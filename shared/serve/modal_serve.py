"""
Modal inference server for Swedish Sovereign AI using vLLM.

Serves both vanilla and LoRA fine-tuned Mistral-7B with dynamic adapter switching
per request. Includes semantic router for automatic model selection based on prompt.

Uses synchronous vLLM for stability - Modal handles scaling.

Usage:
    # Deploy (creates persistent endpoint)
    modal deploy src/serve/modal_serve.py

    # Run temporarily for testing
    modal serve src/serve/modal_serve.py

    # Test with auto-routing (default)
    curl -X POST https://your-app--inference-server-chat.modal.run \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "Vad är reporäntan?"}]
      }'

    # Force specific model (bypass router)
    curl -X POST https://your-app--inference-server-chat.modal.run \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "Hello"}],
        "use_finetuned": false
      }'
"""

import modal

# Modal app configuration
app = modal.App("swedish-sovereign-ai-serve")

# Use the same volume as training to access LoRA adapters
volume = modal.Volume.from_name("sovereign-model-vol", create_if_missing=True)
VOLUME_PATH = "/vol"
ADAPTER_PATH = "/vol/adapters"

# Model configuration - must match training (src/models/riksbanken/train.py)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# LoRA adapter name for vLLM
LORA_NAME = "riksbanken"

# GPU configuration - A10G has 24GB, sufficient for 7B model in fp16
GPU_TYPE = "A10G"

# Router configuration
ROUTER_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ROUTE_RIKSBANKEN = "riksbanken"
ROUTE_GENERAL = "general"


# Build Modal image with vLLM, router, and pre-downloaded model weights
def download_models():
    """Download model weights at image build time."""
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    # Download LLM weights
    snapshot_download(
        "mistralai/Mistral-7B-Instruct-v0.3",
        ignore_patterns=["*.pt", "*.bin"],  # Only get safetensors
    )

    # Download router encoder
    SentenceTransformer(ROUTER_ENCODER)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "fastapi[standard]",
        "pydantic>=2.0",
        "huggingface_hub",
        "semantic-router",
        "sentence-transformers",
    )
    .run_function(download_models)  # Cache models in image
)


# Router artifact path on volume (JSON format - reconstructed at load time)
ROUTER_PATH = "/vol/router/semantic_router.json"


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=600,
    volumes={VOLUME_PATH: volume},
    scaledown_window=600,  # 10 min idle before scale down
    max_containers=1,  # Limit to 1 container to prevent runaway scaling
)
class InferenceServer:
    """vLLM inference server with LoRA switching and semantic routing."""

    @modal.enter()
    def load_model(self):
        """Load vLLM engine, LoRA adapter, and semantic router."""
        import os
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        print("=" * 60)
        print("Starting vLLM Inference Server")
        print("=" * 60)

        # Check if LoRA adapters exist
        self.lora_available = os.path.exists(ADAPTER_PATH) and os.path.exists(
            os.path.join(ADAPTER_PATH, "adapter_config.json")
        )
        if self.lora_available:
            print(f"LoRA adapters found at: {ADAPTER_PATH}")
        else:
            print(f"WARNING: No LoRA adapters at {ADAPTER_PATH}")
            print("Fine-tuned model will not be available.")

        # Load semantic router from volume (reconstruct from JSON)
        self.router_available = False
        self.router = None
        if os.path.exists(ROUTER_PATH):
            print(f"\nLoading semantic router from: {ROUTER_PATH}")
            self._load_router_from_json(ROUTER_PATH)
        else:
            print(f"WARNING: No router data found at {ROUTER_PATH}")
            print("Auto-routing will not be available.")

        # Load vLLM engine with LoRA support
        print(f"\nLoading model: {BASE_MODEL}")
        self.llm = LLM(
            model=BASE_MODEL,
            enable_lora=self.lora_available,
            max_lora_rank=64,
            max_model_len=4096,
            dtype="float16",
            enforce_eager=True,  # Faster startup, slightly slower inference
        )

        # Create LoRA request object for finetuned inference
        if self.lora_available:
            self.lora_request = LoRARequest(
                lora_name=LORA_NAME,
                lora_int_id=1,
                lora_path=ADAPTER_PATH,
            )
            print(f"LoRA adapter '{LORA_NAME}' registered")

        print("\nvLLM Inference server ready!")
        print("=" * 60)

    def _load_router_from_json(self, path: str):
        """Load and reconstruct semantic router from JSON data."""
        import json
        from semantic_router import Route
        from semantic_router.routers import SemanticRouter
        from semantic_router.encoders import HuggingFaceEncoder

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        encoder_model = data.get("encoder_model", ROUTER_ENCODER)
        print(f"  Encoder: {encoder_model}")

        encoder = HuggingFaceEncoder(name=encoder_model)

        routes = []
        for route_name, utterances in data.get("routes", {}).items():
            routes.append(Route(name=route_name, utterances=utterances))
            print(f"  Route '{route_name}': {len(utterances)} examples")

        self.router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
        self.router_available = True
        print("Semantic router reconstructed successfully")

    def route_prompt(self, prompt: str) -> str:
        """
        Route a prompt to the appropriate model.

        Returns:
            'riksbanken' for domain-specific queries, 'general' for everything else
        """
        if not self.router_available:
            return ROUTE_GENERAL

        result = self.router(prompt)
        # Default to general if no confident match (safer - use vanilla when uncertain)
        return result.name if result and result.name else ROUTE_GENERAL

    def format_messages(self, messages: list) -> str:
        """Format messages using Mistral chat template."""
        formatted = ""
        for msg in messages:
            if msg["role"] == "user":
                formatted += f"[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                formatted += f" {msg['content']}</s>"
            elif msg["role"] == "system":
                formatted = f"[INST] {msg['content']}\n\n"
        return formatted

    @modal.fastapi_endpoint(method="POST")
    def chat(self, request: dict) -> dict:
        """
        Chat endpoint with auto-routing or manual model selection.

        Request options:
        - auto_route (default: True): Use semantic router to select model
        - use_finetuned: Override auto-routing (True=finetuned, False=vanilla)
        """
        from vllm import SamplingParams

        # Parse request
        messages = request.get("messages", [])
        auto_route = request.get("auto_route", True)
        use_finetuned_override = request.get("use_finetuned")  # None means use auto-routing
        max_tokens = request.get("max_tokens", 512)
        temperature = request.get("temperature", 0.7)

        if not messages:
            return {"error": "messages array is required", "model": None}

        # Validate message format
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                return {"error": "Each message must have 'role' and 'content'", "model": None}
            if msg["role"] not in ("user", "assistant", "system"):
                return {"error": f"Invalid role: {msg['role']}", "model": None}

        # Get the last user message for routing
        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break

        # Determine which model to use
        route = None
        if use_finetuned_override is not None:
            # Manual override
            use_finetuned = use_finetuned_override
        elif auto_route and self.router_available and last_user_msg:
            # Auto-route based on prompt
            route = self.route_prompt(last_user_msg)
            use_finetuned = (route == ROUTE_RIKSBANKEN)
        else:
            # Default to vanilla
            use_finetuned = False

        # Format messages for Mistral
        prompt = self.format_messages(messages)

        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["[INST]", "</s>"],  # Stop when model tries to generate new turn
        )

        # Select model (with or without LoRA)
        if use_finetuned and self.lora_available:
            outputs = self.llm.generate(
                [prompt],
                sampling_params,
                lora_request=self.lora_request,
            )
            model_type = "finetuned"
        else:
            outputs = self.llm.generate([prompt], sampling_params)
            model_type = "vanilla"
            if use_finetuned and not self.lora_available:
                model_type = "vanilla (finetuned unavailable)"

        # Extract generated text
        response_text = outputs[0].outputs[0].text.strip()

        result = {
            "response": response_text,
            "model": model_type,
        }

        # Include routing info if auto-routed
        if route:
            result["route"] = route

        return result

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "base_model": BASE_MODEL,
            "lora_available": self.lora_available,
            "lora_path": ADAPTER_PATH if self.lora_available else None,
            "router_available": self.router_available,
            "engine": "vllm",
        }


@app.local_entrypoint()
def main():
    """Test the server locally."""
    print("=" * 60)
    print("Swedish Sovereign AI - vLLM Inference Server")
    print("=" * 60)
    print("\nTo deploy:")
    print("  modal deploy src/serve/modal_serve.py")
    print("\nTo run temporarily:")
    print("  modal serve src/serve/modal_serve.py")
    print("\nEndpoints:")
    print("  POST /chat - Chat with auto-routing or manual model selection")
    print("  GET /health - Check server status")
    print("\nExample requests:")
    print("\n  # Auto-route (default) - router picks model based on prompt:")
    print('  curl -X POST <url>/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"messages": [{"role": "user", "content": "Vad är reporäntan?"}]}\'')
    print("\n  # Force finetuned model:")
    print('  curl -X POST <url>/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"messages": [...], "use_finetuned": true}\'')
    print("\n  # Force vanilla model:")
    print('  curl -X POST <url>/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"messages": [...], "use_finetuned": false}\'')
