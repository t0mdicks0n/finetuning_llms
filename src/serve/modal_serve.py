"""
Modal inference server for Swedish Sovereign AI using vLLM.

Serves both vanilla and LoRA fine-tuned Mistral-7B with dynamic adapter switching
per request. Uses synchronous vLLM for stability - Modal handles scaling.

Usage:
    # Deploy (creates persistent endpoint)
    modal deploy src/serve/modal_serve.py

    # Run temporarily for testing
    modal serve src/serve/modal_serve.py

    # Test the endpoint
    curl -X POST https://your-app--inference-server-chat.modal.run \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [
          {"role": "user", "content": "Vad är reporäntan?"}
        ],
        "use_finetuned": true
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

# Build Modal image with vLLM and pre-downloaded model weights
def download_model():
    """Download model weights at image build time."""
    from huggingface_hub import snapshot_download
    snapshot_download(
        "mistralai/Mistral-7B-Instruct-v0.3",
        ignore_patterns=["*.pt", "*.bin"],  # Only get safetensors
    )

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",  # Let it install latest - crashes sometimes but recovers
        "fastapi[standard]",
        "pydantic>=2.0",
        "huggingface_hub",
    )
    .run_function(download_model)  # Cache model in image
)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=600,
    volumes={VOLUME_PATH: volume},
    scaledown_window=600,  # 10 min idle before scale down
    max_containers=1,  # Limit to 1 container to prevent runaway scaling
)
class InferenceServer:
    """vLLM inference server with LoRA switching capability."""

    @modal.enter()
    def load_model(self):
        """Load vLLM engine with base model and LoRA adapter."""
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
        Chat endpoint with vanilla/finetuned switching.
        Synchronous - stable and predictable latency.
        """
        from vllm import SamplingParams

        # Parse request
        messages = request.get("messages", [])
        use_finetuned = request.get("use_finetuned", False)
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

        return {
            "response": response_text,
            "model": model_type,
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "base_model": BASE_MODEL,
            "lora_available": self.lora_available,
            "lora_path": ADAPTER_PATH if self.lora_available else None,
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
    print("  POST /chat - Chat with vanilla or finetuned model")
    print("  GET /health - Check server status")
    print("\nExample request:")
    print('  curl -X POST <url>/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"messages": [{"role": "user", "content": "Vad är reporäntan?"}], "use_finetuned": true}\'')
