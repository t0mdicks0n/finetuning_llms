"""
Modal vLLM server for Swedish Sovereign AI.

Serves both vanilla and LoRA fine-tuned Ministral-8B from a single vLLM instance
with dynamic adapter switching per request. Supports multi-turn conversations.

Usage:
    # Deploy (creates persistent endpoint)
    modal deploy src/serve/modal_serve.py

    # Run temporarily for testing
    modal serve src/serve/modal_serve.py

    # Test the endpoint
    curl -X POST https://your-app--vllm-serve.modal.run/chat \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [
          {"role": "user", "content": "Vad är reporäntan?"},
          {"role": "assistant", "content": "Reporäntan är..."},
          {"role": "user", "content": "Hur påverkar den inflationen?"}
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

# Model configuration - must match training (src/train/train.py)
# Ministral 3 model (Dec 2025) - 256K context, Apache 2.0 license
BASE_MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"

# GPU configuration - A10G has 24GB, sufficient for 8B model + LoRA
GPU_TYPE = "A10G"

# Build Modal image with vLLM and FastAPI
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.4",
        "fastapi[standard]",
        "pydantic>=2.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Fast model downloads
    })
)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=600,
    volumes={VOLUME_PATH: volume},
    container_idle_timeout=300,  # 5 min idle before scale down
    allow_concurrent_inputs=10,  # Handle multiple requests
)
class VLLMServer:
    """vLLM server with LoRA hot-switching capability."""

    @modal.enter()
    def start_engine(self):
        """Initialize vLLM engine on container start."""
        import os
        from vllm import LLM
        from transformers import AutoTokenizer

        print("=" * 60)
        print("Starting vLLM Engine")
        print("=" * 60)

        # Check if LoRA adapters exist
        self.lora_available = os.path.exists(ADAPTER_PATH)
        if self.lora_available:
            print(f"LoRA adapters found at: {ADAPTER_PATH}")
        else:
            print(f"WARNING: No LoRA adapters at {ADAPTER_PATH}")
            print("Fine-tuned model will not be available.")

        # Initialize vLLM with LoRA support
        print(f"\nLoading base model: {BASE_MODEL}")
        self.engine = LLM(
            model=BASE_MODEL,
            enable_lora=self.lora_available,
            max_lora_rank=16,  # Must match training LORA_R
            max_model_len=4096,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )

        # Load tokenizer for chat template
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )

        print("\nvLLM engine ready!")
        print("=" * 60)

    @modal.web_endpoint(method="POST")
    def chat(self, request: dict) -> dict:
        """
        Chat endpoint with vanilla/finetuned switching.
        Supports multi-turn conversations.

        Request body:
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ],
                "use_finetuned": true | false,
                "max_tokens": 512,        # optional
                "temperature": 0.7        # optional
            }

        Response:
            {
                "response": "model output",
                "model": "vanilla" | "finetuned"
            }
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

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

        # Apply Mistral chat template to full conversation
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
        )

        # Generate with or without LoRA
        if use_finetuned and self.lora_available:
            lora_request = LoRARequest(
                lora_name="riksbanken",
                lora_int_id=1,
                lora_path=ADAPTER_PATH,
            )
            outputs = self.engine.generate(
                [formatted_prompt],
                sampling_params,
                lora_request=lora_request,
            )
            model_type = "finetuned"
        else:
            outputs = self.engine.generate(
                [formatted_prompt],
                sampling_params,
            )
            model_type = "vanilla"
            if use_finetuned and not self.lora_available:
                model_type = "vanilla (finetuned unavailable)"

        # Extract response text
        response_text = outputs[0].outputs[0].text

        return {
            "response": response_text,
            "model": model_type,
        }

    @modal.web_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "base_model": BASE_MODEL,
            "lora_available": self.lora_available,
            "lora_path": ADAPTER_PATH if self.lora_available else None,
        }


@app.local_entrypoint()
def main():
    """Test the server locally."""
    print("=" * 60)
    print("Swedish Sovereign AI - vLLM Server")
    print("=" * 60)
    print("\nTo deploy:")
    print("  modal deploy src/serve/modal_serve.py")
    print("\nTo run temporarily:")
    print("  modal serve src/serve/modal_serve.py")
    print("\nEndpoints:")
    print("  POST /chat - Chat with vanilla or finetuned model")
    print("  GET /health - Check server status")
    print("\nExample request (single turn):")
    print('  curl -X POST <url>/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"messages": [{"role": "user", "content": "Vad är reporäntan?"}], "use_finetuned": true}\'')
    print("\nExample request (multi-turn):")
    print('  curl -X POST <url>/chat \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{')
    print('      "messages": [')
    print('        {"role": "user", "content": "Vad är reporäntan?"},')
    print('        {"role": "assistant", "content": "Reporäntan är Sveriges styrränta..."},')
    print('        {"role": "user", "content": "Hur påverkar den inflationen?"}')
    print('      ],')
    print('      "use_finetuned": true')
    print('    }\'')
