"""
Modal inference server for Swedish Sovereign AI.

Serves both vanilla and LoRA fine-tuned Mistral-7B with dynamic adapter switching
per request. Supports multi-turn conversations.

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
# Using stable Mistral-7B-Instruct-v0.3 (text-only model)
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# GPU configuration - A10G has 24GB, sufficient for 7B model + LoRA
GPU_TYPE = "A10G"

# Build Modal image - matching training setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "torch>=2.5.0",
        "transformers>=4.36.0",  # Stable release, Mistral-7B fully supported
        "accelerate>=1.2.0",
        "peft>=0.14.0",
        "bitsandbytes>=0.45.0",
        "fastapi[standard]",
        "pydantic>=2.0",
        "sentencepiece",
    )
)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=600,
    volumes={VOLUME_PATH: volume},
    scaledown_window=300,  # 5 min idle before scale down
)
@modal.concurrent(max_inputs=10)
class InferenceServer:
    """Inference server with LoRA switching capability."""

    @modal.enter()
    def load_models(self):
        """Load base model and optionally LoRA adapters on container start."""
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print("=" * 60)
        print("Starting Inference Server")
        print("=" * 60)

        # Check if LoRA adapters exist
        self.lora_available = os.path.exists(ADAPTER_PATH)
        if self.lora_available:
            print(f"LoRA adapters found at: {ADAPTER_PATH}")
        else:
            print(f"WARNING: No LoRA adapters at {ADAPTER_PATH}")
            print("Fine-tuned model will not be available.")

        # Load base model with 4-bit quantization (same as training)
        print(f"\nLoading base model: {BASE_MODEL}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print("Loaded Mistral-7B-Instruct model")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load finetuned model with LoRA adapters if available
        if self.lora_available:
            print(f"\nLoading LoRA adapters from: {ADAPTER_PATH}")
            # Need to load a fresh base model for the finetuned version
            base_model_ft = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.finetuned_model = PeftModel.from_pretrained(base_model_ft, ADAPTER_PATH)
            self.finetuned_model.eval()
            print("LoRA adapters loaded successfully")

        self.base_model.eval()
        print("\nInference server ready!")
        print("=" * 60)

    def generate(self, model, messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response from a model given messages."""
        import torch

        # Apply chat template to full conversation
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Only decode the newly generated tokens (skip the input prompt)
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    @modal.fastapi_endpoint(method="POST")
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

        # Select model
        if use_finetuned and self.lora_available:
            model = self.finetuned_model
            model_type = "finetuned"
        else:
            model = self.base_model
            model_type = "vanilla"
            if use_finetuned and not self.lora_available:
                model_type = "vanilla (finetuned unavailable)"

        # Generate response
        response_text = self.generate(model, messages, max_tokens, temperature)

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
        }


@app.local_entrypoint()
def main():
    """Test the server locally."""
    print("=" * 60)
    print("Swedish Sovereign AI - Inference Server")
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
