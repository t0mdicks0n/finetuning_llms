"""
Export and merge LoRA adapters with base model.

Creates a merged model and optionally converts to GGUF format for offline deployment.

Usage (Modal - recommended):
    # Merge adapters and save full model
    modal run src/export/merge.py

    # Merge and convert to GGUF
    modal run src/export/merge.py --gguf

    # Merge and push to Hugging Face Hub
    modal run src/export/merge.py --push --repo-id your-username/swedish-sovereign-ai

Usage (Local - requires GPU):
    python -m src.export.merge --adapter-path /path/to/adapters --output-dir ./merged
"""

import modal

# Modal app configuration
app = modal.App("swedish-sovereign-ai-export")

# Use the same volume as training
volume = modal.Volume.from_name("sovereign-model-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Base model (must match training)
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Build image with llama.cpp for GGUF conversion
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .pip_install(
        "numpy<2",
        "torch==2.2.0",
        "transformers==4.40.0",
        "accelerate==0.28.0",
        "peft==0.10.0",
        "sentencepiece",
        "huggingface_hub",
    )
    .run_commands(
        # Clone and build llama.cpp for GGUF conversion
        "git clone https://github.com/ggerganov/llama.cpp /llama.cpp",
        "cd /llama.cpp && make -j$(nproc)",
        "pip install /llama.cpp/gguf-py",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def merge_adapters(
    output_name: str = "merged",
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> str:
    """
    Merge LoRA adapters with base model.

    Args:
        output_name: Name for the output directory.
        push_to_hub: Whether to push to Hugging Face Hub.
        hub_repo_id: Repository ID for Hub upload.

    Returns:
        Path to merged model.
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = os.path.join(VOLUME_PATH, "adapters")
    output_path = os.path.join(VOLUME_PATH, output_name)

    if not os.path.exists(adapter_path):
        return f"ERROR: No adapters found at {adapter_path}. Run training first."

    print("=" * 60)
    print("Swedish Sovereign AI - Model Export")
    print("=" * 60)

    # Load base model (full precision for merging)
    print(f"\nLoading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load and merge adapters
    print(f"\nLoading adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("\nMerging adapters with base model...")
    model = model.merge_and_unload()

    # Save merged model
    print(f"\nSaving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Commit volume changes
    volume.commit()

    print(f"\nMerged model saved to: {output_path}")

    # Push to Hub if requested
    if push_to_hub and hub_repo_id:
        print(f"\nPushing to Hugging Face Hub: {hub_repo_id}")
        model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)
        print(f"Successfully pushed to: https://huggingface.co/{hub_repo_id}")

    return output_path


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def convert_to_gguf(
    model_path: str = "merged",
    quantization: str = "q4_k_m",
) -> str:
    """
    Convert merged model to GGUF format.

    Args:
        model_path: Path to merged model (relative to volume).
        quantization: Quantization type (q4_k_m, q5_k_m, q8_0, f16).

    Returns:
        Path to GGUF file.
    """
    import os
    import subprocess

    input_path = os.path.join(VOLUME_PATH, model_path)
    output_dir = os.path.join(VOLUME_PATH, "gguf")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_path):
        return f"ERROR: Model not found at {input_path}. Run merge first."

    print("=" * 60)
    print("GGUF Conversion")
    print("=" * 60)

    # Step 1: Convert to GGUF F16
    print("\nStep 1: Converting to GGUF format...")
    f16_path = os.path.join(output_dir, "model-f16.gguf")

    convert_cmd = [
        "python", "/llama.cpp/convert_hf_to_gguf.py",
        input_path,
        "--outfile", f16_path,
        "--outtype", "f16",
    ]

    result = subprocess.run(convert_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Conversion error: {result.stderr}")
        return f"ERROR: GGUF conversion failed: {result.stderr}"

    print(f"Created F16 GGUF: {f16_path}")

    # Step 2: Quantize if requested
    if quantization != "f16":
        print(f"\nStep 2: Quantizing to {quantization}...")
        quant_path = os.path.join(output_dir, f"model-{quantization}.gguf")

        quant_cmd = [
            "/llama.cpp/llama-quantize",
            f16_path,
            quant_path,
            quantization.upper(),
        ]

        result = subprocess.run(quant_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Quantization error: {result.stderr}")
            return f"ERROR: Quantization failed: {result.stderr}"

        print(f"Created quantized GGUF: {quant_path}")

        # Remove F16 file to save space
        os.remove(f16_path)
        final_path = quant_path
    else:
        final_path = f16_path

    # Commit volume changes
    volume.commit()

    # Get file size
    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"\nFinal GGUF size: {size_mb:.1f} MB")
    print(f"Saved to: {final_path}")

    return final_path


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={VOLUME_PATH: volume},
)
def test_merged_model(prompt: str = "Vad är reporäntan?") -> str:
    """
    Test the merged model with a sample prompt.

    Args:
        prompt: Swedish prompt to test.

    Returns:
        Model response.
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = os.path.join(VOLUME_PATH, "merged")

    if not os.path.exists(model_path):
        return "ERROR: Merged model not found. Run merge first."

    print(f"Loading merged model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Format prompt
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


@app.local_entrypoint()
def main(
    gguf: bool = False,
    quantization: str = "q4_k_m",
    push: bool = False,
    repo_id: str = "",
    test: bool = False,
):
    """
    Main entrypoint for model export.

    Args:
        gguf: Convert to GGUF format after merging.
        quantization: GGUF quantization type (q4_k_m, q5_k_m, q8_0, f16).
        push: Push merged model to Hugging Face Hub.
        repo_id: Hugging Face repo ID (required if --push).
        test: Test the merged model after export.
    """
    print("=" * 60)
    print("Swedish Sovereign AI - Export Pipeline")
    print("=" * 60)

    # Step 1: Merge adapters
    print("\n[1/3] Merging LoRA adapters with base model...")
    merged_path = merge_adapters.remote(
        output_name="merged",
        push_to_hub=push,
        hub_repo_id=repo_id if repo_id else None,
    )
    print(f"Merged model: {merged_path}")

    if "ERROR" in merged_path:
        print(merged_path)
        return

    # Step 2: Convert to GGUF if requested
    if gguf:
        print(f"\n[2/3] Converting to GGUF ({quantization})...")
        gguf_path = convert_to_gguf.remote(
            model_path="merged",
            quantization=quantization,
        )
        print(f"GGUF file: {gguf_path}")
    else:
        print("\n[2/3] Skipping GGUF conversion (use --gguf to enable)")

    # Step 3: Test if requested
    if test:
        print("\n[3/3] Testing merged model...")
        test_prompt = "Vad anser Riksbanken om inflationen?"
        print(f"Prompt: {test_prompt}")
        response = test_merged_model.remote(test_prompt)
        print(f"Response: {response}")
    else:
        print("\n[3/3] Skipping test (use --test to enable)")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Download merged model: modal volume get sovereign-model-vol merged/")
    if gguf:
        print("  - Download GGUF: modal volume get sovereign-model-vol gguf/")
        print("  - Run locally with llama.cpp or Ollama")
    print("  - Run evaluation: modal run src/eval/eval_modal.py --compare")
