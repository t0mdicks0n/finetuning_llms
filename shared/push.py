"""
Push trained LoRA adapters to HuggingFace Hub.

Downloads adapters from Modal volume and uploads to HuggingFace.

Usage:
    # Push model to HuggingFace Hub
    python -m src.model.push_model --model-id your-username/riksbanken-mistral-lora

    # Dry run (show what would be uploaded)
    python -m src.model.push_model --model-id your-username/riksbanken-mistral-lora --dry-run
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# Load environment variables
load_dotenv()

# Local path for downloaded adapters
LOCAL_ADAPTERS_DIR = Path(__file__).parent.parent.parent / "models" / "adapters"

# Base model info
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def download_from_modal() -> Path:
    """Download adapters from Modal volume."""
    print("Downloading adapters from Modal volume...")

    # Create local directory
    LOCAL_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear existing files
    if LOCAL_ADAPTERS_DIR.exists():
        shutil.rmtree(LOCAL_ADAPTERS_DIR)
    LOCAL_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Download from Modal volume
    result = subprocess.run(
        ["pipenv", "run", "modal", "volume", "get", "sovereign-model-vol", "adapters", str(LOCAL_ADAPTERS_DIR)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR: Failed to download from Modal volume")
        print(result.stderr)
        raise RuntimeError("Modal download failed")

    # Modal creates nested adapters/adapters/ structure, find the actual adapter dir
    adapter_path = LOCAL_ADAPTERS_DIR
    nested_path = LOCAL_ADAPTERS_DIR / "adapters"
    if nested_path.exists() and (nested_path / "adapter_config.json").exists():
        adapter_path = nested_path

    print(f"Downloaded adapters to: {adapter_path}")

    # List downloaded files
    files = list(adapter_path.rglob("*"))
    print(f"Files: {[f.name for f in files if f.is_file()]}")

    return adapter_path


def create_model_card(repo_id: str) -> str:
    """Create a model card (README.md) for the HuggingFace repo."""
    return f"""---
language:
  - sv
license: apache-2.0
library_name: peft
base_model: {BASE_MODEL}
tags:
  - swedish
  - riksbanken
  - monetary-policy
  - finance
  - lora
  - mistral
  - instruction-tuning
datasets:
  - tomdickson/riksbanken-qa
---

# Riksbanken Mistral LoRA

Swedish LoRA adapters for Mistral-7B-Instruct, fine-tuned on Riksbanken (Swedish Central Bank) monetary policy reports.

## Model Description

This model is a LoRA (Low-Rank Adaptation) fine-tune of `{BASE_MODEL}` trained on synthetic Q&A pairs generated from Riksbanken's monetary policy reports (2022-2025).

### Training Data

- **Dataset**: [tomdickson/riksbanken-qa](https://huggingface.co/datasets/tomdickson/riksbanken-qa)
- **Examples**: ~5,000 Swedish Q&A pairs
- **Topics**: Monetary policy, inflation, interest rates (reporäntan), economic forecasts

### Training Configuration

- **LoRA rank**: 16
- **LoRA alpha**: 16
- **Target modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Epochs**: 1
- **Learning rate**: 2e-4

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{BASE_MODEL}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")

# Generate
messages = [{{"role": "user", "content": "Vad är reporäntan?"}}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs.to("cuda"), max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Demo

Try the model at: https://swesovereignai.web.app

## Training

See the [Finetuning LLMs](https://github.com/t0mdicks0n/finetuning_llms) project for training code.

## License

Apache 2.0
"""


def push_to_hub(
    model_id: str,
    dry_run: bool = False,
) -> None:
    """
    Push LoRA adapters to HuggingFace Hub.

    Args:
        model_id: HuggingFace model ID (e.g., "username/riksbanken-mistral-lora").
        dry_run: If True, just show what would be uploaded.
    """
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not dry_run:
        print("ERROR: HF_TOKEN not set in environment")
        print("Add it to .env file: HF_TOKEN=hf_your_token")
        return

    # Download from Modal
    try:
        adapter_path = download_from_modal()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    # Check for required files
    adapter_config = adapter_path / "adapter_config.json"
    adapter_model = adapter_path / "adapter_model.safetensors"

    if not adapter_config.exists():
        print(f"ERROR: adapter_config.json not found in {adapter_path}")
        return

    if not adapter_model.exists():
        # Try .bin format
        adapter_model = adapter_path / "adapter_model.bin"
        if not adapter_model.exists():
            print(f"ERROR: No adapter weights found in {adapter_path}")
            return

    print(f"\nAdapter files found:")
    for f in adapter_path.iterdir():
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")

    if dry_run:
        print(f"\n[DRY RUN] Would upload to: {model_id}")
        print(f"  - Adapter config and weights")
        print(f"  - Model card (README.md)")
        return

    # Login to HuggingFace
    print("\nLogging in to HuggingFace...")
    login(token=hf_token)

    # Create API client
    api = HfApi()

    # Create repo if it doesn't exist
    print(f"\nCreating/updating model repo: {model_id}")
    api.create_repo(
        repo_id=model_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    # Upload all adapter files
    print("Uploading adapter files...")
    for file_path in adapter_path.iterdir():
        if file_path.is_file():
            print(f"  Uploading {file_path.name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=model_id,
                repo_type="model",
            )

    # Create and upload model card
    print("Creating model card...")
    readme_content = create_model_card(model_id)
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=model_id,
        repo_type="model",
    )

    print(f"\nModel pushed to: https://huggingface.co/{model_id}")
    print("\nTo load the model:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, \"{model_id}\")")


def main():
    parser = argparse.ArgumentParser(
        description="Push LoRA adapters to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., your-username/riksbanken-mistral-lora)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Swedish Sovereign AI - Push Model to HuggingFace")
    print("=" * 60)

    push_to_hub(
        model_id=args.model_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
