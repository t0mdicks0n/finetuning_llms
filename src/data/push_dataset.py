"""
Push training dataset to HuggingFace Hub.

Uploads the generated Q&A training data as a HuggingFace dataset for:
- Versioning and reproducibility
- Easy loading with `datasets.load_dataset()`
- Sharing and collaboration

Usage:
    # Push dataset to HuggingFace Hub
    python -m src.data.push_dataset --dataset-id your-username/riksbanken-qa

    # Push with a custom split name
    python -m src.data.push_dataset --dataset-id your-username/riksbanken-qa --split train

    # Dry run (show what would be uploaded)
    python -m src.data.push_dataset --dataset-id your-username/riksbanken-qa --dry-run
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# Load environment variables
load_dotenv()

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train_qa.jsonl"


def load_dataset_from_jsonl(path: Path) -> list[dict]:
    """Load dataset from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def create_dataset_card(repo_id: str, num_examples: int, sources: list[str]) -> str:
    """Create a dataset card (README.md) for the HuggingFace repo."""
    return f"""---
language:
  - sv
license: apache-2.0
task_categories:
  - question-answering
  - text-generation
tags:
  - swedish
  - riksbanken
  - monetary-policy
  - finance
  - instruction-tuning
  - synthetic-data
size_categories:
  - 1K<n<10K
---

# Riksbanken Q&A Dataset

Swedish instruction-tuning dataset generated from Riksbanken (Swedish Central Bank) monetary policy reports.

## Dataset Description

This dataset contains **{num_examples:,}** synthetic question-answer pairs in Swedish, covering topics such as:
- Monetary policy decisions
- Inflation targeting
- Interest rate policies (reporäntan)
- Economic forecasts
- Financial stability

### Sources

Generated from {len(sources)} Riksbanken reports:
{chr(10).join(f'- {s}' for s in sorted(sources)[:10])}
{"..." if len(sources) > 10 else ""}

### Format

Each example follows the Mistral instruction format with `messages` field:

```json
{{
  "messages": [
    {{"role": "user", "content": "Vad är reporäntan?"}},
    {{"role": "assistant", "content": "Reporäntan är Riksbankens styrränta..."}}
  ],
  "source": "penningpolitisk-rapport-december-2024",
  "chunk_id": 0,
  "qa_id": 0
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")

# Access examples
for example in dataset["train"]:
    question = example["messages"][0]["content"]
    answer = example["messages"][1]["content"]
    print(f"Q: {{question}}")
    print(f"A: {{answer}}")
```

## Training

This dataset was created for fine-tuning Ministral-8B on Swedish monetary policy domain knowledge.

See the [Finetuning LLMs](https://github.com/t0mdicks0n/finetuning_llms) project for training code.

## Generation

Q&A pairs were generated using Gemini 2.5 Flash with carefully crafted prompts to ensure:
- Diverse question types (factual, explanatory, analytical)
- High-quality Swedish language
- Grounded answers based on source documents

## License

Apache 2.0
"""


def push_to_hub(
    dataset_id: str,
    split: str = "train",
    dry_run: bool = False,
) -> None:
    """
    Push dataset to HuggingFace Hub.

    Args:
        dataset_id: HuggingFace dataset ID (e.g., "username/riksbanken-qa").
        split: Dataset split name.
        dry_run: If True, just show what would be uploaded.
    """
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not dry_run:
        print("ERROR: HF_TOKEN not set in environment")
        print("Add it to .env file: HF_TOKEN=hf_your_token")
        return

    # Load dataset
    if not TRAIN_PATH.exists():
        print(f"ERROR: Training data not found at {TRAIN_PATH}")
        print("Run the Q&A generation pipeline first:")
        print("  python -m src.data.generate_qa")
        return

    print(f"Loading dataset from: {TRAIN_PATH}")
    examples = load_dataset_from_jsonl(TRAIN_PATH)
    print(f"Loaded {len(examples):,} examples")

    # Get unique sources
    sources = list(set(ex.get("source", "unknown") for ex in examples))
    print(f"Sources: {len(sources)} documents")

    # Show sample
    print("\nSample example:")
    if examples:
        sample = examples[0]
        q = sample["messages"][0]["content"][:80]
        a = sample["messages"][1]["content"][:80]
        print(f"  Q: {q}...")
        print(f"  A: {a}...")

    if dry_run:
        print("\n[DRY RUN] Would upload to:", dataset_id)
        print(f"  - {len(examples):,} examples")
        print(f"  - Split: {split}")
        print(f"  - Sources: {len(sources)} documents")
        return

    # Login to HuggingFace
    print("\nLogging in to HuggingFace...")
    login(token=hf_token)

    # Create API client
    api = HfApi()

    # Create dataset if it doesn't exist
    print(f"\nCreating/updating dataset: {dataset_id}")
    api.create_repo(
        repo_id=dataset_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    # Upload the JSONL file
    print(f"Uploading {TRAIN_PATH.name}...")
    api.upload_file(
        path_or_fileobj=str(TRAIN_PATH),
        path_in_repo=f"data/{split}.jsonl",
        repo_id=dataset_id,
        repo_type="dataset",
    )

    # Create and upload dataset card
    print("Creating dataset card...")
    readme_content = create_dataset_card(dataset_id, len(examples), sources)
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=dataset_id,
        repo_type="dataset",
    )

    print(f"\nDataset pushed to: https://huggingface.co/datasets/{dataset_id}")
    print("\nTo load the dataset:")
    print(f'  from datasets import load_dataset')
    print(f'  dataset = load_dataset("{dataset_id}")')


def main():
    parser = argparse.ArgumentParser(
        description="Push Q&A dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="HuggingFace dataset ID (e.g., your-username/riksbanken-qa)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default: train)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Swedish Sovereign AI - Push Dataset to HuggingFace")
    print("=" * 60)

    push_to_hub(
        dataset_id=args.dataset_id,
        split=args.split,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
