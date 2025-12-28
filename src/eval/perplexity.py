"""
Perplexity evaluation for domain adaptation.

Measures how well the model predicts held-out Riksbanken text.
Lower perplexity = better domain fit.
"""

import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# Default model configuration (must match train.py)
BASE_MODEL_NAME = "mistralai/Ministral-3-8B-Instruct-2512"

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
VAL_PATH = DATA_DIR / "processed" / "val.jsonl"


def load_model(
    model_name: str = BASE_MODEL_NAME,
    adapter_path: str | None = None,
    device: str = "cuda",
) -> tuple:
    """
    Load model and tokenizer.

    Args:
        model_name: Base model name/path.
        adapter_path: Optional path to LoRA adapters.
        device: Device to load model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading base model: {model_name}")

    # Quantization config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapters if provided
    if adapter_path:
        print(f"Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        # Use adapter's tokenizer if available
        adapter_tokenizer_path = Path(adapter_path)
        if (adapter_tokenizer_path / "tokenizer.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model.eval()
    return model, tokenizer


def calculate_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 2048,
    stride: int = 512,
) -> dict:
    """
    Calculate perplexity on a list of texts.

    Uses sliding window approach for long texts.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        texts: List of text strings to evaluate.
        max_length: Maximum sequence length.
        stride: Stride for sliding window.

    Returns:
        Dictionary with perplexity metrics.
    """
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    per_text_perplexities = []

    for text in tqdm(texts, desc="Calculating perplexity"):
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )

        seq_len = encodings.input_ids.size(1)

        if seq_len == 0:
            continue

        nlls = []
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # Tokens to predict

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()

            # Mask tokens we've already seen (except the new ones)
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood.item() * trg_len)
            total_tokens += trg_len

            prev_end_loc = end_loc
            if end_loc >= seq_len:
                break

        # Calculate perplexity for this text
        if nlls:
            text_nll = sum(nlls)
            total_loss += text_nll
            text_ppl = math.exp(text_nll / len(nlls) / stride) if nlls else float('inf')
            per_text_perplexities.append(text_ppl)

    # Overall perplexity
    avg_nll = total_loss / total_tokens if total_tokens > 0 else float('inf')
    overall_perplexity = math.exp(avg_nll)

    return {
        "perplexity": overall_perplexity,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "num_texts": len(texts),
        "per_text_mean": sum(per_text_perplexities) / len(per_text_perplexities) if per_text_perplexities else 0,
    }


def load_validation_texts(val_path: Path = VAL_PATH, max_examples: int | None = None) -> list[str]:
    """
    Load validation texts from JSONL file.

    Args:
        val_path: Path to validation JSONL.
        max_examples: Maximum number of examples to load.

    Returns:
        List of text strings.
    """
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    texts = []
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            data = json.loads(line)
            texts.append(data["text"])

    return texts


def evaluate_perplexity(
    adapter_path: str | None = None,
    base_model: str = BASE_MODEL_NAME,
    val_path: Path = VAL_PATH,
    max_examples: int | None = None,
) -> dict:
    """
    Run perplexity evaluation.

    Args:
        adapter_path: Path to LoRA adapters (None for base model).
        base_model: Base model name.
        val_path: Path to validation data.
        max_examples: Maximum examples to evaluate.

    Returns:
        Dictionary with evaluation results.
    """
    # Load validation texts
    print(f"\nLoading validation data from: {val_path}")
    texts = load_validation_texts(val_path, max_examples)
    print(f"Loaded {len(texts)} validation texts")

    # Load model
    model, tokenizer = load_model(base_model, adapter_path)

    # Calculate perplexity
    print("\nCalculating perplexity...")
    results = calculate_perplexity(model, tokenizer, texts)

    model_type = "finetuned" if adapter_path else "base"
    results["model_type"] = model_type
    results["model_name"] = base_model
    results["adapter_path"] = adapter_path

    return results


def compare_models(
    adapter_path: str,
    base_model: str = BASE_MODEL_NAME,
    val_path: Path = VAL_PATH,
    max_examples: int | None = None,
) -> dict:
    """
    Compare base model and finetuned model perplexity.

    Args:
        adapter_path: Path to LoRA adapters.
        base_model: Base model name.
        val_path: Path to validation data.
        max_examples: Maximum examples to evaluate.

    Returns:
        Comparison results.
    """
    print("=" * 60)
    print("PERPLEXITY EVALUATION")
    print("=" * 60)

    # Load validation texts once
    texts = load_validation_texts(val_path, max_examples)
    print(f"Loaded {len(texts)} validation texts\n")

    # Evaluate base model
    print("-" * 40)
    print("Evaluating BASE model...")
    print("-" * 40)
    base_model_obj, base_tokenizer = load_model(base_model, None)
    base_results = calculate_perplexity(base_model_obj, base_tokenizer, texts)

    # Free memory
    del base_model_obj
    torch.cuda.empty_cache()

    # Evaluate finetuned model
    print("\n" + "-" * 40)
    print("Evaluating FINETUNED model...")
    print("-" * 40)
    ft_model, ft_tokenizer = load_model(base_model, adapter_path)
    ft_results = calculate_perplexity(ft_model, ft_tokenizer, texts)

    # Calculate improvement
    ppl_reduction = base_results["perplexity"] - ft_results["perplexity"]
    ppl_reduction_pct = (ppl_reduction / base_results["perplexity"]) * 100

    comparison = {
        "base_model": {
            "name": base_model,
            "perplexity": base_results["perplexity"],
            "avg_nll": base_results["avg_nll"],
        },
        "finetuned_model": {
            "adapter_path": adapter_path,
            "perplexity": ft_results["perplexity"],
            "avg_nll": ft_results["avg_nll"],
        },
        "improvement": {
            "perplexity_reduction": ppl_reduction,
            "perplexity_reduction_pct": ppl_reduction_pct,
        },
        "validation_texts": len(texts),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON RESULTS")
    print("=" * 60)
    print(f"\nBase model perplexity:      {base_results['perplexity']:.2f}")
    print(f"Finetuned model perplexity: {ft_results['perplexity']:.2f}")
    print(f"\nImprovement: {ppl_reduction:.2f} ({ppl_reduction_pct:.1f}% reduction)")

    if ppl_reduction > 0:
        print("\n✓ Finetuning improved domain adaptation!")
    else:
        print("\n✗ Finetuning did not improve perplexity")

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perplexity evaluation")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapters (omit for base model only)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base vs finetuned model",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum validation examples to use",
    )

    args = parser.parse_args()

    if args.compare and args.adapter_path:
        results = compare_models(args.adapter_path, max_examples=args.max_examples)
    else:
        results = evaluate_perplexity(
            adapter_path=args.adapter_path,
            max_examples=args.max_examples,
        )
        print(f"\nPerplexity: {results['perplexity']:.2f}")

    print("\nResults:", json.dumps(results, indent=2, default=str))
