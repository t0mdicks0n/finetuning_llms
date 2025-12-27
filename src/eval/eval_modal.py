"""
Modal-based evaluation for Swedish Sovereign AI.

Runs all evaluations on Modal GPUs since local GPU is not available.

Usage:
    # Compare base vs finetuned (perplexity + domain questions)
    modal run src/eval/eval_modal.py --compare

    # Quick test with fewer samples
    modal run src/eval/eval_modal.py --compare --quick

Note: EuroEval benchmark is currently disabled due to dependency conflicts.
(EuroEval requires accelerate>=1.9.0, but our training stack uses 0.28.0)
"""

import modal
import json
from dataclasses import dataclass

# Modal app configuration
app = modal.App("swedish-sovereign-ai-eval")

# Use the same volume as training to access adapters
volume = modal.Volume.from_name("sovereign-model-vol", create_if_missing=True)
VOLUME_PATH = "/vol"

# Model configuration (must match train.py)
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Build Modal image with eval dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",
        "torch==2.2.0",
        "transformers==4.40.0",
        "datasets==2.18.0",
        "accelerate==0.28.0",
        "peft==0.10.0",
        "bitsandbytes==0.43.0",
        "scipy",
        "sentencepiece",
        "tqdm",
        "rich",
    )
)

# Note: EuroEval has dependency conflicts with our stack (requires accelerate>=1.9.0)
# For now, EuroEval is disabled. Use ScandEval directly if needed.


# ============================================================================
# Domain Questions (embedded for Modal)
# ============================================================================

DOMAIN_QUESTIONS = [
    {
        "question": "Vad är reporäntan och hur används den av Riksbanken?",
        "expected_keywords": ["styrränta", "penningpolitik", "banker", "lån", "procent"],
        "category": "interest_rates",
    },
    {
        "question": "Vad är Riksbankens inflationsmål?",
        "expected_keywords": ["2", "procent", "KPIF", "KPI", "prisstabilitet"],
        "category": "inflation",
    },
    {
        "question": "Förklara begreppet penningpolitisk transmission.",
        "expected_keywords": ["ränta", "ekonomi", "kanal", "kredit", "växelkurs"],
        "category": "monetary_policy",
    },
    {
        "question": "Vad mäter KPIF och varför använder Riksbanken detta mått?",
        "expected_keywords": ["inflation", "konsumentpris", "räntekostnad", "mått", "exkludera"],
        "category": "inflation",
    },
    {
        "question": "Hur påverkar Riksbankens räntebeslut den svenska kronan?",
        "expected_keywords": ["växelkurs", "SEK", "kapital", "flöde", "stark", "svag"],
        "category": "exchange_rates",
    },
    {
        "question": "Vilka verktyg har Riksbanken för att bedriva penningpolitik?",
        "expected_keywords": ["reporänta", "tillgångsköp", "likviditet", "obligationer"],
        "category": "monetary_policy",
    },
    {
        "question": "Vad innebär kvantitativa lättnader och när används det?",
        "expected_keywords": ["tillgångsköp", "obligationer", "likviditet", "stimulera", "nollränta"],
        "category": "monetary_policy",
    },
    {
        "question": "Hur påverkar internationell konjunktur svensk ekonomi enligt Riksbanken?",
        "expected_keywords": ["export", "handel", "global", "tillväxt", "euroområdet"],
        "category": "economy",
    },
    {
        "question": "Vilken roll spelar hushållens skuldsättning för penningpolitiken?",
        "expected_keywords": ["skuld", "bolån", "risker", "finansiell", "stabilitet"],
        "category": "financial_stability",
    },
    {
        "question": "Hur analyserar Riksbanken arbetsmarknaden i sina rapporter?",
        "expected_keywords": ["sysselsättning", "arbetslöshet", "löner", "resursutnyttjande"],
        "category": "labor_market",
    },
]


# ============================================================================
# Helper Functions (run inside Modal)
# ============================================================================

def load_model_internal(adapter_path: str | None = None):
    """Load model inside Modal container."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading base model: {BASE_MODEL_NAME}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def calculate_perplexity_internal(model, tokenizer, texts: list[str], max_length: int = 2048):
    """Calculate perplexity on texts."""
    import math
    import torch
    from tqdm import tqdm

    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    for text in tqdm(texts, desc="Calculating perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encodings.input_ids.to(device)

        if input_ids.size(1) == 0:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        num_tokens = input_ids.size(1)
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_texts": len(texts),
    }


def generate_response_internal(model, tokenizer, question: str, max_new_tokens: int = 256):
    """Generate response to a question."""
    import torch

    messages = [
        {
            "role": "user",
            "content": f"Du är en svensk ekonomisk analytiker som arbetar på Riksbanken. "
                      f"Svara på följande fråga på svenska med korrekt ekonomisk terminologi:\n\n{question}"
        }
    ]

    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


def evaluate_response_internal(response: str, expected_keywords: list[str]) -> dict:
    """Evaluate response based on keyword presence."""
    response_lower = response.lower()
    found = [kw for kw in expected_keywords if kw.lower() in response_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in response_lower]
    score = len(found) / len(expected_keywords) if expected_keywords else 0

    return {"score": score, "found": found, "missing": missing}


# ============================================================================
# Modal Functions
# ============================================================================

@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
    volumes={VOLUME_PATH: volume},
)
def evaluate_perplexity(
    val_texts: list[str],
    use_adapters: bool = True,
    max_texts: int | None = None,
) -> dict:
    """
    Evaluate perplexity on validation texts.

    Args:
        val_texts: List of validation text strings.
        use_adapters: Whether to load finetuned adapters.
        max_texts: Maximum texts to evaluate.

    Returns:
        Perplexity results.
    """
    import os

    adapter_path = os.path.join(VOLUME_PATH, "adapters") if use_adapters else None

    if use_adapters and not os.path.exists(adapter_path):
        return {"error": "No adapters found. Run training first."}

    if max_texts:
        val_texts = val_texts[:max_texts]

    model, tokenizer = load_model_internal(adapter_path)
    results = calculate_perplexity_internal(model, tokenizer, val_texts)
    results["model_type"] = "finetuned" if use_adapters else "base"

    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
    volumes={VOLUME_PATH: volume},
)
def evaluate_domain_questions(use_adapters: bool = True) -> dict:
    """
    Evaluate model on domain-specific questions.

    Args:
        use_adapters: Whether to load finetuned adapters.

    Returns:
        Domain evaluation results.
    """
    import os

    adapter_path = os.path.join(VOLUME_PATH, "adapters") if use_adapters else None

    if use_adapters and not os.path.exists(adapter_path):
        return {"error": "No adapters found. Run training first."}

    model, tokenizer = load_model_internal(adapter_path)

    results = []
    category_scores = {}

    print(f"\nEvaluating {'finetuned' if use_adapters else 'base'} model on {len(DOMAIN_QUESTIONS)} questions...\n")

    for i, q in enumerate(DOMAIN_QUESTIONS, 1):
        print(f"Question {i}/{len(DOMAIN_QUESTIONS)}: {q['question'][:50]}...")

        response = generate_response_internal(model, tokenizer, q["question"])
        eval_result = evaluate_response_internal(response, q["expected_keywords"])

        results.append({
            "question": q["question"],
            "category": q["category"],
            "response": response[:500],  # Truncate for output
            "score": eval_result["score"],
            "found_keywords": eval_result["found"],
            "missing_keywords": eval_result["missing"],
        })

        # Track by category
        cat = q["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(eval_result["score"])

        print(f"  Score: {eval_result['score']:.1%}")

    overall_score = sum(r["score"] for r in results) / len(results)
    category_averages = {cat: sum(s) / len(s) for cat, s in category_scores.items()}

    return {
        "model_type": "finetuned" if use_adapters else "base",
        "overall_score": overall_score,
        "category_scores": category_averages,
        "num_questions": len(DOMAIN_QUESTIONS),
        "detailed_results": results,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def run_comparison(val_texts: list[str], max_texts: int | None = None) -> dict:
    """
    Run full comparison between base and finetuned models.

    Args:
        val_texts: Validation texts for perplexity.
        max_texts: Maximum texts for perplexity eval.

    Returns:
        Complete comparison results.
    """
    import os
    import torch

    adapter_path = os.path.join(VOLUME_PATH, "adapters")
    if not os.path.exists(adapter_path):
        return {"error": "No adapters found. Run training first."}

    if max_texts:
        val_texts = val_texts[:max_texts]

    results = {
        "perplexity": {},
        "domain": {},
    }

    # -------------------------------------------------------------------------
    # Perplexity Evaluation
    # -------------------------------------------------------------------------
    print("=" * 60)
    print(" PERPLEXITY EVALUATION")
    print("=" * 60)

    # Base model perplexity
    print("\n--- Base Model ---")
    base_model, base_tokenizer = load_model_internal(None)
    base_ppl = calculate_perplexity_internal(base_model, base_tokenizer, val_texts)
    results["perplexity"]["base"] = base_ppl
    print(f"Base perplexity: {base_ppl['perplexity']:.2f}")

    # Free memory
    del base_model
    torch.cuda.empty_cache()

    # Finetuned model perplexity
    print("\n--- Finetuned Model ---")
    ft_model, ft_tokenizer = load_model_internal(adapter_path)
    ft_ppl = calculate_perplexity_internal(ft_model, ft_tokenizer, val_texts)
    results["perplexity"]["finetuned"] = ft_ppl
    print(f"Finetuned perplexity: {ft_ppl['perplexity']:.2f}")

    # Calculate improvement
    ppl_reduction = base_ppl["perplexity"] - ft_ppl["perplexity"]
    ppl_reduction_pct = (ppl_reduction / base_ppl["perplexity"]) * 100
    results["perplexity"]["improvement"] = {
        "reduction": ppl_reduction,
        "reduction_pct": ppl_reduction_pct,
    }

    # Free memory before domain eval
    del ft_model
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Domain Question Evaluation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" DOMAIN KNOWLEDGE EVALUATION")
    print("=" * 60)

    # Base model domain
    print("\n--- Base Model ---")
    base_model, base_tokenizer = load_model_internal(None)
    base_domain = {"results": [], "category_scores": {}}

    for i, q in enumerate(DOMAIN_QUESTIONS, 1):
        print(f"Q{i}: {q['question'][:40]}...")
        response = generate_response_internal(base_model, base_tokenizer, q["question"])
        eval_result = evaluate_response_internal(response, q["expected_keywords"])
        base_domain["results"].append({"score": eval_result["score"], "category": q["category"]})
        print(f"  Score: {eval_result['score']:.1%}")

    base_domain["overall_score"] = sum(r["score"] for r in base_domain["results"]) / len(base_domain["results"])
    results["domain"]["base"] = {"overall_score": base_domain["overall_score"]}

    del base_model
    torch.cuda.empty_cache()

    # Finetuned model domain
    print("\n--- Finetuned Model ---")
    ft_model, ft_tokenizer = load_model_internal(adapter_path)
    ft_domain = {"results": [], "category_scores": {}}

    for i, q in enumerate(DOMAIN_QUESTIONS, 1):
        print(f"Q{i}: {q['question'][:40]}...")
        response = generate_response_internal(ft_model, ft_tokenizer, q["question"])
        eval_result = evaluate_response_internal(response, q["expected_keywords"])
        ft_domain["results"].append({
            "question": q["question"],
            "score": eval_result["score"],
            "category": q["category"],
            "response": response[:300],
            "found": eval_result["found"],
            "missing": eval_result["missing"],
        })
        print(f"  Score: {eval_result['score']:.1%}")

    ft_domain["overall_score"] = sum(r["score"] for r in ft_domain["results"]) / len(ft_domain["results"])
    results["domain"]["finetuned"] = {
        "overall_score": ft_domain["overall_score"],
        "detailed_results": ft_domain["results"],
    }

    # Domain improvement
    domain_improvement = ft_domain["overall_score"] - base_domain["overall_score"]
    results["domain"]["improvement"] = {
        "score_change": domain_improvement,
        "score_change_pct": domain_improvement * 100,
    }

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\nPerplexity:")
    print(f"  Base:      {base_ppl['perplexity']:.2f}")
    print(f"  Finetuned: {ft_ppl['perplexity']:.2f}")
    print(f"  Change:    {ppl_reduction_pct:+.1f}%")

    print(f"\nDomain Knowledge:")
    print(f"  Base:      {base_domain['overall_score']:.1%}")
    print(f"  Finetuned: {ft_domain['overall_score']:.1%}")
    print(f"  Change:    {domain_improvement * 100:+.1f}%")

    if ppl_reduction > 0:
        print("\n✓ Perplexity improved (lower is better)")
    else:
        print("\n✗ Perplexity did not improve")

    if domain_improvement > 0:
        print("✓ Domain knowledge improved")
    else:
        print("✗ Domain knowledge did not improve")

    return results


def run_euroeval_benchmark(task: str = "sentiment-classification") -> dict:
    """
    EuroEval benchmark is currently disabled due to dependency conflicts.

    EuroEval requires accelerate>=1.9.0 but our training stack uses accelerate==0.28.0.
    To run EuroEval, use a separate environment or the standalone euroeval CLI.
    """
    return {
        "task": task,
        "status": "disabled",
        "error": "EuroEval disabled due to dependency conflicts (requires accelerate>=1.9.0)",
    }


# ============================================================================
# Local Entrypoint
# ============================================================================

@app.local_entrypoint()
def main(
    compare: bool = True,
    quick: bool = False,
    include_euroeval: bool = False,
    max_texts: int = 20,
):
    """
    Run evaluation suite on Modal.

    Args:
        compare: Compare base vs finetuned model.
        quick: Use fewer examples for faster eval.
        include_euroeval: Include EuroEval benchmark (slower).
        max_texts: Maximum validation texts for perplexity.
    """
    import json
    from pathlib import Path
    from datetime import datetime

    # Load validation data
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    val_path = data_dir / "val.jsonl"

    if not val_path.exists():
        print("ERROR: Validation data not found!")
        print(f"Expected: {val_path}")
        print("Run the data pipeline first.")
        return

    print(f"Loading validation data from: {val_path}")
    val_texts = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            val_texts.append(data["text"])

    print(f"Loaded {len(val_texts)} validation texts")

    if quick:
        max_texts = min(max_texts, 5)
        print(f"Quick mode: using {max_texts} texts")

    # Run comparison on Modal
    print("\n" + "=" * 60)
    print(" RUNNING EVALUATION ON MODAL")
    print("=" * 60)

    if compare:
        results = run_comparison.remote(val_texts, max_texts=max_texts)
    else:
        # Single model eval
        results = {
            "perplexity": evaluate_perplexity.remote(val_texts, use_adapters=True, max_texts=max_texts),
            "domain": evaluate_domain_questions.remote(use_adapters=True),
        }

    # Run EuroEval if requested
    if include_euroeval:
        print("\nNote: EuroEval is currently disabled due to dependency conflicts.")
        print("(EuroEval requires accelerate>=1.9.0, our stack uses 0.28.0)")
        results["euroeval"] = run_euroeval_benchmark("sentiment-classification")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"modal_eval_{timestamp}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to: {results_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print(" FINAL RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    if "perplexity" in results and "improvement" in results["perplexity"]:
        ppl = results["perplexity"]
        print(f"\nPerplexity:")
        print(f"  Base:      {ppl['base']['perplexity']:.2f}")
        print(f"  Finetuned: {ppl['finetuned']['perplexity']:.2f}")
        print(f"  Improvement: {ppl['improvement']['reduction_pct']:.1f}%")

    if "domain" in results and "improvement" in results["domain"]:
        domain = results["domain"]
        print(f"\nDomain Knowledge:")
        print(f"  Base:      {domain['base']['overall_score']:.1%}")
        print(f"  Finetuned: {domain['finetuned']['overall_score']:.1%}")
        print(f"  Improvement: {domain['improvement']['score_change_pct']:+.1f}%")

    print("\nDone!")
