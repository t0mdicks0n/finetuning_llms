"""
Domain-specific evaluation for Swedish financial/monetary policy knowledge.

Tests the model's understanding of Riksbanken terminology and concepts.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Default model configuration
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass
class DomainQuestion:
    """A domain-specific evaluation question."""
    question: str
    expected_keywords: list[str]  # Keywords that should appear in a good answer
    category: str  # e.g., "monetary_policy", "inflation", "interest_rates"


# Swedish Riksbanken domain questions
# These test knowledge of Swedish monetary policy terminology and concepts
DOMAIN_QUESTIONS = [
    # Monetary Policy Basics
    DomainQuestion(
        question="Vad är reporäntan och hur används den av Riksbanken?",
        expected_keywords=["styrränta", "penningpolitik", "banker", "lån", "procent"],
        category="interest_rates",
    ),
    DomainQuestion(
        question="Vad är Riksbankens inflationsmål?",
        expected_keywords=["2", "procent", "KPIF", "KPI", "prisstabilitet"],
        category="inflation",
    ),
    DomainQuestion(
        question="Förklara begreppet penningpolitisk transmission.",
        expected_keywords=["ränta", "ekonomi", "kanal", "kredit", "växelkurs"],
        category="monetary_policy",
    ),

    # Economic Indicators
    DomainQuestion(
        question="Vad mäter KPIF och varför använder Riksbanken detta mått?",
        expected_keywords=["inflation", "konsumentpris", "räntekostnad", "mått", "exkludera"],
        category="inflation",
    ),
    DomainQuestion(
        question="Hur påverkar Riksbankens räntebeslut den svenska kronan?",
        expected_keywords=["växelkurs", "SEK", "kapital", "flöde", "stark", "svag"],
        category="exchange_rates",
    ),

    # Policy Tools
    DomainQuestion(
        question="Vilka verktyg har Riksbanken för att bedriva penningpolitik?",
        expected_keywords=["reporänta", "tillgångsköp", "likviditet", "obligationer"],
        category="monetary_policy",
    ),
    DomainQuestion(
        question="Vad innebär kvantitativa lättnader och när används det?",
        expected_keywords=["tillgångsköp", "obligationer", "likviditet", "stimulera", "nollränta"],
        category="monetary_policy",
    ),

    # Swedish Economic Context
    DomainQuestion(
        question="Hur påverkar internationell konjunktur svensk ekonomi enligt Riksbanken?",
        expected_keywords=["export", "handel", "global", "tillväxt", "euroområdet"],
        category="economy",
    ),
    DomainQuestion(
        question="Vilken roll spelar hushållens skuldsättning för penningpolitiken?",
        expected_keywords=["skuld", "bolån", "risker", "finansiell", "stabilitet"],
        category="financial_stability",
    ),
    DomainQuestion(
        question="Hur analyserar Riksbanken arbetsmarknaden i sina rapporter?",
        expected_keywords=["sysselsättning", "arbetslöshet", "löner", "resursutnyttjande"],
        category="labor_market",
    ),
]


def load_model(
    model_name: str = BASE_MODEL_NAME,
    adapter_path: str | None = None,
) -> tuple:
    """Load model and tokenizer."""
    print(f"Loading base model: {model_name}")

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

    if adapter_path:
        print(f"Loading LoRA adapters from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
) -> str:
    """Generate a response to a question."""
    # Format for Mistral Instruct
    messages = [
        {
            "role": "user",
            "content": f"Du är en svensk ekonomisk analytiker som arbetar på Riksbanken. "
                      f"Svara på följande fråga på svenska med korrekt ekonomisk terminologi:\n\n{question}"
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant response
    if "[/INST]" in response:
        response = response.split("[/INST]")[-1].strip()

    return response


def evaluate_response(response: str, expected_keywords: list[str]) -> dict:
    """
    Evaluate a response based on keyword presence.

    Args:
        response: Model's response text.
        expected_keywords: Keywords that should appear in a good answer.

    Returns:
        Evaluation metrics.
    """
    response_lower = response.lower()

    found_keywords = []
    missing_keywords = []

    for keyword in expected_keywords:
        # Check for keyword (case-insensitive)
        if keyword.lower() in response_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    keyword_score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

    return {
        "keyword_score": keyword_score,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "response_length": len(response),
    }


def run_domain_evaluation(
    adapter_path: str | None = None,
    base_model: str = BASE_MODEL_NAME,
    questions: list[DomainQuestion] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run domain evaluation on a model.

    Args:
        adapter_path: Path to LoRA adapters (None for base model).
        base_model: Base model name.
        questions: List of questions to evaluate (defaults to DOMAIN_QUESTIONS).
        verbose: Whether to print detailed output.

    Returns:
        Evaluation results.
    """
    if questions is None:
        questions = DOMAIN_QUESTIONS

    # Load model
    model, tokenizer = load_model(base_model, adapter_path)

    results = []
    category_scores = {}

    model_type = "finetuned" if adapter_path else "base"
    print(f"\n{'=' * 60}")
    print(f"DOMAIN EVALUATION - {model_type.upper()} MODEL")
    print(f"{'=' * 60}\n")

    for i, q in enumerate(questions, 1):
        if verbose:
            print(f"Question {i}/{len(questions)}: {q.question[:50]}...")

        # Generate response
        response = generate_response(model, tokenizer, q.question)

        # Evaluate
        eval_result = evaluate_response(response, q.expected_keywords)

        result = {
            "question": q.question,
            "category": q.category,
            "response": response,
            "keyword_score": eval_result["keyword_score"],
            "found_keywords": eval_result["found_keywords"],
            "missing_keywords": eval_result["missing_keywords"],
        }
        results.append(result)

        # Track category scores
        if q.category not in category_scores:
            category_scores[q.category] = []
        category_scores[q.category].append(eval_result["keyword_score"])

        if verbose:
            print(f"  Score: {eval_result['keyword_score']:.1%}")
            print(f"  Found: {eval_result['found_keywords']}")
            if eval_result["missing_keywords"]:
                print(f"  Missing: {eval_result['missing_keywords']}")
            print()

    # Calculate overall metrics
    overall_score = sum(r["keyword_score"] for r in results) / len(results)

    category_averages = {
        cat: sum(scores) / len(scores)
        for cat, scores in category_scores.items()
    }

    summary = {
        "model_type": model_type,
        "adapter_path": adapter_path,
        "overall_score": overall_score,
        "num_questions": len(questions),
        "category_scores": category_averages,
        "detailed_results": results,
    }

    # Print summary
    print("-" * 40)
    print(f"Overall Score: {overall_score:.1%}")
    print("\nBy Category:")
    for cat, score in sorted(category_averages.items()):
        print(f"  {cat}: {score:.1%}")

    return summary


def compare_models(
    adapter_path: str,
    base_model: str = BASE_MODEL_NAME,
    questions: list[DomainQuestion] | None = None,
) -> dict:
    """
    Compare base model and finetuned model on domain questions.

    Args:
        adapter_path: Path to LoRA adapters.
        base_model: Base model name.
        questions: Questions to evaluate.

    Returns:
        Comparison results.
    """
    print("=" * 60)
    print("DOMAIN KNOWLEDGE EVALUATION")
    print("=" * 60)

    # Evaluate base model
    base_results = run_domain_evaluation(
        adapter_path=None,
        base_model=base_model,
        questions=questions,
        verbose=True,
    )

    # Free memory
    torch.cuda.empty_cache()

    # Evaluate finetuned model
    ft_results = run_domain_evaluation(
        adapter_path=adapter_path,
        base_model=base_model,
        questions=questions,
        verbose=True,
    )

    # Calculate improvement
    score_improvement = ft_results["overall_score"] - base_results["overall_score"]

    comparison = {
        "base_model": {
            "overall_score": base_results["overall_score"],
            "category_scores": base_results["category_scores"],
        },
        "finetuned_model": {
            "overall_score": ft_results["overall_score"],
            "category_scores": ft_results["category_scores"],
        },
        "improvement": {
            "overall": score_improvement,
            "overall_pct": score_improvement * 100,
        },
    }

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"\nBase model score:      {base_results['overall_score']:.1%}")
    print(f"Finetuned model score: {ft_results['overall_score']:.1%}")
    print(f"\nImprovement: {score_improvement:+.1%}")

    if score_improvement > 0:
        print("\n✓ Finetuning improved domain knowledge!")
    else:
        print("\n✗ No improvement in domain knowledge")

    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Domain knowledge evaluation")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapters",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base vs finetuned model",
    )

    args = parser.parse_args()

    if args.compare and args.adapter_path:
        results = compare_models(args.adapter_path)
    else:
        results = run_domain_evaluation(adapter_path=args.adapter_path)

    # Save results
    output_path = Path(__file__).parent / "domain_eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        # Remove detailed responses for cleaner JSON
        results_clean = {k: v for k, v in results.items() if k != "detailed_results"}
        json.dump(results_clean, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
