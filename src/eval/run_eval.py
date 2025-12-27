"""
Main evaluation runner for Swedish Sovereign AI.

Runs all evaluation components and produces a comparison report between
the base Mistral model and the finetuned model.

Usage:
    # Compare base vs finetuned (all evals)
    python -m src.eval.run_eval --adapter-path /path/to/adapters --compare

    # Quick eval (perplexity + domain only, skip EuroEval)
    python -m src.eval.run_eval --adapter-path /path/to/adapters --compare --quick

    # Base model only
    python -m src.eval.run_eval

    # Finetuned model only
    python -m src.eval.run_eval --adapter-path /path/to/adapters
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Default configuration
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RESULTS_DIR = Path(__file__).parent / "results"


def run_perplexity_eval(
    adapter_path: str | None = None,
    compare: bool = False,
    max_examples: int | None = None,
) -> dict:
    """Run perplexity evaluation."""
    from src.eval.perplexity import compare_models, evaluate_perplexity

    print("\n" + "=" * 70)
    print(" PERPLEXITY EVALUATION")
    print("=" * 70)

    if compare and adapter_path:
        return compare_models(adapter_path, max_examples=max_examples)
    else:
        return evaluate_perplexity(adapter_path=adapter_path, max_examples=max_examples)


def run_domain_eval(
    adapter_path: str | None = None,
    compare: bool = False,
) -> dict:
    """Run domain knowledge evaluation."""
    from src.eval.domain_eval import compare_models, run_domain_evaluation

    print("\n" + "=" * 70)
    print(" DOMAIN KNOWLEDGE EVALUATION")
    print("=" * 70)

    if compare and adapter_path:
        return compare_models(adapter_path)
    else:
        return run_domain_evaluation(adapter_path=adapter_path)


def run_euroeval(
    adapter_path: str | None = None,
    compare: bool = False,
    quick: bool = False,
) -> dict:
    """Run EuroEval benchmark."""
    from src.eval.euroeval_benchmark import compare_models, run_benchmark, quick_benchmark

    print("\n" + "=" * 70)
    print(" EUROEVAL BENCHMARK")
    print("=" * 70)

    if quick:
        if compare and adapter_path:
            # Run quick benchmark on both
            print("Running quick benchmark (sentiment only)...")
            base_results = quick_benchmark(model_id=BASE_MODEL_NAME)
            ft_results = quick_benchmark(adapter_path=adapter_path)
            return {
                "base_model": base_results,
                "finetuned_model": ft_results,
            }
        else:
            return quick_benchmark(adapter_path=adapter_path)
    else:
        if compare and adapter_path:
            return compare_models(adapter_path)
        else:
            return run_benchmark(adapter_path=adapter_path)


def generate_report(results: dict, output_path: Path) -> str:
    """
    Generate a markdown report from evaluation results.

    Args:
        results: Evaluation results dictionary.
        output_path: Where to save the report.

    Returns:
        Report content as string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Swedish Sovereign AI - Evaluation Report

**Generated:** {timestamp}
**Base Model:** {BASE_MODEL_NAME}
**Adapter Path:** {results.get('adapter_path', 'N/A')}

---

## Summary

"""
    # Perplexity summary
    if "perplexity" in results:
        ppl = results["perplexity"]
        if "improvement" in ppl:
            report += f"""### Perplexity (Domain Adaptation)

| Model | Perplexity |
|-------|------------|
| Base | {ppl['base_model']['perplexity']:.2f} |
| Finetuned | {ppl['finetuned_model']['perplexity']:.2f} |

**Improvement:** {ppl['improvement']['perplexity_reduction_pct']:.1f}% reduction

"""
        else:
            report += f"""### Perplexity
- Score: {ppl.get('perplexity', 'N/A')}

"""

    # Domain eval summary
    if "domain" in results:
        domain = results["domain"]
        if "improvement" in domain:
            report += f"""### Domain Knowledge (Riksbanken Terminology)

| Model | Score |
|-------|-------|
| Base | {domain['base_model']['overall_score']:.1%} |
| Finetuned | {domain['finetuned_model']['overall_score']:.1%} |

**Improvement:** {domain['improvement']['overall_pct']:+.1f}%

"""
        else:
            report += f"""### Domain Knowledge
- Score: {domain.get('overall_score', 'N/A')}

"""

    # EuroEval summary
    if "euroeval" in results:
        euroeval = results["euroeval"]
        report += """### EuroEval Benchmark (Swedish)

"""
        if "base_model" in euroeval and "finetuned_model" in euroeval:
            report += "| Task | Base Status | Finetuned Status |\n"
            report += "|------|-------------|------------------|\n"

            base_results = euroeval.get("base_model", {}).get("results", {})
            ft_results = euroeval.get("finetuned_model", {}).get("results", {})

            all_tasks = set(base_results.keys()) | set(ft_results.keys())
            for task in sorted(all_tasks):
                base_status = base_results.get(task, {}).get("status", "N/A")
                ft_status = ft_results.get(task, {}).get("status", "N/A")
                report += f"| {task} | {base_status} | {ft_status} |\n"
        else:
            for task, result in euroeval.items():
                if isinstance(result, dict):
                    status = result.get("status", "unknown")
                    report += f"- {task}: {status}\n"

    report += """
---

## Interpretation

### Perplexity
Lower perplexity indicates better prediction of domain text. A significant reduction
shows the model has learned the language patterns of Riksbanken reports.

### Domain Knowledge
Higher keyword score means the model uses more domain-appropriate terminology
when answering questions about Swedish monetary policy.

### EuroEval
Standard benchmarks for Swedish NLU. These may not improve with domain finetuning
since we're specializing for a narrow domain, not general Swedish.

---

## Recommendations

"""
    # Add recommendations based on results
    if "perplexity" in results and "improvement" in results["perplexity"]:
        ppl_improvement = results["perplexity"]["improvement"]["perplexity_reduction_pct"]
        if ppl_improvement > 10:
            report += "- ✓ Strong domain adaptation in perplexity\n"
        elif ppl_improvement > 0:
            report += "- ~ Modest domain adaptation, consider more training data\n"
        else:
            report += "- ✗ No perplexity improvement, review training data quality\n"

    if "domain" in results and "improvement" in results["domain"]:
        domain_improvement = results["domain"]["improvement"]["overall_pct"]
        if domain_improvement > 10:
            report += "- ✓ Strong improvement in domain terminology\n"
        elif domain_improvement > 0:
            report += "- ~ Some improvement in terminology, model is learning\n"
        else:
            report += "- ✗ No terminology improvement, consider instruction tuning\n"

    report += "\n---\n\n*Report generated by Swedish Sovereign AI evaluation suite*\n"

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def run_all_evals(
    adapter_path: str | None = None,
    compare: bool = False,
    quick: bool = False,
    skip_euroeval: bool = False,
    max_perplexity_examples: int | None = None,
) -> dict:
    """
    Run all evaluation components.

    Args:
        adapter_path: Path to LoRA adapters.
        compare: Whether to compare base vs finetuned.
        quick: Run quick version (fewer examples, sentiment only).
        skip_euroeval: Skip EuroEval benchmark (faster).
        max_perplexity_examples: Limit perplexity eval examples.

    Returns:
        Combined results dictionary.
    """
    print("=" * 70)
    print(" SWEDISH SOVEREIGN AI - FULL EVALUATION")
    print("=" * 70)
    print(f"\nBase Model: {BASE_MODEL_NAME}")
    print(f"Adapter Path: {adapter_path or 'None (base model only)'}")
    print(f"Compare Mode: {compare}")
    print(f"Quick Mode: {quick}")
    print(f"Skip EuroEval: {skip_euroeval}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "base_model": BASE_MODEL_NAME,
        "adapter_path": adapter_path,
        "compare_mode": compare,
    }

    # 1. Perplexity evaluation
    try:
        max_examples = 10 if quick else max_perplexity_examples
        results["perplexity"] = run_perplexity_eval(
            adapter_path=adapter_path,
            compare=compare,
            max_examples=max_examples,
        )
    except Exception as e:
        print(f"\nPerplexity eval failed: {e}")
        results["perplexity"] = {"error": str(e)}

    # 2. Domain knowledge evaluation
    try:
        results["domain"] = run_domain_eval(
            adapter_path=adapter_path,
            compare=compare,
        )
    except Exception as e:
        print(f"\nDomain eval failed: {e}")
        results["domain"] = {"error": str(e)}

    # 3. EuroEval benchmark (optional)
    if not skip_euroeval:
        try:
            results["euroeval"] = run_euroeval(
                adapter_path=adapter_path,
                compare=compare,
                quick=quick,
            )
        except Exception as e:
            print(f"\nEuroEval failed: {e}")
            results["euroeval"] = {"error": str(e)}
    else:
        print("\nSkipping EuroEval benchmark...")
        results["euroeval"] = {"status": "skipped"}

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"eval_results_{timestamp}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nRaw results saved to: {results_path}")

    # Generate report
    report_path = RESULTS_DIR / f"eval_report_{timestamp}.md"
    report = generate_report(results, report_path)
    print(f"Report saved to: {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETE")
    print("=" * 70)

    if compare and "perplexity" in results and "improvement" in results.get("perplexity", {}):
        ppl = results["perplexity"]
        print(f"\nPerplexity: {ppl['improvement']['perplexity_reduction_pct']:.1f}% improvement")

    if compare and "domain" in results and "improvement" in results.get("domain", {}):
        domain = results["domain"]
        print(f"Domain Knowledge: {domain['improvement']['overall_pct']:+.1f}% improvement")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run evaluation suite for Swedish Sovereign AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare base vs finetuned (all evals)
    python -m src.eval.run_eval --adapter-path ./adapters --compare

    # Quick comparison (faster)
    python -m src.eval.run_eval --adapter-path ./adapters --compare --quick

    # Skip EuroEval (just perplexity + domain)
    python -m src.eval.run_eval --adapter-path ./adapters --compare --skip-euroeval

    # Base model only
    python -m src.eval.run_eval
        """,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapters",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model vs finetuned model",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation (fewer examples)",
    )
    parser.add_argument(
        "--skip-euroeval",
        action="store_true",
        help="Skip EuroEval benchmark (faster)",
    )
    parser.add_argument(
        "--max-perplexity-examples",
        type=int,
        default=None,
        help="Maximum examples for perplexity eval",
    )
    parser.add_argument(
        "--perplexity-only",
        action="store_true",
        help="Only run perplexity evaluation",
    )
    parser.add_argument(
        "--domain-only",
        action="store_true",
        help="Only run domain evaluation",
    )
    parser.add_argument(
        "--euroeval-only",
        action="store_true",
        help="Only run EuroEval benchmark",
    )

    args = parser.parse_args()

    # Single eval mode
    if args.perplexity_only:
        results = run_perplexity_eval(
            adapter_path=args.adapter_path,
            compare=args.compare,
            max_examples=args.max_perplexity_examples,
        )
        print(json.dumps(results, indent=2, default=str))
        return

    if args.domain_only:
        results = run_domain_eval(
            adapter_path=args.adapter_path,
            compare=args.compare,
        )
        return

    if args.euroeval_only:
        results = run_euroeval(
            adapter_path=args.adapter_path,
            compare=args.compare,
            quick=args.quick,
        )
        print(json.dumps(results, indent=2, default=str))
        return

    # Full evaluation
    run_all_evals(
        adapter_path=args.adapter_path,
        compare=args.compare,
        quick=args.quick,
        skip_euroeval=args.skip_euroeval,
        max_perplexity_examples=args.max_perplexity_examples,
    )


if __name__ == "__main__":
    main()
