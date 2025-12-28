"""
EuroEval (formerly ScandEval) benchmark wrapper.

Runs standardized Swedish language benchmarks for:
- Sentiment classification (SweReC)
- Named Entity Recognition
- Question Answering (ScandiQA)
- And more...

Installation:
    pip install euroeval[all]

Usage:
    python -m src.eval.euroeval_benchmark --model <model-id>
    python -m src.eval.euroeval_benchmark --compare --adapter-path /path/to/adapters
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Default model (must match train.py)
BASE_MODEL_NAME = "mistralai/Ministral-3-8B-Instruct-2512"

# Swedish tasks to run (subset for faster evaluation)
SWEDISH_TASKS = [
    "sentiment-classification",  # SweReC
    "named-entity-recognition",  # Swedish NER
    "reading-comprehension",     # ScandiQA
]

# Full task list (takes longer)
ALL_SWEDISH_TASKS = [
    "sentiment-classification",
    "named-entity-recognition",
    "reading-comprehension",
    "linguistic-acceptability",
    "knowledge",
    "common-sense-reasoning",
]


def check_euroeval_installed() -> bool:
    """Check if EuroEval is installed."""
    try:
        import euroeval
        return True
    except ImportError:
        return False


def install_euroeval():
    """Install EuroEval package."""
    print("Installing EuroEval...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "euroeval[all]"
    ])
    print("EuroEval installed successfully!")


def run_euroeval_cli(
    model_id: str,
    tasks: list[str] | None = None,
    language: str = "sv",
    output_dir: Path | None = None,
) -> dict:
    """
    Run EuroEval benchmark using CLI.

    Args:
        model_id: HuggingFace model ID or local path.
        tasks: List of tasks to run (None for all Swedish tasks).
        language: Language code (sv for Swedish).
        output_dir: Directory to save results.

    Returns:
        Benchmark results.
    """
    if tasks is None:
        tasks = SWEDISH_TASKS

    if output_dir is None:
        output_dir = Path(__file__).parent / "euroeval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for task in tasks:
        print(f"\n{'=' * 50}")
        print(f"Running EuroEval: {task}")
        print(f"{'=' * 50}")

        cmd = [
            "euroeval",
            "--model", model_id,
            "--task", task,
            "--language", language,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min timeout per task
            )

            if result.returncode == 0:
                # Parse output for scores
                output = result.stdout
                results[task] = {
                    "status": "success",
                    "output": output,
                }
                print(output)
            else:
                results[task] = {
                    "status": "error",
                    "error": result.stderr,
                }
                print(f"Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            results[task] = {
                "status": "timeout",
                "error": "Task timed out after 30 minutes",
            }
            print("Task timed out")

        except Exception as e:
            results[task] = {
                "status": "error",
                "error": str(e),
            }
            print(f"Error: {e}")

    return results


def run_euroeval_python(
    model_id: str,
    tasks: list[str] | None = None,
    language: str = "sv",
) -> dict:
    """
    Run EuroEval benchmark using Python API.

    Args:
        model_id: HuggingFace model ID or local path.
        tasks: List of tasks to run.
        language: Language code.

    Returns:
        Benchmark results.
    """
    try:
        from euroeval import Benchmarker
    except ImportError:
        print("EuroEval not installed. Installing...")
        install_euroeval()
        from euroeval import Benchmarker

    if tasks is None:
        tasks = SWEDISH_TASKS

    benchmarker = Benchmarker()
    results = {}

    for task in tasks:
        print(f"\n{'=' * 50}")
        print(f"Running EuroEval: {task} (Swedish)")
        print(f"{'=' * 50}")

        try:
            result = benchmarker.benchmark(
                model=model_id,
                task=task,
                language=language,
            )
            results[task] = {
                "status": "success",
                "scores": result,
            }
            print(f"Result: {result}")

        except Exception as e:
            results[task] = {
                "status": "error",
                "error": str(e),
            }
            print(f"Error: {e}")

    return results


def merge_and_save_model(
    base_model: str,
    adapter_path: str,
    output_path: str,
) -> str:
    """
    Merge LoRA adapters with base model for EuroEval.

    EuroEval expects a complete model, so we need to merge adapters.

    Args:
        base_model: Base model name.
        adapter_path: Path to LoRA adapters.
        output_path: Where to save merged model.

    Returns:
        Path to merged model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Merging adapters for EuroEval...")
    print(f"  Base model: {base_model}")
    print(f"  Adapters: {adapter_path}")

    # Load base model (full precision for merging)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load and merge adapters
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    print(f"  Saving to: {output_path}")
    model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)

    return output_path


def run_benchmark(
    model_id: str | None = None,
    adapter_path: str | None = None,
    base_model: str = BASE_MODEL_NAME,
    tasks: list[str] | None = None,
    use_cli: bool = False,
) -> dict:
    """
    Run EuroEval benchmark on a model.

    Args:
        model_id: Model ID to benchmark (if not using adapters).
        adapter_path: Path to LoRA adapters (will merge with base).
        base_model: Base model for merging.
        tasks: Tasks to run.
        use_cli: Whether to use CLI instead of Python API.

    Returns:
        Benchmark results.
    """
    if not check_euroeval_installed():
        print("EuroEval not installed.")
        print("Install with: pip install euroeval[all]")
        return {"error": "EuroEval not installed"}

    # Determine model to benchmark
    if adapter_path:
        # Need to merge adapters first
        merged_path = Path(adapter_path).parent / "merged_for_eval"
        model_id = merge_and_save_model(base_model, adapter_path, str(merged_path))
    elif model_id is None:
        model_id = base_model

    print(f"\nBenchmarking model: {model_id}")

    if use_cli:
        return run_euroeval_cli(model_id, tasks)
    else:
        return run_euroeval_python(model_id, tasks)


def compare_models(
    adapter_path: str,
    base_model: str = BASE_MODEL_NAME,
    tasks: list[str] | None = None,
) -> dict:
    """
    Compare base model and finetuned model on EuroEval benchmarks.

    Args:
        adapter_path: Path to LoRA adapters.
        base_model: Base model name.
        tasks: Tasks to run.

    Returns:
        Comparison results.
    """
    print("=" * 60)
    print("EUROEVAL BENCHMARK COMPARISON")
    print("=" * 60)

    # Benchmark base model
    print("\n" + "-" * 40)
    print("Benchmarking BASE model...")
    print("-" * 40)
    base_results = run_benchmark(
        model_id=base_model,
        tasks=tasks,
    )

    # Benchmark finetuned model
    print("\n" + "-" * 40)
    print("Benchmarking FINETUNED model...")
    print("-" * 40)
    ft_results = run_benchmark(
        adapter_path=adapter_path,
        base_model=base_model,
        tasks=tasks,
    )

    comparison = {
        "base_model": {
            "model": base_model,
            "results": base_results,
        },
        "finetuned_model": {
            "adapter_path": adapter_path,
            "results": ft_results,
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EUROEVAL COMPARISON RESULTS")
    print("=" * 60)
    print("\nBase Model Results:")
    for task, result in base_results.items():
        status = result.get("status", "unknown")
        print(f"  {task}: {status}")

    print("\nFinetuned Model Results:")
    for task, result in ft_results.items():
        status = result.get("status", "unknown")
        print(f"  {task}: {status}")

    return comparison


def quick_benchmark(
    model_id: str | None = None,
    adapter_path: str | None = None,
) -> dict:
    """
    Run a quick benchmark with just sentiment classification.

    Good for testing the pipeline without waiting for full eval.
    """
    return run_benchmark(
        model_id=model_id,
        adapter_path=adapter_path,
        tasks=["sentiment-classification"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EuroEval benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to benchmark",
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
        help="Compare base vs finetuned model",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (sentiment only)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all Swedish tasks (slower)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use CLI instead of Python API",
    )

    args = parser.parse_args()

    tasks = ALL_SWEDISH_TASKS if args.all_tasks else SWEDISH_TASKS

    if args.quick:
        results = quick_benchmark(args.model, args.adapter_path)
    elif args.compare and args.adapter_path:
        results = compare_models(args.adapter_path, tasks=tasks)
    else:
        results = run_benchmark(
            model_id=args.model,
            adapter_path=args.adapter_path,
            tasks=tasks,
            use_cli=args.cli,
        )

    # Save results
    output_path = Path(__file__).parent / "euroeval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
