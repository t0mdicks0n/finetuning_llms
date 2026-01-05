"""
Evaluation suite for Swedish Sovereign AI.

Components:
    - perplexity: Measures domain adaptation via perplexity on held-out data
    - domain_eval: Tests Riksbanken terminology and monetary policy knowledge
    - euroeval_benchmark: Runs standardized Swedish language benchmarks
    - run_eval: Main runner that combines all evaluations
    - eval_modal: Modal-based evaluation (runs on cloud GPU)

Usage (Modal - recommended, no local GPU needed):
    # Full comparison on Modal GPU
    modal run src/eval/eval_modal.py --compare

    # Quick eval on Modal
    modal run src/eval/eval_modal.py --compare --quick

    # Include EuroEval benchmark
    modal run src/eval/eval_modal.py --compare --include-euroeval

Usage (Local - requires GPU):
    python -m src.eval.run_eval --adapter-path /path/to/adapters --compare
    python -m src.eval.run_eval --adapter-path /path/to/adapters --compare --skip-euroeval
"""

from shared.eval.perplexity import evaluate_perplexity, compare_models as compare_perplexity
from shared.eval.domain_eval import run_domain_evaluation, compare_models as compare_domain
from shared.eval.euroeval_benchmark import run_benchmark as run_euroeval

__all__ = [
    "evaluate_perplexity",
    "compare_perplexity",
    "run_domain_evaluation",
    "compare_domain",
    "run_euroeval",
]
