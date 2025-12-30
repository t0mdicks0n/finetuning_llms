"""
Build and evaluate the semantic router for model selection.

Uses semantic-router library to classify prompts as either:
- riksbanken: Swedish monetary policy, central banking, Riksbanken
- general: Everything else

Usage:
    python -m src.models.router.train
    python -m src.models.router.train --evaluate
"""

import json
import pickle
from pathlib import Path

from .config import (
    ENCODER_MODEL,
    ROUTE_RIKSBANKEN,
    ROUTE_GENERAL,
    RIKSBANKEN_EXAMPLES_PATH,
    GENERAL_EXAMPLES_PATH,
    TEST_EXAMPLES_PATH,
    ROUTER_ARTIFACT_PATH,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def load_examples(path: Path) -> list[str]:
    """Load example utterances from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            examples.append(data["text"])
    return examples


def build_router():
    """
    Build the semantic router from training examples.

    Returns:
        SemanticRouter: Configured semantic router
    """
    from semantic_router import Route
    from semantic_router.routers import SemanticRouter
    from semantic_router.encoders import HuggingFaceEncoder

    # Load examples
    riksbanken_path = PROJECT_ROOT / RIKSBANKEN_EXAMPLES_PATH
    general_path = PROJECT_ROOT / GENERAL_EXAMPLES_PATH

    if not riksbanken_path.exists():
        raise FileNotFoundError(
            f"Riksbanken examples not found at {riksbanken_path}\n"
            "Run: python -m src.models.router.generate_examples"
        )

    riksbanken_examples = load_examples(riksbanken_path)
    print(f"Loaded {len(riksbanken_examples)} riksbanken examples")

    general_examples = []
    if general_path.exists():
        general_examples = load_examples(general_path)
        print(f"Loaded {len(general_examples)} general examples")
    else:
        print("WARNING: No general examples found, router will only have riksbanken route")

    # Create encoder
    print(f"\nLoading encoder: {ENCODER_MODEL}")
    encoder = HuggingFaceEncoder(name=ENCODER_MODEL)

    # Define routes
    routes = [
        Route(
            name=ROUTE_RIKSBANKEN,
            utterances=riksbanken_examples,
        ),
    ]

    if general_examples:
        routes.append(
            Route(
                name=ROUTE_GENERAL,
                utterances=general_examples,
            )
        )

    # Build router
    print("Building semantic router...")
    router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

    return router


def evaluate_router(router) -> dict:
    """
    Evaluate router accuracy on test set.

    Returns:
        dict with accuracy metrics
    """
    test_path = PROJECT_ROOT / TEST_EXAMPLES_PATH

    if not test_path.exists():
        print(f"No test set found at {test_path}")
        return {}

    # Load test examples
    test_examples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            test_examples.append(json.loads(line))

    print(f"\nEvaluating on {len(test_examples)} test examples...")

    correct = 0
    results_by_route = {ROUTE_RIKSBANKEN: {"correct": 0, "total": 0}, ROUTE_GENERAL: {"correct": 0, "total": 0}}

    errors = []
    for example in test_examples:
        text = example["text"]
        expected = example["route"]

        result = router(text)
        # Default to general if no match (safer - use vanilla model when uncertain)
        predicted = result.name if result and result.name else ROUTE_GENERAL

        results_by_route[expected]["total"] += 1
        if predicted == expected:
            correct += 1
            results_by_route[expected]["correct"] += 1
        else:
            errors.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "expected": expected,
                "predicted": predicted,
            })

    accuracy = correct / len(test_examples) if test_examples else 0

    print(f"\nOverall accuracy: {accuracy:.1%} ({correct}/{len(test_examples)})")
    for route, stats in results_by_route.items():
        if stats["total"] > 0:
            route_acc = stats["correct"] / stats["total"]
            print(f"  {route}: {route_acc:.1%} ({stats['correct']}/{stats['total']})")

    if errors:
        print(f"\nMisclassified examples ({len(errors)}):")
        for err in errors[:5]:  # Show first 5
            print(f"  - \"{err['text']}\"")
            print(f"    Expected: {err['expected']}, Got: {err['predicted']}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_examples),
        "by_route": results_by_route,
        "errors": errors,
    }


def save_router(router, path: Path):
    """Save router to pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(router, f)
    print(f"Saved router to {path}")


def load_router(path: Path):
    """Load router from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def demo_router(router):
    """Interactive demo of the router."""
    print("\n" + "=" * 60)
    print("Router Demo (type 'quit' to exit)")
    print("=" * 60)

    test_queries = [
        "Vad är reporäntan?",
        "Hur ser Riksbankens inflationsprognos ut?",
        "Vilken är huvudstaden i Japan?",
        "Vad är fotosyntesen?",
        "Hur påverkar penningpolitiken hushållen?",
        "Vem skrev Pippi Långstrump?",
    ]

    print("\nSample classifications:")
    for query in test_queries:
        result = router(query)
        route_name = result.name if result and result.name else "(no match)"
        print(f"  [{route_name:12}] {query}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build and evaluate semantic router")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--save", action="store_true", help="Save router to artifacts")
    parser.add_argument("--load", action="store_true", help="Load existing router instead of building")
    args = parser.parse_args()

    print("=" * 60)
    print("Semantic Router - Training & Evaluation")
    print("=" * 60)

    artifact_path = PROJECT_ROOT / ROUTER_ARTIFACT_PATH

    if args.load and artifact_path.exists():
        print(f"\nLoading router from {artifact_path}")
        router = load_router(artifact_path)
    else:
        router = build_router()

    if args.evaluate:
        evaluate_router(router)

    if args.demo:
        demo_router(router)

    if args.save:
        save_router(router, artifact_path)

    if not (args.evaluate or args.demo or args.save):
        # Default: evaluate and demo
        evaluate_router(router)
        demo_router(router)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
