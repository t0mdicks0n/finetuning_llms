"""
Generate training examples for the semantic router.

1. Extracts riksbanken questions from existing Q&A training data
2. Generates general Swedish questions using Gemini API

Usage:
    python -m src.models.router.generate_examples
    python -m src.models.router.generate_examples --dry-run
    python -m src.models.router.generate_examples --riksbanken-only
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

from .config import (
    RIKSBANKEN_EXAMPLES_PATH,
    GENERAL_EXAMPLES_PATH,
    TEST_EXAMPLES_PATH,
)

load_dotenv()

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ROUTER_DIR = DATA_DIR / "router"

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


def get_model():
    return genai.GenerativeModel(
        "gemini-2.0-flash",
        generation_config=genai.GenerationConfig(temperature=0.8),
    )


def extract_riksbanken_questions(max_examples: int = 200) -> list[str]:
    """
    Extract questions from existing Q&A training data.

    These are perfect domain examples - questions about Riksbanken,
    Swedish monetary policy, inflation, interest rates, etc.
    """
    train_path = PROCESSED_DIR / "train_qa.jsonl"

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Run the Q&A generation pipeline first:")
        print("  python -m src.data.generate_qa")
        return []

    questions = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            question = example["messages"][0]["content"]
            questions.append(question)

    # Shuffle and limit
    random.shuffle(questions)
    questions = questions[:max_examples]

    print(f"Extracted {len(questions)} riksbanken questions from training data")
    return questions


GENERAL_QUESTIONS_PROMPT = """
Generera 50 varierade frågor på svenska som en användare kan ställa till en AI-assistent.

Frågorna ska vara UTANFÖR området svensk penningpolitik och Riksbanken. De ska täcka:

1. Svensk geografi och historia (10 frågor)
   - Exempel: "Vilken är Sveriges näst största stad?", "När blev Sverige medlem i EU?"

2. Vetenskap och natur (10 frågor)
   - Exempel: "Vad är fotosyntesen?", "Hur fungerar gravitationen?"

3. Kultur och samhälle (10 frågor)
   - Exempel: "Vem skrev Pippi Långstrump?", "Vad är midsommar?"

4. Internationellt (10 frågor)
   - Exempel: "Vilken är huvudstaden i Japan?", "Hur många invånare har Kina?"

5. Vardagsfrågor och praktiskt (10 frågor)
   - Exempel: "Hur lagar man pannkakor?", "Vad betyder ordet 'lagom'?"

VIKTIGT:
- Undvik ALLA frågor om ekonomi, banker, räntor, inflation, valutor eller finansmarknader
- Frågorna ska vara naturliga och varierade i stil
- Blanda korta och längre frågor
- Inkludera både faktafrågor och förklaringsfrågor

Returnera ENDAST en JSON-array med frågor, utan markdown:
["fråga 1", "fråga 2", ...]
"""


def generate_general_questions(num_batches: int = 2) -> list[str]:
    """
    Generate general Swedish questions using Gemini.

    These are questions outside the Riksbanken domain that should
    be routed to the vanilla model.
    """
    model = get_model()
    all_questions = []

    for i in range(num_batches):
        print(f"  Generating batch {i + 1}/{num_batches}...")
        try:
            response = model.generate_content(GENERAL_QUESTIONS_PROMPT)
            text = response.text

            # Parse JSON
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end > 0:
                questions = json.loads(text[start:end])
                all_questions.extend(questions)
                print(f"    Got {len(questions)} questions")
        except Exception as e:
            print(f"    Error in batch {i + 1}: {e}")

    # Deduplicate
    unique_questions = list(set(all_questions))
    print(f"Generated {len(unique_questions)} unique general questions")

    return unique_questions


def save_examples(questions: list[str], output_path: Path, route: str):
    """Save questions to JSONL file with route label."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for q in questions:
            example = {"text": q, "route": route}
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(questions)} examples to {output_path}")


def create_test_set(
    riksbanken_questions: list[str],
    general_questions: list[str],
    test_size: int = 50,
) -> list[dict]:
    """
    Create a balanced test set for evaluating router accuracy.
    """
    # Take equal samples from each
    n_each = test_size // 2

    riksbanken_test = random.sample(riksbanken_questions, min(n_each, len(riksbanken_questions)))
    general_test = random.sample(general_questions, min(n_each, len(general_questions)))

    test_examples = [
        {"text": q, "route": "riksbanken"} for q in riksbanken_test
    ] + [
        {"text": q, "route": "general"} for q in general_test
    ]

    random.shuffle(test_examples)
    return test_examples


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate router training examples")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--riksbanken-only", action="store_true", help="Only extract riksbanken examples")
    parser.add_argument("--max-riksbanken", type=int, default=200, help="Max riksbanken examples")
    parser.add_argument("--general-batches", type=int, default=2, help="Number of Gemini batches for general questions")
    args = parser.parse_args()

    print("=" * 60)
    print("Router Training Data Generation")
    print("=" * 60)

    # Extract riksbanken questions
    print("\n1. Extracting riksbanken questions from Q&A data...")
    riksbanken_questions = extract_riksbanken_questions(max_examples=args.max_riksbanken)

    if not riksbanken_questions:
        return

    # Generate general questions
    general_questions = []
    if not args.riksbanken_only:
        if not os.environ.get("GEMINI_API_KEY"):
            print("\nWARNING: GEMINI_API_KEY not set, skipping general question generation")
            print("Add it to .env file to generate general questions")
        else:
            print("\n2. Generating general questions via Gemini...")
            general_questions = generate_general_questions(num_batches=args.general_batches)

    if args.dry_run:
        print("\n[DRY RUN] Would save:")
        print(f"  - {len(riksbanken_questions)} riksbanken examples")
        print(f"  - {len(general_questions)} general examples")
        print("\nSample riksbanken questions:")
        for q in riksbanken_questions[:5]:
            print(f"  - {q[:80]}...")
        if general_questions:
            print("\nSample general questions:")
            for q in general_questions[:5]:
                print(f"  - {q[:80]}...")
        return

    # Save examples
    print("\n3. Saving examples...")
    save_examples(
        riksbanken_questions,
        PROJECT_ROOT / RIKSBANKEN_EXAMPLES_PATH,
        "riksbanken",
    )

    if general_questions:
        save_examples(
            general_questions,
            PROJECT_ROOT / GENERAL_EXAMPLES_PATH,
            "general",
        )

        # Create test set
        print("\n4. Creating test set...")
        test_examples = create_test_set(riksbanken_questions, general_questions)
        test_path = PROJECT_ROOT / TEST_EXAMPLES_PATH
        with open(test_path, "w", encoding="utf-8") as f:
            for ex in test_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved {len(test_examples)} test examples to {test_path}")

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Review examples: head data/router/riksbanken_examples.jsonl")
    print("  2. Train router: python -m src.models.router.train")
    print("=" * 60)


if __name__ == "__main__":
    main()
