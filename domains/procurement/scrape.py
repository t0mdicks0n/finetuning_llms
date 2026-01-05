"""
Scraper for Upphandlingsmyndigheten Frågeportalen.

Extracts Q&A pairs from the Swedish public procurement authority's Q&A portal.

Usage:
    python -m src.models.procurement.scrape
    python -m src.models.procurement.scrape --dry-run
    python -m src.models.procurement.scrape --category inkopsprocessen --limit 50
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .config import FRAGEPORTALEN_BASE_URL, FRAGEPORTALEN_CATEGORIES


# Directories
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "procurement"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# API endpoint for fetching Q&A listings
API_BASE_URL = "https://www.upphandlingsmyndigheten.se/api/sv/questionportal/kundo/dialogs"

# Mapping from URL category names to API topic names
CATEGORY_TO_API_TOPIC = {
    "inkopsprocessen": "process",
    "ovriga-fragor": "Q",
    "hallbarhet": "hallbarhet",
    "offentlighet-och-sekretess": "offentlighet_sekretess",
    "statsstod": "statsstod",
    "innovation-och-dialog": "innovation",
}

# Request settings
HEADERS_HTML = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0; educational purposes)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
}
HEADERS_JSON = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0; educational purposes)",
    "Accept": "application/json",
    "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
}
REQUEST_DELAY = 0.1  # Short delay between requests


@dataclass
class QAPair:
    """A single Q&A pair from Frågeportalen."""
    id: str
    url: str
    category: str
    question: str
    answer: str
    published_date: str | None = None
    updated_date: str | None = None
    tags: list[str] | None = None


def fetch_questions_from_api(category: str, limit: int | None = None) -> list[dict]:
    """
    Fetch all questions from a category using the JSON API.

    The API supports fetching all items at once with a large 'fetch' parameter.
    Returns list of dicts with question metadata.
    """
    api_topic = CATEGORY_TO_API_TOPIC.get(category)
    if not api_topic:
        print(f"  Warning: Unknown category '{category}', skipping")
        return []

    # Fetch all items at once (API supports up to 2000+)
    fetch_size = limit if limit else 2000
    url = f"{API_BASE_URL}?dialog.category={api_topic}&fetch={fetch_size}"

    response = requests.get(url, headers=HEADERS_JSON, timeout=60)
    response.raise_for_status()

    data = response.json()
    total = data.get("total", 0)
    items = data.get("items", [])

    print(f"  Category '{category}' (API topic: {api_topic}): {total} questions, fetched {len(items)}")

    questions = []
    for item in items:
        if limit and len(questions) >= limit:
            break

        question_id = item.get("id", "")
        url = item.get("url", "")
        heading = item.get("heading", "")
        text = item.get("text", "")  # This is the question text
        tags = item.get("tags", [])
        state = item.get("conversationState", "")

        # Only include answered questions
        if question_id and url and state == "ANSWERED":
            questions.append({
                "id": str(question_id),
                "url": f"https://www.upphandlingsmyndigheten.se{url}" if url.startswith("/") else url,
                "heading": heading,
                "question_text": text,
                "category": category,
                "tags": tags,
            })

    return questions


def extract_question_ids_from_category(category: str, limit: int | None = None) -> list[dict]:
    """
    Get all question IDs and URLs from a category.

    Uses the JSON API for reliable pagination.
    Returns list of dicts with 'id', 'url', 'heading', 'question_text', 'category', 'tags'.
    """
    return fetch_questions_from_api(category, limit)


def fetch_question_page(url: str) -> str:
    """Fetch the HTML content of a question page."""
    response = requests.get(url, headers=HEADERS_HTML, timeout=30)
    response.raise_for_status()
    return response.text


def parse_question_page(html: str, question_meta: dict) -> list[QAPair]:
    """
    Parse a question page and extract all Q&A pairs from the thread.

    The page structure uses 'dialog-block' divs:
    - Block 0: Original question
    - Block 1: Official answer
    - Block 2: Follow-up question (if any)
    - Block 3: Follow-up answer (if any)
    - ... and so on

    Returns list of QAPair objects (empty list if no valid Q&A found).
    """
    soup = BeautifulSoup(html, "html.parser")
    qa_pairs = []

    dialog_blocks = soup.find_all("div", class_="dialog-block")

    if len(dialog_blocks) < 2:
        # Not enough blocks for a Q&A pair
        return []

    # Get tags from question_meta
    tags = question_meta.get("tags", [])

    # Process pairs of blocks (question, answer)
    # Blocks come in pairs: 0+1, 2+3, 4+5, etc.
    for i in range(0, len(dialog_blocks) - 1, 2):
        question_block = dialog_blocks[i]
        answer_block = dialog_blocks[i + 1]

        question_text = question_block.get_text(separator=" ", strip=True)
        answer_text = answer_block.get_text(separator="\n\n", strip=True)

        # Skip if either is empty or answer is too short
        if not question_text or not answer_text:
            continue
        if len(answer_text) < 50:
            continue

        # For the first pair, use the question_text from listing if available
        # (it's usually cleaner)
        if i == 0 and question_meta.get("question_text"):
            question_text = question_meta["question_text"]

        # Create unique ID for follow-up pairs
        pair_id = question_meta["id"] if i == 0 else f"{question_meta['id']}_followup_{i // 2}"

        qa_pairs.append(QAPair(
            id=pair_id,
            url=question_meta["url"],
            category=question_meta["category"],
            question=question_text,
            answer=answer_text,
            tags=tags,
        ))

    return qa_pairs


def fetch_and_parse_question(q_meta: dict) -> list[QAPair]:
    """Fetch and parse a single question page. Used for parallel execution."""
    try:
        html = fetch_question_page(q_meta["url"])
        return parse_question_page(html, q_meta)
    except Exception:
        return []


def scrape_category(category: str, limit: int | None = None, max_workers: int = 20) -> list[QAPair]:
    """Scrape all Q&A pairs from a category using parallel fetching."""
    print(f"\nScraping category: {category}")

    # Get all question URLs
    questions = extract_question_ids_from_category(category, limit=limit)

    if not questions:
        print(f"  No questions found in {category}")
        return []

    # Fetch and parse questions in parallel
    qa_pairs = []
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_and_parse_question, q): q for q in questions}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Fetching {category}"):
            try:
                pairs = future.result()
                qa_pairs.extend(pairs)
            except Exception:
                errors += 1

    if errors:
        print(f"  ({errors} errors)")
    print(f"  Extracted {len(qa_pairs)} Q&A pairs from {category} ({len(questions)} threads)")
    return qa_pairs


def scrape_all_categories(
    categories: list[str] | None = None,
    limit_per_category: int | None = None,
    save_incrementally: bool = True,
) -> list[QAPair]:
    """Scrape Q&A pairs from all categories."""
    if categories is None:
        categories = FRAGEPORTALEN_CATEGORIES

    all_qa_pairs = []

    for category in categories:
        qa_pairs = scrape_category(category, limit=limit_per_category)
        all_qa_pairs.extend(qa_pairs)

        # Save after each category to avoid losing progress
        if save_incrementally and all_qa_pairs:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            save_raw_data(all_qa_pairs, RAW_DIR / "frageportalen_qa.json")

    return all_qa_pairs


def save_raw_data(qa_pairs: list[QAPair], output_path: Path):
    """Save raw scraped data as JSON."""
    data = [asdict(qa) for qa in qa_pairs]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved raw data to: {output_path}")


def convert_to_training_format(qa_pairs: list[QAPair]) -> list[dict]:
    """
    Convert Q&A pairs to Mistral instruction format.

    Same format as Riksbanken training data.
    """
    examples = []

    for qa in qa_pairs:
        example = {
            "messages": [
                {"role": "user", "content": qa.question},
                {"role": "assistant", "content": qa.answer},
            ],
            "source": "frageportalen",
            "category": qa.category,
            "id": qa.id,
        }
        examples.append(example)

    return examples


def save_training_data(examples: list[dict], output_path: Path):
    """Save training data as JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved training data to: {output_path}")
    print(f"Total examples: {len(examples)}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Frågeportalen Q&A data")
    parser.add_argument("--category", type=str, help="Scrape only this category")
    parser.add_argument("--limit", type=int, help="Limit questions per category")
    parser.add_argument("--dry-run", action="store_true", help="Only list questions, don't fetch content")
    args = parser.parse_args()

    print("=" * 60)
    print("Procurement Expert - Frågeportalen Scraper")
    print("=" * 60)

    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Determine categories to scrape
    categories = [args.category] if args.category else FRAGEPORTALEN_CATEGORIES

    if args.dry_run:
        print("\n[DRY RUN] Listing questions only:\n")
        for category in categories:
            questions = extract_question_ids_from_category(category, limit=args.limit or 5)
            print(f"\n{category}:")
            for q in questions[:5]:
                heading = q.get('heading', 'No heading')[:60]
                print(f"  - {heading}...")
        return

    # Scrape all categories
    qa_pairs = scrape_all_categories(
        categories=categories,
        limit_per_category=args.limit,
    )

    if not qa_pairs:
        print("\nNo Q&A pairs extracted!")
        return

    print(f"\n{'=' * 60}")
    print(f"Total Q&A pairs extracted: {len(qa_pairs)}")
    print(f"{'=' * 60}")

    # Save raw data
    raw_path = RAW_DIR / "frageportalen_qa.json"
    save_raw_data(qa_pairs, raw_path)

    # Convert and save training data
    examples = convert_to_training_format(qa_pairs)

    # Split 90/10
    if len(examples) > 50:
        split_idx = int(len(examples) * 0.9)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        save_training_data(train_examples, PROCESSED_DIR / "train.jsonl")
        save_training_data(val_examples, PROCESSED_DIR / "val.jsonl")

        print(f"\nTrain set: {len(train_examples)} examples")
        print(f"Validation set: {len(val_examples)} examples")
    else:
        save_training_data(examples, PROCESSED_DIR / "train.jsonl")

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Review data: head data/procurement/processed/train.jsonl")
    print("  2. Train: modal run domains/procurement/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
