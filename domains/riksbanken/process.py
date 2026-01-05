"""
Process downloaded Riksbanken PDFs into training data.

Extracts text from PDFs, cleans headers/footers, and outputs JSONL.
"""

import json
import re
from pathlib import Path

import pymupdf  # fitz

# Directories
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# Patterns to filter out (headers, footers, page numbers)
FILTER_PATTERNS = [
    r"^\s*\d+\s*$",  # Standalone page numbers
    r"^\s*Sida\s+\d+",  # "Sida X" (Page X in Swedish)
    r"^\s*Page\s+\d+",  # "Page X"
    r"PENNINGPOLITISK RAPPORT",  # Header
    r"SVERIGES RIKSBANK",  # Header
    r"^\s*\d{4}\s*$",  # Standalone years
    r"^\s*[A-Z]{2,}\s*$",  # All-caps short strings (likely headers)
]

# Compiled regex for efficiency
FILTER_REGEX = [re.compile(p, re.IGNORECASE) for p in FILTER_PATTERNS]

# Minimum line length to keep (filters out noise)
MIN_LINE_LENGTH = 20

# Chunk size for training (in characters)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


def should_filter_line(line: str) -> bool:
    """Check if a line should be filtered out."""
    line = line.strip()

    # Filter empty or very short lines
    if len(line) < MIN_LINE_LENGTH:
        return True

    # Filter lines matching known patterns
    for pattern in FILTER_REGEX:
        if pattern.search(line):
            return True

    return False


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract and clean text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Cleaned text content.
    """
    doc = pymupdf.open(pdf_path)
    text_parts = []

    for page_num, page in enumerate(doc):
        # Extract text from page
        text = page.get_text()

        # Split into lines and filter
        lines = text.split("\n")
        filtered_lines = [
            line.strip()
            for line in lines
            if not should_filter_line(line)
        ]

        # Join remaining lines
        page_text = " ".join(filtered_lines)

        # Clean up multiple spaces
        page_text = re.sub(r"\s+", " ", page_text)

        if page_text.strip():
            text_parts.append(page_text.strip())

    doc.close()

    # Join all pages
    full_text = "\n\n".join(text_parts)

    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks for training.

    Args:
        text: Full text to chunk.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near the boundary
            search_start = max(start + chunk_size - 200, start)
            search_end = min(start + chunk_size + 200, len(text))
            search_text = text[search_start:search_end]

            # Find last sentence boundary in search window
            for punct in [". ", ".\n", "? ", "! "]:
                last_idx = search_text.rfind(punct)
                if last_idx != -1:
                    end = search_start + last_idx + len(punct)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# Instruction prompts for chat-formatted training
INSTRUCTION_PROMPTS = [
    "Du är en Riksbanken-analytiker. Skriv en analys av den penningpolitiska situationen.",
    "Analysera den svenska ekonomin ur ett centralbanks-perspektiv.",
    "Skriv ett avsnitt till en penningpolitisk rapport.",
    "Beskriv inflationsutvecklingen och penningpolitiska överväganden.",
    "Ge en ekonomisk analys i Riksbankens stil.",
]


def create_training_examples(chunks: list[str], source: str, use_chat_format: bool = True) -> list[dict]:
    """
    Create training examples from text chunks.

    Args:
        chunks: List of text chunks.
        source: Source document identifier.
        use_chat_format: If True, wrap in Mistral chat template.

    Returns:
        List of training examples.
    """
    import random

    examples = []

    for i, chunk in enumerate(chunks):
        if use_chat_format:
            # Pick a random instruction prompt
            instruction = random.choice(INSTRUCTION_PROMPTS)
            # Format as Mistral chat template
            text = f"<s>[INST] {instruction} [/INST] {chunk}</s>"
        else:
            text = chunk

        example = {
            "text": text,
            "source": source,
            "chunk_id": i,
        }
        examples.append(example)

    return examples


def process_all_pdfs() -> list[dict]:
    """
    Process all PDFs in the raw directory.

    Returns:
        List of all training examples.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")

    all_examples = []
    total_chars = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(text):,} characters")
        total_chars += len(text)

        # Chunk text
        chunks = chunk_text(text)
        print(f"  Created {len(chunks)} chunks")

        # Create training examples
        examples = create_training_examples(chunks, pdf_path.stem)
        all_examples.extend(examples)

    print()
    print(f"Total: {len(all_examples)} training examples")
    print(f"Total text: {total_chars:,} characters ({total_chars / 1_000_000:.1f} MB)")

    return all_examples


def save_jsonl(examples: list[dict], output_path: Path):
    """Save examples to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")


def main():
    """Main entry point."""
    # Process all PDFs
    examples = process_all_pdfs()

    if not examples:
        print("ERROR: No training examples created!")
        print("Make sure to run the scraper first: python -m src.data.scrape")
        return

    # Save to JSONL
    output_path = PROCESSED_DIR / "train.jsonl"
    save_jsonl(examples, output_path)

    # Also save a small validation set (last document)
    if len(examples) > 50:
        # Use last ~10% as validation
        split_idx = int(len(examples) * 0.9)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        save_jsonl(train_examples, PROCESSED_DIR / "train.jsonl")
        save_jsonl(val_examples, PROCESSED_DIR / "val.jsonl")

        print()
        print(f"Train set: {len(train_examples)} examples")
        print(f"Validation set: {len(val_examples)} examples")


if __name__ == "__main__":
    main()
