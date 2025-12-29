"""
Generate high-quality Q&A pairs from Riksbanken documents using Gemini API.

Creates proper instruction-tuning data in Mistral's format:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    python -m src.data.generate_qa
    python -m src.data.generate_qa --dry-run  # Preview without API calls
    python -m src.data.generate_qa --max-chunks 10  # Limit for testing
"""

import json
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
import google.generativeai as genai
import pymupdf

from src.data.process import should_filter_line

# Load environment variables from .env
load_dotenv()

# Directories
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# Configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Generation config
generation_config = genai.GenerationConfig(
    temperature=0.7,
)

# Use Gemini 2.5 Flash (stable)
def get_model():
    return genai.GenerativeModel(
        "gemini-2.5-flash",
        generation_config=generation_config,
    )


# Chunk size limits for paragraph grouping
MIN_CHUNK_SIZE = 400   # Don't create tiny chunks
MAX_CHUNK_SIZE = 1200  # Don't exceed this


def extract_paragraphs_from_pdf(pdf_path: Path) -> list[str]:
    """
    Extract paragraphs from PDF, preserving natural text structure.

    Returns list of paragraphs (each is a coherent text block).
    """
    doc = pymupdf.open(pdf_path)
    paragraphs = []

    for page in doc:
        text = page.get_text()

        # Split on double newlines (paragraph breaks)
        raw_paragraphs = re.split(r'\n\s*\n', text)

        for para in raw_paragraphs:
            # Clean up the paragraph
            lines = para.split('\n')
            cleaned_lines = [
                line.strip()
                for line in lines
                if line.strip() and not should_filter_line(line)
            ]

            if cleaned_lines:
                # Join lines within paragraph with space
                cleaned_para = ' '.join(cleaned_lines)
                # Clean multiple spaces
                cleaned_para = re.sub(r'\s+', ' ', cleaned_para).strip()

                if len(cleaned_para) >= 50:  # Skip tiny fragments
                    paragraphs.append(cleaned_para)

    doc.close()
    return paragraphs


def chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    """
    Group paragraphs into chunks without splitting mid-paragraph.

    Creates chunks between MIN_CHUNK_SIZE and MAX_CHUNK_SIZE chars.
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        # If single paragraph exceeds max, include it as its own chunk
        if para_size > MAX_CHUNK_SIZE:
            # Flush current chunk first
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            # Add large paragraph as its own chunk
            chunks.append(para)
            continue

        # Check if adding this paragraph would exceed max
        if current_size + para_size + 2 > MAX_CHUNK_SIZE and current_chunk:
            # Flush current chunk if it meets minimum
            if current_size >= MIN_CHUNK_SIZE:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0

        # Add paragraph to current chunk
        current_chunk.append(para)
        current_size += para_size + 2  # +2 for \n\n separator

    # Don't forget the last chunk
    if current_chunk and current_size >= MIN_CHUNK_SIZE:
        chunks.append('\n\n'.join(current_chunk))
    elif current_chunk and chunks:
        # Append small remainder to previous chunk
        chunks[-1] += '\n\n' + '\n\n'.join(current_chunk)
    elif current_chunk:
        # Only chunk, include even if small
        chunks.append('\n\n'.join(current_chunk))

    return chunks


QA_GENERATION_PROMPT = """
Du är en kunnig och pedagogisk expert på svensk penningpolitik, Riksbankens arbete och ekonomi i allmänhet.
Du svarar som en hjälpsam assistent som vill att läsaren verkligen ska förstå ämnet i ett bredare sammanhang.

Baserat på följande text från en Riksbanken-rapport, generera 3-5 högkvalitativa fråga-svar-par på svenska.

VIKTIGT - Generera OLIKA frågetyper (minst en av varje typ):
1. Faktafråga - konkreta fakta med specifika siffror eller nivåer (t.ex. "Vilken nivå...", "Hur mycket...")
2. Förklaringsfråga - förklara koncept eller samband (varför, hur fungerar, vad innebär)
3. Analysfråga - tolkning och konsekvenser för hushåll/företag/samhället (vad betyder detta för...)
4. Jämförelsefråga - jämför med andra länder, tidigare perioder, eller relaterade koncept (hur skiljer sig..., jämfört med...)

KRAV på frågorna:
- Frågor ska vara sådana som en verklig användare skulle ställa till en ekonomiexpert
- Fokusera på generaliserbar kunskap om penningpolitik, ekonomi och finansmarknader
- Koppla gärna Riksbankens arbete till bredare ekonomiska koncept (t.ex. hur påverkar detta bostadsmarknaden, kronkursen, sparande)
- UNDVIK frågor om "texten", "dokumentet", "rapporten", "avsnittet" eller "nästa sida"
- UNDVIK frågor om dokumentstruktur, layout eller hänvisningar
- Frågor ska kunna besvaras utan tillgång till källdokumentet

KRAV på svaren:
- Utförliga och pedagogiska (8-12 meningar, cirka 200-300 ord)
- Ge bakgrund och kontext - förklara HUR saker hänger ihop, inte bara VAD
- Förklara på ett tillgängligt sätt, som om du hjälper någon verkligen förstå ett komplext ämne
- Inkludera konkreta siffror, årtal och exempel när möjligt (t.ex. "när inflationen var 10% under 2022...")
- Koppla till praktiska konsekvenser för vanliga människor (hushåll, låntagare, sparare, företag)
- Sätt in svensk kontext där relevant (t.ex. Sveriges bostadsmarknad, kronans utveckling, svensk exportindustri)
- Avsluta med en sammanfattande mening om vad detta innebär i praktiken
- Svara som om du är en expert som förklarar, UTAN att referera till någon specifik text eller dokument

TEXT:
{chunk}

Returnera ENDAST valid JSON utan markdown-formatering:
[{{"question": "...", "answer": "..."}}]
"""


def clean_and_parse_json(text: str) -> list[dict]:
    """Clean up Gemini's text output and parse as JSON."""
    # Find JSON array in response
    start = text.find('[')
    end = text.rfind(']') + 1

    if start == -1 or end == 0:
        raise ValueError("No JSON array found in response")

    json_str = text[start:end]
    return json.loads(json_str)


def generate_qa_for_chunk(chunk: str, source: str, chunk_id: int) -> list[dict]:
    """
    Generate Q&A pairs for a single chunk using Gemini.

    Returns list of examples in Mistral instruction format.
    """
    model = get_model()
    prompt = QA_GENERATION_PROMPT.format(chunk=chunk)

    try:
        response = model.generate_content(prompt)
        qa_pairs = clean_and_parse_json(response.text)

        # Convert to Mistral instruction format
        examples = []
        for i, qa in enumerate(qa_pairs):
            if "question" in qa and "answer" in qa:
                example = {
                    "messages": [
                        {"role": "user", "content": qa["question"]},
                        {"role": "assistant", "content": qa["answer"]},
                    ],
                    "source": source,
                    "chunk_id": chunk_id,
                    "qa_id": i,
                }
                examples.append(example)

        return examples

    except Exception as e:
        print(f"  Error processing chunk {chunk_id}: {e}")
        return []


def generate_qa_for_chunk_wrapper(args):
    """Wrapper for parallel processing."""
    chunk, source, chunk_id = args
    return generate_qa_for_chunk(chunk, source, chunk_id)


def process_all_pdfs(max_chunks: int | None = None, dry_run: bool = False) -> list[dict]:
    """
    Process all PDFs and generate Q&A pairs.

    Args:
        max_chunks: Limit number of chunks to process (for testing).
        dry_run: If True, just show what would be processed without API calls.

    Returns:
        List of training examples in Mistral instruction format.
    """
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    # Collect all chunks using paragraph-aware chunking
    all_chunks = []
    for pdf_path in pdf_files:
        print(f"\nExtracting: {pdf_path.name}")
        paragraphs = extract_paragraphs_from_pdf(pdf_path)
        chunks = chunk_paragraphs(paragraphs)
        print(f"  {len(paragraphs)} paragraphs -> {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append((chunk, pdf_path.stem, i))

    print(f"\nTotal chunks: {len(all_chunks)}")

    if max_chunks:
        all_chunks = all_chunks[:max_chunks]
        print(f"Limited to: {max_chunks} chunks")

    if dry_run:
        print("\n[DRY RUN] Would generate Q&A for these chunks:")
        for chunk, source, chunk_id in all_chunks[:5]:
            print(f"  - {source} chunk {chunk_id}: {chunk[:100]}...")
        return []

    # Generate Q&A pairs in parallel
    print("\nGenerating Q&A pairs with Gemini 2.0 Flash...")
    all_examples = []

    # Use ThreadPoolExecutor for parallel API calls (10 concurrent, safe for Gemini pay-as-you-go)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(generate_qa_for_chunk_wrapper, args): args
            for args in all_chunks
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            args = futures[future]
            source, chunk_id = args[1], args[2]

            try:
                examples = future.result()
                all_examples.extend(examples)
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{len(all_chunks)} chunks, {len(all_examples)} Q&A pairs")
            except Exception as e:
                print(f"  Error in {source} chunk {chunk_id}: {e}")

    print(f"\nGenerated {len(all_examples)} Q&A pairs total")
    return all_examples


def filter_low_quality(examples: list[dict]) -> list[dict]:
    """
    Filter out low-quality Q&A pairs.

    Removes:
    - Very short answers (<50 chars)
    - Generic questions that don't need domain knowledge
    - Duplicates or near-duplicates
    """
    filtered = []
    seen_questions = set()

    generic_patterns = [
        r"^vad handlar texten om",
        r"^vad står i texten",
        r"^sammanfatta",
        r"^beskriv texten",
    ]

    for example in examples:
        question = example["messages"][0]["content"].lower()
        answer = example["messages"][1]["content"]

        # Skip short answers
        if len(answer) < 50:
            continue

        # Skip generic questions
        is_generic = any(re.match(p, question) for p in generic_patterns)
        if is_generic:
            continue

        # Skip duplicates (simple check)
        q_normalized = re.sub(r'\s+', ' ', question.strip())
        if q_normalized in seen_questions:
            continue
        seen_questions.add(q_normalized)

        filtered.append(example)

    removed = len(examples) - len(filtered)
    if removed > 0:
        print(f"Filtered out {removed} low-quality examples")

    return filtered


def save_jsonl(examples: list[dict], output_path: Path):
    """Save examples to JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Q&A training data using Gemini")
    parser.add_argument("--max-chunks", type=int, help="Limit chunks for testing")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    parser.add_argument("--no-filter", action="store_true", help="Skip quality filtering")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set in environment")
        print("Add it to .env file and run: source .env")
        return

    print("=" * 60)
    print("Swedish Sovereign AI - Q&A Generation Pipeline")
    print("=" * 60)
    print(f"Model: gemini-2.5-flash")
    print(f"Chunk size: {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} chars (paragraph-aware)")

    # Generate Q&A pairs
    examples = process_all_pdfs(max_chunks=args.max_chunks, dry_run=args.dry_run)

    if not examples:
        if not args.dry_run:
            print("No examples generated!")
        return

    # Filter low-quality examples
    if not args.no_filter:
        examples = filter_low_quality(examples)

    # Create train/val split (90/10)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if len(examples) > 50:
        split_idx = int(len(examples) * 0.9)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        save_jsonl(train_examples, PROCESSED_DIR / "train_qa.jsonl")
        save_jsonl(val_examples, PROCESSED_DIR / "val_qa.jsonl")

        print(f"\nTrain set: {len(train_examples)} examples")
        print(f"Validation set: {len(val_examples)} examples")
    else:
        save_jsonl(examples, PROCESSED_DIR / "train_qa.jsonl")
        print(f"\nTotal: {len(examples)} examples (no val split - too few)")

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Review generated Q&A: head data/processed/train_qa.jsonl")
    print("  2. Update train.py to use train_qa.jsonl")
    print("  3. Retrain: modal run src/train/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
