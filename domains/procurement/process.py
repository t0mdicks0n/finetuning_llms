"""
Post-process scraped Frågeportalen data for training.

Cleans the Q&A pairs to produce assistant-style responses:
- Removes greetings and sign-offs
- Cleans HTML from questions
- Removes website-specific references
- Standardizes formatting
- Enriches follow-up questions with context (optionally using LLM)

Usage:
    python -m src.models.procurement.process
    python -m src.models.procurement.process --input data/procurement/raw/frageportalen_qa.json
    python -m src.models.procurement.process --use-llm  # Use LLM for high-quality enrichment
"""

import json
import os
import re
from pathlib import Path


# Directories
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "procurement"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def clean_html(text: str) -> str:
    """Remove HTML tags and convert to plain text."""
    # Replace <br> tags with newlines
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

    # Remove other HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')

    return text


def remove_greeting(text: str) -> str:
    """Remove greeting lines from the start of an answer."""
    lines = text.split('\n')

    # Patterns for greetings (with optional "igen" and name)
    # Name pattern includes hyphens and accented characters
    name_pattern = r'[A-ZÅÄÖÉÈËa-zåäöéèë]+(?:-[A-ZÅÄÖÉÈËa-zåäöéèë]+)?'
    greeting_patterns = [
        rf'^(Hej|Hello|Hi|Hejsan)(\s+igen)?\s+{name_pattern}[,!.]?\s*$',  # "Hej Anna," or "Hej Nils-Erik,"
        r'^(Hej|Hello|Hi|Hejsan)(\s+igen)?[,!.]?\s*$',  # Just "Hej" or "Hej igen,"
        r'^Tack för (din |er )?(fråga|frågan)[,!.]?\s*$',
        r'^Tack för att du (kontaktar|hör av dig)[^.]*[.]?\s*$',
    ]

    # Remove leading empty lines and greeting lines
    while lines:
        line = lines[0].strip()
        if not line:
            lines.pop(0)
            continue

        is_greeting = False
        for pattern in greeting_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_greeting = True
                break

        if is_greeting:
            lines.pop(0)
            # Also remove empty line after greeting
            if lines and not lines[0].strip():
                lines.pop(0)
        else:
            break

    return '\n'.join(lines)


def remove_signoff(text: str) -> str:
    """Remove sign-off lines from the end of an answer."""
    lines = text.split('\n')

    # Patterns for sign-offs
    signoff_patterns = [
        r'^(Med )?(vänlig |vänliga )?(hälsning|hälsningar)[,!]?\s*$',
        r'^(Best|Kind) regards[,!]?\s*$',
        r'^(Vänligen|Mvh|Vh)[,]?\s*$',
        r'^/\s*[A-ZÅÄÖa-zåäö\s]+$',  # /Name format
        r'^[A-ZÅÄÖ][a-zåäö]+\s*$',  # Just a name at the end
    ]

    # Remove trailing empty lines and sign-off lines
    while lines:
        line = lines[-1].strip()
        if not line:
            lines.pop()
            continue

        is_signoff = False
        for pattern in signoff_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_signoff = True
                break

        if is_signoff:
            lines.pop()
        else:
            break

    return '\n'.join(lines)


def clean_website_references(text: str) -> str:
    """Clean or remove website-specific references."""
    # Remove entire "Läs mer" sections - these are website-specific links
    # Common patterns:
    # - "Läs mer\n\nSe även inlägget..."
    # - "Läs mer\n\nLäs även inläggen..."
    # - "Läs mer\n\nPå vår webbplats..."

    # Remove "Läs mer" followed by content referencing inlägg/webbplats/vägledning
    text = re.sub(
        r'\n\nLäs mer\n\n(Se även |Läs även |Läs mer i |Läs gärna |Läs om |I |Mer information om |På vår webbplats |För ytterligare |För en mer |Om |Du hittar |Ta del av )[^\n]+(\n\n[^\n]+)*$',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Remove "Läs mer" at the very end
    text = re.sub(r'\n\nLäs mer\s*$', '', text, flags=re.IGNORECASE)

    # Remove "Läs mer" followed by any website reference content
    text = re.sub(
        r'\n\nLäs mer\n\n[^\n]*(inlägg|webbplats|vägledning)[^\n]*(\n\n[^\n]+)*$',
        '',
        text,
        flags=re.IGNORECASE
    )

    # Remove "Läs mer" sections that reference their website
    text = re.sub(
        r'\n*Läs mer\s*\n+Läs mer om[^\n]+på vår webbplats\.?\s*',
        '\n',
        text,
        flags=re.IGNORECASE
    )

    # Remove standalone "Läs mer" headers followed by links
    text = re.sub(
        r'\n*Läs mer\s*\n+[^\n]*upphandlingsmyndigheten[^\n]*\n*',
        '\n',
        text,
        flags=re.IGNORECASE
    )

    # Remove "Läs mer" sections with "Du kan läsa mer om" or "Läs gärna mer om"
    text = re.sub(
        r'\n*Läs mer\s*\n+(Du kan |Läs gärna )?läsa? mer om[^\n]*\.?\s*',
        '\n',
        text,
        flags=re.IGNORECASE
    )

    # Remove "på vår webbplats" references
    text = re.sub(r'\s*på vår webbplats\.?', '.', text)

    # Remove Frågeportal references
    text = re.sub(r'\s*i vår Frågeportal\.?', '.', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*i Frågeportalen\.?', '.', text, flags=re.IGNORECASE)

    # Remove references to specific posts/articles on their site
    text = re.sub(
        r'Läs mer om[^\n]*i inlägget\s*\n*[^\n]*\.\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    return text


def clean_source_references(text: str) -> str:
    """Clean up source/reference sections to be more neutral."""
    # Keep legal references but remove "Källhänvisning" header if standalone
    text = re.sub(r'\n*Källhänvisning\s*\n+', '\n\nKällor:\n', text)

    return text


def fix_broken_inline_links(text: str) -> str:
    """
    Fix broken inline link formatting from HTML extraction.

    Links get extracted with their text on separate lines like:
    - "The Directive (\n\n2014/24/EU\n\n) states..."
    - "lagen om offentlig upphandling (\n\nLOU\n\n)"
    - "9 kap. 1 §\n\nLOU\n\n)."
    - "Bilaga 3 till\n\nLOU\n\n– definition"
    - "in a\n\njoint statement\n\nbetween..."

    This rejoins them into proper inline text.
    """
    # Fix parenthesized references split across lines: (\n\nTEXT\n\n) -> (TEXT)
    # Handles both (\n\nLOU\n\n) and (\nLOU\n) patterns
    text = re.sub(r'\(\s*\n+([^)\n]+)\n+\)', r'(\1)', text)

    # Also fix (\nTEXT) without closing newline
    text = re.sub(r'\(\s*\n+([^)\n]+)\)', r'(\1)', text)

    # Fix short uppercase abbreviations between newlines: \n\nLOU\n\n -> LOU
    # These are often law abbreviations like LOU, LUF, LUK, ESPD, etc.
    text = re.sub(r'\n\n([A-ZÅÄÖ]{2,6})\n\n', r' \1 ', text)

    # Fix hyphenated terms like "LOU-direktivet", "ESPD-systemet"
    text = re.sub(r'\n\n([A-ZÅÄÖ]{2,6}-[a-zåäö]+)\n\n', r' \1 ', text)

    # Fix common Swedish words that appear isolated (often link anchors)
    # These should be joined inline, not on separate lines
    common_words = ['tröskelvärdet', 'bilaga 2', 'webbplats', 'ramavtal']
    for word in common_words:
        text = re.sub(rf'\n\n({word})\n\n', r' \1 ', text, flags=re.IGNORECASE)

    # Fix orphaned short words between newlines (often link text)
    # "word\n\noch\n\nmore" -> "word och more"
    orphan_words = ['och', 'eller', 'samt', 'inte', 'får', 'ska', 'kan', 'har', 'är', 'att']
    for word in orphan_words:
        text = re.sub(rf'\n\n({word})\n\n', r' \1 ', text, flags=re.IGNORECASE)

    # Fix orphaned punctuation
    text = re.sub(r'\n\n([.,;:])\n\n', r'\1 ', text)

    # Fix "a/an" article followed by link text on next line
    # "in a\n\njoint statement" -> "in a joint statement"
    text = re.sub(r'\b(a|an|the|ett|en|den|det)\s*\n\n+([A-Za-zÅÄÖåäö][^\n]{0,50})\n\n', r'\1 \2 ', text, flags=re.IGNORECASE)

    # Fix standalone link text between double newlines (common pattern)
    # "word\n\nLinkText\n\nmore" -> "word LinkText more"
    # Only match short capitalized phrases that look like link anchors
    text = re.sub(r'(\w)\s*\n\n([A-ZÅÄÖ][a-zåäö]{2,25})\n\n(\w)', r'\1 \2 \3', text)

    return text


def fix_broken_section_headers(text: str) -> str:
    """
    Fix section headers that got separated from their content.

    Pattern like:
    "Legal ground\n\nfor alternative means"
    becomes:
    "Legal ground for alternative means"
    """
    # Common header words followed by newlines then lowercase continuation
    header_words = [
        'Legal ground', 'Mandatory', 'Voluntary', 'Background',
        'Rättslig grund', 'Bakgrund', 'Sammanfattning', 'Slutsats',
    ]

    for header in header_words:
        # Match header followed by newlines then lowercase word
        pattern = rf'({header})\s*\n\n+(för |for |om |av |till |on |of |to )'
        text = re.sub(pattern, r'\1 \2', text, flags=re.IGNORECASE)

    return text


def clean_orphaned_words(text: str) -> str:
    """
    Clean orphaned words that appear on their own line.

    These often come from link text extraction where a connector word
    like "och" or "eller" was part of a link and ends up on its own line.
    """
    lines = text.split('\n')
    cleaned_lines = []

    # Common Swedish connector words that shouldn't be alone on a line
    orphan_words = {'och', 'eller', 'samt', 'men', 'för', 'till', 'av', 'med', 'om', 'på'}

    for i, line in enumerate(lines):
        stripped = line.strip().lower()

        # If line is just a short connector word, merge with next line
        if stripped in orphan_words and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line:
                # Merge this word with the next line
                lines[i + 1] = f"{line.strip()} {next_line}"
                continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    # Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize multiple newlines to max 2 (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove spaces at start/end of lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def remove_question_pleasantries(text: str) -> str:
    """Remove polite phrases from questions that don't add value for LLM training."""
    # Name pattern includes hyphens and accented characters
    name_pattern = r'[A-ZÅÄÖÉÈËa-zåäöéèë]+(?:-[A-ZÅÄÖÉÈËa-zåäöéèë]+)?'

    # First remove inline greetings at the start (e.g., "Hej! När..." -> "När...")
    # This needs to run first so "Hej Gustav, Är..." becomes "Gustav, Är..."
    inline_greeting_patterns = [
        rf'^{name_pattern}[!]\s+Tack för svar[.!]?\s*',  # "Linnea! Tack för svar. "
        r'^(Hej|Hello|Hi|Hejsan)(\s+igen)?[,!.\s]+',  # "Hej! ", "Hej, ", "Hej "
    ]
    for pattern in inline_greeting_patterns:
        text = re.sub(pattern, '', text, count=1, flags=re.IGNORECASE)

    # Remove leading name followed by comma (e.g., "Gustav, Är det..." -> "Är det...")
    text = re.sub(rf'^{name_pattern},\s*', '', text, count=1)

    # Remove leading name(s) followed by capital letter (e.g., "Terese Är..." -> "Är...")
    # Run in a loop to handle multiple names
    while True:
        new_text = re.sub(rf'^{name_pattern}\s+(?=[A-ZÅÄÖÉÈË])', '', text, count=1)
        if new_text == text:
            break
        text = new_text

    lines = text.split('\n')

    # Patterns for question greetings/pleasantries on their own line
    start_patterns = [
        r'^(Hej|Hello|Hi|Hejsan)\s*[,.!]?\s*$',  # "Hej" "Hej." "Hej !" "Hej,"
        r'^(Hej|Hello|Hi|Hejsan)(\s+igen)?\s*[,.!]?\s+[A-ZÅÄÖa-zåäö]+[,.!]?\s*$',  # Hej + name
        r'^Tack\s+(för|på förhand)[^.]*[.!]?\s*$',
    ]

    # Remove leading pleasantries
    while lines:
        line = lines[0].strip()
        if not line:
            lines.pop(0)
            continue

        is_pleasantry = False
        for pattern in start_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_pleasantry = True
                break

        if is_pleasantry:
            lines.pop(0)
            if lines and not lines[0].strip():
                lines.pop(0)
        else:
            break

    # Patterns for closing pleasantries
    end_patterns = [
        r'^Tack\s*(på förhand|i förväg)?[,!.]?\s*[A-ZÅÄÖa-zåäö]*\s*$',  # "Tack på förhand Åke"
        r'^(Med )?(vänlig |vänliga )?(hälsning|hälsningar?)[,!]?\s*[A-ZÅÄÖa-zåäö]*\s*$',  # "Med vänlig hälsning Johanna"
        r'^Mvh[,]?\s*$',
        r'^/\s*[A-ZÅÄÖa-zåäö\s]+$',  # /Name format
    ]

    # Remove trailing pleasantries
    while lines:
        line = lines[-1].strip()
        if not line:
            lines.pop()
            continue

        is_pleasantry = False
        for pattern in end_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_pleasantry = True
                break

        if is_pleasantry:
            lines.pop()
        else:
            break

    text = '\n'.join(lines)

    # Also remove inline sign-offs at the end (e.g., "...enmansbolag Med vänlig hälsning Johanna")
    text = re.sub(r'\s*(Med )?(vänlig |vänliga )?(hälsning|hälsningar?)\s*[A-ZÅÄÖa-zåäö]*\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*Mvh\s*[A-ZÅÄÖa-zåäö]*\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*Tack\s*(på förhand|i förväg)?\s*[A-ZÅÄÖa-zåäö]*\s*$', '', text, flags=re.IGNORECASE)

    return text


def clean_question(question: str) -> str:
    """Clean a question text."""
    text = question

    # Remove HTML
    text = clean_html(text)

    # Remove greeting from question (some users start with "Hej,")
    text = remove_greeting(text)

    # Remove polite pleasantries that don't add value
    text = remove_question_pleasantries(text)

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text


def clean_answer(answer: str) -> str:
    """Clean an answer text for assistant-style output."""
    text = answer

    # Remove HTML (shouldn't be present but just in case)
    text = clean_html(text)

    # Remove greeting
    text = remove_greeting(text)

    # Remove sign-off
    text = remove_signoff(text)

    # Clean website references
    text = clean_website_references(text)

    # Clean source references
    text = clean_source_references(text)

    # Fix broken link formatting (must come before orphan word cleaning)
    text = fix_broken_inline_links(text)
    text = fix_broken_section_headers(text)

    # Clean orphaned words from link extraction
    text = clean_orphaned_words(text)

    # Normalize whitespace
    text = normalize_whitespace(text)

    return text


def is_followup(qa_id: str) -> bool:
    """Check if this is a follow-up question based on ID pattern."""
    return "_followup_" in qa_id


def needs_context_enrichment(question: str) -> bool:
    """
    Check if a follow-up question needs context from the original Q&A.

    Returns True for questions that explicitly reference the previous conversation
    and would be confusing without that context.
    """
    question_lower = question.lower()

    # Severe: Explicit quotes or references to previous answer
    severe_patterns = [
        r'du (skriver|skrev|nämner|nämnde|anger|angav)',  # "you write/wrote/mention"
        r'i (ditt|ert) (svar|inlägg)',  # "in your answer/post"
        r'enligt (ditt|ert) svar',  # "according to your answer"
        r'som du (skrev|nämnde|angav)',  # "as you wrote/mentioned"
        r'citatet',  # "the quote"
        r'"[^"]{10,}"',  # Quoted text from previous answer
    ]

    for pattern in severe_patterns:
        if re.search(pattern, question_lower):
            return True

    # Moderate: References to previous discussion
    moderate_patterns = [
        r'^(ja,?\s+)?(men|och|så)\s',  # Starts with "yes, but/and/so"
        r'^(jag )?(förstår|förstod)',  # "I understand/understood"
        r'ang(ående)?\s+(det|detta)',  # "regarding this/that"
        r'menar du (att|med)',  # "do you mean that/by"
        r'vad (menas|menar du) med',  # "what is meant by"
        r'syftar du på',  # "are you referring to"
        r'ovan(stående)?',  # "above/aforementioned"
        r'(det|detta) (du|ni) (beskriver|beskrev)',  # "what you describe"
    ]

    for pattern in moderate_patterns:
        if re.search(pattern, question_lower):
            return True

    return False


def create_enriched_question(followup_q: str, original_q: str, original_a: str) -> str:
    """
    Create an enriched, self-contained question from a follow-up.

    Prepends context from the original Q&A to make the follow-up understandable
    without the conversational history.
    """
    # Truncate original answer if very long (keep first ~500 chars for context)
    max_answer_len = 800
    if len(original_a) > max_answer_len:
        # Try to cut at a sentence boundary
        truncated = original_a[:max_answer_len]
        last_period = truncated.rfind('.')
        if last_period > max_answer_len // 2:
            truncated = truncated[:last_period + 1]
        original_a = truncated + " [...]"

    enriched = f"""Bakgrund: En fråga ställdes om: "{original_q}"

Svaret inkluderade: {original_a}

Uppföljningsfråga: {followup_q}"""

    return enriched


# LLM-based enrichment for high-quality question rewriting
_llm_model = None


def get_llm_model():
    """Get or initialize the LLM model for question rewriting."""
    global _llm_model
    if _llm_model is None:
        try:
            from dotenv import load_dotenv
            import google.generativeai as genai

            load_dotenv()

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            genai.configure(api_key=api_key)

            generation_config = genai.GenerationConfig(
                temperature=0.7,
            )

            _llm_model = genai.GenerativeModel(
                "gemini-2.5-pro",
                generation_config=generation_config,
            )
        except ImportError:
            raise ImportError("google-generativeai package not installed")
    return _llm_model


def rewrite_followup_with_llm(
    followup_q: str, original_q: str, original_a: str
) -> str | None:
    """
    Use LLM to rewrite a follow-up question into a standalone question.

    Returns the rewritten question, or None if rewriting fails.
    """
    model = get_llm_model()

    # Truncate answer if very long
    max_answer_len = 1500
    if len(original_a) > max_answer_len:
        truncated = original_a[:max_answer_len]
        last_period = truncated.rfind('.')
        if last_period > max_answer_len // 2:
            truncated = truncated[:last_period + 1]
        original_a = truncated

    prompt = f"""Du är en expert på att omformulera uppföljningsfrågor till fristående frågor för träning av AI-assistenter.

Givet en ursprunglig fråga, dess svar, och en uppföljningsfråga - skriv om uppföljningsfrågan så att den blir en komplett, fristående fråga som kan förstås utan den tidigare konversationen.

VIKTIGT:
- Behåll det ursprungliga syftet och innebörden
- Inkludera nödvändig kontext från den ursprungliga frågan/svaret
- Skriv på svenska
- Håll frågan koncis men komplett
- Ta bort ALL artighet: inga hälsningar (Hej, Tack för svar, etc.), inga namn, inga avslutningar (Mvh, Vänliga hälsningar, etc.)
- Skriv som en neutral, professionell fråga - inte som ett e-postmeddelande
- Returnera ENDAST den omformulerade frågan, inget annat

URSPRUNGLIG FRÅGA:
{original_q}

SVAR PÅ URSPRUNGLIG FRÅGA:
{original_a}

UPPFÖLJNINGSFRÅGA ATT OMFORMULERA:
{followup_q}

OMFORMULERAD FRISTÅENDE FRÅGA:"""

    try:
        response = model.generate_content(prompt)
        rewritten = response.text.strip()

        # Basic validation
        if len(rewritten) < 20 or len(rewritten) > 2000:
            return None
        if not rewritten.endswith('?'):
            rewritten += '?'

        return rewritten
    except Exception as e:
        print(f"    LLM rewrite failed: {e}")
        return None


def _rewrite_single_followup(args: tuple) -> tuple[str, str | None]:
    """Worker function for parallel rewriting."""
    qa_id, followup_q, orig_q, orig_a = args
    rewritten = rewrite_followup_with_llm(followup_q, orig_q, orig_a)
    return (qa_id, rewritten)


def rewrite_followups_batch(
    followups: list[dict], qa_index: dict[str, dict], max_workers: int = 10
) -> dict[str, str]:
    """
    Rewrite multiple follow-up questions using LLM in parallel.

    Returns a dict mapping followup_id -> rewritten_question.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    # Prepare all tasks
    tasks = []
    for qa in followups:
        qa_id = qa.get("id", "")
        followup_q = clean_question(qa.get("question", ""))

        original = get_original_qa(qa_id, qa_index)
        if not original:
            continue

        orig_q = clean_question(original.get("question", ""))
        orig_a = clean_answer(original.get("answer", ""))
        tasks.append((qa_id, followup_q, orig_q, orig_a))

    print(f"\n  Rewriting {len(tasks)} follow-ups with LLM ({max_workers} parallel workers)...")

    results = {}
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_rewrite_single_followup, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="  LLM rewriting"):
            try:
                qa_id, rewritten = future.result()
                if rewritten:
                    results[qa_id] = rewritten
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"    Worker error: {e}")

    print(f"  LLM rewriting complete: {len(results)} success, {failed} failed")
    return results


def is_thanks_only(text: str) -> bool:
    """Check if text is just a thanks/acknowledgment without a real question."""
    text_lower = text.lower().strip()

    # Common "thanks only" patterns
    thanks_patterns = [
        r'^(hej[!,]?\s*)?(tack|utmärkt|bra|perfekt|jättebra|toppen)[!.\s]*$',
        r'^(hej[!,]?\s*)?(tack|utmärkt|bra|perfekt)[!,\s]+(för\s+)?(svaret|svar|hjälpen|info)[!.\s]*$',
        r'^tack\s+så\s+mycket[!.\s]*$',
        r'^(hej[!,]?\s*)?utmärkt\s+svar[!.\s]*(tack[!.\s]*)?$',
    ]

    for pattern in thanks_patterns:
        if re.match(pattern, text_lower):
            return True
    return False


def process_qa_pair(qa: dict, skip_question_clean: bool = False) -> dict | None:
    """
    Process a single Q&A pair.

    Returns None if the pair should be filtered out.

    Args:
        qa: The Q&A dict with question, answer, etc.
        skip_question_clean: If True, skip question cleaning (used for pre-enriched questions)
    """
    if skip_question_clean:
        question = qa.get("question", "")
    else:
        question = clean_question(qa.get("question", ""))
    answer = clean_answer(qa.get("answer", ""))

    # Filter out if question or answer is too short after cleaning
    if len(question) < 20:
        return None
    if len(answer) < 100:
        return None

    # Filter out "thanks only" follow-ups that aren't real questions
    if is_thanks_only(question):
        return None

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "source": "frageportalen",
        "category": qa.get("category", ""),
        "id": qa.get("id", ""),
        "tags": qa.get("tags", []),
    }


def build_qa_index(raw_data: list[dict]) -> dict[str, dict]:
    """
    Build an index mapping IDs to Q&A pairs for quick lookup.

    Used to find original questions when processing follow-ups.
    """
    index = {}
    for qa in raw_data:
        qa_id = qa.get("id", "")
        if qa_id and qa_id not in index:
            index[qa_id] = qa
    return index


def get_original_qa(followup_id: str, qa_index: dict[str, dict]) -> dict | None:
    """
    Get the original Q&A for a follow-up question.

    Follow-up IDs have format: {base_id}_followup_{n}
    Returns the original Q&A dict or None if not found.
    """
    if "_followup_" not in followup_id:
        return None

    base_id = followup_id.split("_followup_")[0]
    return qa_index.get(base_id)


def process_raw_data(input_path: Path, use_llm: bool = False) -> list[dict]:
    """
    Process raw scraped data into clean training examples.

    Args:
        input_path: Path to raw JSON data
        use_llm: If True, use LLM to rewrite context-dependent follow-ups
                 into high-quality standalone questions
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Processing {len(raw_data)} raw Q&A pairs...")

    # Deduplicate by ID (API returns duplicates for some categories)
    seen_ids = set()
    unique_data = []
    duplicate_count = 0
    for qa in raw_data:
        qa_id = qa.get("id", "")
        if qa_id not in seen_ids:
            seen_ids.add(qa_id)
            unique_data.append(qa)
        else:
            duplicate_count += 1

    if duplicate_count > 0:
        print(f"Removed {duplicate_count} duplicates, {len(unique_data)} unique entries")

    # Build index for linking follow-ups to originals
    qa_index = build_qa_index(unique_data)

    # Identify follow-ups that need enrichment
    followups_needing_enrichment = []
    for qa in unique_data:
        qa_id = qa.get("id", "")
        if is_followup(qa_id):
            cleaned_q = clean_question(qa.get("question", ""))
            if needs_context_enrichment(cleaned_q):
                followups_needing_enrichment.append(qa)

    print(f"Found {len(followups_needing_enrichment)} follow-ups needing context enrichment")

    # If using LLM, rewrite all context-dependent follow-ups upfront
    llm_rewrites = {}
    if use_llm and followups_needing_enrichment:
        llm_rewrites = rewrite_followups_batch(followups_needing_enrichment, qa_index)

    processed = []
    filtered_count = 0
    enriched_count = 0
    llm_enriched_count = 0
    followup_count = 0

    for qa in unique_data:
        qa_id = qa.get("id", "")
        question = qa.get("question", "")

        # Handle follow-up questions
        if is_followup(qa_id):
            followup_count += 1

            # Clean the question first to check if it needs enrichment
            cleaned_question = clean_question(question)

            # Check if this follow-up needs context enrichment
            if needs_context_enrichment(cleaned_question):
                # Try LLM rewrite first if available
                if qa_id in llm_rewrites:
                    qa_rewritten = qa.copy()
                    qa_rewritten["question"] = llm_rewrites[qa_id]

                    result = process_qa_pair(qa_rewritten, skip_question_clean=True)
                    if result:
                        result["enriched"] = "llm"
                        processed.append(result)
                        enriched_count += 1
                        llm_enriched_count += 1
                    else:
                        filtered_count += 1
                    continue

                # Fall back to template-based enrichment
                original = get_original_qa(qa_id, qa_index)
                if original:
                    orig_q_clean = clean_question(original.get("question", ""))
                    orig_a_clean = clean_answer(original.get("answer", ""))

                    enriched_q = create_enriched_question(
                        cleaned_question, orig_q_clean, orig_a_clean
                    )

                    qa_enriched = qa.copy()
                    qa_enriched["question"] = enriched_q

                    result = process_qa_pair(qa_enriched, skip_question_clean=True)
                    if result:
                        result["enriched"] = "template"
                        processed.append(result)
                        enriched_count += 1
                    else:
                        filtered_count += 1
                    continue

        # Standard processing for non-enriched questions
        result = process_qa_pair(qa)
        if result:
            processed.append(result)
        else:
            filtered_count += 1

    print(f"Processed: {len(processed)} examples")
    print(f"  - Follow-ups: {followup_count}")
    print(f"  - Enriched: {enriched_count} ({llm_enriched_count} via LLM, {enriched_count - llm_enriched_count} via template)")
    print(f"Filtered out: {filtered_count} examples")

    return processed


def save_training_data(examples: list[dict], train_path: Path, val_path: Path):
    """Save processed data with train/val split."""
    # Shuffle for better split (optional - could keep category grouping)
    import random
    random.seed(42)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    # 90/10 split
    split_idx = int(len(shuffled) * 0.9)
    train_examples = shuffled[:split_idx]
    val_examples = shuffled[split_idx:]

    # Save train
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Save val
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved training data:")
    print(f"  Train: {len(train_examples)} examples -> {train_path}")
    print(f"  Val: {len(val_examples)} examples -> {val_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process scraped Q&A data for training")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "frageportalen_qa.json",
        help="Path to raw scraped data",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="Show N example transformations",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM (Gemini) to rewrite context-dependent follow-up questions into high-quality standalone questions",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run the scraper first: python -m src.models.procurement.scrape")
        return

    print("=" * 60)
    print("Procurement Expert - Data Processing")
    if args.use_llm:
        print("  (Using LLM for follow-up enrichment)")
    print("=" * 60)

    # Load and show examples if requested
    if args.show_examples > 0:
        with open(args.input, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        print(f"\nShowing {args.show_examples} example transformations:\n")
        for i, qa in enumerate(raw_data[:args.show_examples]):
            print(f"{'='*60}")
            print(f"Example {i+1}")
            print(f"{'='*60}")

            print("\n--- ORIGINAL QUESTION ---")
            print(qa.get("question", "")[:300])

            print("\n--- CLEANED QUESTION ---")
            print(clean_question(qa.get("question", ""))[:300])

            print("\n--- ORIGINAL ANSWER (first 400 chars) ---")
            print(qa.get("answer", "")[:400])

            print("\n--- CLEANED ANSWER (first 400 chars) ---")
            print(clean_answer(qa.get("answer", ""))[:400])
            print()
        return

    # Process data
    processed = process_raw_data(args.input, use_llm=args.use_llm)

    if not processed:
        print("No examples after processing!")
        return

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_training_data(
        processed,
        PROCESSED_DIR / "train.jsonl",
        PROCESSED_DIR / "val.jsonl",
    )

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Review: head data/procurement/processed/train.jsonl")
    print("  2. Train: modal run domains/procurement/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
