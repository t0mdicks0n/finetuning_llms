"""Quick test of LLM rewriting for follow-up questions."""
import json
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

generation_config = genai.GenerationConfig(temperature=0.7)
model = genai.GenerativeModel("gemini-2.5-pro", generation_config=generation_config)

# Load raw data
with open("data/procurement/raw/frageportalen_qa.json", "r") as f:
    raw_data = json.load(f)

# Deduplicate and build index
seen = set()
unique = []
for qa in raw_data:
    if qa["id"] not in seen:
        seen.add(qa["id"])
        unique.append(qa)
qa_index = {qa["id"]: qa for qa in unique}

# Find follow-ups needing context
def needs_context(q):
    q = q.lower()
    patterns = [
        r'du (skriver|skrev|nämner|nämnde)',
        r'i (ditt|ert) (svar|inlägg)',
        r'^(ja,?\s+)?(men|och|så)\s',
        r'menar du (att|med)',
        r'ovan(stående)?',
    ]
    for p in patterns:
        if re.search(p, q):
            return True
    return False

followups = []
for qa in unique:
    if "_followup_" in qa["id"] and needs_context(qa.get("question", "")):
        followups.append(qa)
        if len(followups) >= 5:
            break

# Rewrite function
def rewrite(followup_q, original_q, original_a):
    # Truncate answer
    if len(original_a) > 1500:
        original_a = original_a[:1500]
        last_period = original_a.rfind('.')
        if last_period > 750:
            original_a = original_a[:last_period + 1]

    prompt = f"""Du är en expert på att omformulera uppföljningsfrågor till fristående frågor.

Givet en ursprunglig fråga, dess svar, och en uppföljningsfråga - skriv om uppföljningsfrågan så att den blir en komplett, fristående fråga som kan förstås utan den tidigare konversationen.

VIKTIGT:
- Behåll det ursprungliga syftet och innebörden
- Inkludera nödvändig kontext från den ursprungliga frågan/svaret
- Skriv på svenska
- Håll frågan koncis men komplett
- Returnera ENDAST den omformulerade frågan, inget annat

URSPRUNGLIG FRÅGA:
{original_q}

SVAR PÅ URSPRUNGLIG FRÅGA:
{original_a}

UPPFÖLJNINGSFRÅGA ATT OMFORMULERA:
{followup_q}

OMFORMULERAD FRISTÅENDE FRÅGA:"""

    response = model.generate_content(prompt)
    return response.text.strip()

# Test on each
print("TESTING LLM REWRITES ON 5 EXAMPLES (gemini-2.5-pro)\n")
print("="*70)

for i, fu in enumerate(followups):
    base_id = fu["id"].split("_followup_")[0]
    orig = qa_index.get(base_id)

    print(f"\n{'='*70}")
    print(f"Example {i+1}: {fu['id']}")
    print(f"{'='*70}")

    print(f"\nORIGINAL FOLLOW-UP:")
    print(fu["question"][:300] + ("..." if len(fu["question"]) > 300 else ""))

    rewritten = rewrite(fu["question"], orig["question"], orig["answer"])

    print(f"\n→ LLM REWRITTEN:")
    print(rewritten)
    print()
