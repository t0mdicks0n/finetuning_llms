# Prompt Routing Plan: Domain-Specific Model Selection

## Problem Statement

Our finetuned Riksbanken model (V2) excels at Swedish monetary policy questions but suffers from **catastrophic forgetting** on general knowledge. For example, it confidently claims Kyoto is Japan's capital.

Rather than trying to fix this through training data (adding general knowledge examples, self-routing prompts), we can solve it architecturally: **route each prompt to the appropriate model before generation**.

## Goal

Build a routing layer that classifies incoming prompts and directs them to:

| Route | Model | Use Case |
|-------|-------|----------|
| `riksbanken` | Finetuned Mistral + LoRA | Swedish monetary policy, Riksbanken, inflation, interest rates, KPIF, etc. |
| `general` | Vanilla Mistral-7B-Instruct | Everything else |

This gives users the best of both worlds: domain expertise when relevant, full general capabilities otherwise.

---

## Routing Approaches

### Option A: Semantic Router (Embedding-based) - Recommended

**How it works:**
1. Define routes with example utterances (20-50 per route)
2. Pre-encode all examples into embedding vectors
3. At runtime: embed user query → find nearest route via cosine similarity → route to model

**Architecture:**
```
User prompt
    ↓
Sentence Transformer (encode)  [~5-10ms]
    ↓
Cosine similarity vs route centroids  [~1ms]
    ↓
Route decision (riksbanken | general)
    ↓
Model inference (with/without LoRA adapters)
```

**Pros:**
- Extremely fast (~10ms overhead)
- No LLM call for routing
- Simple to implement and debug
- Works well for binary classification with clear domains

**Cons:**
- Requires curated example utterances
- Less flexible for edge cases
- No built-in confidence scores (but can compute similarity threshold)

**Library:** [aurelio-labs/semantic-router](https://github.com/aurelio-labs/semantic-router)

**Encoder options:**
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - Good for Swedish, 118M params
- `intfloat/multilingual-e5-small` - Smaller, fast
- Local HuggingFace encoder for fully offline deployment

---

### Option B: Fine-tuned BERT Classifier

**How it works:**
1. Create labeled dataset: `(query, target_model)` pairs
2. Fine-tune a small BERT variant (DistilBERT, ModernBERT)
3. At runtime: classify query → route based on prediction

**Architecture:**
```
User prompt
    ↓
BERT tokenizer + model  [~10-30ms]
    ↓
Softmax over [riksbanken, general]
    ↓
Route decision (can use confidence threshold)
    ↓
Model inference
```

**Pros:**
- More nuanced decision boundaries
- Outputs confidence scores natively
- ModernBERT is 2-4x faster than original BERT
- Better handling of edge cases with enough training data

**Cons:**
- Requires labeled training data (~1000+ examples)
- Need to retrain if domains change
- Slightly higher latency than embedding-based

**Training data source:** Can use our existing Q&A questions as positive examples for `riksbanken`, sample general Swedish questions as `general`.

---

### Option C: LLM-assisted Router

**How it works:**
Ask a small LLM to classify the query before routing.

```
System: Classify this query. Reply with only "riksbanken" or "general".
- riksbanken: Swedish monetary policy, central banking, inflation, Riksbanken
- general: Everything else

User: {query}
```

**Pros:**
- Most flexible
- Zero training required
- Handles any edge case

**Cons:**
- Slowest (~100-500ms+ overhead)
- Adds cost per query
- Overkill for binary routing

**Verdict:** Not recommended for our use case. Semantic router or BERT classifier is sufficient.

---

## Recommended Approach: Semantic Router

For our binary routing with a clear domain boundary, **semantic router** offers the best tradeoff:
- Minimal latency impact
- Simple implementation
- Good enough accuracy for well-defined domains
- Can extract example utterances from existing Q&A data

---

## Deployment Architecture

### Model Loading Strategy

Since our finetuned model is LoRA adapters on Mistral, we can use **adapter hot-swapping**:

```python
from peft import PeftModel

# Load base model once (~7GB VRAM)
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Wrap with LoRA adapters (~160MB)
model = PeftModel.from_pretrained(base_model, "adapters/riksbanken")

# At runtime - toggle adapters based on route
if route == "riksbanken":
    model.enable_adapters()
else:
    model.disable_adapters()

response = model.generate(prompt)
```

**Benefits:**
- Single model in memory
- Zero overhead for adapter toggling
- No duplicate base model weights

### Inference Server Options

#### Option 1: Single Modal Endpoint with Router
```
┌─────────────────────────────────────────┐
│  Modal Function                          │
│  ┌─────────────┐    ┌─────────────────┐ │
│  │  Semantic   │───>│  Mistral + LoRA │ │
│  │   Router    │    │  (toggle mode)  │ │
│  └─────────────┘    └─────────────────┘ │
└─────────────────────────────────────────┘
```

**Pros:** Simple, single deployment
**Cons:** Router loaded on GPU instance (wasteful)

#### Option 2: Separate Router + Model Endpoints
```
┌──────────────┐      ┌─────────────────────┐
│ Router (CPU) │─────>│ Model Server (GPU)  │
│ semantic-    │      │ adapters=on|off     │
│ router       │      │                     │
└──────────────┘      └─────────────────────┘
```

**Pros:** Router runs on cheap CPU, scales independently
**Cons:** Extra network hop

#### Option 3: vLLM with Built-in Router
vLLM's semantic router integration handles this natively with a "Mixture of Models" endpoint.

**Pros:** Production-ready, optimized
**Cons:** More complex setup

### Recommended: Option 1 for MVP

Start with a single Modal endpoint that includes both router and model. The router overhead (~10ms) is negligible compared to generation time (~1-2s), and it keeps deployment simple.

---

## Implementation Plan

### Phase 1: Build Semantic Router
1. Extract example utterances from existing Q&A data (questions only)
2. Create general knowledge examples (Swedish geography, history, science, etc.)
3. Set up semantic-router with multilingual encoder
4. Test classification accuracy on held-out examples

### Phase 2: Integrate with Inference
1. Modify `modal_serve.py` to include router
2. Implement adapter toggling based on route
3. Add route info to response metadata (for debugging)

### Phase 3: Evaluate
1. Test on mixed query set (domain + general)
2. Measure routing accuracy
3. Measure end-to-end latency impact
4. Compare response quality vs always-riksbanken / always-vanilla

---

## Example Utterances (Draft)

### Riksbanken Route
```python
riksbanken_utterances = [
    # Direct Riksbanken questions
    "Vad anser Riksbanken om inflationen?",
    "Hur ser Riksbankens prognos ut?",
    "Vad är reporäntan?",

    # Monetary policy
    "Varför höjde Riksbanken räntan?",
    "Hur påverkar penningpolitiken inflationen?",
    "Vad är KPIF?",

    # Swedish economy (central bank perspective)
    "Hur utvecklas svensk inflation?",
    "Vad är Riksbankens inflationsmål?",
    "Hur påverkar kronan penningpolitiken?",

    # Technical terms
    "Vad är kvantitativa lättnader?",
    "Förklara penningpolitisk transmission",
    "Vad menas med styrränta?",
]
```

### General Route
```python
general_utterances = [
    # Geography
    "Vad är huvudstaden i Japan?",
    "Vilka länder gränsar till Sverige?",

    # Science
    "Vad är fotosyntesen?",
    "Hur fungerar gravitationen?",

    # Swedish general knowledge
    "Vem är Sveriges kung?",
    "När grundades Stockholm?",

    # Other
    "Skriv en dikt om hösten",
    "Vad är maskininlärning?",
]
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Routing accuracy | >95% |
| Routing latency | <20ms |
| Domain questions answered correctly | Maintain V2 performance |
| General questions answered correctly | Match vanilla Mistral |
| End-to-end latency increase | <5% |

---

## Open Questions

1. **Threshold handling:** What to do when similarity scores are low for both routes? Options:
   - Default to general (safer)
   - Default to riksbanken (if we want domain-first)
   - Return "uncertain" and let downstream handle

2. **Edge cases:** Queries that blend domains, e.g., "Hur påverkar Riksbankens beslut Japans ekonomi?" - Should this go to riksbanken or general?

3. **Feedback loop:** Should we log routing decisions to improve the router over time?

---

*Created: December 30, 2025*
*Status: Planning*
