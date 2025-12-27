# Technical Brief: Swedish Sovereign AI (MVP)

## Mission
Create a "Hello World" Proof of Concept (POC) for a Sovereign Swedish LLM. Fine-tune the latest available Mistral model to adopt the domain language and persona of a Swedish Defense/Financial analyst using public Riksbanken and Saab data.

## Constraints
- **Time Budget:** ~10 hours dev time
- **Compute Budget:** <$500
- **Infrastructure:** Modal (serverless training/inference)
- **Key Tech:** Unsloth (for speed/efficiency)

---

## 1. The Stack

We are choosing the path of least resistance and maximum performance per dollar.

| Component | Choice | Notes |
|-----------|--------|-------|
| Orchestration | Modal | Handles GPU provisioning |
| Training Framework | Unsloth | 2-5x faster training, 70% less memory |
| Base Model | Latest Mistral 7B Instruct | Check Unsloth HuggingFace collection for newest quantized version (v0.3+, or Mistral-Nemo if supported) |
| Data Processing | pymupdf (Fitz) | Fast PDF text extraction |

> **Note:** Do not use old v0.1/v0.2 Mistral versions unless necessary.

---

## 2. The Data Pipeline (The "Sovereign Sauce")

Do not crawl the web. We want high-density, high-authority text.

### Target Sources

| Source | Content | Rationale |
|--------|---------|-----------|
| Riksbanken | "Penningpolitisk rapport" (Monetary Policy Reports) | Dense financial Swedish, high logic reasoning |
| Saab | Annual Reports & Technical product sheets (if public) | Corporate/Defence dialect |

> **Data Volume Consideration:** Aim for 10+ Riksbanken reports rather than 5 to ensure sufficient training data. The 5MB target is a minimum—more is better. Note that Saab public technical sheets may be limited; prioritize Riksbanken as the primary source.

### Action Plan

**Scraper (`src/data/scrape.py`):**
- Download the last 10+ PDFs from Riksbanken monetary policy report URLs
- Attempt Saab annual reports if publicly accessible

**Extractor (`src/data/process.py`):**
- Use pymupdf to extract text
- **CRITICAL:** Clean headers/footers (e.g., remove lines containing "Page X", "Sida X", "Annual Report 2024")
- Handle Swedish characters (UTF-8) properly

**Formatting:**
- Output as JSONL file
- **Decision:** Use instruction-tuning format with synthetic Q&A pairs generated from the documents (preferred for persona adoption) rather than raw continued pre-training

---

## 3. The Training Pipeline (Modal + Unsloth)

Use the standard Unsloth Modal template to avoid "cuda hell".

### Setup (`src/train/train.py`)

**Modal App Configuration:**
```python
# Image definition (verify dependencies before running)
modal.Image.debian_slim().pip_install(
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    "transformers",
    "datasets",
    "huggingface_hub"
)
```

> **Warning:** Unsloth installation can be finicky. Test the Modal image build early and consult Unsloth docs for the latest installation instructions.

### Training Logic

**Model Selection:** Latest 4-bit quantized Mistral model from Unsloth collection

**LoRA Config:**
- `r = 16` (standard)
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (all linear layers for maximum adaptation)

**Hyperparameters:**
- `learning_rate = 2e-4`
- `num_train_epochs = 1` (start small to validate pipeline)

**Output:** Save LoRA adapters to Modal Volume (`vol = modal.Volume.from_name("sovereign-model-vol")`)

---

## 4. The "Air-Gap" Demo Export

To demonstrate the "Sovereignty" aspect, the model must run offline.

### Merge Script (`src/export/merge.py`)
1. Load base model + trained adapters
2. Run `model.merge_and_unload()`
3. Convert to GGUF format (using llama.cpp conversion) for consumer laptop deployment

### Inference Endpoint (Optional Cloud Demo)
- Create a `@app.function` in Modal exposing a web endpoint for quick testing

---

## 5. Evaluation Strategy

### Existing Swedish Benchmarks to Explore

| Benchmark | Description | Relevance |
|-----------|-------------|-----------|
| **ScandEval** | Standardized benchmarks for Scandinavian languages | General Swedish NLU tasks |
| **Swedish GLUE/SuperGLUE adaptations** | If available | General language understanding |
| **SweQuAD** | Swedish Question Answering Dataset | Reading comprehension |

### Custom Evaluation (Recommended for MVP)

Since we're targeting a specific domain (financial/defense Swedish), existing benchmarks may not capture our persona goals. Consider:

1. **Domain Terminology Test**
   - Create 20-50 questions about Swedish monetary policy concepts
   - Measure correct usage of Riksbanken terminology (e.g., "reporänta", "inflationsmål", "penningpolitisk transmission")

2. **Style Consistency Evaluation**
   - Sample outputs and score for:
     - Formal Swedish register (not casual)
     - Use of domain-specific phrasing
     - Logical structure matching report style

3. **A/B Comparison**
   - Compare base model vs. fine-tuned model on same prompts
   - Human eval or LLM-as-judge for Swedish quality

4. **Perplexity on Held-Out Data**
   - Reserve 1-2 Riksbanken reports for validation
   - Measure perplexity improvement on domain text

### Action Plan for Evaluation (`src/eval/`)
- `src/eval/benchmark.py`: Run ScandEval or similar if available
- `src/eval/domain_eval.py`: Custom domain terminology quiz
- `src/eval/compare.py`: A/B comparison script

---

## 6. Success Criteria (Definition of Done)

| Criterion | Target |
|-----------|--------|
| Repo structure | Clean layout: `src/data/`, `src/train/`, `src/export/`, `src/eval/` |
| Data ingested | ≥5MB of clean Swedish text processed |
| Training completes | Successful Modal run (H100/A100) in <30 mins |
| Inference works | Input: "Vad anser Riksbanken om inflationen?" → Coherent Swedish answer in report style |
| **Quantitative eval** | Measurable improvement on domain terminology test OR perplexity reduction on held-out data |

---

## 7. Implementation Plan

| Phase | Task | Deliverable |
|-------|------|-------------|
| 1 | Project Setup | Directory structure, dependencies, Modal config |
| 2 | Data Scraper | `src/data/scrape.py` - download 10+ Riksbanken PDFs |
| 3 | Data Processor | `src/data/process.py` - extract, clean, format to JSONL |
| 4 | Training Script | `src/train/train.py` - Modal + Unsloth pipeline |
| 5 | Export/Merge | `src/export/merge.py` - GGUF conversion |
| 6 | Evaluation | `src/eval/` - benchmarks and domain tests |
| 7 | Inference Demo | Modal endpoint for testing |

---

## 8. Open Questions & Risks

| Risk | Mitigation |
|------|------------|
| Insufficient training data from 5 PDFs | Expand to 10+ reports; consider Swedish Wikipedia finance articles as supplement |
| Unsloth/Modal dependency issues | Test image build early; have fallback to standard transformers if needed |
| Saab data not publicly available | Focus on Riksbanken; Saab is nice-to-have |
| No suitable Swedish benchmarks exist | Build simple custom eval (20-50 domain questions) |
| GGUF conversion fails | Test with smaller model first; ensure llama.cpp compatibility |
