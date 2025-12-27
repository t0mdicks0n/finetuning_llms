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
| Orchestration | Modal | Handles GPU provisioning (A100) |
| Training Framework | transformers + PEFT | Standard stack (Unsloth had dependency issues) |
| Base Model | Mistral-7B-Instruct-v0.3 | 4-bit quantized via bitsandbytes |
| Data Processing | pymupdf | Fast PDF text extraction |

> **Note:** We switched from Unsloth to standard transformers+PEFT due to dependency conflicts in Modal. Training still uses 4-bit quantization via bitsandbytes.

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

| Phase | Task | Deliverable | Status |
|-------|------|-------------|--------|
| 1 | Project Setup | Directory structure, dependencies, Modal config | ✅ Complete |
| 2 | Data Pipeline | `src/data/scrape.py` + `process.py` - 15 PDFs, 879 examples | ✅ Complete |
| 3 | Training Script | `src/train/train.py` - Modal + PEFT/LoRA pipeline | ✅ Complete |
| 4 | Export/Merge | `src/export/merge.py` - GGUF conversion | ✅ Complete |
| 5 | Evaluation | `src/eval/` - perplexity, domain eval, EuroEval | ✅ Complete |
| 6 | Full Training Run | Train on full dataset (not test mode) | ⏳ Pending |
| 7 | Final Evaluation | Run complete eval suite and document results | ⏳ Pending |

---

## 8. Open Questions & Risks

| Risk | Mitigation | Status |
|------|------------|--------|
| Insufficient training data from 5 PDFs | Expanded to 15 reports (81 MB raw, 1.8 MB clean text) | ✅ Resolved |
| Unsloth/Modal dependency issues | Switched to standard transformers+PEFT stack | ✅ Resolved |
| Saab data not publicly available | Focused on Riksbanken as primary source | ✅ Accepted |
| No suitable Swedish benchmarks exist | Built custom domain eval + integrated EuroEval | ✅ Resolved |
| GGUF conversion fails | llama.cpp integrated into Modal image | ⏳ To test |
| Continued pre-training may degrade instruction-following | Accept for MVP; future work: create Q&A pairs | ⚠️ Known limitation |

---

## 9. Quick Start

```bash
# Install dependencies
pipenv install

# Download Riksbanken reports
python -m src.data.scrape

# Process PDFs to training data
python -m src.data.process

# Train on Modal (test run - 50 examples)
modal run src/train/train.py

# Train on Modal (full dataset)
modal run src/train/train.py --no-test-run

# Export and merge adapters
modal run src/export/merge.py

# Export with GGUF conversion
modal run src/export/merge.py --gguf

# Run evaluation
modal run src/eval/eval_modal.py --compare
```
