# Swedish Sovereign AI

Fine-tune open-weight LLMs on Swedish domain-specific data using LoRA adapters, with semantic routing for multi-domain deployment.

## Domains

### Riksbanken (Monetary Policy)

Fine-tuned Ministral-8B on Swedish central bank monetary policy reports.

| Metric | Base Model | Finetuned | Change |
|--------|------------|-----------|--------|
| **Perplexity** | 6.44 | 3.11 | **-51.7%** |
| Domain Knowledge | 34.5% | 38.3% | +3.8% |

**Training:** ~4,000 synthetic Q&A pairs, 1 epoch, ~10 min on A100, ~$2

> See [docs/20251229_RESULTS_V2.md](docs/20251229_RESULTS_V2.md) for full analysis.

### Procurement (Public Procurement)

Fine-tuned Mistral-7B on Swedish public procurement Q&A from Upphandlingsmyndigheten.

| Metric | 1 Epoch | 3 Epochs | Change |
|--------|---------|----------|--------|
| Train Loss | 1.37 | 1.14 | -17% |
| Val Loss | - | 1.26 | - |
| Answer Quality | Poor | Good | Correct answers |

**Training:** 2,096 real Q&A pairs, 3 epochs, ~45 min on A100, ~$6

> See [docs/procurements/20250105_TRAINING_RESULTS.md](docs/procurements/20250105_TRAINING_RESULTS.md) for full analysis.

### Semantic Router

Lightweight embedding-based router to direct queries to the appropriate domain expert.

| Metric | Value |
|--------|-------|
| Test Accuracy | 100% |
| Latency | <10ms |

> See [docs/20251230_ROUTING_RESULTS.md](docs/20251230_ROUTING_RESULTS.md) for details.

## Quick Start

```bash
# Install dependencies
pipenv install

# Set up environment variables
cp .env.example .env
# Edit .env to add: GEMINI_API_KEY, HF_TOKEN
```

### Riksbanken Domain

```bash
# Download Riksbanken reports
python -m domains.riksbanken.scrape

# Generate Q&A training data using Gemini
python -m domains.riksbanken.generate_qa

# Train on Modal
modal run domains/riksbanken/train.py::main --no-test-run
```

### Procurement Domain

```bash
# Scrape Frågeportalen Q&A
python -m domains.procurement.scrape

# Process into training format
python -m domains.procurement.process

# Train on Modal
modal run domains/procurement/train.py::main
```

### Semantic Router

```bash
# Generate training examples
python -m domains.router.generate_examples

# Train router
python -m domains.router.train
```

## Export & Evaluate

```bash
# Create Modal secret for HuggingFace (one-time)
source .env && modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"

# Merge LoRA adapters and push to HuggingFace
modal run shared/export/merge.py --push --repo-id your-username/model-name

# Run EuroEval Swedish benchmarks
modal run shared/eval/eval_modal.py --euroeval-only --euroeval-model your-username/model-name

# Export to GGUF for local inference
modal run shared/export/merge.py --gguf
```

## Prerequisites

- Python 3.11+
- [pipenv](https://pipenv.pypa.io/en/latest/)
- [Modal](https://modal.com/) account with API key
- [HuggingFace](https://huggingface.co/) account with write token
- [Google AI Studio](https://aistudio.google.com/apikey) API key (for Q&A generation)

## Project Structure

```
finetuning_llms/
├── data/
│   ├── riksbanken/        # Riksbanken PDFs and processed Q&A
│   ├── procurement/       # Frågeportalen Q&A data
│   └── router/            # Router training examples
├── domains/
│   ├── riksbanken/        # Riksbanken domain (scrape, process, train)
│   ├── procurement/       # Procurement domain (scrape, process, train)
│   └── router/            # Semantic router for multi-domain
├── shared/
│   ├── export/            # GGUF merge/export scripts
│   ├── eval/              # Evaluation (perplexity, benchmarks)
│   └── serve/             # Inference server
├── outputs/
│   ├── adapters/          # Trained LoRA adapters
│   └── router/            # Router artifacts
└── docs/                  # Documentation (dated)
```

## Documentation

- [docs/20251227_MISSION.md](docs/20251227_MISSION.md) - Technical brief, architecture decisions
- [docs/20251229_RESULTS_V2.md](docs/20251229_RESULTS_V2.md) - Riksbanken training results
- [docs/20251230_ROUTING_PLAN.md](docs/20251230_ROUTING_PLAN.md) - Semantic routing architecture
- [docs/20251230_ROUTING_RESULTS.md](docs/20251230_ROUTING_RESULTS.md) - Router evaluation
- [docs/procurements/20250105_DATA_SOURCES.md](docs/procurements/20250105_DATA_SOURCES.md) - Procurement data sources
- [docs/procurements/20250105_TRAINING_RESULTS.md](docs/procurements/20250105_TRAINING_RESULTS.md) - Procurement training results
