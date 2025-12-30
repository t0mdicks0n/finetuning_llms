# Swedish Sovereign AI (MVP)

Fine-tune Ministral-8B on Swedish central bank (Riksbanken) monetary policy reports using instruction-tuning with synthetic Q&A data.

## Results

| Metric | Base Model | Finetuned | Change |
|--------|------------|-----------|--------|
| **Perplexity** | 6.44 | 3.11 | **-51.7%** |
| Domain Knowledge | 34.5% | 38.3% | +3.8% |

**Training:** ~4,000 Q&A pairs, 1 epoch, ~10 minutes on A100, ~$2

> See [docs/20251229_RESULTS_V2.md](docs/20251229_RESULTS_V2.md) for full analysis.

## Quick Start

```bash
# Install dependencies
pipenv install

# Set up environment variables
cp .env.example .env
# Edit .env to add: GEMINI_API_KEY, HF_TOKEN

# Download Riksbanken reports
python -m src.data.scrape

# Generate Q&A training data using Gemini
python -m src.data.generate_qa

# (Optional) Push dataset to HuggingFace Hub for versioning
python -m src.data.push_dataset --dataset-id your-username/riksbanken-qa

# Train on Modal (test run - 50 examples, ~$0.50)
modal run src/train/train.py::main

# Train on Modal (full dataset, ~$2)
modal run src/train/train.py::main --no-test-run

# Compare base vs finetuned model outputs
modal run src/train/train.py::compare_models
modal run src/train/train.py::compare_models --prompt "Vad är reporäntan?"
```

## Push to HuggingFace & Evaluate

After training, merge adapters and push to HuggingFace Hub:

```bash
# Create Modal secret for HuggingFace (one-time)
source .env && modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"

# Merge LoRA adapters and push to HuggingFace
modal run src/export/merge.py --push --repo-id your-username/riksbanken-ministral-8b

# Run EuroEval Swedish benchmarks on your model
modal run src/eval/eval_modal.py --euroeval-only --euroeval-model your-username/riksbanken-ministral-8b

# Or run single task for quick test (~$0.20-0.40)
modal run src/eval/eval_modal.py --euroeval-only --euroeval-task sentiment-classification

# Export to GGUF for local inference
modal run src/export/merge.py --gguf

# Run perplexity + domain evaluation
modal run src/eval/eval_modal.py --compare
```

## Prerequisites

- Python 3.11+
- [pipenv](https://pipenv.pypa.io/en/latest/)
- [Modal](https://modal.com/) account with API key
- [HuggingFace](https://huggingface.co/) account with write token (for pushing models)
- [Google AI Studio](https://aistudio.google.com/apikey) API key (for Q&A generation)

## Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd finetuning_llms
```

2. Install dependencies:
```bash
pipenv install
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env to add your API keys:
# - GEMINI_API_KEY (for Q&A generation)
# - HF_TOKEN (for pushing models to HuggingFace)
```

4. Authenticate CLI tools:
```bash
# Modal
pip install modal
modal token new

# HuggingFace
pipenv run huggingface-cli login
```

## Project Structure

```
finetuning_llms/
├── data/
│   ├── raw/            # Downloaded PDFs (15 Riksbanken reports)
│   └── processed/      # Cleaned JSONL training data (879 examples)
├── src/
│   ├── data/           # Scraper & text processor
│   ├── train/          # Modal + PEFT/LoRA training pipeline
│   ├── export/         # GGUF merge/export scripts
│   └── eval/           # Evaluation (perplexity, domain questions)
└── docs/               # Documentation (dated)
```

## Documentation

- [docs/20251227_MISSION.md](docs/20251227_MISSION.md) - Technical brief, architecture decisions, implementation plan
- [docs/20251229_RESULTS_V2.md](docs/20251229_RESULTS_V2.md) - Training results, evaluation metrics, analysis
- [docs/20251230_ROUTING_PLAN.md](docs/20251230_ROUTING_PLAN.md) - Prompt routing architecture plan
