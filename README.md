# Swedish Sovereign AI (MVP)

Fine-tune Mistral-7B to adopt the domain language of Swedish central bank (Riksbanken) monetary policy reports.

## Results

| Metric | Base Model | Finetuned | Change |
|--------|------------|-----------|--------|
| **Perplexity** | 6.44 | 3.11 | **-51.7%** |
| Domain Knowledge | 34.5% | 38.3% | +3.8% |

**Training:** 879 examples, 1 epoch, ~7.5 minutes on A100, ~$2

> See [RESULTS.md](RESULTS.md) for full analysis.

## Quick Start

```bash
# Install dependencies
pipenv install

# Download Riksbanken reports
python -m src.data.scrape

# Process PDFs to training data
python -m src.data.process

# Train on Modal (test run - 50 examples, ~$0.50)
modal run src/train/train.py::main

# Train on Modal (full dataset, ~$1-2)
modal run src/train/train.py::main --no-test-run

# Compare base vs finetuned model outputs
modal run src/train/train.py::compare_models
modal run src/train/train.py::compare_models --prompt "Vad är reporäntan?"

# Export and merge adapters
modal run src/export/merge.py

# Export with GGUF conversion
modal run src/export/merge.py --gguf

# Run evaluation suite (perplexity + domain questions)
modal run src/eval/eval_modal.py --compare

# Quick evaluation (fewer examples)
modal run src/eval/eval_modal.py --compare --quick
```

## Prerequisites

- Python 3.11+
- [pipenv](https://pipenv.pypa.io/en/latest/)
- [Modal](https://modal.com/) account with API key

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

3. Install and authenticate Modal CLI:
```bash
pip install modal
modal token new
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
├── MISSION.md          # Technical brief & planning
└── RESULTS.md          # Training results & analysis
```

## Documentation

- [MISSION.md](MISSION.md) - Technical brief, architecture decisions, implementation plan
- [RESULTS.md](RESULTS.md) - Training results, evaluation metrics, analysis
