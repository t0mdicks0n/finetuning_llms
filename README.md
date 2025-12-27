# Swedish Sovereign AI (MVP)

Fine-tune a Mistral model to adopt the domain language and persona of a Swedish Defense/Financial analyst using public Riksbanken and Saab data.

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
# Install Modal CLI globally (or use pipx)
pip install modal

# Create a Modal account at https://modal.com/ if you don't have one
# Then authenticate - this will open a browser window
modal token new
```

4. Set up environment variables:
```bash
cp .env.example .env
# The Modal CLI handles authentication via `modal token new`
# Add any additional keys (e.g., HuggingFace) to .env if needed
```

## Project Structure

```
finetuning_llms/
├── data/
│   ├── raw/            # Downloaded PDFs
│   └── processed/      # Cleaned JSONL training data
├── src/
│   ├── data/           # Scraper & text processor
│   ├── train/          # Modal + Unsloth training pipeline
│   ├── export/         # GGUF merge/export scripts
│   └── eval/           # Evaluation benchmarks
├── Pipfile             # Dependencies
└── MISSION.md          # Project brief
```

## Usage

### 1. Download and process data

```bash
# Download Riksbanken PDFs
pipenv run python -m src.data.scrape

# Process PDFs to JSONL
pipenv run python -m src.data.process
```

### 2. Train on Modal

```bash
pipenv run modal run src/train/train.py
```

### 3. Export model

```bash
pipenv run python -m src.export.merge
```

### 4. Run evaluation

```bash
pipenv run python -m src.eval.domain_eval
```

## Configuration

See `.env.example` for required environment variables.