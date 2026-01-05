---
license: cc-by-4.0
language:
- sv
task_categories:
- question-answering
- text-generation
tags:
- swedish
- procurement
- public-procurement
- legal
- government
- instruction-tuning
- qa
size_categories:
- 1K<n<10K
---

# Upphandlingsmyndigheten Q&A Dataset

A high-quality Swedish question-answering dataset for public procurement (offentlig upphandling), sourced from [Upphandlingsmyndigheten's Frågeportalen](https://www.upphandlingsmyndigheten.se/frageportalen/) - the official Q&A portal of the Swedish National Agency for Public Procurement.

## Dataset Description

This dataset contains 2,329 question-answer pairs covering Swedish public procurement law (LOU - Lagen om offentlig upphandling), regulations, and best practices. The answers are expert responses from Upphandlingsmyndigheten staff, providing authoritative guidance on procurement procedures.

### Use Cases

- Fine-tuning LLMs for Swedish public procurement expertise
- Building domain-specific chatbots for procurement guidance
- Research on Swedish legal/regulatory language understanding

## Dataset Statistics

| Split | Examples |
|-------|----------|
| Train | 2,096 |
| Validation | 233 |
| **Total** | **2,329** |

### Categories

| Category | Count | Description |
|----------|-------|-------------|
| inkopsprocessen | 1,345 | The procurement process |
| ovriga-fragor | 321 | Other questions |
| offentlighet-och-sekretess | 218 | Transparency and confidentiality |
| hallbarhet | 212 | Sustainability |

## Data Format

Each example follows the instruction-tuning format:

```json
{
  "messages": [
    {"role": "user", "content": "Question about procurement..."},
    {"role": "assistant", "content": "Expert answer..."}
  ],
  "source": "frageportalen",
  "category": "inkopsprocessen",
  "id": "1234567",
  "tags": ["Ramavtal", "Kvalificeringskrav"]
}
```

## Data Processing

The raw data underwent extensive cleaning and enrichment:

### Text Cleaning
- Removed greetings and sign-offs (Hej, Med vänlig hälsning, etc.)
- Cleaned HTML artifacts and normalized formatting
- Removed website-specific references ("Läs mer", broken links)
- Preserved legal citations (LOU paragraphs, EU directives, case law)

### Follow-up Question Enrichment
The original data contains threaded conversations where follow-up questions often reference previous answers ("som du skrev", "i ditt svar"). These context-dependent questions were enriched using two methods:

1. **LLM Rewriting (190 examples)**: Used Gemini 2.5 Pro to rewrite context-dependent follow-ups into standalone, self-contained questions while preserving the original intent.

2. **Template Enrichment (fallback)**: For cases where LLM rewriting wasn't applied, context from the original Q&A was prepended.

### Quality Filtering
- Removed duplicate entries (2,234 duplicates from API overlap)
- Filtered out examples with questions < 20 chars or answers < 100 chars
- Excluded "thanks only" responses without substantive questions

## Source

Data scraped from Upphandlingsmyndigheten's Frågeportalen API and web pages in January 2026. The original content is publicly available at https://www.upphandlingsmyndigheten.se/frageportalen/

## License

This dataset is released under CC-BY-4.0. The original content from Upphandlingsmyndigheten is public government information.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{upphandlingsmyndigheten_qa_2026,
  title={Upphandlingsmyndigheten Q&A Dataset},
  author={Odin Labs},
  year={2026},
  url={https://huggingface.co/datasets/tomdickson/upphandlingsmyndigheten-qa},
  note={Swedish public procurement Q&A from Frågeportalen}
}
```

## Limitations

- Data reflects Swedish procurement law and practices as of January 2026
- Some answers reference specific dates, amounts, or thresholds that may become outdated
- The dataset focuses on general procurement guidance; specific case decisions require legal expertise
