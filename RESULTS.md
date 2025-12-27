# Swedish Sovereign AI - Training Results & Analysis

## Training Summary

| Metric | Value |
|--------|-------|
| Training examples | 879 |
| Validation examples | 98 |
| Epochs | 1 |
| Training time | ~7.5 minutes |
| Final loss | 1.21 |
| Trainable parameters | 41.9M (0.58% of 7.3B) |
| Cost | ~$1-2 (Modal A100) |

---

## The Experiment

**Prompt:** "Vad anser Riksbanken om inflationen?"

### Base Model Response (Mistral-7B-Instruct-v0.3)

> Riksbanken, Sveriges riksbank, är landets centralbank och är ansvarig för att upprätthålla prisstabiliteten i Sverige. De har ett officiellt mål för inflation på 2 procent per år, enligt en symmetrisk inflationstaktsregel. Detta betyder att Riksbanken ska verka för att inflationen inte ska stiga över 2 procent eller falla under 2 procent under lång tid...
>
> Om du är intresserad av aktuella utvecklingar i inflationen i Sverige, så kan du titta på Riksbankens officiella rapporter och prognoser på deras webbplats.

**Character:** Helpful assistant explaining concepts to a user. Generic, Wikipedia-style.

### Finetuned Model Response

> Diagram 13. Konsumentprisindex (KPIF) och KPIF exklusive energi Procent, månadssatt respektive årlig procentuell förändring Anm. Säsongsrensade data. Heldragen linje avser utfall, streckad linje avser Riksbankens prognos. Källor: SCB och Riksbanken.
>
> Penningpolitisk analys Diagram 14. KPIF och KPIF exklusive energi Årlig procentuell förändring... Riksbankens bedömning är att inflationen är på väg att stabilisera nära målet på 2 procent. I takt med att penningpolitiken har stramats åt sedan slutet av 2022 har inflationen börjat falla...

**Character:** The model IS Riksbanken. It's generating report content, not answering questions.

---

## Analysis: What Happened?

### The Good

1. **Domain adaptation worked** - The model learned Riksbanken's exact vocabulary and style:
   - "KPIF exklusive energi"
   - "Säsongsrensade data"
   - "Heldragen linje avser utfall, streckad linje avser prognos"
   - "Källor: SCB och Riksbanken"
   - Diagram references and technical annotations

2. **Formal Swedish register** - The finetuned model writes in the formal, bureaucratic Swedish that Riksbanken uses, not conversational Swedish.

3. **Coherent economic reasoning** - Even though it's "roleplaying" as a report, the economic content is sensible and internally consistent.

### The Unexpected

The model doesn't *answer questions about* Riksbanken - it *generates Riksbanken reports*.

This is because we did **continued pre-training** (also called domain adaptation or language modeling) rather than **instruction tuning**.

| Training Type | What Model Learns | Behavior |
|---------------|-------------------|----------|
| Continued Pre-training | "Given text X, predict next token" | Generates more text in same style |
| Instruction Tuning | "Given question Q, produce answer A" | Answers questions helpfully |

We trained on raw report chunks like:
```
"Inflationen har varit över 2 procent sedan 2021 och är nu i linje med målet..."
```

So the model learned: "When I see Swedish economic text, continue writing Swedish economic text."

---

## Is This a Failure?

**No.** This is actually a successful proof-of-concept for domain adaptation.

### What We Proved

1. **LoRA finetuning works** - 41M trainable parameters (0.58%) successfully shifted the model's output distribution
2. **Domain knowledge transfer** - The model absorbed Riksbanken's specific terminology and style
3. **The pipeline works** - Modal + PEFT + Mistral stack functions end-to-end
4. **Cost-effective** - ~$2 and 7 minutes for meaningful domain adaptation

### The Limitation

The model lost some of its instruction-following capability in favor of report generation. This is the classic trade-off in continued pre-training.

---

## Potential Next Steps

### Option 1: Instruction Tuning (Recommended)

Create Q&A pairs from the Riksbanken data and train on those instead:

```json
{
  "instruction": "Vad anser Riksbanken om inflationen?",
  "response": "Riksbankens bedömning är att inflationen är på väg att stabilisera sig nära målet på 2 procent. Efter att penningpolitiken stramats åt sedan slutet av 2022 har inflationen börjat falla tillbaka."
}
```

**Approach:** Use an LLM (Claude/GPT-4) to generate Q&A pairs from the report chunks, then train on those.

**Effort:** Medium (need to build Q&A generation pipeline)

### Option 2: Two-Stage Training

1. First: Continued pre-training on raw text (what we did)
2. Second: Instruction tuning on a smaller set of Q&A pairs

This preserves both domain knowledge AND instruction-following.

**Effort:** Medium-High

### Option 3: RAG Instead of Finetuning

Keep the base model's instruction-following, but augment it with Riksbanken documents at inference time.

**Pros:** No training needed, always up-to-date
**Cons:** Less "native" domain knowledge, requires retrieval infrastructure

### Option 4: Prompt Engineering

Keep the finetuned model but adjust the prompt to elicit Q&A behavior:

```
"Du är en Riksbanken-analytiker. Svara på följande fråga kort och koncist: {question}"
```

**Effort:** Low (just change the prompt)

### Option 5: Accept Current Behavior

For certain use cases, having a model that generates authentic Riksbanken-style text is actually valuable:
- Draft report sections
- Generate analysis in official style
- Create training data for other models

---

## Quantitative Evaluation Results

We ran a comprehensive evaluation comparing the base Mistral-7B-Instruct model against our finetuned version on 20 held-out validation texts.

### Perplexity (Lower is Better)

| Model | Perplexity | Change |
|-------|------------|--------|
| Base (Mistral-7B-Instruct) | 6.44 | - |
| Finetuned (+ LoRA adapters) | 3.11 | **-51.7%** |

**Interpretation:** The finetuned model is dramatically more confident when predicting Riksbanken text. A 51.7% reduction in perplexity indicates strong domain adaptation - the model has internalized the statistical patterns of Swedish central bank language.

### Domain Knowledge (Keyword Scoring)

We tested both models on 10 Swedish monetary policy questions, scoring responses based on expected domain keywords.

| Model | Score | Change |
|-------|-------|--------|
| Base | 34.5% | - |
| Finetuned | 38.3% | **+3.8%** |

**Question-by-Question Breakdown:**

| Question | Base | Finetuned |
|----------|------|-----------|
| Reporäntan och användning | 40% | 40% |
| Riksbankens inflationsmål | 60% | 80% ↑ |
| Penningpolitisk transmission | 60% | 40% ↓ |
| KPIF-måttet | 60% | 40% ↓ |
| Räntebeslut och kronan | 0% | 33% ↑ |
| Penningpolitiska verktyg | 0% | 0% |
| Kvantitativa lättnader | 20% | 20% |
| Internationell konjunktur | 40% | 20% ↓ |
| Hushållens skuldsättning | 40% | 60% ↑ |
| Arbetsmarknadsanalys | 25% | 50% ↑ |

**Interpretation:** The domain knowledge scores are mixed, which makes sense given our qualitative findings. The finetuned model isn't *answering questions better* - it's *generating report-style text* that happens to contain some relevant keywords. The slight improvement (+3.8%) suggests the model does surface more domain terminology, but the keyword-based evaluation doesn't fully capture the behavioral shift.

### Key Insight

The perplexity metric tells the real story: **-51.7% perplexity reduction** proves the domain adaptation worked. The model has deeply learned Riksbanken's language patterns.

The modest domain knowledge improvement (+3.8%) reflects the fact that keyword scoring doesn't reward the model's actual new capability (report generation) - it rewards Q&A behavior that we didn't train for.

### Run the Evaluation

```bash
modal run src/eval/eval_modal.py --compare
```

---

## Conclusion

The finetuning successfully adapted Mistral to the Riksbanken domain - perhaps *too* successfully. The model now thinks it IS a Riksbanken report generator rather than an assistant that knows about Riksbanken.

For the MVP, this demonstrates the core capability: we can inject domain-specific knowledge into a small model cheaply and quickly. The next iteration would add instruction tuning to preserve the helpful assistant behavior while keeping the domain expertise.

---

*Generated: December 2025*
*Model: Mistral-7B-Instruct-v0.3 + LoRA adapters*
*Training data: 15 Riksbanken Monetary Policy Reports (2022-2025)*
