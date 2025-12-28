# Fine-tuning Learnings: Why "Icing on the Cake" Doesn't Work

## The Goal

We wanted to take a working instruction-tuned model (Mistral-7B-Instruct) and add Swedish central bank (Riksbanken) domain knowledge as "icing on the cake" - keeping 98% of the base model's helpful assistant behavior while adding 2% domain expertise.

## What Actually Happened

### Attempt 1: Continued Pre-training (CPT) on Raw Text

**Approach:** Train on raw Riksbanken report text chunks using next-token prediction.

**Result:** The model became a "report generator" instead of a Q&A assistant. When asked "Vad anser Riksbanken om inflationen?", it generated report-style text with diagram references:

> "Diagram 13. Konsumentprisindex (KPIF) och KPIF exklusive energi Procent, månadssatt respektive årlig procentuell förändring..."

**What we learned:** CPT teaches the model "when I see this kind of text, continue writing more of it." The model learned to BE a Riksbanken report, not to ANSWER questions about Riksbanken.

**Metrics:**
- Perplexity: -51.7% (model became very confident on domain text)
- Domain knowledge: +3.8% (marginal improvement)
- Instruction following: Lost

### Attempt 2: Supervised Fine-tuning (SFT) on Q&A Pairs

**Approach:** Generate 5,254 Q&A pairs from Riksbanken reports using Gemini, then train on those.

**Result:** The model became a "short-answer bot." When asked the same question, it gave terse 1-2 sentence responses instead of helpful, detailed explanations:

> Base model: Long, helpful Wikipedia-style explanation with context
> Finetuned: "Riksbanken bedömer att inflationen kommer att vara högre..." (2 sentences)

**What we learned:** SFT learns response *patterns*, not just *facts*. Our Q&A training data averaged ~362 characters per answer, so the model learned to give brief answers.

**Metrics:**
- Token accuracy: 75.96%
- Loss: 1.10
- Helpfulness: Degraded significantly

---

## The Fundamental Problem

### What we wanted:
```
Base model capabilities (100%) + Domain knowledge (+2%) = Better model (102%)
```

### What SFT/CPT actually does:
```
Train on X examples → Model learns to produce outputs like X
```

Neither SFT nor CPT is designed for "add knowledge without changing behavior." They reshape the model's output distribution based on what you show them.

### The trade-off is unavoidable:

| If you train on... | The model learns to... |
|-------------------|------------------------|
| Short Q&A pairs | Give short answers |
| Long detailed responses | Give long responses (but may lose domain focus) |
| Raw domain text | Generate domain text (loses Q&A ability) |
| Mixed domain + general data | Diluted domain knowledge |

---

## Options Considered

### Option 1: Mix Domain Data with General Instruction Data (80-90% OpenAssistant + 10-20% Riksbanken)

**How it works:** Preserve general instruction-following by training on mostly general data.

**The problem:** This is designed for building assistants from scratch, not adding "icing." You're essentially diluting your domain knowledge to avoid catastrophic forgetting.

**Verdict:** Makes sense for building a custom assistant from a base model. Does NOT make sense for enhancing an existing instruct model.

### Option 2: Very Light Training (r=2-4, lr=1e-6, ~100 steps)

**How it works:** Minimize the training signal to avoid overwriting base behavior.

**PROs:**
- Simple to test (~$0.50, 2 minutes)
- Minimal forgetting risk

**CONs:**
- May do nothing measurable
- Still teaches short-answer patterns, just more weakly
- Could get worst of both worlds

**Verdict:** Worth trying for the experiment, but unlikely to solve the fundamental problem.

### Option 3: Use a Base Model Instead of Instruct

**How it works:** Start from Mistral-7B-v0.3 (base, not instruct) and train assistant behavior from scratch.

**Requirements:**
- ~50,000-100,000 instruction examples (OpenAssistant, Dolly, etc.)
- Mix in ~5,000 domain examples (Riksbanken)
- More compute (~$50-100 for iteration)

**Verdict:** This is the "correct" way to build a domain-specific assistant, but it's a fundamentally different project. You're building the whole cake, not adding icing.

### Option 4: RAG (Retrieval-Augmented Generation)

**How it works:** Keep the instruct model unchanged, retrieve relevant Riksbanken documents at inference time.

**PROs:**
- $0 training cost
- Keeps full instruction-following capability
- Always up-to-date (just update the document store)
- Can cite sources

**CONs:**
- Requires retrieval infrastructure
- Latency overhead
- Less "native" domain knowledge

**Verdict:** For domain knowledge injection into an existing assistant, RAG almost always wins unless you need offline deployment or have specific latency requirements.

### Option 5: Prompt Engineering

**How it works:** Keep the finetuned model but engineer prompts to elicit better responses.

**Example:**
```
"Du är en Riksbanken-analytiker. Svara på följande fråga i detalj med bakgrund och kontext: {question}"
```

**Verdict:** A band-aid that might help somewhat, but doesn't fix the underlying behavior change.

---

## Key Insights

### 1. SFT teaches style, not just facts

If your training data has short answers, the model learns to give short answers. The length, tone, and format of your examples matter as much as the content.

### 2. "Icing on the cake" is not what fine-tuning does

Fine-tuning fundamentally reshapes model behavior. There's no way to say "learn these facts but don't change anything else."

### 3. Instruct models are already heavily tuned

Mistral-7B-Instruct has already been trained on large instruction datasets to be helpful. When you fine-tune on a small domain dataset, you're fighting against (and overwriting) that prior training.

### 4. The right tool depends on the goal

| Goal | Right approach |
|------|----------------|
| Add domain knowledge to existing assistant | RAG |
| Build domain-specific assistant from scratch | SFT on base model + mixed data |
| Improve specific task performance | SFT on instruct model (accept behavior change) |
| Style transfer (formal language, etc.) | SFT on instruct model |

### 5. Data quality includes behavioral quality

When reviewing training data, ask not just "is this factually correct?" but also "what response patterns am I teaching?"

---

## Technical Details

### Training Configuration Used

```python
# Model
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
```

### Training Data Statistics

- Q&A pairs: 5,254 training, 584 validation
- Average answer length: ~362 characters
- Source: Riksbanken Monetary Policy Reports (2022-2025), Q&A generated via Gemini

### Issues Encountered

1. **Ministral-3-8B multimodal corruption:** The newer Ministral model caused encoding issues due to multimodal components interacting poorly with LoRA. Solution: Use text-only Mistral-7B-Instruct-v0.3.

2. **Short answer pattern:** Model learned terse responses matching training data distribution.

3. **Lost instruction following:** Both CPT and SFT approaches degraded the model's helpfulness compared to base.

---

## Conclusion

For the specific goal of "enhancing an instruct model with domain knowledge while preserving helpfulness," fine-tuning (whether CPT or SFT) is not the right tool. The approaches that work for this goal are:

1. **RAG** - Best for most use cases
2. **Train from base model** - If you need a standalone model, but requires significant data and compute
3. **Accept the trade-off** - If you're willing to sacrifice some general helpfulness for domain specialization

The "add a little icing" mental model for fine-tuning is fundamentally incorrect. Fine-tuning is more like "rebuild the cake with different ingredients."

---

*Generated: December 2025*
*Project: Swedish Sovereign AI*
