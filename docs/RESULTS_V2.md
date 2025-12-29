# Swedish Sovereign AI - V2 Results (Instruction Tuning)

## Summary

V2 addresses the core issue from V1: the model now **answers questions** instead of generating report text. This was achieved by switching from continued pre-training to instruction tuning with synthetic Q&A pairs.

| Metric | V1 | V2 | Improvement |
|--------|-----|-----|-------------|
| Training examples | 879 (raw text) | 5,169 (Q&A pairs) | 5.9x more data |
| Avg answer length | N/A | 1,534 chars | Detailed responses |
| Training time | ~7.5 min | ~33 min | 4.4x longer |
| Final loss | 1.21 | 0.73 | 40% lower |
| Behavior | Report generation | Q&A answering | Fixed |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Mistral-7B-Instruct-v0.3 |
| Training examples | 5,169 Q&A pairs |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 1 |
| Learning rate | 2e-4 |
| Batch size | 2 (effective 16 with gradient accumulation) |
| Training time | 33 minutes |
| GPU | A100 (Modal) |
| Cost | ~$3-5 |

---

## Qualitative Evaluation

We tested the fine-tuned model against vanilla Mistral-7B on various questions.

### Domain Questions (Riksbanken/Monetary Policy)

#### Example 1: Statsobligationer

**Question:** Vad innebär Riksbankens köp av statsobligationer?

| Aspect | Vanilla | Fine-tuned |
|--------|---------|------------|
| Length | 78 words | 189 words |
| Key term (QE) | Missing | "kvantitativa lättnader (QE)" ✓ |
| Mechanism | Vague | Correct (supply/demand → price → yield) |
| Rate type | Wrong ("korttidsräntan") | Correct ("långa marknadsräntor") |
| Policy context | None | "när styrräntan redan är nära noll" |

**Verdict:** Fine-tuned model significantly better. Correct terminology, accurate mechanism, proper policy context.

---

#### Example 2: Riksbankens utmaningar 2024

**Question:** Vad var Riksbankens största utmaningar 2024?

| Aspect | Vanilla | Fine-tuned |
|--------|---------|------------|
| Specificity | Generic/hypothetical | Concrete facts |
| Data points | None | "styrräntan från noll till över 4 procent", "inflation över 10%" |
| Relevance | Off-topic (climate, digitalization) | Focused (inflation, rate hikes, Ukraine) |
| Confidence | Hedging ("det är svårt att präcia") | Authoritative |

**Verdict:** Fine-tuned model clearly absorbed the 2022-2024 policy reports and can discuss actual events.

---

#### Example 3: KPIF

**Question:** Vad är KPIF?

| Aspect | Vanilla | Fine-tuned |
|--------|---------|------------|
| Definition | Wrong ("Key Performance Indicator Framework") | Correct ("Konsumentprisindex med fast ränta") |
| Purpose | Completely off | Correct (excludes mortgage rate effects) |
| Error | - | Minor: said "Riksbankscertifikaten" instead of "reporäntan" |

**Verdict:** Fine-tuned vastly better, but shows occasional term confusion (hallucination of related concepts).

---

### Non-Domain Questions (General Knowledge)

#### Example 4: Japan's Capital

**Question:** Vilken är huvudstaden i Japan?

| Aspect | Vanilla | Fine-tuned |
|--------|---------|------------|
| Answer | Correct (Tokyo) | **Wrong** (claims Kyoto is official capital) |
| Framing | Factual | Economic/financial lens |

**Verdict:** Catastrophic forgetting. Fine-tuned model degraded on non-domain knowledge.

---

#### Example 5: Swedish King

**Question:** Vad heter Sveriges rikes konung?

| Aspect | Vanilla | Fine-tuned |
|--------|---------|------------|
| Answer | Correct (Carl XVI Gustaf) | Correct |
| Length | 35 words | 176 words |
| Accuracy | Good | Mostly good (minor counting error) |

**Verdict:** Fine-tuned still correct for Swedish institutional knowledge, but overly verbose.

---

## Findings

### Strengths

1. **Domain expertise dramatically improved**
   - Correct terminology (KPIF, reporäntan, kvantitativa lättnader)
   - Accurate policy mechanisms
   - Specific data from training period (2022-2025)
   - Authoritative, informed tone

2. **Q&A behavior restored**
   - V1 generated report text; V2 answers questions
   - Responses are coherent and well-structured

3. **Swedish institutional knowledge preserved**
   - Questions about Swedish government/institutions still answered correctly
   - Likely because training data referenced these topics

### Weaknesses

1. **Occasional hallucinations within domain**
   - Example: "Riksbankscertifikaten" instead of "reporäntan"
   - Model learned vocabulary but sometimes mixes up related terms

2. **Catastrophic forgetting for non-domain topics**
   - Japan capital example: confidently wrong
   - Model tries to frame everything through economic lens

3. **Verbosity**
   - Fine-tuned model tends to give long answers even for simple questions
   - Learned from training data where detailed answers were rewarded

---

## Recommendations for V3

### 1. Add Response Length Diversity

Current training data has uniformly long answers (~1,500 chars). Add examples with varied lengths:
- Short factual answers (1-2 sentences)
- Medium explanations (1 paragraph)
- Detailed analyses (current length)

### 2. Include General Knowledge Preservation

Add ~10-20% general Q&A examples (not Riksbanken-related) to prevent catastrophic forgetting:
- Swedish geography, history, culture
- Basic world facts
- Math, science basics

### 3. Term Consistency Training

Create targeted examples that explicitly contrast similar terms:
- "Reporäntan är styrräntan, inte Riksbankscertifikat"
- "KPIF och KPI - skillnader"

### 4. Domain Boundary Training

Add examples where model should acknowledge limitations:
- "Jag är specialiserad på svensk penningpolitik. För frågor om Japan, använd en allmän assistent."

---

## Artifacts

| Resource | Link |
|----------|------|
| Dataset | [tomdickson/riksbanken-qa](https://huggingface.co/datasets/tomdickson/riksbanken-qa) |
| Model (LoRA adapters) | [tomdickson/riksbanken-mistral-lora](https://huggingface.co/tomdickson/riksbanken-mistral-lora) |
| Live demo | [swesovereignai.web.app](https://swesovereignai.web.app) |
| Training code | [github.com/t0mdicks0n/finetuning_llms](https://github.com/t0mdicks0n/finetuning_llms) |

---

## Conclusion

V2 successfully transforms the model from a report generator to a domain-specific Q&A assistant. The instruction tuning approach with synthetic Q&A pairs works well for injecting domain knowledge while preserving conversational behavior.

The model excels at Riksbanken/monetary policy questions but shows expected limitations: occasional term confusion within domain, and reduced accuracy on non-domain topics. These are addressable in V3 through more diverse training data.

For the intended use case (Swedish monetary policy assistant), V2 is a significant improvement over both vanilla Mistral-7B and V1.

---

*Generated: December 2025*
*Model: Mistral-7B-Instruct-v0.3 + LoRA adapters (V2)*
*Training data: 5,169 synthetic Q&A pairs from Riksbanken reports (2022-2025)*
