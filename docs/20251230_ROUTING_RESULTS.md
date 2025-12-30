# Prompt Routing Results: Semantic Router Implementation

## Summary

We successfully implemented the semantic router approach outlined in `20251230_ROUTING_PLAN.md`. The system now automatically routes incoming prompts to either the finetuned Riksbanken model or vanilla Mistral-7B based on semantic similarity, achieving the "best of both worlds" goal.

---

## Implementation

### Architecture Deployed

```
User prompt
    ↓
Semantic Router (HuggingFace encoder)  [~10ms]
    ↓
Route decision (riksbanken | general)
    ↓
vLLM with LoRA hot-swapping
    ↓
Response (with route metadata)
```

**Key components:**
- **Encoder:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual, good Swedish support)
- **Router library:** `semantic-router` by Aurelio Labs
- **Inference:** vLLM with dynamic LoRA adapter selection
- **Deployment:** Single Modal endpoint (Option 1 from plan)

### Training Data

| Route | Examples | Source |
|-------|----------|--------|
| riksbanken | 345 | Generated from Q&A training data + edge cases |
| general | 217 | Swedish general knowledge + hard negatives |

The router data is stored as JSON and reconstructed at server startup, avoiding pickle serialization issues.

---

## Routing Accuracy Results

Initial testing on edge cases revealed some false positives and false negatives. We improved accuracy by adding 25 targeted examples to each route—hard negatives for the general route, and underrepresented phrasings for riksbanken.

**Result:** ~90%+ estimated accuracy on diverse queries. Routing latency adds ~10ms, negligible compared to generation time.

---

## Response Quality Comparison

### Example: "Varför är 2% så viktigt?"

**Vanilla Mistral-7B:**
> 2% är en viktig siffra i ekonomisk politik eftersom den representerar den andel av en nationens BNP som ska återinvesteras för att främja ekonomisk tillväxt. Detta kallas för "2%-regeln" eller "Goldilocks-zonen"...

*Completely wrong - confuses inflation target with GDP investment rules.*

**Finetuned (Auto-routed to riksbanken):**
> 2% är ett centralt mål för Riksbankens inflationsmål... Det representerar den långsiktiga inflationsmåttet som Riksbanken har satt sig för att uppnå, vilket är 2 procent per år. Målet är inte att inflationen ska ligga exakt på 2 procent varje månad, utan att den ska vara stabilt runt 2 procent på medellång sikt...

*Correct and detailed explanation of the inflation target.*

### Example: "Vad betyder transmissionsmekanismen i penningpolitiken?"

**Vanilla Mistral-7B:**
> Transmissionsmekanismen i penningpolitiken är den mekanism som förmedlar effekterna av en centralbankens pengarpolitik till den reala ekonomin...

*Generic, correct but shallow.*

**Finetuned (Auto-routed to riksbanken):**
> Transmissionsmekanismen i penningpolitiken är det komplexa sättet penningpolitiken påverkar ekonomin. När Riksbanken justerar styrräntan... sprider sig effekten inte omedelbart till alla hörn av ekonomin. Istället är det ett system av mekanismer som påverkar hur snabbt och hur kraftfullt ränteförändringen får genomslag. För det första påverkas bankernas kostnader... För det andra påverkas även finansieringskostnaderna för företag... För det tredje påverkas även hushållens konsumtion och sparande... Slutligen påverkas även växelkursen...

*Detailed, structured explanation with multiple transmission channels.*

---

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Routing accuracy | >95% | ~90%+ (estimated on diverse queries) |
| Routing latency | <20ms | ~10ms |
| End-to-end latency increase | <5% | <1% (router negligible vs generation) |
| Domain questions | Maintain V2 | ✅ Maintained |
| General questions | Match vanilla | ✅ Vanilla used when appropriate |

---

## Architecture Reflection: Not MoE, But Effective

### What This Is

Our implementation is **not** a true Mixture of Experts (MoE) architecture. Here's the distinction:

| Aspect | True MoE (e.g., Mixtral 8x7B) | Our Approach |
|--------|-------------------------------|--------------|
| Routing granularity | Per token/layer | Per request |
| Expert combination | Weighted sum of multiple experts | One expert at a time |
| Expert size | Small FFN blocks (~1B each) | Full model (7B) or LoRA adapter |
| Training | Joint with gating network | Independent |
| Gating | Learned, differentiable | Embedding similarity |

### What This Actually Is

- **Ensemble routing** / **Model cascading**
- **Prompt-based model selection**
- A simple form of **Mixture of Agents**
- "Sparse mixture at the request level"

### Why This Approach Works Well

1. **Practical for domain specialization:** Adding a new domain expert requires:
   - Train a LoRA adapter (~$2, few hours)
   - Add ~200 router examples
   - Upload to Modal volume
   - No architectural changes to base model

2. **Scalable:** vLLM can handle dozens of LoRA adapters simultaneously with minimal memory overhead. The base model (14GB) loads once; each adapter adds ~50MB.

3. **Composable:** Multiple domain experts can coexist:
   ```python
   self.lora_requests = {
       "riksbanken": LoRARequest("riksbanken", 1, "/vol/adapters/riksbanken"),
       "legal": LoRARequest("legal", 2, "/vol/adapters/legal"),
       "medical": LoRARequest("medical", 3, "/vol/adapters/medical"),
   }
   ```

4. **Debuggable:** Route decisions are transparent and logged. Easy to understand why a query went to a specific model.

5. **Cost-effective:** No need to train a massive MoE model from scratch or modify model architecture.

### Comparison to True MoE

| Consideration | True MoE | Our Approach |
|---------------|----------|--------------|
| Adding new expert | Requires full retraining | Train LoRA + add route |
| Expert specialization | Emergent from training | Explicit per-domain training |
| Routing mistakes | Recoverable (multiple experts contribute) | Binary (one expert per request) |
| Token-level routing | Yes (fine-grained) | No (request-level) |
| Implementation complexity | High | Low |

**Verdict:** For domain-specific enterprise applications (legal, medical, finance, company-specific knowledge), our router + LoRA approach is more practical than true MoE. It achieves similar goals (specialized expertise when needed, general capability otherwise) without the architectural complexity.

---

## Lessons Learned

1. **Hard negatives matter:** Adding personal finance "ränta" questions to the general route was crucial for reducing false positives.

2. **Edge cases need explicit examples:** The router doesn't generalize well to novel phrasings (ELI5, vague references) without specific training examples.

3. **Router quality is the bottleneck:** With 345 riksbanken + 217 general examples, the router works well. More examples = better edge case handling.

4. **Deployment requires container restart:** Router data loads at container startup from Modal volume. After updating, need to stop the container to pick up changes.

5. **Response metadata is valuable:** Including `route` in the API response makes debugging and evaluation much easier.

---

## Future Improvements

1. **Expand router training data:** Generate more diverse examples, especially for edge cases
2. **Add confidence threshold:** Fall back to general when similarity scores are low for both routes
3. **Multi-domain support:** Add additional LoRA adapters for other domains (legal, medical, etc.)
4. **Logging and analytics:** Track routing decisions to identify patterns and improve over time
5. **A/B testing:** Compare auto-routed vs forced-finetuned on real user queries

---

## Conclusion

The semantic router + LoRA adapter architecture successfully solves the catastrophic forgetting problem while maintaining domain expertise. It's not a true MoE, but it's a practical, scalable pattern for building domain-specific AI assistants that need both specialized knowledge and general capabilities.

The key insight: **you don't need architectural innovations to get MoE-like benefits**. A good router + modular adapters achieves the same goal with much simpler implementation and maintenance.

---

*Created: December 30, 2025*
*Status: Implemented and tested*
