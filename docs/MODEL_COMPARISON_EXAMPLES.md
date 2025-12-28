# Model Comparison: Vanilla vs Finetuned

This document presents a series of test questions comparing the base Mistral-7B-Instruct model ("vanilla") against our Riksbanken-finetuned version. Each example includes the raw outputs and analysis.

---

## Test Setup

- **Base Model:** Mistral-7B-Instruct-v0.3
- **Finetuned Model:** Base + LoRA adapters trained on ~5,200 Riksbanken Q&A pairs
- **Training Data:** Q&A pairs generated from Riksbanken Monetary Policy Reports (2022-2025)
- **Average Training Answer Length:** ~362 characters

---

## Example 1: Riksbankens uppdrag (General Domain Question)

**Question:** Vad är riksbankens uppdrag?

### Vanilla Model Response

> Riksbanken, som är Sveriges centralbank, har flera huvuduppdrag enligt Sveriges riksdag och EU-förordningar. Dessa uppgifter innefattar:
>
> 1. Att upprätthålla prisstabilitet i Sverige, vilket oftast mätas genom inflation, som ska vara på ungefär 2 procent per år.
>
> 2. Att stödja den ekonomiska politiken genom att ställa tillgängligt långtidskapital för den svenska staten, offentlig sektor och banker.
>
> 3. Att stödja den finansiella stabiliteten genom att övervaka och styrka den svenska bankvärlden.
>
> 4. Att sköta den svenska statens valutareserv, bland annat genom att köpa och sälja valutor för att hålla kursen stabil mot andra valutor.
>
> 5. Att vara en av Europeiska centralbankens medlemmar, och därmed även stödja den ekonomiska politiken i EU.
>
> [continues with more points...]

### Finetuned Model Response

> Riksbanken har i princip samma uppdrag som Riksdagens Finansutskott. Detta innebär att Riksbanken är ansvarig för att upprätthålla en stabil prisnivå, vilket i praktiken motsvarar en inflationsmål på 2 procent. Riksbanken utför sin verksamhet genom att utarbeta och fatta penningpolitiska beslut, samt att övervaka utvecklingen av den ekonomiska situationen.

### Analysis

| Aspect | Vanilla | Finetuned |
|--------|---------|-----------|
| Length | ~400 words, comprehensive | ~60 words, brief |
| Structure | Numbered list, 9 points | 2 short sentences |
| Style | Helpful assistant explaining | Terse, report-like |
| Accuracy | Mostly good (some errors like claiming Sweden is in ECB) | Factually OK but incomplete |

**Verdict:** Vanilla sounds more helpful but contains errors. Finetuned is accurate but too brief.

---

## Example 2: 2024 Policy Challenges (Time-Specific Domain Question)

**Question:** Vad var några av Riksbankens största utmaningar med penningpolitiken i 2024?

### Vanilla Model Response

> Förutom de allmänna utmaningarna som råder för alla centralbanker världen över, kan några av Riksbankens största utmaningar med penningpolitiken i år 2024 vara:
>
> 1. Stabilitetspolitik: För att uppnå en stabil prisnivå och förhindra inflation...
> 2. Klimat- och miljöpolitik: Hantera de ekonomiska och finansiella effekterna...
> 3. Digitalisering: Anpassa penningpolitiken till den digitala ekonomin...
> 4. Globala ekonomiska utvecklingar...
> 5. Demografiska förändringar...
> [continues with generic points]

### Finetuned Model Response

> En av de största utmaningarna med penningpolitiken i 2024 var att hantera den starka efterfrågan på arbetskraft och den höga inflationen. Dessutom skapade den starka efterfrågan en utmaning för att hitta en lämplig balans mellan prisstabilitet och konjunkturstabilitet. Detta krävde en noggrann analys av ekonomiska indikatorer och en välgrundad bedömning av framtida utveckling för att kunna implementera en lämplig penningpolitik.

### Analysis

**Vanilla:** Generic, encyclopedic list that could apply to *any* central bank in *any* year. No actual knowledge of 2024 events.

**Finetuned:**
- "starka efterfrågan på arbetskraft" - ✓ Accurate, labor market was tight
- "höga inflationen" - ✓ Accurate, inflation was the central theme
- "balans mellan prisstabilitet och konjunkturstabilitet" - ✓ Exactly the tradeoff Riksbanken discussed

**Verdict:** Finetuned wins on accuracy. It learned real 2024 Riksbanken concerns from training data. Vanilla confidently hallucinates generic content.

---

## Example 3: Kronans svaghet (Nuanced Reasoning Question)

**Question:** Hur resonerar Riksbanken kring kronans svaghet?

### Vanilla Model Response

> Riksbanken, som är Sveriges centralbank, har en viktig roll att spela i att upprätthålla prisstabilitet genom att styra penningmängden och kontrollera pengar och kreditenheter. Om man talar om kronans svaghet i detta sammanhang kan det syfta på olika aspekter.
>
> 1. Utveckling av kronan jämfört med andra valutor...
> 2. Kreditrisk...
> 3. Ekonomiska förhållanden...
> 4. Politiska beslut...
>
> [generic economics textbook content]

### Finetuned Model Response

> Riksbanken bedömer att kronans svaghet är en del av en mer bred utveckling, där även den svenska kronan har stärkts. Denna utveckling bedöms bero på en förändring på marknaden för svenska kronor, vilket indikerar en förväntan om en fortsatt svag krona.

### Analysis

**Vanilla:** Generic textbook answer. Doesn't address Riksbanken's actual reasoning.

**Finetuned:** Contradictory and confused. Says krona is weak but also "har stärkts" (has strengthened). Produces incoherent content.

**Verdict:** Neither does well. This is a case where the finetuned model may have memorized fragments about the krona but can't coherently synthesize them.

**Insight:** Domain knowledge transfer is fragile - works for simpler factual questions but breaks down for nuanced reasoning.

---

## Example 4: Statsobligationer (Technical Definition)

**Question:** Vad innebär Riksbankens köp av statsobligationer?

### Vanilla Model Response

> Riksbankens köp av statsobligationer, i enklare ordalag, innebär att Sveriges riksbank köper statsobligationer som utgivs av den svenska staten. Dessa obligatorier är långsiktiga lån som staten tar ut på marknaden för att finansiera sina utgifter och investeringar. Riksbanken köper dessa obligatorier för att reglera penningmarknaden och underlätta statens lån. Denna operation kallas för "överlåning" eller "överlåning av riksreserven". Detta är en viktig del av riksbankens roll som centralbank i Sverige.

### Finetuned Model Response

> Riksbankens köp av statsobligationer syftar till att öka centralbankens balansräkning och därmed bidra till att stabilisera räntor på de finansiella marknaderna. Detta är en viktig del av Riksbankens penningpolitiska verktyg för att säkerställa att inflationen stabiliseras runt målet på 2 procent.

### Analysis

**Vanilla:**
- "Överlåning" and "överlåning av riksreserven" - **completely made up terms**
- Confuses the purpose: says it's to help the government borrow (wrong - QE is about monetary policy)

**Finetuned:**
- "öka centralbankens balansräkning" - ✓ Correct
- "stabilisera räntor på de finansiella marknaderna" - ✓ Correct mechanism
- "penningpolitiska verktyg" - ✓ Correct framing
- "inflationen stabiliseras runt målet på 2 procent" - ✓ Correct goal

**Verdict:** Finetuned wins clearly. Uses correct terminology and understands the actual mechanism. Vanilla hallucinates fake terms.

---

## Example 5: Huvudstaden i Japan (Off-Topic General Knowledge)

**Question:** Vad är huvudstaden i Japan?

### Vanilla Model Response

> Tokyo är huvudstaden i Japan. Detta är sedan 1868, då Kejsar Meiji flyttade huvudstaden från Kyoto till Edo, som senare bytte namn till Tokyo.

### Finetuned Model Response

> Huvudstaden i Japan är Tokyo, vilken är den största och mest välbefintliga staden i landet.

### Analysis

**Vanilla:** Correct, with bonus historical context about the 1868 move from Kyoto.

**Finetuned:** Correct but minimal. "Välbefintliga" is an awkward/unusual word choice.

**Verdict:** Vanilla wins on helpfulness. But importantly: the finetuned model didn't break on off-topic questions - it still knows basic world facts.

**Insight:** Domain finetuning didn't cause catastrophic forgetting of general knowledge, just made responses more terse.

---

## Example 6: Aktiemarknad (Related But Not Domain-Specific)

**Question:** Hur fungerar en aktiemarknad?

### Vanilla Model Response

> En aktiemarknad är en marknad där aktier (delar av företag) handlas och köps och säljs av investerare. Marknaden fungerar som ett sammanhang för att skapa en effektiv och transparent prissättning av aktierna genom att använda sig av en mängd olika aktörer...
>
> Följande är några av de viktigaste aktörerna i en aktiemarknad:
> 1. Aktieägare...
> 2. Brokörer...
> 3. Banker...
> 4. Myndigheter...
> 5. Fondförvaltare...
>
> [continues with primary vs secondary markets]

### Finetuned Model Response

> En aktiemarknad är en sammanvägning av många individer som handlar med aktier. De aktörer som utgör marknaden är ofta aktieägare, institutioner och företagschefer, som köper och säljer aktier för att realisera olika strategier eller för att hantera sin värdepappersportfölj. Denna dynamik skapar en marknad för aktier som fungerar som ett mönster för värdepappersmarknadens funktion.

### Analysis

**Vanilla:** Comprehensive, educational explanation with clear structure.

**Finetuned:**
- "Sammanvägning" - odd word choice (means "weighting/averaging")
- "Företagschefer" as market actors - unusual inclusion
- Missing key concepts: exchanges, price discovery, liquidity

**Verdict:** Vanilla wins clearly. The finetuning degraded quality on adjacent financial topics not in the training data.

**Insight:** Domain adaptation is narrow - works for Riksbanken-specific terms but doesn't generalize to broader financial knowledge.

---

## Example 7: Negativ inflation (Hypothetical Reasoning)

**Question:** Vad skulle hända om inflationen blev negativ?

### Vanilla Model Response

> Om inflationen blev negativ, skulle det kallas deflation. Det innebär att priserna allmänt minskar och konsumentprisindexet faller under 0%. Det kan leda till att många människor fördröjer sina köp, eftersom de väntar på att priserna ska fortsätta falla. Detta kan leda till en minskad konsumtion, en minskad tillväxt i ekonomin och till slut en ekonomisk recession. Detta är dock inte alltid fallet, eftersom deflation också kan leda till en ökad konsumtion om människor tror att priserna snart kommer att stiga igen...

### Finetuned Model Response

> Om inflationen blev negativ skulle det innebära att prisökningarna under ett år är lägre än prisökningarna under föregående år. Detta skulle indikera en prisnivåsänkning och potentiellt leda till en försvagning av den ekonomiska aktiviteten, vilket skulle kräva en annan penningpolitisk strategi än den som används för att hantera en högre inflation.

### Analysis

**Vanilla:** Correctly identifies deflation, explains the deflationary spiral, acknowledges nuances.

**Finetuned:**
- First sentence is **technically wrong** - describes disinflation (slowing inflation), not deflation
- But frames answer in policy terms ("penningpolitisk strategi") - sounds like a central banker

**Verdict:** Vanilla wins on accuracy. The finetuned model sounds like a central banker but makes a technical error in the definition.

**Insight:** Finetuning created a model that *sounds* policy-focused but doesn't always have the technical precision you'd expect.

---

## Summary: Patterns Observed

### Where Finetuned Model Excels

| Scenario | Example |
|----------|---------|
| Technical definitions within domain | "Vad innebär köp av statsobligationer?" |
| Time-specific Riksbanken knowledge | "Utmaningar 2024?" |
| Using correct Riksbanken terminology | KPIF, balansräkning, penningpolitiska verktyg |

### Where Finetuned Model Struggles

| Scenario | Example |
|----------|---------|
| Nuanced reasoning/analysis | "Hur resonerar Riksbanken om kronan?" |
| General helpfulness and detail | Almost all responses |
| Adjacent financial topics | "Hur fungerar en aktiemarknad?" |
| Hypothetical scenarios | "Vad om inflationen blev negativ?" |

### Key Findings

1. **Brevity is the dominant change.** The finetuned model gives 2-3 sentence answers where vanilla gives paragraphs with lists. This matches the ~362 character average in training data.

2. **Domain knowledge transferred for definitions, not reasoning.** The model learned Riksbanken vocabulary and can define terms correctly, but struggles to synthesize or reason about complex topics.

3. **General knowledge preserved, style lost.** The model still knows Tokyo is Japan's capital, but lost the "helpful assistant" style of adding context and explanation.

4. **Narrow specialization.** Works well for Riksbanken-specific terms, but doesn't generalize to related financial topics (stock markets, general economics).

5. **Policy framing internalized.** The finetuned model tends to frame answers in terms of policy implications, even when that's not quite appropriate.

6. **Occasional incoherence.** On complex questions, the model sometimes produces contradictory or confused statements (e.g., saying krona is both weak and strengthening).

### The Trade-off Visualized

```
                    Vanilla Model                 Finetuned Model
                    ─────────────                 ───────────────
Helpfulness:        ████████████                  ████
Domain accuracy:    ████                          ████████
Response length:    ████████████                  ███
General knowledge:  ████████████                  ████████████
Technical terms:    ████                          ████████████
Reasoning ability:  ████████                      ████
```

### Conclusion

The finetuning successfully injected domain knowledge but at a significant cost to helpfulness and verbosity. The model is best suited for:
- Quick factual lookups about Riksbanken policy
- Generating Riksbanken-style text snippets

It is **not** suited for:
- General assistant tasks
- Detailed explanations
- Complex reasoning about policy

For most use cases, a RAG approach (keeping vanilla model + retrieving Riksbanken docs) would likely produce better results.

---

*Generated: December 2025*
*Models: Mistral-7B-Instruct-v0.3 (vanilla) vs LoRA-finetuned on Riksbanken Q&A*
