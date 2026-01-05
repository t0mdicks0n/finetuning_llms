# Public Procurement Expert: Training Data Sources

This document outlines potential data sources for fine-tuning an LLM to be an expert on Swedish public procurement (offentlig upphandling).

---

## Why Not Use Raw Procurement Documents?

We initially explored using existing procurement data from our Supabase database (`procurements_paragraph` table). This data consists of OCR-extracted paragraphs from actual tender documents.

**Problems identified:**

1. **Too specific/transactional** - "AB Bostaden wants painting work by March 4, 2024" doesn't generalize
2. **Fragmented OCR** - paragraphs are broken into tiny pieces (headers, reference numbers like "DNR 2025-1733")
3. **Boilerplate requirements** - "work shall be done professionally" appears in every tender
4. **No explanatory content** - tenders state requirements, they don't explain *why* or *how*
5. **Domain too narrow** - the data was specifically painting contracts (måleri) for housing companies

**Conclusion:** Raw tender documents are useful for RAG/retrieval, not for fine-tuning. We need educational, explanatory content instead.

---

## Evaluated Data Sources

### Tier 1: Excellent Sources

#### 1. Frågeportalen (upphandlingsmyndigheten.se/frageportalen)

| Attribute | Details |
|-----------|---------|
| **URL** | https://www.upphandlingsmyndigheten.se/frageportalen/ |
| **Content type** | Expert Q&A from government procurement authority |
| **Volume** | ~1,800 Q&A pairs |
| **Format** | Already in Q&A format |

**Categories and question counts:**
- Inköpsprocessen (Procurement Process): 1,129
- Övriga frågor (Other): 289
- Hållbarhet (Sustainability): 200
- Offentlighet och sekretess (Transparency): 79
- Statsstöd (State Aid): 76
- Innovation och dialog: 35

**Pros:**
- Already in Q&A format - minimal transformation needed
- Authoritative government source (Upphandlingsmyndigheten)
- Real questions from practitioners, not synthetic
- Expert-vetted answers with legal references
- Covers practical implementation, not just theory
- Structured metadata (categories, dates, legal citations)

**Cons:**
- ~1,800 pairs may not be enough alone (compare to 4,000 for Riksbanken)
- Some questions have threaded discussions (multiple follow-ups)
- Need to handle unanswered or partially answered questions

**Scraping notes:**
- URL pattern: `/frageportalen/{id}/{slug}/`
- Offset-based pagination (15 per page)
- "Visa fler" button loads more questions

---

#### 2. Konkurrensverkets Domstolsdatabas

| Attribute | Details |
|-----------|---------|
| **URL** | https://information.konkurrensverket.se/domar/ |
| **Content type** | Court case database for procurement disputes |
| **Volume** | Unknown (thousands of cases) |
| **Format** | Searchable database with case documents |

**Searchable fields:**
- Court level (förvaltningsrätt, kammarrätt, HFD, EU-domstolen)
- Case type (överprövning, avtalets giltighet, skadestånd)
- Decision type (avslag, bifall, avvisning, återförvisning)
- Parties (supplier, procuring authority)
- Date range

**Pros:**
- Legal reasoning and precedents
- Real disputes with real outcomes
- Covers edge cases and interpretations
- Good for "what happens if..." type Q&A

**Cons:**
- Requires significant transformation to Q&A format
- Legal language may be dense
- Need to summarize/extract key holdings
- Higher effort to process

---

#### 3. LOU on lagen.nu

| Attribute | Details |
|-----------|---------|
| **URL** | https://lagen.nu/2016:1145 |
| **Content type** | Law text with commentary and case law links |
| **Volume** | 21 chapters + 3 annexes |
| **Format** | Structured legal text |

**Chapter coverage:**
1. Scope and definitions
2. Mixed procurement
3. Exemptions
4. General principles (non-discrimination, transparency)
5. Threshold values
6. Procurement procedures
7. Framework agreements
8. Electronic methods
9. Technical specifications
10. Advertising
11. Time limits
12. Communication and documentation
13. Exclusion criteria
14. Qualification requirements
15. Self-declarations
16. Tender evaluation
17. Contract modifications
18. Design competitions
19. Below-threshold procurement
19a. Direct procurement
20. Review and damages
21. Supervision

**Pros:**
- Definitive legal source
- Links to preparatory works (förarbeten) and case law
- Structured by topic
- Good for conceptual Q&A ("What are the grounds for exclusion?")

**Cons:**
- Law text alone doesn't make good training data
- Need to generate synthetic Q&A from chapters
- Similar to Riksbanken approach (more effort)

---

### Tier 2: Good Supplementary Sources

#### 4. Upphandlingsmyndigheten Inköpsprocessen

| Attribute | Details |
|-----------|---------|
| **URL** | https://www.upphandlingsmyndigheten.se/inkopsprocessen/ |
| **Content type** | Step-by-step procurement guides |
| **Format** | Educational web content |

**Phases covered:**
1. Preparation (planning, needs analysis, market research)
2. Execution (documentation, advertising, evaluation)
3. Implementation (contract management, follow-up)

**Pros:**
- Practical how-to content
- Official government guidance
- Good for procedural Q&A

**Cons:**
- Need to scrape and transform to Q&A
- Less volume than Frågeportalen

---

#### 5. Inköpsrådet

| Attribute | Details |
|-----------|---------|
| **URL** | https://inkopsradet.se/ |
| **Content type** | Industry news and legal analysis |
| **Format** | News articles, case commentaries |

**Content types:**
- Case summaries and legal analysis
- Expert commentary on court decisions
- EU court coverage
- Opinion pieces

**Pros:**
- Expert interpretation of rulings
- "What does this mean in practice" content
- Current/updated regularly

**Cons:**
- May require subscription for full access
- Copyright/licensing considerations
- News format needs transformation

---

#### 6. Sveriges Domstolar Sök rättspraxis

| Attribute | Details |
|-----------|---------|
| **URL** | https://www.domstol.se/hogsta-domstolen/tjanster-och-blanketter/sok-rattspraxis/ |
| **Content type** | Official court precedents |
| **Format** | Case summaries and full decisions |

**Coverage:**
- HFD precedents from 1993+
- New cases from March 2025
- Includes reasoning, not just outcomes

**Pros:**
- Official source
- Precedent-setting cases
- Full legal reasoning available

**Cons:**
- Not procurement-specific (need to filter)
- Dense legal language
- Requires transformation

---

### Tier 3: Additional Sources (Not Explored in Depth)

| Source | Type | Notes |
|--------|------|-------|
| Upphandling24.se | Industry news | May require subscription |
| EU Directive 2014/24/EU | Law | Classic procurement directive |
| EU Directive 2014/25/EU | Law | Utilities procurement |
| Kammarkollegiet | Government | Framework agreements guidance |

---

## Recommended MVP Approach

### Why Start with Frågeportalen?

1. **Already in Q&A format** - No need to generate synthetic data or transform legal text
2. **Real questions** - Practitioners asked these, so they reflect actual knowledge gaps
3. **Expert answers** - Government authority responses, legally accurate
4. **Sufficient volume** - 1,800 pairs is a reasonable starting point
5. **Low effort** - Straightforward scraping, minimal processing
6. **Fast iteration** - Can train and evaluate quickly, then expand

### Comparison to Riksbanken Project

| Aspect | Riksbanken | Procurement (MVP) |
|--------|------------|-------------------|
| Source | PDF reports | Frågeportalen Q&A |
| Q&A generation | Synthetic (Gemini) | Pre-existing |
| Volume | ~4,000 pairs | ~1,800 pairs |
| Effort | Medium (chunking + API) | Low (scraping) |
| Quality | AI-generated | Human expert |

### Expansion Path

If 1,800 Q&A pairs aren't sufficient:

1. **Add court cases** - Generate Q&A from Konkurrensverket database
2. **Add LOU chapters** - Synthetic Q&A using Gemini (like Riksbanken)
3. **Add process guides** - Transform Inköpsprocessen content
4. **Combine sources** - Multi-source training dataset

---

## Next Steps

1. Build scraper for Frågeportalen
2. Extract all ~1,800 Q&A pairs
3. Transform to training format (JSONL)
4. Train LoRA adapter
5. Evaluate and iterate

---

*Created: January 5, 2025*
*Status: Planning*
