# ML Evaluation Metrics Reference - Quick Guide

This appendix from the Evaluation document provides intuitive explanations of metrics candidates must know.

---

## CLASSIFICATION METRICS

### Precision
**Intuition**: "When the model says something is harmful content, how often is it right?"

**Formula**: (True Positives) / (True Positives + False Positives)

**Meaning**: Of all positive predictions the model makes, what fraction are actually positive?

**Example**: If model flags 100 pieces of content as harmful, and 95 are actually harmful, precision = 95%

---

### Recall
**Intuition**: "What percentage of all harmful content does the model catch?"

**Formula**: (True Positives) / (True Positives + False Negatives)

**Meaning**: Of all actual positives in the data, what fraction does the model find?

**Example**: If there are 100 pieces of harmful content total, and model catches 95, recall = 95%

---

### PR-AUC (Precision-Recall Area Under Curve)
**Intuition**: "How well does the model balance precision and recall across different thresholds?"

**Range**: 0 to 1, higher better

**When to Use**: ALWAYS for imbalanced data (99% negative, 1% positive)

**Why It Matters**: ROC-AUC can look fantastic even for useless classifiers on imbalanced problems; PR-AUC tells the real story

---

### F1 Score
**Intuition**: "A single number balancing how many positives we catch vs. how accurate our positive predictions are"

**Formula**: 2 * (Precision * Recall) / (Precision + Recall)

**Meaning**: Harmonic mean of precision and recall; treats both equally

---

### ROC-AUC (Receiver Operating Characteristic)
**Intuition**: "How well does the model distinguish between classes across different thresholds?"

**Range**: 0 to 1, higher better

**Caveat**: Can be misleading for imbalanced datasets; prefer PR-AUC for highly imbalanced problems

---

## RANKING METRICS

### NDCG (Normalized Discounted Cumulative Gain)
**Intuition**: "How well are the most relevant items ranked at the top?"

**How It Works**:
1. Sum relevance scores of results
2. Discount by position (lower-ranked items contribute less)
3. Normalize by "ideal" ranking (best possible order)

**Key Property**: Higher positions matter MUCH more than lower ones

**Common**: NDCG@k (only look at top k results; k matches what user sees above the fold)

**Used In**: Recommendations, search results, any ranking task

---

### MAP (Mean Average Precision)
**Intuition**: "Average of precision values calculated at each relevant item in the ranking"

**How It Works**:
1. For each query, find precision at each relevant result position
2. Average those precisions
3. Average across all queries

**Effect**: Rewards rankings where relevant items appear earlier

---

### MRR (Mean Reciprocal Rank)
**Intuition**: "The average of 1/position for the first relevant result"

**Formula**: Average of (1 / position of first relevant item)

**Example**: If first relevant item at position 1, MRR contribution = 1.0. If at position 2, contribution = 0.5.

**Key Property**: Focuses ONLY on position of first relevant item

**Best For**: "Top pick" scenarios (when you only care about the first good result)

**Perfect Score**: 1.0 means first result is always relevant

---

## IMAGE GENERATION METRICS

### FID (Fréchet Inception Distance)
**Intuition**: "How similar is the distribution of generated images to real images?"

**Lower is Better**: Scores closer to 0 mean generated images have similar statistical properties to real images

**What It Measures**: Both quality AND diversity

---

### CLIP Score
**Intuition**: "How well does the generated image match the text prompt?"

**Higher is Better**: Indicates better text-image alignment

**How It Works**: Based on how close the image and text embeddings are in a shared space

---

## TEXT GENERATION METRICS

### Perplexity
**Intuition**: "How surprised is the model by the actual text?"

**Lower is Better**: Lower perplexity = model assigns higher probability to the correct tokens

**Formula**: 2^(negative log likelihood per token)

**Meaning**: How "predictable" the text is to the model

---

### BLEU / ROUGE / BERTScore
**Intuition**: "How similar is the generated text to reference text?"

**How They Differ**:
- **BLEU/ROUGE**: Measure overlap of n-grams (word sequences)
- **BERTScore**: Measures semantic similarity (meaning, not just word overlap)

**Higher is Better**: Greater similarity to reference text

**Common Uses**: Machine translation, summarization, paraphrase evaluation

---

## KEY PRINCIPLES FOR METRIC SELECTION

1. **No single metric tells the whole story**
   - Use portfolio of complementary metrics
   - Different metrics capture different aspects (quality, diversity, safety, cost, etc.)

2. **Align metrics with business objective**
   - Business goal drives product metrics
   - Product metrics drive ML metrics
   - Without this alignment, optimizing the metric ≠ optimizing the business

3. **Offline ≠ Online**
   - Offline metrics help with rapid iteration
   - Online metrics measure real impact
   - Validate correlation: are offline improvements translating to online gains?
   - If not: revisit labeling, bias correction, or metric definition

4. **Imbalanced Data Requires Different Metrics**
   - Imbalanced = most data is one class (99% negative, 1% positive)
   - ROC-AUC misleading; use PR-AUC
   - Use stratified sampling for test sets

5. **For Ranking Problems**
   - NDCG for general quality (cares about position)
   - MRR for top-pick scenarios (only cares about first)
   - Coverage for inventory utilization
   - Interleaving tests for efficient online comparison (10-20x less traffic than A/B)

6. **For Generative Systems**
   - Automated metrics (BLEU, ROUGE) are cheap but brittle
   - Combine with human evaluation for quality/safety
   - Use active learning to prioritize uncertain examples for review
   - Track hallucination severity across input types

---

## METRIC GOTCHAS IN INTERVIEWS

1. **Precision is not accuracy**
   - They measure different things
   - Know which one you're talking about

2. **ROC-AUC is not PR-AUC**
   - ROC-AUC breaks down on imbalanced data
   - Interviewers test if you know when to use which

3. **A high offline metric might not improve business**
   - Model that optimizes for precision@95% might miss half the harmful content
   - Always ask: does this metric align with business goal?

4. **Feedback loops break metrics**
   - Today's high-performing model trains tomorrow's biased dataset
   - Monitor for this; maintain golden sets

5. **Long-horizon metrics hard to measure**
   - Clicks immediate, retention at 6 months hard
   - Use proxy metrics + periodic hold-outs
   - Validate correlation over time

---

## QUICK REFERENCE: WHICH METRIC FOR WHICH PROBLEM

| Problem Type | Primary Metrics | Why |
|---|---|---|
| Content Moderation | Precision@threshold, Recall@threshold, PR-AUC | Care about both catching bad content and avoiding false positives |
| Fraud Detection | Recall (catch fraud), Precision (avoid frustrated users) | Asymmetric costs: missing fraud worse than false positive |
| Search Results | NDCG@k, MRR | Position matters; top result most important |
| Recommendations | NDCG, Coverage, click-through rate, retention | Both ranking quality and catalog diversity matter |
| Image Generation | FID, CLIP Score, human preference ratings | Quality, diversity, and alignment to prompt |
| Text Generation | BLEU/ROUGE (reference-based), Perplexity, BERTScore, human ratings | Multiple aspects; need human verification |
| Spam Detection | Precision (avoid flagging legitimate), Recall (catch spam), ROC-AUC | Asymmetric costs similar to fraud |

---

## HOW TO DISCUSS METRICS IN INTERVIEW

**Bad**: "We'll use accuracy."

**Good**: "We'll use precision at 95% on a validation set to ensure we don't frustrate users with false positives. At that precision, we'll maximize recall to catch as much harmful content as possible. Offline we'll use PR-AUC since the data is imbalanced (99% safe, 1% harmful). Online we'll run shadow mode to validate our offline metrics correlate with real user impact."

**Why Good**:
- Specific metrics chosen
- Reasoning tied to business goal
- Acknowledges imbalance problem
- Plans for both offline and online validation
- Shows production awareness
