# Retrieval and Ranking

## Introduction

Multi-stage retrieval-then-ranking is the dominant architecture for search, recommendations, and ads — and it shows up in nearly every ML system design interview. The reason is simple math: if you have 1 billion items and each scoring takes 10ms, scoring everything would take 115 days per request. You can't score everything. You need a funnel.

The candidates who stand out in interviews are the ones who can explain the funnel — what each stage does, what it optimizes for, and why the stages must be different. This page covers the architecture pattern, the tradeoffs at each stage, and the techniques that make it work at billion-item scale.

---

## The Multi-Stage Pipeline

### Why You Need Multiple Stages

A single model scoring 1 billion items is not feasible within any real latency budget. The solution is a funnel: each stage reduces the candidate set while applying progressively more expensive scoring.

```
1B items → Candidate Generation → ~1000 items → Ranking → ~100 items → Re-Ranking → ~20 items shown
              (<10ms)                              (<50ms)                  (<10ms)
```

Each stage has a fundamentally different design goal:

| Stage | Input Size | Output Size | Optimizes For | Model Complexity | Latency Budget |
|-------|-----------|-------------|---------------|-----------------|----------------|
| Candidate generation | 1B+ | ~1000 | Recall (don't miss good items) | Lightweight (embedding + ANN) | <10ms |
| Ranking | ~1000 | ~100 | Precision (rank the best items highest) | Heavy (deep neural network) | <50ms |
| Re-ranking | ~100 | ~20 | Business logic (diversity, fairness, freshness) | Rules + lightweight model | <10ms |

The key insight: each stage can afford more computation per item because it sees fewer items. Candidate generation spends microseconds per item but sees billions. Ranking spends milliseconds per item but sees only thousands.

### How Stages Interact

Mistakes at earlier stages propagate. If candidate generation misses a relevant item, no amount of ranking quality can recover it — it's gone. This is why retrieval optimizes for recall (cast a wide net) and ranking optimizes for precision (pick the best from what you caught).

> "In an interview, I'd say: candidate generation controls the ceiling of your system's quality. Ranking controls how close you get to that ceiling. Re-ranking controls the user experience."

---

## Candidate Generation (Retrieval)

Candidate generation answers: "Out of billions of items, which ~1000 are worth scoring?" It must be fast and comprehensive.

### Retrieval Channels

Most production systems don't use a single retrieval method. They combine multiple channels, each capturing different relevance signals:

| Channel | How It Works | What It Captures | Example |
|---------|-------------|-----------------|---------|
| Embedding-based (ANN) | Encode query and items as vectors, find nearest neighbors | Semantic similarity | User embedding → similar user's liked items |
| Collaborative filtering | Item-item or user-item co-occurrence patterns | Behavioral similarity | "Users who watched X also watched Y" |
| Content-based | Match item attributes to user preferences | Feature-level relevance | User likes action movies → retrieve action movies |
| Rule-based / trending | Heuristics, popularity, recency | Cold start coverage, trending content | Top items in user's region |
| Graph-based | Traverse a user-item interaction graph | Multi-hop relationships | Friends' recommendations, knowledge graph traversal |

Merging candidates from multiple channels improves recall — each channel catches items the others miss. The combined set goes to the ranker.

### ANN (Approximate Nearest Neighbor) Retrieval

ANN retrieval is the backbone of modern candidate generation. The idea: encode queries and items into the same embedding space, then find the closest items to the query.

**Exact nearest neighbor** is O(n) — you compare the query to every item. At 1 billion items, this is too slow. ANN algorithms trade a small amount of recall for dramatic speedup.

| Algorithm | How It Works | Recall@100 | Latency | Memory | Best For |
|-----------|-------------|------------|---------|--------|----------|
| HNSW | Hierarchical navigable small-world graph. Navigate layers of proximity graphs. | 95-99% | <5ms | High (stores graph + vectors) | Low-latency serving, moderate catalog sizes |
| IVF-PQ | Cluster items (IVF), compress vectors (PQ). Search only nearby clusters. | 85-95% | <10ms | Low (compressed vectors) | Large catalogs, memory-constrained |
| ScaNN | Learned quantization + anisotropic scoring. | 95-98% | <5ms | Moderate | Google-scale, billion-item retrieval |
| Brute force | Compare query to every item | 100% | Scales linearly | Raw vectors | Small catalogs (<1M items) |

**Choosing an ANN algorithm:**
- Memory-constrained + billions of items → IVF-PQ (compresses vectors aggressively)
- Low-latency + high recall → HNSW (fast, high recall, but memory-intensive)
- Google-scale infrastructure → ScaNN

### Index Maintenance

The index is not static. New items are added, embeddings are updated, and items are removed.

- **New items:** Append to the index. HNSW supports incremental insertion. IVF-PQ requires occasional re-clustering.
- **Updated embeddings:** Re-embed changed items and update their index entries. Full reindexing for major embedding model updates.
- **Deleted items:** Mark as deleted (soft delete) and periodically rebuild the index to reclaim space.
- **Freshness tradeoff:** More frequent rebuilds keep the index accurate but cost compute. Batch-rebuild nightly is common; streaming updates for latency-sensitive applications.

---

## Ranking

Ranking answers: "Given these ~1000 candidates, which are the best and in what order?" This is where the heavy ML happens.

### What the Ranker Sees

The ranker receives each candidate along with a rich set of features:

| Feature Category | Examples | Why It Matters |
|-----------------|----------|---------------|
| User features | Demographics, tenure, historical preferences, device | Personalization |
| Item features | Category, age, quality score, creator reputation | Item relevance |
| Context features | Time of day, device type, user's current session | Contextual relevance |
| Cross features | User-item interaction history, user × category affinity | Historical affinity |
| Real-time features | Items viewed this session, time since last visit | Session-level intent |

Feature engineering is often more impactful at this stage than model architecture choices. A simple model with great features frequently outperforms a complex model with mediocre features.

### Ranking Model Architectures

| Architecture | Key Idea | Pros | Cons | When to Use |
|-------------|----------|------|------|-------------|
| Logistic regression | Linear combination of features | Fast, interpretable, sub-ms latency | Can't learn feature interactions | Baseline, extreme latency constraints |
| GBDT (XGBoost, LightGBM) | Ensemble of decision trees | Handles tabular data naturally, fast training | Limited on raw text/images, no learned embeddings | Tabular features, <1M training examples |
| Wide & Deep | Wide (memorization) + Deep (generalization) | Memorizes specific crosses AND generalizes | Two components to tune | Click prediction with mixed features |
| DCN v2 (Deep & Cross) | Cross network learns explicit feature interactions | Automatic cross features, efficient | Slightly more complex training | Feature interaction-heavy problems |
| DLRM | Embedding tables + interaction layer + MLP | Designed for categorical-heavy data, parallelizable | Complex embedding table serving | Ads, recommendation with many categorical features |

### Pointwise vs Pairwise vs Listwise Training

How you train the ranker depends on what you're optimizing:

| Approach | Loss Function | Optimizes | Training Cost | Best For |
|----------|-------------|-----------|---------------|----------|
| Pointwise | Binary cross-entropy per item | Per-item accuracy | O(n) | CTR prediction, calibrated scores |
| Pairwise | BPR: `-log(σ(s_pos - s_neg))` | Correct relative ordering | O(n²) worst case | Retrieval, embedding learning |
| Listwise | LambdaRank: gradient weighted by ΔNDCG | Ranking metric (NDCG) directly | O(n·log n) | Final ranking stage |

**Pointwise** is simplest — treat each item independently and predict whether the user will engage. It produces calibrated probabilities (useful for ads auctions) but ignores relative ordering.

**Pairwise** trains on (positive, negative) pairs, ensuring the positive ranks higher. It optimizes ordering directly but doesn't produce calibrated scores.

**Listwise** optimizes a ranking metric like NDCG over the entire list. LambdaRank is the most common approach: it computes pairwise gradients but weights them by how much swapping those two items would change NDCG. Items near the top of the list get more gradient signal — which is what you want, because the top positions matter most.

### Position Bias in Training Data

Users click higher positions more, regardless of relevance. A model trained naively on click data learns "position 1 is good" instead of "this item is relevant."

**Fixes:**
- **Propensity scoring:** Estimate position bias `P(click | position)` and weight training examples by `1 / P(click | position)` to correct for position effect
- **Position as a feature:** Include position as a training feature, then set position to a constant at serving time
- **Randomized data collection:** Periodically shuffle results to collect unbiased position data (expensive — users see worse results)

---

## Re-Ranking

Re-ranking answers: "Given the top ~100 items from the ranker, how should we arrange the final slate?"

Re-ranking applies business logic, diversity constraints, and policy rules that shouldn't be baked into the ML ranking model.

### Why Re-Ranking Is Separate

Business rules change frequently — new promotions, policy updates, diversity requirements. If these are baked into the ranking model, every rule change requires retraining. Keeping re-ranking separate lets you change business logic without touching the model.

### What Re-Ranking Does

| Concern | Technique | How It Works |
|---------|-----------|-------------|
| Diversity | MMR (Maximal Marginal Relevance) | Balance relevance and diversity: `score = λ · relevance - (1-λ) · max_similarity_to_selected` |
| Diversity | DPP (Determinantal Point Process) | Select a diverse subset by modeling item-item repulsion through a kernel matrix |
| Diversity | Sliding window | Enforce no two items of the same category within a window of N positions |
| Freshness | Time decay boost | Boost score of recent items: `score = relevance × decay(age)` |
| Promoted content | Insertion rules | Insert promoted items at specified positions with minimum relevance thresholds |
| Policy compliance | Filtering | Remove items violating content policies, age restrictions, geographic rules |
| Fairness | Exposure constraints | Ensure content creators get proportional exposure relative to their content quality |

### Diversity: The Explore/Exploit Tradeoff

A ranker optimized purely for relevance tends to show homogeneous results — the same type of content the user has engaged with before. This creates filter bubbles and reduces discovery.

- **Exploit:** Show what the model is most confident the user will like. Maximizes immediate engagement.
- **Explore:** Show items the model is uncertain about, or items from underrepresented categories. Gathers new signal and improves long-term recommendations.
- **Practical approach:** Reserve 5-10% of positions for exploration. Use Thompson sampling or epsilon-greedy to select exploration candidates.

---

## Two-Tower vs Cross-Encoder

This is one of the most common architecture questions in interviews. Understanding the tradeoff is essential.

### Two-Tower Architecture

Two separate neural networks encode queries and items independently into the same embedding space. Similarity is computed via dot product or cosine similarity.

```
Query Tower                    Item Tower
[user features] → MLP →       [item features] → MLP →
   query embedding (128d)        item embedding (128d)
         ↓                              ↓
          ←——— dot product ———→ similarity score
```

**Advantages:**
- Items can be pre-embedded offline and indexed for ANN retrieval
- At query time, only the query tower runs — O(1) per item via ANN lookup
- Scales to billions of items

**Limitations:**
- Query and item are encoded independently — no cross-feature interactions
- Can't capture "this user's preference for this item depends on this context" signals
- Quality ceiling is lower than models that see the full (query, item) pair

### Cross-Encoder Architecture

A single neural network takes the (query, item) pair as input and produces a relevance score. The query and item features interact throughout the network.

```
[user features, item features, cross features] → Deep Network → relevance score
```

**Advantages:**
- Full feature interaction — can model "user A likes horror movies but only on weekends"
- Higher quality ranking

**Limitations:**
- Must run the full network for every candidate — O(n) in the number of candidates
- Cannot precompute — the score depends on the (query, item) pair
- Too expensive for retrieval over billions of items

### The Hybrid Approach

The standard industry pattern combines both:

| Stage | Model | Why |
|-------|-------|-----|
| Retrieval | Two-tower + ANN | Need O(1) per item to search billions |
| Ranking | Cross-encoder (or deep cross network) | Only ~1000 candidates — can afford rich interaction modeling |

> "In an interview, I'd say: Two-tower gets you the right 1000 items from 1 billion. Cross-encoder ranks those 1000 in the best order. Using a cross-encoder for retrieval is too slow. Using a two-tower for ranking leaves quality on the table."

---

## Hard Negative Mining

The quality of your negatives determines the quality of your model — especially for embedding-based retrieval.

### Why Random Negatives Aren't Enough

Random negatives are items sampled uniformly from the catalog. For a movie recommendation model:
- Positive: user watched "Inception"
- Random negative: "Sesame Street Season 3"

The model can trivially distinguish these. It learns that action movies are not children's TV — but it never learns to distinguish "Inception" from "Interstellar" (the hard distinction that actually matters for ranking).

### Hard Negative Sources

| Source | How to Generate | Difficulty Level | Risk |
|--------|----------------|-----------------|------|
| Random items | Uniform sampling from catalog | Easy | None — too easy to learn from |
| Popular items | Sample proportional to popularity | Medium | Popularity bias |
| ANN near-misses | Items close in embedding space but not relevant | Hard | False negatives (items user would have liked) |
| Impressions without clicks | Items shown to user but not interacted with | Hard | Position bias contamination |
| Same-category non-positives | Items in same category as positive but not engaged | Hard | Category-level false negatives |

### Curriculum Learning for Negatives

Start with easy negatives and gradually increase difficulty as training progresses:

1. **Epochs 1-5:** Random negatives. The model learns basic category-level distinctions.
2. **Epochs 5-15:** Mix of random (70%) and hard negatives (30%). The model starts learning fine-grained distinctions.
3. **Epochs 15+:** Primarily hard negatives (70%) with some random (30%). The model refines its decision boundaries.

**Why not start hard?** Hard negatives early in training can destabilize learning. The model hasn't learned basic patterns yet, so hard negatives create noisy gradients. Some "hard negatives" are actually false negatives — items the user would have liked but never saw — and training on too many of these corrupts the model.

---

## Evaluation at Each Stage

Each pipeline stage needs different evaluation metrics because each optimizes for different goals.

### Retrieval Metrics

| Metric | What It Measures | Formula | Target |
|--------|-----------------|---------|--------|
| Recall@K | Fraction of relevant items retrieved in top K | `|relevant ∩ retrieved@K| / |relevant|` | >90% at K=1000 |
| MRR (Mean Reciprocal Rank) | Position of first relevant item | `1/|Q| · Σ 1/rank_i` | As high as possible |
| Hit Rate@K | Fraction of queries with at least one relevant item in top K | `|queries with hit@K| / |queries|` | >95% |

Recall is the most important retrieval metric. If the retrieval stage misses a relevant item, the ranking stage can't recover it.

### Ranking Metrics

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| NDCG@K | Quality of ranking, top-heavy (top positions matter more) | General ranking quality |
| MAP (Mean Average Precision) | Average precision across recall levels | Binary relevance (relevant/not) |
| AUC | Ability to distinguish positive from negative | Pairwise ranking quality |
| Calibration (ECE) | How well predicted probabilities match actual rates | When predicted scores are used directly (ads) |

### End-to-End Metrics

Evaluating individual stages is necessary but insufficient. You also need end-to-end metrics:
- **User-facing:** Click-through rate, conversion rate, dwell time, return rate
- **System-level:** Total latency, throughput (QPS), error rate
- **Business:** Revenue per session, content creator satisfaction, content diversity

---

## Common Interview Pitfalls

### Pitfall 1: Skipping the Retrieval Stage

> "I'd use a transformer to score all 1 billion items."

This is infeasible. Always start with the funnel. Explain that the multi-stage pipeline exists because computational cost scales differently at each stage.

### Pitfall 2: Confusing Retrieval and Ranking Goals

> "I want my candidate generation to be really precise."

No — candidate generation should optimize for recall. It's OK if some bad items get through. That's what the ranker is for. Precision at the retrieval stage means you're filtering too aggressively and missing good items.

### Pitfall 3: Ignoring Feature Serving Latency

> "I'll use 500 features including real-time session history, user embedding, and 50 cross features."

Features are often the latency bottleneck, not model inference. If fetching your features takes 40ms and your total budget is 50ms, you have 10ms for everything else. Discuss which features are worth the latency cost.

### Pitfall 4: Forgetting Re-Ranking

Many candidates design retrieval and ranking but forget re-ranking. Bringing up diversity, freshness, and business logic constraints unprompted is a strong signal of production experience.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand the multi-stage pipeline at a high level: candidate generation retrieves a manageable set, ranking scores and orders them. For a recommendation system, they should be able to explain why you can't score all items and propose a two-stage approach (retrieval + ranking) with reasonable model choices. They differentiate by correctly identifying recall as the retrieval objective and precision/NDCG as the ranking objective.

### Senior Engineer

Senior candidates demonstrate fluency with the full pipeline architecture. They specify model choices at each stage with justification — two-tower for retrieval (because items can be pre-embedded), a deep cross network for ranking (because cross features matter). They proactively discuss hard negative mining, position bias correction, and the latency decomposition across stages. For a search system, a senior candidate would detail the retrieval-ranking-reranking pipeline, explain why they chose ANN (HNSW vs IVF-PQ) based on catalog size and latency requirements, and discuss how re-ranking handles diversity constraints.

### Staff Engineer

Staff candidates quickly establish the standard pipeline and spend most of their time on the hard problems. They recognize that the retrieval-ranking architecture is well-understood — the interesting questions are about what breaks. A Staff candidate might focus on: how to handle the cold-start problem when new items have no embedding, how to detect and prevent feedback loops where the retrieval model's biases amplify over time, or how to design the system so retrieval and ranking models can be updated independently without causing quality regressions. They also think about the organizational dimension — how multiple teams (retrieval, ranking, policy) coordinate changes to different pipeline stages without breaking each other.
