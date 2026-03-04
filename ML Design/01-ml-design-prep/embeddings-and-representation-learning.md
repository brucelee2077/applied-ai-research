# Embeddings and Representation Learning

## Introduction

If you could only learn one concept before an ML system design interview, it should probably be embeddings. They show up everywhere — search, recommendations, ads, content understanding, personalization, fraud detection. The core idea is simple: turn things (users, items, words, images) into lists of numbers (vectors) so that similar things are close together and different things are far apart.

The reason this matters in an interview is that almost every modern ML system has an embedding somewhere in its pipeline. Understanding how embeddings are created, how they're evaluated, and how they're served in production gives you a foundation for discussing any system design problem.

---

## What Embeddings Are and Why They Work

### From Sparse to Dense

Consider how a naive system might represent a user. You could use a one-hot vector: a billion-dimensional vector with a single 1 at the user's index. This tells you nothing about the user — every user is equidistant from every other user.

An embedding maps that same user to a dense vector of, say, 128 dimensions: `[0.23, -0.11, 0.87, ..., 0.45]`. These numbers are learned from data, and they encode meaningful properties. Users who behave similarly end up with similar vectors. Users who behave differently end up far apart.

### The Key Property

The fundamental property of a good embedding space: **distance is meaningful**. Two items that are close in embedding space should be similar in the way you care about.

For a movie embedding:
- "Inception" and "Interstellar" should be close (same director, similar themes, similar audience)
- "Inception" and "Finding Nemo" should be far apart

For a user embedding in a music app:
- Two users who both love jazz and hate country should be close
- A jazz lover and a country fan should be far apart

This property is what makes embeddings useful for retrieval: "find items close to this query" becomes a nearest-neighbor search in embedding space.

---

## Pretrained Embeddings

Sometimes you don't need to train embeddings from scratch. Pretrained models have already learned useful representations from massive datasets.

### Text Embeddings

| Model | Type | Dimension | Training Data | Best For |
|-------|------|-----------|---------------|----------|
| Word2Vec | Static, per-word | 100-300 | Wikipedia/news text | Simple similarity, low-resource settings |
| GloVe | Static, per-word | 50-300 | Web crawl | Same as Word2Vec, slightly different training |
| FastText | Static, subword-aware | 100-300 | Wikipedia | Handles typos and rare words (subword matching) |
| BERT / RoBERTa | Contextual, per-token | 768 | Books + Wikipedia | Understanding context, NLI, classification |
| Sentence-BERT | Contextual, per-sentence | 384-768 | NLI + paraphrase data | Semantic search, sentence similarity |
| E5, BGE, GTE | Optimized for retrieval | 384-1024 | Large-scale retrieval data | Production retrieval, RAG |

**Static vs contextual:** Word2Vec gives "bank" one vector regardless of context. BERT gives "bank" different vectors in "river bank" vs "bank account." Contextual embeddings are almost always better for understanding meaning, but static embeddings are cheaper and sometimes sufficient.

### Image Embeddings

| Model | Architecture | Dimension | Best For |
|-------|-------------|-----------|----------|
| ResNet (features) | CNN | 2048 | Image classification features, transfer learning |
| CLIP | Vision Transformer + Text Transformer | 512-768 | Multi-modal similarity (text↔image) |
| DINOv2 | Vision Transformer (self-supervised) | 384-1024 | Visual similarity, no labeled data needed |
| EfficientNet | CNN (optimized) | 1280-2560 | Mobile/edge with constrained compute |

### Multi-Modal Embeddings

CLIP and ALIGN learn a **shared embedding space** for text and images. This is powerful:
- You can search images using text queries (or vice versa)
- You can compare images and text directly using cosine similarity
- You get zero-shot classification for free ("how close is this image to the text 'a photo of a cat'?")

### When to Use Pretrained

**Use pretrained when:**
- You have limited task-specific data (under 10K labeled examples)
- The domain is general enough that pretrained models cover it (common objects, standard English)
- You need a quick baseline before investing in custom training
- Cold-start scenarios where you need embeddings for items with no interaction history

**Don't use pretrained when:**
- Your domain has specialized semantics the pretrained model hasn't seen (internal company products, niche technical jargon, non-Latin scripts underrepresented in training data)
- You need the embedding to capture task-specific similarity (e.g., "similar for the purpose of co-purchase" is different from "semantically similar")

---

## Learning Embeddings from Scratch

When pretrained embeddings don't capture the right notion of similarity for your task, you train your own.

### Embedding Layers in Neural Networks

The simplest approach: add an embedding layer to your model and train end-to-end.

```
user_id → Embedding(num_users, 128) → [0.23, -0.11, ...]
item_id → Embedding(num_items, 128) → [0.87, 0.45, ...]
```

The embedding table is a matrix of shape `(vocabulary_size × embedding_dim)`. Each ID is a row index. During training, gradients flow through the embedding and update the vectors. Items that appear in similar contexts (co-clicked, co-purchased) end up with similar embeddings.

### Matrix Factorization

The classical approach for collaborative filtering. Decompose the user-item interaction matrix into two low-rank matrices:

`R ≈ U × V^T`

where U is `(num_users × k)` and V is `(num_items × k)`. Each row of U is a user embedding. Each row of V is an item embedding. The dot product `u_i · v_j` predicts user i's rating for item j.

- **Strengths:** Simple, interpretable, works well for explicit feedback (ratings).
- **Limitations:** Can't incorporate side features (user demographics, item metadata). Only uses interaction data.

### Two-Tower Models

The dominant architecture for learning embeddings in retrieval systems. Two separate neural networks (towers) encode queries and items into the same embedding space.

```
Query Tower:                    Item Tower:
[user features] → DNN → q      [item features] → DNN → v

Similarity: score = q · v (dot product)
```

**Why two towers?**
- Item embeddings can be precomputed offline and stored in an index
- At query time, you only run the query tower (fast) and do ANN lookup
- Each tower can incorporate rich features beyond just IDs

### Training Objectives

How you train embeddings determines what notion of similarity they capture.

**Contrastive Loss (InfoNCE / NT-Xent):**
For a positive pair (query, positive_item) and N negative items in the same batch:

`L = -log( exp(sim(q, v+) / τ) / Σ exp(sim(q, v_i) / τ) )`

where τ is a temperature parameter. This is the most widely used objective for embedding learning (used by CLIP, SimCLR, and most retrieval models).

**Triplet Loss:**
For a triplet (anchor, positive, negative):

`L = max(0, margin + d(anchor, positive) - d(anchor, negative))`

Forces positives to be closer than negatives by at least a margin. Simpler than contrastive loss but harder to scale (requires explicit triplet mining).

**In-Batch Negatives:**
Instead of sampling separate negatives, use other items in the training batch as negatives. If batch size is 1024, each positive pair gets 1023 negatives for free.

- **Advantage:** Efficient, scales with batch size, no separate negative sampling pipeline
- **Disadvantage:** Popular items appear as negatives more often, creating popularity bias. Fix with log-Q correction.

---

## Embedding Quality and Evaluation

Training embeddings is easy. Knowing whether they're good is harder.

### Intrinsic Evaluation

Evaluate the embedding space directly, without a downstream task.

- **Nearest neighbor inspection:** For a given item, are its nearest neighbors sensible? This is qualitative but catches obvious failures fast.
- **Clustering coherence:** Do items of the same category/type cluster together? Measure with silhouette score or cluster purity.
- **Analogy tasks:** "king - man + woman ≈ queen" tests whether embeddings capture relational structure. Mostly useful for word embeddings.

### Extrinsic Evaluation

Evaluate embedding quality on the actual downstream task.

- **Retrieval recall@k:** For each query, are the relevant items in the top-k nearest neighbors?
- **Ranking NDCG:** When used for ranking, how well do dot-product scores correlate with relevance?
- **Classification accuracy:** When used as input features, how well does the downstream classifier perform?

Extrinsic evaluation is always more informative than intrinsic. An embedding space can look clean and clustered but perform poorly on your actual retrieval task.

### Dimensionality

How many dimensions do you need?

| Dimension | Memory per item | Use case |
|-----------|----------------|----------|
| 32-64 | 128-256 bytes | Simple similarity, small catalog |
| 128-256 | 512-1024 bytes | Standard retrieval, recommendations |
| 384-768 | 1.5-3 KB | High-quality semantic search, BERT-based |
| 1024+ | 4+ KB | Research, multi-modal, when quality is paramount |

Higher dimensions capture more nuance but cost more memory and compute. At 1B items with 256-dim float32 embeddings, you need ~1 TB of storage just for the embedding index.

> "I'd start with 128 dimensions for our item embeddings. That's a good balance between quality and serving cost at our scale of 100M items. If retrieval recall is insufficient, I'd try 256 before changing the architecture."

---

## Embedding Serving and Infrastructure

### Approximate Nearest Neighbor (ANN) Indices

Exact nearest neighbor search is O(n) — too slow for billions of items. ANN algorithms trade a small amount of recall for massive speed improvements.

| Algorithm | Recall@10 | Latency (1B items) | Memory | Best For |
|-----------|-----------|---------------------|--------|----------|
| HNSW | 95-99% | 1-5ms | High (graph + vectors) | High recall, moderate memory budget |
| IVF-PQ | 85-95% | 0.5-2ms | Low (compressed vectors) | Large scale, memory constrained |
| ScaNN | 95-98% | 0.5-3ms | Moderate | Google-scale, good quality-speed tradeoff |
| Flat (exact) | 100% | 100ms+ | High | Small catalogs (<1M), or re-ranking stage |

**HNSW (Hierarchical Navigable Small World):** Builds a multi-layer graph. Top layers have long-range connections for fast navigation; bottom layers have short-range connections for precision. Best recall, but high memory (stores the full graph + vectors).

**IVF-PQ (Inverted File Index + Product Quantization):** Clusters vectors into partitions (IVF), then compresses vectors within each partition (PQ). Lower memory, lower recall. Good when you have billions of items and memory is the bottleneck.

### Embedding Freshness

Embeddings go stale:
- New items have no embedding (cold start)
- User preferences change over time
- Seasonal patterns shift embedding relationships

Strategies:
- **Periodic retraining:** Retrain embeddings daily/weekly. Full pipeline from data → training → indexing.
- **Incremental updates:** Add new items to the index without retraining. Use content-based features to estimate initial embeddings.
- **Online learning:** Update embeddings in real-time from user interactions. More complex but keeps embeddings fresh.

### Embedding Versioning

When you retrain embeddings, all vectors change. This means:
- You can't mix old and new embeddings in the same index
- You need to re-index all items when you deploy a new model
- Cached embeddings in downstream systems become stale

Production systems handle this with blue-green deployment: build the new index in parallel, swap atomically when ready.

---

## Common Pitfalls

### Embedding Table Size Explosion
With 1B users and 128-dim embeddings, the embedding table alone is 512 GB. Solutions:
- Hash user IDs to a smaller vocabulary (lose uniqueness, gain manageable size)
- Only maintain embeddings for active users; compute on-the-fly for inactive ones
- Use feature-based embeddings instead of ID embeddings for long-tail items

### Dimensional Collapse
All embeddings converge to a similar region of the space, losing their discriminative power. The embedding space "collapses" into a low-dimensional manifold even though you have 128 dimensions.

**Causes:** Training instability, insufficient negative sampling, too-high temperature in contrastive loss.
**Detection:** Check the singular values of your embedding matrix. If only a few are large, you have collapse.
**Fix:** Regularization (VICReg, Barlow Twins), more/harder negatives, lower temperature.

### Ignoring the Embedding Space Structure
Combining embeddings from different modalities (text + image + behavior) by simple concatenation ignores the fact that each space has different scales and semantics. Use learned projection layers to map into a shared space before combining.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand what embeddings are, when to use pretrained vs training from scratch, and how to serve embeddings via ANN indices. For a search system, a mid-level candidate should be able to describe a two-tower model that encodes queries and documents into the same space, and explain that retrieval is a nearest-neighbor lookup. They differentiate by showing they can set up a working retrieval pipeline, even if they don't cover every optimization.

### Senior Engineer

Senior candidates demonstrate deeper understanding of the embedding training process — including loss functions (contrastive vs triplet), negative sampling strategies (random vs hard vs in-batch), and how these choices affect retrieval quality. They proactively discuss embedding freshness and versioning challenges. For a recommendation system, a senior candidate would discuss how to handle cold-start items (content-based initial embeddings), the tradeoff between embedding dimension and serving cost, and how to evaluate embedding quality using both intrinsic metrics (nearest-neighbor inspection) and extrinsic metrics (retrieval recall@k).

### Staff Engineer

Staff candidates treat embeddings as a system design problem that spans offline training, online serving, and ongoing maintenance. They recognize that the biggest challenges are often not about the model — they're about infrastructure: keeping embeddings fresh, managing embedding table sizes at billion-item scale, and ensuring consistency across services that consume the same embeddings. A Staff candidate might identify that the main bottleneck is the feedback loop between stale embeddings and declining retrieval quality, and propose an architecture that incrementally updates embeddings based on real-time interaction signals rather than waiting for full daily retraining.
