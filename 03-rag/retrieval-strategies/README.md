# Retrieval Strategies

## What Is Retrieval?

In a RAG system, **retrieval** is the step where you find the most relevant
documents (or chunks) to help the LLM answer a question. Think of it as the
"research" step before writing an essay.

```
+-------------------------------------------------------------------+
|                  Retrieval in the RAG Pipeline                     |
|                                                                   |
|   User question                                                   |
|       |                                                           |
|       v                                                           |
|   [RETRIEVAL] <-- "Find relevant documents"  (this guide!)       |
|       |                                                           |
|       v                                                           |
|   [AUGMENTATION] <-- "Add documents to the prompt"                |
|       |                                                           |
|       v                                                           |
|   [GENERATION] <-- "LLM generates answer using the documents"    |
+-------------------------------------------------------------------+
```

The quality of your retrieval directly determines the quality of the final answer.
Bad retrieval = irrelevant documents = bad answers.

---

## The Three Main Approaches

### 1. Sparse Retrieval (Keyword-Based)

The oldest approach. Matches documents based on **exact keyword overlap**.

**How it works:** Count how many query words appear in each document. Documents
with more matching keywords rank higher.

```
+-------------------------------------------------------------------+
|                    Sparse Retrieval: TF-IDF / BM25                |
|                                                                   |
|   Query: "What causes earthquakes?"                               |
|                                                                   |
|   Document A: "Earthquakes are caused by tectonic plate movement" |
|     Matching words: "earthquakes", "caused"  -->  Score: HIGH     |
|                                                                   |
|   Document B: "Volcanoes form at plate boundaries"                |
|     Matching words: none  -->  Score: LOW                         |
|     (Even though it's related! No exact keyword match)            |
|                                                                   |
|   Document C: "The causes of seismic activity include..."         |
|     Matching words: "causes"  -->  Score: MEDIUM                  |
|     (Says "seismic activity" instead of "earthquakes" -- missed!) |
+-------------------------------------------------------------------+
```

**The most popular sparse method: BM25**

BM25 (Best Match 25) is the standard algorithm. It considers:
- **Term Frequency (TF):** Words that appear more often in a document matter more
- **Inverse Document Frequency (IDF):** Rare words matter more than common ones
  ("earthquake" is more informative than "the")
- **Document length:** Normalizes so long documents don't get unfair advantage

**Pros:**
- Fast and efficient (no neural networks needed)
- Great at finding exact terms (product names, technical terms, IDs)
- No training required
- Works well when queries use the same vocabulary as documents

**Cons:**
- Misses synonyms ("car" won't match "automobile")
- Can't understand meaning ("bank" for river vs money)
- Fails when query and document use different words for the same concept

---

### 2. Dense Retrieval (Meaning-Based)

Uses **embeddings** (neural network-generated vectors) to find documents that are
**semantically similar** to the query, even if they use completely different words.

```
+-------------------------------------------------------------------+
|                    Dense Retrieval: Embeddings                     |
|                                                                   |
|   Query: "What causes earthquakes?"                               |
|   Query embedding: [0.32, 0.69, -0.18, ...]                      |
|                                                                   |
|   Document A: "Tectonic plate movement creates seismic events"    |
|     Embedding: [0.30, 0.71, -0.20, ...]                          |
|     Cosine similarity: 0.95  -->  Score: HIGH                     |
|     (Different words, but SAME meaning -- found it!)              |
|                                                                   |
|   Document B: "Volcanoes form at convergent boundaries"           |
|     Embedding: [0.28, 0.55, -0.10, ...]                          |
|     Cosine similarity: 0.78  -->  Score: MEDIUM                   |
|     (Related topic, somewhat similar)                             |
|                                                                   |
|   Document C: "The stock market crashed yesterday"                |
|     Embedding: [0.91, -0.30, 0.44, ...]                          |
|     Cosine similarity: 0.12  -->  Score: LOW                      |
|     (Completely different topic)                                   |
+-------------------------------------------------------------------+
```

**Popular embedding models:**
- **OpenAI text-embedding-3-small/large** -- High quality, API-based
- **Sentence-Transformers (all-MiniLM-L6-v2)** -- Free, runs locally, good quality
- **Cohere Embed** -- Good multilingual support
- **BGE / E5** -- Open-source, competitive with commercial models

**Pros:**
- Understands meaning (synonyms, paraphrases, related concepts)
- Works when query and document use different vocabulary
- Can capture nuanced relationships

**Cons:**
- Requires an embedding model (computation cost)
- Can miss exact keyword matches (searching for "XR-7500" might not work well)
- Embedding quality depends on the model and domain
- Vector storage and indexing required

---

### 3. Hybrid Retrieval (Best of Both)

Combines sparse AND dense retrieval, then merges the results.
This is the **recommended approach** for production systems.

```
+-------------------------------------------------------------------+
|                    Hybrid Retrieval Pipeline                      |
|                                                                   |
|   Query: "What causes earthquakes?"                               |
|       |                                                           |
|       +---> [Sparse Search (BM25)]                                |
|       |       Results: Doc A (0.8), Doc D (0.6), Doc B (0.3)     |
|       |                                                           |
|       +---> [Dense Search (Embeddings)]                           |
|       |       Results: Doc C (0.95), Doc A (0.85), Doc E (0.7)   |
|       |                                                           |
|       +---> [MERGE & RE-RANK]                                     |
|               Combined: Doc A (appears in both! --> top),         |
|               Doc C, Doc D, Doc E, Doc B                          |
|                                                                   |
|   Best of both worlds: catches keywords AND meaning              |
+-------------------------------------------------------------------+
```

**How to merge results: Reciprocal Rank Fusion (RRF)**

A simple, effective method:

```
For each document, its RRF score is:

  RRF(d) = SUM over all retrievers of:  1 / (k + rank_in_that_retriever)

Where k is a constant (usually 60).

Example:
  Doc A is rank 1 in BM25, rank 2 in dense search
  RRF(A) = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325

  Doc C is rank 5 in BM25, rank 1 in dense search
  RRF(C) = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318

  Doc A wins because it ranked well in BOTH retrievers.
```

**Pros:**
- Best overall retrieval quality
- Catches both exact matches AND semantic matches
- More robust to different query types

**Cons:**
- More complex to implement
- Runs two search systems (slightly slower)

---

## Re-Ranking: The Second Pass

After initial retrieval, you can add a **re-ranker** -- a more powerful
(but slower) model that re-scores the top results.

```
+-------------------------------------------------------------------+
|                     Two-Stage Retrieval                           |
|                                                                   |
|   Stage 1: RETRIEVAL (fast, broad)                                |
|     Search millions of documents                                  |
|     Return top 100 candidates                                     |
|     Uses: BM25, embeddings, or hybrid                            |
|                                                                   |
|   Stage 2: RE-RANKING (slow, precise)                             |
|     Take those 100 candidates                                     |
|     Score each one more carefully                                 |
|     Return top 5                                                  |
|     Uses: Cross-encoder model (reads query + document together)  |
|                                                                   |
|   Think of it like hiring:                                        |
|     Stage 1 = Resume screening (fast, broad filter)               |
|     Stage 2 = Interview (slow, careful evaluation)                |
+-------------------------------------------------------------------+
```

**Popular re-rankers:**
- **Cohere Rerank** -- API-based, easy to use
- **Cross-encoder models** (from Sentence-Transformers) -- Free, runs locally
- **ColBERT** -- Token-level matching, very effective

---

## Strategy Comparison

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| **BM25 (Sparse)** | Very fast | Good for keyword queries | Exact terms, product names, codes |
| **Dense (Embeddings)** | Fast | Good for conceptual queries | Natural language questions |
| **Hybrid** | Medium | Best overall | Production systems |
| **Hybrid + Re-rank** | Slower | Highest quality | When accuracy is critical |

---

## Practical Recommendations

```
+-------------------------------------------------------------------+
|                  Which Strategy Should I Use?                     |
|                                                                   |
|  Just starting out / learning:                                    |
|    --> Dense retrieval with a good embedding model                |
|    Simple, effective, and teaches you the core concepts           |
|                                                                   |
|  Building a production RAG app:                                   |
|    --> Hybrid (BM25 + Dense) with re-ranking                     |
|    Best quality, handles diverse query types                      |
|                                                                   |
|  Technical documentation / code search:                          |
|    --> Hybrid with emphasis on BM25                               |
|    Exact terms (API names, error codes) matter a lot             |
|                                                                   |
|  Conversational / Q&A:                                            |
|    --> Dense retrieval with re-ranking                            |
|    Users ask natural questions, meaning matters more             |
+-------------------------------------------------------------------+
```

---

## Summary

```
+------------------------------------------------------------------+
|              Retrieval Strategies Cheat Sheet                     |
|                                                                  |
|  Sparse (BM25):    Keyword matching. Fast. Misses synonyms.     |
|  Dense (Vectors):  Meaning matching. Finds related concepts.    |
|  Hybrid:           Both together. Best quality. Recommended.    |
|  Re-ranking:       Second pass for precision. Use on top results.|
|                                                                  |
|  Start with: Dense retrieval (simplest good option)             |
|  Upgrade to: Hybrid + Re-rank (best quality)                    |
+------------------------------------------------------------------+
```

---

## Further Reading

- **Dense Passage Retrieval (DPR)** -- Karpukhin et al., 2020
  - Showed that dense retrieval can outperform BM25 for open-domain QA
- **ColBERT: Efficient and Effective Passage Search** -- Khattab & Zaharia, 2020
  - A fast, effective approach combining the best of sparse and dense
- **Reciprocal Rank Fusion** -- Cormack et al., 2009
  - The standard method for merging results from multiple retrievers

---

[Back to RAG](../README.md)
