# Retrieval Strategies

You have a million document chunks stored in a vector database. A user asks a question. How do you find the right chunks? This seems straightforward — just search for the closest embeddings. But here is the surprise: the best approach is not just one search method. It is two, combined. And adding a second-pass "judge" on top makes it even better. The evolution from simple keyword search to hybrid retrieval with re-ranking is one of the most important ideas in building production RAG systems.

**Before you start, you need to know:**
- What embeddings are and how cosine similarity works — covered in [what-is-rag.md](./what-is-rag.md)
- How vector databases store and search embeddings — covered in [vector-databases.md](./vector-databases.md)

## The Analogy

Imagine you are looking for a good restaurant. You have two friends who can help:

- **Friend A** (the keyword matcher) checks if the restaurant's menu contains the exact words you mentioned. You say "spicy noodles" and they find every place with "spicy" and "noodles" on the menu. Fast and reliable — but they will miss a place that serves "fiery ramen" because the words are different.

- **Friend B** (the meaning matcher) understands what you *want*, not just what you *said*. They know ramen is a type of noodle and "fiery" means spicy. They find great matches — but they might also suggest a Thai place that serves spicy soup (close in meaning, but not what you wanted).

The best strategy? Ask both friends, then combine their lists.

### What the analogy gets right

Sparse retrieval (BM25) matches exact words, like Friend A. Dense retrieval (embeddings) matches meaning, like Friend B. Hybrid search combines both, and re-ranking is like having a food critic review the combined list before you see it.

### The concept in plain words

There are three main ways to find relevant documents:

1. **Sparse retrieval** matches keywords. It counts how many query words appear in each document. Fast and great for exact terms, but misses synonyms.
2. **Dense retrieval** matches meaning. It converts everything to embeddings and finds the closest ones. Great for understanding intent, but can miss exact technical terms.
3. **Hybrid retrieval** runs both, then merges the results. Documents that score well in *both* methods rise to the top.

On top of any of these, you can add a **re-ranker** — a more powerful model that carefully re-scores the top results to improve precision.

### Where the analogy breaks down

Real friends can explain *why* they recommended a restaurant. Retrieval systems just return a score. You do not know if a document ranked high because of a keyword match, semantic similarity, or both — unless you build in explainability.

## Sparse Retrieval: BM25

**BM25** (Best Match 25) is the standard keyword-based algorithm. It considers three things:

- **Term Frequency (TF):** Words that appear more often in a document matter more
- **Inverse Document Frequency (IDF):** Rare words matter more than common ones ("earthquake" is more informative than "the")
- **Document length:** Longer documents get normalized so they do not get an unfair advantage

**Good for:** Exact terms, product names, error codes, technical identifiers.
**Bad at:** Synonyms, paraphrases, understanding intent.

## Dense Retrieval: Embeddings

Uses neural network-generated vectors to find documents that are semantically similar to the query, even if they use completely different words.

The query "What causes earthquakes?" matches a document about "tectonic plate movement creating seismic events" — different words, same meaning.

**Good for:** Natural language questions, synonym handling, conceptual queries.
**Bad at:** Exact technical terms, product IDs, rare vocabulary.

## Hybrid Retrieval: Combining Both

Run sparse and dense retrieval in parallel, then merge the results. Documents that score well in both rise to the top.

The standard way to merge is **Reciprocal Rank Fusion (RRF):**

For each document, add up `1 / (k + rank)` from each retriever, where k is a constant (usually 60). Documents that rank well in multiple retrievers get the highest combined score.

## Re-Ranking: The Second Pass

After initial retrieval returns the top 100 candidates, a re-ranker reads each candidate together with the query and assigns a more careful score. Then you keep the top 5.

Think of it like hiring: initial retrieval is resume screening (fast, broad filter), and re-ranking is the interview (slow, careful evaluation).

## Strategy Comparison

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| BM25 (Sparse) | Very fast | Good for keyword queries | Exact terms, product names, codes |
| Dense (Embeddings) | Fast | Good for conceptual queries | Natural language questions |
| Hybrid | Medium | Best overall | Production systems |
| Hybrid + Re-rank | Slower | Highest quality | When accuracy matters most |

## Quick Check — Can You Answer These?

- When would sparse retrieval (BM25) outperform dense retrieval?
- What is Reciprocal Rank Fusion and why does it work?
- Why is re-ranking slower but more accurate than initial retrieval?

If you cannot answer one, go back and re-read that part. That is completely normal.

## Victory Lap

You just learned the retrieval strategies used by every serious search and RAG system in production today. Google, Bing, and enterprise search all use hybrid retrieval with re-ranking. When someone tells you "just use embeddings," you now know that is only part of the story — and you know exactly when keywords win, when meaning wins, and how to combine them.

Ready to go deeper? The interview deep-dive covers the full BM25 formula derivation, RRF sensitivity analysis, sparse-vs-dense failure mode construction, and the complexity analysis interviewers expect. See [retrieval-strategies-interview.md](./retrieval-strategies-interview.md).

---

[Back to RAG module](./README.md)
