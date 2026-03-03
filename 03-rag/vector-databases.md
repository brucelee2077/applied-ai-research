# Vector Databases

You can turn any piece of text into a list of numbers (an embedding) that captures its meaning. But once you have millions of these number lists, how do you quickly find the one closest to your question? You cannot compare them one by one — that would take forever. This is the problem vector databases solve, and the clever tricks they use are surprisingly elegant.

**Before you start, you need to know:**
- What an embedding is (a list of numbers that captures meaning) — covered in [what-is-rag.md](./what-is-rag.md)
- What cosine similarity measures (how close two embeddings are) — covered in [what-is-rag.md](./what-is-rag.md)

## The Analogy

Imagine a library with millions of books. A regular database is like a card catalog — you can look up books by title, author, or category. But what if you want to find books that are similar in meaning to a question you have? The card catalog cannot do that.

A **vector database** is like a librarian who has read every book and organized them by topic in a special way. When you ask a question, the librarian does not search alphabetically — they walk straight to the right section because they know how the ideas connect.

### What the analogy gets right

Just like the librarian uses their understanding of content to find relevant books quickly, a vector database uses mathematical similarity to find the closest embeddings. And just like a good library organizes books so you do not have to search every shelf, vector databases use indexing algorithms to avoid comparing every single vector.

### The concept in plain words

A vector database stores embeddings and lets you search by meaning instead of keywords. You give it a question (converted to an embedding), and it finds the stored embeddings that are closest — meaning the documents that are most similar to your question.

The hard part is doing this fast. With a million vectors, comparing your question to every single one takes too long. So vector databases use clever indexing tricks: they organize the vectors into groups, build graph connections between similar ones, or compress them to save memory. These tricks trade a tiny bit of accuracy for a huge speed gain.

### Where the analogy breaks down

A real librarian understands books deeply. A vector database does not understand anything — it just calculates distances between lists of numbers. It can be fooled if the embeddings are bad, even though the concepts are related.

## Regular Database vs. Vector Database

| | Regular Database | Vector Database |
|---|---|---|
| **Query** | "Find books by Author X" | "Find documents about plant nutrition" |
| **Method** | Exact match on a field | Find nearest embeddings by distance |
| **Strength** | Precise, structured lookups | Meaning-based, fuzzy matching |

## Distance Metrics

To find the closest vectors, you need a way to measure distance. There are three common methods:

| Metric | How It Works | When to Use |
|--------|-------------|-------------|
| **Cosine Similarity** | Measures the angle between two vectors | Most common default. Works well when vector lengths vary. |
| **Euclidean Distance** | Measures the straight-line distance between two points | When magnitude matters (e.g., popularity-weighted embeddings). |
| **Dot Product** | Combines direction and magnitude | When vectors are already normalized. Fast to compute. |

Most vector databases use cosine similarity by default. A cosine similarity of 1.0 means identical meaning, 0.0 means unrelated, and -1.0 means opposite meaning.

## Indexing: How Databases Search Fast

Searching through millions of vectors one by one would be too slow. Vector databases use **indexing algorithms** to speed things up:

### Flat (Brute Force)

Compare the query against every single vector. Slowest but 100% accurate. Use only for small datasets (under 10K vectors).

### IVF (Inverted File Index)

Cluster the vectors into groups. When a query comes in, figure out which cluster it is closest to, then only search within that cluster (and maybe a few neighboring ones).

Think of it like going to the right section of a library first, then searching within that section. Much faster, but you might miss a relevant document that got placed in a different cluster.

### HNSW (Hierarchical Navigable Small World)

Build a graph where similar vectors are connected. Searching means hopping from node to node, getting closer to the answer at each step.

Think of it like asking a friend who knows someone who knows someone — a chain of connections that gets you to the right person fast.

### PQ (Product Quantization)

Compress each vector into a smaller representation. This saves memory and makes comparisons faster, but the compression loses some information.

| Algorithm | Speed | Accuracy | Memory | Best For |
|-----------|-------|----------|--------|----------|
| Flat | Slow | 100% | High | Small datasets, ground truth |
| IVF | Fast | ~95% | Medium | Medium datasets |
| HNSW | Very fast | ~98% | High | Most production use cases |
| PQ | Fastest | ~90% | Low | Very large datasets, memory-limited |

## Popular Vector Databases

| Database | Type | Best For | Complexity |
|----------|------|----------|------------|
| **FAISS** | Library (in-memory) | Prototyping, research | Low |
| **ChromaDB** | Lightweight DB | Small projects, getting started | Low |
| **Pinecone** | Managed cloud service | Production apps, no infrastructure management | Low |
| **Weaviate** | Self-hosted / cloud | Full-featured apps, hybrid search | Medium |
| **Qdrant** | Self-hosted / cloud | Performance-critical apps, filtering | Medium |
| **Milvus** | Self-hosted | Large-scale production, billions of vectors | High |
| **pgvector** | PostgreSQL extension | Already using Postgres, don't want another DB | Low |

**Just learning?** Start with ChromaDB or FAISS.
**Building for production?** Consider Pinecone (managed) or Qdrant (self-hosted).
**Already using PostgreSQL?** Add pgvector as an extension.

## Quick Check — Can You Answer These?

- What is the difference between a regular database search and a vector database search?
- Why can't you just compare every vector to the query (brute force) when you have millions of vectors?
- In your own words, how does HNSW speed up search?

If you cannot answer one, go back and re-read that part. That is completely normal.

## Victory Lap

You now understand how the "memory" behind every modern AI search system works. When you use ChatGPT with web search, when Google finds relevant results for vague queries, when Spotify recommends similar songs — vector databases and similarity search are behind all of it. You know the distance metrics, the indexing tricks, and the trade-offs. That is a big deal.

Ready to go deeper? The interview deep-dive covers the math behind HNSW construction, IVF clustering, Product Quantization distortion bounds, and the complexity analysis interviewers expect. See [vector-databases-interview.md](./vector-databases-interview.md).

---

[Back to RAG module](./README.md)
