# Vector Databases

## What's a Vector Database?

Imagine you have a library with millions of books. A regular database is like a card
catalog -- you can look up books by title, author, or category. But what if you want
to find books that are **similar in meaning** to a question you have? The card catalog
can't do that.

A **vector database** is a special kind of database designed for **similarity search**.
Instead of looking up exact matches (like "find the book titled X"), it finds things
that are **close in meaning** ("find documents similar to my question").

```
+-------------------------------------------------------------------+
|          Regular Database vs. Vector Database                     |
|                                                                   |
|  Regular Database:                                                 |
|    Query: "Find all books by Author X"                            |
|    Method: Exact match on the "author" field                      |
|    Result: Books 4, 17, 23                                        |
|                                                                   |
|  Vector Database:                                                  |
|    Query: "How do plants make food?"                              |
|    Method: Find documents whose MEANING is closest                |
|    Result: "Photosynthesis explained", "How leaves work",         |
|            "The biology of plant nutrition"                        |
|                                                                   |
|  The vector database understood the CONCEPT, not just keywords!   |
+-------------------------------------------------------------------+
```

---

## How It Works: Embeddings

Before a vector database can find "similar" things, it needs a way to represent
meaning as **numbers**. That's what **embeddings** do.

An **embedding** is a list of numbers (a "vector") that represents the meaning
of a piece of text. Similar texts get similar number patterns.

```
Sentence: "The cat sat on the mat"
Embedding: [0.12, -0.45, 0.78, 0.33, -0.21, ...]   (hundreds of numbers)

Sentence: "A kitten rested on the rug"
Embedding: [0.14, -0.43, 0.76, 0.31, -0.19, ...]   (very similar numbers!)

Sentence: "The stock market crashed today"
Embedding: [0.91, 0.22, -0.56, 0.08, 0.67, ...]   (very different numbers!)
```

The vector database stores these embeddings and can quickly find which ones are
closest to your query.

```
+-------------------------------------------------------------------+
|                How Vector Search Works                             |
|                                                                   |
|   Step 1: Convert your documents into embeddings                  |
|           (done once, stored in the database)                     |
|                                                                   |
|        "How photosynthesis works"  -->  [0.3, 0.7, -0.2, ...]    |
|        "History of the Roman Empire" --> [0.8, -0.1, 0.5, ...]    |
|        "Plant biology basics"      -->  [0.35, 0.68, -0.15, ...] |
|                                                                   |
|   Step 2: Convert your QUERY into an embedding                    |
|                                                                   |
|        "How do plants make food?"  -->  [0.32, 0.69, -0.18, ...] |
|                                                                   |
|   Step 3: Find the CLOSEST embeddings in the database             |
|                                                                   |
|        Closest match: "How photosynthesis works" (distance: 0.05) |
|        Second match:  "Plant biology basics"     (distance: 0.08) |
|        Far away:      "History of Roman Empire"  (distance: 0.91) |
+-------------------------------------------------------------------+
```

---

## Distance Metrics: How "Close" Is Close?

To find the closest vectors, we need a way to measure distance.
There are three common methods:

| Metric | How It Works | Analogy |
|--------|-------------|---------|
| **Cosine Similarity** | Measures the angle between two vectors | "Are these arrows pointing in the same direction?" |
| **Euclidean Distance** | Measures the straight-line distance | "How far apart are these two points on a map?" |
| **Dot Product** | Combines direction and magnitude | "How aligned are these and how strong are they?" |

```
+-------------------------------------------------------------------+
|              Cosine Similarity (most common)                      |
|                                                                   |
|   Cosine = 1.0   -->  Identical meaning                          |
|   Cosine = 0.8   -->  Very similar                                |
|   Cosine = 0.5   -->  Somewhat related                            |
|   Cosine = 0.0   -->  Unrelated                                   |
|   Cosine = -1.0  -->  Opposite meaning                            |
|                                                                   |
|   Most vector databases use cosine similarity by default.         |
+-------------------------------------------------------------------+
```

---

## Popular Vector Databases

Here's a comparison of the most popular options:

| Database | Type | Best For | Complexity |
|----------|------|----------|------------|
| **FAISS** | Library (in-memory) | Prototyping, research, single-machine use | Low |
| **ChromaDB** | Lightweight DB | Small projects, getting started, local dev | Low |
| **Pinecone** | Managed cloud service | Production apps, no infrastructure management | Low |
| **Weaviate** | Self-hosted / cloud | Full-featured apps, hybrid search | Medium |
| **Qdrant** | Self-hosted / cloud | Performance-critical apps, filtering | Medium |
| **Milvus** | Self-hosted | Large-scale production, billions of vectors | High |
| **pgvector** | PostgreSQL extension | Already using Postgres, don't want another DB | Low |

```
+-------------------------------------------------------------------+
|              Choosing a Vector Database                            |
|                                                                   |
|   Just learning / prototyping?                                    |
|     --> ChromaDB or FAISS (easiest to set up)                     |
|                                                                   |
|   Building a production app?                                      |
|     --> Pinecone (managed) or Qdrant (self-hosted)                |
|                                                                   |
|   Already using PostgreSQL?                                       |
|     --> pgvector (just add an extension)                          |
|                                                                   |
|   Need to handle billions of vectors?                             |
|     --> Milvus or Weaviate                                        |
+-------------------------------------------------------------------+
```

### FAISS (Facebook AI Similarity Search)

- **What:** A library from Meta for fast similarity search
- **Pros:** Extremely fast, works on GPU, great for research
- **Cons:** Not a full database (no persistence by default), requires more code
- **Use when:** You need speed and are comfortable with Python

### ChromaDB

- **What:** An open-source, lightweight vector database
- **Pros:** Dead simple to get started, Python-native, good for RAG
- **Cons:** Less mature, limited scaling
- **Use when:** You're building your first RAG system

### Pinecone

- **What:** A fully managed cloud vector database
- **Pros:** Zero infrastructure, auto-scaling, easy API
- **Cons:** Costs money, vendor lock-in, data leaves your machine
- **Use when:** You want production-ready without managing servers

---

## Key Operations

Every vector database supports these core operations:

```
+-------------------------------------------------------------------+
|                  Vector Database Operations                       |
|                                                                   |
|  1. UPSERT (Insert/Update)                                       |
|     Add a document and its embedding to the database              |
|     db.upsert(id="doc1", vector=[0.1, 0.3, ...],                |
|               metadata={"source": "textbook"})                    |
|                                                                   |
|  2. QUERY (Search)                                                |
|     Find the K most similar documents to a query vector           |
|     results = db.query(vector=[0.2, 0.35, ...], top_k=5)         |
|                                                                   |
|  3. FILTER                                                        |
|     Combine vector search with metadata filters                   |
|     results = db.query(vector=..., top_k=5,                      |
|                        filter={"source": "textbook"})             |
|                                                                   |
|  4. DELETE                                                        |
|     Remove documents by ID or filter                              |
|     db.delete(ids=["doc1", "doc2"])                               |
+-------------------------------------------------------------------+
```

---

## Indexing: How Databases Search Fast

Searching through millions of vectors one by one would be too slow. Vector databases
use **indexing algorithms** to speed things up:

| Algorithm | How It Works | Speed vs. Accuracy |
|-----------|-------------|-------------------|
| **Flat (Brute Force)** | Compare against every vector | Slowest but 100% accurate |
| **IVF (Inverted File)** | Cluster vectors, only search nearby clusters | Fast, ~95% accurate |
| **HNSW (Hierarchical NSW)** | Build a graph connecting similar vectors | Very fast, ~98% accurate |
| **PQ (Product Quantization)** | Compress vectors to save memory | Fastest, ~90% accurate |

```
Think of it like finding a book in a library:

Flat:  Walk through every shelf, check every book
       (accurate but slow)

IVF:   Go to the right SECTION first (Science, History...),
       then search within that section
       (much faster, might miss a misplaced book)

HNSW:  Ask a librarian who points you to another librarian
       who points you to the right shelf -- a chain of
       "people who know people"
       (very fast, almost always finds the right book)
```

---

## Summary

```
+------------------------------------------------------------------+
|               Vector Databases Cheat Sheet                       |
|                                                                  |
|  What:     Databases designed for similarity search              |
|  How:      Store embeddings (number lists representing meaning)  |
|  Why:      Find semantically similar content, not just keywords  |
|                                                                  |
|  Key concepts:                                                   |
|    - Embeddings turn text into numbers                           |
|    - Cosine similarity measures how close two vectors are        |
|    - Indexing algorithms (HNSW, IVF) make search fast            |
|                                                                  |
|  For beginners: Start with ChromaDB or FAISS                    |
|  For production: Consider Pinecone, Qdrant, or Weaviate         |
+------------------------------------------------------------------+
```

---

## Further Reading

- **Efficient Estimation of Word Representations in Vector Space** -- Mikolov et al., 2013
  (Word2Vec -- the paper that started the embeddings revolution)
- **Billion-scale similarity search with GPUs** -- Johnson et al., 2019 (FAISS paper)
- ChromaDB documentation -- Great getting-started tutorials
- Pinecone learning center -- Excellent visual guides to vector search

---

[Back to RAG](../README.md)
