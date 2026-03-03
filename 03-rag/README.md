# Retrieval-Augmented Generation (RAG)

Have you ever asked a chatbot a question and gotten a confident but completely wrong answer? Large language models are powerful, but they have a big weakness: they can only use what they learned during training. Ask about your company's latest product, yesterday's news, or a private document — and they guess. Sometimes they guess wrong. RAG fixes this.

**Before you start, you need to know:**
- What a vector is (a list of numbers) — introduced in the first notebook
- Basic Python (variables, functions, loops)
- No ML knowledge required — everything is explained from scratch

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. The idea is simple: before the language model answers your question, it first *looks up* relevant information from a knowledge base. Then it uses that information to give a grounded answer.

Think of it like an open-book exam. Without RAG, the model has to answer from memory. With RAG, it can flip through its notes first.

This analogy gets the core idea right: the model uses external material to answer better. It breaks down because the "flipping through notes" part is actually a sophisticated search process involving embeddings, vector similarity, and retrieval strategies — which is what this module teaches you.

## What You Will Learn

By the end of this module, you will understand:
- How computers understand and search through text using embeddings
- How to split documents into searchable pieces (chunking)
- How vector databases store and search by meaning
- Different ways to find relevant information (sparse, dense, hybrid retrieval)
- How to build a complete RAG pipeline from scratch

## Coverage Map

### RAG Fundamentals

| Topic | Depth | Files |
|-------|-------|-------|
| What is RAG — embeddings, vector similarity, the RAG pipeline | [Applied] | [what-is-rag.md](./what-is-rag.md) · [01_what_is_rag.ipynb](./01_what_is_rag.ipynb) |

### Core Techniques

| Topic | Depth | Files |
|-------|-------|-------|
| Chunking Techniques — fixed-size, overlapping, sentence, semantic, recursive | [Core] | [chunking-techniques.md](./chunking-techniques.md) · [chunking-techniques-interview.md](./chunking-techniques-interview.md) · [02_chunking_techniques.ipynb](./02_chunking_techniques.ipynb) · [02_chunking_techniques_experiments.ipynb](./02_chunking_techniques_experiments.ipynb) |
| Vector Databases — indexing algorithms, distance metrics, popular DBs | [Core] | [vector-databases.md](./vector-databases.md) · [vector-databases-interview.md](./vector-databases-interview.md) · [03_vector_databases.ipynb](./03_vector_databases.ipynb) · [03_vector_databases_experiments.ipynb](./03_vector_databases_experiments.ipynb) |
| Retrieval Strategies — sparse, dense, hybrid search, re-ranking | [Core] | [retrieval-strategies.md](./retrieval-strategies.md) · [retrieval-strategies-interview.md](./retrieval-strategies-interview.md) · [04_retrieval_strategies.ipynb](./04_retrieval_strategies.ipynb) · [04_retrieval_strategies_experiments.ipynb](./04_retrieval_strategies_experiments.ipynb) |

### End-to-End Systems

| Topic | Depth | Files |
|-------|-------|-------|
| Building a RAG Pipeline — document loading, prompt engineering, evaluation | [Applied] | [building-rag-pipeline.md](./building-rag-pipeline.md) · [05_building_rag_pipeline.ipynb](./05_building_rag_pipeline.ipynb) |
| Agentic RAG — routing, self-correction, multi-step retrieval | [Awareness] | [README.md#agentic-rag](#agentic-rag) |

## Study Plan

Follow the notebooks **in order** — each builds on the previous one:

| # | Topic | What You Will Learn | Difficulty |
|---|-------|---------------------|-----------|
| 01 | [What is RAG?](01_what_is_rag.ipynb) | RAG concepts, embeddings, vector similarity, the RAG pipeline | Beginner |
| 02 | [Chunking Techniques](02_chunking_techniques.ipynb) | Fixed-size, overlapping, sentence-based, semantic, recursive chunking | Beginner |
| 03 | [Vector Databases](03_vector_databases.ipynb) | How vector DBs work, indexing algorithms (Flat, IVF, HNSW), popular DBs | Beginner+ |
| 04 | [Retrieval Strategies](04_retrieval_strategies.ipynb) | TF-IDF, BM25, dense retrieval, hybrid search, re-ranking | Intermediate |
| 05 | [Building a RAG Pipeline](05_building_rag_pipeline.ipynb) | End-to-end pipeline, prompt engineering, evaluation, failure modes | Intermediate |

### Prerequisites

- **Python basics** (variables, functions, loops)
- **No ML knowledge required** — everything is explained from scratch
- Only uses `numpy` and `matplotlib` (no paid APIs or complex setups needed)

### Suggested Approach

1. **Read the markdown file first** for each topic to understand the concepts
2. **Run each code cell** in the notebook and observe the outputs
3. **Try the checkpoint questions** at the end of each section
4. **Experiment!** Change parameters and see what happens

## Key Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| **RAG** | Giving an LLM reference material so it can answer questions about your documents |
| **Embedding** | Converting text into a list of numbers that captures its meaning (like GPS coordinates for words) |
| **Vector Similarity** | Measuring how close two embeddings are (how similar their meanings are) |
| **Chunking** | Splitting large documents into smaller, searchable pieces |
| **Vector Database** | A special database that finds items by meaning, not just keywords |
| **Dense Retrieval** | Finding documents using embeddings (understands meaning) |
| **Sparse Retrieval** | Finding documents using keywords (TF-IDF, BM25) |
| **Hybrid Retrieval** | Combining dense + sparse for best results |
| **Re-ranking** | Re-scoring retrieved results with a more powerful model |

## Agentic RAG

Standard RAG follows a fixed pipeline: retrieve, then generate. Agentic RAG makes the system smarter by adding decision-making. The system can:

- **Route queries** — decide whether a question needs retrieval at all, or can be answered from the model's own knowledge
- **Self-correct** — check its own answer and re-retrieve if the quality is low
- **Multi-step retrieval** — break a complex question into sub-questions and retrieve for each one

Agentic RAG builds on everything in this module (chunking, vector search, retrieval strategies) and adds an orchestration layer on top. For a deep dive into agentic RAG with full implementation examples, see [genAI design / RAG](../genAI%20design/06-rag/).

## Key Papers

- **RAG** — [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- **DPR** — [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020)
- **Self-RAG** — [Self-Reflective Retrieval-Augmented Generation](https://arxiv.org/abs/2310.11511) (Asai et al., 2023)
- **CRAG** — [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) (Yan et al., 2024)

## Useful Resources

- [ChromaDB](https://docs.trychroma.com/) — Easy-to-use vector database
- [FAISS](https://github.com/facebookresearch/faiss) — Fast similarity search library
- [Sentence Transformers](https://www.sbert.net/) — Popular embedding models

---

[Back to Main](../README.md)
