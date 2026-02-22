# 3️⃣ Retrieval-Augmented Generation (RAG)

## Overview

Building LLM systems enhanced with external knowledge through retrieval mechanisms. This module takes you from zero to building intelligent, agentic RAG systems — no prior ML knowledge required!

## 🎯 What You'll Learn

By the end of this module, you'll understand:
- How computers understand and search through text using embeddings
- How to split documents into searchable pieces (chunking)
- How vector databases store and search by meaning
- Different ways to find relevant information (sparse, dense, hybrid retrieval)
- How to build a complete RAG pipeline from scratch
- How to make RAG systems "smart" with agents (agentic RAG using LangGraph)

## 📚 Study Plan

Follow the notebooks **in order** — each builds on the previous one:

| # | Notebook | What You'll Learn | Difficulty |
|---|----------|-------------------|-----------|
| 01 | [What is RAG?](01_what_is_rag.ipynb) | RAG concepts, embeddings, vector similarity, the RAG pipeline | ⭐ Beginner |
| 02 | [Chunking Techniques](02_chunking_techniques.ipynb) | Fixed-size, overlapping, sentence-based, semantic, recursive chunking | ⭐ Beginner |
| 03 | [Vector Databases](03_vector_databases.ipynb) | How vector DBs work, indexing algorithms (Flat, IVF, HNSW), popular DBs | ⭐⭐ Beginner+ |
| 04 | [Retrieval Strategies](04_retrieval_strategies.ipynb) | TF-IDF, BM25, dense retrieval, hybrid search, re-ranking | ⭐⭐ Intermediate |
| 05 | [Building a RAG Pipeline](05_building_rag_pipeline.ipynb) | End-to-end pipeline, prompt engineering, evaluation, failure modes | ⭐⭐⭐ Intermediate |
| 06 | [Agentic RAG with LangGraph](06_agentic_rag_langgraph.ipynb) | Routing, self-correction, multi-step retrieval, LangGraph patterns | ⭐⭐⭐ Advanced |

### Prerequisites

- **Python basics** (variables, functions, loops)
- **No ML knowledge required** — everything is explained from scratch!
- Only uses `numpy` and `matplotlib` (no paid APIs or complex setups needed)

### Suggested Approach

1. **Read the markdown cells first** to understand the concepts
2. **Run each code cell** and observe the outputs
3. **Try the "Test Your Understanding" questions** at the end of each notebook
4. **Experiment!** Change parameters and see what happens

## 📂 Directory Structure

```
03-rag/
├── README.md                          ← You are here
├── 01_what_is_rag.ipynb               ← Start here!
├── 02_chunking_techniques.ipynb
├── 03_vector_databases.ipynb
├── 04_retrieval_strategies.ipynb
├── 05_building_rag_pipeline.ipynb
├── 06_agentic_rag_langgraph.ipynb     ← Final notebook
├── chunking-techniques/               ← Supplementary materials
│   └── README.md
├── retrieval-strategies/
│   └── README.md
├── vector-databases/
│   └── README.md
└── experiments/                       ← Your experiments go here
```

## Key Concepts

| Concept | Simple Explanation |
|---------|-------------------|
| **RAG** | Giving an LLM reference material so it can answer questions about YOUR documents |
| **Embedding** | Converting text into a list of numbers that captures its meaning (like GPS coordinates for words) |
| **Vector Similarity** | Measuring how close two embeddings are (how similar their meanings are) |
| **Chunking** | Splitting large documents into smaller, searchable pieces |
| **Vector Database** | A special database that finds items by meaning, not just keywords |
| **Dense Retrieval** | Finding documents using embeddings (understands meaning) |
| **Sparse Retrieval** | Finding documents using keywords (TF-IDF, BM25) |
| **Hybrid Retrieval** | Combining dense + sparse for best results |
| **Re-ranking** | Re-scoring retrieved results with a more powerful model |
| **Agentic RAG** | RAG systems that can route, self-correct, and perform multi-step retrieval |
| **LangGraph** | A Python library for building agent workflows as graphs |

## Key Papers

- **RAG** — [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020)
- **DPR** — [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (Karpukhin et al., 2020)
- **Self-RAG** — [Self-Reflective Retrieval-Augmented Generation](https://arxiv.org/abs/2310.11511) (Asai et al., 2023)
- **CRAG** — [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) (Yan et al., 2024)

## Useful Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ChromaDB](https://docs.trychroma.com/) — Easy-to-use vector database
- [FAISS](https://github.com/facebookresearch/faiss) — Fast similarity search library
- [Sentence Transformers](https://www.sbert.net/) — Popular embedding models

---

[Back to Main](../README.md) | [Previous: Fine-Tuning](../02-fine-tuning/README.md) | [Next: Prompt Engineering](../04-prompt-engineering/README.md)
