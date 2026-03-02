# 📚 Chapter 06: Retrieval-Augmented Generation (RAG)

> **Teaching AI to Read Before Answering — Like an Open-Book Exam!**
> This chapter covers how products like Perplexity.ai and ChatPDF actually work under the hood.

---

## 🎮 Tell Me Like I'm 12

Imagine you're taking a test in school. There are three ways to answer questions:

1. **Memorize everything** (Fine-tuning) — You study so hard that you memorize every fact. Great for stuff that never changes, but what about yesterday's news? 🧠💥
2. **Get hints from the teacher** (Prompt Engineering) — The teacher whispers some context before each question. Helpful, but limited! 🤫
3. **Open-book exam** (RAG) — You bring your textbooks and look things up! You still need to understand the material, but you can find specific facts on the fly. 📖✅

**RAG = the open-book exam approach.** The AI searches through documents, finds relevant passages, and then writes an answer based on what it found. That's how Perplexity.ai gives you sourced answers, and how ChatPDF lets you "talk" to a PDF!

---

## 🗺️ What's In This Chapter?

```
THE RAG PIPELINE
================

  Your Question        Find Relevant Info       Generate Answer
  "What is X?"    →    📄📄📄 search docs   →   🤖 write response
                       (retrieval)               (generation)

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   01_retrieval_fundamentals.ipynb                           │
  │   ├── Finetuning vs Prompt Engineering vs RAG               │
  │   ├── Document Parsing & Chunking                           │
  │   ├── Vector Embeddings (text → numbers)                    │
  │   ├── Nearest Neighbor Search (ANN)                         │
  │   └── FAISS for fast vector search                          │
  │                                                             │
  │   02_generation_and_evaluation.ipynb                        │
  │   ├── Prompt Engineering Principles & Techniques            │
  │   ├── RAFT (Retrieval-Augmented Fine-Tuning)                │
  │   ├── Evaluation: Context Relevance, Faithfulness,          │
  │   │    Answer Relevance, Answer Correctness                 │
  │   └── Overall System Design & Interview Walkthrough         │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Concepts at a Glance

### What Is RAG?

**Retrieval-Augmented Generation** = Retrieve relevant documents + Generate an answer from them.

Instead of relying solely on what the LLM "memorized" during training, RAG lets the model **look things up** in real-time. Think of it as giving the AI a library card. 📚

**Real Products Using RAG:**
| Product | How It Uses RAG |
|---------|----------------|
| **Perplexity.ai** | Searches the web, retrieves pages, generates sourced answers |
| **ChatPDF** | Parses your uploaded PDF, chunks it, retrieves relevant sections to answer your questions |
| **GitHub Copilot Chat** | Retrieves relevant code from your repo to answer coding questions |
| **Notion AI** | Searches your workspace documents to generate contextual responses |
| **Enterprise chatbots** | Search internal knowledge bases (Confluence, docs) to answer employee questions |

---

### 🥊 The Big Three: Fine-tuning vs Prompt Engineering vs RAG

| Aspect | Fine-tuning 🏋️ | Prompt Engineering 🎯 | RAG 📚 |
|--------|-----------------|----------------------|--------|
| **Analogy** | Memorizing textbooks | Getting hints from teacher | Open-book exam |
| **How it works** | Retrain the model on new data | Craft clever prompts with context | Retrieve docs, stuff into prompt |
| **Knowledge updates** | Must retrain (expensive! 💰) | Update prompt template | Update document index (cheap! ✅) |
| **Hallucination risk** | Medium (can still make stuff up) | Medium | Low (grounded in real docs) |
| **Cost** | High (GPU, training time) | Low (just API calls) | Medium (embedding + storage + API) |
| **Latency** | Fast inference (knowledge baked in) | Fast (one API call) | Slower (search + generate) |
| **Best for** | Domain-specific style/behavior | Simple tasks, quick wins | Knowledge-intensive Q&A |
| **Data freshness** | Stale after training cutoff | Limited by context window | Always up-to-date ✨ |
| **Traceability** | ❌ Can't cite sources | ❌ Can't cite sources | ✅ Can cite exact passages |

**When to use what:**
- 🏋️ **Fine-tuning:** You need the model to behave differently (tone, format, domain-specific reasoning)
- 🎯 **Prompt Engineering:** Quick wins, prototyping, or when the context fits in the prompt
- 📚 **RAG:** The model needs access to large, changing knowledge bases it wasn't trained on

---

### 📄 Document Parsing

Before you can search documents, you need to **extract clean text** from messy file formats (PDFs, HTML, Word docs, scanned images).

| Approach | How It Works | Good For | Limitations |
|----------|-------------|----------|-------------|
| **Rule-based** | Regex, HTML parsers, PDF extractors | Clean, structured docs | Breaks on complex layouts |
| **AI-based (Layout-Parser)** | ML models detect text regions, tables, figures | Scanned docs, complex PDFs | Slower, needs GPU |

---

### ✂️ Document Chunking

Once you have clean text, you split it into **chunks** — smaller pieces that fit into the LLM's context window and can be individually retrieved.

| Strategy | How It Works | Pros | Cons |
|----------|-------------|------|------|
| **Length-based** | Split every N characters/tokens | Simple, predictable | Cuts mid-sentence 😬 |
| **Overlap chunking** | Length-based but with overlap between chunks | Preserves context at boundaries | Redundant storage |
| **Regex-based** | Split on paragraph breaks, sentences | Respects natural boundaries | Uneven chunk sizes |
| **HTML/Markdown splitters** | Split on headers (h1, h2, ##) | Preserves document structure | Only works for structured docs |
| **Semantic chunking** | Use embeddings to detect topic shifts | Most meaningful chunks | Expensive to compute |

**The golden rule:** Chunks should be big enough to contain a complete thought, but small enough to be specific. Think Goldilocks! 🐻

---

### 🔢 Vector Embeddings

How do you search through millions of text chunks to find the relevant ones? You turn text into **vectors** (lists of numbers) and find which vectors are "close" to each other!

```
"The cat sat on the mat"  →  [0.23, -0.45, 0.12, ..., 0.67]   (768 numbers)
"A kitten rested on a rug" →  [0.21, -0.43, 0.14, ..., 0.65]  (similar! ✅)
"Stock prices rose today"  →  [-0.56, 0.78, -0.34, ..., -0.12] (different! ❌)
```

**Text Encoder:** An encoder-only Transformer (like BERT or Sentence-BERT) that converts text → vectors.

**Image Encoder:** A Vision Transformer (ViT) or CNN that converts images → vectors.

**CLIP:** A model that maps both text AND images into the **same** vector space, so you can search images with text queries! 🖼️↔️📝

---

### 🔍 Nearest Neighbor Search

Given a query vector, find the most similar document vectors. The naive approach (compare to every vector) is too slow for millions of documents.

**Approximate Nearest Neighbor (ANN)** trades a tiny bit of accuracy for massive speed gains:

| Category | Algorithm | Analogy | Speed | Accuracy |
|----------|-----------|---------|-------|----------|
| **Tree-based** | KD-Tree, Annoy | Organizing a library with nested sections | ⭐⭐⭐ | ⭐⭐⭐ |
| **Hashing (LSH)** | Locality-Sensitive Hashing | Sorting by zip code to find neighbors | ⭐⭐⭐⭐ | ⭐⭐ |
| **Clustering** | IVF (Inverted File Index) | Looking only in the right neighborhood | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Graph-based** | HNSW | Skip list / express train stops | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**HNSW (Hierarchical Navigable Small World)** is the current champion — used by most production systems. Think of it like a subway system: express trains (top layers) get you to the neighborhood, then local trains (bottom layers) get you to the exact stop.

**FAISS** (by Meta) is the most popular library for vector search. It supports all the ANN methods above.

---

### 🎯 Prompt Engineering for RAG

Once you retrieve relevant chunks, you need to **engineer the prompt** to help the LLM generate a good answer.

**Five Key Principles:**
1. 📝 **Be specific and clear** — Tell the model exactly what you want
2. 📐 **Provide structure** — Use delimiters, sections, formatting
3. 🎭 **Assign a role** — "You are a helpful research assistant..."
4. 📋 **Give examples** — Few-shot learning (show input→output pairs)
5. 🔗 **Chain reasoning** — "Think step by step" (Chain-of-Thought)

**Key Techniques:**
- **Chain-of-Thought (CoT):** "Let's think step by step..." — forces the model to show its reasoning
- **Few-shot prompting:** Provide 2-3 examples of the desired input→output format
- **Role-specific prompting:** "You are an expert lawyer reviewing contracts..."
- **User-context prompting:** Include user preferences, history, or constraints

---

### 📊 Evaluation Triad (How Do We Know RAG Is Working?)

RAG evaluation has **four dimensions** — think of it as a report card for your RAG system:

```
                    ┌─────────────────────┐
                    │   USER QUESTION     │
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │    RETRIEVAL        │──── 1️⃣ Context Relevance
                    │  (find documents)   │     "Did we find the right docs?"
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │    GENERATION       │──── 2️⃣ Faithfulness
                    │  (write answer)     │     "Is the answer supported by the docs?"
                    └─────────┬───────────┘
                              │
                    ┌─────────▼───────────┐
                    │    FINAL ANSWER     │──── 3️⃣ Answer Relevance
                    └─────────┬───────────┘     "Does it actually answer the question?"
                              │
                    ┌─────────▼───────────┐
                    │   GROUND TRUTH      │──── 4️⃣ Answer Correctness
                    └─────────────────────┘     "Is the answer factually correct?"
```

| Dimension | What It Measures | Key Metrics |
|-----------|-----------------|-------------|
| **1️⃣ Context Relevance** | Did retrieval find useful docs? | Hit Rate, MRR, NDCG, Precision@k |
| **2️⃣ Faithfulness** | Is the answer grounded in retrieved docs? | Claim verification, NLI-based scoring |
| **3️⃣ Answer Relevance** | Does the answer address the question? | Semantic similarity, LLM-as-judge |
| **4️⃣ Answer Correctness** | Is the answer factually right? | Exact match, F1, human evaluation |

---

### 🏗️ Overall System Design

A production RAG system has way more than just "retrieve and generate." Here's the full picture:

```
┌──────────────────────────────────────────────────────────────────┐
│                     OFFLINE INDEXING PIPELINE                    │
│                                                                  │
│  Raw Docs → Parse → Chunk → Embed → Store in Vector DB          │
│  📄📄📄    📝       ✂️      🔢        🗄️                        │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     ONLINE QUERY PIPELINE                        │
│                                                                  │
│  User Query → Safety Filter → Query Expansion → Embed Query     │
│     ❓           🛡️              🔄                🔢            │
│                                                                  │
│  → Retrieve Top-K → Re-rank → Build Prompt → Generate Answer    │
│     🔍               📊         📝              🤖               │
│                                                                  │
│  → Safety Filter → Return with Citations                         │
│     🛡️              📎                                           │
└──────────────────────────────────────────────────────────────────┘
```

**Key components you should know for interviews:**
- **Safety filtering:** Block harmful/PII content at input AND output
- **Query expansion:** Rewrite user query for better retrieval (e.g., HyDE — hypothetical document embeddings)
- **Re-ranking:** Use a cross-encoder to re-score retrieved chunks for better precision
- **Citation generation:** Map answer claims back to source documents

---

## 📓 Notebook Guide

### 01 - Retrieval Fundamentals

**The first half of RAG: finding the right information.**

This notebook starts with the "open-book exam" analogy, compares fine-tuning vs prompt engineering vs RAG, and then dives deep into every component of the retrieval pipeline: document parsing, chunking strategies, vector embeddings, and approximate nearest neighbor search.

You'll implement text chunking with overlap, see how text becomes vectors, build a simple clustering-based ANN, and use FAISS for real vector search.

**Key concepts:** chunking, embeddings, cosine similarity, ANN, HNSW, FAISS, IVF

**Plain English:** Before the AI can answer your question about a document, it needs to (1) chop the document into bite-sized pieces, (2) convert each piece into a list of numbers, and (3) find which pieces are closest to your question. That's retrieval.

---

### 02 - Generation & Evaluation

**The second half of RAG: writing the answer and measuring quality.**

This notebook covers prompt engineering principles (how to write prompts that make the LLM behave), RAFT (combining RAG with fine-tuning), and the full evaluation framework. You'll learn context relevance metrics (hit rate, MRR, NDCG), faithfulness checking, and answer correctness evaluation.

Ends with a complete system design walkthrough, quiz, and interview talking points.

**Key concepts:** CoT prompting, few-shot, evaluation triad, MRR, NDCG, faithfulness, RAFT

**Plain English:** Once you've found the right document chunks, you need to (1) write a really good prompt that includes those chunks, (2) get the LLM to generate a faithful answer, and (3) measure whether the whole pipeline is actually working. That's generation + evaluation.

---

## 🎯 Interview Cheat Sheet

### Opening Statement
> "RAG is a technique that augments LLM generation with external knowledge retrieval. Instead of relying solely on parametric knowledge (what the model memorized during training), RAG retrieves relevant documents at inference time and conditions the generation on them. This gives us updatable knowledge, source traceability, and reduced hallucination."

### Key Talking Points

**1. Why RAG over fine-tuning?**
- Knowledge can be updated without retraining (just update the index)
- Provides source attribution (you can cite which document the answer came from)
- Reduces hallucination (answer is grounded in real documents)
- More cost-effective for knowledge-intensive tasks

**2. Chunking strategy matters a LOT**
- Too small = missing context. Too large = noise dilutes the signal
- Overlap between chunks preserves boundary context
- Semantic chunking > fixed-length for quality (but more expensive)
- Typical chunk sizes: 256-512 tokens with 10-20% overlap

**3. Embedding model selection**
- Sentence-BERT / E5 / GTE for text-only
- CLIP for multi-modal (text + image)
- Dimension-accuracy tradeoff: 768d is common, 384d for efficiency
- Fine-tune embedding model on your domain for best results

**4. HNSW is king for ANN**
- Hierarchical graph with O(log N) search time
- Best accuracy-speed tradeoff of all ANN methods
- Parameters: M (connections per node), ef (search beam width)
- Used by Pinecone, Weaviate, Qdrant, pgvector

**5. Evaluation is non-negotiable**
- Context relevance: Are you retrieving the right chunks? (MRR, NDCG)
- Faithfulness: Is the answer grounded in the chunks? (claim verification)
- Answer relevance: Does the answer address the question? (semantic similarity)
- Answer correctness: Is it factually right? (F1 vs ground truth)

### Common Interview Questions

| Question | Key Points |
|----------|-----------|
| "Design a RAG system for X" | Clarify requirements → chunking strategy → embedding model → vector DB → retrieval → re-ranking → prompt template → evaluation |
| "How do you handle hallucinations?" | Faithful prompting ("Only answer based on the context"), claim verification, citation generation, confidence thresholds |
| "How do you evaluate RAG?" | The triad: context relevance (retrieval quality), faithfulness (grounding), answer relevance + correctness |
| "RAG vs fine-tuning?" | RAG for dynamic knowledge + traceability. Fine-tuning for behavior/style changes. Can combine (RAFT). |
| "How do you scale to millions of docs?" | ANN indexing (HNSW/IVF), sharding, quantization (PQ), tiered retrieval (sparse → dense → re-rank) |
| "What about multi-modal RAG?" | CLIP embeddings for shared text-image space, image chunking, multi-modal re-ranking |

### System Design Template

```
1. REQUIREMENTS
   - What types of documents? (PDF, web, code, images?)
   - Latency requirements? (real-time chat vs batch)
   - Scale? (thousands vs millions of documents)
   - Accuracy vs speed tradeoff?

2. INDEXING PIPELINE (offline)
   - Document parsing (rule-based vs AI-based)
   - Chunking strategy (length + overlap + semantic)
   - Embedding model selection
   - Vector database (Pinecone, Weaviate, Qdrant, pgvector)

3. QUERY PIPELINE (online)
   - Safety filtering (input)
   - Query expansion / rewriting
   - Retrieval (ANN search, top-K)
   - Re-ranking (cross-encoder)
   - Prompt construction
   - LLM generation
   - Safety filtering (output)
   - Citation generation

4. EVALUATION
   - Context relevance metrics (MRR, NDCG, Precision@k)
   - Faithfulness evaluation
   - Answer quality metrics
   - A/B testing in production

5. DEPLOYMENT
   - Index update strategy (batch vs streaming)
   - Caching (query cache, embedding cache)
   - Monitoring (retrieval quality, latency, cost)
   - Feedback loop (user clicks, thumbs up/down)
```

---

## Prerequisites

- Python 3.8+
- NumPy, Matplotlib
- Basic understanding of Transformers and LLMs (Chapters 02-04)

```bash
pip install numpy matplotlib scikit-learn faiss-cpu
```

---

## Key Terms

| Term | Meaning |
|------|---------|
| **RAG** | Retrieval-Augmented Generation — retrieve docs then generate answers from them |
| **Chunking** | Splitting documents into smaller pieces for retrieval |
| **Embedding** | Converting text/images into numerical vectors that capture meaning |
| **Vector database** | A database optimized for storing and searching high-dimensional vectors |
| **ANN** | Approximate Nearest Neighbor — fast (but approximate) similarity search |
| **HNSW** | Hierarchical Navigable Small World — the most popular ANN algorithm |
| **FAISS** | Facebook AI Similarity Search — a library for efficient vector search |
| **Cosine similarity** | A measure of how similar two vectors are (1 = identical, 0 = unrelated) |
| **Re-ranking** | A second-pass scoring of retrieved chunks for better precision |
| **Faithfulness** | Whether the generated answer is supported by the retrieved documents |
| **RAFT** | Retrieval-Augmented Fine-Tuning — fine-tune the LLM to be better at RAG |
| **HyDE** | Hypothetical Document Embeddings — generate a fake answer, then search for similar real docs |

---

[Previous: Image Captioning](../05-image-captioning/) | [Back to Overview](../README.md) | [Next: Face Generation](../07-realistic-face-generation/)
