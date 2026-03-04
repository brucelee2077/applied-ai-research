# RAG Architecture Patterns

## Introduction

RAG (Retrieval-Augmented Generation) is the most common architecture pattern in production genAI systems. The reason is pragmatic: LLMs hallucinate facts, have knowledge cutoffs, and can't access private data. RAG solves all three by retrieving relevant documents before generating an answer — grounding the model's output in actual source material.

Nearly every enterprise genAI application uses some form of RAG: customer support bots retrieving help articles, legal assistants searching case law, internal knowledge tools searching company documents, coding assistants pulling from documentation. Understanding the full RAG pipeline — from document ingestion to retrieval to generation — at production depth is essential for Staff-level genAI interviews.

---

## Why RAG Exists

### The Three Problems RAG Solves

| Problem | What Happens Without RAG | How RAG Fixes It |
|---------|-------------------------|-----------------|
| Knowledge cutoff | Model doesn't know about events/changes after its training data cutoff | Retrieve current information from an up-to-date knowledge base |
| Hallucination | Model generates plausible-sounding but incorrect facts | Ground responses in retrieved documents; model can cite sources |
| Private data | Model has no access to proprietary/internal information | Retrieve from private document stores without putting data in model weights |

### RAG vs Fine-Tuning

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| Knowledge freshness | Up-to-date (update documents anytime) | Stale until retrained |
| Setup cost | Low-medium (build ingestion pipeline) | High (curate data, train, evaluate) |
| Per-query cost | Higher (retrieval + longer prompt) | Lower (no retrieval, shorter prompt) |
| Hallucination | Reduced (grounded in sources) | Not reduced (knowledge in weights, can still hallucinate) |
| Knowledge boundary | Clear — can only answer from indexed documents | Fuzzy — hard to know what the model learned |
| Attribution | Can cite specific sources | Cannot attribute to specific training examples |

**When to choose RAG:** You need accurate, attributable, up-to-date answers grounded in specific documents.
**When to choose fine-tuning:** You need the model to internalize a style, format, or behavioral pattern — not specific facts.
**When to use both:** RAG for knowledge grounding + fine-tuning for output style/format.

---

## The RAG Pipeline

```
Documents → Chunking → Embedding → Vector Store (offline, batch)
                                          ↓
User Query → Query Processing → Retrieval → Re-ranking → Context Assembly → LLM → Response
                                                                                     ↓
                                                                              Citation/Attribution
```

---

## Document Ingestion

### Chunking Strategies

Documents must be split into chunks before embedding. Chunk size is one of the most impactful decisions in RAG quality.

| Strategy | How It Works | Chunk Size | Pros | Cons |
|----------|-------------|------------|------|------|
| Fixed-size | Split every N tokens | 256-1024 tokens | Simple, predictable | Splits mid-sentence, breaks context |
| Sentence-level | Split at sentence boundaries | 1-3 sentences | Preserves sentence meaning | Chunks may be too small for complex topics |
| Paragraph-level | Split at paragraph boundaries | 1-3 paragraphs | Natural topic boundaries | Inconsistent sizes, some paragraphs are huge |
| Semantic | Use embedding similarity to detect topic shifts | Variable | Best topic coherence | Slower, more complex |
| Recursive / hierarchical | Split large chunks, then subdivide if still too large | Variable | Balanced | Implementation complexity |

**Chunk size tradeoffs:**
- **Too small (< 100 tokens):** Loses context. A chunk like "Yes, that's correct" is meaningless without surrounding text.
- **Too large (> 1000 tokens):** Includes noise — irrelevant information dilutes the useful content. Also consumes more of the context window.
- **Sweet spot:** 256-512 tokens for most applications. Experiment with your specific documents.

**Overlap:** Chunks should overlap by 10-20% to prevent information loss at boundaries. If a key sentence falls at a chunk boundary, the overlap ensures at least one chunk contains it fully.

### Metadata Preservation

Attach metadata to each chunk for filtering and attribution:
- **Source document:** title, URL, author, date
- **Section hierarchy:** chapter → section → subsection
- **Document type:** FAQ, product doc, legal brief, knowledge article
- **Timestamp:** When the document was last updated

Metadata enables hybrid retrieval: "Find chunks about billing from documents updated in the last 30 days."

---

## Embedding and Indexing

### Embedding Model Selection

| Model Category | Examples | Dimension | Best For |
|---------------|---------|-----------|----------|
| General-purpose | OpenAI text-embedding-3, Cohere embed-v3 | 256-3072 | General text, multi-domain |
| Retrieval-optimized | E5, BGE, GTE | 384-1024 | Semantic search, RAG |
| Domain-specific | Fine-tuned on legal/medical/code data | Varies | Domain-specific retrieval |
| Multilingual | multilingual-e5, Cohere multilingual | 384-1024 | Cross-language retrieval |

**Key decision:** General-purpose embeddings work surprisingly well for most applications. Fine-tune only when retrieval quality on your specific domain is demonstrably insufficient.

### Dimensionality Tradeoffs

| Dimension | Storage per 1M chunks | Search Latency | Quality |
|-----------|----------------------|----------------|---------|
| 384 | ~1.5 GB | Fastest | Good |
| 768 | ~3 GB | Fast | Better |
| 1536 | ~6 GB | Moderate | Best for general |
| 3072 | ~12 GB | Slower | Marginal improvement |

For most RAG applications, 384-768 dimensions offer the best cost-quality tradeoff.

### Vector Store Options

| Store | Type | Scaling | Best For |
|-------|------|---------|----------|
| FAISS | Library (in-process) | Single machine | Prototyping, small-medium scale |
| Pinecone | Managed service | Auto-scaling | Production, no infra management |
| Weaviate | Self-hosted or managed | Horizontal | Hybrid search, complex filtering |
| Qdrant | Self-hosted or managed | Horizontal | High-performance, custom scoring |
| pgvector | Postgres extension | Single machine | Existing Postgres infra, small-medium |
| ChromaDB | Library / server | Single machine | Prototyping, local development |

### Hybrid Search

Combine semantic search (vector similarity) with keyword search (BM25). Often outperforms either alone.

**Why hybrid works:** Semantic search catches paraphrases and conceptual matches. Keyword search catches exact terms (product names, error codes, acronyms) that embedding models sometimes miss.

**Implementation:** Run both searches in parallel, combine scores with a weighted average:

`final_score = α × semantic_score + (1-α) × keyword_score`

Typical α: 0.5-0.7 (slightly favor semantic).

---

## Retrieval

### Top-k Retrieval

**How many chunks to retrieve?** Typically 3-10. More chunks = more context for the model, but also more noise and higher cost.

| k | Pros | Cons |
|---|------|------|
| 3-5 | Focused, low noise, low cost | May miss relevant information |
| 5-10 | Good coverage, balanced | Some noise, moderate cost |
| 10-20 | High recall | More noise, higher cost, may exceed context window |

### Re-Ranking

After initial vector retrieval, use a cross-encoder to re-rank the top-k chunks for better precision.

**Why:** Vector retrieval is fast but approximate. A cross-encoder (which processes the full query-chunk pair jointly) is more accurate but too slow to run over the entire corpus. Use vector search for recall (find potentially relevant chunks), then re-ranking for precision (pick the actually relevant ones).

**Popular re-rankers:** Cohere Rerank, BGE Reranker, cross-encoder models from sentence-transformers.

### Query Transformation

The user's query is often not the best search query. Transform it before retrieval:

| Technique | How It Works | When to Use |
|-----------|-------------|-------------|
| HyDE (Hypothetical Document Embeddings) | Generate a hypothetical answer, embed that, search for similar chunks | Queries that are questions (vs keyword searches) |
| Multi-query | Generate multiple reformulations of the query, retrieve for each, merge results | Ambiguous or multi-faceted queries |
| Step-back prompting | Rewrite the specific query as a more general one | Very specific queries that need broader context |
| Query decomposition | Split complex queries into sub-queries, retrieve for each | Multi-part questions |

### Metadata Filtering

Apply filters before or after vector search:
- **Pre-filtering:** Only search within documents matching metadata criteria. Faster but may miss relevant chunks in other categories.
- **Post-filtering:** Search all documents, then filter results. More thorough but slower.

---

## Generation with Retrieved Context

### Prompt Construction

How you present retrieved chunks to the LLM matters significantly:

```
System: You are a helpful assistant that answers questions based on the provided context.
Only use information from the context below. If the context doesn't contain the answer,
say "I don't have enough information to answer this question."

Context:
[Chunk 1: {{chunk_1_text}}]
Source: {{chunk_1_source}}

[Chunk 2: {{chunk_2_text}}]
Source: {{chunk_2_source}}

[Chunk 3: {{chunk_3_text}}]
Source: {{chunk_3_source}}

User question: {{user_query}}

Answer the question based on the context above. Cite your sources.
```

**Key prompt design decisions:**
- **Order:** Put the most relevant chunks first (models pay more attention to the beginning)
- **Source attribution:** Include source metadata so the model can cite specific documents
- **Faithfulness instruction:** Explicitly instruct the model to only use provided context
- **"I don't know" instruction:** Tell the model what to do when the context doesn't contain the answer

### Handling Contradictory Sources

When retrieved documents disagree, the model needs guidance:

- **Prefer recent:** "If sources conflict, prefer the most recently updated source."
- **Acknowledge conflict:** "If sources disagree, present both perspectives and note the disagreement."
- **Domain-specific rules:** "For medical information, always prefer peer-reviewed sources over blog posts."

### Context Window Management

When retrieved chunks exceed the context window:
- **Truncation:** Keep only the top-N most relevant chunks. Simple but may lose information.
- **Summarization:** Summarize retrieved chunks before including them. Preserves information but adds latency.
- **Map-reduce:** Process each chunk independently with the query, then combine the sub-answers. Handles large context at the cost of latency.
- **Hierarchical:** Retrieve at multiple granularities (paragraph-level for specific facts, document-level for context).

---

## Advanced RAG Patterns

### Multi-Hop RAG

Some questions require information from multiple documents that must be connected:

1. Retrieve initial documents for the query
2. Generate an intermediate answer or follow-up query
3. Retrieve additional documents based on the intermediate result
4. Generate the final answer from all retrieved context

**When to use:** Complex questions like "What's the total revenue from products mentioned in our Q3 earnings call?" (need to find the products, then find revenue for each).

### Agentic RAG

The LLM decides when to retrieve, what to retrieve, and how many times.

**How it works:**
1. LLM receives the user query
2. LLM decides: "Do I need to search?" If yes, generates a search query
3. Retrieval results are returned to the LLM
4. LLM evaluates: "Do I have enough information?" If not, generates another search query
5. Repeat until the LLM has sufficient context, then generate the final answer

**Advantages:** Adapts retrieval to query complexity. Simple questions → one retrieval. Complex questions → multiple retrievals.
**Risks:** More LLM calls → higher cost and latency. The LLM might retrieve unnecessarily or get stuck in loops.

### Graph RAG

Combine document retrieval with knowledge graph traversal.

**When to use:** When relationships between entities matter (organizational charts, product dependencies, legal case references). Vector similarity alone can't capture structured relationships.

### Self-RAG

The model evaluates its own retrieval quality and decides whether to use retrieved context, ignore it, or retrieve again.

**How it works:** Special tokens in the model's vocabulary signal:
- "Is retrieval needed?" → [Retrieve] / [No Retrieve]
- "Is this passage relevant?" → [Relevant] / [Irrelevant]
- "Is the response supported by the passage?" → [Supported] / [Not Supported]

---

## Evaluation

### Retrieval Quality

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Recall@K | Fraction of relevant chunks in top K | >90% at K=10 |
| Precision@K | Fraction of top K chunks that are relevant | >50% at K=5 |
| MRR | Position of first relevant chunk | As close to 1.0 as possible |
| Context relevance | % of retrieved context actually useful for the answer | >70% |

### Generation Quality

| Dimension | What It Measures | How to Evaluate |
|-----------|-----------------|----------------|
| Faithfulness | Does the answer match the retrieved context? (No hallucination) | LLM-as-judge or NLI model |
| Relevance | Does the answer address the user's question? | LLM-as-judge with rubric |
| Completeness | Does the answer cover all aspects of the question? | Human evaluation |
| Citation accuracy | Are citations correct? Do they support the claims? | Automated verification |

### End-to-End Evaluation

- **Human evaluation:** Gold standard. Have domain experts rate answers for accuracy and helpfulness.
- **LLM-as-judge:** Use a strong model to evaluate a weaker model's answers against a rubric.
- **User signals:** In production, track thumbs up/down, regeneration rate, follow-up question rate.

### Failure Analysis

| Failure Mode | Symptom | Root Cause | Fix |
|-------------|---------|-----------|-----|
| Retrieval miss | Correct information exists but wasn't retrieved | Poor chunking, embedding model mismatch, query-document vocabulary gap | Improve chunking, try hybrid search, query transformation |
| Context ignored | Retrieved relevant chunks but model didn't use them | Too much context, relevant chunk buried in noise, poor prompt | Reduce k, re-rank, improve prompt |
| Hallucination despite context | Model generates facts not in retrieved documents | Weak faithfulness instruction, model's parametric knowledge overrides | Stronger faithfulness prompt, lower temperature |
| Wrong source cited | Model attributes claim to wrong document | Multiple similar chunks, unclear source metadata | Improve metadata, fewer chunks |

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand the basic RAG pipeline: embed documents, store in a vector database, retrieve relevant chunks at query time, and include them in the LLM prompt. For a customer support bot, they should propose indexing help articles, retrieving the most relevant ones for each user query, and instructing the model to answer based on the retrieved context. They differentiate by mentioning chunk size as an important parameter and recognizing that the model should be instructed to say "I don't know" when the context doesn't contain the answer.

### Senior Engineer

Senior candidates can design a production RAG pipeline with depth. They discuss chunking strategies and tradeoffs, hybrid search (vector + keyword), re-ranking for precision, and prompt design for faithfulness. For a legal research assistant, a senior candidate would propose semantic chunking to preserve legal argument structure, hybrid search to catch specific case citations that semantic search might miss, and a re-ranking step to ensure the most relevant passages are prioritized. They bring up evaluation: retrieval recall, faithfulness checks, and the importance of monitoring hallucination rates in production.

### Staff Engineer

Staff candidates think about RAG as a system with failure modes and optimization opportunities. They recognize that the hardest RAG problem isn't retrieval — it's knowing when retrieval fails. A Staff candidate might propose a confidence calibration layer that estimates whether the retrieved context actually answers the query, and routes low-confidence queries to a human or to a multi-hop retrieval path. They also think about the operational aspects: how to keep the knowledge base current (ingestion pipeline freshness), how to handle document versioning (which version of a policy document is authoritative?), and how to detect knowledge base gaps (queries that consistently fail to find relevant context signal missing documentation).
