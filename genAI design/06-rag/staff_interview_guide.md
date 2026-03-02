# Retrieval-Augmented Generation (RAG) System — Staff/Principal Interview Guide

---

## How to Use This Guide

This guide is structured for interviewers and candidates preparing for staff- or principal-level ML design interviews. The interview is **45 minutes** total. Each section includes an **interviewer prompt**, the **signal being tested**, and **four-level model answers** representing the candidate response quality spectrum.

**Rating Levels:**
- **No Hire** — Fundamental misunderstanding or silence
- **Lean No Hire** — Partial understanding, significant gaps, needs heavy prompting
- **Lean Hire** — Correct understanding, hits main points, minor gaps
- **Strong Hire** — Deep, nuanced, first-principles reasoning, proactively addresses trade-offs, demonstrates platform-level thinking

**Interviewer Notes:**
- Spend the first minute reading the prompt aloud and giving the candidate time to think silently.
- Do not volunteer information unless the candidate is stuck for more than 90 seconds.
- Use the follow-up probes listed under each section to differentiate Hire from Strong Hire.
- The principal-level bar requires connecting individual design decisions to broader organizational or platform impact.

**Time Budget:**

| Section | Time |
|---|---|
| Problem Statement & Clarification | 5 min |
| ML Problem Framing | 5 min |
| Data & Preprocessing | 8 min |
| Model Architecture Deep Dive | 12 min |
| Evaluation | 5 min |
| Serving Architecture | 7 min |
| Edge Cases & Failure Modes | 5 min |
| Principal-Level Platform Thinking | 3 min |

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

> "Design a Retrieval-Augmented Generation system for an enterprise knowledge base assistant. The system should answer employee questions by finding and synthesizing information from internal company documents. Walk me through your approach."

### Signal Being Tested

Does the candidate recognize the core RAG tension (retrieval quality determines answer quality) and ask the right questions about document corpus, freshness requirements, and faithfulness guarantees?

### Six Clarification Dimensions

| Dimension | Why It Matters |
|---|---|
| **Corpus size and type** | 10K docs vs. 10M docs; PDFs/HTML/Slack — shapes indexing pipeline |
| **Freshness requirements** | Real-time updates vs. nightly re-index — determines streaming vs. batch indexing |
| **Faithfulness requirements** | Legal/compliance contexts need strict grounding; factual drift is unacceptable |
| **Multi-hop reasoning** | Single document answers vs. synthesis across multiple documents |
| **Access control** | Employee should only see documents they have permission to access |
| **Latency SLA** | Interactive Q&A (< 2s) vs. async report generation (minutes OK) |

### Follow-up Probes

- "What changes about your design if the documents update every minute vs. once a day?"
- "How do you handle the case where no documents in the corpus answer the question?"
- "What is the risk of providing wrong information in an enterprise context, and how does that shape your design?"

---

### Model Answers — Section 1

**No Hire:**
"I would fine-tune GPT on the company's documents." No understanding of retrieval, grounding, or why fine-tuning alone is insufficient for document-grounded Q&A.

**Lean No Hire:**
Understands that RAG involves retrieval + generation but doesn't probe for faithfulness requirements, access control, or the corpus update problem.

**Lean Hire:**
Asks about corpus size, update frequency, and faithfulness. Recognizes that access control is a non-negotiable enterprise requirement. Notes that multi-hop reasoning (synthesizing across multiple documents) is harder than single-document lookup.

**Strong Hire Answer (first-person):**

RAG's core value proposition is grounding LLM outputs in retrieved evidence, which reduces hallucination and enables citing sources. But the system is only as good as its retrieval quality — if we retrieve wrong or irrelevant documents, the LLM will generate wrong answers that sound confident.

Let me clarify six dimensions before proceeding.

First, corpus characteristics. How many documents? What formats (PDF, Word, Confluence, Slack, email)? Each format requires a different parser. For 10K documents, a single-server vector index is fine. For 10M documents, I need a distributed vector store (Pinecone, Weaviate, PG Vector with horizontal sharding). The document structure also matters — a well-structured technical manual with headings and sections can be chunked more intelligently than a stream of emails.

Second, freshness. Enterprise knowledge changes: policies update, products change, org charts evolve. If the document index is stale, the system answers questions with outdated information — sometimes worse than no answer at all. Is nightly re-indexing sufficient, or do I need real-time streaming ingestion?

Third, faithfulness requirements. In a legal or compliance context, the system should not synthesize claims that aren't explicitly stated in the retrieved documents. Hallucination is not just embarrassing — it could create legal liability ("Our AI said the contract says X"). Faithfulness needs to be both trained for and evaluated continuously.

Fourth, multi-hop reasoning. Some questions require synthesizing information from multiple documents. "What is our refund policy for international orders?" may require looking at both the refund policy document and the international shipping document. Simple single-document retrieval won't work.

Fifth, access control. An employee in Marketing should not receive answers sourced from confidential HR documents. The retrieval must respect document-level permissions tied to the user's identity. This is a non-negotiable enterprise requirement.

Sixth, latency. Interactive chat requires < 2s end-to-end. Retrieval + generation typically takes 0.5–1s for retrieval and 1–3s for generation with a large LLM. Meeting the SLA requires careful optimization.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "How do you formally frame RAG as an ML problem? What are the learned and non-learned components?"

### Signal Being Tested

Does the candidate understand that RAG has three ML components (retrieval model, generation model, and optionally a re-ranker) and that each can be separately optimized?

### Follow-up Probes

- "What does the retrieval model learn? What is its training objective?"
- "Is the retrieval model frozen or jointly trained with the generation model? What are the trade-offs?"
- "What is the difference between sparse retrieval (BM25) and dense retrieval (DPR)? When does each win?"

---

### Model Answers — Section 2

**No Hire:**
"I would use keyword search to find documents and feed them to GPT." Cannot frame as an ML problem or explain dense retrieval.

**Lean No Hire:**
Knows dense retrieval exists but cannot describe bi-encoder vs. cross-encoder architectures or their trade-offs.

**Lean Hire:**
Correctly describes the RAG pipeline: query encoder → nearest-neighbor retrieval → LLM generation with retrieved context. Can explain DPR bi-encoder training with in-batch negatives.

**Strong Hire Answer (first-person):**

RAG has three ML components that can each be independently optimized:

**Component 1: Retrieval Model (Bi-encoder)**
The retrieval model embeds the user query and all documents into a shared embedding space, then finds the top-k documents by cosine similarity. This is a bi-encoder architecture: a query encoder `E_q` and a document encoder `E_d` (typically BERT-based, or a smaller transformer):
```
score(q, d) = E_q(q) · E_d(d) / (||E_q(q)|| · ||E_d(d)||)
```
Training uses contrastive learning (DPR — Dense Passage Retrieval):
```
L_retrieval = -log [exp(s(q, d+)/τ) / (exp(s(q, d+)/τ) + Σ_j exp(s(q, d_j^-)/τ))]
```
where d+ is the relevant document and d_j^- are in-batch negatives.

The retrieval model must be extremely fast (millions of documents in milliseconds) — this is why we use a bi-encoder (pre-compute all document embeddings offline) rather than a cross-encoder (which processes query+document jointly and cannot be pre-computed).

**Component 2: Re-ranker (Cross-encoder, optional)**
After retrieving top-50 candidates with the bi-encoder, a cross-encoder re-ranker refines the ordering by processing (query, document) pairs jointly:
```
score_rerank(q, d) = BERT([CLS] q [SEP] d [SEP])[CLS projection → scalar]
```
Cross-encoders are 10–100× slower than bi-encoders but much more accurate (they see both query and document together, enabling fine-grained relevance estimation). Running a cross-encoder on top-50 candidates (instead of all M documents) makes this tractable.

**Component 3: LLM Generator**
The LLM receives the top-k retrieved documents concatenated with the query:
```
prompt = f"Context: {doc_1}\n{doc_2}\n...\n{doc_k}\nQuestion: {query}\nAnswer:"
p(answer | query, docs) = LLM(prompt)
```
The LLM is typically a frozen pretrained model; RAG-specific fine-tuning can improve faithfulness but is expensive.

**Sparse vs. Dense Retrieval:**
- *BM25 (sparse)*: TF-IDF based bag-of-words matching; excellent for exact keyword matches; no neural inference required; fails for semantic similarity ("fix" vs. "repair" are unrelated to BM25)
- *DPR (dense)*: semantic similarity in embedding space; handles paraphrases and synonyms; requires neural inference; fails for rare or highly specific terminology not in training distribution

Hybrid: run both BM25 and DPR in parallel, then combine scores (RRF — Reciprocal Rank Fusion) for the retrieval stage. This typically outperforms either alone.

---

## Section 3: Data & Preprocessing (8 min)

### Interviewer Prompt

> "Walk me through the document indexing pipeline. How do you preprocess documents before storing them in the vector index?"

### Signal Being Tested

Does the candidate understand chunking strategies, embedding models, and the preprocessing steps needed for different document types?

### Follow-up Probes

- "How do you decide on chunk size? What are the trade-offs?"
- "What is the challenge with PDFs specifically?"
- "How do you handle documents with tables, code blocks, or figures?"

---

### Model Answers — Section 3

**No Hire:**
"I would embed each document and store it." No understanding of chunking or why full-document embedding is insufficient.

**Lean No Hire:**
Knows chunking is needed but cannot explain the chunk size trade-off or describe preprocessing for different document formats.

**Lean Hire:**
Describes paragraph-level chunking, overlap between chunks, and the chunk size trade-off (larger chunks have more context but dilute the embedding signal). Can describe PDF parsing challenges.

**Strong Hire Answer (first-person):**

The document ingestion pipeline is the unglamorous but critical component of RAG. Poor chunking and preprocessing leads to retrieval failures even with a perfect embedding model.

**Document parsing by format:**
- *HTML/Markdown*: use structure-aware parsers (BeautifulSoup, Python-Markdown). Preserve headings as metadata.
- *PDF*: the hardest format. PDFs are layout-based, not semantically structured. `pdfplumber` or `pymupdf` extract text, but column layouts, headers/footers, footnotes, and figures require heuristic handling. Tables in PDFs are particularly challenging — they often parse as garbled text.
- *Office documents (Word, PowerPoint)*: `python-docx` preserves structure; slide text often lacks context.
- *HTML with tables*: convert tables to a serialized text representation (column1: value1, column2: value2) or markdown table format.

**Chunking strategy:**
The fundamental trade-off: smaller chunks = better embedding precision (the embedding captures a specific fact); larger chunks = more context for the LLM to synthesize (answer requires surrounding context).

Typical production approach:
- *Semantic chunking*: split at paragraph or section boundaries rather than fixed token counts. A 300-token paragraph with a coherent topic is better than a 300-token window that splits mid-sentence.
- *Chunk size*: 256–512 tokens per chunk is a common sweet spot. Each chunk should answer one "atomic" question.
- *Overlap*: 10–20% overlap between consecutive chunks prevents information loss at chunk boundaries.

**Metadata attachment:**
Each chunk is stored with metadata: {document_id, document_title, section_heading, page_number, author, creation_date, access_control_list}. This metadata enables:
- Access control: filter by user permissions before returning chunks
- Freshness: filter out chunks from documents updated before a certain date
- Source citation: show the user exactly which document/section each answer comes from

**Embedding model selection:**
For enterprise knowledge bases, I use a domain-adapted embedding model. General-purpose embeddings (OpenAI text-embedding-ada-002, sentence-transformers) work for general language but may underperform on specialized domains (legal, medical, technical). I evaluate embedding models on a domain-specific retrieval benchmark before deploying.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "Walk me through the full RAG architecture — from query to answer. Be specific about the retrieval step, the re-ranking step, and how context is injected into the LLM."

### Signal Being Tested

Does the candidate understand the full RAG pipeline including HNSW approximate nearest-neighbor search, re-ranking, prompt construction, and the generation step? Can they explain why each component is necessary?

### Follow-up Probes

- "How does HNSW (approximate nearest-neighbor) work? What are its precision vs. recall trade-offs?"
- "Why do you need a re-ranker if your bi-encoder is already good?"
- "How do you handle the case where the top-k documents contain contradictory information?"

---

### Model Answers — Section 4

**No Hire:**
"I would use cosine similarity to find documents." Cannot describe HNSW or re-ranking.

**Lean No Hire:**
Describes the retrieval pipeline at a high level but cannot explain HNSW, the bi-encoder/cross-encoder distinction, or prompt construction.

**Lean Hire:**
Correctly explains bi-encoder retrieval → cross-encoder re-ranking → LLM generation. Can describe HNSW as approximate nearest-neighbor. Describes prompt construction with retrieved context.

**Strong Hire Answer (first-person):**

Let me walk through the full pipeline from user query to final answer.

**Step 1: Query Encoding**
The user's query is embedded using the bi-encoder query encoder:
```
q_emb = E_q(query) ∈ R^{768}
```
This takes ~5ms with a small (110M parameter) BERT-based model.

**Step 2: Approximate Nearest-Neighbor Retrieval (HNSW)**
The document index stores pre-computed embeddings for all chunks. We retrieve top-k=50 candidates by cosine similarity using HNSW (Hierarchical Navigable Small World) graph:
```
top_50_chunks = HNSW_index.search(q_emb, k=50)
```
HNSW builds a multi-layer graph where each node (document embedding) is connected to its approximate nearest neighbors at each layer. Search traverses the graph greedily from the top (sparse) layer to the bottom (dense) layer. Key properties:
- Query time: O(log N) vs. O(N) for brute force
- Recall@50: typically 95–99% vs. 100% for exact search (acceptable trade-off)
- Memory: the graph structure adds ~3–5× overhead over raw embeddings

**Step 3: Re-Ranking (Cross-Encoder)**
From 50 candidates, a cross-encoder scores each (query, chunk) pair jointly:
```
relevance_i = CrossEncoder([CLS] query [SEP] chunk_i [SEP])
```
The cross-encoder can see the full interaction between query and chunk — much more accurate than the bi-encoder's inner product. We keep top-k=5 after re-ranking.

The bi-encoder + cross-encoder combination is the industry standard (first-stage dense retrieval + second-stage neural re-ranking) for production RAG systems.

**Step 4: Access Control Filtering**
Before passing to the LLM, apply user permission filtering: remove any chunk where the document's access_control_list does not include the current user. This must happen after re-ranking (so we retrieve the most relevant documents first) but before LLM generation (so we don't leak unauthorized content).

**Step 5: Prompt Construction**
```
system_prompt = "Answer the question based only on the provided context. If the context doesn't contain the answer, say 'I don't have information about this.'"

context_block = "\n\n".join([
    f"[Source: {chunk.doc_title}, p.{chunk.page}]\n{chunk.text}"
    for chunk in top_k_chunks
])

final_prompt = f"{system_prompt}\n\nContext:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
```

The "based only on the provided context" instruction is the key faithfulness constraint — it instructs the model not to hallucinate beyond the retrieved evidence.

**Step 6: LLM Generation**
The prompt (typically 2000–5000 tokens) is processed by the LLM (GPT-4, Claude, or open-source equivalent). The LLM generates a response that synthesizes information from the retrieved chunks.

**Handling contradictory documents:**
When retrieved documents contain conflicting information (document A says policy X, document B says policy Y), the LLM should report the contradiction rather than picking one. I add an explicit instruction: "If the context contains contradictory information, present both views and note the contradiction." This is especially important for policy documents with different version dates.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate a RAG system? What metrics do you use for retrieval quality and generation quality separately?"

### Signal Being Tested

Does the candidate understand RAG-specific metrics (MRR, NDCG, faithfulness, answer relevance) and the importance of evaluating retrieval and generation independently?

### Follow-up Probes

- "What is MRR and how does it differ from precision@k?"
- "How do you measure faithfulness — whether the answer is grounded in retrieved documents?"
- "What is RAGAs and how would you use it?"

---

### Model Answers — Section 5

**No Hire:**
"I would check if the answers are correct." Cannot describe retrieval-specific or faithfulness-specific metrics.

**Lean No Hire:**
Mentions accuracy or BLEU. Cannot distinguish retrieval evaluation from generation evaluation or describe faithfulness metrics.

**Lean Hire:**
Correctly describes MRR and NDCG for retrieval; mentions faithfulness as a generation quality metric. Knows that retrieval and generation should be evaluated independently.

**Strong Hire Answer (first-person):**

RAG evaluation must separately measure retrieval quality and generation quality — a failure in either component leads to bad answers, but the fix is different depending on which component failed.

**Retrieval Evaluation:**

*Recall@k*: what fraction of the time is the correct document in the top-k retrieved results? For a Q&A system, Recall@5 (is the answer in the top 5 retrieved chunks?) is the primary retrieval metric.

*MRR (Mean Reciprocal Rank)*: average of 1/rank across queries, where rank is the position of the first relevant document:
```
MRR = (1/|Q|) Σ_{q=1}^{|Q|} 1/rank_q
```
MRR ranges from 0 to 1; higher is better. MRR=1.0 means every query's relevant document is the top result. MRR differs from Precision@k by rewarding systems that rank relevant documents higher.

*NDCG@k (Normalized Discounted Cumulative Gain)*: accounts for graded relevance (highly relevant vs. somewhat relevant) and position discount:
```
DCG@k = Σ_{i=1}^{k} rel_i / log_2(i+1)
NDCG@k = DCG@k / IDCG@k (IDCG = ideal ordering)
```
For RAG, NDCG is less commonly used than MRR because relevance labels are typically binary (relevant/not relevant).

**Generation Evaluation:**

*Faithfulness*: is every claim in the answer supported by the retrieved documents? Evaluated using an NLI (Natural Language Inference) model: for each sentence in the answer, check if it is entailed by at least one retrieved chunk.
```
Faithfulness = |claims_entailed_by_context| / |total_claims_in_answer|
```
Faithfulness should be > 0.95 for enterprise use cases. Unfaithful claims = hallucination.

*Answer Relevance*: is the answer responsive to the question? Evaluated by generating questions from the answer and checking if they match the original question (using a question generation model + similarity metric).

*Context Precision*: are the retrieved chunks actually relevant to the question? Prevents the model from being "correctly faithful to wrong documents."

**RAGAs framework:**
RAGAs (Retrieval Augmented Generation Assessment) provides a standardized evaluation framework that computes faithfulness, answer relevance, context recall, and context precision using an LLM as judge. I use RAGAs for automated monitoring and combine it with human evaluation on a monthly sample.

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "Walk me through the serving infrastructure for a RAG system handling thousands of concurrent enterprise queries."

### Signal Being Tested

Does the candidate understand the latency budget split among retrieval, re-ranking, and LLM generation? Can they describe the vector database serving requirements and caching strategies?

### Follow-up Probes

- "How do you partition the latency budget across the three stages (retrieval, re-ranking, LLM generation)?"
- "What can you cache in a RAG system? What can't you cache?"
- "How do you keep the vector index fresh when documents update?"

---

### Model Answers — Section 6

**No Hire:**
"I would run the pipeline on a server." No understanding of latency breakdown or vector index operations.

**Lean No Hire:**
Describes the pipeline at a high level but cannot provide latency estimates or describe vector index serving (HNSW, sharding).

**Lean Hire:**
Correctly estimates latency breakdown, describes HNSW serving, and identifies what can be cached (query embeddings, retrieved chunks). Notes document freshness as a separate infrastructure concern.

**Strong Hire Answer (first-person):**

The total latency budget for interactive RAG is ~2 seconds. Let me break down how it is allocated:

**Latency budget (target: 2s end-to-end):**
- Query embedding: ~10ms (small bi-encoder)
- HNSW retrieval (top-50): ~20ms (sub-millisecond per query at steady state, ~20ms including overhead)
- Cross-encoder re-ranking (50→5): ~100ms (cross-encoder over 50 (query, chunk) pairs)
- Access control filtering: ~5ms
- Prompt construction: ~5ms
- LLM generation (500 tokens): ~1000–1500ms (dominant cost)
- **Total: ~1200–1650ms** — within 2s budget

**Vector Index Serving:**
I use a managed vector database (Pinecone, Weaviate, or Milvus) for production. Key requirements:
- HNSW index with `ef_search=128` parameter (controls recall vs. speed)
- Horizontal sharding across multiple nodes for large corpora (> 10M documents)
- Metadata filtering at index time (access control, freshness)
- Replication for high availability (3 replicas minimum)

**Document freshness pipeline:**
When a document is updated:
1. Re-chunk the updated document
2. Delete old chunks from vector index (by document_id)
3. Re-embed and insert new chunks
4. Update metadata in the document store

For real-time updates, use a streaming pipeline (Kafka → embedding service → vector store). For nightly batch updates, a simpler batch job suffices. The freshness SLA drives the pipeline complexity.

**Caching strategy:**
- *What to cache*: query embeddings (popular queries are re-asked frequently; LRU cache of embedding → top-k chunks), LLM responses for exact query matches (if content is static), retrieved chunks for frequently accessed documents.
- *What not to cache*: responses that depend on dynamic context (user-specific documents), or any response whose source documents have been updated since the cached response was generated.

**LLM serving optimization:**
The LLM generation is the bottleneck. Optimizations: prompt prefix caching (cache the encoded representation of the system prompt), speculative decoding, and INT8 quantization. For very high QPS, deploy a dedicated LLM serving cluster (vLLM with continuous batching) rather than using a third-party API that may introduce unpredictable tail latency.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the most critical failure modes of a RAG system in an enterprise context?"

### Signal Being Tested

Does the candidate identify retrieval failures (no relevant document), faithfulness failures (hallucination despite retrieval), and access control failures (showing unauthorized content)?

### Follow-up Probes

- "What happens when no document in the corpus answers the question? How do you detect this?"
- "How do you prevent the LLM from ignoring the retrieved context and relying on its parametric knowledge?"
- "What is a 'lost in the middle' failure and how do you mitigate it?"

---

### Model Answers — Section 7

**No Hire:**
Cannot describe RAG-specific failure modes. Generic "wrong answers."

**Lean No Hire:**
Mentions "retrieval failure" but cannot describe it mechanically or propose detection/mitigation strategies.

**Lean Hire:**
Identifies no-document, hallucination-despite-context, and lost-in-the-middle failures. Proposes confidence thresholding and faithfulness classifiers.

**Strong Hire Answer (first-person):**

RAG has failure modes at both the retrieval and generation layers.

**1. Retrieval failure (no relevant document):**
The correct answer is not in the knowledge base, or the embedding model fails to retrieve the relevant chunk (vocabulary mismatch, poorly chunked documents). The LLM then either halluculates or says "I don't know."

Detection: retrieval confidence score — if the highest-similarity chunk has cosine similarity < 0.7, flag the query for "low confidence" and either return a "no information found" response or escalate to a human agent. Never force a generation step on irretrievable queries.

**2. Faithfulness failure (hallucination despite good retrieval):**
The LLM ignores the retrieved context and generates from its parametric knowledge. This can happen when: the retrieved context contradicts the model's parametric beliefs (the model "trusts" itself over the evidence), the context is too long and the model skips parts, or the model generalizes beyond what the context supports.

Mitigation: (a) explicit faithfulness instruction in the system prompt; (b) post-generation faithfulness check using NLI classifier; (c) fine-tune the LLM on RAG-specific examples to improve context following.

**3. Lost in the Middle:**
Research shows that LLMs pay less attention to information in the middle of long contexts — they attend to the beginning and end of the context most reliably. If the relevant chunk is at position 3 of 5, it may be partially ignored.

Mitigation: re-order retrieved chunks so the most relevant (highest cross-encoder score) appear first and last, not in the middle.

**4. Access control leakage:**
A bug in the permission filtering step causes a user to receive answers sourced from documents they are not authorized to see. This is a critical security failure in enterprise settings.

Mitigation: defense in depth — apply permission filtering at both the retrieval layer (HNSW metadata filter) and the prompt construction layer (verify permissions before including each chunk). Log every query with the source documents used; audit logs for unauthorized access patterns.

**5. Stale information:**
The document corpus is not updated in time; the system answers based on outdated policy documents. A user asking about the current refund policy receives an answer from a policy document that was superseded three months ago.

Mitigation: include document creation/modification date in source citations; if answering from a document older than a freshness threshold (e.g., 6 months), include a disclaimer: "Note: this information is from a document last updated X months ago. Please verify with HR."

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "You've built a RAG system for one enterprise customer. Now you're building a multi-tenant RAG platform serving 100 enterprise customers, each with their own private knowledge base. What are the most important architectural decisions?"

### Signal Being Tested

Does the candidate think about tenant isolation, shared infrastructure economics, and the challenges of multi-tenant vector indexing?

### Follow-up Probes

- "How do you isolate tenant data in a shared vector store?"
- "What shared infrastructure provides the most cost leverage across tenants?"

---

### Model Answers — Section 8

**No Hire:**
"Give each customer their own server." No consideration of shared infrastructure economics.

**Lean No Hire:**
Suggests shared embedding model but doesn't address tenant isolation in the vector store or the cost leverage of shared LLM serving.

**Lean Hire:**
Describes per-tenant namespaces in the vector store, shared embedding and LLM serving infrastructure, and access control as a per-tenant configuration.

**Strong Hire Answer (first-person):**

Multi-tenant RAG has two competing requirements: strong tenant isolation (customer A never sees customer B's data) and infrastructure sharing for cost efficiency.

**Tenant isolation in the vector store:**
Each tenant gets a dedicated namespace (Pinecone namespaces, Weaviate classes, Milvus partitions) within a shared vector cluster. Queries for tenant A only search within their namespace. For customers with strict compliance requirements (HIPAA, SOC2), dedicated indexes (not just namespaces) are needed.

**Shared infrastructure for cost leverage:**
- *Embedding model*: one shared embedding service handles all tenants. Embedding is compute-intensive but customer-agnostic — the same model produces embeddings for all knowledge bases.
- *LLM serving*: one shared LLM serving cluster (with continuous batching) handles generation for all tenants. The system prompt is the only per-tenant variation.
- *Evaluation pipeline*: shared RAGAs evaluation harness runs across all tenants; aggregate metrics detect when system-wide quality changes.

**Per-tenant customization:**
- Custom embedding models for domain-specific tenants (a medical tenant uses a medically pre-trained embedding model)
- Custom chunk sizes and indexing strategies per tenant (legal documents benefit from different chunking than Slack messages)
- Per-tenant freshness schedules (financial services needs real-time; others can tolerate daily)

**Security model:**
Never mix vectors from different tenants in the same index partition. Any metadata filtering bug could return cross-tenant results. The partition boundary is the primary isolation mechanism; secondary is query-time tenant ID validation.

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**Bi-encoder retrieval score:**
```
score(q, d) = E_q(q) · E_d(d) / (||E_q(q)|| · ||E_d(d)||)
```

**DPR contrastive training loss:**
```
L_retrieval = -log [exp(s(q, d+)/τ) / (exp(s(q, d+)/τ) + Σ_j exp(s(q, d_j^-)/τ))]
```

**MRR (Mean Reciprocal Rank):**
```
MRR = (1/|Q|) Σ_{q=1}^{|Q|} 1/rank_q
```

**NDCG@k:**
```
DCG@k = Σ_{i=1}^{k} rel_i / log_2(i+1)
NDCG@k = DCG@k / IDCG@k
```

**Faithfulness (NLI-based):**
```
Faithfulness = |{s ∈ answer : ∃ c ∈ context, NLI(c, s) = entail}| / |sentences(answer)|
```

**RAG generation prompt:**
```
p(answer | q, docs) = LLM("Context: {docs}\nQuestion: {q}\nAnswer:")
```

**Reciprocal Rank Fusion (hybrid BM25 + dense):**
```
RRF(q, d) = Σ_i 1 / (k + rank_i(d))
where k=60 (smoothing constant), sum over retrieval methods
```

**Precision@k:**
```
P@k = |relevant docs in top-k| / k
```

### Vocabulary Cheat Sheet

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation; grounds LLM output in retrieved documents |
| **Bi-encoder** | Separate encoders for query and document; enables pre-computation |
| **Cross-encoder** | Joint encoding of (query, document); more accurate, slower |
| **HNSW** | Hierarchical Navigable Small World; approximate nearest-neighbor graph |
| **DPR** | Dense Passage Retrieval; bi-encoder with contrastive training |
| **BM25** | Sparse TF-IDF based retrieval; strong for exact keyword matches |
| **MRR** | Mean Reciprocal Rank; average 1/rank of first relevant document |
| **NDCG** | Normalized Discounted Cumulative Gain; position-discounted graded relevance |
| **Faithfulness** | Fraction of answer claims entailed by retrieved context |
| **Answer Relevance** | Does the answer address the question asked? |
| **RAGAs** | RAG Assessment framework; standardized faithfulness + relevance evaluation |
| **Chunking** | Splitting documents into segments for embedding and retrieval |
| **Namespace** | Logical partition in vector store for tenant isolation |
| **Lost in the middle** | LLMs attend less to context in middle positions of long prompts |
| **RRF** | Reciprocal Rank Fusion; combines multiple retrieval system rankings |
| **NLI** | Natural Language Inference; determines if one text entails/contradicts another |

### Key Numbers Table

| Metric | Value |
|---|---|
| Target end-to-end latency (interactive) | < 2s |
| Query embedding latency (bi-encoder) | ~10ms |
| HNSW retrieval latency | ~20ms |
| Cross-encoder re-ranking (50 chunks) | ~100ms |
| LLM generation (500 tokens) | ~1000–1500ms |
| Good MRR for enterprise RAG | > 0.85 |
| Good Recall@5 for enterprise RAG | > 0.90 |
| Target faithfulness score | > 0.95 |
| Typical chunk size | 256–512 tokens |
| Chunk overlap | 10–20% |
| HNSW recall@50 vs. brute force | 95–99% |
| Recommended top-k for retrieval | 50 (before re-ranking) |
| Recommended top-k for LLM context | 3–5 (after re-ranking) |

### Rapid-Fire Day-Before Review

1. **Bi-encoder vs. cross-encoder?** Bi-encoder: pre-compute embeddings, fast retrieval; cross-encoder: joint encoding, accurate re-ranking but slow
2. **Why is RAG better than fine-tuning for factual Q&A?** RAG cites sources, handles document updates without retraining, less hallucination
3. **MRR formula?** (1/|Q|) Σ 1/rank_q — average reciprocal rank of first relevant document
4. **What is faithfulness?** Fraction of answer claims that are entailed by the retrieved context (NLI-based)
5. **Lost in the middle?** LLMs pay less attention to middle context; put most relevant chunks at beginning/end
6. **BM25 vs. DPR trade-off?** BM25 wins on exact keyword matches; DPR wins on semantic similarity; hybrid (RRF) beats both
7. **HNSW property?** O(log N) query time, 95–99% recall vs. brute force, ~3–5× memory overhead
8. **How to handle no-document queries?** Flag as low confidence (cosine similarity < 0.7); return "no information found" rather than hallucinating
9. **Access control in multi-tenant?** Per-tenant namespace + query-time tenant ID validation + audit logging
10. **RAGAs framework metrics?** Faithfulness, answer relevance, context precision, context recall
