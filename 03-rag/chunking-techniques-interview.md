> **What this file covers**
> - 🎯 Why chunk size is a precision-recall trade-off with a measurable optimum
> - 🧮 Embedding quality vs chunk size: how the centroid effect degrades retrieval
> - ⚠️ 4 failure modes: mid-sentence cuts, topic contamination, density mismatch, coreference loss
> - 📊 Complexity analysis for all 5 strategies — time, memory, scaling bottlenecks
> - 💡 Overlap cost-benefit: storage vs boundary coverage, diminishing returns
> - 🏭 Production: per-document-type tuning, metadata strategy, incremental re-indexing
> - Staff/Principal Q&A with all four hiring levels shown (6 questions)

---

# Chunking Techniques — Interview Deep-Dive

This file assumes you have read [chunking-techniques.md](./chunking-techniques.md) and understand the six strategies (fixed-size, overlapping, sentence-based, paragraph-based, semantic, recursive), the Goldilocks problem, and why chunking matters for retrieval quality. Everything here is for Staff/Principal depth.

---

## 🧮 Embedding Quality vs Chunk Size

A chunk's embedding is a single point in vector space that represents the entire chunk's meaning. When a chunk contains one focused topic, that point is precise. When a chunk contains multiple topics, the embedding becomes the centroid of those topics — a point in between that represents none of them well.

```
🧮 The centroid effect:

    embedding(chunk) ≈ mean(embedding(sentence_1), ..., embedding(sentence_k))

    For a chunk with k sentences about topic A:
      → embedding lands squarely in the "topic A" region
      → high cosine similarity with queries about topic A

    For a chunk with sentences about topics A and B:
      → embedding lands between the "topic A" and "topic B" regions
      → moderate cosine similarity with queries about either topic
      → neither query finds this chunk as a top result
```

This is why chunk size matters mathematically: larger chunks are more likely to span multiple topics, producing centroid embeddings that match no query well.

**Measuring the effect:** Compute pairwise cosine similarity between all sentences within a chunk. If the minimum intra-chunk similarity is below 0.4, the chunk likely spans a topic boundary. The embedding for that chunk is a poor representative of either topic.

---

## 🧮 Overlap Cost-Benefit Analysis

Overlap prevents information loss at chunk boundaries. The trade-off is storage and index size.

```
🧮 Storage overhead from overlap:

    Without overlap:
      n_chunks = ceil(D / C)
      total_stored = D

    With overlap fraction α (e.g., α = 0.15 for 15%):
      step_size = C × (1 - α)
      n_chunks = ceil(D / step_size)
      total_stored ≈ D / (1 - α)

    Example: D = 100K chars, C = 300, α = 0.15
      Without overlap: 334 chunks, 100K chars stored
      With 15% overlap: 393 chunks (+18%), 118K chars stored (+18%)

    The overhead is exactly 1/(1-α) - 1:
      10% overlap → 11% more storage
      15% overlap → 18% more storage
      20% overlap → 25% more storage
      50% overlap → 100% more storage (every char stored twice)
```

**Diminishing returns:** Overlap beyond 20% rarely improves retrieval recall. The boundary region that benefits from overlap is typically one sentence (15-30 tokens). Once overlap exceeds one sentence length, the additional overlap duplicates content that is already well-represented in the chunk interior. Empirically, 10-20% overlap captures 90%+ of the boundary coverage benefit.

---

## 🗺️ Chunk Size as Precision-Recall Trade-off

```
                    Precision                        Recall
                        ▲                              ▲
                   1.0  │  ╲                      1.0  │        ╱
                        │    ╲                         │      ╱
                        │      ╲                       │    ╱
                        │        ╲                     │  ╱
                   0.0  │──────────►              0.0  │──────────►
                        small → large                  small → large
                        chunk size                     chunk size

    Small chunks: high precision (focused), low recall (answer may be split)
    Large chunks: low precision (noisy), high recall (answer likely included)
    Optimal: the chunk size that maximizes F1 on your query distribution
```

🎯 **Key insight:** chunk size is a hyperparameter, not a constant. The optimal value depends on your query distribution, document type, and embedding model. Treat 200-500 tokens as a starting point, then tune against an evaluation set.

---

## ⚠️ Failure Modes

### 1. Mid-Sentence Cuts (Semantic Incompleteness)

Fixed-size chunking can split a sentence across two chunks. The resulting fragments are incomplete semantic units. Transformer-based encoders (BERT variants) are trained on complete sentences; the [CLS] token's representation degrades when input truncates mid-sentence.

**Impact:** 5-15% drop in MRR on datasets with precise factual questions.

**Detection:** Check if chunk text starts with a lowercase letter or ends without terminal punctuation.

**Fix:** Use sentence-aware chunking or add overlap of at least one sentence length.

### 2. Topic Contamination (Centroid Embedding)

A chunk that spans a paragraph boundary between Topic A and Topic B produces an embedding that is the centroid of both. This is silently terrible — the embedding will not be the nearest neighbor for either topic query.

**Detection:** Compute pairwise cosine similarity between sentences within each chunk. If minimum intra-chunk similarity is below 0.4, the chunk probably spans a topic shift.

**Fix:** Use semantic chunking to detect topic boundaries, or use paragraph-based splitting.

### 3. Information Density Mismatch

Fixed-size chunking treats all text as having the same information density per token. In practice, legal contracts, academic abstracts, and medical notes have 3-5x the information density of narrative prose. A single chunk size across document types guarantees that one of them is miscalibrated.

**Detection:** Compare retrieval precision across document types. If one type consistently underperforms, its chunk size is wrong.

**Fix:** Per-document-type chunk size tuning, driven by a held-out evaluation set.

### 4. Coreference Loss

Narrative text uses backward coreference — pronouns that refer to entities introduced in the previous paragraph. A chunk that starts with "He said the plan was approved" is useless without knowing who "He" is. The embedding captures the action but not the actor.

**Detection:** Check if chunks begin with pronouns (he, she, it, they, this, that).

**Fix:** Increase overlap to carry forward entity mentions. 20-25% overlap typically captures the referent.

---

## 📊 Complexity Analysis

| Strategy | Chunking Time | Embedding Time | Total Dominant Cost | Memory |
|----------|--------------|----------------|--------------------|---------|
| Fixed-size | O(D) | O(D/C × L² × d_model) | Embedding | O(D/C × d_model) |
| Fixed + overlap | O(D) | O(D/(C(1-α)) × L² × d_model) | Embedding | O(D/(C(1-α)) × d_model) |
| Sentence-based | O(D) | O(D/C × L² × d_model) | Embedding | O(D/C × d_model) |
| Semantic | O(S × L² × d_model) | O(D/C × L² × d_model) | Boundary detection | O(S × d_model) |
| Recursive | O(D log D) | O(D/C × L² × d_model) | Embedding | O(D/C × d_model) |

Where:
- D = total document length in characters
- C = target chunk size
- α = overlap fraction
- S = number of sentences in the document
- L = average sentence length in tokens
- d_model = embedding model dimension

**The real scaling bottleneck** is not the chunking algorithm itself — it is the embedding computation that runs on the chunks afterward. Semantic chunking adds an extra embedding step (embedding every sentence for boundary detection), which makes it roughly 2× the embedding cost of fixed-size chunking.

At corpus scale (1M+ documents), the architecture answer is to decouple the chunking pipeline from the indexing pipeline. Chunking runs offline in a batch job, outputs chunk boundaries, and the indexing pipeline handles embedding in parallel.

---

## 💡 Design Trade-offs

| | Fixed-size | Fixed + overlap | Sentence | Semantic | Recursive |
|---|---|---|---|---|---|
| Implementation | 5 lines of code | 10 lines | 15 lines + sentence splitter | Embedding model + threshold tuning | 30 lines, recursive logic |
| Chunk quality | Low — cuts arbitrarily | Medium — boundaries covered | Medium — clean sentences | High — topic-coherent | High — structure-aware |
| Chunk size variance | None (uniform) | None (uniform) | Moderate | High | Moderate |
| External dependencies | None | None | Sentence tokenizer | Embedding model | None |
| Best document types | Any (baseline) | Any (default choice) | Well-written prose | Mixed-topic documents | Structured (MD, HTML, PDF) |
| Tuning parameters | chunk_size | chunk_size, overlap | chunk_size | threshold, embedding model | chunk_size, separator list |
| When it fails | Mid-sentence cuts | Overlap bloat at high α | Variable sizes, coreference | Threshold sensitivity, gradual drift | Unstructured text with no hierarchy |

---

## 🏭 Production Considerations

### Per-Document-Type Configuration

Production corpora contain multiple document types: legal contracts, support tickets, product docs, blog posts. Each has different information density and structure. The fix is a document classifier that routes each document to its own chunking config:

- Legal contracts: 50-150 tokens, clause-boundary splitting, minimal overlap (10-20 tokens)
- News articles: 150-300 tokens, paragraph-based, 20-25% overlap
- Technical docs: 200-400 tokens, recursive (uses markdown headers), 15% overlap
- Support tickets: Full ticket as one chunk (typically < 500 tokens)

### Metadata Strategy

Store rich metadata with every chunk from day one: doc_id, chunk_index, char_start, char_end, source_filename, page_number, section_header. This metadata is nearly free to store and extremely expensive to reconstruct later. It enables:

- Retrieval filtering (only search recent documents, only search a category)
- Source attribution (show the user where the answer came from)
- Re-indexing without re-parsing (change chunk strategy, re-embed from stored boundaries)

### Incremental Re-indexing

When documents change, you need to update the index. Naive approach: re-chunk and re-embed the entire corpus. Better approach: detect which documents changed (by hash or timestamp), re-chunk only those documents, re-embed only the affected chunks, and update the index in place. For very large corpora, diff at the paragraph level to avoid re-chunking unchanged sections of a modified document.

---

## Staff/Principal Interview Depth

---

**Q1: What are the failure modes of fixed-size chunking, and when does it silently hurt retrieval quality?**

---
**No Hire**
*Interviewee:* "Fixed-size chunking can sometimes cut sentences in the middle, which makes the chunks less readable."
*Interviewer:* The candidate identifies the surface symptom but shows no understanding of the downstream consequence. They do not connect mid-sentence cuts to degraded embedding quality, retrieval precision, or LLM answer quality. No discussion of when this matters most versus when it matters less. No failure mode analysis, no measurement suggestion.
*Criteria — Met:* Basic awareness that cuts happen. *Missing:* Mechanism of how mid-sentence cuts degrade embeddings, when it matters versus when it does not, how to detect the problem, quantitative intuition about severity.

**Weak Hire**
*Interviewee:* "Fixed-size chunking can split sentences in half, which hurts the embedding because the chunk now contains a partial thought. This makes retrieval worse because the query will not match a fragment as well as a complete sentence. I would use a larger chunk size or add overlap to reduce this."
*Interviewer:* The candidate understands the mechanism at a conceptual level and offers a mitigation. But they stop short of the full picture: they do not quantify how much embedding quality degrades, do not discuss the second-order failure mode (overlap increases recall but also increases index size and retrieval latency), and do not mention the most dangerous silent failure — mid-paragraph topic switches that create chunks spanning two unrelated topics.
*Criteria — Met:* Mechanism, basic mitigation. *Missing:* Silent failures, second-order effects of overlap, measurement strategy, domain sensitivity.

**Hire**
*Interviewee:* "There are a few failure modes. First and most discussed: sentence boundary cuts. The chunk starts or ends mid-sentence, which forces the embedding model to represent an incomplete semantic unit. Empirically, this drops retrieval recall by 10-20% in benchmarks on dense factual text. Second, and more dangerous in practice: mid-paragraph topic transitions. If your chunk boundary falls in the middle of a passage that pivots from one topic to another, the resulting embedding is a centroid of both topics and is a poor match for either. This happens silently — the embedding does not raise an error; it just points nowhere useful. Third: fixed-size assumes uniform information density. Legal contracts have high density (every clause matters), while introductory textbooks have low density (most sentences rephrase the same idea). A chunk size calibrated for textbooks will over-chunk legal text, and vice versa. To detect these failures, I would run an embedding visualization — if I see chunks with high internal cosine variance (sentences within the chunk pulling in different directions), that is a sign the chunk spans multiple topics."
*Interviewer:* Strong answer. The candidate identifies three distinct failure modes, distinguishes visible failures from silent ones, introduces the information density dimension, and proposes a concrete detection method. What would push this to Strong Hire: discussion of how the failure rate scales with document type and domain, and connection to downstream evaluation metrics.
*Criteria — Met:* Three failure modes, silent vs. visible distinction, information density, detection approach. *Missing:* Quantitative evaluation framework, domain-specific sensitivity analysis.

**Strong Hire**
*Interviewee:* "Let me separate the failure modes by mechanism. The first mechanism is semantic incompleteness: when a chunk boundary cuts a sentence, the resulting embedding is forced to represent a partial syntactic unit. Transformer-based encoders like BERT variants are trained on complete sentences; the [CLS] token's representation degrades measurably when input truncates mid-sentence. In my experience this shows up as a 5-15% drop in MRR on datasets with precise factual questions. The second mechanism is topic contamination: a chunk that spans a paragraph boundary between Topic A and Topic B produces an embedding that is the centroid of both. This is silently terrible — the embedding will not be the nearest neighbor for either topic query. The way I detect this is to compute pairwise cosine similarity between sentences within each chunk. If the minimum intra-chunk similarity is below 0.4, the chunk is probably spanning a topic shift. The third mechanism is density mismatch: fixed-size chunking treats all text as having the same information density per token. In practice, legal contracts, academic abstracts, and medical notes have 3-5x the information density of narrative prose. Using a single chunk size across document types guarantees that one of them will be miscalibrated. The fix is per-document-type chunk size tuning, which in production I would drive with a held-out evaluation set where chunk size is the hyperparameter. Finally, there is a retrieval-precision trade-off that people miss: smaller chunks increase precision (the retrieved chunk is more tightly focused on the query) but decrease recall (a fact requiring multiple sentences may be split across chunks, and neither chunk alone triggers retrieval). The right chunk size is the one that maximizes F1 on your specific query distribution — and that is not 256 tokens by default, it is whatever your eval set tells you."
*Interviewer:* This is a clear Strong Hire. The candidate moves beyond the common answer, uses precise vocabulary (MRR, centroid embedding, intra-chunk similarity), proposes a concrete detection heuristic, connects to production evaluation practice, and closes with the key insight that chunk size is a hyperparameter that should be tuned against your own data — not chosen by convention. The answer demonstrates systems-level thinking and measurement discipline.
*Criteria — Met:* Three failure modes with precise mechanisms, silent vs. visible failures, detection heuristic with a concrete threshold, density mismatch, precision-recall framing, production evaluation practice.

---

**Q2: How would you choose chunk size for a production RAG system over legal contracts versus short news articles?**

---
**No Hire**
*Interviewee:* "I would use 512 tokens for legal contracts because they are longer, and maybe 256 tokens for news articles because they are shorter."
*Interviewer:* The candidate has the intuition backwards and has not thought about the mechanism. Longer documents do not require larger chunks — document length and optimal chunk size are nearly independent. Legal contracts need smaller chunks precisely because information density is high and every clause is a distinct retrievable unit. The candidate is mapping document length to chunk size, which is the wrong variable.
*Criteria — Met:* Awareness that different document types might need different sizes. *Missing:* Correct direction of the effect, information density reasoning, how to measure the right size.

**Weak Hire**
*Interviewee:* "Legal contracts have very dense, precise language where every clause matters. I would use smaller chunks, maybe 100-200 tokens, so each clause gets its own embedding. News articles are more narrative and each paragraph covers one event, so 300-500 tokens feels more natural. I would run some experiments to confirm."
*Interviewer:* The reasoning is now correct — information density driving chunk size. The candidate knows the direction of the effect and connects it to domain structure. What is missing: no mention of how to measure "correct," no discussion of the overlap parameter, and no connection to the query distribution.
*Criteria — Met:* Correct direction, information density intuition, domain-structural reasoning, experimental mindset. *Missing:* Evaluation framework, overlap tuning, query distribution sensitivity.

**Hire**
*Interviewee:* "The right mental model is to match chunk size to the granularity of retrievable units in your domain. For legal contracts, the retrievable unit is usually a single clause or subsection — typically 50-150 tokens. Going larger risks fusing two clauses with different legal meanings into one embedding, making it impossible to retrieve one without the other. I would also use minimal overlap (10-20 tokens) because clauses are designed to be independent. For news articles, the retrievable unit is usually a paragraph (150-300 tokens), and I would use larger overlap (20-30%) because news writing often uses backward coreference — pronouns that refer to entities introduced in the previous paragraph. Without overlap, retrieval can return a paragraph that starts with 'He said' and the LLM has no idea who 'He' is. The way I would validate this: I would build an evaluation set of 50-100 questions across both domains, with ground-truth answers, and measure RAGAS answer relevance and faithfulness as I sweep chunk size from 64 to 1024 tokens. The chunk size that maximizes F1 on the eval set is the one I deploy."
*Interviewer:* Strong. The candidate uses domain structure to derive chunk size, understands backward coreference as a specific failure mode that drives the overlap decision, and proposes a concrete evaluation protocol. What would push to Strong Hire: discussion of hierarchical chunking and the latency implications of different chunk sizes at scale.
*Criteria — Met:* Domain-structural granularity reasoning, backward coreference, overlap tuning, evaluation framework. *Missing:* Hierarchical chunking, latency implications.

**Strong Hire**
*Interviewee:* "I would approach this as a domain-specific retrieval unit analysis. The core question is: what is the smallest meaningful unit that can stand alone as an answer to a plausible query? For legal contracts, that is a clause — 50-150 tokens, sometimes shorter for definitions. Fusing clauses is dangerous because contract language is designed so that each clause is an independent legal proposition; merging two clauses creates an embedding that misrepresents both. I would use a clause-boundary splitter if the contract is structured (and most are — numbered sections, clear delimiters), falling back to sentence-boundary splitting for recitals and preambles. I would set overlap to exactly one sentence (15-30 tokens) to handle the rare cases where a clause continues a thought from the previous one. For news articles, the answer changes because the query distribution is different. News readers query at the event level ('what happened at the trial') not the clause level. The retrievable unit is a paragraph, 150-300 tokens. But news writing uses heavy backward coreference, so I would use 20-25% overlap to carry forward entity mentions. More importantly, I would run a two-level index: one at the paragraph level for specific fact retrieval, one at the article level for summary-style queries. The routing logic to decide which level to query would be a small classifier on the query — 'summarize' queries go to article-level, 'what specifically' queries go to paragraph-level. To validate, I would use a held-out eval set with labels for both query types, measuring RAGAS faithfulness, answer relevance, and context precision. Context precision is the metric most sensitive to chunk size — it drops sharply when chunks are too large because the LLM gets distracted by irrelevant content in the same chunk."
*Interviewer:* Clear Strong Hire. The answer demonstrates first-principles domain analysis, addresses the full stack from document structure through evaluation, introduces hierarchical indexing with query routing, and names a specific evaluation metric (context precision) that is directly sensitive to the decision being made. The candidate is ready to lead this project.
*Criteria — Met:* Domain-structural granularity analysis, clause-boundary splitting, hierarchical indexing with query routing, overlap tuning with specific rationale, full evaluation framework with domain-sensitive metrics.

---

**Q3: Walk me through the computational complexity of building a semantic chunking pipeline at scale. Where are the bottlenecks?**

---
**No Hire**
*Interviewee:* "Semantic chunking is slower than fixed-size chunking because it uses embeddings. The main bottleneck is probably the embedding model."
*Interviewer:* Directionally correct but not useful. The candidate has not reasoned about the actual complexity, has not distinguished between the components, and cannot tell you how much slower or why.
*Criteria — Met:* Awareness that embeddings add cost. *Missing:* Complexity analysis, component breakdown, specific bottlenecks, optimization strategies.

**Weak Hire**
*Interviewee:* "The bottleneck is embedding computation. If you have N sentences, you need N forward passes through the embedding model, each of O(L² × d) where L is sentence length and d is model width. Then you need N-1 similarity comparisons. Compared to fixed-size chunking which is O(N), semantic chunking is much more expensive. You could batch the embeddings to use GPU parallelism."
*Interviewer:* The candidate can write a complexity expression and identifies the right bottleneck. Batching is the right first mitigation. What is missing: the candidate has not reasoned about the full pipeline (chunks still need to be re-embedded for the index after boundary detection), has not discussed the window-size parameter, and has not addressed index build time vs. embedding time as separate costs.
*Criteria — Met:* Correct complexity expression for embedding, correct bottleneck identification, batching mitigation. *Missing:* Full pipeline accounting, window parameter, index build vs. chunking cost breakdown.

**Hire**
*Interviewee:* "Let me break the pipeline into stages. Stage 1: sentence splitting — O(D) where D is total document length in characters, essentially free. Stage 2: sentence embedding — this is the bottleneck. If you have S sentences, you are running S forward passes of O(L² × d_model) per pass. In practice, L is short (15-30 tokens per sentence) so this is fast per sentence, but S can be large. A 100-page document might have 500 sentences. With batching on a GPU, you can do 64 sentences per pass, bringing wall time down significantly. Stage 3: similarity computation — O(S) comparisons of O(d) dot products = O(S × d). Trivial relative to embedding. Stage 4: re-embedding the chunks — this is often overlooked. After you have identified chunk boundaries, you need to re-embed the complete chunks for your vector index. Chunk count is typically S/5 to S/10, so this adds maybe 20% to the total embedding cost. The real scaling problem is at corpus level: if you have 1M documents, the total embedding compute is substantial and requires a multi-hour job on a GPU cluster. The fix is a two-stage architecture: run semantic chunking offline in a batch pipeline with horizontal scaling, cache the chunk boundaries, and only re-run when documents change."
*Interviewer:* Strong answer with concrete numbers and a production-aware conclusion. The candidate counts FLOPs, correctly identifies the re-embedding stage as a hidden cost, and proposes an offline batch architecture. What would elevate to Strong Hire: discussing the window parameter trade-off and the impact of embedding model size on the accuracy-cost trade-off.
*Criteria — Met:* Staged complexity analysis, batching optimization, re-embedding accounting, offline batch architecture. *Missing:* Window parameter analysis, model size vs. accuracy trade-off.

**Strong Hire**
*Interviewee:* "The complexity depends on the comparison strategy. Most semantic chunking implementations use a sliding window: compare each sentence i to the next k sentences (k=3 is common). This gives O(S × k) similarity computations after embedding, which is linear and cheap. But if you compare all pairs — O(S²) — costs explode for long documents. The main cost is still the embedding stage: O(S × L² × d_model / batch_size) wall time on GPU. For a 110M parameter encoder with L=25, d_model=768, the attention computation per sentence is roughly 25² × 768 = 480K FLOPs. For S=500 sentences in a 100-page document, that is 240M FLOPs per document. At GPU throughput of ~10¹³ FLOPs/s with batching, this is very fast per document. The real cost is not the chunking; it is the index-build embedding that runs on the chunks after boundaries are identified. If semantic chunking produces 80 chunks per document from 500 sentences, those 80 chunks need to be embedded at their full length (200-500 tokens each), which is 5-10x more expensive per unit than the short sentence embeddings used for boundary detection. At corpus scale: 1M documents × 80 chunks × 400 tokens per chunk is a real engineering problem. The architectural answer is to decouple the chunking pipeline from the indexing pipeline. Chunking runs offline, outputs (doc_id, chunk_boundaries) tuples, and the indexing pipeline runs embedding in parallel. You also want incremental updates — when a document changes, you re-chunk and re-embed only that document's chunks, not the entire corpus. One more thing: there is an accuracy-cost trade-off in the embedding model you use for boundary detection. For chunking, you do not need your production embedding model (which might be large and expensive). A smaller, faster model like MiniLM (22M parameters, 10x cheaper) works almost as well for boundary detection because you are only looking for topic shifts, not fine-grained semantic similarity. You save the expensive model for index-build embedding."
*Interviewer:* Exceptional. The candidate has command of the full complexity picture, correctly identifies that index-build embedding is the dominant cost (not boundary detection), proposes a decoupled architecture with incremental updates, and introduces the insight that a cheaper model suffices for boundary detection. This is the kind of systems-level thinking that separates staff-level engineers from senior engineers.
*Criteria — Met:* Window strategy comparison, correct identification of index-build embedding as dominant cost, corpus-scale calculation, decoupled architecture, incremental update strategy, cheap-model-for-detection optimization.

---

**Q4: Your team has been using a 512-token fixed-size chunking strategy with 50-token overlap for 6 months. A product manager says users are complaining that the chatbot "misses context" in long answers. How do you diagnose what is wrong and what chunking changes would you consider?**

---
**No Hire**
*Interviewee:* "I would just switch to semantic chunking because it is smarter than fixed-size chunking."
*Interviewer:* The candidate makes a recommendation without any diagnosis. They do not know what "misses context" means in retrieval terms, have not considered that the problem might not be in chunking at all, and are proposing a significant system change based on a vague symptom.
*Criteria — Met:* Awareness that an alternative exists. *Missing:* Diagnostic framework, problem decomposition, hypothesis generation, cost-benefit of the proposed change.

**Weak Hire**
*Interviewee:* "I would look at some examples of queries where the chatbot misses context and see if the relevant information is being retrieved. If not, the chunks might be too small and splitting the information. If the relevant chunks are being retrieved but the answer is still missing context, the problem might be in how the LLM uses the context. I would try increasing the chunk size or switching to sentence-based chunking."
*Interviewer:* The candidate has the right diagnostic instinct — separate retrieval quality from generation quality — and connects the symptom to concrete hypotheses. What is missing: no mention of the overlap as a diagnostic point (512-token chunks with 50-token overlap have only ~10% overlap), no systematic evaluation framework, no consideration of the cost of changing a 6-month-old production system.
*Criteria — Met:* Retrieval vs. generation decomposition, reasonable hypotheses, actionable changes. *Missing:* Overlap analysis, evaluation framework, production change cost.

**Hire**
*Interviewee:* "First I would try to understand whether 'misses context' is a retrieval problem or a generation problem. I would pull a sample of 20-30 failed queries and manually annotate: does the correct answer exist in the retrieved chunks? If yes, the problem is in the LLM's use of context. If no, the problem is in retrieval, which is affected by chunking. For retrieval failures, my first hypothesis for 512-token chunks with 50-token overlap: the overlap is only 10%, which means boundary information that spans more than 50 tokens falls into neither chunk. I would test this by increasing overlap to 20-25% (100-128 tokens) while holding chunk size constant. This is a low-risk change — same number of chunks approximately, more boundary coverage. If the problem is that answers require information from across multiple sections of a document, that is a different issue that neither larger overlap nor semantic chunking directly solves. That is a retrieval strategy problem (retrieve more chunks, do query decomposition) not a chunking problem. I would measure the impact of each change using a held-out eval set with RAGAS context recall before deploying."
*Interviewer:* Strong. The candidate correctly leads with diagnosis before prescription, decomposes the problem into retrieval vs. generation failure modes, generates a low-risk first change (increased overlap), and correctly identifies that some "missing context" complaints are actually retrieval strategy problems not chunking problems. What would push to Strong Hire: discussion of hierarchical indexing and explicit risk/cost analysis for each proposed change.
*Criteria — Met:* Diagnosis before prescription, retrieval vs. generation decomposition, overlap analysis, low-risk incremental change, measurement framework, distinction between chunking and retrieval strategy problems. *Missing:* Hierarchical indexing, explicit risk/cost analysis.

**Strong Hire**
*Interviewee:* "I would approach this as a debugging exercise before it is a design exercise. The symptom 'misses context' could trace to at least four different failure modes, and the right fix depends on which one is actually happening. Failure mode 1: the relevant chunk is not retrieved. This is a retrieval recall problem. Diagnosis: for each failed query, embed the query, compute cosine similarity to all chunks, and check whether the ground-truth answer chunk is in the top-50 even if not top-5. If yes, the recall is fine but the ranking needs work — consider hybrid search or re-ranking. If no, the chunk boundary might be splitting the answer — overlap too small. Fix: increase overlap from 50 to 100-128 tokens. Failure mode 2: the answer requires connecting information across multiple retrieved chunks. This is a context synthesis problem. The LLM gets chunks A and B but needs to understand their relationship. Diagnosis: check if the answer is achievable from any single retrieved chunk. If the answer requires two chunks, that is expected and the LLM should handle it — if it does not, the problem is in the prompt, not the chunking. Failure mode 3: the context window is filled with irrelevant chunks, burying the relevant ones. This is a precision problem. 512-token chunks are on the larger side; if you are retrieving top-5, you are handing the LLM 2,500 tokens of context. Fix: smaller chunks (200-300 tokens) with higher top-K gives more coverage with better precision per chunk. Failure mode 4: the relevant information truly spans more than 512 tokens and a single chunk cannot contain it. Fix: store a separate index of section-level summaries alongside the chunk-level index. When a query requires broad context, route to the section-level index. Diagnose which failure mode by manually annotating 30 failed queries, measuring RAGAS context recall and faithfulness for each case. Fix the most common failure mode first. Do not change the chunking strategy until you know it is a chunking problem — it might be a retrieval problem, a prompt problem, or a generation problem."
*Interviewer:* Exceptional. This answer demonstrates the systems-level debugging discipline of a staff engineer. The four-failure-mode taxonomy is correct and complete. The candidate does not propose any fix without first proposing a diagnostic test, correctly identifies that multiple different "missing context" symptoms have different root causes and different fixes, and closes with the critical insight that chunking is one of four possible root causes and should not be changed until it is confirmed to be the cause.
*Criteria — Met:* Four distinct failure mode taxonomy, diagnosis before prescription, RAGAS-based evaluation, precision vs. recall trade-off analysis, hierarchical indexing, prompt-level vs. chunking-level distinction, incremental change strategy.

---

**Q5: A startup is building a RAG system from scratch with limited engineering resources. They ask you to recommend a single chunking strategy they can implement in a day and iterate on later. What would you recommend, and how would you know when to upgrade?**

---
**No Hire**
*Interviewee:* "I would recommend semantic chunking because it produces the best results."
*Interviewer:* Semantic chunking is the most expensive to implement and operate. It requires embedding infrastructure, a similarity computation step, threshold tuning, and has no pre-built implementation in pure Python without external dependencies. For a team with limited resources who needs something in a day, this is the wrong answer.
*Criteria — Met:* Awareness of semantic chunking as a high-quality approach. *Missing:* Practical implementation cost assessment, appropriate recommendation for the context.

**Weak Hire**
*Interviewee:* "I would recommend overlapping fixed-size chunking. It is easy to implement in a day and handles the boundary problem. I would start with 300-400 tokens and 15-20% overlap. When to upgrade: if users complain that answers are incomplete or miss context, consider switching to sentence-based or semantic chunking."
*Interviewer:* Correct recommendation. The candidate has matched the approach to the context. The "when to upgrade" guidance is reasonable but vague — there is no measurement criterion.
*Criteria — Met:* Correct recommendation, practical implementation awareness, basic upgrade signal. *Missing:* Measurement-based upgrade criteria, upgrade roadmap.

**Hire**
*Interviewee:* "Start with overlapping fixed-size chunking. Here is the full specification: chunk size 256-400 tokens, overlap 15-20% (40-80 tokens). This can be implemented in about 30 lines of Python with no external dependencies. The implementation is also nearly identical in structure to what you would need for sentence-based chunking later, so you are not throwing work away. When to upgrade, in order of signal strength: first, measure context precision with a small eval set (20-50 questions). If context precision is below 0.6, your chunks are probably too large and should be smaller. Second, if users frequently rephrase questions to get complete answers, that is a recall signal. Increase overlap first, then consider sentence-based chunking. Third, if your corpus has very mixed document types and you are seeing quality variance by document type, move to per-document-type configurations before investing in semantic chunking. The order of upgrades: overlapping fixed-size (day 1) → sentence-based with overlap (week 2-4) → recursive for structured documents (month 1-2) → semantic for premium quality (month 3+)."
*Interviewer:* Strong recommendation with a full specification, clear upgrade triggers, and an upgrade roadmap. The connection between failure symptoms and upgrade priorities is practical and actionable. What would push to Strong Hire: discussion of the metadata that should be stored alongside each chunk and how that metadata enables future upgrades without re-indexing.
*Criteria — Met:* Specific implementation recommendation, overlap specification, measurement-based upgrade criteria, ordered upgrade roadmap. *Missing:* Chunk metadata strategy, re-indexing cost of upgrades.

**Strong Hire**
*Interviewee:* "For a one-day implementation with limited resources, the answer is overlapping fixed-size chunking with a specific implementation note: store rich metadata with each chunk, because that metadata is worth more than the chunking strategy upgrade you will do in month two. Specification: chunk size 300 tokens, overlap 15% (45 tokens), implemented in pure Python with no dependencies. Each chunk stored with: doc_id, chunk_index, char_start, char_end, source_filename, page_number if available, section_header if detectable. Why metadata matters more than chunking strategy early on: when you upgrade to sentence-based or semantic chunking later, you will want to re-index. If you have stored the source provenance for each chunk, re-indexing is a re-embedding job — you already have the chunk boundaries. If you have not, you are re-parsing your entire corpus. Second, metadata enables retrieval filtering (only search chunks from documents updated in the last 30 days, or only from a specific category), which can substitute for better chunking quality when you are resource-constrained. For upgrade triggers, I would use three signals. Signal 1: measure context precision on a 30-question eval set weekly. When it drops below 0.55, start the sentence-based upgrade. Signal 2: when the corpus grows past 10K documents, measure retrieval latency. If P99 latency exceeds 500ms, optimize the index before improving chunking. Signal 3: when you have users who consistently ask multi-paragraph questions, that is the signal to invest in semantic chunking. The upgrade path: day 1 overlapping fixed-size, week 4 sentence-based, month 2 recursive for structured document types, month 3 semantic for the highest-value user segment, measured by a feature flag."
*Interviewer:* Exceptional. The metadata recommendation is the insight that separates an architect from an implementer — storing rich metadata is essentially free upfront but extremely expensive to reconstruct later, and it enables upgrade paths and filtering that compound over time. The three concrete upgrade signals are all measurable and actionable. The feature-flag rollout for semantic chunking shows production deployment awareness.
*Criteria — Met:* Correct recommendation, full chunk specification, metadata strategy with upgrade implications, three measurement-based upgrade triggers, ordered upgrade roadmap with feature flagging, production deployment awareness.

---

**Q6: You are designing a RAG system for a corpus that includes PDFs with embedded tables, Markdown files with code blocks, and plain-text emails. How does the heterogeneous document format affect your chunking strategy?**

---
**No Hire**
*Interviewee:* "I would use the same chunking strategy for all document types since they are all just text."
*Interviewer:* Tables, code blocks, and email threads are not just text — they have structure that character-based splitting destroys. A table row split across chunk boundaries becomes meaningless. A code block split between an if-statement and its body produces unexecutable fragments. The candidate has no awareness of document structure as a variable in chunking design.
*Criteria — Met:* None. *Missing:* Awareness of document structure impact, format-specific chunking requirements.

**Weak Hire**
*Interviewee:* "Different document types need different treatment. Tables probably should not be chunked — they should be kept together as a single chunk. Code blocks should also be kept whole. Emails are more like regular text so fixed-size or sentence-based chunking works. I would write format-specific chunking logic."
*Interviewer:* The candidate correctly identifies that tables and code blocks are structural units that should not be split. The direction is right. What is missing: no concrete implementation strategy for table extraction from PDFs, no discussion of how tables should be represented for embedding, and no mention of the hybrid approach for documents that mix text and structured elements.
*Criteria — Met:* Format-specific awareness, correct treatment of tables and code blocks. *Missing:* PDF table extraction, table representation for embedding, hybrid documents.

**Hire**
*Interviewee:* "Each format has a different structure, and the chunking strategy needs to match the structure. For PDFs with tables: the first problem is extraction — PDF parsers often mangle tables into plain text that loses the row/column structure. Use a tool like pdfplumber or Camelot to extract tables as structured data. For embedding, do not embed the raw table — convert it to natural language ('The Q3 revenue for product X was $12M, up 15% from Q2'). Each table becomes one or two chunks depending on size. For Markdown with code blocks: use the triple-backtick fence as a hard boundary. Never split inside a code block. For embeddings, code chunks should be embedded using a code-aware model (CodeBERT, CodeT5) if code retrieval is important, because general-purpose text embedders do not represent code semantics well. For emails: emails have a specific structure — subject, header, quoted thread, new reply. The new reply is the most useful chunk; the quoted thread is often redundant. Strip the quoted reply before chunking. Then use sentence-based chunking on the new reply text. The overarching pattern: document parsing and chunking are not the same step. Parse first to identify structure, then apply format-specific chunking rules."
*Interviewer:* Strong. The candidate has concrete strategies for each format, correctly identifies PDF table extraction as a parsing problem, introduces code-aware embeddings, and has a smart preprocessing step for email threading. What would push to Strong Hire: discussion of how to handle documents that mix formats, and the indexing implications of using different embedding models for different chunk types.
*Criteria — Met:* Format-specific strategies for all three types, PDF table extraction as pre-chunking step, natural language table representation, code-aware embeddings, email thread stripping, parsing vs. chunking distinction. *Missing:* Hybrid documents, multi-model embedding alignment.

**Strong Hire**
*Interviewee:* "This is a document pipeline design problem as much as a chunking problem. The key insight is that chunking happens after parsing, and parsing heterogeneous formats correctly is the hard part. PDFs with tables: tables in PDFs are stored as layout instructions, not semantic structure. Standard PDF text extraction converts them into sequences of cells read left-to-right across rows, which looks like gibberish as plain text. Use pdfplumber or Camelot for table detection and extraction. The extracted table becomes a DataFrame. For indexing, I would store the table in two forms: (1) a natural language summary chunk for semantic embedding, and (2) the raw structured data (CSV or JSON) for exact lookup queries. This gives you both semantic search and structured query capabilities. Markdown with code blocks: use triple-backtick boundaries as hard chunk separators. Code blocks get their own chunks. For embedding code chunks, the choice matters: general-purpose embedders are trained on natural language and represent code poorly. Code-specific embedders (CodeBERT, GraphCodeBERT) understand variable names and control flow. If code search is a primary use case, use code-specific embedders for code chunks and a separate index. If code search is secondary, a general embedder applied to a concatenation of the code's docstring and a plain-English description is often sufficient and avoids the multi-index problem. Emails: strip quoted reply content before chunking — look for '>' prefixes or 'On [date], [person] wrote:' markers. Store the email metadata (from, to, subject, date, thread_id) as chunk metadata for filtering. The indexing architecture: I would run separate parsing pipelines for each format type, each producing a standardized chunk representation (text, embedding_model_hint, metadata), then route to embedding appropriately. All chunks feed into a single HNSW index if using a unified embedding model, or two indices (text and code) with query routing if using separate models. The query routing is simple: if the query contains code keywords or function call syntax, add results from the code index."
*Interviewer:* Exceptional. This answer demonstrates senior systems architect-level thinking across the full document processing pipeline. The dual-representation strategy for tables is a real production technique. The code embedding model choice discussion is precise and acknowledges the multi-index complexity. The routing architecture ties it all together.
*Criteria — Met:* Format-specific parsing strategies, dual-representation for tables, code embedding model choice with trade-off analysis, email thread stripping, chunk metadata strategy, unified vs. multi-index architecture with query routing, production deployment awareness.

---

## Key Takeaways

🎯 1. **Chunk size is a hyperparameter, not a constant** — the optimal value depends on your query distribution, document type, and embedding model. Tune it against an eval set.
🎯 2. **The centroid effect is the core failure mechanism** — chunks spanning multiple topics produce embeddings that match no query well. Detect with intra-chunk cosine similarity.
   3. **Overlap of 10-20% captures 90%+ of boundary coverage benefit** — going beyond 20% yields diminishing returns at increasing storage cost.
⚠️ 4. **Information density varies 3-5x across document types** — a single chunk size for mixed corpora guarantees miscalibration for at least one type.
   5. **Semantic chunking's bottleneck is index-build embedding, not boundary detection** — use a cheap model for boundaries, save the expensive model for the index.
🎯 6. **Metadata is worth more than your chunking strategy upgrade** — store doc_id, chunk_index, char_start, char_end, source_filename from day one.
   7. **Diagnosis before prescription** — "misses context" can be a chunking problem, a retrieval problem, a prompt problem, or a generation problem. Determine which before changing anything.
   8. **Heterogeneous corpora need format-specific parsing before chunking** — tables, code blocks, and emails each have structural rules that character-based splitting destroys.
   9. **The upgrade path is predictable:** overlapping fixed-size → sentence-based → recursive → semantic, each triggered by measurement, not intuition.

---

[← Back to Chunking Techniques (Layer 1)](./chunking-techniques.md) | [Back to RAG module](./README.md)
