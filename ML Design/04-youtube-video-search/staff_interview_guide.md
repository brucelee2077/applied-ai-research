# YouTube Video Search System — Staff/Principal Interview Guide

## How to Use This Guide

This guide covers a complete 45-minute staff/principal ML design interview for a YouTube-like text-based video search system. Hire and Strong Hire answers are written in first-person candidate voice.

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

*"Design a video search system for a platform like YouTube. Users type a text query and receive a ranked list of relevant videos. Walk me through your approach."*

### What to Clarify — 6 Dimensions

| Dimension | Question | Why It Matters |
|-----------|----------|---------------|
| **Business objective** | Maximize user satisfaction? Watch time? Ad revenue? | Determines ranking objective — watch time vs. click vs. explicit relevance |
| **Scale** | How many videos? Billion+? | Determines indexing strategy |
| **Latency** | Sub-100ms? Sub-500ms? | Affects model complexity choices |
| **Data availability** | Do we have (query, video) click pairs? Human labels? | Determines whether supervised or self-supervised training |
| **Interaction types** | Text-only queries? Or multimodal (image + text)? | Multimodal requires cross-modal architecture |
| **Constraints** | Personalization? Multiple languages? | Personalized search needs user-context models |

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use Elasticsearch to index video titles and return videos matching the query terms."*

BM25/Elasticsearch is a reasonable baseline but misses semantic understanding. More importantly, shows no ML thinking — no embedding model, no learning from user feedback.

---

#### ⚠️ Weak Hire Answer

*"Can I ask — how many videos are we dealing with? And do we have training data of (query, video) pairs?"*

Gets scale and data questions but misses: latency SLA, personalization, business objective, and the crucial distinction between visual search and text search.

---

#### ✅ Hire Answer (Staff)

*"Before designing, let me clarify a few things that change the architecture significantly.*

*First, the business objective: are we optimizing for click-through rate on results, watch time after clicking, or something like user-reported search satisfaction? Each has different implications. CTR optimization can lead to clickbait titles. Watch time is better correlated with satisfaction.*

*Second, scale: how many videos? I'll assume ~1 billion based on YouTube's scale. This drives whether we can do exact retrieval or need ANN.*

*Third, latency: for interactive search, users expect results in under 500ms. That gives us maybe 100ms for the ML model. This constrains model size.*

*Fourth, training data: do we have human-annotated (query, relevant video) pairs, or only click logs? Click logs are noisier but abundant. Human annotations are higher quality but expensive.*

*Fifth, languages: text search across 100+ languages requires multilingual models. I'll assume we start with English for simplicity.*

*Finally, personalization: should search results account for user history? YouTube's search is semi-personalized. I'll assume non-personalized for now.*

*Proceeding with: 1 billion videos, <500ms SLA, 10 million (query, video) click pairs available, English only, non-personalized."*

---

#### 🌟 Strong Hire Answer (Principal)

*"The framing question has an architectural decision embedded in it: is this primarily a text matching problem or a semantic understanding problem?*

*Text matching (BM25/TF-IDF) works when the user's query words appear literally in the video's metadata. But users often describe content semantically — querying 'dogs learning tricks' should find videos about training puppies even if the title says 'puppy obedience training.' This semantic gap is why ML-based search outperforms BM25.*

*The system design has three distinct subsystems, and good candidates identify all three upfront:*
1. *Text search: match query terms to video titles/descriptions/transcripts using inverted index (Elasticsearch)*
2. *Visual/semantic search: match query semantics to video visual content using cross-modal embeddings*
3. *Ranking: combine signals from both subsystems with personalization and quality signals*

*Most candidates design one of these and forget the others. The production YouTube system does all three.*

*On evaluation: 'relevant' is ambiguous for video search. Is a 3-second clip of what the user wanted relevant? Is a 2-hour video where the topic appears for 5 minutes relevant? I'd want to understand what user satisfaction means — I'd propose using 'user watched > 30 seconds after clicking' as the primary relevance signal, which captures both click intent and content delivery.*"

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

*"How would you frame this as an ML problem?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"Input is the text query, output is a ranked list of videos. I'd train a model to predict which videos are relevant to the query."*

No discussion of why retrieval beats classification, or how text and video representations work together.

---

#### ⚠️ Weak Hire Answer

*"I'd use a two-tower model: one tower for the text query, one for the video, trained to maximize similarity between relevant pairs."*

Right architecture name, but no detail on what 'similarity' means, how the training objective works, or why this is the right approach vs. alternatives.

---

#### ✅ Hire Answer (Staff)

*"This is a cross-modal retrieval problem: given a text query, retrieve the most semantically relevant videos from a large corpus.*

*The ML framing: learn two embedding functions:*
- *f_text: text query → ℝ^d*
- *f_video: video → ℝ^d*

*such that sim(f_text(q), f_video(v)) is high when video v is relevant to query q and low otherwise.*

*Why retrieval not classification: if we trained a pairwise classifier (query, video) → P(relevant), we'd need to run inference against all 1B videos per query — infeasible. With pre-computed video embeddings and ANN search, we compute f_text(q) once and find nearest videos in O(log N). This is the key architectural decision.*

*The system has two search paths in parallel:*
1. *Semantic search: f_text(q) → ANN over video embedding index → semantically similar videos*
2. *Text/keyword search: query terms → Elasticsearch → videos with matching metadata*

*These are complementary: semantic search handles paraphrase and concept queries; keyword search handles specific title/phrase matches.*

*Fusion: combine the top results from both paths via a fusion layer (weighted sum or learned ranker).*

*Training objective: maximize similarity of positive (query, video) pairs while minimizing similarity of negative pairs — contrastive learning with InfoNCE loss.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to flag a subtle but important design choice: what granularity to compute the video embedding at.*

*A video can be 20 seconds or 2 hours. If we embed the full video as a single vector, we lose the ability to find content within long videos. A 1-hour documentary that discusses dogs for 5 minutes should be surfaced for a dog-related query, but embedding the whole video dilutes the signal.*

*Three strategies:*
1. *Video-level embedding: embed the full video. Fast index, low recall for specific topics in long videos.*
2. *Segment-level embedding: segment videos into 30-60 second clips, embed each, index all segments. High recall, but index grows by ~10x and we need to de-duplicate (same video's multiple segments showing up).*
3. *Multi-vector representation: embed key frames + audio transcript segments + metadata separately, store multiple vectors per video, and use multi-vector retrieval (ColBERT-style) at query time.*

*For a YouTube-scale system, segment-level with deduplication is the right tradeoff. It gives good recall for long-video specific topics without the complexity of multi-vector retrieval.*

*The ML objective also needs careful definition. We have positive labels from clicks, but click bias is severe in search: the top result gets 30-40% of clicks regardless of relevance. Using raw clicks as labels trains the model to rank popular/well-titled videos, not necessarily the most relevant ones.*

*Mitigation: use 'watch time after click' as the label weight — a video the user clicked and watched for 60%+ is a higher-quality positive than one they clicked and bounced from after 3 seconds."*

---

## Section 3: Data & Feature Engineering (8 min)

### Interviewer Prompt

*"Walk me through data preparation and feature engineering."*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use the video title as a text feature and extract frames as image features."*

No mention of normalization, tokenization, embedding models, or how to combine modalities.

---

#### ⚠️ Weak Hire Answer

*"For text, I'd use BERT to embed the query. For video, I'd use a CNN on sampled frames and average the frame embeddings."*

Right models named but no detail on preprocessing, tokenization, frame sampling strategy, or training data construction.

---

#### ✅ Hire Answer (Staff)

*"Let me cover both text and video pipelines, since they're quite different.*

**Text data preparation:**

1. Normalization:
   - Lowercase
   - Remove punctuation except meaningful ones (commas, periods in numbers)
   - Strip accents (é → e for matching)
   - Lemmatization (running → run) for keyword matching layer

2. Tokenization: WordPiece (BERT's tokenizer)
   - Handles OOV (out-of-vocabulary) via subword units: 'skateboarding' → ['skate', '##board', '##ing']
   - Vocabulary: 30,000 tokens

3. Sequence handling: truncate to 128 tokens for short queries; 512 for document processing

4. Text embedding model: BERT-base (110M parameters, 768-dim CLS token)
   - Why not BoW/TF-IDF: loses word order and semantic meaning ('bank' as financial vs. river)
   - Why not word2vec: no contextual embeddings (same word, different meaning)
   - BERT: contextual, pre-trained on large corpus, handles polysemy

**Video data preparation:**

1. Frame sampling: sample 1 frame per second or keyframe detection (uniform sampling avoids bias toward fast-cut videos)
   - Typical: sample 8-32 frames per 1-minute video segment

2. Frame preprocessing:
   - Decode to RGB
   - Resize to 224×224 (bilinear interpolation)
   - Normalize: subtract ImageNet mean, divide by std

3. Frame embedding: ViT-B/16 (Vision Transformer) → 768-dim CLS token per frame
   - Why ViT over ResNet: better global context, captures scene-level semantics relevant to video search

4. Video-level embedding: average pooling over all frame embeddings → one 768-dim video embedding
   - Alternative: attention-weighted pooling (learn which frames are most representative)

**Training data construction:**

- Available: 10M (query, video) pairs with click signal
- Positive: (query, clicked video after search) with watch_time > 30s
- Negative: (query, shown video not clicked) — in-batch negatives during training
- Class imbalance: most queries have 1-5 positive videos; others are 0 (null queries) or 1000+ (popular queries)
- Temporal split for evaluation: train on months 1-8, val on month 9, test on month 10

**Feature hashing for OOV words:**
- At serving time, rare words may not be in BERT's vocabulary
- Subword tokenization handles this (BERT's WordPiece breaks unknown words into known subpieces)
- Fallback: feature hashing maps any string to a fixed-size hash space, avoids OOV but loses semantics*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to go deeper on the multimodal fusion strategy, because it's where the most product value comes from.*

*Text search + visual search find different videos. Text search is best for 'how to change a tire' (keyword match to tutorial titles). Visual search is best for 'that song with the blue umbrella' (semantic query matching video visual content). The fusion strategy determines how we combine them.*

**Fusion strategies:**

1. *Score-level fusion:* Both systems produce ranked lists with relevance scores. Combine: final_score = α * text_score + (1-α) * visual_score. Simple, interpretable. Problem: scores from different systems are on different scales and need normalization.

2. *Feature-level fusion:* At the ranking stage, take the text embedding and video embedding, concatenate them, and run through a ranker. The ranker learns the optimal combination. More powerful but requires labeled ranking data.

3. *Query-type routing:* Classify the query as 'keyword intent' vs. 'semantic intent', route to the appropriate system. Over-simplification: most queries have both intents.

*Production recommendation: score-level fusion for candidate retrieval (simple, low latency), feature-level fusion in the ranking stage (uses richer signals for the small candidate set).*

**Why average frame pooling isn't optimal:**

*Simple average of frame embeddings gives equal weight to talking-head frames, black transition frames, and action-packed frames. This dilutes the signal. Better approaches:*
1. *Keyframe detection: use scene-change detection to identify representative frames*
2. *Attention-weighted pooling: learn a weight for each frame based on its relevance to a 'generic query' distribution*
3. *Temporal embeddings: treat the video as a sequence, use a lightweight Transformer over the frame sequence to get a context-aware video embedding — this captures narrative structure, not just content*

*For a first version, average pooling over keyframes is good enough. For production, temporal Transformer gets an additional 2-5% nDCG.*"

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

*"Deep dive on the model. What architectures would you consider and what would you deploy?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use BERT for text and CNN for video, concatenate the features, and train a classifier."*

Concatenating and classifying doesn't scale — you'd need to run inference on all 1B videos per query. Completely infeasible.

---

#### ⚠️ Weak Hire Answer

*"Two-tower model: BERT for text, CNN for video. Train with contrastive learning. ANN for retrieval."*

Correct but surface-level. No detail on training objective, negative sampling, temperature, or why two-tower works for retrieval.

---

#### ✅ Hire Answer (Staff)

*"Let me walk through the progression and explain why simpler approaches fail.*

**Why not a single fused model:**
If I concatenate query and video features and pass through a single transformer (like a cross-encoder), I get the best possible relevance score — the model sees the full interaction between query and video. But at inference, I must run this for all 1B videos per query. Even at 1ms/video, that's 1 billion ms = 11 days per query. Completely infeasible.*

*This is why retrieval-then-ranking is the only viable architecture at scale:*
- *Retrieval stage: bi-encoder (two-tower) — pre-compute video embeddings, run ANN at query time*
- *Ranking stage: cross-encoder — run only on top-K candidates from retrieval*

**Two-Tower Architecture (Retrieval):**

*Text encoder:*
- Input: tokenized query, max 128 tokens
- Architecture: BERT-base (12 layers, 768 hidden, 12 heads)
- Output: [CLS] token embedding, 768-dim → linear projection to 256-dim

*Video encoder:*
- Input: sampled frames (N=8 frames per video segment)
- Architecture: ViT-B/16 per frame → average pooling → 768-dim → linear projection to 256-dim

*Similarity: cosine similarity between 256-dim text and video embeddings.*

*Training (InfoNCE / in-batch negatives):*
```
L = -(1/B) Σ_i log[ exp(cos(t_i, v_i) / τ) / Σ_j exp(cos(t_i, v_j) / τ) ]
```
*where (t_i, v_i) is the i-th (query embedding, positive video embedding) pair, and the denominator sums over all B videos in the batch (1 positive + B-1 negatives). Temperature τ = 0.07.*

*Why this works for retrieval: pre-compute all video embeddings offline. At query time, run text encoder (20ms) then ANN search (10ms) → top-500 candidates in 30ms.*

**Why BM25 doesn't suffice:**
BM25 retrieves videos whose titles/descriptions literally contain query words. It misses:
- Paraphrase: 'car crash' vs. 'vehicle accident'
- Concept: 'dogs being goofy' (no video has this exact title)
- Cross-lingual (future): English query matching Spanish-title video

The two-tower model captures semantic meaning through the embedding space — 'car crash' and 'vehicle accident' map to nearby embeddings even without shared words.

**Ranking Stage (Cross-Encoder):**

*Input: top-500 retrieved candidates*
*Architecture: a 6-layer Transformer that jointly processes [CLS] + query tokens + [SEP] + video metadata tokens*
*Output: relevance score per candidate*
*Why cross-encoder for ranking: it sees the full query-video interaction, enabling much higher precision*
*Latency: ~50ms for 500 candidates on GPU*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to talk about three things the standard answer misses: knowledge distillation from cross-encoder to bi-encoder, handling long-tail queries, and the evaluation-training feedback loop.*

**Knowledge distillation:**

*The cross-encoder (used for ranking) achieves much higher relevance quality than the bi-encoder (used for retrieval). But we can't use the cross-encoder for retrieval at scale.*

*Trick: use the cross-encoder as a 'teacher' to distill knowledge into the bi-encoder 'student.' Process:*
1. Train cross-encoder on labeled (query, video, relevance) pairs*
2. Run cross-encoder on training data to get soft relevance scores*
3. Train bi-encoder using the cross-encoder scores as labels (distillation loss + contrastive loss)*
*This is the ColBERT/SPLADE/dense retrieval distillation paradigm.*

*Result: the bi-encoder learns to approximate the cross-encoder's quality within the constraints of the two-tower architecture. Recall@100 improves from ~80% to ~90%.*

**Long-tail query handling:**

*Head queries ('cat videos', 'how to cook pasta') have abundant click data. Tail queries ('obscure 1980s Bulgarian folk music') have no click data.*

*Problem: the two-tower model is trained on click-heavy queries and performs poorly on tail queries.*

*Solutions:*
1. *Text-only fallback: for queries with <10 historical clicks, fall back to BM25 keyword search*
2. *Query clustering: represent each tail query by its nearest head query in query embedding space, use head query's click data as pseudo-labels*
3. *Self-supervised video embedding: pre-train video encoder on all videos (not just those with click data) using visual-text contrastive learning on (video, auto-generated transcript) pairs — this gives good embeddings for all videos including obscure ones*

**Training-evaluation feedback loop:**

*As the model improves, it changes what videos users see, which changes what users click, which changes the training data. This is the training-serving feedback loop.*

*The insidious problem: the model trains on clicks on its own recommendations. A video that the model undervalued in the past gets few clicks (because it was rarely shown). The new model continues to undervalue it (because there are no clicks to learn from). This is exposure bias.*

*Break the loop:*
1. *5% random recommendation injection: log clicks on randomly shown videos as 'exploration data'; use this unbiased data for evaluation*
2. *IPS correction in training: weight each training example by 1/P(model showed this video), where P is the model's probability of showing that video — this re-weights rare-shown videos*
3. *Counterfactual evaluation: estimate what performance would have been with a different random policy using the doubly robust estimator:*
```
DR = IPW + direct_method_correction
```*"

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

*"How do you measure quality offline and online?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"Offline metrics for video search require care because there's typically only one or a few correct videos per query (unlike image search where many images might be relevant).*

**Offline metrics:**

*Mean Reciprocal Rank (MRR): primary metric*
```
MRR = (1/|Q|) Σ_{i=1}^{|Q|} 1/rank_i
```
*where rank_i is the position of the first relevant video for query i. MRR rewards putting the correct video as high as possible.*

*Why MRR over Precision@k: for single-answer queries (where there's only one correct video), precision@k doesn't reward ranking it at position 1 vs. position k equally. MRR does.*

*Why not mAP: mAP requires multiple known-relevant items per query. With click data, we typically have only 1-2 clicked videos per query — mAP is noisy with so few positives.*

*Recall@k: secondary metric — does the correct video appear in the top-k results?*
- *For retrieval stage evaluation: Recall@100 (is the correct video in the top-100 candidates?)*
- *For end-to-end evaluation: Recall@5, Recall@10*

**Online metrics (A/B test):**
- *Primary: Total watch time after search (this is YouTube's north star)*
- *Secondary: CTR@3 (click-through on top-3 results), video completion rate post-click*
- *Guardrail: session abandonment rate (user searches but watches nothing)*
- *Counter-metric: zero-result sessions*

**Offline-online gap for video search:**
1. *Click bias: offline ground truth is 'what users clicked' — but clicks reflect the previous model's rankings*
2. *No-impression gap: videos never shown by previous model have no click data; new model may find better results that look 'wrong' offline*
3. *Query distribution shift: search trends change (new events, new content) — an offline test set from 3 months ago doesn't represent current queries*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to raise a metric design question that goes beyond MRR and CTR: measuring long-term search quality.*

*Short-term metrics (CTR, watch time) can be gamed. A model that learns to rank viral videos highly will have great short-term CTR but poor satisfaction for users looking for specific content.*

*Long-term metric: 'search success rate' — did the user find what they were looking for? Proxy: after clicking a search result and watching it, did the user close the app satisfied (no immediate re-search) or immediately search again?*

*Define: search success = user watched >30% of the video AND didn't repeat the same query within 5 minutes.*

*This metric is harder to optimize for (requires longitudinal user tracking) but better predicts actual user satisfaction.*

*A/B test design considerations:*
- *Experiment unit: should we randomize at user level or query level? User-level randomization gives cleaner causal estimates (no crossover) but requires more users. Query-level randomization has carryover effects (user learns about the new system).*
- *Minimum detectable effect: if baseline success rate is 65% with σ=15%, detecting a 1% lift requires n = 2*(225)*(1.96+0.84)² / (0.0065²) ≈ 3.3M queries. Easy to achieve at YouTube scale.*
- *Novelty effects: video search results change, users notice. Run A/B test for at least 2 weeks to separate novelty from sustained quality.*"

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

*"How does the serving system work end-to-end?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"The serving architecture has two offline pipelines and one online pipeline.*

**Offline Pipeline 1 — Video Embedding Index:**
1. For every video: sample 8 frames, run ViT-B/16 on each → average → 256-dim video embedding
2. Build FAISS IVF-PQ index over 1B video embeddings
3. Refresh daily to include new uploads (incremental indexing)
4. Total size: 1B × 32 bytes (IVF-PQ compressed) = 32GB — fits in RAM on 1-2 servers

**Offline Pipeline 2 — Elasticsearch Text Index:**
1. For every video: index title, description, auto-generated transcript
2. Standard TF-IDF/BM25 inverted index
3. Update in near-real-time as new videos upload

**Online Pipeline (per query):**
1. User types query → query text sent to search service (latency: 5ms)
2. Parallel fan-out:
   a. Semantic search: text encoder (BERT, 30ms) → FAISS ANN search (10ms) → top-200 semantic results
   b. Keyword search: BM25 on Elasticsearch (20ms) → top-200 keyword results
3. Fusion: merge and deduplicate top-200 from each path → top-500 candidates (5ms)
4. Ranking: cross-encoder scores top-500 on GPU (50ms)
5. Re-ranking: quality signals (view count, upload recency, spam filter), diversity (max 3 from same channel) (10ms)
6. Return top-10 results to user
7. Total: ~130ms end-to-end

**Training-serving skew prevention:**
- Same tokenization library used in training and serving
- Feature schema validation on startup
- Feature distributions logged and compared daily between training and serving*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to discuss two production challenges that don't appear in most system designs: the index freshness problem and the model versioning problem.*

**Index freshness:**
YouTube ingests ~500 hours of video per minute. For a 1B video corpus, that's ~0.05% daily refresh. Embedding new videos and inserting them into the index needs to happen within minutes of upload for search to work.*

*Solution: a streaming embedding pipeline:*
1. New video uploaded → video processing pipeline (thumbnail generation, transcription)*
2. Video embedding pipeline subscribes to the same event stream → runs ViT-B in parallel → generates 256-dim embedding*
3. Embedding written to the 'new video buffer' (a small HNSW index of recent videos)*
4. Daily batch job merges new video buffer into the main IVF-PQ index*

*Query fan-out hits both the main IVF-PQ index (stable, 99% of corpus) and the new video buffer (HNSW, recent uploads). This gives near-real-time freshness for new content without re-indexing the full corpus.*

**Model versioning:**
When we deploy a new version of the text or video encoder, the old pre-computed video embeddings are in the old embedding space. The new query embeddings are in the new space. Incompatible.*

*Option 1: re-embed all 1B videos (takes days at scale). Not practical for frequent model updates.*
*Option 2: train a compatibility mapping — a linear projection P such that P * old_embedding ≈ new_embedding, trained on a sample of videos embedded by both models. Apply P to all old embeddings without re-running the full pipeline. Quality loss of ~1-3% recall, but deployable in hours.*
*Option 3: use a frozen backbone and only fine-tune the projection head. The backbone embeddings are stable; only the projection changes. This makes compatibility trivial.*

*Production recommendation: option 3 for incremental fine-tuning, option 2 for major model version changes.*"

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Model Answers by Level

#### ✅ Hire Answer (Staff)

**5 Failure Modes:**

**1. Long-Tail Queries (No Training Data)**
- *What:* Queries for obscure topics have no click history → model defaults to popularity-based ranking
- *Detection:* Monitor query coverage: what fraction of queries have <10 historical clicks?
- *Mitigation:* BM25 fallback for zero/sparse click queries; query expansion via LLM to find related queries with better coverage

**2. Temporal Queries ('latest' / 'yesterday')**
- *What:* 'Latest iPhone review' should return newest videos. Embedding models don't capture temporal intent.
- *Detection:* Monitor CTR for queries containing temporal terms ('new', 'latest', 'today', '2024')
- *Mitigation:* Query classification: detect temporal queries → boost by recency in ranking

**3. Multi-language Videos**
- *What:* Video is in Spanish but user queries in English. Semantic model misses it.
- *Detection:* Track recall for non-English videos in English queries
- *Mitigation:* Multilingual BERT (mBERT) or XLM-R for text encoder; auto-translate video titles for text index

**4. New Video Cold Start**
- *What:* Video uploaded 1 hour ago has no clicks → scores low on popularity signal
- *Detection:* Monitor recommendation rate for videos <24h old
- *Mitigation:* Separate recency signal in ranking; content-quality signals from video metadata (channel authority, category relevance)

**5. Transcript Quality**
- *What:* Auto-generated transcripts have errors, especially for accents, technical terms. This hurts text-based retrieval.
- *Detection:* Compare search recall for videos with human-verified vs. auto-generated transcripts
- *Mitigation:* Semantic search is more robust than keyword search to transcript errors; use visual embeddings as fallback

---

#### 🌟 Strong Hire Answer (Principal)

*[Extends above with:]*

**6. Query Intent Ambiguity**
- *What:* 'Java' = programming language? Coffee? Island? Model must guess.
- *Detection:* Track user reformulation rate (user searches same query again immediately)
- *Mitigation:* Multi-intent retrieval: retrieve top candidates for each probable intent, show diverse results

**7. Coordinated Manipulation**
- *What:* Bad actors click on specific videos to inflate their ranking
- *Detection:* Velocity monitoring: anomalous click rate increase on specific videos
- *Mitigation:* Click quality scoring: weight clicks by user authenticity signals, diversity of user sources

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"Build vs. buy:*
- *Text index (Elasticsearch): buy — best-in-class for keyword search, well-supported*
- *ANN index (FAISS/ScaNN): buy — Meta's FAISS is the industry standard, used at this exact scale*
- *Text encoder (BERT): start with pre-trained (HuggingFace), fine-tune on domain data*
- *Video encoder (ViT): start with pre-trained (torchvision/timm), fine-tune*

*Cross-team sharing:*
- *Text encoder (query understanding) is used by: search, ads targeting, content moderation, recommendation*
- *Video encoder is used by: search, content moderation, copyright detection, recommendation*
- *These are platform assets — one team should own each, expose as APIs*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The platform opportunity here is a universal query understanding service.*

*Today, every ML product team at YouTube runs its own query understanding model. Search has a query encoder. Ads has a query intent classifier. Recommendations has a 'trending query' model. Each team trains these independently on its own labeled data.*

*The platform investment: a shared query encoder service that runs once per query and serves all downstream ML systems. This query representation could include:*
- *Embedding (256-dim): for semantic retrieval*
- *Intent classification (informational/navigational/transactional): for routing logic*
- *Entity mentions (video title, channel name): for structured metadata matching*

*Running this once instead of 5 times saves compute, and training it on all available signals (search clicks + ad clicks + recommendation engagement) gives a richer representation than any single team's data.*

*Roadmap:*
1. *Deploy two-tower retrieval + BM25 fusion (Q1)*
2. *Add cross-encoder ranking (Q2)*
3. *Build query understanding platform service (Q3)*
4. *Extend to multilingual (Q4)*"

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**InfoNCE (In-Batch Contrastive Loss):**
```
L = -(1/B) Σ_{i=1}^{B} log[ exp(cos(t_i, v_i) / τ) / Σ_{j=1}^{B} exp(cos(t_i, v_j) / τ) ]
```

**Cosine Similarity:**
```
cos(t, v) = (t · v) / (||t|| * ||v||)
```

**Mean Reciprocal Rank (MRR):**
```
MRR = (1/|Q|) Σ_{i=1}^{|Q|} 1/rank_i
```

**BM25 Score:**
```
BM25(q, d) = Σ_{t∈q} IDF(t) * tf(t,d) * (k+1) / (tf(t,d) + k*(1 - b + b*|d|/avgdl))
```
where k=1.5, b=0.75 are tuning parameters.

**Doubly Robust Estimator (for counterfactual evaluation):**
```
DR = (1/n) Σ_i [L_i / P(shown_i) - (1/P(shown_i) - 1) * L̂_i]
```

### Vocabulary Cheat Sheet

| Term | Definition |
|------|-----------|
| Bi-encoder (two-tower) | Separate encoders for query and document; enables ANN retrieval |
| Cross-encoder | Joint encoding of query + document; highest quality but O(N) per query |
| BM25 | Best-Match 25: probabilistic text retrieval with TF-IDF weighting |
| Retrieval-then-ranking | Two-stage: cheap retrieval for candidates, expensive ranking for top-K |
| MRR | Mean Reciprocal Rank: average of 1/rank for first relevant item |
| Recall@k | Fraction of queries where correct answer appears in top-k results |
| Knowledge distillation | Training a simpler model (student) to mimic a complex one (teacher) |
| Index freshness | How up-to-date the ANN index is relative to the video corpus |
| Exposure bias | Click data only reflects videos the previous model chose to show |
| Feature hashing | Mapping variable-length strings to a fixed-size feature vector |

### Key Numbers

| Metric | Value |
|--------|-------|
| Corpus size | 1 billion videos |
| Training pairs | 10 million (query, video) |
| End-to-end SLA | <500ms |
| Text encoder output | 256-dim |
| Video encoder output | 256-dim |
| FAISS index size (1B at 32B/v) | 32GB |
| Text encoder latency | 20-30ms (BERT) |
| ANN search latency | 10ms (FAISS IVF-PQ) |
| Cross-encoder latency (500 docs) | 50ms on GPU |
| YouTube upload rate | ~500 hours/minute |
| Frames sampled per video | 8-32 |
| Temperature τ | 0.07 |

### Rapid-Fire Day-Before Review

**Q: Why use a two-tower model instead of a cross-encoder for retrieval?**
A: Cross-encoder requires joint query-video encoding, which means O(N) inference per query (N=1B videos). Two-tower pre-computes video embeddings offline; query time is O(1) encoder + O(log N) ANN. The two-tower is ~1000x faster at inference, at some cost in quality.

**Q: Why MRR instead of Precision@k for video search?**
A: Video search often has only 1 correct answer per query. Precision@k can't distinguish a correct video at rank 1 vs. rank k. MRR directly measures the rank of the first correct result.

**Q: How do you handle queries with no training data (cold start)?**
A: BM25/Elasticsearch fallback for low-click queries. Self-supervised video embeddings (from visual content + transcripts) ensure all videos have embeddings even without click data. LLM-based query expansion to find related queries with better coverage.

**Q: What is knowledge distillation from cross-encoder to bi-encoder?**
A: Train the cross-encoder first (it's more accurate). Then use the cross-encoder to score training pairs and use those scores as soft labels to fine-tune the bi-encoder. The bi-encoder learns to approximate the cross-encoder's quality within the constraints of separate encoding.
