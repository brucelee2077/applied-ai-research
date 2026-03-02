# Visual Search System — Staff/Principal Interview Guide

## How to Use This Guide

This guide covers a complete 45-minute staff/principal ML design interview for a Pinterest-like visual search system. Each section provides the interviewer's prompt and model answers at four calibration levels. Hire and Strong Hire answers are written in first-person candidate voice.

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

*"Design a visual search system for a platform like Pinterest, where users upload or crop an image and receive visually similar images. How would you approach this?"*

### What to Clarify — 6 Dimensions

| Dimension | Question | Why It Matters |
|-----------|----------|---------------|
| **Business objective** | Are we optimizing for CTR on results, session length, or purchase conversion? | Determines relevance definition |
| **Scale** | How many images in the corpus? 1 billion? 200 billion? | Drives ANN index choice (HNSW vs. IVF-PQ) |
| **Latency** | What's the SLA for search results? | Determines whether we can afford exact NN search |
| **Data availability** | Do we have labeled similar/dissimilar pairs, or must we use self-supervision? | Shapes training approach |
| **Interaction types** | Just image-to-image? Or text+image queries (multimodal)? | If multimodal, need cross-modal embeddings |
| **Constraints** | Copyright restrictions? Duplicate detection? NSFW filtering? | Re-ranking layer requirements |

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd extract features from images using a CNN and find the most similar images using nearest neighbor search."*

Names a technique without any understanding of scale, training, or why nearest neighbor search at 100B+ images requires specialized infrastructure.

---

#### ⚠️ Weak Hire Answer

*"Can I ask — how many images are we dealing with? And what counts as 'similar'?" — gets scale and similarity definition. Doesn't ask about latency, training data, business objective, or interaction types.*

---

#### ✅ Hire Answer (Staff)

*"Before I start designing, I want to nail down a few things that fundamentally change the architecture.*

*First, scale: how many images are in the corpus? If it's under 100 million, I can use exact nearest neighbor search. Above 1 billion, I need approximate methods (ANN). Above 100 billion, I need a distributed index. I'll assume we're in the 100-200 billion range based on a Pinterest-scale system.*

*Second, what's 'similar'? Visual similarity can mean: same object, same style/aesthetic, same color palette, same scene type. These require different training objectives. For a product search use case, same-object matters. For a mood board use case, same aesthetic matters. I'll assume visual similarity means 'same object or style that a user would consider relevant.'*

*Third, latency SLA. Sub-100ms feels right for an interactive search feature. This rules out exact search over 100B images (would take seconds) and constrains the ANN index choice.*

*Fourth, training data: do we have click logs of which results users clicked? Or must I use self-supervised learning? Click logs enable supervised training with actual user feedback as labels.*

*Fifth, are there content restrictions? Pinterest has strong NSFW and copyright policies. The re-ranking layer must incorporate safety filters.*

*I'll proceed assuming: 100-200B images, <100ms SLA, click logs available, image-only queries (not multimodal), and safety filtering required."*

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to understand not just the functional requirements but the north star metric, because it determines several non-obvious design decisions.*

*On the business objective: Pinterest's stated mission is inspiration, not search utility. This means 'visually similar' should include stylistically related images even if they don't contain the exact same object. A query image of a mid-century modern chair should surface other mid-century modern furniture, not just identical chairs. This is an aesthetic embedding problem, not a strict object-identity problem. The distinction affects training: object-identity requires supervised labeling, aesthetic similarity requires contrastive learning on user engagement signals.*

*On scale: 100-200B images is a known hard constraint. At that scale, even storing one 512-dim float32 embedding per image requires 200B × 512 × 4 bytes ≈ 400TB of raw embedding storage. We can't fit this in RAM for a single-machine HNSW index. We need distributed ANN with quantization — IVF-PQ can compress to ~8-16 bytes per vector, bringing this to 1.6-3.2TB — feasible on a cluster.*

*On data: I'd want to understand the quality of click data. At Pinterest scale, 'user clicked image in search results' is noisy because of position bias (top results get clicked regardless of quality) and diversity bias (users click to explore, not necessarily because they found the most relevant item). I'd propose a cleaner signal: 'user saved image to a board' is a stronger indicator of relevance than click, and 'user repinned' is stronger still. Training on saves rather than clicks produces better embeddings for the aesthetic similarity task.*

*My design proceeds assuming: 100-200B images, <100ms SLA, save/repin as positive labels, IVF-PQ distributed index, aesthetic+object similarity as the target.*"

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

*"How do you frame this as an ML problem? What's the input, what's the output?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"Input is a query image, output is a list of similar images. I'd train a classifier to predict similarity."*

Classifier framing is wrong — you can't run a classifier against 200B images. Doesn't understand this is a retrieval/embedding problem.

---

#### ⚠️ Weak Hire Answer

*"I'd frame it as a representation learning problem: learn an embedding function that maps similar images to nearby points in a latent space. Then nearest neighbor search finds the most similar images."*

Correct framing, but no discussion of what 'similar' means, how to define positive/negative pairs, or why representation learning is the right approach vs. pairwise classification.

---

#### ✅ Hire Answer (Staff)

*"This is a metric learning / representation learning problem. The goal is to learn an embedding function f(image) → ℝ^d such that similar images map to nearby vectors and dissimilar images map to far-apart vectors in the d-dimensional space.*

*The ML task is: given a query image q, retrieve the top-K images from the corpus that are most similar to q according to some distance metric in embedding space.*

*I frame this as a retrieval problem, not a classification problem. The key distinction: for classification, you'd train a model that takes (q, candidate) pairs and outputs a similarity score — this is O(N) per query at inference time. For retrieval with pre-computed embeddings, you compute f(q) once and do ANN search — this is O(log N) per query. At 200B images, retrieval is the only feasible approach.*

*The ML objective during training is to learn f such that:*
- *sim(f(q), f(positive)) > sim(f(q), f(negative)) + margin*

*where 'positive' is an image the user found relevant and 'negative' is not. This is formalized as contrastive learning with a triplet or InfoNCE loss.*

*Input: query image (any resolution, RGB) → preprocessed to 224×224 RGB → normalized*
*Output: ordered list of (image_id, similarity_score) pairs*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The framing question has a subtle design decision embedded in it: what similarity metric should the embedding space use?*

*Dot product similarity and cosine similarity are different metrics. Dot product = ‖q‖ * ‖v‖ * cos(θ). If vectors are L2-normalized, they're identical. But unnormalized, high-norm vectors dominate dot product search — this creates a bias toward embeddings of 'prominent' or 'high-energy' images.*

*For visual search, I'd use L2-normalized embeddings with cosine similarity, which is equivalent to dot product after normalization. Normalizing removes the confound of image 'prominence' and focuses the similarity on direction in embedding space (semantic content).*

*I'd also think carefully about the embedding space dimension. Larger dimension = more expressive = better recall, but slower ANN search (linear in dimension for IVF-PQ distance computation). For 200B images, there's a sweet spot around d=256-512: enough for aesthetic detail, not so large that ANN is unusably slow. I'd ablate this empirically.*

*Finally: the query could be an image crop, not just a full image. A crop of a specific object in a cluttered scene should retrieve images of that object, not images of the full scene. This requires a crop-aware training strategy: (1) generate training pairs where (full image, crop of same image) are positives; (2) use object detection to localize salient regions and create crop-based positive pairs. This is non-trivial but important for the product experience."*

---

## Section 3: Data & Feature Engineering (8 min)

### Interviewer Prompt

*"Walk me through your data strategy. How do you construct training data? What preprocessing is involved?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use the image pixels directly as features."*

Can't use raw pixels for 200B images. No understanding of embedding models.

---

#### ⚠️ Weak Hire Answer

*"I'd use a pre-trained CNN to extract features and then fine-tune on our data. For training pairs, I'd use images the same user interacted with as positives."*

Right direction but lacks detail on preprocessing, augmentation strategy, negative sampling, label noise, and class imbalance.

---

#### ✅ Hire Answer (Staff)

*"Let me cover data schema, label construction, preprocessing, and augmentation.*

**Data sources:**
- Images table: image_id, owner_id, upload_time, manual_tags, image_url
- Users table: user_id, username, age, gender, city
- Interactions table: user_id, query_image_id, displayed_image_id, rank_position, is_click, dwell_time

**Preprocessing pipeline for every image:**
1. Decode to RGB (handle PNG, JPEG, WebP)
2. Resize to 224×224 (bilinear interpolation; or 384×384 for ViT-L)
3. Normalize: subtract ImageNet mean [0.485, 0.456, 0.406], divide by std [0.229, 0.224, 0.225]
4. Center crop at inference; random crop during training (data augmentation)

**Data augmentation (offline, before training):**
- Random horizontal flip (p=0.5)
- Random crop (80-100% of image, then resize to 224×224)
- Color jitter (brightness ±0.4, contrast ±0.4, saturation ±0.4, hue ±0.1)
- Random grayscale (p=0.2)
- Gaussian blur (p=0.5)

*Why augment: this creates diverse views of the same image. If we define (view1, view2) of the same image as a positive pair, contrastive learning will push the embedding model to be invariant to these augmentations — which is exactly the invariance we want (same content, different lighting, different crop).*

**Label construction (3 approaches, in order of signal quality):**

1. *Self-supervision via augmentation (no labeled data needed):* Positive pair = two augmented views of the same image. Negative = different images in the batch. This is SimCLR/MoCo. Advantage: infinite data. Disadvantage: doesn't incorporate actual user relevance feedback.

2. *Click-based labels:* Positive pair = (query image, clicked result image). Negative = (query image, shown but not clicked image). Advantage: captures user intent. Disadvantage: position bias (top results clicked regardless of quality), sparse labels.

3. *Save-based labels:* Positive pair = (query image, saved/repinned image). Higher quality than clicks — user took an explicit action. Use this when available.

*For production, combine: pre-train with self-supervision (handles the full 200B image corpus), then fine-tune on click/save signal (aligns with user preferences).*

**Class imbalance:** Typically 1-5% of shown images are clicked. Handle with negative sampling: for each positive (query, clicked), sample 4-9 random negatives from the batch (in-batch negatives in contrastive learning are implicit).*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to highlight the hard negative mining problem, which is crucial for training a high-quality visual search model.*

*With random in-batch negatives, the model sees mostly 'easy negatives' — a photo of a red dress vs. a photo of a mountain. Easy negatives don't teach the model fine-grained discrimination. The model plateaus at mediocre quality.*

*Hard negatives are near-misses: a red dress vs. a slightly different red dress, or an Eames chair vs. a similar mid-century chair. Training on hard negatives forces the model to learn fine-grained distinctions.*

*Hard negative mining strategies:*
1. *Semi-hard negatives (triplet mining): for each anchor, find negatives where the anchor-negative distance < anchor-positive distance + margin. These are 'harder' than random but not so hard they confuse the model early in training.*
2. *In-batch hard negatives: within each batch, for each query image, the hardest negative is the image in the batch most similar to the query that isn't the positive. Sort batch images by similarity to query, take the hardest negatives.*
3. *Offline hard negative mining: after training a first model, compute all pairwise similarities in a subset of the corpus, find the hardest negatives, and build a curated hard negative dataset for fine-tuning.*

*One danger: false negatives. An image that IS visually similar to the query but wasn't clicked (because it was never shown, or the user missed it) will be treated as a negative. This injects noise. Mitigation: use only explicit negative signals (user explicitly said 'not relevant') for the hardest negatives, or use a deduplication step to remove images too similar to the positive from the negative set.*

*Full preprocessing pipeline for production:*
1. Image decoding + RGB conversion: handle 10+ image formats
2. Resolution normalization: 224×224 for ResNet/ViT-S, 384×384 for ViT-L
3. Pixel normalization: ImageNet mean/std subtraction
4. Feature extraction: run pre-trained backbone, extract final layer or penultimate layer features
5. Dimensionality reduction: optional PCA or learned linear projection to d=256 for index efficiency*"

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

*"What model architecture would you use? Walk me through the training procedure."*

### Model Answers by Level

#### ❌ No Hire Answer

*"ResNet-50 pre-trained on ImageNet, then fine-tune on our data."*

No discussion of the training objective, loss function, or why fine-tuning for classification isn't the right approach for retrieval.

---

#### ⚠️ Weak Hire Answer

*"I'd use a siamese network or contrastive learning approach. Two images go through the same encoder, and we minimize the distance for similar pairs and maximize it for dissimilar pairs."*

Correct direction but no detail on loss function math, architecture specifics, or why contrastive > siamese.

---

#### ✅ Hire Answer (Staff)

*"Let me walk through baseline → intermediate → production, explaining why each simpler version fails.*

**Baseline: CNN encoder + L2 distance**
- Architecture: ResNet-50 pre-trained on ImageNet-1K, extract penultimate layer features (2048-dim), L2-normalize, use cosine similarity
- Why it fails: ResNet is trained for classification (1000 ImageNet categories). Its feature space clusters by object class but not by visual aesthetics or fine-grained style. Two very different chairs both cluster near 'chair' but their stylistic differences aren't captured. Recall on a real visual search benchmark is ~60-70%.

**Intermediate: SimCLR (Self-supervised contrastive learning)**

*Architecture: ResNet-50 or ViT-S backbone + projection head (2-layer MLP, 2048→512→128).*

*Training:*
1. For each image x in batch, generate two augmented views: x_1 = aug1(x), x_2 = aug2(x)
2. Run both through the encoder+projection head: z_1 = f(x_1), z_2 = f(x_2)
3. Apply InfoNCE (NT-Xent) loss:
```
L = -log[ exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) ]
```
where sim is cosine similarity, τ = 0.07, and the denominator sums over all 2N-2 other examples in the batch (the 2N-2 negatives).

*Why this works: the model learns to map two views of the same image to similar vectors, and different images to dissimilar vectors. The encoder is forced to extract the invariant content of the image, ignoring augmentation-specific features like lighting and crop.*

*Why it fails for retrieval: self-supervised learning doesn't optimize for user-defined relevance. It learns that two crops of the same image are similar, but not that a user's query image of a red chair is most similar to other red chairs with similar style. We need a human-feedback signal.*

**Production: Supervised Contrastive Learning + Hard Negatives**

*Architecture: ViT-B/16 backbone (better than ResNet for semantic understanding), projection head, embedding dimension d=256 (post-projection).*

*Training (2 phases):*
1. Pre-train with SimCLR on full 200B image corpus (unsupervised — no labels needed)
2. Fine-tune with click/save labels: positive pairs = (query, saved_result), hard negatives = similar but not saved images

*Loss function (Supervised Contrastive):*
```
L_supcon = -Σ_{i∈I} (1/|P(i)|) * Σ_{p∈P(i)} log[ exp(sim(z_i, z_p)/τ) / Σ_{a∈A(i)} exp(sim(z_i, z_a)/τ) ]
```
where P(i) is all positive examples for anchor i, A(i) = all other examples in batch.

*Key advantage of 2-phase training: the self-supervised pre-training gives a strong generic visual feature extractor. The fine-tuning step then aligns it with user preference signals, without starting from scratch.*

*At inference: remove projection head, use backbone output (or frozen backbone + fine-tuned linear layer). The 256-dim embedding is computed once per image and indexed in ANN.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to discuss three things that go beyond standard contrastive learning: the choice between ResNet and ViT, the temperature sensitivity, and the embedding space geometry.*

**ResNet vs. ViT for visual search:**

*ViT (Vision Transformer) has largely displaced ResNet for visual embeddings in production systems. The reason: ViT uses global self-attention, which captures long-range relationships within an image — important for aesthetic understanding where the relationship between different parts of an image (e.g., color harmony across an interior design image) matters. ResNet's local convolutional filters miss these global patterns.*

*ViT-B/16 (patch size 16, base model) is the sweet spot: better than ResNet-50 on visual search benchmarks, comparable inference speed (with INT8 quantization), and smaller than ViT-L/14.*

**Temperature τ in InfoNCE:**

*Temperature controls the concentration of the similarity distribution. Low τ = hard attention on the nearest negatives. High τ = soft attention spread across all negatives.*

*Effect on training:*
- τ too low (< 0.05): model collapses — all embeddings pushed to antipodal points, gradients vanish
- τ too high (> 0.3): model doesn't learn fine distinctions — all negatives treated equally
- Optimal range: τ = 0.07-0.15 for visual search; tune via held-out retrieval recall

**Embedding space geometry:**

*One underappreciated issue: trained embeddings tend to cluster into 'cones' in embedding space (the dimensional collapse problem). Embeddings use only a small subspace of the full d-dimensional space, which limits ANN recall because the effective dimensionality is lower than d.*

*Mitigation:*
1. L2 normalization before computing loss (forces embeddings onto a hypersphere)
2. Uniformity loss: add L_uniform = log E[exp(-2 * ||z_i - z_j||²)] as a regularizer to spread embeddings uniformly across the hypersphere
3. Batch normalization in the projection head

*The uniformity + alignment loss from Wang & Isola (2020) is the principled formulation:*
```
L_alignment = E_{(x,y)~p_pos} ||f(x) - f(y)||²
L_uniformity = log E_{(x,y)~p_data} exp(-2||f(x) - f(y)||²)
L_total = L_alignment + λ * L_uniformity
```
*This explicitly optimizes for two desiderata: aligned embeddings for similar pairs, and uniformly distributed embeddings across the sphere.*"

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

*"How do you measure the quality of this system, offline and online?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd compute accuracy by checking if the returned images are in the same category as the query."*

Category matching is a proxy, not a real quality metric. Doesn't understand ranking quality.

---

#### ⚠️ Weak Hire Answer

*"I'd use Precision@k offline. Online I'd measure CTR."*

Precision@k is problematic for graded relevance. CTR has position bias.

---

#### ✅ Hire Answer (Staff)

*"For offline evaluation, I'd use nDCG@10 as the primary metric.*

*Why nDCG and not Precision@k:*
- Precision@k treats all results as equally good or bad. nDCG gives graded relevance (saved image > clicked image > impression) and discounts lower positions.
- Precision@k doesn't penalize having a relevant image at rank 3 vs. rank 1. nDCG does.

*nDCG computation:*
```
DCG@k = Σ_{i=1}^{k} (2^rel_i - 1) / log₂(i+1)
IDCG@k = DCG@k for the ideal ordering
nDCG@k = DCG@k / IDCG@k
```
*where rel_i ∈ {0, 1, 2}: 0 = not relevant, 1 = clicked, 2 = saved. This is a graded relevance formulation.*

*Why not MRR (Mean Reciprocal Rank): MRR only considers the first relevant result. For visual search, users often benefit from multiple relevant results, so MRR misses the quality of the full list.*

*Why not Recall@k alone: Recall@k doesn't account for ranking order within the top-k.*

*Evaluation protocol:*
1. Hold out 10% of user sessions (temporal split — use newest sessions)
2. For each session, take the query image, use the model to retrieve top-50 results
3. Compute nDCG@10 against the ground truth (images the user actually clicked/saved)
4. Average across sessions

*Online metrics:*
- CTR@5 (click-through rate on top-5 results) — but correct for position bias
- Save rate (saves per search session) — higher quality signal
- Session depth (how many results users scroll through) — measures quality/diversity
- Zero-result sessions (sessions where user clicks nothing and abandons) — lower is better*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The most important thing about evaluation for visual search is that 'ground truth' is not well-defined. Unlike web search where there's a single correct answer, visual search has a distributional notion of relevance. Any of 10,000 red chairs might be 'relevant' for a red chair query.*

*This means standard IR metrics can mislead. nDCG@10 measured against click history only captures what the previous model showed. A new model that surfaces high-quality but never-previously-shown images will look worse offline even if users would love those results online.*

*Better evaluation approaches:*

1. *Expert annotation:* For a held-out set of 1000 queries, have human annotators rate the top-10 results on a 5-point relevance scale. Compute mean nDCG@10 against annotator judgments. More expensive but less biased.

2. *Interleaving:* In a small online experiment, interleave results from model A and model B for the same query. Users implicitly compare both. The model whose results get more saves/clicks in the interleaved position wins. More sensitive than A/B (requires 10x fewer users).

3. *Embedding quality metrics:*
   - Alignment: average cosine similarity between positive pairs in a held-out set
   - Uniformity: average pairwise similarity across random pairs (lower = more uniform)
   These don't require labeled data and correlate with downstream retrieval quality.

*For the offline-online gap: visual search has a strong novelty effect. New model often shows images users haven't seen before → initial CTR boost from novelty, but long-term quality matters more. Run A/B tests for at least 2 weeks to separate novelty from sustained quality.*"

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

*"Walk me through the serving architecture. How do you serve results at scale with low latency?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd run the model on a server, compare the query embedding to all image embeddings, and return the most similar ones."*

Linear scan over 200B images takes ~minutes. Completely infeasible.

---

#### ⚠️ Weak Hire Answer

*"I'd use approximate nearest neighbor search. Pre-compute all image embeddings and build an index."*

Right direction, no detail on which ANN algorithm, why, latency numbers, or serving pipeline.

---

#### ✅ Hire Answer (Staff)

*"The serving architecture has two distinct pipelines: an offline indexing pipeline and an online prediction pipeline.*

**Offline Indexing Pipeline:**
1. Run all 200B images through the trained embedding model (ViT-B/16) → 256-dim L2-normalized embeddings
2. Build ANN index: IVF-PQ (Inverted File with Product Quantization)
   - IVF: partition embedding space into K=65536 clusters (k-means)
   - PQ: compress each 256-dim residual into 32 subquantizers × 8 bits = 32 bytes per vector
   - Index size: 200B × 32 bytes = 6.4TB (distributed across ~100 index servers)
3. Store image metadata (image_id, thumbnail URL, owner) in a key-value store (Redis/BigTable)
4. Re-index daily to add new images; incremental index updates for freshness

**Online Prediction Pipeline (per query):**
1. User uploads/crops query image → edge CDN processes crop
2. Embedding service: run ViT-B forward pass on 224×224 query → 256-dim embedding (5-10ms on GPU)
3. ANN service: query IVF-PQ index with nprobe=64 → top-500 candidate image IDs (10-20ms)
4. Re-ranking service (20ms):
   - Fetch image metadata for 500 candidates
   - Apply safety filter (NSFW classifier)
   - Remove duplicates (near-exact copies)
   - Apply diversity constraint (max 5 images per owner)
   - Return top 50 results to client
5. Total: ~40-50ms end-to-end

**ANN trade-offs:**

| Algorithm | Recall@100 | Latency (200B) | Memory/vector |
|-----------|-----------|---------------|---------------|
| Exact scan | 100% | ~minutes | 1024 bytes (fp32, d=256) |
| HNSW | 98% | ~30ms (100M only) | ~400 bytes |
| IVF-PQ | 87-92% | ~15ms | ~32 bytes |

*For 200B images, only IVF-PQ is feasible in RAM. HNSW would require 80TB RAM for the graph structure — not viable even on large clusters.*

**Training-serving skew prevention:**
- The preprocessing (resize, normalize) is implemented once as a shared library
- Embeddings are logged at query time and used for training data collection
- Model version is tagged in every embedding; incompatible versions are detected automatically*"

---

#### 🌟 Strong Hire Answer (Principal)

*"Let me go deeper on the distributed index architecture and the index freshness problem.*

**Distributed IVF-PQ at 200B scale:**

*A single IVF-PQ index over 200B images at 32 bytes/vector = 6.4TB. This doesn't fit on one machine. We shard the index: partition the 200B images into 100 shards of 2B images each. Each shard has its own IVF-PQ index fitting in ~64GB RAM per machine.*

*Query fanout: a query broadcasts to all 100 shards simultaneously, each shard returns top-K results, a merger aggregates and re-ranks. This adds ~5ms for the aggregation step but allows horizontal scaling.*

*An alternative: hierarchical IVF. A coarse index (on a single machine) narrows the search to relevant image clusters across the full 200B corpus. Only a subset of shards are queried. This reduces the fanout from 100 to ~10, but at the cost of ~2% recall loss.*

**Index freshness:**

*200B images × 1% daily upload rate = 2B new images/day. We can't re-index the entire corpus daily.*

*Incremental indexing strategy:*
1. New images (< 7 days old) go into a 'hot index' (HNSW on a small number of machines, 14B images max = 5.6TB) — exact recall, fast query
2. Old images (> 7 days) go into the 'cold index' (IVF-PQ, full 200B corpus)
3. Queries fan out to both hot and cold indexes, merge results

*This gives better recall for new images (HNSW precision) while maintaining cost efficiency for the full corpus.*

**Embedding model versioning:**

*When we retrain the embedding model (e.g., fine-tune on new user data), all pre-computed index embeddings become stale — the new model's embedding space is different. We can't re-index 200B images immediately.*

*Mitigation: train a 'compatibility layer' — a linear projection from old embedding space to new embedding space, learned on a sample of images that have been embedded by both models. Apply this projection to old embeddings without re-indexing. Acceptable quality loss (~1-2% recall) while full re-indexing completes in the background.*"

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

*"What are the key failure modes?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

**5 Failure Modes:**

**1. Near-Duplicate Flooding**
- *What:* Corpus has many near-identical images (scraped from same source). ANN returns 10 copies of the same image.
- *Detection:* Monitor average pairwise cosine similarity of top-10 results. High average = low diversity.
- *Mitigation:* Max marginal relevance (MMR) in re-ranking: iteratively select images that are both highly relevant AND dissimilar to already-selected results.

**2. Adversarial Queries**
- *What:* Users query with NSFW images or images designed to elicit harmful content via visual similarity.
- *Detection:* NSFW classifier on query image before embedding.
- *Mitigation:* Block queries that trigger NSFW classifier. Separately, audit ANN results through a safety filter before returning to user.

**3. Distribution Shift — New Image Styles**
- *What:* Model trained on 2020-era image styles performs poorly on new aesthetic trends (e.g., AI-generated art).
- *Detection:* Monitor embedding distribution shift over time (KL divergence on quarterly basis).
- *Mitigation:* Quarterly fine-tuning on recent saves/clicks data.

**4. Crop-Query Failure**
- *What:* User crops a small specific object from a large scene. Model embeds the full crop context including background, returning images with similar backgrounds rather than similar objects.
- *Detection:* Evaluate on crop-specific benchmark: user-reported satisfaction for crop queries.
- *Mitigation:* Training pairs that include (object crop, other images of same object) as positives. Object detection + attention mechanism to focus embedding on the salient object.

**5. Cold Start for New Image Categories**
- *What:* New product category (e.g., a new type of fashion item) has no training examples. Model can't find similar images.
- *Detection:* Monitor zero-click rate for queries about newly trending search terms.
- *Mitigation:* Few-shot adaptation: use CLIP's zero-shot embeddings as a fallback when the fine-tuned model has low confidence.

---

#### 🌟 Strong Hire Answer (Principal)

*[Extends above with:]*

**6. Copyright and Content Attribution**
- *What:* Returning visually similar copyrighted images without attribution violates DMCA.
- *Detection:* Flag images with known copyright metadata.
- *Mitigation:* Pre-index copyright status; filter or annotate results with license info.

**7. Feedback Loop — Amplifying Trending Aesthetics**
- *What:* If fine-tuning on saves, popular aesthetic styles accumulate more saves → more representation in training → embedding space biased toward popular styles → niche aesthetics become harder to find.
- *Detection:* Track recall separately for 'tail' image categories (rare styles, niche cultural aesthetics).
- *Mitigation:* Upsampling rare categories in fine-tuning; separate evaluation track for diversity metrics.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

*"How does this fit into the broader platform?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"The embedding infrastructure built for visual search is a platform asset, not a product feature.*

*Build vs. buy:*
- ANN index: use FAISS (Meta OSS) — don't build custom. FAISS is used by Facebook, Google, Spotify at scale.
- Embedding model: build (fine-tune from OSS backbone like ViT). Core competitive advantage.
- Feature store for embeddings: build on top of Redis Cluster. Operational complexity is manageable.

*Cross-team sharing:*
- The same image embedding model powers: visual search, ad relevance (show ads similar to content user is looking at), content moderation (find images similar to known NSFW images), duplicate detection.
- One team maintains the embedding model; other teams consume embeddings as an API.

*Org design:*
- 'Embeddings platform' team: owns model training, indexing, serving, monitoring
- 'Visual search product' team: owns retrieval pipeline, re-ranking, UI
- Clear API boundary between them*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The strategic opportunity here is a universal embedding layer.*

*Every Pinterest product uses images. The pin recommendation system, the visual search, the ad targeting system, the board organization system — all of them need to understand visual similarity. Currently, each team trains its own embedding model with its own training signal and its own index. This is ~5x the infrastructure cost and creates embedding spaces that are incompatible with each other.*

*The platform investment: train a single foundation visual embedding model on all available signals (saves, clicks, pins, board co-occurrence, caption-image pairs), build one shared embedding index, and expose it as an internal API. Each product team then adds a lightweight task-specific fine-tuning layer on top.*

*This is the ViT → CLIP → multimodal journey. The natural extension of this visual search system is a unified visual + text embedding space (CLIP-style), so users can query with either an image or text and get the same quality results. This is the platform that enables semantic search, not just visual search.*

*Roadmap:*
1. Launch visual-only search with SimCLR + fine-tuning (Q1)
2. Build shared embedding platform serving 3+ teams (Q2-Q3)
3. Extend to multimodal (image + text) CLIP-style model (Q4)
4. Expose embedding API to external creators (Year 2)*"

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**InfoNCE (NT-Xent) Contrastive Loss:**
```
L = -log[ exp(sim(z_i, z_j) / τ) / Σ_{k=1, k≠i}^{2N} exp(sim(z_i, z_k) / τ) ]
sim(u, v) = u^T v / (||u|| * ||v||)   (cosine similarity)
```

**Supervised Contrastive Loss:**
```
L_supcon = -Σ_{i∈I} (1/|P(i)|) Σ_{p∈P(i)} log[ exp(sim(z_i,z_p)/τ) / Σ_{a∈A(i)} exp(sim(z_i,z_a)/τ) ]
```

**nDCG:**
```
DCG@k = Σ_{i=1}^{k} (2^rel_i - 1) / log₂(i+1)
nDCG@k = DCG@k / IDCG@k
```

**Uniformity + Alignment (Wang & Isola 2020):**
```
L_align = E_{(x,y)~pos} ||f(x) - f(y)||²
L_uniform = log E_{x,y~data} exp(-2||f(x)-f(y)||²)
L = L_align + λ * L_uniform
```

**Cosine Similarity:**
```
cos(q, k) = (q · k) / (||q|| * ||k||)
```

### Vocabulary Cheat Sheet

| Term | Definition |
|------|-----------|
| Contrastive learning | Training by comparing positive and negative pairs to learn embeddings |
| SimCLR | Simple framework for contrastive learning of visual representations |
| InfoNCE | Loss for contrastive learning using all batch items as negatives |
| Hard negative mining | Selecting challenging negatives near the decision boundary |
| IVF-PQ | Inverted File + Product Quantization: memory-efficient ANN index |
| HNSW | Hierarchical navigable small world: graph-based ANN index |
| L2 normalization | ||v|| = 1 — projecting embeddings onto the unit hypersphere |
| Dimensional collapse | Embeddings using only a low-dimensional subspace of ℝ^d |
| MMR | Maximum Marginal Relevance: diversifying re-ranking |
| Augmentation | Random transformations creating multiple views of the same image |
| Temperature τ | Controls sharpness of softmax in contrastive loss |

### Key Numbers

| Metric | Value |
|--------|-------|
| Corpus size | 100-200 billion images |
| End-to-end SLA | <100ms |
| Embedding dimension | 256-512 |
| IVF-PQ memory/vector | 32 bytes |
| IVF-PQ total memory (200B) | 6.4TB |
| HNSW recall @top-100 | ~98% |
| IVF-PQ recall @top-100 | ~87-92% |
| ANN search latency | 10-20ms (IVF-PQ) |
| ViT-B forward pass | 5-10ms on GPU |
| Augmentation batch size | 4096-8192 (SimCLR) |
| Temperature τ | 0.07-0.15 |
| Embedding index shards | ~100 for 200B images |

### Rapid-Fire Day-Before Review

**Q: Why IVF-PQ over HNSW for 200B images?**
A: HNSW requires ~400 bytes/vector for the graph = 80TB for 200B vectors — impossible in RAM. IVF-PQ compresses to 32 bytes/vector = 6.4TB — feasible on a cluster.

**Q: What's the key difference between SimCLR and supervised contrastive learning?**
A: SimCLR uses augmented views of the same image as positives (no labels). Supervised contrastive uses actual user feedback (clicks/saves) as positives, giving better alignment with user intent.

**Q: How do you handle near-duplicate flooding in results?**
A: Maximum Marginal Relevance (MMR) in re-ranking: iteratively select images that are both relevant AND dissimilar to already-selected results.

**Q: What is temperature τ in InfoNCE, and what happens if it's too low?**
A: τ controls softmax sharpness. Too low (< 0.05): model attends only to the very nearest negatives, gradients saturate, training instability. Too high (> 0.3): all negatives treated equally, model doesn't learn fine distinctions.

**Q: Why pre-train self-supervised then fine-tune on clicks?**
A: Self-supervised pre-training gives a strong visual feature extractor without needing labeled data (scales to full 200B corpus). Fine-tuning aligns the embedding space with user relevance feedback. The combination outperforms either alone.
