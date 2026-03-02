# Video Recommendation System — Staff/Principal Interview Guide

## How to Use This Guide

This guide simulates a complete 45-minute staff/principal ML design interview for a YouTube-like homepage video recommendation system. Each section contains the interviewer's prompts and model answers at four calibration levels. Study the **Hire** and **Strong Hire** answers carefully — they are written in first-person candidate voice so you can internalize exact phrasing, not just concepts.

**Time allocation:**
- Section 1 (Clarification): 5 min
- Section 2 (Framing): 5 min
- Section 3 (Data/Features): 8 min
- Section 4 (Model): 12 min
- Section 5 (Evaluation): 5 min
- Section 6 (Serving): 7 min
- Section 7 (Edge Cases): 5 min
- Section 8 (Principal): 3 min

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

*"Design a video recommendation system for a platform like YouTube. The system should recommend videos to users on the homepage. Walk me through how you'd approach this."*

### What to Clarify — 6 Dimensions

| Dimension | Question to Ask | Why It Matters |
|-----------|----------------|----------------|
| **Business objective** | Are we optimizing for engagement, revenue, or user satisfaction? | Determines ML objective — watch time vs. clicks vs. explicit likes |
| **Scale** | How many videos, how many users, what's the growth rate? | Drives architecture — billions of videos requires multi-stage retrieval |
| **Latency** | What's the SLA? Sub-100ms? Sub-500ms? | Determines whether we can use heavy models in ranking |
| **Data availability** | What interaction signals do we have — clicks, watch time, likes, searches? | Determines feature set and label construction |
| **Interaction types** | Homepage recommendation vs. related videos vs. search? | Homepage = pure collaborative filtering context; related = content-based |
| **Constraints** | Cold start requirements? Multilingual? Privacy regulations (GDPR)? | Cold start forces content features in both towers |

### Model Answers by Level

#### ❌ No Hire Answer

*"Sure, I'd use a neural network to recommend videos. I'd take user features and video features and train a model to predict which videos the user wants to watch. Let me start drawing the architecture."*

**Why it fails:** Skips clarification entirely. Jumps straight to solution. No business objective defined. No scale discussed. Will build the wrong thing. Interviewer has no idea if this candidate can scope ambiguous problems — a core staff-level skill.

---

#### ⚠️ Weak Hire Answer

*"Before I start, can I ask — are we doing homepage recommendations or related videos? And roughly how many videos are we dealing with?"*

Gets the surface scope question and scale. Doesn't ask about ML objective, latency constraints, available interaction signals, or cold start. Will proceed with incomplete information and likely miss the multi-objective complexity of the problem.

---

#### ✅ Hire Answer (Staff)

*"Before I dive in, I want to make sure I'm building the right thing, so let me ask a few clarifying questions.*

*First, on the business objective — are we primarily optimizing for watch time, click-through rate, or some combination of explicit and implicit engagement signals? The reason this matters is that optimizing purely for clicks leads to clickbait, optimizing purely for completion rate biases toward short videos, and optimizing purely for watch time can favor addictive but low-quality content. I'd want to understand the product team's north star metric.*

*Second, on scale — how many videos are in the corpus? How many daily active users? I'm assuming we're in the billions-of-videos range, which has major architectural implications for how I'll design the retrieval layer.*

*Third, on latency — what's the end-to-end SLA? The answer determines whether I can afford a heavy ranking model. Sub-200ms end-to-end is very different from sub-50ms.*

*Fourth, on data — what interaction signals are available? Specifically, do we have watch duration, completion rate, likes, shares, comments, and search queries? Each of these has different noise levels and different correlations with true user satisfaction.*

*Fifth, is this homepage recommendation specifically, or does this also include related videos and notifications? These have very different ML framings — homepage needs pure personalization without a query, while related videos is conditioned on a seed video.*

*Finally, cold start — do we have a significant fraction of new users with no history? And are there new videos being uploaded continuously? Both affect how I architect the system.*

*Based on those answers, I'll tailor the design. My assumption going in is: we're YouTube-scale (~10 billion videos), the SLA is under 200ms, we have rich interaction signals, the focus is homepage recommendation, and we need to handle cold start for both new users and new videos."*

**What makes this Hire-level:** Asks all 6 dimensions. Explains *why* each dimension matters. Summarizes assumptions before moving on. Shows that asking good questions is itself a signal of experience.

---

#### 🌟 Strong Hire Answer (Principal)

*"Happy to dive in — let me first make sure we're solving the right problem, because the right system design depends heavily on which constraints actually bind.*

*On the business objective: I want to understand not just what we're optimizing today, but what the business regrets optimizing in retrospect. YouTube famously optimized watch time and found it created radicalization feedback loops. A principal-level system design should anticipate second-order effects. Are we adding any long-term user satisfaction signal, like survey scores or explicit dislikes, as a constraint on top of the engagement objective? I'd propose we treat this as a constrained optimization: maximize watch time subject to a minimum threshold on user-reported satisfaction, rather than as pure watch time maximization.*

*On scale: I'll assume ~10 billion videos and hundreds of millions of daily active users. The important number isn't just the corpus size — it's the ratio of videos per user session. If we're targeting a feed of 20 recommendations and the corpus has 10 billion videos, we need a retrieval system that can efficiently narrow that by a factor of 500 million. That's a multi-stage pipeline question, not just a model question.*

*On latency: I want a breakdown. The 200ms budget isn't monolithic — there's network latency, feature retrieval, model inference, and reranking. For a staff-level design I'd want to allocate roughly: 20ms feature retrieval, 30ms for candidate generation ANN search, 80ms for ranking model inference, 20ms for reranking and business logic, leaving 50ms margin. These numbers constrain which models are feasible.*

*On data: the interaction signals matter not just for features but for label construction. One thing many candidates miss is the label delay problem — if we define a positive label as 'user watched >50% of video', that label is only available after the user finishes watching, which could be hours later. This affects how we structure the training pipeline.*

*I'd also want to understand the privacy constraints. GDPR's right to be forgotten means user embeddings need to be deletable. That affects the model architecture — we need to be able to re-compute a user's embedding without their history, which is a design constraint.*

*With that context, I'll proceed assuming: 10B videos, <200ms SLA, rich interaction data, homepage recommendations, cold start significant for new users, long-term engagement considered alongside short-term watch time, and GDPR compliance required."*

**What makes this Strong Hire:** Identifies the *regret* problem with the business objective proactively. Allocates the latency budget quantitatively before even starting. Raises the label delay problem unprompted. Raises GDPR constraints the interviewer didn't mention. Shows platform/org thinking (second-order effects, constrained optimization framing).

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

*"How do you frame this as an ML problem? What's the input, what's the output, and what ML objective would you use?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"The input is user features and video features, and the output is a ranked list of videos. I'd train a classifier to predict whether the user likes each video."*

No justification of objective choice. Doesn't discuss the business objective → ML objective translation. Doesn't discuss task category or why ranking/retrieval vs. classification matters at scale.

---

#### ⚠️ Weak Hire Answer

*"I'd frame this as a ranking problem. The model takes user features and video features as input and outputs a relevance score per video. I'd use watch time as the label — longer watch time means more relevant."*

Reasonable but misses: the clickbait problem with raw engagement, multi-objective complexity, the difference between pointwise and listwise ranking, and why retrieval-then-ranking is necessary at scale.

---

#### ✅ Hire Answer (Staff)

*"Let me work through the business objective → ML objective translation explicitly, because there are several wrong choices that seem right at first.*

*The business wants to increase user engagement. The naive translation is 'maximize predicted watch time per session.' The problem with that is watch time conflates video quality with video length — a 10-minute mediocre video that a user half-watches looks better than a 2-minute excellent video watched completely.*

*The better framing is: predict relevance, where relevance is a combination of explicit and implicit signals. I'd define a positive label as: the user either (1) explicitly liked or shared the video, OR (2) watched at least 50% of it. This is more robust to length bias and better correlates with actual satisfaction.*

*With that label definition, the ML task is a binary classification problem at training time: given a (user, video) pair, predict P(user finds video relevant). But at serving time, we're solving a ranking problem: given a user, return the top-K most relevant videos from a corpus of 10 billion.*

*This distinction is important: the same model serves two different inference modes. During candidate generation, we run approximate nearest neighbor search on pre-computed video embeddings — this is a retrieval problem. During ranking, we score a small candidate set with a richer model — this is a classification problem. We need architectures that support both.*

*The input to the overall system is: user ID + context (device, time of day, location) + optional session context (recently watched). The output is an ordered list of ~20 video recommendations with predicted relevance scores.*

*I'd categorize this as a hybrid retrieval-ranking system, not a pure classification system, because the scale makes exhaustive pairwise scoring infeasible."*

---

#### 🌟 Strong Hire Answer (Principal)

*"The framing question is actually where most systems go wrong, so I want to be careful here.*

*First, the standard recommendation objective — maximize predicted engagement — has a known failure mode: it creates feedback loops. If we predict that users in demographic D like content of type T, we show them more of T, which increases their interaction with T, which makes our model predict even higher relevance for T. This is the filter bubble problem, and it's a direct consequence of optimizing a single engagement proxy.*

*A more principled framing is constrained multi-objective optimization: maximize primary engagement (watch time or completion rate) subject to a diversity constraint (the recommendation set covers at least K distinct topic clusters) and a user satisfaction floor (weekly active user rate stays above a threshold). This is architecturally more complex but avoids the degenerate feedback loop.*

*For the ML task specifically: I'd use a two-stage formulation. Stage 1 is a retrieval task — a two-tower model that learns a shared embedding space for users and videos, trained with InfoNCE contrastive loss. The objective is to bring (user, relevant-video) embeddings close together and push (user, irrelevant-video) embeddings apart. This is not a pointwise classification task — it's a contrastive learning task, which is more appropriate because we care about relative ordering, not absolute scores.*

*Stage 2 is a ranking task — a more complex model that takes the ~1000 retrieved candidates and scores them with rich cross-feature interactions. Here I'd use a multi-task learning objective that jointly predicts: P(click), P(watch >50%), P(like), P(share), P(explicit skip). Each task gets its own head but shares a base tower. The final ranking score is a weighted combination: score = w1 * P(watch>50%) + w2 * P(like) + w3 * P(share) - w4 * P(skip).*

*The reason I separate retrieval and ranking with different objectives is that they serve different purposes: retrieval needs to be computationally compatible with ANN search (requires embedding similarity), while ranking can afford heavy cross-feature interaction models (runs on a small candidate set).*

*Input specification:*
- *Retrieval: user embedding (learned from demographics + history) + context embedding*
- *Ranking: concatenated user features, video features, and user-video interaction context*

*Output: ordered list of video IDs with predicted multi-task relevance scores, final ranking by weighted sum."*

---

## Section 3: Data & Feature Engineering (8 min)

### Interviewer Prompt

*"Walk me through the data sources, how you'd construct labels, and what features you'd use. What preprocessing is involved?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use user watch history as features and videos the user has watched as positive labels. I'd join the user table with the video table and interaction logs."*

No feature taxonomy. No encoding strategy. No discussion of label noise or class imbalance. Doesn't know that watch history needs to be aggregated into embeddings, not used raw.

---

#### ⚠️ Weak Hire Answer

*"I'd use user demographics (age, gender, location) and video metadata (title, category, duration) as features. For labels, videos the user watched would be positives, and random unshown videos would be negatives."*

Better, but missing: encoding strategies per feature type, user historical interaction features (the most important features), label noise due to accidental clicks, class imbalance handling, temporal features.

---

#### ✅ Hire Answer (Staff)

*"Let me walk through this systematically by feature type, since different feature types require very different encoding strategies.*

**Data sources:**
- Video table: video_id, duration, language, title, tags, upload_timestamp, like_count, view_count
- User table: user_id, age, gender, city, country, account_created_at
- Interaction log: user_id, video_id, interaction_type (click/watch/like/share/search/skip), watch_duration, impression_timestamp

**Label construction:**
I'd define a positive label as: `watch_duration / video_duration >= 0.5 OR interaction_type = 'like'`. This is a binary label combining implicit (watch completion) and explicit (like) signals. The threshold of 50% watch duration is a design choice — you could tune it.*

*A critical issue: positive labels are delayed. The label 'watched 50%+' is only observable after the video finishes playing, which could be 30+ minutes after the recommendation was served. For training, this is fine since we're using historical data. But for continual learning pipelines, label delay means we can't immediately use fresh data — we need a wait window of ~2 hours before a training example is complete.*

*Class imbalance: in a typical recommendation system, maybe 5-10% of impressions result in a positive interaction. This is severe imbalance. I'd handle it through negative sampling — for each positive, I'd include 4-9 randomly sampled negatives from the same time window, plus hard negatives (videos shown but explicitly skipped).*

**Feature taxonomy and encoding:**

*Video features:*
- `video_id`: embedding lookup, dim=128 (learned end-to-end)
- `duration`: bucketized into [0-1min, 1-5min, 5-15min, 15-60min, 60+min] → one-hot, dim=5
- `language`: embedding lookup, dim=16
- `title`: BERT embeddings (pre-trained, fine-tuned), 768-dim → projected to 128-dim via linear layer
- `tags`: CBOW (average of word embeddings), pre-trained on video corpus, projected to 128-dim

*User demographic features:*
- `age`: bucketized [<18, 18-25, 25-35, 35-50, 50+] → one-hot
- `gender`: one-hot, handle missing with 'unknown' category

*User contextual features (query-time):*
- `time_of_day`: bucketized into 4-hour windows → one-hot, dim=6
- `device_type`: [mobile, tablet, desktop, TV] → one-hot
- `day_of_week`: one-hot, dim=7

*User historical features (most important):*
- `watched_video_embeddings`: take the last 50 watched video IDs, look up their embeddings, average them → 128-dim 'watch history embedding'
- `liked_video_embeddings`: same process for liked videos → 128-dim
- `search_history`: take last 20 search queries, embed each with BERT, average → 128-dim

*These averaged embeddings are the single most predictive feature category. A user's watch history, encoded as an average of video embeddings, creates a 'taste profile' vector in the same embedding space as videos — this is what powers collaborative filtering.*

*For training, I'd split data temporally: train on interactions from weeks 1-8, validate on week 9, test on week 10. Never shuffle and split randomly — that would cause future data leakage."*

---

#### 🌟 Strong Hire Answer (Principal)

*"I'll cover the feature engineering, but I want to highlight two non-obvious design decisions that junior engineers miss.*

*First, the watch history embedding aggregation strategy matters more than people realize. Simply averaging the last N video embeddings treats all history equally. Better approaches: (1) exponential decay weighting — recent watches weighted higher than old ones; (2) attention-weighted aggregation — learn which past videos are most relevant for predicting the current request; (3) separate the 'short-term interest' embedding (last 5 videos) from the 'long-term interest' embedding (last 100 videos) and use both as separate features. YouTube's production system uses a learned attention mechanism over watch history rather than simple averaging.*

*Second, label construction has a hidden bias problem. If we define positives as 'watched >50%', then our training data is biased toward videos that were recommended in the first place. Videos that were never shown get no labels. This is the exposure bias or selection bias problem. The model learns to predict engagement on the distribution of historically recommended content, not on the full video corpus. This systematically underestimates the relevance of videos that have rarely been shown — typically newer videos and niche content. Mitigation: (1) counterfactual debiasing via inverse propensity scoring, (2) exploration strategies (ε-greedy: 5% of recommendations are random to gather unbiased signal), (3) separate the exploration model from the exploitation model.*

*Full feature taxonomy:*

*Video features (video tower input):*
- ID embedding: dim=256, learned
- Title: BERT [CLS] token, 768→256 via 2-layer MLP
- Tags: CBOW, 300-dim word2vec → average → 256-dim linear projection
- Duration: log-transform then bucketize (log scale handles power-law distribution), 6 buckets → embedding dim=8
- Upload recency: log(hours_since_upload), continuous scaled
- Engagement stats: log(view_count), log(like_count), like/view ratio — these are real-time statistics from the feature store, updated hourly

*User features (user tower input):*
- Demographics: age bucket embedding (dim=8), gender embedding (dim=4)
- Watch history: last 50 videos → look up video embeddings → weighted average with recency decay: e^(-λt) weight per video. Output: 256-dim
- Liked videos: last 20 → weighted average, 256-dim
- Search history: last 10 queries → BERT each → average → 256-dim
- Contextual: time_of_day embedding (dim=8), device_type (dim=4), day_of_week (dim=8)

*Cross-features (only in ranking stage, not retrieval):*
- User-video language match: binary
- Video topic overlap with watch history: cosine similarity between topic distributions
- User's historical engagement with this video's creator/channel

*For feature freshness: static user demographics are computed daily. Watch history embeddings are updated every 5 minutes (streaming). Video engagement stats (views, likes) are updated hourly via a stream processing job. The feature store has separate online (Redis, low-latency) and offline (BigTable, high-throughput) stores.*"

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

*"Walk me through your model choices. Start with a baseline and progress to what you'd deploy in production."*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use a deep learning model with user and video features concatenated together, then several dense layers to predict relevance."*

No architectural justification. Doesn't understand why concatenating user and video features doesn't scale to billions of videos. No discussion of loss functions.

---

#### ⚠️ Weak Hire Answer

*"I'd start with collaborative filtering using matrix factorization, then upgrade to a two-tower neural network. The two-tower model has one tower for users and one for videos, and they're trained together."*

Names the right models but doesn't explain why each works mechanistically, why matrix factorization fails, what the training objective is, or how inference works.

---

#### ✅ Hire Answer (Staff)

*"I'll walk through a progression from baseline to production, explaining the failure mode of each simpler model to justify moving to the next.*

**Baseline: Matrix Factorization**

*The idea: decompose the user-video interaction matrix R (users × videos) into two low-rank matrices: U (users × d) and V (videos × d), such that R ≈ U · V^T. For user i and video j, predicted relevance = u_i · v_j (dot product).*

*Loss function (WALS — Weighted Alternating Least Squares):*
```
L = Σ_{(i,j) observed} w_ij * (r_ij - u_i^T * v_j)^2
  + λ_u * ||U||² + λ_v * ||V||²
```
*where w_ij is a higher weight for positive interactions and a lower weight (e.g., 0.01) for unobserved pairs. The regularization terms prevent overfitting.*

*Why it fails for production: (1) cold start — new users and new videos have no interaction history, so we can't initialize their embeddings; (2) can't incorporate side features like video title, user demographics; (3) the embedding space only captures user-video co-occurrence, not content semantics.*

**Intermediate: Two-Tower Neural Network (Candidate Generation)**

*Architecture:*
- *User tower: concatenate all user features → 4 MLP layers [512 → 256 → 128 → 64] → L2-normalized 64-dim user embedding*
- *Video tower: concatenate all video features → same MLP architecture → L2-normalized 64-dim video embedding*
- *Similarity: dot product of user embedding and video embedding*

*Training objective (InfoNCE / in-batch negatives):*
```
L = -log[ exp(sim(u, v+) / τ) / (exp(sim(u, v+) / τ) + Σ_k exp(sim(u, v_k^-) / τ)) ]
```
*where v+ is the positive (watched) video, v_k^- are negative videos (in-batch), τ is temperature (hyperparameter, ~0.07). This is the contrastive loss — it pulls user and positive video embeddings together and pushes negatives apart.*

*Why dot product at inference enables ANN search: once we have all video embeddings pre-computed, finding the top-K videos for a user is equivalent to maximum inner product search (MIPS) over the video embedding matrix. FAISS or ScaNN can do this in O(log N) with high recall using HNSW or IVF-PQ indexing. Cosine similarity would also work but requires normalizing vectors; dot product is the natural inner product in the embedding space.*

*Why the two-tower model over matrix factorization: (1) incorporates side features in both towers, solving cold start; (2) video embeddings can be pre-computed offline, so inference cost is just the user tower forward pass + ANN search; (3) scales to billions of videos via pre-computed index.*

**Production: Two-Tower for Retrieval + Deep Ranking Model**

*The ranking model takes the ~1000 candidates from retrieval and scores them with cross-feature interactions. I'd use a deep neural network that concatenates user and video embeddings with cross-feature signals:*
```
Input: [user_embedding || video_embedding || cross_features]
       (64 + 64 + N_cross = 128 + N_cross dimensions)
→ 4 MLP layers [512 → 256 → 128 → 64] → multi-task output heads
```

*Why not use the two-tower model for ranking: two-tower doesn't allow user-video cross-attention (the towers are separate). The cross-feature interactions (e.g., user's historical engagement with this creator, user-video language match) are critical for high-precision ranking and require late fusion.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"Let me give you the full mechanistic picture, including the parts that matter most for production systems.*

**Matrix Factorization — and why WALS over SGD:**

*The standard matrix factorization formulation uses squared error loss on observed pairs. The critical insight in WALS is treating unobserved pairs as 'soft negatives' with low weight rather than ignoring them entirely. If we only optimize on observed (user, video) pairs, the model never learns to push apart user-video pairs that shouldn't be recommended — it can assign high scores to everything. The weight w_ij for unobserved pairs (typically 0.001 to 0.01) acts as a weak negative signal across the entire unobserved matrix.*

*WALS alternates: fix V, solve for optimal U analytically (closed-form linear system), then fix U, solve for V. Each step is a ridge regression. This converges faster than SGD for sparse matrices because it exploits the closed-form solution.*

**Two-Tower — the subtle design decisions:**

*Embedding dimension is a key hyperparameter. Larger dimensions (256-512) give more expressive power but slower ANN search (linear in dimension for IVF-PQ). For candidate generation, I'd use dim=64-128 because we need fast ANN search. For ranking, I'd expand to 256-512 because we're only scoring 1000 candidates.*

*Temperature τ in InfoNCE controls the sharpness of the distribution. Small τ (0.05-0.07) creates a sharper softmax — the model focuses more on the hardest negatives. Large τ (0.2-0.5) creates a softer distribution. I'd tune τ as a hyperparameter; the right value depends on how many in-batch negatives we have.*

*Hard negative mining is essential for training quality. In-batch negatives are randomly sampled — most are easy to distinguish. Training only on easy negatives leads to embedding collapse (all embeddings near each other). I'd add hard negatives: videos that are topically similar to the positive but weren't clicked. These come from the video's topic cluster.*

**ANN search — HNSW vs IVF-PQ:**

*HNSW (Hierarchical Navigable Small World): graph-based index. Nodes are connected to neighbors at multiple scales (like skip lists). Search traverses from coarse to fine levels. Recall: ~98% at top-100. Latency: ~5-10ms for 100M vectors. Memory: ~150 bytes/vector at dim=64 → ~15GB for 100M vectors. Does NOT scale to 10B vectors in RAM.*

*IVF-PQ (Inverted File with Product Quantization): coarse quantization partitions space into clusters (IVF), then product quantization compresses residuals. Recall: ~85-92% at top-100 with nprobe=50. Latency: ~3ms for 100M vectors. Memory: ~8-16 bytes/vector → ~80-160GB for 10B vectors (feasible on 1-2 machines). For 10B videos, IVF-PQ is the only RAM-feasible option.*

*Concrete numbers: IVF with 65536 clusters, PQ with 64 subspaces of 8 bits each, nprobe=64 → ~90% recall at ~5ms latency on 10B vectors.*

**Position bias and its impact on training:**

*A critical non-obvious problem: videos at position 1 in the recommendation list receive ~5-10x more clicks than videos at position 5, independent of content quality. If we naively use 'clicked' as the positive label, our training data is confounded by position — the model learns 'predict what gets shown at the top' rather than 'predict what users want.' This leads to a rich-get-richer dynamic for already-popular content.*

*Mitigation using Inverse Propensity Scoring (IPS):*
```
L_IPS = Σ_i (L_i / P(examined | position_i))
```
*where P(examined | position_i) is the empirical click-through rate at each position for random content (estimated from exploration traffic). This re-weights the loss to undo the position bias.*"

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

*"How do you evaluate this system, both offline and online? What's the gap between offline and online performance?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use accuracy and then run an A/B test."*

Accuracy is wrong for ranking. No understanding of ranking metrics. No offline-online gap analysis.

---

#### ⚠️ Weak Hire Answer

*"I'd use precision@k for offline evaluation and CTR for online. I'd run an A/B test comparing the new model to the baseline."*

Gets the right metric categories but doesn't justify the choices or explain the offline-online gap.

---

#### ✅ Hire Answer (Staff)

*"Let me walk through offline metrics, online metrics, and then the tricky offline-online gap.*

**Offline metrics:**

*For the retrieval stage (two-tower model), the primary metric is Recall@K: what fraction of videos a user actually watched appear in the top-K retrieved candidates. I'd use K=500 (generous retrieval budget). This tells me how many 'good' candidates make it into the ranking stage.*

*For the ranking stage and end-to-end evaluation, I'd use nDCG@10 (Normalized Discounted Cumulative Gain):*
```
DCG@p = Σ_{i=1}^{p} (2^rel_i - 1) / log₂(i+1)
nDCG@p = DCG@p / IDCG@p
```
*where rel_i is the relevance score of the video at rank i (0 or 1 for binary labels, or graded if we have different signal strengths), and IDCG is the DCG of the ideal ranking. nDCG captures both precision and ranking order.*

*Why not just Precision@k: precision doesn't penalize having relevant items ranked lower than irrelevant ones. nDCG gives diminishing credit for relevant items at lower positions.*

*I'd also track Diversity@k: the fraction of distinct topic clusters represented in the top-k recommendations. Pure relevance optimization tends to collapse diversity.*

**Online metrics (A/B test):**
- Primary: Total watch time per session (the north star)
- Secondary: CTR (click-through rate), video completion rate, explicit positive feedback rate (likes/shares per session)
- Guardrail: daily active users (DAU), 7-day user retention. We don't want to increase short-term engagement at the cost of long-term retention.
- Counter-metric: skip rate, explicit dislikes, 'not interested' signals

**Offline-online gap:**

*The gap exists for several reasons specific to recommendation:*
1. *Popularity bias in offline data: our historical data reflects what the previous model showed users. A new model's recommendations on items with no impression history look 'bad' offline even if they'd be great online.*
2. *Distribution shift: offline test set is from a past time period; user tastes and video corpus evolve.*
3. *Interaction effects: offline evaluation treats each recommendation independently. Online, there are session-level effects — watching video A makes video B more or less appealing.*
4. *Novelty effects: new recommendations cause immediate CTR changes not captured offline.*

*To bridge the gap: I'd run a small exploration experiment (5% of traffic) to collect unbiased signal on new model candidates before full A/B launch."*

---

#### 🌟 Strong Hire Answer (Principal)

*"The offline-online gap is actually the most interesting part of recommendation system evaluation, and it's where most teams get burned.*

*The fundamental problem is counterfactual estimation: we want to know 'how would users have engaged with model B's recommendations?' but we only observed engagement under model A's recommendations. This is a causal inference problem.*

*The naive approach — train offline on historical data, evaluate on a held-out time period — is systematically biased. If model A showed political content to users and users engaged with it (because it was the only thing shown), model B that shows diverse content will appear worse offline even if users would prefer it online. This is the exposure bias problem.*

*Better approaches to offline evaluation:*
1. *Counterfactual IPS estimator: weight each offline evaluation sample by 1/P(model_A showed this item), the inverse propensity score. This re-weights the evaluation to approximate what would happen under uniform exploration. Requires knowing the logging policy's propensities.*
2. *Interleaving experiments: instead of A/B (different user cohorts), interleave recommendations from both models in a single session. Users implicitly compare both models' output. Much more sensitive than A/B, requires fewer users to detect improvements.*
3. *Surrogate metrics that correlate with long-term outcomes: watch time has short-term-long-term tension. User retention at 90 days is a better proxy for value delivered than watch time, but it takes 90 days to measure. I'd build a short-term surrogate metric trained to predict 90-day retention from 7-day behavior.*

*The A/B test design also needs care:*
- *Minimum detectable effect: if watch time has standard deviation σ=15 min/day and we want to detect a 1% lift (δ=0.5 min), the sample size needed is n = 2σ²(z_α + z_β)² / δ² ≈ 2*(225)*(1.96+0.84)² / (0.25) ≈ 14,000 users per variant. Very achievable.*
- *Network effects: video recommendation has network externalities (popular videos affect everyone's recommendations). Standard user-level randomization can violate SUTVA (stable unit treatment value assumption). Better to use geo-based or day-of-week-based holdouts.*
- *Multiple testing correction: if we're testing multiple metrics simultaneously (watch time, CTR, diversity), use Bonferroni correction: α_adjusted = α / num_metrics.*"

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

*"Walk me through the serving architecture. How do you serve recommendations at scale with low latency?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd deploy the model on a server and call it when users request recommendations."*

No multi-stage pipeline. No understanding of why you can't run a complex model against 10 billion videos.

---

#### ⚠️ Weak Hire Answer

*"I'd use a two-stage pipeline: first candidate generation, then ranking. The candidate generation uses a lightweight model to narrow down to a few hundred candidates, and then the ranking model scores those."*

Gets the high-level right but no details on how candidate generation works, latency budget, feature store, or training-serving skew.

---

#### ✅ Hire Answer (Staff)

*"The serving architecture is driven by a simple constraint: we have 10 billion videos and a 200ms budget. Running any non-trivial model against all 10B videos per request is impossible — even just doing a dot product against 10B 64-dim vectors takes ~100ms. So we need multi-stage retrieval.*

**Three-stage pipeline:**

*Stage 1 — Candidate Generation (target: <30ms):*
- Input: user features (pulled from feature store)
- Compute user embedding via the lightweight user tower (forward pass, ~2ms)
- Query the pre-built ANN index (FAISS/ScaNN with IVF-PQ) over 10B video embeddings (~10ms)
- Also query several specialized retrievers in parallel:
  - CF-based retrieval (above)
  - Trending videos retriever (top N by recent engagement)
  - Location-based retriever (local events, local creators)
  - Fresh content retriever (new uploads within 24h, prevents new video starvation)
- Merge and deduplicate across retrievers → ~1000 candidates

*Stage 2 — Ranking (target: <80ms):*
- Input: 1000 candidates + user features
- For each candidate: fetch video features from feature store (batch Redis lookup, ~10ms for 1000 items)
- Run ranking model: concatenate user + video + cross-features → MLP → multi-task output (~50ms for 1000 items)
- Output: ranked list of ~100 videos

*Stage 3 — Re-ranking (target: <20ms):*
- Input: top 100 from ranking
- Apply business rules:
  - Filter out already-watched videos (in last 7 days)
  - Apply diversity enforcement (max 3 from same creator in top 20)
  - Boost videos in viewer's declared interests
  - De-duplicate near-identical content
  - Apply safety filters (age restrictions, region restrictions)
- Output: top 20 videos for the homepage

**Feature store design:**
- Offline store (BigQuery/S3): batch features computed daily (user demographics, video metadata)
- Online store (Redis/Memcached): real-time features updated continuously (recent watch history, trending scores)
- The unified feature API abstracts over both stores — the serving code requests features by name, not by store type

**Training-serving skew prevention:**
- The model is trained on features computed by the same feature pipeline as serving. We use a shared feature computation library, not separate training and serving code.
- Features are logged at serving time and stored for training — this ensures training data matches serving distribution
- Schema validation: the feature store enforces typed schemas; if a feature changes shape, the serving pipeline catches it before the model sees corrupted input"*

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to go deeper on two things that are usually glossed over: the latency budget decomposition and the training-serving skew problem.*

**Latency budget decomposition (200ms total):**
```
Network round trip (client → edge → datacenter): 20ms
Feature store lookup (online features for user): 15ms
Stage 1: User tower forward pass: 3ms
Stage 1: ANN search (FAISS IVF-PQ, 10B vectors): 12ms
Stage 1: Multiple retriever fan-out + merge: 5ms
Stage 2: Batch feature fetch for 1000 candidates: 15ms
Stage 2: Ranking model inference (1000 items): 55ms
Stage 3: Re-ranking + business logic: 10ms
Network response + serialization: 15ms
Buffer: 50ms
Total: 200ms
```

*The binding constraint is Stage 2 ranking inference — 55ms for 1000 items. If we're using a 4-layer MLP with 512 hidden units, forward pass for one item is ~0.05ms, so 1000 items is ~50ms sequentially. But we can batch all 1000 items and run them through the MLP in a single GPU forward pass — actual latency with batching is closer to ~5ms. So the 55ms budget also includes feature assembly and result serialization.*

**Training-serving skew — the root cause and solution:**

*Training-serving skew is one of the most insidious production bugs. The symptom: the model performs well offline but worse in production. The cause is any divergence between the feature values seen at training time vs. serving time.*

*Root causes:*
1. *Feature computation logic divergence: training uses Python pandas code; serving uses Java or C++ code with slightly different numerical behavior (e.g., float32 vs float64, different handling of null values)*
2. *Feature freshness mismatch: training uses daily-computed features; serving uses hourly-computed features. If the model learned on stale features, it can be confused by fresh ones*
3. *Log-train skew: serving logs compressed or sampled features to save space; training re-computes features from raw data. Small differences compound*

*The canonical solution is a feature store with a unified computation graph:*
- *The feature transformations (bucketing, normalization, embedding lookups) are defined once as a DAG*
- *The same DAG runs in both the batch training pipeline and the online serving pipeline*
- *Features are logged at serving time and fed back into training — so training exactly replicates what serving computed*
- *Feature distributions are monitored: a daily KL divergence check between training feature distributions and serving feature distributions. If D_KL(train || serve) > threshold, page oncall*

**Continual learning without catastrophic forgetting:**

*The model needs to be updated frequently (ideally daily) to capture trending topics and recent user behavior. Naive retraining from scratch is safe but expensive. Fine-tuning on recent data is fast but risks catastrophic forgetting — the model unlearns good behavior on underrepresented content categories.*

*The production approach: (1) fine-tune on recent data (last 7 days) with a lower learning rate; (2) add a replay buffer — randomly sample 10% of examples from an older training set and mix into fine-tuning batches; (3) use elastic weight consolidation (EWC) to penalize changes to weights that were most important for past predictions. This maintains performance on tail content while adapting to current trends."*

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

*"What are the key failure modes you'd watch for? How do you detect and mitigate them?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"The main issue is accuracy. We'd monitor accuracy and retrain if it drops."*

---

#### ⚠️ Weak Hire Answer

*"Cold start is a problem for new users and new videos. I'd handle it with content-based features. Popularity bias is another issue."*

---

#### ✅ Hire Answer (Staff)

**5 Failure Modes with Detection + Mitigation:**

**1. Filter Bubbles / Rabbit Holes**
- *What happens:* Optimizing engagement → model recommends more of what user already engages with → feedback loop → user only sees narrow content type → loss of long-term retention
- *Detection:* Track topic distribution entropy of recommendations per user over time. If entropy drops below threshold, flag. Survey a sample of users weekly on satisfaction diversity.
- *Mitigation:* Add diversity constraint in re-ranking (force at least 30% of recommendations from outside the user's most-interacted topic cluster). Track long-term retention alongside short-term engagement.

**2. Cold Start — New Videos**
- *What happens:* New videos have no interaction data → no user-video pairs → two-tower model ranks them lower → they never get impressions → self-fulfilling prophecy
- *Detection:* Monitor recommendation rate for videos by age (hours since upload). If videos <24h old get <X% share of impressions, flag.
- *Mitigation:* Fresh content retriever that explicitly boosts new uploads. Use content tower (title, tags, description) embedding as proxy until interaction data accumulates. Policy: guarantee minimum impressions (e.g., 1000) for any new video within 24 hours.

**3. Popularity Bias**
- *What happens:* Popular videos accumulate more engagement data → ranked higher → get more impressions → more engagement → rich-get-richer. Niche content never surfaces.
- *Detection:* Gini coefficient of impression distribution across videos. High Gini = high concentration.
- *Mitigation:* IPS (inverse propensity scoring) in training. Exploration policy (5% random recommendations). Separate 'rising content' retriever that boosts videos with unusually high recent engagement relative to their history.

**4. Position Bias in Training Data**
- *What happens:* Video at position 1 gets 5x more clicks than position 5 regardless of quality. If we use 'clicked' as label, the model learns to recommend what was already at position 1.
- *Detection:* Controlled experiment: occasionally swap position 1 and position 5 and measure if CTR follows the item or the position.
- *Mitigation:* IPS correction as described. Log position at serving time and include it as a feature at training time with a propensity-based sample weight.

**5. Engagement Bait (Clickbait Thumbnails)**
- *What happens:* Videos with misleading thumbnails get high clicks but low watch completion. Optimizing for clicks rewards these; optimizing for watch time penalizes them.
- *Detection:* Track the ratio click_rate / completion_rate per video. Outliers (very high click, very low completion) are engagement bait.
- *Mitigation:* Penalize videos with click/completion ratio > 3 standard deviations from mean. Add 'satisfaction score' head to multi-task model trained on explicit dislikes and 'not interested' signals.*"

---

#### 🌟 Strong Hire Answer (Principal)

*[Extends Hire answer with:]*

**6. Training-Serving Skew (systematic)**
- *Detection:* Daily KL divergence monitoring between training and serving feature distributions. Set threshold at D_KL > 0.01 for any individual feature.

**7. Feedback Loop Amplification**
- *What happens:* Model → recommendations → user behavior → new training data → model reinforces same behavior. Small biases in initial model amplify over time.
- *Detection:* Shadow log a 5% exploration policy (random recommendations). Compare engagement on exploration vs. exploitation traffic over time. If the gap grows, exploitation recommendations are increasingly optimizing for model-induced behavior, not true preferences.
- *Mitigation:* Periodic resets to exploration-based training data. Set minimum exploration budget as a policy constraint.

*"The deeper principle: every failure mode I described is a specific instance of Goodhart's Law — when a measure becomes a target, it ceases to be a good measure. Watch time becomes a target → engagement bait. Clicks become a target → clickbait. Diversity score becomes a target → superficial diversity without actual value diversity. The solution is to have multiple metrics in tension with each other and monitor all of them simultaneously."*

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

*"How does this system fit into the broader ML platform? What would you build vs. buy? What are the org design implications?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd build everything custom for maximum control."*

---

#### ⚠️ Weak Hire Answer

*"I'd use cloud services for infrastructure but build the models ourselves."*

---

#### ✅ Hire Answer (Staff)

*"Build vs. buy decisions by component:*

- *ANN index (HNSW/IVF-PQ): Buy — use FAISS (Meta) or ScaNN (Google). These are heavily optimized, battle-tested libraries. The customization needed (adding custom distance functions or hybrid searches) is available via their APIs. Building custom ANN search from scratch has no competitive advantage.*

- *Feature store: Build (but build on top of open-source). Feast or Tecton provide the framework; we customize for our scale and consistency requirements. The feature store is a strategic asset — it serves ranking, ads, and search, so centralizing it is high leverage.*

- *Two-tower model architecture: Build. This is core IP. But use PyTorch or JAX, not custom ML frameworks.*

- *Training infrastructure: Buy — use managed distributed training (AWS SageMaker, GCP Vertex, or internal compute platform). Not a differentiator.*

**Cross-team sharing opportunities:**
- The user tower (user embedding model) is valuable beyond recommendation. The ads team, the search team, and the creator tools team all need a user representation. By making the user tower a platform service, one team maintains it and all teams benefit.
- The video embedding model is similarly shared: search uses video embeddings for text-to-video retrieval; ads uses it for ad-video relevance; the recommendation system uses it for collaborative filtering.

**Org design implications:**
- One team owns the embedding infrastructure (user tower, video tower, feature store) and is a service provider to product teams
- Individual product teams (homepage rec, watch-next rec, search rec) own the retrieval and ranking models for their surface
- A data quality team owns label construction and signal quality monitoring
- This structure avoids duplication of the expensive embedding infrastructure while preserving product team autonomy"*

---

#### 🌟 Strong Hire Answer (Principal)

*"The most important org design question isn't 'who builds what' — it's 'how do you prevent the ML platform from becoming a bottleneck while preventing each team from reinventing the wheel.'*

*The pattern I've seen work: a centralized ML platform team that owns (1) the feature store, (2) training infrastructure, (3) model serving infrastructure, and (4) embedding serving for the shared user and item towers. Product teams own the retrieval and ranking logic that sits on top of this platform.*

*The failure mode is a platform team that tries to own the models too. That creates a coordination overhead where every product experiment requires a platform team review cycle. Better to give product teams self-service access to the platform APIs.*

*For this specific system, the roadmap prioritization would be:*
1. *Launch with matrix factorization + simple ranking model (3 months). Get signal from real users.*
2. *Add two-tower candidate generation, move to multi-stage pipeline (2 months). Quality uplift is the biggest gain here.*
3. *Add multi-task ranking model with watch completion + like + skip tasks (2 months). Addresses engagement bait.*
4. *Add long-term user health metrics to the optimization objective (ongoing). This is the hardest because it requires multi-week A/B tests and causal inference.*

*The 'should we worry about filter bubbles now or later' question is a product philosophy decision as much as a technical one. My recommendation: instrument for filter bubble detection from day 1 (entropy metrics, diversity scores), but don't add diversity constraints until you have evidence the problem is occurring at scale. Over-engineering for second-order effects before launch slows down the team and may never be needed at your scale."*

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**WALS Matrix Factorization Loss:**
```
L = Σ_{(i,j) observed} w_ij * (r_ij - u_i^T v_j)^2 + λ_u ||U||² + λ_v ||V||²
```
where w_ij = 1 for positive interactions, 0.01 for unobserved pairs.

**InfoNCE Contrastive Loss:**
```
L = -log[ exp(sim(q, k+) / τ) / Σ_{i=1}^{N} exp(sim(q, k_i) / τ) ]
```
where q = query embedding, k+ = positive key embedding, k_i = all keys (1 positive + N-1 negatives), τ = temperature.

**Normalized Discounted Cumulative Gain (nDCG@p):**
```
DCG@p = Σ_{i=1}^{p} (2^rel_i - 1) / log₂(i+1)
IDCG@p = DCG@p computed on ideal ranking
nDCG@p = DCG@p / IDCG@p
```

**Position Bias IPS Correction:**
```
L_IPS = Σ_i (L_i / P(examined | position_i))
```
where P(examined | position_i) is estimated from exploration traffic.

**Engagement Ranking Score (Multi-task):**
```
score(u, v) = w1 * P(watch>50% | u,v) + w2 * P(like | u,v) + w3 * P(share | u,v) - w4 * P(skip | u,v)
```

**A/B Test Sample Size:**
```
n = 2σ²(z_α + z_β)² / δ²
```
e.g., σ=15min, δ=0.5min (1% lift), α=0.05, β=0.2: n ≈ 14,000 users per variant.

**KL Divergence for Feature Drift Detection:**
```
D_KL(P || Q) = Σ_x P(x) log(P(x) / Q(x))
```

---

### Vocabulary Cheat Sheet

| Term | Definition |
|------|-----------|
| Two-tower model | Neural network with separate encoders for users and items, trained with contrastive loss |
| WALS | Weighted Alternating Least Squares — MF training algorithm |
| InfoNCE | Info Noise Contrastive Estimation — contrastive loss using in-batch negatives |
| ANN | Approximate Nearest Neighbor — fast similarity search |
| IVF-PQ | Inverted File with Product Quantization — compressed ANN index |
| HNSW | Hierarchical Navigable Small World — graph-based ANN index |
| Exposure bias | Training data only contains interactions with recommended items, not the full corpus |
| Feedback loop | Model recommendations affect user behavior which feeds back into training |
| Filter bubble | Narrow personalization creating an information echo chamber |
| Training-serving skew | Divergence between feature computation at training vs. inference time |
| Position bias | Click rate confounded by item position in the recommendation list |
| IPS | Inverse Propensity Scoring — technique to correct for selection/position bias |
| Hard negative mining | Selecting challenging negative examples to improve contrastive training |
| Label delay | Positive labels (watch completion) are only observable after the interaction completes |
| Multi-task learning | Single model with shared layers and separate output heads for each task |
| Feature store | Infrastructure for computing, storing, and serving features at low latency |
| Candidate generation | First stage of multi-stage retrieval: narrows corpus from billions to thousands |
| Re-ranking | Final stage: applies business logic and diversity constraints to ranked list |

---

### Key Numbers to Memorize

| Metric | Value |
|--------|-------|
| Corpus size | 10 billion videos |
| End-to-end SLA | <200ms |
| Candidate generation output | ~1000 candidates |
| Ranking model output | ~100 candidates |
| Final recommendation count | 20-50 videos |
| Embedding dimension (retrieval) | 64-128 |
| Embedding dimension (ranking) | 256-512 |
| HNSW recall @top-100 | ~98% |
| HNSW latency (100M vectors) | ~5-10ms |
| IVF-PQ recall @top-100 | ~85-92% |
| IVF-PQ memory (10B at dim=64) | ~80-160GB |
| Watch history window | Last 50 videos |
| Exploration traffic budget | 5% |
| Class imbalance ratio | ~10:1 (neg:pos) |
| Negative sampling ratio | 4-9 negatives per positive |
| Temperature (τ) | 0.05-0.1 |
| Model update frequency | Daily batch + hourly fine-tune |
| Feature freshness (online store) | 5-minute lag |

---

### Rapid-Fire Day-Before Review

**Q: Why use dot product instead of cosine similarity for ANN?**
A: Dot product = MIPS (maximum inner product search), directly compatible with FAISS. Cosine requires normalized vectors; if vectors are L2-normalized, dot product = cosine. Dot product is faster and matches training objective.

**Q: Why InfoNCE over hinge loss for contrastive learning?**
A: InfoNCE (softmax over in-batch negatives) naturally handles variable numbers of negatives, benefits from large batch sizes, and has a clear information-theoretic interpretation (maximizing mutual information). Hinge loss requires careful margin tuning.

**Q: What's the single most impactful feature for recommendation?**
A: User watch history embeddings — the average of recently watched video embeddings. This is a compressed representation of user taste in the same semantic space as candidate videos.

**Q: Why separate candidate generation from ranking?**
A: Running a complex ranking model against 10B videos is infeasible (~100ms just for dot products). Candidate generation (cheap model + ANN) narrows to 1000 candidates; ranking then uses a complex model on that small set.

**Q: How do you prevent the filter bubble?**
A: Multiple retrievers (not just personalized CF), diversity constraint in re-ranking, entropy monitoring, exploration traffic, and treating long-term retention as a guardrail metric alongside short-term engagement.

**Q: What causes training-serving skew?**
A: Different feature computation code in training vs. serving, feature freshness differences, log compression at serving, or distribution shift in production data.
