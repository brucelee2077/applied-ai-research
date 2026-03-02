# Harmful Content Detection System — Staff/Principal Interview Guide

## How to Use This Guide

This guide simulates a complete 45-minute staff/principal ML design interview for a social media harmful content detection system. Hire and Strong Hire answers are in first-person candidate voice.

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

*"Design a system to detect harmful content on a social media platform. Walk me through your approach."*

### What to Clarify — 6 Dimensions

| Dimension | Question | Why It Matters |
|-----------|----------|---------------|
| **Business objective** | Is the primary goal reducing harmful impressions or reducing prevalence? | Determines precision/recall trade-off |
| **Scale** | How many posts per day? 500M? 1B? | Real-time detection vs. batch processing |
| **Latency** | Must content be blocked before publication or can we post-publish? | Pre-publication requires very low latency (<100ms) |
| **Data availability** | Do we have labeled harmful content? How much? | Determines model approach |
| **Interaction types** | What harm types? Violence, nudity, hate speech, all? | Each harm type may need different modeling strategies |
| **Constraints** | Explainability required? User appeals? Multilingual? | Appeals pipeline, GDPR right to explanation |

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd build a text classifier using BERT to detect harmful posts."*

Text-only classifier misses image and video content. No understanding of scale, latency, or the multi-modality of social media posts.

---

#### ⚠️ Weak Hire Answer

*"I'd ask — what types of harm are we detecting? And how many posts per day?"*

Gets harm types and scale. Misses: latency constraints (pre vs. post publication), explainability requirement, multilingual, appeal pipeline.

---

#### ✅ Hire Answer (Staff)

*"This is a high-stakes system — false positives (incorrectly removing benign content) harm creators and free speech, while false negatives (missing harmful content) harm viewers. I want to get the requirements right.*

*First, harm types: are we covering all categories (violence, nudity, hate speech, self-harm, illegal goods) or starting with a subset? Each category has very different false-positive risk profiles — nudity detection is prone to false positives on medical/artistic content; hate speech is highly context-dependent.*

*Second, scale: 500M posts/day at Facebook scale. That's ~5,800 posts/second. This is a real-time streaming problem, not batch.*

*Third, latency: is the requirement to block before publication (pre-publication), within seconds of publication, or batch after the fact? Pre-publication requires <100ms end-to-end. Post-publication within seconds is more achievable.*

*Fourth, explainability: if we remove a post, can we tell the user why? GDPR Article 22 requires this for automated decisions. This affects architecture — I'd want separate output heads per harm category.*

*Fifth, multilingual: does the platform serve global users? Text understanding across 50+ languages requires multilingual models.*

*Sixth, appeals: users who disagree with takedowns can appeal. The appeals pipeline feeds back into training as hard negative mining.*

*I'll proceed with: 500M posts/day, all 5 harm types, post-publication detection (within 5 seconds acceptable), explainability required, multilingual, appeals pipeline included."*

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to dig into the business objective more carefully, because 'detect harmful content' is underspecified and the choice of objective drives the entire system.*

*Two different objectives lead to very different systems:*
1. *Minimize prevalence: the fraction of harmful posts that exist on the platform at any time. This is a supply-side metric — are we removing harmful content quickly?*
2. *Minimize harmful impressions: the number of times users see harmful content. This is a demand-side metric — does harmful content actually reach people?*

*They're not equivalent. A viral harmful post seen by 10 million users is much worse than 1,000 non-viral harmful posts each seen by 5 users, even though the latter has higher 'prevalence.' If we optimize for prevalence, we focus on removing any harmful post quickly. If we optimize for harmful impressions, we prioritize catching high-virality harmful content even if we miss some low-virality posts.*

*I'd argue harmful impressions is the right business objective. It directly measures user harm.*

*On the precision-recall trade-off: different harm types have different asymmetry of errors:*
- *Child safety (CSAM): recall >> precision. A false negative is catastrophic. Even 1% false negative rate is unacceptable. Acceptable false positive rate: much higher.*
- *Nudity on a general platform: more balanced. Aggressive detection causes too many false positives on artistic content.*
- *Hate speech: highly context-dependent. Very hard to automate reliably. More human-in-the-loop needed.*

*The system design should have different confidence thresholds and escalation paths per harm type, not a single binary decision.*"

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

*"How do you frame this as an ML problem? What's the input, output, and ML task?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"Binary classification: is the post harmful or not?"*

Single binary label loses explainability. Can't tell users which specific harm type was detected.

---

#### ⚠️ Weak Hire Answer

*"Multi-label classification: for each post, predict a binary label for each harm type (violence: yes/no, nudity: yes/no, etc.)."*

Gets multi-label right but doesn't explain why multi-task is better than multi-label with separate models, or discuss the fusion strategy for multimodal content.

---

#### ✅ Hire Answer (Staff)

*"Let me work through this carefully because there are several valid framings and the choice matters.*

*Option 1: Single binary classifier (harmful/not). Advantage: simple. Disadvantage: can't explain which harm type. Fails explainability requirement.*

*Option 2: One binary classifier per harm type (5 models). Advantage: each model specializes. Disadvantage: 5x training cost, no knowledge sharing between related tasks.*

*Option 3: Multi-label classification — one model, 5 output heads. Problem: some harm types require very different feature transformations. Violence detection needs temporal features (motion). Nudity detection needs spatial features (body parts). Treating them identically may hurt both.*

*Option 4: Multi-task learning (chosen) — shared base layers + task-specific heads.*
- *Shared base: learns common representations (is this content disturbing at all?)*
- *Task-specific heads: fine-tuned for each harm type's specific patterns*
- *Benefits: cheaper than 5 separate models, knowledge transfer between related tasks (learning 'disturbing imagery' helps both violence and self-harm), explainable (which head triggered?)*

*Input: a social media post is multimodal — text, image(s), video(s), author metadata, user reactions.*

*Fusion strategy — early fusion (chosen over late fusion):*
- *Early fusion: combine all modalities into a single feature vector before modeling*
- *Late fusion (rejected): separate models per modality, combine predictions at end*
- *Why early fusion: some harm is only detectable from the combination of modalities, not individual modalities. Example: a meme where an innocuous image becomes harmful with a specific text overlay. Late fusion would miss this cross-modal interaction.*

*Output: P(harm_type_k | post) for each harm type k, plus a human-readable explanation of which features contributed.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The multi-task framing is right, but I want to add nuance about the label construction problem, which is where content moderation systems typically go wrong.*

*Two label sources:*
1. *User reports (natural labels): fast, scalable, but noisy. Users may report content they disagree with politically, not just genuinely harmful content. False report rates are high for hate speech (weaponized reporting).*
2. *Human review (hand labels): accurate, but human reviewers cause known issues: reviewer burnout from exposure to harmful content, inter-rater disagreement on ambiguous cases (especially hate speech), high cost at scale.*

*The practical approach: use user reports to identify candidates for human review. Human reviewers confirm or reject. Use confirmed cases as training labels, rejected cases as hard negatives.*

*The capacity constraint is real: at 500M posts/day, you can only human-review ~10M posts/day (a 2% sample, assuming 10,000 reviewers reviewing 1,000 posts/day). This means 98% of labeling must come from the model's own predictions (active learning) or proxy signals.*

*The implication for multi-task learning: don't train all harm types with the same label quality. Violence and nudity have higher-quality labels (more visual, less ambiguous). Hate speech has lower-quality labels (highly context-dependent). Reflect this in the loss function — weight tasks by label quality:*
```
L_total = Σ_k w_k * L_k(θ_shared, θ_k)
```
*where w_k is lower for high-noise harm types like hate speech and higher for high-confidence types like CSAM.*"

---

## Section 3: Data & Feature Engineering (8 min)

### Interviewer Prompt

*"Walk me through the features you'd use and how you'd engineer them."*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use the post text and image as features."*

No processing pipeline, no encoding strategy, no author features, no user reaction features.

---

#### ⚠️ Weak Hire Answer

*"For text, I'd use BERT embeddings. For images, I'd use a CNN. I'd concatenate the features and feed them to a classifier."*

Right models but no detail on multilingual handling, how to aggregate user reactions, what author features capture.

---

#### ✅ Hire Answer (Staff)

*"Let me walk through the full feature taxonomy.*

**Text features:**
- Model: DistilmBERT (distilled multilingual BERT, 6 layers, 66M parameters)
- Why DistilmBERT over full mBERT: 40% smaller, 60% faster inference at 97% of mBERT's quality. At 500M posts/day, speed matters.
- Output: 768-dim CLS token embedding
- Preprocessing: truncate to 512 tokens; for longer posts, use sliding window with max-pooling

**Image features:**
- Model: CLIP visual encoder (ViT-B/32) or SimCLR-pretrained ResNet-50
- Why CLIP: its visual-language pre-training makes it robust to adversarial image modifications (text overlaid on images, memes)
- Output: 512-dim image embedding
- Preprocessing: resize to 224×224, normalize. For multi-image posts: average embeddings

**Video features:**
- Extract keyframes (1 per second or scene-change detection)
- Run image encoder on each keyframe
- Temporal aggregation: average pooling or 1D temporal convolution over frame sequence
- For audio: speech-to-text then embed text; or MFCC features for audio classification

**User reaction features:**
- like_count, share_count, report_count (log-transformed to handle power-law distribution)
- comment_count, comment_embeddings (average of DistilmBERT embeddings of comments)
- Why reactions matter: user reports are a real-time signal. A post receiving 500 reports within 1 minute is almost certainly harmful, even before the model runs.

**Author features:**
- violation_history_count: number of previous policy violations (last 90 days)
- profanity_rate: fraction of user's posts containing flagged terms (last 90 days)
- account_age: bucketized (new accounts post more harmful content than old ones)
- follower_count: log-transformed
- demographics: age bucket (one-hot), gender (one-hot), country (embedding, high cardinality)

**Contextual features:**
- time_of_day: bucketized [6-bucket]
- device_type: one-hot [mobile, desktop, API bot]

**Feature concatenation:**
```
final_features = concat([text_emb (768), image_emb (512), author_features (~20), reaction_features (~10), context_features (~15)])
→ total: ~1325-dim input vector to multi-task MLP
```*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to discuss three non-obvious feature engineering decisions.*

**1. Comment embeddings as cross-modal signal:**

*User comments on a post are often the most informative signal for certain harm types. A post showing a person standing at the edge of a building might be ambiguous — but if the comments say 'please don't do this' or 'this is so dangerous,' those comments encode user understanding of the post's harmfulness. This is cross-modal signal: the image is ambiguous, the text reaction makes it clear.*

*The comment aggregation strategy matters: simple average embedding dilutes rare but highly informative comments. Better approach: attention-weighted aggregation where comments containing harm-related vocabulary receive higher weights. Or: train a comment harmfulness classifier and use the max of that classifier's scores across comments.*

**2. Network-level features:**

*For coordinated inauthentic behavior (multiple accounts posting similar harmful content), individual post features miss the pattern. Graph-level features are needed:*
- *Is this post being reshared across a network of new accounts (bot amplification)?*
- *Is the same image being posted by multiple different accounts within 1 hour (coordinated campaign)?*
- *Does the author have many connections to previously-flagged harmful content creators?*

*These network features require a graph database lookup and add latency. For the high-confidence, real-time detection path, skip them. For the slower, high-stakes review path, include them.*

**3. Historical post features from same author:**

*The same author posting 3 times in the last hour with near-identical text is a signal of spamming/inauthentic behavior, independent of the post content. These 'author burst features' are computed in real-time from a sliding window event log.*

*Concretely:*
- *posts_last_1h: count of posts by this author in last 1 hour*
- *image_hash_similarity_last_24h: max cosine similarity between current post image and author's recent post images*
- *keyword_overlap_last_24h: max Jaccard similarity of keywords between current post and author's recent posts*

*These features don't require running any ML model — they're computed from simple database lookups and are available in <5ms.*"

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

*"Deep dive on the model architecture."*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd train a single neural network on all the features."*

No multi-task reasoning. No understanding of shared vs. task-specific components.

---

#### ⚠️ Weak Hire Answer

*"Multi-task learning with shared layers and separate output heads for each harm type. Each head predicts the probability for its harm type."*

Right concept but no detail on architecture specifics, loss function, class imbalance handling, or calibration.

---

#### ✅ Hire Answer (Staff)

*"The architecture is a multi-task deep neural network.*

**Shared base tower:**
```
Input: 1325-dim concatenated feature vector
→ Dense layer: 1325 → 512, ReLU, BatchNorm, Dropout(0.3)
→ Dense layer: 512 → 256, ReLU, BatchNorm, Dropout(0.3)
→ Dense layer: 256 → 128, ReLU
→ Shared representation: 128-dim
```

**Task-specific heads (one per harm type):**
```
For each harm type k ∈ {violence, nudity, hate_speech, self_harm, illegal_goods}:
→ Dense layer: 128 → 64, ReLU
→ Dense layer: 64 → 1, Sigmoid
→ Output: P(harm_type_k | features)
```

**Loss function (multi-task):**
```
L_total = Σ_k w_k * L_k
L_k = Focal Loss(y_k, ŷ_k) = -α(1-ŷ_k)^γ * y_k * log(ŷ_k) - (1-α) * ŷ_k^γ * (1-y_k) * log(1-ŷ_k)
```
- γ = 2 (focusing parameter — down-weights easy examples)
- α = 0.25 (class balance weight — up-weights positive class)

**Why Focal Loss over standard binary cross-entropy:**
- Class imbalance: harmful posts are typically <1% of all posts
- Standard BCE focuses too much on the 99% majority (non-harmful examples)
- Focal Loss down-weights easy negatives, forces model to focus on the hard cases near the decision boundary
- This dramatically improves recall on the rare positive class

**Class imbalance handling (additional):**
1. Oversample positives: upsample harmful examples to 1:10 positive:negative ratio
2. Class-weighted sampling: ensure each batch has a minimum fraction of positives for each harm type
3. Loss weighting: w_k in the total loss can compensate for task-level imbalance

**Calibration:**
After training, the model's raw outputs may not be well-calibrated probabilities. For a content moderation system, we rely on thresholds to decide whether to remove content. Uncalibrated models give poor threshold behavior.*

*Calibration using Platt Scaling:*
```
P_calibrated = sigmoid(a * logit(ŷ) + b)
```
*Parameters a, b learned on a held-out calibration set.*

*Why calibration matters: if P(violence)=0.6 predicts that 80% of posts at that score are actually violent, our thresholds will be systematically off. Calibration makes P(violence)=0.6 mean 60% of posts at that score are actually violent.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to go deeper on two production issues: the multi-task weight tuning problem and adversarial robustness.*

**Multi-task weight tuning:**

*With 5 harm type heads, the loss weights w_k in L_total = Σ w_k * L_k are critical hyperparameters. Setting them naively (all equal) will cause the model to focus on the harm type with the most training examples and underfit on rare categories.*

*Two approaches:*
1. *Manual tuning: set w_k proportional to 1/n_k (inverse class frequency). Intuitive but doesn't account for relative task difficulty.*
2. *Gradient normalization (GradNorm): automatically adjusts weights based on the relative rate of learning of each task. Tasks learning too slowly get higher weights.*

*For CSAM (child safety) in particular: set a very high w_CSAM even if CSAM examples are rare. The cost of missing CSAM is catastrophically higher than the cost of a false positive.*

**Adversarial robustness:**

*Bad actors know our system exists and actively try to evade it. Common evasion techniques:*
1. *Text evasion: misspellings ('v10lence'), unicode substitution, adding irrelevant text to dilute the harmful content*
2. *Image evasion: overlaying noise patterns that fool neural networks while remaining invisible to humans, slight color shifts that change CNN activations*
3. *Semantic evasion: using coded language (dog whistles) that humans recognize as hate speech but NLP models don't*

*Mitigation:*
- *For text evasion: character-level models and subword tokenization are more robust to misspellings than word-level. Train on augmented data with typical evasion patterns.*
- *For image evasion: adversarial training (augment training data with adversarially-perturbed images using FGSM: x_adv = x + ε * sign(∇_x L))*
- *For semantic evasion: use RLHF-style annotation — train annotators to label human-understood meaning, not just surface text. Use knowledge graphs of known dog whistles.*

*The adversarial arms race is ongoing. Monitor for performance degradation on new content patterns, with a dedicated 'adversarial content' team that probes the system monthly.*"

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

*"How do you evaluate the system?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"Evaluation for content moderation requires thinking carefully about which metric to optimize, because precision and recall have asymmetric costs.*

**Offline metrics:**

*For each harm type, track:*
- *Precision: of all content we remove, what fraction was actually harmful? (False positive rate)*
- *Recall: of all actual harmful content, what fraction do we remove? (False negative rate)*
- *F1: harmonic mean — but choose the β to reflect the precision/recall priority:*
```
F_β = (1+β²) * P * R / (β²P + R)
```
*β > 1 means recall is more important (CSAM: use F_2). β < 1 means precision is more important (nuanced hate speech: use F_0.5).*

*PR-AUC (Area Under Precision-Recall Curve): summary metric that captures performance across all thresholds. More informative than ROC-AUC for heavily imbalanced problems, because ROC-AUC can be misleadingly high when negatives vastly outnumber positives.*

**Online (business/product) metrics:**

- *Harmful Impressions: number of views of harmful content per day. The primary business metric. Measures actual harm to users.*
- *Prevalence: fraction of all posts that are harmful and still live. Secondary metric.*
- *Valid Appeals: fraction of removals that are successfully appealed. Measures false positive rate from the creator's perspective. Target: <5%.*
- *Proactive Rate: fraction of harmful content detected by the model before user reports. Measures how much we're doing vs. waiting for users to report.*
- *Per-category metrics: track each harm type separately. Violence going up while nudity is down is actionable information.*

**Why Harmful Impressions > Prevalence:**
A viral harmful video with 10M views is worse than 1,000 non-viral harmful posts each with 100 views (100K impressions total). Prevalence treats these equally. Harmful Impressions captures the actual scale of exposure.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to raise the problem of evaluation dataset bias and the appeal audit.*

**Evaluation dataset bias:**

*Our offline evaluation set is created by human reviewers. Human reviewers have known biases:*
- *Inter-rater disagreement: for hate speech, annotator agreement is often only 60-70%. This means 30-40% of labels in our eval set are 'uncertain.'*
- *Cultural context: a US-trained annotator may label content from other cultures inconsistently.*
- *Recency: if the eval set was created 6 months ago, it doesn't reflect new evasion patterns or new harm categories.*

*The calibration of our metrics depends on the quality of the eval set. A model that achieves 90% F1 on a noisy eval set may be substantially worse in production.*

*Mitigation:*
1. *Stratify evaluation by annotator agreement: report metrics separately on 'high-agreement' (annotators agree >90%) and 'low-agreement' cases*
2. *Refresh evaluation set quarterly with new examples*
3. *Use inter-annotator agreement as a sample weight in metric calculation*

**Appeal audit as ground truth:**

*User appeals are underutilized as training signal. When a user successfully appeals a takedown, that's confirmed evidence of a false positive. When an appeal is denied (upheld removal), that's confirmed evidence of a true positive.*

*Design a structured appeal review process:*
1. *User submits appeal → human reviewer sees the post + model's confidence + which features contributed*
2. *Reviewer approves or denies appeal*
3. *Approved appeals (false positives) → added to training set as high-confidence negatives*
4. *Denied appeals → added to training set as high-confidence positives*

*This creates a continuously improving hard-negative mining pipeline from the appeals process itself.*"

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

*"How does the serving system work at 500M posts/day?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"The serving system has two routing paths based on confidence.*

**High-level flow:**
```
New post published
       ↓
Harmful Content Detection Service
(runs multi-task model inference)
       ↓
      /    \
High confidence   Low confidence
  (P > 0.9)       (0.3 < P < 0.9)
      ↓                ↓
Violation         Demoting Service
Enforcement       - Reduce visibility
Service           - Queue for human review
- Remove post     - Flag for manual decision
- Notify user
- Log reason
```

**Infrastructure:**

*500M posts/day = 5,800 posts/second. The inference pipeline must handle this throughput.*

1. *Stream processing: posts enter a Kafka queue. Multiple consumers pull from the queue and run inference.*
2. *GPU inference workers: multi-task model loaded on GPU. Batch inference: collect 64 posts, run forward pass, emit results in ~50ms for the batch.*
3. *Feature retrieval: for each post, fetch author features from Redis feature store (<5ms), user reaction features from another Redis store (updated every minute)*
4. *Pre-computed embeddings: text, image, video embeddings are computed as part of the post-processing pipeline (parallel to inference). The harm detection model consumes these pre-computed embeddings, not raw content.*
5. *Output routing: confidence score determines action (remove, demote, pass)*

**Latency budget (per post, P99):**
- Feature retrieval: 10ms
- Embedding lookup: 5ms (pre-computed by upload pipeline)
- Model inference (GPU batch): 15ms
- Routing decision + enforcement: 5ms
- Total: 35ms (well within 5-second post-publication window)*

**Human review queue:**
- Posts with 0.3 < P < 0.9 are demoted and queued for human review
- Queue prioritization: by virality signal (posts with high engagement at top), by harm type urgency (CSAM at top)
- SLA for human review: CSAM within 1 hour, others within 24 hours*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to discuss the 'demote vs. remove' architecture more carefully, because it has important product implications.*

**Why a two-tier action system:**

*Binary remove/don't-remove has a fatal flaw: the threshold that minimizes harmful impressions and the threshold that minimizes valid appeals are different. Setting the threshold at P=0.5 maximizes recall but causes ~20% false positive rate. Setting it at P=0.9 minimizes false positives but misses 30% of harmful content.*

*The demote action solves this: at P=0.5-0.9, content is demotion (its virality is reduced — it doesn't appear in recommendations, trending, or amplified distribution) but it's not removed. Users can still access it via direct URL.*

*This is the 'reduce without removing' strategy. It limits harm (fewer harmful impressions) without the creator-facing impact of a false positive removal. The valid appeals rate for demotion is much lower than for removal.*

**The appeals pipeline as a platform feature:**

*The appeals pipeline shouldn't be an afterthought. It's a platform for creators to dispute moderation decisions and for the company to get ground truth labels.*

*Architecture:*
1. *Creator submits appeal → appeal enters a prioritized queue*
2. *Appeal review interface: reviewer sees post, model confidence for each harm type, feature importance (SHAP values showing which features drove the decision)*
3. *Reviewer verdict: uphold (true positive) or overturn (false positive)*
4. *Regardless of verdict: the post is added to a high-quality label pool for model retraining*
5. *Monthly analysis: if valid appeals rate for a harm type exceeds threshold, trigger model review for that category*

*The SHAP-based explainability in the review interface is critical: it tells reviewers not just what the model decided, but why. This makes reviews faster and more consistent.*"

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Model Answers by Level

#### ✅ Hire Answer (Staff)

**5 Failure Modes:**

**1. Context-Dependent Harm (Sarcasm, Irony, Parody)**
- *What:* A satirical post that quotes harmful language to mock it. Model classifies the surface text as harmful without understanding the intent.
- *Detection:* Track appeals that cite 'satire/parody' as the reason. Monitor valid appeals rate for accounts with high follower counts (more likely to be satire/news).
- *Mitigation:* Add 'post_context' features: account category (news, satire, parody — from account verification), historical post tone. Train on satire-specific examples.

**2. Cultural Context Blindness**
- *What:* Gestures, symbols, or phrases that are benign in one culture and harmful in another. A US-trained model miscategorizes culturally-specific content from other regions.
- *Detection:* Track valid appeals rate by country. Higher appeals in specific countries signals cultural context issues.
- *Mitigation:* Regional model fine-tuning on country-specific annotation. Regional human review queues with culturally-knowledgeable reviewers.

**3. Adversarial Inputs**
- *What:* Users add invisible pixel perturbations to images, misspell words, or use Unicode lookalikes to evade detection.
- *Detection:* Periodic adversarial probing by a red team. Monitor distribution shift in text/image feature space.
- *Mitigation:* Adversarial training (FGSM augmentation). Character-level robustness for text. Hash-based near-duplicate detection to catch image perturbations.

**4. Coordinated Posting Campaigns**
- *What:* Large-scale coordinated posting of similar harmful content from many accounts simultaneously overwhelms the moderation queue.
- *Detection:* Velocity alerts: if 1000+ posts with the same image hash appear within 5 minutes, escalate.
- *Mitigation:* Deduplication at ingestion: if a post is near-identical to an already-flagged post, auto-apply the same moderation decision. Graph-level features detecting coordinated behavior.

**5. Label Lag for New Harm Categories**
- *What:* New harm types emerge (e.g., deepfake non-consensual intimate imagery appeared as a new category). Existing model has no training data for it.
- *Detection:* Monitor 'other' harm type reports; if a new pattern appears in appeals, escalate for policy team review.
- *Mitigation:* Zero-shot classification: use CLIP's open-vocabulary capabilities to detect new categories without labeled data. Establish a 'new harm type' policy process with clear thresholds for when to build a specialized model.

---

#### 🌟 Strong Hire Answer (Principal)

*[Extends above with:]*

**6. Reviewer Fatigue and Bias**
- *What:* Human reviewers exposed to large volumes of violent/abusive content develop psychological harm (a documented problem at Facebook/Twitter). Fatigued reviewers also make inconsistent decisions.
- *Detection:* Inter-rater agreement monitoring on shared review sets. Track reviewer accuracy degradation over time.
- *Mitigation:* Mandatory content exposure limits per reviewer per day. Rotation between benign and harmful content queues. Mental health support programs. Prioritize automation to minimize human review volume.

**7. Policy-Model Lag**
- *What:* Community standards policies evolve faster than ML models. A new policy prohibiting a specific type of content takes months to collect labeled data and retrain the model.
- *Detection:* Track cases where human reviewers are removing content that the model classifies as safe.
- *Mitigation:* Rule-based filters for new categories before ML training data is available. Active learning: human reviewers flag novel cases → immediate escalation → these cases become seed training data.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"Build vs. buy:*
- *CLIP/DistilmBERT: use pre-trained from Hugging Face, fine-tune on our data. Don't train from scratch.*
- *Kafka: buy (managed) — streaming infrastructure is not a differentiator*
- *Model serving: use Triton or TorchServe for GPU inference*
- *Label tooling: build internally — the annotation interface must integrate with the appeals pipeline and policy workflow*

*Cross-team sharing:*
- *The multimodal feature extraction pipeline (BERT + CLIP embeddings of posts) is used by: content moderation, spam detection, recommendation system, search. Centralize this as a 'content understanding' platform service.*
- *The author/account feature store is shared with the fraud and spam detection teams.*

*Org design: Safety & Integrity as a separate team from product development. This ensures moderation decisions aren't influenced by engagement metrics.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The deepest platform question is: who has the authority to change the model's behavior, and how fast can they do it?*

*In a content moderation crisis (e.g., a viral harmful event), you need to adjust thresholds or add new rules within minutes, not after a multi-week model retraining cycle. The architecture needs to support this:*

1. *Threshold management system: separate from the model itself. A 'configuration service' that stores per-harm-type thresholds and per-region adjustments. Changes take effect in seconds without model redeployment.*

2. *Rule layer above the model: for new harm categories or known adversarial patterns, maintain a rule engine that can be updated by policy teams without engineering support. 'Block all posts containing image hash X' or 'Automatically review all posts from account Y.'*

3. *Emergency response protocol: a runbook for increasing sensitivity (lower thresholds + more human review) during live crises, with automatic escalation to C-level visibility.*

*The org implication: policy teams need to own the threshold configuration and the rule engine. ML engineers own the model. This separation of concerns enables rapid policy response without engineering bottlenecks.*

*On regulatory compliance: the EU's Digital Services Act (DSA) requires platforms with >45M EU users to conduct annual independent audits of their content moderation systems. This is not a technical choice — it's a legal requirement. Build audit logging and explainability from day 1.*"

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**Multi-task Loss:**
```
L_total = Σ_{k=1}^{K} w_k * L_k(θ_shared, θ_k)
```

**Binary Cross-Entropy:**
```
L = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
```

**Focal Loss (for class imbalance):**
```
FL(pt) = -α * (1-pt)^γ * log(pt)
where pt = ŷ if y=1, else 1-ŷ
α = 0.25 (class balance), γ = 2.0 (focusing)
```

**Platt Scaling (calibration):**
```
P_calibrated = sigmoid(a * logit(ŷ) + b)
```

**F-beta Score:**
```
F_β = (1+β²) * P * R / (β²*P + R)
β=2 for recall-heavy (CSAM), β=0.5 for precision-heavy (hate speech)
```

**Harmful Impressions:**
```
Harmful_Impressions = Σ_{harmful posts} view_count(post)
```

**FGSM Adversarial Perturbation:**
```
x_adv = x + ε * sign(∇_x L(θ, x, y))
```

### Vocabulary Cheat Sheet

| Term | Definition |
|------|-----------|
| Multi-task learning | Shared model with task-specific heads; transfers knowledge between tasks |
| Early fusion | Combining modalities into one feature vector before the model |
| Late fusion | Separate models per modality, combine predictions |
| Focal loss | Loss function that down-weights easy examples (addresses class imbalance) |
| Calibration | Making predicted probabilities match empirical frequencies |
| Platt scaling | Linear calibration of logit outputs |
| Prevalence | Fraction of all posts that are harmful and live |
| Harmful impressions | Number of user views of harmful content |
| Valid appeals | Moderation decisions successfully overturned by user appeal |
| Proactive rate | Harmful content detected before user reports it |
| DSA | Digital Services Act (EU regulation for large platforms) |
| Hard negative mining | Using incorrectly-moderated content (false positives/negatives) as training examples |
| Adversarial training | Training on adversarially-perturbed examples for robustness |

### Key Numbers

| Metric | Value |
|--------|-------|
| Post volume | 500M+ posts/day = 5,800/second |
| Human review capacity | ~10M/day (10,000 reviewers × 1,000 posts/day) |
| DistilmBERT parameters | 66M (vs. 110M for BERT-base) |
| DistilmBERT quality | 97% of BERT on most tasks |
| Class imbalance | ~1% positive (harmful) rate |
| Focal loss α | 0.25 |
| Focal loss γ | 2.0 |
| High confidence threshold | P > 0.9 (remove) |
| Low confidence range | 0.3 < P < 0.9 (demote + review) |
| Human review SLA (CSAM) | 1 hour |
| Human review SLA (other) | 24 hours |
| Valid appeals target | <5% of removals |
| Inference latency target | <35ms P99 |
| DSA audit requirement | Annual for 45M+ EU users |

### Rapid-Fire Day-Before Review

**Q: Why multi-task learning over 5 separate models?**
A: Cheaper to train and serve (one model, not five), transfer learning between related harm types, explainable (which head triggered), shared multimodal feature extraction.

**Q: Why Focal Loss for content moderation?**
A: Class imbalance: <1% of posts are harmful. Standard BCE focuses on 99% negatives. Focal Loss down-weights easy negatives via (1-pt)^γ, forcing model to learn from hard positives.

**Q: What's the difference between Prevalence and Harmful Impressions?**
A: Prevalence = fraction of live posts that are harmful (supply-side). Harmful Impressions = number of times users view harmful content (demand-side). A viral harmful video matters more than many non-viral ones; Harmful Impressions captures this.

**Q: Why early fusion over late fusion?**
A: Cross-modal harm (e.g., meme where image+text combination is harmful but neither is alone) requires the model to see all modalities simultaneously. Late fusion trains separate models and combines predictions — it misses interactions between modalities.

**Q: How does the appeals pipeline improve the model?**
A: Successful appeals = confirmed false positives → hard negatives for training. Failed appeals = confirmed true positives → reinforce positive class training. The appeals process is a continuously-improving label collection system.
