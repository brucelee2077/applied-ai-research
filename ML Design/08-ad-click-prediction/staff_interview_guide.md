# Ad Click Prediction System — Staff/Principal Interview Guide

## How to Use This Guide

This guide covers a complete 45-minute staff/principal ML design interview for social media ad click prediction (CTR prediction). Hire and Strong Hire answers are written in first-person candidate voice.

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

*"Design an ad click prediction system for a social media platform. The system should predict whether a user will click on an ad shown on their timeline. Walk me through your approach."*

### What to Clarify — 6 Dimensions

| Dimension | Question | Why It Matters |
|-----------|----------|---------------|
| **Business objective** | Revenue maximization? CTR optimization? Advertiser ROI? | These can be in tension — a high-CTR ad may not convert well |
| **Scale** | How many ads? How many users? How many impressions/day? | Billions of ads × hundreds of millions of users = extreme sparsity |
| **Latency** | <50ms? <200ms? | Determines whether we can use complex models in the critical path |
| **Data availability** | Click logs, conversion data, hide/feedback signals? | Determines label construction; conversion has severe label delay |
| **Interaction types** | Multiple ad placements (timeline, stories, search)? | Different surfaces have different click patterns |
| **Constraints** | Continual learning requirement? Privacy (ATT, GDPR)? | Apple ATT dramatically changed signal availability; GDPR affects user-level features |

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use a neural network to predict click probability based on user and ad features."*

No clarification. No architectural thinking. Doesn't address the most critical constraint (continual learning) or the feature sparsity challenge.

---

#### ⚠️ Weak Hire Answer

*"I'd ask — how many ads/impressions per day, and what's the latency requirement?"*

Gets scale and latency but misses: continual learning (critical for ad systems), privacy constraints (ATT, GDPR), conversion delay problem, auction mechanics.

---

#### ✅ Hire Answer (Staff)

*"Ad click prediction is one of the most technically demanding ML problems because of the combination of extreme scale, extreme sparsity, and extreme freshness requirements. Let me clarify before designing.*

*First, business objective: are we predicting click probability to optimize CTR, or are we predicting expected revenue (CTR × bid × conversion probability) for auction ranking? Pure CTR optimization ignores advertiser ROI and can favor cheap ads with misleading copy.*

*Second, scale: how many ad impressions per day? I'll assume 100B+ impressions/day. At this scale, even sub-millisecond decisions matter enormously.*

*Third, latency: the prediction must happen within the ad auction, which typically has a 100ms total budget. I'd want 20-30ms for the ML component.*

*Fourth, continual learning: I've seen that ad CTR prediction degrades severely without fresh model updates. Is continuous learning required, or is daily retraining sufficient? The answer changes the entire serving architecture.*

*Fifth, data: do we have conversion data (user clicked ad → purchased)? Conversion events have severe label delay — users might convert hours or days after seeing an ad. This complicates label construction.*

*Sixth, privacy: has Apple's App Tracking Transparency (ATT) reduced cross-app signal? Does GDPR restrict user-level behavioral targeting in the EU?*

*I'll proceed assuming: 100B impressions/day, <30ms for prediction, continual learning required (daily fine-tune minimum), click is the primary label, conversion is secondary, ATT impact is significant.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to start with the auction mechanics because they fundamentally shape the ML objective.*

*Most social media ad systems use a second-price auction (Vickrey auction variant). The ad that ranks first wins the impression, but pays the price of the second-highest bid. The ranking score is: eCPM = CTR_predicted × bid × quality_score.*

*The implication for ML: we're not just predicting P(click). Our prediction feeds directly into the auction clearing price. If our model over-estimates P(click) for an advertiser, that advertiser 'wins' auctions they shouldn't win and pays more than they would in a well-calibrated system. They'll find that their actual ROI is below expectation and reduce spend.*

*This is why model calibration is a first-class concern for ad systems, not an afterthought. A well-calibrated model that predicts P(click)=0.03 should be right 3% of the time. Advertisers trust the system based on this.*

*On the feedback loop: the ads shown to users are determined by the model. The clicks we observe are on the ads the model ranked highly. This creates a bias toward high-CTR ads (they get shown more) and against low-CTR ads (they get shown less, so we have less signal on them). This is the exploration-exploitation problem in ad serving.*

*On privacy: Apple ATT reduced cross-app tracking signal. For iOS users, behavioral data from outside Facebook/Instagram is largely gone. Differential privacy requirements from GDPR further limit personalization depth. The model must be robust to partial feature absence.*"

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

*"How do you frame this as an ML problem?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"Binary classification: predict if user clicks the ad."*

Correct framing but zero depth. Misses the calibration requirement, label delay, or why multi-output might be valuable.

---

#### ⚠️ Weak Hire Answer

*"Binary classification: for each (user, ad) pair, predict P(click). Use cross-entropy loss. Positive = click, negative = impression without click."*

Right but incomplete. Doesn't discuss calibration, label construction subtleties, or how this feeds into auction ranking.

---

#### ✅ Hire Answer (Staff)

*"The ML task is: given features of a user U and an ad A shown to user U at time t, predict P(click | U, A, context).*

*Output: a calibrated probability (not just a score), because this probability feeds directly into the auction clearing price.*

*Input specification:*
- User features: demographics, contextual (device, time), behavioral history (past clicks, interests)
- Ad features: advertiser ID, campaign ID, category, creative content (image/video embedding), historical CTR
- Context features: current page, user's recent activity, time of day

*Label construction:*
- *Positive (y=1): user clicked the ad within a time window t_window after impression*
- *Negative (y=0): ad was shown (visible on screen for > 1 second) but not clicked within t_window*
- *Special case: user explicitly hid the ad → strongly negative label (can weight higher)*

*Why 'visible for >1 second' matters: if we count all impressions as negatives, many are scroll-past impressions where the user never saw the ad. These are not true negatives — the user wasn't given a chance to click. Viewability threshold prevents this noisy negative problem.*

*Label delay: conversion events (purchase after click) are delayed by hours to days. For click prediction, the delay is short (<5 seconds typically). For conversion prediction (needed for full-funnel optimization), we need delayed label handling.*

*This is fundamentally a binary classification problem, but with two important nuances: (1) the model must be well-calibrated (predicted probabilities should match empirical click rates), and (2) the system must support continual learning (the model is retrained frequently with fresh data).*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I'd frame this as a multi-task prediction problem with calibration as a hard constraint.*

*Tasks:*
1. *P(click | U, A): primary task for auction ranking*
2. *P(conversion | click, U, A): secondary task for advertiser value optimization*
3. *P(hide | U, A): negative signal — users explicitly hiding ads*

*Why predict all three: the auction ranking score should be: eCPM = P(click) × bid × P(conversion|click). Including conversion gives advertisers better ROI and prevents optimizing for click farms.*

*The multi-task formulation shares a base representation for all three tasks, enabling knowledge transfer. Clicks are far more frequent than conversions, so the conversion prediction model benefits from the rich signal learned in click prediction.*

*Calibration requirement: I'd treat calibration as a hard constraint, not a nice-to-have. After training, run isotonic regression or Platt scaling to calibrate the click score:*
```
P_calibrated = sigmoid(a * logit(ŷ) + b)  [Platt scaling]
```
*Validate calibration using a reliability diagram: bin predictions by score, plot mean predicted vs. mean actual click rate per bin. A perfectly calibrated model plots on the diagonal.*

*The label delay problem: conversions can happen days after an ad impression. During that delay, we don't know if the impression led to a conversion. Two approaches:*
1. *Wait-and-train: delay model training until the conversion window closes (up to 7 days). More accurate labels, less fresh model.*
2. *Survival analysis framing: treat 'time to conversion' as a right-censored event. Model P(conversion by time T | features). Train incrementally as conversions arrive.*"

---

## Section 3: Data & Feature Engineering (8 min)

### Interviewer Prompt

*"Walk me through the data and feature engineering."*

### Model Answers by Level

#### ❌ No Hire Answer

*"User age, gender, and ad category as features."*

Completely misses the most important features (behavioral history, ad historical performance) and encoding strategies.

---

#### ⚠️ Weak Hire Answer

*"User demographics, ad category, and historical CTR for the ad. Encode categorical features with embeddings."*

Right direction but misses: cross-features, encoding strategies for high-cardinality IDs, behavioral features, interaction features.

---

#### ✅ Hire Answer (Staff)

*"Let me organize features by category, since encoding strategy depends on feature type.*

**Ad features:**
- `ad_id` (cardinality: billions): embedding lookup, dim=32. The ID embedding captures ad-specific quality signals beyond what other features encode.
- `advertiser_id` (cardinality: millions): embedding lookup, dim=16
- `campaign_id`: embedding lookup, dim=16
- `category` (cardinality: ~1000): embedding lookup, dim=32
- `subcategory`: embedding, dim=16
- `ad_image_embedding`: SimCLR or CLIP visual encoder → 256-dim (pre-computed, stored in feature store)
- `historical_ctr`: ad's average click-through rate (last 7 days), float — crucial signal for new users where personalization fails
- `impression_count`: log-transformed, captures ad freshness/fatigue

**User features:**
- `user_id` (cardinality: billions): embedding, dim=32
- Demographics: age bucket (embedding, dim=8), gender (embedding, dim=4), country (embedding, dim=16)
- Contextual: device_type (one-hot), time_of_day (embedding, dim=8), day_of_week (embedding, dim=4)
- `recently_clicked_ad_ids`: list of last 10 ad IDs user clicked → embeddings → average, dim=32. Crucial for personalization.
- `user_click_rate`: user's historical click rate across all ads (last 30 days) — captures user's general propensity to click ads
- `user_category_affinities`: user's engagement rate per ad category — high-cardinality sparse feature, use feature hashing

**Cross-features (explicit feature interactions):**
- `user_category × ad_category`: whether user's historical category preference matches ad category (0/1)
- `user_country × ad_target_country`: geo targeting match

**Interaction features:**
- `dwell_time`: how long ad was visible before action
- `scroll_velocity`: how fast user was scrolling (fast scroll = less attention)

**Key encoding decisions:**
- High-cardinality IDs (user_id, ad_id): embedding lookup tables, learned end-to-end
- One-hot for low-cardinality categoricals (<50 values)
- Embeddings for medium-cardinality categoricals (50-10000 values)
- Log transform for power-law distributed counts (impression_count, follower_count)

**Training data construction:**
Each training example = one ad impression. Features: user + ad + context at impression time. Label: clicked (1) or not clicked (0) within 5 minutes.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to highlight three non-obvious feature engineering decisions.*

**1. Feature freshness and the feature store design:**

*User behavioral features (recent clicks, affinities) change rapidly. An ad system is only as good as the freshness of its features. The feature store needs two tiers:*
- *Online (Redis): user behavioral features updated in near-real-time (< 5 minute lag). Accessed at prediction time.*
- *Offline (Hive/BigQuery): historical aggregate features computed daily.*

*Concretely: 'user clicked on sports ads in the last 24 hours' is a hot feature (needs real-time update). 'User's 30-day average click rate' is a cold feature (daily batch is fine).*

**2. Label construction with delayed signals:**

*Clicks are observed quickly. But 'quality clicks' (user stayed on the advertiser's page for > 30 seconds) are delayed and more valuable. And conversions are delayed by days.*

*For the primary click prediction task: use 5-minute window. For a secondary 'quality click' signal: use 30-second post-click dwell time as a delayed label, weight 2x.*

*For the conversion prediction task: use a delayed label pipeline. After impression, set up a 7-day window. As conversions come in, add them to a queue. The training pipeline reads from this queue with appropriate time delay.*

**3. Cross-features and why they matter for ads:**

*The most predictive signals for ad CTR are interactions between user features and ad features, not individual features in isolation. A user who likes cooking is much more likely to click on a cooking-related ad — but this requires the model to learn the (user_likes_cooking × ad_is_cooking) interaction.*

*Simple neural networks struggle with this because sparse input features (ID embeddings) need to interact in ways that require many training examples to learn.*

*Factorization Machines (FM) solve this by modeling all pairwise feature interactions via embedding dot products:*
```
y_FM = w_0 + Σ_i w_i * x_i + Σ_i Σ_{j>i} <v_i, v_j> * x_i * x_j
```
*where v_i is a learned k-dim embedding for feature i. This captures the cooking-user × cooking-ad interaction implicitly, even for (user, ad category) pairs with few training examples.*"

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

*"Walk me through your model choices from baseline to production."*

### Model Answers by Level

#### ❌ No Hire Answer

*"Logistic regression as a baseline, then gradient boosting, then a neural network."*

Lists models without explaining why each fails. Doesn't know DeepFM or why it's specifically appropriate for ad click prediction.

---

#### ⚠️ Weak Hire Answer

*"I'd use DeepFM: a combination of factorization machines and deep neural network. The FM component captures pairwise feature interactions, the DNN captures higher-order interactions."*

Names the right model but no mathematical detail, no explanation of why simpler models fail, no discussion of calibration or continual learning.

---

#### ✅ Hire Answer (Staff)

*"Let me walk through the model progression systematically, explaining why each simpler model fails.*

**Model 1: Logistic Regression (baseline)**
```
y = sigmoid(w^T x + b)
```
*Why it fails: (1) can't model non-linear relationships; (2) feature interactions must be hand-crafted; (3) can't use embedding layers for high-cardinality categoricals.*

**Model 2: Feature Crossing + Logistic Regression**
- Manually create interaction features: `country=US AND language=English → new feature`
- *Why it fails: combinatorial explosion of features; requires domain knowledge to enumerate useful interactions; misses subtle interactions.*

**Model 3: Gradient Boosted Decision Trees (GBDT)**
- Auto-discovers feature interactions via tree splits
- *Why it fails for production: (1) can't use embedding layers — categorical features must be one-hot encoded → sparse, high-dimensional; (2) continual learning is expensive — GBDT must retrain from scratch (no incremental update); (3) 5-minute update requirement is infeasible.*

**Model 4: Neural Network**
- Dense layers over concatenated feature embeddings
- Can use embedding layers for IDs
- Supports incremental learning
- *Why it fails vs. DeepFM: struggles to explicitly model pairwise feature interactions — the most important signal for CTR.*

**Model 5: Factorization Machines (FM)**
```
y_FM = w_0 + Σ_i w_i * x_i + Σ_i Σ_{j>i} <v_i, v_j> * x_i * x_j
```
*Every feature has a k-dim embedding v_i. Pairwise interaction between features i and j = dot product of their embeddings.*

*Why this is powerful: for sparse features like user ID and ad ID, there may be very few direct (user_A, ad_B) training examples. But through shared embeddings, FM learns that 'user_A is similar to users who clicked ad category X' and 'ad_B is in category X.' This is collaborative filtering via feature interactions.*

*Why FM alone fails: only captures second-order (pairwise) interactions. Misses higher-order patterns like (young user × mobile device × evening × video ad) four-way interaction.*

**Model 6: DeepFM (SELECTED)**

*Architecture:*
```
Input features (user + ad + context)
          ↓
    Embedding Layer
    (all features → dense vectors)
          ↓
   ┌──────┴──────┐
   │ FM Component │  │ Deep Component │
   │              │  │                │
   │ First-order: │  │ MLP layers:    │
   │ Σ w_i * x_i  │  │ concat → 512 → │
   │              │  │ 256 → 128 → 64 │
   │ Second-order: │  │                │
   │ Σ<v_i,v_j>   │  │ ReLU + BN +    │
   │ * x_i * x_j  │  │ Dropout(0.3)   │
   └──────┬──────┘  └──────┬─────────┘
          ↓                 ↓
          └────────┬────────┘
                   ↓
           y = sigmoid(y_FM + y_DNN)
           → P(click)
```

*Loss function:*
```
L = -(1/N) Σ [y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
```

*Why DeepFM wins:*
1. *FM component: efficiently captures pairwise interactions (second-order)*
2. *DNN component: captures complex, higher-order interactions*
3. *Shared embedding layer: both components share the same input embeddings — FM and DNN reinforce each other*
4. *Embedding layers: handle high-cardinality categorical features naturally*
5. *Incremental update: can fine-tune with gradient descent on new batches → supports 5-minute refresh requirement*

*Evaluation metric: Normalized Cross-Entropy (NCE):*
```
NCE = CE_model / CE_baseline
where CE_baseline = -[p * log(p) + (1-p) * log(1-p)], p = empirical CTR
```
*NCE < 1 means the model beats predicting average CTR for everyone. Typical production NCE: 0.80-0.90.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to extend the DeepFM discussion with three production considerations: calibration, position bias, and the exploration-exploitation problem.*

**Calibration (critical for ad auctions):**

*Uncalibrated models break the auction. If our model predicts P(click)=0.05 for an ad but the true CTR is 0.02, the advertiser wins more auctions than they should and pays too much. They notice this when their campaign ROI is below expectation and reduce spend.*

*Two calibration methods:*

*1. Platt Scaling (post-hoc linear calibration):*
```
P_calibrated = sigmoid(a * logit(ŷ) + b)
```
*Fit parameters a, b on a held-out calibration set. Fast, but only a linear correction.*

*2. Isotonic Regression (non-parametric):*
*Fit a piecewise-constant monotone function mapping raw scores to calibrated probabilities. More flexible than Platt scaling. Use for production; validate with reliability diagrams.*

*Calibration must be re-run after each model update. A freshly fine-tuned model may have slightly different calibration characteristics.*

**Position bias correction:**

*Ads at the top of the feed get 3-5x more clicks than ads further down, regardless of ad quality. If we train on raw click data, the model learns 'top positions get clicks' rather than 'good ads get clicks.'*

*Correction: use Inverse Propensity Scoring (IPS):*
```
L_IPS = -(1/N) Σ_i [ y_i / e(pos_i) * log(ŷ_i) + (1 - y_i/e(pos_i)) * log(1-ŷ_i) ]
```
*where e(pos_i) = P(click | position_i, random ad) = the empirical click rate at that position for randomly-shown ads.*

*In practice: collect exploration traffic (5% of impressions show randomly-ordered ads). Estimate e(pos) empirically. Apply IPS weighting to all training examples.*

**Exploration-exploitation tradeoff:**

*The exploitation policy (always show highest-CTR ads) causes a feedback loop: high-CTR ads → more impressions → more click data → refined predictions → still highest CTR → even more impressions. Low-CTR ads never get impression data.*

*Problem: we may be systematically missing ads that would perform well with the right audience segment but have never been shown to that segment.*

*Solution: ε-greedy exploration: 5% of impressions → randomly selected ad (not the highest-predicted CTR). Log these clicks as exploration data. Use exploration data for unbiased evaluation and causal training.*"

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

*"How do you evaluate the system offline and online?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"**Offline metrics:**

*1. Cross-Entropy (CE): primary metric for binary classification*
```
CE = -(1/N) Σ [y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
```
*Lower is better.*

*2. Normalized Cross-Entropy (NCE): measures improvement over baseline*
```
NCE = CE_model / CE_baseline
CE_baseline = -[p * log(p) + (1-p) * log(1-p)], p = empirical CTR
NCE < 1: better than baseline
```
*NCE < 1 means the model predicts click rates better than predicting the average CTR for everyone. Typical range: 0.80-0.90 for good production models.*

*3. AUC-ROC: measures discrimination (how well the model ranks click vs. no-click pairs)*

*4. Calibration metrics: Mean Absolute Error between predicted CTR and actual CTR across deciles of predictions (the reliability diagram). A well-calibrated model has MAE < 0.001.*

**Why not Accuracy:** with 1-2% click rate, a model predicting 'no click always' achieves 98-99% accuracy. Useless.*

**Online metrics (A/B test):**
- *CTR lift: % increase in click-through rate. Primary metric for click prediction.*
- *Revenue lift: % increase in ad revenue (CPM). The ultimate business metric.*
- *Conversion rate: post-click conversion. Indicates ad quality beyond just clicks.*
- *Hide rate: % of users hiding the ad. Guards against aggressive targeting (precision metric).*
- *RPM (Revenue Per Mille): revenue per 1000 impressions. Composite metric.*

**A/B test design:**
- *Randomize at user level (not impression level) to avoid within-user contamination*
- *Run for 2+ weeks to account for novelty effects and day-of-week variation*
- *Minimum detectable effect: if baseline CTR=2%, target 0.1% absolute improvement. With σ≈0.14, n = 2*(0.02)*(1.96+0.84)² / (0.001)² ≈ 313K users per variant — easily achievable*"

---

#### 🌟 Strong Hire Answer (Principal)

*"Beyond the standard metrics, I want to discuss three evaluation dimensions that matter for an ad system at scale.*

**1. Per-segment calibration:**

*Global calibration hides segment-specific failures. A model might be well-calibrated on average but systematically over-predict for mobile users and under-predict for desktop users. This matters because advertisers often run device-targeted campaigns.*

*Track calibration broken down by: device type, user age group, ad category, geo region. Flag segments where |predicted_CTR - actual_CTR| > 0.005 (5 basis points).*

**2. Advertiser fairness:**

*A CTR prediction model that systematically under-predicts for smaller advertisers creates a marketplace fairness problem: small advertisers' ads are shown less because the model predicts low CTR, so they never get the chance to improve. This is the same exposure bias problem but from the supply side.*

*Measure: plot predicted CTR vs. actual CTR for advertisers stratified by spend (large vs. small advertisers). Check for systematic bias against smaller advertisers.*

**3. Long-term revenue vs. short-term CTR:**

*A model optimized purely for CTR can increase short-term revenue while decreasing advertiser satisfaction (users learn to ignore ads that look like clickbait → ad fatigue → CTR drops over longer horizon).*

*Metric: advertiser renewal rate — did advertisers renew their campaigns after running with this system? This is a 30-90 day lagged metric that captures long-term value.*

*The tension: short-term CTR metrics drive model tuning, but advertiser renewal rate drives long-term business value. The right approach is to track both and accept short-term CTR reductions if they improve long-term renewal rates.*"

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

*"Walk me through the serving architecture, including the continual learning pipeline."*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"The serving system has three interconnected pipelines.*

**Pipeline 1: Data Preparation**
- *Data stream: every impression + click event → Kafka*
- *Batch feature computation (daily): user demographics, 30-day behavioral aggregates → stored in offline feature store (Hive)*
- *Online feature computation (real-time): recent clicks (last 1 hour), current location, device → stored in online feature store (Redis)*
- *Feature store API: abstracts batch + online stores; serving reads features by name*

**Pipeline 2: Continual Learning**
- *New impressions + clicks stream in continuously*
- *Every N minutes: pull fresh training batch from stream*
- *Fine-tune DeepFM on fresh batch (gradient descent, small learning rate)*
- *Evaluate on held-out validation set (last 1 hour of data)*
- *Deploy if validation loss improves; rollback to previous model if it degrades*
- *Critical: even 5-minute delay in model updates causes measurable CTR degradation*

**Pipeline 3: Prediction**

*Stage 1 — Candidate Generation (<5ms):*
- Advertiser targeting constraints filter relevant ads from billions to ~hundreds
- Rules-based: geography, demographics, content safety, budget caps
- Output: ~200-500 relevant ad candidates

*Stage 2 — Ranking (<20ms):*
- For each candidate: fetch user features (Redis, batch) + ad features (feature store)
- Run DeepFM forward pass on GPU (batched, all candidates at once)
- Output: P(click) per candidate

*Stage 3 — Re-ranking + Auction (<5ms):*
- Compute eCPM = P(click) × bid for each candidate
- Apply diversity constraints (max 2 ads from same advertiser per page)
- Apply brand safety rules (ad category vs. page content match)
- Run second-price auction: highest eCPM wins, pays second-highest eCPM
- Return winning ad

*Total latency: <30ms P50, <50ms P99*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The continual learning pipeline has several subtle failure modes that are worth discussing.*

**Why models degrade without frequent updates:**

*Ad click patterns are highly non-stationary. News events, sports results, cultural moments cause sudden shifts in user interest. A model trained 24 hours ago has no signal about today's trending content. An advertiser running a Super Bowl ad needs the model to recognize the Super Bowl context immediately.*

*The degradation curve is steep: a 5-minute-stale model loses ~0.1% absolute CTR vs. real-time; a 1-hour-stale model loses ~0.5-1%. Over billions of impressions, this is millions of dollars per hour.*

**Catastrophic forgetting in continual learning:**

*If we fine-tune on only the most recent data, the model forgets stable patterns from older data. Example: seasonal patterns (holiday shopping vs. summer travel) that appear reliably but infrequently. Pure online learning on recent data loses these.*

*Solutions:*
1. *Replay buffer: when fine-tuning on new data, mix in a random sample of older training examples (10-20% of each batch). This prevents forgetting without requiring full retraining.*
2. *Elastic weight consolidation (EWC): penalize changes to weights that were important for predicting past examples.*
3. *Warm start from previous model: initialize fine-tuning from the previous version's weights, not random. Small learning rate (1e-4 vs. 1e-3 for full training) to avoid catastrophic updates.*

**Feature store consistency:**

*A subtle but critical problem: the features used at serving time must exactly match the features used in training. If the serving pipeline computes 'user_click_rate' as clicks / (impressions + 1e-6) and the training pipeline computes it as clicks / max(impressions, 1), predictions will drift.*

*Solution: a shared feature computation library. The exact same Python/Spark code that computes batch features for training also runs in the online serving path (via a serving runtime that executes the computation graph). Feature schemas are versioned.*

**Model versioning and rollback:**

*The continual learning pipeline produces a new model every N minutes. If a bad batch of training data (e.g., a click fraud attack) corrupts the model, you need to be able to roll back to the last good version in seconds.*

*Architecture: every model version gets a timestamp and hash. The serving infrastructure reads the 'active model pointer' from a config service. To roll back: update the config service → all serving replicas switch to the previous model within 10 seconds.*"

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Model Answers by Level

#### ✅ Hire Answer (Staff)

**5 Failure Modes:**

**1. Click Fraud**
- *What:* Bots or click farms click on ads to drain competitor budgets (invalid traffic)
- *Detection:* Click velocity anomalies (100 clicks in 1 second from same IP), unusual user agent strings, click patterns inconsistent with organic behavior
- *Mitigation:* Invalid click scoring model (separate from CTR model). Filter fraudulent clicks from training data. Retroactive refund to advertisers for detected fraud.

**2. Label Delay for Conversions**
- *What:* User clicks ad, converts 3 days later. During those 3 days, the impression looks like 'click but no conversion' in training data.
- *Detection:* Compare conversion rate at t=1h vs t=7 days. Significant difference means label delay is material.
- *Mitigation:* Survival analysis model: P(conversion by time T | click, features). This handles censored observations (conversions not yet observed).

**3. Model Degradation During Events**
- *What:* Major news event (e.g., COVID lockdowns) causes sudden massive shift in user behavior. Model trained on pre-event data predicts incorrectly.
- *Detection:* Monitor NCE hourly. If NCE degrades by > 0.02 in 1 hour, trigger alert.
- *Mitigation:* Increase continual learning frequency during detected distribution shift. Weight recent data more heavily (exponential decay of older training examples).

**4. Cold Start for New Advertisers/Ads**
- *What:* New advertiser or new ad creative has no click history → no historical CTR feature → model defaults to average
- *Detection:* Track CTR prediction accuracy for ads with < 1000 impressions. Compare to ads with > 10000 impressions.
- *Mitigation:* Use ad content embedding (SimCLR) as proxy for historical CTR in early stage. Use advertiser-level CTR as prior for new ads. Thompson Sampling for exploration: show new ads more to estimate their true CTR faster.

**5. Privacy Signal Loss (ATT)**
- *What:* Apple ATT removes cross-app behavioral data for iOS users. Personalization degrades for ~50% of mobile users.
- *Detection:* Track NCE separately for iOS vs. Android users. Post-ATT, iOS NCE should increase (model performs worse = higher NCE).
- *Mitigation:* Contextual targeting: use page content, time, and device type (without user history). Cohort-based targeting (Google's Topics API model). On-device ML for personalization without data leaving the device.

---

#### 🌟 Strong Hire Answer (Principal)

*[Extends above with:]*

**6. Auction Gaming by Advertisers**
- *What:* Sophisticated advertisers learn to bid just above the threshold to win auctions at minimum cost, or coordinate to suppress competitors' ads.
- *Detection:* Analyze bid distributions by advertiser. Unusually tight bid clusters may indicate gaming.
- *Mitigation:* Auction design matters: second-price auctions are strategy-proof in theory but break down with multiple ad slots. Consider Vickrey-Clarke-Groves (VCG) mechanism for multi-slot pages.

**7. Feature Distribution Shift**
- *What:* As platform demographics change (e.g., younger users join), the training data distribution shifts. Model trained on historical data doesn't represent current users.
- *Detection:* KL divergence monitoring on input feature distributions. D_KL(train_distribution || serving_distribution) > 0.01 triggers alert.
- *Mitigation:* Data freshness weighting: more recent examples get higher weights in training. Exponential decay with half-life of 30 days.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"Build vs. buy:*
- *DeepFM training: build on top of PyTorch. The model is core IP.*
- *Feature store: build on top of Redis + Hive. Central infrastructure shared across all ML teams.*
- *Kafka streaming: buy (managed Confluent). Infrastructure, not differentiator.*
- *Model serving: Triton Inference Server (NVIDIA OSS). Excellent GPU batching, standard.*

*Cross-team sharing:*
- *User feature store shared with: recommendation, search, content moderation*
- *Ad embedding model shared with: brand safety, ad quality scoring, creative optimization*
- *The auction logic is shared across: timeline ads, video ads, stories ads — one auction service*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The platform opportunity in ad click prediction is the 'ads brain' concept: a unified ML platform for all ad-related predictions.*

*Currently, different teams might own:*
- *CTR prediction model*
- *Conversion prediction model*
- *Ad quality scoring (is the ad trustworthy?)*
- *Brand safety scoring (is this page appropriate for ads?)*
- *Budget pacing model (how to spread budget over time)*

*Each team trains its own model, maintains its own feature pipelines, and runs its own serving infrastructure. This is ~5x the cost.*

*The platform investment: a unified 'ads understanding' model with multiple output heads (CTR, conversion, quality, safety) trained jointly on shared features. One feature store. One serving infrastructure.*

*The organizational challenge: these models are owned by different teams (monetization, policy, finance). To build a shared platform, you need a cross-functional team with authority to set standards and a clear API contract between the platform team and the product teams.*

*This is the same build-vs-buy problem but at the org level: you're deciding whether to 'build' each team's ML capability independently or 'buy' a shared platform service. The shared platform is the right call for infrastructure like feature stores and serving, but wrong for model architecture (product teams should own that).*

*Privacy roadmap: Apple ATT is phase 1. Likely, all browsers will remove third-party cookies and cross-site tracking within 3 years. The ML model needs to perform acceptably on contextual targeting alone (no user history). Run an experiment today: hold out 10% of users and only show them contextually-targeted ads. Measure the revenue gap. This gives you the cost of losing personalization, which informs how much to invest in privacy-preserving alternatives (FLoC, federated learning).*"

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**DeepFM:**
```
y = sigmoid(y_FM + y_DNN)
y_FM = w_0 + Σ_i w_i*x_i + Σ_i Σ_{j>i} <v_i, v_j> * x_i * x_j
y_DNN = MLP(concat(v_1, v_2, ..., v_m))
```

**Binary Cross-Entropy:**
```
L = -(1/N) Σ [y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i)]
```

**Normalized Cross-Entropy:**
```
NCE = CE_model / CE_baseline
CE_baseline = -[p * log(p) + (1-p) * log(1-p)], p = empirical CTR
```
(NCE < 1: better than predicting average CTR)

**eCPM (auction ranking score):**
```
eCPM = P(click) × bid × quality_score
```

**Platt Scaling (calibration):**
```
P_calibrated = sigmoid(a * logit(ŷ) + b)
```

**Focal Loss (for class imbalance):**
```
FL(pt) = -α * (1-pt)^γ * log(pt)
α = 0.25, γ = 2.0
```

**IPS (position bias correction):**
```
L_IPS = -(1/N) Σ_i y_i / e(pos_i) * log(ŷ_i)
e(pos_i) = P(click | position_i, random ad)
```

**Survival Analysis (delayed labels):**
```
P(conversion by time T | click, features) = 1 - exp(-Λ(T | features))
```

### Vocabulary Cheat Sheet

| Term | Definition |
|------|-----------|
| DeepFM | Deep + Factorization Machine: captures pairwise and higher-order feature interactions |
| FM component | Factorization machine: pairwise interactions via embedding dot products |
| Deep component | MLP that captures complex higher-order interactions |
| eCPM | Effective Cost Per Mille: predicted revenue per 1000 impressions = CTR × bid |
| Second-price auction | Winner pays the second-highest bid, not their own bid (strategy-proof) |
| Calibration | Ensuring predicted probabilities match empirical frequencies |
| Platt scaling | Linear calibration: sigmoid(a*logit(ŷ)+b) |
| NCE | Normalized Cross-Entropy: CE_model / CE_baseline |
| Label delay | Conversion events observable only after a time delay |
| Click fraud | Invalid clicks by bots/farms to drain competitor budgets |
| ATT | Apple App Tracking Transparency: restricts cross-app behavioral data |
| Position bias | Click rate inflated by ad position, not ad quality |
| IPS | Inverse Propensity Scoring: corrects for position/selection bias in training |
| Continual learning | Frequent model updates on streaming data without full retraining |
| Catastrophic forgetting | Model losing old knowledge when fine-tuned on new data |
| Exploration traffic | Random ad serving to collect unbiased click signal |

### Key Numbers

| Metric | Value |
|--------|-------|
| Impressions per day | 100B+ |
| Ad universe | Billions of ads |
| Prediction latency | <30ms P50, <50ms P99 |
| Continual learning frequency | Every 5 minutes (minimum) |
| Click fraud rate | 1-5% of impressions |
| Class imbalance | ~1-2% click rate |
| Calibration target (MAE) | < 0.001 per decile |
| Exploration traffic | 5% of impressions |
| Conversion label delay | Up to 7 days |
| Feature store latency (online) | <5ms (Redis) |
| Model rollback SLA | <10 seconds |
| Valid NCE range | 0.80-0.90 for good models |

### Rapid-Fire Day-Before Review

**Q: Why DeepFM over a plain DNN for ad click prediction?**
A: Ad click prediction relies heavily on feature interactions — (user likes cooking) × (ad is cooking) → high CTR. Plain DNN learns interactions implicitly but needs many examples. FM's explicit pairwise interaction via embedding dot products captures sparse feature interactions even with few observations.

**Q: Why is calibration a first-class requirement for ad systems?**
A: Predicted CTR × bid = eCPM = auction ranking score. If P(click) is systematically off, the wrong ads win auctions. Advertisers discover ROI is below expectation → reduce spend → revenue loss. Calibration ensures the auction pricing is fair.

**Q: What is NCE and what does NCE=0.85 mean?**
A: NCE = CE_model / CE_baseline where baseline predicts average CTR. NCE=0.85 means the model has 15% lower cross-entropy loss than predicting average CTR for everyone — a significant improvement.

**Q: How do you handle the 5-minute continual learning requirement without catastrophic forgetting?**
A: (1) Fine-tune on recent data with small learning rate; (2) replay buffer — mix 10-20% old examples into each new batch; (3) warm start from previous model weights.

**Q: Why is Apple ATT impactful and how do you mitigate it?**
A: ATT removes cross-app behavioral signals for iOS users (50%+ of mobile). Mitigation: contextual targeting (page content, time, device), cohort-based targeting (Google Topics), on-device ML for personalization without data transmission.
