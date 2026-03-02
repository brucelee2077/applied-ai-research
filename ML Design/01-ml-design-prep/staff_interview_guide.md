# ML Design Interview Preparation Guide — Staff/Principal Bar

---

## How to Use This Guide

This is the orientation document you read **first** — before any domain-specific module (recommendation systems, search, ads, safety, etc.). Its job is to install the mental operating system for ML design interviews at the Staff and Principal level.

A Staff or Principal ML interview is not a knowledge test. It is a judgment test. The interviewer is not checking whether you memorized transformer architectures or know the formula for nDCG. They are evaluating whether you think the way a senior technical leader thinks: starting from business reality, moving systematically through trade-off space, and landing on a design that is defensible, practical, and appropriately scoped.

This guide is organized around the 8-step framework that underlies every well-structured ML design answer. But more importantly, it explains the *reasoning* behind each step, the *signals* that separate hire from no-hire, and the *meta-strategy* for managing a 45-minute session under pressure.

**How to study this guide:**

1. Read it end to end once, slowly. Do not skim the model answers.
2. Internalize the 4-level quality rubric (No Hire → Weak Hire → Hire → Strong Hire). Every time you practice a problem, score yourself honestly against it.
3. Memorize Section 10 (formulas and key numbers) so they are available to you without cognitive overhead during the interview.
4. Use this guide as the frame, then study the domain-specific modules (recommendation, search, ads, safety) as the content that fills the frame.

A typical Staff ML interview is 45 minutes. A Principal interview may run 60 minutes with a harder follow-up phase. This guide covers both.

---

## The 8-Step ML Design Framework

Every ML system design problem — no matter the domain — can be answered using this 8-step structure. The steps are not arbitrary; they mirror the actual lifecycle of an ML product from concept to production. Interviewers at top companies have internalized this lifecycle and will notice immediately if you skip or conflate steps.

The eight steps are:

1. **Clarifying Requirements** — Establish what you are building, for whom, at what scale, under what constraints.
2. **Frame as ML Task** — Translate the business goal into a precise ML objective with defined inputs and outputs.
3. **Data Preparation** — Identify data sources, storage, feature engineering strategies, and label construction.
4. **Model Development** — Select and justify a model architecture, describe training procedure, handle class imbalance.
5. **Evaluation** — Define offline and online metrics; design an A/B test.
6. **Deployment and Serving** — Design the serving architecture; address latency, throughput, and model compression.
7. **Monitoring** — Plan for drift detection, data quality, and operational alerting.
8. **Infrastructure** — Address platform dependencies, compute, data pipelines, and system reliability.

The following eight sections of this guide give a deep dive on each step. But first, understand the meta-principle:

> **The framework is a communication tool, not a checklist.** You are not filling out a form. You are telling a coherent story about an ML system. The framework keeps that story logically ordered. At the Staff/Principal level, what matters most is the quality of reasoning inside each step, not mechanical completion of every sub-bullet.

---

## Section 1: Mastering Requirements Clarification

### The 6 Dimensions You Must Cover

Requirements clarification is the most underrated part of the ML design interview. Junior and mid-level candidates often rush past it to get to the "interesting" parts (models, architectures). This is a mistake. The 2–3 minutes you spend on clarification determine whether the rest of your answer is targeted and coherent or generic and unfocused.

Cover these six dimensions in every interview:

**1. Business Objective**
What is the company actually trying to achieve? Increase revenue? Reduce churn? Improve safety? The ML system is a means to this end. If you design a system that optimizes the wrong proxy, you fail even if the ML is technically brilliant.

**2. Scale**
How many users? How many items? How many requests per second? What is the data volume? Scale determines whether simple baselines are sufficient or whether you need distributed training, approximate nearest neighbor search, or specialized serving infrastructure.

**3. Latency**
What is the acceptable end-to-end latency? 10ms? 100ms? 1 second? Latency constraints directly gate model complexity, the use of real-time vs. precomputed features, and the feasibility of online learning.

**4. Data Availability**
What labeled data exists today? How is it collected? Is the label clean or noisy? Is the data biased in ways that matter? This determines whether you train from scratch, fine-tune a pretrained model, or use weak supervision.

**5. Interaction Types**
Is this a one-shot prediction or a multi-turn interaction? Is user feedback explicit (ratings, clicks) or implicit (dwell time, skips)? The interaction type determines how you construct training labels and how you close the feedback loop.

**6. Constraints**
Privacy (GDPR, CCPA), fairness (protected attributes), latency budget, memory budget, regulatory requirements. These are non-negotiable guardrails that must be acknowledged early.

### Why Each Dimension Matters

Skipping any dimension leads to a specific class of design error:

- Skipping **business objective** → optimizing a proxy metric that doesn't move the business needle (e.g., maximizing CTR at the expense of user satisfaction).
- Skipping **scale** → proposing a design that works for 10K users but fails at 100M.
- Skipping **latency** → recommending a 500ms deep learning model for a search autocomplete system.
- Skipping **data availability** → designing a supervised system when only 500 labeled examples exist.
- Skipping **interaction types** → constructing labels from clicks without correcting for position bias.
- Skipping **constraints** → proposing a model that uses protected attributes in a lending or hiring context.

### Model Answers by Level

The following model answers use the prompt: "Design a recommendation system for a video streaming platform."

#### No Hire Answer

"Sure, I'd build a collaborative filtering model. You collect user watch history, build a user-item matrix, apply matrix factorization, and then recommend the top-k items with highest predicted scores. For evaluation I'd use precision and recall. For serving, I'd precompute recommendations nightly and store them in a database."

**Why this is a No Hire:** No clarifying questions were asked. The scale is unknown. The business objective is assumed. The candidate jumped directly to a solution without establishing the problem. The answer is generic and could apply to any recommendation system circa 2012.

#### Weak Hire Answer

"Before I start, let me ask a few questions. Is this for a company like Netflix or YouTube? How many users do we have? What kind of videos — long form or short form? OK, assuming 50M users and long-form content, I'd use collaborative filtering with matrix factorization. I'd train on watch history. The main metric is recall at 10. I'd serve from a precomputed table. If latency is tight, I'd cache recommendations. I'd monitor for performance degradation over time."

**Why this is a Weak Hire:** Questions were asked, which is good. But the questions were shallow — asking "Netflix or YouTube?" is flavor, not substance. The candidate didn't probe the six dimensions with precision. The business objective (engagement? retention? subscriber growth?) was not established. Scale was accepted as a given without implications being drawn out. The answer hit the right topic areas but at surface depth.

#### Hire Answer (Staff)

"Before I propose any architecture, I want to make sure I understand the problem well enough to design the right system. Let me ask a structured set of questions across six dimensions.

First, the business objective: are we optimizing for engagement (watch time, session length), subscriber retention, or new subscriber acquisition? These lead to very different objective functions. Optimizing for click-through rate can maximize short-term engagement but damage long-term retention if we recommend clickbait. I'd want to understand the primary KPI and whether there are guardrail metrics — for example, maximizing watch time while not letting satisfaction scores fall below a threshold.

Second, scale: roughly how many monthly active users, how many items in the catalog, and what is the expected QPS for the recommendation endpoint? 50M users and 500K videos is a very different engineering problem than 1B users and 5M videos. At large scale, approximate nearest neighbor search becomes necessary, the user-item matrix doesn't fit in memory, and we need to think about distributed training.

Third, latency: what is the acceptable p99 latency for serving recommendations? If this is a homepage refresh that a user sees every session, 200ms might be acceptable. If it's an autoplay trigger between episodes, we probably have a few seconds of precomputed buffer and latency matters less. This determines whether we can afford real-time feature retrieval or must rely on precomputed embeddings.

Fourth, data availability: what interaction signals do we have — plays, completions, ratings, skips? How much history per user? New users with no history require a cold-start strategy. Also, is there any editorial or human-curated data we can use for supervised pretraining?

Fifth, interaction types: is feedback explicit (star ratings) or implicit (watch percentage, rewatch, shares)? Implicit feedback is harder to work with because a non-click doesn't mean dislike — it might mean the item was never seen. This matters a lot for how we construct training labels and whether we need to correct for exposure bias.

Sixth, constraints: any regulatory requirements around content in certain regions? COPPA for under-13 users? Fairness requirements around content diversity — for example, ensuring that recommendations don't create filter bubbles? Memory or compute constraints on the recommendation service?

Based on your answers, I'd adjust my design accordingly. For now, let me assume: primary objective is 7-day retention, scale is 100M users and 2M videos, latency budget is 100ms p99, implicit feedback (watch events), and no hard regulatory constraints beyond standard data privacy. With those assumptions, here's how I'd approach the system..."

**Why this is a Hire (Staff):** The candidate demonstrated systematic command of the 6 dimensions without prompting. Each question was accompanied by an explanation of *why it matters* — this signals that the candidate understands the implications, not just the taxonomy. The answer concluded with explicit assumptions, which is critical: it prevents the rest of the design from being built on unstated foundations.

#### Strong Hire Answer (Principal)

"I want to approach this differently than I would have five years ago. The instinct is to immediately ask about scale and latency and then sketch a two-tower architecture. But at the Principal level, I think the most valuable thing I can do first is to challenge whether we're solving the right problem.

So let me start with first principles. A video recommendation system exists to serve multiple stakeholders simultaneously: users want to discover content they'll enjoy, creators want distribution for their work, and the business wants to generate subscription revenue and maintain platform health. These objectives can align, but they frequently conflict. A purely engagement-maximizing system will favor established creators and sensationalist content, because those generate reliable watch time. A content-diversity objective might reduce aggregate watch time but improve long-term platform health and creator ecosystem vitality. I want to understand how the business thinks about these trade-offs before I design any system.

Concretely, the questions I'd ask are: What are the one or two metrics that, if they moved, would most directly indicate the recommendation system is working? And what are the guardrail metrics — the things we absolutely cannot let degrade, even if the primary metric improves? I've seen systems that were technically impressive but destroyed the business because the objective function didn't capture what actually mattered.

On the data side, I want to understand not just what data exists but what the *data generating process* looks like. User interaction data is not a random sample of preferences — it's a sample of what the system has already shown users. This exposure bias is a fundamental challenge: if the current system never shows a user documentaries, we have no data about whether that user would enjoy documentaries. Any model trained on this data will perpetuate the existing system's biases. At the Principal level, I'd want to address this architecturally: consider an explore-exploit framework where some fraction of traffic is randomized for unbiased data collection, or use counterfactual estimation techniques like inverse propensity scoring.

The six dimensions I'd probe in detail are: (1) the primary business KPI and guardrail metrics, (2) user base size and composition including cold-start fraction, (3) catalog size, freshness requirements, and whether new items need to be surfaced quickly, (4) latency budget decomposed by stage — retrieval vs. ranking vs. re-ranking each have different constraints, (5) interaction signals available and their quality, specifically how noisy the implicit feedback is, and (6) regulatory and fairness constraints including whether the platform operates in markets with content restriction requirements.

Beyond these six, I'd ask one question that most candidates miss: what does the existing system look like, and what are its known failure modes? Designing a recommendation system in a greenfield context is a different problem from improving an existing one that already serves 100M users. In the latter case, your design choices must account for system continuity, rollback risk, and the organizational cost of migrating infrastructure. I want to design a system that can be actually shipped, not just a theoretically optimal one.

After this dialogue, I'd explicitly state my assumptions, acknowledge the ones I'm least sure about, and propose we revisit them as the design evolves. This is how I'd approach the problem at the Principal level: treat requirements clarification as an active investigation of problem structure, not a formality before getting to the 'real' engineering."

**Why this is a Strong Hire (Principal):** The candidate reframed the problem from the start, demonstrating systems thinking beyond the immediate technical question. The discussion of multi-stakeholder objectives, exposure bias, and counterfactual estimation reflects genuine production experience at scale. The point about existing system failure modes is something only a Principal-level practitioner surfaces — it signals organizational awareness. The answer is not longer for its own sake; every additional point connects to a specific design implication. This is the signal of a principal: they see around corners.

---

## Section 2: ML Problem Framing — Business to ML Objective Translation

The most consequential decision in any ML system is the formulation of the ML objective. A misformulated objective produces a system that is technically correct but commercially wrong.

**The translation process has four steps:**
1. Identify the business outcome.
2. Identify a measurable proxy that correlates with the business outcome.
3. Specify the ML task (classification, regression, ranking, generation).
4. Define the input space, output space, and the interpretation of the output.

### Examples Across Domains

**Recommendation:**
- Business objective: increase 7-day retention.
- Proxy: predict probability of a user watching at least 70% of a recommended video.
- ML task: binary classification (watched ≥ 70% vs. not).
- Input: user embedding, item embedding, context features (time of day, device).
- Output: scalar probability in [0, 1].
- Note: CTR is a tempting but dangerous proxy — it optimizes for click, not for satisfied watch. The distinction matters enormously.

**Search:**
- Business objective: improve user task completion rate.
- Proxy: predict relevance of each candidate document to a query.
- ML task: pointwise regression or listwise ranking.
- Input: query representation, document representation, interaction features.
- Output: relevance score used to rank candidates.
- Note: At the Principal level, recognize that relevance alone is insufficient — diversity, freshness, and query intent ambiguity must all be handled.

**Safety / Content Moderation:**
- Business objective: reduce harmful content exposure while minimizing false positive removal of legitimate content.
- ML task: binary or multi-class classification.
- Output: probability of policy violation per category.
- Note: This is inherently a multi-objective problem. Framing it as a single classifier collapses the nuance. At the Staff/Principal level, decompose into: is this content harmful (detection model) × what is the appropriate action (policy model).

**Ads:**
- Business objective: maximize revenue.
- Proxy: predict expected revenue per impression = P(click) × P(conversion | click) × bid × value.
- ML task: multiple — CTR prediction (classification) + conversion prediction (classification) + revenue calibration (regression).
- Note: The cascade structure matters. Errors in CTR prediction compound with errors in conversion prediction. At the Principal level, consider end-to-end training approaches (like post-click conversion models) to reduce this compounding.

### 4-Level Rubric

**No Hire:** States the ML task without explanation. "We'd use a classifier."

**Weak Hire:** Correctly identifies the ML task and specifies input/output. "We'd train a binary classifier to predict whether a user clicks on a recommendation, using user features, item features, and context."

**Hire (Staff):** Discusses the proxy-metric gap and proposes a formulation that better aligns with the business objective. Acknowledges that CTR ≠ satisfaction, and proposes a richer label (watch completion rate, downstream engagement) or a multi-task objective that balances short-term and long-term signals.

**Strong Hire (Principal):** Surfaces the objective function design as a first-class decision with organizational implications. Proposes multi-task learning where tasks correspond to different business objectives (engagement, quality, safety), discusses how to weight them, acknowledges that the weighting is a product decision not a technical one, and describes a process for calibrating weights using experimentation. Also discusses the feedback loop problem: the model's outputs change what data gets collected, which changes future model training. Proposes architectural solutions (epsilon-greedy exploration, bandit frameworks, IPS correction) to manage this loop.

---

## Section 3: Data & Feature Engineering Mastery

### Feature Taxonomy

Features in ML systems fall into five categories:

**1. User features** — who the user is: demographics (age, location), historical behavior (watch history, purchase history), real-time context (current session activity).

**2. Item features** — what the item is: content metadata (genre, duration, creator), embeddings (visual, textual, audio), popularity signals (global view count, trending score).

**3. Context features** — situational signals: time of day, day of week, device type, network quality, geographic region. Context features are frequently underweighted by junior candidates but are often highly predictive.

**4. Cross features** — interactions between feature pairs: user_age × content_genre, device_type × content_duration. DeepFM and similar architectures are specifically designed to learn cross features automatically.

**5. Label-derived features** — predictions from upstream models used as features downstream. For example, using a content quality score as a feature in the ranking model.

### Encoding Strategies

**Continuous features:** Apply normalization (subtract mean, divide by std) or standardization for bounded features. Use log transformation for heavy-tailed distributions (e.g., view counts, revenue). Clip outliers at the 99th percentile before scaling.

**Categorical features with low cardinality (< ~50 categories):** One-hot encoding. Sparse but interpretable.

**Categorical features with high cardinality (> ~50 categories):** Embedding lookup. For user IDs and item IDs, learned embeddings are the foundation of most modern recommendation systems. Initialize randomly, learn end-to-end. For very large catalogs, use hash embeddings to control memory.

**Ordinal categorical features:** Integer encoding preserves ordinal information. Do not one-hot encode ordinal features — it destroys the ordering.

**Missing values:** Do not delete rows with missing values unless the missingness rate is very low (< 1%) and you have confirmed the data is missing completely at random (MCAR). Use median imputation for continuous features, mode imputation for categorical, or a learned imputation model for high-stakes features. Always add a binary indicator feature `feature_is_missing` to let the model learn from the missingness pattern itself.

### Label Construction Principles

Label construction is where most production systems make their most consequential errors. Key principles:

**Positive label definition:** Be precise. "User clicked" is not the same as "user found relevant." "User watched ≥ 30 seconds" is not the same as "user watched ≥ 70%." Each definition produces a different model with different biases.

**Negative sampling:** In implicit feedback settings, you do not have true negatives — you have unobserved interactions, which could mean "not shown," "shown but ignored," or "shown, seen, and actively disliked." Treat these differently. Random negatives from the corpus are not equivalent to hard negatives (items that were shown but not engaged with).

**Temporal splits:** Always split your data by time, not randomly. Random splits allow future information to leak into training. A model trained with random splits will appear to have much better offline metrics than it will achieve in production.

**IPS correction for position bias:** When learning from user interactions with ranked lists, items at position 1 receive far more exposure than items at position 10, independent of their quality. Failing to correct for this produces models that learn to recommend items that appear at the top of lists, not items that users actually prefer. The IPS-corrected loss is:

```
L_IPS = Σ (L_i / e_i)
```

where `e_i = P(examined | position i)` is the examination probability at position i, estimated from randomization experiments or propensity models.

### 4-Level Rubric

**No Hire:** Lists feature types without discussing encoding or label quality. Proposes random train/test split.

**Weak Hire:** Correctly describes encoding strategies. Mentions the need for temporal splits. Does not discuss IPS or label noise.

**Hire (Staff):** Describes a coherent feature engineering pipeline. Discusses missing value treatment, outlier handling, and embedding strategies. Raises label quality as a concern and proposes a specific strategy for handling implicit feedback (e.g., watch completion rate with IPS correction for position bias). Discusses the cold-start problem and how to handle new items/users without history.

**Strong Hire (Principal):** Treats feature engineering as an ongoing system, not a one-time design. Describes a feature store architecture that enables feature sharing across teams and models. Discusses the train-serve skew problem (features computed differently at training vs. serving time) as a major source of production degradation and describes architectural solutions (log-and-replay for training, feature store for serving). Proposes a data flywheel strategy: each generation of the model collects better training data for the next generation. Discusses fairness implications of feature selection — which features encode protected attributes implicitly, and what to do about it.

---

## Section 4: Model Architecture Decision Making

### The Baseline-to-Production Progression

A principled model selection process always starts simple and increases complexity only when justified by evidence. The progression is:

**Tier 1: Non-ML baseline**
A rule-based system or popularity-based ranker. This sets the floor and is often surprisingly competitive. It also gives you a debugging baseline — if your ML model underperforms the popularity ranker, something is wrong.

**Tier 2: Simple ML models**
Logistic regression, decision trees, linear regression. These are interpretable, fast to train, and easy to debug. They also expose whether the feature engineering is doing the right work. If a logistic regression with well-engineered features performs well, you may not need deep learning.

**Tier 3: Complex ML models**
Gradient boosted trees (XGBoost, LightGBM) for tabular data. Two-tower neural networks for retrieval. Transformer-based models for sequential behavior. These require more data, more compute, and more careful hyperparameter tuning.

**Tier 4: Specialized architectures**
DeepFM, DIN (Deep Interest Network), DLRM for recommendation. ColBERT, DPR for dense retrieval in search. These are domain-specific architectures developed specifically for the problem at hand. Justify them by specific properties of the problem (e.g., high-cardinality categorical features, sequential dependencies, multi-modal inputs).

### Key Architecture Decisions

**Two-tower architecture (retrieval):** One tower encodes the query (user + context), one tower encodes the candidate item. Dot product similarity at inference time. Enables precomputation of item embeddings and fast approximate nearest neighbor (ANN) search at serving. The trade-off is that the dot product interaction is shallow — the towers cannot interact until the final similarity computation. This limits the expressiveness of cross-feature interactions.

**DeepFM:** Combines a factorization machine (FM) component for second-order feature interactions with a deep neural network (DNN) component for higher-order interactions:

```
y = sigmoid(y_FM + y_DNN)
```

Useful when you have many categorical features with complex interaction patterns. Commonly used in CTR prediction.

**WALS (Weighted Alternating Least Squares):** Matrix factorization for collaborative filtering with implicit feedback. The objective function is:

```
L = Σ w_ui(r_ui - u_i^T v_u)^2 + λ(||U||² + ||V||²)
```

where `w_ui` is a confidence weight (higher for observed interactions), `r_ui` is the implicit feedback signal, `u_i` is the user embedding, `v_u` is the item embedding, and `λ` is the regularization coefficient. WALS is efficient because it alternates between fixing U and solving for V, and fixing V and solving for U — each of which is a closed-form least squares problem.

**Contrastive learning (InfoNCE):** Used to train dense retrievers and general-purpose embeddings. The InfoNCE loss pulls together representations of similar items and pushes apart representations of dissimilar items:

```
L = -log[exp(sim(q,k+)/τ) / Σ exp(sim(q,ki)/τ)]
```

where `q` is the query representation, `k+` is the positive key, `ki` are the negative keys, and `τ` is the temperature hyperparameter. Lower temperature makes the distribution more peaked (harder negatives receive higher loss). Hard negative mining — selecting informative negatives that are similar to the query but not relevant — is critical for training quality embeddings.

### Training Considerations

**Class imbalance:** In CTR prediction, negative rates of 99%+ are common. Options:
- Downsampling negatives (with or without IPS correction).
- Upsampling positives (less preferred — can cause overfitting).
- Class-weighted loss: assign higher weight to the minority class.
- Focal loss (designed specifically for class imbalance in object detection, applicable elsewhere): `FL(pt) = -α(1-pt)^γ * log(pt)` where `α` balances positive/negative class frequency and `γ` downweights easy examples.

**Training from scratch vs. fine-tuning:** Fine-tuning a pretrained model is almost always preferable when a relevant pretrained model exists. It requires less data, trains faster, and typically generalizes better. However, you must ensure the pretraining distribution is compatible with your target distribution. For text-based systems, fine-tuning BERT/RoBERTa/LLaMA variants is standard. For image/video systems, fine-tuning CLIP or a vision transformer is standard.

### 4-Level Rubric

**No Hire:** Names a model without justification. "I'd use a neural network."

**Weak Hire:** Describes a reasonable architecture with some justification. Knows that collaborative filtering is used for recommendation, BERT for text.

**Hire (Staff):** Presents the baseline-to-production progression. Justifies each step up in complexity. Discusses two-tower vs. cross-attention trade-offs explicitly. Mentions specific hyperparameters and their effects (embedding dimension, regularization, temperature). Addresses class imbalance with a concrete strategy.

**Strong Hire (Principal):** Treats model architecture as part of a system, not a standalone component. Discusses how retrieval and ranking interact (does the retrieval model's recall bound the ranking model's precision?). Considers multi-task learning for jointly optimizing engagement, quality, and safety. Raises the question of model maintainability: a simpler model that engineers can debug and improve incrementally may be preferable to a state-of-the-art architecture that is a black box. Proposes an architecture that enables continuous learning (online model updates from streaming data) and addresses the catastrophic forgetting problem when fine-tuning pre-trained models on new data.

---

## Section 5: Evaluation Strategy

### Offline Metrics

Choose your offline metric based on the ML task:

**Classification:**
- Accuracy: do not use when classes are imbalanced.
- Precision = TP / (TP + FP): of items predicted positive, what fraction are actually positive.
- Recall = TP / (TP + FN): of all actual positives, what fraction did we predict positive.
- F1 = 2 * (Precision * Recall) / (Precision + Recall): harmonic mean; balances precision and recall.
- AUC-ROC: discrimination ability of the classifier across all thresholds. Use when you need threshold-agnostic evaluation.
- PR-AUC: better than ROC-AUC for imbalanced datasets.

**Regression:**
- MSE = (1/n) Σ(y - ŷ)²: heavily penalizes large errors. Sensitive to outliers.
- MAE = (1/n) Σ|y - ŷ|: more robust to outliers.
- RMSE: same units as target variable; interpretable.

**Ranking:**
- nDCG (Normalized Discounted Cumulative Gain):
```
DCG_p = Σ (2^rel_i - 1) / log₂(i+1)
nDCG = DCG / IDCG
```
where `rel_i` is the relevance of the item at position i and IDCG is the ideal (maximum possible) DCG. nDCG is the standard metric for learning-to-rank systems.

- MRR (Mean Reciprocal Rank):
```
MRR = (1/|Q|) Σ 1/rank_i
```
where `rank_i` is the rank of the first relevant item for query i. MRR is simple and interpretable; it gives full credit to the first relevant result.

- MAP (Mean Average Precision): average of precision at each recall level; useful when multiple relevant items exist.

### Online Metrics

Offline metrics are necessary but not sufficient. A model can improve offline metrics while degrading business metrics. Online metrics are the ground truth.

**Common online metrics by domain:**
- Recommendation: CTR (click-through rate), watch time, 7-day retention, session length.
- Search: CTR, NDCG measured from user behavior, task completion rate.
- Ads: CTR, conversion rate, RPM (revenue per thousand impressions), ROAS (return on ad spend).
- Safety: false positive rate (legitimate content incorrectly removed), recall on policy violations.

### A/B Test Design

An A/B test is the standard method for online evaluation. Design it correctly:

**Power analysis:** Determine the minimum sample size before running the experiment.

```
n = 2σ²(z_α + z_β)² / δ²
```

where:
- `σ²` is the variance of the metric.
- `z_α` is the z-score for the significance level (e.g., 1.96 for α = 0.05).
- `z_β` is the z-score for the desired power (e.g., 0.84 for β = 0.20, i.e., 80% power).
- `δ` is the minimum detectable effect — the smallest improvement you care about detecting.

This formula tells you how many users you need in each arm of the experiment. Running an underpowered A/B test and concluding "no significant effect" is a Type II error — a false negative.

**Common A/B test design mistakes:**
- Peeking: checking results before the planned end date and stopping when p < 0.05. This inflates the false positive rate. Use sequential testing or pre-commit to a test duration.
- Network effects: if users interact (social networks, marketplaces), treatment can leak into control. Use cluster-based randomization.
- Novelty effect: users may engage more with any new system simply because it's new. Run experiments long enough to account for this (typically 2-4 weeks).
- Metric dilution: if most users in the experiment are ineligible for the feature, the per-user treatment effect is diluted. Restrict analysis to triggered users.

### Calibration

A model that ranks well (high AUC) may be poorly calibrated — its predicted probabilities do not reflect true frequencies. Calibration matters for systems where scores are used directly (e.g., ad auction bid prices, risk scoring).

**Platt scaling** is a post-hoc calibration method that fits a logistic regression on top of the raw model score:

```
P(y=1|f(x)) = 1 / (1 + exp(Af(x) + B))
```

where A and B are calibration parameters learned on a held-out set. Platt scaling is widely used in production ad systems to recalibrate CTR predictions after any model update.

### 4-Level Rubric

**No Hire:** Mentions accuracy and A/B testing without depth. Does not know nDCG or MRR.

**Weak Hire:** Correctly selects metrics for the task. Knows basic A/B testing. Cannot derive the power formula or discuss calibration.

**Hire (Staff):** Proposes a complete offline/online evaluation plan. Discusses the proxy-metric gap. Knows the power formula and can reason about experiment duration. Discusses Platt scaling for calibration. Addresses common A/B test pitfalls.

**Strong Hire (Principal):** Treats evaluation as a continuous organizational process, not a one-time experiment. Proposes a metrics hierarchy (primary KPIs, secondary metrics, guardrail metrics) and explains how to arbitrate conflicts. Discusses Bayesian A/B testing as an alternative to frequentist testing. Raises the interplay between evaluation and feedback loops — A/B test results are themselves training data for the next model iteration, so experiment design must account for long-term effects, not just in-experiment metrics. Discusses counterfactual evaluation (offline simulation of A/B test outcomes) for cases where running live experiments is expensive.

---

## Section 6: Production Systems Thinking

### Serving Architecture Patterns

A production ML serving system must handle: (1) feature retrieval, (2) model inference, (3) result post-processing, and (4) logging.

**Two-stage (retrieval + ranking):** The standard pattern for recommendation and search. Stage 1 retrieves O(100-1000) candidates from a corpus of millions using approximate nearest neighbor (ANN) search (e.g., FAISS, ScaNN, HNSW). Stage 2 ranks the candidates using a heavier model that can afford to process fewer items with more features. This pattern decouples recall (maximized in retrieval) from precision (maximized in ranking).

**Multi-stage:** Adds a re-ranking stage after the main ranker. Re-ranking can apply business rules (diversity constraints, freshness boosts, policy filters), personalization signals that are expensive to compute at scale, or a slate-level optimization (considering item interactions within the recommendation set).

**Batch vs. online prediction:**
- Batch: precompute predictions for all users nightly. Low serving latency (cache lookup). Stale — does not respond to within-session behavior. Good for non-time-sensitive recommendations.
- Online: compute predictions in real time using fresh features. High serving latency cost. Responsive to current context. Required for search, ads, and real-time feeds.
- Hybrid: precompute item embeddings, compute user embeddings in real time. A common pattern that balances freshness with latency.

**Latency budget allocation:** A 100ms budget decomposed into stages might look like:
- Feature retrieval from feature store: ~10ms
- ANN retrieval: ~20ms
- Feature assembly for ranking: ~15ms
- Ranking model inference: ~30ms
- Post-processing and logging: ~10ms
- Network overhead: ~15ms

If any stage exceeds its budget, the system misses the SLA. Each stage must be independently optimized.

### Model Compression

When latency constraints are tight, compress the model:

**Knowledge distillation:** Train a smaller student model to mimic the output distribution of a larger teacher model. The student is trained on the teacher's soft predictions (probabilities) rather than hard labels. Soft predictions contain more information about the model's uncertainty and inter-class relationships. The distillation loss combines the standard cross-entropy loss on hard labels with a KL divergence term on soft labels from the teacher.

**Pruning:** Remove weights with small magnitude (unstructured pruning) or remove entire channels/heads (structured pruning). Structured pruning maps more naturally to hardware acceleration. Pruning typically requires fine-tuning after pruning to recover accuracy.

**Quantization:** Reduce weight and activation precision from FP32 to INT8 or FP16. INT8 quantization typically reduces model size by 4× and improves inference throughput by 2-4× with minimal accuracy loss. Post-training quantization (PTQ) applies quantization after training; quantization-aware training (QAT) simulates quantization during training for higher accuracy.

### Cloud vs. On-Device Deployment

**Cloud inference:** Centralized, easy to update, can use large models, but requires network round-trip. Required for server-side business logic.

**On-device inference:** No network latency, works offline, privacy-preserving. Requires small model (< ~100MB), careful runtime engineering (Core ML, TensorFlow Lite, ONNX Runtime). Required for real-time audio processing, camera-based features, low-latency keyboard suggestions.

### 4-Level Rubric

**No Hire:** Describes a monolithic serving endpoint. Does not discuss latency decomposition.

**Weak Hire:** Knows the two-stage pattern. Describes batch vs. online. Does not discuss ANN or model compression.

**Hire (Staff):** Designs a complete multi-stage serving architecture. Allocates latency budget explicitly. Discusses model compression options. Addresses the batch vs. online hybrid pattern.

**Strong Hire (Principal):** Treats the serving system as a platform problem. Discusses how the serving infrastructure should be shared across ML use cases (shared feature store, shared model registry, shared A/B testing framework). Proposes capacity planning (what scale of hardware is needed at launch vs. in six months, and how to provision it incrementally). Discusses graceful degradation: what happens when the ranking model is unavailable — fall back to the retrieval model; when the retrieval model is unavailable — fall back to the popularity ranker. This is systems thinking at the organizational level.

---

## Section 7: Failure Mode Reasoning

A senior ML practitioner's most valuable skill is the ability to enumerate the ways a system can fail *before* it fails. Interviewers at the Staff/Principal level explicitly probe for this.

### Common Failure Mode Categories

**Data quality failures:**
- Training data mislabeled at scale (e.g., labeling pipeline bug).
- Training/serving feature skew (features computed differently in training vs. serving).
- Data leakage (future information in training features).
- Distribution shift (training distribution diverges from serving distribution over time).

**Model failures:**
- Overfit to spurious correlations in training data.
- Poor calibration (scores are not well-calibrated probabilities).
- Failure on underrepresented subgroups (model works well on average but fails badly for certain user cohorts).
- Feedback loop amplification (model recommendations change behavior, which changes training data, which reinforces model biases).

**Serving failures:**
- Feature retrieval latency spike causes cache misses, system falls back to degraded experience.
- Model version mismatch (different model versions serving different users during rollout).
- Embedding index not refreshed after catalog update (new items never surface in retrieval).
- Numerical instability in score computation (NaN propagation, overflow).

**Organizational failures:**
- Metric misalignment (the model optimizes what it's told to optimize, but the business objective wasn't correctly specified).
- Evaluation methodology error (A/B test was underpowered; novelty effect was mistaken for genuine improvement).
- Deployment mistake (new model deployed to 100% of traffic without canary testing).

### 4-Level Rubric

**No Hire:** Cannot enumerate failure modes. Does not consider data quality.

**Weak Hire:** Lists a few failure modes (distribution shift, overfitting) without specifics.

**Hire (Staff):** Systematically walks through failure mode categories. Describes specific failure modes relevant to the domain. Proposes monitoring strategies to detect each failure mode. Discusses rollback procedures.

**Strong Hire (Principal):** Frames failure mode reasoning as a proactive organizational practice. Proposes pre-mortems as a standard process before any major deployment. Discusses the difference between recoverable failures (model quality degrades; rollback in minutes) and unrecoverable failures (bad model runs for months and corrupts downstream training data). Raises the concept of a model audit: periodic evaluation on held-out validation sets from multiple time periods to detect drift proactively rather than reactively.

---

## Section 8: Principal-Level Thinking

This section addresses what separates a Staff answer from a Principal answer at the meta level. The four themes below are what Principal candidates demonstrate throughout the interview, not just in designated moments.

### Theme 1: Org Design and Platform Thinking

A Principal engineer thinks about how the system they are designing will be used and maintained by a team of engineers over years, not just by themselves in the current quarter.

Concretely: when designing a recommendation system, a Staff engineer designs a good recommendation system. A Principal engineer designs a recommendation platform that enables multiple product teams to build recommendation experiences without re-implementing retrieval, ranking, or feature engineering from scratch. The difference is abstraction and generalizability.

Platform thinking manifests as: "I'd build the retrieval layer as a generic service that takes a query embedding and returns candidates from any catalog. The product teams can plug in their own ranking models and re-ranking policies on top. This way, when the video team, the podcast team, and the live events team each need recommendation capabilities, they build on the same infrastructure."

### Theme 2: Build vs. Buy

A Principal engineer can articulate when to build custom ML infrastructure vs. when to use existing tools, managed services, or third-party vendors.

The rule of thumb: build when the capability is a core differentiator and you have the team to maintain it; buy or use managed services when the capability is commodity and your time is better spent elsewhere.

In ML systems, candidates for "buy" include: vector databases (Pinecone, Weaviate), feature stores (Tecton, Feast), ML platforms (Vertex AI, SageMaker), monitoring (Arize, Evidently). Candidates for "build" include: proprietary ranking models, custom loss functions, domain-specific embedding models.

### Theme 3: Multi-Stakeholder Objective Management

At the Principal level, you are expected to navigate conflicting objectives across teams. The recommendation system serves creators (who want distribution), users (who want discovery), and the business (which wants revenue). Safety wants to reduce harmful content; growth wants to maximize engagement. These conflict. A Principal engineer proposes a framework for managing this: explicit guardrail metrics, weighted multi-task objectives, and governance processes for adjusting weights.

### Theme 4: Long-Term System Evolution

A Principal engineer thinks about how the system will need to evolve over the next 1-3 years and designs for that evolution. This means: using interfaces that are stable across model changes, separating feature engineering from model training so features can be reused, building evaluation infrastructure that doesn't need to be rebuilt for each new model, and planning the data collection strategy that will generate training signal for future model generations.

### 4-Level Rubric

**No Hire:** Does not surface any of these themes.

**Weak Hire:** Mentions platform thinking or build vs. buy in passing without substance.

**Hire (Staff):** Addresses 1-2 of these themes with concrete examples specific to the problem domain.

**Strong Hire (Principal):** Naturally integrates all four themes throughout the interview, not just when explicitly asked. Demonstrates that these are habitual patterns of thought, not talking points retrieved on demand.

---

## Section 9: The Interview Meta-Strategy

### Time Management (45-Minute Allocation)

A 45-minute ML design interview, optimally structured:

| Phase | Time | Activity |
|---|---|---|
| Requirements Clarification | 5-7 min | Ask the 6 dimensions; state assumptions |
| ML Problem Framing | 3-5 min | Translate business objective to ML objective |
| Data & Features | 7-8 min | Sources, engineering, label construction |
| Model Architecture | 8-10 min | Baseline progression; justify complexity |
| Evaluation | 5-6 min | Offline metrics + A/B test design |
| Serving & Deployment | 5-6 min | Architecture, latency, compression |
| Monitoring | 3-4 min | Drift, operational, ML metrics |
| Follow-up / Deep Dive | 5-8 min | Interviewer-directed exploration |

**Key timing principles:**

- Do not spend more than 8 minutes on model architecture in a 45-minute interview. Candidates who spend 20 minutes on model architecture are signaling that they think this is the important part. It is not. The important parts are requirements, framing, and evaluation.
- Leave 5 minutes of buffer for follow-up. If you finish early, use the buffer to propose extensions or acknowledge limitations you didn't address.
- If the interviewer redirects you, follow them immediately. They are telling you what they want to evaluate. Insisting on finishing a section you were talking about is a signal of poor communication skill.

**Pacing heuristic:** After requirements clarification (~7 minutes), you should be able to state your problem framing in 60 seconds or less. If you cannot state "I am formulating this as a [classification/ranking/generation] problem where the input is X and the output is Y" concisely, you have not finished requirements clarification — you just think you have.

### Common Mistakes at Each Level

**No Hire mistakes:**
- Jumping to a solution before asking a single clarifying question.
- Proposing a solution that does not match the stated constraints (e.g., 500ms deep learning model for a 10ms autocomplete requirement).
- Using undefined jargon ("I'd use a transformer") without explaining the architecture.
- Confusing ML metrics with business metrics.
- Proposing random train/test split.

**Weak Hire mistakes:**
- Asking clarifying questions but not integrating the answers into the design.
- Describing each step in isolation without connecting them (e.g., choosing a model without explaining why it matches the data distribution).
- Naming the right concepts (AUC, nDCG, knowledge distillation) without being able to explain or derive them.
- Treating model selection as the primary decision rather than as one of many.
- Not surfacing any trade-offs.

**Hire (Staff) gaps:**
- Strong on individual components but does not connect them into a coherent system story.
- Good at technical depth but does not raise organizational or business implications.
- Handles the initial question well but struggles with follow-up questions that push on edge cases.
- Does not proactively surface failure modes.

**Principal gaps:**
- Very technically strong but does not demonstrate platform thinking or org-level judgment.
- Raises platform concerns but cannot ground them in concrete design decisions.
- Does not adjust the level of abstraction based on the interviewer's signals.

### How to Handle Follow-up Questions

Follow-up questions in a Staff/Principal interview fall into three types:

**1. Depth probe:** "Can you explain how WALS actually works?" or "Walk me through the math of nDCG."
Response strategy: go to first principles. Do not just state the formula — explain the intuition behind it. What problem is the formula solving? What would go wrong without this approach?

**2. Constraint change:** "What if we reduced the latency requirement to 20ms?" or "What if we had no historical data?"
Response strategy: treat this as a new requirements clarification. State explicitly how the constraint change affects your design. "With a 20ms budget, I can no longer afford a real-time ranking model. I'd shift to fully precomputed embeddings and use ANN lookup exclusively. The trade-off is that recommendations are less contextually responsive."

**3. Challenge:** "I don't think A/B testing works here because of network effects. What would you do?"
Response strategy: engage with the challenge directly. Agree where the interviewer is right. Propose an alternative. "You're right — if users interact with each other, standard A/B tests suffer from SUTVA violations. I'd switch to cluster-based randomization, where the unit of randomization is a social graph cluster rather than an individual user. Geo-based holdouts are another option if the platform has strong geographic segmentation."

In all three cases: **think out loud**. The interviewer is evaluating your reasoning process, not just your final answer. A wrong answer arrived at by excellent reasoning is often better than a correct answer arrived at by lucky intuition.

### Signaling Staff vs. Principal Level

The key behavioral differences that distinguish Staff from Principal in the interview:

| Behavior | Staff | Principal |
|---|---|---|
| Scope | Deep on the assigned problem | Spontaneously expands to adjacent concerns |
| Trade-offs | Discusses trade-offs when asked | Proactively surfaces trade-offs before being asked |
| Failure modes | Identifies common failure modes | Identifies novel failure modes specific to the system |
| Business connection | Connects ML decisions to product metrics | Connects ML decisions to business strategy and org design |
| Uncertainty | Acknowledges uncertainty when present | Uses uncertainty to structure the investigation |
| Asking questions | Asks good clarifying questions | Asks questions that reframe the problem |

---

## Section 10: Cross-Cutting Concepts Reference

### Mathematical Formulations You Must Know

These formulas must be immediately available during the interview. Study them until you can write them from memory, state the intuition behind each term, and explain when each is used.

**Binary Cross-Entropy Loss:**
```
L = -[y * log(p) + (1-y) * log(1-p)]
```
Used for binary classification. `y` is the true label (0 or 1), `p` is the predicted probability. Minimizing this loss is equivalent to maximizing the likelihood under a Bernoulli model.

**WALS (Weighted Alternating Least Squares):**
```
L = Σ w_ui(r_ui - u_i^T v_u)^2 + λ(||U||² + ||V||²)
```
Used for implicit feedback collaborative filtering. `w_ui` controls the confidence we place in observing interaction `(u, i)`. Higher weight for observed interactions; lower weight (but nonzero) for unobserved pairs.

**InfoNCE (Contrastive Loss):**
```
L = -log[exp(sim(q, k+)/τ) / Σ exp(sim(q, ki)/τ)]
```
Used to train dense embedding models. Pulls query closer to its positive key, pushes it away from negative keys. Temperature `τ` controls the sharpness of the distribution.

**Focal Loss:**
```
FL(pt) = -α(1 - pt)^γ * log(pt)
```
Modification of cross-entropy for class-imbalanced settings. The `(1 - pt)^γ` term down-weights easy examples (high `pt`), focusing training on hard negatives. Standard values: `γ = 2`, `α = 0.25`.

**DeepFM:**
```
y = sigmoid(y_FM + y_DNN)
```
Combines factorization machine (explicit pairwise feature interactions) with DNN (implicit higher-order interactions). All input features are shared between the FM and DNN components.

**nDCG:**
```
DCG_p = Σ_{i=1}^{p} (2^rel_i - 1) / log₂(i+1)
nDCG = DCG / IDCG
```
Measures ranking quality. Exponential gain function amplifies the importance of highly relevant items. Logarithmic discount penalizes relevant items placed at lower positions.

**MRR (Mean Reciprocal Rank):**
```
MRR = (1/|Q|) Σ_{i=1}^{|Q|} 1/rank_i
```
Mean of reciprocal ranks of the first relevant result across a set of queries. Simple and interpretable. Only gives credit to the first relevant result.

**IPS Correction:**
```
L_IPS = Σ (L_i / e_i)
```
Inverse propensity scoring for position bias correction. `e_i = P(examined | position i)` is the examination probability at position i. Dividing by the propensity up-weights losses on items that were rarely shown and down-weights losses on items that were heavily promoted.

**A/B Test Power:**
```
n = 2σ²(z_α + z_β)² / δ²
```
Minimum sample size per arm. `σ²` is metric variance, `z_α` is the one-tailed critical value for significance level α, `z_β` is the critical value for power level (1-β), `δ` is the minimum detectable effect.

**Platt Scaling (Calibration):**
```
P(y=1|f(x)) = 1 / (1 + exp(Af(x) + B))
```
Post-hoc calibration. A and B are fit by logistic regression on a held-out calibration set. Used to convert raw model scores to well-calibrated probabilities.

### The 10 Must-Know Loss Functions

**1. Binary Cross-Entropy (BCE)**
```
L = -[y * log(p) + (1-y) * log(1-p)]
```
Use case: binary classification. Foundation of logistic regression and neural network binary classifiers. Output of sigmoid layer directly minimizes this loss.

**2. Categorical Cross-Entropy (Softmax Loss)**
```
L = -Σ y_k * log(p_k)
```
Use case: multi-class classification. `y_k` is the one-hot label, `p_k` is the softmax probability for class k. Minimizing this is equivalent to maximizing log-likelihood under a categorical distribution.

**3. Mean Squared Error (MSE)**
```
L = (1/n) Σ (y_i - ŷ_i)²
```
Use case: regression. Penalizes large errors quadratically. Sensitive to outliers. The optimal point predictor under MSE is the conditional mean.

**4. Mean Absolute Error (MAE)**
```
L = (1/n) Σ |y_i - ŷ_i|
```
Use case: regression with outliers. More robust than MSE. The optimal point predictor under MAE is the conditional median.

**5. Huber Loss**
```
L_δ = (1/2)(y - ŷ)² if |y - ŷ| ≤ δ
      δ(|y - ŷ| - δ/2) otherwise
```
Use case: regression. Combines the best of MSE (smooth near zero) and MAE (robust to outliers). `δ` is a hyperparameter controlling the transition point.

**6. Hinge Loss (SVM Loss)**
```
L = max(0, 1 - y * ŷ)
```
Use case: binary classification in SVMs. Zero for correctly classified examples with sufficient margin. Penalizes examples that are in the margin or misclassified.

**7. Focal Loss**
```
FL(pt) = -α(1 - pt)^γ * log(pt)
```
Use case: class-imbalanced classification (object detection, rare event prediction). The modulating factor `(1 - pt)^γ` reduces the loss contribution of easy examples, focusing training on hard examples.

**8. InfoNCE (Contrastive Loss)**
```
L = -log[exp(sim(q, k+)/τ) / Σ exp(sim(q, ki)/τ)]
```
Use case: self-supervised learning, dense retrieval, representation learning. Trains embeddings by contrasting positive and negative pairs.

**9. Triplet Loss**
```
L = max(0, d(a, p) - d(a, n) + margin)
```
Use case: metric learning, face recognition, image similarity. Ensures anchor `a` is closer to positive `p` than negative `n` by at least `margin`. Hard triplet mining is critical for efficient training.

**10. KL Divergence**
```
KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
```
Use case: variational autoencoders (regularization term), knowledge distillation (matching student to teacher distribution), A/B test Bayesian analysis. Measures how distribution P diverges from reference distribution Q. Not symmetric: KL(P||Q) ≠ KL(Q||P).

### The 8 Most Common Serving Architectures

**1. Precomputed Batch Serving**
- Pattern: Run model offline (nightly), store results in key-value store, serve from cache at query time.
- Latency: ~1ms (cache lookup).
- Freshness: stale by up to 24 hours.
- Use case: non-personalized rankings, cold-start fallback, daily digest recommendations.

**2. Two-Tower Real-Time Retrieval + Ranking**
- Pattern: Precompute item embeddings, compute user embedding in real time, ANN search for candidates, rank with heavier model.
- Latency: 50-150ms end to end.
- Freshness: user embedding is fresh; item embeddings updated on a schedule.
- Use case: homepage recommendation, search ranking.

**3. Streaming Online Inference**
- Pattern: Model receives streaming event data (Kafka/Kinesis), updates predictions continuously.
- Latency: depends on event processing lag.
- Use case: fraud detection, real-time bidding, anomaly detection.

**4. Edge/On-Device Inference**
- Pattern: Model runs on user device (phone, browser, IoT device) using TFLite, CoreML, or ONNX.
- Latency: <10ms.
- Privacy: data never leaves device.
- Use case: keyboard next-word prediction, on-device wake word detection, image classification for AR.

**5. Cascade / Multi-Stage Pipeline**
- Pattern: Funnel architecture. Stage 1 retrieves O(1000) candidates, Stage 2 ranks to O(100), Stage 3 re-ranks to O(10).
- Each stage uses progressively more features and compute.
- Use case: e-commerce search, video recommendation, ad serving.

**6. Ensemble Serving**
- Pattern: Multiple models run in parallel; outputs are aggregated (average, weighted average, learned combiner).
- More robust than single model.
- Latency cost: parallel inference, not sequential.
- Use case: A/B testing with gradual rollout, domain expert ensembles, multi-modal fusion.

**7. Model-as-Microservice (REST/gRPC)**
- Pattern: Model exposed as a microservice via REST or gRPC API. Called by application servers.
- Standard pattern for team-owned models.
- Use case: virtually any ML model in a service-oriented architecture.

**8. Embedding-as-a-Service**
- Pattern: Model produces embeddings stored in a vector database. Downstream systems query the vector DB for similarity search.
- Decouples embedding model from retrieval logic.
- Use case: semantic search, image similarity, document retrieval.

### Key Numbers to Memorize

These are the "back of the envelope" numbers you need to reason about system scale:

**Scale anchors:**
- 1M users, 100 RPS: small-scale system
- 10M users, 1K RPS: medium-scale system
- 100M users, 10K RPS: large-scale system
- 1B users, 100K RPS: hyperscale

**Latency anchors:**
- Memory access (RAM): ~100ns
- SSD access: ~100μs
- Network round trip (same datacenter): ~1ms
- Network round trip (cross-region): ~100ms
- ANN search (FAISS, 1M vectors, 128 dims): ~5ms
- BERT inference (single sentence, GPU): ~10ms
- Transformer generation (100 tokens, A100 GPU): ~200ms

**Storage anchors:**
- 1 billion user-item interaction pairs: ~32GB (at 32 bytes per record)
- Embedding matrix (100M items × 128 dims, float32): ~50GB
- BERT-base parameters: 110M → ~440MB at FP32, ~110MB at INT8
- LLaMA-7B parameters: 7B → ~28GB at FP32, ~7GB at INT8

**A/B testing anchors:**
- Minimum detectable effect for engagement metrics: typically 0.5–1% relative lift
- Minimum experiment duration for novelty effect wash-out: 2 weeks
- Typical A/B test size for a 1% MDE at 80% power on a 10% baseline metric: ~1M users per arm

**Model training anchors:**
- BERT fine-tuning on typical NLP task: ~hours on 1 A100
- Two-tower model training on 100M pairs: ~hours on 4-8 A100s
- GPT-3 style pretraining from scratch: months on thousands of A100s
- LightGBM on 100M tabular rows: ~minutes on a single machine

---

## Section 11: Rapid-Fire Cheat Sheet

Use this section for final review before an interview. Each entry is a single concept compressed to its most essential form.

**Requirements → always cover:** business objective, scale, latency, data availability, interaction types, constraints.

**ML task formulation → always specify:** input space, output space, task type (classification/regression/ranking), proxy metric and its gap from business objective.

**Label construction → always address:** positive definition, negative sampling strategy, temporal split, position bias correction (IPS).

**Feature engineering → always address:** missing value treatment (imputation + indicator), scaling (normalization/standardization/log), encoding (one-hot for low cardinality, embedding for high cardinality), temporal leakage.

**Model progression → always start with:** non-ML baseline → logistic regression → gradient boosted trees → deep learning → specialized architecture. Justify each step up.

**Offline metrics by task:**
- Binary classification: AUC-ROC, PR-AUC, F1
- Regression: MSE, MAE, RMSE
- Ranking: nDCG, MRR, MAP

**Online metrics by domain:**
- Recommendation: CTR, watch time, retention
- Search: task completion, SERP CTR, satisfaction surveys
- Ads: CTR, CVR, RPM, ROAS
- Safety: FPR, recall on violations, escalation rate

**Serving → always address:** batch vs. online, two-stage retrieval-ranking, ANN library (FAISS/ScaNN/HNSW), latency budget decomposition, model compression (distillation/pruning/quantization), fallback/degradation strategy.

**A/B testing → always address:** power analysis (state the formula), minimum detectable effect, experiment duration, common pitfalls (peeking, network effects, novelty effect).

**Monitoring → always address:** input distribution shift, output distribution shift, data quality checks, operational metrics (latency/throughput/error rate), scheduled retraining trigger.

**Class imbalance → options:** downsampling negatives, upsampling positives, class-weighted loss, focal loss. State the trade-offs.

**Cold start → options:** content-based fallback, popularity ranking, knowledge transfer from similar users/items, exploration policy.

**Position bias → solution:** IPS correction: `L_IPS = Σ (L_i / e_i)`.

**Feedback loop → solution:** explore-exploit policy (epsilon-greedy, Thompson sampling, UCB), counterfactual estimation, periodic randomization.

**Calibration → solution:** Platt scaling: `P(y=1|f(x)) = 1 / (1 + exp(Af(x) + B))`.

**What separates levels in one sentence each:**
- No Hire → Weak Hire: ask clarifying questions.
- Weak Hire → Hire: integrate answers into a coherent design with explicit trade-offs.
- Hire → Strong Hire: surface organizational implications, platform thinking, and failure modes proactively.
- Strong Hire → "Exceptional": reframe the problem in a way that reveals something the interviewer hadn't considered.

---

## Closing Notes

The purpose of this guide is to make the 8-step framework automatic, so that your cognitive bandwidth during the interview is freed up for the thing that actually matters: high-quality reasoning about the specific problem in front of you.

A checklist that you mechanically execute is a Weak Hire. A framework that you use as a lens to generate insight is a Hire. A conversation that you structure and redirect in ways that reveal deeper problems is a Strong Hire.

Before every practice session, state this goal explicitly: "I am not trying to complete the checklist. I am trying to build a coherent argument for a system that solves a specific business problem under specific constraints, and to surface the most important trade-offs and failure modes."

The difference between a good ML practitioner and a great one is not the depth of their knowledge about any single technique. It is the quality of judgment they exercise when the right technique is not obvious — which is true of every real problem that matters.

Study this guide. Practice the domain modules. Score yourself honestly. Good luck.

---

*End of Staff/Principal ML Design Interview Preparation Guide*
