# Event Recommendation System — Staff/Principal Interview Guide

---

## How to Use This Guide

This guide is structured as a realistic mock interview for a Staff or Principal ML Engineer role at a company like Eventbrite, Meetup, or any large consumer platform with event discovery. The problem tests your ability to go beyond "I would use a neural network" and demonstrate deep, production-grade thinking across the full ML lifecycle.

**How to run the interview:**
- Time budget: 45 minutes total. Each section has a suggested duration.
- Read the interviewer prompt aloud, then let the candidate answer before consulting the model answers.
- Use the 4-level rubric (No Hire → Hire → Strong Hire) to calibrate. Do not expect candidates to hit every point — use the rubric to assess depth and breadth of thinking.
- Push with follow-up questions from the "Probing Questions" list at the end of each section.
- Debrief using the appendix formulas to verify quantitative claims.

**What separates Staff from Senior at this level:**
A Senior engineer gives correct answers. A Staff engineer gives correct answers AND explains the *tradeoffs*, *failure modes*, and *business implications* of each decision. A Principal engineer additionally thinks about platform abstractions, organizational leverage, and long-term system evolution. Watch for candidates who spontaneously raise concerns you did not ask about — that is the strongest signal.

**Scoring summary:**

| Level | Description |
|---|---|
| No Hire | Missing fundamentals; cannot frame ML task; vague on features or evaluation |
| Weak Hire | Correct answers but shallow; follows prompts without depth; misses key tradeoffs |
| Hire | Correct and complete; explains tradeoffs; proactively raises failure modes |
| Strong Hire | All of the above + platform thinking, organizational leverage, principled prioritization under constraints |

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

> "Design an event recommendation system for a platform like Eventbrite. Users can browse events, register for tickets, and invite friends. The business goal is to maximize ticket sales. Walk me through how you'd approach this problem."

Before the candidate begins designing, listen for whether they ask clarifying questions. This is the first signal: Staff engineers do not start building until they understand the problem space.

### The 6 Clarifying Dimensions

A strong candidate should probe at least 4 of these 6 dimensions unprompted:

**1. Scale**
- How many events are on the platform at any given time? (~1 million total, ~100K active/upcoming)
- How many users? (~10 million daily active users)
- How many recommendation requests per second? (~10K QPS at peak)

**2. User context**
- What user data is available at request time? (current location via GPS/IP)
- Is the user logged in or anonymous?
- Do we have historical behavior (past registrations, clicks, searches)?

**3. Business objective**
- Is the goal raw registrations, paid ticket revenue, or something else?
- Are we optimizing for the user, the organizer, or the platform?
- Are there any organizer SLAs or fairness constraints?

**4. Cold start**
- What fraction of users are new vs. returning?
- What fraction of events are new vs. established?

**5. Freshness and time sensitivity**
- Events expire — a concert tomorrow is different from one next month.
- Should we boost events that are about to happen?

**6. Latency and serving**
- What is the acceptable p99 latency for the recommendation endpoint?
- Is this a real-time ranked feed or a pre-computed batch list?

---

### Model Answers by Level

#### No Hire Answer
"I would recommend events based on the user's location and past interests. I'd use collaborative filtering to find similar users and recommend what they liked."

*Why this fails:* No clarifying questions. No discussion of business objective precision. Jumps to a specific technique without establishing requirements. Collaborative filtering alone ignores the critical time dimension of events.

---

#### Weak Hire Answer
"Before I start, I want to clarify a few things. How many users and events do we have? Is this a real-time feed or a batch job? 

Given the scale, I'd frame this as a ranking problem. We have user data, event data, and interaction data. I'd build features from all three and train a model to predict whether a user will register for an event. For the model, I'd start with logistic regression for a baseline and then move to something more powerful like XGBoost or a neural network.

The business metric would be registrations per recommendation session, and I'd evaluate offline with some held-out test set."

*Why this is weak hire:* Asks some clarifying questions (good), gets the ML framing right (good), but the answers lack specificity. "Some held-out test set" is not an evaluation strategy. No mention of the time sensitivity of events, cold start, or class imbalance.

---

#### Hire Answer
"Let me ask a few clarifying questions before I start. First, what does 'maximize ticket sales' mean precisely — are we optimizing for total registration count, paid revenue, or something like session-to-purchase conversion? The answer changes the label definition.

Second, what user signals are available at inference time? I'm assuming we have location (GPS or IP), but do we have browsing history in the current session, or only historical registered events?

Third, scale: roughly how many events are live at any given time, and what's our QPS for recommendations?

Fourth, how do we handle events that expire? A concert tomorrow and a conference three months from now have very different urgency dynamics.

Given what I know: I'd frame this as a Learning to Rank problem, specifically pointwise binary classification where the label is 'did the user register for this event?' The input is a (user, event) pair and the output is a probability of registration, which we then rank to produce the final ordered list.

The reason I choose pointwise LTR over pairwise or listwise is pragmatic: pointwise maps cleanly to existing binary classification infrastructure, has a well-understood loss function (binary cross-entropy), and scales to millions of (user, event) pairs without requiring paired samples during training. We can always upgrade to listwise later if we find the top-of-list quality is insufficient.

One thing I want to flag upfront: this is a heavily class-imbalanced problem. Registrations are rare events — maybe 1-2% of impressions convert. This will require either focal loss, weighted sampling, or careful threshold calibration at serving time."

*Why this is a hire:* Asks precise questions, establishes the ML framing with justification, proactively raises the class imbalance problem without being prompted, explains the tradeoff between LTR formulations.

---

#### Strong Hire Answer
*(Includes everything in the Hire answer, plus:)*

"I also want to think about what 'success' means from a two-sided marketplace perspective. Pure registration optimization could create a feedback loop where we only recommend popular events, starving new organizers of visibility. This is a supply-side health problem — if organizers don't get traction on early events, they stop using the platform, which reduces supply diversity for users. So I'd want to add an organizer fairness constraint to the objective from the beginning, not as an afterthought.

Additionally, the business metric 'ticket sales' conflates free events and paid events. I would separate the objective: for free events, optimize for registrations; for paid events, optimize for GMV (price × registrations). The recommendation system may need to be aware of price as a proxy for organizer monetization.

Finally, I'd confirm: is this a cold-start-heavy problem? If a large fraction of users have fewer than 5 past interactions, pure collaborative filtering will fail, and we need a strong content-based fallback. I'd design the feature schema to degrade gracefully when history is sparse."

*Why this is strong hire:* Raises the two-sided marketplace supply-side health problem without prompting. Distinguishes free vs. paid events in the objective. Designs for degradation under cold start. Demonstrates systems thinking beyond the immediate ML task.

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

> "Walk me through exactly how you'd frame this as an ML problem. What is the training data unit? What is the label? What is the objective function?"

### Ground Truth

**ML Task:** Learning to Rank — Pointwise binary classification.

**Training data unit:** A (user, event) pair observed at impression time.

**Label:** 1 if the user registered for the event following the impression, 0 otherwise.

**Objective:** Minimize binary cross-entropy (or focal loss to handle class imbalance).

**Why pointwise over pairwise/listwise:**
- Pointwise: Each (user, event) pair is scored independently. Simple, scalable, directly maps to probability calibration.
- Pairwise: Requires (user, event_A, event_B) triples where one was preferred. Better for learning relative order but 2× data construction cost.
- Listwise: Optimizes the full list ordering (e.g., NDCG directly). Best quality but expensive.
- For a production system at this scale, pointwise is the correct starting point.

---

### Model Answers by Level

#### No Hire Answer
"I'd use a recommendation system. The label would be whether the user clicked on the event."

*Why this fails:* Clicks (impressions) are not the target — registrations are. Conflating CTR with conversion is a fundamental mistake. "Recommendation system" is not an ML problem framing.

---

#### Weak Hire Answer
"I'd frame it as a binary classification problem. For each user-event pair, I predict whether the user will register. The label is 1 for registration and 0 for no registration. I'd train on historical impression logs with those labels."

*Why this is weak hire:* Correct framing but no discussion of class imbalance, label construction challenges (position bias — events shown first get more registrations regardless of quality), or the difference between random negatives and hard negatives.

---

#### Hire Answer
"I'd frame this as pointwise Learning to Rank — binary classification over (user, event) pairs. The label is 1 if a registration occurred after an impression, 0 otherwise.

A few important nuances in label construction:

First, position bias: events shown at the top of the feed are registered for more often simply because of visibility, not because they're better matches. If I train naively on impression logs, the model learns to score high-position events higher, which is a self-reinforcing loop. I'd correct for this using Inverse Propensity Scoring (IPS) to weight training examples by the inverse probability of being shown at that position.

Second, class imbalance: registration rate is probably around 1-2% of impressions. I'd use focal loss rather than standard binary cross-entropy:

$$\text{FL}(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$$

where $p_t$ is the model's predicted probability for the true class, $\alpha$ balances positive/negative class weight (typically 0.25), and $\gamma$ controls focus on hard examples (typically 2.0). When $\gamma = 0$, this reduces to standard cross-entropy.

Third, impression vs. non-impression negatives: I should only use events that were actually shown (impressed) to the user as negatives, not all events in the catalog. Using all catalog events as negatives would bias the model against popular events that simply happen to have been impressed many times."

*Why this is a hire:* Correct framing with nuance. Raises position bias, proposes IPS, gives the focal loss formula, distinguishes impression negatives from catalog negatives.

---

#### Strong Hire Answer
*(Includes everything in the Hire answer, plus:)*

"I want to also think about the temporal structure of the label. Events happen at a specific time, and a registration 30 days before an event has different behavioral semantics than a registration 30 minutes before. I'd want to include 'time until event at impression time' as a feature, and potentially segment evaluation by registration lead time.

Also, for the training window: I'd use a sliding window of recent data (e.g., 90 days) rather than all historical data, because user preferences and event landscapes shift over time. Events from 2 years ago trained the model on a pre-pandemic preference distribution that may not reflect current behavior.

One more framing decision: should this be a single model for all users, or should we have separate models for different user segments? A user who has registered for 50 events has much richer history than a new user. I'd start with a single model with user history features that gracefully degrade to defaults when history is sparse, rather than maintaining separate models — simpler operationally and often better in practice because the shared model can transfer learning across user segments."

*Why this is strong hire:* Adds temporal label semantics, training window choice justification, and a principled argument for single vs. multi-model architecture.

---

## Section 3: Data & Feature Engineering (8 min)

### Interviewer Prompt

> "Walk me through the features you'd use. We have four tables: User (user_id, username, age, gender, city), Event (event_id, host_user_id, category, subcategory, description, price), Friendship (user_id_1, user_id_2, timestamp), and Interactions (impressions, registrations, invites). What features do you build, and how?"

### Ground Truth Feature Schema

#### Location Features
1. **Distance (bucketized):** Raw distance in miles between user's current location and event venue. Bucketized into: <0.5 mi, 0.5-1 mi, 1-5 mi, 5-10 mi, 10-25 mi, 25-50 mi, 50+ mi. Bucketization avoids the model learning spurious precision and makes the feature robust to GPS noise.
2. **Same city (binary):** 1 if event is in user's home city.
3. **Same country (binary):** 1 if event is in user's home country.
4. **Estimated travel time:** Google Maps / transit API call, bucketized similarly.
5. **Walk/bike/transit score:** From a service like Walk Score API. Captures whether the venue is accessible without a car, which matters for urban vs. suburban users.
6. **Historical location affinity:** Cosine similarity between the event's geo-cluster and the centroid of the user's past attended event locations. Captures "does this user typically attend events in this neighborhood?"

#### Time Features
7. **Time until event (bucketized):** <1 hour, 1-24 hours, 1-7 days, 7-30 days, 30+ days. Critical for urgency modeling.
8. **Day-of-week preference similarity:** Compare event's day of week to the histogram of the user's past event days. E.g., if a user always attends weekend events and this event is on a Tuesday, score is low.
9. **Hour-of-day preference similarity:** Same idea for start time.
10. **Estimated travel time vs. available time:** Does the user have enough time to get there?

#### Social Features
11. **Number of friends registered:** Count of user's friends who have registered for this event (from Friendship × Interactions join).
12. **Number of friend invitations:** Count of invitations received from friends for this event.
13. **Host is friend (binary):** 1 if the event host (host_user_id) is in the user's friend graph.
14. **Event popularity — total registrations:** Log-transformed count of all registrations.
15. **Registration-to-impression ratio:** registrations / impressions. Proxy for event quality/appeal. Requires smoothing for new events: $\text{smoothed\_ratio} = \frac{\text{registrations} + \alpha}{\text{impressions} + \beta}$ where $\alpha, \beta$ are prior counts (e.g., 1 and 10).

#### User Features
16. **Age (bucketized):** 18-24, 25-34, 35-44, 45-54, 55+. Bucketized to avoid treating age as a continuous linear variable.
17. **Gender (one-hot):** If gender is binary in the schema: [male, female, unspecified].
18. **User activity level:** Number of events registered in past 30/90 days. Proxy for engagement level.

#### Event Features
19. **Price bucket:** Free, $1-$24, $25-$99, $100-$499, $500+. Bucketized because the relationship between price and registration probability is highly nonlinear.
20. **Category/subcategory (one-hot or embedding):** Music, Sports, Arts, etc.
21. **Description embedding similarity:** Embed event description using a sentence transformer (e.g., SBERT). Compute cosine similarity to the centroid of embeddings of user's past registered events. Captures semantic interest alignment beyond category labels.

$$\text{sim}(\mathbf{u}, \mathbf{e}) = \frac{\mathbf{u} \cdot \mathbf{e}}{||\mathbf{u}|| \cdot ||\mathbf{e}||}$$

where $\mathbf{u}$ is the mean embedding of user's past events and $\mathbf{e}$ is the current event's embedding.

---

### Model Answers by Level

#### No Hire Answer
"I'd use features like user age, event category, event location, and event price."

*Why this fails:* Correct instinct but zero depth. No discussion of feature engineering (bucketization, embeddings, ratio smoothing), no mention of social signals, no mention of how to handle sparse history.

---

#### Weak Hire Answer
"I'd build three groups of features: user features (age, gender, historical preferences), event features (category, price, description), and contextual features (distance to event, time until event). I'd also add social features like how many friends have registered.

For the description, I'd use TF-IDF or a simple embedding to capture topic similarity to the user's interests."

*Why this is weak hire:* Covers the right categories but lacks engineering specifics. No discussion of bucketization, no formula for smoothed ratio, no mention of position bias correction, no mention of how to compute "historical preference similarity" for location or time.

---

#### Hire Answer
"I'd organize features into five groups. Let me walk through the most interesting ones with implementation details.

**Location features** are critical because events are inherently local. Raw GPS distance is noisy and has a nonlinear relationship with willingness to attend, so I'd bucketize: <0.5 mi, 0.5-1 mi, 1-5 mi, 5-25 mi, 25+ mi. Beyond raw distance, I'd add a 'historical location affinity' feature: take the geo-cluster centroids of all events the user has attended historically, and compute the cosine similarity between that distribution and the current event's location. This captures whether a user typically attends events in this part of the city, which raw distance from home doesn't capture. I'd also add walk/transit scores from a third-party API — a user without a car in a city cares deeply about whether an event is walkable.

**Social features** are some of the highest-signal features in an event context. The number of friends who have registered for an event is a strong proxy for social relevance. I'd also distinguish between organic registrations (friends registered on their own) and friend invitations (friends explicitly invited me), since the latter is a stronger signal. I'd compute: (a) friend registration count, (b) friend invitation count, (c) whether the host is a direct friend (binary), and (d) event-level popularity as a registration/impression ratio with Laplace smoothing.

For the smoothed ratio:
$$r_\text{smooth} = \frac{N_\text{reg} + \alpha}{N_\text{imp} + \beta}$$
where I'd set $\alpha = 1, \beta = 10$ as a weak prior based on platform average CTR (~10%). This prevents new events with 1 impression and 1 registration from getting a 100% ratio.

**Description embedding similarity** is important for capturing user interest beyond category labels. I'd embed event descriptions with a pretrained sentence encoder (e.g., SBERT/all-MiniLM-L6-v2) and store user interest profiles as the mean embedding of their past 50 registered events. At inference time, compute cosine similarity between the current event's embedding and the user's interest vector. This is stored in a feature store and updated in near-real-time as new registrations occur.

One thing I want to flag: for new users with fewer than 5 past events, the interest embedding is unreliable. I'd fall back to category-level one-hot features for sparse users, and use the embedding only when history is sufficient."

*Why this is a hire:* Detailed engineering specifics, Laplace smoothing formula, discussion of new user fallback, differentiation of friend registrations vs. invitations, mention of feature store for the embedding similarity feature.

---

#### Strong Hire Answer
*(Includes everything in the Hire answer, plus:)*

"I want to raise two cross-cutting concerns about this feature set.

First, **training-serving skew.** Several of these features involve joins at training time (e.g., friend registrations requires joining Friendship × Interactions) and then need to be recomputed in real-time at serving. If the training pipeline computes these features differently from the serving pipeline — different time windows, different join logic, different null handling — the model will underperform in production. My mitigation: use a shared feature store with versioned feature definitions. Training reads features from the feature store using the same code path as serving, computed at the time of the training impression. This is non-trivial to implement but essential for correctness.

Second, **feature leakage.** For time-based features like 'time until event,' I need to compute this as of the impression timestamp, not the current time. If I accidentally compute it at training time relative to today, the model learns the wrong thing. I'd implement strict point-in-time correctness in the training data pipeline.

Third, **feature importance monitoring in production.** I'd log feature distributions at serving time and compare them to training distributions using Population Stability Index (PSI) or Jensen-Shannon divergence. If a feature's distribution drifts (e.g., 'time until event' shifts because we're in a seasonally slow period), I can catch it before it impacts recommendations.

On feature prioritization: if I have to launch with a limited feature set, I'd start with location (distance bucket), time until event, category match, and friend registration count — these four features cover the majority of the variance in user utility. Everything else is incremental."

*Why this is strong hire:* Raises training-serving skew with a specific mitigation (feature store with shared code path), point-in-time correctness, feature drift monitoring with specific metrics (PSI, JS divergence), and principled feature prioritization for MVP.

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

> "What model would you use? Walk me through the options you considered and why you landed on your recommendation. What happens if we want to update the model continuously as new events and registrations come in?"

### Ground Truth — Model Comparison

| Model | Pros | Cons | Verdict |
|---|---|---|---|
| Logistic Regression | Interpretable, fast, calibrated | Linear decision boundary, no feature interactions | Baseline |
| Decision Trees | Fast, handles non-linearity | Overfits badly, unstable | Not recommended alone |
| GBDT / XGBoost | Excellent on structured data, robust | Batch training only, slow to update | Strong baseline |
| Neural Network | Continual learning, handles embeddings, powerful | Needs data volume, harder to debug | Production target |

**Recommended path:** XGBoost baseline → Neural Network for production, with online learning.

---

### Model Answers by Level

#### No Hire Answer
"I'd use a neural network because it's the most powerful. I'd train it on user and event features and optimize for click-through rate."

*Why this fails:* No discussion of tradeoffs, wrong target metric (CTR not registrations), no mention of training strategy or baseline.

---

#### Weak Hire Answer
"I'd start with logistic regression as a baseline for interpretability, then move to XGBoost because it works well on structured tabular data. If performance isn't good enough, I'd upgrade to a neural network. For continuous updates, I'd retrain the model periodically, maybe daily."

*Why this is weak hire:* Correct trajectory but shallow. No depth on why XGBoost beats LR on this problem, no discussion of XGBoost's limitations for streaming, no architecture details for the neural network, no discussion of what "periodic retraining" means operationally.

---

#### Hire Answer
"Let me walk through four model families and then make a recommendation.

**Logistic Regression:** This is my mandatory first baseline. It's interpretable, fast to train and serve (single matrix multiply at inference), naturally calibrated in terms of probability outputs, and easy to debug. Its fundamental limitation is that it can only learn linear decision boundaries — it cannot capture the interaction between 'user is in the same city' AND 'event is tomorrow' that makes an event highly relevant. We can add hand-crafted interaction features, but this doesn't scale. LR is my day-1 baseline to establish a production floor.

**Decision Trees:** Fast and non-linear, but deep trees overfit badly on sparse features (like one-hot categories), and shallow trees underfit. Not competitive with ensembles. I'd skip standalone decision trees.

**GBDT / XGBoost:** This is the workhorse for structured tabular data. XGBoost handles mixed feature types (continuous, categorical, binary) without much preprocessing, is robust to outliers, captures feature interactions through tree splits, and has excellent off-the-shelf calibration. I'd expect a significant lift over LR on this feature set because the value of many features is highly interaction-dependent (distance matters much more if the event is tonight than if it's next month). 

The critical limitation of XGBoost for this problem: it requires batch training. We cannot update a gradient-boosted tree incrementally when a new registration comes in. In an event recommendation context, this is a real problem because: (1) new events are created constantly, (2) early registration signals for new events are highly predictive but ephemeral, and (3) user preferences can shift within a session. If we retrain daily, we miss the early-life signals for events created today.

**Neural Network:** The production target. Key advantages:
1. Supports online/continual learning via SGD — we can update model weights with each new batch of interactions, capturing same-day signals.
2. Can jointly embed categorical features (category, subcategory, user_id) in a learned latent space, which generalizes better than one-hot encoding.
3. Can incorporate the description embedding as a dense input naturally.
4. Scales to very large feature vocabularies.

My recommended architecture: a two-tower model with a shallow MLP on top.
- User tower: dense embedding from user_id + user feature vector (age, gender, activity level, interest embedding)
- Event tower: dense embedding from event_id + event feature vector (price, category, description embedding)
- Merge layer: concatenate user and event towers + contextual features (distance, time until event, social signals)
- Output: single sigmoid unit predicting P(registration)

Loss: Focal loss with $\alpha = 0.25$, $\gamma = 2.0$:

$$\text{FL}(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$$

**Recommendation:** Launch with XGBoost for speed of iteration and explainability. Run it in parallel with an NN after ~3 months of production data collection. Graduate to NN when it shows statistically significant lift in A/B test and the ML infrastructure team has built the online learning pipeline."

*Why this is a hire:* Covers all four models with specific tradeoffs, explains XGBoost's batch training limitation in context, proposes a concrete NN architecture, gives the focal loss formula, provides a sequenced deployment roadmap.

---

#### Strong Hire Answer
*(Includes everything in the Hire answer, plus:)*

"I want to go deeper on the XGBoost vs. NN decision for streaming, because I think this is the crux of the architecture choice.

XGBoost's batch-only constraint isn't just a performance issue — it creates a structural staleness problem. An event organizer posts a new concert tonight. It starts getting impressions. Early registrations in the first hour are a strong signal of virality. With daily batch retraining, this signal doesn't propagate into the model until tomorrow, by which time the event may already be sold out or over. The model is perpetually behind on new events.

There are two mitigation strategies that let you keep XGBoost while partially addressing this:
1. **Feature store freshness:** Even if the model weights are stale, you can have real-time features like 'registrations in last 1 hour' flowing through the feature store. The old model weights will at least see the updated feature values at serving time. This is not as good as online learning but is much better than fully stale features.
2. **Hourly retraining:** Train XGBoost every hour on a rolling window of the last 7 days. Computationally expensive (~100GB dataset, 30 min training time) but feasible on a cluster. This is the pragmatic bridge while the NN online learning pipeline is being built.

For the neural network's online learning pipeline, I'd use mini-batch SGD with a learning rate schedule that decays for old examples:
- New registrations: standard learning rate $\eta = 0.001$
- Reweighted older examples in each mini-batch to prevent catastrophic forgetting

One risk of online learning I want to call out: **model drift from distribution shift.** If we update the model continuously based on user interactions, and user interactions are shaped by the model's recommendations, we get a feedback loop. The model can converge to a local optimum (e.g., only recommending concerts because that category got the most engagement, which caused users to only see concerts, which caused concerts to dominate the training data). I'd mitigate this with:
1. Epsilon-greedy exploration: with probability $\epsilon = 0.05$, serve a random recommendation to gather unbiased signal.
2. Counterfactual logging: log model scores for all candidates, not just those shown.
3. Periodic retraining from scratch to reset to an unbiased starting point."

*Why this is strong hire:* Explains the structural staleness problem with a concrete scenario, proposes two XGBoost mitigations, discusses the feedback loop problem in online learning with specific mitigations (epsilon-greedy, counterfactual logging), demonstrates awareness of catastrophic forgetting.

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

> "How do you evaluate this system? Walk me through both offline and online metrics. Why would you choose one metric over another?"

### Ground Truth

**Offline metric (primary):** Mean Average Precision (mAP)

**Online metrics:**
- CTR (clicks / impressions) — leading indicator, easy to measure
- Conversion rate (registrations / impressions) — primary business metric
- Bookmark / save rate — intent signal
- Revenue lift (GMV change in A/B test) — ultimate business metric

**Why mAP over AUC-ROC:**
- mAP is order-sensitive: it rewards the model for putting relevant items higher in the list. AUC-ROC treats the problem as a binary classification without regard to rank order.
- For a recommendation feed, rank order matters enormously. A model that puts the most relevant event at position 1 vs. position 5 has very different user experience implications.
- mAP directly optimizes for the ranked output we care about.

**Why conversion rate over CTR:**
- CTR measures whether users clicked on a recommendation. But users can click out of curiosity and not register.
- Conversion rate measures whether the recommendation actually drove the business outcome (registration).
- A model that learns to surface clickbait events (great thumbnail, misleading title) will have high CTR but low conversion rate.
- Optimizing for CTR creates misalignment between the metric and the business goal.

---

### Model Answers by Level

#### No Hire Answer
"I'd use accuracy to evaluate the model. If the model is accurate, it's working."

*Why this fails:* Accuracy is meaningless for a class-imbalanced problem. A model that always predicts 0 (no registration) achieves 98-99% accuracy. No discussion of ranking metrics.

---

#### Weak Hire Answer
"I'd use AUC-ROC offline because it measures the model's ability to distinguish registrations from non-registrations. Online, I'd track CTR and registrations per session."

*Why this is weak hire:* AUC-ROC is a reasonable choice but not the best for ranking. Weak hire explains what CTR measures but doesn't articulate why it might be misleading vs. conversion rate.

---

#### Hire Answer
"For offline evaluation, my primary metric would be Mean Average Precision (mAP), not AUC-ROC. Let me explain why.

AUC-ROC measures the probability that a randomly chosen positive example is scored higher than a randomly chosen negative example. It's a good general-purpose discriminative metric. But for a ranking problem, we care specifically about how well the model orders items — does the most relevant event appear at position 1 or position 5?

mAP directly captures this. For each query (user), we compute Average Precision:
$$\text{AP} = \frac{1}{|R|} \sum_{k=1}^{n} P(k) \cdot \text{rel}(k)$$

where $|R|$ is the number of relevant items (registrations), $P(k)$ is the precision at position $k$, and $\text{rel}(k)$ is 1 if the item at position $k$ is relevant, 0 otherwise. mAP averages AP over all queries.

The key difference: if the user eventually registers for an event, but the model ranked that event at position 50, mAP penalizes this. AUC-ROC does not — it only cares whether the positive example was scored higher than a random negative, regardless of absolute rank.

For online metrics, I'd use conversion rate (registrations / impressions) as the primary metric, not CTR. CTR measures engagement with the recommendation, not the business outcome. A model optimized for CTR can learn to surface visually attractive events with misleading descriptions — the user clicks, lands on the page, realizes it's not what they wanted, and bounces. Conversion rate measures whether the recommendation actually drove registration, which is the business goal.

I'd also track:
- **Bookmark rate:** Users who save an event without registering immediately. This is intent signal that may convert later.
- **Revenue lift in A/B test:** For paid events, did the new model increase total GMV?
- **Organizer diversity:** Are we recommending events from a healthy diversity of organizers, or are we over-indexing on a few popular ones?

For the A/B test design: I'd run a holdback experiment with 90% treatment / 10% control, stratified by user cohort (new vs. returning users), for at least 2 weeks to capture weekly seasonality. Statistical significance at $p < 0.05$ with pre-registered primary metric (conversion rate)."

*Why this is a hire:* Gives the mAP formula with explanation, articulates clearly why conversion rate beats CTR with a concrete failure mode example, includes organizer diversity as a metric (platform health), and specifies A/B test design details.

---

#### Strong Hire Answer
*(Includes everything in the Hire answer, plus:)*

"I want to add a nuance on mAP for this specific domain. Standard mAP weights all relevant items equally. But for event recommendations, a registration for an event happening tomorrow is much more valuable than a registration for an event in 3 months — the former represents immediate GMV. I'd consider a time-discounted variant of AP where the reward for a correct ranking is weighted by proximity to the event date.

Also, offline mAP computed on historical data has a survivorship bias problem: we only observe registrations for events that were shown to users. Events that the model might have ranked highly but were never shown have no label. This is the standard partial observability problem in offline ranking evaluation. To mitigate this, I'd periodically inject 'evaluation events' — events that are forced into a random sample of recommendation feeds purely for evaluation purposes, not because the model ranked them highly. This gives us unbiased signal on held-out events.

One more thing: I'd track mAP separately for different user cohorts — cold-start users (<5 registrations historically) vs. warm users. The model should be evaluated on its weakest segments, not just aggregate performance, because the cold-start segment is often the largest and most important for growth."

*Why this is strong hire:* Proposes time-discounted mAP for business alignment, identifies survivorship bias in offline evaluation with a specific mitigation, and advocates for per-cohort evaluation to surface cold-start performance.

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

> "Walk me through the serving architecture. How does a recommendation request flow through the system? How do you keep the model fresh?"

### Ground Truth — Serving Architecture

**Two-stage retrieval + ranking pipeline:**

1. **Event Filtering (Candidate Generation):** Rule-based and lightweight ML. Reduce 1M events → ~500 candidates.
   - Hard filters: events in the future, in accessible geographic area, matching user's language preference
   - Soft filters: ANN similarity search if embedding-based retrieval is in use
   
2. **Ranking (ML model):** Score all ~500 candidates with the full feature set. Return top-K.

**Online Learning Pipeline:**

1. New registrations/interactions stream into a message queue (Kafka)
2. Feature computation service enriches events with context features
3. Model update service runs mini-batch SGD on the enriched examples
4. New model weights deployed to serving layer (canary → full rollout)
5. Model evaluation gate: automated rollback if online metrics degrade

**Feature Store Architecture:**
- Static features (user demographics, event metadata): computed offline, stored in Redis/DynamoDB, low-latency reads
- Dynamic features (friend registrations, event popularity): computed in near-real-time via streaming pipeline (Spark Streaming / Flink)
- Request-time features (distance, time until event): computed at serving time from request context

---

### Model Answers by Level

#### No Hire Answer
"I'd have a backend service that takes a user ID, looks up their preferences, and returns a list of recommended events."

*Why this fails:* No discussion of the two-stage pipeline, feature store, latency constraints, or model freshness.

---

#### Weak Hire Answer
"I'd use a two-stage pipeline: first filter down to a manageable set of candidate events using fast rules, then rank them with the ML model. The ML model would be served via a model serving layer like TensorFlow Serving or TorchServe. For freshness, I'd retrain the model daily on new data."

*Why this is weak hire:* Correct two-stage structure but shallow. No details on how the filter works, how features are computed at serving time, latency budget, or the tradeoffs of daily vs. continuous training.

---

#### Hire Answer
"The serving pipeline has three main components that I'll walk through in order of request flow.

**Stage 1: Event Filtering (Candidate Generation)**

When a user opens the app, we have ~1 million events on the platform. Running the full ML model on all 1 million events at every request is computationally infeasible at 10K QPS. The first stage reduces this to a manageable candidate set (~300-500 events) using fast, rule-based filters:
- Events in the future (hard filter)
- Events within a configurable distance radius (100 miles default, configurable)
- Events not already registered for by this user
- Events matching user's language/region preference

This stage should complete in <10ms. It runs on the catalog database with appropriate spatial indexes (PostGIS or Elasticsearch geo-queries).

**Stage 2: Feature Assembly**

For the ~500 candidates, we need to assemble the full feature vector in real-time. This is where the feature store comes in:
- **Static features** (user demographics, event category, price): pre-computed, stored in Redis. Read latency: ~1ms for a batch of 500 events.
- **Dynamic features** (friend registrations, event popularity): updated by a streaming pipeline (Kafka → Flink → Redis). Updated every 5 minutes. Read latency: ~2ms.
- **Request-time features** (distance, time until event): computed at serving time from the request's GPS coordinates and current timestamp. Latency: ~1ms.

Total feature assembly: ~5ms for 500 candidates.

**Stage 3: ML Scoring and Ranking**

The assembled features go into the ML model (XGBoost or NN). For XGBoost, scoring 500 examples takes ~5ms on CPU. For NN, it takes ~3ms on GPU. Return the top 20 ranked events.

Total end-to-end latency budget: ~20ms, well within a 100ms p99 target.

**Model Freshness — Online Learning Pipeline:**

For the NN, I'd implement a continuous training pipeline:
1. User interactions (registrations, clicks) stream into Kafka
2. A feature enrichment service reads the interaction event and assembles the training example (joining in features from the feature store at the time of the interaction)
3. A training service collects mini-batches (e.g., 1024 examples) and runs a gradient update
4. Updated weights are pushed to a model registry (MLflow) with automated evaluation
5. A canary deployment system serves the new model to 5% of traffic, monitors online metrics for 30 minutes, then promotes to 100% if metrics are stable

**Training-serving skew mitigation:**
The biggest risk in this pipeline is computing features differently at training time vs. serving time. My mitigation: the feature enrichment service in the training pipeline is the *same code* as the feature assembly service in the serving pipeline, reading from the *same feature store*. The training example's features are assembled at the time of the interaction event, using the feature store's historical snapshots. This ensures the feature computation logic is identical between training and serving."

*Why this is a hire:* Covers all three serving stages with latency numbers, explains the feature store architecture with three tiers, describes the online learning pipeline end-to-end, and articulates the training-serving skew mitigation clearly.

---

#### Strong Hire Answer
*(Includes everything in the Hire answer, plus:)*

"I want to raise two architectural concerns that are easy to miss.

**Shadow mode deployment for model upgrades:** When I ship a new model version, I don't want to immediately A/B test it — that exposes real users to a potentially worse model. Instead, I'd run the new model in shadow mode: it receives the same requests as the production model, generates scores, but those scores are not used for serving. We compare the shadow model's rankings to the production model's rankings and check for statistical differences. Only after shadow mode shows the new model is at least as good do we graduate to an A/B test.

**Feature freshness vs. latency tradeoff:** Dynamic features like 'friend registrations in the last hour' require a streaming pipeline. Every minute of lag in that pipeline means the model is making decisions with stale social signals. However, the streaming pipeline is expensive and operationally complex. The question I'd ask: what is the business value of knowing a friend registered 5 minutes ago vs. 1 hour ago? For events happening tomorrow or next week, 1-hour lag is probably fine. For events happening today, 5-minute lag matters. I'd architect the pipeline with configurable freshness SLAs per feature, so we can tune the tradeoff.

**Back-pressure and failure handling:** The online learning pipeline has a risk of overloading the training service during peak traffic. I'd add: (a) a rate limiter on the Kafka consumer so training processes at most N examples per second, (b) a dead letter queue for failed training examples so we don't silently drop data, and (c) a circuit breaker that pauses online learning if model quality metrics drop below a threshold, reverting to the last known-good checkpoint."

*Why this is strong hire:* Shadow mode deployment, feature freshness SLA per feature type, back-pressure handling in online learning — all production-grade concerns that are rarely discussed at the Senior level.

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Interviewer Prompt

> "What are the main ways this system can fail? Walk me through at least 4 failure modes and how you'd handle them."

### Ground Truth — 6 Failure Modes

#### 1. Time-Sensitive Event Starvation
**Problem:** An event happening in 6 hours has very few impressions and registrations, so the model scores it low due to low popularity features. But it may be exactly what the user wants (last-minute plans).
**Solution:** Add a "time urgency boost" rule: events within 24 hours get a multiplicative score boost in the ranking stage. This is a business rule, not a model feature, so it cannot be gamed.

#### 2. Popular Event Ticket Scalping / Artificial Inflation
**Problem:** A bad actor creates fake accounts and "registers" for events to make them appear popular, driving the model to recommend them to real users.
**Solution:** Anomaly detection on registration velocity (sudden spike in registrations from new accounts). Rate limiting and fraud scoring on registration events. Weight registrations from accounts > 30 days old more heavily than new accounts.

#### 3. Cold Start — New Organizer
**Problem:** A new organizer creates their first event. It has no impressions, no registrations, no social signals. The model has no signal to rank it.
**Solution:** 
- For organizer cold start: use category/subcategory + location + description embedding to compute a content-based score as a fallback.
- Introduce an "exploration budget" per recommendation session: 1-2 slots reserved for new organizer events that pass content-based quality filters.
- Incentivize organizers to provide rich descriptions and high-quality photos (which improve the embedding quality).

#### 4. Filter Bubble / Diversity Collapse
**Problem:** The model learns that a user always attends music events in their neighborhood, so it only shows music events in their neighborhood. The user never discovers events outside their normal pattern, reducing long-term engagement.
**Solution:** Diversity constraint in the ranking stage: enforce minimum category diversity in the final top-K (e.g., at most 50% of top 10 from the same category). Periodically inject "serendipity events" based on exploration policy.

#### 5. Seasonal Distribution Shift
**Problem:** The model is trained on data from summer (outdoor events, festivals). In winter, the distribution of events and user preferences shifts (indoor events, holidays). The model's features don't generalize.
**Solution:** Weight recent training data more heavily (exponential decay on sample weights). Monitor feature distribution shift with PSI. Consider seasonal models or fine-tuning on rolling windows.

#### 6. Position Bias in Training Data
**Problem:** The model is trained on impression logs, where events shown at position 1 have much higher registration rates than position 10, purely due to visibility. The model learns to score high-position events higher, which is a self-reinforcing loop.
**Solution:** Apply Inverse Propensity Scoring (IPS) to weight training examples. The propensity score $e(k)$ is the probability that an event was shown at position $k$, estimated from randomized experiments.

$$\text{IPS weight} = \frac{1}{e(k)}$$

This upweights registrations from low-position events (likely high-quality but hidden) and downweights registrations from position 1 (partly driven by visibility).

---

### Model Answers by Level

#### No Hire Answer
"The main failure mode is if the model doesn't have enough data. We should collect more data."

*Why this fails:* Completely vague. No specific failure modes, no mitigations.

---

#### Weak Hire Answer
"Some failure modes: cold start for new users and events, filter bubbles where we only recommend the same types of events, and the system being gamed by fake registrations.

For cold start, I'd use content-based recommendations as a fallback. For filter bubbles, I'd add diversity constraints. For gaming, I'd use fraud detection."

*Why this is weak hire:* Correct at a high level but no implementation depth. No discussion of time-sensitive events (domain-specific), no position bias, no seasonal shift.

---

#### Hire Answer
"I'll walk through six failure modes specific to this system.

**1. Time-sensitive event under-ranking.** Events happening in the next few hours have very few impressions and registrations — the model's popularity signals are low, so it ranks them poorly. But for a user making last-minute plans, a local event tonight is exactly the right recommendation. I'd add a time-urgency multiplier at the re-ranking stage: events within 6 hours get a 2× boost, events within 24 hours get a 1.5× boost. This is a post-model business rule, not a model feature.

**2. Cold start for new organizers.** A first-time organizer has zero historical signal. I'd use content-based ranking (description embedding similarity to user interests, category match) as a fallback for events with fewer than 50 impressions. I'd also reserve 1-2 'exploration slots' per recommendation request for new organizers to ensure they get early signal.

**3. Artificial popularity inflation (gaming).** Bad actors register for events using fake accounts to boost popularity features. I'd weight registration-from-account-age: accounts < 30 days old contribute 10% of the weight of a registration from a 1-year-old account. I'd also apply anomaly detection on registration velocity: a 10× spike in registrations in 30 minutes triggers a fraud review hold.

**4. Filter bubble / diversity collapse.** The model converges to recommending only the user's established categories. I'd enforce category diversity in the top-K: at most 3 out of 10 recommended events from the same category. This is a re-ranking constraint applied after scoring.

**5. Seasonal distribution shift.** The model trained on summer data under-ranks indoor winter events. I'd use a rolling training window (90 days, exponentially weighted) and monitor feature distribution drift with Population Stability Index (PSI): PSI > 0.2 triggers a model retraining alert.

**6. Position bias.** Events ranked at the top of the feed get more registrations purely due to visibility, not relevance. I'd apply IPS weighting during training:
$$w_i = \frac{1}{P(\text{shown at position } k_i)}$$
where $P(\text{shown at position } k_i)$ is estimated from randomized exposure experiments."

*Why this is a hire:* Six specific failure modes with concrete mitigations, domain-specific (time urgency), quantified thresholds, IPS formula.

---

#### Strong Hire Answer
*(Includes everything in the Hire answer, plus:)*

"I want to add two systemic failure modes that are harder to detect.

**7. Model collapse during online learning.** In the online learning pipeline, if a new model version is bad and gets deployed before the automated rollback triggers, it can corrupt the feedback loop: bad recommendations → fewer registrations → model sees fewer positive labels → model degrades further. This is a runaway degradation spiral. I'd mitigate with: (a) a minimum registration rate monitor — if registration rate drops >20% in 30 minutes, pause online learning and roll back to the last checkpoint; (b) dead band on model updates — require a minimum number of examples (e.g., 10K) before any weight update to prevent noise-driven updates.

**8. Organizer churn from zero early impressions.** If the event filtering stage never surfaces a new organizer's events because they score below the threshold, the organizer gets zero impressions, gets frustrated, and leaves the platform. This is an invisible supply-side failure — the recommendation system appears healthy (high CTR, high conversion) but the supply of events is slowly degrading. I'd monitor 'organizer impression rate within 7 days of event creation' as a platform health metric and set a minimum guaranteed impressions SLA for new organizers (e.g., 500 impressions for any event within 24 hours of creation)."

*Why this is strong hire:* Identifies model collapse spiral in online learning (advanced), quantifies rollback triggers, and raises the invisible supply-side churn problem with a specific monitoring metric and SLA — classic Principal-level platform thinking.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Interviewer Prompt

> "Step back and think about this system from a platform perspective. How does building this well — or badly — affect the health of the two-sided marketplace?"

### Ground Truth

This section assesses whether the candidate can think beyond the ML system to its effects on the ecosystem:

1. **Two-sided marketplace dynamics:** More relevant recommendations → more registrations → organizers succeed → they create more/better events → better supply for users → better recommendations. This is the virtuous cycle. Breaking it at any point (e.g., starving new organizers) can collapse the supply side.

2. **Organizer fairness:** If the recommendation system only surfaces events from already-popular organizers, new organizers can never build a following. This creates a winner-take-all dynamic that reduces supply diversity. Some marketplaces (YouTube, Spotify) have faced criticism for exactly this pattern.

3. **Supply-demand balance:** During peak seasons (summer, holidays), there are many events and many users — the matching problem is easier. During off-seasons, supply is thin and the model should shift toward showing events further away or in new categories to maintain user engagement.

4. **Trust and transparency:** Organizers need to trust that the platform is giving them fair exposure. Publishing an "organizer health score" with guidance on how to improve event discoverability (better descriptions, better photos, better pricing) creates a healthier supply side.

---

### Model Answers by Level

#### No Hire Answer
"The platform would be healthier if the recommendations are better."

*Why this fails:* Tautological. No understanding of marketplace dynamics.

---

#### Hire Answer
"The recommendation system is not just a user experience feature — it's the primary demand generation mechanism for organizers. If the system recommends well, organizers succeed, which attracts more and better organizers, which improves user supply, which makes the recommendations even better. This virtuous cycle is the core flywheel of a marketplace.

The failure mode I'd worry most about is organizer supply concentration. If the algorithm consistently rewards established organizers (because they have more historical data and social proof), new organizers never get traction, gradually stop using the platform, and supply diversity shrinks. This is the 'rich get richer' problem in recommendation systems.

My interventions:
1. **Guaranteed exploration budget:** Reserve 10% of recommendation slots for events with fewer than 500 impressions, regardless of model score. This is an explicit fairness mechanism that gives new events a chance.
2. **Organizer health dashboard:** Give organizers visibility into how their events are performing in discovery, and actionable guidance on improving discoverability (description quality, accurate location, competitive pricing).
3. **Supply health metrics:** Track 'new organizer retention rate' (what fraction of organizers who create a first event create a second). If this drops, the recommendation system is probably not giving new organizers enough early signal.

On supply-demand balance: during peak seasons, matching is easy. During off-seasons, the system should expand its radius (recommend events farther away, in new categories) to maintain user engagement even when local supply is thin."

*Why this is a hire:* Articulates the flywheel, identifies supply concentration as the key failure mode, proposes three concrete interventions, and addresses supply-demand seasonality.

---

#### Strong Hire Answer
*(Includes everything in the Hire answer, plus:)*

"I want to think about this from a product strategy perspective. The recommendation system is a platform capability — it should be designed as a reusable service that can power multiple surfaces (home feed, search ranking, email recommendations, push notifications), not as a one-off feature.

If I architect it as a platform service, I get several benefits:
1. One model serves multiple surfaces, so training data is pooled and the model is stronger.
2. Organizers get consistent exposure across surfaces — if their event is well-matched to a user, they see it everywhere, not just in the feed.
3. The feature store becomes a shared platform asset that other teams (fraud, pricing, A/B testing) can build on.

On the bilateral fairness question: I've talked a lot about organizer fairness, but there's also user fairness. The system should not exploit users' emotional triggers (fear of missing out, social pressure) to drive registrations for events they'll regret attending. This is a product integrity question that affects long-term trust and platform health. I'd measure 'post-event satisfaction' via surveys or implicit signals (did the user subsequently register for more events from the same organizer?) to detect cases where the model is optimizing for short-term conversion at the expense of user satisfaction.

Finally: the recommendation system has pricing power. If we can accurately predict which events a user wants, we can inform dynamic pricing — organizers can charge more for events with predicted high demand. This is a monetization opportunity but also a fairness question for users. I'd recommend keeping the recommendation and pricing systems separated by an abstraction layer, so that pricing optimization doesn't contaminate the ranking objective."

*Why this is strong hire:* Platform-as-a-service thinking, bilateral user fairness, post-event satisfaction monitoring, and clean separation of recommendation and pricing — these are Principal-level concerns about long-term system integrity and organizational leverage.

---

## Section 9: Appendix — Key Formulas & Reference

### Focal Loss

For class-imbalanced binary classification:

$$\text{FL}(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$$

**Parameters:**
- $p_t$: model's estimated probability for the true class
- $\alpha \in [0, 1]$: class weight balancing factor (typically 0.25 for rare positives)
- $\gamma \geq 0$: focusing parameter (typically 2.0)
- When $\gamma = 0, \alpha = 0.5$: reduces to standard binary cross-entropy
- As $\gamma$ increases, the loss focuses more on hard, misclassified examples

### Mean Average Precision (mAP)

$$\text{AP}(q) = \frac{1}{|R_q|} \sum_{k=1}^{n} P_q(k) \cdot \text{rel}_q(k)$$

$$\text{mAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}(q)$$

**Variables:**
- $q$: a query (user recommendation request)
- $|R_q|$: number of relevant items (ground truth registrations) for query $q$
- $P_q(k)$: precision at cut-off $k$ in the ranked list
- $\text{rel}_q(k)$: 1 if the item at position $k$ is relevant, 0 otherwise
- $|Q|$: total number of queries

### Binary Cross-Entropy Loss

$$\mathcal{L}_\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

### Inverse Propensity Scoring (IPS) Weight

For debiasing position bias in training:

$$w_i = \frac{1}{P(\text{shown at position } k_i \mid \text{query}_i)}$$

where the propensity $P(\text{shown at position } k)$ is estimated from randomized exposure experiments (random ranking for a held-out fraction of traffic).

### Laplace-Smoothed Popularity Ratio

$$r_\text{smooth} = \frac{N_\text{reg} + \alpha}{N_\text{imp} + \beta}$$

**Typical values:** $\alpha = 1, \beta = 10$ (encodes a prior that ~10% of impressions convert, consistent with platform average).

### Cosine Similarity for Interest Embedding

$$\text{sim}(\mathbf{u}, \mathbf{e}) = \frac{\mathbf{u} \cdot \mathbf{e}}{||\mathbf{u}||_2 \cdot ||\mathbf{e}||_2}$$

where:
- $\mathbf{u} = \frac{1}{|H_u|} \sum_{j \in H_u} \mathbf{e}_j$: mean embedding of user's past registered events $H_u$
- $\mathbf{e}$: embedding of the current candidate event

### Population Stability Index (PSI) for Feature Drift

$$\text{PSI} = \sum_{i=1}^{n} (P_{\text{serving},i} - P_{\text{training},i}) \cdot \ln\left(\frac{P_{\text{serving},i}}{P_{\text{training},i}}\right)$$

**Thresholds:**
- PSI < 0.1: no significant change
- 0.1 ≤ PSI < 0.2: moderate change, monitor closely
- PSI ≥ 0.2: significant change, trigger retraining

### Concrete System Numbers

| Parameter | Value |
|---|---|
| Total events on platform | ~1,000,000 |
| Active/upcoming events | ~100,000 |
| Daily active users | ~10,000,000 |
| Peak QPS for recommendations | ~10,000 |
| Candidate set after filtering | ~300-500 per request |
| Target p99 serving latency | <100ms |
| Stage 1 filtering latency | <10ms |
| Feature assembly latency (500 candidates) | ~5ms |
| XGBoost scoring latency (500 candidates) | ~5ms |
| NN scoring latency (500 candidates, GPU) | ~3ms |
| Training impression-to-registration rate | ~1-2% |
| Recommended focal loss α | 0.25 |
| Recommended focal loss γ | 2.0 |
| Online learning mini-batch size | 1,024 |
| Feature store cache TTL (dynamic features) | 5 minutes |
| Model canary rollout window | 5% traffic, 30 min |
| A/B test minimum duration | 2 weeks |
| Exploration budget (new organizer events) | 10% of slots |

### Feature Store Architecture

```
At serving time:
  ┌──────────────────────┐
  │   Request Context    │  → Time until event, GPS distance (computed inline)
  └──────────────────────┘
            +
  ┌──────────────────────┐
  │   Static Features    │  → User demographics, event metadata (Redis, ~1ms)
  └──────────────────────┘
            +
  ┌──────────────────────┐
  │   Dynamic Features   │  → Friend registrations, event popularity (Redis, updated ~5min)
  └──────────────────────┘
            ↓
  ┌──────────────────────┐
  │    ML Model Scorer   │  → XGBoost / NN, ~5ms
  └──────────────────────┘
            ↓
  ┌──────────────────────┐
  │   Re-ranking Rules   │  → Time urgency boost, diversity constraints
  └──────────────────────┘
```

### Two-Stage Retrieval Pipeline Summary

```
1M events
    │
    ▼ Stage 1: Event Filtering (rule-based, <10ms)
~500 candidates
    │
    ▼ Stage 2: Feature Assembly (feature store reads, ~5ms)
~500 scored candidates
    │
    ▼ Stage 3: ML Ranking (XGBoost/NN, ~5ms)
Top-20 ranked events
    │
    ▼ Stage 4: Re-ranking (business rules: urgency, diversity)
Final recommendation list
```

### Online Learning Pipeline Summary

```
User interaction (registration, click)
    │
    ▼ Kafka event stream
Feature enrichment service (same code as serving feature store)
    │
    ▼ Training examples assembled
Mini-batch buffer (accumulate 1,024 examples)
    │
    ▼ Gradient update (NN mini-batch SGD)
Model registry (MLflow)
    │
    ▼ Canary deployment (5% traffic, 30min eval)
    │
    ├─ Metrics OK → Promote to 100%
    └─ Metrics degrade → Automatic rollback
```

---

*Guide prepared for Staff/Principal ML Engineer interview calibration. All numbers are illustrative and based on typical consumer recommendation system benchmarks. Adjust thresholds based on actual platform data.*

---

## Section 10: Extended Deep Dives

### Deep Dive A: XGBoost Architecture and Why It Works on This Feature Set

XGBoost (eXtreme Gradient Boosting) is a specific implementation of gradient boosted decision trees (GBDT) that has dominated structured data competitions since its introduction in 2016. Understanding *why* it works well on this feature set requires understanding its learning mechanism.

**How GBDT works:**

GBDT trains an ensemble of $M$ shallow decision trees sequentially. Each tree $f_m$ is trained to predict the residual error of the previous ensemble:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot f_m(\mathbf{x})$$

where $\eta$ is the learning rate (shrinkage). The final prediction is:

$$\hat{y} = F_M(\mathbf{x}) = \sum_{m=1}^{M} \eta \cdot f_m(\mathbf{x})$$

Each tree $f_m$ is fit by minimizing the second-order Taylor approximation of the loss at the current residuals. XGBoost extends GBDT with:
1. **L1 and L2 regularization on tree leaf weights** (prevents overfitting on sparse features)
2. **Column (feature) subsampling** (like Random Forest, reduces variance)
3. **Row subsampling per tree** (faster training, adds diversity)
4. **Approximate split finding** (allows handling of large datasets efficiently)
5. **Sparsity-aware split** (natively handles missing values / sparse one-hot features)

**Why XGBoost works specifically on this feature set:**

The event recommendation feature set has several characteristics that align well with GBDT:

1. **High-order feature interactions:** The value of 'distance < 1 mile' is very different when 'time until event < 2 hours' vs. 'time until event > 7 days'. Decision tree splits naturally capture these interactions: a tree can learn a split like "if distance < 1 mile AND time < 2 hours, predict high registration probability." Logistic regression would need these interactions to be hand-crafted as explicit cross-features.

2. **Mixed feature types:** The feature set contains binary features (same city, host is friend), ordinal features (distance bucket, price bucket), dense continuous features (popularity ratio), and high-cardinality categoricals (category, subcategory). XGBoost handles all of these without requiring normalization, one-hot encoding of continuous features, or special preprocessing.

3. **Robustness to outliers:** Raw popularity counts (e.g., 1M registrations for a viral event) can be extreme outliers. Tree-based models are unaffected by outliers because splits only compare feature values ordinally (is feature value > threshold?), not cardinally. In contrast, neural networks with linear inputs are sensitive to feature scale.

4. **Small-to-medium dataset size:** For a feature set with ~100 features and ~100M training examples (1 year of impression logs), XGBoost converges in minutes on a single machine. Neural networks would require much longer training and more careful hyperparameter tuning.

**XGBoost hyperparameter choices for this problem:**

| Hyperparameter | Recommended Range | Rationale |
|---|---|---|
| `n_estimators` | 500-2000 | More trees = better with regularization |
| `max_depth` | 4-8 | Deep enough for 3-way interactions |
| `learning_rate` | 0.01-0.05 | Smaller = better generalization |
| `subsample` | 0.7-0.9 | Row subsampling for variance reduction |
| `colsample_bytree` | 0.6-0.8 | Feature subsampling |
| `scale_pos_weight` | `N_neg / N_pos` ≈ 50-100 | Handles class imbalance |
| `min_child_weight` | 10-50 | Prevents splits on very few examples |

Note: for class imbalance, XGBoost's `scale_pos_weight` is equivalent to upweighting positives. This is an alternative to focal loss when using XGBoost (which does not natively support focal loss without a custom objective function).

---

### Deep Dive B: Neural Network Architecture for Event Recommendation

The recommended neural network architecture for production is a **Wide & Deep** model (inspired by Google's 2016 paper) or its evolution, a **Two-Tower model with interaction layers**.

**Wide & Deep Architecture:**

$$\hat{y} = \sigma(\mathbf{w}_\text{wide}^\top [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{W}_\text{deep}^\top \mathbf{a}^{(L)} + b)$$

- **Wide component:** Linear model on raw features + hand-crafted cross-features. Memorizes specific feature combinations seen in training. Example: "user in San Francisco AND category = Tech → high registration." Fast inference, interpretable.
- **Deep component:** Multi-layer perceptron on dense feature embeddings. Generalizes to unseen feature combinations by learning low-dimensional representations.
- **Joint training:** Both components are trained simultaneously.

**Two-Tower Architecture (preferred for retrieval scaling):**

```
User Input          Event Input
   │                    │
   ▼                    ▼
User Feature        Event Feature
  Vector              Vector
   │                    │
   ▼                    ▼
User Embedding      Event Embedding
Tower (MLP)         Tower (MLP)
   │                    │
   └────────┬───────────┘
            ▼
    Dot Product / Concat
            +
    Contextual Features
    (distance, time, social)
            │
            ▼
    Output MLP → sigmoid
            │
            ▼
    P(registration)
```

**Why two-tower for serving:** Once trained, user embeddings can be pre-computed and cached. At serving time, only the event tower runs for each candidate. This reduces serving cost from O(N_features × N_candidates) to O(user_embedding + N_candidates × event_embedding).

**Embedding dimensions:**

| Feature | Embedding Dim | Rationale |
|---|---|---|
| User ID | 64 | High cardinality, dense embedding |
| Event category | 16 | ~50 categories |
| Event subcategory | 32 | ~500 subcategories |
| City | 32 | ~10K cities |
| Description (SBERT) | 384 → 64 | Project down from SBERT |

**MLP layer sizes:**
- User tower: [512, 256, 128]
- Event tower: [512, 256, 128]
- Merge + context: [256, 128, 64]
- Output: [1] + sigmoid

**Dropout:** 0.3 at each hidden layer. **BatchNorm:** applied before each activation. **Activation:** ReLU throughout, sigmoid at output.

**Continual learning setup:**
- Mini-batch SGD, batch size 1024, learning rate $\eta = 0.001$ with cosine annealing
- Experience replay: 20% of each mini-batch consists of randomly sampled examples from the last 30 days to prevent catastrophic forgetting of older patterns
- Parameter snapshots every 6 hours for rollback capability

---

### Deep Dive C: A/B Testing Design for Recommendation Systems

Designing A/B tests for recommendation systems is more complex than for simple UI experiments because of the **network effects** and **spillover** between treatment and control groups.

**Standard A/B test design:**

Split users randomly into treatment (new model) and control (old model). Hold test for a minimum of 2 weeks (captures weekly seasonality). Statistical significance at $p < 0.05$ with pre-registered primary metric (conversion rate).

**Sample size calculation:**

For a two-sample proportion test:

$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2p(1-p)}{\delta^2}$$

where:
- $z_{\alpha/2} = 1.96$ (for $\alpha = 0.05$, two-tailed)
- $z_\beta = 0.84$ (for 80% power)
- $p$: baseline conversion rate (~0.02)
- $\delta$: minimum detectable effect (0.002 = 10% relative lift)

Plugging in: $n \approx \frac{(1.96 + 0.84)^2 \cdot 2 \times 0.02 \times 0.98}{(0.002)^2} \approx 150,000$ users per variant.

At 10M DAU and 90/10 split: 1M control users. We exceed the required sample size in less than 1 day — but we still run for 2 weeks due to day-of-week effects.

**Novelty effect:** Users who see a new recommendation UI may engage more simply because it's new. This inflates early A/B test metrics. Mitigation: pre-commit to measuring the effect at 2 weeks, not stopping early.

**Network effect spillover:** If a user in the treatment group registers for an event and invites a friend in the control group, the control user's behavior is contaminated. For social features, this is hard to fully eliminate. Mitigation: use geo-based splitting (treatment = certain cities, control = other cities) to reduce cross-group interactions.

**Guardrail metrics:** In addition to the primary metric (conversion rate), define guardrail metrics whose violation triggers an automatic stop:
- Recommendation latency p99 > 500ms → stop
- Overall registration rate drops >10% → stop (safety net)
- Error rate of recommendation API > 1% → stop

---

### Deep Dive D: Training Data Pipeline

**Data collection flow:**

```
User opens app
    │
    ▼
Recommendation service returns ranked list of events
    │
    ├── Log impression event: (user_id, event_id, rank_position, timestamp, model_version)
    │
User scrolls / clicks / registers
    │
    ├── Log interaction event: (user_id, event_id, interaction_type, timestamp)
    │
    ▼
Kafka topic: user_interactions
    │
    ▼
Stream processing (Flink/Spark Streaming)
    - Join impression and interaction events by (user_id, event_id, session_id)
    - Label: 1 if interaction_type = 'registration', 0 if only impression within session
    │
    ▼
Labeled training example: (user_id, event_id, label, impression_timestamp)
    │
    ▼
Feature store lookup: enrich with features as of impression_timestamp
    │
    ▼
Training example stored to data warehouse (BigQuery / Redshift)
```

**Important: point-in-time correctness.** When we look up features for a training example, we must use the feature values as of the `impression_timestamp`, not the current time. If we accidentally use the current values of 'number of friends registered,' we're leaking future information: a user registers because of an event's early momentum, but we train the model with the final high friend-registration count, which wasn't available when the impression was made.

Implementation: the feature store must support time-travel queries: `GET feature(user_id=X, event_id=Y, as_of=impression_timestamp)`. This requires storing historical feature snapshots, which adds storage cost (~10% overhead) but is essential for correctness.

**Training data volume and window:**

| Parameter | Value | Rationale |
|---|---|---|
| Training window | Rolling 90 days | Captures seasonality without over-weighting stale data |
| Positive examples | ~5M per 90 days | ~1-2% of impressions convert |
| Negative examples | ~250M per 90 days | All non-registering impressions |
| Downsampling ratio for negatives | 10:1 (negative:positive) | Balance for training; adjust model threshold at serving |
| Total training set size | ~55M examples | Manageable for XGBoost and NN |

**Label expiration:** If a user was impressed with an event but hasn't registered within 7 days and the event has passed, label = 0. We do not count deferred registrations (registered after event ended) as positives.

---

### Deep Dive E: Model Interpretability and Debugging

For a production recommendation system, model interpretability is not just a nice-to-have — it's essential for debugging, building trust with organizers, and regulatory compliance.

**XGBoost interpretability tools:**

1. **Feature importance (gain):** The total gain in model performance attributed to each feature across all splits. Typical output for this feature set:
   - Distance bucket: ~18% gain
   - Time until event: ~15% gain
   - Friend registration count: ~12% gain
   - Category match: ~10% gain
   - Description similarity: ~9% gain
   - ... remaining features

2. **SHAP values (SHapley Additive exPlanations):** A game-theoretic approach to assign each feature a contribution to the prediction for a specific example:
   $$\hat{y} = \phi_0 + \sum_{j=1}^{M} \phi_j$$
   where $\phi_0$ is the base rate prediction and $\phi_j$ is the SHAP value for feature $j$. SHAP is additive, consistent, and provides local (per-example) explanations. Essential for debugging "why was this event recommended?" queries from organizers.

**Example SHAP explanation:**
```
Event "Jazz Night at Blue Room" recommended to User 12345 because:
  Base rate:                 +0.012
  Distance: 0.3 miles        +0.031
  Time until event: 4 hours  +0.024
  Friend registered: 2 friends +0.019
  Category: Music (user pref) +0.015
  Price: Free                +0.008
  Final prediction:           0.109 (10.9% registration probability)
```

**Neural network interpretability:**

For NNs, SHAP is more expensive to compute (requires background dataset and forward passes). Practical alternatives:
1. **Integrated Gradients:** Attribute the prediction to input features by integrating gradients from a baseline to the actual input. Computationally tractable and theoretically grounded.
2. **Attention visualization** (if using transformer-based description encoder): Visualize which words in the event description contributed most to the interest similarity score.

**Production monitoring dashboards:**

1. **Feature importance stability:** Plot SHAP feature importances over time. If 'friend registration count' suddenly drops from 12% to 2% importance, a bug in the social feature pipeline is likely.
2. **Score distribution:** Plot the histogram of model output scores daily. A shift toward lower scores across the board may indicate a problem with the training data.
3. **Error analysis:** For events with very high predicted probability (>0.8) that received zero registrations, investigate: was the event cancelled? Was it a fraudulent event? This informs data quality improvements.

---

### Section 11: Probing Questions for Interviewers

Use these questions to push candidates to the next level of depth. Good candidates should be able to answer at least 60% of questions for their target level.

**Section 1 probes:**
- "What if the user is traveling and not in their home city? How does your system handle this?"
- "What if an event is virtual (online) — how does location factor in?"
- "How would you handle a user who has never registered for any events? What data do you have?"

**Section 2 probes:**
- "Why did you choose binary classification rather than regression (predicting actual registration count)?"
- "If we wanted to optimize for GMV rather than registrations, what changes?"
- "Explain what 'position bias' is and how it affects this problem specifically."

**Section 3 probes:**
- "For the description embedding, which specific model would you use? What embedding dimension?"
- "How do you handle the cold start for a brand new event with zero registrations and zero impressions?"
- "What does 'historical location preference' mean for a user who moved cities 3 months ago?"

**Section 4 probes:**
- "If you had to ship something in 2 weeks, what would you ship and why?"
- "What happens to your XGBoost model if a new event category is added to the platform?"
- "How do you prevent catastrophic forgetting in your online learning NN setup?"

**Section 5 probes:**
- "mAP assumes binary relevance (registered or not). But a user who registered and attended is more 'relevant' than one who registered but didn't attend. How would you modify the metric?"
- "How do you know if your offline mAP improvement will translate to online metric improvement?"
- "What is the minimum statistically significant A/B test effect size you'd care about?"

**Section 6 probes:**
- "What happens if the feature store goes down during serving?"
- "How do you handle the case where a new event is created while a recommendation request is in flight?"
- "What is your rollback strategy if a bad model version is deployed?"

**Section 7 probes:**
- "How would you detect if an organizer is buying fake registrations to game the system?"
- "What happens during a major local event (e.g., Super Bowl week) where many normally active users are away?"
- "How would you handle events that get cancelled after they've been recommended?"

**Section 8 probes:**
- "How would you design the recommendation system differently if organizers paid for placement (sponsored events)?"
- "If you were the CTO, what would be your top 3 investment priorities for the recommendation system over the next 12 months?"
- "How do you balance personalization (giving users what they want) with serendipity (exposing them to new things)?"

---

### Section 12: Common Mistakes to Watch For

**Mistake 1: Confusing CTR and conversion rate as equivalent metrics.**
CTR (clicks/impressions) measures engagement. Conversion rate (registrations/impressions) measures business outcomes. A model that maximizes CTR may surface clickbait events that get clicks but not registrations. Always tie the primary metric to the business goal.

**Mistake 2: Not accounting for class imbalance.**
A registration rate of 1-2% means a trivial model that always predicts 0 achieves 98-99% accuracy. Candidates who propose accuracy as a metric have not understood the data distribution.

**Mistake 3: Ignoring temporal dynamics.**
Events have expiration dates. A recommendation system that ignores the remaining time until event is fundamentally broken — recommending an event that already happened is a user experience disaster. The time dimension must appear in both features (time until event) and candidate filtering (future events only).

**Mistake 4: Training-serving skew.**
Many candidates describe a training pipeline and a serving pipeline as if they are completely separate systems. In practice, the most common source of production ML failures is features being computed differently between training and serving. Candidates should proactively address this.

**Mistake 5: Single-metric tunnel vision.**
Optimizing only for conversion rate can degrade organizer supply diversity, user trust, or long-term engagement. Staff/Principal candidates should spontaneously mention guardrail metrics, organizer health, and long-term platform health alongside the primary metric.

**Mistake 6: Underspecifying the candidate generation stage.**
"Run the model on all events" is infeasible at 1M events × 10K QPS. The two-stage retrieval + ranking pattern is standard knowledge at Staff level. Candidates who jump straight to model scoring without a candidate generation stage are missing a fundamental component.

**Mistake 7: Treating online learning as free.**
Candidates sometimes suggest "just use online learning" without discussing the risks: model collapse, feedback loops, infrastructure complexity, the need for rollback mechanisms. Online learning is a significant engineering investment with real risks, not a drop-in solution.

