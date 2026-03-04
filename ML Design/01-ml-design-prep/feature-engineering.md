# Feature Engineering

## Introduction

Feature engineering is probably the highest-leverage activity in most ML system design interviews — and most candidates handle it poorly. They either spend 15 minutes listing every feature they can think of (the "feature dump") or they breeze past it in 30 seconds and jump to model architecture. Both are red flags.

The reality is that in production ML systems, the choice and quality of features often matters more than the choice of model. A well-featured logistic regression will beat a poorly-featured deep network almost every time. Your job in the interview is to show that you can think systematically about what signals are available, how to encode them, and which ones are most predictive — all while keeping the conversation moving.

---

## The Feature Engineering Mindset

Before diving into specific feature types, here's the mental model: every feature you propose should answer two questions.

1. **Why is this predictive?** What signal does this feature carry about the label you're trying to predict?
2. **How do I encode it?** What representation does the model actually receive?

If you can't answer both, don't propose the feature. A feature that "might be useful" but you can't explain why is worse than no feature — it wastes interview time and signals that you're guessing rather than reasoning.

> "I want to focus on the five features I think will have the biggest impact, and for each one I'll explain why it's predictive and how I'd encode it."

This is what a strong candidate says. It shows structure, focus, and an awareness of time.

---

## Categorical Features

Categorical features are variables that take on discrete values — things like `user_country`, `device_type`, `item_category`, or `video_language`.

### One-Hot Encoding

The simplest approach: create a binary column for each possible value. `user_country = "US"` becomes a vector like `[1, 0, 0, ..., 0]` with a 1 in the US position.

- **When to use:** Low cardinality (fewer than ~50 unique values). Countries, device types, content categories.
- **When it fails:** High cardinality. If you one-hot encode `item_id` with 1B items, you get a 1B-dimensional sparse vector. That's not going to work.

### Embedding Lookup

For high-cardinality categoricals, you learn a dense vector (embedding) for each value. `item_id = 12345` maps to a learned vector like `[0.23, -0.11, 0.87, ...]` of dimension 32-256.

- **When to use:** Cardinality above ~50, especially entity IDs (user_id, item_id, creator_id, query tokens).
- **How it works:** An embedding table is a matrix of shape `(num_values × embedding_dim)`. Each value is a row index. The model learns the embedding values during training.
- **Rule of thumb for dimension:** Start with `min(50, cardinality^0.25)` and tune from there.

### Target Encoding

Replace each category value with the mean of the target variable for that category. `city = "San Francisco"` becomes `0.73` if the average click rate for SF users is 73%.

- **When to use:** Gradient-boosted tree models (XGBoost, LightGBM) where you can't use embeddings.
- **Pitfall:** Leaks target information if you don't do it carefully. Always compute target means on the training fold only, and use smoothing to handle categories with few samples.

### The Cardinality Decision

| Cardinality | Encoding | Why |
|-------------|----------|-----|
| < 10 | One-hot | Simple, interpretable, minimal overhead |
| 10-50 | One-hot or embedding | Either works; embedding if the model is neural |
| 50-10K | Embedding | Too sparse for one-hot, enough data to learn embeddings |
| 10K-1B+ | Embedding with hashing | Embedding table size becomes a problem; hash to a fixed vocabulary |

---

## Numerical Features

Numerical features are continuous values — prices, counts, durations, distances, scores.

### Normalization

Raw numerical features can have wildly different scales. `user_age` ranges from 13-100, while `view_count` ranges from 0 to 1B. Without normalization, the model gives more weight to larger-scale features.

- **Z-score normalization:** `(x - mean) / std`. Centers around 0 with unit variance. Use when the distribution is roughly Gaussian.
- **Min-max normalization:** `(x - min) / (max - min)`. Scales to [0, 1]. Use when you need bounded values (e.g., for similarity scores).
- **Robust scaling:** `(x - median) / IQR`. Use when you have outliers — it's not affected by extreme values.

### Log Transforms

Many real-world distributions are heavily right-skewed: prices, follower counts, view counts, dwell times. A few items have enormous values while most are small.

Applying `log(x + 1)` (the +1 avoids log(0)) compresses the range and makes the distribution more Gaussian-like, which helps most models learn better.

> "View counts are heavily right-skewed — a few viral videos have billions of views while most have under 1000. I'd apply a log transform to compress this range."

### Bucketing / Binning

Sometimes it's better to discretize a continuous feature into bins.

- **When to bucket:** The relationship between the feature and the label is non-linear and step-wise. For example, age groups behave differently: 13-17, 18-24, 25-34, etc.
- **When NOT to bucket:** You lose information. If the model can learn non-linear relationships on its own (neural networks), bucketing can actually hurt.
- **How to choose bins:** Domain knowledge (age groups), quantiles (equal-frequency bins), or learned (tree models find optimal splits automatically).

### Missing Values

Missing values are more common than most candidates realize. A feature might be missing because the data wasn't collected, the user didn't fill it in, or the feature pipeline failed.

| Strategy | When to use | Example |
|----------|-------------|---------|
| Impute with mean/median | Feature is missing at random, low missing rate | Fill missing `age` with median age |
| Impute with 0 or -1 | Missing has a meaningful interpretation | Fill missing `time_since_last_purchase` with -1 (no purchase history) |
| Add a binary indicator | Missingness itself is informative | `has_profile_photo = 0` predicts lower engagement |
| Let the model handle it | Tree models (XGBoost) handle NaN natively | Don't impute — let the tree learn the best split for missing |

---

## Text Features

Text is everywhere in ML systems: search queries, item titles, product descriptions, reviews, comments.

### Bag-of-Words and TF-IDF

The simplest representations. Bag-of-words counts word occurrences. TF-IDF weights words by how unique they are across documents.

- **When they're still useful:** Baseline models, sparse features for gradient-boosted trees, when you need interpretability (which words drove the prediction?), and when compute is constrained.
- **Limitations:** No word order, no semantics. "The cat sat on the mat" and "The mat sat on the cat" are identical.

### Pretrained Embeddings

Dense vector representations from pretrained models.

| Model | Type | Dimension | Best for |
|-------|------|-----------|----------|
| Word2Vec / GloVe | Static (one vector per word) | 100-300 | Simple similarity, low compute |
| FastText | Static, subword-aware | 100-300 | Handles typos and rare words |
| BERT / sentence-transformers | Contextual (vector depends on context) | 768 | Semantic understanding, search, matching |
| Distilled models (MiniLM, TinyBERT) | Contextual, smaller | 384 | Production serving with latency constraints |

- **When to use pretrained:** You don't have enough domain-specific text data to train from scratch, or you need general semantic understanding.
- **When NOT to use pretrained:** Your domain has specialized vocabulary the pretrained model hasn't seen (medical, legal, internal jargon). In that case, fine-tune or train domain-specific embeddings.

### Tokenization Choices

How you tokenize text affects downstream performance. BPE (used by GPT models), WordPiece (used by BERT), and SentencePiece all handle subwords differently. In an interview, you probably won't go deep on tokenization for feature engineering — but mentioning that tokenization choice matters for multilingual systems or code is a nice signal.

---

## Temporal Features

Time is one of the most predictive — and most underused — feature categories. Users behave differently on weekday mornings vs weekend nights. Items have seasonal patterns. Trends change over time.

### Cyclical Encoding

Time-of-day, day-of-week, and month-of-year are cyclical: 11pm is closer to 1am than to 3pm, but a raw hour encoding (23 vs 1 vs 15) doesn't capture this.

Encode cyclical features using sine and cosine:
- `hour_sin = sin(2π * hour / 24)`
- `hour_cos = cos(2π * hour / 24)`

This gives you two features that correctly represent the circular nature of time.

### Recency Features

How recently something happened is often more predictive than what happened.

- `time_since_last_click` — seconds/minutes since the user's last interaction
- `time_since_item_published` — freshness of the content
- `time_since_last_login` — engagement recency

These features often benefit from a log transform (recent events matter more than distant ones) or exponential decay weighting.

### Aggregation Windows

A single aggregation window hides important patterns. A user who clicked 100 times in the last hour is different from one who clicked 100 times in the last month.

Always use **multiple time windows**:
- Last 1 hour, last 24 hours, last 7 days, last 30 days
- For each window: count, sum, mean, max, min, trend

> "I'd compute the user's click count over four time windows — last hour, last day, last week, last month. The ratio between recent and historical activity is itself a strong feature: a user whose hourly rate is 10x their monthly average is probably in an active session."

### Trend Features

Is the metric going up or down? Trend features capture directional changes:
- `clicks_7d / clicks_30d` — ratio indicates acceleration or deceleration
- `views_this_week - views_last_week` — absolute change
- Slope of a linear fit over the last N data points

---

## Cross Features and Interactions

Individual features capture one dimension. Cross features capture how dimensions interact.

### Explicit Feature Crosses

Combine two features into one: `user_age_bucket × item_category`. This creates a new feature for each combination (e.g., "18-24 × Sports", "25-34 × Music").

- **Why they help:** A 20-year-old's preference for sports content is different from a 60-year-old's. The model needs to learn separate patterns for each combination.
- **Why models struggle without them:** Linear models and shallow networks can't easily learn interactions from raw features. They need explicit crosses.

### Automated Feature Interaction

Deep learning architectures can learn feature interactions automatically, but some architectures are better at it:

| Architecture | How it handles interactions | Best for |
|---|---|---|
| Wide & Deep | Wide (linear) component memorizes crosses; deep component generalizes | Click prediction with both memorization and generalization |
| DeepFM | Factorization machine layer + DNN jointly learn interactions | Feature-rich tabular data |
| DCN v2 (Deep & Cross Network) | Cross network explicitly models bounded-degree interactions | When you want explicit high-order crosses without manual engineering |

### When Cross Features Matter Most

Cross features have the biggest impact when:
1. The relationship between features is non-additive (age AND gender together predict differently than each alone)
2. You're using a linear or shallow model that can't learn interactions automatically
3. You have strong domain knowledge about which features interact

They matter less when:
1. You're using a deep network with enough capacity to learn interactions
2. You have abundant data for the model to discover patterns itself

---

## Feature Freshness and Serving

This is where Senior candidates become Staff candidates. Knowing what features to compute is table stakes. Knowing how to *serve* them in production is what separates levels.

### Batch vs Real-Time Features

Not all features need to be computed in real-time. The tradeoff is freshness vs cost.

| Feature type | Update frequency | Examples | Infrastructure |
|---|---|---|---|
| **Static** | Rarely (days/weeks) | User demographics, item metadata | Batch pipeline, key-value store |
| **Batch** | Hourly/daily | User's 30-day click count, item popularity score | Batch pipeline → feature store |
| **Near-real-time** | Minutes | Trending topics, recent user activity aggregates | Streaming pipeline (Kafka → Flink) |
| **Real-time** | Per-request | Current session context, device info, time of day | Computed at request time |

### Feature Store Architecture

A feature store provides a unified interface for serving features to models in production.

- **Offline store:** Stores historical feature values for training. Backed by data warehouse (BigQuery, Hive).
- **Online store:** Serves the latest feature values for inference. Backed by low-latency store (Redis, DynamoDB).
- **Streaming pipeline:** Computes near-real-time features and writes to both stores.

### The Cost of Real-Time Features

Real-time features are expensive:
- Each one adds latency to the prediction path
- Each one requires a streaming pipeline to maintain
- Each one is a potential point of failure

Before proposing a real-time feature, ask: "Does the freshness of this feature measurably improve the prediction?" If a daily-updated feature is 95% as good as a real-time one, the daily version is almost always the right choice.

### Feature Drift Detection

Features can silently change over time:
- An upstream data pipeline changes its schema
- A feature starts returning nulls or stale values
- User behavior shifts and feature distributions change

Monitor feature distributions in production. Alert on:
- Distribution shift (KL divergence, KS test, PSI > 0.1)
- Null rate changes (sudden spike in missing values)
- Range violations (values outside expected bounds)

---

## Interview Strategy

### The 5-Feature Rule

In an interview, don't list 30 features. Pick the **5 most impactful features** and for each one:
1. State the feature and its encoding
2. Explain why it's predictive (what signal it carries)
3. Note any engineering challenges (freshness, missing values, cardinality)

This demonstrates judgment — you're not just brainstorming, you're prioritizing.

### Structure Your Feature Discussion

Start with the most predictive signal and work down:

> "The single most predictive feature for click-through prediction is the user's historical CTR on this content category. I'd encode this as a numerical feature with a log transform, computed over multiple time windows — last day, last week, last month."

Then move to the next most important feature. Don't let the interviewer ask "what about X?" — anticipate the important features proactively.

### Avoid the Feature Dump Tarpit

Feature discussions are easily the biggest tarpit for candidates. There's always one more feature you could mention. Set a time limit for yourself:

> "I could keep going on features, but I want to make sure we have time for modeling and evaluation. Let me earmark feature interactions as something I'd explore further, and move on to model selection."

**Green Flags**
- You selected features with clear reasoning about predictive value
- You mentioned encoding choices and why they matter
- You discussed freshness and serving tradeoffs for at least one feature
- You kept the discussion focused and moved on when appropriate

**Red Flags**
- You listed 20 features without explaining why any of them are predictive
- You ignored encoding entirely (treating "user_id" as a raw number)
- You didn't mention temporal features or aggregation windows
- You spent 15 minutes on features and ran out of time for modeling

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should be able to identify the main feature categories for a given problem (user features, item features, context features) and propose reasonable encodings. They should know the difference between one-hot and embedding representations and when to use each. A mid-level candidate working on a recommendation system should mention user interaction history, item metadata, and temporal features as core signals. They differentiate by showing they can build a workable feature set without getting lost in the weeds.

### Senior Engineer

Senior candidates demonstrate deeper feature engineering intuition. They proactively discuss feature freshness (batch vs real-time), handle missing values thoughtfully, and propose cross features or interaction terms that capture non-obvious patterns. For a click prediction system, a senior candidate would discuss position bias in click features, the importance of multiple aggregation windows, and how to handle cold-start users who have no feature history. They bring up practical considerations like feature store architecture and feature drift without being prompted.

### Staff Engineer

Staff candidates treat feature engineering as a system design problem, not just a modeling problem. They focus on which features provide the highest marginal lift relative to their serving cost. They recognize that the best feature engineering often looks like data engineering — getting access to new data sources, building streaming pipelines for real-time signals, or creating feedback loops where model outputs become features for downstream models. A Staff candidate might point out that the biggest opportunity isn't better features on existing data, but instrumenting a new user behavior signal that nobody is currently collecting.
