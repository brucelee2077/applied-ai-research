# Ad Click Prediction on Social Platforms

> **Interview Module 08** -- Based on ByteByteGo ML System Design Interview, Chapter 8

---

## Table of Contents

1. [What Is Ad Click Prediction and Why It Matters](#what-is-ad-click-prediction-and-why-it-matters)
2. [Clarifying Requirements](#clarifying-requirements)
3. [Framing the Problem as an ML Task](#framing-the-problem-as-an-ml-task)
4. [Data Preparation](#data-preparation)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
7. [Evaluation](#evaluation)
8. [Serving Architecture](#serving-architecture)
9. [Advanced Talking Points](#advanced-talking-points)
10. [Interview Cheat Sheet](#interview-cheat-sheet)

---

## What Is Ad Click Prediction and Why It Matters

### The Simple Explanation (Like You Are 12)

Imagine you run a lemonade stand and you want to put up posters around town. You only have enough money for 5 posters, so you need to pick the 5 spots where thirsty people are most likely to see them and actually come buy lemonade.

Now imagine you are Google or Facebook. Millions of companies want to show their ads to billions of users. Your job is to figure out: **"If I show THIS ad to THIS person right NOW, how likely are they to click on it?"**

Why does this matter so much?

- **This is how tech companies make money.** Google made over $200 billion in 2023 from ads alone. Facebook/Meta makes about 97% of its revenue from ads. If the ad system gets even 0.1% better at predicting clicks, that can mean billions of dollars in extra revenue.
- **Better predictions = happier everyone.** Users see ads they actually find useful instead of annoying ones. Advertisers get more bang for their buck. The platform makes more money. It is a three-way win.
- **It is one of the most common ML system design interview questions** because it touches nearly every important concept: ranking, feature engineering, real-time serving, continual learning, and calibration.

### The Staff-Level Technical Framing

Ad click prediction is a **pointwise Learning-to-Rank (LTR)** problem where we train a binary classifier to estimate `P(click | ad, user, context)`. The predicted click-through rate (pCTR) feeds into an **ad auction** system (typically second-price or VCG-based) that determines which ads are shown and how much advertisers pay. Because revenue is directly proportional to `pCTR * bid`, even small improvements in calibration or discrimination translate to substantial revenue gains.

The system must handle:
- **Massive scale**: billions of predictions per day with <100ms latency
- **Extreme sparsity**: millions of categorical features, most of which are zero
- **Continual learning**: even a 5-minute delay in model updates can degrade performance
- **Calibration**: predicted probabilities must be well-calibrated (not just well-ranked) because they feed into the auction

---

## Clarifying Requirements

Before jumping into a solution, always clarify requirements with the interviewer. Here are the key questions from the PDF:

| Question | Answer |
|----------|--------|
| Business objective? | Maximize revenue |
| Ad types and placement? | Image ads on user timeline only (simplified) |
| Same ad shown multiple times? | Yes (no fatigue period for simplicity) |
| Negative feedback? | Users can "hide this ad" |
| Training data construction? | User + ad data, labels from interactions |
| Continual learning? | Critical -- even 5-min delay hurts performance |

### How to Define Negative Labels (Tricky!)

This is a classic interview question. How do you know a user "did not click"?

- **Option 1 (Simple):** Any impression not clicked within `t` seconds is negative.
- **Option 2 (Better):** Only count as negative if the ad was visible on screen for a minimum dwell time (e.g., 5 seconds) but not clicked. A 0.4-second scroll-by should not count.
- **Option 3 (Even Better):** Use explicit negative feedback ("hide this ad") as strong negatives.

In practice, companies use sophisticated delayed-feedback models and attribution windows.

---

## Framing the Problem as an ML Task

### ML Objective

**Business goal:** Maximize ad revenue
**ML objective:** Predict if a user will click an ad -- `P(click | user, ad, context)`

This is because correctly predicting click probabilities lets the system display relevant ads, which leads to increased revenue.

### Input and Output

```
Input:  A user (with their profile + history)
Output: A ranked list of ads, ordered by click probability
```

### Why Pointwise Learning-to-Rank?

Think of it like this: instead of comparing ads against each other (pairwise), we score each ad independently. Each `(user, ad)` pair gets a probability score, and we rank by that score.

```
Model Input:  (user_features, ad_features, context_features)
Model Output: P(click) in [0, 1]
```

This is modeled as **binary classification**: click = 1, no click = 0.

---

## Data Preparation

### Raw Data Sources

**1. Ads Table**

| Ad ID | Advertiser ID | Ad Group ID | Campaign ID | Category | Subcategory |
|-------|--------------|-------------|-------------|----------|-------------|
| 1     | 1            | 4           | 7           | travel   | hotel       |
| 2     | 7            | 2           | 9           | insurance| car         |
| 3     | 9            | 6           | 28          | travel   | airline     |

**2. Users Table**

| ID | Username | Age | Gender | City | Country |
|----|----------|-----|--------|------|---------|

**3. User-Ad Interactions Table**

| User ID | Ad ID | Interaction Type | Dwell Time | Location |
|---------|-------|-----------------|------------|----------|
| 11      | 6     | Impression      | 5 sec      | ...      |
| 11      | 7     | Impression      | 0.4 sec    | ...      |
| 4       | 20    | Click           | -          | ...      |
| 11      | 6     | Conversion      | -          | ...      |

### Constructing Training Labels

For every ad impression, construct a training example:

- **Positive label (y=1):** User clicks the ad within `t` seconds after it is shown
- **Negative label (y=0):** User does NOT click within `t` seconds

Note: `t` is a hyperparameter tuned via experimentation.

### The Sparse Feature Challenge

Ad click prediction systems deal with massive, sparse feature spaces:
- "Ad category" has hundreds of possible values
- "Advertiser ID" and "User ID" can have millions of unique values
- Most features are zero (sparse)

This is a unique challenge that drives many model architecture decisions.

---

## Feature Engineering

### Ad Features

| Feature | Why Important | How to Prepare |
|---------|--------------|----------------|
| **IDs** (advertiser, campaign, ad group, ad) | Capture unique characteristics of each entity | Embedding layer converts sparse IDs into dense vectors |
| **Image/Video** | Visual content signals what the ad is about (airplane = travel) | Pre-trained model (e.g., SimCLR) converts to feature vector |
| **Category & Subcategory** | Helps model understand ad topic | Provided by advertiser from predefined list |
| **Impression & Click Numbers** | Historical CTR indicates ad quality | Numerical values: total impressions/clicks per ad, advertiser, campaign |

### User Features

| Feature | Why Important | How to Prepare |
|---------|--------------|----------------|
| **Demographics** (age, gender, city, country) | Basic user profile | Standard encoding |
| **Contextual info** (device, time of day) | Context affects click behavior | Standard encoding |
| **Clicked ads history** | Previous clicks indicate interests (many insurance clicks = likely to click insurance again) | Same embedding approach as ad features |
| **Historical engagement stats** (total views, click rate) | Past engagement predicts future engagement | Scale to similar range |

### The Key Insight About Feature Interactions

Individual features alone are not enough. The magic happens when features **interact**:
- "Young" + "Basketball" together predict clicks on sports ads
- "USA" + "Football" together predict clicks on NFL ads
- Neither feature alone is as predictive as the combination

This is why we need models that can learn **feature crosses** automatically.

---

## Model Development

### The Progression of Models (Simple to Complex)

Think of this like building a rocket. You start with a paper airplane, then a model rocket, then the real thing:

#### 1. Logistic Regression (LR) -- The Paper Airplane

**How it works:** Linear combination of features passed through sigmoid.

**Pros:** Fast to train, easy to implement, great baseline.

**Cons:**
- Cannot solve non-linear problems (linear decision boundary)
- Cannot capture feature interactions (the effect of one feature does not depend on another)

**Verdict:** Use as a baseline only.

#### 2. Feature Crossing + LR -- Adding Wings

**How it works:**
1. Manually create new features by combining existing ones (e.g., country x language)
2. Feed original + crossed features into LR

**Pros:** Can capture some pairwise interactions.

**Cons:**
- Manual process requiring human domain expertise
- Cannot capture complex interactions from thousands of sparse features
- Crossing increases sparsity even more

**Verdict:** Better than plain LR, but too manual.

#### 3. Gradient Boosted Decision Trees (GBDT) -- The Model Rocket

**Pros:** Interpretable, good performance.

**Cons:**
- Inefficient for continual learning (cannot fine-tune; must retrain from scratch)
- Cannot train embedding layers (bad for sparse categorical features)

#### 4. GBDT + LR -- Facebook's Classic Approach

**How it works:**
1. Train GBDT to learn the task
2. Use GBDT for feature extraction/selection (not prediction)
3. Feed extracted features + original features into LR

**Feature Selection:** Use decision trees to select most important features.
**Feature Extraction:** Use leaf node indices as new binary features.

**Pros:** GBDT-produced features are more predictive than originals.

**Cons:**
- Still cannot capture complex interactions
- Continual learning is slow (GBDT fine-tuning takes time)

#### 5. Neural Networks -- Getting Serious

**Option A: Single NN** -- Standard feedforward network on concatenated features.
**Option B: Two-Tower** -- Separate user encoder and ad encoder; similarity = click probability.

**Cons for Ad Click Prediction:**
- Sparse features mean not enough data points for NN to learn effectively
- Difficult to capture all pairwise interactions with large feature counts

#### 6. Deep & Cross Network (DCN) -- Google's Approach (2017)

Two parallel networks:
- **Deep network:** DNN that learns complex, generalizable features
- **Cross network:** Automatically captures feature interactions (learns crosses)

Outputs are concatenated for final prediction. Available in stacked or parallel variants.

**Pros:** Automatically learns feature crosses (no manual engineering).
**Cons:** Cross network only models certain interactions; may miss some.

#### 7. Factorization Machines (FM) -- The Elegant Solution

**The Formula:**
```
y_hat(x) = w_0 + sum_i(w_i * x_i) + sum_i(sum_j(<v_i, v_j> * x_i * x_j))
            |___ bias ___|  |____ LR term ____|  |________ pairwise interactions ________|
```

**How it works:**
- First two terms = logistic regression (linear combination)
- Third term = dot product of learned embeddings for every pair of features
- Automatically models ALL pairwise feature interactions

**Pros:** Efficient with sparse data, captures all pairwise interactions.
**Cons:** Cannot learn higher-order interactions (only pairwise).

#### 8. Deep Factorization Machines (DeepFM) -- The Real Rocket

Combines the best of both worlds:
- **FM component:** Captures low-level pairwise feature interactions
- **DNN component:** Captures sophisticated higher-order features

This is widely used in production at major tech companies.

**Optional upgrade:** GBDT + DeepFM (GBDT transforms features first, then DeepFM processes them). This has won ad prediction competitions but slows continual learning.

### Recommended Interview Answer

> "I would start with LR as a baseline, then experiment with DCN and DeepFM, as both are widely used in the tech industry."

---

## Evaluation

### Offline Metrics

#### Cross-Entropy (CE)

```
CE = -sum_i [y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i)]
```

- Measures how close predicted probabilities are to ground truth
- CE = 0 means perfect predictions
- **Lower CE = better model**
- Also used as the training loss function

#### Normalized Cross-Entropy (NCE)

```
NCE = CE(ML model) / CE(Simple baseline)
```

Where the simple baseline always predicts the background CTR (average CTR in training data).

- **NCE < 1:** Model outperforms the baseline (good!)
- **NCE >= 1:** Model is no better than always predicting the average CTR (bad!)

This is important because raw CE can be misleading when CTR varies across datasets.

### Online Metrics

| Metric | Formula | Why It Matters |
|--------|---------|---------------|
| **CTR** | clicked_ads / shown_ads | Directly related to revenue |
| **Conversion Rate** | conversions / impressions | Advertiser satisfaction (they pay for results) |
| **Revenue Lift** | % revenue increase over time | The ultimate business metric |
| **Hide Rate** | hidden_ads / shown_ads | Measures false positives (irrelevant ads shown) |

---

## Serving Architecture

### Three Pipelines

The production system has three key pipelines:

#### 1. Data Preparation Pipeline

Two types of feature computation:

**Batch Features (Static):**
- Change rarely (e.g., ad image, ad category)
- Computed periodically (every few days/weeks) via batch jobs
- Stored in a feature store for fast retrieval

**Online Features (Dynamic):**
- Change frequently (e.g., impression count, click count)
- Computed at query time
- Must be fast (<10ms)

#### 2. Continual Learning Pipeline

- Fine-tunes the model on new training data continuously
- Evaluates the new model against metrics
- Deploys the model only if it improves metrics
- Ensures the prediction pipeline always uses the most up-to-date model

Even a 5-minute delay in model updates can damage performance!

#### 3. Prediction Pipeline

```
User Query
    |
    v
[Candidate Generation] -- Uses advertiser targeting criteria (age, gender, country)
    |                      Narrows millions of ads to small subset
    v
[Ranking Service]       -- Fetches features from feature store + online computation
    |                      Uses model to predict P(click) for each candidate
    v
[Re-Ranking Service]    -- Applies business logic and heuristics
    |                      e.g., increase ad diversity, remove duplicates
    v
Top K Ads (ranked by click probability)
```

Key design decisions:
- **Online prediction only** (cannot use batch prediction because of dynamic features)
- **Two-stage architecture**: candidate generation (fast, coarse) then ranking (slow, precise)
- Feature store shared between training and serving to avoid training-serving skew

---

## Advanced Talking Points

These are topics to mention if time remains in the interview:

### 1. Data Leakage
In ranking and recommendation systems, it is critical to avoid data leakage. For example, using future click data to predict current clicks. Always ensure temporal ordering in train/test splits.

### 2. Model Calibration
In ad click prediction, calibration is essential because predicted probabilities feed directly into the auction. A model that predicts 0.3 should mean 30% of those impressions actually get clicked.

Calibration techniques:
- **Platt scaling:** Fit a sigmoid on validation set
- **Isotonic regression:** Non-parametric calibration
- **Temperature scaling:** Single parameter to sharpen/soften predictions

### 3. Position Bias
Ads shown in higher positions get more clicks regardless of relevance. Must account for this in training data.

### 4. Delayed Feedback
Users might click an ad hours after seeing it. Need attribution windows and delayed feedback models.

### 5. Cold Start
New ads and new users have no interaction history. Solutions include content-based features and exploration strategies.

---

## Interview Cheat Sheet

### 30-Second Pitch

> "Ad click prediction estimates P(click | user, ad, context) to maximize revenue. I would frame it as pointwise LTR with binary classification. Key features are ad IDs/categories (embedded), user demographics/history, and contextual signals. I would start with LR as a baseline, then use DeepFM which combines factorization machines for pairwise interactions with a DNN for higher-order features. For evaluation, I would use NCE offline and CTR/revenue lift online. The serving architecture uses candidate generation, ranking, and re-ranking with continual learning to keep the model fresh."

### Common Follow-Up Questions

| Question | Key Points |
|----------|-----------|
| Why not just use a deep neural network? | Sparse features, hard to capture all pairwise interactions |
| How do you handle cold-start ads? | Content features, exploration, multi-armed bandits |
| What about position bias? | Include position as feature in training, use inverse propensity weighting |
| How do you ensure calibration? | Platt scaling, isotonic regression, NCE metric |
| Why is continual learning important? | User interests and ad inventory change rapidly; 5-min delay hurts |
| How do you handle the sparse feature space? | Embeddings, hashing trick, factorization machines |

---

## Notebooks in This Module

| Notebook | Topics |
|----------|--------|
| `01_ad_click_system_design.ipynb` | Complete system design walkthrough with code |
| `02_feature_engineering_deep_dive.ipynb` | Encoding, crosses, real-time vs batch features |
| `03_deep_ctr_models.ipynb` | LR, DeepFM, DCN, DIN with PyTorch implementations |
| `04_interview_walkthrough.ipynb` | Full mock interview simulation |

---

## References

1. Real-time ML models in production at scale (continual learning)
2. AdTech fundamentals (beyond ML scope)
3. SimCLR -- A Simple Framework for Contrastive Learning of Visual Representations
4. Feature crossing techniques
5. Decision tree based feature generation
6. DCN -- Deep & Cross Network for Ad Click Predictions (Google, 2017)
7. DCN V2 -- Improved Deep & Cross Network
8. DCN architecture details
9. Factorization Machines (Rendle, 2010)
10. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
11. GBDT + DeepFM competition winners
12. Data leakage in ranking systems
13. Avoiding data leakage in recommendation systems
