# Event Recommendation System -- ML System Design Interview Guide

## What Is This About?

Imagine your phone could tell you about the coolest events happening near you this weekend -- concerts, basketball games, art shows, comedy nights. How does it know what **you** would like? That is exactly what an event recommendation system does. It is like having a super-smart friend who knows your taste in events, knows where you live, knows what your friends are doing, and picks out the perfect events just for you.

This guide covers the complete design of an **Eventbrite-style event recommendation system**, based on the ByteByteGo ML System Design Interview chapter. It is written so a 12-year-old can follow along, but with every staff-level technical detail intact.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Clarifying Requirements](#clarifying-requirements)
3. [Framing the ML Problem](#framing-the-ml-problem)
4. [Data Preparation](#data-preparation)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Serving Architecture](#serving-architecture)
9. [Advanced Talking Points](#advanced-talking-points)
10. [Interview Cheat Sheet](#interview-cheat-sheet)

---

## The Big Picture

**Business context:** Eventbrite is a platform where people create, browse, and register for events. The recommendation system personalizes what each user sees, so they find events they actually want to attend -- which means more ticket sales for the platform.

**Why is this hard compared to, say, Netflix recommendations?**

Think about it this way. A movie stays on Netflix forever. You can watch it today or next year. But a concert? It happens once, on one date, in one place, and then it is gone. This makes event recommendation fundamentally different:

- **Events are ephemeral** -- they expire after they happen.
- **Cold-start is constant** -- new events pop up every day with zero interaction history.
- **Location matters enormously** -- a concert in Tokyo is useless to someone in Chicago.
- **Time matters** -- recommending a concert that starts in 20 minutes and is 2 hours away is unhelpful.
- **Social dynamics** -- you are more likely to go if your friends are going.

---

## Clarifying Requirements

In an interview, always start by asking clarifying questions. Here is the complete set from the chapter:

| Question | Answer |
|----------|--------|
| Business objective? | Increase ticket sales |
| Only events, or also hotels/restaurants? | Only events |
| Events are one-time, ephemeral? | Yes -- once finished, no more registrations |
| What event attributes exist? | Description, price, location, date/time, category |
| Annotated training data? | No hand-labeled data; use interaction logs |
| User location available? | Yes (users agree to share) |
| Can users be friends? | Yes, bidirectional friendships |
| Can users invite others? | Yes |
| RSVP or registration? | Registration only (simplified) |
| Free or paid events? | Both |
| Scale? | ~1 million events/month, ~1 million DAU |
| External APIs? | Google Maps / map services available for distance/travel time |

**Summary statement:** Design a personalized event recommendation system. Input = a user. Output = top-k events ranked by relevance. Primary goal = maximize event registrations (which drives ticket sales).

---

## Framing the ML Problem

### ML Objective

Translate "increase ticket sales" into: **maximize the number of event registrations**.

### Input / Output

- **Input:** A user (with their profile, location, history)
- **Output:** Top-k events ranked by predicted relevance

### Approach: Learning to Rank (LTR)

There are three broad approaches to recommendation:

1. **Simple rules** -- recommend popular events (good baseline, poor personalization)
2. **Embedding-based** -- content-based or collaborative filtering
3. **Ranking problem** -- Learning to Rank (LTR)

We choose LTR, specifically the **pointwise** approach using **binary classification**.

#### The Three LTR Flavors

| Approach | How It Works | Examples |
|----------|-------------|----------|
| **Pointwise** | Predict relevance of each item independently | Logistic Regression, NN |
| **Pairwise** | Given two items, predict which is more relevant | RankNet, LambdaRank, LambdaMART |
| **Listwise** | Predict optimal ordering of an entire list | SoftRank, ListNet, AdaRank |

Pairwise and listwise are more accurate but harder to implement. For this design, we use **pointwise binary classification**: for each (user, event) pair, predict the probability the user will register.

> **12-year-old version:** Imagine you have a stack of event flyers. For each flyer, you ask: "Would this specific person sign up? Yes or no?" You score them all and show the ones most likely to get a "yes."

---

## Data Preparation

### Raw Data Available

**Users Table**
| Field | Example |
|-------|---------|
| ID | 42 |
| Username | alice_m |
| Age | 28 |
| Gender | F |
| City | Miami |
| Country | US |

**Events Table**
| Field | Example |
|-------|---------|
| ID | 101 |
| Host User ID | 5 |
| Category / Subcategory | Music / Concert |
| Description | "Dua Lipa Tour in Miami" |
| Price | $200-$900 |
| Location | Miami, FL |
| Date/Time | 2024-03-15 20:00 |

**Friendship Table**
| User ID 1 | User ID 2 | Timestamp |
|-----------|-----------|-----------|
| 28 | 3 | 1658451341 |

**Interactions Table**
| User ID | Event ID | Interaction Type | Value |
|---------|----------|-----------------|-------|
| 4 | 18 | Impression | - |
| 4 | 18 | Register | Confirmation# |
| 4 | 18 | Invite | User 9 |

### Constructing the Training Dataset

For each (user, event) pair from interaction data:
- **Label = 1** if the user registered for the event
- **Label = 0** if the user saw the event (impression) but did not register

**Class imbalance problem:** Users explore tens or hundreds of events before registering for one, so negatives vastly outnumber positives.

**Solutions:**
- Focal loss or class-balanced loss
- Undersample the majority class

---

## Feature Engineering

This is the heart of the system. Because events are ephemeral and cold-start is constant, we rely heavily on well-crafted features rather than interaction history.

### 1. Location-Related Features

> **12-year-old version:** "Is this event close enough for me to actually get there?"

| Feature | Description |
|---------|-------------|
| **Walk score** | 0-100 measure of walkability (from Google Maps / OSM), bucketized into 5 categories |
| **Walk score similarity** | Difference between event's walk score and user's average walk score of past events |
| **Transit score / Bike score** | Similar accessibility metrics + their similarities |
| **Same country?** | Binary: 1 if user and event in same country, else 0 |
| **Same city?** | Binary: 1 if user and event in same city, else 0 |
| **Distance bucket** | Distance bucketized: 0 (<1mi), 1 (1-5mi), 2 (5-20mi), 3 (20-50mi), 4 (50-100mi), 5 (100+mi) |
| **Distance similarity** | Difference between this event's distance and user's average past-event distance |

### 2. Time-Related Features

> **12-year-old version:** "Is this event at a time that works for me, and is there enough time to plan?"

| Feature | Description |
|---------|-------------|
| **Remaining time bucket** | Time until event: 0 (<1hr), 1 (1-2hr), ..., 8 (7+ days). One-hot encoded. |
| **Remaining time similarity** | Difference from user's average registration-to-event time |
| **Estimated travel time** | From external API, bucketized |
| **Travel time similarity** | Compared to user's historical average |
| **Day-of-week profile** | Vector of size 7: rate of past attendance per day (e.g., user never attends Monday events) |
| **Hour-of-day profile** | Similar vector for time-of-day preference |
| **Day/hour similarity** | How well the event's timing matches user's profile |

### 3. Social-Related Features

> **12-year-old version:** "Are my friends going? Did someone invite me?"

| Feature | Description |
|---------|-------------|
| **Number registered** | Total users registered for this event |
| **Registration ratio** | Registered users / impressions |
| **Registered user similarity** | Compared to user's previously registered events |
| **Friends registered** | Number of user's friends registered for this event |
| **Friend registration ratio** | Registered friends / total friends |
| **Friend registration similarity** | Compared to past events |
| **Invitations from friends** | Number of friends who invited user to this event |
| **Invitations from others** | Number of non-friend users who invited user |
| **Host is friend?** | Binary: 1 if event host is user's friend |
| **Past events by host** | How often user attended this host's previous events |

### 4. User-Related Features

| Feature | Description |
|---------|-------------|
| **Gender** | One-hot encoded |
| **Age bucket** | Bucketized into categories, one-hot encoded |

### 5. Event-Related Features

| Feature | Description |
|---------|-------------|
| **Price bucket** | 0: Free, 1: $1-99, 2: $100-499, 3: $500-1999, 4: $2000+ |
| **Price similarity** | Difference from user's average past event price |
| **Description similarity** | TF-IDF cosine similarity between event description and user's past event descriptions |

### Key Feature Engineering Discussion Points

- **Batch vs. streaming features:** Static features (age, gender, description) computed periodically; dynamic features (registered count, remaining time) computed in real-time.
- **Feature computation efficiency:** Instead of computing distance as a feature, pass both locations to the model and let it learn useful representations.
- **Decay factor:** Give more weight to recent user interactions.
- **Embedding learning:** Convert users and events into embedding vectors as alternative features.
- **Bias awareness:** Features from user attributes (age, gender) can create discrimination -- important to monitor.

---

## Model Development

### Model Candidates

#### 1. Logistic Regression

**Pros:**
- Fast inference (weighted sum of features)
- Fast training
- Interpretable (feature weights show importance)
- Good when data is linearly separable

**Cons:**
- Cannot learn non-linear relationships
- Suffers from multicollinearity
- Too simple for complex feature interactions in our system

#### 2. Decision Tree

**Pros:**
- Fast training and inference
- No data normalization needed
- Interpretable (visualize the tree)

**Cons:**
- Axis-parallel decision boundaries (suboptimal)
- Overfitting (sensitive to small data changes)
- Rarely used alone in practice

#### 3. Ensemble Methods

**Bagging (Random Forest):**
- Trains multiple trees in parallel on data subsets
- Reduces variance (overfitting)
- Does not increase training time much (parallel)
- Limitation: does not help with high bias (underfitting)

**Boosting (AdaBoost, XGBoost, Gradient Boost):**
- Trains weak classifiers sequentially, each fixing previous mistakes
- Reduces both bias and variance
- Slower training/inference (sequential)
- Preferred over bagging in practice

#### 4. GBDT (Gradient-Boosted Decision Trees)

**Pros:**
- No data preparation needed
- Reduces both variance and bias
- Works well with structured/tabular data
- XGBoost variant strong in ML competitions

**Cons:**
- Many hyperparameters to tune
- Does not work well on unstructured data (images, text)
- **Not suitable for continual learning** (must retrain from scratch)

#### 5. Neural Network

**Pros:**
- Supports continual learning (fine-tune on new data)
- Works with unstructured data
- Highly expressive (non-linear decision boundaries)

**Cons:**
- Computationally expensive to train
- Sensitive to input data (needs normalization, scaling)
- Requires large training data
- Black-box (not interpretable)

### Recommended Strategy

1. **Start with XGBoost** as baseline -- fast to implement, works well with structured features
2. **Graduate to Neural Network** for production -- continual learning is critical because:
   - Massive training data available from user interactions
   - Data likely not linearly separable
   - Need to adapt continuously to new events, users, and changing preferences

### Loss Function

**Binary cross-entropy** -- standard for binary classification.

---

## Evaluation Metrics

### Offline Metrics

| Metric | Verdict | Why |
|--------|---------|-----|
| Precision@k / Recall@k | Not ideal | Do not consider ranking quality |
| MRR (Mean Reciprocal Rank) | Not ideal | Focuses on rank of *first* relevant item; we have multiple relevant events |
| nDCG | Good but... | Best for non-binary relevance scores |
| **mAP (Mean Average Precision)** | **Best fit** | Works with binary relevance (registered or not), considers ranking quality |

### Online Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **CTR** | clicked events / impressions | Are users clicking on recommendations? |
| **Conversion rate** | registrations / impressions | Are users actually registering? (More meaningful than CTR) |
| **Bookmark rate** | bookmarked events / impressions | Interest signal even without registration |
| **Revenue lift** | Revenue increase from recommendations | Direct business impact |

> **Why not just CTR?** Some events are clickbait -- high clicks but low registrations. Conversion rate is the more meaningful metric.

---

## Serving Architecture

The system has two main pipelines:

### 1. Online Learning Pipeline

Because events are ephemeral and cold-start is constant, the model must be **continuously fine-tuned** with new data.

This pipeline:
- Ingests new interaction data (registrations, clicks, new events)
- Continuously trains updated models
- Evaluates new models
- Deploys them to production

### 2. Prediction Pipeline

When a user opens the app:

**Step 1: Event Filtering**
- Input: the query user
- Narrows 1 million events down to hundreds of candidates
- Uses simple rules: location filter, category filter, user preferences
- Example: user sets "concerts only" filter

**Step 2: Ranking Service**
- Takes user + candidate events from filtering
- Computes features for each (user, event) pair
- Runs the model to predict registration probability
- Sorts events by predicted probability
- Returns top-k events

**Feature Computation:**
- **Static features** (age, gender, event description) retrieved from feature store
- **Dynamic features** (registered count, remaining time) computed in real-time

---

## Advanced Talking Points

These are topics an interviewer might probe deeper on:

1. **Bias in recommendations** -- Position bias (users click top results more), popularity bias (popular events get more exposure), demographic bias from user attributes.

2. **Feature crossing** -- Combining features (e.g., age x category) for more expressive representations.

3. **Diversity and freshness** -- Users want varied recommendations, not just the same category. Techniques: MMR (Maximal Marginal Relevance), category-aware re-ranking.

4. **Privacy and security** -- Using live location and personal attributes raises privacy concerns. Differential privacy, data minimization, anonymization.

5. **Two-sided marketplace fairness** -- Hosts are suppliers, users are demand. System should not only optimize for users; hosts need fair exposure too.

6. **Data leakage** -- When constructing datasets, ensure no future information leaks into training features (e.g., using registration count that includes the target user's own registration).

7. **Model update frequency** -- How often to retrain? Balance freshness vs. computational cost. Continual learning with NNs helps here.

---

## Interview Cheat Sheet

### 30-Second Elevator Pitch

"We build a pointwise Learning-to-Rank system that predicts the probability a user will register for each candidate event. We engineer features across five categories -- location, time, social, user, and event -- to handle the cold-start problem inherent in ephemeral events. We start with XGBoost as a baseline, then move to a neural network for continual learning. Events are filtered by location/rules first, then ranked. We evaluate offline with mAP and online with conversion rate and revenue lift."

### Key Differentiators From Standard Recommendation

1. Events are ephemeral (expire after they happen)
2. Constant cold-start (new events daily with no history)
3. Location is critical (not just user preference -- physical accessibility)
4. Time sensitivity (remaining time, travel time, day/hour preferences)
5. Strong social signals (friends attending, invitations)

### Framework: How to Walk Through This in 45 Minutes

| Time | Topic |
|------|-------|
| 0-5 min | Clarifying requirements |
| 5-10 min | Frame as ML problem (LTR, pointwise binary classification) |
| 10-20 min | Feature engineering (the star of this design) |
| 20-30 min | Model selection (LR -> Decision Tree -> GBDT -> NN) |
| 30-35 min | Evaluation (mAP offline, conversion rate online) |
| 35-42 min | Serving architecture (filtering + ranking, online learning) |
| 42-45 min | Advanced topics (bias, diversity, privacy, two-sided marketplace) |

---

## Files in This Module

| File | Description |
|------|-------------|
| `01_event_recommendation_system_design.ipynb` | Complete system design with code |
| `02_location_and_time_features.ipynb` | Deep dive into geospatial, temporal, social, and cold-start features |
| `03_ranking_and_personalization.ipynb` | Ranking models, diversity, notification optimization |
| `04_interview_walkthrough.ipynb` | Full mock interview simulation |
