# Similar Listings on Vacation Rental Platforms

## ML System Design Interview Module

> **The Simple Version**: Imagine you are looking at a treehouse on Airbnb and the website says "you might also like these other cool treehouses" -- how does the computer know which ones are similar? That is the exact problem we are solving here.

---

## Table of Contents

1. [What is Similar Listing Recommendation?](#what-is-similar-listing-recommendation)
2. [Clarifying Requirements](#clarifying-requirements)
3. [Framing the Problem as an ML Task](#framing-the-problem-as-an-ml-task)
4. [Data Preparation](#data-preparation)
5. [Model Development](#model-development)
6. [Training Data Construction (Negative Sampling)](#training-data-construction)
7. [Loss Function Design](#loss-function-design)
8. [Improving the Loss Function](#improving-the-loss-function)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Serving Architecture](#serving-architecture)
11. [Additional Talking Points](#additional-talking-points)
12. [Notebook Index](#notebook-index)

---

## What is Similar Listing Recommendation?

**Simple explanation**: When you look at a vacation rental (like a beach house on Airbnb), the website shows you other homes that are similar -- maybe in the same neighborhood, with a similar price, or with the same number of beds. The system needs to figure out which homes are "similar" without anyone manually telling it.

**Technical explanation**: Similar listing recommendation is a **session-based recommendation system** that, given a listing a user is currently viewing, produces a ranked list of other listings the user is likely to click on next. Unlike traditional recommendation systems that model long-term user preferences, this system focuses on **short-term intent** derived from the user's current browsing session.

Platforms that use this pattern:
- **Airbnb**: Similar accommodation listings
- **Amazon**: Similar products ("Customers who viewed this also viewed...")
- **Expedia**: Similar travel experiences
- **Vrbo**: Similar vacation rentals
- **Instagram**: Similar accounts (Explore feature)

---

## Clarifying Requirements

In an interview, always start by asking clarifying questions. Here is the canonical set for this problem:

| Question | Answer |
|----------|--------|
| What is the business objective? | Increase the number of bookings |
| What is the definition of "similarity"? | Same neighborhood, city, price range, etc. |
| Are results personalized per user? | For simplicity, treat logged-in and anonymous users equally |
| How many listings on the platform? | ~5 million |
| What data can we use for training? | User-listing interactions only (not listing attributes) |
| Cold start -- new listings? | New listings can appear in recommendations after 1 day |

**Summary of the problem statement**: Design a "similar listings" feature for a vacation rental platform. Input: a listing the user is currently viewing. Output: a ranked list of similar listings. Works for both anonymous and logged-in users. ~5M listings. Business goal: increase bookings.

---

## Framing the Problem as an ML Task

### Defining the ML Objective

**Simple explanation**: If someone clicks on a beach house, then clicks on another beach house nearby, then another -- the computer notices a pattern. It learns that these houses are "similar" because people keep clicking on them in the same browsing session. The ML goal is to predict: "Given the house you are looking at right now, which house will you click on next?"

**Technical explanation**: The ML objective is to accurately predict which listing the user will click next, given the listing they are currently viewing. We leverage the observation that listings clicked sequentially in a session typically share characteristics (location, price range, amenities).

### System Input and Output

```
Input:  A listing the user is currently viewing (listing ID)
Output: A ranked list of listings, sorted by P(user clicks on them)
```

### Why Session-Based Recommendation?

**Simple explanation**: Traditional recommendation systems are like a friend who has known you for years and knows your general taste. Session-based systems are like a smart store assistant who watches what you are looking at RIGHT NOW and suggests similar things in the moment.

**Technical depth**:

| Aspect | Traditional RecSys | Session-Based RecSys |
|--------|-------------------|---------------------|
| User interest model | Long-term, context-independent | Short-term, context-dependent |
| Interest dynamics | Stable, changes slowly | Dynamic, evolves fast |
| Key signal | Historical interactions (months/years) | Recent browsing session (minutes/hours) |
| Goal | Learn generic user interests | Understand current intent |
| Best for | Netflix movie recs, Spotify playlists | Airbnb similar listings, e-commerce "also viewed" |

For vacation rental search, recent clicks are far more informative than clicks from months ago. A user searching for "beachfront condos in Miami" right now does not benefit from their search for "ski cabins in Aspen" from last winter.

### The Embedding Approach

**Simple explanation**: Think of every listing as a dot on a huge map. But instead of a geographic map, it is a "similarity map." Houses that people tend to browse together get placed close together on this map. To find similar listings, we just look for the closest dots.

**Technical explanation**: We train a model that maps each listing into a dense embedding vector in a d-dimensional space (e.g., d=32 or d=64). Listings that frequently co-occur in users' browsing sessions will have embedding vectors in close proximity. To recommend similar listings, we perform nearest neighbor search in this embedding space.

This approach is directly inspired by **Word2Vec** (Mikolov et al., 2013), where words that appear in similar contexts get similar embeddings. Here, instead of words in sentences, we have listings in browsing sessions.

---

## Data Preparation

### Available Data

**1. Users Table**

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Unique user identifier |
| name | string | User name |
| age | int | User age |
| ... | ... | Other user attributes |

**2. Listings Table**

| Column | Type | Description |
|--------|------|-------------|
| listing_id | string | Unique listing identifier |
| host_id | string | Owner of the listing |
| price | float | Nightly price |
| num_beds | int | Number of beds |
| city | string | City location |
| neighborhood | string | Neighborhood |
| ... | ... | Other listing attributes |

**3. User-Listing Interactions Table**

| Column | Type | Description |
|--------|------|-------------|
| user_id | string | Who interacted |
| listing_id | string | Which listing |
| interaction_type | enum | impression / click / booking |
| timestamp | datetime | When it happened |

### Feature Engineering: Search Sessions

**Simple explanation**: Imagine you are house-hunting online. You click on House A, then House B, then House C, and finally you book House D. That whole sequence (A -> B -> C -> D) is called a "search session." We collect millions of these sessions from all users.

**Technical explanation**: A **search session** is defined as a sequence of clicked listing IDs, followed by an eventually booked listing, without interruption. We extract these from the interaction data.

Example session: `[L1, L2, L3, L4, L5_booked]`

The user clicked through listings L1-L4, then booked L5. This entire sequence becomes one training example.

**Important design decision**: The model only uses browsing history (co-occurrence of listings in sessions). It does NOT use listing attributes (price, location) or user attributes (age, location) during training. The embeddings implicitly learn these relationships from behavioral data.

---

## Model Development

### Model Selection

**Simple explanation**: We use a relatively simple neural network -- not a giant deep one. It is similar to the famous Word2Vec model but for listings instead of words.

**Technical explanation**: We use a **shallow neural network** (similar to Skip-gram Word2Vec architecture) to learn listing embeddings. The architecture maps each listing to a dense vector. Key hyperparameters (embedding dimension, number of layers, context window size) are tuned via experiments.

### Training Process

1. **Initialize** listing embeddings as random vectors
2. **Slide a window** across each search session
3. For each window position:
   - The **center listing** is the target
   - **Context listings** (within the window) should have similar embeddings
   - **Non-context listings** should have dissimilar embeddings
4. Update embeddings via backpropagation
5. **Retrain daily** to adapt to new listings and interactions

---

## Training Data Construction

### Negative Sampling

**Simple explanation**: To teach the computer what "similar" means, we also need to show it what "NOT similar" looks like. So we create two kinds of pairs:
- **Positive pairs**: "These two listings appeared near each other in someone's browsing session, so they are similar"
- **Negative pairs**: "This listing was randomly chosen and has nothing to do with the other one, so they are NOT similar"

**Technical explanation**: We use the **negative sampling** technique from Word2Vec. For each search session:

1. Slide a context window across the session
2. **Positive pairs** (label=1): (center listing, each context listing within the window)
3. **Negative pairs** (label=0): (center listing, randomly sampled listings outside the window)

Example with window size = 2 and session `[A, B, C, D, E]`:
- When center = C: Positive pairs = (C,A), (C,B), (C,D), (C,E). Negative pairs = (C, random1), (C, random2), ...

---

## Loss Function Design

### Basic Loss Function

**Simple explanation**: The loss function is like a score that tells the computer how wrong it is. If the computer says two similar houses are far apart (bad!) or two unrelated houses are close together (also bad!), the loss goes up. Training makes the loss go down.

**Technical explanation**: The loss computation follows three steps:

1. **Compute distance**: Calculate dot product between two listing embedding vectors
2. **Sigmoid activation**: Convert distance to probability in [0, 1]
3. **Cross-entropy loss**: Standard binary classification loss between predicted probability and ground truth label

The loss formula:

```
L = -sum_{(c,p) in D_pos} log(sigmoid(v_c . v_p)) - sum_{(c,n) in D_neg} log(sigmoid(-v_c . v_n))
```

Where:
- `c` = center listing, `p` = positive (context) listing, `n` = negative listing
- `v_c`, `v_p`, `v_n` = embedding vectors
- `D_pos` = set of positive pairs (center, context) -- push together
- `D_neg` = set of negative pairs (center, random) -- push apart

---

## Improving the Loss Function

The basic loss has two shortcomings. Both are critical interview talking points.

### Problem 1: No Signal from Booked Listings

**Simple explanation**: The basic training only teaches the model "these houses were clicked near each other." But clicking is not the same as booking. We want to especially push each listing close to the one the user actually booked, because booking = real success.

**Technical explanation**: During training, the center listing embedding is pushed toward context listings (clicked neighbors), but NOT specifically toward the eventually booked listing. This produces embeddings optimized for click prediction, not booking prediction.

**Solution -- Global Context**: Treat the eventually booked listing as a **global context** that remains in every positive pair throughout the entire session. As the window slides, listings enter and leave the context, but the booked listing always stays.

For each position in the window, add an extra positive pair: `(center_listing, booked_listing)`.

### Problem 2: Easy Negatives from Different Regions

**Simple explanation**: If you randomly pick a "not similar" listing, it will probably be in a completely different city. That is too easy -- of course a beach house in Miami is different from a cabin in Alaska. The hard question is: among all the beach houses in Miami, which ones are similar and which are not?

**Technical explanation**: Random negative sampling produces negatives predominantly from different geographic regions. The model easily learns to distinguish cross-region listings but fails to differentiate listings within the same region.

**Solution -- Hard Negatives from Same Region**: For each center listing, sample a negative listing from the **same neighborhood** that is NOT in the center listing's context window. These "hard negatives" force the model to learn fine-grained distinctions within the same area.

### Updated Loss Function

```
L = -sum_{D_pos} log(sig(v_c . v_p))        # Positive pairs (clicked neighbors)
    -sum_{D_neg} log(sig(-v_c . v_n))        # Random negative pairs
    -sum_{D_book} log(sig(v_c . v_book))     # Booked listing as global context
    -sum_{D_hard} log(sig(-v_c . v_hard))    # Hard negatives from same region
```

Four components:
1. Push center toward clicked context listings
2. Push center away from random listings
3. Push center toward the eventually booked listing (global context)
4. Push center away from same-region non-context listings (hard negatives)

---

## Evaluation Metrics

### Offline Metrics

**Average Rank of the Eventually Booked Listing**

**Simple explanation**: After the model is trained, we test it like this: for a real browsing session where someone eventually booked a listing, we ask the model to rank all listings by similarity to the first click. If the model puts the booked listing near the top of that ranking, the model is good.

**Technical explanation**: For each session in the validation set:
1. Take the first clicked listing
2. Compute similarities to all other listings in the embedding space
3. Rank listings by similarity
4. Record the rank position of the eventually booked listing
5. Average across all sessions

Lower average rank = better model. If the new model ranks booked listings higher (lower rank number) than the old model, the new model has learned better embeddings.

### Online Metrics

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Click-Through Rate (CTR)** | % of users who see recommended listings and click on them | Measures user engagement. Higher CTR means more exploration, which increases booking probability |
| **Session Book Rate** | % of search sessions that end in a booking | Directly tied to business objective (increase bookings). Higher session book rate = more revenue |

CTR alone is not sufficient because clicks do not equal bookings. Session book rate is the primary business metric.

---

## Serving Architecture

### System Overview

The serving system has three main pipelines:

```
+------------------+     +------------------+     +--------------------+
|  Training        |     |  Indexing         |     |  Prediction        |
|  Pipeline        |---->|  Pipeline         |---->|  Pipeline          |
+------------------+     +------------------+     +--------------------+
  - Daily retraining      - Pre-compute all        - Embedding fetcher
  - New interactions        listing embeddings     - Nearest neighbor
  - Fine-tune model       - Update index table     - Re-ranking service
```

### Training Pipeline

- **Frequency**: Daily retraining
- **Input**: New user-listing interactions, new listings
- **Output**: Updated model with new listing embeddings
- **Purpose**: Ensures the model adapts to new listings and changing user behavior

### Indexing Pipeline

**Simple explanation**: Instead of computing similarities on the fly (which would be slow with 5 million listings), we pre-calculate and store every listing's "similarity fingerprint" (embedding) in a big lookup table.

**Technical explanation**: After training, pre-compute embeddings for all 5M listings and store them in an **index table** (e.g., FAISS, ScaNN, or Annoy). The pipeline:
- Adds embeddings for new listings
- Re-computes all embeddings when a new model is deployed
- Updates the index table accordingly

### Prediction Pipeline

The prediction pipeline has three components:

#### 1. Embedding Fetcher Service

**If the listing has been seen during training**: Directly fetch its pre-computed embedding from the index table.

**If the listing is new (cold start)**:
- Use a heuristic: take the embedding of a **geographically nearby listing**
- Once enough interaction data accumulates (typically within 1 day), the daily training pipeline will learn the new listing's embedding

#### 2. Nearest Neighbor Service

**Simple explanation**: Once we have the "fingerprint" of the listing the user is looking at, we search through all 5 million fingerprints to find the closest matches. Since checking all 5 million would be too slow, we use a shortcut method that is almost as accurate but much faster.

**Technical explanation**: Compute similarity between the current listing's embedding and all other listings. Use **Approximate Nearest Neighbor (ANN)** search for efficiency:
- **FAISS** (Facebook AI Similarity Search)
- **ScaNN** (Google)
- **HNSW** (Hierarchical Navigable Small World graphs)

ANN trades a tiny amount of accuracy for massive speed improvements (e.g., searching 5M vectors in <10ms instead of seconds).

#### 3. Re-Ranking Service

**Simple explanation**: The nearest neighbor search gives us a raw list of similar listings, but we need to apply some common-sense rules. For example, remove listings that are too expensive for the user's filter, or remove listings in a different city.

**Technical explanation**: Post-processing layer that applies:
- **User filters**: Remove listings above the user's price filter
- **Geographic constraints**: Remove listings in different cities
- **Business rules**: Ensure diversity, remove unavailable listings
- **Display logic**: Final ordering for the UI

---

## Additional Talking Points

These are bonus topics to discuss if time remains in the interview:

1. **Positional Bias**: Listings shown at the top of results get more clicks regardless of true relevance. Address this with position-aware training or inverse propensity weighting.

2. **Random Walk Approaches**: Compare session-based embedding to graph-based approaches using random walks on a listing interaction graph. Random Walks with Restart (RWR) can capture both local and global graph structure.

3. **In-Session Personalization**: For logged-in users, incorporate their longer-term booking history to personalize the session-based recommendations. Airbnb uses this to boost listings similar to users' past bookings.

4. **Seasonality**: Vacation rentals are highly seasonal (beach houses in summer, ski cabins in winter). Incorporate temporal features or train separate seasonal models.

---

## Notebook Index

| Notebook | Topics |
|----------|--------|
| [01_similar_listing_system_design.ipynb](01_similar_listing_system_design.ipynb) | Problem definition, metrics, architecture, data pipeline, feature engineering, model architecture |
| [02_embedding_techniques.ipynb](02_embedding_techniques.ipynb) | Listing2Vec, session-based embeddings, negative sampling, loss function improvements |
| [03_ranking_and_serving.ipynb](03_ranking_and_serving.ipynb) | Re-ranking, business rules, serving architecture, cold start handling |
| [04_interview_walkthrough.ipynb](04_interview_walkthrough.ipynb) | Complete mock interview walkthrough from start to finish |

---

## Key References

1. Instagram Explore embeddings for account recommendations
2. Airbnb listing embeddings for similar listings (KDD 2018 paper: "Real-time Personalization using Embeddings for Search Ranking at Airbnb")
3. Word2Vec (Mikolov et al., 2013) - the foundational embedding approach
4. Negative Sampling technique for efficient embedding training
5. Positional bias in recommendation systems
6. Random walk-based graph embeddings (DeepWalk, Node2Vec)
7. Random Walks with Restart (RWR) for recommendation
8. Seasonal modeling in recommendation systems
