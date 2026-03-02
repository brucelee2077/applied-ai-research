# Video Recommendation System - ML System Design Interview Guide

> **Source**: ByteByteGo ML System Design Interview, Chapter 6
> **Difficulty**: Staff-level ML Engineering
> **Interview Time**: 45 minutes

---

## Table of Contents

1. [What is Video Recommendation?](#what-is-video-recommendation)
2. [Clarifying Requirements](#clarifying-requirements)
3. [Framing the ML Problem](#framing-the-ml-problem)
4. [Types of Recommendation Systems](#types-of-recommendation-systems)
5. [Data Preparation](#data-preparation)
6. [Feature Engineering](#feature-engineering)
7. [Model Development](#model-development)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Serving Architecture](#serving-architecture)
10. [Interview Cheat Sheet](#interview-cheat-sheet)

---

## What is Video Recommendation?

**Simple explanation**: Imagine you open YouTube or Netflix. There are *billions* of videos, but the homepage only shows you maybe 20-50. How does the system pick *those specific videos* for you out of billions? That is the video recommendation problem. It is like having a friend who knows everything you have ever watched, what you liked, what you skipped, and who can instantly pick the perfect next video for you -- except this "friend" is a machine learning system that does it for hundreds of millions of users simultaneously.

**Why it matters**: Recommendation systems are the backbone of platforms like YouTube, Netflix, TikTok, and Spotify. They directly drive user engagement, retention, and revenue. YouTube reported that over 70% of watch time comes from recommended videos, not searches. Getting recommendations right is a billion-dollar problem.

**The core challenge**: You have ~10 billion videos and hundreds of millions of users. You need to serve personalized recommendations in under 200 milliseconds. You cannot score every video for every user -- that would be 10 billion x hundreds of millions of computations per request. You need a smart, multi-stage pipeline.

---

## Clarifying Requirements

In a real interview, always start by clarifying. Here is the typical dialogue:

| Question | Answer |
|----------|--------|
| What is the business objective? | Increase user engagement |
| What type of recommendation? | Homepage personalized videos (not "similar to current video") |
| Are users worldwide? | Yes, global users, videos in different languages |
| Can we use interaction data? | Yes -- clicks, watches, likes, impressions |
| Do playlists exist? | No (simplification) |
| How many videos? | ~10 billion |
| Latency requirement? | < 200 milliseconds |

**Key takeaway**: The system recommends personalized videos on the homepage, not "related videos" on a watch page. This is a harder problem because you have no immediate context signal (the user is not watching anything yet).

---

## Framing the ML Problem

### Defining the ML Objective

The business objective is "increase user engagement," but we need to translate that into a concrete ML objective. There are several options, each with trade-offs:

| ML Objective | Pros | Cons |
|---|---|---|
| **Maximize clicks** | Easy to measure | Leads to clickbait; users click but do not watch |
| **Maximize completed videos** | Good signal of satisfaction | Biased toward short videos |
| **Maximize total watch time** | Captures deep engagement | May favor addictive but low-quality content |
| **Maximize number of relevant videos** | Most control over quality signals | Requires defining "relevance" carefully |

**Best choice: Maximize relevant videos**. We define "relevant" as: a user explicitly likes the video OR watches at least half of it. This combines explicit feedback (likes) with implicit feedback (watch time) and avoids the pitfalls of pure click optimization or pure watch-time optimization.

**Simple analogy**: Imagine you are picking lunch for a friend. "Maximize clicks" is like picking food with the prettiest packaging -- they might grab it but hate the taste. "Maximize completed videos" is like only picking tiny snacks they can finish quickly. "Maximize relevant videos" is like picking food they will actually enjoy eating, based on what they have liked before.

### Input and Output

- **Input**: A user (their profile, history, context)
- **Output**: A ranked list of videos sorted by predicted relevance scores

### ML Category

This is a **hybrid filtering** problem that combines collaborative filtering and content-based filtering in a sequential pipeline.

---

## Types of Recommendation Systems

### Content-Based Filtering

**Simple explanation**: "If you liked ski videos before, here are more ski videos." The system looks at the *features of the videos* you enjoyed and finds similar ones.

**How it works**:
1. User A engaged with videos X and Y
2. Video Z is similar to X and Y (based on features like tags, title, category)
3. Recommend video Z to User A

**Pros**:
- Can recommend brand-new videos (no interaction data needed for the video)
- Captures unique interests of individual users

**Cons**:
- Difficult to discover new interests (if you only watched cooking videos, it will only recommend cooking)
- Requires manual feature engineering and domain knowledge

### Collaborative Filtering (CF)

**Simple explanation**: "People who are similar to you liked this video, so you might like it too." The system does not look at video features at all -- it only looks at who watched what.

**How it works**:
1. Find a user B who is similar to User A (they watched many of the same videos)
2. Find a video Z that User B watched but User A has not seen
3. Recommend video Z to User A

**Pros**:
- No domain knowledge needed (does not need to understand video content)
- Can discover new interest areas (you might get cooking recommendations even though you never searched for cooking, because similar users like cooking)
- Efficient to compute

**Cons**:
- **Cold-start problem**: Cannot recommend for new users or new videos with no interaction history
- Struggles with niche interests (hard to find similar users if your tastes are very unique)

### Comparison Table

| Aspect | Content-Based | Collaborative Filtering |
|--------|:---:|:---:|
| Handle new videos | Yes | No |
| Discover new interests | No | Yes |
| No domain knowledge needed | No | Yes |
| Efficiency | Lower | Higher |

### Hybrid Filtering (Our Choice)

**Simple explanation**: Why not use both? First, use collaborative filtering to quickly find a big pool of candidate videos. Then, use content-based filtering to carefully rank them using video features.

This is exactly what YouTube does in practice (described in their 2016 paper). The two methods are complementary -- CF finds candidates fast, content-based ranking refines quality.

---

## Data Preparation

### Available Data Sources

**1. Video Data**

| Field | Example | Type |
|-------|---------|------|
| Video ID | 1 | Categorical |
| Length | 28 seconds | Numerical |
| Manual Tags | "Dog, Family" | Text |
| Title | "Our lovely dog playing!" | Text |
| Likes | 138 | Numerical |
| Views | 5300 | Numerical |

**2. User Data**

| Field | Type |
|-------|------|
| User ID | Categorical |
| Username | Text |
| Age | Numerical |
| Gender | Categorical |
| City | Categorical |
| Country | Categorical |

**3. User-Video Interaction Data**

| User ID | Video ID | Interaction Type | Value |
|---------|----------|-----------------|-------|
| 4 | 18 | Like | - |
| 2 | 18 | Impression | 8 seconds |
| 2 | 6 | Watch | 46 minutes |
| 6 | 9 | Click | - |
| 9 | - | Search | "Basics of clustering" |
| 8 | 6 | Comment | "Amazing video. Thanks" |

This interaction data is the gold mine. It tells us not just *what* users watched, but *how* they interacted.

---

## Feature Engineering

### Video Features

| Feature | How to Prepare | Why It Matters |
|---------|---------------|----------------|
| **Video ID** | Embedding layer (learned during training) | Unique identifier for CF |
| **Duration** | Raw numerical value | Some users prefer short vs. long videos |
| **Language** | Embedding layer | Users prefer specific languages |
| **Tags** | Pre-trained CBOW embeddings | Describe video content |
| **Title** | Pre-trained BERT embeddings | Rich semantic representation |

### User Features

**Demographics**: Age, gender, language -- represented via embedding layers.

**Contextual Information**:
- **Time of day**: A software engineer might watch educational videos in the evening
- **Device**: Mobile users may prefer shorter videos
- **Day of week**: Weekend vs. weekday viewing habits differ

**Historical Interactions**:
- **Search history**: Use BERT to embed each query, then average all query embeddings into a fixed-size vector
- **Liked videos**: Map video IDs to embeddings, then average
- **Watched videos and impressions**: Same approach as liked videos

**Key engineering insight**: All variable-length histories (searches, liked videos, watched videos) are converted to fixed-size vectors by averaging their embeddings. This is a simple but effective approach.

---

## Model Development

### Model 1: Matrix Factorization

**Simple explanation**: Imagine a huge table where every row is a user and every column is a video. Each cell says whether that user liked that video. Most cells are empty (a user has only seen a tiny fraction of all videos). Matrix factorization tries to fill in the missing cells by discovering hidden patterns.

**How it works**:
1. Build a **feedback matrix** (users x videos) with 1 for positive interactions, 0 or empty for others
2. Decompose this matrix into two smaller matrices: **User embeddings** (U) and **Video embeddings** (V)
3. The product U x V approximates the original feedback matrix
4. To predict relevance: compute dot product between a user's embedding and a video's embedding

**Feedback matrix construction**: We combine explicit feedback (likes, shares) and implicit feedback (clicks, watch time) because:
- Explicit-only: Too sparse (few users bother to click "like")
- Implicit-only: Noisy (a click does not mean enjoyment)
- Combined: Best of both worlds

**Loss Function Choices**:

| Loss Function | Description | Problem |
|---|---|---|
| Sum over observed pairs only | Only penalizes errors on known interactions | Does not learn from missing data; all-ones matrix scores zero loss |
| Sum over all pairs | Treats unobserved as negative (zero) | Sparse matrix means negatives dominate; predictions collapse to zero |
| **Weighted combination** (our choice) | Weighted sum of observed + unobserved losses | Hyperparameter W balances the two; works well in practice |

**Optimization**:
- **SGD**: Standard gradient descent
- **WALS (Weighted Alternating Least Squares)**: Fix U, optimize V. Fix V, optimize U. Repeat. Converges faster and is parallelizable. This is our choice.

**Inference**: Dot product between user embedding and video embedding gives relevance score (e.g., 0.32 for user 2, video 5).

**Pros**: Fast training, fast serving (embeddings are static/precomputed)
**Cons**: Only uses interaction data (no user/video features), cannot handle new users (cold start)

### Model 2: Two-Tower Neural Network

**Simple explanation**: Think of two separate "expert analyzers." One expert studies everything about the user (age, watch history, search history, etc.) and summarizes the user as a single vector. The other expert studies everything about a video (title, tags, duration, etc.) and summarizes the video as a single vector. If the two vectors are close together in space, the video is relevant to the user.

**Architecture**:
- **User Tower**: Takes all user features (demographics + context + history) as input, outputs a user embedding vector
- **Video Tower**: Takes all video features (or just video ID for CF mode) as input, outputs a video embedding vector
- **Similarity**: Dot product or cosine similarity between the two embeddings

**Dataset Construction**:
- **Positive pairs**: User explicitly liked the video OR watched at least half
- **Negative pairs**: Random videos the user has not interacted with, or videos the user explicitly disliked
- **Important**: Dataset is imbalanced (many more negatives than positives) -- use techniques like negative sampling, class weighting, or oversampling

**Loss Function**: Cross-entropy (binary classification: relevant or not relevant)

**Inference**: Compute user embedding at query time, then use **Approximate Nearest Neighbor (ANN)** search to find the k most similar video embeddings efficiently.

**Two modes**:
- **Content-based mode**: Video tower uses full video features (title, tags, duration, etc.)
- **CF mode**: Video tower is just an embedding lookup layer (video ID to embedding), no other video features

**Pros**: Uses rich features, handles new users via their features
**Cons**: Slower serving (must compute user embedding at query time), more expensive to train

### Matrix Factorization vs. Two-Tower: Head-to-Head

| Aspect | Matrix Factorization | Two-Tower NN |
|--------|:---:|:---:|
| Training cost | Better (fewer params) | Worse (more params) |
| Inference speed | Better (static embeddings) | Worse (compute at query time) |
| Cold-start handling | Worse (cannot handle new users) | Better (uses user features) |
| Recommendation quality | Worse (no features) | Better (rich features) |

---

## Evaluation Metrics

### Offline Metrics

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **Precision@k** | Proportion of relevant videos in top-k recommendations | Standard relevance check |
| **mAP (mean Average Precision)** | Ranking quality of recommendations | When relevance is binary (relevant/not) |
| **Diversity** | How different recommended videos are from each other | Prevent "echo chamber" recommendations |

**Diversity detail**: Calculate average pairwise similarity (cosine similarity) between all recommended videos. Low similarity = high diversity. But diversity alone is not enough -- diverse but irrelevant recommendations are useless. Always pair with relevance metrics.

### Online Metrics

| Metric | Formula/Description | Caveat |
|--------|-------------------|--------|
| **CTR (Click-Through Rate)** | clicked / recommended | Cannot detect clickbait |
| **Completed videos** | Count of fully watched recommended videos | May bias toward short content |
| **Total watch time** | Sum of time spent on recommended videos | Core engagement signal |
| **Explicit feedback** | Count of likes/dislikes on recommended videos | Most accurate but sparse |

---

## Serving Architecture

### The Multi-Stage Pipeline

**Simple explanation**: You cannot carefully analyze 10 billion videos for every user request in 200ms. Instead, use a funnel:

```
10 billion videos
        |
   [Candidate Generation]  -- fast & rough, narrows to ~thousands
        |
   ~1,000-10,000 candidates
        |
   [Scoring / Ranking]     -- slow & precise, scores each candidate
        |
   ~100-500 scored videos
        |
   [Re-Ranking]            -- business rules, diversity, freshness
        |
   ~20-50 final recommendations displayed to user
```

### Stage 1: Candidate Generation

**Goal**: Narrow billions to thousands. Prioritize speed over accuracy (false positives are OK here).

**Model choice**: Two-tower neural network in CF mode (fast, handles new users)

**Workflow**:
1. Compute user embedding from user tower
2. Use ANN (Approximate Nearest Neighbor) service to find top-k most similar video embeddings
3. Return ranked candidates

**Multiple candidate generators**: In practice, use several generators in parallel:
- CF-based generator (similar users liked these)
- Trending/popular videos generator
- Location-based generator
- New videos generator (for exploration/freshness)

This diversity of sources ensures recommendations cover different reasons a user might enjoy a video.

### Stage 2: Scoring (Ranking)

**Goal**: Precisely score each candidate video using a heavy model with rich features.

**Model**: Content-based ranking model that takes user features + video features as input and predicts a relevance score for each candidate.

Since we only have thousands of candidates (not billions), we can afford to use a compute-intensive model with many features.

### Stage 3: Re-Ranking

**Goal**: Apply business rules and post-processing:
- Remove inappropriate/flagged content
- Enforce diversity (do not show 10 cooking videos in a row)
- Boost fresh/trending content
- Apply regional/legal restrictions
- Balance exploration vs. exploitation

### Cold-Start Problem Solutions

| Scenario | Solution |
|----------|----------|
| New user, no history | Use demographic features (age, country, language) in two-tower model |
| New video, no interactions | Use content features (title, tags, duration) in content-based filtering |
| Completely cold (new user + new platform) | Show popular/trending videos as fallback |

---

## Interview Cheat Sheet

### The 5-Minute Framework

1. **Clarify requirements** (1 min): Business objective, type of recommendation, scale, latency
2. **Frame the ML problem** (2 min): ML objective, input/output, ML category (hybrid filtering)
3. **Data and features** (3 min): Video features, user features (demographics + context + history)
4. **Model architecture** (5 min): Matrix factorization vs. two-tower, loss functions, training
5. **Evaluation** (3 min): Offline (precision@k, mAP, diversity) + Online (CTR, watch time, feedback)
6. **Serving pipeline** (5 min): Candidate generation -> scoring -> re-ranking
7. **Deep dives** (remaining time): Cold start, ANN search, scalability, A/B testing

### Common Follow-Up Questions

**Q: Why not just use one model?**
A: Scoring 10 billion videos per request is computationally infeasible in 200ms. The multi-stage pipeline is a speed/accuracy trade-off.

**Q: How do you handle the cold-start problem?**
A: Two-tower models use user features (not just interaction history) so new users can get recommendations from demographics. New videos use content features. Fallback to popular/trending videos.

**Q: Why WALS over SGD for matrix factorization?**
A: WALS converges faster and is parallelizable. It alternates between fixing user embeddings and video embeddings, solving each as a least squares problem.

**Q: How does ANN search work?**
A: Libraries like FAISS, ScaNN, or Annoy build index structures (e.g., IVF, HNSW) that allow finding approximate nearest neighbors in sub-linear time instead of scanning all embeddings.

**Q: How do you handle data imbalance in training?**
A: Negative sampling (randomly sample negatives), class weighting, or oversampling positives. The ratio of negatives to positives matters -- typically 3:1 to 10:1 works well.

**Q: Why combine explicit and implicit feedback?**
A: Explicit feedback (likes) is accurate but sparse. Implicit feedback (watch time, clicks) is abundant but noisy. Combining them gives both coverage and quality.

---

## References

1. YouTube recommendation system paper
2. Google's "Deep Neural Networks for YouTube Recommendations" (2016)
3. CBOW (Continuous Bag of Words) -- Mikolov et al.
4. BERT -- Devlin et al.
5. Weighted matrix factorization for implicit feedback
6. SGD optimization
7. WALS (Weighted Alternating Least Squares)

---

## Notebooks in This Module

| Notebook | Topic |
|----------|-------|
| `01_recommendation_system_design.ipynb` | Full system design walkthrough |
| `02_candidate_generation.ipynb` | Candidate generation: CF, two-tower, ANN |
| `03_ranking_models.ipynb` | Deep ranking models and multi-task learning |
| `04_interview_walkthrough.ipynb` | Complete mock interview simulation |
