# 10 - Personalized News Feed: Complete Interview Guide

## What Is a Personalized News Feed?

**Imagine this:** You open Instagram or TikTok after school. There are thousands of new posts from your friends, pages you follow, and groups you're in. But your phone screen can only show a few posts at a time. How does the app decide which posts to show you *first*?

That is what a **personalized news feed** does. It is a smart sorting system that looks at all the new posts available and picks the ones **you** are most likely to enjoy, putting them at the top of your timeline.

Every major social platform does this:
- **Facebook** ranks posts from friends, groups, and pages based on your past behavior
- **Twitter/X** mixes chronological tweets with "top tweets" it thinks you will like
- **LinkedIn** prioritizes professional content based on your career interests and network

Without personalization, your feed would just be a firehose of posts in reverse-chronological order. You would scroll past hundreds of boring posts to find the one that actually matters to you. Personalization solves this by acting like a really smart friend who pre-reads everything and says, "Hey, I think you will love *these*."

---

## The Big Picture: System Design Overview

### The Core Problem

> **Given a user, return a ranked list of posts (unseen posts or posts with unseen comments), ordered by how engaging each post is to that specific user. Do this in under 200 milliseconds for 2 billion daily active users.**

### Why It Matters Commercially

Users who see engaging content stay on the platform longer. Longer sessions mean more ad impressions. More ad impressions mean more revenue. Facebook, for example, generates almost all of its revenue from ads inserted between news feed posts.

---

## Step 1: Clarifying Requirements (Interview Essentials)

In a real interview, you must clarify before building. Here is the essential dialogue:

| Question | Answer |
|----------|--------|
| What is the goal? | Keep users engaged so they stay on the platform longer (which increases ad revenue) |
| What are "new activities"? | Unseen posts AND posts with unseen comments |
| What content types exist? | Text, images, videos, or any combination |
| What does "engaging" mean? | Posts that users are most likely to interact with (click, like, share, comment) |
| What engagement types exist? | Click, like, comment, share, hide, block, friend request, dwell time, skip |
| Latency requirement? | Ranked posts must be displayed within 200ms |
| Scale? | ~3 billion total users, ~2 billion DAU, each checking feeds ~2x/day |

---

## Step 2: Frame as an ML Task

### Defining the ML Objective

There are three options for what to optimize:

**Option 1: Maximize implicit reactions (clicks, dwell time)**
- Advantage: Lots of training data (everyone clicks)
- Disadvantage: A click does not mean the user actually liked the post (clickbait problem)

**Option 2: Maximize explicit reactions (likes, shares)**
- Advantage: Stronger signal of true user preference
- Disadvantage: Very few users actually like/share posts (sparse data problem)

**Option 3: Maximize a weighted combination of both (BEST CHOICE)**
- Assigns different weights to different reactions based on business value
- Captures both passive and active engagement signals

### Reaction Weights

| Reaction | Click | Like | Comment | Share | Friend Request |
|----------|-------|------|---------|-------|----------------|
| Weight   | 1     | 5    | 10      | 20    | 30             |

Negative reactions (hide, block) get negative weights.

**Why Option 3 wins:** It lets the business tune what matters. A share is worth more than a like because it brings new eyeballs to the platform. A block is a strong negative signal that should reduce a post's score.

### Input/Output Specification

- **Input:** A user (user ID)
- **Output:** A ranked list of posts sorted by engagement score (descending)

### ML Category: Pointwise Learning to Rank (LTR)

We use **multiple binary classifiers** (one per reaction type) to predict the probability that a user will perform each reaction on each post. Then we compute:

```
Engagement Score = sum(P(reaction_i) * weight_i)
```

**Example:**
| Reaction | P(reaction) | Weight | Score |
|----------|-------------|--------|-------|
| Click    | 0.23        | 1      | 0.23  |
| Like     | 0.48        | 5      | 2.40  |
| Comment  | 0.12        | 10     | 1.20  |
| Share    | 0.04        | 20     | 0.80  |
| Friend   | 0.001       | 30     | 0.03  |
| **Total Engagement Score** | | | **4.66** |

---

## Step 3: Data Preparation

### Raw Data Sources

1. **Users Table:** ID, username, age, gender, city, country
2. **Posts Table:** Author ID, textual content, hashtags, mentions, images/videos, timestamp
3. **User-Post Interactions Table:** User ID, Post ID, interaction type (like, share, comment, click, block, impression), interaction value
4. **Friendship Table:** User ID 1, User ID 2, timestamp of friendship formation, close friend/family flag

### Feature Engineering

#### A. Post Features

| Feature | What It Is | Why It Matters | How to Prepare |
|---------|-----------|----------------|----------------|
| **Textual content** | The main text body of a post | Determines what the post is about | Use a pre-trained language model like BERT to convert text into an embedding vector |
| **Images/Videos** | Media attached to a post | Can extract signals (e.g., unsafe content, topics) | Use pre-trained models like ResNet or CLIP to create embedding vectors |
| **Reactions** | Number of likes, shares, comments, etc. | Posts with many likes are more likely to be engaging | Scale numerical values to a similar range |
| **Hashtags** | Topic tags on a post | Group content by topic; help match user interests | Tokenize with Viterbi algorithm, then vectorize with TF-IDF or Word2Vec (not Transformers -- hashtags are short phrases, no context needed) |
| **Post age** | Time since the post was created | Users prefer newer content | Bucketize into categories: <1hr, 1-5hr, 5-24hr, 1-7d, 7-30d, >30d, then one-hot encode |

#### B. User Features

| Feature | What It Is | Why It Matters | How to Prepare |
|---------|-----------|----------------|----------------|
| **Demographics** | Age, gender, country | Basic user profile for personalization | Standard encoding |
| **Contextual info** | Device type, time of day | Behavior varies by context (mobile vs desktop, morning vs night) | Encode categorically |
| **Historical interactions** | All posts the user liked, shared, commented on | Past behavior predicts future behavior | Extract features from interacted posts |
| **Mentioned in post** | Whether the user is @mentioned | Users pay more attention to posts mentioning them | Binary feature (0 or 1) |

#### C. User-Author Affinity Features (MOST IMPORTANT)

According to research, **affinity between user and author is the single most predictive factor** for engagement on Facebook.

| Feature | What It Is | Why It Matters |
|---------|-----------|----------------|
| **Like/click/comment/share rate** | Rate at which user reacted to this author's previous posts | A like rate of 0.95 means the user almost always likes this author's posts |
| **Length of friendship** | Days since they became friends | Longer friendships often correlate with stronger ties |
| **Close friends/family flag** | Whether they marked each other as close friends or family | Users pay much more attention to close friends and family |

---

## Step 4: Model Development

### Why Neural Networks?

1. **Handle unstructured data** (text, images) natively
2. **Embedding layers** for categorical features
3. **Fine-tune pre-trained models** (BERT, ResNet) end-to-end -- impossible with tree-based models

### Architecture Choice

#### Option 1: N Independent DNNs (one per reaction type)
- Train a separate network for click prediction, like prediction, share prediction, etc.
- **Drawbacks:**
  - Expensive: N models to train and maintain
  - Sparse reactions (e.g., shares) have too little training data for their dedicated model

#### Option 2: Multi-Task DNN (BEST CHOICE)
- **One shared backbone** with **N task-specific heads**
- The shared layers learn common patterns across all reactions
- Each head specializes in predicting one reaction type
- **Advantages:**
  - Shared representations improve learning for sparse reactions
  - Much cheaper to train and maintain (one model instead of N)
  - Tasks can help each other (transfer learning between tasks)

### Handling Passive Users

Many users scroll but never click, like, or share. For them, the multi-task DNN predicts near-zero probabilities for all explicit reactions, making ranking meaningless.

**Solution: Add two implicit reaction tasks:**
1. **Dwell-time prediction** (regression): How long will the user look at this post?
2. **Skip prediction** (binary): Will the user spend less than 0.5 seconds on this post?

These signals capture engagement even when users never explicitly react.

### Model Training

#### Constructing the Dataset

For each **binary classification task** (like, click, share, comment, etc.):
- **Positive examples:** User performed the reaction on a post (user_features, post_features) -> label = 1
- **Negative examples:** User saw the post (impression) but did NOT perform the reaction -> label = 0
- **Balance the dataset:** Subsample negatives to match positives (since impressions without reaction far outnumber reactions)

For the **dwell-time regression task:**
- Each impression is a data point
- The label is the actual dwell time (continuous value)

#### Loss Function

The overall loss combines task-specific losses:

```
L_total = sum(alpha_i * L_i)
```

Where:
- Binary classification tasks use **binary cross-entropy loss**
- Dwell-time regression uses **MAE, MSE, or Huber loss**
- Alpha weights control relative importance of each task

---

## Step 5: Evaluation

### Offline Metrics

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **Precision / Recall** | Per-reaction prediction accuracy | Per-task evaluation |
| **ROC-AUC** | Trade-off between true positive rate and false positive rate | Summarizes classifier performance in a single number |

### Online Metrics

| Metric | Formula | What It Captures |
|--------|---------|-----------------|
| **CTR (Click-Through Rate)** | clicked posts / impressions | Basic engagement (but vulnerable to clickbait) |
| **Reaction Rates** | liked posts / impressions (similarly for share, comment, hide, block, skip) | Explicit engagement signals -- stronger than CTR |
| **Total Time Spent** | Aggregate dwell time over a period (e.g., 1 week) | Captures both passive and active user engagement |
| **User Satisfaction Survey** | Explicit user feedback | Most accurate but expensive to collect |

**Key insight for interviews:** CTR alone is insufficient because clickbait inflates CTR without increasing real engagement. Always mention reaction rates and total time spent as complementary metrics.

---

## Step 6: Serving Architecture

The production system has two main pipelines:

### Data Preparation Pipeline
- Processes raw data into features
- Stores pre-computed features in a feature store for low-latency access
- Similar to Ad Click Prediction pipeline

### Prediction Pipeline (3 stages)

#### 1. Retrieval Service
- Fetches all unseen posts and posts with unseen comments for the user
- Must be efficient given billions of posts (uses inverted indexes, social graph traversal)
- Produces a **candidate set** (thousands of posts)

#### 2. Ranking Service
- Runs the multi-task DNN on each candidate post
- Computes the weighted engagement score for each post
- Sorts posts by score
- Produces a **ranked list**

#### 3. Re-Ranking Service
- Applies additional business logic and user filters on top of ML scores
- Examples:
  - Boost posts about topics the user explicitly expressed interest in (e.g., soccer)
  - Enforce diversity (don't show 10 posts from the same author in a row)
  - Inject sponsored content at appropriate positions
  - Apply content policy filters (remove misinformation, NSFW, etc.)
  - Handle viral posts appropriately

---

## Step 7: Other Talking Points (Advanced)

These are topics to discuss if there is time remaining in the interview:

1. **Viral posts:** Posts going viral create thundering-herd problems. Need special caching and distribution strategies.

2. **Cold-start for new users:** No interaction history means no affinity features. Solutions include:
   - Content-based recommendations using demographics
   - Popularity-based ranking as a fallback
   - Exploration strategies to quickly learn preferences

3. **Positional bias:** Users are more likely to click posts at the top regardless of quality. Solutions:
   - Train with position as a feature, then set position to a default at inference
   - Use inverse propensity weighting

4. **Retraining frequency:** How often should the model be retrained?
   - Depends on how fast user preferences change
   - Common approach: daily retraining with hourly feature updates

---

## Quick Interview Cheat Sheet

```
1. CLARIFY: What platform? What reactions? Latency? Scale?
2. ML OBJECTIVE: Weighted engagement score (clicks*1 + likes*5 + comments*10 + shares*20)
3. DATA: Users, Posts, Interactions, Friendship graph
4. FEATURES: Post features + User features + User-Author affinity (most important!)
5. MODEL: Multi-task DNN with shared backbone + N heads (one per reaction)
6. PASSIVE USERS: Add dwell-time (regression) and skip (binary) tasks
7. TRAINING: Binary cross-entropy for classification, Huber for regression
8. OFFLINE METRICS: ROC-AUC per task
9. ONLINE METRICS: CTR, reaction rates, total time spent, user surveys
10. SERVING: Retrieval -> Ranking -> Re-ranking (< 200ms total)
11. ADVANCED: Viral posts, cold start, positional bias, retraining frequency
```

---

## References

1. News Feed ranking in Facebook - engineering.fb.com
2. Twitter's news feed system - blog.twitter.com
3. LinkedIn's News Feed system - engineering.linkedin.com
4. BERT paper - arxiv.org/pdf/1810.04805.pdf
5. ResNet model - arxiv.org/pdf/1512.03385.pdf
6. CLIP model - openai.com/blog/clip/
7. Viterbi algorithm - Wikipedia
8. TF-IDF - Wikipedia
9. Word2Vec - Wikipedia
10. Serving a billion personalized news feeds - YouTube
11. Mean Absolute Error - Wikipedia
12. Mean Squared Error - Wikipedia
13. Huber loss - Wikipedia

---

## Notebooks in This Module

| # | Notebook | What You Will Learn |
|---|----------|---------------------|
| 01 | [01_news_feed_system_design.ipynb](01_news_feed_system_design.ipynb) | End-to-end system design overview: clarifying requirements, ML framing, feature engineering, multi-task model architecture, serving pipeline (retrieval -> ranking -> re-ranking), and evaluation metrics. Includes running PyTorch code for the full multi-task DNN and a simulated serving pipeline. |
| 02 | [02_ranking_and_personalization.ipynb](02_ranking_and_personalization.ipynb) | Deep dive into the ranking model: user-author affinity scoring with time decay, content understanding via NLP and CV embeddings, social graph features, embedding layers in PyTorch, the full FullNewsFeedRanker model, and negative feedback handling (hide, unfollow, report, block). |
| 03 | [03_multi_task_and_engagement.ipynb](03_multi_task_and_engagement.ipynb) | Deep dive into multi-task learning: why multi-task beats N independent models (sparse data, shared representations, cost), full PyTorch implementation with 6 heads (click, like, comment, share, dwell-time, skip), weighted loss function with Huber for regression, user-author affinity features, engagement score computation, positional bias with inverse propensity weighting, and cold-start fallback for new users. |
| 04 | [04_interview_walkthrough.ipynb](04_interview_walkthrough.ipynb) | Complete 45-minute mock interview simulation: phase-by-phase timeline visualization, full interviewer/candidate dialogue, whiteboard-style architecture diagram, live engagement score demo, 6 common follow-up questions with staff-level answers, scoring rubric across junior/senior/staff levels, 30-second elevator pitch, and a printable key-phrases cheat sheet. |
