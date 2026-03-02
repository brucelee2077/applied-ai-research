# People You May Know (PYMK) - ML System Design Interview Module

## What is "People You May Know"?

**Imagine you join a new school.** You don't know anyone yet, but the school counselor somehow figures out exactly which kids you'd want to be friends with -- maybe they were in your old soccer team, live on your street, or are friends with your best friend from summer camp. That's basically what "People You May Know" (PYMK) does on LinkedIn and Facebook!

PYMK is a feature on social networks that looks at everything about you -- your school, workplace, friends, location, and even who you've been looking at on the platform -- and suggests people you might want to connect with. Behind the scenes, it's a sophisticated ML system that processes billions of relationships to find the perfect suggestions.

---

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Clarifying Requirements](#clarifying-requirements)
3. [Framing as an ML Task](#framing-as-an-ml-task)
4. [Data Preparation](#data-preparation)
5. [Feature Engineering](#feature-engineering)
6. [Model Development (GNNs)](#model-development)
7. [Training the Model](#training-the-model)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Serving Architecture](#serving-architecture)
10. [System Design](#system-design)
11. [Advanced Topics](#advanced-topics)
12. [Interview Tips](#interview-tips)

---

## Problem Definition

**The Goal:** Design a PYMK system similar to LinkedIn's. The system takes a user as input and outputs a ranked list of potential connections.

**Why it matters:** PYMK helps users discover new connections and grow their professional/social networks. It is one of the most impactful features on platforms like LinkedIn and Facebook because it directly drives network growth, which is a core business metric.

**Scale:**
- ~1 billion total users
- ~300 million daily active users (DAU)
- Average user has ~1,000 connections
- Friendship is **symmetrical** (both sides must agree)
- Social graphs are relatively **static** (don't change drastically day to day)

---

## Clarifying Requirements

In an interview, always start by clarifying:

| Question | Answer |
|----------|--------|
| What is the motivation? | Help users discover connections and grow networks |
| What factors matter most? | Education, work experience, social context |
| Is friendship symmetrical? | Yes -- both sides must accept |
| How many total users? | ~1 billion |
| Daily active users? | ~300 million |
| Average connections per user? | ~1,000 |
| Is the social graph dynamic? | No -- relatively stable over short periods |

---

## Framing as an ML Task

### ML Objective
Maximize the number of **formed connections** between users. This directly maps to network growth.

### Input/Output
- **Input:** A user
- **Output:** A ranked list of potential connections, ordered by likelihood of connecting

### Two Approaches

#### 1. Pointwise Learning to Rank (LTR)

Think of it like this: you take two kids' profiles, feed them into a machine, and it says "70% chance they'll be friends." Simple, but it ignores the social context entirely.

- Binary classification: takes two users, predicts probability of connection
- **Drawback:** Ignores social context (the "neighborhood" around each user)
- Doesn't know that User A and User B have 10 mutual friends

#### 2. Edge Prediction (Graph-Based) -- THE RECOMMENDED APPROACH

Now imagine instead of just looking at two kids' profiles, you look at the entire friend map of the school. You can see that User A and User B both hang out with the same group of 10 kids. That's way more powerful!

- Takes the **entire social graph** as input
- Predicts the probability of an edge (connection) between two nodes (users)
- Uses Graph Neural Networks (GNNs) to process graph structure
- Captures social context: mutual connections, neighborhood structure, triadic closure

**Why edge prediction wins:** Consider two scenarios:
- **Scenario 1:** User A and User B share 4 mutual connections (C, D, E, F) -- high chance of connecting
- **Scenario 2:** User A and User B have separate friend groups with no overlap -- low chance of connecting

The graph-based approach captures this difference; pointwise LTR cannot.

---

## Data Preparation

### Raw Data Sources

#### 1. User Data
| Field | Examples |
|-------|----------|
| Demographics | Age, gender, city, country |
| Education | School, degree, major, years attended |
| Work | Company, title, industry, tenure |
| Skills | Programming languages, certifications |

**Key Challenge:** Data standardization. "Computer Science" and "CS" mean the same thing but look different.

Solutions:
- Force selection from predefined lists
- Use heuristics to group representations
- Use ML-based methods (clustering, language models) to group similar attributes

#### 2. Connection Data
Each row = a connection between two users + timestamp.

| User ID 1 | User ID 2 | Timestamp |
|-----------|-----------|-----------|
| 28 | 3 | 1658451341 |
| 7 | 39 | 1659281720 |

#### 3. Interaction Data
All user activities on the platform:

| User ID | Interaction Type | Value |
|---------|-----------------|-------|
| 11 | Connection request | user_id_8 |
| 8 | Accepted connection | user_id_11 |
| 11 | Profile view | user_id_21 |
| 4 | Search | "Scott Belsky" |
| 11 | Comment | [user_id_4, "Very interesting..."] |

---

## Feature Engineering

### User Features

Think of these as the "stats" on each person's trading card:

- **Demographics:** Age, gender, city, country
- **Connection counts:** Number of connections, followers, following, pending requests
- **Account age:** Newer accounts might be spam; older accounts are more trustworthy
- **Received reactions:** Total likes, shares, comments (shows how active/popular someone is)

### User-User Affinity Features

These measure how "close" two users are to each other -- like measuring the distance between two kids' friend circles:

#### Education & Work Affinity
- **Schools in common:** Did they attend the same school?
- **Contemporaries at school:** Did they overlap in years? (Way more predictive than just same school)
- **Same major:** Binary feature -- same field of study?
- **Companies in common:** Worked at the same places?
- **Same industry:** Both in tech? Both in finance?

#### Social Affinity
- **Profile visits:** How many times User A looked at User B's profile
- **Mutual connections:** THE most important feature! If two users have many common friends, they're very likely to connect.
- **Time-discounted mutual connections:** This is a clever one -- recent mutual connections are worth MORE than old ones.

**Why time-discounting matters:**
- **Scenario 1:** User A formed connections recently (network is growing) -- more likely to add User B
- **Scenario 2:** User A's connections are all old (stable network) -- User A probably already knows about User B and chose NOT to connect

---

## Model Development

### Why Graph Neural Networks (GNNs)?

Remember, we formulated PYMK as an edge prediction task on a social graph. GNNs are neural networks specifically designed to operate on graph data.

**How GNNs work (simple version):**
1. Each user (node) starts with their feature vector (age, gender, connections, etc.)
2. Each connection (edge) has its own feature vector (mutual connections, profile visits, etc.)
3. The GNN passes messages between connected nodes, so each node learns about its neighborhood
4. After several rounds of message passing, each node has an **embedding** that captures both its own features AND its graph context
5. To predict if two users will connect: compute the **dot product** of their embeddings

**GNN Variants:**
- **GCN** (Graph Convolutional Network) -- basic version
- **GraphSAGE** -- samples and aggregates from neighbors (scales better)
- **GAT** (Graph Attention Network) -- uses attention to weigh neighbors differently
- **GIT** -- another variant with different architecture

---

## Training the Model

### Constructing the Training Dataset

This is a 3-step process:

#### Step 1: Create Graph Snapshot at Time t
Take the entire social graph at a specific moment in time. This is the model's input.

#### Step 2: Compute Node and Edge Features
- **Node features:** Extract user features (age, gender, account age, connection count, etc.)
- **Edge features:** Extract user-user affinity features (mutual connections, profile visits, overlapping school time, etc.)

#### Step 3: Create Labels Using Time t+1
- Look at the graph at time t+1
- **Positive labels:** Pairs of users who formed NEW connections between t and t+1
- **Negative labels:** Pairs of users who did NOT connect

This temporal split ensures no data leakage -- the model learns to predict the future from the past.

---

## Evaluation Metrics

### Offline Metrics

#### For the GNN Model
- **ROC-AUC:** Since edge prediction is binary classification (connect or not), ROC-AUC measures how well the model distinguishes positive pairs from negative pairs

#### For the PYMK System
- **mAP (mean Average Precision):** Since the output is a ranked list and each recommendation has a binary outcome (connect or not), mAP captures ranking quality

### Online Metrics

#### 1. Total Connection Requests Sent (last X days)
- Measures if the model increases outreach
- **Drawback:** Users might spam requests that never get accepted

#### 2. Total Connection Requests Accepted (last X days)
- THE key metric -- measures actual network growth
- A new connection only forms when the recipient accepts
- This is the true north star metric

---

## Serving Architecture

### The Efficiency Challenge

With 1 billion users, we'd need to compare every user against all others. That's 10^18 comparisons -- completely impractical!

### Solution 1: Friends of Friends (FoF)

**The key insight:** According to Meta research, **92% of new friendships form via FoF!**

- Average user has 1,000 friends
- FoF = 1,000 x 1,000 = 1,000,000 candidates
- Search space reduced from 1 billion to 1 million (1000x reduction!)

### Solution 2: Batch vs. Online Prediction

| Aspect | Online Prediction | Batch Prediction |
|--------|-------------------|------------------|
| When | Real-time when user loads page | Pre-computed periodically |
| Latency | Can be slow | Instant (fetched from DB) |
| Waste | None (only for active users) | Some (pre-compute for all) |
| Freshness | Always current | May be slightly stale |

**Recommendation: Batch Prediction** for two reasons:
1. Computing PYMK for 300M DAU in real-time is too slow
2. Social graphs don't change quickly, so pre-computed results stay relevant

**Smart batch strategies:**
- Re-compute every 7 days for established users
- Re-compute daily for newer users (their networks grow faster)
- Pre-compute more candidates than needed, show unseen ones each time

---

## System Design

The complete PYMK system has **two pipelines**:

### Pipeline 1: PYMK Generation Pipeline (Offline/Batch)

```
User --> FoF Service --> Candidate Connections (2-hop neighbors)
                              |
                              v
                     Scoring Service (GNN Model)
                              |
                              v
                     Ranked PYMK List --> Database
```

1. **FoF Service:** For each user, finds all 2-hop neighbors (friends of friends)
2. **Scoring Service:** Uses the GNN model to score each candidate, producing a ranked list
3. **Storage:** Stores ranked PYMK lists in a database

### Pipeline 2: Prediction Pipeline (Online)

```
User Request --> PYMK Service --> Check Database
                                      |
                        +--------------+--------------+
                        |                             |
                   Pre-computed                  Not Found
                   PYMK exists                        |
                        |                             v
                        v                    One-time request to
                  Return PYMK                Generation Pipeline
```

### Optimization Talking Points

- Pre-compute PYMK **only for active users** (saves 80% of compute)
- Use a **lightweight ranker** to reduce FoF candidates before the expensive GNN scoring
- Add a **re-ranking service** for diversity in the final list

---

## Advanced Topics

### Personalized Random Walks
An alternative efficient method for recommendations. Start at a user's node and randomly walk through the graph, keeping track of which nodes you visit most often. Those frequently-visited nodes are good connection candidates.

### Graph-Level Prediction Tasks
Beyond PYMK, GNNs can do:
- **Graph-level:** Is this molecule toxic? (entire graph classification)
- **Node-level:** Is this user a spammer? (node classification)
- **Edge-level:** Will these two users connect? (our PYMK task)

### Handling Scale
- Graph partitioning for distributed processing
- Mini-batch training with neighborhood sampling (GraphSAGE approach)
- Approximate nearest neighbor search for embedding similarity

---

## Interview Tips

### Structure Your Answer

1. **Clarify requirements** (2-3 minutes)
2. **Define ML objective** (1 minute)
3. **Choose approach** (explain pointwise LTR vs edge prediction, choose edge prediction)
4. **Data preparation** (3-4 minutes)
5. **Feature engineering** (5 minutes -- this is where you shine)
6. **Model selection** (GNN, explain why)
7. **Training** (temporal split, positive/negative labels)
8. **Metrics** (offline: ROC-AUC, mAP; online: requests sent/accepted)
9. **Serving** (FoF optimization, batch prediction)
10. **System design** (two pipelines)

### Key Points to Remember

- **Mutual connections** is the single most important feature
- **92% of new friendships** come from friends-of-friends
- **Batch prediction** is preferred because social graphs evolve slowly
- **Time-discounted mutual connections** is a clever feature that impresses interviewers
- Always mention the **privacy** angle: don't reveal who viewed profiles
- The **re-ranking service** for diversity is a great optimization to mention

### Common Follow-Up Questions

1. "How would you handle cold-start users?" -- Use content-based features (school, company) instead of graph features
2. "How do you avoid creepy recommendations?" -- Don't use phone contacts without consent, don't surface "stalker" signals
3. "How would you add diversity?" -- Re-ranking layer with diversity constraints
4. "What about privacy?" -- Don't reveal profile views in recommendations, comply with data regulations
5. "How do you handle scale?" -- FoF narrowing, batch prediction, graph partitioning

---

## References

1. Clustering and language models for data standardization
2. Meta study: 92% of new friendships via FoF; mutual connections as top feature
3. GCN (Graph Convolutional Networks)
4. GraphSAGE (Sample and Aggregate)
5. GAT (Graph Attention Networks)
6. GIT
7. GNN survey/textbook
8. Personalized Random Walks for recommendations

---

## Notebooks in This Module

| Notebook | Topic |
|----------|-------|
| [01_pymk_system_design.ipynb](01_pymk_system_design.ipynb) | Full system design: problem definition, metrics, architecture, features, model |
| [02_graph_based_approaches.ipynb](02_graph_based_approaches.ipynb) | Social graph analysis, GNNs, node embeddings, community detection |
| [03_ranking_and_privacy.ipynb](03_ranking_and_privacy.ipynb) | Ranking candidates, multi-objective optimization, privacy, notifications |
| [04_interview_walkthrough.ipynb](04_interview_walkthrough.ipynb) | Complete mock interview walkthrough from start to finish |
