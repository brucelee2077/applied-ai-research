# People You May Know — Staff/Principal Interview Guide

## How to Use This Guide

This guide covers a complete 45-minute staff/principal ML design interview for a LinkedIn-style "People You May Know" (PYMK) connection recommendation system. Hire and Strong Hire answers are in first-person candidate voice.

---

## Section 1: Problem Statement & Clarification (5 min)

### Interviewer Prompt

*"Design a 'People You May Know' system for a professional social network like LinkedIn. The system recommends connections users might want to add. Walk me through your approach."*

### What to Clarify — 6 Dimensions

| Dimension | Question | Why It Matters |
|-----------|----------|---------------|
| **Business objective** | Maximize new connections? Engagement? Network growth? | Connections sent vs. accepted are very different metrics |
| **Scale** | How many users? How active is the graph? | 1B users × 1B users comparison is infeasible — needs candidate pruning |
| **Latency** | Real-time or acceptable batch? | Social graph is static → batch pre-compute is viable |
| **Data availability** | Connection data, profile views, search history? | Rich behavioral signals beyond just connections |
| **Interaction types** | Symmetric connections? Asymmetric follows? | LinkedIn has symmetric connections (both must accept) |
| **Constraints** | Privacy concerns? Echo chamber risk? | Privacy: PYMK can infer sensitive information (medical patients, support groups) |

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd recommend users with the most mutual friends."*

Correct intuition but this is a rule, not an ML system. No discussion of scale, ranking, or richer features.

---

#### ⚠️ Weak Hire Answer

*"I'd ask about scale and the definition of a connection. Then I'd use a binary classifier to predict if two users will connect."*

Gets scale and framing but misses: the graph structure is the fundamental feature, batch vs. online prediction, candidate generation strategy, privacy concerns.

---

#### ✅ Hire Answer (Staff)

*"This is a graph edge prediction problem at scale. Let me clarify a few things.*

*First, the business objective: are we maximizing connection requests sent or connection requests accepted? These are different. A high requests-sent rate could mean we're suggesting lots of connections — some good, some not. Accepted requests directly measure successful new connections. I'd optimize for accepted connections.*

*Second, scale: how many total users? Active users? I'll assume LinkedIn-scale: ~1 billion total users, 300 million daily active. The key challenge: for each user, we can't compare against all other users to find connection candidates. We need smart candidate generation.*

*Third, latency: does PYMK need to be real-time (computed when the user loads their profile) or can it be pre-computed? The social graph changes slowly (connections don't change dramatically day to day). Pre-computing recommendations daily is acceptable.*

*Fourth, graph structure: LinkedIn has symmetric connections (mutual consent). The graph average degree is ~1,000 connections. This means the 2-hop neighborhood (friends-of-friends) is roughly 1,000 × 1,000 = 1M candidates per user — manageable.*

*Fifth, privacy: PYMK has a known privacy issue. If two users both joined a support group that uses LinkedIn for professional networking, showing them in each other's PYMK list reveals they're both in the same group, which may be sensitive. This is a design constraint.*

*I'll proceed with: 1B users, 300M DAU, ~1,000 average connections, batch pre-computation acceptable, optimize for accepted connections, privacy-aware design required.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to start with the privacy dimension because it's the most underappreciated complexity in PYMK design.*

*The Andreessen Horowitz example: a person joins an HIV support group hosted on a professional platform. Another group member appears in their PYMK list. The recommendation system has just leaked to both users that they share a sensitive health condition.*

*This isn't hypothetical. LinkedIn faced exactly this kind of privacy criticism in 2013 when PYMK revealed HR consultants to employees being managed out. The recommendation system implicitly revealed internal HR processes.*

*The engineering implication: PYMK must have configurable privacy guards:*
1. *Group membership privacy: if both users are members of the same group and the group has 'private' setting, don't use group co-membership as a PYMK signal*
2. *Profile view privacy: if User A viewed User B's profile in private mode, don't surface this as a connection signal*
3. *Mutual connection suppression: in highly-sensitive communities (healthcare, legal, law enforcement), suppress PYMK signals from group co-membership*

*On the core ML question: PYMK is fundamentally a graph problem. Two nodes are likely to connect if their graph neighborhoods overlap. A binary classifier that treats each (user_A, user_B) pair independently misses this graph context. Graph Neural Networks, which propagate information through the graph structure, are the right architecture.*

*Scale challenge: 1B users, each with ~1M FoF candidates, is 1 quadrillion pairs to score. The candidate generation strategy (FoF pruning) must reduce this to a manageable set before the ML model runs.*"

---

## Section 2: ML Problem Framing (5 min)

### Interviewer Prompt

*"How do you frame this as an ML problem?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"Binary classification: predict if two users will connect."*

Not wrong, but treating pairs independently ignores graph context — the single most important signal in connection prediction.

---

#### ⚠️ Weak Hire Answer

*"Learning to rank: given a target user, rank all candidate users by their probability of connecting."*

Better framing, but still doesn't mention graph structure or why GNN is the right model.

---

#### ✅ Hire Answer (Staff)

*"The ML task is graph edge prediction: given a snapshot of the social graph at time t, predict which new edges (connections) will form by time t+k.*

*This is fundamentally different from a standard binary classification problem because the features are graph-structured. The most important signal — how many mutual connections two users have — is a graph-level feature, not an individual-level feature.*

*Formally:*
- *Input: social graph G = (V, E) where V = users, E = current connections, plus node features (user profiles) and edge features (connection timestamps)*
- *Output: for a target user u, a ranked list of candidate users v₁, v₂, ..., v_k predicted to form a connection with u*

*Why not treat as independent binary classification:*
- *User A's probability of connecting with User B depends on their common friends (graph structure)*
- *User A's probability of connecting with User B is influenced by whether A recently connected with mutual friends of B*
- *Graph neural networks (GNNs) learn to propagate these neighborhood signals*

*Training data construction:*
- *Snapshot of graph at time t*
- *Positive labels: user pairs who connected between t and t+1*
- *Negative labels: user pairs who were in each other's FoF neighborhood but did not connect*

*Note: this is a temporal graph prediction problem — we must be careful not to use future graph structure as features when predicting past connections.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The ML framing question has a subtle trap: the choice of what 'negative' examples to use matters enormously for the model you learn.*

*If we use random non-connections as negatives (e.g., two users who have never been in the same FoF neighborhood), we learn a model that distinguishes 'people who have reason to connect' from 'completely random pairs.' This is easy — any shared geography or industry is sufficient.*

*The harder and more useful problem: distinguish 'people who will connect' from 'people who had every opportunity to connect but didn't.' These are both in the same FoF neighborhood, have many mutual connections, and similar profiles — but one pair connects and the other doesn't.*

*This is why FoF-sampled negatives are critical: for each (user_A, user_B) positive pair, the negatives should be sampled from user_A's FoF neighborhood who share similar numbers of mutual connections with user_A but didn't connect.*

*The GNN learns the subtle signals that distinguish these cases:*
- *Are their mutual connections from the same company? → Strong signal*
- *Did user_A recently connect with users from user_B's company? → Momentum signal*
- *Does user_B match user_A's historical connection-making pattern? → Preference signal*

*These are all graph signals that a standard classifier without neighborhood context would miss.*

*One more framing consideration: the connection request flow is asymmetric even though connections are symmetric. User_A must send the request; User_B must accept. The model should predict P(A sends request to B) × P(B accepts request from A). These have different drivers: A's propensity to send requests is about A's behavior, B's propensity to accept is about their selectivity.*"

---

## Section 3: Data & Feature Engineering (8 min)

### Interviewer Prompt

*"What features would you use? Walk me through the feature engineering."*

### Model Answers by Level

#### ❌ No Hire Answer

*"I'd use mutual friend count and whether they went to the same school."*

Two features. Misses all the rich graph signals.

---

#### ⚠️ Weak Hire Answer

*"I'd use user profile features (company, school, location) and graph features like mutual connections and profile views."*

Better but no detail on how to compute these at scale, how to encode them, or the time-discounting approach.

---

#### ✅ Hire Answer (Staff)

*"Features fall into four categories: user features, user-user affinity features, graph-based features, and interaction features.*

**User features (node features):**
- `account_age`: log-transformed days
- `num_connections`: log-transformed
- `industry`: embedding lookup, dim=16
- `job_title`: BERT embedding of job title, projected to 32-dim
- `education_institution`: embedding, dim=16
- `location`: city embedding (dim=8), country embedding (dim=4)
- `profile_completeness`: 0-1 score (% of profile fields filled)

**User-user affinity features:**
- `same_company_current`: binary
- `same_company_history`: binary (ever worked at same company)
- `same_school`: binary
- `same_industry`: binary
- `mutual_connection_count`: raw count (very strong predictor)
- `profile_view_count_A→B`: how many times A viewed B's profile (last 90 days)
- `profile_view_count_B→A`: how many times B viewed A's profile

**Graph-based features (the most important):**

*1. Number of mutual connections:*
```
|N(A) ∩ N(B)| = size of intersection of A's and B's neighbor sets
```
*This is the single strongest predictor. LinkedIn has reported it as the top feature.*

*2. Time-discounted mutual connections:*
```
weighted_mutual = Σ_{c ∈ N(A)∩N(B)} exp(-λ * t_c)
where t_c = days since the most recent of (A-c connection, B-c connection)
```
*Recent mutual connections weight more — they indicate active network growth in overlapping areas.*

*3. Common communities:*
- Number of LinkedIn Groups both users belong to (excluding private groups)
- Jaccard similarity of group memberships

*4. Social distance:*
- Shortest path length between A and B in the graph (1=direct connection, 2=FoF, etc.)
- For PYMK, most candidates are distance 2 (FoF). Some are distance 3 (FoFoF).

**Interaction features:**
- `search_A_found_B`: did user A search for terms matching B's profile?
- `coworker_viewed_B`: has any of A's current colleagues viewed B's profile?
- `recruiter_match`: both users are in the same company's recruiting pipeline*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to highlight three features that are often missed but are highly predictive.*

**1. Temporal momentum:**

*When a user is actively building their network (just started a new job, just graduated), they connect rapidly with many people. Being in the same temporal wave as another user is predictive.*

*Feature: 'connection velocity' — connections made by user A in the last 30 days. Users with high recent velocity are in 'networking mode' and are more likely to both send and accept connection requests.*

*Feature: 'shared connection velocity' — count of mutual connections formed within the last 30 days. If A and B share 3 mutual connections that were all formed this month, they're probably all expanding the same professional circle simultaneously.*

**2. Reciprocal signals:**

*LinkedIn is symmetric (both must accept), but the underlying interest may be asymmetric. If user A has viewed user B's profile 5 times but B has never viewed A's profile, A is more interested in B than vice versa. The model should predict both the sending and acceptance probability separately.*

*Feature: `profile_view_asymmetry = log((A→B views + 1) / (B→A views + 1))`. Positive value → A is more interested.*

**3. Graph motifs:**

*Certain graph substructures are highly predictive:*
- *Triangle closure: A is connected to C, B is connected to C, A-B not connected yet → A and B are predicted to connect (common friend C can facilitate)*
- *Path extension: A-B-C-D chain, if A and D just connected through C, D is more likely to connect with A*

*These motifs are exactly what GNNs learn from the graph structure through message passing. But you can also engineer them as explicit features:*
- `triangle_completion_count`: how many triangles would be closed if A and B connect?
- `common_connection_degree`: average degree of mutual connections (mutual connections that are more 'connective' are stronger bridges)*"

---

## Section 4: Model Architecture Deep Dive (12 min)

### Interviewer Prompt

*"What model architecture would you use? Why a GNN?"*

### Model Answers by Level

#### ❌ No Hire Answer

*"A random forest on mutual friend count and profile similarity."*

Not wrong as a baseline, but no understanding of graph structure or why GNN is architecturally appropriate.

---

#### ⚠️ Weak Hire Answer

*"GNN — it takes the social graph as input and learns user embeddings by aggregating information from neighbors."*

Right but no detail on the message passing mechanism, training procedure, or why GNN outperforms simpler models.

---

#### ✅ Hire Answer (Staff)

*"Let me walk through the architecture progression.*

**Baseline: Logistic Regression / XGBoost on hand-crafted features**
- Features: mutual connection count, same company, same school, profile view counts
- Pros: fast, interpretable, easy to deploy
- Why it fails: misses higher-order graph signals. Two users with 5 mutual friends but all in a very tight cluster are less likely to connect than two users with 5 mutual friends spanning diverse networks. Feature engineering can't capture this topology efficiently.

**Better: Graph Neural Network (GNN)**

*GNNs process graph-structured data through a series of 'message passing' layers. Each layer: every node aggregates information from its neighbors and updates its own representation.*

*Layer l message passing:*
```
m_v^(l) = AGG({h_u^(l-1) : u ∈ N(v)})  [aggregate neighbor representations]
h_v^(l) = UPDATE(h_v^(l-1), m_v^(l))     [update node representation]
```
*where AGG is typically mean/sum/max, and UPDATE is a linear layer + activation.*

*After L layers of message passing, node v's representation h_v^(L) encodes information from L-hop neighborhoods.*

*For PYMK:*
- *Input node features: [user_demographics, job_title_embedding, industry_embedding, profile_completeness]*
- *2-3 message passing layers → captures up to 2-3 hop neighborhood structure*
- *User embedding after L layers: 64-dim*
- *Connection prediction: dot product of user_A embedding and user_B embedding → P(connect)*

*Training:*
- *Positive examples: user pairs who connected between t and t+1 (from graph snapshot)*
- *Negative examples: FoF pairs who did NOT connect*
- *Loss: binary cross-entropy*
- *Train on graph at time t, predict edges at time t+1*

*Why GNN over feature engineering: GNN learns which graph patterns predict connections, without requiring us to enumerate them manually. Triangle closing, bridge formation, community expansion — all learned implicitly.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to go deeper on three aspects: the GNN variant choice, inductive vs. transductive learning, and the practical challenge of training on a 1B-node graph.*

**GNN variant choice:**

*There are many GNN variants: GCN, GraphSAGE, GAT, Graph Transformer. For PYMK, the key question is: can we efficiently compute user embeddings for new users and new connection patterns?*

*GCN (Graph Convolutional Network):* Aggregates with fixed learned weights. Efficient but doesn't distinguish the importance of different neighbors.

*GAT (Graph Attention Network):* Attention weights for different neighbors — more important mutual connections get higher weight. Better for PYMK where 'quality' mutual connections matter more than 'quantity.' But O(degree²) attention computation can be expensive for high-degree nodes.

*GraphSAGE (Graph SAGE):* Inductive — learns an embedding function that can generalize to unseen nodes. Aggregates neighbor features, not neighbor IDs. Critical for PYMK because:
- New users join the platform daily → need embeddings for nodes not seen during training
- GraphSAGE handles this by learning from node features, not just node identity

*For PYMK, I'd use GraphSAGE with mean aggregation:*
```
h_N(v)^(k) = MEAN({h_u^(k-1) : u ∈ N(v)})
h_v^(k) = σ(W^(k) · concat(h_v^(k-1), h_N(v)^(k)))
```

**Training on 1B-node graphs:**

*A 1B-node graph cannot fit in GPU memory. Standard GNN training requires full batch gradient descent over the entire graph. This is infeasible.*

*Solution: mini-batch training with neighborhood sampling.*
1. *Sample a mini-batch of target nodes (source and destination users)*
2. *For each target node, sample a fixed-size neighborhood (e.g., 25 1-hop neighbors, 10 2-hop neighbors)*
3. *Compute embeddings for this sampled subgraph only*
4. *Compute loss on the target node pairs in the mini-batch*

*This is the GraphSAGE neighborhood sampling approach. Memory scales with the sampling budget, not the total graph size.*

**Inductive evaluation:**

*The model must generalize to new users. Evaluate on users who joined after the training cutoff — these nodes were never seen during training. A good GNN (GraphSAGE) should still produce meaningful embeddings for new users based on their features and connections to existing users.*

*Metric for new user performance: precision@10 for PYMK on users in their first 30 days (before significant graph context accumulates).*"

---

## Section 5: Evaluation (5 min)

### Interviewer Prompt

*"How do you evaluate PYMK quality?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"**Offline metrics:**

*ROC-AUC: measures how well the model ranks positive (will-connect) pairs above negative (won't-connect) pairs. Measures discrimination quality.*

*mAP (mean Average Precision): for each user, compute the average precision of the ranked connection recommendation list. Average across all users.*
```
AP@k = (1/R) Σ_{i=1}^{k} precision@i * rel_i
mAP = (1/|U|) Σ_u AP@k_u
```
*where R = total relevant items, rel_i = 1 if item i is relevant (will connect)*

*Offline evaluation protocol:*
- *Take graph at time t*
- *Features from graph at t*
- *Labels: connections formed between t and t+30 days*
- *Evaluate mAP and AUC on held-out user set*

**Online metrics:**

*Connection Requests Sent: volume metric. Necessary but not sufficient — could just mean we're showing lots of random recommendations.*

*Connection Requests Accepted: direct measure of quality. A recommendation is successful if both users want to connect. Accepted rate = accepted / sent.*
```
Accepted rate = connections accepted / connection requests sent from PYMK
```
*This is the primary online metric. A higher accepted rate means better recommendations.*

*Network growth: connected components per user, or average shortest path length (lower = denser network). Long-term metric.*

**A/B test design:**
- *Randomize at user level*
- *Run for 2 weeks (connection behavior isn't daily — many users connect in batches)*
- *Primary metric: accepted rate. Secondary: connections per session.*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to flag a measurement challenge specific to connection recommendations: the baseline acceptance rate varies enormously by user segment.*

*Power users (recruiters, salespeople) have base connection acceptance rates of 60-70% because they're highly responsive. Regular users have rates of 10-20%. If your A/B test shows improvement in overall acceptance rate, it might just be because the treatment algorithm showed more power users. Not a real quality improvement.*

*Better analysis: stratify A/B test results by user type (power user vs. casual user) and report improvement within each segment.*

*Second issue: the 'connection request sent' metric is gameable. An algorithm that sends PYMK notifications to users at 9am when they're most likely to click will show high 'requests sent' without any improvement in model quality.*

*Counter-metric: sent/impression ratio — of all PYMK recommendations shown, what fraction did the user act on? This normalizes for notification-driven behavior.*

*Long-term metric: is the new connection still active (messaging, profile views) 30 days later? A PYMK that creates meaningful connections is more valuable than one that creates connections users ignore.*"

---

## Section 6: Serving Architecture (7 min)

### Interviewer Prompt

*"How does the serving system work at 1B user scale?"*

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"The fundamental constraint: scoring 1B × 1B user pairs per recommendation request is computationally infeasible. We need two optimizations: candidate pruning and batch pre-computation.*

**Optimization 1: Friends-of-Friends (FoF) candidate generation**

*Instead of comparing user A against all 1B users, restrict to A's 2-hop neighborhood:*
- *A's direct connections: ~1,000 users*
- *Each direct connection's connections: ~1,000 users each*
- *FoF candidates: ~1,000 × 1,000 = ~1M candidates (after deduplication)*

*This 1,000,000x reduction in candidates makes the scoring problem tractable.*

*FoF generation at scale:*
1. *Precompute adjacency lists for all users (stored in graph database: Neo4j or custom)*
2. *For each user u, do a 2-hop BFS over the adjacency list → FoF candidates*
3. *Filter: remove direct connections (already connected), remove users who sent/received a request from u in the last 7 days*

**Optimization 2: Batch pre-computation**

*The social graph changes slowly. Most users' PYMK list will be the same tomorrow as today. Rather than computing PYMK in real-time when a user loads their profile, pre-compute PYMK lists for all users once per day.*

*This batch approach:*
- *Runs once daily for all 300M DAU users*
- *Stores results in a key-value store (Redis): user_id → [recommended user IDs, scores]*
- *At request time, serving layer simply reads from KV store: ~1ms latency*

**Full pipeline:**

*PYMK Generation Pipeline (daily batch):*
1. Feature computation: extract user features, compute graph-based features from graph snapshot
2. FoF Service: for each active user, compute FoF candidates (~1M per user, run in parallel across machines)
3. Pre-filter: apply hard filters (location, industry relevance) to reduce ~1M → ~10K candidates
4. Scoring: run GNN on 10K candidates per user → predicted connection probability
5. Ranking: sort by score, take top 20
6. Write to KV store: user_id → [(candidate_id, score), ...]

*Online Prediction Pipeline (real-time):*
1. User loads profile page
2. PYMK Service: receive request with user_id
3. KV lookup: fetch pre-computed list from Redis (<1ms)
4. Real-time freshness filter: remove recently-connected users
5. Return top 10-20 recommendations*"

---

#### 🌟 Strong Hire Answer (Principal)

*"I want to go deeper on the FoF candidate generation at scale, because naive implementations are much slower than necessary.*

**FoF generation complexity:**

*For each of 300M active users, we're doing a 2-hop BFS over a graph with average degree 1,000. Naive BFS over the full graph for each user is O(300M × 1,000 × 1,000) = O(300 trillion) edge traversals. Even at 1B edge traversals/second, this takes 300,000 seconds (~3.5 days) just for FoF generation. Clearly infeasible.*

*Optimizations:*

*1. Sharded graph storage: partition the graph by user ID across many machines. Each machine holds a shard of users and their adjacency lists. BFS for a given user mostly hits local shard, with cross-shard lookups for distant connections.*

*2. Precompute partial FoF: for each user, precompute and cache their 1-hop neighborhood (direct connections). The 2-hop neighborhood is then the union of 1-hop neighborhoods of all 1-hop neighbors — one join operation.*

*3. Batch FoF computation using distributed SQL/Spark:*
```sql
-- Compute FoF for all users in one MapReduce pass
SELECT c1.user_id AS source, c2.user_id AS fof, COUNT(*) AS mutual_count
FROM connections c1
JOIN connections c2 ON c1.target_id = c2.user_id
WHERE c1.user_id != c2.user_id
  AND c1.user_id NOT IN (SELECT target_id FROM connections WHERE user_id = c1.user_id)
GROUP BY c1.user_id, c2.user_id
HAVING mutual_count >= 1
```
*This 2-table self-join computes all FoF pairs with their mutual connection count in a single distributed SQL query. At 1B nodes, this requires a distributed SQL engine (Spark, Presto) with careful partitioning.*

*4. Limit to active users: only pre-compute PYMK for users active in the last 30 days. This reduces the computation from 1B to ~300M users.*

**Graph database considerations:**

*For real-time graph queries (e.g., mutual connection count between two specific users at serving time), we need a graph database with O(1) edge lookup:*
- *Neo4j: mature, ACID, Cypher query language, good for up to ~100B edges*
- *Amazon Neptune: managed graph DB, good for AWS deployments*
- *Custom adjacency list in Redis: for the simplest access patterns, sorted sets in Redis can represent adjacency lists with O(log N) lookup*

**Privacy-aware candidate generation:**

*Some connections should never appear in PYMK due to privacy settings:*
- *Users who blocked each other*
- *Users who the privacy settings engine has flagged as 'do not surface'*
- *Users who are in private communities where co-membership is sensitive*

*These exclusions are maintained in a 'PYMK suppression list' per user, checked as a filter step after candidate generation.*"

---

## Section 7: Edge Cases & Failure Modes (5 min)

### Model Answers by Level

#### ✅ Hire Answer (Staff)

**5 Failure Modes:**

**1. Echo Chamber Formation**
- *What:* PYMK recommends users with many mutual connections → users connect with people already in their network cluster → network becomes increasingly insular
- *Detection:* Track network heterophily: are connections being formed between users from different industries/locations, or only within existing clusters?
- *Mitigation:* Diversity constraint: ensure top-10 PYMK list includes users from at least 3 different industries or companies. Measure and report cross-cluster connection rate.

**2. Privacy Inference Attacks**
- *What:* If two users appear in each other's PYMK, it reveals they have shared connections or co-membership in a group. This can reveal sensitive co-memberships (medical, legal, support groups).
- *Detection:* Privacy review: analyze whether certain group co-memberships reliably generate PYMK recommendations for members.
- *Mitigation:* Group privacy flags: if a LinkedIn Group is marked 'private' by the group admin, do not use co-membership as a PYMK signal. Give users privacy settings for which signals can be used in PYMK.

**3. Celebrity/Influencer Node Problem**
- *What:* High-degree nodes (thought leaders, famous executives) appear in many users' FoF neighborhoods. PYMK recommends them to everyone, cluttering recommendations with aspirational connections that are never accepted.
- *Detection:* Track recommendation acceptance rate by recommended user's degree. Very high-degree users should have lower acceptance rate.
- *Mitigation:* Penalize high-degree candidates in ranking. A mutual connection with a 1,000-connection user provides less signal than a mutual connection with a 100-connection user (Katz centrality weighting).

**4. Graph Sparsity for New Users**
- *What:* New users have 0-5 connections → FoF neighborhood is tiny → few candidates → low-quality PYMK → new users can't find relevant connections → churn
- *Detection:* Track PYMK click rate / acceptance rate for users with < 10 connections.
- *Mitigation:* Profile-based PYMK for cold start: use company/school/location profile matching as primary signal before graph signals are available. As graph grows, gradually shift to graph-based signals.

**5. Stale Batch Recommendations**
- *What:* Pre-computed daily PYMK lists become stale. User connects with someone, making some PYMK candidates now direct connections. Or user blocks someone in PYMK.
- *Detection:* Track % of shown PYMK recommendations that are already connections or blocked (indicates staleness).
- *Mitigation:* Apply real-time freshness filter: before serving, remove direct connections and blocked users from the pre-computed list. Accept slightly reduced list size as cost.*

---

#### 🌟 Strong Hire Answer (Principal)

*[Extends above with:]*

**6. Demographic Bubble Amplification**
- *What:* In a professional network with existing demographic biases (e.g., tech industry has more male engineers), PYMK amplifies these biases by recommending more connections within existing demographic clusters
- *Detection:* Audit: track whether PYMK acceptance rates differ by recommended user demographics. If female engineers less likely to appear in or accept PYMK from mixed-gender clusters, the system is amplifying segregation.
- *Mitigation:* Algorithmic fairness audit: run PYMK through a fairness evaluation that checks demographic parity of recommendations. Consider adding diversity constraints across demographic dimensions.

**7. Network Effects and Manipulation**
- *What:* Bad actors create many accounts that mutually connect, artificially inflating mutual connection count to appear in PYMK for target users (spam, influence operations).
- *Detection:* Velocity alerts: unusual connection spikes from a cluster of accounts. Connection quality scoring: are connections between real users or between low-activity accounts?
- *Mitigation:* Weight mutual connections by account quality score. New accounts (< 30 days old) with low profile completeness have lower weight as mutual connections.

---

## Section 8: Principal-Level — Platform Thinking (3 min)

### Model Answers by Level

#### ✅ Hire Answer (Staff)

*"Build vs. buy:*
- *Graph database (adjacency lists): build custom on top of Redis or RocksDB for the access patterns we need (2-hop BFS, mutual connection count)*
- *GNN training framework: build on PyTorch Geometric (excellent OSS library for GNNs at scale)*
- *Distributed graph processing: Apache Spark GraphX or custom MapReduce for FoF generation*
- *Online KV store: Redis Cluster (managed)*

*Cross-team sharing:*
- *The user embeddings learned by the GNN are useful beyond PYMK: job recommendations (who's connected to companies similar to your network?), alumni features (who worked at the same companies as you?), news feed ranking (weight posts from close-network users higher)*
- *The graph infrastructure (adjacency lists, FoF computation) is shared with graph-based search features*"

---

#### 🌟 Strong Hire Answer (Principal)

*"The strategic platform question for PYMK is: how do graph signals fit into the broader personalization platform?*

*LinkedIn (and similar platforms) have three distinct graph types:*
1. *Connection graph: symmetric professional connections*
2. *Follower graph: asymmetric content following*
3. *Interaction graph: weighted edges from messages, profile views, content engagement*

*PYMK currently uses only the connection graph. The richer opportunity is combining all three:*
- *Someone who has viewed your profile 5 times but isn't connected → strong PYMK candidate*
- *Someone who is connected to your followers but not your connections → bridge connection opportunity*
- *Someone your connections frequently message and engage with → likely professionally relevant*

*The platform investment: a unified graph embedding service that ingests all three graph types and produces user embeddings that encode all relationship dimensions simultaneously. This is the 'LinkedIn identity' embedding — the richest representation of who each user is professionally.*

*Org design implication: the graph platform team must have access to all three graph types. Currently, the messaging graph might be owned by the messaging team and treated as sensitive. A graph platform requires organizational alignment to share these signals appropriately, with strong privacy governance.*

*Roadmap:*
1. *Deploy FoF + GNN PYMK (Q1)*
2. *Add profile view signals to GNN (Q2)*
3. *Build unified graph platform with all three graph types (Q3-Q4)*
4. *Expose user graph embeddings as internal API for job recommendations, content ranking, etc. (Year 2)*"

---

## Section 9: Appendix — Key Formulas & Reference

### Mathematical Formulations

**GNN Message Passing (GraphSAGE):**
```
h_N(v)^(k) = MEAN({h_u^(k-1) : u ∈ N(v)})
h_v^(k) = σ(W^(k) · CONCAT(h_v^(k-1), h_N(v)^(k)))
```

**Edge Prediction:**
```
P(A, B connect) = sigmoid(h_A^(L) · h_B^(L))
```

**Time-Discounted Mutual Connections:**
```
w_mutual(A, B) = Σ_{c ∈ N(A)∩N(B)} exp(-λ * t_c)
where t_c = days since most recent connection involving c among A-c and B-c
```

**FoF Scale:**
```
|FoF(u)| ≈ |N(u)| × avg_degree ≈ 1,000 × 1,000 = 1M candidates
After deduplication and filtering: ~100K-500K unique FoF
```

**Mean Average Precision:**
```
AP@k = (1/R) Σ_{i=1}^{k} precision@i × rel_i
mAP = (1/|U|) Σ_u AP@k_u
```

**Accepted Rate:**
```
Accepted Rate = # connection requests accepted / # connection requests sent from PYMK
```

**Graph Heterophily (for echo chamber detection):**
```
H = # cross-cluster edges / # total edges
(H near 0 = echo chambers, H near 1 = mixed connections)
```

### Vocabulary Cheat Sheet

| Term | Definition |
|------|-----------|
| PYMK | People You May Know: connection recommendation feature |
| FoF | Friends-of-Friends: 2-hop neighborhood in the social graph |
| GNN | Graph Neural Network: processes graph-structured data via message passing |
| GraphSAGE | Inductive GNN variant: generalizes to unseen nodes via neighborhood sampling |
| Message passing | GNN operation: nodes aggregate and update using neighbor representations |
| Inductive learning | Model can generalize to new nodes not seen during training |
| Triangle closure | Edge (A,C) likely if A-B-C path exists and A-B, B-C already present |
| FoF candidate generation | Reducing 1B users to ~1M candidates via 2-hop graph traversal |
| Batch pre-computation | Pre-computing PYMK daily for all users; serve from KV store at query time |
| Katz centrality | Graph centrality: weights paths to high-degree nodes less |
| Echo chamber | Network where users only connect with similar others |
| Transductive GNN | GNN that can only predict for nodes seen during training (vs. inductive) |

### Key Numbers

| Metric | Value |
|--------|-------|
| Total users | 1 billion |
| Daily active users | 300 million |
| Average connections | ~1,000 per user |
| FoF candidates (raw) | ~1M per user |
| FoF candidates (after filtering) | ~10K-100K |
| KV store lookup latency | <1ms |
| Batch pre-computation frequency | Daily |
| GNN message passing layers | 2-3 |
| User embedding dimension | 64 |
| PYMK list size (served) | 10-20 candidates |
| Connection acceptance rate (benchmark) | 10-20% (regular users), 60-70% (power users) |
| Network density (LinkedIn) | ~1,000 connections / 1B users = 0.0001% |

### Rapid-Fire Day-Before Review

**Q: Why is PYMK a graph problem, not a binary classification problem?**
A: The most predictive signal — mutual connection structure — is a graph-level feature. Two users with 10 mutual friends in the same company cluster are much more likely to connect than two users with 10 mutual friends across random companies. Graph structure captures this. Independent feature engineering can't.

**Q: Why batch pre-computation instead of real-time?**
A: The social graph changes slowly (connections don't change dramatically day-to-day). Pre-computing recommendations daily is sufficient and allows expensive GNN inference to happen offline. Real-time serving is just a KV store lookup: <1ms.

**Q: How do you handle the 1B × 1B scale problem?**
A: FoF candidate generation: restrict each user's candidates to their 2-hop neighborhood (~1M candidates from ~1,000 connections × 1,000 FoF). This reduces 1B × 1B = 1 quadrillion comparisons to 1B × 1M = 1 quadrillion still... but only for active users (300M × 1M = manageable with distributed compute). Further filtered to ~10K before GNN scoring.

**Q: What is the primary online metric and why?**
A: Connection requests ACCEPTED (not just sent). 'Sent' measures how many recommendations users acted on. 'Accepted' measures successful new connections — the actual business goal. A model that generates many low-quality requests users ignore has good 'sent' but poor 'accepted' rate.

**Q: What's the privacy risk in PYMK and how do you mitigate it?**
A: Co-membership in private groups or shared sensitive context (medical, legal) can be inferred from PYMK suggestions. Mitigation: private group membership flag suppresses co-membership as a PYMK signal. Users get privacy settings controlling which signals feed PYMK.
