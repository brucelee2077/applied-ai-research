# ML System Design Cross-Cutting Reference Materials

## Overview

These three documents are foundational reference materials that ALL ML system design candidates are expected to master. They establish:
1. A common interview delivery framework
2. Core concepts for evaluating ML systems (generalization and evaluation)
3. General knowledge about different problem types and their evaluation strategies

---

## FILE 1: SYSTEM DESIGN DELIVERY FRAMEWORK
*The interview structure that guides all ML system design interviews*

### Purpose
Provides a structured timeline and approach for candidates to follow during a 45-minute ML system design interview. This is the meta-framework that ensures candidates cover all critical areas without getting stuck in rabbit holes.

### Overall Structure (45 minutes)

| Section | Time | Purpose |
|---------|------|---------|
| Problem Framing | 5-7 min | Understand the problem, establish business objective, set ML objective |
| High-level Design | 2-3 min | Block diagram showing how components fit together |
| Data and Features | 10 min | Discuss training data, features, encodings, representations |
| Modeling | 10 min | Baseline models, model selection, model architecture details |
| Inference and Evaluation | 7 min | Evaluation metrics (offline/online), inference optimizations |
| Deep Dives | Remaining | Edge cases, scaling, monitoring, or interviewer-driven topics |

---

## SECTION 1: PROBLEM FRAMING (5-7 minutes)

### Three Core Tasks
1. **Clarify the Problem** — Ask targeted questions to understand scope and constraints
2. **Establish Business Objective** — Define what "success" means for the organization
3. **Decide on ML Objective** — Translate business goal into concrete ML task and metrics

### Key Concepts

#### Clarify the Problem
- Ask about users, pain points, current solutions
- Understand scale: DAU, RPS, real-time vs batch requirements
- Identify constraints: latency, privacy, cost
- CRITICAL: Do not jump straight in; probe to uncover what makes the problem interesting

#### Establish Business Objective
- Business objective ≠ naive ML objective
  - Example: Content moderation → business goal is "reduce harmful exposure to users", not "high accuracy"
  - Example: Posts with high views are infinitely more important than low-view posts
- Be specific: "increase CTR on recommendations" beats "improve user experience"
- Be directional but not obsessed with precision ("increase CTR by 10%" is overspecifying if current CTR unknown)
- In real teams, engineers spend years optimizing narrow business objectives; understand this deeply

#### Decide on ML Objective
- Identify ML task type: classification, regression, ranking, clustering, etc.
- Define what success looks like in ML terms
- Identify key metrics for evaluation
- Example: E-commerce → ranking model predicting P(purchase | user history), optimize for precision@k or NDCG
- Don't obsess over loss function hyperparameters or precision levels; these are too vague

### Green Flags (What Interviewers Want to See)
✓ Detailed questions that uncover the problem's core challenge
✓ Clear business objective that guides optimization
✓ Concrete ML objective with clarity for downstream work

### Red Flags (What Sinks Candidates)
✗ Jumping to naive ML objective without understanding business
✗ Questions that miss what makes the problem interesting
✗ Vague ML objective that doesn't guide the rest of the design

---

## SECTION 2: HIGH-LEVEL DESIGN (2-3 minutes)

### Purpose
Communicate how all system pieces fit together. Usually a simple block diagram (inputs → components → outputs).

### Key Principles
- Don't get hung up on perfecting the diagram; it's a communication tool, not a deliverable
- Include full lifecycle: data inputs through actions taken
- Avoid SWE-level details (database choices, API design) unless it's an ML infra interview
- Watch for interesting nuances that emerge only when walking the full lifecycle

---

## SECTION 3: DATA AND FEATURES (10 minutes)

### Three Subsections

#### Training Data
- Identify data sources and whether new data collection is needed
- Think in three buckets: supervised, semi-supervised, unsupervised
  - Most candidates acknowledge supervised; great solutions leverage orders of magnitude more data in other buckets
- Consider: data quality, labeling process, bias, whether to use direct labels or proxy signals (clicks, interactions)
- Address cold-start problems and data biases
- **Pitfall**: Don't assume perfect data; real systems spend most time on data collection and preparation

#### Features
- Start with raw data fields, then think about useful transformations
- Consider temporal aspects and domain knowledge
- **CRITICAL MISTAKE**: Don't dump a long list of random features; this shows no insight
- Prioritize based on predictive power and implementation feasibility
- Distinguish online features (queried fresh) vs offline/batch features (precomputed in feature store)
- Example: user demographics, purchase history, browsing patterns, price, time of day, co-purchasing similarity, recency, frequency

#### Encodings and Representations
- How to represent categorical, numerical, text, image data
- Options: one-hot encoding, embeddings, bag-of-words, transformers, normalization, missing value handling
- Example: embeddings for products/users in shared space, one-hot for categories, normalized prices, pre-trained LMs for descriptions

### Green Flags
✓ Creative use of supervised + semi-supervised + unsupervised data
✓ Impactful features with clear hypotheses for why they're predictive
✓ Clear discussion of encoding and representation

### Red Flags
✗ Laundry list of features, many not impactful or practical
✗ Ambiguous how features are represented or used by the model

---

## SECTION 4: MODELING (10 minutes)

### Three Subsections

#### Benchmark Models
- Start simple, not with the latest fancy model
- Examples: heuristics, simple statistical models, basic ML algorithms
- Purpose: understand the problem, provide a reference point
- Example for recommendations: popularity-based or basic collaborative filtering
- Benefit: quickly moves discussion from theoretical to practical trade-offs

#### Model Selection
- Discuss appropriate model families for the problem
- Consider trade-offs: cost, complexity, latency, interpretability, predictive power
- Evaluate classical models vs deep learning
- Show breadth: draw from approaches tested in last 2-3 years, not cutting-edge papers
- Choose ONE model to elaborate on (not enough time for all)
- Interviewers want to see you understand trade-offs, not just follow trends

#### Model Architecture
- Key components, layers, parameters, activation functions
- Regularization strategy (dropout, L2, early stopping)
- Loss function
- Example: two-tower architecture with separate embeddings, fully connected layers, ReLU, sigmoid output, dropout/L2 regularization
- **Bullshit test**: Does the interviewer believe you've built this before? Be confident in defense of your choices
- It's OK to say "I'm more familiar with X, but here's what I know about Y and how I'd generalize"
- The question: "Can we take this engineer's knowledge and apply it in a new domain?"

### Green Flags
✓ Simple, fast baseline for comparison
✓ Described multiple approaches with trade-offs
✓ Sufficient detail to explain model architecture

### Red Flags
✗ Jump to complex model without considering simpler options
✗ Hand-waved architecture details
✗ No evidence of having built something similar

---

## SECTION 5: INFERENCE AND EVALUATION (7 minutes)

### Evaluation Design and Metrics
- Offline evaluation: use historical data to estimate production performance
- Online evaluation: measure actual impact on real users
- Example metrics: precision, recall, NDCG, MAP for recommendations; CTR, conversion, AOV for business
- **Critical**: Always tie ML metrics back to business objective
  - Good ML metrics + bad business outcomes = not valuable

### Inference Considerations
- Practical operational aspects: scale, latency, cost
- Optimizations: model distillation, caching, quantization, pruning
- Importance varies by problem: offline small-scale ≠ online massive-scale
- **Why this matters**: Candidates stuck in Jupyter miss this, which is a deal-breaker for applied roles

### Green Flags
✓ Clear offline and online evaluation metrics tied to business objectives
✓ Practical inference constraints considered (latency, cost, scale)
✓ Concrete optimizations proposed where relevant

### Red Flags
✗ Metrics disconnected from business objectives
✗ Ignored practical inference constraints
✗ Proposed complex optimizations without justification

---

## SECTION 6: DEEP DIVES (Remaining time)

### Common Topics (Interviewer-Driven or Candidate-Identified)

#### Handling Edge Cases
- Cold-start: new users/items without interaction history
- Data sparsity, seasonal trends, non-representative training data
- Techniques: content-based filtering, exploration strategies (epsilon-greedy), bias mitigation

#### Scaling Considerations
- How system scales with users/data volume
- Distributed training, efficient serving, caching strategies

#### Monitoring and Maintenance
- What metrics to track in production
- When/how to retrain
- Alerts and automated triggers
- Example: CTR, recommendation diversity, model drift; auto-retrain when performance drops

---

## FILE 2: ML SYSTEM DESIGN EVALUATION
*How to evaluate every type of ML system*

### Purpose
Provides a general evaluation framework applicable to ANY ML problem, then specializes for different system types: classification, recommender systems, search/IR, and generative AI.

### General Evaluation Framework (Applies to All Problems)

Five-layer stack:

1. **Business Objective** (top-level)
   - Start here; work backward to ensure metric is tied to real value
   - Example: "eliminate legal risks" or "reduce harmful exposures"

2. **Product Metrics** (user-facing)
   - What signals indicate success?
   - Example: user retention, CTR, conversion, operational costs

3. **ML Metrics** (technical proxy)
   - Technical metrics that align with product goals
   - Often measurable without new labels/feedback
   - Example: precision, recall, NDCG, accuracy

4. **Evaluation Methodology** (how to measure)
   - Offline evaluation: historical data proxy, rapid iteration
   - Online evaluation: real-world impact measurement

5. **Address Challenges**
   - Class imbalance, label cost, feedback loops, fairness
   - Mitigation strategies

---

## SECTION 1: CLASSIFICATION SYSTEMS

### Characteristics
- Input: structured data + possibly text/image
- Output: classification label
- Often replaces human judgment (so human performance is a good baseline)
- High labeling cost, so evaluate label-efficiently

### Business Objective Examples
- Content moderation: minimize harmful exposure, avoid false positives
- Fraud detection: catch fraud while avoiding user frustration
- Spam detection: block spam while maintaining user trust
- **Key**: downstream action (block/flag) and its business impact

### Product Metrics
- User retention rate
- Time to label/moderate
- User satisfaction scores
- Operational review costs
- Appeal rate for decisions
- Downstream error costs

### ML Metrics
- **Precision**: % of positive predictions that are truly positive ("When model says 'harmful', how often right?")
- **Recall**: % of actual positives model catches ("What % of all harmful content does model find?")
- **Precision-Recall Curve**: shows trade-off across thresholds
- **Threshold tuning**: adjust to balance precision/recall based on business goal
  - Example: fix precision at 95% (human-level), maximize recall at that precision
- Common metrics: Precision@threshold, Recall@threshold, ROC-AUC, F1, PR-AUC
- **CRITICAL**: For imbalanced data (99% negative, 1% positive), PR-AUC >> ROC-AUC
  - ROC-AUC can look great even for useless classifiers on imbalanced problems

### Evaluation Methodology

**Offline**:
- Balanced test set with stratified sampling (adequate representation of all categories)
- Precision-recall curves to find optimal threshold
- Ensure offline evaluation correlates with online outcomes

**Online**:
- Shadow mode: model predicts but doesn't act; reviewers validate
- A/B test: measure technical + business impact (retention, reviewer workload)
- Compare shadow predictions vs reality

### Challenges

#### Class Imbalance
- Real-world: fraud is <1%, legitimate transactions >>99%
- Impacts dataset assembly and metric choice
- Solution: PR-AUC over ROC-AUC, stratified sampling for labeling, active learning to discover minority examples

#### Label Efficiency
- High-quality labels expensive/time-consuming
- Random sampling poor for imbalanced data
- Solution: stratified sampling, active learning to prioritize informative examples

#### Estimating Prevalence
- True prevalence of positives hard to measure in production
- Random sampling unbiased but high variance
- Example: if <1% positive, random 100-sample gives ~1 positive for prevalence estimate

#### Feedback Loops
- Model predictions influence future training data, amplify biases over time
- Solution: inject randomness (explore full space), maintain "golden set" unaffected by model decisions

---

## SECTION 2: RECOMMENDER SYSTEMS

### Characteristics
- Output: ranked list of items (movies, products, posts, users to follow)
- Value in ordering and diversity, not just yes/no
- Evaluation is "right set, right order, right time" not just "right label"
- Feedback loops are self-reinforcing (popular items → more clicks → higher rank → more clicks)

### Business Objective Examples
- Streaming: maximize watch completion (drives retention)
- E-commerce: maximize GMV net of return risk
- Social network: maximize engagement while respecting inventory/policy constraints
- **Key Questions**:
  - What's the dollar value of one more relevant recommendation?
  - What's the opportunity cost of showing irrelevant/policy-violating item?

### Product Metrics
- Session watch/purchase rate per user
- Average Revenue Per User (ARPU) or Gross Merchandise Value (GMV) per user
- Retention / churn-deferral over N days
- Inventory utilization (how evenly is catalog surfaced?)
- Dwell time (creating "doom-scrolling"? harms long-term satisfaction)
- **Tradeoff alert**: Optimizing short-term can cannibalize long-term; interviewers love probing this

### ML Metrics
- **Mean Reciprocal Rank (MRR)**: sensitive to first relevant item; good for "top pick" scenarios
- **NDCG (Normalized Discounted Cumulative Gain)**: discounts by log-rank; all-rounder
- **Hit@K / Recall@K**: fraction of sessions with ≥1 relevant item in top K
- **Coverage**: proportion of catalog shown over time window (critical for cold-start sellers, long-tail content)
- **Calibration**: does score distribution match observed engagement probabilities?
- **Key insight**: Offline ranking metrics only approximate user utility; online validation critical

### Evaluation Methodology

**Offline**:
- Leave-one-interaction-out test set (user in train and test, but different timestamps)
- Replay candidate-gen + ranking pipeline
- Watch for temporal leakage (including tomorrow's interactions in today's training = spectacular but fake)

**Online**:
- Shadow-rank mode: model reorders but served order from baseline; compare CTR for position changes
- A/B test: measure short-term (CTR) and long-term (retention) metrics
  - If they diverge, you've found a trap!
- Interleaving tests: show mixed list to same user (paired test) vs A/B (unpaired)
  - Interleaving reduces variance 10-20x vs A/B for same power

### Challenges

#### Evaluation Horizon
- Easy: measure clicks immediately
- Hard: measure true goal (renewed subscription 6 months later)
- Solution: proxy metrics + periodic hold-out cohorts, counterfactual evaluation with importance weighting

#### Feedback Loops
- Model output influences future training data (echo chamber)
- Solution: periodically inject exploration traffic (epsilon-greedy, Thompson Sampling), maintain "golden" uniformly-sampled set, retrain with counterfactual logging

#### A/B Test Validity
- Can fail silently if treatment model collects data control never sees
- Solution: always plan for replayability; log all candidates and features, not just ranked list

---

## SECTION 3: SEARCH & INFORMATION RETRIEVAL SYSTEMS

### Characteristics
- Input: query (text, voice, image)
- Output: ranked list of results
- Must rank thousands of candidates, very tight latency (e.g., 100ms)
- Balance: relevance, speed, business impact

### Business Objective Examples
- Web search: satisfied users return → ad revenue
- E-commerce search: higher relevance → conversion, AOV lift
- Enterprise search: faster retrieval → lower employee time-to-answer, support costs
- **Key**: Clicks alone poor proxy for satisfaction; measure bounce rates, reformulations, etc.

### Product Metrics
- Query success rate (did session end with click/purchase?)
- Click-through rate (CTR) on first page
- Time to first meaningful interaction
- Session abandon rate
- Revenue/conversions per search
- Latency-p99
- Query reformulation rate (proxy for dissatisfaction)

### ML Metrics
- **Precision@k / Recall@k**: % of top k results relevant vs % of relevant docs retrieved
  - k fixed as product parameter (e.g., 10 above the fold)
- **Mean Reciprocal Rank (MRR)**: position of first relevant result
- **NDCG@k**: weights high-rank relevance more heavily
- **Mean Average Precision (MAP)**: averages precision across recall levels
- **Hit Rate / Success@k**: at least one relevant doc in top k
- **CRITICAL**: Click logs biased by presentation ("presentation bias")
  - Need debiasing: inverse-propensity weighting, deterministic interleaving
  - Interviewers love probing this!

### Evaluation Methodology

**Offline**:
- Held-out set of (query, doc, graded-relevance) triples
- Compute NDCG@k, MRR, latency, cost
- Maintain diversity: head/torso/long-tail queries, freshness-sensitive vs evergreen, different locales

**Online**:
- Shadow mode: rank but don't serve; check latency & safety
- A/B or interleaving: measure CTR, revenue, latency, downstream (purchases, page views)
- Verify offline gains correlate with online wins

### Challenges

#### Query Ambiguity
- Many queries have multiple intents ("jaguar" = animal, car, sports team)
- Solution: intent classification, diversification (cover multiple intents), user behavior analysis, query refinement tracking

#### Long-Tail & Sparse Judgments
- Most queries unique/rare; can't label all
- Solution: active learning, query clustering (share judgments), synthetic query generation, transfer learning, LLM zero-shot evaluation

#### Freshness & Recency
- Must balance fresh content with quality
- Solution: track temporal relevance, monitor crawl latency, classify queries for freshness requirements, time-based decay functions

#### Feedback Loops
- Popular results get more clicks → higher rank → more clicks (self-reinforcing)
- Solution: inject ranking randomness, inverse propensity scoring, interleaving tests, golden sets, diversity metrics

---

## SECTION 4: GENERATIVE AI SYSTEMS

### Characteristics
- Output: new content (text, images, code, audio)
- "Correctness" often subjective (multiple valid outputs)
- Expensive reference answers, slow human review
- Need smart sampling + proxy metrics

### Business Objective Examples
- Support chatbot: deflect tickets without frustrating customers → retention + cost
- Image tool: boost ad CTR while avoiding brand-unsafe content → revenue + brand safety
- Code generation: boost developer productivity while minimizing hallucinations → engineer satisfaction + quality

### Product Metrics
- Task success rate (e.g., ticket fully resolved)
- Average handle time (when humans intervene)
- User satisfaction / Net Promoter Score (NPS)
- Brand-safety incident rate & review cost
- Latency
- Downstream engagement (clicks, watch-time)

### ML Metrics
Use a portfolio of metrics (not just one):

**Automated Overlap Scores** (cheap but brittle):
- BLEU, ROUGE, METEOR

**Semantic Similarity**:
- BERTScore, BLEURT

**Factuality**:
- Task-specific fact checkers, hallucination rate

**Safety**:
- Toxicity models, bias scores, hate-speech detectors

**Diversity**:
- Self-BLEU, distinct-n

**Human Ratings**:
- Pairwise preference, Likert scale

**Custom Metrics**:
- % of refusals, % of hallucinations

### Evaluation Methodology

**Offline**:
- Stratified test set: intents, languages, edge-cases, policy red-lines
- Multiple reference answers or pairwise ranking
- Quality metrics, toxicity detectors, domain slicing

**Online**:
- Shadow mode: generate but hide; log quality signals
- A/B test: new model on % of traffic; monitor KPIs + safety dashboards
- Keep "golden canary" set from legacy model for drift detection

### Challenges

#### Subjective Quality
- No single "correct" answer
- Solution: multi-reference evaluation, expert review workflows, preference learning models, break into aspects (fluency, coherence, style), re-evaluate hard cases

#### Hallucination & Factual Consistency
- Confident but incorrect statements
- Solution: source attribution, automated fact checking, retrieval augmentation, calibration metrics, hallucination taxonomies
- **Key insight**: Hallucination severity explodes on rare/niche prompts (exactly ones reviewers won't see often); target them deliberately

#### Safety & Policy Compliance
- Toxic, biased, illicit outputs = legal + brand risk
- Solution: red-team testing, adversarial evaluation, focus on false-negative rates (missing bad output > false-positive)

#### Evaluation Cost
- Human review expensive + slow
- Solution: active learning (prioritize uncertain), automated filters (fast screening), efficient sampling, proxy metrics correlated with human judgment, stratified sampling for full input coverage

#### Distribution Shift
- User behavior/content patterns evolve
- Solution: drift detection systems, rolling windows in test sets (hold-out by time, not randomly), stable benchmark canaries, version control model behavior

---

## FILE 3: GENERALIZATION
*Core concept for ALL ML systems*

### Purpose
Generalization is THE core goal of ML: train a model that performs well on unseen data. Covers overfitting, underfitting, data drift, regularization, and production failure modes.

---

## SECTION 1: OVERFITTING AND UNDERFITTING

### Overfitting
- Model learns training data too well; memorizes noise instead of underlying patterns
- High variance: small changes in training data → wildly different learned patterns
- Performs great on training, terribly on test/production
- Analogy: student memorizes practice exams word-for-word but fails on rephrased questions

### Underfitting
- Model too simple to capture patterns
- High bias: strong, wrong assumptions prevent learning
- Performs poorly on both training and test data
- Analogy: lazy student skims examples, learned nothing useful

### Key Interview Insight
- Candidates say "avoid overfitting" without explaining what it means or how to detect it
- **Be specific**: talk about training vs validation loss curves, have a measurement plan

### Spotting the Difference (Diagnostic)

Plot training loss and validation loss over epochs:

| Pattern | What It Means |
|---------|---------------|
| Both high, decreasing slowly, plateauing early | Underfitting: model not learning much |
| Training low & steady, validation close to training, small gap | Good fit: model generalizing well |
| Training keeps decreasing, validation stops improving or increases | Overfitting: model memorizing training data |

### Important: Strategic Data Holdout
- Which data to hold out for validation matters!
- Random holdout ≠ always right
- **Example failure**: Stock prediction, random stock holdout
  - Model still sees market-wide trends even without Disney stock in training
  - False impression of performance; fails in real deployment
- Solution: Often need time-based split (temporal structure matters)

### Real-World Detection
- Best detector: model bombs in production or degrades quickly over time
- Interview question: "Model underperforms in production; what do you do?"

---

## SECTION 2: MODEL CAPACITY AND DATA REQUIREMENTS

### Model Capacity
- Roughly: how complex a function the model can represent
- Usually discussed as: number of trainable parameters

### Two Examples
- **Low-capacity models** (linear): hard to overfit, prone to underfitting (can't learn sine wave)
- **High-capacity models** (deep net): can learn complex functions, prone to overfitting

### Key Rule
**High-capacity models need more data to generalize well**

- Give huge model tiny dataset → overfit immediately (can memorize every example)
- This is a red flag in interviews: proposing massive models with limited data signals inexperience

### Modern Scale Context
- GPT-3: 175B parameters
- Typical image classifier: 20-50M parameters
- Each parameter: thing model can tune during training
- More flexibility → more capacity to fit complex patterns AND noise

### Interview Guidance
- Don't dismiss simple models ("logistic regression")
- Example good answer: "We don't have enough data for large model from scratch. Start with logistic regression: won't learn feature interactions, but fast, won't overfit, gives good baseline to compare against."

---

## SECTION 3: TRANSFER LEARNING AND SMALL DATA

### Problem
- Want power of large model with limited data
- Training from scratch → immediate overfitting

### Solution: Transfer Learning
- Take pre-trained model (already learned useful features on large dataset)
- Fine-tune on smaller dataset
- Already learned: edges, textures, shapes, objects (for vision); language structure (for NLP)
- Only train final layers on task-specific patterns

### Examples
- **BERT**: 110M parameters, fine-tune on few thousand examples for text classification (already knows language)
- **ResNet**: classify niche product images with hundreds per category (already knows visual features from ImageNet)

### How It Works in Practice
- Lower layers learn general features (edges, word patterns)
- Higher layers learn task-specific features (objects, sentiment)
- Freeze lower layers, train final layers only → avoid overfitting

### Common Surgery
- Extract pre-trained model + layers
- Freeze initial layers (prevent overfitting)
- Add trainable layers on top (new classification head, LoRA adapter, etc.)

### Interview Guidance
Example: "I'll take a pre-trained model and freeze initial layers. Then train with limited data to fine-tune for our task."

---

## SECTION 4: DATA AUGMENTATION, SELF/SEMI-SUPERVISED

### Data Augmentation
- Generate synthetic training examples via transformations (rotations, crops, paraphrasing)
- Exposes model to more variety
- Very common for vision; trickier for NLP/other domains
- Works best when you understand which corruptions naturally occur in your data
- **Not a silver bullet**: generating diverse, high-signal examples often harder than the original problem

### Interview Tip
- Question to ask: "Why not just use LLM to generate labels directly?"
- Shows you understand the trade-off

### Self-Supervised Learning
- Use large unlabeled data to learn representations first
- Model creates own supervision signal: masked word prediction, corrupted image reconstruction, next frame prediction
- Similar to transfer learning, but starting from scratch using unlabeled data to guide training

### Semi-Supervised Learning
- Use small labeled data + large unlabeled data simultaneously
- Techniques: pseudo-labeling, consistency regularization
- Works when labeling expensive but unlabeled data plentiful

### Interview Signal
- Show you understand small-data is common (new products, niche domains, cold-start)
- Know when to reach for transfer learning vs simpler model

---

## SECTION 5: DATA DRIFT

### What It Is
- Distribution of production data changes over time compared to training data
- Model learned patterns from history, but real world changed
- Different from overfitting: model generalizes well when deployed, degrades over time as world changes

### Example
- Recommendation system trained on summer behavior deployed; now it's winter
- Different patterns, different user behavior

---

## Types of Data Drift

### Covariate Shift
- Distribution of input features changes; relationship between features and labels stays same
- Example: recommendation system, summer vs winter browsing patterns
- If trained on winter data correctly, model would still work

### Prior Probability Shift (Label Drift)
- Distribution of target variable changes
- Example: fraud was 1%, now 3%
- Model's decision threshold might not be calibrated anymore

### Concept Drift
- **Nastiest type**: relationship between features and labels actually changes
- User preferences evolve, competitors launch, regulations change, world events happen
- Patterns model learned are just wrong now
- Example: social media platform tastes change, nobody wants what used to be popular

### Interview Prep
- Expect questions on handling this
- Need: detection + remediation

---

## Detecting Data Drift

### Weapons in Arsenal
1. **Monitor prediction distributions**: if fraud model suddenly flags 10x more transactions, something changed
2. **Monitor feature distributions**: track mean, variance, percentiles; set alerts for drift beyond thresholds
3. **Monitor performance metrics**: track accuracy/precision/recall on labeled production data; degradation signals drift
4. **Model retraining cadence**: compare performance on same hold-out test set over time; degradation on fixed test = drift signal

### Interview Signal
- Data drift detection is hard; many teams aggressively remediate instead of detect
- Discussing how you'd monitor shows production awareness

---

## Handling Data Drift

### Start Here: Retrain Regularly
- Most important: **retrain regularly**
- Less common than expected (production pipelines hard!)
- Ensures model has fresh, near-current data
- Interview tip: it's free to suggest doing what people should be doing anyway

### Advanced Approaches (Trade-offs)

#### Online Learning
- Systems that need rapid adaptation
- Some models (logistic regression) can update continuously on new data
- Hard with deep learning
- Risk: catastrophic forgetting (forget old patterns while learning new)
- Common in fraud (needs rapid adaptation)

#### Online Embedding Learning
- Freeze model weights, update embeddings/parameters continuously
- Massive engineering challenge
- Required for rapid adaptation in recommendation systems (TikTok, Instagram)

#### Ensemble Approaches
- Keep multiple models trained on different time periods
- Weight predictions based on which period best matches current data
- Hedges against drift but increases serving costs
- Use multi-armed bandit or tournament to choose model

#### Human-in-the-Loop
- Route uncertain predictions to humans for review
- Human feedback provides fresh training data reflecting current patterns
- For high-stakes applications

### Interview Guidance
- Start with basic hygiene: retrain regularly, get fresh labeled data
- If still concerns: online learning might be good fit

---

## SECTION 6: REGULARIZATION

### Purpose
- Force models to learn robust patterns instead of memorizing noise
- Constrain model during training → harder to memorize
- Trade some training performance for better generalization
- Helps both with initial overfitting AND graceful degradation during drift

### Dropout and Layer Normalization

#### Dropout
- Randomly disable neurons during training (each neuron probability p of turning off, e.g., 0.5)
- Forces network to learn redundant representations (can't rely on single neuron)
- At test time: all neurons active, scale down by (1-p) (or use inverted dropout: scale up during training by 1/(1-p))
- Extremely effective for large networks

#### Layer Normalization
- Normalize activations across features within each training example
- Stabilizes training, mild regularization effect
- **Key**: LayerNorm >> BatchNorm for transformers/modern architectures
  - BatchNorm normalizes across batch (issues with small batch sizes, awkward for sequences)
  - LayerNorm normalizes per-example (no cross-example coupling)

### When to Use
- Proposing deep models? Must discuss dropout and layer normalization
- For transformers + modern: always layer norm
- For older CNNs: often batch norm

### L2 Regularization (Ridge / Weight Decay)

- Add penalty to loss function proportional to square of weights
- Large weights penalized more than small
- Weights stay smaller, more evenly distributed
- Model can't rely too heavily on single feature → prevents noise fitting

#### When to Use
- **Almost always**: L2 is default regularization
- Cheap, effective, doesn't complicate training
- Tune regularization strength (lambda/alpha) on validation set
  - Too much → underfitting
  - Too little → overfitting

#### Interview Guidance
- L2 is first regularization technique to mention
- Simple, well-understood, works across model types

### L1 Regularization (Lasso)

- Penalty proportional to absolute value of weights
- Unlike L2, pushes weights all the way to zero
- Sparse models: many weights exactly zero
- Effect: feature selection (only important features survive)

#### When to Use
- When you have many features, suspect most aren't useful
- Gives interpretability (shows which features matter)
- Common in linear models/logistic regression
- Less common in deep learning
- Useful for performance-constrained scenarios: prune features to simpler model

---

## SECTION 7: EARLY STOPPING

### Concept
- Overfit models' validation loss eventually increases even as training loss decreases
- Exploit this: stop training when validation performance stops improving
- Monitor validation loss during training, halt if no improvement for N epochs
- Use model from best epoch

### When to Use
- **Always**: free and effective
- Not a replacement for other regularization, but good safety net

---

## SECTION 8: SUMMARY & KEY CONCEPTS

### The Generalization Failure Modes (Production Impact)
1. **Overfitting**: memorize training data
2. **Underfitting**: too simple
3. **Data Drift**: world changed after deployment

### Detection Strategy
- Compare training vs validation loss curves (spot overfitting/underfitting)
- Monitor feature distributions, prediction distributions, performance metrics over time (spot drift)

### Mitigation Strategy
1. **Model Capacity**: match model size to data availability; high-capacity needs more data
2. **Transfer Learning**: use pre-trained models when data limited
3. **Data Augmentation**: generate synthetic examples
4. **Self/Semi-Supervised**: leverage unlabeled data
5. **Regularization**: L2, dropout, layer norm, early stopping constrain learning
6. **Monitoring**: detect drift early
7. **Retraining**: regular retraining on fresh data

### Interview Signal
- Talk about generalization whenever picking architecture, estimating data needs, designing evaluation
- Be specific: concrete techniques, how you'd measure
- Connect to production reality: models must work for real users, not just notebooks
- Show production experience: you've optimized real systems, not just read papers

---

## CROSS-CUTTING PATTERNS FOR ALL ML SYSTEM DESIGNS

### 1. Business Objective Must Drive Everything
- Start with business goal, not ML goal
- Work backward: business → product metrics → ML metrics
- Example: "reduce harmful content exposure to users" ≠ "maximize accuracy"
- Tie every decision (model, data, evaluation) back to business impact

### 2. Evaluation is Multi-Layer
- Business Objective → Product Metrics → ML Metrics → Evaluation Methodology → Challenges Addressed
- Every system needs BOTH offline and online evaluation
- Offline doesn't equal online; need correlation validation
- Offline rapid iteration, online real impact

### 3. Data Matters More Than Model
- Model selection gets attention, data gets ignored
- Reality: data collection/preparation dominates real project time
- Discuss: data sources, labeling process, quality issues, biases, proxy signals
- Avoid: perfect data assumption

### 4. Simplicity Wins Until It Doesn't
- Always start with baseline (simple model, heuristic)
- Trade-offs: simple fast vs accurate complex
- Don't jump to fancy model without justifying complexity
- Classic models (logistic, collaborative filtering) often surprisingly good

### 5. Constraints Drive Design
- Scale requirements (DAU, RPS, batch vs real-time)
- Latency (100ms? 1s? 1h?)
- Cost (inference expensive?)
- Privacy (federated? on-device?)
- These radically change architecture

### 6. Inference is Real
- Model accuracy is necessary, not sufficient
- Must serve predictions: latency, throughput, cost, scale
- Optimizations: caching, quantization, distillation, pruning
- Jupyter success ≠ production success

### 7. Monitor or Die
- Model degrades in production (drift, feedback loops, data quality issues)
- Must monitor: feature distributions, prediction distributions, performance metrics
- Set up alerts and retraining triggers
- Without this, model slowly fails and nobody knows

### 8. Feedback Loops Everywhere
- Classification: model predictions influence future labels (biases amplify)
- Recommendations: popular items more clicks → higher rank → more clicks (echo chamber)
- Search: position bias (users click what's at top more, regardless of quality)
- Solution: inject randomness, maintain golden sets, retrain on debiased data

### 9. Edge Cases are Features
- Cold-start: new users/items/queries without history
- Distribution shift: query ambiguity, time sensitivity, long tail
- These aren't bugs; they're what real systems spend time on

### 10. Interview Success = Communication
- Delivery framework: follow the structure, don't get stuck
- Be specific: "precision @95%" beats "high accuracy"
- Show trade-offs: this vs that, costs/benefits
- Connect to real-world: I've built this before; I understand the operational reality
