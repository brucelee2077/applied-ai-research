# Loss Functions

## Introduction

The loss function is the single most important modeling decision you make. It defines what "good" means to your model. Pick the wrong loss and your model will optimize for something you don't care about — and it will do so with impressive efficiency.

In interviews, candidates who can articulate *why* they chose a specific loss function — and what goes wrong with alternatives — stand out clearly. This page covers the loss functions that matter for ML system design, organized by problem type.

---

## Classification Losses

### Binary Cross-Entropy (Log Loss)

The standard for binary classification:

`L = -[y · log(p) + (1-y) · log(1-p)]`

where y is the true label (0 or 1) and p is the predicted probability.

- **When it works:** Balanced datasets, well-calibrated probability estimates needed (ads CTR, risk scoring).
- **When it fails:** Extreme class imbalance — the model learns to predict the majority class and achieves low loss.
- **Key property:** Produces calibrated probabilities (if p=0.3, the event happens ~30% of the time). This matters for ad auctions where predicted CTR directly affects pricing.

### Focal Loss

Addresses class imbalance by downweighting easy examples:

`L = -α · (1-p)^γ · log(p)`

- γ controls how much easy examples are downweighted. γ=0 is standard cross-entropy. γ=2 is typical.
- α balances the weight between positive and negative classes.
- **When to use:** Content moderation (harmful <0.1%), fraud detection, any task with severe class imbalance.
- **Why it works better than class weighting:** It doesn't just upweight rare classes — it focuses on *hard* examples specifically. An easy negative gets low weight even though it's in the majority class.

### Label Smoothing

Replace hard labels (0/1) with soft labels (ε/(K-1), 1-ε):

- **Why:** Prevents the model from becoming over-confident. A model trained with hard labels pushes logits to ±∞, which hurts calibration and generalization.
- **Typical ε:** 0.1 (so labels become 0.05 and 0.95 for binary).
- **Tradeoff:** Improves generalization but slightly degrades calibration — the model's probabilities are less sharp.

### Asymmetric Losses

When false positives and false negatives have different costs:

- Content moderation: a false negative (missing harmful content) is much worse than a false positive (incorrectly flagging benign content)
- Medical screening: missing a disease is worse than a false alarm
- Spam filtering: blocking a legitimate email is worse than letting spam through

**Implementation:** Use different loss weights for positive and negative examples, or set different decision thresholds at inference time.

---

## Ranking Losses

For systems where the order of items matters more than individual scores.

### Pointwise

Treat each item independently. Predict a relevance score using regression or classification.

- **Advantage:** Simple — just binary cross-entropy or MSE per item.
- **Disadvantage:** Ignores relative ordering. Two items with scores 0.51 and 0.49 are treated the same as 0.99 and 0.01.
- **When to use:** When you need calibrated scores (ads, where predicted CTR affects pricing).

### Pairwise (BPR, Hinge)

Given a pair (positive, negative), ensure the positive is scored higher:

`L = max(0, margin + score_neg - score_pos)`

or the Bayesian Personalized Ranking (BPR) version:

`L = -log(σ(score_pos - score_neg))`

- **Advantage:** Directly optimizes for correct ordering.
- **Disadvantage:** Number of pairs grows quadratically. Need smart pair sampling.
- **When to use:** Recommendation retrieval, when relative order matters more than absolute scores.

### Listwise (LambdaRank, ListMLE)

Optimize a ranking metric (like NDCG) directly over the full list.

- LambdaRank: Weight pairwise gradients by the change in NDCG that swapping two items would cause. Items near the top get more gradient.
- ListMLE: Treat the list as a sequence and maximize the likelihood of the correct ordering.
- **Advantage:** Directly optimizes the metric you care about.
- **Disadvantage:** More complex to implement, harder to train.
- **When to use:** Final ranking stage where NDCG/MAP is your evaluation metric.

| Approach | Optimizes | Training complexity | Best for |
|----------|-----------|---------------------|----------|
| Pointwise | Per-item accuracy | O(n) | Scoring/calibration |
| Pairwise | Relative ordering | O(n²) worst case | Retrieval, embedding learning |
| Listwise | Ranking metric (NDCG) | O(n·log(n)) | Final ranking stage |

---

## Contrastive and Metric Learning Losses

For learning embeddings where distance/similarity is meaningful.

### InfoNCE / NT-Xent

The dominant contrastive loss:

`L = -log( exp(sim(q, k+) / τ) / Σ_i exp(sim(q, k_i) / τ) )`

- sim is cosine similarity, τ is a temperature parameter.
- **Temperature matters:** Low τ (0.05-0.1) makes the distribution sharp — the model must distinguish hard negatives. High τ (0.5-1.0) makes it softer — easier to train but weaker discrimination.
- **Used by:** CLIP, SimCLR, most modern retrieval models.
- **Scales with batch size:** More negatives per positive = better gradient signal. Batch sizes of 4K-64K are common.

### Triplet Loss

`L = max(0, d(anchor, positive) - d(anchor, negative) + margin)`

- Simpler than InfoNCE but requires explicit triplet mining.
- **Hard negative mining is critical:** Random negatives are too easy — the model converges quickly without learning useful distinctions. Hard negatives (items similar to the positive but not relevant) force the model to learn fine-grained differences.
- **When to use:** When you have explicit positive/negative pairs and want simple implementation.

---

## Multi-Task Losses

When your model predicts multiple things simultaneously.

### Weighted Sum

`L_total = w₁·L₁ + w₂·L₂ + w₃·L₃`

- **Problem:** Manual weight tuning. Weights depend on the relative scale of each loss, which changes during training.
- **Quick fix:** Normalize each loss to similar scale before weighting.

### Uncertainty Weighting (Kendall et al.)

Learn weights automatically from task uncertainty:

`L_total = (1/2σ₁²)·L₁ + (1/2σ₂²)·L₂ + log(σ₁) + log(σ₂)`

- Each σ represents homoscedastic uncertainty for that task.
- Tasks with higher uncertainty (noisier labels) get lower weight.
- **Advantage:** No manual weight tuning, adapts during training.

### Gradient Conflict

When task gradients point in opposite directions, the model can't improve on both simultaneously.

- **Detection:** Compute cosine similarity between task gradients. Negative = conflict.
- **PCGrad:** Project conflicting gradients so they don't interfere. If task A's gradient conflicts with task B's, project A's gradient onto the normal plane of B's.
- **GradNorm:** Dynamically balance gradient magnitudes across tasks to ensure all tasks train at similar rates.

---

## The Loss-Metric Gap

The training loss is rarely the same as your evaluation metric. NDCG is not differentiable. Business metrics (revenue, user satisfaction) are not differentiable. You always optimize a surrogate.

**Key insight for interviews:** The gap between loss and metric is where surprises hide. A model that achieves low cross-entropy loss might have poor NDCG because it's well-calibrated but poorly ordered. Understanding this gap — and how to minimize it — is a Staff-level skill.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should know binary cross-entropy and when to use it. For a recommendation system, they should propose a reasonable loss (cross-entropy for CTR prediction, or a pairwise loss for ranking). They should recognize that class imbalance requires special treatment. Mid-level candidates differentiate by picking a loss function that aligns with the task objective, even if they don't discuss alternatives in depth.

### Senior Engineer

Senior candidates articulate the tradeoff between different loss families. They know when pointwise losses are sufficient (calibrated scoring) and when pairwise or listwise losses are needed (ranking optimization). For a retrieval system, a senior candidate would propose contrastive loss with in-batch negatives and discuss temperature tuning. They bring up multi-task losses and the challenge of balancing objectives without being asked.

### Staff Engineer

Staff candidates focus on the loss-metric gap and how to close it. They recognize that the loss function encodes implicit assumptions about what "good" means, and changing the loss can dramatically change model behavior. A Staff candidate might point out that optimizing click-through rate (cross-entropy) is straightforward, but the business actually cares about long-term user satisfaction — which requires a different formulation entirely, perhaps involving delayed rewards or multi-objective optimization with satisfaction constraints.
