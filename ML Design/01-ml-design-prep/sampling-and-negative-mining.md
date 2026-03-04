# Sampling and Negative Mining

## Introduction

How you construct training examples — especially negatives — has an outsized impact on model quality. In many systems, improving your negative sampling strategy gives a bigger quality boost than changing the model architecture. Yet in interviews, candidates often describe the model and loss function in detail while hand-waving through "we'd sample some negatives."

The core challenge: in most ML systems, positive examples are rare (clicks, purchases, follows) and negative examples are overwhelmingly abundant (everything the user didn't interact with). You can't train on all negatives — there are too many. So you must sample. And how you sample determines what the model learns to distinguish.

---

## The Negative Sampling Problem

### Why Negatives Matter

Consider a recommendation system with 10 million items. A typical user interacts with maybe 100 items per month. That leaves 9,999,900 items the user didn't interact with. You have two problems:

1. **Scale:** Training on all 9,999,900 negatives per user is computationally infeasible.
2. **Quality:** Most of those negatives are trivially easy to distinguish from the positives. A jazz fan not clicking on a tractor repair video tells the model nothing useful.

The negatives you choose shape what the model learns. Easy negatives teach broad category distinctions ("jazz is not tractor repair"). Hard negatives teach fine-grained distinctions ("this jazz album is better than that jazz album for this user"). You need both — but the ratio matters.

### Positive-Negative Ratios

| Domain | Typical Positive Rate | Neg:Pos Ratio in Training | Why |
|--------|----------------------|--------------------------|-----|
| Search ranking | 1-5% CTR | 5:1 to 20:1 | Enough clicked results per query |
| Recommendations | 0.1-1% engagement | 10:1 to 100:1 | Very sparse interactions |
| Ads CTR | 0.1-2% CTR | 50:1 to 200:1 | Need accurate calibration |
| Content moderation | 0.01-0.1% harmful | 100:1 to 1000:1 | Extreme class imbalance |
| Fraud detection | 0.001-0.01% fraud | 1000:1+ | Extremely rare positives |

Higher neg:pos ratios give the model more information per positive, but with diminishing returns. Beyond ~100:1, additional random negatives rarely help — you need harder negatives instead.

---

## Sampling Strategies

### Random Negative Sampling

The simplest approach: sample items uniformly at random from the catalog as negatives.

**How it works:**
- For each positive (user, item+), sample K items uniformly from the item catalog
- Each sampled item becomes a negative: (user, item-)

**When it works:**
- Initial model training (the model needs to learn basic patterns first)
- Problems with abundant positives and clear category boundaries
- As a baseline to compare against more sophisticated strategies

**When it fails:**
- The model converges quickly because random negatives are trivially distinguishable
- For retrieval models that need to make fine-grained distinctions
- When the catalog has a long tail — most random negatives are obscure items no one would consider

**The "easy negative" problem in practice:** A model trained only on random negatives learns to distinguish broad categories (action vs documentary) but can't rank within a category (which action movie is best for this user). This ceiling becomes visible when offline metrics plateau despite more training data.

### Hard Negative Mining

Hard negatives are items that the model finds difficult to distinguish from positives. They are the training examples that teach the model the most.

**Sources of hard negatives:**

| Source | How to Generate | Quality | Cost |
|--------|----------------|---------|------|
| Impressions without clicks | Log items shown to user but not interacted with | High — user saw and rejected them | Free (comes from serving logs) |
| ANN near-misses | Find items close to the positive in embedding space that aren't relevant | High — tests embedding quality | Moderate (requires ANN lookup per training example) |
| Same-category negatives | Items in the same category/genre as the positive | Medium-high | Low |
| Previous model's top-K | Items the current model ranks highly but aren't relevant | High — targets model weaknesses | Moderate (requires model inference) |

**The false negative problem:**

The hardest negatives may not actually be negatives. A user who didn't click "Inception" may have already seen it, may not have noticed it, or may have been planning to watch it later. Training the model to rank "Inception" below truly irrelevant items corrupts learning.

**Mitigation strategies:**
- **Mixing ratios:** Combine hard negatives (30%) with random negatives (70%). The random negatives prevent the model from overfitting to false negatives.
- **Confidence filtering:** Only use hard negatives where you're confident the item is truly irrelevant (e.g., the user saw the item, scrolled past it, and engaged with the next item).
- **Margin filtering:** Exclude negatives that are too close to the positive in embedding space — they're likely false negatives.

### In-Batch Negatives

A technique used heavily in contrastive learning and two-tower retrieval models.

**How it works:**
- Within a training batch of B examples, each example's positive pair is used
- For each example, the other B-1 examples' positives serve as negatives
- No explicit negative sampling needed — negatives come from the batch itself

**Example:** A batch of 4 (user, item) pairs:
```
(user_A, item_1)  ← positive pair
(user_B, item_2)  ← positive pair
(user_C, item_3)  ← positive pair
(user_D, item_4)  ← positive pair
```

For user_A: positive = item_1, negatives = {item_2, item_3, item_4}

**Advantages:**
- No separate negative sampling pipeline — negatives are "free"
- Scales with batch size — larger batches = more negatives = better gradient signal
- Efficient GPU utilization — all computation happens within the batch

**Disadvantages:**
- **Popularity bias:** Popular items appear in more batches, so they serve as negatives more often. The model learns to push popular items' scores down, even when they're relevant.
- **Batch size dependency:** Small batches = few negatives = weak learning signal. Large batches (4K-64K) are common but require significant GPU memory.
- **No control over difficulty:** In-batch negatives are sampled by the data loader, not by relevance — most will be easy.

**Log-Q correction for popularity bias:**

The probability of item_j appearing as an in-batch negative is proportional to its frequency in the training data. Popular items are overrepresented as negatives. The correction:

`corrected_score(i, j) = score(i, j) - log(frequency(j))`

This subtracts the log-frequency of each negative, counteracting the bias toward pushing popular items down.

### Negative Sampling with Frequency-Based Weighting

Instead of uniform random sampling, sample negatives proportional to a function of their frequency.

**Word2Vec's approach:** Sample negatives proportional to `frequency^0.75`. The 0.75 exponent smooths the distribution — common items are still sampled more often, but less aggressively than their true frequency. This balances learning about common items (which the model will encounter at serving time) with seeing enough rare items.

**When to use:** Embedding learning, when the item frequency distribution is highly skewed (follows a power law, as most real catalogs do).

---

## Curriculum Learning for Negatives

Start with easy negatives and gradually increase difficulty as training progresses.

### Why Curriculum Matters

Hard negatives early in training are problematic:
1. The model hasn't learned basic patterns yet, so hard negative gradients are noisy
2. Some hard negatives are actually false negatives, and the model can't distinguish these early on
3. Training is unstable — loss oscillates instead of decreasing

Starting easy and increasing difficulty lets the model build a solid foundation before tackling fine-grained distinctions.

### A Practical Schedule

| Training Phase | Negative Mix | What the Model Learns |
|---------------|-------------|----------------------|
| Phase 1 (epochs 1-5) | 100% random | Basic category-level distinctions |
| Phase 2 (epochs 5-15) | 70% random + 30% hard | Fine-grained within-category distinctions |
| Phase 3 (epochs 15+) | 30% random + 70% hard | Decision boundary refinement |

**Triggering difficulty increases:**
- **Epoch-based:** Switch at fixed epoch boundaries. Simple but arbitrary.
- **Loss-based:** Increase difficulty when training loss plateaus. Adapts to model progress.
- **Gradient-based:** Increase difficulty when gradient norm drops (model is learning little from current negatives).

### Self-Adversarial Negatives

A related technique: use the model's own predictions to generate negatives.

1. Run the current model on the training data
2. Items the model scores highly but aren't relevant become the next batch's hard negatives
3. Retrain on these self-generated hard negatives

This creates a feedback loop where the model continuously challenges itself. The risk is the same as with any hard negative strategy — some "hard negatives" are false negatives.

---

## Sampling Strategies by Problem Type

Different problems need different negative strategies because the nature of positives, negatives, and biases varies.

| Problem | Positive Signal | Negative Strategy | Key Consideration |
|---------|----------------|-------------------|-------------------|
| Search ranking | Click + dwell > 30s | Impressions without clicks + random | Position bias correction needed |
| Recommendations | Watch > 50%, explicit ratings | Random + ANN near-misses | Popularity debiasing, cold start |
| Ads CTR prediction | Click | Impressions without clicks | Calibration: negative sampling rate affects predicted probabilities |
| Content moderation | Human-labeled harmful | Random benign content | Adversarial examples, evolving threats |
| Embedding learning | Co-occurrence, semantic similarity | In-batch + hard negatives | Dimensional collapse risk with too-easy negatives |
| Fraud detection | Confirmed fraud | Random legitimate transactions | Extreme imbalance (>1000:1), temporal patterns |

### Search Ranking: Position Bias Correction

In search, "impressions without clicks" is the most natural negative source — the user saw the item and chose not to click it. But this signal is confounded by position: users click higher positions more, regardless of relevance.

**Propensity scoring corrects for this:**

`weight(example) = 1 / P(examine | position)`

Items in lower positions get higher training weights because users are less likely to examine them — so a non-click from position 10 is weaker evidence of irrelevance than a non-click from position 1.

Propensity scores can be estimated by:
- Running occasional randomized experiments (shuffle results, observe position-independent CTR)
- Using position-dependent click models (cascade model, DBN)

### Ads CTR: Calibration and Sampling Rate

In ads systems, the predicted CTR directly affects bid pricing. If you train with a 1:10 positive:negative ratio but serve at a true 1:500 ratio, your predicted CTR will be miscalibrated (too high).

**Calibration correction:** If training negative sampling rate is `q` and true negative rate is `p`, apply:

`calibrated_CTR = predicted_CTR / (predicted_CTR + (1 - predicted_CTR) × (q / p))`

Or simply: train with the true positive rate, using techniques like focal loss or asymmetric loss to handle the extreme imbalance. Some teams train with downsampled negatives for efficiency and post-hoc calibrate using Platt scaling.

### Embedding Learning: Dimensional Collapse

When training embedding models with contrastive loss, if negatives are too easy, the model can "cheat" — it collapses all embeddings into a low-dimensional subspace and still achieves low loss.

**Signs of dimensional collapse:**
- Embedding dimensions are highly correlated
- The effective rank of the embedding matrix is much lower than the dimensionality
- Retrieval quality plateaus despite decreasing loss

**Fixes:**
- Use harder negatives (in-batch + mined hard negatives)
- Add a uniformity loss that penalizes embedding concentration
- Temperature tuning in contrastive loss — lower temperature forces harder distinctions

---

## Debiasing Sampled Training Data

No matter how you sample, the training data contains biases from how it was collected. The model will learn these biases unless you correct for them.

### Types of Bias in Training Data

| Bias Type | What Happens | Source | Correction |
|-----------|-------------|--------|-----------|
| Position bias | Higher positions get more clicks | Search/recommendation UI | Propensity scoring, position-aware training |
| Popularity bias | Popular items dominate positives and negatives | Power-law item distribution | Inverse frequency weighting, temperature sampling |
| Presentation bias | Users only interact with what they see | Previous model's selections | Counterfactual learning (IPS weighting) |
| Selection bias | Training data reflects past model behavior | Closed-loop data collection | Exploration, randomized data collection |

### Inverse Propensity Scoring (IPS)

The key idea: weight each training example by the inverse probability of it being observed.

`weighted_loss = Σ (loss_i / P(observe item_i))`

If a user was unlikely to see an item (it was in position 20, or it was a niche item), and they still clicked it, that click is a strong signal — weight it higher. If a user was very likely to see an item (position 1, popular item), a click is weaker evidence — weight it lower.

**The variance problem:** IPS weights can be very large for rare observations, causing high-variance gradients. Clipping the weights (e.g., cap at 100) or using doubly-robust estimators reduces variance at the cost of some bias.

### Counterfactual Learning

The broader framework: learn what would have happened if a different model had been serving.

- **Logged data:** (query, items_shown, clicks) from the current production model
- **Goal:** Train a new model that would produce better results than the production model
- **Challenge:** We only observe feedback for items the production model chose to show

**Off-policy evaluation:** Before A/B testing, estimate the new model's performance using logged data. IPS-weighted replay evaluation upweights rare items to correct for the production model's selection bias.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand that you need to sample negatives (can't train on all non-interactions) and that random sampling is the starting point. For a recommendation system, they should propose random negative sampling with a reasonable ratio (e.g., 10:1 negative:positive) and recognize that class imbalance affects training. They differentiate by mentioning that the choice of negatives affects what the model learns — not all negatives are equally informative.

### Senior Engineer

Senior candidates demonstrate understanding of the negative quality spectrum. They can explain why hard negatives improve model quality and describe specific sources (impressions without clicks, ANN near-misses, same-category negatives). For a retrieval system, a senior candidate would propose in-batch negatives for contrastive training, discuss the popularity bias problem with log-Q correction, and bring up curriculum learning — starting easy and increasing difficulty. They proactively mention position bias correction when discussing search or recommendation negatives.

### Staff Engineer

Staff candidates recognize that negative sampling strategy is often more impactful than model architecture. They think about the sampling pipeline as a system design problem: how to efficiently mine hard negatives at scale, how to detect when the negative distribution has drifted (changing the model's learning signal), and how to avoid the false negative trap. A Staff candidate might point out that the biggest risk with hard negatives is silently training on false negatives — items the user would have liked but never saw — and propose monitoring the overlap between "hard negatives" and "items the user later engages with" as a data quality signal.
