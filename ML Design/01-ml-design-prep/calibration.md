# Calibration

## Introduction

Most ML interviews focus on ranking quality — can the model put the best items at the top? But there's a class of systems where ranking is not enough. When predicted probabilities are used directly — to set ad auction prices, to decide moderation thresholds, or to combine scores from multiple models — those probabilities need to be accurate as probabilities, not just as orderings.

A model is **calibrated** if when it predicts 30% probability, the event actually happens about 30% of the time. A model can rank items perfectly and still be wildly miscalibrated. And a calibrated model can produce accurate probabilities while ranking items poorly. These are different properties, and knowing when each one matters is a key signal of production experience.

---

## Calibration vs Ranking

### When Ranking Is Enough

If all you need is to sort items from best to worst, calibration doesn't matter. A model that predicts [0.9, 0.8, 0.7] and a model that predicts [0.001, 0.0005, 0.0001] produce the same ranking. For pure ranking tasks (search results, feed recommendations), ranking metrics like NDCG and AUC are what matter.

### When Calibration Is Essential

| System | Why Calibration Matters | What Goes Wrong Without It |
|--------|------------------------|---------------------------|
| Ad auctions | Predicted CTR × bid = expected value per impression. Miscalibrated CTR → wrong auction outcomes. | Over-predicted CTR: advertiser overpays, loses trust. Under-predicted CTR: platform loses revenue. |
| Content moderation | Decision threshold: flag content if P(harmful) > 0.8. Miscalibrated → threshold doesn't mean what you think. | Over-confident: too many false positives → user complaints. Under-confident: harmful content slips through. |
| Multi-model score combination | Combining P(click) × P(purchase) to estimate P(conversion). Only works if both are calibrated. | Scores on different scales produce nonsensical combined predictions. |
| Risk assessment | Credit scoring, fraud probability. Regulatory requirements for accurate probability estimates. | Systematically under- or over-estimating risk violates regulatory standards and misallocates resources. |
| Decision-making thresholds | Any system with a cutoff: "approve if P > 0.6". | The cutoff doesn't correspond to the risk tolerance you intend. |

> "In an interview, I'd frame it this way: ranking tells you which items are better. Calibration tells you how much better. If you need to make a decision based on the actual probability — not just the ordering — you need calibration."

---

## Measuring Calibration

### Reliability Diagrams (Calibration Curves)

The most intuitive calibration diagnostic. Group predictions into bins by predicted probability, then plot predicted probability vs actual frequency.

**How to read it:**
- **Perfect calibration:** Points fall on the diagonal (predicted = actual)
- **Over-confident:** Points below the diagonal (model predicts higher probability than reality)
- **Under-confident:** Points above the diagonal (model predicts lower probability than reality)
- **S-shaped curve:** Under-confident at extremes, reasonable in the middle — common for neural networks

### Expected Calibration Error (ECE)

The most common scalar calibration metric:

`ECE = Σ (|bin_accuracy - bin_confidence| × bin_size / total_samples)`

For each bin:
- `bin_confidence` = average predicted probability in that bin
- `bin_accuracy` = fraction of positive examples in that bin
- `bin_size` = number of examples in that bin

**Intuition:** ECE is the weighted average of how far each bin deviates from perfect calibration. Lower is better. ECE = 0 means perfect calibration.

**Typical values:**
- ECE < 0.02 = well-calibrated
- ECE 0.02-0.05 = acceptable for most applications
- ECE > 0.10 = poorly calibrated, needs fixing

**Limitations of ECE:**
- Sensitive to bin size and number of bins
- Can be misleading with extreme class imbalance (most examples in one bin)
- A model can have low ECE overall but be poorly calibrated for specific subgroups

### Maximum Calibration Error (MCE)

The worst-case bin deviation:

`MCE = max(|bin_accuracy - bin_confidence|)`

Useful when you need guarantees — "the model is never off by more than X." Important for safety-critical applications where the worst case matters more than the average.

### Segment-Level Calibration

A model can be well-calibrated overall but poorly calibrated for specific segments:

| Segment | Predicted Avg CTR | Actual CTR | Calibration Ratio |
|---------|------------------|------------|-------------------|
| Overall | 2.0% | 2.1% | 0.95 (good) |
| New users (<7 days) | 3.5% | 1.8% | 1.94 (over-predicted by 2x) |
| Mobile US | 2.2% | 2.0% | 1.10 (good) |
| International desktop | 1.0% | 2.5% | 0.40 (under-predicted by 2.5x) |

The overall calibration hides the segment-level problems. Always check calibration across key segments: user tenure, platform, geography, content type.

---

## Calibration Methods

### Platt Scaling

Fit a logistic regression on the model's raw output (logit) to produce calibrated probabilities.

`P_calibrated = 1 / (1 + exp(-(a · logit + b)))`

where `a` and `b` are learned on a held-out calibration set.

**Pros:** Simple, only 2 parameters, works well for binary classification, preserves ranking.
**Cons:** Assumes the calibration function is sigmoid-shaped. Can't fix complex miscalibration patterns.
**When to use:** First thing to try. Works surprisingly well for most neural networks and tree models.

### Temperature Scaling

A special case of Platt scaling with only one parameter:

`P_calibrated = softmax(logits / T)`

- T > 1: softens probabilities (reduces over-confidence)
- T < 1: sharpens probabilities (increases confidence)
- T = 1: no change (already calibrated)

**Pros:** Single parameter, very easy to implement, widely used for neural network outputs, preserves ranking.
**Cons:** Can only globally scale confidence — can't fix different miscalibration in different probability ranges.
**When to use:** The default for multi-class neural networks. Often sufficient on its own.

### Isotonic Regression

Fits a non-parametric, monotonically increasing step function from predicted scores to calibrated probabilities.

**Pros:** Flexible — can fix any calibration pattern, no assumptions about the shape.
**Cons:** Needs more calibration data (hundreds to thousands of examples per bin). Can overfit with small calibration sets.
**When to use:** When Platt scaling is insufficient (complex miscalibration patterns), and you have enough calibration data.

### Histogram Binning

Divide predicted probabilities into bins. For each bin, replace all predicted probabilities with the bin's actual positive rate.

**Pros:** Simplest possible approach, highly interpretable.
**Cons:** Coarse — resolution depends on number of bins. Doesn't preserve fine-grained ranking within bins.
**When to use:** When interpretability is paramount, or as a diagnostic tool.

### Method Comparison

| Method | Parameters | Preserves Ranking? | Data Needed | Handles Complex Patterns? | Best For |
|--------|-----------|-------------------|-------------|---------------------------|----------|
| Platt scaling | 2 (a, b) | Yes | Small (100+) | No (sigmoid only) | Binary classification, first attempt |
| Temperature scaling | 1 (T) | Yes | Small (100+) | No (global scaling only) | Multi-class neural networks |
| Isotonic regression | Non-parametric | Yes (monotonic) | Medium (1000+) | Yes | Complex miscalibration patterns |
| Histogram binning | # bins | Within bins only | Large (5000+) | Yes | Interpretability, diagnostics |

**The practical recommendation:** Start with temperature scaling. If that's insufficient, try Platt scaling. If the calibration curve is non-sigmoid, use isotonic regression.

---

## Calibration in Practice

### The Calibration Set

Calibration must be done on data separate from the training data. Using training data for calibration is circular — the model has already memorized these examples.

**Options:**
- **Held-out calibration set:** Split your data into train/validation/calibration. Use calibration set only for fitting the calibration function.
- **Cross-calibration:** Use k-fold — train on k-1 folds, calibrate on the remaining fold, rotate. More data-efficient but more complex.

### Calibration Drift

Calibration is not permanent. As the data distribution changes over time, the calibration function becomes stale.

**Causes of calibration drift:**
- User behavior changes (seasonal patterns, trending events)
- New user segments join the platform
- Upstream feature changes alter the model's score distribution
- The world changes (economic shifts, cultural events)

**Monitoring:**
- Track calibration ratio (predicted / actual) daily, weekly
- Alert when calibration ratio deviates from 1.0 by more than 10-15%
- Monitor segment-level calibration drift separately — overall calibration can mask segment-level problems

**Recalibration cadence:**
- Ads systems: recalibrate daily or weekly (fast-moving distributions)
- Content moderation: recalibrate weekly or monthly
- Slow-moving domains (credit scoring): monthly or quarterly

### Calibration After Model Updates

When you retrain the model, the calibration function likely needs to be retrained too. The new model's score distribution may differ from the old model's.

**Pitfall:** Deploying a new model with the old calibration function. The scores are on a different scale, so the calibration function maps them incorrectly.

**Best practice:** Always retrain calibration as part of the model deployment pipeline. Automate it.

### Multi-Model Calibration

When combining predictions from multiple models, each model must be independently calibrated.

**Example:** In ads, you might combine:
- P(click) from a CTR model
- P(purchase | click) from a conversion model
- P(click) × P(purchase | click) = P(conversion)

If P(click) is over-calibrated by 2x and P(purchase|click) is under-calibrated by 0.5x, the errors might cancel on average — but they won't cancel for specific segments. Each model needs its own calibration.

---

## When Calibration Goes Wrong

### Over-Confident Models

The model predicts extreme probabilities (0.95, 0.02) when the true probabilities are moderate (0.7, 0.15).

**Causes:**
- Overfitting to training data
- No regularization or label smoothing
- Deep neural networks tend to be over-confident (especially with cross-entropy loss)
- Training set is too clean / too small

**Impact:** Decision thresholds become too aggressive. A "flag if P > 0.8" threshold catches far more than intended because the model assigns 0.9+ to borderline cases.

**Fix:** Temperature scaling (T > 1) softens probabilities. Label smoothing during training prevents extreme confidence.

### Under-Confident Models

The model predicts cautious probabilities (0.5, 0.45, 0.55) when the true probabilities are more extreme (0.1, 0.9, 0.95).

**Causes:**
- Excessive regularization (L2, dropout)
- Label smoothing with too-high epsilon
- Ensemble models (averaging predictions compresses the range)
- Training on very noisy labels

**Impact:** The model can't distinguish between confident and uncertain predictions. All predictions cluster around 0.5.

**Fix:** Temperature scaling (T < 1) sharpens probabilities. Reduce regularization. If using an ensemble, calibrate the ensemble output.

### The Ads Revenue Problem

In ad auction systems, calibration errors directly affect revenue:

- **Over-predicted CTR:** The ad wins the auction and pays too much. Short-term: platform earns more. Long-term: advertiser sees poor ROI, reduces spend, platform loses the advertiser.
- **Under-predicted CTR:** The ad loses auctions it should win. The platform shows worse ads and earns less. The advertiser misses impressions they would have paid for.

A 10% calibration error in a system serving billions of impressions per day translates to millions of dollars in misallocated revenue — daily. This is why ads teams invest heavily in calibration infrastructure and monitoring.

---

## Calibration and Negative Sampling

An important interaction that many candidates miss: negative sampling affects calibration.

If you train with a 1:10 negative:positive sampling ratio but the true ratio is 1:500, your model's predicted probabilities will be systematically too high. The model was trained on a world where positives are 50x more common than reality.

**Correction approaches:**
- **Calibration offset:** After training, apply a correction factor based on the sampling ratio
- **Prior correction:** `P_corrected = P_model × (true_neg_rate / training_neg_rate) / (P_model × (true_neg_rate / training_neg_rate) + (1 - P_model))`
- **Train with true ratio:** Use focal loss or asymmetric weighting to handle the extreme imbalance, avoiding the need for downsampling
- **Post-hoc Platt scaling:** Fit calibration on a held-out set with the true class ratio

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand that calibration means predicted probabilities should match actual frequencies. They should know that ranking and calibration are different properties, and that calibration matters for systems like ads where predicted probabilities are used directly. They differentiate by mentioning at least one calibration method (Platt scaling or temperature scaling) and recognizing that extreme class imbalance can affect calibration.

### Senior Engineer

Senior candidates can explain when calibration matters and when ranking quality is sufficient. They discuss reliability diagrams, ECE, and at least two calibration methods with tradeoffs. For an ads system, a senior candidate would explain how miscalibrated CTR predictions affect auction outcomes and advertiser experience. They proactively bring up the interaction between negative sampling rate and calibration, segment-level calibration monitoring, and the need for periodic recalibration as data distributions shift.

### Staff Engineer

Staff candidates treat calibration as a system design problem, not just a post-processing step. They recognize that calibration connects to business outcomes directly — miscalibrated ads predictions misallocate revenue, miscalibrated moderation thresholds affect user safety. A Staff candidate might point out that the hardest calibration problem is segment-level miscalibration: a model that looks well-calibrated overall can systematically over-predict for new users or under-predict for certain geographies, and these segment-level errors compound in multi-model systems where scores are combined. They propose calibration monitoring as part of the production ML pipeline, with automated alerts for drift and segment-level dashboards.
