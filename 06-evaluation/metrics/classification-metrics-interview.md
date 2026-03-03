> **What this file covers**
> - 🎯 Why accuracy fails under class imbalance — the base rate trap
> - 🧮 Full formulas for precision, recall, F1, micro/macro/weighted averaging with worked examples
> - 🧮 ROC curves, AUC, PR curves — when each is appropriate
> - 🧮 Calibration, Brier score, reliability diagrams
> - ⚠️ 5 failure modes: class imbalance, threshold sensitivity, label noise, metric gaming, leakage
> - 📊 Computational complexity of each metric
> - 💡 Micro vs macro vs weighted averaging — design trade-offs
> - 🏭 Production considerations: monitoring, threshold selection, multi-class extensions
> - Staff/Principal Q&A with all four hiring levels shown

---

# Classification Metrics — Interview Deep-Dive

This file assumes you have read [classification-metrics.md](./classification-metrics.md) and have the intuition for TP/FP/FN/TN, the confusion matrix, precision, recall, and F1. Everything here is for Staff/Principal depth.

---

## 🗺️ Concept Flow

```
              Raw model output
                    │
                    ▼
           Predicted probabilities
           (e.g., 0.73 for "spam")
                    │
                    ▼
            Apply threshold (e.g., > 0.5)
                    │
                    ▼
           Binary predictions
           (e.g., "spam" or "not spam")
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Confusion     ROC/PR      Calibration
    Matrix       Curves       Check
        │           │           │
        ▼           ▼           ▼
   Precision   AUC score    Brier score
   Recall      (threshold-  (are the
   F1          independent) probabilities
   Accuracy                 trustworthy?)
```

The key insight: most classification metrics depend on a **threshold**. Changing the threshold trades precision for recall. ROC-AUC and PR-AUC evaluate the model across all thresholds at once.

---

## 🧮 The Core Formulas

All formulas build from the confusion matrix:

```
🧮 Confusion matrix entries:

    TP = true positives   (predicted positive, actually positive)
    FP = false positives  (predicted positive, actually negative)
    FN = false negatives  (predicted negative, actually positive)
    TN = true negatives   (predicted negative, actually negative)

    N = TP + FP + FN + TN  (total examples)
```

### Accuracy

The fraction of all predictions that are correct.

```
🧮 Accuracy:

    Accuracy = (TP + TN) / (TP + FP + FN + TN)
```

Worked example: TP = 80, FP = 10, FN = 20, TN = 890.
Accuracy = (80 + 890) / (80 + 10 + 20 + 890) = 970 / 1000 = **97.0%**

But notice: there are only 100 actual positives (TP + FN = 80 + 20) out of 1000 examples. A model that always says "negative" would get 900/1000 = 90% accuracy. Accuracy is misleading here.

### Precision

Of all the examples the model called positive, what fraction actually were positive?

```
🧮 Precision:

    Precision = TP / (TP + FP)

    Denominator = everything the model labeled as positive
```

Worked example: TP = 80, FP = 10.
Precision = 80 / (80 + 10) = 80 / 90 = **88.9%**

Precision is undefined when TP + FP = 0 (the model never predicts positive). By convention, set it to 0 in this case.

### Recall (Sensitivity, True Positive Rate)

Of all the examples that actually are positive, what fraction did the model find?

```
🧮 Recall:

    Recall = TP / (TP + FN)

    Denominator = everything that actually IS positive
```

Worked example: TP = 80, FN = 20.
Recall = 80 / (80 + 20) = 80 / 100 = **80.0%**

Recall is undefined when TP + FN = 0 (no actual positives exist). This is a dataset problem, not a model problem.

### F1 Score

The harmonic mean of precision and recall. It penalizes extreme imbalances between the two.

```
🧮 F1 Score:

    F1 = 2 × (Precision × Recall) / (Precision + Recall)

    Equivalently, from the confusion matrix:

    F1 = 2·TP / (2·TP + FP + FN)
```

Worked example: Precision = 88.9%, Recall = 80.0%.
F1 = 2 × (0.889 × 0.800) / (0.889 + 0.800) = 2 × 0.711 / 1.689 = 1.422 / 1.689 = **84.2%**

**Why harmonic mean, not arithmetic mean?**

The arithmetic mean of 100% and 0% is 50% — which sounds acceptable. The harmonic mean of 100% and 0% is 0% — which correctly signals a broken model. The harmonic mean punishes low values harder than the arithmetic mean. A model cannot score a high F1 by being excellent on one metric and terrible on the other.

### Specificity (True Negative Rate)

Of all the actual negatives, what fraction did the model correctly identify as negative?

```
🧮 Specificity:

    Specificity = TN / (TN + FP)
```

Specificity is the "recall for the negative class." It matters when you care about false alarm rate — for example, in medical screening where unnecessary follow-up tests are expensive.

### False Positive Rate (Fall-out)

```
🧮 FPR:

    FPR = FP / (FP + TN) = 1 - Specificity
```

The FPR is the x-axis of the ROC curve.

---

## 🧮 F-beta: Generalizing F1

Sometimes precision and recall are not equally important. The F-beta score lets you control the trade-off.

```
🧮 F-beta score:

    F_β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)

    β > 1 → weights recall more heavily
    β < 1 → weights precision more heavily
    β = 1 → standard F1 (equal weight)
```

Common choices:
- **F0.5** — precision matters twice as much. Used for content moderation, search results.
- **F1** — equal balance. General-purpose default.
- **F2** — recall matters twice as much. Used for disease screening, security threats.

---

## 🧮 Multi-Class Averaging: Micro, Macro, Weighted

When there are more than two classes, you compute per-class precision, recall, and F1, then combine them. The combining strategy matters a lot.

### Micro averaging

Pool all TP, FP, FN across all classes, then compute once.

```
🧮 Micro-averaged precision:

    Precision_micro = Σ TPᵢ / (Σ TPᵢ + Σ FPᵢ)

    Sum is over all classes i = 1, ..., C.
```

Micro averaging treats every example equally, regardless of which class it belongs to. This means large classes dominate. For balanced datasets, micro F1 equals accuracy.

### Macro averaging

Compute per-class metrics, then take the unweighted mean.

```
🧮 Macro-averaged precision:

    Precision_macro = (1/C) × Σ Precisionᵢ

    Each class gets equal vote regardless of size.
```

Macro averaging treats every class equally. A rare class with 10 examples has the same influence as a common class with 10,000 examples. Use macro when all classes matter equally, even if some are rare.

### Weighted averaging

Compute per-class metrics, then take a weighted mean using class support (number of true examples per class) as weights.

```
🧮 Weighted-averaged precision:

    Precision_weighted = Σ (nᵢ / N) × Precisionᵢ

    Where nᵢ = number of true examples in class i, N = total examples.
```

Weighted averaging is a middle ground. Larger classes contribute more, but small classes are not ignored like in micro averaging.

### Comparison

|                  | Micro           | Macro           | Weighted         |
|------------------|-----------------|-----------------|------------------|
| Each example     | Equal weight    | Unequal (big class matters more per-metric) | Proportional     |
| Each class       | Unequal (big class dominates) | Equal weight    | Proportional     |
| Best when        | You care about overall correctness | All classes equally important | You want per-class fairness proportional to size |
| Hides            | Poor performance on rare classes | Dataset size distribution | Nothing, but harder to interpret |
| For balanced data | = accuracy      | = micro         | = micro = macro  |

**🎯 Key interview insight:** if an interviewer asks "what F1 did your model get?" and the dataset is imbalanced, the FIRST follow-up should be: "micro or macro?" A good micro F1 can hide terrible performance on rare classes. This distinction separates junior from senior engineers.

---

## 🧮 ROC Curve and AUC

The ROC (Receiver Operating Characteristic) curve plots TPR (recall) against FPR at every possible threshold.

```
🧮 ROC curve construction:

    For each threshold t from 1.0 down to 0.0:
      1. Classify examples: positive if predicted probability ≥ t, else negative
      2. Compute TPR = TP / (TP + FN)
      3. Compute FPR = FP / (FP + TN)
      4. Plot the point (FPR, TPR)

    AUC-ROC = area under this curve (0.0 to 1.0)
```

**Interpretation:**
- AUC = 1.0: perfect classifier (exists a threshold that separates all positives from all negatives)
- AUC = 0.5: random classifier (the diagonal line)
- AUC < 0.5: worse than random (label your predictions backwards and you improve)

**🎯 Key property:** AUC-ROC is equivalent to the probability that a randomly chosen positive example has a higher predicted score than a randomly chosen negative example.

### When ROC-AUC Misleads

Under severe class imbalance, ROC-AUC can look excellent while the model is practically useless.

Why: the x-axis is FPR = FP / (FP + TN). When TN is huge (many negatives), even a large absolute number of false positives produces a small FPR. The ROC curve stays close to the top-left corner, AUC stays high, but precision may be terrible.

Example: 10 positives, 10,000 negatives. Model gets 8 TP, 100 FP. Recall = 0.80, FPR = 100/10000 = 0.01. ROC looks great. But Precision = 8/108 = 7.4%. Useless in practice.

---

## 🧮 PR Curve and AUC-PR

The PR (Precision-Recall) curve plots precision against recall at every threshold.

```
🧮 PR curve construction:

    For each threshold t from 1.0 down to 0.0:
      1. Classify examples: positive if predicted probability ≥ t
      2. Compute Precision = TP / (TP + FP)
      3. Compute Recall = TP / (TP + FN)
      4. Plot the point (Recall, Precision)

    AUC-PR = area under this curve
```

**Baseline for PR curve:** a random classifier has AUC-PR equal to the positive class proportion (e.g., 1% if positives are 1% of data). This is much more informative than the 0.5 baseline of ROC.

### ROC-AUC vs PR-AUC

| | ROC-AUC | PR-AUC |
|---|---|---|
| Baseline | 0.5 (random) | π (positive rate) |
| Affected by class imbalance | Weakly — can look good despite poor precision | Strongly — directly reflects real-world performance |
| Use when | Classes are roughly balanced | Positives are rare (fraud, disease, anomaly detection) |
| Threshold-free | Yes | Yes |
| What it hides | Low precision under imbalance | Nothing — it is the harsher judge |

**🎯 Rule of thumb for interviews:** If the dataset is imbalanced, default to PR-AUC. If the interviewer mentions "we optimized for AUC," immediately ask "ROC-AUC or PR-AUC?" This is a staff-level signal.

---

## 🧮 Calibration and Brier Score

A model is **calibrated** if its predicted probabilities match observed frequencies. If it says "80% chance of spam" for a group of emails, roughly 80% of those emails should actually be spam.

### Brier Score

```
🧮 Brier score:

    BS = (1/N) × Σ (pᵢ - yᵢ)²

    Where:
      pᵢ = predicted probability for example i
      yᵢ = true label (0 or 1) for example i
      N  = number of examples

    BS ranges from 0 (perfect) to 1 (worst possible)
```

Brier score is the mean squared error of predicted probabilities. It rewards both correct ranking AND calibration.

### Reliability Diagrams

To check calibration visually:
1. Bin examples by predicted probability (e.g., 0.0–0.1, 0.1–0.2, ..., 0.9–1.0)
2. For each bin, compute the actual positive rate
3. Plot predicted probability vs actual positive rate
4. A perfectly calibrated model lies on the diagonal

**Common calibration failures:**
- **Overconfident:** predicted probabilities are more extreme than reality (0.95 when true rate is 0.70)
- **Underconfident:** predicted probabilities are too close to 0.5 (0.60 when true rate is 0.90)
- **Shifted:** the model is systematically too high or too low

**Post-hoc calibration methods:**
- **Platt scaling:** fit a logistic regression on the model's logits using a held-out calibration set
- **Temperature scaling:** divide logits by a learned scalar T before softmax (T > 1 reduces confidence)
- **Isotonic regression:** non-parametric, more flexible, needs more calibration data

---

## ⚠️ Failure Modes

### 1. Class Imbalance Trap

**What happens:** accuracy looks excellent because the model learns to predict the majority class.

**Example:** 1% positive rate. Always predict negative → 99% accuracy. F1 = 0%, recall = 0%.

**How to detect:** check per-class metrics. If recall for any class is near zero, the model has learned to ignore it.

**How to fix:** use stratified sampling, class weights, oversampling (SMOTE), or switch metrics to macro F1 or PR-AUC.

### 2. Threshold Sensitivity

**What happens:** the model's ranking is good, but the default 0.5 threshold is wrong for the problem.

**Example:** a fraud detector with 0.1% fraud rate. At threshold 0.5, recall is 60%. At threshold 0.3, recall jumps to 90% with acceptable precision loss.

**How to detect:** plot the PR curve. If the curve is far above the baseline, the model is good — the threshold is the problem.

**How to fix:** choose threshold based on the business cost of FP vs FN, not the arbitrary 0.5 default. Some methods: maximize F1 on validation set, or use cost-sensitive threshold selection.

### 3. Label Noise

**What happens:** ground truth labels are wrong for some examples. The model learns from noisy labels, and evaluation against noisy labels produces unreliable metrics.

**Example:** 5% of "spam" labels are actually legitimate emails (annotation error). Precision appears lower than it truly is because some "false positives" are actually correct predictions with wrong labels.

**How to detect:** have multiple annotators label a subset. Compute inter-annotator agreement (Cohen's Kappa). If agreement is low, the labels themselves are unreliable.

**How to fix:** clean labels, use confident learning to identify likely mislabeled examples, report metrics with confidence intervals.

### 4. Metric Gaming

**What happens:** the model or training process is optimized to inflate the chosen metric without improving real performance.

**Example:** optimizing for accuracy on an imbalanced dataset leads the model to ignore rare classes entirely. Or, optimizing for recall without a precision constraint creates a model that predicts positive for everything.

**How to detect:** always report multiple complementary metrics. Accuracy + F1 + per-class recall at minimum.

**How to fix:** use composite metrics (F1 balances precision and recall). Set minimum thresholds for each metric during model selection.

### 5. Train-Evaluation Leakage

**What happens:** information from the test set leaks into training, making metrics unrealistically optimistic.

**Example:** normalizing features using statistics from the entire dataset (including test set). Or time-series data split randomly instead of chronologically, letting the model "see the future."

**How to detect:** metrics that are dramatically better than expected. Performance drops sharply in production vs evaluation.

**How to fix:** strict train/validation/test splits. For time series, use temporal splits. Never compute any statistic on test data before final evaluation.

---

## 📊 Computational Complexity

| Metric | Time Complexity | Space Complexity | Notes |
|--------|----------------|------------------|-------|
| Confusion matrix | O(N) | O(C²) | One pass through predictions; C = number of classes |
| Precision/Recall/F1 | O(N) | O(C) | Computed from confusion matrix |
| Accuracy | O(N) | O(1) | Single-pass counter |
| ROC curve | O(N log N) | O(N) | Sort predictions, then sweep thresholds |
| AUC-ROC | O(N log N) | O(N) | Trapezoidal integration of ROC curve |
| PR curve | O(N log N) | O(N) | Sort predictions, sweep thresholds |
| Brier score | O(N) | O(1) | Single-pass sum of squared errors |
| Calibration (binned) | O(N) | O(B) | One pass; B = number of bins |

All metrics are fast relative to training. The bottleneck is never the metric computation itself — it is generating the predictions.

---

## 💡 Design Trade-offs

### Which Metric to Optimize During Training

| Decision | Option A | Option B |
|----------|----------|----------|
| Loss function vs metric | Cross-entropy (differentiable, smooth gradients) | F1 (non-differentiable, must use surrogate) |
| Single metric vs composite | Simpler model selection, clearer signal | Prevents gaming, but harder to optimize |
| Threshold-dependent vs independent | F1 at fixed threshold (simple, but threshold matters) | AUC (evaluates all thresholds, but harder to act on) |
| Calibrated vs uncalibrated | Probabilities are trustworthy (needed for decision-making) | Ranking is correct but probabilities are arbitrary |

### Multi-Class Strategies

| Strategy | When to use |
|----------|------------|
| One-vs-rest (OvR) | Each class has its own binary classifier. Simple, parallelizable. |
| One-vs-one (OvO) | Train C(C-1)/2 pairwise classifiers. Better for small C, expensive for large C. |
| Softmax (multinomial) | Single model, joint probability distribution over all classes. Standard for neural networks. |

---

## 🏭 Production Considerations

### Monitoring Metrics in Production

In production, you rarely have labels immediately. Strategies:
- **Delayed labels:** for some tasks (ad clicks, customer churn), the true label arrives hours or days later. Set up pipelines to compute metrics once labels arrive.
- **Proxy metrics:** use correlated signals (user engagement, complaint rate) as early indicators.
- **Drift detection:** monitor the distribution of predicted probabilities. If it shifts, the model's environment has changed even if you cannot measure accuracy directly.

### Threshold Selection in Practice

- Never use 0.5 as a default without justification
- Common methods: maximize F1 on validation set, maximize profit (requires cost model for FP and FN), or set recall floor then maximize precision
- The threshold should be re-evaluated periodically — optimal threshold can change as the input distribution shifts

### Fairness Metrics

For models affecting people (hiring, lending, criminal justice), standard metrics are not enough. You must also check:
- **Equalized odds:** TPR and FPR are equal across protected groups
- **Demographic parity:** positive prediction rate is equal across groups
- **Calibration across groups:** predicted probabilities are calibrated within each group

These often conflict — you cannot satisfy all fairness criteria simultaneously (Impossibility theorem by Chouldechova 2017 and Kleinberg et al. 2016).

---

## Staff/Principal Interview Depth

### Q1: A model achieves 95% accuracy on a binary classification task with 90% negative class. Is this good?

---
**No Hire**
*Interviewee:* "95% accuracy is pretty good. That's above 90% so the model is learning something."
*Interviewer:* The candidate does not recognize the imbalance issue. They compare accuracy to an arbitrary threshold rather than to the baseline.
*Criteria — Met:* none / *Missing:* baseline comparison, per-class analysis, awareness of imbalance

**Weak Hire**
*Interviewee:* "With 90% negatives, a majority-class classifier gets 90% accuracy, so 95% is only 5 percentage points above baseline. I'd want to see precision and recall for the positive class."
*Interviewer:* Good baseline awareness. But lacks specific next steps or alternative metrics.
*Criteria — Met:* baseline comparison, imbalance awareness / *Missing:* specific metrics to check, concrete recommendations

**Hire**
*Interviewee:* "The 90/10 split means accuracy baseline is 90%. The model's 95% is only a 50% error reduction. I'd immediately check: (1) recall on the positive class — is it above 80%? (2) precision — what's the false alarm rate? (3) macro F1 to see performance independent of class size. If this is a high-stakes application like fraud or disease, I'd also look at the PR curve since ROC-AUC can be misleadingly high under imbalance."
*Interviewer:* Solid analysis. Connects to the right metrics and explains the reasoning. Would be even stronger with cost-sensitivity discussion.
*Criteria — Met:* baseline, imbalance, per-class metrics, PR-AUC reasoning / *Missing:* cost-sensitive threshold selection, calibration consideration

**Strong Hire**
*Interviewee:* "95% accuracy at 90/10 imbalance gives error rate 5% vs baseline 10% — a 50% reduction, which is modest. The real question is: what are the costs of FP and FN? In fraud detection, missing a fraudulent transaction (FN) might cost $10,000 while a false alert (FP) costs $5 for manual review. I'd compute expected cost = FP × $5 + FN × $10,000 and select the threshold minimizing this. For the metric: PR-AUC over ROC-AUC because with 10% positives we're in imbalance territory where ROC can hide poor precision. I'd also check calibration — if we're using the model's probability for downstream decisions (like setting review priority), Platt scaling on a held-out calibration set is standard. Finally, I'd check per-group fairness if this affects people — equalized odds across demographic groups."
*Interviewer:* Full depth. Cost-sensitive framing, correct metric choice with justification, calibration awareness, fairness consideration. Staff-level answer.
*Criteria — Met:* baseline, imbalance, cost-sensitive analysis, PR-AUC, calibration, fairness / *Missing:* nothing

---

### Q2: When would you choose macro F1 over micro F1, and when would that choice be wrong?

---
**No Hire**
*Interviewee:* "I usually just use F1. I'm not sure what macro and micro mean exactly."
*Interviewer:* Lacks fundamental knowledge of multi-class metrics.
*Criteria — Met:* none / *Missing:* micro/macro definitions, trade-off awareness

**Weak Hire**
*Interviewee:* "Macro F1 gives equal weight to each class. Micro F1 gives equal weight to each example. I'd use macro when classes are imbalanced so small classes aren't ignored."
*Interviewer:* Correct definitions and basic reasoning. But does not explore when this choice could go wrong.
*Criteria — Met:* correct definitions / *Missing:* failure cases, specific examples, design judgment

**Hire**
*Interviewee:* "Macro F1 treats all classes equally regardless of size. Use it when every class matters — for example, in a medical diagnosis system where a rare disease still needs to be detected. Micro F1 weights by class frequency, so it's essentially accuracy for multi-class problems. The failure case for macro: if one rare class has 3 examples and the model gets 2/3 right, that class's F1 has high variance. One mislabeled example swings macro F1 significantly. For unstable rare classes, weighted F1 or stratified bootstrapped confidence intervals are better."
*Interviewer:* Good. Identifies the variance problem with small classes. Would be stronger with a concrete production example and connection to business decisions.
*Criteria — Met:* definitions, trade-offs, failure mode / *Missing:* business context, alternative approaches

**Strong Hire**
*Interviewee:* "Macro F1 gives each class equal vote. This is correct when all classes are equally important — intent classification in a chatbot where every intent must work, or multi-disease diagnosis. But it can be wrong in two ways: (1) tiny classes with high F1 variance dominate the average — I've seen a class with 5 examples flip macro F1 by 3 points depending on the random seed. Fix: report bootstrap confidence intervals, or exclude classes below a minimum support threshold from the macro average and report them separately. (2) In e-commerce product classification with 1000 categories, some categories drive 60% of revenue. Macro F1 treats a niche category selling 2 items/year the same as the top category. Here you want revenue-weighted F1 or at minimum weighted F1 with class importance weights. The meta-point: there's no universal best averaging strategy. The choice should follow from 'which mistakes cost the most in this specific system,' not from a default."
*Interviewer:* Exceptional. Identifies both the statistical instability problem and the business-importance mismatch. Proposes concrete fixes for each. The meta-point about connecting metrics to business cost is staff-level thinking.
*Criteria — Met:* definitions, both failure modes, concrete fixes, business context, meta-principle / *Missing:* nothing

---

### Q3: Explain the difference between ROC-AUC and PR-AUC. When does each one lie to you?

---
**No Hire**
*Interviewee:* "ROC-AUC is the area under the ROC curve. It measures how well the model separates classes. I think PR-AUC is similar but uses precision and recall."
*Interviewer:* Surface-level. No understanding of when each metric fails.
*Criteria — Met:* basic definition of ROC-AUC / *Missing:* PR-AUC specifics, failure modes, imbalance connection

**Weak Hire**
*Interviewee:* "ROC plots TPR vs FPR. AUC is the area under it, 0.5 is random, 1.0 is perfect. PR curve plots precision vs recall. PR-AUC is better for imbalanced data because ROC can look good even when precision is low."
*Interviewer:* Correct high-level understanding and the key recommendation. But cannot explain WHY ROC misleads under imbalance.
*Criteria — Met:* correct definitions, imbalance recommendation / *Missing:* mathematical explanation of why, specific examples

**Hire**
*Interviewee:* "ROC uses FPR = FP/(FP+TN) on the x-axis. When TN is huge (many negatives), even large absolute FP counts produce small FPR. So ROC stays high. PR uses Precision = TP/(TP+FP) which directly feels the FP count. Example: 100 positives, 100,000 negatives. Model gets 90 TP, 500 FP. Recall = 90%, FPR = 500/100,000 = 0.5%. ROC looks amazing. Precision = 90/590 = 15%. PR-AUC would correctly flag this as a bad model. Conversely, PR-AUC can lie when the positive class is well-separated but very prevalent — though this is rare in practice."
*Interviewer:* The worked example is convincing. Correctly explains the mechanism. Good staff-level answer.
*Criteria — Met:* mechanism explanation, worked example, correct recommendation / *Missing:* probabilistic interpretation of AUC, when ROC is actually preferred

**Strong Hire**
*Interviewee:* "ROC-AUC has a clean probabilistic interpretation: it's the probability that a randomly chosen positive scores higher than a randomly chosen negative. This is threshold-independent and class-balance-independent — which is both its strength and its weakness. The problem: under 1% positive rate, you can have ROC-AUC of 0.99 while Precision at any useful recall level is below 10%. The FPR denominator is dominated by the massive TN count, hiding thousands of false positives. PR-AUC's baseline is the positive rate π, so 0.5 PR-AUC at π = 0.01 means 50x better than random, while 0.5 ROC-AUC means no better than random. When does ROC-AUC win? When the operating point requires high specificity — spam filtering where you need FPR < 0.001 — the ROC curve's left region is more informative than the PR curve. Also, ROC-AUC is better for comparing models when the positive rate differs across datasets, since the baseline doesn't shift. But for any deployment with rare positives — fraud, disease, anomaly — PR-AUC is the honest metric."
*Interviewer:* Complete. Probabilistic interpretation, mechanism of failure, numerical example, when ROC actually wins, and the baseline comparison point. This is staff-level depth.
*Criteria — Met:* everything / *Missing:* nothing

---

### Q4: You deploy a model and its F1 drops from 0.92 in evaluation to 0.78 in production. Walk me through your debugging process.

---
**No Hire**
*Interviewee:* "I'd retrain with more data to get the F1 back up."
*Interviewer:* Jumps to a solution without diagnosing the problem.
*Criteria — Met:* none / *Missing:* systematic debugging, root cause analysis

**Weak Hire**
*Interviewee:* "The data distribution probably shifted. I'd compare the production data distribution to the training data and see what changed."
*Interviewer:* Correct instinct, but no systematic framework. Does not consider other causes.
*Criteria — Met:* distribution shift hypothesis / *Missing:* systematic debugging, alternative causes, specific checks

**Hire**
*Interviewee:* "I'd check in order: (1) Data leakage in evaluation — was there test-train contamination? Re-split and re-evaluate. (2) Feature pipeline bugs — are production features computed the same way as training? Check for missing values, different encodings, stale features. (3) Distribution shift — compare input feature distributions between training and production. KL divergence or KS test on each feature. (4) Temporal shift — if the data is time-sensitive, the training data might be stale. (5) Subgroup analysis — maybe the model is fine overall but a new subpopulation appeared that it was never trained on."
*Interviewer:* Systematic and ordered by likelihood. Covers the major causes. Would be stronger with concrete metrics for monitoring each cause.
*Criteria — Met:* systematic approach, multiple hypotheses, actionable checks / *Missing:* monitoring infrastructure, long-term prevention

**Strong Hire**
*Interviewee:* "I'd attack this in layers. First, rule out evaluation bugs: replay a sample of production data through the offline evaluation pipeline. If offline F1 on production data is also 0.78, the model genuinely performs worse — no pipeline bug. If offline F1 is still 0.92, there's a serving bug (feature computation mismatch, model version mismatch, batch normalization in wrong mode). Second, characterize the drop: is it precision, recall, or both? Precision drop suggests new FP patterns (new types of negative examples the model hasn't seen). Recall drop suggests the positive class changed (concept drift). Third, slice the data: performance by time period (sudden vs gradual drop), by feature segments (geography, device, user type). Find where the gap concentrates. Fourth, check label quality: if production labels come from a different process than training labels (e.g., delayed feedback vs manual annotation), the metrics may not be comparable at all. For prevention: I'd set up automated monitoring with (a) prediction distribution dashboards, (b) feature drift detectors comparing rolling windows to training baseline, (c) per-slice performance tracking with alerting when any slice drops below threshold."
*Interviewer:* Exceptionally structured. The layered approach (bug vs model vs data vs labels) is exactly how senior engineers debug this in practice. The monitoring prevention plan shows production maturity.
*Criteria — Met:* systematic layers, precision/recall decomposition, slicing, label quality, monitoring plan / *Missing:* nothing

---

## Key Takeaways

🎯 1. Accuracy is only meaningful when classes are balanced — always check the baseline rate
🎯 2. F1 uses the harmonic mean because it penalizes extreme imbalances — arithmetic mean hides broken models
   3. Micro averaging weights examples equally, macro weights classes equally, weighted is proportional
🎯 4. Use PR-AUC instead of ROC-AUC when positives are rare — ROC hides poor precision under imbalance
   5. A well-calibrated model's predicted probabilities match observed frequencies — check with reliability diagrams
⚠️ 6. Five failure modes to know: class imbalance, threshold sensitivity, label noise, metric gaming, leakage
   7. In production, monitor prediction distributions even when you lack labels — distribution shift precedes metric drops
   8. The threshold should come from business costs (FP cost vs FN cost), not from the arbitrary 0.5 default
🎯 9. When an interviewer asks about metrics, the first question should always be: "what is the class distribution?"
