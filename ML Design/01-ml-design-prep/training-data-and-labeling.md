# Training Data and Labeling

## Introduction

Here's something that experienced ML engineers know but candidates rarely say in interviews: the data matters more than the model. A mediocre model trained on excellent data will beat an excellent model trained on mediocre data almost every time. Yet in interviews, most candidates spend 80% of their time on model architecture and 20% on data — the exact inverse of what production ML teams actually do.

This page covers how to think about training data in a system design interview: what signals are available, how to label data when ground truth is expensive, how to handle noise and bias, and when to invest in better data vs better models.

---

## Types of Training Signal

Every ML system needs labeled training data. The question is where those labels come from.

### Explicit Labels (Human-Annotated)

Humans look at examples and assign labels. "Is this image harmful? Yes/No." "Rate this translation from 1-5." "Is this the correct address for this business?"

- **Strengths:** Highest quality. You get exactly what you define.
- **Weaknesses:** Expensive ($0.05-$50 per label depending on complexity), slow (days to weeks for a batch), hard to scale.
- **When to use:** When the task is well-defined and you need high precision. Content moderation policies, search relevance judgments, medical diagnoses.

### Implicit Signals (User Behavior)

Users generate labels through their actions: clicks, purchases, dwell time, skips, saves, shares, follows, unfollows.

| Signal | What it indicates | Noise level | Example |
|--------|-------------------|-------------|---------|
| Click | Interest (maybe) | High — position bias, accidental clicks | User clicked on a search result |
| Purchase | Strong preference | Low — real commitment | User bought the product |
| Dwell time > 30s | Engagement | Medium — could be confusion | User spent 2 minutes reading an article |
| Skip (< 3s) | Disinterest | Medium — could be accidental | User scrolled past a video immediately |
| Save/bookmark | Future intent | Low | User saved a recipe |
| Share | Strong endorsement | Low | User shared an article with friends |
| Explicit rating | Stated preference | Medium — rating inflation, mood effects | User gave 4 stars |

- **Strengths:** Abundant, free, continuously generated.
- **Weaknesses:** Noisy, biased (position bias, presentation bias, selection bias), doesn't capture what users *would have liked* but never saw.
- **When to use:** At scale, when human labeling is too expensive. Recommendations, search ranking, ads.

> "Clicks are our most abundant signal, but they're noisy — a user clicking on position 1 tells us less about relevance than a user clicking on position 5. I'd use dwell time as a secondary signal to filter out accidental clicks."

### Programmatic Labels (Weak Supervision)

Write labeling functions — heuristic rules that automatically assign noisy labels. Then aggregate multiple noisy labels into a cleaner one.

- **How it works:** Write 10-50 labeling functions, each capturing a different heuristic. "If the text contains profanity → label as toxic." "If the sender domain is in a known spam list → label as spam." Each function is noisy on its own, but aggregated (using tools like Snorkel), they produce reasonably accurate labels.
- **Strengths:** Fast to iterate, scalable, captures domain expertise in code.
- **Weaknesses:** Limited by the quality and coverage of your heuristics. Fails on subtle cases.
- **When to use:** Bootstrapping a new task with no labeled data. Getting from 0 to "good enough" quickly.

### Semi-Supervised and Self-Supervised

- **Semi-supervised:** Train on a small labeled set + large unlabeled set. Pseudo-labeling: train a model on labeled data, predict labels for unlabeled data, add high-confidence predictions to training set, repeat.
- **Self-supervised:** Create labels from the data itself. Mask a word and predict it (BERT). Predict the next token (GPT). Predict whether two patches come from the same image (DINO).
- **When to use:** You have abundant unlabeled data but labeling is expensive. Pretraining before fine-tuning on your task-specific labeled data.

---

## Human Labeling

When you need high-quality labels, you need a well-designed annotation pipeline.

### Annotation Guideline Design

The single biggest factor in label quality is the annotation guidelines. Vague guidelines produce noisy labels.

**Bad guideline:** "Label this comment as toxic or not toxic."
**Good guideline:** "A comment is toxic if it contains any of the following: (a) threats of violence toward a person or group, (b) slurs targeting race, gender, or sexual orientation, (c) explicit encouragement of self-harm. Sarcasm and criticism of ideas (not people) are NOT toxic. See examples 1-10 for edge cases."

Rules for good guidelines:
1. Define every category with specific criteria, not subjective terms
2. Include 10+ labeled examples covering edge cases
3. Explicitly define what does NOT belong in each category
4. Update guidelines iteratively as annotators encounter ambiguous cases

### Inter-Annotator Agreement

How often do two annotators agree on the same label?

- **Cohen's kappa:** Agreement corrected for chance. κ > 0.8 is excellent, 0.6-0.8 is good, < 0.4 means your guidelines need work.
- **Fleiss' kappa:** Extends to more than two annotators.

If agreement is low, the problem is usually the guidelines — not the annotators. Fix the guidelines, add examples, hold calibration sessions.

### Quality Control Mechanisms

| Mechanism | How it works | What it catches |
|-----------|-------------|-----------------|
| Gold questions | Mix pre-labeled examples into the queue. Flag annotators who get them wrong. | Low-quality annotators, random clicking |
| Double labeling | Have 2-3 annotators label each example. Use majority vote or adjudication. | Individual annotator errors |
| Calibration sessions | Periodic meetings where annotators discuss disagreements. | Guideline ambiguity, concept drift in labeling |
| Annotator performance tracking | Track per-annotator accuracy over time. | Fatigue, declining quality |

### Crowdsourcing vs Expert Labeling

| | Crowdsourcing | Expert Labeling |
|---|---|---|
| Cost | $0.01-$0.50 per label | $1-$50 per label |
| Quality | Variable, needs quality control | High, consistent |
| Speed | Fast (thousands per hour) | Slow (tens per hour) |
| Best for | Simple tasks (sentiment, image classification) | Complex tasks (medical imaging, legal review) |
| Platforms | Amazon MTurk, Scale AI, Labelbox | Internal teams, specialized vendors |

---

## Implicit Feedback as Labels

Most production ML systems are trained on implicit signals, not human labels. This introduces several systematic biases.

### Click ≠ Relevance

A user clicking on a search result doesn't mean it was relevant. They might have:
- Clicked because it was in position 1 (position bias)
- Clicked because the snippet was misleading (clickbait)
- Clicked accidentally (fat finger on mobile)
- Not clicked on a great result because they never scrolled that far

### Position Bias Correction

Users click higher positions more, regardless of relevance. If you train on raw click data, your model learns "position 1 is good" instead of "relevant content is good."

**Inverse Propensity Scoring (IPS):** Weight each click by `1 / P(click | position)`. Clicks at lower positions (which are less likely) get higher weight, correcting for the bias.

**Position-aware models:** Add position as a feature during training, then set it to a constant during inference. The model learns to factor out position from its relevance prediction.

### Negative Sampling

In implicit feedback, you observe positives (things the user interacted with) but not true negatives (you don't know what the user would have disliked — you only know what they didn't see).

| Strategy | How it works | Tradeoff |
|----------|-------------|----------|
| Random negatives | Sample random items as negatives | Easy, but too easy for the model to distinguish |
| Impression-based negatives | Items shown but not clicked | Better, but confounded by position bias |
| Hard negatives | Items similar to positives but not interacted with | Best learning signal, but risk of false negatives |
| In-batch negatives | Other items in the same training batch | Efficient, but biased toward popular items |

### Counterfactual Learning

Training data reflects what the previous model showed. You can't learn about items the old model never surfaced. This creates a feedback loop where the model only gets better at predicting engagement for the types of content it already recommends.

Solutions:
- **Randomized data collection:** Show a small percentage of random items to get unbiased interaction data. Costs short-term engagement.
- **IPS reweighting:** Upweight interactions with items that were unlikely to be shown by the old model.
- **Off-policy evaluation:** Evaluate new models on logged data using importance sampling.

---

## Data Quality Issues

### Label Noise

Real-world labels are noisy. Annotators disagree. Implicit signals are ambiguous. Programmatic labels have errors.

**Impact:** Label noise hurts model calibration more than model accuracy. A model trained on 10% noisy labels might still rank items correctly, but its predicted probabilities will be unreliable.

**Mitigation:**
- Noise-robust losses: symmetric cross-entropy, generalized cross-entropy, forward correction
- Confident learning: identify and remove likely mislabeled examples using model predictions
- Label smoothing: replace hard labels (0/1) with soft labels (0.05/0.95) to account for inherent uncertainty

### Class Imbalance

In content moderation, less than 0.1% of content is harmful. In fraud detection, less than 0.01% of transactions are fraudulent. In click prediction, CTR is typically 1-5%.

| Technique | How it works | When to use |
|-----------|-------------|-------------|
| Oversampling (SMOTE) | Generate synthetic examples of the minority class | Small datasets, tabular features |
| Undersampling | Remove examples of the majority class | Large datasets where you can afford to discard data |
| Class-weighted loss | Multiply loss by inverse class frequency | Always a good starting point |
| Focal loss | `L = -(1-p)^γ · log(p)` — downweights easy examples, focuses on hard ones | When the model is too confident on easy negatives |

### Selection Bias

Your training data only contains what the previous system chose to show. Users only interact with items they were shown. You never observe what would have happened if a different set of items was shown.

This is a fundamental problem. It means your model learns "what works given the old model's choices," not "what would work if we could show anything."

---

## Data Augmentation

When you don't have enough training data, create more by modifying existing examples.

### Text Augmentation
- **Synonym replacement:** Swap words with synonyms. "The movie was great" → "The film was excellent."
- **Back-translation:** Translate to another language and back. Produces natural paraphrases.
- **Random insertion/deletion:** Remove or insert random words. Simple but effective regularizer.
- **LLM-based paraphrasing:** Use a language model to generate diverse paraphrases. Higher quality, higher cost.

### Image Augmentation
- **Geometric:** Random crop, flip, rotation, scaling. Always useful.
- **Color:** Jittering, brightness, contrast, saturation. Helps with lighting variation.
- **CutMix/MixUp:** Blend two images and their labels. Strong regularizer.
- **Domain-specific:** Weather simulation for autonomous driving, lighting variation for retail images.

### When Augmentation Helps vs Hurts
- **Helps:** Small datasets, when the augmentations preserve the label (flipping a cat image is still a cat).
- **Hurts:** When augmentations change the label (flipping a "turn left" sign makes it "turn right"), or when they introduce unrealistic artifacts that the model overfits to.

---

## Data Collection Strategy for Interviews

### How to Discuss Data in a System Design Interview

Most candidates jump straight to features and models. A strong candidate starts with data:

> "Before I talk about the model, I want to understand what training data we have. For a recommendation system, our strongest signal is probably user interaction logs — what users clicked, watched, or purchased. But I want to understand: how much of this data do we have? How far back does it go? What biases should I be aware of?"

### The Bootstrapping Path

What if you're starting a new ML system with no labeled data?

1. **Heuristics:** Start with simple rules (keyword matching for content moderation, popularity-based for recommendations). Use these to build the first version.
2. **Weak supervision:** Write labeling functions that capture domain expertise. Aggregate noisy labels.
3. **Active learning:** Use the initial model to identify the most informative examples to label. Get maximum value from each human label.
4. **Human review:** Label a small, high-quality dataset to evaluate and fine-tune the model.
5. **Scale with implicit feedback:** Once the system is live, use user behavior as training signal. But maintain a human-labeled evaluation set.

### The Cost-Signal Tradeoff

**Cheap noisy data** (clicks, logs) is abundant but biased. **Expensive clean data** (human labels) is accurate but scarce. The right strategy depends on the problem:

- For ranking and recommendations: mostly implicit signals, with a small human-labeled evaluation set
- For content moderation: human labels for policy-critical decisions, weak supervision for scale
- For medical/legal: expert labels are non-negotiable — the cost of errors is too high

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should identify the main data sources for a given problem and recognize the difference between explicit and implicit labels. For a recommendation system, they should mention user interaction logs as the primary signal and understand that clicks are noisy. They differentiate by proposing a reasonable training data pipeline — even if they don't cover every bias or quality issue.

### Senior Engineer

Senior candidates demonstrate awareness of data quality challenges. They proactively bring up position bias in click data, discuss negative sampling strategies, and recognize that class imbalance requires specific treatment (weighted loss, focal loss). For a content moderation system, a senior candidate would discuss the tradeoff between crowdsourced labeling (cheap, noisy) and expert labeling (expensive, accurate), and propose a quality control pipeline with gold questions and inter-annotator agreement monitoring.

### Staff Engineer

Staff candidates recognize that data quality is often the bottleneck, not model architecture. They think about the feedback loop between model predictions and future training data — how a biased model generates biased training data that reinforces its biases. A Staff candidate might propose that the highest-impact intervention isn't a better model, but instrumenting a new data source, implementing counterfactual logging to debias training data, or building an active learning pipeline that directs expensive human labeling budget to the examples where it matters most.
