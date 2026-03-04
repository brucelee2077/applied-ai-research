# Fairness and Bias

## Introduction

Fairness questions come up in nearly every Staff-level ML system design interview — and the candidates who handle them well are the ones who bring them up first. When you proactively mention fairness as a design constraint, it signals that you've built systems that affect real people and that you understand the consequences of getting it wrong.

The fundamental challenge: ML models learn patterns from historical data, and historical data reflects the world as it was — including its biases. A hiring model trained on past hiring decisions will learn existing biases against underrepresented groups. A content moderation system trained on English-language data will perform worse on other languages. A recommendation system that optimizes engagement will create filter bubbles. These aren't bugs in the model — they're the model doing exactly what it was trained to do. Fixing them requires intentional design decisions.

---

## Types of Bias in ML Systems

Bias enters ML systems through multiple pathways. Knowing where bias originates is the first step toward addressing it.

| Bias Type | What Happens | Example | When It Enters |
|-----------|-------------|---------|---------------|
| Historical bias | Training data reflects past societal biases | Hiring data where women were historically underrepresented in engineering roles | Data collection |
| Representation bias | Certain groups are underrepresented in training data | Image recognition trained mostly on light-skinned faces performs poorly on dark-skinned faces | Data collection |
| Measurement bias | Features are proxies that correlate with protected attributes | ZIP code as a feature correlates with race; vocabulary complexity correlates with education level | Feature engineering |
| Aggregation bias | Treating heterogeneous groups as homogeneous | One model for all users ignores that medical symptoms present differently across demographics | Model design |
| Feedback loop bias | Model predictions influence future data, amplifying existing biases | Model recommends less content from underrepresented creators → they get less engagement → model learns to recommend them less | Deployment |
| Selection bias | Training data only includes examples that passed a previous filter | Loan default model trained only on approved loans misses patterns in rejected applicants | Data collection |
| Label bias | Ground truth labels reflect human biases | Content moderation labels harsher on African American Vernacular English than equivalent white speech patterns | Labeling |

### Feedback Loop Bias in Detail

Feedback loops deserve special attention because they're the most dangerous — they're silent and self-reinforcing.

**How it works:**
1. Model is slightly biased against group X (perhaps from historical data)
2. Group X gets worse recommendations / lower rankings / fewer opportunities
3. Group X has less engagement (because the content shown to them is worse)
4. Lower engagement becomes negative training signal
5. Model becomes more biased against group X
6. Repeat — bias amplifies with each training cycle

**Detection:** Monitor model performance metrics disaggregated by demographic group over time. If performance gaps are widening monotonically, a feedback loop is likely active.

**Prevention:** Exploration mechanisms (show diverse content regardless of model confidence), periodic audits comparing model behavior across groups, separate training data collection that isn't influenced by model predictions.

---

## Fairness Definitions

There is no single definition of "fair." Different definitions capture different values, and they are often mathematically incompatible — satisfying one can require violating another.

### Group Fairness Definitions

| Definition | Formula | What It Requires | Intuition |
|-----------|---------|-----------------|-----------|
| Demographic parity | `P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)` | Equal positive prediction rates across groups | "Each group gets the same fraction of positive outcomes" |
| Equalized odds | `P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)` AND `P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)` | Equal TPR and FPR across groups | "Each group has the same accuracy" |
| Equal opportunity | `P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)` | Equal TPR across groups (relaxes FPR constraint) | "Qualified people in each group have equal chance of positive prediction" |
| Predictive parity | `P(Y=1 | Ŷ=1, A=0) = P(Y=1 | Ŷ=1, A=1)` | Equal precision across groups | "When the model says 'yes,' it's equally likely to be right for both groups" |

Where `A` is the protected attribute (e.g., gender, race), `Y` is the true label, and `Ŷ` is the prediction.

### Individual Fairness

**Principle:** Similar individuals should receive similar predictions, regardless of group membership.

`d_output(f(x_1), f(x_2)) ≤ L · d_input(x_1, x_2)`

If two people have similar qualifications (small input distance), their predictions should be similar (small output distance). The challenge: defining "similar" — what distance metric captures meaningful similarity without encoding bias?

### The Impossibility Theorem

Demographic parity, equalized odds, and predictive parity cannot all be satisfied simultaneously (except in trivial cases where base rates are equal across groups or the model is perfect).

**Why it matters for interviews:** When an interviewer asks about fairness, they don't want you to list all definitions. They want you to pick the right one for the problem and explain the tradeoff.

| Problem | Most Relevant Definition | Why |
|---------|------------------------|-----|
| Hiring | Equal opportunity | Qualified candidates from each group should have equal chance of being selected |
| Content moderation | Equalized odds | Both false positive rate (wrongly flagging benign content) and false negative rate (missing harmful content) should be equal across groups |
| Ads targeting | Demographic parity | Housing, employment, and credit ads must reach all demographics equally (legal requirement) |
| Recommendations | Exposure fairness (not a standard definition) | Content creators from each group should get proportional exposure relative to content quality |
| Loan approval | Predictive parity + equal opportunity | Among approved loans, default rates should be similar across groups; among qualified applicants, approval rates should be similar |

---

## Measuring Bias

### Disaggregated Evaluation

The most basic and most important step: compute all your standard evaluation metrics separately for each demographic group.

**Instead of reporting:**
> "Model accuracy: 94%"

**Report:**

| Group | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Group A | 96% | 94% | 97% | 95.5% |
| Group B | 89% | 85% | 92% | 88.4% |
| Group C | 82% | 78% | 85% | 81.4% |

Now you can see that the model works significantly worse for Group C. The 94% overall average was hiding a 14-point gap.

### Fairness Gap Metrics

| Metric | Formula | What It Captures |
|--------|---------|-----------------|
| Accuracy gap | `max(accuracy) - min(accuracy)` across groups | Worst-case accuracy difference |
| FPR ratio | `max(FPR) / min(FPR)` across groups | How much more often the worst group gets false positives |
| Four-fifths rule | `min(selection_rate) / max(selection_rate)` | Legal threshold: ratio < 0.8 indicates adverse impact (used in US employment law) |
| Equal opportunity gap | `max(TPR) - min(TPR)` across groups | Difference in ability to correctly identify positives |

### Intersectional Analysis

Bias may only appear at the intersection of multiple attributes. A model might perform equally well for men and women, and equally well for each racial group, but perform poorly for women of a specific racial background.

**Check intersections of:** gender × race, age × geography, language × socioeconomic status.

The number of intersections grows combinatorially, so focus on the intersections most relevant to your problem and most likely to be underrepresented in training data.

### Proxy Detection

Even if you remove protected attributes from the model's features, the model can still discriminate — other features may serve as proxies.

| Feature | What It Proxies | Why |
|---------|----------------|-----|
| ZIP code | Race, income | Residential segregation |
| First name | Gender, ethnicity | Naming patterns correlate with demographics |
| University attended | Socioeconomic status | University prestige correlates with family income |
| Language / dialect | Ethnicity, geography | Different communities use different language patterns |

**Detection:** Train a classifier to predict the protected attribute from the model's features. If it can predict well, proxies exist. Use feature importance on this classifier to identify which features are the strongest proxies.

---

## Mitigation Strategies

Bias can be addressed at three stages: before training (pre-processing), during training (in-processing), and after training (post-processing).

### Pre-Processing

Fix the data before the model sees it.

| Technique | How It Works | Pros | Cons |
|-----------|-------------|------|------|
| Re-sampling | Oversample underrepresented groups, undersample overrepresented groups | Simple, model-agnostic | Can cause overfitting on oversampled minority data |
| Re-weighting | Assign higher training weights to underrepresented groups | Simple, preserves all data | Doesn't fix quality issues in underrepresented data |
| Data augmentation | Generate synthetic examples for underrepresented groups | Increases effective dataset size | Generated examples may not capture real-world complexity |
| Representation learning | Learn fair representations that remove protected attribute information | Fixes bias at the representation level | May remove useful information correlated with protected attribute |

### In-Processing

Build fairness into the model's training objective.

| Technique | How It Works | Pros | Cons |
|-----------|-------------|------|------|
| Adversarial debiasing | Add a discriminator that tries to predict protected attribute from model output; penalize the model when the discriminator succeeds | Directly targets the bias signal | Harder to train (adversarial instability), no guarantee it removes all bias |
| Fairness constraints | Add fairness penalty to loss: `L_total = L_task + λ · L_fairness` | Explicit control over fairness-accuracy tradeoff | Need to choose λ (fairness-accuracy tradeoff), may reduce overall accuracy |
| Calibration constraints | Add per-group calibration terms to the loss | Ensures calibrated predictions for each group separately | Increases training complexity |

### Post-Processing

Adjust the model's outputs after training.

| Technique | How It Works | Pros | Cons |
|-----------|-------------|------|------|
| Threshold adjustment | Use different decision thresholds for different groups to equalize TPR or FPR | Simple, doesn't require retraining, easy to adjust | Requires knowing group membership at serving time |
| Per-group calibration | Calibrate predictions within each group separately | Ensures accurate probabilities for each group | Requires sufficient data per group for calibration |
| Score transformation | Apply group-specific score transformations | Flexible | Can feel like "moving the goalposts" |

**Which stage to intervene?**
- **Pre-processing** is best when the bias is in the data (representation gaps, label bias)
- **In-processing** is best when the bias comes from model behavior (learning proxy features)
- **Post-processing** is best when you need a quick fix without retraining, or when you need to meet specific fairness criteria

In practice, the most effective approaches combine interventions at multiple stages.

---

## Fairness in Specific Systems

### Recommendations and Search

**Exposure fairness:** Do all content creators get fair exposure relative to their content quality? A recommendation system that consistently promotes a small set of creators creates a winner-take-all dynamic that discourages new and diverse creators.

**Filter bubbles:** Engagement-optimized recommendations show users more of what they've already engaged with, reducing content diversity over time. This disproportionately affects users in smaller demographic groups whose preferences are underrepresented.

**Mitigation:** Diversity constraints in re-ranking (ensure no single creator dominates), exploration mechanisms (surface content from underrepresented creators), exposure audits (compare creator exposure to content quality metrics).

### Ads

**Legal requirements:** In the US, housing, employment, and credit ads must not discriminate based on protected characteristics (Fair Housing Act, EEOC, ECOA). Platforms must ensure these ad categories reach diverse audiences regardless of targeting.

**Technical implementation:** Restrict targeting options for sensitive ad categories, enforce minimum demographic reach, audit delivery algorithms for disparate impact.

### Content Moderation

**Cross-cultural norms:** What counts as harmful varies across cultures. A model trained primarily on US English data may flag legitimate speech in other dialects or cultural contexts.

**Disparate enforcement:** Studies have shown automated moderation disproportionately flags content from minority communities (e.g., AAVE being flagged as toxic at higher rates than equivalent non-AAVE speech).

**Mitigation:** Multilingual training data with diverse annotators, per-language/per-community calibration of moderation thresholds, human review for edge cases, regular disparate impact audits.

### Hiring

**Legal framework:** In the US, the four-fifths rule states that if the selection rate for a protected group is less than 80% of the group with the highest selection rate, adverse impact is presumed.

**Example:** If 60% of Group A applicants are selected and only 30% of Group B, the ratio is 30/60 = 0.5, which is below 0.8. This creates legal liability.

**Mitigation:** Monitor selection rates by demographic group throughout the pipeline, ensure model features don't encode demographic proxies, validate that the model's predictions are justified by job-relevant qualifications.

---

## The Fairness-Accuracy Tradeoff

Enforcing fairness constraints typically reduces overall model accuracy. This is a real tension that must be acknowledged and managed.

**Why the tradeoff exists:** If the true base rates differ between groups (e.g., different click rates across demographics due to historical patterns), a model that maximizes accuracy will exploit these differences. Constraining the model to treat groups equally prevents it from exploiting real statistical differences — which reduces raw accuracy.

**How to manage the tradeoff:**
- Quantify the accuracy cost of each fairness intervention
- Present the tradeoff to stakeholders: "We can reduce the accuracy gap from 14 points to 3 points at a cost of 2 points of overall accuracy"
- The right tradeoff depends on the domain: regulated industries (hiring, lending) have legal minimums; consumer products need to balance user experience with equity

> "In an interview, I'd frame it this way: the goal isn't zero bias — that's often not achievable without making the model useless. The goal is to make the tradeoff explicit, choose the fairness definition that best fits the problem, and demonstrate that you've measured and managed the impact on all groups."

---

## Interview Strategy

### How to Bring Up Fairness

Don't wait for the interviewer to ask. In the monitoring or evaluation section of your design, say something like:

> "For this content moderation system, I'd want to monitor false positive and false negative rates disaggregated by language and demographic group. If we see significant gaps — say, AAVE being flagged at 2x the rate of standard American English — that's a signal we need to investigate our training data and potentially add per-group calibration or threshold adjustment."

This shows production awareness without turning your design into a fairness lecture.

### What NOT to Do

- Don't claim you'd "just remove protected attributes." Proxy features will carry the same signal.
- Don't treat fairness as an afterthought. Mention it as a design constraint alongside latency, accuracy, and cost.
- Don't over-index on definitions. The interviewer wants to see that you can apply fairness thinking to a real problem, not that you memorized demographic parity vs equalized odds.
- Don't assume you know which groups are affected. Propose measuring first, then intervening based on what the data shows.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should recognize that ML models can produce biased outcomes and that monitoring for disparate impact is necessary. For a recommendation system, they should mention checking performance across demographic groups and ensuring the model doesn't systematically disadvantage certain users. They differentiate by showing awareness that removing protected attributes isn't sufficient — proxy features can carry the same information.

### Senior Engineer

Senior candidates proactively design fairness into the system. They can name specific bias types relevant to the problem (feedback loops in recommendations, label bias in content moderation) and propose concrete mitigation strategies: disaggregated evaluation, re-sampling for representation balance, per-group threshold adjustment. For a content moderation system, a senior candidate would discuss the risk of disparate enforcement across languages and demographics, propose multilingual evaluation splits, and bring up the tension between global consistency and cultural context in harm definitions.

### Staff Engineer

Staff candidates think about fairness as a systemic property, not just a model property. They recognize that the most impactful biases come from feedback loops and system-level effects, not from individual model decisions. A Staff candidate might point out that in a recommendation system, the biggest fairness concern isn't model accuracy — it's the feedback loop where the model's own recommendations shape future training data, creating a rich-get-richer dynamic for popular creators and a cold-start trap for underrepresented ones. They propose system-level interventions: exploration budgets, creator exposure audits, longitudinal fairness monitoring that tracks equity metrics over months and years, and organizational processes for responding when disparities are detected.
