# Experimentation and A/B Testing

## Introduction

Every ML system design interview ends with "how would you test this?" — and most candidates give a shallow answer. "We'd A/B test it" is not enough. Interviewers want to hear about randomization units, sample size, guardrail metrics, and the subtle ways experiments can go wrong.

Experimentation is where ML engineering meets product engineering. A model that improves offline metrics but degrades online metrics is worse than no change at all. Understanding how to design, run, and interpret experiments is what separates engineers who build things that ship from engineers who build things that sit on a shelf.

---

## A/B Testing Fundamentals

### Randomization Unit

The most important design decision: what gets randomized?

| Unit | Pros | Cons | Use When |
|------|------|------|----------|
| User | Consistent experience per user, clean measurement | Can't test features that affect all users on a page | Most recommendation, personalization experiments |
| Session | Can test within-session effects | Same user sees different treatments | Testing session-level features (layout, ranking) |
| Request | Maximum statistical power | Inconsistent user experience | Testing backend changes invisible to users |
| Geo/cluster | Handles network effects | Low statistical power, confounders | Marketplace, social network experiments |

**Rule of thumb:** Randomize at the user level unless you have a specific reason not to. Users seeing inconsistent experiences across sessions creates noise and frustration.

### Sample Size and Duration

Before launching, determine:
- **Minimum Detectable Effect (MDE):** What's the smallest improvement worth detecting? A 0.01% CTR improvement probably isn't worth shipping. A 1% improvement definitely is.
- **Statistical power (1-β):** Probability of detecting a real effect. Standard: 80%.
- **Significance level (α):** Probability of a false positive. Standard: 5%.
- **Required sample size:** Calculated from MDE, power, and baseline metrics. For small effects, you need millions of users.

**Duration considerations:**
- Run for at least 1-2 weeks to capture day-of-week effects
- Longer for experiments measuring long-term engagement (4-8 weeks)
- Check for novelty effects: initial improvement that fades as users habituate

### Metric Design

**Primary metric:** The one thing you're optimizing. Must connect to business value. Only one — having multiple primary metrics inflates false positive rates.

**Secondary metrics:** Additional signals that help interpret results. "CTR went up — but did dwell time also go up, or are we just generating more clickbait?"

**Guardrail metrics:** Things that must NOT degrade. Latency, crash rate, revenue, user complaints. If a guardrail metric degrades, the experiment fails regardless of primary metric improvement.

---

## Common A/B Testing Pitfalls

### Peeking

Checking results daily and stopping when you see significance. This inflates your false positive rate dramatically — you're essentially running multiple hypothesis tests.

**Fix:** Use sequential testing (always-valid p-values) that accounts for multiple looks. Or pre-commit to a fixed duration and sample size before launching.

### Multiple Comparisons

Running 20 variants means one will appear significant by chance at α=0.05.

**Fix:**
- Bonferroni correction: divide α by the number of comparisons (conservative)
- False Discovery Rate (FDR): control the proportion of false positives among significant results (less conservative)
- Pre-register your primary metric: one metric, one comparison

### Network Effects

In social networks and marketplaces, one user's treatment affects another user's outcomes. If I see better recommendations and share a video with you (in the control group), your engagement changes because of my treatment.

**Fix:** Cluster randomization — randomize at the level of social clusters or geographic regions. Lower statistical power but captures interference effects.

### Survivorship Bias

Users who don't like the new experience leave. You only measure outcomes for users who stuck around — making the treatment look better than it is.

**Fix:** Include all users who entered the experiment, not just those who completed it. Intent-to-treat analysis.

---

## Beyond Simple A/B Tests

### Multi-Armed Bandits

Instead of fixed 50/50 allocation, adaptively route more traffic to the winning variant.

- **Thompson sampling:** Maintain a posterior distribution for each variant's performance. Sample from the posteriors and choose the variant with the highest sample.
- **Advantage:** Reduces "regret" — less traffic goes to the losing variant.
- **Disadvantage:** Harder to get clean statistical significance. Useful when you care more about optimization than measurement.
- **When to use:** Testing many variants (e.g., headline variations), optimizing in real-time, when the cost of showing a bad variant is high.

### Interleaving Experiments

For ranking systems, interleave results from control and treatment in the same list. Each user sees a mix.

- **Advantage:** Much more sensitive than A/B testing — needs 10-100x fewer samples to detect the same effect size.
- **How:** Team Draft Interleaving — alternate between picking items from control and treatment lists.
- **When to use:** Comparing two ranking algorithms. Not suitable for testing entirely different UIs.

### Shadow Mode / Dark Launch

Run the new model in parallel with production without serving its results. Compare outputs offline.

- **Advantage:** Zero risk to users. Catches bugs and regressions before live traffic.
- **Disadvantage:** Can't measure user behavior responses (no engagement data for the shadow model).
- **When to use:** Before A/B testing — verify the model produces sensible outputs first.

---

## Offline Evaluation

### Test Set Construction

**Temporal splits, not random splits.** In any system with time dynamics (recommendations, search, ads), a random split leaks future information into the training set. Always split by time: train on data before time T, evaluate on data after T.

### Replay Evaluation

Simulate online behavior using logged data. Take the logged (query, shown items, user feedback) tuples and ask: "What would the new model have shown? If it showed the same items the user clicked on, count it as a win."

- **Limitation:** Can only evaluate items that were actually shown. Can't evaluate items the old model never surfaced.
- **Fix:** IPS-weighted replay evaluation — upweight rare items to correct for the old model's bias.

---

## Interpreting Results

### Statistical vs Practical Significance

A p-value < 0.05 means the result is unlikely due to chance. It does NOT mean the result is worth shipping.

- A 0.001% CTR improvement might be statistically significant with 100M users but not worth the engineering cost to maintain.
- Always report effect size and confidence intervals, not just p-values.

### Heterogeneous Treatment Effects

The average treatment effect can hide important segment-level differences:
- New users might love the change while power users hate it
- Mobile users might benefit while desktop users are harmed
- US users might improve while international users degrade

Always slice results by key segments: user tenure, platform, country, engagement level.

### Long-Term Effects

Some changes look good short-term but degrade long-term:
- Clickbait optimization increases CTR but decreases return visits
- Engagement optimization creates filter bubbles that reduce content diversity
- Aggressive recommendation diversity decreases short-term engagement but improves long-term retention

Consider running holdout experiments — keep a small percentage of users on the old system permanently to measure long-term divergence.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should understand the basics of A/B testing: randomization, sample size, and primary vs guardrail metrics. For a recommendation system, they should propose an A/B test with user-level randomization, identify appropriate metrics (CTR, dwell time, retention), and recognize that experiments need to run for at least a week. They differentiate by showing awareness of at least one pitfall (peeking, novelty effects).

### Senior Engineer

Senior candidates design experiments rigorously. They specify randomization units with justification, discuss metric hierarchies (primary, secondary, guardrail), and proactively bring up pitfalls like network effects or multiple comparisons. For a search system, a senior candidate would propose interleaving experiments for comparing ranking algorithms and explain why interleaving is more sensitive than A/B testing. They discuss offline evaluation with temporal splits as a prerequisite to online experiments.

### Staff Engineer

Staff candidates think about experimentation as an organizational capability, not just a statistical technique. They recognize that the biggest risks are not statistical — they're operational (experiments running too long, metric definitions drifting, teams cherry-picking results). A Staff candidate might propose an experimentation framework with automated guardrail checks, pre-registered analysis plans, and mandatory long-term holdouts for major model changes. They also understand the tension between experimentation velocity (ship fast, learn fast) and rigor (don't ship something that hurts users in ways you didn't measure).
