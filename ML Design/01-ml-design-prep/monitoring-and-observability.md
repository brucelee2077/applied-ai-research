# Monitoring and Observability

## Introduction

Staff candidates proactively bring up monitoring. It's one of the clearest signals of production experience. Anyone can design a model architecture on a whiteboard — but knowing what can go wrong after deployment, how to detect it, and what to do about it takes real-world experience.

The most dangerous failures in ML systems are silent. The model keeps serving predictions, latency stays normal, no errors in the logs — but the predictions are gradually getting worse because the data shifted, a feature pipeline broke, or a feedback loop amplified a bias. By the time someone notices, users have already been affected for days or weeks.

---

## What to Monitor

| Category | What to Track | Why It Matters | Alert Threshold |
|----------|---------------|----------------|-----------------|
| **Model performance** | Prediction quality (accuracy, NDCG, precision) | Detect model degradation | >5% drop from baseline |
| **Data quality** | Feature distributions, null rates, schema violations | Catch upstream data issues before they reach the model | Distribution shift PSI > 0.1 |
| **System health** | Latency (p50/p95/p99), throughput, error rates | Ensure SLA compliance | p99 > 2x target |
| **Business metrics** | Revenue, engagement, conversion rates | Connect model to business impact | >2% drop vs control |
| **Fairness metrics** | Performance across demographic groups | Detect and prevent bias amplification | >10% gap between groups |

### The Monitoring Stack

```
Raw Predictions → Logging → Metrics Pipeline → Dashboards → Alerting → Runbooks
         ↓                                           ↓
    Audit Trail                              Anomaly Detection
```

---

## Data and Feature Monitoring

Feature monitoring catches problems before they affect predictions.

### Feature Distribution Shift

Monitor the statistical distribution of each input feature and alert when it changes.

| Metric | What It Detects | How It Works |
|--------|-----------------|-------------|
| PSI (Population Stability Index) | Overall distribution shift | Bin feature values, compare bin proportions. PSI > 0.1 = investigate, > 0.25 = alert. |
| KS Test (Kolmogorov-Smirnov) | Maximum distributional distance | Compare CDFs of training vs serving distributions. |
| KL Divergence | Information-theoretic distance | Asymmetric — measures how much serving distribution diverges from training. |

### Schema Validation

Catch structural changes in input data:
- **Type checking:** Did a numeric feature start receiving strings?
- **Range checking:** Are values within expected bounds? (Age = 250 is wrong.)
- **Null rate monitoring:** Did the null rate for a feature spike from 1% to 50%?
- **Volume monitoring:** Did the number of incoming records drop by 90%?

These are cheap to implement and catch the most common upstream failures.

### Feature Coverage

What percentage of predictions have each feature available? If a critical feature drops from 99% coverage to 60%, the model is running on partial information for 40% of requests — and the predictions for those requests are likely degraded.

---

## Model Performance Monitoring

### Online vs Offline Metric Divergence

Your offline metrics (computed on held-out test sets) and online metrics (computed on live traffic) should be correlated. When they diverge, something is wrong:

| Scenario | Likely Cause |
|----------|-------------|
| Offline improves, online degrades | Training-serving skew, feature pipeline difference |
| Offline stable, online degrades | Data drift in serving data, upstream change |
| Both degrade | Model is stale, needs retraining |
| Offline degrades, online stable | Test set is stale or unrepresentative |

### Delayed Feedback

For some systems, ground truth arrives days or weeks later:
- **Ad conversions:** User might purchase days after clicking
- **Content quality:** A video's long-term engagement pattern takes weeks to stabilize
- **User retention:** Whether a recommendation led to a return visit is only known later

**Solution:** Track proxy metrics in real-time (click-through rate, dwell time) alongside delayed true metrics (conversions, retention). Alert on proxy metric changes immediately; validate with true metrics later.

### Segmented Monitoring

Overall metrics can hide problems in specific segments:

> "Our model's overall accuracy is 94% and stable. But when I slice by user tenure, accuracy for new users (< 7 days) dropped from 88% to 72% last week. The new user onboarding flow changed and broke a feature."

Monitor separately for: user tenure, platform (mobile/desktop/web), country/language, content type, traffic source.

---

## Alerting and Response

### Alert Design

Too many alerts → alert fatigue → real alerts get ignored. Too few → real problems go undetected.

| Severity | When to use | Response | Example |
|----------|-------------|----------|---------|
| **Informational** | Unusual but not urgent | Review next business day | Feature distribution shifted slightly |
| **Warning** | Potential problem, needs investigation | Investigate within hours | Prediction quality dropped 3% |
| **Critical** | Active degradation | Investigate immediately | Latency p99 exceeded 5x target |
| **Page** | User-impacting incident | Drop everything | Model serving errors > 1% of traffic |

### Runbooks

For every alert, write a runbook: a step-by-step procedure for diagnosing and resolving the problem.

A runbook should answer:
1. What does this alert mean?
2. What's the immediate user impact?
3. What should I check first? (dashboard links, log queries)
4. What are the common causes and their fixes?
5. When should I escalate?

### Automated Rollback

For model deployment, define automated rollback triggers:
- Prediction error rate exceeds threshold → revert to previous model version
- Latency exceeds 2x baseline for >5 minutes → revert
- Business metric (revenue, engagement) drops >X% → revert

Automated rollback catches problems faster than humans can — but you need to be confident your rollback mechanism itself is reliable.

---

## Common Failure Modes

### Silent Model Degradation

The model still serves predictions, no errors, latency is fine — but prediction quality is slowly declining.

**Causes:**
- Data drift: user behavior changed, but the model was trained on old patterns
- Feature pipeline change: an upstream team changed a data format or schema
- Concept drift: the relationship between features and labels changed (seasonal, cultural shifts)

**Detection:** Track prediction quality metrics over time with anomaly detection. Compare serving predictions against a shadow model or random baseline.

### Training-Serving Skew

The model sees different data at training time vs serving time.

**Causes:**
- Different code paths for feature computation (Python script for training, Java service for serving)
- Time-of-access differences (training uses daily snapshots, serving uses real-time values)
- Missing features at serving time that were available during training

**Detection:** Log serving-time features and periodically compare their distributions against training features.

**Prevention:** Use a feature store that serves the same features to both training and serving pipelines.

### Feedback Loop Amplification

The model's predictions influence future training data, amplifying existing patterns.

- Recommendation model promotes popular items → popular items get more engagement → model learns to promote them even more
- Content moderation model under-detects a category → that category grows → model falls further behind

**Detection:** Monitor diversity metrics (coverage, entropy) and segment-level performance over time. If diversity decreases monotonically, a feedback loop is likely active.

---

## Production ML Lifecycle

### Model Retraining Cadence

| Cadence | When to use | Infrastructure |
|---------|-------------|----------------|
| Daily | Fast-moving domains (news, trending, ads) | Automated pipeline with validation gates |
| Weekly | Moderate drift (recommendations, search) | Scheduled pipeline with human review |
| Monthly | Slow-moving domains (credit scoring, fraud) | Manual trigger with extensive validation |
| Triggered | When drift detection fires | Event-driven pipeline |

### Model Validation Before Deployment

Never deploy a model without automated validation:
1. **Offline metric gates:** New model must match or exceed previous model on held-out test set
2. **Canary deployment:** Route 1% of traffic to new model, monitor metrics, gradually increase
3. **Shadow comparison:** Run new model in parallel with production, compare outputs
4. **A/B test:** Full experiment with user-level randomization (for significant changes)

### Incident Postmortems

After every production ML incident:
1. What happened? (Timeline, impact)
2. Why did it happen? (Root cause)
3. Why wasn't it caught sooner? (Monitoring gap)
4. What will we do to prevent recurrence? (New monitors, tests, or processes)

Update your monitoring and alerting after every postmortem. The monitoring system should be a living artifact that improves with every failure.

---

## What is Expected at Each Level?

### Mid-Level Engineer

Mid-level candidates should mention that monitoring is needed and identify the basic categories: model accuracy, latency, and error rates. For a recommendation system, they should propose tracking CTR and latency as primary serving metrics. They differentiate by showing awareness that models can degrade over time and need retraining.

### Senior Engineer

Senior candidates proactively design a monitoring strategy. They discuss data quality monitoring (feature distributions, null rates), segmented metric tracking, and training-serving skew detection. For a content moderation system, a senior candidate would propose monitoring precision/recall by content category, tracking false positive rates per policy type, and setting up alerts for distribution shifts in incoming content. They bring up retraining cadence and model validation gates without being prompted.

### Staff Engineer

Staff candidates treat monitoring as a system design problem, not an afterthought. They recognize that the most dangerous ML failures are silent and propose proactive detection mechanisms: shadow models that continuously evaluate serving quality, automated drift detection that triggers retraining, and diversity metrics that detect feedback loops before they cause harm. A Staff candidate might point out that the hardest monitoring problem isn't detection — it's attribution. When a business metric drops, is it the model, the features, the data, or an external factor? They design monitoring systems that support rapid root cause analysis, not just alerting.
