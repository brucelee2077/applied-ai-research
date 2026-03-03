> **What this file covers**
> - 🎯 Inter-Annotator Agreement: why raw agreement is misleading and Kappa corrects for chance
> - 🧮 Cohen's Kappa (2 annotators), Fleiss' Kappa (3+ annotators), Krippendorff's Alpha — full derivations
> - 🧮 ELO rating systems for pairwise comparisons (Chatbot Arena)
> - ⚠️ 5 failure modes: order bias, position bias, anchoring, evaluator fatigue, LLM-as-Judge self-preference
> - 📊 Sample size and power analysis for human evaluation studies
> - 💡 Direct assessment vs pairwise comparison vs ranking — design trade-offs
> - 🏭 LLM-as-Judge: when it works, when it fails, calibration strategies
> - Staff/Principal Q&A with all four hiring levels shown

---

# Human Evaluation — Interview Deep-Dive

This file assumes you have read [human-evaluation.md](./human-evaluation.md) and have the intuition for rating scales, inter-annotator agreement, and A/B testing. Everything here is for Staff/Principal depth.

---

## 🗺️ Concept Flow

```
            Define evaluation criteria
            (fluency, helpfulness, safety, ...)
                        │
                        ▼
            Choose evaluation method
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
       Direct       Pairwise     Ranking
       Assessment   Comparison   (Best-Worst)
       (1-5 scale)  (A vs B)    (rank k items)
            │           │           │
            ▼           ▼           ▼
       Collect human judgments
       (3+ annotators per item)
                        │
                        ▼
       Measure inter-annotator agreement
       (Kappa, Alpha)
                        │
              ┌─────────┼─────────┐
              ▼                   ▼
        High agreement        Low agreement
        (κ > 0.6)            (κ < 0.4)
        → results reliable    → revise guidelines,
                               retrain annotators,
                               or simplify criteria
```

---

## 🧮 Cohen's Kappa (Two Annotators)

### Why Raw Agreement Is Misleading

Two annotators rate 100 items as "good" or "bad." They agree on 90 out of 100. Raw agreement = 90%. Sounds great?

Not necessarily. If 95% of items are "good," two random annotators who both always say "good" would agree 90.25% of the time by pure chance. The 90% agreement adds almost nothing beyond chance.

**Cohen's Kappa corrects for chance agreement.**

```
🧮 Cohen's Kappa:

    κ = (p_o - p_e) / (1 - p_e)

    Where:
      p_o = observed agreement (fraction of items where annotators agree)
      p_e = expected agreement by chance

    For binary labels (positive/negative):
      p_e = p₁₊ × p₊₁ + p₁₋ × p₊₋

      Where:
        p₁₊ = fraction of items annotator 1 labeled positive
        p₊₁ = fraction of items annotator 2 labeled positive
        p₁₋ = fraction of items annotator 1 labeled negative
        p₊₋ = fraction of items annotator 2 labeled negative
```

### Worked Example

```
                   Annotator 2
                   Good    Bad
Annotator 1  Good  80      5      → 85 total
             Bad   5       10     → 15 total
                   85      15     → 100 total

p_o = (80 + 10) / 100 = 0.90

p_e = (85/100 × 85/100) + (15/100 × 15/100)
    = 0.7225 + 0.0225
    = 0.745

κ = (0.90 - 0.745) / (1 - 0.745)
  = 0.155 / 0.255
  = 0.608

Interpretation: 90% raw agreement but only 0.608 Kappa.
The baseline was already 74.5% by chance.
```

### Kappa Interpretation Scale

| Kappa | Interpretation |
|-------|---------------|
| < 0 | Worse than chance (systematic disagreement) |
| 0.0 – 0.20 | Slight agreement |
| 0.21 – 0.40 | Fair agreement |
| 0.41 – 0.60 | Moderate agreement |
| 0.61 – 0.80 | Substantial agreement |
| 0.81 – 1.00 | Almost perfect agreement |

**🎯 Key insight:** κ < 0.4 means the evaluation criteria are too subjective or the guidelines are too vague. Fix the guidelines before trusting the labels.

---

## 🧮 Fleiss' Kappa (Three or More Annotators)

Cohen's Kappa works for exactly two annotators. When you have three or more annotators rating the same items, use Fleiss' Kappa.

```
🧮 Fleiss' Kappa:

    κ = (P̄ - P̄_e) / (1 - P̄_e)

    Where:
      N = number of items
      n = number of annotators per item
      k = number of categories

      For each item i, for each category j:
        n_ij = number of annotators who assigned item i to category j

      P̄ = (1 / (N × n × (n-1))) × Σᵢ Σⱼ n_ij × (n_ij - 1)
         = average pairwise agreement across all items

      P̄_e = Σⱼ p_j²
         where p_j = (1 / (N × n)) × Σᵢ n_ij
         = proportion of all assignments to category j
```

Fleiss' Kappa generalizes Cohen's Kappa but does NOT require the same annotators for every item. This is important for crowdsourced annotation where different workers rate different items.

---

## 🧮 Krippendorff's Alpha

Krippendorff's Alpha is the most general agreement measure. It handles:
- Any number of annotators
- Missing data (not every annotator rates every item)
- Different measurement scales (nominal, ordinal, interval, ratio)

```
🧮 Krippendorff's Alpha:

    α = 1 - D_o / D_e

    Where:
      D_o = observed disagreement
      D_e = expected disagreement by chance

    For nominal data:
      D_o = (1 / (n × N)) × Σ_items Σ_{c≠k} n_ic × n_ik / (n_i - 1)
      D_e computed from marginal category frequencies

    α = 1:   perfect agreement
    α = 0:   agreement equals chance
    α < 0:   systematic disagreement
```

**When to use which:**

| Metric | Annotators | Missing data | Scale types | Use when |
|--------|-----------|-------------|-------------|----------|
| Cohen's κ | Exactly 2 | No | Nominal | Two fixed annotators rate all items |
| Fleiss' κ | 3+ (variable) | No | Nominal | Crowdsourcing with rotating annotators |
| Krippendorff's α | Any | Yes | Any | General-purpose, missing data, ordinal/interval scales |

---

## 🧮 ELO Rating System for Pairwise Comparison

The Chatbot Arena (LMSYS) uses ELO ratings to rank language models from pairwise human preferences. This is the same system used in chess.

```
🧮 ELO update after a match:

    Expected score of A against B:
      E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    After the match:
      R_A_new = R_A + K × (S_A - E_A)

    Where:
      R_A, R_B = current ratings of A and B
      S_A = actual outcome (1 = A wins, 0.5 = tie, 0 = A loses)
      K = update factor (controls sensitivity to new results)
      400 = scaling constant (from chess)
```

**Why ELO for LLM evaluation?**

1. **Transitive ranking:** if A beats B and B beats C, ELO predicts A beats C with calibrated probabilities
2. **Efficient sampling:** you do not need every pair to be compared — ELO converges with a subset
3. **Dynamic updating:** new models can be added without re-evaluating all pairs
4. **Confidence intervals:** rating uncertainty shrinks with more comparisons

**Limitations of ELO for LLMs:**
- Assumes transitive preferences (but humans are not always transitive)
- Does not capture dimension-specific quality (model A may be better at math, B at creativity)
- Sensitive to the distribution of prompts used for comparison
- Order effects: the model shown first may have an advantage

---

## ⚠️ Failure Modes

### 1. Position Bias (Order Effects)

**What happens:** when comparing two outputs side by side, annotators tend to prefer the first one they read (primacy effect) or the last one (recency effect), independent of quality.

**How to detect:** randomly swap the order for each item. Compare win rates in position 1 vs position 2. If they differ significantly, position bias is present.

**How to fix:** always randomize presentation order. Report position-balanced results. If bias persists, use separate evaluation (rate each output independently, then compare ratings).

### 2. Anchoring Effect

**What happens:** the first item an annotator rates sets an implicit reference point. Subsequent items are judged relative to the first, not on their own merits.

**How to fix:** include calibration items at the start (items with known quality levels). Randomize item order per annotator.

### 3. Evaluator Fatigue

**What happens:** annotation quality degrades over time. After evaluating 50+ items, annotators become less careful, defaulting to "average" ratings.

**How to detect:** plot agreement per batch. If it drops in later batches, fatigue is present.

**How to fix:** limit sessions to 30-50 items. Insert attention checks (items with obviously correct answers). Remove annotators who fail attention checks.

### 4. LLM-as-Judge Self-Preference

**What happens:** when using an LLM to judge outputs from itself and other models, it tends to prefer its own outputs. GPT-4 as a judge rates GPT-4 outputs higher than Llama outputs, even when human judges disagree.

**How to detect:** compare LLM-judge rankings to human rankings. If the LLM consistently favors its own outputs, self-preference is present.

**How to fix:** use a different LLM as judge than the ones being evaluated. Or use multi-judge (multiple LLMs judging, aggregate results). Always calibrate against a subset of human judgments.

### 5. Guideline Ambiguity

**What happens:** vague evaluation criteria lead to different annotators applying different standards. "Rate helpfulness 1-5" means different things to different people.

**How to detect:** low inter-annotator agreement (κ < 0.4).

**How to fix:** provide concrete examples for each rating level. Include "anchor" examples: "This is a 2 because... This is a 4 because..." Run a calibration session where annotators discuss disagreements before the main evaluation.

---

## 📊 Sample Size and Power Analysis

### How Many Items to Evaluate

The required sample size depends on the expected effect size (how different the two systems are) and desired statistical power.

For pairwise comparison (A vs B, binary preference):

```
🧮 Sample size for proportion test:

    n = (z_{α/2} + z_β)² × 2 × p̄(1 - p̄) / (p_1 - p_2)²

    Where:
      p_1, p_2 = expected win rates for A and B
      p̄ = (p_1 + p_2) / 2
      z_{α/2} = z-score for significance level (1.96 for α = 0.05)
      z_β = z-score for power (0.84 for 80% power)
```

**Rules of thumb:**
- Detecting a 60/40 preference split: ~200 items
- Detecting a 55/45 preference split: ~800 items
- Detecting a 52/48 preference split: ~5,000 items

### How Many Annotators Per Item

- **Minimum:** 3 annotators (allows majority vote and Fleiss' Kappa)
- **Standard:** 5 annotators (reduces noise, allows outlier detection)
- **High-stakes:** 7+ annotators (medical, legal, safety-critical)

More annotators per item costs more but reduces measurement noise. More items with fewer annotators gives broader coverage. The trade-off depends on whether you need precise per-item scores or reliable aggregate comparisons.

---

## 💡 Design Trade-offs

### Evaluation Method Comparison

| Method | Strengths | Weaknesses | Best for |
|--------|-----------|------------|----------|
| Direct assessment (1-5 scale) | Rich signal per item, fine-grained | Scale calibration varies across annotators | Absolute quality measurement |
| Pairwise comparison (A vs B) | Simpler judgment, more reliable | Only relative ranking, O(n²) pairs | Comparing 2-3 systems |
| Best-worst scaling | Efficient, handles many items | More complex for annotators | Ranking many items |
| ELO-based (Chatbot Arena) | Dynamic, scalable, handles many models | Needs many comparisons, prompt-distribution-dependent | Large-scale model ranking |

### LLM-as-Judge vs Human Evaluation

| | LLM-as-Judge | Human Evaluation |
|---|---|---|
| Cost | $0.01-0.10 per item | $0.50-5.00 per item |
| Speed | Seconds per item | Minutes per item |
| Consistency | Perfect (deterministic at temp=0) | Variable (human disagreement) |
| Bias | Self-preference, verbosity preference | Position bias, fatigue |
| Captures | Factuality, coherence, format | Creativity, helpfulness, nuance |
| When to use | Development iteration, large-scale screening | Final quality assessment, subjective tasks |

**🎯 Key insight:** LLM-as-Judge is not a replacement for human evaluation. It is a fast, cheap proxy that works well for some criteria (factual accuracy, format compliance) and poorly for others (creativity, cultural sensitivity). Use it for development iteration; use humans for final assessment.

---

## 🏭 Production Considerations

### Building a Human Evaluation Pipeline

1. **Define criteria** — list the specific qualities to evaluate (max 3-5 per task to avoid fatigue)
2. **Write guidelines** — with concrete examples for each rating level
3. **Recruit annotators** — diverse backgrounds reduce bias
4. **Calibration round** — annotators evaluate the same 20 items, discuss disagreements
5. **Main annotation** — with attention checks every 20 items
6. **Compute agreement** — Fleiss' κ or Krippendorff's α
7. **Remove low-quality annotators** — those who fail attention checks or have κ < 0.2 with the group
8. **Aggregate** — majority vote for categorical, mean for ordinal (after outlier removal)

### Chatbot Arena Design

The LMSYS Chatbot Arena is the most influential human evaluation benchmark for LLMs. Key design decisions:

- **Blind evaluation:** users do not know which model generated each response
- **User-generated prompts:** not researcher-selected, avoiding prompt curation bias
- **Pairwise comparison:** "Which response is better?" — simpler than rating scales
- **ELO calculation:** with bootstrap confidence intervals over 100K+ comparisons
- **Category-specific ratings:** separate ELO for coding, math, reasoning, creative writing

---

## Staff/Principal Interview Depth

### Q1: You computed inter-annotator agreement and got Kappa = 0.35. What do you do?

---
**No Hire**
*Interviewee:* "0.35 seems decent. I'd move forward with the annotations."
*Interviewer:* Does not recognize that 0.35 indicates poor agreement.
*Criteria — Met:* none / *Missing:* Kappa interpretation, remediation steps

**Weak Hire**
*Interviewee:* "That's fair agreement, which isn't great. I'd try to improve the guidelines or train the annotators better."
*Interviewer:* Correctly identifies the problem but offers only vague remediation.
*Criteria — Met:* correct interpretation / *Missing:* specific debugging steps, root cause analysis

**Hire**
*Interviewee:* "κ = 0.35 is fair agreement — below the 0.6 threshold I'd want for reliable labels. Before investing more in annotation, I'd diagnose the cause: (1) analyze which items have the most disagreement — is it a subset of ambiguous items or widespread? (2) check if specific annotator pairs have low agreement — one bad annotator can drag down the average, (3) review the guidelines — are the rating criteria concrete enough? Do they include anchor examples? (4) run a calibration session where annotators discuss their disagreements on 10-20 items. After fixing the root cause, re-annotate a small batch and recompute κ."
*Interviewer:* Systematic debugging approach. Identifies multiple possible causes and actionable fixes.
*Criteria — Met:* correct interpretation, systematic diagnosis, actionable remediation / *Missing:* statistical analysis of disagreement patterns

**Strong Hire**
*Interviewee:* "κ = 0.35 means only about 35% of agreement beyond chance is captured. I would not use these labels for training or final evaluation — they are unreliable. My debugging process: (1) Compute pairwise κ for all annotator pairs to identify if one annotator is the outlier. If removing them brings κ > 0.6, replace them. (2) Confusion matrix of disagreements: are annotators confusing adjacent categories (3 vs 4 on a 5-point scale) or distant ones (1 vs 5)? Adjacent confusion suggests collapsing categories — maybe 5-point is too fine and 3-point would work. Distant confusion suggests fundamental disagreement about criteria. (3) Stratify by item type: if κ is 0.7 on factual questions and 0.1 on creativity questions, the problem is with the creativity criteria, not the whole study. (4) Revise guidelines with concrete examples at each rating level, run a calibration session, then re-annotate 50 items and verify κ > 0.6 before scaling up. If we cannot achieve κ > 0.6 even after revision, the evaluation criteria may be inherently too subjective for this task, and I'd switch to pairwise comparison (which typically has higher agreement than absolute rating)."
*Interviewer:* Complete diagnostic framework. Pairwise κ analysis, confusion matrix, stratification, category reduction, and method switching. Staff-level evaluation design.
*Criteria — Met:* everything / *Missing:* nothing

---

### Q2: How would you evaluate a conversational AI system where there is no single "correct" answer?

---
**No Hire**
*Interviewee:* "I'd use BLEU or ROUGE to compare the output to some reference answers."
*Interviewer:* Applies the wrong tool — automatic metrics do not capture conversational quality.
*Criteria — Met:* none / *Missing:* human evaluation necessity, multi-criteria framework

**Weak Hire**
*Interviewee:* "I'd use human evaluation. Have people rate the responses on quality — maybe 1 to 5."
*Interviewer:* Right direction but too vague. "Quality" is not a well-defined criterion.
*Criteria — Met:* human evaluation necessity / *Missing:* specific criteria, study design, agreement measurement

**Hire**
*Interviewee:* "For conversational AI with no single correct answer, I'd design a multi-criteria human evaluation: (1) Helpfulness — did the response address the user's need? (2) Factual accuracy — is the information correct? (3) Naturalness — does it sound like a human wrote it? (4) Safety — does it contain harmful or inappropriate content? I'd use a 5-point Likert scale per criterion with 3 annotators per item. Measure Fleiss' κ and require κ > 0.6. For model comparison, I'd also add pairwise evaluation: show users two responses and ask which they prefer. Report both absolute scores and pairwise win rates."
*Interviewer:* Well-structured evaluation plan with concrete criteria and methodology. Would be stronger with sample size considerations.
*Criteria — Met:* multi-criteria design, Likert scale, agreement, pairwise comparison / *Missing:* sample size, bias mitigation, ongoing monitoring

**Strong Hire**
*Interviewee:* "This is exactly the problem human evaluation was designed for. My design: multi-turn conversations with real users, evaluating on 4 criteria (helpfulness, accuracy, naturalness, safety) using a 3-point scale (not 5 — coarser scales get higher agreement). For each criterion, guidelines would include 2 anchor examples per rating level. Study design: 500 conversations (based on power analysis for detecting 5-point differences in mean ratings at α=0.05, power=0.80), 3 annotators per conversation, randomized presentation order. For model comparison: pairwise preference on the same 500 conversations, with position randomization. I'd report: (1) per-criterion mean scores with 95% CIs, (2) Fleiss' κ per criterion, (3) pairwise win rate with bootstrap CIs, (4) ELO if comparing more than 2 models. Bias mitigation: blind evaluation (annotators don't know which model produced what), position randomization, attention checks. For ongoing monitoring: I'd set up a Chatbot-Arena-style continuous evaluation where a fraction of production traffic gets pairwise comparison from users, giving us a real-time quality signal that adapts as the user population changes."
*Interviewer:* Production-grade evaluation design. Power analysis, scale choice justification, bias mitigation, multiple reporting formats, and continuous monitoring. Staff-level answer.
*Criteria — Met:* everything / *Missing:* nothing

---

### Q3: What are the strengths and limitations of using LLM-as-Judge?

---
**No Hire**
*Interviewee:* "LLM-as-Judge is great because it's cheap and fast. You can just use GPT-4 to rate outputs instead of hiring people."
*Interviewer:* Only sees the benefits. No awareness of limitations.
*Criteria — Met:* cost/speed benefits / *Missing:* any limitations

**Weak Hire**
*Interviewee:* "LLM-as-Judge is fast and cheap but has biases. For example, it might prefer longer responses or its own outputs. It's good for development but not for final evaluation."
*Interviewer:* Identifies key limitations but at surface level.
*Criteria — Met:* speed benefit, verbosity bias, self-preference / *Missing:* specific examples, calibration strategies, when it works well

**Hire**
*Interviewee:* "LLM-as-Judge trades cost for reliability. Strengths: 100x cheaper than human eval, perfectly consistent (no fatigue), can evaluate thousands of items in minutes. Limitations: (1) self-preference bias — GPT-4 as judge rates GPT-4 outputs 10-20% higher than human judges do, (2) verbosity preference — longer, more detailed responses score higher even when conciseness is better, (3) cannot reliably judge creativity, humor, or cultural sensitivity, (4) position bias — the response presented first tends to get higher scores. For calibration: evaluate a subset with both LLM and humans, compute correlation, and adjust the LLM threshold. Report the human-LLM agreement rate alongside results."
*Interviewer:* Quantified self-preference bias, identified multiple limitations, proposed calibration. Good answer.
*Criteria — Met:* strengths, multiple limitations, calibration / *Missing:* when it works well, specific production setup

**Strong Hire**
*Interviewee:* "LLM-as-Judge has a specific operating envelope. It works well for: factual accuracy checking, format compliance, instruction following, safety violations, and code correctness — tasks where there is an objectively assessable criterion. It fails for: creative writing quality, conversational engagement, cultural appropriateness, and subtle tone differences — tasks requiring human sensibility. Key biases: (1) self-preference (mitigate by using a different model family as judge), (2) verbosity preference (mitigate by explicitly instructing 'conciseness is valued'), (3) position bias (mitigate by evaluating each response separately rather than comparatively, or averaging forward and reverse order), (4) sycophancy — the judge may be reluctant to give harsh ratings. My production setup: use LLM-as-Judge for 100% of traffic (cheap screening), sample 5% for human evaluation (calibration), compute Pearson correlation between LLM and human scores per criterion, and only trust criteria where r > 0.7. If correlation drops below 0.7 for any criterion, that criterion gets human evaluation only. This gives you the speed of LLM-as-Judge where it's reliable and the accuracy of human evaluation where it's not."
*Interviewer:* Clear operating envelope, specific mitigation for each bias, hybrid production architecture with correlation-based trust. Staff-level systems design.
*Criteria — Met:* everything / *Missing:* nothing

---

## Key Takeaways

🎯 1. Raw agreement is misleading — always use Kappa or Alpha to correct for chance agreement
🎯 2. κ < 0.4 means the guidelines need revision before the labels can be trusted
   3. Cohen's κ for 2 annotators, Fleiss' κ for 3+, Krippendorff's α for missing data and ordinal scales
   4. Pairwise comparison is more reliable than absolute rating — humans are better at comparing than scoring
⚠️ 5. Five biases to know: position bias, anchoring, fatigue, LLM self-preference, guideline ambiguity
🎯 6. LLM-as-Judge works for factual/format tasks but fails for creativity/nuance — calibrate against humans
   7. Sample size of ~500 items detects 5-point mean differences at standard power
   8. The Chatbot Arena uses ELO with blind pairwise comparison — the current gold standard for LLM ranking
   9. Always randomize presentation order, include attention checks, and measure agreement
