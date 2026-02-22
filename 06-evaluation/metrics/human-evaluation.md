# Human Evaluation

## Why Do We Need Humans?

Imagine you wrote a poem and asked a calculator to grade it. The calculator could
count the words, check the spelling, and verify the rhyming pattern. But could it
tell you if the poem was **beautiful**? **Moving**? **Creative**?

No way! Some things only humans can judge.

**Human evaluation** is when we ask real people to read AI-generated text and rate
its quality. It's the "gold standard" of evaluation -- more reliable than automatic
metrics, but also more expensive and slower.

```
+-------------------------------------------------------------------+
|            Automatic Metrics vs. Human Evaluation                 |
|                                                                   |
|   Automatic (BLEU, ROUGE, etc.):                                  |
|     Speed:     FAST (seconds)                                      |
|     Cost:      FREE (just run the code)                            |
|     Quality:   Misses nuance                                       |
|     Best for:  Quick comparisons during development                |
|                                                                   |
|   Human Evaluation:                                                |
|     Speed:     SLOW (hours to days)                                |
|     Cost:      EXPENSIVE (pay human judges)                        |
|     Quality:   Catches subtlety and nuance                         |
|     Best for:  Final quality checks, creative tasks                |
+-------------------------------------------------------------------+
```

---

## When Do You Need Human Evaluation?

Not every task needs human judges. Here's a guide:

```
+-------------------------------------------------------------------+
|                  Do I Need Human Evaluation?                      |
|                                                                   |
|   Task: "What is 2 + 2?"                                         |
|   Answer: Either right or wrong                                   |
|   Need humans? NO -- a computer can check this                    |
|                                                                   |
|   Task: "Translate this sentence to French"                       |
|   Answer: Mostly one correct answer, some flexibility             |
|   Need humans? MAYBE -- BLEU works for rough checks,             |
|                humans help for nuance                              |
|                                                                   |
|   Task: "Write a creative story about a dragon"                   |
|   Answer: Infinite valid answers, quality is subjective           |
|   Need humans? YES -- only humans can judge creativity            |
|                                                                   |
|   Task: "Is this chatbot response helpful and safe?"              |
|   Answer: Depends on context, tone, accuracy                     |
|   Need humans? YES -- safety and helpfulness need human judgment  |
+-------------------------------------------------------------------+
```

**Rule of thumb:** The more subjective or creative the task, the more you need
human evaluation.

---

## What Do Human Judges Rate?

When you ask humans to evaluate AI output, you need to tell them WHAT to look for.
These are called **evaluation criteria** (or "dimensions").

### Common Evaluation Criteria

| Criterion | What the Judge Asks | Example |
|-----------|-------------------|---------|
| **Fluency** | "Does this sound natural? Is the grammar correct?" | "The cat sat on mat" -- not fluent (missing "the") |
| **Coherence** | "Does this make sense? Do the ideas connect logically?" | A summary that jumps between unrelated topics is incoherent |
| **Relevance** | "Does this actually answer the question / match the task?" | Asked about dogs, AI talks about cats -- not relevant |
| **Factual accuracy** | "Is the information correct?" | "The Earth is the 5th planet from the Sun" -- factually wrong |
| **Helpfulness** | "Would this response actually help someone?" | A technically correct but confusing answer isn't helpful |
| **Safety** | "Could this response cause harm?" | Responses that are offensive, biased, or dangerous |
| **Creativity** | "Is this original and interesting?" | A generic story vs. a unique one with surprising twists |

### Rating Scales

Judges typically use one of these scales:

```
+-------------------------------------------------------------------+
|                    Common Rating Scales                            |
|                                                                   |
|  LIKERT SCALE (most common):                                     |
|  Rate from 1-5:                                                   |
|    1 = Terrible                                                   |
|    2 = Poor                                                       |
|    3 = Okay                                                       |
|    4 = Good                                                       |
|    5 = Excellent                                                  |
|                                                                   |
|  BINARY:                                                          |
|    Yes/No, Good/Bad, Accept/Reject                                |
|                                                                   |
|  RANKING:                                                         |
|    "Which response is better, A or B?"                            |
|    (Called "pairwise comparison")                                  |
|                                                                   |
|  BEST-WORST:                                                      |
|    "Of these 3 responses, which is BEST                           |
|     and which is WORST?"                                          |
+-------------------------------------------------------------------+
```

---

## How to Set Up a Human Evaluation

Setting up a good human evaluation is like designing a fair exam.
Here are the key steps:

### Step 1: Define Clear Guidelines

Write instructions that are so clear, anyone can follow them. Ambiguous
guidelines lead to inconsistent ratings.

```
BAD guideline:   "Rate how good the response is"
                  (What does "good" mean? Good grammar?
                   Good information? Good humor?)

GOOD guideline:  "Rate the FLUENCY of the response from 1-5.
                  A score of 5 means the text sounds like a
                  native English speaker wrote it with no
                  grammar errors. A score of 1 means it has
                  many grammar errors and sounds unnatural."
```

### Step 2: Include Examples

Show judges what each rating looks like. These are called **anchor examples**.

```
+-------------------------------------------------------------------+
|                     Fluency Anchor Examples                       |
|                                                                   |
|  Score 5: "The cat sat quietly on the warm windowsill,           |
|            watching the birds outside."                           |
|                                                                   |
|  Score 3: "The cat was sit on window, watching bird."            |
|                                                                   |
|  Score 1: "Cat window sit warm bird the watching is on."         |
+-------------------------------------------------------------------+
```

### Step 3: Choose Your Judges

```
+-------------------------------------------------------------------+
|                    Who Should Judge?                               |
|                                                                   |
|  CROWDSOURCE WORKERS (e.g., Amazon Mechanical Turk):             |
|    Pro: Cheap, fast, lots of judges                               |
|    Con: Variable quality, may rush through                        |
|    Best for: Simple yes/no judgments                               |
|                                                                   |
|  DOMAIN EXPERTS (e.g., doctors for medical AI):                   |
|    Pro: High quality, can catch subtle errors                     |
|    Con: Expensive, hard to find, slow                             |
|    Best for: Specialized tasks (legal, medical, scientific)       |
|                                                                   |
|  REGULAR USERS:                                                   |
|    Pro: Represents actual end users                               |
|    Con: May not follow guidelines carefully                       |
|    Best for: Chatbot/assistant evaluations                        |
+-------------------------------------------------------------------+
```

### Step 4: Use Multiple Judges

Never rely on just one person's opinion. Use at least 3 judges per item
and average (or majority-vote) their ratings.

---

## Inter-Annotator Agreement (IAA)

This is a fancy term for: **"Do the judges agree with each other?"**

If Judge A says the response is a 5 and Judge B says it's a 2, something
is wrong -- either the guidelines are unclear, or the judges need more training.

### Why IAA Matters

```
+-------------------------------------------------------------------+
|              High Agreement vs. Low Agreement                     |
|                                                                   |
|  HIGH AGREEMENT (good!):                                          |
|    Judge A: 4    Judge B: 4    Judge C: 5                        |
|    "All judges roughly agree. We can trust this rating."          |
|                                                                   |
|  LOW AGREEMENT (problem!):                                        |
|    Judge A: 1    Judge B: 5    Judge C: 3                        |
|    "Judges disagree wildly. The guidelines might be unclear,      |
|     or the task is very subjective."                              |
+-------------------------------------------------------------------+
```

### Measuring Agreement: Cohen's Kappa

The most common measure of agreement is **Cohen's Kappa**. It's a number
between -1 and 1.

```
What's special about Cohen's Kappa?

Regular agreement just asks: "What % of the time did judges agree?"
But this has a problem -- judges might agree just by CHANCE.

If there are only 2 choices (Yes/No), two random judges would agree
50% of the time just by luck!

Cohen's Kappa adjusts for chance:

                  Actual agreement - Chance agreement
  Kappa = ------------------------------------------------
                    1 - Chance agreement

In simple terms: "How much better is the agreement compared to
random guessing?"
```

**Interpreting Kappa:**

| Kappa Score | Agreement Level | What It Means |
|-------------|----------------|---------------|
| < 0.0 | Less than chance | Judges are contradicting each other |
| 0.0 - 0.20 | Slight | Almost random -- guidelines need work |
| 0.21 - 0.40 | Fair | Some agreement, but lots of disagreement |
| 0.41 - 0.60 | Moderate | Reasonable -- acceptable for some tasks |
| 0.61 - 0.80 | Substantial | Good -- guidelines are working well |
| 0.81 - 1.00 | Almost perfect | Excellent -- judges consistently agree |

**For most NLP evaluation, aim for Kappa > 0.6 (substantial agreement).**

---

## Common Evaluation Setups

### A/B Testing (Pairwise Comparison)

The simplest and most reliable method. Show the judge two responses
and ask: "Which one is better?"

```
+-------------------------------------------------------------------+
|                    A/B Test Setup                                  |
|                                                                   |
|  Prompt: "Explain why the sky is blue"                            |
|                                                                   |
|  Response A:                                                      |
|  "The sky appears blue because molecules in the atmosphere        |
|   scatter shorter wavelengths of light (blue) more than longer    |
|   wavelengths (red). This is called Rayleigh scattering."        |
|                                                                   |
|  Response B:                                                      |
|  "The sky is blue because of the way sunlight interacts with      |
|   our atmosphere. Think of sunlight as a mix of all colors.       |
|   Blue light bounces around more in the air, so that's the        |
|   color we see the most when we look up."                         |
|                                                                   |
|  Judge's choice:  [ ] A is better                                 |
|                   [ ] B is better                                 |
|                   [ ] Tie                                          |
+-------------------------------------------------------------------+
```

**Advantages:**
- Easy for judges (just pick the better one)
- High agreement between judges
- Don't need to define what "4 out of 5" means

**Disadvantage:**
- Need many comparisons if you have many models

### Likert Scale Rating

Rate each response independently on a scale (usually 1-5).

```
  Rate this response for HELPFULNESS:

  Response: "Photosynthesis is the process by which plants
  convert sunlight into energy..."

  [ ] 1 - Not helpful at all
  [ ] 2 - Slightly helpful
  [ ] 3 - Moderately helpful
  [ ] 4 - Very helpful
  [ ] 5 - Extremely helpful
```

**Advantage:** Can rate many models at once
**Disadvantage:** Different judges interpret scales differently
("My 3 might be your 4")

---

## LLM-as-Judge: A New Approach

A recent trend is using **another AI to judge AI outputs**. Instead of paying
humans, you ask a powerful LLM (like GPT-4 or Claude) to rate responses.

```
+-------------------------------------------------------------------+
|                     LLM-as-Judge Setup                            |
|                                                                   |
|   You give the LLM judge:                                         |
|     1. The original question                                      |
|     2. The response to evaluate                                   |
|     3. Clear rating criteria                                      |
|     4. A rating scale                                             |
|                                                                   |
|   The LLM returns:                                                |
|     - A rating (e.g., 4/5)                                        |
|     - An explanation of why                                       |
|                                                                   |
|   Pros:                   Cons:                                   |
|   - Fast and cheap        - Can be biased (prefers its own style) |
|   - Consistent ratings    - May miss subtle issues                |
|   - Scalable              - Not a replacement for human judgment  |
|                             for high-stakes decisions              |
+-------------------------------------------------------------------+
```

LLM-as-Judge is useful for **rapid iteration during development**, but for
final evaluations and published results, human evaluation is still preferred.

---

## Common Pitfalls to Avoid

```
+-------------------------------------------------------------------+
|              Common Mistakes in Human Evaluation                  |
|                                                                   |
|  1. VAGUE GUIDELINES                                              |
|     "Rate how good this is"  -->  Good HOW? Be specific!         |
|                                                                   |
|  2. TOO FEW JUDGES                                               |
|     Using 1 judge  -->  Use at least 3 per item                  |
|                                                                   |
|  3. NO CALIBRATION                                                |
|     Judges jump right in  -->  Do a practice round first         |
|     with known examples so judges calibrate their scales          |
|                                                                   |
|  4. JUDGE FATIGUE                                                 |
|     Evaluating 500 items  -->  Judges get tired and sloppy       |
|     Keep sessions under 1 hour, take breaks                       |
|                                                                   |
|  5. ORDER BIAS                                                    |
|     Always showing Model A first  -->  Randomize the order!      |
|     Judges tend to prefer whichever they see first                |
|                                                                   |
|  6. NOT MEASURING AGREEMENT                                       |
|     Skipping IAA  -->  Always compute Cohen's Kappa              |
|     If judges disagree, your results aren't reliable              |
+-------------------------------------------------------------------+
```

---

## Summary

```
+------------------------------------------------------------------+
|                Human Evaluation Cheat Sheet                       |
|                                                                  |
|  What:     Real people judge AI output quality                   |
|  When:     Subjective tasks (creativity, helpfulness, safety)    |
|  How:      Define criteria, write guidelines, use 3+ judges      |
|  Measure:  Use Cohen's Kappa to check judge agreement            |
|                                                                  |
|  Methods:                                                        |
|    A/B testing -- "Which is better, A or B?" (simplest)          |
|    Likert scale -- Rate 1-5 on specific criteria                 |
|    LLM-as-Judge -- Use AI to judge AI (fast but imperfect)       |
|                                                                  |
|  Golden rule: Clear guidelines + multiple judges + measure       |
|               agreement = reliable evaluation                     |
+------------------------------------------------------------------+
```

---

## Further Reading

- **Chatbot Arena** (lmsys.org) -- A large-scale human evaluation platform where users
  vote on which AI chatbot gives better responses
- **Best Practices for Human Evaluation of Machine Translation** -- A comprehensive
  guide to setting up rigorous human evaluations
- **Judging LLM-as-a-Judge** -- Research on when and how to use AI as an evaluator
- **Inter-Annotator Agreement** -- Artstein & Poesio, 2008 -- The definitive survey
  on measuring agreement between human judges

---

[Back to Metrics](./README.md) | [Back to Evaluation](../README.md)
