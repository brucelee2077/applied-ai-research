# Classification Metrics

A hospital builds an AI to detect a rare disease. Out of every 1,000 patients, only 10 actually have it. The AI team announces: "Our model is 99% accurate!" Everyone celebrates. But here is the problem — the model just says "no disease" for every single patient. It gets 990 out of 1,000 right by doing absolutely nothing useful. The 10 patients who actually need help? All missed.

99% accuracy. Completely useless. How is that possible?

The answer is that accuracy alone is not enough. You need a toolbox of metrics, each one checking a different kind of mistake. That is what this file is about.

---

**Before you start, you need to know:**
- What a model does — it takes input (like an image or text) and makes a prediction (like "cat" or "spam")
- What "correct" and "incorrect" mean for a prediction — the model either gets it right or wrong
- No math needed for this file

---

## The Grading Analogy

Imagine a teacher who grades a stack of essays as "pass" or "fail."

There are two ways the teacher can mess up:
- **False alarm:** The teacher fails a student who actually wrote a good essay. The student is upset — they did nothing wrong.
- **Missed catch:** The teacher passes a student who actually wrote a terrible essay. The student gets credit they do not deserve.

Different situations make different mistakes worse. If this is a final exam for medical school, missing a bad doctor-to-be (missed catch) is far more dangerous than accidentally re-testing a good student (false alarm).

**What the analogy gets right:** classification metrics are exactly about counting these two types of mistakes separately and deciding which one matters more for your situation.

**Where the analogy breaks down:** a real AI model gives predictions on thousands or millions of examples at once, not one essay at a time. The metrics are about patterns across all those predictions together.

---

## The Confusion Matrix — Your Score Card

Before we get to any metric, we need a way to organize all the predictions a model makes. That tool is called the **confusion matrix**. It is just a 2x2 table.

Say we have a spam filter. It looks at emails and says "spam" or "not spam." For each email, one of four things happens:

```
                        What the model predicted
                    +--------------+--------------+
                    |  "It's Spam" | "Not Spam"   |
    +---------------+--------------+--------------+
    | Actually Spam |     TP       |     FN       |
    |               | "Caught it!" | "Missed it!" |
    +---------------+--------------+--------------+
    | Actually NOT  |     FP       |     TN       |
    | Spam          | "False       | "Got it      |
    |               |  alarm!"     |  right!"     |
    +---------------+--------------+--------------+
```

- **TP (True Positive):** The model said spam, and it WAS spam. Correct!
- **FP (False Positive):** The model said spam, but it was NOT spam. A false alarm.
- **FN (False Negative):** The model said not spam, but it WAS spam. A missed catch.
- **TN (True Negative):** The model said not spam, and it was NOT spam. Correct!

Every metric you will learn is just a different way of combining these four numbers.

---

## Accuracy — The Simplest Metric

**Accuracy** is the percentage of predictions the model got right — both the spam it caught and the real emails it left alone.

Think of it like a test score: out of 100 questions, how many did you get right?

Accuracy works well when the classes are balanced — when there are roughly equal numbers of spam and non-spam emails. But remember the hospital example from the start? When one class is much rarer than the other, accuracy can be misleading. A model that always says "no disease" gets high accuracy by ignoring the very thing we care about.

---

## Precision — "When You Said Yes, Were You Right?"

**Precision** answers this question: out of all the emails the model labeled as spam, how many were actually spam?

Think of a cautious detective. This detective only makes an arrest when they are very sure. They rarely arrest innocent people (few false alarms), but sometimes criminals escape because the detective was too careful (missed catches).

High precision means: when the model says something is positive, you can trust it.

Precision matters most when **false alarms are expensive**. If your spam filter accidentally blocks an important email from your boss, that is a real problem.

---

## Recall — "Did You Find Them All?"

**Recall** answers the opposite question: out of all the emails that actually ARE spam, how many did the model catch?

Think of an eager detective. This detective arrests everyone who looks even slightly suspicious. They catch all the criminals (no missed catches), but they also arrest a lot of innocent people (many false alarms).

High recall means: you are not missing any real positives.

Recall matters most when **missed catches are dangerous**. If a disease screening test misses a sick patient, that patient does not get treatment.

---

## The Precision-Recall Trade-Off

Here is the key insight: precision and recall pull in opposite directions.

- If you want to catch ALL the spam (high recall), you have to be more aggressive. But that means more false alarms (lower precision).
- If you want to avoid ALL false alarms (high precision), you have to be more cautious. But that means more spam slips through (lower recall).

You cannot maximize both at the same time. Every real system must decide which mistakes are more acceptable.

```
HIGH PRECISION, LOW RECALL:
"I only mark emails as spam when I'm REALLY sure"
  Result: Few false alarms, but lots of spam gets through

LOW PRECISION, HIGH RECALL:
"I mark EVERYTHING suspicious as spam"
  Result: Catches all spam, but also blocks real emails!
```

---

## F1 Score — The Balanced Grade

Sometimes you need a single number that balances precision and recall. That number is the **F1 score**.

Think of it like an overall grade that penalizes you if either precision OR recall is low. You cannot cheat the F1 score by being great at one and terrible at the other — both need to be decent for the F1 score to be high.

The F1 score is most useful when you care about both types of mistakes roughly equally and need one number to compare models.

---

## When to Use Which Metric

| Situation | Best metric | Why |
|-----------|------------|-----|
| Balanced classes, simple task | Accuracy | Fair representation when classes are equal |
| False alarms are costly (spam filter, content moderation) | Precision | You want to trust the "yes" predictions |
| Missing positives is dangerous (disease detection, fraud) | Recall | You cannot afford to miss real cases |
| Both mistakes matter equally | F1 Score | Balances precision and recall |
| Classes are very imbalanced (rare disease, fraud) | NOT accuracy | Accuracy will lie to you — use precision, recall, or F1 |

---

**Quick check — can you answer these?**
- Why can a model with 99% accuracy still be useless?
- What is the difference between a false positive and a false negative?
- When would you care more about recall than precision?

If any of these feel unclear, go back and re-read that section. That is completely normal.

---

## You Just Learned the Language of Model Evaluation

Every ML paper, every product launch, every interview question about model quality uses these metrics. When someone says "our model achieves 0.95 F1 on the benchmark," you now know exactly what that means — and more importantly, what questions to ask next. "What about the precision-recall breakdown? Is the dataset balanced? Which mistakes are they hiding?"

These metrics are used everywhere — from Google's spam filter to hospital diagnostic systems to self-driving cars. You just picked up the vocabulary that every ML engineer uses daily.

---

Ready to go deeper? The [interview deep-dive](./classification-metrics-interview.md) covers the full math (micro/macro/weighted averaging, ROC curves, AUC, calibration), failure modes, and staff-level interview questions with worked examples.
