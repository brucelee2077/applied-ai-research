# Evaluation Metrics

## What's a Metric?

A **metric** is just a number that tells you how good (or bad) your AI model is doing.
Think of it like the score on a test -- it gives you a quick way to answer "Is this
model any good?"

Different tasks need different metrics, just like different sports have different
scoring rules. You wouldn't use basketball scores to judge a swimming race!

```
+-------------------------------------------------------------------+
|                   Choosing the Right Metric                       |
|                                                                   |
|   What is your model doing?                                       |
|                                                                   |
|   Predicting text / Language model  -->  Perplexity               |
|   Translating languages            -->  BLEU                      |
|   Summarizing text                 -->  ROUGE                     |
|   Classifying (spam/not spam)      -->  Accuracy, F1, Precision   |
|   Creative writing / Chat          -->  Human Evaluation          |
|   All of the above / General       -->  Benchmarks (MMLU, etc.)   |
+-------------------------------------------------------------------+
```

---

## Metrics Comparison Table

Here's a quick reference for all the metrics covered in this section:

| Metric | What It Measures | Score Range | Higher or Lower = Better? | Best For |
|--------|-----------------|-------------|--------------------------|----------|
| **Accuracy** | % of correct answers | 0% - 100% | Higher | Simple classification |
| **Precision** | How exact the model is | 0 - 1 | Higher | When false alarms are costly |
| **Recall** | How complete the model is | 0 - 1 | Higher | When missing things is costly |
| **F1 Score** | Balance of precision & recall | 0 - 1 | Higher | General classification |
| **[Perplexity](./perplexity.md)** | Model's "surprise" at text | 1 to infinity | **Lower** | Language model quality |
| **[BLEU](./bleu-rouge.md)** | Translation word overlap | 0 - 1 | Higher | Machine translation |
| **[ROUGE](./bleu-rouge.md)** | Summary content overlap | 0 - 1 | Higher | Text summarization |
| **[Human Eval](./human-evaluation.md)** | Human judgment scores | Varies | Higher | Subjective / creative tasks |

---

## How to Pick the Right Metric

```
                           START
                             |
                             v
                   What is your task?
                    /        |        \
                   v         v         v
            Classification  Generation  Language
            (Yes/No,        (Writing,   Modeling
             Spam/Ham,       Translation,(Predicting
             Cat/Dog)        Summary)    next words)
                |               |            |
                v               v            v
          Use Accuracy     Is there a    Use Perplexity
          Precision        "correct"
          Recall, F1       reference?
                          /          \
                         v            v
                       YES            NO
                        |              |
                        v              v
                  Translation?    Use Human
                  Use BLEU        Evaluation
                  Summary?
                  Use ROUGE
```

---

## Reading Order

If you're new to evaluation metrics, read these pages in order:

1. **[Perplexity](./perplexity.md)** -- Start here to understand how language models
   are measured. Uses a fun "guessing game" analogy.

2. **[BLEU & ROUGE](./bleu-rouge.md)** -- Learn how we score translations and summaries
   by comparing them to reference answers.

3. **[Human Evaluation](./human-evaluation.md)** -- When numbers aren't enough, learn
   how to set up human judges to rate AI quality.

---

## Key Takeaway

No single metric tells the whole story. The best evaluations use **multiple metrics
together**. Think of it like a report card -- you wouldn't judge a student by just
their math grade. You'd look at all their subjects, plus teacher comments (human
evaluation), to get the full picture.

```
+-------------------------------------------------------------------+
|                    A Good Evaluation Uses:                         |
|                                                                   |
|   1. Automatic metrics (fast, cheap, good for development)        |
|      +                                                            |
|   2. Human evaluation (slow, expensive, good for final checks)    |
|      =                                                            |
|   Complete picture of model quality                               |
+-------------------------------------------------------------------+
```

---

[Back to Evaluation](../README.md)
