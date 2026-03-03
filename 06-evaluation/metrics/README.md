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
|   Classifying (spam/not spam)      -->  Accuracy, F1, Precision   |
|   Predicting text / Language model  -->  Perplexity               |
|   Translating languages            -->  BLEU                      |
|   Summarizing text                 -->  ROUGE                     |
|   Creative writing / Chat          -->  Human Evaluation          |
|   All of the above / General       -->  Benchmarks (MMLU, etc.)   |
+-------------------------------------------------------------------+
```

---

## Coverage Map

| Topic | Depth | Files |
|-------|-------|-------|
| Classification Metrics — accuracy, precision, recall, F1, confusion matrix | [Core] | [classification-metrics.md](./classification-metrics.md) · [classification-metrics-interview.md](./classification-metrics-interview.md) · [01_classification_metrics.ipynb](./01_classification_metrics.ipynb) · [01_classification_metrics_experiments.ipynb](./01_classification_metrics_experiments.ipynb) |
| Perplexity — measuring language model quality | [Core] | [perplexity.md](./perplexity.md) · [perplexity-interview.md](./perplexity-interview.md) · [02_perplexity.ipynb](./02_perplexity.ipynb) · [02_perplexity_experiments.ipynb](./02_perplexity_experiments.ipynb) |
| BLEU & ROUGE — translation and summarization scoring | [Core] | [bleu-rouge.md](./bleu-rouge.md) · [bleu-rouge-interview.md](./bleu-rouge-interview.md) · [03_bleu_rouge.ipynb](./03_bleu_rouge.ipynb) · [03_bleu_rouge_experiments.ipynb](./03_bleu_rouge_experiments.ipynb) |
| Human Evaluation — annotator agreement, pairwise comparison, LLM-as-Judge | [Core] | [human-evaluation.md](./human-evaluation.md) · [human-evaluation-interview.md](./human-evaluation-interview.md) · [04_human_evaluation.ipynb](./04_human_evaluation.ipynb) · [04_human_evaluation_experiments.ipynb](./04_human_evaluation_experiments.ipynb) |

---

## Metrics Comparison Table

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

1. **[Classification Metrics](./classification-metrics.md)** -- Start here to understand
   accuracy, precision, recall, F1, and why 99% accuracy can be useless.

2. **[Perplexity](./perplexity.md)** -- Learn how language models are measured using a
   "guessing game" analogy.

3. **[BLEU & ROUGE](./bleu-rouge.md)** -- Learn how we score translations and summaries
   by comparing them to reference answers.

4. **[Human Evaluation](./human-evaluation.md)** -- When numbers aren't enough, learn
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
