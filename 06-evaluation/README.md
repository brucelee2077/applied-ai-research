# 6. Evaluation & Benchmarking

## What is Evaluation?

Imagine you bake a cake. How do you know if it's any good? You could taste it yourself,
ask your friends to rate it, or enter it in a baking contest where judges score it
against other cakes.

**Evaluation in AI works the same way.** When we build an AI model (like a chatbot or
a translator), we need ways to measure: _"Is this model actually doing a good job?"_

Without evaluation, we'd be flying blind -- we'd have no idea if our AI is getting
better, getting worse, or just making things up.

```
+-----------------------------------------------------------------------+
|                    Why Evaluation Matters                              |
|                                                                       |
|   Without evaluation:          With evaluation:                       |
|                                                                       |
|   "I think the model           "The model gets 92% of                 |
|    is pretty good?"    vs       answers correct, up from              |
|         /                       85% last week"                        |
|      (shrug)                         |                                |
|                                (confident!)                           |
+-----------------------------------------------------------------------+
```

---

## Wait, What Are We Evaluating?

Before we dive in, let's cover some terms you'll see everywhere:

**AI Model** -- A computer program that has learned patterns from data. Think of it
like a student who studied millions of examples and can now answer new questions.

**LLM (Large Language Model)** -- A specific type of AI model that works with text.
It has read billions of pages and can write, translate, summarize, and answer questions.

**Key terms explained simply:**

| Term | Simple Explanation |
|------|-------------------|
| **Metric** | A number that tells you how good (or bad) your model is. Like a grade on a test |
| **Benchmark** | A standardized test for AI models -- like the SATs, but for AI. Everyone takes the same test so you can compare scores fairly |
| **Ground truth** | The correct answer. When testing a model, you need to know what the RIGHT answer is so you can check if the model got it right |
| **Accuracy** | The simplest metric: what percentage of answers did the model get correct? (e.g., 90% = got 9 out of 10 right) |
| **Precision** | Out of all the things the model SAID were correct, how many actually were? (If it says "these 10 emails are spam," and 8 really are, precision = 80%) |
| **Recall** | Out of all the things that ARE correct, how many did the model find? (If there are 10 spam emails and the model only caught 6, recall = 60%) |
| **F1 Score** | A single number that balances precision and recall. Think of it as the "overall grade" when you care about both |
| **Loss** | How wrong the model is. Lower loss = better model. Like golf -- lower scores win |
| **Overfitting** | When a model memorizes the test answers instead of actually learning. It scores great on practice tests but fails on new ones |

---

## The Two Big Questions

Evaluation boils down to two questions:

### 1. Automatic Evaluation: "Can a computer grade it?"

Some tasks have clear right/wrong answers. A computer can check them automatically.

```
+-----------------------------------------------+
|        Automatic Evaluation Example            |
|                                                |
|  Question: "What is 2 + 2?"                   |
|  Model's answer: "4"                          |
|  Correct answer:  "4"                          |
|  Grade: CORRECT                                |
|                                                |
|  This is easy for a computer to check!         |
+-----------------------------------------------+
```

We use **metrics** (like BLEU, ROUGE, and Perplexity) to automatically score models.
These are covered in the [Metrics](./metrics/) section.

### 2. Human Evaluation: "Do we need a human to judge it?"

Some tasks don't have one right answer. Is a poem "good"? Is a summary "helpful"?
Only humans can judge these things well.

```
+-----------------------------------------------+
|         Human Evaluation Example               |
|                                                |
|  Task: "Write a bedtime story about a cat"    |
|                                                |
|  Model A: writes a boring, technically         |
|           correct story                        |
|  Model B: writes a creative, engaging story    |
|           with a small grammar mistake         |
|                                                |
|  A computer might score A higher (no errors)   |
|  A human would probably prefer B (more fun!)   |
+-----------------------------------------------+
```

This is why we need both automatic AND human evaluation. They each catch
different things.

---

## Precision, Recall, and F1 -- The Big Three

These three metrics show up everywhere in ML. Let's use a simple example to
understand them.

**Scenario:** You have a basket of 10 fruits. 4 are apples, 6 are oranges.
You ask your AI to find all the apples.

```
Actual fruits:   [A] [A] [A] [A] [O] [O] [O] [O] [O] [O]
                  4 apples         6 oranges

AI's picks:      [A] [A] [A] [O]
                  3 correct     1 wrong (it thought this orange was an apple!)
                  (but it missed 1 real apple)
```

Now let's calculate:

```
Precision = Correct picks / Total picks = 3 / 4 = 75%
            "Of everything I picked, how many were actually apples?"

Recall    = Correct picks / Total apples = 3 / 4 = 75%
            "Of all the real apples, how many did I find?"

F1 Score  = 2 x (Precision x Recall) / (Precision + Recall)
          = 2 x (0.75 x 0.75) / (0.75 + 0.75) = 75%
            "A balanced grade combining both"
```

**Why do we need both Precision and Recall?**

```
+----------------------------------------------------------------+
|  Imagine a spam filter for your email:                         |
|                                                                |
|  HIGH PRECISION, LOW RECALL:                                   |
|  "I only mark emails as spam when I'm REALLY sure"            |
|   Result: Few mistakes, but lots of spam gets through          |
|                                                                |
|  LOW PRECISION, HIGH RECALL:                                   |
|  "I mark EVERYTHING suspicious as spam"                        |
|   Result: Catches all spam, but also blocks real emails!       |
|                                                                |
|  F1 SCORE: Helps you find the sweet spot between them          |
+----------------------------------------------------------------+
```

---

## The Confusion Matrix -- Sounds Scary, Isn't!

A confusion matrix is just a 2x2 table that shows what your model got right
and wrong. Let's use our spam filter example:

```
                        What the model predicted
                    +--------------+--------------+
                    |  "It's Spam" | "Not Spam"   |
    +---------------+--------------+--------------+
    | Actually Spam |     TP       |     FN       |
  A |               | (True Pos.)  | (False Neg.) |
  c |               | "Correct!    | "Oops, missed|
  t |               |  Caught it!" |  this spam"  |
  u +---------------+--------------+--------------+
  a | Actually NOT  |     FP       |     TN       |
  l | Spam          | (False Pos.) | (True Neg.)  |
    |               | "Oops, this  | "Correct!    |
    |               |  was real"   |  It's real"  |
    +---------------+--------------+--------------+

TP = True Positive  -- Model said spam, and it WAS spam (correct!)
FP = False Positive -- Model said spam, but it WASN'T spam (mistake!)
FN = False Negative -- Model said not spam, but it WAS spam (mistake!)
TN = True Negative  -- Model said not spam, and it WASN'T spam (correct!)
```

From this table, you can calculate everything:
- **Precision** = TP / (TP + FP) -- "Of my spam predictions, how many were right?"
- **Recall** = TP / (TP + FN) -- "Of all real spam, how much did I catch?"
- **Accuracy** = (TP + TN) / Total -- "Overall, how many did I get right?"

---

## Study Plan

Here's the recommended order to learn the topics in this module. Start from the top
and work your way down -- each topic builds on the one before it.

```
    START HERE
        |
        v
+---------------------------+
|  1. Metrics Overview      |  Learn what metrics are and when
|     (metrics/README.md)   |  to use which one
+-----------+---------------+
            |
            v
+---------------------------+
|  2. Perplexity            |  The go-to metric for language models.
|     (metrics/)            |  "How surprised is the model?"
+-----------+---------------+
            |
            v
+---------------------------+
|  3. BLEU & ROUGE          |  Metrics for translation and
|     (metrics/)            |  summarization tasks
+-----------+---------------+
            |
            v
+---------------------------+
|  4. Human Evaluation      |  When numbers aren't enough --
|     (metrics/)            |  getting human judges involved
+-----------+---------------+
            |
            v
+---------------------------+
|  5. Benchmarks            |  The standardized "exams" used to
|     (benchmarks/)         |  compare AI models fairly
+-----------+---------------+
            |
            v
+---------------------------+
|  6. Experiments           |  Try it yourself! Evaluate
|     (experiments/)        |  models hands-on
+---------------------------+
```

**Prerequisites:** Basic understanding of what an AI/ML model is. If you've read
through the earlier modules (especially [00-Neural Networks](../00-neural-networks/)
and [01-Transformers](../01-transformers/)), you're all set. If not, the explanations
here are written to be self-contained.

---

## Directory Structure

```
06-evaluation/
+-- README.md                          # You are here
+-- metrics/                           # How to measure model quality
|   +-- README.md                      #   Overview & comparison of all metrics
|   +-- perplexity.md                  #   How "surprised" is the model?
|   +-- bleu-rouge.md                  #   Scoring translations & summaries
|   +-- human-evaluation.md            #   When you need human judges
+-- benchmarks/                        # Standardized tests for AI
|   +-- README.md                      #   Major benchmarks explained
+-- notebooks/                         # Hands-on Jupyter notebooks
|   +-- 01_classification_metrics.ipynb #   Accuracy, Precision, Recall, F1
|   +-- 02_perplexity.ipynb            #   Perplexity from scratch + GPT-2
|   +-- 03_bleu_rouge.ipynb            #   BLEU & ROUGE from scratch
|   +-- 04_benchmarks_demo.ipynb       #   Explore real benchmarks + MMLU
+-- experiments/                       # Your own experiments
    +-- (your experiments go here!)
```

---

## Key Concepts at a Glance

| Concept | One-Line Summary | When You'll Use It |
|---------|-----------------|-------------------|
| **Accuracy** | % of answers the model got right | Simple classification tasks |
| **Precision & Recall** | How exact vs. how complete the model is | When mistakes have different costs (spam, medical) |
| **F1 Score** | Balanced grade of precision + recall | When you need one number to compare models |
| **Perplexity** | How "surprised" the model is by text | Evaluating language models |
| **BLEU** | Does the translation match the reference? | Machine translation |
| **ROUGE** | Does the summary capture key content? | Text summarization |
| **Human Evaluation** | Real people judge the output quality | Creative tasks, chatbots, anything subjective |
| **Benchmarks** | Standardized exams for fair comparison | Comparing models against each other |

---

## Key Papers

If you want to go deeper, these are the landmark research papers in evaluation:

- **BLEU: a Method for Automatic Evaluation of Machine Translation** -- Papineni et al., 2002
  - The foundational paper that introduced BLEU, still the most widely used translation metric
- **ROUGE: A Package for Automatic Evaluation of Summaries** -- Lin, 2004
  - Introduced the ROUGE family of metrics for measuring summarization quality
- **GLUE: A Multi-Task Benchmark and Analysis Platform** -- Wang et al., 2018
  - Created a standardized benchmark for testing language understanding (like SATs for AI)
- **SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding** -- Wang et al., 2019
  - A harder version of GLUE, because models got too good at the original
- **Measuring Massive Multitask Language Understanding (MMLU)** -- Hendrycks et al., 2021
  - Tests AI across 57 subjects from math to law to medicine
- **Beyond the Imitation Game (BIG-Bench)** -- Srivastava et al., 2022
  - A massive collaborative benchmark with 200+ tasks to test model capabilities

---

[Back to Main](../README.md) | [Previous: Multimodal](../05-multimodal/README.md) | [Next: Deployment](../07-deployment/README.md)
