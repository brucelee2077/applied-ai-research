# Benchmarks

GPT-4 scores 86% on a test. Claude scores 89%. Gemini scores 84%. But wait — how do we know they all took the same test? And what if the test does not cover what actually matters? Welcome to the world of AI benchmarks — where measuring intelligence is harder than building it.

---

**Before you start, you need to know:**
- What a language model does at a high level — it predicts text
- No math needed for this file
- No prior knowledge of specific models needed

---

## What's a Benchmark?

Imagine every school in the country gave students a completely different final exam.
How would you compare students across schools? You couldn't! That's why standardized
tests (like the SAT) exist -- everyone takes the SAME test, so you can compare scores
fairly.

**Benchmarks are standardized tests for AI models.** They're pre-made collections of
questions and tasks that every AI model can be tested on, so we can compare them
fairly.

```
+-------------------------------------------------------------------+
|                    Why Benchmarks Matter                           |
|                                                                   |
|   Without benchmarks:                                              |
|     "Our model is really good!"                                    |
|     "How good?"                                                    |
|     "Trust us, it's great!"                                        |
|     (No way to verify or compare)                                  |
|                                                                   |
|   With benchmarks:                                                 |
|     "Our model scores 89.5% on MMLU"                               |
|     "That's better than Model X (85.2%) but not                    |
|      as good as Model Y (92.1%)"                                   |
|     (Clear, fair, verifiable comparison)                            |
+-------------------------------------------------------------------+
```

---

## The Major Benchmarks

Here's a guide to the most important benchmarks you'll see in AI research.
Think of them as different "subjects" in school -- each one tests different
skills.

### GLUE (General Language Understanding Evaluation)

**The SAT of AI** -- tests basic language understanding.

```
+-------------------------------------------------------------------+
|   GLUE -- Released: 2018                                          |
|                                                                   |
|   What it tests: Can the model understand English?                |
|                                                                   |
|   Tasks include:                                                  |
|   - Sentiment analysis:  "Is this movie review positive           |
|                           or negative?"                            |
|   - Sentence similarity: "Do these two sentences mean             |
|                           the same thing?"                         |
|   - Textual entailment:  "If sentence A is true, must            |
|                           sentence B also be true?"                |
|                                                                   |
|   Number of tasks: 9                                              |
|   Score: Average across all tasks (0-100)                          |
|   Status: SOLVED -- modern models score above human level         |
+-------------------------------------------------------------------+
```

**Example GLUE question:**

```
Task: Sentence Similarity
Sentence A: "The cat is sleeping on the couch."
Sentence B: "A feline is napping on the sofa."
Question: Do these mean the same thing?
Answer: Yes

Task: Sentiment Analysis
Sentence: "This movie was a complete waste of time."
Question: Is this positive or negative?
Answer: Negative
```

### SuperGLUE

**The "harder SAT"** -- created because models got too good at GLUE.

```
+-------------------------------------------------------------------+
|   SuperGLUE -- Released: 2019                                     |
|                                                                   |
|   What it tests: Harder language understanding tasks              |
|                                                                   |
|   Why it exists: Models started scoring BETTER than humans        |
|   on GLUE, so researchers made a harder version!                  |
|                                                                   |
|   Harder tasks include:                                           |
|   - Reading comprehension with tricky questions                   |
|   - Causal reasoning: "Why did X happen?"                         |
|   - Word sense disambiguation: "Does 'bank' mean a               |
|     financial institution or a river bank?"                        |
|                                                                   |
|   Number of tasks: 8                                              |
|   Score: Average across all tasks (0-100)                          |
|   Status: Also essentially solved by modern LLMs                  |
+-------------------------------------------------------------------+
```

### MMLU (Massive Multitask Language Understanding)

**The "college entrance exam"** -- tests knowledge across 57 subjects.

```
+-------------------------------------------------------------------+
|   MMLU -- Released: 2021                                          |
|                                                                   |
|   What it tests: Does the model actually KNOW things?             |
|                                                                   |
|   57 subjects including:                                          |
|   - STEM: Math, physics, computer science, biology                |
|   - Humanities: History, philosophy, law                           |
|   - Social Sciences: Economics, psychology, politics               |
|   - Professional: Medicine, accounting, engineering                |
|                                                                   |
|   Format: Multiple choice (A, B, C, or D)                         |
|   Total questions: ~16,000                                         |
|   Score: % correct (0-100%)                                        |
|                                                                   |
|   This is one of the MOST cited benchmarks today.                 |
|   When someone says "Model X scores 90% on MMLU,"                 |
|   they mean it correctly answers 90% of questions across          |
|   all 57 subjects.                                                 |
+-------------------------------------------------------------------+
```

**Example MMLU questions:**

```
Subject: Astronomy
Q: What is the approximate age of the universe?
A) 6,000 years  B) 1 billion years  C) 13.8 billion years  D) 100 billion years
Answer: C

Subject: US History
Q: The Monroe Doctrine was primarily aimed at:
A) Asia  B) Europe  C) Africa  D) Australia
Answer: B

Subject: Computer Science
Q: What is the time complexity of binary search?
A) O(1)  B) O(log n)  C) O(n)  D) O(n^2)
Answer: B
```

### BIG-Bench (Beyond the Imitation Game)

**The "everything test"** -- a massive collaboration with 200+ tasks.

```
+-------------------------------------------------------------------+
|   BIG-Bench -- Released: 2022                                     |
|                                                                   |
|   What it tests: A HUGE variety of capabilities                   |
|                                                                   |
|   200+ tasks contributed by 450+ researchers, including:          |
|   - Logical reasoning and puzzles                                 |
|   - Code understanding                                            |
|   - Multilingual tasks                                            |
|   - Mathematical reasoning                                        |
|   - Common sense reasoning                                        |
|   - And many more creative challenges                             |
|                                                                   |
|   Fun task examples:                                               |
|   - "Finish the joke" (tests humor understanding)                  |
|   - "Navigate a text-based maze" (tests spatial reasoning)         |
|   - "Identify logical fallacies" (tests critical thinking)         |
|                                                                   |
|   BIG-Bench Hard (BBH): A subset of the hardest 23 tasks          |
|   that models still struggle with                                  |
+-------------------------------------------------------------------+
```

### HumanEval

**The "coding interview"** -- tests if the model can write code.

```
+-------------------------------------------------------------------+
|   HumanEval -- Released: 2021 (by OpenAI)                         |
|                                                                   |
|   What it tests: Can the model write working Python code?         |
|                                                                   |
|   How it works:                                                    |
|   - Give the model a function description                          |
|   - The model writes the code                                      |
|   - Run the code against test cases                                |
|   - Score: % of problems solved correctly                          |
|                                                                   |
|   Number of problems: 164                                          |
|   Metric: pass@k (% of problems solved in k attempts)             |
+-------------------------------------------------------------------+
```

**Example HumanEval problem:**

```python
def has_close_elements(numbers: list, threshold: float) -> bool:
    """Check if any two numbers in the list are closer
    to each other than the given threshold.

    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3)
    True
    """
    # Model writes the code here
```

### HellaSwag

**The "common sense test"** -- tests if the model understands how the world works.

```
+-------------------------------------------------------------------+
|   HellaSwag -- Released: 2019                                     |
|                                                                   |
|   What it tests: Common sense reasoning                           |
|                                                                   |
|   Format: Given a scenario, pick the most logical                 |
|   continuation from 4 choices                                      |
|                                                                   |
|   Example:                                                         |
|   "A woman is standing in front of a washing machine.              |
|    She opens the door and..."                                      |
|                                                                   |
|   A) "...puts clothes inside" (correct -- makes sense!)           |
|   B) "...starts dancing salsa"                                    |
|   C) "...pulls out a guitar"                                      |
|   D) "...begins painting the ceiling"                             |
|                                                                   |
|   Humans score ~95%. Early models scored ~40%.                    |
|   Modern LLMs now score 85%+                                      |
+-------------------------------------------------------------------+
```

### TruthfulQA

**The "honesty test"** -- tests if the model gives truthful answers.

```
+-------------------------------------------------------------------+
|   TruthfulQA -- Released: 2022                                    |
|                                                                   |
|   What it tests: Does the model give TRUTHFUL answers,            |
|   or does it repeat common misconceptions?                         |
|                                                                   |
|   Example:                                                         |
|   Q: "Can you see the Great Wall of China from space?"            |
|                                                                   |
|   Common (wrong) answer: "Yes, you can!"                          |
|   Truthful answer: "No, it's too narrow to see from              |
|   space with the naked eye."                                      |
|                                                                   |
|   This benchmark is important because LLMs tend to                |
|   repeat popular misconceptions they learned from                 |
|   training data, even when they're wrong.                         |
|                                                                   |
|   Number of questions: 817                                         |
|   Categories: Health, law, finance, politics, etc.                 |
+-------------------------------------------------------------------+
```

---

## Benchmark Comparison Table

| Benchmark | Year | What It Tests | # of Tasks/Questions | Status |
|-----------|------|--------------|---------------------|--------|
| **GLUE** | 2018 | Basic language understanding | 9 tasks | Solved |
| **SuperGLUE** | 2019 | Harder language understanding | 8 tasks | Mostly solved |
| **HellaSwag** | 2019 | Common sense reasoning | 10k questions | High scores |
| **MMLU** | 2021 | Knowledge across 57 subjects | ~16k questions | Active |
| **HumanEval** | 2021 | Code generation (Python) | 164 problems | Active |
| **TruthfulQA** | 2022 | Truthfulness vs misconceptions | 817 questions | Active |
| **BIG-Bench** | 2022 | 200+ diverse capabilities | 200+ tasks | Active |

---

## The Benchmark Treadmill

An important thing to understand: **benchmarks get "solved" over time.**

```
+-------------------------------------------------------------------+
|              The Benchmark Lifecycle                               |
|                                                                   |
|   Year 1: New benchmark released!                                 |
|           Humans score 90%, best AI scores 60%                    |
|           "AI has a long way to go"                                |
|                                                                   |
|   Year 2: New models score 80%                                    |
|           "Great progress!"                                        |
|                                                                   |
|   Year 3: Models score 95%, beating humans                        |
|           "Benchmark is solved"                                    |
|                                                                   |
|   Year 4: Researchers create a HARDER benchmark                   |
|           "Back to square one!"                                    |
|                                                                   |
|   This happened with GLUE --> SuperGLUE --> MMLU --> ...          |
+-------------------------------------------------------------------+
```

This is why the AI field constantly creates new benchmarks -- the old ones
become too easy as models improve.

---

## Limitations of Benchmarks

Benchmarks are useful but not perfect:

```
+-------------------------------------------------------------------+
|            What Benchmarks Can and Can't Tell You                 |
|                                                                   |
|  CAN tell you:                                                    |
|  - How models compare on specific tasks                           |
|  - Whether models are improving over time                         |
|  - Rough strengths and weaknesses                                 |
|                                                                   |
|  CAN'T tell you:                                                  |
|  - If the model is safe to deploy in production                   |
|  - If users will actually like the model's responses              |
|  - If the model works well on YOUR specific task                  |
|  - If the model is biased or unfair                               |
|                                                                   |
|  Important: A model scoring well on benchmarks might still        |
|  perform poorly on YOUR use case. Always test on your             |
|  own data too!                                                    |
+-------------------------------------------------------------------+
```

**Other concerns:**
- **Data contamination** -- If a model accidentally trained on benchmark questions,
  its score is inflated (like a student who got the answer key before the test)
- **Teaching to the test** -- Models can be optimized for benchmarks without
  being generally useful
- **Cultural bias** -- Many benchmarks are English-centric and reflect Western
  cultural knowledge

---

## Leaderboards

**Leaderboards** are public rankings of models by benchmark score. Key ones include:

- **Open LLM Leaderboard** (Hugging Face) -- Ranks open-source models across multiple
  benchmarks
- **Chatbot Arena** (lmsys.org) -- Ranks chatbots based on human preference votes
  (real users pick the better response in blind A/B tests)
- **HELM** (Stanford) -- Holistic Evaluation of Language Models across many dimensions

---

**Quick check — can you answer these?**
- What is the difference between GLUE and MMLU?
- Why do benchmarks get "solved" over time?
- Name two reasons why a high benchmark score does not guarantee a model will work well for your task.

If any of these feel unclear, go back and re-read that section. That is completely normal.

---

## Summary

```
+------------------------------------------------------------------+
|                    Benchmarks Cheat Sheet                         |
|                                                                  |
|  What:     Standardized tests for fair model comparison          |
|  Why:      "My model is better" needs proof -- benchmarks        |
|            provide that proof                                     |
|  Key ones: MMLU (knowledge), HumanEval (coding),                |
|            BIG-Bench (everything), TruthfulQA (honesty)          |
|                                                                  |
|  Remember:                                                       |
|    - Benchmarks get solved over time (treadmill effect)           |
|    - High scores don't guarantee real-world performance          |
|    - Always test on your own data too                            |
|    - Watch out for data contamination                            |
+------------------------------------------------------------------+
```

---

## You Now Speak the Language of AI Evaluation

Every time you read an AI paper, a product launch, or a model comparison blog post, you will see benchmark names: MMLU, HumanEval, BIG-Bench, HellaSwag. You now know what each one tests, why they exist, and — just as importantly — why they are not the whole story. You can read a leaderboard and know what the numbers mean and what they leave out.

---

## Further Reading

- **GLUE: A Multi-Task Benchmark and Analysis Platform** -- Wang et al., 2018
- **SuperGLUE: A Stickier Benchmark** -- Wang et al., 2019
- **Measuring Massive Multitask Language Understanding** -- Hendrycks et al., 2021
- **Beyond the Imitation Game (BIG-Bench)** -- Srivastava et al., 2022
- **Evaluating Large Language Models Trained on Code (HumanEval)** -- Chen et al., 2021
- **TruthfulQA: Measuring How Models Mimic Human Falsehoods** -- Lin et al., 2022

---

[Back to Evaluation](../README.md)
