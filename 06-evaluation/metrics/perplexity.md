# Perplexity

You train a language model for a week. It cost real money. Now you ask: "Is this model any good at language?" But how do you measure something as vague as "understanding language"?

Here is a clever trick: you give the model a sentence and at each word, you ask: "What do you think comes next?" If the model is good, it will not be surprised. If the model is bad, every word will shock it. Perplexity turns that surprise into a single number — and that number is one of the most important metrics in all of language modeling.

---

**Before you start, you need to know:**
- What a language model does — it predicts the next word in a sentence
- What "probability" means — a number between 0 and 1 that says how likely something is
- No math needed for this file

---

## The Guessing Game Analogy

Imagine you are playing a guessing game. Your friend tells you a story one word at a time, and after each word, you have to guess the NEXT word.

- "The cat sat on the ___" — You would probably guess **"mat"** or **"chair"**. Easy!
- "The cat sat on the quantum ___" — Hmm... **"computer"**? That is unexpected.

A language model plays this game with every sentence it reads. **Perplexity measures how surprised the model is** by the words it sees.

**What the analogy gets right:** perplexity really is about surprise at each word. A model with low perplexity is like a friend who knows you so well they can finish your sentences.

**Where the analogy breaks down:** you guess one word at a time and either get it right or wrong. The model assigns probabilities to every possible word simultaneously — it is not picking a single guess, but spreading its confidence across all options.

---

## Low Perplexity = Good, High Perplexity = Bad

```
+-------------------------------------------------------------+
|                  Perplexity = Surprise Level                 |
|                                                              |
|   "The dog chased the cat"                                   |
|    Model thinks: "Yeah, I expected that"                     |
|    Perplexity: LOW (around 10-20)                            |
|                                                              |
|   "The refrigerator philosophized about jazz"                |
|    Model thinks: "Wait, WHAT?"                               |
|    Perplexity: HIGH (could be 100+)                          |
|                                                              |
|   Rule of thumb: LOWER perplexity = BETTER model             |
+-------------------------------------------------------------+
```

Think of perplexity as a golf score — lower is better.

---

## Perplexity as "Number of Choices"

Here is the most useful way to think about perplexity: **it tells you how many words the model is choosing between at each step.**

Imagine a simple language with only 10 possible next words at any point.

**A bad model (random guessing):** Every word is equally likely. The model is choosing from 10 equally likely options. Perplexity = 10.

**A decent model:** The model has learned some patterns. For "The cat sat on the ___", it narrows it down to maybe 3–4 likely options. Perplexity = 3.5.

**A great model:** The model is very confident. It is almost sure the next word is "mat." Perplexity = 1.2.

```
+------------------------------------------------------------------+
|   Perplexity = 1     Perfect! The model always knows              |
|                      the next word (impossible in practice)        |
|                                                                   |
|   Perplexity = 10    The model is choosing between                |
|                      ~10 equally likely words                      |
|                                                                   |
|   Perplexity = 100   The model is confused among                  |
|                      ~100 possible words                           |
|                                                                   |
|   Perplexity = 1000  The model is basically lost!                 |
+------------------------------------------------------------------+
```

---

## Why Do We Care About Perplexity?

Perplexity is the **go-to metric for language models** (like GPT, Claude, LLaMA, and others). It tells us three things:

1. **Is this model good at understanding language?** A model with lower perplexity on normal text has learned language patterns better.

2. **Is this model improving?** If you train a model more and perplexity goes down, you are making progress.

3. **Which model is better?** Compare perplexity scores — lower wins.

```
+-------------------------------------------+
|        Comparing Two Models               |
|                                           |
|   Model A: Perplexity = 45               |
|   Model B: Perplexity = 22               |
|                                           |
|   Model B is better! It's less            |
|   surprised by language, meaning          |
|   it understands it better.               |
+-------------------------------------------+
```

---

## How It Works (No Math Version)

The model reads a sentence word by word. At each word, it says: "How likely did I think this word was?"

If the model thought the word was very likely — not very surprised. If the model thought the word was unlikely — very surprised.

Perplexity takes the average surprise across all the words, then converts it into a single number that represents "how many choices the model was confused between."

That is all there is to it. The exact formula lives in the [interview deep-dive](./perplexity-interview.md) if you want the full derivation.

---

## What Is a "Good" Perplexity Score?

There is no universal "good" number — it depends on the task and dataset. But here are some rough guidelines:

| Perplexity Range | What It Means | Example |
|-----------------|---------------|---------|
| 1 – 10 | Excellent (or possibly overfitting!) | Model that memorized the data |
| 10 – 50 | Very good for modern LLMs | GPT-class models on clean text |
| 50 – 100 | Decent, room for improvement | Smaller or older models |
| 100 – 500 | Struggling | Underfitting, not enough training |
| 500+ | Basically guessing randomly | Untrained model |

---

## What Perplexity Cannot Tell You

```
+-------------------------------------------------------------------+
|           Perplexity is useful but not perfect                     |
|                                                                   |
|   CAN tell you:                CAN'T tell you:                    |
|   - Is the model learning?     - Is the output actually helpful?  |
|   - Which model predicts        - Is the text creative or         |
|     text better?                  interesting?                    |
|   - Is training working?       - Is the model factually correct?  |
|                                - Would a human prefer it?         |
+-------------------------------------------------------------------+
```

Four important limitations:

1. **Same dataset only** — You cannot compare perplexity across different datasets. A score of 20 on one dataset is not comparable to 20 on a different dataset.
2. **Does not measure quality** — Low perplexity does not mean the text is good, helpful, or correct.
3. **Can be gamed** — A model that just repeats common phrases will have low perplexity but will not be useful.
4. **Vocabulary matters** — Models with different vocabularies (different ways of splitting words into tokens) are not directly comparable.

---

## Real-World Example: Tracking Training Progress

```
Dataset: 1000 customer service conversations

Model v1 (early training):
  Perplexity = 85
  "The model is still learning. It's confused among ~85 possible
   words at each step."

Model v2 (more training):
  Perplexity = 28
  "Much better! Now it's only confused among ~28 possible words.
   It's learned common customer service patterns."

Model v3 (even more training):
  Perplexity = 25
  "Slightly better, but diminishing returns. The model might
   be close to its best on this data."
```

This is how researchers track training progress — perplexity should go down over time as the model learns.

---

**Quick check — can you answer these?**
- What does a perplexity of 50 mean in plain words?
- Why can you not compare perplexity scores across different datasets?
- A model has perplexity 15 on training data but perplexity 200 on new data. What is probably happening?

If any of these feel unclear, go back and re-read that section. That is completely normal.

---

## You Just Learned How Language Models Are Graded

Every time a new language model is released — GPT-4, LLaMA 3, Gemini — one of the first numbers reported is perplexity. You now know what that number means, why lower is better, and when it can be misleading. This is the same metric used by every major AI lab to track whether their models are learning language well.

---

Ready to go deeper? The [interview deep-dive](./perplexity-interview.md) covers the full derivation, cross-entropy connection, vocabulary dependence, BPE effects, and staff-level interview questions with worked examples.

---

[Back to Metrics](./README.md) | [Back to Evaluation](../README.md)
