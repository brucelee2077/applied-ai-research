# Perplexity

## What is Perplexity?

Imagine you're playing a guessing game. Your friend is telling you a story one word at
a time, and after each word, you have to guess the NEXT word.

- "The cat sat on the ___" -- You'd probably guess **"mat"** or **"chair"**. Easy!
- "The cat sat on the quantum ___" -- Hmm... **"computer"**? That's unexpected.

**Perplexity measures how SURPRISED a language model is** when it reads text.

- **Low perplexity** = The model is NOT surprised. It predicted the words easily.
  This means the text is "normal" and the model understands the language well.
- **High perplexity** = The model is VERY surprised. It had a hard time predicting
  the words. Either the text is unusual, or the model isn't very good.

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

---

## Why Do We Care About Perplexity?

Perplexity is the **go-to metric for language models** (like GPT, Claude, LLaMA, etc.).
It tells us:

1. **Is this model good at understanding language?**
   A model with lower perplexity on normal text has learned language patterns better.

2. **Is this model improving?**
   If we train a model more and perplexity goes down, we're making progress.

3. **Which model is better?**
   Compare perplexity scores -- lower wins (like golf scores).

```
+-------------------------------------------+
|        Comparing Two Models               |
|                                           |
|   Model A: Perplexity = 45               |
|   Model B: Perplexity = 22               |
|                                           |
|   Model B is better! It's less           |
|   surprised by language, meaning          |
|   it understands it better.               |
+-------------------------------------------+
```

---

## The Intuition: A Guessing Game

Let's make this concrete with numbers. Imagine a really simple language where
there are only 10 possible next words at any point.

```
The model needs to guess the next word. There are 10 choices:

    [apple] [banana] [cat] [dog] [eat]
    [fish]  [good]   [hat] [ice] [jump]
```

**Scenario 1: Random guessing (bad model)**

The model has NO idea, so every word is equally likely. It's like rolling a
10-sided die. The model is choosing from 10 equally likely options.

Perplexity = 10 (it's confused among 10 choices)

**Scenario 2: Pretty good model**

The model has learned some patterns. For the sentence "The cat sat on the ___",
it narrows it down to maybe 3-4 likely options.

Perplexity = 3.5 (it's only confused among ~3-4 choices)

**Scenario 3: Excellent model**

The model is very confident. It's almost sure the next word is "mat."

Perplexity = 1.2 (barely confused at all!)

```
+------------------------------------------------------------------+
|              Perplexity as "Number of Choices"                    |
|                                                                  |
|   Perplexity = 1     Perfect! The model always knows             |
|                      the next word (impossible in practice)       |
|                                                                  |
|   Perplexity = 10    The model is choosing between               |
|                      ~10 equally likely words                     |
|                                                                  |
|   Perplexity = 100   The model is confused among                 |
|                      ~100 possible words                          |
|                                                                  |
|   Perplexity = 1000  The model is basically lost!                |
|                                                                  |
|   Think of it as: "On average, how many words is the             |
|   model choosing between at each step?"                          |
+------------------------------------------------------------------+
```

---

## The Math (Don't Panic!)

If you're curious about the formula, here it is. But the intuition above is
really all you need.

### Step 1: Probability

For each word in a sentence, the model assigns a **probability** -- how likely
it thinks that word is.

```
Sentence: "The cat sat"

The model's predictions:
  P("The")  = 0.10   (10% chance -- "The" is a common start)
  P("cat")  = 0.05   (5% chance -- after "The", many words are possible)
  P("sat")  = 0.20   (20% chance -- after "The cat", "sat" is pretty likely)
```

### Step 2: Log probability

We take the **logarithm** (log) of each probability. Why? Because probabilities
are tiny numbers that get even tinier when multiplied. Logs turn multiplication
into addition, which is easier for computers.

```
What's a logarithm? Think of it as the "reverse" of powers:

  log2(8) = 3    because 2^3 = 8
  log2(4) = 2    because 2^2 = 4
  log2(2) = 1    because 2^1 = 2

For perplexity, we use natural log (ln), but the idea is the same.
```

### Step 3: Average and exponentiate

```
                    1
Perplexity = exp( - --- x SUM of ln(P(each word)) )
                    N

Where:
  N = number of words in the text
  P(each word) = the probability the model assigned to that word
  ln = natural logarithm
  exp = e raised to the power of (the reverse of ln)
```

**In plain English:** We average how "surprised" the model is at each word,
then convert that average surprise back into a single number.

### Worked Example

```
Sentence: "The cat sat" (3 words)

Model probabilities:
  P("The")  = 0.10
  P("cat")  = 0.05
  P("sat")  = 0.20

Step 1: Take log of each probability
  ln(0.10) = -2.30
  ln(0.05) = -3.00
  ln(0.20) = -1.61

Step 2: Average them
  Average = (-2.30 + -3.00 + -1.61) / 3 = -6.91 / 3 = -2.30

Step 3: Negate and exponentiate
  Perplexity = exp(2.30) = 10.0

Interpretation: On average, the model was choosing between
about 10 equally likely words at each step.
```

---

## What's a "Good" Perplexity Score?

There's no universal "good" number -- it depends on the task and dataset.
But here are some rough guidelines:

| Perplexity Range | What It Means | Example |
|-----------------|---------------|---------|
| 1 - 10 | Excellent (or possibly overfitting!) | Model that memorized the data |
| 10 - 50 | Very good for modern LLMs | GPT-class models on clean text |
| 50 - 100 | Decent, room for improvement | Smaller or older models |
| 100 - 500 | Struggling | Underfitting, not enough training |
| 500+ | Basically guessing randomly | Untrained model |

**Important caveats:**
- You can only compare perplexity scores on the **same dataset**. A score of 20
  on one dataset isn't comparable to a score of 20 on a different dataset.
- A model might have low perplexity but still generate boring or repetitive text.
  Perplexity measures prediction, not creativity.

---

## Limitations -- What Perplexity Can't Tell You

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
|                                                                   |
|   Think of a student who can predict the next word in a           |
|   sentence perfectly, but can't actually have a conversation.     |
|   That's the gap perplexity can't measure.                        |
+-------------------------------------------------------------------+
```

1. **Same dataset only** -- You can't compare perplexity across different datasets
2. **Doesn't measure quality** -- Low perplexity doesn't mean the text is good,
   helpful, or correct
3. **Can be gamed** -- A model that just repeats common phrases will have low
   perplexity but won't be useful
4. **Vocabulary matters** -- Models with different vocabularies aren't directly
   comparable

---

## Real-World Example

Let's say you're building a chatbot and training two versions:

```
Dataset: 1000 customer service conversations

Model v1 (trained for 1 day):
  Perplexity = 85
  "The model is still learning. It's confused among ~85 possible
   words at each step."

Model v2 (trained for 1 week):
  Perplexity = 28
  "Much better! Now it's only confused among ~28 possible words.
   It's learned common customer service patterns."

Model v3 (trained for 2 weeks):
  Perplexity = 25
  "Slightly better, but diminishing returns. The model might
   be close to its best on this data."
```

This is how researchers track training progress -- perplexity should go
down over time as the model learns.

---

## Summary

```
+------------------------------------------------------------------+
|                     Perplexity Cheat Sheet                        |
|                                                                  |
|   What:     How "surprised" a model is by text                   |
|   Good:     Lower is better (like golf scores)                   |
|   Think of: "How many words is the model choosing between?"      |
|   Used for: Comparing language models, tracking training         |
|   Caution:  Only compare on same dataset, doesn't measure        |
|             quality or helpfulness                                |
+------------------------------------------------------------------+
```

---

## Further Reading

- **Language modeling evaluation** -- Most introductory NLP courses cover perplexity
  in their language modeling chapters
- **A Closer Look at Perplexity** -- Various blog posts break down the math further
- For practical usage, see evaluation libraries like Hugging Face's `evaluate`

---

[Back to Metrics](./README.md) | [Back to Evaluation](../README.md)
