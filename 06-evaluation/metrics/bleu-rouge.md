# BLEU & ROUGE

Here is a puzzle: you ask an AI to translate "The dog is happy" into French. It outputs "Le chien est joyeux." A human translator wrote "Le chien est content." Both are perfectly correct translations — but the AI used completely different words from the reference. How do you score that automatically?

This is the core challenge of measuring translation and summarization quality. BLEU and ROUGE are the two most widely used metrics for this problem. They are not perfect — that puzzle above is one of their known weaknesses — but they are fast, standardized, and good enough to have been the default metrics in NLP for over two decades.

---

**Before you start, you need to know:**
- What a "reference" answer is — the correct or ideal output written by a human
- What precision and recall mean — covered in [classification-metrics.md](./classification-metrics.md)
- No math needed for this file

---

## The Grading Analogy

Imagine you asked two students to summarize the same book chapter. How would you grade them? You would probably compare their summaries to a "model answer" and see how much they match.

**BLEU and ROUGE work the same way.** They compare text that an AI generated to a reference (the correct or ideal answer) and measure how similar they are.

- **BLEU** is used mostly for **translation** — "Did the AI translate this correctly?"
- **ROUGE** is used mostly for **summarization** — "Did the AI capture the key points?"

**What the analogy gets right:** both metrics really do compare the AI's output word-by-word against a reference answer, just like a teacher checking answers against an answer key.

**Where the analogy breaks down:** a real teacher understands meaning — they would give full credit for "happy" and "joyful" because they mean the same thing. BLEU and ROUGE only match exact words. Synonyms get zero credit.

```
+----------------------------------------------------------------+
|                   BLEU vs ROUGE at a Glance                    |
|                                                                |
|   BLEU asks:  "How much of the AI's output matches            |
|                the reference?"                                  |
|                (Focuses on PRECISION -- being correct)          |
|                                                                |
|   ROUGE asks: "How much of the reference is captured           |
|                in the AI's output?"                             |
|                (Focuses on RECALL -- being complete)            |
+----------------------------------------------------------------+
```

---

## Understanding BLEU

BLEU stands for Bilingual Evaluation Understudy. It answers: "How many words and phrases in the AI's translation also appear in the correct translation?"

### Word Matching: The Simplest Version

```
Reference (correct translation):
  "The cat is on the mat"

AI's translation:
  "The cat is on the mat"

Every word matches! BLEU score = very high (close to 1.0)
```

But what if the AI wrote something slightly different?

```
Reference: "The cat is on the mat"
AI output: "The cat sits on a mat"

Matching words: "The", "cat", "on", "mat" = 4 out of 6 words
Non-matching:   "sits" (vs "is"), "a" (vs "the")

BLEU score = lower (but still decent because most words match)
```

### N-grams: Not Just Single Words

BLEU does not just check individual words. It also checks **n-grams** — groups of consecutive words.

```
What are n-grams? Just groups of words next to each other:

Sentence: "The cat sat on the mat"

1-grams (single words):  "The", "cat", "sat", "on", "the", "mat"
2-grams (pairs):         "The cat", "cat sat", "sat on", "on the", "the mat"
3-grams (triples):       "The cat sat", "cat sat on", "sat on the", "on the mat"
4-grams (quads):         "The cat sat on", "cat sat on the", "sat on the mat"
```

**Why check groups of words?** Because word ORDER matters!

```
Reference: "The cat sat on the mat"
Bad AI:    "mat the on sat cat The"

Single word check: 6 out of 6 match! (100%)
But wait -- the sentence makes no sense!

2-gram check: "mat the" -- nope! "the on" -- nope!
              0 out of 5 match (0%)

N-grams catch word order problems that single words miss.
```

### How BLEU Works (Plain Language)

BLEU follows these steps:

1. Count how many 1-grams (single words) in the AI's output match the reference
2. Count how many 2-grams (word pairs) match
3. Count how many 3-grams and 4-grams match
4. Combine all four scores together
5. Apply a **brevity penalty** if the AI's output is much shorter than the reference

**Why the brevity penalty?** Without it, an AI could cheat:

```
Reference: "The beautiful cat sat gracefully on the soft mat"
AI output: "The"

Word precision = 1/1 = 100% (the one word it said was correct!)
But that's obviously a terrible translation.

The brevity penalty reduces the score when the AI's output
is much shorter than the reference.
```

### BLEU Score Ranges

| BLEU Score | Quality | What It Means |
|-----------|---------|---------------|
| 0.0 | Terrible | Nothing matches at all |
| 0.1 – 0.2 | Poor | Some words match but mostly wrong |
| 0.2 – 0.3 | Okay | Understandable but has many errors |
| 0.3 – 0.4 | Good | Good translation with some issues |
| 0.4 – 0.5 | Very good | High quality translation |
| 0.5+ | Excellent | Near-human quality (rare!) |

---

## Understanding ROUGE

ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. Where BLEU asks "How much of what the AI said is correct?" (precision), ROUGE asks "How much of the important stuff did the AI include?" (recall).

This makes ROUGE a natural fit for **summarization**, where we want to make sure the AI captured all the key points.

### ROUGE Variants

```
+-------------------------------------------------------------------+
|                     ROUGE Family                                  |
|                                                                   |
|  ROUGE-1:  Compare individual WORDS (1-grams)                    |
|            "Did the summary include the important words?"          |
|                                                                   |
|  ROUGE-2:  Compare WORD PAIRS (2-grams)                          |
|            "Did the summary preserve word combinations?"           |
|                                                                   |
|  ROUGE-L:  Find the LONGEST matching sequence                    |
|            "What's the longest stretch of text that matches?"      |
+-------------------------------------------------------------------+
```

### ROUGE Worked Example

```
Reference summary (written by a human):
  "The president signed the new climate bill into law today"

AI's summary:
  "The president signed a new bill on climate change"

ROUGE-1 (word overlap):
  Matching words: "The", "president", "signed", "new", "climate", "bill"
  Reference has 10 words, 6 match
  ROUGE-1 Recall = 6 out of 10 = 60%

ROUGE-2 (pair overlap):
  Reference pairs: "The president", "president signed", "signed the", ...
  AI pairs:        "The president", "president signed", "signed a", ...
  Matching pairs: "The president", "president signed"
  Fewer matches because word ORDER matters more with pairs.
```

### ROUGE-L: Longest Common Subsequence

ROUGE-L uses a clever trick. Instead of checking exact consecutive matches, it finds the longest sequence of words that appear in the same ORDER in both texts. The words do not have to be right next to each other.

```
Reference: "The cat sat on the mat"
AI output: "The cat was sitting on a mat"

Longest Common Subsequence: "The cat" ... "on" ... "mat" = 4 words in order
(skips "was sitting" and "a")

This is forgiving of minor wording differences while still
checking that the key content appears in the right order.
```

---

## BLEU vs ROUGE: When to Use Which?

| | BLEU | ROUGE |
|---|------|-------|
| **Focus** | Precision (is the output correct?) | Recall (is the output complete?) |
| **Best for** | Machine translation | Text summarization |
| **Question it answers** | "How much of what the AI wrote matches the reference?" | "How much of the reference did the AI capture?" |
| **Score range** | 0 to 1 (higher = better) | 0 to 1 (higher = better) |

```
+-------------------------------------------------------------------+
|              Think of it like a school test:                      |
|                                                                   |
|  BLEU = "Of the answers you wrote, how many were correct?"        |
|          (You can score high by only answering easy questions)     |
|                                                                   |
|  ROUGE = "Of all the questions on the test, how many did you      |
|           get right?"                                             |
|          (Rewards attempting and getting everything right)         |
+-------------------------------------------------------------------+
```

---

## What BLEU and ROUGE Miss

Both metrics have real limitations:

**1. Word matching is not understanding.** "The dog is happy" and "The puppy is joyful" mean the same thing, but BLEU and ROUGE give low scores because the words are different.

**2. Same words can mean different things.** "The bank is by the river" and "I went to the bank for money by the river" share many words, but "bank" means different things.

**3. Only one "right" answer.** There are many ways to say the same thing. These metrics penalize valid alternatives.

Despite these limitations, BLEU and ROUGE remain widely used because they are fast, automatic, standardized, and correlate reasonably well with human judgment for their specific tasks. For tasks where these limitations matter most, use [Human Evaluation](./human-evaluation.md) alongside these metrics.

---

**Quick check — can you answer these?**
- What is the difference between what BLEU measures and what ROUGE measures?
- Why does BLEU use n-grams instead of just single words?
- Give an example of a good translation that BLEU would score poorly.

If any of these feel unclear, go back and re-read that section. That is completely normal.

---

## You Just Learned How AI Translations and Summaries Are Graded

Every machine translation paper since 2002 reports BLEU scores. Every summarization paper reports ROUGE. When Google Translate improves, they measure it with BLEU. When a new summarization model is released, they compare it using ROUGE. You now know exactly what those numbers mean, why they are useful, and where they fall short. That knowledge puts you ahead of most people who just see the numbers without understanding what is behind them.

---

Ready to go deeper? The [interview deep-dive](./bleu-rouge-interview.md) covers the exact formulas (modified n-gram precision with clipping, brevity penalty derivation, ROUGE-L LCS algorithm), failure modes, BERTScore/METEOR/CIDEr alternatives, and staff-level interview questions.

---

[Back to Metrics](./README.md) | [Back to Evaluation](../README.md)
