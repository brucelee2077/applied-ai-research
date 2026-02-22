# BLEU & ROUGE

## The Big Idea

Imagine you asked two students to summarize the same book chapter. How would you
grade them? You'd probably compare their summaries to a "model answer" and see
how much they match.

**BLEU and ROUGE work the same way.** They compare text that an AI generated
to a "reference" (the correct or ideal answer) and measure how similar they are.

- **BLEU** = Used mostly for **translation** ("Did the AI translate this correctly?")
- **ROUGE** = Used mostly for **summarization** ("Did the AI capture the key points?")

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

## Understanding BLEU (Bilingual Evaluation Understudy)

Don't worry about the fancy name. BLEU just means: "How many words and phrases
in the AI's translation also appear in the correct translation?"

### The Simplest Version: Word Matching

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

BLEU doesn't just check individual words. It also checks **n-grams** -- groups
of consecutive words.

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

This is why n-grams are important -- they check word ORDER too.
```

### How BLEU Is Calculated (Simplified)

```
+------------------------------------------------------------------+
|                     BLEU Calculation Steps                        |
|                                                                  |
|  1. Count matching 1-grams  -->  Precision_1                    |
|  2. Count matching 2-grams  -->  Precision_2                    |
|  3. Count matching 3-grams  -->  Precision_3                    |
|  4. Count matching 4-grams  -->  Precision_4                    |
|  5. Average them together (using geometric mean)                 |
|  6. Apply "brevity penalty" (if AI's text is too short)          |
|                                                                  |
|  BLEU = Brevity_Penalty x exp(average of log precisions)         |
+------------------------------------------------------------------+
```

**Why the brevity penalty?** Without it, an AI could cheat:

```
Reference: "The beautiful cat sat gracefully on the soft mat"
AI output: "The"

Precision = 1/1 = 100% (the one word it said was correct!)
But that's obviously a terrible translation.

The brevity penalty reduces the score when the AI's output
is much shorter than the reference.
```

### BLEU Score Ranges

| BLEU Score | Quality | What It Means |
|-----------|---------|---------------|
| 0.0 | Terrible | Nothing matches at all |
| 0.1 - 0.2 | Poor | Some words match but mostly wrong |
| 0.2 - 0.3 | Okay | Understandable but has many errors |
| 0.3 - 0.4 | Good | Good translation with some issues |
| 0.4 - 0.5 | Very good | High quality translation |
| 0.5+ | Excellent | Near-human quality (rare!) |

---

## Understanding ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Another fancy name, but ROUGE is just BLEU's cousin with a different focus.

Where BLEU asks "How much of what the AI said is correct?" (precision),
ROUGE asks "How much of the important stuff did the AI include?" (recall).

This makes ROUGE perfect for **summarization**, where we want to make sure
the AI captured all the key points.

### ROUGE Variants

There are several types of ROUGE. Here are the main ones:

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
  ROUGE-1 Recall = 6/10 = 0.60

  AI output has 9 words, 6 match
  ROUGE-1 Precision = 6/9 = 0.67

  F1 = 2 x (0.60 x 0.67) / (0.60 + 0.67) = 0.63

ROUGE-2 (pair overlap):
  Reference pairs: "The president", "president signed", "signed the",
                   "the new", "new climate", "climate bill", etc.
  AI pairs:        "The president", "president signed", "signed a",
                   "a new", "new bill", etc.
  Matching pairs: "The president", "president signed"
  Fewer matches because word ORDER matters more with pairs.
```

### ROUGE-L: Longest Common Subsequence

ROUGE-L uses a clever trick called **Longest Common Subsequence (LCS)**.
Instead of checking exact consecutive matches, it finds the longest sequence
of words that appear in the same ORDER in both texts (they don't have to be
right next to each other).

```
Reference: "The cat sat on the mat"
AI output: "The cat was sitting on a mat"

LCS: "The cat" ... "on" ... "mat" = 4 words in order
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
| **Typical use** | Comparing translation models | Comparing summarization models |

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

## Limitations -- What BLEU and ROUGE Miss

Both BLEU and ROUGE have significant limitations:

```
+-------------------------------------------------------------------+
|         Why These Metrics Aren't Perfect                          |
|                                                                   |
|  1. WORD MATCHING ISN'T UNDERSTANDING                            |
|                                                                   |
|     Reference: "The dog is happy"                                 |
|     AI output: "The puppy is joyful"                              |
|     BLEU/ROUGE: Low score! (different words)                      |
|     Reality: This is a GREAT answer! Same meaning!                |
|                                                                   |
|  2. SAME WORDS DON'T MEAN SAME THING                            |
|                                                                   |
|     Reference: "The bank is by the river"                         |
|     AI output: "I went to the bank for money by the river"       |
|     BLEU/ROUGE: High score! (many matching words)                 |
|     Reality: "bank" means different things!                       |
|                                                                   |
|  3. ONLY ONE "RIGHT" ANSWER                                     |
|                                                                   |
|     There are many ways to say the same thing.                    |
|     These metrics penalize valid alternatives.                    |
+-------------------------------------------------------------------+
```

Despite these limitations, BLEU and ROUGE remain widely used because:
- They're fast and automatic (no humans needed)
- They're standardized (everyone uses the same formula)
- They correlate reasonably well with human judgment for their specific tasks
- They're useful for comparing models during development

For tasks where these limitations matter most, use [Human Evaluation](./human-evaluation.md)
alongside these metrics.

---

## Summary

```
+------------------------------------------------------------------+
|                  BLEU & ROUGE Cheat Sheet                        |
|                                                                  |
|  BLEU:                                                           |
|    What:     Measures translation quality via word matching       |
|    Focus:    Precision (is the output correct?)                   |
|    Range:    0 to 1 (higher = better, 0.4+ is very good)         |
|    Method:   Compares n-grams + brevity penalty                  |
|                                                                  |
|  ROUGE:                                                          |
|    What:     Measures summary quality via content overlap         |
|    Focus:    Recall (did it capture the key points?)              |
|    Range:    0 to 1 (higher = better)                             |
|    Variants: ROUGE-1 (words), ROUGE-2 (pairs), ROUGE-L (longest) |
|                                                                  |
|  Both:                                                           |
|    Limitation: Word matching != understanding meaning             |
|    Use with:   Human evaluation for best results                  |
+------------------------------------------------------------------+
```

---

## Further Reading

- **BLEU: a Method for Automatic Evaluation of Machine Translation** -- Papineni et al., 2002
  - The original paper that introduced BLEU
- **ROUGE: A Package for Automatic Evaluation of Summaries** -- Lin, 2004
  - The original paper that introduced ROUGE
- **BERTScore** -- Zhang et al., 2020
  - A newer metric that uses word meanings (embeddings) instead of exact word matching,
    addressing many BLEU/ROUGE limitations

---

[Back to Metrics](./README.md) | [Back to Evaluation](../README.md)
