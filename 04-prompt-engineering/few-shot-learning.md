# Few-Shot Learning

A team at Google gave GPT-3 a simple question: "Is this movie review positive or negative?" The AI got it right about 60% of the time. Then they added just three examples before the question — same AI, same question. Accuracy jumped to over 90%. Three examples. That is the power of few-shot learning.

---

**Before you start, you need to know:**
- What a prompt is (the text you type to tell the AI what to do) — covered in [README.md](./README.md)
- What zero-shot means (asking the AI with no examples) — covered in [README.md](./README.md)

---

## What is Few-Shot Learning?

**The analogy: teaching a friend a card game.** Imagine you want to teach a friend a new card game. You could hand them the rulebook and say "figure it out" — that is zero-shot. You could play one round and let them watch — that is one-shot. Or you could play a few rounds so they see the pattern — that is few-shot. Which way would your friend learn fastest? Probably the last one.

**What the analogy gets right:** the friend does not memorize the rules from the examples. They watch the pattern — what cards get played when, what counts as a win — and figure out the game by seeing it in action. That is exactly what the AI does. It looks at the examples in your prompt and picks up the pattern.

**The concept in plain words:** few-shot learning means giving the AI a few examples of what you want, right inside your prompt, so it figures out the pattern and follows it for your actual question.

**Where the analogy breaks down:** your friend remembers the rules after learning them. The AI does not remember anything between conversations. Every time you send a new prompt, you need to include the examples again.

```
┌──────────────────────────────────────────────────────────────────┐
│                  Zero-Shot vs Few-Shot (Visual)                  │
│                                                                  │
│   ZERO-SHOT:                                                     │
│   ┌─────────────────────────────────────┐                        │
│   │ "Translate 'hello' to French"       │──> AI guesses ──> "Bonjour"│
│   └─────────────────────────────────────┘                        │
│   No examples given. AI relies entirely on its training.         │
│                                                                  │
│   FEW-SHOT:                                                      │
│   ┌─────────────────────────────────────┐                        │
│   │ Example 1: "cat" -> "gato"          │                        │
│   │ Example 2: "dog" -> "perro"         │                        │
│   │ Example 3: "house" -> "casa"        │                        │
│   │                                     │                        │
│   │ Now translate: "car" -> ???         │──> AI sees ──> "coche" │
│   └─────────────────────────────────────┘    the pattern         │
│   AI learns the pattern from examples: English -> Spanish.       │
└──────────────────────────────────────────────────────────────────┘
```

---

## The Key Idea: In-Context Learning

Here is the wild part: the AI does not actually "retrain" or "study" when you give it examples. It just looks at the examples **right there in your prompt** and figures out the pattern on the fly. This is called **in-context learning** — the AI learns from the context (the text around your question).

```
┌──────────────────────────────────────────────────────────┐
│                   In-Context Learning                     │
│                                                          │
│   Traditional learning:   Study for weeks → Take test    │
│   (like school)                                          │
│                                                          │
│   In-context learning:    See pattern in prompt → Answer │
│   (like a puzzle)         "Oh, I see what you want!"     │
└──────────────────────────────────────────────────────────┘
```

The AI does not change or update its brain. It just uses the examples as clues to understand what you are asking for.

---

## Zero-Shot vs One-Shot vs Few-Shot

Let's see all three approaches side by side with a real task:
**classifying movie reviews as positive or negative.**

### Zero-Shot (No examples)

```
Prompt:
  "Is this movie review positive or negative?
   Review: 'The acting was terrible and the plot made no sense.'
   Answer:"

AI Response: "Negative"
```

This works for simple tasks because the AI already understands sentiment from its training. But what if you want a specific output format?

### One-Shot (1 example)

```
Prompt:
  "Classify movie reviews.

   Review: 'I loved every minute of it!'
   Classification: POSITIVE

   Review: 'The acting was terrible and the plot made no sense.'
   Classification:"

AI Response: "NEGATIVE"
```

Now the AI knows you want the word "POSITIVE" or "NEGATIVE" in all caps. One example showed the format.

### Few-Shot (2-5 examples)

```
Prompt:
  "Classify movie reviews.

   Review: 'I loved every minute of it!'
   Classification: POSITIVE

   Review: 'Worst movie I've ever seen.'
   Classification: NEGATIVE

   Review: 'It was okay, nothing special.'
   Classification: NEUTRAL

   Review: 'The acting was terrible and the plot made no sense.'
   Classification:"

AI Response: "NEGATIVE"
```

Now the AI also knows there is a NEUTRAL option. The extra examples taught it something the one-shot version could not.

---

**Quick check — can you answer these?**
- What is the difference between zero-shot, one-shot, and few-shot?
- Does the AI retrain when you give it examples? (Hint: no — it uses in-context learning)
- When would zero-shot be enough, and when would you need few-shot?

If you cannot answer one, re-read that section. That is completely normal.

---

## When Should You Use Each?

```
┌──────────────────────────────────────────────────────────────┐
│                    Decision Guide                             │
│                                                              │
│   Is your task simple and common?                            │
│   (translation, summarization, basic Q&A)                    │
│       │                                                      │
│       ├── YES ──> Try Zero-Shot first                        │
│       │           (it might just work!)                      │
│       │                                                      │
│       └── NO ──> Do you need a specific output format?       │
│                      │                                       │
│                      ├── YES ──> Use Few-Shot                │
│                      │           (show the format you want)  │
│                      │                                       │
│                      └── NO ──> Does the task have subtle    │
│                                 categories or edge cases?    │
│                                     │                        │
│                                     ├── YES ──> Use Few-Shot │
│                                     │    (show the tricky    │
│                                     │     cases)             │
│                                     │                        │
│                                     └── NO ──> Try One-Shot  │
└──────────────────────────────────────────────────────────────┘
```

---

## How to Pick Good Examples

Picking the right examples is like picking the right practice problems for a test. Here are the rules:

### Rule 1: Cover the Different Cases

If your task has multiple possible answers, show at least one example of each.

```
BAD (all examples are the same type):
  "happy" -> Positive
  "joyful" -> Positive
  "excited" -> Positive
  Now classify: "sad" -> ???          # AI might be biased toward "Positive"

GOOD (covers different types):
  "happy" -> Positive
  "sad" -> Negative
  "okay" -> Neutral
  Now classify: "furious" -> ???      # AI understands all the options
```

### Rule 2: Keep the Format Consistent

Every example should look exactly the same. Same structure, same punctuation, same labels.

```
BAD (inconsistent format):
  Input: "hello" -- Output is POSITIVE
  "goodbye" = negative
  the word "maybe" -> Neutral

GOOD (consistent format):
  Input: "hello" -> Sentiment: POSITIVE
  Input: "goodbye" -> Sentiment: NEGATIVE
  Input: "maybe" -> Sentiment: NEUTRAL
```

### Rule 3: Use Representative Examples

Pick examples that are similar to what you will actually ask about. Do not use overly simple examples if your real task is complex.

```
BAD (too simple for the real task):
  Examples: single words like "good", "bad", "okay"
  Real task: classify a 500-word essay

GOOD (similar complexity to real task):
  Examples: short paragraphs
  Real task: classify a 500-word essay
```

### Rule 4: Don't Use Too Many Examples

More examples = more tokens = more cost and slower responses. Usually **2-5 examples** is the sweet spot.

```
┌──────────────────────────────────────────────────┐
│             Number of Examples Guide              │
│                                                  │
│   1 example    ──  Good for format demos         │
│   2-3 examples ──  Good for most tasks           │
│   4-5 examples ──  Good for complex/subtle tasks │
│   6+ examples  ──  Usually not worth it          │
│                    (wastes tokens, rarely helps)  │
└──────────────────────────────────────────────────┘
```

---

## Real-World Examples

### Example 1: Data Extraction

Extract structured data from messy text:

```
Prompt:
  "Extract the name and age from each sentence.

   Sentence: 'My friend Sarah just turned 25 last week.'
   Name: Sarah
   Age: 25

   Sentence: 'Old Mr. Johnson, who is 82, lives next door.'
   Name: Mr. Johnson
   Age: 82

   Sentence: 'The 16-year-old student, Emma, won the science fair.'
   Name:
   Age:"

AI Response:
  "Name: Emma
   Age: 16"
```

### Example 2: Code Generation

Teach the AI your coding style:

```
Prompt:
  "Convert descriptions to Python functions.

   Description: Add two numbers
   Code:
   def add(a, b):
       return a + b

   Description: Check if a number is even
   Code:
   def is_even(n):
       return n % 2 == 0

   Description: Find the larger of two numbers
   Code:"

AI Response:
  "def find_larger(a, b):
       return a if a > b else b"
```

### Example 3: Creative Writing with Style

Teach a specific humor style:

```
Prompt:
  "Write a funny one-liner about the topic.

   Topic: Mondays
   Joke: Mondays are proof that the universe has a sense of humor, just not a very good one.

   Topic: Cooking
   Joke: I cook with wine. Sometimes I even put it in the food.

   Topic: Exercise
   Joke:"

AI Response:
  "I tried jogging once, but the ice kept falling out of my glass."
```

---

## Common Mistakes to Avoid

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| Inconsistent formatting | AI gets confused about the expected output | Use the exact same format for every example |
| All examples are the same category | AI becomes biased toward that category | Balance your examples across categories |
| Examples too different from real task | AI cannot transfer the pattern | Make examples similar to your real inputs |
| Too many examples | Wastes tokens, hits context limit, no accuracy gain | Stick to 2-5 examples |
| Ambiguous examples | AI learns the wrong pattern | Make each example clear and unambiguous |

---

## Key Takeaways

1. **Few-shot learning** = giving the AI examples in your prompt so it learns the pattern
2. **The AI does not retrain** — it just uses the examples as clues (in-context learning)
3. **2-5 examples** is usually enough
4. **Good examples** are consistent, diverse, and representative
5. **Try zero-shot first** — you might not even need examples for simple tasks

---

## You Just Unlocked a Superpower

You now know how to teach an AI a new task without any training, any data pipeline, or any code. Just a few well-chosen examples in your prompt. This technique — in-context learning — is one of the most surprising discoveries in modern AI. Researchers at OpenAI called it "one of the most remarkable capabilities" of large language models. And you can use it right now.

Ready to go further? Next, learn how to make the AI show its work → [Chain-of-Thought](./chain-of-thought.md)

---

## Further Reading

- **Language Models are Few-Shot Learners** — Brown et al., 2020
  - The foundational paper (GPT-3) that showed large language models can perform new tasks
    with just a few examples in the prompt, without any retraining
- **Rethinking the Role of Demonstrations** — Min et al., 2022
  - Surprising finding: the format of examples matters more than whether the examples are
    correct. The AI learns the pattern/structure, not the specific answers

---

[Back to Module Overview](./README.md) | [Next: Chain-of-Thought](./chain-of-thought.md)
