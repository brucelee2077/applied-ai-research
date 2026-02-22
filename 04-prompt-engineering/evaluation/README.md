# Prompt Evaluation

## Why Evaluate Your Prompts?

Imagine you're baking cookies. You taste one, tweak the recipe, bake again, taste again.
You don't just guess -- you **test** whether the cookies are actually good. Prompt
evaluation works the same way: you systematically check whether your prompts are giving
you good results, and you use that information to improve them.

Without evaluation, you're just guessing. With evaluation, you **know**.

```
┌──────────────────────────────────────────────────────────────────┐
│                   Why Evaluation Matters                          │
│                                                                  │
│   Without evaluation:                                            │
│   Prompt v1 ──> "Seems okay?" ──> Prompt v2 ──> "Maybe better?" │
│                  (guessing)                      (still guessing)│
│                                                                  │
│   With evaluation:                                               │
│   Prompt v1 ──> Score: 6/10 ──> Prompt v2 ──> Score: 8/10       │
│                 (measured)                     (measured)         │
│                 "Accuracy is low"              "Much better!"    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Simple Evaluation Methods

You don't need fancy tools to start evaluating. Here are methods from simplest
to most advanced:

### Method 1: The Eyeball Test (Manual Review)

The simplest approach: **read the AI's output and judge it yourself.**

Create a checklist for what "good" looks like:

```
┌──────────────────────────────────────────────────────────┐
│              Manual Evaluation Checklist                   │
│                                                          │
│   For each AI response, ask:                             │
│                                                          │
│   □ Is the answer factually correct?                     │
│   □ Does it follow the requested format?                 │
│   □ Is it the right length? (not too short, not too long)│
│   □ Is it clear and easy to understand?                  │
│   □ Would I be comfortable sharing this with someone?    │
│                                                          │
│   Score: Count the checkmarks out of 5                   │
└──────────────────────────────────────────────────────────┘
```

**When to use:** Early prototyping, small number of prompts, subjective quality tasks.

### Method 2: Test Cases (Like a Quiz)

Create a set of **test inputs with known correct answers**, then see how many the
AI gets right. Think of it like giving the AI a quiz.

```
Test Cases for a "Sentiment Classifier" prompt:

   Input                                    Expected    AI Got    Correct?
   ─────────────────────────────────────────────────────────────────────
   "I love this product!"                   POSITIVE    POSITIVE    ✓
   "Terrible experience, never again"       NEGATIVE    NEGATIVE    ✓
   "It's okay I guess"                      NEUTRAL     POSITIVE    ✗
   "Best purchase I've ever made!"          POSITIVE    POSITIVE    ✓
   "Waste of money"                         NEGATIVE    NEGATIVE    ✓

   Accuracy: 4/5 = 80%
```

**When to use:** Tasks with clear right/wrong answers (classification, extraction,
math, factual Q&A).

### Method 3: Side-by-Side Comparison (A/B Testing)

Run two different prompts on the **same inputs** and compare which one gives
better results. This is called **A/B testing**.

```
┌──────────────────────────────────────────────────────────────────┐
│                      A/B Testing                                 │
│                                                                  │
│   Same question, two different prompts:                          │
│                                                                  │
│   PROMPT A (simple):                                             │
│   "Summarize this article."                                      │
│   Result: "The article discusses climate change and its effects  │
│            on polar bears..." (Score: 6/10)                      │
│                                                                  │
│   PROMPT B (detailed):                                           │
│   "Summarize this article in 3 bullet points. Each point should  │
│    be one sentence. Focus on actionable takeaways."              │
│   Result: "• Rising temperatures threaten polar bear habitats    │
│            • Ice coverage has decreased 13% per decade           │
│            • Conservation efforts need increased funding"        │
│            (Score: 9/10)                                         │
│                                                                  │
│   Winner: Prompt B                                               │
└──────────────────────────────────────────────────────────────────┘
```

**Important:** Always test on **multiple inputs** (at least 10-20), not just one.
A prompt might work great on one example but fail on others.

---

## Evaluation Metrics (Ways to Score)

Metrics are just fancy words for "ways to measure how good the answer is." Here are
the most common ones, explained simply:

### Accuracy

**What it measures:** How often the AI gets the right answer.

```
Formula:    Accuracy = (number of correct answers) / (total questions)

Example:    AI got 17 out of 20 right
            Accuracy = 17/20 = 85%
```

**Use for:** Classification, factual Q&A, math problems -- anything with a single
correct answer.

### Relevance

**What it measures:** How well the AI's answer actually addresses what you asked.

```
Scoring guide (1-5 scale):

  5 = Perfectly relevant, directly answers the question
  4 = Mostly relevant, minor tangents
  3 = Somewhat relevant, misses some key points
  2 = Mostly off-topic
  1 = Completely irrelevant
```

**Use for:** Open-ended tasks like summarization, explanation, creative writing.

### Consistency

**What it measures:** Does the AI give similar answers when you ask the same thing
multiple times?

```
Ask the same question 5 times:

  Run 1: "The capital of France is Paris."       ──┐
  Run 2: "Paris is the capital of France."       ──┤ All say Paris
  Run 3: "France's capital city is Paris."       ──┤ = Consistent!
  Run 4: "The capital is Paris."                 ──┤
  Run 5: "Paris."                                ──┘

  Consistency = 5/5 = 100%
```

**Use for:** Any task where you need reliable, predictable outputs.

### Format Compliance

**What it measures:** Did the AI follow your formatting instructions?

```
Instruction: "Respond as JSON with keys 'name' and 'age'"

  Response 1: {"name": "Alice", "age": 30}        ✓ Valid JSON, right keys
  Response 2: name: Alice, age: 30                 ✗ Not JSON
  Response 3: {"Name": "Alice", "Age": 30}         ✗ Wrong key casing

  Format compliance: 1/3 = 33%
```

**Use for:** When the output feeds into other software (APIs, databases, reports).

---

## How to Run an Evaluation

Here's a simple step-by-step process:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Evaluation Process                               │
│                                                                  │
│   Step 1: Create test cases                                      │
│   ┌─────────────────────────────────────────────────┐            │
│   │ Write 10-20 inputs with expected outputs        │            │
│   │ Cover easy, medium, and tricky cases            │            │
│   └──────────────────────┬──────────────────────────┘            │
│                          │                                       │
│   Step 2: Run your prompt on all test cases                      │
│   ┌─────────────────────────────────────────────────┐            │
│   │ Feed each test input into your prompt           │            │
│   │ Save the AI's response                          │            │
│   └──────────────────────┬──────────────────────────┘            │
│                          │                                       │
│   Step 3: Score the results                                      │
│   ┌─────────────────────────────────────────────────┐            │
│   │ Compare AI outputs to expected outputs          │            │
│   │ Calculate accuracy, relevance, etc.             │            │
│   └──────────────────────┬──────────────────────────┘            │
│                          │                                       │
│   Step 4: Improve and repeat                                     │
│   ┌─────────────────────────────────────────────────┐            │
│   │ Look at the failures -- what went wrong?        │            │
│   │ Adjust the prompt and run again                 │            │
│   └─────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

### Example: Evaluating a Summary Prompt

```
Step 1: Test cases
  Article 1 (short, simple)    → Expected: 2-3 clear bullet points
  Article 2 (long, technical)  → Expected: 2-3 clear bullet points
  Article 3 (opinion piece)    → Expected: 2-3 clear bullet points

Step 2: Run prompt "Summarize in 3 bullet points"
  Article 1 → ✓ Got 3 bullets, clear and accurate
  Article 2 → ✗ Got 5 bullets (didn't follow instructions)
  Article 3 → ✗ Got 3 bullets but included the author's opinion as fact

Step 3: Score
  Format compliance: 1/3 (only one followed the 3-bullet format)
  Accuracy: 2/3 (two were factually correct)

Step 4: Improve
  New prompt: "Summarize in EXACTLY 3 bullet points. Only include facts,
  not opinions. Each bullet should be one sentence."
  Re-run → Much better results!
```

---

## Tips for Better Evaluation

| Tip | Why |
|-----|-----|
| Use at least 10 test cases | Small samples can be misleading -- one good result doesn't mean the prompt is reliable |
| Include edge cases | Test with tricky inputs: very short text, very long text, ambiguous cases, unusual formats |
| Test one change at a time | If you change 3 things and results improve, you won't know which change helped |
| Save your test set | Keep using the same test cases so you can track improvement over time |
| Have someone else review | You might be biased -- a fresh pair of eyes catches issues you'd miss |
| Track your scores | Keep a simple spreadsheet: prompt version, date, accuracy score, notes |

---

## A Simple Evaluation Spreadsheet

You don't need fancy tools. A simple spreadsheet works great:

```
┌──────────┬───────────┬──────────┬─────────┬───────────┬─────────────────┐
│ Prompt   │ Date      │ Accuracy │ Format  │ Avg Score │ Notes           │
│ Version  │           │          │ Match   │ (1-5)     │                 │
├──────────┼───────────┼──────────┼─────────┼───────────┼─────────────────┤
│ v1       │ Jan 5     │ 60%      │ 70%     │ 3.2       │ Baseline        │
│ v2       │ Jan 6     │ 75%      │ 90%     │ 3.8       │ Added format    │
│ v3       │ Jan 7     │ 85%      │ 95%     │ 4.3       │ Added examples  │
│ v4       │ Jan 8     │ 90%      │ 95%     │ 4.5       │ Added CoT       │
└──────────┴───────────┴──────────┴─────────┴───────────┴─────────────────┘
```

---

## Automated Evaluation (Advanced)

Once you're comfortable with manual evaluation, you can automate it with code.
Here's the basic idea:

```
┌──────────────────────────────────────────────────────────────────┐
│                  Automated Evaluation Pipeline                   │
│                                                                  │
│   Test cases          Run prompt on         Score each            │
│   (input +    ──>     each input    ──>     response     ──>  Report│
│    expected)          via API               automatically        │
│                                                                  │
│   You write           Code does this        Code compares        │
│   these once          automatically         output to expected   │
└──────────────────────────────────────────────────────────────────┘
```

For example, in Python:

```python
test_cases = [
    {"input": "I love it!",    "expected": "POSITIVE"},
    {"input": "Terrible!",     "expected": "NEGATIVE"},
    {"input": "It's okay.",    "expected": "NEUTRAL"},
]

correct = 0
for test in test_cases:
    response = call_llm(my_prompt, test["input"])
    if response.strip() == test["expected"]:
        correct += 1

accuracy = correct / len(test_cases)
print(f"Accuracy: {accuracy:.0%}")    # e.g., "Accuracy: 67%"
```

You can also use **LLM-as-a-judge**: ask a separate AI to evaluate the first AI's
response. This works well for open-ended tasks where there's no single correct answer.

---

## Key Takeaways

1. **Always evaluate** -- guessing whether your prompt works is not enough
2. **Start simple** -- manual review with a checklist is perfectly fine to begin with
3. **Use test cases** -- create inputs with known answers and track accuracy
4. **A/B test** -- compare prompt variations on the same inputs to find the best one
5. **Track your progress** -- even a simple spreadsheet showing scores over time helps
6. **Iterate** -- look at failures, improve the prompt, re-evaluate. Repeat.

---

[Back to Prompt Engineering](../README.md)
