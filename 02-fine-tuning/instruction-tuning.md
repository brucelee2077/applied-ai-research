# Instruction Tuning, RLHF, and DPO

## From Word Predictor to Helpful Assistant

Here is something strange. GPT-3 read most of the internet. It knows facts about history, science, law, and medicine. It can write grammatically correct sentences in dozens of languages. But if you ask it "What is the capital of France?", it might respond with: "What is the capital of Germany? What is the capital of Italy? These are common geography questions..."

It does not answer. It just keeps writing, as if you are in the middle of a document and it is adding the next paragraph. It has all the knowledge in the world, but it does not know how to be *helpful*.

How did we get from that to ChatGPT — a model that answers questions clearly, refuses dangerous requests, and admits when it does not know something?

The answer is **instruction tuning** — and its two powerful extensions, **RLHF** and **DPO**.

---

**Before you start, you need to know:**
- What fine-tuning is — covered in [01_what_is_fine_tuning.ipynb](./01_what_is_fine_tuning.ipynb)
- What LoRA is — covered in [lora.md](./lora.md)
- What "training data" means (examples the model learns from)

---

## The Analogy: A Brilliant Student Who Never Answers Questions

Imagine a student who has read every book in the school library. They know everything — history, math, science, literature. But they have never been in a class. They have never been asked a question and had a teacher say "good answer" or "try again."

If you ask them "What causes rain?", they might start reciting a paragraph from a textbook about cloud formation, weather patterns, atmospheric pressure, and then drift into ocean currents. They are not *wrong*, but they are not *answering your question*.

Now imagine three training steps:

**Step 1 — Supervised Fine-Tuning (SFT).** A teacher gives the student 10,000 flash cards. Each card has a question on the front and a perfect answer on the back. "What causes rain?" → "Water evaporates, rises into the sky, cools down, and falls back as rain." The student practices until they can produce answers that match the cards.

**Step 2 — Reward Model.** The teacher shows the student two answers to the same question and says "this one is better." They do this thousands of times. The student starts to understand *what makes a good answer* — it should be clear, honest, and safe.

**Step 3 — Practice with Feedback (RLHF).** The student writes their own answers. Each time, the teacher gives a score. High scores for helpful answers, low scores for unhelpful or harmful ones. The student gets better and better.

**What this analogy gets right:**
- The student (model) already has the knowledge from reading (pre-training). Instruction tuning does not add new knowledge — it teaches the model how to *use* its knowledge to help people.
- Each step builds on the last. SFT comes first, then RLHF refines it further.
- The quality improves at every step — from useless text completer, to instruction follower, to genuinely helpful assistant.

**Where the analogy breaks down:** A real student understands *why* an answer is good. The model does not. It learns statistical patterns about what kind of text gets high rewards. This can lead to problems like reward hacking — the model finds ways to get high scores without being truly helpful.

---

## The Three Steps in Plain Words

### Step 1: Supervised Fine-Tuning (SFT)

You take the pre-trained model and fine-tune it on thousands of examples that look like this:

```
Question: What is photosynthesis?
Answer: Photosynthesis is how plants turn sunlight, water,
and carbon dioxide into food (glucose) and oxygen.
```

The model learns: when you see a question, give a clear, direct answer. When you see "summarize this", write a short summary. When you see "translate to French", output French.

This is the same kind of fine-tuning we learned about earlier — just with instruction-response data instead of task-specific data.

### Step 2: RLHF (Reinforcement Learning from Human Feedback)

SFT makes the model follow instructions. But it does not always follow them *well*. It might give correct but confusing answers, or answer dangerous questions it should refuse.

RLHF fixes this by learning from human preferences. Humans look at two answers to the same question and pick the better one. A separate model (the **reward model**) learns to predict what humans prefer. Then the main model is trained to generate answers that the reward model rates highly.

### Step 3 (Alternative): DPO (Direct Preference Optimization)

RLHF works, but it is complex — it needs three models running at the same time. In 2023, researchers found a simpler way. DPO skips the reward model entirely. It directly trains the model on preference pairs: make the preferred answer more likely, make the rejected answer less likely. Same idea, much simpler to run.

```
  The ChatGPT Recipe:

  Pre-trained Model ──→ SFT ──→ RLHF ──→ ChatGPT
  (knows everything,    (follows     (follows them
   helps nobody)         instructions) WELL)
```

---

**Quick check — can you answer these?**
- Why can a pre-trained model know many facts but still not be helpful?
- What is the difference between SFT and RLHF?
- Why was DPO invented if RLHF already works?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## What You Just Learned

You now understand the three-step process that turned raw language models into helpful assistants like ChatGPT. SFT teaches the model to follow instructions. RLHF teaches it to follow them *well* — to be helpful, honest, and safe. DPO achieves similar results with a simpler approach.

This is one of the most important developments in AI. Before instruction tuning, large language models were impressive but useless for most people. After instruction tuning, they became the tools that millions of people use every day.

Ready to go deeper? Read the interview-prep version: [instruction-tuning-interview.md](./instruction-tuning-interview.md)
