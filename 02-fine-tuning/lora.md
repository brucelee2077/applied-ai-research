# LoRA (Low-Rank Adaptation)

## The Trick That Changed Everything

Here's a puzzle: a model with 7 billion parameters was fine-tuned for a new task. Researchers looked at how much each parameter changed. The answer shocked them.

Almost all the change could be captured by adjusting just **0.1% of the parameters**. The other 99.9% barely moved at all. It was as if you renovated an entire house but really only needed to add a small extension.

This discovery led to a question: if most of the change is small and structured, why update all 7 billion parameters? Why not just add a small, targeted piece and leave the original model untouched?

That question led to **LoRA** — and it changed how the entire industry does fine-tuning.

---

**Before you start, you need to know:**
- What fine-tuning is — covered in [01_what_is_fine_tuning.ipynb](./01_what_is_fine_tuning.ipynb)
- Why full fine-tuning is expensive — covered in [full-fine-tuning.md](./full-fine-tuning.md)
- What a matrix is (a grid of numbers, like a spreadsheet)

---

## The Analogy: Sticky Notes on a Textbook

Imagine you have a thick textbook — hundreds of pages of general knowledge. Now you need to study for a specific exam. You have two options:

**Option A (Full fine-tuning):** Rewrite the entire textbook with your exam notes mixed in. Every page gets changed. It takes weeks, costs a lot, and if you need the original textbook back for a different exam, it's gone.

**Option B (LoRA):** Leave the textbook exactly as it is. Stick small sticky notes on the relevant pages. Each note adds a tiny bit of new information — a formula here, a definition there. The textbook stays intact. When you're done with the exam, peel off the notes and the original textbook is right there, unchanged.

**What this analogy gets right:**
- The textbook (pre-trained model) stays frozen. Nothing in it changes.
- The sticky notes (LoRA adapters) are small — much smaller than the textbook itself.
- You can have different sets of sticky notes for different exams (different tasks) and swap them in and out.
- The information from the textbook and the sticky notes is combined when you read a page.

```
  Full Fine-Tuning:              LoRA:
  ┌─────────────────┐           ┌─────────────────┐
  │                 │           │  ┌──┐           │
  │  Every page     │           │  │📝│ Sticky    │
  │  is rewritten   │           │  └──┘ note      │
  │                 │           │                 │
  │  (expensive,    │           │  Original page  │
  │   risky)        │           │  is untouched   │
  │                 │           │  (cheap, safe)  │
  └─────────────────┘           └─────────────────┘
     7B params changed             ~7M params added
```

**Where the analogy breaks down:** Real sticky notes just sit on top of the page. LoRA adapters are mathematically mixed into the model's computation — they modify the output of each layer in a precise, structured way. The "mixing" is not random; it follows a specific low-rank decomposition that captures the most important changes efficiently.

---

## LoRA in Plain Words

Here's the core idea in three sentences:

1. **Freeze the entire pre-trained model.** Do not change any of its parameters.
2. **Add a pair of small matrices to each layer.** These are the "adapters." They are much smaller than the original weight matrices.
3. **During training, only update the adapters.** The adapters learn the task-specific changes. The original model stays intact.

Why does this work? Because the change needed for most tasks lives in a **low-rank subspace**. That is a fancy way of saying: the change is simple and structured, not random and complex. You can capture it with a small number of parameters.

Think of it this way. A 7B model has weight matrices with thousands of rows and thousands of columns. When you fine-tune, those matrices change. But if you look at the *change* (the difference between the old matrix and the new matrix), it turns out you can describe that change with just two small matrices multiplied together. Instead of storing thousands × thousands of changes, you store thousands × a small number, and a small number × thousands.

That small number is called the **rank**, and it is usually between 4 and 64. This means LoRA typically adds less than 1% new parameters.

---

## Why LoRA Matters

| | Full Fine-Tuning | LoRA |
|---|---|---|
| Parameters changed | All (e.g., 7 billion) | Tiny fraction (e.g., 4–40 million) |
| GPU memory | ~112 GB for 7B model | ~14 GB for 7B model |
| Risk of forgetting | High — old knowledge can be lost | Very low — original model is frozen |
| Multiple tasks | Need a separate copy per task | Swap small adapter files (~10 MB) |
| Quality | Best possible | 95–99% as good |
| Training speed | Slow | Fast |

---

**Quick check — can you answer these?**
- What is the difference between full fine-tuning and LoRA in terms of what gets changed?
- Why is LoRA cheaper in terms of memory?
- What does "low-rank" mean in simple terms?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## What You Just Learned

You now understand the key insight behind LoRA: the change a model needs for a new task is small and structured. Instead of modifying billions of parameters, you add a small set of new parameters that capture the same change. This makes fine-tuning dramatically cheaper, faster, and safer.

LoRA was published in 2021 and quickly became the default method for fine-tuning large language models. When someone says they "fine-tuned LLaMA" on their laptop, they almost certainly used LoRA.

Ready to go deeper? Read the interview-prep version: [lora-interview.md](./lora-interview.md)
