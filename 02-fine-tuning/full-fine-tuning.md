# Full Fine-Tuning

## Why Would You Rewrite an Entire Brain?

GPT-4 was trained on trillions of words. It can write poetry, explain quantum physics, and translate between 100 languages. But ask it to classify your company's customer tickets into the exact 47 categories your support team uses, and it struggles.

The fix is simple in concept: take the model and train it again, this time on your data. But here is the surprising part — when you do this, you change *every single number* inside the model. All 7 billion of them (or 70 billion, or 175 billion). Every one gets adjusted a tiny bit.

Why would you change *everything* when you only need to teach it one new skill? And what happens to all the knowledge it already had?

---

**Before you start, you need to know:**
- What a neural network does at a high level — covered in [00-neural-networks](../../00-neural-networks/README.md)
- What fine-tuning means — covered in [01_what_is_fine_tuning.ipynb](./01_what_is_fine_tuning.ipynb)
- What "parameters" are (the numbers inside a model that control its behavior)

---

## The Analogy: Renovating a House

You bought a house. It is a nice, well-built house — good foundation, solid walls, working plumbing. But it was designed as a family home, and you need to turn it into a medical clinic.

**Full fine-tuning is like doing a complete renovation.** You change every room. The living room becomes an exam room. The bedroom becomes a lab. The kitchen becomes a break room. You touch every wall, every fixture, every piece of the house.

**What this analogy gets right:**
- The foundation (pre-trained knowledge) was already there. You did not build from scratch.
- Every part of the house gets modified, even parts that were already fine. This is what makes it expensive.
- The result is perfectly customized for your needs. No compromises.

```
  Before (Family Home)               After (Medical Clinic)
  ┌──────────┬──────────┐            ┌──────────┬──────────┐
  │  Living  │  Dining  │            │   Exam   │ Patient  │
  │   Room   │   Room   │            │   Room   │   Room   │
  ├──────────┼──────────┤   ───→     ├──────────┼──────────┤
  │   Bed    │   Bath   │            │   Lab    │  Office  │
  │   room   │   room   │            │          │          │
  └──────────┴──────────┘            └──────────┴──────────┘
  Every room changed. Expensive, but perfectly customized.
```

**Where the analogy breaks down:** When you renovate a house, the old rooms are gone but the house's purpose is clear — it is a clinic now. When you fully fine-tune a model, the old skills can get destroyed by accident. The model might become great at your task but forget how to do basic things it used to do well. This is called **catastrophic forgetting**, and it is a real risk.

---

## Full Fine-Tuning in Plain Words

A neural network is made of millions (or billions) of numbers called **parameters**. These parameters are what the model "knows." They were set during pre-training, when the model read huge amounts of text.

Full fine-tuning works like this:

1. **Start with a pre-trained model.** It already has good parameters from reading lots of text.
2. **Show it your data.** Give it examples of the task you care about — maybe 1,000 customer tickets with the correct category label.
3. **Let it adjust.** For each example, the model makes a prediction. If it is wrong, every parameter in the model gets nudged a tiny bit in the direction that would make the answer more correct.
4. **Repeat.** Do this thousands of times. After enough adjustments, the model gets good at your task.

The key word is *every*. In full fine-tuning, every parameter is allowed to change. A 7-billion-parameter model means 7 billion numbers are being adjusted on every training step.

This is powerful because the model can reshape itself completely for your task. But it is expensive because:
- You need enough GPU memory to hold the model, its gradients, and its optimizer state — roughly **4 times the model size**
- You risk catastrophic forgetting — the model might lose its general abilities while gaining your specific skill

---

## The Cost: Why Full Fine-Tuning Needs So Much Memory

When you train a model, the GPU does not just store the parameters. It also stores:
- **Gradients** — for each parameter, a number that says "which direction should this parameter move?"
- **Optimizer state** — the Adam optimizer keeps two extra numbers per parameter (a running average and a variance)
- **Activations** — intermediate results the model needs to remember for the backward pass

For a 7-billion-parameter model in 16-bit precision:
- Model weights: 14 GB
- Gradients: 14 GB
- Optimizer state (Adam): 56 GB
- Activations: depends on batch size, but typically 10–30 GB

Total: roughly **100+ GB** of GPU memory. That is more than a single A100 GPU (80 GB) can hold.

---

## Catastrophic Forgetting: The Big Risk

When you fine-tune a model on medical data, it gets great at medicine. But it might forget how to do math, lose its grammar skills, or stop understanding history. This happens because every parameter changed — including the parameters that stored those old skills.

You can reduce this risk:
- **Use a very small learning rate.** Make tiny changes instead of big ones.
- **Mix in some general data.** Include a few general-purpose examples alongside your specific data.
- **Stop early.** Do not train for too many steps. The model gets good at your task quickly — extra training just causes more forgetting.
- **Use LoRA instead.** LoRA freezes all the old parameters and adds small new ones. The old skills stay intact.

---

**Quick check — can you answer these?**
- Why is it called "full" fine-tuning? What makes it different from other fine-tuning methods?
- Why does full fine-tuning need roughly 4x the model size in GPU memory?
- What is catastrophic forgetting, and why does it happen during full fine-tuning?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## What You Just Learned

You now understand the simplest and most powerful form of fine-tuning: update every parameter. This is how most models were fine-tuned before 2021. It gives the best possible results — but at a high cost in memory, compute, and the risk of forgetting.

In the next notebook, you will see this in action with code. And after that, you will learn about **LoRA** — a method that gets 95% of the quality of full fine-tuning while changing less than 1% of the parameters.

Ready to go deeper? Read the interview-prep version: [full-fine-tuning-interview.md](./full-fine-tuning-interview.md)
