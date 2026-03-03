# TRL Library Tutorial

You understand RLHF, reward models, PPO for language models, and DPO. You could build all of these from scratch. But should you? Building from scratch is great for learning. For production, you want tested, optimized tools. That is what TRL provides.

---

**Before you start, you need to know:**
- The full RLHF pipeline (SFT, reward modeling, PPO) — covered in previous files in this directory
- What DPO is and how it differs from RLHF — covered in `dpo-and-alternatives.md`

---

## The analogy: the professional toolkit

Imagine you are building a house. You could cut every board by hand, forge your own nails, and mix your own cement. You would learn a lot about construction. But it would take months, and every joint would be slightly off.

Or you could use professional tools: a power saw, factory-made nails, and premixed cement. The house gets built in days instead of months. The joints are straight. And you can focus on the design instead of struggling with basic construction.

**TRL (Transformer Reinforcement Learning) is the professional toolkit for LLM alignment.** It gives you four trainers — one for each step of the RLHF pipeline — so you can focus on your data and your goals instead of wrestling with implementation details.

### What the analogy gets right

- Professional tools (TRL) do not change what you are building — they change how quickly and reliably you build it
- You still need to understand the underlying concepts to use the tools well
- The tools handle engineering details (batching, logging, checkpointing) so you can focus on what matters (data quality, hyperparameters, evaluation)

### The concept in plain words

TRL is a library from Hugging Face that provides four main trainers. Each one handles one step of the alignment pipeline.

**SFTTrainer** handles supervised fine-tuning — Stage 1 of RLHF. You give it a base language model and instruction data (pairs of questions and good answers). It fine-tunes the model to follow instructions. This turns a text predictor into something that can answer questions. The data format is simple: either plain text with "Human:" and "Assistant:" markers, or a list of messages with roles.

**RewardTrainer** handles reward model training — Stage 2 of RLHF. You give it a transformer model and preference data (pairs of responses where a human marked one as better). It trains the model to predict which response humans prefer. The data format is: "chosen" (the preferred response) and "rejected" (the non-preferred one).

**PPOTrainer** handles the RL fine-tuning — Stage 3 of RLHF. This is the most complex trainer. It runs the full PPO loop: generate responses with the current model, score them with the reward model, compute advantages, and update the model. It handles the KL penalty, value head, and advantage estimation automatically. You just provide prompts and a reward function.

**DPOTrainer** handles direct preference optimization — the shortcut that skips Stages 2 and 3. You give it the SFT model and preference data. It trains directly on preferences using a supervised loss. No reward model needed, no PPO loop. The data format is: "prompt", "chosen", and "rejected".

### Where the analogy breaks down

Professional construction tools work the same way every time. TRL trainers have many hyperparameters that change their behavior. A power saw always cuts straight, but DPOTrainer with the wrong beta value will ruin your model. You need to understand the underlying algorithms to set the parameters correctly.

---

**Quick check — can you answer these?**
- What are the four main trainers in TRL?
- Which trainer do you need for RLHF? Which can you skip with DPO?
- What data format does each trainer expect?

If you cannot answer one, re-read that part. That is completely normal.

---

## The two pipelines

Most alignment projects follow one of two paths.

**Path 1: SFT then DPO** — the simple path. Train with SFTTrainer first. Then train with DPOTrainer. Two steps, two trainers, no reward model needed. This is the recommended starting point for most projects.

**Path 2: SFT, then Reward Model, then PPO** — the full RLHF path. Train with SFTTrainer, then RewardTrainer, then PPOTrainer. Three steps, three trainers, more complex but more flexible. Use this when you need online learning, complex reward shaping, or maximum control.

In both paths, you start with SFT. The difference is what comes after.

## LoRA: making it practical

Training a 7-billion-parameter model requires multiple expensive GPUs. Most people do not have that hardware. LoRA solves this problem.

LoRA (Low-Rank Adaptation) freezes the original model and trains tiny additional matrices — less than 1% of the total parameters. This cuts memory use by 10x or more. A 7B model that normally needs 4 GPUs can train on a single GPU with LoRA.

All four TRL trainers support LoRA. You pass a LoRA configuration to the trainer, and it handles everything automatically. This is how most people do alignment in practice.

## Practical tips

**Start small.** Use 1,000-5,000 examples to validate your pipeline before scaling up. A broken pipeline wastes time no matter how much data you have.

**Data quality matters more than quantity.** 5,000 high-quality preference pairs often beat 50,000 noisy ones. Spend time cleaning your data.

**Use defaults first.** TRL's default hyperparameters were tuned on real workloads. Change them only after you understand what each one does and why you want a different value.

**Monitor during training.** For PPO: watch the KL divergence (should stay below 10) and the mean reward (should increase). For DPO: watch the loss (should decrease) and the implicit reward margin (should increase).

---

You now know the tools that companies use to build AI assistants. TRL wraps the same algorithms behind ChatGPT and Claude into a clean, well-tested library. With SFTTrainer and DPOTrainer alone, you can align a language model in a single afternoon.

**Ready to go deeper?** Head to [trl-library-tutorial-interview.md](./trl-library-tutorial-interview.md) for implementation details, failure modes, and interview-grade depth.
