# Fine-Tuning with RLHF

You know every piece of the RLHF puzzle: supervised fine-tuning, reward models, PPO, DPO. Now it is time to put them all together. How do you go from a raw language model to a helpful AI assistant, start to finish?

---

**Before you start, you need to know:**
- The three stages of RLHF — covered in `what-is-rlhf.md`
- How DPO works as an alternative to PPO — covered in `dpo-and-alternatives.md`
- What TRL provides — covered in `trl-library-tutorial.md`

---

## The analogy: the factory assembly line

Imagine a car factory. Raw materials come in one end: steel, glass, rubber. At each station on the assembly line, workers add something: the frame, the engine, the paint. At the end, a finished car rolls out.

Building an aligned language model works the same way. Raw materials come in: a pretrained model, instruction data, preference data. Each stage adds something: instruction-following ability, then preference alignment. At the end, a helpful AI assistant comes out.

The factory analogy also explains why order matters. You cannot paint a car before you build the frame. You cannot align a model with preferences before it can follow instructions. Each stage depends on the one before it.

### What the analogy gets right

- The pipeline has a clear order: each stage builds on the last
- Raw materials (data) go in, a finished product (aligned model) comes out
- Quality control (evaluation) happens at the end to check the product
- If any stage has a problem, the final product suffers

### The concept in plain words

The complete RLHF pipeline has three main stages and one evaluation step.

**Stage 1: Supervised Fine-Tuning (SFT).** Start with a base language model — one that was trained to predict text but does not know how to answer questions. Show it thousands of examples of good question-answer pairs. The model learns the format and style of helpful responses. After this stage, the model can follow instructions, but it does not know which responses are *best*.

**Stage 2: Alignment.** This is where you teach the model what humans prefer. You have two options:

- **Option A (DPO path — recommended):** Go directly from SFT to alignment. Use DPOTrainer with preference data. Two stages total. Simpler, faster, and often just as effective.
- **Option B (PPO path):** Train a reward model first, then run PPO. Three stages total. More complex, but gives you more control and supports online learning.

For most projects, the DPO path is the right choice. You only need the PPO path when you need to generate new responses during training or shape the reward in complex ways.

**Stage 3: Evaluation.** Test the aligned model to make sure it actually improved. Compare its responses to the SFT model. Check that it has not lost basic abilities. Look for signs of reward hacking, mode collapse, or verbose outputs. Use both automated metrics and human evaluation.

### Where the analogy breaks down

In a real factory, each station adds physical parts. In RLHF, each stage changes the same model — adjusting millions of numbers inside the neural network. If you push too hard at Stage 2 (too much alignment), you can damage what Stage 1 built (instruction following). The KL penalty exists to prevent this, but it is a balancing act with no exact analogy in physical manufacturing.

---

**Quick check — can you answer these?**
- What is the recommended path for most alignment projects? (Hint: it has two stages)
- Why does SFT come before alignment?
- What are three things you should check during evaluation?

If you cannot answer one, re-read that part. That is completely normal.

---

## The five pitfalls

Building a complete pipeline means more places where things can go wrong. Here are the five most common problems and how to spot them.

**Reward hacking.** The model finds a trick that scores high on the reward model but is not actually helpful. Symptom: reward scores keep climbing, but the outputs look strange or repetitive. Fix: increase the KL penalty, add human evaluation, use diverse test prompts.

**Mode collapse.** The model gives the same response to every question, or only uses a few sentence patterns. Symptom: very low diversity in outputs. Fix: increase the KL penalty, lower the learning rate, check your preference data for imbalance.

**Catastrophic forgetting.** The alignment process damages the model's basic language abilities. Symptom: the model can no longer answer simple factual questions or generate coherent text. Fix: use LoRA instead of full fine-tuning, lower the learning rate, increase the KL penalty.

**Overfitting to preferences.** The model memorizes the preference training data instead of learning general patterns. Symptom: perfect accuracy on training data, poor performance on new prompts. Fix: use more diverse preference data, add regularization, use early stopping.

**Length bias.** The model learns that longer responses score higher, so it pads every answer with unnecessary detail. Symptom: responses get progressively longer during training. Fix: normalize rewards by response length, include short-response examples in the preference data.

## Evaluation: how to know it worked

Do not rely on a single metric. Use several and check that they all point the same direction.

**Win rate** — show the same prompt to the aligned model and the SFT model, then have a judge (human or another model) pick the better response. The aligned model should win more than 50% of the time.

**KL divergence** — measure how much the aligned model changed from the SFT model. Too low (below 5) means it barely learned anything. Too high (above 15) means it changed too much and might have lost important abilities.

**Harmful output check** — test the model with adversarial prompts designed to produce harmful content. The aligned model should refuse or redirect these, while the SFT model might comply.

**Human evaluation** — the gold standard. Have real people rate the model's responses for helpfulness, accuracy, and safety. Expensive, but nothing else gives the same confidence.

---

You now know how to build an AI assistant from scratch — from a raw pretrained model all the way to a helpful, aligned product. The recommended path is simple: SFT first, then DPO. Two stages, two trainers, one afternoon. The result is a model that understands what you ask and gives answers humans actually prefer.

**Ready to go deeper?** Head to [fine-tuning-with-rlhf-interview.md](./fine-tuning-with-rlhf-interview.md) for the complete implementation details, failure mode analysis, and interview-grade depth.
