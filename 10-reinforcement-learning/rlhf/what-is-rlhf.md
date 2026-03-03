# What is RLHF?

Here is something strange: a language model trained on the entire internet knows how to write poetry, solve math problems, and explain quantum physics. But ask it a simple question like "What is the capital of France?" and it might ramble for three paragraphs, go off-topic, or give you something harmful. It knows everything, but it does not know how to help you. How do you fix that?

---

**Before you start, you need to know:**
- What a language model does at a high level (it predicts the next word)
- What PPO does (it trains a policy using clipped updates) — covered in `../advanced-algorithms/ppo-from-scratch.md`

---

## The analogy: a brilliant student who never learned manners

Imagine a student who has read every book in the library. Every textbook, every novel, every Wikipedia article. They know an incredible amount.

Now you ask them: "What is the capital of France?"

A helpful answer is: "The capital of France is Paris."

But this student might say: "The capital of France is Paris. Speaking of France, did you know that the French Revolution started in 1789? Let me tell you about it..." and keep going for ten minutes. Or they might say: "What is the capital of Germany? Berlin. What about Spain? Madrid." — answering questions you never asked.

The student knows the answer. But nobody ever taught them how to give a *helpful* response. Knowing everything is not the same as knowing how to help.

**RLHF is like hiring a tutor to teach this student how to use their knowledge in a way that actually helps people.**

### What the analogy gets right

- The student (the language model) already has vast knowledge from reading (pretraining)
- The problem is not lack of knowledge — it is lack of judgment about what kind of response is useful
- The tutor (RLHF) does not teach new facts — they teach the student how to *present* what they already know
- After tutoring, the student still knows everything they knew before, but now they also know how to be helpful

### The concept in plain words

RLHF stands for **Reinforcement Learning from Human Feedback**. It is a technique that teaches a language model to generate responses that humans actually prefer. The model already knows how to generate text. RLHF teaches it *which* text to generate.

It works in three stages, like training someone for a job:

**Stage 1: Supervised Fine-Tuning (SFT)** — show the model examples of good responses. Think of it as the student reading model essays. The model learns the format and style of helpful responses. After this stage, the model can produce helpful-looking answers, but it does not know which answer is *best*.

**Stage 2: Reward Model Training** — train a separate model to judge response quality. You show it two responses to the same question and ask: "Which one is better?" Humans provide thousands of these comparisons. The reward model learns to predict which response a human would prefer. Think of it as training a teaching assistant to grade essays the way the professor would.

**Stage 3: PPO Fine-Tuning** — use reinforcement learning to improve the language model. The language model generates a response. The reward model scores it. PPO updates the language model to generate higher-scoring responses. This repeats thousands of times. Think of it as the student writing essays, getting grades, and improving based on feedback.

There is one more important piece: the **KL penalty**. During PPO training, the model gets a penalty for changing too much from the SFT version. Without this penalty, the model might find weird tricks to get high scores from the reward model — like repeating the same phrase that scores well, or generating nonsense that happens to fool the reward model. The KL penalty says: "Improve, but stay close to what you learned."

### Where the analogy breaks down

A real student learns continuously from a single tutor. In RLHF, the reward model is trained once (from human comparisons) and then frozen. The language model optimizes against this frozen judge. If the judge has blind spots, the model might learn to exploit them — this is called *reward hacking*, and it does not happen with human tutors.

---

**Quick check — can you answer these?**
- What problem does RLHF solve? (Hint: it is not about teaching the model new facts)
- What are the three stages, and what does each one do?
- Why do we need a KL penalty during PPO training?

If you cannot answer one, re-read that part. That is completely normal.

---

## Why preferences work better than demonstrations

You might wonder: why not just show the model more examples of good responses? Why go through all this trouble with reward models and PPO?

The answer is scaling. Writing a perfect response to a question takes an expert 5-10 minutes. But comparing two responses and saying "this one is better" takes anyone about 30 seconds. You do not need to be an expert to know which answer is more helpful — you just need to read both and pick one.

This means you can collect preference data 10 times faster and 10 times cheaper than demonstration data. And you can get it from regular people, not just domain experts.

There is a deeper reason too. Some qualities are hard to describe but easy to recognize. "Helpful" is hard to define precisely. "Harmless" is hard to write rules for. "Honest" is hard to program. But when you see two responses side by side, you can usually tell which one is more helpful, more harmless, and more honest. Preferences capture qualities that are hard to put into words.

## The history: from Atari to ChatGPT

RLHF did not start with language models. In 2017, researchers first used human preferences to train agents to play Atari games. In 2020, OpenAI applied it to text summarization and showed it beat supervised fine-tuning. In 2022, the InstructGPT paper established the three-stage pipeline. Later that year, ChatGPT used this pipeline at scale and reached 100 million users in months. Today, nearly every major AI company — OpenAI, Anthropic, Google, Meta — uses RLHF or something like it to align their language models.

---

You just learned the technique behind ChatGPT, Claude, and every modern AI assistant. The three-stage pipeline — SFT, reward modeling, PPO — is the foundation of language model alignment. You now understand the "secret sauce" that turns a text predictor into a helpful assistant.

**Ready to go deeper?** Head to [what-is-rlhf-interview.md](./what-is-rlhf-interview.md) for the full math, failure modes, and interview-grade depth.
