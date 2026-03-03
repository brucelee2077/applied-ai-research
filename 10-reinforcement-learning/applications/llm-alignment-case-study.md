# LLM Alignment Case Study

In 2022, ChatGPT became the fastest-growing app in history. But the raw language model behind it — GPT-3.5 — was not very helpful on its own. It would ramble, ignore instructions, and sometimes say harmful things. What turned it from a clever autocomplete into a useful assistant? A three-stage training process that taught it what humans actually want.

---

**Before you start, you need to know:**
- What RLHF does at a high level (using human feedback to improve a model) — covered in `../rlhf/what-is-rlhf.md`
- What DPO is (a simpler alternative to PPO for preference learning) — covered in `../rlhf/dpo-and-alternatives.md`

---

## The analogy: training an apprentice

Imagine you hire an apprentice who has read every book in the library. They know an enormous amount, but they have never actually done a job. If you ask them to fix a leaky faucet, they might write an essay about the history of plumbing instead of telling you to tighten the fitting.

Training this apprentice happens in three stages.

**Stage 1 — Education (pre-training).** The apprentice reads millions of documents. They learn how language works, what facts exist, and how ideas connect. But they have no idea how to be useful.

**Stage 2 — Job training (supervised fine-tuning / SFT).** You show the apprentice examples of good work. "When someone asks a question, answer it clearly. Use bullet points when listing steps. Offer to help further at the end." The apprentice copies these patterns and learns the format of a helpful response.

**Stage 3 — Mentorship (DPO / RLHF).** You show the apprentice two versions of their work and say "this one is better than that one." Over many examples, the apprentice learns not just the format of a good response, but the subtle qualities that make one response preferred over another — clarity, honesty, appropriate detail, and knowing when to say "I'm not sure."

### What the analogy gets right

- The apprentice (model) starts with broad knowledge but no practical skill
- SFT teaches the structure and format of good responses — like teaching someone what a good report looks like
- DPO/RLHF teaches preferences — the difference between a correct response and a great response
- Each stage builds on the previous one — you cannot skip steps
- The mentorship stage uses comparisons, not absolute scores — "this is better than that" is easier to judge than "rate this on a scale of 1 to 10"

### The concept in plain words

LLM alignment is the process of making a language model do what humans want. It has three parts.

**Supervised fine-tuning (SFT)** takes a pre-trained model and trains it on examples of good instruction-response pairs. The model learns to follow instructions, answer questions directly, and use a helpful tone. This is standard supervised learning — the model predicts the next word in high-quality responses.

**Direct Preference Optimization (DPO)** goes further. It uses pairs of responses where a human has marked one as better than the other. The training objective is simple: increase the probability of the preferred response and decrease the probability of the rejected response, while staying close to the original model. The "staying close" part is important — without it, the model could change so much that it forgets its base knowledge.

**Evaluation** checks whether alignment actually worked. Automated metrics compare the aligned model against a baseline on held-out prompts. LLM-as-judge uses a strong model (like GPT-4) to rate response quality. Human evaluation is the gold standard but expensive. Safety testing tries to make the model produce harmful outputs to check its guardrails.

### Where the analogy breaks down

A real apprentice understands why good work is good — they build judgment. A language model does not understand anything. It learns statistical patterns that correlate with human approval. This means it can learn superficial shortcuts: writing longer responses because longer responses were usually preferred in training, or agreeing with the user because agreement was usually preferred. These shortcuts — reward hacking, length gaming, sycophancy — are the main failure modes of alignment.

---

**Quick check — can you answer these?**
- Why does SFT come before DPO? What would happen if you skipped SFT?
- What does the KL constraint in DPO prevent?
- Name two ways an aligned model can game its reward without actually being helpful.

If you cannot answer one, re-read that part. That is completely normal.

---

You just learned the recipe behind every modern AI assistant — from ChatGPT to Claude to Gemini. The same three stages (pre-train, SFT, preference learning) power all of them. The differences between assistants come down to the quality of training data, the choice of preference algorithm, and how carefully the failure modes are monitored. Every time you chat with an AI and it gives a helpful, honest, structured answer — that is alignment working.

**Ready to go deeper?** Head to [llm-alignment-case-study-interview.md](./llm-alignment-case-study-interview.md) for the full math, failure modes, and interview-grade depth.
