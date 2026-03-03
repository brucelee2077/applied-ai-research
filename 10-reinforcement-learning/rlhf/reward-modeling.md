# Reward Modeling

You want to build an AI assistant that gives helpful answers. You have a language model that can generate many different responses to the same question. Some responses are great. Some are mediocre. Some are bad. How do you tell them apart automatically — without a human reading every single one?

---

**Before you start, you need to know:**
- What RLHF is and why it has three stages — covered in `what-is-rlhf.md`
- What a neural network does at a high level (it takes numbers in and puts numbers out)

---

## The analogy: the essay grader

Imagine a professor who teaches a class of 10,000 students. She cannot grade every essay personally — there are too many. But she has an idea.

She takes 1,000 pairs of essays. For each pair, she reads both and marks which one is better. Then she gives these 1,000 comparisons to a teaching assistant and says: "Learn how I grade. Then grade the rest."

The teaching assistant studies the comparisons. She notices the professor prefers essays that are clear, well-organized, and answer the question directly. She notices the professor dislikes essays that ramble, go off-topic, or state things that are wrong. After studying enough comparisons, the teaching assistant can grade new essays the same way the professor would — even essays the professor never saw.

**The reward model is this teaching assistant.** It learns to score responses the way a human would, by studying thousands of human comparisons.

### What the analogy gets right

- The professor (human annotators) provides comparisons, not absolute scores
- The teaching assistant (reward model) learns the professor's preferences from these comparisons
- Once trained, the teaching assistant can grade thousands of new essays automatically
- The teaching assistant does not need to know the subject matter — she only needs to recognize quality patterns

### The concept in plain words

A **reward model** is a neural network that takes a prompt and a response, and outputs a single number: a score. Higher score means the response is better — more helpful, more accurate, more like what a human would prefer.

The reward model does not learn from absolute scores. It learns from **comparisons**. You show it two responses to the same prompt and tell it which one a human preferred. It learns to give the preferred response a higher score.

Here is how the comparison works. Given two responses A and B to the same prompt, the reward model computes:

- Score for A: say, 7.3
- Score for B: say, 4.1

If humans preferred A, the reward model is correct — it gave A a higher score. The training process pushes the model to make the preferred response score higher and the rejected response score lower. After training on thousands of comparisons, the reward model can score new responses it has never seen before.

This comparison-based approach is called the **Bradley-Terry model**. It is the same math used to rank chess players. You do not need to know exactly how strong a player is — you just need to know who beats whom.

### Where the analogy breaks down

A real teaching assistant can ask the professor for clarification: "What do you mean by 'clear'?" A reward model cannot. It must figure out the professor's preferences entirely from comparisons. If the comparisons do not cover some type of response, the reward model has no idea how to score it — and it might give a wildly wrong score.

---

**Quick check — can you answer these?**
- What does a reward model take as input, and what does it output?
- Why do we train on comparisons instead of having humans assign scores directly?
- What is the Bradley-Terry model, in one sentence?

If you cannot answer one, re-read that part. That is completely normal.

---

## Why comparisons beat absolute scores

You might wonder: why not just have humans rate each response on a scale of 1 to 10? Why go through the trouble of comparisons?

There are three reasons.

**Comparisons are more consistent.** If you ask ten people to rate a response from 1 to 10, you will get ten different numbers. One person's 7 is another person's 5. But if you ask ten people "Is response A or B better?", they agree much more often. Comparisons remove the calibration problem.

**Comparisons are faster.** Reading two short responses and picking the better one takes about 30 seconds. Writing a detailed rubric and assigning a justified score takes minutes. Speed matters when you need hundreds of thousands of data points.

**Comparisons capture subtlety.** Some qualities — like helpfulness, tone, or honesty — are hard to define on a numerical scale. But when you see two responses side by side, you can usually tell which one is more helpful. Comparisons let humans express preferences they could not easily put into numbers.

## What the reward model looks like inside

The reward model is usually a transformer — the same kind of neural network used in the language model itself. In fact, it is often initialized from the same model that was fine-tuned in Stage 1 (SFT). The only difference is the output: instead of predicting the next word, it outputs a single number (the reward score).

The architecture looks like this:
1. Take the prompt and response as input
2. Run them through the transformer to get a rich representation
3. Take the representation of the last token
4. Pass it through a single linear layer to get one number — the score

That is it. The reward model is just a language model with a different head.

## The pitfalls

Reward models are powerful but imperfect. Three things can go wrong.

**Length bias.** The reward model might learn that longer responses score higher — not because they are better, but because the training data happened to prefer longer answers. Then the language model learns to be verbose, padding every answer with unnecessary detail. The fix: normalize scores by response length, and make sure the training data includes cases where the shorter response is better.

**Reward hacking.** The language model might find weird tricks that score high on the reward model but are not actually helpful. For example, repeating a specific phrase that the reward model likes, or generating text that exploits a blind spot. This is why the KL penalty in Stage 3 is so important — it prevents the language model from drifting too far.

**Distribution shift.** The reward model was trained on responses from the SFT model. But during PPO training, the language model changes and generates different kinds of responses. The reward model might give unreliable scores on these new types of responses, because it never saw anything like them during training.

---

You just learned how to build the "judge" in the RLHF pipeline. The reward model turns human preferences into a score that the language model can optimize. Without it, we would need a human to read every response — millions of them. With it, we can train AI assistants at scale.

**Ready to go deeper?** Head to [reward-modeling-interview.md](./reward-modeling-interview.md) for the full math, failure mode analysis, and interview-grade depth.
