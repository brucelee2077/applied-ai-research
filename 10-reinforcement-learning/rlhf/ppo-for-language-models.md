# PPO for Language Models

You have a language model that can generate text, and a reward model that can score how good the text is. Now you need to connect them — make the language model generate text that scores higher. How do you do that without breaking the model?

---

**Before you start, you need to know:**
- What RLHF is and its three stages — covered in `what-is-rlhf.md`
- How a reward model works — covered in `reward-modeling.md`
- What PPO does (clipped policy gradient updates) — covered in `../advanced-algorithms/ppo-from-scratch.md`

---

## The analogy: training a dog

Imagine you are training a dog. The dog already knows how to walk, run, sit, and bark — these are its natural behaviors (pretraining). But you want it to learn specific tricks, like shaking hands or rolling over.

You use a clicker and treats. When the dog does something good, you click and give a treat. When it does something not quite right, you stay silent. Over time, the dog learns which behaviors earn treats and does more of those.

But there is a catch. If you only reward the dog for doing tricks, it might stop being a normal dog. It might refuse to walk, refuse to eat normally, and only do tricks all day. That is not what you want. You want a dog that is still a dog but also knows tricks.

So you keep the dog on a leash during training. The leash says: "You can learn new things, but do not stray too far from your normal self."

**PPO for language models works the same way.** The language model generates text (the dog does things). The reward model scores it (the clicker). PPO updates the model to generate higher-scoring text (the treats). And the KL penalty keeps the model close to its original behavior (the leash).

### What the analogy gets right

- The dog (language model) already has useful abilities before training starts
- The clicker (reward model) provides a signal for what is good
- The treats (PPO gradient) make the dog do more of what gets rewarded
- The leash (KL penalty) prevents the dog from changing too much

### The concept in plain words

PPO for language models takes the same algorithm you learned in the advanced-algorithms section and adapts it for text generation. Here is what changes and what stays the same.

**What stays the same:** The core PPO idea — compute a probability ratio between the new policy and the old policy, clip it to prevent big changes, and use advantages to decide which actions to encourage.

**What changes:** Four things make language models different from standard RL.

**1. The action space is enormous.** In a game, you might have 4 actions: up, down, left, right. In a language model, every action is choosing the next token from a vocabulary of 50,000 or more. The model must pick one word out of fifty thousand at every single step.

**2. Rewards come only at the end.** In a game, you might get a reward after every step. In RLHF, the reward model scores the complete response — all 100 or 200 tokens at once. Individual tokens do not get direct feedback. This makes it hard to figure out which tokens were good and which were bad.

**3. There is a reference model.** Standard PPO does not have a reference policy. RLHF PPO keeps a frozen copy of the SFT model (the reference model) and adds a penalty whenever the current model drifts too far from it. The full reward becomes: reward = RM score - KL penalty.

**4. There is a value head.** To solve the "reward only at the end" problem, a small neural network (the value head) is added on top of the language model. At every token position, the value head predicts: "How much total reward do I expect from here onwards?" This lets PPO compute advantages for each token, even though the actual reward only arrives at the last token.

### Where the analogy breaks down

A dog trainer adjusts in real time — they can see the dog, change the timing of clicks, and adapt. In RLHF, the reward model is frozen during training. If the reward model has blind spots, the language model might learn behaviors that score high but are not actually helpful. The leash (KL penalty) helps, but it does not fully solve this problem.

---

**Quick check — can you answer these?**
- What four things make PPO for language models different from standard PPO?
- What does the value head do, and why is it needed?
- What happens if you remove the KL penalty?

If you cannot answer one, re-read that part. That is completely normal.

---

## How the KL penalty works

The KL penalty is the most important modification to PPO for language models. Here is what it does.

The language model generates a response. For each token in that response, the model assigned a probability. The reference model (the frozen SFT copy) would have assigned a different probability to the same token. The KL penalty measures how much these probabilities differ across the entire response.

If the current model assigns similar probabilities to the reference model, the KL penalty is small. The model has not changed much — good. If the probabilities are very different, the KL penalty is large. The model has drifted far — bad.

The total reward becomes:

**Total reward = RM score - beta x KL divergence**

The parameter beta controls how strong the leash is. If beta is high (say 0.5), the model barely changes from the reference — very safe, but slow to improve. If beta is low (say 0.01), the model can change more freely — faster improvement, but more risk of reward hacking.

Typical values for beta are between 0.05 and 0.2.

## How the value head works

The value head solves a credit assignment problem. The reward model gives one score for the entire response. But the response has 100 tokens. Which tokens were responsible for the high (or low) score?

The value head is a small neural network attached to the language model. At every token position, it looks at the hidden state and predicts: "Given everything generated so far, how much reward do I expect in total?"

With these predictions, PPO can compute an advantage for each token: "Was generating this token better or worse than expected?" Tokens with positive advantage get reinforced. Tokens with negative advantage get suppressed.

The value head adds very few parameters — typically less than 1% of the total model. It shares the same backbone (transformer layers) as the language model. Only the final output layer is different: instead of predicting the next word, it predicts a single number (the expected reward).

## The training loop

Here is what happens during one step of RLHF PPO training:

1. **Generate** — Sample a batch of prompts. The language model generates a response for each prompt.
2. **Score** — The reward model scores each (prompt, response) pair.
3. **Compute KL** — For each token, compute the KL divergence between the current model and the reference model.
4. **Compute rewards** — Total reward = RM score - beta x KL penalty.
5. **Compute advantages** — Use the value head predictions and the total rewards to compute advantages for each token (using GAE, just like standard PPO).
6. **PPO update** — Update the language model and value head using the clipped PPO objective.
7. **Repeat** — Go back to step 1 with the updated model.

This loop runs for thousands of steps. Gradually, the language model learns to generate responses that the reward model scores highly, while staying close to the reference model.

---

You just learned how PPO — the same algorithm used to train game-playing agents — gets adapted for training language models. The key modifications are the KL penalty (to prevent reward hacking), the value head (to assign credit to individual tokens), and the enormous action space (50,000+ tokens per step). This is the engine behind ChatGPT, Claude, and every major AI assistant.

**Ready to go deeper?** Head to [ppo-for-language-models-interview.md](./ppo-for-language-models-interview.md) for the full math, failure modes, and interview-grade depth.
