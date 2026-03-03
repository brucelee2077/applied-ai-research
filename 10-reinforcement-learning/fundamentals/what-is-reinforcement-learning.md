# What is Reinforcement Learning?

Have you ever wondered how a computer can learn to beat the best human player at a game — without anyone telling it the rules? No training data. No answers to copy. It just plays, fails, and gets better. How?

The answer is **reinforcement learning** — and it is one of the most powerful ideas in all of AI.

---

**Before you start, you need to know:**
- What machine learning is at a high level (a computer that learns from data)
- Nothing else. Start here.

---

## The Analogy: Learning to Ride a Bike

Think about the first time you rode a bike.

Nobody handed you a textbook. Nobody showed you 10,000 photos of people riding bikes. Instead, you got on, wobbled, fell, and tried again. When you stayed balanced for a few seconds — that felt great. When you crashed — that hurt. Over time, your brain figured out what works.

**What the analogy gets right:**
- You learn by *doing*, not by studying examples
- Good outcomes (staying balanced) encourage you to repeat what you did
- Bad outcomes (falling) push you to try something different
- Nobody tells you the right answer — you discover it through experience

**The concept in plain words:**

Reinforcement learning (RL) is a type of machine learning where a computer program (called an **agent**) learns by interacting with a world (called an **environment**). The agent tries things, gets feedback (called **rewards**), and slowly figures out the best way to act.

There are three big ideas:
1. **Agent and environment.** The agent acts. The environment responds. This happens over and over in a loop.
2. **Rewards.** After each action, the environment gives the agent a number: positive for good, negative for bad. The agent's goal is to collect as much total reward as possible.
3. **Policy.** The agent's strategy — the set of rules it uses to pick actions. A good policy leads to high rewards. A bad policy leads to low rewards.

**Where the analogy breaks down:** When you ride a bike, you are learning one skill. An RL agent can learn to play chess, control a robot, or write helpful text — the same framework works for very different problems.

---

**Quick check — can you answer these?**
- What are the two main characters in every RL problem?
- How does an agent know if it did something good or bad?
- What is a "policy" in RL?

If you cannot answer one, re-read the section above. That is completely normal.

---

## What You Just Unlocked

Reinforcement learning is the idea behind some of the most impressive AI achievements in history. AlphaGo used RL to beat the world champion at Go. OpenAI Five used RL to beat professional Dota 2 teams. And ChatGPT uses a technique called RLHF (Reinforcement Learning from Human Feedback) to learn what kind of answers humans prefer.

Every one of those systems started with the same simple loop: act, get feedback, learn. You now understand that loop.

---

Ready to go deeper? → [what-is-reinforcement-learning-interview.md](./what-is-reinforcement-learning-interview.md)
