# Multi-Agent Systems

In 2019, OpenAI trained AI agents to play hide and seek. Nobody told the agents to use tools, build walls, or surf on ramps. But after millions of games against each other, the hiders learned to build shelters and the seekers learned to climb over them. When the hiders found a way to lock the seekers out, the seekers discovered they could launch themselves over walls using a ramp. None of these strategies were programmed. They emerged from competition.

---

**Before you start, you need to know:**
- What Q-learning does (learns action values from experience) — covered in `../classic-algorithms/q-learning.md`
- What actor-critic methods are (separate networks for policy and value) — covered in `../policy-gradient/actor-critic.md`

---

## The analogy: a dance troupe learning together

Imagine a dance troupe of four people learning a new routine without a choreographer. Each dancer decides their own moves. The problem: every time one dancer changes what they are doing, the others need to adjust. If dancer A switches from slow movements to fast ones, dancer B's timing is suddenly off.

Now imagine all four dancers are changing their moves at the same time, every rehearsal. What worked in the last rehearsal might not work now because everyone else changed too. The dance floor becomes unpredictable — not because the stage is different, but because the other dancers keep changing.

### What the analogy gets right

- Each dancer (agent) makes their own decisions based on what they see
- What one dancer does affects what the others should do
- Everyone is learning at the same time, which makes the world feel unstable
- The goal might be shared (put on a great show) or conflicting (win a dance battle)
- Nobody planned the final routine — it emerged from practice

### The concept in plain words

Multi-agent reinforcement learning is what happens when multiple RL agents share the same world. There are three settings, depending on what the agents want.

**Cooperative** — all agents want the same thing. A team of warehouse robots needs to move packages without bumping into each other. They all get the same reward: packages delivered. The challenge is coordination — each robot needs to do its part, and they need to avoid getting in each other's way.

**Competitive** — one agent's win is another's loss. A game of chess has two players with opposite goals. This is called zero-sum. The challenge is that your opponent is also learning. Every time you find a winning strategy, your opponent learns to counter it.

**Mixed** — some cooperation, some competition. A soccer match has two teams. Players on the same team cooperate, but the teams compete against each other. This is the most common real-world setting.

The biggest challenge in all three settings is **non-stationarity**. In single-agent RL, the environment stays the same — the rules of the game do not change. In multi-agent RL, the other agents *are* part of the environment, and they are all learning and changing. What worked yesterday might fail today because everyone else adapted. This breaks the convergence guarantees that single-agent algorithms rely on.

The main solution is **centralized training with decentralized execution (CTDE)**. During training, all agents share information — they can see each other's observations, actions, and rewards. This stabilizes learning because each agent understands the full picture. During execution (in the real world), each agent acts using only its own observations. This makes deployment practical — the agents do not need to communicate in real time.

Two important ideas from game theory help understand multi-agent outcomes. A **Nash equilibrium** is a state where no agent can improve by changing their strategy alone — everyone is doing the best they can given what everyone else is doing. **Pareto optimality** means there is no way to make any agent better off without making another worse off. The striking thing is that these two ideas can conflict: in the Prisoner's Dilemma, the Nash equilibrium (both defect) is worse for everyone than the Pareto optimal outcome (both cooperate).

### Where the analogy breaks down

Real dancers have body language, eye contact, and years of physical intuition about movement. RL agents start with none of this. They cannot watch a partner's body and predict what comes next. They learn purely from numerical observations and reward signals, which makes coordination much harder to discover.

---

**Quick check — can you answer these?**
- Why is multi-agent RL harder than single-agent RL?
- What does "centralized training with decentralized execution" mean in practice?
- In the Prisoner's Dilemma, why do both players defect even though cooperation is better for both?

If you cannot answer one, re-read that part. That is completely normal.

---

You just learned the framework behind some of the most impressive AI achievements in recent years — AlphaStar mastering StarCraft, OpenAI Five beating human teams at Dota 2, and agents that develop their own languages to communicate. Multi-agent RL is where individual intelligence becomes collective intelligence, and where the most surprising, unplanned behaviors emerge.

**Ready to go deeper?** Head to [multi-agent-systems-interview.md](./multi-agent-systems-interview.md) for the full math, failure modes, and interview-grade depth.
