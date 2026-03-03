# Comparing Advanced Methods — Choosing the Right Algorithm

> You now know four advanced RL algorithms: TRPO, PPO, SAC, and TD3. Each has a different philosophy. TRPO guarantees safe updates. PPO simplifies TRPO with clipping. SAC maximizes entropy for better exploration. TD3 uses twin critics with a deterministic policy. But which one should you actually use? The answer depends on your task — and there is a simple decision process to follow.

---

**Before you start, you need to know:**
- How TRPO and PPO constrain policy updates — covered in [Trust Region Methods](./trust-region-methods.md) and [PPO From Scratch](./ppo-from-scratch.md)
- How SAC balances reward and entropy — covered in [SAC for Continuous Control](./sac-continuous-control.md)

---

## The analogy: the sports tournament

Imagine a sports tournament with four athletes. Each specializes in different events.

**PPO is the all-rounder.** Good at every event. Maybe not the absolute best in any one, but never bad. If you only get to pick one athlete and you do not know what event is coming, you pick PPO.

**SAC is the creative explorer.** They discover new techniques, try unexpected strategies, and learn quickly from every attempt. They are brilliant at events that require adaptability and continuous precision — but they cannot compete in events with discrete choices (they only do continuous).

**TD3 is the precision specialist.** They commit fully to one move per situation — no randomness. Extremely stable and efficient. Like SAC, they only compete in continuous events. They excel when you need reliability more than exploration.

**TRPO is the theoretician.** They have a mathematical proof that they will never get worse. But their method is complex and slow. In practice, PPO achieves the same results with a fraction of the effort.

No single athlete wins every event. The right choice depends on what you are competing in.

## What the analogy gets right

The parallel captures the real trade-offs. PPO's strength is versatility — it handles discrete and continuous actions, is simple to implement, and has robust defaults. SAC's strength is sample efficiency and exploration in continuous settings. TD3's strength is stability in continuous control. TRPO's strength is theoretical guarantees that rarely matter in practice.

## The concept in plain words

### The two big divides

Two questions split these algorithms into groups:

**On-policy or off-policy?** PPO and TRPO are on-policy — they collect fresh data with the current policy, use it for a few updates, then throw it away. SAC and TD3 are off-policy — they store every experience in a replay buffer and reuse it many times. Off-policy methods need much less data (they are "sample efficient"), but they need more careful engineering to stay stable.

**Stochastic or deterministic?** PPO, SAC, and TRPO output a probability distribution over actions and sample from it. TD3 outputs a single action directly (and adds noise for exploration during training). SAC's stochastic policy gives it natural exploration. TD3's deterministic policy avoids variance from sampling.

### The decision flowchart

Here is how to choose:

1. **Are your actions discrete** (choose from a list, like left/right/up/down)?
   - Yes → **PPO**. It is the only one of these four that handles discrete actions well.

2. **Are your actions continuous** (output a number, like torque or velocity)?
   - Is data expensive (real robots, slow simulations)? → **SAC** or **TD3**. They are off-policy and need far less data.
   - Is data cheap (fast simulations)? → **PPO**. Its stability outweighs SAC's sample efficiency when you can just run more simulations.

3. **Within off-policy, SAC or TD3?**
   - Does the task require good exploration (many possible strategies, sparse rewards)? → **SAC**. Its entropy bonus encourages trying new things.
   - Does the task require stability and precision (the optimal strategy is clear, you just need to execute it)? → **TD3**. No randomness in the policy means less variance.

4. **What about TRPO?**
   - In practice, almost never. PPO achieves similar results with much simpler code. TRPO matters for research papers that need theoretical guarantees.

### When each algorithm shines

**PPO wins when:**
- Actions are discrete (games, language models, RLHF)
- You have lots of compute and fast simulations
- You want the simplest, most reliable option
- You are doing RLHF for language models (PPO is the industry standard)

**SAC wins when:**
- Actions are continuous and data is expensive
- The task benefits from exploration (multiple good strategies exist)
- You want automatic temperature tuning (no manual entropy coefficient)

**TD3 wins when:**
- Actions are continuous and you need maximum stability
- The optimal behavior is clear and you want the policy to commit to it
- SAC's exploration adds unwanted noise

**TRPO wins when:**
- You need a formal guarantee of monotonic improvement (rare in practice)
- You are writing a research paper that builds on TRPO's theory

### The practical reality

In industry, the decision is simpler than it looks:

- **Default to PPO.** It works for almost everything and is hard to mess up.
- **Switch to SAC** if you have continuous actions and need sample efficiency.
- **Try TD3** if SAC is too exploratory for your task.
- **Use TRPO** only if you have a specific theoretical reason.

Most companies and research labs start with PPO and only switch when they hit a specific limitation.

### Compute and memory trade-offs

| | PPO | SAC | TD3 | TRPO |
|---|---|---|---|---|
| **Memory** | Low (no buffer) | High (replay buffer) | High (replay buffer) | Low (no buffer) |
| **Compute per update** | Low | Medium | Medium | High (Fisher matrix) |
| **Environment samples needed** | Many | Few | Few | Many |
| **Implementation complexity** | Simple | Medium | Medium | Hard |

### Reproducibility warning

All RL algorithms are sensitive to random seeds. The same algorithm with different seeds can produce wildly different results. When comparing algorithms, always:

1. Run each algorithm with at least 5 different seeds
2. Report mean and standard deviation
3. Use the same hyperparameter tuning budget for each algorithm

A single run proves nothing. Averages across seeds tell the real story.

## Where the analogy breaks down

Athletes in a tournament compete in the same events with the same rules. RL algorithms face fundamentally different task structures — discrete vs continuous, sparse vs dense rewards, high-dimensional vs low-dimensional. The "tournament" framing suggests a fair competition, but these algorithms are designed for different settings. Comparing SAC on discrete tasks or PPO on sample-limited continuous tasks is not a fair contest — it is using the wrong tool for the job.

---

**Quick check — can you answer these?**
- When would you choose SAC over PPO?
- Why is TRPO rarely used in practice despite having strong theoretical guarantees?
- What is the first algorithm you should try for a new RL project, and why?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Victory lap

You now have a complete toolkit of advanced RL algorithms and you know when to use each one. PPO for general-purpose and RLHF. SAC for sample-efficient continuous control with exploration. TD3 for stable continuous control. TRPO for theoretical guarantees. This is the same set of choices that engineers at OpenAI, DeepMind, and robotics companies make every day. You are thinking about RL the way they do.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [comparing-advanced-methods-interview.md](./comparing-advanced-methods-interview.md).
