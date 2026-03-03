# Temporal Difference Learning

**Imagine two students taking a math test.** Student A finishes the entire test, hands it in, waits for the grade, and only then looks at which problems they got wrong. Student B checks their answer after every single problem — if they got problem 3 wrong, they adjust their approach before starting problem 4. Who do you think learns faster? Student B does. And that is exactly the idea behind temporal difference learning. Instead of waiting until the end to learn, you learn after every single step.

---

**Before you start, you need to know:**
- How Monte Carlo methods work — learning by playing full episodes and averaging returns (covered in [monte-carlo-methods.md](./monte-carlo-methods.md))

---

## The analogy

You are a student taking a 50-question math test. There are two ways to learn from it.

**The Monte Carlo way:** You finish all 50 questions. You hand in the test. A week later, the teacher returns your graded paper. Now you look at every question and figure out what went wrong. But it has been a week. You barely remember what you were thinking on question 12. The feedback is accurate (you know the real score), but it is slow and hard to use.

**The Temporal Difference way:** After you finish question 1, the teacher immediately tells you whether you got it right. You adjust your approach. You do question 2. Immediate feedback again. By question 10, you have already corrected three bad habits. By question 50, you are a much better test-taker than when you started — and you learned *during* the test, not after.

That second approach is temporal difference learning. The "temporal difference" part means: you compare what you *expected* to happen at one moment in time with what you *actually see* at the next moment, and you use that difference to learn.

## What the analogy gets right

- **You learn after every single step, not at the end.** Monte Carlo waits for the full episode to finish. TD learns after each transition — each step from one state to the next. This is faster.
- **You use partial information.** The student does not wait for the full 50-question score. They use the result of just one question to update their strategy. TD methods do the same — they use the reward from one step plus their current guess about the future to update their value estimate.
- **Early corrections prevent later mistakes.** Because the student adjusts after each question, mistakes do not pile up. TD learning works the same way — correcting value estimates early means better decisions throughout the episode.

## The concept in plain words

Here is how temporal difference learning works:

1. **You are in a state.** You have a guess about how good this state is — your current value estimate.

2. **You take one action.** You move to the next state and receive a reward.

3. **You compare.** You look at the reward you just got, plus your guess about how good the *next* state is. This combination is your new, slightly better estimate.

4. **You update.** If this new estimate is higher than your old guess, you nudge your old guess upward. If it is lower, you nudge it downward. The size of the nudge is controlled by a **learning rate** — a small number that keeps you from overreacting to a single step.

The key insight is this: you do not need to know the final outcome. You just need to know what happened in *one step* and what you *think* will happen next. You are using a guess to update a guess. This sounds unreliable, but it works remarkably well — the guesses correct each other over time, like a group of friends checking each other's homework.

The simplest version of this is called **TD(0)** — the "(0)" means you only look one step ahead. There are versions that look more steps ahead, blending the TD idea with Monte Carlo. But TD(0) is the foundation.

## Where the analogy breaks down

A real student gets exact, correct feedback after each question — the teacher tells them the right answer. But in TD learning, the "feedback" is partly made up. The agent uses its own guess about the next state's value, and that guess might be wrong. It is learning from its own imperfect predictions, not from a perfect answer key. Despite this, the method still converges to the right values over time.

---

**Quick check — can you answer these?**
- What is the main advantage of TD learning over Monte Carlo methods?
- What does "using a guess to update a guess" mean in practice?
- Why can TD learning work even in episodes that last a very long time or never end?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## Victory lap

You just learned the idea that Richard Sutton — one of the founding figures of reinforcement learning — has called "the most novel and central idea in the field." Temporal difference learning is the engine behind nearly every modern RL system. It powers the value networks in AlphaGo. It is the foundation of Q-learning and SARSA, which you will learn next. And the concept of learning from moment-to-moment differences, rather than waiting for final outcomes, has even been found in the brain — dopamine neurons appear to compute a signal that looks remarkably like a TD error. You just learned something that connects computer science to neuroscience.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [temporal-difference-learning-interview.md](./temporal-difference-learning-interview.md)
