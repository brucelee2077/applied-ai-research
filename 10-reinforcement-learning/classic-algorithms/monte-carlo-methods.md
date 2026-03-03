# Monte Carlo Methods

**Here is something surprising.** The first programs that learned to play backgammon well did not use any clever tricks. They just played thousands of complete games, wrote down what happened, and slowly figured out which moves led to winning. No teacher. No rules about strategy. Just play, observe, learn. This simple idea — learning from complete experiences — is called **Monte Carlo methods**, and it is one of the oldest and most powerful ideas in reinforcement learning.

---

**Before you start, you need to know:**
- What the Bellman equation does — it says the value of a state depends on the value of the states that come after it (covered in [bellman-equations.md](../bellman-equations.md))
- What a policy is — a rule that tells the agent what to do in each state (covered in [policies-and-value-functions.md](../policies-and-value-functions.md))
- What a value function is — a number that says how good it is to be in a certain state (covered in [policies-and-value-functions.md](../policies-and-value-functions.md))

---

## The analogy

Imagine you are watching a recording of a full basketball game you just played. The game is over. You know the final score. Now you rewind the tape and look at every play you made.

You pause at minute 12. You passed the ball to your teammate, and two plays later, your team scored. Good pass. You write that down — that pass was worth something.

You pause at minute 25. You took a risky three-pointer and missed. The other team grabbed the ball and scored. Bad shot. You write that down too.

By the time you finish reviewing the entire tape, you have a sense of which plays helped and which plays hurt. You did not need a coach telling you what to do. You just watched what happened and learned from the results.

That is Monte Carlo learning.

## What the analogy gets right

- **You need the full game to be over before you learn.** You cannot judge a play until you know how the rest of the game turned out. A risky pass in minute 5 might look bad at first, but if it set up the winning play, it was actually great. Monte Carlo methods wait until the episode (the full game) ends, then look back.
- **You learn from your own experience.** Nobody hands you a textbook of basketball strategy. You learn by playing games and seeing what happens. Monte Carlo methods do the same — they learn from the actual outcomes of actual episodes, not from a model of how the world works.
- **You average over many games.** One game is not enough. Maybe you made a great pass but your teammate fumbled it. That does not mean the pass was bad. After reviewing 100 games, the noise cancels out and you start to see which plays are truly good. Monte Carlo methods average the returns from many episodes to get accurate value estimates.

## The concept in plain words

A Monte Carlo method works like this:

1. **Play a full episode from start to finish.** An episode is a complete experience — a full game, a full maze run, a full round. The agent takes actions, sees what happens, and collects rewards along the way.

2. **Wait until the episode ends.** This is the key difference from other methods. Monte Carlo does not try to learn mid-game. It waits.

3. **Look back at every state you visited.** For each state, add up all the rewards you collected from that state onward until the end. This total is called the **return** for that state.

4. **Update your value estimate.** If you have visited this state in many episodes, your value estimate for it is just the average of all the returns you have seen. Over time, this average gets closer and closer to the true value.

That is the whole idea. Play. Wait. Look back. Average.

There is one important choice to make: if you visit the same state twice in one episode, do you count it once or twice? Counting only the first visit is called **first-visit Monte Carlo**. Counting every visit is called **every-visit Monte Carlo**. Both work. First-visit is more common.

## Where the analogy breaks down

In basketball, you can watch the tape and understand *why* a play worked — you see the defense was out of position, you see your teammate was open. But Monte Carlo methods do not understand *why*. They only know that a state led to a good or bad return. They have no insight into the mechanics. They just track numbers.

---

**Quick check — can you answer these?**
- Why does a Monte Carlo method need to wait until the episode ends before learning?
- If you visit the same state three times in one episode, how does first-visit MC handle that?
- Why do you need many episodes, not just one, to get good value estimates?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## Victory lap

You just learned the foundation that made some of the earliest game-playing AI systems possible. Monte Carlo methods were used in the famous TD-Gammon system for backgammon. They are still used today inside AlphaGo and AlphaZero — when those systems play millions of simulated games against themselves, they are using Monte Carlo ideas to figure out which positions are good and which are bad. Every time you hear about an AI "learning by self-play," Monte Carlo methods are part of the story.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [monte-carlo-methods-interview.md](./monte-carlo-methods-interview.md)
