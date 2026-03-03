# Q-Learning

**Here is a strange trick that makes learning much faster.** Imagine you are trying to find the best restaurant in a new city. You eat at a random place and think, "That was okay, but I bet the *best* item on the menu would have been amazing." So you rate the restaurant not based on what you actually ordered, but based on what you *would have* ordered if you had known better. That sounds like cheating — and in a way it is. But this "optimistic cheating" is exactly what makes Q-learning so powerful. It lets an agent explore randomly while learning as if it were already perfect.

---

**Before you start, you need to know:**
- How temporal difference learning works — updating your guess after every step using the next step's estimate (covered in [temporal-difference-learning.md](./temporal-difference-learning.md))
- What a Q-value is — a number that tells you how good it is to take a specific action in a specific state (covered in [policies-and-value-functions.md](../policies-and-value-functions.md))

---

## The analogy

You are on vacation in a city you have never been to. Every night you pick a restaurant for dinner. Some nights you pick randomly just to explore. Some nights you go back to a place you liked.

After each dinner, you rate the restaurant. But here is the key: you do not rate it based on the dish you actually ate. You rate it based on the best dish on the menu — the one you *would* pick if you went back knowing everything.

So you might visit a sushi place, order a mediocre California roll (because you were exploring), but notice that the chef's omakase special looked incredible. You rate the restaurant high — not because of your actual experience, but because of what the best possible experience would be.

Over time, your restaurant ratings become very accurate, even though you ate random dishes half the time. And when you finally stop exploring and start just going to the highest-rated places and ordering the best dishes, you have an excellent guide.

That is Q-learning. It explores freely. It makes mistakes on purpose. But it always learns as if it will make the best choice next time.

## What the analogy gets right

- **You rate based on the best possible future, not what actually happened.** When the Q-learning agent takes a random action (explores), it does not use that random action to judge the next state. It asks, "What is the best action I *could* take from the next state?" and uses that to update. This is called learning from the **maximum** Q-value.
- **Exploration and learning are separated.** You can eat random dishes (explore) without messing up your ratings (learning). In Q-learning, the agent can follow any exploration strategy — even a completely random one — and still learn the optimal values. The policy the agent follows (the **behavior policy**) is different from the policy it is learning about (the **target policy**). This separation is called **off-policy** learning.
- **You only need one visit to start forming a rating.** You do not need to eat at a restaurant 100 times. One visit, combined with your estimate of the best dish, gives you something to work with. Q-learning also updates after a single step.

## The concept in plain words

Q-learning is a way to learn the value of every action in every state, one step at a time.

Here is how it works:

1. **You are in a state, and you pick an action.** Sometimes you pick the action you think is best. Sometimes you pick a random action to explore. A common rule is called **epsilon-greedy**: with probability epsilon (a small number like 0.1), pick randomly. Otherwise, pick the action with the highest Q-value.

2. **You take the action.** You land in a new state and get a reward.

3. **You ask: what is the BEST action I could take from this new state?** You look up the Q-values for all actions in the new state and pick the biggest one. This is the "optimistic" part. You assume you will play perfectly from now on.

4. **You update.** You combine the reward you just got with that best-possible future value. If this combination is higher than your current Q-value for the state-action pair, you nudge it up. If lower, you nudge it down.

The thing that makes Q-learning special is step 3. The agent does not look at what it *will* actually do next. It looks at what the *best* action would be. This means the agent can explore wildly — take random actions, try weird strategies — and still converge to the optimal Q-values. The exploration does not corrupt the learning.

This is why Q-learning is called an **off-policy** method. The policy it follows (which might be random) is different from the policy it learns about (which is always the best possible one).

## Where the analogy breaks down

In the restaurant analogy, you can see the entire menu and know what the "best dish" is before ordering. In Q-learning, the agent does not actually know which action is truly best — it only knows its current estimates, which might be wrong. The agent's idea of "the best action" changes as it learns. Early on, its estimates are poor, and the "best" action might actually be terrible. The estimates only become accurate after enough exploration.

---

**Quick check — can you answer these?**
- What does "off-policy" mean, and why is Q-learning off-policy?
- Why can Q-learning explore randomly and still learn the optimal values?
- What is the role of the "max" in Q-learning — what is it choosing, and why?

If you cannot answer one, go back and re-read that part. That is completely normal.

---

## Victory lap

You just learned the algorithm that made DeepMind famous. In 2013, DeepMind combined Q-learning with a deep neural network (creating "Deep Q-Networks" or DQN) and trained it to play Atari games at superhuman levels — straight from raw pixels. The agent had no idea what Breakout or Space Invaders was. It just tried random actions, observed rewards (the game score), and used Q-learning to figure out the best thing to do in every situation. That single paper changed the field of AI and led directly to AlphaGo two years later. The foundation of all of it is the algorithm you just learned.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [q-learning-interview.md](./q-learning-interview.md)
