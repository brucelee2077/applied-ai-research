# Game Playing with Reinforcement Learning

In 2013, a computer program learned to play 49 different Atari games — just from looking at the screen and pressing buttons. Nobody told it the rules. Nobody showed it how to play. It figured everything out by itself, and on many games it played better than any human ever had. Three years later, another program beat the world champion at Go, a game so complex that there are more possible board positions than atoms in the universe. How did it learn?

---

**Before you start, you need to know:**
- What a policy does (it chooses actions based on states) — covered in `../fundamentals/policies-and-value-functions.md`
- What DQN is (a neural network that estimates action values) — covered in `../deep-rl/dqn-from-scratch.md`
- What PPO is (a stable policy gradient algorithm) — covered in `../advanced-algorithms/ppo-from-scratch.md`

---

## The analogy: learning to play a new arcade game

Imagine you walk up to an arcade machine you have never seen before. There is a screen showing something colorful, a joystick, and some buttons. You have no instruction manual.

What do you do? You start pressing buttons. Most of what happens makes no sense at first. But then you notice: when you moved the joystick left, a little character on screen moved left. When you pressed a button, the character jumped. You try random things for a while.

Then something good happens — your score goes up! You are not sure why, but you remember what you just did. You try it again. Score goes up again. Over dozens of games, you start to figure out what actions lead to points and what actions lead to losing lives. You develop strategies. You get better and better.

That is exactly how RL agents learn to play games.

### What the analogy gets right

- The agent (player) sees the game screen (state) and chooses buttons to press (actions)
- The agent gets points (rewards) for good play and loses lives (penalties) for mistakes
- The agent learns by trial and error — there is no instruction manual
- Over many games (episodes), the agent discovers which strategies work
- The agent starts bad and gradually becomes good, just like a real player

### The concept in plain words

RL for game playing follows a simple loop:

**Step 1: Look at the screen.** The agent sees the current game state — positions of characters, enemies, objects, and the score.

**Step 2: Choose an action.** The agent picks what to do — move left, move right, jump, fire, or do nothing. Early on, the choices are mostly random. Over time, the agent learns which actions work best in each situation.

**Step 3: See what happens.** The game responds. The character moves. Enemies react. The score might change. The agent gets a reward (positive for good things, negative for bad things).

**Step 4: Learn from the result.** The agent updates its policy — its internal rules for choosing actions. Actions that led to good outcomes become more likely. Actions that led to bad outcomes become less likely.

**Step 5: Repeat.** The agent plays the game thousands or millions of times, getting better with each game.

Different algorithms handle Step 4 differently. DQN learns a value for each action and picks the highest one. PPO directly adjusts the policy to make good actions more likely. A2C does both at the same time. But the overall loop is always the same.

### Classic games and why they matter

Researchers use a set of standard game environments to test and compare RL algorithms:

**CartPole** is the "Hello World" of RL. A pole is balanced on a moving cart. The agent pushes the cart left or right to keep the pole upright. It is simple enough that most algorithms can solve it, which makes it useful for testing whether your code works.

**LunarLander** is a step up. The agent controls a spacecraft and must land it safely between two flags. It has four actions (fire left, right, main engine, or do nothing) and must manage fuel while avoiding crashes.

**MountainCar** is surprisingly hard. A car sits in a valley between two hills. The engine is too weak to drive straight up the hill. The agent must learn to rock back and forth to build momentum — a strategy that takes many games to discover because the reward only comes at the end.

**Atari games** are where it gets exciting. The agent sees raw pixels on the screen and must figure out the game from those pixels alone. This is what DQN was built for, and it worked on 49 different games without being told the rules of any of them.

### Where the analogy breaks down

A real arcade player knows what a character, an enemy, and a coin look like. They bring years of experience understanding visual scenes. An RL agent starts with no knowledge at all — it must learn from scratch that certain pixel patterns mean "enemy" and others mean "reward." This makes the learning process much longer and much harder than for a human.

---

**Quick check — can you answer these?**
- What are the four steps in the game-playing RL loop?
- Why is MountainCar harder than CartPole, even though it has fewer states?
- What makes Atari games special compared to simple control tasks?

If you cannot answer one, re-read that part. That is completely normal.

---

## From simple games to superhuman AI

The journey from CartPole to AlphaGo follows a clear path. Each step used the same core ideas — value functions, policy gradients, neural networks — but applied them to harder and harder problems.

In 1992, TD-Gammon used temporal difference learning with a neural network to play backgammon at expert level. This was one of the first successes of combining RL with neural networks.

In 2013, DeepMind combined DQN with convolutional neural networks and learned to play Atari games from raw pixels. This showed that a single algorithm could handle many different games without being told the rules.

In 2016, AlphaGo combined deep RL with Monte Carlo tree search and beat the world champion at Go. This was widely considered impossible just a few years earlier.

In 2019, AlphaStar reached Grandmaster level at StarCraft II — a game with imperfect information, real-time decisions, and thousands of possible actions at each step.

And in 2022-2024, the same PPO algorithm that was tested on CartPole and Atari was used to align language models through RLHF, creating ChatGPT and Claude. The domain changed, but the algorithm stayed.

---

You just learned how RL agents learn to play games — the same trial-and-error loop that powered the revolution from Atari to superhuman Go to the AI assistants we use today. The game-playing domain is where every major RL idea was first tested and proven, and it remains the best place to build intuition for how these algorithms work.

**Ready to go deeper?** Head to [game-playing-interview.md](./game-playing-interview.md) for the full math, failure modes, and interview-grade depth.
