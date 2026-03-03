# PPO with Stable-Baselines3 — From Scratch to Production

> You just built PPO from scratch. You understand every line — the ratio, the clip, the GAE, the entropy bonus. But would you trust your code to train a robot arm? Or fine-tune a language model? Probably not. Production RL needs more than correct math. It needs tested code, parallelism, logging, checkpointing, and hundreds of small engineering details that took years of community effort to get right. That is what Stable-Baselines3 gives you.

---

**Before you start, you need to know:**
- How PPO works — the clipping trick, multiple epochs, and GAE — covered in [PPO From Scratch](./ppo-from-scratch.md)
- What vectorized environments are and why parallel data collection helps — covered in [A2C and A3C](../policy-gradient/a2c-a3c.md)

---

## The analogy: the professional kitchen

You know how to cook. You have made bread from scratch — measured the flour, kneaded the dough, watched it rise, baked it in your oven. You understand every step.

Now imagine you walk into a professional bakery. They have industrial mixers, proofing cabinets at precise temperatures, conveyor ovens with timers, and a team that has been refining the process for years. The bread is the same recipe. But the kitchen is designed to produce it reliably, at scale, every day.

**Your from-scratch PPO** is like cooking at home. You understand every step. You have full control. But the code might have subtle bugs, it runs on one environment at a time, and it has no logging or checkpointing.

**Stable-Baselines3** is the professional kitchen. Same algorithm, same math, same bread. But the implementation is battle-tested, parallelized, and packed with tools you would have to build yourself: callbacks for monitoring, vectorized environments for speed, save/load for reproducibility.

Knowing how to cook from scratch makes you a better user of the professional kitchen. You understand what the tools are doing and why.

## What the analogy gets right

The parallel captures the real trade-off. From-scratch code is for learning and for novel research where you need to modify the algorithm itself. Library code is for production, for benchmarking, and for anything where reliability matters more than customization. Most practitioners use both: they prototype ideas from scratch, then implement them in or on top of a library.

## The concept in plain words

### What Stable-Baselines3 is

Stable-Baselines3 (SB3) is a Python library that gives you production-ready implementations of common RL algorithms: PPO, A2C, SAC, TD3, DQN, and more. It is built on PyTorch and Gymnasium.

The key word is "stable." The original Stable-Baselines library was created because most RL codebases had bugs that silently produced bad results. SB3 is the successor, rewritten in PyTorch with clean code and thorough testing.

### The three-line quick start

Training PPO with SB3 takes three lines:

1. Create the model: tell it which algorithm, which network type, and which environment
2. Train: tell it how many timesteps to run
3. Save: store the trained model for later

That is it. All the details — rollout collection, GAE computation, clipping, mini-batching, gradient updates — happen inside `model.learn()`. You already know what each of those steps does from the from-scratch notebook. SB3 does them with optimized, tested code.

### Vectorized environments

SB3 can run multiple copies of the same environment in parallel. If you create 4 parallel CartPole environments, each update collects 4 times as much data in the same wall-clock time. This is the same idea as A2C's parallel workers, but handled automatically by the library.

There are two options:
- **DummyVecEnv** — runs all environments in a single process, one after another. Simple, no overhead. Good for fast environments like CartPole.
- **SubprocVecEnv** — runs each environment in its own process, truly in parallel. Good for slow environments like Atari games or physics simulations.

The `make_vec_env()` helper creates either type with one function call.

### Hyperparameters

SB3 uses good defaults. For PPO, the defaults are:

- **clip_range = 0.2** — the clipping parameter you know from the from-scratch version. Maximum 20% policy change per update.
- **n_steps = 2048** — how many steps to collect before each update. More steps means more stable gradients but more memory.
- **batch_size = 64** — how many transitions per mini-batch during the update.
- **n_epochs = 10** — how many passes over the same data per update. Safe because of clipping.
- **learning_rate = 3e-4** — the Adam learning rate. This is usually the first thing to try changing if training is unstable.
- **gae_lambda = 0.95** — the GAE lambda that controls the bias-variance trade-off in advantage estimation.
- **ent_coef = 0.0** — the entropy bonus coefficient. Increase this if the agent stops exploring too early.

These defaults work well for many environments. The most common adjustment is lowering the learning rate for harder tasks.

### Callbacks

Callbacks let you run custom code at specific points during training — after every step, after every rollout, after every update. SB3 provides built-in callbacks for common needs:

- **EvalCallback** — periodically evaluates the agent and saves the best model
- **CheckpointCallback** — saves the model every N steps
- **StopTrainingOnRewardThreshold** — stops training early when the agent reaches a target reward

You can also write custom callbacks by overriding a few methods. This is how you add custom logging, learning rate schedules, or curriculum learning.

### Save and load

Trained models are saved as ZIP files containing the network weights, optimizer state, and hyperparameters. Loading a model restores it exactly. You can also load a model and continue training — useful for fine-tuning or when training is interrupted.

### When to use from-scratch vs. SB3

| | From scratch | Stable-Baselines3 |
|---|---|---|
| **Use for** | Learning, research, novel algorithms | Production, benchmarking, reproducibility |
| **Pros** | Full control, full understanding | Tested, optimized, feature-rich |
| **Cons** | May have bugs, no logging | Less flexible for novel research |

Most practitioners know both: understand the algorithm deeply enough to implement it from scratch, then use SB3 for real work.

## Where the analogy breaks down

A professional kitchen is physically different from a home kitchen — different equipment, different layout, different scale. SB3's PPO and your from-scratch PPO are running the same math, the same algorithm, with the same PyTorch operations. The difference is entirely in the engineering: error handling, parallelism, logging, and testing. A professional kitchen changes the *process* of baking. SB3 only changes the *reliability* of the same process.

---

**Quick check — can you answer these?**
- What is the difference between DummyVecEnv and SubprocVecEnv?
- Why can PPO safely reuse the same data for 10 epochs?
- When would you use from-scratch PPO instead of SB3?

If you cannot answer one, go back and re-read that section. That is completely normal.

---

## Victory lap

You now know how to use the same tool that researchers and engineers use every day to train RL agents. SB3's PPO is what people mean when they say "we trained with PPO." The algorithm is what you learned in the from-scratch notebook. The library is what makes it production-ready. You have both pieces.

---

Ready to go deeper? See the full math, failure modes, and interview-grade Q&A in [ppo-with-stable-baselines-interview.md](./ppo-with-stable-baselines-interview.md).
