# Robotics Simulation with Reinforcement Learning

In 2019, a robotic hand learned to solve a Rubik's Cube — not by being programmed with a solution algorithm, but by practicing millions of times in a simulated world. The hand had never touched a real cube, yet when researchers put a physical cube in its fingers, it solved it. How did a virtual practice session translate to real-world dexterity?

---

**Before you start, you need to know:**
- What a continuous action space is (actions are real numbers, not choices from a list) — covered in `../advanced-algorithms/sac-continuous-control.md`
- What PPO does (it trains a policy using clipped updates) — covered in `../advanced-algorithms/ppo-from-scratch.md`

---

## The analogy: teaching someone to dance

Imagine teaching someone to dance who has never moved their body to music before. In a video game, you press one of four buttons: up, down, left, right. But dancing is not like that. You do not choose "move arm" or "don't move arm." You choose *how much* to move your arm — a little, a lot, fast, slow, and in what direction. Every joint in your body moves a specific amount at a specific speed, all at the same time.

A dance instructor watches you dance and gives feedback. "Good, your arm movement was smooth!" "Bad, you moved your legs too fast and fell over." "Better — but you are using way too much energy." Over hundreds of practice sessions, you learn to coordinate all your joints smoothly.

That is what RL does for robots. Each joint gets a continuous number (how much torque to apply), and the reward function tells the robot how well it danced.

### What the analogy gets right

- The robot (dancer) controls many joints at once — not a single choice, but many continuous values at the same time
- The feedback (reward) combines multiple goals: move forward, do not fall, save energy, be smooth
- Learning happens through practice — the robot tries many different movements and gradually improves
- Coordination is the hardest part — moving one joint well is easy, but coordinating six joints to walk is hard

### The concept in plain words

Robotics simulation uses RL to teach robots how to move. There are three big ideas.

**Continuous control is harder than discrete control.** In a game like CartPole, the agent chooses from two options: push left or push right. In robotics, the agent chooses a real number for each joint — like "apply 0.73 units of torque to the hip and -0.21 to the knee." There are infinitely many possible actions, so the agent cannot try them all. It needs smarter algorithms.

**Reward shaping combines multiple goals.** Walking is not just about moving forward. A good walking reward includes: forward velocity (move fast), control cost (do not waste energy), alive bonus (do not fall down), and smoothness (do not jerk around). Getting this mix right is one of the hardest parts of robotics RL.

**Simulation is where robots learn; the real world is where they perform.** Training a real robot by trial and error is slow (real-time), expensive (hardware breaks), and dangerous (robots fall). Simulation is fast (1000x real-time), cheap (no hardware), and safe (crashing costs nothing). But simulation is not perfect — the friction, weight, and physics in the simulator are different from reality. This difference is called the *sim-to-real gap*.

### The environments

**Pendulum** is the simplest robotics task. One joint, one action (torque), and the goal is to swing the pendulum up and keep it balanced. It is the "Hello World" of continuous control.

**HalfCheetah** is a 2D running robot with 6 joints. The goal is to run as fast as possible without falling. It is the standard benchmark for continuous control algorithms.

**Ant** is a four-legged robot with 8 joints. All four legs must coordinate to walk forward. This is much harder than HalfCheetah because the agent must discover a stable gait.

**Humanoid** is the hardest standard environment — a bipedal robot with 17 joints and 376-dimensional state space. Balancing on two legs while walking forward requires precise coordination of every joint.

### Where the analogy breaks down

A real dancer has proprioception — an innate sense of where their body parts are in space. They also have years of experience with gravity, balance, and movement. An RL agent starts with no physical intuition at all. It does not know that falling hurts or that legs should alternate. It must discover everything from scratch, which is why training takes millions of steps.

---

**Quick check — can you answer these?**
- What makes continuous control harder than discrete control?
- Why does a locomotion reward need multiple components (not just "move forward")?
- What is the sim-to-real gap, and why does it matter?

If you cannot answer one, re-read that part. That is completely normal.

---

## Solving the sim-to-real gap

The biggest practical challenge in robotics RL is making simulated training work on real hardware. Three approaches have proven effective.

**Domain randomization** changes the simulation parameters randomly during training. Every episode, the friction, mass, motor strength, and sensor noise are slightly different. The robot cannot rely on any specific setting, so it learns a policy that works across a range of conditions — including the real world.

**System identification** goes the other direction. Instead of making the policy robust, it makes the simulation accurate. Engineers measure the real robot's friction, weight, and dynamics, then calibrate the simulator to match. This gives a smaller sim-to-real gap, but it requires careful measurement.

**Fine-tuning on real hardware** trains mostly in simulation, then does a small number of updates on the real robot. This combines the speed of simulation with the accuracy of real experience.

---

You just learned how RL teaches robots to move — from swinging a pendulum to walking on two legs. The same algorithms you learned in earlier modules (PPO, SAC) power these robotic systems. The main new challenges are continuous action spaces, multi-objective reward design, and bridging the gap between simulation and reality.

**Ready to go deeper?** Head to [robotics-simulation-interview.md](./robotics-simulation-interview.md) for the full math, failure modes, and interview-grade depth.
