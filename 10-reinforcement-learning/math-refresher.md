# Math Refresher for Reinforcement Learning

This file explains every mathematical symbol and concept used in the RL interview deep-dive files. You do not need to memorize it. Keep it open in a second tab and look things up as you need them.

If you already know a symbol, skip it. If you do not, read the plain English explanation and the small example. That is all you need to keep going.

---

**Before you start, you need to know:**
- Basic arithmetic (addition, multiplication, fractions)
- What a variable is (a letter that stands for a number)

That is it. Everything else is explained below.

---

## Part 1: General Math Notation

### Σ — Summation (adding things up)

**Step 1 — Words.** The Greek letter sigma (Σ) means "add up a list of things." You specify what changes (the index), where to start, and where to stop. It is a shorthand for writing out a long addition.

**Step 2 — Formula.**

```
Σ_{k=0}^{3} x_k = x_0 + x_1 + x_2 + x_3

Where:
  k = the index (it counts from start to stop)
  0 = start value
  3 = stop value
  x_k = the thing being added for each value of k
```

**Step 3 — Worked example.** Suppose x_0 = 2, x_1 = 5, x_2 = 1, x_3 = 7.

```
Σ_{k=0}^{3} x_k = 2 + 5 + 1 + 7 = 15
```

You may also see Σ_a which means "sum over all possible values of a." If there are 4 actions (left, right, up, down), then Σ_a f(a) = f(left) + f(right) + f(up) + f(down).

---

### Π — Product (multiplying things together)

**Step 1 — Words.** The Greek letter pi (Π) means "multiply a list of things together." It works just like Σ, but with multiplication instead of addition.

**Step 2 — Formula.**

```
Π_{k=1}^{3} x_k = x_1 × x_2 × x_3

Where:
  k = the index
  1 = start value
  3 = stop value
```

**Step 3 — Worked example.** Suppose x_1 = 2, x_2 = 3, x_3 = 4.

```
Π_{k=1}^{3} x_k = 2 × 3 × 4 = 24
```

In RL, you will see Π used for the probability of a full trajectory (sequence of actions). If you took 3 actions with probabilities 0.8, 0.6, 0.9, the probability of that trajectory is 0.8 × 0.6 × 0.9 = 0.432.

---

### E[X] — Expected Value (the weighted average)

**Step 1 — Words.** E[X] means "the average outcome you would get if you repeated something many, many times." Each outcome is weighted by how likely it is. Likely outcomes matter more than unlikely ones.

Think of it like this: if you roll a fair 6-sided die, you sometimes get 1, sometimes 6. But on average, you get 3.5. That is the expected value.

**Step 2 — Formula.**

```
E[X] = Σ_x x × P(x)

Where:
  X = a random variable (something whose value depends on chance)
  x = a specific value X can take
  P(x) = the probability of that value happening
```

**Step 3 — Worked example.** A coin flip: heads gives you 10 points, tails gives you 2 points. Each happens with probability 0.5.

```
E[reward] = 10 × 0.5 + 2 × 0.5 = 5 + 1 = 6
```

On average, you earn 6 points per flip.

**Variations you will see:**
- `E_π[...]` — expected value when you follow policy π (your strategy)
- `E_τ[...]` — expected value averaged over trajectories (full episodes)
- `E_{x~D}[...]` — expected value when x is drawn from dataset D

---

### ∇ — Gradient (which direction to move)

**Step 1 — Words.** The upside-down triangle ∇ (called "nabla") means "gradient." A gradient tells you which direction to change your parameters to make a function increase the fastest. It is like standing on a hillside and finding the direction of steepest climb.

If you have a function that depends on multiple parameters, the gradient gives you a separate "push" direction for each parameter.

**Step 2 — Formula.**

```
∇_θ f(θ) = [ ∂f/∂θ₁, ∂f/∂θ₂, ..., ∂f/∂θ_n ]

Where:
  θ = a vector of parameters [θ₁, θ₂, ..., θ_n]
  ∂f/∂θ₁ = how much f changes when you nudge θ₁ a tiny bit
  ∇_θ = "the gradient with respect to θ"
```

**Step 3 — Worked example.** Suppose f(θ₁, θ₂) = θ₁² + 3·θ₂. At θ = [2, 1]:

```
∂f/∂θ₁ = 2·θ₁ = 2·2 = 4   (increasing θ₁ makes f grow by 4)
∂f/∂θ₂ = 3                   (increasing θ₂ always makes f grow by 3)

∇_θ f = [4, 3]
```

To increase f as fast as possible, move θ in the direction [4, 3]. In neural networks, we use the gradient to update weights so the loss decreases (gradient descent) or the reward increases (gradient ascent).

---

### argmax and argmin — "Which input gives the best output?"

**Step 1 — Words.** `max` gives you the biggest value. `argmax` gives you the input that produces the biggest value. The "arg" stands for "argument" — the thing you plug in.

Think of it like a cooking contest. `max` tells you the highest score (98 points). `argmax` tells you which chef got 98 points (Chef B).

**Step 2 — Formula.**

```
max_a f(a) = the largest value of f across all choices of a
argmax_a f(a) = the specific a that produces the largest f(a)
```

**Step 3 — Worked example.** Suppose Q(s, a) gives the value of each action in a state:

```
Q(s, left)  = 3.2
Q(s, right) = 7.1
Q(s, up)    = 5.0
Q(s, down)  = 2.8

max_a Q(s, a)    = 7.1       (the highest value)
argmax_a Q(s, a) = right     (the action that gives 7.1)
```

In RL, `argmax_a Q(s, a)` means "pick the action with the highest Q-value." This is how greedy policies work.

---

### ∈ — "Belongs to" / "Is a member of"

**Step 1 — Words.** The symbol ∈ means "is in" or "belongs to." It says that a value is part of a set or range.

**Step 2 — Formula.**

```
γ ∈ [0, 1)

Where:
  γ = the discount factor
  [0, 1) = the range from 0 to 1, including 0 but not including 1
  Square bracket [ = includes that endpoint
  Round bracket ) = excludes that endpoint
```

**Step 3 — Worked example.**

```
γ ∈ [0, 1) means γ can be 0, 0.5, 0.99, 0.999, but NOT 1.0
r_t ∈ ℝ means the reward can be any real number (positive, negative, or zero)
s ∈ S means state s is one of the states in the state space S
```

---

### ≥, ≤, ≈ — Comparison symbols

**Step 1 — Words.** These compare two values:
- ≥ means "greater than or equal to"
- ≤ means "less than or equal to"
- ≈ means "approximately equal to"

**Step 2 — Formula and examples.**

```
π(a|s) ≥ 0       Every action probability is zero or positive (never negative)
Σ_a π(a|s) = 1   Action probabilities add up to exactly 1
V̂(s) ≈ V^π(s)    Our estimate V̂ is close to (but not exactly) the true value V^π
```

---

### ∀ — "For all" / "For every"

**Step 1 — Words.** The symbol ∀ means "for all" or "for every." It says a statement is true no matter which value you pick.

**Step 2 — Formula and example.**

```
π(a|s) ≥ 0  ∀a, s

This means: "The probability of any action in any state is never negative."
It is true for every possible action a and every possible state s.
```

---

### ∞ — Infinity

**Step 1 — Words.** The symbol ∞ means "goes on forever." In RL, you see it in sums that run forever (infinite horizon tasks) and in convergence conditions.

**Step 2 — Formula and example.**

```
G_t = Σ_{k=0}^{∞} γ^k · r_{t+k}

This sum goes from k=0 to infinity.
It does not actually blow up because γ < 1 makes each term smaller.
γ^0 = 1, γ^1 = 0.99, γ^2 = 0.98, γ^10 = 0.90, γ^100 = 0.37, γ^1000 ≈ 0.00004
The terms shrink so fast that the infinite sum converges to a finite number.
```

---

### log and exp — Logarithm and exponential

**Step 1 — Words.** `exp(x)` (also written `e^x`) makes numbers grow very fast. `log(x)` is the reverse — it asks "what power do I raise e to in order to get x?" They undo each other: log(exp(x)) = x.

In RL, log appears constantly in policy gradients because log probabilities have nice mathematical properties. exp appears in softmax and in the optimal RLHF policy.

**Step 2 — Formula.**

```
exp(x) = e^x    where e ≈ 2.718

log(exp(x)) = x     (they cancel each other)
exp(log(x)) = x

log(a × b) = log(a) + log(b)     ← this is why logs are useful:
                                     they turn products into sums
```

**Step 3 — Worked example.**

```
exp(0) = 1
exp(1) = 2.718
exp(2) = 7.389
exp(-1) = 0.368

log(1) = 0
log(2.718) = 1
log(7.389) = 2
log(0.5) = -0.693

Why log is useful in RL:
  Probability of a trajectory = Π_t π(a_t|s_t)  (multiply many small numbers)
  Log probability = Σ_t log π(a_t|s_t)          (add numbers instead — more stable)
```

---

### ||x|| — Norm (measuring size)

**Step 1 — Words.** Double bars ||x|| measure how "big" something is. There are different types of norms. The most common in RL is the infinity norm ||x||_∞ which finds the largest element.

**Step 2 — Formula.**

```
||x||_∞ = max_i |x_i|     (the biggest absolute value in the vector)

Where:
  |x_i| = absolute value of the i-th element
```

**Step 3 — Worked example.**

```
x = [-3, 1, 2]
||x||_∞ = max(|-3|, |1|, |2|) = max(3, 1, 2) = 3

In RL, the Bellman contraction uses ||V - V'||_∞ to measure how different
two value functions are — specifically, the largest disagreement at any state.
```

---

## Part 2: Probability Notation

### P(A|B) — Conditional probability

**Step 1 — Words.** P(A|B) means "the probability of A happening, given that B has already happened." The vertical bar | is read as "given" or "knowing that."

**Step 2 — Formula.**

```
P(A|B) = P(A and B) / P(B)

Where:
  P(A|B) = probability of A given B
  P(A and B) = probability of both A and B happening
  P(B) = probability of B happening
```

**Step 3 — Worked example.** You have a bag with 3 red balls and 2 blue balls. You draw one ball and it is red. What is the probability the next ball is also red?

```
P(2nd red | 1st red) = 2/4 = 0.5

After removing one red, there are 2 red and 2 blue left (4 total).
```

In RL, you will see:
- `P(s'|s, a)` — probability of ending up in state s', given you are in state s and take action a
- `π(a|s)` — probability of choosing action a, given you are in state s

---

### x ~ P — Sampling notation

**Step 1 — Words.** The tilde ~ means "is drawn from" or "is sampled from." When you write `x ~ P`, it means x is a random value chosen according to the distribution P.

**Step 2 — Formula and examples.**

```
a ~ π(·|s)      Action a is sampled from policy π in state s
τ ~ π_θ         Trajectory τ is generated by running policy π_θ
x ~ D           Input x is drawn from dataset D
ε ~ N(0, 1)     ε is a random number from a standard normal distribution
```

**Step 3 — Worked example.** If π(left|s) = 0.7 and π(right|s) = 0.3, then sampling a ~ π(·|s) gives you "left" 70% of the time and "right" 30% of the time.

---

### Probability distributions

**Step 1 — Words.** A probability distribution tells you how likely each possible outcome is. Every probability must be between 0 and 1, and all probabilities must add up to 1.

**Step 2 — Key distributions in RL.**

```
Categorical distribution (discrete choices):
  π(a|s) = [0.1, 0.6, 0.2, 0.1]  for actions [left, right, up, down]
  Rule: all values ≥ 0, they sum to 1

Gaussian/Normal distribution (continuous values):
  N(μ, σ²)
  Where: μ = mean (center), σ² = variance (spread)
  Used for continuous action spaces (e.g., robot joint angles)

Uniform distribution (equal chance for all):
  Each outcome equally likely
  Used in random exploration
```

---

### σ — Sigmoid function

**Step 1 — Words.** The sigmoid function takes any number and squishes it into the range (0, 1). Large positive numbers become close to 1. Large negative numbers become close to 0. Zero maps to exactly 0.5. It turns raw scores into probabilities.

**Step 2 — Formula.**

```
σ(x) = 1 / (1 + exp(-x))

Where:
  x = any real number
  σ(x) = output between 0 and 1
```

**Step 3 — Worked example.**

```
σ(0)  = 1/(1 + exp(0))  = 1/(1+1) = 0.5
σ(2)  = 1/(1 + exp(-2)) = 1/(1+0.135) = 0.881
σ(-2) = 1/(1 + exp(2))  = 1/(1+7.389) = 0.119
σ(10) ≈ 0.99995         (almost 1)
σ(-10) ≈ 0.00005        (almost 0)
```

In RL, the sigmoid appears in reward modeling (Bradley-Terry model) and DPO where it converts log-probability differences into a preference probability.

---

### KL Divergence — Measuring distance between distributions

**Step 1 — Words.** KL divergence measures how different two probability distributions are. If two distributions are identical, the KL divergence is 0. The more different they are, the larger the number. It is not symmetric: KL(P||Q) ≠ KL(Q||P).

Think of it as: "How surprised would I be if I expected distribution Q but the real distribution is P?"

**Step 2 — Formula.**

```
KL(P ‖ Q) = Σ_x P(x) × log(P(x) / Q(x))

Where:
  P = the "true" or reference distribution
  Q = the "approximate" or new distribution
  Σ_x = sum over all possible values of x
```

**Step 3 — Worked example.** Two policies over 3 actions:

```
P = [0.5, 0.3, 0.2]   (reference policy)
Q = [0.4, 0.4, 0.2]   (new policy)

KL(P ‖ Q) = 0.5 × log(0.5/0.4) + 0.3 × log(0.3/0.4) + 0.2 × log(0.2/0.2)
           = 0.5 × 0.223 + 0.3 × (-0.288) + 0.2 × 0
           = 0.112 - 0.086 + 0
           = 0.026

Small KL (0.026) means the two policies are quite similar.
```

In RL, KL divergence is used in:
- **TRPO/PPO**: constraining how much the policy can change per update
- **RLHF**: penalizing drift from the reference model: β × KL(π_θ ‖ π_ref)

---

### 𝟙 — Indicator function

**Step 1 — Words.** The indicator function 𝟙(condition) equals 1 if the condition is true, and 0 if it is false. It is like a switch: on or off.

**Step 2 — Formula.**

```
𝟙(condition) = { 1  if condition is true
               { 0  if condition is false
```

**Step 3 — Worked example.**

```
𝟙(s_t = s) = 1 if the agent is in state s at time t, 0 otherwise

In eligibility traces:
  e_t(s) = γλ · e_{t-1}(s) + 𝟙(s_t = s)

This adds 1 to the trace of the current state and lets all other traces decay.
```

---

## Part 3: RL-Specific Notation

### π(a|s) — Policy

**Step 1 — Words.** A policy is the agent's strategy — the rule it uses to pick actions. The notation π(a|s) means "the probability of choosing action a when in state s." A deterministic policy always picks the same action in a given state. A stochastic policy has a distribution over actions.

**Step 2 — Formula.**

```
Deterministic:  a = π(s)           maps state to one action
Stochastic:     π(a|s) = P(a_t = a | s_t = s)   probability distribution

Rules:
  π(a|s) ≥ 0  for all a, s         (probabilities are never negative)
  Σ_a π(a|s) = 1  for all s        (probabilities sum to 1 in each state)
```

**Step 3 — Worked example.** Robot at a crossroads with 3 choices:

```
π(left  | crossroads) = 0.1    (rarely goes left)
π(right | crossroads) = 0.7    (usually goes right)
π(straight | crossroads) = 0.2 (sometimes goes straight)

Sum: 0.1 + 0.7 + 0.2 = 1.0 ✓
```

---

### V^π(s) — State-value function

**Step 1 — Words.** V^π(s) answers the question: "How good is it to be in state s, if I follow policy π from now on?" It is the expected (average) total reward the agent will collect starting from state s. High V means a good situation. Low V means a bad one.

**Step 2 — Formula.**

```
V^π(s) = E_π[ G_t | s_t = s ]

Where:
  E_π = expected value when following policy π
  G_t = the total discounted return from time t onward (defined below)
  s_t = s means "starting from state s"
```

**Step 3 — Worked example.** A game with 3 states. Following policy π, you simulate 4 episodes from state A and record the returns:

```
Episode 1: G = 8.5
Episode 2: G = 9.2
Episode 3: G = 7.8
Episode 4: G = 8.1

V^π(A) ≈ (8.5 + 9.2 + 7.8 + 8.1) / 4 = 8.4
```

State A is worth about 8.4 points under this policy. A better policy might make V^π(A) even higher.

---

### Q^π(s, a) — Action-value function

**Step 1 — Words.** Q^π(s, a) answers: "How good is it to take action a in state s, and then follow policy π?" The difference from V is that Q tells you the value of a specific action, not just the state. This is what Q-learning learns.

**Step 2 — Formula.**

```
Q^π(s, a) = E_π[ G_t | s_t = s, a_t = a ]

Where:
  a_t = a means "starting by taking action a"
  After that first action, the agent follows policy π as normal
```

**Step 3 — Worked example.** In state A, you can go left or right:

```
Q^π(A, left)  = 5.2   (going left then following π gives ~5.2 total reward)
Q^π(A, right) = 8.4   (going right then following π gives ~8.4 total reward)

Going right is much better in this state.
V^π(A) = π(left|A) × Q^π(A, left) + π(right|A) × Q^π(A, right)
       = 0.3 × 5.2 + 0.7 × 8.4
       = 1.56 + 5.88 = 7.44
```

---

### A^π(s, a) — Advantage function

**Step 1 — Words.** The advantage measures how much better action a is compared to what you would normally do. Positive advantage means "this action is better than average." Negative means "worse than average." Zero means "exactly as good as what the policy would do anyway."

**Step 2 — Formula.**

```
A^π(s, a) = Q^π(s, a) - V^π(s)

Where:
  Q^π(s, a) = value of taking action a
  V^π(s) = average value across all actions (weighted by policy)
```

**Step 3 — Worked example.** Continuing from the Q example:

```
V^π(A) = 7.44

A^π(A, left)  = Q(A, left)  - V(A) = 5.2 - 7.44 = -2.24  (worse than average)
A^π(A, right) = Q(A, right) - V(A) = 8.4 - 7.44 = +0.96  (better than average)
```

Policy gradient methods use the advantage to decide which actions to reinforce (positive A) and which to discourage (negative A).

---

### G_t — Return (total reward from time t)

**Step 1 — Words.** The return G_t is the total reward the agent collects from time step t onward. Future rewards are discounted by γ — rewards now are worth more than rewards later.

**Step 2 — Formula.**

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ_{k=0}^{∞} γ^k · r_{t+k}

Recursive form:
  G_t = r_t + γ · G_{t+1}

Where:
  r_t = reward at time step t
  γ = discount factor (e.g., 0.99)
```

**Step 3 — Worked example.** Rewards = [1, 0, 10] with γ = 0.9:

```
G_2 = 10                          (last reward, nothing after)
G_1 = 0 + 0.9 × 10 = 9.0         (0 now, plus discounted future)
G_0 = 1 + 0.9 × 9.0 = 9.1        (1 now, plus discounted future)
```

---

### δ_t — TD Error (the surprise signal)

**Step 1 — Words.** The TD error δ_t measures how surprised the agent is after one step. It compares what the agent expected (its current value estimate) with what it actually got (the reward plus the estimated value of the next state). Positive δ means "things went better than expected." Negative δ means "things went worse."

**Step 2 — Formula.**

```
δ_t = r_{t+1} + γ · V(s_{t+1}) - V(s_t)
      ─────────────────────────   ──────
      what we got                 what we expected
      (reward + future estimate)  (current estimate)
```

**Step 3 — Worked example.** V(s) = 5.0, then we get reward = 2 and land in s' where V(s') = 4.0, γ = 0.9:

```
δ = 2 + 0.9 × 4.0 - 5.0
  = 2 + 3.6 - 5.0
  = 0.6

Positive! Things went slightly better than expected.
The agent should increase its estimate of V(s) a little.
```

---

### γ — Discount factor

**Step 1 — Words.** Gamma (γ) controls how much the agent cares about future rewards. γ = 0 means "only care about the next reward." γ = 0.99 means "care about rewards far into the future." It is always between 0 and 1 (not including 1).

**Step 2 — Formula and key property.**

```
γ ∈ [0, 1)

Effective planning horizon ≈ 1 / (1 - γ)

γ = 0.9   → looks ~10 steps ahead
γ = 0.99  → looks ~100 steps ahead
γ = 0.999 → looks ~1000 steps ahead
```

---

### α — Learning rate

**Step 1 — Words.** Alpha (α) controls how much the agent changes its estimates in a single update. A large α (like 0.5) means big changes — learn fast but risk overshooting. A small α (like 0.001) means small, careful changes — learn slowly but more stably.

**Step 2 — Formula.**

```
V(s) ← V(s) + α × [target - V(s)]
                α
                ↑
        How far to move toward the target

α = 0: never learn (V does not change)
α = 1: jump all the way to the target (forget old estimate entirely)
α = 0.1: move 10% of the way toward the target
```

**Step 3 — Worked example.** V(s) = 5.0, target = 8.0, α = 0.1:

```
V(s) ← 5.0 + 0.1 × (8.0 - 5.0)
     = 5.0 + 0.1 × 3.0
     = 5.0 + 0.3
     = 5.3

V moved from 5.0 toward 8.0, but only 10% of the way.
```

---

### r_t — Reward

**Step 1 — Words.** The reward r_t is the number the agent receives from the environment after taking an action at time step t. It tells the agent how good or bad that step was. The agent's goal is to maximize the total (discounted) reward over time.

**Step 2 — Formula.**

```
r_t = R(s_t, a_t, s_{t+1})

Where:
  s_t = state before the action
  a_t = the action taken
  s_{t+1} = the state after the action
  R = the reward function (defined by the environment)
```

---

### P(s'|s, a) — Transition probability

**Step 1 — Words.** P(s'|s, a) is the probability of landing in state s' when you take action a in state s. It describes the dynamics of the environment. In a deterministic environment, P is always 0 or 1. In a stochastic environment, multiple next states are possible.

**Step 2 — Formula.**

```
P(s'|s, a) = probability of transitioning to s' given (s, a)

Rules:
  P(s'|s, a) ≥ 0             (probabilities are non-negative)
  Σ_{s'} P(s'|s, a) = 1      (you always end up somewhere)
```

**Step 3 — Worked example.** In a slippery grid world, when you try to go right:

```
P(actually go right | s, right) = 0.8    (80% chance it works)
P(slip up | s, right) = 0.1              (10% chance you slip up)
P(slip down | s, right) = 0.1            (10% chance you slip down)

Sum: 0.8 + 0.1 + 0.1 = 1.0 ✓
```

---

### V*, Q*, π* — Optimal value functions and policy

**Step 1 — Words.** The star (*) means "optimal" — the best possible. V*(s) is the highest value any policy can achieve in state s. Q*(s, a) is the highest value achievable by starting with action a. π* is the policy that achieves these optimal values.

**Step 2 — Formula.**

```
V*(s) = max_π V^π(s)              (best V over all possible policies)
Q*(s, a) = max_π Q^π(s, a)        (best Q over all possible policies)
π*(s) = argmax_a Q*(s, a)          (pick the action with highest Q*)
```

**Step 3 — Worked example.**

```
Suppose two policies exist:
  Policy A: V^A(start) = 6.0
  Policy B: V^B(start) = 8.5

V*(start) = max(6.0, 8.5) = 8.5
The optimal value of the start state is 8.5 (achieved by policy B).
```

---

### π_θ — Parameterized policy

**Step 1 — Words.** In deep RL, the policy is a neural network. The subscript θ means "the policy's behavior depends on the network's weights θ." When you update θ (through gradient descent), the policy changes — it picks different actions.

**Step 2 — Formula.**

```
π_θ(a|s) = neural_network(s; θ)[a]

Where:
  θ = all weights and biases in the network
  s = the input (current state)
  π_θ(a|s) = output probability for action a
```

---

### π_ref — Reference policy

**Step 1 — Words.** In RLHF, the reference policy is a frozen copy of the model before RL training. The training process uses a KL penalty to prevent the new policy from drifting too far from this reference. This keeps the model coherent and prevents reward hacking.

**Step 2 — Formula.**

```
RLHF objective includes: -β × KL(π_θ ‖ π_ref)

Where:
  π_θ = the policy being trained (changes over time)
  π_ref = the frozen reference (does not change)
  β = how strongly to penalize drift
  KL = the divergence between the two policies
```

---

### θ⁻ — Target network parameters

**Step 1 — Words.** In DQN, the target network is a frozen (or slowly updating) copy of the Q-network. It is used to compute stable training targets. The minus sign (⁻) means "lagging behind the main network." Without it, the targets shift with every update, making training unstable.

**Step 2 — Formula.**

```
Hard update: θ⁻ ← θ           (copy exactly, every C steps)
Soft update: θ⁻ ← τ·θ + (1-τ)·θ⁻   (blend slowly, every step, τ ≈ 0.005)

Where:
  θ = current network weights (updated every step)
  θ⁻ = target network weights (updated slowly)
  τ = blending coefficient (small = slow tracking)
```

---

### r(θ) — Probability ratio (PPO)

**Step 1 — Words.** In PPO, the probability ratio r(θ) compares the new policy to the old policy for a specific action. If r = 1, the policies agree. If r > 1, the new policy made the action more likely. If r < 1, it made the action less likely.

**Step 2 — Formula.**

```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)

Where:
  π_θ = current (new) policy
  π_θ_old = policy used to collect the data
```

**Step 3 — Worked example.**

```
Old policy: π_old(a|s) = 0.3
New policy: π_new(a|s) = 0.45

r = 0.45 / 0.3 = 1.5

The new policy made this action 50% more likely.
PPO would clip this to at most 1.2 (with ε=0.2) to prevent too-large changes.
```

---

### log π — Log probability

**Step 1 — Words.** log π(a|s) is the logarithm of the policy's probability for action a. Since probabilities are between 0 and 1, log probabilities are always negative (or zero). We use log probabilities instead of raw probabilities because: (1) they are more numerically stable (no underflow from multiplying tiny numbers), and (2) the policy gradient theorem involves ∇log π, not ∇π.

**Step 2 — Formula.**

```
log π(a|s) = log( probability of action a in state s )

Since 0 < π(a|s) ≤ 1:
  log π is always ≤ 0
  Higher probability → less negative log probability
```

**Step 3 — Worked example.**

```
π(left|s) = 0.7  → log(0.7) = -0.357   (likely action, small negative)
π(right|s) = 0.3 → log(0.3) = -1.204   (unlikely action, large negative)

In REINFORCE, the gradient is: ∇_θ log π(a|s) × G_t
If G_t is positive, the update makes the taken action MORE likely.
If G_t is negative, the update makes the taken action LESS likely.
```

---

### ε — Epsilon (exploration or clipping)

**Step 1 — Words.** The Greek letter epsilon (ε) appears in two different contexts in RL — same symbol, different meaning depending on where you see it:

1. **ε-greedy exploration:** With probability ε, pick a random action (explore). With probability 1-ε, pick the best known action (exploit).
2. **PPO clipping:** The ratio r(θ) is clipped to [1-ε, 1+ε] to prevent large policy updates.

**Step 2 — Formulas.**

```
ε-greedy:
  action = { random action          with probability ε
           { argmax_a Q(s, a)        with probability 1 - ε

PPO clip:
  clip(r, 1-ε, 1+ε)   keeps r between 0.8 and 1.2 (when ε = 0.2)
```

---

## Part 4: Key Mathematical Operations

### The Log-Derivative Trick

**Step 1 — Words.** The log-derivative trick is the core of the policy gradient theorem. It rewrites a hard-to-compute gradient (the gradient of a probability) into something we can estimate from samples. The idea: the derivative of log(f) equals the derivative of f divided by f. Rearranging, the derivative of f equals f times the derivative of log(f).

**Step 2 — Formula.** Starting from the chain rule:

```
Step A: ∇log f(x) = ∇f(x) / f(x)       (chain rule applied to log)
Step B: Multiply both sides by f(x):
        f(x) × ∇log f(x) = ∇f(x)       (the log-derivative trick)
```

Now apply this to the policy gradient:

```
∇_θ J(θ) = ∇_θ Σ_τ p(τ|θ) R(τ)
         = Σ_τ ∇_θ p(τ|θ) × R(τ)
         = Σ_τ p(τ|θ) × ∇_θ log p(τ|θ) × R(τ)    ← used the trick!
         = E_τ[ ∇_θ log p(τ|θ) × R(τ) ]

We turned a gradient of a probability (hard) into an expectation (easy to sample).
```

**Step 3 — Worked example.** Suppose p(τ|θ) = 0.3 and ∇_θ p(τ|θ) = 0.06:

```
Using the trick:
  ∇_θ log p(τ|θ) = ∇_θ p(τ|θ) / p(τ|θ) = 0.06 / 0.3 = 0.2

Check: p × ∇log p = 0.3 × 0.2 = 0.06 = ∇p ✓
```

---

### Geometric Series — Why discounted sums converge

**Step 1 — Words.** When you add up an infinite series where each term is a fixed fraction of the previous term, the sum converges to a finite number. This is why discounted returns do not blow up — each future reward is multiplied by γ, γ², γ³, etc., and these get smaller and smaller.

**Step 2 — Formula.**

```
Σ_{k=0}^{∞} γ^k = 1 / (1 - γ)    when |γ| < 1

Where:
  γ = the common ratio between consecutive terms
  The formula works because the tail of the series shrinks to nothing
```

**Step 3 — Worked example.** With γ = 0.9:

```
Sum = 1 + 0.9 + 0.81 + 0.729 + 0.656 + ...
    = 1 / (1 - 0.9)
    = 1 / 0.1
    = 10

So the maximum possible return with γ = 0.9 and R_max = 1 is 10.
With γ = 0.99: max return = 1/(1-0.99) = 100.
With γ = 0.5: max return = 1/(1-0.5) = 2.
```

---

### Contraction Mapping — Why value iteration converges

**Step 1 — Words.** A contraction mapping is a function that brings things closer together. If you apply it over and over, everything converges to a single point. The Bellman backup operator is a contraction: no matter what value estimates you start with, repeatedly applying the Bellman equation brings them closer to the true values.

Think of it like this: you guess a number, and I have a rule that moves your guess closer to the right answer. Every time I apply my rule, your error shrinks. Eventually, you reach the right answer.

**Step 2 — Formula.**

```
||T[V] - T[V']||_∞ ≤ γ × ||V - V'||_∞

Where:
  T = the Bellman backup operator (one step of value iteration)
  V, V' = any two value functions
  ||...||_∞ = maximum difference across all states
  γ = the discount factor (and the contraction rate)
```

After k applications, the error shrinks by at least γ^k:

```
||T^k[V] - V*||_∞ ≤ γ^k × ||V - V*||_∞
```

**Step 3 — Worked example.** Two initial guesses, γ = 0.9:

```
V  = [10, 5]    (guess 1)
V' = [0, 0]     (guess 2)
||V - V'||_∞ = max(|10-0|, |5-0|) = 10

After 1 Bellman backup:
  ||T[V] - T[V']||_∞ ≤ 0.9 × 10 = 9.0    (gap shrinks to at most 9)

After 10 backups:
  Error ≤ 0.9^10 × 10 = 3.49               (gap shrinks to at most 3.49)

After 100 backups:
  Error ≤ 0.9^100 × 10 ≈ 0.00003           (practically zero)
```

---

### Importance Sampling — Learning from someone else's data

**Step 1 — Words.** Importance sampling lets you estimate the expected value under one distribution using samples from a different distribution. In RL, this is used when you want to learn about one policy (the target) using data collected by a different policy (the behavior). You correct for the mismatch by weighting each sample.

**Step 2 — Formula.**

```
E_{x~P}[f(x)] = E_{x~Q}[ f(x) × P(x)/Q(x) ]

Where:
  P = the distribution you care about (target)
  Q = the distribution you sampled from (behavior)
  P(x)/Q(x) = the importance weight (correction factor)
```

For a trajectory of T steps:

```
ρ = Π_{t=0}^{T-1} π_target(a_t|s_t) / π_behavior(a_t|s_t)
```

**Step 3 — Worked example.** You want to know the average reward under policy π_target, but your data came from π_behavior:

```
Action taken: a = right
π_target(right|s) = 0.8
π_behavior(right|s) = 0.4

Importance weight: 0.8 / 0.4 = 2.0

This sample counts double because the target policy would have taken
this action twice as often as the behavior policy did.

If π_target(right|s) = 0.2 and π_behavior(right|s) = 0.6:
  Weight = 0.2 / 0.6 = 0.33

This sample counts for only 1/3 because the target policy rarely takes it.
```

**Warning:** Over many steps, importance weights can become very large or very small (they multiply together), causing high variance. This is why PPO uses clipping instead of raw importance sampling.

---

## Quick Reference Table

| Symbol | Name | One-line meaning |
|--------|------|-----------------|
| Σ | Summation | Add up a list |
| Π | Product | Multiply a list |
| E[X] | Expected value | Weighted average outcome |
| ∇ | Gradient | Direction of steepest ascent |
| argmax | Argmax | Input that gives the max output |
| ∈ | Element of | Belongs to a set or range |
| ∀ | For all | True for every value |
| ∞ | Infinity | Goes on forever |
| log / exp | Log / Exponential | Inverse operations; logs turn products into sums |
| \|\|x\|\| | Norm | Measures size of a vector |
| P(A\|B) | Conditional probability | Probability of A given B happened |
| x ~ P | Sampling | x is drawn randomly from distribution P |
| σ(x) | Sigmoid | Squishes any number to (0, 1) |
| KL(P\|\|Q) | KL divergence | How different two distributions are |
| 𝟙 | Indicator | 1 if true, 0 if false |
| π(a\|s) | Policy | Probability of action a in state s |
| V^π(s) | State-value | Expected return from state s under π |
| Q^π(s,a) | Action-value | Expected return from (s, a) under π |
| A^π(s,a) | Advantage | How much better a is than average |
| G_t | Return | Total discounted reward from time t |
| δ_t | TD error | Surprise: actual minus expected |
| γ | Discount factor | How much to value future rewards |
| α | Learning rate | How fast to update estimates |
| r_t | Reward | Scalar feedback at time t |
| P(s'\|s,a) | Transition probability | Chance of landing in s' from (s,a) |
| V*, Q*, π* | Optimal | Best possible value or policy |
| π_θ | Parameterized policy | Policy as a neural network with weights θ |
| π_ref | Reference policy | Frozen pre-training model (RLHF) |
| θ⁻ | Target network | Slowly-updated copy of weights |
| r(θ) | Probability ratio | π_new / π_old (PPO) |
| log π | Log probability | Logarithm of action probability |
| ε | Epsilon | Exploration rate or clip range |
