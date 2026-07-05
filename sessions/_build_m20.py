#!/usr/bin/env python3
"""Builder for M20 — RL Foundations (convert from 10-reinforcement-learning).
Six lessons + review, rendered via _lesson_gen. Run: python3 sessions/_build_m20.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m20")
os.makedirs(OUT, exist_ok=True)

def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — The MDP ----------------
L("day-01-mdp.html", dict(
  qid="m20-d01-mdp", title="Module 20 · Day 1 — The MDP", nav_title="Spiral · M20 Day 1",
  eyebrow="Module 20 · Train · Day 1", h1="The MDP: How We Frame a Decision Problem",
  lead="Before a model can learn to <em>act</em>, we need a clean way to write down \"an agent making decisions over time to collect reward.\" That frame is the Markov Decision Process — states, actions, rewards, and a policy. Every RL method in this module, and every RLHF pipeline later, is built on it.",
  goal="<b>🎯 By the end, you'll be able to:</b> name the four pieces of an MDP (state, action, reward, transition), write the discounted return <code>G = r₀ + γr₁ + γ²r₂ + …</code>, and say what a policy and a value function are.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-policy-gradient.html", next_label="Policy Gradient",
  sections=[
    {"title":"An agent, a world, and a score","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> you can read a function and a weighted sum (from the foundations). Nothing about RL — we start from zero.</div></div>
     <h4>Learning by doing</h4>
     <p>Most models you have seen learn from a fixed dataset with correct answers. <span class="term" data-tip="Reinforcement learning: an agent learns which actions to take by trying them, seeing the reward, and adjusting — no labelled correct action is given.">Reinforcement learning (RL)</span> is different: there is no answer key. An <span class="term" data-tip="The decision-maker. It observes the state and chooses an action.">agent</span> tries actions in a world, sees a reward, and slowly learns which actions pay off.</p>
     <h4>The four pieces</h4>
     <p>We frame this as a <span class="term" data-tip="Markov Decision Process: the standard math frame for RL — states, actions, a reward signal, and transition rules, where the next state depends only on the current state and action.">Markov Decision Process (MDP)</span>:</p>
     <ul>
       <li><b>State</b> <code>s</code> — what the agent sees right now (the game screen, the sentence so far).</li>
       <li><b>Action</b> <code>a</code> — what it can do (press a button, pick the next word).</li>
       <li><b>Reward</b> <code>r</code> — a number scoring the result (points, +1 for a good answer).</li>
       <li><b>Transition</b> — taking action <code>a</code> in state <code>s</code> lands you in a new state <code>s'</code>.</li>
     </ul>
     <p>"Markov" means one thing: the next state depends only on the <em>current</em> state and action, not the whole history. That keeps the math clean.</p>''',
     "gotit":"Got the four pieces"},
    {"title":"It's a video game","body":
     '''<div class="relate">
       <div class="card"><span class="big">🎮</span><h5>You are the agent</h5><p>You look at the screen (the state), press a button (the action), and the game gives you points (the reward) and a new screen (the next state). You never get told the "correct" button — you learn what scores by playing.</p></div>
       <div class="card"><span class="big">🏆</span><h5>You want the high score</h5><p>You don't care about one point now — you care about total points by the end. Sometimes you give up a point now (dodge instead of grab) to score more later. That "total future reward" is the whole game.</p></div>
     </div>
     <p><strong>In one line:</strong> the agent looks at a state, picks an action with its policy, gets a reward and a new state, and repeats — trying to maximize total reward over time.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a real game has fixed rules you can see. In RL the agent usually does <i>not</i> know the transition rules or rewards in advance — it discovers them by trying.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"One step of the loop","body":
     '''<p>Below is a simulated run of one agent–environment step. Watch the policy pick an action, the environment return a reward and next state, and the discounted return build up. <strong>Click all three.</strong></p>'''},
    {"title":"Return, discount, and value","body":
     '''<h4>Step 1 — the return, in words</h4>
     <p>The agent maximizes the <span class="term" data-tip="The total (discounted) reward collected from now until the end of the episode. Written G_t.">return</span> — the sum of all future rewards, not just the next one. But a reward now is usually worth more than the same reward far in the future, so we shrink later rewards with a discount.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       G<sub>t</sub> = r<sub>t</sub> + γ·r<sub>t+1</sub> + γ²·r<sub>t+2</sub> + γ³·r<sub>t+3</sub> + …<br>
       <span class="dim"># r = reward at each step, γ = discount factor between 0 and 1</span>
     </div></div>
     <p>Symbols: <code>G_t</code> = return from step t. <code>r_t</code> = reward at step t. <code>γ</code> (gamma) = the <span class="term" data-tip="A number between 0 and 1 that shrinks future rewards. γ near 1 = far-sighted; γ near 0 = only cares about the next reward.">discount factor</span>. A high γ (0.99) makes the agent far-sighted; a low γ (0.5) makes it grab rewards now.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       rewards = [1, 0, 2],  γ = 0.9<br><br>
       G = 1 + 0.9·0 + 0.9²·2<br>
       G = 1 + 0 + 0.81·2<br>
       G = 1 + 1.62 = <span class="hl">2.62</span>
     </div></div>
     <p>The <span class="term" data-tip="The expected return if you start in a state and follow your policy. V(s) answers: how good is it to be here?">value function</span> <code>V(s)</code> is just the <em>expected</em> return from a state if you follow your policy. It answers "how good is it to be here?" — and it is the backbone of the next five lessons.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — reward misspecification.</b> The agent maximizes exactly the reward you write, not what you meant. A boat-racing agent once learned to spin in circles hitting bonus targets forever instead of finishing the race — its reward looked great while the behaviour was useless. This "reward hacking" is silent: the number climbs, the goal is missed. You catch it only by watching behaviour, not the reward curve.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — the discount γ.</b> A γ near 1 makes the agent value the long term, but returns become noisier and learning slower (more far-future rewards add up). A small γ is stable and fast but short-sighted — it will sacrifice a big later reward for a small one now. Choosing γ is choosing how far ahead the agent plans.</div></div>''',
     "gotit":"Got return, discount, value"},
    {"title":"The agent–environment loop","body":
     '''<p>Here is the MDP as a loop, built one piece at a time: the state, the policy choosing an action, the environment's reward and next state, and the return accumulating. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a gridworld","body":
     '''<p>Understanding ≠ writing it. Build a tiny MDP. Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m20_gridworld.py</code>. Make a 4×4 grid where the agent moves up/down/left/right, gets −1 per step and +10 at a goal cell. Run a fixed policy for one episode, print the state, action, reward, and running discounted return at each step with γ=0.9.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 20 Day 1 artifact.
Create experiments/foundations/m20_gridworld.py: a 4x4 gridworld MDP with -1 step reward, +10 goal. Run one episode under a fixed policy and print state, action, reward, and the running discounted return (gamma=0.9) at each step, then the final return.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m20/day-01-log.md</code>: what final return did you get, and how did it change when you set γ=0.5 vs 0.99?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "policy":{"btn":"① policy: state → action","html":'''<span class="prompt">&gt;&gt;&gt;</span> s = env.reset()   <span class="dim"># starting state</span>
<span class="prompt">&gt;&gt;&gt;</span> a = policy(s)   <span class="dim"># the agent chooses</span>
<span class="hl">a = "right"</span>   <span class="dim"># one of the allowed actions</span>''',
      "take":"<b>①  The policy picks an action.</b> Given the state, the policy (the agent's strategy) outputs what to do. Here it chose \"right\"."},
    "step":{"btn":"② env: reward + next state","html":'''<span class="prompt">&gt;&gt;&gt;</span> s2, r, done = env.step(a)
<span class="hl">r = 1</span>       <span class="dim"># reward for that action</span>
<span class="hl">s2 = (0,1)</span>  <span class="dim"># the new state</span>
<span class="dim"># done = False → the episode continues</span>''',
      "take":"<b>②  The environment responds.</b> It returns a reward (1) and the next state. The agent never sees a \"correct\" action — only this reward."},
    "return":{"btn":"③ accumulate the return","html":'''<span class="prompt">&gt;&gt;&gt;</span> rewards = [1, 0, 2]; gamma = 0.9
<span class="prompt">&gt;&gt;&gt;</span> G = 1 + 0.9*0 + 0.9**2*2
<span class="hl">G = 2.62</span>   <span class="dim"># discounted total future reward</span>''',
      "take":"<b>③  The return is the goal.</b> Sum the future rewards, shrinking later ones by γ. The agent tries to make this as large as possible — that is all of RL."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='200' y='20' width='120' height='34' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='42' fill='#1F6280'>state s</text></g></svg>",
     "note":"<b>Start with the state.</b> Everything the agent needs to decide right now — the game screen, the board, the sentence so far."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='25' width='90' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='85' y='44' fill='#1F6280'>state s</text><text x='150' y='44' fill='#6B645E'>→</text><polygon points='180,25 300,38 300,42 180,55' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='240' y='43' fill='#5E5191'>policy π</text><text x='320' y='44' fill='#6B645E'>→</text><rect x='350' y='25' width='90' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='395' y='44' fill='#9A7208'>action a</text></g></svg>",
     "note":"<b>The policy chooses an action.</b> The policy π maps the state to an action (or to probabilities over actions). It is the thing we will learn."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='25' width='80' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='70' y='44' fill='#9A7208'>action a</text><text x='125' y='44' fill='#6B645E'>→</text><polygon points='150,22 290,38 290,42 150,58' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='220' y='43' fill='#1a5c38'>environment</text><text x='310' y='44' fill='#6B645E'>→</text><rect x='340' y='22' width='150' height='36' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='415' y='44' fill='#1F6280'>reward r + next s'</text></g></svg>",
     "note":"<b>The environment answers.</b> It hands back a reward and the next state. That reward is the only feedback signal the agent gets."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='24' width='220' height='34' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='46' fill='#1F6280'>G = r₀ + γr₁ + γ²r₂ + …</text></g></svg>",
     "note":"<b>Rewards accumulate into the return.</b> The agent sums future rewards, discounting later ones by γ. This return is what it is trying to maximize."},
    {"viz":"<svg viewBox='0 0 520 100'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='35' width='90' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='105' y='54' fill='#1F6280'>state s</text><polygon points='180,38 280,50 280,54 180,66' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='230' y='55' fill='#5E5191'>policy</text><rect x='330' y='35' width='120' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='390' y='54' fill='#1a5c38'>env → r, s'</text><path d='M450,66 Q460,90 260,90 Q90,90 100,66' fill='none' stroke='#6B645E' stroke-width='1.5' stroke-dasharray='4,3'/><text x='260' y='86' fill='#6B645E'>loop: s' becomes the new s</text></g></svg>",
     "note":"<b>The loop closes.</b> The next state becomes the new state, and the cycle repeats until the episode ends. Learn a policy that makes the return large, and you have solved the MDP."},
  ],
  quiz=[
    {"q":"1. In an MDP, what does the agent actually try to maximize?","opts":["the reward on the very next step","the discounted return — total future reward","the number of states","the size of the action set"],"ans":1,"fb":"It maximizes the return (sum of future rewards, discounted by γ), not just the immediate reward — that is why it can sacrifice now for later."},
    {"q":"2. What does the \"Markov\" property say?","opts":["rewards are always positive","the next state depends only on the current state and action, not the full history","the policy never changes","there is exactly one action"],"ans":1,"fb":"Markov = memoryless: given the current state and action, the future does not depend on how you got here. It keeps the math tractable."},
    {"q":"3. Rewards are [2, 4] with γ = 0.5. The return G is:","opts":["6","4","3","2"],"ans":1,"fb":"G = 2 + 0.5·4 = 2 + 2 = 4. Later rewards are shrunk by γ before adding."},
    {"q":"4. Your agent's reward keeps rising but its behaviour is clearly wrong. Most likely cause?","opts":["γ is too small","reward misspecification — it is maximizing the reward you wrote, not what you meant","the state is Markov","V(s) is negative"],"ans":1,"fb":"This is reward hacking from a misspecified reward: the number looks great while the behaviour is useless. You must inspect behaviour, not just the reward curve."},
  ],
  fin={"em":"🎮","h3":"Day 1 complete — you can frame a decision problem!",
       "p":"You now have the MDP: state, action, reward, transition, a policy that chooses actions, and the discounted return <code>G = r₀ + γr₁ + γ²r₂ + …</code> the agent maximizes — plus the value function V(s) as expected return. Next: <b>Day 2</b> — how a policy actually learns, with policy gradients."},
))

# ---------------- Day 2 — Policy Gradient / REINFORCE ----------------
L("day-02-policy-gradient.html", dict(
  qid="m20-d02-pg", title="Module 20 · Day 2 — Policy Gradient", nav_title="Spiral · M20 Day 2",
  eyebrow="Module 20 · Train · Day 2", h1="Policy Gradient: Learning to Act",
  lead="Yesterday we framed the problem. Today the agent learns. A policy gradient makes the policy a network that outputs action probabilities, then nudges those probabilities up for actions that led to high reward and down for actions that led to low reward. This one idea — REINFORCE — is the seed of PPO and every RLHF fine-tune.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why we push up the log-probability of good actions, write the policy-gradient estimate <code>∇J = E[∇log π(a|s)·G]</code>, and say why REINFORCE has high variance.",
  prev_href="day-01-mdp.html", prev_label="The MDP", next_href="day-03-baselines.html", next_label="Baselines",
  sections=[
    {"title":"A policy you can train","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (state, action, reward, return) and that a network can output probabilities with softmax (from the foundations).</div></div>
     <h4>The policy is a network</h4>
     <p>A <span class="term" data-tip="A policy whose action probabilities come from a neural network with weights θ. Training adjusts θ.">parameterized policy</span> <code>π<sub>θ</sub>(a|s)</code> takes the state and outputs a probability for each action. To act, you sample from those probabilities. Training means adjusting the weights θ so good actions get more probable.</p>
     <h4>The core move: reinforce what worked</h4>
     <p>After an episode, look at each action taken and the return that followed. If the return was high, increase that action's probability. If it was low, decrease it. Do this over many episodes and the policy drifts toward high-reward behaviour. That is <span class="term" data-tip="The simplest policy-gradient algorithm: run an episode, then increase the log-probability of each action in proportion to the return that followed it.">REINFORCE</span>.</p>
     <h4>Why gradients, not a table</h4>
     <p>With millions of states (pixels, text) you cannot store a value for each. A network generalizes, and gradient ascent gives a principled way to climb toward more reward. This is why policy gradients scale to LLMs.</p>''',
     "gotit":"Got the idea"},
    {"title":"Coaching by outcome","body":
     '''<div class="relate">
       <div class="card"><span class="big">🏀</span><h5>Repeat what scored</h5><p>A player tries a move and it works — they make a mental note to do it more. A move flops — do it less. They don't get told the perfect move; they adjust from the outcome. That is exactly the policy-gradient update.</p></div>
       <div class="card"><span class="big">🎲</span><h5>Probabilities, not certainties</h5><p>The policy doesn't say "always shoot" — it says "shoot 70% of the time." Good outcomes nudge that 70% up; bad ones nudge it down. Keeping it probabilistic lets the agent keep exploring.</p></div>
     </div>
     <p><strong>In one line:</strong> take an action, see the return, and change the policy's weights to make high-return actions more likely and low-return actions less likely.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a coach judges each move on its own. REINFORCE credits <i>every</i> action in the episode with the <i>whole</i> return — even actions that had nothing to do with the score. That crude credit assignment is exactly why it is noisy (Day 3 fixes it).</div></div>''',
     "gotit":"Got the picture"},
    {"title":"One REINFORCE update","body":
     '''<p>Watch a trajectory get sampled, its return computed, and the policy nudged. <strong>Click all three.</strong></p>'''},
    {"title":"The log-probability trick","body":
     '''<h4>Step 1 — what we want</h4>
     <p>We want to change θ to raise the expected return <code>J(θ)</code>. So we need the gradient <code>∇J</code> — "which way to nudge θ to get more reward."</p>
     <h4>Step 2 — the estimate</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       ∇J(θ) = E[ ∇<sub>θ</sub> log π<sub>θ</sub>(a|s) · G ]<br>
       <span class="dim"># for each action taken, its log-prob gradient, scaled by the return G</span>
     </div></div>
     <p>Symbols: <code>π<sub>θ</sub>(a|s)</code> = probability of the taken action. <code>∇log π</code> = the direction in weight-space that makes that action more likely (the <span class="term" data-tip="The gradient of the log-probability of an action. Following it makes that action more likely; it is the handle policy gradient pushes on.">score function</span>). <code>G</code> = the return that followed. Multiply them: a large positive G pushes hard toward the action; a negative G pushes away.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       action "up" had prob 0.5, return G = +3  →  push log-prob of "up" UP (×3)<br>
       action "down" had prob 0.5, return G = −2  →  push log-prob of "down" DOWN (×2)<br>
       <span class="dim"># update: θ ← θ + α · ∇log π(a|s) · G</span>
     </div></div>
     <p>The update rule is gradient <em>ascent</em>: <code>θ ← θ + α·∇log π(a|s)·G</code>. Over many episodes the good actions win out.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — high variance.</b> Using the full episode return G as the weight makes the gradient extremely noisy: the same action can get a big or tiny G just because of luck later in the episode. Training limps along or stalls, and it is not obvious why — the loss is not "wrong," just deafeningly noisy. This is the single biggest weakness of vanilla REINFORCE and the reason for the next three lessons.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — policy-based vs value-based.</b> Policy gradients directly optimize the thing you care about and handle stochastic and continuous actions naturally (great for robotics and text generation). The price: they are on-policy and sample-hungry, and high-variance. Value-based methods (like Q-learning) reuse data and have lower variance, but struggle with continuous actions.</div></div>''',
     "gotit":"Got the log-prob trick"},
    {"title":"From trajectory to update","body":
     '''<p>Here is REINFORCE built up: sample a trajectory, compute the return for each step, and turn it into a weighted push on the policy. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it on CartPole","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m20_reinforce.py</code>. Train a small policy network on CartPole with REINFORCE: run episodes, compute discounted returns, and update with the loss <code>-mean(log π(a|s) · G)</code>. Print the average episode length every 20 episodes and watch it climb.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 20 Day 2 artifact.
Create experiments/foundations/m20_reinforce.py: REINFORCE on CartPole (gym). A small MLP policy outputs action probabilities; each episode compute discounted returns and update with loss = -mean(log_prob(a) * G). Print average return every 20 episodes and note how noisy the curve is.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m20/day-02-log.md</code>: did the average return climb, and how noisy (jumpy) was the learning curve?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "sample":{"btn":"① sample a trajectory","html":'''<span class="prompt">&gt;&gt;&gt;</span> traj = run_episode(policy)
<span class="dim"># states, actions, rewards for one run</span>
<span class="hl">actions = ["up","up","down"]</span>
<span class="hl">rewards = [1, 1, 0]</span>''',
      "take":"<b>①  Play one episode.</b> Sample actions from the policy and record what happened. This is the data for one update."},
    "returns":{"btn":"② compute returns","html":'''<span class="prompt">&gt;&gt;&gt;</span> G = discounted_returns(rewards, gamma=0.9)
<span class="hl">G = [1.9, 1.0, 0.0]</span>
<span class="dim"># each action is credited with the reward that followed it</span>''',
      "take":"<b>②  Credit each action with its return.</b> The first action gets 1 + 0.9·1 + 0.81·0 = 1.9. High return → we will push that action up."},
    "update":{"btn":"③ update the policy","html":'''<span class="prompt">&gt;&gt;&gt;</span> loss = -mean(log_prob(a) * G)
<span class="prompt">&gt;&gt;&gt;</span> loss.backward(); opt.step()
<span class="hl">P(up | s0): 0.50 → 0.58</span>   <span class="dim"># good action made more likely</span>''',
      "take":"<b>③  Nudge the weights.</b> Gradient ascent raises the log-probability of actions with high return. \"up\" went from 0.50 to 0.58. Repeat over many episodes."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>trajectory: s₀,a₀,r₀ → s₁,a₁,r₁ → …</text></g></svg>",
     "note":"<b>Run an episode.</b> Sample actions from the current policy and record the states, actions, and rewards."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>Gₜ = rₜ + γrₜ₊₁ + …</text></g></svg>",
     "note":"<b>Compute the return for each step.</b> Each action taken is credited with the discounted sum of rewards that came after it."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>∇log π(a|s) · G</text></g></svg>",
     "note":"<b>Weight the log-prob gradient by the return.</b> Positive return → push the action's probability up; negative → push it down. The size of the push scales with G."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='120' y='30' fill='#6B645E'>before</text><rect x='70' y='38' width='100' height='16' rx='3' fill='#E4F2F7' stroke='#2A7B9B'/><rect x='70' y='38' width='50' height='16' rx='3' fill='#2A7B9B'/><text x='400' y='30' fill='#6B645E'>after</text><rect x='350' y='38' width='100' height='16' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='350' y='38' width='62' height='16' rx='3' fill='#2D8B55'/></g></svg>",
     "note":"<b>The good action gets more probable.</b> One gradient step raises P(good action). Over many episodes the policy concentrates on high-return behaviour."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><polyline points='40,55 90,50 140,52 190,40 240,46 290,32 340,38 390,22 440,28 480,15' fill='none' stroke='#C99A12' stroke-width='2'/><text x='260' y='68' fill='#9A7208'>average return over episodes — noisy but rising</text></g></svg>",
     "note":"<b>The reward climbs — noisily.</b> Because the whole return weights every action, the curve is jumpy. That noise is the variance problem Day 3 attacks with a baseline."},
  ],
  quiz=[
    {"q":"1. In REINFORCE, what gets pushed up for a high-return action?","opts":["the reward","the log-probability of that action under the policy","the discount factor","the number of episodes"],"ans":1,"fb":"We do gradient ascent on log π(a|s) scaled by the return, so high-return actions become more probable."},
    {"q":"2. Why use ∇log π rather than a lookup table of the best action?","opts":["tables are faster","a network generalizes across huge state spaces (pixels, text) where a table is impossible","log is required by γ","to avoid rewards"],"ans":1,"fb":"With millions of states you cannot tabulate; a parameterized policy generalizes, and the gradient tells you how to improve it."},
    {"q":"3. An action was taken and the return that followed was G = −2. The update will:","opts":["make that action more likely","make that action less likely (push its log-prob down)","not change anything","increase γ"],"ans":1,"fb":"A negative return flips the sign of the push, so the action's probability is decreased."},
    {"q":"4. Your REINFORCE training is extremely jumpy and slow. The core cause is:","opts":["the learning rate is negative","high variance — the full episode return is a noisy weight for each action","γ = 1 is illegal","the softmax is broken"],"ans":1,"fb":"Vanilla REINFORCE weights every action by the whole (luck-laden) return, so gradients are high-variance. Baselines and advantages (next lessons) fix this."},
  ],
  fin={"em":"📈","h3":"Day 2 complete — your agent learns!",
       "p":"You can now train a policy: <code>∇J = E[∇log π(a|s)·G]</code> pushes up the log-probability of high-return actions and down for low-return ones, via gradient ascent <code>θ ← θ + α∇log π·G</code>. You also saw REINFORCE's Achilles heel — high variance. Next: <b>Day 3</b> — a baseline that tames it."},
))
# ---------------- Day 3 — Baselines & Variance Reduction ----------------
L("day-03-baselines.html", dict(
  qid="m20-d03-baseline", title="Module 20 · Day 3 — Baselines", nav_title="Spiral · M20 Day 3",
  eyebrow="Module 20 · Train · Day 3", h1="Baselines: Judge an Action Against Average",
  lead="REINFORCE works but is painfully noisy — it weights every action by the whole episode's return. The fix is small and beautiful: subtract a baseline. Instead of \"this action got return 12,\" ask \"this action got 12 versus an average of 10, so +2.\" That subtraction slashes the noise without biasing the answer.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why subtracting a baseline lowers variance, say why a state-only baseline adds no bias, and use <code>V(s)</code> as the natural baseline.",
  prev_href="day-02-policy-gradient.html", prev_label="Policy Gradient", next_href="day-04-advantage-gae.html", next_label="Advantage & GAE",
  sections=[
    {"title":"Subtract a reference point","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 2 (the policy-gradient update <code>∇log π · G</code>) and the value function V(s) from Day 1.</div></div>
     <h4>The problem, restated</h4>
     <p>REINFORCE scales each action's push by the raw return G. But G swings wildly from episode to episode, so the gradient is a noisy mess. We want the <em>same</em> direction on average, with far less noise.</p>
     <h4>The fix: subtract a baseline</h4>
     <p>Replace G with <code>(G − b(s))</code>, where <span class="term" data-tip="A reference value subtracted from the return before weighting the gradient. Using a state-dependent baseline b(s) lowers variance without changing the average gradient.">baseline</span> <code>b(s)</code> is a reference return for that state. Now an action is judged by how much it beat the reference, not by its raw score. If every action in a state gets a high G, the baseline is high too, so the differences — the signal — stand out.</p>
     <h4>The natural choice</h4>
     <p>The best simple baseline is the value function <code>V(s)</code>: the average return you expect from that state. Then <code>G − V(s)</code> asks "did this action do better or worse than typical here?" That quantity has a name we meet tomorrow — the advantage.</p>''',
     "gotit":"Got the idea"},
    {"title":"Grading on a curve","body":
     '''<div class="relate">
       <div class="card"><span class="big">📊</span><h5>Raw scores mislead</h5><p>A test score of 80 tells you little until you know the class average. If the average was 60, an 80 is excellent; if it was 95, it's weak. Judging against the average — the curve — reveals the true signal.</p></div>
       <div class="card"><span class="big">🎯</span><h5>Same ranking, less noise</h5><p>Subtracting the average doesn't change <i>who</i> did better than whom — it just removes the shared, uninformative part. The policy still learns the same thing, but from a much steadier signal.</p></div>
     </div>
     <p><strong>In one line:</strong> subtract the expected return of the state so each action is scored by how much it beat average — same learning signal, far less noise.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a class average is fixed after the test. Here the baseline V(s) is itself learned and keeps changing as the policy improves — a lagging or wrong baseline can actually <i>hurt</i>.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"See the variance drop","body":'''<p>Watch raw returns (noisy), then the same returns centred by a baseline. <strong>Click all three.</strong></p>'''},
    {"title":"Why it's unbiased","body":
     '''<h4>Step 1 — the new estimate</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       ∇J(θ) = E[ ∇log π(a|s) · (G − b(s)) ]<br>
       <span class="dim"># subtract a state-only baseline b(s) from the return</span>
     </div></div>
     <h4>Step 2 — why the average gradient doesn't change</h4>
     <p>The key fact: <code>E[ ∇log π(a|s) · b(s) ] = 0</code> whenever <code>b</code> depends only on the state, not the action. The probabilities over actions always sum to 1, so their gradient sums to 0; multiplying by a constant-per-state baseline still sums to 0. So subtracting <code>b(s)</code> removes noise but leaves the <em>expected</em> gradient unchanged — it is <span class="term" data-tip="An estimate whose average equals the true value. Subtracting a state-only baseline keeps the policy gradient unbiased.">unbiased</span>.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       three actions in a state, returns G = [10, 12, 8]<br>
       baseline b(s) = V(s) = 10   <span class="dim"># the average return here</span><br><br>
       G − b = [0, +2, −2]<br>
       <span class="dim"># action 2 beat average (push up), action 3 was below (push down),</span><br>
       <span class="dim"># action 1 was exactly average (no push) — the shared "10" is gone</span>
     </div></div>
     <p>Before, all three pushed <em>up</em> (all returns positive), which is mostly noise. After, only the differences remain — a much cleaner signal.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — a stale or action-dependent baseline.</b> If your learned baseline lags far behind the current policy, the centring is off and variance can stay high. Worse, if the baseline accidentally depends on the <i>action</i> (not just the state), the <code>E[∇log π · b] = 0</code> proof breaks and you inject <b>bias</b> — the policy converges to the wrong thing, quietly. Keep the baseline a function of state only.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — accuracy vs cost.</b> A better baseline (a well-trained value network) cuts variance more, but you must spend compute and data learning it, and a bad value estimate can mislead. A constant baseline (like the batch mean return) is free and helps a little; a learned V(s) helps a lot but adds a second model to train.</div></div>''',
     "gotit":"Got why it's unbiased"},
    {"title":"Centring the returns","body":'''<p>Here is the baseline built up: raw returns, the state baseline, the subtraction, and the tighter gradient. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove the variance drop","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Extend your Day 2 REINFORCE: subtract the batch-mean return as a baseline, then a learned V(s). Log the <em>variance</em> of the gradient (or the jumpiness of the return curve) with and without the baseline.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 20 Day 3 artifact.
Create experiments/foundations/m20_baseline.py: REINFORCE on CartPole with (a) no baseline, (b) batch-mean baseline, (c) a learned V(s) baseline. Plot/print the return curve and gradient variance for each, showing the baseline reduces variance.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m20/day-03-log.md</code>: how much steadier was the curve with the V(s) baseline vs none?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "raw":{"btn":"① raw returns (noisy)","html":'''<span class="prompt">&gt;&gt;&gt;</span> returns = [10, 12, 8]
<span class="dim"># all positive → every action pushed UP</span>
<span class="hl">gradient signal ≈ [+10, +12, +8]</span>   <span class="dim"># mostly shared noise</span>''',
      "take":"<b>①  Raw returns are noisy.</b> All three actions get a big positive push — most of that is the shared baseline, not real signal about which action was better."},
    "sub":{"btn":"② subtract baseline V(s)","html":'''<span class="prompt">&gt;&gt;&gt;</span> b = mean(returns)   <span class="dim"># V(s) ≈ 10</span>
<span class="prompt">&gt;&gt;&gt;</span> centered = [g - b for g in returns]
<span class="hl">centered = [0, +2, -2]</span>''',
      "take":"<b>②  Subtract the state's expected return.</b> Now only the differences remain: action 2 was above average, action 3 below, action 1 exactly average."},
    "var":{"btn":"③ variance drops","html":'''<span class="prompt">&gt;&gt;&gt;</span> var(raw), var(centered)
<span class="hl">4.0   →   2.67</span>   <span class="dim"># lower variance, same mean direction</span>
<span class="dim"># unbiased: E[∇log π · b(s)] = 0</span>''',
      "take":"<b>③  Same signal, less noise.</b> The centred version has lower variance but the same average gradient — learning gets steadier without bias."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='260' y='18' fill='#6B645E'>raw returns — all large and positive</text><rect x='120' y='30' width='60' height='24' fill='#E4F2F7' stroke='#2A7B9B'/><text x='150' y='47' fill='#1F6280'>10</text><rect x='230' y='30' width='60' height='24' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='47' fill='#1F6280'>12</text><rect x='340' y='30' width='60' height='24' fill='#E4F2F7' stroke='#2A7B9B'/><text x='370' y='47' fill='#1F6280'>8</text></g></svg>",
     "note":"<b>Raw returns.</b> Every action gets a big positive weight — the gradient is dominated by the shared level, which carries no information about which action was best."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='180' y='16' width='160' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>baseline b(s) = V(s) = 10</text></g></svg>",
     "note":"<b>The baseline.</b> V(s) is the expected return from this state — the reference we subtract. Best learned by a small value network."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='260' y='18' fill='#6B645E'>after subtracting the baseline</text><rect x='120' y='30' width='60' height='24' fill='#F5F0E8' stroke='#6B645E'/><text x='150' y='47' fill='#5A544E'>0</text><rect x='230' y='30' width='60' height='24' fill='#E8F5EE' stroke='#2D8B55'/><text x='260' y='47' fill='#1a5c38'>+2</text><rect x='340' y='30' width='60' height='24' fill='#FDE8E8' stroke='#C93B3B'/><text x='370' y='47' fill='#C93B3B'>-2</text></g></svg>",
     "note":"<b>Centred advantages.</b> Only the differences survive: above-average actions get pushed up, below-average down, average ones not at all."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>∇log π(a|s) · (G − b(s))</text></g></svg>",
     "note":"<b>The new gradient.</b> Same expected direction as before (unbiased, because b depends only on the state) but far less variance — training steadies."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><polyline points='40,50 90,20 140,55 190,25 240,50 290,22 340,52 390,28 440,48 480,30' fill='none' stroke='#C93B3B' stroke-width='1.5' opacity='.5'/><polyline points='40,45 90,40 140,42 190,35 240,37 290,30 340,32 390,26 440,28 480,22' fill='none' stroke='#2D8B55' stroke-width='2'/><text x='260' y='66' fill='#6B645E'>red = no baseline (jumpy) · green = with baseline (steady)</text></g></svg>",
     "note":"<b>The payoff.</b> With a baseline the learning curve is much smoother, converging faster and more reliably — for the price of learning V(s)."},
  ],
  quiz=[
    {"q":"1. What does subtracting a baseline b(s) do to the policy gradient?","opts":["adds bias to speed things up","lowers variance while keeping the average gradient the same","removes the reward","increases γ"],"ans":1,"fb":"A state-only baseline reduces variance without changing the expected gradient — same learning, steadier."},
    {"q":"2. Why must the baseline depend only on the state, not the action?","opts":["for speed","because a state-only baseline keeps the estimate unbiased (E[∇log π · b] = 0); an action-dependent one injects bias","because actions are secret","to avoid softmax"],"ans":1,"fb":"The proof that the baseline cancels in expectation relies on it being constant across actions in a state. An action-dependent baseline breaks it and biases the update."},
    {"q":"3. Returns are [6, 10, 14], baseline V(s)=10. The centred values are:","opts":["[6, 10, 14]","[-4, 0, +4]","[16, 20, 24]","[0, 0, 0]"],"ans":1,"fb":"Subtract 10 from each: [−4, 0, +4]. Only the deviations from average remain as signal."},
    {"q":"4. Your learned baseline lags far behind the improving policy. Likely effect?","opts":["perfect, low-variance updates","poor centring, so variance stays high (and an action-dependent baseline would add bias)","the reward disappears","γ becomes 1"],"ans":1,"fb":"A stale baseline centres poorly, so much of the variance survives. Keep V(s) tracking the current policy — and never let the baseline depend on the action."},
  ],
  fin={"em":"📉","h3":"Day 3 complete — you tamed the noise!",
       "p":"You now know the baseline trick: replace G with <code>G − b(s)</code> to judge each action against the state's average. Using <code>V(s)</code> as the baseline cuts variance a lot and stays unbiased because a state-only baseline cancels in expectation. Next: <b>Day 4</b> — that <code>G − V(s)</code> quantity has a name (the advantage) and a tunable estimator (GAE)."},
))

# ---------------- Day 4 — Advantage & GAE ----------------
L("day-04-advantage-gae.html", dict(
  qid="m20-d04-gae", title="Module 20 · Day 4 — Advantage & GAE", nav_title="Spiral · M20 Day 4",
  eyebrow="Module 20 · Train · Day 4", h1="Advantage & GAE: The Bias–Variance Knob",
  lead="Yesterday's <code>G − V(s)</code> has a name: the advantage — how much better an action is than average. Today we estimate it two ways (one noisy, one biased) and blend them with a single dial, λ, that trades bias against variance. This is the exact quantity PPO and every RLHF run optimizes.",
  goal="<b>🎯 By the end, you'll be able to:</b> define the advantage <code>A = Q − V</code>, write the one-step TD error <code>δ = r + γV(s') − V(s)</code>, and explain how GAE's λ trades bias for variance.",
  prev_href="day-03-baselines.html", prev_label="Baselines", next_href="day-05-actor-critic.html", next_label="Actor-Critic",
  sections=[
    {"title":"How much better than average?","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 3's baseline <code>G − V(s)</code>, and the value function V(s) from Day 1.</div></div>
     <h4>The advantage</h4>
     <p>The <span class="term" data-tip="A(s,a) = Q(s,a) − V(s): how much better taking action a in state s is than the state's average. Positive = above average, negative = below.">advantage</span> <code>A(s,a) = Q(s,a) − V(s)</code> measures exactly what Day 3 hinted at: how much better this action is than the typical action in this state. Here <code>Q(s,a)</code> is the expected return if you take action a, and <code>V(s)</code> is the state's average.</p>
     <h4>Two ways to estimate it</h4>
     <p>You never know the true advantage, so you estimate it. Two extremes:</p>
     <ul>
       <li><b>Monte Carlo</b> (use the full return): unbiased, but high variance — it waits for the whole episode.</li>
       <li><b>One-step TD</b> (use just <code>r + γV(s')</code>): low variance, but biased — it trusts a possibly-wrong V(s').</li>
     </ul>
     <h4>The knob between them</h4>
     <p><span class="term" data-tip="Generalized Advantage Estimation: blends one-step and multi-step advantage estimates with a factor λ, trading bias (low λ) against variance (high λ).">GAE</span> blends the two with a dial λ. λ near 0 leans on one-step TD (low variance, some bias); λ near 1 leans on the full return (unbiased, high variance). One knob controls the whole trade-off.</p>''',
     "gotit":"Got the advantage"},
    {"title":"Surprise, good or bad","body":
     '''<div class="relate">
       <div class="card"><span class="big">😮</span><h5>Better than expected</h5><p>You expected an okay meal (that's V — the average). It turned out great. The <i>positive surprise</i> is the advantage. Do more of what caused it. A disappointing meal is a negative advantage — do less.</p></div>
       <div class="card"><span class="big">🎚️</span><h5>How far to look</h5><p>Judge the meal on the first bite (fast, but a bad first bite can mislead) or the whole meal (accurate, but you wait to the end). λ is the dial between "trust the first bite" and "wait for the whole thing."</p></div>
     </div>
     <p><strong>In one line:</strong> the advantage is how much better an action turned out than expected, and λ sets how many steps of real reward you use before trusting your value estimate.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: "expected" here is a learned V(s) that is itself imperfect. Leaning too hard on it (low λ) means trusting a guess; leaning on full returns (high λ) means trusting luck.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Compute a TD error","body":'''<p>Watch the one-step TD error, then a multi-step estimate, then the λ blend. <strong>Click all three.</strong></p>'''},
    {"title":"TD error and the λ blend","body":
     '''<h4>Step 1 — the one-step TD error</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       δ<sub>t</sub> = r<sub>t</sub> + γ·V(s<sub>t+1</sub>) − V(s<sub>t</sub>)<br>
       <span class="dim"># actual reward + discounted next-state value, minus what we expected</span>
     </div></div>
     <p>Symbols: <code>δ</code> (delta) = the <span class="term" data-tip="Temporal-difference error: the gap between the value we expected, V(s), and a one-step estimate r + γV(s'). It is a one-step advantage estimate.">TD error</span>. It says "was this step better or worse than V predicted?" A positive δ is a pleasant surprise.</p>
     <h4>Step 2 — GAE stacks TD errors</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       Â<sub>t</sub><sup>GAE(λ)</sup> = δ<sub>t</sub> + (γλ)·δ<sub>t+1</sub> + (γλ)²·δ<sub>t+2</sub> + …<br>
       <span class="dim"># λ=0 → just δ_t (one step). λ=1 → the full Monte-Carlo advantage.</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       r = 1,  V(s) = 5,  V(s') = 6,  γ = 0.9<br><br>
       δ = 1 + 0.9·6 − 5 = 1 + 5.4 − 5 = <span class="hl">1.4</span><br>
       <span class="dim"># the step beat expectations by 1.4 → this action gets a positive push</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — bootstrapping on a wrong value.</b> Low-λ estimates lean on <code>V(s')</code>. Early in training V is garbage, so δ can point the wrong way and the errors <i>compound</i> down the chain — the policy confidently learns from a bad critic. It looks like it is learning (low-variance, smooth updates) while heading somewhere wrong. High λ avoids this but brings back the variance.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — λ is the bias–variance dial.</b> λ→1: unbiased (uses real returns) but high variance and slow. λ→0: low variance but biased (trusts an imperfect V). Practitioners use λ≈0.95 with γ≈0.99 as a sweet spot. There is no free lunch — you are choosing where on the bias–variance curve to sit.</div></div>''',
     "gotit":"Got the λ knob"},
    {"title":"Building the advantage","body":'''<p>Here is GAE built up: Q vs V gives the advantage, δ is the one-step version, and λ stacks δ's into a tunable estimate. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with GAE","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Add GAE to your Day 3 agent. Compute per-step TD errors δ, then the GAE advantage for λ ∈ {0, 0.5, 0.95, 1.0}. Print how the advantage estimates and the return curve change with λ.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 20 Day 4 artifact.
Create experiments/foundations/m20_gae.py: compute GAE advantages on CartPole rollouts for lambda in {0, 0.5, 0.95, 1.0} with gamma=0.99. Print the TD errors and resulting advantage estimates, and show how learning stability/return changes with lambda.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m20/day-04-log.md</code>: which λ gave the best trade-off, and what went wrong at λ=0 and λ=1?</div></div>''',
     "gotit":"Done — on to Day 5"},
  ],
  demos={
    "td":{"btn":"① one-step TD error δ","html":'''<span class="prompt">&gt;&gt;&gt;</span> r, V_s, V_next, gamma = 1, 5, 6, 0.9
<span class="prompt">&gt;&gt;&gt;</span> delta = r + gamma*V_next - V_s
<span class="hl">delta = 1.4</span>   <span class="dim"># step beat expectation by 1.4</span>''',
      "take":"<b>①  The TD error is a one-step advantage.</b> δ = reward + discounted next value − current value. Positive means the step went better than V(s) predicted."},
    "mc":{"btn":"② full-return estimate","html":'''<span class="prompt">&gt;&gt;&gt;</span> A_mc = full_return - V_s
<span class="hl">A_mc = 2.1</span>   <span class="dim"># unbiased, but noisy (uses whole episode)</span>''',
      "take":"<b>②  The Monte-Carlo advantage.</b> Use the real full return instead of a one-step guess: unbiased, but it waits for the episode to end and is noisy."},
    "gae":{"btn":"③ blend with λ","html":'''<span class="prompt">&gt;&gt;&gt;</span> A_gae = gae(deltas, gamma=0.9, lam=0.95)
<span class="hl">A_gae = 1.7</span>   <span class="dim"># between one-step and full-return</span>
<span class="dim"># lam=0 → δ only; lam=1 → full return</span>''',
      "take":"<b>③  GAE blends them.</b> λ dials between the low-variance one-step δ and the unbiased full return. λ≈0.95 is the usual sweet spot."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='130' y='16' width='260' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>A(s,a) = Q(s,a) − V(s)</text></g></svg>",
     "note":"<b>The advantage.</b> How much better an action is than the state's average. Positive advantage → make the action more likely."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='110' y='16' width='300' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>δₜ = rₜ + γV(sₜ₊₁) − V(sₜ)</text></g></svg>",
     "note":"<b>The one-step TD error.</b> A cheap, low-variance advantage estimate — but it trusts V(s'), which may be wrong early on (bias)."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>Â = δₜ + (γλ)δₜ₊₁ + (γλ)²δₜ₊₂ + …</text></g></svg>",
     "note":"<b>GAE stacks TD errors.</b> Each further step is discounted by γλ. More terms = closer to the full return; fewer = closer to one-step."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='90' y='30' fill='#5E5191'>λ=0</text><text x='90' y='46' fill='#6B645E'>low var, biased</text><line x1='150' y1='40' x2='370' y2='40' stroke='#6B645E' stroke-width='2'/><circle cx='340' cy='40' r='5' fill='#2D8B55'/><text x='430' y='30' fill='#5E5191'>λ=1</text><text x='430' y='46' fill='#6B645E'>unbiased, noisy</text></g></svg>",
     "note":"<b>λ is the bias–variance dial.</b> Slide toward 0 for stability (biased); toward 1 for correctness (noisy). λ≈0.95 (green) is the common sweet spot."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>∇log π(a|s) · Â(s,a)</text></g></svg>",
     "note":"<b>The final update.</b> Weight the log-prob gradient by the advantage estimate. This is precisely what actor-critic (tomorrow) and PPO optimize."},
  ],
  quiz=[
    {"q":"1. The advantage A(s,a) measures:","opts":["the raw reward","how much better an action is than the state's average (Q − V)","the discount factor","the number of actions"],"ans":1,"fb":"A = Q − V: the extra return from this action versus the typical action in the state. Positive = above average."},
    {"q":"2. With r=2, V(s)=3, V(s')=4, γ=1, the TD error δ is:","opts":["3","2","9","-1"],"ans":0,"fb":"δ = r + γV(s') − V(s) = 2 + 1·4 − 3 = 3. A positive δ means the step beat the value estimate."},
    {"q":"3. What does GAE's λ control?","opts":["the learning rate","the trade-off between bias (low λ) and variance (high λ)","the number of actions","the reward scale"],"ans":1,"fb":"λ blends one-step TD (low variance, biased) and full returns (unbiased, high variance). It is the bias–variance knob."},
    {"q":"4. Early in training with λ near 0, updates are smooth but the policy drifts wrong. Why?","opts":["λ must be negative","it is bootstrapping on an untrained V(s'); the bias compounds down the chain","the reward is too large","γ is missing"],"ans":1,"fb":"Low λ leans on V(s'), which is inaccurate early on, so biased errors compound — smooth but misdirected. Raising λ trades that bias back for variance."},
  ],
  fin={"em":"🎚️","h3":"Day 4 complete — you hold the bias–variance knob!",
       "p":"You can now define the advantage <code>A = Q − V</code>, compute the one-step TD error <code>δ = r + γV(s') − V(s)</code>, and use GAE's λ to trade bias against variance. This advantage is the exact signal the next lesson's actor-critic — and later PPO/RLHF — pushes on. Next: <b>Day 5</b> — actor and critic, together."},
))

# ---------------- Day 5 — Actor-Critic ----------------
L("day-05-actor-critic.html", dict(
  qid="m20-d05-ac", title="Module 20 · Day 5 — Actor-Critic", nav_title="Spiral · M20 Day 5",
  eyebrow="Module 20 · Train · Day 5", h1="Actor-Critic: A Performer and a Coach",
  lead="We have a policy that learns (the actor) and a value function that judges (the critic). Actor-critic runs them together: the critic estimates V(s) and hands the actor an advantage each step, and the actor updates immediately — no waiting for the episode to end. This is the architecture under A2C, PPO, and RLHF.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain the actor and critic roles, write both loss functions, and say why the coupled system can be unstable (the deadly triad).",
  prev_href="day-04-advantage-gae.html", prev_label="Advantage & GAE", next_href="day-06-on-off-policy.html", next_label="On- vs Off-Policy",
  sections=[
    {"title":"Two networks, one loop","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 2 (the actor's policy-gradient update) and Day 4 (the advantage from a value function).</div></div>
     <h4>The two roles</h4>
     <p>The <span class="term" data-tip="The policy network in actor-critic. It chooses actions and is updated to raise the probability of high-advantage actions.">actor</span> is the policy — it picks actions. The <span class="term" data-tip="The value network in actor-critic. It estimates V(s) and provides the advantage the actor learns from.">critic</span> is a value network — it estimates V(s) and tells the actor how good each step was, as an advantage.</p>
     <h4>Why put them together</h4>
     <p>REINFORCE waits until the episode ends to compute returns. Actor-critic uses the critic's V(s') to estimate the future <em>immediately</em>, so it can update every single step (online). The critic supplies the low-variance advantage from Day 4; the actor turns it into a policy update.</p>
     <h4>The loop</h4>
     <p>Each step: the actor acts, the environment returns a reward, the critic computes the TD error / advantage, the actor nudges its action probabilities by that advantage, and the critic nudges V(s) toward the observed target. Both improve together.</p>''',
     "gotit":"Got the two roles"},
    {"title":"Performer and coach","body":
     '''<div class="relate">
       <div class="card"><span class="big">🎭</span><h5>The actor performs</h5><p>A performer tries moves on stage. They can't judge their own performance well in the moment — they need feedback.</p></div>
       <div class="card"><span class="big">🧑‍🏫</span><h5>The critic coaches</h5><p>A coach watches and says, after each move, "that was better than usual" (positive advantage) or "worse" (negative). The performer adjusts immediately, not at the end of the show.</p></div>
     </div>
     <p><strong>In one line:</strong> the actor acts, the critic scores each step as an advantage, and the actor adjusts right away — a tight feedback loop instead of end-of-episode grading.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a real coach already knows what "good" is. Here the critic is <i>also learning</i> — a beginner coaching a beginner. If the critic is wrong, it misleads the actor, which changes the data, which further confuses the critic.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"One actor-critic step","body":'''<p>Watch the critic estimate V, the advantage get computed, and both networks update. <strong>Click all three.</strong></p>'''},
    {"title":"Two losses, updated together","body":
     '''<h4>Step 1 — the actor loss</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       actor_loss = − log π(a|s) · Â(s,a)<br>
       <span class="dim"># raise the probability of actions with positive advantage</span>
     </div></div>
     <p>Same policy-gradient idea as Day 2, but weighted by the critic's advantage Â instead of the raw return — much lower variance.</p>
     <h4>Step 2 — the critic loss</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       target = r + γ·V(s')          <span class="dim"># the bootstrap target</span><br>
       critic_loss = ( V(s) − target )²   <span class="dim"># plain MSE regression</span>
     </div></div>
     <p>The critic just does regression: make <code>V(s)</code> match the one-step target <code>r + γV(s')</code>. As it gets accurate, the advantages get accurate, and the actor learns well.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       r=1, V(s)=5, V(s')=6, γ=0.9<br>
       advantage Â = δ = 1 + 0.9·6 − 5 = <span class="hl">1.4</span>  → actor pushes this action UP<br>
       critic target = 1 + 0.9·6 = 6.4;  V(s)=5 → nudge V(s) up toward 6.4
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the deadly triad.</b> Combine (1) bootstrapping (the critic uses its own V(s') in the target), (2) function approximation (V is a neural net), and (3) shifting data (off-policy or a changing policy) and value estimates can <b>diverge</b> — blow up to nonsense — with no error thrown. The actor then chases a broken critic. Mitigations: target networks, careful learning rates, gradient clipping, keeping the critic ahead of the actor.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — actor-critic vs REINFORCE.</b> Actor-critic gives low-variance, online, per-step updates (fast, data-efficient) — but the critic introduces <b>bias</b> and a second network to tune, plus the instability above. REINFORCE is unbiased and dead simple, but high-variance and only updates once per episode. Most modern RL (A2C, PPO, RLHF) chooses actor-critic and works hard to stabilize it.</div></div>''',
     "gotit":"Got both losses"},
    {"title":"The actor-critic loop","body":'''<p>Here it is built up: the shared state, the actor's action, the critic's value, the advantage feedback, and the two updates. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with A2C","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Build A2C on CartPole: an actor head and a critic head (can share a trunk). Each step, compute the advantage with the critic, update the actor by <code>-log π · Â</code> and the critic by MSE to <code>r + γV(s')</code>. Print both losses and the return.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 20 Day 5 artifact.
Create experiments/foundations/m20_a2c.py: A2C on CartPole with a shared-trunk actor and critic. Each step compute advantage = r + gamma*V(s') - V(s); actor_loss = -(log_prob * advantage.detach()); critic_loss = advantage^2. Print actor_loss, critic_loss, and average return; note it converges faster and steadier than REINFORCE.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m20/day-05-log.md</code>: did A2C converge faster and steadier than your REINFORCE run?</div></div>''',
     "gotit":"Done — on to Day 6"},
  ],
  demos={
    "critic":{"btn":"① critic estimates V","html":'''<span class="prompt">&gt;&gt;&gt;</span> V_s = critic(s); V_next = critic(s2)
<span class="hl">V(s) = 5.0   V(s') = 6.0</span>
<span class="dim"># the critic's guess of expected return</span>''',
      "take":"<b>①  The critic estimates value.</b> It predicts V(s) and V(s') — the expected return from each state. These feed the advantage."},
    "adv":{"btn":"② compute the advantage","html":'''<span class="prompt">&gt;&gt;&gt;</span> A = r + gamma*V_next - V_s
<span class="dim"># = 1 + 0.9*6 - 5</span>
<span class="hl">A = 1.4</span>   <span class="dim"># step beat the critic's expectation</span>''',
      "take":"<b>②  The advantage.</b> The TD error tells the actor this step was 1.4 better than expected — so push this action up."},
    "upd":{"btn":"③ update actor + critic","html":'''<span class="prompt">&gt;&gt;&gt;</span> actor_loss  = -(log_prob * A)
<span class="prompt">&gt;&gt;&gt;</span> critic_loss = (V_s - (r + gamma*V_next))**2
<span class="hl">both step every timestep (online)</span>''',
      "take":"<b>③  Both learn each step.</b> The actor raises the action's probability by the advantage; the critic regresses V toward the target. No waiting for the episode."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='200' y='16' width='120' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>state s</text></g></svg>",
     "note":"<b>Shared state.</b> Both networks read the same state — often through a shared trunk with two heads."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='30' width='80' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='80' y='48' fill='#1F6280'>state</text><polygon points='140,30 250,42 250,46 140,58' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='195' y='47' fill='#5E5191'>actor π</text><rect x='270' y='30' width='80' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='310' y='48' fill='#9A7208'>action</text></g></svg>",
     "note":"<b>The actor acts.</b> The policy chooses an action and the environment returns a reward and next state."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='30' width='80' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='80' y='48' fill='#1F6280'>state</text><polygon points='140,30 250,42 250,46 140,58' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='195' y='47' fill='#1a5c38'>critic V</text><rect x='270' y='30' width='120' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='330' y='48' fill='#1a5c38'>V(s), V(s')</text></g></svg>",
     "note":"<b>The critic values it.</b> It estimates V(s) and V(s') so we can form the advantage without waiting for the episode to finish."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='35' fill='#9A7208'>Â = r + γV(s') − V(s)</text></g></svg>",
     "note":"<b>The advantage feedback.</b> The critic's TD error becomes the actor's signal: positive → make the action more likely."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='30' width='190' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='135' y='48' fill='#5E5191'>actor: −log π · Â</text><rect x='290' y='30' width='190' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='385' y='48' fill='#1a5c38'>critic: (V − target)²</text></g></svg>",
     "note":"<b>Both update — every step.</b> Actor raises good-action probability; critic regresses V toward r+γV(s'). Watch for the deadly triad and stabilize it."},
  ],
  quiz=[
    {"q":"1. In actor-critic, what is the critic's job?","opts":["choose actions","estimate V(s) and provide the advantage the actor learns from","store the replay buffer","set the reward"],"ans":1,"fb":"The critic is the value network: it estimates V(s), which yields the advantage that the actor (the policy) uses to update."},
    {"q":"2. How is actor-critic different from REINFORCE in timing?","opts":["it waits two episodes","it can update every step (online) using the critic's V(s'), not just at episode end","it never updates","it ignores rewards"],"ans":1,"fb":"By bootstrapping with V(s'), actor-critic estimates the future immediately, enabling per-step online updates instead of end-of-episode ones."},
    {"q":"3. The critic's loss is:","opts":["cross-entropy over actions","MSE between V(s) and the target r + γV(s')","the negative advantage","the discount factor"],"ans":1,"fb":"The critic does regression: minimize (V(s) − (r + γV(s')))². Accurate V → accurate advantages."},
    {"q":"4. Your value estimates blow up to huge numbers mid-training. This is:","opts":["expected and fine","the deadly triad — bootstrapping + function approximation + shifting data can diverge","a reward bug only","impossible"],"ans":1,"fb":"The deadly triad: combining bootstrapping, a neural-net value function, and shifting data can cause divergence. Target networks, clipping, and careful LRs help."},
  ],
  fin={"em":"🎭","h3":"Day 5 complete — actor and critic in sync!",
       "p":"You can now run actor-critic: the critic estimates V(s) and hands the actor a low-variance advantage each step, the actor updates by <code>-log π · Â</code>, and the critic regresses toward <code>r + γV(s')</code> — online, every step. You also met the deadly triad, the reason this needs stabilizing. Next: <b>Day 6</b> — on- vs off-policy, the last framing you need before RLHF."},
))

# ---------------- Day 6 — On- vs Off-Policy ----------------
L("day-06-on-off-policy.html", dict(
  qid="m20-d06-onoff", title="Module 20 · Day 6 — On- vs Off-Policy", nav_title="Spiral · M20 Day 6",
  eyebrow="Module 20 · Train · Day 6", h1="On-Policy vs Off-Policy",
  lead="One last distinction runs through all of RL and decides how PPO and RLHF are built: are you learning from data your current policy just generated (on-policy), or from older/other data you saved (off-policy)? It's the difference between practising your own swing and studying tapes — and it sets the sample-efficiency vs stability trade-off.",
  goal="<b>🎯 By the end, you'll be able to:</b> define on-policy and off-policy, explain the importance-sampling correction, and say why PPO is \"almost on-policy\" and why RLHF is expensive because of it.",
  prev_href="day-05-actor-critic.html", prev_label="Actor-Critic", next_href="review.html", next_label="Review Gate",
  sections=[
    {"title":"Whose data are you learning from?","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 2–5 — policy gradients, advantages, and actor-critic.</div></div>
     <h4>The two families</h4>
     <p><span class="term" data-tip="Learning only from data generated by the current policy. REINFORCE, A2C, and PPO are on-policy. Old data must be discarded.">On-policy</span> methods learn only from data the <em>current</em> policy just produced. Once you update the policy, that data is stale and thrown away. <span class="term" data-tip="Learning from data generated by a different or older policy — e.g. from a replay buffer. Q-learning and DQN are off-policy.">Off-policy</span> methods can learn from data generated by an older or entirely different policy, so you can store and reuse it.</p>
     <h4>Why it matters</h4>
     <p>Off-policy is <b>sample-efficient</b>: you reuse each experience many times (a replay buffer). On-policy is <b>sample-hungry</b>: fresh data every update. But off-policy needs a correction, because the data came from a different distribution than the one you are now optimizing.</p>
     <h4>Where the methods land</h4>
     <p>REINFORCE, A2C, PPO → on-policy. Q-learning, DQN, SAC → off-policy. This single choice shapes the whole training system: its cost, its stability, and its infrastructure.</p>''',
     "gotit":"Got the two families"},
    {"title":"Your swing vs the tapes","body":
     '''<div class="relate">
       <div class="card"><span class="big">🏌️</span><h5>On-policy: your own swing</h5><p>You practise, watch <i>your own</i> current swing, and adjust. Every rep is fresh and exactly relevant — but you can only learn from what you just did, so it's slow.</p></div>
       <div class="card"><span class="big">📼</span><h5>Off-policy: study the tapes</h5><p>You watch a huge library of old games — yours and others'. Tons of data to learn from cheaply, but those players had different styles, so you must adjust for "they'd have played this differently than I would now."</p></div>
     </div>
     <p><strong>In one line:</strong> on-policy learns only from your current behaviour (fresh but slow), off-policy reuses saved behaviour (cheap but needs correcting for the mismatch).</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: watching tapes is harmless, but in RL reusing far-off-policy data without correction actively breaks the math — the gradient points the wrong way.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Reuse vs discard","body":'''<p>Watch on-policy discard old data, off-policy reuse a buffer, and the importance-sampling fix. <strong>Click all three.</strong></p>'''},
    {"title":"Importance sampling and PPO","body":
     '''<h4>Step 1 — the mismatch</h4>
     <p>If data came from an old policy <code>π<sub>old</sub></code> but you want to improve the new <code>π<sub>new</sub></code>, you cannot just reuse it — the actions were sampled with the wrong probabilities. You must reweight.</p>
     <h4>Step 2 — the importance-sampling ratio</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       ratio  r(θ) = π<sub>new</sub>(a|s) / π<sub>old</sub>(a|s)<br>
       estimate = E[ r(θ) · Â ]   <span class="dim"># reweight old data by how likely π_new would take it</span>
     </div></div>
     <p>The <span class="term" data-tip="Reweighting samples from one distribution to estimate an expectation under another, using the ratio of probabilities. Lets off-policy data be reused — but the ratio's variance explodes when the policies differ a lot.">importance-sampling</span> ratio corrects for the mismatch. But if <code>π<sub>new</sub></code> and <code>π<sub>old</sub></code> differ a lot, the ratio can be huge or tiny — variance explodes.</p>
     <h4>Step 3 — why PPO is \"almost on-policy\"</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       PPO reuses each batch for a few epochs (mildly off-policy)…<br>
       …but <b>clips</b> the ratio to [1−ε, 1+ε] (ε≈0.2) so π_new can't stray far from π_old.<br>
       <span class="dim"># small reuse for efficiency, clipping for stability</span>
     </div></div>
     <p>PPO takes a middle path: reuse a batch a few times (efficiency), but clip the ratio so the policy never moves too far (stability). That is why RLHF, built on PPO, must keep generating fresh samples from the model being trained.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — off-policy distribution shift.</b> Reuse data from a policy that has since changed a lot and the importance ratios blow up: a few samples dominate the gradient, or it silently points the wrong way. Training looks like it is running but learns garbage. This is why on-policy methods refuse to reuse stale data at all, and why PPO clips.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — sample efficiency vs stability.</b> Off-policy (replay buffers) squeezes many updates from each sample — vital when environment steps are expensive — but is harder to stabilize. On-policy is simple and stable but throws data away, so it needs far more environment interaction. For RLHF this trade-off is money: on-policy PPO must run the LLM to generate fresh rollouts every update, which is the dominant cost.</div></div>''',
     "gotit":"Got on- vs off-policy"},
    {"title":"Data source → reuse or discard","body":'''<p>Here it is built up: the data source, the policy-match check, and the two paths. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove the difference","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Compare on-policy A2C (discard data each update) with a simple off-policy variant using a replay buffer and importance-sampling weights. Track sample efficiency (return vs environment steps) and note when the IS ratio blows up.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 20 Day 6 artifact.
Create experiments/foundations/m20_on_off_policy.py on CartPole: (a) on-policy A2C discarding data each update; (b) an off-policy variant reusing a replay buffer with importance-sampling ratios. Plot return vs environment steps to show off-policy is more sample-efficient, and print when IS ratios blow up.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m20/day-06-log.md</code>: which was more sample-efficient, and did the off-policy IS ratios ever explode?</div></div>''',
     "gotit":"Done — on to the review gate"},
  ],
  demos={
    "on":{"btn":"① on-policy: discard","html":'''<span class="prompt">&gt;&gt;&gt;</span> data = rollout(policy)   <span class="dim"># fresh from current policy</span>
<span class="prompt">&gt;&gt;&gt;</span> policy.update(data)
<span class="prompt">&gt;&gt;&gt;</span> data = None   <span class="hl"># thrown away — now stale</span>''',
      "take":"<b>①  On-policy discards.</b> After one update the data no longer matches the policy, so it is thrown away. Fresh, relevant — but wasteful."},
    "off":{"btn":"② off-policy: reuse buffer","html":'''<span class="prompt">&gt;&gt;&gt;</span> buffer.add(rollout(old_policy))
<span class="prompt">&gt;&gt;&gt;</span> for _ in range(many):
<span class="prompt">...</span>     batch = buffer.sample()   <span class="hl"># reuse old data</span>
<span class="prompt">...</span>     policy.update(batch)''',
      "take":"<b>②  Off-policy reuses.</b> A replay buffer lets each experience train the policy many times — sample-efficient, but the data is from an older policy."},
    "is":{"btn":"③ importance-sampling fix","html":'''<span class="prompt">&gt;&gt;&gt;</span> ratio = pi_new(a|s) / pi_old(a|s)
<span class="hl">ratio = 1.8</span>   <span class="dim"># reweight the stale sample</span>
<span class="prompt">&gt;&gt;&gt;</span> estimate = ratio * advantage
<span class="dim"># if policies differ a lot, ratio explodes → high variance</span>''',
      "take":"<b>③  Correct the mismatch.</b> The ratio π_new/π_old reweights old data. PPO clips this ratio to keep π_new close to π_old and stay stable."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>data from some policy</text></g></svg>",
     "note":"<b>Start with the data.</b> Some experience — states, actions, rewards. The key question: which policy generated it?"},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='35' fill='#9A7208'>is it from the CURRENT policy?</text></g></svg>",
     "note":"<b>The deciding question.</b> If the data came from the policy you're updating right now, it's on-policy; if from an older/other policy, it's off-policy."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='24' width='190' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='135' y='43' fill='#1a5c38'>on-policy → update, discard</text><rect x='290' y='24' width='190' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='385' y='43' fill='#1F6280'>off-policy → buffer + reuse</text></g></svg>",
     "note":"<b>Two paths.</b> On-policy uses the data once then discards it. Off-policy stores it in a buffer and reuses it many times — but must correct the mismatch."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='35' fill='#5E5191'>ratio = π_new(a|s) / π_old(a|s)</text></g></svg>",
     "note":"<b>Importance sampling.</b> Off-policy reuse needs this ratio to reweight old samples — but the ratio's variance explodes if the policies diverge."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>PPO: reuse a few epochs + clip ratio to 1±ε</text></g></svg>",
     "note":"<b>PPO's middle path.</b> Reuse each batch a little (efficiency) but clip the ratio so the policy can't stray far (stability). This is the engine under RLHF."},
  ],
  quiz=[
    {"q":"1. What defines an on-policy method?","opts":["it uses a replay buffer","it learns only from data generated by the current policy, discarding stale data","it never updates the policy","it ignores the advantage"],"ans":1,"fb":"On-policy (REINFORCE, A2C, PPO) learns only from fresh data produced by the current policy; once updated, old data is stale."},
    {"q":"2. Why is off-policy learning more sample-efficient?","opts":["it uses larger rewards","it reuses each stored experience many times from a replay buffer","it has no critic","it discards data faster"],"ans":1,"fb":"A replay buffer lets each sample train the policy repeatedly, so fewer environment interactions are needed — at the cost of a correction and harder stability."},
    {"q":"3. The importance-sampling ratio π_new/π_old is dangerous when:","opts":["it equals 1","the new and old policies differ a lot, so the ratio explodes and variance blows up","the reward is positive","γ is 0.9"],"ans":1,"fb":"Large policy divergence makes the ratio huge or tiny, so a few samples dominate the gradient — high variance. PPO clips the ratio to prevent this."},
    {"q":"4. Why is RLHF (PPO-based) expensive to run?","opts":["it uses a huge replay buffer","it is on-policy-ish, so it must keep generating fresh rollouts from the LLM being trained","it never uses a GPU","it has no reward model"],"ans":1,"fb":"PPO is nearly on-policy, so RLHF must repeatedly sample fresh generations from the current model — that generation is the dominant cost."},
  ],
  fin={"em":"🔁","h3":"Day 6 complete — you've got the full RL frame!",
       "p":"You can now place any RL method: on-policy (learn from current data, discard it — REINFORCE/A2C/PPO) vs off-policy (reuse a buffer with importance sampling — Q-learning/DQN). You saw why PPO clips to stay near π_old and why RLHF's on-policy generation is its dominant cost. That completes RL Foundations — you're ready for the review gate, then post-training."},
))

# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m20-review", title="Module 20 · Review Gate", nav_title="Spiral · M20 Review",
  eyebrow="Module 20 · The Gate — You Understand RL", h1="Module 20 — Review Gate",
  lead="Checkpoint. The gate question: <b>can I frame a problem as an MDP, derive the policy gradient, tame its variance, and place any method as on- or off-policy?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. Everything in post-training (M21) — SFT, reward models, PPO, DPO, GRPO — assumes this RL frame is solid.",
  prev_href="day-06-on-off-policy.html", prev_label="Day 6 · On- vs Off-Policy",
  checks=[
    ["Day 1","I can name the four MDP pieces (state, action, reward, transition), write the discounted return <code>G = r₀ + γr₁ + …</code>, and say what V(s) is."],
    ["Day 2","I can write the policy gradient <code>∇J = E[∇log π(a|s)·G]</code> and explain why REINFORCE is high-variance."],
    ["Day 3","I can explain why subtracting a state baseline <code>b(s)</code> lowers variance without adding bias."],
    ["Day 4","I can define the advantage <code>A = Q − V</code>, write the TD error <code>δ = r + γV(s') − V(s)</code>, and say what GAE's λ trades off."],
    ["Day 5","I can describe actor-critic (both losses) and name the deadly triad as its instability risk."],
    ["Day 6","I can define on- vs off-policy, explain the importance-sampling ratio, and say why PPO clips."],
  ],
  quiz=[
    {"q":"1. The agent maximizes:","opts":["the next reward only","the discounted return over the whole episode","the number of states","the policy entropy"],"ans":1,"fb":"Return = sum of discounted future rewards. It is why the agent can trade a small reward now for a larger one later (Day 1)."},
    {"q":"2. REINFORCE's core weakness is:","opts":["it is biased","high variance — the full return is a noisy weight","it needs a replay buffer","it can't use γ"],"ans":1,"fb":"Weighting every action by the whole return is unbiased but very noisy — the motivation for baselines and advantages (Day 2–4)."},
    {"q":"3. With r=1, V(s)=4, V(s')=5, γ=1, the TD error δ is:","opts":["1","2","0","5"],"ans":1,"fb":"δ = r + γV(s') − V(s) = 1 + 5 − 4 = 2 (Day 4)."},
    {"q":"4. In actor-critic, the critic is trained with:","opts":["policy gradient","MSE regression toward r + γV(s')","the importance ratio","no loss"],"ans":1,"fb":"The critic regresses V(s) toward the bootstrap target r + γV(s') (Day 5)."},
    {"q":"5. PPO clips the ratio π_new/π_old to:","opts":["make rewards larger","keep the updated policy close to the data-generating policy, for stability","remove the critic","switch to off-policy"],"ans":1,"fb":"Clipping stops the policy from moving too far from π_old, controlling the importance-sampling variance — nearly-on-policy but data-reusing (Day 6)."},
  ],
  verdict_pass="you can frame decisions as MDPs, derive and stabilize the policy gradient, use advantages and actor-critic, and place any method on the on/off-policy map. You're ready for post-training (M21): SFT, reward models, PPO, and DPO.",
  verdict_fail="note which day tripped you up and re-run just that lesson. The chain — <code>MDP</code> → <code>policy gradient</code> → <code>baseline</code> → <code>advantage/GAE</code> → <code>actor-critic</code> → <code>on/off-policy</code> — should feel familiar before RLHF.",
  complete_label="Mark Module 20 complete",
  fin={"em":"🏆","h3":"Module 20 — passed!",
       "p":"You built the whole RL frame from scratch: MDPs and returns, the policy gradient, variance reduction with baselines and advantages, actor-critic, and the on/off-policy distinction. This is the exact foundation PPO, DPO, and GRPO stand on in the next module."},
  score_target=4,
))
print("M20 built: 6 lessons + review")

