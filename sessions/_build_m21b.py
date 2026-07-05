#!/usr/bin/env python3
"""Builder for M21b — Reasoning-RL & Verifiable Rewards (GRPO/RLVR).
Convert 10-rl/rlhf/grpo + authored. 5 lessons + review.
Run: python3 sessions/_build_m21b.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m21b")
os.makedirs(OUT, exist_ok=True)
def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — GRPO ----------------
L("day-01-grpo.html", dict(
  qid="m21b-d01-grpo", title="Module 21b · Day 1 — GRPO", nav_title="Spiral · M21b Day 1",
  eyebrow="Module 21b · Train · Day 1", h1="GRPO: Reinforcement Learning Without a Critic",
  lead="PPO (Module 21a) needs a value network — the critic — to judge each step. That critic is a whole extra model to train and store. GRPO throws it away with a simple trick: for each prompt, sample a <em>group</em> of answers, and judge each answer against the group's own average. The group is the baseline. This is the algorithm behind the 2026 reasoning models.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why GRPO removes the value critic, compute a group-relative advantage <code>(r − mean)/std</code>, and say what GRPO trades for that simplicity.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-rlvr.html", next_label="RLVR",
  sections=[
    {"title":"Drop the critic","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> M20 (advantage, baselines, PPO clip) and M21a Day 6 (PPO for LLMs, the KL leash). GRPO is a small change to that.</div></div>
     <h4>The cost of a critic</h4>
     <p>In PPO the advantage came from a <span class="term" data-tip="The value network in actor-critic/PPO that estimates V(s). It is a second large model to train and keep in memory.">critic</span> (the value head estimating <code>V</code>). For a large language model that critic is expensive — another big network, more memory, another thing that can go wrong.</p>
     <h4>The GRPO idea</h4>
     <p><span class="term" data-tip="Group Relative Policy Optimization: for each prompt, sample a group of answers and use the group's mean reward as the baseline, removing the need for a value critic.">GRPO (Group Relative Policy Optimization)</span> removes the critic. For one prompt it samples a whole <em>group</em> of answers (say 8). It scores them all, then judges each answer by how far above or below the <em>group average</em> it scored. The group's mean reward <em>is</em> the baseline.</p>
     <h4>Why this works</h4>
     <p>Module 20 taught that any state-only baseline reduces variance without bias. The group mean is exactly such a baseline — computed on the fly from samples, no separate model needed. So GRPO keeps the variance-reduction benefit of a baseline while deleting the critic. That simplicity is why it scales to huge reasoning runs.</p>''',
     "gotit":"Got the idea"},
    {"title":"Grading on the group's curve","body":
     '''<div class="relate">
       <div class="card"><span class="big">📝</span><h5>One exam, many students</h5><p>A teacher gives the same hard problem to 8 students and grades each one <i>relative to the group's average</i>. No need for an external "expected score" model — the group itself sets the bar. Above average? Do more of that. Below? Less.</p></div>
       <div class="card"><span class="big">📐</span><h5>Normalize by the spread</h5><p>If everyone scored close together, a small lead means a lot; if scores were all over the place, the same lead means little. Dividing by the spread (standard deviation) makes the signal fair across easy and hard prompts.</p></div>
     </div>
     <p><strong>In one line:</strong> sample many answers to the same prompt, and reward each by how far above the group's average it scored — the group replaces the critic.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a teacher can grade one student alone. GRPO <i>must</i> generate a whole group per prompt to have an average at all — no group, no baseline.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Compute group advantages","body":'''<p>Watch a group of answers sampled, scored, and turned into group-relative advantages. <strong>Click all three.</strong></p>'''},
    {"title":"The group-relative advantage","body":
     '''<h4>Step 1 — sample a group and score it</h4>
     <p>For a prompt <code>x</code>, sample <code>G</code> answers from the current policy and get a reward <code>r_i</code> for each (from a reward model, or tomorrow's verifier).</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       Â_i = ( r_i − mean(r) ) / ( std(r) + ε )<br>
       <span class="dim"># each answer's advantage = how many std-devs above/below the group mean</span><br>
       then: PPO-style clipped update using Â_i, plus the usual KL penalty to the reference
     </div></div>
     <p>Symbols: <code>r_i</code> = reward of answer i. <code>mean(r)</code>, <code>std(r)</code> = mean and spread over the group. <code>ε</code> = a tiny number so we never divide by zero. Every answer in the group gets the <em>same</em> advantage value applied to all its tokens.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       group of 4 answers, rewards r = [1, 0, 1, 0]<br><br>
       mean = 0.5,  std = 0.5<br>
       Â = (r − 0.5) / 0.5 = [ +1, −1, +1, −1 ]<br>
       <span class="dim"># the two correct answers are pushed up, the two wrong ones down</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — a zero-variance group.</b> If every answer in the group gets the <i>same</i> reward (all correct, or all wrong), then <code>std = 0</code> and every advantage is 0 — the prompt contributes <b>no learning signal at all</b>. On easy prompts (all right) or very hard ones (all wrong), GRPO silently learns nothing from them, wasting the rollouts. You catch it by tracking the fraction of prompts with all-equal rewards and curating prompt difficulty.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — no critic vs many samples.</b> GRPO saves a whole value network (memory, training complexity, and a source of instability). The price: you must generate <code>G</code> answers per prompt (say 8–64), so it spends far more <i>inference</i> per prompt. It trades the critic's cost for generation cost — a good deal when generation is cheap relative to holding a second big model.</div></div>''',
     "gotit":"Got the group advantage"},
    {"title":"GRPO, built up","body":'''<p>Here it is assembled: the prompt, the group of sampled answers, their rewards, the group mean baseline, the normalized advantages, and the update. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with GRPO advantages","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21b_grpo.py</code>. For several prompts, sample a group of answers, assign toy rewards, and compute group-relative advantages <code>(r − mean)/std</code>. Show what happens to the advantages when a group has all-equal rewards (std = 0).</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21b Day 1 artifact.
Create experiments/foundations/m21b_grpo.py: implement GRPO advantage computation — sample G answers per prompt (toy), reward them, compute (r - mean)/(std + eps) per answer. Show a normal group and a zero-variance group (all rewards equal) where every advantage collapses to 0. Print both.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21b/day-01-log.md</code>: what happened to the advantages when the whole group scored the same?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "sample":{"btn":"① sample a group","html":'''<span class="prompt">&gt;&gt;&gt;</span> answers = policy.generate(prompt, n=4)
<span class="dim"># G=4 answers to the SAME prompt</span>
<span class="hl">["...42", "...41", "...42", "...50"]</span>''',
      "take":"<b>①  Sample a group per prompt.</b> Instead of one answer, generate several. The group will supply its own baseline — no critic needed."},
    "score":{"btn":"② score each","html":'''<span class="prompt">&gt;&gt;&gt;</span> r = [reward(a) for a in answers]
<span class="hl">r = [1, 0, 1, 0]</span>   <span class="dim"># two correct, two wrong</span>
<span class="prompt">&gt;&gt;&gt;</span> mean(r), std(r)
<span class="hl">0.5, 0.5</span>''',
      "take":"<b>②  Score the group.</b> Reward each answer, then compute the group mean and spread. The mean is the baseline."},
    "adv":{"btn":"③ group-relative advantage","html":'''<span class="prompt">&gt;&gt;&gt;</span> A = [(x - 0.5)/0.5 for x in r]
<span class="hl">A = [+1, -1, +1, -1]</span>
<span class="dim"># correct answers up, wrong ones down — no value network used</span>''',
      "take":"<b>③  Judge against the group.</b> (r − mean)/std gives each answer's advantage. Correct answers get +, wrong get −. That's GRPO — critic deleted."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='170' y='12' width='180' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='30' fill='#1F6280'>one prompt x</text></g></svg>",
     "note":"<b>Start with one prompt.</b> A single question — say a math problem the model must solve."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='18' width='100' height='26' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='80' y='35' fill='#5E5191'>answer 1</text><rect x='150' y='18' width='100' height='26' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='200' y='35' fill='#5E5191'>answer 2</text><rect x='270' y='18' width='100' height='26' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='320' y='35' fill='#5E5191'>answer 3</text><rect x='390' y='18' width='100' height='26' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='440' y='35' fill='#5E5191'>answer 4</text></g></svg>",
     "note":"<b>Sample a group of answers.</b> The policy generates G answers to that one prompt. This group is the raw material for the baseline."},
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='80' y='30' fill='#2D8B55'>r=1</text><text x='200' y='30' fill='#C93B3B'>r=0</text><text x='320' y='30' fill='#2D8B55'>r=1</text><text x='440' y='30' fill='#C93B3B'>r=0</text></g></svg>",
     "note":"<b>Score each answer.</b> A reward for each — here two correct (1) and two wrong (0). Tomorrow these rewards come from a verifier."},
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='150' y='12' width='220' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='31' fill='#9A7208'>mean = 0.5   (the baseline)</text></g></svg>",
     "note":"<b>The group mean is the baseline.</b> No value network — the group scores its own reference point on the fly."},
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='80' y='30' fill='#2D8B55'>+1</text><text x='200' y='30' fill='#C93B3B'>-1</text><text x='320' y='30' fill='#2D8B55'>+1</text><text x='440' y='30' fill='#C93B3B'>-1</text></g></svg>",
     "note":"<b>Normalized advantages drive the update.</b> (r − mean)/std pushes above-average answers up and below-average down, then a PPO-style clipped step (with the KL leash) improves the policy — all without a critic."},
  ],
  quiz=[
    {"q":"1. What does GRPO remove compared with PPO?","opts":["the reward","the value critic (it uses the group mean as the baseline instead)","the KL penalty","the policy"],"ans":1,"fb":"GRPO drops the value network; the mean reward over a sampled group serves as the baseline, cutting a whole model."},
    {"q":"2. Rewards in a group are [2, 0, 2, 0] with mean 1 and std 1. The advantages are:","opts":["[2,0,2,0]","[+1,−1,+1,−1]","[0,0,0,0]","[1,1,1,1]"],"ans":1,"fb":"Â = (r − mean)/std = ([2,0,2,0] − 1)/1 = [+1, −1, +1, −1]."},
    {"q":"3. Why is the group mean a valid baseline?","opts":["it is always zero","a state-only baseline reduces variance without bias (M20), and the group mean is one","it replaces the reward","it needs a critic"],"ans":1,"fb":"From M20: subtracting a baseline that doesn't depend on the action leaves the gradient unbiased. The group mean qualifies and is computed from samples."},
    {"q":"4. A prompt's whole group scores identically (all correct). What signal does GRPO get from it?","opts":["a strong positive push","none — std=0 so all advantages are 0, and the rollouts are wasted","a negative push","double reward"],"ans":1,"fb":"Zero variance → zero advantage for every answer → no learning signal. Too-easy or too-hard prompts silently teach nothing; curate difficulty."},
  ],
  fin={"em":"👥","h3":"Day 1 complete — RL without a critic!",
       "p":"You now know GRPO: sample a group of answers per prompt and judge each by <code>(r − mean)/std</code>, using the group mean as the baseline so no value critic is needed. You saw the zero-variance trap and the compute trade-off (more samples instead of a critic). Next: <b>Day 2</b> — where those rewards come from a verifier, not a learned model."},
))

# ---------------- Day 2 — RLVR ----------------
L("day-02-rlvr.html", dict(
  qid="m21b-d02-rlvr", title="Module 21b · Day 2 — RLVR", nav_title="Spiral · M21b Day 2",
  eyebrow="Module 21b · Train · Day 2", h1="RLVR: Rewards You Can Trust",
  lead="Yesterday's reward could come from a learned reward model — which, as Module 21a warned, can be gamed. But for some tasks the reward is <em>objective</em>: is the math answer correct? Do the unit tests pass? RLVR uses these automatic checks as the reward. No reward model to train, nothing to drift — just a checker that says right or wrong. This is what turned base models into strong reasoners.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain what a verifiable reward is, contrast it with a learned reward model, and name where RLVR works and where it can't.",
  prev_href="day-01-grpo.html", prev_label="GRPO", next_href="day-03-prm-vs-orm.html", next_label="PRM vs ORM",
  sections=[
    {"title":"Let the answer check itself","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (GRPO gives a reward-shaped signal) and M21a Days 4–5 (reward models and how they get gamed).</div></div>
     <h4>The problem with learned rewards</h4>
     <p>A reward model is a guess about quality, and Module 21a showed it can be gamed (reward overoptimization). For open-ended tasks that's unavoidable. But for many valuable tasks, we have something better: a way to <em>check</em> the answer directly.</p>
     <h4>Verifiable rewards</h4>
     <p><span class="term" data-tip="Reinforcement Learning from Verifiable Rewards: the reward comes from an automatic checker (e.g. is the math answer correct, do the unit tests pass) rather than a learned reward model.">RLVR (RL from Verifiable Rewards)</span> replaces the reward model with an automatic <span class="term" data-tip="A deterministic checker that returns whether an answer is correct — a math answer-key, a test suite for code, a proof checker.">verifier</span>. Did the model's final number match the known answer? Did its code pass the tests? Reward 1 for yes, 0 for no. No model to train, nothing subjective.</p>
     <h4>Why it's powerful</h4>
     <p>The reward is objective and cannot drift. You can generate unlimited training signal for any problem you can auto-check, and the model cannot "flatter" a checker the way it flatters a reward model. Pair RLVR with GRPO — sample a group, verify each, compute group advantages — and you get the modern reasoning-RL recipe.</p>''',
     "gotit":"Got the idea"},
    {"title":"An answer key vs a human grader","body":
     '''<div class="relate">
       <div class="card"><span class="big">🔑</span><h5>The answer key</h5><p>A multiple-choice test grades itself against a key — instant, objective, unbudgeable. That's a verifier: the math answer either equals 42 or it doesn't.</p></div>
       <div class="card"><span class="big">🧪</span><h5>The test suite</h5><p>For code, a set of unit tests is the key: run the model's function, see if all tests pass. The reward writes itself, and you can't sweet-talk a test.</p></div>
     </div>
     <p><strong>In one line:</strong> when the answer can be checked automatically, use that check as the reward — objective, cheap, and impossible to flatter.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: an answer key covers exactly the tested cases. A weak test suite has gaps, and the model can still slip through them (verifier gaming) — a real failure mode, not a clean solve.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Verify an answer","body":'''<p>Watch a generation, an automatic check, and the resulting reward — for a correct and a wrong answer. <strong>Click all three.</strong></p>'''},
    {"title":"The verifiable reward","body":
     '''<h4>Step 1 — the reward is a check</h4>
     <p>Generate an answer, extract the final result, and run it through a verifier. The reward is whether it passed.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       r(x, y) = 1   if verify(y) is correct<br>
       r(x, y) = 0   otherwise<br>
       <span class="dim"># (or a graded score: fraction of unit tests passed)</span><br>
       feed r into GRPO: Â_i = (r_i − mean)/std over the group
     </div></div>
     <p>The verifier is a plain function — a math answer-checker, a test runner, a proof checker. There is no reward model, so nothing to train and little to game (as long as the check is tight).</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       prompt: "12 × 7 = ?"   gold answer: 84<br><br>
       answer A: "…so it's 84"   → verify → <span class="hl">r = 1</span><br>
       answer B: "…it's 82"      → verify → <span class="hl">r = 0</span><br>
       code task, 10 tests: 7 pass → graded <span class="hl">r = 0.7</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — verifier gaming & reward sparsity.</b> Two quiet failures. (1) <b>Gaming:</b> a loose checker (e.g. tests only easy cases, or matches the answer string anywhere in the output) lets the model pass without truly solving — it learns the exploit, and your accuracy metric lies. (2) <b>Sparsity:</b> if the model gets everything wrong at first, every reward is 0, GRPO's group has zero variance, and there is no gradient to climb — training stalls before it starts. Fix (1) with airtight checks; fix (2) with easier curriculum, partial credit, or a warm SFT start.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — verifiable vs general.</b> Verifiable rewards are objective and un-gameable (no RM drift), which is why reasoning models trained this way are so strong at math and code. But they only exist where you can auto-check — you cannot verify "is this poem beautiful?" or "is this advice kind?" For those, you still need a reward model or human preferences. Frontier training uses RLVR where it can and reward models where it must.</div></div>''',
     "gotit":"Got verifiable rewards"},
    {"title":"RLVR + GRPO, assembled","body":'''<p>Here it is built up: the prompt, a group of answers, the verifier checking each, the 0/1 rewards, and the group-relative update. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with a verifier","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21b_rlvr.py</code>. Write a verifier for a small task (e.g. arithmetic answer-checking or running unit tests), generate several answers, and turn pass/fail into rewards. Combine with the Day 1 GRPO advantage. Show a verifier-gaming case where a loose check passes a wrong answer.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21b Day 2 artifact.
Create experiments/foundations/m21b_rlvr.py: define a strict and a loose verifier for arithmetic answers (loose = substring match anywhere in output). Generate answers, compute 0/1 rewards, feed into GRPO group advantages. Demonstrate the loose verifier being gamed (a wrong answer passes) and reward sparsity when all answers are wrong (std=0).</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21b/day-02-log.md</code>: how did the loose verifier get gamed, and what fixed it?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "gen":{"btn":"① generate an answer","html":'''<span class="prompt">&gt;&gt;&gt;</span> y = policy.generate("12 × 7 = ?")
<span class="hl">y = "12 times 7 is 84."</span>
<span class="dim"># the model shows work and gives a final number</span>''',
      "take":"<b>①  Generate.</b> The model produces an answer, ideally with reasoning and a clear final result to check."},
    "check":{"btn":"② verify it","html":'''<span class="prompt">&gt;&gt;&gt;</span> final = extract_answer(y)   <span class="dim"># 84</span>
<span class="prompt">&gt;&gt;&gt;</span> verify(final, gold=84)
<span class="hl">True</span>   <span class="dim"># objective, no reward model</span>''',
      "take":"<b>②  Check against the key.</b> Extract the final answer and compare to the gold answer (or run the tests). Deterministic and objective."},
    "reward":{"btn":"③ reward = pass/fail","html":'''<span class="prompt">&gt;&gt;&gt;</span> r = 1 if verify(final, gold) else 0
<span class="hl">r = 1</span>   <span class="dim"># wrong answer would give r = 0</span>
<span class="dim"># feed r into GRPO group advantages</span>''',
      "take":"<b>③  Turn the check into reward.</b> 1 for correct, 0 for wrong (or graded). No model to train or game — then GRPO uses it."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='12' width='220' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='30' fill='#1F6280'>prompt: 12 x 7 = ? (gold 84)</text></g></svg>",
     "note":"<b>Start with a checkable prompt.</b> A task where the correct answer is known or testable — math, code, logic."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='16' width='120' height='24' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='100' y='32' fill='#5E5191'>...84</text><rect x='200' y='16' width='120' height='24' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='260' y='32' fill='#5E5191'>...82</text><rect x='360' y='16' width='120' height='24' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='420' y='32' fill='#5E5191'>...84</text></g></svg>",
     "note":"<b>Sample a group of answers.</b> As in GRPO — several attempts at the same prompt."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='33' fill='#9A7208'>verifier(answer) → correct?</text></g></svg>",
     "note":"<b>Verify each automatically.</b> A deterministic checker — answer-key or test suite — decides right or wrong. No learned model."},
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='100' y='30' fill='#2D8B55'>r=1</text><text x='260' y='30' fill='#C93B3B'>r=0</text><text x='420' y='30' fill='#2D8B55'>r=1</text></g></svg>",
     "note":"<b>Rewards write themselves.</b> Correct → 1, wrong → 0 (or graded by fraction of tests passed). Objective and un-gameable, given a tight check."},
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='12' width='280' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='31' fill='#1a5c38'>GRPO: Â = (r − mean)/std → update</text></g></svg>",
     "note":"<b>Plug into GRPO.</b> Group-relative advantages from verifiable rewards. This RLVR + GRPO recipe is what produced the 2026 reasoning models."},
  ],
  quiz=[
    {"q":"1. What is a verifiable reward?","opts":["a reward-model score","a reward from an automatic, objective check (answer-key, unit tests)","a human rating","the KL penalty"],"ans":1,"fb":"RLVR's reward comes from a deterministic checker, not a learned model — objective and un-gameable when the check is tight."},
    {"q":"2. Compared with a learned reward model, a verifier:","opts":["is easier to overoptimize","cannot drift and needs no training, but only works where you can auto-check","requires more GPUs to train","gives fuzzier signal"],"ans":1,"fb":"There's no model to train or drift; the limit is that it only applies to auto-checkable tasks like math and code."},
    {"q":"3. A code task has 10 unit tests; the model passes 7. A graded verifiable reward is:","opts":["1.0","0.7","0.0","7.0"],"ans":1,"fb":"Graded reward = fraction passed = 7/10 = 0.7. Binary would be 0 unless all pass."},
    {"q":"4. Your loose verifier matches the gold number anywhere in the output, and accuracy looks amazing — but the model isn't really solving. This is:","opts":["a perfect result","verifier gaming — the model exploits the weak check without solving the task","reward sparsity","a KL problem"],"ans":1,"fb":"A loose check is an exploitable target: the model learns to trip the checker, not to solve. Tighten the verifier (exact final-answer extraction, hidden tests)."},
  ],
  fin={"em":"🔑","h3":"Day 2 complete — objective rewards!",
       "p":"You now know RLVR: replace the learned reward model with an automatic verifier (answer-key, test suite) giving 0/1 or graded reward, then feed it into GRPO. It's objective and un-gameable where checks are tight — but only works on auto-checkable tasks, and watch for verifier gaming and reward sparsity. Next: <b>Day 3</b> — should you reward the final answer, or every reasoning step?"},
))
# ---------------- Day 3 — PRM vs ORM ----------------
L("day-03-prm-vs-orm.html", dict(
  qid="m21b-d03-prm", title="Module 21b · Day 3 — PRM vs ORM", nav_title="Spiral · M21b Day 3",
  eyebrow="Module 21b · Train · Day 3", h1="Reward the Answer, or Every Step?",
  lead="A reasoning model writes a long chain of steps before its final answer. If you only reward the final answer (an outcome reward), a chain that stumbles through wrong steps but lands on the right number gets full credit — and the model learns bad habits. A process reward scores each step, so it can reward good reasoning and punish a wrong step even when the final answer is right. That is the ORM vs PRM choice.",
  goal="<b>🎯 By the end, you'll be able to:</b> distinguish ORM (outcome) from PRM (process) rewards, explain the long-chain credit-assignment problem, and name the cost of each.",
  prev_href="day-02-rlvr.html", prev_label="RLVR", next_href="day-04-star-rejection.html", next_label="STaR & Best-of-N",
  sections=[
    {"title":"Where do you put the reward?","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–2 (a reward drives GRPO; verifiers give it) and the idea of a chain-of-thought (a model reasoning step by step before answering).</div></div>
     <h4>Two places to score</h4>
     <p>A reasoning answer is a long chain: step 1, step 2, …, final answer. You can reward it in two places.</p>
     <ul>
       <li>An <span class="term" data-tip="Outcome Reward Model: scores only the final answer of a reasoning chain — one reward at the end.">ORM (outcome reward model)</span> scores only the final answer. One number for the whole chain.</li>
       <li>A <span class="term" data-tip="Process Reward Model: scores each individual step of a reasoning chain, giving a dense, per-step signal.">PRM (process reward model)</span> scores every step. A number for each line of reasoning.</li>
     </ul>
     <h4>The credit-assignment problem</h4>
     <p>With only an outcome reward on a 30-step chain, which steps deserve the credit? All of them get the same signal — even the wrong ones — because the reward can't tell them apart. This is the <span class="term" data-tip="The difficulty of figuring out which of many actions deserves credit or blame for a delayed reward. Long reasoning chains make it acute.">credit-assignment problem</span> from M20, made worse by long chains.</p>
     <h4>Why PRM helps</h4>
     <p>A per-step reward points directly at the good and bad steps, so the model learns <em>which reasoning</em> works, not just which final answers happen to be right. Denser signal, better learning — at a cost we'll see.</p>''',
     "gotit":"Got the two choices"},
    {"title":"Grading the working, not just the answer","body":
     '''<div class="relate">
       <div class="card"><span class="big">🔢</span><h5>Only the final answer (ORM)</h5><p>A math teacher who marks only the boxed answer gives full marks to a student who made two errors that cancelled out. The student learns nothing about their broken method.</p></div>
       <div class="card"><span class="big">✍️</span><h5>Mark every line (PRM)</h5><p>A teacher who marks each line of working catches the wrong step even when the final answer is right, and rewards clean reasoning. The student learns the <i>method</i>, not just the lucky answer.</p></div>
     </div>
     <p><strong>In one line:</strong> outcome rewards grade only the final answer; process rewards grade each step, so the model learns good reasoning instead of lucky answers.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a teacher understands each step's correctness for free. A PRM must be <i>trained</i> to judge steps, which needs someone to label thousands of steps as good or bad — the real cost of PRMs.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Score a chain two ways","body":'''<p>Watch a reasoning chain scored by an ORM (final only), then a PRM (per step), then the credit each step receives. <strong>Click all three.</strong></p>'''},
    {"title":"Lucky-wrong reasoning","body":
     '''<h4>Step 1 — the two reward shapes</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       ORM:  reward = score(final_answer)            <span class="dim"># one number, at the end</span><br>
       PRM:  reward_t = score(step_t) for each step  <span class="dim"># a number per step</span>
     </div></div>
     <h4>Step 2 — the failure ORM misses</h4>
     <p>Consider a chain that makes an error in step 3 but, by luck, still reaches the correct final answer.</p>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       chain:  step1 ✓   step2 ✓   step3 ✗ (error)   step4 ✓   final ✓ (correct by luck)<br><br>
       ORM reward:  final is correct → <span class="hl">+1 for the whole chain</span>  (rewards the broken step 3!)<br>
       PRM rewards: [+1, +1, <span class="hl">−1</span>, +1]  (catches step 3 as bad)
     </div></div>
     <p>The ORM reinforces the flawed reasoning because the outcome happened to be right. The PRM pinpoints the bad step. On long chains, this difference compounds — models trained on outcome-only rewards learn to reach answers through unreliable reasoning.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — ORM rewards lucky-wrong reasoning.</b> With outcome-only rewards, a model can score perfectly while reasoning badly: right answers via wrong methods get reinforced. Accuracy on the training distribution looks great, then the model fails on harder problems where the flawed method no longer luckily cancels out. The reward curve hides it; only step-level inspection (or a PRM) reveals it.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — PRM signal vs label cost.</b> A PRM gives a dense, well-targeted signal that trains better reasoning and enables step-level search — but it needs step-by-step labels (expensive human or model annotation) and those labels can be noisy or gameable. An ORM is cheap (just check the final answer, e.g. via RLVR) but gives sparse, poorly-targeted credit. Many teams use cheap ORM/verifier rewards at scale and reserve PRMs for the hardest reasoning.</div></div>''',
     "gotit":"Got lucky-wrong reasoning"},
    {"title":"Outcome vs process, built up","body":'''<p>Here it is assembled: the reasoning chain, the single ORM reward at the end, the per-step PRM rewards, and the lucky-wrong case. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove the ORM/PRM gap","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21b_prm_orm.py</code>. Build toy reasoning chains where some reach the right answer via a wrong step. Score them with an ORM (final-only) and a PRM (per-step) and show the ORM rewarding a lucky-wrong chain that the PRM penalizes.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21b Day 3 artifact.
Create experiments/foundations/m21b_prm_orm.py: represent reasoning chains as lists of (step, is_correct) plus a final-answer-correct flag. Implement an ORM (reward = final correct) and a PRM (per-step rewards). Construct a "lucky-wrong" chain (a wrong step but correct final answer) and print how ORM rewards it while PRM flags the bad step.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21b/day-03-log.md</code>: what did the ORM reward that the PRM caught?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "orm":{"btn":"① ORM: score the answer","html":'''<span class="prompt">&gt;&gt;&gt;</span> chain = [s1, s2, s3_wrong, s4]; final = 84 (correct)
<span class="prompt">&gt;&gt;&gt;</span> orm_reward = 1 if final_correct else 0
<span class="hl">orm_reward = 1</span>   <span class="dim"># whole chain gets +1, wrong step included</span>''',
      "take":"<b>①  ORM scores only the end.</b> One reward for the whole chain. It can't see that step 3 was wrong — the answer was right, so everything gets +1."},
    "prm":{"btn":"② PRM: score each step","html":'''<span class="prompt">&gt;&gt;&gt;</span> prm_rewards = [prm(s) for s in chain]
<span class="hl">[+1, +1, -1, +1]</span>
<span class="dim"># step 3 flagged as bad even though the final answer is right</span>''',
      "take":"<b>②  PRM scores every step.</b> A per-step signal catches the wrong step 3 that the ORM missed — the model learns which reasoning is good."},
    "credit":{"btn":"③ credit assignment","html":'''<span class="prompt">&gt;&gt;&gt;</span> # ORM: all steps share one reward → can't tell them apart
<span class="prompt">&gt;&gt;&gt;</span> # PRM: each step gets its own → precise credit
<span class="hl">PRM localizes credit; ORM smears it</span>''',
      "take":"<b>③  PRM localizes credit.</b> On long chains, per-step rewards point at the exact good/bad steps, solving the credit-assignment problem outcome rewards struggle with."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='14' width='90' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='75' y='30' fill='#1a5c38'>step1 ✓</text><rect x='130' y='14' width='90' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='175' y='30' fill='#1a5c38'>step2 ✓</text><rect x='230' y='14' width='90' height='24' rx='4' fill='#FDE8E8' stroke='#C93B3B'/><text x='275' y='30' fill='#C93B3B'>step3 ✗</text><rect x='330' y='14' width='90' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='375' y='30' fill='#1a5c38'>step4 ✓</text><rect x='430' y='14' width='70' height='24' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='465' y='30' fill='#1F6280'>final ✓</text></g></svg>",
     "note":"<b>A reasoning chain.</b> Steps lead to a final answer. Here step 3 is wrong, yet the final answer is (luckily) correct."},
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='330' y='12' width='170' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='415' y='31' fill='#9A7208'>ORM: +1 (final only)</text></g></svg>",
     "note":"<b>ORM rewards only the outcome.</b> The final answer is correct, so the whole chain — including the broken step 3 — gets +1."},
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><text x='75' y='30' fill='#2D8B55'>+1</text><text x='175' y='30' fill='#2D8B55'>+1</text><text x='275' y='30' fill='#C93B3B'>-1</text><text x='375' y='30' fill='#2D8B55'>+1</text></g></svg>",
     "note":"<b>PRM rewards each step.</b> It flags step 3 as bad (−1) even though the final answer was right — catching the flawed reasoning."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='33' fill='#C93B3B'>ORM reinforces the lucky-wrong chain</text></g></svg>",
     "note":"<b>The lucky-wrong trap.</b> Outcome-only rewards teach the model to reach answers through unreliable reasoning — fine on easy problems, fragile on hard ones."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='18' width='200' height='26' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='140' y='35' fill='#1F6280'>ORM: cheap, sparse credit</text><rect x='280' y='18' width='200' height='26' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='380' y='35' fill='#5E5191'>PRM: dense, costly labels</text></g></svg>",
     "note":"<b>The choice.</b> ORM is cheap but smears credit; PRM localizes credit but needs expensive step labels. Use cheap verifiable outcomes at scale, PRMs for the hardest reasoning."},
  ],
  quiz=[
    {"q":"1. What does a process reward model (PRM) score?","opts":["only the final answer","each individual reasoning step","the prompt length","the KL divergence"],"ans":1,"fb":"A PRM gives a reward per step, versus an ORM which scores only the final outcome."},
    {"q":"2. Why does outcome-only reward struggle on long reasoning chains?","opts":["chains are too short","credit assignment — one end reward can't say which of many steps was good or bad","it needs a critic","it ignores the answer"],"ans":1,"fb":"A single delayed reward smears credit across all steps, so the model can't tell good reasoning from bad — the credit-assignment problem."},
    {"q":"3. A chain has a wrong step 3 but a correct final answer. What does an ORM give it?","opts":["a negative reward for step 3","full positive reward (it only sees the correct final answer)","zero","it depends on step 1"],"ans":1,"fb":"The ORM only checks the outcome; since the final answer is correct it rewards the whole chain — reinforcing the flawed step 3 (lucky-wrong)."},
    {"q":"4. The main cost of using a PRM is:","opts":["it needs more GPUs to run","it requires expensive per-step labels (and those can be noisy/gameable)","it can't be combined with GRPO","it only works on code"],"ans":1,"fb":"PRMs need step-level annotations — costly to collect and imperfect — which is why cheap outcome/verifier rewards are used at scale and PRMs reserved for hard reasoning."},
  ],
  fin={"em":"✍️","h3":"Day 3 complete — you can grade reasoning!",
       "p":"You now know ORM vs PRM: outcome rewards score only the final answer (cheap, but reward lucky-wrong reasoning and smear credit), while process rewards score each step (dense, better credit, but need costly step labels). This is the credit-assignment problem from M20, made sharp by long chains. Next: <b>Day 4</b> — self-improvement without RL: STaR, rejection sampling, and best-of-N."},
))

# ---------------- Day 4 — STaR, Rejection Sampling & Best-of-N ----------------
L("day-04-star-rejection.html", dict(
  qid="m21b-d04-star", title="Module 21b · Day 4 — STaR & Best-of-N", nav_title="Spiral · M21b Day 4",
  eyebrow="Module 21b · Train · Day 4", h1="Self-Improvement Without RL",
  lead="You don't always need full reinforcement learning to make a model reason better. A model can improve itself: generate many solutions, keep only the ones that are correct, and fine-tune on those. That's STaR and rejection sampling — self-training on your own best work. And at test time, best-of-N sampling squeezes out more accuracy by trying several answers and picking the best.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain the STaR / rejection-sampling loop, describe best-of-N at inference, and say why this is simpler but weaker than full RL.",
  prev_href="day-03-prm-vs-orm.html", prev_label="PRM vs ORM", next_href="day-05-rl-distill-flywheel.html", next_label="RL→Distill Flywheel",
  sections=[
    {"title":"Learn from your own successes","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 2 (a verifier can tell correct from wrong) and M21a Day 2 (SFT = fine-tune on good examples).</div></div>
     <h4>The idea</h4>
     <p>If you can <em>check</em> answers (Day 2), you can bootstrap. Have the model generate many solutions to each problem, use the verifier to <span class="term" data-tip="Keeping only the samples that pass a check (e.g. reach the correct answer) and discarding the rest, then training on the kept ones.">reject</span> the wrong ones, and fine-tune the model on the correct solutions it produced. It learns from its own best work.</p>
     <h4>STaR</h4>
     <p><span class="term" data-tip="Self-Taught Reasoner: generate reasoning chains, keep those that reach the correct answer, fine-tune on them, and repeat — the model teaches itself to reason.">STaR (Self-Taught Reasoner)</span> loops this: generate rationales → keep the ones that got the right answer → SFT on them → repeat. Each round the model gets a bit better, so its next batch of solutions is better, so the training data improves.</p>
     <h4>Best-of-N at test time</h4>
     <p>The same idea helps at inference. <span class="term" data-tip="At inference, sample N answers and pick the best one (by a verifier or reward model), trading extra compute for higher accuracy.">Best-of-N</span>: sample N answers, use a verifier or reward model to pick the best, and return that. More samples → higher chance one is correct → better accuracy, for more compute.</p>''',
     "gotit":"Got the idea"},
    {"title":"Keep your good drafts","body":
     '''<div class="relate">
       <div class="card"><span class="big">🗂️</span><h5>Study your wins</h5><p>A student solves a problem five ways, checks which attempts were right, throws away the wrong ones, and studies the correct solutions. Next time they're better — and their new attempts are better still. That loop is STaR.</p></div>
       <div class="card"><span class="big">🎯</span><h5>Submit your best try</h5><p>On the real exam, if allowed, they'd work the problem a few times and hand in the attempt they're most confident is right. That's best-of-N: more tries, pick the best.</p></div>
     </div>
     <p><strong>In one line:</strong> generate many attempts, keep the correct ones to train on (STaR), and at test time submit the best of several tries (best-of-N).</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a student learns from mistakes too. STaR throws wrong attempts away entirely — it only imitates successes, with no signal from failures. That's the key difference from full RL, which uses both.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Run the STaR loop","body":'''<p>Watch many solutions generated, the wrong ones rejected, and an SFT round on the survivors. <strong>Click all three.</strong></p>'''},
    {"title":"The bootstrap loop","body":
     '''<h4>Step 1 — the STaR loop</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       repeat:<br>
       &nbsp;&nbsp;1. for each problem, sample K solutions from the model<br>
       &nbsp;&nbsp;2. keep only solutions the verifier marks correct  (reject the rest)<br>
       &nbsp;&nbsp;3. SFT the model on the kept (problem, correct-solution) pairs<br>
       <span class="dim"># the model teaches itself; each round the kept set gets better</span>
     </div></div>
     <h4>Step 2 — best-of-N at inference</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       sample N answers → pick argmax over verifier/reward-model score → return it
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       100 problems, sample 8 solutions each = 800 attempts<br>
       verifier keeps the correct ones → say 240 correct solutions survive<br>
       SFT on those 240 → model improves → next round keeps more (say 320)<br>
       <span class="dim"># the loop compounds as long as the model can solve *some* new problems</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — diversity collapse / narrow style.</b> Because STaR only trains on successes, it amplifies whatever solution style already works and drops everything else. Over rounds the model can collapse to one narrow way of solving, losing diversity — and it stops improving on anything it couldn't already sometimes solve (nothing correct survives the filter for the truly-hard problems). Training looks stable while coverage quietly narrows; track solution diversity and accuracy on hard held-out problems.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — simplicity vs power.</b> STaR/rejection sampling is just \"generate, filter, SFT\" — simple, stable, offline, no RL machinery. But it only imitates successes: it uses no negative signal (unlike GRPO/PPO, which push down bad answers), so it's generally weaker and plateaus sooner. Best-of-N buys accuracy at inference but multiplies compute per query. Teams often start with rejection sampling, then move to full RL for the last gains.</div></div>''',
     "gotit":"Got the bootstrap loop"},
    {"title":"The self-improvement loop","body":'''<p>Here it is built up: many samples, the verifier filter, the kept correct set, the SFT round, and the loop compounding. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove it with rejection sampling","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21b_star.py</code>. On a small checkable task, sample K solutions per problem, filter to correct ones with a verifier, and SFT on the survivors for a couple of rounds. Track accuracy and solution diversity across rounds.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21b Day 4 artifact.
Create experiments/foundations/m21b_star.py: a STaR/rejection-sampling loop on a small arithmetic-reasoning task — sample K=8 solutions per problem, keep verifier-correct ones, SFT on them, repeat 2-3 rounds. Print accuracy and a diversity measure per round, and also best-of-N accuracy for N in {1,4,16}.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21b/day-04-log.md</code>: did accuracy rise across rounds, and did solution diversity shrink?</div></div>''',
     "gotit":"Done — on to Day 5"},
  ],
  demos={
    "gen":{"btn":"① generate many solutions","html":'''<span class="prompt">&gt;&gt;&gt;</span> sols = [policy.generate(problem) for _ in range(8)]
<span class="hl">8 attempts per problem</span>
<span class="dim"># some correct, some wrong</span>''',
      "take":"<b>①  Generate a batch of attempts.</b> Sample many solutions per problem. Diversity here is what makes the filtering useful."},
    "reject":{"btn":"② reject the wrong ones","html":'''<span class="prompt">&gt;&gt;&gt;</span> kept = [s for s in sols if verify(s)]
<span class="hl">kept 3 of 8</span>   <span class="dim"># only correct solutions survive</span>
<span class="dim"># the 5 wrong ones are discarded (no negative signal)</span>''',
      "take":"<b>②  Keep only the successes.</b> The verifier filters to correct solutions. Wrong attempts are thrown away — STaR learns only from wins."},
    "sft":{"btn":"③ SFT on the survivors","html":'''<span class="prompt">&gt;&gt;&gt;</span> sft(model, kept_correct_solutions)
<span class="hl">model improves → next round keeps more</span>
<span class="dim"># repeat: a self-teaching loop</span>''',
      "take":"<b>③  Train on your best work.</b> Fine-tune on the kept solutions. The improved model then produces better attempts next round — the loop compounds."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='16' width='70' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='75' y='32' fill='#5E5191'>sol 1</text><rect x='120' y='16' width='70' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='155' y='32' fill='#5E5191'>sol 2</text><rect x='200' y='16' width='70' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><text x='235' y='32' fill='#5E5191'>sol 3</text><text x='300' y='32' fill='#6B645E'>… × 8</text></g></svg>",
     "note":"<b>Generate many solutions.</b> The model attempts each problem several times, producing diverse candidate reasoning chains."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='160' y='14' width='200' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='33' fill='#9A7208'>verifier: correct?</text></g></svg>",
     "note":"<b>Filter with the verifier.</b> Check each attempt. Only the correct ones move forward; the rest are rejected (and their signal lost)."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='120' y='16' width='90' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='165' y='32' fill='#1a5c38'>correct 1</text><rect x='220' y='16' width='90' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='265' y='32' fill='#1a5c38'>correct 2</text><rect x='320' y='16' width='90' height='24' rx='4' fill='#E8F5EE' stroke='#2D8B55'/><text x='365' y='32' fill='#1a5c38'>correct 3</text></g></svg>",
     "note":"<b>Keep the correct solutions.</b> These verified-good reasoning chains become the training set — the model's own best work."},
    {"viz":"<svg viewBox='0 0 520 50'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='12' width='220' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='31' fill='#1F6280'>SFT on the kept solutions</text></g></svg>",
     "note":"<b>Fine-tune on the survivors.</b> Plain SFT (M21a Day 2) on the correct solutions. The model absorbs its own successful reasoning."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><path d='M120,40 Q260,10 400,40' fill='none' stroke='#6B645E' stroke-width='1.5' stroke-dasharray='4,3'/><text x='260' y='52' fill='#6B645E'>better model → better attempts → repeat</text></g></svg>",
     "note":"<b>The loop compounds.</b> Each round the model solves a bit more, so more correct solutions survive to train on. Simple and stable — but only imitates successes (no negative signal), so it's weaker than full RL."},
  ],
  quiz=[
    {"q":"1. What does STaR / rejection sampling train on?","opts":["all generated solutions","only the solutions a verifier marks correct (successes are kept, failures discarded)","only wrong solutions","human demonstrations"],"ans":1,"fb":"It keeps verifier-correct solutions and SFTs on them — self-training on the model's own best work."},
    {"q":"2. What is best-of-N at inference?","opts":["training for N epochs","sampling N answers and returning the best by a verifier/reward model","using N GPUs","N reward models"],"ans":1,"fb":"Best-of-N trades extra compute for accuracy: draw N answers, pick the best one. More N → higher chance a good answer is among them."},
    {"q":"3. Why is STaR generally weaker than full RL (GRPO/PPO)?","opts":["it uses a critic","it only imitates successes and uses no negative signal from wrong answers","it can't use a verifier","it needs a reward model"],"ans":1,"fb":"STaR discards failures entirely, so unlike RL it never pushes down bad answers — less signal, earlier plateau."},
    {"q":"4. After several STaR rounds, the model solves the same problems but its solutions all look identical and it stopped improving on hard ones. This is:","opts":["ideal convergence","diversity collapse — training only on successes narrows the style and can't crack problems it never solves","a verifier bug","reward hacking"],"ans":1,"fb":"Imitating only successes amplifies one style and offers no path on problems where nothing correct ever survives the filter. Track diversity and hard-set accuracy."},
  ],
  fin={"em":"🗂️","h3":"Day 4 complete — the model teaches itself!",
       "p":"You now know self-improvement: STaR / rejection sampling generate many solutions, keep the verifier-correct ones, and SFT on them in a compounding loop; best-of-N picks the best of several tries at inference. It's simple, stable, and offline — but only imitates successes (no negative signal), so it's weaker than full RL and can collapse diversity. Next: <b>Day 5</b> — the flywheel that ties RL, distillation, and self-improvement together."},
))
# ---------------- Day 5 — The RL to Distill Flywheel ----------------
L("day-05-rl-distill-flywheel.html", dict(
  qid="m21b-d05-flywheel", title="Module 21b · Day 5 — RL→Distill Flywheel", nav_title="Spiral · M21b Day 5",
  eyebrow="Module 21b · Train · Day 5", h1="The RL → Distill Flywheel",
  lead="Reasoning RL is expensive — you can only afford to run it on so many models. But once you have one strong reasoner, you can bottle its skill cheaply: have it generate high-quality reasoning traces, then fine-tune smaller or fresh models on those traces. That's distillation. Chain it with RL and you get a flywheel that turns a lot of compute on one model into many capable models — the pattern behind the 2026 open reasoning models.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain the RL→distill flywheel, say why distillation is so much cheaper than RL, and name the ceiling it can't cross on its own.",
  prev_href="day-04-star-rejection.html", prev_label="STaR & Best-of-N", next_href="review.html", next_label="Review Gate",
  sections=[
    {"title":"Bottle the reasoning","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–4 (GRPO, RLVR, self-improvement), M21a Day 2 (SFT), and M9c distillation as a concept (a small student learns from a big teacher).</div></div>
     <h4>RL is expensive; traces are cheap</h4>
     <p>Running GRPO/RLVR to build a strong reasoner takes a lot of compute. But once you have that reasoner, generating its reasoning is just inference — cheap. So capture it: have the strong model produce many high-quality <span class="term" data-tip="A full reasoning chain from a strong model — the steps plus the final answer. Used as training data to teach another model.">traces</span> (worked solutions), then train other models on them.</p>
     <h4>Distillation</h4>
     <p><span class="term" data-tip="Training a smaller or fresh 'student' model to imitate the outputs of a strong 'teacher' model — here, its reasoning traces — via ordinary SFT.">Distillation</span> here is just SFT (M21a Day 2) on the teacher's traces. A small student model can absorb a lot of the teacher's reasoning skill this way — far more cheaply than discovering it through RL itself.</p>
     <h4>The flywheel</h4>
     <p>Now chain it: RL a strong model → distill its traces into a smaller/fresh model → optionally RL <em>that</em> model further on verifiable rewards → distill again. Each turn converts expensive RL on one model into broad capability across many. That compounding loop is the flywheel.</p>''',
     "gotit":"Got the idea"},
    {"title":"The master and the apprentices","body":
     '''<div class="relate">
       <div class="card"><span class="big">👨‍🏫</span><h5>One master, hard-won skill</h5><p>A master craftsperson spent years (expensive RL) getting great. You can't clone the years — but you can have them produce many worked examples.</p></div>
       <div class="card"><span class="big">🧑‍🎓</span><h5>Many apprentices, fast</h5><p>Apprentices study those worked examples and get good quickly (cheap distillation). Some apprentices then push further on their own (more RL) and produce even better examples for the next batch.</p></div>
     </div>
     <p><strong>In one line:</strong> pay once for RL on a strong model, then cheaply copy its reasoning into many models by training them on its traces — and repeat.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a bright apprentice can eventually surpass the master through their own practice. A distilled student, trained only to imitate, cannot exceed its teacher <i>unless</i> it gets its own fresh learning signal (its own RL on verifiable rewards).</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Turn the flywheel","body":'''<p>Watch a strong teacher generate traces, a student distilled on them, and the improved student feeding the next round. <strong>Click all three.</strong></p>'''},
    {"title":"Why it compounds — and its ceiling","body":
     '''<h4>Step 1 — the loop</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       1. teacher = RL-tuned strong reasoner (expensive, done once)<br>
       2. traces = teacher.generate(problems)      <span class="dim"># cheap inference</span><br>
       3. student = SFT(base_model, traces)          <span class="dim"># cheap distillation</span><br>
       4. (optional) student = RLVR(student)         <span class="dim"># student earns its own new signal</span><br>
       5. teacher ← best student; repeat
     </div></div>
     <h4>Step 2 — why distillation is cheap</h4>
     <p>RL needs many rollouts, reward evaluation, and careful stabilization. Distillation is one supervised pass over fixed traces — orders of magnitude cheaper per unit of capability transferred. That asymmetry is the whole reason the flywheel works.</p>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       RL a 70B reasoner: cost = 100 units (once)<br>
       generate 1M traces: cost = 5 units<br>
       distill into a 7B student: cost = 3 units → a small model with most of the reasoning<br>
       <span class="dim"># one expensive RL run → many cheap, capable students</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — inheriting the teacher's flaws.</b> Distillation copies <i>whatever</i> the teacher does — including its mistakes, biases, and blind spots. If the teacher has a systematic error (a wrong method that usually works, a safety gap), every student inherits it, and now it's spread across your whole model family. It looks like faithful learning; it's faithful copying of flaws. Guard by verifying traces (keep only verifier-correct ones, like Day 4) and evaluating students independently, not just against the teacher.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — cheap breadth vs a capped ceiling.</b> Distillation spreads a teacher's skill cheaply and fast, but a pure student <b>cannot exceed its teacher</b> — it only imitates. To push past the teacher you must inject new signal: give the student its own RL on verifiable rewards (step 4), so it can discover solutions the teacher never found. Distillation scales what you have; RL is what actually raises the frontier.</div></div>''',
     "gotit":"Got the flywheel"},
    {"title":"The flywheel, assembled","body":'''<p>Here it is built up: the expensive RL teacher, its cheap traces, the distilled student, the optional student RL, and the loop feeding back. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove the flywheel","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m21b_flywheel.py</code>. Use a stronger model (or a hand-written \"teacher\") to generate verified-correct traces on a small task, distill (SFT) a weaker student on them, and measure the student's accuracy before vs after. Show the student approaching — but not exceeding — the teacher.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 21b Day 5 artifact.
Create experiments/foundations/m21b_flywheel.py: simulate the RL->distill flywheel on a small task — a "teacher" produces verified-correct traces, distill (SFT) a weaker "student" on them, and plot student accuracy before/after vs the teacher's. Show the student approaching but not exceeding the teacher unless it gets its own verifiable-reward step.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m21b/day-05-log.md</code>: how close did the student get to the teacher, and what would let it surpass it?</div></div>''',
     "gotit":"Done — on to the review gate"},
  ],
  demos={
    "teach":{"btn":"① teacher makes traces","html":'''<span class="prompt">&gt;&gt;&gt;</span> teacher = rl_tuned_reasoner   <span class="dim"># expensive, built once</span>
<span class="prompt">&gt;&gt;&gt;</span> traces = teacher.generate(problems)
<span class="hl">1,000,000 high-quality reasoning traces</span>   <span class="dim"># cheap inference</span>''',
      "take":"<b>①  Bottle the skill.</b> The costly RL reasoner just runs inference to produce many worked solutions — cheap to generate, rich in reasoning."},
    "distill":{"btn":"② distill into a student","html":'''<span class="prompt">&gt;&gt;&gt;</span> student = sft(base_7B, traces)   <span class="dim"># plain SFT on the traces</span>
<span class="hl">a 7B model with most of the 70B's reasoning</span>
<span class="dim"># orders of magnitude cheaper than RL from scratch</span>''',
      "take":"<b>②  Copy it cheaply.</b> One supervised pass teaches a small student most of the teacher's reasoning — far cheaper than the student discovering it via RL."},
    "loop":{"btn":"③ close the loop","html":'''<span class="prompt">&gt;&gt;&gt;</span> student = rlvr(student)   <span class="dim"># optional: its own new signal</span>
<span class="prompt">&gt;&gt;&gt;</span> teacher = best(student, teacher)   <span class="dim"># repeat</span>
<span class="hl">expensive RL once → many capable models</span>''',
      "take":"<b>③  Turn the flywheel.</b> Optionally RL the student on verifiable rewards so it can surpass the teacher, then use the best model as the next teacher."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='140' y='14' width='240' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='33' fill='#1F6280'>RL teacher (expensive, once)</text></g></svg>",
     "note":"<b>Start with a strong RL-tuned reasoner.</b> Built with GRPO/RLVR at real cost — but only once."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='33' fill='#9A7208'>generate traces (cheap inference)</text></g></svg>",
     "note":"<b>Generate reasoning traces.</b> Just inference — the teacher writes many worked solutions. Cheap relative to the RL that created it."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='33' fill='#5E5191'>distill: SFT a student on traces</text></g></svg>",
     "note":"<b>Distill into a student.</b> Ordinary SFT on the traces transfers most of the reasoning to a smaller or fresh model — cheaply."},
    {"viz":"<svg viewBox='0 0 520 55'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='33' fill='#1a5c38'>optional: RLVR the student (new signal)</text></g></svg>",
     "note":"<b>Give the student its own RL (optional).</b> Verifiable-reward RL lets the student discover solutions the teacher never found — the only way to exceed the teacher."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><path d='M120,42 Q260,10 400,42' fill='none' stroke='#6B645E' stroke-width='1.5' stroke-dasharray='4,3'/><text x='260' y='54' fill='#6B645E'>best student → next teacher → repeat</text></g></svg>",
     "note":"<b>Close the loop.</b> The best model becomes the next teacher. One expensive RL investment compounds into a family of capable models — the flywheel."},
  ],
  quiz=[
    {"q":"1. In the RL→distill flywheel, what is 'distillation'?","opts":["running RL on the student","SFT-training a student on the teacher's reasoning traces","deleting the teacher","a verifier"],"ans":1,"fb":"Distillation here is plain SFT on the strong teacher's generated traces — cheaply transferring its reasoning to another model."},
    {"q":"2. Why is distillation so much cheaper than RL?","opts":["it uses bigger models","it's one supervised pass over fixed traces, versus RL's many rollouts + reward eval + stabilization","it needs no data","it skips the teacher"],"ans":1,"fb":"RL is expensive (rollouts, rewards, instability); distillation is a single SFT pass over already-generated traces — orders of magnitude cheaper per unit of capability."},
    {"q":"3. A pure distilled student can, at best:","opts":["far exceed its teacher","approach but not exceed its teacher (it only imitates)","ignore the teacher","only match a random model"],"ans":1,"fb":"Imitation caps the student at the teacher's level; surpassing it requires new signal — the student's own RL on verifiable rewards."},
    {"q":"4. Your teacher has a subtle systematic reasoning error. After distilling 5 students, what's the risk?","opts":["the error disappears","every student inherits the same flaw — it's copied across the whole family","only one student has it","the students fix it automatically"],"ans":1,"fb":"Distillation faithfully copies the teacher's flaws too. Verify traces (keep only correct ones) and evaluate students independently to avoid spreading a systematic error."},
  ],
  fin={"em":"⚙️","h3":"Day 5 complete — you can scale reasoning!",
       "p":"You now know the RL→distill flywheel: RL one strong reasoner (expensive, once), generate its traces (cheap), distill them into many students via SFT, optionally RL those students on verifiable rewards to push past the teacher, and repeat. Distillation spreads skill cheaply but can't exceed the teacher alone, and it copies flaws. That completes Reasoning-RL — on to the review gate."},
))

# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m21b-review", title="Module 21b · Review Gate", nav_title="Spiral · M21b Review",
  eyebrow="Module 21b · The Gate — You Understand Reasoning-RL", h1="Module 21b — Review Gate",
  lead="Checkpoint. The gate question: <b>can I explain GRPO, verifiable rewards, process vs outcome rewards, self-improvement, and the distillation flywheel — the 2026 reasoning recipe?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. This is the exact stack behind the current frontier reasoning models — worth locking in.",
  prev_href="day-05-rl-distill-flywheel.html", prev_label="Day 5 · RL→Distill Flywheel",
  checks=[
    ["Day 1","I can explain GRPO — sample a group, use the group mean as the baseline, advantage <code>(r − mean)/std</code>, no critic — and the zero-variance trap."],
    ["Day 2","I can explain RLVR — an automatic verifier gives the reward — and where it works (math/code) vs where it can't (open-ended)."],
    ["Day 3","I can distinguish ORM (final answer) from PRM (per step), and explain lucky-wrong reasoning and the credit-assignment gap."],
    ["Day 4","I can describe STaR / rejection sampling (keep correct solutions, SFT, repeat) and best-of-N, and why they're weaker than full RL."],
    ["Day 5","I can explain the RL→distill flywheel, why distillation is cheap, and why a student can't exceed its teacher without its own RL."],
  ],
  quiz=[
    {"q":"1. GRPO removes which PPO component?","opts":["the policy","the value critic (using the group mean as the baseline)","the reward","the KL penalty"],"ans":1,"fb":"GRPO drops the value network; the group's mean reward is the baseline (Day 1)."},
    {"q":"2. Group rewards are [1,1,0,0], mean 0.5, std 0.5. Advantages:","opts":["[1,1,0,0]","[+1,+1,−1,−1]","[0,0,0,0]","[0.5,0.5,0.5,0.5]"],"ans":1,"fb":"(r − 0.5)/0.5 = [+1,+1,−1,−1] (Day 1)."},
    {"q":"3. A verifiable reward comes from:","opts":["a learned reward model","an automatic, objective check (answer-key or tests)","human preference pairs","the KL term"],"ans":1,"fb":"RLVR uses a deterministic verifier, not a learned model — objective and hard to game (Day 2)."},
    {"q":"4. An outcome-only reward on a long chain risks:","opts":["being too dense","rewarding lucky-wrong reasoning (right answer via a wrong step)","needing step labels","ignoring the answer"],"ans":1,"fb":"ORM credits the whole chain for a correct final answer, reinforcing flawed steps — the lucky-wrong problem a PRM catches (Day 3)."},
    {"q":"5. A pure distilled student relative to its teacher can:","opts":["always beat it","approach but not exceed it without its own RL signal","never learn anything","only match noise"],"ans":1,"fb":"Imitation caps the student at the teacher; new verifiable-reward RL is needed to surpass it (Day 5)."},
  ],
  verdict_pass="you understand the modern reasoning-RL recipe end to end — GRPO's critic-free group baseline, verifiable rewards, process vs outcome credit, self-improvement by rejection sampling, and the distillation flywheel. You've completed the post-training arc.",
  verdict_fail="note which day tripped you up and re-run just that lesson. The chain — <code>GRPO</code> → <code>RLVR</code> → <code>PRM vs ORM</code> → <code>STaR / best-of-N</code> → <code>RL→distill flywheel</code> — should feel familiar.",
  complete_label="Mark Module 21b complete",
  fin={"em":"🏆","h3":"Module 21b — passed!",
       "p":"You now know how 2026's reasoning models are trained: GRPO removes the critic with a group baseline, RLVR gives objective rewards, process rewards fix credit assignment, rejection sampling self-improves offline, and the RL→distill flywheel scales one expensive reasoner into many. That completes the RL and post-training arc (M20–M21)."},
  score_target=4,
))
print("M21b built: 5 lessons + review")


