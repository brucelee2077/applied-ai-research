---
quest_id: wf2-d02-activations
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 2 — Activation Functions"
module_label: "Module 2 · Train · Day 2"
title: "Activation Functions"
subtitle: "The Bend That Makes Depth Matter"
brand_sub: "Foundations · M2 Day 2"
spine: "bend"
nav_prev_href: "../day-01-single-neuron/lesson.html"
nav_prev_label: "A Single Neuron"
nav_next_href: "../day-03-layers-forward-pass/lesson.html"
nav_next_label: "Layers & the Forward Pass"
fin_title: "Module 2 · Day 2 complete! 🏆"
fin_body: "Nice work — you've completed <b>Activation Functions</b>.<br>Next up: <b>Layers &amp; the Forward Pass</b>."
notebook_yardstick: 00-neural-networks/fundamentals/03_activation_functions.ipynb
---

@@@ hero
@lede Yesterday a neuron ended with a mysterious third step: the activation. Today you find out why it's there — and it's a big deal. Without it, stacking a hundred layers would be no better than one. The activation is the little bend that lets deep networks learn complicated things.
@goal By the end, you'll be able to: say what an activation adds (a bend, called non-linearity), show why layers with no activation collapse into one, and describe ReLU, sigmoid, and tanh.

@@@ concept id=c1 tag="The problem" title="Straight + straight is still straight" gotit="Got the problem"
Here's the catch that makes today matter. A weighted sum plus a bias is a [[linear||A straight-line relationship: output changes in fixed proportion to input. A matmul plus bias is linear.]] step — picture a perfectly straight ruler. Now stack two of them: lay one straight ruler on top of another and you still get… a single straight ruler. Two neuron-layers with no bend between them collapse into one layer, so all that extra depth buys you nothing.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Two linear layers W1 then W2 collapse into a single matrix W1 times W2"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="40" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="70" y="65" fill="#5E5191">W₁</text><text x="115" y="65" fill="#6B645E">then</text><rect x="150" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="180" y="65" fill="#5E5191">W₂</text><text x="250" y="65" fill="#6B645E">=</text><rect x="290" y="40" width="140" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="360" y="65" fill="#C93B3B" font-weight="bold">one matrix W₁@W₂</text><text x="260" y="105" fill="#C93B3B" font-size="11">no activation → two layers become one</text></g></svg>
%%%

#### Why depth alone buys nothing
Stacking two linear layers gives `(x@W1)@W2 = x@(W1@W2)`. Since `W1@W2` is just another matrix, the two layers are secretly one. Ten layers, a hundred — still one linear layer. Depth without a bend is an illusion.

!!! c-info 🪜
<b>Math Ladder — why linear ∘ linear collapses.</b>
<br><b>1 · In words:</b> two plain matmuls back-to-back, with no bend between them, do the same job as one matmul — so the second layer bought you nothing.
<br><b>2 · Scalars first (no matrix algebra needed):</b> chain two linear steps `f(x)=a·x+p` then `g(u)=b·u+q`. Compose them: `g(f(x)) = b·(a·x+p)+q = (ab)·x + (bp+q)` — still `slope·x + shift`, i.e. one linear step. Two linear layers with no bend collapse to one.
<br><b>3 · Now the matrix version (our anchor pair):</b> `x=[1,2]`, `W₁=[[1,2],[0,1]]`, `W₂=[[1,0],[3,1]]`. Then `(x W₁) W₂ = [13,4]`, and `W₁ W₂ = [[7,2],[3,1]]` so `x (W₁ W₂) = [13,4]` — identical. The two linear layers really are one matrix.
<br><b>4 · Sanity check:</b> now clip with a bend first — `ReLU(x W₁) W₂` — and any negative entries get zeroed <em>before</em> the second matmul, producing values no single matmul can. That clip is the entire reason activations exist.
!!!

!!! c-warn 😕
<b>为什么这里容易卡住 · Why this trips people up:</b> activation 看起来只是"最后随手加的一个函数"，很容易被当成不重要的一步 — the activation looks like an optional extra you tack on at the end. In plain terms: without it, stacking 100 layers is mathematically the same as having <b>one</b> layer — all that depth silently collapses. The activation is the single thing that makes "deep" worth anything.
!!!

@@@ concept id=c2 tag="The fix" title="A bend between the layers" gotit="Got the fix"
So how do you stop the collapse? You put a bend between the layers. That bend is the [[activation function||A simple function applied at the end of a neuron, after the weighted sum + bias. It adds non-linearity so stacked layers can learn complex shapes.]] — a small [[non-linearity||Not a straight line. A non-linear step between layers lets a deep network bend and combine features to learn complex patterns.]] applied at the very end of a neuron. Put a bend between each layer and the layers can no longer be squashed into one. That is the entire reason deep networks work.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Inserting ReLU between the two layers stops the collapse"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="30" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="57" y="65" fill="#5E5191">W₁</text><rect x="100" y="45" width="70" height="30" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="135" y="65" fill="#1a5c38" font-weight="bold">ReLU</text><rect x="185" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="212" y="65" fill="#5E5191">W₂</text><text x="275" y="65" fill="#6B645E">≠</text><rect x="305" y="45" width="150" height="30" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2" stroke-dasharray="4,3"/><text x="380" y="65" fill="#6B645E">any single W</text><text x="250" y="105" fill="#2D8B55" font-size="11">a bend in the middle → cannot be folded flat</text></g></svg>
%%%

#### The chain to remember
The logic is one short chain: `linear ∘ linear = linear` → depth alone buys nothing → a bend breaks it → the second layer builds on the first's non-linear features. Put a bend between W₁ and W₂ and the whole thing is no longer linear — it can't be collapsed into one matrix. Three common bends do this job: **ReLU**, **sigmoid**, and **tanh**. Let's meet each one.

@@@ concept id=c3 tag="Meet ReLU" title="ReLU — a one-way valve" gotit="Met ReLU"
The most common bend is the simplest. Think of a **one-way valve**: it lets positive signals flow straight through, and it blocks anything negative (sets it to 0). That is [[ReLU||A one-way valve: it lets positive numbers pass and turns negatives into 0.]] — simple, fast, and the default in almost every modern network.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="ReLU curve: flat at zero for negatives, a straight rising line for positives"><g font-family="monospace"><line x1="60" y1="95" x2="320" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="190" y1="25" x2="190" y2="110" stroke="#E5DFD6" stroke-width="1"/><path d="M70 95 L190 95 L300 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">negatives → 0</text><text x="260" y="30" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text x="400" y="70" font-size="13" fill="#1a5c38" font-family="monospace">ReLU(x)=max(0,x)</text></g></svg>
%%%

%%% demo id=relu label="run it"
code: relu(np.array([-3, -1, 0, 2, 5]))
out: array([0, 0, 0, 2, 5])
take: <b>ReLU = max(0, z).</b> It zeroes every negative and lets positives pass through unchanged. Left of zero it is flat (output 0); right of zero it is the line y = z. One cheap operation — this is the default activation in modern networks.
%%%

#### The formula
`ReLU(z) = max(0, z)` — keep positives, zero negatives; range `[0, ∞)`. It is one `max`, which makes it dirt cheap to compute. That speed is a big reason it became the modern default hidden-layer bend.

@@@ concept id=c4 tag="Meet sigmoid" title="Sigmoid — a soft dimmer" gotit="Met sigmoid"
The next bend is softer. Think of a **dimmer switch**: instead of hard on/off, it squashes any number gently into the range 0 to 1 — a smooth "off → on." That is [[sigmoid||A soft dimmer switch that gently squashes any number into the range 0 to 1.]]. It is great when you want a probability, but it flattens out at the extremes.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Sigmoid curve: an S shape rising smoothly from 0 to 1"><g font-family="monospace"><line x1="60" y1="95" x2="320" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="190" y1="25" x2="190" y2="110" stroke="#E5DFD6" stroke-width="1"/><path d="M70 92 C 150 92, 165 40, 300 38" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><text x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">→ near 0</text><text x="270" y="30" font-size="11" fill="#5E5191" text-anchor="middle">→ near 1</text><text x="400" y="70" font-size="12" fill="#5E5191" font-family="monospace">sigmoid → (0,1)</text></g></svg>
%%%

%%% demo id=sigmoid label="run it"
code: sigmoid(np.array([-2, 0, 2]))
out: array([0.119, 0.5  , 0.881])
take: <b>Sigmoid squashes into (0, 1).</b> A smooth "off → on": sigmoid(0)=0.5, big negatives → near 0, big positives → near 1 (the shown values are rounded — sigmoid(2)≈0.8808). Handy for a probability, but it flattens (saturates) at the extremes.
%%%

#### The formula
`sigmoid(z) = 1 / (1 + e^(−z))` — squash any number into `(0, 1)`. It is an S-curve: `sigmoid(0)=0.5`, and the two ends flatten toward 0 and 1. That flattening is a feature for probabilities but a problem for learning, as you'll see soon.

@@@ concept id=c5 tag="Meet tanh" title="tanh — the zero-centered cousin" gotit="Met tanh"
The third bend is sigmoid's **zero-centered cousin**. [[tanh||Squashes any number into (−1, 1), and passes through the origin: tanh(0)=0. Zero-centered, unlike sigmoid.]] is the same smooth S-curve shape, but it runs from `−1` to `+1` and passes through the origin, so a layer's outputs stay centered on 0. That centering helps the next layer learn. tanh was the default hidden-layer bend before ReLU took over, and it still shows up — for example in older recurrent networks.

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="tanh curve: an S shape through the origin, rising from near minus one up to near plus one"><g font-family="monospace"><line x1="60" y1="70" x2="460" y2="70" stroke="#E5DFD6" stroke-width="1.5"/><line x1="260" y1="16" x2="260" y2="124" stroke="#E5DFD6" stroke-width="1"/><path d="M70 120 C 190 118, 205 74, 260 70 C 315 66, 330 22, 450 20" fill="none" stroke="#9A5A12" stroke-width="2.5"/><text x="110" y="136" font-size="11" fill="#6B645E" text-anchor="middle">→ near −1</text><text x="410" y="15" font-size="11" fill="#9A5A12" text-anchor="middle">→ near +1</text><circle cx="260" cy="70" r="4" fill="#C93B3B"/><text x="300" y="62" font-size="11" fill="#C93B3B" text-anchor="middle">tanh(0)=0</text><text x="400" y="98" font-size="12" fill="#9A5A12">tanh → (−1, 1)</text></g></svg>
%%%

#### Three bends, side by side
Here are all three you'll see most, in one table.

%%% table
:: ReLU :: sigmoid :: tanh
Formula :: `max(0, z)` :: `1/(1+e⁻ᶻ)` :: `(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)`
Range :: `[0, ∞)` :: `(0, 1)` :: `(−1, 1)`
Zero-centered? :: no :: no (centers at 0.5) :: yes (centers at 0)
Slope near 0 :: `1` for z>0, `0` for z<0 :: ≈ `0.25` :: `1`
Saturates? :: only for z<0 (flat 0 → can "die") :: both tails (→ 0 and → 1) :: both tails (→ −1 and → 1)
Typical use :: hidden layers (modern default) :: binary output / gate :: hidden layers (older RNNs); zero-centered
%%%

@@@ concept id=c6 tag="When bends break" title="Dead ReLUs and saturation" gotit="Got the failure mode"
A bend can silently stop learning. This is the staff-level part of today: the activation you pick decides whether a neuron keeps improving. Don't take the "Slope near 0" row on faith — **watch the slope**. The dashed curve below is each activation's slope `f′(z)`; drag the marker into the flat tails and see it collapse toward `0`. That vanishing slope <em>is</em> saturation (sigmoid/tanh) and the dead-ReLU zero — the exact reason a stuck unit stops learning.

%%% viz src=../../viz/activation-derivatives.html title="activations and their slopes" caption="interactive — pick an activation, drag z; watch the slope flatten"
%%%

!!! c-warn ⚠️
<b>Failure mode (silent): saturated units that stop learning.</b> A `sigmoid` squashes into `(0, 1)`; when its input is far from zero (say `z = 10` or `z = −10`), the output flattens near `1` or `0` and barely moves when the input changes. A `ReLU` that always sees a negative input outputs a flat `0` — its one-way valve has jammed shut, and nothing flows through it again. That is a "dead" ReLU. In both cases the neuron's output is nearly constant, so training can no longer nudge it, and the network quietly learns less than it could. Nothing crashes; the loss just plateaus higher than expected. A senior engineer catches this in <b>code review</b> by logging activation statistics — the fraction of ReLUs stuck at zero, or how many sigmoids sit pinned near 0 or 1 — instead of only watching the loss number.
!!!

!!! c-info ⚖️
<b>Trade-off: ReLU's speed and sparsity vs. its dead-neuron risk.</b> Choosing `ReLU` buys you a cheap activation (just a `max`) that keeps gradients healthy in deep stacks and turns off many neurons at once, which is often useful. What you give up is safety at zero: any neuron pushed permanently negative dies and never comes back. Softer bends that never fully flatten (Leaky ReLU, GELU — a valve left with a permanent small leak: they keep a slight slope for negative inputs) trade a little extra math and less sparsity for never dying. Which one to use, and where, is a <b>design-review</b> decision — commonly ReLU-family in the hidden layers, sigmoid or softmax only at the output.
!!!

!!! c-ok 🎤
<b>Say this in an interview:</b> "An activation is the per-layer non-linearity. Without it `(x W₁) W₂ = x (W₁ W₂)` — the whole stack collapses to one linear map, so depth adds nothing. ReLU = max(0, z) is the cheap default and dodges vanishing gradients; sigmoid and softmax live at the output for probabilities. The silent failure is saturation — dead ReLUs stuck at 0, or sigmoids pinned near 0/1 — where the unit stops learning and the loss plateaus, so I log activation statistics, not just the loss."
!!!

@@@ concept id=c7 tag="The limit" title="One line can't do XOR" gotit="Got the limit"
Here is the simplest thing one straight line can never do. A single neuron is <b>one straight line</b> (Day 1). Ask it for [[XOR||XOR: output 1 when exactly one of two inputs is on, else 0. Its two classes sit on opposite diagonals, so no single straight line separates them.]] — "on when exactly one input is on" — and it can never get all four cases right, because XOR is not [[linearly separable||A dataset is linearly separable if one straight line (or flat plane) can put every class on its own side. XOR is the classic example that is NOT.]]. Slide the line any way you like below: the best you ever manage is 3 of 4. Then add a hidden layer — two lines, composed through a bend, carve out the region a single line never could.

%%% viz src=../../viz/xor-limit.html title="why one neuron can't do XOR" caption="interactive — slide the single line (stuck at 3/4), then add a hidden layer"
%%%

#### Why a frontier lab cares
`ReLU` is the classic workhorse: it is dirt cheap (just a `max`) and it dodges the [[vanishing-gradient||When gradients shrink toward zero as they pass back through many layers, so early layers barely learn. Sigmoid suffers from this; ReLU largely avoids it.]] problem that plagues sigmoid in deep stacks (you'll meet gradients on Day 5). It is also why different frontier labs pick different bends: GPT-2 and GPT-3 use **GELU**, and Llama uses **SwiGLU** — smoother curves that are one real answer to the dead-neuron risk from the last unit, staying slightly sloped where a plain ReLU goes flat. Sigmoid and its cousin [[softmax||Turns a vector of scores into probabilities that sum to 1. Used at the output of a classifier.]] live mostly at the output, where you want probabilities. The activation is the quiet reason "deep" learning is a thing at all.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four questions, instant feedback. Answer all four to complete today.
%%% quiz
q: What does an activation function add to a network? | a:1 | More parameters | Non-linearity (a bend) | Faster matmuls | A bias term | fb: The activation is the non-linear bend; without it, layers stay linear and depth is wasted.
q: With NO activation, ten stacked linear layers behave like: | a:1 | a much more powerful model | a single linear layer | a random function | a sigmoid | fb: (x W₁)W₂…W₁₀ = x (W₁…W₁₀) — one combined matrix. Depth buys nothing without a bend.
q: `ReLU(z)` equals: | a:1 | 1/(1+e^(−z)) | max(0, z) | z² | −z | fb: ReLU keeps positives and zeroes negatives: max(0, z).
q: Your network trains without error, but the loss stops dropping early and stays high. You log the hidden ReLU outputs and find most are exactly 0 for every input in the batch. Most likely explanation? | a:1 | Sigmoid at the output is always a bug | Many ReLU units are "dead": their input stays negative for all inputs, so max(0, z) is always 0 and they can no longer learn — the network is effectively much smaller than it looks | The GPU ran out of memory, which always makes ReLU output zero | A high loss with zero activations means the network has already converged | fb: A ReLU that always sees a negative input outputs a flat 0 for every example — a dead unit. Many dead units mean much of the network does nothing, so the loss plateaus. You look at activation statistics (fraction of zeros), not at a crash, because the failure is silent.
%%%

@@@ produce id=produce tag="Produce" title="Watch two layers collapse — then break the collapse" gotit="Done"
Here's the experiment worth running yourself. Build two linear layers with no bend between them and watch them fold into a single matrix — then drop a `ReLU` in the middle and watch that collapse break. **Predict** first: will `(x@W1)@W2` and `x@(W1@W2)` print the same numbers or different ones? Then run it and **observe**. Seeing them match, and then differ the moment the bend goes in, is the whole lesson in two lines of output. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-02-activations/experiment.py`. Write `relu`, `sigmoid`, and `tanh`, and print all three as a table over the grid `np.linspace(-6, 6, 13)`. **Notice** that ReLU zeros negatives, sigmoid stays in `(0,1)`, and tanh stays in `(−1,1)` and is 0 at 0. Also print `sigmoid'(0) ≈ 0.25` and note that ReLU's gradient is `0` for every negative input. Then use the anchor pair `W1=[[1,2],[0,1]]`, `W2=[[1,0],[3,1]]` and show that `(x@W1)@W2` equals `x@(W1@W2)` for a random `x` — print `W1@W2` (expect `[[7,2],[3,1]]`), proving two linear layers = one. Run with `python3 sessions/m02-the-neuron/day-02-activations/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 2 Day 2 artifact.

Create sessions/m02-the-neuron/day-02-activations/experiment.py that, with a comment on each step:
1. Defines relu(z)=np.maximum(0,z), sigmoid(z)=1/(1+np.exp(-z)), and tanh(z)=np.tanh(z); prints all three as a table over grid = np.linspace(-6,6,13) (ReLU zeros negatives, sigmoid in (0,1), tanh in (-1,1) and 0 at 0). Also prints sigmoid'(0)=0.25 and notes ReLU's gradient is 0 for negative inputs.
2. Shows the collapse: with W1=[[1,2],[0,1]], W2=[[1,0],[3,1]] and a random x, asserts np.allclose((x@W1)@W2, x@(W1@W2)) and prints W1@W2 (expect [[7,2],[3,1]]) — two linear layers equal one.
3. Then inserts relu between them: shows relu(x@W1)@W2 is NOT equal to any single x@W, i.e. the bend prevents the collapse.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the collapse, then the cure)
- Your `relu`, `sigmoid`, and `tanh` print sensible values over the grid `np.linspace(-6, 6, 13)`: ReLU zeros negatives, sigmoid stays inside `(0, 1)`, tanh stays inside `(−1, 1)` and is 0 at 0.
- You print `sigmoid'(0) ≈ 0.25` and note ReLU's gradient is `0` for negative inputs — the two facts behind saturation and "dead" (jammed-shut) ReLUs.
- The moment to **watch**: `(x@W1)@W2` prints the <em>same</em> numbers as `x@(W1@W2)` for a random `x` — two linear layers really did collapse to one — and then, once you slip a `ReLU` between them, the two stop matching. That break is the bend earning its keep.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-02-activations/log.md`: (1) why a 100-layer network with no activations is no better than one layer; (2) ReLU vs sigmoid in one line; (3) what a "dead" or "saturated" unit is and how you'd detect it.
!!!

@@@ fin
