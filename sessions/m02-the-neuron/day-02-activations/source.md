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
fin_body: "Nice work — you've completed <b>Activation Functions</b>. You just unlocked the one small idea that makes deep learning possible: the bend.<br>Next up: <b>Layers &amp; the Forward Pass</b>."
notebook_yardstick: 00-neural-networks/fundamentals/03_activation_functions.ipynb
---

@@@ hero
@lede Welcome back! Yesterday your neuron ended on a mysterious third step — the activation — and we left it hanging. Today we solve that mystery together. Here's the surprising part: without that one little step, stacking a hundred layers is no better than having just one. If that sounds strange right now, don't worry — by the end it will feel obvious.
@goal Together we'll meet the "bends" one at a time — the crude on/off switch people started with, then ReLU, sigmoid, and tanh. We'll see why depth without a bend is a mirage, watch a bend quietly "die," and meet the tiny fix that brings it back. And every time a formula shows up, we'll say it out loud in plain words first — no symbol left unexplained.

@@@ concept id=c1 tag="The problem" title="Straight + straight is still straight" gotit="Got the problem"
Let's start with a puzzle. A neuron does a weighted sum and adds a bias. In plain words, that is a straight-line rule — picture a perfectly straight **ruler**. Now here is the puzzle: what happens if you stack two straight rulers?

First, a quick friendly cheat-sheet. Here are the words we'll meet today, in plain English — just glance at it now, and come back whenever a term feels fuzzy.

%%% jargon
activation | the little "bend" added at the very end of a neuron, after the weighted sum
non-linearity | anything that is not a straight line; the bend that lets layers learn curved patterns
ReLU | a one-way valve: passes positive numbers, turns negatives into 0
sigmoid | a soft dimmer: squashes any number smoothly into the range 0 to 1
tanh | like sigmoid, but the range is −1 to 1 and it is centered on 0
step function | the oldest bend: a hard on/off switch (output 0 or 1)
slope | how steep a curve is at a point — how much the output moves when you nudge the input
saturation | when a curve goes flat, so its slope is ≈ 0 and the neuron stops responding
dead ReLU | a ReLU stuck outputting 0 forever, so it can no longer learn
sparsity | many neurons output exactly 0 at once — a useful "only the ones that matter speak up"
vanishing gradient | learning signals shrink toward zero as they pass back through many layers
XOR | "on when exactly one of two inputs is on" — the classic pattern one straight line can't split
%%%

Now, the puzzle's answer. Lay one straight ruler on top of another and you still get… a single straight ruler. Two neuron-layers with no bend between them fold back into one layer. So all that extra depth buys you nothing.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Two linear layers W1 then W2 collapse into a single matrix W1 times W2"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="40" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="70" y="65" fill="#5E5191">W₁</text><text x="115" y="65" fill="#6B645E">then</text><rect x="150" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="180" y="65" fill="#5E5191">W₂</text><text x="250" y="65" fill="#6B645E">=</text><rect x="290" y="40" width="140" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="360" y="65" fill="#C93B3B" font-weight="bold">one matrix W₁@W₂</text><text x="260" y="105" fill="#C93B3B" font-size="11">no activation → two layers become one</text></g></svg>
%%%

**What the ruler picture gets right:** each layer really is a straight-line rule, and laying straight rules end to end keeps things straight. **Where it breaks down:** a ruler is one fixed line, while a layer has many weights it can tune — but no matter how it tunes them, the *shape* stays straight. That is the part that matters.

#### Why depth alone buys nothing
Here is the same idea in one line. Stacking two linear layers gives `(x@W1)@W2 = x@(W1@W2)`. The piece `W1@W2` is just another single matrix. So the two layers are secretly one. Ten layers, a hundred — still one straight-line layer. Depth without a bend is an illusion.

If that feels slippery, you're in good company — almost everyone finds "the layers collapse" strange the first time. The little worked example below makes it click.

!!! c-info 🪜
<b>Math Ladder — why linear ∘ linear collapses.</b>
<br><b>1 · In words:</b> two plain matmuls back-to-back, with no bend between them, do the same job as one matmul — so the second layer bought you nothing.
<br><b>2 · Scalars first (no matrix algebra needed):</b> chain two linear steps `f(x)=a·x+p` then `g(u)=b·u+q`. Compose them: `g(f(x)) = b·(a·x+p)+q = (ab)·x + (bp+q)` — still `slope·x + shift`, i.e. one linear step. Two linear layers with no bend collapse to one.
<br><b>3 · Now the matrix version (our anchor pair):</b> `x=[1,2]`, `W₁=[[1,2],[0,1]]`, `W₂=[[1,0],[3,1]]`. Then `(x W₁) W₂ = [13,4]`, and `W₁ W₂ = [[7,2],[3,1]]` so `x (W₁ W₂) = [13,4]` — identical. The two linear layers really are one matrix.
<br><b>4 · Sanity check:</b> now clip with a bend first — `ReLU(x W₁) W₂` — and any negative entries get zeroed <em>before</em> the second matmul, producing values no single matmul can. That clip is the entire reason activations exist.
!!!

!!! c-warn 😕
<b>这一步很多人一开始都会觉得奇怪，很正常 · This step feels strange to almost everyone at first — that's completely normal.</b> The activation looks like an optional extra you tack on at the end, so it's easy to shrug off. In plain terms: without it, stacking 100 layers is mathematically the same as having <b>one</b> layer — all that depth silently folds flat. The activation is the single thing that makes "deep" worth anything. Sit with that for a moment; it's the whole reason today matters.
!!!

@@@ concept id=c2 tag="The fix" title="A bend between the layers" gotit="Got the fix"
So how do we stop the collapse? We put a **bend** between the layers. That bend is the [[activation function||A simple function applied at the end of a neuron, after the weighted sum + bias. It adds non-linearity so stacked layers can learn complex shapes.]] — a small [[non-linearity||Not a straight line. A non-linear step between layers lets a deep network bend and combine features to learn complex patterns.]] applied at the very end of a neuron.

Think of it like a **crease in a sheet of paper**. Flat paper laid on flat paper stays flat. But fold a crease into it first, and now you can build shapes a flat sheet never could. Put a bend between each layer and the layers can no longer be squashed into one.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Inserting ReLU between the two layers stops the collapse"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="30" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="57" y="65" fill="#5E5191">W₁</text><rect x="100" y="45" width="70" height="30" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="135" y="65" fill="#1a5c38" font-weight="bold">ReLU</text><rect x="185" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="212" y="65" fill="#5E5191">W₂</text><text x="275" y="65" fill="#6B645E">≠</text><rect x="305" y="45" width="150" height="30" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2" stroke-dasharray="4,3"/><text x="380" y="65" fill="#6B645E">any single W</text><text x="250" y="105" fill="#2D8B55" font-size="11">a bend in the middle → cannot be folded flat</text></g></svg>
%%%

**What the paper-crease picture gets right:** one fold unlocks shapes a flat sheet can't make, just as one bend unlocks patterns a straight-line stack can't. **Where it breaks down:** paper only creases where you fold it, while an activation bends the signal at *every* neuron, on every example — far more folds than you could make by hand.

And that's the whole secret. **You just unlocked why deep learning is even possible.** The rest of today is meeting the bends themselves — starting with the very first one anyone tried.

@@@ concept id=c3 tag="The first bend" title="The step function — a hard on/off switch" gotit="Met the step"
The first activation people used was the simplest bend you can imagine: a hard **on/off switch**. Picture a light switch on a wall. Flip it and the light is either fully **on** or fully **off** — nothing in between. That is the [[step function||The original activation: output 1 if the input is at or above zero, else 0. A hard on/off switch with no in-between — and slope 0 everywhere, so a network can't learn from it.]].

In words: if the input is at or above zero, output `1` (the neuron "fires"). If it's below zero, output `0` (it stays quiet). In symbols, `step(z) = 1 if z ≥ 0, else 0`.

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Step function: output 0 for negative inputs, jumps to 1 at zero, then flat at 1 for positive inputs"><g font-family="monospace"><line x1="60" y1="100" x2="330" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><line x1="195" y1="30" x2="195" y2="118" stroke="#E5DFD6" stroke-width="1"/><path d="M70 100 L195 100 L195 48 L305 48" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="195" cy="48" r="3.5" fill="#8A6D3B"/><text x="125" y="117" font-size="11" fill="#6B645E" text-anchor="middle">output 0 (off)</text><text x="255" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">output 1 (on)</text><text x="415" y="72" font-size="12" fill="#8A6D3B">step → {0, 1}</text><text x="195" y="133" font-size="10" fill="#9A938A" text-anchor="middle">jump at 0 — no in-between, and flat (slope 0) everywhere else</text></g></svg>
%%%

%%% demo id=step label="run it"
code: step(np.array([-2, -1, 0, 1, 2]))
out: array([0, 0, 1, 1, 1])
take: <b>Step = 1 if z ≥ 0 else 0.</b> A hard on/off switch: everything below zero is 0, everything at or above zero is 1. Notice there is no "how much" — only "yes" or "no."
%%%

**What the light-switch picture gets right:** it's the perfect image for all-or-nothing, no middle ground. **Where it breaks down:** a wall switch is flipped by *you*, but here the *input* decides — and unlike a dimmer, there's no in-between, which turns out to be its fatal flaw.

#### Why we moved on from it
The step function looks tidy, so why doesn't anyone use it in hidden layers today? Because it is **flat everywhere** — its slope is `0` on both sides of the jump. Learning works by nudging weights in the direction that lowers the error, and a flat function gives no hint about which way to nudge: tweak the input a little and the output doesn't budge at all. So the network gets no signal to improve. The step function is a piece of history — the original 1950s "perceptron" used it — and its flatness is exactly *why* the smoother, slope-carrying bends we meet next took over.

@@@ concept id=c4 tag="Meet ReLU" title="ReLU — a one-way valve" gotit="Met ReLU"
The bend that fixed all that is almost as simple as the switch — but it keeps a useful slope. Picture a **one-way valve** in a pipe: it lets water flow through in the forward direction, and blocks it the other way. That is [[ReLU||A one-way valve: it lets positive numbers pass and turns negatives into 0. The modern default activation.]]: it lets positive signals pass straight through, and blocks anything negative (sets it to 0).

In symbols: `ReLU(z) = max(0, z)`. Read it out loud: "take the bigger of 0 and z." If `z` is positive, `z` wins and passes through. If `z` is negative, `0` wins. That's the whole rule.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="ReLU curve: flat at zero for negatives, a straight rising line for positives"><g font-family="monospace"><line x1="60" y1="95" x2="320" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="190" y1="25" x2="190" y2="110" stroke="#E5DFD6" stroke-width="1"/><path d="M70 95 L190 95 L300 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">negatives → 0</text><text x="260" y="30" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text x="400" y="70" font-size="13" fill="#1a5c38" font-family="monospace">ReLU(x)=max(0,x)</text></g></svg>
%%%

%%% demo id=relu label="run it"
code: relu(np.array([-3, -1, 0, 2, 5]))
out: array([0, 0, 0, 2, 5])
take: <b>ReLU = max(0, z).</b> It zeroes every negative and lets positives pass through unchanged. Left of zero it is flat (output 0); right of zero it is the line y = z.
%%%

**What the valve picture gets right:** flow one way, blocked the other — exactly ReLU's "positives pass, negatives stop." **Where it breaks down:** a real valve is either open or shut, but ReLU passes positives at their *full* size (5 stays 5, 100 stays 100) rather than a fixed trickle. Its range is `[0, ∞)` — zero and up.

Here's the payoff: unlike the flat step function, ReLU's positive side has a real slope of `1`, so learning gets a clear signal there. Cheap to compute (just one `max`) and a usable slope — that combination is why ReLU is the default bend in almost every modern network.

@@@ concept id=c5 tag="Meet sigmoid" title="Sigmoid — a soft dimmer" gotit="Met sigmoid"
The next bend is gentler. Instead of the hard on/off of the step function, picture a **dimmer switch**: it slides smoothly from off to fully on, passing through every level of brightness in between. That is [[sigmoid||A soft dimmer switch that gently squashes any number into the range 0 to 1.]]. It takes any number and squashes it smoothly into the range 0 to 1.

In symbols: `sigmoid(z) = 1 / (1 + e^(−z))`. In words: one, divided by (one plus e-to-the-minus-z). You don't need to compute `e` by hand — the only thing to remember is that the answer always lands between 0 and 1, and `sigmoid(0) = 0.5` sits right in the middle.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Sigmoid curve: an S shape rising smoothly from 0 to 1"><g font-family="monospace"><line x1="60" y1="95" x2="320" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="190" y1="25" x2="190" y2="110" stroke="#E5DFD6" stroke-width="1"/><path d="M70 92 C 150 92, 165 40, 300 38" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><text x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">→ near 0</text><text x="270" y="30" font-size="11" fill="#5E5191" text-anchor="middle">→ near 1</text><text x="400" y="70" font-size="12" fill="#5E5191" font-family="monospace">sigmoid → (0,1)</text></g></svg>
%%%

%%% demo id=sigmoid label="run it"
code: sigmoid(np.array([-2, 0, 2]))
out: array([0.119, 0.5  , 0.881])
take: <b>Sigmoid squashes into (0, 1).</b> A smooth "off → on": sigmoid(0)=0.5, big negatives land near 0, big positives near 1 (the values are rounded — sigmoid(2)≈0.8808). Because it lives between 0 and 1, it's handy when you want a probability.
%%%

**What the dimmer picture gets right:** the smooth, gradual slide from off to on, no hard jump. **Where it breaks down:** a real dimmer is evenly graded across its whole travel, but sigmoid *flattens out* at both ends — push the input far in either direction and the output barely changes. Hold onto that flattening; it becomes a real problem for learning, and we'll see exactly why in a couple of units.

@@@ concept id=c6 tag="Meet tanh" title="tanh — the zero-centered cousin" gotit="Met tanh"
The third smooth bend is sigmoid's close **cousin**. [[tanh||Squashes any number into (−1, 1), and passes through the origin: tanh(0)=0. Zero-centered, unlike sigmoid.]] has the same gentle S-shape, but it runs from `−1` to `+1` and passes through the middle at zero (so `tanh(0) = 0`). Because its outputs are balanced around 0, we say it is **zero-centered**, and that balance helps the next layer learn. tanh was the go-to hidden-layer bend before ReLU took over, and it still shows up today — for example inside older recurrent networks.

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="tanh curve: an S shape through the origin, rising from near minus one up to near plus one"><g font-family="monospace"><line x1="60" y1="70" x2="460" y2="70" stroke="#E5DFD6" stroke-width="1.5"/><line x1="260" y1="16" x2="260" y2="124" stroke="#E5DFD6" stroke-width="1"/><path d="M70 120 C 190 118, 205 74, 260 70 C 315 66, 330 22, 450 20" fill="none" stroke="#9A5A12" stroke-width="2.5"/><text x="110" y="136" font-size="11" fill="#6B645E" text-anchor="middle">→ near −1</text><text x="410" y="15" font-size="11" fill="#9A5A12" text-anchor="middle">→ near +1</text><circle cx="260" cy="70" r="4" fill="#C93B3B"/><text x="300" y="62" font-size="11" fill="#C93B3B" text-anchor="middle">tanh(0)=0</text><text x="400" y="98" font-size="12" fill="#9A5A12">tanh → (−1, 1)</text></g></svg>
%%%

**What the "cousin" framing gets right:** same S-shape family, same smooth squashing. **Where it breaks down:** tanh isn't just a shifted sigmoid — it's stretched to reach both −1 and +1, and it shares sigmoid's habit of flattening at the ends.

#### So far, a quick breath
We've met four bends now: the crude **step** (flat, can't learn), the **ReLU** valve (cheap, keeps a slope for positives), the **sigmoid** dimmer (smooth 0→1), and its **tanh** cousin (smooth −1→1). Here they are side by side — a reference you can come back to.

%%% table
:: ReLU :: sigmoid :: tanh
Formula :: `max(0, z)` :: `1/(1+e⁻ᶻ)` :: `(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)`
Range :: `[0, ∞)` :: `(0, 1)` :: `(−1, 1)`
Zero-centered? :: no :: no (centers at 0.5) :: yes (centers at 0)
Slope near 0 :: `1` for z>0, `0` for z<0 :: ≈ `0.25` :: `1`
Saturates? :: only for z<0 (flat 0 → can "die") :: both tails (→ 0 and → 1) :: both tails (→ −1 and → 1)
Typical use :: hidden layers (modern default) :: binary output / gate :: hidden layers (older RNNs); zero-centered
%%%

#### The three bends — and their slopes — on one picture
The bottom half of this picture shows the **slope** of each bend, and it's the one idea to carry into the next unit.

%%% svg
<svg viewBox="0 0 520 380" role="img" aria-label="Top panel: ReLU, sigmoid and tanh curves overlaid on shared axes. Bottom panel: their slopes f prime of z overlaid, sigmoid slope peaking near 0.25 and tanh slope peaking near 1.0."><g font-family="monospace" font-size="12">
<!-- ===== TOP PANEL: the three curves ===== -->
<text x="60" y="22" font-size="13" fill="#3A342E" font-weight="bold">The bend  f(z)</text>
<!-- axes -->
<line x1="60" y1="110" x2="470" y2="110" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="265" y1="36" x2="265" y2="160" stroke="#E5DFD6" stroke-width="1"/>
<text x="475" y="114" fill="#9A938A" font-size="10" text-anchor="start">z</text>
<!-- tanh: S through origin, from near -1 up to near +1 (brown #9A5A12) -->
<path d="M70 155 C 190 153, 225 112, 265 110 C 305 108, 340 67, 460 65" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<!-- sigmoid: S from near 0 (bottom) up to near 1, passing through 0.5 at z=0 (purple #7C6DAA) -->
<path d="M70 152 C 190 150, 230 90, 265 87 C 300 84, 340 46, 460 44" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<!-- ReLU: flat 0 for negatives, rising line for positives (green #2D8B55) -->
<path d="M70 110 L265 110 L440 46" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<!-- curve labels -->
<text x="450" y="60" fill="#2D8B55" text-anchor="end" font-weight="bold">ReLU</text>
<text x="405" y="38" fill="#7C6DAA" text-anchor="start">sigmoid</text>
<text x="405" y="78" fill="#9A5A12" text-anchor="start">tanh</text>
<!-- ===== BOTTOM PANEL: the slopes ===== -->
<text x="60" y="212" font-size="13" fill="#3A342E" font-weight="bold">The slope  f′(z)</text>
<!-- axes -->
<line x1="60" y1="340" x2="470" y2="340" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="265" y1="230" x2="265" y2="352" stroke="#E5DFD6" stroke-width="1"/>
<text x="475" y="344" fill="#9A938A" font-size="10" text-anchor="start">z</text>
<!-- peak reference lines -->
<line x1="60" y1="248" x2="470" y2="248" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="252" fill="#9A5A12" font-size="10" text-anchor="end">1.0</text>
<line x1="60" y1="317" x2="470" y2="317" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="321" fill="#7C6DAA" font-size="10" text-anchor="end">0.25</text>
<!-- ReLU slope: 0 for z<0 (flat on axis), 1 for z>0 (flat at the 1.0 line) with a step at 0 (green) -->
<path d="M70 340 L265 340 L265 248 L460 248" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<!-- tanh slope: bell peaking ~1.0 at z=0 (brown) -->
<path d="M70 339 C 200 338, 235 252, 265 248 C 295 252, 330 338, 460 339" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<!-- sigmoid slope: lower, flatter bell peaking ~0.25 at z=0 (purple) -->
<path d="M70 340 C 205 339, 240 319, 265 317 C 290 319, 325 339, 460 340" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<!-- slope labels + peak annotations -->
<text x="272" y="244" fill="#9A5A12" font-size="11" text-anchor="start">tanh′ peaks ≈ 1.0</text>
<text x="272" y="313" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid′ peaks ≈ 0.25</text>
<text x="450" y="240" fill="#2D8B55" text-anchor="end" font-weight="bold">ReLU′ = 1 (z&gt;0)</text>
<text x="130" y="333" fill="#2D8B55" font-size="10" text-anchor="middle">ReLU′ = 0 (z&lt;0)</text>
</g></svg>
%%%

#### What is a "slope," in plain words?
Take a breath — this is the one word that unlocks the rest of the day. The **slope** of a bend at a point is just *how steep the curve is right there*. Put another way: how much does the output move when you nudge the input a tiny bit? A steep slope means the neuron reacts strongly. A slope near `0` — a flat stretch — means the neuron barely reacts, so learning there crawls to a stop. Look at the bottom panel: sigmoid's slope never climbs above about `0.25`, tanh's reaches about `1.0`, and ReLU's is a flat `0` for negatives or a plain `1` for positives. (The exact recipe for a slope is called the *derivative* — you'll do that math on Day 5. Today the picture is all you need.) Keep this in your pocket: the next unit is about what happens when that slope goes flat.

@@@ concept id=c7 tag="When bends break" title="Dead ReLUs and saturation" gotit="Got the failure mode"
Here's something that surprises a lot of people: a bend can quietly *stop learning*, without any error or crash. The bend you pick decides whether a neuron keeps improving — so this is the part that separates a working model from a stuck one. And you don't have to take the "Slope near 0" row on faith. **Watch the slope yourself.** Drag the marker below into the flat tails of each curve and see the slope collapse toward `0`.

%%% viz src=../../viz/activation-derivatives.html title="activations and their slopes" caption="interactive — pick an activation, drag z; watch the slope flatten"
%%%

When the slope goes flat, learning stalls. That flat-tail stall has two everyday names, and here's what each looks like:

!!! c-warn ⚠️
<b>Two ways a bend goes quiet (and how to spot it).</b> <b>Saturation:</b> a `sigmoid` squashes into `(0, 1)`; when its input is far from zero (say `z = 10` or `z = −10`), the output flattens near `1` or `0` and barely moves — its slope is ≈ 0. <b>Dead ReLU:</b> a `ReLU` that always sees a negative input outputs a flat `0` — its one-way valve has jammed shut and nothing flows through it again (this is most often triggered by too high a learning rate, which shoves the unit's input permanently negative). In both cases the neuron's output is nearly constant, so training can no longer nudge it, and the network quietly learns less than it could. Nothing crashes — the loss just plateaus higher than you'd expect. The way a senior engineer catches this: log activation statistics (the fraction of ReLUs stuck at zero, or how many sigmoids sit pinned near 0 or 1) instead of only watching the loss number. That habit is a small superpower.
!!!

@@@ concept id=c8 tag="The cure" title="Leaky ReLU — a valve with a small leak" gotit="Got the cure"
A dead ReLU sounds bad — so here's the good news: one tiny tweak fixes it. Picture the same one-way valve as ReLU, but this time it's left with a **tiny permanent leak**. Positives still flow straight through. Negatives, instead of being shut off completely, are let through as a small trickle. That is [[Leaky ReLU||Like ReLU, but negatives get a small slope (α·z, e.g. α=0.01) instead of a flat 0 — so the unit keeps a tiny slope on the negative side and can't fully die.]].

Why does a leak help? Because the negative side now has a small but *non-zero* slope. So a unit that drifts negative still gets a learning signal, and it can climb back to life instead of dying. In symbols: `leaky(z) = z` for positives, or `α·z` for negatives, where `α` (the Greek letter "alpha") is a small number like `0.01`. In words: negatives aren't zeroed — they're shrunk to a tiny trickle.

%%% svg
<svg viewBox="0 0 520 145" role="img" aria-label="Leaky ReLU: a gentle downward slope for negative inputs instead of a flat zero, and a rising line for positive inputs; a dashed line marks where plain ReLU would be flat at zero"><g font-family="monospace"><line x1="60" y1="92" x2="330" y2="92" stroke="#E5DFD6" stroke-width="1.5"/><line x1="195" y1="24" x2="195" y2="116" stroke="#E5DFD6" stroke-width="1"/><path d="M70 92 L195 92" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><path d="M70 104 L195 92 L305 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="120" y="120" font-size="11" fill="#2D8B55" text-anchor="middle">tiny leak (slope α)</text><text x="118" y="84" font-size="10" fill="#C93B3B" text-anchor="middle">ReLU: flat 0 (can die)</text><text x="258" y="34" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text x="415" y="70" font-size="12" fill="#1a5c38">leaky(z)=max(αz, z)</text></g></svg>
%%%

%%% demo id=leaky label="run it"
code: leaky_relu(np.array([-5, -2, 0, 2, 5]), alpha=0.01)
out: array([-0.05, -0.02,  0.  ,  2.  ,  5.  ])
take: <b>Leaky ReLU = max(αz, z), with α small (≈0.01).</b> Positives are identical to ReLU; negatives become a small trickle (−5 → −0.05) instead of a flat 0. That trickle keeps a slope on the negative side, so the unit can't get permanently stuck.
%%%

**What the "leaky valve" picture gets right:** the tiny leak is exactly the small negative slope that keeps the neuron alive. **Where it breaks down:** a real leak is an accident you'd want to fix, but here the leak is the *feature* — deliberately built in.

A couple of close relatives take the same idea further: **PReLU** *learns* the best `α` during training, and **GELU** / **ELU** use a smooth curve on the negative side instead of a straight leak. They all trade a hair more computation for a bend that never fully dies.

!!! c-info ⚖️
<b>Trade-off: ReLU's speed and sparsity vs. its dead-neuron risk.</b> Choosing `ReLU` buys you a cheap activation (just a `max`) that keeps gradients healthy in deep stacks and turns off many neurons at once — a useful <b>sparsity</b> (lots of exact zeros, so each layer leans on only the units that matter). What you give up is safety at zero: any neuron pushed permanently negative dies and never comes back. Softer bends that never fully flatten (Leaky ReLU, GELU — a valve with a permanent small leak) trade a little extra math and less sparsity for never dying. Which one to use, and where, is a <b>design-review</b> decision — usually ReLU-family in the hidden layers, sigmoid or softmax only at the output.
!!!

!!! c-ok 🎤
<b>Once it clicks, here's how you'd explain it (say, in an interview):</b> "An activation is the per-layer non-linearity. Without it `(x W₁) W₂ = x (W₁ W₂)` — the whole stack collapses to one straight-line map, so depth adds nothing. ReLU = max(0, z) is the cheap default and dodges vanishing gradients; sigmoid and softmax live at the output for probabilities. The silent failure is saturation — dead ReLUs stuck at 0, or sigmoids pinned near 0/1 — where the slope goes to zero and the unit stops learning, so I log activation statistics, not just the loss. If units are dying I reach for Leaky ReLU or GELU, which keep a small slope on the negative side." Notice you can already say every piece of that — that's how far you've come today.
!!!

@@@ concept id=c9 tag="The limit" title="One line can't do XOR" gotit="Got the limit"
Let's end with the puzzle that started this whole field — and see why bends matter, from the other side. A single neuron draws **one straight line** (that was Day 1). Now ask it for [[XOR||XOR: output 1 when exactly one of two inputs is on, else 0. Its two classes sit on opposite diagonals, so no single straight line separates them.]] — "on when exactly one of two inputs is on." Try as you might, one straight line can never get all four cases right, because XOR is not [[linearly separable||A dataset is linearly separable if one straight line (or flat plane) can put every class on its own side. XOR is the classic example that is NOT.]]. Slide the line any way you like below — the best you'll ever manage is 3 of 4.

%%% viz src=../../viz/xor-limit.html title="why one neuron can't do XOR" caption="interactive — slide the single line (stuck at 3/4), then add a hidden layer"
%%%

Then flip on the hidden layer. Two lines, joined through a bend, carve out the region a single line never could. That's the same lesson as the whole day, seen from a new angle: the bend is what lets simple pieces combine into something a straight line can't reach.

#### Why a frontier lab cares
`ReLU` is the classic workhorse: dirt cheap (just a `max`), and it sidesteps the [[vanishing-gradient||When gradients shrink toward zero as they pass back through many layers, so early layers barely learn. Sigmoid suffers from this; ReLU largely avoids it.]] problem that plagues sigmoid in deep stacks (you'll meet gradients properly on Day 5). It's also why different labs pick different bends: GPT-2 and GPT-3 use **GELU**, and Llama uses **SwiGLU** — smoother versions of the same "keep a small slope so the bend never dies" idea you just met with Leaky ReLU. Sigmoid and its cousin [[softmax||Turns a vector of scores into probabilities that sum to 1. Used at the output of a classifier.]] live mostly at the output, where you want probabilities. The activation really is the quiet reason "deep" learning works at all — and now you know why.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does an activation function add to a network? | a:1 | More parameters | Non-linearity (a bend) | Faster matmuls | A bias term | fb: The activation is the non-linear bend; without it, layers stay linear and depth is wasted.
q: With NO activation, ten stacked linear layers behave like: | a:1 | a much more powerful model | a single linear layer | a random function | a sigmoid | fb: (x W₁)W₂…W₁₀ = x (W₁…W₁₀) — one combined matrix. Depth buys nothing without a bend.
q: Why did the old step function fall out of use for hidden layers? | a:2 | It was too slow to compute | It only worked on images | It is flat everywhere (slope 0), so it gives no signal to learn from | It outputs negative numbers | fb: The step function's slope is 0 on both sides, so nudging the input never moves the output — the network gets no direction to improve. Smooth/valve bends keep a usable slope.
q: Your network trains without error, but the loss stops dropping early and stays high. You log the hidden ReLU outputs and find most are exactly 0 for every input in the batch. Most likely explanation — and a fix? | a:1 | Sigmoid at the output is always a bug | Many ReLU units are "dead": their input stays negative for all inputs, so max(0, z) is always 0 and they can no longer learn — switching those units to Leaky ReLU keeps a small negative slope so they can recover | The GPU ran out of memory, which always makes ReLU output zero | A high loss with zero activations means the network has already converged | fb: A ReLU that always sees a negative input outputs a flat 0 for every example — a dead unit. Many dead units mean much of the network does nothing, so the loss plateaus. You spot it from activation statistics (fraction of zeros), and Leaky ReLU (a small negative slope) is the standard cure.
%%%

@@@ produce id=produce tag="Produce" title="Watch two layers collapse — then break the collapse" gotit="Done"
Time to see today's big idea with your own eyes. You'll build two linear layers with no bend between them and watch them fold into a single matrix — then drop a `ReLU` in the middle and watch that collapse break. **Predict first:** will `(x@W1)@W2` and `x@(W1@W2)` print the same numbers, or different ones? Then run it and **watch.** Seeing them match, and then differ the moment the bend goes in, is the whole lesson in two lines of output. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-02-activations/experiment.py`. Write `step`, `relu`, `leaky_relu`, `sigmoid`, and `tanh`, and print all five as a table over the grid `np.linspace(-6, 6, 13)`. **Notice** that the step is a flat 0/1 jump, ReLU zeros negatives, leaky_relu lets a small trickle through the negatives, sigmoid stays in `(0,1)`, and tanh stays in `(−1,1)` and is 0 at 0. Also print `sigmoid'(0) ≈ 0.25` and note that ReLU's slope is `0` for every negative input while leaky_relu's is a small `α`. Then use the anchor pair `W1=[[1,2],[0,1]]`, `W2=[[1,0],[3,1]]` and show that `(x@W1)@W2` equals `x@(W1@W2)` for a random `x` — print `W1@W2` (expect `[[7,2],[3,1]]`), proving two linear layers = one. Run with `python3 sessions/m02-the-neuron/day-02-activations/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 2 Day 2 artifact.

Create sessions/m02-the-neuron/day-02-activations/experiment.py that, with a comment on each step:
1. Defines step(z)=(z>=0).astype(float), relu(z)=np.maximum(0,z), leaky_relu(z,a=0.01)=np.where(z>0,z,a*z), sigmoid(z)=1/(1+np.exp(-z)), and tanh(z)=np.tanh(z); prints all five as a table over grid = np.linspace(-6,6,13) (step is a 0/1 jump, ReLU zeros negatives, leaky_relu lets a small trickle through, sigmoid in (0,1), tanh in (-1,1) and 0 at 0). Also prints sigmoid'(0)=0.25 and notes ReLU's slope is 0 for negative inputs while leaky_relu's is a small alpha.
2. Shows the collapse: with W1=[[1,2],[0,1]], W2=[[1,0],[3,1]] and a random x, asserts np.allclose((x@W1)@W2, x@(W1@W2)) and prints W1@W2 (expect [[7,2],[3,1]]) — two linear layers equal one.
3. Then inserts relu between them: shows relu(x@W1)@W2 is NOT equal to any single x@W, i.e. the bend prevents the collapse.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the collapse, then the cure)
- Your five bends print sensible values over the grid `np.linspace(-6, 6, 13)`: the step is a flat 0/1 jump, ReLU zeros negatives, leaky_relu lets a small trickle through the negatives, sigmoid stays inside `(0, 1)`, tanh stays inside `(−1, 1)` and is 0 at 0.
- You print `sigmoid'(0) ≈ 0.25` and note ReLU's slope is `0` for negative inputs (the seed of "dead" ReLUs) while leaky_relu keeps a small slope `α` there — the seed of the cure.
- The moment to **watch**: `(x@W1)@W2` prints the <em>same</em> numbers as `x@(W1@W2)` for a random `x` — two linear layers really did collapse to one — and then, once you slip a `ReLU` between them, the two stop matching. That break is the bend earning its keep.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-02-activations/log.md`: (1) why a 100-layer network with no activations is no better than one layer; (2) ReLU vs sigmoid in one line; (3) what a "dead" or "saturated" unit is, how you'd detect it, and the one-line fix.
!!!

@@@ fin
