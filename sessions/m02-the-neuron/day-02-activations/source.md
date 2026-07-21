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
@lede Welcome back! Here's a tiny experiment you can picture right now: try to draw a smiley face using only a **ruler**. You can't — a ruler makes straight lines, and a face is all curves. To draw anything real, your hand has to *bend*. A neural network hits the exact same wall, and today you'll meet its fix: a small step called the **activation** — the "bend." Yesterday your neuron ended on this mysterious third step and we left it hanging; today we solve it together. And here's the twist that surprises almost everyone: without that one little bend, stacking a hundred layers is no better than having just one. It's also the quiet trick that lets something like ChatGPT curve its way to an answer. Sounds strange? Good — hold that feeling; by the end it'll feel obvious, and a little bit magic.
@goal Together we'll meet the "bends" one at a time — the crude on/off switch people started with, then ReLU, sigmoid, and tanh. You'll see why depth without a bend is a mirage, catch a bend quietly "dying" in the act, and pocket the tiny fixes that bring learning back. Every failure we meet is a little puzzle with a neat answer — and every time a formula shows up, we say it out loud in plain words first. No symbol left unexplained.

@@@ concept id=c1 tag="The problem" title="Straight + straight is still straight" gotit="Got the problem"
Let's start with a puzzle. A neuron does a weighted sum and adds a bias. In plain words, that is a straight-line rule — picture a perfectly straight **ruler**. Now here is the puzzle: what happens if you stack two straight rulers?

Now, the puzzle's answer. Lay one straight ruler on top of another and you still get… a single straight ruler. Two neuron-layers with no bend between them fold back into one layer. So all that extra depth buys you nothing. Engineers have a name for this fold-back — the **linear collapse** — and beating it is the whole reason today exists.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Two straight rulers laid one on top of the other still make a single straight edge. The everyday picture for two straight-line layers folding into one."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Two straight rulers stacked = still one straight edge</text><rect x="40" y="42" width="260" height="20" rx="3" fill="#FCF3DC" stroke="#C99A12"/><line x1="60" y1="52" x2="280" y2="52" stroke="#9A7A10" stroke-width="1"/><line x1="70" y1="46" x2="70" y2="58" stroke="#9A7A10"/><line x1="110" y1="46" x2="110" y2="58" stroke="#9A7A10"/><line x1="150" y1="46" x2="150" y2="58" stroke="#9A7A10"/><line x1="190" y1="46" x2="190" y2="58" stroke="#9A7A10"/><line x1="230" y1="46" x2="230" y2="58" stroke="#9A7A10"/><text x="312" y="56" fill="#6B645E">ruler 1</text><rect x="40" y="66" width="260" height="20" rx="3" fill="#EFEAF7" stroke="#7C6DAA"/><line x1="60" y1="76" x2="280" y2="76" stroke="#5E5191" stroke-width="1"/><line x1="70" y1="70" x2="70" y2="82" stroke="#5E5191"/><line x1="110" y1="70" x2="110" y2="82" stroke="#5E5191"/><line x1="150" y1="70" x2="150" y2="82" stroke="#5E5191"/><line x1="190" y1="70" x2="190" y2="82" stroke="#5E5191"/><line x1="230" y1="70" x2="230" y2="82" stroke="#5E5191"/><text x="312" y="80" fill="#6B645E">ruler 2</text><text x="260" y="110" text-anchor="middle" fill="#6B645E">↓ stack them ↓</text><rect x="40" y="120" width="260" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><line x1="60" y1="131" x2="280" y2="131" stroke="#C93B3B" stroke-width="1.4"/><text x="316" y="135" fill="#C93B3B">still ONE straight edge</text><text x="170" y="158" text-anchor="middle" fill="#C93B3B" font-size="10">no bend → depth adds nothing</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Two linear layers W1 then W2 collapse into a single matrix W1 times W2"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="40" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="70" y="65" fill="#5E5191">W₁</text><text x="115" y="65" fill="#6B645E">then</text><rect x="150" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="180" y="65" fill="#5E5191">W₂</text><text x="250" y="65" fill="#6B645E">=</text><rect x="290" y="40" width="140" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="360" y="65" fill="#C93B3B" font-weight="bold">one matrix W₁@W₂</text><text x="260" y="105" fill="#C93B3B" font-size="11">no activation → two layers become one</text></g></svg>
%%%

**What the ruler picture gets right:** each layer really is a straight-line rule, and laying straight rules end to end keeps things straight. **Where it breaks down:** a ruler is one fixed line, while a layer has many weights it can tune — but no matter how it tunes them, the *shape* stays straight. That is the part that matters.

#### Why depth alone buys nothing
Here's the same idea in plain words: stack two straight-line layers and the math quietly folds their weights together into a *single* set of weights — so the two layers are secretly one. Ten layers, a hundred — still one straight-line layer. Depth without a bend is an illusion. (Want the actual numbers? They're in the optional peek just below, and you'll run them yourself in **Produce**.)

If that feels slippery, you're in good company — almost everyone finds "the layers collapse" strange the first time. The tiny everyday example below makes it click.

!!! c-info 🪜
<b>Optional peek (skippable) — why two straights make one.</b> No matrix math needed to feel this. Take two straight-line steps: first "triple it" (3 × the number), then "double that and add one." Do them back to back and the combined rule is just "six times the number, plus one" (6 × input + 1) — still a single straight line. Two separate steps quietly collapsed into one; a third or a hundredth would too. That is the whole trap: straight ∘ straight is always straight. (In <b>Produce</b> you'll watch the matrix version prove itself — <code>(x·W₁)·W₂</code> lands on the exact same numbers as <code>x·(W₁·W₂)</code> — until you slip a bend between the layers and they finally disagree.)
!!!

Watch the two steps fold into one — the input `4` takes both paths and lands on the same number, and the two straight lines are secretly the same line:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two straight-line steps triple-it then double-and-add-one applied to 4 give 25, exactly matching the single combined rule six-times-plus-one; below, both rules draw the same straight line"><g font-family="monospace" font-size="12" text-anchor="middle">
<text x="45" y="35" fill="#3A342E" font-weight="bold">4</text>
<rect x="70" y="20" width="88" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="114" y="37" fill="#5E5191">×3 → 12</text>
<rect x="178" y="20" width="120" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="238" y="37" fill="#5E5191">×2 +1 → 25</text>
<text x="330" y="37" fill="#6B645E">=</text>
<rect x="352" y="18" width="140" height="30" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="422" y="38" fill="#C93B3B" font-weight="bold">6×4 +1 = 25</text>
<line x1="60" y1="170" x2="470" y2="170" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="80" y1="70" x2="80" y2="176" stroke="#E5DFD6" stroke-width="1"/>
<path d="M80 168 L300 96 L440 78" fill="none" stroke="#7C6DAA" stroke-width="4" opacity="0.5"/>
<path d="M80 168 L440 78" fill="none" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="5,4"/>
<text x="360" y="150" fill="#C93B3B" font-size="10" text-anchor="middle">both rules → one straight line</text>
</g></svg>
%%%

!!! c-warn 😕
<b>这一步很多人一开始都会觉得奇怪，很正常 · This step feels strange to almost everyone at first — that's completely normal.</b> The activation looks like an optional extra you tack on at the end, so it's easy to shrug off. In plain terms: without it, stacking 100 layers is mathematically the same as having <b>one</b> layer — all that depth silently folds flat. The activation is the single thing that makes "deep" worth anything. Sit with that for a moment; it's the whole reason today matters.
!!!

@@@ concept id=c2 tag="The fix" title="A bend between the layers" gotit="Got the fix"
So how do we stop the collapse? We put a **bend** between the layers. That bend is the [[activation function||A simple function applied at the end of a neuron, after the weighted sum + bias. It adds non-linearity so stacked layers can learn complex shapes.]] — a small [[non-linearity||Not a straight line. A non-linear step between layers lets a deep network bend and combine features to learn complex patterns.]] applied at the very end of a neuron.

Think of it like a **crease in a sheet of paper**. Flat paper laid on flat paper stays flat. But fold a crease into it first, and now you can build shapes a flat sheet never could. Put a bend between each layer and the layers can no longer be squashed into one.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="Left: a flat sheet of paper stays flat. Right: the same sheet with one folded crease stands up into a shape. A single crease unlocks shapes flat paper cannot make."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One crease unlocks shapes flat paper can't make</text><polygon points="40,120 200,120 220,90 60,90" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="50" y1="110" x2="200" y2="110" stroke="#E5DFD6"/><text x="128" y="145" text-anchor="middle" fill="#6B645E">flat paper</text><text x="128" y="160" text-anchor="middle" fill="#9A938A" font-size="10">stays flat (no shape)</text><text x="255" y="105" fill="#B8AEA2" font-size="16">→</text><path d="M300 130 L390 130 L410 100 L360 60 L340 95 Z" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><line x1="360" y1="60" x2="340" y2="95" stroke="#5E5191" stroke-width="2.5"/><line x1="340" y1="95" x2="390" y2="130" stroke="#5E5191" stroke-width="1"/><text x="420" y="92" fill="#5E5191">the crease</text><text x="420" y="106" fill="#5E5191">= the bend</text><text x="350" y="152" text-anchor="middle" fill="#2D8B55">creased paper stands up</text><text x="350" y="166" text-anchor="middle" fill="#9A938A" font-size="10">can't be squashed flat</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Inserting ReLU between the two layers stops the collapse"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="30" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="57" y="65" fill="#5E5191">W₁</text><rect x="100" y="45" width="70" height="30" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="135" y="65" fill="#1a5c38" font-weight="bold">ReLU</text><rect x="185" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="212" y="65" fill="#5E5191">W₂</text><text x="275" y="65" fill="#6B645E">≠</text><rect x="305" y="45" width="150" height="30" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2" stroke-dasharray="4,3"/><text x="380" y="65" fill="#6B645E">any single W</text><text x="250" y="105" fill="#2D8B55" font-size="11">a bend in the middle → cannot be folded flat</text></g></svg>
%%%

**What the paper-crease picture gets right:** one fold unlocks shapes a flat sheet can't make, just as one bend unlocks patterns a straight-line stack can't. **Where it breaks down:** paper only creases where you fold it, while an activation bends the signal at *every* neuron, on every example — far more folds than you could make by hand.

And that's the whole secret. **You just unlocked why deep learning is even possible.** The rest of today is meeting the bends themselves — starting with the very first one anyone tried.

@@@ concept id=c3 tag="The first bend" title="The step function — a hard on/off switch" gotit="Met the step"
The first activation people ever tried was the simplest bend you can imagine: a hard **on/off switch**. Picture a light switch on a wall. Flip it and the light is either fully **on** or fully **off** — nothing in between. That is the [[step function||The original activation: output 1 if the input is at or above zero, else 0. A hard on/off switch with no in-between — and slope 0 everywhere, so a network can't learn from it.]], the great-grandparent of every bend we'll meet today.

In words: if the input is at or above zero, output `1` (the neuron "fires"). If it's below zero, output `0` (it stays quiet). In symbols, `step(z) = 1 if z ≥ 0, else 0`.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A wall light switch in two states. On the left the switch is down, the bulb is dark and off. On the right the switch is up, the bulb is bright and on. Nothing in between."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A wall switch: fully OFF or fully ON — no in-between</text><rect x="60" y="40" width="70" height="100" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><rect x="84" y="92" width="22" height="40" rx="4" fill="#D8D2C8" stroke="#8f8a80"/><text x="95" y="156" text-anchor="middle" fill="#6B645E">switch down</text><circle cx="200" cy="70" r="20" fill="#EFEAF0" stroke="#9A938A" stroke-width="1.5"/><text x="200" y="76" text-anchor="middle" font-size="16">🌑</text><text x="200" y="120" text-anchor="middle" fill="#6B645E">input < 0</text><text x="200" y="136" text-anchor="middle" fill="#6B645E" font-weight="bold">output 0 (off)</text><rect x="330" y="40" width="70" height="100" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><rect x="354" y="48" width="22" height="40" rx="4" fill="#F2D97A" stroke="#9A7A10"/><text x="365" y="156" text-anchor="middle" fill="#6B645E">switch up</text><circle cx="470" cy="70" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="470" y="76" text-anchor="middle" font-size="16">💡</text><text x="470" y="120" text-anchor="middle" fill="#8A6D3B">input ≥ 0</text><text x="470" y="136" text-anchor="middle" fill="#8A6D3B" font-weight="bold">output 1 (on)</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Step function: output 0 for negative inputs, jumps to 1 at zero, then flat at 1 for positive inputs"><g font-family="monospace"><line x1="60" y1="100" x2="330" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><line x1="195" y1="30" x2="195" y2="118" stroke="#E5DFD6" stroke-width="1"/><path d="M70 100 L195 100 L195 48 L305 48" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="195" cy="48" r="3.5" fill="#8A6D3B"/><text x="125" y="117" font-size="11" fill="#6B645E" text-anchor="middle">output 0 (off)</text><text x="255" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">output 1 (on)</text><text x="415" y="72" font-size="12" fill="#8A6D3B">step → {0, 1}</text><text x="195" y="133" font-size="10" fill="#9A938A" text-anchor="middle">jump at 0 — no in-between, and flat (slope 0) everywhere else</text></g></svg>
%%%

%%% demo id=step label="run it"
code: step(np.array([-2, -1, 0, 1, 2]))
out: array([0, 0, 1, 1, 1])
take: <b>Step = 1 if z ≥ 0 else 0.</b> A hard on/off switch: everything below zero is 0, everything at or above zero is 1. Notice there is no "how much" — only "yes" or "no."
%%%

**What the light-switch picture gets right:** it's the perfect image for all-or-nothing, no middle ground. **Where it breaks down:** a wall switch is flipped by *you*, but here the *input* decides — and unlike a dimmer, there's no in-between, which turns out to be its fatal flaw.

Here's the little puzzle it leaves us. The step function looks tidy, so why doesn't anyone use it in hidden layers today? Because it is **flat everywhere** — its slope is `0` on both sides of the jump. Learning works by nudging weights in the direction that lowers the error, and a flat function gives no hint about which way to nudge: tweak the input a little and the output doesn't budge. So the network gets no signal to improve. That single flaw is exactly *why* the smoother, slope-carrying bends we meet next took over — and now you can already explain why the very first neural net design got stuck.

@@@ concept id=c4 tag="Meet ReLU" title="ReLU — a one-way valve" gotit="Met ReLU"
The bend that fixed all that is almost as simple as the switch — but it keeps a useful slope. Picture a **one-way valve** in a pipe: it lets water flow through in the forward direction, and blocks it the other way. That is [[ReLU||A one-way valve: it lets positive numbers pass and turns negatives into 0. The modern default activation.]]: it lets positive signals pass straight through, and blocks anything negative (sets it to 0). This is the bend inside almost every modern network — the one you'll reach for by default.

In symbols: `ReLU(z) = max(0, z)`. Read it out loud: "take the bigger of 0 and z." If `z` is positive, `z` wins and passes through. If `z` is negative, `0` wins. That's the whole rule.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve in a pipe. Top: a positive signal pushes forward and water flows straight through. Bottom: a negative signal is blocked and nothing passes, output zero."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">ReLU = a one-way valve</text><rect x="40" y="38" width="440" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><polygon points="250,42 262,55 250,68" fill="#2D8B55"/><line x1="250" y1="38" x2="250" y2="72" stroke="#2D8B55" stroke-width="1"/><text x="90" y="59" fill="#276b45">positive →</text><text x="330" y="59" fill="#276b45" font-weight="bold">flows through 💧</text><text x="470" y="59" text-anchor="end" fill="#2D8B55">→</text><rect x="40" y="96" width="440" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><line x1="250" y1="92" x2="250" y2="134" stroke="#C93B3B" stroke-width="3"/><rect x="244" y="100" width="12" height="26" fill="#C93B3B"/><text x="95" y="117" fill="#8a3b3b">negative ←</text><text x="330" y="117" fill="#C93B3B" font-weight="bold">BLOCKED → 0</text><text x="260" y="154" text-anchor="middle" fill="#6B645E" font-size="10">flows one way, blocked the other — that's max(0, z)</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="ReLU curve: flat at zero for negatives, a straight rising line for positives"><g font-family="monospace"><line x1="60" y1="95" x2="320" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="190" y1="25" x2="190" y2="110" stroke="#E5DFD6" stroke-width="1"/><path d="M70 95 L190 95 L300 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">negatives → 0</text><text x="260" y="30" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text x="400" y="70" font-size="13" fill="#1a5c38" font-family="monospace">ReLU(x)=max(0,x)</text></g></svg>
%%%

%%% demo id=relu label="run it"
code: relu(np.array([-3, -1, 0, 2, 5]))
out: array([0, 0, 0, 2, 5])
take: <b>ReLU = max(0, z).</b> It zeroes every negative and lets positives pass through unchanged. Left of zero it is flat (output 0); right of zero it is the line y = z.
%%%

**What the valve picture gets right:** flow one way, blocked the other — exactly ReLU's "positives pass, negatives stop." **Where it breaks down:** a real valve is either open or shut, but ReLU passes positives at their *full* size (5 stays 5, 100 stays 100) rather than a fixed trickle. Its range is `[0, ∞)` — zero and up.

Here's the payoff: unlike the flat step function, ReLU's positive side has a real slope of `1`, so learning gets a clear signal there. And because it zeroes every negative, many neurons output exactly `0` at once — a handy [[sparsity||Many neurons output exactly 0 at the same time, so only part of the network is active per example — cheaper, and "only the units that matter speak up."]] where only the units that matter speak up. Cheap to compute (just one `max`), a usable slope, and free sparsity — that combination is why ReLU is the default bend almost everywhere.

@@@ concept id=c5 tag="Meet sigmoid" title="Sigmoid — a soft dimmer" gotit="Met sigmoid"
The next bend is gentler, and it's the one behind every "how confident are you, 0 to 100%?" answer a model gives. Instead of the hard on/off of the step function, picture a **dimmer switch**: it slides smoothly from off to fully on, passing through every level of brightness in between. That is [[sigmoid||A soft dimmer switch that gently squashes any number into the range 0 to 1.]]. It takes any number and squashes it smoothly into the range 0 to 1.

In symbols: `sigmoid(z) = 1 / (1 + e^(−z))`. In words: one, divided by (one plus e-to-the-minus-z). You don't need to compute `e` by hand — the only thing to remember is that the answer always lands between 0 and 1, and `sigmoid(0) = 0.5` sits right in the middle.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A dimmer switch as a slider track. On the left, the slider is low and the bulb is dark near zero. In the middle it is half bright at one half. On the right it is full and the bulb is fully bright near one. It slides smoothly through every level."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid = a dimmer that slides smoothly off → on</text><rect x="60" y="70" width="400" height="14" rx="7" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="110" cy="77" r="12" fill="#5E5191"/><text x="110" y="106" text-anchor="middle" fill="#6B645E">low</text><text x="110" y="120" text-anchor="middle" fill="#9A938A" font-size="10">→ near 0</text><circle cx="260" cy="52" r="16" fill="#EFEAF0" stroke="#9A938A"/><text x="260" y="58" text-anchor="middle" font-size="14">🌑</text><circle cx="350" cy="48" r="18" fill="#F6EFCF" stroke="#C99A12"/><text x="350" y="54" text-anchor="middle" font-size="14">🔆</text><circle cx="430" cy="44" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="430" y="51" text-anchor="middle" font-size="16">💡</text><text x="260" y="106" text-anchor="middle" fill="#5E5191">middle</text><text x="260" y="120" text-anchor="middle" fill="#9A938A" font-size="10">= 0.5</text><text x="430" y="106" text-anchor="middle" fill="#8A6D3B">full</text><text x="430" y="120" text-anchor="middle" fill="#9A938A" font-size="10">→ near 1</text><text x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">no hard jump — every brightness in between lives in (0, 1)</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Sigmoid curve: an S shape rising smoothly from 0 to 1"><g font-family="monospace"><line x1="60" y1="95" x2="320" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="190" y1="25" x2="190" y2="110" stroke="#E5DFD6" stroke-width="1"/><path d="M70 92 C 150 92, 165 40, 300 38" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><text x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">→ near 0</text><text x="270" y="30" font-size="11" fill="#5E5191" text-anchor="middle">→ near 1</text><text x="400" y="70" font-size="12" fill="#5E5191" font-family="monospace">sigmoid → (0,1)</text></g></svg>
%%%

%%% demo id=sigmoid label="run it"
code: sigmoid(np.array([-2, 0, 2]))
out: array([0.119, 0.5  , 0.881])
take: <b>Sigmoid squashes into (0, 1).</b> A smooth "off → on": sigmoid(0)=0.5, big negatives land near 0, big positives near 1 (the values are rounded — sigmoid(2)≈0.8808). Because it lives between 0 and 1, it's handy when you want a probability.
%%%

**What the dimmer picture gets right:** the smooth, gradual slide from off to on, no hard jump. **Where it breaks down:** a real dimmer is evenly graded across its whole travel, but sigmoid *flattens out* at both ends — push the input far in either direction and the output barely changes. Hold onto that flattening; it hides a puzzle we'll crack in a couple of units.

@@@ concept id=c6 tag="Meet tanh" title="tanh — a balanced see-saw" gotit="Met tanh"
Picture a **see-saw** in a playground — a long plank balanced on a bar in the middle. When no one is on it, it rests perfectly level. Push down hard on the left and the left end drops all the way down; push down hard on the right and the right end rises all the way up. Now put numbers on that picture: the plank tilts down to `−1` at its lowest, up to `+1` at its highest, and rests at exactly `0` when nothing pushes it. That balanced, centered swing is [[tanh||Squashes any number into (−1, 1), and passes through the origin: tanh(0)=0. It is zero-centered, so its outputs are balanced above and below zero.]].

In plain words: hand tanh any number and it gives back a value between `−1` and `+1`. Big negatives tip it toward `−1`, big positives tip it toward `+1`, and an input of `0` leaves it resting at `0`. Because its outputs are balanced around 0 — as many below zero as above — we say it is **zero-centered**, and that balance helps the next layer learn.

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A playground see-saw. A plank rests on a central triangular pivot, tilted so its left end drops to minus one and its right end rises to plus one, resting level at zero in the middle."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">tanh = a balanced see-saw: −1 … 0 … +1</text><line x1="90" y1="120" x2="430" y2="70" stroke="#9A5A12" stroke-width="6" stroke-linecap="round"/><polygon points="260,95 240,140 280,140" fill="#C99A12" stroke="#9A7A10"/><line x1="170" y1="140" x2="350" y2="140" stroke="#B8AEA2" stroke-width="2"/><circle cx="90" cy="120" r="7" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="150" text-anchor="middle" fill="#C93B3B">push left</text><text x="90" y="164" text-anchor="middle" fill="#C93B3B" font-weight="bold">−1</text><circle cx="260" cy="95" r="5" fill="#6B645E"/><text x="290" y="90" fill="#6B645E">rests level</text><text x="260" y="64" text-anchor="middle" fill="#2D8B55" font-weight="bold">0</text><circle cx="430" cy="70" r="7" fill="#EAF5EE" stroke="#2D8B55"/><text x="430" y="58" text-anchor="middle" fill="#276b45" font-weight="bold">+1</text><text x="430" y="150" text-anchor="middle" fill="#276b45">push right</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="tanh curve: an S shape through the origin, rising from near minus one up to near plus one"><g font-family="monospace"><line x1="60" y1="70" x2="460" y2="70" stroke="#E5DFD6" stroke-width="1.5"/><line x1="260" y1="16" x2="260" y2="124" stroke="#E5DFD6" stroke-width="1"/><path d="M70 120 C 190 118, 205 74, 260 70 C 315 66, 330 22, 450 20" fill="none" stroke="#9A5A12" stroke-width="2.5"/><text x="110" y="136" font-size="11" fill="#6B645E" text-anchor="middle">→ near −1</text><text x="410" y="15" font-size="11" fill="#9A5A12" text-anchor="middle">→ near +1</text><circle cx="260" cy="70" r="4" fill="#C93B3B"/><text x="300" y="62" font-size="11" fill="#C93B3B" text-anchor="middle">tanh(0)=0</text><text x="400" y="98" font-size="12" fill="#9A5A12">tanh → (−1, 1)</text></g></svg>
%%%

%%% demo id=tanh label="run it"
code: tanh(np.array([-2, 0, 2]))
out: array([-0.964,  0.   ,  0.964])
take: <b>tanh squashes into (−1, 1) and rests at 0.</b> Like a see-saw: the input 0 leaves it level (tanh(0)=0), a big negative tips it toward −1, a big positive tips it toward +1. Balanced above and below zero — that's what "zero-centered" means.
%%%

**What the see-saw picture gets right:** the balanced tilt around a level middle — down to −1, up to +1, resting at 0 — is exactly tanh's zero-centered shape. **Where it breaks down:** a real see-saw keeps tilting the harder you push, but tanh *flattens out* near both ends — once you've pushed the input far enough, tilting harder barely moves the output. Same flattening puzzle as sigmoid, coming up soon.

#### The family it belongs to
tanh has the same smooth S-shape as sigmoid — the two are close cousins. The difference is the range and the resting point: sigmoid runs from `0` to `1` and rests at `0.5`, while tanh runs from `−1` to `+1` and rests at `0`. That zero resting point is why tanh was the go-to hidden-layer bend before ReLU took over, and it still shows up today — for example inside older recurrent networks.

#### Victory lap — you've met the whole family
Take a breath: you now know the four bends every deep-learning engineer uses — the crude **step** (flat, can't learn), the **ReLU** valve (cheap, keeps a slope for positives), the **sigmoid** dimmer (smooth 0→1), and its **tanh** see-saw cousin (smooth −1→1, balanced at 0). A side-by-side table is waiting for you in the one-page recap at the end — nothing to memorize now.

#### The three bends — and their slopes — on one picture
The bottom half of this picture shows the **slope** of each bend, and it's the one idea to carry into the next unit — so it's worth a slow look.

%%% svg
<svg viewBox="0 0 520 380" role="img" aria-label="Top panel: ReLU, sigmoid and tanh curves overlaid on shared axes. Bottom panel: their slopes f prime of z overlaid, sigmoid slope peaking near 0.25 and tanh slope peaking near 1.0."><g font-family="monospace" font-size="12">
<text x="60" y="22" font-size="13" fill="#3A342E" font-weight="bold">The bend  f(z)</text>
<line x1="60" y1="110" x2="470" y2="110" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="265" y1="36" x2="265" y2="160" stroke="#E5DFD6" stroke-width="1"/>
<text x="475" y="114" fill="#9A938A" font-size="10" text-anchor="start">z</text>
<path d="M70 155 C 190 153, 225 112, 265 110 C 305 108, 340 67, 460 65" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<path d="M70 152 C 190 150, 230 90, 265 87 C 300 84, 340 46, 460 44" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<path d="M70 110 L265 110 L440 46" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<text x="450" y="60" fill="#2D8B55" text-anchor="end" font-weight="bold">ReLU</text>
<text x="405" y="38" fill="#7C6DAA" text-anchor="start">sigmoid</text>
<text x="405" y="78" fill="#9A5A12" text-anchor="start">tanh</text>
<text x="60" y="212" font-size="13" fill="#3A342E" font-weight="bold">The slope  f′(z)</text>
<line x1="60" y1="340" x2="470" y2="340" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="265" y1="230" x2="265" y2="352" stroke="#E5DFD6" stroke-width="1"/>
<text x="475" y="344" fill="#9A938A" font-size="10" text-anchor="start">z</text>
<line x1="60" y1="248" x2="470" y2="248" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="252" fill="#9A5A12" font-size="10" text-anchor="end">1.0</text>
<line x1="60" y1="317" x2="470" y2="317" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="321" fill="#7C6DAA" font-size="10" text-anchor="end">0.25</text>
<path d="M70 340 L265 340 L265 248 L460 248" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<path d="M70 339 C 200 338, 235 252, 265 248 C 295 252, 330 338, 460 339" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<path d="M70 340 C 205 339, 240 319, 265 317 C 290 319, 325 339, 460 340" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<text x="272" y="244" fill="#9A5A12" font-size="11" text-anchor="start">tanh′ peaks ≈ 1.0</text>
<text x="272" y="313" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid′ peaks ≈ 0.25</text>
<text x="450" y="240" fill="#2D8B55" text-anchor="end" font-weight="bold">ReLU′ = 1 (z&gt;0)</text>
<text x="130" y="333" fill="#2D8B55" font-size="10" text-anchor="middle">ReLU′ = 0 (z&lt;0)</text>
</g></svg>
%%%

#### What is a "slope," in plain words?
This is the one word that unlocks the rest of the day. The **slope** of a bend at a point is just *how steep the curve is right there* — how much the output moves when you nudge the input a tiny bit. A steep slope means the neuron reacts strongly. A slope near `0` — a flat stretch — means the neuron barely reacts, so learning there crawls to a stop. Look at the bottom panel: sigmoid's slope never climbs above about `0.25` (its steepest point, right in the middle), tanh's reaches about `1.0`, and ReLU's is a flat `0` for negatives or a plain `1` for positives. (The exact recipe for a slope is called the *derivative* — you'll do that math on Day 5. Today the picture is all you need.) Keep this in your pocket: the next two units are little mysteries about what happens when that slope goes flat.

@@@ concept id=c7 tag="Flat = no learning" title="Saturation, and how a learning signal fades away" gotit="Got saturation"
Here's the first little mystery: a smooth bend can quietly *stop learning* — and it happens right in those flat tails we kept flagging. The picture below is the whole idea.

Remember the dimmer that flattens out at both ends? When a `sigmoid` or `tanh` gets a big input — say `z = 10` or `z = −10` — its output is already pinned near the top or bottom, and nudging the input barely moves it. The name for "the curve has gone flat here" is [[saturation||When an activation's input is far from zero, its output is pinned near a limit and its slope is ≈ 0. A saturated neuron barely responds to changes in its input.]]. A saturated neuron has a slope of about `0`.

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="Four people in a line passing a whisper forward. Person one whispers loudly, person two hears a quarter, person three much less, and person four hears only a faint mumble. The learning signal fades to almost nothing through the layers."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A whisper down a line — the learning signal fades</text><circle cx="70" cy="105" r="20" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="70" y="111" text-anchor="middle" font-size="16">🗣️</text><rect x="58" y="58" width="56" height="26" rx="8" fill="#EFEAF7" stroke="#7C6DAA"/><text x="86" y="75" text-anchor="middle" fill="#5E5191">1.0</text><text x="122" y="110" fill="#B8AEA2">→</text><circle cx="180" cy="105" r="20" fill="#EFEAF7" stroke="#9A93C0"/><text x="180" y="111" text-anchor="middle" font-size="14">👂</text><rect x="172" y="66" width="42" height="18" rx="6" fill="#EFEAF7" stroke="#9A93C0"/><text x="193" y="79" text-anchor="middle" fill="#7C6DAA" font-size="10">0.25</text><text x="232" y="110" fill="#B8AEA2">→</text><circle cx="290" cy="105" r="20" fill="#F4F1FA" stroke="#C7BEDF"/><text x="290" y="111" text-anchor="middle" font-size="14">👂</text><rect x="286" y="72" width="30" height="12" rx="5" fill="#F4F1FA" stroke="#C7BEDF"/><text x="301" y="81" text-anchor="middle" fill="#9A93C0" font-size="9">0.06</text><text x="342" y="110" fill="#B8AEA2">→</text><circle cx="400" cy="105" r="20" fill="#FDECEC" stroke="#C93B3B"/><text x="400" y="111" text-anchor="middle" font-size="14">😕</text><text x="400" y="78" text-anchor="middle" fill="#C93B3B" font-size="9">faint mumble</text><text x="260" y="152" text-anchor="middle" fill="#C93B3B" font-size="10">×0.25 each pass → by layer 4 ≈ 0.004 → the vanishing gradient</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Sigmoid S-curve with the two flat tails shaded and marked slope near zero, and the steep middle marked slope near 0.25"><g font-family="monospace"><line x1="50" y1="105" x2="470" y2="105" stroke="#E5DFD6" stroke-width="1.5"/><line x1="260" y1="24" x2="260" y2="122" stroke="#E5DFD6" stroke-width="1"/><rect x="50" y="95" width="90" height="20" fill="#FDE8E8" opacity="0.7"/><rect x="380" y="30" width="90" height="20" fill="#FDE8E8" opacity="0.7"/><path d="M60 108 C 175 108, 205 42, 260 40 C 315 38, 345 33, 460 32" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><text x="95" y="130" font-size="10" fill="#C93B3B" text-anchor="middle">flat tail: slope ≈ 0</text><text x="425" y="25" font-size="10" fill="#C93B3B" text-anchor="middle">flat tail: slope ≈ 0</text><text x="300" y="70" font-size="10" fill="#5E5191" text-anchor="middle">steep middle: slope ≈ 0.25</text></g></svg>
%%%

Now here's why that matters. Learning only nudges a weight when the slope is non-zero — the slope is literally the "which way should I move?" signal. If the slope is ≈ `0`, the nudge is ≈ `0`, so a saturated neuron sits frozen. On its own that's one sleepy neuron. The real trouble shows up when you stack many layers — and that's the puzzle the box below cracks.

!!! c-info 🔎
<b>Optional deep-dive (skippable) — a whisper down a long line.</b> Want to *feel* why deep sigmoid nets stalled for years? Imagine a line of people passing a whispered message from the back of the room to the front. That whisper is the <em>learning signal</em> travelling backward through the layers, telling each early layer how to improve — and it gets multiplied by each layer's slope on the way back. Because a sigmoid's slope tops out around <code>0.25</code> even at its steepest (and a saturated layer multiplies by even less), suppose every person passes on only a <em>quarter</em> of what they heard. Four people in, the message is <code>0.25 × 0.25 × 0.25 × 0.25 ≈ 0.004</code> of the original — a faint mumble. Ten people in, it's essentially silence. The early layers hear nothing, so they never learn. That fading-to-silence is the [[vanishing gradient||A learning signal (gradient) that shrinks toward zero as it is multiplied by each layer's small slope on the way back through a deep network — so the earliest layers stop learning. Saturating activations like sigmoid/tanh cause it; ReLU largely avoids it.]] problem.
<br><br>
<b>The picture:</b>
<br><span style="font-family:monospace">1.0 → 0.25 → 0.06 → 0.016 → ≈0.004</span> — ×0.25 per layer, and by the fourth layer the signal is a faint mumble.
!!!

See the fade — and the rescue — in one picture. Each bar is how loud the learning signal still is that many layers back. Sigmoid's bars shrink to almost nothing; ReLU's stay tall:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two rows of bars. Top row sigmoid: signal strength 1.0 then 0.25 then 0.06 then 0.016 then 0.004, shrinking to almost nothing. Bottom row ReLU: strength stays 1.0 at every layer, all bars full height."><g font-family="monospace" font-size="11" text-anchor="middle">
<text x="60" y="20" fill="#5E5191" font-weight="bold" text-anchor="start">sigmoid · ×0.25 per layer → signal fades</text>
<rect x="60"  y="30" width="46" height="60" fill="#7C6DAA"/><text x="83"  y="103" fill="#6B645E">1.0</text>
<rect x="140" y="75" width="46" height="15" fill="#7C6DAA"/><text x="163" y="103" fill="#6B645E">0.25</text>
<rect x="220" y="86" width="46" height="4"  fill="#7C6DAA"/><text x="243" y="103" fill="#6B645E">0.06</text>
<rect x="300" y="89" width="46" height="1"  fill="#9A93C0"/><text x="323" y="103" fill="#6B645E">0.016</text>
<rect x="380" y="90" width="46" height="1"  fill="#D9D3EC"/><text x="403" y="103" fill="#C93B3B">≈0.004</text>
<text x="60" y="130" fill="#2D8B55" font-weight="bold" text-anchor="start">ReLU · ×1 per layer → signal survives</text>
<rect x="60"  y="140" width="46" height="55" fill="#2D8B55"/><text x="83"  y="207" fill="#6B645E">1.0</text>
<rect x="140" y="140" width="46" height="55" fill="#2D8B55"/><text x="163" y="207" fill="#6B645E">1.0</text>
<rect x="220" y="140" width="46" height="55" fill="#2D8B55"/><text x="243" y="207" fill="#6B645E">1.0</text>
<rect x="300" y="140" width="46" height="55" fill="#2D8B55"/><text x="323" y="207" fill="#6B645E">1.0</text>
<rect x="380" y="140" width="46" height="55" fill="#2D8B55"/><text x="403" y="207" fill="#6B645E">1.0</text>
<text x="460" y="60"  fill="#9A938A" text-anchor="start">layer 1…4 back</text>
</g></svg>
%%%

**The neat answer.** Now look back at the slope picture: `ReLU`'s slope on the positive side is a flat `1`, not `0.25`. Multiply `1 × 1 × 1 × …` down the line and the signal keeps its full strength — no shrinking. That is exactly why swapping saturating sigmoids/tanhs for `ReLU` was the single change that let people train genuinely deep networks. **You can now explain, in one sentence, why sigmoid stalled deep nets and ReLU rescued them** — that's a real interview beat, and you have it. (You'll meet the backward-pass math that actually multiplies these slopes on Day 5; today the whisper picture is all you need.)

@@@ concept id=c8 tag="Dead ReLUs" title="When a ReLU quietly dies" gotit="Got the dead ReLU"
ReLU rescued us from saturation — but it has a sneaky trap of its own, and this one is famously *silent*: nothing crashes, no error prints, the loss just quietly stops dropping. Here's the puzzle — how can a neuron simply give up?

Think of the same **one-way valve** from before, but now imagine it has **jammed shut**. Water piles up behind it and nothing flows through. That is a [[dead ReLU||A ReLU whose input has become negative for every example, so it outputs a flat 0 forever. Its slope there is 0, so training gives it no signal, and it never recovers on its own.]]: a ReLU whose input has drifted negative for *every* example, so it outputs a flat `0` and never opens again.

Don't take the flat spot on faith — **play with it.** Drag the marker into the flat left side of ReLU (or into either tail of sigmoid/tanh) and watch the slope collapse toward `0` with your own eyes. Try to predict the slope before you let go.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve jammed fully shut. Water piles up behind the closed gate and nothing passes through. The pipe on the far side is empty. The valve is frozen and cannot reopen on its own."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A dead ReLU = the valve jammed SHUT</text><rect x="40" y="56" width="200" height="46" rx="4" fill="#DCE9F5" stroke="#7C6DAA" stroke-width="1.5"/><text x="120" y="84" text-anchor="middle" fill="#5E5191">💧 water piles up 💧</text><line x1="250" y1="46" x2="250" y2="112" stroke="#C93B3B" stroke-width="5"/><rect x="242" y="56" width="16" height="46" fill="#C93B3B"/><text x="250" y="40" text-anchor="middle" fill="#C93B3B" font-size="14">🔒</text><rect x="262" y="56" width="216" height="46" rx="4" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5" stroke-dasharray="4,3"/><text x="370" y="84" text-anchor="middle" fill="#9A938A">nothing gets through</text><text x="120" y="126" text-anchor="middle" fill="#6B645E" font-size="10">input stays negative</text><text x="370" y="126" text-anchor="middle" fill="#C93B3B" font-size="10">output frozen at 0</text><text x="260" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">no hand to pry it open — slope 0, so training can't grip it</text></g></svg>
%%%

%%% viz src=../../viz/activation-derivatives.html title="activations and their slopes" caption="interactive — pick an activation, drag z; watch the slope flatten"
%%%

**What the jammed-valve picture gets right:** just like a stuck valve, a dead ReLU passes nothing — its output is frozen at 0 no matter what comes in. **Where it breaks down:** a real jammed valve you can pry open by hand, but a dead ReLU has no hand to reopen it — its slope on the negative side is exactly `0`, so the training signal can't get a grip, and it can't climb back out on its own. That's what makes this failure quietly permanent.

So what jams the valve, and how do you catch it? Two crisp answers:

- **What kills it:** most often a **learning rate that's too large** — the learning rate is the size of the step training takes when it nudges the weights (a real knob you'll tune on Day 8). One too-big step can shove a unit's input permanently negative. A bad random start can do the same before training even begins.
- **How you catch it:** because it's silent, you go looking — **log activation statistics** (the fraction of ReLUs stuck at `0`) instead of only watching the loss. When that dead-fraction creeps up while the loss plateaus, you've found your culprit. That one habit is a real superpower.

!!! c-warn ⚠️
<b>The silent tell:</b> nothing crashes — the loss just plateaus higher than you'd expect while a chunk of your neurons quietly output a constant. Watch the activation stats, not only the loss.
!!!

**You can now diagnose a stuck network from its activations, not just its loss** — the next unit hands you the one-line cure.

@@@ concept id=c9 tag="The cure" title="Leaky ReLU — a valve with a small leak" gotit="Got the cure"
A dead ReLU sounds scary — so here's the delightful part: one tiny tweak fixes it. Picture the same one-way valve, but this time it's left with a **tiny permanent leak**. Positives still flow straight through. Negatives, instead of being shut off completely, are let through as a small trickle. That is [[Leaky ReLU||Like ReLU, but negatives get a small slope (α·z, e.g. α=0.01) instead of a flat 0 — so the unit keeps a tiny slope on the negative side and can't fully die.]].

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve with a tiny leak. Top: positive flow passes straight through as before. Bottom: on the blocked side a small trickle still seeps past the gate instead of being fully stopped, keeping the valve alive."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Leaky ReLU = a valve with a tiny built-in leak</text><rect x="40" y="38" width="440" height="32" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><polygon points="250,42 262,54 250,66" fill="#2D8B55"/><text x="95" y="58" fill="#276b45">positive →</text><text x="330" y="58" fill="#276b45" font-weight="bold">flows through 💧</text><rect x="40" y="92" width="440" height="36" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><line x1="250" y1="88" x2="250" y2="132" stroke="#C99A12" stroke-width="3"/><rect x="244" y="96" width="12" height="20" fill="#C99A12"/><text x="259" y="124" fill="#9A7A10" font-size="9">·····</text><text x="95" y="114" fill="#8A6D3B">negative ←</text><text x="320" y="114" fill="#8A6D3B" font-weight="bold">small trickle (slope α)</text><text x="260" y="150" text-anchor="middle" fill="#2D8B55" font-size="10">the leak = a tiny non-zero slope → the unit can't fully die</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 145" role="img" aria-label="Leaky ReLU: a gentle downward slope for negative inputs instead of a flat zero, and a rising line for positive inputs; a dashed line marks where plain ReLU would be flat at zero"><g font-family="monospace"><line x1="60" y1="92" x2="330" y2="92" stroke="#E5DFD6" stroke-width="1.5"/><line x1="195" y1="24" x2="195" y2="116" stroke="#E5DFD6" stroke-width="1"/><path d="M70 92 L195 92" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><path d="M70 104 L195 92 L305 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="120" y="120" font-size="11" fill="#2D8B55" text-anchor="middle">tiny leak (slope α)</text><text x="118" y="84" font-size="10" fill="#C93B3B" text-anchor="middle">ReLU: flat 0 (can die)</text><text x="258" y="34" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text x="415" y="70" font-size="12" fill="#1a5c38">leaky(z)=max(αz, z)</text></g></svg>
%%%

Why does a leak help? The negative side now has a small but *non-zero* slope. So a unit that drifts negative still gets a learning signal, and it can climb back to life instead of dying. The flat `0` was the whole problem; a gentle slope is the whole fix.

In symbols: `leaky(z) = z` for positives, or `α·z` for negatives, where `α` (the Greek letter "alpha") is a small number like `0.01`. In words: positives pass unchanged, and negatives aren't zeroed — they're shrunk to a tiny trickle.

%%% demo id=leaky label="run it"
code: leaky_relu(np.array([-5, -2, 0, 2, 5]), alpha=0.01)
out: array([-0.05, -0.02,  0.  ,  2.  ,  5.  ])
take: <b>Leaky ReLU = max(αz, z), with α small (≈0.01).</b> Positives are identical to ReLU; negatives become a small trickle (−5 → −0.05) instead of a flat 0. That trickle keeps a slope on the negative side, so the unit can't get permanently stuck.
%%%

**What the "leaky valve" picture gets right:** the tiny leak is exactly the small negative slope that keeps the neuron alive. **Where it breaks down:** a real leak is an accident you'd want to fix, but here the leak is the *feature* — deliberately built in. That's the entire Leaky ReLU idea: a valve with a small leak so it never fully dies. **You now know a failure mode AND its named cure** — that pairing is exactly how engineers think.

The three items below are close relatives you'll *hear about* — you don't need them today, so treat this box as optional and skip ahead if you like.

!!! c-info 🧰
<b>Optional — three relatives you'll hear named (skippable today):</b>
<br><b>· GELU</b> uses a *smooth* curve near zero instead of a straight leak, so it keeps a non-zero slope right around the origin — that's the bend GPT-2 and GPT-3 actually use. Same goal as Leaky ReLU: never fully die.
<br><b>· He (Kaiming) initialization</b> heads off dead ReLUs *before training starts.* When you first build a network you fill its weights with small random numbers; if those are too big, many ReLUs get shoved negative on day one and die before learning anything. He init is the recipe that sizes those starting weights so activations begin at a healthy size — the sensible default for a ReLU network, set with one line.
<br><b>· Xavier / Glorot initialization</b> is He's sibling recipe, used for `tanh`/`sigmoid` networks instead. You'll pick between them when you get to real training (a Day-5 detail).
!!!

!!! c-info ⚖️
<b>Trade-off: ReLU's speed and sparsity vs. its dead-neuron risk.</b> Choosing `ReLU` buys you a cheap activation (just a `max`) that keeps gradients healthy in deep stacks and turns off many neurons at once — a useful <b>sparsity</b> (lots of exact zeros, so each layer leans on only the units that matter). What you give up is safety at zero: any neuron pushed permanently negative dies and never comes back. Softer bends that never fully flatten (Leaky ReLU, GELU) trade a little extra math and less sparsity for never dying. Which one to use, and where, is a <b>design-review</b> decision — usually ReLU-family in the hidden layers, sigmoid or softmax only at the output.
!!!

!!! c-ok 🎤
<b>Once it clicks, here's how you'd explain it (say, in an interview):</b> "An activation is the per-layer non-linearity. Without it `(x W₁) W₂ = x (W₁ W₂)` — the whole stack collapses to one straight-line map, so depth adds nothing. ReLU = max(0, z) is the cheap default and dodges vanishing gradients, because its positive-side slope of 1 doesn't shrink the backward signal the way sigmoid's ≈0.25 slope does through many layers. The silent failure is saturation — dead ReLUs stuck at 0, or sigmoids pinned near 0/1 — where the slope goes to zero and the unit stops learning, so I log activation statistics, not just the loss. If units are dying I reach for Leaky ReLU or GELU, and I default to He initialization so fewer units die at the start." Notice you can already say every piece of that — that's how far you've come today.
!!!

@@@ concept id=c10 tag="Sigmoid's tilt" title="Why plain sigmoid drifts (not zero-centered)" gotit="Got the drift"
One last, gentler mystery — and it's the exact reason tanh was invented as sigmoid's fix. Picture a **shopping cart with one stuck wheel** that always pulls a little to the left. You can still get where you're going, but you zig-zag instead of rolling straight, and it takes longer. Sigmoid does something similar to a network's learning path.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A shopping cart with one wheel stuck turned to the side so it always pulls left. Its path zig-zags across the floor toward the goal instead of rolling straight, so it takes longer."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid's tilt = a cart with one stuck wheel</text><rect x="48" y="48" width="70" height="40" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><line x1="56" y1="56" x2="56" y2="88" stroke="#7C6DAA"/><line x1="110" y1="48" x2="122" y2="36" stroke="#7C6DAA" stroke-width="1.5"/><circle cx="122" cy="34" r="4" fill="#7C6DAA"/><circle cx="62" cy="98" r="9" fill="#FDF9F3" stroke="#6B645E" stroke-width="1.5"/><circle cx="104" cy="98" r="9" fill="#FDF9F3" stroke="#C93B3B" stroke-width="2.5"/><line x1="98" y1="92" x2="110" y2="104" stroke="#C93B3B" stroke-width="2"/><text x="104" y="124" text-anchor="middle" fill="#C93B3B" font-size="9">stuck wheel</text><path d="M150 100 L200 66 L245 100 L295 62 L340 100 L390 60 L435 88" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="300" y="122" text-anchor="middle" fill="#C93B3B" font-size="10">always-positive nudges → zig-zag, slow</text><path d="M150 100 L435 60" fill="none" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="5,4"/><text x="320" y="72" fill="#2D8B55" font-size="10">straight (zero-centered)</text><circle cx="440" cy="58" r="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="440" y="46" text-anchor="middle" fill="#6B645E" font-size="10">goal</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A zig-zagging path from start to goal because every step is nudged in the same sideways direction, versus a straight path"><g font-family="monospace" font-size="11"><circle cx="55" cy="95" r="5" fill="#2D8B55"/><text x="55" y="118" fill="#6B645E" text-anchor="middle">start</text><path d="M55 95 L110 55 L160 90 L215 48 L270 84 L325 42 L380 78 L435 40" fill="none" stroke="#C93B3B" stroke-width="2" stroke-dasharray="1,0"/><text x="245" y="112" fill="#C93B3B" text-anchor="middle">all-positive nudges → zig-zag (sigmoid)</text><path d="M55 95 L440 40" fill="none" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="5,4"/><text x="300" y="60" fill="#2D8B55" text-anchor="middle">straight (zero-centered)</text><circle cx="440" cy="40" r="5" fill="#9A5A12"/><text x="440" y="30" fill="#6B645E" text-anchor="middle">goal</text></g></svg>
%%%

The cause is simple: a `sigmoid` only ever outputs *positive* numbers (everything lands in `(0, 1)`). When every output of a layer is positive, the nudges the next layer receives all get pushed in the same direction at once — that's the stuck wheel, always pulling the same way. Learning still works, but it wanders: the path to a good answer zig-zags instead of going straight, so training is slower than it needs to be. This is milder than a dead unit — nothing stops, it just drags.

The neat fix: `tanh` is **zero-centered** (outputs balanced above and below `0`), so the nudges aren't all shoved one way, and the path straightens out. `ReLU` sidesteps the problem a different way. Together with the vanishing-gradient trouble from two units ago, this is a big part of why plain sigmoid is rare in today's hidden layers.

**What the stuck-wheel picture gets right:** a constant sideways pull really does force a zig-zag instead of a straight line — just like all-positive gradients. **Where it breaks down:** a stuck wheel never reaches the goal, but a zig-zagging network usually still gets there — it just takes more steps. **You now know why tanh exists** — it's the zero-centered answer to sigmoid's drift.

@@@ concept id=c11 tag="The limit" title="One line can't do XOR" gotit="Got the limit"
Let's end with the puzzle that started this whole field — and see why bends matter, from the other side. Picture a **room with four chairs**. Two red chairs sit in the front-left and back-right corners; two blue chairs sit in the front-right and back-left corners — so each color sits on its own **diagonal**. Now I hand you a single straight **rope** pulled tight across the floor, and ask you to put both red chairs on one side and both blue chairs on the other.

%%% svg
<svg viewBox="0 0 520 178" role="img" aria-label="A room floor with four chairs at the corners. Two red chairs sit on one diagonal, two blue chairs on the other. A single straight rope drawn across the floor always leaves one chair on the wrong side — the best you get is three of four."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One straight rope can't split two diagonal pairs</text><rect x="150" y="32" width="220" height="120" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><rect x="172" y="52" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="185" y="70" text-anchor="middle" fill="#C93B3B">🔴</text><rect x="322" y="106" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="335" y="124" text-anchor="middle" fill="#C93B3B">🔴</text><rect x="322" y="52" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="335" y="70" text-anchor="middle" fill="#5E5191">🔵</text><rect x="172" y="106" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="185" y="124" text-anchor="middle" fill="#5E5191">🔵</text><line x1="140" y1="148" x2="390" y2="44" stroke="#9A5A12" stroke-width="3" stroke-dasharray="1,0"/><text x="405" y="46" fill="#9A5A12" font-size="10">straight rope</text><text x="70" y="96" text-anchor="middle" fill="#6B645E" font-size="10">reds on one</text><text x="70" y="110" text-anchor="middle" fill="#6B645E" font-size="10">diagonal,</text><text x="70" y="124" text-anchor="middle" fill="#6B645E" font-size="10">blues the other</text><text x="260" y="170" text-anchor="middle" fill="#C93B3B" font-size="10">any angle → best you get is 3 of 4 (need a bend + a 2nd rope)</text></g></svg>
%%%

%%% viz src=../../viz/xor-limit.html title="why one neuron can't do XOR" caption="interactive — slide the single line (stuck at 3/4), then add a hidden layer"
%%%

**Step 1 — try the single rope, and watch it fail.** Go ahead and slide the rope, angle it any way you like. You can always separate three chairs, but the fourth ends up on the wrong side. One straight rope simply cannot split two diagonal pairs — the best you'll ever manage is 3 of 4.

**Step 2 — name what you just saw.** That room is exactly the [[XOR||XOR: output 1 when exactly one of two inputs is on, else 0. Its two classes sit on opposite diagonals, so no single straight line separates them.]] problem — "on when exactly one of two inputs is on." A single neuron draws **one straight line** (that was Day 1), which is your one straight rope. XOR is not [[linearly separable||A dataset is linearly separable if one straight line (or flat plane) can put every class on its own side. XOR is the classic example that is NOT.]] — no single straight cut works, no matter how you tilt it.

**Step 3 — how a bend rescues it.** Now flip on the hidden layer in the demo. A network gets a *second* rope, and a bend between the layers lets you *fold two lines together*: the first rope splits off one corner, the second handles the rest, and the bend combines them into a boundary a single straight rope never could make. Two simple straight pieces, joined through a bend, carve out the region one line can't reach. That's the same lesson as the whole day, seen from a new angle: the bend is what lets simple pieces combine into something a straight line can't.

**What the room-of-chairs picture gets right:** the diagonal layout is the whole trap — two groups arranged so no single straight cut can ever separate them, just like XOR. **Where it breaks down:** with a real rope you're stuck for good, but a neural network isn't limited to one rope — add a hidden layer with a bend and the two lines combine into a boundary a single rope never could.

#### Why a frontier lab cares
`ReLU` is the classic workhorse: dirt cheap (just a `max`), and it sidesteps the vanishing-gradient problem that plagues sigmoid in deep stacks — its positive-side slope of `1` keeps the backward signal from shrinking (you met that whisper-down-the-line picture earlier, and you'll meet the gradient math on Day 5). It's also why different labs pick different bends: GPT-2 and GPT-3 use **GELU**, and Llama uses **SwiGLU** — smoother versions of the same "keep a small slope so the bend never dies" idea you just met with Leaky ReLU. Sigmoid and its cousin [[softmax||Turns a vector of scores into probabilities that sum to 1. Used at the output of a classifier.]] live mostly at the output, where you want probabilities. The activation really is the quiet reason "deep" learning works at all — and now you know why.

@@@ concept id=c12 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you did the whole thing. Here's a fresh way to hold the whole day at once: picture today as one **road trip on a single road**, and the five beats below as the five signposts you drove past, in order. A trip is one continuous journey with a clear start and finish, and each signpost only makes sense because of the one before it — that's exactly how today's ideas stack: the fix only matters because of the trap, the failures only matter because you'd met the family, and the limit is the view from the far end. **Where the road-trip picture breaks down:** a real trip is a straight line you drive once and never revisit — but these signposts are a loop you'll circle back to for years, and the whole point of today (the bend!) is the opposite of a straight road. Read the five signposts, then keep the cheat-sheets below as your map.

The five signposts, in order:
- **Signpost 1 · The trap.** Straight-line layers stacked together stay one straight line — depth alone buys nothing.
- **Signpost 2 · The fix.** Put a **bend** (an activation) between layers, and now they can't fold flat — that's what makes "deep" mean something.
- **Signpost 3 · The family.** You met four bends: the crude **step** (flat, can't learn), the **ReLU** valve (cheap, keeps a slope for positives), the **sigmoid** dimmer (0→1), and the **tanh** see-saw (−1→1, balanced at 0).
- **Signpost 4 · The two silent failures — each with its cure.** Saturating bends flatten at the tails, so the learning signal fades (**vanishing gradient**) → cure: **ReLU**, whose slope stays 1. A ReLU shoved permanently negative goes **dead** (flat 0 forever) → cures: **Leaky ReLU**, **GELU**, and starting with **He initialization** so fewer die on day one.
- **Signpost 5 · The limit.** A bend lets simple pieces combine, but one neuron is still one straight line — it can't do **XOR**; you need layers.

Here's the one picture to keep — the three working bends side by side.

%%% svg
<svg viewBox="0 0 520 178" role="img" aria-label="A single winding road from Start to Finish with five numbered signposts along it in order: one the trap, two the fix, three the family, four the two silent failures, five the limit. Each signpost follows the one before."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12" font-family="monospace">Today = one road trip, five signposts in order</text><path d="M30 150 C 120 150, 120 60, 210 60 C 300 60, 300 150, 390 150 C 440 150, 470 110, 495 90" fill="none" stroke="#E5DFD6" stroke-width="10"/><path d="M30 150 C 120 150, 120 60, 210 60 C 300 60, 300 150, 390 150 C 440 150, 470 110, 495 90" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="6,6"/><circle cx="30" cy="150" r="6" fill="#2D8B55"/><text x="30" y="172" text-anchor="middle" fill="#276b45">start</text><g><line x1="78" y1="128" x2="78" y2="148" stroke="#9A5A12" stroke-width="2"/><rect x="58" y="110" width="40" height="18" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="78" y="123" text-anchor="middle" fill="#8A6D3B">1 trap</text></g><g><line x1="150" y1="70" x2="150" y2="90" stroke="#9A5A12" stroke-width="2"/><rect x="132" y="52" width="36" height="18" rx="3" fill="#EFEAF7" stroke="#7C6DAA"/><text x="150" y="65" text-anchor="middle" fill="#5E5191">2 fix</text></g><g><line x1="250" y1="70" x2="250" y2="90" stroke="#9A5A12" stroke-width="2"/><rect x="226" y="52" width="48" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="250" y="65" text-anchor="middle" fill="#276b45">3 family</text></g><g><line x1="350" y1="128" x2="350" y2="148" stroke="#9A5A12" stroke-width="2"/><rect x="322" y="110" width="56" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="350" y="123" text-anchor="middle" fill="#C93B3B">4 failures</text></g><g><line x1="455" y1="98" x2="455" y2="118" stroke="#9A5A12" stroke-width="2"/><rect x="432" y="80" width="46" height="18" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="455" y="93" text-anchor="middle" fill="#8A6D3B">5 limit</text></g><circle cx="495" cy="90" r="6" fill="#C93B3B"/><text x="495" y="78" text-anchor="middle" fill="#6B645E">finish 🏁</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Recap: ReLU, sigmoid and tanh curves side by side on shared axes"><g font-family="monospace" font-size="12"><line x1="60" y1="95" x2="470" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="265" y1="24" x2="265" y2="112" stroke="#E5DFD6" stroke-width="1"/><path d="M70 95 L265 95 L440 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><path d="M70 92 C 190 90, 230 44, 265 42 C 300 40, 340 34, 460 33" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><path d="M70 118 C 190 116, 230 96, 265 94 C 300 92, 340 50, 460 48" fill="none" stroke="#9A5A12" stroke-width="2.5"/><text x="450" y="36" fill="#2D8B55" text-anchor="end" font-weight="bold">ReLU</text><text x="405" y="30" fill="#7C6DAA" text-anchor="start">sigmoid</text><text x="405" y="60" fill="#9A5A12" text-anchor="start">tanh</text><text x="130" y="90" fill="#9A938A" font-size="10" text-anchor="middle">negatives</text><text x="360" y="112" fill="#9A938A" font-size="10" text-anchor="middle">positives</text></g></svg>
%%%

#### Cheat-sheet · the three bends at a glance
%%% table
:: :: ReLU :: sigmoid :: tanh
Shape :: one-way valve :: soft dimmer :: balanced see-saw
Range :: `[0, ∞)` :: `(0, 1)` :: `(−1, 1)`
Slope :: `1` for positives, `0` for negatives :: peaks ≈ `0.25` :: peaks ≈ `1.0`
Watch out :: can go **dead** at 0 :: **saturates** + not zero-centered :: **saturates** at the tails
Reach for it :: hidden layers (default) :: an output probability :: when you want a balanced, bounded signal
%%%

#### Cheat-sheet · the words you met today
%%% jargon
activation | the little bend added at the end of a neuron, after the weighted sum
non-linearity | not a straight line — the property a bend adds so depth matters
step function | the original hard on/off bend; flat everywhere, so it can't learn
ReLU | a one-way valve: max(0, z) — passes positives, zeroes negatives
sigmoid | a soft dimmer that squashes any number into (0, 1)
tanh | a see-saw that squashes into (−1, 1) and rests at 0 (zero-centered)
slope | how steep the curve is at a point — a slope near 0 means learning stalls there
saturation | the curve has gone flat, so the slope ≈ 0 and the unit barely responds
vanishing gradient | the learning signal shrinks toward 0 through a deep stack of small slopes
dead ReLU | a ReLU stuck outputting a flat 0 forever, with no slope to recover
Leaky ReLU | ReLU with a tiny negative slope (α·z) so the unit can't fully die
GELU | a smooth ReLU-like bend (used in GPT-2/GPT-3) that never flattens to a hard 0
He initialization | the sensible starting-weight recipe for ReLU nets, so fewer units start dead
XOR | "on when exactly one input is on" — the classic pattern one straight line can't split
%%%

That's the whole day. Next you'll stack these neurons into layers and push data through them — the **forward pass**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does an activation function add to a network? | a:1 | More parameters | Non-linearity (a bend) | Faster matmuls | A bias term | fb: The activation is the non-linear bend; without it, layers stay linear and depth is wasted.
q: With NO activation, ten stacked linear layers behave like: | a:1 | a much more powerful model | a single linear layer | a random function | a sigmoid | fb: (x W₁)W₂…W₁₀ = x (W₁…W₁₀) — one combined matrix. Depth buys nothing without a bend.
q: Why does deep sigmoid suffer the "vanishing gradient" while deep ReLU largely doesn't? | a:2 | Sigmoid uses more memory | ReLU is written in C | Sigmoid's slope tops out near 0.25, so multiplying it through many layers shrinks the backward signal toward 0; ReLU's positive slope is 1, so it doesn't shrink | Sigmoid outputs negative numbers | fb: The backward signal is multiplied by each layer's slope. With ≈0.25 per layer it fades fast (0.25⁴≈0.004); ReLU's slope of 1 keeps it full-strength.
q: Your network trains without error, but the loss stops dropping early and stays high. You log the hidden ReLU outputs and find most are exactly 0 for every input in the batch. Most likely explanation — and a fix? | a:1 | Sigmoid at the output is always a bug | Many ReLU units are "dead": their input stays negative for all inputs, so max(0, z) is always 0 and they can no longer learn — switching those units to Leaky ReLU keeps a small negative slope so they can recover | The GPU ran out of memory, which always makes ReLU output zero | A high loss with zero activations means the network has already converged | fb: A ReLU that always sees a negative input outputs a flat 0 for every example — a dead unit. Many dead units mean much of the network does nothing, so the loss plateaus. You spot it from activation statistics (fraction of zeros), and Leaky ReLU (a small negative slope) is the standard cure.
%%%

@@@ produce id=produce tag="Produce" title="Watch two layers collapse — then break the collapse" gotit="Done"
Time to see today's big idea with your own eyes. You'll build two linear layers with no bend between them and watch them fold into a single matrix — then drop a `ReLU` in the middle and watch that collapse break. **Predict first:** will `(x@W1)@W2` and `x@(W1@W2)` print the same numbers, or different ones? Then run it and **watch.** Seeing them match, and then differ the moment the bend goes in, is the whole lesson in two lines of output. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-02-activations/experiment.py`. Write `step`, `relu`, `leaky_relu`, `sigmoid`, and `tanh`, and print all five as a table over the grid `np.linspace(-6, 6, 13)`. **Notice** that the step is a flat 0/1 jump, ReLU zeros negatives, leaky_relu lets a small trickle through the negatives, sigmoid stays in `(0,1)`, and tanh stays in `(−1,1)` and is 0 at 0. Also print `sigmoid'(0) ≈ 0.25` and note that ReLU's slope is `0` for every negative input while leaky_relu's is a small `α` — the seed of both the "dead ReLU" failure and its cure. Then use the anchor pair `W1=[[1,2],[0,1]]`, `W2=[[1,0],[3,1]]` and show that `(x@W1)@W2` equals `x@(W1@W2)` for a random `x` — print `W1@W2` (expect `[[7,2],[3,1]]`), proving two linear layers = one. Run with `python3 sessions/m02-the-neuron/day-02-activations/experiment.py`.

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
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-02-activations/log.md`: (1) why a 100-layer network with no activations is no better than one layer; (2) ReLU vs sigmoid in one line, mentioning the vanishing gradient; (3) what a "dead" or "saturated" unit is, how you'd detect it, and the one-line fix.
!!!

@@@ fin
