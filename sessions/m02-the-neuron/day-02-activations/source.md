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
@lede Welcome back! Try this in your head: draw a smiley face using only a **ruler**. You can't — a ruler makes straight lines, and a face is all curves. To draw anything real, your hand has to *bend*. A neural network hits that exact same wall, and the fix has a name: the **activation** — the "bend." Yesterday you met the neuron's third beat, *decide*, and its two oldest bends: the hard on/off switch and the smooth dimmer. Today we ask the bigger question nobody answered yesterday — *why does a bend have to be there at all?* — and then meet the rest of the family. And here's how far this one little idea reaches: it's what lets your phone decide in a blink that this face is yours, what shapes the feed that picks your next video, and what lets something like ChatGPT curve its way to an answer. Sounds strange that one bend does all that? Good — hold that feeling; by the end it'll feel obvious, and a little bit magic.
@goal We'll line the bends up side by side — yesterday's on/off switch and dimmer, plus the two you haven't met: **ReLU** and **tanh**. You'll see why depth without a bend is a mirage, catch a bend quietly going to sleep, and pocket the tiny fixes that wake it up. Every failure is a little puzzle with a neat answer, and every formula gets said out loud in plain words first. No symbol left unexplained.

%%% warmup
q: Yesterday you built a single neuron. What three beats does it always do, in order? | a:2 | add → weigh → decide | decide → weigh → add | weigh → add → decide | weigh → decide → add | concept: single-neuron | fb: A neuron scales each input by its weight (weigh), sums them with the bias (add), then turns that total into an answer (decide).
q: Yesterday's "decide" step came in two flavours: a hard on/off switch, and a smooth dimmer. Which one lets a *tiny* nudge to the score make a *tiny* change in the answer? | a:1 | the hard on/off switch | the smooth dimmer (sigmoid) | both, equally | neither of them | concept: sigmoid | fb: The switch jumps from 0 to 1 and is flat everywhere else, so a small nudge changes nothing at all. The dimmer slides, so a small nudge makes a small change. Today you'll see why that one difference decides almost everything.
q: A single neuron draws one straight line to split its inputs. Which pattern could it NOT learn on its own? | a:1 | a single blob of "yes" points | XOR (two classes on opposite diagonals) | points all on one side | an empty dataset | concept: xor-limit | fb: XOR sits on two diagonals, so no single straight line separates it — that's the neuron's hard ceiling, and layers are the fix.
q: In a neuron, how many weights and how many biases are there for two inputs? | a:1 | 2 weights and 2 biases | 2 weights and 1 bias | 1 weight and 1 bias | 1 weight and 2 biases | concept: weights-bias | fb: One weight per input (so two), and exactly one bias for the whole neuron — the bias is a single starting-lean number.
%%%

@@@ concept id=c1 tag="The trap" title="Straight + straight is still straight" gotit="Got the trap"
Let's start with a puzzle you can hold in your hands. A neuron does a weighted sum and adds a bias — in plain words, a straight-line rule. Picture a perfectly straight **ruler**.

Now the puzzle: **what do you get if you stack two straight rulers?**

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Two straight rulers laid one on top of the other still make a single straight edge. The everyday picture for two straight-line layers folding into one."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Two straight rulers stacked = still one straight edge</text><rect x="40" y="42" width="260" height="20" rx="3" fill="#FCF3DC" stroke="#C99A12"/><line x1="60" y1="52" x2="280" y2="52" stroke="#9A7A10" stroke-width="1"/><line x1="70" y1="46" x2="70" y2="58" stroke="#9A7A10"/><line x1="110" y1="46" x2="110" y2="58" stroke="#9A7A10"/><line x1="150" y1="46" x2="150" y2="58" stroke="#9A7A10"/><line x1="190" y1="46" x2="190" y2="58" stroke="#9A7A10"/><line x1="230" y1="46" x2="230" y2="58" stroke="#9A7A10"/><text x="312" y="56" fill="#6B645E">ruler 1</text><rect x="40" y="66" width="260" height="20" rx="3" fill="#EFEAF7" stroke="#7C6DAA"/><line x1="60" y1="76" x2="280" y2="76" stroke="#5E5191" stroke-width="1"/><line x1="70" y1="70" x2="70" y2="82" stroke="#5E5191"/><line x1="110" y1="70" x2="110" y2="82" stroke="#5E5191"/><line x1="150" y1="70" x2="150" y2="82" stroke="#5E5191"/><line x1="190" y1="70" x2="190" y2="82" stroke="#5E5191"/><line x1="230" y1="70" x2="230" y2="82" stroke="#5E5191"/><text x="312" y="80" fill="#6B645E">ruler 2</text><text x="260" y="110" text-anchor="middle" fill="#6B645E">↓ stack them ↓</text><rect x="40" y="120" width="260" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><line x1="60" y1="131" x2="280" y2="131" stroke="#C93B3B" stroke-width="1.4"/><text x="316" y="135" fill="#C93B3B">still ONE straight edge</text><text x="170" y="158" text-anchor="middle" fill="#C93B3B" font-size="10">no bend → depth adds nothing</text></g></svg>
%%%

Answer: still one straight edge. Two neuron-layers with no bend between them fold back into one layer. Engineers call that fold-back the **linear collapse**.

**What the ruler gets right:** each layer really is a straight-line rule. **Where it breaks down:** a ruler is one fixed line, while a layer can tune many weights — but however it tunes them, the *shape* stays straight.

%%% insight
This is the sneaky part: a network with no bend still runs, still trains, still prints a loss that drops. Nothing on your screen ever says "your depth is fake." That's exactly why we go looking for the collapse with our own eyes instead of trusting the loss.
%%%

#### Watch two rulers fold into one
Back to the ruler bench — with numbers this time, so you can check every rung yourself. Take the input `4` and give it two straight-line moves in a row.

%%% steps
step: **Move 1 — "triple it."**  4 → 12
why: one straight-line layer: multiply, that's all
step: **Move 2 — "double that, add one."**  12 → 25
why: a second straight-line layer stacked on the first
step: **The fold.** One single rule — "six times it, plus one" — sends 4 → 25 too
why: the two moves quietly merged into ONE straight-line move; this is the linear collapse
step: **The bad news.** So two layers = one layer
why: a third, or a hundredth, folds in the same way — depth alone buys nothing
%%%

Don't take my word for it. Below, the two stacked moves are drawn as a thick line and the single combined move as a dashed line — and the dashed one lands right on top of the thick one, because they are the very same line:

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="Two straight-line steps, triple-it then double-and-add-one, applied to 4 give 25, exactly matching the single combined rule six-times-plus-one. Below, both rules are drawn as one and the same straight line: the thick line for the two stacked moves lies exactly on top of the dashed line for the single move."><g font-family="monospace" font-size="12" text-anchor="middle">
<text x="45" y="35" fill="#3A342E" font-weight="bold">4</text>
<rect x="70" y="20" width="88" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="114" y="37" fill="#5E5191">×3 → 12</text>
<rect x="178" y="20" width="120" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="238" y="37" fill="#5E5191">×2 +1 → 25</text>
<text x="330" y="37" fill="#6B645E">=</text>
<rect x="352" y="18" width="140" height="30" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="422" y="38" fill="#C93B3B" font-weight="bold">6×4 +1 = 25</text>
<line x1="96" y1="66" x2="128" y2="66" stroke="#7C6DAA" stroke-width="7" opacity="0.4"/>
<text x="136" y="70" fill="#5E5191" font-size="10" text-anchor="start">thick = the two stacked moves</text>
<line x1="96" y1="86" x2="128" y2="86" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="5,4"/>
<text x="136" y="90" fill="#C93B3B" font-size="10" text-anchor="start">dashed = the one combined move — it lands on exactly the same line</text>
<line x1="60" y1="195" x2="470" y2="195" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="80" y1="100" x2="80" y2="201" stroke="#E5DFD6" stroke-width="1"/>
<path d="M80 190 L440 110" fill="none" stroke="#7C6DAA" stroke-width="7" opacity="0.4"/>
<path d="M80 190 L440 110" fill="none" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="5,4"/>
<text x="272" y="178" fill="#C93B3B" font-size="10" text-anchor="middle">one straight line, drawn twice</text>
<text x="486" y="199" fill="#9A938A" font-size="10" text-anchor="middle">input</text>
</g></svg>
%%%

**So far:** two straight moves are always secretly one straight move.

#### The same story with real layers
A layer's weights live in a grid of numbers called a matrix, written `W`. Stack two and the grid-math folds them together exactly like `×3` then `×2` folded into `×6`:

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Two linear layers W1 then W2 collapse into a single matrix W1 times W2"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="40" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="70" y="65" fill="#5E5191">W₁</text><text x="115" y="65" fill="#6B645E">then</text><rect x="150" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="180" y="65" fill="#5E5191">W₂</text><text x="250" y="65" fill="#6B645E">=</text><rect x="290" y="40" width="140" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="360" y="65" fill="#C93B3B" font-weight="bold">one matrix W₁@W₂</text><text x="260" y="105" fill="#C93B3B" font-size="11">no activation → two layers become one</text></g></svg>
%%%

You'll prove that yourself in **Produce** — a couple of lines of output, one big idea.

!!! c-info 🪜
<b>Optional peek (skippable).</b> The one-line version of the fold: two straight-line layers back to back are <code>(x·W₁)·W₂</code>, and grid-math lets you re-bracket that as <code>x·(W₁·W₂)</code> — a single set of weights. The biases fold too: <code>b₁·W₂ + b₂</code> is just one new bias. So the whole stack is still one straight-line-plus-offset rule. Nothing was gained; the second layer never had a chance to add anything new.
!!!

!!! c-warn 😕
<b>这一步很多人一开始都会觉得奇怪，很正常 · This step feels strange to almost everyone at first — that's completely normal.</b> The activation looks like an optional extra tacked on at the end, so it's easy to shrug off. But without it, 100 layers behave like <b>one</b>. Sit with that for a moment — it's the whole reason today matters. And you already understand it, which is a great place to be four minutes in.
!!!

@@@ concept id=c2 tag="The fix" title="A bend between the layers" gotit="Got the fix"
So how do we stop the collapse? We slip a **bend** between the layers.

Picture a sheet of **paper**. Flat paper on flat paper stays flat. Fold one crease into it and suddenly it stands up in shapes a flat sheet could never make. That crease is the whole idea.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="Left: a flat sheet of paper stays flat. Right: the same sheet with one folded crease stands up into a shape. A single crease unlocks shapes flat paper cannot make."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One crease unlocks shapes flat paper can't make</text><polygon points="40,120 200,120 220,90 60,90" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="50" y1="110" x2="200" y2="110" stroke="#E5DFD6"/><text x="128" y="145" text-anchor="middle" fill="#6B645E">flat paper</text><text x="128" y="160" text-anchor="middle" fill="#9A938A" font-size="10">stays flat (no shape)</text><text x="255" y="105" fill="#B8AEA2" font-size="16">→</text><path d="M300 130 L390 130 L410 100 L360 60 L340 95 Z" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><line x1="360" y1="60" x2="340" y2="95" stroke="#5E5191" stroke-width="2.5"/><line x1="340" y1="95" x2="390" y2="130" stroke="#5E5191" stroke-width="1"/><text x="420" y="92" fill="#5E5191">the crease</text><text x="420" y="106" fill="#5E5191">= the bend</text><text x="350" y="152" text-anchor="middle" fill="#2D8B55">creased paper stands up</text><text x="350" y="166" text-anchor="middle" fill="#9A938A" font-size="10">can't be squashed flat</text></g></svg>
%%%

That bend has a name: the [[activation function||A simple function applied at the end of a neuron, after the weighted sum + bias. It adds non-linearity so stacked layers can learn complex shapes.]] — a small [[non-linearity||Not a straight line. A non-linear step between layers lets a deep network bend and combine features to learn complex patterns.]] applied at the very end of a neuron. Yesterday you called it the "decide" step; today you get to see what it is really *for*.

**What the crease gets right:** one fold unlocks shapes flat paper can't make. **Where it breaks down:** paper only creases where *you* fold it, while an activation bends the signal at *every* neuron on *every* example — far more folds than your hands could ever make.

%%% insight
Notice how little this costs. You are not adding layers, parameters, or data — you're adding one tiny bend per neuron, and that alone is what turns "deep" from a marketing word into real extra power.
%%%

#### Where the bend goes, in three beats
Let's follow one number through a layer, so you know exactly which step is the bend.

%%% steps
step: **Mix.** The weights combine the inputs into one running total, `z`
why: this is the straight-line part — mixing is the weights' job, and only theirs
step: **Bend.** The activation reshapes that single number, one number at a time
why: it takes one number in and gives one number out — that's why we call it the bend
step: **Pass on.** The bent value becomes the input of the next layer
why: therefore the next layer starts from something curved, and can't be folded back into the first
%%%

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Inserting ReLU between the two layers stops the collapse"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="30" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="57" y="65" fill="#5E5191">W₁</text><rect x="100" y="45" width="70" height="30" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="135" y="65" fill="#1a5c38" font-weight="bold">ReLU</text><rect x="185" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="212" y="65" fill="#5E5191">W₂</text><text x="275" y="65" fill="#6B645E">≠</text><rect x="305" y="45" width="150" height="30" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2" stroke-dasharray="4,3"/><text x="380" y="65" fill="#6B645E">any single W</text><text x="250" y="105" fill="#2D8B55" font-size="11">a bend in the middle → cannot be folded flat</text></g></svg>
%%%

**So far:** weights mix, the bend curves, and that one curve is what keeps the layers separate. One honest limit while we're here: every bend you'll meet today works on **one number at a time**, so it can't build new combinations of features — combining inputs stays the weights' job. (There is exactly one exception, a bend called *softmax* that sits at the very end and looks at a whole row at once; that's Day 4's story.)

And that's the whole secret. **You just unlocked why deep learning is even possible.** The rest of today is meeting the bends themselves — starting with the one you already met yesterday.

@@@ concept id=c3 tag="The first bend" title="The step function — a hard on/off switch" gotit="Met the step"
Here's the on/off switch from yesterday again — but this time we put it on trial, because today you know what a bend is *for*.

Picture the light switch on your wall. Flip it and the light is fully **on** or fully **off** — nothing in between.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A wall light switch in two states. On the left the switch is down, the bulb is dark and off. On the right the switch is up, the bulb is bright and on. Nothing in between."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A wall switch: fully OFF or fully ON — no in-between</text><rect x="60" y="40" width="70" height="100" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><rect x="84" y="92" width="22" height="40" rx="4" fill="#D8D2C8" stroke="#8f8a80"/><text x="95" y="156" text-anchor="middle" fill="#6B645E">switch down</text><circle cx="200" cy="70" r="20" fill="#EFEAF0" stroke="#9A938A" stroke-width="1.5"/><text x="200" y="76" text-anchor="middle" font-size="16">🌑</text><text x="200" y="120" text-anchor="middle" fill="#6B645E">input &lt; 0</text><text x="200" y="136" text-anchor="middle" fill="#6B645E" font-weight="bold">output 0 (off)</text><rect x="330" y="40" width="70" height="100" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><rect x="354" y="48" width="22" height="40" rx="4" fill="#F2D97A" stroke="#9A7A10"/><text x="365" y="156" text-anchor="middle" fill="#6B645E">switch up</text><circle cx="470" cy="70" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="470" y="76" text-anchor="middle" font-size="16">💡</text><text x="470" y="120" text-anchor="middle" fill="#8A6D3B">input ≥ 0</text><text x="470" y="136" text-anchor="middle" fill="#8A6D3B" font-weight="bold">output 1 (on)</text></g></svg>
%%%

That is the [[step function||The original activation: output 1 if the input is at or above zero, else 0. A hard on/off switch with no in-between — and slope 0 everywhere, so a network can't learn from it.]], the earliest of all these bends — the one yesterday's theme-park height bar was describing. In words: input at or above zero → output `1` (the neuron "fires"); below zero → output `0`.

**What the switch gets right:** all-or-nothing, no middle ground. **Where it breaks down:** a wall switch is flipped by *you*; here the *input* flips it — and there's no dimming, which turns out to be fatal.

%%% demo id=step label="predict, then run"
predict: what does the step function give for the inputs −2, −1, 0, 1, 2? Guess the five outputs before you peek — where exactly does it flip?
code: step(np.array([-2, -1, 0, 1, 2]))
out: array([0., 0., 1., 1., 1.])
take: <b>Step = 1 if z ≥ 0 else 0.</b> Everything below zero is 0, everything at or above zero is 1. (They print as `0.` and `1.` because the code returns decimals.) Notice what's missing: any sense of "how much." Only yes or no.
%%%

#### The switch's fatal flaw
Here's the puzzle it leaves us: the step function looks tidy, so why does nobody use it inside a network today? Look at its shape.

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Step function: output 0 for negative inputs, jumps to 1 at zero, then flat at 1 for positive inputs"><g font-family="monospace"><line x1="60" y1="100" x2="330" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><line x1="195" y1="30" x2="195" y2="118" stroke="#E5DFD6" stroke-width="1"/><path d="M70 100 L195 100 L195 48 L305 48" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="195" cy="48" r="3.5" fill="#8A6D3B"/><text x="125" y="117" font-size="11" fill="#6B645E" text-anchor="middle">output 0 (off)</text><text x="255" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">output 1 (on)</text><text x="415" y="72" font-size="12" fill="#8A6D3B">step → {0, 1}</text><text x="195" y="133" font-size="10" fill="#9A938A" text-anchor="middle">jump at 0 — no in-between, and flat (slope 0) everywhere else</text></g></svg>
%%%

Flat, flat, jump, flat. And "flat" is exactly the wrong thing to be:

%%% steps
step: **The nudge.** Learning improves a network by nudging weights in whichever direction lowers the error
why: to nudge, it needs a hint about which way is "downhill"
step: **The flat stretch.** Wiggle the input of a step function and the output doesn't budge at all
why: therefore there is no hint — flat means no direction, no signal
step: **The escape.** Keep the S-shape, but make it *smooth* so it always has some tilt
why: which is exactly what sigmoid and tanh do — that swap is why yesterday's dimmer exists
%%%

%%% insight
Feel the shape of today's whole story here: the enemy is never the curve you can see — it's **flatness**. Every failure you'll meet from now on is some part of a bend going flat, and every cure puts the tilt back.
%%%

**So far:** the step switch works, but it can't teach. And now you can explain why the very first neural networks stalled for years — nice.

@@@ concept id=c4 tag="Meet ReLU" title="ReLU — a one-way valve" gotit="Met ReLU"
Now the first bend that's brand new to you — and it's the one that actually runs the modern world. It is almost as simple as the switch, but it keeps a tilt.

Picture a **one-way valve** in a water pipe: push from the front and water flows straight through; push from the back and it's blocked.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve in a pipe. Top: a positive signal pushes forward and water flows straight through. Bottom: a negative signal is blocked and nothing passes, output zero."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">ReLU = a one-way valve</text><rect x="40" y="38" width="440" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><polygon points="250,42 262,55 250,68" fill="#2D8B55"/><line x1="250" y1="38" x2="250" y2="72" stroke="#2D8B55" stroke-width="1"/><text x="90" y="59" fill="#276b45">positive →</text><text x="330" y="59" fill="#276b45" font-weight="bold">flows through 💧</text><text x="470" y="59" text-anchor="end" fill="#2D8B55">→</text><rect x="40" y="96" width="440" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><line x1="250" y1="92" x2="250" y2="134" stroke="#C93B3B" stroke-width="3"/><rect x="244" y="100" width="12" height="26" fill="#C93B3B"/><text x="95" y="117" fill="#8a3b3b">negative ←</text><text x="330" y="117" fill="#C93B3B" font-weight="bold">BLOCKED → 0</text><text x="260" y="154" text-anchor="middle" fill="#6B645E" font-size="10">flows one way, blocked the other — that's max(0, z)</text></g></svg>
%%%

That is [[ReLU||A one-way valve: it lets positive numbers pass and turns negatives into 0. The modern default activation.]]: positives pass through untouched, negatives are shut off to `0`. One line of plain words — and it is the bend inside almost every network you use without noticing: the one that unlocks your phone with your face, and the one that picks the next video in your feed. The big language models use a close cousin of it — a smoothed ReLU called **GELU**, which you'll meet in a moment.

**What the valve gets right:** flow one way, blocked the other. **Where it breaks down:** a real valve is open or shut, but ReLU passes positives at their *full* size — 5 stays 5, 100 stays 100.

%%% demo id=relu label="predict, then run"
predict: before you look — what does ReLU do to −3, −1, 0, 2, 5? Which numbers survive, and do the survivors change size?
code: relu(np.array([-3, -1, 0, 2, 5]))
out: array([0, 0, 0, 2, 5])
take: <b>ReLU = max(0, z)</b> — "take the bigger of 0 and z." Negatives become 0; positives come out completely unchanged.
%%%

Said out loud, the formula is friendly: `max(0, z)` means "keep whichever is bigger, zero or z." If `z` is positive, `z` wins. If `z` is negative, `0` wins. That's it — no `e`, no exponent, one comparison.

#### Three gifts from one valve
Back at the pipe: the valve's shape hands us three separate wins. Watch each one appear.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="ReLU curve: flat at zero for negatives, a straight rising line for positives"><g font-family="monospace"><line x1="60" y1="95" x2="320" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="190" y1="25" x2="190" y2="110" stroke="#E5DFD6" stroke-width="1"/><path d="M70 95 L190 95 L300 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">negatives → 0</text><text x="260" y="30" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text x="400" y="70" font-size="13" fill="#1a5c38" font-family="monospace">ReLU(x)=max(0,x)</text></g></svg>
%%%

%%% steps
step: **Gift 1 — a real tilt.** On the positive side the line rises at a steady slope of `1`
why: unlike the flat step switch, learning gets a clear "this way" signal there
step: **Gift 2 — it's dirt cheap.** One comparison per number, no exponent to compute
why: which means you can afford millions of them in every layer of a big model
step: **Gift 3 — free [[sparsity||Many neurons output exactly 0 at the same time, so only part of the network is active per example — cheaper, and "only the units that matter speak up."]].** Roughly half the units get a negative input, so they output exactly `0`
why: therefore only the units that matter speak up on any one example — quieter and cheaper
%%%

Its range is `[0, ∞)` — zero and up, with no ceiling. Push the input higher and the output keeps climbing; ReLU never flattens on the positive side, which is the property the next few units all lean on.

%%% insight
One question people love to ask: what's the slope at exactly `z = 0`, where the two straight pieces meet in a kink? Honest answer: there isn't one. Everyone just agrees to call it `0` and moves on — landing *exactly* on zero in floating-point numbers basically never happens.
%%%

**So far:** cheap, tilted, sparse — that trio is why ReLU is the default bend almost everywhere.

@@@ concept id=c5 tag="Meet sigmoid" title="Sigmoid — a soft dimmer" gotit="Met sigmoid"
Back to the gentle bend you already slid yesterday — the one behind every "how confident are you, 0 to 100%?" answer a model ever gives. Today we look at its two flat ends, and at one quirk that made people invent a whole other bend.

Picture that **dimmer switch** instead of an on/off switch. Slide it and the room brightens smoothly through every level in between.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A dimmer switch as a slider track. On the left, the slider is low and the bulb is dark near zero. In the middle it is half bright at one half. On the right it is full and the bulb is fully bright near one. It slides smoothly through every level."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid = a dimmer that slides smoothly off → on</text><rect x="60" y="70" width="400" height="14" rx="7" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="110" cy="77" r="12" fill="#5E5191"/><text x="110" y="106" text-anchor="middle" fill="#6B645E">low</text><text x="110" y="120" text-anchor="middle" fill="#9A938A" font-size="10">→ near 0</text><circle cx="260" cy="52" r="16" fill="#EFEAF0" stroke="#9A938A"/><text x="260" y="58" text-anchor="middle" font-size="14">🌑</text><circle cx="350" cy="48" r="18" fill="#F6EFCF" stroke="#C99A12"/><text x="350" y="54" text-anchor="middle" font-size="14">🔆</text><circle cx="430" cy="44" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="430" y="51" text-anchor="middle" font-size="16">💡</text><text x="260" y="106" text-anchor="middle" fill="#5E5191">middle</text><text x="260" y="120" text-anchor="middle" fill="#9A938A" font-size="10">= 0.5</text><text x="430" y="106" text-anchor="middle" fill="#8A6D3B">full</text><text x="430" y="120" text-anchor="middle" fill="#9A938A" font-size="10">→ near 1</text><text x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">no hard jump — every brightness in between lives in (0, 1)</text></g></svg>
%%%

That smooth slide is [[sigmoid||A soft dimmer switch that gently squashes any number into the range 0 to 1.]]. Hand it any number at all and it hands back a brightness between `0` and `1` — never below, never above.

**What the dimmer gets right:** the gradual slide, no hard jump. **Where it breaks down:** a real dimmer is evenly graded all the way along, but sigmoid *flattens* at both ends — slide past a point and the bulb barely changes. Hold onto that; it's a puzzle we crack shortly.

%%% demo id=sigmoid label="predict, then run"
predict: sigmoid turns any number into a brightness in (0, 1). Guess its answer for −2, 0 and 2 — and especially: what do you think sigmoid(0) is?
code: sigmoid(np.array([-2, 0, 2]))
out: array([0.119, 0.5  , 0.881])
take: <b>Sigmoid squashes into (0, 1).</b> Dead centre, sigmoid(0) = 0.5 — half brightness. Big negatives land near 0, big positives near 1 (values rounded; sigmoid(2) ≈ 0.8808).
%%%

#### Reading its shape, one feature at a time
Same dimmer, now drawn as a curve on real axes — the floor of the picture is `0`, the dashed ceiling is `1`, and the dot marks half brightness right at `z = 0`:

%%% svg
<svg viewBox="0 0 520 145" role="img" aria-label="Sigmoid curve on labelled axes. The horizontal axis is the output value zero. A dashed ceiling line marks one. The S-shaped curve rises from just above zero on the left, passes exactly through one half at z equals zero, and levels off just under one on the right, so the whole curve stays inside zero to one."><g font-family="monospace"><line x1="60" y1="110" x2="330" y2="110" stroke="#E5DFD6" stroke-width="1.5"/><line x1="60" y1="50" x2="330" y2="50" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/><text x="56" y="54" font-size="10" fill="#7C6DAA" text-anchor="end">1.0</text><text x="56" y="114" font-size="10" fill="#9A938A" text-anchor="end">0.0</text><line x1="190" y1="40" x2="190" y2="122" stroke="#E5DFD6" stroke-width="1"/><path d="M70 109 C 130 109, 162 82, 190 80 C 218 78, 250 51, 310 50" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><circle cx="190" cy="80" r="3.5" fill="#C93B3B"/><text x="152" y="74" font-size="10" fill="#C93B3B" text-anchor="end">0.5 at z=0</text><text x="115" y="126" font-size="11" fill="#6B645E" text-anchor="middle">→ near 0</text><text x="272" y="44" font-size="11" fill="#5E5191" text-anchor="middle">→ near 1</text><text x="398" y="86" font-size="12" fill="#5E5191" font-family="monospace">sigmoid → (0,1)</text><text x="190" y="138" font-size="10" fill="#9A938A" text-anchor="middle">the curve never leaves the band between 0 and 1</text></g></svg>
%%%

%%% steps
step: **The range.** Every output lands strictly between `0` and `1`
why: which is why sigmoid reads like a confidence dial — perfect for "how sure are you?"
step: **The one-way climb.** Bigger input always means brighter output, never dimmer
why: it only ever goes up, so the answer is never ambiguous
step: **The two flat ends.** Far out on either side the curve lies almost flat
why: therefore a very confident unit stops reacting — remember flat means no learning signal
%%%

%%% insight
This is the bend that made deep learning *possible* in the 1990s and then quietly held it back for a decade. Same curve, two opposite verdicts — the reason is hiding in those flat ends, and you're a couple of units away from seeing it.
%%%

In symbols it's `sigmoid(z) = 1 / (1 + e^(−z))`: "one, divided by one plus e-to-the-minus-z." You never compute that by hand.

!!! c-info 🧪
<b>Optional (skippable) — one practical wrinkle.</b> If you ever hand-write that formula, a very negative <code>z</code> makes <code>e^(−z)</code> enormous and your computer can overflow to <code>inf</code>. The fix is free: call the library version (<code>torch.sigmoid</code>, <code>jax.nn.sigmoid</code>), which handles the two sides separately and never blows up.
!!!

#### One more quirk: the dimmer only ever pushes one way
Here's a small, gentler mystery — and it's the reason the *next* bend exists. Every sigmoid output is a positive brightness. Nothing in `(0, 1)` is ever negative. Sounds harmless, right?

Picture pushing a **shopping cart with one stuck wheel** that always pulls a little to the left. You still reach the milk. You just zig-zag the whole way there.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A shopping cart with one wheel stuck turned to the side so it always pulls left. Its path zig-zags across the floor toward the goal instead of rolling straight, so it takes longer."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid's one-sided pull = a cart with one stuck wheel</text><rect x="48" y="48" width="70" height="40" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><line x1="56" y1="56" x2="56" y2="88" stroke="#7C6DAA"/><line x1="110" y1="48" x2="122" y2="36" stroke="#7C6DAA" stroke-width="1.5"/><circle cx="122" cy="34" r="4" fill="#7C6DAA"/><circle cx="62" cy="98" r="9" fill="#FDF9F3" stroke="#6B645E" stroke-width="1.5"/><circle cx="104" cy="98" r="9" fill="#FDF9F3" stroke="#C93B3B" stroke-width="2.5"/><line x1="98" y1="92" x2="110" y2="104" stroke="#C93B3B" stroke-width="2"/><text x="104" y="124" text-anchor="middle" fill="#C93B3B" font-size="9">stuck wheel</text><path d="M150 100 L200 66 L245 100 L295 62 L340 100 L390 60 L435 88" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="300" y="122" text-anchor="middle" fill="#C93B3B" font-size="10">always-positive nudges → zig-zag, slow</text><path d="M150 100 L435 60" fill="none" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="5,4"/><text x="320" y="72" fill="#2D8B55" font-size="10">straight (zero-centered)</text><circle cx="440" cy="58" r="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="440" y="46" text-anchor="middle" fill="#6B645E" font-size="10">goal</text></g></svg>
%%%

**What the stuck wheel gets right:** a constant sideways pull really does force a zig-zag. **Where it breaks down:** a stuck cart may never arrive, while a zig-zagging network usually does — it simply takes more steps.

%%% steps
step: **The one-sided output.** Sigmoid can only ever hand the next layer *positive* numbers
why: everything it produces lives in `(0, 1)`, so a negative value is impossible for it
step: **The shared shove.** So the nudges arriving at a downstream weight all point the same way at once
why: which means the network can't correct one direction without dragging the others along — that's the stuck wheel
step: **The fix, in one word: balance.** We want a bend whose outputs sit evenly above *and* below `0`
why: therefore nudges come in both signs, the sideways pull cancels, and the path straightens
%%%

%%% insight
Nothing breaks here — no crash, no error. The network still arrives; it just wanders on the way. Keep that in your pocket: most of today's failures are like this. They don't break your code, they quietly tax it.
%%%

**So far:** sigmoid is a smooth confidence dial with two suspiciously flat ends and a one-sided pull. The bend that fixes the pull is next — and it has the best analogy of the day.

@@@ concept id=c6 tag="Meet tanh" title="tanh — the balanced see-saw" gotit="Met tanh"
So we want a smooth bend that pushes both ways. Someone had exactly that idea, and the everyday picture for it is the best one today — plus there's a live widget at the end so you can feel all these bends under your own finger.

Picture a playground **see-saw**: a plank resting on a bar in the middle. Nobody on it → perfectly level. Push the left side down hard → left end hits the ground. Push the right side → right end goes all the way up.

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A playground see-saw. A plank rests on a central triangular pivot, tilted so its left end drops to minus one and its right end rises to plus one, resting level at zero in the middle."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">tanh = a balanced see-saw: −1 … 0 … +1</text><line x1="90" y1="120" x2="430" y2="70" stroke="#9A5A12" stroke-width="6" stroke-linecap="round"/><polygon points="260,95 240,140 280,140" fill="#C99A12" stroke="#9A7A10"/><line x1="170" y1="140" x2="350" y2="140" stroke="#B8AEA2" stroke-width="2"/><circle cx="90" cy="120" r="7" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="150" text-anchor="middle" fill="#C93B3B">push left</text><text x="90" y="164" text-anchor="middle" fill="#C93B3B" font-weight="bold">−1</text><circle cx="260" cy="95" r="5" fill="#6B645E"/><text x="290" y="90" fill="#6B645E">rests level</text><text x="260" y="64" text-anchor="middle" fill="#2D8B55" font-weight="bold">0</text><circle cx="430" cy="70" r="7" fill="#EAF5EE" stroke="#2D8B55"/><text x="430" y="58" text-anchor="middle" fill="#276b45" font-weight="bold">+1</text><text x="430" y="150" text-anchor="middle" fill="#276b45">push right</text></g></svg>
%%%

Put numbers on the plank: lowest is `−1`, highest is `+1`, resting is exactly `0`. That balanced swing is [[tanh||Squashes any number into (−1, 1), and passes through the origin: tanh(0)=0. It is zero-centered, so its outputs are balanced above and below zero.]] — and because its outputs sit evenly above and below zero, we call it **zero-centered**. That is the cure for the stuck wheel you just met.

**What the see-saw gets right:** the balanced tilt around a level middle. **Where it breaks down:** a real see-saw keeps tilting the harder you push, but tanh flattens near both ends — the same flattening as sigmoid.

%%% demo id=tanh label="predict, then run"
predict: tanh is the see-saw. What does it give for −2, 0 and 2? The middle one is the giveaway — what should a see-saw with nobody on it read?
code: tanh(np.array([-2, 0, 2]))
out: array([-0.964,  0.   ,  0.964])
take: <b>tanh squashes into (−1, 1) and rests at 0.</b> Input 0 leaves it level, a big negative tips it toward −1, a big positive toward +1 — balanced above and below zero. That's "zero-centered," so the cart rolls straight.
%%%

#### One word to carry forward: slope
tanh and sigmoid are close cousins — same smooth S, different plank height and resting point. Sigmoid runs `0 → 1` and rests at `0.5`; tanh runs `−1 → +1` and rests at `0`. To see *why* that difference matters, the rest of the day turns on a single word, so let's make it concrete. The **slope** of a bend at a point is *how steep the curve is right there* — how much the output moves when you nudge the input a tiny bit.

%%% steps
step: **Steep slope** → the neuron reacts strongly to small changes
why: lots of tilt means a clear "which way to nudge" signal, so learning moves fast
step: **Slope near 0 (flat)** → the neuron barely reacts at all
why: no tilt, no direction — learning there crawls to a stop, exactly the step switch's problem
step: **So each bend has a slope story, not just a shape**
why: that's why the picture below draws the curves on top and their slopes underneath
%%%

The bottom half of this picture is the one to look at slowly — it's the whole next unit in advance:

%%% svg
<svg viewBox="0 0 520 380" role="img" aria-label="Top panel: ReLU, sigmoid and tanh drawn on shared labelled axes with gridlines at minus one, zero and plus one. Sigmoid runs from zero up through one half at z equals zero to one; tanh runs from minus one through zero up to plus one; ReLU is flat at zero for negatives then rises straight past one. Bottom panel: their slopes, with sigmoid's slope peaking at about 0.25 and tanh's at about 1.0, both at z equals zero, and ReLU's slope a flat 0 then a flat 1."><g font-family="monospace" font-size="12">
<text x="60" y="22" font-size="13" fill="#3A342E" font-weight="bold">The bend  f(z)</text>
<rect x="292" y="13" width="10" height="10" fill="#2D8B55"/><text x="307" y="22" fill="#2D8B55" font-size="11" text-anchor="start">ReLU</text>
<rect x="352" y="13" width="10" height="10" fill="#7C6DAA"/><text x="367" y="22" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid</text>
<rect x="432" y="13" width="10" height="10" fill="#9A5A12"/><text x="447" y="22" fill="#9A5A12" font-size="11" text-anchor="start">tanh</text>
<line x1="60" y1="65" x2="470" y2="65" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<line x1="60" y1="110" x2="470" y2="110" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="60" y1="155" x2="470" y2="155" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="69" fill="#9A938A" font-size="10" text-anchor="end">+1</text>
<text x="56" y="114" fill="#9A938A" font-size="10" text-anchor="end">0</text>
<text x="56" y="159" fill="#9A938A" font-size="10" text-anchor="end">−1</text>
<line x1="265" y1="36" x2="265" y2="162" stroke="#E5DFD6" stroke-width="1"/>
<text x="477" y="114" fill="#9A938A" font-size="10" text-anchor="start">z</text>
<path d="M70 156 C 190 154, 228 113, 265 110 C 300 107, 330 66, 460 64" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<path d="M70 108 C 195 107, 238 90, 265 87.5 C 292 85, 350 68, 460 66" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<path d="M70 110 L265 110 L440 46" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<circle cx="265" cy="87.5" r="3.5" fill="#C93B3B"/>
<text x="256" y="80" fill="#C93B3B" font-size="10" text-anchor="end">sigmoid(0)=0.5</text>
<text x="60" y="212" font-size="13" fill="#3A342E" font-weight="bold">The slope  f′(z)</text>
<line x1="60" y1="340" x2="470" y2="340" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="265" y1="230" x2="265" y2="352" stroke="#E5DFD6" stroke-width="1"/>
<text x="477" y="344" fill="#9A938A" font-size="10" text-anchor="start">z</text>
<line x1="60" y1="248" x2="470" y2="248" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="252" fill="#9A5A12" font-size="10" text-anchor="end">1.0</text>
<line x1="60" y1="317" x2="470" y2="317" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="321" fill="#7C6DAA" font-size="10" text-anchor="end">0.25</text>
<path d="M70 340 L265 340 L265 248 L460 248" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<path d="M70 339 C 200 338, 235 252, 265 248 C 295 252, 330 338, 460 339" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<path d="M70 340 C 205 339, 240 319, 265 317 C 290 319, 325 339, 460 340" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<text x="272" y="244" fill="#9A5A12" font-size="11" text-anchor="start">tanh′ peaks ≈ 1.0 at z=0</text>
<text x="272" y="313" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid′ peaks ≈ 0.25 at z=0</text>
<text x="450" y="240" fill="#2D8B55" text-anchor="end" font-weight="bold">ReLU′ = 1 (z&gt;0)</text>
<text x="130" y="333" fill="#2D8B55" font-size="10" text-anchor="middle">ReLU′ = 0 (z&lt;0)</text>
</g></svg>
%%%

Three numbers off that bottom panel, and they explain the next hour of your life:

- **sigmoid** — slope never gets above about `0.25`
- **tanh** — slope reaches about `1.0`
- **ReLU** — slope is either a flat `0` (negatives) or a plain `1` (positives)

#### Now go bend them yourself
Enough reading — this one is yours to play with. Pick an activation, drag the marker along `z`, and watch the slope reading change. Try to **call the number before you let go**: drag into either tail of sigmoid or tanh, or into ReLU's flat left side, and see how close to `0` you can drive it.

%%% viz src=../../viz/activation-derivatives.html title="activations and their slopes" caption="interactive — pick an activation, drag z; watch the slope flatten"
%%%

%%% insight
That flattening you just felt with your finger is the split that organises every activation ever invented. **Bounded and saturating** (sigmoid, tanh): they squash into a fixed range and go flat at the ends. **Unbounded and non-saturating** (the ReLU family): no ceiling, never flat on the positive side. Almost every choice you'll make comes down to which side of that line you want.
%%%

(The exact recipe for a slope is called the *derivative*, and you'll do that math on Day 5. Today the picture is genuinely enough.)

#### Victory lap — you've met the whole family
Breathe. You now know the four bends every deep-learning engineer can name — three you'll actually use, and one you'll know why nobody uses:

- **step** — the crude switch: flat, can't learn (the museum piece)
- **ReLU** — the valve: cheap, keeps a tilt for positives
- **sigmoid** — the dimmer: smooth `0 → 1`, one-sided
- **tanh** — the see-saw: smooth `−1 → +1`, balanced at `0`

One honest footnote so you're never caught out: **ReLU is one-sided too** — its outputs are all `≥ 0`, so it has the same drift as sigmoid. The field accepts that trade because ReLU's slope of `1` pays for it, and normalization layers (a later day) clean up the rest.

**So far:** four bends, and one number each — their slope. Next we find out what happens when that number goes flat. Nothing to memorise; a side-by-side table is waiting for you in the recap.

@@@ concept id=c7 tag="The whisper" title="The whisper down the line" gotit="Got the whisper"
Here's the first real mystery of the day, and it's a lovely one: a perfectly good smooth bend can quietly *stop learning*.

Picture a line of friends passing a **whisper** from the back of a room to the front. Each person hears a bit less than the one before. By the fourth person it's a mumble; by the tenth, silence.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="Five people standing in a line passing a whisper forward. The first speaks at full strength 1.0, the next hears about 0.25, the next about 0.06, the next about 0.016, and the last hears only about 0.004 — a faint mumble. Labels under each person count how many layers back they are, from zero back to four back."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A whisper down a line — the learning signal fades</text>
<circle cx="58" cy="104" r="19" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="58" y="110" text-anchor="middle" font-size="16">🗣️</text><rect x="34" y="58" width="48" height="22" rx="7" fill="#EFEAF7" stroke="#7C6DAA"/><text x="58" y="74" text-anchor="middle" fill="#5E5191">1.0</text>
<text x="100" y="110" fill="#B8AEA2">→</text>
<circle cx="155" cy="104" r="19" fill="#EFEAF7" stroke="#9A93C0"/><text x="155" y="110" text-anchor="middle" font-size="14">👂</text><rect x="132" y="60" width="46" height="20" rx="6" fill="#EFEAF7" stroke="#9A93C0"/><text x="155" y="74" text-anchor="middle" fill="#7C6DAA" font-size="10">≈0.25</text>
<text x="197" y="110" fill="#B8AEA2">→</text>
<circle cx="253" cy="104" r="19" fill="#F4F1FA" stroke="#C7BEDF"/><text x="253" y="110" text-anchor="middle" font-size="14">👂</text><rect x="231" y="62" width="44" height="18" rx="6" fill="#F4F1FA" stroke="#C7BEDF"/><text x="253" y="75" text-anchor="middle" fill="#9A93C0" font-size="9">≈0.06</text>
<text x="295" y="110" fill="#B8AEA2">→</text>
<circle cx="351" cy="104" r="19" fill="#F8F6FC" stroke="#D9D3EC"/><text x="351" y="110" text-anchor="middle" font-size="14">👂</text><rect x="328" y="63" width="48" height="17" rx="6" fill="#F8F6FC" stroke="#D9D3EC"/><text x="352" y="76" text-anchor="middle" fill="#9A93C0" font-size="9">≈0.016</text>
<text x="393" y="110" fill="#B8AEA2">→</text>
<circle cx="449" cy="104" r="19" fill="#FDECEC" stroke="#C93B3B"/><text x="449" y="110" text-anchor="middle" font-size="14">😕</text><text x="449" y="76" text-anchor="middle" fill="#C93B3B" font-size="9">≈0.004</text>
<text x="58" y="138" text-anchor="middle" fill="#9A938A" font-size="9">0 back</text><text x="155" y="138" text-anchor="middle" fill="#9A938A" font-size="9">1 back</text><text x="253" y="138" text-anchor="middle" fill="#9A938A" font-size="9">2 back</text><text x="351" y="138" text-anchor="middle" fill="#9A938A" font-size="9">3 back</text><text x="449" y="138" text-anchor="middle" fill="#C93B3B" font-size="9">4 back</text>
<text x="260" y="166" text-anchor="middle" fill="#C93B3B" font-size="10">×0.25 at every hand-off → four layers back it is ≈0.004 → the vanishing gradient</text></g></svg>
%%%

That whisper is the **learning signal** travelling backward through the layers, telling each early layer how to improve. Each layer it passes gets multiplied by that layer's slope — so a small slope means a quieter whisper. ("0 layers back" is where the signal starts, at the end of the network.)

**What the whisper gets right:** each hand-off loses a fixed fraction, so the loss compounds. **Where it breaks down:** real whispers get *garbled*; here the message stays correct, it just gets too faint to act on.

#### First, why one neuron goes quiet
You already felt this with your own finger a minute ago. When you dragged `z` far out into sigmoid's tail, the slope reading fell toward `0` — the output was pinned near the top and stopped reacting. That is [[saturation||When an activation's input is far from zero, its output is pinned near a limit and its slope is ≈ 0. A saturated neuron barely responds to changes in its input.]], and it has an easy cure: keep `z` small so units stay in the steep middle, or use a bend that never flattens.

A saturated neuron has a slope of about `0`, and you already know what that means: no tilt, no nudge, no learning. One sleepy neuron. Annoying, but survivable.

%%% insight
Here's why it bites much harder than "one sleepy neuron." Nothing in your code errors. Nothing prints a warning. The loss just settles a bit higher than it should and sits there — for hours. This failure's whole trick is looking like a bad idea rather than a broken network.
%%%

#### Then, why depth makes it brutal
Back to the whisper line — and now put sigmoid's numbers on it. Its steepest slope is only about `0.25`, so at *best* each layer passes back a quarter of what it heard. **Guess before you look:** four layers back, is the whisper down to a half? A tenth? Hold a number in your head, then read the bars.

Each bar below is how loud the learning signal still is, that many layers back. Both rows are drawn to the same height scale, so you can compare them straight across — and no arithmetic is needed, the picture does it for you:

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="Two rows of bars drawn to the same height scale. Top row, sigmoid: the signal is a full bar at 1.0 zero layers back, then about 0.25, about 0.06, about 0.016 and about 0.004 — shrinking to almost nothing. Bottom row, ReLU: every bar is full height at 1.0, from zero layers back to four layers back."><g font-family="monospace" font-size="11" text-anchor="middle">
<text x="52" y="18" fill="#5E5191" font-weight="bold" text-anchor="start">sigmoid · ×0.25 per hand-off → the signal fades</text>
<text x="470" y="18" fill="#9A938A" font-size="9" text-anchor="end">same scale in both rows: 1.0 = 60 px</text>
<line x1="52" y1="92" x2="440" y2="92" stroke="#E5DFD6" stroke-width="1.5"/>
<rect x="60"  y="32" width="46" height="60" fill="#7C6DAA"/><text x="83"  y="106" fill="#6B645E">1.0</text><text x="83"  y="120" fill="#9A938A" font-size="9">0 back</text>
<rect x="140" y="77" width="46" height="15" fill="#7C6DAA"/><text x="163" y="106" fill="#6B645E">≈0.25</text><text x="163" y="120" fill="#9A938A" font-size="9">1 back</text>
<rect x="220" y="88" width="46" height="4"  fill="#7C6DAA"/><text x="243" y="106" fill="#6B645E">≈0.06</text><text x="243" y="120" fill="#9A938A" font-size="9">2 back</text>
<rect x="300" y="91" width="46" height="1"  fill="#9A93C0"/><text x="323" y="106" fill="#6B645E">≈0.016</text><text x="323" y="120" fill="#9A938A" font-size="9">3 back</text>
<rect x="380" y="91" width="46" height="1"  fill="#D9D3EC"/><text x="403" y="106" fill="#C93B3B">≈0.004</text><text x="403" y="120" fill="#C93B3B" font-size="9">4 back</text>
<text x="52" y="152" fill="#2D8B55" font-weight="bold" text-anchor="start">ReLU · ×1 per hand-off → the signal survives</text>
<line x1="52" y1="224" x2="440" y2="224" stroke="#E5DFD6" stroke-width="1.5"/>
<rect x="60"  y="164" width="46" height="60" fill="#2D8B55"/><text x="83"  y="238" fill="#6B645E">1.0</text>
<rect x="140" y="164" width="46" height="60" fill="#2D8B55"/><text x="163" y="238" fill="#6B645E">1.0</text>
<rect x="220" y="164" width="46" height="60" fill="#2D8B55"/><text x="243" y="238" fill="#6B645E">1.0</text>
<rect x="300" y="164" width="46" height="60" fill="#2D8B55"/><text x="323" y="238" fill="#6B645E">1.0</text>
<rect x="380" y="164" width="46" height="60" fill="#2D8B55"/><text x="403" y="238" fill="#6B645E">1.0</text>
<text x="470" y="238" fill="#9A938A" font-size="9" text-anchor="end">same 0…4 layers back</text>
</g></svg>
%%%

If the top row surprised you, good — almost everyone guesses "about a half." Three short beats and it's yours:

%%% steps
step: **The fade.** Every hand-off multiplies the whisper by sigmoid's best slope, `0.25`
why: four hand-offs later it is a rounding error, `≈ 0.004` — that top row of bars
step: **The name.** Early layers hear only a mumble, so they barely change: the [[vanishing gradient||A learning signal (gradient) that shrinks toward zero as it is multiplied by each layer's small slope on the way back through a deep network — so the earliest layers stop learning. Saturating activations like sigmoid/tanh cause it; ReLU largely avoids it.]]
why: therefore a deep sigmoid net trains its last layers and quietly abandons its first ones
step: **The rescue.** Swap in ReLU, whose positive-side slope is exactly `1`
why: multiplying by 1 changes nothing, so the whisper arrives at full volume — that bottom row of bars
%%%

Struggling to hold it all? That's the normal reaction, not a bad sign — the ladder here is one idea repeated, not four new ones. If you'd like a nudge, open the hints below one at a time.

%%% hint
t1: Don't try to hold four layers in your head at once. Look at ONE hand-off: a person hears "1.0" and passes on "0.25." What single number did they multiply by?
t2: They multiplied by 0.25 — sigmoid's steepest slope. Doing that four times shrinks 1.0 to about 0.004, which is why the fourth bar is invisible.
t3: Each layer multiplies the learning signal by a small slope (≈0.25 for sigmoid), so many layers shrink it toward zero and the earliest layers stop learning. That fade is the vanishing gradient, and ReLU's slope of 1 is what stops it.
%%%

That single swap — saturating sigmoids out, ReLU in — is what let people finally train genuinely deep networks. **You can now explain in one sentence why sigmoid stalled deep nets and ReLU rescued them.** That is a real interview answer, and it's yours.

!!! c-ok 🧰
<b>Three more cures, so you know the names.</b> Keep <code>z</code> small so units stay in the steep middle. Prefer <code>tanh</code> over <code>sigmoid</code> when you need an S-curve (slope up to 1.0, not 0.25). And two standard tools you'll meet properly later: <b>He</b> / <b>Xavier initialization</b> (recipes for sensible starting weights) and <b>normalization layers</b> (they re-centre each layer's numbers as training runs).
!!!

**So far:** flat tails + depth = a whisper that dies. ReLU's tilt of 1 is the fix. (The backward-pass math that actually multiplies these slopes is Day 5's job.)

@@@ concept id=c8 tag="Which bend?" title="Inside the factory vs. the shipping door" gotit="Got the two decisions"
Time to cash in what you've learned — this is the fun part, and it's a break from hunting failures. You've met the whole family; now you get to *choose*.

Picture a **factory**. Inside, along the conveyor belt, every station uses the same cheap stamping tool over and over, because cheap and reliable is exactly what you want a hundred times a second. Then, at the **shipping door**, you pick a box that fits the thing you're actually sending: a bottle needs a bottle-shaped box, a book needs a flat one.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A factory picture in two halves. On the left, a conveyor belt with three identical ReLU stamping stations above it, labelled as the hidden layers where you pick one cheap bend and repeat it. On the right, a shipping door holding three different shaped boxes labelled yes-no, which-one, and number, labelled as the output layer where you match the box to the answer's shape."><g font-family="monospace" font-size="10">
<text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Inside the factory: the same cheap tool. At the shipping door: the right box.</text>
<rect x="28" y="96" width="270" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/>
<circle cx="48" cy="109" r="7" fill="#FDF9F3" stroke="#7C6DAA"/><circle cx="163" cy="109" r="7" fill="#FDF9F3" stroke="#7C6DAA"/><circle cx="278" cy="109" r="7" fill="#FDF9F3" stroke="#7C6DAA"/>
<rect x="60" y="62" width="52" height="26" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="86" y="79" text-anchor="middle" fill="#1a5c38">ReLU</text>
<rect x="137" y="62" width="52" height="26" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="163" y="79" text-anchor="middle" fill="#1a5c38">ReLU</text>
<rect x="214" y="62" width="52" height="26" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="240" y="79" text-anchor="middle" fill="#1a5c38">ReLU</text>
<text x="163" y="142" text-anchor="middle" fill="#5E5191">the hidden layers — one cheap bend, repeated</text>
<text x="163" y="158" text-anchor="middle" fill="#9A938A">decision 1: pick ONE default and move on</text>
<text x="313" y="114" fill="#B8AEA2" font-size="16">→</text>
<rect x="340" y="50" width="150" height="78" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/>
<text x="415" y="66" text-anchor="middle" fill="#8A6D3B">🚪 shipping door</text>
<rect x="348" y="74" width="44" height="20" rx="3" fill="#FFFDF6" stroke="#9A7A10"/><text x="370" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">yes/no</text>
<rect x="396" y="74" width="44" height="20" rx="3" fill="#FFFDF6" stroke="#9A7A10"/><text x="418" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">which?</text>
<rect x="444" y="74" width="42" height="20" rx="3" fill="#FFFDF6" stroke="#9A7A10"/><text x="465" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">number</text>
<text x="415" y="110" text-anchor="middle" fill="#8A6D3B" font-size="9">a different box for a different job</text>
<text x="415" y="142" text-anchor="middle" fill="#8A6D3B">the output layer</text>
<text x="415" y="158" text-anchor="middle" fill="#9A938A">decision 2: match the answer's shape</text>
</g></svg>
%%%

That's the one thing to carry out of today about *choosing*: a network makes **two separate activation decisions**, and people mix them up constantly. Inside the [[hidden layers||Every layer between the input and the final output layer. Their job is to build useful internal features, not to produce the answer.]] you want a cheap, reliable bend. At the [[output layer||The last layer, whose numbers are the model's actual answer. Its activation has to match the SHAPE of the answer you want.]] you want the answer's shape.

**What the factory gets right:** one tool inside, a fitted box at the door. **Where it breaks down:** a factory's stations are physically different machines, while every hidden layer here runs the exact same little bend — the sameness is the point.

You've met all three doors already without noticing. Your phone's face unlock ends in one **yes/no** — is this you? A feed that ranks videos ends in a **which-one-of-many**. A model that guesses a house price or tomorrow's temperature ends in **a plain number**. Same conveyor inside; three different doors.

#### Decision 1 is easy: take the default
**Use `ReLU` in the hidden layers unless you have a specific reason not to.** That's it. Reach for `GELU` in a transformer, `Leaky ReLU` if units go silent (that's the very next unit), `tanh` if you truly need a bounded, balanced signal. Otherwise: ReLU, move on, spend your thinking somewhere it pays.

#### Decision 2 is the one people get wrong
Now the interesting mistake — and it's easy to make, because it feels like being consistent. Below are **two different jobs**, and each door fails a different one. Job A: say a house price of `350,000`. Job B: say a temperature change of `−20`. **Predict which door survives each job before you read the picture.**

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Two different jobs tested on three output doors. Job A is to say the house price 350,000: with sigmoid at the output the answer is squashed to about 1.0, so it can never say 350,000, while with no activation at all the number passes straight through unchanged and arrives intact. Job B is to say a negative answer, minus 20 degrees: ReLU clips it to 0, because ReLU passes a large positive like 350,000 unchanged but can never output a negative number. Each door fails a different job."><g font-family="monospace" font-size="11">
<text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two jobs. Which door can even SAY the answer?</text>
<text x="26" y="38" fill="#8A6D3B" font-size="10" text-anchor="start">JOB A · say the house price: 350,000</text>
<text x="122" y="63" text-anchor="end" fill="#5E5191">sigmoid</text>
<rect x="130" y="46" width="96" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="178" y="63" text-anchor="middle" fill="#6B645E" font-size="10">350000 in</text>
<text x="234" y="63" fill="#B8AEA2">→</text>
<rect x="252" y="46" width="96" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="300" y="63" text-anchor="middle" fill="#C93B3B" font-size="10">≈ 1.0 out</text>
<text x="358" y="61" fill="#C93B3B" font-size="9" text-anchor="start">❌ squashed — can't even say 5</text>
<text x="122" y="99" text-anchor="end" fill="#9A5A12">no bend (linear)</text>
<rect x="130" y="82" width="96" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="178" y="99" text-anchor="middle" fill="#6B645E" font-size="10">350000 in</text>
<text x="234" y="99" fill="#B8AEA2">→</text>
<rect x="252" y="82" width="96" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="300" y="99" text-anchor="middle" fill="#276b45" font-size="10">350000 out</text>
<text x="358" y="97" fill="#2D8B55" font-size="9" text-anchor="start">✅ arrives intact</text>
<line x1="26" y1="120" x2="494" y2="120" stroke="#E5DFD6" stroke-width="1"/>
<text x="26" y="140" fill="#8A6D3B" font-size="10" text-anchor="start">JOB B · say a temperature change: −20</text>
<text x="122" y="165" text-anchor="end" fill="#2D8B55">ReLU</text>
<rect x="130" y="148" width="96" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="178" y="165" text-anchor="middle" fill="#6B645E" font-size="10">−20 in</text>
<text x="234" y="165" fill="#B8AEA2">→</text>
<rect x="252" y="148" width="96" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="300" y="165" text-anchor="middle" fill="#C93B3B" font-size="10">0 out</text>
<text x="358" y="163" fill="#C93B3B" font-size="9" text-anchor="start">❌ clipped — no negatives</text>
<text x="260" y="196" text-anchor="middle" fill="#6B645E" font-size="10">ReLU would pass 350,000 through fine — it just can never say −20</text>
<text x="260" y="212" text-anchor="middle" fill="#6B645E" font-size="10">so match the door to the SHAPE of the answer, job by job</text>
</g></svg>
%%%

%%% steps
step: **The mistake.** Habit says "activations go at the end of layers," so a `sigmoid` or `ReLU` gets stuck on the answer too
why: it *looks* tidy and consistent, which is exactly why this one slips through code review
step: **The damage.** `sigmoid` can never output `5`, and `tanh` can never output `100` — every large answer is squashed to the ceiling
why: therefore your model is not wrong by a little; it is physically unable to say the right number
step: **The fix, by job.** No bend at all (plain linear) when the answer is any number; a `sigmoid` when the answer is one yes/no
why: match the door to the shape of the answer, and the model can at least *express* the truth
%%%

%%% insight
This is the cleanest capability limit in the whole day, and it's worth saying out loud: **a bounded bend cannot express a big answer.** Not "does it badly" — *cannot*. The ceiling isn't a training problem you can fix with more data or more layers; it's built into the shape of the curve.
%%%

Which box goes with which job — the which-one-of-many door (called **softmax**), and the loss function that pairs with each — is Day 4's whole story. Today you only need the split: **inside, one cheap bend; at the door, match the answer.**

**So far:** two decisions, not one. Hidden layers: take ReLU and move on. Output layer: think, every time. **You can now answer "which activation should I use?" like someone who has shipped a model.**

@@@ concept id=c9 tag="The jammed valve" title="When a ReLU quietly jams shut" gotit="Got the jammed valve"
ReLU saved us from the fading whisper — and then brought a puzzle of its own, one that's genuinely fun to solve because it is so *silent*: nothing crashes, no error prints, the loss just stops dropping.

Picture that one-way valve again, but this time it has **jammed shut**. Water piles up behind it; nothing gets through, ever.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve jammed fully shut. Water piles up behind the closed gate and nothing passes through. The pipe on the far side is empty. The valve is frozen and cannot reopen on its own."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A dead ReLU = the valve jammed SHUT</text><rect x="40" y="56" width="200" height="46" rx="4" fill="#DCE9F5" stroke="#7C6DAA" stroke-width="1.5"/><text x="120" y="84" text-anchor="middle" fill="#5E5191">💧 water piles up 💧</text><line x1="250" y1="46" x2="250" y2="112" stroke="#C93B3B" stroke-width="5"/><rect x="242" y="56" width="16" height="46" fill="#C93B3B"/><text x="250" y="40" text-anchor="middle" fill="#C93B3B" font-size="14">🔒</text><rect x="262" y="56" width="216" height="46" rx="4" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5" stroke-dasharray="4,3"/><text x="370" y="84" text-anchor="middle" fill="#9A938A">nothing gets through</text><text x="120" y="126" text-anchor="middle" fill="#6B645E" font-size="10">input stays negative</text><text x="370" y="126" text-anchor="middle" fill="#C93B3B" font-size="10">output frozen at 0</text><text x="260" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">no hand to pry it open — slope 0, so training can't grip it</text></g></svg>
%%%

That is a [[dead ReLU||A ReLU whose input has become negative for every example, so it outputs a flat 0 forever. Its slope there is 0, so training gives it no signal, and it never recovers on its own.]]: a unit whose input has drifted negative for *every* example, so it outputs a flat `0` and never opens again.

**What the jammed valve gets right:** nothing passes; the output is frozen. **Where it breaks down:** a real valve you can pry open by hand — a dead ReLU has no handle, because its slope out there is exactly `0`. That's what makes this one permanent.

%%% insight
Read that again slowly: the unit is not *wrong*, it's *gone*. Let 100 units die in a layer of 300 and you are paying for 300 neurons while only 200 of them learn anything — and the loss curve will never tell you.
%%%

#### What jams the valve, and how you catch it
Two crisp beats, then you're armed.

%%% steps
step: **The shove.** One oversized training step pushes a unit's input permanently negative
why: the learning rate is how big a step training takes — a real knob you'll meet on Day 5 when we walk downhill; too big, and a unit can be flung past no return
step: **The freeze.** Out there the slope is exactly 0, so no future signal can move it back
why: therefore the unit is stuck at 0 forever — dead, not merely quiet
step: **The tell.** Log the *fraction of ReLUs sitting at exactly 0*, not just the loss
why: when that fraction creeps up while the loss plateaus, you've found your culprit in one glance
%%%

Now play detective with the numbers themselves. Two things get printed each epoch — the loss, and the fraction of hidden ReLUs sitting at exactly `0`:

%%% demo id=deadtell label="predict, then run"
predict: the loss stops improving after epoch 40. Which of the two numbers below tells you *why* — and what do you expect the fraction of zeros to do?
code: h = relu(X @ W1 + b1)            # one batch through the hidden layer
code: print(f"epoch {epoch} · loss {loss:.2f} · zeros {(h == 0).mean():.2f}")
out: epoch 10 · loss 0.42 · zeros 0.51
out: epoch 40 · loss 0.31 · zeros 0.58
out: epoch 70 · loss 0.31 · zeros 0.79
out: epoch 100 · loss 0.31 · zeros 0.80
take: <b>The loss says "stuck"; the zeros say "why."</b> About half the units at 0 is healthy ReLU sparsity. But watch the third column climb to `0.79`, then `0.80`, and *park* there while the loss flatlines at `0.31` — that is units dying, one shove at a time. One extra printed number turns a mystery into a diagnosis.
%%%

!!! c-warn ⚠️
<b>And the opposite failure, in one line.</b> ReLU has no ceiling, so starting weights that are too <em>large</em> make values grow layer after layer instead of dying — activations explode toward huge numbers. Same two cures: sensible starting weights (<b>He initialization</b>) and normalization layers.
!!!

**So far:** ReLU can die silently, and activation stats are how you spot it. **You can now diagnose a stuck network from its activations instead of guessing** — and the good news is that the cure is one line of code, waiting in the very next unit.

@@@ concept id=c10 tag="The rescue" title="Leaky ReLU — one small leak brings it back" gotit="Got the rescue"
Here's the delightful part: that permanent death has a fix you can picture in five seconds and type in one line.

Take the same one-way valve — and leave it with a **tiny permanent leak**. Positives still flow through as before. Negatives, instead of being fully shut off, seep past as a small trickle.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve with a tiny leak. Top: positive flow passes straight through as before. Bottom: on the blocked side a small trickle still seeps past the gate instead of being fully stopped, keeping the valve alive."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Leaky ReLU = a valve with a tiny built-in leak</text><rect x="40" y="38" width="440" height="32" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><polygon points="250,42 262,54 250,66" fill="#2D8B55"/><text x="95" y="58" fill="#276b45">positive →</text><text x="330" y="58" fill="#276b45" font-weight="bold">flows through 💧</text><rect x="40" y="92" width="440" height="36" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><line x1="250" y1="88" x2="250" y2="132" stroke="#C99A12" stroke-width="3"/><rect x="244" y="96" width="12" height="20" fill="#C99A12"/><text x="259" y="124" fill="#9A7A10" font-size="9">·····</text><text x="95" y="114" fill="#8A6D3B">negative ←</text><text x="320" y="114" fill="#8A6D3B" font-weight="bold">small trickle (slope α)</text><text x="260" y="150" text-anchor="middle" fill="#2D8B55" font-size="10">the leak = a tiny non-zero slope → the unit can't fully die</text></g></svg>
%%%

That's [[Leaky ReLU||Like ReLU, but negatives get a small slope (α·z, e.g. α=0.01) instead of a flat 0 — so the unit keeps a tiny slope on the negative side and can't fully die.]]. In plain words: negatives aren't zeroed, they're shrunk.

**What the leaky valve gets right:** the trickle is exactly the small slope that keeps the unit alive. **Where it breaks down:** a real leak is a fault you'd fix — here the leak is the *feature*, deliberately built in.

#### Bring the dead unit back, live
Take the jammed unit from the last section — its input is stuck out at `z = −3`. Watch what each bend does as that input drifts from `−5` to `−3`, because *movement* is exactly what a slope is. **Predict Leaky ReLU's numbers before you reveal them.**

%%% demo id=leaky label="swap the bend — predict, then run"
predict: ReLU gives 0 at z = −5, −4 and −3 — that row never moves, and the stillness IS the death. What does Leaky ReLU (α = 0.01) give for the same three? And what do you think it does to the positives, 2 and 5?
code: relu(np.array([-5., -4., -3.]))
code: leaky_relu(np.array([-5., -4., -3.]), alpha=0.01)
code: leaky_relu(np.array([-5., -2., 0., 2., 5.]), alpha=0.01)
out: array([0., 0., 0.])
out: array([-0.05, -0.04, -0.03])
out: array([-0.05, -0.02,  0.  ,  2.  ,  5.  ])
take: <b>The whole rescue, in three rows.</b> ReLU's row never budges as `z` climbs — no movement means no slope, so nothing can pull the unit back. Leaky ReLU's row rises by `0.01` at every step: tiny, but not zero, so a real nudge finally reaches the unit and it can walk back to life. The third row shows what that costs you on the positive side: nothing at all — `2` and `5` come out exactly as ReLU would leave them.
%%%

#### Why one small leak undoes a permanent death
Walk it through with the valve in mind. Here is the same fix drawn as a shape — the dashed red line is where plain ReLU sat flat, and the green line is the tilt we gave it instead:

%%% svg
<svg viewBox="0 0 520 155" role="img" aria-label="Leaky ReLU drawn against plain ReLU. For negative inputs, plain ReLU is a flat dashed line at zero while leaky ReLU slopes gently downward, so it keeps a small non-zero slope there. For positive inputs both rise as the same straight line. A note says the leak is drawn larger than real so it is visible."><g font-family="monospace"><line x1="60" y1="92" x2="330" y2="92" stroke="#E5DFD6" stroke-width="1.5"/><line x1="195" y1="24" x2="195" y2="116" stroke="#E5DFD6" stroke-width="1"/><path d="M70 92 L195 92" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><path d="M70 104 L195 92 L305 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="120" y="120" font-size="11" fill="#2D8B55" text-anchor="middle">tiny leak (slope α)</text><text x="118" y="84" font-size="10" fill="#C93B3B" text-anchor="middle">ReLU: flat 0 (can die)</text><text x="258" y="34" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text x="415" y="70" font-size="12" fill="#1a5c38">leaky(z)=max(αz, z)</text><text x="195" y="140" font-size="9" fill="#9A938A" text-anchor="middle">the leak is drawn much bigger than the real α = 0.01, so you can see it</text></g></svg>
%%%

%%% steps
step: **The old dead end.** Plain ReLU's negative side is flat, slope exactly `0`
why: a unit pushed out there gets no signal, so it can never come back — the jammed valve
step: **The leak.** Give that side a small tilt instead — `α·z`, with `α` (the Greek letter "alpha") around `0.01`
why: small enough that ReLU's behaviour barely changes, big enough that the slope is no longer zero
step: **The way back.** A non-zero slope means a real nudge arrives
why: therefore the unit can climb back toward positive and start working again — no death, no rescue mission
%%%

%%% insight
Notice the *shape* of this fix, because you'll see it again and again in machine learning: the problem was a hard zero, and the cure was not a clever algorithm — just refusing to let a number be exactly zero. Most of the modern activations after ReLU are variations on that one move.
%%%

**Victory lap.** Stop and look at what you're now holding: two silent failures, both understood, both with a named cure you could type today. That failure-plus-cure pairing *is* how engineers think about design — and you built it yourself, in about twenty minutes.

!!! c-ok 🧰
<b>Three relatives you'll hear named.</b>
<br><b>· GELU</b> and <b>ELU</b> curve smoothly near zero instead of using a straight leak, so they keep a non-zero slope around the origin — where dying happens. (Far out on the negative side they do flatten too; <b>Leaky ReLU</b> is the one with a slope everywhere.) GELU is the bend inside modern transformer models.
<br><b>· He (Kaiming) initialization</b> heads off dead units <em>before</em> training starts: it sizes the random starting weights so activations begin in a healthy range. The default for a ReLU network, set with one line.
<br><b>· Xavier / Glorot initialization</b> is the sibling recipe for <code>tanh</code>/<code>sigmoid</code> layers.
!!!

!!! c-info ⚖️
<b>Trade-off: ReLU's speed and sparsity vs. its dead-unit risk.</b> ReLU buys a dirt-cheap bend, healthy gradients in deep stacks, and lots of exact zeros (<b>sparsity</b>). What you give up is safety at zero. Softer bends that are never exactly zero near the origin (Leaky ReLU, ELU, GELU) trade a little extra math and less sparsity for units that don't die.
!!!

@@@ concept id=c11 tag="Two ropes" title="One rope can't do XOR — two can" gotit="Got the limit"
Let's finish with the puzzle that started this entire field — and see the bend from the other side.

Picture a **room with four chairs**, one in each corner. Two red chairs sit on one diagonal; two blue chairs sit on the other. I hand you a single straight **rope**, pulled tight across the floor, and ask you to put both reds on one side and both blues on the other.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A room floor with four chairs at the corners. Two red chairs sit on one diagonal, two blue chairs on the other. A single straight rope drawn across the floor separates only the bottom-left blue chair, so the top-right blue chair ends up on the same side as both reds — three of the four chairs are right and one is wrong, and no angle does better."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One straight rope can't split two diagonal pairs</text><rect x="150" y="32" width="220" height="120" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><rect x="172" y="52" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="185" y="70" text-anchor="middle" fill="#C93B3B">🔴</text><rect x="322" y="106" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="335" y="124" text-anchor="middle" fill="#C93B3B">🔴</text><rect x="322" y="52" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="335" y="70" text-anchor="middle" fill="#5E5191">🔵</text><rect x="172" y="106" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="185" y="124" text-anchor="middle" fill="#5E5191">🔵</text><line x1="140" y1="36" x2="276" y2="162" stroke="#9A5A12" stroke-width="3"/><text x="284" y="152" fill="#9A5A12" font-size="10" text-anchor="start">straight rope</text><circle cx="335" cy="65" r="20" fill="none" stroke="#C93B3B" stroke-width="1.5" stroke-dasharray="4,3"/><text x="376" y="62" fill="#C93B3B" font-size="10" text-anchor="start">this 🔵 ends up</text><text x="376" y="76" fill="#C93B3B" font-size="10" text-anchor="start">with the reds</text><text x="72" y="96" text-anchor="middle" fill="#6B645E" font-size="10">reds on one</text><text x="72" y="110" text-anchor="middle" fill="#6B645E" font-size="10">diagonal,</text><text x="72" y="124" text-anchor="middle" fill="#6B645E" font-size="10">blues the other</text><text x="260" y="186" text-anchor="middle" fill="#C93B3B" font-size="10">any angle → best you get is 3 of 4 (you need a bend + a 2nd rope)</text></g></svg>
%%%

Go on, try it. Any angle you like — then flip on the hidden layer and watch the second rope appear.

%%% viz src=../../viz/xor-limit.html title="why one neuron can't do XOR" caption="interactive — slide the single line (stuck at 3/4), then switch on the hidden layer"
%%%

**What the room of chairs gets right:** the diagonal layout is the entire trap. **Where it breaks down:** with a real rope you're stuck for good — a network can lay down a *second* rope, which changes everything.

#### From "impossible" to "easy" in three moves

%%% steps
step: **The failure you just felt.** Slide the rope anywhere: three chairs land right, the fourth is always wrong
why: one straight cut can't separate two diagonal pairs — the best score is 3 of 4, forever
step: **The name for it.** That room is the [[XOR||XOR: output 1 when exactly one of two inputs is on, else 0. Its two classes sit on opposite diagonals, so no single straight line separates them.]] problem — "on when exactly one input is on"
why: a single neuron draws exactly one straight line (that was Day 1), so XOR is not [[linearly separable||A dataset is linearly separable if one straight line (or flat plane) can put every class on its own side. XOR is the classic example that is NOT.]] — and no activation changes that, because one neuron is still one boundary
step: **The rescue: two ropes, laid side by side.** Switch on the hidden layer in the demo and *two* cuts appear at once
why: therefore one cut slices off the (0,0) corner and the other slices off the (1,1) corner, leaving both reds in the stripe between them — the two cuts are drawn side by side by two hidden units, and the bend is what stops the output layer's combination of them from folding back into one straight line
%%%

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="Left panel: one straight rope across the four chairs cuts off only the top-left red chair, so the bottom-right red chair ends up on the wrong side — three of four correct. Right panel: two roughly parallel cuts, each drawn by its own hidden unit from the same input, leave a middle stripe that contains both red points, while each blue corner is cut off outside the stripe — four of four correct."><g font-family="monospace" font-size="10"><text x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">one rope → 3 of 4</text><rect x="50" y="26" width="160" height="100" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><circle cx="80" cy="52" r="7" fill="#C93B3B"/><circle cx="180" cy="100" r="7" fill="#C93B3B"/><circle cx="180" cy="52" r="7" fill="#5E5191"/><circle cx="80" cy="100" r="7" fill="#5E5191"/><line x1="46" y1="89" x2="150" y2="24" stroke="#9A5A12" stroke-width="2.5"/><circle cx="180" cy="100" r="13" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="3,3"/><text x="130" y="142" text-anchor="middle" fill="#C93B3B">this 🔴 is on the wrong side</text><text x="255" y="80" fill="#B8AEA2" font-size="18">→</text><text x="395" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">two cuts, side by side → 4 of 4</text><rect x="315" y="26" width="160" height="100" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><polygon points="315,62 315,26 341,26 475,90 475,126 449,126" fill="#EAF5EE" opacity="0.85"/><path d="M315 62 L449 126" fill="none" stroke="#2D8B55" stroke-width="2.5"/><path d="M341 26 L475 90" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="322" y="80" fill="#276b45" font-size="8" text-anchor="start">unit A</text><text x="430" y="44" fill="#276b45" font-size="8" text-anchor="start">unit B</text><circle cx="345" cy="52" r="7" fill="#C93B3B"/><circle cx="445" cy="100" r="7" fill="#C93B3B"/><circle cx="445" cy="52" r="7" fill="#5E5191"/><circle cx="345" cy="100" r="7" fill="#5E5191"/><text x="395" y="142" text-anchor="middle" fill="#2D8B55">both 🔴 inside the stripe, each 🔵 cut off outside</text><text x="395" y="156" text-anchor="middle" fill="#9A938A" font-size="9">two hidden units cut in parallel; the output layer combines them</text></g></svg>
%%%

%%% insight
Look at that stripe: it isn't a smooth curve — it is built from **two straight cuts**, each drawn by its own hidden unit from the same input, and then combined by the output layer. The bend is what stops that combination from folding back into one straight line (remember the rulers). That is what a ReLU network always builds: flat pieces joined at kinks. Stack enough of them and you can trace any shape you like, the way a hundred short straight fence panels can follow the edge of a round pond. That is also its honest limit — a ReLU net *approximates* a smooth curve with many small kinks; it never matches one exactly.
%%%

**So far:** the bend doesn't give one neuron new powers — it lets *many* simple pieces combine into something no straight line could reach. (And no bend rescues a network that's too small or fed useless inputs; the activation changes the *shape* of the response, never what the layer can see.)

#### Why the big labs care
`ReLU` is the one you'll meet most often — dirt cheap, and it keeps the backward whisper loud. That's why the **ReLU family** sits inside the model that reads a photo, the one that ranks your feed, and the one that answers your questions.

"Family" is the important word there. Different labs pick different cousins for exactly the reason you'd pick Leaky ReLU — never let the bend flatten where it matters. Modern transformer models reach for **GELU** (a smoothed ReLU), and some newer ones use a gated cousin called **SwiGLU**. Sigmoid and its cousin [[softmax||Turns a vector of scores into probabilities that sum to 1. Used at the output of a classifier.]] mostly live at the shipping door, where you want probabilities — that's Day 4's story.

The bend really is the quiet reason "deep" learning works at all, and now you know why.

@@@ concept id=c12 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you did the whole thing. Here's a way to hold it all at once: today was one **road trip**, and these were the five signposts you drove past, in order. Each one only makes sense because of the one before it.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A straight road from start to finish with five numbered signposts standing along it in order: one the trap, two the bend, three the family of bends, four the quiet failures, five the limits and the door."><g font-family="monospace" font-size="9">
<text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Today = one road trip, five signposts in order</text>
<rect x="20" y="88" width="480" height="16" rx="3" fill="#E5DFD6"/>
<line x1="26" y1="96" x2="494" y2="96" stroke="#FDF9F3" stroke-width="2" stroke-dasharray="10,8"/>
<circle cx="24" cy="96" r="6" fill="#2D8B55"/><text x="24" y="118" text-anchor="middle" fill="#276b45">start</text>
<line x1="86" y1="68" x2="86" y2="88" stroke="#9A5A12" stroke-width="2"/><rect x="60" y="48" width="52" height="20" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="86" y="62" text-anchor="middle" fill="#8A6D3B">1 trap</text>
<line x1="172" y1="104" x2="172" y2="124" stroke="#9A5A12" stroke-width="2"/><rect x="146" y="124" width="52" height="20" rx="3" fill="#EFEAF7" stroke="#7C6DAA"/><text x="172" y="138" text-anchor="middle" fill="#5E5191">2 bend</text>
<line x1="258" y1="68" x2="258" y2="88" stroke="#9A5A12" stroke-width="2"/><rect x="228" y="48" width="60" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="258" y="62" text-anchor="middle" fill="#276b45">3 family</text>
<line x1="344" y1="104" x2="344" y2="124" stroke="#9A5A12" stroke-width="2"/><rect x="312" y="124" width="64" height="20" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="344" y="138" text-anchor="middle" fill="#C93B3B">4 failures</text>
<line x1="430" y1="68" x2="430" y2="88" stroke="#9A5A12" stroke-width="2"/><rect x="400" y="48" width="60" height="20" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="430" y="62" text-anchor="middle" fill="#8A6D3B">5 limits</text>
<circle cx="496" cy="96" r="6" fill="#C93B3B"/><text x="488" y="118" text-anchor="middle" fill="#6B645E">finish 🏁</text>
</g></svg>
%%%

**Where the road-trip picture breaks down:** a real trip is a straight road you drive once — these signposts are a loop you'll circle back to for years, and today's whole point (the bend!) is the opposite of straight.

#### Now rebuild the road yourself, one "therefore" at a time
Don't re-read the day — *re-derive* it. Each link below is forced by the one above it, so if you can say the first line out loud, the rest follow.

%%% steps
step: **Link 1 — the trap.** Stack straight-line layers and they fold back into one straight line
why: therefore depth on its own buys you nothing — that was the pair of rulers
step: **Link 2 — so insert a bend.** One small non-linearity between layers and they can't fold flat
why: weights mix, the bend curves; therefore we now need to pick *which* bend
step: **Link 3 — the danger is flatness.** The hard on/off switch has slope 0 everywhere, and the smooth S-curves keep a tilt only in the middle
why: small slopes multiply layer after layer, so the backward whisper fades — the vanishing gradient
step: **Link 4 — ReLU keeps a tilt of exactly `1` on the positive side.** Cheap, sparse, and the signal survives depth
why: therefore hidden layers default to ReLU (or GELU) — but its flat `0` side can jam a unit shut, so we refuse the hard zero with a leak (Leaky ReLU) plus He initialization
step: **Link 5 — the door, and the ceiling.** The output layer must match the *shape* of the answer, and one neuron is still one straight cut
why: therefore a bend lets many simple pieces combine, but XOR still needs layers — which is exactly where tomorrow starts
%%%

Here are those same five links drawn as a staircase, each one resting on the one above it. This is the picture to rebuild in your head tomorrow morning:

%%% svg
<svg viewBox="0 0 520 235" role="img" aria-label="The day rebuilt as one chain of consequences, drawn as a staircase where each step rests on the one above it. Link one: straight plus straight is still straight. Therefore link two: we insert a bend. Therefore link three: flatness is the danger, because a slope near zero gives no learning signal. Therefore link four: ReLU keeps a slope of exactly one on the positive side, though its flat zero side can jam, so we add a small leak. Therefore link five: match the output door to the shape of the answer, and remember that XOR still needs layers."><g font-family="monospace" font-size="10">
<text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The day as one chain — each link forced by the one above</text>
<rect x="24" y="28" width="320" height="26" rx="5" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="34" y="45" fill="#C93B3B" text-anchor="start">1 · straight + straight = still straight</text>
<line x1="184" y1="54" x2="184" y2="68" stroke="#B8AEA2" stroke-width="1.5"/><text x="192" y="66" fill="#9A938A" font-size="8" text-anchor="start">therefore</text>
<rect x="44" y="68" width="320" height="26" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="54" y="85" fill="#5E5191" text-anchor="start">2 · so we insert a bend</text>
<line x1="204" y1="94" x2="204" y2="108" stroke="#B8AEA2" stroke-width="1.5"/><text x="212" y="106" fill="#9A938A" font-size="8" text-anchor="start">therefore</text>
<rect x="64" y="108" width="320" height="26" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="74" y="125" fill="#8A6D3B" text-anchor="start">3 · flat slope = no learning signal</text>
<line x1="224" y1="134" x2="224" y2="148" stroke="#B8AEA2" stroke-width="1.5"/><text x="232" y="146" fill="#9A938A" font-size="8" text-anchor="start">therefore</text>
<rect x="84" y="148" width="320" height="26" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="94" y="165" fill="#276b45" text-anchor="start">4 · ReLU: slope 1 — but add a leak</text>
<line x1="244" y1="174" x2="244" y2="188" stroke="#B8AEA2" stroke-width="1.5"/><text x="252" y="186" fill="#9A938A" font-size="8" text-anchor="start">therefore</text>
<rect x="104" y="188" width="320" height="26" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="114" y="205" fill="#5E5191" text-anchor="start">5 · match the door; XOR needs layers</text>
</g></svg>
%%%

#### Cheat-sheet · the one picture to keep
The three working bends side by side, drawn on the same axes so you can read each range straight off:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Recap figure: ReLU, sigmoid and tanh on one set of labelled axes with gridlines at minus one, zero and plus one. ReLU is flat at zero for negatives then rises straight past plus one. Sigmoid rises from just above zero, passes exactly through one half at z equals zero, and heads toward plus one. Tanh passes exactly through zero at z equals zero and is the same distance below the axis on the left as above it on the right, running from minus one to plus one."><g font-family="monospace" font-size="12">
<rect x="292" y="13" width="10" height="10" fill="#2D8B55"/><text x="307" y="22" fill="#2D8B55" font-size="11" text-anchor="start">ReLU</text>
<rect x="352" y="13" width="10" height="10" fill="#7C6DAA"/><text x="367" y="22" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid</text>
<rect x="432" y="13" width="10" height="10" fill="#9A5A12"/><text x="447" y="22" fill="#9A5A12" font-size="11" text-anchor="start">tanh</text>
<line x1="60" y1="45" x2="470" y2="45" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<line x1="60" y1="95" x2="470" y2="95" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="60" y1="145" x2="470" y2="145" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="49" fill="#9A938A" font-size="10" text-anchor="end">+1</text>
<text x="56" y="99" fill="#9A938A" font-size="10" text-anchor="end">0</text>
<text x="56" y="149" fill="#9A938A" font-size="10" text-anchor="end">−1</text>
<line x1="265" y1="30" x2="265" y2="155" stroke="#E5DFD6" stroke-width="1"/>
<path d="M70 145 C 160 145, 235 99, 265 95 C 295 91, 370 49, 460 45" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<path d="M70 93 C 160 93, 235 73, 265 70 C 295 67, 370 55, 460 53" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<path d="M70 95 L265 95 L455 25" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<circle cx="265" cy="70" r="3.5" fill="#C93B3B"/><text x="256" y="64" fill="#C93B3B" font-size="10" text-anchor="end">sigmoid(0)=0.5</text>
<circle cx="265" cy="95" r="3.5" fill="#C93B3B"/><text x="276" y="112" fill="#C93B3B" font-size="10" text-anchor="start">tanh(0)=0</text>
<text x="130" y="88" fill="#9A938A" font-size="10" text-anchor="middle">negatives</text>
<text x="380" y="140" fill="#9A938A" font-size="10" text-anchor="middle">positives</text>
<text x="260" y="168" fill="#6B645E" font-size="10" text-anchor="middle">ReLU keeps climbing past 1 · sigmoid stays inside (0,1) · tanh stays inside (−1,1)</text>
</g></svg>
%%%

#### Cheat-sheet · the three bends at a glance
%%% table
:: :: ReLU :: sigmoid :: tanh
Shape :: one-way valve :: soft dimmer :: balanced see-saw
Range :: `[0, ∞)` :: `(0, 1)` :: `(−1, 1)`
Slope :: `1` for positives, `0` for negatives :: peaks ≈ `0.25` at z=0 :: peaks ≈ `1.0` at z=0
Watch out :: can go **dead** at 0; also **not zero-centered** :: **saturates** at both tails; **not zero-centered** :: **saturates** at both tails
Reach for it :: hidden layers (the default) :: an output probability :: a balanced, bounded signal
%%%

#### Cheat-sheet · the words you met today
%%% jargon
activation | the little bend added at the end of a neuron, after the weighted sum
non-linearity | not a straight line — the property a bend adds so depth matters
linear collapse | stacked straight-line layers folding back into one layer
step function | the original hard on/off bend; flat everywhere, so it can't learn
ReLU | a one-way valve: max(0, z) — passes positives, zeroes negatives
sigmoid | a soft dimmer that squashes any number into (0, 1)
tanh | a see-saw that squashes into (−1, 1) and rests at 0
zero-centered | outputs sit evenly above and below 0 (tanh yes; sigmoid and ReLU no)
slope | how steep the curve is at a point — a slope near 0 means learning stalls there
saturation | the curve has gone flat, so the slope ≈ 0 and the unit barely responds
vanishing gradient | the learning signal shrinks toward 0 through a deep stack of small slopes
dead ReLU | a ReLU stuck outputting a flat 0 forever, with no slope to recover
sparsity | many units output exactly 0 at once, so only the ones that matter speak up
Leaky ReLU | ReLU with a tiny negative slope (α·z) so the unit can't fully die — a slope everywhere
GELU | a smooth ReLU-like bend (used in modern transformers); never exactly zero near the origin, though it does flatten far out on the negative side
He initialization | the sensible starting-weight recipe for ReLU nets, so fewer units start dead
Xavier initialization | the sibling recipe for tanh/sigmoid layers
hidden layer | any layer between the input and the final output layer
output layer | the last layer — its bend must match the shape of the answer you want
XOR | "on when exactly one input is on" — the classic pattern one straight line can't split
%%%

!!! c-ok 🎤
<b>If someone asks you what an activation is, here's the whole day in four sentences:</b> "It's the per-layer bend. Without it the stack collapses — <code>(x W₁) W₂ = x (W₁ W₂)</code>, and the two biases fold into one new bias as well, so depth adds nothing at all. ReLU = max(0, z) is the cheap default and keeps the backward signal alive, because its positive-side slope of 1 doesn't shrink it the way sigmoid's ≈0.25 does. The failures are silent — saturated units and dead ReLUs — so I watch activation statistics, not just the loss, and I reach for Leaky ReLU or GELU with He initialization." Every piece of that is something you can already explain in your own words.
!!!

That's the whole day. Next you'll stack these neurons into layers and push data through them — the **forward pass**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does an activation function add to a network? | a:1 | More parameters | Non-linearity (a bend) | Faster matrix multiplies (matmuls) | A bias term | fb: The activation is the non-linear bend; without it, layers stay linear and depth is wasted.
q: With NO activation, ten stacked linear layers behave like: | a:1 | a much more powerful model | a single linear layer | a random function | a sigmoid | fb: (x W₁)W₂…W₁₀ = x (W₁…W₁₀) — one combined matrix, and the ten biases fold into one new bias too. Depth buys nothing without a bend.
q: Why does deep sigmoid suffer the "vanishing gradient" while deep ReLU largely doesn't? | a:2 | Sigmoid uses more memory | ReLU is written in C | Sigmoid's slope tops out near 0.25, so multiplying it through many layers shrinks the backward signal toward 0; ReLU's positive slope is 1, so it doesn't shrink | Sigmoid outputs negative numbers | fb: The backward signal is multiplied by each layer's slope. With ≈0.25 per layer it fades fast (0.25⁴ ≈ 0.004); ReLU's slope of 1 keeps it full-strength.
q: Your network trains without error, but the loss stops dropping early and stays high. You log the hidden ReLU outputs and find most are exactly 0 for every input in the batch. Most likely explanation — and a fix? | a:1 | Sigmoid at the output is always a bug | Many ReLU units are "dead": their input stays negative for all inputs, so max(0, z) is always 0 and they can no longer learn — switching those units to Leaky ReLU keeps a small negative slope so they can recover | The GPU ran out of memory, which always makes ReLU output zero | A high loss with zero activations means the network has already converged | fb: A ReLU that always sees a negative input outputs a flat 0 for every example — a dead unit. Many dead units mean much of the network does nothing, so the loss plateaus. You spot it from activation statistics (fraction of zeros), and Leaky ReLU (a small negative slope) is the standard cure.
%%%

@@@ produce id=produce tag="Produce" title="Watch two layers collapse — then break the collapse" gotit="Done"
Time to see today's big idea with your own eyes. You'll build two straight-line layers with no bend between them and watch them fold into a single matrix — then drop a `ReLU` in the middle and watch the fold stop working. **Predict first:** will `(x@W1)@W2` and `x@(W1@W2)` print the same numbers, or different ones? Then run it and **watch.** Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-02-activations/experiment.py`. Write `step`, `relu`, `leaky_relu`, `sigmoid`, and `tanh`, and print all five over the grid `np.linspace(-6, 6, 13)`. **Notice** that the step is a flat 0/1 jump, ReLU zeros negatives, `leaky_relu` lets a small trickle through, sigmoid stays in `(0,1)`, and tanh stays in `(−1,1)` and is 0 at 0. Also print `sigmoid'(0) ≈ 0.25` and note that ReLU's slope is `0` for every negative input while `leaky_relu`'s is a small `alpha` — the seed of both the dead-ReLU failure and its cure.

Then use the anchor pair `W1=[[1,2],[0,1]]`, `W2=[[1,0],[3,1]]` with `x_a=[[1.0, -3.0]]` and `x_b=[[-1.0, 3.0]]` (they add up to all zeros), and run the two paths side by side:

- **the collapse:** `(x_a@W1)@W2` equals `x_a@(W1@W2)` — print `W1@W2` (expect `[[7,2],[3,1]]`), proving two straight-line layers = one matrix.
- **the break:** a straight-line stack always *adds up* — `g(x_a) + g(x_b)` equals `g(x_a + x_b)` exactly. **Watch** the bent path fail that test: `f(x) = relu(x@W1)@W2` gives `f(x_a) + f(x_b) = [[4, 1]]` but `f(x_a + x_b) = [[0, 0]]`. Since every single matrix adds up, no single matrix can be doing this — the bend really did buy you something new.

Run with `python3 sessions/m02-the-neuron/day-02-activations/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 2 artifact.

Create sessions/m02-the-neuron/day-02-activations/experiment.py that, with a comment on each step:
1. Defines step(z)=(z>=0).astype(float), relu(z)=np.maximum(0,z), leaky_relu(z, alpha=0.01)=np.where(z>0, z, alpha*z), sigmoid(z)=1/(1+np.exp(-z)), and tanh(z)=np.tanh(z); prints all five as a table over grid = np.linspace(-6,6,13) (step is a 0/1 jump printed as floats, ReLU zeros negatives, leaky_relu lets a small trickle through, sigmoid in (0,1), tanh in (-1,1) and 0 at 0). Also prints sigmoid'(0)=0.25 and notes ReLU's slope is 0 for negative inputs while leaky_relu's is the small alpha.
2. Shows the collapse: with W1=[[1,2],[0,1]], W2=[[1,0],[3,1]] and x_a=[[1.0, -3.0]], asserts np.allclose((x_a@W1)@W2, x_a@(W1@W2)) and prints W1@W2 (expect [[7,2],[3,1]]) — two linear layers equal one.
3. Then proves the bend breaks that, the honest way, with x_b=[[-1.0, 3.0]] so x_a+x_b is all zeros. For the straight path g(x)=x@W1@W2, check g(x_a)+g(x_b) == g(x_a+x_b) (a linear map always adds up). For the bent path f(x)=relu(x@W1)@W2, print f(x_a), f(x_b), f(x_a)+f(x_b) and f(x_a+x_b), and assert they are NOT equal (expect [[4,1]] vs [[0,0]]). Add a comment saying that because every linear map adds up, failing this additivity test proves no single matrix can reproduce f.
Keep the keyword name alpha (leaky_relu(z, alpha=0.01)) so the calls in the lesson work unchanged, and keep relu as np.maximum(0, z) so integer inputs print as integers.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the collapse, then the cure)
- Your five bends print sensible values over the grid: the step is a flat 0/1 jump, ReLU zeros negatives, `leaky_relu` lets a small trickle through, sigmoid stays inside `(0, 1)`, tanh stays inside `(−1, 1)` and is 0 at 0.
- You print `sigmoid'(0) ≈ 0.25` and note ReLU's slope is `0` for negative inputs (the seed of dead units) while `leaky_relu` keeps a small slope `alpha` there — the seed of the cure.
- The first moment to **watch**: `(x_a@W1)@W2` prints the <em>same</em> numbers as `x_a@(W1@W2)` — two straight-line layers really did collapse into one.
- The second moment to **watch**: the straight path adds up perfectly (`g(x_a) + g(x_b) = g(x_a + x_b) = [[0, 0]]`), while the bent path does not (`[[4, 1]]` versus `[[0, 0]]`). Every single matrix adds up, so that mismatch is proof the bend cannot be folded flat. That break is the bend earning its keep.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-02-activations/log.md`: (1) why a 100-layer network with no activations is no better than one layer; (2) ReLU vs sigmoid in one line, mentioning the vanishing gradient; (3) what a "dead" or "saturated" unit is, how you'd detect it, and the one-line fix.
!!!

@@@ fin
