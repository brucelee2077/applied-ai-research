---
quest_id: wf3-d03-training-loop
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 6 — The Training Loop"
module_label: "Module 2 · Train · Day 6"
title: "The Training Loop"
subtitle: "Where the Network Actually Learns"
brand_sub: "Foundations · M2 Day 6"
spine: "free-throw"
nav_prev_href: "../day-05-gradients-backprop/lesson.html"
nav_prev_label: "Gradients & Backpropagation"
nav_next_href: "../day-07-optimizers/lesson.html"
nav_next_label: "Optimizers (SGD, Momentum, Adam)"
fin_title: "Module 2 · Day 6 complete! 🏆"
fin_body: "Nice work — you've assembled the <b>training loop</b>: forward pass → loss → backward pass → update, repeated over batches and epochs until the loss falls. That loop is where a network actually learns.<br>Next up: <b>Optimizers</b> — smarter ways to take each downhill step."
notebook_yardstick: 00-neural-networks/fundamentals/08_training_loop.ipynb
coverage_topics:
  - {topic: the four-phase loop, keywords: [four steps, forward pass, loss, backward pass, update, over and over]}
  - {topic: forward pass phase, keywords: [make a guess, weighted sum, prediction]}
  - {topic: loss phase, keywords: [how wrong, one number, loss]}
  - {topic: backward pass phase, keywords: [gradient, which way, backpropagation, blame]}
  - {topic: update phase, keywords: [the update, nudge the aim, shift each weight, new weight, small step in the gradient]}
  - {topic: minus sign goes downhill, keywords: [minus sign, opposite way, subtract]}
  - {topic: learning rate step size, keywords: [learning rate, step size, how big a step]}
  - {topic: epoch vs iteration, keywords: [epoch, one full pass, iteration, one step, one update]}
  - {topic: loss curve, keywords: [loss curve, plot the loss, goes down, watch it fall]}
  - {topic: convergence, keywords: [convergence, flattens out, stops improving, levels off]}
  - {topic: perceptron ancestor, keywords: [perceptron, rosenblatt, only on mistakes, fixed rule, crude ancestor]}
  - {topic: lr too high failure, keywords: [too high, overshoot, diverges, blows up, nan, oscillates]}
  - {topic: lr scheduling decay remedy, keywords: [learning-rate schedule, decay, shrink the learning rate, smaller steps later]}
  - {topic: lr too low failure, keywords: [too low, crawls, barely moves, too tiny]}
  - {topic: zero grad failure, keywords: [reset the gradients, gradients pile up, accumulate, forget to reset]}
  - {topic: zero grad remedy, keywords: [zero the gradients, reset the gradients each step, clear the gradients]}
  - {topic: shuffle failure, keywords: [same fixed order, fixed order, biased by that order, never shuffling]}
  - {topic: shuffle remedy, keywords: [shuffle the data, reshuffle each epoch, mix up the order]}
  - {topic: too few epochs underfit, keywords: [too few epochs, stopping too early, still high, underfit, train longer]}
  - {topic: overfitting from over-training, keywords: [memorizes, overfitting, train too long, validation loss rises]}
  - {topic: early stopping remedy, keywords: [early stopping, stop when validation stops improving, held-out]}
  - {topic: capability limit xor, keywords: [xor, non-linearly separable, more iterations never, plateaus above zero, single neuron cannot]}
  - {topic: local minimum plateau, keywords: [local minimum, plateau, non-convex, not the global best]}
---

@@@ hero
@lede Think about learning to sink a **free-throw** in basketball. You step up, you shoot, and the ball clanks off the left of the rim. So you aim a hair to the right and shoot again — a little closer. Clank, adjust, shoot. Clank, adjust, shoot. Nobody sinks it clean on the first try; you get good by *repeating one small fix, over and over*, until the ball drops through. That exact rhythm — *try, measure the miss, nudge, repeat* — is the whole secret of how a neural network learns. It has a name: the **training loop**, and it is the engine humming underneath every model you've heard of, from the tiny neuron you'll train today all the way up to ChatGPT. Over the last few days you built each separate piece — the neuron's guess, the loss that scores the miss, the gradient that says which way to nudge. Today the pieces click together into one repeating cycle. Press "go" and watch a single number — the loss — fall, shot after shot, until the network can sink it.
@goal Together we'll snap the pieces into one loop you could run in your sleep: **forward** (take the shot) → **loss** (measure the miss) → **backward** (which way was I off?) → **update** (nudge the aim), then loop again. You'll meet the **learning rate** — how big a nudge you make each shot — and the words **epoch** and **iteration** for counting your practice. You'll watch the **loss curve** slide downhill and flatten out when the network has (roughly) got it. Then we'll poke at the sneaky ways practice goes wrong — nudging too hard, too soft, forgetting to reset, drilling in a fixed rut, quitting too early, or grinding so long you just memorize — each a little puzzle with a neat one-line fix. Every formula shows up in plain words first; any heavy math sits in a skippable box.

@@@ concept id=c1 tag="The practice loop" title="The training loop — four steps, on repeat" gotit="Got the loop"
Let's stay on the basketball court, because you already know this loop by heart. Picture one **free-throw** practice cycle. **(1)** You shoot the ball — that's your best try right now. **(2)** You *look* at where it landed — a foot left of the rim — and measure the miss. **(3)** You figure out *which way* you were off — "I leaned left." **(4)** You *nudge* your aim a little to the right for next time. Then you do the whole thing again. Four little steps, over and over. That is the training loop, and every single piece of it is something you already built this module.

%%% svg
<svg viewBox="0 0 520 232" role="img" aria-label="A basketball free-throw practice loop drawn as four steps arranged in a circle: shoot the ball, see where it landed, figure out which way you were off, nudge your aim. Arrows connect them in a ring, showing the cycle repeats."><g font-family="monospace" font-size="10.5"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One free-throw practice cycle — repeated until it drops clean</text><circle cx="140" cy="70" r="40" fill="#E7F0F5" stroke="#2A7B9B"/><text x="140" y="64" text-anchor="middle" fill="#1F6280">🏀 shoot</text><text x="140" y="78" text-anchor="middle" fill="#1F6280" font-size="9">(take the shot)</text><circle cx="380" cy="70" r="40" fill="#FDECEC" stroke="#C93B3B"/><text x="380" y="64" text-anchor="middle" fill="#C93B3B">📏 measure</text><text x="380" y="78" text-anchor="middle" fill="#C93B3B" font-size="9">(how far off?)</text><circle cx="380" cy="180" r="40" fill="#EDE9F8" stroke="#7C6DAA"/><text x="380" y="174" text-anchor="middle" fill="#5E5191">🧭 which way?</text><text x="380" y="188" text-anchor="middle" fill="#5E5191" font-size="9">(I leaned left)</text><circle cx="140" cy="180" r="40" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="174" text-anchor="middle" fill="#1a5c38">🎯 nudge aim</text><text x="140" y="188" text-anchor="middle" fill="#1a5c38" font-size="9">(a little right)</text><g stroke="#B8AEA2" stroke-width="2" fill="#B8AEA2"><path d="M180 62 l160 0"/><polygon points="340,62 331,57 331,67"/><path d="M380 110 l0 30"/><polygon points="380,140 375,131 385,131"/><path d="M340 188 l-160 0"/><polygon points="180,188 189,183 189,193"/><path d="M140 140 l0 -30"/><polygon points="140,110 135,119 145,119"/></g><text x="260" y="130" text-anchor="middle" fill="#6B645E" font-size="9">round and round — each loop, a little closer</text></g></svg>
%%%

**What the free-throw picture gets right:** you improve by *repeating* one small correction — no single shot fixes everything, but hundreds of tiny fixes add up. That is exactly how a network trains. **Where it breaks down:** a basketball player adjusts one thing at a time (aim, then arc, then power), but a network nudges *every* one of its knobs at once, on every single loop. It is like fixing your aim, arc, power, stance, and breathing all in one motion, thousands of times.

#### The four steps, in network words
Each step on the court is a phase of the [[training loop||the four-step cycle — forward, loss, backward, update — repeated over and over until the network's mistakes get small]] you'll run in code:

- **Step 1 · Forward pass** — the network takes its shot: run an input through the neuron to get a **prediction** (the weighted-sum-then-activation you built on earlier days).
- **Step 2 · Loss** — measure the miss: the [[loss||one number saying how wrong the guess is; lower is better, 0 is perfect]] scores how far the guess landed from the true answer.
- **Step 3 · Backward pass** — which way was I off? [[backpropagation||the backward sweep from yesterday that hands every weight its gradient — which way to nudge it]] computes the **gradient** for every weight.
- **Step 4 · Update** — nudge the aim: shift each weight a small step in the direction that lowers the loss.

Then jump back to Step 1 and do it all again. Here is the loop laid out flat, so you can see one lap feed into the next.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="The four phases of the training loop drawn as a horizontal conveyor: forward pass gives a prediction, loss measures how wrong, backward pass gives gradients, update nudges the weights. A curved arrow loops from update back to forward, labelled repeat."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One lap of the training loop — then it repeats</text><rect x="14" y="44" width="108" height="34" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text x="68" y="59" text-anchor="middle" fill="#1F6280" font-size="9.5">1 · forward pass</text><text x="68" y="71" text-anchor="middle" fill="#1F6280" font-size="9">→ prediction</text><text x="128" y="63" fill="#B8AEA2">→</text><rect x="142" y="44" width="96" height="34" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="190" y="59" text-anchor="middle" fill="#C93B3B" font-size="9.5">2 · loss</text><text x="190" y="71" text-anchor="middle" fill="#C93B3B" font-size="9">→ how wrong</text><text x="244" y="63" fill="#B8AEA2">→</text><rect x="258" y="44" width="112" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text x="314" y="59" text-anchor="middle" fill="#5E5191" font-size="9.5">3 · backward pass</text><text x="314" y="71" text-anchor="middle" fill="#5E5191" font-size="9">→ gradients</text><text x="376" y="63" fill="#B8AEA2">→</text><rect x="390" y="44" width="112" height="34" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="446" y="59" text-anchor="middle" fill="#1a5c38" font-size="9.5">4 · update</text><text x="446" y="71" text-anchor="middle" fill="#1a5c38" font-size="9">→ nudge weights</text><path d="M446 82 C 446 130, 68 130, 68 84" fill="none" stroke="#C99A12" stroke-width="2" stroke-dasharray="4,3"/><polygon points="68,84 63,94 73,94" fill="#C99A12"/><text x="260" y="126" text-anchor="middle" fill="#9A7208" font-size="10">↻ repeat — every lap, the loss drops a little</text></g></svg>
%%%

The one line to keep: **forward → loss → backward → update, on repeat.** The first three steps you already built; today's new magic is Step 4, the actual nudge. Let's zoom into it next — the moment the network truly *changes*.

@@@ concept id=c2 tag="Nudge the aim" title="The update — the one step that changes the weights" gotit="Got the update"
This is the heart of the loop, so let's slow down. Back on the court: your last **free-throw** clanked off the *left* of the rim. Your body already knows what to do — aim a little to the *right*. Not all the way across the gym (you'd overcorrect and miss right), just a *small nudge* opposite to the miss. The network does the same thing with each weight. Yesterday's gradient told it "nudging this weight up makes the miss *worse*" — so it does the obvious thing and nudges the weight the **opposite** way, a small step toward *less* miss.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A basketball hoop seen from above. The last shot landed left of the rim, marked with an X. A red arrow shows the miss pointing left. A green arrow shows the aim being nudged the opposite way, a small step to the right, toward the center of the rim."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Miss went left → nudge the aim the OPPOSITE way, a small step right</text><circle cx="300" cy="96" r="46" fill="none" stroke="#C99A12" stroke-width="3"/><text x="300" y="152" text-anchor="middle" fill="#9A7208" font-size="9">the rim (target)</text><text x="188" y="100" text-anchor="middle" fill="#C93B3B" font-size="16">✕</text><text x="188" y="120" text-anchor="middle" fill="#C93B3B" font-size="9">last shot landed here</text><path d="M270 96 l-70 0" stroke="#C93B3B" stroke-width="2.5"/><polygon points="200,96 210,91 210,101" fill="#C93B3B"/><text x="235" y="86" text-anchor="middle" fill="#C93B3B" font-size="9">the miss →</text><path d="M300 70 l44 -20" stroke="#2D8B55" stroke-width="2.5"/><polygon points="344,50 333,50 338,60" fill="#2D8B55"/><text x="392" y="50" text-anchor="middle" fill="#2D8B55" font-size="9">nudge aim: small step</text><text x="392" y="63" text-anchor="middle" fill="#2D8B55" font-size="9">the opposite way</text></g></svg>
%%%

**What the aim-nudge picture gets right:** the correction is *small* and points *opposite* to the miss — nudge too hard and you'll fly past to the other side. **Where it breaks down:** a shooter feels the miss and guesses the fix; the network doesn't guess — the gradient gives it the exact opposite direction for every weight, no feeling required.

#### The update rule, in plain words
The update is one short sentence: **new weight = old weight − a small step in the gradient's direction.** The gradient points *uphill* (toward more loss), so the **minus sign** is what turns the network around to walk *downhill*. That "how big a step" number has a name — the [[learning rate||the step size: how big a nudge you make each loop. Too big overshoots, too small crawls]] — and you'll meet it properly in the next concept. Here is the whole rule as one line:

%%% formula
expr: w ← w − lr × (gradient of the loss for w)
note: "w" is a weight, "lr" is the learning rate (step size). The MINUS sign walks downhill — opposite the uphill gradient. The bias updates the exact same way: b ← b − lr × (its gradient).
%%%

Now watch it actually move. Say one weight should really be `3.0`, but it starts at `0.5` — like standing a big step left of the rim. Each loop, the gradient points back toward `3.0`, and with a learning rate of `0.25` the weight takes a *fraction* of that gap each time. Predict it first: will it jump straight to `3.0`, or creep up? Then run it.

%%% demo id=update label="run it — one weight, five nudges"
code: w=0.5; target=3.0; lr=0.25
code: for i in range(5): grad = 2*(w-target); w = w - lr*grad; print(round(w,3))
out: 1.75
out: 2.375
out: 2.688
out: 2.844
out: 2.922
take: <b>Update = old weight − lr × gradient.</b> The weight climbs 0.5 → 1.75 → 2.375 → 2.69 → 2.84 → 2.92, closing in on 3.0 a little each loop — never one giant leap. That patient, repeated nudge is exactly the free-throw drill. Run the loop longer and it lands right on 3.0.
%%%

And to *see* that creep, here is the same climb drawn as bars — the weight stepping toward its target, the gap shrinking each loop.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A bar chart of one weight climbing over five loops from 0.5 toward its target of 3.0. Bars rise 0.5, 1.75, 2.375, 2.69, 2.84, 2.92, each closer to a dashed line at 3.0. The gap between the bar top and the target shrinks every loop."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One weight nudging toward its target (3.0) — gap shrinks each loop</text><line x1="46" y1="140" x2="500" y2="140" stroke="#E5DFD6" stroke-width="1.5"/><line x1="46" y1="42" x2="500" y2="42" stroke="#2D8B55" stroke-width="1.3" stroke-dasharray="5,3"/><text x="502" y="45" fill="#2D8B55" font-size="9">target 3.0</text><g fill="#2A7B9B"><rect x="60" y="123" width="44" height="17"/><rect x="132" y="83" width="44" height="57"/><rect x="204" y="63" width="44" height="77"/><rect x="276" y="53" width="44" height="87"/><rect x="348" y="48" width="44" height="92"/><rect x="420" y="45" width="44" height="95"/></g><g fill="#1F6280" text-anchor="middle"><text x="82" y="118">0.5</text><text x="154" y="78">1.75</text><text x="226" y="58">2.38</text><text x="298" y="48">2.69</text><text x="370" y="43">2.84</text><text x="442" y="40">2.92</text></g><g fill="#6B645E" text-anchor="middle" font-size="9"><text x="82" y="154">start</text><text x="154" y="154">loop 1</text><text x="226" y="154">loop 2</text><text x="298" y="154">loop 3</text><text x="370" y="154">loop 4</text><text x="442" y="154">loop 5</text></g></g></svg>
%%%

That single move — subtract a small multiple of the gradient — is called **gradient descent**, and it is Step 4 of every loop. It is also the great-grandchild of a much cruder idea from the 1950s. Let's meet its ancestor next — it makes today's smooth version feel almost magical.

@@@ concept id=c3 tag="The old way" title="The perceptron — the crude ancestor of the loop" gotit="Got the ancestor"
Before the smooth loop you just met, there was a much blunter one, and seeing it makes today's version click. Imagine an old-school **free-throw** coach with only one rule: he stays silent while you're sinking shots, and the instant you *miss*, he blows a whistle and shoves your elbow a fixed inch — always the same shove, whether you missed by a hair or by a mile. If your shot was already good, he says *nothing* — no praise, no fine-tuning. That was the very first learning machine, the [[perceptron||Frank Rosenblatt's 1958 learning machine: it nudged weights by a fixed amount, and only when it got an example wrong]], built by Frank Rosenblatt in 1958.

%%% svg
<svg viewBox="0 0 520 184" role="img" aria-label="An old-fashioned coach with a whistle. On the left, a made shot: the coach is silent, no correction. On the right, a missed shot: the coach blows the whistle and gives a fixed-size elbow shove, the same size every time regardless of how big the miss was."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The perceptron coach: silent on hits, one fixed shove on any miss</text><line x1="260" y1="30" x2="260" y2="170" stroke="#E5DFD6" stroke-dasharray="4,3"/><rect x="20" y="34" width="222" height="130" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="131" y="54" text-anchor="middle" fill="#1a5c38">shot was GOOD</text><text x="131" y="92" text-anchor="middle" font-size="24">🏀 ✓</text><text x="131" y="126" text-anchor="middle" fill="#1a5c38" font-size="9.5">coach: silent 🤐</text><text x="131" y="144" text-anchor="middle" fill="#6B645E" font-size="9">no nudge at all</text><rect x="278" y="34" width="222" height="130" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="389" y="54" text-anchor="middle" fill="#C93B3B">shot MISSED</text><text x="389" y="92" text-anchor="middle" font-size="24">🏀 ✕</text><text x="389" y="126" text-anchor="middle" fill="#C93B3B" font-size="9.5">whistle 📣 + fixed shove</text><text x="389" y="144" text-anchor="middle" fill="#6B645E" font-size="9">same size, tiny miss or huge miss</text></g></svg>
%%%

**What the strict-coach picture gets right:** the perceptron only reacts to *mistakes*, and every correction is the *same fixed size*. **Where it breaks down:** a coach can at least see *how badly* you missed even if he ignores it; the perceptron had no smooth "how wrong" number at all, so it truly couldn't tell a near-miss from a disaster.

#### Why the modern loop is smoother
The perceptron had two rough edges, and fixing both is exactly what today's loop does:

- It only corrected on a **mistake**. A shot that was *nearly* perfect got zero feedback, so it never got polished.
- Every nudge was the **same fixed size**, no matter how big the miss.

Today's loop swaps in a smooth [[loss||one number saying how wrong the guess is; lower is better]] and its **gradient**, so *every* example — even a near-perfect one — gives a **graded** correction: a big miss earns a big nudge, a tiny miss earns a tiny one. Here's the two side by side.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Before-and-after comparison. Top row, the perceptron: correct examples get no bar, wrong examples all get the same fixed-height bar. Bottom row, gradient descent: every example gets a bar whose height matches how big its miss was, small misses give small bars, big misses give big bars."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Size of correction per example — fixed vs graded</text><text x="16" y="42" fill="#C93B3B" font-size="10">perceptron (fixed / only on miss):</text><line x1="30" y1="86" x2="360" y2="86" stroke="#E5DFD6"/><g fill="#C93B3B"><rect x="60" y="58" width="26" height="28"/><rect x="150" y="58" width="26" height="28"/><rect x="285" y="58" width="26" height="28"/></g><g fill="#6B645E" text-anchor="middle" font-size="8"><text x="73" y="98">miss</text><text x="118" y="98">✓ hit: 0</text><text x="163" y="98">miss</text><text x="208" y="98">✓ hit: 0</text><text x="253" y="98">✓ hit: 0</text><text x="298" y="98">miss</text></g><text x="16" y="128" fill="#2D8B55" font-size="10">gradient descent (graded / every example):</text><line x1="30" y1="172" x2="360" y2="172" stroke="#E5DFD6"/><g fill="#2D8B55"><rect x="60" y="150" width="26" height="22"/><rect x="105" y="166" width="26" height="6"/><rect x="150" y="140" width="26" height="32"/><rect x="195" y="163" width="26" height="9"/><rect x="240" y="158" width="26" height="14"/><rect x="285" y="132" width="26" height="40"/></g><text x="200" y="186" text-anchor="middle" fill="#6B645E" font-size="8">bar height = how big the miss — everyone gets a fair, graded nudge</text></g></svg>
%%%

That single upgrade — a graded nudge for *everyone*, every loop — is why the modern loop learns so much more smoothly than its 1958 ancestor. Now back to today's loop, and the one knob that decides how big each nudge is.

@@@ concept id=c4 tag="How big a nudge" title="The learning rate — and two ways it bites" gotit="Got the learning rate"
Now the single most important dial in the whole loop. Think of walking down a hill to a valley in the dark, and you can only take **fixed-size steps**. Take *baby* steps and you'll get there, but it takes all night. Take *giant leaps* and you might overshoot the valley completely — bound down one side, up the other, back and forth, never settling. The right stride is *big enough to make progress, small enough not to fly past the bottom*. That stride length is the [[learning rate||the step size: how far the network moves each loop. Written "lr". Too big overshoots, too small crawls]] — often written `lr` — and it multiplies every nudge in the update rule.

%%% svg
<svg viewBox="0 0 520 188" role="img" aria-label="Three walkers heading into a valley. Left: tiny baby steps, barely moving down the slope, labelled too small crawls. Middle: medium steps landing near the bottom, labelled just right. Right: giant leaps bouncing from one wall of the valley to the other and back, overshooting, labelled too big overshoots."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Step size into the valley: too small, just right, too big</text><path d="M20 40 Q 90 130 160 130 Q 175 130 180 40" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#2A7B9B" stroke-width="1.6" fill="none"><path d="M40 62 l10 8"/><path d="M50 70 l10 7"/><path d="M60 77 l9 6"/></g><text x="95" y="160" text-anchor="middle" fill="#1F6280" font-size="9">too small</text><text x="95" y="174" text-anchor="middle" fill="#6B645E" font-size="8">crawls forever</text><path d="M195 40 Q 265 130 335 130 Q 350 130 355 40" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#2D8B55" stroke-width="1.8" fill="none"><path d="M215 60 l30 34"/><path d="M245 94 l28 34"/></g><circle cx="273" cy="128" r="4" fill="#2D8B55"/><text x="270" y="160" text-anchor="middle" fill="#1a5c38" font-size="9">just right</text><text x="270" y="174" text-anchor="middle" fill="#6B645E" font-size="8">lands at the bottom</text><path d="M370 40 Q 440 130 510 130 Q 515 130 518 40" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#C93B3B" stroke-width="1.8" fill="none"><path d="M388 58 l70 66"/><path d="M458 124 l60 -70"/></g><text x="445" y="160" text-anchor="middle" fill="#C93B3B" font-size="9">too big</text><text x="445" y="174" text-anchor="middle" fill="#6B645E" font-size="8">overshoots, bounces</text></g></svg>
%%%

**What the stepping-into-a-valley picture gets right:** one dial (stride length) decides between crawling, arriving, and bouncing past. **Where it breaks down:** a hiker can shorten her stride as she nears the bottom by feel; a network uses the *same* learning rate every loop unless you deliberately shrink it — which is exactly the remedy we'll reach in a moment.

#### Puzzle 1 — the learning rate too *high*
Set `lr` too big and each nudge overshoots the bottom. The weight leaps past the target, the loss goes *up*, so the next nudge leaps back even harder — the loss **oscillates**, then **diverges**, and can blow up to `NaN` ("not a number", what you get from math gone infinite). **Cause:** the step is larger than the distance to the bottom. **Remedy:** *lower the learning rate.* You spot this instantly on the loss curve — it should slope *down*; if it zig-zags up, your `lr` is too big.

#### Puzzle 2 — the learning rate too *low*
Set `lr` too tiny and every nudge is a grain of sand. The loss *does* fall, but so slowly that training **crawls** — you'd wait hours for progress an hour of the right `lr` would give in seconds. **Cause:** steps too small to make real headway. **Remedy:** *raise the learning rate* until the loss falls briskly without zig-zagging.

Here are all three side by side — the shape of the loss curve tells you which one you're in.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three loss curves over loops. Green just-right curve slides smoothly down and flattens near zero. Blue too-low curve slopes down very gently, barely dropping, still high at the end. Red too-high curve zig-zags upward and diverges off the top."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Read the loss curve → diagnose the learning rate</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="20" y="90" fill="#6B645E" font-size="9" transform="rotate(-90 20 90)">loss</text><text x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">loops →</text><path d="M52 40 C 140 130, 300 146, 490 148" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text x="405" y="140" fill="#2D8B55" font-size="9">just right ✓ (falls, flattens)</text><path d="M52 46 C 200 60, 360 74, 490 84" fill="none" stroke="#2A7B9B" stroke-width="2.2"/><text x="360" y="70" fill="#1F6280" font-size="9">too low (crawls, still high)</text><path d="M52 120 L 100 60 L 148 132 L 196 44 L 244 140 L 292 30 L 330 150" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text x="130" y="34" fill="#C93B3B" font-size="9">too high (zig-zags → diverges → NaN)</text></g></svg>
%%%

#### The best-of-both remedy — a learning-rate schedule
Here's the trick that gets both: start with a *big* stride to cover ground fast, then *shrink* it as you near the valley so you settle gently instead of bouncing. Deliberately lowering `lr` as training goes on is a [[learning-rate schedule||a plan that shrinks the learning rate over time — big early steps for speed, small late steps to settle; also called decay]] (also called *decay*). Big early steps for speed, small late steps to settle — the smart middle path.

!!! c-info 🔬
<b>Optional (skippable) — the update rule, symbol by symbol.</b> For weight `w` with gradient `g` and learning rate `η` (the Greek letter "eta", the usual symbol for the learning rate): the update is `w ← w − η·g`. If `η` is bigger than roughly `2 / (curvature of the loss)` the step overshoots and the loss grows — that's the "too high" failure in one inequality. A schedule replaces the single `η` with a shrinking sequence `η₁ > η₂ > η₃ > …`. You never need this to run the loop — the loss curve tells you everything by eye.
!!!

With the stride set sensibly, the loop *will* march the loss down. But how do we *count* all this practice — and how do we know when to stop? That's the next piece.

@@@ concept id=c5 tag="Counting practice" title="Epochs, iterations, and the loss curve" gotit="Got epochs & the curve"
Now we need words for *how much* practice, and a way to *watch* it pay off. Picture studying a **deck of flashcards** for a test. Flipping through *one* card and fixing what you got wrong is a single small round of practice. Flipping through the *whole deck* once — every card, front to back — is one full pass. You'd repeat the whole deck many times before the test. Training uses those same two counts, with fancier names.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A deck of flashcards. One highlighted card is labelled iteration, one update. A bracket around the whole row of cards is labelled epoch, one full pass through all cards. Below, three copies of the deck labelled epoch 1, epoch 2, epoch 3 show the deck being repeated."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Flashcards: one card = an iteration, one full deck = an epoch</text><g><rect x="40" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="92" y="40" width="40" height="52" rx="4" fill="#E7F0F5" stroke="#2A7B9B" stroke-width="2"/><rect x="144" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="196" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="248" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/></g><path d="M112 96 l0 12" stroke="#2A7B9B"/><text x="112" y="122" text-anchor="middle" fill="#1F6280" font-size="9">1 iteration</text><text x="112" y="134" text-anchor="middle" fill="#6B645E" font-size="8">(one update)</text><path d="M40 34 l248 0" stroke="#7C6DAA" stroke-width="1.6"/><text x="164" y="30" text-anchor="middle" fill="#5E5191" font-size="9">1 epoch = whole deck once</text><g font-size="8" fill="#6B645E"><rect x="330" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text x="347" y="102" text-anchor="middle">epoch 1</text><rect x="392" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text x="409" y="102" text-anchor="middle">epoch 2</text><rect x="454" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text x="471" y="102" text-anchor="middle">epoch 3</text></g><text x="409" y="130" text-anchor="middle" fill="#6B645E" font-size="8">repeat the deck many times</text></g></svg>
%%%

**What the flashcards picture gets right:** one card is one small update, the whole deck is one full pass, and you repeat the deck many times. **Where it breaks down:** with flashcards you flip one card at a time; a network often grabs a small *handful* of examples per update instead of exactly one — but that's a knob (batching) we'll meet in a later module. For now, one example per nudge is a perfect mental model.

#### Two counts to keep straight
- An [[iteration||one loop of forward → loss → backward → update, i.e. one weight update, from one example (or one small batch)]] (also called a *step*) is a single trip around the four-step loop — **one update**.
- An [[epoch||one full pass over all of your training data — many iterations back to back]] is one full pass over *all* your training data. If you have 100 examples and update after each, one epoch is 100 iterations.

You train for many epochs, so the whole deck gets seen again and again. The natural next question: how do you *know* it's working?

#### The loss curve — your progress report
Every loop, jot down the loss and plot it against the loop number. That plot is the [[loss curve||a plot of the loss over loops; a falling curve is proof the network is learning]], and it is the single most useful picture in all of training. A curve that slides *down* is a network that is learning. When the curve **flattens out** near a low value — barely dropping anymore — the network has (roughly) finished; that flattening is called [[convergence||when the loss stops dropping much and levels off — the loop has mostly finished learning]]. Watch a healthy run: a steep drop early, then a gentle glide, then flat.

%%% svg
<svg viewBox="0 0 520 186" role="img" aria-label="A healthy loss curve over epochs. It starts high, drops steeply in the first few epochs, then bends and glides down more gently, then flattens into a nearly horizontal line near the bottom, where a label reads converged: loss stops dropping."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A healthy loss curve: steep drop, gentle glide, then flat</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">epochs →</text><path d="M52 36 C 110 120, 180 140, 260 146 C 340 149, 420 150, 494 150" fill="none" stroke="#2D8B55" stroke-width="2.4"/><circle cx="70" cy="52" r="3.5" fill="#2D8B55"/><text x="110" y="52" fill="#1a5c38" font-size="9">steep drop (fast learning)</text><circle cx="200" cy="141" r="3.5" fill="#2D8B55"/><text x="240" y="128" fill="#1a5c38" font-size="9">gentle glide</text><circle cx="470" cy="150" r="3.5" fill="#C99A12"/><text x="360" y="140" fill="#9A7208" font-size="9">flat = converged</text></g></svg>
%%%

Two failures hide in this curve, and they sit at opposite ends of the *time* dial — stop the loop too early or run it too long. Those are the next puzzle, and the last piece of the healthy loop.

@@@ concept id=c6 tag="Too little, too much" title="Puzzle — stopping too early, and drilling too long" gotit="Got early/late stopping"
How *long* should the loop run? Both extremes bite, and you've felt both when studying for a test. Study **too little** and you walk in unprepared — you bomb it. Study **too much in the wrong way** — memorizing the exact practice answers word-for-word — and you ace the practice sheet but freeze on the real exam, because it asks the *same ideas* with *different numbers*. Good practice lives in between: enough to learn the pattern, not so much that you memorize the exact questions.

%%% svg
<svg viewBox="0 0 520 188" role="img" aria-label="A student studying for a test, shown three ways. Left: barely studied, sad face, labelled too little, underfit. Middle: studied just enough, happy face, labelled just right. Right: memorized every practice answer word for word, confused on the real exam, labelled too much, memorized."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Studying for a test: too little, just right, too much</text><rect x="16" y="34" width="150" height="132" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="91" y="70" text-anchor="middle" font-size="26">😟</text><text x="91" y="104" text-anchor="middle" fill="#C93B3B">too little</text><text x="91" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">barely practised</text><text x="91" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ bombs the test</text><text x="91" y="156" text-anchor="middle" fill="#C93B3B" font-size="9">(underfit)</text><rect x="185" y="34" width="150" height="132" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="260" y="70" text-anchor="middle" font-size="26">🙂</text><text x="260" y="104" text-anchor="middle" fill="#1a5c38">just right</text><text x="260" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">learned the pattern</text><text x="260" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ handles new questions</text><rect x="354" y="34" width="150" height="132" rx="6" fill="#FDF3D6" stroke="#C99A12"/><text x="429" y="70" text-anchor="middle" font-size="26">😵</text><text x="429" y="104" text-anchor="middle" fill="#9A7208">too much</text><text x="429" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">memorized the answers</text><text x="429" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ freezes on new ones</text><text x="429" y="156" text-anchor="middle" fill="#9A7208" font-size="9">(overfit)</text></g></svg>
%%%

**What the studying picture gets right:** too little and too much both hurt, in opposite ways — one from not enough practice, one from the wrong kind. **Where it breaks down:** a student *knows* when they're just memorizing; a network can't tell on its own — its practice score keeps looking great even as it starts memorizing, so you need a *separate* check to catch it.

#### Puzzle — too few epochs (underfit)
Stop the loop too soon and the loss is still high — the network hasn't finished descending. This is **underfitting**: not enough practice. **Cause:** too few loops. **Remedy:** *train for more epochs*, and watch the loss curve until it flattens (converges).

#### Puzzle — too many epochs (overfit)
Run the loop *too* long and something sneaky happens: the training loss keeps dropping, but the network stops learning the real *pattern* and starts **memorizing the exact training examples**. That's [[overfitting||when the network memorizes the training data instead of the pattern, so it does great on data it has seen and badly on new data]]. **Cause:** too many loops relative to how much the data can teach. The catch: the *training* loss looks better and better — memorizing lowers it — so you can't spot the problem from training loss alone.

#### The remedy — a held-out check and early stopping
The fix is to keep a small slice of data the network *never trains on* — a [[validation set||a slice of data held back from training, used only to check if the network handles data it hasn't practised on]] — and measure the loss on it each epoch. While the network is truly learning, *both* losses fall. The moment it starts memorizing, the training loss keeps falling but the **validation loss turns back up** — that fork is your alarm. Stopping right at that turn is [[early stopping||halt training when the validation loss stops improving and starts rising — right before memorizing takes over]]. Watch the fork:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two loss curves over epochs. The blue training-loss curve falls steadily and keeps dropping toward zero. The red validation-loss curve falls at first alongside it, reaches a lowest point, then turns back upward. A vertical dashed line at that lowest point is labelled stop here, early stopping. The region after it is shaded and labelled overfitting."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Training loss vs validation loss — stop at the fork</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text x="270" y="174" text-anchor="middle" fill="#6B645E" font-size="9">epochs →</text><rect x="300" y="26" width="200" height="124" fill="#FDF3D6" opacity="0.55"/><text x="400" y="42" text-anchor="middle" fill="#9A7208" font-size="9">overfitting zone</text><path d="M52 40 C 160 110, 320 138, 494 148" fill="none" stroke="#2A7B9B" stroke-width="2.2"/><text x="410" y="140" fill="#1F6280" font-size="9">training loss (keeps falling)</text><path d="M52 46 C 150 96, 280 108, 300 108 C 380 108, 440 90, 494 70" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text x="360" y="66" fill="#C93B3B" font-size="9">validation loss (turns up!)</text><line x1="300" y1="26" x2="300" y2="150" stroke="#2D8B55" stroke-width="1.8" stroke-dasharray="5,3"/><circle cx="300" cy="108" r="4" fill="#2D8B55"/><text x="300" y="166" text-anchor="middle" fill="#1a5c38" font-size="9">⬆ stop here (early stopping)</text></g></svg>
%%%

So the loop has a Goldilocks amount of practice: run *enough* to converge, stop *before* the validation loss climbs. Before the recap, two easy housekeeping slips — and then one honest wall the loop can't climb.

@@@ concept id=c7 tag="Two easy slips" title="Two housekeeping traps — reset, and shuffle" gotit="Got reset & shuffle"
Two more traps, and these are the kind that bite *everyone* the first time — not because the idea is hard, but because they're easy to *forget*. Think of a shared **Etch A Sketch** (the red toy where you turn two knobs to draw, then shake it to erase). If you start a new drawing *without shaking the old one off first*, your new lines pile on top of the old — a scribbled mess. In the loop, the "old drawing" is last loop's gradient: if you don't clear it, this loop's gradient piles on top and the nudge goes haywire.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Two Etch A Sketch toys side by side. Left: not shaken clear first, so a new clean line is drawn on top of an old scribble, making a mess, labelled forgot to reset: gradients pile up. Right: shaken clear first, so only the new clean line shows, labelled reset first: clean gradient each loop."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Clear the sketch before you draw — reset the gradient each loop</text><rect x="24" y="34" width="212" height="120" rx="10" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><path d="M44 70 q30 30 60 -10 q30 40 60 0 q20 -20 50 20" fill="none" stroke="#B8AEA2" stroke-width="1.4"/><path d="M50 110 L110 60 L170 118 L216 74" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="130" y="140" text-anchor="middle" fill="#C93B3B" font-size="9">forgot to reset → old + new pile up</text><rect x="284" y="34" width="212" height="120" rx="10" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><path d="M310 110 L370 60 L430 118 L476 74" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="390" y="140" text-anchor="middle" fill="#1a5c38" font-size="9">reset first → clean gradient this loop</text></g></svg>
%%%

**What the Etch A Sketch picture gets right:** you must *clear the old marks* before each new drawing, or they add together. **Where it breaks down:** shaking an Etch A Sketch is a choice you make when it looks messy; in the loop the piling-up is *silent and automatic* — the tools add new gradients onto old ones by default, so you must reset *every* loop, on purpose.

#### Puzzle 1 — forgetting to reset the gradients
Most training tools *add* each loop's gradient onto whatever was there before (it's a default that helps in fancier setups). So if you don't clear it, loop 2's gradient sits on top of loop 1's, loop 3's on top of that — the nudge gets bigger and more wrong every loop. **Cause:** gradients accumulate by default. **Remedy:** [[zero the gradients||reset every weight's gradient to 0 at the start of each loop, before the backward pass — so each nudge uses only this loop's signal]] at the start of every loop (often literally a `zero_grad()` call). Watch the difference with real numbers — same gradient of `2` each loop, with and without a reset:

%%% demo id=zerograd label="run it — reset vs pile-up"
code: g = 2   # this loop's gradient, same every loop
code: acc = 0
code: for i in range(3): acc = acc + g; print('no reset -> used gradient', acc)
code: for i in range(3): used = g; print('with reset -> used gradient', used)
out: no reset -> used gradient 2
out: no reset -> used gradient 4
out: no reset -> used gradient 6
out: with reset -> used gradient 2
out: with reset -> used gradient 2
out: with reset -> used gradient 2
take: <b>Forget to reset and the gradient piles up: 2 → 4 → 6</b>, so the nudge grows wrong every loop. Reset each loop and it stays a clean 2. One tiny line — zero the gradients — saves the whole run.
%%%

#### Puzzle 2 — never shuffling the data
Here's the other slip. If you always feed the examples in the *same fixed order* — like only ever shooting **free-throws** from the exact same spot on the floor — the network's nudges get biased by that order, and it never generalizes to the rest of the court. **Cause:** a fixed, correlated order. **Remedy:** [[shuffle the data||mix the order of your training examples before each epoch, so the network doesn't lock onto the order itself]] before each epoch — reshuffle the deck every pass so no ordering can sneak into the weights.

!!! c-warn ⚠️
<b>The two silent slips:</b> both of these run *without any error message* — the loop keeps going, the numbers just come out wrong. Reset the gradients every loop, and reshuffle the data every epoch. Two one-line habits that separate a loop that learns from one that quietly doesn't.
!!!

You now know a healthy loop *and* its traps. One last honest truth before the recap: some targets a single-neuron loop simply cannot learn, no matter how perfectly you run it.

@@@ concept id=c8 tag="Where it hits a wall" title="Two honest limits of the loop" gotit="Got the limits"
The loop is powerful, but it is not magic — and knowing its edges is what separates someone who *uses* it from someone who *understands* it. Two honest walls. First, think about trying to draw a perfect **circle using only a straight ruler**. You can practise your ruler-drawing all year, and every stroke is a straight line — you will *never* get a circle out of a tool that can only make straight lines. More practice can't add a power the tool never had.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A grid of four points showing the XOR pattern: bottom-left and top-right are one class, top-left and bottom-right are the other. A dashed straight line tries to separate them but always leaves one point on the wrong side. A caption says no single straight line can split these — a one-neuron loop plateaus above zero."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A straight line can't split XOR — no matter how long you train</text><line x1="90" y1="40" x2="90" y2="150" stroke="#B8AEA2"/><line x1="90" y1="150" x2="240" y2="150" stroke="#B8AEA2"/><circle cx="110" cy="130" r="9" fill="#2A7B9B"/><circle cx="220" cy="60" r="9" fill="#2A7B9B"/><circle cx="110" cy="60" r="9" fill="#C93B3B"/><circle cx="220" cy="130" r="9" fill="#C93B3B"/><line x1="80" y1="150" x2="240" y2="55" stroke="#9A938A" stroke-width="1.8" stroke-dasharray="5,3"/><text x="165" y="172" text-anchor="middle" fill="#6B645E" font-size="8.5">a straight cut always mis-sorts one dot</text><line x1="300" y1="40" x2="300" y2="150" stroke="#B8AEA2"/><line x1="300" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="285" y="94" fill="#6B645E" font-size="8" transform="rotate(-90 285 94)">loss</text><text x="400" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">loops →</text><path d="M306 50 C 340 96, 380 104, 494 104" fill="none" stroke="#C93B3B" stroke-width="2.2"/><line x1="300" y1="104" x2="494" y2="104" stroke="#C99A12" stroke-width="1" stroke-dasharray="3,3"/><text x="410" y="98" fill="#9A7208" font-size="8.5">stuck above 0 forever</text></g></svg>
%%%

**What the ruler picture gets right:** some jobs need a tool the ruler simply doesn't have, and repetition can't conjure it. **Where it breaks down:** you *can* draw a circle by adding a second tool — a compass; likewise a network learns XOR by adding more neurons in a hidden layer. The limit is the *single* neuron, not networks in general — that's exactly why we stack them, coming in a later module.

#### Wall 1 — a single neuron can't learn XOR
[[XOR||"exclusive or": output 1 when the two inputs differ, 0 when they match — the four points can't be split by any single straight line]] is the classic example. A single neuron can only carve the input with *one straight line*, and no straight line splits the XOR pattern. Run the loop for a *million* loops and the loss just **plateaus above zero** — flat, never reaching a good score. More iterations never beat a *representational* ceiling: if the tool can't express the answer, practice can't find it.

#### Wall 2 — getting stuck in a local dip
The second wall is subtler. The loss landscape isn't always a single clean bowl; it can be lumpy, with small dips scattered around the real valley. This is called a [[non-convex||a bumpy landscape with more than one dip, so the lowest nearby point may not be the lowest point overall]] landscape. Gradient descent only ever feels the slope *right where it stands*, so it can settle into a small dip — a [[local minimum||a spot where every direction is uphill nearby, so the loop stops — even though a deeper valley exists elsewhere]] — think of a hiker who stops in a roadside ditch, sees only uphill all around, and declares "this is the bottom!" while the real valley is over the next ridge. The loss flattens, so it *looks* converged, but it isn't the best possible.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A bumpy loss landscape with two dips. A ball has rolled into the shallower left dip, a local minimum, and stopped, with uphill on both sides. The deeper right dip, the global minimum, is lower but unreached. Labels mark local minimum stuck here and global minimum the real best."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A bumpy landscape: the loop can stop in a shallow dip</text><path d="M30 60 C 90 130, 150 120, 190 96 C 230 72, 270 150, 340 152 C 400 154, 440 90, 500 60" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="170" cy="108" r="7" fill="#C99A12" stroke="#fff" stroke-width="1.5"/><text x="170" y="88" text-anchor="middle" fill="#9A7208" font-size="9">🛑 stuck here</text><text x="150" y="130" text-anchor="middle" fill="#9A7208" font-size="8.5">local minimum</text><circle cx="320" cy="150" r="6" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="360" y="150" fill="#1a5c38" font-size="9">← global minimum</text><text x="330" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">the real best (deeper), unreached</text></g></svg>
%%%

**The honest takeaway:** a flat, converged loss curve is *good news* but not a *guarantee* you found the lowest possible loss — you found the bottom of *whatever bowl you happened to roll into*. (Good news for later: in the huge networks behind real models, these dips are usually shallow and not a big problem in practice — but it's an honest limit worth knowing.) Now let's gather the whole day onto one page.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you just built the engine that makes every neural network learn. Here's the whole day held in one picture: our **free-throw** drill, now labelled with real training words. Read the beats as one lap of practice, and keep the cheat-sheets below as your map. **Where the free-throw picture finally breaks down:** a player fixes one thing per shot, but the loop nudges every knob at once, thousands of times — that scale is the only reason it can learn things a person never could.

The loop you built, in order:
- **Step 1 · Forward** — take the shot: run the input to a prediction.
- **Step 2 · Loss** — measure the miss: one number for how wrong the guess is.
- **Step 3 · Backward** — which way off: backprop hands every weight its gradient.
- **Step 4 · Update** — nudge the aim: `w ← w − lr × gradient`; the **minus** walks downhill.
- **Repeat** over **iterations** (one update) and **epochs** (one full pass), watching the **loss curve** fall and flatten (**converge**).
- The **learning rate** sets the stride; the **perceptron** (1958) was the crude, fixed-nudge ancestor.

Here's the one picture to keep — the four steps, the knob, the curve, and where each trap strikes.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Recap diagram. Top: the four-step loop forward, loss, backward, update in a ring with a repeat arrow, and a learning-rate dial feeding the update step. Middle: a loss curve falling and flattening at convergence. Bottom: a row listing the six traps and their one-line cures."><g font-family="monospace" font-size="9"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">The whole day: the loop, the stride, the curve, the traps</text><rect x="16" y="26" width="86" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="59" y="43" text-anchor="middle" fill="#1F6280">1 forward</text><rect x="120" y="26" width="70" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="155" y="43" text-anchor="middle" fill="#C93B3B">2 loss</text><rect x="208" y="26" width="94" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text x="255" y="43" text-anchor="middle" fill="#5E5191">3 backward</text><rect x="320" y="26" width="82" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="361" y="43" text-anchor="middle" fill="#1a5c38">4 update</text><g stroke="#B8AEA2" stroke-width="1.6" fill="#B8AEA2"><path d="M102 39 l16 0"/><polygon points="118,39 111,35 111,43"/><path d="M190 39 l16 0"/><polygon points="206,39 199,35 199,43"/><path d="M302 39 l16 0"/><polygon points="318,39 311,35 311,43"/></g><path d="M361 52 C 361 70, 59 70, 59 54" fill="none" stroke="#C99A12" stroke-width="1.6" stroke-dasharray="3,2"/><polygon points="59,54 55,62 63,62" fill="#C99A12"/><text x="210" y="66" text-anchor="middle" fill="#9A7208">↻ repeat</text><rect x="418" y="26" width="86" height="26" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text x="461" y="40" text-anchor="middle" fill="#9A7208">learning rate</text><text x="461" y="49" text-anchor="middle" fill="#9A7208" font-size="7.5">(stride into update)</text><line x1="46" y1="86" x2="46" y2="140" stroke="#B8AEA2"/><line x1="46" y1="140" x2="250" y2="140" stroke="#B8AEA2"/><path d="M50 92 C 90 132, 160 138, 246 139" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="150" y="100" fill="#1a5c38">loss curve → converges (flat)</text><text x="280" y="100" fill="#6B645E">read the curve:</text><text x="280" y="114" fill="#C93B3B">zig-zag up → lr too high</text><text x="280" y="128" fill="#2A7B9B">barely moves → lr too low</text><text x="280" y="142" fill="#9A7208">val loss turns up → overfit</text><line x1="16" y1="152" x2="504" y2="152" stroke="#E5DFD6"/><text x="16" y="168" fill="#C93B3B">lr high → lower it / schedule</text><text x="270" y="168" fill="#C93B3B">lr low → raise it</text><text x="16" y="182" fill="#C93B3B">no reset → zero the gradients</text><text x="270" y="182" fill="#C93B3B">fixed order → shuffle each epoch</text><text x="16" y="196" fill="#C93B3B">too few epochs → train longer</text><text x="270" y="196" fill="#C93B3B">too many → early stopping</text><text x="16" y="212" fill="#6B645E">limit: one neuron can't learn XOR · GD can stop in a local dip</text></g></svg>
%%%

#### Cheat-sheet · the loop, its knobs, and the six traps
%%% table
:: Piece / trap :: In one line :: Remember
forward → loss → backward → update :: one lap of the loop :: repeat until the loss flattens
update rule :: w ← w − lr × gradient :: the MINUS walks downhill
learning rate (lr) :: the step size each loop :: too big overshoots, too small crawls
epoch vs iteration :: full data pass vs one update :: many iterations make one epoch
lr too high :: loss zig-zags up, → NaN :: cure: lower lr / schedule (decay)
lr too low :: loss barely moves :: cure: raise lr
forgot to reset gradients :: they pile up silently :: cure: zero the gradients each loop
fixed data order :: updates biased by order :: cure: shuffle each epoch
too few epochs :: underfit, loss still high :: cure: train longer, watch convergence
too many epochs :: overfit, memorizes :: cure: early stopping on validation loss
one-neuron limit :: can't learn XOR, plateaus :: cure: more neurons (later module)
local minimum :: stops in a shallow dip :: converged ≠ globally best
%%%

#### Cheat-sheet · the words you met today
%%% jargon
training loop | the four-step cycle — forward, loss, backward, update — repeated until the loss gets small
forward pass | run the input through the network to get a prediction (a guess)
loss | one number saying how wrong the guess is; lower is better
backward pass | backpropagation: hands every weight its gradient (which way to nudge)
update | w ← w − lr × gradient; the minus sign steps downhill toward less loss
learning rate | the step size each loop; too big overshoots, too small crawls
learning-rate schedule | shrinking the learning rate over time — big steps early, small steps late (decay)
epoch | one full pass over all the training data
iteration | one loop / one weight update, from one example (or small batch)
loss curve | a plot of the loss over loops; a falling, flattening curve means it's learning
convergence | the loss levels off near a low value — the loop has mostly finished
perceptron | Rosenblatt's 1958 machine: a fixed nudge, only on mistakes — the crude ancestor
zero the gradients | reset every gradient to 0 each loop so nudges don't pile up
shuffle | mix the data order each epoch so order doesn't bias the weights
overfitting | memorizing the training data instead of the pattern — great on seen data, bad on new
validation set | held-out data used only to check generalization, never to train
early stopping | halt when the validation loss stops improving, before overfitting takes over
XOR | a target no single straight line (one neuron) can split, no matter how long you train
local minimum | a shallow dip the loop can settle in that isn't the globally lowest loss
%%%

That's the whole day. Tomorrow you keep the loop but swap Step 4 for something smarter than a plain nudge — **optimizers** like Momentum and Adam, which take the same downhill idea and make it faster and steadier.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What are the four steps of the training loop, in order? | a:2 | loss → forward → update → backward | update → backward → loss → forward | forward pass → loss → backward pass → update | backward → update → forward → loss | fb: One lap is forward (take the shot) → loss (measure the miss) → backward (which way off?) → update (nudge the aim), then repeat. It's the free-throw drill: try, measure, correct, again.
q: In the update rule w ← w − lr × gradient, what does the MINUS sign do, and what does lr control? | a:1 | Minus makes training faster; lr sets how many layers | The gradient points uphill, so minus steps the weight downhill (toward less loss); lr is the step size — how big each nudge is | Minus deletes bad weights; lr is the number of epochs | Minus is just notation with no effect; lr is the loss value | fb: The gradient points UPHILL (toward more loss), so subtracting it walks the weight downhill toward less loss. The learning rate (lr) scales that step: too big overshoots and the loss diverges, too small and training crawls.
q: Your loss curve zig-zags upward and blows up to NaN. What's wrong, and what's the fix? | a:2 | The learning rate is too low; fix: lower it more | You have too few epochs; fix: shuffle the data | The learning rate is too high — each step overshoots the bottom, so the loss diverges; fix: lower the learning rate (or use a schedule that shrinks it over time) | The gradients weren't reset; fix: train for more epochs | fb: A loss that grows and oscillates means each step is bigger than the distance to the minimum — the classic too-high learning rate. Lower it, or use a learning-rate schedule (decay) so early steps are big for speed and late steps are small to settle.
q: Training loss keeps falling but validation loss starts rising. What is happening, and what's the remedy? | a:1 | The learning rate is too low; remedy: raise it | The network is overfitting — memorizing the training data instead of the pattern; remedy: early stopping (halt when validation loss stops improving) | The gradients are piling up; remedy: zero them each loop | The neuron can't represent the target; remedy: shuffle the data | fb: When the two curves fork — training loss down, validation loss up — the network has stopped learning the pattern and started memorizing the training examples. That's overfitting. Early stopping halts right at the fork, keeping the version that generalizes best.
%%%

@@@ produce id=produce tag="Produce" title="Run a real training loop and watch the loss fall" gotit="Done"
Time to run the engine yourself. You'll build the full four-step loop for one neuron and **watch** the loss drop, loop after loop, then poke the learning rate to *feel* the too-high and too-low failures. **Predict first:** with a sensible learning rate, will the loss fall smoothly to near zero, or jump around? Then set the learning rate way too high and predict again — will it fall, or blow up? Run it and **watch** which prediction was right. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-06-training-loop/experiment.py`. (1) Set up one neuron learning a simple target: pick `x`, a true `target`, start `w` and `b` at small numbers, and a learning rate `lr = 0.1`. (2) Write the loop for, say, 50 iterations: **forward** `pred = w*x + b`; **loss** `L = (pred - target)**2`; **backward** `grad_w = 2*(pred-target)*x` and `grad_b = 2*(pred-target)`; **update** `w = w - lr*grad_w` and `b = b - lr*grad_b`. Print the loss every few loops and **watch** it fall. (3) Now rerun with `lr = 1.5` and **observe** the loss zig-zag and blow up, then with `lr = 0.0005` and **observe** it barely move. Run with `python3 sessions/m02-the-neuron/day-06-training-loop/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 2 Day 6 artifact.

Create sessions/m02-the-neuron/day-06-training-loop/experiment.py that, with a comment on each step:
1. Trains ONE neuron with a full training loop. Setup: x=2.0, target=1.0, w=0.0, b=0.0, lr=0.1.
2. Loops 50 iterations doing the four steps: forward pred=w*x+b; loss L=(pred-target)**2; backward grad_w=2*(pred-target)*x and grad_b=2*(pred-target); update w=w-lr*grad_w, b=b-lr*grad_b. Store the loss each loop and print it every 10 loops so we can watch it fall toward ~0 (convergence).
3. Re-runs the same loop twice more to show the learning-rate failures: once with lr=1.5 (watch the loss oscillate and blow up / diverge) and once with lr=0.0005 (watch the loss barely move — crawling). Print the final loss for each lr and a one-line comment naming the failure.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the loop, learning)
- With `lr = 0.1` the **loss falls fast at first, then flattens near zero** — a healthy loss curve, converging. That flattening is the network having (roughly) learned it.
- With `lr = 1.5` the **loss zig-zags and blows up** — each step overshoots the bottom. That's the too-high failure, live; the cure is a smaller `lr`.
- With `lr = 0.0005` the **loss barely moves** after 50 loops — the too-low failure; the cure is a bigger `lr`. Seeing all three back to back is exactly how engineers pick a learning rate by eye.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-06-training-loop/log.md`: (1) the four steps of the loop, in order; (2) what the learning rate controls, and what happens if it's too high or too low; (3) one trap from today (reset, shuffle, underfit, overfit, or a capability limit) and its one-line cure.
!!!

@@@ fin
