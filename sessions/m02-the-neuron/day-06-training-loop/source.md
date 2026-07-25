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
@lede Think about learning to sink a **free-throw**. You step up, you shoot — clank, off the left of the rim. So you aim a hair to the right and shoot again. A little closer. Clank, adjust, shoot. Nobody sinks it clean on the first try. You get good by *repeating one small fix, over and over*, until the ball drops through. That rhythm — *try, measure the miss, nudge, repeat* — is the whole secret of how a neural network learns. It has a name: the **training loop**. It is the engine humming underneath every model you've heard of: the tiny neuron you'll train today, the face-unlock on a phone, the thing that picks your next video, all the way up to ChatGPT. You already built each separate piece — the guess, the loss that scores the miss, the gradient that says which way to nudge. Today they click together into one repeating cycle. Press go, and watch one number fall, shot after shot, until the network can sink it.
@goal You'll snap the pieces into one loop: **forward** (take the shot) → **loss** (measure the miss) → **backward** (which way was I off?) → **update** (nudge the aim), then go again. You'll meet the **learning rate** — how bold each nudge is — and count practice in **epochs** and **iterations**. You'll read a falling **loss curve** the way an engineer does. Then we'll poke the sneaky ways practice goes wrong, each a small puzzle with a one-line fix. Plain words come first every time; any heavy math sits in a skippable box.
%%% warmup
q: From yesterday: the gradient points which way, and which way do we step to learn? | a:1 | It points downhill, and we step the same way | It points uphill (toward more loss), and we step the OPPOSITE way, downhill | It points sideways, and we don't move | It points at the answer, and we jump straight there | concept: gradient-downhill | fb: The gradient points UPHILL toward more loss, so we step the opposite way — downhill toward less loss. That minus sign is exactly today's update.
q: From yesterday: what does backpropagation do in one backward sweep? | a:2 | Redoes the whole forward pass once per weight | Picks the biggest weight and deletes it | Multiplies local slopes backward to hand EVERY weight its gradient at once | Guesses each weight's blame with a hunch | concept: backprop-sweep | fb: Backprop starts at the loss and sweeps backward, multiplying local slopes, so every weight gets its gradient in ONE pass. Today that gradient is what the update actually uses.
q: From day 4: what is the loss, in one line? | a:0 | One number saying how wrong the guess is; lower is better | The number of layers in the network | The speed of training | A random starting weight | concept: loss-signal | fb: The loss is a single number scoring how wrong the guess is — lower is better, 0 is perfect. Today we watch it fall, loop after loop.
%%%

@@@ concept id=c1 tag="The practice loop" title="The training loop — four steps, on repeat" gotit="Got the loop"
Let's stay on the basketball court, because you already know this loop by heart. Picture one **free-throw** practice cycle. **(1)** You shoot. **(2)** You *look* at where it landed — a foot left of the rim. **(3)** You work out *which way* you were off: "I leaned left." **(4)** You *nudge* your aim a little to the right. Then you do the whole thing again. Four little steps, over and over — and every single one is something you already built this module.

%%% svg
<svg viewBox="0 0 520 232" role="img" aria-label="A basketball free-throw practice loop drawn as four steps arranged in a circle: shoot the ball, see where it landed, figure out which way you were off, nudge your aim. Arrows connect them in a ring, showing the cycle repeats."><g font-family="monospace" font-size="10.5"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One free-throw practice cycle — repeated until it drops clean</text><circle cx="140" cy="70" r="40" fill="#E7F0F5" stroke="#2A7B9B"/><text x="140" y="64" text-anchor="middle" fill="#1F6280">🏀 shoot</text><text x="140" y="78" text-anchor="middle" fill="#1F6280" font-size="9">(take the shot)</text><circle cx="380" cy="70" r="40" fill="#FDECEC" stroke="#C93B3B"/><text x="380" y="64" text-anchor="middle" fill="#C93B3B">📏 measure</text><text x="380" y="78" text-anchor="middle" fill="#C93B3B" font-size="9">(how far off?)</text><circle cx="380" cy="180" r="40" fill="#EDE9F8" stroke="#7C6DAA"/><text x="380" y="174" text-anchor="middle" fill="#5E5191">🧭 which way?</text><text x="380" y="188" text-anchor="middle" fill="#5E5191" font-size="9">(I leaned left)</text><circle cx="140" cy="180" r="40" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="174" text-anchor="middle" fill="#1a5c38">🎯 nudge aim</text><text x="140" y="188" text-anchor="middle" fill="#1a5c38" font-size="9">(a little right)</text><g stroke="#B8AEA2" stroke-width="2" fill="#B8AEA2"><path d="M180 62 l160 0"/><polygon points="340,62 331,57 331,67"/><path d="M380 110 l0 30"/><polygon points="380,140 375,131 385,131"/><path d="M340 188 l-160 0"/><polygon points="180,188 189,183 189,193"/><path d="M140 140 l0 -30"/><polygon points="140,110 135,119 145,119"/></g><text x="260" y="130" text-anchor="middle" fill="#6B645E" font-size="9">round and round — each loop, a little closer</text></g></svg>
%%%

**What the picture gets right:** you improve by *repeating* one small correction — no single shot fixes everything, but hundreds of tiny fixes add up.

**Where it breaks down:** a player fixes one thing at a time (aim, then arc, then power). A network nudges *every* knob at once, on every lap.

#### Now the same four steps, in network words
Same rhythm, new vocabulary. Read one rung at a time — the left side is the move, the right side is why it's there.

%%% steps
step: 1 · the shot — the **forward pass**
why: run an input through the neuron to **make a guess**. It's the **weighted sum** from day 1 plus its bend from day 2 — and the guess it spits out is called the **prediction**.
step: 2 · the miss — the **loss**
why: **one number** saying **how wrong** that guess is. Lower is better, 0 is perfect. This is the "ball landed a foot left" measurement.
step: 3 · the blame — the **backward pass**
why: **backpropagation** sweeps backward from the loss and hands every weight its **gradient** — **which way** that weight would have to move to make the miss bigger. That's the "I leaned left" **blame** report.
step: 4 · the nudge — **the update**
why: **shift each weight** a small step **against** the gradient — the direction that *lowers* the loss. This is the only step that actually changes the network.
step: ↻ then jump back to step 1
why: one lap barely helps. Hundreds of laps is what learning *is*.
%%%

%%% insight
Here's the part that surprises everyone: nothing in this loop is clever. There is no "understanding" step, no moment where the network figures anything out. It just repeats four boring moves — guess, measure, blame, nudge — a few hundred thousand times. That stubborn repetition, and nothing else, is what turns a pile of random numbers into something that can finish your sentences.
%%%

So far: four steps, in a ring, forever. Now here's the same lap drawn flat, so you can watch one lap feed straight into the next.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="The four phases of the training loop drawn as a horizontal conveyor: forward pass gives a prediction, loss measures how wrong, backward pass gives gradients, update nudges the weights. A curved arrow loops from update back to forward, labelled repeat."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One lap of the training loop — then it repeats</text><rect x="14" y="44" width="108" height="34" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text x="68" y="59" text-anchor="middle" fill="#1F6280" font-size="9.5">1 · forward pass</text><text x="68" y="71" text-anchor="middle" fill="#1F6280" font-size="9">→ prediction</text><text x="128" y="63" fill="#B8AEA2">→</text><rect x="142" y="44" width="96" height="34" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="190" y="59" text-anchor="middle" fill="#C93B3B" font-size="9.5">2 · loss</text><text x="190" y="71" text-anchor="middle" fill="#C93B3B" font-size="9">→ how wrong</text><text x="244" y="63" fill="#B8AEA2">→</text><rect x="258" y="44" width="112" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text x="314" y="59" text-anchor="middle" fill="#5E5191" font-size="9.5">3 · backward pass</text><text x="314" y="71" text-anchor="middle" fill="#5E5191" font-size="9">→ gradients</text><text x="376" y="63" fill="#B8AEA2">→</text><rect x="390" y="44" width="112" height="34" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="446" y="59" text-anchor="middle" fill="#1a5c38" font-size="9.5">4 · update</text><text x="446" y="71" text-anchor="middle" fill="#1a5c38" font-size="9">→ nudge weights</text><path d="M446 82 C 446 130, 68 130, 68 84" fill="none" stroke="#C99A12" stroke-width="2" stroke-dasharray="4,3"/><polygon points="68,84 63,94 73,94" fill="#C99A12"/><text x="260" y="126" text-anchor="middle" fill="#9A7208" font-size="10">↻ repeat — every lap, the loss drops a little</text></g></svg>
%%%

The one line to keep: **forward → loss → backward → update, on repeat.**

Steps 1–3 you already own. Step 4 is today's new magic — the single moment the network *changes*. Let's zoom in.

@@@ concept id=c2 tag="Nudge the aim" title="The update — the one step that changes the weights" gotit="Got the update"
This is the heart of the loop, so let's slow right down. Your last **free-throw** clanked off the *left* of the rim. Your body already knows the fix: aim a little to the *right*. Not all the way across the gym — you'd overcorrect and miss right — just a *small nudge*, the **opposite way** from the miss. The network does exactly this with each weight, and it never has to guess: yesterday's gradient already told it "pushing this weight up makes the miss *worse*."

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A basketball hoop seen from above. The last shot landed left of the rim, marked with an X. A red arrow shows the miss pointing left. A green arrow shows the aim being nudged the opposite way, a small step to the right, toward the center of the rim."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Miss went left → nudge the aim the OPPOSITE way, a small step right</text><circle cx="300" cy="96" r="46" fill="none" stroke="#C99A12" stroke-width="3"/><text x="300" y="152" text-anchor="middle" fill="#9A7208" font-size="9">the rim (target)</text><text x="188" y="100" text-anchor="middle" fill="#C93B3B" font-size="16">✕</text><text x="188" y="120" text-anchor="middle" fill="#C93B3B" font-size="9">last shot landed here</text><path d="M270 96 l-70 0" stroke="#C93B3B" stroke-width="2.5"/><polygon points="200,96 210,91 210,101" fill="#C93B3B"/><text x="235" y="86" text-anchor="middle" fill="#C93B3B" font-size="9">the miss →</text><path d="M300 70 l44 -20" stroke="#2D8B55" stroke-width="2.5"/><polygon points="344,50 333,50 338,60" fill="#2D8B55"/><text x="392" y="50" text-anchor="middle" fill="#2D8B55" font-size="9">nudge aim: small step</text><text x="392" y="63" text-anchor="middle" fill="#2D8B55" font-size="9">the opposite way</text></g></svg>
%%%

**What the picture gets right:** the correction is *small* and points *opposite* to the miss.

**Where it breaks down:** a shooter *feels* the fix. The network doesn't feel anything — the gradient handed it the exact opposite direction, for every weight at once.

#### Building the update rule, one piece at a time
Five short rungs and you'll have the whole thing. Notice we're just writing down what your arm already did.

%%% steps
step: start from the aim you have now — the old weight
why: nobody starts over from scratch each shot. You keep your stance and adjust it.
step: ask the gradient which way is *worse*
why: the gradient points **uphill**, toward MORE loss. It's the direction of the miss.
step: turn around — **subtract** it
why: that little **minus sign** is the whole trick. Going the **opposite way** from "worse" means going toward *better*, downhill.
step: take only a *fraction* of that direction — multiply it by the **learning rate**
why: a full-size jump would fly past the rim. The learning rate decides **how big a step** you take for a given slope; you'll meet it properly in concept 4, right after we say hello to the loop's 1958 ancestor.
step: that gives you the **new weight**, and every weight in the network gets the same treatment on the same lap
why: this is **the update**. Do it to `w`, do it to `b`, do it to all million of them — one lap, one nudge each.
%%%

Put those five rungs on one line and you get the rule people actually write down:

%%% formula
expr: w ← w − lr × (gradient of the loss for w)
note: "w" is a weight, "lr" is the learning rate (also called the step size). The MINUS sign walks downhill — opposite the uphill gradient. The bias updates the exact same way: b ← b − lr × (its gradient).
%%%

%%% insight
Why *small* steps, when we know the direction? Because the gradient is only honest *right where you stand*. It's the slope under your feet, not a map of the whole hill. Trust it for a short step and you go down; trust it for a giant leap and you'll land somewhere it never promised anything about. That single fact explains almost every training disaster in this lesson.
%%%

So far: subtract a small piece of the gradient. Now let's watch a real weight do it. It starts at `0.5` and its ideal home is `3.0` — like standing one big step left of the rim.

%%% demo id=update label="run it — one weight, five nudges"
predict: With a learning rate of 0.25, does the weight jump straight to 3.0 on the first nudge, creep up in shrinking hops, or overshoot past it? Pick one, then reveal.
code: w=0.5; target=3.0; lr=0.25
code: for i in range(5): grad = 2*(w-target); w = w - lr*grad; print(round(w,3))
out: 1.75
out: 2.375
out: 2.688
out: 2.844
out: 2.922
take: <b>Shrinking hops.</b> 0.5 → 1.75 → 2.375 → 2.69 → 2.84 → 2.92. Each nudge closes a chunk of the remaining gap, so as the gap shrinks the gradient shrinks — and the hops shrink with it, all on their own. That patient repeated nudge IS the free-throw drill. Run it longer and it settles right on 3.0.
%%%

And here is that same climb as bars, so you can *see* the gap closing instead of reading numbers.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A bar chart of one weight climbing over five loops from 0.5 toward its target of 3.0. Bars rise 0.5, 1.75, 2.375, 2.69, 2.84, 2.92, each closer to a dashed line at 3.0. The gap between the bar top and the target shrinks every loop."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One weight nudging toward its target (3.0) — gap shrinks each loop</text><line x1="46" y1="140" x2="500" y2="140" stroke="#E5DFD6" stroke-width="1.5"/><line x1="46" y1="42" x2="500" y2="42" stroke="#2D8B55" stroke-width="1.3" stroke-dasharray="5,3"/><text x="502" y="45" fill="#2D8B55" font-size="9">target 3.0</text><g fill="#2A7B9B"><rect x="60" y="123" width="44" height="17"/><rect x="132" y="83" width="44" height="57"/><rect x="204" y="63" width="44" height="77"/><rect x="276" y="53" width="44" height="87"/><rect x="348" y="48" width="44" height="92"/><rect x="420" y="45" width="44" height="95"/></g><g fill="#1F6280" text-anchor="middle"><text x="82" y="118">0.5</text><text x="154" y="78">1.75</text><text x="226" y="58">2.38</text><text x="298" y="48">2.69</text><text x="370" y="43">2.84</text><text x="442" y="40">2.92</text></g><g fill="#6B645E" text-anchor="middle" font-size="9"><text x="82" y="154">start</text><text x="154" y="154">loop 1</text><text x="226" y="154">loop 2</text><text x="298" y="154">loop 3</text><text x="370" y="154">loop 4</text><text x="442" y="154">loop 5</text></g></g></svg>
%%%

That one move — subtract a small multiple of the gradient — has a name: **gradient descent**. It is Step 4 of every loop ever run.

It is also the great-grandchild of a much cruder idea from 1958. Meeting the ancestor makes today's version feel almost luxurious.

@@@ concept id=c3 tag="The old way" title="The perceptron — the crude ancestor of the loop" gotit="Got the ancestor"
Imagine an old-school **free-throw** coach with exactly one rule. While you're sinking shots he says *nothing* — no praise, no tips. The instant you *miss*, he blows a whistle and shoves your elbow — the same shove no matter **how badly** you missed. Airball off the backboard? One shove. Rattled out by a fingernail? The same shove. That coach is a real machine: the [[perceptron||Frank Rosenblatt's 1958 learning machine: it corrected the weights only when it got an example wrong, and its correction carried no measure of how wrong]], built by Frank **Rosenblatt** in 1958 — the first thing anyone ever called a learning machine, and the crude ancestor of everything you did in the last concept.

%%% svg
<svg viewBox="0 0 520 184" role="img" aria-label="An old-fashioned coach with a whistle. On the left, a made shot: the coach is silent, no correction. On the right, a missed shot: the coach blows the whistle and gives an elbow shove whose size carries no information about how big the miss was."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The perceptron coach: silent on hits, one blind shove on any miss</text><line x1="260" y1="30" x2="260" y2="170" stroke="#E5DFD6" stroke-dasharray="4,3"/><rect x="20" y="34" width="222" height="130" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="131" y="54" text-anchor="middle" fill="#1a5c38">shot was GOOD</text><text x="131" y="92" text-anchor="middle" font-size="24">🏀 ✓</text><text x="131" y="126" text-anchor="middle" fill="#1a5c38" font-size="9.5">coach: silent 🤐</text><text x="131" y="144" text-anchor="middle" fill="#6B645E" font-size="9">no nudge at all</text><rect x="278" y="34" width="222" height="130" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="389" y="54" text-anchor="middle" fill="#C93B3B">shot MISSED</text><text x="389" y="92" text-anchor="middle" font-size="24">🏀 ✕</text><text x="389" y="126" text-anchor="middle" fill="#C93B3B" font-size="9.5">whistle 📣 + a shove</text><text x="389" y="144" text-anchor="middle" fill="#6B645E" font-size="9">no measure of HOW wrong</text></g></svg>
%%%

**What the picture gets right:** it reacts **only on mistakes**, and the correction carries no information about the size of the miss — a **fixed rule**, not a measured one.

**Where it breaks down:** a real coach can at least *see* how badly you missed, even if he ignores it. The perceptron had no "how wrong" number at all — no loss, nothing to plot.

#### Two rough edges — and the upgrade that fixed both
This is a short story with a happy ending, and it explains why today's loop looks the way it does.

%%% steps
step: rough edge 1 — silence on the shots that barely squeaked in
why: a ball that rattled around the rim and *dropped* counted as a hit, so it got zero feedback. Therefore the barely-right cases never got polished, and the machine stopped improving the moment it was merely correct.
step: rough edge 2 — the shove ignores how wrong you were
why: a hair-thin miss and a wild airball earn the same-size correction, because the perceptron has no number for **how wrong** it was. That means the machine could not aim its effort where the error actually was.
step: the upgrade — a smooth loss, and its gradient
why: today's loop scores *every* example with a number, then reads the slope of that number. Which means a big miss earns a big nudge, a tiny miss earns a tiny one, and even a barely-right shot gets polished — a **graded** correction for everyone, on every lap.
step: the bonus you get for free
why: because there's now a number, you can *plot* it. The perceptron had nothing to watch; you get a curve that tells you at a glance whether the run is healthy.
%%%

%%% insight
Notice the shape of that story, because it repeats through all of machine learning: someone replaces a hard yes/no rule with a smooth number you can measure, and suddenly you can take derivatives, take small steps, and watch progress. "Make it smooth so you can nudge it" is arguably the single most reused idea in the field.
%%%

So far: a blind shove versus a graded nudge. Here's what that difference looks like across six practice shots.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Before-and-after comparison. Top row, the perceptron: correct examples get no bar, wrong examples all get the same-height bar because the correction ignores how big the miss was. Bottom row, gradient descent: every example gets a bar whose height matches how big its miss was, small misses give small bars, big misses give big bars."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Size of correction per example — blind vs graded</text><text x="16" y="42" fill="#C93B3B" font-size="10">perceptron (only on a miss, size ignores how wrong):</text><line x1="30" y1="86" x2="360" y2="86" stroke="#E5DFD6"/><g fill="#C93B3B"><rect x="60" y="58" width="26" height="28"/><rect x="150" y="58" width="26" height="28"/><rect x="285" y="58" width="26" height="28"/></g><g fill="#6B645E" text-anchor="middle" font-size="8"><text x="73" y="98">miss</text><text x="118" y="98">✓ hit: 0</text><text x="163" y="98">miss</text><text x="208" y="98">✓ hit: 0</text><text x="253" y="98">✓ hit: 0</text><text x="298" y="98">miss</text></g><text x="16" y="128" fill="#2D8B55" font-size="10">gradient descent (graded / every example):</text><line x1="30" y1="172" x2="360" y2="172" stroke="#E5DFD6"/><g fill="#2D8B55"><rect x="60" y="150" width="26" height="22"/><rect x="105" y="166" width="26" height="6"/><rect x="150" y="140" width="26" height="32"/><rect x="195" y="163" width="26" height="9"/><rect x="240" y="158" width="26" height="14"/><rect x="285" y="132" width="26" height="40"/></g><text x="200" y="186" text-anchor="middle" fill="#6B645E" font-size="8">bar height = how big the miss — everyone gets a fair, graded nudge</text></g></svg>
%%%

You just placed today's loop in history — 1958's blind shove became a graded, measurable nudge. Nice.

Now back to your loop, and its single most important dial.

@@@ concept id=c4 tag="How big a nudge" title="The learning rate — and two ways it bites" gotit="Got the learning rate"
Picture walking down a hill into a valley, in the dark. You can't see the bottom, but you can *feel* the tilt of the ground under your boots. Before you set off you choose one rule — how **boldly** you turn that tilt into a step — and you are not allowed to change your mind halfway down. Choose timid and every step is a shuffle. Choose bold and steep ground launches you a long way. Here's the lovely part: with *one* setting, your steps are naturally long high on the steep slope and short down where the ground flattens. The steps shrink by themselves. What stays the same all walk long is the boldness.

%%% svg
<svg viewBox="0 0 520 206" role="img" aria-label="Three walkers going down the same valley with three different boldness settings. Left: a timid setting, steps stay tiny and the walker barely descends. Middle: a good setting, the steps start long on the steep slope and get shorter on their own as the ground flattens, landing at the bottom. Right: a too-bold setting, each leap flies clean across the valley and lands higher up the far side."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One boldness setting for the whole walk — the STEP still changes with the ground</text><path d="M18 46 Q 95 152 166 46" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#2A7B9B" stroke-width="1.8" fill="none"><path d="M30 60 l9 9"/><path d="M39 69 l8 8"/><path d="M47 77 l8 7"/><path d="M55 84 l7 7"/></g><circle cx="62" cy="91" r="4" fill="#2A7B9B"/><text x="92" y="172" text-anchor="middle" fill="#1F6280" font-size="9.5">timid setting</text><text x="92" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">tiny steps → still up the slope</text><path d="M186 46 Q 263 152 334 46" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#2D8B55" stroke-width="2" fill="none"><path d="M196 58 l26 38"/><path d="M222 96 l21 22"/><path d="M243 118 l13 10"/></g><circle cx="262" cy="131" r="4.5" fill="#2D8B55"/><text x="260" y="172" text-anchor="middle" fill="#1a5c38" font-size="9.5">good setting</text><text x="260" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">steps shrink on their own → lands</text><path d="M354 46 Q 431 152 502 46" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#C93B3B" stroke-width="2" fill="none"><path d="M370 64 l106 34"/><path d="M476 98 l-116 -50"/></g><circle cx="360" cy="48" r="4" fill="#C93B3B"/><text x="428" y="172" text-anchor="middle" fill="#C93B3B" font-size="9.5">too bold</text><text x="428" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">each leap lands higher up</text></g></svg>
%%%

That boldness setting is the [[learning rate||the multiplier on every nudge: step = learning rate × gradient. Also called the step size. Too bold flings you past the bottom; too timid crawls]] — written `lr` — and it multiplies every single nudge, so it decides **how big a step** you take on any given slope. (You'll also hear it called the **step size**.) It is the single most fiddled-with number in the whole field: the team training the model that ranks your video feed, and the one that unlocks a phone with a face, are tuning this exact dial today.

**What the picture gets right:** one setting decides between crawling, arriving, and flying past the bottom forever.

**Where it breaks down:** the loop's real step is `lr × gradient`, so it *already* shrinks by itself as you near the bottom — those are the shrinking hops you just watched in concept 2. What stays fixed is `lr`, the multiplier. That's exactly why a bold `lr` can still fling you past the bottom even from close up: the slope down there is small, but a big enough multiplier turns a small slope into a big step.

#### When the setting is too bold — the loss climbs
This one is worth savouring, because it feels backwards: a *bolder* step makes the loss go **up**. Three rungs and you'll have the whole runaway.

%%% steps
step: the leap — a bold multiplier can carry you *past* the bottom
why: crossing the bottom is fine — you're still closer than you were, so the loss still falls. It only turns bad when the step carries you **more than twice** the distance you had left: then you land *farther* out than you started.
step: the runaway — farther out means steeper ground, which means a longer leap still
why: same multiplier, bigger slope, therefore a bigger jump — back across the bottom and farther out again. So the **weight bounces back and forth across the bottom while the loss climbs every lap**. It **diverges**, and it **blows up** until the number is too big for the computer (it prints `inf`) — and then the arithmetic stops making sense at all: `NaN`, "not a number".
step: the fix — shrink the multiplier
why: **Cause:** the learning rate was **too high**, so each step **overshoot**s by more than twice the distance left. **Remedy:** lower it until the curve slopes down instead of climbing.
%%%

So the whole story turns on *how far* past the bottom you land. Here are the two cases side by side, with the distance-to-the-bottom written on each hop.

%%% svg
<svg viewBox="0 0 520 214" role="img" aria-label="Two bowls compared. Left bowl: a hop overshoots the bottom a little, distance left goes from 1.0 to 0.4 to 0.16, so the ball crosses back and forth over the bottom but still closes in and the loss falls. Right bowl: each hop overshoots by more than twice the distance left, so distance goes 1.0 to 1.5 to 2.25, landing farther out every lap and the loss climbs."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">How far past the bottom you land decides everything</text><text x="95" y="34" text-anchor="middle" fill="#2D8B55" font-size="10">overshoot a LITTLE (under 2×)</text><path d="M24 56 Q 95 162 166 56" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><line x1="95" y1="60" x2="95" y2="118" stroke="#E5DFD6" stroke-dasharray="3,3"/><text x="95" y="130" text-anchor="middle" fill="#6B645E" font-size="8">bottom</text><circle cx="155" cy="71" r="4.5" fill="#2D8B55"/><text x="172" y="70" fill="#1a5c38" font-size="8.5">1.0</text><circle cx="71" cy="103" r="4.5" fill="#2D8B55"/><text x="52" y="100" fill="#1a5c38" font-size="8.5">0.4</text><circle cx="105" cy="108" r="4.5" fill="#2D8B55"/><text x="112" y="118" fill="#1a5c38" font-size="8.5">0.16</text><path d="M152 76 C 120 100, 90 104, 74 100" fill="none" stroke="#2D8B55" stroke-width="1.4" stroke-dasharray="4,3"/><path d="M75 107 C 88 114, 96 112, 102 110" fill="none" stroke="#2D8B55" stroke-width="1.4" stroke-dasharray="4,3"/><text x="95" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">ball crosses the bottom — but still closes in ✓</text><text x="95" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">distance left: 1.0 → 0.4 → 0.16 → …</text><text x="385" y="34" text-anchor="middle" fill="#C93B3B" font-size="10">overshoot MORE than 2×</text><path d="M314 56 Q 385 162 456 56" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><line x1="385" y1="60" x2="385" y2="118" stroke="#E5DFD6" stroke-dasharray="3,3"/><text x="385" y="130" text-anchor="middle" fill="#6B645E" font-size="8">bottom</text><circle cx="415" cy="100" r="4.5" fill="#C93B3B"/><text x="424" y="110" fill="#C93B3B" font-size="8.5">1.0</text><circle cx="340" cy="88" r="4.5" fill="#C93B3B"/><text x="322" y="84" fill="#C93B3B" font-size="8.5">1.5</text><circle cx="453" cy="61" r="4.5" fill="#C93B3B"/><text x="462" y="58" fill="#C93B3B" font-size="8.5">2.25</text><path d="M411 96 C 390 84, 366 80, 344 84" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><path d="M344 84 C 390 70, 430 62, 449 60" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><text x="385" y="150" text-anchor="middle" fill="#C93B3B" font-size="9">lands FARTHER out every lap → the loss climbs ✕</text><text x="385" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">distance left: 1.0 → 1.5 → 2.25 → …</text><line x1="20" y1="178" x2="500" y2="178" stroke="#E5DFD6"/><text x="260" y="196" text-anchor="middle" fill="#2C2A28" font-size="9.5">Overshooting is not the enemy. Overshooting by more than DOUBLE is.</text><text x="260" y="208" text-anchor="middle" fill="#6B645E" font-size="8.5">left: messy but it lands · right: runaway</text></g></svg>
%%%

%%% insight
This is the failure you will actually hit, over and over, for the rest of your ML life — and the loss curve tells you instantly. A curve that **climbs** instead of falling is almost never a "hard problem"; it's almost always a learning rate that's **too high**. That one reflex will save you hours. And notice how narrow the honest rule is: a hop that crosses the bottom is *fine*, even good. It's only the hop that lands farther out than it started that eats the run.
%%%

Now stop reading and **play with it.** Drag the learning-rate slider below. Predict before each drag: at a tiny `lr`, does the ball reach the bottom? Then crank it up and watch three different behaviours appear — a smooth slide, a ball that hops across the bottom and still lands, and finally a ball that never settles at all.

%%% viz src=../../viz/gradient-descent.html title="Gradient descent playground — drag the learning rate" caption="Drag the learning-rate slider and step the ball down the bowl. Small setting = many tiny hops. Middle settings (try around 0.5–0.9) make the BALL hop back and forth ACROSS the bottom while still closing in — messy but it lands. Push past about 1.0 and it stops settling at all: every hop lands farther out and the loss climbs. Find that flip point."
%%%

#### When the setting is too timid — the loss crawls
Same dial, opposite bite, much less drama. Each nudge is a grain of sand: the loss *does* fall, but it **crawls** — after a thousand laps it **barely moves**. **Cause:** a learning rate **too low**, so the steps are **too tiny** to make headway. **Remedy:** raise it until the loss falls briskly *without* climbing.

So far: **too high** climbs, **too low** crawls. And you can tell which one you're in from the *shape* of the curve alone.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three loss curves over loops. Green just-right curve slides smoothly down and flattens near zero. Blue too-low curve slopes down very gently, barely dropping, still high at the end. Red too-high curve climbs steeply lap after lap and runs off the top of the chart."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Read the loss curve → diagnose the learning rate</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="20" y="90" fill="#6B645E" font-size="9" transform="rotate(-90 20 90)">loss</text><text x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">loops →</text><path d="M52 40 C 140 130, 300 146, 490 148" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text x="392" y="140" fill="#2D8B55" font-size="9">just right ✓ (falls, flattens)</text><path d="M52 60 C 200 74, 360 86, 490 96" fill="none" stroke="#2A7B9B" stroke-width="2.2"/><text x="352" y="82" fill="#1F6280" font-size="9">too low (crawls, still high)</text><path d="M52 60 C 90 58, 120 50, 146 40 C 164 34, 176 30, 184 27" fill="none" stroke="#C93B3B" stroke-width="2.4"/><polygon points="184,26 175,33 182,36" fill="#C93B3B"/><text x="196" y="36" fill="#C93B3B" font-size="9">too high (climbs → diverges → NaN)</text><text x="196" y="48" fill="#6B645E" font-size="8">runs off the top of the chart</text></g></svg>
%%%

One honest footnote on that red line: in a small loop like today's the climb is a clean, steady rise, every lap bigger than the last. In a real run with millions of weights the same runaway usually looks *jagged* — different weights blow up at different moments — so people often describe it as "spiky". Either way the trend is the same, and the diagnosis is the same: up and away means lower the learning rate.

#### The neat trick: start bold, then soften
A bold setting is great *far* from the bottom and terrible *near* it. So do the obvious thing: start bold, then **shrink the learning rate** as training goes on. **Smaller steps later** means you cover ground fast early and settle gently at the end. A plan that does this is a [[learning-rate schedule||a plan that shrinks the learning rate over time — bold early steps for speed, gentle late steps to settle; also called decay]], also called **decay**, and essentially every real model you've heard of is trained with one.

!!! c-ok 🎯
<b>Coming soon — Day 8 is a whole day on this one dial.</b> Today you only need what the loop needs: `lr` is the multiplier, too bold climbs, too timid crawls, and a schedule softens it over time. <i>How</i> to hunt for a good value (sweeping it by factors of ten and reading each curve) and <i>how</i> to design the shrink (the shapes a schedule can take) are Day 8's whole job. Park them there and enjoy the ride.
!!!

%%% hint
t1: Confused why a BIG step makes the loss go UP instead of just learning faster? Picture the walker in the valley again — think about *how far past* the bottom one leap can carry you.
t2: Say the bottom is 1 step to your right, and your leap lands you 3 steps to the right. You started 1 away; now you are 2 away — farther than before. Next lap the ground out there is steeper, so the leap is bigger still, and you land farther out again.
t3: Crossing the bottom is fine; landing farther out than you started is not. That happens when the step is more than **twice** the distance you had left — so the loss grows instead of falling. Lower the learning rate (the multiplier) and every step drops back under that line.
%%%

!!! c-info 🔬
<b>Optional (skippable) — the same thing in symbols.</b> For weight `w` with gradient `g` and learning rate `η` (the Greek letter "eta", the usual symbol): the update is `w ← w − η·g`. The step *length* is `η × |g|`, and on a bowl-shaped loss `|g|` grows with your distance from the bottom — which is why the step shrinks all by itself as you approach, with `η` untouched. The run only runs away when `η` is bigger than roughly `2 / (curvature of the loss)`; that inequality is exactly the "more than twice the distance left" rule, written in symbols. A schedule replaces the single `η` with a shrinking sequence `η₁ > η₂ > η₃ > ��`. You never need any of this to run a loop — the curve tells you everything by eye.
!!!

Dial sorted — and that's **two of the six traps** already in your kit. Next: how do we *count* all this practice, and how do we know when to stop?

@@@ concept id=c5 tag="Counting practice" title="Epochs, iterations, and the loss curve" gotit="Got epochs & the curve"
Picture studying with a **deck of flashcards** before a test. Flipping *one* card and fixing what you got wrong is one small round of practice. Flipping through the *whole deck*, front to back, is one full pass. And you'd go through the deck many times before test day. Training counts practice in exactly those two units — it just gives them fancier names. Once you have the names you unlock the single most useful picture in all of machine learning: a plot of your own progress.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A deck of flashcards. One highlighted card is labelled iteration, one update. A bracket around the whole row of cards is labelled epoch, one full pass through all cards. Below, three copies of the deck labelled epoch 1, epoch 2, epoch 3 show the deck being repeated."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Flashcards: one card = an iteration, one full deck = an epoch</text><g><rect x="40" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="92" y="40" width="40" height="52" rx="4" fill="#E7F0F5" stroke="#2A7B9B" stroke-width="2"/><rect x="144" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="196" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="248" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/></g><path d="M112 96 l0 12" stroke="#2A7B9B"/><text x="112" y="122" text-anchor="middle" fill="#1F6280" font-size="9">1 iteration</text><text x="112" y="134" text-anchor="middle" fill="#6B645E" font-size="8">(one update)</text><path d="M40 34 l248 0" stroke="#7C6DAA" stroke-width="1.6"/><text x="164" y="30" text-anchor="middle" fill="#5E5191" font-size="9">1 epoch = whole deck once</text><g font-size="8" fill="#6B645E"><rect x="330" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text x="347" y="102" text-anchor="middle">epoch 1</text><rect x="392" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text x="409" y="102" text-anchor="middle">epoch 2</text><rect x="454" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text x="471" y="102" text-anchor="middle">epoch 3</text></g><text x="409" y="130" text-anchor="middle" fill="#6B645E" font-size="8">repeat the deck many times</text></g></svg>
%%%

**What the picture gets right:** one card is one small update, the whole deck is one full pass, and you repeat the deck many times.

**Where it breaks down:** with flashcards you flip one card at a time. A network often grabs a small *handful* of cards per update, and that handful has a name: a [[batch||the handful of examples used for ONE update. Today our batch is a single example; how big to make the handful is a knob you'll tune in a later module]]. Today's batch is exactly one example, which keeps the counting simple — the handful size is a knob for a later module.

#### The two counts, and then the plot
Three short rungs. The third one is the payoff.

%%% steps
step: one card, one fix — an **iteration**
why: an [[iteration||one loop of forward → loss → backward → update, i.e. one weight update]] (also called **one step**) is a single trip round the four-step loop. So an iteration is exactly **one update** — nothing more.
step: the whole deck once — an **epoch**
why: an [[epoch||one full pass over all of your training data — many iterations back to back]] is **one full pass** over all your training data. Therefore 100 examples with one update each = 100 iterations in one epoch. Epochs are just iterations, bundled.
step: write down the loss after every lap, then **plot the loss**
why: that plot is your progress report. Because the loss is one number per lap, you can literally **watch it fall** — the network's whole education on one line.
%%%

Quick play before we look at the plot — guess the numbers first, they're friendlier than they look.

%%% demo id=count label="predict, then run — how many nudges hide inside '3 epochs'?"
predict: You have 200 training examples and you nudge once per example. How many iterations is ONE epoch? And after 3 epochs, how many nudges has the network taken — 3, 200, or something much bigger?
code: examples = 200                 # rows in the training set
code: iters_per_epoch = examples     # one nudge per example today (batch of 1)
code: for epoch in range(1, 4): print('after epoch', epoch, '-> total iterations:', epoch*iters_per_epoch)
out: after epoch 1 -> total iterations: 200
out: after epoch 2 -> total iterations: 400
out: after epoch 3 -> total iterations: 600
take: <b>200 iterations per epoch, 600 nudges after only 3 epochs.</b> That's why "just 3 epochs" already means hundreds of tiny corrections — and why nobody counts iterations out loud. Epochs are the human-sized unit.
%%%

%%% insight
This little plot you're about to meet is the instrument panel every ML engineer stares at all day — the same graph is up on a screen somewhere right now for the model that picks your next song. And you can read it already: it's how you tell "my model is learning" from "my model is broken" without knowing a single thing about the data. That's a genuinely professional skill, and you got it for the price of one graph.
%%%

So far: iterations, epochs, and a number to plot. Here's what a *healthy* run looks like.

%%% svg
<svg viewBox="0 0 520 186" role="img" aria-label="A healthy loss curve over epochs. It starts high, drops steeply in the first few epochs, then bends and glides down more gently, then flattens into a nearly horizontal line near the bottom, where a label reads converged: loss stops dropping."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A healthy loss curve: steep drop, gentle glide, then flat</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">epochs →</text><path d="M52 36 C 110 120, 180 140, 260 146 C 340 149, 420 150, 494 150" fill="none" stroke="#2D8B55" stroke-width="2.4"/><circle cx="70" cy="52" r="3.5" fill="#2D8B55"/><text x="110" y="52" fill="#1a5c38" font-size="9">steep drop (fast learning)</text><circle cx="200" cy="141" r="3.5" fill="#2D8B55"/><text x="240" y="128" fill="#1a5c38" font-size="9">gentle glide</text><circle cx="470" cy="150" r="3.5" fill="#C99A12"/><text x="360" y="140" fill="#9A7208" font-size="9">flat = converged</text></g></svg>
%%%

Read it left to right in three beats. A steep drop, while there's plenty to learn. Then a gentle glide, because the nudges get smaller as the gap closes. Then the curve **flattens out**.

That last beat has a name. So what is convergence, exactly? When the loss **levels off** and **stops improving**, we say the run reached [[convergence||when the loss stops dropping much and levels off — the loop has mostly finished learning]] — the loop has mostly finished. One more little guess before you move on:

%%% demo id=flatten label="predict, then run — does the loss ever hit exactly 0?"
predict: Here are the losses from a healthy run, printed every 10 laps. Before you look: does the last number land exactly on 0, or just get very close and then stall?
code: losses = [(0,1.0),(10,0.1216),(20,0.0148),(30,0.0018),(40,0.00022),(50,0.00003)]
code: for lap, loss in losses: print(f'lap {lap:>2}: loss {loss:.5f}')
out: lap  0: loss 1.00000
out: lap 10: loss 0.12160
out: lap 20: loss 0.01480
out: lap 30: loss 0.00180
out: lap 40: loss 0.00022
out: lap 50: loss 0.00003
take: <b>Close, never exactly 0.</b> Watch the row labels — each printed line is <i>ten</i> laps apart, and each of those ten-lap jumps cuts the loss to about an eighth. One single lap trims it only to about 0.81 of what it was, so the per-lap drops get microscopic near the end. The curve <b>goes down</b> and then looks flat even though it is still creeping. That's convergence: not "finished", just "no longer worth another lap".
%%%

So: a [[loss curve||a plot of the loss over loops; a falling curve is proof the network is learning]] that **goes down** and then flattens is the shape you're hunting for. Anything else is a clue — and reading those clues is exactly what the next sections hand you.

@@@ concept id=c6 tag="Two free wins" title="Two one-line habits that make every run trustworthy" gotit="Got reset & shuffle"
Good news first: this section is two **free wins**. Two habits, one line of code each, and once you've met them you will never lose an afternoon to either again — which is more than most people can say, because these two bite *everybody* exactly once. Think of an **Etch A Sketch**: the red toy where two knobs draw a line and you shake it to erase. Start a new drawing without shaking the old one off and your fresh clean line lands on top of yesterday's scribble. In the loop, the "old drawing" is last lap's gradient.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Two Etch A Sketch toys side by side. Left: not shaken clear first, so a new clean line is drawn on top of an old scribble, making a mess, labelled forgot to reset: gradients pile up. Right: shaken clear first, so only the new clean line shows, labelled reset first: clean gradient each loop."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Clear the sketch before you draw — reset the gradient each loop</text><rect x="24" y="34" width="212" height="120" rx="10" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><path d="M44 70 q30 30 60 -10 q30 40 60 0 q20 -20 50 20" fill="none" stroke="#B8AEA2" stroke-width="1.4"/><path d="M50 110 L110 60 L170 118 L216 74" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="130" y="140" text-anchor="middle" fill="#C93B3B" font-size="9">forgot to reset → old + new pile up</text><rect x="284" y="34" width="212" height="120" rx="10" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><path d="M310 110 L370 60 L430 118 L476 74" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="390" y="140" text-anchor="middle" fill="#1a5c38" font-size="9">reset first → clean gradient this loop</text></g></svg>
%%%

**What the picture gets right:** you must clear the old marks before each new drawing, or they add together.

**Where it breaks down:** you shake an Etch A Sketch when it *looks* messy. In the loop the piling-up is silent and automatic — nothing looks wrong, so you have to reset on purpose, every lap.

#### Win 1 · clear the sketch — reset the gradients
Three rungs, and the middle one is why this is nasty rather than merely annoying.

%%% steps
step: the default — the tools **accumulate**
why: most training tools *add* each new gradient onto whatever was already sitting there. It's a deliberate default that helps in fancier setups, therefore nobody warns you about it.
step: the pile-up — **gradients pile up** lap after lap
why: if you **forget to reset**, lap 2 uses lap 1 + lap 2, lap 3 uses all three, and so on. Which means your effective step grows every lap — you accidentally re-created the "too bold" runaway from the last section, with the learning rate untouched.
step: the fix — one line, at the top of every lap
why: [[zero the gradients||reset every weight's gradient to 0 at the start of each loop, before the backward pass — so each nudge uses only this loop's signal]] — literally `zero_grad()` in most frameworks. **Remedy:** **reset the gradients each step** — **clear the gradients** so each nudge carries only today's signal.
%%%

%%% insight
Why this one earns a callout: there is no error message. No crash, no warning, no red text — the loop runs happily and the numbers quietly rot. Silent bugs are the expensive kind, and "did I clear the gradients?" belongs on your reflex list forever.
%%%

Same gradient of `2` arriving every lap. Guess what the loop *uses* in each case.

%%% demo id=zerograd label="predict, then run — reset vs pile-up"
predict: If the same gradient (2) shows up on all three laps and you never clear the gradients, is the gradient the loop uses on lap 3 still 2, or something bigger? How much bigger?
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
take: <b>2 → 4 → 6.</b> Forget to reset and by lap 3 the nudge is three times too big — and it keeps growing. Clear the gradients and it stays a clean 2 forever. One tiny line saves the whole run.
%%%

#### Win 2 · don't practise from one spot — shuffle the data
The second win is about *order*, and it's a one-liner too. Practising **free-throw** after **free-throw** from the exact same spot on the floor makes you great at that one spot and useless everywhere else. Feed your examples in the **same fixed order** every epoch and the nudges get **biased by that order** — and **never shuffling** really bites when your data happens to arrive sorted (all the class-0 rows, then all the class-1 rows), because each stretch of laps drags the weights one way and the next stretch drags them back.

The remedy is delightfully cheap: [[shuffle the data||mix the order of your training examples before each epoch, so the network can't lock onto the order itself]] — **reshuffle each epoch**. **Mix up the order** and the problem is simply gone.

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="Top row: cards arriving in a fixed sorted order, all class zero cards first then all class one cards, with an arrow showing the weights being dragged one way then yanked back the other way. Bottom row: the same cards shuffled into a mixed order, with a straight steady arrow showing balanced nudges."><g font-family="monospace" font-size="9"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Same data, two orders — sorted drags, shuffled steadies</text><text x="14" y="42" fill="#C93B3B" font-size="10">never shuffling (sorted):</text><g><rect x="150" y="30" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="178" y="30" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="206" y="30" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="234" y="30" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="266" y="30" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="294" y="30" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="322" y="30" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="350" y="30" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/></g><path d="M150 66 L258 92" stroke="#C93B3B" stroke-width="2"/><path d="M258 92 L374 62" stroke="#C93B3B" stroke-width="2"/><polygon points="374,62 364,62 368,71" fill="#C93B3B"/><text x="400" y="82" fill="#C93B3B" font-size="9">weights get yanked</text><text x="400" y="94" fill="#6B645E" font-size="8">one way, then back</text><line x1="14" y1="106" x2="504" y2="106" stroke="#E5DFD6"/><text x="14" y="126" fill="#2D8B55" font-size="10">reshuffle each epoch:</text><g><rect x="150" y="118" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="178" y="118" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="206" y="118" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="234" y="118" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="266" y="118" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="294" y="118" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="322" y="118" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="350" y="118" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/></g><path d="M150 152 L374 152" stroke="#2D8B55" stroke-width="2"/><polygon points="374,152 364,148 364,156" fill="#2D8B55"/><text x="400" y="150" fill="#2D8B55" font-size="9">steady, balanced</text><text x="400" y="162" fill="#6B645E" font-size="8">no order bias</text></g></svg>
%%%

!!! c-warn ⚠️
<b>Both wins are silent if you skip them.</b> No error, no crash — the loop runs and the numbers just come out wrong. So: reset the gradients every lap, reshuffle the data every epoch. Two one-line habits that separate a loop that learns from one that only looks like it does.
!!!

🎉 **Two wins banked** — that's **four of the six traps** down, and they cost you a paragraph each. Now let's put the traps down for a minute and go *play*, because there's something rather lovely to see.

@@@ concept id=c7 tag="Watch it work" title="Fun break — what the loop is actually moving" gotit="Had a play"
Numbers falling in a column are fine. But what is the loop *actually doing* out there in the world? Here's a picture you can hold. You're on a gym floor with basketballs on one side of the room and footballs on the other, all jumbled together. You lay a long **rope** on the floor and try to split them: balls on this side, footballs on that side. The rope is too far left? Slide it right. Tilted the wrong way? Spin it a little. Every nudge from the loop is one of those slides or spins — and after enough of them the rope sits exactly where it should. That's it. That's what training *looks* like from outside.

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="A gym floor seen from above with two kinds of balls scattered on it. Left panel: a rope laid on the floor in the wrong place, so some balls are on the wrong side, with arrows showing it can be slid and spun. Right panel: after many nudges the rope sits in the right place and every ball is on its correct side."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A rope on the gym floor: slide it, spin it, until the two kinds are split</text><rect x="14" y="28" width="232" height="150" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><text x="130" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">rope in the wrong place</text><g font-size="13"><text x="46" y="84">🏀</text><text x="76" y="112">🏀</text><text x="52" y="140">🏀</text><text x="104" y="76">🏀</text><text x="150" y="94">🏈</text><text x="186" y="70">🏈</text><text x="196" y="118">🏈</text><text x="156" y="146">🏈</text></g><line x1="60" y1="60" x2="96" y2="166" stroke="#9A7208" stroke-width="3"/><text x="30" y="60" fill="#9A7208" font-size="8">rope</text><g stroke="#2D8B55" stroke-width="1.8" fill="#2D8B55"><path d="M108 150 l30 0"/><polygon points="138,150 130,146 130,154"/></g><text x="164" y="166" fill="#1a5c38" font-size="8">slide →</text><text x="122" y="56" fill="#1a5c38" font-size="8">↻ spin</text><rect x="272" y="28" width="232" height="150" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="388" y="46" text-anchor="middle" fill="#1a5c38" font-size="10">after many nudges ✓</text><g font-size="13"><text x="292" y="84">🏀</text><text x="320" y="112">🏀</text><text x="298" y="140">🏀</text><text x="330" y="76">🏀</text><text x="418" y="94">🏈</text><text x="446" y="70">🏈</text><text x="456" y="118">🏈</text><text x="422" y="146">🏈</text></g><line x1="378" y1="56" x2="392" y2="170" stroke="#2D8B55" stroke-width="3"/><text x="388" y="190" text-anchor="middle" fill="#6B645E" font-size="8.5">every ball on its own side — the loop moved the rope there, one nudge at a time</text></g></svg>
%%%

**What the picture gets right:** the loop is not doing anything mysterious — it is *sliding and spinning one line* until the two groups fall on opposite sides.

**Where it breaks down:** a rope is one line on a flat floor. A real network has many lines in many more directions than you can draw — but every one of them gets nudged by exactly the loop you just built.

#### Your turn — grab the rope
This one is pure play. There is nothing to get wrong here, so just drag things and enjoy it. **Predict first:** if you slide the line without spinning it, can you still split the two groups? Then find out.

%%% viz src=../../viz/neuron-boundary.html title="Drag the weights and move the line yourself" caption="You are now the training loop. Drag w₁, w₂ and b, and watch the line slide and spin across the dots. Try to split the two colours by hand — then think about how many tiny nudges the loop would need to find that same line on its own."
%%%

%%% insight
Here is the fun part, and it is genuinely true: the loop you built today is the *same* loop used to train the models that write essays and draw pictures. Nobody invented a fancier engine for them. They have more weights, more data, a smarter version of Step 4, and enormous computers — but the cycle is still forward → loss → backward → update. You have just learned the engine, at the size where you can watch every part of it move. Everything after this is *scale*.
%%%

Let's count that scale for a laugh, because the numbers are silly.

%%% demo id=scale label="predict, then run — how many nudges is 'a big training run'?"
predict: Your neuron takes 50 nudges in the experiment at the end of this lesson. A big model's training run takes... a thousand times more? A million times more? Guess an order of magnitude, then reveal.
code: mine  = 50                       # nudges in today's little experiment
code: theirs = 500_000                 # updates in a serious training run
code: print('my nudges   :', mine)
code: print('their nudges:', theirs)
code: print('same four steps, run', theirs // mine, 'times more often')
out: my nudges   : 50
out: their nudges: 500000
out: same four steps, run 10000 times more often
take: <b>About ten thousand times more laps — of the exact same four steps.</b> No new idea, just an enormous amount of patience (and electricity). Which means the thing you understood today is not a toy version of how learning works. It <i>is</i> how learning works.
%%%

Break over — and you've earned it. Now the one real thinking question of the day: how *long* should the loop run?

@@@ concept id=c8 tag="How long to practise" title="How long to practise — and why more can be worse" gotit="Got early/late stopping"
This section hides a surprising idea: *more practice can make a model worse.*

You already know both halves from school. Study **too little** and you walk into the exam unprepared. Study **too much of the wrong kind** — memorizing the practice sheet answers word-for-word — and you ace the practice sheet but freeze on the real exam, because it asks the same ideas with different numbers.

Good practice lives in between. The loop has exactly the same Goldilocks problem.

%%% svg
<svg viewBox="0 0 520 188" role="img" aria-label="A student studying for a test, shown three ways. Left: barely studied, sad face, labelled too little, underfit. Middle: studied just enough, happy face, labelled just right. Right: memorized every practice answer word for word, confused on the real exam, labelled too much, memorized."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Studying for a test: too little, just right, too much</text><rect x="16" y="34" width="150" height="132" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="91" y="70" text-anchor="middle" font-size="26">😟</text><text x="91" y="104" text-anchor="middle" fill="#C93B3B">too little</text><text x="91" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">barely practised</text><text x="91" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ bombs the test</text><text x="91" y="156" text-anchor="middle" fill="#C93B3B" font-size="9">(underfit)</text><rect x="185" y="34" width="150" height="132" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="260" y="70" text-anchor="middle" font-size="26">🙂</text><text x="260" y="104" text-anchor="middle" fill="#1a5c38">just right</text><text x="260" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">learned the pattern</text><text x="260" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ handles new questions</text><rect x="354" y="34" width="150" height="132" rx="6" fill="#FDF3D6" stroke="#C99A12"/><text x="429" y="70" text-anchor="middle" font-size="26">😵</text><text x="429" y="104" text-anchor="middle" fill="#9A7208">too much</text><text x="429" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">memorized the answers</text><text x="429" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ freezes on new ones</text><text x="429" y="156" text-anchor="middle" fill="#9A7208" font-size="9">(overfit)</text></g></svg>
%%%

**What the picture gets right:** too little and too much both hurt, in opposite ways — one from not enough practice, one from the wrong kind.

**Where it breaks down:** a student *knows* when they're only memorizing. A network doesn't — its practice score keeps looking great, so you need a separate check to catch it.

#### Quitting too early — the easy half
One line and you're done with this one. Cut the loop short and the loss is **still high** — the network simply hadn't finished walking downhill yet. **Cause: too few epochs** (**stopping too early**). **Remedy:** **train longer** and watch the curve until it flattens. That not-finished-yet state has a name: **underfit** (you will also see it written **underfitting**).

#### Drilling too long — the sneaky half (training too long)
Now the good one, and the most important idea of the day. Follow the beats — and notice that the second one is completely invisible if you only look at the training loss.

%%% steps
step: the honest phase — real pattern, still plenty to learn
why: both the practice score and the real-exam score improve together. Every lap is genuine progress.
step: the turn — the real signal runs out
why: there's nothing general left to learn, therefore the loop starts fitting the little accidents that exist *only* in your training rows. That is **overfitting**: the network **memorizes** instead of generalizing.
step: the trap — the training loss still looks great
why: memorizing lowers the training loss! Which means the number you were happily watching now *lies* to you. **Cause:** you **train too long** for the amount your data can teach.
step: the alarm — keep a slice of data back and score it too
why: on a [[validation set||a slice of data held back from training, used only to check whether the network handles data it has never practised on]] the network has never practised, so memorizing doesn't help there. Therefore the moment memorizing starts, the **validation loss rises** while training loss keeps falling. Two curves fork — and the fork is your alarm bell.
step: the fix — **early stopping**
why: [[early stopping||halt training when the validation loss stops improving and starts rising — right before memorizing takes over]] means you halt at the fork: **stop when validation stops improving**, and keep the version that scored best on **held-out** data rather than the one that memorized best.
%%%

%%% insight
Sit with this one, because it's the first time the number you're optimizing becomes your enemy. The loop is doing *exactly* what you asked — drive the training loss down — and that is precisely how it goes wrong. Almost all of "real" machine learning is this lesson on repeat: what you can measure is never quite what you actually want.
%%%

Enough words — let's catch the fork in the act. Six epochs of a real-ish run, both scores printed side by side.

%%% demo id=fork label="predict, then run — which epoch would you keep?"
predict: Training loss falls all six epochs. Validation loss does *not*. Which epoch has the BEST (lowest) validation score — the one early stopping would keep? Guess an epoch number, then reveal.
code: train = [0.90, 0.42, 0.21, 0.12, 0.07, 0.04]   # score on rows it practises on
code: val   = [0.95, 0.50, 0.31, 0.28, 0.33, 0.41]   # score on held-out rows
code: for e, (t, v) in enumerate(zip(train, val), 1): print(f'epoch {e}: train {t:.2f} | val {v:.2f}')
out: epoch 1: train 0.90 | val 0.95
out: epoch 2: train 0.42 | val 0.50
out: epoch 3: train 0.21 | val 0.31
out: epoch 4: train 0.12 | val 0.28
out: epoch 5: train 0.07 | val 0.33
out: epoch 6: train 0.04 | val 0.41
take: <b>Epoch 4 — that's the one to keep.</b> Validation loss turns upward at epoch 5 (0.28 → 0.33), so epoch 4 was its best score. After that, train keeps improving (0.12 → 0.04, looks fantastic!) while val gets steadily worse. Epochs 5 and 6 made the model <i>worse at its actual job</i>. Early stopping notices the rise at epoch 5, keeps the epoch-4 weights, and walks away.
%%%

Here's the same story as the picture you'll actually stare at.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two loss curves over epochs. The blue training-loss curve falls steadily and keeps dropping toward zero. The red validation-loss curve falls at first alongside it, reaches a lowest point, then turns back upward. A vertical dashed line at that lowest point is labelled stop here, early stopping. The region after it is shaded and labelled overfitting."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Training loss vs validation loss — stop at the fork</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text x="270" y="174" text-anchor="middle" fill="#6B645E" font-size="9">epochs →</text><rect x="300" y="26" width="200" height="124" fill="#FDF3D6" opacity="0.55"/><text x="400" y="42" text-anchor="middle" fill="#9A7208" font-size="9">overfitting zone</text><path d="M52 40 C 160 110, 320 138, 494 148" fill="none" stroke="#2A7B9B" stroke-width="2.2"/><text x="410" y="140" fill="#1F6280" font-size="9">training loss (keeps falling)</text><path d="M52 46 C 150 96, 280 108, 300 108 C 380 108, 440 90, 494 70" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text x="360" y="66" fill="#C93B3B" font-size="9">validation loss (turns up!)</text><line x1="300" y1="26" x2="300" y2="150" stroke="#2D8B55" stroke-width="1.8" stroke-dasharray="5,3"/><circle cx="300" cy="108" r="4" fill="#2D8B55"/><text x="300" y="166" text-anchor="middle" fill="#1a5c38" font-size="9">⬆ stop here (early stopping)</text></g></svg>
%%%

That's the Goldilocks rule in one image: run *enough* to converge, stop *before* the red line turns up.

🎉 **Victory lap.** You can now read a loss curve and name what's wrong with a run — too bold, too timid, no reset, fixed order, quit too early, drilled too long. **All six traps, with a one-line cure each**, and honestly that's most of what a working engineer does on a bad day. One honest truth left, and it's the interesting kind: the thing the loop *cannot* do, no matter how perfectly you tune it.

@@@ concept id=c9 tag="Where it hits a wall" title="The honest limit — when tuning cannot help" gotit="Got the limits"
This last idea is the one that separates someone who *uses* a tool from someone who *understands* it — and it's a satisfying one, because it hands you a diagnosis nobody taught you before. Try this: draw a perfect **circle using only a straight ruler**. Practise all year. Every stroke you make will be a straight line, so you'll never get a circle — not because you practised badly, but because the tool cannot make that shape. More repetition can't add a power a tool never had.

%%% svg
<svg viewBox="0 0 520 184" role="img" aria-label="On the left, a hand holding a straight ruler drawing straight strokes that try and fail to form a circle: the target circle is dashed and the ruler strokes form a jagged polygon that never matches it. On the right, a compass draws a clean circle in one sweep. Caption: no amount of practice makes a ruler draw a circle; you need a different tool."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A ruler can only make straight strokes — practice never gives it a curve</text><rect x="14" y="28" width="240" height="142" rx="6" fill="#FDECEC" stroke="#C93B3B"/><circle cx="130" cy="96" r="48" fill="none" stroke="#B8AEA2" stroke-width="1.4" stroke-dasharray="4,4"/><path d="M96 62 L164 62 L178 96 L164 130 L96 130 L82 96 Z" fill="none" stroke="#C93B3B" stroke-width="2"/><rect x="60" y="146" width="84" height="10" rx="2" fill="#E5DFD6" stroke="#9A938A"/><text x="102" y="155" text-anchor="middle" fill="#6B645E" font-size="7">ruler</text><text x="196" y="152" fill="#C93B3B" font-size="9">still not</text><text x="196" y="163" fill="#C93B3B" font-size="9">a circle ✕</text><rect x="272" y="28" width="234" height="142" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><circle cx="380" cy="96" r="48" fill="none" stroke="#2D8B55" stroke-width="2.4"/><path d="M380 96 L416 62" stroke="#9A938A" stroke-width="2"/><circle cx="380" cy="96" r="3" fill="#9A938A"/><text x="452" y="100" fill="#1a5c38" font-size="9">compass ✓</text><text x="452" y="112" fill="#6B645E" font-size="8">a different tool</text><text x="389" y="160" text-anchor="middle" fill="#1a5c38" font-size="9">add the right tool, not more practice</text></g></svg>
%%%

**What the picture gets right:** some jobs need a power the tool simply doesn't have, and repetition cannot conjure it.

**Where it breaks down:** you *can* get a circle — by adding a compass. Likewise a network gets curved boundaries by adding more neurons in a hidden layer. The wall belongs to the *single* neuron, not to networks in general.

#### The ruler wall — one neuron and [[XOR||"exclusive or": output 1 when the two inputs differ, 0 when they match — the four points can't be split by any single straight line]]
XOR is the classic. Four points; the two that *match* belong to one class, the two that *differ* to the other. A single neuron is a ruler: it can only cut the space with **one straight line** — the same rope you were dragging two sections ago.

%%% steps
step: the shape — XOR is **non-linearly separable**
why: no single straight line can put both matching points on one side and both differing points on the other. Try it in your head; then try it for real in the widget below.
step: the run — train it anyway, and the loss **plateaus above zero**
why: the loop still works perfectly! It walks downhill as far as downhill goes, then sits on a **plateau** — flat, but stuck at a bad score, because a **single neuron cannot** represent the answer.
step: the trap — reaching for more epochs
why: **more iterations never** fix this. A flat-but-high curve after a long, healthy run is your hint that the *model* is wrong, not the *knobs*.
step: the fix — change the tool, not the practice
why: add neurons (a hidden layer) and the pair of them carve two lines, which together solve XOR. That's the compass — and it's the whole reason the next module stacks layers.
%%%

%%% insight
This is the most useful diagnosis in the whole day: is my run failing because the **knobs** are wrong (too bold, too few epochs, forgot to shuffle) or because the **model** cannot express the answer? Everything before this section fixes knobs. Nothing before this section can fix a ruler being asked for a circle. Telling those two apart is the skill — and now you have it.
%%%

Don't take my word for it — go **try** to split XOR with one line. Drag the three sliders. Then tick "add a hidden layer" and watch the wall disappear.

%%% viz src=../../viz/xor-limit.html title="One line, four points, no solution" caption="Drag w₁, w₂ and b to move the single straight boundary. Predict first: can you find ANY setting that sorts all four points? Then tick 'add a hidden layer' — two lines, and suddenly it's solvable."
%%%

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A grid of four points showing the XOR pattern: bottom-left and top-right are one class, top-left and bottom-right are the other. A dashed straight line tries to separate them but always leaves one point on the wrong side. Beside it, a loss curve falls then flattens on a plateau well above zero."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A straight cut can't split XOR → the loss flattens above zero</text><line x1="90" y1="40" x2="90" y2="150" stroke="#B8AEA2"/><line x1="90" y1="150" x2="240" y2="150" stroke="#B8AEA2"/><circle cx="110" cy="130" r="9" fill="#2A7B9B"/><circle cx="220" cy="60" r="9" fill="#2A7B9B"/><circle cx="110" cy="60" r="9" fill="#C93B3B"/><circle cx="220" cy="130" r="9" fill="#C93B3B"/><line x1="80" y1="150" x2="240" y2="55" stroke="#9A938A" stroke-width="1.8" stroke-dasharray="5,3"/><text x="165" y="172" text-anchor="middle" fill="#6B645E" font-size="8.5">a straight cut always mis-sorts one dot</text><line x1="300" y1="40" x2="300" y2="150" stroke="#B8AEA2"/><line x1="300" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="285" y="94" fill="#6B645E" font-size="8" transform="rotate(-90 285 94)">loss</text><text x="400" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">loops →</text><path d="M306 50 C 340 96, 380 104, 494 104" fill="none" stroke="#C93B3B" stroke-width="2.2"/><line x1="300" y1="104" x2="494" y2="104" stroke="#C99A12" stroke-width="1" stroke-dasharray="3,3"/><text x="410" y="98" fill="#9A7208" font-size="8.5">stuck above 0 forever</text></g></svg>
%%%

#### One small footnote — the shallow dip
One more honest line, and it's a short one. Our valley picture assumed a single clean bowl; a lumpy landscape with several dips is called [[non-convex||a bumpy landscape with more than one dip, so the lowest nearby point may not be the lowest point overall]]. Because the gradient only feels the ground under your boots, the loop can roll into a small dip, find uphill in every direction, and stop there — a [[local minimum||a spot where every direction is uphill nearby, so the loop stops, even though a deeper valley exists elsewhere]], like a hiker who settles in a roadside ditch and announces "this is the bottom!" The curve flattens, so it *looks* converged: you did find the bottom of *this* dip, and that is **not the global best**.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A bumpy loss landscape with two dips. A ball has rolled into the shallower left dip and is sitting exactly at its lowest point, a local minimum, with uphill ground on both sides. The deeper right dip, the global minimum, is lower but unreached."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A bumpy landscape: the loop can stop in a shallow dip</text><path d="M24 44 C 60 44, 84 120, 120 120 C 156 120, 172 80, 208 80 C 244 80, 264 152, 300 152 C 336 152, 460 44, 500 44" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="120" cy="113" r="7" fill="#C99A12" stroke="#fff" stroke-width="1.5"/><text x="120" y="100" text-anchor="middle" fill="#9A7208" font-size="8.5">🛑 stuck here (local minimum)</text><circle cx="300" cy="145" r="6" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="316" y="148" fill="#1a5c38" font-size="9">← global minimum</text><text x="360" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">the real best (deeper), unreached</text><text x="120" y="140" text-anchor="middle" fill="#6B645E" font-size="8">uphill both ways → the loop stops</text></g></svg>
%%%

Cheerful footnote to the footnote: in the enormous networks behind real models these dips turn out to be mostly shallow and rarely a practical problem. It's an honest limit, not a reason to worry — and now you know it exists.

Now let's gather the whole day onto one page.

@@@ concept id=c10 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you just built the engine that makes every neural network learn, and you can now diagnose six ways it goes wrong from a single glance at a graph. Here's the whole day on one page: our **free-throw** drill, wearing its real training names. **Where the free-throw picture finally breaks down:** a player fixes one thing per shot; the loop nudges every knob at once, thousands of times — and that scale is the only reason it can learn things no person could.

The loop you built, in order:
- **Step 1 · Forward** — take the shot: run the input to a prediction.
- **Step 2 · Loss** — measure the miss: one number for how wrong the guess is.
- **Step 3 · Backward** — which way off: backprop hands every weight its gradient.
- **Step 4 · Update** — nudge the aim: `w ← w − lr × gradient`; the **minus** walks downhill.
- **Repeat** over **iterations** (one update, on one batch — today a batch of one example) and **epochs** (one full pass), watching the **loss curve** fall and flatten (**converge**).
- The **learning rate** is the multiplier on every nudge; the **perceptron** (1958) was the crude ancestor that corrected only on mistakes.

Here's the one picture to keep — the four steps, the dial, the curve, and where each trap strikes.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Recap diagram. Top: the four-step loop forward, loss, backward, update in a ring with a repeat arrow, and a learning-rate dial feeding the update step. Middle: a loss curve falling and flattening at convergence. Bottom: a row listing the six traps and their one-line cures."><g font-family="monospace" font-size="9"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">The whole day: the loop, the dial, the curve, the traps</text><rect x="16" y="26" width="86" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="59" y="43" text-anchor="middle" fill="#1F6280">1 forward</text><rect x="120" y="26" width="70" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="155" y="43" text-anchor="middle" fill="#C93B3B">2 loss</text><rect x="208" y="26" width="94" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text x="255" y="43" text-anchor="middle" fill="#5E5191">3 backward</text><rect x="320" y="26" width="82" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="361" y="43" text-anchor="middle" fill="#1a5c38">4 update</text><g stroke="#B8AEA2" stroke-width="1.6" fill="#B8AEA2"><path d="M102 39 l16 0"/><polygon points="118,39 111,35 111,43"/><path d="M190 39 l16 0"/><polygon points="206,39 199,35 199,43"/><path d="M302 39 l16 0"/><polygon points="318,39 311,35 311,43"/></g><path d="M361 52 C 361 70, 59 70, 59 54" fill="none" stroke="#C99A12" stroke-width="1.6" stroke-dasharray="3,2"/><polygon points="59,54 55,62 63,62" fill="#C99A12"/><text x="210" y="66" text-anchor="middle" fill="#9A7208">↻ repeat</text><rect x="418" y="26" width="86" height="26" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text x="461" y="40" text-anchor="middle" fill="#9A7208">learning rate</text><text x="461" y="49" text-anchor="middle" fill="#9A7208" font-size="7.5">(× every nudge)</text><line x1="46" y1="86" x2="46" y2="140" stroke="#B8AEA2"/><line x1="46" y1="140" x2="250" y2="140" stroke="#B8AEA2"/><path d="M50 92 C 90 132, 160 138, 246 139" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="150" y="100" fill="#1a5c38">loss curve → converges (flat)</text><text x="280" y="100" fill="#6B645E">read the curve:</text><text x="280" y="114" fill="#C93B3B">climbs every lap → lr too high</text><text x="280" y="128" fill="#2A7B9B">barely moves → lr too low</text><text x="280" y="142" fill="#9A7208">val loss turns up → overfit</text><line x1="16" y1="152" x2="504" y2="152" stroke="#E5DFD6"/><text x="16" y="168" fill="#C93B3B">lr high → lower it / schedule</text><text x="270" y="168" fill="#C93B3B">lr low → raise it</text><text x="16" y="182" fill="#C93B3B">no reset → zero the gradients</text><text x="270" y="182" fill="#C93B3B">fixed order → shuffle each epoch</text><text x="16" y="196" fill="#C93B3B">too few epochs → train longer</text><text x="270" y="196" fill="#C93B3B">too many → early stopping</text><text x="16" y="212" fill="#6B645E">limit: one neuron can't learn XOR · GD can stop in a local dip</text></g></svg>
%%%

#### One cheat-sheet · the words, the knobs, and the six traps
%%% table
:: Word / trap :: In plain English :: Remember
training loop :: forward → loss → backward → update, on repeat :: repeat until the loss flattens
forward pass :: run the input through the neuron to get a prediction :: the shot
loss :: one number saying how wrong the guess is :: lower is better, 0 is perfect
backward pass :: backpropagation hands every weight its gradient :: which way was I off?
update :: `w ← w − lr × gradient` :: the MINUS walks downhill
learning rate (lr) :: the multiplier on each nudge (the step size) :: too bold overshoots, too timid crawls
schedule (decay) :: shrink the learning rate over time :: bold steps early, gentle steps late (Day 8 goes deep)
iteration vs epoch :: one update vs one full pass over the data :: many iterations make one epoch
batch :: the handful of examples behind one update :: today, one example per update
loss curve · convergence :: the loss plotted per lap; flat = mostly finished :: the instrument panel
trap · lr too high :: a step lands farther out than it started → the loss climbs every lap, then NaN :: cure: lower lr / use a schedule
trap · lr too low :: loss barely moves :: cure: raise lr
trap · forgot to reset :: gradients pile up silently, so the step grows :: cure: zero the gradients each loop
trap · fixed data order :: not shuffling data means the updates get biased by the order :: cure: shuffle each epoch
trap · too few epochs :: underfit — loss still high :: cure: train longer, watch it flatten
trap · too many epochs :: overfit — memorizes; validation loss rises :: cure: early stopping on held-out data
limit · one neuron :: can't learn XOR; plateaus above zero :: more neurons, not more epochs
limit · local minimum :: settles in a shallow dip of a non-convex landscape :: converged ≠ globally best
%%%

%%% jargon
training loop | the four-step cycle that repeats until the loss stops falling
forward pass | run the input through the neuron to get a prediction
loss | one number scoring how wrong the prediction is
backward pass | backpropagation, which hands every weight its gradient
the update | `w ← w − lr × gradient` — the one step that changes the network
learning rate (lr) | how bold each nudge is; also called the step size
schedule / decay | a plan that shrinks the learning rate as training goes on
iteration | one trip round the loop — exactly one weight update
epoch | one full pass over all the training data
batch | the handful of examples behind a single update
loss curve | the loss plotted lap by lap — your instrument panel
convergence | the loss levels off and stops improving
underfit | stopped too soon; the loss is still high
overfitting | the network memorizes the training rows instead of the pattern
early stopping | halt at the point where validation loss stops improving
local minimum | a shallow dip the loop can settle in, on a non-convex landscape
%%%

That's the whole day. Tomorrow you keep the loop and swap Step 4 for something smarter than a plain nudge — **optimizers** like Momentum and Adam, which take this same downhill idea and make it faster and steadier.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What are the four steps of the training loop, in order? | a:2 | loss → forward → update → backward | update → backward → loss → forward | forward pass → loss → backward pass → update | backward → update → forward → loss | fb: One lap is forward (take the shot) → loss (measure the miss) → backward (which way off?) → update (nudge the aim), then repeat. It's the free-throw drill: try, measure, correct, again.
q: In the update rule w ← w − lr × gradient, what does the MINUS sign do, and what does lr control? | a:1 | Minus makes training faster; lr sets how many layers | The gradient points uphill, so minus steps the weight downhill (toward less loss); lr is the multiplier on that step — how bold each nudge is | Minus deletes bad weights; lr is the number of epochs | Minus is just notation with no effect; lr is the loss value | fb: The gradient points UPHILL (toward more loss), so subtracting it walks the weight downhill toward less loss. The learning rate multiplies that step: too bold and it overshoots until the loss diverges, too timid and training crawls.
q: Your loss climbs higher on every single lap and finally blows up to NaN. What's wrong, and what's the fix? | a:2 | The learning rate is too low; fix: lower it more | You have too few epochs; fix: shuffle the data | The learning rate is too high — each step overshoots by more than twice the distance left, so the weight lands farther out every lap and the loss climbs; fix: lower the learning rate (or use a schedule that shrinks it over time) | The gradients weren't reset; fix: train for more epochs | fb: A loss that grows lap after lap means each step carries the weight past the bottom by MORE than it started away from it, so the weight bounces across the bottom and lands farther out every time. (Crossing the bottom by a little is harmless — the ball zig-zags but still settles.) That's the classic too-high learning rate: lower it, or use a schedule (decay) so early steps are bold for speed and late steps gentle to settle.
q: Training loss keeps falling but validation loss starts rising. What is happening, and what's the remedy? | a:1 | The learning rate is too low; remedy: raise it | The network is overfitting — memorizing the training data instead of the pattern; remedy: early stopping (halt when validation loss stops improving) | The gradients are piling up; remedy: zero them each loop | The neuron can't represent the target; remedy: shuffle the data | fb: When the two curves fork — training loss down, validation loss up — the network has stopped learning the pattern and started memorizing the training examples. That's overfitting. Early stopping halts right at the fork, keeping the version that generalizes best.
%%%

@@@ produce id=produce tag="Produce" title="Run a real training loop and watch the loss fall" gotit="Done"
Time to run the engine yourself. You'll build the full four-step loop for one neuron and **watch** the loss drop, lap after lap, then poke the learning rate to *feel* the too-bold and too-timid settings. **Predict first:** with a sensible learning rate, will the loss fall smoothly toward zero, or jump around? Then set the learning rate way too high and predict again — fall, or blow up? Run it and **watch** which prediction was right. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-06-training-loop/experiment.py`. (1) Set up one neuron learning a simple target: `x = 2.0`, `target = 1.0`, start `w = 0.0` and `b = 0.0`, learning rate `lr = 0.01`. (2) Write the loop for 50 iterations: **forward** `pred = w*x + b`; **loss** `L = (pred - target)**2`; **backward** `grad_w = 2*(pred-target)*x` and `grad_b = 2*(pred-target)`; **update** `w = w - lr*grad_w` and `b = b - lr*grad_b`. Print the loss every 10 laps and **watch** it fall. (3) Now rerun with `lr = 1.5` and **observe** the prediction flip sign every lap while the loss explodes, then with `lr = 0.00005` and **observe** it barely move. Run with `python3 sessions/m02-the-neuron/day-06-training-loop/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 6 artifact.

Create sessions/m02-the-neuron/day-06-training-loop/experiment.py that, with a comment on each step:
1. Trains ONE neuron with a full training loop. Setup: x=2.0, target=1.0, w=0.0, b=0.0, lr=0.01.
2. Loops 50 iterations doing the four steps: forward pred=w*x+b; loss L=(pred-target)**2; backward grad_w=2*(pred-target)*x and grad_b=2*(pred-target); update w=w-lr*grad_w, b=b-lr*grad_b. Store the loss each loop and print it every 10 loops so we can watch it fall: it should start at 1.0 and slide down to roughly 0.12, 0.015, 0.0018, 0.0002 — a steep drop then a gentle glide, flattening near zero (convergence).
3. Re-runs the same loop twice more to show the learning-rate failures: once with lr=1.5 (print the prediction as well as the loss, and watch the PREDICTION flip sign every loop — 0, 15, -195, 2745 — while the LOSS climbs to an astronomical number: divergence) and once with lr=0.00005 (watch the loss go from 1.000 to only about 0.95 in 50 loops — crawling). Print the final loss for each lr and a one-line comment naming the failure.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the loop, learning)
- With `lr = 0.01` the loss starts at `1.000000` and the every-10-laps prints read roughly `0.1216 → 0.0148 → 0.0018 → 0.00022`, ending near `0.00003`. That's the healthy shape from concept 5: a **steep drop, a gentle glide, then flat** near zero — converging. **Watch** how the drop per lap shrinks as the gap shrinks, all on its own, with `lr` untouched.
- With `lr = 1.5` the **prediction flips sign every lap while the loss explodes** — the guess swings `0 → 15 → -195 → 2745 → …` past the target and back, and because the loss is that miss squared it only ever grows: by the last print it is an astronomical number (around `10¹¹²`), and if you keep looping it becomes `inf` and then `NaN`. That's the too-bold setting, live: each step lands farther from the bottom than it started. The cure is a smaller `lr`.
- With `lr = 0.00005` the loss goes from `1.000` to only about `0.952` in 50 laps — it **barely moves**. That's the too-timid setting; the cure is a bigger `lr`. Seeing all three back to back is exactly how engineers pick a learning rate by eye.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-06-training-loop/log.md`: (1) the four steps of the loop, in order; (2) what the learning rate multiplies, and what happens if it's too high or too low; (3) one trap from today (reset, shuffle, underfit, overfit, or a capability limit) and its one-line cure.
!!!

@@@ fin
