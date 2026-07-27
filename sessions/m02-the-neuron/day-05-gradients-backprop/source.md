---
quest_id: wf3-d02-backprop
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 5 — Gradients & Backpropagation"
module_label: "Module 2 · Train · Day 5"
title: "Gradients & Backpropagation"
subtitle: "Which Way Is Downhill?"
brand_sub: "Foundations · M2 Day 5"
spine: "hiker in fog"
nav_prev_href: "../day-04-loss/lesson.html"
nav_prev_label: "Loss Functions"
nav_next_href: "../day-06-training-loop/lesson.html"
nav_next_label: "The Training Loop"
fin_title: "Module 2 · Day 5 complete! 🏆"
fin_body: "Nice work — you've met the <b>gradient</b>, the arrow that points along the steepest slope <em>uphill</em> on the loss surface, which is exactly why we step the opposite way — downhill — and <b>backpropagation</b>, the chain-rule trick that hands every weight its own share of the blame. Now the network knows which way to step.<br>Next up: <b>The Training Loop</b> — where these steps repeat until the network learns."
notebook_yardstick: 00-neural-networks/fundamentals/07_backpropagation.ipynb
coverage_topics:
  - {topic: loss/error signal, keywords: [how wrong, one number, error signal]}
  - {topic: gradient, keywords: [gradient, which way is downhill, steepest]}
  - {topic: step opposite the gradient, keywords: [opposite, downhill, minus]}
  - {topic: forward pass, keywords: [forward pass, weighted sum, save the intermediate]}
  - {topic: computation graph, keywords: [small steps, chain of operations, computation graph]}
  - {topic: local derivative, keywords: [local slope, one small step, immediately downstream]}
  - {topic: chain rule, keywords: [chain rule, multiply the slopes, chain of effects]}
  - {topic: upstream downstream gradient, keywords: [incoming gradient, upstream, pass it further back]}
  - {topic: backward pass backprop, keywords: [backward pass, backpropagation, from the loss back]}
  - {topic: backprop reuse efficiency, keywords: [one sweep, reuse, instead of re-deriving]}
  - {topic: gradient of the weighted sum, keywords: [input value times, big inputs, weight's gradient]}
  - {topic: activation derivative in chain, keywords: [activation's slope multiplies, sigmoid slope, relu slope]}
  - {topic: gradient with respect to bias, keywords: [bias, passes straight through, slope 1]}
  - {topic: vanishing gradient, keywords: [vanishing gradient, multiply toward 0, saturating, signal fades]}
  - {topic: relu remedy for vanishing, keywords: [relu slope stays 1, switch to relu]}
  - {topic: dead relu zero gradient, keywords: [dead relu, permanently negative, slope 0, never learns]}
  - {topic: leaky relu remedy, keywords: [leaky relu, small negative slope, sensible init]}
  - {topic: exploding gradient, keywords: [exploding gradient, huge values, overshoot]}
  - {topic: gradient clipping remedy, keywords: [gradient clipping, clip, careful init]}
  - {topic: zero-init symmetry, keywords: [zero-initialization, identical gradients, symmetry, stay identical]}
  - {topic: small random init remedy, keywords: [small random, break the symmetry]}
  - {topic: numerical gradient check, keywords: [numerical gradient check, finite difference, nudge and measure]}
---

@@@ hero
@lede Imagine you're standing on a hillside in thick **fog**. You can't see the valley, but you want to walk down to the lowest point. So you do the one thing you still can: you feel the ground right under your boots, notice which way it tips *down*, and take a step that way. Then you feel again, and step again. That simple move — *feel the slope, step downhill* — is the whole secret behind how every neural network learns: the face-unlock model on your phone, the thing that picks your next recommended video, and the giant one behind ChatGPT. Yesterday you built the network's **score** — one number for how wrong its guess was. Today you learn the magic that reads that score and whispers to every single knob inside the network: *"nudge this way to make the score a little lower."* That whisper has a name — the **gradient** — and the clever trick that computes it for thousands of knobs at once is called **backpropagation**. By the end you'll know exactly which way is downhill.
@goal Together we'll turn a foggy hillside into a precise recipe. You'll meet the **gradient** (the arrow that points along the slope), watch the network **save its work** on the way forward so it can trace blame on the way back, learn the one multiplying trick — the **chain rule** — that hands every knob its fair share of the blame, and see the whole **backward pass** sweep through in a single pass. You'll get sliders to make the downhill signal fade, die, or explode with your own hand, a 30-second trick to prove your gradients are right, and a few sneaky traps with neat one-line cures. Every formula shows up in plain words first; any heavy math sits in a skippable box.

%%% warmup
q: Yesterday's loss — what is it, in one line? | a:1 | The number of layers in the network | One number saying how wrong a guess is; lower is better, 0 is perfect | The step that changes the weights | The speed the network trains at | concept: loss-one-number | fb: A loss turns "how far off was that guess?" into one honest number. Lower is better — today's gradient is what reads that number and finds which way to shrink it.
q: You're guessing house prices and a few freak mansions are wild outliers. Which loss keeps one crazy point from hijacking the score? | a:2 | MSE, because it squares each miss | Cross-entropy | MAE (or Huber), because it takes the plain miss size instead of squaring it | Counting mistakes | concept: mse-vs-mae-outlier | fb: MSE squares every miss, so one giant error explodes and drowns out the rest. MAE uses the plain miss size, so an outlier counts in proportion — Huber is the smooth middle ground.
q: Yesterday you learned a loss only judges — it does not fix. What actually changes the weights? | a:2 | The loss function itself | The activation function | A separate step called the optimizer, which reads the loss's slope | The number of training examples | concept: loss-scores-not-fix | fb: A loss only scores and hands over a slope; it never touches a weight. The optimizer reads that slope and steps the weights — and today's gradient is exactly the slope it reads.
%%%

@@@ concept id=c1 tag="The downhill arrow" title="The gradient — which way is downhill?" gotit="Got the gradient"
Let's start on that foggy hillside, before any math. You're a **hiker in fog**, and the ground under your boots is the network's mistake: high ground means a very wrong guess, low ground means a good one. Yesterday's [[loss||one number saying how wrong the network's guess was; lower is better, 0 is perfect]] is your *height* on this hill — one number for how wrong you are right now. You can't see the whole valley through the fog. But you *can* feel the tilt of the ground right where you stand, and step the way it tips down. Do that again and again and you'll reach the bottom — the lowest mistake.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A hiker standing in fog on a hillside that is high on the left and low on the right. Under the hiker's boots a small dashed patch of ground is visible. A red arrow points up the slope to the left, labelled the gradient, steepest way UP, more mistake. A green arrow points down the slope to the right, labelled our step, downhill, equals minus the gradient. Height is labelled how wrong you are."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A hiker in fog: feel the tilt underfoot, step downhill</text><path d="M20 62 C 140 62, 260 152, 500 162" fill="none" stroke="#B8AEA2" stroke-width="2"/><rect x="20" y="24" width="480" height="32" fill="#EDEAE4" opacity="0.85"/><text x="260" y="44" text-anchor="middle" fill="#9A938A" font-size="10">— fog · you can't see the whole hill —</text><text x="24" y="72" fill="#6B645E" font-size="10">height = how wrong</text><text x="148" y="74" font-size="20">🥾</text><circle cx="160" cy="92" r="26" fill="none" stroke="#7C6DAA" stroke-width="1.3" stroke-dasharray="3,3"/><path d="M144 88 l-40 -20" stroke="#C93B3B" stroke-width="2.5"/><polygon points="104,68 116,69 112,77" fill="#C93B3B"/><text x="72" y="98" text-anchor="middle" fill="#C93B3B" font-size="10">the gradient →</text><text x="72" y="110" text-anchor="middle" fill="#C93B3B" font-size="10">steepest way UP</text><text x="72" y="122" text-anchor="middle" fill="#C93B3B" font-size="10">(more mistake)</text><path d="M176 100 l56 24" stroke="#2D8B55" stroke-width="2.5"/><polygon points="232,124 220,124 224,116" fill="#2D8B55"/><text x="240" y="118" fill="#2D8B55" font-size="10">our step → DOWNHILL</text><text x="240" y="131" fill="#2D8B55" font-size="10">= minus the gradient</text><text x="200" y="152" text-anchor="middle" fill="#7C6DAA" font-size="9">the patch under your boots</text><text x="60" y="150" text-anchor="middle" fill="#6B645E" font-size="10">high = very wrong</text><text x="430" y="184" text-anchor="middle" fill="#2D8B55" font-size="10">low = good guess</text></g></svg>
%%%

**What the hiker-in-fog picture gets right:** you only ever need the *local* tilt — the ground right under your boots — to know which way to step, and you reach the bottom one small step at a time. **Where it breaks down:** a real hill has just two directions to lean (north–south, east–west), but a real network has *thousands* of knobs, so its "hill" tips in thousands of directions at once.

#### From boots to knobs
Let's walk the hiker into the network, one small idea at a time. Each knob inside the neuron is a [[weight||a number the neuron multiplies its input by; learning means finding good values for the weights]] (or a bias). Here's the move: instead of feeling the ground with *boots*, we ask the same tilt question about *each knob*.

%%% steps
step: the test lean — nudge ONE knob up by a hair
why: that's the hiker leaning one foot one way; a tiny test move, not a real step yet
step: the reading — did the mistake go up or down, a lot or a little?
why: that answer is this knob's own slope, and it usefully splits into two parts
step: the two parts — the SIGN says which way, the SIZE says how much this knob matters right now
why: therefore a steep knob deserves a big correction, and a nearly-flat one barely matters yet
step: the whole arrow — collect that reading for every knob at once
why: that stack of slopes IS the gradient, and it points the steepest way UPHILL (more mistake)
%%%

So the [[gradient||the arrow that says, for every weight at once, which way to nudge it to change the loss the fastest — and how steeply]] is not mysterious — it's that four-rung ladder, run for every knob. Which means it answers today's title question directly: *which way is downhill?* It's the way the arrow **isn't** pointing.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A single dip-shaped loss curve. A dot sits on the right side. A red arrow points up-and-right along the slope labelled gradient points uphill, more error. A green arrow points the opposite way, down toward the bottom of the dip, labelled we step the opposite way, downhill."><g font-family="monospace" font-size="11"><path d="M40 30 C 160 150, 360 150, 480 30" fill="none" stroke="#B8AEA2" stroke-width="2.5"/><circle cx="410" cy="92" r="6" fill="#C99A12" stroke="#fff" stroke-width="2"/><path d="M416 88 l38 -28" stroke="#C93B3B" stroke-width="2.5"/><polygon points="454,60 443,61 448,70" fill="#C93B3B"/><text x="440" y="52" fill="#C93B3B" text-anchor="middle" font-size="10">gradient → uphill (more error)</text><path d="M404 96 l-40 26" stroke="#2D8B55" stroke-width="2.5"/><polygon points="364,122 375,120 370,111" fill="#2D8B55"/><text x="300" y="140" fill="#2D8B55" font-size="10">we step the OPPOSITE way → downhill</text><text x="258" y="120" text-anchor="middle" fill="#6B645E" font-size="10">the bottom = smallest mistake</text><text x="40" y="24" fill="#6B645E" font-size="10">loss (height) ↑</text></g></svg>
%%%

The one line to remember: **the gradient points uphill, and to learn we step the opposite way — downhill.** That's why the update you meet tomorrow carries a **minus** sign: subtract the gradient and you walk *down* the hill instead of up it.

%%% insight
Notice what we never needed: a map of the valley. The hiker never sees the bottom — and neither does ChatGPT while it trains. Every network you've heard of learned by feeling the ground under its feet, thousands of knobs at a time, and stepping the opposite way. That's the whole idea, and you already have it.
%%%

So far: we know what we *want* — one slope per knob. The rest of today is one question: *how does the network actually feel that slope for every knob, cheaply?* That's backpropagation, and it starts with something the neuron did yesterday.

@@@ concept id=c2 tag="Save your work" title="The forward pass — and why it saves its notes" gotit="Got the forward pass"
Before a network can trace its mistake backward, it has to make a guess *forward* first — and, quietly, keep a few notes along the way. Think about **baking a cake**. You add flour, then sugar, then eggs, then bake — one step feeds the next. If the cake comes out wrong, you don't throw out the kitchen; you look back at your notes — *"how much sugar did I add? how long did it bake?"* — to figure out which step to change. If you wrote nothing down, you'd be guessing blind.

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A baking assembly line: flour then sugar then eggs then bake then taste. Under each step a small sticky note records what was used. A caption says keep the notes so if the cake is wrong you know which step to fix."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Baking forward — and keeping a note at every step</text><rect x="24" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="61" y="59" text-anchor="middle" fill="#9A7208">flour</text><text x="104" y="59" fill="#B8AEA2">→</text><rect x="120" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="157" y="59" text-anchor="middle" fill="#9A7208">sugar</text><text x="200" y="59" fill="#B8AEA2">→</text><rect x="216" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="253" y="59" text-anchor="middle" fill="#9A7208">eggs</text><text x="296" y="59" fill="#B8AEA2">→</text><rect x="312" y="40" width="74" height="30" rx="5" fill="#F3ECDB" stroke="#C99A12"/><text x="349" y="59" text-anchor="middle" fill="#9A7208">bake</text><text x="392" y="59" fill="#B8AEA2">→</text><rect x="408" y="40" width="88" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="452" y="59" text-anchor="middle" fill="#1a5c38">taste 😋</text><g><rect x="30" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text x="61" y="107" text-anchor="middle" fill="#5E5191" font-size="9">2 cups</text><rect x="126" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text x="157" y="107" text-anchor="middle" fill="#5E5191" font-size="9">1 cup</text><rect x="222" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text x="253" y="107" text-anchor="middle" fill="#5E5191" font-size="9">3 eggs</text><rect x="318" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text x="349" y="107" text-anchor="middle" fill="#5E5191" font-size="9">30 min</text></g><text x="260" y="140" text-anchor="middle" fill="#6B645E">the sticky notes = the saved values — so a wrong cake tells you which step to fix</text></g></svg>
%%%

**What the baking picture gets right:** the steps run in order, each feeding the next, and *keeping notes* is what lets you trace a bad result back to its cause. **Where it breaks down:** a baker keeps notes for a person to read later; the network keeps its notes for *itself* — it re-uses them a fraction of a second later, on the way back.

#### The recipe, one step at a time
The [[forward pass||running the input through the neuron to get a guess — and saving the in-between values for the trip back]] is just the neuron making its guess, the same recipe you built earlier this module. Here it is as a recipe card, exactly like the cake:

%%% steps
step: the mix — multiply each input by its weight, add the bias
why: this in-between number is the [[pre-activation||the weighted sum plus bias, before the activation bends it; often written z]] `z` — the weighted sum, our "batter"
step: the bake — bend `z` with the activation (sigmoid or ReLU from Day 2) to get the output `a`
why: that bend is what makes a network more than a straight line; `a` is the neuron's guess
step: the taste — compare `a` to the true answer to get today's mistake number
why: that's yesterday's loss, the height of our hill
step: the notes — save the intermediate values `x`, `z` and `a` as each one is made
why: these are the sticky notes on the cake tin, and the backward trip re-uses every single one
%%%

That last rung is the quiet, crucial one. Save the intermediate values now, and the trip backward becomes almost free. Skip them, and you'd have to re-bake the whole cake for every question you ask.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="The neuron's forward pass as a conveyor: input x flows into weighted-sum-plus-bias giving z, then activation giving a, then loss. Under z and a, saved-note tags show the values are stored for the backward trip."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Forward pass → guess, saving z and a as it goes</text><rect x="16" y="42" width="72" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text x="52" y="61" text-anchor="middle" fill="#1F6280">input x</text><text x="94" y="61" fill="#B8AEA2">→</text><rect x="112" y="42" width="118" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="171" y="57" text-anchor="middle" fill="#9A7208" font-size="9.5">weighted sum + bias</text><text x="171" y="68" text-anchor="middle" fill="#9A7208" font-size="9">= z</text><text x="236" y="61" fill="#B8AEA2">→</text><rect x="254" y="42" width="96" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text x="302" y="57" text-anchor="middle" fill="#5E5191" font-size="9.5">activation</text><text x="302" y="68" text-anchor="middle" fill="#5E5191" font-size="9">= a (guess)</text><text x="356" y="61" fill="#B8AEA2">→</text><rect x="374" y="42" width="126" height="30" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="437" y="61" text-anchor="middle" fill="#C93B3B" font-size="9.5">loss (how wrong)</text><rect x="130" y="96" width="82" height="24" rx="3" fill="#F3ECDB" stroke="#C99A12" stroke-dasharray="3,2"/><text x="171" y="112" text-anchor="middle" fill="#9A7208" font-size="9">📌 saved z</text><rect x="266" y="96" width="72" height="24" rx="3" fill="#EFEAF7" stroke="#B3A6D4" stroke-dasharray="3,2"/><text x="302" y="112" text-anchor="middle" fill="#5E5191" font-size="9">📌 saved a</text><text x="260" y="146" text-anchor="middle" fill="#6B645E">these saved notes get re-used on the trip back — nothing recomputed</text></g></svg>
%%%

%%% insight
Here's why this bites in real life: those saved notes are also the reason training a big model eats so much memory. The network is holding on to every in-between value it made, for every example in the batch, just so the backward trip can read them. When you hear "we ran out of GPU memory," this is often the culprit — sticky notes, by the billion.
%%%

There's a second gift hiding in that recipe card. Written as a *chain of operations* — multiply, add, bend, score — the neuron becomes a [[computation graph||a picture of the math as a chain of tiny operations, so each little step has a clear, simple slope of its own]]: an assembly line where each box does one tiny thing.

Why care? Because each of those **small steps** has a *simple* slope of its own — and stitching simple slopes together is the whole backward trick. On to the stitch.

@@@ concept id=c3 tag="Chain of effects" title="Local slopes and the chain rule" gotit="Got the chain rule"
Here's the one idea that makes the whole thing tick, and you already understand it from planning a **road trip**. Push the gas pedal a little harder → your *speed* goes up a little. More speed → you cover more *distance* each hour. More distance → you pay more *money* at the pump. Now ask: *"how much does one extra push on the pedal cost me?"* You don't need a mega-formula. You follow the **chain of effects** and **multiply** the little hops.

%%% svg
<svg viewBox="0 0 520 148" role="img" aria-label="A road-trip chain with three hops: pedal affects speed, speed affects distance, distance affects money. Each of the three arrows carries one small local effect, written as times 2, times 1.5 and times 0.5. A caption says multiply all three little effects to get how the pedal affects the money."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Road trip: three little effects, multiplied</text><rect x="30" y="48" width="84" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text x="72" y="67" text-anchor="middle" fill="#1F6280">🦶 pedal</text><rect x="166" y="48" width="84" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="208" y="67" text-anchor="middle" fill="#9A7208">speed</text><rect x="302" y="48" width="92" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="348" y="67" text-anchor="middle" fill="#9A7208">distance</text><rect x="446" y="48" width="64" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="478" y="67" text-anchor="middle" fill="#1a5c38">💰 money</text><g stroke="#C93B3B" stroke-width="2" fill="#C93B3B"><path d="M116 63 l40 0"/><polygon points="164,63 156,59 156,67"/><path d="M252 63 l40 0"/><polygon points="300,63 292,59 292,67"/><path d="M396 63 l40 0"/><polygon points="444,63 436,59 436,67"/></g><text x="140" y="94" text-anchor="middle" fill="#C93B3B">× 2</text><text x="140" y="106" text-anchor="middle" fill="#6B645E" font-size="9">mph per push</text><text x="276" y="94" text-anchor="middle" fill="#C93B3B">× 1.5</text><text x="276" y="106" text-anchor="middle" fill="#6B645E" font-size="9">miles per mph</text><text x="420" y="94" text-anchor="middle" fill="#C93B3B">× 0.5</text><text x="420" y="106" text-anchor="middle" fill="#6B645E" font-size="9">dollars per mile</text><text x="260" y="132" text-anchor="middle" fill="#6B645E">three small "local" effects, multiplied → the pedal's effect on money</text></g></svg>
%%%

**What the road-trip picture gets right:** you never need one giant formula for "pedal → money." You break it into easy one-hop effects and multiply. **Where it breaks down:** on a road trip the effects mostly go one way (more pedal → more money), but inside a network a slope can be negative — nudging a weight *up* can push the loss *down*. The multiplying still works; the sign just tells you which way to lean.

#### Two words, then we drive
Each little arrow above is a [[local derivative||the slope of just one tiny step — how much its OWN output moves when its OWN input moves a hair, ignoring the rest of the chain]] — a fancy name for one easy question: *"if I nudge this box's input a hair, how much does its output move?"* Just that **one small step** — this box, and nothing else. The rest of the chain can wait its turn.

The rule for gluing those local slopes together is the [[chain rule||to get how the far-away loss depends on an early quantity, multiply the local slopes all along the path between them]]: **multiply the slopes** all along the path. That's the whole rule. Let's drive it with the three numbers from the picture.

%%% steps
step: pedal → speed: one more push adds `2` mph — local slope = `2`
why: the first hop only knows about the pedal and the speedometer; nothing else exists yet
step: speed → distance: each mph adds `1.5` miles — running product `2 × 1.5 = 3`
why: therefore one push now buys 3 extra miles; we are carrying the effect forward by multiplying
step: distance → money: each mile costs `$0.5` at the pump — running product `3 × 0.5 = 1.5`
why: that's the answer — one extra push on the pedal costs about **$1.50**
step: notice what we never wrote: a single "pedal → money" formula
why: three easy one-hop slopes, multiplied. That's why this scales to a network of thousands of steps
%%%

Watch that running product grow, hop by hop:

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A running-product ladder for the chain rule. Three multiply steps: after pedal to speed the product is 2, after speed to distance it is 3, after distance to money it is 1.5. Bars show the running product changing at each hop."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Running product: multiply one local slope at a time</text><line x1="40" y1="118" x2="500" y2="118" stroke="#E5DFD6" stroke-width="1.5"/><rect x="70" y="58" width="66" height="60" fill="#2A7B9B"/><text x="103" y="52" text-anchor="middle" fill="#1F6280">2.0</text><text x="103" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 2 (pedal→speed)</text><rect x="230" y="28" width="66" height="90" fill="#C99A12"/><text x="263" y="22" text-anchor="middle" fill="#9A7208">3.0</text><text x="263" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 1.5 (speed→dist)</text><rect x="390" y="73" width="66" height="45" fill="#2D8B55"/><text x="423" y="67" text-anchor="middle" fill="#1a5c38">1.5</text><text x="423" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 0.5 (dist→money)</text><text x="163" y="90" text-anchor="middle" fill="#C93B3B" font-size="13">×</text><text x="343" y="90" text-anchor="middle" fill="#C93B3B" font-size="13">×</text></g></svg>
%%%

%%% demo id=chain label="run it — multiply the little slopes"
predict: before you look — do the three slopes 2, 1.5 and 0.5 get ADDED into 4, or MULTIPLIED into 1.5? Guess, then reveal.
code: pedal_to_speed=2; speed_to_dist=1.5; dist_to_money=0.5; pedal_to_speed*speed_to_dist*dist_to_money
out: 1.5
take: <b>Chain rule = multiply the local slopes.</b> 2 × 1.5 × 0.5 = 1.5 — one extra push on the pedal costs about $1.50. Break a scary "how does the start affect the end?" question into easy one-hop slopes and multiply. That single move is the engine of backprop.
%%%

#### One last word, and it's the one that flows
As we work back along the chain, the slope we've built up *so far* — the running product arriving at a box from the loss side — is called the [[upstream gradient||the gradient that has already arrived at a step from the loss side; multiply it by the step's own local slope to pass it further back]]. It was handed over by the box **immediately downstream** of you — the next box along toward the loss. (When engineers say **error signal**, this travelling number is what they mean: not the loss itself, but the blame flowing backward.) The rhythm at every box is a two-beat move:

%%% steps
step: take the incoming gradient (what arrived from the loss side)
why: you don't recompute anything — the box downstream already did that work for you
step: multiply it by THIS box's own local slope, then pass it further back
why: the box you hand it to now sees YOUR number as its incoming gradient — that's how the slope flows
%%%

%%% insight
Stop and appreciate this, because it's the reason deep networks are possible at all: each box only needs to know two things — what arrived from downstream, and its own one-step slope. It never needs to know the whole network. That locality is why the same trick works for 2 layers or 200.
%%%

If the road-trip multiply felt fiddly, that's normal — it is the one genuinely new idea today, and everything else is this idea reused. You just met the engine of backprop. Next we start it up.

@@@ concept id=c4 tag="Trace the blame" title="The backward pass — one sweep, all the blame" gotit="Got backprop"
Now the trick clicks into place. Imagine a student who just got a test back with a low score. The lazy way to improve: re-take the *entire* test after changing one study habit, then re-take it again after changing the next — a fresh full test per habit. Exhausting, and most of the work is repeated. The smart student does something better: she starts from the wrong answers and traces the blame *backward* — *"this mistake came from that step, which came from this habit"* — giving each habit its fair share in **one careful pass** over the graded test.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A student tracing blame backward from a low test score to specific study habits, in one pass. On the left a slow way redoes the whole test once per habit; on the right the smart way traces blame from the score back through each step to each habit in a single sweep."><g font-family="monospace" font-size="10"><text x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">Slow: redo the whole test per habit</text><rect x="30" y="30" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="60" y="45" text-anchor="middle" fill="#C93B3B">test #1</text><rect x="30" y="58" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="60" y="73" text-anchor="middle" fill="#C93B3B">test #2</text><rect x="30" y="86" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="60" y="101" text-anchor="middle" fill="#C93B3B">test #3</text><text x="130" y="73" fill="#6B645E">…once per habit 😩</text><line x1="258" y1="24" x2="258" y2="150" stroke="#E5DFD6"/><text x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">Smart: trace blame back ONCE</text><rect x="452" y="44" width="56" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="480" y="61" text-anchor="middle" fill="#C93B3B">score</text><rect x="356" y="44" width="56" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text x="384" y="61" text-anchor="middle" fill="#5E5191">step</text><rect x="278" y="44" width="52" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="304" y="61" text-anchor="middle" fill="#1F6280">habit</text><g stroke="#2D8B55" stroke-width="2.5" fill="#2D8B55"><path d="M452 57 l-40 0"/><polygon points="412,57 420,53 420,61"/><path d="M356 57 l-26 0"/><polygon points="330,57 338,53 338,61"/></g><text x="390" y="96" text-anchor="middle" fill="#2D8B55">one backward sweep 🎯</text><text x="390" y="112" text-anchor="middle" fill="#6B645E">every habit gets its blame at once</text></g></svg>
%%%

**What the student picture gets right:** blame flows backward from the final score to each cause, and one traced pass beats redoing everything once per cause. **Where it breaks down:** a student judges her habits with fuzzy hunches; the network's blame is exact — each step's share is precisely the incoming gradient times that step's local slope, the chain rule you just met.

#### The sweep, one move at a time
The [[backward pass||backpropagation: starting at the loss, walk backward through the saved steps, multiplying local slopes, to get every weight's gradient in one sweep]] — **backpropagation** — is that smart student, made exact. Watch her walk backward through our cake-recipe graph:

%%% steps
step: start at the score — begin from the loss back, where the mistake is measured
why: this is the graded test in her hand; the very first slope is "how does the guess change the loss?"
step: one hop back — multiply by the activation's own local slope
why: same two-beat move as the road trip: incoming gradient × this box's local slope
step: read the sticky note — the activation's slope needs the `z` the forward pass saved
why: that's why we kept notes; nothing gets recomputed, it's just looked up
step: arrive at a knob — the number sitting there IS that knob's gradient
why: its share of the blame, exact, no debate
step: keep going — hand the number further back and repeat
why: one walk hands EVERY weight its gradient, because they all share the same running product
%%%

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="The neuron's graph with green gradient arrows sweeping backward from the loss to the activation, then to the weighted sum, then to the weights. Each arrow carries a numbered marker, and a legend below names them: one, multiply by the loss slope which uses the saved a; two, multiply by the activation's own slope which uses the saved z; three, multiply by the input x, which is each weight's gradient."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Backward pass: one sweep from loss → every weight's gradient</text><rect x="16" y="42" width="64" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text x="48" y="61" text-anchor="middle" fill="#1F6280">weights</text><rect x="110" y="42" width="108" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="164" y="61" text-anchor="middle" fill="#9A7208" font-size="9">weighted sum → z</text><rect x="248" y="42" width="90" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text x="293" y="61" text-anchor="middle" fill="#5E5191" font-size="9">activation → a</text><rect x="368" y="42" width="132" height="30" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="434" y="61" text-anchor="middle" fill="#C93B3B" font-size="9">loss (start here)</text><g stroke="#2D8B55" stroke-width="2.5" fill="#2D8B55"><path d="M368 90 l-22 0"/><polygon points="338,90 346,86 346,94"/><path d="M248 90 l-22 0"/><polygon points="218,90 226,86 226,94"/><path d="M110 90 l-22 0"/><polygon points="80,90 88,86 88,94"/></g><g fill="#2D8B55"><circle cx="353" cy="76" r="8"/><circle cx="233" cy="76" r="8"/><circle cx="95" cy="76" r="8"/></g><g fill="#ffffff" text-anchor="middle" font-size="10"><text x="353" y="80">1</text><text x="233" y="80">2</text><text x="95" y="80">3</text></g><text x="44" y="110" text-anchor="middle" fill="#1a5c38" font-size="9">gradient! 🎯</text><line x1="30" y1="120" x2="490" y2="120" stroke="#E5DFD6"/><g fill="#2D8B55"><circle cx="44" cy="136" r="7"/><circle cx="44" cy="156" r="7"/><circle cx="44" cy="176" r="7"/></g><g fill="#ffffff" text-anchor="middle" font-size="9"><text x="44" y="139">1</text><text x="44" y="159">2</text><text x="44" y="179">3</text></g><g fill="#5A544E" font-size="9.5"><text x="58" y="139">× the loss slope — uses the saved a 📌</text><text x="58" y="159">× the activation's own slope — uses the saved z 📌</text><text x="58" y="179">× the input x — and that IS each weight's gradient</text></g></g></svg>
%%%

So far: forward to make a guess and leave notes, backward to spread the blame. Two sweeps, and the network knows how to improve.

#### Why this is such a big deal
Here's the part worth getting excited about. The slow way is re-deriving each weight's effect from scratch — one full run of the whole network per weight — which repeats almost all of the same work every single time. A real network has *millions* of weights, so that road is hopeless.

%%% insight
Backprop's win is [[reuse||computing every weight's gradient in a single backward sweep that shares its work, instead of redoing the chain separately for each weight]]: **one sweep** gives you every weight's gradient — **instead of re-deriving** each weight's effect separately — because they all ride the same running product. This single fact is why training a large model costs roughly *two* passes instead of a million. No backprop, no ChatGPT — the arithmetic simply would not fit in a human lifetime.
%%%

You've now got the engine and the sweep. Next: what blame actually lands on each part of the neuron.

@@@ concept id=c5 tag="Each part's share" title="The gradient at each knob — three rules, plus one for the road" gotit="Got the rules"
Time to open the neuron and see who gets blamed for what. Picture a **group project** where everyone shares one grade. It only feels fair if your share matches how much you actually *did*: the teammate who wrote most of the report carries most of the responsibility, and the one who just showed up gets a light touch. The neuron splits its blame the exact same way — and it comes down to a few tiny rules, one per part.

%%% svg
<svg viewBox="0 0 520 158" role="img" aria-label="A group project sharing one grade. Three teammates get different-sized slices of blame: the big contributor gets a big slice, the medium contributor a medium slice, the tag-along a tiny slice. Caption: fair blame matches how much each part actually did."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Group project: your share of blame = how much you did</text><rect x="40" y="40" width="120" height="80" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="34" text-anchor="middle" fill="#6B645E" font-size="9">did a lot 📝</text><text x="100" y="86" text-anchor="middle" fill="#C93B3B" font-size="13">big blame</text><rect x="200" y="60" width="120" height="60" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="260" y="54" text-anchor="middle" fill="#6B645E" font-size="9">did some</text><text x="260" y="96" text-anchor="middle" fill="#9A7208" font-size="12">medium</text><rect x="360" y="98" width="120" height="22" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="420" y="92" text-anchor="middle" fill="#6B645E" font-size="9">just showed up</text><text x="420" y="113" text-anchor="middle" fill="#1a5c38" font-size="10">tiny blame</text><text x="260" y="142" text-anchor="middle" fill="#6B645E">the neuron splits its blame the same fair way — a few simple rules</text></g></svg>
%%%

**What the group-project picture gets right:** blame is shared in proportion to contribution — do more, own more of the result. **Where it breaks down:** teammates argue about who did what; the neuron's split is exact and instant, just the chain-rule multiply.

Before the rules, one behavior to feel in your own hands. When the gradient reaches the activation it gets *multiplied* by the activation's slope — and that slope changes a lot with the input. **Predict first:** where do you think sigmoid's slope is biggest — out in the tails, or in the middle? Now drag `z` and find out:

%%% svg
<svg id="sl-svg" viewBox="0 0 520 214" role="img" aria-label="Interactive activation slope. Pick sigmoid, tanh, or ReLU and drag the input z. A dot rides the activation curve and a short tangent line shows the slope there — steep in the middle, flat in the tails. A horizontal reference line marks where the output is zero, and it moves to the right place for each activation, so tanh's negative half is visible below it. The slope is the exact number that multiplies the gradient passing back through."><g font-family="monospace" font-size="11"><rect x="60" y="30" width="400" height="160" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line id="sl-axis" x1="60" y1="190" x2="460" y2="190" stroke="#D8D2C8" stroke-width="1" stroke-dasharray="4,3"/><text id="sl-axlab" x="64" y="185" fill="#9A938A" font-size="9">output = 0</text><line x1="260" y1="30" x2="260" y2="190" stroke="#EFE9DF" stroke-width="1"/><path id="sl-curve" d="M60 190 L260 110 L460 40" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><line id="sl-tan" x1="234" y1="150" x2="286" y2="70" stroke="#C99A12" stroke-width="3"/><circle id="sl-dot" cx="260" cy="115" r="6" fill="#C99A12" stroke="#fff" stroke-width="2"/><text x="466" y="40" fill="#9A938A" font-size="9">high</text><text x="466" y="188" fill="#9A938A" font-size="9">low</text><text x="264" y="204" fill="#9A938A" font-size="9">z = 0</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px">
<span>activation:</span>
<button id="sl-sig" type="button" style="cursor:pointer;border:2px solid #7C6DAA;background:#EDE9F8;color:#5E5191;border-radius:5px;padding:2px 8px;font-family:monospace">sigmoid</button>
<button id="sl-tanh" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">tanh</button>
<button id="sl-relu" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">ReLU</button>
</div>
<label>drag z <input id="sl-z" type="range" min="-6" max="6" step="0.2" value="0" style="width:60%;accent-color:#C99A12;vertical-align:middle"></label>
<div id="sl-out" style="margin-top:6px;color:#2C2A28">at z = 0.0, sigmoid slope = <b>0.25</b> → a gradient of 1.00 passes back as <b>0.25</b> <span style="color:#C99A12">(shrinks to a quarter)</span></div>
<div style="margin-top:4px;color:#6B645E;font-size:.92em">Switch to <b>tanh</b> and watch the dashed "output = 0" line jump to the middle of the box — tanh is the one activation here that can go <i>negative</i>, and its curve dips below that line. Note on ReLU at exactly <code>z = 0</code>: the curve has a sharp corner there, so it has no single slope — the slope is 0 just to the left and 1 just to the right. Everyone simply picks one by convention (we show 1); nudge <code>z</code> either side and you'll see the honest values.</div>
</div>
<script>(function(){
  var zs=document.getElementById('sl-z');if(!zs)return;
  var curve=document.getElementById('sl-curve'),dot=document.getElementById('sl-dot'),tan=document.getElementById('sl-tan'),
      out=document.getElementById('sl-out'),axis=document.getElementById('sl-axis'),axlab=document.getElementById('sl-axlab'),
      bSig=document.getElementById('sl-sig'),bTanh=document.getElementById('sl-tanh'),bRelu=document.getElementById('sl-relu');
  var act='sigmoid';
  var X0=60,X1=460,ZLO=-6,ZHI=6;
  function xpx(z){return X0+(z-ZLO)/(ZHI-ZLO)*(X1-X0);}
  var xscale=(X1-X0)/(ZHI-ZLO);
  var defs={
    sigmoid:{f:function(z){return 1/(1+Math.exp(-z));},d:function(z){var s=1/(1+Math.exp(-z));return s*(1-s);},ypx:function(f){return 190-f*150;},yscale:150},
    tanh:{f:function(z){return Math.tanh(z);},d:function(z){var t=Math.tanh(z);return 1-t*t;},ypx:function(f){return 110-f*70;},yscale:70},
    relu:{f:function(z){return Math.max(0,z);},d:function(z){return z>=0?1:0;},ypx:function(f){return 190-f/6*150;},yscale:25}
  };
  function setBtns(){[[bSig,'sigmoid'],[bTanh,'tanh'],[bRelu,'relu']].forEach(function(p){
    var sel=(p[1]===act);
    p[0].style.borderColor=sel?'#7C6DAA':'#E5DFD6';p[0].style.background=sel?'#EDE9F8':'#FDF9F3';p[0].style.color=sel?'#5E5191':'#6B645E';});}
  function paint(){
    var D=defs[act],z=+zs.value;
    // put the "output = 0" reference line where THIS activation's zero actually is
    var zeroY=D.ypx(0);
    axis.setAttribute('y1',zeroY.toFixed(1));axis.setAttribute('y2',zeroY.toFixed(1));
    axlab.setAttribute('y',(zeroY-5).toFixed(1));
    var d='';for(var i=0;i<=80;i++){var zz=ZLO+(ZHI-ZLO)*i/80;d+=(i?'L':'M')+xpx(zz).toFixed(1)+' '+D.ypx(D.f(zz)).toFixed(1)+' ';}
    curve.setAttribute('d',d.trim());
    var dotx=xpx(z),doty=D.ypx(D.f(z)),sl=D.d(z);
    dot.setAttribute('cx',dotx.toFixed(1));dot.setAttribute('cy',doty.toFixed(1));
    var pxslope=sl*D.yscale/xscale,L=28;
    tan.setAttribute('x1',(dotx-L).toFixed(1));tan.setAttribute('y1',(doty+L*pxslope).toFixed(1));
    tan.setAttribute('x2',(dotx+L).toFixed(1));tan.setAttribute('y2',(doty-L*pxslope).toFixed(1));
    var col=sl>=0.5?'#2D8B55':(sl>=0.1?'#C99A12':'#C93B3B');
    var word=sl>=0.5?'passes strongly':(sl>=0.1?'shrinks':(sl>0?'nearly dies (flat tail)':'a full stop — nothing gets through'));
    var kink=(act==='relu'&&Math.abs(z)<0.001)?' <span style="color:#6B645E">— right at the corner, so this is the convention, not a real slope</span>':'';
    tan.setAttribute('stroke',col);dot.setAttribute('fill',col);
    out.innerHTML='at z = '+z.toFixed(1)+', '+act+' slope = <b>'+sl.toFixed(2)+'</b> → a gradient of 1.00 passes back as <b>'+sl.toFixed(2)+'</b> <span style="color:'+col+'">('+word+')</span>'+kink;
  }
  zs.addEventListener('input',paint);
  bSig.addEventListener('click',function(){act='sigmoid';setBtns();paint();});
  bTanh.addEventListener('click',function(){act='tanh';setBtns();paint();});
  bRelu.addEventListener('click',function(){act='relu';setBtns();paint();});
  setBtns();paint();
})();</script>
%%%

**What that slider is telling you:** the gradient does not pass through the activation untouched — the **activation's slope multiplies** it, using the `z` your forward pass saved. Biggest in the middle (and even at its peak, sigmoid slope only reaches about `0.25`), nearly flat in the tails. Keep that number `0.25` in your pocket; it's the villain of the next unit.

Now the rules, in the order the backward walk actually meets them: first the activation takes its cut, then whatever survives gets split between the weight and the bias — and finally one extra rule for travelling further back.

#### Rule 1 · the activation charges a toll
Whatever gradient arrives at the neuron from the loss side is first scaled by the activation's slope at the saved `z` — the thing you just felt on the slider. Call the number that gets *through* the toll `δ` ("delta"); that is the gradient that reaches the knobs.

- **[[sigmoid||an S-shaped activation that squashes numbers into 0–1; its slope is σ(z)(1−σ(z)), which peaks at only 0.25]]** — the **sigmoid slope** `σ(z)(1−σ(z))` tops out near `0.25` in the middle, and goes tiny in the tails. An expensive toll.
- **[[ReLU||an activation that passes positives and zeros negatives; its slope is a clean 1 for positive inputs, 0 for negative]]** — the **ReLU slope** is a clean `1` (positive `z`) or `0` (negative `z`). Free passage, or a full stop.

#### Rule 2 · the weight's share = its input × the gradient that got through the toll
A weight only touches the answer *through* the input it multiplies. So take `δ` — the gradient that survived the toll — and multiply it by that weight's own **input value**. Nothing else joins in. That one multiply is the **weight's gradient**, the whole story.

%%% steps
step: the loud teammate — a weight fed a BIG input gets a big gradient
why: it had more say in the answer, therefore it owns more of the blame
step: the quiet teammate — a weight fed a tiny input barely moves
why: it hardly affected the guess, so correcting it would hardly help
step: the rule of thumb — **big inputs** cause big weight updates
why: that's also why we scale our data later; a runaway input makes a runaway update
%%%

#### Rule 3 · the bias travels free
The [[bias||the neuron's constant offset, added after the weighted sum; nudging it moves z one-for-one]] has **slope 1** — nudge it by a hair and `z` moves by exactly that hair. So `δ` **passes straight through** to the bias, unchanged. No second toll at all.

#### Rule 4 · one for the road — going further back, multiply by the weight
Rules 1–3 finish this neuron's own knobs. But in a real network the input `x` was itself the *output of the neuron behind it*, so the sweep has to keep travelling. The weighted-sum box has one more local slope, pointing back the way we came: nudge the **input** by a hair and `z` moves by `w` — the weight it gets multiplied by. Therefore the gradient handed to the previous layer is `δ × w`.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="The weighted-sum box with three exits for the gradient. Delta arrives from the activation. Exit one goes down to the weight and is delta times the input x. Exit two goes down to the bias and is delta times one. Exit three goes left, further back to the previous layer, and is delta times the weight w. A caption says the weight matters twice: once as a knob to fix, once as the toll on the road back."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Three exits out of the weighted-sum box</text><rect x="196" y="34" width="132" height="34" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><text x="262" y="55" text-anchor="middle" fill="#9A7208">z = w·x + b</text><text x="400" y="55" fill="#5A544E" font-size="9.5">δ arrives here</text><g stroke="#2D8B55" stroke-width="2.2" fill="#2D8B55"><path d="M392 51 l-56 0"/><polygon points="336,51 344,47 344,55"/><path d="M240 68 l0 34"/><polygon points="240,102 236,94 244,94"/><path d="M290 68 l0 34"/><polygon points="290,102 286,94 294,94"/><path d="M196 51 l-56 0"/><polygon points="140,51 148,47 148,55"/></g><rect x="188" y="106" width="104" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="240" y="123" text-anchor="middle" fill="#1F6280" font-size="9">weight: δ × x</text><rect x="300" y="106" width="96" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="348" y="123" text-anchor="middle" fill="#1F6280" font-size="9">bias: δ × 1</text><rect x="18" y="38" width="118" height="28" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text x="77" y="56" text-anchor="middle" fill="#5E5191" font-size="9">back a layer: δ × w</text><text x="77" y="80" text-anchor="middle" fill="#6B645E" font-size="9">← keeps the sweep going</text><line x1="30" y1="146" x2="490" y2="146" stroke="#E5DFD6"/><text x="260" y="166" text-anchor="middle" fill="#5A544E" font-size="9.5">the weight matters twice: as a knob to fix (δ × x), and as the toll on the road back (δ × w)</text></g></svg>
%%%

That little `× w` looks harmless. Hold on to it — it is the reason the next unit's dial can go both *below* and *above* 1, and the reason "size your starting weights sensibly" is a real cure and not just advice.

%%% demo id=rules label="run it — split the blame three ways"
predict: the input is `x = 2.0` and the same gradient `0.3` arrives at the neuron. Will the weight's gradient come out BIGGER than the bias's, smaller, or exactly equal? By what factor?
code: x=2.0; w=0.5; grad_into_neuron=0.3; relu_slope=1.0  # input, weight, gradient arriving from the loss side, ReLU slope for positive z
code: delta=grad_into_neuron*relu_slope           # Rule 1: pay the activation's toll
code: w_grad=delta*x; b_grad=delta*1.0            # Rule 2: × its input · Rule 3: bias travels free
code: back=delta*w                                # Rule 4: what gets handed to the previous layer
code: print("delta",delta," weight grad",w_grad," bias grad",b_grad," passed back",back)
out: delta 0.3  weight grad 0.6  bias grad 0.3  passed back 0.15
take: <b>Big input → big weight gradient.</b> The toll was free here (ReLU slope 1), so δ = 0.3. The weight's blame is δ × its input = 0.3 × 2.0 = 0.6 — exactly twice the bias's 0.3, because its input was 2×. And what travels one layer further back is δ × w = 0.3 × 0.5 = 0.15: this weight was smaller than 1, so the signal got quieter on the way. Same chain-rule multiply, four exits.
%%%

%%% insight
Did you catch the shape of all four rules? They are the *same* move — incoming gradient × one local slope — wearing four different hats. That's the whole reason a beginner can hold backprop in their head: there is only ever one rule, applied at whatever box you happen to be standing on.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the rules in symbols.</b> Peek only if you like notation; the plain lines above are all you need. With loss slope <code>g = ∂L/∂a</code> arriving at the neuron and activation slope <code>a'(z)</code>: Rule 1 is <code>δ = g · a'(z)</code>; Rule 2 is <code>∂L/∂wᵢ = δ · xᵢ</code>; Rule 3 is <code>∂L/∂b = δ · 1 = δ</code>; Rule 4 is <code>∂L/∂xᵢ = δ · wᵢ</code>. Every symbol is just "incoming gradient × one local slope."
!!!

Those rules *are* the neuron's backward pass — and notice each one is safe and boring on its own: a multiply by a slope, by an input, by a plain `1`, or by a weight. Nice work; that's the mechanism, done. The drama only starts when you **chain many of them together**, which is exactly the next puzzle.

@@@ concept id=c6 tag="The fading signal" title="Puzzle — when the downhill signal fades to nothing" gotit="Got vanishing gradients"
Here's your first puzzle, and it's the most famous one in deep learning. Play the **telephone game**: a whisper passed down a long line of kids, each repeating it a little quieter, until the last kid hears nothing at all. Now hold that picture next to what you just learned — travelling back one layer means multiplying by *two* small things (the activation's slope, then the weight), and you saw sigmoid's slope is at most `0.25`. So what reaches the *first* layer after four of those multiplies?

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A line of kids playing telephone. A whisper starts loud on the right and gets quieter at each kid to the left, shrinking to nothing. Caption: multiply by a small amount at every layer and the gradient fades to zero."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Telephone: each layer multiplies by a small amount → the whisper fades</text><text x="470" y="70" font-size="24">🧒</text><text x="470" y="98" text-anchor="middle" fill="#C93B3B" font-size="10">loud</text><text x="368" y="70" font-size="22">🧒</text><text x="378" y="98" text-anchor="middle" fill="#C99A12" font-size="10">×0.25</text><text x="270" y="70" font-size="19">🧒</text><text x="280" y="98" text-anchor="middle" fill="#C99A12" font-size="10">×0.25</text><text x="176" y="70" font-size="16">🧒</text><text x="184" y="98" text-anchor="middle" fill="#9A938A" font-size="10">×0.25</text><text x="92" y="70" font-size="12">🧒</text><text x="96" y="98" text-anchor="middle" fill="#B8AEA2" font-size="10">×0.25</text><text x="40" y="70" font-size="9" fill="#B8AEA2">·</text><g stroke="#B8AEA2" stroke-width="1.5" fill="#B8AEA2"><path d="M460 60 l-70 0"/><polygon points="390,60 398,56 398,64"/><path d="M360 60 l-70 0"/><polygon points="290,60 298,56 298,64"/><path d="M262 60 l-70 0"/><polygon points="192,60 200,56 200,64"/><path d="M168 60 l-64 0"/><polygon points="104,60 112,56 112,64"/></g><text x="60" y="130" fill="#9A938A" font-size="10">last kid hears ≈ nothing 😴</text><text x="470" y="130" text-anchor="end" fill="#C93B3B" font-size="10">loss end (loud) →</text></g></svg>
%%%

%%% hint
t1: You already know one sigmoid layer multiplies the gradient by a slope that is at most 0.25. What happens to that number when a SECOND layer multiplies by 0.25 again?
t2: Multiply it out one layer at a time: 0.25, then 0.25 × 0.25 = 0.0625, then × 0.25 again = 0.0156. Each layer makes it about four times smaller.
t3: Stacking many layers means multiplying many numbers below 1, so the gradient shrinks toward 0 and the early layers stop learning — that's the vanishing gradient.
%%%

**What the telephone picture gets right:** repeatedly shrinking a signal, step after step, drives it to almost nothing — and the *far* end (the early layers) suffers most. **Where it breaks down:** in telephone the message also gets garbled; here the gradient stays perfectly "correct," it just gets so tiny that the early weights barely move.

#### Walk down the line of kids
This is the [[vanishing gradient||when many small per-layer multiplies stack up, the gradient shrinks toward 0, so early layers stop learning]] problem. To keep the numbers friendly, say each layer's weights are sized so the `× w` part is about `1`, leaving the sigmoid's `0.25` to do the shrinking. Let's read the volume at each kid:

%%% steps
step: kid 1 (the last layer) — gradient `1` × `0.25` = `0.25`
why: one sigmoid toll booth already took three quarters of the signal
step: kid 2 — `× 0.25` again = `0.0625`
why: the shrink compounds; each layer makes the whisper about four times quieter
step: kid 3 — `0.0156`
why: we are still only three layers deep and the signal is already 1.5% of what it was
step: kid 4 (the FIRST layer) — under `0.004`
why: that's the freeze — the early layers get almost no downhill signal, so they stop learning
%%%

Now put your own hand on the dial. It sets **what each layer multiplies the gradient by** — the activation's slope times the weight, the two tolls from Rule 1 and Rule 4 rolled into one number. **Predict first:** to keep the gradient alive across 8 layers, does that number need to be *below* 1, *about* 1, or *above* 1? Drag and see — and while you're there, push it past 1 and watch what happens the other way:

%%% svg
<svg id="vg-svg" viewBox="0 0 520 200" role="img" aria-label="Interactive gradient-through-depth ladder. A slider sets what each layer multiplies the gradient by, which is the activation slope times the weight, and eight bars show the running product of the gradient after it has travelled back one, two, up to eight layers. Bar heights are on a logarithmic scale, so an equal drop in height means an equal multiplying cut. At 0.25 per layer the bars step down evenly and the printed values collapse toward zero, which is the vanishing gradient. At 1 the bars stay level. Above 1 the bars run off the top, which is the exploding gradient."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The gradient after each layer — you set the per-layer multiply</text><text x="34" y="31" fill="#9A938A" font-size="9">bar height = LOG scale · each equal drop is an equal ×cut — read the printed number for true size</text><line x1="34" y1="150" x2="500" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><line x1="34" y1="67.5" x2="500" y2="67.5" stroke="#B8AEA2" stroke-width="1" stroke-dasharray="4,3"/><text x="500" y="63" text-anchor="end" fill="#9A938A" font-size="9">= 1.0 · signal unchanged</text><g id="vg-bars"><rect id="vg-b0" x="46" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b1" x="100" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b2" x="154" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b3" x="208" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b4" x="262" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b5" x="316" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b6" x="370" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b7" x="424" y="149" width="40" height="1" fill="#C99A12"/></g><g font-size="8.5" text-anchor="middle" fill="#6B645E"><text id="vg-v0" x="66" y="146">·</text><text id="vg-v1" x="120" y="146">·</text><text id="vg-v2" x="174" y="146">·</text><text id="vg-v3" x="228" y="146">·</text><text id="vg-v4" x="282" y="146">·</text><text id="vg-v5" x="336" y="146">·</text><text id="vg-v6" x="390" y="146">·</text><text id="vg-v7" x="444" y="146">·</text></g><g font-size="9" text-anchor="middle" fill="#9A938A"><text x="66" y="166">1 back</text><text x="120" y="166">2</text><text x="174" y="166">3</text><text x="228" y="166">4</text><text x="282" y="166">5</text><text x="336" y="166">6</text><text x="390" y="166">7</text><text x="444" y="166">8</text></g><text x="260" y="188" text-anchor="middle" fill="#6B645E" font-size="9.5">how many layers the gradient has travelled back · 8 = the earliest layer</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px">
<span>try:</span>
<button id="vg-sig" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">sigmoid, tidy weights → 0.25</button>
<button id="vg-relu" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">ReLU, tidy weights → 1.0</button>
<button id="vg-big" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">ReLU, big weights → 1.6</button>
</div>
<label>each layer multiplies by <input id="vg-s" type="range" min="0.05" max="2" step="0.05" value="0.25" style="width:50%;accent-color:#C99A12;vertical-align:middle"></label>
<div id="vg-out" style="margin-top:6px;color:#2C2A28">multiply <b>0.25</b> per layer → after 8 layers the gradient is <b>×1.5e-5</b> <span style="color:#C93B3B">(vanished — the early layers get nothing)</span></div>
<div style="margin-top:4px;color:#6B645E;font-size:.92em">This one number is the activation's slope <b>times</b> the weight. Sigmoid alone caps its half at 0.25; a weight bigger than 1 can push the product above 1 — which is why the dial reaches past 1 at all.</div>
</div>
<script>(function(){
  var sl=document.getElementById('vg-s');if(!sl)return;
  var out=document.getElementById('vg-out'),BASE=150,H=110;
  var bars=[],vals=[];
  for(var i=0;i<8;i++){bars.push(document.getElementById('vg-b'+i));vals.push(document.getElementById('vg-v'+i));}
  function fmt(p){
    if(p>=1000||p<0.001)return '×'+p.toExponential(1);
    if(p>=10)return '×'+p.toFixed(1);
    return '×'+p.toFixed(p<0.1?4:3);
  }
  function hgt(p){
    var h=H*(Math.log(p)/Math.LN10+6)/8;
    return Math.max(1,Math.min(H,h));
  }
  function paint(){
    var s=+sl.value,p=1,col,word;
    for(var i=0;i<8;i++){
      p=p*s;
      var h=hgt(p);
      col=(p>3)?'#C93B3B':(p<0.05?'#C93B3B':(p<0.5?'#C99A12':'#2D8B55'));
      bars[i].setAttribute('y',(BASE-h).toFixed(1));
      bars[i].setAttribute('height',h.toFixed(1));
      bars[i].setAttribute('fill',col);
      vals[i].setAttribute('y',(BASE-h-4).toFixed(1));
      vals[i].textContent=fmt(p);
      vals[i].setAttribute('fill',col);
    }
    word=(p>3)?'<span style="color:#C93B3B">(exploded — one step overshoots the whole valley)</span>'
        :(p<0.05?'<span style="color:#C93B3B">(vanished — the early layers get nothing)</span>'
        :(p<0.5?'<span style="color:#C99A12">(fading — the early layers learn slowly)</span>'
        :'<span style="color:#2D8B55">(healthy — the signal survives all 8 layers)</span>'));
    out.innerHTML='multiply <b>'+s.toFixed(2)+'</b> per layer → after 8 layers the gradient is <b>'+fmt(p)+'</b> '+word;
  }
  sl.addEventListener('input',paint);
  document.getElementById('vg-sig').addEventListener('click',function(){sl.value=0.25;paint();});
  document.getElementById('vg-relu').addEventListener('click',function(){sl.value=1;paint();});
  document.getElementById('vg-big').addEventListener('click',function(){sl.value=1.6;paint();});
  paint();
})();</script>
%%%

Slide to `0.25` and each bar drops by the *same* height as the one before — one equal step shorter, every time. But those equal steps are a **log scale**, and that is the sneaky bit: each equal drop means *four times smaller*.

Do it eight times and you have `0.25⁸` — a `65,536×` cut, which is exactly why the printed number ends at about `×1.5e-5` (that notation just means 15 millionths) while the bars only look a few times shorter. Trust the numbers over the picture here — small multiplies **multiply toward 0** far faster than any height can show.

So far we have shown one setting of the dial. Try the other two and today's whole cast of traps fits on three rungs:

%%% steps
step: the fade — leave the dial at `0.25`
why: every bar is one equal height shorter than the last, and the printed numbers collapse toward zero
step: the flat line — slide to `1.0`
why: the bars sit level on the dashed line all the way back to the earliest layer, so the signal survives
step: the runaway — push past `1`
why: they climb instead of drop, and near the top of the dial they run clean off the picture — hold that thought, you'll meet that trap by name shortly
%%%

%%% demo id=vanish label="run it — the fade in five numbers"
predict: after 5 sigmoid layers, is the gradient about 0.1, about 0.01, or about 0.001? Commit to a guess, then reveal.
code: per_layer=0.25; [round(per_layer**n,4) for n in range(1,6)]  # gradient after 1..5 sigmoid layers
out: [0.25, 0.0625, 0.0156, 0.0039, 0.001]
take: <b>Small multiplies drive the gradient toward zero.</b> After 5 sigmoid layers the gradient is ×0.001 — the early layers barely move. This is why deep sigmoid networks were so hard to train for years.
%%%

It's our **hiker in fog** again, but the ground feels flatter and flatter the deeper the layer, until she genuinely cannot tell which way is down. And the main cause has a name: sigmoid is a **saturating** activation — out in its flat tails it barely reacts at all, so the **signal fades**.

%%% insight
This is the sneaky part, and it's worth a moment: the network does not crash. It trains happily, prints a loss, uses your GPU — while its first layers secretly learn almost nothing. For about two decades this quiet fade is what kept neural networks shallow. You are looking at the bug that held back a whole field.
%%%

#### The neat fix — a slope that refuses to shrink
So the cure writes itself: stop multiplying by tiny numbers. **Switch to ReLU**, whose slope for any positive input is a clean `1` (press the "ReLU, tidy weights" button above and watch the bars go level). The **ReLU slope stays 1** however many layers you stack, so the *activation* stops eating the signal — multiply by `1` a hundred times and the whisper is still exactly as loud.

That leaves only the other half of the per-layer multiply: the `× w` from Rule 4. Size the starting weights sensibly and that half also lands near `1`, so the product neither fades nor grows. Sizing the starting weights well is exactly what people mean by **careful initialization** — there are named recipes for it, and you'll meet them when you build your first multi-layer network.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Before and after: with sigmoid the per-layer multiply is about 0.25 and the gradient fades to near zero across four layers; with ReLU and sensibly sized weights the per-layer multiply is about 1 and the gradient stays full height. Two rows of bars compared."><g font-family="monospace" font-size="10"><text x="140" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">sigmoid → ×0.25 per layer → fades</text><rect x="40" y="34" width="30" height="60" fill="#C99A12"/><rect x="86" y="79" width="30" height="15" fill="#C99A12"/><rect x="132" y="90" width="30" height="4" fill="#C93B3B"/><rect x="178" y="93" width="30" height="1" fill="#C93B3B"/><line x1="34" y1="94" x2="216" y2="94" stroke="#E5DFD6"/><text x="120" y="112" text-anchor="middle" fill="#6B645E" font-size="9">early layers freeze 💀</text><line x1="258" y1="24" x2="258" y2="104" stroke="#E5DFD6"/><text x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">ReLU + sane weights → ×≈1 → stays loud</text><rect x="300" y="34" width="30" height="60" fill="#2D8B55"/><rect x="346" y="34" width="30" height="60" fill="#2D8B55"/><rect x="392" y="34" width="30" height="60" fill="#2D8B55"/><rect x="438" y="34" width="30" height="60" fill="#2D8B55"/><line x1="294" y1="94" x2="476" y2="94" stroke="#E5DFD6"/><text x="384" y="112" text-anchor="middle" fill="#1a5c38" font-size="9">signal survives every layer 🎉</text></g></svg>
%%%

**One puzzle solved, and it's the big one.** You just learned the cause (small multiplies stacked) *and* the cure (ReLU's slope of 1, plus sanely sized weights), which is more than most people can explain about deep learning. Let's cash that in on something you can use today.

@@@ concept id=c7 tag="Check your work" title="The gradient check — measure it two ways" gotit="Got the gradient check"
Breather time, and this one is a gift you get to keep for the rest of your life in this field. You can now compute the downhill slope with the chain rule — but how do you know you didn't slip a sign or fat-finger a symbol? Think of **weighing flour two ways**: once by reading the recipe's math, and once by putting the bowl on a kitchen scale. If the two numbers agree, you trust your recipe. If they disagree, you know a step is wrong — *before* you bake a ruined cake. That is the whole trick, and it takes about thirty seconds.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Two ways to get the same slope compared on a balance scale. On the left, backprop's computed gradient. On the right, a measured slope from nudging the weight up and down. When they match, the check passes; a mismatch flags a bug."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two ways to the same slope — do they match?</text><rect x="34" y="46" width="150" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="109" y="63" text-anchor="middle" fill="#5E5191">backprop's gradient</text><text x="109" y="78" text-anchor="middle" fill="#6B645E" font-size="9">(the chain rule)</text><rect x="336" y="46" width="150" height="40" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="411" y="63" text-anchor="middle" fill="#1a5c38">measured slope</text><text x="411" y="78" text-anchor="middle" fill="#6B645E" font-size="9">(nudge &amp; re-check the loss)</text><text x="260" y="60" text-anchor="middle" font-size="20">⚖️</text><text x="260" y="80" text-anchor="middle" fill="#6B645E" font-size="10">≈ equal?</text><text x="150" y="122" fill="#2D8B55" font-size="10">match → trust your code ✓</text><text x="360" y="122" fill="#C93B3B" font-size="10" text-anchor="end">mismatch → hunt the bug 🐛</text></g></svg>
%%%

**What the two-weighings picture gets right:** computing a number and *measuring* it independently is the surest way to catch a mistake — agreement builds trust, disagreement flags a bug. **Where it breaks down:** a kitchen scale is exact, but our measured slope is only an *estimate* (it uses a tiny nudge, not a truly zero-size step), so you check that the two are *close*, not identical to the last digit.

#### Put the weight on the scale
A slope is just "how much the output moves when you nudge the input." So let's **nudge and measure** it, with no calculus at all — this is our **hiker** stamping her boot to double-check the tilt she thinks she feels:

%%% steps
step: nudge up — add a tiny `ε` to one weight and read the loss
why: that's putting a pinch more flour in and re-weighing; one forward pass, nothing clever
step: nudge down — subtract the same `ε` and read the loss again
why: measuring both sides (a **finite difference**) cancels most of the error, therefore a sharper reading
step: divide — `(loss(w+ε) − loss(w−ε)) / (2ε)`
why: change in height ÷ distance moved — the plain definition of a slope
step: compare — does it match what backprop computed?
why: if yes, your backward pass is right. If no, you have a bug, and you found it in seconds
%%%

That four-rung recipe is the **numerical gradient check**. Two loss readings and a divide — that's it.

So far we have shown the recipe. Now watch it agree with backprop on a case where you already know the right answer:

%%% demo id=gcheck label="run it — two ways, one slope"
predict: for the toy loss `f(w) = w²` at `w = 3`, the true slope is `2w = 6`. Will the nudge-and-measure version land on 6.0, or drift noticeably off?
code: f=lambda w: w**2; w=3.0; eps=1e-5  # toy loss f(w)=w^2, whose true slope at w is 2w=6
code: measured=(f(w+eps)-f(w-eps))/(2*eps); backprop=2*w; print("measured",round(measured,4)," backprop",backprop)
out: measured 6.0  backprop 6.0
take: <b>They match → the gradient is right.</b> The measured slope (nudge up and down, divide by 2ε) is 6.0, and backprop says 6.0 too. When these disagree, you've got a bug in your backward pass — this cheap check catches it before you waste a training run.
%%%

%%% insight
Why this is a habit and not a chore: a wrong gradient does not throw an error. It trains, it prints numbers, it just quietly learns the wrong thing for six hours. Thirty seconds of gradient-checking on a toy example is the cheapest confidence you will ever buy in this field.
%%%

!!! c-warn ⚠️
<b>Use it to check, not to train.</b> Measuring the slope needs <i>two</i> full forward passes <i>per weight</i> — one for <code>w+ε</code> and one for <code>w−ε</code> — which is hopelessly slow for millions of weights. Backprop gets them all in one sweep. So you gradient-check on a tiny toy to prove the code is right, then let fast backprop do the real work.
!!!

**Victory lap: you can now *prove* your own gradients.** That is a genuinely professional habit, it fits on one page, and you picked it up in about five minutes — engineers at every lab run this exact check before trusting a new model.

Keep it in your pocket, because the next three beats are the classic ways learning goes wrong even when your math is perfect. With the check in hand you'll be able to tell "my code is buggy" apart from "my code is fine, my *setup* is bad" — and that distinction is most of debugging.

First up, the sneakiest one, and it hides in the very first line of any training script.

@@@ concept id=c8 tag="The twin trap" title="Puzzle — when every neuron becomes the same neuron" gotit="Got symmetry breaking"
One quick puzzle, and it's the sneakiest of the day because nothing about it looks broken. Picture **identical twins** who copy each other's every move: same breakfast, same route to school, same answer on every quiz. Two people, one behavior. Now think about how you might start a fresh network. Setting every weight to the same tidy value — all `0`, say — feels neat and fair. It is also the twins.

%%% svg
<svg viewBox="0 0 520 152" role="img" aria-label="Identical twins in lockstep. Two children with the same clothes take the same step at the same moment, with a caption saying same start, same move, forever. Beside them a label says a whole layer of neurons started at the same value behaves like one neuron, and the cure is a small random start."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Identical twins: same start → same move → forever</text><text x="90" y="72" font-size="34">👯</text><rect x="150" y="42" width="150" height="34" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="225" y="63" text-anchor="middle" fill="#C93B3B">same step, always</text><g stroke="#C93B3B" stroke-width="2" fill="#C93B3B"><path d="M306 59 l38 0"/><polygon points="344,59 336,55 336,63"/></g><rect x="350" y="42" width="156" height="34" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="428" y="57" text-anchor="middle" fill="#C93B3B" font-size="9.5">a whole layer acting</text><text x="428" y="70" text-anchor="middle" fill="#C93B3B" font-size="9.5">like ONE neuron</text><line x1="30" y1="96" x2="490" y2="96" stroke="#E5DFD6"/><text x="260" y="116" text-anchor="middle" fill="#2D8B55">cure: give each neuron a small RANDOM starting value 🎲</text><text x="260" y="136" text-anchor="middle" fill="#6B645E" font-size="9.5">a slightly different start is all it takes for them to grow into different jobs</text></g></svg>
%%%

**What the twins picture gets right:** the sameness is *self-sustaining* — identical starts plus identical experiences keep them identical, however long you wait. **Where it breaks down:** real twins eventually differ because life hands them different days; a network gets exactly the same inputs and the same rules for every clone, so nothing ever breaks the tie on its own.

#### Why the clones never drift apart
Follow the blame and you'll see the trap close:

%%% steps
step: the innocent choice — start every weight in a layer at the same value
why: this is **zero-initialization** (or any all-same start), and it feels tidy and fair
step: the collapse — every neuron computes the same thing, so backprop hands them **identical gradients**
why: same inputs, same local slopes, and the same outgoing wire carrying blame back — therefore the same share arrives at each unit, to the last digit
step: the trap — they update identically and **stay identical** forever
why: a hundred neurons behave like one; you paid for a layer and got a single unit
step: the fix — start at **small random** values to **break the symmetry**
why: the layer is no longer symmetric, so the units stop being interchangeable and each is free to grow into its own detector
%%%

%%% svg
<svg viewBox="0 0 520 208" role="img" aria-label="Two panels comparing starting weights. Left panel: three tanh neurons whose incoming weights all start at 0.30 and whose wires out are also equal, so their local slopes match and backprop hands all three the identical gradient minus 0.45; they stay clones forever and the whole layer acts like one neuron. Right panel: three tanh neurons whose incoming weights start at three different small values, minus 0.10, 0.45 and 0.65, so their local slopes differ and the gradients coming back differ too, minus 0.59, minus 0.40 and minus 0.26; the three units grow into three different detectors. Every wire out is 0.5 in both panels, so the weight in is the only thing that changed."><g font-family="monospace" font-size="10.5"><text x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">❌ all weights start at the SAME value</text><text x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">✅ small random start</text><rect x="14" y="24" width="232" height="136" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="274" y="24" width="232" height="136" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><text x="130" y="39" text-anchor="middle" fill="#5A544E" font-size="9">same weight in → same steepness → same blame</text><text x="390" y="39" text-anchor="middle" fill="#5A544E" font-size="9">own weight in → own steepness → own blame</text><g stroke="#C93B3B" fill="#F7E4E4"><circle cx="60" cy="66" r="15"/><circle cx="60" cy="102" r="15"/><circle cx="60" cy="138" r="15"/></g><g fill="#C93B3B" text-anchor="middle" font-size="8.5"><text x="60" y="69">.30</text><text x="60" y="105">.30</text><text x="60" y="141">.30</text></g><g stroke="#C93B3B" stroke-width="1.8"><line x1="75" y1="66" x2="143" y2="66"/><line x1="75" y1="102" x2="143" y2="102"/><line x1="75" y1="138" x2="143" y2="138"/></g><g fill="#C93B3B" font-size="9.5"><text x="147" y="69">grad −0.45</text><text x="147" y="105">grad −0.45</text><text x="147" y="141">grad −0.45</text></g><text x="130" y="156" text-anchor="middle" fill="#C93B3B" font-size="9.5">3 neurons behaving as 1</text><g stroke="#2D8B55" fill="#E6F2EA"><circle cx="320" cy="66" r="15"/><circle cx="320" cy="102" r="15"/><circle cx="320" cy="138" r="15"/></g><g fill="#2D8B55" text-anchor="middle" font-size="8.5"><text x="320" y="69">−.10</text><text x="320" y="105">.45</text><text x="320" y="141">.65</text></g><g stroke="#2D8B55" stroke-width="1.8"><line x1="335" y1="66" x2="403" y2="66"/><line x1="335" y1="102" x2="403" y2="102"/><line x1="335" y1="138" x2="403" y2="138"/></g><g fill="#2D8B55" font-size="9.5"><text x="407" y="69">grad −0.59</text><text x="407" y="105">grad −0.40</text><text x="407" y="141">grad −0.26</text></g><text x="390" y="156" text-anchor="middle" fill="#2D8B55" font-size="9.5">3 neurons doing 3 things</text><text x="260" y="177" text-anchor="middle" fill="#5A544E" font-size="8.5">every wire out is .5 in both panels — the weight IN is the only thing that changed (tanh units)</text><text x="260" y="191" text-anchor="middle" fill="#5A544E" font-size="9">symmetric all through — same weights in AND same wires out → same forever</text><text x="260" y="204" text-anchor="middle" fill="#5A544E" font-size="9">randomness is not sloppiness — it is what lets a layer specialize</text></g></svg>
%%%

So far we have shown the trap as a picture. Now let's settle it with arithmetic. We'll use **tanh** units here — the smooth S-curve from Day 2 — because a tanh unit's local slope changes as its `z` changes, so a hair of difference in the weight *going in* has somewhere to show up. Let's find out whether that is enough to break the tie.

%%% demo id=twins label="run it — does a hair of difference break the tie?"
predict: three hidden units start with the same weight `0.3` and see the same input. You nudge just ONE of those starting weights to `0.31`. Does that unit's share of the blame change — or do all three stay identical?
code: import math; x=1.5; t=1.0                                        # one input, one true answer
code: def hid(w):    return [math.tanh(wi*x) for wi in w]              # 3 hidden units, each bends its OWN weighted input
code: def guess(w,v): return sum(vi*hi for vi,hi in zip(v,hid(w)))     # the single output adds up what it hears
code: def blame(w,v): return [round(2*(guess(w,v)-t)*vi*(1-hi*hi)*x,4)+0.0 for vi,hi in zip(v,hid(w))]
code: #   ↑ one unit's share = 2·(guess−t) · its wire out v · tanh's slope at ITS OWN z · the input x
code: print("A  all three start the same  ", blame([0.30,0.30,0.30], [0.5,0.5,0.5]))
code: print("B  one weight nudged to 0.31 ", blame([0.31,0.30,0.30], [0.5,0.5,0.5]))
code: print("C  every weight its own value", blame([-0.10,0.45,0.65], [0.5,0.5,0.5]))
code: print("D  every weight starts at 0  ", blame([0.00,0.00,0.00], [0.0,0.0,0.0]))
out: A  all three start the same   [-0.4527, -0.4527, -0.4527]
out: B  one weight nudged to 0.31  [-0.4395, -0.4451, -0.4451]
out: C  every weight its own value [-0.5938, -0.3971, -0.2649]
out: D  every weight starts at 0   [0.0, 0.0, 0.0]
take: <b>Start a whole layer the same and it gets one shared answer back; give the units different weights and they stop being interchangeable.</b> Row A is the twins: identical weights bend the input identically, so all three units get the very same `-0.4527` and move as one forever. Row B breaks the tie with a hair — the nudged unit now sits at a different place on the tanh curve, where the curve has a different steepness, so its share is `-0.4395`, while the two you left alone still match each other at `-0.4451` (they still share a weight). Row C does the cure properly: give every weight its own small random value and all three shares differ. Row D is the tidiest-looking start and the worst of the four — every share is exactly `0`.
%%%

!!! c-info 🔬
<b>Optional (skippable) — what if the units use ReLU or sigmoid instead of tanh?</b> The one rule that holds for every activation is the trap itself: start a layer completely symmetric — same weights in <i>and</i> same wires out — and it stays symmetric forever. How fast the CURE bites is what varies.<br>
<b>Sigmoid behaves like tanh:</b> its steepness also changes with `z`, so a nudged weight gives that unit a different share on the very first step.<br>
<b>ReLU is slower:</b> its slope is a flat `1` for every positive `z`, so the nudge moves all three shares together — from `-0.4875` to `-0.4763` — changing them without making them differ.<br>
<b>What breaks it anyway:</b> a unit sitting at a positive `z` still <i>outputs</i> something different, so the wire leading out of it picks up its own gradient on the first step, and the layer stops being a set of clones.<br>
<b>Careful with "different is always enough":</b> it isn't. Shove a ReLU unit's `z` negative and it outputs `0` whatever its weight is — which is exactly the dead unit you meet next.
!!!

#### One extra beat — why the all-zero start is the worst of all
Row D deserves its own sentence. Rule 4 said the blame reaching a hidden unit travels through the wire leading *out* of it. Start every weight at `0` and every one of those wires is `0` too, so on the first step every weight in the hidden layer gets a gradient of exactly `0`. For a `tanh` or `ReLU` unit — whose output is `0` when `z` is `0` — it stays stuck there: the hidden layer never learns anything at all.

The only thing left that can still move is the output's own bias (Rule 3), so the network quietly settles on predicting one constant number for every input. And note which trap this is: it is the twins in their most extreme form, so the cure is a **small random start** — not Leaky ReLU. Nothing here is a dead unit; the whole layer is simply mute. A small random start fixes both sides at once: each unit gets its own weight in *and* its own wire out.

%%% insight
Why this one bites hardest: a same-start network trains without a single error message. The loss even goes down a little — one neuron's worth (and from an all-zero start with `tanh` or `ReLU` units the hidden layer learns nothing at all, yet the loss usually *still* creeps down, because that one output bias is drifting toward the average answer). You would stare at that code for hours. Knowing this trap by name is worth an entire afternoon of debugging, today.
%%%

**One line of code, one whole class of bug avoided.** Every framework already hands you small random starting weights by default, and now you know *why* that default exists — you'd have invented it yourself. Two more cousins to meet, and they are the loud ones.

@@@ concept id=c9 tag="Stuck &amp; squealing" title="Puzzle — the stuck switch and the mic squeal" gotit="Got the two cures"
Two last puzzles, and this time you get to break things on purpose. Picture two everyday annoyances. First, a **light switch jammed OFF** — you flick it and nothing happens, because the thing that would move it is the very thing that's broken. Second, a **microphone held too close to its own speaker**: the sound feeds itself, louder and louder, until the whole room covers its ears. One trap goes silent, the other goes deafening — and both are the same per-layer multiply, just at the two extremes: `×0` and `×big`.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Two everyday things drawn side by side. On the left a wall light switch jammed in the off position with a hand flicking it and nothing happening, labelled goes silent, multiply by zero. On the right a microphone pointed at a loudspeaker with sound waves feeding back and growing, labelled goes deafening, multiply by something big."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two loud cousins: one goes silent, one goes deafening</text><rect x="18" y="26" width="228" height="126" rx="8" fill="#FDFCF9" stroke="#E5DFD6"/><rect x="88" y="44" width="46" height="70" rx="6" fill="#EDEAE4" stroke="#9A938A"/><rect x="96" y="84" width="30" height="24" rx="3" fill="#C93B3B"/><text x="111" y="130" text-anchor="middle" fill="#C93B3B">OFF · jammed</text><text x="152" y="76" font-size="20">👆</text><text x="176" y="96" text-anchor="middle" fill="#6B645E" font-size="9">flick…</text><text x="176" y="108" text-anchor="middle" fill="#6B645E" font-size="9">nothing</text><text x="132" y="146" text-anchor="middle" fill="#C93B3B" font-size="10.5">goes SILENT · ×0</text><rect x="274" y="26" width="228" height="126" rx="8" fill="#FDFCF9" stroke="#E5DFD6"/><text x="308" y="84" font-size="24">🎤</text><rect x="428" y="56" width="40" height="52" rx="4" fill="#EDEAE4" stroke="#9A938A"/><circle cx="448" cy="82" r="12" fill="none" stroke="#9A938A"/><g fill="none" stroke="#C93B3B" stroke-width="1.6"><path d="M350 78 q10 -10 0 -20"/><path d="M362 84 q18 -16 0 -34"/><path d="M374 90 q26 -22 0 -48"/></g><g fill="none" stroke="#C93B3B" stroke-width="1.6"><path d="M418 82 q-10 8 0 16"/><path d="M406 82 q-16 12 0 26"/></g><text x="388" y="130" text-anchor="middle" fill="#C93B3B">sound feeds itself → squeal</text><text x="388" y="146" text-anchor="middle" fill="#C93B3B" font-size="10.5">goes DEAFENING · ×big</text></g></svg>
%%%

**What the two pictures get right:** each is *self-sustaining* — the jammed switch has nothing left to move it, and the squeal is its own cause. **Where they break down:** a real switch can be fixed with a screwdriver from outside, but a network only ever has its own gradients to work with, so a unit at `×0` truly cannot rescue itself. That is exactly why the cures change the *rules of the road* rather than the unit.

#### The stuck switch — a **dead ReLU**
ReLU's **slope 0** for negatives is what makes this one possible. Follow the jam:

%%% steps
step: the shove — one oversized update pushes a unit's `z` **permanently negative**
why: ReLU outputs 0 there, so the unit says nothing at all
step: the jam — its slope is 0 too, therefore its gradient is 0
why: zero gradient means zero update, so `z` stays negative — the switch cannot flip itself back
step: the verdict — this unit **never learns** again; it is a **dead ReLU**
why: nothing in normal training can rescue it, because a rescue would need a gradient
step: the fix — **Leaky ReLU** keeps a **small negative slope** (about `0.01`, gently rising, instead of flat `0`)
why: a trickle of gradient always gets through, so a shoved unit can climb back out
%%%

Now go break both of them yourself. The left dial shoves a unit's `z` around (tick **Leaky** to change the rules of the road); the right dial cranks the weight size that every layer multiplies by (tick **clip** to put a ceiling on it). **Predict first:** on the left, what gradient does the unit get once `z` goes negative? On the right, what does a weight of `3` do to the gradient after six layers?

%%% svg
<svg id="tc-svg" viewBox="0 0 520 236" role="img" aria-label="Two interactive panels. The left panel is a unit dashboard: a light switch drawing shows on or off, and readouts show the unit's z, its output, its slope, and a bar for the gradient reaching its weight. When z is negative with plain ReLU the bar is empty and the verdict reads dead. Ticking Leaky ReLU leaves a small non-zero bar and the verdict reads it can recover. The right panel shows six bars for the gradient after travelling back one to six layers as the weight size changes. Small weights keep the bars low, a weight of three makes every bar taller than the last and the verdict reads exploded. Ticking clip holds any bar that reaches the ceiling at the ceiling line, leaving the smaller earlier bars alone, so the held bars end up level with each other, and the verdict then reports the size the gradient would have reached and the smaller size clipping passed on instead."><g font-family="monospace" font-size="10"><text x="132" y="20" text-anchor="middle" fill="#2C2A28" font-size="11.5">🔌 the stuck switch</text><text x="388" y="20" text-anchor="middle" fill="#2C2A28" font-size="11.5">🎤 the squeal</text><rect x="14" y="28" width="236" height="178" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="270" y="28" width="236" height="178" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="34" y="44" width="40" height="60" rx="6" fill="#EDEAE4" stroke="#9A938A"/><rect id="tc-knob" x="40" y="74" width="28" height="24" rx="3" fill="#C93B3B"/><text id="tc-state" x="54" y="118" text-anchor="middle" fill="#C93B3B" font-size="9">OFF</text><text id="tc-z" x="92" y="58" fill="#5A544E">z = -1.0</text><text id="tc-out" x="92" y="76" fill="#5A544E">output = 0.00</text><text id="tc-slope" x="92" y="94" fill="#5A544E">slope = 0.00</text><text x="34" y="140" fill="#6B645E" font-size="9">gradient reaching its weight (bar stretched, so a sliver still shows)</text><rect x="34" y="146" width="196" height="14" rx="3" fill="#EFE9DF"/><rect id="tc-bar" x="34" y="146" width="2" height="14" rx="3" fill="#C93B3B"/><text id="tc-verdict" x="132" y="180" text-anchor="middle" fill="#C93B3B" font-size="10">dead — zero gradient, forever</text><text id="tc-verdict2" x="132" y="196" text-anchor="middle" fill="#6B645E" font-size="9">a rescue would need a gradient it cannot get</text><line x1="284" y1="176" x2="496" y2="176" stroke="#E5DFD6" stroke-width="1.5"/><g id="tc-bars"><rect id="tc-e0" x="292" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e1" x="326" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e2" x="360" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e3" x="394" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e4" x="428" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e5" x="462" y="174" width="26" height="2" fill="#2D8B55"/></g><line id="tc-clipline" x1="284" y1="130.7" x2="496" y2="130.7" stroke="#1a5c38" stroke-width="1.6" stroke-dasharray="4,3" stroke-opacity="0"/><text id="tc-cliplab" x="496" y="126.7" text-anchor="end" fill="#1a5c38" font-size="8.5" opacity="0">clip ceiling</text><g font-size="7.5" text-anchor="middle" fill="#6B645E" paint-order="stroke" stroke="#FDF9F3" stroke-width="1.5" stroke-linejoin="round"><text id="tc-ev0" x="305" y="170">·</text><text id="tc-ev1" x="339" y="170">·</text><text id="tc-ev2" x="373" y="170">·</text><text id="tc-ev3" x="407" y="170">·</text><text id="tc-ev4" x="441" y="170">·</text><text id="tc-ev5" x="475" y="170">·</text></g><text x="388" y="188" text-anchor="middle" fill="#9A938A" font-size="8.5">layers travelled back: 1 · 2 · 3 · 4 · 5 · 6</text><text id="tc-everdict" x="388" y="202" text-anchor="middle" fill="#2D8B55" font-size="10">healthy — the signal survives</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">
<label>shove z <input id="tc-zs" type="range" min="-4" max="4" step="0.2" value="-1" style="width:150px;accent-color:#C99A12;vertical-align:middle"></label>
<label><input id="tc-leaky" type="checkbox"> use <b>Leaky</b> ReLU</label>
</div>
<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:6px">
<label>weight size <input id="tc-ws" type="range" min="0.5" max="3" step="0.1" value="1" style="width:150px;accent-color:#C99A12;vertical-align:middle"></label>
<label><input id="tc-clip" type="checkbox"> <b>clip</b> the gradient at 5</label>
</div>
<div style="margin-top:6px;color:#6B645E;font-size:.92em">Left: drag `z` below 0 with plain ReLU and the gradient bar empties — that unit is finished. Tick Leaky and the same unit keeps a sliver. Right: the bars are the gradient after 1…6 layers when each layer multiplies by the weight size; the printed numbers are the true values. With clip ticked, only the bars that actually reach the ceiling are held there — the earlier, smaller ones are left alone — and the numbers then show the clipped size the optimizer would really use, not the raw one.</div>
</div>
<script>(function(){
  var zs=document.getElementById('tc-zs');if(!zs)return;
  var leaky=document.getElementById('tc-leaky'),ws=document.getElementById('tc-ws'),clip=document.getElementById('tc-clip'),
      knob=document.getElementById('tc-knob'),state=document.getElementById('tc-state'),
      tz=document.getElementById('tc-z'),tout=document.getElementById('tc-out'),tsl=document.getElementById('tc-slope'),
      bar=document.getElementById('tc-bar'),ver=document.getElementById('tc-verdict'),ver2=document.getElementById('tc-verdict2'),
      cline=document.getElementById('tc-clipline'),clab=document.getElementById('tc-cliplab'),ever=document.getElementById('tc-everdict');
  var eb=[],ev=[];
  for(var i=0;i<6;i++){eb.push(document.getElementById('tc-e'+i));ev.push(document.getElementById('tc-ev'+i));}
  var LEAK=0.01, INCOMING=1.0, BASE=176, H=120, CLIP=5;
  function hgt(p){var h=H*(Math.log(p)/Math.LN10+1)/4.5;return Math.max(2,Math.min(H,h));}
  function paintUnit(){
    var z=+zs.value, on=(z>=0), lk=leaky.checked;   /* c5 shows slope 1 at z=0 — match it */
    var out=on?z:(lk?LEAK*z:0), slope=on?1:(lk?LEAK:0), grad=INCOMING*slope;
    knob.setAttribute('y',on?'50':'74');knob.setAttribute('fill',on?'#2D8B55':'#C93B3B');
    state.textContent=on?'ON':'OFF';state.setAttribute('fill',on?'#2D8B55':'#C93B3B');
    tz.textContent='z = '+z.toFixed(1);
    tout.textContent='output = '+out.toFixed(3);
    tsl.textContent='slope = '+slope.toFixed(2);
    var w=(grad<=0)?0:Math.max(8,Math.sqrt(grad)*196);
    bar.setAttribute('width',w.toFixed(1));
    if(on){bar.setAttribute('fill','#2D8B55');ver.setAttribute('fill','#2D8B55');
      ver.textContent='alive — full gradient gets through';ver2.textContent='this unit is learning normally';}
    else if(lk){bar.setAttribute('fill','#C99A12');ver.setAttribute('fill','#C99A12');
      ver.textContent='shoved, but NOT dead — gradient '+grad.toFixed(2);ver2.textContent='the trickle can push z back above 0 · that is Leaky ReLU';}
    else {bar.setAttribute('fill','#C93B3B');ver.setAttribute('fill','#C93B3B');
      ver.textContent='dead — zero gradient, forever';ver2.textContent='a rescue would need a gradient it cannot get';}
  }
  function paintChain(){
    var w=+ws.value, doClip=clip.checked, p=1, raw=1, last=1, didClip=false;
    cline.setAttribute('stroke-opacity',doClip?'1':'0');clab.setAttribute('opacity',doClip?'1':'0');
    for(var i=0;i<6;i++){
      p=p*w; raw=raw*w;
      if(doClip&&p>CLIP){p=CLIP;didClip=true;}
      last=p;
      // Colour by the SAME bands as the verdict below and as the c6 ladder, on the value the
      // bar actually shows. No special case for a clipped bar: clipping bounds the step, it
      // does not turn a big gradient into a healthy one, so a bar pinned at the ceiling stays
      // in the band its number falls in. Otherwise a clipped bar reads green while the
      // verdict quoting that same number reads red.
      var h=hgt(p),col=(p>3||p<0.05)?'#C93B3B':(p<0.5?'#C99A12':'#2D8B55');
      eb[i].setAttribute('y',(BASE-h).toFixed(1));eb[i].setAttribute('height',h.toFixed(1));
      eb[i].setAttribute('fill',col);
      // With the ceiling showing, a tall bar's value label would land on the same line as
      // the "clip ceiling" text (both at y ~ 127), so print it INSIDE the bar. Use dark ink,
      // not white: white on the amber fill (#C99A12) is only ~2.6:1 at this size. The group
      // carries a light halo (paint-order stroke) so dark ink stays legible on green, amber
      // and red alike.
      var crowded=doClip&&(BASE-h-3)<135;
      ev[i].setAttribute('y',(crowded?(BASE-h+10):(BASE-h-3)).toFixed(1));
      ev[i].textContent=(p>=100?p.toExponential(1):p.toFixed(p<0.1?3:(p<10?2:1)));   /* 3dp under 0.1 so 0.046656 prints 0.047, not a "0.05" that straddles the band edge */
      ev[i].setAttribute('fill',crowded?'#1F1B16':col);
    }
    // Bands are the c6 ladder exactly (>3 exploded, <0.05 vanished, <0.5 fading), so the same
    // product cannot read "exploded" in one widget and "healthy" in the other.
    // Clipping is NOT its own band. The ceiling (5) sits above the exploded line (3), so a
    // "clipped, all sane now" verdict would contradict the red bars it is sitting under.
    // Clipping bounds the STEP; it does not make the gradient healthy. So report the raw
    // product that WOULD have arrived, and say what clipping did to it.
    if(didClip){ever.setAttribute('fill','#C93B3B');
      ever.textContent='would have hit '+(raw>=100?raw.toExponential(1):raw.toFixed(1))
        +' → clipped to '+CLIP+' before the step';}
    else if(last>3){ever.setAttribute('fill','#C93B3B');
      ever.textContent='exploded → '+(last>=100?last.toExponential(1):last.toFixed(1))+' · one step overshoots the valley';}
    else if(last<0.05){ever.setAttribute('fill','#C93B3B');
      ever.textContent='vanished → '+last.toExponential(1)+' · the early layers get nothing';}
    else if(last<0.5){ever.setAttribute('fill','#C99A12');
      ever.textContent='fading → '+last.toFixed(2)+' · the early layers learn slowly';}
    else {ever.setAttribute('fill','#2D8B55');ever.textContent='healthy — the signal survives';}
  }
  function paint(){paintUnit();paintChain();}
  zs.addEventListener('input',paint);ws.addEventListener('input',paint);
  leaky.addEventListener('change',paint);clip.addEventListener('change',paint);
  paint();
})();</script>
%%%

**What you just made happen:** on the left, a negative `z` with plain ReLU empties the bar to exactly `0` — and once it's `0`, nothing can refill it. Tick **Leaky** and the same shoved unit keeps a sliver of gradient, enough to climb back. **Sensible init** helps too: start the weights at reasonable sizes and units don't get shoved off the cliff on day one.

#### The squeal — the **exploding gradient**
Now the right-hand dial. Crank the weight size up and the running product from Rule 4 stops shrinking and starts snowballing:

%%% steps
step: the squeal — the numbers each layer multiplies by are big instead of tiny, so the running product grows
why: same chain of multiplies as the fade, just pointed the other way
step: the blow-up — a few layers of that and you get **huge values**: this is the **exploding gradient**
why: the gradient is still "correct" — it is just enormous, and one giant step **overshoots** the whole valley, so the loss often comes back as `inf` or `NaN`
step: the fix — **gradient clipping**: set a ceiling and **clip** anything above it before you step
why: tick the clip box and every bar that reaches the ceiling stops right there — one monster number can never launch you off the hill again
%%%

%%% demo id=explode label="run it — the squeal in six numbers"
predict: multiply the gradient by 3 at each of six layers. Does it reach roughly 18, roughly 100, or over 700?
code: g=1.0; big=3.0; [round(g:=g*big,1) for _ in range(6)]  # gradient multiplied by a big amount each layer
out: [3.0, 9.0, 27.0, 81.0, 243.0, 729.0]
take: <b>Big multiplies drive the gradient toward infinity.</b> ×3 per layer and the gradient rockets from 3 to 729 in six steps — a step that size overshoots wildly, and the loss often prints as inf or NaN. Clip it, or size the starting weights so the per-layer multiply sits near 1.
%%%

And **careful init** is the other half of the cure: sensible starting weights keep the per-layer multiplies near `1` in the first place, so the product neither fades nor explodes.

%%% insight
Notice the pattern across both dials you just drove: nobody had to change the *neuron*. Both cures change the rules the gradient travels under — leave a trickle open, or put a ceiling on. That's how almost every training fix in this field works, right up to the models you use every day.
%%%

#### One dial, four ways to go wrong
Here's the payoff for sitting through the puzzles. They are not four things to memorise — they are one number behaving badly. Every layer the gradient passes through multiplies it by something (the activation's slope times the weight); the whole story is *what that something is*:

%%% svg
<svg viewBox="0 0 520 186" role="img" aria-label="One dial showing the per-layer multiplier from 0 to above 1. At exactly 0 the label reads dead unit, cured by Leaky ReLU. Below 1 the label reads the fade, cured by ReLU. At about 1 the label reads healthy, the signal survives. Above 1 the label reads explode, cured by clipping. A separate note below shows that if every unit multiplies by the same value, they become twins, cured by small random init."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The same multiply, four ways — what does each layer multiply by?</text><line x1="40" y1="76" x2="480" y2="76" stroke="#B8AEA2" stroke-width="2"/><g stroke="#B8AEA2"><line x1="40" y1="70" x2="40" y2="82"/><line x1="200" y1="70" x2="200" y2="82"/><line x1="330" y1="70" x2="330" y2="82"/><line x1="480" y1="70" x2="480" y2="82"/></g><text x="40" y="96" text-anchor="middle" fill="#6B645E" font-size="9">0</text><text x="200" y="96" text-anchor="middle" fill="#6B645E" font-size="9">under 1</text><text x="330" y="96" text-anchor="middle" fill="#6B645E" font-size="9">≈ 1</text><text x="480" y="96" text-anchor="middle" fill="#6B645E" font-size="9">over 1</text><circle cx="40" cy="76" r="7" fill="#C93B3B"/><text x="40" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">dead unit 🔌</text><text x="40" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">→ Leaky ReLU</text><circle cx="200" cy="76" r="7" fill="#C99A12"/><text x="200" y="46" text-anchor="middle" fill="#C99A12" font-size="10">the fade 📞</text><text x="200" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">→ ReLU (slope 1)</text><circle cx="330" cy="76" r="7" fill="#2D8B55"/><text x="330" y="46" text-anchor="middle" fill="#2D8B55" font-size="10">healthy 🎉</text><text x="330" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">signal survives</text><circle cx="480" cy="76" r="7" fill="#C93B3B"/><text x="470" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">explode 🎤</text><text x="470" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">→ clip it</text><line x1="40" y1="122" x2="480" y2="122" stroke="#E5DFD6"/><text x="46" y="144" fill="#C93B3B" font-size="10">👯 and if EVERY unit multiplies by the very same thing → twins</text><text x="46" y="160" fill="#2D8B55" font-size="10">→ small random starting weights, so the units differ</text><text x="260" y="180" text-anchor="middle" fill="#6B645E" font-size="9.5">aim for multiplies near 1, plus a little healthy variety between units</text></g></svg>
%%%

!!! c-info 🧭
<b>Keep it straight:</b> all four traps are the same math seen from different angles. <b>Fade</b> = multiplying by numbers under 1. <b>Explode</b> = multiplying by numbers over 1. <b>Dead unit</b> = multiplying by exactly 0. <b>Twins</b> = every unit multiplying by the <i>same</i> thing. The cures all aim for the same sweet spot: per-layer multiplies near 1, and a little healthy variety between units.
!!!

**Four traps, four quick cures — and that was the hardest stretch of the day, now behind you.** One page to tie it all together.

@@@ concept id=c10 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you just learned how a network figures out which way to improve, which is genuinely the core idea of modern AI. Here's a way to hold the whole day at once: picture our **hiker in fog** one more time, and read the beats below as the steps of a single descent, in order. Each step only made sense because of the one before it — you can't trace blame backward until you've saved your work going forward. **Where the hiker picture breaks down:** a hiker takes the downhill step herself, but today you only learned how to *find* the downhill direction; actually taking the step is tomorrow's job (the optimizer).

**The trail you walked, in order:**

- **Step 1 · The downhill arrow.** The **gradient** answers "which way is downhill, for every weight?" It points **uphill** (more error), so to learn we step the **opposite** way, downhill — that's the **minus** sign in tomorrow's update.
- **Step 2 · Save your work.** The **forward pass** makes a guess and quietly saves its in-between values (`z`, `a`) — the sticky notes the backward trip re-uses.
- **Step 3 · Chain of effects.** A **local slope** is one tiny step's effect; the **chain rule** multiplies the local slopes along the path to get the whole effect.
- **Step 4 · Trace the blame.** **Backpropagation** starts at the loss and sweeps backward, multiplying local slopes, handing every weight its gradient in **one** pass — the reason big networks are trainable at all.
- **Step 5 · Each part's share.** Four tiny rules, in backward order: the **activation charges a toll** (multiply by its slope at the saved `z`, giving `δ`); the **weight's share = its input × δ** (big inputs → big updates); the **bias travels free** (slope 1, so the gradient passes straight through); and to travel **one layer further back**, multiply by the **weight** (`δ × w`).
- **Step 6 · The fading signal.** **Vanishing**: per-layer multiplies below 1 (sigmoid's slope is ≤ 0.25) stack up and drive the gradient toward 0, so early layers freeze — cure: **ReLU**, whose slope stays 1, plus sanely sized starting weights.
- **Step 7 · Check your work.** The **gradient check** measures the same slope a second way (nudge ±ε, divide) and compares — 30 seconds that catch a silent bug.
- **Step 8 · The twin trap.** Start every weight at the same value and every neuron gets **identical gradients**, so they **stay identical** forever — cure: **small random** starting weights to break the symmetry.
- **Step 9 · Stuck and squealing.** **Dead ReLU** (unit shoved permanently negative, slope 0, never learns → **Leaky ReLU**, a small negative slope, plus sensible init) and **exploding** (big multiplies snowball to huge values → **clip** the gradient + careful init).

Here's the one picture to keep — the whole flow, forward then back.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Recap flow: forward pass goes right from input through weighted sum to activation to loss, saving z and a; backward pass returns left multiplying local slopes to give each weight its gradient. Below, four traps and their cures are listed."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The whole day: forward saves, backward traces blame</text><rect x="16" y="30" width="60" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="46" y="47" text-anchor="middle" fill="#1F6280">input</text><rect x="120" y="30" width="86" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="163" y="47" text-anchor="middle" fill="#9A7208">sum → z 📌</text><rect x="250" y="30" width="86" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text x="293" y="47" text-anchor="middle" fill="#5E5191">act → a 📌</text><rect x="380" y="30" width="86" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="423" y="47" text-anchor="middle" fill="#C93B3B">loss</text><g stroke="#2A7B9B" stroke-width="1.8" fill="#2A7B9B"><path d="M76 43 l40 0"/><polygon points="116,43 108,39 108,47"/><path d="M206 43 l40 0"/><polygon points="246,43 238,39 238,47"/><path d="M336 43 l40 0"/><polygon points="376,43 368,39 368,47"/></g><text x="250" y="24" text-anchor="middle" fill="#2A7B9B" font-size="9">forward →</text><g stroke="#2D8B55" stroke-width="1.8" fill="#2D8B55"><path d="M380 70 l-40 0"/><polygon points="340,70 348,66 348,74"/><path d="M250 70 l-40 0"/><polygon points="210,70 218,66 218,74"/><path d="M120 70 l-40 0"/><polygon points="80,70 88,66 88,74"/></g><text x="250" y="86" text-anchor="middle" fill="#2D8B55" font-size="9">← backward: × local slopes → each weight's gradient</text><line x1="30" y1="100" x2="490" y2="100" stroke="#E5DFD6"/><text x="30" y="120" fill="#C93B3B">fade → ReLU</text><text x="160" y="120" fill="#C93B3B">dead → Leaky ReLU</text><text x="330" y="120" fill="#C93B3B">explode → clip</text><text x="30" y="140" fill="#C93B3B">twins → small random init</text><text x="240" y="140" fill="#2D8B55">check → measure the slope two ways</text><text x="260" y="164" text-anchor="middle" fill="#6B645E">aim for per-layer multiplies near 1, a little variety between units</text></g></svg>
%%%

#### Cheat-sheet · each part's gradient, and the traps

%%% table
:: Part / trap :: In one line :: Remember
activation :: multiplies the gradient by its slope at the saved `z`, giving `δ` :: sigmoid ≤ 0.25 (tiny in tails), ReLU is 1 or 0
weight gradient :: its input × `δ` (the gradient after the activation's toll) :: big input → big update
bias gradient :: `δ`, unchanged :: slope 1 → passes straight through
one layer further back :: `δ × w` — multiply by the weight you came through :: this is why weight SIZE decides fade vs explode
vanishing :: per-layer multiplies under 1 stack up toward 0 :: cure: ReLU (slope stays 1) + sane weight sizes
dead ReLU :: unit stuck negative, slope 0 forever :: cure: Leaky ReLU (small negative slope) + sensible init
exploding :: per-layer multiplies over 1 snowball toward ∞ :: cure: gradient clipping + careful init
same-value init (twins) :: identical weights get identical gradients :: cure: small random init
gradient check :: measure the slope a second way and compare :: use it to check, never to train
%%%

#### Cheat-sheet · the words you met today

%%% jargon
gradient | the arrow saying which way to nudge every weight to change the loss fastest; it points uphill, so we step the opposite way
forward pass | running the input through the neuron to a guess, saving z and a for the trip back
computation graph | the neuron's math seen as a chain of tiny steps, each with an easy local slope
local derivative | the slope of just one tiny step — how much its own output moves when its own input moves a hair
chain rule | multiply the local slopes along the path to get how the loss depends on an early weight
upstream gradient | the gradient arriving at a step from the loss side (the "error signal"); × the step's local slope to pass it further back
delta (δ) | the gradient that got through the activation's toll — what the weight and bias then share
backpropagation | one backward sweep from the loss that multiplies local slopes to give every weight its gradient
vanishing gradient | per-layer multiplies below 1 (sigmoid's slope ≤ 0.25) stacked over many layers shrink the gradient toward 0, so early layers stop learning
dead ReLU | a unit stuck outputting 0 with slope 0, so it never learns again → cure: Leaky ReLU
exploding gradient | per-layer multiplies above 1 blow the gradient up huge → cure: gradient clipping
gradient check | measure a slope by nudging a weight ±ε and compare it to backprop, to catch bugs
%%%

That's the whole day. Tomorrow you take the downhill step for real — the **training loop**, where these gradients actually change the weights, over and over, until the network learns.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)

%%% quiz
q: What does the gradient tell the network, and which way do we step? | a:2 | The final loss value; we step toward it | How many layers to use; we step randomly | Which way is uphill (more error) for every weight; we step the opposite way, downhill | The learning rate; we don't step | fb: The gradient points UPHILL — the direction that makes the loss bigger. So learning means stepping the opposite way, downhill, which is exactly why tomorrow's update carries a minus sign. (A very common mix-up is to say it "points downhill" — it doesn't; we flip it.)
q: What is the chain rule, in one line? | a:1 | Add up the loss of every layer | Multiply the local slopes along the path to get how an early weight affects the far-away loss | Pick the layer with the biggest weight | Run the network backward with new random weights | fb: Multiply, don't add. Pedal → speed → distance → money: three easy one-hop slopes multiplied give the whole effect, and that single move is the engine of backprop.
q: Why do deep sigmoid networks suffer from vanishing gradients, and what's the common cure? | a:2 | Sigmoid uses too much memory; cure: bigger batches | The loss is negative; cure: flip its sign | The sigmoid's slope is at most 0.25, so multiplying it in at every layer shrinks the gradient toward 0; cure: ReLU, whose slope stays 1 for positive inputs | Sigmoid outputs are too large; cure: clip them | fb: Every sigmoid layer multiplies the gradient by at most 0.25, so the whisper fades about 4× per layer. ReLU's slope is a clean 1 for positive inputs, so the activation stops eating the signal — the weights still have to be sized sensibly for the rest.
q: A ReLU unit is stuck outputting 0 with slope 0 and never learns. What is this, and how do you fix it? | a:1 | Exploding gradient; fix by lowering the learning rate | A dead ReLU; fix with Leaky ReLU, which keeps a small nonzero slope on the negative side, plus sensible weight initialization | Vanishing gradient; fix with a bigger network | Zero-init symmetry; fix by clipping | fb: Slope 0 means gradient 0, so the unit can never climb back — a dead ReLU. Leaky ReLU keeps a small rising slope (about 0.01) on the negative side, so a trickle of gradient always gets through.
%%%

@@@ produce id=produce tag="Produce" title="Watch a gradient flow backward — and check it two ways" gotit="Done"
Time to see today's big idea with your own eyes. You'll run a tiny neuron forward, trace the gradient backward through it by hand with the rules, and then *check your work* by measuring the same slope a totally different way. **Predict first:** for a neuron with input `x = 2` and a positive `z`, will the *weight's* gradient be bigger or smaller than the *bias's* gradient — and by what factor? Then run it and **watch** the weight's gradient come out exactly `x` times the bias's, because the weight's blame is scaled by its input while the bias just passes `δ` through. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py`. (1) Do a forward pass for one neuron: `z = w*x + b` then `a = relu(z)`, and a squared-error loss `L = (a - target)**2`; *save* `x`, `z`, `a` as you go. (2) Backprop by hand: the gradient arriving from the loss is `incoming = 2*(a-target)`, the ReLU slope is `1` if `z>0` else `0`, so Rule 1 gives `delta = incoming * relu_slope`, then Rule 2 gives `w_grad = delta * x` and Rule 3 gives `b_grad = delta` — **notice** `w_grad` is `x` times `b_grad`. (3) **Check it:** write `loss_of_w(w)` and compute the measured slope `(loss_of_w(w+1e-5) - loss_of_w(w-1e-5)) / (2e-5)`, then print it next to your `w_grad` and **watch** them match. Run with `python3 sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 5 artifact.
Create sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py that, with a comment on each step:
1. Runs a forward pass for one neuron: given x=2.0, w=0.5, b=0.1, target=1.0, compute z=w*x+b, a=relu(z) (relu = max(0,z)), and loss L=(a-target)**2; save x, z, a.
2. Backprops by hand with the lesson's rules: incoming = 2*(a-target); relu_slope = 1.0 if z>0 else 0.0; Rule 1 delta = incoming*relu_slope; Rule 2 w_grad = delta*x; Rule 3 b_grad = delta. Print all three and show w_grad == x * b_grad (big input -> big weight update).
3. Gradient-checks w_grad: define loss_of_w(w) that redoes the forward pass and returns L, then measured = (loss_of_w(w+1e-5) - loss_of_w(w-1e-5))/(2e-5); print measured next to w_grad and show they match to ~4 decimals.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the gradient, traced and verified)
- The **weight's gradient is exactly `x` times the bias's** — with `x = 2`, twice as big — because a weight's blame is scaled by its input while the bias passes `δ` straight through.
- Flip `z` negative (try a big negative `b`) and **both gradients become `0`** — the ReLU slope is `0` there, so that unit gets no signal at all. That's a dead ReLU, live in front of you.
- The **measured slope matches your hand-computed `w_grad`** to about four decimals — proof your backward pass is correct, exactly the check a careful engineer runs before trusting the code.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-05-gradients-backprop/log.md</code>: (1) what the gradient is, and which way we step; (2) why the forward pass saves its in-between values; (3) one trap from today (vanishing, dead ReLU, exploding, or twins) and its one-line cure.
!!!

@@@ fin

