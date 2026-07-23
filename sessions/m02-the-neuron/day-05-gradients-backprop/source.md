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
fin_body: "Nice work — you've met the <b>gradient</b>, the arrow that points straight downhill on the loss surface, and <b>backpropagation</b>, the chain-rule trick that hands every weight its own share of the blame. Now the network knows which way to step.<br>Next up: <b>The Training Loop</b> — where these steps repeat until the network learns."
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
@lede Imagine you're standing on a hillside in thick **fog**. You can't see the valley, but you want to walk down to the lowest point. So you do the one thing you still can: you feel the ground right under your boots, notice which way it tips *down*, and take a step that way. Then you feel again, and step again. That simple move — *feel the slope, step downhill* — is the whole secret behind how every neural network learns, from a tiny one you'll build today all the way up to the giant one behind ChatGPT. Yesterday you built the network's **score** — one number for how wrong its guess was. Today you learn the magic that reads that score and whispers to every single knob inside the network: *"nudge this way to make the score a little lower."* That whisper has a name — the **gradient** — and the clever trick that computes it for thousands of knobs at once is called **backpropagation**. By the end you'll know exactly which way is downhill.
@goal Together we'll turn a foggy hillside into a precise recipe. You'll meet the **gradient** (the arrow pointing downhill), watch the network **save its work** on the way forward so it can trace blame on the way back, learn the one multiplying trick — the **chain rule** — that hands every knob its fair share of the blame, and see the whole **backward pass** sweep through in a single pass. Then we'll poke at a few sneaky ways the downhill signal fades, dies, or explodes — each a little puzzle with a neat one-line fix — and finish with a trick to check your own work. Every formula shows up in plain words first; any heavy math sits in a skippable box.

@@@ concept id=c1 tag="The downhill arrow" title="The gradient — which way is downhill?" gotit="Got the gradient"
Let's start on that foggy hillside, before any math. You're a **hiker in fog**, and the ground under your boots is the network's mistake: high ground means a very wrong guess, low ground means a good one. Yesterday's [[loss||one number saying how wrong the network's guess was; lower is better, 0 is perfect]] is your *height* on this hill — call it the **error signal**, since it's the one number that tells you how wrong you are right now. You can't see the whole valley through the fog. But you *can* feel the tilt of the ground right where you stand, and step the way it tips down. Do that again and again and you'll reach the bottom — the lowest mistake.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A hiker standing in fog on a hillside. Grey fog hides the valley. Under the hiker's boots a small patch of ground is visible, tipping downhill. A red arrow points down the slope, labelled the gradient points the steepest way down; height is labelled how wrong the guess is."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A hiker in fog: feel the tilt underfoot, step downhill</text><path d="M20 60 C 140 60, 260 150, 500 160" fill="none" stroke="#B8AEA2" stroke-width="2"/><rect x="20" y="24" width="480" height="40" fill="#EDEAE4" opacity="0.85"/><text x="260" y="48" text-anchor="middle" fill="#9A938A" font-size="10">— fog · you can't see the valley —</text><text x="150" y="70" font-size="20">🥾</text><circle cx="160" cy="92" r="30" fill="none" stroke="#7C6DAA" stroke-width="1.3" stroke-dasharray="3,3"/><text x="160" y="130" text-anchor="middle" fill="#7C6DAA" font-size="9">visible patch underfoot</text><path d="M170 96 l40 22" stroke="#C93B3B" stroke-width="2.5"/><polygon points="210,118 199,113 202,124" fill="#C93B3B"/><text x="238" y="118" fill="#C93B3B" font-size="10">the gradient →</text><text x="238" y="132" fill="#C93B3B" font-size="10">steepest way DOWN</text><text x="58" y="150" fill="#6B645E" font-size="10">high = very wrong</text><text x="428" y="176" fill="#2D8B55" font-size="10">low = good guess</text></g></svg>
%%%

**What the hiker-in-fog picture gets right:** you only ever need the *local* tilt — the ground right under your boots — to know which way to step, and you reach the bottom one small step at a time. That is exactly how a network learns. **Where it breaks down:** a real hill has just two directions to lean (north–south, east–west), but a real network has *thousands* of knobs, so its "hill" tips in thousands of directions at once. The gradient handles all of them together — one downhill arrow with a component for every knob.

#### The gradient, in plain words
Each knob inside the neuron is a [[weight||a number the neuron multiplies its input by; learning means finding good values for the weights]] (or a bias). The [[gradient||the arrow that says, for every weight at once, which way to nudge it to change the loss the fastest — and how steeply]] answers one question for every weight: *"if I nudge this weight up a hair, does the mistake go up or down, and how fast?"* Collect that answer for every weight and you get the gradient — a list of slopes, one per knob. It points the **steepest way uphill** (toward *more* mistake). So to go *down*, you do the obvious thing: step the **opposite** way, against the arrow.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A single dip-shaped loss curve. A dot sits on the right side. A red arrow points up-and-right along the slope labelled gradient points uphill. A green arrow points the opposite way, down toward the bottom of the dip, labelled we step the opposite way, downhill."><g font-family="monospace" font-size="11"><path d="M40 30 C 160 150, 360 150, 480 30" fill="none" stroke="#B8AEA2" stroke-width="2.5"/><circle cx="410" cy="92" r="6" fill="#C99A12" stroke="#fff" stroke-width="2"/><path d="M416 88 l38 -28" stroke="#C93B3B" stroke-width="2.5"/><polygon points="454,60 443,61 448,70" fill="#C93B3B"/><text x="440" y="52" fill="#C93B3B" text-anchor="middle" font-size="10">gradient → uphill (more error)</text><path d="M404 96 l-40 26" stroke="#2D8B55" stroke-width="2.5"/><polygon points="364,122 375,120 370,111" fill="#2D8B55"/><text x="300" y="140" fill="#2D8B55" font-size="10">we step the OPPOSITE way → downhill</text><text x="258" y="120" text-anchor="middle" fill="#6B645E" font-size="10">the bottom = smallest mistake</text><text x="40" y="24" fill="#6B645E" font-size="10">loss (height) ↑</text></g></svg>
%%%

The one line to remember: **the gradient points uphill, and to learn we step the opposite way — downhill.** That's the same *feel the slope, step down* move as our hiker, just written for thousands of knobs at once. The rest of today is one question: *how does the network actually feel that slope for every knob?* That's the trick called backpropagation, and it starts with something the neuron did yesterday.

@@@ concept id=c2 tag="Save your work" title="The forward pass — and why it saves its notes" gotit="Got the forward pass"
Before a network can trace its mistake backward, it has to make a guess *forward* first — and, quietly, keep a few notes along the way. Think about **baking a cake**. You add flour, then sugar, then eggs, then bake — one step feeds the next. If the cake comes out wrong, you don't throw out the kitchen; you look back at your notes — *"how much sugar did I add? how long did it bake?"* — to figure out which step to change. If you didn't write anything down, you'd be guessing blind.

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A baking assembly line: flour then sugar then eggs then bake then taste. Under each step a small sticky note records what was used. A caption says keep the notes so if the cake is wrong you know which step to fix."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Baking forward — and keeping a note at every step</text><rect x="24" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="61" y="59" text-anchor="middle" fill="#9A7208">flour</text><text x="104" y="59" fill="#B8AEA2">→</text><rect x="120" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="157" y="59" text-anchor="middle" fill="#9A7208">sugar</text><text x="200" y="59" fill="#B8AEA2">→</text><rect x="216" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="253" y="59" text-anchor="middle" fill="#9A7208">eggs</text><text x="296" y="59" fill="#B8AEA2">→</text><rect x="312" y="40" width="74" height="30" rx="5" fill="#F3ECDB" stroke="#C99A12"/><text x="349" y="59" text-anchor="middle" fill="#9A7208">bake</text><text x="392" y="59" fill="#B8AEA2">→</text><rect x="408" y="40" width="88" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="452" y="59" text-anchor="middle" fill="#1a5c38">taste 😋</text><g><rect x="30" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text x="61" y="107" text-anchor="middle" fill="#5E5191" font-size="9">2 cups</text><rect x="126" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text x="157" y="107" text-anchor="middle" fill="#5E5191" font-size="9">1 cup</text><rect x="222" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text x="253" y="107" text-anchor="middle" fill="#5E5191" font-size="9">3 eggs</text><rect x="318" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text x="349" y="107" text-anchor="middle" fill="#5E5191" font-size="9">30 min</text></g><text x="260" y="140" text-anchor="middle" fill="#6B645E">the sticky notes = the saved values — so a wrong cake tells you which step to fix</text></g></svg>
%%%

**What the baking picture gets right:** the steps run in order, each feeding the next, and *keeping notes* is what lets you trace a bad result back to its cause. **Where it breaks down:** a baker keeps notes for a person to read later; the network keeps its notes for *itself* — it'll re-use them a fraction of a second later, on the way back. Nothing is thrown away until the backward trace is done.

#### The neuron's forward pass, step by step
The [[forward pass||running the input through the neuron to get a guess — and saving the in-between values for the trip back]] is just the neuron making its guess, the same recipe you built earlier this module. It runs the input through two small steps and then scores it:

- **Step 1 — the weighted sum.** Multiply each input by its weight, add the bias. This in-between number has a name, the [[pre-activation||the weighted sum plus bias, before the activation bends it; often written z]] (often called `z`).
- **Step 2 — the activation.** Bend `z` with the activation (like sigmoid or ReLU from Day 2) to get the neuron's output `a` — its guess.
- **Step 3 — the loss.** Compare the guess `a` to the true answer to get today's mistake number.

Here's the quiet, crucial part: at each step the neuron **saves the value it just made** — the input, `z`, and `a`. Those saved notes are exactly the sticky notes on the cake. In a moment you'll see the trip backward *re-use* every one of them.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="The neuron's forward pass as a conveyor: input x flows into weighted-sum-plus-bias giving z, then activation giving a, then loss. Under z and a, saved-note tags show the values are stored for the backward trip."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Forward pass → guess, saving z and a as it goes</text><rect x="16" y="42" width="72" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text x="52" y="61" text-anchor="middle" fill="#1F6280">input x</text><text x="94" y="61" fill="#B8AEA2">→</text><rect x="112" y="42" width="118" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="171" y="57" text-anchor="middle" fill="#9A7208" font-size="9.5">weighted sum + bias</text><text x="171" y="68" text-anchor="middle" fill="#9A7208" font-size="9">= z</text><text x="236" y="61" fill="#B8AEA2">→</text><rect x="254" y="42" width="96" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text x="302" y="57" text-anchor="middle" fill="#5E5191" font-size="9.5">activation</text><text x="302" y="68" text-anchor="middle" fill="#5E5191" font-size="9">= a (guess)</text><text x="356" y="61" fill="#B8AEA2">→</text><rect x="374" y="42" width="126" height="30" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="437" y="61" text-anchor="middle" fill="#C93B3B" font-size="9.5">loss (how wrong)</text><rect x="130" y="96" width="82" height="24" rx="3" fill="#F3ECDB" stroke="#C99A12" stroke-dasharray="3,2"/><text x="171" y="112" text-anchor="middle" fill="#9A7208" font-size="9">📌 saved z</text><rect x="266" y="96" width="72" height="24" rx="3" fill="#EFEAF7" stroke="#B3A6D4" stroke-dasharray="3,2"/><text x="302" y="112" text-anchor="middle" fill="#5E5191" font-size="9">📌 saved a</text><text x="260" y="146" text-anchor="middle" fill="#6B645E">these saved notes get re-used on the trip back — nothing recomputed</text></g></svg>
%%%

There's a second gift hiding here. Writing the neuron as a *chain of small steps* — multiply, add, bend, score — turns it into a [[computation graph||a picture of the math as a chain of tiny operations, so each little step has a clear, simple slope of its own]]: a little assembly line where each box does one tiny thing. That matters because each *tiny* step has a *tiny, easy* slope — and stitching those easy slopes together is the whole backward trick. On to the stitch.

@@@ concept id=c3 tag="Chain of effects" title="Local slopes and the chain rule" gotit="Got the chain rule"
Here's the one idea that makes the whole thing tick, and it's something you already understand from planning a **road trip**. Push the gas pedal a little harder → your *speed* goes up a little. More speed → you cover more *distance* per hour. More distance → you burn more *fuel* → you pay more *money*. Now ask: *"how much does one extra push on the pedal cost me?"* You don't need a mega-formula. You just follow the chain and **multiply** the little effects: how much speed the pedal adds, times how much distance the speed adds, times how much money the distance costs. Multiply the little hops and you get the whole trip.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A road-trip chain: pedal affects speed affects distance affects fuel affects money. Each arrow carries a small local effect. A caption says multiply the little effects along the chain to get how the pedal affects the money."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Road trip: multiply the little effects along the chain</text><rect x="10" y="46" width="72" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text x="46" y="65" text-anchor="middle" fill="#1F6280">🦶 pedal</text><rect x="122" y="46" width="72" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="158" y="65" text-anchor="middle" fill="#9A7208">speed</text><rect x="234" y="46" width="78" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="273" y="65" text-anchor="middle" fill="#9A7208">distance</text><rect x="352" y="46" width="64" height="30" rx="5" fill="#F3ECDB" stroke="#C99A12"/><text x="384" y="65" text-anchor="middle" fill="#9A7208">fuel</text><rect x="446" y="46" width="64" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="478" y="65" text-anchor="middle" fill="#1a5c38">💰 money</text><g stroke="#C93B3B" stroke-width="2" fill="#C93B3B"><path d="M82 61 l38 0"/><polygon points="120,61 112,57 112,65"/><path d="M194 61 l38 0"/><polygon points="232,61 224,57 224,65"/><path d="M312 61 l38 0"/><polygon points="350,61 342,57 342,65"/><path d="M416 61 l28 0"/><polygon points="444,61 436,57 436,65"/></g><text x="46" y="108" fill="#C93B3B" font-size="9">×</text><text x="158" y="108" fill="#C93B3B" font-size="9">×</text><text x="273" y="108" fill="#C93B3B" font-size="9">×</text><text x="384" y="108" fill="#C93B3B" font-size="9">×</text><text x="260" y="132" text-anchor="middle" fill="#6B645E">each arrow = one small "local" effect · multiply them all → pedal's effect on money</text></g></svg>
%%%

**What the road-trip picture gets right:** you never need one giant formula for "pedal → money." You break it into easy one-hop effects and multiply. **Where it breaks down:** on a road trip the effects mostly go one way (more pedal → more money), but inside a network a slope can be negative (nudging a weight *up* can push the loss *down*) — the multiplying still works, the sign just tells you which way to lean.

#### Two words: local slope, and the chain rule
Each little arrow above is a [[local derivative||the slope of just one tiny step — how much its output moves when its own input moves a hair, ignoring the rest of the chain]] — a fancy name for one small, easy question: *"if I nudge this box's input a hair, how much does its output move?"* Just the one step, nothing else. The rule for gluing the local slopes together is the [[chain rule||to get how the far-away loss depends on an early quantity, multiply the local slopes all along the path between them]]: **to get the effect of an early thing on a far-away thing, multiply the local slopes all along the path.**

Watch it with real numbers, one hop at a time — follow the running product:

- **Pedal → speed:** one more push adds `2` mph. *(local slope = 2)*
- **Speed → distance:** each mph adds `1.5` extra miles. Running product so far: `2 × 1.5 = 3`.
- **Distance → money:** each mile costs `$0.5`. Final: `3 × 0.5 = 1.5`.

So one extra push on the pedal costs about **$1.50** — found by multiplying three tiny slopes, never a monster formula. Watch the running product grow with each hop:

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A running-product ladder for the chain rule. Three multiply steps: after pedal to speed the product is 2, after speed to distance it is 3, after distance to money it is 1.5. Bars show the running product changing at each hop."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Running product: multiply one local slope at a time</text><line x1="40" y1="118" x2="500" y2="118" stroke="#E5DFD6" stroke-width="1.5"/><rect x="70" y="58" width="66" height="60" fill="#2A7B9B"/><text x="103" y="52" text-anchor="middle" fill="#1F6280">2.0</text><text x="103" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 2 (pedal→speed)</text><rect x="230" y="28" width="66" height="90" fill="#C99A12"/><text x="263" y="22" text-anchor="middle" fill="#9A7208">3.0</text><text x="263" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 1.5 (speed→dist)</text><rect x="390" y="73" width="66" height="45" fill="#2D8B55"/><text x="423" y="67" text-anchor="middle" fill="#1a5c38">1.5</text><text x="423" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 0.5 (dist→money)</text><text x="163" y="90" text-anchor="middle" fill="#C93B3B" font-size="13">×</text><text x="343" y="90" text-anchor="middle" fill="#C93B3B" font-size="13">×</text></g></svg>
%%%

%%% demo id=chain label="run it — multiply the little slopes"
code: pedal_to_speed=2; speed_to_dist=1.5; dist_to_money=0.5; pedal_to_speed*speed_to_dist*dist_to_money
out: 1.5
take: <b>Chain rule = multiply the local slopes.</b> 2 × 1.5 × 0.5 = 1.5 — one extra push on the pedal costs about $1.50. Break a scary "how does the start affect the end?" question into easy one-hop slopes and multiply. That single move is the engine of backprop.
%%%

One more word you'll need in a moment. As we work back along the chain, the slope we've built up *so far* — the running product arriving at a box from the loss side — is called the [[upstream gradient||the gradient that has already arrived at a step from the loss side; multiply it by the step's own local slope to pass it further back]]. At each box you take that incoming (upstream) number, multiply by the box's own local slope, and hand the result to the next box back. That handed-along number becomes the *downstream* box's upstream gradient — and so the slope flows backward, one easy multiply at a time. That flowing-backward multiply *is* backpropagation, which we meet next.

@@@ concept id=c4 tag="Trace the blame" title="The backward pass — one sweep, all the blame" gotit="Got backprop"
Now the whole trick clicks into place. Imagine a student who just got a test back with a low score. The lazy way to improve is to re-take the entire test after changing *one* study habit, then again after changing the *next* — a fresh full test for every single habit. Exhausting, and most of the work is repeated. The smart student does something better: she starts from the wrong answers and traces the blame *backward* — *"this mistake came from that step, which came from this habit"* — assigning each habit its fair share in **one careful pass** over the graded test. Same answer, a fraction of the effort.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A student tracing blame backward from a low test score to specific study habits, in one pass. On the left a slow way redoes the whole test once per habit; on the right the smart way traces blame from the score back through each step to each habit in a single sweep."><g font-family="monospace" font-size="10"><text x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">Slow: redo the whole test per habit</text><rect x="30" y="30" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="60" y="45" text-anchor="middle" fill="#C93B3B">test #1</text><rect x="30" y="58" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="60" y="73" text-anchor="middle" fill="#C93B3B">test #2</text><rect x="30" y="86" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="60" y="101" text-anchor="middle" fill="#C93B3B">test #3</text><text x="130" y="73" fill="#6B645E">…once per habit 😩</text><line x1="258" y1="24" x2="258" y2="150" stroke="#E5DFD6"/><text x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">Smart: trace blame back ONCE</text><rect x="452" y="44" width="56" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="480" y="61" text-anchor="middle" fill="#C93B3B">score</text><rect x="356" y="44" width="56" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text x="384" y="61" text-anchor="middle" fill="#5E5191">step</text><rect x="278" y="44" width="52" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="304" y="61" text-anchor="middle" fill="#1F6280">habit</text><g stroke="#2D8B55" stroke-width="2.5" fill="#2D8B55"><path d="M452 57 l-40 0"/><polygon points="412,57 420,53 420,61"/><path d="M356 57 l-26 0"/><polygon points="330,57 338,53 338,61"/></g><text x="390" y="96" text-anchor="middle" fill="#2D8B55">one backward sweep 🎯</text><text x="390" y="112" text-anchor="middle" fill="#6B645E">every habit gets its blame at once</text></g></svg>
%%%

**What the student picture gets right:** blame flows backward from the final score to each cause, and doing it in one traced pass beats redoing everything once per cause. **Where it breaks down:** a student judges her habits with fuzzy hunches; the network's "blame" is exact — each step's blame is precisely the upstream gradient times that step's local slope, the chain rule from a moment ago.

#### Backpropagation, in one breath
The [[backward pass||backpropagation: starting at the loss, walk backward through the saved steps, multiplying local slopes, to get every weight's gradient in one sweep]] — **backpropagation** — is that smart student, made exact. It's also our hiker feeling the slope, but for a whole chain of steps at once. Start at the loss end. Walk *backward* through the very same steps the forward pass took, and at each step do the one move you just learned: take the gradient arriving from the loss side (upstream), multiply by this step's local slope, and pass it to the next step back. Re-use the saved forward notes (`z`, `a`, the input) so nothing is recomputed. When the sweep reaches a weight, the number sitting there *is* that weight's gradient — its share of the blame. One backward walk hands **every** weight its gradient at once.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="The neuron's graph with a green gradient arrow sweeping backward from loss through activation to weighted-sum to the weights, re-using the saved z and a notes, so every weight gets its gradient in one pass."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Backward pass: one sweep from loss → every weight's gradient</text><rect x="16" y="46" width="64" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text x="48" y="65" text-anchor="middle" fill="#1F6280">weights</text><rect x="110" y="46" width="108" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="164" y="61" text-anchor="middle" fill="#9A7208" font-size="9">weighted sum → z</text><rect x="248" y="46" width="90" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text x="293" y="61" text-anchor="middle" fill="#5E5191" font-size="9">activation → a</text><rect x="368" y="46" width="132" height="30" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="434" y="65" text-anchor="middle" fill="#C93B3B" font-size="9">loss (start here)</text><g stroke="#2D8B55" stroke-width="2.5" fill="#2D8B55"><path d="M368 90 l-30 0"/><polygon points="338,90 346,86 346,94"/><path d="M248 90 l-30 0"/><polygon points="218,90 226,86 226,94"/><path d="M110 90 l-30 0"/><polygon points="80,90 88,86 88,94"/></g><text x="293" y="106" text-anchor="middle" fill="#2D8B55" font-size="9">× local slope (uses saved a)</text><text x="164" y="106" text-anchor="middle" fill="#2D8B55" font-size="9">× local slope (uses saved z)</text><text x="48" y="106" fill="#1a5c38" font-size="9" text-anchor="middle">gradient!</text><rect x="120" y="128" width="88" height="22" rx="3" fill="#F3ECDB" stroke="#C99A12" stroke-dasharray="3,2"/><text x="164" y="143" text-anchor="middle" fill="#9A7208" font-size="9">📌 saved z re-used</text><rect x="250" y="128" width="86" height="22" rx="3" fill="#EFEAF7" stroke="#B3A6D4" stroke-dasharray="3,2"/><text x="293" y="143" text-anchor="middle" fill="#5E5191" font-size="9">📌 saved a re-used</text></g></svg>
%%%

#### Why this is such a big deal
The slow way — re-deriving each weight's effect from scratch — would repeat almost all the work for every weight. A real network has *millions* of weights, so that would be hopeless. Backprop's win is [[reuse||computing every weight's gradient in a single backward sweep that shares its work, instead of redoing the chain separately for each weight]]: **one backward sweep gives you the gradient of every weight**, sharing the exact same running product along the way. That efficiency is the reason we can train giant networks at all — the whisper to every knob, computed in a single pass. Next: what that gradient actually *is* for each part of the neuron.

@@@ concept id=c5 tag="Each part's share" title="The gradient at each knob — three tiny rules" gotit="Got the three rules"
Time to open the neuron and see what blame lands on each part. Picture a **group project** where everyone shares one grade. It only feels fair if your share of the blame matches how much you actually *did*: the teammate who wrote most of the report gets most of the responsibility, and someone who just showed up gets a light touch. The neuron splits its blame the exact same way — and it comes down to just three tiny, memorable rules, one per part.

%%% svg
<svg viewBox="0 0 520 158" role="img" aria-label="A group project sharing one grade. Three teammates get different-sized slices of blame: the big contributor gets a big slice, the medium contributor a medium slice, the tag-along a tiny slice. Caption: fair blame matches how much each part actually did."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Group project: your share of blame = how much you did</text><rect x="40" y="40" width="120" height="80" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="34" text-anchor="middle" fill="#6B645E" font-size="9">did a lot 📝</text><text x="100" y="86" text-anchor="middle" fill="#C93B3B" font-size="13">big blame</text><rect x="200" y="60" width="120" height="60" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="260" y="54" text-anchor="middle" fill="#6B645E" font-size="9">did some</text><text x="260" y="96" text-anchor="middle" fill="#9A7208" font-size="12">medium</text><rect x="360" y="98" width="120" height="22" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="420" y="92" text-anchor="middle" fill="#6B645E" font-size="9">just showed up</text><text x="420" y="113" text-anchor="middle" fill="#1a5c38" font-size="10">tiny blame</text><text x="260" y="142" text-anchor="middle" fill="#6B645E">the neuron splits its blame the same fair way — three simple rules</text></g></svg>
%%%

**What the group-project picture gets right:** blame is shared in proportion to contribution — do more, own more of the result. **Where it breaks down:** teammates argue about who did what, but the neuron's split is exact and instant — each part's share is just the chain-rule multiply, no debate.

Before the rules, one behavior to feel in your hands. When the gradient reaches the activation it gets *multiplied* by the activation's slope — and that slope changes a lot with the input. Drag `z` and watch it:

%%% svg
<svg id="sl-svg" viewBox="0 0 520 214" role="img" aria-label="Interactive activation slope. Pick sigmoid, tanh, or ReLU and drag the input z. A dot rides the activation curve and a short tangent line shows the slope there — steep in the middle, flat in the tails. The slope is the exact number that multiplies the gradient passing back through."><g font-family="monospace" font-size="11"><rect x="60" y="30" width="400" height="160" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line id="sl-axis" x1="60" y1="190" x2="460" y2="190" stroke="#EFE9DF" stroke-width="1"/><line x1="260" y1="30" x2="260" y2="190" stroke="#EFE9DF" stroke-width="1"/><path id="sl-curve" d="M60 190 L260 110 L460 40" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><line id="sl-tan" x1="234" y1="150" x2="286" y2="70" stroke="#C99A12" stroke-width="3"/><circle id="sl-dot" cx="260" cy="115" r="6" fill="#C99A12" stroke="#fff" stroke-width="2"/><text x="466" y="40" fill="#9A938A" font-size="9">high</text><text x="466" y="188" fill="#9A938A" font-size="9">low</text><text x="264" y="204" fill="#9A938A" font-size="9">z = 0</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px">
<span>activation:</span>
<button id="sl-sig" type="button" style="cursor:pointer;border:2px solid #7C6DAA;background:#EDE9F8;color:#5E5191;border-radius:5px;padding:2px 8px;font-family:monospace">sigmoid</button>
<button id="sl-tanh" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">tanh</button>
<button id="sl-relu" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">ReLU</button>
</div>
<label>drag z <input id="sl-z" type="range" min="-6" max="6" step="0.2" value="0" style="width:60%;accent-color:#C99A12;vertical-align:middle"></label>
<div id="sl-out" style="margin-top:6px;color:#2C2A28">at z = 0.0, sigmoid slope = <b>0.25</b> → a gradient of 1.00 passes back as <b>0.25</b> <span style="color:#C99A12">(shrinks to a quarter)</span></div>
</div>
<script>(function(){
  var zs=document.getElementById('sl-z');if(!zs)return;
  var curve=document.getElementById('sl-curve'),dot=document.getElementById('sl-dot'),tan=document.getElementById('sl-tan'),
      out=document.getElementById('sl-out'),axis=document.getElementById('sl-axis'),
      bSig=document.getElementById('sl-sig'),bTanh=document.getElementById('sl-tanh'),bRelu=document.getElementById('sl-relu');
  var act='sigmoid';
  var X0=60,X1=460,ZLO=-6,ZHI=6;
  function xpx(z){return X0+(z-ZLO)/(ZHI-ZLO)*(X1-X0);}
  var xscale=(X1-X0)/(ZHI-ZLO);
  var defs={
    sigmoid:{f:function(z){return 1/(1+Math.exp(-z));},d:function(z){var s=1/(1+Math.exp(-z));return s*(1-s);},ypx:function(f){return 190-f*150;},yscale:150,peak:'0.25'},
    tanh:{f:function(z){return Math.tanh(z);},d:function(z){var t=Math.tanh(z);return 1-t*t;},ypx:function(f){return 110-f*70;},yscale:70,peak:'1.0'},
    relu:{f:function(z){return Math.max(0,z);},d:function(z){return z>0?1:0;},ypx:function(f){return 190-f/6*150;},yscale:25,peak:'1 or 0'}
  };
  function setBtns(){[[bSig,'sigmoid'],[bTanh,'tanh'],[bRelu,'relu']].forEach(function(p){
    var on=(p[1]===act)||(p[1]==='relu'&&act==='relu');var sel=(p[1]===act);
    p[0].style.borderColor=sel?'#7C6DAA':'#E5DFD6';p[0].style.background=sel?'#EDE9F8':'#FDF9F3';p[0].style.color=sel?'#5E5191':'#6B645E';});}
  function paint(){
    var D=defs[act],z=+zs.value;
    var d='';for(var i=0;i<=80;i++){var zz=ZLO+(ZHI-ZLO)*i/80;d+=(i?'L':'M')+xpx(zz).toFixed(1)+' '+D.ypx(D.f(zz)).toFixed(1)+' ';}
    curve.setAttribute('d',d.trim());
    var dotx=xpx(z),doty=D.ypx(D.f(z)),sl=D.d(z);
    dot.setAttribute('cx',dotx.toFixed(1));dot.setAttribute('cy',doty.toFixed(1));
    var pxslope=sl*D.yscale/xscale,L=28;
    tan.setAttribute('x1',(dotx-L).toFixed(1));tan.setAttribute('y1',(doty+L*pxslope).toFixed(1));
    tan.setAttribute('x2',(dotx+L).toFixed(1));tan.setAttribute('y2',(doty-L*pxslope).toFixed(1));
    var col=sl>=0.5?'#2D8B55':(sl>=0.1?'#C99A12':'#C93B3B');
    var word=sl>=0.5?'passes strongly':(sl>=0.1?'shrinks':'nearly dies (flat tail)');
    tan.setAttribute('stroke',col);dot.setAttribute('fill',col);
    out.innerHTML='at z = '+z.toFixed(1)+', '+act+' slope = <b>'+sl.toFixed(2)+'</b> → a gradient of 1.00 passes back as <b>'+sl.toFixed(2)+'</b> <span style="color:'+col+'">('+word+')</span>';
  }
  zs.addEventListener('input',paint);
  bSig.addEventListener('click',function(){act='sigmoid';setBtns();paint();});
  bTanh.addEventListener('click',function(){act='tanh';setBtns();paint();});
  bRelu.addEventListener('click',function(){act='relu';setBtns();paint();});
  setBtns();paint();
})();</script>
%%%

**Why this picture matters:** the gradient doesn't pass through the activation untouched — it gets *multiplied* by the activation's slope at the `z` you saved on the forward pass. Drag `z` and watch that slope: biggest in the middle (yet even at its peak sigmoid only reaches about `0.25`), nearly flat in the tails. That small-and-shrinking slope is the villain of the next unit, so get a feel for it now.

#### The three rules, each in one plain line
Every rule below is just the chain-rule move (upstream gradient × a local slope) applied to one part. Let `g` be the gradient flowing *into* the neuron from the loss side.

- **Rule 1 — a weight's gradient = its input × the incoming gradient.** A weight only touches the answer *through* the input it multiplies. So a weight fed a **big input** gets a **big** gradient (it had more say, so it gets more blame); a weight fed a tiny input barely moves. This is why loud inputs cause loud weight updates.
- **Rule 2 — the activation multiplies the gradient by its slope.** Whatever gradient reaches the activation gets scaled by the activation's slope at the saved `z`. For [[sigmoid||an S-shaped activation that squashes numbers into 0–1; its slope is σ(z)(1−σ(z)), which peaks at only 0.25]] that slope is `σ(z)(1−σ(z))` — biggest in the middle (at most `0.25`), tiny in the tails. For [[ReLU||an activation that passes positives and zeros negatives; its slope is a clean 1 for positive inputs, 0 for negative]] it's a clean `1` (positive `z`) or `0` (negative `z`).
- **Rule 3 — the bias just passes the gradient through.** The [[bias||the neuron's constant offset, added after the weighted sum; nudging it moves z one-for-one]] has slope `1` — nudge it by a hair and `z` moves by exactly that hair. So the gradient arriving at the bias passes **straight through**, unchanged.

Watch all three on tiny numbers — the whole blame-split in one run:

%%% demo id=rules label="run it — split the blame three ways"
code: x=2.0; incoming_g=0.3; relu_slope=1.0  # input, gradient into the neuron, ReLU slope for positive z
code: w_grad=incoming_g*relu_slope*x; b_grad=incoming_g*relu_slope*1.0; print("weight grad",w_grad," bias grad",b_grad)
out: weight grad 0.6  bias grad 0.3
take: <b>Big input → big weight gradient.</b> The weight's blame is input(2.0) × the gradient through the activation(0.3×1) = 0.6 — twice the bias's 0.3, exactly because its input was 2×. The bias just passed the 0.3 straight through (slope 1). Same chain-rule multiply, three parts.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the three rules in symbols.</b> Peek only if you like notation; the plain lines above are all you need. With loss slope <code>g = ∂L/∂a</code> arriving at the neuron and activation slope <code>a'(z)</code>: the pre-activation gradient is <code>δ = g · a'(z)</code>; then <code>∂L/∂wᵢ = δ · xᵢ</code> (Rule 1), <code>∂L/∂b = δ · 1 = δ</code> (Rule 3), and Rule 2 is the <code>a'(z)</code> factor itself. Every symbol is just "upstream gradient × one local slope."
!!!

Those three rules *are* the neuron's backward pass. Notice each is safe and boring on its own — a multiply by an input, a slope, or a plain `1`. The drama only starts when you **chain many of them together**, which is exactly the next puzzle.

@@@ concept id=c6 tag="The fading signal" title="Puzzle — when the downhill signal fades to nothing" gotit="Got vanishing gradients"
Here's your first puzzle, and it's the most famous one in deep learning. You just felt in your hands that the sigmoid's slope is small everywhere — at most `0.25`, and even tinier out in its flat tails. Now remember backprop's move: at every layer, **multiply** by that slope. So what happens when you stack *many* layers, each multiplying by a small number? Play the **telephone game**: a whisper passed down a long line of kids, each repeating it a little quieter, until the last kid hears nothing at all. That's the trap.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A line of kids playing telephone. A whisper starts loud on the right and gets quieter at each kid to the left, shrinking to nothing. Caption: multiply by a small slope at every layer and the gradient fades to zero."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Telephone: each layer multiplies by a small slope → the whisper fades</text><text x="470" y="70" font-size="24">🧒</text><text x="470" y="98" text-anchor="middle" fill="#C93B3B" font-size="10">loud</text><text x="368" y="70" font-size="22">🧒</text><text x="378" y="98" text-anchor="middle" fill="#C99A12" font-size="10">×0.25</text><text x="270" y="70" font-size="19">🧒</text><text x="280" y="98" text-anchor="middle" fill="#C99A12" font-size="10">×0.25</text><text x="176" y="70" font-size="16">🧒</text><text x="184" y="98" text-anchor="middle" fill="#9A938A" font-size="10">×0.25</text><text x="92" y="70" font-size="12">🧒</text><text x="96" y="98" text-anchor="middle" fill="#B8AEA2" font-size="10">×0.25</text><text x="40" y="70" font-size="9" fill="#B8AEA2">·</text><g stroke="#B8AEA2" stroke-width="1.5" fill="#B8AEA2"><path d="M460 60 l-70 0"/><polygon points="390,60 398,56 398,64"/><path d="M360 60 l-70 0"/><polygon points="290,60 298,56 298,64"/><path d="M262 60 l-70 0"/><polygon points="192,60 200,56 200,64"/><path d="M168 60 l-64 0"/><polygon points="104,60 112,56 112,64"/></g><text x="60" y="130" fill="#9A938A" font-size="10">last kid hears ≈ nothing 😴</text><text x="470" y="130" text-anchor="end" fill="#C93B3B" font-size="10">loss end (loud) →</text></g></svg>
%%%

**What the telephone picture gets right:** repeatedly shrinking a signal, step after step, drives it to almost nothing — and the *far* end (the early layers) suffers most. **Where it breaks down:** in telephone the message also gets garbled; here the gradient stays "correct," it just gets so tiny that the early weights barely move — they effectively stop learning.

#### Watch the fade — the decay ladder
This is the [[vanishing gradient||when many small activation slopes multiply together, the gradient shrinks toward 0, so early layers stop learning]] problem. The sigmoid's slope maxes out at just `0.25`. So through 4 sigmoid layers the gradient is multiplied by about `0.25 × 0.25 × 0.25 × 0.25` — watch the bars shrink toward the floor:

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A decay ladder of bars: starting gradient 1, then after one sigmoid layer 0.25, after two 0.0625, after three 0.0156, after four 0.0039. Each bar is a quarter the height of the one before, shrinking toward the floor."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Multiply by 0.25 each sigmoid layer → gradient collapses</text><line x1="40" y1="138" x2="500" y2="138" stroke="#E5DFD6" stroke-width="1.5"/><rect x="56" y="30" width="52" height="108" fill="#2D8B55"/><text x="82" y="24" text-anchor="middle" fill="#1a5c38">1.0</text><text x="82" y="152" text-anchor="middle" fill="#6B645E" font-size="9">start</text><rect x="150" y="111" width="52" height="27" fill="#C99A12"/><text x="176" y="105" text-anchor="middle" fill="#9A7208">0.25</text><text x="176" y="152" text-anchor="middle" fill="#6B645E" font-size="9">1 layer</text><rect x="244" y="131" width="52" height="7" fill="#C99A12"/><text x="270" y="124" text-anchor="middle" fill="#9A7208">0.0625</text><text x="270" y="152" text-anchor="middle" fill="#6B645E" font-size="9">2 layers</text><rect x="338" y="136" width="52" height="2" fill="#C93B3B"/><text x="364" y="129" text-anchor="middle" fill="#C93B3B">0.0156</text><text x="364" y="152" text-anchor="middle" fill="#6B645E" font-size="9">3 layers</text><rect x="432" y="137.4" width="52" height="0.6" fill="#C93B3B"/><text x="458" y="131" text-anchor="middle" fill="#C93B3B">0.0039</text><text x="458" y="152" text-anchor="middle" fill="#6B645E" font-size="9">4 layers 💀</text></g></svg>
%%%

By the fourth layer the gradient is under `0.004` — the early layers get almost no downhill signal, so they freeze. It's our hiker feeling flatter and flatter ground the deeper the layer, until she can't tell which way is down at all. **Predict how small it gets, then run it:**

%%% demo id=vanish label="run it — the fade"
code: slope=0.25; [round(slope**n,4) for n in range(1,6)]  # gradient after 1..5 sigmoid layers
out: [0.25, 0.0625, 0.0156, 0.0039, 0.001]
take: <b>Small slopes multiply toward zero.</b> After 5 sigmoid layers the gradient is ×0.001 — the early layers barely move. This is why deep sigmoid networks were so hard to train for years.
%%%

#### The neat fix — a slope that doesn't shrink
The cure is to stop multiplying by tiny numbers. Swap sigmoid for **ReLU**, whose slope for any positive input is a clean `1`. Multiply by `1` as many times as you like and the signal stays exactly as loud — no fade. That's the whole reason ReLU took over deep learning: it keeps the gradient alive through many layers.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Before and after: sigmoid multiplies the gradient by 0.25 each layer and it fades to near zero; ReLU multiplies by 1 each layer and it stays full height. Two rows of bars compared."><g font-family="monospace" font-size="10"><text x="140" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">sigmoid ×0.25 → fades</text><rect x="40" y="34" width="30" height="60" fill="#C99A12"/><rect x="86" y="79" width="30" height="15" fill="#C99A12"/><rect x="132" y="90" width="30" height="4" fill="#C93B3B"/><rect x="178" y="93" width="30" height="1" fill="#C93B3B"/><line x1="34" y1="94" x2="216" y2="94" stroke="#E5DFD6"/><text x="120" y="112" text-anchor="middle" fill="#6B645E" font-size="9">early layers freeze 💀</text><line x1="258" y1="24" x2="258" y2="104" stroke="#E5DFD6"/><text x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">ReLU ×1 → stays loud</text><rect x="300" y="34" width="30" height="60" fill="#2D8B55"/><rect x="346" y="34" width="30" height="60" fill="#2D8B55"/><rect x="392" y="34" width="30" height="60" fill="#2D8B55"/><rect x="438" y="34" width="30" height="60" fill="#2D8B55"/><line x1="294" y1="94" x2="476" y2="94" stroke="#E5DFD6"/><text x="384" y="112" text-anchor="middle" fill="#1a5c38" font-size="9">signal survives every layer 🎉</text></g></svg>
%%%

**One puzzle solved, and it's the big one.** A fading gradient (cause: small slopes multiplied over and over) → ReLU (cure: a slope of 1 that doesn't shrink). But ReLU has a small dark side of its own, plus two cousin traps — quick ones — coming next.

@@@ concept id=c7 tag="Three quick traps" title="Three cousin traps — dead units, explosions, and twins" gotit="Got the three cures"
Three fast puzzles, each with a neat one-line cure. Picture three everyday things: a **light switch stuck OFF**, a **microphone squeal** that feeds on itself, and **identical twins** who copy each other's every move. Each one is a real way learning can go wrong — and each has a tidy fix an engineer reaches for by reflex.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Three cousin traps drawn as icons: a light switch stuck off labelled dead ReLU cured by Leaky ReLU; a microphone squeal snowballing labelled exploding gradient cured by clipping; identical twins moving in lockstep labelled zero-init symmetry cured by small random init."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Three quick traps — each with a one-line cure</text><rect x="18" y="30" width="150" height="118" rx="8" fill="#FDFCF9" stroke="#E5DFD6"/><text x="93" y="52" text-anchor="middle" font-size="24">🔌</text><text x="93" y="74" text-anchor="middle" fill="#C93B3B">switch stuck OFF</text><text x="93" y="94" text-anchor="middle" fill="#6B645E">dead ReLU: slope 0</text><text x="93" y="108" text-anchor="middle" fill="#6B645E">never learns</text><text x="93" y="132" text-anchor="middle" fill="#2D8B55">cure: Leaky ReLU</text><text x="93" y="144" text-anchor="middle" fill="#2D8B55">+ sensible init</text><rect x="185" y="30" width="150" height="118" rx="8" fill="#FDFCF9" stroke="#E5DFD6"/><text x="260" y="52" text-anchor="middle" font-size="24">🎤</text><text x="260" y="74" text-anchor="middle" fill="#C93B3B">feedback SQUEAL</text><text x="260" y="94" text-anchor="middle" fill="#6B645E">exploding: ×big ×big</text><text x="260" y="108" text-anchor="middle" fill="#6B645E">→ overshoot</text><text x="260" y="132" text-anchor="middle" fill="#2D8B55">cure: clip the gradient</text><text x="260" y="144" text-anchor="middle" fill="#2D8B55">+ careful init</text><rect x="352" y="30" width="150" height="118" rx="8" fill="#FDFCF9" stroke="#E5DFD6"/><text x="427" y="52" text-anchor="middle" font-size="24">👯</text><text x="427" y="74" text-anchor="middle" fill="#C93B3B">identical twins</text><text x="427" y="94" text-anchor="middle" fill="#6B645E">zero init: same grad</text><text x="427" y="108" text-anchor="middle" fill="#6B645E">stay identical</text><text x="427" y="132" text-anchor="middle" fill="#2D8B55">cure: small RANDOM</text><text x="427" y="144" text-anchor="middle" fill="#2D8B55">starting weights</text></g></svg>
%%%

#### Trap 1 · the dead ReLU (a switch stuck off)
ReLU's slope is `0` for any negative input. If a unit gets pushed so its `z` is *always* negative, it outputs `0` forever *and* its slope is always `0` — so its gradient is always `0`, and it never updates again. That's a [[dead ReLU||a ReLU unit stuck outputting 0 with slope 0, so it gets no gradient and never learns again]]: a light switch jammed off. **The cure:** [[Leaky ReLU||a ReLU variant with a small nonzero slope for negatives (like 0.01), so a unit is never fully dead]] — give negatives a tiny slope (say `0.01`) instead of a flat `0`, so a trickle of gradient always gets through. Sensible starting weights help too, so units don't get shoved off a cliff on day one.

#### Trap 2 · the exploding gradient (a feedback squeal)
Flip the fade around. If the numbers you multiply are **big** instead of tiny, the running product *grows* — a mic held too close to its speaker, squealing louder and louder. This is the [[exploding gradient||when large weights or slopes multiply together, the gradient blows up huge, so the update overshoots wildly]]: the update becomes a giant leap that overshoots the valley and lands somewhere worse. **Predict what happens after a few big multiplies, then run it:**

%%% demo id=explode label="run it — the squeal"
code: g=1.0; big=3.0; [round(g:=g*big,1) for _ in range(6)]  # gradient multiplied by a big slope each layer
out: [3.0, 9.0, 27.0, 81.0, 243.0, 729.0]
take: <b>Big slopes multiply toward infinity.</b> ×3 per layer and the gradient rockets from 3 to 729 in six steps — a step this size overshoots wildly. <b>The cure:</b> gradient <b>clipping</b> — cap the gradient at a ceiling before you step. Careful starting weights keep the multiplies near 1 to begin with.
%%%

The cure named in the demo: [[gradient clipping||cap the gradient's size at a ceiling before taking a step, so one huge value can't cause a wild overshoot]] — set a ceiling and never step further than that in one go. Careful weight initialization keeps the per-layer multiplies close to `1`, so the product neither fades nor explodes.

#### Trap 3 · the twin trap (start them all the same)
Last one, and it's sneaky. What if you start *every* weight at exactly `0` (or all the same value)? Then every neuron computes the same thing, so backprop hands every neuron the **identical** gradient, so they all update **identically** — forever. They're identical twins copying each other's every move; a hundred neurons behave like one. This is [[zero-init symmetry||starting all weights equal so every neuron gets the same gradient and stays identical, wasting the whole layer]]. **The cure:** start the weights at **small random** values — a tiny bit different each — so each neuron gets a slightly different gradient and they grow into different, useful detectors. Randomness *breaks the symmetry*.

!!! c-info 🧭
<b>Keep it straight:</b> all four traps are the same math seen from different angles. <em>Fade</em> = multiplying by numbers under 1. <em>Explode</em> = multiplying by numbers over 1. <em>Dead unit</em> = multiplying by exactly 0. <em>Twins</em> = every unit multiplying by the <em>same</em> thing. The cures all aim for the sweet spot: slopes near 1, and a little healthy variety between units.
!!!

**Three traps, three quick cures — done.** One last trick, and it's the one that lets you *trust* your own backprop code.

@@@ concept id=c8 tag="Check your work" title="The gradient check — measure it two ways" gotit="Got the gradient check"
You've computed the downhill slope with the chain rule — but how do you know you didn't make a sign slip or a typo? Here's the trick every careful engineer keeps in their back pocket, and it's pure common sense. Think of **weighing flour two ways**: once by reading the recipe's math, and once by actually putting it on a scale. If the two numbers match, you trust your recipe. If they disagree, you know a step is wrong — *before* you bake a ruined cake.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Two ways to get the same slope compared on a balance scale. On the left, backprop's computed gradient. On the right, a measured slope from nudging the weight up and down. When they match, the check passes; a mismatch flags a bug."><g font-family="monospace" font-size="10.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two ways to the same slope — do they match?</text><rect x="34" y="46" width="150" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="109" y="63" text-anchor="middle" fill="#5E5191">backprop's gradient</text><text x="109" y="78" text-anchor="middle" fill="#6B645E" font-size="9">(the chain rule)</text><rect x="336" y="46" width="150" height="40" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="411" y="63" text-anchor="middle" fill="#1a5c38">measured slope</text><text x="411" y="78" text-anchor="middle" fill="#6B645E" font-size="9">(nudge &amp; re-check the loss)</text><text x="260" y="60" text-anchor="middle" font-size="20">⚖️</text><text x="260" y="80" text-anchor="middle" fill="#6B645E" font-size="10">≈ equal?</text><text x="150" y="122" fill="#2D8B55" font-size="10">match → trust your code ✓</text><text x="360" y="122" fill="#C93B3B" font-size="10" text-anchor="end">mismatch → hunt the bug 🐛</text></g></svg>
%%%

**What the two-weighings picture gets right:** computing a number and *measuring* it independently is the surest way to catch a mistake — agreement builds trust, disagreement flags a bug. **Where it breaks down:** a scale is exact, but the measured slope is only an *estimate* (it uses a tiny nudge, not a true zero step), so you check they're *close*, not identical to the last digit.

#### How to measure a slope without any calculus
A slope is just "how much the output moves when you nudge the input." So *measure* it: nudge one weight up by a tiny `ε`, see how much the loss rose; nudge it down by `ε`, see how much it fell; the slope is the change divided by the distance you moved. This is the [[numerical gradient check||estimate a slope by nudging a weight up and down by a tiny ε and measuring the loss change, then compare it to backprop's gradient]] — in one plain line, `slope ≈ (loss(w+ε) − loss(w−ε)) / (2ε)`. No chain rule, just two loss readings and a divide. If this measured slope matches what backprop computed, your backprop is correct. **Predict whether the two match, then run it:**

%%% demo id=gradcheck label="run it — two ways, one slope"
code: f=lambda w: w**2; w=3.0; eps=1e-5  # toy loss f(w)=w^2, whose true slope at w is 2w=6
code: measured=(f(w+eps)-f(w-eps))/(2*eps); backprop=2*w; print("measured",round(measured,4)," backprop",backprop)
out: measured 6.0  backprop 6.0
take: <b>They match → the gradient is right.</b> The measured slope (nudge up and down, divide by 2ε) is 6.0, and backprop says 6.0 too. When these disagree, you've got a bug in your backward pass — this cheap check catches it before you waste a training run.
%%%

!!! c-warn ⚠️
<b>Use it to check, not to train.</b> Measuring the slope needs a full forward pass <em>per weight</em> — hopelessly slow for millions of weights. Backprop gets them all in one sweep. So you gradient-check on a tiny toy to prove the code is right, then let fast backprop do the real work.
!!!

**That's the whole toolkit.** You can compute the downhill slope, split it fairly across every knob, dodge the ways it fades or explodes, and *check* your own work. One page to tie it together.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you just learned how a network figures out which way to improve. Here's a way to hold the whole day at once: picture our **hiker in fog** again, and read the beats below as the steps of one descent, in order. Each step only made sense because of the one before it — you can't trace blame backward until you've saved your work going forward. **Where the hiker picture breaks down:** a hiker takes the downhill step herself, but today you only learned how to *find* the downhill direction — actually taking the step is tomorrow's job (the optimizer). Read the trail, then keep the cheat-sheets below as your map.

The trail you walked, in order:
- **Step 1 · The downhill arrow.** The **gradient** answers "which way is downhill, for every weight?" It points *uphill* (more error), so to learn we step the *opposite* way, downhill.
- **Step 2 · Save your work.** The **forward pass** makes a guess and quietly saves its in-between values (`z`, `a`) — the sticky notes the backward trip re-uses.
- **Step 3 · Chain of effects.** A **local slope** is one tiny step's effect; the **chain rule** multiplies the local slopes along the path to get the whole effect.
- **Step 4 · Trace the blame.** **Backpropagation** starts at the loss and sweeps backward, multiplying local slopes, handing every weight its gradient in *one* pass — the reason big networks are trainable.
- **Step 5 · Each part's share.** Three tiny rules: weight gradient = **input × incoming gradient** (big inputs → big updates); the **activation multiplies by its slope**; the **bias passes the gradient straight through**.
- **Step 6 · The traps.** *Vanishing* (small slopes multiply → 0 → use **ReLU**), *dead ReLU* (stuck-off unit → **Leaky ReLU** + init), *exploding* (big slopes multiply → ∞ → **clip** + init), *twins* (all-equal weights stay identical → **small random init**). And a **gradient check** measures the slope two ways to catch bugs.

Here's the one picture to keep — the whole flow, forward then back.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Recap flow: forward pass goes right from input through weighted sum to activation to loss, saving z and a; backward pass returns left multiplying local slopes to give each weight its gradient. Below, four traps and their cures are listed."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The whole day: forward saves, backward traces blame</text><rect x="16" y="30" width="60" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="46" y="47" text-anchor="middle" fill="#1F6280">input</text><rect x="120" y="30" width="86" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="163" y="47" text-anchor="middle" fill="#9A7208">sum → z 📌</text><rect x="250" y="30" width="86" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text x="293" y="47" text-anchor="middle" fill="#5E5191">act → a 📌</text><rect x="380" y="30" width="86" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="423" y="47" text-anchor="middle" fill="#C93B3B">loss</text><g stroke="#2A7B9B" stroke-width="1.8" fill="#2A7B9B"><path d="M76 43 l40 0"/><polygon points="116,43 108,39 108,47"/><path d="M206 43 l40 0"/><polygon points="246,43 238,39 238,47"/><path d="M336 43 l40 0"/><polygon points="376,43 368,39 368,47"/></g><text x="250" y="24" text-anchor="middle" fill="#2A7B9B" font-size="9">forward →</text><g stroke="#2D8B55" stroke-width="1.8" fill="#2D8B55"><path d="M380 70 l-40 0"/><polygon points="340,70 348,66 348,74"/><path d="M250 70 l-40 0"/><polygon points="210,70 218,66 218,74"/><path d="M120 70 l-40 0"/><polygon points="80,70 88,66 88,74"/></g><text x="250" y="86" text-anchor="middle" fill="#2D8B55" font-size="9">← backward: × local slopes → each weight's gradient</text><line x1="30" y1="100" x2="490" y2="100" stroke="#E5DFD6"/><text x="30" y="120" fill="#C93B3B">fade → ReLU</text><text x="160" y="120" fill="#C93B3B">dead → Leaky ReLU</text><text x="330" y="120" fill="#C93B3B">explode → clip</text><text x="30" y="140" fill="#C93B3B">twins → small random init</text><text x="240" y="140" fill="#2D8B55">check → measure the slope two ways</text><text x="260" y="164" text-anchor="middle" fill="#6B645E">aim for slopes near 1, a little variety between units</text></g></svg>
%%%

#### Cheat-sheet · each part's gradient, and the traps
%%% table
:: Part / trap :: In one line :: Remember
weight gradient :: input × the gradient flowing in :: big input → big update
activation :: multiplies by its slope at saved z :: sigmoid ≤ 0.25 (tiny in tails), ReLU is 1 or 0
bias gradient :: the incoming gradient, unchanged :: slope 1 → passes straight through
vanishing :: small slopes multiply toward 0 :: cure: ReLU (slope stays 1)
dead ReLU :: unit stuck negative, slope 0 forever :: cure: Leaky ReLU + sensible init
exploding :: big slopes multiply toward ∞ :: cure: gradient clipping + careful init
zero-init twins :: all-equal weights get identical gradients :: cure: small random init
%%%

#### Cheat-sheet · the words you met today
%%% jargon
gradient | the arrow saying which way to nudge every weight to change the loss fastest; points uphill, so we step the opposite way
forward pass | running the input through the neuron to a guess, saving z and a for the trip back
computation graph | the neuron's math seen as a chain of tiny steps, each with an easy local slope
local derivative | the slope of just one tiny step — how much its output moves when its input moves a hair
chain rule | multiply the local slopes along the path to get how the loss depends on an early weight
upstream gradient | the gradient arriving at a step from the loss side; × the step's local slope to pass it further back
backpropagation | one backward sweep from the loss that multiplies local slopes to give every weight its gradient
vanishing gradient | small slopes (sigmoid ≤ 0.25) multiplied over many layers shrink the gradient toward 0 → early layers stop learning
dead ReLU | a unit stuck outputting 0 with slope 0, so it never learns again → cure Leaky ReLU
exploding gradient | big slopes multiplied blow the gradient up huge → cure gradient clipping
gradient check | measure a slope by nudging a weight ±ε and compare to backprop, to catch bugs
%%%

That's the whole day. Tomorrow you take the downhill step for real — the **training loop**, where these gradients get used to actually change the weights, over and over, until the network learns.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does the gradient tell the network, and which way do we step? | a:2 | The final loss value; we step toward it | How many layers to use; we step randomly | Which way is uphill (more error) for every weight; we step the opposite way, downhill | The learning rate; we don't step | fb: The gradient points uphill — toward MORE error — for every weight at once. To learn, the network steps the OPPOSITE way, downhill, toward less error. That's the whole "feel the slope, step down" idea.
q: What is the chain rule, in one line? | a:1 | Add up the loss of every layer | Multiply the local slopes along the path to get how an early weight affects the far-away loss | Pick the layer with the biggest weight | Run the network backward with new random weights | fb: The chain rule multiplies the small "local" slopes hop by hop — like pedal → speed → distance → money on a road trip. That single multiply, swept backward, is exactly what backpropagation does.
q: Why do deep sigmoid networks suffer from vanishing gradients, and what's the common cure? | a:2 | Sigmoid uses too much memory; cure: bigger batches | The loss is negative; cure: flip its sign | The sigmoid's slope is at most 0.25, so multiplying it across many layers shrinks the gradient toward 0; cure: ReLU, whose slope stays 1 for positive inputs | Sigmoid outputs are too large; cure: clip them | fb: Backprop multiplies by the activation's slope at every layer. Sigmoid's slope tops out at 0.25, so across many layers 0.25 × 0.25 × … collapses toward 0 and early layers stop learning. ReLU's slope is a clean 1 for positive inputs, so the signal survives.
q: A ReLU unit is stuck outputting 0 with slope 0 and never learns. What is this, and how do you fix it? | a:1 | Exploding gradient; fix by lowering the learning rate | A dead ReLU; fix with Leaky ReLU (a small nonzero negative slope) plus sensible weight initialization | Vanishing gradient; fix with a bigger network | Zero-init symmetry; fix by clipping | fb: A dead ReLU is pushed permanently negative, so it outputs 0 with slope 0 — its gradient is always 0 and it can't recover. Leaky ReLU gives negatives a small slope (like 0.01) so a trickle of gradient always flows; good initialization keeps units off the cliff to begin with.
%%%

@@@ produce id=produce tag="Produce" title="Watch a gradient flow backward — and check it two ways" gotit="Done"
Time to see today's big idea with your own eyes. You'll run a tiny neuron forward, trace the gradient backward through it by hand with the three rules, and then **check your work** by measuring the same slope a totally different way. **Predict first:** for a neuron with input `x = 2` and a positive `z`, will the *weight's* gradient be bigger or smaller than the *bias's* gradient — and by what factor? Then run it and **watch** the weight's gradient come out exactly `x` times the bias's, because the weight's blame is scaled by its input while the bias just passes the gradient through. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py`. (1) Do a forward pass for one neuron: `z = w*x + b` then `a = relu(z)`, and a squared-error loss `L = (a - target)**2`; **save** `x`, `z`, `a` as you go. (2) Backprop by hand with the three rules: the gradient into the neuron is `2*(a-target)`, the ReLU slope is `1` if `z>0` else `0`, then `w_grad = incoming * relu_slope * x` and `b_grad = incoming * relu_slope` — **notice** `w_grad` is `x` times `b_grad`. (3) **Check it:** write `loss_of_w(w)` and compute the measured slope `(loss_of_w(w+1e-5) - loss_of_w(w-1e-5)) / (2e-5)`, then print it next to your `w_grad` and **watch** them match. Run with `python3 sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 5 artifact.

Create sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py that, with a comment on each step:
1. Runs a forward pass for one neuron: given x=2.0, w=0.5, b=0.1, target=1.0, compute z=w*x+b, a=relu(z) (relu = max(0,z)), and loss L=(a-target)**2; save x, z, a.
2. Backprops by hand with the three rules: incoming = 2*(a-target); relu_slope = 1.0 if z>0 else 0.0; w_grad = incoming*relu_slope*x; b_grad = incoming*relu_slope. Print both and show w_grad == x * b_grad (big input -> big weight update).
3. Gradient-checks w_grad: define loss_of_w(w) that redoes the forward pass and returns L, then measured = (loss_of_w(w+1e-5) - loss_of_w(w-1e-5))/(2e-5); print measured next to w_grad and show they match to ~4 decimals.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the gradient, traced and verified)
- The **weight's gradient is exactly `x` times the bias's** — with `x = 2`, twice as big — because a weight's blame is scaled by its input while the bias passes the gradient straight through.
- Flip `z` negative (try a big negative `b`) and **both gradients become `0`** — the ReLU slope is `0` there, so that unit gets no signal. That's a dead ReLU, live.
- The **measured slope matches your hand-computed `w_grad`** to about four decimals — proof your backward pass is correct, exactly the check a careful engineer runs before trusting the code.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-05-gradients-backprop/log.md`: (1) what the gradient is, and which way we step; (2) why the forward pass saves its in-between values; (3) one trap from today (vanishing, dead ReLU, exploding, or twins) and its one-line cure.
!!!

@@@ fin
