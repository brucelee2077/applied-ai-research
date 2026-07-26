---
quest_id: wf2-d01-neuron
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 1 — A Single Neuron"
module_label: "Module 2 · Train · Day 1"
title: "A Single Neuron"
subtitle: "The Smallest Piece of a Learning Machine"
brand_sub: "Foundations · M2 Day 1"
spine: "brain cell: weigh → add → decide"
nav_prev_href: "../../m01-shape-of-data/review.html"
nav_prev_label: "M1 · Review Gate"
nav_next_href: "../day-02-activations/lesson.html"
nav_next_label: "Activation Functions"
fin_title: "Module 2 · Day 1 complete! 🏆"
fin_body: "Nice work — you've met the <b>single neuron</b>: it weighs its inputs, adds them up with a bias, and decides. That tiny weigh → add → decide is the atom every big model is built from.<br>Next up: <b>Activation Functions</b> — the little bend that lets neurons stack into something deep."
notebook_yardstick: 00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb
coverage_topics:
  - {topic: single neuron, keywords: [artificial neuron, a neuron, one neuron]}
  - {topic: inputs, keywords: [inputs, raw number]}
  - {topic: weights, keywords: [weight, trust dial]}
  - {topic: bias, keywords: [bias, starting mood, head start]}
  - {topic: weighted sum, keywords: [weighted sum, pre-activation, running total]}
  - {topic: activation function, keywords: [activation function, decide step, the activation]}
  - {topic: step function, keywords: [step function, mcculloch, perceptron, height bar]}
  - {topic: sigmoid, keywords: [sigmoid, dimmer, s-curve, s-shaped]}
  - {topic: decision boundary, keywords: [decision boundary, straight line, chalk line]}
  - {topic: biological neuron inspiration, keywords: [brain cell, inspired by]}
  - {topic: capability limit XOR, keywords: [xor, linearly separable, one line can]}
  - {topic: layers and networks fix, keywords: [layer, hidden layer, neural network, deep learning]}
  - {topic: forward propagation, keywords: [forward propagation, forward pass, conveyor]}
  - {topic: real-world applications, keywords: [image recognition, speech recognition, language translation, game playing, recommendation, self-driving]}
  - {topic: how networks learn, keywords: [guess, check error, adjust, repeat, training is the search]}
  - {topic: common misconceptions, keywords: [myth, magic black box, bigger is always, think like a human, pattern]}
  - {topic: optional further reading, keywords: [further reading, michael nielsen, 3blue1brown]}
---

@@@ hero
@lede Picture asking three friends whether to see a movie tonight. One friend has amazing taste, so you *lean on* what they say. One is right about half the time, so you only half-listen. One always hates everything, so you barely count their vote. You quietly weigh each friend, add it all up in your head, and land on a decision: yes or no. That little "weigh → add → decide" is *exactly* what one **neuron** does — a tiny math copy of a **brain cell**. And here's the wild part: connect a few million of these little vote-counters into a network and you get the thing behind ChatGPT finishing your sentence, your phone recognizing your face, or an app translating a language. Today you meet that single piece, all on its own. It's simpler than you fear, and by the end you'll honestly be able to say, "I know what the smallest part of an AI actually does."
@goal Together we'll build one neuron from nothing: its inputs, its **weights** (how much it trusts each input), its **bias** (its starting mood), the little sum it adds up, and the "decide" step at the end. Then you'll drag a real neuron's dials and watch it draw a line, catch the one pattern it *can't* learn, and meet the neat fix — the **layers** and **networks** that the word *deep* in "deep learning" points at. Every new word gets a plain-English meaning the moment it shows up. One promise about the math: anything in a box marked **Optional (skippable)** really is optional — skip every one and you'll still follow the whole day.

%%% warmup
q: This is Day 1 — no neural-network knowledge needed yet. Quick readiness check: what does "multiply" do to two numbers like 0.5 and 2? | a:2 | it adds them to get 2.5 | it picks the bigger one | it scales one by the other to get 1.0 | it makes them both zero | concept: prereq-multiply | fb: Multiplying scales one number by the other: 0.5 × 2 = 1.0. Today a neuron does exactly this to each input.
q: A neuron will hand you a plain list of numbers to work with, like [2, 3]. What is that — a single number, or a few numbers grouped together? | a:1 | a single number | a few numbers grouped in a list | a word | a picture | concept: prereq-list-of-numbers | fb: It's a small list (a group) of numbers. You met data as lists of numbers in M1 — today those become a neuron's inputs.
%%%

@@@ concept id=c1 tag="What it is" title="A neuron: weigh → add → decide" gotit="Got the idea"
Let's start where the name comes from. Deep inside your head sit tiny cells called **brain cells** — the wiring that lets you read this sentence.

Each brain cell has little arms that catch signals from other cells. Some arms it trusts a lot; some it barely listens to. It gathers all those signals, sizes them up, and when it hears *enough*, it "fires" and passes a signal along. One small yes-or-no call.

An artificial [[neuron||A tiny math function: it takes a few input numbers, scales each by how much it matters, adds them up, and produces one output number.]] copies that idea with plain arithmetic — and that's the whole machine. Really.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A brain cell drawn like a blob with three input arms on the left catching signals, and one output arm on the right. Beneath it, the same idea as three labelled boxes: weigh the inputs, add them up, then decide."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A brain cell → a neuron: catch signals, weigh them, fire</text><line x1="40" y1="55" x2="150" y2="80" stroke="#B8AEA2" stroke-width="2"/><line x1="40" y1="80" x2="150" y2="85" stroke="#B8AEA2" stroke-width="2"/><line x1="40" y1="108" x2="150" y2="92" stroke="#B8AEA2" stroke-width="2"/><circle cx="40" cy="55" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="40" cy="80" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="40" cy="108" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="22" y="59" text-anchor="end" fill="#6B645E">in</text><text x="22" y="84" text-anchor="end" fill="#6B645E">in</text><text x="22" y="112" text-anchor="end" fill="#6B645E">in</text><path d="M150 60 Q 195 40 235 70 Q 260 100 225 118 Q 175 128 152 100 Q 140 78 150 60 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="192" y="94" text-anchor="middle" fill="#276b45">neuron</text><line x1="238" y1="86" x2="320" y2="86" stroke="#2D8B55" stroke-width="2.5"/><polygon points="320,80 334,86 320,92" fill="#2D8B55"/><text x="345" y="90" fill="#276b45">one number out →</text><rect x="60" y="150" width="90" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="105" y="169" text-anchor="middle" fill="#8A6D3B">1 · weigh</text><text x="163" y="169" fill="#B8AEA2">→</text><rect x="185" y="150" width="90" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="230" y="169" text-anchor="middle" fill="#5E5191">2 · add</text><text x="288" y="169" fill="#B8AEA2">→</text><rect x="310" y="150" width="90" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="355" y="169" text-anchor="middle" fill="#276b45">3 · decide</text><text x="260" y="196" text-anchor="middle" fill="#6B645E" font-size="10">every neuron, big model or small, is just these three steps</text></g></svg>
%%%

**The picture is honest about the shape** — many signals in, each trusted differently, summed, one signal out. **It stops being true about the biology:** a real brain cell is alive and wildly complicated, while an artificial neuron is a pinch of multiply-and-add. So: *inspired by*, never "the same as." (First myth busted; we'll collect a few more at the end.)

#### The three beats, one line each
These three names are the skeleton of the entire day, so let's walk them slowly.

%%% steps
step: weigh
why: each incoming number gets multiplied by a number that says how much it matters — that multiplier is a *weight*, the arm the brain cell trusts (or doesn't)
step: add
why: pile all the weighed numbers into one running total, therefore many opinions become one score — plus a little starting nudge called the *bias*
step: decide
why: pass that one total through a final rule that turns it into the neuron's answer, which is the moment the cell "fires" or stays quiet
%%%

%%% insight
Here's why those three little words are worth memorising: they never change. Not for a bigger neuron, not for a deeper network, not for the model writing your emails. Learn **weigh → add → decide** today and you have the shape of *every* neuron you will ever meet — the rest of this module just fills in each beat with detail.
%%%

So far: a neuron is a tiny **brain cell** that likes to *weigh → add → decide*. That's the shape to hold. Now let's meet each beat as its own unit — starting with the trust dials.

@@@ concept id=c2 tag="Weights" title="Weights — how much you trust each input" gotit="Got weights"
Time for the first beat: **weigh**. Back to the movie night.

Your friend with amazing taste? You turn their "trust dial" way up. The friend who hates everything? You turn theirs way down — maybe even into the negatives, so their thumbs-up actually *pushes you away*. Each friend gets their own dial, and it decides how loudly that friend's opinion lands in your head.

A neuron does the same. Every input arrives with its own [[weight||One number per input that scales how much that input matters. Big weight = "listen closely"; near-zero = "ignore"; negative = "count this against."]] — one number that says how much to trust it.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="Three trust dials, one per friend. Each dial is a gauge: the needle points straight up at zero, leans right for a positive weight and left for a negative one. Friend one's needle leans well to the right, weight 0.9, strong trust. Friend two's leans slightly right, weight 0.5, half trust. Friend three's leans left onto the negative side, weight minus 0.6, counting against."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A weight = a trust dial, one per input (straight up = 0)</text><g><circle cx="110" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><path d="M76 80 A 34 34 0 0 1 110 46" fill="none" stroke="#C93B3B" stroke-width="3" opacity="0.55"/><path d="M110 46 A 34 34 0 0 1 144 80" fill="none" stroke="#2D8B55" stroke-width="3" opacity="0.55"/><text x="110" y="38" text-anchor="middle" fill="#9A938A" font-size="9">0</text><text x="64" y="84" text-anchor="middle" fill="#C93B3B" font-size="10">−</text><text x="156" y="84" text-anchor="middle" fill="#276b45" font-size="10">+</text><line x1="110" y1="80" x2="128" y2="59" stroke="#2D8B55" stroke-width="3"/><circle cx="110" cy="80" r="4" fill="#2D8B55"/><text x="110" y="132" text-anchor="middle" fill="#6B645E">good-taste friend</text><text x="110" y="148" text-anchor="middle" fill="#276b45" font-weight="bold">weight = 0.9 (high)</text></g><g><circle cx="260" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><path d="M226 80 A 34 34 0 0 1 260 46" fill="none" stroke="#C93B3B" stroke-width="3" opacity="0.55"/><path d="M260 46 A 34 34 0 0 1 294 80" fill="none" stroke="#2D8B55" stroke-width="3" opacity="0.55"/><text x="260" y="38" text-anchor="middle" fill="#9A938A" font-size="9">0</text><text x="214" y="84" text-anchor="middle" fill="#C93B3B" font-size="10">−</text><text x="306" y="84" text-anchor="middle" fill="#276b45" font-size="10">+</text><line x1="260" y1="80" x2="270" y2="54" stroke="#7C6DAA" stroke-width="3"/><circle cx="260" cy="80" r="4" fill="#7C6DAA"/><text x="260" y="132" text-anchor="middle" fill="#6B645E">half-listened friend</text><text x="260" y="148" text-anchor="middle" fill="#5E5191" font-weight="bold">weight = 0.5 (half)</text></g><g><circle cx="410" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><path d="M376 80 A 34 34 0 0 1 410 46" fill="none" stroke="#C93B3B" stroke-width="3" opacity="0.55"/><path d="M410 46 A 34 34 0 0 1 444 80" fill="none" stroke="#2D8B55" stroke-width="3" opacity="0.55"/><text x="410" y="38" text-anchor="middle" fill="#9A938A" font-size="9">0</text><text x="364" y="84" text-anchor="middle" fill="#C93B3B" font-size="10">−</text><text x="456" y="84" text-anchor="middle" fill="#276b45" font-size="10">+</text><line x1="410" y1="80" x2="397" y2="55" stroke="#C93B3B" stroke-width="3"/><circle cx="410" cy="80" r="4" fill="#C93B3B"/><text x="410" y="132" text-anchor="middle" fill="#6B645E">hates-everything friend</text><text x="410" y="148" text-anchor="middle" fill="#C93B3B" font-weight="bold">weight = −0.6 (against)</text></g><text x="260" y="180" text-anchor="middle" fill="#6B645E" font-size="10">input × weight = how loudly that input lands</text></g></svg>
%%%

**The dial gets it right:** each input has its own setting that scales its say, exactly like turning a knob. **One catch:** you set friend-dials once and forget them, but a neuron's weights are *meant* to be tuned over and over until it stops making mistakes — that tuning is called **training**, and it's what Days 5–6 are for. Today the dials just sit at numbers we hand them.

#### Reading a dial: three settings worth knowing
Stay with the dials for a second, because their whole vocabulary is three positions.

%%% steps
step: dial turned up (weight 0.9)
why: "listen closely" — this input shouts, therefore it dominates the total
step: dial near zero (weight 0.05)
why: "basically ignore you" — the input is still there, but it barely moves the score
step: dial into the negative (weight −0.6)
why: "count this *against*" — which means a bigger input now pushes the answer toward NO, the hates-everything friend in numbers
%%%

%%% insight
This is the small part that turns out to be very powerful. A weight is not just "importance" — its **sign** flips the *meaning* of an input. Same input, positive dial: evidence for. Same input, negative dial: evidence against. One number, two jobs: *how much*, and *which way*.

You already lean on this daily: your spam filter turns its dial on "the words FREE MONEY appear" way up, and its dial on "the sender is in your contacts" *negative* — that input is evidence **against** spam, so it drags the score back down. If "negative weight" still feels slippery, that's normal; it clicks the moment you watch it multiply, which is right now.
%%%

#### See a weight do its one job
A weight has exactly one job: multiply. Let's watch it, with your guess first.

%%% demo id=weigh label="predict, then run"
predict: input 2 with weight 0.5, and input 3 with weight −1.0 — which one comes out negative, and how big?
code: inputs = np.array([2, 3]); weights = np.array([0.5, -1.0]); inputs * weights
out: array([ 1., -3.])
take: <b>Each input is scaled by its own weight.</b> The first input <code>2</code> with weight <code>0.5</code> becomes <code>1</code> — halved, "half-trusted." The second input <code>3</code> with weight <code>−1.0</code> becomes <code>−3</code> — flipped negative, "counted against." That's the whole "weigh" beat: one multiply per input.
%%%

So a weight is a trust dial made of arithmetic. Turn it up, the input shouts; toward zero, it whispers; negative, it argues the other way.

**You just learned the first of a neuron's two sets of numbers.** The second is one lonely number called the bias, and it's next.

@@@ concept id=c3 tag="Bias" title="Bias — the neuron's starting mood" gotit="Got bias"
Here's a feeling you know well. Some mornings you wake up already cheerful — it takes almost nothing to tip you into a "yes." Other mornings you wake up grumpy, and even good news barely moves you.

That built-in starting mood, before *anything* happens today, is the everyday picture of a [[bias||A single extra number added to the neuron's sum. It shifts the neuron's starting point — how easily it leans toward a "yes" before any input arrives.]]. A neuron carries one — and only one.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Two runners at the start of a race. One has a head start ahead of the line (a positive bias, an easy yes). One starts behind the line (a negative bias, a hard yes). The bias just shifts where the neuron begins before any input."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Bias = a head start (or a handicap) before the race begins</text><line x1="260" y1="40" x2="260" y2="150" stroke="#B8AEA2" stroke-width="2" stroke-dasharray="4,4"/><text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">start line (0)</text><circle cx="360" cy="70" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="360" y="75" text-anchor="middle" font-size="13">🏃</text><line x1="274" y1="70" x2="346" y2="70" stroke="#2D8B55" stroke-width="2"/><polygon points="346,64 360,70 346,76" fill="#2D8B55"/><text x="410" y="74" fill="#276b45">+bias: a head start</text><text x="410" y="88" fill="#276b45" font-size="10">easy "yes"</text><circle cx="150" cy="120" r="14" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="150" y="125" text-anchor="middle" font-size="13">🏃</text><line x1="246" y1="120" x2="174" y2="120" stroke="#C93B3B" stroke-width="2"/><polygon points="174,114 160,120 174,126" fill="#C93B3B"/><text x="110" y="102" text-anchor="middle" fill="#C93B3B">−bias: starts behind</text><text x="110" y="140" text-anchor="middle" fill="#C93B3B" font-size="10">hard "yes"</text></g></svg>
%%%

**The head start gets it right:** the bias really does move where the neuron *begins* — how far it leans before hearing a single input. **One catch:** your mood drifts by itself all day; a neuron's bias is a fixed number that only moves during training (Days 5–6).

#### One bias, not one per input
Keep the runner in mind and this all stays simple.

%%% steps
step: the head start is added *once*, at the end
why: the neuron weighs and adds all its inputs first, then drops the bias on top — one nudge for the whole race, not one per runner
step: a big positive bias = the cheerful morning
why: the neuron already leans "yes," therefore even weak input tips it over
step: a big negative bias = the grumpy morning
why: the neuron starts behind the line, which means the inputs must work hard to win it over
%%%

%%% insight
Here's the freedom this one extra number buys: the neuron gets to hold an **opinion before the evidence arrives**. It can wake up already leaning yes, or already leaning no, and make the inputs argue it out of that lean. Your spam filter uses exactly that — it leans **NO** before reading a word, because most of your mail isn't spam.

Take the bias away and that freedom goes: the neuron is pinned to "start at zero," so it can only ever say yes when the **weighed** inputs add up past zero. (Weighed, not raw — remember the friend whose dial is negative can drag a big cheerful input *down*.) Real questions are rarely balanced on zero like that. Small number, big freedom.
%%%

#### Watch the bias shift the total
The weighed inputs from last unit summed to `1 + (−3) = −2`. Now add a bias of `1`.

%%% demo id=bias label="predict, then run"
predict: total was −2, and the head start is +1 — does the answer land at −1, or at −3?
code: weighed = np.array([1.0, -3.0]); bias = 1.0; weighed.sum() + bias
out: -1.0
take: <b>Bias is added once, on top of the summed inputs.</b> The inputs summed to <code>−2</code>; the bias <code>+1</code> nudged the total up to <code>−1</code>. A bigger bias would push it toward "yes"; a negative bias would drag it toward "no." One number, added last.
%%%

So far: weights say *how much each input matters*; the bias says *where the neuron starts from*. Two kinds of number, and that's the neuron's entire memory. Now watch them team up into the neuron's one real calculation.

@@@ concept id=c4 tag="The sum" title="Adding it all up — the neuron's one number" gotit="Got the sum"
Now the "add" beat, where weights and bias finally work together.

Picture a **shopping cart at the till**. Each item's price gets scanned, the register keeps a running total, and at the end the cashier adds one flat service fee on top. One final number on the receipt: your total.

A neuron builds its answer the exact same way, and that number has a name: the [[weighted sum||The running total: every input times its weight, all added up, plus the bias. Also called the pre-activation, because it comes just before the final "decide" step. Written z.]] — often called the **pre-activation**, because it comes right *before* the final "decide" step. Engineers give it a short nickname: `z`.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A shopping-cart receipt. Two scanned items each show price times a weight, they add into a running total, then a flat service fee (the bias) is added on the bottom line to give the final total z."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The weighted sum z = ring up each item, add the fee</text><rect x="150" y="30" width="220" height="150" rx="4" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="150" y1="52" x2="370" y2="52" stroke="#E5DFD6"/><text x="260" y="47" text-anchor="middle" fill="#6B645E">— RECEIPT —</text><text x="166" y="74" fill="#5E5191">input₁ × w₁</text><text x="354" y="74" text-anchor="end" fill="#3A342E">1.0</text><text x="166" y="98" fill="#5E5191">input₂ × w₂</text><text x="354" y="98" text-anchor="end" fill="#3A342E">−3.0</text><line x1="166" y1="110" x2="354" y2="110" stroke="#B8AEA2"/><text x="166" y="128" fill="#6B645E">running total</text><text x="354" y="128" text-anchor="end" fill="#3A342E">−2.0</text><text x="166" y="150" fill="#8A6D3B">+ bias (fee)</text><text x="354" y="150" text-anchor="end" fill="#8A6D3B">+1.0</text><line x1="166" y1="160" x2="354" y2="160" stroke="#3A342E" stroke-width="1.5"/><text x="166" y="176" fill="#2D8B55" font-weight="bold">TOTAL  z</text><text x="354" y="176" text-anchor="end" fill="#2D8B55" font-weight="bold">−1.0</text></g></svg>
%%%

**The receipt gets it right:** scan each item, keep a running total, add one fee at the end — that's the neuron's exact order. **One catch:** on a receipt every price is positive, but a weighed input can be *negative*, so an item can *subtract* from the total, which no real till ever does.

#### The one line of math, said out loud
%%% formula
expr: z = w₁·x₁ + w₂·x₂ + … + b
note: z is the total; each x is an input; each w is that input's weight; b is the one bias. In words: "weigh every input, add them up, then add the bias."
%%%

Read it left to right: `w₁·x₁` is "input one times its weight," you add on `w₂·x₂` for input two, keep going for every input, and finally add `b`, the bias. That's every symbol. Nothing hidden.

%%% insight
Notice how *small* that line is. Two multiplies and two adds — a calculator could do it. And yet: this exact line, copied across millions of tiny neurons, is what writes poetry and spots tumours. The power was never in the formula's cleverness; it's in how many copies get stacked and tuned. That's a genuinely surprising fact about AI, and you now know it first-hand.
%%%

#### Ring up the receipt, one line at a time
Let's build `z` the way the till does — one item, then the next, then the fee. Watch the running total change at each rung.

%%% steps
step: start the receipt at 0
why: nothing scanned yet, therefore the running total is empty — this is the till before you unload the cart
step: scan item 1 → 0.5 × 2 = 1.0 → total is 1.0
why: the first input times its own trust dial; a positive dial, so it *adds* to the total
step: scan item 2 → −1.0 × 3 = −3.0 → total drops to −2.0
why: the negative dial means this item *subtracts* — which is why the total goes down even though the input was a cheerful 3
step: add the fee → −2.0 + 1.0 = −1.0
why: the bias lands once, at the end; the head start lifts the total from −2.0 to −1.0
step: read the bottom line → z = −1.0
why: that single number is everything the neuron has to say, right before it decides
%%%

Same ladder, drawn to scale as bars — because a picture of a running total is far faster to read than a line of arithmetic. Every `1.0` of value is the same bar height, so you can read the sizes straight off:

%%% svg
<svg viewBox="0 0 520 264" role="img" aria-label="A running-total bar chart drawn to scale, thirty pixels per unit, against a zero line with tick marks at plus one, zero, minus one, minus two and minus three. The first bar is plus one for the first weighed input, running total one. The second bar is minus three, three times as tall as the first and pointing down, dropping the running total to minus two. The third bar is plus one for the bias, lifting the total to minus one. The final short downward bar is z equals minus one."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">The total z builds up piece by piece (30 px = 1.0, everywhere)</text><line x1="50" y1="90" x2="480" y2="90" stroke="#EFE9DF" stroke-width="1" stroke-dasharray="3,3"/><line x1="50" y1="150" x2="480" y2="150" stroke="#EFE9DF" stroke-width="1" stroke-dasharray="3,3"/><line x1="50" y1="180" x2="480" y2="180" stroke="#EFE9DF" stroke-width="1" stroke-dasharray="3,3"/><line x1="50" y1="210" x2="480" y2="210" stroke="#EFE9DF" stroke-width="1" stroke-dasharray="3,3"/><line x1="50" y1="120" x2="480" y2="120" stroke="#B8AEA2" stroke-width="1.5"/><text x="44" y="94" text-anchor="end" fill="#9A938A" font-size="9">+1</text><text x="44" y="124" text-anchor="end" fill="#9A938A" font-size="9">0</text><text x="44" y="154" text-anchor="end" fill="#9A938A" font-size="9">−1</text><text x="44" y="184" text-anchor="end" fill="#9A938A" font-size="9">−2</text><text x="44" y="214" text-anchor="end" fill="#9A938A" font-size="9">−3</text><rect x="70" y="90" width="70" height="30" fill="#7C6DAA" opacity="0.85"/><text x="105" y="110" fill="#fff">+1.0</text><text x="105" y="230" fill="#6B645E" font-size="10">w₁·x₁</text><text x="105" y="244" fill="#5E5191" font-size="10">total: 1.0</text><rect x="175" y="120" width="70" height="90" fill="#C93B3B" opacity="0.85"/><text x="210" y="172" fill="#fff">−3.0</text><text x="210" y="230" fill="#6B645E" font-size="10">w₂·x₂</text><text x="210" y="244" fill="#C93B3B" font-size="10">total: −2.0</text><rect x="280" y="90" width="70" height="30" fill="#C99A12" opacity="0.85"/><text x="315" y="110" fill="#fff">+1.0</text><text x="315" y="230" fill="#6B645E" font-size="10">bias b</text><text x="315" y="244" fill="#8A6D3B" font-size="10">total: −1.0</text><rect x="390" y="120" width="70" height="30" fill="#2D8B55"/><text x="425" y="140" fill="#fff">z</text><text x="425" y="230" fill="#276b45" font-size="11" font-weight="bold">z = −1.0</text><text x="260" y="260" fill="#9A938A" font-size="9">the −3.0 bar really is three times the +1.0 bar — that's why the total dips below zero</text></g></svg>
%%%

%%% demo id=zsum label="predict, then run"
predict: you rang it up by hand above — does the code agree that z lands at −1.0?
code: x = np.array([2, 3]); w = np.array([0.5, -1.0]); b = 1.0; z = (w * x).sum() + b; z
out: -1.0
take: <b>z = (w·x summed) + b = −1.0.</b> Two inputs weighed to <code>1.0</code> and <code>−3.0</code>, summed to <code>−2.0</code>, plus the bias <code>1.0</code>, lands at <code>z = −1.0</code>. Exactly the receipt you built by hand.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the same sum in one breath.</b> <code>0.5 × 2 = 1.0</code>; <code>−1.0 × 3 = −3.0</code>; <code>1.0 + (−3.0) = −2.0</code>; <code>−2.0 + 1.0 = −1.0</code>. Two multiplies, two adds. You'll type it yourself in <b>Produce</b>.
!!!

**You just did the only real arithmetic a neuron ever does.** If the bars made sense, you are past the hardest arithmetic of the day — everything left is what *happens* to that one number `z`.

@@@ concept id=c5 tag="Decide" title="The decide step — turning z into an answer" gotit="Got the decide step"
The neuron has its one number, `z`. But a raw score isn't yet an *answer*.

Picture a **judge at the end of a talent show**. The scores are in and totalled — that's `z` — and now the judge has to turn that total into a verdict the audience understands: "you're through" or "you're out," or maybe a confidence like "87%." Turning a total into a usable answer is the last beat, **decide**.

The step that does it is the [[activation function||The last step of a neuron: it takes the weighted sum z and turns it into the neuron's output. There's a whole family of them — a step, a smooth S-curve, and more.]]. Carry away this shape: a neuron is always the **linear part** (the weigh-and-add that gives `z`) *then* the **activation** (the decide step on top of `z`).

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A talent-show judge. On the left, a scoreboard shows the total score z. An arrow leads to a judge who turns that score into a verdict: a yes or no, or a confidence percentage. The judge is the activation function."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The activation = a judge turning the score z into a verdict</text><rect x="40" y="60" width="120" height="60" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="100" y="82" text-anchor="middle" fill="#5E5191" font-size="10">weighted sum</text><text x="100" y="105" text-anchor="middle" fill="#3A342E" font-size="16" font-weight="bold">z = −1.0</text><text x="100" y="140" text-anchor="middle" fill="#6B645E" font-size="10">the linear part</text><line x1="164" y1="90" x2="230" y2="90" stroke="#B8AEA2" stroke-width="2.5"/><polygon points="230,84 244,90 230,96" fill="#B8AEA2"/><circle cx="300" cy="90" r="34" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="300" y="86" text-anchor="middle" font-size="18">⚖️</text><text x="300" y="104" text-anchor="middle" fill="#276b45" font-size="9">judge</text><text x="300" y="150" text-anchor="middle" fill="#276b45" font-size="10">the activation</text><line x1="336" y1="90" x2="400" y2="90" stroke="#B8AEA2" stroke-width="2.5"/><polygon points="400,84 414,90 400,96" fill="#B8AEA2"/><rect x="418" y="66" width="90" height="48" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="463" y="86" text-anchor="middle" fill="#8A6D3B" font-size="10">verdict</text><text x="463" y="104" text-anchor="middle" fill="#8A6D3B" font-size="10">yes / no / 87%</text></g></svg>
%%%

**The judge gets it right:** a raw total is not a verdict, and something must *convert* one into the other — that conversion is the activation's whole job. **One catch:** a human judge has hunches; a neuron's activation is one fixed rule applied the same way every time.

#### Watch the judge build a verdict
Don't take "it turns `z` into an answer" on faith — let's watch the judge work, using the simplest possible rule (a hard yes/no at zero).

%%% steps
step: the score arrives — z = −1.0
why: straight off the receipt you built last unit; the judge invents nothing, it only reads what the sum handed over
step: the judge asks its one question — "is z ≥ 0?"
why: that's the entire rule, therefore the judge needs no opinion of its own
step: the answer here is *no*, because −1.0 sits below 0
why: this is the fork — flip z positive and this same question answers "yes" instead
step: read off the verdict → 0, meaning "off"
why: the no-answer forces a 0; a yes-answer would have forced a 1 ("on")
%%%

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Three steps that build the verdict from a raw score. Step one: the raw score z equals minus one. Step two: ask is z at or above zero? Here the answer is no. Step three: the verdict is zero, meaning off. Each step feeds the next."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">z → ask one question → the verdict</text><rect x="30" y="55" width="130" height="66" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="95" y="76" fill="#6B645E" font-size="10">1 · the raw score</text><text x="95" y="100" fill="#3A342E" font-size="16" font-weight="bold">z = −1.0</text><text x="95" y="116" fill="#9A938A" font-size="9">(from weigh + add)</text><line x1="162" y1="88" x2="192" y2="88" stroke="#B8AEA2" stroke-width="2"/><polygon points="192,83 202,88 192,93" fill="#B8AEA2"/><rect x="205" y="55" width="130" height="66" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="270" y="76" fill="#8A6D3B" font-size="10">2 · ask one thing</text><text x="270" y="98" fill="#3A342E" font-size="12">is z ≥ 0 ?</text><text x="270" y="116" fill="#C93B3B" font-size="11" font-weight="bold">no (−1.0 &lt; 0)</text><line x1="337" y1="88" x2="367" y2="88" stroke="#B8AEA2" stroke-width="2"/><polygon points="367,83 377,88 367,93" fill="#B8AEA2"/><rect x="380" y="55" width="120" height="66" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="440" y="76" fill="#6B645E" font-size="10">3 · the verdict</text><text x="440" y="100" fill="#C93B3B" font-size="16" font-weight="bold">0 (off)</text><text x="440" y="116" fill="#9A938A" font-size="9">the neuron's answer</text><text x="260" y="150" fill="#6B645E" font-size="10">raw score → one question → answer: that's the whole "decide" beat</text><text x="260" y="166" fill="#9A938A" font-size="9">a positive z would flip step 2 to "yes" and the verdict to 1 (on)</text></g></svg>
%%%

%%% insight
This little three-box picture is your defence against the biggest myth in AI. People say neural nets are "magic black boxes." But look: the answer was *assembled* — a score, one question, a verdict. No magic anywhere. Every answer ChatGPT gives you is built from millions of steps exactly this plain. Knowing that is genuinely a superpower in a room full of people who don't.
%%%

#### It's a whole family, not one rule
Here's the fun part: there isn't *one* activation. There's a family, and choosing between them is a real design decision.

- A blunt on/off "yes or no" — the one you just watched the judge use.
- A smooth "how confident, 0 to 100%."
- Several more you'll meet tomorrow.

So far: every neuron is *linear part, then activation*. Next up, the very first activation anyone ever tried — and the puzzle it left behind.

@@@ concept id=c6 tag="First activation" title="The step function — a hard on/off switch" gotit="Met the step"
Let's meet the very first activation anyone ever built. It's the great-grandparent of the whole family, from the 1940s, and its idea is beautifully simple.

Picture the **height bar at a theme-park ride**: "you must be *this* tall to ride." A tall-enough rider clears the bar and gets waved on. A too-short rider is turned away at the gate. There's no "you're 80% on the ride" — a hard yes or no, decided by one fixed bar.

That is the [[step function||The original activation: if the weighted sum z is at or above a threshold (zero), output 1 (fire); otherwise output 0. A hard on/off switch, with nothing in between.]] — the "decide" rule inside the earliest artificial neurons (people called them the **McCulloch–Pitts neuron**, and later the **Perceptron**).

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A theme-park height bar. On the left a short person is below the bar and is turned away (output 0, off). On the right a tall person clears the bar and is let on (output 1, on). It is a hard cutoff with nothing in between."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Step = a height bar: tall enough (on) or not (off)</text><line x1="260" y1="45" x2="260" y2="150" stroke="#C99A12" stroke-width="3"/><text x="260" y="42" text-anchor="middle" fill="#8A6D3B" font-size="10">the bar (z = 0)</text><rect x="120" y="105" width="16" height="45" fill="#C93B3B" opacity="0.8"/><circle cx="128" cy="98" r="9" fill="#C93B3B" opacity="0.8"/><text x="128" y="164" text-anchor="middle" fill="#C93B3B" font-size="10">below the bar</text><text x="60" y="90" fill="#C93B3B" font-weight="bold">z &lt; 0 → 0 (off)</text><rect x="384" y="70" width="16" height="80" fill="#2D8B55"/><circle cx="392" cy="62" r="9" fill="#2D8B55"/><text x="392" y="164" text-anchor="middle" fill="#276b45" font-size="10">clears the bar</text><text x="420" y="90" text-anchor="end" fill="#276b45" font-weight="bold">z ≥ 0 → 1 (on)</text></g></svg>
%%%

**The height bar gets it right:** one fixed bar, two outcomes, no middle ground. **One catch:** at a real ride *you* stand at the bar, while here the neuron's own number `z` is what gets measured — and the bar never tells you *how far* over or under you were.

#### Line four riders up at the bar
Let's build the step one rider at a time. Each rider is a score `z`; the attendant only asks "did you reach the bar (z = 0)?"

%%% steps
step: rider z = −1.2 → waved away, 0
why: well below the bar, therefore an easy "no"
step: rider z = −0.3 → waved away, 0
why: closer, but still short — and the attendant gives no credit for being close
step: rider z = 0.1 → waved on, 1
why: *just* clears the bar, which means the verdict flips the instant the bar is reached
step: rider z = 2.5 → waved on, 1
why: towers over the bar, yet gets the *identical* answer as 0.1 — the bar never asks HOW tall
%%%

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Four riders lined up by their score z at a theme-park height bar. The bar sits at z equals zero. A rider with z of minus one point two is below the bar and turned away, output zero. A rider with z of minus zero point three is still below the bar, output zero. A rider with z of zero point one just clears the bar, output one. A rider with z of two point five clears it easily, output one. The verdict flips from zero to one the moment a rider reaches the bar."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Build the step: each rider meets the bar, gets waved on (1) or away (0)</text><line x1="270" y1="34" x2="270" y2="150" stroke="#C99A12" stroke-width="3"/><text x="270" y="30" fill="#8A6D3B" font-size="9">the bar (z = 0)</text><text x="120" y="30" fill="#C93B3B" font-size="9">below → away (0)</text><text x="420" y="30" fill="#276b45" font-size="9">at/above → on (1)</text><line x1="40" y1="150" x2="500" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><g><circle cx="90" cy="128" r="8" fill="#C93B3B" opacity="0.85"/><rect x="86" y="136" width="8" height="14" fill="#C93B3B" opacity="0.85"/><text x="90" y="172" fill="#6B645E" font-size="9">z = −1.2</text><text x="90" y="186" fill="#C93B3B" font-size="10" font-weight="bold">→ 0</text></g><g><circle cx="200" cy="128" r="8" fill="#C93B3B" opacity="0.85"/><rect x="196" y="136" width="8" height="14" fill="#C93B3B" opacity="0.85"/><text x="200" y="172" fill="#6B645E" font-size="9">z = −0.3</text><text x="200" y="186" fill="#C93B3B" font-size="10" font-weight="bold">→ 0</text></g><g><circle cx="330" cy="120" r="8" fill="#2D8B55"/><rect x="326" y="128" width="8" height="22" fill="#2D8B55"/><text x="330" y="172" fill="#6B645E" font-size="9">z = 0.1</text><text x="330" y="186" fill="#276b45" font-size="10" font-weight="bold">→ 1</text></g><g><circle cx="450" cy="112" r="8" fill="#2D8B55"/><rect x="446" y="120" width="8" height="30" fill="#2D8B55"/><text x="450" y="172" fill="#6B645E" font-size="9">z = 2.5</text><text x="450" y="186" fill="#276b45" font-size="10" font-weight="bold">→ 1</text></g><text x="260" y="204" fill="#9A938A" font-size="9">notice: 0.1 and 2.5 both just say "on" — the bar never asks HOW tall, only "tall enough?"</text></g></svg>
%%%

Draw that snap as a curve and you get the step's famous shape — flat at `0`, a sudden jump at the bar, flat at `1`:

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Step function curve: output 0 for negative inputs, a sudden jump up to 1 at zero, then flat at 1 for positive inputs."><g font-family="monospace"><line x1="60" y1="100" x2="340" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><line x1="200" y1="30" x2="200" y2="118" stroke="#E5DFD6" stroke-width="1"/><path d="M70 100 L200 100 L200 48 L320 48" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="200" cy="48" r="3.5" fill="#8A6D3B"/><text x="130" y="117" font-size="11" fill="#6B645E" text-anchor="middle">output 0 (away)</text><text x="265" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">output 1 (on)</text><text x="200" y="133" font-size="10" fill="#9A938A" text-anchor="middle">the same snap, drawn: a cliff at the bar — no in-between</text><text x="420" y="76" font-size="12" fill="#8A6D3B">step → {0, 1}</text></g></svg>
%%%

%%% demo id=step label="predict, then run"
predict: same four riders. Will 0.1 and 2.5 get *different* answers, or the same one?
code: step(np.array([-1.2, -0.3, 0.1, 2.5]))
out: array([0., 0., 1., 1.])
take: <b>Step = 1 if z ≥ 0, else 0.</b> The two below the bar become <code>0</code>, the two at or above it become <code>1</code>. Notice <code>0.1</code> and <code>2.5</code> give the identical answer — the bar never asks "how tall," only "tall enough?"
%%%

%%% insight
And *that* is the puzzle the step leaves behind. Improving a neuron means nudging its dials a hair and checking whether the answer got a little better. But the height bar is a **cliff**: nudge a rider by a millimetre and nothing budges — until they tumble over the edge and it flips completely. No small change, no hint, no direction to improve in.

For *one* neuron there was a clever way around it: Rosenblatt's **Perceptron** (1957) touched its dials only when it got an example *wrong*, nudging them toward the right answer. It worked — the first neuron that learned its own weights. But stack neurons (the thing that makes networks powerful) and "which way do I nudge a dial in the *middle* of the stack?" has no answer at all, because a cliff hands back no direction. Swapping the cliff for a gentle slope is what makes a whole stack improvable — the machinery of Day 5.
%%%

You just diagnosed a real design problem, on Day 1. **The fix is beautifully simple:** swap the cliff for a ramp — and that's next.

@@@ concept id=c7 tag="Smooth version" title="Sigmoid — a smooth dimmer, not a cliff" gotit="Met sigmoid"
So we want the "decide" step to change *gradually*, so a small nudge to the dials makes a small, visible change in the answer.

Swap the light *switch* for a **dimmer**. A dimmer slides smoothly from off to fully on, passing through every level of brightness in between — dim, medium, bright. Nudge it a hair and the light changes a hair.

That smooth dimmer is the [[sigmoid||A smooth S-shaped activation that squashes any number into the range 0 to 1. It replaces the step's hard cliff with a gentle slide, so a small change in z makes a small change in the output.]] activation: it takes any score `z` and squashes it into the range 0 to 1 — read it as "how confident, from 0% to 100%."

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A dimmer switch shown as a horizontal slider. On the left the slider is low and the bulb is dark, near zero. In the middle it is half bright at one half. On the right it is full and the bulb is fully bright, near one. It slides smoothly through every level in between."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid = a dimmer that slides smoothly off → on</text><rect x="60" y="78" width="400" height="14" rx="7" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="110" cy="85" r="12" fill="#5E5191"/><text x="110" y="114" text-anchor="middle" fill="#6B645E">low</text><text x="110" y="128" text-anchor="middle" fill="#9A938A" font-size="10">→ near 0</text><circle cx="260" cy="58" r="16" fill="#EFEAF0" stroke="#9A938A"/><text x="260" y="64" text-anchor="middle" font-size="14">🌗</text><circle cx="350" cy="54" r="18" fill="#F6EFCF" stroke="#C99A12"/><text x="350" y="60" text-anchor="middle" font-size="14">🔆</text><circle cx="430" cy="50" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="430" y="57" text-anchor="middle" font-size="16">💡</text><text x="260" y="114" text-anchor="middle" fill="#5E5191">middle</text><text x="260" y="128" text-anchor="middle" fill="#9A938A" font-size="10">= 0.5</text><text x="430" y="114" text-anchor="middle" fill="#8A6D3B">full</text><text x="430" y="128" text-anchor="middle" fill="#9A938A" font-size="10">→ near 1</text><text x="260" y="158" text-anchor="middle" fill="#6B645E" font-size="10">no hard jump — every level in between is a real answer in (0, 1)</text></g></svg>
%%%

**The dimmer gets it right:** a smooth, gradual slide from off to on, every brightness in between a real setting. **One catch:** a real dimmer is evenly graded end to end, but sigmoid *flattens out* at the far ends — push the score very high or very low and the light barely changes. (That flattening hides a puzzle you'll crack tomorrow.)

#### Slide the dimmer through three positions
Stay on the dimmer knob and read off what the light does.

%%% steps
step: slide it far left (z = −4) → about 0.02
why: the bulb is nearly dark, therefore the neuron is saying "almost certainly no" — but not a flat zero, it still leaves a sliver
step: park it dead centre (z = 0) → exactly 0.5
why: perfectly undecided, which is the honest answer when the evidence cancels out
step: slide it far right (z = +4) → about 0.98
why: nearly full brightness — "almost certainly yes," and again it never quite reaches 1
step: nudge it anywhere by a hair → the light changes by a hair
why: that's the whole win over the cliff; a small change in z always makes a small, readable change in the answer
%%%

Here is the point of this unit in one figure — the step's cliff (left) versus sigmoid's smooth ramp (right). Look for the dot: at `z = 0` the ramp sits exactly halfway up, at `0.5`.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="Left panel: the step function as a sharp cliff jumping from 0 to 1. Right panel: the sigmoid as a smooth S-shaped ramp that passes exactly halfway up, at 0.5, where z equals zero, then keeps rising toward 1. The cliff gives no in-between; the ramp does."><g font-family="monospace" font-size="11"><text x="130" y="20" text-anchor="middle" fill="#C93B3B" font-size="11">step: a cliff (no in-between)</text><line x1="40" y1="120" x2="230" y2="120" stroke="#E5DFD6" stroke-width="1.5"/><line x1="135" y1="45" x2="135" y2="132" stroke="#E5DFD6" stroke-width="1"/><path d="M50 120 L135 120 L135 60 L225 60" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="135" cy="60" r="3" fill="#8A6D3B"/><text x="130" y="150" text-anchor="middle" fill="#9A938A" font-size="10">tiny nudge → nothing moves</text><text x="390" y="20" text-anchor="middle" fill="#276b45" font-size="11">sigmoid: a smooth ramp</text><line x1="295" y1="118" x2="490" y2="118" stroke="#E5DFD6" stroke-width="1.5"/><line x1="392" y1="45" x2="392" y2="132" stroke="#E5DFD6" stroke-width="1"/><line x1="295" y1="88" x2="490" y2="88" stroke="#E5DFD6" stroke-width="1" stroke-dasharray="3,3"/><path d="M300 117 C 342 117, 372 110, 392 88 C 412 66, 442 59, 485 58" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><circle cx="392" cy="88" r="4" fill="#5E5191"/><text x="404" y="84" fill="#5E5191" font-size="9">0.5 at z = 0</text><text x="300" y="132" fill="#9A938A" font-size="9">near 0</text><text x="482" y="52" text-anchor="end" fill="#9A938A" font-size="9">near 1</text><text x="390" y="150" text-anchor="middle" fill="#2D8B55" font-size="10">tiny nudge → answer moves a little</text></g></svg>
%%%

%%% demo id=sigmoid label="predict, then run"
predict: five scores from −4 to +4 go into the dimmer. Which one comes out at exactly 0.5?
code: sigmoid(np.array([-4.0, -1.0, 0.0, 1.0, 4.0]))
out: array([0.018, 0.269, 0.5  , 0.731, 0.982])
take: <b>Sigmoid squashes any score into (0, 1).</b> A smooth off → on: <code>sigmoid(0) = 0.5</code> sits dead centre, big negatives land near <code>0</code>, big positives near <code>1</code> (values rounded). Because every answer lands between 0 and 1, it reads like a confidence — perfect when you want "how sure, 0 to 100%."
%%%

%%% insight
Feel the difference in one sentence: with the step, a small nudge changed *nothing* (or flipped everything). With the dimmer, a small nudge makes a small, honest change you can *follow* — and "follow the change" is literally how every model on earth learns. That swap, cliff → ramp, is what lets us improve a whole *stack* of neurons at once instead of just one.
%%%

**You now know both bookends of the activation family:** the original hard step, and the smooth sigmoid that replaced it. Tomorrow you'll meet the rest of the family. Today, let's watch the *whole* neuron run start to finish.

@@@ concept id=c8 tag="Forward pass" title="Forward propagation — run the whole neuron" gotit="Got the forward pass"
Let's run the whole brain cell end to end: **weigh → add → decide**, in one clean sweep.

Picture a **factory conveyor belt**. Raw parts go on at one end, move steadily to the right through station after station, and a finished product rolls off the far end. Nothing goes backward; each station does its one job and passes the work along.

A neuron computes its answer exactly like that, and the sweep has a name: [[forward propagation||Pushing the inputs left-to-right through the neuron — first the weighted sum, then the activation — to produce the output. Also called the forward pass.]], or the **forward pass**.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A factory conveyor belt moving left to right. Inputs enter on the left, pass a weigh-and-add station that produces the score z, then a decide station holding the activation, and a finished output rolls off the right end. Everything flows one way."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Forward pass = a conveyor: inputs in, one answer out</text><rect x="30" y="120" width="460" height="10" rx="5" fill="#D8D2C8"/><circle cx="45" cy="135" r="8" fill="#B8AEA2"/><circle cx="120" cy="135" r="8" fill="#B8AEA2"/><circle cx="260" cy="135" r="8" fill="#B8AEA2"/><circle cx="400" cy="135" r="8" fill="#B8AEA2"/><circle cx="475" cy="135" r="8" fill="#B8AEA2"/><rect x="40" y="55" width="70" height="45" rx="5" fill="#FDF9F3" stroke="#B8AEA2"/><text x="75" y="74" text-anchor="middle" fill="#6B645E" font-size="10">inputs</text><text x="75" y="90" text-anchor="middle" fill="#3A342E" font-size="10">x = [2, 3]</text><text x="122" y="82" fill="#B8AEA2">→</text><rect x="145" y="55" width="110" height="45" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="200" y="74" text-anchor="middle" fill="#5E5191" font-size="10">weigh &amp; add</text><text x="200" y="90" text-anchor="middle" fill="#3A342E" font-size="10">z = −1.0</text><text x="267" y="82" fill="#B8AEA2">→</text><rect x="290" y="55" width="110" height="45" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="345" y="74" text-anchor="middle" fill="#276b45" font-size="10">decide (sigmoid)</text><text x="345" y="90" text-anchor="middle" fill="#3A342E" font-size="10">→ 0.27</text><text x="412" y="82" fill="#B8AEA2">→</text><rect x="430" y="55" width="60" height="45" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="460" y="74" text-anchor="middle" fill="#8A6D3B" font-size="10">output</text><text x="460" y="90" text-anchor="middle" fill="#8A6D3B" font-size="10">0.27</text><text x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="10">one direction only — left to right, no going back</text></g></svg>
%%%

**The conveyor gets it right:** work moves one way, each station does its job once, a finished result comes off the end. **One catch:** a factory belt never reverses, but a neuron *does* have a backward trip — that's how it learns (backprop, a later day). Today we only ride the belt forward.

#### Ride the belt, station by station
Nothing new here — just your three beats, bolted onto the conveyor in order.

%%% steps
step: station 0 · the parts arrive → x = [2, 3]
why: two raw input numbers get placed on the belt; the neuron never sees anything else about the world
step: station 1 · weigh & add → z = −1.0
why: each input meets its trust dial and the bias lands on top, therefore many numbers become one score
step: station 2 · decide (sigmoid) → 0.269
why: the dimmer turns that score into a readable confidence, which means the answer is now something a human can use
step: off the end · output 0.269
why: a soft "probably no, about 27% yes" — one number, from one pass, in one direction
%%%

%%% demo id=forward label="predict, then run"
predict: z will be −1.0 again. Since sigmoid(0) = 0.5, will the final answer sit above or below 0.5?
code: x = np.array([2, 3]); w = np.array([0.5, -1.0]); b = 1.0; z = (w*x).sum() + b; out = sigmoid(z); (z, round(out, 3))
out: (-1.0, 0.269)
take: <b>That's a full forward pass.</b> The inputs flowed through weigh-and-add to <code>z = −1.0</code>, then the sigmoid "decide" step turned that into <code>0.269</code> — a soft "probably no, about 27% yes." Weigh → add → decide, start to finish, in one direction.
%%%

%%% insight
Pause on this, because it's a bigger moment than it looks: you just ran the *same computation* the largest models on earth run. Not a simplified cartoon of it — the actual operation. When a giant model answers you, millions of these little forward passes fire at once, on the same conveyor, in the same order. The scale is different. The step is identical.
%%%

So far: a complete neuron, computed end to end. Now for the twist — this tidy little machine has one thing it flat-out *cannot* do.

@@@ concept id=c9 tag="The line" title="A neuron draws one straight line" gotit="Got the line"
Here's a beautiful way to *see* what a neuron really does.

Imagine a **playground with kids scattered around**, and you're picking two teams. You grab a piece of chalk and draw one straight line across the ground: everyone on this side, team A; everyone on that side, team B. One straight chalk line, and the whole yard is split in two.

A neuron makes exactly that kind of call. Picture every possible pair of inputs as a dot on a flat map — the neuron's weights and bias together draw **one straight line** across it, a [[decision boundary||The straight line (or flat plane) a neuron draws to split its inputs into two sides — "yes" on one side, "no" on the other. Its position and tilt come from the weights and bias.]].

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A playground map with scattered dots. One straight chalk line splits them into two teams: green dots on one side get a yes, red dots on the other get a no. The line's tilt is set by the weights and its position by the bias."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A neuron = one straight chalk line splitting the yard</text><rect x="150" y="32" width="220" height="130" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="160" y1="150" x2="360" y2="44" stroke="#9A5A12" stroke-width="3"/><text x="378" y="46" fill="#9A5A12" font-size="10">the line</text><circle cx="195" cy="70" r="7" fill="#2D8B55"/><circle cx="230" cy="58" r="7" fill="#2D8B55"/><circle cx="215" cy="95" r="7" fill="#2D8B55"/><circle cx="260" cy="72" r="7" fill="#2D8B55"/><text x="205" y="120" text-anchor="middle" fill="#276b45" font-size="10">"yes" side</text><circle cx="300" cy="120" r="7" fill="#C93B3B"/><circle cx="330" cy="135" r="7" fill="#C93B3B"/><circle cx="315" cy="105" r="7" fill="#C93B3B"/><circle cx="345" cy="118" r="7" fill="#C93B3B"/><text x="330" y="90" text-anchor="middle" fill="#C93B3B" font-size="10">"no" side</text><text x="70" y="80" text-anchor="middle" fill="#6B645E" font-size="10">weights</text><text x="70" y="94" text-anchor="middle" fill="#6B645E" font-size="10">tilt it</text><text x="70" y="126" text-anchor="middle" fill="#6B645E" font-size="10">bias</text><text x="70" y="140" text-anchor="middle" fill="#6B645E" font-size="10">slides it</text></g></svg>
%%%

**The chalk line gets it right:** the neuron carves its world into two sides with a single straight cut, and where that cut sits is *all* it controls. **One catch:** with more than two inputs the "line" becomes a flat sheet in a space too big to sketch — same idea, no drawing.

#### Grab the dials and move the line yourself
Forget my word for it — **play.** Three things to try, and predict each before you let go:

- Drag **w₁** or **w₂** and watch the line **tilt**. Weights set which direction counts as "more yes."
- Drag **b** and watch the same-angled line **slide** across the yard. Nothing tilts; it just shifts.
- Now try to make it **bend**. Any combination you like. (Watch the **yes ✓** label too — drag both weights to −2 and it jumps to the opposite corner.)

%%% svg
<svg id="db-svg" viewBox="0 0 520 232" role="img" aria-label="Interactive decision boundary. Drag the weight and bias sliders and watch the neuron's straight line tilt and slide across a map of dots, recoloring each dot green for yes or red for no. The yes and no labels move to whichever corner is actually on that side."><g font-family="monospace" font-size="11"><rect x="60" y="30" width="400" height="180" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="120" x2="460" y2="120" stroke="#EFE9DF" stroke-width="1"/><line x1="260" y1="30" x2="260" y2="210" stroke="#EFE9DF" stroke-width="1"/><line id="db-line" x1="60" y1="30" x2="460" y2="210" stroke="#9A5A12" stroke-width="3"/><circle id="db-p0" cx="110" cy="66" r="7" fill="#C93B3B"/><circle id="db-p1" cx="180" cy="102" r="7" fill="#C93B3B"/><circle id="db-p2" cx="310" cy="53" r="7" fill="#2D8B55"/><circle id="db-p3" cx="380" cy="84" r="7" fill="#2D8B55"/><circle id="db-p4" cx="140" cy="161" r="7" fill="#C93B3B"/><circle id="db-p5" cx="290" cy="183" r="7" fill="#C93B3B"/><circle id="db-p6" cx="400" cy="147" r="7" fill="#2D8B55"/><circle id="db-p7" cx="220" cy="143" r="7" fill="#C93B3B"/><text id="db-lt" x="435" y="45" text-anchor="middle" fill="#276b45" font-size="10">yes ✓</text><text id="db-lb" x="85" y="199" text-anchor="middle" fill="#C93B3B" font-size="10">no ✗</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
<label>w₁ <input id="db-w1" type="range" min="-2" max="2" step="0.1" value="1" style="accent-color:#9A5A12;vertical-align:middle" aria-label="weight one"> <b id="db-w1v">1.0</b></label>
<label>w₂ <input id="db-w2" type="range" min="-2" max="2" step="0.1" value="1" style="accent-color:#9A5A12;vertical-align:middle" aria-label="weight two"> <b id="db-w2v">1.0</b></label>
<label>b <input id="db-b" type="range" min="-2" max="2" step="0.1" value="0" style="accent-color:#9A5A12;vertical-align:middle" aria-label="bias"> <b id="db-bv">0.0</b></label>
</div>
<div id="db-out" style="margin-top:6px;color:#2C2A28">boundary: <b>1.0·x₁ + 1.0·x₂ + 0.0 = 0</b> — the weights tilt this line, the bias slides it.</div>
</div>
<script>(function(){
  var w1=document.getElementById('db-w1');if(!w1)return;
  var w2=document.getElementById('db-w2'),b=document.getElementById('db-b'),
      line=document.getElementById('db-line'),out=document.getElementById('db-out'),
      w1v=document.getElementById('db-w1v'),w2v=document.getElementById('db-w2v'),bv=document.getElementById('db-bv'),
      lt=document.getElementById('db-lt'),lb=document.getElementById('db-lb');
  var pts=[[-1.5,1.2],[-0.8,0.4],[0.5,1.5],[1.2,0.8],[-1.2,-0.9],[0.3,-1.4],[1.4,-0.6],[-0.4,-0.5]];
  function px(x){return 60+(x+2)/4*400;}
  function py(y){return 210-(y+2)/4*180;}
  // where the boundary meets the edges of the -2..2 box. DE-DUPLICATED: when the line
  // passes exactly through a corner, the x1-scan and the x2-scan find the SAME point, and
  // pushing it twice used to draw a zero-length (invisible) line while the dots stayed split.
  function ends(a,c,d){var lo=-2,hi=2,e=[];
    function add(x1,x2){
      for(var i=0;i<e.length;i++){if(Math.abs(e[i][0]-x1)<1e-6&&Math.abs(e[i][1]-x2)<1e-6)return;}
      e.push([x1,x2]);}
    if(Math.abs(c)>1e-9){[lo,hi].forEach(function(x1){var x2=-(a*x1+d)/c;if(x2>=lo-1e-9&&x2<=hi+1e-9)add(x1,x2);});}
    if(Math.abs(a)>1e-9){[lo,hi].forEach(function(x2){var x1=-(c*x2+d)/a;if(x1>=lo-1e-9&&x1<=hi+1e-9)add(x1,x2);});}
    return e;}
  function sgn(n){return (n<0?'− ':'+ ')+Math.abs(n).toFixed(1);}
  // the yes/no labels are NOT fixed: each one MOVES to a corner that is really on that side
  var corners=[[-1.75,1.75],[1.75,1.75],[-1.75,-1.75],[1.75,-1.75]];
  function place(el,txt,col,p,show){if(!el)return;
    el.textContent=txt;el.setAttribute('fill',col);
    el.setAttribute('x',px(p[0]).toFixed(1));el.setAttribute('y',py(p[1]).toFixed(1));
    el.setAttribute('opacity',show?'1':'0');}
  function labels(a,c,d){
    var best=null,worst=null;
    corners.forEach(function(p){var z=a*p[0]+c*p[1]+d;
      if(!best||z>best.z)best={p:p,z:z};
      if(!worst||z<worst.z)worst={p:p,z:z};});
    place(lt,'yes ✓','#276b45',best.p,best.z>=0);
    place(lb,'no ✗','#C93B3B',worst.p,worst.z<0);
  }
  function paint(){
    var a=+w1.value,c=+w2.value,d=+b.value;
    w1v.textContent=a.toFixed(1);w2v.textContent=c.toFixed(1);bv.textContent=d.toFixed(1);
    var yes=0;
    for(var i=0;i<pts.length;i++){var z=a*pts[i][0]+c*pts[i][1]+d;var el=document.getElementById('db-p'+i);
      if(el)el.setAttribute('fill',z>=0?'#2D8B55':'#C93B3B');if(z>=0)yes++;}
    labels(a,c,d);
    var e=ends(a,c,d);
    if(e.length>=2){line.setAttribute('x1',px(e[0][0]).toFixed(1));line.setAttribute('y1',py(e[0][1]).toFixed(1));
      line.setAttribute('x2',px(e[1][0]).toFixed(1));line.setAttribute('y2',py(e[1][1]).toFixed(1));line.setAttribute('opacity','1');}
    else line.setAttribute('opacity','0');
    out.innerHTML='boundary: <b>'+a.toFixed(1)+'·x₁ '+sgn(c)+'·x₂ '+sgn(d)+' = 0</b> — '+yes+' of 8 dots land on the “yes” side.';
  }
  [w1,w2,b].forEach(function(el){el.addEventListener('input',paint);});paint();
})();</script>
%%%

%%% insight
Did you try to bend it? Everyone does. Tilt, slide, flip which side is "yes" — but no combination of those sliders will ever give you a curve or an L-shape. That stubborn straightness is not a limitation of the sliders; it *is* the neuron. And in about two minutes it's going to cost us something real.
%%%

**You just saw, with your own hands, the single most important fact about one neuron: it draws exactly one straight line.** Hold that thought — it's about to become the neuron's one true limit.

@@@ concept id=c10 tag="The limit" title="One line can't do XOR" gotit="Got the limit"
Now the famous puzzle that stumped this whole field for years — and it starts as a party trick you can picture in your head.

Picture a **room with four chairs**, one in each corner, seen from above like a map. Two red chairs sit **top-left and bottom-right**; two blue chairs sit **top-right and bottom-left** — so each colour sits on its own **diagonal**. I hand you one straight **rope** pulled tight across the floor: put both reds on one side and both blues on the other.

Go on, try to picture it. You can't. Slide the rope, angle it any way you like — one straight rope always leaves one chair on the wrong side.

That room is the [[XOR||"On when exactly one of two inputs is on." Its two answer-groups sit on opposite diagonals, so no single straight line can separate them.]] pattern: "yes when exactly one of two inputs is on." And a single neuron, you just saw, draws exactly **one straight line** — your one straight rope. XOR is not [[linearly separable||A pattern is linearly separable if one straight line can put every "yes" on one side and every "no" on the other. XOR is the classic pattern that is NOT — no single line works.]].

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="A room floor seen from above with four chairs at the corners. Two red chairs sit on one diagonal (top-left and bottom-right); two blue chairs sit on the other (top-right and bottom-left). One straight rope crosses the floor so that only the bottom-right red chair is below it, while the top-left red chair, the top-right blue chair and the bottom-left blue chair are all above it. The top-left red chair is therefore stranded on the blues' side, so this rope gets three of the four chairs right, and no angle does better."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One straight rope can't split two diagonal pairs</text><rect x="150" y="30" width="220" height="138" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><line x1="160" y1="166" x2="365" y2="71" stroke="#9A5A12" stroke-width="3"/><text x="376" y="66" fill="#9A5A12" font-size="10">straight rope</text><rect x="172" y="45" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="185" y="64" text-anchor="middle" fill="#C93B3B">🔴</text><circle cx="185" cy="58" r="21" fill="none" stroke="#C99A12" stroke-width="2" stroke-dasharray="3,3"/><rect x="322" y="45" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="335" y="64" text-anchor="middle" fill="#5E5191">🔵</text><rect x="172" y="119" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="185" y="138" text-anchor="middle" fill="#5E5191">🔵</text><rect x="322" y="119" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="335" y="138" text-anchor="middle" fill="#C93B3B">🔴</text><text x="70" y="88" text-anchor="middle" fill="#6B645E" font-size="10">reds: top-left +</text><text x="70" y="102" text-anchor="middle" fill="#6B645E" font-size="10">bottom-right;</text><text x="70" y="116" text-anchor="middle" fill="#6B645E" font-size="10">blues the other</text><text x="440" y="120" text-anchor="middle" fill="#6B645E" font-size="10">above the rope:</text><text x="440" y="134" text-anchor="middle" fill="#5E5191" font-size="10">2 blue + 1 red</text><text x="440" y="152" text-anchor="middle" fill="#6B645E" font-size="10">below the rope:</text><text x="440" y="166" text-anchor="middle" fill="#C93B3B" font-size="10">1 red</text><text x="260" y="186" text-anchor="middle" fill="#C99A12" font-size="10">the ringed red chair is stranded on the blues' side</text><text x="260" y="200" text-anchor="middle" fill="#C93B3B" font-size="10">any angle → best you get is 3 of 4</text></g></svg>
%%%

**The chairs get it right:** the diagonal layout *is* the trap — two groups arranged so no single straight cut can separate them. **One catch:** with a real rope you're stuck for good, but a network isn't limited to one rope (that's the next unit's rescue).

If you're mid-shrug going "wait, surely *some* angle works" — good. That's the exact feeling everybody has here, including the researchers who argued about it in print. Want to wrestle it yourself first? Here's a ladder of nudges.

%%% hint
t1: Don't worry about the math yet — just look at the four chairs. Same-colored chairs sit on opposite corners (a diagonal), not next to each other. Ask yourself: could one straight rope ever keep two things that sit on opposite corners on the same side?
t2: Try it slowly. Put the rope so it fences off the top-left chair alone — good, that one's handled. Now its same-color partner is way down at the bottom-right, on the far side of your rope. To grab it too you'd have to swing the rope over there — but then you let one of the OTHER color slip onto your side. You keep trading one mistake for another, which is why the ceiling is 3 of 4.
t3: One neuron draws exactly one straight line, and XOR's two groups sit on crossing diagonals — so no single straight line can ever put all the "yes"s on one side and all the "no"s on the other. That's the whole limit.
%%%

#### Try to beat it yourself — you can't
Drag the slider to slide the one straight rope across the floor. The chair that escapes gets a ring around it. (Curious about the fix? Tick **add a hidden layer** to drop in a second rope — that's the next unit, previewed.)

%%% svg
<svg id="xor-svg" viewBox="0 0 520 236" role="img" aria-label="Interactive XOR. Slide a single straight rope across four corner chairs and watch it catch at most three of four, with a ring around the one that escapes. Tick the hidden-layer box to add a second rope that forms a band around the two red chairs and catches all four."><g font-family="monospace" font-size="11"><rect x="150" y="30" width="220" height="180" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line id="xor-l2" x1="0" y1="0" x2="0" y2="0" stroke="#2D8B55" stroke-width="3" stroke-dasharray="6,4" opacity="0"/><line id="xor-l1" x1="190" y1="106" x2="274" y2="190" stroke="#9A5A12" stroke-width="3"/><circle id="xor-miss" cx="330" cy="50" r="15" fill="none" stroke="#C99A12" stroke-width="2.5" stroke-dasharray="3,3" opacity="1"/><circle cx="190" cy="190" r="9" fill="#5E5191"/><circle cx="190" cy="50" r="9" fill="#C93B3B"/><circle cx="330" cy="190" r="9" fill="#C93B3B"/><circle cx="330" cy="50" r="9" fill="#5E5191"/><text x="150" y="224" fill="#5E5191" font-size="9">🔵 off = both/neither on</text><text x="300" y="224" fill="#C93B3B" font-size="9">🔴 on = exactly one</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
<label>slide the rope <input id="xor-c" type="range" min="0.3" max="1.7" step="0.05" value="0.6" style="accent-color:#9A5A12;vertical-align:middle" aria-label="rope position"></label>
<label style="color:#276b45"><input id="xor-hl" type="checkbox" style="vertical-align:middle"> add a hidden layer (a 2nd rope)</label>
</div>
<div id="xor-out" style="margin-top:6px;color:#2C2A28">one straight rope catches <b>3 of 4</b> — one chair always escapes (ringed).</div>
</div>
<script>(function(){
  var cs=document.getElementById('xor-c');if(!cs)return;
  var hl=document.getElementById('xor-hl'),l1=document.getElementById('xor-l1'),l2=document.getElementById('xor-l2'),
      miss=document.getElementById('xor-miss'),out=document.getElementById('xor-out');
  function px(x){return 190+x*140;}
  function py(y){return 190-y*140;}
  // edge hits, de-duplicated (a rope through a corner is found twice; drawing e[0]->e[1]
  // would then be a zero-length, invisible line)
  function ends(c){var e=[];
    function add(x1,x2){
      for(var i=0;i<e.length;i++){if(Math.abs(e[i][0]-x1)<1e-6&&Math.abs(e[i][1]-x2)<1e-6)return;}
      e.push([x1,x2]);}
    [0,1].forEach(function(x1){var x2=c-x1;if(x2>=0-1e-9&&x2<=1+1e-9)add(x1,x2);});
    [0,1].forEach(function(x2){var x1=c-x2;if(x1>=0-1e-9&&x1<=1+1e-9)add(x1,x2);});return e;}
  function setLine(el,c){var e=ends(c);if(e.length>=2){el.setAttribute('x1',px(e[0][0]).toFixed(1));el.setAttribute('y1',py(e[0][1]).toFixed(1));
    el.setAttribute('x2',px(e[1][0]).toFixed(1));el.setAttribute('y2',py(e[1][1]).toFixed(1));el.setAttribute('opacity','1');}}
  var P=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]; // x1,x2,trueClass
  function paint(){
    if(hl.checked){
      setLine(l1,0.5);setLine(l2,1.5);miss.setAttribute('opacity','0');cs.disabled=true;
      out.innerHTML='two ropes make a <b>band</b> around the two red chairs — inside = “on”, outside = “off”. All <b>4 of 4</b> split ✓ — that’s a hidden layer.';
      return;
    }
    cs.disabled=false;l2.setAttribute('opacity','0');
    var c=+cs.value;setLine(l1,c);
    var A=[],B=[];P.forEach(function(p){(p[0]+p[1]>=c?A:B).push(p);});
    function maj(arr){var o=arr.filter(function(p){return p[2]===1;}).length;return o*2>=arr.length?1:0;}
    var la=maj(A),lb=maj(B),wrong=[];
    A.forEach(function(p){if(p[2]!==la)wrong.push(p);});B.forEach(function(p){if(p[2]!==lb)wrong.push(p);});
    if(wrong.length){miss.setAttribute('cx',px(wrong[0][0]));miss.setAttribute('cy',py(wrong[0][1]));miss.setAttribute('opacity','1');}
    else miss.setAttribute('opacity','0');
    out.innerHTML='one straight rope catches <b>'+(4-wrong.length)+' of 4</b> — one chair always escapes (ringed). Slide it anywhere: never 4.';
  }
  cs.addEventListener('input',paint);hl.addEventListener('change',paint);paint();
})();</script>
%%%

%%% insight
This is not you being bad at sliding a rope. It's a *proof*, and it once cost the field years of funding: one neuron draws one straight line, and some patterns can't be split by any straight line. So whenever you hear "neural nets can solve any problem" — you now have the four-chair counterexample. Knowing exactly where a tool stops is what separates someone who *uses* AI from someone who *understands* it, and you just did that on Day 1.
%%%

**You've found the exact edge of a single neuron's power.** The fix is waiting in the very next unit, and it's almost embarrassingly simple.

@@@ concept id=c11 tag="Layers & networks" title="Join neurons into a network and the line can bend" gotit="Got layers"
So one rope can't split the four chairs. The fix is almost silly once you see it: **use more than one rope.**

Give yourself a *second* straight rope and you can fence off a strip of floor — one rope on each side of a diagonal band. Together they wrap two chairs inside the strip while the other two sit outside. Two simple straight pieces, combining into something bent.

That's exactly what happens when you put several neurons side by side. A [[layer||A row of neurons working side by side, all reading the SAME inputs. Each neuron draws its own straight line; together the layer draws several lines at once.]] is a **row of neurons** that all read the same inputs at the same time — a panel of judges, each scoring the same act with their own opinion.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A layer drawn as a row of two neurons side by side. The same two inputs on the left fan out to both neurons. Each neuron draws its own straight line, so the layer produces two lines from the same inputs. This is like a panel of judges each scoring the same act."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A layer = a row of neurons, all reading the SAME inputs</text><circle cx="55" cy="70" r="9" fill="#EFEAF7" stroke="#7C6DAA"/><text x="55" y="74" text-anchor="middle" fill="#5E5191" font-size="9">x₁</text><circle cx="55" cy="120" r="9" fill="#EFEAF7" stroke="#7C6DAA"/><text x="55" y="124" text-anchor="middle" fill="#5E5191" font-size="9">x₂</text><line x1="64" y1="70" x2="180" y2="65" stroke="#B8AEA2"/><line x1="64" y1="120" x2="180" y2="75" stroke="#B8AEA2"/><line x1="64" y1="70" x2="180" y2="130" stroke="#B8AEA2"/><line x1="64" y1="120" x2="180" y2="140" stroke="#B8AEA2"/><rect x="180" y="52" width="120" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="240" y="70" text-anchor="middle" fill="#276b45" font-size="10">neuron A</text><text x="240" y="86" text-anchor="middle" fill="#6B645E" font-size="9">draws line 1</text><rect x="180" y="112" width="120" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="240" y="130" text-anchor="middle" fill="#276b45" font-size="10">neuron B</text><text x="240" y="146" text-anchor="middle" fill="#6B645E" font-size="9">draws line 2</text><text x="120" y="44" text-anchor="middle" fill="#6B645E" font-size="10">same inputs</text><text x="120" y="170" text-anchor="middle" fill="#9A938A" font-size="9">fan out to every neuron</text><path d="M312 74 Q 340 90 312 134" fill="none" stroke="#B8AEA2" stroke-width="1.5"/><text x="380" y="98" text-anchor="middle" fill="#276b45" font-size="10">2 neurons</text><text x="380" y="114" text-anchor="middle" fill="#276b45" font-size="10">= 2 lines at once</text></g></svg>
%%%

**The panel of judges gets it right:** every neuron in a layer sees the same inputs and forms its own opinion (its own line). **One catch:** real judges influence each other; neurons in a layer never talk — they work in parallel, and only their outputs get combined, by the *next* layer.

#### From one rope to a bendy boundary
Here is the whole rescue, one rung at a time. Keep the ropes in mind — each rung is one more rope, or one more knot.

%%% steps
step: one neuron → one rope
why: exactly what you played with; a single straight cut, therefore stuck at 3 of 4
step: a layer of two neurons → two ropes at once
why: both read the same inputs but hold different dials, which means two independent straight cuts across the same floor
step: feed those outputs into a *second* layer → the knot
why: the next neuron reads "which side of rope 1?" and "which side of rope 2?" — so it can demand *inside both*, and that combination is a band, not a line
step: the band catches all four chairs → 4 of 4
why: the straightness never went away; it got *combined*, therefore the boundary bends without any single neuron bending
%%%

That stack of layers has a name: a [[neural network||Several layers of neurons connected in a chain: each layer's outputs become the next layer's inputs. The network as a whole can bend and fold boundaries a single neuron never could.]] — neurons wired in a chain, each layer feeding the next.

%%% insight
Sit with how strange and lovely this is. Nobody taught any neuron to draw a curve. Every single one is still doing the same dumb straight cut you built in ten minutes. The bending is an *emergent* property of stacking — the limit didn't get fixed, it got **outgrown by teamwork**. That is genuinely the central trick of all deep learning, and you just watched it happen with two ropes.
%%%

#### The three kinds of layer, by where they sit
Layers get names from their position in the chain. Three names, one line each.

- The [[input layer||The raw numbers you feed in — the very first values that enter the network. Not neurons that compute, just the inputs handed to the first real layer.]] — the raw inputs you hand in (`x₁`, `x₂`, …). No computing; just the numbers at the very front.
- A [[hidden layer||A layer of neurons in the MIDDLE of a network — between the inputs and the final answer. It's "hidden" because you never read its outputs directly; they only feed the next layer. Hidden layers are what let the boundary bend.]] — any layer of neurons in the *middle*. "Hidden" because you never read its numbers; they exist only to feed the next layer. This is the layer that split the four chairs.
- The [[output layer||The LAST layer, whose result is the network's final answer — the number (or numbers) you actually read and use.]] — the very last layer, whose result is the answer you actually use.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A small neural network in three named columns. On the left, the input layer is two raw input values. In the middle, a hidden layer of two neurons, each connected to both inputs. On the right, an output layer of one neuron producing the final answer. Arrows flow left to right from inputs through the hidden layer to the output."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A network: input layer → hidden layer → output layer</text><circle cx="70" cy="75" r="12" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="70" y="79" text-anchor="middle" fill="#6B645E" font-size="9">x₁</text><circle cx="70" cy="135" r="12" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="70" y="139" text-anchor="middle" fill="#6B645E" font-size="9">x₂</text><text x="70" y="172" text-anchor="middle" fill="#6B645E" font-size="10">input layer</text><text x="70" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(raw inputs)</text><line x1="82" y1="75" x2="248" y2="75" stroke="#B8AEA2"/><line x1="82" y1="75" x2="248" y2="135" stroke="#B8AEA2"/><line x1="82" y1="135" x2="248" y2="75" stroke="#B8AEA2"/><line x1="82" y1="135" x2="248" y2="135" stroke="#B8AEA2"/><circle cx="260" cy="75" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><circle cx="260" cy="135" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="260" y="172" text-anchor="middle" fill="#276b45" font-size="10">hidden layer</text><text x="260" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(the bend lives here)</text><line x1="274" y1="75" x2="426" y2="105" stroke="#B8AEA2"/><line x1="274" y1="135" x2="426" y2="105" stroke="#B8AEA2"/><circle cx="440" cy="105" r="14" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="440" y="172" text-anchor="middle" fill="#8A6D3B" font-size="10">output layer</text><text x="440" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(the answer)</text><line x1="454" y1="105" x2="500" y2="105" stroke="#C99A12" stroke-width="2"/><polygon points="500,100 512,105 500,110" fill="#C99A12"/></g></svg>
%%%

Real networks stack many hidden layers with thousands of neurons each — far too many to draw — but the shape is always this same input → hidden → output chain.

#### That's what "deep" means
Now the word you've heard a hundred times finally has a plain meaning.

Stack **many hidden layers** one after another and we call the network **deep** — training it is [[deep learning||Training a neural network that has many hidden layers stacked in a chain. "Deep" literally means "many layers deep"; each layer bends the boundary a little more.]]. That's the whole secret behind the term: "deep" just means "many layers deep."

Each extra hidden layer bends the boundary a little more, so a deep stack carves shapes wildly more intricate than any single line. ChatGPT is a very deep network — a tall stack built out of exactly these neurons, plus a few extra parts you'll meet in later modules.

!!! c-warn ⚠️
<b>Myth to dodge — "bigger is always better."</b> It's tempting to think "more layers, more neurons, always smarter." Not so: a network that's too big for its data just <i>memorizes</i> the examples instead of learning the real pattern (you'll meet this as <b>overfitting</b> later), and it costs far more to train and run. The right size fits the problem — bigger is a tool, not a magic wand.
!!!

#### See the fix: one line fails, a hidden layer bends it
Same four chairs, split by one line (fails) versus two lines from a hidden layer (works). The band wraps the two **red** chairs — the same pair the live widget wraps. Either diagonal pair can be the one inside the band; what matters is that a band is a shape no single line could ever make.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Left: one straight line tries to separate two diagonal pairs of dots and fails. It leaves the top-left red dot on the same side as both blue dots, so only three of the four are right. Right: two parallel straight lines from a hidden layer form a band along the other diagonal. Both red dots (top-left and bottom-right) fall inside the band; both blue dots (top-right and bottom-left) fall outside it. The band cleanly separates the two classes, four of four."><g font-family="monospace" font-size="11"><text x="128" y="20" text-anchor="middle" fill="#C93B3B" font-size="11">one neuron: one line (fails)</text><rect x="33" y="30" width="190" height="132" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="43" y1="152" x2="213" y2="67" stroke="#9A5A12" stroke-width="2.5"/><circle cx="68" cy="60" r="6" fill="#C93B3B"/><circle cx="188" cy="128" r="6" fill="#C93B3B"/><circle cx="188" cy="60" r="6" fill="#5E5191"/><circle cx="68" cy="128" r="6" fill="#5E5191"/><circle cx="68" cy="60" r="14" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="3,3"/><text x="128" y="180" text-anchor="middle" fill="#C93B3B" font-size="10">stuck at 3 of 4 (ringed red is wrong)</text><text x="392" y="20" text-anchor="middle" fill="#276b45" font-size="11">hidden layer: two lines bend (works)</text><rect x="297" y="30" width="190" height="132" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="325" y1="30" x2="470" y2="112" stroke="#2D8B55" stroke-width="2.5"/><line x1="315" y1="76" x2="462" y2="159" stroke="#2D8B55" stroke-width="2.5"/><circle cx="332" cy="60" r="6" fill="#C93B3B"/><circle cx="452" cy="128" r="6" fill="#C93B3B"/><circle cx="452" cy="60" r="6" fill="#5E5191"/><circle cx="332" cy="128" r="6" fill="#5E5191"/><text x="392" y="180" text-anchor="middle" fill="#276b45" font-size="10">band holds both reds → 4 of 4</text></g></svg>
%%%

Joining simple straight pieces really can fence off a shape neither piece could handle alone. With real ropes *you* tie the knot; in a network the "knot" is a tiny math step called the activation (tomorrow's whole lesson), and the network finds the arrangement by **training** (Days 5–6). Want to watch it happen? Flip on the **"add a hidden layer"** toggle back in the XOR widget above: the second rope appears, forms exactly this band, and all four chairs finally split.

<strong>You now know a single neuron <em>and</em> how neurons join into layers and networks to outgrow its limit.</strong> That door — stacking, plus the bend that makes stacking worth anything — is exactly where Day 2 begins.

@@@ concept id=c12 tag="What it powers" title="Where these little neurons grow up" gotit="Saw the payoff"
Before we wrap, let's zoom *all* the way out — because the tiny weigh → add → decide you just learned is the exact seed inside the tech you use every day.

Picture a single **LEGO brick**. On its own it's almost nothing — a bump of plastic. Snap a few thousand together and you can build a whole castle. One neuron is that brick.

%%% svg
<svg viewBox="0 0 520 206" role="img" aria-label="Six labelled tiles showing real-world uses of neural networks: image recognition (a camera), speech recognition (a person speaking), language translation (a globe), game playing (a game controller), recommendation systems (a thumbs-up), and self-driving cars (a car). In the middle sits one small neuron brick — the same brick all six are built from."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">One neuron (the brick) → the things people build with millions of them</text><rect x="30" y="30" width="140" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="50" font-size="15">📸</text><text x="100" y="68" fill="#276b45">image recognition</text><rect x="190" y="30" width="140" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="260" y="50" font-size="15">🗣️</text><text x="260" y="68" fill="#5E5191">speech recognition</text><rect x="350" y="30" width="140" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="420" y="50" font-size="15">🌐</text><text x="420" y="68" fill="#8A6D3B">language translation</text><rect x="205" y="84" width="110" height="34" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="260" y="99" font-size="13">🧱</text><text x="260" y="113" fill="#6B645E">one neuron</text><rect x="30" y="126" width="140" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="100" y="146" font-size="15">🎮</text><text x="100" y="164" fill="#8A6D3B">game playing</text><rect x="190" y="126" width="140" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="260" y="146" font-size="15">👍</text><text x="260" y="164" fill="#5E5191">recommendations</text><rect x="350" y="126" width="140" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="420" y="146" font-size="15">🚗</text><text x="420" y="164" fill="#276b45">self-driving cars</text><text x="260" y="192" fill="#9A938A" font-size="9">same brick, different castle</text></g></svg>
%%%

**The LEGO brick gets it right:** one simple unit, repeated and connected in huge numbers, builds wildly different things. **One catch:** bricks are snapped together *by hand* into a fixed shape, while a network tunes itself to the job by **training** (Day 6, the training loop).

#### Six castles built from your brick
One line each. Every one of these is stacks of the neuron you now understand.

- **📸 Image recognition** — your phone unlocking at your face: early layers spot edges, later layers combine them into eyes, then whole faces.
- **🗣️ Speech recognition** — your voice assistant turning sound into words: layers pick out sound-pieces, then syllables, then words.
- **🌐 Language translation** — an app turning English into French: the network reads one sentence's meaning and writes it out in another language.
- **🎮 Game playing** — the AI that beat the world's best Go players: it looks at the board and scores which move is most likely to win.
- **👍 Recommendation systems** — the "you might also like" on a streaming app: it weighs what you've watched to guess what you'll enjoy next.
- **🚗 Self-driving cars** — a car reading the road: networks find lanes, cars and pedestrians in the camera feed and decide when to brake.

%%% insight
Here's the part that surprises everyone: not one of those six is a *different kind* of machine. Same brick, same weigh → add → decide, different stack and different training data. When someone tells you "AI for medicine" and "AI for translation" are worlds apart — they're mostly the same bricks in a different castle. You are now in on that.
%%%

#### How does it *get good* at any of this?
Every castle starts out **useless**. A fresh network's weights are random numbers, so its first guesses are basically coin-flips. Think of learning to bake: taste a too-sweet cake, use less sugar next time, taste again — repeat until it's right. A network improves in exactly that loop.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A four-step learning loop drawn as a cycle. Step one: guess with the current weights. Step two: measure the error, how wrong the guess was. Step three: adjust the weights a little to reduce the error. Step four: repeat. An arrow loops from step four back to step one."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">How a network learns: guess → check error → adjust → repeat</text><rect x="30" y="55" width="90" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="75" y="74" fill="#5E5191">1 · guess</text><text x="75" y="90" fill="#9A938A" font-size="9">(current weights)</text><line x1="120" y1="77" x2="160" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="160,72 170,77 160,82" fill="#B8AEA2"/><rect x="172" y="55" width="96" height="44" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="220" y="74" fill="#C93B3B">2 · check error</text><text x="220" y="90" fill="#9A938A" font-size="9">how wrong?</text><line x1="268" y1="77" x2="308" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="308,72 318,77 308,82" fill="#B8AEA2"/><rect x="320" y="55" width="96" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="368" y="74" fill="#276b45">3 · adjust</text><text x="368" y="90" fill="#9A938A" font-size="9">nudge weights</text><line x1="416" y1="77" x2="452" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="452,72 462,77 452,82" fill="#B8AEA2"/><rect x="464" y="55" width="46" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="487" y="80" fill="#8A6D3B">4 · repeat</text><path d="M487 99 Q 487 130 260 130 Q 75 130 75 101" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="4,3"/><polygon points="70,105 75,95 80,105" fill="#C99A12"/><text x="260" y="145" fill="#9A938A" font-size="9">do it thousands of times and the guesses get good — that's training</text></g></svg>
%%%

%%% steps
step: guess
why: run a forward pass with whatever weights it has now — exactly the conveyor you rode, mistakes and all
step: check the error
why: compare the guess to the right answer, therefore the network gets a number for "how wrong was I?"
step: adjust
why: nudge each weight and the bias a little in the direction that shrinks that error — the taste-less-sugar move
step: repeat, thousands of times
why: each pass is a tiny correction, which means the guesses creep from coin-flip to genuinely good
%%%

In one sentence: **training is the search for the weights and biases that make the network wrong as little as possible.**

The machinery behind each rung has its own day ahead:
- *How wrong was the guess?* → a **loss function**, **Day 4**.
- *Which way to nudge each weight?* → **gradient descent** and **backpropagation**, **Day 5** and beyond.
- *The loop that runs it all thousands of times?* → **the training loop**, **Day 6**.

Today you built the thing that does the guessing. **Next you'll sharpen its "decide" step, then teach it to teach itself.** That's the arc of this whole module — and you're standing right at the start of it.

@@@ concept id=c13 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — that was a real journey.

Here's a nice way to hold it: picture yourself at the end of a hike, laying out the **postcards you collected at each stop**. At every viewpoint you picked up one small picture, and now, spread on the table, they tell the whole climb in order. (Postcards are souvenirs you put away; these are working parts you'll pick back up every day from here.)

The postcards from today's trail, in order:

- **Stop 1 · The brain cell.** A neuron is a tiny copy of a brain cell: **weigh → add → decide**, and that's the whole machine.
- **Stop 2 · Inputs & weights.** The raw numbers you feed in are the **inputs**. Each gets one **weight** — a trust dial: big = "listen," near-zero = "ignore," negative = "count against."
- **Stop 3 · Bias.** One extra number, the neuron's starting mood — a head start toward "yes," or a handicap.
- **Stop 4 · The weighted sum z.** Weigh every input, add them up, add the bias → one number `z` (the pre-activation).
- **Stop 5 · The activation (decide).** A step turns `z` into the answer. You met the hard **step** (a cliff — no direction to nudge in, so a *stack* of them can't be improved) and the smooth **sigmoid** (a dimmer, 0→1, so it can). Always *linear part, then activation.*
- **Stop 6 · The line, and its limit.** Weights + bias draw **one straight line**. That's the ceiling: one line can't do **XOR**.
- **Stop 7 · Layers, networks, depth.** A **layer** is a row of neurons on the same inputs; chain layers into a **network** (**input layer** → **hidden layer** → **output layer**) and the lines combine and bend. Many hidden layers = **deep** (**deep learning**).
- **Stop 8 · The payoff.** Millions of these bricks power image and speech recognition, translation, game playing, recommendations and self-driving cars — and they get good by **guess → check error → adjust → repeat** (training, Days 4–6).

Here's the one picture to keep — the whole neuron on one line, and where it grows into a network.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="The complete neuron on one line: two inputs each times a weight, summed with a bias into z, then an activation turns z into the output. Below, a note that weights and bias draw one straight line, which cannot do XOR alone, and that stacking neurons into a hidden layer bends the boundary."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The whole neuron: weigh → add → decide</text><circle cx="40" cy="55" r="7" fill="#EFEAF7" stroke="#7C6DAA"/><text x="40" y="59" text-anchor="middle" fill="#5E5191" font-size="9">x₁</text><circle cx="40" cy="100" r="7" fill="#EFEAF7" stroke="#7C6DAA"/><text x="40" y="104" text-anchor="middle" fill="#5E5191" font-size="9">x₂</text><line x1="47" y1="57" x2="150" y2="72" stroke="#B8AEA2" stroke-width="1.5"/><text x="95" y="58" fill="#6B645E" font-size="9">×w₁</text><line x1="47" y1="98" x2="150" y2="80" stroke="#B8AEA2" stroke-width="1.5"/><text x="95" y="105" fill="#6B645E" font-size="9">×w₂</text><rect x="150" y="58" width="80" height="36" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="190" y="80" text-anchor="middle" fill="#5E5191" font-size="10">Σ + b → z</text><line x1="230" y1="76" x2="290" y2="76" stroke="#B8AEA2" stroke-width="2"/><polygon points="290,70 302,76 290,82" fill="#B8AEA2"/><rect x="305" y="58" width="90" height="36" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="350" y="80" text-anchor="middle" fill="#276b45" font-size="10">activation</text><line x1="395" y1="76" x2="450" y2="76" stroke="#B8AEA2" stroke-width="2"/><polygon points="450,70 462,76 450,82" fill="#B8AEA2"/><rect x="465" y="58" width="45" height="36" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="487" y="80" text-anchor="middle" fill="#8A6D3B" font-size="10">out</text><text x="260" y="132" text-anchor="middle" fill="#6B645E" font-size="10">weights + bias draw ONE straight line…</text><text x="260" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">…can't do XOR alone → stack into a hidden layer to bend it</text><text x="260" y="168" text-anchor="middle" fill="#9A938A" font-size="9">many hidden layers = "deep" (deep learning)</text></g></svg>
%%%

#### Three myths to leave behind
- **"Neural nets think like a human brain."** No — a real brain cell is a living thing; an artificial neuron is only multiply-and-add. *Inspired by*, not the same as.
- **"They're magic black boxes."** No — every answer is *assembled* from plain steps (weigh, add, decide), exactly the ones you ran today. What they're genuinely great at is **finding patterns** in data.
- **"Bigger is always better" / "they can solve any problem."** No — one neuron provably can't even do XOR, and a network too big for its data just memorizes. The right *size* and *shape* beats raw size.

#### Cheat-sheet · the neuron's parts at a glance
One table, worth bookmarking. Every word you met today is tucked into the optional list underneath it.

%%% table
:: Part :: Everyday picture :: What it does
Input :: a raw number you hand in :: the value the neuron reads
Weight :: a trust dial (one per input) :: scales how much an input matters (negative = counts against)
Bias :: a starting mood / head start :: shifts where the neuron leans before any input
Weighted sum z :: a shopping receipt total :: every input × its weight, added, plus bias (the pre-activation)
Activation :: a judge's verdict :: turns z into the answer — step (hard 0/1) or sigmoid (smooth 0→1)
Decision boundary :: one chalk line on a playground :: the straight cut weights + bias draw
The limit :: one rope, four chairs :: one line can't do XOR (not linearly separable)
Layer & network :: a panel of judges, chained :: input → hidden → output; many hidden layers = deep
%%%

~~~html
<details style="margin:14px 0;border:1px solid #E5DFD6;border-radius:8px;padding:10px 14px;background:#FDF9F3">
<summary style="cursor:pointer;font-weight:600;color:#5A544E">Optional (skippable) · every word you met today, in one list</summary>
<div style="font-family:monospace;font-size:.86em;line-height:1.95;color:#3A342E;margin-top:8px">
<b>neuron</b> — a tiny math function that weighs its inputs, adds them up, and decides<br>
<b>input</b> — a raw number handed to the neuron; the input layer is all the raw inputs together<br>
<b>weight</b> — one number per input: how much that input matters (negative = counts against)<br>
<b>bias</b> — one extra number added to the sum: the neuron's starting lean toward "yes"<br>
<b>weighted sum (z)</b> — every input times its weight, summed, plus the bias; also called the pre-activation<br>
<b>activation function</b> — the last step that turns z into the neuron's output<br>
<b>step function</b> — the first activation: fires 1 when z reaches zero or higher, else 0<br>
<b>sigmoid</b> — a smooth S-curve that squashes z into (0, 1) — the "how confident" answer<br>
<b>decision boundary</b> — the one straight line a neuron draws to split its inputs into two sides<br>
<b>forward propagation</b> — pushing inputs left-to-right through the neuron to get the output<br>
<b>XOR</b> — "on when exactly one input is on" — the pattern one straight line can't split<br>
<b>linearly separable</b> — when one straight line can cleanly split every class; XOR is not<br>
<b>layer</b> — a row of neurons all reading the same inputs<br>
<b>hidden layer</b> — a middle layer, between inputs and answer — where the boundary bends<br>
<b>output layer</b> — the last layer, whose result is the network's final answer<br>
<b>neural network</b> — layers of neurons chained so each layer feeds the next<br>
<b>deep learning</b> — training a network with many hidden layers stacked ("deep" = many layers)<br>
<b>training</b> — slowly adjusting the weights and bias so the network makes fewer mistakes (Days 5–6)
</div>
</details>
~~~

!!! c-info 📚
<b>Want more before Day 2? (all optional further reading)</b> Nothing here is required. If today lit a spark, the friendliest next read is the free online book <b>"Neural Networks and Deep Learning"</b> by Michael Nielsen (chapter 1 covers this exact neuron), or <b>3Blue1Brown</b>'s short video series <b>"Neural Networks"</b> on YouTube for a beautiful visual tour. Skip them with a clear conscience — Day 2 assumes only what you learned right here.
!!!

That's the whole day. Next you'll zoom in on that "decide" step — the **activation** — and see the little bend that lets neurons stack into something genuinely deep.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a nudge to scroll back, not a failing grade.)
%%% quiz
q: What are the three beats a single neuron always does, in order? | a:2 | add → weigh → decide | decide → add → weigh | weigh → add → decide | weigh → decide → add | fb: A neuron scales each input by its weight (weigh), sums them with the bias (add), then turns that total into an answer (decide).
q: A neuron has two inputs. How many weights and how many biases does it have? | a:1 | 1 weight and 2 biases | 2 weights and 1 bias | 2 weights and 2 biases | 1 weight and 1 bias | fb: One weight per input (so 2), and exactly one bias for the whole neuron — the bias is a single starting-lean number, not one per input. Training adjusts all three of those numbers.
q: Why did people move from the hard step function to a smooth activation like sigmoid? | a:2 | The step used too much memory | The step outputs negative numbers | The step is a cliff, so a tiny nudge to the weights changes the output by nothing (or flips it entirely) — a smooth curve lets a small nudge make a small, followable change, which is what you need to improve a whole stack of neurons | Sigmoid is faster to compute | fb: A cliff hands back no direction to nudge in. That is survivable for one neuron (the Perceptron's mistake rule managed it), but a stack of neurons needs the smooth version — you will see exactly why on Day 5.
q: One neuron can't get all four XOR points right. What is the fix, and what do we call a network with many such stacked layers? | a:1 | Raise the learning rate | Stack neurons into a hidden layer so their lines combine into a bent boundary; a network with many hidden layers is called "deep" (deep learning) | Give the neuron more weights | Use a negative bias | fb: One neuron = one straight line, and XOR is not linearly separable. A hidden layer draws several lines that combine into a band to split all four. Stacking many hidden layers is what "deep" learning means.
%%%

@@@ produce id=produce tag="Produce" title="Build a neuron and watch it decide" gotit="Done"
Time to see today with your own eyes. You'll code one neuron — weights, bias, weighted sum, activation — run a full forward pass, and then watch it *fail* on XOR so the limit is real, not just a story.

**Predict first:** for inputs `x = [2, 3]`, weights `w = [0.5, −1.0]`, and bias `b = 1.0`, what's `z`? And will `sigmoid(z)` land above or below `0.5`? Then run it and **watch.** Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-01-single-neuron/experiment.py`. Write `sigmoid(z)` and `step(z)`, then a `neuron(x, w, b)` that returns `sigmoid((w*x).sum() + b)`.

**Notice** that for `x=[2,3]`, `w=[0.5,−1.0]`, `b=1.0` you get `z = −1.0` and `sigmoid(z) ≈ 0.269` — a soft "probably no." Print `z` and the output so you can see both.

Then set up the four XOR points `[0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0` and try to find a single `w, b` that a step-neuron gets all four right — **observe** that you can never beat 3 of 4, because one neuron draws one straight line. Run with `python3 sessions/m02-the-neuron/day-01-single-neuron/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 1 artifact.

Create sessions/m02-the-neuron/day-01-single-neuron/experiment.py that, with a comment on each step:
1. Defines sigmoid(z)=1/(1+np.exp(-z)) and step(z)=(z>=0).astype(float).
2. Defines neuron(x, w, b) that computes z=(w*x).sum()+b and returns (z, sigmoid(z)). For x=[2,3], w=[0.5,-1.0], b=1.0 it should print z=-1.0 and sigmoid(z)≈0.269 (a soft "probably no").
3. Shows the single-neuron limit on XOR: the four points [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0. Loop over a grid of candidate (w1, w2, b), classify each XOR point with a step-neuron (fire 1 if z>=0), and print the BEST accuracy any single line achieves — it should top out at 3 of 4 (0.75), never 4 of 4, because one neuron draws one straight line and XOR is not linearly separable.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (a neuron deciding, then hitting its limit)
- For `x=[2,3]`, `w=[0.5,−1.0]`, `b=1.0` you print `z = −1.0` and `sigmoid(z) ≈ 0.269` — a full forward pass, ending in a soft "about 27% yes."
- Flipping the bias more positive pushes the output above `0.5`; more negative pushes it down — the bias really is the starting lean.
- The moment to **watch**: no single `(w, b)` gets all four XOR points right — the best any straight line manages is **3 of 4**. That ceiling is the whole point: one neuron = one line, and stacking neurons into a hidden layer is the way out.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-01-single-neuron/log.md`: (1) the three beats a neuron does and what a weight vs. a bias each controls; (2) why we prefer a smooth sigmoid over the hard step; (3) why one neuron can't do XOR, and the one-line fix (a hidden layer).
!!!

@@@ fin
