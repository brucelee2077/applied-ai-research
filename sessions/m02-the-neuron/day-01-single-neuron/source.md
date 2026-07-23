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
@lede Picture asking three friends whether to see a movie tonight. One friend has amazing taste, so you *lean on* what they say. One is hit-or-miss, so you only half-listen. One always hates everything, so you barely count their vote. You quietly weigh each friend, add it all up in your head, and land on a decision: yes or no. That little "weigh → add → decide" is *exactly* what one **neuron** does — and here's the wild part: connect a few million of these tiny vote-counters into a network and you get the thing behind ChatGPT finishing your sentence, your phone recognizing your face, or an app translating a language. Today you meet that single piece, all on its own. It's simpler than you fear, and by the end you'll honestly be able to say, "I know what the smallest part of an AI actually does."
@goal Together we'll build one neuron from nothing: its inputs, its **weights** (how much it trusts each input), its **bias** (its starting mood), the little sum it adds up, and the "decide" step at the end. Then you'll see how neurons join into **layers** and **networks** — the very thing the word *deep* in "deep learning" points at. You'll drag a real neuron's dials and watch it draw a line, catch the one pattern it *can't* learn, and meet the neat fix. Every new word gets a plain-English meaning the moment it shows up — no symbol left unexplained.

%%% warmup
q: This is Day 1 — no neural-network knowledge needed yet. Quick readiness check: what does "multiply" do to two numbers like 0.5 and 2? | a:2 | it adds them to get 2.5 | it picks the bigger one | it scales one by the other to get 1.0 | it makes them both zero | concept: prereq-multiply | fb: Multiplying scales one number by the other: 0.5 × 2 = 1.0. Today a neuron does exactly this to each input.
q: A neuron will hand you a plain list of numbers to work with, like [2, 3]. What is that — a single number, or a few numbers grouped together? | a:1 | a single number | a few numbers grouped in a list | a word | a picture | concept: prereq-list-of-numbers | fb: It's a small list (a group) of numbers. You met data as lists of numbers in M1 — today those become a neuron's inputs.
%%%

@@@ concept id=c1 tag="What it is" title="A neuron: weigh → add → decide" gotit="Got the idea"
Let's start where the name comes from. Deep inside your head sit tiny cells called **brain cells** — the wiring that lets you read this sentence. Each brain cell has little arms that catch signals from other cells, and when it hears *enough* of them, it "fires" and passes a signal along. Some incoming arms it trusts a lot; some it barely listens to. It gathers all those signals, sizes them up, and makes one small yes-or-no call.

An artificial [[neuron||A tiny math function: it takes a few input numbers, scales each by how much it matters, adds them up, and produces one output number.]] copies that idea with plain arithmetic. It takes a few numbers coming in — those raw numbers you hand it are its **inputs** — decides how much each one matters, adds them together, and produces one number out. Three beats, always in this order: **weigh → add → decide**. That is the whole machine. Really.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A brain cell drawn like a blob with three input arms on the left catching signals, and one output arm on the right. Beneath it, the same idea as three labelled boxes: weigh the inputs, add them up, then decide."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A brain cell → a neuron: catch signals, weigh them, fire</text><line x1="40" y1="55" x2="150" y2="80" stroke="#B8AEA2" stroke-width="2"/><line x1="40" y1="80" x2="150" y2="85" stroke="#B8AEA2" stroke-width="2"/><line x1="40" y1="108" x2="150" y2="92" stroke="#B8AEA2" stroke-width="2"/><circle cx="40" cy="55" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="40" cy="80" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="40" cy="108" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="22" y="59" text-anchor="end" fill="#6B645E">in</text><text x="22" y="84" text-anchor="end" fill="#6B645E">in</text><text x="22" y="112" text-anchor="end" fill="#6B645E">in</text><path d="M150 60 Q 195 40 235 70 Q 260 100 225 118 Q 175 128 152 100 Q 140 78 150 60 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="192" y="94" text-anchor="middle" fill="#276b45">neuron</text><line x1="238" y1="86" x2="320" y2="86" stroke="#2D8B55" stroke-width="2.5"/><polygon points="320,80 334,86 320,92" fill="#2D8B55"/><text x="345" y="90" fill="#276b45">one number out →</text><rect x="60" y="150" width="90" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="105" y="169" text-anchor="middle" fill="#8A6D3B">1 · weigh</text><text x="163" y="169" fill="#B8AEA2">→</text><rect x="185" y="150" width="90" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="230" y="169" text-anchor="middle" fill="#5E5191">2 · add</text><text x="288" y="169" fill="#B8AEA2">→</text><rect x="310" y="150" width="90" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="355" y="169" text-anchor="middle" fill="#276b45">3 · decide</text><text x="260" y="196" text-anchor="middle" fill="#6B645E" font-size="10">every neuron, big model or small, is just these three steps</text></g></svg>
%%%

**What the brain-cell picture gets right:** the shape is spot on — many signals come in, each is trusted differently, they get summed, and one signal goes out. **Where it breaks down:** a real brain cell is a living, electrical, wildly complicated thing; an artificial neuron is only a pinch of multiply-and-add. It borrows the *idea*, not the biology. So think "inspired by," never "the same as" — that's your first myth busted, and we'll gather a few more at the end.

#### The three beats, in a sentence each
- **Weigh** — each input gets multiplied by a number that says how much it matters (that number is a *weight*, coming up next).
- **Add** — pile all the weighed inputs into one running total, plus a little starting nudge (that nudge is the *bias*).
- **Decide** — pass that total through one last step that turns it into the neuron's answer.

You'll meet each beat as its own unit today. Right now, just hold the shape: a neuron is a tiny **brain cell** that likes to *weigh → add → decide*. Everything else is detail.

@@@ concept id=c2 tag="Weights" title="Weights — how much you trust each input" gotit="Got weights"
Time for the first beat: **weigh**. Back to the movie night. Your friend with amazing taste? You turn their "trust dial" way up. The friend who hates everything? You turn theirs way down — maybe even into the negatives, so their thumbs-up actually *pushes you away*. Each friend gets their own dial, set once and left there, and it decides how loudly that friend's opinion lands in your head.

A neuron does the same. Every input arrives with its own [[weight||One number per input that scales how much that input matters. Big weight = "listen closely"; near-zero = "ignore"; negative = "count this against."]] — a number that says how much to trust it. Multiply each input by its weight and you've *weighed* it. Big weight means "listen closely." Near-zero means "ignore this one." A negative weight means "the more of this, the *less* I want it." Same brain cell as before — some incoming arms it trusts, some it doesn't.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three trust dials, one per friend. Friend one's dial is turned high (strong trust). Friend two's is near the middle (half trust). Friend three's points into the negative (counts against). Each dial is the weight for that input."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A weight = a trust dial, one per input</text><g><circle cx="110" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="110" y1="80" x2="134" y2="56" stroke="#2D8B55" stroke-width="3"/><circle cx="110" cy="80" r="4" fill="#2D8B55"/><text x="110" y="132" text-anchor="middle" fill="#6B645E">good-taste friend</text><text x="110" y="148" text-anchor="middle" fill="#276b45" font-weight="bold">weight = 0.9 (high)</text></g><g><circle cx="260" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="260" y1="80" x2="260" y2="48" stroke="#7C6DAA" stroke-width="3"/><circle cx="260" cy="80" r="4" fill="#7C6DAA"/><text x="260" y="132" text-anchor="middle" fill="#6B645E">so-so friend</text><text x="260" y="148" text-anchor="middle" fill="#5E5191" font-weight="bold">weight = 0.3 (low)</text></g><g><circle cx="410" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="410" y1="80" x2="386" y2="100" stroke="#C93B3B" stroke-width="3"/><circle cx="410" cy="80" r="4" fill="#C93B3B"/><text x="410" y="132" text-anchor="middle" fill="#6B645E">hates-everything friend</text><text x="410" y="148" text-anchor="middle" fill="#C93B3B" font-weight="bold">weight = −0.6 (counts against)</text></g><text x="260" y="178" text-anchor="middle" fill="#6B645E" font-size="10">input × weight = how loudly that input lands</text></g></svg>
%%%

**What the trust-dial picture gets right:** each input really does have its own fixed setting that scales its say, exactly like turning a knob up or down. **Where it breaks down:** you set your friend-dials once and rarely touch them — but a neuron's weights are meant to be *tuned* over and over until it gets things right. Slowly adjusting the weights (and the bias) so the neuron makes fewer mistakes has a name: **training**. That's a whole later day (Day 5); today the dials just sit at whatever numbers we hand them.

#### See a weight do its one job
A weight has exactly one job: multiply. Let's watch it. Say an input is `2` and its weight is `0.5`. **Predict first:** what's the weighed value? Then run it.

%%% demo id=weigh label="run it"
code: inputs = np.array([2, 3]); weights = np.array([0.5, -1.0]); inputs * weights
out: array([ 1., -3.])
take: <b>Each input is scaled by its own weight.</b> The first input <code>2</code> with weight <code>0.5</code> becomes <code>1</code> — halved, "half-trusted." The second input <code>3</code> with weight <code>−1.0</code> becomes <code>−3</code> — flipped negative, "counted against." That's the whole "weigh" beat: one multiply per input.
%%%

So a weight is just a trust dial made of arithmetic. Turn it up, the input shouts; turn it toward zero, it whispers; turn it negative, it argues the other way. **You just learned the first of a neuron's two sets of numbers** — the weights. The second is one lonely number called the bias, and it's next.

@@@ concept id=c3 tag="Bias" title="Bias — the neuron's starting mood" gotit="Got bias"
Here's a feeling you know well. Some mornings you wake up already cheerful — it takes almost nothing to tip you into a "yes." Other mornings you wake up grumpy, and even good news barely moves you. That built-in starting mood, before *anything* happens today, is the everyday picture of a [[bias||A single extra number added to the neuron's sum. It shifts the neuron's starting point — how easily it leans toward a "yes" before any input arrives.]].

A neuron carries one — and only one — extra number called the bias. After it weighs and adds up its inputs, it adds this one number on top. A big positive bias is the cheerful mood: the neuron leans "yes" easily. A big negative bias is the grumpy mood: it takes a lot of input to win it over. Unlike weights, there's just a single bias for the whole neuron — one starting mood, not one per input.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Two runners at the start of a race. One has a head start ahead of the line (a positive bias, an easy yes). One starts behind the line (a negative bias, a hard yes). The bias just shifts where the neuron begins before any input."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Bias = a head start (or a handicap) before the race begins</text><line x1="260" y1="40" x2="260" y2="150" stroke="#B8AEA2" stroke-width="2" stroke-dasharray="4,4"/><text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">start line (0)</text><circle cx="360" cy="70" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="360" y="75" text-anchor="middle" font-size="13">🏃</text><line x1="274" y1="70" x2="346" y2="70" stroke="#2D8B55" stroke-width="2"/><polygon points="346,64 360,70 346,76" fill="#2D8B55"/><text x="410" y="74" fill="#276b45">+bias: a head start</text><text x="410" y="88" fill="#276b45" font-size="10">easy "yes"</text><circle cx="150" cy="120" r="14" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="150" y="125" text-anchor="middle" font-size="13">🏃</text><line x1="246" y1="120" x2="174" y2="120" stroke="#C93B3B" stroke-width="2"/><polygon points="174,114 160,120 174,126" fill="#C93B3B"/><text x="110" y="102" text-anchor="middle" fill="#C93B3B">−bias: starts behind</text><text x="110" y="140" text-anchor="middle" fill="#C93B3B" font-size="10">hard "yes"</text></g></svg>
%%%

**What the mood / head-start picture gets right:** the bias really does move the neuron's *starting point* — how far it already leans before it hears a single input. **Where it breaks down:** your mood drifts on its own all day, but a neuron's bias is one fixed number that only changes when the neuron is being trained (Day 5). Today it's a dial we set and leave.

#### Watch the bias shift the total
Same weighed inputs as before, but now add a bias on the end. **Predict first:** the weighed inputs summed to `1 + (−3) = −2`. With a bias of `1`, what's the final total? Then run it.

%%% demo id=bias label="run it"
code: weighed = np.array([1.0, -3.0]); bias = 1.0; weighed.sum() + bias
out: -1.0
take: <b>Bias is added once, on top of the summed inputs.</b> The inputs summed to <code>−2</code>; the bias <code>+1</code> nudged the total up to <code>−1</code>. A bigger bias would push it toward "yes"; a negative bias would drag it toward "no." One number, added last.
%%%

That's it — the bias is the neuron's thumb on the scale, set before any input speaks. Weights say *how much each input matters*; the bias says *where the neuron starts from*. With both in hand, you're ready to see them team up into the neuron's one real calculation.

@@@ concept id=c4 tag="The sum" title="Adding it all up — the neuron's one number" gotit="Got the sum"
Now the "add" beat, where weights and bias finally work together. Picture a **shopping cart at the till**. Each item's price gets scanned (that's an input times its weight — a weighed value), the register keeps a running total, and at the end the cashier adds one flat service fee on top (that's the bias). One final number on the receipt: your total. A neuron builds its answer the exact same way.

The number a neuron gets from "weigh, then add, then add the bias" has a name: the [[weighted sum||The running total: every input times its weight, all added up, plus the bias. Also called the pre-activation, because it comes just before the final "decide" step. Written z.]], often called the **pre-activation** because it comes right *before* the final "decide" step. Engineers give it a short nickname: `z`.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A shopping-cart receipt. Two scanned items each show price times a weight, they add into a running total, then a flat service fee (the bias) is added on the bottom line to give the final total z."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The weighted sum z = ring up each item, add the fee</text><rect x="150" y="30" width="220" height="150" rx="4" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="150" y1="52" x2="370" y2="52" stroke="#E5DFD6"/><text x="260" y="47" text-anchor="middle" fill="#6B645E">— RECEIPT —</text><text x="166" y="74" fill="#5E5191">input₁ × w₁</text><text x="354" y="74" text-anchor="end" fill="#3A342E">1.0</text><text x="166" y="98" fill="#5E5191">input₂ × w₂</text><text x="354" y="98" text-anchor="end" fill="#3A342E">−3.0</text><line x1="166" y1="110" x2="354" y2="110" stroke="#B8AEA2"/><text x="166" y="128" fill="#6B645E">running total</text><text x="354" y="128" text-anchor="end" fill="#3A342E">−2.0</text><text x="166" y="150" fill="#8A6D3B">+ bias (fee)</text><text x="354" y="150" text-anchor="end" fill="#8A6D3B">+1.0</text><line x1="166" y1="160" x2="354" y2="160" stroke="#3A342E" stroke-width="1.5"/><text x="166" y="176" fill="#2D8B55" font-weight="bold">TOTAL  z</text><text x="354" y="176" text-anchor="end" fill="#2D8B55" font-weight="bold">−1.0</text></g></svg>
%%%

**What the receipt picture gets right:** the "scan each item, keep a running total, add one fee at the end" is exactly the order a neuron adds things up. **Where it breaks down:** on a receipt every price is positive, but a neuron's weighed values can be *negative* (a negative weight, remember) — so items can *subtract* from the total, which a real till never does.

The one line of math, said out loud in plain words:

%%% formula
expr: z = w₁·x₁ + w₂·x₂ + … + b
note: z is the total; each x is an input; each w is that input's weight; b is the one bias. In words: "weigh every input, add them up, then add the bias."
%%%

Read it left to right: `w₁·x₁` is "input one times its weight," you add on `w₂·x₂` for input two, keep going for every input, and finally add `b`, the bias. That's every symbol. Nothing hidden.

#### Watch the total build up, one piece at a time
Never trust a formula on faith — build it. Each bar below is one piece landing on the running total, ending at `z`. This is the receipt, drawn:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A running-total bar chart. Start at zero. Add plus one for the first weighed input. Add minus three for the second, dropping the total to minus two. Add plus one for the bias, ending at minus one, the value z."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">The total z builds up piece by piece</text><line x1="50" y1="120" x2="480" y2="120" stroke="#B8AEA2" stroke-width="1.5"/><text x="40" y="124" text-anchor="end" fill="#9A938A" font-size="10">0</text><rect x="70" y="90" width="70" height="30" fill="#7C6DAA" opacity="0.85"/><text x="105" y="110" fill="#fff">+1.0</text><text x="105" y="140" fill="#6B645E" font-size="10">w₁·x₁</text><text x="105" y="153" fill="#5E5191" font-size="10">total: 1.0</text><rect x="175" y="120" width="70" height="60" fill="#C93B3B" opacity="0.85"/><text x="210" y="155" fill="#fff">−3.0</text><text x="210" y="196" fill="#6B645E" font-size="10">w₂·x₂</text><text x="210" y="209" fill="#C93B3B" font-size="10">total: −2.0</text><rect x="280" y="90" width="70" height="30" fill="#C99A12" opacity="0.85"/><text x="315" y="110" fill="#fff">+1.0</text><text x="315" y="140" fill="#6B645E" font-size="10">bias b</text><text x="315" y="153" fill="#8A6D3B" font-size="10">total: −1.0</text><rect x="390" y="120" width="70" height="30" fill="#2D8B55"/><text x="425" y="140" fill="#fff">z</text><text x="425" y="170" fill="#276b45" font-size="11" font-weight="bold">z = −1.0</text></g></svg>
%%%

%%% demo id=zsum label="run it"
code: x = np.array([2, 3]); w = np.array([0.5, -1.0]); b = 1.0; z = (w * x).sum() + b; z
out: -1.0
take: <b>z = (w·x summed) + b = −1.0.</b> Two inputs weighed to <code>1.0</code> and <code>−3.0</code>, summed to <code>−2.0</code>, plus the bias <code>1.0</code>, lands at <code>z = −1.0</code>. That single number <code>z</code> is everything the neuron has to say — right before it decides.
%%%

!!! c-info 🪜
<b>Optional peek (skippable) — the same sum, spelled out.</b> With inputs <code>x = [2, 3]</code>, weights <code>w = [0.5, −1.0]</code>, and bias <code>b = 1.0</code>: first <code>0.5 × 2 = 1.0</code>, then <code>−1.0 × 3 = −3.0</code>. Add them: <code>1.0 + (−3.0) = −2.0</code>. Add the bias: <code>−2.0 + 1.0 = −1.0</code>. So <code>z = −1.0</code>. That's the whole calculation — two multiplies, two adds. You'll type it yourself in <b>Produce</b>.
!!!

**You just did the only real arithmetic a neuron ever does.** Everything left is what happens to that one number `z` next — the "decide" beat.

@@@ concept id=c5 tag="Decide" title="The decide step — turning z into an answer" gotit="Got the decide step"
The neuron has its one number, `z`. But a raw score isn't yet an *answer*. Picture a **judge at the end of a talent show**. The scores are in and totalled — that's `z` — and now the judge has to turn that total into a verdict the audience understands: "you're through" or "you're out," or maybe a confidence like "87%." Turning a total into a usable answer is the last beat, **decide**.

The step that does it is the [[activation function||The last step of a neuron: it takes the weighted sum z and turns it into the neuron's output. There's a whole family of them — a step, a smooth S-curve, and more.]]. Here's the shape to carry away: a neuron is always the **linear part** (the weigh-and-add that gives `z`) *then* the **activation** (the decide step on top of `z`). Same two-part pattern in every neuron, from the tiniest to the ones inside ChatGPT.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A talent-show judge. On the left, a scoreboard shows the total score z. An arrow leads to a judge who turns that score into a verdict: a yes or no, or a confidence percentage. The judge is the activation function."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The activation = a judge turning the score z into a verdict</text><rect x="40" y="60" width="120" height="60" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="100" y="82" text-anchor="middle" fill="#5E5191" font-size="10">weighted sum</text><text x="100" y="105" text-anchor="middle" fill="#3A342E" font-size="16" font-weight="bold">z = −1.0</text><text x="100" y="140" text-anchor="middle" fill="#6B645E" font-size="10">the linear part</text><line x1="164" y1="90" x2="230" y2="90" stroke="#B8AEA2" stroke-width="2.5"/><polygon points="230,84 244,90 230,96" fill="#B8AEA2"/><circle cx="300" cy="90" r="34" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="300" y="86" text-anchor="middle" font-size="18">⚖️</text><text x="300" y="104" text-anchor="middle" fill="#276b45" font-size="9">judge</text><text x="300" y="150" text-anchor="middle" fill="#276b45" font-size="10">the activation</text><line x1="336" y1="90" x2="400" y2="90" stroke="#B8AEA2" stroke-width="2.5"/><polygon points="400,84 414,90 400,96" fill="#B8AEA2"/><rect x="418" y="66" width="90" height="48" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="463" y="86" text-anchor="middle" fill="#8A6D3B" font-size="10">verdict</text><text x="463" y="104" text-anchor="middle" fill="#8A6D3B" font-size="10">yes / no / 87%</text></g></svg>
%%%

**What the judge picture gets right:** a raw total is not the same as a verdict, and something has to *convert* one into the other — that conversion is the activation's whole job. **Where it breaks down:** a human judge weighs feelings and hunches; a neuron's activation is one fixed, simple rule applied the same way every single time — no mood, no mercy.

#### Watch the verdict get built, step by step
Don't take "it turns `z` into an answer" on faith — let's construct that answer in three little steps, using the simplest activation (a hard yes/no at zero) so you can *see* each step land. Start from the raw score, ask the one question, read off the verdict:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Three steps that build the verdict from a raw score. Step one: the raw score z equals minus one. Step two: ask is z at or above zero? Here the answer is no. Step three: the verdict is zero, meaning off. Each step feeds the next."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">z → ask one question → the verdict</text><rect x="30" y="55" width="130" height="66" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="95" y="76" fill="#6B645E" font-size="10">1 · the raw score</text><text x="95" y="100" fill="#3A342E" font-size="16" font-weight="bold">z = −1.0</text><text x="95" y="116" fill="#9A938A" font-size="9">(from weigh + add)</text><line x1="162" y1="88" x2="192" y2="88" stroke="#B8AEA2" stroke-width="2"/><polygon points="192,83 202,88 192,93" fill="#B8AEA2"/><rect x="205" y="55" width="130" height="66" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="270" y="76" fill="#8A6D3B" font-size="10">2 · ask one thing</text><text x="270" y="98" fill="#3A342E" font-size="12">is z ≥ 0 ?</text><text x="270" y="116" fill="#C93B3B" font-size="11" font-weight="bold">no (−1.0 &lt; 0)</text><line x1="337" y1="88" x2="367" y2="88" stroke="#B8AEA2" stroke-width="2"/><polygon points="367,83 377,88 367,93" fill="#B8AEA2"/><rect x="380" y="55" width="120" height="66" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="440" y="76" fill="#6B645E" font-size="10">3 · the verdict</text><text x="440" y="100" fill="#C93B3B" font-size="16" font-weight="bold">0 (off)</text><text x="440" y="116" fill="#9A938A" font-size="9">the neuron's answer</text><text x="260" y="150" fill="#6B645E" font-size="10">raw score → one question → answer: that's the whole "decide" beat</text><text x="260" y="166" fill="#9A938A" font-size="9">a positive z would flip step 2 to "yes" and the verdict to 1 (on)</text></g></svg>
%%%

Read the three boxes left to right. **Box 1** is the raw score `z = −1.0`, straight from the weigh-and-add you just built. **Box 2** asks the activation's one question — here, "is `z` at or above zero?" — and the answer is *no*, because `−1.0` sits below `0`. **Box 3** reads off the verdict that answer forces: `0`, meaning "off." Flip `z` to a positive number and box 2 answers "yes," so box 3 becomes `1` ("on"). The answer isn't magic — it's assembled from the score by asking one question. (That's a second myth we'll come back to: neural nets look magic, but every answer is built from plain steps like this.)

#### It's a whole family, not one rule
Here's the fun part: there isn't *one* activation — there's a family of them, and choosing between them is a real design decision. A blunt on/off "yes or no" (the one you just watched). A smooth "how confident, 0 to 100%." Others you'll meet tomorrow. Today you'll meet the two that started it all: the original hard switch, and the smooth curve that replaced it. **You now know the shape of every neuron — linear part, then activation.** Next: the very first activation anyone ever tried.

@@@ concept id=c6 tag="First activation" title="The step function — a hard on/off switch" gotit="Met the step"
Let's meet the very first activation anyone ever built — it's the great-grandparent of the whole family, from way back in the 1940s, and its idea is beautifully simple. Picture the **height bar at a theme-park ride**: "you must be *this* tall to ride." A tall-enough rider clears the bar and gets waved on. A too-short rider is turned away at the gate. There's no "you're 80% on the ride." It's a hard yes or no, decided by one fixed bar.

That is the [[step function||The original activation: if the weighted sum z is at or above a threshold (zero), output 1 (fire); otherwise output 0. A hard on/off switch, with nothing in between.]] — the "decide" rule inside the earliest artificial neurons (people called them the **McCulloch–Pitts neuron** and later the **Perceptron**). The rule: if `z` reaches the bar (zero), the neuron fires and outputs `1`; if not, it stays quiet at `0`. On or off. Nothing in between.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A theme-park height bar. On the left a short person is below the bar and is turned away (output 0, off). On the right a tall person clears the bar and is let on (output 1, on). It is a hard cutoff with nothing in between."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Step = a height bar: tall enough (on) or not (off)</text><line x1="260" y1="45" x2="260" y2="150" stroke="#C99A12" stroke-width="3"/><text x="260" y="42" text-anchor="middle" fill="#8A6D3B" font-size="10">the bar (z = 0)</text><rect x="120" y="105" width="16" height="45" fill="#C93B3B" opacity="0.8"/><circle cx="128" cy="98" r="9" fill="#C93B3B" opacity="0.8"/><text x="128" y="164" text-anchor="middle" fill="#C93B3B" font-size="10">below the bar</text><text x="60" y="90" fill="#C93B3B" font-weight="bold">z &lt; 0 → 0 (off)</text><rect x="384" y="70" width="16" height="80" fill="#2D8B55"/><circle cx="392" cy="62" r="9" fill="#2D8B55"/><text x="392" y="164" text-anchor="middle" fill="#276b45" font-size="10">clears the bar</text><text x="420" y="90" text-anchor="end" fill="#276b45" font-weight="bold">z ≥ 0 → 1 (on)</text></g></svg>
%%%

**What the height-bar picture gets right:** one fixed bar splits the world into "on" and "off," with no middle ground — that's the step exactly. **Where it breaks down:** at a real ride, *you* stand at the bar; here the neuron's own number `z` is what gets measured against it. And unlike a friendly attendant, the step gives zero hint about *how far* over or under the bar you were — just the verdict.

#### Build it up from the height bar: watch riders sorted as z climbs
Let's construct the step one rider at a time, straight from the height-bar analogy. Line four riders up by their score `z`. Each walks to the bar; the attendant looks only at "did they reach the bar (z = 0)?" and waves them **on (1)** or **away (0)**. Watch the verdict flip the instant a rider crosses the bar — that flip *is* the step function being built:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Four riders lined up by their score z at a theme-park height bar. The bar sits at z equals zero. A rider with z of minus one point two is below the bar and turned away, output zero. A rider with z of minus zero point three is still below the bar, output zero. A rider with z of zero point one just clears the bar, output one. A rider with z of two point five clears it easily, output one. The verdict flips from zero to one the moment a rider crosses the bar."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Build the step: each rider meets the bar, gets waved on (1) or away (0)</text><line x1="270" y1="34" x2="270" y2="150" stroke="#C99A12" stroke-width="3"/><text x="270" y="30" fill="#8A6D3B" font-size="9">the bar (z = 0)</text><text x="120" y="30" fill="#C93B3B" font-size="9">below → away (0)</text><text x="420" y="30" fill="#276b45" font-size="9">at/above → on (1)</text><line x1="40" y1="150" x2="500" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><g><circle cx="90" cy="128" r="8" fill="#C93B3B" opacity="0.85"/><rect x="86" y="136" width="8" height="14" fill="#C93B3B" opacity="0.85"/><text x="90" y="172" fill="#6B645E" font-size="9">z = −1.2</text><text x="90" y="186" fill="#C93B3B" font-size="10" font-weight="bold">→ 0</text></g><g><circle cx="200" cy="128" r="8" fill="#C93B3B" opacity="0.85"/><rect x="196" y="136" width="8" height="14" fill="#C93B3B" opacity="0.85"/><text x="200" y="172" fill="#6B645E" font-size="9">z = −0.3</text><text x="200" y="186" fill="#C93B3B" font-size="10" font-weight="bold">→ 0</text></g><g><circle cx="330" cy="120" r="8" fill="#2D8B55"/><rect x="326" y="128" width="8" height="22" fill="#2D8B55"/><text x="330" y="172" fill="#6B645E" font-size="9">z = 0.1</text><text x="330" y="186" fill="#276b45" font-size="10" font-weight="bold">→ 1</text></g><g><circle cx="450" cy="112" r="8" fill="#2D8B55"/><rect x="446" y="120" width="8" height="30" fill="#2D8B55"/><text x="450" y="172" fill="#6B645E" font-size="9">z = 2.5</text><text x="450" y="186" fill="#276b45" font-size="10" font-weight="bold">→ 1</text></g><text x="260" y="204" fill="#9A938A" font-size="9">notice: 0.1 and 2.5 both just say "on" — the bar never asks HOW tall, only "tall enough?"</text></g></svg>
%%%

Read the four riders left to right. The first two (`z = −1.2` and `z = −0.3`) both fall short of the bar, so both are waved **away → 0**. The third (`z = 0.1`) *just* clears it and is waved **on → 1**; the fourth (`z = 2.5`) clears it easily — but gets the *same* verdict, `1`. That's the key thing you can see: the moment a rider crosses the bar the answer snaps from `0` to `1`, and after that, being taller changes nothing. Draw that snap as a curve and you get the step's famous shape — flat at `0`, a sudden jump at the bar, flat at `1`:

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Step function curve: output 0 for negative inputs, a sudden jump up to 1 at zero, then flat at 1 for positive inputs."><g font-family="monospace"><line x1="60" y1="100" x2="340" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><line x1="200" y1="30" x2="200" y2="118" stroke="#E5DFD6" stroke-width="1"/><path d="M70 100 L200 100 L200 48 L320 48" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="200" cy="48" r="3.5" fill="#8A6D3B"/><text x="130" y="117" font-size="11" fill="#6B645E" text-anchor="middle">output 0 (away)</text><text x="265" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">output 1 (on)</text><text x="200" y="133" font-size="10" fill="#9A938A" text-anchor="middle">the same snap, drawn: a cliff at the bar — no in-between</text><text x="420" y="76" font-size="12" fill="#8A6D3B">step → {0, 1}</text></g></svg>
%%%

%%% demo id=step label="run it"
code: step(np.array([-1.2, -0.3, 0.1, 2.5]))
out: array([0., 0., 1., 1.])
take: <b>Step = 1 if z ≥ 0, else 0.</b> The same four riders: the two below the bar become <code>0</code>, the two at or above it become <code>1</code>. Notice <code>0.1</code> and <code>2.5</code> give the identical answer <code>1</code> — the bar never asks "how tall," only "tall enough?"
%%%

Here's the little puzzle it leaves us — and it's *why* the next activation exists. The step is a cliff. Learning works by nudging the dials a tiny bit and checking if the answer got a little better. But a tiny nudge to a cliff does *nothing* until you tumble right over the edge — the output doesn't budge, so the neuron gets no hint which way to improve. **You can already explain why the earliest neurons got stuck.** The fix is coming next: swap the cliff for a smooth ramp.

@@@ concept id=c7 tag="Smooth version" title="Sigmoid — a smooth dimmer, not a cliff" gotit="Met sigmoid"
So we want the "decide" step to change *gradually*, so a small nudge to the dials makes a small, visible change in the answer. Swap the light *switch* for a **dimmer**. A dimmer slides smoothly from off to fully on, passing through every level of brightness in between — dim, medium, bright. Nudge it a hair and the light changes a hair. That gentle slide is the fix the step never had.

That smooth dimmer is the [[sigmoid||A smooth S-shaped activation that squashes any number into the range 0 to 1. It replaces the step's hard cliff with a gentle slide, so a small change in z makes a small change in the output.]] activation. It takes any score `z` and squashes it smoothly into the range 0 to 1 — think of it as "how confident, from 0% to 100%." Big negative scores land near `0`, big positives near `1`, and a score of exactly `0` sits right in the middle at `0.5`.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A dimmer switch shown as a horizontal slider. On the left the slider is low and the bulb is dark, near zero. In the middle it is half bright at one half. On the right it is full and the bulb is fully bright, near one. It slides smoothly through every level in between."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid = a dimmer that slides smoothly off → on</text><rect x="60" y="78" width="400" height="14" rx="7" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="110" cy="85" r="12" fill="#5E5191"/><text x="110" y="114" text-anchor="middle" fill="#6B645E">low</text><text x="110" y="128" text-anchor="middle" fill="#9A938A" font-size="10">→ near 0</text><circle cx="260" cy="58" r="16" fill="#EFEAF0" stroke="#9A938A"/><text x="260" y="64" text-anchor="middle" font-size="14">🌗</text><circle cx="350" cy="54" r="18" fill="#F6EFCF" stroke="#C99A12"/><text x="350" y="60" text-anchor="middle" font-size="14">🔆</text><circle cx="430" cy="50" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="430" y="57" text-anchor="middle" font-size="16">💡</text><text x="260" y="114" text-anchor="middle" fill="#5E5191">middle</text><text x="260" y="128" text-anchor="middle" fill="#9A938A" font-size="10">= 0.5</text><text x="430" y="114" text-anchor="middle" fill="#8A6D3B">full</text><text x="430" y="128" text-anchor="middle" fill="#9A938A" font-size="10">→ near 1</text><text x="260" y="158" text-anchor="middle" fill="#6B645E" font-size="10">no hard jump — every level in between is a real answer in (0, 1)</text></g></svg>
%%%

**What the dimmer picture gets right:** the smooth, gradual slide from off to on, with every brightness in between being a real setting — that's the heart of sigmoid. **Where it breaks down:** a real dimmer is evenly graded end to end, but sigmoid *flattens out* at the far ends — push the score very high or very low and the light barely changes. (That flattening hides a puzzle you'll crack tomorrow — hold the thought.)

#### The cliff becomes a ramp — see them side by side
Here is the whole point of this unit in one picture: the step's cliff (left) versus sigmoid's smooth ramp (right). The ramp is why the neuron can finally learn from tiny nudges.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="Left panel: the step function as a sharp cliff jumping from 0 to 1. Right panel: the sigmoid as a smooth S-shaped ramp rising gently from near 0 to near 1. The cliff gives no in-between; the ramp does."><g font-family="monospace" font-size="11"><text x="130" y="20" text-anchor="middle" fill="#C93B3B" font-size="11">step: a cliff (no in-between)</text><line x1="40" y1="120" x2="230" y2="120" stroke="#E5DFD6" stroke-width="1.5"/><line x1="135" y1="45" x2="135" y2="132" stroke="#E5DFD6" stroke-width="1"/><path d="M50 120 L135 120 L135 60 L225 60" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="135" cy="60" r="3" fill="#8A6D3B"/><text x="130" y="150" text-anchor="middle" fill="#9A938A" font-size="10">tiny nudge → nothing moves</text><text x="390" y="20" text-anchor="middle" fill="#276b45" font-size="11">sigmoid: a smooth ramp</text><line x1="295" y1="120" x2="490" y2="120" stroke="#E5DFD6" stroke-width="1.5"/><line x1="392" y1="45" x2="392" y2="132" stroke="#E5DFD6" stroke-width="1"/><path d="M300 116 C 360 116, 372 64, 392 62 C 412 60, 424 58, 485 57" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><text x="390" y="150" text-anchor="middle" fill="#2D8B55" font-size="10">tiny nudge → answer moves a little</text></g></svg>
%%%

%%% demo id=sigmoid label="run it"
code: sigmoid(np.array([-4.0, -1.0, 0.0, 1.0, 4.0]))
out: array([0.018, 0.269, 0.5  , 0.731, 0.982])
take: <b>Sigmoid squashes any score into (0, 1).</b> A smooth off → on: <code>sigmoid(0) = 0.5</code> sits dead center, big negatives land near <code>0</code>, big positives near <code>1</code> (values rounded). Because every answer lands between 0 and 1, it reads like a confidence — perfect when you want "how sure, 0 to 100%."
%%%

Feel the difference: with the step, a small nudge to a dial changed *nothing* (or flipped the whole answer). With sigmoid, a small nudge makes a small, honest change you can follow — which is exactly what learning needs. **You now know both bookends of the activation family: the original hard step, and the smooth sigmoid that replaced it.** Tomorrow you'll meet the rest. Today, let's watch the *whole* neuron run start to finish.

@@@ concept id=c8 tag="Forward pass" title="Forward propagation — run the whole neuron" gotit="Got the forward pass"
Let's run the whole brain cell end to end: **weigh → add → decide**, in one clean sweep. Picture a **factory conveyor belt**. Raw parts go on at one end, move steadily to the right through station after station, and a finished product rolls off the far end. Nothing goes backward; each station does its one job and passes the work along. A neuron computes its answer exactly like that — data enters, flows left to right, one number rolls out.

That left-to-right sweep has a name: [[forward propagation||Pushing the inputs left-to-right through the neuron — first the weighted sum, then the activation — to produce the output. Also called the forward pass.]], or the **forward pass**. It's just the three beats you already know, done in order, in one direction.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A factory conveyor belt moving left to right. Inputs enter on the left, pass a weigh-and-add station that produces the score z, then a decide station holding the activation, and a finished output rolls off the right end. Everything flows one way."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Forward pass = a conveyor: inputs in, one answer out</text><rect x="30" y="120" width="460" height="10" rx="5" fill="#D8D2C8"/><circle cx="45" cy="135" r="8" fill="#B8AEA2"/><circle cx="120" cy="135" r="8" fill="#B8AEA2"/><circle cx="260" cy="135" r="8" fill="#B8AEA2"/><circle cx="400" cy="135" r="8" fill="#B8AEA2"/><circle cx="475" cy="135" r="8" fill="#B8AEA2"/><rect x="40" y="55" width="70" height="45" rx="5" fill="#FDF9F3" stroke="#B8AEA2"/><text x="75" y="74" text-anchor="middle" fill="#6B645E" font-size="10">inputs</text><text x="75" y="90" text-anchor="middle" fill="#3A342E" font-size="10">x = [2, 3]</text><text x="122" y="82" fill="#B8AEA2">→</text><rect x="145" y="55" width="110" height="45" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="200" y="74" text-anchor="middle" fill="#5E5191" font-size="10">weigh &amp; add</text><text x="200" y="90" text-anchor="middle" fill="#3A342E" font-size="10">z = −1.0</text><text x="267" y="82" fill="#B8AEA2">→</text><rect x="290" y="55" width="110" height="45" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="345" y="74" text-anchor="middle" fill="#276b45" font-size="10">decide (sigmoid)</text><text x="345" y="90" text-anchor="middle" fill="#3A342E" font-size="10">→ 0.27</text><text x="412" y="82" fill="#B8AEA2">→</text><rect x="430" y="55" width="60" height="45" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="460" y="74" text-anchor="middle" fill="#8A6D3B" font-size="10">output</text><text x="460" y="90" text-anchor="middle" fill="#8A6D3B" font-size="10">0.27</text><text x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="10">one direction only — left to right, no going back</text></g></svg>
%%%

**What the conveyor picture gets right:** work moves one way, each station does its job once, and a finished result comes off the end — that's the forward pass exactly. **Where it breaks down:** a factory belt never runs in reverse, but a neuron *does* have a backward trip — that's how it learns (it's called backprop, and it's a later day). Today we only ride the belt forward.

#### Run the whole neuron, one line
Everything you built today, in a single pass — weigh, add the bias, then decide with sigmoid:

%%% demo id=forward label="run it"
code: x = np.array([2, 3]); w = np.array([0.5, -1.0]); b = 1.0; z = (w*x).sum() + b; out = sigmoid(z); (z, round(out, 3))
out: (-1.0, 0.269)
take: <b>That's a full forward pass.</b> The inputs flowed through weigh-and-add to <code>z = −1.0</code>, then the sigmoid "decide" step turned that into <code>0.269</code> — a soft "probably no, about 27% yes." Weigh → add → decide, start to finish, in one direction.
%%%

That's a complete neuron, computed exactly the way the biggest models compute *theirs* — millions of these little forward passes, all at once. **You just ran your first neural computation, whole.** Now for the twist: this tidy little machine has one thing it flat-out *cannot* do.

@@@ concept id=c9 tag="The line" title="A neuron draws one straight line" gotit="Got the line"
Here's a beautiful way to *see* what a neuron really does. Imagine a **playground with kids scattered around**, and you're picking two teams. You grab a piece of chalk and draw one straight line across the ground: everyone on this side, team A; everyone on that side, team B. One straight chalk line, and the whole yard is split in two. A neuron makes exactly that kind of call.

When a neuron looks at two inputs, you can picture every possible input as a dot on a flat map. The neuron's weights and bias together draw **one straight line** across that map — a [[decision boundary||The straight line (or flat plane) a neuron draws to split its inputs into two sides — "yes" on one side, "no" on the other. Its position and tilt come from the weights and bias.]]. Dots on one side get a "yes," dots on the other get a "no." The **weights** tilt the line; the **bias** slides it. That's the geometry of weigh → add → decide.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A playground map with scattered dots. One straight chalk line splits them into two teams: green dots on one side get a yes, red dots on the other get a no. The line's tilt is set by the weights and its position by the bias."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A neuron = one straight chalk line splitting the yard</text><rect x="150" y="32" width="220" height="130" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="160" y1="150" x2="360" y2="44" stroke="#9A5A12" stroke-width="3"/><text x="378" y="46" fill="#9A5A12" font-size="10">the line</text><circle cx="195" cy="70" r="7" fill="#2D8B55"/><circle cx="230" cy="58" r="7" fill="#2D8B55"/><circle cx="215" cy="95" r="7" fill="#2D8B55"/><circle cx="260" cy="72" r="7" fill="#2D8B55"/><text x="205" y="120" text-anchor="middle" fill="#276b45" font-size="10">"yes" side</text><circle cx="300" cy="120" r="7" fill="#C93B3B"/><circle cx="330" cy="135" r="7" fill="#C93B3B"/><circle cx="315" cy="105" r="7" fill="#C93B3B"/><circle cx="345" cy="118" r="7" fill="#C93B3B"/><text x="330" y="90" text-anchor="middle" fill="#C93B3B" font-size="10">"no" side</text><text x="70" y="80" text-anchor="middle" fill="#6B645E" font-size="10">weights</text><text x="70" y="94" text-anchor="middle" fill="#6B645E" font-size="10">tilt it</text><text x="70" y="126" text-anchor="middle" fill="#6B645E" font-size="10">bias</text><text x="70" y="140" text-anchor="middle" fill="#6B645E" font-size="10">slides it</text></g></svg>
%%%

**What the chalk-line picture gets right:** the neuron really does carve its world into two sides with a single straight cut, and where that cut sits is *all* it can control. **Where it breaks down:** a chalk line lives on a flat playground you can see; a real neuron often splits a space with many more than two inputs — the "line" becomes a flat sheet (a *plane*) in a space too big to draw. The idea is the same; you just can't sketch it past three inputs.

#### Grab the dials and move the line yourself
Don't take this on faith — **play with it.** Drag the weight and bias sliders below and watch the neuron's line tilt and slide across the map in real time. Try to *predict* which way the line will swing before you let go of a slider.

%%% svg
<svg id="db-svg" viewBox="0 0 520 232" role="img" aria-label="Interactive decision boundary. Drag the weight and bias sliders and watch the neuron's straight line tilt and slide across a map of dots, recoloring each dot green for yes or red for no."><g font-family="monospace" font-size="11"><rect x="60" y="30" width="400" height="180" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="120" x2="460" y2="120" stroke="#EFE9DF" stroke-width="1"/><line x1="260" y1="30" x2="260" y2="210" stroke="#EFE9DF" stroke-width="1"/><line id="db-line" x1="60" y1="30" x2="460" y2="210" stroke="#9A5A12" stroke-width="3"/><circle id="db-p0" cx="110" cy="66" r="7" fill="#C93B3B"/><circle id="db-p1" cx="180" cy="102" r="7" fill="#C93B3B"/><circle id="db-p2" cx="310" cy="53" r="7" fill="#2D8B55"/><circle id="db-p3" cx="380" cy="84" r="7" fill="#2D8B55"/><circle id="db-p4" cx="140" cy="161" r="7" fill="#C93B3B"/><circle id="db-p5" cx="290" cy="183" r="7" fill="#C93B3B"/><circle id="db-p6" cx="400" cy="147" r="7" fill="#2D8B55"/><circle id="db-p7" cx="220" cy="143" r="7" fill="#C93B3B"/><text x="468" y="42" fill="#276b45" font-size="10">yes ✓</text><text x="468" y="206" fill="#C93B3B" font-size="10">no ✗</text></g></svg>
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
      w1v=document.getElementById('db-w1v'),w2v=document.getElementById('db-w2v'),bv=document.getElementById('db-bv');
  var pts=[[-1.5,1.2],[-0.8,0.4],[0.5,1.5],[1.2,0.8],[-1.2,-0.9],[0.3,-1.4],[1.4,-0.6],[-0.4,-0.5]];
  function px(x){return 60+(x+2)/4*400;}
  function py(y){return 210-(y+2)/4*180;}
  function ends(a,c,d){var lo=-2,hi=2,e=[];
    if(Math.abs(c)>1e-9){[lo,hi].forEach(function(x1){var x2=-(a*x1+d)/c;if(x2>=lo-1e-9&&x2<=hi+1e-9)e.push([x1,x2]);});}
    if(Math.abs(a)>1e-9){[lo,hi].forEach(function(x2){var x1=-(c*x2+d)/a;if(x1>=lo-1e-9&&x1<=hi+1e-9)e.push([x1,x2]);});}
    return e;}
  function sgn(n){return (n<0?'− ':'+ ')+Math.abs(n).toFixed(1);}
  function paint(){
    var a=+w1.value,c=+w2.value,d=+b.value;
    w1v.textContent=a.toFixed(1);w2v.textContent=c.toFixed(1);bv.textContent=d.toFixed(1);
    var yes=0;
    for(var i=0;i<pts.length;i++){var z=a*pts[i][0]+c*pts[i][1]+d;var el=document.getElementById('db-p'+i);
      if(el)el.setAttribute('fill',z>=0?'#2D8B55':'#C93B3B');if(z>=0)yes++;}
    var e=ends(a,c,d);
    if(e.length>=2){line.setAttribute('x1',px(e[0][0]).toFixed(1));line.setAttribute('y1',py(e[0][1]).toFixed(1));
      line.setAttribute('x2',px(e[1][0]).toFixed(1));line.setAttribute('y2',py(e[1][1]).toFixed(1));line.setAttribute('opacity','1');}
    else line.setAttribute('opacity','0');
    out.innerHTML='boundary: <b>'+a.toFixed(1)+'·x₁ '+sgn(c)+'·x₂ '+sgn(d)+' = 0</b> — '+yes+' of 8 dots land on the “yes” side.';
  }
  [w1,w2,b].forEach(function(el){el.addEventListener('input',paint);});paint();
})();</script>
%%%

Notice the one thing you can never do here: no matter how you set the dials, the boundary stays a **straight line**. You can tilt it, slide it, flip which side is "yes" — but you can't bend it into a curve or an L-shape. **You just saw, with your own hands, the single most important fact about one neuron: it draws exactly one straight line.** Hold that — it's about to become the neuron's one real limit.

@@@ concept id=c10 tag="The limit" title="One line can't do XOR" gotit="Got the limit"
Now the famous puzzle that stumped this whole field for years. Picture a **room with four chairs**, one in each corner. Two red chairs sit in the front-left and back-right corners; two blue chairs sit in the front-right and back-left — so each color sits on its own **diagonal**. I hand you one straight **rope** pulled tight across the floor, and ask: put both red chairs on one side and both blue chairs on the other. Go on — try to picture it.

You can't. Slide the rope, angle it any way you like — one straight rope always leaves one chair on the wrong side. The best you'll ever manage is three of four. That room is the [[XOR||"On when exactly one of two inputs is on." Its two answer-groups sit on opposite diagonals, so no single straight line can separate them.]] pattern: "yes when exactly one of two inputs is on." And a single neuron, you just saw, draws exactly **one straight line** — your one straight rope. XOR is not [[linearly separable||A pattern is linearly separable if one straight line can put every "yes" on one side and every "no" on the other. XOR is the classic pattern that is NOT — no single line works.]], so one neuron simply cannot learn it.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A room floor with four chairs at the corners. Two red chairs sit on one diagonal, two blue chairs on the other. A single straight rope drawn across the floor always leaves one chair on the wrong side — the best you can do is three of four."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One straight rope can't split two diagonal pairs</text><rect x="150" y="34" width="220" height="120" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><rect x="172" y="54" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="185" y="72" text-anchor="middle" fill="#C93B3B">🔴</text><rect x="322" y="108" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="335" y="126" text-anchor="middle" fill="#C93B3B">🔴</text><rect x="322" y="54" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="335" y="72" text-anchor="middle" fill="#5E5191">🔵</text><rect x="172" y="108" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="185" y="126" text-anchor="middle" fill="#5E5191">🔵</text><line x1="140" y1="150" x2="390" y2="46" stroke="#9A5A12" stroke-width="3"/><text x="405" y="48" fill="#9A5A12" font-size="10">straight rope</text><text x="70" y="94" text-anchor="middle" fill="#6B645E" font-size="10">reds on one</text><text x="70" y="108" text-anchor="middle" fill="#6B645E" font-size="10">diagonal,</text><text x="70" y="122" text-anchor="middle" fill="#6B645E" font-size="10">blues the other</text><text x="260" y="176" text-anchor="middle" fill="#C93B3B" font-size="10">any angle → best you get is 3 of 4</text></g></svg>
%%%

**What the room-of-chairs picture gets right:** the diagonal layout is the whole trap — two groups arranged so no single straight cut can ever separate them, which is precisely XOR. **Where it breaks down:** with a real rope you're stuck for good, but a network isn't limited to one rope (that's the next unit's rescue).

#### Try to beat it yourself — you can't
**Step 1 — slide the single rope and watch it fail.** In the widget above, drag the slider to slide the one straight rope across the floor. You'll always fence off three chairs, never the fourth — the one that escapes gets a ring around it, and you're stuck at 3 of 4 no matter where you slide. (Curious about the fix? Tick **add a hidden layer** to drop in a second rope — that's the next unit, previewed.)

%%% hint
t1: Don't worry about the math yet — just look at the four chairs. Same-colored chairs sit on opposite corners (a diagonal), not next to each other. Ask yourself: could one straight rope ever keep two things that sit on opposite corners on the same side?
t2: Try it slowly. Put the rope so it fences off the top-left chair alone — good, that one's handled. Now its same-color partner is way down at the bottom-right, on the far side of your rope. To grab it too you'd have to swing the rope over there — but then you let one of the OTHER color slip onto your side. You keep trading one mistake for another.
t3: One neuron draws exactly one straight line, and XOR's two groups sit on crossing diagonals — so no single straight line can ever put all the "yes"s on one side and all the "no"s on the other. That's the whole limit.
%%%

%%% svg
<svg id="xor-svg" viewBox="0 0 520 236" role="img" aria-label="Interactive XOR. Slide a single straight rope across four corner chairs and watch it catch at most three of four, with a ring around the one that escapes. Tick the hidden-layer box to add a second rope that forms a band and catches all four."><g font-family="monospace" font-size="11"><rect x="150" y="30" width="220" height="180" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line id="xor-l2" x1="0" y1="0" x2="0" y2="0" stroke="#2D8B55" stroke-width="3" stroke-dasharray="6,4" opacity="0"/><line id="xor-l1" x1="190" y1="106" x2="274" y2="190" stroke="#9A5A12" stroke-width="3"/><circle id="xor-miss" cx="330" cy="50" r="15" fill="none" stroke="#C99A12" stroke-width="2.5" stroke-dasharray="3,3" opacity="1"/><circle cx="190" cy="190" r="9" fill="#5E5191"/><circle cx="190" cy="50" r="9" fill="#C93B3B"/><circle cx="330" cy="190" r="9" fill="#C93B3B"/><circle cx="330" cy="50" r="9" fill="#5E5191"/><text x="150" y="224" fill="#5E5191" font-size="9">🔵 off = both/neither on</text><text x="300" y="224" fill="#C93B3B" font-size="9">🔴 on = exactly one</text></g></svg>
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
  function ends(c){var e=[];[0,1].forEach(function(x1){var x2=c-x1;if(x2>=0-1e-9&&x2<=1+1e-9)e.push([x1,x2]);});
    [0,1].forEach(function(x2){var x1=c-x2;if(x1>=0-1e-9&&x1<=1+1e-9)e.push([x1,x2]);});return e;}
  function setLine(el,c){var e=ends(c);if(e.length>=2){el.setAttribute('x1',px(e[0][0]).toFixed(1));el.setAttribute('y1',py(e[0][1]).toFixed(1));
    el.setAttribute('x2',px(e[1][0]).toFixed(1));el.setAttribute('y2',py(e[1][1]).toFixed(1));el.setAttribute('opacity','1');}}
  var P=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]; // x1,x2,trueClass
  function paint(){
    if(hl.checked){
      setLine(l1,0.5);setLine(l2,1.5);miss.setAttribute('opacity','0');cs.disabled=true;
      out.innerHTML='two ropes make a <b>band</b> — inside = “on”, outside = “off”. All <b>4 of 4</b> split ✓ — that’s a hidden layer.';
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

**Step 2 — name what you saw.** That's the limit of *one* neuron: it draws one straight line, and some patterns (like XOR) can't be split by any single line, no matter the tilt. This isn't a bug in your dial-setting — it's a hard ceiling on what one straight line can do. So beware the myth that a neural net "can solve *any* problem": one neuron alone provably can't even do XOR. **You've just found the exact edge of a single neuron's power** — and knowing a tool's limit is as important as knowing its use. The fix is waiting in the very next unit.

@@@ concept id=c11 tag="Layers & networks" title="Join neurons into a network and the line can bend" gotit="Got layers"
So one rope can't split the four chairs. The fix is almost silly once you see it: **use more than one rope.** Give yourself a *second* straight rope and a way to join them, and now you can fence off a strip of floor — one rope on each side of a diagonal band, and together they wrap the two chairs in the middle while the other two sit outside. Two simple straight pieces combine into something bent.

That's exactly what happens when you put several neurons side by side. A [[layer||A row of neurons working side by side, all reading the SAME inputs. Each neuron draws its own straight line; together the layer draws several lines at once.]] is just a **row of neurons** that all read the same inputs at the same time — think of a panel of judges, each scoring the same act with their own opinion. One neuron draws one line; a layer of, say, two neurons draws two lines at once.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A layer drawn as a row of two neurons side by side. The same two inputs on the left fan out to both neurons. Each neuron draws its own straight line, so the layer produces two lines from the same inputs. This is like a panel of judges each scoring the same act."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A layer = a row of neurons, all reading the SAME inputs</text><circle cx="55" cy="70" r="9" fill="#EFEAF7" stroke="#7C6DAA"/><text x="55" y="74" text-anchor="middle" fill="#5E5191" font-size="9">x₁</text><circle cx="55" cy="120" r="9" fill="#EFEAF7" stroke="#7C6DAA"/><text x="55" y="124" text-anchor="middle" fill="#5E5191" font-size="9">x₂</text><line x1="64" y1="70" x2="180" y2="65" stroke="#B8AEA2"/><line x1="64" y1="120" x2="180" y2="75" stroke="#B8AEA2"/><line x1="64" y1="70" x2="180" y2="130" stroke="#B8AEA2"/><line x1="64" y1="120" x2="180" y2="140" stroke="#B8AEA2"/><rect x="180" y="52" width="120" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="240" y="70" text-anchor="middle" fill="#276b45" font-size="10">neuron A</text><text x="240" y="86" text-anchor="middle" fill="#6B645E" font-size="9">draws line 1</text><rect x="180" y="112" width="120" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="240" y="130" text-anchor="middle" fill="#276b45" font-size="10">neuron B</text><text x="240" y="146" text-anchor="middle" fill="#6B645E" font-size="9">draws line 2</text><text x="120" y="44" text-anchor="middle" fill="#6B645E" font-size="10">same inputs</text><text x="120" y="170" text-anchor="middle" fill="#9A938A" font-size="9">fan out to every neuron</text><path d="M312 74 Q 340 90 312 134" fill="none" stroke="#B8AEA2" stroke-width="1.5"/><text x="380" y="98" text-anchor="middle" fill="#276b45" font-size="10">2 neurons</text><text x="380" y="114" text-anchor="middle" fill="#276b45" font-size="10">= 2 lines at once</text></g></svg>
%%%

**What the panel-of-judges picture gets right:** every neuron in a layer sees the very same inputs and forms its own separate opinion (its own line) — just like judges scoring one act. **Where it breaks down:** real judges argue and influence each other; the neurons in a layer never talk to each other — they work in parallel, and only their outputs get combined afterward, by the *next* layer.

#### Stack the layers and you've built a network
Here's the move that changes everything. Feed the outputs of one layer as the *inputs* to another layer, and you've stacked them. That stack is a [[neural network||Several layers of neurons connected in a chain: each layer's outputs become the next layer's inputs. The network as a whole can bend and fold boundaries a single neuron never could.]] — neurons wired in a chain, layer feeding layer. Every neuron still draws just one straight line, but the second layer gets to *combine* the first layer's lines — through the little "bend" you'll meet tomorrow — into a boundary that curves and folds. The neuron's limit doesn't vanish; it gets *outgrown* by teamwork.

The layers even have names, based on where they sit in the chain:
- The [[input layer||The raw numbers you feed in — the very first values that enter the network. Not neurons that compute, just the inputs handed to the first real layer.]] is just the raw inputs you hand in — the numbers at the very front (`x₁`, `x₂`, …).
- A [[hidden layer||A layer of neurons in the MIDDLE of a network — between the inputs and the final answer. It's "hidden" because you never read its outputs directly; they only feed the next layer. Hidden layers are what let the boundary bend.]] is any layer of neurons in the *middle* — between the inputs and the answer. It's called "hidden" because you never look at its numbers directly; they exist only to feed the next layer. Adding even one hidden layer is what let the four chairs finally split.
- The [[output layer||The LAST layer, whose result is the network's final answer — the number (or numbers) you actually read and use.]] is the very last layer — its result is the answer you actually read and use.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A small neural network in three named columns. On the left, the input layer is two raw input values. In the middle, a hidden layer of two neurons, each connected to both inputs. On the right, an output layer of one neuron producing the final answer. Arrows flow left to right from inputs through the hidden layer to the output."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A network: input layer → hidden layer → output layer</text><circle cx="70" cy="75" r="12" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="70" y="79" text-anchor="middle" fill="#6B645E" font-size="9">x₁</text><circle cx="70" cy="135" r="12" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="70" y="139" text-anchor="middle" fill="#6B645E" font-size="9">x₂</text><text x="70" y="172" text-anchor="middle" fill="#6B645E" font-size="10">input layer</text><text x="70" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(raw inputs)</text><line x1="82" y1="75" x2="248" y2="75" stroke="#B8AEA2"/><line x1="82" y1="75" x2="248" y2="135" stroke="#B8AEA2"/><line x1="82" y1="135" x2="248" y2="75" stroke="#B8AEA2"/><line x1="82" y1="135" x2="248" y2="135" stroke="#B8AEA2"/><circle cx="260" cy="75" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><circle cx="260" cy="135" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="260" y="172" text-anchor="middle" fill="#276b45" font-size="10">hidden layer</text><text x="260" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(the bend lives here)</text><line x1="274" y1="75" x2="426" y2="105" stroke="#B8AEA2"/><line x1="274" y1="135" x2="426" y2="105" stroke="#B8AEA2"/><circle cx="440" cy="105" r="14" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="440" y="172" text-anchor="middle" fill="#8A6D3B" font-size="10">output layer</text><text x="440" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(the answer)</text><line x1="454" y1="105" x2="500" y2="105" stroke="#C99A12" stroke-width="2"/><polygon points="500,100 512,105 500,110" fill="#C99A12"/></g></svg>
%%%

**What the "named columns" picture gets right:** the inputs enter at the left, flow through one or more hidden layers in the middle, and the output layer at the right hands back the answer — that left-to-right chain is exactly a network. **Where it breaks down:** real networks can have many hidden layers and thousands of neurons per layer — so many you couldn't draw them — but the shape is always this same input → hidden → output chain.

#### That's what "deep" means
Now the word you've heard a hundred times finally has a plain meaning. When a network has **many hidden layers stacked one after another**, we call it **deep** — and training such a network is [[deep learning||Training a neural network that has many hidden layers stacked in a chain. "Deep" literally means "many layers deep"; each layer bends the boundary a little more.]]. That's the whole secret behind the term: "deep" just means "many layers deep." Each extra hidden layer bends the boundary a little more, so a *deep* stack can carve shapes wildly more intricate than any single line. ChatGPT is a very deep network — a tall chain of these layers.

!!! c-warn ⚠️
<b>Myth to dodge — "bigger is always better."</b> It's tempting to think "more layers, more neurons, always smarter." Not so: a network that's too big for its data just *memorizes* the examples instead of learning the real pattern (you'll meet this as **overfitting** later), and it costs far more to train and run. The right size fits the problem — bigger is a tool, not a magic wand.
!!!

#### See the fix: one line fails, a hidden layer bends it
Here is the whole point in one picture — the same four chairs, split by one line (fails) versus by two lines from a hidden layer (works). The two lines form a **band** that fences the two blue dots inside while the two reds stay outside:

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="Left: one straight line tries to separate two diagonal pairs of dots and fails, getting three of four. Right: two parallel straight lines from a hidden layer form a diagonal band. Both blue dots (top-right and bottom-left) fall inside the band; both red dots (top-left and bottom-right) fall outside it. The band cleanly separates the two classes, four of four."><g font-family="monospace" font-size="11"><text x="128" y="20" text-anchor="middle" fill="#C93B3B" font-size="11">one neuron: one line (fails)</text><rect x="33" y="34" width="190" height="120" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><circle cx="68" cy="60" r="6" fill="#C93B3B"/><circle cx="188" cy="128" r="6" fill="#C93B3B"/><circle cx="188" cy="60" r="6" fill="#5E5191"/><circle cx="68" cy="128" r="6" fill="#5E5191"/><line x1="43" y1="140" x2="213" y2="55" stroke="#9A5A12" stroke-width="2.5"/><text x="128" y="172" text-anchor="middle" fill="#C93B3B" font-size="10">stuck at 3 of 4</text><text x="392" y="20" text-anchor="middle" fill="#276b45" font-size="11">hidden layer: two lines bend (works)</text><rect x="297" y="34" width="190" height="120" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><circle cx="332" cy="60" r="6" fill="#C93B3B"/><circle cx="452" cy="128" r="6" fill="#C93B3B"/><circle cx="452" cy="60" r="6" fill="#5E5191"/><circle cx="332" cy="128" r="6" fill="#5E5191"/><line x1="346" y1="152" x2="466" y2="84" stroke="#2D8B55" stroke-width="2.5"/><line x1="318" y1="104" x2="438" y2="36" stroke="#2D8B55" stroke-width="2.5"/><text x="392" y="172" text-anchor="middle" fill="#276b45" font-size="10">band holds both blues → 4 of 4</text></g></svg>
%%%

**What the two-ropes picture gets right:** joining simple straight pieces really can fence off a shape neither piece could handle alone — that's the whole reason we stack neurons into a hidden layer. **Where it breaks down:** with real ropes *you* tie the knot; in a network the "knot" is a tiny math step called the activation (tomorrow's whole lesson), and the network finds how to arrange the lines by training (Day 5). Today, just hold the headline: *one neuron = one line; a hidden layer of neurons = a boundary that can bend.*

#### See it live
Flip on the **"add a hidden layer"** toggle back in the XOR demo above and watch the second line appear and the four chairs finally split cleanly — the limit, outgrown, right in front of you. **You now know a single neuron, and how neurons join into layers and networks to outgrow its limit.** That door — stacking, and the bend that makes stacking worth it — is exactly where Day 2 begins.

@@@ concept id=c12 tag="What it powers" title="Where these little neurons grow up" gotit="Saw the payoff"
Before we wrap, let's zoom *all* the way out — because the tiny weigh → add → decide you just learned is the exact seed inside the tech you use every day. Picture a single **LEGO brick**. On its own it's almost nothing — a bump of plastic. But snap a few thousand together and you can build a whole castle. One neuron is that brick; the things below are the castles people have already built from it.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Six labelled tiles showing real-world uses of neural networks: image recognition (a camera), speech recognition (a microphone), language translation (a speech bubble), game playing (a game controller), recommendation systems (a thumbs-up), and self-driving cars (a car). At the center, one small neuron brick feeds all six."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">One neuron (the brick) → the things people build with millions of them</text><rect x="30" y="34" width="140" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="52" font-size="15">📸</text><text x="100" y="70" fill="#276b45">image recognition</text><rect x="190" y="34" width="140" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="260" y="52" font-size="15">🗣️</text><text x="260" y="70" fill="#5E5191">speech recognition</text><rect x="350" y="34" width="140" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="420" y="52" font-size="15">🌐</text><text x="420" y="70" fill="#8A6D3B">language translation</text><rect x="70" y="130" width="130" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="135" y="148" font-size="15">🎮</text><text x="135" y="166" fill="#8A6D3B">game playing</text><rect x="320" y="130" width="130" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="385" y="148" font-size="15">🚗</text><text x="385" y="166" fill="#276b45">self-driving cars</text><rect x="205" y="96" width="110" height="34" rx="6" fill="#FDF9F3" stroke="#B8AEA2"/><text x="260" y="110" font-size="13">🧱</text><text x="260" y="125" fill="#6B645E">one neuron</text><text x="135" y="192" fill="#5E5191" font-size="10">👍 recommendation systems (like/skip)</text><text x="385" y="192" fill="#9A938A" font-size="9">same brick, different castle</text></g></svg>
%%%

**What the LEGO-brick picture gets right:** the same simple unit, repeated and connected in huge numbers, builds wildly different things — that's genuinely how one neuron scales into all of these. **Where it breaks down:** LEGO bricks are snapped together *by hand* into a fixed shape; a network isn't hand-assembled — it *tunes itself* to the job by training (the very next idea, and Day 5).

Here's the tour — six real jobs, one line each, all built from stacks of the neuron you now understand:
- **📸 Image recognition** — your phone unlocking at your face: early layers spot edges, later layers combine them into eyes, then whole faces.
- **🗣️ Speech recognition** — your voice assistant turning sound into words: layers pick out little sound-pieces, then syllables, then words.
- **🌐 Language translation** — an app turning English into French: the network reads the meaning of one sentence and writes it out in another language.
- **🎮 Game playing** — the AI that beat the world's best Go players: the network looks at the board and scores which move is most likely to win.
- **👍 Recommendation systems** — the "you might also like" on a streaming app: the network weighs what you've watched to guess what you'll enjoy next.
- **🚗 Self-driving cars** — a car reading the road: networks find lanes, cars, and pedestrians in the camera feed and decide when to brake.

#### How does it *get good* at any of this? (a peek at learning)
Every one of those castles starts out **useless** — a fresh network's weights are just random numbers, so its first guesses are basically coin-flips. It gets good through a simple loop you can say in one breath. Think of learning to bake: taste a too-sweet cake, use less sugar next time, taste again — repeat until it's right.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A four-step learning loop drawn as a cycle. Step one: guess with the current weights. Step two: measure the error, how wrong the guess was. Step three: adjust the weights a little to reduce the error. Step four: repeat. An arrow loops from step four back to step one."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">How a network learns: guess → check error → adjust → repeat</text><rect x="30" y="55" width="90" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="75" y="74" fill="#5E5191">1 · guess</text><text x="75" y="90" fill="#9A938A" font-size="9">(current weights)</text><line x1="120" y1="77" x2="160" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="160,72 170,77 160,82" fill="#B8AEA2"/><rect x="172" y="55" width="96" height="44" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="220" y="74" fill="#C93B3B">2 · check error</text><text x="220" y="90" fill="#9A938A" font-size="9">how wrong?</text><line x1="268" y1="77" x2="308" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="308,72 318,77 308,82" fill="#B8AEA2"/><rect x="320" y="55" width="96" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="368" y="74" fill="#276b45">3 · adjust</text><text x="368" y="90" fill="#9A938A" font-size="9">nudge weights</text><line x1="416" y1="77" x2="452" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="452,72 462,77 452,82" fill="#B8AEA2"/><rect x="464" y="55" width="46" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="487" y="80" fill="#8A6D3B">4 · repeat</text><path d="M487 99 Q 487 130 260 130 Q 75 130 75 101" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="4,3"/><polygon points="70,105 75,95 80,105" fill="#C99A12"/><text x="260" y="145" fill="#9A938A" font-size="9">do it thousands of times and the guesses get good — that's training</text></g></svg>
%%%

The loop has four beats: **(1) guess** with the weights it has now, **(2) check the error** — how far off the guess was from the right answer, **(3) adjust** the weights a little to shrink that error, and **(4) repeat**, thousands of times, until the guesses get good. In one sentence: **training is the search for the weights and biases that make the network wrong as little as possible.** That's the whole idea — and the machinery that does each step has its own days ahead:
- *How wrong was the guess?* is measured by a **loss function** — **Day 4**.
- *Which way to nudge each weight?* is worked out by **gradient descent** and **backpropagation** — **Day 5** and beyond.

Today you built the thing that does the guessing. **Next you'll sharpen its "decide" step, then learn how it teaches itself.** That's the arc of this whole module — and you're already standing at the start of it.

@@@ concept id=c13 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — that was a real journey. Here's a fresh way to hold it: picture yourself at the end of a hike, laying out the **postcards you collected at each stop** along the trail. You didn't just walk; at every viewpoint you picked up one small postcard, and now, spread on the table, they tell the whole story of the climb in order — start, middle, summit. Today had stops just like that, each with its own little picture. Flip through them once and the whole day snaps back into place. **Where the postcard picture breaks down:** postcards are souvenirs you look at and put away, but these ideas are *working parts* — you'll pick each one back up and build on it in the days ahead, so they're less "trip mementos" and more "tools you'll keep using."

The postcards from today's trail, in order:
- **Stop 1 · The brain cell.** A neuron is a tiny copy of a brain cell: it does **weigh → add → decide**, and that's the whole machine.
- **Stop 2 · Inputs & weights.** The raw numbers you feed in are the **inputs**. Each input gets one **weight** — a trust dial: big = "listen," near-zero = "ignore," negative = "count against."
- **Stop 3 · Bias.** One extra number, the neuron's starting mood — a head start toward "yes," or a handicap.
- **Stop 4 · The weighted sum z.** Weigh every input, add them up, add the bias → one number `z` (the pre-activation).
- **Stop 5 · The activation (decide).** A step turns `z` into the answer. You met the hard **step** (a cliff, can't learn from tiny nudges) and the smooth **sigmoid** (a dimmer, 0→1, so it can). A neuron is always *linear part, then activation.*
- **Stop 6 · The line, and its limit.** Weights + bias draw **one straight line** to split inputs. That's the ceiling: one line can't do **XOR**.
- **Stop 7 · Layers, networks, and depth.** A **layer** is a row of neurons on the same inputs; chain layers into a **network** (**input layer** → **hidden layer** → **output layer**), and the lines combine and bend — a **hidden layer** solves XOR. Stack many hidden layers and it's called **deep** (that's **deep learning**).
- **Stop 8 · The payoff.** Millions of these bricks power image and speech recognition, translation, game playing, recommendations, and self-driving cars — and they get good by a **guess → check error → adjust → repeat** loop (training, Days 4–5).

Here's the one picture to keep — the whole neuron on one line, and where it grows into a network.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="The complete neuron on one line: two inputs each times a weight, summed with a bias into z, then an activation turns z into the output. Below, a note that weights and bias draw one straight line, which cannot do XOR alone, and that stacking neurons into a hidden layer bends the boundary."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The whole neuron: weigh → add → decide</text><circle cx="40" cy="55" r="7" fill="#EFEAF7" stroke="#7C6DAA"/><text x="40" y="59" text-anchor="middle" fill="#5E5191" font-size="9">x₁</text><circle cx="40" cy="100" r="7" fill="#EFEAF7" stroke="#7C6DAA"/><text x="40" y="104" text-anchor="middle" fill="#5E5191" font-size="9">x₂</text><line x1="47" y1="57" x2="150" y2="72" stroke="#B8AEA2" stroke-width="1.5"/><text x="95" y="58" fill="#6B645E" font-size="9">×w₁</text><line x1="47" y1="98" x2="150" y2="80" stroke="#B8AEA2" stroke-width="1.5"/><text x="95" y="105" fill="#6B645E" font-size="9">×w₂</text><rect x="150" y="58" width="80" height="36" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="190" y="80" text-anchor="middle" fill="#5E5191" font-size="10">Σ + b → z</text><line x1="230" y1="76" x2="290" y2="76" stroke="#B8AEA2" stroke-width="2"/><polygon points="290,70 302,76 290,82" fill="#B8AEA2"/><rect x="305" y="58" width="90" height="36" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="350" y="80" text-anchor="middle" fill="#276b45" font-size="10">activation</text><line x1="395" y1="76" x2="450" y2="76" stroke="#B8AEA2" stroke-width="2"/><polygon points="450,70 462,76 450,82" fill="#B8AEA2"/><rect x="465" y="58" width="45" height="36" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="487" y="80" text-anchor="middle" fill="#8A6D3B" font-size="10">out</text><text x="260" y="132" text-anchor="middle" fill="#6B645E" font-size="10">weights + bias draw ONE straight line…</text><text x="260" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">…can't do XOR alone → stack into a hidden layer to bend it</text><text x="260" y="168" text-anchor="middle" fill="#9A938A" font-size="9">many hidden layers = "deep" (deep learning)</text></g></svg>
%%%

#### Three myths to leave behind
You met these along the way — here they are in one spot so they stick:
- **"Neural nets think like a human brain."** No — a real brain cell is a living thing; an artificial neuron is only multiply-and-add. *Inspired by*, not the same as.
- **"They're magic black boxes."** No — every answer is *assembled* from plain steps (weigh, add, decide), exactly like the ones you ran today. What they're genuinely great at is **finding patterns** in data.
- **"Bigger is always better" / "they can solve any problem."** No — one neuron provably can't even do XOR, and a network too big for its data just memorizes. The right *size* and *shape* for the job beats raw size.

#### Cheat-sheet · the neuron's parts at a glance
%%% table
:: Part :: Everyday picture :: What it does
Input :: a raw number you hand in :: the value the neuron reads
Weight :: a trust dial (one per input) :: scales how much an input matters
Bias :: a starting mood / head start :: shifts where the neuron leans before any input
Weighted sum z :: a shopping receipt total :: every input × its weight, added, plus bias
Activation :: a judge's verdict :: turns z into the neuron's answer (step, sigmoid, …)
Decision line :: one chalk line on a playground :: the straight boundary weights + bias draw
The limit :: one rope, four chairs :: one line can't do XOR
Layer & network :: a panel of judges, chained :: many neurons combine lines into a bent boundary
%%%

#### Cheat-sheet · the words you met today
%%% jargon
neuron | a tiny math function that weighs its inputs, adds them up, and decides
input | a raw number handed to the neuron; the input layer is all the raw inputs together
weight | one number per input — how much that input matters (negative = counts against)
bias | one extra number added to the sum — the neuron's starting lean toward "yes"
weighted sum (z) | every input times its weight, summed, plus the bias — the pre-activation
activation function | the last step that turns z into the neuron's output
step function | the first activation — a hard on/off switch (fires 1 above zero, else 0)
sigmoid | a smooth S-curve that squashes z into (0, 1) — the "how confident" answer
decision boundary | the one straight line a neuron draws to split its inputs into two sides
forward propagation | pushing inputs left-to-right through the neuron to get the output
XOR | "on when exactly one input is on" — the pattern one straight line can't split
linearly separable | when one straight line can cleanly split every class; XOR is not
layer | a row of neurons all reading the same inputs
hidden layer | a middle layer, between inputs and answer — where the boundary bends
output layer | the last layer, whose result is the network's final answer
neural network | layers of neurons chained so each layer feeds the next
deep learning | training a network with many hidden layers stacked ("deep" = many layers)
training | slowly adjusting the weights and bias so the network makes fewer mistakes (Day 5)
%%%

!!! c-info 📚
<b>Want more before Day 2? (all optional)</b> Nothing here is required. If today lit a spark, the friendliest next read is the free online book <b>"Neural Networks and Deep Learning"</b> by Michael Nielsen (chapter 1 covers this exact neuron), or 3Blue1Brown's short video series <b>"Neural Networks"</b> on YouTube for a beautiful visual tour. Skip them with a clear conscience — Day 2 assumes only what you learned right here.
!!!

That's the whole day. Next you'll zoom in on that "decide" step — the **activation** — and see the little bend that lets neurons stack into something genuinely deep.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a nudge to scroll back, not a failing grade.)
%%% quiz
q: What are the three beats a single neuron always does, in order? | a:2 | add → weigh → decide | decide → add → weigh | weigh → add → decide | weigh → decide → add | fb: A neuron scales each input by its weight (weigh), sums them with the bias (add), then turns that total into an answer (decide).
q: A neuron has two inputs. How many weights and how many biases does it have? | a:1 | 1 weight and 2 biases | 2 weights and 1 bias | 2 weights and 2 biases | 1 weight and 1 bias | fb: One weight per input (so 2), and exactly one bias for the whole neuron — the bias is a single starting-lean number, not one per input.
q: Why did people move from the hard step function to a smooth activation like sigmoid? | a:2 | The step used too much memory | The step outputs negative numbers | The step is a cliff, so a tiny nudge to the weights changes the output by nothing (or flips it entirely) — a smooth curve lets a small nudge make a small, followable change, which is what learning needs | Sigmoid is faster to compute | fb: Learning works by nudging the dials and watching the answer change a little. A step gives no in-between, so it hands back no useful signal; sigmoid's smooth ramp does.
q: One neuron can't get all four XOR points right. What is the fix, and what do we call a network with many such stacked layers? | a:1 | Raise the learning rate | Stack neurons into a hidden layer so their lines combine into a bent boundary; a network with many hidden layers is called "deep" (deep learning) | Give the neuron more weights | Use a negative bias | fb: One neuron = one straight line, and XOR is not linearly separable. A hidden layer draws several lines that combine and bend to split all four. Stacking many hidden layers is what "deep" learning means.
%%%

@@@ produce id=produce tag="Produce" title="Build a neuron and watch it decide" gotit="Done"
Time to see today with your own eyes. You'll code one neuron — weights, bias, weighted sum, activation — run a full forward pass, and then watch it *fail* on XOR so the limit is real, not just a story. **Predict first:** for inputs `x = [2, 3]`, weights `w = [0.5, −1.0]`, and bias `b = 1.0`, what's `z`? And will `sigmoid(z)` land above or below `0.5`? Then run it and **watch.** Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-01-single-neuron/experiment.py`. Write `sigmoid(z)` and `step(z)`, then a `neuron(x, w, b)` that returns `sigmoid((w*x).sum() + b)`. **Notice** that for `x=[2,3]`, `w=[0.5,−1.0]`, `b=1.0` you get `z = −1.0` and `sigmoid(z) ≈ 0.269` — a soft "probably no." Print `z` and the output so you can see both. Then set up the four XOR points `[0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0` and try to find a single `w, b` that a step-neuron gets all four right — **observe** that you can never beat 3 of 4, because one neuron draws one straight line. Run with `python3 sessions/m02-the-neuron/day-01-single-neuron/experiment.py`.

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
