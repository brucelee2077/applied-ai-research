---
quest_id: wf8-d04-batchnorm
mode: concept
donor: v9-base.donor
page_title: "Module 6 · Day 4 — Batch Normalization"
module_label: "Module 06 · Represent · Day 4"
title: "Batch Normalization"
subtitle: "Keeping Each Layer's Signal in a Steady Range"
brand_sub: "Foundations · M6 Day 4"
spine: "thermostat"
nav_prev_href: "../day-03-cnn-classifier/lesson.html"
nav_prev_label: "A Small CNN Classifier"
nav_next_href: "../day-05-data-augmentation/lesson.html"
nav_next_label: "Data Augmentation"
fin_title: "Module 6 · Day 4 complete! 🏆"
fin_body: "Nice — you've met <b>Batch Normalization</b>: a <b>thermostat</b> inside the network that re-centers each layer's signal to a steady range every batch, then lets the model scale and shift it back as needed. That one habit lets you train deeper networks faster and with far less fuss — it's in almost every modern vision model.<br>Next up: <b>Data Augmentation</b>."
notebook_yardstick: null
---

@@@ hero
@lede Have you ever walked from a warm room into a freezing hallway and back, over and over, and felt your body never quite settle? Now picture a **thermostat** on the wall. It watches the temperature, and whenever the room drifts too hot or too cold, it gently nudges it back toward a comfortable middle. You barely notice — the room just *stays* pleasant, so you can get on with your day. Yesterday you built a whole CNN and even met the trap where a deep stack "stops learning" because the signal fades on the way down. Today you meet the fix that trap was waiting for: a tiny **thermostat** you drop *inside* the network, between its layers. Every batch of pictures, it checks whether the numbers flowing through are drifting too hot or too cold, and gently re-centers them to a steady, comfortable range — so every layer keeps getting a signal it can actually learn from. This one habit, called **batch normalization**, is the reason we can train deep networks fast and without fuss — it lives inside almost every modern vision model you've heard of.
@goal Together we'll build the thermostat one dial at a time. You'll see the drift problem it solves, the exact re-centering step (subtract the batch's average, divide by its spread), the tiny safety number that stops a divide-by-zero, the two learnable knobs that hand the network its freedom back, where the layer sits in the stack, how it quietly remembers a running average so a single test picture gets a stable answer, and — as friendly puzzles — the three ways it can trip you up, each with its fix. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="The drift problem" title="Why a deep stack drifts: the signal wanders too hot or too cold" gotit="Got the drift"
Let's start with the problem the thermostat is here to fix — because once you *feel* the problem, the fix will make instant sense.

Picture a long line of people playing the **telephone game**: the first person whispers a message, the next repeats what they heard, and so on down the line. Tiny changes creep in at every step. By the end, "let's meet at nine" has drifted into "let's eat some wine." Nobody meant to change it — small shifts just pile up over a long chain. The numbers flowing through a deep network do the same thing. Each layer multiplies, adds, and bends its input a little, so by the time the signal reaches a deep layer, its typical size can have drifted far too **big** (everything huge) or far too **small** (everything near zero). This wandering of the signal's scale as it passes through many layers is the drift problem — historically people called it [[internal covariate shift||A fancy old name for a simple idea: as signals pass through many layers of a network, the typical scale and center of the numbers keeps shifting during training, which makes deep networks slow and touchy to train.]].

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A telephone-game analogy for signal drift. Five people stand in a line, each whispering to the next. Speech bubbles show the message drifting: let's meet at nine, then meet at nine, then eat at nine, then eat some wine. A caption says small shifts pile up over a long chain, just like a signal wandering through deep layers."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Telephone game: small shifts pile up down a long chain</text><g fill="#C99A12"><circle cx="60" cy="80" r="16"/><circle cx="160" cy="80" r="16"/><circle cx="260" cy="80" r="16"/><circle cx="360" cy="80" r="16"/><circle cx="460" cy="80" r="16"/></g><g stroke="#B8AEA2"><line x1="76" y1="80" x2="144" y2="80"/><line x1="176" y1="80" x2="244" y2="80"/><line x1="276" y1="80" x2="344" y2="80"/><line x1="376" y1="80" x2="444" y2="80"/></g><g font-size="7.5" fill="#6B645E"><text x="60" y="118">"meet at nine"</text><text x="160" y="118">"meet at nine"</text><text x="260" y="118">"eat at nine"</text><text x="360" y="118">"eat some wine"</text><text x="460" y="118">"eat some wine?"</text></g><g fill="#6B645E" font-size="8"><text x="60" y="140">layer 1</text><text x="160" y="140">layer 2</text><text x="260" y="140">layer 3</text><text x="360" y="140">layer 4</text><text x="460" y="140">layer 5</text></g><text x="260" y="164" fill="#8a3b3b" font-size="9">the signal's scale drifts the same way through deep layers</text></g></svg>
%%%

**What the telephone-game picture gets right:** each step makes only a small change, but over a long chain those small changes pile up into a big drift, and no single step is "to blame." **Where it breaks down:** in telephone the *meaning* changes; in a network it's the *scale* of the numbers (too big or too tiny) that drifts, and — unlike a party game — the network can actively fix it, which is the whole point of today.

#### Why drift makes deep networks slow and touchy
This drift isn't just untidy — it makes training genuinely painful. Watch what happens to a signal that starts at a healthy `1.0` but gets multiplied by about `1.6` at each of five layers:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A growth ladder showing a signal drifting too hot. Five bars rise left to right: 1.0, then 1.6, then 2.6, then 4.1, then 6.6, each about 1.6 times the one before. The bars grow tall. A caption says multiply by a bigger-than-one factor each layer and the signal blows up hot; the opposite fades it cold."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Signal × 1.6 each layer → it drifts too "hot" (blows up)</text><line x1="45" y1="150" x2="480" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><rect x="60" y="130" width="55" height="20" fill="#2D8B55"/><text x="87" y="122" fill="#276b45">1.0</text><text x="87" y="166" fill="#6B645E" font-size="8">layer 1</text><rect x="150" y="118" width="55" height="32" fill="#C99A12"/><text x="177" y="110" fill="#8A6D3B">1.6</text><text x="177" y="166" fill="#6B645E" font-size="8">layer 2</text><rect x="240" y="98" width="55" height="52" fill="#C99A12"/><text x="267" y="90" fill="#8A6D3B">2.6</text><text x="267" y="166" fill="#6B645E" font-size="8">layer 3</text><rect x="330" y="68" width="55" height="82" fill="#C93B3B" opacity="0.8"/><text x="357" y="60" fill="#8a3b3b">4.1</text><text x="357" y="166" fill="#6B645E" font-size="8">layer 4</text><rect x="420" y="28" width="55" height="122" fill="#C93B3B"/><text x="447" y="20" fill="#8a3b3b">6.6</text><text x="447" y="166" fill="#6B645E" font-size="8">layer 5</text></g></svg>
%%%

Read the ladder: `1.0 → 1.6 → 2.6 → 4.1 → 6.6` — the signal keeps getting hotter. (Flip the factor to `0.6` and it fades toward zero instead — the "too cold" case, which is exactly yesterday's vanishing-signal trap.) When the scale drifts like this, the learning steps have to be *tiny* and *careful* or training blows up, and you must fuss over exactly how the weights were first set. **That's the whole ache batch normalization removes.** The fix is delightfully simple: put a little thermostat between the layers that re-centers the signal every time — which is the very next concept.

@@@ concept id=c2 tag="The thermostat" title="The core move: re-center the signal to a steady range" gotit="Got the thermostat"
Here's the heart of the whole day. To stop the signal from drifting hot or cold, we do the most natural thing in the world: measure where it is right now, and pull it back to a comfortable middle. That's the **thermostat** move.

Picture the **thermostat** on the wall again. It does two simple things. First it *reads* the room's temperature — how far from comfortable it is. Then it *nudges* the room back toward a steady target, say 20°C, so the room neither bakes nor freezes. A [[batch normalization||A layer dropped inside a network that re-centers the numbers flowing through it: it subtracts their average and divides by their spread, so each layer's signal stays in a steady range. Nicknamed "batch norm" or "BN".]] layer — batch norm for short — is exactly this thermostat for the numbers flowing through the network. It reads the signal, then re-centers it so its average sits at `0` and its spread is about `1`. Neither too hot (huge) nor too cold (near zero) — just a steady, comfortable range every layer can learn from.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A thermostat analogy for batch normalization. A round wall thermostat in the middle with a dial pointing to a comfortable target. On the left, incoming numbers are scattered too hot and too cold. An arrow through the thermostat leads to the right, where the same numbers are re-centered around a comfortable middle. A caption says read the signal, then nudge it back to a steady range, average 0, spread 1."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Batch norm = a thermostat: read the signal, nudge it to a steady range</text><text x="80" y="42" fill="#8a3b3b">drifting numbers</text><g transform="translate(40,52)" font-size="10"><text x="10" y="10" fill="#C93B3B">98</text><text x="55" y="24" fill="#276b45">2</text><text x="20" y="50" fill="#C93B3B">-77</text><text x="60" y="66" fill="#C99A12">41</text><text x="8" y="80" fill="#276b45">0.3</text></g><g transform="translate(200,60)"><circle cx="60" cy="55" r="46" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><circle cx="60" cy="55" r="34" fill="#FDF9EF" stroke="#C99A12"/><line x1="60" y1="55" x2="60" y2="26" stroke="#C93B3B" stroke-width="2.5"/><text x="60" y="115" fill="#8A6D3B" font-size="8">target: 0, spread 1</text></g><text x="330" y="65" fill="#B8AEA2" font-size="14">→</text><text x="440" y="42" fill="#276b45">steady numbers</text><g transform="translate(395,55)" font-size="10"><text x="8" y="12" fill="#276b45">1.2</text><text x="52" y="24" fill="#276b45">-0.4</text><text x="14" y="46" fill="#276b45">-1.0</text><text x="54" y="60" fill="#276b45">0.6</text><text x="10" y="78" fill="#276b45">-0.3</text></g><text x="440" y="150" fill="#276b45" font-size="8">avg ≈ 0, spread ≈ 1</text></g></svg>
%%%

**What the thermostat picture gets right:** it measures the current state and then actively pulls it back toward a fixed, comfortable target — it doesn't just hope things stay nice, it *corrects* them. **Where it breaks down:** a home thermostat targets one number (temperature); batch norm targets *two* at once — it fixes both the *center* (average → 0) and the *spread* (→ 1) of a whole cloud of numbers.

#### From words to the two-step move
The thermostat move is just two small steps, and it's worth seeing them plainly. Batch norm looks at a whole batch of numbers for one feature and does this:

1. **Subtract the average.** Find the average of the batch, then subtract it from every number. Now the numbers are *centered* — their new average is `0`. (This kills the "too hot on one side" drift.)
2. **Divide by the spread.** Find the [[standard deviation||A single number that measures how spread out a set of values is: small when they huddle together, large when they're scattered wide.]] — how spread out the numbers are — and divide every number by it. Now the *spread* is about `1`. (This kills the "way too big" or "way too tiny" drift.)

Let's watch it happen on five real numbers. **Predict first:** the raw values are `[2, 4, 4, 4, 6]` — after re-centering, what should their new average be? Then run it:

%%% demo id=normalize label="re-center five numbers"
code: normalize([2, 4, 4, 4, 6])   # subtract avg, divide by spread
out: avg = 4.0,  spread = 1.26  →  [-1.58, 0.0, 0.0, 0.0, 1.58]   (new avg = 0, new spread = 1)
take: <b>Batch norm = subtract the average, divide by the spread.</b> The middle value 4 (the average) becomes exactly 0, and the values that were far out (2 and 6) land at about ±1.58. Whatever came in, it leaves centered at 0 with a spread near 1 — a steady range.
%%%

**Notice:** the average `4` slid to `0`, and the whole cloud got squeezed to a spread of `1`. That single move — done fresh at every layer, every batch — is what keeps the signal from ever wandering hot or cold. But *which* numbers does batch norm average over? That subtle choice is the next concept.

@@@ concept id=c3 tag="Across the batch" title="Whose average? The whole batch, one feature at a time" gotit="Got the batch stats"
The thermostat needs to read a temperature before it can nudge — but *whose* temperature? Batch norm makes a very specific, clever choice here, and it's the choice hiding in the *name*.

Imagine a teacher grading a **class of 32 students on one test**. To fairly curve the scores, the teacher doesn't look at a single student — one score tells you nothing about "normal." Instead they gather *all 32* scores for that one test, find the class average, and see how spread out they are. Now they can say "this score is above average, that one below." Batch norm does the same: to normalize one feature, it gathers that feature's value across the **whole batch of pictures** in front of it right now, and computes the average and spread from that group. These per-group numbers are the [[mini-batch statistics||The average and spread of a feature, measured across all the examples in the current batch of data, not from a single example. "Batch" in batch norm refers to exactly this.]] — and this is why it's called *batch* norm.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A class-of-students analogy for mini-batch statistics. A grid shows 8 students' scores for one test as a column of numbers. A bracket around the whole column is labelled the batch. Arrows point to two computed numbers below: the class average and the class spread. A caption says gather one feature across the whole batch, then compute its average and spread; one student alone tells you nothing."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Batch norm reads one feature across the WHOLE batch of examples</text><g transform="translate(120,34)"><rect x="0" y="0" width="120" height="128" fill="#FCF3DC" stroke="#C99A12"/><text x="60" y="-6" fill="#8A6D3B" font-size="8">one feature, 8 examples</text><g fill="#8A6D3B"><text x="60" y="20">example 1 → 62</text><text x="60" y="40">example 2 → 55</text><text x="60" y="60">example 3 → 71</text><text x="60" y="80">example 4 → 48</text><text x="60" y="100">example 5 → 66</text><text x="60" y="120">example 6 → 58</text></g><path d="M-8 0 V128" stroke="#5E5191" stroke-width="2"/><text x="-34" y="64" fill="#5E5191" font-size="8" transform="rotate(-90 -34 64)">the batch</text></g><text x="300" y="70" fill="#B8AEA2" font-size="13">→</text><g transform="translate(330,50)"><rect x="0" y="0" width="150" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="75" y="19" fill="#276b45">class average ≈ 60</text><rect x="0" y="40" width="150" height="30" fill="#EAF0FA" stroke="#5E5191"/><text x="75" y="59" fill="#5E5191">class spread ≈ 8</text></g><text x="260" y="186" fill="#6B645E" font-size="8">one example alone tells you nothing — you need the group</text></g></svg>
%%%

**What the classroom picture gets right:** you can only tell whether a score is high or low by comparing it to the *group's* average and spread — a lone number has no meaning without the crowd. **Where it breaks down:** a teacher curves once at the end; batch norm re-reads a *fresh* batch and re-computes these numbers on *every single training step*, so the "class" keeps changing.

#### One thermostat per channel
For a convolution's feature maps (from Days 1–3), batch norm keeps *one thermostat per [[channel||One of the stacked grids in a feature map — one filter's found-it map. Batch norm treats each channel as its own feature, with its own average and spread.]]*. Each channel is one filter's found-it map, so it gets its own average and spread, pooled across every picture in the batch *and* every position in that channel's grid. Channel 1 (say, "edge detector") gets one thermostat; channel 2 ("blob detector") gets its own. They don't share a dial — each pattern is kept steady on its own terms.

#### The tiny safety number that prevents a crash
There's one small catch in "divide by the spread." What if a channel is nearly flat — every value almost the same — so its spread is *almost zero*? Dividing by (almost) zero explodes into a giant or undefined number, and training crashes. The fix is a tiny safety cushion: add a very small number — called [[epsilon||A tiny fixed number (like 0.00001) added inside the square root when dividing by the spread, so you never divide by zero when a channel's spread is almost nothing.]] (written `ε`, and typically about `0.00001`) — inside the divide, so the denominator can never be exactly zero. Watch the difference it makes when the spread is basically nothing:

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A before-and-after of the epsilon safety number. On the left, labelled without epsilon, dividing a value by a spread of 0.0000 gives an explosion, drawn as a huge red bar shooting off the top marked crash. On the right, labelled with epsilon, dividing by 0.0000 plus a tiny epsilon gives a small safe number, drawn as a short green bar marked safe. A caption says epsilon stops the divide-by-zero."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">A nearly-flat channel: divide by spread ≈ 0 → epsilon saves you</text><line x1="55" y1="140" x2="465" y2="140" stroke="#E5DFD6" stroke-width="1.5"/><text x="150" y="36" fill="#8a3b3b">without ε: 3 ÷ 0.0000</text><rect x="120" y="28" width="60" height="112" fill="#C93B3B"/><text x="150" y="24" fill="#8a3b3b" font-size="14">↑</text><text x="150" y="158" fill="#8a3b3b" font-size="8">explodes → crash ✗</text><text x="360" y="36" fill="#276b45">with ε: 3 ÷ √(0 + ε)</text><rect x="330" y="96" width="60" height="44" fill="#2D8B55"/><text x="360" y="90" fill="#276b45">≈ 949</text><text x="360" y="158" fill="#276b45" font-size="8">big but finite → safe ✓</text></g></svg>
%%%

**What you just saw:** without `ε` the divide blows up; with the tiny `ε` cushion the answer stays finite and training keeps going. It costs almost nothing and it's *always* there in a real batch norm layer. **You now have the full read-and-nudge move** — but pure "average 0, spread 1" is actually a little *too* strict. The next concept gives the network its freedom back.

@@@ concept id=c4 tag="Two free knobs" title="Scale and shift: handing the network its freedom back" gotit="Got scale & shift"
Forcing every layer to always sit at "average 0, spread 1" is a bit like a thermostat that *locks* every room to exactly 20°C — comfortable for most, but what about the sauna that *needs* to be hot, or the fridge that *needs* to be cold? Sometimes a layer genuinely wants a different center or a different spread. So batch norm adds two little knobs the network can turn for itself.

Picture that wall thermostat again, but now with **two extra dials** the network is allowed to spin. One dial is a **volume knob** — it multiplies the steadied signal to make it louder (bigger spread) or quieter (smaller spread). The other is a **slider** — it shifts the whole thing up or down to re-center it wherever the layer likes. After batch norm steadies the signal to 0-and-1, these two dials let the network scale and shift it back to whatever range actually helps. The volume knob is called [[gamma||A learnable scale knob (written γ) that multiplies the normalized signal. It lets the network choose how wide the spread should be — even wider than 1 if that helps.]] (`γ`, the scale) and the slider is called [[beta||A learnable shift knob (written β) that is added to the scaled signal. It lets the network choose where the center should sit — not just 0.]] (`β`, the shift).

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A two-dials analogy for the learnable scale and shift. A steady signal centered at 0 with spread 1 enters on the left. It passes a volume knob labelled gamma (scale) which widens or narrows the spread, then a slider labelled beta (shift) which moves the center up or down. On the right the signal comes out at a new center and new spread of the network's choosing. A caption says gamma sets the spread, beta sets the center; the network learns both."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">After steadying, two learnable dials give the range back: γ (scale), β (shift)</text><g transform="translate(30,60)"><rect x="0" y="0" width="80" height="60" fill="#FDF9EF" stroke="#C99A12"/><text x="40" y="28" fill="#8A6D3B" font-size="8">steady in</text><text x="40" y="44" fill="#276b45" font-size="8">avg 0, spread 1</text></g><text x="128" y="94" fill="#B8AEA2">→</text><g transform="translate(150,54)"><circle cx="45" cy="40" r="34" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><line x1="45" y1="40" x2="45" y2="14" stroke="#2D8B55" stroke-width="2.5"/><text x="45" y="94" fill="#276b45" font-size="8">γ · volume (spread)</text></g><text x="252" y="94" fill="#B8AEA2">→</text><g transform="translate(275,60)"><rect x="0" y="20" width="90" height="8" fill="#EAF0FA" stroke="#5E5191"/><rect x="56" y="12" width="10" height="24" fill="#5E5191"/><text x="45" y="60" fill="#5E5191" font-size="8">β · slider (center)</text></g><text x="382" y="94" fill="#B8AEA2">→</text><g transform="translate(405,60)"><rect x="0" y="0" width="90" height="60" fill="#FCF3DC" stroke="#C99A12"/><text x="45" y="28" fill="#8A6D3B" font-size="8">out: any center,</text><text x="45" y="44" fill="#8A6D3B" font-size="8">any spread it wants</text></g></g></svg>
%%%

**What the two-dials picture gets right:** the steady 0-and-1 signal is just a *starting point*, and the two knobs let the network re-shape it to whatever range it prefers — including leaving it steady. **Where it breaks down:** real dials are set by a person's hand; `γ` and `β` are set by *learning* — the training loop from yesterday nudges them along with every other weight, so the network discovers its own best center and spread.

#### Why the two knobs matter so much
Here's the clever part: because the network can learn `γ` and `β` freely, it *never loses* any option it had before. In particular, if a layer decides normalizing was a bad idea, it can learn `γ` = the old spread and `β` = the old average and **completely undo** the normalization. So batch norm never *takes away* freedom — it only *adds* a helpful starting point plus two knobs. That's why "too rigid" is never a real worry.

The whole layer is now three moves in a row — steady it, then scale, then shift — and here's the one-line rule in plain words: take the steadied value, multiply by the scale `γ`, and add the shift `β`.

!!! c-info 📎 Optional (skippable)
For one value `x` in a channel with batch average `μ` and batch spread `σ`, the full batch norm output is `y = γ · (x − μ) / √(σ² + ε) + β`. The `(x − μ) / √(σ² + ε)` part is the thermostat (steady to 0-and-1 with the `ε` safety cushion); the `γ ·` and `+ β` are the two learnable knobs. You never compute this by hand — the library does — but every symbol is something you already met: `μ` and `σ` are the batch's average and spread, `ε` is the safety number, `γ` and `β` are the scale and shift.
!!!

Watch the two knobs re-shape a steadied signal. **Predict:** a signal steady at spread `1`, center `0`, is passed through `γ = 3` and `β = 5`. What center and spread come out?

%%% demo id=scaleshift label="turn the two knobs"
code: scale_shift(signal_avg=0, signal_spread=1, gamma=3, beta=5)
out: new center = 0·3 + 5 = 5,   new spread = 1·3 = 3   →  steady-at-0-and-1 becomes centered-at-5-spread-3
take: <b>γ scales, β shifts.</b> The volume knob γ=3 tripled the spread (1→3); the slider β=5 moved the center from 0 to 5. The network learns these two numbers, so it can pick any range it wants — or set γ=old-spread and β=old-average to undo the whole thing.
%%%

**Notice:** `γ` set the new spread and `β` set the new center — exactly the two things the thermostat had fixed. **You now understand the whole batch norm layer:** steady, then scale, then shift. Next: where in the network does this little machine actually sit?

@@@ concept id=c5 tag="Where it sits" title="Placement: the thermostat goes between the layer and the bend" gotit="Got the placement"
A thermostat only helps if it's in the *right spot* — bolt it inside the oven and it's useless. So where in the stack does batch norm go? There's a natural home for it, and once you see it you'll spot it in every architecture diagram.

Think of a **water tap with a filter fitted right after it**, before the water reaches your glass. The tap pours out water that might be a bit gritty; the filter cleans it up *before* you drink. Order matters: filter-then-glass, not glass-then-filter. In a network, the "tap" is the convolution or dense layer (it pours out raw numbers), and the "glass" is the activation — the ReLU bend from yesterday that decides what passes. Batch norm is the filter you fit *between* them: it steadies the raw numbers right after the layer produces them, and right *before* the bend reads them. So the usual order is **layer → batch norm → activation**.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A tap-and-filter analogy for batch norm placement. Left to right: a tap labelled conv or dense layer pours raw numbers, then a filter labelled batch norm steadies them, then a glass labelled ReLU activation receives clean steady water. A caption says the thermostat sits between the layer and the bend: layer, then batch norm, then activation."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Batch norm sits BETWEEN the layer and the bend: layer → BN → activation</text><g transform="translate(40,50)"><rect x="0" y="0" width="110" height="60" fill="#FCF3DC" stroke="#C99A12"/><text x="55" y="26" fill="#8A6D3B">conv / dense</text><text x="55" y="44" fill="#8A6D3B" font-size="8">the "tap" (raw numbers)</text></g><text x="165" y="84" fill="#B8AEA2" font-size="13">→</text><g transform="translate(195,50)"><rect x="0" y="0" width="120" height="60" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="60" y="26" fill="#5E5191">batch norm</text><text x="60" y="44" fill="#5E5191" font-size="8">the "filter" (steady it)</text></g><text x="330" y="84" fill="#B8AEA2" font-size="13">→</text><g transform="translate(360,50)"><rect x="0" y="0" width="120" height="60" fill="#FDECEC" stroke="#C93B3B"/><text x="60" y="26" fill="#8a3b3b">ReLU activation</text><text x="60" y="44" fill="#8a3b3b" font-size="8">the "glass" (the bend)</text></g><text x="260" y="150" fill="#6B645E" font-size="8">clean, steady numbers reach the bend</text></g></svg>
%%%

**What the tap-and-filter picture gets right:** the filter cleans the water *right after* the tap and *before* the glass — the same in-between spot batch norm takes, and the order can't be swapped without losing the point. **Where it breaks down:** a water filter removes dirt (throws stuff away); batch norm removes *nothing* — it just re-centers the same numbers, and `γ`/`β` even let it put the scale back if the layer wants.

#### Slotting it into yesterday's block
Remember the repeating block from Day 3: `conv → ReLU → pool`. Batch norm slides right in after the conv, giving the modern block **conv → batch norm → ReLU → pool**. For conv feature maps it's applied per channel, exactly as we said — one thermostat per found-it map. Here's the upgraded block:

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="An upgraded convolutional block. Four stacked colored bars in order: conv (slide stencils), batch norm (steady per channel), ReLU (add the bend), pool (shrink). The batch norm bar is highlighted as the new insert between conv and ReLU. A caption says the modern block adds one thermostat right after the conv."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">The modern block: batch norm slots in right after the conv</text><g transform="translate(150,30)"><rect x="0" y="0" width="220" height="24" fill="#FCF3DC" stroke="#C99A12"/><text x="110" y="16" fill="#8A6D3B">conv  (slide learned stencils)</text><rect x="0" y="28" width="220" height="24" fill="#EAF0FA" stroke="#5E5191" stroke-width="2.5"/><text x="110" y="44" fill="#5E5191">batch norm  (steady per channel)  ← new</text><rect x="0" y="56" width="220" height="24" fill="#FDECEC" stroke="#C93B3B"/><text x="110" y="72" fill="#8a3b3b">ReLU  (add the bend)</text><rect x="0" y="84" width="220" height="24" fill="#EAF5EE" stroke="#2D8B55"/><text x="110" y="100" fill="#276b45">pool  (shrink)</text></g><text x="260" y="150" fill="#6B645E" font-size="8">one thermostat per channel, fitted between conv and the bend</text></g></svg>
%%%

**You now know exactly where the thermostat lives:** right after the conv (or dense) layer, before the bend, one per channel. That's the same spot in nearly every modern vision network. But so far the thermostat reads the *current batch* — what happens at test time, when you might have just **one** picture and no batch to average? That beautiful little wrinkle is next.

@@@ concept id=c6 tag="Train vs test" title="The running average: a memory for when there's no batch" gotit="Got train vs test"
Here's a puzzle the thermostat has to solve. During training it always sees a big batch of pictures, so it can compute an average and spread. But when your model is *finished* and someone shows it a single photo to classify, there's no batch — you can't take the "average of one picture." So what does the thermostat read? The answer is lovely: it keeps a *memory*.

Think of a shopkeeper working out the **"typical" number of customers per hour**. On any single day, they don't just use *today's* count — a rainy Tuesday would fool them. Instead they keep a **running average**: each day they blend the new count into a long-running estimate, mostly keeping the old memory and letting today nudge it a little. Over time this settles into a stable "typical" number they can trust even on a weird day. Batch norm does exactly this during training: alongside each batch's own average and spread, it quietly keeps a slowly-updated [[running statistics||A slowly-updated memory of the typical average and spread, blended a little from each training batch. At test time the layer uses these frozen numbers instead of a live batch.]] memory — a blended, long-run average and spread.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A running-average analogy. A row of daily customer counts, each a small bar of varying height, feeds into a single smooth running-average line that stays steady while the daily bars jump around. A caption says blend each day a little into a long-run memory; the running average stays stable even on a weird day."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Running average: blend each batch a little into a steady long-run memory</text><line x1="50" y1="150" x2="470" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><g fill="#C99A12" opacity="0.7"><rect x="60" y="90" width="20" height="60"/><rect x="110" y="60" width="20" height="90"/><rect x="160" y="110" width="20" height="40"/><rect x="210" y="70" width="20" height="80"/><rect x="260" y="100" width="20" height="50"/><rect x="310" y="55" width="20" height="95"/><rect x="360" y="95" width="20" height="55"/><rect x="410" y="80" width="20" height="70"/></g><text x="180" y="46" fill="#8A6D3B" font-size="8">each batch jumps around</text><path d="M60 96 Q200 90 440 92" fill="none" stroke="#5E5191" stroke-width="3"/><text x="400" y="82" fill="#5E5191" font-size="8">running average (steady)</text></g></svg>
%%%

**What the shopkeeper picture gets right:** you build a trustworthy "typical" value by gently blending each new day into a long memory, so one strange day can't throw it off. **Where it breaks down:** a shopkeeper *chooses* to keep a running number; batch norm does it *automatically* during training — you never touch it, it just accumulates in the background, ready for later.

#### Two modes: the same layer behaves differently
So batch norm has two behaviors, and switching between them is the single most important habit to remember:

%%% table
:: Mode :: Which average/spread does it use? :: Why
Training :: the **live batch's own** numbers :: it has a full batch, and using live numbers is what steadies the drift while learning
Evaluation (test) :: the **frozen running-average** memory :: a single test picture has no batch, so it reads the remembered typical numbers → a stable, deterministic answer
%%%

The rule in one line: **train mode reads the live batch; eval mode reads the frozen memory.** In code you flip this with one switch (in PyTorch, `model.eval()`; in training, `model.train()`). Here's the whole idea in one picture — the same layer, two modes:

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A two-mode diagram for batch norm. On the left, training mode: a full batch of pictures feeds the layer, which reads the live batch average and spread. On the right, evaluation mode: a single picture feeds the layer, which instead reads the frozen running-average memory. A caption says train uses the live batch, eval uses the remembered numbers so one picture gets a stable answer."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Same layer, two modes: live batch (train) vs. frozen memory (eval)</text><g transform="translate(30,36)"><text x="110" y="8" fill="#276b45">TRAINING</text><g fill="#FCF3DC" stroke="#C99A12"><rect x="10" y="20" width="24" height="30"/><rect x="40" y="20" width="24" height="30"/><rect x="70" y="20" width="24" height="30"/><rect x="100" y="20" width="24" height="30"/><rect x="130" y="20" width="24" height="30"/></g><text x="82" y="66" fill="#8A6D3B" font-size="8">a full batch</text><rect x="40" y="80" width="140" height="34" fill="#EAF5EE" stroke="#2D8B55"/><text x="110" y="94" fill="#276b45" font-size="8">reads the LIVE batch's</text><text x="110" y="106" fill="#276b45" font-size="8">own avg & spread</text></g><line x1="260" y1="36" x2="260" y2="160" stroke="#E5DFD6" stroke-dasharray="4,3"/><g transform="translate(290,36)"><text x="110" y="8" fill="#5E5191">EVALUATION (test)</text><rect x="90" y="20" width="24" height="30" fill="#FCF3DC" stroke="#C99A12"/><text x="102" y="66" fill="#8A6D3B" font-size="8">just one picture</text><rect x="40" y="80" width="140" height="34" fill="#EAF0FA" stroke="#5E5191"/><text x="110" y="94" fill="#5E5191" font-size="8">reads the FROZEN running</text><text x="110" y="106" fill="#5E5191" font-size="8">avg & spread (memory)</text></g><text x="260" y="176" fill="#6B645E" font-size="8">one test picture still gets a stable, repeatable answer</text></g></svg>
%%%

**You've now met the thermostat's memory.** During training it steadies with the live batch *and* quietly builds a running average; at test time it switches to that frozen memory so a lone picture gets a rock-steady answer. **This train/eval split is the source of the classic batch norm bugs** — which is exactly what the next concept turns into three friendly puzzles.

@@@ concept id=c7 tag="What you gain" title="The payoff: faster, bolder, less fussy training" gotit="Got the payoff"
Time for a victory lap. You've built the whole thermostat — now let's see *why* people put it in almost every network. The payoff is big and very practical, and it's why batch norm was one of the most influential ideas in deep learning.

Think of learning to ride a bike **with training wheels** versus without. Without them, you inch along terrified — one wrong lean and you topple, so you creep. Bolt on training wheels and suddenly you can pedal *hard* and *fast*, take bigger turns, and not stress about the exact way you first sat on the seat, because the wheels keep you from tipping over. Batch norm is training wheels for a deep network: because the signal can't drift wildly hot or cold anymore, you can train *boldly* — bigger steps, faster progress — without toppling.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A training-wheels analogy for what batch norm buys you. On the left, without batch norm: a wobbly bike creeping slowly with a note tiny careful steps. On the right, with batch norm: a steady bike with training wheels moving fast with a note big bold steps, and three small badges: faster training, bigger learning rate, less fussy about initialization. A caption says batch norm is training wheels; you can ride boldly."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Batch norm = training wheels: you can train boldly instead of creeping</text><g transform="translate(40,40)"><rect x="0" y="0" width="180" height="110" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="20" fill="#8a3b3b">WITHOUT batch norm</text><circle cx="55" cy="70" r="18" fill="none" stroke="#8a3b3b" stroke-width="2"/><circle cx="120" cy="70" r="18" fill="none" stroke="#8a3b3b" stroke-width="2"/><text x="90" y="100" fill="#8a3b3b" font-size="8">tiny, careful steps — or it topples</text></g><g transform="translate(300,40)"><rect x="0" y="0" width="180" height="110" fill="#EAF5EE" stroke="#2D8B55"/><text x="90" y="20" fill="#276b45">WITH batch norm</text><circle cx="55" cy="66" r="16" fill="none" stroke="#276b45" stroke-width="2"/><circle cx="120" cy="66" r="16" fill="none" stroke="#276b45" stroke-width="2"/><circle cx="42" cy="86" r="6" fill="none" stroke="#2D8B55"/><circle cx="133" cy="86" r="6" fill="none" stroke="#2D8B55"/><text x="90" y="104" fill="#276b45" font-size="8">big bold steps — the wheels keep it up</text></g></g></svg>
%%%

**What the training-wheels picture gets right:** the safety of not toppling lets you go faster and worry less about getting the start perfect — the same three gifts batch norm gives training. **Where it breaks down:** training wheels stay bolted on forever, but a network *keeps* its batch norm at test time too (using the frozen memory) — the wheels never come off, they just switch to a steadier mode.

#### The three concrete wins
Spelled out plainly, dropping batch norm into a network buys you three things:

1. **Bigger learning rates.** Because the signal stays in a steady range, you can take *larger* training steps without the loss exploding. Bigger steps mean the network reaches a good answer in fewer laps.
2. **Faster, steadier training.** The loss falls sooner and with less jitter, so you spend less time babysitting the run.
3. **Less fragile to initialization.** Before batch norm, you had to set the very first weights *just so* (a whole craft of its own) or training would stall or blow up. The thermostat re-centers the signal anyway, so a merely-okay start is now good enough.

Watch the difference in one picture — the loss curve with and without the thermostat:

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Two loss curves over training steps. The without-batch-norm curve is jittery and falls slowly, staying high. The with-batch-norm curve is smooth and dives quickly to a low loss. A caption says batch norm makes the loss fall faster and smoother, so you can use bigger steps."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Loss over training: with batch norm it falls faster and smoother</text><line x1="55" y1="150" x2="475" y2="150" stroke="#B8AEA2"/><line x1="55" y1="30" x2="55" y2="150" stroke="#B8AEA2"/><text x="265" y="170" fill="#6B645E" font-size="8">training steps →</text><text x="42" y="90" fill="#6B645E" font-size="8" transform="rotate(-90 42 90)">loss</text><path d="M60 44 L90 70 L110 55 L140 95 L170 80 L210 110 L260 100 L320 118 L400 112 L470 116" fill="none" stroke="#C93B3B" stroke-width="1.8"/><text x="360" y="104" fill="#8a3b3b" font-size="8">without BN: jittery, slow</text><path d="M60 46 Q130 120 260 138 Q360 144 470 142" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="320" y="132" fill="#276b45" font-size="8">with BN: smooth, fast ↓</text></g></svg>
%%%

**That's the whole reason batch norm is everywhere.** Faster, bolder, less fussy — three real gifts from one tiny thermostat. You've unlocked the trick that made training deep networks *practical* rather than a black art. **But no tool is free of gotchas** — the next concept turns the three classic batch norm mistakes into puzzles you'll never be caught by.

@@@ concept id=c8 tag="Three gotchas" title="Where it trips you up: three puzzles, one fix each" gotit="Got the gotchas"
Batch norm is wonderful, but every one of its powers rests on the same foundation: *having a good batch to read*. Poke at that foundation and three things go wrong. The good news — each is a well-understood puzzle with a clean fix. Meet all three now so none of them ever surprises you.

Think of taking an **opinion poll** to guess what a whole town thinks. Three ways it goes wrong: (1) you ask only **three people** — too few to trust, so your "average opinion" is basically noise; (2) you try to poll a town of **just one person** — there's no "town average" to speak of; (3) you poll a big crowd on Monday, then on Tuesday keep quoting a *totally different fresh crowd's* mood as if it were the settled answer — inconsistent, depending on who happened to show up. Batch norm hits the same three traps, because it too leans on a decent-sized, sensible group to read from.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Three panels naming the three batch norm gotchas as polling problems. Panel 1, small batch: only three tiny figures with a note too few, noisy average. Panel 2, batch of one: a single figure with a note no group to average. Panel 3, forgot eval mode: a shifting crowd with a note answer depends on who shows up. Each panel names the gotcha."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Three ways batch norm trips you — one clean fix each</text><g transform="translate(30,30)"><rect x="0" y="0" width="140" height="100" fill="#FDECEC" stroke="#C93B3B"/><text x="70" y="18" fill="#8a3b3b">① tiny batch</text><g fill="#C93B3B"><circle cx="45" cy="55" r="8"/><circle cx="70" cy="55" r="8"/><circle cx="95" cy="55" r="8"/></g><text x="70" y="86" fill="#8a3b3b" font-size="8">too few → noisy average</text></g><g transform="translate(190,30)"><rect x="0" y="0" width="140" height="100" fill="#FDF2E0" stroke="#C99A12"/><text x="70" y="18" fill="#8A6D3B">② batch of one</text><circle cx="70" cy="55" r="10" fill="#C99A12"/><text x="70" y="86" fill="#8A6D3B" font-size="8">no group to average</text></g><g transform="translate(350,30)"><rect x="0" y="0" width="140" height="100" fill="#EAF0FA" stroke="#5E5191"/><text x="70" y="18" fill="#5E5191">③ forgot eval mode</text><g fill="#5E5191" opacity="0.7"><circle cx="40" cy="50" r="6"/><circle cx="62" cy="60" r="6"/><circle cx="86" cy="48" r="6"/><circle cx="104" cy="62" r="6"/></g><text x="70" y="86" fill="#5E5191" font-size="8">answer depends on the crowd</text></g></g></svg>
%%%

**What the polling picture gets right:** every failure comes from a *bad sample* — too few, none, or the wrong crowd — not from the idea being wrong. **Where it breaks down:** a bad poll gives an obviously-shaky number; batch norm often fails *quietly* — the code still runs, the answers are just subtly worse — which is why you learn to name these by sight.

#### Gotcha 1 · The batch is too small → the reading is noisy
Batch norm's average and spread are only trustworthy if the batch is big enough. With just a **handful of examples** per batch, those numbers bounce around randomly from step to step — noisy, unreliable — and batch norm *hurts* instead of helps. **The fix:** switch to a normalization that doesn't lean on the batch at all. Two named cousins do this — [[Group Normalization||A normalization that steadies over a group of channels within a single example, so it doesn't depend on the batch size at all — the go-to fix when batches are small.]] (steadies over groups of channels within one example) and [[Layer Normalization||A normalization that steadies over all the features within a single example, independent of the batch. It's the standard choice in transformers.]] (steadies over all features within one example). Both read from *one example*, so a tiny batch no longer matters. You'll meet these properly in the normalization family later.

#### Gotcha 2 · A single test picture → there's no batch to average
At test time you often feed the model **one** picture. You can't take "the average of one" — and even if you tried to use its lone statistics, the output would be unstable and meaningless. **The fix is one you already built:** the *running-statistics memory* from the last concept. In eval mode, batch norm ignores the (nonexistent) batch and reads its frozen, remembered average and spread — so a single picture gets a stable, deterministic answer. This is *why* the running average exists at all.

#### Gotcha 3 · Forgetting to switch to eval mode → wobbly predictions
This is the single most common batch norm bug, and now you can see exactly why it bites. If you forget to flip the model into **eval mode** at test time, batch norm keeps using the *live* batch's statistics. So the very same picture gets a *different* prediction depending on which other pictures happen to share its batch — spooky and wrong. **The fix is one line:** put the model in eval mode (e.g. `model.eval()`) so it uses the frozen running statistics. Here's the before-and-after:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A before-and-after of forgetting eval mode. On the left, labelled forgot eval mode, the same cat picture in two different batches gets two different answers, cat 0.7 and cat 0.4, marked inconsistent. On the right, labelled model.eval(), the same cat in either batch gets the same answer, cat 0.7 and cat 0.7, marked stable. A caption says eval mode uses the frozen memory so the answer never depends on the batch."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Forget eval mode → the same picture gets different answers</text><g transform="translate(30,32)"><rect x="0" y="0" width="200" height="110" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="18" fill="#8a3b3b">forgot eval mode (live batch)</text><text x="100" y="46" fill="#8a3b3b" font-size="8">cat in batch A → "cat 0.7"</text><text x="100" y="72" fill="#8a3b3b" font-size="8">same cat in batch B → "cat 0.4"</text><text x="100" y="98" fill="#8a3b3b">inconsistent ✗</text></g><text x="255" y="90" fill="#B8AEA2" font-size="13">→</text><g transform="translate(285,32)"><rect x="0" y="0" width="205" height="110" fill="#EAF5EE" stroke="#2D8B55"/><text x="102" y="18" fill="#276b45">model.eval() (frozen memory)</text><text x="102" y="46" fill="#276b45" font-size="8">cat in batch A → "cat 0.7"</text><text x="102" y="72" fill="#276b45" font-size="8">same cat in batch B → "cat 0.7"</text><text x="102" y="98" fill="#276b45">stable ✓</text></g></g></svg>
%%%

#### The honest capability limits
Behind those three gotchas sit two plain limits worth stating outright:

- **It leans on the batch.** Batch norm is fundamentally unreliable when the batch is very small or size 1, because it *needs* enough examples to estimate an average and spread. And during training, the very same example can be normalized a little differently depending on which other examples share its batch — its output is not a function of that example alone.
- **It adds no nonlinearity.** Batch norm only *rescales* numbers (subtract, divide, scale, shift) — all straight-line operations. It does **not** replace the activation. You still need a bend (ReLU) to give the network its power; batch norm just keeps the signal healthy so the bend can do its job.

!!! c-warn ⚠️
<b>The quiet trap:</b> batch norm never throws an error when you forget eval mode or use a tiny batch — the code runs fine and the numbers look plausible. The damage shows up only as *slightly wrong, inconsistent* predictions. When results wobble at test time, check eval mode and batch size *first*.
!!!

**You now hold all three gotchas and their fixes:** tiny batch → use Group/Layer norm; single test picture → the running memory handles it; forgot eval mode → call `model.eval()`. None is scary once named. Let's gather the whole day onto one page.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
You dropped a whole thermostat inside a neural network today. Before we file the day away, let's lay every piece out on one page.

Think of the last day of a **school field trip**, when you sit down with your **scrapbook**. All day you collected little things — a ticket stub, a leaf, a photo, a postcard. Now you lay them out on **one page**, in order, so a single glance replays the whole trip. You're not *doing* anything new; you're *gathering* what you already have so it fits in one look, and each item is a hook that pulls back the memory. This recap is your scrapbook page for the thermostat: every idea you met, side by side, in order.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A scrapbook-page analogy for the recap. A single page holds several taped-on keepsakes laid out in order: a ticket stub, a photo, a leaf, and a postcard, each a small labelled card. A caption says one page gathers the whole day; each keepsake is a hook back to a moment."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A recap is a scrapbook page: the whole day gathered in one glance</text><rect x="40" y="30" width="440" height="120" fill="#FDF9EF" stroke="#C99A12" stroke-width="1.5" rx="4"/><g transform="translate(70,52) rotate(-6)"><rect x="0" y="0" width="72" height="52" fill="#FCF3DC" stroke="#C99A12"/><text x="36" y="30" fill="#8A6D3B">drift</text></g><g transform="translate(175,58) rotate(4)"><rect x="0" y="0" width="72" height="52" fill="#EAF0FA" stroke="#5E5191"/><text x="36" y="30" fill="#5E5191">thermostat</text></g><g transform="translate(285,50) rotate(-3)"><rect x="0" y="0" width="72" height="52" fill="#EAF5EE" stroke="#2D8B55"/><text x="36" y="30" fill="#276b45">γ · β</text></g><g transform="translate(392,58) rotate(5)"><rect x="0" y="0" width="72" height="52" fill="#FDECEC" stroke="#C93B3B"/><text x="36" y="30" fill="#8a3b3b">eval</text></g><text x="260" y="168" fill="#6B645E">laid out in order — each keepsake a hook back to a moment</text></g></svg>
%%%

**What the scrapbook picture gets right:** nothing new is created — you gather what you already collected into one ordered page. **Where it breaks down:** scrapbook keepsakes just sit there, but these ideas *connect* — the drift problem calls for the thermostat, which needs `γ`/`β` to stay flexible, which needs a running memory for eval — so this "page" is really a working machine.

Here's the whole layer as one connected picture — raw numbers in, steadied, scaled, shifted, out:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A one-page summary of the batch norm layer. Left to right: raw drifting numbers enter, then subtract the batch average and divide by the batch spread plus epsilon to steady them, then multiply by gamma and add beta to scale and shift, then the steady output leaves toward the ReLU. Below, a note shows training reads the live batch while eval reads the frozen running memory."><g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">The whole layer: steady (−μ, ÷spread+ε) → scale (γ) → shift (β)</text><g transform="translate(20,40)"><rect x="0" y="0" width="80" height="44" fill="#FDECEC" stroke="#C93B3B"/><text x="40" y="20" fill="#8a3b3b">raw numbers</text><text x="40" y="34" fill="#8a3b3b">(drifting)</text></g><text x="108" y="66" fill="#B8AEA2">→</text><g transform="translate(122,40)"><rect x="0" y="0" width="120" height="44" fill="#EAF0FA" stroke="#5E5191"/><text x="60" y="18" fill="#5E5191">− batch avg,</text><text x="60" y="34" fill="#5E5191">÷ spread (+ε)</text></g><text x="250" y="66" fill="#B8AEA2">→</text><g transform="translate(264,40)"><rect x="0" y="0" width="110" height="44" fill="#EAF5EE" stroke="#2D8B55"/><text x="55" y="18" fill="#276b45">× γ  (scale)</text><text x="55" y="34" fill="#276b45">+ β  (shift)</text></g><text x="382" y="66" fill="#B8AEA2">→</text><g transform="translate(396,40)"><rect x="0" y="0" width="100" height="44" fill="#FCF3DC" stroke="#C99A12"/><text x="50" y="20" fill="#8A6D3B">steady out</text><text x="50" y="34" fill="#8A6D3B">→ ReLU</text></g><text x="260" y="118" fill="#6B645E">train: read the LIVE batch  ·  eval: read the FROZEN running memory</text><text x="260" y="140" fill="#9A938A" font-size="8">(and it quietly updates that memory on every training step)</text></g></svg>
%%%

The day in a few beats:
- **The drift.** In a deep stack the signal's scale wanders too hot or too cold (the old name: *internal covariate shift*), making training slow and touchy.
- **The thermostat.** Batch norm re-centers the signal: subtract the batch average, divide by the batch spread — so it sits at average `0`, spread `1`.
- **Across the batch, per channel.** It reads the average/spread across the *whole batch* (not one example), one thermostat per channel. A tiny *epsilon* inside the divide prevents a divide-by-zero.
- **Two free knobs.** After steadying, it multiplies by a learnable *scale* `γ` and adds a learnable *shift* `β`, so the network can pick any range — even undo the normalization.
- **Where it sits.** Between the layer and the bend: `conv → batch norm → ReLU → pool`.
- **Train vs test.** Training uses the live batch's numbers *and* builds a *running-average memory*; evaluation uses that frozen memory, so a single test picture gets a stable answer.
- **The payoff.** Bigger learning rates, faster steadier training, and far less fuss about how the weights were first set.
- **Three gotchas.** Tiny batch → noisy stats (use Group/Layer norm); single test picture → running memory handles it; forgot eval mode → call `model.eval()`.
- **The limits.** It leans on a good batch (bad at size 1), an example's output can depend on its batch-mates, and it adds *no* nonlinearity — you still need the bend.

#### Cheat-sheet · the thermostat at a glance
%%% table
:: Step :: What it does :: In one line
Steady :: subtract batch average, divide by batch spread :: `(x − μ) / √(σ² + ε)` → avg 0, spread 1
Epsilon :: tiny number inside the divide :: never divide by zero
Scale & shift :: multiply by γ, add β (both learned) :: gives the range back; can undo the steadying
Placement :: between layer and activation, per channel :: `conv → BN → ReLU → pool`
Training mode :: use the live batch's average/spread :: + quietly update the running memory
Eval mode :: use the frozen running-average memory :: stable answer for a single picture
%%%

#### Cheat-sheet · the words you met today
%%% jargon
batch normalization | a thermostat layer: re-centers the signal to average 0, spread 1
internal covariate shift | the old name for the signal's scale drifting through deep layers
mini-batch statistics | the average and spread measured across the whole current batch
standard deviation | one number for how spread out a set of values is (the "spread")
channel | one filter's found-it map; batch norm keeps one thermostat per channel
epsilon (ε) | a tiny number added inside the divide so you never divide by zero
gamma (γ) | the learnable scale knob — sets the spread after steadying
beta (β) | the learnable shift knob — sets the center after steadying
running statistics | a slowly-updated memory of the typical average/spread, used at test time
train mode | uses the live batch's numbers (and updates the running memory)
eval mode | uses the frozen running memory, so one picture gets a stable answer
Group / Layer Normalization | batch-independent cousins — the fix for very small batches
%%%

That's the whole day. You can now explain what batch norm does, why it exists, where it sits, how it behaves differently at train and test time, and the three ways it can trip you — plus the fix for each. Next you'll help it even more by *feeding* the network more variety, with **day-05, Data Augmentation**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is the core move batch normalization makes on a layer's signal? | a:2 | it adds a bend (nonlinearity) | it drops random units | it subtracts the batch average and divides by the batch spread, re-centering the signal to about average 0, spread 1 | it doubles every value | fb: Batch norm is a thermostat: it reads the signal's average and spread across the batch, then re-centers it to a steady range (avg 0, spread 1). It rescales — it does NOT add a bend.
q: After steadying the signal, why does batch norm add the two learnable knobs γ (scale) and β (shift)? | a:1 | to speed up the divide | so the network can freely re-scale and re-center the signal — even undo the normalization — instead of being locked to avg 0, spread 1 | to prevent divide-by-zero | to count the classes | fb: Forcing every layer to sit at exactly avg 0, spread 1 would be too rigid. γ and β are learned, so the network picks its own center and spread — and can even set them to undo the normalization if that helps. (The tiny ε, not γ/β, is what prevents divide-by-zero.)
q: At test time you classify a single picture. Which statistics does batch norm use, and why? | a:2 | the live picture's own average and spread | the training set re-loaded each time | the frozen running-average memory it built during training, because a single picture has no batch to average | random values | fb: One picture has no batch, so batch norm switches (in eval mode) to the running-average average and spread it accumulated during training — giving a stable, deterministic answer.
q: A model's test predictions for the same picture keep changing depending on which other pictures share its batch. What went wrong, and the fix? | a:1 | learning rate too high; lower it | eval mode was forgotten, so batch norm still uses live batch stats; fix by putting the model in eval mode (model.eval()) | epsilon too small; raise it | too few classes; add more | fb: Without eval mode, batch norm reads the live batch, so an example's answer depends on its batch-mates. Switching to eval mode makes it use the frozen running statistics — the answer stops depending on the batch.
%%%

@@@ produce id=produce tag="Produce" title="Build the thermostat, then watch the drift disappear" gotit="Done"
Time to build the thermostat yourself and *watch* it work. You'll take a batch of drifting numbers — some huge, some tiny — run the batch norm steps by hand, and **print the average and spread before and after** so you can see them snap to `0` and `1`. Then you'll add the `γ`/`β` knobs and the eval-mode running memory. **Predict first:** if a channel's values across the batch are `[10, 20, 30, 40, 50]`, what is their average? After you subtract that average and divide by the spread, what should the *new* average be, and roughly the new spread? Then run it and **observe** the numbers land where you predicted, that a wrong-but-huge value gets pulled back into range, and that flipping to eval mode uses the frozen memory instead of the live batch. Pick one path.

#### Option A · write it yourself
Create `sessions/m06-cnns-vision-encoders/day-04-batch-normalization/experiment.py`. Build a tiny batch norm step and **print the average and spread at every stage**. Start with a fake batch for one channel, `x = np.array([10., 20., 30., 40., 50.])`, and **print** its average (`x.mean()`) and spread (`x.std()`). Then **steady** it: `mu = x.mean(); sigma = x.std(); eps = 1e-5; x_hat = (x - mu) / np.sqrt(sigma**2 + eps)` — **print** `x_hat`, and **notice** its new average is about `0` and its new spread is about `1`. Then add the two knobs: pick `gamma = 2.0`, `beta = 5.0`, compute `y = gamma * x_hat + beta`, **print** it, and **observe** its average is now about `beta` (5) and its spread about `gamma` (2). Finally, mimic the running memory: keep `running_mean` and `running_var`, blend each batch with `momentum = 0.1` (`running_mean = 0.9*running_mean + 0.1*mu`), and **print** how, in "eval mode", using `running_mean`/`running_var` instead of the live `mu`/`sigma` gives a fixed answer for a single value no matter what batch it arrives in. Run with `python3 sessions/m06-cnns-vision-encoders/day-04-batch-normalization/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 6 Day 4 artifact.

Create sessions/m06-cnns-vision-encoders/day-04-batch-normalization/experiment.py that implements a tiny batch normalization step by hand, with a comment on each step and printing values at every step:
1. Makes a fake batch for one channel x = np.array([10., 20., 30., 40., 50.]) and prints its mean and std (the "before" — a drifting, off-center signal).
2. Steadies it: mu = x.mean(); sigma = x.std(); eps = 1e-5; x_hat = (x - mu) / np.sqrt(sigma**2 + eps). Print x_hat and assert its mean is ~0 (np.isclose(x_hat.mean(), 0, atol=1e-6)) and its std is ~1. Point out this is the thermostat move.
3. Adds the learnable knobs: gamma = 2.0, beta = 5.0; y = gamma * x_hat + beta. Print y and point out its mean is now ~beta and its std ~gamma — the network can pick any range, or set gamma=old-std, beta=old-mean to undo normalization.
4. Shows epsilon matters: make a near-flat channel flat = np.array([3., 3., 3., 3.]) with std 0; show that dividing by np.sqrt(0 + eps) is finite (no crash), whereas dividing by 0 would be inf/nan.
5. Mimics running statistics: initialize running_mean=0.0, running_var=1.0, momentum=0.1; update running_mean = (1-momentum)*running_mean + momentum*mu and similarly running_var with sigma**2. Then define an "eval" normalize that uses running_mean/running_var instead of the live batch, and show that a single value (say 25.0) normalized in eval mode gives the SAME answer regardless of what other numbers are in its batch, while live-batch normalization would change with the batch. 
6. Prints a one-line takeaway: "batch norm = subtract batch mean, divide by batch spread (+eps) -> avg 0 spread 1; then scale by gamma and shift by beta; train uses the live batch, eval uses the frozen running memory."
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (predict, then confirm)
- The raw batch has average `30` and a wide spread; after steadying, the new average is about `0.0` and the new spread about `1.0` — the thermostat snapped it to a steady range.
- With `γ = 2, β = 5`, the output's average moves to about `5` and its spread to about `2` — the two knobs set the center and spread, exactly as promised.
- The near-flat channel does **not** crash, because `ε` keeps the divide finite.
- In eval mode, a single value normalized with the *frozen running memory* gives the **same** answer no matter what batch it arrives in — while live-batch normalization would wobble with the batch. That's the whole reason eval mode exists.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m06-cnns-vision-encoders/day-04-batch-normalization/log.md`: (1) the two steps of the core batch-norm move and what each fixes; (2) what γ and β each let the network do; (3) why train mode and eval mode read different statistics, and one bug that follows from forgetting eval mode.
!!!

@@@ fin








