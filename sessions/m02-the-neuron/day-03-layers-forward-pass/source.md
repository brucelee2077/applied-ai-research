---
quest_id: wf2-d03-forward
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 3 — Layers & the Forward Pass"
module_label: "Module 2 · Train · Day 3"
title: "Layers & the Forward Pass"
subtitle: "How Neurons Team Up to Make One Guess"
brand_sub: "Foundations · M2 Day 3"
spine: "assembly line"
nav_prev_href: "../day-02-activations/lesson.html"
nav_prev_label: "Activation Functions"
nav_next_href: "../review-part-a.html"
nav_next_label: "Review Gate"
fin_title: "Module 2 · Day 3 complete! 🏆"
fin_body: "Nice work — you've completed <b>Layers & the Forward Pass</b>. You can now send numbers through a whole network, one layer at a time, and read a prediction off the end.<br>Next up: the <b>Review Gate</b> — a quick checkpoint on everything so far."
notebook_yardstick: 00-neural-networks/fundamentals/05_forward_propagation.ipynb
coverage_topics:
  - {topic: "Layer as a stack of parallel neurons reading the same input", keywords: ["row of neurons that all read the same input", "read the same input"]}
  - {topic: "Weight matrix W: one row per neuron, one column per input feature", keywords: ["weight matrix", "rows = neurons, columns = inputs", "row per neuron"]}
  - {topic: "Bias vector b: one bias per neuron, added after the weighted sum", keywords: ["bias vector", "one bias number per neuron", "one bias per neuron"]}
  - {topic: "Layer pre-activation z = W x + b as one matrix-vector multiply-plus-add", keywords: ["matrix-vector multiply", "z = W·x + b", "every neuron's weighted sum at once"]}
  - {topic: "Elementwise activation a = f(z): the day-02 bend applied to each entry of z", keywords: ["elementwise", "a = f(z)", "each entry of"]}
  - {topic: "Layer types by role: input, hidden, output layer", keywords: ["input layer", "hidden layer", "output layer"]}
  - {topic: "Stacking layers: output of one layer becomes input of the next; sizes must chain", keywords: ["output list of one layer becomes the input list of the next", "sizes must match", "columns = the previous layer"]}
  - {topic: "Forward pass: the ordered, one-direction chain input to prediction", keywords: ["forward pass", "one-direction trip", "one direction, fixed order"]}
  - {topic: "Dimension / shape bookkeeping: rows = neurons, columns = incoming features", keywords: ["shape bookkeeping", "rows = neurons, columns = inputs", "columns = the previous layer"]}
  - {topic: "Why a nonlinearity must sit between layers (else linear collapse); remedy = insert the day-02 activation", keywords: ["linear collapse", "bend between the layers", "put a bend"]}
  - {topic: "Depth as the source of representational power: multiple hidden layers", keywords: ["depth", "several hidden layers", "richer patterns from the last"]}
  - {topic: "Reading the network as a pipeline / composition of functions", keywords: ["composition", "functions nested inside functions", "W₂·f₁"]}
---

@@@ hero
@lede Picture a **car assembly line** in a factory. A bare metal frame rolls in one end. It stops at the first station, where a team bolts on the engine. It rolls to the next station, where another team adds the doors. Station after station, each team does one small job and passes the car along — and a finished car rolls out the far end. A neural network works exactly like this line: your numbers roll in one end, each **layer** is a station full of workers, and a finished **guess** rolls out the other end. That single trip down the line — input to answer — is called the **forward pass**, and it's the exact thing happening every time ChatGPT reads your message and begins to reply. Yesterday you met one worker (a neuron) and the little bend it adds. Today you watch a whole line of them team up.
@goal Together we'll build the assembly line one piece at a time: line up many neurons into a **layer**, learn the neat trick that runs a whole layer's math in one step, then chain layers into a network and push a real input all the way down to a prediction. You'll catch the two classic ways the line jams — and the simple fix for each — and by the end you'll trace a number from input to answer with your finger. Every formula shows up in plain words first. No symbol left unexplained.

@@@ concept id=c1 tag="A team of workers" title="A layer is a row of neurons doing the same job" gotit="Got the layer"
Let's start at the first station on our assembly line. Yesterday you built one worker — one neuron — that looks at the input, does a weighted sum, adds a bias, and applies its little bend. One worker, one job.

But a real station doesn't have one worker. It has a **whole row of them, working side by side**. Picture a row of graders at the front of a classroom, and every grader gets a photocopy of the *same* test. Each grader is checking for a different thing — one looks for spelling, one for math, one for neat handwriting — and each hands back their own score. Same test in, many different scores out. A [[layer||A row of neurons that all read the same input at the same time. Each neuron has its own weights and bias, so each one looks for a different pattern and produces its own number.]] is exactly that: a row of neurons that all read the *same* input vector, but each has its **own** weights and its **own** bias, so each one is hunting for a different pattern.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A stack of photocopies of the same test is handed to a row of four graders. Each grader checks a different thing — spelling, math, neatness, grammar — and each writes back their own score."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same test → a row of graders → each returns its own score</text><rect x="18" y="70" width="58" height="70" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><rect x="24" y="64" width="58" height="70" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><rect x="30" y="58" width="58" height="70" rx="4" fill="#FFFFFF" stroke="#5E5191" stroke-width="2"/><text x="59" y="80" text-anchor="middle" fill="#5E5191" font-size="10">the</text><text x="59" y="96" text-anchor="middle" fill="#5E5191" font-size="10">same</text><text x="59" y="112" text-anchor="middle" fill="#5E5191" font-size="10">test 📄</text><text x="59" y="152" text-anchor="middle" fill="#6B645E" font-size="9">photocopies</text><line x1="90" y1="55" x2="128" y2="40" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="80" x2="128" y2="78" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="105" x2="128" y2="116" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="120" x2="128" y2="154" stroke="#B8AEDA" stroke-width="1.3"/><rect x="130" y="28" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="45" fill="#276b45">🧑 grader: spelling</text><text x="286" y="45" fill="#B8AEA2">→</text><text x="300" y="45" fill="#6B645E">score a₁</text><rect x="130" y="66" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="83" fill="#276b45">🧑 grader: math</text><text x="286" y="83" fill="#B8AEA2">→</text><text x="300" y="83" fill="#6B645E">score a₂</text><rect x="130" y="104" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="121" fill="#276b45">🧑 grader: neatness</text><text x="286" y="121" fill="#B8AEA2">→</text><text x="300" y="121" fill="#6B645E">score a₃</text><rect x="130" y="142" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="159" fill="#276b45">🧑 grader: grammar</text><text x="286" y="159" fill="#B8AEA2">→</text><text x="300" y="159" fill="#6B645E">score a₄</text><text x="430" y="96" text-anchor="middle" fill="#5E5191" font-size="11">one layer =</text><text x="430" y="112" text-anchor="middle" fill="#5E5191" font-size="11">the whole row</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="One input vector fans out to four neurons in a layer; each neuron reads the same input and produces its own output number"><g font-family="monospace" font-size="12">
<rect x="20" y="80" width="70" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/>
<text x="55" y="98" text-anchor="middle" fill="#5E5191">input</text>
<text x="55" y="114" text-anchor="middle" fill="#5E5191">x</text>
<line x1="90" y1="100" x2="150" y2="40" stroke="#B8AEDA" stroke-width="1.5"/>
<line x1="90" y1="100" x2="150" y2="80" stroke="#B8AEDA" stroke-width="1.5"/>
<line x1="90" y1="100" x2="150" y2="120" stroke="#B8AEDA" stroke-width="1.5"/>
<line x1="90" y1="100" x2="150" y2="160" stroke="#B8AEDA" stroke-width="1.5"/>
<circle cx="180" cy="40" r="20" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="180" y="45" text-anchor="middle" fill="#1a5c38">N₁</text>
<circle cx="180" cy="80" r="20" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="180" y="85" text-anchor="middle" fill="#1a5c38">N₂</text>
<circle cx="180" cy="120" r="20" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="180" y="125" text-anchor="middle" fill="#1a5c38">N₃</text>
<circle cx="180" cy="160" r="20" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="180" y="165" text-anchor="middle" fill="#1a5c38">N₄</text>
<text x="180" y="192" text-anchor="middle" fill="#6B645E" font-size="11">the layer: 4 neurons, same input each</text>
<line x1="200" y1="40" x2="300" y2="40" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="200" y1="80" x2="300" y2="80" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="200" y1="120" x2="300" y2="120" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="200" y1="160" x2="300" y2="160" stroke="#8FBFA3" stroke-width="1.5"/>
<text x="330" y="45" fill="#6B645E">→ own score a₁</text>
<text x="330" y="85" fill="#6B645E">→ own score a₂</text>
<text x="330" y="125" fill="#6B645E">→ own score a₃</text>
<text x="330" y="165" fill="#6B645E">→ own score a₄</text>
</g></svg>
%%%

**What the row-of-graders picture gets right:** every grader reads the same test, yet each returns a different score because each cares about a different thing — just like the neurons in a layer share one input but hold different weights. **Where it breaks down:** the graders talk and think, while a neuron only does a fixed sum-then-bend; and the graders act one after another, but the neurons in a layer all fire at the *same time* on the same input.

#### The one thing to hold onto
A layer takes in a small list of numbers and hands back a small list of numbers — one number per neuron. If the station has 4 neurons, the input goes in and 4 numbers come out. That "list of numbers" is called a **vector**, and it's how every number travels down our assembly line. Next we'll find a beautifully compact way to store all those workers' settings in one place.

@@@ concept id=c2 tag="The settings sheet" title="One grid holds every neuron's weights" gotit="Got the weight matrix"
Our station on the assembly line has 4 workers, and each one has its own list of weights. That's a lot of little lists floating around. How do we keep them tidy?

Think of a **school timetable pinned to the wall** — a grid. Each **row** is one student's schedule; each **column** is a time slot. One neat grid holds everyone's plan at once, and you can find any single cell by looking up its row and column. We do the same for a layer: we stack every neuron's weights into one grid called the [[weight matrix||A grid of numbers holding every neuron's weights for a layer. One row per neuron, one column per input feature. Written W.]], written **W**. There's one **row per neuron** and one **column per input number**. So if a layer has 4 neurons and each reads 3 input numbers, W is a grid that is 4 rows tall and 3 columns wide — its **shape** is `[4, 3]`.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A school timetable pinned to a wall as a grid. Each row is one student's schedule, each column is a time slot, and any single class is found by its row and column."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A timetable: one grid holds everyone's plan 📌</text><circle cx="140" cy="36" r="4" fill="#C99A12" stroke="#9A5A12"/><circle cx="380" cy="36" r="4" fill="#C99A12" stroke="#9A5A12"/><rect x="110" y="44" width="300" height="120" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="155" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">9am</text><text x="225" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">10am</text><text x="295" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">11am</text><text x="365" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">12pm</text><line x1="190" y1="66" x2="190" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="260" y1="66" x2="260" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="330" y1="66" x2="330" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="90" x2="410" y2="90" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="115" x2="410" y2="115" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="140" x2="410" y2="140" stroke="#E6D08A" stroke-width="1"/><text x="104" y="82" text-anchor="end" fill="#5E5191" font-size="9">Ann →</text><text x="104" y="107" text-anchor="end" fill="#5E5191" font-size="9">Ben →</text><text x="104" y="132" text-anchor="end" fill="#5E5191" font-size="9">Cam →</text><text x="104" y="157" text-anchor="end" fill="#5E5191" font-size="9">Dan →</text><text x="155" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text x="225" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text x="295" y="82" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="365" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text x="155" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text x="225" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text x="295" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text x="365" y="107" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="155" y="132" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="225" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text x="295" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text x="365" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text x="155" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text x="225" y="157" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="295" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text x="365" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text x="260" y="182" text-anchor="middle" fill="#6B645E" font-size="10">one row per person = one row per neuron in W</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A weight matrix W drawn as a 4-row, 3-column grid. Each row is one neuron's weights, each column lines up with one input feature. A bias vector b sits beside it with one bias per row."><g font-family="monospace" font-size="12">
<text x="40" y="24" fill="#3A342E" font-weight="bold">W — one row per neuron, one column per input</text>
<text x="150" y="46" text-anchor="middle" fill="#9A938A" font-size="10">x₁</text>
<text x="200" y="46" text-anchor="middle" fill="#9A938A" font-size="10">x₂</text>
<text x="250" y="46" text-anchor="middle" fill="#9A938A" font-size="10">x₃</text>
<rect x="125" y="55" width="150" height="120" rx="4" fill="none" stroke="#7C6DAA" stroke-width="2"/>
<line x1="175" y1="55" x2="175" y2="175" stroke="#D8CFEC" stroke-width="1"/>
<line x1="225" y1="55" x2="225" y2="175" stroke="#D8CFEC" stroke-width="1"/>
<line x1="125" y1="85" x2="275" y2="85" stroke="#D8CFEC" stroke-width="1"/>
<line x1="125" y1="115" x2="275" y2="115" stroke="#D8CFEC" stroke-width="1"/>
<line x1="125" y1="145" x2="275" y2="145" stroke="#D8CFEC" stroke-width="1"/>
<text x="118" y="74" text-anchor="end" fill="#2D8B55" font-size="10">N₁ →</text>
<text x="118" y="104" text-anchor="end" fill="#2D8B55" font-size="10">N₂ →</text>
<text x="118" y="134" text-anchor="end" fill="#2D8B55" font-size="10">N₃ →</text>
<text x="118" y="164" text-anchor="end" fill="#2D8B55" font-size="10">N₄ →</text>
<text x="150" y="74" text-anchor="middle" fill="#5E5191">.2</text><text x="200" y="74" text-anchor="middle" fill="#5E5191">-.1</text><text x="250" y="74" text-anchor="middle" fill="#5E5191">.5</text>
<text x="150" y="104" text-anchor="middle" fill="#5E5191">.0</text><text x="200" y="104" text-anchor="middle" fill="#5E5191">.3</text><text x="250" y="104" text-anchor="middle" fill="#5E5191">-.2</text>
<text x="150" y="134" text-anchor="middle" fill="#5E5191">.4</text><text x="200" y="134" text-anchor="middle" fill="#5E5191">.1</text><text x="250" y="134" text-anchor="middle" fill="#5E5191">.0</text>
<text x="150" y="164" text-anchor="middle" fill="#5E5191">-.3</text><text x="200" y="164" text-anchor="middle" fill="#5E5191">.2</text><text x="250" y="164" text-anchor="middle" fill="#5E5191">.6</text>
<text x="200" y="200" text-anchor="middle" fill="#6B645E" font-size="11">shape [4 neurons, 3 inputs]</text>
<rect x="340" y="55" width="34" height="120" rx="4" fill="#FDF6EC" stroke="#9A5A12" stroke-width="2"/>
<text x="357" y="46" text-anchor="middle" fill="#9A5A12" font-size="11">b</text>
<text x="357" y="74" text-anchor="middle" fill="#9A5A12">b₁</text><text x="357" y="104" text-anchor="middle" fill="#9A5A12">b₂</text><text x="357" y="134" text-anchor="middle" fill="#9A5A12">b₃</text><text x="357" y="164" text-anchor="middle" fill="#9A5A12">b₄</text>
<text x="415" y="118" fill="#6B645E" font-size="11">one bias</text><text x="415" y="134" fill="#6B645E" font-size="11">per neuron</text>
</g></svg>
%%%

**What the timetable picture gets right:** one grid stores everyone's settings tidily, and each row belongs to one person — exactly how each row of W belongs to one neuron. **Where it breaks down:** a timetable is just for reading, but W is used in a live calculation, and — unlike a fixed schedule — its numbers get nudged and improved during training (that's a later day).

#### The bias comes along too
Each neuron also has one **bias** — a single number it adds after its weighted sum, to shift its result up or down. Stack those, one per neuron, and you get a short list called the [[bias vector||A short list holding one bias number for each neuron in the layer. Written b. It has the same length as the number of neurons.]], written **b**. If the layer has 4 neurons, `b` has 4 numbers — that is one bias per neuron. So a layer is fully described by just two things: the grid **W** and the little list **b**.

Hold onto that shape rule — *rows = neurons, columns = inputs* — because in the very next unit it becomes the safety-check that keeps the whole assembly line from jamming.

@@@ concept id=c3 tag="One step, not many" title="z = W·x + b — the whole station in one move" gotit="Got the matrix step"
Here's the moment the assembly line gets its superpower. On Day 1, each neuron did its own weighted sum, one at a time — worker 1 multiplies and adds, then worker 2, then worker 3. Fine for a few workers. But a real layer can have *thousands*, and doing them one by one would be painfully slow.

Think of a **cashier scanning a full cart at self-checkout** versus adding up prices in your head one item at a time. The machine reads the whole cart and gives you one total in a single beep. Math has the same "one beep" move for our grid of weights: it's called a [[matrix-vector multiply||A single operation that multiplies a grid of numbers (the weight matrix) by a list of numbers (the input) to produce a new list — computing every neuron's weighted sum at once.]]. You feed it the whole grid **W** and the whole input list **x**, and in one move it produces *every* neuron's weighted sum at once. Then you add the bias list **b**. The result is a list we call **z**.

In one line: **z = W·x + b**. Say it out loud in plain words: "multiply the weight grid by the input list, then add the bias list." That's the entire station's math in a single step — and it's the reason networks run fast on the graphics chips (GPUs) that are built to do exactly this multiply.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A self-checkout scanner reads a whole shopping cart in one beep and prints one total, next to a slow person adding item prices one at a time in their head."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Scan the whole cart in one BEEP → one total</text><rect x="18" y="34" width="230" height="58" rx="8" fill="#FDECEC" stroke="#E0A9A0"/><text x="133" y="52" text-anchor="middle" fill="#8a3b3b">slow way 😓</text><text x="133" y="70" text-anchor="middle" fill="#9a6b64" font-size="10">add prices one item at a time</text><text x="133" y="84" text-anchor="middle" fill="#9a6b64" font-size="10">$2 … +$3 … +$1 … in your head</text><rect x="18" y="104" width="230" height="64" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="64" y="124" text-anchor="middle" fill="#276b45">🛒 cart</text><rect x="110" y="114" width="46" height="44" rx="6" fill="#FFFFFF" stroke="#2D8B55" stroke-width="2"/><text x="133" y="133" text-anchor="middle" font-size="14">📷</text><text x="133" y="150" text-anchor="middle" fill="#276b45" font-size="9">scan</text><text x="176" y="140" fill="#2D8B55" font-size="14">BEEP!</text><text x="133" y="166" text-anchor="middle" fill="#276b45" font-size="9">the fast way</text><text x="268" y="110" fill="#B8AEA2" font-size="16">→</text><rect x="292" y="104" width="210" height="64" rx="8" fill="#FCF3DC" stroke="#C99A12"/><text x="397" y="128" text-anchor="middle" fill="#9A5A12">🧾 one TOTAL</text><text x="397" y="148" text-anchor="middle" fill="#9A5A12" font-size="10">every neuron's sum</text><text x="397" y="162" text-anchor="middle" fill="#9A5A12" font-size="10">at once = z = W·x + b</text></g></svg>
%%%

%%% viz src=../../viz/matmul.html title="a matrix times a vector, cell by cell" caption="interactive — this is z = W·x built up one cell at a time; press play and watch each neuron's total form"
%%%

**What the self-checkout picture gets right:** one action handles the whole cart at once, just as one matrix-vector multiply handles every neuron at once. **Where it breaks down:** the cashier just sums prices, but the multiply pairs each input with each neuron's weight *first*, then sums — it's a weighted total, not a plain one.

#### Watch one number get born (a tiny worked example)
Let's compute the first neuron's weighted sum by hand so the "one beep" isn't a mystery. Take input `x = [2, 3]` and let neuron 1's weights (the first row of W) be `[0.5, -1]`.

- Multiply each input by its matching weight: `2 × 0.5 = 1.0`, and `3 × (−1) = −3.0`.
- Add those up: `1.0 + (−3.0) = −2.0`. That's the weighted sum for neuron 1.

The bias `b` is one more number added on top afterward — a shift up or down that we'll fold in below. The matrix multiply just does that same little sum for **every** row of W at the same time — one weighted sum born per neuron, all in one move. No new math, only every neuron at once.

Don't take it on faith — **predict the four weighted sums, then run it.** Here's the full `[4, 3]` grid `W` from the last unit meeting the input `x = [2, 3, 1]`, in one beep. This is `W·x` on its own — the pure multiply, **no bias added yet** — so you can reproduce it straight from the grid you just saw. Try to guess even the first number before you reveal it. (Row 1 is `[.2, -.1, .5]`, so its sum is `2×.2 + 3×(−.1) + 1×.5 = .4 − .3 + .5 = .6`.)

%%% demo id=wx label="run W·x (the multiply, no bias yet)"
code: W @ np.array([2, 3, 1])        # W is [4,3], x is length 3
out: array([0.6, 0.7, 1.1, 0.6])
take: <b>One multiply, four weighted sums born.</b> The whole `[4, 3]` grid met the length-3 input and produced one weighted sum per neuron — a length-4 list, all in a single move. Add the bias list `b` on top of this and you have the full `z = W·x + b`.
%%%

!!! c-info 🪜
<b>Optional (skippable) — reading the shapes.</b> The multiply only works when the shapes line up. A `[4, 3]` weight grid (4 neurons, 3 inputs) times a length-3 input list gives a length-4 result — one number per neuron. In shorthand: <code>[4, 3] · [3] → [4]</code>. The middle "3" is shared and cancels out; the outer numbers, 4 and 3, tell you the sizes coming out and going in. You'll see this shape-matching turn into a real safety check in a later unit.
!!!

Next: this list is raw. Remember yesterday's bend? It's about to come back — one neuron per entry.

@@@ concept id=c4 tag="The bend, per worker" title="a = f(z) — each worker adds their bend" gotit="Got the activation step"
The multiply gave us a list `z` — one raw score per neuron. But raw scores aren't the final word. Remember yesterday's **bend**, the activation function? Each worker on the line applies their own bend to their own score before passing it on.

Think of a row of **stamps**. The multiply lays out a row of raw forms (the numbers in `z`), and then each form gets stamped — the same kind of stamp, pressed onto each one separately. The stamp doesn't mix the forms together; it just touches each one on its own. That "same rule, applied to each entry on its own" is what we mean by [[elementwise||Applied to each number in a list separately, one at a time, with no mixing between them. The bend f runs on each entry of z independently.]]. We take the bend `f` (yesterday's ReLU, or sigmoid, or tanh) and run it on each entry of `z` by itself. The result is the layer's real output list, which we call **a**.

In one line: **a = f(z)**. In plain words: "apply the bend `f` to every number in `z`, one at a time." So the full station is just two moves: first `z = W·x + b` (the multiply), then `a = f(z)` (the bend).

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A row of four raw paper forms, each getting the same rubber stamp pressed onto it separately. One stamp per form, no form touches another — the same rule applied to each on its own."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same stamp, pressed on each form on its own</text><text x="55" y="38" text-anchor="middle" fill="#6B645E" font-size="10">raw forms (z)</text><rect x="30" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="98" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="166" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="234" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><line x1="38" y1="58" x2="74" y2="58" stroke="#D8CFEC"/><line x1="38" y1="70" x2="74" y2="70" stroke="#D8CFEC"/><line x1="106" y1="58" x2="142" y2="58" stroke="#D8CFEC"/><line x1="106" y1="70" x2="142" y2="70" stroke="#D8CFEC"/><line x1="174" y1="58" x2="210" y2="58" stroke="#D8CFEC"/><line x1="174" y1="70" x2="210" y2="70" stroke="#D8CFEC"/><line x1="242" y1="58" x2="278" y2="58" stroke="#D8CFEC"/><line x1="242" y1="70" x2="278" y2="70" stroke="#D8CFEC"/><rect x="38" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="56" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="106" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="124" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="174" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="192" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="242" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="260" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><text x="56" y="128" text-anchor="middle" font-size="18">🔨</text><text x="124" y="128" text-anchor="middle" font-size="18">🔨</text><text x="192" y="128" text-anchor="middle" font-size="18">🔨</text><text x="260" y="128" text-anchor="middle" font-size="18">🔨</text><text x="158" y="148" text-anchor="middle" fill="#6B645E" font-size="9">the SAME stamp f, one per form — no form touches another</text><rect x="330" y="56" width="172" height="56" rx="8" fill="#EFEAF7" stroke="#7C6DAA"/><text x="416" y="80" text-anchor="middle" fill="#5E5191">elementwise:</text><text x="416" y="98" text-anchor="middle" fill="#5E5191" font-size="10">a = f(z), each entry alone</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A list z of four raw numbers, each passed separately through a ReLU bend, producing the output list a. Negatives become zero, positives pass through."><g font-family="monospace" font-size="12">
<text x="55" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold">z (raw)</text>
<rect x="20" y="40" width="70" height="110" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/>
<text x="55" y="66" text-anchor="middle" fill="#5E5191">-1.0</text>
<text x="55" y="92" text-anchor="middle" fill="#5E5191">2.4</text>
<text x="55" y="118" text-anchor="middle" fill="#5E5191">-0.3</text>
<text x="55" y="144" text-anchor="middle" fill="#5E5191">0.8</text>
<text x="200" y="26" text-anchor="middle" fill="#1a5c38" font-weight="bold">apply bend f, each on its own</text>
<line x1="95" y1="60" x2="300" y2="60" stroke="#8FBFA3" stroke-width="1.5"/><text x="200" y="55" text-anchor="middle" fill="#2D8B55" font-size="10">ReLU: max(0, z)</text>
<line x1="95" y1="86" x2="300" y2="86" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="95" y1="112" x2="300" y2="112" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="95" y1="138" x2="300" y2="138" stroke="#8FBFA3" stroke-width="1.5"/>
<text x="380" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold">a (output)</text>
<rect x="345" y="40" width="70" height="110" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text x="380" y="66" text-anchor="middle" fill="#1a5c38">0.0</text>
<text x="380" y="92" text-anchor="middle" fill="#1a5c38">2.4</text>
<text x="380" y="118" text-anchor="middle" fill="#1a5c38">0.0</text>
<text x="380" y="144" text-anchor="middle" fill="#1a5c38">0.8</text>
<text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">negatives → 0, positives pass — one entry at a time</text>
</g></svg>
%%%

**What the row-of-stamps picture gets right:** the same stamp touches each form separately, changing none of the others — exactly how the bend hits each entry of `z` alone. **Where it breaks down:** a stamp always leaves the same mark, but the bend's effect depends on the number underneath (a `−1` becomes `0`, a `2.4` stays `2.4`) — the rule is fixed, the result is not.

#### Why "each on its own" matters
The multiply is where neurons *mix* the input together. The bend does the opposite — it touches each neuron's score alone, adding no mixing. That clean split (mix in the multiply, bend elementwise) is why the two-step recipe `z = W·x + b` then `a = f(z)` describes *every* dense layer you'll ever build. Learn it once, reuse it forever.

You now have one complete station. Next we snap stations together into a full line.

@@@ concept id=c5 tag="Snap them together" title="Input, hidden, output — and how they chain" gotit="Got the stacking rule"
One station is nice, but a factory has *many* stations in a row — and this is the moment our whole assembly line comes alive. Picture the finished part rolling off station 1 straight onto station 2's bench, then onto station 3, all the way down. Each station takes whatever the last one handed it, does its own work, and passes it forward. Nothing skips ahead, nothing waits — the part just flows from bench to bench until a finished thing rolls off the end. Feel that flow first, because *that flow is the whole idea* — now we just give the stations their names.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A car assembly line of three stations connected by a conveyor belt. A bare frame rolls in, station one adds a part, passes it along the belt to station two, then station three, and a finished car rolls off the end."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Parts roll bench → bench, one way, station to station</text><rect x="20" y="90" width="470" height="18" rx="9" fill="#ECE6DC" stroke="#C0B8AC"/><circle cx="45" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="120" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="200" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="290" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="380" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="465" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><text x="36" y="66" text-anchor="middle" font-size="16">🔩</text><text x="36" y="84" text-anchor="middle" fill="#5E5191" font-size="9">frame in</text><rect x="96" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="135" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 1</text><text x="135" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ engine</text><rect x="196" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="235" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 2</text><text x="235" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ doors</text><rect x="296" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="335" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 3</text><text x="335" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ wheels</text><text x="430" y="62" text-anchor="middle" font-size="18">🚗</text><text x="430" y="82" text-anchor="middle" fill="#9A5A12" font-size="9">car out</text><text x="82" y="130" fill="#7C6DAA" font-size="9">input layer</text><text x="235" y="130" text-anchor="middle" fill="#2D8B55" font-size="9">hidden layers</text><text x="430" y="130" text-anchor="middle" fill="#9A5A12" font-size="9">output layer</text><text x="260" y="152" text-anchor="middle" fill="#6B645E" font-size="10">each station hands its part to the next — output of one = input of the next</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="An input list flows into a hidden layer of 4 neurons, whose output flows into an output layer of 2 neurons, producing the final answer. Sizes must chain: the hidden layer's 4 outputs must match the output layer's 4 inputs."><g font-family="monospace" font-size="11">
<rect x="15" y="65" width="60" height="50" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/>
<text x="45" y="86" text-anchor="middle" fill="#5E5191">input</text><text x="45" y="102" text-anchor="middle" fill="#5E5191">3 nums</text>
<line x1="75" y1="90" x2="140" y2="90" stroke="#B8AEDA" stroke-width="1.5"/><text x="107" y="82" text-anchor="middle" fill="#9A938A" font-size="9">→</text>
<rect x="140" y="45" width="90" height="90" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text x="185" y="30" text-anchor="middle" fill="#1a5c38" font-weight="bold">hidden</text>
<text x="185" y="88" text-anchor="middle" fill="#1a5c38">4 neurons</text><text x="185" y="104" text-anchor="middle" fill="#1a5c38">→ 4 nums</text>
<line x1="230" y1="90" x2="300" y2="90" stroke="#8FBFA3" stroke-width="1.5"/><text x="265" y="82" text-anchor="middle" fill="#2D8B55" font-size="9">4 out = 4 in</text>
<rect x="300" y="55" width="90" height="70" rx="6" fill="#FDF6EC" stroke="#9A5A12" stroke-width="2"/>
<text x="345" y="40" text-anchor="middle" fill="#9A5A12" font-weight="bold">output</text>
<text x="345" y="90" text-anchor="middle" fill="#9A5A12">2 neurons</text><text x="345" y="106" text-anchor="middle" fill="#9A5A12">→ 2 nums</text>
<line x1="390" y1="90" x2="455" y2="90" stroke="#E0B87A" stroke-width="1.5"/>
<text x="470" y="94" text-anchor="middle" fill="#6B645E">answer</text>
<text x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="10">each layer's output length must equal the next layer's input length</text>
</g></svg>
%%%

Look at the three stations in the picture and give each its name by where it sits on the line:

- The first block, the raw part rolling on, is the [[input layer||The raw features you feed in — the numbers describing one example, like the pixels of an image. It's not a computing layer; it's just the starting list.]]: the list you feed in — the numbers describing one example. It doesn't compute anything; it's just the frame rolling onto the line.
- The middle block is a [[hidden layer||A middle layer between input and output. It builds intermediate patterns from the previous layer's output. A network can have several stacked hidden layers.]]: it takes the previous list and builds a richer, more useful list of patterns. A network can have several of these in a row.
- The last block is the [[output layer||The final layer. Its output list is the network's answer — one number for a yes/no guess, or several for a multi-way choice.]]: its output list *is* the network's answer.

And the flow you just felt has a precise name too: **the output list of one layer becomes the input list of the next.** The finished part from the hidden station rolls straight onto the output station's bench. Feed the input into the hidden layer, take its output `a`, and hand *that* to the output layer as if it were a fresh input.

**What the assembly-line picture gets right:** parts flow one way, station to station, each handing its work to the next — exactly how a layer's output becomes the next layer's input. **Where it breaks down:** on a real line a station can pass a part backward for rework, but the forward pass only ever flows *forward* — no going back (the "going back" part is learning, and that's a later day).

#### The one rule that makes them chain
For the output station to accept the hidden station's part, the sizes must match: **this layer's number of inputs = the previous layer's number of neurons.** If the hidden layer has 4 neurons, it outputs a list of 4 numbers — so the output layer *must* be built to read 4 numbers (its W needs 4 columns). Get that wrong and the parts don't fit. That single sizing rule is the heart of the failure we tackle next — and now you know exactly why it matters.

@@@ concept id=c6 tag="The full trip" title="The forward pass — one trip down the line" gotit="Got the forward pass"
Now we run the whole thing. Feed one input in, let each station do its two-step job, hand the result to the next, and read the answer off the end. That single, front-to-back trip is the [[forward pass||The ordered, one-direction trip that turns an input into a prediction: input → layer 1 → bend → layer 2 → … → output. Also called a forward propagation or, on a trained network, inference.]] — the star of today.

Picture **dominoes standing in a row**. You tip the first one, it knocks the next, which knocks the next, all the way to the end — one direction, in a fixed order, no skipping. The forward pass is that toppling chain: input tips the hidden layer, the hidden layer tips the output layer, and the last layer's fall *is* the prediction.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A row of standing dominoes tipping over in sequence. A finger tips the first domino, which knocks the second, then the third, all falling one direction in a fixed order until the last one falls — that fall is the prediction."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Tip the first → the whole row topples, one way, in order</text><text x="30" y="64" font-size="18">👆</text><g transform="rotate(-32 78 118)"><rect x="66" y="78" width="24" height="56" rx="3" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.8"/></g><text x="78" y="152" text-anchor="middle" fill="#5E5191" font-size="9">input</text><g transform="rotate(-16 150 110)"><rect x="138" y="70" width="24" height="64" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/></g><text x="150" y="152" text-anchor="middle" fill="#276b45" font-size="9">layer 1</text><rect x="218" y="66" width="24" height="68" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="230" y="152" text-anchor="middle" fill="#276b45" font-size="9">bend</text><rect x="296" y="66" width="24" height="68" rx="3" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.8"/><text x="308" y="152" text-anchor="middle" fill="#9A5A12" font-size="9">layer 2</text><rect x="374" y="66" width="24" height="68" rx="3" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.8"/><text x="386" y="152" text-anchor="middle" fill="#9A5A12" font-size="9">output</text><g transform="rotate(66 452 118)"><rect x="440" y="84" width="24" height="56" rx="3" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.8"/></g><text x="478" y="110" text-anchor="middle" fill="#C93B3B" font-size="14">💥</text><text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">no skipping, no going back — the last fall IS the prediction</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="The forward pass as an ordered chain: input, then layer 1 multiply, then bend, then layer 2 multiply, arriving at the output prediction. Arrows point one direction only."><g font-family="monospace" font-size="10">
<rect x="10" y="70" width="52" height="40" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="36" y="94" text-anchor="middle" fill="#5E5191">x</text>
<text x="72" y="94" fill="#9A938A">→</text>
<rect x="86" y="70" width="66" height="40" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="119" y="88" text-anchor="middle" fill="#1a5c38">W₁·x+b₁</text><text x="119" y="102" text-anchor="middle" fill="#1a5c38">= z₁</text>
<text x="160" y="94" fill="#9A938A">→</text>
<rect x="174" y="70" width="54" height="40" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="1.8"/><text x="201" y="94" text-anchor="middle" fill="#9A5A12">f(z₁)=a₁</text>
<text x="236" y="94" fill="#9A938A">→</text>
<rect x="250" y="70" width="70" height="40" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="285" y="88" text-anchor="middle" fill="#1a5c38">W₂·a₁+b₂</text><text x="285" y="102" text-anchor="middle" fill="#1a5c38">= ŷ</text>
<text x="328" y="94" fill="#9A938A">→</text>
<rect x="346" y="65" width="120" height="50" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="406" y="86" text-anchor="middle" fill="#C93B3B" font-weight="bold">prediction</text><text x="406" y="102" text-anchor="middle" fill="#C93B3B">ŷ (raw)</text>
<text x="260" y="40" text-anchor="middle" fill="#3A342E" font-weight="bold" font-size="12">one direction, fixed order — the forward pass</text>
<text x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">multiply, bend, multiply → read the answer</text>
</g></svg>
%%%

**What the dominoes picture gets right:** the strict one-way order — each step only happens after the one before it, and the last fall is the result. **Where it breaks down:** dominoes are identical and just fall, but each layer does real, different math on the numbers; and unlike dominoes you can't run this backward for free — going backward is a separate, harder trip you'll meet on training days.

#### The whole network in one honest line
Read the picture right to left as a nesting: the output layer's answer is just a multiply (no bend today), and *its* input was a bend applied to a multiply — the hidden layer's work. Written as one nested expression, our 2-layer network is:

`ŷ = W₂·f₁(W₁·x + b₁) + b₂`

Say it in plain words: "take the first layer's total, bend it, then run that through the second layer's multiply-plus-bias." The inner `f₁(W₁·x + b₁)` is the hidden layer's output list `a₁`; the outer `W₂·a₁ + b₂` is the raw output `ŷ`. It looks busy, but it's just our two-step recipe on the hidden layer, followed by one more multiply for the output. Reading a network as this stack of steps-inside-steps is called seeing it as a [[composition||A function made by feeding one function's output into the next. A network is a chain of layers composed together: the output of each becomes the input of the next.]] — and it's the honest one-line summary of everything we built today.

Don't just read the busy line — **watch a number actually make the trip.** Predict what comes out, then run the whole forward pass and watch the length change from `3` in, to `4` after the hidden layer, down to `2` at the output:

%%% demo id=fwd label="run the full forward pass"
code: a1 = relu(W1 @ x + b1); yhat = W2 @ a1 + b2; (a1.shape, yhat)   # x is length 3
out: ((4,), array([ 1.32, -0.07]))
take: <b>One trip down the line, one prediction out.</b> The length-3 input became a length-4 hidden list `a1` (bent by ReLU), which became the length-2 raw answer `ŷ`. That is the entire nested line `ŷ = W₂·f₁(W₁·x + b₁) + b₂`, run start to finish — the forward pass in two lines of code.
%%%

!!! c-info 🪜
<b>Optional (skippable) — where's the output bend?</b> Notice there's no `f₂` wrapped around the whole thing: today the output layer does its multiply-plus-bias and stops, handing back a *raw* list. That's on purpose — the forward pass we're building ends in a plain list of numbers. Some networks *do* add a final bend on the output (a sigmoid or softmax) to turn that raw list into a clean probability, but that belongs with the **loss** lesson next, so we leave the output raw for now.
!!!

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it:</b> "The forward pass is the ordered chain input → (W·x + b) → activation → next layer → … → output. Each hidden dense layer is one matrix-vector multiply plus a bias, then an elementwise activation. The whole network is a composition of those layers — the output of one is the input of the next — and it runs strictly in one direction to produce a prediction." You can already say every piece of that.

**You just ran a full network in your head.** Next: the two ways this line jams, and the one-line fix for each.

@@@ concept id=c7 tag="Jam #1: bad fit" title="When the parts don't fit (shape mismatch)" gotit="Got the shape fix"
Here's the first way the assembly line jams — and it's the single most common error a beginner hits. Good news: it's noisy and easy to fix once you know the rule.

Picture trying to plug a **3-prong plug into a 4-hole socket**. It just won't seat — the counts don't match, and you notice instantly. That's a [[shape mismatch||When one layer outputs a list of one length but the next layer was built to read a different length, so the matrix multiply can't line up. The program stops with a dimension error.]]: layer 1 outputs a list of length 4, but layer 2's weight grid was built with only 3 columns — so its multiply can't line up with the 4 numbers coming in. The program stops with a "dimension mismatch" error.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A three-prong electrical plug held in front of a socket that has four holes. The prong count and the hole count do not match, so the plug cannot seat — a visible, instant mismatch."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">3 prongs into a 4-hole socket — won't seat</text><rect x="40" y="52" width="70" height="60" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="75" y="46" text-anchor="middle" fill="#276b45" font-size="9">layer-1 plug</text><line x1="110" y1="66" x2="150" y2="66" stroke="#2D8B55" stroke-width="5"/><line x1="110" y1="82" x2="150" y2="82" stroke="#2D8B55" stroke-width="5"/><line x1="110" y1="98" x2="150" y2="98" stroke="#2D8B55" stroke-width="5"/><text x="75" y="88" text-anchor="middle" fill="#276b45" font-weight="bold">3 prongs</text><text x="178" y="86" text-anchor="middle" fill="#C93B3B" font-size="22">✗</text><rect x="210" y="46" width="84" height="72" rx="8" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="252" y="40" text-anchor="middle" fill="#C93B3B" font-size="9">layer-2 socket</text><circle cx="252" cy="62" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><circle cx="252" cy="78" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><circle cx="252" cy="94" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><circle cx="252" cy="110" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><text x="278" y="90" fill="#C93B3B" font-weight="bold">4 holes</text><text x="167" y="140" text-anchor="middle" fill="#C93B3B" font-size="10">3 ≠ 4 → you notice instantly (loud error)</text><rect x="330" y="52" width="172" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="416" y="74" text-anchor="middle" fill="#276b45">fix: match the counts</text><text x="416" y="92" text-anchor="middle" fill="#276b45" font-size="10">columns = previous layer's neurons</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A layer outputs a length-4 list, but the next layer's weight grid has only 3 columns, so they cannot connect. The fix: make columns equal the previous layer's neuron count."><g font-family="monospace" font-size="11">
<rect x="20" y="55" width="70" height="70" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text x="55" y="40" text-anchor="middle" fill="#1a5c38">layer 1 out</text>
<text x="55" y="80" text-anchor="middle" fill="#1a5c38">length</text><text x="55" y="98" text-anchor="middle" fill="#1a5c38" font-weight="bold">4</text>
<text x="120" y="94" fill="#C93B3B" font-size="20">✗</text>
<rect x="155" y="55" width="120" height="70" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/>
<text x="215" y="40" text-anchor="middle" fill="#C93B3B">layer 2's W</text>
<text x="215" y="82" text-anchor="middle" fill="#C93B3B">only 3</text><text x="215" y="100" text-anchor="middle" fill="#C93B3B">columns</text>
<text x="215" y="145" text-anchor="middle" fill="#C93B3B" font-size="10">4 ≠ 3 → won't connect</text>
<text x="300" y="70" fill="#6B645E">fix:</text>
<rect x="330" y="55" width="170" height="50" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text x="415" y="76" text-anchor="middle" fill="#1a5c38">give layer 2's W</text><text x="415" y="94" text-anchor="middle" fill="#1a5c38" font-weight="bold">4 columns</text>
<text x="415" y="130" text-anchor="middle" fill="#2D8B55" font-size="10">columns = previous layer's neurons</text>
</g></svg>
%%%

**What the plug-and-socket picture gets right:** mismatched counts simply can't connect, and you find out right away. **Where it breaks down:** a plug is a physical object you can't force, while the mismatch here is a rule the computer checks — same idea, but it fails as a printed error, not a physical block.

#### The named rule that never lets it happen
The remedy is the shape rule you already met, now used as a checklist. Whenever you build a layer's weight grid **W**:

- **rows = the number of neurons in *this* layer** (how many outputs you want), and
- **columns = the number of inputs coming in = the number of neurons in the *previous* layer.**

Walk the whole line once with your finger, checking "does each layer's column count equal the layer before it's neuron count?" and every part will fit. This little habit — call it **shape bookkeeping** — is exactly what experienced engineers do first when a network won't run. Unlike the next jam, this one is loud: the program *tells* you. The next jam is silent, and far sneakier.

@@@ concept id=c8 tag="Jam #2: silent collapse" title="Layers with no bend fold into one" gotit="Got the collapse fix"
Now the sneaky jam — the one that doesn't crash, doesn't print an error, and quietly wastes all your hard work. It's the exact idea from yesterday, seen on the assembly line.

Picture a line where **every station is just a conveyor belt that scales the part** — station 1 doubles its length, station 2 triples it. Run a part through both and it's simply six times as long. You didn't need two stations; **one** belt set to "×6" does the identical job. Stack a hundred such belts and you *still* only get one scaling. That's the [[linear collapse||When you stack layers with no activation between them, the whole stack behaves like a single layer. All the depth is wasted because straight-line steps combine into one straight-line step.]]: layers with no bend between them fold into a single layer, and all your depth vanishes.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Two conveyor belts in a row that only scale a bar's length. The first stretches it two times, the second three times. A dashed arrow shows one single belt set to six times does the exact same job — two belts collapse into one."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two scaling belts (×2 then ×3) = one belt set to ×6</text><rect x="20" y="56" width="22" height="10" rx="2" fill="#7C6DAA"/><text x="31" y="48" text-anchor="middle" fill="#5E5191" font-size="9">part</text><rect x="52" y="48" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><text x="97" y="65" text-anchor="middle" fill="#276b45">belt ×2</text><rect x="150" y="56" width="44" height="10" rx="2" fill="#7C6DAA"/><rect x="200" y="48" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><text x="245" y="65" text-anchor="middle" fill="#276b45">belt ×3</text><rect x="298" y="56" width="132" height="10" rx="2" fill="#7C6DAA"/><text x="364" y="48" text-anchor="middle" fill="#5E5191" font-size="9">6× longer</text><text x="445" y="64" fill="#C93B3B">=</text><rect x="20" y="104" width="22" height="10" rx="2" fill="#7C6DAA"/><rect x="52" y="96" width="238" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.8"/><text x="171" y="113" text-anchor="middle" fill="#C93B3B" font-weight="bold">one belt ×6 — same result</text><rect x="298" y="104" width="132" height="10" rx="2" fill="#7C6DAA"/><text x="210" y="144" text-anchor="middle" fill="#C93B3B" font-size="10">no bend → two stations do the work of one (depth wasted)</text><text x="260" y="164" text-anchor="middle" fill="#2D8B55" font-size="10">a bend between them → they stay two real layers</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Two linear layers with no activation collapse into one single equivalent layer. Inserting an activation between them keeps them distinct."><g font-family="monospace" font-size="12" text-anchor="middle">
<rect x="30" y="45" width="55" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="57" y="67" fill="#5E5191">W₁·x</text>
<text x="102" y="66" fill="#6B645E" font-size="11">then</text>
<rect x="130" y="45" width="55" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="157" y="67" fill="#5E5191">W₂·</text>
<text x="215" y="66" fill="#C93B3B">=</text>
<rect x="240" y="42" width="150" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="315" y="66" fill="#C93B3B" font-weight="bold">one layer (W₂·W₁)·x</text>
<text x="210" y="102" fill="#C93B3B" font-size="10">no bend → depth wasted</text>
<text x="315" y="130" fill="#2D8B55" font-size="11">put a bend between them → they stay two real layers</text>
</g></svg>
%%%

**What the conveyor-belt picture gets right:** plain scaling steps really do multiply into a single scaling, so extra stations add nothing. **Where it breaks down:** a belt only scales, but a layer also *mixes and shifts* the numbers — yet the punchline holds: without a bend, all that mixing still collapses into one single mixing step.

#### See the collapse happen — with real numbers
The picture says two straight layers fold into one. Let's *prove* it with numbers you can check by hand. Take a tiny input `x = [1, 2]` and two small weight grids (biases 0 to keep it clean):

- `W1 = [[1, 2], [0, 1]]` — the first station.
- `W2 = [[1, 0], [3, 1]]` — the second station.

Here's the whole build-up in one figure — feed `x` through both grids **step by step** on the top path, and through their single combined grid `W₂·W₁` on the bottom path. Watch both paths land on the **same** two numbers:

%%% svg
<svg viewBox="0 0 520 230" role="img" aria-label="Two-layer path: x times W1 gives [5,2], then times W2 gives [5,17]. Bottom path: the combined grid W2 times W1 equals [[1,2],[3,7]], and that times x gives [5,17]. Both paths reach the same result, showing two linear layers collapse into one."><g font-family="monospace" font-size="11">
<text x="18" y="22" fill="#3A342E" font-weight="bold" font-size="12">Top path — two layers, one at a time</text>
<rect x="18" y="34" width="52" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="44" y="55" text-anchor="middle" fill="#5E5191">x=[1,2]</text>
<text x="78" y="55" fill="#9A938A">→</text>
<rect x="94" y="34" width="76" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="132" y="49" text-anchor="middle" fill="#1a5c38">W₁·x</text><text x="132" y="62" text-anchor="middle" fill="#1a5c38">=[5,2]</text>
<text x="178" y="55" fill="#9A938A">→</text>
<rect x="194" y="34" width="86" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="237" y="49" text-anchor="middle" fill="#1a5c38">W₂·[5,2]</text><text x="237" y="62" text-anchor="middle" fill="#1a5c38">=[5,17]</text>
<text x="288" y="55" fill="#9A938A">→</text>
<rect x="304" y="30" width="96" height="42" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="352" y="55" text-anchor="middle" fill="#C93B3B" font-weight="bold">[5, 17]</text>
<line x1="18" y1="92" x2="502" y2="92" stroke="#E4DED5" stroke-width="1"/>
<text x="18" y="118" fill="#3A342E" font-weight="bold" font-size="12">Bottom path — one combined grid</text>
<text x="18" y="140" fill="#6B645E">first multiply the grids:</text>
<rect x="180" y="122" width="150" height="30" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="1.8"/><text x="255" y="142" text-anchor="middle" fill="#9A5A12">W₂·W₁ = [[1,2],[3,7]]</text>
<rect x="18" y="170" width="52" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="44" y="191" text-anchor="middle" fill="#5E5191">x=[1,2]</text>
<text x="78" y="191" fill="#9A938A">→</text>
<rect x="94" y="170" width="120" height="34" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="1.8"/><text x="154" y="185" text-anchor="middle" fill="#9A5A12">(W₂·W₁)·x</text><text x="154" y="198" text-anchor="middle" fill="#9A5A12">=[5,17]</text>
<text x="222" y="191" fill="#9A938A">→</text>
<rect x="238" y="166" width="96" height="42" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="286" y="191" text-anchor="middle" fill="#C93B3B" font-weight="bold">[5, 17]</text>
<text x="400" y="185" fill="#2D8B55" font-weight="bold">same result!</text>
<text x="400" y="200" fill="#6B645E" font-size="10">two layers = one</text>
</g></svg>
%%%

Don't take the picture's word for it — **predict, then run.** With no bend, will the two-step path and the single combined grid `W₂·W₁` give the *same* answer, or different? Guess, then reveal:

%%% demo id=collapse label="run: two linear layers vs one combined grid"
code: two_step = W2 @ (W1 @ x); combined = W2 @ W1; one_step = combined @ x; (combined.tolist(), two_step.tolist(), one_step.tolist())
out: ([[1, 2], [3, 7]], [5, 17], [5, 17])
take: <b>Identical — [5, 17] both ways.</b> With no bend, running `x` through `W₁` then `W₂` gives the exact same numbers as running it through the single grid `W₂·W₁ = [[1,2],[3,7]]`. The two layers really did fold into one. Add a bend between them and this match breaks — the collapse is gone.
%%%

#### The fix you already own
The remedy is the star of yesterday: put a **bend** — the activation `f` — between the layers. That elementwise `a = f(z)` step from unit 4 is not decoration; it is the thing that stops the collapse. With a bend between the layers, `W₁` and `W₂` can no longer be multiplied into one grid, so the two layers stay two genuinely different transforms.

!!! c-warn 🤫
<b>Failure mode (silent):</b> stack linear layers with no activation and nothing breaks — the code runs, shapes fit, a number comes out. But your 5-layer network secretly has the power of *one* layer, so it learns almost nothing and you can't tell why. The tell is a model that stays stubbornly weak no matter how many layers you add. The fix is one line: an activation between every pair of layers.
!!!

#### Why depth is worth having (once the bend is in)
Here's the flip side, and it's the exciting part. *With* a bend between layers, each extra hidden layer can build on the patterns the last one found — simple pieces combined into richer ones. That stacking of pattern-on-pattern is where a network's real power comes from, and it has a name: [[depth||Having several hidden layers in a row. With an activation between them, each layer builds richer patterns from the last — this is where a network's power comes from.]]. Several hidden layers, with bends between them, can build almost anything, drawing richer patterns from the last — one layer alone only draws simple shapes. Depth is only an illusion *without* the bend — and a superpower *with* it. That single trade is the whole reason "deep" learning is called deep.

@@@ concept id=c9 tag="What it can't do yet" title="A forward pass guesses — it doesn't learn (yet)" gotit="Got the limits"
Let's be honest about what today's machine can and can't do — because knowing the edge of a tool is what separates a beginner from someone who really gets it.

Picture a brand-new **vending machine that's plugged in but never stocked**. Press a button and it *will* dispense something — the mechanism works perfectly. But since nobody loaded it with the right snacks, what drops out is basically random. Today's forward pass is exactly that: it runs flawlessly and always produces an answer, but the weights `W` and biases `b` are still whatever they started as, so the answer is **just a guess**. The forward pass *uses* the weights; it never *changes* them.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A vending machine plugged into the wall with its light on. A hand presses a button and an item drops into the tray, but the shelves are empty or randomly filled, so whatever drops out is a random guess."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Plugged in, works — but never stocked → random drop</text><rect x="150" y="30" width="120" height="128" rx="8" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="210" y="46" text-anchor="middle" fill="#5E5191" font-size="9">💡 powered on</text><rect x="160" y="52" width="70" height="58" rx="3" fill="#FFFFFF" stroke="#B8AEDA"/><line x1="160" y1="71" x2="230" y2="71" stroke="#E5DFD6"/><line x1="160" y1="90" x2="230" y2="90" stroke="#E5DFD6"/><line x1="195" y1="52" x2="195" y2="110" stroke="#E5DFD6"/><text x="177" y="66" text-anchor="middle" fill="#B8AEA2" font-size="9">empty</text><text x="213" y="85" text-anchor="middle" fill="#B8AEA2" font-size="9">?</text><text x="177" y="104" text-anchor="middle" fill="#B8AEA2" font-size="9">?</text><circle cx="248" cy="66" r="5" fill="#FCF3DC" stroke="#C99A12"/><circle cx="248" cy="82" r="5" fill="#FCF3DC" stroke="#C99A12"/><text x="245" y="98" fill="#9A5A12" font-size="9">👆</text><rect x="160" y="128" width="70" height="22" rx="3" fill="#FDE8E8" stroke="#C93B3B"/><text x="195" y="143" text-anchor="middle" fill="#C93B3B" font-size="9">random drop 🎲</text><text x="110" y="70" text-anchor="middle" fill="#6B645E" font-size="10">press →</text><text x="300" y="70" fill="#6B645E" font-size="10">it runs...</text><text x="300" y="88" fill="#C93B3B" font-size="10">...but the answer</text><text x="300" y="104" fill="#C93B3B" font-size="10">is just a guess</text><text x="300" y="124" fill="#9A938A" font-size="9">restocking =</text><text x="300" y="138" fill="#9A938A" font-size="9">training (later)</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="The forward pass produces a prediction from fixed weights, but there is no arrow going back to update the weights. That backward, weight-updating trip is training, coming later."><g font-family="monospace" font-size="11">
<rect x="20" y="55" width="60" height="40" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="50" y="79" text-anchor="middle" fill="#5E5191">input</text>
<text x="90" y="79" fill="#2D8B55">→</text>
<rect x="108" y="50" width="120" height="50" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="168" y="70" text-anchor="middle" fill="#1a5c38">forward pass</text><text x="168" y="87" text-anchor="middle" fill="#1a5c38" font-size="10">(fixed W, b)</text>
<text x="236" y="79" fill="#2D8B55">→</text>
<rect x="254" y="55" width="90" height="40" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="299" y="79" text-anchor="middle" fill="#C93B3B">a guess</text>
<path d="M299 100 C 299 130, 168 130, 168 102" fill="none" stroke="#C0B8AC" stroke-width="1.6" stroke-dasharray="5,4"/><text x="235" y="145" text-anchor="middle" fill="#9A938A" font-size="10">updating W back this way = training (later days)</text>
</g></svg>
%%%

**What the vending-machine picture gets right:** the machine works whether or not it's stocked well — just as the forward pass runs whether or not the weights are any good. **Where it breaks down:** you restock a vending machine by hand, but a network restocks *itself* through training — a clever backward trip we haven't taken yet.

#### Three honest edges of today's tool
- **It doesn't learn.** Getting good weights — the backward trip that nudges `W` and `b` toward better answers — is **gradient descent and backpropagation**, coming on the training days. Today we only go forward.
- **The output is raw, not yet a probability.** The last layer hands you a plain list of numbers — just a multiply plus a bias, with no final bend. Turning that raw list into a clean "80% cat" or picking one class is the job of a special output step — **sigmoid** for a yes/no, **softmax** for a many-way choice — which you'll meet with the **loss** function next. So don't expect today's numbers to look like percentages yet.
- **One layer alone is limited.** A single layer is still just one multiply plus one bend — it can't capture every pattern. The way past that ceiling is **depth**, exactly the stacking-with-bends you just learned.

Knowing these edges *is* the win. You now know precisely where the forward pass stops and where training begins — which is exactly the map for the days ahead. One page to gather it all, and you're done.

@@@ concept id=c10 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built a whole working machine today. Here's a fresh way to hold it all at once: think of the day like the **layers of a sandwich you just made**. The bread on the bottom is your input; each filling you add — the cheese, then the tomato, then the lettuce — is one layer's work stacked on the last; and the top slice you press down is the final prediction. To describe your sandwich to a friend, you name the fillings in order, bottom to top — exactly how you read a network bottom to top, input to answer. **Where the sandwich picture breaks down:** in a sandwich the fillings just sit there, but in a network each layer actively *reshapes* what the layer below handed it (and later even retunes itself during training) — the order matters not just for stacking, but because each layer's math feeds the next. Read the six beats below, then keep the cheat-sheets as your map.

The six beats of today, in order:
- **1 · The layer.** A row of neurons that all read the same input, each with its own weights and bias — like graders marking the same test for different things.
- **2 · The settings.** All those weights live in one grid **W** (rows = neurons, columns = inputs), and the biases in one list **b** (one bias per neuron).
- **3 · The multiply.** One move — `z = W·x + b` — computes every neuron's weighted sum at once. Fast, and built for GPUs.
- **4 · The bend.** Apply the activation to each entry on its own: `a = f(z)`. That's one complete hidden layer, in two moves.
- **5 · The chain.** Feed each layer's output into the next; input → hidden → output. Sizes must match: columns = the previous layer's neuron count.
- **6 · The full trip.** The **forward pass** runs the chain one direction, start to finish, to produce a prediction. For our 2-layer network the output stays raw (no final bend): `ŷ = W₂·f₁(W₁·x + b₁) + b₂`.

And the two jams, with their fixes: a **shape mismatch** (loud) → set columns = previous neurons; a **linear collapse** (silent) → put a bend between every pair of layers. Plus the honest edge: the forward pass *guesses*, it doesn't *learn* — that's training, next.

Here's the one picture to keep — the full forward pass on a line.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A stacked sandwich seen from the side. Bottom slice of bread is the input, then a cheese layer, a tomato layer, a lettuce layer stacked in order, and the top slice pressed down is the final prediction — read bottom to top."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Read a network like a sandwich — bottom to top 🥪</text><rect x="120" y="40" width="180" height="18" rx="9" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><text x="330" y="53" fill="#9A5A12" font-size="10">← top slice = prediction ŷ</text><rect x="128" y="58" width="164" height="16" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="330" y="70" fill="#276b45" font-size="10">← lettuce = output layer</text><rect x="128" y="74" width="164" height="16" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.4"/><text x="330" y="86" fill="#C93B3B" font-size="10">← tomato = hidden layer 2</text><rect x="128" y="90" width="164" height="16" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.4"/><text x="330" y="102" fill="#9A5A12" font-size="10">← cheese = hidden layer 1</text><rect x="120" y="106" width="180" height="18" rx="9" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><text x="330" y="119" fill="#5E5191" font-size="10">← bottom slice = input x</text><text x="210" y="146" text-anchor="middle" fill="#6B645E" font-size="10">name the fillings in order = read layers input → answer</text></g></svg>
%%%

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Recap: the forward pass as a one-direction chain — input, hidden layer multiply and bend, output layer multiply, raw prediction."><g font-family="monospace" font-size="10">
<rect x="12" y="52" width="46" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="35" y="73" text-anchor="middle" fill="#5E5191">x</text>
<text x="64" y="72" fill="#9A938A">→</text>
<rect x="80" y="52" width="94" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="127" y="66" text-anchor="middle" fill="#1a5c38">W₁·x+b₁</text><text x="127" y="80" text-anchor="middle" fill="#9A5A12">→ bend</text>
<text x="180" y="72" fill="#9A938A">→</text>
<rect x="196" y="52" width="94" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="243" y="66" text-anchor="middle" fill="#1a5c38">W₂·a₁+b₂</text><text x="243" y="80" text-anchor="middle" fill="#6B645E">(raw)</text>
<text x="296" y="72" fill="#9A938A">→</text>
<rect x="312" y="48" width="96" height="42" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="360" y="73" text-anchor="middle" fill="#C93B3B" font-weight="bold">prediction ŷ</text>
<text x="260" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold" font-size="12">the forward pass — one trip down the line</text>
<text x="260" y="116" text-anchor="middle" fill="#6B645E" font-size="10">multiply → bend → multiply → read the answer</text>
</g></svg>
%%%

#### Cheat-sheet · the two moves of every layer
%%% table
:: :: what it does :: in one line :: shape story
The multiply :: mixes the input — every neuron's weighted sum at once :: `z = W·x + b` :: `[neurons, inputs]·[inputs] → [neurons]`
The bend :: adds the non-linearity, each entry on its own :: `a = f(z)` :: same length in and out
%%%

#### Cheat-sheet · the words you met today
%%% jargon
layer | a row of neurons that all read the same input, each with its own weights and bias
weight matrix W | a grid of all a layer's weights — rows = neurons, columns = inputs
bias vector b | a short list, one bias number per neuron in the layer
matrix-vector multiply | one operation that computes every neuron's weighted sum at once
elementwise | applied to each number in a list on its own, with no mixing between them
input / hidden / output layer | the raw features / the middle pattern-builders / the final answer
forward pass | the one-direction trip input → layers → output that produces a prediction
composition | a network read as functions nested inside functions, output of one feeding the next
shape mismatch | the loud jam: a layer's input length doesn't match the previous layer's output
linear collapse | the silent jam: layers with no bend fold into one, wasting depth
depth | several hidden layers with bends between them — where a network's power comes from
%%%

That's the whole day. Next you'll learn how to *score* the network's guess and start teaching it to improve — the **loss** function and the road to training.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: A layer has 5 neurons and each neuron reads a 3-number input. What is the shape of its weight matrix W? | a:1 | [3, 5] | [5, 3] | [5, 5] | [3, 3] | fb: Rows = neurons, columns = inputs. 5 neurons and 3 inputs give a [5, 3] grid — one row per neuron, one column per input feature.
q: In one dense layer, the correct order of the two moves is: | a:0 | multiply first (z = W·x + b), then bend (a = f(z)) | bend first, then multiply | bend, then bend again | multiply, then multiply again | fb: A layer first computes z = W·x + b (mixing the inputs), then applies the activation elementwise: a = f(z).
q: You stack 6 layers but put NO activation between them. The network runs fine and gives numbers, yet it barely learns anything. Why? | a:2 | The shapes don't match, so it silently skips layers | It ran out of GPU memory | With no bend between them, the linear layers collapse into a single layer — so 6 layers have the power of 1, and depth is wasted; the fix is an activation between each pair | Six layers is always too many | fb: No non-linearity means W₁…W₆ multiply into one grid — the whole stack acts like one layer. Insert an activation between layers and they stay distinct.
q: After a forward pass, why is the network's output usually a bad guess at first? | a:1 | The forward pass has a bug | The weights W and biases b start untrained — the forward pass USES them but never updates them, so the answer is basically random until training (gradient descent) tunes them | The activation is missing | The input was the wrong shape | fb: The forward pass only runs the numbers through fixed weights. Making those weights good is training (gradient descent + backpropagation), which comes later.
%%%

@@@ produce id=produce tag="Produce" title="Push a number down the whole line — and watch it collapse without a bend" gotit="Done"
Time to see today's assembly line run with your own eyes. You'll build a tiny 2-layer network by hand and push one input all the way down to a prediction — then run the sneaky experiment: rip the bend out and watch two layers collapse into one. **Predict first:** with NO activation between the layers, will a 2-layer network give the *same* answer as a single cleverly-chosen layer, or a different one? Then run it and **watch.** Seeing them match without the bend, and differ the moment you add it, is the whole lesson in a few lines of output. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py`. Build a forward pass by hand:
1. Make a layer function `layer(x, W, b, f)` that returns `f(W @ x + b)` — one multiply-plus-bias, then an elementwise bend. Print the shape of `W @ x + b` at each step so you can **watch** the vector length change from layer to layer.
2. Set up a network: input `x` of length 3 → hidden layer of 4 neurons (so `W1` is shape `[4, 3]`, `b1` length 4, bend = `relu`) → output layer of 2 neurons (so `W2` is shape `[2, 4]`, `b2` length 2, no activation on output — the raw prediction, as in today's lesson). **Notice** how the hidden layer's 4 outputs must equal the output layer's 4 inputs — that's the chaining rule.
3. Run the full forward pass and print the final length-2 prediction.
4. The collapse demo: build the SAME 2-layer network but with **no** activation on the hidden layer either (use `f = identity` throughout). Show that its output equals a single layer with `W_combined = W2 @ W1` and `b_combined = W2 @ b1 + b2` — proving two linear layers = one. Then switch the hidden bend back to `relu` and show the outputs **no longer match** any single layer. Run with `python3 sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 2 Day 3 artifact.

Create sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py that, with a comment on each step:
1. Defines relu(z)=np.maximum(0,z) and a layer(x, W, b, f) that returns f(W @ x + b); prints the shape of W @ x + b at each call.
2. Builds a 2-layer forward pass: input x length 3 → hidden layer W1 shape [4,3], b1 length 4, activation relu → output layer W2 shape [2,4], b2 length 2 (no activation on output — the raw prediction). Prints the intermediate vector lengths (3 → 4 → 2) and the final length-2 prediction.
3. Shows the linear collapse: run the SAME network with NO activation anywhere (identity on the hidden layer too) and assert np.allclose(out, (W2 @ W1) @ x + (W2 @ b1 + b2)) — two linear layers equal one layer with W_combined = W2 @ W1. Print W2 @ W1.
4. Then put relu back on the hidden layer and show the output is NOT equal to that single combined layer — the bend prevents the collapse.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the trip, then the collapse)
- The vector length changes as it flows: length **3** in → **4** after the hidden layer → **2** at the output. Watching the length change is watching each station reshape the part.
- With **no** activation, the 2-layer output matches the single combined layer `W2 @ W1` exactly — two linear layers really did collapse into one.
- The moment to **watch**: switch the hidden bend back to `relu` and the two stop matching. That break is depth earning its keep — the same lesson as Day 2, now on a full network.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-03-layers-forward-pass/log.md`: (1) the two moves inside one layer, in order, and the shape rule for W; (2) what the forward pass is, in one sentence, and why its first guess is usually bad; (3) the two ways the line jams (shape mismatch, linear collapse) and the one-line fix for each.
!!!

@@@ fin
