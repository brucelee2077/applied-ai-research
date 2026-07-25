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
@goal Together we'll build the assembly line one piece at a time: line up many neurons into a **layer**, learn the neat trick that runs a whole layer's math in one step, then chain layers into a network and push a real input all the way down to a prediction. You'll see what depth actually buys you, catch the two classic ways the line jams — with the simple fix for each — and by the end you'll trace a number from input to answer with your finger. Every formula shows up in plain words first. No symbol left unexplained.
%%% warmup
q: Yesterday's puzzle: you stack two straight-line layers with NO bend between them. What do you get? | a:1 | An even more powerful two-layer network | Still just one straight-line layer — the depth folds flat | A curved boundary for free | A network that runs twice as fast | concept: linear-collapse | fb: Straight ∘ straight is still straight. With no activation between them, the two layers' weights fold into one — depth buys nothing until you add a bend.
q: What does ReLU do to a number? | a:2 | Squashes it into the range 0 to 1 | Turns every number into 0 or 1 | Passes positives straight through, turns negatives into 0 (max(0, z)) | Adds a small bias to it | concept: relu | fb: ReLU = max(0, z), a one-way valve: positives flow through unchanged, negatives become 0. It's the default hidden-layer bend.
q: Why did deep sigmoid networks stall for years, while ReLU rescued them? | a:0 | Sigmoid's slope tops out near 0.25, so multiplying it through many layers shrinks the learning signal toward 0 (vanishing gradient); ReLU's positive slope is 1, so the signal survives | Sigmoid uses too much memory | ReLU has more parameters | Sigmoid can only handle small numbers | concept: vanishing-gradient | fb: The backward signal is multiplied by each layer's slope. With ≈0.25 per layer it fades fast; ReLU's slope of 1 keeps it full strength — that's why swapping to ReLU let people train deep nets.
%%%

@@@ concept id=c1 tag="A team of workers" title="A layer is a row of neurons doing the same job" gotit="Got the layer"
Let's walk up to the first station on our assembly line.

Yesterday you built one worker — one neuron. It looks at the input, does a weighted sum, adds a bias, applies its little bend. One worker, one job. Nice, but lonely.

A real station doesn't have one worker. It has a **whole row of them, side by side.**

Picture a row of graders at the front of a classroom. Every grader gets a photocopy of the *same* test. One hunts for spelling, one checks the math, one looks at the handwriting — and each hands back their own score. Same test in, many different scores out.

A [[layer||A row of neurons that all read the same input at the same time. Each neuron has its own weights and bias, so each one looks for a different pattern and produces its own number.]] is exactly that: a row of neurons that all read the same input, where each neuron has its **own** weights and its **own** bias — so each one hunts for a different pattern.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A stack of photocopies of the same test is handed to a row of four graders. Each grader checks a different thing — spelling, math, neatness, grammar — and each writes back their own score."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same test → a row of graders → each returns its own score</text><rect x="18" y="70" width="58" height="70" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><rect x="24" y="64" width="58" height="70" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><rect x="30" y="58" width="58" height="70" rx="4" fill="#FFFFFF" stroke="#5E5191" stroke-width="2"/><text x="59" y="80" text-anchor="middle" fill="#5E5191" font-size="10">the</text><text x="59" y="96" text-anchor="middle" fill="#5E5191" font-size="10">same</text><text x="59" y="112" text-anchor="middle" fill="#5E5191" font-size="10">test 📄</text><text x="59" y="152" text-anchor="middle" fill="#6B645E" font-size="9">photocopies</text><line x1="90" y1="55" x2="128" y2="40" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="80" x2="128" y2="78" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="105" x2="128" y2="116" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="120" x2="128" y2="154" stroke="#B8AEDA" stroke-width="1.3"/><rect x="130" y="28" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="45" fill="#276b45">🧑 grader: spelling</text><text x="286" y="45" fill="#B8AEA2">→</text><text x="300" y="45" fill="#6B645E">score a₁</text><rect x="130" y="66" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="83" fill="#276b45">🧑 grader: math</text><text x="286" y="83" fill="#B8AEA2">→</text><text x="300" y="83" fill="#6B645E">score a₂</text><rect x="130" y="104" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="121" fill="#276b45">🧑 grader: neatness</text><text x="286" y="121" fill="#B8AEA2">→</text><text x="300" y="121" fill="#6B645E">score a₃</text><rect x="130" y="142" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="159" fill="#276b45">🧑 grader: grammar</text><text x="286" y="159" fill="#B8AEA2">→</text><text x="300" y="159" fill="#6B645E">score a₄</text><text x="430" y="96" text-anchor="middle" fill="#5E5191" font-size="11">one layer =</text><text x="430" y="112" text-anchor="middle" fill="#5E5191" font-size="11">the whole row</text></g></svg>
%%%

**What the row-of-graders picture gets right:** every grader reads the same test, yet each returns a different score, because each cares about a different thing. **Where it breaks down:** graders talk and think, while a neuron only does a fixed sum-then-bend — and graders take turns, while the neurons in a layer all fire at the *same* moment.

%%% insight
Here's why the row matters. One neuron can ask the input exactly **one** question ("does this look tall?"). A row of four asks four questions at once — and that little committee of answers is what the next station gets to work with. Every big model you've heard of is this trick, repeated: rows of workers, each asking their own question.
%%%

#### One input in, four answers out
So what actually happens when the input arrives at the station? Let's watch it, one move at a time.

%%% steps
step: hand the same input list to every neuron in the row
why: no neuron gets a special copy — this is the "photocopy of the same test" move
step: each neuron uses its OWN weights and its OWN bias
why: that's why four neurons reading identical numbers can return four different answers
step: nobody inside the row talks to anybody else
why: therefore all four can be computed at the very same time — which is exactly what makes layers fast
step: collect the four answers into one small list
why: that list is the station's finished part, ready to roll to the next station
%%%

**So far:** a layer takes in a small list of numbers and hands back a small list of numbers — one number per neuron. Four neurons in, four numbers out.

That "list of numbers" has a name: a **vector**. It is how every number travels down our assembly line, station to station.

If that felt like a lot of new words for one screen, that's normal — you only need one idea from this unit: *a layer is a row, not a worker.* Next we find a beautifully tidy place to keep all those workers' settings.

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

@@@ concept id=c2 tag="The settings sheet" title="One grid holds every neuron's weights" gotit="Got the weight matrix"
Our station has four workers, and each one carries its own little list of weights. That's four lists floating around — and a real station has hundreds. Where do we keep them all?

Think of a **school timetable pinned to the wall.** One grid. Each **row** is one student's schedule; each **column** is a time slot. Everyone's plan lives in one place, and you find any single class by looking up its row and its column.

We do exactly that for a layer. We stack every neuron's weights into one grid called the [[weight matrix||A grid of numbers holding every neuron's weights for a layer. One row per neuron, one column per input feature. Written W.]], written **W**.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A school timetable pinned to a wall as a grid. Each row is one student's schedule, each column is a time slot, and any single class is found by its row and column."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A timetable: one grid holds everyone's plan 📌</text><circle cx="140" cy="36" r="4" fill="#C99A12" stroke="#9A5A12"/><circle cx="380" cy="36" r="4" fill="#C99A12" stroke="#9A5A12"/><rect x="110" y="44" width="300" height="120" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="155" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">9am</text><text x="225" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">10am</text><text x="295" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">11am</text><text x="365" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">12pm</text><line x1="190" y1="66" x2="190" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="260" y1="66" x2="260" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="330" y1="66" x2="330" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="90" x2="410" y2="90" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="115" x2="410" y2="115" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="140" x2="410" y2="140" stroke="#E6D08A" stroke-width="1"/><text x="104" y="82" text-anchor="end" fill="#5E5191" font-size="9">Ann →</text><text x="104" y="107" text-anchor="end" fill="#5E5191" font-size="9">Ben →</text><text x="104" y="132" text-anchor="end" fill="#5E5191" font-size="9">Cam →</text><text x="104" y="157" text-anchor="end" fill="#5E5191" font-size="9">Dan →</text><text x="155" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text x="225" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text x="295" y="82" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="365" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text x="155" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text x="225" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text x="295" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text x="365" y="107" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="155" y="132" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="225" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text x="295" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text x="365" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text x="155" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text x="225" y="157" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="295" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text x="365" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text x="260" y="182" text-anchor="middle" fill="#6B645E" font-size="10">one row per person = one row per neuron in W</text></g></svg>
%%%

**What the timetable picture gets right:** one grid stores everyone's settings tidily, and each row belongs to one person. **Where it breaks down:** a timetable is only for reading, but W sits inside a live calculation — and unlike a fixed schedule, its numbers get nudged and improved during training (a later day).

#### The shape rule: rows = neurons, columns = inputs
Keep the timetable in your head and just swap the labels. Two moves, and then the payoff they hand you:

%%% steps
step: give the grid one row per neuron
why: a row IS a worker's settings, exactly like a row was one student's whole schedule
step: give the grid one column per input number
why: a column lines up with one incoming number, exactly like a column was one time slot
%%%

**So far:** those two moves fix the size of the grid. A layer of 4 neurons reading 3 numbers gets a grid 4 tall and 3 wide — written as shape `[4, 3]`, read out loud as *"four neurons, three inputs."* Every weight in the whole station now lives at one address: row for the worker, column for the input.

%%% insight
Notice what just happened: the layer stopped being a crowd of neurons and became **two objects** — one grid and one short list. That is genuinely all a layer is. When you hear that a model "has 175 billion parameters", those parameters are the numbers sitting in grids exactly like this one.
%%%

#### The bias comes along too
Each neuron also has one **bias** — a single number it adds after its weighted sum, nudging its result up or down. Stack them, and you get a short list called the [[bias vector||A short list holding one bias number for each neuron in the layer. Written b. It has the same length as the number of neurons.]], written **b**.

There is one bias per neuron, so a 4-neuron layer has exactly 4 of them — one bias number per neuron, no more, no less. Look at the tall thin box on the right of the picture below and you can count them.

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
<text x="150" y="104" text-anchor="middle" fill="#5E5191">.0</text><text x="200" y="104" text-anchor="middle" fill="#5E5191">.3</text><text x="250" y="104" text-anchor="middle" fill="#5E5191">-.1</text>
<text x="150" y="134" text-anchor="middle" fill="#5E5191">.4</text><text x="200" y="134" text-anchor="middle" fill="#5E5191">.1</text><text x="250" y="134" text-anchor="middle" fill="#5E5191">.0</text>
<text x="150" y="164" text-anchor="middle" fill="#5E5191">-.3</text><text x="200" y="164" text-anchor="middle" fill="#5E5191">.2</text><text x="250" y="164" text-anchor="middle" fill="#5E5191">.6</text>
<text x="200" y="200" text-anchor="middle" fill="#6B645E" font-size="11">shape [4 neurons, 3 inputs]</text>
<rect x="340" y="55" width="60" height="120" rx="4" fill="#FDF6EC" stroke="#9A5A12" stroke-width="2"/>
<text x="370" y="46" text-anchor="middle" fill="#9A5A12" font-size="11">b</text>
<text x="370" y="74" text-anchor="middle" fill="#9A5A12">0.0</text><text x="370" y="104" text-anchor="middle" fill="#9A5A12">-0.5</text><text x="370" y="134" text-anchor="middle" fill="#9A5A12">0.0</text><text x="370" y="164" text-anchor="middle" fill="#9A5A12">-2.0</text>
<text x="412" y="118" fill="#6B645E" font-size="11">one bias</text><text x="412" y="134" fill="#6B645E" font-size="11">per neuron</text>
</g></svg>
%%%

Those exact numbers are our running example for the whole day — one grid `W`, one bias list `b`. We'll feed them the same input `x = [1, 2, 3]` in every unit from here on, so you can always check the page against your own arithmetic.

Hold onto that shape rule — *rows = neurons, columns = inputs* — because later today it becomes the safety check that keeps the whole line from jamming.

@@@ concept id=c3 tag="One step, not many" title="z = W·x + b — the whole station in one move" gotit="Got the matrix step"
This is the moment the assembly line gets its superpower.

On Day 1, each neuron did its own weighted sum, one at a time: worker 1 multiplies and adds, then worker 2, then worker 3. Fine for three workers. A real layer can have *thousands*, and one-at-a-time would crawl.

Picture a **self-checkout scanner** versus adding up prices in your head. In your head you go "$2… plus $3… plus $1…" item by item. The scanner reads the whole cart and gives you one total in a single beep.

Math has that same "one beep" move for our grid. It is called a [[matrix-vector multiply||A single operation that multiplies a grid of numbers (the weight matrix) by a list of numbers (the input) to produce a new list — computing every neuron's weighted sum at once.]]: you hand it the whole grid **W** and the whole input list **x**, and in one move it gives you every neuron's weighted sum at once. Then you add the bias list **b**. The result is a list we call **z**.

In one line: **z = W·x + b**. Out loud, in plain words: *"multiply the weight grid by the input list, then add the bias list."*

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A self-checkout scanner reads a whole shopping cart in one beep and prints one total, next to a slow person adding item prices one at a time in their head."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Scan the whole cart in one BEEP → one total</text><rect x="18" y="34" width="230" height="58" rx="8" fill="#FDECEC" stroke="#E0A9A0"/><text x="133" y="52" text-anchor="middle" fill="#8a3b3b">slow way 😓</text><text x="133" y="70" text-anchor="middle" fill="#9a6b64" font-size="10">add prices one item at a time</text><text x="133" y="84" text-anchor="middle" fill="#9a6b64" font-size="10">$2 … +$3 … +$1 … in your head</text><rect x="18" y="104" width="230" height="64" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="64" y="124" text-anchor="middle" fill="#276b45">🛒 cart</text><rect x="110" y="114" width="46" height="44" rx="6" fill="#FFFFFF" stroke="#2D8B55" stroke-width="2"/><text x="133" y="133" text-anchor="middle" font-size="14">📷</text><text x="133" y="150" text-anchor="middle" fill="#276b45" font-size="9">scan</text><text x="176" y="140" fill="#2D8B55" font-size="14">BEEP!</text><text x="133" y="166" text-anchor="middle" fill="#276b45" font-size="9">the fast way</text><text x="268" y="110" fill="#B8AEA2" font-size="16">→</text><rect x="292" y="104" width="210" height="64" rx="8" fill="#FCF3DC" stroke="#C99A12"/><text x="397" y="128" text-anchor="middle" fill="#9A5A12">🧾 one TOTAL</text><text x="397" y="148" text-anchor="middle" fill="#9A5A12" font-size="10">every neuron's sum</text><text x="397" y="162" text-anchor="middle" fill="#9A5A12" font-size="10">at once = z = W·x + b</text></g></svg>
%%%

**What the self-checkout picture gets right:** one action handles the whole cart at once, just as one multiply handles every neuron at once. **Where it breaks down:** the scanner only sums prices, while the multiply first pairs each input with each neuron's weight and *then* sums — a weighted total, not a plain one.

Now go play with it, because this widget lets you look *inside* the beep. Two things to try, in this order:

1. **Hover or tap any green cell** on the right-hand grid. The row of `A` and the column of `B` that build that one cell light up — that is a single neuron's weighted sum, assembling in front of you.
2. **Drag the `n` slider down to 1.** Now `B` is a single column — exactly our input list `x` — and the whole right-hand grid becomes one column of totals: one number per neuron. That is `W·x`, live.

%%% viz src=../../viz/matmul.html title="a matrix times a vector, cell by cell" caption="interactive — hover or tap any green cell to light up the row and column that build it; drag the n slider to 1 so B is a single column, our input x, and each green cell is one neuron's weighted sum"
%%%

#### Watch one number get born
Before we trust the beep, let's do one neuron by hand — with the real grid from the last unit. Our input is `x = [1, 2, 3]`, and neuron 1 is the **first row** of `W`: `[.2, -.1, .5]`.

%%% steps
step: pair each input with its matching weight
why: 1 goes with .2, 2 goes with −.1, and 3 goes with .5 — the row lines up with the input, slot by slot
step: multiply each pair: 1 × .2 = .2, then 2 × (−.1) = −.2, then 3 × .5 = 1.5
why: a big weight means "this input matters a lot to me"; a negative weight means "this counts against me"
step: add the pieces up: .2 + (−.2) + 1.5 = 1.5
why: that single number is neuron 1's weighted sum — one worker's whole opinion, in one number
step: now do that for EVERY row of the grid, in the same instant
why: that's the beep — the same little sum as before, one per neuron, all at once
%%%

**So far:** nothing new has been invented. The matrix multiply is the Day-1 weighted sum, done for every row of W at the same time.

%%% insight
Why does anyone care that it's *one* move instead of a loop? Because "one move" is a shape that graphics chips (GPUs) were built to eat. The same numbers, hundreds of times faster. That is the real reason big models are trained as stacks of matrix multiplies and not as loops over neurons.
%%%

#### Predict, then run it
Here's the full `[4, 3]` grid meeting `x = [1, 2, 3]` in one beep. This is `W·x` alone — the pure multiply, **no bias added yet** — so you can check every number straight from the grid you just saw. You already hand-computed the first one.

%%% demo id=wx label="run W·x (the multiply, no bias yet)"
predict: you got 1.5 for neuron 1 by hand. How MANY numbers come out of the beep in total — and will your 1.5 be one of them?
code: W = np.array([[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]])   # 4 neurons, 3 inputs
code: x = np.array([1., 2., 3.])
code: W @ x
out: array([1.5, 0.3, 0.6, 1.9])
take: <b>One multiply, four weighted sums born.</b> Your hand-computed 1.5 is sitting right there in slot 1. The whole `[4, 3]` grid met the length-3 input and produced one weighted sum per neuron — a length-4 list, in a single move. Add the bias list `b` on top and you have the full `z = W·x + b`.
%%%

!!! c-info 🪜
<b>Optional (skippable) — reading the shapes.</b> The multiply only works when the shapes line up. A <code>[4, 3]</code> weight grid (4 neurons, 3 inputs) times a length-3 input list gives a length-4 result — one number per neuron. In shorthand: <code>[4, 3] · [3] → [4]</code>. The middle "3" is shared and cancels out; the outer numbers tell you the size going in and the size coming out. This becomes a real safety check later today.
!!!

That list is still raw, and the bias hasn't even landed yet. Remember yesterday's bend? Both are about to arrive — one per worker.

@@@ concept id=c4 tag="The bend, per worker" title="a = f(z) — each worker adds their bend" gotit="Got the activation step"
The beep gave us four raw weighted sums. Two small things still have to happen before the part can roll on: each worker adds its own bias, and then each worker adds its own bend.

Picture a row of **rubber stamps.** The multiply laid out a row of raw forms (the numbers in `z`). Now each form gets stamped — the same stamp, pressed onto each form separately. The stamp never mixes two forms together. It just touches one, then the next.

That "same rule, applied to each entry on its own" is what we mean by [[elementwise||Applied to each number in a list separately, one at a time, with no mixing between them. The bend f runs on each entry of z independently.]]. We take the bend `f` — yesterday's ReLU, or sigmoid, or tanh — and run it on each entry of `z` by itself. The result is the layer's real output list, which we call **a**.

In one line: **a = f(z)**. In plain words: *"apply the bend to every number in z, one at a time."*

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A row of four raw paper forms, each getting the same rubber stamp pressed onto it separately. One stamp per form, no form touches another — the same rule applied to each on its own."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same stamp, pressed on each form on its own</text><text x="55" y="38" text-anchor="middle" fill="#6B645E" font-size="10">raw forms (z)</text><rect x="30" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="98" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="166" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="234" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><line x1="38" y1="58" x2="74" y2="58" stroke="#D8CFEC"/><line x1="38" y1="70" x2="74" y2="70" stroke="#D8CFEC"/><line x1="106" y1="58" x2="142" y2="58" stroke="#D8CFEC"/><line x1="106" y1="70" x2="142" y2="70" stroke="#D8CFEC"/><line x1="174" y1="58" x2="210" y2="58" stroke="#D8CFEC"/><line x1="174" y1="70" x2="210" y2="70" stroke="#D8CFEC"/><line x1="242" y1="58" x2="278" y2="58" stroke="#D8CFEC"/><line x1="242" y1="70" x2="278" y2="70" stroke="#D8CFEC"/><rect x="38" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="56" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="106" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="124" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="174" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="192" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="242" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="260" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><text x="56" y="128" text-anchor="middle" font-size="18">🔨</text><text x="124" y="128" text-anchor="middle" font-size="18">🔨</text><text x="192" y="128" text-anchor="middle" font-size="18">🔨</text><text x="260" y="128" text-anchor="middle" font-size="18">🔨</text><text x="158" y="148" text-anchor="middle" fill="#6B645E" font-size="9">the SAME stamp f, one per form — no form touches another</text><rect x="330" y="56" width="172" height="56" rx="8" fill="#EFEAF7" stroke="#7C6DAA"/><text x="416" y="80" text-anchor="middle" fill="#5E5191">elementwise:</text><text x="416" y="98" text-anchor="middle" fill="#5E5191" font-size="10">a = f(z), each entry alone</text></g></svg>
%%%

**What the row-of-stamps picture gets right:** the same stamp touches each form separately and changes none of the others. **Where it breaks down:** a stamp always leaves the same mark, but the bend's *result* depends on the number underneath — a `−1` becomes `0`, a `2.4` stays `2.4`. The rule is fixed; the outcome is not.

#### Finish the station with our own numbers
We left the beep holding `W·x = [1.5, 0.3, 0.6, 1.9]`. Now the bias lands, and then the stamp. Watch what the bias quietly does to two of the workers.

%%% steps
step: add the bias list — z = [1.5, 0.3, 0.6, 1.9] + [0.0, −0.5, 0.0, −2.0] = [1.5, −0.2, 0.6, −0.1]
why: each worker gets ITS own bias, and worker 4's big −2.0 just dragged a healthy 1.9 below zero
step: entry 1 is 1.5 → ReLU gives 1.5
why: ReLU is max(0, z), so a positive passes straight through, untouched — worker 1 is shouting and gets heard
step: entry 2 is −0.2 → ReLU gives 0.0
why: therefore a negative gets flattened to zero — worker 2 stays quiet on this input
step: entry 3 is 0.6 → ReLU gives 0.6
why: notice worker 1 being loud did NOT rescue worker 2; each entry of z is judged completely alone
step: entry 4 is −0.1 → ReLU gives 0.0
why: four entries in, four entries out — the bend never changes the length of the list
%%%

**So far:** the station's finished part is `a = [1.5, 0.0, 0.6, 0.0]`. Here is that exact ladder, drawn — raw list on the left, bent list on the right.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A list z of four numbers 1.5, -0.2, 0.6, -0.1, each passed separately through a ReLU bend, producing the output list a of 1.5, 0, 0.6, 0. Negatives become zero, positives pass through unchanged."><g font-family="monospace" font-size="12">
<text x="55" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold">z (raw)</text>
<rect x="20" y="40" width="70" height="110" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/>
<text x="55" y="66" text-anchor="middle" fill="#5E5191">1.5</text>
<text x="55" y="92" text-anchor="middle" fill="#5E5191">-0.2</text>
<text x="55" y="118" text-anchor="middle" fill="#5E5191">0.6</text>
<text x="55" y="144" text-anchor="middle" fill="#5E5191">-0.1</text>
<text x="200" y="26" text-anchor="middle" fill="#1a5c38" font-weight="bold">apply bend f, each on its own</text>
<line x1="95" y1="60" x2="300" y2="60" stroke="#8FBFA3" stroke-width="1.5"/><text x="200" y="55" text-anchor="middle" fill="#2D8B55" font-size="10">ReLU: max(0, z)</text>
<line x1="95" y1="86" x2="300" y2="86" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="95" y1="112" x2="300" y2="112" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="95" y1="138" x2="300" y2="138" stroke="#8FBFA3" stroke-width="1.5"/>
<text x="380" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold">a (output)</text>
<rect x="345" y="40" width="70" height="110" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text x="380" y="66" text-anchor="middle" fill="#1a5c38">1.5</text>
<text x="380" y="92" text-anchor="middle" fill="#1a5c38">0.0</text>
<text x="380" y="118" text-anchor="middle" fill="#1a5c38">0.6</text>
<text x="380" y="144" text-anchor="middle" fill="#1a5c38">0.0</text>
<text x="440" y="92" fill="#C93B3B" font-size="10">← flattened</text>
<text x="440" y="144" fill="#C93B3B" font-size="10">← flattened</text>
<text x="260" y="172" text-anchor="middle" fill="#6B645E" font-size="10">negatives → 0, positives pass — one entry at a time</text>
</g></svg>
%%%

%%% insight
Here's the clean split worth remembering. The multiply is where neurons **mix** the input together. The bend does the opposite — it touches each score **alone**, mixing nothing. That's why the two-move recipe `z = W·x + b`, then `a = f(z)`, describes *every* dense layer you will ever build, in any framework, for the rest of your life. Learn it once, reuse it forever.
%%%

Two of our four workers just went silent on this input, and that is completely normal — a layer is a committee where only the relevant members speak up. (If a worker went quiet on *every* input, that would be a real problem with a real name, and you'll meet it on a later day.)

You now own one complete station: mix, then bend. Next we snap stations together into a full line.

@@@ concept id=c5 tag="Snap them together" title="Input, hidden, output — and how they chain" gotit="Got the stacking rule"
One station is nice. A factory has *many* — and this is where our assembly line finally comes alive.

Picture the part rolling off station 1 straight onto station 2's bench, then onto station 3, all the way down. Each station takes whatever the last one handed it, does its own work, and passes it forward. Nothing skips ahead. Nothing waits.

Feel that flow first, because *the flow is the whole idea.* All we do now is give the stations their names.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A car assembly line of three stations connected by a conveyor belt. A bare frame rolls in, station one adds a part, passes it along the belt to station two, then station three, and a finished car rolls off the end."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Parts roll bench → bench, one way, station to station</text><rect x="20" y="90" width="470" height="18" rx="9" fill="#ECE6DC" stroke="#C0B8AC"/><circle cx="45" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="120" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="200" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="290" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="380" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="465" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><text x="36" y="66" text-anchor="middle" font-size="16">🔩</text><text x="36" y="84" text-anchor="middle" fill="#5E5191" font-size="9">frame in</text><rect x="96" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="135" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 1</text><text x="135" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ engine</text><rect x="196" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="235" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 2</text><text x="235" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ doors</text><rect x="296" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="335" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 3</text><text x="335" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ wheels</text><text x="430" y="62" text-anchor="middle" font-size="18">🚗</text><text x="430" y="82" text-anchor="middle" fill="#9A5A12" font-size="9">car out</text><text x="82" y="130" fill="#7C6DAA" font-size="9">input layer</text><text x="235" y="130" text-anchor="middle" fill="#2D8B55" font-size="9">hidden layers</text><text x="430" y="130" text-anchor="middle" fill="#9A5A12" font-size="9">output layer</text><text x="260" y="152" text-anchor="middle" fill="#6B645E" font-size="10">each station hands its part to the next — output of one = input of the next</text></g></svg>
%%%

**What the assembly-line picture gets right:** parts flow one way, station to station, each handing its work to the next. **Where it breaks down:** a real line can pass a part *backward* for rework, but the forward pass only ever flows forward. (Going backward is learning — a later day.)

#### Name the three stations by where they sit
- The frame rolling on is the [[input layer||The raw features you feed in — the numbers describing one example, like the pixels of an image. It's not a computing layer; it's just the starting list.]]: the numbers describing one example. It computes nothing; it's just the starting list.
- The middle station is a [[hidden layer||A middle layer between input and output. It builds intermediate patterns from the previous layer's output. A network can have several stacked hidden layers.]]: it turns the previous list into a richer list of patterns. A network can have several of these in a row.
- The last station is the [[output layer||The final layer. Its output list is the network's answer — one number for a yes/no guess, or several for a multi-way choice.]]: its output list *is* the network's answer.

**So far:** three names, one flow. Now the rule that makes the flow legal.

%%% insight
Why "hidden"? Because nobody ever tells that middle station what to produce. There's no correct answer for it to copy — it *invents* its own useful features from whatever the layer below hands it. That freedom is where a network's cleverness actually lives, and it's why we care so much about what happens between the input and the answer.
%%%

#### The one rule that makes them chain
The flow has a precise version: **the output list of one layer becomes the input list of the next.** The hidden station's finished part rolls straight onto the output station's bench. So the sizes must match — and here's how you check it, in three moves you can do with your finger.

%%% steps
step: count the neurons in the layer you just built — our hidden layer has 4
why: 4 neurons means it hands forward a list of exactly 4 numbers, no more, no less
step: build the NEXT layer's grid with 4 columns
why: columns = the previous layer's neuron count, because each column must catch one incoming number
step: walk the whole line once, layer by layer, asking "columns here = neurons before?"
why: therefore a size clash is caught before you ever hit Run — sizes must match at every joint
%%%

Read the diagram below as that finger-walk, drawn: 3 numbers in, our 4-neuron hidden layer, then a 2-neuron output layer. See how the "4 out" and the "4 in" meet in the middle.

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

Nice — you can now read any network diagram in the world: count the boxes, check the joints. Hold that finger-walk; it is about to save you from the first jam.

@@@ concept id=c6 tag="The full trip" title="The forward pass — one trip down the line" gotit="Got the forward pass"
Now we run the whole thing, end to end. This is today's star.

Picture **dominoes standing in a row.** You tip the first one. It knocks the next, which knocks the next, all the way to the end. One direction, in a fixed order, no skipping and no going back.

Feed one input in, let each station do its two-move job, hand the result along, and read the answer off the end. That single front-to-back trip is the [[forward pass||The ordered, one-direction trip that turns an input into a prediction: input → layer 1 → bend → layer 2 → … → output. Also called forward propagation, or inference on a trained network.]] — a one-direction trip from numbers to a guess. The input tips the hidden layer, the hidden layer tips the output layer, and the last domino's fall *is* the prediction.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A row of standing dominoes tipping over in sequence. A finger tips the first domino, which knocks the second, then the third, all falling one direction in a fixed order until the last one falls — that fall is the prediction."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Tip the first → the whole row topples, one way, in order</text><text x="30" y="64" font-size="18">👆</text><g transform="rotate(-32 78 118)"><rect x="66" y="78" width="24" height="56" rx="3" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.8"/></g><text x="78" y="152" text-anchor="middle" fill="#5E5191" font-size="9">input</text><g transform="rotate(-16 150 110)"><rect x="138" y="70" width="24" height="64" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/></g><text x="150" y="152" text-anchor="middle" fill="#276b45" font-size="9">layer 1</text><rect x="218" y="66" width="24" height="68" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="230" y="152" text-anchor="middle" fill="#276b45" font-size="9">bend</text><rect x="296" y="66" width="24" height="68" rx="3" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.8"/><text x="308" y="152" text-anchor="middle" fill="#9A5A12" font-size="9">layer 2</text><rect x="374" y="66" width="24" height="68" rx="3" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.8"/><text x="386" y="152" text-anchor="middle" fill="#9A5A12" font-size="9">output</text><g transform="rotate(66 452 118)"><rect x="440" y="84" width="24" height="56" rx="3" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.8"/></g><text x="478" y="110" text-anchor="middle" fill="#C93B3B" font-size="14">💥</text><text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">no skipping, no going back — the last fall IS the prediction</text></g></svg>
%%%

**What the dominoes picture gets right:** the strict one-way order — each step only happens after the one before it, and the last fall is the result. **Where it breaks down:** dominoes are identical and just fall over, while each layer does real, different math. And you can't run this backward for free — the backward trip is a separate, harder journey you'll meet on the training days.

%%% insight
This little trip is not a toy. Every time you send ChatGPT a message, your words become numbers and take exactly this ride — in one direction, layer after layer, until numbers come out the far end and get turned back into words. Same shape of journey you're about to trace with your finger. Just a much longer line.
%%%

#### Take the trip, one station at a time
Our network, with the numbers we've been building all day: 3 numbers in, our 4-neuron hidden layer with a ReLU bend, then a 2-neuron output layer whose grid `W₂` is `[[.5, .5, 0, .5], [−.5, 0, 1, .5]]` with both biases 0.

%%% steps
step: the part rolls on — x = [1, 2, 3]
why: this is the input layer, just the raw numbers describing one example
step: station 1 mixes — z₁ = W₁·x + b₁ = [1.5, −0.2, 0.6, −0.1]
why: one beep gives all 4 hidden neurons their weighted sums, so the list is now length 4
step: station 1 bends — a₁ = f(z₁) = [1.5, 0.0, 0.6, 0.0]
why: the stamp hits each of the 4 entries alone; this is the step that stops the collapse we meet later
step: the hand-off — a₁ rolls onto station 2's bench
why: the hidden layer's output IS the output layer's input; that's the chaining rule, live
step: station 2 mixes — ŷ = W₂·a₁ + b₂, so ŷ₁ = .5(1.5) + .5(0) + 0(0.6) + .5(0) = 0.75
why: 2 output neurons, so the list shrinks from 4 to 2 — and today we stop here, raw
step: read ŷ off the end of the line
why: those 2 numbers are the network's guess; the trip is over, one direction, fixed order
%%%

**So far:** six moves, and every single one you already learned. The whole trip drawn as one line looks like this.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="The forward pass as an ordered chain: input, then layer 1 multiply, then bend, then layer 2 multiply, arriving at the output prediction. Arrows point one direction only."><g font-family="monospace" font-size="10">
<rect x="10" y="70" width="52" height="40" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="36" y="88" text-anchor="middle" fill="#5E5191">x</text><text x="36" y="102" text-anchor="middle" fill="#5E5191">[1,2,3]</text>
<text x="72" y="94" fill="#9A938A">→</text>
<rect x="86" y="70" width="66" height="40" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="119" y="88" text-anchor="middle" fill="#1a5c38">W₁·x+b₁</text><text x="119" y="102" text-anchor="middle" fill="#1a5c38">= z₁</text>
<text x="160" y="94" fill="#9A938A">→</text>
<rect x="174" y="70" width="54" height="40" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="1.8"/><text x="201" y="94" text-anchor="middle" fill="#9A5A12">f(z₁)=a₁</text>
<text x="236" y="94" fill="#9A938A">→</text>
<rect x="250" y="70" width="70" height="40" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="285" y="88" text-anchor="middle" fill="#1a5c38">W₂·a₁+b₂</text><text x="285" y="102" text-anchor="middle" fill="#1a5c38">= ŷ</text>
<text x="328" y="94" fill="#9A938A">→</text>
<rect x="346" y="65" width="140" height="50" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="416" y="86" text-anchor="middle" fill="#C93B3B" font-weight="bold">prediction ŷ</text><text x="416" y="102" text-anchor="middle" fill="#C93B3B">[0.75, -0.15] (raw)</text>
<text x="260" y="40" text-anchor="middle" fill="#3A342E" font-weight="bold" font-size="12">one direction, fixed order — the forward pass</text>
<text x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">length 3 → 4 → 2: multiply, bend, multiply → read the answer</text>
</g></svg>
%%%

#### The whole network in one honest line
Read that chain right to left, as boxes inside boxes, and the six moves squeeze into one line:

`ŷ = W₂·f₁(W₁·x + b₁) + b₂`

In plain words: *"take the first station's total, bend it, then run that through the second station's multiply-plus-bias."*

The inner `f₁(W₁·x + b₁)` is the hidden layer's output `a₁`. The outer `W₂·f₁(…) + b₂` is the raw answer. Reading a network as boxes inside boxes like this — functions nested inside functions — is called seeing it as a [[composition||A function made by feeding one function's output into the next. A network is a chain of layers composed together: the output of each becomes the input of the next.]].

It looks busy on the page. It is exactly the six steps you just walked. If your eyes slid off it the first time, that's the normal reaction to nested notation — trace the boxes in the picture instead, and the line reads itself.

%%% demo id=fwd label="run the full forward pass"
predict: the input has 3 numbers. How many come out the far end — 3, 4, or 2? And what happened to the list in between?
code: W1 = np.array([[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]]); b1 = np.array([0., -.5, 0., -2.])
code: W2 = np.array([[.5,.5,0.,.5],[-.5,0.,1.,.5]]);                    b2 = np.array([0., 0.])
code: x  = np.array([1., 2., 3.])
code: a1 = relu(W1 @ x + b1); yhat = W2 @ a1 + b2; (a1.shape, yhat)
out: ((4,), array([ 0.75, -0.15]))
take: <b>One trip down the line, one prediction out.</b> Length 3 in → length 4 across the hidden layer (bent by ReLU into `[1.5, 0, 0.6, 0]`) → length 2 at the output. Check the first answer yourself: `.5(1.5) + .5(0) + 0(0.6) + .5(0) = 0.75`. That is the whole nested line `ŷ = W₂·f₁(W₁·x + b₁) + b₂`, run start to finish, in one line of code.
%%%

!!! c-info 🪜
<b>Optional (skippable) — where's the output bend, and why is one answer negative?</b> Notice there's no <code>f₂</code> wrapped around the outside: today the output layer does its multiply-plus-bias and stops, handing back a <i>raw</i> list — which is why <code>-0.15</code> is allowed to be negative. Some networks do add a final bend (a sigmoid or a softmax) to turn that raw list into a clean probability, but that belongs with the <b>loss</b> lesson next, so we leave the output raw for now.
!!!

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it:</b> "The forward pass is the ordered chain input → (W·x + b) → activation → next layer → … → output. Each hidden dense layer is one matrix-vector multiply plus a bias, then an elementwise activation. The whole network is a composition of those layers — the output of one is the input of the next — and it runs strictly in one direction to produce a prediction." You can already say every piece of that.

**You just ran a whole network in your head.** Next: the first way this line jams — and it takes one line to fix.

@@@ concept id=c7 tag="Jam #1: bad fit" title="When the parts don't fit (shape mismatch)" gotit="Got the shape fix"
Here's a puzzle every single beginner runs into, usually within their first hour of writing a network. The good news: it's loud, honest, and takes one line to fix.

Picture holding a **4-prong plug in front of a 3-hole socket.** You don't need a manual. The counts don't match, so it simply won't seat, and you know instantly.

That's a [[shape mismatch||When one layer outputs a list of one length but the next layer was built to read a different length, so the matrix multiply can't line up. The program stops with a dimension error.]]: layer 1 hands forward a list of 4 numbers, but layer 2's grid was built with only 3 columns — so its multiply has nowhere to put the fourth number. The program stops and prints a "dimension mismatch" error.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A four-prong electrical plug held in front of a socket that has only three holes. The prong count and the hole count do not match, so the plug cannot seat — a visible, instant mismatch."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">4 prongs into a 3-hole socket — won't seat</text><rect x="40" y="46" width="70" height="72" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="75" y="40" text-anchor="middle" fill="#276b45" font-size="9">layer-1 plug</text><line x1="110" y1="58" x2="150" y2="58" stroke="#2D8B55" stroke-width="5"/><line x1="110" y1="74" x2="150" y2="74" stroke="#2D8B55" stroke-width="5"/><line x1="110" y1="90" x2="150" y2="90" stroke="#2D8B55" stroke-width="5"/><line x1="110" y1="106" x2="150" y2="106" stroke="#2D8B55" stroke-width="5"/><text x="75" y="86" text-anchor="middle" fill="#276b45" font-weight="bold">4 prongs</text><text x="178" y="86" text-anchor="middle" fill="#C93B3B" font-size="22">✗</text><rect x="210" y="50" width="84" height="64" rx="8" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="252" y="42" text-anchor="middle" fill="#C93B3B" font-size="9">layer-2 socket</text><circle cx="252" cy="66" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><circle cx="252" cy="82" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><circle cx="252" cy="98" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><text x="278" y="86" fill="#C93B3B" font-weight="bold">3 holes</text><text x="167" y="140" text-anchor="middle" fill="#C93B3B" font-size="10">4 ≠ 3 → you notice instantly (loud error)</text><rect x="330" y="52" width="172" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="416" y="74" text-anchor="middle" fill="#276b45">fix: match the counts</text><text x="416" y="92" text-anchor="middle" fill="#276b45" font-size="10">columns = previous layer's neurons</text></g></svg>
%%%

**What the plug-and-socket picture gets right:** mismatched counts can't connect, and you find out right away. **Where it breaks down:** a plug is a physical thing you can't force, while this mismatch is a rule the computer checks — same idea, but it fails as a printed message, not a physical block.

%%% insight
Be glad this one is loud. A crash with a line number is the friendliest kind of bug there is: it tells you *exactly* where the parts don't fit. The other jam waiting for you today says nothing at all — and quietly wastes your afternoon. Loud beats silent, every time.
%%%

#### The habit that never lets it happen
The fix is the shape rule from unit 2, used as a checklist. Engineers call this **shape bookkeeping**, and it's genuinely the first thing a professional does when a network won't run.

%%% steps
step: write the sizes down BEFORE writing any code — 3 → 4 → 2
why: that little arrow chain is your map; every grid in the network has to agree with it
step: for each layer, set rows = the number of neurons in THIS layer
why: rows are the workers, so rows decide how many numbers roll out
step: set columns = the previous layer's neuron count
why: columns are the sockets, so columns must equal the number of prongs arriving
step: print the shape after every layer while the code runs
why: therefore a mismatch shows up at the exact station where it starts, not three layers later
%%%

**So far:** rows are what a layer *produces*, columns are what it *accepts*. Get those two right at every joint and the parts always fit — as the picture below shows, the whole fix is "give layer 2 a fourth column".

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A layer outputs a length-4 list, but the next layer's weight grid has only 3 columns, so they cannot connect. The fix: make columns equal the previous layer's neuron count, giving layer 2 four columns."><g font-family="monospace" font-size="11">
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

One jam solved, and you fixed it with a rule you already had — that's your first professional debugging habit, earned on day 3. Now for the fun part: what all this stacking actually *buys* you.

@@@ concept id=c8 tag="Why depth wins" title="Stacked layers build bigger ideas" gotit="Got depth"
Time for the payoff. Why bother with a second station at all — why not one enormous row of workers?

Think about building with **LEGO.** You never build a house straight out of loose studs. You snap a few studs into a **brick**. You line bricks up into a **wall**. You stand walls together into a **room**, and rooms into a **house**. Each stage builds from the pieces the stage before it made — and that's the only reason a kid can build something as complicated as a house at all.

Layers do exactly that. This is [[depth||Having several hidden layers in a row. With an activation between them, each layer builds richer patterns from the last — this is where a network's power comes from.]]: several hidden layers in a row, each one drawing richer patterns from the last.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A LEGO build-up in four stages: loose studs snap into a single brick, bricks line up into a wall, walls stand together into a room, and rooms make a house. Each stage is built from the pieces the previous stage made."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">studs → brick → wall → house: each stage uses the last stage's pieces 🧱</text><circle cx="30" cy="70" r="5" fill="#C99A12" stroke="#9A5A12"/><circle cx="46" cy="70" r="5" fill="#C99A12" stroke="#9A5A12"/><circle cx="38" cy="86" r="5" fill="#C99A12" stroke="#9A5A12"/><text x="38" y="120" text-anchor="middle" fill="#9A5A12">studs</text><text x="38" y="134" text-anchor="middle" fill="#9A938A" font-size="9">the input</text><text x="72" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="94" y="62" width="60" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><circle cx="106" cy="60" r="4" fill="#FCF3DC" stroke="#C99A12"/><circle cx="124" cy="60" r="4" fill="#FCF3DC" stroke="#C99A12"/><circle cx="142" cy="60" r="4" fill="#FCF3DC" stroke="#C99A12"/><text x="124" y="120" text-anchor="middle" fill="#9A5A12">one brick</text><text x="124" y="134" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 1</text><text x="168" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="190" y="48" width="100" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="190" y="68" width="100" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><text x="240" y="120" text-anchor="middle" fill="#276b45">a wall</text><text x="240" y="134" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 2</text><text x="304" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="336" y="62" width="86" height="30" rx="3" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.8"/><path d="M330 62 L379 34 L428 62 Z" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.8"/><rect x="368" y="74" width="22" height="18" fill="#FFFFFF" stroke="#7C6DAA"/><text x="379" y="120" text-anchor="middle" fill="#5E5191">a house 🏠</text><text x="379" y="134" text-anchor="middle" fill="#9A938A" font-size="9">the answer</text><text x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="10">no stage skips ahead — each one is only possible because of the one before it</text></g></svg>
%%%

**What the LEGO picture gets right:** big things are built from middle-sized things, which are built from small things — no stage can skip ahead. **Where it breaks down:** *you* decide what a LEGO wall should look like, but nobody tells a hidden layer what to build. It invents its own bricks from the data, which is stranger and much more powerful.

%%% insight
This is why a photo app can look at a grid of raw pixels and say "cat". No single row of workers could ever answer that from raw dots. But a stack can: dots → edges → an eye, an ear, a whisker → cat. Each layer re-describes what the layer below handed it, in slightly bigger words. That re-description *is* what "deep" actually buys you.
%%%

#### Watch the ideas grow, stage by stage
Take a photo of a face and follow the assembly line up.

%%% steps
step: hidden layer 1 finds tiny edges — a light-dark border here, a corner there
why: it can only see raw pixels, so tiny local contrasts are all it CAN find
step: hidden layer 2 combines those edges into parts — a curve of an eyelid, the line of a nose
why: therefore layer 2 never touches a pixel; it works on layer 1's edge-words, which is why its ideas are bigger
step: hidden layer 3 combines parts into a whole face
why: two eyes above a nose above a mouth is a pattern in PART-language, unsayable in pixel-language
step: the output layer reads that description and answers "this is Sam"
why: the answer was easy — because three layers of re-describing made it easy
%%%

**So far:** each layer speaks a slightly bigger language than the one below. Here is that growth drawn in one figure.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A feature hierarchy in four stages. Raw pixels become small edges, edges become face parts like an eye and a nose, parts become a whole face, and the output names the person. Each stage is drawn bigger and more meaningful than the last."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Each layer re-describes the last in slightly bigger words</text><rect x="18" y="42" width="70" height="70" rx="4" fill="#F3F0EA" stroke="#C0B8AC"/><g fill="#B8AEA2"><rect x="24" y="48" width="12" height="12"/><rect x="48" y="48" width="12" height="12"/><rect x="36" y="66" width="12" height="12"/><rect x="60" y="78" width="12" height="12"/><rect x="24" y="90" width="12" height="12"/></g><text x="53" y="128" text-anchor="middle" fill="#6B645E">raw pixels</text><text x="53" y="142" text-anchor="middle" fill="#9A938A" font-size="9">the input x</text><text x="98" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="122" y="42" width="70" height="70" rx="4" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.6"/><path d="M130 100 L160 55" stroke="#5E5191" stroke-width="2" fill="none"/><path d="M150 100 L184 70" stroke="#5E5191" stroke-width="2" fill="none"/><path d="M132 62 L176 62" stroke="#5E5191" stroke-width="2" fill="none"/><text x="157" y="128" text-anchor="middle" fill="#5E5191">edges</text><text x="157" y="142" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 1</text><text x="202" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="226" y="42" width="70" height="70" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><ellipse cx="252" cy="66" rx="14" ry="8" fill="#FFF" stroke="#2D8B55" stroke-width="1.6"/><circle cx="252" cy="66" r="3" fill="#2D8B55"/><path d="M272 82 L272 100 L282 100" stroke="#2D8B55" stroke-width="1.8" fill="none"/><text x="261" y="128" text-anchor="middle" fill="#276b45">parts</text><text x="261" y="142" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 2</text><text x="306" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="330" y="42" width="70" height="70" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><circle cx="365" cy="77" r="26" fill="#FFF" stroke="#C99A12" stroke-width="1.6"/><circle cx="356" cy="70" r="3" fill="#9A5A12"/><circle cx="374" cy="70" r="3" fill="#9A5A12"/><path d="M356 88 Q365 94 374 88" stroke="#9A5A12" stroke-width="1.6" fill="none"/><text x="365" y="128" text-anchor="middle" fill="#9A5A12">a face</text><text x="365" y="142" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 3</text><text x="410" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="430" y="60" width="76" height="34" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="468" y="82" text-anchor="middle" fill="#C93B3B" font-weight="bold">"it's Sam"</text><text x="468" y="128" text-anchor="middle" fill="#9A938A" font-size="9">output layer</text><text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">pixels → edges → parts → a face → a name</text></g></svg>
%%%

#### Your turn — try to beat one layer, then give it a second
Here's the classic 60-second experiment that made the whole field want depth. Four dots. Two classes, sitting on opposite diagonals (this pattern has a name: **XOR**).

- **Drag the three sliders** and try to place ONE straight line so both blue dots are on one side and both orange dots are on the other. Really try. The best anyone ever manages is 3 out of 4.
- Now **tick "add a hidden layer"** — a second line appears, and the two lines together fence off exactly the right region. Four out of four.

%%% viz src=../../viz/xor-limit.html title="one line versus a hidden layer" caption="interactive — drag w1, w2, b to move your single dividing line (you cannot get all four dots right), then tick add a hidden layer and watch a second line appear"
%%%

That little failure is real history. In 1969, Marvin Minsky and Seymour Papert showed that a single layer of neurons cannot express XOR — and interest in neural networks cooled for years. The way out was the thing you just clicked: add a hidden layer.

**Victory lap.** You just re-ran, in one minute, the experiment that explains why every modern network is deep. Several hidden layers, each drawing richer patterns from the last, can build almost anything, while one layer alone only draws simple shapes. Depth is the whole game — with one condition attached, which is the next unit.

@@@ concept id=c9 tag="Jam #2: the quiet one" title="Layers with no bend fold into one" gotit="Got the collapse fix"
So depth is the superpower. Here's the one condition attached to it — and it's a single line of code, which is a wonderful price for that much power. It's yesterday's idea, now seen on the whole assembly line.

Picture a line where **every station is just a conveyor belt that stretches the part.** Station 1 doubles its length. Station 2 triples it. Run a part through both and it comes out six times longer.

But hold on — you didn't need two stations. **One** belt set to "×6" does the identical job. Stack a hundred such belts and you still only get one stretch.

That's the [[linear collapse||When you stack layers with no activation between them, the whole stack behaves like a single layer. All the depth is wasted, because straight-line steps combine into one straight-line step.]]: layers with no bend between them fold into a single layer, and all your hard-won depth vanishes.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Two conveyor belts in a row that only scale a bar's length. The first stretches it two times, the second three times. A dashed arrow shows one single belt set to six times does the exact same job — two belts collapse into one."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two scaling belts (×2 then ×3) = one belt set to ×6</text><rect x="20" y="56" width="22" height="10" rx="2" fill="#7C6DAA"/><text x="31" y="48" text-anchor="middle" fill="#5E5191" font-size="9">part</text><rect x="52" y="48" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><text x="97" y="65" text-anchor="middle" fill="#276b45">belt ×2</text><rect x="150" y="56" width="44" height="10" rx="2" fill="#7C6DAA"/><rect x="200" y="48" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><text x="245" y="65" text-anchor="middle" fill="#276b45">belt ×3</text><rect x="298" y="56" width="132" height="10" rx="2" fill="#7C6DAA"/><text x="364" y="48" text-anchor="middle" fill="#5E5191" font-size="9">6× longer</text><text x="445" y="64" fill="#C93B3B">=</text><rect x="20" y="104" width="22" height="10" rx="2" fill="#7C6DAA"/><rect x="52" y="96" width="238" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.8"/><text x="171" y="113" text-anchor="middle" fill="#C93B3B" font-weight="bold">one belt ×6 — same result</text><rect x="298" y="104" width="132" height="10" rx="2" fill="#7C6DAA"/><text x="210" y="144" text-anchor="middle" fill="#C93B3B" font-size="10">no bend → two stations do the work of one (depth wasted)</text><text x="260" y="164" text-anchor="middle" fill="#2D8B55" font-size="10">a bend between them → they stay two real layers</text></g></svg>
%%%

**What the conveyor-belt picture gets right:** plain scaling steps really do multiply into a single scaling, so extra stations add nothing. **Where it breaks down:** a belt only stretches, while a layer also *mixes and shifts* the numbers — but the punchline survives: with no bend, all that mixing still folds into one single mixing step.

%%% insight
Here's why this one bites so hard. Nothing breaks. The code runs, the shapes fit, a number comes out, the loss even goes down a little. Your 5-layer network is quietly doing the work of one layer, and nothing on your screen says so. That's why we're going to *see* the collapse with our own eyes instead of trusting it.
%%%

#### Catch the collapse with real numbers
Let's shrink the line to a size you can check on paper. Two tiny stations — call their grids **A** and **B** (same roles as `W₁` and `W₂`, just smaller) — and one 2-number input `u`, with biases left at 0 to keep it clean:

`u = [−3, 1]`, `A = [[1, 2], [0, 1]]`, `B = [[1, 0], [3, 1]]`

%%% steps
step: run station A — A·u = [1(−3) + 2(1), 0(−3) + 1(1)] = [−1, 1]
why: ordinary weighted sums, exactly like unit 3; no bend applied, because that's the mistake we're studying
step: run station B — B·[−1, 1] = [1(−1) + 0(1), 3(−1) + 1(1)] = [−1, −2]
why: two full stations done, and [−1, −2] is this little network's answer
step: now the shortcut — multiply the two grids FIRST: B·A = [[1, 2], [3, 7]]
why: one grid that claims to do both stations' work in a single move
step: run that single grid once — [[1,2],[3,7]]·[−3, 1] = [1(−3)+2(1), 3(−3)+7(1)] = [−1, −2]
why: the same answer, therefore the two stations really were one station in disguise
step: name it — this is the collapse
why: two layers of effort, one layer of power, and not one error message to warn you
step: now the fix — bend the middle list before passing it on: relu([−1, 1]) = [0, 1], then B·[0, 1] = [0, 1]
why: [0, 1] is nothing like [−1, −2], so with a bend the two stations are finally doing two different jobs
%%%

**So far:** without a bend, two paths land on the same answer; with a bend, they part company. Here is that whole story in one figure — collapse on top, the fix underneath.

%%% svg
<svg viewBox="0 0 520 260" role="img" aria-label="Top: input u equals minus three, one, times A gives minus one, one, times B gives minus one, minus two; and the combined grid B times A also gives minus one, minus two, so the two paths match and the layers collapse. Bottom: with a ReLU bend the middle list becomes zero, one, and the answer becomes zero, one, which no longer matches the combined grid."><g font-family="monospace" font-size="11">
<text x="18" y="20" fill="#C93B3B" font-weight="bold" font-size="12">NO bend — the two paths match (collapse)</text>
<rect x="18" y="30" width="58" height="32" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="47" y="50" text-anchor="middle" fill="#5E5191">u=[-3,1]</text>
<text x="82" y="50" fill="#9A938A">→</text>
<rect x="98" y="30" width="74" height="32" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="135" y="44" text-anchor="middle" fill="#1a5c38">A·u</text><text x="135" y="57" text-anchor="middle" fill="#1a5c38">=[-1,1]</text>
<text x="178" y="50" fill="#9A938A">→</text>
<rect x="194" y="30" width="92" height="32" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="240" y="44" text-anchor="middle" fill="#1a5c38">B·[-1,1]</text><text x="240" y="57" text-anchor="middle" fill="#1a5c38">=[-1,-2]</text>
<text x="292" y="50" fill="#9A938A">→</text>
<rect x="308" y="26" width="96" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="356" y="50" text-anchor="middle" fill="#C93B3B" font-weight="bold">[-1, -2]</text>
<rect x="98" y="76" width="188" height="30" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="1.8"/><text x="192" y="95" text-anchor="middle" fill="#9A5A12">one grid B·A = [[1,2],[3,7]]</text>
<text x="292" y="95" fill="#9A938A">→</text>
<rect x="308" y="76" width="96" height="30" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="356" y="96" text-anchor="middle" fill="#C93B3B" font-weight="bold">[-1, -2]</text>
<text x="416" y="70" fill="#C93B3B" font-weight="bold">same!</text><text x="416" y="86" fill="#6B645E" font-size="10">two layers</text><text x="416" y="100" fill="#6B645E" font-size="10">= one layer</text>
<line x1="18" y1="126" x2="502" y2="126" stroke="#E4DED5" stroke-width="1"/>
<text x="18" y="150" fill="#2D8B55" font-weight="bold" font-size="12">WITH a bend — the match breaks (depth is real)</text>
<rect x="18" y="164" width="58" height="32" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="47" y="184" text-anchor="middle" fill="#5E5191">u=[-3,1]</text>
<text x="82" y="184" fill="#9A938A">→</text>
<rect x="98" y="164" width="74" height="32" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="135" y="178" text-anchor="middle" fill="#1a5c38">A·u</text><text x="135" y="191" text-anchor="middle" fill="#1a5c38">=[-1,1]</text>
<text x="178" y="184" fill="#9A938A">→</text>
<rect x="194" y="164" width="92" height="32" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="2"/><text x="240" y="178" text-anchor="middle" fill="#9A5A12">relu → [0,1]</text><text x="240" y="191" text-anchor="middle" fill="#9A5A12" font-size="9">the -1 is flattened</text>
<text x="292" y="184" fill="#9A938A">→</text>
<rect x="308" y="160" width="96" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="356" y="184" text-anchor="middle" fill="#1a5c38" font-weight="bold">[0, 1]</text>
<text x="416" y="178" fill="#2D8B55" font-weight="bold">different!</text><text x="416" y="194" fill="#6B645E" font-size="10">no single grid</text><text x="416" y="208" fill="#6B645E" font-size="10">can do this</text>
<text x="260" y="238" text-anchor="middle" fill="#6B645E" font-size="10">the ONE thing that changed: a bend on the middle list</text>
</g></svg>
%%%

%%% mathladder
title: why two layers with no bend fold into one
words: doing station A and then station B is the same as doing one station whose grid is B·A
formula: B·(A·u) = (B·A)·u
numbers: u = [-3, 1], A = [[1,2],[0,1]], B = [[1,0],[3,1]] → both paths give [-1, -2]
sanity: put a bend in the middle and the match breaks — B·relu(A·u) = [0, 1]
%%%

%%% hint
t1: Feeling lost on how two grids become one? Start smaller. Forget the whole grid — just track ONE number through both stations and ask whether a single station could have done the same job.
t2: Take u = [-3, 1]. Station A: A·u = [1(-3) + 2(1), 0(-3) + 1(1)] = [-1, 1]. Station B: B·[-1, 1] = [-1, -2]. Now the shortcut: multiply the grids first, B·A = [[1, 2], [3, 7]], and run that once on u — you get [-1, -2] too. Same answer, one station.
t3: With no bend, doing A then B is the exact same math as one combined grid B·A — so the two layers are secretly one, and all the depth is wasted.
%%%

Don't take the picture's word for it. **Guess first, then reveal.**

%%% demo id=collapse label="run: two linear layers vs one combined grid vs one bend"
predict: three outputs come back. Which two will be identical — and will the ReLU one join them or break away?
code: A = np.array([[1,2],[0,1]]); B = np.array([[1,0],[3,1]]); u = np.array([-3,1])
code: no_bend = B @ (A @ u); combined = (B @ A) @ u; with_bend = B @ relu(A @ u)
code: (B @ A).tolist(), no_bend.tolist(), combined.tolist(), with_bend.tolist()
out: ([[1, 2], [3, 7]], [-1, -2], [-1, -2], [0, 1])
take: <b>The first two match; the bend breaks away.</b> With no bend, `A` then `B` is exactly the single grid `B·A = [[1,2],[3,7]]` — both give `[-1, -2]`, so the two layers really did fold into one. Add the bend and you get `[0, 1]`: a different answer that no single grid can produce. One line of code, and your depth is real again.
%%%

!!! c-info 🪜
<b>Optional (skippable) — when the bend looks like it does nothing.</b> Try the same demo with <code>u = [1, 2]</code>. Now <code>A·u = [5, 2]</code> — both entries are already positive, so ReLU has nothing to flatten and the bent answer matches the collapsed one exactly. The bend only changes the result once some middle number goes <b>negative</b>. So when you test for the collapse, pick an input that drives a hidden total below zero — otherwise your test passes for the wrong reason.
!!!

#### The fix you already own
The remedy is yesterday's star: **put a bend** — the activation `f` — between the layers. That elementwise `a = f(z)` step from unit 4 is not decoration; it is the thing that stops the collapse. With a bend between the layers, `A` and `B` can no longer be multiplied into one grid, so the two stations stay two genuinely different transforms.

!!! c-warn 🤫
<b>Failure mode (silent):</b> stack linear layers with no activation and nothing breaks — the code runs, the shapes fit, a number comes out. But your 5-layer network secretly has the power of <i>one</i> layer, so it learns almost nothing and you can't tell why. The tell is a model that stays stubbornly weak no matter how many layers you add. The fix is one line: an activation between every pair of layers.
!!!

Depth is an illusion *without* the bend, and a superpower *with* it. That single trade is the whole reason "deep" learning is called deep — and you just proved it with two tiny grids and one multiply.

@@@ concept id=c10 tag="The honest edge" title="A forward pass guesses — training comes next" gotit="Got the edge"
One honest thing to say about today's machine, and it's the most exciting sentence in the lesson — because it's the door into everything next.

Picture a brand-new **vending machine that's plugged in but never stocked.** Press a button and it *will* dispense something — the mechanism is flawless. But nobody loaded the right snacks, so whatever drops out is basically random.

Today's forward pass is that machine. It runs perfectly and always produces an answer. But the weights `W` and biases `b` are still whatever they started as, so the answer is **just a guess.** The forward pass *uses* the weights; it never *changes* them.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A vending machine plugged into the wall with its light on. A hand presses a button and an item drops into the tray, but the shelves are empty or randomly filled, so whatever drops out is a random guess."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Plugged in, works — but never stocked → random drop</text><rect x="150" y="30" width="120" height="128" rx="8" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="210" y="46" text-anchor="middle" fill="#5E5191" font-size="9">💡 powered on</text><rect x="160" y="52" width="70" height="58" rx="3" fill="#FFFFFF" stroke="#B8AEDA"/><line x1="160" y1="71" x2="230" y2="71" stroke="#E5DFD6"/><line x1="160" y1="90" x2="230" y2="90" stroke="#E5DFD6"/><line x1="195" y1="52" x2="195" y2="110" stroke="#E5DFD6"/><text x="177" y="66" text-anchor="middle" fill="#B8AEA2" font-size="9">empty</text><text x="213" y="85" text-anchor="middle" fill="#B8AEA2" font-size="9">?</text><text x="177" y="104" text-anchor="middle" fill="#B8AEA2" font-size="9">?</text><circle cx="248" cy="66" r="5" fill="#FCF3DC" stroke="#C99A12"/><circle cx="248" cy="82" r="5" fill="#FCF3DC" stroke="#C99A12"/><text x="245" y="98" fill="#9A5A12" font-size="9">👆</text><rect x="160" y="128" width="70" height="22" rx="3" fill="#FDE8E8" stroke="#C93B3B"/><text x="195" y="143" text-anchor="middle" fill="#C93B3B" font-size="9">random drop 🎲</text><text x="110" y="70" text-anchor="middle" fill="#6B645E" font-size="10">press →</text><text x="300" y="70" fill="#6B645E" font-size="10">it runs...</text><text x="300" y="88" fill="#C93B3B" font-size="10">...but the answer</text><text x="300" y="104" fill="#C93B3B" font-size="10">is just a guess</text><text x="300" y="124" fill="#9A938A" font-size="9">restocking =</text><text x="300" y="138" fill="#9A938A" font-size="9">training (later)</text></g></svg>
%%%

**What the vending-machine picture gets right:** the machine works whether or not it's stocked well, just as the forward pass runs whether or not the weights are any good. **Where it breaks down:** you restock a vending machine by hand, but a network restocks *itself* through training — a clever backward trip we haven't taken yet.

%%% insight
Read this as a preview, not a complaint. Every piece of machinery you built today is exactly what training needs in order to work: it *scores* the guess you just produced, then walks back down the same line nudging every `W` and `b` a little. You didn't build half a network today — you built the half that has to exist first.
%%%

#### The missing arrow, drawn as a ghost
Getting *good* weights means nudging `W` and `b` toward better answers, which needs a trip in the opposite direction — **gradient descent and backpropagation**, coming on the training days. Look at the dashed arrow below: it's drawn as a ghost, because today it doesn't exist yet. Filling it in is the rest of this module.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="The forward pass produces a prediction from fixed weights, but there is no arrow going back to update the weights. That backward, weight-updating trip is training, coming later."><g font-family="monospace" font-size="11">
<rect x="20" y="55" width="60" height="40" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="50" y="79" text-anchor="middle" fill="#5E5191">input</text>
<text x="90" y="79" fill="#2D8B55">→</text>
<rect x="108" y="50" width="120" height="50" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="168" y="70" text-anchor="middle" fill="#1a5c38">forward pass</text><text x="168" y="87" text-anchor="middle" fill="#1a5c38" font-size="10">(fixed W, b)</text>
<text x="236" y="79" fill="#2D8B55">→</text>
<rect x="254" y="55" width="90" height="40" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="299" y="79" text-anchor="middle" fill="#C93B3B">a guess</text>
<path d="M299 100 C 299 130, 168 130, 168 102" fill="none" stroke="#C0B8AC" stroke-width="1.6" stroke-dasharray="5,4"/><text x="235" y="145" text-anchor="middle" fill="#9A938A" font-size="10">updating W back this way = training (starts next lesson)</text>
</g></svg>
%%%

Knowing exactly where the forward pass stops and training begins *is* the win — that's the map for every day ahead. One page to gather it all, and you're done.

@@@ concept id=c11 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built a whole working machine today.

Here's a fresh way to hold it all at once: think of the day like the **layers of a sandwich you just made.** The bottom slice of bread is your input. Each filling you add — cheese, then tomato, then lettuce — is one layer's work stacked on the last. The top slice you press down is the final prediction. To describe your sandwich to a friend you name the fillings in order, bottom to top — exactly how you read a network, input to answer.

**Where the sandwich picture breaks down:** fillings just sit there, while each layer actively *reshapes* what the layer below handed it (and later even retunes itself during training).

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A stacked sandwich seen from the side. Bottom slice of bread is the input, then a cheese layer, a tomato layer, a lettuce layer stacked in order, and the top slice pressed down is the final prediction — read bottom to top."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Read a network like a sandwich — bottom to top 🥪</text><rect x="120" y="40" width="180" height="18" rx="9" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><text x="330" y="53" fill="#9A5A12" font-size="10">← top slice = prediction ŷ</text><rect x="128" y="58" width="164" height="16" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="330" y="70" fill="#276b45" font-size="10">← lettuce = output layer</text><rect x="128" y="74" width="164" height="16" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.4"/><text x="330" y="86" fill="#C93B3B" font-size="10">← tomato = hidden layer 2</text><rect x="128" y="90" width="164" height="16" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.4"/><text x="330" y="102" fill="#9A5A12" font-size="10">← cheese = hidden layer 1</text><rect x="120" y="106" width="180" height="18" rx="9" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><text x="330" y="119" fill="#5E5191" font-size="10">← bottom slice = input x</text><text x="210" y="146" text-anchor="middle" fill="#6B645E" font-size="10">name the fillings in order = read layers input → answer</text></g></svg>
%%%

#### The seven beats of today, in order
- **1 · The layer.** A row of neurons that all read the same input, each with its own weights and bias — graders marking one test for different things.
- **2 · The settings.** All the weights live in one grid **W** (rows = neurons, columns = inputs); the biases in one list **b** (one bias per neuron).
- **3 · The multiply.** One move — `z = W·x + b` — gives every neuron's weighted sum at once. Fast, and built for GPUs.
- **4 · The bend.** Apply the activation to each entry on its own: `a = f(z)`. That's one complete hidden layer, in two moves.
- **5 · The chain.** Each layer's output feeds the next; input → hidden → output. Sizes must match: columns = the previous layer's neuron count.
- **6 · The full trip.** The **forward pass** runs the chain one direction, start to finish. Our network, output left raw: `ŷ = W₂·f₁(W₁·x + b₁) + b₂`, which took `[1, 2, 3]` to `[0.75, -0.15]`.
- **7 · The payoff.** **Depth** — several hidden layers, each drawing richer patterns from the last — is what lets pixels become "a face". One layer alone can't even split XOR.

#### The two jams, and the honest edge
- **Shape mismatch** (loud — it crashes) → set columns = the previous layer's neurons, and print shapes as you go.
- **Linear collapse** (silent — it just wastes depth) → put a bend between every pair of layers. Test it with an input that drives a middle number negative, or the bend will look like it does nothing.
- **The edge:** the forward pass *guesses*, it doesn't *learn*. That's training, and it starts next.

Here's the one picture to keep.

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
depth | several hidden layers with bends between them — where a network's power comes from
XOR | the four-dot pattern no single layer can split; the classic reason hidden layers exist
shape bookkeeping | writing the sizes down and checking every joint before you run
shape mismatch | the loud jam: a layer's input length doesn't match the previous layer's output
linear collapse | the silent jam: layers with no bend fold into one, wasting depth
%%%

That's the whole day. Next you'll learn how to *score* the network's guess and start teaching it to improve — the **loss** function, and the road to training.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: A layer has 5 neurons and each neuron reads a 3-number input. What is the shape of its weight matrix W? | a:1 | [3, 5] | [5, 3] | [5, 5] | [3, 3] | fb: Rows = neurons, columns = inputs. 5 neurons and 3 inputs give a [5, 3] grid — one row per neuron, one column per input feature.
q: In one dense layer, the correct order of the two moves is: | a:0 | multiply first (z = W·x + b), then bend (a = f(z)) | bend first, then multiply | bend, then bend again | multiply, then multiply again | fb: A layer first computes z = W·x + b (mixing the inputs), then applies the activation elementwise: a = f(z).
q: You stack 6 layers but put NO activation between them. The network runs fine and gives numbers, yet it barely learns anything. Why? | a:2 | The shapes don't match, so it silently skips layers | It ran out of GPU memory | With no bend between them, the linear layers collapse into a single layer — so 6 layers have the power of 1, and depth is wasted; the fix is an activation between each pair | Six layers is always too many | fb: No non-linearity means W₁…W₆ multiply into one grid — the whole stack acts like one layer. Insert an activation between layers and they stay distinct.
q: After a forward pass, why is the network's output usually a bad guess at first? | a:1 | The forward pass has a bug | The weights W and biases b start untrained — the forward pass USES them but never updates them, so the answer is basically random until training (gradient descent) tunes them | The activation is missing | The input was the wrong shape | fb: The forward pass only runs the numbers through fixed weights. Making those weights good is training (gradient descent + backpropagation), which comes later.
%%%

@@@ produce id=produce tag="Produce" title="Push a number down the whole line — and watch it collapse without a bend" gotit="Done"
Time to see today's assembly line run with your own eyes. You'll build the exact 2-layer network from this lesson and push `x = [1, 2, 3]` all the way down to a prediction — then run the sneaky experiment: rip the bend out and watch two layers collapse into one. **Predict first:** with NO bend between them, will a 2-layer network give the *same* answer as a single cleverly-chosen layer, or a different one? Then run it and **watch.** Seeing them match without the bend, and part ways the moment you add it, is the whole lesson in a few lines of output. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py`. Build a forward pass by hand:
1. Make a layer function `layer(x, W, b, f)` that returns `f(W @ x + b)` — one multiply-plus-bias, then an elementwise bend. Print the shape of `W @ x + b` inside it so you can **watch** the vector length change from station to station.
2. Set up today's network: `x = [1, 2, 3]` → hidden layer of 4 neurons (`W1 = [[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]]`, `b1 = [0, -.5, 0, -2]`, bend = `relu`) → output layer of 2 neurons (`W2 = [[.5,.5,0,.5],[-.5,0,1,.5]]`, `b2 = [0, 0]`, no activation on the output — the raw prediction, as in the lesson). **Notice** that the hidden layer's 4 outputs must equal the output layer's 4 inputs — that's the chaining rule.
3. Run the full forward pass and print `a1` and the final length-2 prediction. You should see `a1 = [1.5, 0, 0.6, 0]` and `yhat = [0.75, -0.15]` — check `0.75` by hand and confirm the page wasn't lying to you.
4. The collapse demo, on the tiny pair from the lesson: `A = [[1,2],[0,1]]`, `B = [[1,0],[3,1]]`, `u = [-3, 1]`. Show that `B @ (A @ u)` equals `(B @ A) @ u` exactly — two linear layers really are one. Then put `relu` in the middle and show `B @ relu(A @ u)` is **different**. One trap to avoid: your input must drive a middle number **negative** (that's why `u = [-3, 1]` and not `[1, 2]`), otherwise ReLU has nothing to flatten and your test would pass for the wrong reason. Run with `python3 sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 3 artifact.

Create sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py that, with a comment on each step:
1. Defines relu(z)=np.maximum(0,z), identity(z)=z, and a layer(x, W, b, f) that returns f(W @ x + b); prints the shape of W @ x + b at each call.
2. Builds this exact 2-layer forward pass: x = [1., 2., 3.]; hidden W1 = [[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]], b1 = [0., -.5, 0., -2.], activation relu; output W2 = [[.5,.5,0.,.5],[-.5,0.,1.,.5]], b2 = [0., 0.], no activation (raw prediction). Print the lengths 3 -> 4 -> 2, print a1 and yhat, and assert np.allclose(a1, [1.5, 0., 0.6, 0.]) and np.allclose(yhat, [0.75, -0.15]).
3. Shows the linear collapse on a tiny pair: A = [[1,2],[0,1]], B = [[1,0],[3,1]], u = [-3, 1]. Print B @ A, then assert np.array_equal(B @ (A @ u), (B @ A) @ u) — two linear layers equal one layer whose grid is B @ A.
4. Then put relu between them and show the collapse breaks: assert not np.array_equal(B @ relu(A @ u), (B @ A) @ u). IMPORTANT: the test input must make A @ u contain a NEGATIVE entry (u = [-3, 1] gives A @ u = [-1, 1]), otherwise relu changes nothing and this assert would pass vacuously. Also print what happens with u = [1, 2] (A @ u = [5, 2], both positive) to show the bend changing nothing there, and say in a comment why.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the trip, then the collapse)
- The vector length changes as it flows: length **3** in → **4** after the hidden layer → **2** at the output. Watching the length change is watching each station reshape the part.
- The hidden bend bites: `z1 = [1.5, -0.2, 0.6, -0.1]` becomes `a1 = [1.5, 0, 0.6, 0]`, and the final raw prediction is `[0.75, -0.15]`.
- With **no** bend, the two-layer output `[-1, -2]` matches the single combined grid `B @ A = [[1,2],[3,7]]` exactly — two linear layers really did collapse into one.
- The moment to **watch**: drop `relu` into the middle and the answer jumps to `[0, 1]`, which no single grid can produce. That break is depth earning its keep — the same lesson as Day 2, now on a full network.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-03-layers-forward-pass/log.md`: (1) the two moves inside one layer, in order, and the shape rule for W; (2) what the forward pass is, in one sentence, and why its first guess is usually bad; (3) the two ways the line jams (shape mismatch, linear collapse) and the one-line fix for each.
!!!

@@@ fin
