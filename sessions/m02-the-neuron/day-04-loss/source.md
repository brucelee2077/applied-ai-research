---
quest_id: wf3-d01-loss
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 4 — Loss Functions"
module_label: "Module 2 · Train · Day 4"
title: "Loss Functions"
subtitle: "How the Network Measures Its Own Mistakes"
brand_sub: "Foundations · M2 Day 4"
spine: "golf score"
nav_prev_href: "../review-part-a.html"
nav_prev_label: "M2 · Review Gate"
nav_next_href: "../day-05-gradients-backprop/lesson.html"
nav_next_label: "Gradients & Backpropagation"
fin_title: "Module 2 · Day 4 complete! 🏆"
fin_body: "Nice work — you've completed <b>Loss Functions</b>. You can now put a single number on how wrong a guess is — the score the whole network is about to chase down.<br>Next up: <b>Gradients & Backpropagation</b> — how the network figures out which way to improve."
notebook_yardstick: 00-neural-networks/fundamentals/06_loss_functions.ipynb
coverage_topics:
  - {topic: what a loss function is, keywords: [one honest number, how wrong]}
  - {topic: prediction vs target pairing, keywords: [prediction, target, how wrong]}
  - {topic: mean squared error, keywords: [mean squared error, squared misses, "mse ="]}
  - {topic: mse squaring property, keywords: [square, big misses count more, always positive]}
  - {topic: mean absolute error, keywords: [mean absolute error, plain miss sizes, absolute value]}
  - {topic: cross-entropy log loss, keywords: [cross-entropy, log loss, prob of the correct]}
  - {topic: cross-entropy punishes confident wrong, keywords: [confidently wrong, "-log", shoots up]}
  - {topic: binary cross-entropy, keywords: [binary cross-entropy, yes/no, sigmoid]}
  - {topic: categorical cross-entropy softmax, keywords: [categorical cross-entropy, softmax, sum to 1]}
  - {topic: logits raw scores, keywords: [logits, raw output scores]}
  - {topic: loss averaged over dataset, keywords: [average, cost, whole batch]}
  - {topic: loss must be smooth differentiable, keywords: [smooth, slope, "0/1 loss", counting mistakes]}
  - {topic: mse+sigmoid learns slowly, keywords: [stall, flat sigmoid slope, confidently wrong]}
  - {topic: remedy bce for slow mse, keywords: [cross-entropy is the fix, push stays]}
  - {topic: outlier dominates mse, keywords: [outlier, hijack]}
  - {topic: remedy mae, keywords: [switch to mae, huber]}
  - {topic: cross-entropy log(0) blow-up, keywords: ["log(0)", "-log(0)", infinity, nan]}
  - {topic: remedy epsilon clipping, keywords: [epsilon clipping, clamp]}
  - {topic: interpreting the loss value, keywords: [lower is better, goal is to make loss go down]}
  - {topic: capability limit only scores, keywords: [only scores, does not change any weight, optimizer]}
---

@@@ hero
@lede Welcome back! Picture a round of **mini-golf**. At each hole you tap the ball, it stops somewhere, and you write **one number** on your card: how far off you were. Low number, great hole. High number, rough one. A neural network keeps a scorecard exactly like that — and today you'll build it. Here's the part that should make you lean in: that one little number is what every model on earth, from the spam filter in your inbox to the model behind ChatGPT, spends its entire life trying to shrink. Learn to read the **golf score**, and you finally see what the whole game is *for*.
@goal Together we'll build the scorecard from scratch — one score for guessing numbers, a calm cousin for messy data, one for yes/no calls, one for picking among many. You'll *feel* why a confident-but-wrong guess should sting most, learn to read a falling score like a pro, catch three sneaky ways the score gets fooled (each a quick puzzle with a one-line fix), and learn the one honest thing a score can't do alone. Every formula arrives in plain words first. Heavy math sits in a skippable box.
%%% warmup
q: From yesterday — which way does a forward pass flow through the network? | a:2 | backward, from answer to input | in a loop, over and over | one direction, from input straight to the prediction | it skips around between layers | concept: forward-pass | fb: The forward pass is a one-way trip: input → hidden layer → output → prediction. No going back — that backward trip is training, coming later.
q: From yesterday — what happens if you stack layers with NO bend (activation) between them? | a:1 | they run twice as fast | the whole stack collapses into a single layer, so all the depth is wasted | the shapes stop matching and it crashes | each layer learns a different pattern | concept: linear-collapse | fb: With no activation between them, the straight-line layers fold into one — 5 layers get the power of 1. The fix is a bend between every pair of layers.
q: From yesterday — a layer with 4 neurons, each reading a 3-number input. What shape is its weight grid W? | a:0 | [4, 3] — one row per neuron, one column per input | [3, 4] | [4, 4] | [3, 3] | concept: weight-matrix-shape | fb: Rows = neurons, columns = inputs. 4 neurons reading 3 inputs give a [4, 3] grid.
%%%

@@@ concept id=c1 tag="The scorecard" title="What a loss is — one honest number" gotit="Got what a loss is"
Let's start with the picture, before any math. In mini-golf your **score** is how far off you landed, and lower is always better. A perfect tap is a low number. The whole game is shooting lower next time.

A network keeps the same card. After it guesses, we hand it one number for how far off that guess landed.

That number is the [[loss||One number that says how wrong a prediction is. Lower is better; 0 means the guess was perfect.]], and the little rule that computes it is the [[loss function||The rule that takes the prediction and the true answer and returns a single "how wrong" number.]].

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A mini-golf scorecard on a clipboard. Hole 1 scored 1, a low number, marked great. Hole 2 scored 7, a high number, marked rough. A note reads lower is better and 0 is perfect — a network keeps the same card."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Your golf card: one number per hole — lower is better</text><rect x="150" y="28" width="220" height="118" rx="8" fill="#FDFCF9" stroke="#C9B77A" stroke-width="2"/><rect x="228" y="22" width="64" height="12" rx="4" fill="#B8AEA2"/><line x1="150" y1="58" x2="370" y2="58" stroke="#E5DFD6"/><text x="166" y="50" fill="#6B645E">hole</text><text x="300" y="50" fill="#6B645E">score</text><text x="166" y="82" fill="#2C2A28">1  ⛳</text><rect x="286" y="70" width="30" height="18" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="301" y="83" text-anchor="middle" fill="#2D8B55">1</text><text x="324" y="83" fill="#2D8B55" font-size="10">great!</text><text x="166" y="110" fill="#2C2A28">2  ⛳</text><rect x="286" y="98" width="30" height="18" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="301" y="111" text-anchor="middle" fill="#C93B3B">7</text><text x="324" y="111" fill="#C93B3B" font-size="10">rough</text><text x="260" y="136" text-anchor="middle" fill="#6B645E" font-size="10">0 = perfect shot · a network keeps this exact card</text><text x="40" y="92" font-size="30">⛳</text><text x="55" y="120" text-anchor="middle" fill="#6B645E" font-size="10">how far</text><text x="55" y="133" text-anchor="middle" fill="#6B645E" font-size="10">off?</text></g></svg>
%%%

**What the golf card gets right:** one tidy number, lower is better, 0 is perfect. **Where it breaks down:** a golf score is a whole number you read at the end, but a loss is a *smooth* number the network reads constantly — and smooth is what lets us find *which way* to improve tomorrow.

#### Beat 1 · the two things a loss compares
Every loss looks at exactly two things, and nothing else:

- the **prediction** — the guess that rolled out of yesterday's forward pass;
- the **target** (also called the **label**) — the true answer the data came with.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A guess lands near a target; the loss is that gap turned into one number, like a golf score"><g font-family="monospace" font-size="13" text-anchor="middle"><circle cx="360" cy="60" r="26" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="360" y="64" fill="#1a5c38" font-size="11">target</text><circle cx="150" cy="60" r="7" fill="#2A7B9B"/><text x="150" y="90" fill="#1F6280" font-size="11">guess</text><line x1="160" y1="60" x2="332" y2="60" stroke="#C99A12" stroke-width="2" stroke-dasharray="5,4"/><text x="248" y="48" fill="#9A7208" font-size="11">how far off?</text><text x="248" y="115" fill="#6B645E" font-size="11">loss = that gap, as one number (lower is better)</text></g></svg>
%%%

Hand a loss function that `(prediction, target)` pair and it gives back one **how wrong** number. That's it.

%%% insight
Here's the sentence to tattoo on your brain: **all of training is just changing the weights to make that number smaller.** Every trick you will ever read about — every clever architecture, every giant GPU cluster — is in service of shrinking this one honest number. So it's worth ten minutes.
%%%

#### Beat 2 · one hole, or the whole round?
*So far* we've scored **one** guess — one hole. Real training has thousands of examples, so we play the whole round and average the card.

%%% steps
step: score hole 1 → loss on example 1
why: pair up (prediction, target), get one number for how wrong that single guess was
step: do it for every example in the batch
why: each one gets its own little score, exactly like each hole gets its own number
step: average them all → the **cost**
why: one number for the whole batch, so the network has a single thing to chase down
%%%

That averaged number has its own name: the [[cost||The loss averaged over all the training examples: the single number the whole network tries to shrink.]] (you'll also hear "objective"). Same idea as one hole — just spread over the whole book of examples instead of one page.

Notice we took the **average**, not the sum. That's deliberate: an average stays the same size whether your batch has 8 examples or 800, therefore your training settings don't secretly change when you change the batch size.

!!! c-info 🏭
<b>A real one to keep you hungry:</b> a spam filter, a photo tagger, a chatbot — every one of them is trained by shrinking the same kind of number you're about to build with your own hands. Same idea, wildly different scale.
!!!

@@@ concept id=c2 tag="Why smooth" title="Counting mistakes gives no hint — so we go smooth" gotit="Got why smooth wins"
Before anything fancy, let's try the scorecard a kid would invent: just **count the mistakes**. Ten guesses, three wrong, score is 3. Simple. Honest. And completely useless for learning — which is a lovely little puzzle.

Picture a quiz graded with a rubber stamp: **✓ or ✗**, nothing in between. You missed by a hair? ✗. You guessed something wild? Also ✗. The stamp tells you *that* you were wrong, never *how* wrong.

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A quiz graded with a red rubber stamp of check or X. A near-miss answer and a wild-guess answer both get the same red X. The stamp says whether you were wrong, never how wrong, and gives no hint which way to lean."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The rubber stamp: only ✓ or ✗ — nothing in between</text><rect x="40" y="30" width="210" height="52" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><text x="52" y="50" fill="#6B645E">so close — off by a hair</text><text x="52" y="70" fill="#2C2A28">answer: 41  (true: 42)</text><text x="228" y="70" font-size="26" fill="#C93B3B">✗</text><rect x="40" y="96" width="210" height="52" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><text x="52" y="116" fill="#6B645E">wild guess — miles off</text><text x="52" y="136" fill="#2C2A28">answer: 900 (true: 42)</text><text x="228" y="136" font-size="26" fill="#C93B3B">✗</text><rect x="282" y="46" width="78" height="78" rx="10" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="321" y="92" text-anchor="middle" font-size="34" fill="#C93B3B">✗</text><text x="321" y="140" text-anchor="middle" fill="#6B645E" font-size="10">same stamp</text><text x="392" y="78" fill="#C93B3B">tells you THAT</text><text x="392" y="94" fill="#C93B3B">you were wrong,</text><text x="392" y="110" fill="#6B645E">never HOW wrong —</text><text x="392" y="126" fill="#6B645E">no way to lean better</text></g></svg>
%%%

**What the rubber stamp gets right:** counting mistakes is what you care about *in the end* — how many did I get right? **Where it breaks down:** a stamp can't say "you were almost right, lean a little this way." A network learns *only* by following little hints toward better.

#### The freeze — watch the count refuse to move
This crude score even has a proper name, the **0/1 loss** (1 for wrong, 0 for right). Put it under a microscope and nudge one weight.

%%% steps
step: nudge a weight a hair — the guess creeps from 41 toward 42, the true answer
why: real progress! Now: does the score notice?
step: re-read the count → still 1. Still 1. Still 1 — then at one exact point it snaps to 0
why: flat, flat, flat, then a cliff. Flat means no "warmer, colder" — nothing to follow.
step: the fix — measure HOW FAR off, not whether
why: distance changes with every nudge, therefore there is always a hint to follow
%%%

Here's that story as one picture — a cliff you can't climb, next to a ramp you can always slide down:

%%% svg
<svg viewBox="0 0 520 155" role="img" aria-label="Counting mistakes is a flat staircase with a cliff, giving no slope to follow; a smooth loss is a gentle ramp you can always slide down"><g font-family="monospace" font-size="11"><text x="130" y="22" fill="#3A342E" font-weight="bold">count mistakes — a cliff</text><line x1="40" y1="130" x2="250" y2="130" stroke="#E5DFD6" stroke-width="1.5"/><path d="M50 55 L143 55 L143 120 L245 120" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text x="90" y="48" fill="#C93B3B">wrong = 1 (flat)</text><text x="200" y="113" fill="#2D8B55">right = 0</text><text x="145" y="148" fill="#6B645E" text-anchor="middle">no slope to follow — stuck</text><text x="400" y="22" fill="#3A342E" font-weight="bold">smooth loss — a ramp</text><line x1="300" y1="130" x2="500" y2="130" stroke="#E5DFD6" stroke-width="1.5"/><path d="M312 50 C 360 95, 430 122, 495 128" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="400" y="148" fill="#6B645E" text-anchor="middle">always a hint: nudge downhill</text></g></svg>
%%%

%%% insight
The sneaky part, worth feeling properly: a model trained on the count doesn't crash or complain. It runs, reports a number, and **learns absolutely nothing** — because a network improves by reading the **slope** of the loss, and a flat staircase has no slope to read.
%%%

So every real loss is a **smooth** stand-in for the crude count: it measures *how far off* you were, which means the score wiggles a little with every nudge, which means there is always a downhill direction. That's the rule behind every score in the rest of today.

!!! c-info 🧭
<b>Keep these two jobs straight:</b> counting mistakes is a fine way to <em>report</em> a grade — that's basically "accuracy", the number you'd tell a person. It's a terrible thing to <em>train</em> on. So we train on a smooth loss and report accuracy separately. Two jobs, two tools.
!!!

@@@ concept id=c3 tag="MSE" title="MSE — the score for guessing numbers" gotit="Got MSE"
Now the fun starts — your first real golf score. Say the network is guessing a *number*: a house price, tomorrow's temperature, someone's age.

That task is called [[regression||A task where the network predicts a plain number (a price, a temperature), not a category.]], and its go-to score is [[MSE||Mean Squared Error: measure how far each guess is off, square it, then average. The default loss for predicting numbers.]] — "mean squared error". Heavy name, playground idea.

Picture **cornhole**: tossing beanbags at a hole in a board. For each toss you measure how far the bag landed from the hole. "Short by a foot" and "long by a foot" are equally bad — you don't care about the *direction* of a miss, only its **size**. And a wild toss into the neighbour's yard should count for a *lot* more than one that barely missed.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Cornhole misses: short and long by the same amount are equally bad; a wild toss should hurt far more"><g font-family="monospace" font-size="11" text-anchor="middle"><circle cx="260" cy="60" r="14" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="260" y="64" fill="#1a5c38" font-size="10">hole</text><circle cx="210" cy="60" r="6" fill="#2A7B9B"/><text x="200" y="88" fill="#1F6280">short 1ft</text><circle cx="310" cy="60" r="6" fill="#2A7B9B"/><text x="320" y="88" fill="#1F6280">long 1ft</text><text x="260" y="30" fill="#6B645E">both equally bad — size, not direction</text><circle cx="470" cy="60" r="8" fill="#C93B3B"/><text x="450" y="88" fill="#C93B3B">wild toss</text><text x="450" y="104" fill="#C93B3B">should hurt a LOT more</text></g></svg>
%%%

**What cornhole gets right:** direction doesn't matter, size does, and a wild miss should hurt far more than a near one. **Where it breaks down:** your worst toss is capped by how hard you can throw — MSE has no cap at all. Remember that; it's the very next puzzle.

#### Watch one score get born
Say the network predicted `[2.5, 0.0, 2.0]` and the true targets were `[3.0, −0.5, 2.0]`. Three tosses. Let's turn them into one number, one move at a time.

%%% steps
step: the misses — subtract: pred − target = [−0.5, 0.5, 0.0]
why: how far off each toss landed. Notice the trap: add these up and the first two cancel to zero!
step: the squeeze — square each miss: [0.25, 0.25, 0.0]
why: squaring kills the minus signs, so misses are always positive and can never cancel out
step: the same squeeze also blows up the big ones
why: twice as wrong costs four times as much, therefore big misses count more than small ones
step: the average — (0.25 + 0.25 + 0) / 3 = 0.167
why: one tidy score for all three tosses. A perfect set of guesses gives exactly 0.
%%%

Same three moves, drawn left to right so you can *see* the numbers change shape:

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Three steps of MSE drawn as a flow: misses, then squared misses, then their average of 0.167"><g font-family="monospace" font-size="12" text-anchor="middle"><text x="90" y="30" fill="#6B645E">1 · misses</text><rect x="30" y="42" width="120" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="90" y="62" fill="#9A7208">[−0.5, 0.5, 0]</text><text x="185" y="62" fill="#6B645E">→</text><text x="290" y="30" fill="#6B645E">2 · square each</text><rect x="220" y="42" width="140" height="30" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="290" y="62" fill="#C93B3B">[0.25, 0.25, 0]</text><text x="390" y="62" fill="#6B645E">→</text><text x="465" y="30" fill="#6B645E">3 · average</text><rect x="415" y="42" width="95" height="32" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="462" y="63" fill="#1a5c38" font-weight="bold">0.167</text><text x="260" y="118" fill="#6B645E" font-size="11">square → direction gone, big misses blown up; average → one MSE number</text></g></svg>
%%%

In one line of plain words: **MSE = the average of the squared misses.** That's the whole score for numeric guesses.

%%% demo id=mse label="reveal the score"
predict: three misses of −0.5, +0.5 and 0. Will the score come out negative, zero, or a small positive number? Guess before you click.
code: pred=np.array([2.5,0,2]); target=np.array([3,-0.5,2]); np.mean((pred-target)**2)
out: 0.16666666666666666
take: <b>MSE = mean of squared errors.</b> Misses [−0.5, 0.5, 0] → squares [0.25, 0.25, 0] → mean ≈ 0.167. Small, because the guess is close. A far-off guess prints a much bigger number — lower always means a better fit.
%%%

%%% insight
Did you feel the near-miss in step 1? Two real misses of −0.5 and +0.5 would have **added to exactly zero** — a score of "perfect" for a model that missed twice. That's why the squaring isn't decoration; it's the thing that keeps the scorecard honest.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the formula in symbols.</b> Only peek if symbols help you; the four moves above are all you need. <code>MSE = (1/N) Σᵢ (predᵢ − targetᵢ)²</code>. Read it right to left: take each miss <code>(predᵢ − targetᵢ)</code>, square it, add them all up (the <code>Σ</code>), divide by <code>N</code> examples to make it an average. Same moves, written tight.
!!!

*So far:* one loss, fully built. **MSE is the default whenever the answer is a number** — reach for it first, and only switch away for the reason you're about to meet.

@@@ concept id=c4 tag="Outliers → MAE" title="Puzzle — when one wild point hijacks the score" gotit="Got MAE"
First puzzle of the day, and you've felt this one at a lunch table. You want the **average weight** of ten friends. Nine of you are ordinary kids — and the tenth guest is a sumo wrestler. Add, divide by ten, and the "average" comes out heavier than any real kid at the table.

One giant value dragged the whole average up toward itself. Now remember what MSE does *before* averaging: it **squares**. Can you see the trap coming?

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A lunch table of nine ordinary kids and one huge sumo wrestler guest. The average-weight marker gets dragged far up toward the sumo, ending heavier than any real kid, so it no longer describes the nine kids."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One sumo guest hijacks the average of ten</text><text x="30" y="64" font-size="22">🧒🧒🧒</text><text x="30" y="92" font-size="22">🧒🧒🧒</text><text x="30" y="120" font-size="22">🧒🧒🧒</text><text x="48" y="140" fill="#6B645E" font-size="10">nine ordinary kids</text><text x="150" y="104" font-size="48">🥋</text><text x="168" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">one giant guest</text><line x1="238" y1="40" x2="238" y2="150" stroke="#E5DFD6"/><text x="258" y="52" fill="#6B645E" font-size="10">weight scale →</text><line x1="248" y1="116" x2="330" y2="116" stroke="#2D8B55" stroke-width="2" stroke-dasharray="4,3"/><text x="338" y="120" fill="#2D8B55" font-size="10">where the 9 kids sit</text><line x1="248" y1="66" x2="440" y2="66" stroke="#C93B3B" stroke-width="2.5"/><text x="338" y="58" fill="#C93B3B" font-size="10">the "average" dragged UP here</text><path d="M300 108 C 330 96, 360 82, 388 70" fill="none" stroke="#C93B3B" stroke-width="1.5" stroke-dasharray="3,3"/><polygon points="388,70 380,70 385,77" fill="#C93B3B"/><text x="320" y="166" text-anchor="middle" fill="#6B645E" font-size="10">the number now fits nobody real — and MSE SQUARES it first</text></g></svg>
%%%

**What the sumo picture gets right:** one value far bigger than the rest pulls the whole average toward itself.

**Where it breaks down:** the sumo is real and belongs at the table, so his weight *should* count. A data [[outlier||A single data point far away from all the rest. Because MSE squares errors, one outlier can dominate the whole loss.]] is often just a typo or a freak case — and MSE doesn't merely add it, it **squares** it first.

#### The hijack, in three moves
You're grading house-price guesses. Feel the score slip out of your hands.

%%% steps
step: the honest misses — most guesses are off by a few thousand dollars
why: exactly what you'd hope for. A reasonable model on reasonable houses.
step: one freak mansion, missed by a million — and MSE **squares** it first
why: a million squared dwarfs a few thousand squared, therefore that single row now outweighs the whole crowd
step: the **hijack** — after averaging, the score is basically "how did you do on that ONE house?"
why: your good guesses stopped mattering. One bad row is shouting over ten thousand good ones.
%%%

#### The neat fix — stop squaring
Don't square each miss. Just take its plain size — its [[absolute value||The size of a number, ignoring its sign: |−7| = 7. Used by MAE so a miss counts by its size, not its size-squared.]], meaning how far off it was with the sign ignored — and average those.

That's [[MAE||Mean Absolute Error: average of the plain miss sizes (no squaring). Each miss counts in proportion, so one outlier can't dominate.]], **mean absolute error**: the average of the **plain miss sizes**. A miss of a million now counts as a million, *not* a million squared, which means it can no longer explode past everybody else. So when outliers are hijacking your score, you **switch to MAE**.

Your turn to break it — and to watch the cure hold. Below are twenty ordinary misses of about 2, plus one wild one **you** control. Predict what each score does, then drag:

%%% svg
<svg id="olx-svg" viewBox="0 0 520 196" role="img" aria-label="Interactive outlier explorer. Twenty ordinary misses of about two sit next to one wild miss you drag from 1 up to 1000. Two score bars on a shared log scale show the squared score MSE running away while the plain score MAE grows only gently."><g font-family="monospace" font-size="11"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Drag the wild miss — watch which score runs away</text><line x1="26" y1="160" x2="250" y2="160" stroke="#E5DFD6" stroke-width="1.5"/><g id="olx-small" fill="#2D8B55"><rect x="30" y="150" width="4" height="10"/><rect x="38" y="146" width="4" height="14"/><rect x="46" y="155" width="4" height="5"/><rect x="54" y="150" width="4" height="10"/><rect x="62" y="146" width="4" height="14"/><rect x="70" y="155" width="4" height="5"/><rect x="78" y="150" width="4" height="10"/><rect x="86" y="146" width="4" height="14"/><rect x="94" y="155" width="4" height="5"/><rect x="102" y="150" width="4" height="10"/><rect x="110" y="146" width="4" height="14"/><rect x="118" y="155" width="4" height="5"/><rect x="126" y="150" width="4" height="10"/><rect x="134" y="146" width="4" height="14"/><rect x="142" y="155" width="4" height="5"/><rect x="150" y="150" width="4" height="10"/><rect x="158" y="146" width="4" height="14"/><rect x="166" y="155" width="4" height="5"/><rect x="174" y="150" width="4" height="10"/><rect x="182" y="146" width="4" height="14"/></g><text x="106" y="176" text-anchor="middle" fill="#2D8B55" font-size="10">20 ordinary misses (~2 each)</text><rect id="olx-wild" x="212" y="120" width="18" height="40" fill="#C93B3B"/><text x="221" y="176" text-anchor="middle" fill="#C93B3B" font-size="10">the wild one</text><line x1="278" y1="160" x2="504" y2="160" stroke="#E5DFD6" stroke-width="1.5"/><rect id="olx-mse" x="300" y="140" width="44" height="20" fill="#C93B3B"/><text x="322" y="176" text-anchor="middle" fill="#C93B3B" font-size="10">MSE (squared)</text><text id="olx-msev" x="322" y="134" text-anchor="middle" fill="#C93B3B" font-size="10">4.8</text><rect id="olx-mae" x="420" y="146" width="44" height="14" fill="#2A7B9B"/><text x="442" y="176" text-anchor="middle" fill="#1F6280" font-size="10">MAE (plain size)</text><text id="olx-maev" x="442" y="140" text-anchor="middle" fill="#1F6280" font-size="10">2.0</text><text x="391" y="34" text-anchor="middle" fill="#6B645E" font-size="9">both bars share one squashed scale</text></g></svg>
<div style="margin-top:8px">
<div style="display:flex;justify-content:space-between;font-size:.9em;color:#5A544E"><span>One weird house: missed by <b id="olx-w">2</b></span><span>drag me →</span></div>
<input id="olx-r" type="range" min="1" max="1000" value="2" step="1" aria-label="size of the one wild miss" style="width:100%;margin:6px 0;accent-color:#C93B3B">
<div id="olx-out" style="text-align:center;font-size:1.02em;color:#2C2A28">MSE <b>4.8</b> · MAE <b>2.0</b> &nbsp;<span style="color:#2D8B55">no outlier yet — both scores agree</span></div>
</div>
<script>(function(){
  var s=document.getElementById('olx-r');if(!s)return;
  var small=[2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3];
  var sum=0,sumsq=0;for(var i=0;i<small.length;i++){sum+=small[i];sumsq+=small[i]*small[i];}
  var baseM=sumsq/small.length, baseA=sum/small.length, N=small.length+1;
  var bM=document.getElementById('olx-mse'),bA=document.getElementById('olx-mae'),
      tM=document.getElementById('olx-msev'),tA=document.getElementById('olx-maev'),
      wild=document.getElementById('olx-wild'),wtxt=document.getElementById('olx-w'),
      out=document.getElementById('olx-out');
  var BASE=160,MAXH=118,LOGMAX=Math.log10(60000);
  function hOf(v){return Math.max(6,Math.min(MAXH,MAXH*Math.log10(Math.max(v,1.01))/LOGMAX));}
  function paint(){
    var w=Math.max(1,+s.value);
    var mse=(sumsq+w*w)/N, mae=(sum+w)/N;
    var hM=hOf(mse),hA=hOf(mae),hW=Math.max(6,Math.min(MAXH,MAXH*Math.log10(Math.max(w,1.01))/Math.log10(1000)));
    if(bM){bM.setAttribute('y',(BASE-hM).toFixed(1));bM.setAttribute('height',hM.toFixed(1));}
    if(bA){bA.setAttribute('y',(BASE-hA).toFixed(1));bA.setAttribute('height',hA.toFixed(1));}
    if(wild){wild.setAttribute('y',(BASE-hW).toFixed(1));wild.setAttribute('height',hW.toFixed(1));}
    if(tM){tM.setAttribute('y',(BASE-hM-6).toFixed(1));tM.textContent=mse<100?mse.toFixed(1):Math.round(mse);}
    if(tA){tA.setAttribute('y',(BASE-hA-6).toFixed(1));tA.textContent=mae.toFixed(1);}
    if(wtxt)wtxt.textContent=w;
    var gM=mse/baseM,gA=mae/baseA;
    var word=w<10?'still tame — both scores agree':(w<120?'MSE is pulling ahead already':'MSE has been hijacked — MAE is only nudged');
    var col=w<10?'#2D8B55':(w<120?'#C99A12':'#C93B3B');
    if(out)out.innerHTML='MSE <b>'+(mse<100?mse.toFixed(1):Math.round(mse))+'</b> (×'+Math.round(gM)+' bigger) · MAE <b>'+mae.toFixed(1)+'</b> (×'+Math.round(gA)+') &nbsp;<span style="color:'+col+'">'+word+'</span>';
  }
  s.addEventListener('input',paint);paint();
})();</script>
%%%

%%% insight
Slide it to the far right and read the two multipliers. One wild row blows the squared score up by **thousands of times**; the plain score grows only about twenty-fold. Both scores notice the freak — but that gap is the whole difference between an outlier being *amplified* and an outlier being merely *felt*.
%%%

%%% demo id=mae label="reveal both scores"
predict: twenty ordinary misses of about 2, then one wild 1000 crashes the party. One score grows about 24×, the other about 10,000×. Which is which?
code: small=np.array([2,3,1]*6+[2,3]); big=np.append(small,1000)
code: print("no outlier   MSE",round(np.mean(small**2),1)," MAE",round(np.mean(np.abs(small)),1))
code: print("with outlier MSE",round(np.mean(big**2),1)," MAE",round(np.mean(np.abs(big)),1))
out: no outlier   MSE 4.8  MAE 2.0
out: with outlier MSE 47623.7  MAE 49.6
take: <b>MAE = mean of the plain miss sizes.</b> One wild row multiplies MSE by about <b>10,000×</b> (4.8 → 47,624) but MAE by only about <b>24×</b> (2.0 → 49.6). MAE still <em>feels</em> the outlier — it just never lets it be squared into a monster, so the honest misses keep their say.
%%%

There's a best-of-both cousin worth knowing by name: [[Huber loss||A hybrid loss that acts like MSE for small misses (smooth near zero) and like MAE for large misses (calm about outliers).]] acts like **MSE for small misses** and like **MAE for big ones** — smooth near the target, calm about outliers. It's what people reach for when they want both.

!!! c-info ⚖️
<b>The trade-off in one breath:</b> MSE punishes big misses hardest (great on clean data); MAE treats every miss in proportion (great when a few crazy points would hijack the score); Huber sits in between. Which one you pick quietly decides what "doing well" even <em>means</em> — a real design call, not a detail.
!!!

**Victory lap:** puzzle one, solved. You turned a failure into a named cure — outliers hijack MSE, MAE (or Huber) fixes it. That failure→remedy move is *exactly* how engineers think. Two more puzzles later today, and both are quicker than this one.

@@@ concept id=c5 tag="Yes/No → BCE" title="Cross-entropy — the score for confident guesses" gotit="Got cross-entropy"
Switch games — and meet the score behind almost every classifier you have ever used. Instead of a number, the network makes a **yes/no** call: spam or not spam, cat or dog, sick or healthy. That's [[binary classification||A task where the network picks between exactly two options: yes/no, spam/not-spam.]].

Think of a **weather forecaster**. She says "90% chance of rain" and it rains — great, barely a ding. But she shouts "99% chance of sun!" and it *pours* — that's a disaster: loud, certain, and dead wrong. A good scorecard should barely tap the first and absolutely hammer the second.

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="Two weather forecasters. Top: one loudly bets 99 percent sun but it rains — a huge surprise, a huge loss. Bottom: one carefully bets 90 percent rain and it rains — tiny surprise, near-zero loss. Cross-entropy measures that surprise."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Cross-entropy = your SURPRISE when the truth appears</text><rect x="12" y="34" width="176" height="40" rx="8" fill="#FDECEC" stroke="#E0A9A0"/><text x="100" y="52" text-anchor="middle" fill="#8a3b3b">bet ☀️ 99% SUN</text><text x="100" y="67" text-anchor="middle" font-size="9" fill="#9a6b64">loud &amp; certain</text><text x="198" y="58" fill="#B8AEA2">→</text><rect x="214" y="34" width="86" height="40" rx="8" fill="#EFEAF7" stroke="#B3A6D4"/><text x="257" y="58" text-anchor="middle" fill="#5E5191">truth: ☔ RAIN</text><text x="308" y="58" fill="#B8AEA2">→</text><rect x="324" y="34" width="184" height="40" rx="8" fill="#FDE8E8" stroke="#C93B3B"/><text x="338" y="60" font-size="16">😱</text><text x="364" y="52" fill="#C93B3B">HUGE surprise</text><text x="364" y="67" font-size="9" fill="#C93B3B">→ huge loss</text><rect x="12" y="108" width="176" height="40" rx="8" fill="#EAF5EE" stroke="#A7D2B8"/><text x="100" y="126" text-anchor="middle" fill="#276b45">bet ☔ 90% RAIN</text><text x="100" y="141" text-anchor="middle" font-size="9" fill="#5a8a6e">careful &amp; humble</text><text x="198" y="132" fill="#B8AEA2">→</text><rect x="214" y="108" width="86" height="40" rx="8" fill="#EFEAF7" stroke="#B3A6D4"/><text x="257" y="132" text-anchor="middle" fill="#5E5191">truth: ☔ RAIN</text><text x="308" y="132" fill="#B8AEA2">→</text><rect x="324" y="108" width="184" height="40" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="338" y="134" font-size="16">😌</text><text x="364" y="126" fill="#2D8B55">tiny surprise</text><text x="364" y="141" font-size="9" fill="#2D8B55">→ loss ≈ 0</text><text x="260" y="182" text-anchor="middle" font-size="10" fill="#6B645E">Same for cats, spam, anything: bet big on the WRONG answer → big shock → big loss.</text></g></svg>
%%%

**What the forecaster gets right:** the penalty depends on *confidence*, not just right-vs-wrong. **Where it breaks down:** a real forecaster feels embarrassment; the network feels nothing. The whole sting lives in the math.

#### How the answer comes out as a probability
Before we can score confidence, the network has to *express* it. Two quick moves:

%%% steps
step: the last layer spits out a raw, untidied number — the **logits**
why: these are just the raw output scores. They can be −4.2 or +17; they are not percentages yet.
step: a **sigmoid** squashes that into a number between 0 and 1
why: now you can read it as "I'm 90% sure this is spam" — a confidence you can actually score
%%%

Those raw scores are the [[logits||The raw output scores of the last layer, before a sigmoid or softmax turns them into probabilities.]], and the squash is the [[sigmoid||A function that squashes any number into a probability between 0 and 1 — the natural output for a yes/no call.]] you met on Day 2.

The matching score for that kind of output is [[binary cross-entropy||The loss for yes/no predictions made as a probability. It barely dings confident-and-right guesses and hammers confident-and-wrong ones. Also called log loss.]] — also called **log loss**.

#### The one-line rule, in plain words
Cross-entropy looks at exactly one thing: the **prob of the correct** answer. Then it scores `−log` of it.

%%% steps
step: find the probability the model gave the TRUE answer
why: everything else it said is irrelevant — only the truth's share counts
step: take −log of that probability
why: `log` is the shape you met in M1; the minus flips it so small probability → big number
step: near 1 → the score is near 0
why: confident and right, so barely any penalty. This is the "😌" row of the picture.
step: sliding toward 0 → the score shoots up
why: confidently wrong, and the punishment grows without any ceiling. The "😱" row.
%%%

`log` quietly turns "probability" into "surprise". Now stop reading and **drag the dial** — watch that surprise climb a cliff:

%%% svg
<svg id="cex-svg" viewBox="0 0 520 190" role="img" aria-label="Interactive cross-entropy: drag your model's confidence in the correct answer and watch the loss — near zero when confident and right, exploding upward when confident and wrong"><g font-family="monospace" font-size="11"><line x1="60" y1="150" x2="484" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><line x1="60" y1="24" x2="60" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><path id="cex-curve" d="M62 30 C 140 84, 230 130, 320 142 C 384 149, 440 150, 478 150" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><circle id="cex-dot" cx="308" cy="142" r="6.5" fill="#C99A12" stroke="#fff" stroke-width="2"/><text x="272" y="174" fill="#6B645E" text-anchor="middle">your model's confidence in the CORRECT answer  →</text><text x="26" y="92" fill="#6B645E" text-anchor="middle" transform="rotate(-90 26 92)">loss (surprise)</text><text x="478" y="144" fill="#2D8B55" text-anchor="end">sure &amp; right → ≈ 0</text><text x="90" y="40" fill="#C93B3B">sure &amp; WRONG → 😱</text></g></svg>
<div style="margin-top:8px">
<div style="display:flex;justify-content:space-between;font-size:.9em;color:#5A544E"><span>Truth: it really <b>is</b> a cat 🐱.</span><span>Your model was <b id="cex-pct">60%</b> sure.</span></div>
<input id="cex-p" type="range" min="1" max="99" value="60" step="1" aria-label="model confidence in the correct answer" style="width:100%;margin:6px 0;accent-color:#7C6DAA">
<div id="cex-out" style="text-align:center;font-size:1.02em;color:#2C2A28">loss = −log(0.60) = <b>0.51</b> 😐 <span style="color:#C99A12">hedging — a medium ding</span></div>
</div>
<script>(function(){
  var s=document.getElementById('cex-p');if(!s)return;
  var dot=document.getElementById('cex-dot'),curve=document.getElementById('cex-curve'),
      pct=document.getElementById('cex-pct'),out=document.getElementById('cex-out');
  var X0=60,X1=478,Y0=150,YT=30,LMAX=5;
  function xOf(p){return X0+p*(X1-X0);}
  function yOf(L){return Y0-Math.min(L,LMAX)/LMAX*(Y0-YT);}
  var d='';for(var i=0;i<=100;i++){var q=0.01+(0.99-0.01)*i/100;d+=(i?'L':'M')+xOf(q).toFixed(1)+' '+yOf(-Math.log(q)).toFixed(1)+' ';}
  if(curve)curve.setAttribute('d',d.trim());
  function paint(){
    var p=Math.max(1,Math.min(99,+s.value))/100,L=-Math.log(p);
    var col=p>=0.8?'#2D8B55':(p>=0.4?'#C99A12':'#C93B3B');
    var face=p>=0.8?'😌':(p>=0.4?'😐':'😱');
    var word=p>=0.8?'confident and right — barely a tap':(p>=0.4?'hedging — a medium ding':'confident and WRONG — brutal!');
    if(dot){dot.setAttribute('cx',xOf(p).toFixed(1));dot.setAttribute('cy',yOf(L).toFixed(1));dot.setAttribute('fill',col);}
    if(pct)pct.textContent=Math.round(p*100)+'%';
    if(out)out.innerHTML='loss = −log('+p.toFixed(2)+') = <b>'+L.toFixed(2)+'</b> '+face+' &nbsp;<span style="color:'+col+'">'+word+'</span>';
  }
  s.addEventListener('input',paint);paint();
})();</script>
%%%

%%% insight
Slide it all the way left and look at the number. That cliff is not a bug — it's the *point*. A model that hedges at 50% is only mildly punished, but one that is loudly, **confidently wrong** gets hammered, which means cross-entropy teaches a network to be humble unless it's sure. That single design choice is why classifiers behave sensibly.
%%%

%%% demo id=ce label="reveal the three scores"
predict: the model's confidence in the right answer goes 0.3 → 0.6 → 0.9. Does the loss rise or fall — and does it fall evenly?
code: [round(-np.log(p),3) for p in [0.3, 0.6, 0.9]]
out: [1.204, 0.511, 0.105]
take: <b>Cross-entropy = −log(probability of the correct answer).</b> As confidence in the right answer grows (0.3 → 0.6 → 0.9), the loss falls (1.204 → 0.511 → 0.105). Get it right with full confidence and the score glides toward 0.
%%%

#### The quiet pairing rule
*So far, so good.* One thing trips up every beginner: **each loss expects the output in a matching shape.** You can't bolt any loss onto any output.

%%% table
:: The task :: What the output looks like :: The loss to use
Guess a number (regression) :: a raw number (linear output) :: MSE (or MAE / Huber)
Yes / no (binary) :: one probability, 0 to 1, from a sigmoid :: binary cross-entropy
%%%

Get the pairing wrong and the score still computes *a* number — just the wrong one. That's a sneaky bug, and you'll meet it soon. **You can now pick the right scorecard for a yes/no game.** But what if there are more than two answers?

@@@ concept id=c6 tag="Many answers → softmax" title="More than two choices — softmax + categorical cross-entropy" gotit="Got softmax"
Real life rarely stops at yes/no. Is this photo a **cat, a dog, or a bird**? Is this digit a **0, 1, 2 … or 9**? Picking one answer from three or more choices is [[multi-class classification||Picking one answer out of three or more choices, like cat vs dog vs bird.]] — and here's the happy surprise: it's the *same* score you just learned, gently stretched.

Picture **one whole pizza that you slice up** among the choices. You have 100% of a pizza — your total confidence — and you cut it: a big slice for "cat", a smaller one for "dog", a sliver for "bird". The slices always add up to exactly one whole pizza. Never more, never less.

%%% svg
<svg viewBox="0 0 520 182" role="img" aria-label="One whole pizza of confidence sliced among three choices. Cat gets a big slice covering two thirds of the pizza at 66 percent, dog gets a smaller slice at 24 percent, and bird gets a thin sliver at 10 percent. The slices always add up to exactly one whole pizza."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One pizza of confidence, sliced across every choice — always adds to 1</text><circle cx="128" cy="104" r="58" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><path d="M128 104 L186 104 A58 58 0 1 1 96.9 55.0 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><path d="M128 104 L96.9 55.0 A58 58 0 0 1 174.9 69.9 Z" fill="#E7F0F5" stroke="#2A7B9B" stroke-width="1.5"/><path d="M128 104 L174.9 69.9 A58 58 0 0 1 186 104 Z" fill="#F3ECDB" stroke="#C99A12" stroke-width="1.5"/><text x="111" y="136" text-anchor="middle" fill="#276b45" font-size="11">🐱 .66</text><text x="134" y="73" text-anchor="middle" fill="#1F6280" font-size="9">.24</text><rect x="228" y="48" width="180" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="236" y="64" fill="#276b45">🐱 cat</text><text x="402" y="64" text-anchor="end" fill="#276b45">.66 big slice</text><rect x="228" y="78" width="65" height="22" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="234" y="94" fill="#1F6280">🐶</text><text x="302" y="94" fill="#1F6280">dog .24</text><rect x="228" y="108" width="27" height="22" rx="4" fill="#F3ECDB" stroke="#C99A12"/><text x="232" y="124" font-size="10">🐦</text><text x="264" y="124" fill="#9A7208">bird .10 — the sliver</text><text x="228" y="150" fill="#6B645E" font-size="10">chip widths match the slice sizes</text><text x="228" y="168" fill="#1a5c38" font-size="10">.66 + .24 + .10 = one whole pizza</text></g></svg>
%%%

The step that cuts the pizza — turning the raw logits into slices that add to 1 — is called [[softmax||A step that turns a list of raw scores (logits) into probabilities that add up to 1 — the natural output for choosing among 3+ classes.]].

**What the pizza gets right:** a fixed whole to share out, so giving more to one choice *means* less for the others. **Where it breaks down:** you could cut real slices any size by hand, but softmax sets the sizes automatically from the raw scores — always positive, always adding to one whole.

#### From raw scores to slices to a score
Watch three raw numbers become a score, one move at a time.

%%% steps
step: the raw scores arrive — logits [2.0, 1.0, 0.1]
why: just the last layer's untidied output. Nothing here is a percentage yet.
step: **softmax slices the pizza** → [0.66, 0.24, 0.10]
why: bigger score gets a bigger slice, all slices positive, and together they sum to 1
step: read the slice you gave the CORRECT class — 0.66 for cat
why: same rule as yes/no — only the truth's share matters
step: score it: −log(0.66) ≈ 0.416
why: fat slice for the right answer → small loss. Thin sliver → big loss. Identical heart.
%%%

%%% svg
<svg viewBox="0 0 520 152" role="img" aria-label="Softmax turns three raw scores into three probabilities that add up to one. The cat segment fills two thirds of the bar, the dog segment is a quarter of it, and the bird segment is a thin sliver."><g font-family="monospace" font-size="11"><text x="20" y="30" fill="#6B645E">raw scores (logits)</text><rect x="20" y="42" width="120" height="26" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="80" y="60" fill="#9A7208" text-anchor="middle">[2.0, 1.0, 0.1]</text><text x="152" y="56" fill="#6B645E">— softmax →</text><text x="160" y="72" fill="#9A938A" font-size="9">cuts the pizza</text><text x="280" y="30" fill="#6B645E">probabilities (they add to 1)</text><rect x="280" y="42" width="145" height="26" fill="#2D8B55"/><rect x="425" y="42" width="53" height="26" fill="#2A7B9B"/><rect x="478" y="42" width="22" height="26" fill="#C99A12"/><text x="352" y="60" fill="#fff" text-anchor="middle">cat .66</text><text x="451" y="60" fill="#fff" text-anchor="middle" font-size="10">dog .24</text><line x1="489" y1="70" x2="489" y2="82" stroke="#9A7208"/><text x="489" y="94" fill="#9A7208" text-anchor="middle" font-size="10">bird .10</text><text x="20" y="120" fill="#6B645E" font-size="11">one whole "pizza" of confidence, sliced across every class → always sums to 1.0</text><text x="20" y="142" fill="#1a5c38" font-size="11">loss = −log(the slice you gave the CORRECT class) — same rule as yes/no</text></g></svg>
%%%

This multi-class version has a name: [[categorical cross-entropy||The loss for picking one of 3+ classes. It reads the softmax probability you gave the correct class and scores −log of it — the same rule as binary cross-entropy, spread across many classes.]]. Notice you learned nothing genuinely new — you stretched one idea over more slices.

%%% insight
The pizza rule has a consequence that's easy to miss: because the slices must **sum to 1**, the classes *compete*. Giving "cat" more confidence forces "dog" and "bird" to give some up. That's why softmax is right for "pick exactly one" — and why it's the wrong tool when several answers can be true at once.
%%%

%%% demo id=cce label="reveal the score"
predict: the model gave the correct class (cat) a 0.66 slice. Will the loss be closer to 0.4, or closer to 4?
code: probs=[0.66,0.24,0.10]; correct=0; round(-np.log(probs[correct]),3)  # 3 classes, true class = cat
out: 0.416
take: <b>Categorical cross-entropy = −log(the slice you gave the correct class).</b> Cat got 0.66, so the loss is −log(0.66) ≈ 0.416. Slice the correct class thinner and this climbs; hand it nearly the whole pizza and it drops toward 0.
%%%

**The whole classification toolkit in one breath:** yes/no → **sigmoid + binary cross-entropy**; one-of-many → **softmax + categorical cross-entropy**. Same `−log` heart, two shapes of output. That's what you'll carry into every model you build from here.

@@@ concept id=c7 tag="Slow MSE" title="Puzzle — why MSE stalls on yes/no calls" gotit="Got the stall"
Second puzzle, and it starts with a tempting shortcut. MSE is so simple — why not use it for yes/no too? Score "yes" as `1`, "no" as `0`, take the squared miss. It runs. It gives a perfectly valid number. And then the network learns **painfully slowly**, exactly when it's most wrong.

Picture shoving a heavy box toward home across a floor that turns to thick **mud** the farther out you are. Right when the box is far away and *should* be racing back, the mud is thickest — so each shove barely moves it.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A person shoving a heavy box toward home across a floor that turns to thick mud far from home. Where the box is farthest and most wrong the mud is thickest, so it barely budges per shove, while near home the floor is clean."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">MSE + sigmoid: the floor turns to mud right where you're most wrong</text><rect x="20" y="96" width="230" height="38" fill="#EDE4D4"/><path d="M20 96 q 20 -8 40 0 q 20 -8 40 0 q 20 -8 40 0 q 20 -8 40 0 q 20 -8 40 0 q 15 -6 30 0" fill="none" stroke="#8a6d3b" stroke-width="2"/><text x="120" y="124" text-anchor="middle" fill="#6b4f22" font-size="10">THICK mud</text><rect x="250" y="96" width="250" height="38" fill="#F2F4F0"/><text x="400" y="120" text-anchor="middle" fill="#6B645E" font-size="10">clean floor</text><rect x="34" y="66" width="36" height="30" rx="3" fill="#C99A12" stroke="#9A7208"/><text x="52" y="86" text-anchor="middle" fill="#fff" font-size="10">box</text><text x="14" y="58" font-size="20">🧍</text><path d="M74 80 l22 0" stroke="#C93B3B" stroke-width="2"/><polygon points="96,80 88,76 88,84" fill="#C93B3B"/><text x="70" y="152" fill="#C93B3B" font-size="10">far from home &amp; wrong → barely budges 😩</text><text x="320" y="152" fill="#6B645E" font-size="10">push should be strongest here — but it's weakest</text></g></svg>
%%%

That's MSE on a yes/no call: the learning push fades to almost nothing at the worst possible moment. **Where the mud picture breaks down:** real mud slows a box *everywhere*, but this slowdown bites in one specific spot — when the model is **confidently wrong**.

Want to work it out yourself first? One rung at a time:

%%% hint
t1: No math — just the picture. On a yes/no call the sigmoid output flattens at the extreme ends. When the model is *confidently* wrong ("0.99 spam!" but it isn't), is it sitting on the flat part of that squash, or the steep part?
t2: MSE's push = the plain miss × the sigmoid's steepness. At p = 0.99 that steepness is only 0.99 × 0.01 ≈ 0.01 — so MSE's push is about a hundredth of the plain miss. Cross-entropy's push IS the plain miss.
t3: MSE + sigmoid multiplies the push by the sigmoid's near-zero slope exactly when the model is confidently wrong, so it barely moves — that's the mud. Cross-entropy cancels that slope so the push stays strong there.
%%%

#### The stall, in three moves
Back to the box in the mud. Here's what the mud actually *is*.

%%% steps
step: the box sits far out — the model says 0.99 "spam" and the truth is "not spam"
why: as wrong as a model can get, and exactly the moment you want the biggest correction
step: the **flat sigmoid slope** — out at 0.99 the squash has gone almost flat: 0.99 × 0.01 ≈ 0.01
why: MSE multiplies its push BY that flatness, therefore the push shrinks to about a hundredth of the plain miss — a whisper instead of a shove. That is the mud, and that is the **stall**: the most-wrong examples get the weakest correction.
step: the cure — cross-entropy is built so that flat factor CANCELS
why: its push is simply the plain miss (≈ 0.99 here), so the push stays loud precisely where MSE went quiet
%%%

Same two scorecards, measured at two moments — mildly wrong, then confidently wrong. Watch the red bar shrink while the green one grows:

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="Paired bars at two moments. When the model is mildly wrong at p equals 0.6, the MSE push is 0.14 and the cross-entropy push is 0.60. When the model is confidently wrong at p equals 0.99, the MSE push shrinks to about 0.01 while the cross-entropy push grows to 0.99. MSE fades exactly where cross-entropy gets stronger."><g font-family="monospace" font-size="11"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The learning push: same two moments, two scorecards</text><line x1="62" y1="152" x2="500" y2="152" stroke="#E5DFD6" stroke-width="1.5"/><line x1="62" y1="34" x2="62" y2="152" stroke="#E5DFD6" stroke-width="1.5"/><text x="30" y="100" text-anchor="middle" fill="#6B645E" font-size="10" transform="rotate(-90 30 100)">learning push</text><rect x="90" y="36" width="12" height="10" fill="#C93B3B"/><text x="108" y="45" fill="#C93B3B" font-size="10">MSE + sigmoid</text><rect x="230" y="36" width="12" height="10" fill="#2D8B55"/><text x="248" y="45" fill="#2D8B55" font-size="10">cross-entropy</text><rect x="140" y="138" width="26" height="14" fill="#C93B3B"/><text x="153" y="132" text-anchor="middle" fill="#C93B3B" font-size="10">0.14</text><rect x="176" y="92" width="26" height="60" fill="#2D8B55"/><text x="189" y="86" text-anchor="middle" fill="#2D8B55" font-size="10">0.60</text><text x="171" y="169" text-anchor="middle" fill="#6B645E" font-size="10">mildly wrong (p=0.6)</text><rect x="360" y="149" width="26" height="3" fill="#C93B3B"/><text x="373" y="143" text-anchor="middle" fill="#C93B3B" font-size="10">0.01</text><rect x="396" y="53" width="26" height="99" fill="#2D8B55"/><text x="409" y="47" text-anchor="middle" fill="#2D8B55" font-size="10">0.99</text><text x="391" y="169" text-anchor="middle" fill="#6B645E" font-size="10">confidently WRONG (p=0.99)</text><path d="M166 141 L358 150" fill="none" stroke="#C93B3B" stroke-width="1.2" stroke-dasharray="4,3"/><text x="262" y="137" text-anchor="middle" fill="#C93B3B" font-size="10">MSE fades to a whisper ↘</text><path d="M202 90 L394 55" fill="none" stroke="#2D8B55" stroke-width="1.2" stroke-dasharray="4,3"/><text x="298" y="64" text-anchor="middle" fill="#2D8B55" font-size="10">cross-entropy grows to a shout ↗</text><text x="260" y="188" text-anchor="middle" fill="#6B645E" font-size="10">right where the fix is needed most, one push dies and the other gets louder</text></g></svg>
%%%

%%% insight
Read the two red bars again: **0.14 → 0.01**. MSE's push got *smaller* as the mistake got *worse*. The strange part: MSE + sigmoid is not broken — it computes a fine, correct number. It just whispers the correction when it should be shouting. If that took two reads, you're in good company.
%%%

%%% demo id=push label="reveal both pushes"
predict: the model says 0.99 but the truth is 0. One push is about 0.99, the other about 0.01 — which one belongs to MSE?
code: p=0.99; sig_slope=p*(1-p); mse_push=round((p-0)*sig_slope,4); ce_push=round(p-0,4); print("MSE push",mse_push," cross-entropy push",ce_push)
out: MSE push 0.0098  cross-entropy push 0.99
take: <b>MSE's push nearly vanishes; cross-entropy's stays loud.</b> Confidently wrong at p=0.99, the sigmoid's steepness is only ≈0.01, so MSE's push is the plain miss <em>times</em> that — roughly two orders of magnitude weaker than cross-entropy's push, which is the plain miss itself (0.99). Same mistake, wildly different correction.
%%%

**The neat fix:** swap in binary cross-entropy. Where MSE-plus-sigmoid whispers "meh, barely move", **cross-entropy is the fix** that shouts "you're loudly wrong — fix this *now*". That's the whole reason classifiers train on cross-entropy.

!!! c-info 🔎
<b>Optional (skippable) — one line of "why".</b> With a sigmoid output, the MSE learning signal carries a factor of the sigmoid's slope, which is ≈ 0 in the flat tails (Day 2) — so a confident-wrong unit gets almost no signal.<br>
<b>Why cross-entropy escapes it:</b> it is built so that factor cancels: its signal is simply proportional to the plain error <code>(prediction − target)</code>, which is <em>largest</em> when the model is most wrong.<br>
<b>One bookkeeping note:</b> the pushes above use the tidy <code>½·(pred − target)²</code> form of MSE, which is why no stray factor of 2 shows up; the shrinking-versus-growing story is identical either way.<br>
<b>Where this goes next:</b> you'll see the actual slopes multiplied out on Day 5.
!!!

**Victory lap:** two puzzles down, and both cures fit on a sticky note — *outliers? use MAE.* *Yes/no? never MSE, use cross-entropy.* Now let's take a breath and do something easier and rather satisfying: learn to *read* the score you've been building.

@@@ concept id=c8 tag="Read the score" title="Reading the score like a pro — is it actually working?" gotit="Got how to read it"
Breather time — and a genuinely useful skill. You've built three scorecards. Now: how do you *look* at a loss number and know whether your model is doing well? This is the part that makes you feel like an insider, because there's a trick to it.

Picture a stack of **golf cards taped to the fridge** — one per week for a whole season. Nobody stares at a single card and asks "is 46 a good number?" You look at the *stack* and ask one thing: **is the line going down?**

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A fridge door with a season of golf cards taped to it, showing weekly scores of 46, 41, 37, 34 and 33. A line drawn through them slopes downward. A note says nobody asks whether one card is good; you ask whether the line is going down."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A season on the fridge — you read the trend, not one card</text><rect x="26" y="28" width="300" height="128" rx="10" fill="#F4F5F3" stroke="#C4C9C2" stroke-width="2"/><rect x="34" y="60" width="10" height="34" rx="4" fill="#B8AEA2"/><g font-size="10"><rect x="58" y="44" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="80" y="58" text-anchor="middle" fill="#6B645E">wk 1</text><text x="80" y="72" text-anchor="middle" fill="#C93B3B">46</text><rect x="112" y="52" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="134" y="66" text-anchor="middle" fill="#6B645E">wk 2</text><text x="134" y="80" text-anchor="middle" fill="#C99A12">41</text><rect x="166" y="62" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="188" y="76" text-anchor="middle" fill="#6B645E">wk 3</text><text x="188" y="90" text-anchor="middle" fill="#C99A12">37</text><rect x="220" y="74" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="242" y="88" text-anchor="middle" fill="#6B645E">wk 4</text><text x="242" y="102" text-anchor="middle" fill="#2D8B55">34</text><rect x="266" y="82" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="288" y="96" text-anchor="middle" fill="#6B645E">wk 5</text><text x="288" y="110" text-anchor="middle" fill="#2D8B55">33</text></g><path d="M80 46 L134 56 L188 66 L242 78 L288 86" fill="none" stroke="#2D8B55" stroke-width="2" stroke-dasharray="5,3"/><text x="176" y="136" text-anchor="middle" fill="#1a5c38" font-size="10">the line goes DOWN — you're getting better ⛳</text><text x="344" y="60" fill="#6B645E">one card alone:</text><text x="344" y="76" fill="#C93B3B">"is 46 good?" 🤷</text><text x="344" y="104" fill="#6B645E">the stack:</text><text x="344" y="120" fill="#2D8B55">"it's dropping!" 🎉</text></g></svg>
%%%

**What the fridge gets right:** the direction over time is the whole story, and one number on its own tells you almost nothing. **Where it breaks down:** in golf you can eventually shoot a perfect round, but a loss usually flattens out somewhere above zero — and a *perfectly* zero training loss is often bad news (a story for a later day).

%%% insight
This is why "is my loss good?" is the wrong question, and it trips up nearly everyone. `0.42` is fantastic on one task and terrible on another — the units and the number of classes change everything. **Lower is better**, always, but *lower than what?* You're two minutes from knowing exactly how to answer that.
%%%

#### Four curves — spot the healthy one
Every training run draws a line: the loss after each step. Four shapes, and you can name all four after this one picture.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Four training-loss curves side by side. First: a curve that drops steeply then flattens, a healthy run. Second: a flat line that never moves, nothing is learning. Third: a jagged line that trends down but jumps around, the step size is too big. Fourth: a curve that falls and then shoots off the top into NaN."><g font-family="monospace" font-size="10"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Four loss curves — which one is a healthy run?</text><rect x="18" y="30" width="112" height="84" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><line x1="24" y1="108" x2="124" y2="108" stroke="#E5DFD6"/><path d="M26 40 C 52 84, 82 102, 122 105" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="74" y="128" text-anchor="middle" fill="#3A342E">drops → flattens</text><text x="74" y="142" text-anchor="middle" fill="#2D8B55">✓ healthy</text><rect x="146" y="30" width="112" height="84" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><line x1="152" y1="108" x2="252" y2="108" stroke="#E5DFD6"/><path d="M152 64 L252 64" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text x="202" y="128" text-anchor="middle" fill="#3A342E">never moves</text><text x="202" y="142" text-anchor="middle" fill="#C93B3B">✗ not learning</text><rect x="274" y="30" width="112" height="84" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><line x1="280" y1="108" x2="380" y2="108" stroke="#E5DFD6"/><path d="M280 42 L294 74 L306 50 L320 86 L334 58 L348 94 L362 70 L378 98" fill="none" stroke="#C99A12" stroke-width="2.5"/><text x="330" y="128" text-anchor="middle" fill="#3A342E">down but jumpy</text><text x="330" y="142" text-anchor="middle" fill="#9A7208">⚠ step too big</text><rect x="402" y="30" width="112" height="84" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><line x1="408" y1="108" x2="508" y2="108" stroke="#E5DFD6"/><path d="M408 44 C 428 72, 442 90, 452 96" fill="none" stroke="#C93B3B" stroke-width="2.5"/><path d="M456 96 L456 36" fill="none" stroke="#C93B3B" stroke-width="2" stroke-dasharray="4,3"/><text x="482" y="44" fill="#C93B3B" font-size="9">nan 💥</text><text x="458" y="128" text-anchor="middle" fill="#3A342E">falls, then 💥</text><text x="458" y="142" text-anchor="middle" fill="#C93B3B">✗ next puzzle!</text></g></svg>
%%%

%%% steps
step: the healthy shape — a steep drop, then a slow flattening
why: the easy mistakes get fixed first, therefore progress naturally slows down. That's the shape you want on your fridge.
step: dead flat from step one → nothing is learning
why: often the wrong loss for the output shape, or a step size so tiny nothing moves
step: jagged but trending down → alive, just clumsy
why: each step is overshooting a bit. Usually harmless; smaller steps smooth it out.
step: falls, then leaps to `nan` → the run has poisoned itself
why: that one is coming up next, and after the next unit you'll fix it in one line
%%%

#### The 5-second sanity check — what does pure guessing score?
Here's the insider trick, and it's *lovely*. Before trusting any run, ask: **what would a model that knows nothing score?** For cross-entropy that number is easy — a pure guesser spreads its pizza evenly, so its loss is `ln K` for `K` choices. Drag the dial and watch that bar move:

%%% svg
<svg id="cl-svg" viewBox="0 0 520 156" role="img" aria-label="Interactive chance-level meter. Drag the number of classes and the marker moves along a loss axis from 0 to 5, showing the loss a pure guesser would score. Left of the marker is the learning zone; right of it means the model is still guessing."><g font-family="monospace" font-size="11"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The 5-second check: where does pure guessing land?</text><rect id="cl-good" x="70" y="58" width="55" height="28" fill="#EAF5EE" stroke="#2D8B55"/><rect id="cl-bad" x="125" y="58" width="345" height="28" fill="#FDECEC" stroke="#E0A9A0"/><line id="cl-mark" x1="125" y1="44" x2="125" y2="100" stroke="#C99A12" stroke-width="2.5"/><text id="cl-lab" x="125" y="38" text-anchor="middle" fill="#9A7208" font-size="10">0.69</text><line x1="70" y1="100" x2="470" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><text x="70" y="116" fill="#6B645E" font-size="10">0</text><text x="470" y="116" text-anchor="end" fill="#6B645E" font-size="10">5</text><text x="150" y="134" text-anchor="middle" fill="#2D8B55" font-size="10">← real learning lives here</text><text x="400" y="134" text-anchor="middle" fill="#C93B3B" font-size="10">still guessing →</text><text x="270" y="150" text-anchor="middle" fill="#6B645E" font-size="10">cross-entropy loss value →</text></g></svg>
<div style="margin-top:8px">
<div style="display:flex;justify-content:space-between;font-size:.9em;color:#5A544E"><span>How many choices? <b id="cl-k">2</b> (yes/no → 2, digits → 10)</span><span>drag me →</span></div>
<input id="cl-r" type="range" min="2" max="100" value="2" step="1" aria-label="number of classes" style="width:100%;margin:6px 0;accent-color:#C99A12">
<div id="cl-out" style="text-align:center;font-size:1.02em;color:#2C2A28">A pure guesser scores <b>0.69</b> &nbsp;<span style="color:#2D8B55">so a yes/no model stuck at 0.69 has learned nothing yet</span></div>
</div>
<script>(function(){
  var s=document.getElementById('cl-r');if(!s)return;
  var good=document.getElementById('cl-good'),bad=document.getElementById('cl-bad'),
      mark=document.getElementById('cl-mark'),lab=document.getElementById('cl-lab'),
      ktxt=document.getElementById('cl-k'),out=document.getElementById('cl-out');
  var X0=70,X1=470,LMAX=5;
  function paint(){
    var k=Math.max(2,+s.value), chance=Math.log(k);
    var x=X0+Math.min(chance,LMAX)/LMAX*(X1-X0);
    if(good)good.setAttribute('width',Math.max(1,x-X0).toFixed(1));
    if(bad){bad.setAttribute('x',x.toFixed(1));bad.setAttribute('width',Math.max(1,X1-x).toFixed(1));}
    if(mark){mark.setAttribute('x1',x.toFixed(1));mark.setAttribute('x2',x.toFixed(1));}
    if(lab){lab.setAttribute('x',Math.min(Math.max(x,84),456).toFixed(1));lab.textContent=chance.toFixed(2);}
    if(ktxt)ktxt.textContent=k;
    var word=k<=2?'a yes/no model stuck at 0.69 has learned nothing yet':(k<=12?'below this line, your model really is picking up the pattern':'with this many choices, even a good model starts high — compare to the line, not to 0');
    if(out)out.innerHTML='A pure guesser scores <b>'+chance.toFixed(2)+'</b> &nbsp;<span style="color:#2D8B55">so '+word+'</span>';
  }
  s.addEventListener('input',paint);paint();
})();</script>
%%%

%%% demo id=chance label="reveal the guessing scores"
predict: a 10-class digit model that just guesses at random — is its cross-entropy about 0, about 0.5, or about 2.3?
code: [round(-np.log(1/k),3) for k in (2, 10, 1000)]   # a pure guesser gives every class 1/k
out: [0.693, 2.303, 6.908]
take: <b>Chance level = ln K.</b> Two choices → 0.693, ten → 2.303, a thousand → 6.908. So a 10-class run sitting at 2.3 has learned <em>nothing</em>, while 2.3 on a 1000-class problem is real progress. Numbers only mean something next to a baseline.
%%%

There's a matching baseline for numbers, too: if you always guess the average target, MSE lands on the **spread of your data**. Beat that and your model is genuinely adding something.

So the habit is short and sweet: **the goal is to make loss go down**, and you judge the number by comparing it to a baseline, never to zero.

**Victory lap:** you can now open somebody's training log, glance at the curve and the baseline, and say something true about it in five seconds. That's a real engineer's move. One puzzle left — and it's the one hiding in that fourth curve.

@@@ concept id=c9 tag="log(0) blow-up" title="Puzzle — when cross-entropy explodes to infinity" gotit="Got the clip"
Last puzzle, and it's the sharp edge of cross-entropy — the one behind that fourth curve. What if the model is *absolutely certain* and wrong: it declares "0% chance this is spam" and it **is** spam?

Picture a wall you're told never to touch, with a meter that charges more the closer your bumper creeps. A foot away: a little. Six inches: double. Three inches: double again. The charge climbs with **no ceiling** — and if you actually kiss the wall, the meter refuses to show a number at all.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A car bumper creeping toward a wall you must never touch, with a charge meter. A foot away costs a little, six inches double, three inches double again, climbing with no ceiling. Kissing the wall shows no number at all. A tiny epsilon margin keeps the bumper off the wall."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The closer to the wall, the bigger the charge — with no ceiling</text><rect x="468" y="30" width="22" height="120" fill="#D9CFC0" stroke="#8a7d68"/><text x="479" y="96" text-anchor="middle" fill="#6b4f22" font-size="10" transform="rotate(90 479 96)">the WALL — never touch</text><rect x="60" y="70" width="56" height="30" rx="4" fill="#2A7B9B"/><text x="88" y="90" text-anchor="middle" fill="#fff" font-size="10">🚗</text><path d="M116 85 l40 0" stroke="#6B645E" stroke-width="1.5" stroke-dasharray="4,3"/><polygon points="156,85 148,81 148,89" fill="#6B645E"/><rect x="110" y="120" width="36" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="128" y="133" text-anchor="middle" fill="#2D8B55">$1</text><rect x="250" y="120" width="36" height="18" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="268" y="133" text-anchor="middle" fill="#9A7208">$2</text><rect x="370" y="120" width="40" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="390" y="133" text-anchor="middle" fill="#C93B3B">$4…💥</text><text x="128" y="156" text-anchor="middle" fill="#6B645E" font-size="9">a foot away</text><text x="268" y="156" text-anchor="middle" fill="#6B645E" font-size="9">6 inches</text><text x="390" y="156" text-anchor="middle" fill="#6B645E" font-size="9">3 inches — no ceiling</text><line x1="456" y1="30" x2="456" y2="150" stroke="#2D8B55" stroke-width="1.5" stroke-dasharray="4,3"/><text x="448" y="52" text-anchor="end" fill="#2D8B55" font-size="9">stop here (ε)</text><text x="448" y="64" text-anchor="end" fill="#2D8B55" font-size="9">tiny safe margin</text></g></svg>
%%%

That runaway bill is exactly `−log` as the probability on the true answer slides toward `0`. **Where the meter breaks down:** a real meter has a top price it can't pass. The math has no cap — right at the wall the penalty is a true `+∞`.

#### One certain mistake, three quick beats

%%% steps
step: the model kisses the wall — it gives the true answer a probability of exactly 0.0
why: not "very small" — actually 0.0, because floating-point rounding flattened a tiny number
step: the blow-up — the score becomes `-log(0)`, which is `+∞`, and infinity soon turns into a **NaN**
why: the log's climb has no ceiling, so this really is infinite
step: the spread — that one NaN leaks into every step after it
why: nothing crashes, no error appears. The numbers just quietly turn to garbage.
%%%

Here's the cliff, plus the little green line that saves you from it:

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="The minus-log curve shoots to infinity as probability approaches zero; clamping the probability just above zero with a tiny epsilon keeps the loss finite"><g font-family="monospace" font-size="11"><line x1="70" y1="130" x2="470" y2="130" stroke="#E5DFD6" stroke-width="1.5"/><line x1="90" y1="20" x2="90" y2="130" stroke="#E5DFD6" stroke-width="1"/><path d="M96 24 C 120 60, 170 112, 260 124 C 350 130, 420 131, 460 131" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><text x="150" y="30" fill="#C93B3B">p → 0 : −log(p) → +∞  (then 💥 NaN)</text><line x1="112" y1="20" x2="112" y2="135" stroke="#2D8B55" stroke-width="1.5" stroke-dasharray="4,3"/><text x="118" y="150" fill="#2D8B55">clamp at ε (tiny)</text><text x="330" y="150" fill="#6B645E">probability of correct answer →</text></g></svg>
%%%

A [[NaN||"Not a Number" — what a computer makes from an undefined operation like ∞ − ∞. Once one appears it spreads through every later step.]] is the villain, and it never announces itself.

%%% insight
Picture the morning after: you left a model training overnight, you come back, and the loss column reads `nan, nan, nan…` all the way down. No error message. Everyone hits this once — and after today you'll recognise it in five seconds.
%%%

#### The neat fix — keep the bumper off the wall
Before taking the log, nudge every probability into a tiny safe margin — say `[0.0000001, 0.9999999]`.

That margin is [[epsilon||A tiny number (like 1e-7) used as a safety margin, so you never take log(0).]] (`ε`), and squeezing the number into it — also called a **clamp** — is [[epsilon clipping||Clamping every predicted probability into a tiny range like [ε, 1−ε] before the log, so cross-entropy can never blow up to infinity.]]. Now `−log` sees a big *finite* number instead of infinity: a fair punishment instead of a poisoned run.

%%% demo id=clip label="reveal the clamped score"
predict: a certain-and-wrong p = 0 would score infinity. Clamped to 0.0000001, does it become something like 2, 16, or a million?
code: p=0.0; print("clamped",round(-np.log(max(p,1e-7)),3))  # clamp p into [1e-7, 1-1e-7]
out: clamped 16.118
take: <b>Epsilon clipping keeps the loss finite.</b> A confident-wrong <code>p=0</code> gives <code>−log(0)=+∞</code>, which later leaks out as a NaN. Clamped to <code>1e-7</code> it becomes <code>≈16.1</code> — a large, honest penalty training can actually use. Crisis averted with one <code>max</code>.
%%%

!!! c-info 🔎
<b>Optional (skippable):</b> real libraries go one better — they compute the loss straight from the raw logits (the "from-logits" trick) so <code>log(0)</code> never happens at all. The idea you now hold — keep the probability off the cliff edge — is exactly its spirit.
!!!

**Victory lap:** that's **three puzzles, three cures**, all yours — and you can say each cure in one breath. Outliers → MAE. Yes/no → cross-entropy. `p = 0` → clamp with ε. One honest thing left: what a score simply *can't* do.

@@@ concept id=c10 tag="The limit" title="A score judges — it doesn't fix (that's a separate job)" gotit="Got the limit"
Let's end on the honest truth about what today's tool *can't* do — because knowing the edge of a tool is what separates a beginner from someone who really gets it.

Picture a **bathroom scale**. It shows a number, and a lower number might be your goal. But the scale itself has never once made anyone lighter. It only *reports*. The eating and the running do the changing.

%%% svg
<svg viewBox="0 0 520 152" role="img" aria-label="A bathroom scale showing a weight number with a goal arrow pointing down. A note says the scale only reports the number and has never once made anyone lighter; the eating and running do the changing."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The scale reports the number — it never made anyone lighter</text><rect x="70" y="38" width="150" height="90" rx="14" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><rect x="96" y="56" width="98" height="40" rx="6" fill="#FDFCF9" stroke="#B3A6D4"/><text x="145" y="84" text-anchor="middle" fill="#5E5191" font-size="18">0.42</text><text x="145" y="116" text-anchor="middle" fill="#6B645E" font-size="10">bathroom scale = the loss</text><path d="M250 70 l40 0" stroke="#6B645E" stroke-width="1.5"/><polygon points="290,70 282,66 282,74" fill="#6B645E"/><text x="270" y="62" text-anchor="middle" fill="#2D8B55" font-size="10">goal: ↓ lower</text><text x="330" y="58" fill="#6B645E">it only REPORTS.</text><text x="330" y="78" fill="#C93B3B">it never moves a weight.</text><text x="330" y="104" fill="#2D8B55">the eating &amp; running</text><text x="330" y="120" fill="#2D8B55">(a separate step)</text><text x="330" y="136" fill="#2D8B55">do the changing</text></g></svg>
%%%

**Where the scale breaks down:** a scale is silent about *how* to change, but a loss is a little more helpful. Because it's smooth, it also hands over a **slope** — a "which way is downhill" hint.

#### Step onto the scale — four beats
Let's walk the analogy all the way to the edge, one beat at a time. Keep that scale picture in your head: today's loss reads `0.42`.

%%% steps
step: **the reading** — the scale shows 68.4 kg; your loss shows 0.42
why: one honest number for "where you stand right now". That is today's entire job: (prediction, target) → 0.42.
step: **the tilt** — stand off-centre and the needle leans; a smooth loss leans too, and that lean is a slope
why: it whispers "downhill is *this* way" — a direction. A real bathroom scale can't even do this much, so the loss is already one step ahead of the analogy.
step: **the full stop** — the scale has never once made anyone lighter, and neither has a loss
why: it **only scores**, and it **does not change any weight**. Not one, not ever. Reading 0.42 all day long changes nothing.
step: **the mover** — a separate step reads that slope and actually edits the weights: the **optimizer**
why: the eating and the running of our analogy. Tomorrow you learn how that slope is computed (**gradients and backpropagation**); the step that spends it — the optimizer itself — gets its own day soon after.
%%%

So the day ends with a clean seam: today's tool judges, and a different tool fixes.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="A loss scores a prediction and hands over a slope, but a separate optimizer step is what actually changes the weights; the slope itself is computed on day five"><g font-family="monospace" font-size="12" text-anchor="middle"><rect x="30" y="40" width="130" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="95" y="58" fill="#5E5191">loss (today)</text><text x="95" y="73" fill="#6B645E" font-size="10">reads how wrong</text><text x="196" y="58" fill="#6B645E">→ slope →</text><text x="196" y="74" fill="#9A938A" font-size="9">Day 5: how it's found</text><rect x="255" y="40" width="150" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="330" y="58" fill="#1a5c38">optimizer</text><text x="330" y="73" fill="#6B645E" font-size="10">actually moves weights</text><text x="460" y="58" fill="#9A938A" font-size="10">judge,</text><text x="460" y="73" fill="#9A938A" font-size="10">then fix</text></g></svg>
%%%

That separate step is the [[optimizer||The separate step that reads the loss's slope and actually changes the weights to make the loss smaller. It gets its own lesson shortly after gradients.]] — the eating and the running of our analogy.

%%% insight
This split is *good news*, not a limitation to mourn. Because scoring and fixing are separate jobs, you can swap either one without touching the other — keep your loss, try a smarter optimizer; keep your optimizer, change what "wrong" means. That clean seam is why the same handful of losses shows up in every model ever built.
%%%

One more honest edge, and it's worth ten seconds: a loss only measures agreement with the labels **you** handed it. Hand it wrong labels and it will cheerfully, precisely march your model toward the wrong answer. The score is a mirror, not a judge of your data.

**You now know exactly where a loss stops.** It's the judge, not the fix — and tomorrow you start building the fix. One page to gather it all, and you're done.

@@@ concept id=c11 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built the whole scorecard. Here's a way to hold the day at once: picture it as a single **round of golf**, with the beats below as the holes you played, in order. Each hole only made sense because of the one before it.

**Where the golf-round picture breaks down:** a real round ends when you sink the last putt, but this scorecard is one you'll come back to for years.

The holes you played, in order:
- **Hole 1 · The score.** A **loss** turns "how wrong was that guess?" into one number (lower is better), comparing a **prediction** to a **target**. The **cost** is that loss averaged over all examples.
- **Hole 2 · Why smooth.** Counting mistakes (the **0/1 loss**) is flat, so it gives no direction. Every real loss is **smooth** — it measures how far off. Report accuracy separately.
- **Hole 3 · Guessing numbers.** **MSE** = average of the squared misses; squaring makes misses always positive and big misses count more. Soft spot: one **outlier** hijacks it → **MAE** or **Huber**.
- **Hole 4 · Yes/no and one-of-many.** **Cross-entropy** = `−log(prob of the correct answer)`. Pair **sigmoid → binary cross-entropy** for yes/no, **softmax → categorical cross-entropy** for one-of-many.
- **Hole 5 · Reading the score.** Watch the *trend*, not one number: a healthy curve drops then flattens. Compare against a baseline — a pure guesser scores `ln K` (0.69 for yes/no, 2.30 for ten classes).
- **Hole 6 · Three traps, three cures.** Outliers → MAE. MSE **stalls** on confident-wrong yes/no calls → cross-entropy. A `p=0` sends cross-entropy to `+∞`/NaN → **epsilon clipping**.
- **Hole 7 · The honest limit.** A loss only scores and hands over a slope. It never moves a weight — that's the **optimizer**, which arrives shortly after tomorrow's gradients.

%%% svg
<svg viewBox="0 0 520 216" role="img" aria-label="A mini-golf scorecard for today's round listing the seven holes played in order: the score, why smooth, guessing numbers, yes-no and one-of-many, reading the score, three traps three cures, and the honest limit. Each hole builds on the one before."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Today's round — the holes you played, in order ⛳</text><rect x="30" y="28" width="460" height="176" rx="8" fill="#FDFCF9" stroke="#C9B77A" stroke-width="2"/><text x="48" y="52" fill="#9A7208">1 ·</text><text x="78" y="52" fill="#2C2A28">The score — one number, lower is better</text><text x="48" y="76" fill="#9A7208">2 ·</text><text x="78" y="76" fill="#2C2A28">Why smooth — a stamp gives no hint</text><text x="48" y="100" fill="#9A7208">3 ·</text><text x="78" y="100" fill="#2C2A28">Guessing numbers — MSE (outlier → MAE)</text><text x="48" y="124" fill="#9A7208">4 ·</text><text x="78" y="124" fill="#2C2A28">Yes/no &amp; one-of-many — cross-entropy</text><text x="48" y="148" fill="#9A7208">5 ·</text><text x="78" y="148" fill="#2C2A28">Reading it — watch the trend vs a baseline</text><text x="48" y="172" fill="#9A7208">6 ·</text><text x="78" y="172" fill="#2C2A28">Three traps, three cures — outlier, stall, log(0)</text><text x="48" y="196" fill="#9A7208">7 ·</text><text x="78" y="196" fill="#2C2A28">The honest limit — it scores, doesn't fix</text><path d="M40 58 C 24 92, 24 124, 40 158" fill="none" stroke="#B8AEA2" stroke-width="1.5" stroke-dasharray="3,3"/><text x="18" y="118" fill="#6B645E" font-size="9" transform="rotate(-90 18 118)">each builds on the last</text></g></svg>
%%%

And the one picture to keep — the four workhorse scores at a glance:

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Recap: MSE and MAE score numeric guesses; binary and categorical cross-entropy score classification; every one is a smooth how-wrong number"><g font-family="monospace" font-size="12" text-anchor="middle"><text x="140" y="26" fill="#3A342E" font-weight="bold">guess a NUMBER</text><rect x="40" y="40" width="90" height="34" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="85" y="61" fill="#9A7208">MSE</text><rect x="150" y="40" width="90" height="34" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="195" y="61" fill="#9A7208">MAE</text><text x="140" y="92" fill="#6B645E" font-size="10">→ how far off, averaged</text><text x="390" y="26" fill="#3A342E" font-weight="bold">pick a CLASS</text><rect x="290" y="40" width="100" height="34" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="340" y="61" fill="#5E5191" font-size="11">BCE (yes/no)</text><rect x="400" y="40" width="100" height="34" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="450" y="61" fill="#5E5191" font-size="11">CCE (many)</text><text x="390" y="92" fill="#6B645E" font-size="10">→ −log(prob of correct)</text><text x="260" y="120" fill="#1a5c38" font-size="11">every one: a smooth "how wrong" number the network shrinks</text></g></svg>
%%%

#### Cheat-sheet · which score to reach for
%%% table
:: Your task :: Output shape :: Loss :: Watch out for
Guess a number :: raw number (linear) :: MSE :: one outlier hijacks it → switch to MAE / Huber
Yes / no :: one probability (sigmoid) :: binary cross-entropy :: never use MSE here — it stalls
One of 3+ classes :: probabilities that sum to 1 (softmax) :: categorical cross-entropy :: `log(0)` → clamp with ε
%%%

#### Cheat-sheet · the words you met today
%%% jargon
loss | one number for how wrong ONE prediction is; lower is better, 0 is perfect
cost | the loss averaged over the whole batch — the single number training shrinks
prediction | the guess the forward pass produced
target | the true answer (also called the label) the loss compares against
0/1 loss | just counting mistakes; flat, so it can't be trained on
MSE | mean squared error — average of the squared misses; punishes big misses hardest
MAE | mean absolute error — average of the plain miss sizes; calm about outliers
Huber loss | MSE for small misses, MAE for big ones — the best-of-both middle ground
logits | the raw output scores of the last layer, before sigmoid/softmax make probabilities
sigmoid | squashes a number into one probability (0 to 1) — for yes/no
softmax | turns raw scores into probabilities that sum to 1 — for 3+ classes
binary cross-entropy | −log(prob of the correct answer) for yes/no; hammers confident-and-wrong
categorical cross-entropy | the same −log(prob of correct), spread over 3+ classes
chance level | the loss a pure guesser gets: ln K for K classes (0.69 for yes/no) — your baseline
epsilon clipping | clamp probabilities to [ε, 1−ε] so −log can never blow up to infinity
optimizer | the separate step that reads the loss's slope and actually changes the weights (its own lesson, just after gradients)
%%%

That's the whole day. Next you'll see how the network *finds* this score's slope — **gradients and backpropagation**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is a loss function, in one line? | a:1 | The step that updates the weights | One number saying how wrong a prediction is, comparing it to the target (lower is better) | The network's number of layers | The learning rate | fb: A loss takes (prediction, target) and turns "how far off was that guess?" into one number. Lower is better; 0 is a perfect guess. The step that updates weights is the optimizer.
q: You're predicting house prices and a few freak mansions are wild outliers. Which loss keeps one crazy point from hijacking the score? | a:2 | MSE, because it squares errors | Cross-entropy | MAE (or Huber), because it doesn't square the miss | counting mistakes | fb: MSE squares each miss, so one giant error explodes and dominates. MAE takes the plain miss size, so an outlier counts in proportion — Huber is the smooth middle ground.
q: For "cat vs dog vs bird" (three classes), which output-and-loss pairing is right? | a:2 | linear output + MSE | sigmoid + binary cross-entropy | softmax + categorical cross-entropy | no loss is needed | fb: Softmax turns the raw scores (logits) into probabilities that sum to 1 — one whole pizza sliced across the classes — and categorical cross-entropy scores −log of the slice you gave the correct class. Sigmoid + binary cross-entropy is for yes/no; MSE is for numbers.
q: Why does cross-entropy beat MSE for a yes/no classifier? | a:2 | Cross-entropy uses less memory | It counts mistakes directly | Its learning push stays strong when the model is confidently wrong, while MSE+sigmoid goes nearly flat there and stalls | It never needs a probability | fb: With a sigmoid output, MSE's push shrinks toward 0 exactly when the model is confidently wrong. Cross-entropy keeps a strong push right there, so it fixes loud mistakes fast.
%%%

@@@ produce id=produce tag="Produce" title="Watch a loss reward confidence — and punish being loudly wrong" gotit="Done"
Time to see today's big idea with your own eyes. You'll compute MSE and cross-entropy on tiny examples and **watch** how each reacts as a guess gets better or worse. **Predict first:** as a classifier's probability on the *correct* answer climbs from 0.1 to 0.99, does cross-entropy go up or down — and where does it change fastest: in the middle of that climb, or right down at the confident-wrong end? Then run it and **notice** whether your guess held. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-04-loss/experiment.py`. (1) Write `mse(pred, target)` and `mae(pred, target)`, and print both for `pred=[2.5, 0, 2]`, `target=[3, −0.5, 2]` — **notice** MSE prints `≈ 0.167`. (2) Now stage the hijack on a crowd: twenty houses each missed by `0.5`, then bolt on **one** mansion missed by `100`, and print both scores before and after — **watch** MSE leap from `0.25` to about `476` while MAE only walks from `0.5` to about `5.2`. (3) Write `cross_entropy(p)` as `−log(p)` and print it for `p = [0.99, 0.6, 0.1, 1e-7]`, plus the two step-to-step drops (`0.1→0.6` and `0.6→0.99`) — **notice** it falls toward 0 as `p` nears 1, shoots up as `p` nears 0, and that clamping `p` with `max(p, 1e-7)` keeps a `p=0` case finite (`≈16.1`) instead of `+∞`. Run with `python3 sessions/m02-the-neuron/day-04-loss/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 4 artifact.

Create sessions/m02-the-neuron/day-04-loss/experiment.py that, with a comment on each step:
1. Defines mse(pred,target)=np.mean((pred-target)**2) and mae(pred,target)=np.mean(np.abs(pred-target)); prints both for pred=[2.5,0,2], target=[3,-0.5,2] (expect MSE~0.167, MAE~0.333).
2. Stages the outlier hijack on a CROWD: 20 examples each missed by 0.5, then the same 20 plus ONE mansion missed by 100. Print MSE and MAE for both batches (expect MSE 0.25 -> ~476.4, MAE 0.5 -> ~5.2) and print each growth factor, so the numbers show MSE is amplified about 1900x while MAE only grows about 10x — the outlier is FELT by MAE, not amplified.
3. Defines cross_entropy(p)=-np.log(np.clip(p,1e-7,1-1e-7)) and prints it for p in [0.99,0.6,0.1,1e-7]; show it falls toward 0 as p->1 and shoots up as p->0, and that the clip keeps p=0 finite (~16.1) instead of +infinity.
4. Prints WHERE cross-entropy changes fastest: the drop from p=0.1 to 0.6 versus the drop from p=0.6 to 0.99 (expect ~1.79 vs ~0.50), so the numbers show it changes fastest near p->0 — the confident-wrong end — and flattens as p->1.
5. Prints a one-line reminder that a loss only SCORES and hands over a slope — it does not change any weight (that is the optimizer, a later lesson).
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the scorecard, reacting)
- **MSE** on the close guess prints `≈ 0.167` (MAE `≈ 0.333`) — a small score, because the guesses are close.
- The crowd test: adding **one** mansion missed by `100` sends MSE from `0.25` to `≈ 476` — about **1,900× bigger** — while MAE only goes `0.5` → `≈ 5.2`, about **10×**. Same single bad row: MSE *amplified* it, MAE merely *felt* it. That's the hijack, live, and the reason MAE is the calmer scorecard.
- **Cross-entropy** falls toward `0` as the probability on the correct answer climbs to `0.99`, and shoots up as it slides toward `0.1` — the confident-and-wrong sting, live.
- **Where it changes fastest** (your prediction from the top): the drop from `p=0.1` to `p=0.6` is about `1.79`, while `p=0.6` to `p=0.99` is only about `0.50`. So cross-entropy moves fastest down at the **confident-wrong end** (`p → 0`) and flattens out as `p → 1`. If you guessed "in the middle", that's a great mistake to have made once — the steepness lives at the bad end, which is exactly what makes the loss push hardest there.
- The `p=1e-7` case prints a big but **finite** number (`≈16.1`) instead of `+∞` — that's epsilon clipping saving you from a NaN.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-04-loss/log.md`: (1) what a loss is, and what the "cost" adds on top; (2) which loss you'd pick for house prices vs. spam-or-not vs. cat/dog/bird, and why; (3) one trap from today (outlier, stall, or log(0)) and its one-line cure.
!!!

@@@ fin
