---
quest_id: wf5-d01-mlp-mnist
mode: concept
donor: v9-base.donor
page_title: "Module 5 · Day 1 — Your First Model (MLP on MNIST)"
module_label: "Module 5 · Train · Day 1"
title: "Your First Model"
subtitle: "An MLP That Reads Handwriting"
brand_sub: "Foundations · M5 Day 1"
spine: "stack"
nav_prev_href: "../../m03-attention/review.html"
nav_prev_label: "M3 · Review Gate"
nav_next_href: "../day-02-backward-pass/lesson.html"
nav_next_label: "The Backward Pass, Wired by Hand"
fin_title: "Module 5 · Day 1 complete! 🏆"
fin_body: "Nice work — you've built <b>Your First Model</b>: a multi-layer perceptron that turns 784 pixels into 10 digit-scores. You stacked layers, put a bend between them, and ran a forward pass end to end.<br>Next up: <b>The Backward Pass, Wired by Hand</b> — where this untrained model finally starts to learn."
notebook_yardstick: 00-neural-networks/fundamentals/09_complete_mnist_implementation.ipynb
coverage_topics:
  - {topic: MNIST task, keywords: [mnist, 28×28, handwritten, 10 classes]}
  - {topic: flattening the image, keywords: [flatten, 784, reshape]}
  - {topic: neuron building block, keywords: [weighted sum, bias, neuron]}
  - {topic: layer as a matrix multiply, keywords: [weight matrix, "x @ w", one matmul]}
  - {topic: multi-layer perceptron, keywords: [multi-layer perceptron, mlp, fully-connected, dense]}
  - {topic: hidden layer, keywords: [hidden layer, internal features]}
  - {topic: nonlinear activation relu, keywords: [relu, nonlinear, activation]}
  - {topic: why nonlinearity is required, keywords: [collapse, one linear layer, straight-line map]}
  - {topic: output layer logits, keywords: [logits, 10 scores, output layer]}
  - {topic: forward pass, keywords: [forward pass, left to right, pixels into scores]}
  - {topic: reading the prediction, keywords: [argmax, highest score, guessed digit]}
  - {topic: parameter count intuition, keywords: [parameters, weights + biases, wider]}
  - {topic: random initialization, keywords: [random, initialization, untrained, ~10%]}
  - {topic: single perceptron ancestor, keywords: [single perceptron, one straight line, linear neuron]}
  - {topic: failure linear collapse, keywords: [no activation, collapse, composition of linear]}
  - {topic: failure underfitting, keywords: [too few hidden units, underfitting, widen]}
  - {topic: failure dead relu, keywords: [dead relu, permanently negative, leaky relu]}
  - {topic: he initialization, keywords: [he init, he initialization, sensible init, variance-preserving, relu init]}
  - {topic: single neuron capability limit, keywords: [xor, not linearly separable, curved boundary]}
  - {topic: mlp spatial-structure limit, keywords: [ignores 2d, spatial structure, convolutional]}
---

@@@ hero
@lede Imagine a machine that has never seen a number in its life, and yet by the end of this hour it can look at your messy, hand-scrawled "7" and just *know* it's a seven. How? You are going to build it — and the surprising part is you already have every piece. Today we stack the tiny building block from Module 2 into your first real network.
@goal By the end you can describe a 2-layer MLP for handwritten digits (784 → 128 → 10), run one forward pass in your head, read off its guess, and say exactly why you must put a bend between the layers.

@@@ concept id=c1 tag="The task" title="70,000 handwritten digits" gotit="Got the task"
Let's start with the job. Picture a big shoebox holding **70,000 small photos**, each one a single handwritten digit — a wobbly 3, a rushed 7, a careful 0. Every photo is tiny and gray: a grid of **28 by 28 dots**, where each dot (a **pixel**) is just a brightness number from 0 (black) to 255 (white). This box is called [[MNIST||70,000 small grayscale images of handwritten digits 0–9, each 28×28 pixels — the classic "hello world" dataset of neural-network classification.]], and it is the "hello world" of neural networks. Your job: look at one photo and answer with the right digit, 0 through 9. Ten possible answers — we call each answer a **class**.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A 28 by 28 grid of pixels showing a handwritten digit on the left, an arrow pointing to ten class labels 0 through 9 on the right"><g font-family="monospace" font-size="12"><rect x="30" y="20" width="90" height="90" rx="4" fill="#FDF9F3" stroke="#7C6DAA" stroke-width="2"/><path d="M50 40 L100 40 L70 100" fill="none" stroke="#3A342E" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/><text x="75" y="125" fill="#6B645E" text-anchor="middle">28×28 pixels</text><text x="160" y="70" fill="#6B645E" text-anchor="middle">→ guess →</text><g text-anchor="middle"><rect x="230" y="52" width="260" height="34" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="360" y="74" fill="#5E5191" font-weight="bold">0  1  2  3  4  5  6  7  8  9</text><text x="360" y="108" fill="#6B645E" font-size="11">10 classes — one is the true digit</text></g></g></svg>
%%%

**What the shoebox picture gets right:** it captures the scale (a huge pile of separate examples) and the honest goal (name the digit). **Where it breaks down:** a real photo box lets you flip through pages, but our model sees every image as a flat sheet of numbers — it has no idea "up" or "next to" mean anything. Hold that thought; it comes back at the very end.

#### Why we stack toward this
One neuron from Module 2 can barely tell a bright pixel from a dark one. Reading a *digit* — a whole shape — is far harder. So the plan for today is to **stack** simple pieces into something that can. Everything below is a step up that stack.

@@@ concept id=c2 tag="Flatten it" title="From a grid to one long row" gotit="Got the flatten"
A neuron eats a *list* of numbers, not a *grid*. So before the model can look at an image, we have to lay the grid out flat. Picture unraveling a knitted scarf: the 28 rows of the grid get laid end to end into **one single row of 28 × 28 = 784 numbers**. This is called [[flattening||Reshaping a 2-D image grid into a 1-D vector by laying its rows end to end, so a dense layer can read it as one list of inputs.]] the image, and the long row it produces is the **input vector**.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A small pixel grid on the left, an arrow, and a long thin row of boxes on the right labeled 784 numbers"><g font-family="monospace" font-size="12"><g stroke="#7C6DAA" stroke-width="1.2" fill="#EDE9F8"><rect x="30" y="35" width="18" height="18"/><rect x="48" y="35" width="18" height="18"/><rect x="66" y="35" width="18" height="18"/><rect x="30" y="53" width="18" height="18"/><rect x="48" y="53" width="18" height="18"/><rect x="66" y="53" width="18" height="18"/><rect x="30" y="71" width="18" height="18"/><rect x="48" y="71" width="18" height="18"/><rect x="66" y="71" width="18" height="18"/></g><text x="57" y="105" fill="#6B645E" text-anchor="middle">28×28 grid</text><text x="130" y="66" fill="#6B645E" text-anchor="middle">→ unravel →</text><g stroke="#2D8B55" stroke-width="1.2" fill="#E8F5EE"><rect x="185" y="53" width="15" height="18"/><rect x="200" y="53" width="15" height="18"/><rect x="215" y="53" width="15" height="18"/><rect x="230" y="53" width="15" height="18"/><rect x="245" y="53" width="15" height="18"/><rect x="260" y="53" width="15" height="18"/><rect x="275" y="53" width="15" height="18"/><rect x="290" y="53" width="15" height="18"/><rect x="305" y="53" width="15" height="18"/></g><text x="330" y="66" fill="#6B645E">… </text><text x="300" y="42" fill="#1a5c38" text-anchor="middle" font-weight="bold">one row: 784 numbers</text><text x="300" y="90" fill="#6B645E" text-anchor="middle" font-size="11">shape (784,) — the input vector</text></g></svg>
%%%

**What the scarf picture gets right:** the grid is not thrown away — every dot is still there, just re-laid into a single line. **Where it breaks down:** unraveling a scarf ruins it, but here we keep every number; the cost is subtler — the flattened row forgets which pixels used to sit *next to* each other. Again, remember that; it is exactly the limit we hit at the end.

So one image becomes a vector of shape `(784,)`. That vector is what the very first layer of the network will read.

@@@ concept id=c3 tag="One line only" title="A single neuron draws one straight line" gotit="Got the limit"
Before we stack anything, let's meet the ancestor — and see why one piece is not enough. A single [[neuron||The building block from Module 2: it takes a weighted sum of its inputs, adds a bias, then applies an activation. On its own it draws one straight dividing line.]] (the building block from Module 2) computes a weighted sum of its inputs plus a bias. Geometrically, that does exactly one thing: it draws **one straight line** and calls everything on one side "yes" and everything on the other side "no." This lone-neuron model has a historical name — the **single perceptron**, the 1950s ancestor of everything we build today.

Here is an everyday picture. Imagine a **strict bouncer at a door who is only allowed to use one straight rope** to split the crowd. He can angle the rope any way he likes, but it stays a single straight line — everyone on one side gets in, everyone on the other side is out. If the "let in" people happen to sit neatly on one side, he does great. **What this bouncer picture gets right:** one neuron, like one straight rope, makes exactly one clean straight cut. **Where it breaks down:** a real bouncer can walk around, look people up, and change his mind case by case; the neuron cannot — the rope is *forced* to stay straight, so if the people he must let in are scattered in opposite corners, no single straight rope can ever separate them.

That corner-scattered case has a famous name: **XOR** — "on when exactly *one* of two inputs is on." Its two classes sit on opposite corners, so any straight line you draw leaves at least one point on the wrong side. Slide the line below every which way — you top out at 3 of 4. Then tick "add a hidden layer" and watch two lines, joined together, finally carve out the region one line never could.

%%% viz src=../../viz/xor-limit.html title="why one neuron can't do XOR" caption="interactive — slide the single line (stuck at 3/4), then add a hidden layer"
%%%

**What this shows:** a single perceptron is stuck at a **straight** boundary, so any curved or diagonal-corner pattern beats it. This is the [[capability limit||A single linear neuron can only separate data that one straight line splits — it cannot solve non-linearly-separable problems like XOR, or the curved boundaries between digit shapes.]] of a single linear neuron, and it is exactly why we cannot read curvy digit shapes with one neuron. The fix — the whole rest of this lesson — is to **stack** layers of neurons *with a bend between them*, so simple straight pieces combine into curved boundaries.

@@@ concept id=c4 tag="A layer" title="A whole layer in one multiply" gotit="Got the layer"
So we need more than one neuron. Put a **group of neurons side by side, all reading the same input**, and you have a [[layer||A group of neurons that all read the same input vector; its weights form a matrix W and its biases a vector b, computed together as z = x @ W + b.]]. Here is the beautiful part: you don't run each neuron one at a time. You stack every neuron's weights into one **weight matrix `W`** and every neuron's bias into one **bias vector `b`**, and compute the entire layer in a single matrix multiply.

Think of a **panel of 128 judges at a talent show, all watching the same act at once**. Each judge listens to the same 784 details of the performance, weighs them their own way, and holds up one score. Instead of walking down the row asking each judge separately, you collect all 128 scorecards in a single sweep. **What the judging-panel picture gets right:** every judge (neuron) reads the *same* input and each produces exactly one number, and you gather them together in one pass — that "one pass" *is* the matrix multiply. **Where it breaks down:** real judges talk to each other and change their minds; the neurons in one layer are independent and silent — each computes its score with no idea what its neighbors decided.

In words: multiply the input row by the weight matrix, then add the bias row. In symbols: `z = x @ W + b`. The `@` means matrix-multiply. Let's watch the shapes so nothing feels like magic:

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Shape diagram: x of shape 1 by 784, times W of shape 784 by 128, plus b of shape 128, equals z of shape 1 by 128"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="20" y="48" width="70" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="55" y="63" fill="#1a5c38" font-weight="bold">x</text><text x="55" y="77" fill="#6B645E" font-size="10">(1,784)</text><text x="105" y="70" fill="#6B645E">@</text><rect x="125" y="48" width="80" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="165" y="63" fill="#5E5191" font-weight="bold">W</text><text x="165" y="77" fill="#6B645E" font-size="10">(784,128)</text><text x="220" y="70" fill="#6B645E">+</text><rect x="238" y="48" width="66" height="34" rx="5" fill="#FDF3E8" stroke="#C99A12" stroke-width="2"/><text x="271" y="63" fill="#8A6D3B" font-weight="bold">b</text><text x="271" y="77" fill="#6B645E" font-size="10">(128,)</text><text x="320" y="70" fill="#6B645E">=</text><rect x="340" y="48" width="80" height="34" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="380" y="63" fill="#C93B3B" font-weight="bold">z</text><text x="380" y="77" fill="#6B645E" font-size="10">(1,128)</text><text x="260" y="112" fill="#6B645E" font-size="11">784 numbers in → 128 numbers out, all in one matrix-multiply</text></g></svg>
%%%

**What to notice in the shapes:** the middle number matches (784 lines up with 784) and cancels, leaving `(1, 128)` — one output number per neuron. So a layer with 128 neurons turns 784 inputs into 128 outputs in a single step. Stacking the weights into a matrix is not just tidy; it is why a whole layer costs one fast multiply instead of 128 slow loops.

@@@ concept id=c5 tag="Stack + bend" title="The MLP: stack layers, bend between them" gotit="Got the MLP"
Now we **stack**. Put layers back to back and you get a [[multi-layer perceptron||An MLP: an input layer, one or more hidden layers, and an output layer, each layer fully connected to the next. The simplest real neural network.]] — an **MLP**. Each layer is **fully connected** (also called **dense**): every input feeds every neuron in the next layer. The layers in the middle — the ones that are neither the raw input nor the final answer — are the [[hidden layers||The intermediate layer(s) between input and output. They let the network build its own internal features — combinations of pixels that hint at a stroke or a curve.]]. They are where the network builds its own internal features: little combinations of pixels that hint at "a loop up top" or "a straight stroke down the side."

But here is the trap. If you stack two plain layers with **nothing** between them, they collapse. Picture **folding a paper map**. Two straight folds, one after the other, still leave you with flat, straight creases — you cannot fold a map into a curved bowl no matter how many straight folds you add. But now **crumple** the map once in the middle before the second fold, and suddenly the paper can take a shape no sequence of flat folds could. **What the map-folding picture gets right:** stacking plain (linear) layers is like stacking flat folds — you stay flat and straight, and the second layer buys you nothing new. The crumple is the bend, and it is what unlocks shapes the flat folds never could. **Where it breaks down:** a real crumple is random and messy; the network's bend is a precise, simple rule (ReLU: keep positives, zero negatives) applied the same way every time, so the network can *learn* which bends help.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Two weight matrices W1 then W2 with nothing between them collapse into a single matrix; with a ReLU bend between them they cannot collapse"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="30" y="20" width="55" height="28" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="57" y="39" fill="#5E5191">W₁</text><text x="100" y="39" fill="#6B645E" font-size="11">then</text><rect x="130" y="20" width="55" height="28" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="157" y="39" fill="#5E5191">W₂</text><text x="215" y="39" fill="#6B645E">=</text><rect x="245" y="16" width="130" height="34" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="310" y="38" fill="#C93B3B" font-weight="bold">one matrix W₁@W₂</text><text x="410" y="39" fill="#C93B3B" font-size="10">no bend → 1 layer</text><rect x="30" y="72" width="55" height="28" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="57" y="91" fill="#5E5191">W₁</text><rect x="95" y="72" width="62" height="28" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="126" y="91" fill="#1a5c38" font-weight="bold">ReLU</text><rect x="167" y="72" width="55" height="28" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="194" y="91" fill="#5E5191">W₂</text><text x="252" y="91" fill="#2D8B55" font-size="11" text-anchor="start">a bend → cannot fold flat → real depth</text></g></svg>
%%%

**What the picture says:** two matrix-multiplies back to back are the same as **one** bigger matrix-multiply, `(x@W1)@W2 = x@(W1@W2)`. No matter how many plain layers you stack, they secretly fold into a single straight-line map — and we just saw a single straight line can't even do XOR. The cure is to insert a [[nonlinear activation||A simple function applied after a layer's matmul+bias that is not a straight line — ReLU is the default. It keeps stacked layers from collapsing into one linear map.]] — a **bend** — between the layers. The default bend is **ReLU**: `ReLU(z) = max(0, z)`, "keep positives, zero out negatives."

!!! c-info 🪜
<b>Math Ladder — why linear ∘ linear collapses.</b>
<br><b>1 · In words:</b> two plain matmuls with no bend between them do the same job as one matmul, so the second layer bought you nothing.
<br><b>2 · Scalars first:</b> chain `f(x)=a·x+p` then `g(u)=b·u+q`. Compose: `g(f(x)) = b·(a·x+p)+q = (ab)·x + (bp+q)` — still `slope·x + shift`, one linear step.
<br><b>3 · Matrix version:</b> `(x@W1)@W2 = x@(W1@W2)`, and `W1@W2` is just one matrix. The two linear layers really are one.
<br><b>4 · Sanity check:</b> now clip with a bend first — `ReLU(x@W1)@W2` — and negatives get zeroed <em>before</em> the second matmul, producing values no single matmul can. That clip is why the hidden layer earns its keep.
!!!

@@@ concept id=c6 tag="Forward pass" title="Pixels in, ten scores out" gotit="Got the forward pass"
Now let's run the whole thing left to right — the [[forward pass||The left-to-right flow that turns an input into an output: input → hidden layer (matmul + bias + activation) → output layer, producing the final scores.]]. This is the exact route every image takes, and it is just the two layers you now understand, wired in order. The final layer, called the **output layer**, has exactly **10** neurons — one per digit class — and its 10 outputs are called [[logits||The raw, un-normalized scores the output layer produces — one per class. A bigger logit means the model leans toward that class. Turning them into probabilities is softmax, on Day 4.]]: raw scores, one per possible digit.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Forward pass: input 784, matmul W1 plus b1 then ReLU to a hidden layer of 128, matmul W2 plus b2 to an output of 10 logits"><g font-family="monospace" font-size="12" text-anchor="middle"><rect x="15" y="55" width="70" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="50" y="70" fill="#1a5c38" font-weight="bold">image</text><text x="50" y="84" fill="#6B645E" font-size="10">(1,784)</text><text x="110" y="66" fill="#6B645E" font-size="10">@W₁+b₁</text><text x="110" y="80" fill="#2D8B55" font-size="10">ReLU</text><rect x="150" y="55" width="80" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="190" y="70" fill="#5E5191" font-weight="bold">hidden</text><text x="190" y="84" fill="#6B645E" font-size="10">(1,128)</text><text x="262" y="70" fill="#6B645E" font-size="10">@W₂+b₂</text><rect x="300" y="55" width="80" height="34" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="340" y="70" fill="#C93B3B" font-weight="bold">logits</text><text x="340" y="84" fill="#6B645E" font-size="10">(1,10)</text><text x="418" y="74" fill="#6B645E">→ digit</text><text x="260" y="122" fill="#6B645E" font-size="11">784 pixels → 128 hidden features → 10 class scores, in one left-to-right pass</text></g></svg>
%%%

The whole model fits on one line: `logits = ReLU(x @ W1 + b1) @ W2 + b2`. That line is written the way math nests — the *innermost* part happens *first*. So to follow the actual left-to-right flow of the data (matching the picture above), read the formula **from the inside out**, starting at `x`:

- **Step 1 — Input (start here):** the flattened image `x`, shape `(1, 784)`. This is the `x` sitting deepest inside the formula.
- **Step 2 — Hidden layer:** `h = ReLU(x @ W1 + b1)` — `W1` is `(784, 128)`, so `h` is `(1, 128)`. The ReLU bend is what keeps this from collapsing into the next layer.
- **Step 3 — Output layer (end here):** `logits = h @ W2 + b2` — `W2` is `(128, 10)`, so `logits` is `(1, 10)`: ten scores, one per digit.

In short: the data marches `x → hidden → logits` (left to right in the picture), which is the same as reading the one-line formula from its inner `x` outward. Nothing is going backward — "inside-out" and "left-to-right" describe the *same* single trip.

**Notice the shapes flow:** 784 → 128 → 10. Each `@` cancels its matching middle number, and the last one lands on 10 because there are 10 digits to score. That left-to-right flow, turning pixels into 10 scores, *is* the forward pass — the same shape of computation as a giant language model, only smaller.

@@@ concept id=c7 tag="Read + count" title="Reading the guess, and counting the knobs" gotit="Got the readout"
The forward pass ends with 10 scores. How do we turn that into an answer? We take the **biggest** one. The position of the largest logit is the model's guessed digit — this "which index is biggest?" operation is called [[argmax||Short for 'arg-max': it returns the *position* of the largest value, not the value itself. Here it turns 10 logits into a single guessed digit 0–9.]]. If the score for slot 7 is the highest, the model guesses "7." That is the entire read-out; probabilities (softmax) come on Day 4, but the *guess* is already just argmax.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Ten logit bars for digits 0 through 9, with the bar for digit 7 tallest and highlighted, and an arrow reading argmax equals 7"><g font-family="monospace" font-size="11" text-anchor="middle"><g fill="#C9C2D8"><rect x="40" y="60" width="20" height="20"/><rect x="70" y="55" width="20" height="25"/><rect x="100" y="65" width="20" height="15"/><rect x="130" y="50" width="20" height="30"/><rect x="160" y="62" width="20" height="18"/><rect x="190" y="58" width="20" height="22"/><rect x="220" y="66" width="20" height="14"/></g><rect x="250" y="24" width="20" height="56" fill="#C93B3B"/><g fill="#C9C2D8"><rect x="280" y="57" width="20" height="23"/><rect x="310" y="63" width="20" height="17"/></g><g fill="#6B645E"><text x="50" y="94">0</text><text x="80" y="94">1</text><text x="110" y="94">2</text><text x="140" y="94">3</text><text x="170" y="94">4</text><text x="200" y="94">5</text><text x="230" y="94">6</text><text x="260" y="94" fill="#C93B3B" font-weight="bold">7</text><text x="290" y="94">8</text><text x="320" y="94">9</text></g><text x="430" y="45" fill="#C93B3B" font-weight="bold">argmax = 7</text><text x="430" y="62" fill="#6B645E" font-size="10">tallest bar wins</text></g></svg>
%%%

#### The knobs the network will learn
Every number inside `W1, b1, W2, b2` is a **parameter** — a knob the network will eventually tune. Count them: `W1` is 784×128 ≈ 100k, `W2` is 128×10 = 1,280, plus the biases. So this small model already has about **100,000 knobs**. Intuition to keep: a **wider** hidden layer (256 or 1024 instead of 128) means *more* parameters — more capacity to represent shapes, but also more to tune.

#### Right now, every knob is random
We have not trained anything yet. When you first build the model, all those weights start as small **random** numbers. So a fresh, untrained network is guessing blindly — over 10 digits, that is about **10% accuracy**, no better than a coin with ten sides. That is the correct, expected starting point. Learning (Day 2 onward) is the slow process of turning those random knobs into good ones.

@@@ concept id=c8 tag="When it breaks" title="Three quiet ways it goes wrong" gotit="Got the failure modes"
A model can be wired perfectly and still quietly fail — no crash, just bad results. Three failures are worth knowing on day one, because each has a clear cause and a clear fix. Here they are side by side:

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="Three failure modes: linear collapse fixed by adding ReLU, underfitting fixed by widening the hidden layer, and dead ReLU fixed by He initialization or Leaky ReLU"><g font-family="monospace" font-size="11"><rect x="20" y="20" width="480" height="38" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="30" y="36" fill="#C93B3B" font-weight="bold">1 · Linear collapse</text><text x="30" y="51" fill="#6B645E">no bend → stacked layers = one layer  ·  FIX: insert ReLU between layers</text><rect x="20" y="64" width="480" height="38" rx="5" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.5"/><text x="30" y="80" fill="#8A6D3B" font-weight="bold">2 · Underfitting</text><text x="30" y="95" fill="#6B645E">too few hidden units → can't separate classes  ·  FIX: widen the hidden layer</text><rect x="20" y="108" width="480" height="38" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="30" y="124" fill="#5E5191" font-weight="bold">3 · Dead ReLU</text><text x="30" y="139" fill="#6B645E">unit stuck outputting 0 forever  ·  FIX: He initialization (or Leaky ReLU / GELU)</text></g></svg>
%%%

Let's name each one plainly:

#### 1 · Linear collapse
**Symptom:** a deep-looking model that learns no better than a single layer. **Cause:** you forgot the bend — a stack of plain linear layers is still, mathematically, one linear map (the collapse from c5). **Remedy:** insert a nonlinear activation (ReLU) between the layers. This is why the hidden layer *must* have an activation.

#### 2 · Underfitting
**Symptom:** the model can't even get the *practice* digits right. **Cause:** the hidden layer is too small — too few units to represent the features needed to tell a 4 from a 9. **Remedy:** widen the hidden layer (add capacity) so it has enough knobs to represent the shapes.

#### 3 · Dead ReLU
**Symptom:** some hidden units output `0` for *every* input and never recover. **Cause:** the weights and bias push that unit's pre-activation permanently negative, so `max(0, z)` is always 0 — often from a bad random start (or, later, too large a learning rate). **Remedy:** the named technique is [[He initialization||A weight-initialization scheme designed for ReLU networks: it draws each weight from a distribution whose variance is scaled by the number of inputs (roughly 2/n_in), so signals keep a sensible size layer to layer and units are far less likely to start dead.]] — a **variance-preserving** starting scheme built for ReLU: instead of picking weights from any old random range, it scales the random spread by the number of inputs (about `2/n_in`), so units are far less likely to be born permanently negative. If units still die, swap in **Leaky ReLU / GELU**, which keep a tiny slope on the negative side so a stuck unit can climb back to life.

!!! c-warn ⚠️
<b>How a senior engineer catches these:</b> none of the three throws an error. You spot them by <b>logging activation statistics</b> — the fraction of ReLU units stuck at zero, the train-vs-practice gap — instead of only staring at the final accuracy number. A silent failure is still a failure.
!!!

@@@ concept id=c9 tag="The ceiling" title="Why this model can't be the last word" gotit="Got the ceiling"
End with honesty about what you built. Your MLP is real and it works — but remember concepts c1 and c2, where we flattened the 28×28 grid into one long row. The moment we did that, the model **lost all sense of where pixels sit**. To the network, two pixels that were touching in the image are no more related than two on opposite corners.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A 3 shifted to a different corner of the frame produces a completely different flattened row, so the MLP sees it as unrelated"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="30" y="20" width="70" height="70" rx="4" fill="#FDF9F3" stroke="#7C6DAA" stroke-width="1.5"/><path d="M45 32 Q70 30 62 45 Q80 50 55 62" fill="none" stroke="#3A342E" stroke-width="4"/><text x="65" y="105" fill="#6B645E">a "3", top-left</text><rect x="150" y="20" width="70" height="70" rx="4" fill="#FDF9F3" stroke="#7C6DAA" stroke-width="1.5"/><path d="M180 55 Q205 53 197 68 Q215 73 190 85" fill="none" stroke="#3A342E" stroke-width="4"/><text x="185" y="105" fill="#6B645E">same "3", moved</text><text x="285" y="52" fill="#C93B3B" font-weight="bold">→ MLP sees a</text><text x="285" y="68" fill="#C93B3B" font-weight="bold">totally different row</text><text x="285" y="88" fill="#6B645E" font-size="10">it never learned "shift doesn't matter"</text></g></svg>
%%%

**The limit:** because it flattens the image, the MLP ignores **2-D spatial structure** and treats a shifted digit as a brand-new pattern. That is the [[capability limit||This architecture flattens away the 2-D layout, so it cannot exploit spatial structure or the fact that a shifted digit is the same digit — a convolutional network, taught in a later module, is built to fix exactly this.]] of this design — and it is exactly what a **convolutional network** (a later module) is built to fix, by keeping the grid and scanning it with small filters. You have not built the best possible digit reader. You have built the *first* one — and understood every knob in it. That is the whole point of today.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions with instant feedback. If one trips you up, that's a cue to scroll back — not a failing grade.
%%% quiz
q: An MNIST image is a 28×28 grid. What shape is it after flattening for the MLP? | a:1 | (28, 28) | (784,) | (10,) | (128,) | fb: Flattening lays the 28×28 = 784 pixels into one long row of shape (784,) — the input vector the first layer reads.
q: Why must you put a nonlinear activation (like ReLU) between the two layers? | a:2 | It makes the matmul faster | It saves memory | Without it, the two stacked layers collapse into a single linear map | It converts logits to probabilities | fb: (x@W1)@W2 = x@(W1@W2) — no bend means the stack is secretly one linear layer, which can't even do XOR. ReLU is the bend that gives real depth.
q: The output layer produces 10 logits. How does the model pick its guessed digit? | a:1 | The average of the 10 scores | argmax — the position of the largest logit | The first positive score | Whichever is closest to 0.5 | fb: The guess is argmax(logits): the index of the biggest score. Slot 7 highest → guess "7." (Turning scores into probabilities is softmax, on Day 4.)
q: Some hidden units output 0 for every input and never recover — a "dead ReLU." Which named fix targets this at the start? | a:2 | Widen the output layer | Convert logits to probabilities | He initialization (variance-preserving weights for ReLU) | Flatten the image differently | fb: Dead ReLUs often come from a bad random start. He initialization scales the weight spread by ~2/n_in so units are far less likely to be born permanently negative; Leaky ReLU / GELU are the fallback.
%%%

@@@ produce id=produce tag="Produce" title="Build the untrained model and read its guess" gotit="Done"
Time to see today's model with your own eyes — no training yet, just the forward pass you now understand. **Before you write any code, predict two things:** (1) what shape will the output be for one image, and (2) roughly what accuracy will this *untrained*, random-weight model get on the test digits — 10%, 50%, or 95%? Write your two guesses down, then run it and **watch** whether you were right. Pick one path.

#### Option A · write it yourself
Create `sessions/m04-first-model-mlp/day-01-mlp-mnist/experiment.py`. Load MNIST, flatten each image to shape `(784,)`, and scale pixels to `[0,1]`. Initialize random weights `W1 (784,128), b1 (128,), W2 (128,10), b2 (10,)` using **He initialization** (scale each weight by roughly `sqrt(2/n_in)`). Write the forward pass `logits = relu(x @ W1 + b1) @ W2 + b2`, print the shape at every step (expect `(batch, 784) → (batch, 128) → (batch, 10)`), take `argmax` for the guesses, and print the accuracy of this untrained model. Run with `python3 sessions/m04-first-model-mlp/day-01-mlp-mnist/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5 Day 1 artifact.

Create sessions/m04-first-model-mlp/day-01-mlp-mnist/experiment.py that, with a comment on each step, builds an UNTRAINED 2-layer MLP for MNIST in NumPy (no training yet — forward pass only):
1. Loads MNIST, flattens each 28x28 image to shape (784,), scales pixels to [0,1].
2. Randomly initializes W1 (784,128), b1 (128,), W2 (128,10), b2 (10,) using He initialization (weights ~ N(0,1) * sqrt(2/n_in); biases zero).
3. Defines relu(z)=np.maximum(0,z) and the forward pass logits = relu(x @ W1 + b1) @ W2 + b2, printing the shape at each step (batch,784)->(batch,128)->(batch,10).
4. Takes argmax(logits) as the predicted digit and prints the accuracy of this untrained model — it should be roughly 10% (chance over 10 classes).
5. Also demonstrates the collapse: show (x@W1)@W2 == x@(W1@W2) for random small W1,W2, i.e. two linear layers with NO relu equal one.
Then run it and paste the untrained accuracy at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- The forward pass maps a batch of shape `(batch, 784)` to `(batch, 10)` — one row of 10 scores per image. Did you predict the output shape?
- The untrained model scores around **10%** — chance level over 10 digits. It has all the right *wiring* but random *knobs*. Was your accuracy guess high or low?
- Without a ReLU, `(x@W1)@W2` and `x@(W1@W2)` print the *same* numbers — proof that two linear layers really are one, and the whole reason the hidden layer needs its bend.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m04-first-model-mlp/day-01-mlp-mnist/log.md`: (1) the shape at each step of the forward pass; (2) why an untrained model gets ~10%; (3) the one reason you must put a bend between two layers.

@@@ fin
