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
@lede You know how you can spot a friend's messy handwriting on a birthday card — a wobbly "7" — in half a second, without even thinking? Your eyes just *know* it's a seven. Today you build a machine that learns to do the same thing. And the wild part: it's made of the exact same tiny building block you met in Module 2, just **stacked**. This is the same shape of machine behind the tools you've heard of — image apps, ChatGPT — only small enough to hold in your head.
@goal By the end you can describe a 2-layer MLP for handwritten digits (784 → 128 → 10), run one forward pass in your head, read off its guess, and say exactly why you must put a bend between the layers.

@@@ concept id=c1 tag="The task" title="70,000 handwritten digits" gotit="Got the task"
Let's start with the job — and it's a fun one. Picture a big shoebox holding **70,000 small photos**, each one a single handwritten digit — a wobbly 3, a rushed 7, a careful 0. Every photo is tiny and gray: a grid of **28 by 28 dots**, where each dot (a **pixel**) is just a brightness number from 0 (black) to 255 (white). This box is called [[MNIST||70,000 small grayscale images of handwritten digits 0–9, each 28×28 pixels — the classic "hello world" dataset of neural-network classification.]], and it is the "hello world" of neural networks — the very first thing people teach a machine to read. Your job: look at one photo and answer with the right digit, 0 through 9. Ten possible answers — we call each answer a **class**.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A shoebox of 28 by 28 pixel photos of a handwritten digit on the left, an arrow pointing to ten class labels 0 through 9 on the right"><g font-family="monospace" font-size="12"><rect x="30" y="20" width="90" height="90" rx="4" fill="#FDF9F3" stroke="#7C6DAA" stroke-width="2"/><path d="M50 40 L100 40 L70 100" fill="none" stroke="#3A342E" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"/><text x="75" y="125" fill="#6B645E" text-anchor="middle">28×28 pixels</text><text x="160" y="70" fill="#6B645E" text-anchor="middle">→ guess →</text><g text-anchor="middle"><rect x="230" y="52" width="260" height="34" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="360" y="74" fill="#5E5191" font-weight="bold">0  1  2  3  4  5  6  7  8  9</text><text x="360" y="108" fill="#6B645E" font-size="11">10 classes — one is the true digit</text></g></g></svg>
%%%

**What the shoebox picture gets right:** it captures the scale (a huge pile of separate examples) and the honest goal (name the digit). **Where it breaks down:** a real shoebox lets you flip through photos and see each one as a picture, but our model sees every image as a flat sheet of numbers — it has no idea "up" or "next to" mean anything. Hold that thought; it comes back at the very end.

#### Why we stack toward this
One neuron from Module 2 can barely tell a bright pixel from a dark one. Reading a *digit* — a whole shape — is far harder. So the plan for today is to **stack** simple pieces into something that can. Everything below is a step up that stack. Ready? Let's build.

@@@ concept id=c2 tag="Flatten it" title="From a grid to one long row" gotit="Got the flatten"
A neuron eats a *list* of numbers, not a *grid*. So before the model can look at an image, we lay the grid out flat. Picture unraveling a knitted scarf: the 28 rows of the grid get laid end to end into **one single row of 28 × 28 = 784 numbers**. This is called [[flattening||Reshaping a 2-D image grid into a 1-D vector by laying its rows end to end, so a dense layer can read it as one list of inputs.]] the image, and the long row it produces is the **input vector**.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A small pixel grid on the left, an arrow, and a long thin row of boxes on the right labeled 784 numbers"><g font-family="monospace" font-size="12"><g stroke="#7C6DAA" stroke-width="1.2" fill="#EDE9F8"><rect x="30" y="35" width="18" height="18"/><rect x="48" y="35" width="18" height="18"/><rect x="66" y="35" width="18" height="18"/><rect x="30" y="53" width="18" height="18"/><rect x="48" y="53" width="18" height="18"/><rect x="66" y="53" width="18" height="18"/><rect x="30" y="71" width="18" height="18"/><rect x="48" y="71" width="18" height="18"/><rect x="66" y="71" width="18" height="18"/></g><text x="57" y="105" fill="#6B645E" text-anchor="middle">28×28 grid</text><text x="130" y="66" fill="#6B645E" text-anchor="middle">→ unravel →</text><g stroke="#2D8B55" stroke-width="1.2" fill="#E8F5EE"><rect x="185" y="53" width="15" height="18"/><rect x="200" y="53" width="15" height="18"/><rect x="215" y="53" width="15" height="18"/><rect x="230" y="53" width="15" height="18"/><rect x="245" y="53" width="15" height="18"/><rect x="260" y="53" width="15" height="18"/><rect x="275" y="53" width="15" height="18"/><rect x="290" y="53" width="15" height="18"/><rect x="305" y="53" width="15" height="18"/></g><text x="330" y="66" fill="#6B645E">… </text><text x="300" y="42" fill="#1a5c38" text-anchor="middle" font-weight="bold">one row: 784 numbers</text><text x="300" y="90" fill="#6B645E" text-anchor="middle" font-size="11">shape (784,) — the input vector</text></g></svg>
%%%

**What the scarf picture gets right:** the grid is not thrown away — every dot is still there, just re-laid into a single line. **Where it breaks down:** unraveling a scarf ruins it, but here we keep every number; the cost is subtler — the flattened row forgets which pixels used to sit *next to* each other. Again, remember that; it is exactly the limit we hit at the end.

So one image becomes a vector of shape `(784,)`. That vector is what the very first layer of the network will read.

@@@ concept id=c3 tag="One line only" title="A single neuron draws one straight line" gotit="Got the limit"
Before we stack anything, let's meet the ancestor — and see why one piece is not enough. Imagine a **strict bouncer at a party door who is only allowed to use one straight rope** to split the crowd. He can angle the rope any way he likes, but it stays a single straight line — everyone on one side gets in, everyone on the other side is out. If the guests he must let in happen to sit neatly on one side, he does a great job. A single [[neuron||The building block from Module 2: it takes a weighted sum of its inputs, adds a bias, then applies an activation. On its own it draws one straight dividing line.]] (the building block from Module 2) is exactly this bouncer: it computes a weighted sum of its inputs plus a bias, which geometrically draws **one straight line** and calls one side "yes" and the other "no." This lone-neuron model has a historical name — the **single perceptron**, the 1950s ancestor of everything we build today.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A bouncer's single straight rope splitting a crowd of dots into an in group and an out group, with one dot on the wrong side"><g font-family="monospace" font-size="11"><rect x="20" y="15" width="480" height="100" rx="6" fill="#FDF9F3" stroke="#7C6DAA" stroke-width="1.5"/><line x1="120" y1="20" x2="360" y2="110" stroke="#C93B3B" stroke-width="3"/><text x="240" y="30" fill="#C93B3B" font-weight="bold" text-anchor="middle">one straight rope</text><g fill="#2D8B55"><circle cx="150" cy="35" r="6"/><circle cx="190" cy="45" r="6"/><circle cx="230" cy="40" r="6"/><circle cx="180" cy="65" r="6"/></g><text x="180" y="90" fill="#1a5c38" text-anchor="middle">"in" side</text><g fill="#7C6DAA"><circle cx="340" cy="70" r="6"/><circle cx="400" cy="60" r="6"/><circle cx="420" cy="90" r="6"/></g><text x="400" y="40" fill="#5E5191" text-anchor="middle">"out" side</text><circle cx="290" cy="95" r="6" fill="#C93B3B"/><text x="290" y="112" fill="#C93B3B" text-anchor="middle">stuck on the wrong side</text></g></svg>
%%%

**What the bouncer picture gets right:** one neuron, like one straight rope, makes exactly one clean straight cut. **Where it breaks down:** a real bouncer can walk around, check IDs, and change his mind case by case; the neuron cannot — the rope is *forced* to stay straight, so if the people he must let in are scattered in opposite corners, no single straight rope can ever separate them.

That corner-scattered case has a famous name: **XOR** — "on when exactly *one* of two inputs is on." Its two classes sit on opposite corners, so any straight line you draw leaves at least one point on the wrong side. Slide the line below every which way — you top out at 3 of 4. Then tick "add a hidden layer" and watch two lines, joined together, finally carve out the region one line never could. Go on — try to beat 3/4 with the single line first:

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

**What this shows:** a single perceptron is stuck at a **straight** boundary, so any curved or diagonal-corner pattern beats it. This is the [[capability limit||A single linear neuron can only separate data that one straight line splits — it cannot solve non-linearly-separable problems like XOR, or the curved boundaries between digit shapes.]] of a single linear neuron, and it is exactly why we cannot read curvy digit shapes with one neuron. The fix — the whole rest of this lesson — is to **stack** layers of neurons *with a bend between them*, so simple straight pieces combine into curved boundaries.

@@@ concept id=c4 tag="A layer" title="A whole layer in one multiply" gotit="Got the layer"
So we need more than one neuron. Think of a **panel of 128 judges at a talent show, all watching the same act at once**. Each judge listens to the same 784 details of the performance, weighs them their own way, and holds up one score. Put a **group of neurons side by side, all reading the same input**, and you have exactly that panel — we call it a [[layer||A group of neurons that all read the same input vector; its weights form a matrix W and its biases a vector b, computed together as z = x @ W + b.]].

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="One input row of 784 numbers fanning out to a vertical panel of 128 judges, each holding up one score"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="20" y="58" width="90" height="24" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="65" y="74" fill="#1a5c38">input · 784</text><g stroke="#C9C2D8" stroke-width="1"><line x1="112" y1="70" x2="230" y2="25"/><line x1="112" y1="70" x2="230" y2="55"/><line x1="112" y1="70" x2="230" y2="85"/><line x1="112" y1="70" x2="230" y2="115"/></g><g fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"><rect x="230" y="16" width="120" height="20" rx="10"/><rect x="230" y="46" width="120" height="20" rx="10"/><rect x="230" y="76" width="120" height="20" rx="10"/><rect x="230" y="106" width="120" height="20" rx="10"/></g><g fill="#5E5191"><text x="290" y="30">judge 1 → score</text><text x="290" y="60">judge 2 → score</text><text x="290" y="90">judge 3 → score</text><text x="290" y="120">…judge 128</text></g><text x="450" y="66" fill="#6B645E">128</text><text x="450" y="80" fill="#6B645E">scores</text></g></svg>
%%%

**What the judging-panel picture gets right:** every judge (neuron) reads the *same* input and each produces exactly one number, and you gather all 128 in one sweep — that "one sweep" is the whole trick. **Where it breaks down:** real judges talk to each other and change their minds; the neurons in one layer are independent and silent — each computes its score with no idea what its neighbors decided.

Here's the beautiful part: you don't ask each judge one at a time. You stack every judge's weights into one **weight matrix `W`** and every bias into one **bias vector `b`**, and collect all 128 scores in a single **matrix multiply** (written `@`). In one narrated line: multiply the input row by the weight matrix, then add the bias — `z = x @ W + b`.

Let's *watch* that one sweep build up, judge by judge, so the multiply is a picture, not algebra. The input row slides across the weight matrix; each column of `W` is one judge's opinion, and the running total fills in the output one slot at a time:

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Build-up: the input row multiplies the weight matrix column by column, each column producing one output slot, accumulating into a 128-long output row"><g font-family="monospace" font-size="10" text-anchor="middle"><text x="70" y="18" fill="#1a5c38" font-weight="bold">input row x (784 wide)</text><g fill="#E8F5EE" stroke="#2D8B55" stroke-width="1"><rect x="20" y="24" width="16" height="16"/><rect x="36" y="24" width="16" height="16"/><rect x="52" y="24" width="16" height="16"/><rect x="68" y="24" width="16" height="16"/><rect x="84" y="24" width="16" height="16"/></g><text x="115" y="36" fill="#6B645E">…</text><text x="300" y="18" fill="#5E5191" font-weight="bold">W: each column = one judge</text><g stroke="#7C6DAA" stroke-width="1"><rect x="200" y="24" width="22" height="120" fill="#EDE9F8"/><rect x="222" y="24" width="22" height="120" fill="#DDD3F0"/><rect x="244" y="24" width="22" height="120" fill="#EDE9F8"/><rect x="266" y="24" width="22" height="120" fill="#DDD3F0"/></g><text x="330" y="90" fill="#6B645E">… 128 columns</text><g><text x="245" y="164" fill="#C99A12" font-weight="bold">↓ each column · row → one number</text></g><text x="430" y="34" fill="#C93B3B" font-weight="bold">output z</text><g fill="#FDE8E8" stroke="#C93B3B" stroke-width="1"><rect x="400" y="44" width="26" height="16"/><rect x="400" y="60" width="26" height="16"/><rect x="400" y="76" width="26" height="16"/></g><text x="413" y="56" fill="#C93B3B">z₀</text><text x="413" y="72" fill="#C93B3B">z₁</text><text x="413" y="88" fill="#C93B3B">z₂</text><text x="440" y="112" fill="#6B645E">…filling</text><text x="440" y="126" fill="#6B645E">to 128</text></g></svg>
%%%

Column 0 of `W` times the row gives `z₀`; column 1 gives `z₁`; keep going and the output fills up to 128 numbers — one per judge, all from one sweep. Want to *see* the numbers move? Predict the output length, then run this tiny two-judge version:

%%% demo id=layer label="run one small layer"
code: x = np.array([1., 2., 3.]); W = np.array([[1.,0.],[0.,1.],[2.,1.]]); b = np.array([0.5, 0.5]); z = x @ W + b; z
out: array([7.5, 5.5])   # z0 = 1*1 + 2*0 + 3*2 + 0.5 = 7.5 ;  z1 = 1*0 + 2*1 + 3*1 + 0.5 = 5.5
take: <b>3 inputs, 2 judges → 2 scores in one multiply.</b> Each output number is one column of W dotted with the input, plus that judge's bias. Scale the same idea up: 784 in, 128 out.
%%%

!!! c-info 🧮
<b>Optional (skippable) — the shapes.</b> The one-line formula is `z = x @ W + b`. Shapes: `x` is `(1, 784)`, `W` is `(784, 128)`, `b` is `(128,)`, so `z` is `(1, 128)`. The middle number (784) lines up and cancels, leaving one output per neuron. Stacking the weights into a matrix isn't just tidy — it's why a whole layer costs one fast multiply instead of 128 slow loops.
!!!

@@@ concept id=c5 tag="Stack + bend" title="The MLP: stack layers, bend between them" gotit="Got the MLP"
Now we **stack**. Picture **folding a paper map**. Two straight folds, one after the other, still leave you with flat, straight creases — you cannot fold a map into a curved bowl no matter how many flat folds you add. But **crumple** the map once in the middle before the second fold, and suddenly the paper takes a shape no sequence of flat folds ever could. Stacking layers is the same story: two plain layers back to back are two flat folds, and putting a bend between them is the crumple. Stack layers this way and you have a [[multi-layer perceptron||An MLP: an input layer, one or more hidden layers, and an output layer, each layer fully connected to the next. The simplest real neural network.]] — an **MLP**. Each layer is **fully connected** (also called **dense**): every input feeds every neuron in the next layer.

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Folding a paper map: two flat folds stay flat, but a crumple in the middle unlocks a curved shape"><g font-family="monospace" font-size="11" text-anchor="middle"><text x="130" y="18" fill="#6B645E">two flat folds → still flat</text><path d="M30 40 L120 40 L210 40" fill="none" stroke="#7C6DAA" stroke-width="3"/><line x1="80" y1="34" x2="80" y2="46" stroke="#C93B3B" stroke-width="1.5"/><line x1="165" y1="34" x2="165" y2="46" stroke="#C93B3B" stroke-width="1.5"/><text x="120" y="66" fill="#C93B3B" font-size="10">no new shape</text><text x="390" y="18" fill="#6B645E">crumple in the middle → new shape</text><path d="M300 55 L340 30 L380 60 L420 32 L470 58" fill="none" stroke="#2D8B55" stroke-width="3"/><text x="385" y="80" fill="#1a5c38" font-size="10">the bend unlocks curves</text><line x1="255" y1="20" x2="255" y2="90" stroke="#DDD3F0" stroke-width="1.5"/></g></svg>
%%%

**What the map-folding picture gets right:** stacking plain (linear) layers is like stacking flat folds — you stay flat and straight, and the second layer buys nothing new; the crumple is the bend, and it unlocks shapes flat folds never could. **Where it breaks down:** a real crumple is random and messy; the network's bend is a precise, simple rule applied the same way every time, so the network can *learn* which bends help.

The middle layers — neither the raw input nor the final answer — are the [[hidden layers||The intermediate layer(s) between input and output. They let the network build its own internal features — combinations of pixels that hint at a stroke or a curve.]], where the net builds its own features ("a loop up top," "a straight stroke"). But here is the trap: stack two plain layers with **nothing** between them and they fold flat — two matmuls back to back are *secretly one bigger matmul*. Let's fold the two weight boxes together and see:

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="Top: W1 then W2 with no bend fold together into a single matrix, one layer. Bottom: W1, a ReLU bend, then W2 cannot be folded flat, giving real depth"><g font-family="monospace" font-size="12" text-anchor="middle"><text x="20" y="24" fill="#6B645E" font-size="10" text-anchor="start">NO bend:</text><rect x="30" y="30" width="48" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="54" y="47" fill="#5E5191">W₁</text><text x="90" y="47" fill="#6B645E" font-size="10">fold</text><rect x="120" y="30" width="48" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="144" y="47" fill="#5E5191">W₂</text><text x="188" y="47" fill="#6B645E">→</text><rect x="210" y="26" width="120" height="34" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="270" y="47" fill="#C93B3B" font-weight="bold">one matrix W₁@W₂</text><text x="415" y="47" fill="#C93B3B" font-size="10">= just 1 layer 😞</text><text x="20" y="104" fill="#6B645E" font-size="10" text-anchor="start">WITH bend:</text><rect x="30" y="110" width="48" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="54" y="127" fill="#5E5191">W₁</text><rect x="86" y="110" width="56" height="26" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="114" y="127" fill="#1a5c38" font-weight="bold">ReLU</text><rect x="150" y="110" width="48" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="174" y="127" fill="#5E5191">W₂</text><text x="212" y="127" fill="#2D8B55">→</text><text x="240" y="127" fill="#2D8B55" text-anchor="start" font-size="11">can't fold flat → real depth 🎉</text></g></svg>
%%%

The cure is to insert a [[nonlinear activation||A simple function applied after a layer's matmul+bias that is not a straight line — ReLU is the default. It keeps stacked layers from collapsing into one linear map.]] — the **bend** — between the layers. The default bend is **ReLU**: `ReLU(z) = max(0, z)`, "keep positives, zero out negatives." With the bend in place, negatives get clipped *before* the second matmul, producing values no single matmul could — that's why the hidden layer earns its keep. And recall from c3: a single straight-line map can't even do XOR, so a flat-folded stack is stuck at that same ceiling.

!!! c-info 🪜
<b>Optional (skippable) — Math Ladder, why linear ∘ linear collapses.</b>
<br><b>1 · In words:</b> two plain matmuls with no bend between them do the same job as one matmul, so the second layer bought you nothing.
<br><b>2 · Scalars first:</b> chain `f(x)=a·x+p` then `g(u)=b·u+q`. Compose: `g(f(x)) = b·(a·x+p)+q = (ab)·x + (bp+q)` — still `slope·x + shift`, one linear step.
<br><b>3 · Matrix version:</b> `(x@W1)@W2 = x@(W1@W2)`, and `W1@W2` is just one matrix. The two linear layers really are one.
<br><b>4 · Sanity check:</b> clip with a bend first — `ReLU(x@W1)@W2` — and negatives get zeroed <em>before</em> the second matmul, producing values no single matmul can. That clip is the crumple.
!!!

@@@ concept id=c6 tag="Forward pass" title="Pixels in, ten scores out" gotit="Got the forward pass"
Now let's run the whole thing left to right. Picture a **factory assembly line**. A raw photo enters at one end. Station 1 studies it and stamps 128 rough notes. A bend-station throws away the negative notes. Station 2 reads those notes and prints a final scorecard with 10 boxes — one per digit — and that scorecard drops off the end of the belt. That end-to-end trip is called the [[forward pass||The left-to-right flow that turns an input into an output: input → hidden layer (matmul + bias + activation) → output layer, producing the final scores.]]. The last station is the **output layer** — exactly **10** neurons, one per digit — and its 10 numbers are called [[logits||The raw, un-normalized scores the output layer produces — one per class. A bigger logit means the model leans toward that class. Turning them into probabilities is softmax, on Day 4.]]: raw scores, one per possible digit.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="An assembly line: a raw image enters, station one plus a bend produces 128 notes, station two prints a 10-box scorecard that drops off the belt"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="10" y="110" width="500" height="14" rx="7" fill="#DDD3F0"/><g fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"><rect x="15" y="55" width="60" height="40" rx="4"/></g><text x="45" y="78" fill="#1a5c38">raw</text><text x="45" y="92" fill="#6B645E" font-size="9">image</text><text x="110" y="50" fill="#5E5191" font-size="9">station 1</text><rect x="95" y="55" width="60" height="40" rx="4" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="125" y="80" fill="#5E5191" font-size="9">W₁+b₁</text><text x="185" y="72" fill="#2D8B55" font-size="9">bend</text><rect x="170" y="60" width="30" height="30" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="185" y="80" fill="#1a5c38" font-size="8">ReLU</text><rect x="215" y="55" width="70" height="40" rx="4" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="250" y="74" fill="#5E5191" font-size="9">128 notes</text><text x="320" y="50" fill="#5E5191" font-size="9">station 2</text><rect x="300" y="55" width="60" height="40" rx="4" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="330" y="80" fill="#5E5191" font-size="9">W₂+b₂</text><rect x="380" y="55" width="70" height="40" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="415" y="74" fill="#C93B3B" font-weight="bold" font-size="9">10-box card</text><text x="485" y="78" fill="#6B645E" font-size="9">→</text><text x="260" y="140" fill="#6B645E" font-size="10">raw photo in → 10 scores off the end of the belt</text></g></svg>
%%%

**What the assembly-line picture gets right:** the photo enters raw, each station transforms it in order, and a finished, labeled scorecard comes out the far end — you never skip a station. **Where it breaks down:** on a real belt one item moves down through the stations; here all the stations fire at once on the whole batch of images in a single matrix multiply — there's no single item crawling along.

The whole line fits on one line of code: `logits = ReLU(x @ W1 + b1) @ W2 + b2`. Instead of decoding that nested formula, just *watch the shape change* at each station — this is the build-up, drawn:

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Build-up: the shape (1,784) shrinks to (1,128) after layer 1, ReLU zeros the negative bars, then it expands to (1,10) logits"><g font-family="monospace" font-size="11" text-anchor="middle"><text x="60" y="20" fill="#1a5c38" font-weight="bold">1 · in</text><g fill="#E8F5EE" stroke="#2D8B55" stroke-width="1"><rect x="20" y="40" width="80" height="60"/></g><text x="60" y="75" fill="#1a5c38">(1,784)</text><text x="125" y="72" fill="#6B645E">→</text><text x="185" y="20" fill="#5E5191" font-weight="bold">2 · hidden</text><g fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1"><rect x="160" y="55" width="52" height="45"/></g><text x="186" y="82" fill="#5E5191">(1,128)</text><text x="235" y="72" fill="#2D8B55">→</text><text x="300" y="20" fill="#2D8B55" font-weight="bold">3 · ReLU zeros ⊖</text><g font-size="9"><rect x="270" y="55" width="10" height="20" fill="#2D8B55"/><rect x="282" y="65" width="10" height="10" fill="#C9C2D8"/><rect x="294" y="55" width="10" height="20" fill="#2D8B55"/><rect x="306" y="65" width="10" height="10" fill="#C9C2D8"/><rect x="318" y="58" width="10" height="17" fill="#2D8B55"/></g><text x="295" y="92" fill="#6B645E" font-size="9">negatives → 0</text><text x="345" y="72" fill="#C93B3B">→</text><text x="440" y="20" fill="#C93B3B" font-weight="bold">4 · logits</text><g fill="#FDE8E8" stroke="#C93B3B" stroke-width="1"><rect x="410" y="52" width="60" height="48"/></g><text x="440" y="80" fill="#C93B3B">(1,10)</text><text x="260" y="135" fill="#6B645E" font-size="10">shapes flow: 784 → 128 (bend zeros negatives) → 10 scores</text></g></svg>
%%%

**Notice the shapes flow:** 784 shrinks to 128 at station 1, the bend zeros out the negative notes, then it expands to 10 scores at station 2 — one per digit. Each `@` cancels its matching middle number, and the last one lands on 10 because there are 10 digits to score. That left-to-right trip, turning pixels into 10 scores, *is* the forward pass — the same shape of computation as a giant language model, only smaller.

!!! c-info 🧮
<b>Optional (skippable) — reading the nested formula.</b> `logits = ReLU(x @ W1 + b1) @ W2 + b2` is written the way math nests: the *innermost* part happens *first*. Read it inside-out, starting at `x`: (1) input `x` is `(1,784)`; (2) `h = ReLU(x @ W1 + b1)` with `W1 (784,128)` gives `h (1,128)`; (3) `logits = h @ W2 + b2` with `W2 (128,10)` gives `logits (1,10)`. "Inside-out" and "left-to-right" describe the *same* single trip — nothing goes backward.
!!!

@@@ concept id=c7 tag="Read + count" title="Reading the guess, and counting the knobs" gotit="Got the readout"
The forward pass ends with 10 scores. How do we turn that into an answer? Picture standing in a room where **ten people are each shouting a number** — one for each digit. You don't average them or ask who spoke first; you just point at the **loudest** voice. The position of the loudest voice is the model's guess. That "which one is loudest?" pick is called [[argmax||Short for 'arg-max': it returns the *position* of the largest value, not the value itself. Here it turns 10 logits into a single guessed digit 0–9.]]. If slot 7 is loudest, the model guesses "7."

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Ten logit bars for digits 0 through 9, with the bar for digit 7 tallest and highlighted, and an arrow reading argmax equals 7"><g font-family="monospace" font-size="11" text-anchor="middle"><g fill="#C9C2D8"><rect x="40" y="60" width="20" height="20"/><rect x="70" y="55" width="20" height="25"/><rect x="100" y="65" width="20" height="15"/><rect x="130" y="50" width="20" height="30"/><rect x="160" y="62" width="20" height="18"/><rect x="190" y="58" width="20" height="22"/><rect x="220" y="66" width="20" height="14"/></g><rect x="250" y="24" width="20" height="56" fill="#C93B3B"/><g fill="#C9C2D8"><rect x="280" y="57" width="20" height="23"/><rect x="310" y="63" width="20" height="17"/></g><g fill="#6B645E"><text x="50" y="94">0</text><text x="80" y="94">1</text><text x="110" y="94">2</text><text x="140" y="94">3</text><text x="170" y="94">4</text><text x="200" y="94">5</text><text x="230" y="94">6</text><text x="260" y="94" fill="#C93B3B" font-weight="bold">7</text><text x="290" y="94">8</text><text x="320" y="94">9</text></g><text x="430" y="45" fill="#C93B3B" font-weight="bold">argmax = 7</text><text x="430" y="62" fill="#6B645E" font-size="10">loudest voice wins</text></g></svg>
%%%

**What the shouting-room picture gets right:** argmax cares only about *which* voice is loudest, not by how much. **Where it breaks down:** real shouting has a volume you can compare; a logit's exact height only becomes a real "how sure" number after softmax turns it into a probability — that's Day 4. For today, the *guess* is just argmax.

#### The knobs the network will learn
Every number inside `W1, b1, W2, b2` is a **parameter** — a tunable knob. Picture a **studio mixing board covered in tuning knobs**: turning each one changes the sound a little, and there are a *lot* of them. How many? Let's stack them up into a running total and watch it climb:

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A running total bar accumulating parameter counts: W1 784x128 about 100352, plus b1 128, plus W2 128x10 1280, plus b2 10, reaching about 101770 total knobs"><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#6B645E">stacking up the knobs (each bar adds to the running total)</text><rect x="60" y="30" width="300" height="18" rx="3" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="210" y="43" fill="#5E5191">+ W₁ = 784 × 128 ≈ 100,352</text><rect x="60" y="54" width="304" height="14" rx="3" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.5"/><text x="212" y="65" fill="#8A6D3B" font-size="9">+ b₁ = 128</text><rect x="60" y="74" width="316" height="14" rx="3" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="218" y="85" fill="#5E5191" font-size="9">+ W₂ = 128 × 10 = 1,280</text><rect x="60" y="94" width="318" height="14" rx="3" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.5"/><text x="219" y="105" fill="#8A6D3B" font-size="9">+ b₂ = 10</text><rect x="60" y="116" width="320" height="24" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="220" y="132" fill="#C93B3B" font-weight="bold">running total ≈ 101,770 knobs</text><text x="450" y="130" fill="#C93B3B" font-size="11" font-weight="bold">~100k</text></g></svg>
%%%

So this small model already has about **100,000 knobs**. **What the mixing-board picture gets right:** each knob nudges the output a little, and training is the slow process of finding a good setting for all of them at once. **Where it breaks down:** a sound engineer turns knobs by ear, one at a time; the network will turn *all* 100,000 at once using math (Day 2). Intuition to keep: a **wider** hidden layer (256 or 1024 instead of 128) means *more* knobs — more capacity to represent shapes, but also more to tune.

#### Right now, every knob is random
We haven't trained anything yet, so every knob starts on a small **random** setting. A fresh, untrained network is guessing blindly — over 10 digits, that's about **10% accuracy**, no better than a ten-sided coin. That's the correct, expected starting point. Learning (Day 2 onward) is turning those random knobs into good ones.

@@@ concept id=c8 tag="When it breaks" title="Three quiet ways it goes wrong" gotit="Got the failure modes"
Here's a fun puzzle to end on: your model can be wired *perfectly* and still quietly fail — no crash, just bad guesses. Think of a **car that turns on but won't drive**. The engine is wired right, the key works, yet the car stalls — and there are a few classic reasons why. Same with an MLP: three quiet stalls, each with a clear cause and a one-move fix. **Where the car picture breaks down:** a stalled car makes a noise or a warning light; these model stalls are *silent* — the code runs fine and only your accuracy looks off.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="Three failure modes shown as a symptom, cause, and fix table: linear collapse fixed by ReLU, underfitting fixed by widening, dead ReLU fixed by He init or Leaky ReLU"><g font-family="monospace" font-size="11"><rect x="20" y="20" width="480" height="38" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="30" y="36" fill="#C93B3B" font-weight="bold">1 · Linear collapse</text><text x="30" y="51" fill="#6B645E">no bend → stacked layers = one layer  ·  FIX: insert ReLU between layers</text><rect x="20" y="64" width="480" height="38" rx="5" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.5"/><text x="30" y="80" fill="#8A6D3B" font-weight="bold">2 · Underfitting</text><text x="30" y="95" fill="#6B645E">too few hidden units → can't separate classes  ·  FIX: widen the hidden layer</text><rect x="20" y="108" width="480" height="38" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="30" y="124" fill="#5E5191" font-weight="bold">3 · Dead ReLU</text><text x="30" y="139" fill="#6B645E">unit stuck outputting 0 forever  ·  FIX: He initialization (or Leaky ReLU / GELU)</text></g></svg>
%%%

The figure above is the whole story in one glance — three stalls, three one-move fixes:
- **Linear collapse** — you forgot the bend (the crumple from c5), so the deep-looking model learns no better than one layer. *Fix:* insert a ReLU between the layers.
- **Underfitting** — the hidden layer is too small to tell a 4 from a 9. *Fix:* widen it — more knobs, more capacity.
- **Dead ReLU** — some units get shoved permanently negative, so `max(0,z)` is always 0 and they stop learning. *Fix:* [[He initialization||A weight start built for ReLU: it scales the random spread by the number of inputs (about 2/n_in) so units are far less likely to be born stuck at zero. The variance math is a later deep-dive.]], a ReLU-friendly random start; if units still die, swap in **Leaky ReLU / GELU**, which keep a tiny negative slope so a stuck unit can climb back to life.

!!! c-warn ⚠️
<b>The senior's trick (one line):</b> none of these throws an error — you catch them by <em>logging activation statistics</em> (how many ReLU units are stuck at 0, the train-vs-test gap), not by staring at final accuracy. There's also a fourth stall — <b>overfitting</b>, where the model memorizes the practice digits and flunks new ones; a train/test split reveals it, and full regularization is a later day.
!!!

@@@ concept id=c9 tag="The ceiling" title="Why this model can't be the last word" gotit="Got the ceiling"
End with a little honesty about what you built — it's real and it works, but it has a ceiling, and seeing the ceiling is what makes the *next* model exciting. Remember flattening from c2? Picture reading a book by **cutting every line of text out and taping them all into one long paper strip**. You can still read every word — but you've thrown away the *page layout*. Now nudge the whole poem down by one line before you cut: the strip you tape together is completely different, even though the poem is the same. That's exactly what flattening does to a digit.

Let's build the problem in three steps so you *see* it happen:

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Three steps: a 3 on a grid, the grid flattened into a row, then the same 3 shifted down one row producing a completely different flattened row"><g font-family="monospace" font-size="10" text-anchor="middle"><text x="70" y="18" fill="#6B645E">1 · a "3" on the grid</text><rect x="35" y="26" width="70" height="70" rx="4" fill="#FDF9F3" stroke="#7C6DAA" stroke-width="1.5"/><path d="M50 38 Q75 36 67 51 Q85 56 60 68" fill="none" stroke="#3A342E" stroke-width="4"/><text x="70" y="112" fill="#6B645E" font-size="9">top rows lit</text><text x="255" y="18" fill="#6B645E">2 · flatten to one row</text><g stroke="#2D8B55" stroke-width="0.8"><rect x="180" y="46" width="12" height="14" fill="#2D8B55"/><rect x="192" y="46" width="12" height="14" fill="#2D8B55"/><rect x="204" y="46" width="12" height="14" fill="#E8F5EE"/><rect x="216" y="46" width="12" height="14" fill="#E8F5EE"/><rect x="228" y="46" width="12" height="14" fill="#E8F5EE"/><rect x="240" y="46" width="12" height="14" fill="#E8F5EE"/><rect x="252" y="46" width="12" height="14" fill="#E8F5EE"/><rect x="264" y="46" width="12" height="14" fill="#E8F5EE"/></g><text x="290" y="57" fill="#1a5c38" font-size="9">…lit near the start</text><text x="255" y="96" fill="#C93B3B">3 · shift the "3" down one row</text><g stroke="#C93B3B" stroke-width="0.8"><rect x="180" y="120" width="12" height="14" fill="#FDE8E8"/><rect x="192" y="120" width="12" height="14" fill="#FDE8E8"/><rect x="204" y="120" width="12" height="14" fill="#FDE8E8"/><rect x="216" y="120" width="12" height="14" fill="#C93B3B"/><rect x="228" y="120" width="12" height="14" fill="#C93B3B"/><rect x="240" y="120" width="12" height="14" fill="#FDE8E8"/><rect x="252" y="120" width="12" height="14" fill="#FDE8E8"/><rect x="264" y="120" width="12" height="14" fill="#FDE8E8"/></g><text x="300" y="131" fill="#C93B3B" font-size="9">…lit slots ALL moved</text><text x="260" y="160" fill="#6B645E" font-size="10">same digit, totally different row → MLP thinks it's a new pattern</text></g></svg>
%%%

Walk the three steps: (1) the "3" lights up the top of the grid; (2) flattening lays those lit dots near the *start* of the 784-row; (3) shift the same "3" down one row, and every lit dot lands in a *different* slot of the row. The model never learned "a shift doesn't change the digit," so to it the shifted 3 is a brand-new pattern. **What the book-strip picture gets right:** you keep every symbol but lose the layout, so moving the content shifts the whole strip. **Where it breaks down:** a person re-reading the taped strip can still tell it's the same poem; the MLP genuinely cannot — it has no built-in notion of "nearby" or "moved."

**The limit:** because it flattens the image, the MLP ignores **2-D spatial structure** and treats a shifted digit as new. That's the [[capability limit||This architecture flattens away the 2-D layout, so it cannot exploit spatial structure or the fact that a shifted digit is the same digit — a convolutional network, taught in a later module, is built to fix exactly this.]] of this design — and it's exactly what a **convolutional network** (a later module) is built to fix, by keeping the grid and scanning it with small filters. You haven't built the best possible digit reader. You've built the *first* one — and understood every knob in it. That's the whole point of today.

@@@ concept id=c10 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — look how far you climbed. You started with a shoebox of 70,000 tiny photos and a bouncer's single straight rope, and you ended holding a full network that turns raw pixels into a digit guess. Here is the whole day as one staircase — each rung is a thing you now understand, and they **stack** on top of each other exactly the way the network does.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A staircase of the day's ideas, bottom to top: flatten the image, one neuron draws one line, a layer is one matmul, stack layers with a ReLU bend, forward pass to ten logits, argmax reads the guess"><g font-family="monospace" font-size="11"><rect x="20" y="164" width="150" height="26" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="30" y="181" fill="#1a5c38">1 · flatten → (784,)</text><rect x="60" y="134" width="180" height="26" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="70" y="151" fill="#C93B3B">2 · one neuron = one line</text><rect x="100" y="104" width="200" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="110" y="121" fill="#5E5191">3 · a layer = one matmul z=x@W+b</text><rect x="140" y="74" width="230" height="26" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="150" y="91" fill="#1a5c38">4 · stack layers + ReLU bend = MLP</text><rect x="180" y="44" width="230" height="26" rx="4" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.5"/><text x="190" y="61" fill="#8A6D3B">5 · forward pass → 10 logits</text><rect x="220" y="14" width="200" height="26" rx="4" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="230" y="31" fill="#C93B3B">6 · argmax = the guess</text></g></svg>
%%%

#### The arc in six beats — each with the picture that anchored it
1. **The task (the shoebox):** classify a 28×28 handwritten digit into one of 10 classes — MNIST, our shoebox of 70,000 photos.
2. **Flatten (the scarf):** unravel the grid into one row of `(784,)` numbers so a neuron can read it — but the row forgets which pixels were neighbors.
3. **One line is not enough (the bouncer's rope):** a single neuron draws one straight boundary — it can't even do XOR, so we must **stack**.
4. **A layer + a bend (the judging panel + the folding map):** a layer is one matmul `z = x @ W + b` — 128 judges scoring at once; a stack of plain layers folds flat, so we **stack** them with a ReLU crumple between — that's the MLP.
5. **Forward pass (the assembly line):** `logits = ReLU(x @ W1 + b1) @ W2 + b2`, shapes flowing 784 → 128 → 10 down the belt.
6. **Read + limits (the shouting room + the mixing board + the car + the book-strip):** `argmax` points at the loudest of 10 voices; the ~100,000 knobs on the mixing board start random, so untrained ≈ 10%; watch for the three quiet car-stalls (collapse / underfitting / dead-ReLU); and remember the book-strip — flattening throws away 2-D structure, which CNNs fix later.

#### Cheat-sheet — every new word from today, in plain English
%%% jargon
MNIST | 70,000 tiny 28×28 grayscale photos of handwritten digits 0–9 — the classic first dataset
flatten | lay a 28×28 grid end-to-end into one row of 784 numbers (the input vector)
neuron | weighted sum of inputs + a bias; on its own it draws one straight line
layer | a group of neurons reading the same input; computed as one matmul z = x @ W + b
MLP | multi-layer perceptron — input layer, hidden layer(s), output layer, stacked and dense
hidden layer | the in-between layer where the net builds its own features
ReLU | the bend: max(0, z) — keep positives, zero negatives — stops the stack from collapsing
forward pass | the left-to-right trip: pixels → hidden → logits
logits | the 10 raw output scores, one per digit (softmax turns them into probabilities — Day 4)
argmax | returns the position of the biggest score → the guessed digit
parameter | one tunable knob (a weight or a bias); this model has ~100,000
He initialization | ReLU-friendly random start, spread scaled by ~2/n_in, so units aren't born dead
%%%

#### The three quiet stalls at a glance
%%% table
:: Failure :: Cause :: Fix
Linear collapse :: no bend between layers :: insert ReLU
Underfitting :: hidden layer too small :: widen the hidden layer
Dead ReLU :: unit stuck permanently negative :: He init (or Leaky ReLU / GELU)
%%%

Keep this page. Everything from Day 2 onward — the backward pass, gradient descent, loss — is about turning those ~100,000 **random** knobs into good ones, one nudge at a time.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions with instant feedback. If one trips you up, that's a cue to scroll back — not a failing grade.
%%% quiz
q: An MNIST image is a 28×28 grid. What shape is it after flattening for the MLP? | a:1 | (28, 28) | (784,) | (10,) | (128,) | fb: Flattening lays the 28×28 = 784 pixels into one long row of shape (784,) — the input vector the first layer reads.
q: Why must you put a nonlinear activation (like ReLU) between the two layers? | a:2 | It makes the matmul faster | It saves memory | Without it, the two stacked layers collapse into a single linear map | It converts logits to probabilities | fb: (x@W1)@W2 = x@(W1@W2) — no bend means the stack is secretly one linear layer, which can't even do XOR. ReLU is the crumple that gives real depth.
q: The output layer produces 10 logits. How does the model pick its guessed digit? | a:1 | The average of the 10 scores | argmax — the position of the largest logit | The first positive score | Whichever is closest to 0.5 | fb: The guess is argmax(logits): the position of the loudest voice. Slot 7 highest → guess "7." (Turning scores into probabilities is softmax, on Day 4.)
q: Some hidden units output 0 for every input and never recover — a "dead ReLU." Which named fix targets this at the start? | a:2 | Widen the output layer | Convert logits to probabilities | He initialization (variance-preserving weights for ReLU) | Flatten the image differently | fb: Dead ReLUs often come from a bad random start. He initialization scales the weight spread by ~2/n_in so units are far less likely to be born permanently negative; Leaky ReLU / GELU are the fallback.
%%%

@@@ produce id=produce tag="Produce" title="Build the untrained model and read its guess" gotit="Done"
Time to see today's model with your own eyes — no training yet, just the forward pass you now understand. **Before you write any code, predict two things:** (1) what shape will the output be for one image, and (2) roughly what accuracy will this *untrained*, random-weight model get on the test digits — 10%, 50%, or 95%? Write your two guesses down, then run it and **watch** whether you were right. Pick one path.

#### Option A · write it yourself
Create `sessions/m04-first-model-mlp/day-01-mlp-mnist/experiment.py`. Load MNIST, flatten each image to shape `(784,)`, and scale pixels to `[0,1]`. Initialize random weights `W1 (784,128), b1 (128,), W2 (128,10), b2 (10,)` using **He initialization** (scale each weight by roughly `sqrt(2/n_in)`). Write the forward pass `logits = relu(x @ W1 + b1) @ W2 + b2`, print the shape at every step (expect `(batch, 784) → (batch, 128) → (batch, 10)`), take `argmax` for the guesses, and print the accuracy of this untrained model. Run with `python3 sessions/m04-first-model-mlp/day-01-mlp-mnist/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 5 Day 1 artifact.

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
