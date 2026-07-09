---
quest_id: wf2-d01-neuron
donor: m02-day-01.donor
mode: exemplar
notebook_yardstick: 00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb
spine: "brain cell: weigh → add → decide"
module_label: "Module 2 · Train · Day 1"
title: "A Single Neuron"
subtitle: "The Smallest Piece of a Learning Machine"
brand_sub: "Foundations · M2 Day 1"
page_title: "Module 2 · Day 1 — A Single Neuron"
nav_prev_href: "../../m01-shape-of-data/review.html"
nav_prev_label: "M1 · Review Gate"
nav_next_href: "../day-02-activations/lesson.html"
nav_next_label: "Activation Functions"
fin_title: "Module 2 · Day 1 complete! 🏆"
fin_body: "Nice work — you've completed <b>A Single Neuron</b>.<br>Next up: <b>Activation Functions</b>."
sidebar:
  - {target: home, label: "Start here"}
  - {target: s1, label: "1 · What is it"}
  - {target: s2, label: "2 · Intuition"}
  - {target: s3, label: "3 · Playground"}
  - {target: s4, label: "4 · Mechanism &amp; why"}
  - {target: s5, label: "5 · Build it up"}
  - {target: s6, label: "6 · Quiz"}
  - {target: s7, label: "7 · Produce"}
anchors:
  x: "[2, 3]"
  w: "[0.5, −1]"
  b: "1"
---

@@@ hero
@lede
Right now, without even trying, your brain is soaking up thousands of tiny signals — the shapes on this screen, the sounds around you — and quietly deciding which ones matter and which to ignore. That one habit, *weigh things, then decide*, is the whole idea behind a **neural network**. Its smallest piece is called a **neuron**, and a neuron does just one very human job: it looks at some numbers, decides how much each one matters, adds them up, and gives a verdict. Today you'll build one by hand — and once this little piece clicks, everything bigger turns out to be just *many of these*.
@goal
By the time you close this tab, one neuron will feel obvious: you'll walk through its three steps — **weigh the inputs → add a bias → decide with an activation** — and compute one by hand.

@@@ section id=s1 num=1 numclass=s-what data_sec=what tag="What is it" title="What a neuron is (and what a network is made of)" gotit="Got what a neuron is — let's go"
!!! c-info 🧭
**What this assumes:** Module 1 — arrays, the dot product / matmul (Day 4), and broadcasting. **No neural-network background needed** — this is the first piece.
!!!

#### First, the picture — hold this before any words
Picture one brain cell for a moment. It is listening to lots of little signals coming in at once, but it does not treat them all the same — some incoming connections are strong and some are weak, so some signals count for more. The cell **adds everything up**, and if the total is strong enough it **fires**; if not, it stays quiet. That is the entire idea of a neuron: **weigh each input by how much it matters, add them up, then decide.** Hold on to that one picture. Every word in the rest of this section is just a name for one part of that single sentence — not a new thing to be scared of.

#### The words you'll meet today — plain English first
Before any formula, here is every term you'll run into today, said in plain words. You don't need to memorize these — just glance down the list so no word feels new when it shows up.

%%% jargon
neuron | The smallest piece of a neural network. It does three tiny steps: weigh the inputs, add a bias, then decide with an activation.
weight | A number that says how much one input matters to this neuron. Training is the process of adjusting these.
bias | One extra number added after the weighted sum. It nudges the result up or down before the decision.
weighted sum | Multiply each input by its weight and add them all up. This is exactly a **dot product** from Module 1.
activation | A simple decision step at the very end (like ReLU = keep positives, zero out negatives) that turns the total into the output.
neural network | Many neurons wired together in layers, tuned by training to turn inputs into useful answers.
%%%

#### What is a neural network?
A [[neural network||A machine that learns patterns from examples by adjusting a big set of numbers called weights. Built from many small units (neurons) arranged in layers.]] is a machine that learns patterns from examples by tuning a big pile of numbers called **weights**. It is built from many tiny repeated units — **neurons** — stacked into layers. Today we zoom all the way in to one neuron.

#### What is a neuron?
A [[neuron||The basic unit of a neural network. It does three steps: weighted sum of inputs, add a bias, then apply an activation function.]] is a tiny function with exactly three steps:
1. **Weighted sum** — multiply each input by its [[weight||A number that says how much one input matters to this neuron. Weights are what training adjusts.]] and add them up. (This is exactly a **dot product** from M1 Day 4.)
2. **Add a bias** — add one extra number, the [[bias||A single number added after the weighted sum. It shifts the result up or down before the activation.]], that shifts the result.
3. **Activation** — pass that result through a simple function (an [[activation function||A simple function applied at the end of a neuron, e.g. ReLU = max(0, x). It adds the non-linearity that lets networks learn complex patterns.]]) to get the output.

#### How does it relate to what you already know?
Step 1 is something you can already do: `np.dot(w, x)` — the weighted sum *is* a dot product. So a neuron is just `activation(dot(w, x) + b)`. Why the activation at the end? That's the secret sauce — next lesson. Keep going 👇

!!! c-warn 😕
**为什么这里容易卡住 · Why this trips people up:** 很多人一上来就盯着 `w·x+b` 这个公式，却没问它到底在做什么 — people meet `w·x+b` and treat it as a magic formula to memorize. In plain terms: a neuron only does one human thing — "score each input by how much it matters, then decide." The formula is just the arithmetic of that one sentence, not a new idea.
!!!

@@@ section id=s2 num=2 numclass=s-study data_sec=intuition tag="Intuition" title="From a brain cell to a math function" gotit="Got the picture"
#### Now make that picture exact
You already have the picture from Section 1 — a cell that weighs its inputs, adds them up, and decides. Because the word "neuron" is borrowed from the brain, let's line the two up piece by piece so the picture becomes something you can compute. A real **brain neuron** collects tiny electrical signals from many neighbors through branches called [[dendrites||The branches of a brain neuron that receive incoming signals from other neurons.]]. Some of those connections are strong and some are weak, so some signals count for more than others. The cell body **adds all the signals up**, and if the total is strong enough to cross a **firing threshold**, the neuron "fires" — sending its own signal out along a long wire called the [[axon||The output wire of a brain neuron that carries its signal onward to other neurons.]]. If the total is too weak, it stays quiet.

An **artificial neuron** keeps that exact shape — take in many signals, weigh them, add them up, then decide one output — but turns each piece into arithmetic you can compute by hand. Here is the piece-by-piece mapping:

%%% table
Brain neuron :: Artificial neuron :: What it is
Dendrites taking in signals :: the inputs `x` :: the numbers coming in
Synapse strength (some connections matter more) :: the weights `w` :: how much each input counts
Cell body summing the signals :: the weighted sum `w·x` :: add up the weighted inputs
Firing threshold (fire only if strong enough) :: the bias `b` + activation :: shift the total, then decide the output
Axon carrying the signal onward :: the neuron's output :: the one number passed to the next layer
%%%

Read straight down the middle column and you get the whole formula, in order: take the inputs `x`, weigh them with `w`, sum to `w·x`, shift by the bias `b`, then let the activation decide the output — exactly `output = activation(w·x + b)`, which Section 4 works out with real numbers.

#### A second, everyday picture (optional)
If the brain feels too abstract, here is the same three steps as one judge scoring a contestant:

%%% cards
⚖️ | Inputs × weights = a judge's score | Each input (talent, stage presence…) gets a **weight** — how much this judge cares about it. The judge multiplies and adds them into one score.
🎚️ | Bias + activation = the verdict | The judge has a personal **bias** (a grumpy judge starts lower), then squashes the total into a clean verdict with an **activation**.
%%%

**In one line:** a neuron weighs its inputs, adds a bias, and squashes the result into an output — the same three steps whether you picture a brain cell or a judge.

!!! c-warn ⚠️
Where the brain analogy breaks: it is an *inspiration*, not a copy. A real neuron fires as electrical spikes spread over time and slowly rewires itself through chemistry; an artificial neuron is a single clean equation whose **weights are learned by gradient descent** (Day 5), not by biology — there is no biological "backprop" in your head. And one artificial neuron alone is weak: the power comes from **many** neurons in **layers**, which is just a matmul, as you'll see at the end.
!!!

**直觉 / Intuition:** "neuron" 这个词来自大脑 — 一个脑神经元把很多输入信号按强弱加起来，够强就"点火"输出，否则保持安静。人工神经元照搬了这个结构：输入 `x`、权重 `w`（每个输入有多重要）、加权求和 `w·x`、再用偏置 `b` 加激活函数决定输出。The technical point: one neuron maps many input numbers down to a single output number, and a **weight** is literally "how much this input matters to me."

@@@ section id=s4 num=4 numclass=s-study data_sec=why tag="Mechanism &amp; why" title="The formula — and why one neuron scales to billions" gotit="Got the formula"
#### The formula

%%% formula
expr: output = activation( w · x + b )
note: `w · x` = weighted sum (a dot product), `b` = bias (one number), `activation` = a simple squashing function like ReLU. The value `w · x + b`, *before* the activation, is called the **pre-activation** (often written `z`).
%%%

Worked small: inputs `x = [2, 3]`, weights `w = [0.5, −1]`, bias `b = 1`. Weighted sum = `2·0.5 + 3·(−1) = −2`. Add bias → `−1`. ReLU(−1) = `0`. Output = **0**.

!!! c-info 💡
**Why the bias is not optional.** The weighted sum `w·x` on its own can only *rotate* the line that separates "high output" from "low output" — and that line is stuck passing through the origin (all-zero input always scores zero). The bias `b` *shifts* that line away from the origin. Drop it and the neuron loses the ability to lean "fire" or "don't fire" independent of the inputs — that is a real loss of what it can express, not a minor knob.
!!!

#### See the boundary move
Don't take the "rotate" and "shift" words on faith — watch them. Drag the weights and the dividing line **rotates**; drag the bias and the whole line **slides**. Set the bias to **0** and notice the line is pinned to the origin, no matter how you turn the weights.

%%% viz src="../../viz/neuron-boundary.html" title="interactive: a neuron's decision boundary" caption="interactive — drag w₁, w₂, and b; watch the line rotate and shift."

%%% mathladder
title: one neuron: `output = activation(w·x + b)`
words: score each input by its weight and add them up, nudge the total with a bias, then squash the result through an activation.
formula: `output = activation( w·x + b )`. `x` = the inputs; `w` = one weight per input; `w·x` = the weighted sum (a dot product); `b` = bias (one number); `activation` = a squash like `ReLU = max(0, ·)`.
numbers: `x=[2,3]`, `w=[0.5,−1]`, `b=1`. Weighted sum = `2·0.5 + 3·(−1) = −2`; + bias → `−1`; `ReLU(−1) = 0`. Output = **0**.
sanity: `w` must hold exactly one number per input (here 2 weights for 2 inputs), and the output is a single number, not a list. A ReLU output of `0` just means the pre-activation was ≤ 0 — expected, not a bug.
%%%

#### Why a frontier lab cares
This is the **atom**. Line up many neurons that all read the same inputs, and computing them all at once is a single **matmul**: `x @ W + b`, then the activation — exactly M1 Day 4. Here is the payoff worth waiting for: GPT-3 has a weight matrix with **49,152 columns**, and every one of those columns *is* one neuron exactly like the one you just built, wired to 12,288 inputs instead of a handful. A frontier model is billions of these neurons, stacked into layers. And the **weights** are precisely what training adjusts.

!!! c-info 🔗
**Remember this chain:** a neuron = weighted sum + bias + activation → many neurons at once = a matmul (a layer) → stacked layers = a network → training tunes the weights. Everything later is scale on top of this.
!!!

#### The staff lens — one silent failure, one trade-off
A neuron is tiny, but two things about it are worth naming precisely — a bug that never announces itself, and a design choice you make the moment you create weights.

!!! c-warn ⚠️
**Failure mode (silent): symmetric initialization.** The weights start as *random* numbers — they are not values you set by hand. Here is the trap: if you start every weight at the *same* value (all zeros is the classic mistake), then in a layer every neuron computes the same output *and* later gets the same update, so they never become different from one another. A 512-neuron layer then has the power of a single neuron. Nothing crashes — the loss even drops a little. A senior engineer catches this in **code review** by spotting `np.zeros(...)` used for weights, and confirms it by printing the row-to-row variance of a weight matrix after a few steps: if every row is identical, symmetry never broke. (You will feel the full force of this once we have layers on Day 3 and updates on Day 5.)
!!!

!!! c-info ⚖️
**Trade-off: the scale of the random start.** Random breaks the symmetry, but *how big* those random numbers are is a real decision. Too large and the pre-activation `z = w·x + b` comes out huge, pushing the neuron to an extreme before learning even begins; too small and the signal shrinks toward zero as it passes through many layers. The fix is to scale the spread to the number of inputs (the "fan-in") — you will meet the named recipes (Xavier for smooth activations, He for ReLU) on Day 6. Today's point: weights are random *with a chosen scale*, not arbitrary.
!!!

~~~html
<div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "A neuron is <code>activation(w·x + b)</code> — a weighted sum of inputs, a bias that shifts the decision boundary off the origin, then a non-linear squash. The weighted sum is exactly a dot product, so a whole layer of neurons is one matmul <code>x @ W + b</code> followed by the activation; stack those and you have the network, and training only ever adjusts <code>w</code> and <code>b</code>. The failure juniors miss is symmetric initialization — if all weights start equal, every unit in a layer receives the same gradient and the layer collapses to one effective neuron, so weights are drawn randomly and scaled to fan-in while biases can safely start at zero."</div></div>
~~~

@@@ section id=s7 num=7 numclass=s-produce data_sec=produce tag="Produce" title="Predict the output, then run it — watch ReLU decide" gotit="Done"
Here's the fun part. **Before you write any code, predict the answer:** for `x=[2,3]`, `w=[0.5,−1]`, `b=1`, will the neuron's output come out positive, negative, or exactly **zero**? Jot the guess down. Then build the neuron in three lines and find out whether ReLU zeroed it — that little "I was right / oh, I was wrong" moment is what makes the day stick. Pick one path:

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-01-single-neuron/experiment.py`. Write `def neuron(x, w, b): return float(np.maximum(0.0, np.dot(w, x) + b))`, then run it on `x=[2,3]`, `w=[0.5,-1]`, `b=1` (expect 0.0) and on a case that stays positive (expect a positive number). Print each step. Run with `python3 sessions/m02-the-neuron/day-01-single-neuron/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the **frontier-experiment-lab** skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 2 Day 1 artifact.

Create sessions/m02-the-neuron/day-01-single-neuron/experiment.py that, with a comment on each step:
1. Defines relu(z) = np.maximum(0, z).
2. Defines neuron(x, w, b) = relu(np.dot(w, x) + b), printing the weighted sum, the value after bias, and the final output.
3. Runs it on x=[2,3], w=[0.5,-1], b=1 (expect weighted sum -2, +bias -1, relu -> 0.0).
4. Runs it on x=[2,3], w=[1,1], b=0 (expect weighted sum 5, relu -> 5.0) to show a positive case.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- Your `neuron(x, w, b)` prints three visible steps: the weighted sum, the value after adding the bias, and the final output.
- On `x=[2,3], w=[0.5,-1], b=1` the pre-activation lands at `−1`, so ReLU squashes it to `0.0` — the "don't fire" case. On a positive case like `w=[1,1], b=0` the total passes straight through as a positive number — the "fire" case.
- The thing worth noticing: nothing crashed and no error appeared, yet a negative pre-activation quietly turned into `0`. That silent zero is ReLU doing its one job — and you predicted it before the computer told you.

!!! c-info 📓
**5-minute research log · 5 分钟研究笔记:** 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m02-the-neuron/day-01-single-neuron/log.md`: (1) in your own words, what the three steps of a neuron are; (2) why the weighted sum is a dot product; (3) one thing still fuzzy (why the activation goes last is fair game).
!!!

@@@ js name=DEMOS
var DEMOS = {
  wsum:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">2</span>, <span class="num">3</span>])      <span class="dim"># two inputs</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = np.array([<span class="num">0.5</span>, <span class="num">-1.0</span>])  <span class="dim"># one weight per input</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.dot(w, x)          <span class="dim"># weighted sum = a dot product</span>\n'+
    '<span class="hl">-2.0</span>',
    take:'<b>①  Step 1 — weighted sum.</b> Multiply each input by its weight and add: 2·0.5 + 3·(−1) = −2. This is exactly the dot product from M1 Day 4.'},
  bias:{html:'<span class="prompt">&gt;&gt;&gt;</span> b = <span class="num">1.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> z = np.dot(w, x) + b   <span class="dim"># add the bias</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> z\n<span class="hl">-1.0</span>',
    take:'<b>②  Step 2 — add the bias.</b> One number shifts the result: z = −2 + 1 = −1. This z is the "pre-activation" value.'},
  act:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">def</span> <span class="fn">relu</span>(v): <span class="kw">return</span> np.maximum(<span class="num">0</span>, v)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> relu(z)              <span class="dim"># step 3: squash the pre-activation</span>\n'+
    '<span class="ok">0.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> relu(np.array([<span class="num">-1</span>, <span class="num">0</span>, <span class="num">3</span>]))  <span class="dim"># zeroes negatives, keeps positives</span>\n'+
    '<span class="ok">array([0, 0, 3])</span>',
    take:'<b>③  Step 3 — activation.</b> ReLU = max(0, z) squashes z into the output: relu(−1) = 0. The neuron\'s final output is <b>0</b>. Full neuron: relu(w·x + b).'}
};

@@@ js name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Two inputs feeding a neuron'><g font-family='monospace' font-size='14' text-anchor='middle'><rect x='60' y='30' width='60' height='34' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='90' y='53' fill='#1F6280'>x₁ = 2</text><rect x='60' y='72' width='60' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='90' y='93' fill='#1F6280'>x₂ = 3</text><text x='260' y='70' fill='#6B645E' font-size='12'>two inputs going into one neuron</text></g></svg>",
  note:"<b>Start with the inputs.</b> A neuron receives a list of numbers — here <code>x = [2, 3]</code>. These could be pixel values, word features, anything. The neuron will turn them into one output."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Each input multiplied by its weight'><g font-family='monospace' font-size='13' text-anchor='middle'><text x='80' y='45' fill='#1F6280'>x₁·w₁ = 2·0.5</text><text x='80' y='70' fill='#1F6280'>= 1.0</text><text x='80' y='100' fill='#9A7208'>x₂·w₂ = 3·(−1)</text><text x='260' y='72' fill='#6B645E'>weights say how much each input matters</text></g></svg>",
  note:"<b>Multiply by weights.</b> Each input gets a <b>weight</b> — how much this neuron cares about it. <code>2·0.5 = 1.0</code> and <code>3·(−1) = −3</code>. Weights are the numbers training will learn."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Sum the weighted inputs into z equals minus 2'><g font-family='monospace' font-size='15' text-anchor='middle'><text x='150' y='55' fill='#2C2A28'>1.0 + (−3.0)</text><text x='300' y='55' fill='#6B645E'>=</text><rect x='335' y='36' width='60' height='34' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='365' y='59' fill='#5E5191' font-weight='bold'>−2.0</text><text x='260' y='92' fill='#6B645E' font-size='12'>the weighted sum (a dot product) → z</text></g></svg>",
  note:"<b>Add them up — the weighted sum.</b> <code>1.0 + (−3.0) = −2.0</code>. This single number is exactly <code>np.dot(w, x)</code>, the dot product you learned in M1 Day 4."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Add the bias to get z equals minus 1'><g font-family='monospace' font-size='15' text-anchor='middle'><text x='150' y='55' fill='#2C2A28'>−2.0 + 1.0</text><text x='280' y='55' fill='#6B645E'>=</text><rect x='315' y='36' width='60' height='34' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='345' y='59' fill='#9A7208' font-weight='bold'>−1.0</text><text x='260' y='92' fill='#6B645E' font-size='12'>+ bias b = 1 → the pre-activation z</text></g></svg>",
  note:"<b>Add the bias.</b> One extra number shifts the result: <code>−2 + 1 = −1</code>. The bias lets a neuron lean toward firing or not, independent of the inputs."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='ReLU activation turns z equals minus 1 into output 0'><g font-family='monospace'><line x1='60' y1='95' x2='300' y2='95' stroke='#E5DFD6' stroke-width='1.5'/><line x1='180' y1='30' x2='180' y2='110' stroke='#E5DFD6' stroke-width='1'/><path d='M80 95 L180 95 L280 45' fill='none' stroke='#2D8B55' stroke-width='2.5'/><text x='170' y='125' font-size='11' fill='#6B645E' text-anchor='middle'>ReLU: flat for z&lt;0, rising for z&gt;0</text><circle cx='140' cy='95' r='4' fill='#C93B3B'/><text x='140' y='84' font-size='11' fill='#C93B3B' text-anchor='middle'>z=−1</text><text x='420' y='70' font-size='14' font-family='monospace' fill='#1a5c38' font-weight='bold' text-anchor='middle'>output = 0</text></g></svg>",
  note:"<b>Squash with the activation.</b> ReLU = max(0, z). Our z = −1 is negative, so the output is <code>0</code>. A positive z would pass through unchanged. That's the full neuron: <code>relu(w·x + b)</code>."},
 {viz:"<svg viewBox='0 0 520 150' role='img' aria-label='Many neurons side by side form a layer computed as x @ W + b'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='55' width='60' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='70' y='75' fill='#1F6280'>x (1,3)</text><text x='115' y='75' fill='#6B645E'>@</text><rect x='135' y='45' width='80' height='50' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='175' y='74' fill='#5E5191'>W (3,4)</text><text x='230' y='75' fill='#6B645E'>+ b →</text><rect x='300' y='55' width='70' height='30' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='335' y='75' fill='#1a5c38' font-weight='bold'>4 neurons</text><text x='400' y='75' fill='#6B645E'>→ act</text><text x='260' y='120' fill='#2C2A28' font-size='12'>a LAYER = many neurons at once = x @ W + b, then activation</text><text x='260' y='140' fill='#6B645E' font-size='11'>stack layers → a network · billions of these = a frontier model</text></g></svg>",
  note:"<b>Widen to a layer.</b> Put four neurons side by side, each with its own weights, and computing them all at once is one matmul: <code>x @ W + b</code>, then the activation — exactly M1 Day 4. (This picture uses a 3-input example just to show a wider layer — same idea as our 2-input neuron, only broader.) Stack layers and you have a network. A neuron is the atom; scale is the whole story from here."}
];

@@@ js name=QS
var QS=[
 {q:'1. What are the three steps a neuron does, in order?',
  opts:['Add bias, activation, weighted sum','Weighted sum, add bias, activation','Activation, weighted sum, add bias','Matmul, softmax, slice'],
  ans:1, fb:'Right. Weighted sum (w·x) → add the bias b → apply the activation. output = activation(w·x + b).'},
 {q:'2. The "weighted sum" of inputs and weights is which operation from Module 1?',
  opts:['Broadcasting','The dot product (part of matmul)','Slicing','Reshaping'],
  ans:1, fb:'Right. Multiply pairwise and add = a dot product, exactly np.dot(w, x) from M1 Day 4.'},
 {q:'3. What is the bias\'s job in a neuron?',
  opts:['It sets the number of inputs','It shifts the pre-activation value up or down','It is the activation function','It stores the output'],
  ans:1, fb:'Right. The bias is one number added after the weighted sum; it lets the neuron lean higher or lower independent of the inputs.'},
 {q:'4. You build a layer of many neurons and start every weight at <code>0</code>. Training runs, no error is thrown, but the whole layer behaves as if it were a single neuron. Why?',
  opts:['A zero start is illegal and should have crashed','With identical starting weights, every neuron in the layer computes the same output and receives the same update, so they never become different — the layer has the power of one unit','Zero weights make the dot product undefined','The bias grew too large and saturated every neuron'],
  ans:1, fb:'Right. Identical weights → identical outputs → identical gradients → the neurons stay copies of each other forever. This is why weights start <b>random</b> (with a sensible scale), never all-equal. Nothing crashes — the failure is silent, so you catch it by checking that a weight matrix\'s rows actually differ.'}
];
