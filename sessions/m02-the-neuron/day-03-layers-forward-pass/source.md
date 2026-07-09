---
quest_id: wf2-d03-forward
donor: m02-day-03.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 2 · Day 3 — Layers &amp; the Forward Pass"
spine: "assembly line"
notebook_yardstick: 00-neural-networks/fundamentals/05_forward_propagation.ipynb
---

@@@ region name=title
<title>Module 2 · Day 3 — Layers &amp; the Forward Pass</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M2 Day 3</div>
@@@ region name=sidebar_nav
<nav aria-label="Sections">
      <div class="nav-group-label">Module 02 · Train</div>
      <button class="nav-link" data-target="home"><span class="nl-dot"></span>Start here</button>
      <button class="nav-link" data-target="s1"><span class="nl-dot"></span>1 · What is it</button>
      <button class="nav-link" data-target="s2"><span class="nl-dot"></span>2 · Intuition</button>
      <button class="nav-link" data-target="s3"><span class="nl-dot"></span>3 · Playground</button>
      <button class="nav-link" data-target="s4"><span class="nl-dot"></span>4 · Mechanism &amp; why</button>
      <button class="nav-link" data-target="s5"><span class="nl-dot"></span>5 · Build it up</button>
      <button class="nav-link" data-target="s6"><span class="nl-dot"></span>6 · Quiz</button>
      <button class="nav-link" data-target="s7"><span class="nl-dot"></span>7 · Produce</button>
    </nav>
@@@ region name=nav_prev
<a class="lnav prev" href="../day-02-activations/lesson.html"><span class="d">← Prev</span><span class="t">Activation Functions</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../review-part-a.html"><span class="d">Next →</span><span class="t">Review Gate</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 2 · Train · Day 3</span>
      <h1>Layers &amp; the Forward Pass<span class="sub">From One Neuron to a Prediction</span></h1>
      <p class="lede">Every time anyone sends a prompt to a model, a server answers with exactly one operation: a <strong>forward pass</strong>. Today you build it. You already have a neuron and you know why the activation matters — a forward pass is just chaining layers (input → layer → layer → output) and reading the answer off the end.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>By the time you close this tab, "running a model" will feel like nothing more than passing data down a line: you'll build a 2-layer forward pass, watch the shapes hand off from input to output, and see why this <em>is</em> the model doing its job.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">A layer, and the forward pass</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (a neuron), Day 2 (activations), and M1 D4 (a layer = <code>x@W+b</code>). This lesson just chains them. <b>Nothing new needed.</b></div></div>

      <h4>First, the picture — hold this before any definition</h4>
      <p>Picture a factory assembly line. Raw material comes in one end; each station does one small job and hands the piece to the next; a finished product rolls off the far end. A neural network runs the same way: your input walks through a line of <strong>layers</strong>, each one reshaping it a little, and the answer drops out at the end. That walk from input to answer has a name — the <strong>forward pass</strong> — and it is literally what happens every time a model is used. Keep that picture; the two definitions below are just names for the line and the walk along it.</p>

      <h4>What is a layer?</h4>
      <p>A <span class="term" data-tip="A group of neurons computed together: activation(x @ W + b). One matmul + bias + activation.">layer</span> is a group of neurons computed in one shot: <code>activation(x @ W + b)</code>. That's M1's matmul, plus a bias, then M2's activation. Its output becomes the input to the next layer.</p>

      <h4>What is a forward pass?</h4>
      <p>A <span class="term" data-tip="Running the input through every layer in order to produce the output (the prediction). Called forward because data flows input → output.">forward pass</span> is running the input through every layer, in order, to produce the final output. Data flows <em>forward</em>: <code>input → layer 1 → layer 2 → … → output</code>. That final output is the model's answer.</p>

      <h4>How does it relate to what you know?</h4>
      <p>Each layer is a line you can already write: <code>h = relu(x @ W + b)</code>. A forward pass just does that repeatedly, feeding each layer's output into the next. The one rule to respect: the shapes must <strong>chain</strong> (M1 Day 4) — each layer's output width must match the next layer's expected input width. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> vLLM, TensorRT-LLM, and SGLang are real, widely used software built to do just one thing at massive scale: run the forward pass you're about to build.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> "layer" 和 "forward pass" 听起来很高级，让人以为里面藏着新魔法 — the words <i>layer</i> and <i>forward pass</i> sound advanced, as if new machinery is involved. In plain terms: a layer is just yesterday's neuron math done for many neurons at once (one matmul + bias + activation), and a forward pass is only "do that layer, then feed its output into the next." The one real gotcha is keeping the widths lined up.</div></div>
      <button class="gotit" type="button">Got layer + forward pass — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">An assembly line</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>You already have the assembly-line picture from Section 1 — here it is as two cards, one for the stations and one for what rolls off the end:</p>
      <div class="relate">
        <div class="card"><span class="big">🏭</span><h5>Layers = stations on a line</h5><p>Raw material (the input) enters. Each station (layer) reshapes it a bit and passes it on. Nothing skips ahead — it flows one station at a time.</p></div>
        <div class="card"><span class="big">📦</span><h5>Forward pass = the finished product</h5><p>What rolls off the end of the line is the output — the model's prediction. Running the line once, start to finish, is one forward pass.</p></div>
      </div>
      <p><strong>In one line:</strong> a forward pass sends the input down an assembly line of layers; the thing that comes out the end is the prediction.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a real factory can't run backwards, but a network can — running <em>backwards</em> to adjust the weights is called backpropagation, and it's how the model learns (Day 5). Today is only the forward direction.</div></div>
      <p><b>直觉 / Intuition:</b> 直觉上，forward pass 就是数据在流水线上"从左走到右" — 每一站（layer）把它加工一下，最后出来的就是预测。The technical point: each layer applies <code>activation(x @ W + b)</code> and hands its output to the next layer as input, so the whole network is just those transforms chained end to end.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The recipe, the shape chain, and why this is "the model"</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The recipe</h4>
      <div class="callout c-ok"><span class="ic">🧮</span><div>One layer: <code>h = activation(x @ W + b)</code>. A forward pass: repeat, feeding each output into the next — <code>h₁ = relu(x @ W₁ + b₁)</code>, then <code>out = h₁ @ W₂ + b₂</code>.</div></div>
      <p>One convention we hold for the rest of the module: inputs are <b>rows</b>, and a layer is <code>x @ W</code> with <code>W</code> shaped <code>(in, out)</code>. The last layer usually <b>skips</b> the activation, so its output is a vector of <span class="term" data-tip="The raw, un-squashed scores a network outputs before any softmax/sigmoid. Higher = more strongly predicted. Turned into probabilities by softmax on Day 4.">raw scores (logits)</span> — you turn those into probabilities later with softmax (Day 4).</p>

      <h4>The shape chain</h4>
      <p>Widths must line up (M1 Day 4): input <code>(1, 3)</code> → after W₁ <code>(3, 4)</code> → hidden <code>(1, 4)</code> → after W₂ <code>(4, 2)</code> → output <code>(1, 2)</code>. Each layer's output width is the next layer's input width. Get one wrong and you get the classic matmul shape error.</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — the forward-pass shape chain</b>
      <br><b>1 · In words:</b> each layer's output width must equal the next layer's input width; the widths hand off down the chain like batons.
      <br><b>2 · The formula:</b> <code>h₁ = relu(x @ W₁ + b₁)</code>, then <code>out = h₁ @ W₂ + b₂</code>. Shapes: <code>x (1,d₀)</code>, <code>W₁ (d₀,d₁)</code>, <code>W₂ (d₁,d₂)</code>; each matmul's shared inner dim must match (M1 Day 4).
      <br><b>3 · Tiny numbers:</b> <code>x (1,3)</code> → <code>@ W₁ (3,4)</code> → <code>h₁ (1,4)</code> → <code>@ W₂ (4,2)</code> → <code>out (1,2)</code>. The 3 and the 4 cancel in turn; only the outer widths survive.
      <br><b>4 · Sanity check:</b> read the widths as <code>3 → 4 → 2</code>. If <code>W₂</code> were <code>(3,2)</code> instead of <code>(4,2)</code>, the second matmul's inner dims (4 vs 3) wouldn't match and it crashes — the classic shape error.</div></div>
      <div class="callout c-ok"><span class="ic">🧮</span><div><b>A tiny forward pass with real numbers.</b> The shapes above traced a 3→4→2 net; here is an even smaller <code>2→2→1</code> net run with actual numbers so you can see the arithmetic. Input <code>x=[2,3]</code>, <code>W₁=[[1,−1],[0,1]]</code>, <code>b₁=[0,−4]</code>: pre-activation <code>x@W₁+b₁ = [2,1]+[0,−4] = [2,−3]</code>; ReLU zeroes the negative → hidden <code>h=[2,0]</code>. Then <code>W₂=[[1],[1]]</code>, <code>b₂=[0]</code>: output <code>h@W₂+b₂ = [2]</code>. Notice the ReLU actually changed the answer — without it the <code>−3</code> would have flowed on and the output would be <code>[−1]</code>.</div></div>

      <h4>Why a frontier lab cares</h4>
      <p>The forward pass <em>is</em> <span class="term" data-tip="Running a trained model on new input to get an answer. The forward pass is inference. Serving = doing this for real users at scale.">inference</span> — running the model to get an answer. Every time you send a prompt to a model, a server runs exactly the operation you just built. The engines from the teaser above spend most of their code on exactly the batching trade-off you're about to meet below. That trade-off is how many requests to group together before running one shared forward pass. The reverse direction, which nudges the weights to make future forward passes better, is <span class="term" data-tip="Running the network backward to compute how each weight should change to reduce the error. How models learn. Covered on Day 5.">backpropagation</span> — Day 5.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> neuron → layer (matmul + bias + activation) → forward pass (chain the layers) → the output is the prediction = inference. Training (M3) runs this backward to improve the weights.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>Chaining layers is simple to write, which is exactly why one small omission slips through unnoticed, and one serving choice always comes up.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): the missing activation between layers.</b> If you write <code>h1 = x @ W1 + b1</code> and forget the <code>relu</code>, then feed <code>h1</code> straight into the next layer, the code still runs and the output shape is unchanged. But two stacked <em>linear</em> layers collapse into one: <code>(x W1) W2 = x (W1 W2)</code> (from Day 2). Your "deep" network is secretly a single linear layer, so it can only draw a straight line and quietly learns far less. On the assembly line, that is two neighboring stations that only pass the piece along without ever shaping it — the line is secretly one station shorter than it looks. Nothing crashes and the shapes all match. A senior engineer catches this in <b>code review</b> by checking that every hidden layer has a non-linear activation between it and the next matmul — not by reading the loss, which just looks disappointing. A sibling silent bug is <b>broadcasting</b>: a bias added on the wrong axis, or a <code>(batch,1)</code> vs <code>(batch,)</code> shape, produces a valid tensor of the wrong meaning with no error — which is exactly why you print shapes at every step and assert the ones you expect.</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: batching the forward pass.</b> A forward pass is inference, and you can run many inputs at once by stacking them into a batch — the matmul <code>X @ W</code> handles a whole batch in one shot. Bigger batches use the hardware far more efficiently (higher <b>throughput</b>, more answers per second), but each individual request waits longer while the batch fills up (higher <b>latency</b>). How large a batch to serve is a classic <b>design-review</b> decision: it trades one user's response time against the total cost of serving everyone.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "A forward pass chains layers — each is <code>activation(x @ W + b)</code> and its output feeds the next as input — which is exactly inference; every prompt to a served model runs this. Two things I watch: the shape chain (each layer's output width = the next layer's input width, or the matmul crashes) and the silent bug of a missing activation between layers, which collapses the whole stack to one linear map with no crash. At serving time the key knob is batch size — bigger batches raise throughput but hurt per-request latency."</div></div>
      <button class="gotit" type="button">Got the forward pass</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Predict the shapes, then run the line</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Here's the fun part. <b>Before you run anything, predict:</b> you send a <code>(1,3)</code> input down a <code>(3,4)</code> station then a <code>(4,2)</code> station — what shape rolls off the end? Write your guess. Then build the two-layer forward pass and watch the widths hand off <code>3 → 4 → 2</code>. For the real payoff, delete the <code>relu</code> between the layers and watch a genuinely different number come out — proof the bend is doing real work. Pick one path:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py</code>. Make <code>W1 (3,4)</code>, <code>b1 (4,)</code>, <code>W2 (4,2)</code>, <code>b2 (2,)</code>. Write <code>forward(x)</code> that computes <code>h = relu(x@W1+b1)</code> then <code>out = h@W2+b2</code>, printing the shape after every step for <code>x</code> of shape <code>(1,3)</code>; confirm the output is <code>(1,2)</code>. Recompute one hidden unit by hand and check it matches. Then run a batch of 4 inputs <code>(4,3)</code>, confirm output <code>(4,2)</code>, and that row 0 equals the single-input result. Run with <code>python3 sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 2 Day 3 artifact.

Create sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py that, with a comment on each step:
1. Defines relu(z)=np.maximum(0,z).
2. Sets up a tiny 2-layer network: W1 shape (3,4), b1 (4,), W2 (4,2), b2 (2,) (random values, fixed seed).
3. Defines forward(x): h = relu(x@W1 + b1); out = h@W2 + b2; return out — printing x.shape, h.shape, out.shape at each step.
4. Runs forward on x of shape (1,3), confirms the output shape is (1,2), and prints the output vector as 'the prediction'.
5. Runs a batch of 4 inputs (shape (4,3)) through forward, confirms the output shape is (4,2), and checks row 0 equals the single-input result.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>What you should see (check your prediction)</h4>
      <ul>
        <li>Your <code>forward(x)</code> prints the shape after every step for <code>x</code> of shape <code>(1,3)</code>, ending at output <code>(1,2)</code> — the widths hand off <code>3 → 4 → 2</code> just as you predicted.</li>
        <li>The hidden step uses a real <code>relu</code> non-linearity, not a bare matmul; post-ReLU hidden values are all ≥ 0.</li>
        <li>One hand-computed hidden unit matches the vectorized <code>x@W1+b1</code> entry (to ~1e-10), and a batch of 4 prints output <code>(4,2)</code> with row 0 equal to the single-input result.</li>
        <li>You can trace the width chain <code>3 → 4 → 2</code> aloud and say where a wrong width would crash.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-03-layers-forward-pass/log.md</code>: (1) trace the shapes 3 → 4 → 2 in your own words; (2) what a forward pass has to do with inference; (3) what breaks if you forget the activation between layers.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 2 · Day 3 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Layers &amp; the Forward Pass</b>.<br>Next up: <b>Review Gate</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  layer1:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([[<span class="num">1.</span>, <span class="num">2.</span>, <span class="num">3.</span>]])   <span class="dim"># (1,3) input</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> W1 = np.random.rand(<span class="num">3</span>, <span class="num">4</span>); b1 = np.zeros(<span class="num">4</span>)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> h = relu(x @ W1 + b1)   <span class="dim"># layer 1: matmul + bias + activation</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> h.shape\n<span class="hl">(1, 4)</span>',
    take:'<b>①  Layer 1.</b> relu(x @ W1 + b1) turns the width-3 input into a width-4 hidden vector <code>h</code>. One matmul, one bias, one activation — exactly a layer.'},
  layer2:{html:'<span class="prompt">&gt;&gt;&gt;</span> W2 = np.random.rand(<span class="num">4</span>, <span class="num">2</span>); b2 = np.zeros(<span class="num">2</span>)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> out = h @ W2 + b2       <span class="dim"># layer 2: usually no activation at the end</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> out.shape\n<span class="hl">(1, 2)</span>',
    take:'<b>②  Layer 2.</b> Feed layer 1\'s output <code>h</code> (width 4) into layer 2 → width 2. The widths chain: 4 must match W2\'s 4. The last layer often skips ReLU so you keep raw scores.'},
  forward:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">def</span> <span class="fn">forward</span>(x):\n'+
    '<span class="dim">...</span>     h = relu(x @ W1 + b1)   <span class="dim"># layer 1</span>\n'+
    '<span class="dim">...</span>     <span class="kw">return</span> h @ W2 + b2      <span class="dim"># layer 2</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> forward(x).shape\n<span class="hl">(1, 2)</span>',
    take:'<b>③  The forward pass.</b> input → layer 1 → layer 2 → output, in one function. Shapes flowed (1,3) → (1,4) → (1,2). That final vector is the model\'s <b>prediction</b>.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 90' role='img' aria-label='Input vector of shape (1,3)'><g font-family='monospace' font-size='13' text-anchor='middle'><rect x='210' y='30' width='100' height='34' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='52' fill='#1F6280'>input x (1,3)</text></g></svg>",
  note:"<b>Start with the input.</b> One example with 3 features, shape <code>(1, 3)</code>. This is what enters the network — a single item going down the assembly line."},
 {viz:"<svg viewBox='0 0 520 100' role='img' aria-label='Layer 1 matmul and bias: x times W1 plus b1 gives shape (1,4)'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='30' y='40' width='70' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='65' y='60' fill='#1F6280'>x (1,3)</text><text x='115' y='60' fill='#6B645E'>@</text><rect x='135' y='36' width='70' height='38' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='170' y='60' fill='#5E5191'>W₁ (3,4)</text><text x='220' y='60' fill='#6B645E'>+ b₁ →</text><rect x='300' y='40' width='120' height='30' rx='5' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='360' y='60' fill='#9A7208'>(1,4) pre-activation</text></g></svg>",
  note:"<b>Layer 1 — the matmul + bias.</b> <code>x @ W₁ + b₁</code> maps the width-3 input to a width-4 vector (M1 Day 4). This is the pre-activation, <code>z</code>."},
 {viz:"<svg viewBox='0 0 520 90' role='img' aria-label='ReLU applied gives the hidden vector h of shape (1,4)'><g font-family='monospace' font-size='13' text-anchor='middle'><rect x='120' y='30' width='90' height='34' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='165' y='52' fill='#9A7208'>(1,4) z</text><text x='240' y='52' fill='#6B645E'>ReLU →</text><rect x='320' y='30' width='120' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='380' y='52' fill='#1a5c38' font-weight='bold'>hidden h (1,4)</text></g></svg>",
  note:"<b>Apply the activation.</b> ReLU turns the pre-activation into the <b>hidden</b> vector <code>h</code> (still width 4). Layer 1 is done: <code>h = relu(x @ W₁ + b₁)</code>."},
 {viz:"<svg viewBox='0 0 520 100' role='img' aria-label='Layer 2: h times W2 plus b2 gives the output of shape (1,2)'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='30' y='40' width='70' height='30' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='65' y='60' fill='#1a5c38'>h (1,4)</text><text x='115' y='60' fill='#6B645E'>@</text><rect x='135' y='36' width='70' height='38' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='170' y='60' fill='#5E5191'>W₂ (4,2)</text><text x='220' y='60' fill='#6B645E'>+ b₂ →</text><rect x='300' y='40' width='120' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2.5'/><text x='360' y='60' fill='#1F6280' font-weight='bold'>output (1,2)</text></g></svg>",
  note:"<b>Layer 2 — to the output.</b> Feed <code>h</code> into a second layer: <code>h @ W₂ + b₂</code> → width 2. The last layer usually skips the activation, giving raw output scores of shape <code>(1, 2)</code>."},
 {viz:"<svg viewBox='0 0 520 90' role='img' aria-label='Shape chain 3 to 4 to 2, inner widths must match'><g font-family='monospace' font-size='14' text-anchor='middle'><rect x='40' y='34' width='60' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='70' y='54' fill='#1F6280'>3</text><text x='120' y='54' fill='#6B645E'>→</text><rect x='150' y='34' width='60' height='30' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='180' y='54' fill='#1a5c38'>4</text><text x='230' y='54' fill='#6B645E'>→</text><rect x='260' y='34' width='60' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='290' y='54' fill='#1F6280'>2</text><text x='420' y='54' fill='#6B645E' font-size='11'>each width feeds the next</text></g></svg>",
  note:"<b>The shape chain.</b> Widths flow <code>3 → 4 → 2</code>. Every layer's output width must equal the next layer's input width — otherwise you hit the matmul shape error from M1 Day 4. Trace the chain and you can trace the whole network."},
 {viz:"<svg viewBox='0 0 520 100' role='img' aria-label='The full forward pass from input to prediction is inference; the reverse is training'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='30' y='38' width='300' height='34' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='180' y='60' fill='#1F6280' font-weight='bold'>input → layers → output = FORWARD PASS (inference)</text><text x='355' y='60' fill='#6B645E'>·</text><text x='435' y='55' fill='#C93B3B' font-size='11'>reverse =</text><text x='435' y='70' fill='#C93B3B' font-size='11'>training (M3)</text></g></svg>",
  note:"<b>That's the whole thing.</b> Input to prediction in one sweep = a <b>forward pass</b> = <b>inference</b> — what runs every time anyone uses a model. Running it backward to improve the weights is training — the training half of this module ahead (Days 4–5): loss, gradients, backprop."}
];
@@@ region name=QS
var QS=[
 {q:'1. A "forward pass" is:',
  opts:['adjusting the weights to reduce error','running the input through the layers, in order, to produce the output','a single matrix multiply','the bias term'],
  ans:1, fb:'Right. Forward pass = feed the input through every layer to get the output (the prediction). The reverse is training.'},
 {q:'2. To chain two layers, each layer\'s output width must equal:',
  opts:['the batch size','the next layer\'s input width','the number of layers','1'],
  ans:1, fb:'Right. Widths must match so the matmuls line up (M1 Day 4): output width of one layer = input width of the next.'},
 {q:'3. One layer computes:',
  opts:['x + W','activation(x @ W + b)','x @ x','just a bias'],
  ans:1, fb:'Right. A layer is a matmul plus bias plus an activation: activation(x @ W + b).'},
 {q:'4. You build a 3-layer network. It runs with no error and the output shapes are all correct, but it can only fit straight-line patterns and its loss barely beats a single linear model. Where do you look first, and why?',
  opts:['Add more layers, since more depth always fixes a weak model','Check that a non-linear activation sits between every pair of layers — if one is missing, the linear layers collapse into a single linear layer and depth buys nothing','The batch size is wrong, which is the only thing that limits what a network can fit','A weak model with correct shapes means the data is unlearnable'],
  ans:1, fb:'Right. Two linear layers with no activation between them satisfy (x W1) W2 = x (W1 W2) — one combined matrix. A forgotten activation makes a "deep" network secretly linear. Shapes still match and nothing errors, so you look at the activations, not the shapes.'}
];
