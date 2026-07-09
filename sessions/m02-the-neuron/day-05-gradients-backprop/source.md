---
quest_id: wf3-d02-backprop
donor: m02-day-05.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 2 · Day 5 — Gradients &amp; Backpropagation"
spine: "hiker in fog"
notebook_yardstick: 00-neural-networks/fundamentals/07_backpropagation.ipynb
---

@@@ region name=title
<title>Module 2 · Day 5 — Gradients &amp; Backpropagation</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M2 Day 5</div>
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
<a class="lnav prev" href="../day-04-loss/lesson.html"><span class="d">← Prev</span><span class="t">Loss Functions</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-06-training-loop/lesson.html"><span class="d">Next →</span><span class="t">The Training Loop</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 2 · Train · Day 5</span>
      <h1>Gradients &amp; Backpropagation<span class="sub">Which Way Is Downhill?</span></h1>
      <p class="lede">You have a loss — one number saying how wrong the model is. Now the key question: which way should each weight move to make that number smaller? The answer is the <strong>gradient</strong> (the slope), and <strong>backpropagation</strong> is the clever trick that computes it for every weight at once.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><b>By the end, you'll be able to:</b> say what a gradient is, do one gradient-descent step by hand, and describe backprop as the chain rule run backward through the network.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">Gradient, gradient descent, backprop</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 4 (the loss), Day 3 (the forward pass), and the idea of a slope. <b>No heavy calculus</b> — we keep it to "slope = steepness."</div></div>

      <h4>First, the picture — hold this before any formula</h4>
      <p>Imagine you are standing on a foggy hillside and you want to reach the bottom of the valley. You can't see far, but you <em>can</em> feel which way the ground tilts most steeply downward under your feet. So you take a small step that way, feel again, and step again — and little by little you walk down to the low ground. That downhill feeling under your feet is the <strong>gradient</strong>, and walking down by small steps is <strong>gradient descent</strong>. Hold that picture; the three terms below are just names for the feeling and the walk.</p>

      <h4>What is a gradient?</h4>
      <p>A <span class="term" data-tip="The slope of the loss with respect to a weight: how much the loss changes if you nudge that weight. It points in the direction the loss increases fastest.">gradient</span> is the <strong>slope</strong> of the loss as you wiggle one weight — how much the loss goes up (or down) if you increase that weight a little. A positive gradient means "increasing this weight raises the loss," so to <em>lower</em> the loss you move the other way.</p>

      <h4>What is gradient descent?</h4>
      <p><span class="term" data-tip="The rule for improving weights: subtract a small fraction of the gradient. w = w - learning_rate * gradient. Repeated, it walks the loss downhill.">Gradient descent</span> is the rule for using that slope: take a small step in the <em>downhill</em> direction — against the gradient — then feel the slope again and step again, so the loss drops a little each time. The <span class="term" data-tip="The step size in gradient descent. Too big overshoots; too small learns slowly.">learning rate</span> is how big each step is. (The exact one-line update rule, with a worked step, is in Section 4.)</p>

      <h4>What is backpropagation?</h4>
      <p>A network has millions of weights. <span class="term" data-tip="The algorithm that computes the gradient of the loss for every weight, efficiently, by applying the chain rule backward from the loss to the inputs.">Backpropagation</span> is the efficient algorithm that computes the gradient for <strong>all</strong> of them in one sweep — running <em>backward</em> from the loss to the inputs, using the chain rule. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> jax.grad and torch.autograd — the exact tools frontier labs train with — are just automated versions of the chain rule you're about to run by hand.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> backprop 有个吓人的名声，好像是什么高深的黑魔法 — backprop has a scary reputation, as if it were deep black magic. In plain terms: it's just one question asked for every weight — "if I nudge you a tiny bit, does the loss go up or down, and how fast?" The gradient is the answer, and backprop is only the chain rule applied backward so you get all those answers in one sweep.</div></div>
      <button class="gotit" type="button">Got the three words — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">A hiker in the fog</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>You already have the foggy-hill picture from Section 1 — here it is as two cards, one for the slope underfoot and one for the walk downhill:</p>
      <div class="relate">
        <div class="card"><span class="big">🥾</span><h5>Gradient = the slope underfoot</h5><p>You're on a foggy hill (the loss). You can't see the bottom, but you can feel which way is steepest down. That feeling is the gradient. Step that way.</p></div>
        <div class="card"><span class="big">🔁</span><h5>Descent = many small steps</h5><p>Take a small step downhill, feel the slope again, step again. Repeat. You gradually reach a low point — a model with small loss. Step size = learning rate.</p></div>
      </div>
      <p><strong>In one line:</strong> the gradient says which way is downhill for each weight; gradient descent takes small steps that way until the loss is low.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: the loss landscape has millions of dimensions (one per weight), not two. And backprop is what lets you feel the slope in <em>all</em> those dimensions at once, cheaply — a single hiker couldn't.</div></div>
      <p><b>直觉 / Intuition:</b> 直觉上，gradient 就是你脚下那块地的"坡度" — 它只告诉你哪边更低、有多陡；朝那边迈一小步，重复很多次，就走到了谷底。The technical point: the gradient is the slope of the loss with respect to each weight, and gradient descent repeatedly nudges each weight a little in the downhill direction.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The chain rule, backward — and why it runs all of AI</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The update rule</h4>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>gradient = slope of loss w.r.t. w</code>, then <code>w ← w − learning_rate × gradient</code>. Repeat over many steps.</div></div>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — one gradient-descent step on <code>f(w)=w²</code></b>
      <br><b>1 · In words:</b> the gradient is how fast the loss changes as <code>w</code> changes; step <code>w</code> a little against that slope to lower the loss.
      <br><b>2 · The formula:</b> <code>gradient = df/dw</code>, then <code>w ← w − lr × gradient</code>. For <code>f(w)=w²</code>, <code>df/dw = 2w</code>; <code>lr</code> = learning rate (step size).
      <br><b>3 · Tiny numbers:</b> at <code>w=3</code>, gradient <code>= 2·3 = 6</code>. With <code>lr=0.1</code>: <code>w ← 3 − 0.1·6 = 2.4</code>, and the loss drops from <code>9</code> to <code>5.76</code>. Step again: <code>2.4 → 1.92 → …</code> toward 0.
      <br><b>4 · Sanity check:</b> the sign points downhill — a positive slope moves <code>w</code> left, a negative slope moves it right. Numerically, <code>(f(w+h)−f(w−h))/(2h)</code> at <code>w=3</code> with tiny <code>h</code> gives ≈ 6, matching <code>2w</code>.</div></div>

      <h4>Why backprop, and the chain rule</h4>
      <p>A weight deep in the network affects the loss only through everything after it. The <span class="term" data-tip="Calculus rule: to get the slope through a chain of steps, multiply the local slopes of each step together.">chain rule</span> says: the slope through a chain of steps is the <strong>product of the local slopes</strong>. Backprop starts at the loss and walks backward, multiplying local slopes layer by layer, so every weight's gradient falls out in one pass — reusing the values already computed in the forward pass.</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — the chain rule with real numbers (one path).</b> Take a tiny neuron with a sigmoid output and a squared-error loss: <code>z = w·x + b</code>, <code>a = sigmoid(z)</code>, <code>L = (a − y)²</code>. The chain rule stacks three local slopes: <code>dL/dw = dL/da · da/dz · dz/dw</code>.
      <br><b>1 · Forward first:</b> with <code>x=2, w=0.5, b=0, y=1</code>: <code>z = 1</code>, <code>a = sigmoid(1) ≈ 0.731</code>, <code>L ≈ 0.072</code>.
      <br><b>2 · Then multiply backward:</b> <code>dL/da = 2(a−y) ≈ −0.538</code>; <code>da/dz = a(1−a) ≈ 0.197</code>; <code>dz/dw = x = 2</code>.
      <br><b>3 · Product:</b> <code>dL/dw ≈ −0.538 · 0.197 · 2 ≈ −0.212</code>. Negative → nudging <code>w</code> <em>up</em> lowers the loss. That single number is exactly what backprop produces for every weight at once.
      <br><b>4 · Note:</b> this is a from-scratch illustration of the mechanics — we pair sigmoid with squared error only to get a non-trivial middle slope <code>a(1−a)</code>; for real classification you'd use cross-entropy (Day 4), not MSE.</div></div>

      <h4>Why a frontier lab cares</h4>
      <p>Backprop is how <strong>every</strong> neural network learns. It's the reverse of the forward pass and costs about twice as much (so a training step ≈ 3× a forward pass — one forward, two forward-equivalents backward — which is exactly where the "6" in Kaplan et al.'s <code>C ≈ 6ND</code> compute formula comes from). This is exactly what the tools from the teaser above compute automatically (<span class="term" data-tip="Automatic differentiation: the framework computes gradients for you, e.g. jax.grad. You will meet it in the JAX modules.">autodiff</span>). Knowing it's just "chain rule, backward" demystifies the whole thing.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> loss → gradient (slope for each weight) → step downhill (w − lr·grad) → backprop computes all gradients in one backward sweep → repeat = training. That plain rule <code>w ← w − lr·grad</code> is <b>vanilla gradient descent</b> — the baseline every optimizer on Day 7 (momentum, Adam) builds on. Tomorrow we assemble it into the training loop.</div></div>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>One honest caveat: the hill is bumpy.</b> Our <code>f(w)=w²</code> bowl has a single lowest point, but the real loss over millions of weights is <b>non-convex</b> — full of local dips, flat plateaus, and saddle points. Gradient descent does <em>not</em> find the global minimum; it settles into <em>a</em> good-enough valley. That is usually fine (many valleys give low loss), and it's exactly why the tricks on Days 7–8 — momentum, the noise from small batches, learning-rate schedules — exist: to move well on a bumpy surface, not to guarantee the deepest point.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>Backprop multiplies many local slopes together, and that single fact is the source of both a quiet training failure and a real memory choice.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): vanishing gradients.</b> The chain rule multiplies one local slope per layer. If those slopes are each a little below <code>1</code> (which sigmoid activations produce), then in a deep network you multiply many small numbers together and the gradient that reaches the early layers shrinks toward <code>0</code>. Those early layers then barely change, no matter how long you train. Nothing crashes — the loss just stops improving and sits there. A senior engineer catches this in <b>code review</b> by logging the size (norm) of the gradients per layer: if the early-layer gradients are orders of magnitude smaller than the late-layer ones, the signal is dying on the way back. The mirror failure is <b>exploding gradients</b>: if the local slopes are mostly above <code>1</code>, the product blows up and the gradients (and the loss) shoot to <code>inf</code>/<code>NaN</code> — the fix there is gradient clipping (Day 8). (Vanishing is exactly why ReLU and residual connections exist.)</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: store activations or recompute them.</b> Backprop reuses the values from the forward pass, so the fast way is to keep every layer's activations in memory — but for a big model that memory is huge. The alternative, <span class="term" data-tip="Gradient checkpointing: during the backward pass, re-run parts of the forward pass to regenerate activations instead of storing them, trading extra compute for less memory.">gradient checkpointing</span>, throws most activations away and re-runs part of the forward pass during backward to get them back. You save a lot of memory but pay extra compute (more FLOPs per step). Whether to checkpoint is a <b>design-review</b> decision that trades GPU memory against training speed.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "The gradient is the slope of the loss w.r.t. each weight; the update is <code>w ← w − lr × grad</code>. Backprop is the chain rule applied backward from the loss — it multiplies local slopes layer by layer and reuses forward-pass values, so all gradients come out in one backward sweep at roughly 2× a forward pass. The silent failure is vanishing gradients: multiply many sub-1 slopes (sigmoids) through a deep net and the early-layer gradients shrink to ~0 and stop learning — which is why ReLU and residual connections exist. I'd log per-layer gradient norms to catch it."</div></div>
      <button class="gotit" type="button">Got backprop</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Roll a weight downhill, then prove backprop is just the chain rule</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Here's the payoff you can watch happen. Roll a weight down the <code>f(w)=w²</code> bowl and see the loss fall <code>9 → 5.76 → …</code> toward 0. Then the real "aha": compute one neuron's gradient <em>by hand</em> with the chain rule (<code>dL/dw ≈ −0.212</code>) and check it against a finite-difference estimate — they agree to better than <code>1e-6</code>. That match is proof that backprop is nothing more than the chain rule computed exactly. Pick one path:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py</code>, in two parts.<br>
      <b>Part (a) — gradient descent.</b> For <code>f(w)=w²</code>, compute the numerical slope at <code>w=3</code> (expect ≈6). Then loop 10 times doing <code>w = w − 0.1·(2·w)</code>, printing <code>w</code> and <code>w²</code> each step, and watch the loss fall toward 0.<br>
      <b>Part (b) — backprop by hand.</b> Take one sigmoid neuron with squared-error loss: <code>z=w·x+b</code>, <code>a=sigmoid(z)</code>, <code>L=(a−y)²</code>, using the Math-Ladder anchors <code>x=2, w=0.5, b=0, y=1</code>. Compute the three chain pieces <code>dL/da=2(a−y)</code>, <code>da/dz=a(1−a)</code>, <code>dz/dw=x</code> (and <code>dz/db=1</code>), print them, and print their products <code>dL/dw ≈ −0.212</code> and <code>dL/db ≈ −0.106</code>. Then verify each analytic gradient against a central finite difference <code>(L(θ+h)−L(θ−h))/(2h)</code> with <code>h=1e-5</code> and confirm they agree to better than <code>1e-6</code>. Run with <code>python3 sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 2 Day 5 artifact.

Create sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py that, with a comment on each step:
1. Defines f(w)=w**2 (a simple loss bowl).
2. Computes a numerical gradient at w=3 via (f(w+h)-f(w-h))/(2h) with h=1e-5; prints it (expect ~6.0) and notes the exact slope is 2w.
3. Runs gradient descent: w=3.0; for 10 steps do grad=2*w; w = w - 0.1*grad; print round(w,3), round(w**2,3).
4. Confirms w marches toward 0 and the loss w**2 falls toward 0.
5. Part (b): verify backprop by hand on one sigmoid neuron with squared error — z=w*x+b, a=sigmoid(z), L=(a-y)**2, with x=2, w=0.5, b=0, y=1.
6. Prints the chain pieces dL/da=2*(a-y), da/dz=a*(1-a), dz/dw=x, dz/db=1; then dL/dw = dL/da*da/dz*dz/dw (expect about -0.212) and dL/db = dL/da*da/dz*dz/db (expect about -0.106).
7. Checks both analytic gradients against central finite differences (L(theta+h)-L(theta-h))/(2h) with h=1e-5 and asserts they agree to < 1e-6.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>What you should see (check your prediction)</h4>
      <ul>
        <li>For <code>f(w)=w²</code> your numerical slope at <code>w=3</code> comes out ≈ 6 (matching <code>2w</code>), and your 10-step loop with <code>w = w − 0.1·(2w)</code> drives <code>w</code> and the loss <code>w²</code> toward 0.</li>
        <li><b>Backprop check:</b> for the sigmoid neuron with <code>x=2, w=0.5, b=0, y=1</code>, your printed chain pieces multiply to <code>dL/dw ≈ −0.212</code> and <code>dL/db ≈ −0.106</code> — the same numbers as the Math Ladder above.</li>
        <li><b>Finite-difference proof:</b> the analytic <code>dL/dw</code> and <code>dL/db</code> each match a central finite-difference estimate to better than <code>1e-6</code>, showing backprop is just the chain rule computed exactly.</li>
        <li>You can state in one sentence what the <em>sign</em> of the gradient tells you and what the learning rate controls.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-05-gradients-backprop/log.md</code>: (1) why backprop is "just the chain rule run backward"; (2) what the finite-difference check proves about a gradient you derived by hand; (3) what "vanishing gradients" means and how you'd detect it.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 2 · Day 5 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Gradients &amp; Backpropagation</b>.<br>Next up: <b>The Training Loop</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  slope:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">def</span> <span class="fn">f</span>(w): <span class="kw">return</span> w**<span class="num">2</span>     <span class="dim"># a simple loss (a bowl)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w, h = <span class="num">3.0</span>, <span class="num">1e-5</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> (f(w+h) - f(w-h)) / (<span class="num">2</span>*h)   <span class="dim"># slope = gradient at w=3</span>\n'+
    '<span class="hl">6.0000000...</span>',
    take:'<b>①  The gradient is the slope.</b> For f(w)=w², the slope at w=3 is 6 (exactly 2w). It says the loss rises 6× as fast as w — so to lower the loss, move w the OTHER way.'},
  step:{html:'<span class="prompt">&gt;&gt;&gt;</span> w, lr, grad = <span class="num">3.0</span>, <span class="num">0.1</span>, <span class="num">6.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = w - lr * grad     <span class="dim"># gradient-descent step: go downhill</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w\n<span class="hl">2.4</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="num">3.0</span>**<span class="num">2</span>, <span class="num">2.4</span>**<span class="num">2</span>   <span class="dim"># loss before vs after</span>\n'+
    '<span class="ok">(9.0, 5.76)</span>',
    take:'<b>②  One step downhill.</b> Subtract lr×gradient: w goes 3.0 → 2.4, and the loss falls 9.0 → 5.76. The learning rate (0.1) is how big a step you take.'},
  descend:{html:'<span class="prompt">&gt;&gt;&gt;</span> w = <span class="num">3.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">for</span> _ <span class="kw">in</span> range(<span class="num">5</span>):\n'+
    '<span class="dim">...</span>     grad = <span class="num">2</span>*w\n'+
    '<span class="dim">...</span>     w = w - <span class="num">0.1</span>*grad\n'+
    '<span class="dim">...</span>     print(round(w,<span class="num">3</span>), round(w**<span class="num">2</span>,<span class="num">3</span>))\n'+
    '<span class="ok">2.4   5.76</span>\n<span class="ok">1.92  3.686</span>\n<span class="ok">1.536 2.359</span>\n<span class="ok">1.229 1.51</span>\n<span class="ok">0.983 0.966</span>',
    take:'<b>③  Repeat → roll to the bottom.</b> Each step re-measures the slope and moves downhill; w marches toward 0, where the loss is smallest. That is training: many small gradient-guided steps.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='The loss as a bowl-shaped curve over one weight'><g font-family='monospace'><line x1='60' y1='110' x2='460' y2='110' stroke='#E5DFD6' stroke-width='1.5'/><path d='M90 40 Q 260 160 430 40' fill='none' stroke='#2A7B9B' stroke-width='2.5'/><text x='260' y='125' font-size='11' fill='#6B645E' text-anchor='middle'>weight w →</text><text x='40' y='70' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 70)'>loss</text><circle cx='260' cy='108' r='4' fill='#2D8B55'/><text x='260' y='100' font-size='10' fill='#2D8B55' text-anchor='middle'>lowest loss</text></g></svg>",
  note:"<b>Picture the loss over one weight.</b> Change the weight and the loss traces a curve — here a bowl. Training wants to reach the bottom. But the network can't see the whole curve; it only stands at one point."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='The slope of the loss at the current weight is the gradient'><g font-family='monospace'><line x1='60' y1='110' x2='460' y2='110' stroke='#E5DFD6' stroke-width='1.5'/><path d='M90 40 Q 260 160 430 40' fill='none' stroke='#2A7B9B' stroke-width='2'/><circle cx='350' cy='83' r='5' fill='#C93B3B'/><line x1='300' y1='108' x2='400' y2='58' stroke='#C93B3B' stroke-width='2.5'/><text x='360' y='45' font-size='11' fill='#C93B3B'>slope here = gradient</text><text x='350' y='104' font-size='10' fill='#6B645E' text-anchor='middle'>current w</text></g></svg>",
  note:"<b>Measure the slope where you stand: the gradient.</b> It tells you the loss is rising to the right here — so downhill is to the <b>left</b>. The gradient is just this slope, per weight."},
 {viz:"<svg viewBox='0 0 520 100' role='img' aria-label='Gradient descent update rule: new weight equals old weight minus learning rate times gradient'><g font-family='monospace' font-size='14' text-anchor='middle'><rect x='90' y='34' width='340' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='56' fill='#1a5c38' font-weight='bold'>w ← w − learning_rate × gradient</text><text x='260' y='90' font-size='11' fill='#6B645E'>subtract a little of the slope → move downhill</text></g></svg>",
  note:"<b>Step downhill.</b> The update rule: <code>w = w − lr × gradient</code>. Subtracting the gradient moves against the uphill direction, i.e. downhill. The learning rate <code>lr</code> sets how far you step."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Repeated steps roll the weight down to the bottom of the bowl'><g font-family='monospace'><line x1='60' y1='110' x2='460' y2='110' stroke='#E5DFD6' stroke-width='1.5'/><path d='M90 40 Q 260 160 430 40' fill='none' stroke='#2A7B9B' stroke-width='2'/><circle cx='350' cy='83' r='4' fill='#C99A12'/><circle cx='315' cy='98' r='4' fill='#C99A12'/><circle cx='288' cy='105' r='4' fill='#C99A12'/><circle cx='262' cy='108' r='5' fill='#2D8B55'/><text x='300' y='60' font-size='11' fill='#9A7208'>step, step, step…</text><text x='262' y='100' font-size='10' fill='#2D8B55' text-anchor='middle'>bottom</text></g></svg>",
  note:"<b>Repeat and you roll to the bottom.</b> Each step re-measures the slope and moves again. After many small steps the weight settles near the lowest loss. That is gradient descent."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='The chain rule multiplies local slopes backward through the layers'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='40' width='70' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='75' y='60' fill='#1F6280'>layer 1</text><rect x='150' y='40' width='70' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='185' y='60' fill='#1F6280'>layer 2</text><rect x='260' y='40' width='70' height='30' rx='5' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='295' y='60' fill='#C93B3B'>loss</text><path d='M330 78 L150 78' stroke='#9A7208' stroke-width='2' marker-end='url(#ar)'/><text x='240' y='96' fill='#9A7208' font-size='11'>multiply local slopes, going backward</text><defs><marker id='ar' markerWidth='7' markerHeight='7' refX='5' refY='3' orient='auto'><path d='M0 0 L6 3 L0 6 z' fill='#9A7208'/></marker></defs></g></svg>",
  note:"<b>The chain rule, backward.</b> A deep weight touches the loss only through the layers after it. The chain rule multiplies each step's local slope together — and backprop does this from the loss backward, so every weight's gradient falls out in one sweep."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Training = forward pass then backward pass, repeated'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='30' y='40' width='150' height='32' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='105' y='61' fill='#1F6280'>forward → loss</text><rect x='200' y='40' width='170' height='32' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='285' y='61' fill='#9A7208'>backward → gradients</text><rect x='390' y='40' width='110' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='445' y='61' fill='#1a5c38'>update w</text><text x='260' y='96' fill='#6B645E'>…then repeat. This loop is training.</text></g></svg>",
  note:"<b>Why it runs all of AI.</b> Every training step is: forward pass → loss → backprop (gradients) → update the weights → repeat. Backprop costs roughly twice a forward pass, so one training step is ~3× a forward pass (one forward + two-forward-equivalents backward). Frameworks automate it (<code>jax.grad</code>), but it's just chain-rule-backward. Tomorrow we wrap it in the training loop."}
];
@@@ region name=QS
var QS=[
 {q:'1. A gradient tells you:',
  opts:['how many layers the model has','the slope of the loss w.r.t. a weight — which way (and how much) to change it to cut the loss','the batch size','the final prediction'],
  ans:1, fb:'Right. The gradient is the loss\'s slope for a weight; it points uphill, so you move the opposite way to reduce the loss.'},
 {q:'2. Gradient descent updates a weight by:',
  opts:['w = w + gradient','w = w − learning_rate × gradient','w = gradient','w = w × learning_rate'],
  ans:1, fb:'Right. Subtract a small fraction (the learning rate) of the gradient — a step downhill.'},
 {q:'3. Backpropagation is based on which calculus rule?',
  opts:['the product rule','the chain rule','the quotient rule','the power rule'],
  ans:1, fb:'Right. The chain rule multiplies local slopes along the path; backprop applies it backward from the loss.'},
 {q:'4. You train a deep network with sigmoid activations. It runs with no error, but the loss barely moves and the early layers\' weights hardly change while the last layers do learn. You log the gradient size at each layer and see the early-layer gradients are tiny. What is happening, and why?',
  opts:['The learning rate is exactly zero, which is the only way weights stop changing','Vanishing gradients: the chain rule multiplies many small (<1) local slopes together, so the gradient shrinks toward 0 by the time it reaches the early layers, and they barely update','The batch size is too large, which always freezes the early layers','Early layers are supposed to never change, so this is normal'],
  ans:1, fb:'Right. Backprop multiplies one local slope per layer. Sigmoid slopes are less than 1, so a product of many of them collapses toward 0 for the early layers — vanishing gradients. Training stalls with no crash. You look at per-layer gradient norms, which is exactly why ReLU and residual connections were introduced.'}
];
