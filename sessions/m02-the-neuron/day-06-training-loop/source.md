---
quest_id: wf3-d03-training-loop
donor: m02-day-06.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 2 · Day 6 — The Training Loop"
spine: "free-throw"
notebook_yardstick: 00-neural-networks/fundamentals/08_training_loop.ipynb
---

@@@ region name=title
<title>Module 2 · Day 6 — The Training Loop</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M2 Day 6</div>
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
<a class="lnav prev" href="../day-05-gradients-backprop/lesson.html"><span class="d">← Prev</span><span class="t">Gradients &amp; Backpropagation</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-07-optimizers/lesson.html"><span class="d">Next →</span><span class="t">Optimizers (SGD, Momentum, Adam)</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 2 · Train · Day 6</span>
      <h1>The Training Loop<span class="sub">Where the Network Actually Learns</span></h1>
      <p class="lede">This is the day it all comes alive. The forward pass, the loss, and the gradients you built stop being separate ideas and click into one short cycle — the <strong>training loop</strong> — that a network runs over and over until it gets good. Scaled up, this exact loop is how every model you've heard of was trained.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>By the time you close this tab, "training" will just be a loop you can trace: you'll list its four steps, run it by hand to watch a loss fall, and know what an epoch, a batch, and "overfit one batch" mean.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">The loop that turns pieces into learning</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 3 (forward pass), Day 4 (loss), Day 5 (gradients, backprop, gradient descent). Today just assembles them. <b>Nothing new needed.</b></div></div>

      <h4>First, the picture — hold this before the steps</h4>
      <p>Think about practicing free throws in basketball. You take a shot, you see how far you missed, you work out what to adjust, and you shoot again — and after enough shoot-check-adjust cycles, you get good. A network learns in exactly that rhythm: it makes a guess, measures how wrong it was, works out which way to nudge its weights, and tries again. Keep that picture; the four steps below are just that one rhythm written down precisely.</p>

      <h4>What is the training loop?</h4>
      <p>The <span class="term" data-tip="The repeating cycle that trains a model: forward pass → loss → backprop (gradients) → update weights → repeat.">training loop</span> is the four-step cycle repeated many times:</p>
      <ol>
        <li><strong>Forward pass</strong> — run the input through the network to get a prediction (Day 3).</li>
        <li><strong>Loss</strong> — measure how wrong it is, one number (Day 4).</li>
        <li><strong>Backprop</strong> — compute the gradient for every weight (Day 5).</li>
        <li><strong>Update</strong> — nudge each weight downhill: <code>w ← w − lr × grad</code>.</li>
      </ol>
      <p>Then repeat. Each pass makes the prediction a little better.</p>

      <h4>Two words you'll see</h4>
      <p>An <span class="term" data-tip="One full pass over the entire training dataset.">epoch</span> is one full pass over all the training data. A <span class="term" data-tip="A small chunk of the dataset processed together in one step, so you update weights often without loading everything at once.">batch</span> is a chunk processed in one step. Real training loops over batches, and over epochs. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Meta has reported training Llama 3's biggest model on more than 15 trillion tokens — the exact loop you're about to run, just repeated an enormous number of times.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> 前几天学了 forward、loss、gradient，但很多人还是不知道它们怎么"合"成训练 — you've met the forward pass, loss, and gradients, yet it's easy to still not see how they fit together. In plain terms: the training loop is just those pieces in a cycle — predict, score, find the slope, nudge the weights — run over and over. There is no hidden fifth step.</div></div>
      <button class="gotit" type="button">Got the loop — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">Practicing free throws</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>You already have the free-throw picture from Section 1 — here it is as two cards, one for the single shoot-check-adjust cycle and one for why the repetition is the whole point:</p>
      <div class="relate">
        <div class="card"><span class="big">🏀</span><h5>Shoot, check, adjust</h5><p>Take a shot (forward pass), see how far you missed (loss), work out what to fix (gradients), adjust your form (update). Then shoot again.</p></div>
        <div class="card"><span class="big">🔁</span><h5>Repetition is the training</h5><p>One shot barely improves you. Thousands of shoot-check-adjust cycles do. Same for a network: the loop, run many times, is the learning.</p></div>
      </div>
      <p><strong>In one line:</strong> the training loop is shoot-check-adjust for weights, repeated until the loss is low.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a network can "memorize" the practice set and still miss new shots — that's <span class="term" data-tip="When a model fits the training data well but fails on new, unseen data. Checked by evaluating on held-out data.">overfitting</span>. So we also check the loss on data the model has not trained on. More on that in the evaluation modules.</div></div>
      <p><b>直觉 / Intuition:</b> 直觉上，training loop 就是"投篮 → 看偏了多少 → 调整 → 再投"，一遍遍重复直到稳定。The technical point: each iteration runs forward → loss → backprop → weight update, and repeating it drives the loss down — the whole of "training" is this one cycle, scaled up.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The loop in code, and why it IS frontier training</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The loop, in plain code</h4>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>for epoch in range(E):</code><br>&nbsp;&nbsp;<code>for batch in data:</code><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>reset_grads()</code>&nbsp;&nbsp;<span style="color:var(--muted)"># frameworks accumulate — clear last step's grads (PyTorch: optimizer.zero_grad())</span><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>pred = forward(batch.x)</code><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>loss = loss_fn(pred, batch.y)</code><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>grads = backprop(loss)</code><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>weights = weights − lr × grads</code></div></div>
      <div class="callout c-info"><span class="ic">💡</span><div><b>Where do the weights start?</b> Before the first pass the weights are small <b>random</b> numbers — never all-equal (the symmetry trap from Day 1), and scaled to the layer's fan-in so the pre-activations don't start too big or too small (the named recipes are Xavier and He). The loop does all the rest.</div></div>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — one training step of <code>pred = w·x</code></b>
      <br><b>1 · In words:</b> predict, measure how wrong, find the slope of the loss w.r.t. <code>w</code>, then nudge <code>w</code> against that slope. Repeat.
      <br><b>2 · The formula:</b> <code>pred = w·x</code>; <code>loss = (pred − y)²</code>; <code>grad = 2(pred − y)·x</code>; <code>w ← w − lr·grad</code>. Here <code>x=2, y=6</code>, start <code>w=1</code>, <code>lr=0.05</code>.
      <br><b>3 · Tiny numbers:</b> step 1 — <code>pred = 2</code>, <code>loss = (2−6)² = 16</code>, <code>grad = 2(−4)(2) = −16</code>, <code>w ← 1 − 0.05(−16) = 1.8</code>. Next step the loss is smaller and <code>w</code> keeps climbing toward the true answer <code>3</code>.
      <br><b>4 · Sanity check:</b> at <code>w=3</code>, <code>pred = 6 = y</code>, so <code>loss = 0</code> and <code>grad = 0</code> — the update stops moving. A loss that <em>rises</em> instead means <code>lr</code> is too big.</div></div>

      <h4>A sanity check you'll use forever</h4>
      <p>Before training for real, run the loop on a <strong>single batch</strong> until the loss hits ~0. If it can't, your gradients aren't flowing (a bug). "Overfit one batch" is the first thing engineers do to prove a new model can learn at all — you'll see it in the JAX modules.</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Epoch, step, batch — with a number.</b> A <b>batch</b> is the chunk of examples used for one weight update; a <b>step</b> (iteration) is one such update; an <b>epoch</b> is one full pass over the data. With 100 examples and batch size 25, one epoch = <code>100/25 = 4</code> steps. The batch's gradient is the <b>mean of the per-example gradients</b>, so a batch gives a <em>noisy but unbiased</em> estimate of the true whole-dataset gradient — a smaller batch means a noisier estimate. In a real loop you also measure the loss on a held-out slice each epoch (the <b>validation</b> loss) — that's the signal Day 9 uses to catch overfitting.</div></div>

      <h4>Why a frontier lab cares</h4>
      <p>This exact loop, scaled up, <em>is</em> how frontier models are trained — that's the Llama 3 run from the teaser above, just repeated an almost unimaginable number of times. How big to make each repetition is exactly the trade-off you're about to meet below: the batch size. Tuning the <span class="term" data-tip="Settings you choose, not learn: learning rate, batch size, number of epochs. They strongly affect training.">hyperparameters</span> — learning rate, batch size — is much of the craft.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> forward → loss → backprop → update, repeated over batches and epochs = training. One batch to near-zero loss proves it learns; scaled to a datacenter, it's how every model you've heard of was made. The core loop is now assembled — the next three days refine <em>how</em> it steps (optimizers Day 7, learning rate Day 8) and whether it truly learned (generalization Day 9).</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>The loop is only four lines, and a missing fifth one is the most common quiet bug in all of training.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): forgetting to reset the gradients each step.</b> In common frameworks (like PyTorch), gradients <em>add up</em> across steps by default — each backward pass piles its gradient on top of the last one instead of replacing it. If you forget the reset (in PyTorch, <code>optimizer.zero_grad()</code>) at the top of the loop, the update uses a growing sum of gradients from every past batch. On the free-throw court, that is like letting the feedback from every shot you have ever taken pile on top of this one's — you keep over-correcting for misses you already fixed. The loop still runs and the loss still changes, so nothing crashes — but training behaves strangely and learns poorly for no obvious reason. A senior engineer catches this in <b>code review</b> by checking that the loop clears the gradients before each backward pass.</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: how big to make each batch.</b> A <span class="term" data-tip="The number of examples processed together in one training step before the weights are updated.">batch</span> is how many examples you process before one weight update. Big batches give a smoother, more reliable gradient estimate and use the hardware efficiently, but need much more memory and sometimes generalize slightly worse. Small batches are noisy and slower per example on the hardware, but use little memory and the noise can even help the model escape bad spots. Choosing the batch size is a <b>design-review</b> decision that trades memory and hardware efficiency against gradient quality.</div></div>
      <table style="width:100%;border-collapse:collapse;margin:14px 0;font-size:.88rem"><thead><tr style="text-align:left;border-bottom:2px solid var(--line2)"><th style="padding:.5rem .6rem"></th><th style="padding:.5rem .6rem;color:var(--q)">Large batch</th><th style="padding:.5rem .6rem;color:var(--v)">Small batch</th></tr></thead><tbody>
        <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Gradient estimate</td><td style="padding:.5rem .6rem;color:var(--ink2)">smoother, lower-noise</td><td style="padding:.5rem .6rem;color:var(--ink2)">noisier</td></tr>
        <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Hardware use</td><td style="padding:.5rem .6rem;color:var(--ink2)">efficient (high throughput)</td><td style="padding:.5rem .6rem;color:var(--ink2)">underutilized</td></tr>
        <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Memory</td><td style="padding:.5rem .6rem;color:var(--ink2)">high</td><td style="padding:.5rem .6rem;color:var(--ink2)">low</td></tr>
        <tr><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Generalization</td><td style="padding:.5rem .6rem;color:var(--ink2)">can be slightly worse; needs LR retuning</td><td style="padding:.5rem .6rem;color:var(--ink2)">noise can help escape bad spots</td></tr>
      </tbody></table>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "The training loop is four steps repeated over batches and epochs: forward pass → loss → backprop for gradients → weight update <code>w ← w − lr·grad</code>. That cycle, scaled to a datacenter, is how every model is trained. My first move on any new model is 'overfit one batch' — run the loop on a single batch until the loss hits ~0; if it can't, gradients aren't flowing and there's a bug. The main knob is batch size, trading gradient quality and hardware efficiency against memory."</div></div>
      <button class="gotit" type="button">Got the loop</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Predict where it lands, then run the loop</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Here's the fun part. <b>Before you run anything, predict:</b> after 20 shoot-check-adjust steps, where does <code>w</code> end up and what happens to the loss? Write your guess. Then run the loop and watch <code>w</code> climb toward <code>3</code> while the loss collapses toward <code>0</code> — and notice that the <em>first</em> step moves <code>w</code> the most (gradient <code>−16</code>) while later steps barely nudge it. That slowdown you see <em>is</em> the loss curve flattening out. Pick one path:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m02-the-neuron/day-06-training-loop/experiment.py</code>. Fit <code>pred = w·x</code> to <code>x=2, y=6</code>. Start <code>w=1.0</code>; loop 20 steps doing: <code>pred=w*x</code>; <code>loss=(pred-y)**2</code>; <code>grad=2*(pred-y)*x</code>; <code>w=w-0.05*grad</code>; print step, w, loss. Confirm <code>w → 3</code> and <code>loss → 0</code>. Run with <code>python3 sessions/m02-the-neuron/day-06-training-loop/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 2 Day 6 artifact.

Create sessions/m02-the-neuron/day-06-training-loop/experiment.py that, with a comment on each step:
1. Sets a tiny model pred = w*x, with x=2.0, target y=6.0 (ideal w=3).
2. Runs a training loop: w=1.0; for 20 steps do pred=w*x; loss=(pred-y)**2; grad=2*(pred-y)*x; w = w - 0.05*grad; print step, round(w,4), round(loss,4).
3. Confirms w converges toward 3.0 and loss toward 0 — the four-step loop (forward, loss, gradient, update) in action.
4. Adds a comment explaining that scaling this loop to many weights, batches, and chips is how real models train.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>What you should see (check your prediction)</h4>
      <ul>
        <li>Your loop prints <b>labeled stages</b> — the step number, <code>w</code>, and <code>loss</code> — at every iteration, so each of the four loop steps (forward → loss → backprop → update) is visible in the output.</li>
        <li>Your loop prints step, <code>w</code>, and <code>loss</code> each iteration for the <code>pred=w·x</code>, <code>x=2, y=6</code> problem.</li>
        <li>Over ~20 steps <code>w → 3</code> and <code>loss → 0</code>, with the loss decreasing (no blow-up).</li>
        <li>You can name which of the four loop steps maps to which earlier lesson (forward, loss, backprop, update).</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-06-training-loop/log.md</code>: (1) the four steps of the loop in order; (2) what "overfit one batch" proves and why you'd run it first; (3) what happens to the loss if the learning rate is too big.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 2 · Day 6 complete! 🏆</h3>
      <p>Nice work — you've completed <b>The Training Loop</b>.<br>Next up: <b>Optimizers (SGD, Momentum, Adam)</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  onestep:{html:'<span class="prompt">&gt;&gt;&gt;</span> x, y = <span class="num">2.0</span>, <span class="num">6.0</span>      <span class="dim"># want pred = w*x to equal 6</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = <span class="num">1.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> pred = w * x            <span class="dim"># 1) forward → 2.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> loss = (pred - y)**<span class="num">2</span>     <span class="dim"># 2) loss → 16.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> grad = <span class="num">2</span>*(pred - y)*x     <span class="dim"># 3) gradient → -16.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = w - <span class="num">0.05</span>*grad       <span class="dim"># 4) update → 1.8</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w, (w*x - y)**<span class="num">2</span>          <span class="dim"># new w, new loss</span>\n'+
    '<span class="ok">(1.8, 5.76)</span>',
    take:'<b>①  One full loop step.</b> forward (2.0) → loss (16) → gradient (−16) → update (w: 1.0 → 1.8). The loss fell 16 → 5.76. All four steps, once.'},
  loop:{html:'<span class="prompt">&gt;&gt;&gt;</span> w = <span class="num">1.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">for</span> step <span class="kw">in</span> range(<span class="num">5</span>):\n'+
    '<span class="dim">...</span>     pred = w*x\n'+
    '<span class="dim">...</span>     grad = <span class="num">2</span>*(pred-y)*x\n'+
    '<span class="dim">...</span>     w = w - <span class="num">0.05</span>*grad\n'+
    '<span class="dim">...</span>     print(step, round(w,<span class="num">3</span>), round((w*x-y)**<span class="num">2</span>,<span class="num">3</span>))\n'+
    '<span class="ok">0 1.8   5.76</span>\n<span class="ok">1 2.28  2.074</span>\n<span class="ok">2 2.568 0.746</span>\n<span class="ok">3 2.741 0.269</span>\n<span class="ok">4 2.844 0.097</span>',
    take:'<b>②  Repeat → it learns.</b> w climbs toward the ideal 3.0 and the loss falls toward 0. The model is fitting the data, one gradient step at a time.'},
  epochs:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">for</span> epoch <span class="kw">in</span> range(num_epochs):     <span class="dim"># one epoch = one full pass over the data</span>\n'+
    '<span class="dim">...</span>     <span class="kw">for</span> batch <span class="kw">in</span> data:            <span class="dim"># process a chunk at a time</span>\n'+
    '<span class="dim">...</span>         pred  = forward(batch.x)\n'+
    '<span class="dim">...</span>         loss  = loss_fn(pred, batch.y)\n'+
    '<span class="dim">...</span>         grads = backprop(loss)      <span class="dim"># gradients for every weight</span>\n'+
    '<span class="dim">...</span>         weights = weights - lr*grads  <span class="dim"># update</span>',
    take:'<b>③  Real training, same four steps.</b> Wrap them in two loops: over epochs (full passes through the data) and over batches (chunks). Scale to many chips and trillions of tokens → frontier training.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 150' role='img' aria-label='The training loop as a four-step cycle: forward, loss, backprop, update'><g font-family='monospace' font-size='11.5' text-anchor='middle'><rect x='200' y='16' width='120' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='34' fill='#1F6280'>1 · forward</text><rect x='360' y='58' width='120' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='420' y='76' fill='#C93B3B'>2 · loss</text><rect x='200' y='104' width='120' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='122' fill='#9A7208'>3 · backprop</text><rect x='40' y='58' width='120' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='100' y='76' fill='#1a5c38'>4 · update</text><path d='M320 30 Q 400 40 405 56' fill='none' stroke='#6B645E' stroke-width='1.5' marker-end='url(#a)'/><path d='M400 86 Q 360 110 322 116' fill='none' stroke='#6B645E' stroke-width='1.5' marker-end='url(#a)'/><path d='M200 118 Q 120 110 118 88' fill='none' stroke='#6B645E' stroke-width='1.5' marker-end='url(#a)'/><path d='M110 56 Q 150 30 198 28' fill='none' stroke='#6B645E' stroke-width='1.5' marker-end='url(#a)'/><defs><marker id='a' markerWidth='7' markerHeight='7' refX='5' refY='3' orient='auto'><path d='M0 0 L6 3 L0 6 z' fill='#6B645E'/></marker></defs></g></svg>",
  note:"<b>The loop is a four-step cycle.</b> Forward pass → loss → backprop → update, then back to the top. Each piece is a lesson you've already done; here they connect into a circle that runs over and over."},
 {viz:"<svg viewBox='0 0 520 100' role='img' aria-label='One step lowers the loss from 16 to 5.76'><g font-family='monospace' font-size='13' text-anchor='middle'><rect x='90' y='36' width='120' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='150' y='56' fill='#C93B3B'>loss 16.0</text><text x='260' y='56' fill='#6B645E'>→ one step →</text><rect x='340' y='36' width='120' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='400' y='56' fill='#9A7208'>loss 5.76</text></g></svg>",
  note:"<b>Run it once.</b> A single trip round the loop already lowers the loss (16 → 5.76 in our toy). The weight moved a little toward the value that fits the data."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Repeating the loop drives the loss down toward zero'><g font-family='monospace'><line x1='60' y1='105' x2='460' y2='105' stroke='#E5DFD6' stroke-width='1.5'/><path d='M75 35 C 160 60, 240 95, 440 100' fill='none' stroke='#2D8B55' stroke-width='2.5'/><text x='250' y='122' font-size='11' fill='#6B645E' text-anchor='middle'>training steps →</text><text x='40' y='65' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 65)'>loss</text><text x='400' y='60' font-size='12' fill='#1a5c38' font-family='monospace'>→ 0</text></g></svg>",
  note:"<b>Repeat and it learns.</b> Many steps drive the loss down toward 0; the weight settles at the value that fits the data. Repetition is the whole trick — one step barely helps, thousands do."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='A batch is a chunk of data; an epoch is one full pass over all of it'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='45' width='40' height='26' rx='4' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='84' y='45' width='40' height='26' rx='4' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='128' y='45' width='40' height='26' rx='4' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='172' y='45' width='40' height='26' rx='4' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='60' y='90' font-size='10' fill='#6B645E'>batch</text><text x='300' y='62' fill='#2C2A28'>4 batches = 1 epoch (one full pass)</text></g></svg>",
  note:"<b>Batches and epochs.</b> The data is split into <b>batches</b> (chunks); one step processes one batch. One full pass over every batch is an <b>epoch</b>. Real training runs many epochs."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Overfit one batch until loss reaches zero as a sanity check'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='90' y='40' width='150' height='32' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='165' y='61' fill='#1F6280'>loop on ONE batch</text><text x='265' y='61' fill='#6B645E'>→</text><rect x='300' y='40' width='150' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='375' y='61' fill='#1a5c38'>loss ≈ 0 ✓</text><text x='260' y='96' fill='#6B645E'>proves gradients flow and the model can learn</text></g></svg>",
  note:"<b>The engineer's first check: overfit one batch.</b> Run the loop on a single batch until the loss hits ~0. If it can't, gradients aren't flowing — a bug. This is the very first thing you do with a new model (you'll see it in the JAX modules)."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='The same loop scaled to a datacenter trains frontier models'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='42' width='150' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='115' y='62' fill='#1F6280'>same 4-step loop</text><text x='210' y='62' fill='#6B645E'>×</text><rect x='240' y='42' width='240' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='360' y='62' fill='#9A7208'>1000s of chips · trillions of tokens</text><text x='260' y='96' fill='#2C2A28'>= how every frontier model is trained</text></g></svg>",
  note:"<b>Why it runs the field.</b> This exact loop, scaled to thousands of accelerators and trillions of tokens, is how frontier models are trained — and the scaling laws (M1 Day 5) predict how the loss falls as you add compute. You've now assembled the core engine: represent → forward → loss → gradients → loop. Still ahead: sharper ways to take the step (optimizers, learning rate) and how to tell it truly learned (Days 7–9)."}
];
@@@ region name=QS
var QS=[
 {q:'1. The training loop\'s four core steps, in order?',
  opts:['loss, update, forward, backprop','forward pass, loss, backprop, update','update, forward, loss, backprop','backprop, forward, update, loss'],
  ans:1, fb:'Right. Forward → loss → backprop (gradients) → update weights, then repeat.'},
 {q:'2. What is one epoch?',
  opts:['one weight update','one full pass over the training data','one layer','one batch'],
  ans:1, fb:'Right. An epoch = one complete pass over all the training data (usually many batches).'},
 {q:'3. Your training loop runs with no error, but the loss jumps around wildly and the model learns badly. You are using PyTorch, and you notice the loop never calls <code>optimizer.zero_grad()</code>. What is the most likely cause?',
  opts:['The learning rate is negative, which is the only thing that makes loss unstable','Gradients accumulate across steps by default, so without resetting them each step the update uses a growing sum of every past batch\'s gradient — the loop runs fine but training misbehaves','The data has too few examples, which always makes loss jump','A jumping loss with no error means training is finished'],
  ans:1, fb:'Right. In frameworks like PyTorch, each backward pass adds to the existing gradients instead of replacing them. Forgetting to zero them means the update uses a stale, growing sum. Nothing crashes — you look at whether the loop clears gradients before each backward pass.'},
 {q:'4. "Overfit a single batch" is a sanity check that:',
  opts:['the data is large enough','gradients flow and the model can actually learn','the GPU is fast','the loss is cross-entropy'],
  ans:1, fb:'Right. If the loop can drive one batch\'s loss to ~0, gradients are flowing and the model can learn. If not, there\'s a bug.'}
];
