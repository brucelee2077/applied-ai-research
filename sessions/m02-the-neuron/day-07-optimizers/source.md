---
quest_id: wf3-d04-optimizers
donor: m02-day-07.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 2 · Day 7 — Optimizers: SGD → Momentum → Adam"
spine: "rolling ball"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 2 · Day 7 — Optimizers: SGD → Momentum → Adam</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M2 Day 7</div>
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
<a class="lnav prev" href="../day-06-training-loop/lesson.html"><span class="d">← Prev</span><span class="t">The Training Loop</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-08-learning-rate/lesson.html"><span class="d">Next →</span><span class="t">Learning-Rate Intuition</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 2 · Train · Day 7</span>
      <h1>Optimizers: SGD → Momentum → Adam<span class="sub">Smarter ways to step downhill</span></h1>
      <p class="lede">You already know how to step downhill — but a plain step is slow and shaky. Today three step-takers race for the same valley floor: <strong>SGD</strong> walks one careful step at a time, <strong>Momentum</strong> rolls like a real ball that coasts through bumps, and <strong>Adam</strong> rolls with a per-weight sense of the ground. Same destination — very different speed getting there.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>By the time you close this tab, you'll be able to write the SGD, momentum, and Adam update rules, step SGD and momentum by hand on the same weight to see momentum pull ahead, and say why Adam costs about twice the optimizer memory.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">The rule that decides how to step</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> the training loop (Day 6), gradients and gradient descent (Day 5). Today we only change <b>step 4 — the update</b>. <b>Nothing new needed.</b></div></div>

      <h4>First, the picture — hold this before the rules</h4>
      <p>Picture the loss as a hilly valley you're trying to reach the bottom of. You could inch down one careful step at a time — or you could let a ball roll and coast through the little dips instead of stopping at each one. That is the whole story of today: three ways of moving downhill, from a careful walk to a smart rolling ball. Keep that picture; the definitions below just give each mover its name.</p>

      <h4>What is an optimizer?</h4>
      <p>An <span class="term" data-tip="The rule that turns gradients into a weight change each step: SGD, momentum, and Adam are the common ones.">optimizer</span> is the rule that decides how to change the weights once you have the gradients. The forward pass, the loss, and backprop stay the same. Only the last step — the update — is what an optimizer controls.</p>

      <h4>Three optimizers, from simple to smart</h4>
      <ol>
        <li><strong><span class="term" data-tip="Stochastic Gradient Descent: the plain update rule w ← w − lr × grad, using the gradient from one batch.">SGD</span></strong> — step straight downhill by <code>lr × grad</code>. Simple and honest, but it can be slow and shaky.</li>
        <li><strong><span class="term" data-tip="An optimizer that keeps a running velocity: it adds a fraction of the last step to the new one, so the weight rolls through small bumps.">Momentum</span></strong> — keep a running speed (velocity). Add a bit of the last step to the new one, so the weight builds up speed in a steady direction.</li>
        <li><strong><span class="term" data-tip="Adaptive Moment Estimation: an optimizer that keeps a running average of the gradient (m) and of the gradient squared (v), giving each weight its own step size.">Adam</span></strong> — give every weight its own step size, using running averages of the gradient and of the gradient squared.</li>
      </ol>
      <p>They all move downhill. The smarter ones just choose the step more cleverly. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Microsoft's ZeRO, used inside the DeepSpeed library, exists to shard Adam's doubled memory cost across many GPUs — a direct fix for the "two extra numbers" you're about to meet.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> 一看到 SGD、Momentum、Adam 三个名字，很容易以为它们是三种完全不同的算法 — three names (SGD, Momentum, Adam) make it look like three unrelated algorithms. In plain terms: all three do the same job — step the weights downhill — and each just remembers a little more about the past to steer better. Adam is momentum plus a per-weight step size, nothing more mysterious.</div></div>
      <button class="gotit" type="button">Got it — let's meet them</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">Rolling a ball downhill, three ways</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>You already have the rolling-ball picture from Section 1 — here are the three movers side by side, each a different way down the same valley:</p>
      <div class="relate">
        <div class="card"><span class="big">🐾</span><h5>SGD — tiny careful steps</h5><p>You take one small step in the steepest downhill direction, stop, look again, step again. Safe, but slow, and you stall in every little dip.</p></div>
        <div class="card"><span class="big">⚽</span><h5>Momentum — a real ball</h5><p>Now let a ball roll. It builds up speed in the downhill direction, so a tiny bump does not stop it — it coasts right over.</p></div>
      </div>
      <div class="relate">
        <div class="card"><span class="big">🤖</span><h5>Adam — a smart ball</h5><p>The ball watches the ground. Where the slope has been steady, it takes bigger, confident steps. Where the slope keeps flipping, it takes smaller, careful ones.</p></div>
        <div class="card"><span class="big">🎯</span><h5>Same goal, better path</h5><p>All three head for the same valley bottom. Momentum and Adam just pick smarter step sizes, so they usually get there faster and with less shaking.</p></div>
      </div>
      <p><strong>In one line:</strong> SGD walks, momentum rolls, and Adam rolls with a per-weight sense of the ground.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a rolling ball feels real gravity every instant, but an optimizer only "feels" the slope once per step, using the gradient from one <span class="term" data-tip="A small chunk of the dataset processed together in one training step.">batch</span> of data. So its speed is a stored number it updates by hand, not true physics.</div></div>
      <p><b>直觉 / Intuition:</b> 直觉上，optimizer 就是一个"下山教练" — SGD 每步只看当前坡度，Momentum 记住了速度所以能冲过小坑，Adam 还给每条腿单独调步幅。The technical point: all three implement <code>w ← w − (step)</code>; they differ only in how the step is computed from the gradient history.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The three update rules, with real numbers</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>1 · SGD — step straight downhill</h4>
      <p><b>In words:</b> move the weight by the learning rate times the gradient, in the downhill direction.</p>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>w ← w − lr × g</code><br><span style="color:var(--muted)">w = weight · lr = learning rate (step size) · g = gradient (slope of the loss)</span></div></div>
      <p><b>Worked example:</b> start <code>w = 1.0</code>, gradient <code>g = −16</code>, <code>lr = 0.05</code>. Step = <code>0.05 × (−16) = −0.8</code>. New <code>w = 1.0 − (−0.8) = 1.8</code>. That single move fits the data a little better. This <em>is</em> the vanilla gradient descent you named on Day 5 — the baseline the next two rules build on.</p>

      <h4>2 · Momentum — add a running velocity</h4>
      <p><b>In words:</b> keep a stored speed <code>v</code>. Each step, keep most of the old speed and add the new gradient. Then move by that speed. This lets the weight coast in a steady direction.</p>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>v ← β × v + g</code>&nbsp;&nbsp;then&nbsp;&nbsp;<code>w ← w − lr × v</code><br><span style="color:var(--muted)">v = velocity (running speed) · β = momentum, e.g. 0.9 (how much old speed to keep) · g = gradient</span></div></div>
      <p><b>Worked example (same first step):</b> start <code>v = 0</code>, <code>g = −16</code>, <code>β = 0.9</code>. New <code>v = 0.9 × 0 + (−16) = −16</code>. Then <code>w = 1.0 − 0.05 × (−16) = 1.8</code>. The <b>first</b> step matches SGD exactly, because there is no speed built up yet.</p>
      <div class="callout c-info"><span class="ic">🔎</span><div><b>The difference shows up on step two.</b> At <code>w=1.8</code> the next gradient is <code>−9.6</code>. Momentum keeps its stored speed: <code>v = 0.9 × (−16) + (−9.6) = −24.0</code>, so it moves <code>0.05 × 24.0 = 1.2</code>. Plain SGD, with no memory, only moves <code>0.05 × 9.6 = 0.48</code>. The stored speed makes momentum cover <b>2.5× the ground</b> — it is picking up speed. (These are the exact numbers your 15-step artifact prints.)</div></div>

      <h4>3 · Adam — a per-weight step size</h4>
      <p><b>In words:</b> keep two running averages for each weight — one of the gradient (<code>m</code>), one of the gradient squared (<code>v</code>). Divide the step by the square root of <code>v</code>, so weights with big, noisy gradients take smaller steps and quiet ones take bigger steps.</p>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>m ← β₁ m + (1−β₁) g</code><br><code>v ← β₂ v + (1−β₂) g²</code><br><code>w ← w − lr × m̂ / (√v̂ + ε)</code><br><span style="color:var(--muted)">m = average gradient · v = average of gradient² · m̂,v̂ = bias-corrected (divide by 1−βᵗ so the cold start toward 0 is fixed) · typical β₁=0.9, β₂=0.999, ε=1e-8</span></div></div>
      <p>You do not need to compute Adam by hand today. The idea to keep: it stores <b>two</b> extra numbers per weight, <code>m</code> and <code>v</code>. Adam is really <b>RMSProp</b> (the per-weight <code>/√v</code> scaling) plus <b>momentum</b> (the running <code>m</code>) — its ancestors are AdaGrad and RMSProp.</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — SGD vs momentum, one step on <code>pred=w·x</code></b>
      <br><b>1 · In words:</b> SGD steps by the current slope; momentum first blends the new slope into a running speed, then steps by that speed, so repeated downhill pushes build up.
      <br><b>2 · The formula:</b> SGD: <code>w ← w − lr·g</code>. Momentum: <code>v ← β·v + g</code>, then <code>w ← w − lr·v</code>. Here <code>x=2, y=6</code>, <code>w=1</code>, <code>lr=0.05</code>, <code>β=0.9</code>, start <code>v=0</code>, <code>g = 2(w·x − y)·x</code>.
      <br><b>3 · Tiny numbers:</b> step 1 both see <code>g = 2(2−6)(2) = −16</code>. SGD → <code>w = 1.8</code>; momentum <code>v = 0.9·0 + (−16) = −16</code> → same <code>w = 1.8</code>. But on step 2 momentum's <code>v</code> still carries the old −16, so it moves <em>further</em> than SGD and pulls ahead.
      <br><b>4 · Sanity check:</b> with <code>β=0</code>, <code>v = g</code> and momentum becomes exactly SGD. Bigger <code>β</code> keeps more past speed; too big and it overshoots the bottom before slowing.</div></div>

      <h4>Why a frontier lab cares about that "two extra numbers"</h4>
      <p>Adam keeps <code>m</code> and <code>v</code> for <em>every</em> weight. So the <span class="term" data-tip="The extra numbers an optimizer stores per weight; for Adam that is m and v, roughly doubling the memory beyond the weights themselves.">optimizer state</span> is about <b>2× the size of the model's weights</b> — extra memory a frontier lab must pay for on every accelerator. That's exactly the "two extra numbers" cost ZeRO/DeepSpeed from the teaser above was built to fix. When you reach the memory and parallelism modules (Module 10), this "Adam doubles your state" fact is a big reason huge models are split across many chips.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> SGD steps by <code>lr × g</code>; momentum stores a speed so it rolls through bumps; Adam stores <code>m</code> and <code>v</code> per weight for a smart per-weight step — at the cost of ~2× optimizer memory.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>Adam's power comes from the two extra numbers it stores per weight — and those numbers are also where a quiet bug and a real memory bill both come from.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): resuming training without the optimizer state.</b> When you save and reload a model, the weights are the obvious thing to save — but Adam's smart step depends on its running averages <code>m</code> and <code>v</code> for every weight. If you reload only the weights and let the optimizer start fresh, <code>m</code> and <code>v</code> reset to zero — like carrying the ball back to the same spot but forgetting all the speed it had built up, so it starts rolling from a dead stop. Training resumes with no error, but Adam has forgotten every per-weight step size it learned, so the first steps after resume are crude and can quietly slow or destabilize training. A senior engineer catches this in <b>code review</b> by confirming the checkpoint saves and restores the optimizer state, not just the model weights.</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: Adam's fast convergence vs. its memory bill.</b> Adam usually trains faster and needs less learning-rate tuning than plain SGD, because its per-weight step adapts automatically. You pay for that with memory: storing <code>m</code> and <code>v</code> roughly doubles the optimizer state beyond the weights themselves. At frontier scale that extra memory decides how a model must be split across chips. Choosing Adam versus a lighter optimizer is a <b>design-review</b> decision that trades convergence speed against memory footprint.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "Every optimizer does <code>w ← w − step</code>; they differ in how the step uses gradient history. SGD uses the raw gradient; momentum keeps a running velocity <code>v ← βv + g</code> to roll through noise and small bumps; Adam adds a per-weight adaptive step by tracking running averages of the gradient and its square. Adam usually converges faster with less lr tuning, but it stores <code>m</code> and <code>v</code> per weight — roughly 2× the model's weights in optimizer state — which is exactly why memory-sharding like ZeRO exists at scale."</div></div>
      <button class="gotit" type="button">Got the three rules</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Watch momentum overtake SGD</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Here's the race worth watching. <b>Before you run it, predict:</b> SGD and momentum take the <em>same</em> first step (both land at <code>w=1.8</code>) — so when does momentum pull ahead? Run both on the same tiny problem and watch step 2: momentum's stored speed hits <code>v = −24.0</code> and it covers <b>2.5× the ground</b> SGD does, then reaches the bottom first. That head start is the whole point of keeping a velocity. Pick one path:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m02-the-neuron/day-07-optimizers/experiment.py</code>. Fit <code>pred = w·x</code> to <code>x=2, y=6</code>. Run two loops of 15 steps from <code>w=1.0</code>, <code>lr=0.05</code>: one plain SGD (<code>w = w − lr*grad</code>), one momentum (<code>v = 0.9*v + grad</code>; <code>w = w − lr*v</code>, start <code>v=0</code>). Print step, w, and loss for each. Confirm momentum reaches <code>w ≈ 3</code> in fewer steps. Run with <code>python3 sessions/m02-the-neuron/day-07-optimizers/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 2 Day 7 artifact.

Create sessions/m02-the-neuron/day-07-optimizers/experiment.py that, with a comment on each step:
1. Sets a tiny model pred = w*x, with x=2.0, target y=6.0 (ideal w=3), lr=0.05.
2. Runs plain SGD from w=1.0 for 15 steps: pred=w*x; grad=2*(pred-y)*x; w = w - lr*grad; print step, round(w,4), round((w*x-y)**2,4).
3. Runs momentum from w=1.0, v=0.0, beta=0.9 for 15 steps: grad=2*(w*x-y)*x; v = beta*v + grad; w = w - lr*v; print step, round(w,4), round(loss,4).
4. Prints how many steps each needed to get loss below 0.01, showing momentum converges faster.
5. Adds a comment noting that Adam keeps two extra numbers (m and v) per weight, so its optimizer state is about 2x the model weights.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>What you should see (check your prediction)</h4>
      <ul>
        <li>Both loops (SGD and momentum) print step, <code>w</code>, and <code>loss</code> from <code>w=1.0</code> on the <code>x=2, y=6</code> problem.</li>
        <li>Momentum reaches <code>w ≈ 3</code> in <em>fewer</em> steps than plain SGD — you can point to the step where it pulls ahead.</li>
        <li>You can explain in one sentence what the velocity <code>v</code> adds over plain SGD.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-07-optimizers/log.md</code>: (1) what all three optimizers have in common; (2) on which step momentum first pulled ahead of SGD, and why; (3) why Adam costs ~2× the weights in memory.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 2 · Day 7 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Optimizers: SGD → Momentum → Adam</b>.<br>Next up: <b>Learning-Rate Intuition</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  sgd:{html:'<span class="prompt">&gt;&gt;&gt;</span> x, y, lr = <span class="num">2.0</span>, <span class="num">6.0</span>, <span class="num">0.05</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = <span class="num">1.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> pred = w * x            <span class="dim"># forward → 2.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> grad = <span class="num">2</span>*(pred - y)*x     <span class="dim"># gradient → -16.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = w - lr*grad          <span class="dim"># SGD update: w - 0.05*(-16)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w\n'+
    '<span class="ok">1.8</span>',
    take:'<b>①  Plain SGD.</b> Step straight downhill by lr × grad: 0.05 × (−16) = −0.8, so w goes 1.0 → 1.8. Honest and simple. No memory of past steps.'},
  momentum:{html:'<span class="prompt">&gt;&gt;&gt;</span> w, v, beta = <span class="num">1.0</span>, <span class="num">0.0</span>, <span class="num">0.9</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> grad = <span class="num">2</span>*(w*x - y)*x     <span class="dim"># gradient → -16.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> v = beta*v + grad        <span class="dim"># 0.9*0 + (-16) = -16.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = w - lr*v             <span class="dim"># same as SGD on step 1</span>\n'+
    '<span class="hl"># --- step 2, next grad = -9.6 (at w=1.8) ---</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> v = <span class="num">0.9</span>*(-<span class="num">16</span>) + (-<span class="num">9.6</span>)   <span class="dim"># v = -24.0  (speed builds up)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> move = lr*v              <span class="dim"># 0.05*24.0 = 1.2  vs SGD 0.48</span>\n'+
    '<span class="ok">momentum moves 1.2, SGD moves 0.48</span>',
    take:'<b>②  Momentum builds speed.</b> Step 1 matches SGD (no speed yet). By step 2 the stored velocity makes it move 1.2 vs SGD\'s 0.48 — 2.5× as far, in the same steady direction.'},
  adam:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># Adam keeps TWO running averages per weight</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> m = beta1*m + (<span class="num">1</span>-beta1)*grad     <span class="dim"># average gradient</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> v = beta2*v + (<span class="num">1</span>-beta2)*grad**<span class="num">2</span>  <span class="dim"># average of grad squared</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = w - lr * m / (v**<span class="num">0.5</span> + eps) <span class="dim"># per-weight step size</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># memory cost, per weight:</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> stored = {<span class="str">"w"</span>: <span class="num">1</span>, <span class="str">"m"</span>: <span class="num">1</span>, <span class="str">"v"</span>: <span class="num">1</span>}\n'+
    '<span class="ok"># 2 optimizer numbers (m, v) per weight ≈ 2× the weights in optimizer state</span>',
    take:'<b>③  Adam stores m and v.</b> Each weight gets its own step size from the running gradient (m) and gradient-squared (v). Cost: 2 extra numbers per weight → about 2× optimizer-state memory (bridge to Module 10).'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='The optimizer only changes step four of the training loop'><g font-family='monospace' font-size='11.5' text-anchor='middle'><rect x='20' y='44' width='96' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='68' y='63' fill='#1F6280'>forward</text><rect x='140' y='44' width='80' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='180' y='63' fill='#C93B3B'>loss</text><rect x='244' y='44' width='100' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='294' y='63' fill='#9A7208'>backprop</text><rect x='368' y='44' width='132' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='3'/><text x='434' y='63' fill='#1a5c38'>UPDATE ← here</text><text x='434' y='96' font-size='10' fill='#6B645E'>the optimizer lives here</text></g></svg>",
  note:"<b>Only the update changes.</b> Forward, loss, and backprop stay exactly as in Day 6. The optimizer is just a smarter <b>step 4</b> — how to turn gradients into a weight change."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='SGD steps straight downhill by learning rate times gradient'><g font-family='monospace' font-size='12' text-anchor='middle'><path d='M40 90 Q 260 -10 480 90' fill='none' stroke='#2A7B9B' stroke-width='2'/><circle cx='80' cy='78' r='8' fill='#C99A12' stroke='#7a5a08' stroke-width='1.5'/><circle cx='150' cy='55' r='8' fill='#C99A12' stroke='#7a5a08' stroke-width='1.5' opacity='0.55'/><path d='M92 76 L138 58' stroke='#6B645E' stroke-width='1.3' stroke-dasharray='3 3' marker-end='url(#a2)'/><text x='260' y='108' fill='#2C2A28'>w ← w − lr × g   (one small step)</text><defs><marker id='a2' markerWidth='7' markerHeight='7' refX='5' refY='3' orient='auto'><path d='M0 0 L6 3 L0 6 z' fill='#6B645E'/></marker></defs></g></svg>",
  note:"<b>1 · SGD — walk downhill.</b> Move by <b>lr × gradient</b>, one small step at a time. From w = 1.0 with grad −16 and lr 0.05, the step is 0.8, so w → 1.8. Simple, but it can be slow and shaky."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Momentum stores a velocity that grows over steps'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='42' width='150' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='115' y='62' fill='#5E5191'>v ← β·v + g</text><text x='210' y='62' fill='#6B645E'>then</text><rect x='250' y='42' width='150' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='325' y='62' fill='#1a5c38'>w ← w − lr·v</text><text x='260' y='100' fill='#2C2A28'>step 2: v = 0.9·(−16) + (−9.6) = −24.0</text></g></svg>",
  note:"<b>2 · Momentum — add a running speed.</b> Keep most of the old velocity (β = 0.9) and add the new gradient. Step 1 matches SGD, but by step 2 the stored speed makes it move <b>more than twice as far</b> in the steady downhill direction."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Momentum rolls straight through a small bump that stops SGD'><g font-family='monospace' font-size='12'><path d='M30 80 C 140 80, 180 55, 220 80 C 260 100, 340 30, 490 30' fill='none' stroke='#2A7B9B' stroke-width='2'/><circle cx='170' cy='63' r='8' fill='#9399B2' stroke='#555' stroke-width='1.5'/><text x='150' y='105' font-size='10' fill='#C93B3B'>SGD stalls at the bump</text><circle cx='430' cy='36' r='8' fill='#C99A12' stroke='#7a5a08' stroke-width='1.5'/><text x='360' y='20' font-size='10' fill='#1a5c38'>momentum coasts over</text></g></svg>",
  note:"<b>Why the speed helps.</b> A tiny bump or flat spot can stop plain SGD. Momentum arrives with built-up speed, so it <b>coasts over the bump</b> and keeps heading for the valley. That is the whole point of storing a velocity."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Adam gives each weight its own step size from two running averages'><g font-family='monospace' font-size='11.5' text-anchor='middle'><rect x='30' y='16' width='200' height='26' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='130' y='33' fill='#1F6280'>m ← avg of gradient</text><rect x='30' y='50' width='200' height='26' rx='5' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='130' y='67' fill='#9A7208'>v ← avg of gradient²</text><text x='300' y='38' fill='#6B645E'>→</text><rect x='330' y='33' width='170' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='415' y='53' fill='#1a5c38'>w − lr·m/√v</text><text x='260' y='104' fill='#2C2A28'>steady slope → big step · noisy slope → small step</text></g></svg>",
  note:"<b>3 · Adam — a per-weight step size.</b> It keeps a running average of the gradient (m) and of the gradient squared (v) for every weight. Dividing by √v makes noisy weights take small steps and steady ones take big steps."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Adam roughly doubles memory because it stores m and v per weight'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='60' y='44' width='90' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='105' y='64' fill='#1F6280'>weights</text><text x='168' y='64' fill='#6B645E'>+</text><rect x='190' y='44' width='60' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='220' y='64' fill='#5E5191'>m</text><rect x='258' y='44' width='60' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='288' y='64' fill='#9A7208'>v</text><text x='340' y='64' fill='#6B645E'>≈</text><rect x='368' y='44' width='120' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='428' y='64' fill='#C93B3B'>2× extra state</text><text x='260' y='104' fill='#2C2A28'>a real memory cost on every chip → Module 10</text></g></svg>",
  note:"<b>Why a frontier lab cares.</b> Adam stores <b>m</b> and <b>v</b> for every weight, so the optimizer state is about <b>2× the model's weights</b> — extra memory on every accelerator. ZeRO/DeepSpeed exists to shard that state across GPUs instead. That is a big reason huge models are split across many chips (Module 10). You now know how the step is chosen."}
];
@@@ region name=QS
var QS=[
 {q:'1. Which part of the training loop does an optimizer control?',
  opts:['the forward pass','the update step (how gradients change the weights)','the loss function','the data loading'],
  ans:1, fb:'Right. Forward, loss, and backprop stay the same. The optimizer only decides step 4 — how to turn gradients into a weight change.'},
 {q:'2. You stop a training run and resume it later. You load the saved model weights, but your code creates a brand-new Adam optimizer instead of restoring the saved one. Training resumes with no error, yet the loss briefly gets worse right after resuming. Where do you look first, and why?',
  opts:['The dataset changed, which is the only thing that can raise the loss on resume','A fresh Adam optimizer has its running averages m and v reset to zero, so it has forgotten every per-weight step size it had learned — the first steps after resume are crude and can bump the loss up','The learning rate doubled itself on resume, which always happens','A loss bump on resume is impossible and must be a display glitch'],
  ans:1, fb:'Right. Adam\'s smart step depends on m and v per weight. Restoring only the model weights and starting the optimizer fresh throws that state away, so Adam behaves crudely for the first steps. Nothing crashes — you check that the checkpoint saves and restores the optimizer state, not just the weights.'},
 {q:'3. Starting w=1.0 with gradient −16 and lr=0.05, what is w after one SGD step?',
  opts:['1.8','−0.8','16.8','0.2'],
  ans:0, fb:'Right. Step = lr × grad = 0.05 × (−16) = −0.8, so w = 1.0 − (−0.8) = 1.8. The first momentum step gives the same 1.8 (no speed built up yet).'},
 {q:'4. Why does Adam use about twice the optimizer memory of SGD?',
  opts:['it trains for twice as long','it stores two extra numbers per weight (m and v)','it uses two GPUs','it doubles the learning rate'],
  ans:1, fb:'Right. Adam keeps a running average of the gradient (m) and of the gradient squared (v) for every weight — about 2× extra state, which matters at frontier scale (Module 10).'}
];
