---
quest_id: wf2-d02-activations
donor: m02-day-02.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 2 · Day 2 — Activation Functions"
spine: "valve / dimmer"
notebook_yardstick: 00-neural-networks/fundamentals/03_activation_functions.ipynb
---

@@@ region name=title
<title>Module 2 · Day 2 — Activation Functions</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M2 Day 2</div>
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
<a class="lnav prev" href="../day-01-single-neuron/lesson.html"><span class="d">← Prev</span><span class="t">A Single Neuron</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-03-layers-forward-pass/lesson.html"><span class="d">Next →</span><span class="t">Layers &amp; the Forward Pass</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 2 · Train · Day 2</span>
      <h1>Activation Functions<span class="sub">The Bend That Makes Depth Matter</span></h1>
      <p class="lede">Yesterday a neuron ended with a mysterious third step: the activation. Today you find out why it's there — and it's a big deal. Without it, stacking a hundred layers would be no better than one. The activation is the little bend that lets deep networks learn complicated things.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><b>By the end, you'll be able to:</b> say what an activation adds (non-linearity), show why layers with no activation collapse into one, and describe ReLU, sigmoid, and tanh.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">The neuron's third step, explained</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (a neuron = weighted sum + bias + activation) and M1 Day 4 (matmul, and that stacking layers = matmuls). <b>Nothing new needed.</b></div></div>

      <h4>What is an activation function?</h4>
      <p>An <span class="term" data-tip="A simple function applied at the end of a neuron, after the weighted sum + bias. It adds non-linearity so stacked layers can learn complex shapes.">activation function</span> is the small curve applied at the very end of a neuron, after the weighted sum and bias. It takes the <code>pre-activation</code> value <code>z</code> and bends it into the output.</p>

      <h4>Why does it exist? (the big reason)</h4>
      <p>Here's the catch that makes it essential. A weighted sum plus a bias is a <span class="term" data-tip="A straight-line relationship: output changes in fixed proportion to input. A matmul plus bias is linear.">linear</span> step — picture a perfectly straight ruler. Now stack two of them: lay one straight ruler on top of another and you still get… a single straight ruler. Two neuron-layers with no bend between them collapse into one layer, so all that extra depth buys you nothing. (Section 4 proves this exactly with two small matrices; for now just hold the picture — straight stacked on straight is still straight.)</p>
      <p>The activation adds <span class="term" data-tip="Not a straight line. A non-linear step between layers lets a deep network bend and combine features to learn complex patterns.">non-linearity</span> — a bend. Put a bend between each layer and the layers can no longer be squashed into one. That is the entire reason deep networks work. Three common bends: <strong>ReLU</strong>, <strong>sigmoid</strong>, and <strong>tanh</strong>. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> Llama's models use an activation called SwiGLU; GPT-2 and GPT-3 use one called GELU. Different frontier labs pick different bends — but every one of them exists for the exact reason you're about to see.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> activation 看起来只是"最后随手加的一个函数"，很容易被当成不重要的一步 — the activation looks like an optional extra you tack on at the end. In plain terms: without it, stacking 100 layers is mathematically the same as having <b>one</b> layer — all that depth silently collapses. The activation is the single thing that makes "deep" worth anything.</div></div>
      <button class="gotit" type="button">Got why it exists — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">A valve and a dimmer switch</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <div class="relate">
        <div class="card"><span class="big">🚰</span><h5>ReLU = a one-way valve</h5><p>Let positive signals flow straight through; block anything negative (set it to 0). Simple, fast, and the default in modern networks.</p></div>
        <div class="card"><span class="big">🎛️</span><h5>Sigmoid = a soft dimmer</h5><p>Squash any number gently into the range 0 to 1 — a smooth "off → on." Great for a probability, but it flattens out at the extremes.</p></div>
      </div>
      <p><strong>In one line:</strong> an activation is a small non-linear gate; stacking linear layers without one is pointless, so the gate is what makes depth worth having.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: these are just two of many activations, and the "gate" isn't on/off — it's a smooth curve the network can differentiate (needed for training — the gradient story on Day 5).</div></div>
      <p><b>直觉 / Intuition:</b> 直觉上，activation 就是每层之间的一个"小弯" — 没有它，一堆直线接起来还是一条直线；有了它，网络才能拐弯、画出复杂形状。The technical point: a stack of linear layers is still one linear map, so a non-linear activation between layers is what lets a deep network represent curves instead of only straight lines.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The two formulas, and why ReLU won</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The three you'll see most</h4>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>ReLU(z) = max(0, z)</code> — keep positives, zero negatives; range <code>[0, ∞)</code>.<br><code>sigmoid(z) = 1 / (1 + e^(−z))</code> — squash into <code>(0, 1)</code>.<br><code>tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)</code> — squash into <code>(−1, 1)</code>, and <b>zero-centered</b>: <code>tanh(0) = 0</code>, unlike sigmoid's <code>0.5</code>.</div></div>
      <p><code>tanh</code> is sigmoid's zero-centered cousin — the same smooth S-curve, but it runs from <code>−1</code> to <code>+1</code> and passes through the origin, so a layer's outputs stay centered on 0 (which helps the next layer learn). It was the default hidden-layer bend before ReLU took over, and it still shows up (for example in older recurrent networks).</p>
      <div class="build-viz" style="margin:14px 0"><svg viewBox="0 0 520 140" role="img" aria-label="tanh curve: an S shape through the origin, rising from near minus one up to near plus one"><g font-family="monospace"><line x1="60" y1="70" x2="460" y2="70" stroke="#E5DFD6" stroke-width="1.5"/><line x1="260" y1="16" x2="260" y2="124" stroke="#E5DFD6" stroke-width="1"/><path d="M70 120 C 190 118, 205 74, 260 70 C 315 66, 330 22, 450 20" fill="none" stroke="#9A5A12" stroke-width="2.5"/><text x="110" y="136" font-size="11" fill="#6B645E" text-anchor="middle">→ near −1</text><text x="410" y="15" font-size="11" fill="#9A5A12" text-anchor="middle">→ near +1</text><circle cx="260" cy="70" r="4" fill="#C93B3B"/><text x="300" y="62" font-size="11" fill="#C93B3B" text-anchor="middle">tanh(0)=0</text><text x="400" y="98" font-size="12" fill="#9A5A12">tanh → (−1, 1)</text></g></svg></div>
      <table style="width:100%;border-collapse:collapse;margin:16px 0;font-size:.86rem">
        <thead><tr style="text-align:left;border-bottom:2px solid var(--line2)"><th style="padding:.5rem .6rem"></th><th style="padding:.5rem .6rem;color:var(--ok-ink)">ReLU</th><th style="padding:.5rem .6rem;color:var(--k)">sigmoid</th><th style="padding:.5rem .6rem;color:var(--v)">tanh</th></tr></thead>
        <tbody>
          <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Formula</td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>max(0, z)</code></td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>1/(1+e⁻ᶻ)</code></td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)</code></td></tr>
          <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Range</td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>[0, ∞)</code></td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>(0, 1)</code></td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>(−1, 1)</code></td></tr>
          <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Zero-centered?</td><td style="padding:.5rem .6rem;color:var(--ink2)">no</td><td style="padding:.5rem .6rem;color:var(--ink2)">no (centers at 0.5)</td><td style="padding:.5rem .6rem;color:var(--ink2)">yes (centers at 0)</td></tr>
          <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Slope near 0</td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>1</code> for z&gt;0, <code>0</code> for z&lt;0</td><td style="padding:.5rem .6rem;color:var(--ink2)">≈ <code>0.25</code></td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>1</code></td></tr>
          <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Saturates?</td><td style="padding:.5rem .6rem;color:var(--ink2)">only for z&lt;0 (flat 0 → can "die")</td><td style="padding:.5rem .6rem;color:var(--ink2)">both tails (→ 0 and → 1)</td><td style="padding:.5rem .6rem;color:var(--ink2)">both tails (→ −1 and → 1)</td></tr>
          <tr><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Typical use</td><td style="padding:.5rem .6rem;color:var(--ink2)">hidden layers (modern default)</td><td style="padding:.5rem .6rem;color:var(--ink2)">binary output / gate</td><td style="padding:.5rem .6rem;color:var(--ink2)">hidden layers (esp. older RNNs); zero-centered</td></tr>
        </tbody>
      </table>

      <h4>Why non-linearity is non-negotiable</h4>
      <p>A composition of linear steps is still linear: <code>(x W₁) W₂ = x (W₁ W₂)</code>. So without a bend between layers, a 100-layer network can only ever draw a straight line — useless for images, language, anything real. One non-linear activation per layer breaks that, and (loosely) lets a big enough network approximate almost any function.</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — why linear ∘ linear collapses</b>
      <br><b>1 · In words:</b> two plain matmuls back-to-back, with no bend between them, do the same job as one matmul — so the second layer bought you nothing.
      <br><b>2 · Scalars first (no matrix algebra needed):</b> chain two linear steps <code>f(x)=a·x+p</code> then <code>g(u)=b·u+q</code>. Compose them: <code>g(f(x)) = b·(a·x+p)+q = (ab)·x + (bp+q)</code> — still <code>slope·x + shift</code>, i.e. one linear step. Two linear layers with no bend collapse to one.
      <br><b>3 · Now the matrix version (our anchor pair):</b> <code>x=[1,2]</code>, <code>W₁=[[1,2],[0,1]]</code>, <code>W₂=[[1,0],[3,1]]</code>. Then <code>(x W₁) W₂ = [13,4]</code>, and <code>W₁ W₂ = [[7,2],[3,1]]</code> so <code>x (W₁ W₂) = [13,4]</code> — identical (the same <code>W₁ W₂</code> the playground prints). Insert an activation <code>a(·)</code> and <code>a(x W₁) W₂</code> can no longer be folded up.
      <br><b>4 · Sanity check:</b> now clip with ReLU first — <code>ReLU(x W₁) W₂</code> — and any negative entries get zeroed <em>before</em> the second matmul, producing values no single matmul can. That clip is the entire reason activations exist.</div></div>

      <h4>Why a frontier lab cares</h4>
      <p><strong>ReLU</strong> is the classic workhorse: it's dirt cheap (just a max) and it avoids the <span class="term" data-tip="When gradients shrink toward zero as they pass back through many layers, so early layers barely learn. Sigmoid suffers from this; ReLU largely avoids it.">vanishing-gradient</span> problem that plagues sigmoid in deep stacks (you'll meet gradients on Day 5). It's also why the systems from the teaser above exist. Their smoother curves are one real answer to the "dead neuron" risk you're about to meet below — a curve that never fully flattens, unlike a plain ReLU. Sigmoid and its cousin <span class="term" data-tip="Turns a vector of scores into probabilities that sum to 1. Used at the output of a classifier.">softmax</span> live mostly at the output, where you want probabilities.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> linear ∘ linear = linear → depth alone buys nothing → a non-linear activation per layer breaks the collapse → deep networks can learn complex functions. The activation is the reason "deep" learning is a thing.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>The activation you pick decides whether a neuron keeps learning. Both common activations have a quiet way to stop learning that throws no error.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): saturated units that stop learning.</b> A <code>sigmoid</code> squashes into <code>(0, 1)</code>; when its input is far from zero (say <code>z = 10</code> or <code>z = −10</code>), the output flattens near <code>1</code> or <code>0</code> and barely moves when the input changes. A <code>ReLU</code> that always sees a negative input outputs a flat <code>0</code> — its one-way valve has jammed shut, and nothing flows through it again. That is a "dead" ReLU. In both cases the neuron's output is nearly constant, so training can no longer nudge it, and the network quietly learns less than it could. Nothing crashes; the loss just plateaus higher than expected. A senior engineer catches this in <b>code review</b> by logging activation statistics — the fraction of ReLUs stuck at zero, or how many sigmoids sit pinned near 0 or 1 — instead of only watching the loss number.</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: ReLU's speed and sparsity vs. its dead-neuron risk.</b> Choosing <code>ReLU</code> buys you a cheap activation (just a <code>max</code>) that keeps gradients healthy in deep stacks and turns off many neurons at once, which is often useful. What you give up is safety at zero: any neuron pushed permanently negative dies and never comes back. Softer activations that never fully flatten (Leaky ReLU, GELU — a valve left with a permanent small leak: they keep a slight slope for negative inputs) trade a little extra math and less sparsity for never dying. Which one to use, and where, is a <b>design-review</b> decision — commonly ReLU-family in the hidden layers, sigmoid or softmax only at the output.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "An activation is the per-layer non-linearity. Without it <code>(x W₁) W₂ = x (W₁ W₂)</code> — the whole stack collapses to one linear map, so depth adds nothing. ReLU = max(0, z) is the cheap default and dodges vanishing gradients; sigmoid and softmax live at the output for probabilities. The silent failure is saturation — dead ReLUs stuck at 0, or sigmoids pinned near 0/1 — where the unit stops learning and the loss plateaus, so I log activation statistics, not just the loss."</div></div>
      <button class="gotit" type="button">Got why depth needs a bend</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Watch two layers collapse — then break the collapse</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Here's the experiment worth running yourself. Build two linear layers with no bend between them and watch them fold into a single matrix — then drop a <code>ReLU</code> in the middle and watch that collapse break. Seeing <code>(x@W1)@W2</code> and <code>x@(W1@W2)</code> print the <em>same</em> numbers, and then differ the moment the valve goes in, is the whole lesson in two lines of output. Pick one path:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m02-the-neuron/day-02-activations/experiment.py</code>. Write <code>relu</code>, <code>sigmoid</code>, and <code>tanh</code>, and print all three as a table over the grid <code>np.linspace(-6, 6, 13)</code> (check ReLU zeros negatives, sigmoid stays in <code>(0,1)</code>, tanh stays in <code>(−1,1)</code> and is 0 at 0). Also print <code>sigmoid'(0) ≈ 0.25</code> and note that ReLU's gradient is <code>0</code> for every negative input. Then use the anchor pair <code>W1=[[1,2],[0,1]]</code>, <code>W2=[[1,0],[3,1]]</code> and show that <code>(x@W1)@W2</code> equals <code>x@(W1@W2)</code> for a random <code>x</code> — print <code>W1@W2</code> (expect <code>[[7,2],[3,1]]</code>), proving two linear layers = one. Run with <code>python3 sessions/m02-the-neuron/day-02-activations/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 2 Day 2 artifact.

Create sessions/m02-the-neuron/day-02-activations/experiment.py that, with a comment on each step:
1. Defines relu(z)=np.maximum(0,z), sigmoid(z)=1/(1+np.exp(-z)), and tanh(z)=np.tanh(z); prints all three as a table over grid = np.linspace(-6,6,13) (ReLU zeros negatives, sigmoid in (0,1), tanh in (-1,1) and 0 at 0). Also prints sigmoid'(0)=0.25 and notes ReLU's gradient is 0 for negative inputs.
2. Shows the collapse: with W1=[[1,2],[0,1]], W2=[[1,0],[3,1]] and a random x, asserts np.allclose((x@W1)@W2, x@(W1@W2)) and prints W1@W2 (expect [[7,2],[3,1]]) — two linear layers equal one.
3. Then inserts relu between them: shows relu(x@W1)@W2 is NOT equal to any single x@W, i.e. the bend prevents the collapse.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>What you should see (the collapse, then the cure)</h4>
      <ul>
        <li>Your <code>relu</code>, <code>sigmoid</code>, and <code>tanh</code> print sensible values over the grid <code>np.linspace(-6, 6, 13)</code>: ReLU zeros negatives, sigmoid stays inside <code>(0, 1)</code>, tanh stays inside <code>(−1, 1)</code> and is 0 at 0.</li>
        <li>You print <code>sigmoid'(0) ≈ 0.25</code> and note ReLU's gradient is <code>0</code> for negative inputs — the two facts behind saturation and "dead" (jammed-shut) ReLUs.</li>
        <li>The moment worth watching: <code>(x@W1)@W2</code> prints the <em>same</em> numbers as <code>x@(W1@W2)</code> for a random <code>x</code> — two linear layers really did collapse to one — and then, once you slip a <code>ReLU</code> between them, the two stop matching. That break is the bend earning its keep.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-02-activations/log.md</code>: (1) why a 100-layer network with no activations is no better than one layer; (2) ReLU vs sigmoid in one line; (3) what a "dead" or "saturated" unit is and how you'd detect it.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 2 · Day 2 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Activation Functions</b>.<br>Next up: <b>Layers &amp; the Forward Pass</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  relu:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">def</span> <span class="fn">relu</span>(x): <span class="kw">return</span> np.maximum(<span class="num">0</span>, x)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> relu(np.array([<span class="num">-3</span>, <span class="num">-1</span>, <span class="num">0</span>, <span class="num">2</span>, <span class="num">5</span>]))\n'+
    '<span class="ok">array([0, 0, 0, 2, 5])</span>',
    take:'<b>①  ReLU = max(0, x).</b> It zeroes every negative and lets positives pass through unchanged. One cheap operation — this is the default activation in modern networks.'},
  sigmoid:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">def</span> <span class="fn">sigmoid</span>(x): <span class="kw">return</span> <span class="num">1</span>/(<span class="num">1</span>+np.exp(-x))\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> sigmoid(np.array([<span class="num">-2</span>, <span class="num">0</span>, <span class="num">2</span>]))\n'+
    '<span class="ok">array([0.119, 0.5  , 0.881])</span>',
    take:'<b>②  Sigmoid squashes into (0, 1).</b> A smooth "off → on": sigmoid(0)=0.5, big negatives → near 0, big positives → near 1 (the shown values are rounded — sigmoid(2)≈0.8808). Handy for a probability, but it flattens at the extremes.'},
  collapse:{html:'<span class="prompt">&gt;&gt;&gt;</span> W1 = np.array([[<span class="num">1</span>,<span class="num">2</span>], [<span class="num">0</span>,<span class="num">1</span>]])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> W2 = np.array([[<span class="num">1</span>,<span class="num">0</span>], [<span class="num">3</span>,<span class="num">1</span>]])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># two linear layers, no activation:</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># (x @ W1) @ W2  ==  x @ (W1 @ W2)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> W1 @ W2\n'+
    '<span class="hl">array([[7, 2],\n       [3, 1]])</span>',
    take:'<b>③  The collapse.</b> Without an activation, two layers fold into one: (x@W1)@W2 = x@(W1@W2), and W1@W2 is just a single matrix. Ten layers, a hundred — still one linear layer. The activation between them is what stops this.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='ReLU curve: flat at zero for negatives, a straight rising line for positives'><g font-family='monospace'><line x1='60' y1='95' x2='320' y2='95' stroke='#E5DFD6' stroke-width='1.5'/><line x1='190' y1='25' x2='190' y2='110' stroke='#E5DFD6' stroke-width='1'/><path d='M70 95 L190 95 L300 40' fill='none' stroke='#2D8B55' stroke-width='2.5'/><text x='120' y='112' font-size='11' fill='#6B645E' text-anchor='middle'>negatives → 0</text><text x='260' y='30' font-size='11' fill='#2D8B55' text-anchor='middle'>positives pass</text><text x='400' y='70' font-size='13' fill='#1a5c38' font-family='monospace'>ReLU(x)=max(0,x)</text></g></svg>",
  note:"<b>ReLU: a one-way valve.</b> Left of zero it's flat (output 0); right of zero it's the line y = x. One <code>max(0, x)</code>. Cheap, and it's what almost every hidden layer uses today."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Sigmoid curve: an S shape rising smoothly from 0 to 1'><g font-family='monospace'><line x1='60' y1='95' x2='320' y2='95' stroke='#E5DFD6' stroke-width='1.5'/><line x1='190' y1='25' x2='190' y2='110' stroke='#E5DFD6' stroke-width='1'/><path d='M70 92 C 150 92, 165 40, 300 38' fill='none' stroke='#7C6DAA' stroke-width='2.5'/><text x='120' y='112' font-size='11' fill='#6B645E' text-anchor='middle'>→ near 0</text><text x='270' y='30' font-size='11' fill='#5E5191' text-anchor='middle'>→ near 1</text><text x='400' y='70' font-size='12' fill='#5E5191' font-family='monospace'>sigmoid → (0,1)</text></g></svg>",
  note:"<b>Sigmoid: a soft dimmer.</b> An S-curve that squashes any number into 0…1 — a smooth on/off. <code>sigmoid(0)=0.5</code>. Useful for probabilities, but it flattens (saturates) at the ends."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Two linear layers W1 then W2 collapse into a single matrix W1 times W2'><g font-family='monospace' font-size='13' text-anchor='middle'><rect x='40' y='45' width='60' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='70' y='65' fill='#5E5191'>W₁</text><text x='115' y='65' fill='#6B645E'>then</text><rect x='150' y='45' width='60' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='180' y='65' fill='#5E5191'>W₂</text><text x='250' y='65' fill='#6B645E'>=</text><rect x='290' y='40' width='140' height='40' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='360' y='65' fill='#C93B3B' font-weight='bold'>one matrix W₁@W₂</text><text x='260' y='105' fill='#C93B3B' font-size='11'>no activation → two layers become one</text></g></svg>",
  note:"<b>The problem: collapse.</b> Stacking two linear layers, <code>(x@W₁)@W₂ = x@(W₁@W₂)</code>. Since <code>W₁@W₂</code> is just another matrix, the two layers are secretly one. Depth without a bend is an illusion."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Inserting ReLU between the two layers stops the collapse'><g font-family='monospace' font-size='13' text-anchor='middle'><rect x='30' y='45' width='55' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='57' y='65' fill='#5E5191'>W₁</text><rect x='100' y='45' width='70' height='30' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2.5'/><text x='135' y='65' fill='#1a5c38' font-weight='bold'>ReLU</text><rect x='185' y='45' width='55' height='30' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='212' y='65' fill='#5E5191'>W₂</text><text x='275' y='65' fill='#6B645E'>≠</text><rect x='305' y='45' width='150' height='30' rx='5' fill='#FDF9F3' stroke='#E5DFD6' stroke-width='2' stroke-dasharray='4,3'/><text x='380' y='65' fill='#6B645E'>any single W</text><text x='250' y='105' fill='#2D8B55' font-size='11'>a bend in the middle → cannot be folded flat</text></g></svg>",
  note:"<b>The cure: a bend between layers.</b> Put <code>ReLU</code> between W₁ and W₂ and the whole thing is no longer linear — it can't be collapsed into one matrix. Now the second layer builds on the first's non-linear features."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Stacking many linear-plus-ReLU layers carves a complex bent shape'><g font-family='monospace'><line x1='50' y1='100' x2='330' y2='100' stroke='#E5DFD6' stroke-width='1.5'/><path d='M60 95 L110 60 L160 80 L210 35 L260 70 L310 45' fill='none' stroke='#2A7B9B' stroke-width='2.5'/><text x='190' y='122' font-size='11' fill='#6B645E' text-anchor='middle'>many linear+ReLU layers</text><text x='420' y='70' font-size='12' fill='#1F6280' font-family='monospace'>complex shapes</text></g></svg>",
  note:"<b>Stack them and you can bend anything.</b> Each layer is <code>x@W + b</code> then ReLU. With enough of these, a network can approximate almost any shape — straight pieces joined at bends. That flexibility is what depth buys you, once the activation is there."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='ReLU used in hidden layers, sigmoid or softmax at the output'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='45' width='200' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='140' y='67' fill='#1a5c38' font-weight='bold'>hidden layers → ReLU (cheap, no vanishing)</text><rect x='280' y='45' width='200' height='34' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='380' y='67' fill='#5E5191' font-weight='bold'>output → sigmoid / softmax</text><text x='260' y='104' fill='#6B645E'>the right activation in the right place</text></g></svg>",
  note:"<b>Why labs default to ReLU.</b> It's one <code>max</code> (fast) and it avoids the vanishing-gradient problem that stalls training in deep sigmoid stacks (gradients come on Day 5). Sigmoid/softmax sit at the output for probabilities. The activation is the quiet reason deep learning works at all."}
];
@@@ region name=QS
var QS=[
 {q:'1. What does an activation function add to a network?',
  opts:['More parameters','Non-linearity (a bend)','Faster matmuls','A bias term'],
  ans:1, fb:'Right. The activation is the non-linear bend; without it, layers stay linear and depth is wasted.'},
 {q:'2. With NO activation, ten stacked linear layers behave like:',
  opts:['a much more powerful model','a single linear layer','a random function','a sigmoid'],
  ans:1, fb:'Right. (x W₁)W₂…W₁₀ = x (W₁…W₁₀) — one combined matrix. Depth buys nothing without a bend.'},
 {q:'3. <code>ReLU(z)</code> equals:',
  opts:['1/(1+e^(−z))','max(0, z)','z²','−z'],
  ans:1, fb:'Right. ReLU keeps positives and zeroes negatives: max(0, z).'},
 {q:'4. Your network trains without any error, but the loss stops dropping early and stays high. You log the hidden ReLU outputs and find that most of them are exactly <code>0</code> for every input in the batch. What is the most likely explanation?',
  opts:['The loss is high because sigmoid was used at the output, which is always a bug','Many ReLU units are "dead": their input stays negative for all inputs, so max(0, z) is always 0 and they can no longer learn — the network is effectively much smaller than it looks','The GPU ran out of memory, which always makes ReLU output zero','A high loss with zero activations means the network has already converged'],
  ans:1, fb:'Right. A ReLU that always sees a negative input outputs a flat 0 for every example — a dead unit. Many dead units mean much of the network does nothing, so the loss plateaus. You look at activation statistics (fraction of zeros), not at a crash, because the failure is silent.'}
];
