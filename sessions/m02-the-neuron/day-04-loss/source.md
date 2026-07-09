---
quest_id: wf3-d01-loss
donor: m02-day-04.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 2 · Day 4 — Loss Functions"
spine: "golf score"
notebook_yardstick: 00-neural-networks/fundamentals/06_loss_functions.ipynb
---

@@@ region name=title
<title>Module 2 · Day 4 — Loss Functions</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M2 Day 4</div>
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
<a class="lnav prev" href="../review-part-a.html"><span class="d">← Prev</span><span class="t">M2 · Review Gate</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-05-gradients-backprop/lesson.html"><span class="d">Next →</span><span class="t">Gradients &amp; Backpropagation</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 2 · Train · Day 4</span>
      <h1>Loss Functions<span class="sub">Giving the Network a Score to Beat</span></h1>
      <p class="lede">Here is the entire goal of training a model, in one sentence: <strong>make one number smaller</strong>. That number is the <strong>loss</strong> — a score for how <strong>wrong</strong> the model's prediction is, where lower is better and 0 is perfect. Your network can already make a prediction (Day 3). Today you give it the score it spends the rest of its life trying to beat.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>By the time you close this tab, "training" will just mean "push the loss down": you'll say what a loss measures, compute mean-squared-error by hand, and see why classification uses cross-entropy (the same log from M1 Day 5).</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">What a loss function is</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 3 (a forward pass produces a prediction) and M1 Day 5 (logs). <b>Nothing new needed</b> — today we just score the prediction.</div></div>

      <h4>First, the picture — hold this before any definition</h4>
      <p>Think of a round of golf. Your score is the number of strokes, and lower is always better — a perfect hole is a low number, and the whole game is trying to shoot lower next time. A network's <strong>loss</strong> is exactly that golf score: one number saying how far off its prediction landed, and training is just "shoot a lower score every round." Keep that picture; the term below — <em>loss function</em> — is only the rule that hands you the score.</p>

      <h4>The words you'll meet today — plain English first</h4>
      <table style="width:100%;border-collapse:collapse;margin:12px 0;font-size:.86rem"><thead><tr style="text-align:left;border-bottom:2px solid var(--line2)"><th style="padding:.4rem .6rem;color:var(--k)">Word</th><th style="padding:.4rem .6rem;color:var(--muted)">Plain-English meaning — before we use it</th></tr></thead><tbody>
      <tr style="border-bottom:1px solid var(--line)"><td style="padding:.4rem .6rem;color:var(--ink);font-weight:600">loss / loss function</td><td style="padding:.4rem .6rem;color:var(--ink2)">One number that says how wrong a prediction is; lower is better and 0 means perfect.</td></tr>
      <tr style="border-bottom:1px solid var(--line)"><td style="padding:.4rem .6rem;color:var(--ink);font-weight:600">MSE (mean squared error)</td><td style="padding:.4rem .6rem;color:var(--ink2)">Measure how far each guess is off, square it, then average — the score for predicting numbers.</td></tr>
      <tr style="border-bottom:1px solid var(--line)"><td style="padding:.4rem .6rem;color:var(--ink);font-weight:600">cross-entropy</td><td style="padding:.4rem .6rem;color:var(--ink2)">The score for choosing a category; big when the model is confidently wrong, near 0 when confidently right.</td></tr>
      <tr style="border-bottom:1px solid var(--line)"><td style="padding:.4rem .6rem;color:var(--ink);font-weight:600">logits</td><td style="padding:.4rem .6rem;color:var(--ink2)">The raw output scores of the model, before they are turned into probabilities.</td></tr>
      <tr style="border-bottom:1px solid var(--line)"><td style="padding:.4rem .6rem;color:var(--ink);font-weight:600">softmax</td><td style="padding:.4rem .6rem;color:var(--ink2)">A step that turns a list of raw scores into probabilities that add up to 1.</td></tr>
      <tr><td style="padding:.4rem .6rem;color:var(--ink);font-weight:600">training</td><td style="padding:.4rem .6rem;color:var(--ink2)">Changing the model's weights so the loss number gets smaller.</td></tr>
      </tbody></table>

      <h4>What is a loss function?</h4>
      <p>A <span class="term" data-tip="A single number measuring how wrong a prediction is, by comparing it to the true answer. Lower = better. Training minimizes it.">loss function</span> takes the model's prediction and the true answer, and returns <strong>one number</strong>: how wrong the prediction is. Lower is better; a perfect prediction gives a loss near 0.</p>

      <h4>Why does it exist?</h4>
      <p>A network can't improve if it can't tell how badly it's doing. The loss is that score. <span class="term" data-tip="Adjusting the weights so the model makes better predictions. It works by making the loss number smaller.">Training</span> is nothing more than: change the weights to make the loss smaller. So the loss is the target the whole training process aims at.</p>

      <h4>The two you'll meet most</h4>
      <ul>
        <li><span class="term" data-tip="Mean Squared Error: average of the squared differences between prediction and target. Used when predicting numbers (regression).">MSE</span> (mean squared error) — for predicting <strong>numbers</strong> (a price, a temperature).</li>
        <li><span class="term" data-tip="A loss for classification: -log(probability the model gave the correct class). Small when confident and right.">Cross-entropy</span> — for predicting a <strong>category</strong> (cat vs dog, which word comes next).</li>
      </ul>
      <p>Both boil the gap between "what the model said" and "the truth" down to one number. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> every GPT-style model — 10 weights or 175 billion — is trained to shrink the exact same cross-entropy loss you're about to compute today.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> 大家都知道训练能让模型"变好"，却说不清到底在优化哪个东西 — people sense training "improves" the model but can't name the one thing it optimizes. In plain terms: everything in training reduces to making a single number — the loss — smaller. Pick that number and "learning" becomes concrete: slide it down.</div></div>
      <button class="gotit" type="button">Got what a loss is — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">A golf score</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>You already have the golf picture from Section 1 — here it is as two cards, one for the score itself and one for how MSE measures it:</p>
      <div class="relate">
        <div class="card"><span class="big">⛳</span><h5>Loss = strokes in golf</h5><p>Lower is better; 0 is perfect. Each prediction is a shot, and the loss says how far from the hole you landed. Training tries to shoot a lower score every round.</p></div>
        <div class="card"><span class="big">📏</span><h5>MSE = distance, squared</h5><p>Measure how far off each guess is, square it (so misses in either direction count, and big misses hurt a lot), then average. One tidy number for the whole batch.</p></div>
      </div>
      <p><strong>In one line:</strong> the loss is a "how wrong" score the network wants to drive toward zero.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: in golf you just record the score; here the loss also has to be <em>smooth</em> so we can compute how to improve it (that's tomorrow — gradients). A jagged score wouldn't tell us which way to adjust.</div></div>
      <p><b>直觉 / Intuition:</b> 直觉上，loss 就是考卷上的"扣分" — 错得越离谱扣得越多，全对就是 0 扣分。The technical point: the loss maps (prediction, truth) to one non-negative "how wrong" number, and it's built to be smooth so tomorrow's gradients can point to which way each weight should move.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The two formulas, and why loss runs everything</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The two formulas</h4>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>MSE = mean( (prediction − target)² )</code> — average squared error, for numbers.<br><code>cross-entropy = − log( p_correct )</code> — for a category, where <code>p_correct</code> is the probability the model assigned to the true class.</div></div>
      <p>Why square in MSE? So positive and negative errors both count, and big misses are punished more. Why the log in cross-entropy? A confident-and-correct model (<code>p_correct</code> near 1) gives <code>−log(1) = 0</code>; a confident-and-wrong one (<code>p_correct</code> near 0) gives a huge loss. The log turns "probability" into "surprise."</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — MSE, <code>mean((pred − target)²)</code></b>
      <br><b>1 · In words:</b> measure how far each prediction is from its target, square it (so misses in either direction count and big misses hurt more), then average over the batch.
      <br><b>2 · The formula:</b> <code>MSE = (1/N) Σᵢ (predᵢ − targetᵢ)²</code>. <code>N</code> = number of examples in the batch; the sum runs over them; the <code>(1/N)</code> makes it a <b>mean</b>.
      <br><b>3 · Tiny numbers:</b> <code>pred=[2.5, 0.0, 2.0]</code>, <code>target=[3.0, −0.5, 2.0]</code>. Errors <code>[−0.5, 0.5, 0.0]</code>; squared <code>[0.25, 0.25, 0.0]</code>; sum <code>0.5</code>; mean <code>= 0.5/3 ≈ 0.167</code>.
      <br><b>4 · Sanity check:</b> MSE is <code>0</code> only when every prediction equals its target, and it can never be negative (squares). Because it is a <b>mean over the batch</b>, doubling the batch with the same errors leaves the loss unchanged — a fact Day 6 leans on.</div></div>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — cross-entropy, <code>−log(p_correct)</code></b>
      <br><b>1 · In words:</b> a loss should be near 0 when the model is confident and right, and huge when it's confident and wrong. Cross-entropy measures the "surprise" of the true answer under the model.
      <br><b>2 · The formula:</b> <code>CE = −log(p_correct)</code>, where <code>p_correct</code> = the probability the model assigned to the true class (a number in (0, 1]).
      <br><b>3 · Tiny numbers:</b> <code>p=0.3 → −log(0.3) ≈ 1.204</code> (large); <code>p=0.6 → ≈ 0.511</code>; <code>p=0.9 → ≈ 0.105</code> (small). As the guess improves, the loss drops — the same order the playground and artifact print.
      <br><b>4 · Sanity check:</b> <code>−log(1) = 0</code> (perfectly confident and correct → zero loss); as <code>p → 0</code> the loss → <code>+∞</code> (confidently wrong is punished without limit). Loss is never negative.</div></div>
      <table style="width:100%;border-collapse:collapse;margin:16px 0;font-size:.88rem">
        <thead><tr style="text-align:left;border-bottom:2px solid var(--line2)"><th style="padding:.5rem .6rem"></th><th style="padding:.5rem .6rem;color:var(--q)">MSE — regression</th><th style="padding:.5rem .6rem;color:var(--k)">Cross-entropy — classification</th></tr></thead>
        <tbody>
          <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Use when</td><td style="padding:.5rem .6rem;color:var(--ink2)">target is a number (price, temperature)</td><td style="padding:.5rem .6rem;color:var(--ink2)">target is a category (cat/dog, next token)</td></tr>
          <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Formula</td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>mean((pred−target)²)</code></td><td style="padding:.5rem .6rem;color:var(--ink2)"><code>−Σ yₖ log pₖ</code> (= <code>−log p_correct</code> for a one-hot label)</td></tr>
          <tr style="border-bottom:1px solid var(--line)"><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Output it pairs with</td><td style="padding:.5rem .6rem;color:var(--ink2)">none (a raw number)</td><td style="padding:.5rem .6rem;color:var(--ink2)">softmax (multi-class) / sigmoid (binary)</td></tr>
          <tr><td style="padding:.5rem .6rem;color:var(--ink);font-weight:600">Gradient when very wrong</td><td style="padding:.5rem .6rem;color:var(--ink2)">can go weak/flat if paired with a squashing output</td><td style="padding:.5rem .6rem;color:var(--ink2)">stays strong (∝ the error) — learning doesn't stall</td></tr>
        </tbody>
      </table>

      <h4>Why a frontier lab cares</h4>
      <p>Loss is <em>the</em> number. Training minimizes it. Every GPT-style model from the teaser above made exactly the choice you're about to see argued below: cross-entropy over MSE. It picks cross-entropy because next-token prediction is choosing a category, not a number. The famous <strong>scaling laws</strong> (M1 Day 5, and Chinchilla) plot <strong>loss vs compute</strong> — the whole game is predicting how low the loss goes as you spend more. "Lower loss = a better model" is the assumption under everything.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> forward pass → prediction → compare to truth with a loss → one "how wrong" number → training pushes it down. Picture that one number as the <b>height of a landscape</b> over all the weights — tomorrow we find which way is downhill and take a step (gradients).</div></div>

      <div class="callout c-info"><span class="ic">🔬</span><div><b>Going deeper — logits → probabilities.</b> The network's last layer emits raw <span class="term" data-tip="The raw, unsquashed output scores of a model, before softmax turns them into probabilities.">logits</span>; <b>softmax</b> turns a vector of logits into probabilities that sum to 1, and cross-entropy then reads off <code>p_correct</code>. In real code the softmax and the log are fused and computed straight from the logits (the <b>log-sum-exp</b> trick — subtract the max first) so you never hit <code>log(0)</code> or overflow <code>e^(big)</code>; that is why loss functions take logits, not probabilities. Two honest caveats: the general multi-class form is <code>−Σₖ yₖ log pₖ</code> (it collapses to <code>−log p_correct</code> when the label is one-hot), and a softmax "probability" is <em>not</em> a calibrated confidence — a model can output 0.99 and still be wrong.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>The loss is just one number, so a wrong loss looks exactly like a right one. Two mistakes here damage training without ever throwing an error.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): feeding probabilities where logits are expected.</b> Many cross-entropy functions (like PyTorch's <code>CrossEntropyLoss</code>) expect raw scores called <span class="term" data-tip="The raw, unsquashed output scores of a model, before softmax turns them into probabilities.">logits</span> and apply the softmax themselves. If you run softmax first and pass the resulting probabilities in, softmax gets applied <em>twice</em>. The loss still computes a finite, plausible-looking number — nothing crashes — but it is the wrong number, so training is weaker and slower for no visible reason. A senior engineer catches this in <b>code review</b> by checking exactly what the loss function expects: does it want logits or probabilities, and is there a stray extra softmax before it?</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: which loss you choose for the task.</b> It's really a choice of scorecard — match it to the game you're actually playing, or you drive down a number that doesn't mean what you think. Using <code>MSE</code> on a classification problem runs perfectly and gives a valid number, so it is tempting — but it treats class labels as plain numbers and gives weak, slow-to-move gradients when the model is confidently wrong. <code>Cross-entropy</code> punishes confident mistakes hard (<code>−log(p)</code> blows up as <code>p</code> nears 0), which trains classifiers much faster, but it needs proper probabilities and can be numerically touchy. Picking the loss that matches the task — MSE for regressing a number, cross-entropy for choosing a class — is a <b>design-review</b> decision that quietly sets how well the whole model can learn.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "The loss is the single scalar training minimizes. For regression I use MSE = mean((pred−target)²); for classification, cross-entropy = −log(p_correct), because next-token prediction is picking a category and cross-entropy punishes confident-wrong predictions hard, giving strong gradients. The classic silent bug is a double softmax — passing probabilities into a loss that expects logits — which yields a finite but wrong number with no crash, so I always check whether the loss wants logits or probabilities."</div></div>
      <button class="gotit" type="button">Got why loss matters</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">See your score drop, round by round</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Here's the fun part. <b>Before you run anything, predict:</b> is <code>−log(0.3)</code> bigger or smaller than <code>−log(0.9)</code>? Jot your guess. Then compute cross-entropy for <code>p = 0.3, 0.6, 0.9</code> and watch the three numbers <code>1.204 → 0.511 → 0.105</code> fall as the model grows more confident — the golf score dropping as your aim improves. Pick one path:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m02-the-neuron/day-04-loss/experiment.py</code>. Write <code>mse(pred, target) = np.mean((pred-target)**2)</code> and <code>cross_entropy(p_correct) = -np.log(p_correct)</code>. Print MSE for a close prediction vs a far one, and cross-entropy for <code>p_correct</code> in <code>[0.3, 0.6, 0.9]</code> — confirm the loss drops as the guess improves. Run with <code>python3 sessions/m02-the-neuron/day-04-loss/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 2 Day 4 artifact.

Create sessions/m02-the-neuron/day-04-loss/experiment.py that, with a comment on each step:
1. Defines mse(pred, target) = np.mean((pred - target)**2).
2. Prints mse for a close prediction ([2.5,0.0,2.0] vs [3.0,-0.5,2.0], expect ~0.167) and a far prediction, showing the far one has higher loss.
3. Defines cross_entropy(p_correct) = -np.log(p_correct).
4. Prints cross_entropy for p_correct in [0.3, 0.6, 0.9] (expect ~1.204, 0.511, 0.105) and notes the loss falls as the model gets more confident in the correct class.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>What you should see (check your prediction)</h4>
      <ul>
        <li>Your <code>mse</code> prints a smaller value for a close prediction than for a far one — the golf score is lower when your aim is better.</li>
        <li>Your <code>cross_entropy(p_correct)</code> on <code>p ∈ [0.3, 0.6, 0.9]</code> prints a loss that <em>drops</em> as <code>p</code> rises (≈ 1.204 → 0.511 → 0.105) — exactly the order you predicted.</li>
        <li>You can say in one sentence why cross-entropy, not MSE, is used for classification.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-04-loss/log.md</code>: (1) why squaring the error (MSE) punishes big misses more; (2) why <code>−log(p)</code> blows up when the model is confidently wrong; (3) when you'd pick MSE vs cross-entropy.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 2 · Day 4 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Loss Functions</b>.<br>Next up: <b>Gradients &amp; Backpropagation</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  mse:{html:'<span class="prompt">&gt;&gt;&gt;</span> pred   = np.array([<span class="num">2.5</span>, <span class="num">0.0</span>, <span class="num">2.0</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> target = np.array([<span class="num">3.0</span>, <span class="num">-0.5</span>, <span class="num">2.0</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> error = pred - target\n<span class="ok">array([-0.5,  0.5,  0. ])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.mean(error**<span class="num">2</span>)      <span class="dim"># mean squared error</span>\n'+
    '<span class="hl">0.16666666666666666</span>',
    take:'<b>①  MSE = average of squared errors.</b> Errors [−0.5, 0.5, 0] → squares [0.25, 0.25, 0] → mean 0.167. Squaring makes every error positive and punishes big misses harder. Small here, because the guess is close.'},
  ce:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># true class is #0; model\'s predicted probabilities:</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> p = np.array([<span class="num">0.7</span>, <span class="num">0.2</span>, <span class="num">0.1</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> -np.log(p[<span class="num">0</span>])       <span class="dim"># cross-entropy = -log(prob of true class)</span>\n'+
    '<span class="hl">0.35667494393873245</span>',
    take:'<b>②  Cross-entropy = −log(p of the correct class).</b> The model gave the right class 0.7, so loss = −log(0.7) ≈ 0.36. If it had said 0.99, loss ≈ 0.01; if 0.1, loss ≈ 2.3. That log is from M1 Day 5.'},
  drop:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">for</span> p_true <span class="kw">in</span> [<span class="num">0.3</span>, <span class="num">0.6</span>, <span class="num">0.9</span>]:\n'+
    '<span class="dim">...</span>     print(p_true, round(-np.log(p_true), <span class="num">3</span>))\n'+
    '<span class="ok">0.3 1.204</span>\n'+
    '<span class="ok">0.6 0.511</span>\n'+
    '<span class="ok">0.9 0.105</span>',
    take:'<b>③  Loss falls as the model improves.</b> As confidence in the correct class rises 0.3 → 0.9, the loss drops 1.20 → 0.11. Training\'s entire job is to push this number down.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 100' role='img' aria-label='Prediction vector versus target vector'><g font-family='monospace' font-size='13' text-anchor='middle'><rect x='60' y='34' width='160' height='32' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='140' y='55' fill='#1F6280'>prediction [2.5, 0, 2]</text><rect x='300' y='34' width='160' height='32' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='380' y='55' fill='#1a5c38'>target [3, −0.5, 2]</text><text x='260' y='55' fill='#6B645E'>vs</text></g></svg>",
  note:"<b>Start with what we have.</b> The forward pass gave a <b>prediction</b>; the data gives the <b>target</b> (the true answer). The loss will turn the gap between them into one number."},
 {viz:"<svg viewBox='0 0 520 90' role='img' aria-label='Error equals prediction minus target'><g font-family='monospace' font-size='13' text-anchor='middle'><text x='140' y='50' fill='#2C2A28'>pred − target</text><text x='250' y='50' fill='#6B645E'>=</text><rect x='300' y='30' width='170' height='32' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='385' y='51' fill='#9A7208'>error [−0.5, 0.5, 0]</text></g></svg>",
  note:"<b>Take the gap: the error.</b> Subtract element-wise: <code>[−0.5, 0.5, 0]</code>. But we can't just add these — the −0.5 and +0.5 would cancel and hide real mistakes."},
 {viz:"<svg viewBox='0 0 520 90' role='img' aria-label='Square each error to make it positive'><g font-family='monospace' font-size='13' text-anchor='middle'><text x='120' y='50' fill='#9A7208'>[−0.5, 0.5, 0]²</text><text x='250' y='50' fill='#6B645E'>→</text><rect x='300' y='30' width='170' height='32' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='385' y='51' fill='#C93B3B'>[0.25, 0.25, 0]</text></g></svg>",
  note:"<b>Square each error.</b> Squaring makes every error positive (so they can't cancel) and makes big misses count much more than small ones. Now every mistake adds to the score."},
 {viz:"<svg viewBox='0 0 520 90' role='img' aria-label='Average the squared errors into one MSE number'><g font-family='monospace' font-size='13' text-anchor='middle'><text x='150' y='50' fill='#2C2A28'>mean([0.25, 0.25, 0])</text><text x='300' y='50' fill='#6B645E'>=</text><rect x='340' y='30' width='120' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2.5'/><text x='400' y='52' fill='#1a5c38' font-weight='bold'>MSE = 0.167</text></g></svg>",
  note:"<b>Average → one number.</b> The mean of the squared errors is the <b>MSE</b>: 0.5 / 3 ≈ 0.167. That single value is the loss for numeric predictions. A perfect prediction would give 0."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Cross-entropy uses minus log of the probability of the correct class'><g font-family='monospace' font-size='13' text-anchor='middle'><text x='140' y='45' fill='#5E5191'>correct-class prob p = 0.7</text><text x='140' y='70' fill='#2C2A28'>loss = −log(0.7)</text><text x='300' y='58' fill='#6B645E'>=</text><rect x='335' y='40' width='130' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='400' y='62' fill='#1a5c38' font-weight='bold'>≈ 0.36</text></g></svg>",
  note:"<b>For categories: cross-entropy.</b> When the model outputs probabilities, the loss is <code>−log(p_correct)</code>. Confident-and-right (p→1) → loss→0; confident-and-wrong (p→0) → huge loss. This is the log from M1 Day 5, put to work."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Loss decreases as the prediction improves, the goal of training'><g font-family='monospace'><line x1='60' y1='95' x2='300' y2='95' stroke='#E5DFD6' stroke-width='1.5'/><line x1='60' y1='25' x2='60' y2='95' stroke='#E5DFD6' stroke-width='1'/><path d='M75 40 C 150 55, 200 88, 290 92' fill='none' stroke='#2D8B55' stroke-width='2.5'/><text x='180' y='112' font-size='11' fill='#6B645E' text-anchor='middle'>training steps →</text><text x='30' y='60' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 30 60)'>loss</text><text x='420' y='60' font-size='12' fill='#1a5c38' font-family='monospace'>lower = better</text></g></svg>",
  note:"<b>Why it matters.</b> Training is the act of pushing this number down, step by step. Scaling laws (M1 Day 5) plot loss vs compute to predict how low it can go. 'Lower loss = better model' is the assumption under the entire field. Tomorrow: how we know which way to nudge each weight to lower it."}
];
@@@ region name=QS
var QS=[
 {q:'1. What does a loss function measure?',
  opts:['How fast the model runs','How wrong the prediction is (one number)','How many layers there are','The batch size'],
  ans:1, fb:'Right. The loss is a single "how wrong" number comparing the prediction to the true answer.'},
 {q:'2. A lower loss means:',
  opts:['a worse prediction','a better prediction','more parameters','a bigger learning rate'],
  ans:1, fb:'Right. Lower loss = closer to the truth. A perfect prediction gives a loss near 0.'},
 {q:'3. MSE computes:',
  opts:['the sum of the predictions','the mean of the squared errors','the largest error','the number of classes'],
  ans:1, fb:'Right. MSE = mean((prediction − target)²) — squared so errors do not cancel and big misses hurt more.'},
 {q:'4. Your classifier trains without any error, but it learns slowly and the loss is oddly small even at the start. You notice your code runs <code>softmax</code> on the output and then passes that into a cross-entropy loss that already applies softmax internally. What is the problem?',
  opts:['Nothing — running softmax twice makes the probabilities more accurate','Softmax is being applied twice, so the loss is computed on the wrong (double-squashed) numbers; it still returns a finite value, so training is quietly weakened with no error to see','The model has too few parameters, which is the only cause of slow learning','A small loss always means the model is already perfect'],
  ans:1, fb:'Right. If the loss function expects logits and applies its own softmax, feeding it probabilities applies softmax twice. The number is still finite and plausible, so nothing crashes — but it is the wrong loss, so learning suffers. You check whether the loss wants logits or probabilities, not for a crash.'}
];
