---
quest_id: wf3-d05-lr
donor: m02-day-08.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 2 · Day 8 — Learning-Rate Intuition"
spine: "stepping stone"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 2 · Day 8 — Learning-Rate Intuition</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M2 Day 8</div>
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
<a class="lnav prev" href="../day-07-optimizers/lesson.html"><span class="d">← Prev</span><span class="t">Optimizers (SGD, Momentum, Adam)</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-09-train-val-test/lesson.html"><span class="d">Next →</span><span class="t">Train / Val / Test &amp; Overfitting</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 2 · Train · Day 8</span>
      <h1>Learning-Rate Intuition<span class="sub">The one knob that makes or breaks training</span></h1>
      <p class="lede">Get one number wrong and a training run that cost a fortune either <strong>explodes</strong> into garbage or <strong>crawls</strong> for hours and never arrives. Get it right and the loss slides down smooth as glass. That number is the <strong>learning rate</strong> — how big a step training takes on each update — and it is the single most important dial in the whole run.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div>By the time you close this tab, you'll be able to describe what happens at a too-big, too-small, and just-right learning rate, read a loss curve to spot each case, and say in one line what warmup and decay do.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">The size of each step</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> gradient descent and the update rule <code>w ← w − lr × grad</code> (Day 5), and optimizers (Day 7). Today we zoom in on the number <code>lr</code>. <b>Nothing new needed.</b></div></div>

      <h4>First, the picture — hold this before any number</h4>
      <p>Imagine crossing a stream by hopping from stone to stone. The gradient already told you which way the far bank is; the only thing left to choose is <em>how big each hop is</em>. Hop too far and you overshoot the next stone into deep water; shuffle too timidly and the sun sets before you cross; get the size right and you land square on each stone and reach the bank quickly. That hop size is the <strong>learning rate</strong>. Keep that picture; the rest of the lesson is just three hop sizes and what each one does.</p>

      <h4>What is the learning rate?</h4>
      <p>The <span class="term" data-tip="How big each weight update is. In the rule w ← w − lr × grad, it is the number lr that scales the gradient. Often written α or lr.">learning rate</span> is the number that decides how far you move on each step. The gradient tells you which way is downhill. The learning rate tells you how big a step to take in that direction. (The exact update rule — <code>w ← w − lr × grad</code> — and the math for when a rate diverges are in Section 4.)</p>

      <h4>It is a choice, not something learned</h4>
      <p>The learning rate is a <span class="term" data-tip="A setting you pick before training, not a value the model learns: learning rate, batch size, number of epochs.">hyperparameter</span> — you pick it before training starts. The model never learns it for you. Because it multiplies every single update, it is the number that most decides whether a run works. Three cases matter: too big, too small, and just right. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> GPT-3, Chinchilla, and Llama 3 all publish their exact learning-rate warmup-and-decay schedule as a named part of how they were trained — not a minor detail, a headline decision.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> learning rate 只是一个小小的数字，很多人以为"随便设一个就行" — the learning rate is just one small number, so it's tempting to treat it as set-and-forget. In plain terms: it sets the <i>size</i> of every step, and it's the difference between a loss that slides down smoothly, one that crawls for hours, and one that explodes to infinity. It's the single most important number in a run.</div></div>
      <button class="gotit" type="button">Got it — let's tune it</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">Crossing a stream on stepping stones</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>You already have the stepping-stones picture from Section 1 — here are the three hop sizes side by side, and what each one does:</p>
      <div class="relate">
        <div class="card"><span class="big">🐘</span><h5>Steps too big</h5><p>You leap too far, miss the stone, and splash into deeper water on the other side — worse than where you started. The bigger the leap, the deeper you land. This is the loss exploding.</p></div>
        <div class="card"><span class="big">🐜</span><h5>Steps too small</h5><p>You shuffle forward a few centimetres at a time. You will get there eventually, but the sun sets first. This is training that crawls and wastes time.</p></div>
      </div>
      <div class="relate">
        <div class="card"><span class="big">🚶</span><h5>Steps just right</h5><p>Each step lands squarely on the next stone, a little closer to the far bank. Smooth, steady, done in good time. This is a healthy learning rate.</p></div>
        <div class="card"><span class="big">🎚️</span><h5>One dial to set</h5><p>The stones (the loss curve) are fixed. Your only choice is step size — the learning rate. Get that one dial right and the crossing is easy.</p></div>
      </div>
      <p><strong>In one line:</strong> the learning rate is your step size across the stream — too big and you overshoot into deep water, too small and you never arrive.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: real stones sit at fixed spots, but the loss "ground" is curved and changes as you move. A step that is safe near the far bank can be too big at the start, which is exactly why runs often <b>shrink the learning rate over time</b> (decay, coming up in section 4).</div></div>
      <p><b>直觉 / Intuition:</b> 直觉上，learning rate 就是过河时"每一步迈多大" — 迈太大掉进深水（发散），迈太小永远走不到对岸（太慢）。The technical point: the learning rate scales the step <code>w ← w − lr·grad</code>; too large and the loss diverges, too small and it converges but painfully slowly.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">Reading the loss curve, and schedules</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>Three cases, read straight off the loss curve</h4>
      <p>You do not have to guess the learning rate blind. You watch the <span class="term" data-tip="A plot of the loss value over training steps. Its shape tells you if the learning rate is too big, too small, or right.">loss curve</span> — the loss plotted against training steps — and its shape tells you which case you are in.</p>
      <ul>
        <li><b>Too small:</b> the curve drops very slowly, almost a straight gentle slope. It will get there, but it wastes hours of compute.</li>
        <li><b>Just right:</b> the curve drops fast at first, then flattens smoothly near the bottom. Clean and steady.</li>
        <li><b>Too big:</b> the curve does not fall. It bounces, or it shoots <b>upward</b> to bigger and bigger numbers (often shown as <code>NaN</code> — "not a number"). Training has diverged.</li>
      </ul>

      <h4>The math behind "too big"</h4>
      <p><b>In words:</b> for a simple bowl-shaped loss, each step multiplies the distance-to-the-bottom by a fixed factor. If that factor is bigger than 1 in size, the distance grows every step instead of shrinking.</p>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>(distance to bottom) ← (1 − lr × c) × (distance to bottom)</code><br><span style="color:var(--muted)">c = curvature of the loss (a fresh symbol — not Day 7's velocity v) · you converge only while |1 − lr × c| &lt; 1</span></div></div>
      <p><b>Worked example:</b> in our toy the curvature is <code>c = 8</code>. At <code>lr = 0.1</code> the factor is <code>1 − 0.8 = 0.2</code> — each step keeps only 20% of the gap, so it shrinks fast. At <code>lr = 0.6</code> the factor is <code>1 − 4.8 = −3.8</code> — bigger than 1 in size, so the gap grows almost 4× every step. That is the explosion you saw in the playground.</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — when does a learning rate converge?</b>
      <br><b>1 · In words:</b> on a simple bowl, each step multiplies the gap to the bottom by a fixed factor; you converge only if that factor is smaller than 1 in size.
      <br><b>2 · The formula:</b> <code>gap ← (1 − lr·c)·gap</code>, converging while <code>|1 − lr·c| &lt; 1</code>. <code>lr</code> = learning rate; <code>c</code> = curvature of the loss (here <code>c = 8</code> for the toy <code>pred=w·x</code> problem).
      <br><b>3 · Tiny numbers:</b> <code>lr=0.005 → 1 − 0.04 = 0.96</code> (shrinks only 4% a step → crawls); <code>lr=0.1 → 0.2</code> (fast); <code>lr=0.6 → 1 − 4.8 = −3.8</code>, size <code>3.8 &gt; 1</code> → the gap grows ~4× a step and explodes.
      <br><b>4 · Sanity check:</b> the tipping point is <code>|1 − lr·c| = 1</code>, i.e. <code>lr = 2/c = 0.25</code> here — below it converges, above it diverges, exactly the crossover you see in the playground.</div></div>

      <h4>Schedules: change the rate as you go (awareness)</h4>
      <p>Real runs rarely keep one fixed rate. Two ideas you will meet later:</p>
      <ul>
        <li><b><span class="term" data-tip="Starting with a very small learning rate and raising it over the first few hundred steps, so an unstable early model does not blow up.">Warmup</span>:</b> start tiny and raise the rate over the first few hundred steps. Early on the model is fragile, so a big step could blow it up. Warmup eases it in.</li>
        <li><b><span class="term" data-tip="Slowly lowering the learning rate as training goes on, so the model can settle gently into a good minimum.">Decay</span>:</b> slowly lower the rate as training goes on. Big steps early cover ground fast; small steps late let the model settle gently into the valley.</li>
      </ul>
      <div class="callout c-info"><span class="ic">🔬</span><div><b>Three things that shift the right rate.</b> (1) <b>Curvature isn't one number.</b> In a real model the <code>c</code> above is a whole range of curvatures across directions; the steepest direction (the largest Hessian eigenvalue) sets the largest stable rate, so a single global <code>lr</code> is always a compromise. (2) <b>Unscaled inputs hurt.</b> If input features live on wildly different scales, the loss bowl becomes lopsided (ill-conditioned) and no single <code>lr</code> suits every weight — normalizing the inputs is the cheapest fix. (3) <b>Batch size and <code>lr</code> move together.</b> A larger batch gives a cleaner gradient, so it usually wants a proportionally larger <code>lr</code> (the "linear scaling rule") — the knobs from Days 6, 7, and 8 are coupled, not independent.</div></div>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Why a frontier lab cares:</b> a good learning rate (and its warmup + decay schedule) is one of the highest-value knobs in a huge training run — exactly why GPT-3, Chinchilla, and Llama 3 (the teaser above) all treat their schedule as a headline methodology decision, not an afterthought. A rate slightly too high can waste millions of dollars of compute by diverging; too low wastes it by crawling. This is a core part of the craft you will see in the scaling and JAX modules.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>A learning rate that blows up is easy to spot. The dangerous case is the one where nothing looks wrong.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): a rate high enough to hurt, but not high enough to explode.</b> A too-big rate that sends the loss to <code>NaN</code> is loud — you notice at once. The quiet failure is a rate that is a bit too high: the loss still drops fast and looks healthy, but it settles at a <em>worse</em> value than a smaller rate would have reached, because the big steps keep bouncing around the bottom of the valley instead of settling into it. On the stream, it's a stride just long enough that you keep splashing at the rim of each stone instead of landing square — you do cross, but soaked and worse off than a cleaner step would leave you. Nothing errors, the curve looks fine, and you ship a model that is silently weaker than it could be. A senior engineer catches this in <b>code review</b> (or in the run logs) by comparing final loss across a few learning rates, not by trusting that "loss went down, so the rate was fine."</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: fast steps vs. stable steps.</b> A larger learning rate covers ground quickly and saves compute, but risks bouncing or diverging. A smaller rate is stable and lands in a better final spot, but crawls and burns hours of expensive GPU time. This is exactly why schedules exist — <b>warmup</b> to ease a fragile fresh model in, then <b>decay</b> to settle gently at the end. Where to set the peak rate and how to shape the schedule is a <b>design-review</b> decision that trades training speed against final quality and stability.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "The learning rate scales each step, <code>w ← w − lr·grad</code>. On a quadratic bowl of curvature <code>c</code>, a step multiplies the gap-to-optimum by <code>(1 − lr·c)</code>, so you converge only while <code>|1 − lr·c| &lt; 1</code> — too small crawls, too big diverges, with a hard tipping point at <code>lr = 2/c</code>. That's why real runs use a schedule: warmup to ease a fresh, fragile model in, then decay to settle at the end. The failure I watch for is a rate that's stable early but too large once curvature rises — the loss looks fine, then suddenly spikes to NaN."</div></div>
      <button class="gotit" type="button">Got the three cases</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Predict which hop size crosses — then run all three</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Here's the test worth running. <b>Before you run it, predict:</b> over 8 full steps, does <code>lr=0.005</code> ever reach <code>w=3</code>? Does <code>lr=0.6</code> settle down or keep growing? Jot your guesses. Then run all three rates and watch which stay under the <code>|1 − lr·c| = 1</code> line — <code>0.005</code> still crawling, <code>0.1</code> landing square on <code>w≈3</code>, and <code>0.6</code> splashing off into ever-bigger numbers. Pick one path:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m02-the-neuron/day-08-learning-rate/experiment.py</code>. Fit <code>pred = w·x</code> to <code>x=2, y=6</code> from <code>w=1.0</code>. Run the same 8-step loop three times with <code>lr = 0.005</code>, <code>0.1</code>, and <code>0.6</code>: <code>grad=2*(w*x-y)*x</code>; <code>w = w - lr*grad</code>; print step, w, loss. Confirm 0.005 crawls, 0.1 reaches <code>w ≈ 3</code>, and 0.6 blows up. Run with <code>python3 sessions/m02-the-neuron/day-08-learning-rate/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 2 Day 8 artifact.

Create sessions/m02-the-neuron/day-08-learning-rate/experiment.py that, with a comment on each step:
1. Sets a tiny model pred = w*x, with x=2.0, target y=6.0 (ideal w=3).
2. Defines a run(lr, steps) function: w=1.0; for each step do grad=2*(w*x-y)*x; w = w - lr*grad; print step, round(w,4), round((w*x-y)**2,4).
3. Calls run(0.005, 8), run(0.1, 8), and run(0.6, 8).
4. Shows that lr=0.005 crawls (loss still high), lr=0.1 reaches w≈3 and loss≈0, and lr=0.6 diverges (loss grows to huge numbers).
5. Adds a comment: the loss shrinks only while |1 - lr*curvature| < 1; here the curvature is 8, so lr must stay below 0.25.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>What you should see (check your prediction)</h4>
      <ul>
        <li>The same 8-step loop runs three times from <code>w=1.0</code> at <code>lr = 0.005, 0.1, 0.6</code>, printing step, <code>w</code>, and loss.</li>
        <li><code>0.005</code> crawls (loss barely moves), <code>0.1</code> reaches <code>w ≈ 3</code>, and <code>0.6</code> blows up (loss grows each step).</li>
        <li>You can name roughly where this toy tips from converging to diverging (near <code>lr = 2/c = 0.25</code>) and why.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-08-learning-rate/log.md</code>: (1) what the learning rate controls, in one sentence; (2) at what lr this toy tips from converging to diverging, and why; (3) why real runs warm up and then decay the rate.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 2 · Day 8 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Learning-Rate Intuition</b>.<br>Next up: <b>Train / Val / Test &amp; Overfitting</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  toosmall:{html:'<span class="prompt">&gt;&gt;&gt;</span> x, y, lr = <span class="num">2.0</span>, <span class="num">6.0</span>, <span class="num">0.005</span>   <span class="dim"># tiny step</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = <span class="num">1.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">for</span> step <span class="kw">in</span> range(<span class="num">5</span>):\n'+
    '<span class="dim">...</span>     grad = <span class="num">2</span>*(w*x-y)*x; w = w - lr*grad\n'+
    '<span class="dim">...</span>     print(step, round(w,<span class="num">3</span>), round((w*x-y)**<span class="num">2</span>,<span class="num">3</span>))\n'+
    '<span class="dim">0 1.08  14.746</span>\n<span class="dim">1 1.157 13.589</span>\n<span class="dim">2 1.231 12.524</span>\n<span class="dim">3 1.301 11.542</span>\n<span class="dim">4 1.369 10.637</span>',
    take:'<b>①  Too small — it crawls.</b> After 5 steps w has barely moved from 1.0 to 1.37 and the loss is still 10.6. Right direction, but you would need hundreds of steps. Wasted compute.'},
  justright:{html:'<span class="prompt">&gt;&gt;&gt;</span> lr = <span class="num">0.1</span>                     <span class="dim"># a healthy step</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = <span class="num">1.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">for</span> step <span class="kw">in</span> range(<span class="num">5</span>):\n'+
    '<span class="dim">...</span>     grad = <span class="num">2</span>*(w*x-y)*x; w = w - lr*grad\n'+
    '<span class="dim">...</span>     print(step, round(w,<span class="num">3</span>), round((w*x-y)**<span class="num">2</span>,<span class="num">3</span>))\n'+
    '<span class="ok">0 2.6   0.64</span>\n<span class="ok">1 2.92  0.026</span>\n<span class="ok">2 2.984 0.001</span>\n<span class="ok">3 2.997 0.0</span>\n<span class="ok">4 2.999 0.0</span>',
    take:'<b>②  Just right — smooth descent.</b> w reaches the ideal 3.0 in about 3 steps and the loss slides to 0. Each step covers real ground without overshooting. This is the learning rate you want.'},
  toobig:{html:'<span class="prompt">&gt;&gt;&gt;</span> lr = <span class="num">0.6</span>                     <span class="dim"># way too big</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> w = <span class="num">1.0</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">for</span> step <span class="kw">in</span> range(<span class="num">4</span>):\n'+
    '<span class="dim">...</span>     grad = <span class="num">2</span>*(w*x-y)*x; w = w - lr*grad\n'+
    '<span class="dim">...</span>     print(step, round(w,<span class="num">3</span>), round((w*x-y)**<span class="num">2</span>,<span class="num">3</span>))\n'+
    '<span class="bad">0 10.6    231.04</span>\n<span class="bad">1 -25.88  3336.22</span>\n<span class="bad">2 112.74  48174.98</span>\n<span class="bad">3 -414.03 695646.74</span>',
    take:'<b>③  Too big — it explodes.</b> Each step jumps past the bottom to a worse point on the far side. w bounces to bigger and bigger numbers and the loss grows without limit. Training has diverged. (This preview stops at 4 steps because it has already blown up; your Produce run does the full 8 steps for all three rates — you just see more of the same explosion here.)'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='The learning rate scales every update step'><g font-family='monospace' font-size='13' text-anchor='middle'><rect x='120' y='38' width='280' height='36' rx='7' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='61' fill='#1a5c38'>w ← w − <tspan fill='#C93B3B' font-weight='bold'>lr</tspan> × grad</text><path d='M300 78 L300 94' stroke='#C93B3B' stroke-width='1.3' marker-end='url(#a5)'/><text x='300' y='106' font-size='10' fill='#C93B3B'>the step size — your one knob</text><defs><marker id='a5' markerWidth='7' markerHeight='7' refX='5' refY='3' orient='auto'><path d='M0 0 L6 3 L0 6 z' fill='#C93B3B'/></marker></defs></g></svg>",
  note:"<b>The learning rate scales every step.</b> The gradient picks the downhill direction; <b>lr</b> decides how far you go. Because it multiplies every update, this one number matters more than almost any other."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Too small: the loss curve drops very slowly'><g font-family='monospace'><line x1='60' y1='105' x2='470' y2='105' stroke='#E5DFD6' stroke-width='1.5'/><path d='M70 40 C 180 55, 320 78, 460 95' fill='none' stroke='#C99A12' stroke-width='2.5'/><text x='260' y='122' font-size='11' fill='#6B645E' text-anchor='middle'>training steps →</text><text x='40' y='68' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 68)'>loss</text><text x='360' y='55' font-size='11' fill='#9A7208'>still high</text></g></svg>",
  note:"<b>Too small — it crawls.</b> The loss slopes down only a little each step. In our toy, 5 steps at lr = 0.005 barely moved the loss from 16 to 10.6. It works, but wastes huge amounts of time and compute."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Just right: the loss curve drops fast then flattens'><g font-family='monospace'><line x1='60' y1='105' x2='470' y2='105' stroke='#E5DFD6' stroke-width='1.5'/><path d='M70 35 C 120 90, 200 102, 460 104' fill='none' stroke='#2D8B55' stroke-width='2.5'/><text x='260' y='122' font-size='11' fill='#6B645E' text-anchor='middle'>training steps →</text><text x='40' y='68' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 68)'>loss</text><text x='400' y='96' font-size='12' fill='#1a5c38'>→ 0</text></g></svg>",
  note:"<b>Just right — smooth descent.</b> The loss drops fast at first, then flattens near the bottom. At lr = 0.1 our toy hit the ideal w = 3 in about 3 steps. This is the shape you are aiming for."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Too big: the loss curve shoots upward and diverges'><g font-family='monospace'><line x1='60' y1='108' x2='470' y2='108' stroke='#E5DFD6' stroke-width='1.5'/><path d='M70 100 L150 70 L200 95 L260 40 L320 78 L380 18 L440 12' fill='none' stroke='#C93B3B' stroke-width='2.5'/><text x='260' y='124' font-size='11' fill='#6B645E' text-anchor='middle'>training steps →</text><text x='40' y='68' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 68)'>loss</text><text x='395' y='30' font-size='12' fill='#C93B3B'>→ NaN</text></g></svg>",
  note:"<b>Too big — it explodes.</b> Each step jumps past the bottom to a worse point. The loss bounces and shoots upward (often ending as <b>NaN</b>). At lr = 0.6 our toy loss grew from 16 to 700,000 in four steps. Training has diverged."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Warmup: the learning rate rises from tiny over the first steps'><g font-family='monospace'><line x1='60' y1='95' x2='470' y2='95' stroke='#E5DFD6' stroke-width='1.5'/><path d='M70 90 L180 35 L460 35' fill='none' stroke='#2A7B9B' stroke-width='2.5'/><text x='260' y='112' font-size='11' fill='#6B645E' text-anchor='middle'>training steps →</text><text x='40' y='60' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 60)'>lr</text><text x='120' y='25' font-size='10' fill='#1F6280'>warmup ramp</text></g></svg>",
  note:"<b>Warmup.</b> Start the learning rate tiny and raise it over the first few hundred steps. A fresh model is fragile; a big early step could blow it up. Warmup eases it in before running at full speed."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Decay: the learning rate falls slowly over training'><g font-family='monospace'><line x1='60' y1='95' x2='470' y2='95' stroke='#E5DFD6' stroke-width='1.5'/><path d='M70 30 L120 30 C 260 32, 360 85, 460 90' fill='none' stroke='#5E5191' stroke-width='2.5'/><text x='260' y='112' font-size='11' fill='#6B645E' text-anchor='middle'>training steps →</text><text x='40' y='60' font-size='11' fill='#6B645E' text-anchor='middle' transform='rotate(-90 40 60)'>lr</text><text x='330' y='60' font-size='10' fill='#5E5191'>decay down</text></g></svg>",
  note:"<b>Decay.</b> Slowly lower the learning rate as training goes on. Big steps early cover ground fast; small steps late let the model settle gently into a good minimum. A good rate plus its warmup-and-decay schedule is a top-value knob in any big run."}
];
@@@ region name=QS
var QS=[
 {q:'1. What does the learning rate control in w ← w − lr × grad?',
  opts:['which direction is downhill','how big each step is','how many layers the model has','the size of the dataset'],
  ans:1, fb:'Right. The gradient picks the downhill direction; the learning rate (lr) scales how far you move each step.'},
 {q:'2. The loss shoots upward to bigger and bigger numbers (or NaN). What went wrong?',
  opts:['the learning rate is too small','the learning rate is too big — steps overshoot and diverge','there is no bug; that is normal','the batch size is too large'],
  ans:1, fb:'Right. Too big a learning rate makes each step jump past the bottom to a worse point, so the loss grows and training diverges.'},
 {q:'3. The loss drops but only a tiny bit each step, over thousands of steps. What is the likely cause?',
  opts:['the learning rate is too small — it crawls','the learning rate is too big','the model has too many weights','the loss function is wrong'],
  ans:0, fb:'Right. A too-small learning rate moves the right way but very slowly, wasting time and compute.'},
 {q:'4. Your training runs cleanly with no error and the loss drops fast, but the final loss is a bit worse than a teammate got on the same setup. Their only difference is a smaller learning rate. What most likely happened, and why is it easy to miss?',
  opts:['Your learning rate was slightly too high, so the steps kept bouncing around the bottom instead of settling in — the loss still went down and never errored, so nothing flagged it','A smaller learning rate always overfits, which is the only explanation','Your model had fewer parameters, which is the only thing that sets the final loss','A worse final loss with no error means the data was corrupted'],
  ans:0, fb:'Right. A rate that is a little too high does not diverge — the loss still falls and looks healthy — but the big steps overshoot the minimum and settle at a worse value. It is easy to miss because nothing crashes. You compare final loss across a few rates rather than trusting that "loss went down."'}
];
