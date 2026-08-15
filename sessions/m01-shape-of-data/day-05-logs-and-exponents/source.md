---
quest_id: wf1-d05-logs
donor: m01-day-05-logs-and-exponents.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 1 · Day 5 — Logs &amp; Exponents"
spine: "shape"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 1 · Day 5 — Logs &amp; Exponents</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M1 Day 5</div>
@@@ region name=sidebar_nav
<nav aria-label="Sections">
      <div class="nav-group-label">Module 01 · Represent</div>
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
<a class="lnav prev" href="../day-04-matmul-and-shapes/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t">Matmul &amp; Shapes</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-06-random-seeds/lesson.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t">Random Seeds &amp; PRNG</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 1 · Represent · Day 5</span>
      <h1>Logs &amp; Exponents<span class="sub">How to Read a Scaling-Law Plot</span></h1>
      <p class="lede">The last bit of math before real ML. The famous <strong>scaling laws</strong> — the curves that predict how much better a model gets as you spend more compute — are <strong>power laws</strong>. On a log-log plot, a power law becomes a straight line. Learn to read that line and you can read the most important charts in the field.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><b>By the end, you'll be able to:</b> say what a log is, turn a power law into a straight line, and read the slope of a log-log line as its exponent.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">Exponents, logs, and power laws</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> just arithmetic and Days 1–4. <b>No calculus.</b> This lesson is lighter on code — it's the reading skill for the plots you'll meet in the scaling-law modules.</div></div>

      <h4>What is an exponent?</h4>
      <p>An <span class="term" data-tip="Repeated multiplication: x^b means multiply x by itself b times. 2^3 = 2·2·2 = 8.">exponent</span> is repeated multiplication: <code>2³ = 2·2·2 = 8</code>. The small number on top says how many times.</p>

      <h4>What is a log?</h4>
      <p>A <span class="term" data-tip="The inverse of an exponent. log10(1000)=3 because 10^3=1000. It answers: 10 to what power gives this number?">log</span> asks the reverse question: "10 to <em>what power</em> gives this number?" Since <code>10³ = 1000</code>, <code>log₁₀(1000) = 3</code>. A log tells you the number of <span class="term" data-tip="A factor of 10. Going from 100 to 1000 is one order of magnitude.">orders of magnitude</span>. Log and exponent <strong>undo each other</strong>.</p>

      <h4>What is a power law?</h4>
      <p>A <span class="term" data-tip="A relationship y = a·x^b, where one quantity is a fixed power of another. Scaling laws are power laws.">power law</span> is a relationship <code>y = a·x^b</code> — one thing is a power of another. Scaling laws (loss vs compute) are power laws. The magic you'll use: draw a power law on <strong>log-log</strong> axes and it becomes a straight <strong>line</strong>, whose slope is the exponent <code>b</code>. Keep going 👇</p>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> 很多人把 log 的规则背下来，却说不出它到底在问什么，于是一看到 log-log 图就懵了。In plain terms: people memorize the log <em>rules</em> but never anchor the one question a log asks — "10 to what power gives this number?" — so a log-log plot feels like a foreign language instead of the simplest chart in ML.</div></div>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> DeepMind's Chinchilla paper trained a 70-billion-parameter model on 1.4 trillion tokens — a ratio of 20 tokens per parameter they found by reading the slope of a log-log line, exactly like you're about to.</div></div>
      <button class="gotit" type="button">Got the three words — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">The Richter scale</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <div class="relate">
        <div class="card"><span class="big">📈</span><h5>A log axis = "×10 per step"</h5><p>On a normal ruler, steps add (+1, +2). On a <strong>log</strong> ruler, each step <strong>multiplies</strong> by 10: 1, 10, 100, 1000 are evenly spaced. Earthquakes (Richter) and loudness (decibels) work this way.</p></div>
        <div class="card"><span class="big">📏</span><h5>Why it helps</h5><p>A log turns "times 10" into "plus 1." Huge ranges (1 to a billion) fit on one axis, and any multiply-relationship becomes a straight line you can measure.</p></div>
      </div>
      <p><strong>In one line:</strong> a log squashes a "multiply" world into an "add" world — that's why power laws straighten out on log axes.</p>
      <p><b>直觉 / Intuition:</b> log 就像一把「每格 ×10」的尺子，把庞大的数字压成好读的小数字。The technical point is that a log converts multiplication into addition, so a range from 1 to a billion fits on one axis and any "multiply" relationship becomes a straight line you can measure.</p>
      <p><b>In code:</b> you meet this as <em>log-space</em> — multiplying hundreds of tiny probabilities underflows to <code>0</code>, so libraries add <code>log</code>-probabilities instead. Same "multiply → add" trick, just used to keep the arithmetic safe.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where it breaks: logs only work for <strong>positive</strong> numbers. And a power law only becomes a straight line when <strong>both</strong> axes are log (a "log-log" plot) — not just one.</div></div>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">Why a power law is a straight line — and why that runs the industry</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The one derivation</h4>
      <p>Start with a power law and take the log of both sides:</p>
      <div class="callout c-ok"><span class="ic">🧮</span><div><code>y = a · x^b</code>&nbsp;&nbsp;→&nbsp;&nbsp;<code>log y = log a + b · log x</code><br><span style="font-size:.85rem;color:var(--text2)">Rename <code>Y = log y</code> and <code>X = log x</code>: that's <code>Y = (log a) + b·X</code> — the equation of a <strong>straight line</strong> with slope <code>b</code>.</span></div></div>
      <p>So on log-log axes, a power law is a line, and its <strong>slope is the exponent</strong>. Two log rules make it work: <code>log(a·b) = log a + log b</code>, and <code>log(x^b) = b·log x</code>.</p>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — a power law becomes a straight line</b>
      <br><b>1 · In words:</b> take the log of both sides of a power law and the curve turns into a straight line whose steepness is the exponent.
      <br><b>2 · The formula:</b> <code>y = a·x^b</code> → <code>log y = log a + b·log x</code> — <code>y</code>, <code>x</code> = the two quantities; <code>a</code> = a constant that sets the height; <code>b</code> = the exponent you want. Rename <code>Y = log y</code>, <code>X = log x</code> and it reads <code>Y = (log a) + b·X</code>, a line with slope <code>b</code>.
      <br><b>3 · Tiny numbers:</b> <code>y = x²</code> with <code>x = 1, 10, 100</code> → <code>y = 1, 100, 10000</code>. Take <code>log₁₀</code>: <code>X = 0, 1, 2</code> and <code>Y = 0, 2, 4</code>. Y climbs 2 for every 1 that X climbs.
      <br><b>4 · Sanity check:</b> slope <code>= (4 − 0) / (2 − 0) = 2</code>, exactly the exponent <code>b = 2</code> you started with. If the slope doesn't match the power you put in, you forgot to <code>log</code> one of the axes.</div></div>

      <h4>Why a frontier lab cares</h4>
      <p>Kaplan et al.'s 2020 scaling-law paper found loss falls with compute at roughly <code>α ≈ 0.05</code> — plot loss vs. compute on log-log axes and you get a nearly straight line with that slope. That's exactly the method behind Chinchilla's 20-tokens-per-parameter finding from the teaser above. Two years later, DeepMind used that same log-log approach on compute-optimal training, and found that tokens and parameters should grow together, not just model size alone. That one exponent — the slope you're about to read off a log-log plot — predicts how much better a model gets for every 10× more compute, which is the business case for spending millions on a training run.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> loss vs compute is a power law → on log-log it's a straight line → the slope is the scaling exponent → the exponent predicts the payoff of more compute. You'll <em>derive</em> these laws in the scaling modules; today you learned to read the axes.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>Logs and exponents are also how we keep very small and very large numbers from breaking the arithmetic — and getting that wrong fails quietly.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): underflow to zero, then <code>log(0)</code> (a bug that doesn't announce itself).</b> A model's probability for one token might be <code>0.001</code>. Multiply a few hundred of those together — as you would to score a sentence — and the product falls below the smallest number the dtype can hold, so it rounds to <em>exactly</em> <code>0.0</code>. No error fires. Then <code>log(0.0)</code> returns <code>-inf</code>, and one <code>-inf</code> poisons the average into <code>nan</code>, which quietly makes the whole loss useless. The fix, which a senior engineer looks for in <b>code review</b>, is to add the logs instead of logging the product: <code>log(a) + log(b) + ...</code> stays in a safe range because it never forms the tiny product at all. Seeing a raw <code>np.log(product_of_probs)</code> is the red flag.</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: work in log-space for safety vs. remembering the numbers changed meaning.</b> Moving to logs turns underflow-prone multiplication into safe addition (<code>log(a·b) = log a + log b</code>) and squashes a huge range into a small one. What you pay: your values are no longer probabilities or counts — they are log-units, so you must combine them by adding (not multiplying) and exponentiate only at the very end to read a real number back. In a <b>design review</b> this is the choice to keep an entire pipeline in log-space: safer arithmetic, but every step must respect that the numbers are logs.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "A scaling law is a power law — <code>loss ∝ compute^(−α)</code> — so on log-log axes it's a straight line and the slope is the exponent. That one number tells you how much loss you buy per 10× more compute, which is the case for spending millions on a run. The nuance: do the arithmetic in log-space — <em>add</em> log-probabilities instead of logging a product — or a long sequence quietly underflows to <code>log(0) = −inf</code>."</div></div>
      <button class="gotit" type="button">Got the derivation</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Leave today's evidence (strongly recommended)</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Understanding ≠ being able to write it. Turn a power law into a line yourself — that's what makes the day count. Pick one:</p>
      <h4>Acceptance criteria</h4>
      <ul>
        <li><code>np.log(np.exp(3))</code> prints <code>3.0</code> — log and exp cancel.</li>
        <li><code>np.log10([1,10,100,1000])</code> prints <code>[0,1,2,3]</code> — each ×10 adds exactly 1.</li>
        <li>For <code>x=[1,10,100]</code>, <code>y=x**2</code>: <code>np.log10(y)</code> rises by 2 for every 1 that <code>np.log10(x)</code> rises, and the computed slope equals the exponent <code>2</code>.</li>
        <li>Runs cleanly with <code>python3 sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code> and prints those three checks.</li>
      </ul>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code>. Show that <code>np.log(np.exp(3))</code> is 3; print <code>np.log10([1,10,100,1000])</code>; then make <code>x = [1,10,100]</code>, <code>y = x**2</code>, and print <code>np.log10(x)</code> and <code>np.log10(y)</code> — check the log values rise by 2 for every 1 (slope 2). Run it with <code>python3 sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">ask Claude to build it</span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Help me build my Module 1 Day 5 artifact.

Create sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py that, with a comment on each step:
1. Prints 2**3, np.exp(1), and np.log(np.exp(3)) to show exponent vs log and that they undo each other.
2. Prints np.log10([1,10,100,1000]) → [0,1,2,3], noting each ×10 adds 1 (orders of magnitude).
3. Builds x=np.array([1,10,100]); y=x**2; prints y, then np.log10(x) and np.log10(y); computes the slope (rise/run) of log10(y) vs log10(x) and asserts it equals 2 (the exponent).
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-05-logs-and-exponents/log.md</code>: (1) in your own words, why does a power law become a straight line on log-log axes? (2) the number or output that surprised you; (3) one thing you're still unsure about.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 1 · Day 5 complete! 🏆</h3>
      <p>You can now read a log-log scaling plot: a power law becomes a straight line, and its slope is the exponent that predicts the payoff of more compute.<br>Next up: <b>Random Seeds &amp; PRNG</b> — make "random" repeat on command.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  explog:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="num">2</span> ** <span class="num">3</span>            <span class="dim"># exponent: 2 multiplied 3 times</span>\n'+
    '<span class="hl">8</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.exp(<span class="num">1</span>)         <span class="dim"># e to the power 1</span>\n'+
    '<span class="ok">2.718281828459045</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.log(np.exp(<span class="num">3</span>))  <span class="dim"># log undoes exp</span>\n'+
    '<span class="ok">3.0</span>',
    take:'<b>①  Log and exp are inverses.</b> An exponent multiplies; a log asks "to what power?" So <code>log(exp(3))</code> gives back 3 — they cancel, like + and −.'},
  oom:{html:'<span class="prompt">&gt;&gt;&gt;</span> np.log10([<span class="num">1</span>, <span class="num">10</span>, <span class="num">100</span>, <span class="num">1000</span>])\n'+
    '<span class="hl">array([0., 1., 2., 3.])</span>',
    take:'<b>②  log10 counts orders of magnitude.</b> Each ×10 adds exactly 1 to the log. That is why 1 all the way to a billion can sit, evenly spaced, on a single axis.'},
  power:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">1</span>, <span class="num">10</span>, <span class="num">100</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> y = x ** <span class="num">2</span>              <span class="dim"># a power law: y = x^2</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> y\n<span class="ok">array([    1,   100, 10000])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.log10(x)\n<span class="hl">array([0., 1., 2.])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.log10(y)\n<span class="hl">array([0., 2., 4.])</span>',
    take:'<b>③  The trick, in numbers.</b> In log space the points are (0,0), (1,2), (2,4) — a perfectly straight line with <b>slope 2</b>. The exponent (2) became the slope. That is how you read a scaling law.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Exponent 2 to the 3 equals 8'><g font-family='monospace' font-size='15' text-anchor='middle'><text x='120' y='55' fill='#1F6280'>2³ = 2·2·2</text><text x='250' y='55' fill='#6B645E'>=</text><rect x='285' y='36' width='44' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='307' y='59' fill='#1a5c38' font-weight='bold'>8</text><text x='260' y='92' fill='#6B645E' font-size='12'>exponent = how many times you multiply</text></g></svg>",
  note:"<b>Start with an exponent.</b> <code>2³</code> means multiply 2 by itself three times → 8. The little number on top is just a count of multiplications."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Log undoes exponent: log2 of 8 is 3, log10 of 1000 is 3'><g font-family='monospace' font-size='15' text-anchor='middle'><text x='130' y='48' fill='#9A7208'>log₂(8) = 3</text><text x='130' y='76' fill='#9A7208'>log₁₀(1000) = 3</text><text x='300' y='62' fill='#6B645E'>←→</text><text x='400' y='62' fill='#1F6280' font-size='12'>the inverse of ^</text></g></svg>",
  note:"<b>A log undoes an exponent.</b> It asks 'to what power?': <code>log₂(8)=3</code> because <code>2³=8</code>; <code>log₁₀(1000)=3</code> because <code>10³=1000</code>. Log and exponent are opposites."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='On a log axis, 1 10 100 1000 are evenly spaced'><g font-family='monospace' font-size='13' text-anchor='middle'><line x1='40' y1='60' x2='480' y2='60' stroke='#5A544E' stroke-width='1.5'/><g fill='#5E5191'><circle cx='70' cy='60' r='4'/><circle cx='200' cy='60' r='4'/><circle cx='330' cy='60' r='4'/><circle cx='460' cy='60' r='4'/></g><text x='70' y='82' fill='#2C2A28'>1</text><text x='200' y='82' fill='#2C2A28'>10</text><text x='330' y='82' fill='#2C2A28'>100</text><text x='460' y='82' fill='#2C2A28'>1000</text><text x='135' y='48' fill='#9A7208' font-size='11'>×10</text><text x='265' y='48' fill='#9A7208' font-size='11'>×10</text><text x='395' y='48' fill='#9A7208' font-size='11'>×10</text></g></svg>",
  note:"<b>A log axis spaces things by ×10.</b> 1, 10, 100, 1000 sit at equal distances — each gap is one order of magnitude. That is how a range from 1 to a billion fits on one screen."},
 {viz:"<svg viewBox='0 0 520 140' role='img' aria-label='y = x squared drawn on normal axes is a rising curve'><g font-family='monospace'><line x1='60' y1='120' x2='60' y2='20' stroke='#5A544E' stroke-width='1.5'/><line x1='60' y1='120' x2='460' y2='120' stroke='#5A544E' stroke-width='1.5'/><path d='M60 120 Q 300 118 300 60 Q 300 25 440 25' fill='none' stroke='#C93B3B' stroke-width='2.5'/><text x='250' y='138' text-anchor='middle' font-size='12' fill='#2C2A28'>x  (normal axis)</text><text x='250' y='45' text-anchor='middle' font-size='12' fill='#C93B3B'>y = x²  →  a curve</text></g></svg>",
  note:"<b>On normal axes, a power law bends.</b> <code>y = x²</code> shoots upward as a curve — hard to read, hard to compare, hard to fit a simple rule to."},
 {viz:"<svg viewBox='0 0 520 150' role='img' aria-label='On log-log axes the same y = x squared is a straight line with slope 2'><g font-family='monospace'><line x1='60' y1='125' x2='60' y2='20' stroke='#5A544E' stroke-width='1.5'/><line x1='60' y1='125' x2='460' y2='125' stroke='#5A544E' stroke-width='1.5'/><line x1='80' y1='120' x2='420' y2='35' stroke='#2D8B55' stroke-width='2.5'/><circle cx='80' cy='120' r='4' fill='#2D8B55'/><circle cx='250' cy='77' r='4' fill='#2D8B55'/><circle cx='420' cy='35' r='4' fill='#2D8B55'/><text x='250' y='143' text-anchor='middle' font-size='12' fill='#2C2A28'>log x</text><text x='30' y='75' text-anchor='middle' font-size='12' fill='#2C2A28' transform='rotate(-90 30 75)'>log y</text><text x='330' y='58' font-size='12' fill='#2D8B55'>a straight line!</text></g></svg>",
  note:"<b>On log-log axes, it straightens.</b> Take the log of both x and y and the same <code>y = x²</code> data falls on a perfectly straight line. This is the single most useful plot type in ML."},
 {viz:"<svg viewBox='0 0 520 160' role='img' aria-label='The slope of the log-log line equals the exponent; scaling law reads loss vs compute with slope alpha'><g font-family='monospace'><line x1='60' y1='120' x2='60' y2='25' stroke='#5A544E' stroke-width='1.5'/><line x1='60' y1='120' x2='440' y2='120' stroke='#5A544E' stroke-width='1.5'/><line x1='80' y1='40' x2='420' y2='110' stroke='#2A7B9B' stroke-width='2.5'/><text x='250' y='60' font-size='12' fill='#2A7B9B'>slope = the exponent</text><text x='250' y='140' text-anchor='middle' font-size='11.5' fill='#2C2A28'>log(compute) →</text><text x='30' y='75' text-anchor='middle' font-size='11.5' fill='#2C2A28' transform='rotate(-90 30 75)'>log(loss)</text><text x='250' y='156' text-anchor='middle' font-size='10.5' fill='#6B645E'>a scaling law: loss falls as a power of compute</text></g></svg>",
  note:"<b>Why it runs the industry.</b> Scaling laws plot loss vs compute on log-log axes; the line's <b>slope is the exponent</b> that predicts how much loss drops per 10× more compute. Read that slope and you've read the chart every frontier lab bets millions on. That closes Module 1 — you now speak the language."}
];
@@@ region name=QS
var QS=[
 {q:'1. A log undoes which operation?',
  opts:['Addition','Exponentiation (raising to a power)','Matrix multiply','Slicing'],
  ans:1, fb:'Right. log and exp are inverses: log(exp(x)) = x, just like + and − cancel.'},
 {q:'2. On a log-log plot, a power law <code>y = a·x^b</code> looks like:',
  opts:['A curve that bends upward','A straight line','A circle','A flat horizontal line'],
  ans:1, fb:'Right. Taking logs turns y = a·x^b into log y = log a + b·log x — a straight line.'},
 {q:'3. For <code>y = a·x^b</code>, the slope of that log-log line equals:',
  opts:['a','b (the exponent)','log a','1'],
  ans:1, fb:'Right. The slope is b — the exponent. The intercept is log a.'},
 {q:'4. Your sentence-scoring code multiplies a few hundred per-token probabilities together and then takes <code>np.log</code> of the product. It runs with no error, but the score comes out as <code>-inf</code> or <code>nan</code> for long sentences only. Where do you look first, and why?',
  opts:['The log function is broken and should be replaced','The product underflowed to exactly 0.0 (many small numbers multiplied), so <code>log(0.0)</code> is <code>-inf</code>; add the logs instead of logging the product','The sentences are simply too rare, so nan is correct','The exponent slope is negative, which always yields nan'],
  ans:1, fb:'Right. Multiplying hundreds of values below 1 drives the product under the smallest representable number, so it rounds to <code>0.0</code> with no warning; <code>log(0.0) = -inf</code> then poisons the average into <code>nan</code>. Only long sentences hit it because they have more factors. The fix is to sum the logs — <code>log(a)+log(b)+...</code> — which never forms the tiny product.'}
];
