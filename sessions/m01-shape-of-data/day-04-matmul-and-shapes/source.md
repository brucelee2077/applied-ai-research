---
quest_id: wf1-d04-matmul
donor: m01-day-04-matmul-and-shapes.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 1 · Day 4 — Matmul &amp; Shapes"
spine: "shape"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 1 · Day 4 — Matmul &amp; Shapes</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M1 Day 4</div>
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
<a class="lnav prev" href="../day-03-broadcasting-dtypes/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t">Broadcasting &amp; dtypes</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-05-logs-and-exponents/lesson.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t">Logs &amp; Exponents</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 1 · Represent · Day 4</span>
      <h1>Matmul &amp; Shapes<span class="sub">The One Operation Every Layer Is Built From</span></h1>
      <p class="lede">Almost all of the arithmetic in a neural network is a single operation: <strong>matrix multiplication</strong>. Learn its shape rule — <code>(m, k) @ (k, n) = (m, n)</code> — and you can read what any layer does to its data, and why chips are built the way they are.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><b>By the end, you'll be able to:</b> read a matmul's output shape from its two input shapes, say why the inner dimensions must match, and explain why a neural-network layer <em>is</em> a matmul.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">What matrix multiplication is</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–3 — arrays, shape, and element-wise math. <b>No new tools</b>. If you multiplied two numbers and summed on Day 1, you already did the core step.</div></div>

      <h4>What is a matmul?</h4>
      <p><span class="term" data-tip="Matrix multiplication: combine two 2-D arrays so each output entry is the dot product of a row of the first and a column of the second. Written with @.">Matmul</span> (matrix multiplication) combines two 2-D arrays into a new one. Each entry of the result is a <span class="term" data-tip="Multiply two equal-length lists element by element, then add up all the products — one number out.">dot product</span>: take a <strong>row</strong> from the first array and a <strong>column</strong> from the second, multiply them pairwise, and add. In code it's the <code>@</code> operator (or <code>np.matmul</code>).</p>

      <h4>The shape rule (the whole lesson in one line)</h4>
      <div class="callout c-ok"><span class="ic">📐</span><div><code>(m, k) @ (k, n) = (m, n)</code> — the <strong>inner</strong> dimensions (<code>k</code>) must match and vanish; the <strong>outer</strong> dimensions (<code>m</code>, <code>n</code>) form the result.</div></div>

      <h4>How does it relate to what you know?</h4>
      <p>On Day 1 you did element-wise math with <code>*</code>. Matmul is different: <code>@</code> does rows-times-columns, not element-by-element. Why care? A neural-network layer <em>is</em> a matmul: <code>y = x @ W</code> (plus a bias). Learn the shape rule and you can trace data through an entire network. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> one NVIDIA H100 chip — the workhorse behind most frontier training runs — can do roughly 1,000 trillion floating-point operations a second (FLOPs) in <code>bfloat16</code>. Almost all of that is spent on matmuls, running exactly the shape rule you're about to learn.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> 大家背下了 <code>(m,k)@(k,n)=(m,n)</code>，却分不清 <code>@</code> 和 <code>*</code> — the rule is memorized, but matmul vs element-wise blur together. In plain terms: <code>@</code> and <code>*</code> look almost identical yet do completely different things, so people write the wrong one and get a legal-shaped array full of wrong numbers.</div></div>
      <button class="gotit" type="button">Got what a matmul is — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">A mixing board</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <div class="relate">
        <div class="card"><span class="big">🎛️</span><h5>Weights = a mixing board</h5><p>Each output channel is a <strong>weighted blend</strong> of all input channels. The weight matrix says how much each input feeds each output. Matmul does every blend at once.</p></div>
        <div class="card"><span class="big">🔀</span><h5>A size-changing machine</h5><p>Feed in a vector of size <code>k</code>, get out a vector of size <code>n</code>. The <code>k × n</code> weight table is the machine; the <code>m</code> is just "do this for <code>m</code> examples at once."</p></div>
      </div>
      <p><strong>In one line:</strong> matmul turns an input of width <code>k</code> into an output of width <code>n</code> using a <code>k × n</code> table of weights, for all <code>m</code> rows at once.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where it trips people: <code>*</code> is <strong>element-wise</strong> (needs matching shapes, from Day 3), while <code>@</code> is <strong>matmul</strong> (needs matching inner dims). They look similar and do totally different things — mixing them up is a classic bug.</div></div>
      <p><b>直觉 / Intuition:</b> 把权重矩阵想成"混音台"，中间那个 <code>k</code> 是"输入声道数"，两边一插上就抵消了 — the shared inner <code>k</code> is the plug that must fit on both sides, then it disappears. The technical point: matmul turns a width-<code>k</code> input into a width-<code>n</code> output through a <code>k×n</code> table, for all <code>m</code> rows at once.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The rule exactly — and why chips are matmul machines</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The rule, precisely</h4>
      <ul>
        <li>To compute <code>A @ B</code>: <code>A</code>'s number of columns must equal <code>B</code>'s number of rows (the inner dims).</li>
        <li>The result has shape <code>(A rows, B columns)</code>.</li>
        <li>Each <code>result[i, j]</code> = the dot product of row <code>i</code> of <code>A</code> with column <code>j</code> of <code>B</code>.</li>
        <li><code>@</code> is matmul; <code>*</code> is element-wise. Never confuse them.</li>
      </ul>
      <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — the matmul shape rule &amp; its cost</b>
      <br><b>1 · In words:</b> the two inner dimensions must match and cancel; the outer two survive as the result; each output entry is one row dotted with one column.
      <br><b>2 · The rule:</b> <code>(m, k) @ (k, n) → (m, n)</code>, costing <code>≈ 2·m·k·n</code> FLOPs. <code>m</code> = rows of A (examples); <code>k</code> = the shared inner width; <code>n</code> = columns of B (output width). The <code>2</code> = one multiply + one add per pair.
      <br><b>3 · Tiny numbers:</b> <code>(2, 3) @ (3, 4) → (2, 4)</code>. Inner <code>3 == 3</code> ✓. Output has <code>2×4 = 8</code> entries, each a 3-term dot product → <code>2·2·3·4 = 48</code> FLOPs.
      <br><b>4 · Sanity check:</b> the <code>k</code> axis appears on both sides and vanishes from the result; if the two inner numbers differ, it's illegal. The result element count is always <code>m·n</code> — never <code>k</code>.</div></div>

      <h4>Why a frontier lab cares</h4>
      <p>A <span class="term" data-tip="A dense (fully-connected / linear) layer: it maps inputs to outputs by y = xW + b, a matmul plus a bias.">dense layer</span> is literally <code>y = x @ W + b</code>. Attention is matmuls too: scores <code>= Q @ Kᵀ</code>, then output <code>= scores @ V</code>. Matmul is where almost all the <span class="term" data-tip="Floating-point operations — a count of the multiply/add work a computation does. Roughly 2·m·k·n for an (m,k)@(k,n) matmul.">FLOPs</span> go — exactly why the H100 chip from the teaser above is built to do almost nothing else. That's also exactly why the compute formula <code>C ≈ 6ND</code> (Kaplan et al., 2020 — you'll meet it in the scaling-law modules) is really a matmul count — and why GPUs and TPUs are, at heart, giant matmul machines.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> a layer is a matmul → matmul dominates the FLOPs → chips are built to do matmuls fast → the whole scaling story becomes "how many matmuls can you afford." And the most common runtime crash in ML is a matmul with mismatched inner dims.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>A mismatched matmul crashes loudly, which is easy to fix. The dangerous matmul bug is the one where the shapes happen to line up but the math is wrong — no crash, no error, just quietly wrong numbers.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): a legal matmul that computes the wrong thing (a bug that doesn't announce itself).</b> Two traps. First, writing <code>*</code> when you meant <code>@</code>: with a bias vector <code>(features,)</code> and an activation <code>(batch, features)</code>, <code>*</code> does not error — it broadcasts an element-wise multiply and returns the right-shaped array full of wrong numbers. Second, matmul on a <strong>square</strong> weight, or feeding the un-transposed operand: <code>A @ B</code> and a wrongly-oriented version can both have legal inner dims, so both run, but only one is the layer you meant. A senior engineer catches this in <b>code review</b> by checking that <code>@</code> (not <code>*</code>) is used for layers and by tracing one <code>result[i, j]</code> against "row <code>i</code> of A dotted with column <code>j</code> of B" — if that dot product isn't the quantity you wanted, the orientation is wrong even though the shape is fine.</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: batching more matmul work vs. memory and latency.</b> Because matmul is where almost all the FLOPs go, chips run it fastest when you hand them big matrices at once — larger batches and wider layers keep the hardware busy and cost less per number. What you pay: bigger matmuls need more memory for their inputs and outputs, and a single huge matmul finishes later than a small one, so interactive latency suffers. In a <b>design review</b> this is the batch-size decision — trade throughput (fill the matmul units) against memory budget and response time.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "A matmul is <code>(m,k)@(k,n)→(m,n)</code>: the inner dims cancel, each output is a row·column dot product, and it costs about <code>2·m·k·n</code> FLOPs — which is why a dense layer <code>y=x@W</code> and attention <code>Q@Kᵀ</code> are essentially the entire FLOP budget of a model. The nuance is that a legal shape isn't a correct matmul: writing <code>*</code> instead of <code>@</code>, or a wrong transpose, can run silently, so I trace one output entry against 'row i dotted with column j'."</div></div>
      <button class="gotit" type="button">Got the rule</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Leave today's evidence (strongly recommended)</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Understanding ≠ being able to write it. Multiply a few matrices and build a layer yourself — that's what makes the day count. Pick one:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py</code>. Multiply a <code>(2,3)</code> by a <code>(3,2)</code> with <code>@</code> and print the result's shape; try a mismatched <code>(2,3) @ (2,3)</code> in a <code>try/except</code> and print the error; build a dense layer <code>x @ W + b</code> with <code>x</code> shape <code>(1,3)</code> and <code>W</code> shape <code>(3,4)</code>; and show that <code>@</code> and <code>*</code> give different results. Run it with <code>python3 sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">ask Claude to build it</span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Help me build my Module 1 Day 4 artifact.

Create sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py that, with a comment on each step:
1. Makes A (2,3) and B (3,2); prints (A @ B).shape (expect (2,2)) and the result, noting the inner 3 matched and vanished.
2. Wraps A @ np.ones((2,3)) in try/except and prints the shape-mismatch error (inner 3 vs 2).
3. Builds a dense layer: x (1,3), W (3,4), b (4,); prints (x @ W + b) and its shape (1,4), with a comment 'this is a neural-net layer'.
4. Shows @ vs *: print A @ B (matmul) and A * A (element-wise) to make the difference obvious.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>Acceptance criteria</h4>
      <ul>
        <li>You multiply <code>(2,3) @ (3,2)</code> with <code>@</code> and print the <code>(2,2)</code> result; you catch the error from a mismatched <code>(2,3) @ (2,3)</code>.</li>
        <li>You build a dense layer <code>x @ W + b</code> and print its output shape, and you show <code>@</code> and <code>*</code> give different results.</li>
        <li>You can state the FLOP cost <code>2·m·k·n</code> for one of your matmuls before running it.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-04-matmul-and-shapes/log.md</code>: (1) in your own words, why must the inner dimensions match; (2) the FLOP count that surprised you; (3) one thing you're still unsure about (<code>@</code> vs <code>*</code> is fair game).</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 1 · Day 4 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Matmul &amp; Shapes</b>.<br>Next up: <b>Logs &amp; Exponents</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  mm:{html:'<span class="prompt">&gt;&gt;&gt;</span> A = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])   <span class="dim"># (2,3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> B = np.array([[<span class="num">1</span>,<span class="num">0</span>], [<span class="num">0</span>,<span class="num">1</span>], [<span class="num">1</span>,<span class="num">0</span>]])   <span class="dim"># (3,2)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> (A @ B).shape      <span class="dim"># inner 3 matches → (2,2)</span>\n'+
    '<span class="hl">(2, 2)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> A @ B\n<span class="ok">array([[ 4,  2],\n       [10,  5]])</span>',
    take:'<b>①  The inner dim matches and vanishes.</b> (2,<b>3</b>) @ (<b>3</b>,2) → the 3\'s touch and disappear; the outer 2 and 2 form the result (2,2). Each entry is a row·column dot product.'},
  bad:{html:'<span class="prompt">&gt;&gt;&gt;</span> A = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])   <span class="dim"># (2,3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> C = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])   <span class="dim"># (2,3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> A @ C\n'+
    '<span class="bad">ValueError: matmul: Input operand 1 has a\n'+
    'mismatch in its core dimension ... (size 3 is\n'+
    'different from 2)</span>',
    take:'<b>②  Inner dims must match.</b> (2,<b>3</b>) @ (<b>2</b>,3): the touching numbers are 3 and 2 — not equal — so matmul refuses. This exact crash is the #1 shape bug in ML code.'},
  layer:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([[<span class="num">1.</span>, <span class="num">2.</span>, <span class="num">3.</span>]])        <span class="dim"># (1,3) one input, 3 features</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> W = np.random.rand(<span class="num">3</span>, <span class="num">4</span>)         <span class="dim"># (3,4) weights</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> b = np.zeros(<span class="num">4</span>)                 <span class="dim"># (4,) bias</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> (x @ W + b).shape       <span class="dim"># (1,3)@(3,4)=(1,4), bias broadcast</span>\n'+
    '<span class="hl">(1, 4)</span>',
    take:'<b>③  This IS a neural-net layer.</b> y = x @ W + b turned 3 inputs into 4 outputs via a 3×4 weight table, then broadcast the bias (Day 3!) across the row. Every dense layer in every model is this line.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Dot product of two length-3 vectors equals 32'><g font-family='monospace' font-size='14' text-anchor='middle'><text x='90' y='40' fill='#1F6280'>[1, 2, 3]</text><text x='90' y='66' fill='#9A7208'>[4, 5, 6]</text><text x='210' y='53' font-size='16' fill='#6B645E'>→</text><text x='330' y='40' fill='#2C2A28'>1·4 + 2·5 + 3·6</text><text x='330' y='66' fill='#2C2A28'>= 4 + 10 + 18</text><text x='430' y='53' font-size='16' fill='#6B645E'>=</text><rect x='455' y='36' width='44' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='477' y='59' fill='#1a5c38' font-weight='bold'>32</text><text x='260' y='108' fill='#6B645E' font-size='12'>dot product: multiply pairwise, then add → ONE number</text></g></svg>",
  note:"<b>Start with a dot product.</b> Multiply two equal-length lists element-by-element, then sum: <code>[1,2,3]·[4,5,6] = 32</code>. One number out. A matmul is just many of these done at once."},
 {viz:"<svg viewBox='0 0 520 140' role='img' aria-label='A row dotted against each column of a matrix produces a row of outputs'><g font-family='monospace' font-size='13' text-anchor='middle'><text x='70' y='70' fill='#1F6280'>row</text><text x='70' y='88' fill='#1F6280'>[1 2 3]</text><text x='160' y='79' fill='#6B645E'>·</text><rect x='195' y='40' width='36' height='80' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><rect x='235' y='40' width='36' height='80' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='213' y='75' fill='#5E5191'>c</text><text x='213' y='95' fill='#5E5191'>o</text><text x='253' y='75' fill='#5E5191'>c</text><text x='253' y='95' fill='#5E5191'>o</text><text x='233' y='134' fill='#6B645E' font-size='11'>each column</text><text x='320' y='79' fill='#6B645E'>→</text><rect x='360' y='63' width='40' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><rect x='402' y='63' width='40' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='380' y='86' fill='#1a5c38'>o1</text><text x='422' y='86' fill='#1a5c38'>o2</text><text x='401' y='118' fill='#6B645E' font-size='11'>one output per column</text></g></svg>",
  note:"<b>One row against many columns.</b> Dot the row with column 1 → first output; with column 2 → second output. A row times a matrix gives a whole row of results — one number per column."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Shape rule (2,3)@(3,4)=(2,4), the inner 3s match and vanish'><g font-family='monospace' font-size='16' text-anchor='middle'><text x='90' y='60' fill='#1F6280'>( 2 ,</text><text x='140' y='60' fill='#C99A12' font-weight='bold'>3</text><text x='158' y='60' fill='#1F6280'>)</text><text x='210' y='60' fill='#6B645E'>@</text><text x='262' y='60' fill='#1F6280'>(</text><text x='282' y='60' fill='#C99A12' font-weight='bold'>3</text><text x='320' y='60' fill='#1F6280'>, 4 )</text><text x='400' y='60' fill='#6B645E'>=</text><rect x='430' y='42' width='72' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='466' y='65' fill='#1a5c38' font-weight='bold'>(2, 4)</text><line x1='140' y1='68' x2='282' y2='68' stroke='#C99A12' stroke-width='1.5' stroke-dasharray='4,3'/><text x='211' y='90' fill='#9A7208' font-size='11'>inner must match → vanishes</text><text x='260' y='116' fill='#2D8B55' font-size='12' font-weight='bold'>outer dims (2 and 4) survive as the result</text></g></svg>",
  note:"<b>The shape rule.</b> <code>(2,3) @ (3,4)</code>: the inner 3's match, cancel out, and the outer 2 and 4 become the result <code>(2,4)</code>. This one rule is all you need to trace shapes through a network."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='Mismatch (2,3)@(2,3) inner 3 vs 2 raises an error'><g font-family='monospace' font-size='16' text-anchor='middle'><text x='110' y='55' fill='#1F6280'>( 2 ,</text><text x='160' y='55' fill='#C93B3B' font-weight='bold'>3</text><text x='178' y='55' fill='#1F6280'>)</text><text x='220' y='55' fill='#6B645E'>@</text><text x='262' y='55' fill='#1F6280'>(</text><text x='282' y='55' fill='#C93B3B' font-weight='bold'>2</text><text x='320' y='55' fill='#1F6280'>, 3 )</text><text x='215' y='80' fill='#C93B3B' font-size='12'>3 ≠ 2</text><rect x='120' y='92' width='280' height='20' rx='5' fill='#FDE8E8' stroke='#C93B3B' stroke-width='1.5'/><text x='260' y='106' font-size='10.5' fill='#C93B3B'>ValueError: matmul core dimension mismatch</text></g></svg>",
  note:"<b>When it crashes.</b> <code>(2,3) @ (2,3)</code>: the inner numbers are 3 and 2 — not equal — so matmul errors before doing anything. Reading the two inner numbers first is how you catch this instantly."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='A dense layer x(1,3) @ W(3,4) + b = (1,4)'><g font-family='monospace' font-size='14' text-anchor='middle'><rect x='40' y='46' width='70' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='75' y='66' fill='#1F6280'>x (1,3)</text><text x='125' y='66' fill='#6B645E'>@</text><rect x='145' y='40' width='80' height='42' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='185' y='65' fill='#5E5191'>W (3,4)</text><text x='240' y='66' fill='#6B645E'>+</text><rect x='258' y='48' width='60' height='28' rx='5' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='288' y='67' fill='#9A7208'>b (4,)</text><text x='335' y='66' fill='#6B645E'>=</text><rect x='358' y='46' width='72' height='30' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='394' y='66' fill='#1a5c38' font-weight='bold'>y (1,4)</text><text x='260' y='108' fill='#2D8B55' font-size='12' font-weight='bold'>y = x @ W + b  —  this is one neural-network layer</text></g></svg>",
  note:"<b>The layer hiding inside.</b> <code>y = x @ W + b</code>: a matmul turns 3 inputs into 4 outputs, then the bias broadcasts across (Day 3). Stack layers like this and you have a neural network. It's matmuls all the way down."},
 {viz:"<svg viewBox='0 0 520 140' role='img' aria-label='Attention is matmuls and matmul dominates the FLOPs on matmul-machine chips'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='34' width='130' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='105' y='54' fill='#1F6280'>scores = Q @ Kᵀ</text><rect x='40' y='72' width='130' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='105' y='92' fill='#1F6280'>out = scores @ V</text><text x='195' y='72' fill='#6B645E'>→</text><rect x='230' y='40' width='250' height='58' rx='8' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='355' y='63' fill='#9A7208' font-weight='bold'>matmul ≈ most of the FLOPs</text><text x='355' y='84' fill='#9A7208'>(so C ≈ 6ND is a matmul count)</text><text x='260' y='126' fill='#2C2A28'>GPUs and TPUs are, at heart, giant matmul machines</text></g></svg>",
  note:"<b>Why it matters at frontier scale.</b> Attention is matmuls (<code>Q @ Kᵀ</code>, then <code>@ V</code>), and matmul is where nearly all the compute goes — that's why the scaling-law formula <code>C ≈ 6ND</code> is really counting matmul work, and why accelerators are designed as matmul engines. Everything you optimize later is about doing matmuls faster or cheaper."}
];
@@@ region name=QS
var QS=[
 {q:'1. What is the output shape of <code>(2, 3) @ (3, 4)</code>?',
  opts:['(3, 3)','(2, 4)','(2, 3)','error'],
  ans:1, fb:'Right. Inner 3\'s match and vanish; outer 2 and 4 form the result → (2, 4).'},
 {q:'2. For <code>A @ B</code> to work, what must match?',
  opts:['A\'s rows and B\'s rows','A\'s columns and B\'s rows (the inner dims)','The dtypes','Nothing — matmul always works'],
  ans:1, fb:'Right. A\'s column count must equal B\'s row count — the two inner dimensions.'},
 {q:'3. A dense (linear) layer computes:',
  opts:['y = x * W (element-wise)','y = x @ W + b (a matmul plus a bias)','y = x + W','y = W only'],
  ans:1, fb:'Right. y = x @ W + b — a matmul that maps inputs to outputs, plus a broadcast bias.'},
 {q:'4. A teammate\'s layer runs with no error and the output has the shape they expected, but the model never learns — loss barely moves. You look at the layer and see <code>y = x * W + b</code> where <code>x</code> is <code>(batch, features)</code> and <code>W</code> is also <code>(batch, features)</code>. Where do you look first, and why?',
  opts:['The learning rate — a stuck loss is always an optimizer problem','The <code>*</code>: it does an element-wise multiply (broadcasting), not the matmul a layer needs, so the shape looks right but every output is the wrong quantity','The dtype — element-wise vs matmul is only a precision issue','Nothing is wrong; <code>*</code> and <code>@</code> are interchangeable when shapes match'],
  ans:1, fb:'Right. <code>*</code> multiplies element by element and broadcasts, so <code>x * W</code> can produce a plausibly-shaped array without ever mixing features the way <code>x @ W</code> does. A real dense layer is <code>y = x @ W + b</code>. The shape check passes, so the bug is silent — you catch it by confirming the operator is <code>@</code>, not <code>*</code>, before blaming the optimizer.'}
];
