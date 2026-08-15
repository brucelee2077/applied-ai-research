---
quest_id: wf1-d03-broadcasting
donor: m01-day-03-broadcasting-dtypes.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 1 · Day 3 — Broadcasting &amp; dtypes"
spine: "shape"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 1 · Day 3 — Broadcasting &amp; dtypes</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M1 Day 3</div>
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
<a class="lnav prev" href="../day-02-indexing-slicing/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t">Indexing &amp; Slicing</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-04-matmul-and-shapes/lesson.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t">Matmul &amp; Shapes</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 1 · Represent · Day 3</span>
      <h1>Broadcasting &amp; dtypes<span class="sub">Math between different shapes</span></h1>
      <p class="lede">You can build arrays, slice them, and read their shape. Now the interesting part: what happens when you try to add two arrays that are <strong>not</strong> the same shape? Most languages make you write a loop. NumPy just does it — a trick called <strong>broadcasting</strong>. Pair it with one more idea, <strong>dtype</strong> (how big each number is), and you have the two levers that quietly set how much memory every model on Earth needs.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><b>By the end, you'll be able to:</b> predict whether two shapes can be added, state the one broadcasting rule, and explain why <code>float32</code> uses half the memory of <code>float64</code>.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
      <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">Two ideas: broadcasting and dtype</span><span class="sec-tag">What is it</span></div>
      <div class="sec-body">
        <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–2 — arrays, shape, and that <code>x + 10</code> adds to every element. <b>No new tools</b> — two ideas that ride on what you have.</div></div>
        <h4>What is broadcasting?</h4>
        <p><span class="term" data-tip="NumPy's rule for doing math between arrays of different shapes: a side that is a single value gets copied ('stamped') across to match the other — without really duplicating the data.">Broadcasting</span> is NumPy's rule for doing math between arrays of <strong>different shapes</strong>. The idea in one line: when one side has a <strong>single value</strong> where the other has many, NumPy copies that one value across to fill the gap — so you never write a loop. You already used the simplest case on Day 1: <code>x + 10</code> copied the single number 10 across every element. In a moment we'll picture this copying as a <strong>stamp</strong> — and that same stamp is all you need to understand the full rule later.</p>
        <h4>What is a dtype?</h4>
        <p>A <span class="term" data-tip="Data type: the single kind and size every element of an array shares, e.g. int64, float32, float64.">dtype</span> (data type) is the <strong>one kind and size</strong> every element in an array shares — like <code>int64</code>, <code>float32</code>, or <code>float64</code>. It decides two things: what values fit (whole numbers vs decimals) and <strong>how many bytes each number costs</strong>.</p>
        <h4>Why put them together today?</h4>
        <p>Both are about <strong>doing math efficiently at scale</strong>. Broadcasting lets one small array (like a bias) act on a huge one with no copies. The dtype decides the memory bill — and that bill is what forces the whole <span class="term" data-tip="Storing numbers in fewer bits (a smaller dtype box) to shrink a model's memory, at the cost of some precision.">quantization</span> story you'll meet later. Keep going 👇</p>
        <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> broadcasting 的规则看着简单，可它<em>不报错的时候</em>最危险 — the rule is short, but its silent successes are the trap. In plain terms: people expect a shape mismatch to crash, but NumPy often "helpfully" stretches a size-1 axis and returns a bigger, wrong array — so the bug ships instead of throwing.</div></div>
        <button class="gotit" type="button">Got the two ideas — let's go</button>
      </div>
    </section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
      <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">A stamp, and a set of boxes</span><span class="sec-tag">Intuition</span></div>
      <div class="sec-body">
        <div class="relate">
          <div class="card"><h5><span class="big" aria-hidden="true">🖊️</span> Broadcasting = one stamp</h5><p>To add 10 to every cell of a grid, you don't write "10" in each cell by hand. You press <strong>one stamp</strong> — the single 10 — across the whole grid at once. That is broadcasting: a small thing pressed out to cover a big thing.</p></div>
          <div class="card"><h5><span class="big" aria-hidden="true">📦</span> dtype = the size of the box</h5><p>Every number lives in a box, and you pick the box size. <code>float64</code> is a big box (8 bytes, lots of precision); <code>float32</code> is half the size (4 bytes); <code>bfloat16</code> is tiny (2 bytes). Smaller box → less memory.</p></div>
        </div>
        <p><strong>In one line:</strong> broadcasting is a stamp that presses a small shape across a big one; dtype is how big each number's box is.</p>
        <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Where the analogy breaks:</b> a real stamp uses ink on every press, but broadcasting never truly copies the data — the "press" is virtual, so it costs no extra memory. And a smaller box is not free: squeeze the number too hard and you lose precision. That tension — smaller vs. accurate — is the whole story of <span class="term" data-tip="Storing numbers in fewer bits (a smaller dtype box) to shrink a model's memory, at the cost of some precision.">quantization</span> you'll meet later.</div></div>
        <p><b>直觉 / Intuition:</b> 把 size-1 的那条轴想成"一个可以无限盖章的印章" — a size-1 axis is a stamp you can press as many times as needed; a size-3 axis is three fixed things that can't fill five slots. The technical point: broadcasting stretches size-1 axes for free (no memory), and that is exactly how a bias vector reaches a whole batch in one op.</p>
        <button class="gotit" type="button">Got the picture</button>
      </div>
    </section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
      <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The one rule, the byte table, and why money is bytes</span><span class="sec-tag">Mechanism &amp; why</span></div>
      <div class="sec-body">
        <h4>The one broadcasting rule</h4>
        <p><strong>The stamp, made precise.</strong> A dimension of size <strong>1</strong> is a stamp — a single thing NumPy can press out as many times as it needs to match its partner. A dimension of size 3 sitting next to a size 5 is three <em>fixed</em> things, not a stamp, so it can't fill five slots.</p>
        <p>To test whether two shapes fit, slide them together like two combs lined up at their <strong>right edge</strong>. Why the right? The rightmost number is the <strong>innermost</strong> count — the actual numbers sitting side by side in one row (like the features of a single example). You match those innermost units first, then work outward.</p>
        <p>At each tooth of the comb, the pair fits if the two numbers are <strong>equal</strong>, or if <strong>one side is 1</strong> (or simply not there — a shorter shape is read as having 1s added on its left). The size-1 / missing side is the stamp that stretches. If a pair is neither equal nor 1 → <strong>error</strong>.</p>
        <div class="callout c-ok"><span class="ic">✅</span><div><code>(2, 3) + (3,)</code> → line up the right edges: <code>3</code> vs <code>3</code> ✓ (equal), then <code>2</code> vs <em>missing</em> ✓ (the shorter shape gets a 1, and a 1 is a stamp) → result <code>(2, 3)</code>.</div></div>
        <div class="callout c-warn"><span class="ic">❌</span><div><code>(2, 3) + (2,)</code> → line up the right edges: <code>3</code> vs <code>2</code> — not equal, and neither one is 1, so neither is a stamp → <strong>error</strong>.</div></div>
        <div class="callout c-info"><span class="ic">🪜</span><div><b>Math Ladder — the broadcasting rule</b>
        <br><b>1 · In words:</b> line the two shapes up at their right edge; each axis pair works if the sizes are equal, or one of them is 1 (a missing axis counts as 1).
        <br><b>2 · The rule:</b> for the k-th axis from the right, <code>out[k] = max(a[k], b[k])</code> <em>only if</em> <code>a[k] == b[k]</code> or <code>min(a[k], b[k]) == 1</code>; otherwise it errors. Here <code>a[k]</code>, <code>b[k]</code> = the two input axis sizes, <code>out[k]</code> = the result axis size.
        <br><b>3 · Tiny numbers:</b> <code>(3, 1)</code> + <code>(1, 4)</code>. Axis −1: <code>1 vs 4</code> → one side is 1 → 4. Axis −2: <code>3 vs 1</code> → one side is 1 → 3. Result <code>(3, 4)</code> — 12 numbers grown from 3 + 4 inputs, with <em>zero</em> copies.
        <br><b>4 · Sanity check:</b> the result axis is never smaller than either input; a size-1 axis stretches, a size-3-vs-5 cannot. If you can't make every pair "equal or a 1", it must error — NumPy never silently guesses which to keep.</div></div>
        <h4>The byte table</h4>
        <ul>
          <li><code>float64</code> = 8 bytes · <code>float32</code> = 4 bytes · <span class="term" data-tip="A 16-bit floating type used a lot in ML: half the memory of float32, keeps the same range with less precision.">bfloat16</span> = 2 bytes each.</li>
          <li>Memory of a tensor = number of elements × bytes/element. Halve the bytes → halve the memory.</li>
        </ul>
        <h4>Why a frontier lab cares</h4>
        <p>Bytes are money and speed. A 7-billion-parameter model in <code>float32</code> needs ~28 GB just for weights; in <code>bfloat16</code>, ~14 GB — the difference between fitting on one chip or not. Broadcasting is how a bias vector <code>(features,)</code> gets added to a whole batch <code>(batch, features)</code> with no copy.</p>
        <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this chain:</b> broadcasting → one op covers many shapes, no copies → less code + less memory. dtype → fewer bytes per number → bigger models fit on the same hardware. <strong>Every quantization trick later is just this dial.</strong></div></div>
        <h4>The staff lens — one silent failure, one trade-off</h4>
        <p>Broadcasting and dtypes are the two places in array math where a bug does not announce itself — the code runs, prints numbers, and is quietly wrong.</p>
        <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): the accidental broadcast (a bug that doesn't announce itself).</b> You expect two vectors of the same length to add element-wise. But if one is a row of shape <code>(n,)</code> and the other is a column of shape <code>(n, 1)</code>, broadcasting does not error — it stretches both and hands back an <code>(n, n)</code> grid (every pair added). No crash, no warning; you just have <code>n</code> times too many numbers, and a mean or loss computed over them looks plausible but is wrong. A senior engineer catches this in <b>code review</b> by asking "print the <code>.shape</code> right after the operation" — if the result gained a dimension you didn't intend, an accidental broadcast happened.</div></div>
        <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: fewer bytes vs. less precision and range.</b> Dropping from <code>float32</code> (4 bytes) to <code>bfloat16</code> (2 bytes) halves the memory bill and lets a bigger model fit on the same chip. What you pay: each number now carries fewer significant digits, so tiny values can round to zero and long sums drift. In a <b>design review</b> the question is not "is float32 or bfloat16 better" — it is "which numbers can tolerate the lower precision (most weights and activations) and which must stay high precision (the running totals that accumulate error)." That split is exactly what mixed-precision training decides.</div></div>
        <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "Broadcasting right-aligns the shapes: each axis pair must be equal or one must be 1, and a size-1 axis is stretched for free — no copy — which is how a <code>(features,)</code> bias adds onto a <code>(batch, features)</code> activation in one op. The nuance is the accidental broadcast: <code>(n,)</code> against <code>(n, 1)</code> doesn't error, it fans out to <code>(n, n)</code>, so I print <code>.shape</code> right after the op to catch a dimension I didn't intend."</div></div>
        <button class="gotit" type="button">Got the rule and the bytes</button>
      </div>
    </section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
      <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Leave today's evidence (strongly recommended)</span><span class="sec-tag">Produce</span></div>
      <div class="sec-body">
        <p>Understanding ≠ being able to write it. Broadcast a few shapes and weigh a few dtypes yourself — that's what makes the day count. Pick one:</p>
        <h4>Option A · write it yourself</h4>
        <p>Create <code>sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py</code>. Add a scalar to an array; add a <code>(3,)</code> row to a <code>(2,3)</code> grid; try <code>(2,3) + (2,)</code> inside a <code>try/except</code> and print the error; then print <code>.dtype</code> and <code>.nbytes</code> for the same data as <code>float64</code> vs <code>float32</code>. Run it with <code>python3 sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py</code>.</p>
        <h4>Option B · let Claude build it, then read it</h4>
        <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
        <div class="prompt">
          <div class="prompt-h"><span class="prompt-l">ask Claude to build it</span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
          <pre class="prompt-t" id="pp">Help me build my Module 1 Day 3 artifact.

Create sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py that, with a comment on each step:
1. Prints x + 10 for x = np.array([1,2,3]) (scalar broadcast).
2. Adds row = np.array([10,20,30]) to g = np.array([[1,2,3],[4,5,6]]) and prints the (2,3) result, noting the (3,) row was stretched across both rows.
3. Wraps g + np.array([1,2]) in try/except and prints the broadcasting error (shapes (2,3) and (2,) do not align).
4. Builds b = np.array([1.,2.,3.]) as float64, makes f = b.astype(np.float32), and prints b.dtype, f.dtype, b.nbytes, f.nbytes — showing float32 uses half the bytes.
Then run it and paste the output at the bottom as a comment.</pre>
        </div>
        <h4>Acceptance criteria</h4>
        <ul>
          <li>You show a scalar broadcast, a <code>(3,)</code> row added to a <code>(2,3)</code> grid, and a caught broadcasting error for <code>(2,3) + (2,)</code>.</li>
          <li>You print <code>.dtype</code> and <code>.nbytes</code> for the same data as <code>float64</code> vs <code>float32</code>, and the float32 bill is exactly half.</li>
          <li>Before running each op, you can predict the result shape from the right-aligned rule.</li>
        </ul>
        <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-03-broadcasting-dtypes/log.md</code>: (1) the one sentence you'd tell a teammate about the broadcasting rule; (2) which broadcast surprised you, or which dtype fact stuck; (3) one thing you're still unsure about.</div></div>
        <button class="gotit" type="button">Done</button>
      </div>
    </section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 1 · Day 3 complete!</h3>
      <p>You can predict a broadcast from two shapes and explain why <code>float32</code> halves the memory of <code>float64</code>.<br>Next up: <b>Matmul &amp; Shapes</b> — the one operation every neural network is built from.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  scalar:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">1</span>, <span class="num">2</span>, <span class="num">3</span>])   <span class="dim"># shape (3,)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x + <span class="num">10</span>          <span class="dim"># the 10 is stretched to every element</span>\n'+
    '<span class="ok">array([11, 12, 13])</span>',
    take:'<b>①  Scalar broadcast — the simplest stamp.</b> One number (10) is pressed across every slot of the array. You did this on Day 1 with <code>x + 10</code> — now it has a name.'},
  row:{html:'<span class="prompt">&gt;&gt;&gt;</span> g = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])  <span class="dim"># (2,3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> row = np.array([<span class="num">10</span>, <span class="num">20</span>, <span class="num">30</span>])       <span class="dim"># (3,)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> g + row       <span class="dim"># row stretched down onto BOTH rows</span>\n'+
    '<span class="ok">array([[11, 22, 33],\n       [14, 25, 36]])</span>',
    take:'<b>②  The general case.</b> Shapes differ — (2,3) vs (3,) — so the (3,) row becomes a stamp pressed <b>down</b> onto both rows of the grid. One bias row, the whole batch, no loop.'},
  dtype:{html:'<span class="prompt">&gt;&gt;&gt;</span> a = np.array([<span class="num">1</span>, <span class="num">2</span>, <span class="num">3</span>]);  a.dtype\n<span class="hl">dtype(\'int64\')</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> b = np.array([<span class="num">1.0</span>, <span class="num">2.0</span>, <span class="num">3.0</span>]);  b.dtype\n<span class="hl">dtype(\'float64\')</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> f = b.astype(np.float32)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> b.nbytes, f.nbytes     <span class="dim"># float64=8B each, float32=4B</span>\n'+
    '<span class="hl">(24, 12)</span>',
    take:'<b>③  dtype = kind + box size.</b> Whole numbers default to int64, decimals to float64. Casting to float32 puts each number in a half-size box (24 → 12 bytes). Half the box → half the memory — the seed of quantization.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Scalar 10 broadcast across a length-3 array'><g font-family='monospace' font-size='14' text-anchor='middle'><rect x='70' y='40' width='46' height='40' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='93' y='66' fill='#1F6280'>1</text><rect x='120' y='40' width='46' height='40' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='143' y='66' fill='#1F6280'>2</text><rect x='170' y='40' width='46' height='40' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='193' y='66' fill='#1F6280'>3</text><text x='240' y='66' font-size='18' fill='#6B645E'>+</text><rect x='265' y='40' width='46' height='40' rx='6' fill='#FCF3DC' stroke='#A67D08' stroke-width='2'/><text x='288' y='66' fill='#6E5205' font-weight='bold'>10</text><text x='288' y='100' font-size='10' fill='#C93B3B'>one number…</text><text x='340' y='66' font-size='18' fill='#6B645E'>=</text><rect x='365' y='40' width='46' height='40' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='388' y='66' fill='#1a5c38' font-weight='bold'>11</text><rect x='413' y='40' width='46' height='40' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='436' y='66' fill='#1a5c38' font-weight='bold'>12</text><rect x='461' y='40' width='46' height='40' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='484' y='66' fill='#1a5c38' font-weight='bold'>13</text></g></svg>",
  note:"<b>Start with the simplest stamp.</b> <code>x + 10</code>: the single number 10 is <b>pressed</b> across all three slots, then added slot-by-slot. You used this on Day 1 without knowing its name."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Broadcasting rule: align shapes from the right, size-1 or missing stretches'><g font-family='monospace' font-size='14'><text x='260' y='24' text-anchor='middle' fill='#2C2A28'>align shapes from the RIGHT →</text><text x='150' y='60' text-anchor='end' fill='#1F6280'>( 2 ,</text><text x='210' y='60' text-anchor='middle' fill='#1F6280' font-weight='bold'>3 )</text><text x='150' y='90' text-anchor='end' fill='#6E5205'>(</text><text x='210' y='90' text-anchor='middle' fill='#6E5205' font-weight='bold'>3 )</text><line x1='198' y1='68' x2='198' y2='78' stroke='#2D8B55' stroke-width='2'/><text x='300' y='60' fill='#6B645E'>3 = 3  ✓</text><text x='300' y='90' fill='#6B645E'>2 vs (missing) ✓ → stretch</text><text x='260' y='118' text-anchor='middle' fill='#2D8B55' font-weight='bold'>compatible → result (2, 3)</text></g></svg>",
  note:"<b>The one rule.</b> Slide the two shapes together at their <b>right edge</b>, like two combs. At each tooth they fit if the numbers are <b>equal</b>, or if <b>one side is 1</b> (a 1 is a stamp — it presses out to any size). Check the rightmost pair first, then move left."},
 {viz:"<svg viewBox='0 0 520 150' role='img' aria-label='A (3,) row broadcast down onto both rows of a (2,3) grid'><g font-family='monospace' font-size='13' text-anchor='middle'><rect x='40' y='40' width='42' height='34' rx='5' fill='#FCF3DC' stroke='#A67D08' stroke-width='2'/><rect x='84' y='40' width='42' height='34' rx='5' fill='#FCF3DC' stroke='#A67D08' stroke-width='2'/><rect x='128' y='40' width='42' height='34' rx='5' fill='#FCF3DC' stroke='#A67D08' stroke-width='2'/><text x='61' y='63' fill='#6E5205'>10</text><text x='105' y='63' fill='#6E5205'>20</text><text x='149' y='63' fill='#6E5205'>30</text><text x='105' y='92' fill='#6B645E' font-size='11'>row (3,)</text><path d='M105 96 L105 112' stroke='#A67D08' stroke-width='2' stroke-dasharray='4,3'/><path d='M105 96 L105 130' stroke='#A67D08' stroke-width='2' stroke-dasharray='4,3'/><text x='210' y='90' font-size='18' fill='#6B645E'>onto</text><rect x='300' y='40' width='42' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><rect x='344' y='40' width='42' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><rect x='388' y='40' width='42' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><rect x='300' y='78' width='42' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><rect x='344' y='78' width='42' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><rect x='388' y='78' width='42' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='321' y='63' fill='#1a5c38'>11</text><text x='365' y='63' fill='#1a5c38'>22</text><text x='409' y='63' fill='#1a5c38'>33</text><text x='321' y='101' fill='#1a5c38'>14</text><text x='365' y='101' fill='#1a5c38'>25</text><text x='409' y='101' fill='#1a5c38'>36</text><text x='365' y='134' fill='#2D8B55' font-weight='bold'>(2,3) + (3,) = (2,3)</text></g></svg>",
  note:"<b>The general case.</b> A <code>(3,)</code> row added to a <code>(2,3)</code> grid: the row is a stamp pressed <b>down</b> onto both rows, then added. This is exactly how a bias vector adds to every row of a batch."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Incompatible shapes (2,3) and (2,) raise a broadcasting error'><g font-family='monospace' font-size='14'><text x='150' y='55' text-anchor='end' fill='#1F6280'>( 2 ,</text><text x='210' y='55' text-anchor='middle' fill='#C93B3B' font-weight='bold'>3 )</text><text x='150' y='88' text-anchor='end' fill='#6E5205'>(</text><text x='210' y='88' text-anchor='middle' fill='#C93B3B' font-weight='bold'>2 )</text><text x='300' y='72' fill='#C93B3B' font-weight='bold'>3 ≠ 2, neither is 1</text><rect x='120' y='100' width='280' height='22' rx='5' fill='#FDE8E8' stroke='#C93B3B' stroke-width='1.5'/><text x='260' y='115' text-anchor='middle' font-size='11' fill='#C93B3B'>ValueError: operands could not be broadcast</text></g></svg>",
  note:"<b>When it fails.</b> <code>(2,3) + (2,)</code>: line up the right edges → 3 vs 2. Not equal, and neither is 1, so neither can be a stamp → NumPy raises a broadcasting error. A row length that doesn't match is a very common bug."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='float64 8 bytes, float32 4 bytes, bfloat16 2 bytes per number'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='60' y='40' width='120' height='34' rx='5' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='120' y='62' fill='#C93B3B' font-weight='bold'>float64 · 8 B</text><rect x='200' y='40' width='90' height='34' rx='5' fill='#FCF3DC' stroke='#A67D08' stroke-width='2'/><text x='245' y='62' fill='#6E5205' font-weight='bold'>float32 · 4 B</text><rect x='310' y='40' width='60' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='340' y='62' fill='#1a5c38' font-weight='bold'>bf16 · 2 B</text><text x='260' y='100' fill='#2C2A28'>same number, smaller box → less memory</text></g></svg>",
  note:"<b>Now the dtype dial.</b> The exact same value costs 8, 4, or 2 bytes depending on its dtype. Choosing a smaller box is the single lever that decides how much memory an array (or a whole model) eats."},
 {viz:"<svg viewBox='0 0 520 140' role='img' aria-label='A 7B model shrinks from 28GB in float32 to 14GB in bfloat16'><g font-family='monospace'><rect x='90' y='30' width='150' height='40' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='165' y='55' text-anchor='middle' font-size='13' fill='#C93B3B' font-weight='bold'>7B · fp32 ≈ 28 GB</text><text x='265' y='55' text-anchor='middle' font-size='16' fill='#6B645E'>→</text><rect x='300' y='38' width='110' height='40' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='355' y='63' text-anchor='middle' font-size='13' fill='#1a5c38' font-weight='bold'>bf16 ≈ 14 GB</text><text x='260' y='108' text-anchor='middle' font-size='12' fill='#2C2A28'>halving the dtype halves the memory bill</text><text x='260' y='128' text-anchor='middle' font-size='11' fill='#6B645E'>fits on one chip, or it doesn&apos;t — this is why labs live in bytes</text></g></svg>",
  note:"<b>Why it matters at frontier scale.</b> Weights = params × bytes/param. A 7-billion-param model is ~28 GB in float32 but ~14 GB in bfloat16 — often the difference between fitting on a single accelerator or not. Every quantization method you'll study is a cleverer way to shrink this box without losing too much precision."}
];
@@@ region name=QS
var QS=[
 {q:'1. You add two arrays you believe are the same-length vectors and expect a vector back, but a later mean comes out strangely small. No error was raised. You print shapes: one is <code>(100,)</code>, the other is <code>(100, 1)</code>, and the sum is <code>(100, 100)</code>. What happened?',
  opts:['NumPy silently truncated the longer array','Broadcasting stretched a row against a column and built a 100&times;100 grid of every pair, so your mean averaged 10,000 numbers instead of 100','The dtypes disagreed and NumPy zero-filled the gap','Nothing is wrong; (100,1) and (100,) are identical shapes'],
  ans:1, fb:'Right. <code>(100,)</code> vs <code>(100, 1)</code> are broadcast-compatible: line up the right edges, the <code>1</code> is a stamp that presses out to 100, then the missing leading dim on the <code>(100,)</code> side presses out too — giving a <code>(100, 100)</code> grid of every pairwise sum. No error fires, so the bug is silent. Fix: print <code>.shape</code> right after the op and reshape one operand (e.g. <code>.ravel()</code>) so both are truly 1-D.'},
 {q:'2. Two dimensions are broadcast-compatible if they are equal, or if one of them is ___.',
  opts:['0','1','even','a float'],
  ans:1, fb:'Right. A dimension of size 1 (or missing) is the stamp — it presses out to match whatever size sits opposite it.'},
 {q:'3. What does an array\'s <code>dtype</code> control?',
  opts:['Its shape','The kind and byte-size of each element (precision & memory)','How many dimensions it has','Which axis is the batch'],
  ans:1, fb:'Right. dtype = the shared type (int/float) and how many bytes each number costs — which sets the memory bill.'},
 {q:'4. Compared to <code>float32</code>, how much memory does <code>bfloat16</code> use per number?',
  opts:['Twice as much','The same','Half','Four times as much'],
  ans:2, fb:'Right. bfloat16 is 2 bytes vs float32\'s 4 — half the memory. That is the core lever behind quantization.'}
];
