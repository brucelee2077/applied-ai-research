---
quest_id: wf1-d02-indexing
donor: m01-day-02-indexing-slicing.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 1 · Day 2 — Indexing &amp; Slicing"
spine: "shape"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 1 · Day 2 — Indexing &amp; Slicing</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M1 Day 2</div>
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
<a class="lnav prev" href="../day-01-arrays/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t">Arrays &amp; the Shape of Data</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-03-broadcasting-dtypes/lesson.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t">Broadcasting &amp; dtypes</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 1 · Represent · Day 2</span>
      <h1>Indexing &amp; Slicing<span class="sub">Grab Any Piece of an Array</span></h1>
      <p class="lede">Yesterday you built arrays and read their <strong>shape</strong>. Today you learn to reach in and pull out exactly the piece you want — one number, a range, a whole row, a whole column. This is how every model grabs a batch, a token range, or one attention head.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><b>By the end, you'll be able to:</b> pull a single element, a slice, and a full row or column out of a 2-D array — and explain why the <code>stop</code> in <code>a:b</code> is left out.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">Indexing vs. slicing — what each one does</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 — you can make an array and read its <code>shape</code>. <b>No new tools today</b> — just two ways of reaching into an array.</div></div>

      <h4>What is indexing?</h4>
      <p><span class="term" data-tip="Grabbing a single element from an array by its position number, counting from 0.">Indexing</span> means grabbing <strong>one element</strong> by its position. Positions count from <strong>0</strong>, exactly like a Python list: <code>x[0]</code> is the first element, <code>x[2]</code> the third.</p>

      <h4>What is slicing?</h4>
      <p><span class="term" data-tip="Grabbing a range of elements with start:stop. The stop position is excluded.">Slicing</span> means grabbing a <strong>range</strong> with <code>start:stop</code>. The catch you must remember forever: the <strong>stop is excluded</strong>. <code>x[1:3]</code> gives positions 1 and 2 — not 3.</p>

      <h4>How does it relate to what you know?</h4>
      <p>This is the same <code>x[0]</code> and <code>x[1:3]</code> you already use on Python lists. Arrays add one new move: for a grid you separate the dimensions with a <strong>comma</strong> — <code>g[row, col]</code>. And a lone colon <code>:</code> means "everything on this <span class="term" data-tip="One dimension of an array. A 2-D array has two axes: axis 0 = rows, axis 1 = columns.">axis</span>."</p>
      <p>Two symbols do all the work today: the <strong>comma</strong> (separate axes) and the <strong>colon</strong> (a range, or "all"). Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> vLLM, one of the most-used serving engines for LLMs, is built around <b>PagedAttention</b> — it slices the KV cache into fixed-size blocks the exact same way an operating system pages memory. Slicing isn't just a classroom trick.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> <code>x[1:3]</code> 看起来"取到 3"，但其实取不到 — the half-open rule feels arbitrary until it burns you. In plain terms: people read <code>start:stop</code> as inclusive, so their batch is one row short (or one too many), and the off-by-one hides for hours because nothing crashes.</div></div>
      <button class="gotit" type="button">Got the two moves — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">A street of houses, then a spreadsheet</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>Before any code, lock in a picture.</p>
      <div class="relate">
        <div class="card"><span class="big">🏠</span><h5>1-D array = a street</h5><p>Houses numbered from 0. An <strong>index</strong> is one house number; a <strong>slice</strong> is a block of consecutive houses (<code>1:3</code> = houses 1 and 2).</p></div>
        <div class="card"><span class="big">📊</span><h5>2-D array = a spreadsheet</h5><p>Point to a cell with <code>[row, col]</code>. A whole row is <code>[r, :]</code>; a whole column is <code>[:, c]</code>. The colon means "all of this side."</p></div>
      </div>
      <p><strong>In one line:</strong> index picks one; slice picks a range; the comma separates the axes; the colon means "everything on this axis."</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a slice doesn't copy the houses — it's a <strong>view</strong>, a window onto the same data, so changing the slice can change the original. We'll treat slices as plain copies for now and revisit that later.</div></div>
      <p><b>直觉 / Intuition:</b> 把 <code>:</code> 想成"整条轴"，把逗号想成"换一条轴" — a colon means "all of this axis", a comma means "now the next axis." The technical point: indexing/slicing is pure address arithmetic on a regular grid, so grabbing a batch or a column is O(1) bookkeeping, not a copy loop.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Software analogy:</b> a slice is like a database <code>LIMIT/OFFSET</code> over rows already sorted on disk — you name a start and a count, and the engine reads that window without touching the rest. Same idea: <code>data[0:32]</code> hands you a training batch by pointer, not by copying 32 rows.</div></div>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The exact rules — and why labs slice all day</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The rules, precisely</h4>
      <ul>
        <li>Positions start at <strong>0</strong>. <code>x[i]</code> is one element.</li>
        <li><code>x[a:b]</code> is from <code>a</code> up to <strong>but not including</strong> <code>b</code>. So its length is <code>b − a</code>.</li>
        <li><code>x[a:b:step]</code> takes every <code>step</code>-th element; negative indices count from the end (<code>x[-1]</code> = last).</li>
        <li>For a grid, use one index or slice <strong>per axis, separated by commas</strong>: <code>g[row, col]</code>. A lone <code>:</code> means the whole axis.</li>
      </ul>

      <h4>Why a frontier lab cares</h4>
      <p>You just met vLLM's <b>PagedAttention</b> above — here's the mechanism underneath it. Slicing is how models move data all day: grab a training <strong>batch</strong> with <code>data[0:32]</code>, or pull one attention head's slice out of a big tensor. vLLM's exact move is to carve the <span class="term" data-tip="The stored keys/values from earlier tokens that a model reuses so it does not recompute them every step. You will meet it later.">KV cache</span> into fixed-size chunks with <code>cache[:, :seq_len]</code> — that's what lets it reuse memory efficiently when two requests share the exact same prompt text, instead of wasting space. It happens millions of times per second.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this causal chain:</b> positions are regular → any sub-region has a fixed address → you grab a batch/row/column with plain arithmetic, no copy-loop → data moves fast. An <strong>off-by-one slice</strong> (forgetting the stop is excluded) is one of the most common bugs there is.</div></div>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Staff / Research Engineer Lens — two silent failures:</b> (1) the off-by-one — <code>x[a:b]</code> has length <code>b − a</code>, so <code>x[1:3]</code> is 2 elements, not 3; drop a token and your labels shift by one silently. (2) the <em>view vs. copy</em> trap — a slice is a window onto the same memory, so writing into <code>batch = data[0:32]</code> can mutate your dataset in place. <b>⚖️ Trade-off:</b> views are free (no copy) but aliased; when in doubt, <code>.copy()</code> costs memory but buys safety — say which one you chose in review.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "Slicing is half-open address arithmetic: <code>x[a:b]</code> is <code>b − a</code> elements and returns a view, not a copy. That's why it's cheap — no data moves — and it's exactly how a batch or a KV-cache block is carved out. The nuance is aliasing: because a slice shares memory, an in-place write can silently corrupt the source, so I copy when I need isolation."</div></div>
      <button class="gotit" type="button">Got the rules</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Leave today's evidence (strongly recommended)</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Understanding ≠ being able to write it. Grab a few pieces yourself — that's what makes the day count. Pick one:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m01-shape-of-data/day-02-indexing-slicing/experiment.py</code>. On a 1-D array, print <code>x[0]</code>, <code>x[-1]</code>, and the slice <code>x[1:3]</code>. On a 2-D array, print one cell <code>g[1, 2]</code>, a whole row <code>g[0, :]</code>, and a whole column <code>g[:, 1]</code>. Run it with <code>python3 sessions/m01-shape-of-data/day-02-indexing-slicing/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">ask Claude to build it</span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Help me build my Module 1 Day 2 artifact.

Create sessions/m01-shape-of-data/day-02-indexing-slicing/experiment.py that, with a comment on each step:
1. Makes x = np.array([10,20,30,40]); prints x[0], x[2], and x[-1].
2. Prints the slices x[1:3] (expect [20,30]), x[:2], and x[::2]; note in a comment that the stop is excluded.
3. Makes g = np.array([[1,2,3],[4,5,6]]); prints g[1,2] (one cell), g[0,:] (a whole row), and g[:,1] (a whole column).
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>Acceptance criteria</h4>
      <ul>
        <li>On a 1-D array you print <code>x[0]</code>, <code>x[-1]</code>, and <code>x[1:3]</code>, and the slice has exactly 2 elements.</li>
        <li>On a 2-D array you print one cell <code>g[1, 2]</code>, a whole row <code>g[0, :]</code>, and a whole column <code>g[:, 1]</code>.</li>
        <li>You can state the length of any <code>x[a:b]</code> as <code>b − a</code> before running it.</li>
        <li>You note in a comment whether each result is a view or a copy.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-02-indexing-slicing/log.md</code>: (1) the one sentence you'd tell a teammate about the half-open slice rule; (2) which slice result surprised you, and why; (3) one thing you're still unsure about (view vs. copy is fair game).</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 1 · Day 2 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Indexing &amp; Slicing</b>.<br>Next up: <b>Broadcasting &amp; dtypes</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  elem:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">10</span>, <span class="num">20</span>, <span class="num">30</span>, <span class="num">40</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x[<span class="num">0</span>]          <span class="dim"># first element — counting starts at 0</span>\n'+
    '<span class="hl">10</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x[<span class="num">2</span>]\n<span class="hl">30</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x[<span class="num">-1</span>]         <span class="dim"># negative counts from the end</span>\n'+
    '<span class="hl">40</span>',
    take:'<b>①  One number by position.</b> Arrays count from <b>0</b>, just like Python lists. <code>x[-1]</code> is a handy shortcut for the last element.'},
  slice:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">10</span>, <span class="num">20</span>, <span class="num">30</span>, <span class="num">40</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x[<span class="num">1</span>:<span class="num">3</span>]        <span class="dim"># start 1, stop 3 — stop is EXCLUDED</span>\n'+
    '<span class="ok">array([20, 30])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x[:<span class="num">2</span>]         <span class="dim"># from the start up to (not incl.) 2</span>\n'+
    '<span class="ok">array([10, 20])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x[::<span class="num">2</span>]        <span class="dim"># every 2nd element</span>\n'+
    '<span class="ok">array([10, 30])</span>',
    take:'<b>②  A range with <code>start:stop</code>.</b> The big gotcha: the <b>stop is excluded</b> — <code>1:3</code> gives positions 1 and 2, not 3. A lone <code>:</code> means the whole axis.'},
  twod:{html:'<span class="prompt">&gt;&gt;&gt;</span> g = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> g[<span class="num">1</span>, <span class="num">2</span>]       <span class="dim"># row 1, column 2</span>\n'+
    '<span class="hl">6</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> g[<span class="num">0</span>, :]       <span class="dim"># row 0, ALL columns</span>\n'+
    '<span class="ok">array([1, 2, 3])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> g[:, <span class="num">1</span>]       <span class="dim"># ALL rows, column 1</span>\n'+
    '<span class="ok">array([2, 5])</span>',
    take:'<b>③  Two dimensions, split by a comma.</b> <code>[row, col]</code> picks one cell; <code>[r, :]</code> is a whole row; <code>[:, c]</code> is a whole column. The comma splits the axes; <code>:</code> means all of that axis.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='A 1-D array with positions labeled 0 to 3 below each value'><text x='260' y='20' text-anchor='middle' font-size='12' fill='#2C2A28' font-family='monospace'>x = [10, 20, 30, 40]</text><g font-family='monospace'><rect x='150' y='34' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='207' y='34' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='264' y='34' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='321' y='34' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='177' y='63' text-anchor='middle' font-size='15' fill='#1F6280'>10</text><text x='234' y='63' text-anchor='middle' font-size='15' fill='#1F6280'>20</text><text x='291' y='63' text-anchor='middle' font-size='15' fill='#1F6280'>30</text><text x='348' y='63' text-anchor='middle' font-size='15' fill='#1F6280'>40</text><text x='177' y='98' text-anchor='middle' font-size='12' fill='#9A7208'>0</text><text x='234' y='98' text-anchor='middle' font-size='12' fill='#9A7208'>1</text><text x='291' y='98' text-anchor='middle' font-size='12' fill='#9A7208'>2</text><text x='348' y='98' text-anchor='middle' font-size='12' fill='#9A7208'>3</text></g><text x='260' y='120' text-anchor='middle' font-size='11' fill='#6B645E' font-family='monospace'>positions (indices) count from 0</text></svg>",
  note:"<b>Every slot has a position number, starting at 0.</b> The value <code>10</code> lives at index 0, <code>40</code> at index 3. Indexing and slicing are just ways of naming these positions."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='x[2] selects the single box at index 2, value 30'><g font-family='monospace'><rect x='150' y='30' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='207' y='30' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='264' y='30' width='55' height='46' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2.5'/><rect x='321' y='30' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='177' y='59' text-anchor='middle' font-size='15' fill='#1F6280'>10</text><text x='234' y='59' text-anchor='middle' font-size='15' fill='#1F6280'>20</text><text x='291' y='59' text-anchor='middle' font-size='15' fill='#C93B3B' font-weight='bold'>30</text><text x='348' y='59' text-anchor='middle' font-size='15' fill='#1F6280'>40</text></g><text x='260' y='98' text-anchor='middle' font-size='13' fill='#C93B3B' font-family='monospace' font-weight='bold'>x[2]  →  30</text></svg>",
  note:"<b>Indexing picks exactly one.</b> <code>x[2]</code> reaches into position 2 and returns the single value <code>30</code>. One index in, one number out."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='x[1:3] selects boxes at index 1 and 2 but excludes index 3'><g font-family='monospace'><rect x='150' y='30' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='207' y='30' width='55' height='46' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2.5'/><rect x='264' y='30' width='55' height='46' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2.5'/><rect x='321' y='30' width='55' height='46' rx='6' fill='#FDF9F3' stroke='#E5DFD6' stroke-width='2' stroke-dasharray='4,3'/><text x='177' y='59' text-anchor='middle' font-size='15' fill='#1F6280'>10</text><text x='234' y='59' text-anchor='middle' font-size='15' fill='#1a5c38' font-weight='bold'>20</text><text x='291' y='59' text-anchor='middle' font-size='15' fill='#1a5c38' font-weight='bold'>30</text><text x='348' y='59' text-anchor='middle' font-size='15' fill='#9399B2'>40</text><text x='348' y='96' text-anchor='middle' font-size='10.5' fill='#C93B3B'>3 excluded</text></g><text x='234' y='112' text-anchor='middle' font-size='13' fill='#2D8B55' font-family='monospace' font-weight='bold'>x[1:3]  →  [20, 30]</text></svg>",
  note:"<b>Slicing picks a range — and the stop is left out.</b> <code>x[1:3]</code> takes positions 1 and 2, but <b>not</b> 3. Length = stop − start = 3 − 1 = 2. Forgetting the excluded stop is the classic off-by-one bug."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='x[-1] selects the last box, value 40'><g font-family='monospace'><rect x='150' y='30' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='207' y='30' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='264' y='30' width='55' height='46' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='321' y='30' width='55' height='46' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2.5'/><text x='177' y='59' text-anchor='middle' font-size='15' fill='#1F6280'>10</text><text x='234' y='59' text-anchor='middle' font-size='15' fill='#1F6280'>20</text><text x='291' y='59' text-anchor='middle' font-size='15' fill='#1F6280'>30</text><text x='348' y='59' text-anchor='middle' font-size='15' fill='#9A7208' font-weight='bold'>40</text></g><text x='260' y='98' text-anchor='middle' font-size='13' fill='#9A7208' font-family='monospace' font-weight='bold'>x[-1]  →  40   (count from the end)</text></svg>",
  note:"<b>Negative indices count backwards.</b> <code>x[-1]</code> is the last element, <code>x[-2]</code> the second-to-last. Handy when you don't know or don't want to compute the length."},
 {viz:"<svg viewBox='0 0 520 150' role='img' aria-label='2-D array with g[1,2] selecting the cell at row 1 column 2, value 6'><text x='260' y='18' text-anchor='middle' font-size='12' fill='#2C2A28' font-family='monospace'>g[row, col]</text><g font-family='monospace' font-size='15' text-anchor='middle'><rect x='200' y='30' width='44' height='38' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='246' y='30' width='44' height='38' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='292' y='30' width='44' height='38' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='200' y='70' width='44' height='38' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='246' y='70' width='44' height='38' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='292' y='70' width='44' height='38' rx='5' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2.5'/><text x='222' y='55' fill='#1F6280'>1</text><text x='268' y='55' fill='#1F6280'>2</text><text x='314' y='55' fill='#1F6280'>3</text><text x='222' y='95' fill='#1F6280'>4</text><text x='268' y='95' fill='#1F6280'>5</text><text x='314' y='95' fill='#C93B3B' font-weight='bold'>6</text></g><text x='170' y='93' text-anchor='end' font-size='10.5' fill='#6B645E' font-family='monospace'>row 1</text><text x='260' y='134' text-anchor='middle' font-size='13' fill='#C93B3B' font-family='monospace' font-weight='bold'>g[1, 2]  →  6</text></svg>",
  note:"<b>Two axes, one comma.</b> In a grid, <code>g[1, 2]</code> means row 1, column 2 → the value <code>6</code>. The number before the comma walks down the rows; the number after walks across the columns."},
 {viz:"<svg viewBox='0 0 520 170' role='img' aria-label='g[0,:] selects the whole first row and g[:,1] selects the whole second column'><g font-family='monospace' font-size='14' text-anchor='middle'><rect x='120' y='28' width='40' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2.5'/><rect x='162' y='28' width='40' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2.5'/><rect x='204' y='28' width='40' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2.5'/><rect x='120' y='64' width='40' height='34' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='162' y='64' width='40' height='34' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='204' y='64' width='40' height='34' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='140' y='51' fill='#1a5c38' font-weight='bold'>1</text><text x='182' y='51' fill='#1a5c38' font-weight='bold'>2</text><text x='224' y='51' fill='#1a5c38' font-weight='bold'>3</text><text x='140' y='87'>4</text><text x='182' y='87'>5</text><text x='224' y='87'>6</text><text x='182' y='118' fill='#2D8B55' font-weight='bold'>g[0, :] → [1,2,3]</text><rect x='320' y='28' width='40' height='34' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='362' y='28' width='40' height='34' rx='5' fill='#FCF3DC' stroke='#C99A12' stroke-width='2.5'/><rect x='404' y='28' width='40' height='34' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='320' y='64' width='40' height='34' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='362' y='64' width='40' height='34' rx='5' fill='#FCF3DC' stroke='#C99A12' stroke-width='2.5'/><rect x='404' y='64' width='40' height='34' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='340' y='51' fill='#1F6280'>1</text><text x='382' y='51' fill='#9A7208' font-weight='bold'>2</text><text x='424' y='51' fill='#1F6280'>3</text><text x='340' y='87'>4</text><text x='382' y='87' fill='#9A7208' font-weight='bold'>5</text><text x='424' y='87'>6</text><text x='382' y='118' fill='#9A7208' font-weight='bold'>g[:, 1] → [2,5]</text></g><text x='260' y='150' text-anchor='middle' font-size='12' fill='#6B645E' font-family='monospace'>the colon means: take EVERYTHING on that axis</text></svg>",
  note:"<b>The colon grabs a whole row or column.</b> <code>g[0, :]</code> fixes the row to 0 and takes all columns → the first row. <code>g[:, 1]</code> takes all rows but fixes the column to 1 → the second column. This exact move is how a model slices one feature, one head, or one batch out of a big tensor."}
];
@@@ region name=QS
var QS=[
 {q:'1. With <code>x = np.array([10,20,30,40])</code>, what is <code>x[1:3]</code>?',
  opts:['[10, 20]','[20, 30]','[20, 30, 40]','[10, 20, 30]'],
  ans:1, fb:'Right. Start at 1, stop at 3 — but the stop is excluded, so you get positions 1 and 2: [20, 30].'},
 {q:'2. From a 2-D array <code>g</code>, what does <code>g[0, :]</code> give you?',
  opts:['The first column','The whole first row','A single element g[0][0]','An error — you need two numbers'],
  ans:1, fb:'Right. Row fixed to 0, colon = all columns → the whole first row. Swap them (g[:, 0]) to get the first column.'},
 {q:'3. What is <code>x[-1]</code>?',
  opts:['The first element','The last element','An error','The second-to-last element'],
  ans:1, fb:'Right. Negative indices count from the end, so -1 is the last element.'},
 {q:'4. NumPy arrays are indexed starting from which number?',
  opts:['1','0','-1','It depends on the array'],
  ans:1, fb:'Right. Position counting starts at 0, exactly like Python lists.'}
];
