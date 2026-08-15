---
quest_id: wf1-d01-arrays
donor: m01-day-01-arrays.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 1 · Day 1 — Numbers, Arrays, and the Shape of Data"
spine: "shape"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 1 · Day 1 — Numbers, Arrays, and the Shape of Data</title>
@@@ region name=brand_sub
<div class="brand-sub">Foundations · M1 Day 1</div>
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
<span class="lnav prev disabled"><span class="d">Start</span><span class="t">Foundations begin</span></span>
@@@ region name=nav_next
<a class="lnav next" href="../day-02-indexing-slicing/lesson.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t">Indexing &amp; Slicing</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 1 · Represent · Day 1</span>
      <h1>Numbers, Arrays,<span class="sub">and the Shape of Data</span></h1>
      <p class="lede">Every image, sentence, and weight a neural network ever touches is the same thing underneath: a grid of numbers called an <strong>array</strong>. Today you learn what an array is — and how to read its <strong>shape</strong>, the one skill every later lesson leans on.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><b>By the end, you'll be able to:</b> create a NumPy array, read its <code>shape</code> and <code>ndim</code> out loud, and say in one sentence why arrays beat plain Python lists for math.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">First, what is an array — and what is NumPy?</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> a little Python — you know a variable, a <code>list</code> like <code>[1, 2, 3]</code>, and a <code>for</code> loop. <b>No math or NumPy background needed</b> — we start from zero.</div></div>

      <h4>What is NumPy?</h4>
      <p><span class="term" data-tip="A Python library for fast math on whole arrays of numbers at once. Imported as np.">NumPy</span> is a Python library for doing math on big collections of numbers <strong>quickly</strong>. It is the foundation almost every other data and machine-learning tool is built on. By convention you import it and call it <code>np</code>:</p>
      <div class="callout c-info"><span class="ic">📦</span><div><code>import numpy as np</code><br><span style="font-size:.85rem;color:var(--text2)">From now on, <code>np.something(...)</code> means "use NumPy's version of something."</span></div></div>

      <h4>What is an array?</h4>
      <p>An <span class="term" data-tip="An ordered grid of numbers, all the same type, stored together in memory.">array</span> is an <strong>ordered grid of numbers</strong>, all of the same type, stored together. You make one from a plain list:</p>
      <div class="callout c-info"><span class="ic">🔢</span><div><code>x = np.array([1, 2, 3])</code><br><span style="font-size:.85rem;color:var(--text2)">This turns a Python list into a NumPy array — a tidy row of three numbers.</span></div></div>

      <h4>How is that different from a Python list?</h4>
      <p>A Python list is a loose bag: it can hold anything (numbers, text, other lists), in any mix, and to do math you have to loop over it item by item. An array is stricter and, because of that, far faster: <strong>one type, one regular grid</strong> — so the computer can crunch every number at once.</p>
      <p>The single most important thing about an array is its <span class="term" data-tip="A tuple giving the size of an array along each dimension, e.g. (2, 3) = 2 rows, 3 columns.">shape</span> — the sizes of its grid. That is today's puzzle. Keep going 👇</p>
      <div class="callout c-info"><span class="ic">🏭</span><div><b>A real one:</b> GPT-3's biggest weight matrix is a rigid grid of shape <code>(12288, 49152)</code> — over 600 million numbers, one matrix, inside one model. Shape is not a toy idea.</div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> "array" 听起来只是 Python list 的另一个名字，于是很多人跳过 shape 直接写代码。In plain terms: people treat an array as "just a fast list" and never learn to read its <code>shape</code> — then every later bug (a matmul that won't line up, a batch axis in the wrong place) is really a shape they never learned to see.</div></div>
      <button class="gotit" type="button">Got what an array is — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">Shopping bag vs. egg carton</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <p>Before any code, lock in a picture.</p>
      <div class="relate">
        <div class="card"><span class="big">🛍️</span><h5>Python list = shopping bag</h5><p>Throw anything in — an apple, a book, another bag. Flexible, but to weigh every apple you must pull them out <strong>one at a time</strong>.</p></div>
        <div class="card"><span class="big">🥚</span><h5>NumPy array = egg carton</h5><p>Fixed slots, all holding the <strong>same kind of thing</strong>, in a neat grid. You can point to any slot by row and column — and act on the whole carton at once.</p></div>
      </div>
      <p><strong>In one line:</strong> a list is flexible but slow for math; an array is rigid but fast, because everything sits in a regular grid.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: an egg carton is only 2-D (rows × columns), but an array can have <strong>3, 4, or more</strong> dimensions — a stack of cartons, a warehouse of stacks. We'll build up to that.</div></div>
      <p><b>直觉 / Intuition:</b> 真正重要的不是"数字"，而是"格子的形状" — the picture that matters is the <em>grid</em>, not the numbers inside it. The technical point: an array bundles one dtype + one shape + a contiguous block of memory, so the hardware can act on the whole grid in a single instruction.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Software analogy:</b> a Python <code>list</code> is a folder of loose files you open one at a time; a NumPy array is one fixed-width database column — same type in every row, so the engine scans it in one pass. That "one pass over a regular grid" is exactly what a GPU does to a weight matrix.</div></div>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">The three facts every array carries — and why labs live in them</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>Every array has three labels</h4>
      <ul>
        <li><strong><span class="term" data-tip="The single data type every element of an array shares, e.g. int32 or float32.">dtype</span></strong> — the one type every element shares (e.g. <code>int32</code>, <code>float32</code>). No mixing.</li>
        <li><strong>shape</strong> — a tuple of the sizes along each dimension: <code>(2, 3)</code> = 2 rows, 3 columns.</li>
        <li><strong><code>ndim</code></strong> — how many dimensions there are (the length of the shape tuple).</li>
      </ul>

      <h4>Why the rigid grid is a superpower</h4>
      <p>Because every element is the same type and packed together, the computer can do one instruction on <strong>many numbers at once</strong> — this is <span class="term" data-tip="Doing one operation on every element of an array at once, in fast compiled code, instead of a Python loop.">vectorization</span>. <code>x + 10</code> adds 10 to a million numbers with no Python loop, running in fast compiled code.</p>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Remember this causal chain:</b> same type + regular grid → packed together in memory → one instruction touches many numbers → fast math → training big models becomes possible. <strong>No grid, no speed.</strong></div></div>

      <h4>Why a frontier lab cares</h4>
      <p>That same <code>(12288, 49152)</code> matrix from the top of this page does not sit alone. It lives inside one of GPT-3's 96 layers, each layer 12,288 numbers wide. A handful of matrices like it repeat in every one of those 96 layers. Every weight, every batch of tokens, every image inside a model is an array (a <span class="term" data-tip="The general word for an array with any number of dimensions (3-D, 4-D, ...). Used everywhere in ML.">tensor</span>). The number-one bug in machine-learning code is a <strong>shape mismatch</strong> — two arrays that don't line up. Reading shapes fluently is the skill that everything later (attention, batching, scaling) is built on. That is why Day 1 is arrays.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Staff / Research Engineer Lens — the silent failure:</b> a shape bug rarely crashes. NumPy will happily broadcast a <code>(32,)</code> against a <code>(32, 1)</code> and hand you a <code>(32, 32)</code> — training keeps running, the loss looks plausible, and you ship a model that learned the wrong thing. <b>⚖️ Trade-off in code review:</b> a one-line <code>assert x.shape == (B, T, C)</code> at each boundary costs nothing and kills the class of bug that otherwise eats an afternoon of debugging.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "An array is a dtype plus a shape over one contiguous block of memory — that is what makes it vectorizable: one instruction over the whole grid. The nuance is that the costliest bugs in ML code aren't crashes, they're silent shape mismatches that broadcast into a wrong-but-plausible result, so I annotate expected shapes at every boundary."</div></div>
      <button class="gotit" type="button">Got the causal chain</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Leave today's evidence (strongly recommended)</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Understanding ≠ being able to write it. Make a few arrays and print their shapes yourself — that's what makes the day count. Pick one:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m01-shape-of-data/day-01-arrays/experiment.py</code>. Build a 1-D array and a 2-D array, print <code>.shape</code> and <code>.ndim</code> for each, do a vectorized <code>x + 10</code>, then <code>reshape</code> a 6-element array into shape <code>(2, 3)</code> and print it. Run it with <code>python3 sessions/m01-shape-of-data/day-01-arrays/experiment.py</code>. (No NumPy? <code>pip install numpy</code>.)</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">ask Claude to build it</span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Help me build my Module 1 Day 1 artifact.

Create sessions/m01-shape-of-data/day-01-arrays/experiment.py that:
1. Builds a 1-D array np.array([1,2,3]); prints it, its .shape (3,), its .ndim (1), and its .dtype.
2. Builds a 2-D array np.array([[1,2,3],[4,5,6]]); prints its .shape (2,3) and .ndim (2).
3. Proves vectorization: prints x + 10 and x * 2 for the 1-D array, with a comment noting no loop was used.
4. Reshapes np.arange(6) into shape (2,3) and prints the result, showing the same 6 numbers rearranged.
Add a one-line comment on each step, then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>Acceptance criteria</h4>
      <ul>
        <li>The script prints <code>.shape</code>, <code>.ndim</code>, and <code>.dtype</code> for both a 1-D and a 2-D array.</li>
        <li>It shows one vectorized op (<code>x + 10</code>) with no Python loop.</li>
        <li>It reshapes a 6-element array to <code>(2, 3)</code>, and the printed grid holds the same six numbers.</li>
        <li>You can read each printed shape out loud, outer-to-inner, without hesitating.</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-01-arrays/log.md</code>: (1) the one sentence you'd tell a teammate about what "shape" is; (2) which shape you found hardest to read, and why; (3) one thing you're still unsure about.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 1 · Day 1 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Numbers, Arrays, and the Shape of Data</b>.<br>Next up: <b>Indexing &amp; Slicing</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  list:{html:'<span class="prompt">&gt;&gt;&gt;</span> nums = [<span class="num">1</span>, <span class="num">2</span>, <span class="num">3</span>]   <span class="dim"># a normal Python list</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> nums + <span class="num">10</span>\n'+
    '<span class="bad">TypeError: can only concatenate list (not "int") to list</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> [n + <span class="num">10</span> <span class="kw">for</span> n <span class="kw">in</span> nums]   <span class="dim"># must loop over every item</span>\n'+
    '<span class="ok">[11, 12, 13]</span>',
    take:'<b>①  A list is flexible but "dumb" about math.</b> <code>nums + 10</code> is an error — to add 10 to each number you must write a loop. Fine for 3 numbers; painfully slow for a million.'},
  arr:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">import</span> numpy <span class="kw">as</span> np\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">1</span>, <span class="num">2</span>, <span class="num">3</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x + <span class="num">10</span>          <span class="dim"># one op, applied to all — no loop</span>\n'+
    '<span class="ok">array([11, 12, 13])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x * <span class="num">2</span>\n'+
    '<span class="ok">array([2, 4, 6])</span>',
    take:'<b>②  A NumPy array does math on every element at once.</b> This is <b>vectorization</b> — it runs in fast compiled code, not a slow Python loop. Same idea scales to millions of numbers.'},
  shape:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">1</span>, <span class="num">2</span>, <span class="num">3</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x.shape       <span class="dim"># a 1-D array: 3 numbers in a row</span>\n'+
    '<span class="hl">(3,)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x.ndim\n<span class="num">1</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> g = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> g.shape       <span class="dim"># 2 rows, 3 columns</span>\n'+
    '<span class="hl">(2, 3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> g.ndim\n<span class="num">2</span>',
    take:'<b>③  Every array carries its shape and ndim.</b> <code>shape</code> = the size along each dimension; <code>ndim</code> = how many dimensions. Reading shapes fluently is THE core ML skill — most bugs are shape mismatches.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='A single number, a scalar, shape empty tuple'><text x='260' y='24' text-anchor='middle' font-size='12' fill='#2C2A28' font-family='monospace'>scalar (0-D)</text><rect x='233' y='38' width='54' height='48' rx='8' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='69' text-anchor='middle' font-size='17' font-family='monospace' fill='#1F6280'>21</text><text x='260' y='108' text-anchor='middle' font-size='12' fill='#6B645E' font-family='monospace'>shape: ()</text></svg>",
  note:"<b>Start with one number.</b> A single value — a temperature, a price — is a <b>scalar</b>. Its shape is the empty tuple <code>()</code>: zero dimensions. Everything bigger is just scalars arranged in a grid."},
 {viz:"<svg viewBox='0 0 520 120' role='img' aria-label='A row of three numbers, a 1-D array, shape (3,)'><text x='260' y='24' text-anchor='middle' font-size='12' fill='#2C2A28' font-family='monospace'>1-D array (vector)</text><rect x='180' y='38' width='54' height='48' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='236' y='38' width='54' height='48' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='292' y='38' width='54' height='48' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='207' y='69' text-anchor='middle' font-size='16' font-family='monospace' fill='#1F6280'>1</text><text x='263' y='69' text-anchor='middle' font-size='16' font-family='monospace' fill='#1F6280'>2</text><text x='319' y='69' text-anchor='middle' font-size='16' font-family='monospace' fill='#1F6280'>3</text><text x='260' y='108' text-anchor='middle' font-size='12' fill='#6B645E' font-family='monospace'>shape: (3,)  ·  3 numbers in a row</text></svg>",
  note:"<b>Put numbers in a row.</b> A single line of values is a <b>1-D array</b> (a vector). Shape <code>(3,)</code> reads as: three numbers along one axis. A list of prices, a word turned into numbers — all vectors."},
 {viz:"<svg viewBox='0 0 520 150' role='img' aria-label='A 2 by 3 grid, a 2-D array, shape (2,3)'><text x='260' y='20' text-anchor='middle' font-size='12' fill='#2C2A28' font-family='monospace'>2-D array (matrix)</text><g font-family='monospace' font-size='15' fill='#1F6280' text-anchor='middle'><rect x='188' y='30' width='48' height='40' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='238' y='30' width='48' height='40' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='288' y='30' width='48' height='40' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='188' y='72' width='48' height='40' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='238' y='72' width='48' height='40' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><rect x='288' y='72' width='48' height='40' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='212' y='56'>1</text><text x='262' y='56'>2</text><text x='312' y='56'>3</text><text x='212' y='98'>4</text><text x='262' y='98'>5</text><text x='312' y='98'>6</text></g><text x='150' y='95' text-anchor='middle' font-size='11' fill='#6B645E' font-family='monospace'>2 rows</text><text x='260' y='134' text-anchor='middle' font-size='12' fill='#6B645E' font-family='monospace'>shape: (2, 3)  ·  2 rows, 3 columns</text></svg>",
  note:"<b>Stack rows into a grid.</b> A <b>2-D array</b> (a matrix) has rows and columns. Shape <code>(2, 3)</code> = 2 rows, 3 columns — read outermost first, left to right. A small greyscale image or a spreadsheet is a matrix."},
 {viz:"<svg viewBox='0 0 520 160' role='img' aria-label='Two stacked grids, a 3-D array, shape (2,2,3)'><text x='260' y='18' text-anchor='middle' font-size='12' fill='#2C2A28' font-family='monospace'>3-D array (tensor)</text><rect x='225' y='34' width='150' height='62' rx='7' fill='#D6E9F1' stroke='#2A7B9B' stroke-width='2'/><rect x='195' y='54' width='150' height='62' rx='7' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><line x1='225' y1='54' x2='195' y2='74' stroke='#2A7B9B' stroke-width='1'/><line x1='375' y1='54' x2='345' y2='74' stroke='#2A7B9B' stroke-width='1'/><text x='270' y='90' text-anchor='middle' font-size='12' font-family='monospace' fill='#1F6280'>grid 0 &amp; grid 1</text><text x='260' y='140' text-anchor='middle' font-size='12' fill='#6B645E' font-family='monospace'>shape: (2, 2, 3)  ·  2 grids, each 2 x 3</text></svg>",
  note:"<b>Stack grids to go 3-D.</b> Shape <code>(2, 2, 3)</code> = 2 grids, each 2 rows by 3 columns. A color image is 3-D: height x width x 3 color channels. Arrays with any number of dimensions are called <b>tensors</b>."},
 {viz:"<svg viewBox='0 0 520 150' role='img' aria-label='Reading the shape tuple (2,3): first number is rows, second is columns'><text x='260' y='60' text-anchor='middle' font-size='30' font-family='monospace' fill='#1F6280' font-weight='bold'>( 2 , 3 )</text><line x1='222' y1='72' x2='210' y2='104' stroke='#C99A12' stroke-width='2' marker-end='url(#a)'/><line x1='298' y1='72' x2='312' y2='104' stroke='#C99A12' stroke-width='2' marker-end='url(#a)'/><text x='200' y='122' text-anchor='middle' font-size='12' fill='#9A7208' font-family='monospace'>outer: 2 rows</text><text x='322' y='122' text-anchor='middle' font-size='12' fill='#9A7208' font-family='monospace'>inner: 3 columns</text><defs><marker id='a' markerWidth='7' markerHeight='7' refX='5' refY='3' orient='auto'><path d='M0 0 L6 3 L0 6 z' fill='#C99A12'/></marker></defs></svg>",
  note:"<b>A shape is just the sizes, outer to inner.</b> Read left to right: the first number is the biggest group, the last is the smallest slot. If you can read a shape, you can predict what an operation will do to it — that is the whole game."},
 {viz:"<svg viewBox='0 0 520 160' role='img' aria-label='An ML batch tensor of shape (32, 8, 4): batch, tokens, features'><rect x='150' y='40' width='220' height='48' rx='8' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2.5'/><text x='260' y='70' text-anchor='middle' font-size='20' font-family='monospace' fill='#1a5c38' font-weight='bold'>(32, 8, 4)</text><text x='185' y='112' text-anchor='middle' font-size='11' fill='#6B645E' font-family='monospace'>32 sentences</text><text x='260' y='112' text-anchor='middle' font-size='11' fill='#6B645E' font-family='monospace'>8 words each</text><text x='340' y='112' text-anchor='middle' font-size='11' fill='#6B645E' font-family='monospace'>4 numbers/word</text><text x='260' y='138' text-anchor='middle' font-size='12' fill='#2D8B55' font-weight='bold'>one tensor a model trains on</text></svg>",
  note:"<b>Why frontier labs live in shapes.</b> A batch of 32 sentences, each 8 words, each word turned into 4 numbers, is one tensor of shape <code>(32, 8, 4)</code>. Every weight, activation, and batch in a model is an array like this — and the #1 bug is a shape that doesn't line up. That is why we started here."}
];
@@@ region name=QS
var QS=[
 {q:'1. What is the shape of <code>np.array([[1,2,3],[4,5,6]])</code>?',
  opts:['(6,)','(3, 2)','(2, 3)','(2, 3, 1)'],
  ans:2, fb:'Right. Read outer-to-inner: 2 rows, then 3 columns → (2, 3).'},
 {q:'2. Why prefer a NumPy array over a Python list for doing math on many numbers?',
  opts:['Arrays can hold different types in each slot','All same type in a regular grid → one operation runs on every element at once (vectorization), fast','Lists cannot store numbers','Arrays are just a shorter way to write lists'],
  ans:1, fb:'Right. The rigid, same-type grid is exactly what lets the computer crunch all elements at once, in fast compiled code.'},
 {q:'3. What does <code>.ndim</code> tell you?',
  opts:['The total number of elements','The number of dimensions (how many numbers are in the shape tuple)','The data type','The largest value in the array'],
  ans:1, fb:'Right. ndim = number of dimensions = the length of the shape tuple. A matrix has ndim 2.'},
 {q:'4. A batch of 32 sentences, each 8 words, each word as 4 numbers — what shape is that tensor?',
  opts:['(4, 8, 32)','(32, 8, 4)','(32, 4)','(8, 32, 4)'],
  ans:1, fb:'Right. Outer-to-inner: 32 sentences, 8 words each, 4 numbers per word → (32, 8, 4).'}
];
