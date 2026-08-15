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
<div class="brand-sub"><span class="lang-en">Foundations · M1 Day 4</span><span class="lang-zh">基础 · M1 第 4 天</span></div>
@@@ region name=sidebar_nav
<nav aria-label="Sections">
      <div class="nav-group-label"><span class="lang-en">Module 01 · Represent</span><span class="lang-zh">Module 01 · 表示</span></div>
      <button class="nav-link" data-target="home"><span class="nl-dot"></span><span class="lang-en">Start here</span><span class="lang-zh">从这里开始</span></button>
      <button class="nav-link" data-target="s1"><span class="nl-dot"></span><span class="lang-en">1 · What is it</span><span class="lang-zh">1 · 这是什么</span></button>
      <button class="nav-link" data-target="s2"><span class="nl-dot"></span><span class="lang-en">2 · Intuition</span><span class="lang-zh">2 · 直觉</span></button>
      <button class="nav-link" data-target="s3"><span class="nl-dot"></span><span class="lang-en">3 · Playground</span><span class="lang-zh">3 · 玩一玩</span></button>
      <button class="nav-link" data-target="s4"><span class="nl-dot"></span><span class="lang-en">4 · Mechanism &amp; why</span><span class="lang-zh">4 · 原理和为什么</span></button>
      <button class="nav-link" data-target="s5"><span class="nl-dot"></span><span class="lang-en">5 · Build it up</span><span class="lang-zh">5 · 一步步搭起来</span></button>
      <button class="nav-link" data-target="s6"><span class="nl-dot"></span><span class="lang-en">6 · Quiz</span><span class="lang-zh">6 · 小测</span></button>
      <button class="nav-link" data-target="s7"><span class="nl-dot"></span><span class="lang-en">7 · Produce</span><span class="lang-zh">7 · 动手做</span></button>
    </nav>
@@@ region name=nav_prev
<a class="lnav prev" href="../day-03-broadcasting-dtypes/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t"><span class="lang-en">Broadcasting &amp; dtypes</span><span class="lang-zh">broadcasting 与 dtype</span></span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-05-logs-and-exponents/lesson.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t"><span class="lang-en">Logs &amp; Exponents</span><span class="lang-zh">log 与 exponent</span></span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker"><span class="lang-en">Module 1 · Represent · Day 4</span><span class="lang-zh">Module 1 · 表示 · 第 4 天</span></span>
      <h1><span class="lang-en">Matmul &amp; Shapes</span><span class="lang-zh">matmul 与 shape</span><span class="sub"><span class="lang-en">The One Operation Every Layer Is Built From</span><span class="lang-zh">每一层都由这一个运算搭出来</span></span></h1>
      <p class="lede"><span class="lang-en">Almost all of the arithmetic in a neural network is a single operation: <strong>matrix multiplication</strong>. Learn its shape rule — <code>(m, k) @ (k, n) = (m, n)</code> — and you can read what any layer does to its data, and why chips are built the way they are.</span><span class="lang-zh">神经网络里几乎所有的算术，都是同一个运算：<strong>matmul（矩阵乘法）</strong>。它有一条 shape（形状）规则：<code>(m, k) @ (k, n) = (m, n)</code>。记住这条规则，你就能看懂任何一层对数据做了什么。你也会明白，芯片为什么被造成今天这个样子。</span></p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><span class="lang-en"><b>By the end, you'll be able to:</b> read a matmul's output shape from its two input shapes, say why the inner dimensions must match, and explain why a neural-network layer <em>is</em> a matmul.</span><span class="lang-zh"><b>学完这一天，你能做到：</b>看着两个输入的 shape，就说出 matmul 输出的 shape。说清楚中间那两个 dimension（维度）为什么必须一样。还能讲明白，为什么神经网络的一层<em>就是</em>一个 matmul。</span></div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h"><span class="lang-en">What matrix multiplication is</span><span class="lang-zh">matmul 到底是什么</span></span><span class="sec-tag"><span class="lang-en">What is it</span><span class="lang-zh">这是什么</span></span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><span class="lang-en"><b>What this assumes:</b> Days 1–3 — arrays, shape, and element-wise math. <b>No new tools</b>. If you multiplied two numbers and summed on Day 1, you already did the core step.</span><span class="lang-zh"><b>你需要先会：</b>第 1 到第 3 天的内容 —— array（数组）、shape，还有 element-wise（一个对一个算）的运算。<b>不需要新工具。</b>第 1 天你把两个数字乘起来再加到一起，那时你就已经做过今天最核心的一步了。</span></div></div>

      <div class="lang-en">
      <h4>What is a matmul?</h4>
      <p><span class="term" data-tip="Matrix multiplication: combine two 2-D arrays so each output entry is the dot product of a row of the first and a column of the second. Written with @.">Matmul</span> (matrix multiplication) combines two 2-D arrays into a new one. Each entry of the result is a <span class="term" data-tip="Multiply two equal-length lists element by element, then add up all the products — one number out.">dot product</span>: take a <strong>row</strong> from the first array and a <strong>column</strong> from the second, multiply them pairwise, and add. In code it's the <code>@</code> operator (or <code>np.matmul</code>).</p>
      </div>
      <div class="lang-zh">
      <h4>matmul 是什么？</h4>
      <p><span class="term" data-tip="矩阵乘法。把两个二维 array 合成一个新的 array。结果里的每个数字，都是第一个 array 的一行和第二个 array 的一列做点积。代码里用 @ 写。">Matmul</span>把两个二维 array 合成一个新的 array。结果里的每一个数字，都是一次 <span class="term" data-tip="把两串一样长的数字一个对一个乘起来，再把所有乘积加到一起。出来的是一个数字。">dot product（点积）</span>。你从第一个 array 里拿一<strong>行</strong>，从第二个 array 里拿一<strong>列</strong>。把它们一个对一个乘起来，再全部加到一起。代码里用 <code>@</code> 这个符号，也可以写 <code>np.matmul</code>。</p>
      </div>

      <h4><span class="lang-en">The shape rule (the whole lesson in one line)</span><span class="lang-zh">shape 规则（一句话就是今天的全部内容）</span></h4>
      <div class="callout c-ok"><span class="ic">📐</span><div><span class="lang-en"><code>(m, k) @ (k, n) = (m, n)</code> — the <strong>inner</strong> dimensions (<code>k</code>) must match and vanish; the <strong>outer</strong> dimensions (<code>m</code>, <code>n</code>) form the result.</span><span class="lang-zh"><code>(m, k) @ (k, n) = (m, n)</code> —— <strong>里面</strong>挨着的那两个 dimension（都写作 <code>k</code>）必须一样。它们一碰上就消失了。<strong>外面</strong>那两个（<code>m</code> 和 <code>n</code>）拼成结果的 shape。</span></div></div>

      <div class="lang-en">
      <h4>How does it relate to what you know?</h4>
      <p>On Day 1 you did element-wise math with <code>*</code>. Matmul is different: <code>@</code> does rows-times-columns, not element-by-element. Why care? A neural-network layer <em>is</em> a matmul: <code>y = x @ W</code> (plus a bias). Learn the shape rule and you can trace data through an entire network. Keep going 👇</p>
      </div>
      <div class="lang-zh">
      <h4>它和你已经会的东西有什么关系？</h4>
      <p>第 1 天你用 <code>*</code> 做 element-wise 的运算。matmul 不一样：<code>@</code> 做的是「行乘列」，不是一个对一个。为什么要在意这件事？因为神经网络的一层<em>就是</em>一个 matmul：<code>y = x @ W</code>，再加上一个 bias（偏置）。记住 shape 规则，你就能一层一层追着数据走完整个网络。继续往下看 👇</p>
      </div>
      <div class="callout c-info"><span class="ic">🏭</span><div><span class="lang-en"><b>A real one:</b> one NVIDIA H100 chip — the workhorse behind most frontier training runs — can do roughly 1,000 trillion floating-point operations a second (FLOPs) in <code>bfloat16</code>. Almost all of that is spent on matmuls, running exactly the shape rule you're about to learn.</span><span class="lang-zh"><b>真实的一个例子：</b>一块 NVIDIA H100 芯片，是今天大多数前沿训练的主力。它用 <code>bfloat16</code> 的时候，一秒钟大约能做 1000 万亿次浮点运算，也就是 FLOPs（浮点运算次数）。这些运算几乎全部花在 matmul 上。它们跑的就是你马上要学的那条 shape 规则。</span></div></div>
      <div class="callout c-warn"><span class="ic">😕</span><div><span class="lang-en"><b>Why this trips people up:</b> the rule <code>(m,k)@(k,n)=(m,n)</code> is memorized, but matmul vs element-wise blur together. In plain terms: <code>@</code> and <code>*</code> look almost identical yet do completely different things, so people write the wrong one and get a legal-shaped array full of wrong numbers.</span><span class="lang-zh"><b>为什么这里容易卡住：</b>大家都背下了 <code>(m,k)@(k,n)=(m,n)</code> 这条规则，却分不清 <code>@</code> 和 <code>*</code>。说白了，这两个符号长得几乎一样，做的事情却完全不同。于是有人写错了符号。程序不报错，array 的 shape 看着也对，可里面装的全是错的数字。</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Got what a matmul is — let's go</span><span class="lang-zh">知道 matmul 是什么了 —— 走吧</span></button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h"><span class="lang-en">A mixing board</span><span class="lang-zh">一台混音台</span></span><span class="sec-tag"><span class="lang-en">Intuition</span><span class="lang-zh">直觉</span></span></div>
  <div class="sec-body">
      <div class="relate">
        <div class="card"><span class="big">🎛️</span><h5><span class="lang-en">Weights = a mixing board</span><span class="lang-zh">weight = 一台混音台</span></h5><p><span class="lang-en">Each output channel is a <strong>weighted blend</strong> of all input channels. The weight matrix says how much each input feeds each output. Matmul does every blend at once.</span><span class="lang-zh">每一路输出，都是所有输入按不同分量<strong>混在一起</strong>的结果。weight（权重）matrix（矩阵）说的是：每个输入往每个输出里放多少。matmul 一次就把所有的混音都做完。</span></p></div>
        <div class="card"><span class="big">🔀</span><h5><span class="lang-en">A size-changing machine</span><span class="lang-zh">一台改尺寸的机器</span></h5><p><span class="lang-en">Feed in a vector of size <code>k</code>, get out a vector of size <code>n</code>. The <code>k × n</code> weight table is the machine; the <code>m</code> is just "do this for <code>m</code> examples at once."</span><span class="lang-zh">放进去一个长度是 <code>k</code> 的 vector（向量），出来一个长度是 <code>n</code> 的 vector。那张 <code>k × n</code> 的 weight 表就是这台机器。<code>m</code> 只是说「一次给 <code>m</code> 个例子都做一遍」。</span></p></div>
      </div>
      <p><span class="lang-en"><strong>In one line:</strong> matmul turns an input of width <code>k</code> into an output of width <code>n</code> using a <code>k × n</code> table of weights, for all <code>m</code> rows at once.</span><span class="lang-zh"><strong>一句话：</strong>matmul 用一张 <code>k × n</code> 的 weight 表，把宽度 <code>k</code> 的输入变成宽度 <code>n</code> 的输出。它一次就把 <code>m</code> 行全部做完。</span></p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><span class="lang-en">Where it trips people: <code>*</code> is <strong>element-wise</strong> (needs matching shapes, from Day 3), while <code>@</code> is <strong>matmul</strong> (needs matching inner dims). They look similar and do totally different things — mixing them up is a classic bug.</span><span class="lang-zh">这里最容易出错：<code>*</code> 是 <strong>element-wise</strong>，它要求两边的 shape 对得上，这是第 3 天讲过的。<code>@</code> 是 <strong>matmul</strong>，它只要求里面那两个 dimension 一样。两个符号长得像，做的事完全不同。把它们搞混，是一个很老很常见的 bug。</span></div></div>
      <p><span class="lang-en"><b>Intuition:</b> picture the weight matrix as a "mixing board" whose inner <code>k</code> is the number of input channels — the shared inner <code>k</code> is the plug that must fit on both sides, then it disappears. The technical point: matmul turns a width-<code>k</code> input into a width-<code>n</code> output through a <code>k×n</code> table, for all <code>m</code> rows at once.</span><span class="lang-zh"><b>直觉：</b>把 weight matrix 想成一台「混音台」。中间那个 <code>k</code> 就是「输入声道数」。两边的插头都得是这个数。插上以后，它就不见了。技术上说：matmul 通过一张 <code>k×n</code> 的表，把宽度 <code>k</code> 的输入变成宽度 <code>n</code> 的输出，而且一次做完 <code>m</code> 行。</span></p>
      <button class="gotit" type="button"><span class="lang-en">Got the picture</span><span class="lang-zh">画面看懂了</span></button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h"><span class="lang-en">The rule exactly — and why chips are matmul machines</span><span class="lang-zh">规则的准确说法 —— 以及芯片为什么是 matmul 机器</span></span><span class="sec-tag"><span class="lang-en">Mechanism &amp; why</span><span class="lang-zh">原理和为什么</span></span></div>
  <div class="sec-body">
      <div class="lang-en">
      <h4>The rule, precisely</h4>
      <ul>
        <li>To compute <code>A @ B</code>: <code>A</code>'s number of columns must equal <code>B</code>'s number of rows (the inner dims).</li>
        <li>The result has shape <code>(A rows, B columns)</code>.</li>
        <li>Each <code>result[i, j]</code> = the dot product of row <code>i</code> of <code>A</code> with column <code>j</code> of <code>B</code>.</li>
        <li><code>@</code> is matmul; <code>*</code> is element-wise. Never confuse them.</li>
      </ul>
      </div>
      <div class="lang-zh">
      <h4>准确的规则</h4>
      <ul>
        <li>要算 <code>A @ B</code>：<code>A</code> 的列数必须等于 <code>B</code> 的行数。这两个数就是里面挨着的那两个 dimension。</li>
        <li>结果的 shape 是 <code>(A rows, B columns)</code>，也就是 A 的行数配 B 的列数。</li>
        <li>每一个 <code>result[i, j]</code>，都是 <code>A</code> 的第 <code>i</code> 行和 <code>B</code> 的第 <code>j</code> 列做一次 dot product。</li>
        <li><code>@</code> 是 matmul，<code>*</code> 是 element-wise。千万别搞混。</li>
      </ul>
      </div>
      <div class="callout c-info"><span class="ic">🪜</span><div><span class="lang-en"><b>Math Ladder — the matmul shape rule &amp; its cost</b>
      <br><b>1 · In words:</b> the two inner dimensions must match and cancel; the outer two survive as the result; each output entry is one row dotted with one column.
      <br><b>2 · The rule:</b> <code>(m, k) @ (k, n) → (m, n)</code>, costing <code>≈ 2·m·k·n</code> FLOPs. <code>m</code> = rows of A (examples); <code>k</code> = the shared inner width; <code>n</code> = columns of B (output width). The <code>2</code> = one multiply + one add per pair.
      <br><b>3 · Tiny numbers:</b> <code>(2, 3) @ (3, 4) → (2, 4)</code>. Inner <code>3 == 3</code> ✓. Output has <code>2×4 = 8</code> entries, each a 3-term dot product → <code>2·2·3·4 = 48</code> FLOPs.
      <br><b>4 · Sanity check:</b> the <code>k</code> axis appears on both sides and vanishes from the result; if the two inner numbers differ, it's illegal. The result element count is always <code>m·n</code> — never <code>k</code>.</span><span class="lang-zh"><b>数学阶梯 —— matmul 的 shape 规则，还有它的代价</b>
      <br><b>1 · 用话说：</b>里面那两个 dimension 必须一样，然后一起消掉。外面那两个留下来，成为结果的 shape。输出里的每一个数字，都是一行和一列做一次 dot product。
      <br><b>2 · 规则：</b><code>(m, k) @ (k, n) → (m, n)</code>，代价大约是 <code>≈ 2·m·k·n</code> 个 FLOPs。<code>m</code> = A 的行数，也就是例子的个数。<code>k</code> = 中间共用的宽度。<code>n</code> = B 的列数，也就是输出的宽度。那个 <code>2</code> 是因为每一对数字要乘一次、再加一次。
      <br><b>3 · 用小数字算一遍：</b><code>(2, 3) @ (3, 4) → (2, 4)</code>。里面 <code>3 == 3</code> ✓。输出有 <code>2×4 = 8</code> 个数字。每个数字都是 3 项加起来的 dot product → <code>2·2·3·4 = 48</code> 个 FLOPs。
      <br><b>4 · 自己检查一下：</b><code>k</code> 这个 axis（轴）在两边都出现，然后从结果里消失。如果里面那两个数字不一样，这个 matmul 就不合法。结果里的数字个数永远是 <code>m·n</code>，绝不会是 <code>k</code>。</span></div></div>

      <div class="lang-en">
      <h4>Why a frontier lab cares</h4>
      <p>A <span class="term" data-tip="A dense (fully-connected / linear) layer: it maps inputs to outputs by y = xW + b, a matmul plus a bias.">dense layer</span> is literally <code>y = x @ W + b</code>. Attention is matmuls too: scores <code>= Q @ Kᵀ</code>, then output <code>= scores @ V</code>. Matmul is where almost all the <span class="term" data-tip="Floating-point operations — a count of the multiply/add work a computation does. Roughly 2·m·k·n for an (m,k)@(k,n) matmul.">FLOPs</span> go — exactly why the H100 chip from the teaser above is built to do almost nothing else. That's also exactly why the compute formula <code>C ≈ 6ND</code> (Kaplan et al., 2020 — you'll meet it in the scaling-law modules) is really a matmul count — and why GPUs and TPUs are, at heart, giant matmul machines.</p>
      </div>
      <div class="lang-zh">
      <h4>前沿实验室为什么在意这件事</h4>
      <p>一个 <span class="term" data-tip="dense layer，也叫全连接层或者 linear layer。它用 y = xW + b 把输入变成输出，就是一个 matmul 加一个 bias。">dense layer（全连接层）</span> 就是 <code>y = x @ W + b</code>，一点不多。attention（注意力）也全是 matmul：先算 scores <code>= Q @ Kᵀ</code>，再算输出 <code>= scores @ V</code>。几乎所有的 <span class="term" data-tip="浮点运算次数。它数的是一次计算里做了多少次乘和加。一个 (m,k)@(k,n) 的 matmul 大约要 2·m·k·n 次。">FLOPs</span> 都花在 matmul 上。上面提过的那块 H100 芯片，就是为了做这件事造出来的，几乎不做别的。算力公式 <code>C ≈ 6ND</code>（Kaplan 等人 2020 年提出，你会在 scaling-law 那几个 module 里见到它）说到底也是在数 matmul。GPU 和 TPU 骨子里就是很大的 matmul 机器。</p>
      </div>
      <div class="callout c-info"><span class="ic">🔗</span><div><span class="lang-en"><b>Remember this chain:</b> a layer is a matmul → matmul dominates the FLOPs → chips are built to do matmuls fast → the whole scaling story becomes "how many matmuls can you afford." And the most common runtime crash in ML is a matmul with mismatched inner dims.</span><span class="lang-zh"><b>记住这条链：</b>一层就是一个 matmul → matmul 吃掉了绝大部分 FLOPs → 芯片被造出来就是为了快速做 matmul → 于是整个 scaling 的故事变成一句话：「你付得起多少个 matmul？」还有一件事：ML 里最常见的运行时崩溃，就是 matmul 里面那两个 dimension 不一样。</span></div></div>

      <div class="lang-en">
      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>A mismatched matmul crashes loudly, which is easy to fix. The dangerous matmul bug is the one where the shapes happen to line up but the math is wrong — no crash, no error, just quietly wrong numbers.</p>
      </div>
      <div class="lang-zh">
      <h4>资深工程师的眼光 —— 一个悄悄发生的错，一个取舍</h4>
      <p>shape 对不上的 matmul 会大声崩掉，这种反而好修。危险的是另一种：shape 刚好对得上，可算的东西是错的。程序不崩，也不报错。只是数字悄悄地错了。</p>
      </div>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><span class="lang-en"><b>Failure mode (silent): a legal matmul that computes the wrong thing (a bug that doesn't announce itself).</b> Two traps. First, writing <code>*</code> when you meant <code>@</code>: with a bias vector <code>(features,)</code> and an activation <code>(batch, features)</code>, <code>*</code> does not error — it broadcasts an element-wise multiply and returns the right-shaped array full of wrong numbers. Second, matmul on a <strong>square</strong> weight, or feeding the un-transposed operand: <code>A @ B</code> and a wrongly-oriented version can both have legal inner dims, so both run, but only one is the layer you meant. A senior engineer catches this in <b>code review</b> by checking that <code>@</code> (not <code>*</code>) is used for layers and by tracing one <code>result[i, j]</code> against "row <code>i</code> of A dotted with column <code>j</code> of B" — if that dot product isn't the quantity you wanted, the orientation is wrong even though the shape is fine.</span><span class="lang-zh"><b>出错的样子（悄悄的）：一个合法的 matmul，算出来的东西却是错的。这个 bug 自己不会喊出来。</b>有两个坑。第一个坑是：你想写 <code>@</code>，手却写成了 <code>*</code>。假设 bias 是 <code>(features,)</code>，activation 是 <code>(batch, features)</code>。这时 <code>*</code> 不会报错。它做的是 broadcasting 加 element-wise 的乘法。它还给你一个 shape 完全正确的 array，里面装满错的数字。第二个坑是：weight 是<strong>方的</strong>，或者你把没做 transpose（转置）的那一边喂了进去。<code>A @ B</code> 和一个方向拧了的版本，里面那两个 dimension 都可能合法。两个都跑得动，可只有一个是你想要的那一层。一个资深工程师在 <b>code review</b> 里怎么抓到它？先看每一层用的是 <code>@</code>，不是 <code>*</code>。再挑一个 <code>result[i, j]</code>，对着「A 的第 <code>i</code> 行和 B 的第 <code>j</code> 列做 dot product」念一遍。如果这个 dot product 不是你想要的那个量，那方向就是拧的，哪怕 shape 看起来很对。</span></div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><span class="lang-en"><b>Trade-off: batching more matmul work vs. memory and latency.</b> Because matmul is where almost all the FLOPs go, chips run it fastest when you hand them big matrices at once — larger batches and wider layers keep the hardware busy and cost less per number. What you pay: bigger matmuls need more memory for their inputs and outputs, and a single huge matmul finishes later than a small one, so interactive latency suffers. In a <b>design review</b> this is the batch-size decision — trade throughput (fill the matmul units) against memory budget and response time.</span><span class="lang-zh"><b>取舍：一次多做一点 matmul，还是省内存加快响应。</b>几乎所有的 FLOPs 都花在 matmul 上。所以你一次递给芯片很大的 matrix，它跑得最快。batch 更大、层更宽，硬件就一直有活干，平均下来每个数字更便宜。你付出的代价是：大 matmul 的输入和输出都要占更多内存。而且一个很大的 matmul，会比一个小的做完得更晚。坐在屏幕前等结果的人，就会觉得慢。在 <b>design review</b> 里，这就是 batch size 怎么定的问题 —— 一边是 throughput（吞吐量），要把 matmul 单元填满；另一边是内存预算和响应时间。</span></div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><span class="lang-en"><b>Say this in an interview:</b> "A matmul is <code>(m,k)@(k,n)→(m,n)</code>: the inner dims cancel, each output is a row·column dot product, and it costs about <code>2·m·k·n</code> FLOPs — which is why a dense layer <code>y=x@W</code> and attention <code>Q@Kᵀ</code> are essentially the entire FLOP budget of a model. The nuance is that a legal shape isn't a correct matmul: writing <code>*</code> instead of <code>@</code>, or a wrong transpose, can run silently, so I trace one output entry against 'row i dotted with column j'."</span><span class="lang-zh"><b>面试时可以这样说：</b>「matmul 就是 <code>(m,k)@(k,n)→(m,n)</code>。里面那两个 dimension 抵消掉，每个输出都是一次行·列的 dot product，代价大约 <code>2·m·k·n</code> 个 FLOPs。所以 dense layer 的 <code>y=x@W</code> 和 attention 的 <code>Q@Kᵀ</code>，基本上就是一个模型全部的 FLOP 预算。细一点说：shape 合法不等于 matmul 算对了。把 <code>@</code> 写成 <code>*</code>，或者 transpose 弄错方向，程序都能悄悄地跑完。所以我会挑一个输出的数字，对着『A 的第 i 行和 B 的第 j 列做 dot product』核一遍。」</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Got the rule</span><span class="lang-zh">规则记住了</span></button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h"><span class="lang-en">Leave today's evidence (strongly recommended)</span><span class="lang-zh">留下今天的证据（很建议做）</span></span><span class="sec-tag"><span class="lang-en">Produce</span><span class="lang-zh">动手做</span></span></div>
  <div class="sec-body">
      <p><span class="lang-en">Understanding ≠ being able to write it. Multiply a few matrices and build a layer yourself — that's what makes the day count. Pick one:</span><span class="lang-zh">看懂 ≠ 写得出来。自己动手乘几个 matrix，再亲手搭一层出来 —— 这一天才算真的过了。挑一个做：</span></p>
      <div class="lang-en">
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py</code>. Multiply a <code>(2,3)</code> by a <code>(3,2)</code> with <code>@</code> and print the result's shape; try a mismatched <code>(2,3) @ (2,3)</code> in a <code>try/except</code> and print the error; build a dense layer <code>x @ W + b</code> with <code>x</code> shape <code>(1,3)</code> and <code>W</code> shape <code>(3,4)</code>; and show that <code>@</code> and <code>*</code> give different results. Run it with <code>python3 sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
      </div>
      <div class="lang-zh">
      <h4>做法 A · 自己写</h4>
      <p>新建 <code>sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py</code>。用 <code>@</code> 把一个 <code>(2,3)</code> 和一个 <code>(3,2)</code> 乘起来，再打印结果的 shape。然后用 <code>try/except</code> 试一个对不上的 <code>(2,3) @ (2,3)</code>，把报错打印出来。接着搭一个 dense layer <code>x @ W + b</code>，<code>x</code> 的 shape 是 <code>(1,3)</code>，<code>W</code> 的 shape 是 <code>(3,4)</code>。最后让 <code>@</code> 和 <code>*</code> 给出不一样的结果。用 <code>python3 sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py</code> 跑起来。</p>
      <h4>做法 B · 让 Claude 写，然后你读</h4>
      <p>把下面这段 prompt 复制回 Claude Code。它会让 Claude Code 帮你新建文件、写好代码，再跑一次。</p>
      </div>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l"><span class="lang-en">ask Claude to build it</span><span class="lang-zh">让 Claude 来写</span></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Help me build my Module 1 Day 4 artifact.

Create sessions/m01-shape-of-data/day-04-matmul-and-shapes/experiment.py that, with a comment on each step:
1. Makes A (2,3) and B (3,2); prints (A @ B).shape (expect (2,2)) and the result, noting the inner 3 matched and vanished.
2. Wraps A @ np.ones((2,3)) in try/except and prints the shape-mismatch error (inner 3 vs 2).
3. Builds a dense layer: x (1,3), W (3,4), b (4,); prints (x @ W + b) and its shape (1,4), with a comment 'this is a neural-net layer'.
4. Shows @ vs *: print A @ B (matmul) and A * A (element-wise) to make the difference obvious.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4><span class="lang-en">Acceptance criteria</span><span class="lang-zh">做到这几条就算过了</span></h4>
      <ul>
        <li><span class="lang-en">You multiply <code>(2,3) @ (3,2)</code> with <code>@</code> and print the <code>(2,2)</code> result; you catch the error from a mismatched <code>(2,3) @ (2,3)</code>.</span><span class="lang-zh">你用 <code>@</code> 算了 <code>(2,3) @ (3,2)</code>，并打印出 <code>(2,2)</code> 的结果。你也接住了 <code>(2,3) @ (2,3)</code> 对不上时的报错。</span></li>
        <li><span class="lang-en">You build a dense layer <code>x @ W + b</code> and print its output shape, and you show <code>@</code> and <code>*</code> give different results.</span><span class="lang-zh">你搭了一个 dense layer <code>x @ W + b</code>，并打印了它输出的 shape。你还让 <code>@</code> 和 <code>*</code> 给出了不一样的结果。</span></li>
        <li><span class="lang-en">You can state the FLOP cost <code>2·m·k·n</code> for one of your matmuls before running it.</span><span class="lang-zh">在跑之前，你就能说出自己某一个 matmul 的 FLOP 代价 <code>2·m·k·n</code>。</span></li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><span class="lang-en"><b>5-minute research log:</b> before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-04-matmul-and-shapes/log.md</code>: (1) in your own words, why must the inner dimensions match; (2) the FLOP count that surprised you; (3) one thing you're still unsure about (<code>@</code> vs <code>*</code> is fair game).</span><span class="lang-zh"><b>5 分钟研究笔记：</b>关掉页面之前，用自己的话在 <code>sessions/m01-shape-of-data/day-04-matmul-and-shapes/log.md</code> 里写三行：（1）里面那两个 dimension 为什么必须一样；（2）哪一个 FLOP 数字让你意外；（3）还有一件你不太确定的事（<code>@</code> 和 <code>*</code> 的区别完全可以写）。</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Done</span><span class="lang-zh">做完了</span></button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3><span class="lang-en">Module 1 · Day 4 complete! 🏆</span><span class="lang-zh">Module 1 · 第 4 天完成！🏆</span></h3>
      <p><span class="lang-en">Nice work — you've completed <b>Matmul &amp; Shapes</b>.<br>Next up: <b>Logs &amp; Exponents</b>.</span><span class="lang-zh">干得好 —— 你学完了 <b>matmul 与 shape</b>。<br>下一站：<b>log 与 exponent</b>。</span></p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  mm:{html:'<span class="prompt">&gt;&gt;&gt;</span> A = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])   <span class="dim"># (2,3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> B = np.array([[<span class="num">1</span>,<span class="num">0</span>], [<span class="num">0</span>,<span class="num">1</span>], [<span class="num">1</span>,<span class="num">0</span>]])   <span class="dim"># (3,2)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> (A @ B).shape      <span class="dim"># inner 3 matches → (2,2)</span>\n'+
    '<span class="hl">(2, 2)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> A @ B\n<span class="ok">array([[ 4,  2],\n       [10,  5]])</span>',
    take:'<span class="lang-en"><b>①  The inner dim matches and vanishes.</b> (2,<b>3</b>) @ (<b>3</b>,2) → the 3\'s touch and disappear; the outer 2 and 2 form the result (2,2). Each entry is a row·column dot product.</span><span class="lang-zh"><b>①  里面那个 dimension 对上了，然后消失。</b>(2,<b>3</b>) @ (<b>3</b>,2) → 两个 3 一碰上就不见了。外面的 2 和 2 拼成结果 (2,2)。里面每一个数字，都是一次行·列的 dot product。</span>'},
  bad:{html:'<span class="prompt">&gt;&gt;&gt;</span> A = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])   <span class="dim"># (2,3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> C = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])   <span class="dim"># (2,3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> A @ C\n'+
    '<span class="bad">ValueError: matmul: Input operand 1 has a\n'+
    'mismatch in its core dimension ... (size 3 is\n'+
    'different from 2)</span>',
    take:'<span class="lang-en"><b>②  Inner dims must match.</b> (2,<b>3</b>) @ (<b>2</b>,3): the touching numbers are 3 and 2 — not equal — so matmul refuses. This exact crash is the #1 shape bug in ML code.</span><span class="lang-zh"><b>②  里面那两个 dimension 必须一样。</b>(2,<b>3</b>) @ (<b>2</b>,3)：挨在一起的两个数字是 3 和 2，不相等，所以 matmul 拒绝动手。这个报错就是 ML 代码里最常见的 shape bug。</span>'},
  layer:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([[<span class="num">1.</span>, <span class="num">2.</span>, <span class="num">3.</span>]])        <span class="dim"># (1,3) one input, 3 features</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> W = np.random.rand(<span class="num">3</span>, <span class="num">4</span>)         <span class="dim"># (3,4) weights</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> b = np.zeros(<span class="num">4</span>)                 <span class="dim"># (4,) bias</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> (x @ W + b).shape       <span class="dim"># (1,3)@(3,4)=(1,4), bias broadcast</span>\n'+
    '<span class="hl">(1, 4)</span>',
    take:'<span class="lang-en"><b>③  This IS a neural-net layer.</b> y = x @ W + b turned 3 inputs into 4 outputs via a 3×4 weight table, then broadcast the bias (Day 3!) across the row. Every dense layer in every model is this line.</span><span class="lang-zh"><b>③  这就是神经网络的一层。</b>y = x @ W + b 用一张 3×4 的 weight 表，把 3 个输入变成 4 个输出。然后它把 bias 沿着这一行 broadcast（广播）开，这是第 3 天讲过的。每个模型里的每一个 dense layer，都是这一行。</span>'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:`<svg viewBox='0 0 520 130' role='img' aria-label='Dot product of two length-3 vectors equals 32'><g font-family='monospace' font-size='14' text-anchor='middle'><text x='90' y='40' fill='#1F6280'>[1, 2, 3]</text><text x='90' y='66' fill='#9A7208'>[4, 5, 6]</text><text x='210' y='53' font-size='16' fill='#6B645E'>→</text><text x='330' y='40' fill='#2C2A28'>1·4 + 2·5 + 3·6</text><text x='330' y='66' fill='#2C2A28'>= 4 + 10 + 18</text><text x='430' y='53' font-size='16' fill='#6B645E'>=</text><rect x='455' y='36' width='44' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='477' y='59' fill='#1a5c38' font-weight='bold'>32</text><text class="lang-en" x='260' y='108' fill='#6B645E' font-size='12'>dot product: multiply pairwise, then add → ONE number</text><text class="lang-zh" x='260' y='108' fill='#6B645E' font-size='12'>dot product：一个对一个乘，再全加起来 → 一个数字</text></g></svg>`,
  note:`<span class="lang-en"><b>Start with a dot product.</b> Multiply two equal-length lists element-by-element, then sum: <code>[1,2,3]·[4,5,6] = 32</code>. One number out. A matmul is just many of these done at once.</span><span class="lang-zh"><b>先从一次 dot product 开始。</b>把两串一样长的数字一个对一个乘起来，再加到一起：<code>[1,2,3]·[4,5,6] = 32</code>。出来一个数字。一个 matmul 就是很多次这样的运算，一次全做完。</span>`},
 {viz:`<svg viewBox='0 0 520 140' role='img' aria-label='A row dotted against each column of a matrix produces a row of outputs'><g font-family='monospace' font-size='13' text-anchor='middle'><text class="lang-en" x='70' y='70' fill='#1F6280'>row</text><text class="lang-zh" x='70' y='70' fill='#1F6280'>一行</text><text x='70' y='88' fill='#1F6280'>[1 2 3]</text><text x='160' y='79' fill='#6B645E'>·</text><rect x='195' y='40' width='36' height='80' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><rect x='235' y='40' width='36' height='80' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='213' y='75' fill='#5E5191'>c</text><text x='213' y='95' fill='#5E5191'>o</text><text x='253' y='75' fill='#5E5191'>c</text><text x='253' y='95' fill='#5E5191'>o</text><text class="lang-en" x='233' y='134' fill='#6B645E' font-size='11'>each column</text><text class="lang-zh" x='233' y='134' fill='#6B645E' font-size='11'>每一列</text><text x='320' y='79' fill='#6B645E'>→</text><rect x='360' y='63' width='40' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><rect x='402' y='63' width='40' height='34' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='380' y='86' fill='#1a5c38'>o1</text><text x='422' y='86' fill='#1a5c38'>o2</text><text class="lang-en" x='401' y='118' fill='#6B645E' font-size='11'>one output per column</text><text class="lang-zh" x='401' y='118' fill='#6B645E' font-size='11'>每列给一个输出</text></g></svg>`,
  note:`<span class="lang-en"><b>One row against many columns.</b> Dot the row with column 1 → first output; with column 2 → second output. A row times a matrix gives a whole row of results — one number per column.</span><span class="lang-zh"><b>一行去碰很多列。</b>这一行和第 1 列做 dot product → 第一个输出。和第 2 列做 → 第二个输出。一行乘一个 matrix，会给你一整行结果。每一列对应一个数字。</span>`},
 {viz:`<svg viewBox='0 0 520 130' role='img' aria-label='Shape rule (2,3)@(3,4)=(2,4), the inner 3s match and vanish'><g font-family='monospace' font-size='16' text-anchor='middle'><text x='90' y='60' fill='#1F6280'>( 2 ,</text><text x='140' y='60' fill='#C99A12' font-weight='bold'>3</text><text x='158' y='60' fill='#1F6280'>)</text><text x='210' y='60' fill='#6B645E'>@</text><text x='262' y='60' fill='#1F6280'>(</text><text x='282' y='60' fill='#C99A12' font-weight='bold'>3</text><text x='320' y='60' fill='#1F6280'>, 4 )</text><text x='400' y='60' fill='#6B645E'>=</text><rect x='430' y='42' width='72' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='466' y='65' fill='#1a5c38' font-weight='bold'>(2, 4)</text><line x1='140' y1='68' x2='282' y2='68' stroke='#C99A12' stroke-width='1.5' stroke-dasharray='4,3'/><text class="lang-en" x='211' y='90' fill='#9A7208' font-size='11'>inner must match → vanishes</text><text class="lang-zh" x='211' y='90' fill='#9A7208' font-size='11'>里面必须一样 → 然后消失</text><text class="lang-en" x='260' y='116' fill='#2D8B55' font-size='12' font-weight='bold'>outer dims (2 and 4) survive as the result</text><text class="lang-zh" x='260' y='116' fill='#2D8B55' font-size='12' font-weight='bold'>外面的 2 和 4 留下来，成为结果</text></g></svg>`,
  note:`<span class="lang-en"><b>The shape rule.</b> <code>(2,3) @ (3,4)</code>: the inner 3's match, cancel out, and the outer 2 and 4 become the result <code>(2,4)</code>. This one rule is all you need to trace shapes through a network.</span><span class="lang-zh"><b>shape 规则。</b><code>(2,3) @ (3,4)</code>：里面的两个 3 对上了，一起消掉。外面的 2 和 4 变成结果 <code>(2,4)</code>。只要这一条规则，你就能一路追着 shape 走完整个网络。</span>`},
 {viz:`<svg viewBox='0 0 520 120' role='img' aria-label='Mismatch (2,3)@(2,3) inner 3 vs 2 raises an error'><g font-family='monospace' font-size='16' text-anchor='middle'><text x='110' y='55' fill='#1F6280'>( 2 ,</text><text x='160' y='55' fill='#C93B3B' font-weight='bold'>3</text><text x='178' y='55' fill='#1F6280'>)</text><text x='220' y='55' fill='#6B645E'>@</text><text x='262' y='55' fill='#1F6280'>(</text><text x='282' y='55' fill='#C93B3B' font-weight='bold'>2</text><text x='320' y='55' fill='#1F6280'>, 3 )</text><text x='215' y='80' fill='#C93B3B' font-size='12'>3 ≠ 2</text><rect x='120' y='92' width='280' height='20' rx='5' fill='#FDE8E8' stroke='#C93B3B' stroke-width='1.5'/><text class="lang-en" x='260' y='106' font-size='10.5' fill='#C93B3B'>ValueError: matmul core dimension mismatch</text><text class="lang-zh" x='260' y='106' font-size='10.5' fill='#C93B3B'>ValueError：里面的 dimension 不一样</text></g></svg>`,
  note:`<span class="lang-en"><b>When it crashes.</b> <code>(2,3) @ (2,3)</code>: the inner numbers are 3 and 2 — not equal — so matmul errors before doing anything. Reading the two inner numbers first is how you catch this instantly.</span><span class="lang-zh"><b>什么时候会崩。</b><code>(2,3) @ (2,3)</code>：里面的两个数字是 3 和 2，不相等。所以 matmul 什么都不做就报错了。先去看里面那两个数字，你就能马上抓到它。</span>`},
 {viz:`<svg viewBox='0 0 520 130' role='img' aria-label='A dense layer x(1,3) @ W(3,4) + b = (1,4)'><g font-family='monospace' font-size='14' text-anchor='middle'><rect x='40' y='46' width='70' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='75' y='66' fill='#1F6280'>x (1,3)</text><text x='125' y='66' fill='#6B645E'>@</text><rect x='145' y='40' width='80' height='42' rx='5' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='185' y='65' fill='#5E5191'>W (3,4)</text><text x='240' y='66' fill='#6B645E'>+</text><rect x='258' y='48' width='60' height='28' rx='5' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='288' y='67' fill='#9A7208'>b (4,)</text><text x='335' y='66' fill='#6B645E'>=</text><rect x='358' y='46' width='72' height='30' rx='5' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='394' y='66' fill='#1a5c38' font-weight='bold'>y (1,4)</text><text class="lang-en" x='260' y='108' fill='#2D8B55' font-size='12' font-weight='bold'>y = x @ W + b  —  this is one neural-network layer</text><text class="lang-zh" x='260' y='108' fill='#2D8B55' font-size='12' font-weight='bold'>y = x @ W + b  ——  这就是神经网络的一层</text></g></svg>`,
  note:`<span class="lang-en"><b>The layer hiding inside.</b> <code>y = x @ W + b</code>: a matmul turns 3 inputs into 4 outputs, then the bias broadcasts across (Day 3). Stack layers like this and you have a neural network. It's matmuls all the way down.</span><span class="lang-zh"><b>藏在里面的那一层。</b><code>y = x @ W + b</code>：一个 matmul 把 3 个输入变成 4 个输出。然后 bias 被 broadcast 开，这是第 3 天讲过的。把这样的层一层一层叠起来，就是一个神经网络。往下看，全都是 matmul。</span>`},
 {viz:`<svg viewBox='0 0 520 140' role='img' aria-label='Attention is matmuls and matmul dominates the FLOPs on matmul-machine chips'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='40' y='34' width='130' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text class="lang-en" x='105' y='54' fill='#1F6280'>scores = Q @ Kᵀ</text><text class="lang-zh" x='105' y='54' fill='#1F6280'>scores = Q @ Kᵀ</text><rect x='40' y='72' width='130' height='30' rx='5' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text class="lang-en" x='105' y='92' fill='#1F6280'>out = scores @ V</text><text class="lang-zh" x='105' y='92' fill='#1F6280'>out = scores @ V</text><text x='195' y='72' fill='#6B645E'>→</text><rect x='230' y='40' width='250' height='58' rx='8' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text class="lang-en" x='355' y='63' fill='#9A7208' font-weight='bold'>matmul ≈ most of the FLOPs</text><text class="lang-zh" x='355' y='63' fill='#9A7208' font-weight='bold'>matmul ≈ 绝大部分 FLOPs</text><text class="lang-en" x='355' y='84' fill='#9A7208'>(so C ≈ 6ND is a matmul count)</text><text class="lang-zh" x='355' y='84' fill='#9A7208'>（所以 C ≈ 6ND 是在数 matmul）</text><text class="lang-en" x='260' y='126' fill='#2C2A28'>GPUs and TPUs are, at heart, giant matmul machines</text><text class="lang-zh" x='260' y='126' fill='#2C2A28'>GPU 和 TPU 骨子里就是很大的 matmul 机器</text></g></svg>`,
  note:`<span class="lang-en"><b>Why it matters at frontier scale.</b> Attention is matmuls (<code>Q @ Kᵀ</code>, then <code>@ V</code>), and matmul is where nearly all the compute goes — that's why the scaling-law formula <code>C ≈ 6ND</code> is really counting matmul work, and why accelerators are designed as matmul engines. Everything you optimize later is about doing matmuls faster or cheaper.</span><span class="lang-zh"><b>放到前沿规模上，它为什么重要。</b>attention 全是 matmul：先 <code>Q @ Kᵀ</code>，再 <code>@ V</code>。几乎所有算力都花在 matmul 上。所以 scaling law 的公式 <code>C ≈ 6ND</code> 数的其实是 matmul 的工作量。加速器也就被设计成了 matmul 引擎。以后你做的每一次优化，都是让 matmul 更快或者更便宜。</span>`}
];
@@@ region name=QS
var QS=[
 {q:'<span class="lang-en">1. What is the output shape of <code>(2, 3) @ (3, 4)</code>?</span><span class="lang-zh">1. <code>(2, 3) @ (3, 4)</code> 输出的 shape 是什么？</span>',
  opts:['(3, 3)','(2, 4)','(2, 3)','<span class="lang-en">error</span><span class="lang-zh">报错</span>'],
  ans:1, fb:'<span class="lang-en">Right. Inner 3\'s match and vanish; outer 2 and 4 form the result → (2, 4).</span><span class="lang-zh">对了。里面的两个 3 对上以后一起消失。外面的 2 和 4 拼成结果 → (2, 4)。</span>'},
 {q:'<span class="lang-en">2. For <code>A @ B</code> to work, what must match?</span><span class="lang-zh">2. 要让 <code>A @ B</code> 算得动，什么必须一样？</span>',
  opts:['<span class="lang-en">A\'s rows and B\'s rows</span><span class="lang-zh">A 的行数和 B 的行数</span>','<span class="lang-en">A\'s columns and B\'s rows (the inner dims)</span><span class="lang-zh">A 的列数和 B 的行数，也就是里面那两个 dimension</span>','<span class="lang-en">The dtypes</span><span class="lang-zh">两边的 dtype（数据类型）</span>','<span class="lang-en">Nothing — matmul always works</span><span class="lang-zh">什么都不用 —— matmul 永远算得动</span>'],
  ans:1, fb:'<span class="lang-en">Right. A\'s column count must equal B\'s row count — the two inner dimensions.</span><span class="lang-zh">对了。A 的列数必须等于 B 的行数。这两个数就是里面那两个 dimension。</span>'},
 {q:'<span class="lang-en">3. A dense (linear) layer computes:</span><span class="lang-zh">3. 一个 dense（linear）layer 算的是：</span>',
  opts:['<span class="lang-en">y = x * W (element-wise)</span><span class="lang-zh">y = x * W，一个对一个乘</span>','<span class="lang-en">y = x @ W + b (a matmul plus a bias)</span><span class="lang-zh">y = x @ W + b，一个 matmul 加一个 bias</span>','y = x + W','<span class="lang-en">y = W only</span><span class="lang-zh">只有 y = W</span>'],
  ans:1, fb:'<span class="lang-en">Right. y = x @ W + b — a matmul that maps inputs to outputs, plus a broadcast bias.</span><span class="lang-zh">对了。y = x @ W + b —— 一个 matmul 把输入变成输出，再加上一个 broadcast 开的 bias。</span>'},
 {q:'<span class="lang-en">4. A teammate\'s layer runs with no error and the output has the shape they expected, but the model never learns — loss barely moves. You look at the layer and see <code>y = x * W + b</code> where <code>x</code> is <code>(batch, features)</code> and <code>W</code> is also <code>(batch, features)</code>. Where do you look first, and why?</span><span class="lang-zh">4. 同事写的一层跑起来不报错，输出的 shape 也和他想的一样。可模型一直学不会，loss（损失）几乎不动。你去看那一层，看到的是 <code>y = x * W + b</code>。这里 <code>x</code> 是 <code>(batch, features)</code>，<code>W</code> 也是 <code>(batch, features)</code>。你先看哪里，为什么？</span>',
  opts:['<span class="lang-en">The learning rate — a stuck loss is always an optimizer problem</span><span class="lang-zh">learning rate —— loss 不动，一定是 optimizer（优化器）的问题</span>','<span class="lang-en">The <code>*</code>: it does an element-wise multiply (broadcasting), not the matmul a layer needs, so the shape looks right but every output is the wrong quantity</span><span class="lang-zh">那个 <code>*</code>：它做的是 element-wise 乘法，还会 broadcast，不是一层需要的 matmul。所以 shape 看着对，可每一个输出都是错的量</span>','<span class="lang-en">The dtype — element-wise vs matmul is only a precision issue</span><span class="lang-zh">dtype —— element-wise 和 matmul 的差别只是精度问题</span>','<span class="lang-en">Nothing is wrong; <code>*</code> and <code>@</code> are interchangeable when shapes match</span><span class="lang-zh">没有问题。shape 对得上的时候，<code>*</code> 和 <code>@</code> 可以互换</span>'],
  ans:1, fb:'<span class="lang-en">Right. <code>*</code> multiplies element by element and broadcasts, so <code>x * W</code> can produce a plausibly-shaped array without ever mixing features the way <code>x @ W</code> does. A real dense layer is <code>y = x @ W + b</code>. The shape check passes, so the bug is silent — you catch it by confirming the operator is <code>@</code>, not <code>*</code>, before blaming the optimizer.</span><span class="lang-zh">对了。<code>*</code> 是一个对一个乘，还会 broadcast。所以 <code>x * W</code> 能做出一个 shape 看着合理的 array，可它从来没有像 <code>x @ W</code> 那样把 features 混到一起。真正的 dense layer 是 <code>y = x @ W + b</code>。shape 检查能过，所以这个 bug 藏得很好。抓它的办法是：先确认运算符是 <code>@</code> 而不是 <code>*</code>，再去怀疑 optimizer。</span>'}
];
