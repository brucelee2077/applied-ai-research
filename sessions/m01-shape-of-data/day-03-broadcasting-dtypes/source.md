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
<div class="brand-sub"><span class="lang-en">Foundations · M1 Day 3</span><span class="lang-zh">基础 · M1 第 3 天</span></div>
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
<a class="lnav prev" href="../day-02-indexing-slicing/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t"><span class="lang-en">Indexing &amp; Slicing</span><span class="lang-zh">index 与 slice</span></span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-04-matmul-and-shapes/lesson.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t"><span class="lang-en">Matmul &amp; Shapes</span><span class="lang-zh">matmul 与 shape</span></span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker"><span class="lang-en">Module 1 · Represent · Day 3</span><span class="lang-zh">Module 1 · 表示 · 第 3 天</span></span>
      <h1><span class="lang-en">Broadcasting &amp; dtypes</span><span class="lang-zh">broadcasting 与 dtype</span><span class="sub"><span class="lang-en">Math between different shapes</span><span class="lang-zh">shape 不一样，怎么做运算</span></span></h1>
      <p class="lede"><span class="lang-en">You can build arrays, slice them, and read their shape. Now the interesting part: what happens when you try to add two arrays that are <strong>not</strong> the same shape? Most languages make you write a loop. NumPy just does it — a trick called <strong>broadcasting</strong>. Pair it with one more idea, <strong>dtype</strong> (how big each number is), and you have the two levers that quietly set how much memory every model on Earth needs.</span><span class="lang-zh">你已经会做 array（数组），会切它，也会读它的 shape（形状）。现在来点有意思的。要是你把两个 shape <strong>不一样</strong>的 array 加起来，会发生什么？大多数语言都要你自己写一个循环。NumPy 直接就把它算出来了。这个本事叫 <strong>broadcasting（广播）</strong>。再配上另一个想法：<strong>dtype（数据类型）</strong>，也就是每个数字占多大。这两个开关合在一起，悄悄决定了地球上每一个 model 要吃掉多少内存。</span></p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><span class="lang-en"><b>By the end, you'll be able to:</b> predict whether two shapes can be added, state the one broadcasting rule, and explain why <code>float32</code> uses half the memory of <code>float64</code>.</span><span class="lang-zh"><b>学完这一天，你能做到：</b>看着两个 shape，就说出它们能不能加起来。能背出 broadcasting 那一条规则。还能讲明白 <code>float32</code> 为什么只花 <code>float64</code> 一半的内存。</span></div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
      <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h"><span class="lang-en">Two ideas: broadcasting and dtype</span><span class="lang-zh">两个想法：broadcasting 和 dtype</span></span><span class="sec-tag"><span class="lang-en">What is it</span><span class="lang-zh">这是什么</span></span></div>
      <div class="sec-body">
        <div class="callout c-info"><span class="ic">🧭</span><div><span class="lang-en"><b>What this assumes:</b> Days 1–2 — arrays, shape, and that <code>x + 10</code> adds to every element. <b>No new tools</b> — two ideas that ride on what you have.</span><span class="lang-zh"><b>你需要先会：</b>第 1 到第 2 天的内容 —— array、shape，还有 <code>x + 10</code> 会加到每一个元素上。<b>不需要新工具。</b>今天只有两个想法，它们都长在你已经会的东西上面。</span></div></div>

        <div class="lang-en">
        <h4>What is broadcasting?</h4>
        <p><span class="term" data-tip="NumPy's rule for doing math between arrays of different shapes: a side that is a single value gets copied ('stamped') across to match the other — without really duplicating the data.">Broadcasting</span> is NumPy's rule for doing math between arrays of <strong>different shapes</strong>. The idea in one line: when one side has a <strong>single value</strong> where the other has many, NumPy copies that one value across to fill the gap — so you never write a loop. You already used the simplest case on Day 1: <code>x + 10</code> copied the single number 10 across every element. In a moment we'll picture this copying as a <strong>stamp</strong> — and that same stamp is all you need to understand the full rule later.</p>
        <h4>What is a dtype?</h4>
        <p>A <span class="term" data-tip="Data type: the single kind and size every element of an array shares, e.g. int64, float32, float64.">dtype</span> (data type) is the <strong>one kind and size</strong> every element in an array shares — like <code>int64</code>, <code>float32</code>, or <code>float64</code>. It decides two things: what values fit (whole numbers vs decimals) and <strong>how many bytes each number costs</strong>.</p>
        <h4>Why put them together today?</h4>
        <p>Both are about <strong>doing math efficiently at scale</strong>. Broadcasting lets one small array (like a bias) act on a huge one with no copies. The dtype decides the memory bill — and that bill is what forces the whole <span class="term" data-tip="Storing numbers in fewer bits (a smaller dtype box) to shrink a model's memory, at the cost of some precision.">quantization</span> story you'll meet later. Keep going 👇</p>
        </div>
        <div class="lang-zh">
        <h4>broadcasting 是什么？</h4>
        <p><span class="term" data-tip="NumPy 的一条规则，让 shape 不一样的 array 也能一起做运算。哪一边只有一个数字，NumPy 就把这个数字复制过去，凑齐另一边的个数。数据并没有真的被复制。">Broadcasting</span> 是 NumPy 的一条规则。它管的是 <strong>shape 不一样</strong>的 array 之间怎么做运算。一句话说清楚：一边只有<strong>一个数字</strong>，另一边有很多个，NumPy 就把这一个数字复制过去，把空位填满。你不用写循环。第 1 天你已经用过最简单的那一种：<code>x + 10</code> 把 10 这一个数字复制到了每一个元素上。等一下我们会把这种复制想成<strong>盖章</strong>。后面那条完整的规则，用的就是同一个印章。</p>
        <h4>dtype 是什么？</h4>
        <p><span class="term" data-tip="数据类型。一个 array 里每个元素共用的那一种类型和大小，比如 int64、float32、float64。">dtype</span>是一个 array 里每个元素<strong>共用的那一种类型和大小</strong>，比如 <code>int64</code>、<code>float32</code> 或者 <code>float64</code>。它决定两件事。第一件：什么样的值放得进去，整数还是小数。第二件：<strong>每个数字要花掉几个字节</strong>。</p>
        <h4>今天为什么把它们放在一起讲？</h4>
        <p>它们说的是同一件事：<strong>在很大的规模上省着算</strong>。broadcasting 让一个很小的 array，比如一个 bias（偏置），去作用在一个很大的 array 上，一份复制都不用做。dtype 决定内存的账单。就是这张账单，逼出了你后面会遇到的整个 <span class="term" data-tip="用更少的位数存数字，也就是换一个更小的 dtype 箱子，好让 model 少占内存。代价是丢掉一些精度。">quantization（量化）</span> 故事。接着往下看 👇</p>
        </div>
        <div class="callout c-warn"><span class="ic">😕</span><div><span class="lang-en"><b>Why this trips people up:</b> the rule is short, but its silent successes are the trap. In plain terms: people expect a shape mismatch to crash, but NumPy often "helpfully" stretches a size-1 axis and returns a bigger, wrong array — so the bug ships instead of throwing.</span><span class="lang-zh"><b>为什么这里容易卡住：</b>broadcasting 的规则看着很简单，可它<em>不报错的时候</em>最危险。说白了：大家以为 shape 对不上就会崩掉。但 NumPy 常常「很好心」地把一条 size-1 的 axis（轴）拉开，然后还给你一个更大的、错的 array。于是这个 bug 跟着代码上线了，它一声都没喊。</span></div></div>
        <button class="gotit" type="button"><span class="lang-en">Got the two ideas — let's go</span><span class="lang-zh">两个想法都懂了 —— 走吧</span></button>
      </div>
    </section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
      <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h"><span class="lang-en">A stamp, and a set of boxes</span><span class="lang-zh">一个印章，和一排箱子</span></span><span class="sec-tag"><span class="lang-en">Intuition</span><span class="lang-zh">直觉</span></span></div>
      <div class="sec-body">
        <div class="relate">
          <div class="card"><h5><span class="big" aria-hidden="true">🖊️</span> <span class="lang-en">Broadcasting = one stamp</span><span class="lang-zh">broadcasting = 一个印章</span></h5><p><span class="lang-en">To add 10 to every cell of a grid, you don't write "10" in each cell by hand. You press <strong>one stamp</strong> — the single 10 — across the whole grid at once. That is broadcasting: a small thing pressed out to cover a big thing.</span><span class="lang-zh">要给一张格子里的每一格都加上 10，你不会一格一格手写「10」。你只按<strong>一个印章</strong>，就是那一个 10，一下盖满整张格子。这就是 broadcasting：一个小东西被压开，盖住一个大东西。</span></p></div>
          <div class="card"><h5><span class="big" aria-hidden="true">📦</span> <span class="lang-en">dtype = the size of the box</span><span class="lang-zh">dtype = 箱子的大小</span></h5><p><span class="lang-en">Every number lives in a box, and you pick the box size. <code>float64</code> is a big box (8 bytes, lots of precision); <code>float32</code> is half the size (4 bytes); <code>bfloat16</code> is tiny (2 bytes). Smaller box → less memory.</span><span class="lang-zh">每一个数字都住在一个箱子里，箱子多大由你挑。<code>float64</code> 是大箱子，8 个字节，精度很足。<code>float32</code> 只有它一半大，4 个字节。<code>bfloat16</code> 很小，2 个字节。箱子更小 → 内存更少。</span></p></div>
        </div>
        <p><span class="lang-en"><strong>In one line:</strong> broadcasting is a stamp that presses a small shape across a big one; dtype is how big each number's box is.</span><span class="lang-zh"><strong>一句话：</strong>broadcasting 是一个印章，把小的 shape 压开、盖到大的 shape 上。dtype 是每个数字的箱子有多大。</span></p>
        <div class="callout c-warn"><span class="ic">⚠️</span><div><span class="lang-en"><b>Where the analogy breaks:</b> a real stamp uses ink on every press, but broadcasting never truly copies the data — the "press" is virtual, so it costs no extra memory. And a smaller box is not free: squeeze the number too hard and you lose precision. That tension — smaller vs. accurate — is the whole story of <span class="term" data-tip="Storing numbers in fewer bits (a smaller dtype box) to shrink a model's memory, at the cost of some precision.">quantization</span> you'll meet later.</span><span class="lang-zh"><b>这个比喻在哪里不成立：</b>真的印章每盖一次都要用墨水。broadcasting 从来没有真的复制数据。那一次「盖」是虚的，所以它不多花内存。还有一件事：更小的箱子不是白拿的。把数字挤得太狠，精度就丢掉了。更小和更准之间的这份拉扯，就是你后面会遇到的 <span class="term" data-tip="用更少的位数存数字，也就是换一个更小的 dtype 箱子，好让 model 少占内存。代价是丢掉一些精度。">quantization</span> 的全部故事。</span></div></div>
        <p><span class="lang-en"><b>Intuition:</b> a size-1 axis is a stamp you can press as many times as needed; a size-3 axis is three fixed things that can't fill five slots. The technical point: broadcasting stretches size-1 axes for free (no memory), and that is exactly how a bias vector reaches a whole batch in one op.</span><span class="lang-zh"><b>直觉：</b>把 size-1 的那条 axis 想成「一个可以一直盖下去的印章」。你要按几次就按几次。size-3 的 axis 是三个固定的东西，它填不满五个位子。技术上说：broadcasting 把 size-1 的 axis 拉开，不花一点内存。一个 bias vector（向量）就是靠这一点，一次盖满整个 batch。</span></p>
        <button class="gotit" type="button"><span class="lang-en">Got the picture</span><span class="lang-zh">画面看懂了</span></button>
      </div>
    </section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
      <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h"><span class="lang-en">The one rule, the byte table, and why money is bytes</span><span class="lang-zh">那一条规则、字节表，还有为什么钱就是字节</span></span><span class="sec-tag"><span class="lang-en">Mechanism &amp; why</span><span class="lang-zh">原理和为什么</span></span></div>
      <div class="sec-body">
        <div class="lang-en">
        <h4>The one broadcasting rule</h4>
        <p><strong>The stamp, made precise.</strong> A dimension of size <strong>1</strong> is a stamp — a single thing NumPy can press out as many times as it needs to match its partner. A dimension of size 3 sitting next to a size 5 is three <em>fixed</em> things, not a stamp, so it can't fill five slots.</p>
        <p>To test whether two shapes fit, slide them together like two combs lined up at their <strong>right edge</strong>. Why the right? The rightmost number is the <strong>innermost</strong> count — the actual numbers sitting side by side in one row (like the features of a single example). You match those innermost units first, then work outward.</p>
        <p>At each tooth of the comb, the pair fits if the two numbers are <strong>equal</strong>, or if <strong>one side is 1</strong> (or simply not there — a shorter shape is read as having 1s added on its left). The size-1 / missing side is the stamp that stretches. If a pair is neither equal nor 1 → <strong>error</strong>.</p>
        </div>
        <div class="lang-zh">
        <h4>broadcasting 只有一条规则</h4>
        <p><strong>把印章说得准一点。</strong>一条 size 是 <strong>1</strong> 的 dimension（维度）就是一个印章。它只有一样东西，NumPy 可以一直按下去，按到和对面一样多。一条 size 是 3 的 dimension 旁边站着一条 size 5，那是三个<em>固定</em>的东西，不是印章。它填不满五个位子。</p>
        <p>要看两个 shape 合不合得上，就把它们像两把梳子一样，从<strong>右边</strong>对齐推到一起。为什么是右边？因为最右边那个数字，数的是<strong>最里面</strong>那一层。它数的是一行里真正并排坐着的那些数字，比如一个例子的那些 feature。你先把最里面这一层对上，再往外一层一层看。</p>
        <p>梳子的每一颗齿上，只要两个数字<strong>一样</strong>，这一对就合得上。或者<strong>有一边是 1</strong>，也合得上。有一边干脆没有这一位，也算合得上 —— 短的那个 shape，会被当成左边补了一串 1。是 1 的那一边、或者空着的那一边，就是会拉开的那个印章。要是一对既不相等、两边又都不是 1 → <strong>报错</strong>。</p>
        </div>
        <div class="callout c-ok"><span class="ic">✅</span><div><span class="lang-en"><code>(2, 3) + (3,)</code> → line up the right edges: <code>3</code> vs <code>3</code> ✓ (equal), then <code>2</code> vs <em>missing</em> ✓ (the shorter shape gets a 1, and a 1 is a stamp) → result <code>(2, 3)</code>.</span><span class="lang-zh"><code>(2, 3) + (3,)</code> → 右边对齐。<code>3</code> 对 <code>3</code> ✓，一样。再看 <code>2</code> 对<em>空着</em> ✓，短的那个 shape 补上一个 1，而 1 就是印章 → 结果是 <code>(2, 3)</code>。</span></div></div>
        <div class="callout c-warn"><span class="ic">❌</span><div><span class="lang-en"><code>(2, 3) + (2,)</code> → line up the right edges: <code>3</code> vs <code>2</code> — not equal, and neither one is 1, so neither is a stamp → <strong>error</strong>.</span><span class="lang-zh"><code>(2, 3) + (2,)</code> → 右边对齐。<code>3</code> 对 <code>2</code>，不一样。两个又都不是 1，谁都当不了印章 → <strong>报错</strong>。</span></div></div>
        <div class="callout c-info"><span class="ic">🪜</span><div><span class="lang-en"><b>Math Ladder — the broadcasting rule</b>
        <br><b>1 · In words:</b> line the two shapes up at their right edge; each axis pair works if the sizes are equal, or one of them is 1 (a missing axis counts as 1).
        <br><b>2 · The rule:</b> for the k-th axis from the right, <code>out[k] = max(a[k], b[k])</code> <em>only if</em> <code>a[k] == b[k]</code> or <code>min(a[k], b[k]) == 1</code>; otherwise it errors. Here <code>a[k]</code>, <code>b[k]</code> = the two input axis sizes, <code>out[k]</code> = the result axis size.
        <br><b>3 · Tiny numbers:</b> <code>(3, 1)</code> + <code>(1, 4)</code>. Axis −1: <code>1 vs 4</code> → one side is 1 → 4. Axis −2: <code>3 vs 1</code> → one side is 1 → 3. Result <code>(3, 4)</code> — 12 numbers grown from 3 + 4 inputs, with <em>zero</em> copies.
        <br><b>4 · Sanity check:</b> the result axis is never smaller than either input; a size-1 axis stretches, a size-3-vs-5 cannot. If you can't make every pair "equal or a 1", it must error — NumPy never silently guesses which to keep.</span><span class="lang-zh"><b>数学阶梯 —— broadcasting 的规则</b>
        <br><b>1 · 用话说：</b>把两个 shape 从右边对齐。每一对 axis 只要 size 一样就行，或者其中一个是 1 也行。少了一条 axis，就当它是 1。
        <br><b>2 · 规则：</b>从右边数第 k 条 axis，<code>out[k] = max(a[k], b[k])</code>。但这<em>只有在</em> <code>a[k] == b[k]</code> 或者 <code>min(a[k], b[k]) == 1</code> 的时候才成立。不然它就报错。这里 <code>a[k]</code> 和 <code>b[k]</code> 是两个输入在这条 axis 上的 size，<code>out[k]</code> 是结果在这条 axis 上的 size。
        <br><b>3 · 用小数字算一遍：</b><code>(3, 1)</code> + <code>(1, 4)</code>。倒数第 1 条 axis：<code>1</code> 对 <code>4</code> → 有一边是 1 → 取 4。倒数第 2 条 axis：<code>3</code> 对 <code>1</code> → 有一边是 1 → 取 3。结果是 <code>(3, 4)</code>。3 + 4 个输入数字长成了 12 个数字，而且<em>一次</em>复制都没做。
        <br><b>4 · 自己检查一下：</b>结果的 axis 永远不会比任何一个输入更小。size-1 的 axis 会拉开，size 3 对 size 5 不行。要是你没法让每一对都做到「一样，或者有一个 1」，它就一定会报错。NumPy 从来不会偷偷猜该留哪一个。</span></div></div>
        <h4><span class="lang-en">The byte table</span><span class="lang-zh">字节表</span></h4>
        <ul>
          <li><span class="lang-en"><code>float64</code> = 8 bytes · <code>float32</code> = 4 bytes · <span class="term" data-tip="A 16-bit floating type used a lot in ML: half the memory of float32, keeps the same range with less precision.">bfloat16</span> = 2 bytes each.</span><span class="lang-zh"><code>float64</code> = 8 个字节 · <code>float32</code> = 4 个字节 · <span class="term" data-tip="一种 16 位的浮点类型，ML 里用得很多。它只占 float32 一半的内存，能表示的范围一样大，但精度更低。">bfloat16</span> = 2 个字节。说的都是每一个数字。</span></li>
          <li><span class="lang-en">Memory of a tensor = number of elements × bytes/element. Halve the bytes → halve the memory.</span><span class="lang-zh">一个 tensor（张量）占的内存 = 元素个数 × 每个元素几个字节。字节减一半 → 内存也减一半。</span></li>
        </ul>
        <div class="lang-en">
        <h4>Why a frontier lab cares</h4>
        <p>Bytes are money and speed. A 7-billion-parameter model in <code>float32</code> needs ~28 GB just for weights; in <code>bfloat16</code>, ~14 GB — the difference between fitting on one chip or not. Broadcasting is how a bias vector <code>(features,)</code> gets added to a whole batch <code>(batch, features)</code> with no copy.</p>
        </div>
        <div class="lang-zh">
        <h4>前沿实验室为什么在意这件事</h4>
        <p>字节就是钱，也是速度。一个 70 亿参数的 model，用 <code>float32</code> 光存 weight（权重）就要大约 28 GB。换成 <code>bfloat16</code>，大约 14 GB。这个差别，决定了它装不装得进一块芯片。而 broadcasting 是这么用的：一个 <code>(features,)</code> 的 bias vector 加到一整批 <code>(batch, features)</code> 上，一次复制都不做。</p>
        </div>
        <div class="callout c-info"><span class="ic">🔗</span><div><span class="lang-en"><b>Remember this chain:</b> broadcasting → one op covers many shapes, no copies → less code + less memory. dtype → fewer bytes per number → bigger models fit on the same hardware. <strong>Every quantization trick later is just this dial.</strong></span><span class="lang-zh"><b>记住这条链：</b>broadcasting → 一个运算就管住很多种 shape，不做复制 → 代码更少，内存更省。dtype → 每个数字的字节更少 → 同一块硬件装得下更大的 model。<strong>后面所有的 quantization 招数，都只是在拧这一个旋钮。</strong></span></div></div>
        <div class="lang-en">
        <h4>The staff lens — one silent failure, one trade-off</h4>
        <p>Broadcasting and dtypes are the two places in array math where a bug does not announce itself — the code runs, prints numbers, and is quietly wrong.</p>
        </div>
        <div class="lang-zh">
        <h4>资深工程师的眼光 —— 一个悄悄发生的错，一个取舍</h4>
        <p>在 array 的运算里，broadcasting 和 dtype 是两个最会藏 bug 的地方。代码跑得动，数字也打印出来了，可它悄悄地就是错的。</p>
        </div>
        <div class="callout c-warn"><span class="ic">⚠️</span><div><span class="lang-en"><b>Failure mode (silent): the accidental broadcast (a bug that doesn't announce itself).</b> You expect two vectors of the same length to add element-wise. But if one is a row of shape <code>(n,)</code> and the other is a column of shape <code>(n, 1)</code>, broadcasting does not error — it stretches both and hands back an <code>(n, n)</code> grid (every pair added). No crash, no warning; you just have <code>n</code> times too many numbers, and a mean or loss computed over them looks plausible but is wrong. A senior engineer catches this in <b>code review</b> by asking "print the <code>.shape</code> right after the operation" — if the result gained a dimension you didn't intend, an accidental broadcast happened.</span><span class="lang-zh"><b>出错的样子（悄悄的）：一次意外的 broadcast。这个 bug 自己不会喊出来。</b>你以为两个一样长的 vector 会一个对一个加起来。可要是其中一个是 shape <code>(n,)</code> 的一行，另一个是 shape <code>(n, 1)</code> 的一列，broadcasting 不会报错。它把两边都拉开，还给你一张 <code>(n, n)</code> 的格子，里面装的是每一对的和。不崩，也不警告。你手上的数字变成了原来的 <code>n</code> 倍。拿它算出来的平均值或者 loss（损失）看着挺像样，其实是错的。一个资深工程师在 <b>code review</b> 里怎么抓到它？他会说一句「在这个运算后面马上打印 <code>.shape</code>」。要是结果多出了一条你没打算要的 dimension，那就是发生了一次意外的 broadcast。</span></div></div>
        <div class="callout c-info"><span class="ic">⚖️</span><div><span class="lang-en"><b>Trade-off: fewer bytes vs. less precision and range.</b> Dropping from <code>float32</code> (4 bytes) to <code>bfloat16</code> (2 bytes) halves the memory bill and lets a bigger model fit on the same chip. What you pay: each number now carries fewer significant digits, so tiny values can round to zero and long sums drift. In a <b>design review</b> the question is not "is float32 or bfloat16 better" — it is "which numbers can tolerate the lower precision (most weights and activations) and which must stay high precision (the running totals that accumulate error)." That split is exactly what mixed-precision training decides.</span><span class="lang-zh"><b>取舍：字节更少，还是精度和范围更大。</b>从 <code>float32</code>（4 个字节）掉到 <code>bfloat16</code>（2 个字节），内存账单少一半，同一块芯片就装得下更大的 model。你付出的代价是：每个数字带的有效位数变少了。很小的值会被四舍五入成 0，很长的一串加法会慢慢跑偏。在 <b>design review</b> 里，该问的不是「float32 和 bfloat16 哪个更好」。该问的是「哪些数字受得住低精度，哪些必须留在高精度」。大多数 weight 和 activation（激活值）受得住。那些一直在累加、会攒起误差的总和，就必须留在高精度。这条分界线，正是 mixed-precision training（混合精度训练）在决定的事。</span></div></div>
        <div class="callout c-ok"><span class="ic">🎤</span><div><span class="lang-en"><b>Say this in an interview:</b> "Broadcasting right-aligns the shapes: each axis pair must be equal or one must be 1, and a size-1 axis is stretched for free — no copy — which is how a <code>(features,)</code> bias adds onto a <code>(batch, features)</code> activation in one op. The nuance is the accidental broadcast: <code>(n,)</code> against <code>(n, 1)</code> doesn't error, it fans out to <code>(n, n)</code>, so I print <code>.shape</code> right after the op to catch a dimension I didn't intend."</span><span class="lang-zh"><b>面试时可以这样说：</b>「broadcasting 把两个 shape 从右边对齐。每一对 axis 要么一样，要么有一个是 1。size-1 的 axis 会被免费拉开，不做复制。所以一个 <code>(features,)</code> 的 bias，能用一个运算就加到 <code>(batch, features)</code> 的 activation 上。细一点说，麻烦在那次意外的 broadcast：<code>(n,)</code> 碰上 <code>(n, 1)</code> 不报错，它会摊成 <code>(n, n)</code>。所以我会在这个运算后面马上打印 <code>.shape</code>，看有没有多出一条我没打算要的 dimension。」</span></div></div>
        <button class="gotit" type="button"><span class="lang-en">Got the rule and the bytes</span><span class="lang-zh">规则和字节都记住了</span></button>
      </div>
    </section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
      <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h"><span class="lang-en">Leave today's evidence (strongly recommended)</span><span class="lang-zh">留下今天的证据（很建议做）</span></span><span class="sec-tag"><span class="lang-en">Produce</span><span class="lang-zh">动手做</span></span></div>
      <div class="sec-body">
        <p><span class="lang-en">Understanding ≠ being able to write it. Broadcast a few shapes and weigh a few dtypes yourself — that's what makes the day count. Pick one:</span><span class="lang-zh">看懂 ≠ 写得出来。自己动手 broadcast 几个 shape，再称一称几个 dtype 有多重 —— 这一天才算真的过了。挑一个做：</span></p>
        <div class="lang-en">
        <h4>Option A · write it yourself</h4>
        <p>Create <code>sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py</code>. Add a scalar to an array; add a <code>(3,)</code> row to a <code>(2,3)</code> grid; try <code>(2,3) + (2,)</code> inside a <code>try/except</code> and print the error; then print <code>.dtype</code> and <code>.nbytes</code> for the same data as <code>float64</code> vs <code>float32</code>. Run it with <code>python3 sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py</code>.</p>
        <h4>Option B · let Claude build it, then read it</h4>
        <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
        </div>
        <div class="lang-zh">
        <h4>做法 A · 自己写</h4>
        <p>新建 <code>sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py</code>。先给一个 array 加上一个 scalar。再把一个 <code>(3,)</code> 的行加到一个 <code>(2,3)</code> 的格子上。然后用 <code>try/except</code> 试一次 <code>(2,3) + (2,)</code>，把报错打印出来。最后拿同一份数据，分别做成 <code>float64</code> 和 <code>float32</code>，打印它们的 <code>.dtype</code> 和 <code>.nbytes</code>。用 <code>python3 sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py</code> 跑起来。</p>
        <h4>做法 B · 让 Claude 写，然后你读</h4>
        <p>把下面这段 prompt 复制回 Claude Code。它会让 Claude Code 帮你新建文件、写好代码，再跑一次。</p>
        </div>
        <div class="prompt">
          <div class="prompt-h"><span class="prompt-l"><span class="lang-en">ask Claude to build it</span><span class="lang-zh">让 Claude 来写</span></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
          <pre class="prompt-t" id="pp">Help me build my Module 1 Day 3 artifact.

Create sessions/m01-shape-of-data/day-03-broadcasting-dtypes/experiment.py that, with a comment on each step:
1. Prints x + 10 for x = np.array([1,2,3]) (scalar broadcast).
2. Adds row = np.array([10,20,30]) to g = np.array([[1,2,3],[4,5,6]]) and prints the (2,3) result, noting the (3,) row was stretched across both rows.
3. Wraps g + np.array([1,2]) in try/except and prints the broadcasting error (shapes (2,3) and (2,) do not align).
4. Builds b = np.array([1.,2.,3.]) as float64, makes f = b.astype(np.float32), and prints b.dtype, f.dtype, b.nbytes, f.nbytes — showing float32 uses half the bytes.
Then run it and paste the output at the bottom as a comment.</pre>
        </div>
        <h4><span class="lang-en">Acceptance criteria</span><span class="lang-zh">做到这几条就算过了</span></h4>
        <ul>
          <li><span class="lang-en">You show a scalar broadcast, a <code>(3,)</code> row added to a <code>(2,3)</code> grid, and a caught broadcasting error for <code>(2,3) + (2,)</code>.</span><span class="lang-zh">你演示了一次 scalar 的 broadcast，把一个 <code>(3,)</code> 的行加到了 <code>(2,3)</code> 的格子上，还接住了 <code>(2,3) + (2,)</code> 的 broadcasting 报错。</span></li>
          <li><span class="lang-en">You print <code>.dtype</code> and <code>.nbytes</code> for the same data as <code>float64</code> vs <code>float32</code>, and the float32 bill is exactly half.</span><span class="lang-zh">同一份数据，你分别用 <code>float64</code> 和 <code>float32</code> 打印了 <code>.dtype</code> 和 <code>.nbytes</code>，而 float32 的账单正好是一半。</span></li>
          <li><span class="lang-en">Before running each op, you can predict the result shape from the right-aligned rule.</span><span class="lang-zh">每一个运算跑之前，你都能用右边对齐那条规则，先说出结果的 shape。</span></li>
        </ul>
        <div class="callout c-info"><span class="ic">📓</span><div><span class="lang-en"><b>5-minute research log:</b> before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-03-broadcasting-dtypes/log.md</code>: (1) the one sentence you'd tell a teammate about the broadcasting rule; (2) which broadcast surprised you, or which dtype fact stuck; (3) one thing you're still unsure about.</span><span class="lang-zh"><b>5 分钟研究笔记：</b>关掉页面之前，用自己的话在 <code>sessions/m01-shape-of-data/day-03-broadcasting-dtypes/log.md</code> 里写三行：（1）关于 broadcasting 那条规则，你会用哪一句话讲给同事听；（2）哪一次 broadcast 让你意外，或者哪一个关于 dtype 的事实你记住了；（3）还有一件你不太确定的事。</span></div></div>
        <button class="gotit" type="button"><span class="lang-en">Done</span><span class="lang-zh">做完了</span></button>
      </div>
    </section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3><span class="lang-en">Module 1 · Day 3 complete!</span><span class="lang-zh">Module 1 · 第 3 天完成！</span></h3>
      <p><span class="lang-en">You can predict a broadcast from two shapes and explain why <code>float32</code> halves the memory of <code>float64</code>.<br>Next up: <b>Matmul &amp; Shapes</b> — the one operation every neural network is built from.</span><span class="lang-zh">你能看着两个 shape 就说出 broadcast 的结果，也能讲明白 <code>float32</code> 为什么把 <code>float64</code> 的内存砍掉一半。<br>下一站：<b>matmul 与 shape</b> —— 每一个神经网络都是用这一个运算搭起来的。</span></p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  scalar:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">1</span>, <span class="num">2</span>, <span class="num">3</span>])   <span class="dim"><span class="lang-en"># shape (3,)</span><span class="lang-zh"># shape 是 (3,)</span></span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> x + <span class="num">10</span>          <span class="dim"><span class="lang-en"># the 10 is stretched to every element</span><span class="lang-zh"># 这个 10 被拉开，盖到每一个元素上</span></span>\n'+
    '<span class="ok">array([11, 12, 13])</span>',
    take:'<span class="lang-en"><b>①  Scalar broadcast — the simplest stamp.</b> One number (10) is pressed across every slot of the array. You did this on Day 1 with <code>x + 10</code> — now it has a name.</span>'+
    '<span class="lang-zh"><b>①  scalar（标量）的 broadcast —— 最简单的一个印章。</b>一个数字（10）被压到 array 的每一个位子上。第 1 天你用 <code>x + 10</code> 做过这件事 —— 现在它有名字了。</span>'},
  row:{html:'<span class="prompt">&gt;&gt;&gt;</span> g = np.array([[<span class="num">1</span>,<span class="num">2</span>,<span class="num">3</span>], [<span class="num">4</span>,<span class="num">5</span>,<span class="num">6</span>]])  <span class="dim"># (2,3)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> row = np.array([<span class="num">10</span>, <span class="num">20</span>, <span class="num">30</span>])       <span class="dim"># (3,)</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> g + row       <span class="dim"><span class="lang-en"># row stretched down onto BOTH rows</span><span class="lang-zh"># 这一行被往下拉开，盖到两行上</span></span>\n'+
    '<span class="ok">array([[11, 22, 33],\n       [14, 25, 36]])</span>',
    take:'<span class="lang-en"><b>②  The general case.</b> Shapes differ — (2,3) vs (3,) — so the (3,) row becomes a stamp pressed <b>down</b> onto both rows of the grid. One bias row, the whole batch, no loop.</span>'+
    '<span class="lang-zh"><b>②  一般的情况。</b>两个 shape 不一样 —— (2,3) 对 (3,) —— 所以那个 (3,) 的行变成一个印章，<b>往下</b>盖到格子的两行上。一行 bias，一整个 batch，不用写循环。</span>'},
  dtype:{html:'<span class="prompt">&gt;&gt;&gt;</span> a = np.array([<span class="num">1</span>, <span class="num">2</span>, <span class="num">3</span>]);  a.dtype\n<span class="hl">dtype(\'int64\')</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> b = np.array([<span class="num">1.0</span>, <span class="num">2.0</span>, <span class="num">3.0</span>]);  b.dtype\n<span class="hl">dtype(\'float64\')</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> f = b.astype(np.float32)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> b.nbytes, f.nbytes     <span class="dim"><span class="lang-en"># float64=8B each, float32=4B</span><span class="lang-zh"># float64 每个 8 字节，float32 每个 4 字节</span></span>\n'+
    '<span class="hl">(24, 12)</span>',
    take:'<span class="lang-en"><b>③  dtype = kind + box size.</b> Whole numbers default to int64, decimals to float64. Casting to float32 puts each number in a half-size box (24 → 12 bytes). Half the box → half the memory — the seed of quantization.</span>'+
    '<span class="lang-zh"><b>③  dtype = 类型 + 箱子大小。</b>整数默认是 int64，小数默认是 float64。转成 float32，每个数字就住进一个只有一半大的箱子（24 → 12 字节）。箱子小一半 → 内存也小一半 —— 这就是 quantization 的种子。</span>'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:'<svg viewBox="0 0 520 130" role="img" aria-label="Scalar 10 broadcast across a length-3 array"><g font-family="monospace" font-size="14" text-anchor="middle"><rect x="70" y="40" width="46" height="40" rx="6" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text x="93" y="66" fill="#1F6280">1</text><rect x="120" y="40" width="46" height="40" rx="6" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text x="143" y="66" fill="#1F6280">2</text><rect x="170" y="40" width="46" height="40" rx="6" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text x="193" y="66" fill="#1F6280">3</text><text x="240" y="66" font-size="18" fill="#6B645E">+</text><rect x="265" y="40" width="46" height="40" rx="6" fill="#FCF3DC" stroke="#A67D08" stroke-width="2"/><text x="288" y="66" fill="#6E5205" font-weight="bold">10</text><text class="lang-en" x="288" y="100" font-size="10" fill="#C93B3B">one number…</text><text class="lang-zh" x="288" y="100" font-size="10" fill="#C93B3B">就一个数字…</text><text x="340" y="66" font-size="18" fill="#6B645E">=</text><rect x="365" y="40" width="46" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="388" y="66" fill="#1a5c38" font-weight="bold">11</text><rect x="413" y="40" width="46" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="436" y="66" fill="#1a5c38" font-weight="bold">12</text><rect x="461" y="40" width="46" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="484" y="66" fill="#1a5c38" font-weight="bold">13</text></g></svg>',
  note:'<span class="lang-en"><b>Start with the simplest stamp.</b> <code>x + 10</code>: the single number 10 is <b>pressed</b> across all three slots, then added slot-by-slot. You used this on Day 1 without knowing its name.</span><span class="lang-zh"><b>先从最简单的印章开始。</b><code>x + 10</code>：那一个数字 10 被<b>按</b>到三个位子上，然后一个位子一个位子地加。第 1 天你就用过它，只是那时还不知道它的名字。</span>'},
 {viz:'<svg viewBox="0 0 520 130" role="img" aria-label="Broadcasting rule: align shapes from the right, size-1 or missing stretches"><g font-family="monospace" font-size="14"><text class="lang-en" x="260" y="24" text-anchor="middle" fill="#2C2A28">align shapes from the RIGHT →</text><text class="lang-zh" x="260" y="24" text-anchor="middle" fill="#2C2A28">两个 shape 从右边对齐 →</text><text x="150" y="60" text-anchor="end" fill="#1F6280">( 2 ,</text><text x="210" y="60" text-anchor="middle" fill="#1F6280" font-weight="bold">3 )</text><text x="150" y="90" text-anchor="end" fill="#6E5205">(</text><text x="210" y="90" text-anchor="middle" fill="#6E5205" font-weight="bold">3 )</text><line x1="198" y1="68" x2="198" y2="78" stroke="#2D8B55" stroke-width="2"/><text x="300" y="60" fill="#6B645E">3 = 3  ✓</text><text class="lang-en" x="300" y="90" fill="#6B645E">2 vs (missing) ✓ → stretch</text><text class="lang-zh" x="300" y="90" fill="#6B645E">2 对上空位 ✓ → 拉开</text><text class="lang-en" x="260" y="118" text-anchor="middle" fill="#2D8B55" font-weight="bold">compatible → result (2, 3)</text><text class="lang-zh" x="260" y="118" text-anchor="middle" fill="#2D8B55" font-weight="bold">合得上 → 结果 (2, 3)</text></g></svg>',
  note:'<span class="lang-en"><b>The one rule.</b> Slide the two shapes together at their <b>right edge</b>, like two combs. At each tooth they fit if the numbers are <b>equal</b>, or if <b>one side is 1</b> (a 1 is a stamp — it presses out to any size). Check the rightmost pair first, then move left.</span><span class="lang-zh"><b>只有这一条规则。</b>把两个 shape 从<b>右边</b>推到一起，像两把梳子。每一颗齿上，两个数字<b>一样</b>就合得上，或者<b>有一边是 1</b> 也行。1 就是印章，你想按多大就按多大。先看最右边那一对，再往左走。</span>'},
 {viz:'<svg viewBox="0 0 520 150" role="img" aria-label="A (3,) row broadcast down onto both rows of a (2,3) grid"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="40" y="40" width="42" height="34" rx="5" fill="#FCF3DC" stroke="#A67D08" stroke-width="2"/><rect x="84" y="40" width="42" height="34" rx="5" fill="#FCF3DC" stroke="#A67D08" stroke-width="2"/><rect x="128" y="40" width="42" height="34" rx="5" fill="#FCF3DC" stroke="#A67D08" stroke-width="2"/><text x="61" y="63" fill="#6E5205">10</text><text x="105" y="63" fill="#6E5205">20</text><text x="149" y="63" fill="#6E5205">30</text><text class="lang-en" x="105" y="92" fill="#6B645E" font-size="11">row (3,)</text><text class="lang-zh" x="105" y="92" fill="#6B645E" font-size="11">一行 (3,)</text><path d="M105 96 L105 112" stroke="#A67D08" stroke-width="2" stroke-dasharray="4,3"/><path d="M105 96 L105 130" stroke="#A67D08" stroke-width="2" stroke-dasharray="4,3"/><text class="lang-en" x="210" y="90" font-size="18" fill="#6B645E">onto</text><text class="lang-zh" x="210" y="90" font-size="18" fill="#6B645E">盖到</text><rect x="300" y="40" width="42" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><rect x="344" y="40" width="42" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><rect x="388" y="40" width="42" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><rect x="300" y="78" width="42" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><rect x="344" y="78" width="42" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><rect x="388" y="78" width="42" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="321" y="63" fill="#1a5c38">11</text><text x="365" y="63" fill="#1a5c38">22</text><text x="409" y="63" fill="#1a5c38">33</text><text x="321" y="101" fill="#1a5c38">14</text><text x="365" y="101" fill="#1a5c38">25</text><text x="409" y="101" fill="#1a5c38">36</text><text x="365" y="134" fill="#2D8B55" font-weight="bold">(2,3) + (3,) = (2,3)</text></g></svg>',
  note:'<span class="lang-en"><b>The general case.</b> A <code>(3,)</code> row added to a <code>(2,3)</code> grid: the row is a stamp pressed <b>down</b> onto both rows, then added. This is exactly how a bias vector adds to every row of a batch.</span><span class="lang-zh"><b>一般的情况。</b>把一个 <code>(3,)</code> 的行加到一个 <code>(2,3)</code> 的格子上：这一行就是印章，<b>往下</b>盖到两行上，然后相加。一个 bias vector 加到一个 batch 的每一行上，用的就是这一招。</span>'},
 {viz:'<svg viewBox="0 0 520 130" role="img" aria-label="Incompatible shapes (2,3) and (2,) raise a broadcasting error"><g font-family="monospace" font-size="14"><text x="150" y="55" text-anchor="end" fill="#1F6280">( 2 ,</text><text x="210" y="55" text-anchor="middle" fill="#C93B3B" font-weight="bold">3 )</text><text x="150" y="88" text-anchor="end" fill="#6E5205">(</text><text x="210" y="88" text-anchor="middle" fill="#C93B3B" font-weight="bold">2 )</text><text class="lang-en" x="300" y="72" fill="#C93B3B" font-weight="bold">3 ≠ 2, neither is 1</text><text class="lang-zh" x="300" y="72" fill="#C93B3B" font-weight="bold">3 ≠ 2，都不是 1</text><rect x="120" y="100" width="280" height="22" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text class="lang-en" x="260" y="115" text-anchor="middle" font-size="11" fill="#C93B3B">ValueError: operands could not be broadcast</text><text class="lang-zh" x="260" y="115" text-anchor="middle" font-size="11" fill="#C93B3B">ValueError: operands could not be broadcast</text></g></svg>',
  note:'<span class="lang-en"><b>When it fails.</b> <code>(2,3) + (2,)</code>: line up the right edges → 3 vs 2. Not equal, and neither is 1, so neither can be a stamp → NumPy raises a broadcasting error. A row length that doesn\'t match is a very common bug.</span><span class="lang-zh"><b>它失败的时候。</b><code>(2,3) + (2,)</code>：右边对齐 → 3 对 2。不一样，两个又都不是 1，谁都当不了印章 → NumPy 抛出一个 broadcasting 报错。行的长度对不上，是一个非常常见的 bug。</span>'},
 {viz:'<svg viewBox="0 0 520 120" role="img" aria-label="float64 8 bytes, float32 4 bytes, bfloat16 2 bytes per number"><g font-family="monospace" font-size="12" text-anchor="middle"><rect x="60" y="40" width="120" height="34" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="120" y="62" fill="#C93B3B" font-weight="bold">float64 · 8 B</text><text class="lang-zh" x="120" y="62" fill="#C93B3B" font-weight="bold">float64 · 8 B</text><rect x="200" y="40" width="90" height="34" rx="5" fill="#FCF3DC" stroke="#A67D08" stroke-width="2"/><text class="lang-en" x="245" y="62" fill="#6E5205" font-weight="bold">float32 · 4 B</text><text class="lang-zh" x="245" y="62" fill="#6E5205" font-weight="bold">float32 · 4 B</text><rect x="310" y="40" width="60" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="340" y="62" fill="#1a5c38" font-weight="bold">bf16 · 2 B</text><text class="lang-en" x="260" y="100" fill="#2C2A28">same number, smaller box → less memory</text><text class="lang-zh" x="260" y="100" fill="#2C2A28">同一个数字，箱子更小 → 更省内存</text></g></svg>',
  note:'<span class="lang-en"><b>Now the dtype dial.</b> The exact same value costs 8, 4, or 2 bytes depending on its dtype. Choosing a smaller box is the single lever that decides how much memory an array (or a whole model) eats.</span><span class="lang-zh"><b>现在来拧 dtype 这个旋钮。</b>完全一样的一个值，可能花 8 个字节、4 个字节，也可能只花 2 个字节。看它的 dtype 是什么。挑一个更小的箱子，就是决定一个 array、或者一整个 model 吃掉多少内存的那一个开关。</span>'},
 {viz:'<svg viewBox="0 0 520 140" role="img" aria-label="A 7B model shrinks from 28GB in float32 to 14GB in bfloat16"><g font-family="monospace"><rect x="90" y="30" width="150" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="165" y="55" text-anchor="middle" font-size="13" fill="#C93B3B" font-weight="bold">7B · fp32 ≈ 28 GB</text><text x="265" y="55" text-anchor="middle" font-size="16" fill="#6B645E">→</text><rect x="300" y="38" width="110" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="355" y="63" text-anchor="middle" font-size="13" fill="#1a5c38" font-weight="bold">bf16 ≈ 14 GB</text><text class="lang-en" x="260" y="108" text-anchor="middle" font-size="12" fill="#2C2A28">halving the dtype halves the memory bill</text><text class="lang-zh" x="260" y="108" text-anchor="middle" font-size="12" fill="#2C2A28">dtype 减一半，内存账单也减一半</text><text class="lang-en" x="260" y="128" text-anchor="middle" font-size="11" fill="#6B645E">fits on one chip, or it doesn&apos;t — this is why labs live in bytes</text><text class="lang-zh" x="260" y="128" text-anchor="middle" font-size="11" fill="#6B645E">装不装得进一块芯片，就看这个 —— 所以实验室天天在算字节</text></g></svg>',
  note:'<span class="lang-en"><b>Why it matters at frontier scale.</b> Weights = params × bytes/param. A 7-billion-param model is ~28 GB in float32 but ~14 GB in bfloat16 — often the difference between fitting on a single accelerator or not. Every quantization method you\'ll study is a cleverer way to shrink this box without losing too much precision.</span><span class="lang-zh"><b>它在前沿规模上为什么重要。</b>weight 占的内存 = 参数个数 × 每个参数几个字节。一个 70 亿参数的 model，用 float32 大约 28 GB，用 bfloat16 大约 14 GB。这常常就决定了它装不装得进一块加速卡。你后面会学的每一种 quantization 方法，都是在更聪明地缩小这个箱子，同时又不丢太多精度。</span>'}
];
@@@ region name=QS
var QS=[
 {q:'<span class="lang-en">1. You add two arrays you believe are the same-length vectors and expect a vector back, but a later mean comes out strangely small. No error was raised. You print shapes: one is <code>(100,)</code>, the other is <code>(100, 1)</code>, and the sum is <code>(100, 100)</code>. What happened?</span>'+
    '<span class="lang-zh">1. 你把两个 array 加起来。你以为它们是两个一样长的 vector，结果也该是一个 vector。可后面算出来的平均值小得奇怪。程序一点都没报错。你把 shape 打印出来看：一个是 <code>(100,)</code>，另一个是 <code>(100, 1)</code>，而它们的和是 <code>(100, 100)</code>。发生了什么？</span>',
  opts:['<span class="lang-en">NumPy silently truncated the longer array</span><span class="lang-zh">NumPy 悄悄把长的那个 array 截短了</span>',
        '<span class="lang-en">Broadcasting stretched a row against a column and built a 100&times;100 grid of every pair, so your mean averaged 10,000 numbers instead of 100</span><span class="lang-zh">broadcasting 把一行和一列都拉开了，搭出一张 100&times;100 的格子，里面是每一对的和。所以你的平均值平均的是 10,000 个数字，不是 100 个</span>',
        '<span class="lang-en">The dtypes disagreed and NumPy zero-filled the gap</span><span class="lang-zh">两边的 dtype 不一样，NumPy 用 0 把空缺填上了</span>',
        '<span class="lang-en">Nothing is wrong; (100,1) and (100,) are identical shapes</span><span class="lang-zh">没什么问题；(100,1) 和 (100,) 是一样的 shape</span>'],
  ans:1, fb:'<span class="lang-en">Right. <code>(100,)</code> vs <code>(100, 1)</code> are broadcast-compatible: line up the right edges, the <code>1</code> is a stamp that presses out to 100, then the missing leading dim on the <code>(100,)</code> side presses out too — giving a <code>(100, 100)</code> grid of every pairwise sum. No error fires, so the bug is silent. Fix: print <code>.shape</code> right after the op and reshape one operand (e.g. <code>.ravel()</code>) so both are truly 1-D.</span>'+
    '<span class="lang-zh">对了。<code>(100,)</code> 和 <code>(100, 1)</code> 是能 broadcast 的：右边对齐，那个 <code>1</code> 是印章，它压开成 100。再看 <code>(100,)</code> 这一边，它前面少了一条 dimension，那一条也压开了。于是你拿到一张 <code>(100, 100)</code> 的格子，里面是每一对的和。它不报错，所以这个 bug 藏得很好。修法：在这个运算后面马上打印 <code>.shape</code>，再把其中一边 reshape 一下，比如用 <code>.ravel()</code>，让两边都真的是 1-D。</span>'},
 {q:'<span class="lang-en">2. Two dimensions are broadcast-compatible if they are equal, or if one of them is ___.</span>'+
    '<span class="lang-zh">2. 两条 dimension 能 broadcast 的条件是：它们相等，或者其中一条是 ___。</span>',
  opts:['0','1','<span class="lang-en">even</span><span class="lang-zh">偶数</span>','<span class="lang-en">a float</span><span class="lang-zh">一个 float（浮点数）</span>'],
  ans:1, fb:'<span class="lang-en">Right. A dimension of size 1 (or missing) is the stamp — it presses out to match whatever size sits opposite it.</span>'+
    '<span class="lang-zh">对了。size 是 1 的那条 dimension，或者干脆空着的那条，就是印章 —— 它会压开，去配上对面的任何 size。</span>'},
 {q:'<span class="lang-en">3. What does an array\'s <code>dtype</code> control?</span>'+
    '<span class="lang-zh">3. 一个 array 的 <code>dtype</code> 管的是什么？</span>',
  opts:['<span class="lang-en">Its shape</span><span class="lang-zh">它的 shape</span>',
        '<span class="lang-en">The kind and byte-size of each element (precision & memory)</span><span class="lang-zh">每个元素的类型和字节大小，也就是精度和内存</span>',
        '<span class="lang-en">How many dimensions it has</span><span class="lang-zh">它有几条 dimension</span>',
        '<span class="lang-en">Which axis is the batch</span><span class="lang-zh">哪一条 axis 是 batch</span>'],
  ans:1, fb:'<span class="lang-en">Right. dtype = the shared type (int/float) and how many bytes each number costs — which sets the memory bill.</span>'+
    '<span class="lang-zh">对了。dtype = 共用的类型（int 还是 float），加上每个数字花几个字节 —— 这就定下了内存的账单。</span>'},
 {q:'<span class="lang-en">4. Compared to <code>float32</code>, how much memory does <code>bfloat16</code> use per number?</span>'+
    '<span class="lang-zh">4. 和 <code>float32</code> 比，<code>bfloat16</code> 每个数字用掉多少内存？</span>',
  opts:['<span class="lang-en">Twice as much</span><span class="lang-zh">两倍</span>',
        '<span class="lang-en">The same</span><span class="lang-zh">一样多</span>',
        '<span class="lang-en">Half</span><span class="lang-zh">一半</span>',
        '<span class="lang-en">Four times as much</span><span class="lang-zh">四倍</span>'],
  ans:2, fb:'<span class="lang-en">Right. bfloat16 is 2 bytes vs float32\'s 4 — half the memory. That is the core lever behind quantization.</span>'+
    '<span class="lang-zh">对了。bfloat16 是 2 个字节，float32 是 4 个字节 —— 内存少一半。这就是 quantization 背后那个核心的开关。</span>'}
];
