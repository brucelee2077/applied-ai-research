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
<div class="brand-sub"><span class="lang-en">Foundations · M1 Day 5</span><span class="lang-zh">基础 · M1 第 5 天</span></div>
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
<a class="lnav prev" href="../day-04-matmul-and-shapes/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t"><span class="lang-en">Matmul &amp; Shapes</span><span class="lang-zh">matmul 与 shape</span></span></a>
@@@ region name=nav_next
<a class="lnav next" href="../day-06-random-seeds/lesson.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t"><span class="lang-en">Random Seeds &amp; PRNG</span><span class="lang-zh">random seed 与 PRNG</span></span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker"><span class="lang-en">Module 1 · Represent · Day 5</span><span class="lang-zh">Module 1 · 表示 · 第 5 天</span></span>
      <h1><span class="lang-en">Logs &amp; Exponents</span><span class="lang-zh">log 与 exponent</span><span class="sub"><span class="lang-en">How to Read a Scaling-Law Plot</span><span class="lang-zh">怎么读懂一张 scaling-law 图</span></span></h1>
      <p class="lede"><span class="lang-en">The last bit of math before real ML. The famous <strong>scaling laws</strong> — the curves that predict how much better a model gets as you spend more compute — are <strong>power laws</strong>. On a log-log plot, a power law becomes a straight line. Learn to read that line and you can read the most important charts in the field.</span><span class="lang-zh">真正开始做 ML 之前，最后一点数学。那些有名的 <strong>scaling law（缩放定律）</strong>，其实都是 <strong>power law（幂律）</strong>。它们画的是这件事：你多花一点 compute（算力），模型会好多少。把一条 power law 画到 log（对数）轴上 —— 横轴纵轴都用 log，也就是 log-log 图 —— 它会变成一条直线。学会读这条直线，你就能读懂这个领域里最重要的那些图。</span></p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><span class="lang-en"><b>By the end, you'll be able to:</b> say what a log is, turn a power law into a straight line, and read the slope of a log-log line as its exponent.</span><span class="lang-zh"><b>学完这一天，你能做到：</b>说清楚 log 是什么。把一条 power law 变成一条直线。还能把 log-log 直线的 slope（斜率）读成它的 exponent（指数）。</span></div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h"><span class="lang-en">Exponents, logs, and power laws</span><span class="lang-zh">exponent、log 和 power law</span></span><span class="sec-tag"><span class="lang-en">What is it</span><span class="lang-zh">这是什么</span></span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><span class="lang-en"><b>What this assumes:</b> just arithmetic and Days 1–4. <b>No calculus.</b> This lesson is lighter on code — it's the reading skill for the plots you'll meet in the scaling-law modules.</span><span class="lang-zh"><b>你需要先会：</b>算术，还有第 1 到第 4 天的内容。<b>不需要微积分。</b>这一天的代码比较少。它教的是一种读图的本领。后面讲 scaling law 的那些 module 里，你会一直用到它。</span></div></div>

      <div class="lang-en">
      <h4>What is an exponent?</h4>
      <p>An <span class="term" data-tip="Repeated multiplication: x^b means multiply x by itself b times. 2^3 = 2·2·2 = 8.">exponent</span> is repeated multiplication: <code>2³ = 2·2·2 = 8</code>. The small number on top says how many times.</p>

      <h4>What is a log?</h4>
      <p>A <span class="term" data-tip="The inverse of an exponent. log10(1000)=3 because 10^3=1000. It answers: 10 to what power gives this number?">log</span> asks the reverse question: "10 to <em>what power</em> gives this number?" Since <code>10³ = 1000</code>, <code>log₁₀(1000) = 3</code>. A log tells you the number of <span class="term" data-tip="A factor of 10. Going from 100 to 1000 is one order of magnitude.">orders of magnitude</span>. Log and exponent <strong>undo each other</strong>.</p>

      <h4>What is a power law?</h4>
      <p>A <span class="term" data-tip="A relationship y = a·x^b, where one quantity is a fixed power of another. Scaling laws are power laws.">power law</span> is a relationship <code>y = a·x^b</code> — one thing is a power of another. Scaling laws (loss vs compute) are power laws. The magic you'll use: draw a power law on <strong>log-log</strong> axes and it becomes a straight <strong>line</strong>, whose slope is the exponent <code>b</code>. Keep going 👇</p>
      </div>
      <div class="lang-zh">
      <h4>exponent 是什么？</h4>
      <p><span class="term" data-tip="重复的乘法。x^b 就是把 x 自己乘 b 次。2^3 = 2·2·2 = 8。">exponent</span> 就是重复的乘法：<code>2³ = 2·2·2 = 8</code>。上面那个小数字，告诉你要乘几次。</p>

      <h4>log 是什么？</h4>
      <p><span class="term" data-tip="exponent 反过来问的那个问题。log10(1000)=3，因为 10^3=1000。它问：10 的几次方等于这个数？">log</span> 问的是反过来的那个问题：「10 的<em>几次方</em>等于这个数？」因为 <code>10³ = 1000</code>，所以 <code>log₁₀(1000) = 3</code>。log 告诉你这个数有几个 <span class="term" data-tip="一个 ×10 就是一个数量级。从 100 到 1000 是一个数量级。">orders of magnitude（数量级）</span>。log 和 exponent 会<strong>互相消掉</strong>。</p>

      <h4>power law 是什么？</h4>
      <p><span class="term" data-tip="一种关系 y = a·x^b。一个量是另一个量的固定次方。scaling law 就是 power law。">power law</span> 是这样一种关系：<code>y = a·x^b</code>。一个东西，是另一个东西的某个次方。scaling law 讲的是 loss（损失）和 compute 的关系，它就是一条 power law。今天要用的妙招是这样：把一条 power law 画在 <strong>log-log</strong> 轴上，它就变成一条直<strong>线</strong>。这条线的 slope，就是 exponent <code>b</code>。继续往下看 👇</p>
      </div>
      <div class="callout c-warn"><span class="ic">😕</span><div><span class="lang-en"><b>Why this trips people up:</b> people memorize the log <em>rules</em> but never anchor the one question a log asks — "10 to what power gives this number?" — so a log-log plot feels like a foreign language instead of the simplest chart in ML.</span><span class="lang-zh"><b>为什么这里容易卡住：</b>很多人把 log 的<em>规则</em>背了下来，却说不出 log 到底在问什么。那一个问题是：「10 的几次方等于这个数？」没有抓住它，一张 log-log 图看起来就像外语。其实它是 ML 里最简单的一种图。</span></div></div>
      <div class="callout c-info"><span class="ic">🏭</span><div><span class="lang-en"><b>A real one:</b> DeepMind's Chinchilla paper trained a 70-billion-parameter model on 1.4 trillion tokens — a ratio of 20 tokens per parameter they found by reading the slope of a log-log line, exactly like you're about to.</span><span class="lang-zh"><b>一个真实的例子：</b>DeepMind 的 Chinchilla 论文用 1.4 万亿个 token，训练了一个 700 亿 parameter（参数）的模型。每个 parameter 配 20 个 token —— 这个比例，是他们读一条 log-log 直线的 slope 读出来的。你马上也要这样读一次。</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Got the three words — let's go</span><span class="lang-zh">三个词都记住了 —— 走吧</span></button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h"><span class="lang-en">The Richter scale</span><span class="lang-zh">地震的震级</span></span><span class="sec-tag"><span class="lang-en">Intuition</span><span class="lang-zh">直觉</span></span></div>
  <div class="sec-body">
      <div class="relate">
        <div class="card"><span class="big">📈</span><h5><span class="lang-en">A log axis = "×10 per step"</span><span class="lang-zh">log 轴 = 「每格 ×10」</span></h5><p><span class="lang-en">On a normal ruler, steps add (+1, +2). On a <strong>log</strong> ruler, each step <strong>multiplies</strong> by 10: 1, 10, 100, 1000 are evenly spaced. Earthquakes (Richter) and loudness (decibels) work this way.</span><span class="lang-zh">普通尺子上，一格一格是加上去的（+1，+2）。<strong>log</strong> 尺子上，每一格是<strong>乘</strong> 10。所以 1、10、100、1000 之间的距离一样宽。地震的震级和声音的大小（分贝）就是这样量出来的。</span></p></div>
        <div class="card"><span class="big">📏</span><h5><span class="lang-en">Why it helps</span><span class="lang-zh">它为什么好用</span></h5><p><span class="lang-en">A log turns "times 10" into "plus 1." Huge ranges (1 to a billion) fit on one axis, and any multiply-relationship becomes a straight line you can measure.</span><span class="lang-zh">log 把「乘 10」变成「加 1」。很大的范围（从 1 到十亿）就能挤进一根轴里。任何「乘」的关系，都会变成一条你量得出来的直线。</span></p></div>
      </div>
      <p><span class="lang-en"><strong>In one line:</strong> a log squashes a "multiply" world into an "add" world — that's why power laws straighten out on log axes.</span><span class="lang-zh"><strong>一句话：</strong>log 把一个「乘」的世界压成一个「加」的世界。所以 power law 在 log 轴上会被拉直。</span></p>
      <p><span class="lang-en"><b>Intuition:</b> a log is like a ruler where every step is ×10, squashing huge numbers down into small, readable ones. The technical point is that a log converts multiplication into addition, so a range from 1 to a billion fits on one axis and any "multiply" relationship becomes a straight line you can measure.</span><span class="lang-zh"><b>直觉：</b>log 就像一把「每格 ×10」的尺子。它把庞大的数字压成好读的小数字。技术上说：log 把乘法变成加法。所以从 1 到十亿的范围能放进一根轴里，任何「乘」的关系也都变成一条你量得出来的直线。</span></p>
      <p><span class="lang-en"><b>In code:</b> you meet this as <em>log-space</em> — multiplying hundreds of tiny probabilities underflows to <code>0</code>, so libraries add <code>log</code>-probabilities instead. Same "multiply → add" trick, just used to keep the arithmetic safe.</span><span class="lang-zh"><b>写代码的时候：</b>你会以 <em>log-space（对数空间）</em> 的样子碰到它。几百个很小的概率乘起来，会 underflow（下溢）成 <code>0</code>。所以库里改成把 <code>log</code> 概率加起来。还是那个「乘 → 加」的招数，只是这次用来保护算术不出错。</span></p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><span class="lang-en">Where it breaks: logs only work for <strong>positive</strong> numbers. And a power law only becomes a straight line when <strong>both</strong> axes are log (a "log-log" plot) — not just one.</span><span class="lang-zh">这个比喻不对的地方：log 只对<strong>正数</strong>有用。而且只有<strong>两根</strong>轴都用 log，也就是画成「log-log」图，power law 才会变成直线。只用一根轴不行。</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Got the picture</span><span class="lang-zh">画面看懂了</span></button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h"><span class="lang-en">Why a power law is a straight line — and why that runs the industry</span><span class="lang-zh">power law 为什么是一条直线 —— 以及它为什么撑起整个行业</span></span><span class="sec-tag"><span class="lang-en">Mechanism &amp; why</span><span class="lang-zh">原理和为什么</span></span></div>
  <div class="sec-body">
      <div class="lang-en">
      <h4>The one derivation</h4>
      <p>Start with a power law and take the log of both sides:</p>
      </div>
      <div class="lang-zh">
      <h4>只有这一个推导</h4>
      <p>从一条 power law 开始，两边同时取 log：</p>
      </div>
      <div class="callout c-ok"><span class="ic">🧮</span><div><span class="lang-en"><code>y = a · x^b</code>&nbsp;&nbsp;→&nbsp;&nbsp;<code>log y = log a + b · log x</code><br><span style="font-size:.85rem;color:var(--text2)">Rename <code>Y = log y</code> and <code>X = log x</code>: that's <code>Y = (log a) + b·X</code> — the equation of a <strong>straight line</strong> with slope <code>b</code>.</span></span><span class="lang-zh"><code>y = a · x^b</code>&nbsp;&nbsp;→&nbsp;&nbsp;<code>log y = log a + b · log x</code><br><span style="font-size:.85rem;color:var(--text2)">把 <code>log y</code> 改名叫 <code>Y</code>，把 <code>log x</code> 改名叫 <code>X</code>：式子就成了 <code>Y = (log a) + b·X</code>。这就是一条<strong>直线</strong>的式子，它的 slope 是 <code>b</code>。</span></span></div></div>
      <p><span class="lang-en">So on log-log axes, a power law is a line, and its <strong>slope is the exponent</strong>. Two log rules make it work: <code>log(a·b) = log a + log b</code>, and <code>log(x^b) = b·log x</code>.</span><span class="lang-zh">所以在 log-log 轴上，一条 power law 就是一条直线。而它的 <strong>slope 就是 exponent</strong>。让这件事成立的是两条 log 规则：<code>log(a·b) = log a + log b</code>，还有 <code>log(x^b) = b·log x</code>。</span></p>
      <div class="callout c-info"><span class="ic">🪜</span><div><span class="lang-en"><b>Math Ladder — a power law becomes a straight line</b>
      <br><b>1 · In words:</b> take the log of both sides of a power law and the curve turns into a straight line whose steepness is the exponent.
      <br><b>2 · The formula:</b> <code>y = a·x^b</code> → <code>log y = log a + b·log x</code> — <code>y</code>, <code>x</code> = the two quantities; <code>a</code> = a constant that sets the height; <code>b</code> = the exponent you want. Rename <code>Y = log y</code>, <code>X = log x</code> and it reads <code>Y = (log a) + b·X</code>, a line with slope <code>b</code>.
      <br><b>3 · Tiny numbers:</b> <code>y = x²</code> with <code>x = 1, 10, 100</code> → <code>y = 1, 100, 10000</code>. Take <code>log₁₀</code>: <code>X = 0, 1, 2</code> and <code>Y = 0, 2, 4</code>. Y climbs 2 for every 1 that X climbs.
      <br><b>4 · Sanity check:</b> slope <code>= (4 − 0) / (2 − 0) = 2</code>, exactly the exponent <code>b = 2</code> you started with. If the slope doesn't match the power you put in, you forgot to <code>log</code> one of the axes.</span><span class="lang-zh"><b>数学阶梯 —— 一条 power law 怎么变成一条直线</b>
      <br><b>1 · 用话说：</b>把一条 power law 两边都取 log，那条弯的线就变成一条直线。这条直线有多陡，就是 exponent。
      <br><b>2 · 式子：</b><code>y = a·x^b</code> → <code>log y = log a + b·log x</code> —— <code>y</code>、<code>x</code> 是那两个量；<code>a</code> 是一个常数，它决定线的高低；<code>b</code> 就是你要的那个 exponent。把 <code>log y</code> 改名叫 <code>Y</code>，<code>log x</code> 改名叫 <code>X</code>，式子就读作 <code>Y = (log a) + b·X</code>，一条 slope 为 <code>b</code> 的直线。
      <br><b>3 · 拿小数字试一试：</b><code>y = x²</code>，取 <code>x = 1, 10, 100</code> → <code>y = 1, 100, 10000</code>。再取 <code>log₁₀</code>：<code>X = 0, 1, 2</code>，<code>Y = 0, 2, 4</code>。X 每涨 1，Y 就涨 2。
      <br><b>4 · 检查一下对不对：</b>slope <code>= (4 − 0) / (2 − 0) = 2</code>，正好是你一开始放进去的 exponent <code>b = 2</code>。如果 slope 和你放进去的次方不一样，那就是有一根轴你忘了取 <code>log</code>。</span></div></div>

      <div class="lang-en">
      <h4>Why a frontier lab cares</h4>
      <p>Kaplan et al.'s 2020 scaling-law paper found loss falls with compute at roughly <code>α ≈ 0.05</code> — plot loss vs. compute on log-log axes and you get a nearly straight line with that slope. That's exactly the method behind Chinchilla's 20-tokens-per-parameter finding from the teaser above. Two years later, DeepMind used that same log-log approach on compute-optimal training, and found that tokens and parameters should grow together, not just model size alone. That one exponent — the slope you're about to read off a log-log plot — predicts how much better a model gets for every 10× more compute, which is the business case for spending millions on a training run.</p>
      </div>
      <div class="lang-zh">
      <h4>前沿实验室为什么在意这个</h4>
      <p>Kaplan 那篇 2020 年的 scaling-law 论文发现：compute 越多，loss 就越小，下降的速度大概是 <code>α ≈ 0.05</code>。把 loss 对 compute 画在 log-log 轴上，你会得到一条几乎笔直的线，它的 slope 就是这个数。上面说的 Chinchilla「每个 parameter 配 20 个 token」，用的正是这个办法。两年之后，DeepMind 把同样的 log-log 做法用在「compute 怎么花最值」这件事上。他们发现 token 和 parameter 应该一起长大，不能只把模型做大。就这一个 exponent —— 你马上要从一张 log-log 图上读出来的那个 slope —— 能预测 compute 每多 10 倍，模型会好多少。这就是有人愿意花几百万美元跑一次训练的理由。</p>
      </div>
      <div class="callout c-info"><span class="ic">🔗</span><div><span class="lang-en"><b>Remember this chain:</b> loss vs compute is a power law → on log-log it's a straight line → the slope is the scaling exponent → the exponent predicts the payoff of more compute. You'll <em>derive</em> these laws in the scaling modules; today you learned to read the axes.</span><span class="lang-zh"><b>记住这条链子：</b>loss 对 compute 是一条 power law → 在 log-log 上它是一条直线 → 这条线的 slope 就是 scaling exponent → 这个 exponent 能预测多花 compute 划不划算。这些定律你会在讲 scaling 的那些 module 里<em>自己推出来</em>。今天你学会的是怎么读这两根轴。</span></div></div>

      <div class="lang-en">
      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>Logs and exponents are also how we keep very small and very large numbers from breaking the arithmetic — and getting that wrong fails quietly.</p>
      </div>
      <div class="lang-zh">
      <h4>资深工程师的眼光 —— 一个悄悄发生的错，一个取舍</h4>
      <p>log 和 exponent 还有一个用处：让很小和很大的数字不要把算术弄坏。这件事做错了，程序不会喊。它只会安安静静地给你一个没用的答案。</p>
      </div>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><span class="lang-en"><b>Failure mode (silent): underflow to zero, then <code>log(0)</code> (a bug that doesn't announce itself).</b> A model's probability for one token might be <code>0.001</code>. Multiply a few hundred of those together — as you would to score a sentence — and the product falls below the smallest number the dtype can hold, so it rounds to <em>exactly</em> <code>0.0</code>. No error fires. Then <code>log(0.0)</code> returns <code>-inf</code>, and one <code>-inf</code> poisons the average into <code>nan</code>, which quietly makes the whole loss useless. The fix, which a senior engineer looks for in <b>code review</b>, is to add the logs instead of logging the product: <code>log(a) + log(b) + ...</code> stays in a safe range because it never forms the tiny product at all. Seeing a raw <code>np.log(product_of_probs)</code> is the red flag.</span><span class="lang-zh"><b>出错的样子（悄悄的）：数字 underflow 成 0，然后你对它取 <code>log(0)</code>。这个 bug 自己不会喊出来。</b>一个模型给某个 token 的概率，可能是 <code>0.001</code>。把几百个这样的数字乘起来 —— 给一句话打分就要这么做 —— 这个乘积会小过 dtype（数据类型）能装下的最小的数。于是它被<em>正好</em>算成 <code>0.0</code>。一个报错都没有。接着 <code>log(0.0)</code> 返回 <code>-inf</code>。一个 <code>-inf</code> 就把平均值毒成 <code>nan</code>，整个 loss 悄悄就没用了。修的办法是把 log 加起来，不要对乘积取 log：<code>log(a) + log(b) + ...</code> 一直待在安全的范围里，因为它根本不会算出那个很小的乘积。一个资深工程师在 <b>code review</b> 里找的就是这个。看到一个光秃秃的 <code>np.log(product_of_probs)</code>，那就是危险信号。</span></div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><span class="lang-en"><b>Trade-off: work in log-space for safety vs. remembering the numbers changed meaning.</b> Moving to logs turns underflow-prone multiplication into safe addition (<code>log(a·b) = log a + log b</code>) and squashes a huge range into a small one. What you pay: your values are no longer probabilities or counts — they are log-units, so you must combine them by adding (not multiplying) and exponentiate only at the very end to read a real number back. In a <b>design review</b> this is the choice to keep an entire pipeline in log-space: safer arithmetic, but every step must respect that the numbers are logs.</span><span class="lang-zh"><b>取舍：在 log-space 里算比较安全，但你得一直记着这些数字的意思变了。</b>换到 log 之后，容易 underflow 的乘法就变成安全的加法（<code>log(a·b) = log a + log b</code>），很大的范围也被压成很小的范围。你付出的代价是：你的数字不再是概率，也不再是个数。它们是 log 单位。所以你要用加法把它们合起来，不能用乘法。也只在最后一步取 exp，才能读回一个真实的数字。在 <b>design review</b> 里，这就是「整条流水线都留在 log-space」这个决定：算术更安全，但每一步都得记住这些数字是 log。</span></div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><span class="lang-en"><b>Say this in an interview:</b> "A scaling law is a power law — <code>loss ∝ compute^(−α)</code> — so on log-log axes it's a straight line and the slope is the exponent. That one number tells you how much loss you buy per 10× more compute, which is the case for spending millions on a run. The nuance: do the arithmetic in log-space — <em>add</em> log-probabilities instead of logging a product — or a long sequence quietly underflows to <code>log(0) = −inf</code>."</span><span class="lang-zh"><b>面试时可以这样说：</b>「一条 scaling law 就是一条 power law —— <code>loss ∝ compute^(−α)</code> —— 所以在 log-log 轴上它是一条直线，这条线的 slope 就是 exponent。这一个数字告诉你 compute 每多 10 倍能换回多少 loss，这就是花几百万美元跑一次训练的理由。细一点说：算术要在 log-space 里做。要把 log 概率<em>加</em>起来，不要对乘积取 log。不然一个长句子会悄悄 underflow 成 <code>log(0) = −inf</code>。」</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Got the derivation</span><span class="lang-zh">推导看懂了</span></button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h"><span class="lang-en">Leave today's evidence (strongly recommended)</span><span class="lang-zh">留下今天的证据（很建议做）</span></span><span class="sec-tag"><span class="lang-en">Produce</span><span class="lang-zh">动手做</span></span></div>
  <div class="sec-body">
      <p><span class="lang-en">Understanding ≠ being able to write it. Turn a power law into a line yourself — that's what makes the day count. Pick one:</span><span class="lang-zh">看懂 ≠ 写得出来。自己动手，把一条 power law 变成一条直线 —— 这一天才算真的过了。挑一个做：</span></p>
      <h4><span class="lang-en">Acceptance criteria</span><span class="lang-zh">做到这几条就算过了</span></h4>
      <ul>
        <li><span class="lang-en"><code>np.log(np.exp(3))</code> prints <code>3.0</code> — log and exp cancel.</span><span class="lang-zh"><code>np.log(np.exp(3))</code> 打印出 <code>3.0</code> —— log 和 exp 互相消掉了。</span></li>
        <li><span class="lang-en"><code>np.log10([1,10,100,1000])</code> prints <code>[0,1,2,3]</code> — each ×10 adds exactly 1.</span><span class="lang-zh"><code>np.log10([1,10,100,1000])</code> 打印出 <code>[0,1,2,3]</code> —— 每一次 ×10，正好让 log 加 1。</span></li>
        <li><span class="lang-en">For <code>x=[1,10,100]</code>, <code>y=x**2</code>: <code>np.log10(y)</code> rises by 2 for every 1 that <code>np.log10(x)</code> rises, and the computed slope equals the exponent <code>2</code>.</span><span class="lang-zh">取 <code>x=[1,10,100]</code>、<code>y=x**2</code>：<code>np.log10(x)</code> 每涨 1，<code>np.log10(y)</code> 就涨 2。而你算出来的 slope 等于 exponent <code>2</code>。</span></li>
        <li><span class="lang-en">Runs cleanly with <code>python3 sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code> and prints those three checks.</span><span class="lang-zh">用 <code>python3 sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code> 能干净地跑起来，并且把上面三项检查都打印出来。</span></li>
      </ul>
      <div class="lang-en">
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code>. Show that <code>np.log(np.exp(3))</code> is 3; print <code>np.log10([1,10,100,1000])</code>; then make <code>x = [1,10,100]</code>, <code>y = x**2</code>, and print <code>np.log10(x)</code> and <code>np.log10(y)</code> — check the log values rise by 2 for every 1 (slope 2). Run it with <code>python3 sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
      </div>
      <div class="lang-zh">
      <h4>做法 A · 自己写</h4>
      <p>新建 <code>sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code>。先证明 <code>np.log(np.exp(3))</code> 是 3。再打印 <code>np.log10([1,10,100,1000])</code>。然后做出 <code>x = [1,10,100]</code> 和 <code>y = x**2</code>，打印 <code>np.log10(x)</code> 和 <code>np.log10(y)</code> —— 检查 log 值是不是每涨 1 就跟着涨 2（slope 是 2）。用 <code>python3 sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py</code> 跑起来。</p>
      <h4>做法 B · 让 Claude 写，然后你读</h4>
      <p>把下面这段 prompt 复制回 Claude Code。它会让 Claude Code 帮你新建文件、写好代码，再跑一次。</p>
      </div>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l"><span class="lang-en">ask Claude to build it</span><span class="lang-zh">让 Claude 来写</span></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Help me build my Module 1 Day 5 artifact.

Create sessions/m01-shape-of-data/day-05-logs-and-exponents/experiment.py that, with a comment on each step:
1. Prints 2**3, np.exp(1), and np.log(np.exp(3)) to show exponent vs log and that they undo each other.
2. Prints np.log10([1,10,100,1000]) → [0,1,2,3], noting each ×10 adds 1 (orders of magnitude).
3. Builds x=np.array([1,10,100]); y=x**2; prints y, then np.log10(x) and np.log10(y); computes the slope (rise/run) of log10(y) vs log10(x) and asserts it equals 2 (the exponent).
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <div class="callout c-info"><span class="ic">📓</span><div><span class="lang-en"><b>5-minute research log:</b> before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-05-logs-and-exponents/log.md</code>: (1) in your own words, why does a power law become a straight line on log-log axes? (2) the number or output that surprised you; (3) one thing you're still unsure about.</span><span class="lang-zh"><b>5 分钟研究笔记：</b>关掉页面之前，用自己的话在 <code>sessions/m01-shape-of-data/day-05-logs-and-exponents/log.md</code> 里写三行：（1）为什么一条 power law 在 log-log 轴上会变成直线？（2）哪一个数字或者哪一段输出让你意外？（3）还有一件你不太确定的事。</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Done</span><span class="lang-zh">做完了</span></button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3><span class="lang-en">Module 1 · Day 5 complete! 🏆</span><span class="lang-zh">Module 1 · 第 5 天完成！🏆</span></h3>
      <p><span class="lang-en">You can now read a log-log scaling plot: a power law becomes a straight line, and its slope is the exponent that predicts the payoff of more compute.<br>Next up: <b>Random Seeds &amp; PRNG</b> — make "random" repeat on command.</span><span class="lang-zh">你现在能读懂一张 log-log 的 scaling 图了：一条 power law 变成一条直线，而它的 slope 就是那个 exponent，能预测多花 compute 划不划算。<br>下一站：<b>random seed 与 PRNG</b> —— 让「随机」听话地重复。</span></p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  explog:{html:'<span class="prompt">&gt;&gt;&gt;</span> <span class="num">2</span> ** <span class="num">3</span>            <span class="dim"><span class="lang-en"># exponent: 2 multiplied 3 times</span><span class="lang-zh"># exponent：2 乘 3 次</span></span>\n'+
    '<span class="hl">8</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.exp(<span class="num">1</span>)         <span class="dim"><span class="lang-en"># e to the power 1</span><span class="lang-zh"># e 的 1 次方</span></span>\n'+
    '<span class="ok">2.718281828459045</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.log(np.exp(<span class="num">3</span>))  <span class="dim"><span class="lang-en"># log undoes exp</span><span class="lang-zh"># log 把 exp 倒回去</span></span>\n'+
    '<span class="ok">3.0</span>',
    take:'<span class="lang-en"><b>①  Log and exp are inverses.</b> An exponent multiplies; a log asks "to what power?" So <code>log(exp(3))</code> gives back 3 — they cancel, like + and −.</span>'+
    '<span class="lang-zh"><b>①  log 和 exp 是一对反过来的动作。</b>exponent 是在乘；log 问的是「几次方？」。所以 <code>log(exp(3))</code> 又把 3 给回来了 —— 它们互相消掉，就像 + 和 − 一样。</span>'},
  oom:{html:'<span class="prompt">&gt;&gt;&gt;</span> np.log10([<span class="num">1</span>, <span class="num">10</span>, <span class="num">100</span>, <span class="num">1000</span>])\n'+
    '<span class="hl">array([0., 1., 2., 3.])</span>',
    take:'<span class="lang-en"><b>②  log10 counts orders of magnitude.</b> Each ×10 adds exactly 1 to the log. That is why 1 all the way to a billion can sit, evenly spaced, on a single axis.</span>'+
    '<span class="lang-zh"><b>②  log10 数的是 orders of magnitude。</b>每一次 ×10，正好让 log 加 1。所以从 1 一直到十亿，都能等距地摆在同一根轴上。</span>'},
  power:{html:'<span class="prompt">&gt;&gt;&gt;</span> x = np.array([<span class="num">1</span>, <span class="num">10</span>, <span class="num">100</span>])\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> y = x ** <span class="num">2</span>              <span class="dim"><span class="lang-en"># a power law: y = x^2</span><span class="lang-zh"># 一条 power law：y = x^2</span></span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> y\n<span class="ok">array([    1,   100, 10000])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.log10(x)\n<span class="hl">array([0., 1., 2.])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.log10(y)\n<span class="hl">array([0., 2., 4.])</span>',
    take:'<span class="lang-en"><b>③  The trick, in numbers.</b> In log space the points are (0,0), (1,2), (2,4) — a perfectly straight line with <b>slope 2</b>. The exponent (2) became the slope. That is how you read a scaling law.</span>'+
    '<span class="lang-zh"><b>③  用数字看这个招数。</b>在 log 空间里，这三个点是 (0,0)、(1,2)、(2,4) —— 一条完全笔直的线，<b>slope 是 2</b>。exponent（2）变成了 slope。读一条 scaling law，就是这样读的。</span>'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Exponent 2 to the 3 equals 8'><g font-family='monospace' font-size='15' text-anchor='middle'><text x='120' y='55' fill='#1F6280'>2³ = 2·2·2</text><text x='250' y='55' fill='#6B645E'>=</text><rect x='285' y='36' width='44' height='34' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='307' y='59' fill='#1a5c38' font-weight='bold'>8</text>"+
   '<text class="lang-en" x="260" y="92" fill="#6B645E" font-size="12">exponent = how many times you multiply</text>'+
   '<text class="lang-zh" x="260" y="92" fill="#6B645E" font-size="12">exponent = 乘几次</text>'+
   "</g></svg>",
  note:'<span class="lang-en">'+"<b>Start with an exponent.</b> <code>2³</code> means multiply 2 by itself three times → 8. The little number on top is just a count of multiplications."+'</span>'+
   '<span class="lang-zh"><b>先从 exponent 开始。</b><code>2³</code> 的意思是把 2 自己乘三次 → 8。上面那个小数字，只是在数你要乘几次。</span>'},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='Log undoes exponent: log2 of 8 is 3, log10 of 1000 is 3'><g font-family='monospace' font-size='15' text-anchor='middle'><text x='130' y='48' fill='#9A7208'>log₂(8) = 3</text><text x='130' y='76' fill='#9A7208'>log₁₀(1000) = 3</text><text x='300' y='62' fill='#6B645E'>←→</text>"+
   '<text class="lang-en" x="400" y="62" fill="#1F6280" font-size="12">the inverse of ^</text>'+
   '<text class="lang-zh" x="400" y="62" fill="#1F6280" font-size="12">^ 反过来</text>'+
   "</g></svg>",
  note:'<span class="lang-en">'+"<b>A log undoes an exponent.</b> It asks 'to what power?': <code>log₂(8)=3</code> because <code>2³=8</code>; <code>log₁₀(1000)=3</code> because <code>10³=1000</code>. Log and exponent are opposites."+'</span>'+
   '<span class="lang-zh"><b>log 把 exponent 倒回去。</b>它问的是「几次方？」：<code>log₂(8)=3</code>，因为 <code>2³=8</code>；<code>log₁₀(1000)=3</code>，因为 <code>10³=1000</code>。log 和 exponent 是一对相反的动作。</span>'},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='On a log axis, 1 10 100 1000 are evenly spaced'><g font-family='monospace' font-size='13' text-anchor='middle'><line x1='40' y1='60' x2='480' y2='60' stroke='#5A544E' stroke-width='1.5'/><g fill='#5E5191'><circle cx='70' cy='60' r='4'/><circle cx='200' cy='60' r='4'/><circle cx='330' cy='60' r='4'/><circle cx='460' cy='60' r='4'/></g><text x='70' y='82' fill='#2C2A28'>1</text><text x='200' y='82' fill='#2C2A28'>10</text><text x='330' y='82' fill='#2C2A28'>100</text><text x='460' y='82' fill='#2C2A28'>1000</text><text x='135' y='48' fill='#9A7208' font-size='11'>×10</text><text x='265' y='48' fill='#9A7208' font-size='11'>×10</text><text x='395' y='48' fill='#9A7208' font-size='11'>×10</text></g></svg>",
  note:'<span class="lang-en">'+"<b>A log axis spaces things by ×10.</b> 1, 10, 100, 1000 sit at equal distances — each gap is one order of magnitude. That is how a range from 1 to a billion fits on one screen."+'</span>'+
   '<span class="lang-zh"><b>log 轴是按 ×10 排开的。</b>1、10、100、1000 之间的距离一样宽 —— 每一格就是一个 order of magnitude。所以从 1 到十亿的范围，能放进一个屏幕里。</span>'},
 {viz:"<svg viewBox='0 0 520 140' role='img' aria-label='y = x squared drawn on normal axes is a rising curve'><g font-family='monospace'><line x1='60' y1='120' x2='60' y2='20' stroke='#5A544E' stroke-width='1.5'/><line x1='60' y1='120' x2='460' y2='120' stroke='#5A544E' stroke-width='1.5'/><path d='M60 120 Q 300 118 300 60 Q 300 25 440 25' fill='none' stroke='#C93B3B' stroke-width='2.5'/>"+
   '<text class="lang-en" x="250" y="138" text-anchor="middle" font-size="12" fill="#2C2A28">x  (normal axis)</text>'+
   '<text class="lang-zh" x="250" y="138" text-anchor="middle" font-size="12" fill="#2C2A28">x（普通轴）</text>'+
   '<text class="lang-en" x="250" y="45" text-anchor="middle" font-size="12" fill="#C93B3B">y = x²  →  a curve</text>'+
   '<text class="lang-zh" x="250" y="45" text-anchor="middle" font-size="12" fill="#C93B3B">y = x²  →  一条曲线</text>'+
   "</g></svg>",
  note:'<span class="lang-en">'+"<b>On normal axes, a power law bends.</b> <code>y = x²</code> shoots upward as a curve — hard to read, hard to compare, hard to fit a simple rule to."+'</span>'+
   '<span class="lang-zh"><b>在普通的轴上，power law 是弯的。</b><code>y = x²</code> 像一条曲线一样往上冲 —— 不好读，不好比，也不好用一条简单的规则去套它。</span>'},
 {viz:"<svg viewBox='0 0 520 150' role='img' aria-label='On log-log axes the same y = x squared is a straight line with slope 2'><g font-family='monospace'><line x1='60' y1='125' x2='60' y2='20' stroke='#5A544E' stroke-width='1.5'/><line x1='60' y1='125' x2='460' y2='125' stroke='#5A544E' stroke-width='1.5'/><line x1='80' y1='120' x2='420' y2='35' stroke='#2D8B55' stroke-width='2.5'/><circle cx='80' cy='120' r='4' fill='#2D8B55'/><circle cx='250' cy='77' r='4' fill='#2D8B55'/><circle cx='420' cy='35' r='4' fill='#2D8B55'/><text x='250' y='143' text-anchor='middle' font-size='12' fill='#2C2A28'>log x</text><text x='30' y='75' text-anchor='middle' font-size='12' fill='#2C2A28' transform='rotate(-90 30 75)'>log y</text>"+
   '<text class="lang-en" x="330" y="58" font-size="12" fill="#2D8B55">a straight line!</text>'+
   '<text class="lang-zh" x="330" y="58" font-size="12" fill="#2D8B55">一条直线！</text>'+
   "</g></svg>",
  note:'<span class="lang-en">'+"<b>On log-log axes, it straightens.</b> Take the log of both x and y and the same <code>y = x²</code> data falls on a perfectly straight line. This is the single most useful plot type in ML."+'</span>'+
   '<span class="lang-zh"><b>在 log-log 轴上，它变直了。</b>把 x 和 y 都取 log，同样的 <code>y = x²</code> 数据就落在一条完全笔直的线上。这是 ML 里最有用的一种图。</span>'},
 {viz:"<svg viewBox='0 0 520 160' role='img' aria-label='The slope of the log-log line equals the exponent; scaling law reads loss vs compute with slope alpha'><g font-family='monospace'><line x1='60' y1='120' x2='60' y2='25' stroke='#5A544E' stroke-width='1.5'/><line x1='60' y1='120' x2='440' y2='120' stroke='#5A544E' stroke-width='1.5'/><line x1='80' y1='40' x2='420' y2='110' stroke='#2A7B9B' stroke-width='2.5'/>"+
   '<text class="lang-en" x="250" y="60" font-size="12" fill="#2A7B9B">slope = the exponent</text>'+
   '<text class="lang-zh" x="250" y="60" font-size="12" fill="#2A7B9B">slope 就是 exponent</text>'+
   '<text class="lang-en" x="250" y="140" text-anchor="middle" font-size="11.5" fill="#2C2A28">log(compute) →</text>'+
   '<text class="lang-zh" x="250" y="140" text-anchor="middle" font-size="11.5" fill="#2C2A28">log(compute) →</text>'+
   "<text x='30' y='75' text-anchor='middle' font-size='11.5' fill='#2C2A28' transform='rotate(-90 30 75)'>log(loss)</text>"+
   '<text class="lang-en" x="250" y="156" text-anchor="middle" font-size="10.5" fill="#6B645E">a scaling law: loss falls as a power of compute</text>'+
   '<text class="lang-zh" x="250" y="156" text-anchor="middle" font-size="10.5" fill="#6B645E">scaling law：loss 按 compute 的次方往下掉</text>'+
   "</g></svg>",
  note:'<span class="lang-en">'+"<b>Why it runs the industry.</b> Scaling laws plot loss vs compute on log-log axes; the line's <b>slope is the exponent</b> that predicts how much loss drops per 10× more compute. Read that slope and you've read the chart every frontier lab bets millions on. That closes Module 1 — you now speak the language."+'</span>'+
   '<span class="lang-zh"><b>它为什么撑起整个行业。</b>scaling law 把 loss 对 compute 画在 log-log 轴上。这条线的 <b>slope 就是 exponent</b>。它能预测 compute 每多 10 倍，loss 会掉多少。读懂这个 slope，你就读懂了每个前沿实验室押上几百万美元的那张图。Module 1 到这里就结束了 —— 这门语言你现在会说了。</span>'}
];
@@@ region name=QS
var QS=[
 {q:'<span class="lang-en">1. A log undoes which operation?</span><span class="lang-zh">1. log 把哪一个运算倒回去？</span>',
  opts:['<span class="lang-en">Addition</span><span class="lang-zh">加法</span>',
        '<span class="lang-en">Exponentiation (raising to a power)</span><span class="lang-zh">乘方（也就是取几次方）</span>',
        '<span class="lang-en">Matrix multiply</span><span class="lang-zh">矩阵乘法</span>',
        '<span class="lang-en">Slicing</span><span class="lang-zh">slice（切片）</span>'],
  ans:1, fb:'<span class="lang-en">Right. log and exp are inverses: log(exp(x)) = x, just like + and − cancel.</span><span class="lang-zh">对了。log 和 exp 是一对反过来的动作：log(exp(x)) = x，就像 + 和 − 会互相消掉一样。</span>'},
 {q:'<span class="lang-en">2. On a log-log plot, a power law <code>y = a·x^b</code> looks like:</span><span class="lang-zh">2. 在一张 log-log 图上，一条 power law <code>y = a·x^b</code> 看起来像：</span>',
  opts:['<span class="lang-en">A curve that bends upward</span><span class="lang-zh">一条往上弯的曲线</span>',
        '<span class="lang-en">A straight line</span><span class="lang-zh">一条直线</span>',
        '<span class="lang-en">A circle</span><span class="lang-zh">一个圆</span>',
        '<span class="lang-en">A flat horizontal line</span><span class="lang-zh">一条平平的横线</span>'],
  ans:1, fb:'<span class="lang-en">Right. Taking logs turns y = a·x^b into log y = log a + b·log x — a straight line.</span><span class="lang-zh">对了。取 log 以后，y = a·x^b 就变成 log y = log a + b·log x —— 一条直线。</span>'},
 {q:'<span class="lang-en">3. For <code>y = a·x^b</code>, the slope of that log-log line equals:</span><span class="lang-zh">3. 对 <code>y = a·x^b</code> 来说，那条 log-log 直线的 slope 等于：</span>',
  opts:['<span class="lang-en">a</span><span class="lang-zh">a</span>',
        '<span class="lang-en">b (the exponent)</span><span class="lang-zh">b（就是那个 exponent）</span>',
        '<span class="lang-en">log a</span><span class="lang-zh">log a</span>',
        '<span class="lang-en">1</span><span class="lang-zh">1</span>'],
  ans:1, fb:'<span class="lang-en">Right. The slope is b — the exponent. The intercept is log a.</span><span class="lang-zh">对了。slope 就是 b，也就是 exponent。这条线和纵轴相交的高度是 log a。</span>'},
 {q:'<span class="lang-en">4. Your sentence-scoring code multiplies a few hundred per-token probabilities together and then takes <code>np.log</code> of the product. It runs with no error, but the score comes out as <code>-inf</code> or <code>nan</code> for long sentences only. Where do you look first, and why?</span><span class="lang-zh">4. 你给句子打分的代码，把每个 token 的概率几百个乘在一起，然后对这个乘积取 <code>np.log</code>。程序一个错都不报，但只有长句子的分数会变成 <code>-inf</code> 或者 <code>nan</code>。你先去看哪里？为什么？</span>',
  opts:['<span class="lang-en">The log function is broken and should be replaced</span><span class="lang-zh">log 这个函数坏了，应该换掉</span>',
        '<span class="lang-en">The product underflowed to exactly 0.0 (many small numbers multiplied), so <code>log(0.0)</code> is <code>-inf</code>; add the logs instead of logging the product</span><span class="lang-zh">很多小数字乘起来，乘积 underflow 成了正好 <code>0.0</code>，所以 <code>log(0.0)</code> 是 <code>-inf</code>；改成把 log 加起来，不要对乘积取 log</span>',
        '<span class="lang-en">The sentences are simply too rare, so nan is correct</span><span class="lang-zh">这些句子本来就太少见，所以 nan 是对的</span>',
        '<span class="lang-en">The exponent slope is negative, which always yields nan</span><span class="lang-zh">exponent 的 slope 是负数，这总会得出 nan</span>'],
  ans:1, fb:'<span class="lang-en">Right. Multiplying hundreds of values below 1 drives the product under the smallest representable number, so it rounds to <code>0.0</code> with no warning; <code>log(0.0) = -inf</code> then poisons the average into <code>nan</code>. Only long sentences hit it because they have more factors. The fix is to sum the logs — <code>log(a)+log(b)+...</code> — which never forms the tiny product.</span><span class="lang-zh">对了。几百个小于 1 的数字乘起来，乘积会小过能表示出来的最小的数。于是它一声不响地变成 <code>0.0</code>。接着 <code>log(0.0) = -inf</code> 把平均值毒成 <code>nan</code>。只有长句子会碰上，因为它要乘的数字更多。修的办法是把 log 加起来 —— <code>log(a)+log(b)+...</code> —— 这样根本不会算出那个很小的乘积。</span>'}
];
