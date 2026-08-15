---
quest_id: wf1-d06-seeds
donor: m01-day-06-random-seeds.donor
mode: exemplar
source_mode: verbatim   # migration mode — regions captured byte-for-byte from the shipped lesson
page_title: "Module 1 · Day 6 — Random Seeds &amp; PRNG"
spine: "shape"
notebook_yardstick: null   # no matching fundamentals notebook (Notebook Smoothness Gate = N/A)
---

@@@ region name=title
<title>Module 1 · Day 6 — Random Seeds &amp; PRNG</title>
@@@ region name=brand_sub
<div class="brand-sub"><span class="lang-en">Foundations · M1 Day 6</span><span class="lang-zh">基础 · M1 第 6 天</span></div>
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
<a class="lnav prev" href="../day-05-logs-and-exponents/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t"><span class="lang-en">Logs &amp; Exponents</span><span class="lang-zh">log 与 exponent</span></span></a>
@@@ region name=nav_next
<a class="lnav next" href="../review.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t"><span class="lang-en">Review Gate</span><span class="lang-zh">复习关卡</span></span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker"><span class="lang-en">Module 1 · Represent · Day 6</span><span class="lang-zh">Module 1 · 表示 · 第 6 天</span></span>
      <h1><span class="lang-en">Random Seeds &amp; PRNG</span><span class="lang-zh">random seed 与 PRNG</span><span class="sub"><span class="lang-en">Make "Random" Repeat on Command</span><span class="lang-zh">让「随机」听话地重复</span></span></h1>
      <p class="lede"><span class="lang-en">Computers cannot make true randomness. They fake it with a recipe. Feed that recipe a starting number — a <strong>seed</strong> — and the "random" numbers come out the same every single time. This tiny trick is what lets you debug a flaky model, and lets other people rerun your result and get the same answer. That is what turns a lucky run into real science.</span><span class="lang-zh">电脑做不出真正的随机。它是拿一份配方假装出来的。你给这份配方一个起始数字 —— 也就是 <strong>seed（随机种子）</strong> —— 那些「随机」数字每次跑出来都会完全一样。这个小招数很有用。它让你能修一个时好时坏的模型。它也让别人重跑你的代码时，能拿到一样的答案。一次靠运气的结果，就这样变成了真正的科学。</span></p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><span class="lang-en"><b>By the end, you'll be able to:</b> say why reproducibility matters, seed NumPy with <code>np.random.seed(0)</code> and the modern <code>np.random.default_rng(0)</code>, and explain what a PRNG really is.</span><span class="lang-zh"><b>学完这一天，你能做到：</b>说清楚 reproducibility（可复现）为什么重要。用 <code>np.random.seed(0)</code> 给 NumPy 设 seed，也会用新写法 <code>np.random.default_rng(0)</code>。还能讲清楚 PRNG（伪随机数生成器）到底是什么。</span></div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h"><span class="lang-en">Seeds, PRNGs, and reproducibility</span><span class="lang-zh">seed、PRNG 和可复现</span></span><span class="sec-tag"><span class="lang-en">What is it</span><span class="lang-zh">这是什么</span></span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><span class="lang-en"><b>What this assumes:</b> just Days 1–5 and how to call a NumPy function. <b>No probability math.</b> This is the last skill of Module 1 — how to make "random" behave.</span><span class="lang-zh"><b>你需要先会：</b>第 1 到第 5 天的内容，还有怎么调用一个 NumPy 函数。<b>不需要概率数学。</b>这是 Module 1 的最后一个本领 —— 怎么让「随机」听话。</span></div></div>

      <div class="lang-en">
      <h4>What is a seed?</h4>
      <p>A <span class="term" data-tip="A starting number you give a random-number generator. The same seed always produces the same sequence of numbers.">seed</span> is one starting number you hand to the random-number maker. Give it the same seed and it always spits out the same "random" numbers, in the same order. Change the seed and you get a whole new sequence.</p>

      <h4>What is a PRNG?</h4>
      <p>A <span class="term" data-tip="Pseudo-Random Number Generator: a fixed recipe that turns a seed into a long sequence of numbers that only look random. Fully decided by the seed.">PRNG</span> is short for <strong>pseudo-random number generator</strong>. "Pseudo" means "fake." It is a fixed recipe: put a seed in, get a long stream of numbers out. The numbers look scattered and unpredictable, but they are not truly random — they are fully decided by the seed.</p>

      <h4>What is reproducibility?</h4>
      <p><span class="term" data-tip="The ability to run the same code and get exactly the same result every time. Others can rerun your experiment and confirm it.">Reproducibility</span> means your program gives the same result every time you run it. It matters for two reasons. First, <strong>debugging</strong>: if a bug only shows up sometimes, you cannot fix it — but a fixed seed makes it show up every time. Second, <strong>science</strong>: other people can rerun your code and check that your result was real, not luck. Keep going 👇</p>
      </div>
      <div class="lang-zh">
      <h4>seed 是什么？</h4>
      <p><span class="term" data-tip="你交给随机数机器的一个起始数字。同一个 seed 永远产生同一串数字，顺序也一样。">seed</span> 就是你交给「随机数机器」的一个起始数字。给它同一个 seed，它吐出来的「随机」数字每次都一样，顺序也一样。换一个 seed，你就得到一串全新的数字。</p>

      <h4>PRNG 是什么？</h4>
      <p><span class="term" data-tip="Pseudo-Random Number Generator，伪随机数生成器。一份固定的配方，把一个 seed 变成一长串看起来随机的数字。这串数字完全由 seed 决定。">PRNG</span> 是 <strong>pseudo-random number generator</strong> 的缩写。「pseudo」的意思就是「假」。它是一份固定的配方：放一个 seed 进去，出来一长串数字。这些数字看起来很散，你也猜不到下一个是什么。但它们不是真的随机 —— 它们完全由 seed 决定。</p>

      <h4>可复现是什么？</h4>
      <p><span class="term" data-tip="同一段代码每次跑出来的结果完全一样。别人也能重跑你的实验，确认你的结果是真的。">Reproducibility</span> 的意思是：你的程序每次跑出来的结果都一样。它重要有两个原因。第一是 <strong>debugging（调试）</strong>：一个 bug 如果只是偶尔出现，你根本没法修。把 seed 固定住，它就每次都出现。第二是 <strong>科学</strong>：别人可以重跑你的代码，检查你的结果是真的，不是碰巧。继续往下看 👇</p>
      </div>
      <div class="callout c-warn"><span class="ic">😕</span><div><span class="lang-en"><b>Why this trips people up:</b> the word "random" makes people think it's uncontrollable, when a seed makes it perfectly repeatable. In plain terms: people either forget to seed (and can't reproduce a bug) or over-share one seed (and silently kill the randomness they wanted) — the failure is invisible either way.</span><span class="lang-zh"><b>为什么这里容易卡住：</b>「随机」这个词让人以为它管不住。其实 seed 让它 100% 可以重复。说白了，出错的样子有两种。一种是忘了设 seed，于是一个 bug 再也复现不出来。另一种是所有地方共用一个 seed，于是你想要的随机悄悄没了。两种都不报错，你看不见。</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Got the three words — let's go</span><span class="lang-zh">三个词都记住了 —— 走吧</span></button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h"><span class="lang-en">A card shuffle with a written recipe</span><span class="lang-zh">照着步骤单洗牌</span></span><span class="sec-tag"><span class="lang-en">Intuition</span><span class="lang-zh">直觉</span></span></div>
  <div class="sec-body">
      <div class="relate">
        <div class="card"><span class="big">🃏</span><h5><span class="lang-en">The seed is the recipe</span><span class="lang-zh">seed 就是那张步骤单</span></h5><p><span class="lang-en">Imagine a shuffle you follow from written steps: "cut at card 7, then card 3, then card 9…". Two people who start from a fresh deck and follow the <strong>same</strong> written steps end up with the <strong>exact same</strong> shuffled order.</span><span class="lang-zh">想一想这样洗牌：你照着一张写好的步骤单做，上面写「在第 7 张切一次，再第 3 张，再第 9 张……」。两个人都从一副没动过的新牌开始，都照<strong>同一张</strong>步骤单做。最后他们手里的牌，顺序<strong>完全一样</strong>。</span></p></div>
        <div class="card"><span class="big">🔁</span><h5><span class="lang-en">Same seed = same shuffle</span><span class="lang-zh">同一个 seed = 同一种洗法</span></h5><p><span class="lang-en">The seed is that written recipe. Give the PRNG seed 0 and you always get shuffle-0. Give it seed 1 and you get a totally different shuffle-1. Every run repeats perfectly.</span><span class="lang-zh">seed 就是那张步骤单。给 PRNG 的 seed 是 0，你每次都得到 0 号洗法。给它 seed 1，你得到完全不同的 1 号洗法。每次跑，都重复得一点不差。</span></p></div>
      </div>
      <p><span class="lang-en"><strong>In one line:</strong> the numbers <em>look</em> shuffled and surprising, but the seed is a recipe that fixes the whole order in advance.</span><span class="lang-zh"><strong>一句话：</strong>这些数字<em>看起来</em>是被洗乱的，让人猜不到。但 seed 是一张步骤单，它提前把整个顺序定死了。</span></p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><span class="lang-en">Where it breaks: a real card shuffle is truly random — you can never repeat it. A PRNG is <strong>not</strong> truly random. It is only unpredictable enough to look random, while staying 100% repeatable from the seed.</span><span class="lang-zh">这个比喻不对的地方：真的用手洗牌是真正的随机 —— 你再也洗不出一样的顺序。PRNG <strong>不是</strong>真正的随机。它只是乱得让你猜不到，看起来像随机。而从同一个 seed 出发，它 100% 可以重复。</span></div></div>
      <p><span class="lang-en"><b>Intuition:</b> the seed isn't the source of randomness, it's the written recipe that fixes the whole order in advance. The technical point: a PRNG is a deterministic chain <code>state → number → new state</code>, and the seed is just the first state.</span><span class="lang-zh"><b>直觉：</b>seed 不是「随机的来源」，而是「那份写好的洗牌步骤」。技术上说：PRNG 是一条固定的链条 <code>state → number → new state</code>，seed 就是链条上的第一个 state（状态）。</span></p>
      <button class="gotit" type="button"><span class="lang-en">Got the picture</span><span class="lang-zh">画面看懂了</span></button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h"><span class="lang-en">How the recipe works — and why it runs the industry</span><span class="lang-zh">配方怎么转 —— 还有它为什么撑起整个行业</span></span><span class="sec-tag"><span class="lang-en">Mechanism &amp; why</span><span class="lang-zh">原理和为什么</span></span></div>
  <div class="sec-body">
      <div class="lang-en">
      <h4>The idea of a PRNG, in plain words</h4>
      <p>A PRNG keeps a hidden <strong>state</strong> — one big number in memory. Each time you ask for a random value, it scrambles that state with a fixed formula, hands you a piece of the result, and saves the new state for next time. So the stream is a chain: state → number → new state → next number, and on and on.</p>
      </div>
      <div class="lang-zh">
      <h4>用人话讲 PRNG 的想法</h4>
      <p>PRNG 心里藏着一个 <strong>state</strong> —— 内存里的一个大数字。你每要一个随机值，它就用一个固定公式把这个 state 搅一遍。它把结果里的一小块交给你。它再把新的 state 存起来，下次用。所以这串数字是一条链：state → number → new state → 下一个 number，一直往下。</p>
      </div>
      <div class="callout c-ok"><span class="ic">🧮</span><div><span class="lang-en"><b>The recipe in one line:</b><br><code>next_state = f(state)</code>, &nbsp; <code>output = g(state)</code><br><span style="font-size:.85rem;color:var(--text2)">Here <code>state</code> is the hidden number in memory, <code>f</code> is a fixed scrambling formula that makes the next state, and <code>g</code> reads a value out of the state. The <strong>seed</strong> is just the very first <code>state</code>. Same seed → same first state → the whole chain repeats exactly.</span></span><span class="lang-zh"><b>一行写完这份配方：</b><br><code>next_state = f(state)</code>, &nbsp; <code>output = g(state)</code><br><span style="font-size:.85rem;color:var(--text2)">这里 <code>state</code> 是内存里那个藏起来的数字。<code>f</code> 是一个固定的搅拌公式，它做出下一个 state。<code>g</code> 从 state 里读出一个值。<strong>seed</strong> 就是最开头那个 <code>state</code>。同一个 seed → 同一个开头 state → 整条链会完全一样地重来。</span></span></div></div>
      <p><span class="lang-en"><strong>Worked example (small numbers).</strong> Say <code>f(s) = (5·s + 3) mod 16</code> and we start with seed <code>state = 1</code>. Then the states go <code>1 → (5·1+3) mod 16 = 8 → (5·8+3) mod 16 = 11 → (5·11+3) mod 16 = 10</code>. The stream <code>1, 8, 11, 10, …</code> looks jumpy and hard to guess, but start again from seed 1 and you get the very same list. Real PRNGs use much bigger, better formulas — but this is the whole idea.</span><span class="lang-zh"><strong>动手算一遍（用小数字）。</strong>就说 <code>f(s) = (5·s + 3) mod 16</code>，我们的 seed 是 <code>state = 1</code>。那么 state 会这样走：<code>1 → (5·1+3) mod 16 = 8 → (5·8+3) mod 16 = 11 → (5·11+3) mod 16 = 10</code>。这串 <code>1, 8, 11, 10, …</code> 跳来跳去，很难猜。但你从 seed 1 再来一次，拿到的还是这一串。真正的 PRNG 用的公式大得多，也好得多。不过想法就是这一个。</span></p>

      <div class="lang-en">
      <h4>Two ways to seed NumPy</h4>
      <p>The old way sets one shared generator for the whole program: <code>np.random.seed(0)</code>, then calls like <code>np.random.rand()</code> draw from it. The modern, recommended way makes your own generator object: <code>rng = np.random.default_rng(0)</code>, then <code>rng.random()</code>. The modern way is safer because two parts of your code can hold two separate generators and never step on each other.</p>

      <h4>A bridge to Module 8 (just so you've seen it)</h4>
      </div>
      <div class="lang-zh">
      <h4>给 NumPy 设 seed 的两种写法</h4>
      <p>老写法给整个程序设一个共用的生成器：<code>np.random.seed(0)</code>，之后像 <code>np.random.rand()</code> 这样的调用都从它取数。新写法是现在推荐的：你自己做一个生成器对象，<code>rng = np.random.default_rng(0)</code>，然后用 <code>rng.random()</code>。新写法更安全。因为你代码里的两个地方可以各拿一个生成器，互相不会踩到对方。</p>

      <h4>先搭一座通向 Module 8 的桥（只是让你先见一眼）</h4>
      </div>
      <div class="callout c-info"><span class="ic">🔗</span><div><span class="lang-en"><b>Awareness only — no code to run today.</b> In <span class="term" data-tip="A numerical computing library used for large ML models. It handles randomness by passing an explicit key you split yourself, instead of a hidden global state.">JAX</span> (Module 8) there is no hidden global state. You pass an explicit <span class="term" data-tip="In JAX, the piece of randomness you carry by hand. You split one key into two to make fresh, independent randomness. Same idea as a NumPy seed, made visible.">key</span> into every random call, and you <strong>split</strong> that key to make fresh randomness. Same idea as a seed — just made visible and explicit so it works across many chips at once. You do not need JAX today; just know the seed idea grows up into an explicit key.</span><span class="lang-zh"><b>只是让你知道 —— 今天不用跑任何代码。</b>在 <span class="term" data-tip="一个做数值计算的库，常用来训练很大的 ML 模型。它不用藏起来的全局状态，而是让你自己传一个 key，并且自己 split 它。">JAX</span>（Module 8）里没有藏起来的全局状态。你每次调用随机函数，都要自己传一个 <span class="term" data-tip="在 JAX 里，随机性是你自己拿在手上的一小块。把一个 key split 成两个，就得到新的随机性，而且两边互相没有关系。和 NumPy 的 seed 是同一个想法，只是摆到了明面上。">key（随机数钥匙）</span>。你还要 <strong>split</strong> 这个 key，才能拿到新的随机性。想法和 seed 一样 —— 只是摆到明面上，这样它才能同时在很多块芯片上用。今天你不需要 JAX。你只要知道 seed 这个想法长大以后，就变成了一个要自己传的 key。</span></div></div>

      <div class="lang-en">
      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>A seed is a promise that the same recipe repeats. That promise becomes a trap when many workers get the <em>same</em> seed — the repetition is now happening where you did not want it.</p>
      </div>
      <div class="lang-zh">
      <h4>资深工程师的眼光 —— 一个悄悄发生的错，一个取舍</h4>
      <p>seed 是一个承诺：同一份配方会重复。可是当很多个 worker（工作进程）都拿到<em>同一个</em> seed，这个承诺就变成陷阱。重复跑到了你不想要它重复的地方。</p>
      </div>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><span class="lang-en"><b>Failure mode (silent): every worker shares one seed, so their "random" streams are identical (a bug that doesn't announce itself).</b> Training often splits data loading across several parallel workers, each meant to draw fresh random augmentations or a fresh shuffle. If each worker starts from the <em>same</em> fixed seed, they all produce the <strong>exact same</strong> stream — so the "random" crops or noise repeat across workers instead of varying. Nothing errors; results are perfectly reproducible; the data just secretly has far less variety than you think, and the model trains worse for no visible reason. A senior engineer catches this in <b>code review</b> by asking "is the seed offset by the worker id (and the epoch)?" — a single global seed shared by all workers is the tell.</span><span class="lang-zh"><b>出错的样子（悄悄的）：每个 worker 共用一个 seed，于是它们的「随机」数列完全一样，而这个 bug 自己不会喊出来。</b>训练时常常把读数据这件事分给好几个并行的 worker。每个 worker 本来该抽出新的随机裁剪，或者新的打乱顺序。可是如果每个 worker 都从<em>同一个</em>固定 seed 出发，它们产生的数列<strong>完全相同</strong>。那些「随机」的裁剪和噪声就在 worker 之间重复，而不是每个都不一样。程序不报错。结果还完全可复现。只是你的数据悄悄比你以为的花样少得多。模型也就训得更差，可你看不出原因。一个资深工程师在 <b>code review</b> 里会问一句：「seed 有没有按 worker id（还有 epoch）错开？」所有 worker 共用一个全局 seed，就是那个破绽。</span></div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><span class="lang-en"><b>Trade-off: reproducibility vs. seeing the true variance.</b> Fixing one seed makes a run repeatable, which is priceless for debugging and for others rerunning your result. What you pay: a single seed shows you one draw of a random process, so a lucky (or unlucky) seed can make a change look better or worse than it really is. In a <b>design review</b> the rule is to fix the seed while debugging, but to report a result across <em>several</em> seeds — the spread across seeds is the honest error bar, and a claim backed by one seed is not yet evidence.</span><span class="lang-zh"><b>取舍：要可复现，还是要看到真实的波动。</b>固定一个 seed 让一次运行可以重复。这对 debug 很值钱，对别人重跑你的结果也很值钱。你付出的代价是：一个 seed 只让你看到这个随机过程的一次抽样。所以一个运气好（或者运气差）的 seed，会让一个改动看起来比实际更好，或者更差。在 <b>design review</b> 里，规矩是这样：debug 的时候固定 seed，但报结果要跑<em>好几个</em> seed。几个 seed 之间的差距，才是诚实的误差范围。只用一个 seed 撑起来的说法，还不算证据。</span></div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><span class="lang-en"><b>Say this in an interview:</b> "A PRNG is fully deterministic — <code>next_state = f(state)</code>, <code>output = g(state)</code>, and the seed is just the first state — so the same seed replays the exact stream, which is what makes runs reproducible. The nuance is two failure modes: no seed means a heisenbug you can't reproduce, and one shared seed across data-loader workers means their 'random' augmentations are identical, so I offset the seed by worker id and epoch and report results across several seeds."</span><span class="lang-zh"><b>面试时可以这样说：</b>「PRNG 是完全确定的 —— <code>next_state = f(state)</code>，<code>output = g(state)</code>，seed 就是第一个 state。所以同一个 seed 会把同一串数字重放一遍，这就是运行可复现的原因。细一点说，出错的样子有两种。一是不设 seed，你会遇到一个复现不出来的 heisenbug。二是所有 data-loader 的 worker 共用一个 seed，它们的『随机』增强就完全一样。所以我会按 worker id 和 epoch 把 seed 错开，报结果时也跑好几个 seed。」</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Got the mechanism</span><span class="lang-zh">原理看懂了</span></button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h"><span class="lang-en">Leave today's evidence (strongly recommended)</span><span class="lang-zh">留下今天的证据（很建议做）</span></span><span class="sec-tag"><span class="lang-en">Produce</span><span class="lang-zh">动手做</span></span></div>
  <div class="sec-body">
      <p><span class="lang-en">Understanding ≠ being able to write it. Make "random" repeat on command yourself — that's what makes the day count. Pick one:</span><span class="lang-zh">看懂 ≠ 写得出来。自己动手，让「随机」听话地重复一次 —— 这一天才算真的过了。挑一个做：</span></p>
      <div class="lang-en">
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m01-shape-of-data/day-06-random-seeds/experiment.py</code>. Draw two numbers with <code>np.random.rand()</code> and no seed (they differ). Then call <code>np.random.seed(0)</code>, draw an array, reseed with <code>np.random.seed(0)</code> again, draw again, and <code>assert</code> the two arrays are equal. Last, make <code>np.random.default_rng(0)</code> and <code>np.random.default_rng(1)</code> and show their first draws differ. Run it with <code>python3 sessions/m01-shape-of-data/day-06-random-seeds/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
      </div>
      <div class="lang-zh">
      <h4>做法 A · 自己写</h4>
      <p>新建 <code>sessions/m01-shape-of-data/day-06-random-seeds/experiment.py</code>。先不设 seed，用 <code>np.random.rand()</code> 抽两个数字（它们不一样）。然后调用 <code>np.random.seed(0)</code>，抽一个 array（数组）。再用 <code>np.random.seed(0)</code> 重设一次，再抽一次，并用 <code>assert</code> 确认两个 array 相等。最后做出 <code>np.random.default_rng(0)</code> 和 <code>np.random.default_rng(1)</code>，让它们的第一次抽样不一样。用 <code>python3 sessions/m01-shape-of-data/day-06-random-seeds/experiment.py</code> 跑起来。</p>
      <h4>做法 B · 让 Claude 写，然后你读</h4>
      <p>把下面这段 prompt 复制回 Claude Code。它会让 Claude Code 帮你新建文件、写好代码，再跑一次。</p>
      </div>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l"><span class="lang-en">ask Claude to build it</span><span class="lang-zh">让 Claude 来写</span></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Help me build my Module 1 Day 6 artifact.

Create sessions/m01-shape-of-data/day-06-random-seeds/experiment.py that, with a comment on each step:
1. Draws two values with np.random.rand() and NO seed, prints both, and notes they differ run to run.
2. Calls np.random.seed(0), draws a = np.random.rand(3); then calls np.random.seed(0) again, draws b = np.random.rand(3); prints a and b and asserts np.array_equal(a, b) — same seed reproduces the exact stream.
3. Makes rng0 = np.random.default_rng(0) and rng1 = np.random.default_rng(1); draws s0 = rng0.random(3) and s1 = rng1.random(3); prints both and asserts NOT np.array_equal(s0, s1) — different seeds give different reproducible streams.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4><span class="lang-en">Acceptance criteria</span><span class="lang-zh">做到这几条就算过了</span></h4>
      <ul>
        <li><span class="lang-en">Two unseeded draws differ; two draws after the <em>same</em> <code>np.random.seed(0)</code> are equal (you <code>assert</code> it).</span><span class="lang-zh">不设 seed 的两次抽样不一样。<em>同一个</em> <code>np.random.seed(0)</code> 之后的两次抽样相等，而且你用 <code>assert</code> 检查了。</span></li>
        <li><span class="lang-en"><code>default_rng(0)</code> and <code>default_rng(1)</code> give different but each-reproducible streams.</span><span class="lang-zh"><code>default_rng(0)</code> 和 <code>default_rng(1)</code> 给出不同的数列，但每一条都能重复出来。</span></li>
        <li><span class="lang-en">You can explain why the same seed always yields the same numbers, in one sentence, without saying "it's random".</span><span class="lang-zh">你能用一句话说清楚，为什么同一个 seed 总是给出同样的数字，而且不用「它是随机的」这句话。</span></li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><span class="lang-en"><b>5-minute research log:</b> before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-06-random-seeds/log.md</code>: (1) in your own words, why the same seed always gives the same "random" numbers; (2) the one place a shared seed would bite you; (3) one thing you're still unsure about.</span><span class="lang-zh"><b>5 分钟研究笔记：</b>关掉页面之前，用自己的话在 <code>sessions/m01-shape-of-data/day-06-random-seeds/log.md</code> 里写三行：（1）为什么同一个 seed 总是给出同样的「随机」数字；（2）共用一个 seed 会在哪一个地方咬你一口；（3）还有一件你不太确定的事。</span></div></div>
      <button class="gotit" type="button"><span class="lang-en">Done</span><span class="lang-zh">做完了</span></button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3><span class="lang-en">Module 1 · Day 6 complete! 🏆</span><span class="lang-zh">Module 1 · 第 6 天完成！🏆</span></h3>
      <p><span class="lang-en">Nice work — you've completed <b>Random Seeds &amp; PRNG</b>.<br>Next up: <b>Review Gate</b>.</span><span class="lang-zh">干得好 —— 你学完了 <b>random seed 与 PRNG</b>。<br>下一站：<b>复习关卡</b>。</span></p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  noseed:{html:'<span class="prompt">&gt;&gt;&gt;</span> np.random.rand()   <span class="dim"><span class="lang-en"># no seed set — first draw</span><span class="lang-zh"># 没设 seed —— 第一次抽</span></span>\n'+
    '<span class="hl">0.417022004702574</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.random.rand()   <span class="dim"><span class="lang-en"># draw again</span><span class="lang-zh"># 再抽一次</span></span>\n'+
    '<span class="hl">0.7203244934421581</span>\n'+
    '<span class="dim"><span class="lang-en"># run the program again tomorrow → two DIFFERENT numbers</span><span class="lang-zh"># 明天再跑一次 → 两个不一样的数字</span></span>',
    take:'<span class="lang-en"><b>①  No seed = no repeat.</b> Each call moves the hidden state forward, so you get a new number. Restart the program and the whole run looks different. Hard to debug, hard to trust.</span>'+
    '<span class="lang-zh"><b>①  不设 seed = 不会重复。</b>每次调用都把藏起来的 state 往前推一格，所以你拿到一个新数字。重启程序，整次运行看起来都不一样。这很难 debug，也很难让人相信。</span>'},
  seeded:{html:'<span class="prompt">&gt;&gt;&gt;</span> np.random.seed(<span class="num">0</span>)     <span class="dim"><span class="lang-en"># set the starting state</span><span class="lang-zh"># 设好开头的 state</span></span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.random.rand(<span class="num">3</span>)\n'+
    '<span class="ok">array([0.5488, 0.7152, 0.6028])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.random.seed(<span class="num">0</span>)     <span class="dim"><span class="lang-en"># reset to the same seed</span><span class="lang-zh"># 用同一个 seed 重设一次</span></span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.random.rand(<span class="num">3</span>)\n'+
    '<span class="ok">array([0.5488, 0.7152, 0.6028])</span>   <span class="dim"><span class="lang-en"># identical!</span><span class="lang-zh"># 一样！</span></span>',
    take:'<span class="lang-en"><b>②  Same seed = same stream.</b> Seed 0 sets the exact starting state, so both draws match to the last digit. This is what makes a run reproducible — every time, forever.</span>'+
    '<span class="lang-zh"><b>②  同一个 seed = 同一串数字。</b>seed 0 把开头的 state 定得死死的，所以两次抽样连最后一位都一样。这就是一次运行能可复现的原因 —— 每一次都这样，永远都这样。</span>'},
  streams:{html:'<span class="prompt">&gt;&gt;&gt;</span> rng0 = np.random.default_rng(<span class="num">0</span>)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> rng1 = np.random.default_rng(<span class="num">1</span>)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> rng0.random(<span class="num">3</span>)\n'+
    '<span class="hl">array([0.6370, 0.2698, 0.0409])</span>   <span class="dim"><span class="lang-en"># seed 0 stream</span><span class="lang-zh"># seed 0 的数列</span></span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> rng1.random(<span class="num">3</span>)\n'+
    '<span class="hl">array([0.5118, 0.9505, 0.1442])</span>   <span class="dim"><span class="lang-en"># seed 1 stream — different</span><span class="lang-zh"># seed 1 的数列 —— 不一样</span></span>',
    take:'<span class="lang-en"><b>③  Different seeds = different, but each is repeatable.</b> <code>default_rng(0)</code> and <code>default_rng(1)</code> are two separate generators. Each gives the same stream every run — but the two streams differ from each other.</span>'+
    '<span class="lang-zh"><b>③  不同的 seed = 不同的数列，但每一条都能重复。</b><code>default_rng(0)</code> 和 <code>default_rng(1)</code> 是两个各自独立的生成器。每一个每次跑都给出同一串数字。但这两串数字彼此不一样。</span>'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:'<svg viewBox="0 0 520 100" role="img" aria-label="A single seed number, 0"><g font-family="monospace" text-anchor="middle"><rect x="210" y="30" width="100" height="42" rx="8" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text class="lang-en" x="260" y="50" fill="#1F6280" font-size="12">seed</text><text class="lang-zh" x="260" y="50" fill="#1F6280" font-size="12">seed</text><text x="260" y="66" fill="#1F6280" font-size="16" font-weight="bold">0</text><text class="lang-en" x="260" y="92" fill="#6B645E" font-size="11">one starting number, chosen by you</text><text class="lang-zh" x="260" y="92" fill="#6B645E" font-size="11">你自己挑的一个起始数字</text></g></svg>',
  note:'<span class="lang-en"><b>Start with a seed.</b> A seed is just one number you pick, like <code>0</code>. On its own it does nothing — it is the starting point for everything that follows.</span><span class="lang-zh"><b>先有一个 seed。</b>seed 就是你挑的一个数字，比如 <code>0</code>。它自己什么也不做 —— 它只是后面所有事情的起点。</span>'},
 {viz:'<svg viewBox="0 0 520 110" role="img" aria-label="The seed goes into a PRNG box and a stream of numbers comes out"><g font-family="monospace" text-anchor="middle"><rect x="20" y="40" width="70" height="36" rx="7" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text class="lang-en" x="55" y="63" fill="#1F6280" font-size="13">seed 0</text><text class="lang-zh" x="55" y="63" fill="#1F6280" font-size="13">seed 0</text><text x="110" y="62" fill="#6B645E" font-size="16">→</text><rect x="130" y="34" width="120" height="48" rx="8" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="190" y="53" fill="#5E5191" font-size="12">PRNG</text><text class="lang-zh" x="190" y="53" fill="#5E5191" font-size="12">PRNG</text><text class="lang-en" x="190" y="70" fill="#5E5191" font-size="10">fixed recipe f, g</text><text class="lang-zh" x="190" y="70" fill="#5E5191" font-size="10">固定配方 f、g</text><text x="268" y="62" fill="#6B645E" font-size="16">→</text><g fill="#9A7208"><rect x="290" y="45" width="36" height="28" rx="5" fill="#FCF3DC" stroke="#C99A12"/><rect x="334" y="45" width="36" height="28" rx="5" fill="#FCF3DC" stroke="#C99A12"/><rect x="378" y="45" width="36" height="28" rx="5" fill="#FCF3DC" stroke="#C99A12"/><rect x="422" y="45" width="36" height="28" rx="5" fill="#FCF3DC" stroke="#C99A12"/></g><g fill="#9A7208" font-size="10"><text x="308" y="63">.54</text><text x="352" y="63">.72</text><text x="396" y="63">.60</text><text x="440" y="63">…</text></g></g></svg>',
  note:'<span class="lang-en"><b>The PRNG turns the seed into a stream.</b> The seed feeds a fixed recipe. Each step scrambles a hidden state and hands you one number, then saves the new state. Out comes a long line of numbers that <em>look</em> random.</span><span class="lang-zh"><b>PRNG 把 seed 变成一串数字。</b>seed 喂给一份固定的配方。每一步都把藏起来的 state 搅一下，交给你一个数字，再把新的 state 存好。出来的就是一长排<em>看起来</em>随机的数字。</span>'},
 {viz:'<svg viewBox="0 0 520 130" role="img" aria-label="Seed 0 twice gives two identical streams"><g font-family="monospace" font-size="11"><rect x="20" y="18" width="64" height="30" rx="6" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text class="lang-en" x="52" y="37" fill="#1F6280" text-anchor="middle">seed 0</text><text class="lang-zh" x="52" y="37" fill="#1F6280" text-anchor="middle">seed 0</text><rect x="20" y="80" width="64" height="30" rx="6" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text class="lang-en" x="52" y="99" fill="#1F6280" text-anchor="middle">seed 0</text><text class="lang-zh" x="52" y="99" fill="#1F6280" text-anchor="middle">seed 0</text><g fill="#1a5c38"><rect x="120" y="18" width="300" height="30" rx="6" fill="#E8F5EE" stroke="#2D8B55"/><text x="270" y="37" text-anchor="middle">0.54  0.72  0.60  0.54 …</text><rect x="120" y="80" width="300" height="30" rx="6" fill="#E8F5EE" stroke="#2D8B55"/><text x="270" y="99" text-anchor="middle">0.54  0.72  0.60  0.54 …</text></g><text class="lang-en" x="460" y="68" text-anchor="middle" fill="#2D8B55" font-size="13" font-weight="bold">same!</text><text class="lang-zh" x="460" y="68" text-anchor="middle" fill="#2D8B55" font-size="13" font-weight="bold">一样！</text></g></svg>',
  note:'<span class="lang-en"><b>Same seed → same stream.</b> Seed 0 today and seed 0 tomorrow start from the exact same state, so the whole stream repeats to the last digit. This is what <code>np.random.seed(0)</code> gives you — a reproducible run.</span><span class="lang-zh"><b>同一个 seed → 同一串数字。</b>今天的 seed 0 和明天的 seed 0 从完全相同的 state 出发。所以整串数字连最后一位都重复一样。这就是 <code>np.random.seed(0)</code> 给你的东西 —— 一次可复现的运行。</span>'},
 {viz:'<svg viewBox="0 0 520 130" role="img" aria-label="Seed 0 and seed 1 give two different streams"><g font-family="monospace" font-size="11"><rect x="20" y="18" width="64" height="30" rx="6" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text class="lang-en" x="52" y="37" fill="#1F6280" text-anchor="middle">seed 0</text><text class="lang-zh" x="52" y="37" fill="#1F6280" text-anchor="middle">seed 0</text><rect x="20" y="80" width="64" height="30" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="52" y="99" fill="#C93B3B" text-anchor="middle">seed 1</text><text class="lang-zh" x="52" y="99" fill="#C93B3B" text-anchor="middle">seed 1</text><rect x="120" y="18" width="300" height="30" rx="6" fill="#E8F5EE" stroke="#2D8B55"/><text x="270" y="37" text-anchor="middle" fill="#1a5c38">0.54  0.72  0.60  0.54 …</text><rect x="120" y="80" width="300" height="30" rx="6" fill="#FDE8E8" stroke="#C93B3B"/><text x="270" y="99" text-anchor="middle" fill="#C93B3B">0.51  0.95  0.14  0.30 …</text><text class="lang-en" x="460" y="68" text-anchor="middle" fill="#C93B3B" font-size="13" font-weight="bold">differ</text><text class="lang-zh" x="460" y="68" text-anchor="middle" fill="#C93B3B" font-size="13" font-weight="bold">不一样</text></g></svg>',
  note:'<span class="lang-en"><b>Different seed → different stream.</b> Change the seed to 1 and you get a whole new, unrelated stream — yet still perfectly repeatable. This is <code>default_rng(0)</code> vs <code>default_rng(1)</code>: two independent, reproducible streams.</span><span class="lang-zh"><b>不同的 seed → 不同的一串数字。</b>把 seed 换成 1，你得到一串全新的、和之前没关系的数字。但它一样可以完美重复。这就是 <code>default_rng(0)</code> 和 <code>default_rng(1)</code> 的区别：两条各自独立、又都能重复的数列。</span>'},
 {viz:'<svg viewBox="0 0 520 130" role="img" aria-label="Two ways to seed NumPy: the old global seed and the modern default_rng"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="30" y="25" width="200" height="80" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text class="lang-en" x="130" y="45" fill="#6B645E" font-size="10">old · one global state</text><text class="lang-zh" x="130" y="45" fill="#6B645E" font-size="10">老写法 · 一个共用状态</text><text class="lang-en" x="130" y="68" fill="#2C2A28">np.random.seed(0)</text><text class="lang-zh" x="130" y="68" fill="#2C2A28">np.random.seed(0)</text><text class="lang-en" x="130" y="88" fill="#2C2A28">np.random.rand()</text><text class="lang-zh" x="130" y="88" fill="#2C2A28">np.random.rand()</text><rect x="290" y="25" width="200" height="80" rx="8" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text class="lang-en" x="390" y="45" fill="#1F6280" font-size="10">modern · your own object</text><text class="lang-zh" x="390" y="45" fill="#1F6280" font-size="10">新写法 · 你自己的对象</text><text class="lang-en" x="390" y="68" fill="#1F6280">rng = default_rng(0)</text><text class="lang-zh" x="390" y="68" fill="#1F6280">rng = default_rng(0)</text><text class="lang-en" x="390" y="88" fill="#1F6280">rng.random()</text><text class="lang-zh" x="390" y="88" fill="#1F6280">rng.random()</text></g></svg>',
  note:'<span class="lang-en"><b>Two ways to seed NumPy.</b> The old way sets one shared generator for the whole program. The modern way makes <code>rng = np.random.default_rng(0)</code>, your own generator object — safer, because two parts of your code never step on each other.</span><span class="lang-zh"><b>给 NumPy 设 seed 的两种写法。</b>老写法给整个程序设一个共用的生成器。新写法做出 <code>rng = np.random.default_rng(0)</code>，一个你自己的生成器对象 —— 更安全，因为你代码里的两个地方不会踩到对方。</span>'},
 {viz:'<svg viewBox="0 0 520 130" role="img" aria-label="In JAX the key is passed explicitly and split into two fresh keys"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="40" y="45" width="90" height="38" rx="7" fill="#E4F2F7" stroke="#2A7B9B" stroke-width="2"/><text class="lang-en" x="85" y="68" fill="#1F6280">key</text><text class="lang-zh" x="85" y="68" fill="#1F6280">key</text><text x="150" y="68" fill="#6B645E" font-size="15">→</text><rect x="175" y="40" width="90" height="48" rx="8" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="220" y="60" fill="#5E5191">split</text><text class="lang-zh" x="220" y="60" fill="#5E5191">split</text><text class="lang-en" x="220" y="78" fill="#5E5191" font-size="10">(explicit)</text><text class="lang-zh" x="220" y="78" fill="#5E5191" font-size="10">（要自己传）</text><text x="285" y="68" fill="#6B645E" font-size="15">→</text><rect x="310" y="24" width="90" height="34" rx="7" fill="#E8F5EE" stroke="#2D8B55"/><text class="lang-en" x="355" y="46" fill="#1a5c38">key A</text><text class="lang-zh" x="355" y="46" fill="#1a5c38">key A</text><rect x="310" y="72" width="90" height="34" rx="7" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="355" y="94" fill="#9A7208">key B</text><text class="lang-zh" x="355" y="94" fill="#9A7208">key B</text><text class="lang-en" x="455" y="68" fill="#6B645E" font-size="10">Module 8</text><text class="lang-zh" x="455" y="68" fill="#6B645E" font-size="10">Module 8</text></g></svg>',
  note:'<span class="lang-en"><b>JAX makes the seed explicit.</b> In Module 8 there is no hidden state — you pass an explicit <code>key</code> into every random call and <code>split</code> it to make fresh randomness. Same idea as a seed, made visible. That closes Module 1 — you now speak the whole language.</span><span class="lang-zh"><b>JAX 把 seed 摆到明面上。</b>在 Module 8 里没有藏起来的 state —— 你每次调用随机函数都要自己传一个 <code>key</code>，再用 <code>split</code> 做出新的随机性。想法和 seed 一样，只是看得见了。Module 1 到这里就学完了 —— 这门语言你现在全都会说了。</span>'}
];
@@@ region name=QS
var QS=[
 {q:'<span class="lang-en">1. What does setting a seed do?</span><span class="lang-zh">1. 设一个 seed 是在做什么？</span>',
  opts:['<span class="lang-en">Makes the numbers truly random</span><span class="lang-zh">让这些数字变成真正的随机</span>',
        '<span class="lang-en">Makes the "random" numbers repeat the same way every run</span><span class="lang-zh">让这些「随机」数字每次运行都以同样的方式重复</span>',
        '<span class="lang-en">Deletes the random generator</span><span class="lang-zh">把随机数生成器删掉</span>',
        '<span class="lang-en">Speeds up NumPy</span><span class="lang-zh">让 NumPy 变快</span>'],
  ans:1, fb:'<span class="lang-en">Right. The same seed gives the same starting state, so the whole stream repeats exactly — that is reproducibility.</span><span class="lang-zh">对了。同一个 seed 给出同一个起始 state，所以整串数字会完全一样地重复。这就是 reproducibility。</span>'},
 {q:'<span class="lang-en">2. Why does reproducibility matter?</span><span class="lang-zh">2. 可复现为什么重要？</span>',
  opts:['<span class="lang-en">It makes numbers bigger</span><span class="lang-zh">它让数字变大</span>',
        '<span class="lang-en">So a flaky bug shows up every time and others can rerun your result</span><span class="lang-zh">这样时好时坏的 bug 每次都会出现，别人也能重跑你的结果</span>',
        '<span class="lang-en">It hides mistakes</span><span class="lang-zh">它把错误藏起来</span>',
        '<span class="lang-en">It is required by Python</span><span class="lang-zh">Python 规定必须这么做</span>'],
  ans:1, fb:'<span class="lang-en">Right. A fixed seed makes bugs repeatable so you can debug them, and lets others confirm your result — that is how it becomes science.</span><span class="lang-zh">对了。固定的 seed 让 bug 每次都能重现，你才修得动。它也让别人能确认你的结果。这样它才算科学。</span>'},
 {q:'<span class="lang-en">3. Is a PRNG truly random?</span><span class="lang-zh">3. PRNG 是真正的随机吗？</span>',
  opts:['<span class="lang-en">Yes, fully random</span><span class="lang-zh">是，完全随机</span>',
        '<span class="lang-en">No — it is a fixed recipe fully decided by the seed, only made to look random</span><span class="lang-zh">不是 —— 它是一份固定的配方，完全由 seed 决定，只是做得看起来像随机</span>',
        '<span class="lang-en">Only on Tuesdays</span><span class="lang-zh">只有星期二才是</span>',
        '<span class="lang-en">Only with a GPU</span><span class="lang-zh">只有用 GPU 的时候才是</span>'],
  ans:1, fb:'<span class="lang-en">Right. "Pseudo" means fake. The seed fixes the whole stream in advance; it just looks scattered enough to pass as random.</span><span class="lang-zh">对了。「pseudo」就是「假」。seed 提前把整串数字定好了。它只是看起来够乱，能冒充随机。</span>'},
 {q:'<span class="lang-en">4. Training runs fine and is perfectly reproducible, but a model with data augmentation across 4 parallel data-loading workers trains worse than expected — as if it saw far less variety than the data holds. No error appears. Where do you look first, and why?</span><span class="lang-zh">4. 训练跑得很顺，而且完全可复现。可是有个模型用了 data augmentation（数据增强），由 4 个并行的 data-loading worker 供数据，它训得比预期差。它就像只见过比数据里少得多的花样。程序也不报错。你先去看哪里，为什么？</span>',
  opts:['<span class="lang-en">The learning rate — low variety is always an optimizer problem</span><span class="lang-zh">learning rate（学习率）—— 花样少一定是 optimizer（优化器）的问题</span>',
        '<span class="lang-en">Whether all 4 workers were given the same fixed seed, so each produced the identical "random" augmentation stream instead of independent ones</span><span class="lang-zh">看这 4 个 worker 是不是拿到了同一个固定 seed。如果是，每个 worker 产生的「随机」增强就完全相同，而不是各自独立的</span>',
        '<span class="lang-en">The dtype of the images, since that controls randomness</span><span class="lang-zh">图片的 dtype（数据类型），因为它管着随机性</span>',
        '<span class="lang-en">Nothing — reproducible training cannot have a data problem</span><span class="lang-zh">不用看 —— 可复现的训练不可能有数据问题</span>'],
  ans:1, fb:'<span class="lang-en">Right. A PRNG stream is fully decided by its seed, so if every worker starts from the same seed they emit the exact same "random" augmentations — the data secretly repeats instead of varying. It stays reproducible and never errors, so the bug is silent. The fix is to offset each worker\'s seed by its worker id (and often the epoch) so the streams are independent.</span><span class="lang-zh">对了。一条 PRNG 数列完全由它的 seed 决定。所以如果每个 worker 都从同一个 seed 出发，它们吐出完全相同的「随机」增强。数据就悄悄在重复，而不是在变化。它还是可复现的，也不报错，所以这个 bug 藏得很好。修法是：把每个 worker 的 seed 按它的 worker id 错开，常常还要按 epoch 错开，这样每条数列才互相独立。</span>'}
];
