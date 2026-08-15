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
<div class="brand-sub">Foundations · M1 Day 6</div>
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
<a class="lnav prev" href="../day-05-logs-and-exponents/lesson.html"><span class="d"><span class="lang-en">← Prev</span><span class="lang-zh">← 上一天</span></span><span class="t">Logs &amp; Exponents</span></a>
@@@ region name=nav_next
<a class="lnav next" href="../review.html"><span class="d"><span class="lang-en">Next →</span><span class="lang-zh">下一天 →</span></span><span class="t">Review Gate</span></a>
@@@ region name=hero
<section id="home" class="hero">
      <span class="kicker">Module 1 · Represent · Day 6</span>
      <h1>Random Seeds &amp; PRNG<span class="sub">Make "Random" Repeat on Command</span></h1>
      <p class="lede">Computers cannot make true randomness. They fake it with a recipe. Feed that recipe a starting number — a <strong>seed</strong> — and the "random" numbers come out the same every single time. This tiny trick is what lets you debug a flaky model, and lets other people rerun your result and get the same answer. That is what turns a lucky run into real science.</p>
      <div class="goal"><span class="gic" aria-hidden="true">🎯</span><div><b>By the end, you'll be able to:</b> say why reproducibility matters, seed NumPy with <code>np.random.seed(0)</code> and the modern <code>np.random.default_rng(0)</code>, and explain what a PRNG really is.</div></div>
    </section>
@@@ region name=s1
<section class="module-section" id="s1" data-sec="what">
  <div class="sec-head"><span class="sec-num s-what">1</span><span class="sec-h">Seeds, PRNGs, and reproducibility</span><span class="sec-tag">What is it</span></div>
  <div class="sec-body">
      <div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> just Days 1–5 and how to call a NumPy function. <b>No probability math.</b> This is the last skill of Module 1 — how to make "random" behave.</div></div>

      <h4>What is a seed?</h4>
      <p>A <span class="term" data-tip="A starting number you give a random-number generator. The same seed always produces the same sequence of numbers.">seed</span> is one starting number you hand to the random-number maker. Give it the same seed and it always spits out the same "random" numbers, in the same order. Change the seed and you get a whole new sequence.</p>

      <h4>What is a PRNG?</h4>
      <p>A <span class="term" data-tip="Pseudo-Random Number Generator: a fixed recipe that turns a seed into a long sequence of numbers that only look random. Fully decided by the seed.">PRNG</span> is short for <strong>pseudo-random number generator</strong>. "Pseudo" means "fake." It is a fixed recipe: put a seed in, get a long stream of numbers out. The numbers look scattered and unpredictable, but they are not truly random — they are fully decided by the seed.</p>

      <h4>What is reproducibility?</h4>
      <p><span class="term" data-tip="The ability to run the same code and get exactly the same result every time. Others can rerun your experiment and confirm it.">Reproducibility</span> means your program gives the same result every time you run it. It matters for two reasons. First, <strong>debugging</strong>: if a bug only shows up sometimes, you cannot fix it — but a fixed seed makes it show up every time. Second, <strong>science</strong>: other people can rerun your code and check that your result was real, not luck. Keep going 👇</p>
      <div class="callout c-warn"><span class="ic">😕</span><div><b>为什么这里容易卡住 · Why this trips people up:</b> "随机"这个词让人以为它不可控，其实 seed 让它 100% 可复现 — the word "random" makes people think it's uncontrollable, when a seed makes it perfectly repeatable. In plain terms: people either forget to seed (and can't reproduce a bug) or over-share one seed (and silently kill the randomness they wanted) — the failure is invisible either way.</div></div>
      <button class="gotit" type="button">Got the three words — let's go</button>
    </div>
</section>
@@@ region name=s2
<section class="module-section" id="s2" data-sec="intuition">
  <div class="sec-head"><span class="sec-num s-study">2</span><span class="sec-h">A card shuffle with a written recipe</span><span class="sec-tag">Intuition</span></div>
  <div class="sec-body">
      <div class="relate">
        <div class="card"><span class="big">🃏</span><h5>The seed is the recipe</h5><p>Imagine a shuffle you follow from written steps: "cut at card 7, then card 3, then card 9…". Two people who start from a fresh deck and follow the <strong>same</strong> written steps end up with the <strong>exact same</strong> shuffled order.</p></div>
        <div class="card"><span class="big">🔁</span><h5>Same seed = same shuffle</h5><p>The seed is that written recipe. Give the PRNG seed 0 and you always get shuffle-0. Give it seed 1 and you get a totally different shuffle-1. Every run repeats perfectly.</p></div>
      </div>
      <p><strong>In one line:</strong> the numbers <em>look</em> shuffled and surprising, but the seed is a recipe that fixes the whole order in advance.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div>Where it breaks: a real card shuffle is truly random — you can never repeat it. A PRNG is <strong>not</strong> truly random. It is only unpredictable enough to look random, while staying 100% repeatable from the seed.</div></div>
      <p><b>直觉 / Intuition:</b> seed 不是"随机的来源"，而是"那份写好的洗牌步骤" — the seed isn't the source of randomness, it's the written recipe that fixes the whole order in advance. The technical point: a PRNG is a deterministic chain <code>state → number → new state</code>, and the seed is just the first state.</p>
      <button class="gotit" type="button">Got the picture</button>
    </div>
</section>
@@@ region name=s4
<section class="module-section" id="s4" data-sec="why">
  <div class="sec-head"><span class="sec-num s-study">4</span><span class="sec-h">How the recipe works — and why it runs the industry</span><span class="sec-tag">Mechanism &amp; why</span></div>
  <div class="sec-body">
      <h4>The idea of a PRNG, in plain words</h4>
      <p>A PRNG keeps a hidden <strong>state</strong> — one big number in memory. Each time you ask for a random value, it scrambles that state with a fixed formula, hands you a piece of the result, and saves the new state for next time. So the stream is a chain: state → number → new state → next number, and on and on.</p>
      <div class="callout c-ok"><span class="ic">🧮</span><div><b>The recipe in one line:</b><br><code>next_state = f(state)</code>, &nbsp; <code>output = g(state)</code><br><span style="font-size:.85rem;color:var(--text2)">Here <code>state</code> is the hidden number in memory, <code>f</code> is a fixed scrambling formula that makes the next state, and <code>g</code> reads a value out of the state. The <strong>seed</strong> is just the very first <code>state</code>. Same seed → same first state → the whole chain repeats exactly.</span></div></div>
      <p><strong>Worked example (small numbers).</strong> Say <code>f(s) = (5·s + 3) mod 16</code> and we start with seed <code>state = 1</code>. Then the states go <code>1 → (5·1+3) mod 16 = 8 → (5·8+3) mod 16 = 11 → (5·11+3) mod 16 = 10</code>. The stream <code>1, 8, 11, 10, …</code> looks jumpy and hard to guess, but start again from seed 1 and you get the very same list. Real PRNGs use much bigger, better formulas — but this is the whole idea.</p>

      <h4>Two ways to seed NumPy</h4>
      <p>The old way sets one shared generator for the whole program: <code>np.random.seed(0)</code>, then calls like <code>np.random.rand()</code> draw from it. The modern, recommended way makes your own generator object: <code>rng = np.random.default_rng(0)</code>, then <code>rng.random()</code>. The modern way is safer because two parts of your code can hold two separate generators and never step on each other.</p>

      <h4>A bridge to Module 8 (just so you've seen it)</h4>
      <div class="callout c-info"><span class="ic">🔗</span><div><b>Awareness only — no code to run today.</b> In <span class="term" data-tip="A numerical computing library used for large ML models. It handles randomness by passing an explicit key you split yourself, instead of a hidden global state.">JAX</span> (Module 8) there is no hidden global state. You pass an explicit <span class="term" data-tip="In JAX, the piece of randomness you carry by hand. You split one key into two to make fresh, independent randomness. Same idea as a NumPy seed, made visible.">key</span> into every random call, and you <strong>split</strong> that key to make fresh randomness. Same idea as a seed — just made visible and explicit so it works across many chips at once. You do not need JAX today; just know the seed idea grows up into an explicit key.</div></div>

      <h4>The staff lens — one silent failure, one trade-off</h4>
      <p>A seed is a promise that the same recipe repeats. That promise becomes a trap when many workers get the <em>same</em> seed — the repetition is now happening where you did not want it.</p>
      <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Failure mode (silent): every worker shares one seed, so their "random" streams are identical (a bug that doesn't announce itself).</b> Training often splits data loading across several parallel workers, each meant to draw fresh random augmentations or a fresh shuffle. If each worker starts from the <em>same</em> fixed seed, they all produce the <strong>exact same</strong> stream — so the "random" crops or noise repeat across workers instead of varying. Nothing errors; results are perfectly reproducible; the data just secretly has far less variety than you think, and the model trains worse for no visible reason. A senior engineer catches this in <b>code review</b> by asking "is the seed offset by the worker id (and the epoch)?" — a single global seed shared by all workers is the tell.</div></div>
      <div class="callout c-info"><span class="ic">⚖️</span><div><b>Trade-off: reproducibility vs. seeing the true variance.</b> Fixing one seed makes a run repeatable, which is priceless for debugging and for others rerunning your result. What you pay: a single seed shows you one draw of a random process, so a lucky (or unlucky) seed can make a change look better or worse than it really is. In a <b>design review</b> the rule is to fix the seed while debugging, but to report a result across <em>several</em> seeds — the spread across seeds is the honest error bar, and a claim backed by one seed is not yet evidence.</div></div>
      <div class="callout c-ok"><span class="ic">🎤</span><div><b>Say this in an interview:</b> "A PRNG is fully deterministic — <code>next_state = f(state)</code>, <code>output = g(state)</code>, and the seed is just the first state — so the same seed replays the exact stream, which is what makes runs reproducible. The nuance is two failure modes: no seed means a heisenbug you can't reproduce, and one shared seed across data-loader workers means their 'random' augmentations are identical, so I offset the seed by worker id and epoch and report results across several seeds."</div></div>
      <button class="gotit" type="button">Got the mechanism</button>
    </div>
</section>
@@@ region name=s7
<section class="module-section" id="s7" data-sec="produce">
  <div class="sec-head"><span class="sec-num s-produce">7</span><span class="sec-h">Leave today's evidence (strongly recommended)</span><span class="sec-tag">Produce</span></div>
  <div class="sec-body">
      <p>Understanding ≠ being able to write it. Make "random" repeat on command yourself — that's what makes the day count. Pick one:</p>
      <h4>Option A · write it yourself</h4>
      <p>Create <code>sessions/m01-shape-of-data/day-06-random-seeds/experiment.py</code>. Draw two numbers with <code>np.random.rand()</code> and no seed (they differ). Then call <code>np.random.seed(0)</code>, draw an array, reseed with <code>np.random.seed(0)</code> again, draw again, and <code>assert</code> the two arrays are equal. Last, make <code>np.random.default_rng(0)</code> and <code>np.random.default_rng(1)</code> and show their first draws differ. Run it with <code>python3 sessions/m01-shape-of-data/day-06-random-seeds/experiment.py</code>.</p>
      <h4>Option B · let Claude build it, then read it</h4>
      <p>Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.</p>
      <div class="prompt">
        <div class="prompt-h"><span class="prompt-l">ask Claude to build it</span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
        <pre class="prompt-t" id="pp">Help me build my Module 1 Day 6 artifact.

Create sessions/m01-shape-of-data/day-06-random-seeds/experiment.py that, with a comment on each step:
1. Draws two values with np.random.rand() and NO seed, prints both, and notes they differ run to run.
2. Calls np.random.seed(0), draws a = np.random.rand(3); then calls np.random.seed(0) again, draws b = np.random.rand(3); prints a and b and asserts np.array_equal(a, b) — same seed reproduces the exact stream.
3. Makes rng0 = np.random.default_rng(0) and rng1 = np.random.default_rng(1); draws s0 = rng0.random(3) and s1 = rng1.random(3); prints both and asserts NOT np.array_equal(s0, s1) — different seeds give different reproducible streams.
Then run it and paste the output at the bottom as a comment.</pre>
      </div>
      <h4>Acceptance criteria</h4>
      <ul>
        <li>Two unseeded draws differ; two draws after the <em>same</em> <code>np.random.seed(0)</code> are equal (you <code>assert</code> it).</li>
        <li><code>default_rng(0)</code> and <code>default_rng(1)</code> give different but each-reproducible streams.</li>
        <li>You can explain why the same seed always yields the same numbers, in one sentence, without saying "it's random".</li>
      </ul>
      <div class="callout c-info"><span class="ic">📓</span><div><b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in <code>sessions/m01-shape-of-data/day-06-random-seeds/log.md</code>: (1) in your own words, why the same seed always gives the same "random" numbers; (2) the one place a shared seed would bite you; (3) one thing you're still unsure about.</div></div>
      <button class="gotit" type="button">Done</button>
    </div>
</section>
@@@ region name=fin
<div class="fin" id="fin" role="status" aria-hidden="true">
      <span class="em" aria-hidden="true">🎉</span>
      <h3>Module 1 · Day 6 complete! 🏆</h3>
      <p>Nice work — you've completed <b>Random Seeds &amp; PRNG</b>.<br>Next up: <b>Review Gate</b>.</p>
    </div>
@@@ region name=DEMOS
var DEMOS = {
  noseed:{html:'<span class="prompt">&gt;&gt;&gt;</span> np.random.rand()   <span class="dim"># no seed set — first draw</span>\n'+
    '<span class="hl">0.417022004702574</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.random.rand()   <span class="dim"># draw again</span>\n'+
    '<span class="hl">0.7203244934421581</span>\n'+
    '<span class="dim"># run the program again tomorrow → two DIFFERENT numbers</span>',
    take:'<b>①  No seed = no repeat.</b> Each call moves the hidden state forward, so you get a new number. Restart the program and the whole run looks different. Hard to debug, hard to trust.'},
  seeded:{html:'<span class="prompt">&gt;&gt;&gt;</span> np.random.seed(<span class="num">0</span>)     <span class="dim"># set the starting state</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.random.rand(<span class="num">3</span>)\n'+
    '<span class="ok">array([0.5488, 0.7152, 0.6028])</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.random.seed(<span class="num">0</span>)     <span class="dim"># reset to the same seed</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> np.random.rand(<span class="num">3</span>)\n'+
    '<span class="ok">array([0.5488, 0.7152, 0.6028])</span>   <span class="dim"># identical!</span>',
    take:'<b>②  Same seed = same stream.</b> Seed 0 sets the exact starting state, so both draws match to the last digit. This is what makes a run reproducible — every time, forever.'},
  streams:{html:'<span class="prompt">&gt;&gt;&gt;</span> rng0 = np.random.default_rng(<span class="num">0</span>)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> rng1 = np.random.default_rng(<span class="num">1</span>)\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> rng0.random(<span class="num">3</span>)\n'+
    '<span class="hl">array([0.6370, 0.2698, 0.0409])</span>   <span class="dim"># seed 0 stream</span>\n'+
    '<span class="prompt">&gt;&gt;&gt;</span> rng1.random(<span class="num">3</span>)\n'+
    '<span class="hl">array([0.5118, 0.9505, 0.1442])</span>   <span class="dim"># seed 1 stream — different</span>',
    take:'<b>③  Different seeds = different, but each is repeatable.</b> <code>default_rng(0)</code> and <code>default_rng(1)</code> are two separate generators. Each gives the same stream every run — but the two streams differ from each other.'}
};
@@@ region name=BUILD
var BUILD=[
 {viz:"<svg viewBox='0 0 520 100' role='img' aria-label='A single seed number, 0'><g font-family='monospace' text-anchor='middle'><rect x='210' y='30' width='100' height='42' rx='8' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='50' fill='#1F6280' font-size='12'>seed</text><text x='260' y='66' fill='#1F6280' font-size='16' font-weight='bold'>0</text><text x='260' y='92' fill='#6B645E' font-size='11'>one starting number, chosen by you</text></g></svg>",
  note:"<b>Start with a seed.</b> A seed is just one number you pick, like <code>0</code>. On its own it does nothing — it is the starting point for everything that follows."},
 {viz:"<svg viewBox='0 0 520 110' role='img' aria-label='The seed goes into a PRNG box and a stream of numbers comes out'><g font-family='monospace' text-anchor='middle'><rect x='20' y='40' width='70' height='36' rx='7' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='55' y='63' fill='#1F6280' font-size='13'>seed 0</text><text x='110' y='62' fill='#6B645E' font-size='16'>→</text><rect x='130' y='34' width='120' height='48' rx='8' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='190' y='53' fill='#5E5191' font-size='12'>PRNG</text><text x='190' y='70' fill='#5E5191' font-size='10'>fixed recipe f, g</text><text x='268' y='62' fill='#6B645E' font-size='16'>→</text><g fill='#9A7208'><rect x='290' y='45' width='36' height='28' rx='5' fill='#FCF3DC' stroke='#C99A12'/><rect x='334' y='45' width='36' height='28' rx='5' fill='#FCF3DC' stroke='#C99A12'/><rect x='378' y='45' width='36' height='28' rx='5' fill='#FCF3DC' stroke='#C99A12'/><rect x='422' y='45' width='36' height='28' rx='5' fill='#FCF3DC' stroke='#C99A12'/></g><g fill='#9A7208' font-size='10'><text x='308' y='63'>.54</text><text x='352' y='63'>.72</text><text x='396' y='63'>.60</text><text x='440' y='63'>…</text></g></svg>",
  note:"<b>The PRNG turns the seed into a stream.</b> The seed feeds a fixed recipe. Each step scrambles a hidden state and hands you one number, then saves the new state. Out comes a long line of numbers that <em>look</em> random."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Seed 0 twice gives two identical streams'><g font-family='monospace' font-size='11'><rect x='20' y='18' width='64' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='52' y='37' fill='#1F6280' text-anchor='middle'>seed 0</text><rect x='20' y='80' width='64' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='52' y='99' fill='#1F6280' text-anchor='middle'>seed 0</text><g fill='#1a5c38'><rect x='120' y='18' width='300' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='270' y='37' text-anchor='middle'>0.54  0.72  0.60  0.54 …</text><rect x='120' y='80' width='300' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='270' y='99' text-anchor='middle'>0.54  0.72  0.60  0.54 …</text></g><text x='460' y='68' text-anchor='middle' fill='#2D8B55' font-size='13' font-weight='bold'>same!</text></g></svg>",
  note:"<b>Same seed → same stream.</b> Seed 0 today and seed 0 tomorrow start from the exact same state, so the whole stream repeats to the last digit. This is what <code>np.random.seed(0)</code> gives you — a reproducible run."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Seed 0 and seed 1 give two different streams'><g font-family='monospace' font-size='11'><rect x='20' y='18' width='64' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='52' y='37' fill='#1F6280' text-anchor='middle'>seed 0</text><rect x='20' y='80' width='64' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='52' y='99' fill='#C93B3B' text-anchor='middle'>seed 1</text><rect x='120' y='18' width='300' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='270' y='37' text-anchor='middle' fill='#1a5c38'>0.54  0.72  0.60  0.54 …</text><rect x='120' y='80' width='300' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='270' y='99' text-anchor='middle' fill='#C93B3B'>0.51  0.95  0.14  0.30 …</text><text x='460' y='68' text-anchor='middle' fill='#C93B3B' font-size='13' font-weight='bold'>differ</text></g></svg>",
  note:"<b>Different seed → different stream.</b> Change the seed to 1 and you get a whole new, unrelated stream — yet still perfectly repeatable. This is <code>default_rng(0)</code> vs <code>default_rng(1)</code>: two independent, reproducible streams."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='Two ways to seed NumPy: the old global seed and the modern default_rng'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='25' width='200' height='80' rx='8' fill='#FDF9F3' stroke='#E5DFD6' stroke-width='1.5'/><text x='130' y='45' fill='#6B645E' font-size='10'>old · one global state</text><text x='130' y='68' fill='#2C2A28'>np.random.seed(0)</text><text x='130' y='88' fill='#2C2A28'>np.random.rand()</text><rect x='290' y='25' width='200' height='80' rx='8' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='390' y='45' fill='#1F6280' font-size='10'>modern · your own object</text><text x='390' y='68' fill='#1F6280'>rng = default_rng(0)</text><text x='390' y='88' fill='#1F6280'>rng.random()</text></g></svg>",
  note:"<b>Two ways to seed NumPy.</b> The old way sets one shared generator for the whole program. The modern way makes <code>rng = np.random.default_rng(0)</code>, your own generator object — safer, because two parts of your code never step on each other."},
 {viz:"<svg viewBox='0 0 520 130' role='img' aria-label='In JAX the key is passed explicitly and split into two fresh keys'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='45' width='90' height='38' rx='7' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='85' y='68' fill='#1F6280'>key</text><text x='150' y='68' fill='#6B645E' font-size='15'>→</text><rect x='175' y='40' width='90' height='48' rx='8' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='220' y='60' fill='#5E5191'>split</text><text x='220' y='78' fill='#5E5191' font-size='10'>(explicit)</text><text x='285' y='68' fill='#6B645E' font-size='15'>→</text><rect x='310' y='24' width='90' height='34' rx='7' fill='#E8F5EE' stroke='#2D8B55'/><text x='355' y='46' fill='#1a5c38'>key A</text><rect x='310' y='72' width='90' height='34' rx='7' fill='#FCF3DC' stroke='#C99A12'/><text x='355' y='94' fill='#9A7208'>key B</text><text x='455' y='68' fill='#6B645E' font-size='10'>Module 8</text></g></svg>",
  note:"<b>JAX makes the seed explicit.</b> In Module 8 there is no hidden state — you pass an explicit <code>key</code> into every random call and <code>split</code> it to make fresh randomness. Same idea as a seed, made visible. That closes Module 1 — you now speak the whole language."}
];
@@@ region name=QS
var QS=[
 {q:'1. What does setting a seed do?',
  opts:['Makes the numbers truly random','Makes the "random" numbers repeat the same way every run','Deletes the random generator','Speeds up NumPy'],
  ans:1, fb:'Right. The same seed gives the same starting state, so the whole stream repeats exactly — that is reproducibility.'},
 {q:'2. Why does reproducibility matter?',
  opts:['It makes numbers bigger','So a flaky bug shows up every time and others can rerun your result','It hides mistakes','It is required by Python'],
  ans:1, fb:'Right. A fixed seed makes bugs repeatable so you can debug them, and lets others confirm your result — that is how it becomes science.'},
 {q:'3. Is a PRNG truly random?',
  opts:['Yes, fully random','No — it is a fixed recipe fully decided by the seed, only made to look random','Only on Tuesdays','Only with a GPU'],
  ans:1, fb:'Right. "Pseudo" means fake. The seed fixes the whole stream in advance; it just looks scattered enough to pass as random.'},
 {q:'4. Training runs fine and is perfectly reproducible, but a model with data augmentation across 4 parallel data-loading workers trains worse than expected — as if it saw far less variety than the data holds. No error appears. Where do you look first, and why?',
  opts:['The learning rate — low variety is always an optimizer problem','Whether all 4 workers were given the same fixed seed, so each produced the identical "random" augmentation stream instead of independent ones','The dtype of the images, since that controls randomness','Nothing — reproducible training cannot have a data problem'],
  ans:1, fb:'Right. A PRNG stream is fully decided by its seed, so if every worker starts from the same seed they emit the exact same "random" augmentations — the data secretly repeats instead of varying. It stays reproducible and never errors, so the bug is silent. The fix is to offset each worker\'s seed by its worker id (and often the epoch) so the streams are independent.'}
];
