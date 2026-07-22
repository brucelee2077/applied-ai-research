---
quest_id: w01-d04-jit
mode: concept
donor: v9-base.donor
page_title: "Week 1 · Day 4 — JAX jit: Compilation with XLA"
module_label: "Thinking in JAX · Week 1 · Day 4"
title: "jit: Compilation With XLA"
subtitle: "Trace Once, Then Run at Full Speed"
brand_sub: "Frontier Lab · Week 1 Day 4"
spine: "blueprint"
nav_prev_href: "../day-03-vmap/lesson.html"
nav_prev_label: "vmap: Vectorization"
nav_next_href: "../day-05-flax-optax/lesson.html"
nav_next_label: "Flax & Optax"
fin_title: "Week 1 · Day 4 complete! 🏆"
fin_body: "You've unlocked <b>jit</b>: wrap a function and JAX traces it once into a <b>blueprint</b>, hands it to the XLA compiler, and reuses that fast compiled version on every later call. The first call pays a one-time compile cost; the rest fly. You also met the catch — jit needs pure functions and can't branch on traced values with a plain Python <code>if</code>.<br>Next up: <b>Flax & Optax</b>."
notebook_yardstick: null
---

@@@ hero
@lede Imagine you're building the very same bookshelf ten times. The first time, you sketch a careful **blueprint**: measure every board, mark every screw hole, plan the whole thing. That first sketch is slow. But once you have the blueprint, every shelf after that is quick — you just *follow the plan*, no re-measuring. That is exactly the feeling of **jit**, a tool in **JAX** (the library behind many of today's biggest AI models, including work at Google DeepMind). You hand `jit` your Python function, and the first time you call it, JAX quietly watches it run and writes down a **blueprint** of every step. Then a fast compiler called XLA turns that blueprint into tight machine code. Every later call skips the sketching and just *runs the plan* — which is why real training loops, that call the same function millions of times, wrap it in `jit`.
@goal Together we'll meet <b>jit</b> — the tool that traces your function once into a blueprint, compiles it, and reuses it at full speed. You'll see how tracing works, what the recorded blueprint (a "jaxpr") looks like, why the compiled version beats running one operation at a time, and the two famous catches — printing inside jit, and using a Python <code>if</code> on a traced value — plus their fixes. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="Trace once, run fast" title="jit: sketch the plan once, then follow it" gotit="Got the big idea"
Let's meet the star of the day. [[jit||Short for "just-in-time compilation." A JAX tool that watches your function run once, writes down a blueprint of the steps, hands that blueprint to a fast compiler, and reuses the compiled version on every later call.]] stands for **just-in-time compilation**, but forget the fancy name for a second and hold onto the picture. When you wrap a function in `jit`, JAX does something clever: the **first** time you call it, JAX watches the whole function run and writes down a **blueprint** — a plan of every math step, in order. Then it hands that plan to a fast compiler (called XLA — we'll meet it soon) that turns the plan into tight machine code. Every call *after* the first one skips all the watching and planning. It just runs the finished plan. Fast.

Here's the mental picture, and it's our home base for the whole day: **drawing a blueprint for a bookshelf.** The first build is slow because you're measuring and sketching the plan. But once the blueprint exists, every next shelf is quick — you just follow it. `jit` sketches the blueprint of your function once, then follows it forever.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A blueprint analogy for jit. On the left, a person carefully drawing a blueprint labelled 'first call: JAX watches and sketches the plan (slow)'. An arrow labelled compile points right. On the right, three identical bookshelves being built quickly from that blueprint, labelled 'every later call: just follow the plan (fast)'. A label at the bottom reads: trace once into a blueprint, then reuse it on every call.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="18" fill="#2C2A28" font-size="12">jit = draw the blueprint ONCE, then follow it on every call</text>
<text x="120" y="44" fill="#9A5A12">first call (slow): sketch the plan</text>
<g transform="translate(58,54)"><rect x="0" y="0" width="120" height="86" rx="6" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><line x1="14" y1="18" x2="106" y2="18" stroke="#C99A12" stroke-dasharray="4 3"/><line x1="14" y1="40" x2="106" y2="40" stroke="#C99A12" stroke-dasharray="4 3"/><line x1="14" y1="62" x2="106" y2="62" stroke="#C99A12" stroke-dasharray="4 3"/><line x1="40" y1="10" x2="40" y2="76" stroke="#C99A12" stroke-dasharray="4 3"/><text x="60" y="84" fill="#8A6D3B" font-size="8">blueprint</text></g>
<text x="205" y="94" fill="#9A5A12" font-size="11">compile →</text>
<text x="400" y="44" fill="#276b45">every later call (fast): follow it</text>
<g transform="translate(300,54)" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"><rect x="0" y="0" width="40" height="86" rx="3"/><line x1="0" y1="28" x2="40" y2="28"/><line x1="0" y1="56" x2="40" y2="56"/><rect x="60" y="0" width="40" height="86" rx="3"/><line x1="60" y1="28" x2="100" y2="28"/><line x1="60" y1="56" x2="100" y2="56"/><rect x="120" y="0" width="40" height="86" rx="3"/><line x1="120" y1="28" x2="160" y2="28"/><line x1="120" y1="56" x2="160" y2="56"/></g>
<text x="260" y="184" fill="#9A938A" font-size="9">the first call pays a one-time cost to sketch + compile; all the rest just run the plan</text>
</g></svg>
%%%

**What the blueprint picture gets right:** you do the careful planning work exactly once, and then every repeat is fast because you just follow the finished plan — just like `jit` traces your function once and reuses the compiled version. **Where it breaks down:** a real blueprint is a static drawing on paper, but `jit`'s blueprint is *runnable* machine code the computer executes directly, and it's also specialized to the exact shapes of the data you first gave it (more on that surprise at the end of the day).

#### The one-line move: wrap and go
Using `jit` is almost embarrassingly simple. You take your function and wrap it — either with `jax.jit(f)` or by writing `@jax.jit` on the line above it. Nothing about the *inside* of your function changes. Let's watch the compile-then-run pattern happen. Predict it first: the first call will feel slow (it's sketching + compiling), and the second call — same function, same shapes — should feel fast.

%%% demo id=wrap label="wrap a function and call it twice"
code: import jax, jax.numpy as jnp
code: def f(x):                 # a plain Python function
code:     return jnp.sin(x) * 2.0 + 1.0
code: fast_f = jax.jit(f)       # wrap it — nothing else changes
code: x = jnp.arange(5.)
code: print(fast_f(x))         # 1st call: JAX traces + compiles, THEN runs
code: print(fast_f(x))         # 2nd call: reuses the compiled blueprint (fast)
out: [1.        2.6829419 2.8185949 1.2822400 -0.5136050]
out: [1.        2.6829419 2.8185949 1.2822400 -0.5136050]
take: <b>Same answer both times — but the work behind them differs.</b> The FIRST call quietly traced your function into a blueprint and compiled it (a one-time cost). The SECOND call skipped all of that and just ran the finished machine code. In a training loop you call the function millions of times, so you pay the compile cost <i>once</i> and reap the speed <i>forever</i>.
%%%

You've got the big idea: **trace once, run fast.** The natural next question is *how* JAX writes that blueprint without knowing your actual numbers yet. The trick is beautiful, and it's the next concept.

@@@ concept id=c2 tag="Tracing with stand-ins" title="Tracing: JAX runs your function on stand-ins, not numbers" gotit="Got tracing"
So how does JAX sketch a blueprint of your function? Here's the clever part. To write the plan, JAX runs your function **once** — but not on your real numbers. It runs it on **stand-ins**: fake values that carry only a *shape* and a *type*, no actual number inside. This one run is called [[tracing||The first-call step where JAX runs your function on stand-in values (not real numbers) to record every operation into a blueprint.]]. As the stand-ins flow through your function — added, multiplied, passed into `sin` — JAX writes down each step they touch. The list of steps it records is the blueprint.

Picture a **stunt double on a movie set.** Before the real actor films a dangerous scene, a stunt double walks through every move: step here, turn there, jump now. The director watches the double and writes down the whole sequence of moves — the *choreography*. The stunt double isn't the real star and doesn't say the real lines; they exist only to *record the moves*. JAX's stand-in value is the stunt double. It has the same "size" as the real data (same shape and type) but no real content — it's there so JAX can watch it move through your function and write down the choreography.

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="A stunt double analogy for tracing. On the left, a stand-in value labelled 'tracer: shape (5,), float32, NO real number' walks through three steps of a function labelled sin, times 2, plus 1. A director watches and writes each step onto a clipboard labelled 'the blueprint (jaxpr)'. A label reads: JAX traces by running the function on a stand-in that has shape and type but no value.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Tracing: run the function on a STAND-IN (shape + type, no number)</text>
<g transform="translate(20,44)"><rect x="0" y="0" width="118" height="42" rx="6" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="59" y="18" fill="#5E5191" font-size="9">tracer (stand-in)</text><text x="59" y="33" fill="#5E5191" font-size="8">shape (5,), float32</text></g>
<text x="150" y="68" fill="#9A5A12" font-size="11">→</text>
<g transform="translate(168,48)" fill="#FDF3E7" stroke="#C99A12"><rect x="0" y="0" width="44" height="34" rx="4"/><text x="22" y="21" fill="#8A6D3B" font-size="8" text-anchor="middle">sin</text></g>
<text x="220" y="68" fill="#9A5A12" font-size="11">→</text>
<g transform="translate(238,48)" fill="#FDF3E7" stroke="#C99A12"><rect x="0" y="0" width="44" height="34" rx="4"/><text x="22" y="21" fill="#8A6D3B" font-size="8" text-anchor="middle">× 2</text></g>
<text x="290" y="68" fill="#9A5A12" font-size="11">→</text>
<g transform="translate(308,48)" fill="#FDF3E7" stroke="#C99A12"><rect x="0" y="0" width="44" height="34" rx="4"/><text x="22" y="21" fill="#8A6D3B" font-size="8" text-anchor="middle">+ 1</text></g>
<path d="M355 65 C 400 65, 420 78, 430 100" stroke="#2D8B55" stroke-width="1.5" fill="none" marker-end="url(#a2)"/>
<defs><marker id="a2" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="#2D8B55"/></marker></defs>
<g transform="translate(360,104)"><rect x="0" y="0" width="150" height="66" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="75" y="16" fill="#1a5c38" font-size="9">the blueprint (jaxpr)</text><text x="75" y="32" fill="#276b45" font-size="8">1. sin(x)</text><text x="75" y="45" fill="#276b45" font-size="8">2. × 2.0</text><text x="75" y="58" fill="#276b45" font-size="8">3. + 1.0</text></g>
<text x="260" y="192" fill="#9A938A" font-size="9">the director (JAX) watches the stand-in move and writes down every step</text>
</g></svg>
%%%

**What the stunt-double picture gets right:** the stand-in walks through the exact same moves the real data will, so JAX can record the whole choreography without ever needing the real content. **Where it breaks down:** a stunt double is a whole separate person, but a JAX stand-in — called a [[tracer||A stand-in value used during tracing. It carries only shape and dtype (type), never a concrete number. Your operations act on it to record the blueprint.]] — is more like a ghost that only remembers *shape* and *type* (`float32`, shape `(5,)`) and forgets the numbers entirely.

#### Why "no real number" matters
This is the single most important thing to hold onto, because every catch later today comes back to it. During tracing, your function's variables are **tracers**, not numbers. A tracer knows it is "a `float32` array of shape `(5,)`" — but ask it "are you bigger than zero?" and it has no answer, because there *is* no number in it yet. The blueprint only records the *steps*, not the values. This is exactly why, later, a plain Python `if x > 0` will break inside `jit`: there's no number to compare.

#### See the blueprint yourself with make_jaxpr
The blueprint JAX records has a name: a [[jaxpr||JAX's name for the recorded blueprint — the little program of operations captured during tracing. You can print it with jax.make_jaxpr(f)(x).]] (say it "jax-per," short for *JAX expression*). It's just the tidy list of operations JAX wrote down. You don't normally need to look at it, but JAX gives you a window — `jax.make_jaxpr` — to *see the plan* it would compile. Predict what our three-step function records, then run it:

%%% demo id=jaxpr label="peek at the recorded blueprint"
code: def f(x):
code:     return jnp.sin(x) * 2.0 + 1.0
code: # make_jaxpr shows the blueprint WITHOUT compiling/running it
code: print(jax.make_jaxpr(f)(jnp.arange(3.)))
out: { lambda ; a:f32[3]. let
out:     b:f32[3] = sin a
out:     c:f32[3] = mul b 2.0
out:     d:f32[3] = add c 1.0
out:   in (d,) }
take: <b>There's the blueprint, in black and white.</b> <code>a:f32[3]</code> is the stand-in — a float32 array of shape 3, with no value shown. Each line is one recorded step: <code>sin</code>, then <code>mul</code> (×2), then <code>add</code> (+1). This little list is exactly what XLA compiles. Notice: no real numbers anywhere — just the <i>shapes</i>, the <i>types</i>, and the <i>steps</i>. That's the whole trick.
%%%

Now you know *what* the blueprint is and *how* JAX writes it. The next question is the fun one: once XLA has this plan, why does the compiled version run so much faster than plain Python? That payoff is the next concept.

@@@ concept id=c3 tag="Why it's faster" title="XLA fusion: one fast trip instead of many little errands" gotit="Got the payoff"
Here's the payoff that makes `jit` worth learning. Without `jit`, JAX runs your function the plain way: **one operation at a time**. It does the `sin`, hands the result back, then does the `× 2`, hands it back, then the `+ 1`. This is called [[eager execution||Running each operation immediately, one at a time — the plain, un-jitted way. Each op is a separate little trip to the chip, which adds up to a lot of overhead.]] — "eager" because each step runs the instant you write it. Each of those steps is a tiny separate errand to the chip, and each errand carries setup overhead. Lots of tiny trips means lots of wasted time. When you `jit` the function, the compiler XLA reads the whole blueprint and does something smarter: it glues neighboring steps together into **one** combined operation. That gluing is called [[fusion||XLA's trick of merging several neighboring operations into one combined kernel, so the data is touched once instead of shuttled back and forth between separate steps.]], and it's why the compiled version flies.

Picture **errands in a town.** You need bread, milk, and stamps. You *could* drive home after each one: drive to the bakery, home; drive to the shop, home; drive to the post office, home — three round trips, tons of driving. Or you plan **one** route that hits all three in a single loop. Same three items, far less driving. Eager execution is the three separate trips. XLA fusion is the one smart route: it plans the whole errand and touches your data once.

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="An errands analogy for XLA fusion. On the left, labelled eager, three separate round-trip loops each going out to one shop and back home, with a note 'three separate trips = slow'. On the right, labelled jit + XLA fusion, one single loop that visits all three shops in order, with a note 'one planned route = fast'. A label reads: fusion glues neighboring operations into one combined trip.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">eager = 3 separate trips · jit+XLA fusion = 1 planned route</text>
<text x="120" y="40" fill="#C93B3B" font-size="11">eager: op-by-op (many trips)</text>
<g transform="translate(40,50)"><circle cx="20" cy="60" r="9" fill="#3A342E"/><text x="20" y="84" fill="#6B645E" font-size="7">home</text><ellipse cx="90" cy="22" rx="26" ry="18" fill="none" stroke="#C93B3B" stroke-dasharray="4 3"/><ellipse cx="90" cy="60" rx="26" ry="18" fill="none" stroke="#C93B3B" stroke-dasharray="4 3"/><ellipse cx="90" cy="98" rx="26" ry="18" fill="none" stroke="#C93B3B" stroke-dasharray="4 3"/><text x="90" y="26" fill="#8a3b3b" font-size="8">sin</text><text x="90" y="64" fill="#8a3b3b" font-size="8">× 2</text><text x="90" y="102" fill="#8a3b3b" font-size="8">+ 1</text></g>
<text x="120" y="178" fill="#C93B3B">three separate trips = slow</text>
<text x="395" y="40" fill="#2D8B55" font-size="11">jit + XLA: one fused route</text>
<g transform="translate(320,50)"><circle cx="10" cy="60" r="9" fill="#3A342E"/><text x="10" y="84" fill="#6B645E" font-size="7">home</text><path d="M18 56 L70 20 L130 20 L130 100 L70 100 L18 66" fill="none" stroke="#2D8B55" stroke-width="2"/><g fill="#EAF5EE" stroke="#2D8B55"><circle cx="70" cy="20" r="12"/><circle cx="130" cy="20" r="12"/><circle cx="130" cy="100" r="12"/></g><text x="70" y="23" fill="#1a5c38" font-size="7">sin</text><text x="130" y="23" fill="#1a5c38" font-size="7">×2</text><text x="130" y="103" fill="#1a5c38" font-size="7">+1</text></g>
<text x="395" y="178" fill="#276b45">one planned route = fast</text>
</g></svg>
%%%

**What the errands picture gets right:** doing many little separate trips wastes a lot of travel, while one planned route that chains them together is far more efficient — just as fusion chains your operations into one pass over the data. **Where it breaks down:** in a town you still drive one road at a time, but on an AI chip the fused kernel can also run its inner work massively in parallel, so the real speed-up is even bigger than the single-route picture suggests.

#### Measure the speed-up (and the async catch)
Talk is cheap — let's *measure* it. There's one subtlety you must know first, or your timing will lie to you. JAX runs work **asynchronously**: when you call a function, it kicks off the work and hands control back to Python *before* the answer is actually ready. So if you time naively, you might time only "how long to *launch* the work," not "how long to *finish* it." The fix is one method: [[block_until_ready||A method you call on a JAX result to make Python wait until the computation has actually finished. Essential for honest timing, because JAX otherwise runs work asynchronously and returns before it's done.]] — it tells Python "wait right here until the real answer lands." Always call it when timing.

!!! c-info 🔬
<b>Optional (skippable) — a heavier function shows the gap bigger.</b> Fusion helps most when there are <i>many</i> element-wise steps in a row to glue together, because each glued step is one fewer round trip. A chain like <code>jnp.sin(x) + jnp.cos(x) * 2 - jnp.tanh(x)</code> has several steps XLA can fuse into a single pass over the array; the eager version makes a separate trip (and a separate temporary array) for each one. More steps in the chain → bigger win from <code>jit</code>.
!!!

Predict it: the `jit`ted version should be faster on repeated calls. Run the timing and watch the gap (numbers vary by machine — the *direction* is the lesson):

%%% demo id=speed label="time eager vs jit (with block_until_ready)"
code: import time
code: def f(x):  return jnp.sin(x) + jnp.cos(x) * 2.0 - jnp.tanh(x)
code: fast_f = jax.jit(f)
code: x = jnp.ones((1_000_000,))
code: _ = fast_f(x).block_until_ready()          # 1st call: pay the compile cost ONCE
code: def bench(fn):
code:     t = time.perf_counter()
code:     for _ in range(100):
code:         fn(x).block_until_ready()           # WAIT for the real answer each time
code:     return (time.perf_counter() - t) * 1e3  # milliseconds for 100 calls
code: print("eager (op-by-op):", round(bench(f), 1), "ms")
code: print("jit   (fused)   :", round(bench(fast_f), 1), "ms")
out: eager (op-by-op): 61.4 ms
out: jit   (fused)   : 18.7 ms
take: <b>The fused version wins.</b> Eager JAX made a separate trip (and a temporary array) for <code>sin</code>, <code>cos</code>, <code>×2</code>, <code>-tanh</code>. XLA fused them into one pass over the million numbers — fewer trips, less memory shuffled. Two habits made the measurement honest: we warmed up once so the compile cost isn't counted, and we called <code>block_until_ready()</code> so we timed the real work, not just the launch.
%%%

You now see *why* `jit` pays off. But every superpower has a catch — and `jit` has two famous ones, both flowing straight from the "no real number during tracing" idea from the last concept. The first catch surprises everyone: printing inside a `jit`ted function. That's next.

@@@ concept id=c4 tag="Catch 1: printing" title="The print puzzle: side effects run only ONCE" gotit="Solved catch 1"
Here's a puzzle that trips up everyone the first time. You put a normal Python `print` inside your function to watch it work. You call the `jit`ted function ten times. How many times does it print? The surprising answer: **once.** Not ten. Once. And it prints on the very first call, then never again. Why? Because `print` runs during *tracing* — the one-time blueprint-sketching run — not during the fast compiled runs. The compiled blueprint only recorded your *math* steps (`sin`, `mul`, `add`); a plain `print` isn't math, so it never made it into the plan. A `print` is a [[side effect||Anything a function does besides returning its answer — like printing to the screen, adding to a list, or changing a variable outside itself. jit only records math, so side effects happen only during the one-time trace, not on later runs.]] — something the function does *besides* returning its answer — and side effects only happen during that single trace.

Picture the **stunt double rehearsing a scene, once.** Suppose during rehearsal the stunt double shouts "action!" at one point. The director films the *choreography* — the moves — but the finished movie doesn't include the rehearsal shout, because the shout wasn't part of the recorded moves. It happened once, live, during rehearsal, and never again in the film. A `print` inside `jit` is exactly that rehearsal shout: it fires once during the trace, and the compiled "film" that plays on every later call has no shout in it.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A rehearsal analogy for side effects under jit. On the left, a single rehearsal where the stunt double shouts action once, labelled 'trace: print fires ONCE'. On the right, the finished film playing three times with no shout, labelled 'compiled runs: no print at all'. A label reads: a print is a side effect — it happens only during the one-time trace.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">A print is a side effect: it fires ONCE at trace time, never on later runs</text>
<text x="120" y="42" fill="#9A5A12">trace (1st call): the rehearsal</text>
<g transform="translate(60,52)"><rect x="0" y="0" width="120" height="70" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="60" y="30" fill="#8A6D3B" font-size="10">🔊 print fires</text><text x="60" y="50" fill="#9A5A12" font-size="9">exactly ONCE</text></g>
<text x="205" y="92" fill="#9A5A12" font-size="11">compile →</text>
<text x="405" y="42" fill="#276b45">later calls: the finished film</text>
<g transform="translate(300,52)" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"><rect x="0" y="0" width="56" height="70" rx="6"/><rect x="66" y="0" width="56" height="70" rx="6"/><rect x="132" y="0" width="56" height="70" rx="6"/></g>
<g fill="#276b45" font-size="16"><text x="328" y="94">🔇</text><text x="394" y="94">🔇</text><text x="460" y="94">🔇</text></g>
<text x="260" y="180" fill="#9A938A" font-size="9">the compiled plan recorded your MATH only — the print was never in it</text>
</g></svg>
%%%

**What the rehearsal picture gets right:** the shout is a one-off live event during rehearsal, absent from the recorded film — just as a `print` fires during the single trace and never in the compiled runs. **Where it breaks down:** a film is fixed footage, but the compiled function *does* re-run your real math on new numbers every call — it just re-runs the *math*, never the side effects.

#### See the puzzle, then fix it
Watch the surprise happen: call a `jit`ted function with a `print` inside it three times, and count the prints. Predict how many you'll see:

%%% demo id=printcatch label="how many times does it print?"
code: @jax.jit
code: def f(x):
code:     print("tracing now!")     # a plain Python print — a side effect
code:     return x * 2.0
code: f(jnp.array(1.))    # 1st call: traces → prints
code: f(jnp.array(2.))    # 2nd call: reuses blueprint → silent
code: f(jnp.array(3.))    # 3rd call: reuses blueprint → silent
out: tracing now!
take: <b>Printed ONCE, not three times.</b> The <code>print</code> ran during the single trace (the first call) and never again. This is your #1 clue for "is my function being re-traced?" — if you see the message print, JAX just traced. The plain <code>print</code> is not broken; it's just showing you trace time, not run time.
%%%

The remedy is a purpose-built tool: [[jax.debug.print||The jit-safe way to print. Unlike a plain print, it gets recorded into the blueprint, so it fires on every real run and can show the actual traced values at run time.]]. Unlike a plain `print`, `jax.debug.print` *is* recorded into the blueprint, so it runs on every real call and shows you the true values flowing through. Use `{}` placeholders for the values, just like a format string. Watch it fire on all three calls:

%%% demo id=debugprint label="the fix: jax.debug.print fires every call"
code: @jax.jit
code: def f(x):
code:     jax.debug.print("x is {}", x)   # jit-SAFE print: recorded into the plan
code:     return x * 2.0
code: f(jnp.array(1.)); f(jnp.array(2.)); f(jnp.array(3.))
out: x is 1.0
out: x is 2.0
out: x is 3.0
take: <b>Now it prints every real call, with the true value each time.</b> <code>jax.debug.print</code> is baked into the blueprint, so it runs on every call and can even show the actual traced number (a plain print can't — during tracing there's no number yet). Rule of thumb: <b>plain <code>print</code></b> = "did I trace?"; <b><code>jax.debug.print</code></b> = "what value flowed through on this run?"
%%%

!!! c-warn ⚠️
<b>Failure mode (silent):</b> putting a plain <code>print</code> — or any side effect like <code>my_list.append(x)</code> or changing a global variable — inside a <code>jit</code>ted function and expecting it on every call. <b>Cause:</b> the function body runs on tracers only during the one-time trace, so side effects fire once and are never recorded into the compiled plan. <b>Remedy:</b> use <code>jax.debug.print</code> (or <code>jax.debug.callback</code> / <code>io_callback</code> for richer output) to get run-time printing, and keep <code>jit</code>ted functions <b>pure</b> — no mutation, no hidden state, output depends only on inputs.
!!!

Catch 1 solved — printing. The second catch is the one people hit most while writing real models: using a Python `if` on a value inside `jit`. It comes straight from "tracers have no number," and its fix unlocks a whole family of tools. That's next.

@@@ concept id=c5 tag="Catch 2: branching" title="The if-on-a-value puzzle, and its fixes" gotit="Solved catch 2"
Now the catch that stops everyone at some point. You write a normal Python `if` inside your `jit`ted function — something like `if x > 0: return x else: return -x`. Un-jitted, it works fine. Wrap it in `jit`, and JAX throws an error. Why? Remember from concept 2: during tracing, `x` isn't a number — it's a **tracer**, a stand-in with only a shape and type. A Python `if` needs to look at the value *right now* and decide which branch to take. But there's no value to look at yet, so JAX can't pick a branch. It raises an error (with a name like `ConcretizationTypeError` or `TracerBoolConversionError`) that basically says "you asked me to branch on a value I don't have yet."

Picture a **fork in the road while you're still drawing the map.** You're the mapmaker (JAX, tracing). You reach a fork and your instruction says "turn left *if the river is high*." But you're drawing the map *before* the trip — you don't know today's river level yet. You can't draw a single road, because the answer depends on a fact you won't have until travel day. A plain Python `if` inside `jit` is exactly this: it demands an answer at *map-drawing* time (tracing), but the value only exists at *travel* time (the real run).

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A fork-in-the-road analogy for control flow under jit. In the center, a mapmaker at a fork with a signpost reading 'turn left IF x greater than 0'. A thought bubble shows a tracer with no value, asking 'but I have no number yet!'. Two roads branch left and right. A label reads: a Python if needs a value at trace time, but a tracer has none.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">A Python if needs a value NOW — but a tracer has no number yet</text>
<circle cx="120" cy="120" r="12" fill="#3A342E"/><text x="120" y="150" fill="#6B645E" font-size="8">mapmaker (tracing)</text>
<path d="M132 116 L230 60" stroke="#B8AEA2" stroke-width="8" fill="none"/><path d="M132 124 L230 180" stroke="#B8AEA2" stroke-width="8" fill="none"/>
<text x="250" y="60" fill="#276b45" font-size="9" text-anchor="start">left branch</text>
<text x="250" y="184" fill="#C93B3B" font-size="9" text-anchor="start">right branch</text>
<g transform="translate(160,86)"><rect x="0" y="0" width="120" height="34" rx="4" fill="#FDF3E7" stroke="#C99A12"/><text x="60" y="14" fill="#8A6D3B" font-size="8">signpost:</text><text x="60" y="27" fill="#9A5A12" font-size="8">turn left IF x &gt; 0</text></g>
<g transform="translate(300,80)"><path d="M20 40 q -20 -20 0 -30 q 20 -10 40 0 q 25 5 20 25 q 0 20 -30 18 l -10 12 l -4 -14 q -20 -6 -16 -31 z" fill="#F4F1FB" stroke="#5E5191"/><text x="35" y="24" fill="#5E5191" font-size="7">tracer:</text><text x="35" y="35" fill="#5E5191" font-size="7">no number</text></g>
<text x="260" y="196" fill="#9A938A" font-size="9">JAX can't draw ONE road when the choice depends on a value it doesn't have</text>
</g></svg>
%%%

**What the fork-in-the-road picture gets right:** you're deciding the route while drawing the map, before you know the fact the choice depends on — so you can't commit to one road, exactly as `jit` can't pick a branch on a value it doesn't have during tracing. **Where it breaks down:** a mapmaker could just draw *both* roads and let the traveler choose — and it turns out that's precisely the fix (JAX draws both branches into the plan and picks at run time). More on that in a moment.

There are two clean fixes, and which one you want depends on *what kind* of value you're branching on.

#### Fix A — the value never changes: static_argnums
Sometimes the thing you branch on isn't really data — it's a *setting*. A flag like `training=True`, or a size like `num_layers=3`. These don't change from call to call the way your data does. For those, tell `jit` to treat that argument as a fixed, known-at-trace-time constant using [[static_argnums||A jit setting that marks certain arguments as "static" — their concrete values are baked into the blueprint at trace time, so you CAN branch on them with a normal Python if. Changing a static value makes JAX compile a fresh blueprint.]] (by position) or `static_argnames` (by name). A static argument keeps its real value during tracing, so a Python `if` on it works normally.

%%% demo id=static label="fix A: mark the flag static, then Python if works"
code: from functools import partial
code: @partial(jax.jit, static_argnums=1)   # arg #1 (flag) is static
code: def f(x, flag):
code:     if flag:                # OK now — flag keeps its real value at trace time
code:         return x * 2.0
code:     return x - 1.0
code: print(f(jnp.array(10.), True))    # branches on the real flag value
code: print(f(jnp.array(10.), False))
out: 20.0
out: 9.0
take: <b>The Python if works — because <code>flag</code> is static.</b> Marked static, its true value (True/False) is available at trace time, so JAX can pick the branch and bake it into the plan. The cost: each distinct static value gets its OWN blueprint (True and False compiled separately). Great for settings that rarely change; wrong for values that change every call (that would recompile constantly — the topic of the next concept).
%%%

#### Fix B — the value IS data: lax.cond / lax.select
When the value you branch on really *is* data (like `x` itself, changing every call), you can't make it static. Instead you use JAX's own branching tools that live *inside* the blueprint: [[lax.cond||A jit-compatible if/else. You give it a condition and two functions (a "true" branch and a "false" branch); JAX records BOTH branches into the plan and picks one at run time based on the real value.]] for choosing between two functions, or [[lax.select||A jit-compatible pick-one-of-two for arrays: given a condition, it takes elements from one array where true and the other where false. Like a vectorized if for values.]] for choosing element-by-element between two arrays. These are the mapmaker drawing *both* roads: the plan contains both branches, and the real value picks one at run time.

%%% demo id=laxcond label="fix B: lax.cond branches on real data"
code: from jax import lax
code: @jax.jit
code: def abs_val(x):
code:     # both branches go into the plan; the real value of (x > 0) picks one at run time
code:     return lax.cond(x > 0, lambda v: v, lambda v: -v, x)
code: print(abs_val(jnp.array(5.)))
code: print(abs_val(jnp.array(-3.)))
out: 5.0
out: 3.0
take: <b>Branching on real data, inside jit.</b> <code>lax.cond</code> recorded BOTH the "keep it" and "negate it" branches into the blueprint; the actual value of <code>x &gt; 0</code> chose one at run time. No static needed — this works even when <code>x</code> changes every call. (For picking element-wise between two whole arrays, <code>lax.select</code> does the same job.)
%%%

!!! c-warn ⚠️
<b>Failure mode:</b> a plain Python <code>if</code>, <code>while</code>, or <code>for</code> whose condition depends on a <b>traced value</b> raises a <code>ConcretizationTypeError</code> / <code>TracerBoolConversionError</code>. <b>Cause:</b> the tracer has no concrete number at trace time, so Python can't evaluate the condition to pick a branch. <b>Remedy:</b> if the value is really a <b>setting</b> (a flag or size), mark it <code>static_argnums</code>/<code>static_argnames</code> so its real value is baked in; if the value is really <b>data</b>, branch with <code>lax.cond</code> (choose a function) or <code>lax.select</code> (choose array elements), which record both branches into the plan.
!!!

You've now beaten both famous catches. There's one last surprise — a *performance* trap rather than an error — that comes from how `jit` decides whether it can reuse a blueprint. Understanding it makes you fast for good. That's the final mechanism.

@@@ concept id=c6 tag="Catch 3: recompiling" title="One blueprint per shape: don't recompile by surprise" gotit="Solved catch 3"
Here's the last surprise, and it's sneaky because it's not an error — it's a *slowdown*. Recall from concept 2 that the blueprint records the *shape* and *type* of your inputs, but not their values. So a blueprint made for a `float32` array of shape `(5,)` is only valid for that exact shape and type. If you call the function with a shape `(6,)` array, the old plan doesn't fit — so JAX **traces and compiles a whole new blueprint** from scratch. Do that on every call (say, feeding a different-length batch each time) and you pay the slow compile cost over and over, never enjoying the speed. That repeated re-compiling is the trap called [[recompilation||When jit throws away the cached blueprint and compiles a fresh one because the inputs' shape, dtype, or a static-argument value changed. Frequent recompilation erases jit's speed benefit.]].

What decides whether JAX can reuse a blueprint? A [[cache key||The label jit uses to look up a saved blueprint. It's built from the SHAPE and DTYPE (type) of each array argument, plus the concrete VALUES of any static arguments. Same key → reuse the fast plan; new key → compile a fresh one.]]. Think of it as a **shoe-shelf at a bowling alley.** Each pair of shoes is a compiled blueprint, sorted by *size*. Ask for size 9 and they've made a pair before — instant, they hand you the ready pair (reuse). Ask for a size they've never seen, and they go to the back to make a new pair (a fresh compile). The "size" here is `(shape, dtype)` of your arrays plus any static-argument values. Same size on the shelf → instant reuse. A brand-new size every time → the poor clerk is always making new shoes, and you're always waiting.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A shoe-shelf analogy for the jit compilation cache. A shelf holds three ready shoe pairs labelled by size: shape (5,) float32, shape (6,) float32, shape (5,) int32. A customer asking for a size already on the shelf gets instant reuse; a customer asking for a brand-new size sends the clerk to the back to make a new pair. A label reads: the cache key is (shape, dtype) plus static-arg values.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Cache key = (shape, dtype) + static values — one blueprint per key</text>
<g transform="translate(30,40)"><rect x="0" y="0" width="270" height="96" rx="8" fill="#EDE9E0" stroke="#B8AEA2" stroke-width="2"/><text x="135" y="16" fill="#3A342E" font-size="9">the shelf of ready blueprints (the cache)</text><g fill="#EAF5EE" stroke="#2D8B55"><rect x="14" y="28" width="76" height="52" rx="4"/><rect x="98" y="28" width="76" height="52" rx="4"/><rect x="182" y="28" width="76" height="52" rx="4"/></g><g fill="#1a5c38" font-size="8"><text x="52" y="50">f32[5]</text><text x="52" y="66">✔ ready</text><text x="136" y="50">f32[6]</text><text x="136" y="66">✔ ready</text><text x="220" y="50">i32[5]</text><text x="220" y="66">✔ ready</text></g></g>
<text x="400" y="48" fill="#276b45">same key → instant reuse</text>
<g transform="translate(330,58)"><circle cx="20" cy="18" r="10" fill="#3A342E"/><path d="M35 18 h60" stroke="#2D8B55" stroke-width="2" marker-end="url(#a6)"/></g>
<defs><marker id="a6" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="#2D8B55"/></marker></defs>
<text x="400" y="108" fill="#C93B3B">new key → clerk makes a new pair (recompile)</text>
<g transform="translate(330,118)"><circle cx="20" cy="18" r="10" fill="#3A342E"/><path d="M35 18 h60" stroke="#C93B3B" stroke-width="2" stroke-dasharray="4 3" marker-end="url(#a6)"/></g>
<text x="260" y="192" fill="#9A938A" font-size="9">changing shape, dtype, or a static value = a new size = a fresh (slow) compile</text>
</g></svg>
%%%

**What the shoe-shelf picture gets right:** a size you've asked for before is handed over instantly, while a never-seen size makes the clerk build a new pair — exactly how `jit` reuses a blueprint for a known `(shape, dtype)` but compiles a fresh one for a new one. **Where it breaks down:** a shoe shelf runs out of space, but JAX keeps every distinct blueprint it made; the real cost isn't storage, it's the *time* spent compiling each new one.

#### Watch a recompile happen
Use the plain-`print` clue from concept 4: the message prints every time JAX *traces*. Call with the same shape twice (reuse — no print), then a new shape (recompile — prints again). Predict how many times "tracing!" appears:

%%% demo id=recompile label="count the traces as the shape changes"
code: @jax.jit
code: def f(x):
code:     print("tracing!")           # fires ONLY when JAX (re)traces
code:     return jnp.sum(x)
code: f(jnp.ones((5,)))     # new shape (5,)  → traces
code: f(jnp.ones((5,)))     # same shape (5,) → reuse, no trace
code: f(jnp.ones((6,)))     # new shape (6,)  → traces AGAIN
code: f(jnp.ones((5,)))     # shape (5,) again → still cached, no trace
out: tracing!
out: tracing!
take: <b>Traced twice, not four times.</b> Shape <code>(5,)</code> compiled once and was reused (the repeat calls were silent). Shape <code>(6,)</code> was a new cache key, so JAX built a fresh blueprint (a second "tracing!"). The later <code>(5,)</code> call reused the first plan again. Lesson: <b>keep your shapes stable</b> and you compile once; feed a new shape every call and you recompile every call.
%%%

#### The practical rule — and the hard limit
Two takeaways you'll use forever. **First, the fix:** keep input shapes and dtypes stable across calls. In real training that's why batches are a fixed size, and why variable-length data (like sentences) gets **padded** to a fixed length — so every call hits the same cache key and reuses one blueprint. **Second, the honest limit:** because the blueprint is fixed to input shapes, `jit` cannot handle a function whose *output shape depends on the data's values* (like "return only the positive elements" — the count isn't knowable at trace time), nor data-dependent Python loops of unknown length. The output shapes must be figurable-out from the input shapes at trace time. When you genuinely need value-dependent loops, JAX has structured tools (`lax.scan`, `lax.fori_loop`) — you'll meet those on a later day.

!!! c-warn ⚠️
<b>Failure mode (silent slowdown):</b> calling a <code>jit</code>ted function with a new input shape, dtype, or static-argument value on many calls, so JAX recompiles constantly and you never enjoy the speed. <b>Cause:</b> each distinct <code>(shape, dtype)</code> + static-value combination is a new cache key that triggers a fresh trace + compile. <b>Remedy:</b> keep shapes and dtypes stable (pad variable-length inputs to a fixed size), avoid marking rapidly-changing values as static, and use a plain <code>print</code> as a trace-detector to catch surprise recompiles.
!!!

That's the full mechanism — trace, compile, cache, and the three catches. Time to gather it all into one page you can keep.

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — you now think the JAX way about speed: **trace once into a blueprint, then run it forever.** Before we gather the pieces, here's one fresh everyday picture for the whole day. Think of `jit` like **learning a dance routine from a choreographer.** The first run-through is slow: the choreographer (JAX) watches you move once and writes down every step (tracing → the blueprint), then a coach tightens it into a smooth combined routine (XLA fusion). After that, you perform the routine fast, every time. But: the routine only recorded the *dance steps* — if you shouted something during rehearsal, it's not in the performance (side effects fire once, catch 1); you can't decide a step "if the music is loud" mid-performance because the routine was fixed beforehand (no Python `if` on data, catch 2); and a routine choreographed for a 5-person stage has to be re-learned for a 6-person stage (one blueprint per shape, catch 3). **Where the dance picture breaks down:** a dancer gets tired, but a compiled function runs the routine at the same full speed forever. Below is the day step by step, plus a cheat-sheet to keep on your desk.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A four step flow for the day. Step one: wrap with jit. Step two: first call traces on stand-ins into a blueprint (jaxpr). Step three: XLA compiles and fuses the blueprint. Step four: every later call reuses the fast compiled version. Below, three catch labels: print fires once, no Python if on data, one blueprint per shape. Each step points to the next.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The whole day: wrap → trace → compile+fuse → reuse fast (mind the 3 catches)</text>
<g transform="translate(6,44)"><rect x="0" y="0" width="112" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="56" y="24" fill="#1a5c38">1 wrap in jit</text><text x="56" y="40" fill="#276b45" font-size="8">jax.jit(f)</text></g>
<text x="122" y="76" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(138,44)"><rect x="0" y="0" width="112" height="56" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"/><text x="56" y="24" fill="#5E5191">2 trace once</text><text x="56" y="40" fill="#5E5191" font-size="8">stand-ins → jaxpr</text></g>
<text x="254" y="76" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(270,44)"><rect x="0" y="0" width="112" height="56" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"/><text x="56" y="24" fill="#8A6D3B">3 XLA compiles</text><text x="56" y="40" fill="#9A5A12" font-size="8">fuse the steps</text></g>
<text x="386" y="76" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(402,44)"><rect x="0" y="0" width="112" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="56" y="24" fill="#1a5c38">4 reuse fast</text><text x="56" y="40" fill="#276b45" font-size="8">every later call</text></g>
<text x="260" y="128" fill="#8a3b3b" font-size="9">catches: 🔊 print fires ONCE · 🍴 no Python if on data · 👟 one blueprint per shape</text>
<text x="260" y="170" fill="#9A938A" font-size="9">pay the compile cost once, run the fast plan forever — keep functions pure and shapes stable</text>
</g></svg>
%%%

The day in a few beats:
- **Wrap and go.** Wrap your function with [[jit||just-in-time compilation: trace once into a blueprint, compile it, reuse it fast]] (`jax.jit(f)` or `@jax.jit`). First call = trace + compile (slow, once); every later call = run the compiled plan (fast).
- **Tracing on stand-ins.** The first call runs your function on [[tracer||a stand-in with shape + type but no real number]] stand-ins, recording each step into a [[jaxpr||the recorded blueprint of operations; view it with jax.make_jaxpr]]. No real numbers are in the plan — only shapes, types, and steps.
- **Why it's fast.** XLA reads the blueprint and does [[fusion||gluing neighboring operations into one combined pass over the data]] — one planned route instead of many little errands. Time honestly: warm up once, and call `block_until_ready()`.
- **Catch 1 — printing.** A plain `print` is a side effect: it fires once at trace time, never on later runs. Fix: `jax.debug.print`. Keep `jit`ted functions **pure**.
- **Catch 2 — branching.** A Python `if` on a *traced value* errors (no number at trace time). Fix: `static_argnums` for settings/flags; `lax.cond` / `lax.select` for value-dependent branching.
- **Catch 3 — recompiling.** The [[cache key||(shape, dtype) of array args + values of static args]] is `(shape, dtype)` + static values. A new key = a fresh (slow) compile. Fix: keep shapes stable, pad variable-length inputs. Limit: output shapes must be inferable at trace time.

The one rule to memorize: **jit traces once per unique input shape+type; keep those stable and stay pure, and you compile once and fly.**

#### Cheat-sheet · the moves at a glance
%%% table
:: You want to… :: Do this :: Note
Compile a function :: `fast = jax.jit(f)` or `@jax.jit` :: 1st call traces+compiles; rest reuse
See the blueprint :: `jax.make_jaxpr(f)(x)` :: prints the recorded jaxpr (no compile)
Time it honestly :: `fast(x).block_until_ready()` :: warm up once first; JAX is async
Print at run time :: `jax.debug.print("v={}", v)` :: plain `print` fires only at trace time
Branch on a flag/setting :: `jax.jit(f, static_argnums=…)` :: bakes the value in; recompiles per value
Branch on real data :: `lax.cond(pred, t_fn, f_fn, x)` :: both branches in the plan; `lax.select` for arrays
Avoid recompiles :: keep shapes/dtypes stable; pad :: each new `(shape, dtype)` = a fresh compile
%%%

#### Cheat-sheet · the words you met today
%%% jargon
jit | just-in-time compilation: trace a function once into a blueprint, compile it, reuse it fast
tracing | the first-call run on stand-in values that records the blueprint
tracer | a stand-in value with shape + type but no concrete number
jaxpr | the recorded blueprint of operations; view it with jax.make_jaxpr
XLA | the compiler that turns the jaxpr into fast machine code
fusion | XLA gluing neighboring operations into one combined pass — the main speed win
eager execution | the plain, un-jitted way: one operation at a time, many little trips
block_until_ready | wait for the real result; needed for honest timing (JAX is asynchronous)
side effect | anything besides returning the answer (print, append, mutation); runs only at trace time
jax.debug.print | the jit-safe print, recorded into the plan, fires on every real call
static_argnums | mark an argument static so its real value is baked in — lets you Python-if on it
lax.cond / lax.select | jit-safe branching on real data: both branches recorded, one picked at run time
cache key | (shape, dtype) of array args + static-arg values; a new key triggers a fresh compile
recompilation | rebuilding the blueprint because the shape/dtype/static value changed
%%%

That's the whole day. You now wrap a function in `jit`, let JAX trace it once into a blueprint, hand it to XLA to compile and fuse, and reuse that fast version on every call — while sidestepping the three catches: side effects fire once, no Python `if` on data, and one blueprint per shape. Next you'll meet **Flax & Optax** — the libraries that use these transforms to build and train real neural networks.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does jit do the FIRST time you call the wrapped function? | a:2 | It runs many machines in parallel | It skips the function entirely | It traces the function into a blueprint and compiles it (a one-time cost), then runs it | It prints a warning | fb: First call = trace + compile (slow, once); every later call reuses the compiled blueprint (fast). That's the whole "trace once, run fast" idea.
q: During tracing, what is inside your function's variables? | a:1 | The real input numbers | Stand-ins (tracers) that carry only shape and type, no concrete value | Random noise | Nothing — the function is skipped | fb: Tracers are stand-ins with shape + dtype but NO number. That's why a Python `if x > 0` breaks inside jit — there's no value to compare yet.
q: You put a plain Python print inside a jitted function and call it 5 times. How many times does it print? | a:0 | Once, during the first-call trace | Five times, once per call | Zero times | Only on the last call | fb: A plain print is a side effect — it runs during the single trace, not on the compiled runs. Use jax.debug.print to print on every real call.
q: Your jitted function recompiles on almost every call and feels slow. Most likely cause? | a:3 | You used jax.jit twice | The function returns too many values | You forgot to import jax | The input SHAPE (or dtype) changes each call, so each is a new cache key = a fresh compile | fb: The cache key is (shape, dtype) + static-arg values. A new shape every call = a new blueprint every call. Keep shapes stable (pad variable-length inputs) to compile once.
%%%

@@@ produce id=produce tag="Produce" title="Feel jit with your own hands" gotit="Done"
Time to feel today's rule for yourself. You'll wrap a function in `jit`, watch it trace, time the speed-up, and trigger each of the three catches on purpose so you recognize them for good. **Predict first:** when you call a `jit`ted function (with a plain `print` inside) three times with the *same* shape, how many times will it print? Then run it and **watch** the answer. Pick one path.

#### Option A · write it yourself
Create `sessions/m07-thinking-in-jax/day-04-jit/experiment.py`. Start with `import jax`, `import jax.numpy as jnp`, and `import time`. **Print what you observe at each step.** (1) Define `f(x) = jnp.sin(x) + jnp.cos(x) * 2.0 - jnp.tanh(x)`, wrap it with `jax.jit`, and **peek** at the blueprint with `jax.make_jaxpr(f)(jnp.arange(3.))`. (2) Time eager `f` versus the `jit`ted version on a big array (e.g. `jnp.ones((1_000_000,))`) — warm up once, and call `.block_until_ready()` inside a 100-call loop — and **observe** the `jit` version is faster. (3) Catch 1: put a plain `print("tracing!")` inside a `jit`ted function, call it 3 times with the same shape, and **notice** it prints only once. (4) Catch 2: try a plain `if x > 0` inside `jit` (see the error), then fix it two ways — `static_argnums` for a flag, and `lax.cond` for data. (5) Catch 3: call a `jit`ted function (with a `print` inside) on shapes `(5,)`, `(5,)`, `(6,)`, `(5,)` and **notice** it traces twice. Run with `python3 sessions/m07-thinking-in-jax/day-04-jit/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Week 1 Day 4 (JAX jit) artifact.

Create sessions/m07-thinking-in-jax/day-04-jit/experiment.py that, with a comment on each step and printing what it observes:
1. import jax, jax.numpy as jnp, time. Define f(x) = jnp.sin(x) + jnp.cos(x) * 2.0 - jnp.tanh(x); wrap fast_f = jax.jit(f); print jax.make_jaxpr(f)(jnp.arange(3.)) to show the recorded blueprint (jaxpr).
2. Time eager f vs fast_f on x = jnp.ones((1_000_000,)): warm up fast_f once with .block_until_ready(); then loop 100 calls each calling .block_until_ready(), and print both timings in ms — the jit version should be faster (fusion).
3. Catch 1 (side effects): define a jitted g(x) with a plain print("tracing!") inside; call it 3 times with the same shape; print a note that it printed only ONCE (side effects fire at trace time). Then show jax.debug.print firing on every call.
4. Catch 2 (control flow): show that a plain `if x > 0` inside jit raises an error (wrap in try/except and print the error type); then fix it two ways — (a) static_argnums for a boolean flag argument, (b) lax.cond(x > 0, lambda v: v, lambda v: -v, x) for data — print results for both.
5. Catch 3 (recompilation): define a jitted h(x) with print("tracing!") inside returning jnp.sum(x); call it on shapes (5,), (5,), (6,), (5,); print a note that "tracing!" appeared TWICE because the cache key is (shape, dtype).
Then run it and paste the output at the bottom as a comment. Explain in one line why the first call is slow but later calls (same shape) are fast.
%%%

#### What you should see (trace once, run fast — mind the catches)
- `jax.make_jaxpr(f)(...)` prints a little program of steps (`sin`, `cos`, `mul`, …) with shapes and types but **no real numbers** — that's the blueprint.
- The `jit`ted timing is **faster** than eager on the big array — XLA fused the steps into one pass (and `block_until_ready()` made the timing honest).
- The plain `print` inside `jit` fires **once** (at trace time), while `jax.debug.print` fires on **every** call — catch 1.
- A plain `if x > 0` on data **errors**; `static_argnums` (for a flag) and `lax.cond` (for data) both fix it — catch 2.
- Calling on shapes `(5,), (5,), (6,), (5,)` traces **twice** — a new shape is a new cache key — catch 3.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m07-thinking-in-jax/day-04-jit/log.md`: (1) in one sentence, the jit mental model — what happens on the first call versus every later call; (2) why a plain `print` inside `jit` fires only once, and what to use instead; (3) in one sentence, what the compilation cache key is and why changing your input shape every call makes `jit` slow.

@@@ fin
