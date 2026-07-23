---
quest_id: w01-d03-vmap
mode: concept
donor: v9-base.donor
page_title: "Week 1 · Day 3 — JAX vmap: Automatic Vectorization"
module_label: "Thinking in JAX · Week 1 · Day 3"
title: "vmap: Automatic Vectorization"
subtitle: "Write It for One, Run It on a Whole Batch"
brand_sub: "Frontier Lab · Week 1 Day 3"
spine: "cookie-cutter"
nav_prev_href: "../day-02-prng-keys/lesson.html"
nav_prev_label: "Randomness: Explicit Keys"
nav_next_href: "../day-04-jit/lesson.html"
nav_next_label: "jit: Compilation"
fin_title: "Week 1 · Day 3 complete! 🏆"
fin_body: "Great — you've unlocked <b>vmap</b>: write your function for a <i>single</i> example, then let JAX stamp it across a whole batch like a <b>cookie-cutter</b> — no hand-written loop, no reshaping. You pick which axis to map with <code>in_axes</code> and where the batch lands with <code>out_axes</code>. It's faster and far cleaner than a Python for-loop.<br>Next up: <b>jit: Compilation</b>."
notebook_yardstick: null
---

@@@ hero
@lede Picture rolling out a flat sheet of cookie dough on the kitchen table. You have one little metal shape — a star — and you press it into the dough over and over: star, star, star, all the way down the sheet. You didn't carve each star by hand. You made **one** shape and *stamped* it across the whole sheet. That is the exact feeling of **vmap**, a tool in **JAX** — the library behind many of today's biggest AI models, including work at Google DeepMind. In JAX you often write a little function for just **one** thing — one picture, one row of numbers, one example. Then you hand it to `vmap`, and JAX presses that same function across a **whole batch** of things at once, like a cookie-cutter stamping the whole sheet. No hand-written loop. No fiddling with shapes. And here's the surprise that makes it a superpower: the stamped-all-at-once version runs *far faster* on the chips that train AI than doing them one at a time — which is why real training code runs a model over thousands of examples this way.
@goal Together we'll meet <b>vmap</b> — the cookie-cutter that turns a one-example function into a whole-batch function. You'll learn to pick which axis is the batch with <code>in_axes</code>, mark a shared ingredient as "don't stamp this" with <code>in_axes=None</code>, place the batch in the output with <code>out_axes</code>, and see why one stamped-together computation beats a slow hand-written loop. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="Write for one" title="Write it for ONE, run it on MANY" gotit="Got the big idea"
Let's meet the star of the day. The big idea of [[vmap||A JAX tool that takes a function you wrote for one example and turns it into a function that runs over a whole batch of examples at once — "vectorizing map." You never write the loop yourself.]] is almost too simple to believe: you write your function for **one** example — one row of numbers, one picture — and then let JAX run it over a **whole batch** of examples for you. A [[batch||A stack of many examples, stored together in one array. If one example is a row of 4 numbers, a batch of 5 is a 5-by-4 block of numbers.]] just means "many examples stacked together in one array."

Here's the mental picture, and it's our home base for the whole day: **a cookie-cutter pressing a sheet of dough.** You make one little shape (your function for one example). Then you stamp it across the whole sheet (the batch). Every stamp is the same shape, done in one smooth sweep — not carved one by one.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A cookie-cutter analogy for vmap. On the left, one star-shaped cookie-cutter labelled 'your function for ONE example'. An arrow labelled vmap points right. On the right, a flat sheet of dough with five identical stars stamped across it in a row, labelled 'run on the WHOLE batch at once'. A label at the bottom reads: one shape, stamped across the whole sheet."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">vmap = a cookie-cutter: one shape, stamped across the whole sheet</text><text x="75" y="46" fill="#6B645E">your function for ONE</text><g transform="translate(45,54)"><path d="M35 2 l7 20 l21 1 l-16 14 l6 21 l-18 -12 l-18 12 l6 -21 l-16 -14 l21 -1 z" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/></g><text x="150" y="88" fill="#9A5A12" font-size="11">vmap →</text><text x="390" y="46" fill="#276b45">run on the WHOLE batch</text><g transform="translate(210,58)"><rect x="0" y="0" width="290" height="70" rx="10" fill="#EDE9E0" stroke="#B8AEA2" stroke-width="2"/><g fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"><path d="M28 8 l5 15 l16 1 l-12 10 l4 16 l-13 -9 l-13 9 l4 -16 l-12 -10 l16 -1 z"/><path d="M85 8 l5 15 l16 1 l-12 10 l4 16 l-13 -9 l-13 9 l4 -16 l-12 -10 l16 -1 z"/><path d="M142 8 l5 15 l16 1 l-12 10 l4 16 l-13 -9 l-13 9 l4 -16 l-12 -10 l16 -1 z"/><path d="M199 8 l5 15 l16 1 l-12 10 l4 16 l-13 -9 l-13 9 l4 -16 l-12 -10 l16 -1 z"/><path d="M256 8 l5 15 l16 1 l-12 10 l4 16 l-13 -9 l-13 9 l4 -16 l-12 -10 l16 -1 z"/></g></g><text x="260" y="172" fill="#9A938A" font-size="9">you write ONE function; vmap presses it across the batch — no loop by hand</text></g></svg>
%%%

**What the cookie-cutter picture gets right:** you build one shape and reuse it across the whole sheet, and every stamp is identical work — just like `vmap` runs your *one* function on every example in the batch. **Where it breaks down:** a cookie-cutter still presses each cookie one after another with your hand, but `vmap` does something better — it turns all the stamps into a *single* combined action the computer runs together, which is why it's so fast (we'll see that in the last concept).

#### The one habit: think about a single example
This is the whole mindset shift. Before `vmap`, you'd think "I have 50 pictures, how do I loop over all 50?" With `vmap` you think about **one** picture, write the function for that, and stop worrying about the 50. Say your function scales one vector by a weight:

%%% demo id=one label="a function written for ONE example"
code: import jax, jax.numpy as jnp
code: def scale(x):        # x is ONE vector, e.g. [1, 2, 3]
code:     return x * 2.0
code: print(scale(jnp.array([1., 2., 3.])))
out: [2. 4. 6.]
take: <b>This function only knows about ONE example.</b> It takes a single vector and doubles it. It has no idea a "batch" exists — and that's exactly the point. You write the simple one-example version, and next you'll hand it to <code>vmap</code> to run on many.
%%%

You've got the big idea. Now the natural question: when I hand my batch to `vmap`, how does it know *which* part of the array is the "many examples" and which part is one example? That's the next concept — and it's just one little word: `in_axes`.

@@@ concept id=c2 tag="Which axis?" title="in_axes: which axis is the batch" gotit="Got in_axes"
Now let's actually stamp the cookie-cutter across a batch. To do that, `vmap` needs one small piece of information: *which direction* in your array holds the many examples. That direction is called the [[batch axis||The direction in an array that runs across the different examples. If a 5-by-4 array holds 5 examples of 4 numbers each, the batch axis is the first one (the "5" side) — it counts the examples.]]. You tell `vmap` which axis it is with a setting called [[in_axes||A setting you give vmap that says which axis of each input is the batch axis to stamp across. in_axes=0 means "the first axis is the batch."]].

Think of your batch as a **stack of pancakes**. The whole stack is your array. Each pancake is one example. The batch axis is the "up the stack" direction — the one that counts the pancakes. `in_axes` just tells the cookie-cutter: "stamp once per pancake, going up the stack." By default `vmap` uses `in_axes=0`, meaning **the first axis is the stack** — the most common way to store a batch.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A stack of pancakes analogy for the batch axis. On the left, five pancakes drawn stacked on top of each other, with an upward arrow labelled 'batch axis (axis 0): counts the examples'. On the right, each single pancake is labelled 'one example: one row of numbers'. A label reads: in_axes=0 means stamp once per pancake, going up the stack."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">in_axes=0: the FIRST axis is the stack — stamp once per pancake</text><g transform="translate(60,42)"><ellipse cx="70" cy="96" rx="70" ry="12" fill="#F0D9A8" stroke="#C99A12"/><ellipse cx="70" cy="76" rx="70" ry="12" fill="#F7E4B8" stroke="#C99A12"/><ellipse cx="70" cy="56" rx="70" ry="12" fill="#F0D9A8" stroke="#C99A12"/><ellipse cx="70" cy="36" rx="70" ry="12" fill="#F7E4B8" stroke="#C99A12"/><ellipse cx="70" cy="16" rx="70" ry="12" fill="#F0D9A8" stroke="#C99A12"/></g><path d="M20 138 v-118" stroke="#2D8B55" stroke-width="2" marker-end="url(#up)"/><defs><marker id="up" markerWidth="7" markerHeight="7" refX="3.5" refY="1" orient="auto"><path d="M0 6 L3.5 0 L7 6 z" fill="#2D8B55"/></marker></defs><text x="205" y="70" fill="#1a5c38" font-size="9" text-anchor="start">← batch axis (axis 0)</text><text x="205" y="86" fill="#276b45" font-size="9" text-anchor="start">counts the examples</text><g transform="translate(330,60)"><ellipse cx="90" cy="14" rx="88" ry="13" fill="#F7E4B8" stroke="#C99A12" stroke-width="1.5"/><text x="90" y="18" fill="#8A6D3B" font-size="9">one pancake = one example</text></g><text x="420" y="105" fill="#9A938A" font-size="9">one row of numbers</text><text x="260" y="180" fill="#9A938A" font-size="9">vmap stamps your one-example function once per pancake, up the stack</text></g></svg>
%%%

**What the pancake-stack picture gets right:** the batch axis is the one direction that counts the separate examples, and `in_axes` points `vmap` at it. **Where it breaks down:** real pancakes have to be flipped one at a time, but `vmap` handles all "pancakes" together — the stack is really the input array, and `in_axes` is just a label saying which axis to slice along.

#### The shape story: mapping over an axis of size N
Here is the single most useful thing to hold onto about `vmap` — what it does to the **shape** of your data. Your one-example function takes a shape and returns a shape. When you `vmap` over a batch axis of size `N`, `vmap` slides an `N` onto the front of the input and onto the output. So a function that turned one vector into one vector now turns `N` vectors into `N` vectors — in one call.

%%% formula
expr: vmap(f, in_axes=0) : shape (N, ...) → shape (N, ...)
note: `f` was written for ONE example (no N). vmap maps over the size-N batch axis and hands you back N results, stacked.
%%%

Let's watch the shape change happen. Predict it first: our `scale` function doubles one vector of 3 numbers. If we hand it a batch of **5** such vectors (shape `(5, 3)`), what shape comes back?

%%% demo id=inaxes label="predict the output shape, then run"
code: def scale(x):  return x * 2.0        # written for ONE vector
code: batch = jnp.ones((5, 3))            # 5 examples, each a vector of 3
code: batched_scale = jax.vmap(scale)     # in_axes=0 by default
code: out = batched_scale(batch)
code: print("in :", batch.shape)
code: print("out:", out.shape)
out: in : (5, 3)
out: out: (5, 3)
take: <b>Same function, now runs on all 5 rows.</b> <code>vmap</code> saw the batch axis of size 5 (axis 0, the default) and stamped <code>scale</code> once per row. You wrote it for ONE vector; you got back all 5 doubled — no loop. The 5 rode along on the front the whole time.
%%%

Now let's *see* that shape transformation, not just read it. Below, one row goes into the plain function and one row comes out. `vmap` wraps that so a whole 5-row block goes in and a 5-row block comes out — the `5` slides straight through:

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="A before and after of vmap's shape transformation. On the top, labelled 'plain scale', a single row of 3 boxes goes into a function box and a single row of 3 boxes comes out. On the bottom, labelled 'vmap(scale)', a 5-by-3 grid goes into the function and a 5-by-3 grid comes out, with the number 5 highlighted as 'the batch rides along the front'. A label reads: mapping over an axis of size N turns shape (3,) into (5, 3)."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Shape contract: vmap slides the batch N onto the front (in AND out)</text><text x="60" y="40" fill="#6B645E" font-size="10">plain scale — shape (3,)</text><g transform="translate(30,48)"><rect x="0" y="0" width="18" height="18" fill="#EAF5EE" stroke="#2D8B55"/><rect x="18" y="0" width="18" height="18" fill="#EAF5EE" stroke="#2D8B55"/><rect x="36" y="0" width="18" height="18" fill="#EAF5EE" stroke="#2D8B55"/></g><text x="110" y="61" fill="#9A5A12" font-size="11">→ f →</text><g transform="translate(150,48)"><rect x="0" y="0" width="18" height="18" fill="#FDF3E7" stroke="#C99A12"/><rect x="18" y="0" width="18" height="18" fill="#FDF3E7" stroke="#C99A12"/><rect x="36" y="0" width="18" height="18" fill="#FDF3E7" stroke="#C99A12"/></g><text x="245" y="61" fill="#9A938A" font-size="9" text-anchor="start">one row in → one row out</text><text x="90" y="102" fill="#276b45" font-size="10">vmap(scale) — shape (5, 3)</text><g transform="translate(30,110)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="18" height="14"/><rect x="18" y="0" width="18" height="14"/><rect x="36" y="0" width="18" height="14"/><rect x="0" y="14" width="18" height="14"/><rect x="18" y="14" width="18" height="14"/><rect x="36" y="14" width="18" height="14"/><rect x="0" y="28" width="18" height="14"/><rect x="18" y="28" width="18" height="14"/><rect x="36" y="28" width="18" height="14"/><rect x="0" y="42" width="18" height="14"/><rect x="18" y="42" width="18" height="14"/><rect x="36" y="42" width="18" height="14"/><rect x="0" y="56" width="18" height="14"/><rect x="18" y="56" width="18" height="14"/><rect x="36" y="56" width="18" height="14"/></g><text x="4" y="185" fill="#2D8B55" font-size="9">↑ the 5 (batch) rides the front</text><text x="110" y="140" fill="#9A5A12" font-size="11">→ f →</text><g transform="translate(150,110)" fill="#FDF3E7" stroke="#C99A12"><rect x="0" y="0" width="18" height="14"/><rect x="18" y="0" width="18" height="14"/><rect x="36" y="0" width="18" height="14"/><rect x="0" y="14" width="18" height="14"/><rect x="18" y="14" width="18" height="14"/><rect x="36" y="14" width="18" height="14"/><rect x="0" y="28" width="18" height="14"/><rect x="18" y="28" width="18" height="14"/><rect x="36" y="28" width="18" height="14"/><rect x="0" y="42" width="18" height="14"/><rect x="18" y="42" width="18" height="14"/><rect x="36" y="42" width="18" height="14"/><rect x="0" y="56" width="18" height="14"/><rect x="18" y="56" width="18" height="14"/><rect x="36" y="56" width="18" height="14"/></g><text x="245" y="140" fill="#9A938A" font-size="9" text-anchor="start">whole block in → whole block out</text></g></svg>
%%%

That shape rule — **map over an axis of size `N` and an `N` appears on the front of the result** — is the thing you'll lean on every single time you use `vmap`. But notice we assumed the batch was the *first* axis. What if your data doesn't want the batch first? And what if one input, like a shared weight, should *not* be stamped at all? Those two questions are the next two concepts.

@@@ concept id=c3 tag="Shared ingredient" title="in_axes=None: don't stamp the shared thing" gotit="Got None"
Real functions usually take more than one input. A common one: your function takes an **example** *and* a **weight** that every example shares. Think of a model where the same weights are used for all 5 pictures. Here's the trap: by default, `vmap` tries to stamp *every* input across the batch. It will happily try to slice your shared weight into 5 pieces too — and either crash or, worse, quietly do the wrong thing. You don't want the weight stamped. You want it **shared**, whole, on every stamp.

Back to the cookie-cutter. The dough is your batch — that's what you stamp across. But the **flour you sprinkle on the counter** is shared: you don't cut the flour into pieces, you use the *same* flour under every cookie. `in_axes=None` is you telling `vmap`: "this ingredient is shared flour — hand it whole to every stamp, don't slice it." The value `None` literally means "there is no batch axis here; broadcast this same value to all."

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A cookie-cutter analogy for in_axes=None. On the left, a stack of dough labelled 'x: the batch, in_axes=0, gets sliced per example'. On the right, a bag of flour labelled 'w: shared, in_axes=None, used whole on every stamp'. Arrows from both meet at three cookie stamps below, each one showing 'one dough slice + the SAME whole flour'. A label reads: in_axes=None means share this value whole across every stamp."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">in_axes=(0, None): slice the dough per example, share the flour whole</text><g transform="translate(50,40)"><rect x="0" y="0" width="90" height="52" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="45" y="22" fill="#1a5c38" font-size="10">x = batch</text><text x="45" y="40" fill="#276b45" font-size="8">in_axes=0 (sliced)</text></g><g transform="translate(380,40)"><rect x="0" y="0" width="90" height="52" rx="6" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="45" y="22" fill="#8A6D3B" font-size="10">w = shared</text><text x="45" y="40" fill="#9A5A12" font-size="8">in_axes=None (whole)</text></g><path d="M95 92 L200 118" stroke="#2D8B55" stroke-width="1.5"/><path d="M425 92 L320 118" stroke="#C99A12" stroke-width="1.5"/><g transform="translate(160,120)"><rect x="0" y="0" width="200" height="50" rx="6" fill="#EDE9E0" stroke="#B8AEA2" stroke-width="1.5"/><text x="100" y="20" fill="#3A342E" font-size="9">each stamp: one dough slice</text><text x="100" y="36" fill="#3A342E" font-size="9">+ the SAME whole flour</text></g><text x="260" y="188" fill="#9A938A" font-size="9">None = "no batch axis here" → broadcast this value to every mapped call</text></g></svg>
%%%

**What the shared-flour picture gets right:** some inputs get sliced per example (the dough) and some are used whole on every stamp (the flour), and `in_axes` lets you say which is which. **Where it breaks down:** flour physically runs out as you bake, but a shared JAX weight isn't used up — the *same* array is handed to all `N` calls at once, not spooned out.

#### Per-argument in_axes: one setting per input
When your function takes several inputs, you pass `in_axes` as a **tuple** — one entry per argument, in order. Each entry is either an integer (which axis is that argument's batch) or `None` (this one is shared). This per-argument control is exactly how you keep the weight whole while still mapping over the examples.

%%% formula
expr: jax.vmap(f, in_axes=(0, None))(x_batch, w)
note: first arg `x_batch` → map over axis 0; second arg `w` → None = shared, broadcast whole to every call
%%%

Let's see the failure and its fix side by side. Predict which one keeps the weight whole:

%%% demo id=none label="the fix: mark the shared weight None"
code: def predict(x, w):  return jnp.dot(x, w)   # one example x, shared weight w
code: xb = jnp.ones((5, 3))       # 5 examples, each length-3
code: w  = jnp.ones((3,))         # ONE shared weight, length-3
code: # in_axes=(0, None): map x over axis 0, share w whole
code: out = jax.vmap(predict, in_axes=(0, None))(xb, w)
code: print("out shape:", out.shape)
out: out shape: (5,)
take: <b>The weight stayed whole; only x was stamped.</b> <code>in_axes=(0, None)</code> told vmap "slice x per example, share w on every call." You get 5 predictions from 5 examples using 1 shared weight. If you had left the default (map BOTH), vmap would try to slice w into 5 too — a shape error, or silently wrong results. Marking shared inputs None is the fix.
%%%

!!! c-warn ⚠️
<b>Failure mode:</b> forgetting <code>None</code> on a <b>shared</b> input (like model weights). By default <code>vmap</code> maps <b>every</b> argument, so it tries to slice the shared value per example too. <b>Cause:</b> every argument defaults to being mapped over axis 0. <b>Remedy:</b> pass a per-argument <code>in_axes</code> tuple and set the shared inputs to <code>None</code> so they're broadcast whole.
!!!

You can now stamp some inputs and share others. But we've still been assuming the batch axis is the *first* one. What if your data stores the batch somewhere else — say, down the columns? That's a one-character change, coming up.

@@@ concept id=c4 tag="Any axis" title="in_axes can be any axis — not just the first" gotit="Got any-axis"
Not every dataset stores the batch on the first axis. Sometimes the examples run *down the columns* instead of *across the rows*. `in_axes` is not stuck at `0` — it can be **any** axis number. Set `in_axes=1` and `vmap` slices along the second axis instead; the batch was never *forced* to go first.

Picture a **book you can read two ways.** Read it row by row (top to bottom), and each row is one example — that's `in_axes=0`. Turn the same page sideways and read it column by column (left to right), and each *column* is one example — that's `in_axes=1`. Same page of numbers; you just pick which direction counts the examples. Choosing the wrong direction doesn't error loudly — it quietly reads the wrong slices, like reading a sideways table straight down and getting nonsense.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A grid of numbers read two ways. On the left, labelled in_axes=0, horizontal arrows point along the rows with a note 'each ROW is one example'. On the right, the same grid labelled in_axes=1, with vertical arrows pointing down the columns and a note 'each COLUMN is one example'. A label reads: in_axes picks WHICH axis counts the examples — pick wrong and you slice the wrong direction."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">in_axes = which direction counts the examples (rows? or columns?)</text><text x="120" y="40" fill="#2D8B55" font-size="11">in_axes=0 → rows</text><g transform="translate(60,50)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="30" height="24"/><rect x="30" y="0" width="30" height="24"/><rect x="60" y="0" width="30" height="24"/><rect x="0" y="24" width="30" height="24"/><rect x="30" y="24" width="30" height="24"/><rect x="60" y="24" width="30" height="24"/><rect x="0" y="48" width="30" height="24"/><rect x="30" y="48" width="30" height="24"/><rect x="60" y="48" width="30" height="24"/></g><path d="M96 62 h34" stroke="#2D8B55" stroke-width="2" marker-end="url(#r)"/><path d="M96 86 h34" stroke="#2D8B55" stroke-width="2" marker-end="url(#r)"/><path d="M96 110 h34" stroke="#2D8B55" stroke-width="2" marker-end="url(#r)"/><defs><marker id="r" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="#2D8B55"/></marker><marker id="d" markerWidth="7" markerHeight="7" refX="3.5" refY="6" orient="auto"><path d="M0 0 L3.5 7 L7 0 z" fill="#5E5191"/></marker></defs><text x="120" y="148" fill="#276b45">each ROW is one example</text><text x="395" y="40" fill="#5E5191" font-size="11">in_axes=1 → columns</text><g transform="translate(350,50)" fill="#F4F1FB" stroke="#5E5191"><rect x="0" y="0" width="30" height="24"/><rect x="30" y="0" width="30" height="24"/><rect x="60" y="0" width="30" height="24"/><rect x="0" y="24" width="30" height="24"/><rect x="30" y="24" width="30" height="24"/><rect x="60" y="24" width="30" height="24"/><rect x="0" y="48" width="30" height="24"/><rect x="30" y="48" width="30" height="24"/><rect x="60" y="48" width="30" height="24"/></g><path d="M365 78 v34" stroke="#5E5191" stroke-width="2" marker-end="url(#d)"/><path d="M395 78 v34" stroke="#5E5191" stroke-width="2" marker-end="url(#d)"/><path d="M425 78 v34" stroke="#5E5191" stroke-width="2" marker-end="url(#d)"/><text x="395" y="148" fill="#5E5191">each COLUMN is one example</text><text x="260" y="186" fill="#9A938A">pick the wrong axis → vmap slices the wrong direction (silently wrong, not an error)</text></g></svg>
%%%

**What the read-two-ways picture gets right:** the same block of numbers can be sliced by rows or by columns, and `in_axes` is just you saying which way to read it. **Where it breaks down:** you read a book one line at a time with your eyes, but `vmap` reads *all* the chosen slices together in one go — the direction is the only thing you're choosing, not the speed.

#### Watch a non-leading axis map
Say your batch of 5 examples is stored as shape `(3, 5)` — 5 examples running *down the columns*, each example of length 3. To map over the columns, set `in_axes=1`. Predict the output shape before you run it:

%%% demo id=axis1 label="map over the columns (axis 1)"
code: def total(x):  return jnp.sum(x)     # sum ONE length-3 example
code: batch_cols = jnp.ones((3, 5))        # 5 examples DOWN the columns
code: out = jax.vmap(total, in_axes=1)(batch_cols)
code: print("in :", batch_cols.shape)
code: print("out:", out.shape)             # one sum per column → 5 sums
out: in : (3, 5)
out: out: (5,)
take: <b>Mapped over axis 1, one result per column.</b> Each column (length 3) was one example; vmap summed each and gave back 5 sums. Had you used the default <code>in_axes=0</code>, it would have summed the 5-long ROWS instead — 3 wrong answers, no error. The batch axis was column-wise, so you pointed vmap at axis 1.
%%%

!!! c-warn ⚠️
<b>Failure mode:</b> mapping the <b>wrong axis</b> — leaving the default <code>in_axes=0</code> when your batch actually lives on a different axis. <b>Cause:</b> the default assumes the batch is the first axis. <b>Remedy:</b> set the integer <code>in_axes</code> (e.g. <code>in_axes=1</code>) to point vmap at the axis that really counts your examples.
!!!

!!! c-info 🔬
<b>Optional (skippable) — pytree-shaped in_axes.</b> When a single argument is a nested container (a [[pytree||A nested bundle of arrays — like a dict or tuple of arrays. JAX treats the whole bundle as one structured value.]], such as a dict of parameters), you can pass an `in_axes` that *mirrors that same structure* — for example `{"w": None, "x": 0}` — so different leaves of the bundle map over different axes (or are shared). You'll meet pytrees fully on a later day; for now just know `in_axes` can be as simple as `0` or as shaped as your input.
!!!

You can now stamp along any axis you like, on many inputs, sharing what should be shared. There's a matching question on the *output* side: once `vmap` gives you back a batch of results, *where* does that new batch dimension land? That's `out_axes`, next.

@@@ concept id=c5 tag="Where it lands" title="out_axes: where the batch lands in the output" gotit="Got out_axes"
`in_axes` decided which axis `vmap` reads from. There's a twin on the way out: [[out_axes||A setting you give vmap that says which axis of the result the new batch dimension should sit on. Default is 0 — the batch lands on the front.]] decides *where* the fresh batch dimension is placed in the result. By default it's `out_axes=0`, so the batch lands on the **front** of the output — the same friendly place it usually starts. But you can put it elsewhere.

Picture a **row of lockers.** Each stamped result is one item. `out_axes` is just the rule for *where you put the shelf of results*: lay them along the top row (front, `out_axes=0`) or stand them up in a side column (`out_axes=1`). The results are the same items — you're only choosing which direction they're arranged in the final array.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="An out_axes analogy showing where the batch dimension lands. On the left, labelled out_axes=0, five result items are laid in a horizontal row across the front. On the right, labelled out_axes=1, the same five items are stacked into a vertical column. A label reads: out_axes only chooses where the batch dimension sits in the result — the values are identical."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">out_axes: where does the new batch dimension sit in the result?</text><text x="115" y="42" fill="#2D8B55" font-size="11">out_axes=0 → front row</text><g transform="translate(35,54)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="20" width="30" height="26"/><rect x="34" y="20" width="30" height="26"/><rect x="68" y="20" width="30" height="26"/><rect x="102" y="20" width="30" height="26"/><rect x="136" y="20" width="30" height="26"/></g><text x="118" y="112" fill="#276b45">batch across the front (shape (5, ...))</text><text x="395" y="42" fill="#5E5191" font-size="11">out_axes=1 → side column</text><g transform="translate(380,52)" fill="#F4F1FB" stroke="#5E5191"><rect x="0" y="0" width="30" height="18"/><rect x="0" y="20" width="30" height="18"/><rect x="0" y="40" width="30" height="18"/><rect x="0" y="60" width="30" height="18"/><rect x="0" y="80" width="30" height="18"/></g><text x="395" y="150" fill="#5E5191">batch down a column (shape (..., 5))</text><text x="260" y="176" fill="#9A938A">same 5 results — out_axes only picks their direction in the final array</text></g></svg>
%%%

**What the lockers picture gets right:** the results are fixed items and `out_axes` only picks the direction they line up in — front row or side column. **Where it breaks down:** lockers hold physical objects you place one by one, but `vmap` produces the whole arranged array in one shot; you're labelling a layout, not carrying items.

#### Watch the batch move
Same computation, two placements. With `out_axes=0` the batch of 5 sits on the front; with `out_axes=1` it sits on the second axis. Predict both shapes, then run:

%%% demo id=outaxes label="move the batch dimension"
code: def make_row(x):  return x * jnp.arange(3.)   # ONE example → length-3 row
code: xb = jnp.ones((5,))                           # 5 scalars
code: front = jax.vmap(make_row, out_axes=0)(xb)    # batch on the front
code: side  = jax.vmap(make_row, out_axes=1)(xb)    # batch on axis 1
code: print("out_axes=0:", front.shape)
code: print("out_axes=1:", side.shape)
out: out_axes=0: (5, 3)
out: out_axes=1: (3, 5)
take: <b>Same numbers, different placement.</b> Each of 5 inputs makes a length-3 row. With <code>out_axes=0</code> the batch (5) lands first → shape <code>(5, 3)</code>. With <code>out_axes=1</code> it lands second → shape <code>(3, 5)</code>. The default (0) puts the batch on the front — the usual, tidy place. You only reach for out_axes when a later step wants the batch somewhere else.
%%%

Now you own both ends of the stamp: `in_axes` (which axis to read) and `out_axes` (where the batch lands). Everything so far has quietly assumed one big promise — that stamping the whole batch together is *faster* than looping by hand. It's time to see *why* that's true, and to learn two moves that push it further: pairing `vmap` with compilation, and stacking two `vmap`s.

@@@ concept id=c6 tag="Why it's fast" title="vmap beats a loop — and pairs with jit and itself" gotit="Got the payoff"
Here's the payoff that makes `vmap` worth learning. You *could* have written a plain Python `for`-loop over your batch — call the function 5 times, glue the results together. That works, but it's slow: the loop makes 5 **separate** trips to the chip, each one a small errand. `vmap` instead builds **one** combined computation that does all 5 at once — a single big trip. On the chips that train AI (which love doing many identical things in parallel), one big batched operation is far faster than many little separate calls.

Picture a **school bus versus 30 separate cars.** Thirty kids all going to the same school: send 30 cars (the loop — each its own driver, its own trip, lots of overhead) or send one bus that carries all 30 together (`vmap` — one trip, everyone at once). Same kids delivered; the bus is dramatically more efficient. `vmap` is the bus: it fuses the many identical trips into one.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A bus versus cars analogy for vmap versus a for-loop. On the left, labelled 'for-loop', five small separate cars each labelled 'one call' drive in a line, with a note 'many separate trips = slow'. On the right, labelled 'vmap', one big bus carries five passengers together, with a note 'one fused trip = fast'. A label reads: vmap fuses many identical calls into one batched computation."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">for-loop = many separate trips · vmap = one fused trip (the bus)</text><text x="115" y="40" fill="#C93B3B" font-size="11">for-loop: 5 separate calls</text><g transform="translate(30,52)" fill="#FDECEC" stroke="#C93B3B"><rect x="0" y="0" width="34" height="20" rx="5"/><rect x="0" y="26" width="34" height="20" rx="5"/><rect x="0" y="52" width="34" height="20" rx="5"/><rect x="0" y="78" width="34" height="20" rx="5"/><rect x="0" y="104" width="34" height="20" rx="5"/></g><g fill="#8a3b3b" font-size="8"><text x="72" y="65" text-anchor="start">call 1</text><text x="72" y="91" text-anchor="start">call 2</text><text x="72" y="117" text-anchor="start">call 3 …</text></g><text x="115" y="168" fill="#C93B3B">many small trips = slow</text><text x="395" y="40" fill="#2D8B55" font-size="11">vmap: one fused call</text><g transform="translate(320,70)"><rect x="0" y="0" width="150" height="48" rx="10" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><circle cx="118" cy="52" r="8" fill="#3A342E"/><circle cx="32" cy="52" r="8" fill="#3A342E"/><g fill="#1a5c38" font-size="8"><text x="30" y="20">😀 😀 😀</text><text x="30" y="36">😀 😀 (all aboard)</text></g></g><text x="395" y="168" fill="#276b45">one big trip = fast</text></g></svg>
%%%

**What the bus picture gets right:** many separate little trips (the loop) carry lots of wasted overhead, while one shared trip (`vmap`) moves everyone together far more efficiently. **Where it breaks down:** a bus still drives one road at a time, but on an AI chip the "bus" can genuinely process all the passengers *simultaneously* — the parallel speed-up is even better than the bus makes it look.

#### The two power-moves
Once you have the bus, two habits make it even better. First, wrap it in [[jit||A JAX tool (coming tomorrow) that compiles your function into one fast, fused machine-code kernel. jit(vmap(f)) squeezes the whole batched computation into a single optimized program.]]: `jit(vmap(f))` compiles the whole batched computation into a single tight kernel — the bus, but now with a perfectly-tuned engine. (You meet `jit` in full tomorrow; today just know it stacks cleanly on top of `vmap`.) Second, you can `vmap` a function that's *already* `vmap`ped — [[nested vmap||Applying vmap twice, so you map over two axes independently. Handy for building a grid of results, like every row paired with every column.]] — to map over **two** axes at once. That's how you build a full grid of results, like pairing every row with every column to make an outer-product table.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Nested vmap building a grid. On the left a column of row-values and on the top a row of column-values feed into a doubly-mapped function, producing a full grid of pairwise results on the right. A label reads: vmap over rows, then vmap over columns, gives every pair — a full matrix."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">nested vmap: map rows, then map columns → every pair (a full grid)</text><text x="55" y="46" fill="#2D8B55">rows (vmap #1)</text><g transform="translate(40,54)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="24" height="20"/><rect x="0" y="22" width="24" height="20"/><rect x="0" y="44" width="24" height="20"/></g><text x="180" y="40" fill="#5E5191">columns (vmap #2)</text><g transform="translate(110,44)" fill="#F4F1FB" stroke="#5E5191"><rect x="0" y="0" width="24" height="18"/><rect x="26" y="0" width="24" height="18"/><rect x="52" y="0" width="24" height="18"/></g><text x="215" y="90" fill="#9A5A12" font-size="12">→</text><g transform="translate(250,44)" fill="#FDF3E7" stroke="#C99A12"><rect x="0" y="0" width="24" height="20"/><rect x="26" y="0" width="24" height="20"/><rect x="52" y="0" width="24" height="20"/><rect x="0" y="22" width="24" height="20"/><rect x="26" y="22" width="24" height="20"/><rect x="52" y="22" width="24" height="20"/><rect x="0" y="44" width="24" height="20"/><rect x="26" y="44" width="24" height="20"/><rect x="52" y="44" width="24" height="20"/></g><text x="288" y="130" fill="#9A5A12">every (row, column) pair</text><text x="260" y="162" fill="#9A938A">two vmaps = two independent batch axes, no double loop by hand</text></g></svg>
%%%

Let's watch nested `vmap` build a pairwise grid. Predict the output shape — 4 rows paired with 3 columns should give a 4-by-3 grid:

%%% demo id=nested label="nested vmap → a full grid"
code: def prod(a, b):  return a * b            # multiply ONE pair
code: rows = jnp.arange(4.)                    # 4 row values
code: cols = jnp.arange(3.)                    # 3 column values
code: # inner vmap maps b over cols; outer vmap maps a over rows
code: grid = jax.vmap(lambda a: jax.vmap(lambda b: prod(a, b))(cols))(rows)
code: print("grid shape:", grid.shape)
out: grid shape: (4, 3)
take: <b>Two vmaps → a 4×3 grid, no hand-written double loop.</b> The inner vmap stamps <code>prod</code> across the 3 columns; the outer vmap stamps that across the 4 rows. You wrote <code>prod</code> for a single pair, and got every pairing back as a grid. This is how outer products and pairwise tables get built in JAX.
%%%

Notice what `vmap` is *not* for: work where each step **depends on the one before it** — like adding up a running total step by step. Those can't all happen at once (step 3 needs step 2's answer first), so `vmap` can't fuse them; JAX has a different tool (`scan`) for genuinely sequential loops. And `vmap` runs the whole batch on **one** device — spreading a batch across *many* machines is yet another tool (`pmap`), for a later day. `vmap` is the master of *identical, independent* work — stamp the same shape, over and over, all at once.

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — you now think the JAX way about batches: **write for one, run on many.** Before we gather the pieces, here's one fresh everyday picture for the whole day. Think of today like **a rubber stamp and an ink pad at a post office.** You carve one stamp (your one-example function). `in_axes` is how you feed the sheet of envelopes in — long edge first or short edge first (which axis is the batch). `in_axes=None` is the ink pad: shared by every envelope, never cut up. `out_axes` is where you stack the finished envelopes. And stamping the whole tray in one press beats inking each envelope by hand (`vmap` beats the loop). **Where the stamp picture breaks down:** a clerk still presses envelopes one at a time, but `vmap` presses the whole tray *together* in a single fused action — that togetherness is the speed. Below is the day step by step, plus a cheat-sheet to keep on your desk.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A five step flow for the day. Step one: write your function for ONE example. Step two: in_axes picks which axis is the batch. Step three: in_axes=None shares an input whole. Step four: out_axes places the batch in the output. Step five: vmap beats a loop and pairs with jit and nested vmap. Each step points to the next."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The whole day: write-for-one → in_axes → None shares → out_axes → fast fused batch</text><g transform="translate(4,54)"><rect x="0" y="0" width="92" height="60" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="46" y="26" fill="#1a5c38">1 write for ONE</text><text x="46" y="42" fill="#276b45" font-size="8">def f(x): ...</text></g><text x="100" y="88" fill="#9A5A12" font-size="13">→</text><g transform="translate(116,54)"><rect x="0" y="0" width="92" height="60" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"/><text x="46" y="26" fill="#8A6D3B">2 in_axes</text><text x="46" y="42" fill="#9A5A12" font-size="8">which axis = batch</text></g><text x="212" y="88" fill="#9A5A12" font-size="13">→</text><g transform="translate(228,54)"><rect x="0" y="0" width="92" height="60" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"/><text x="46" y="26" fill="#5E5191">3 None shares</text><text x="46" y="42" fill="#5E5191" font-size="8">whole weight</text></g><text x="324" y="88" fill="#9A5A12" font-size="13">→</text><g transform="translate(340,54)"><rect x="0" y="0" width="92" height="60" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"/><text x="46" y="26" fill="#8A6D3B">4 out_axes</text><text x="46" y="42" fill="#9A5A12" font-size="8">where batch lands</text></g><text x="436" y="88" fill="#9A5A12" font-size="13">→</text><g transform="translate(452,54)"><rect x="0" y="0" width="66" height="60" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="33" y="26" fill="#1a5c38">5 fast</text><text x="33" y="42" fill="#276b45" font-size="8">beats loop</text></g><text x="260" y="156" fill="#9A938A">one function, stamped across the whole batch — faster than a hand-written loop</text></g></svg>
%%%

The day in five beats:
- **Write for one.** Write your function for a **single** example, then hand it to [[vmap||the cookie-cutter that stamps a one-example function across a whole batch]] to run on the whole [[batch||many examples stacked in one array]] — no hand-written loop.
- **Which axis?** [[in_axes||tells vmap which axis of an input is the batch to map over]] picks the batch axis. Default `in_axes=0` (first axis). Set an integer like `in_axes=1` for a batch on another axis. Pick wrong → silently wrong slices.
- **Share the shared.** `in_axes=None` marks an input (like weights) as **shared** — broadcast whole to every call, not sliced. Pass a per-argument **tuple** to mix mapped and shared inputs.
- **Where it lands.** [[out_axes||tells vmap which axis of the output the new batch dimension sits on]] places the new batch dimension in the result (default `0`, the front).
- **Fast & composable.** `vmap` fuses the batch into **one** computation (far faster than a loop). `jit(vmap(f))` compiles it into one kernel; **nested vmap** maps over two axes to build a grid. It's for *identical, independent* work only — sequential loops want `scan`, many devices want `pmap`.

The shape rule to memorize: **map over an axis of size `N`, and an `N` slides onto the front of the result.**

#### Cheat-sheet · the moves at a glance
%%% table
:: You want to… :: Do this :: Note
Run f on a batch :: `jax.vmap(f)(batch)` :: default maps axis 0 (the front)
Batch on another axis :: `jax.vmap(f, in_axes=1)(batch)` :: point at the axis that counts examples
Share a weight :: `jax.vmap(f, in_axes=(0, None))(x, w)` :: `None` = shared, not sliced
Place the batch in output :: `jax.vmap(f, out_axes=1)(batch)` :: default `out_axes=0` (front)
Compile it :: `jax.jit(jax.vmap(f))(batch)` :: one fused kernel (tomorrow's tool)
Map two axes :: `jax.vmap(jax.vmap(f))(grid)` :: nested → a full grid
%%%

#### Cheat-sheet · the words you met today
%%% jargon
vmap | the cookie-cutter: turns a one-example function into a whole-batch function, no loop
batch | many examples stacked together in one array
batch axis | the array direction that counts the separate examples
in_axes | which axis of each input vmap maps over; an integer, or None for shared, or a tuple/pytree
in_axes=None | mark an input as shared — broadcast whole to every mapped call (e.g. weights)
out_axes | which axis of the output the new batch dimension lands on (default 0, the front)
shape contract | map over an axis of size N → an N appears on the front of the result
nested vmap | vmap twice to map over two axes and build a grid (e.g. an outer product)
jit | tomorrow's tool: compiles a function into one fast fused kernel; jit(vmap(f)) is common
%%%

That's the whole day. You now write a function once, for one example, and let `vmap` stamp it across a whole batch like a cookie-cutter — picking the axis with `in_axes`, sharing with `None`, placing the result with `out_axes`, all faster than a loop. Next you'll meet **jit**, the tool that takes any function — including your `vmap`ped one — and compiles it into one fast, fused program.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is the core idea of vmap? | a:2 | It makes random numbers reproducible | It compiles your function to machine code | Write your function for ONE example, then run it over a whole batch automatically | It spreads work across many machines | fb: vmap is the cookie-cutter: you write a one-example function and vmap stamps it across the whole batch — no hand-written loop. (Compiling is jit; many machines is pmap.)
q: You vmap a function that turns a length-3 vector into a length-3 vector, over a batch of 5 (shape (5, 3)). What comes out? | a:1 | shape (3,) | shape (5, 3) | shape (3, 5) | a single number | fb: The shape contract: map over an axis of size N and an N slides onto the front of the result. Five inputs of shape (3,) → five outputs, stacked as (5, 3).
q: Your function takes an example x AND a shared weight w. How do you keep w whole (not sliced per example)? | a:2 | Use jax.jit instead of vmap | Reshape w to match the batch | Pass in_axes=(0, None) — None marks w as shared/broadcast | Loop over the batch by hand | fb: Every argument is mapped by default, so a shared input needs in_axes=None. The tuple (0, None) maps x over axis 0 and broadcasts w whole to every call.
q: Why is vmap faster than a Python for-loop over the batch? | a:1 | It skips half the examples | It fuses the batch into ONE combined computation instead of many separate calls | It uses less memory by deleting data | It runs on many machines at once | fb: vmap builds one batched computation (the bus carrying everyone) rather than many small separate calls (the loop's many trips) — a big win on accelerators. (Many machines is pmap, a different tool.)
%%%

@@@ produce id=produce tag="Produce" title="Feel vmap with your own hands" gotit="Done"
Time to feel today's rule for yourself. You'll write a function for **one** example, then let `vmap` run it over a whole batch — and watch the shapes to prove it worked. **Predict first:** if you `vmap` a function over a batch axis of size 5, what number should appear on the front of the output shape? Then run it and **watch** the `5` slide in. Pick one path.

#### Option A · write it yourself
Create `sessions/m07-thinking-in-jax/day-03-vmap/experiment.py`. Start with `import jax` and `import jax.numpy as jnp`. **Print every shape so you can see it happen.** (1) Write `def scale(x): return x * 2.0` — a function for ONE vector — and call it on a single vector to see it work. (2) Make a batch of shape `(5, 3)`, wrap `scale` with `jax.vmap`, run it, and **observe** the output shape is `(5, 3)` — the batch rode the front. (3) Add a shared weight: `def predict(x, w): return jnp.dot(x, w)`; make `xb` of shape `(5, 3)` and `w` of shape `(3,)`; call `jax.vmap(predict, in_axes=(0, None))(xb, w)` and **notice** the shape is `(5,)` — `w` stayed whole. (4) Store a batch as shape `(3, 5)` and map over `in_axes=1`; **observe** you get 5 results. (5) Build a pairwise grid with **nested** `vmap` over `jnp.arange(4)` and `jnp.arange(3)`, and **notice** the shape is `(4, 3)`. Run with `python3 sessions/m07-thinking-in-jax/day-03-vmap/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Week 1 Day 3 (JAX vmap) artifact.

Create sessions/m07-thinking-in-jax/day-03-vmap/experiment.py that, with a comment on each step and printing every SHAPE:
1. import jax and jax.numpy as jnp; define scale(x) = x * 2.0 (a function for ONE vector); call it on jnp.array([1., 2., 3.]) and print the result.
2. Make batch = jnp.ones((5, 3)); wrap with jax.vmap(scale); run it; print the input and output shapes — the output must be (5, 3), proving vmap mapped over the batch axis (size 5) with in_axes=0 by default.
3. Show in_axes=None for a shared weight: define predict(x, w) = jnp.dot(x, w); make xb = jnp.ones((5, 3)) and w = jnp.ones((3,)); call jax.vmap(predict, in_axes=(0, None))(xb, w); print the output shape — it must be (5,) because w stayed shared/whole.
4. Show a non-leading batch axis: make batch_cols = jnp.ones((3, 5)) (5 examples DOWN the columns); call jax.vmap(jnp.sum, in_axes=1)(batch_cols); print the output shape — it must be (5,).
5. Show nested vmap building a grid: with rows = jnp.arange(4.) and cols = jnp.arange(3.), compute grid = jax.vmap(lambda a: jax.vmap(lambda b: a * b)(cols))(rows); print grid.shape — it must be (4, 3).
Then run it and paste the output at the bottom as a comment. Explain in one line why vmap is faster than a Python for-loop over the batch (it fuses into one batched computation).
%%%

#### What you should see (write for one, the batch rides along)
- `scale` on one vector `[1, 2, 3]` gives `[2, 4, 6]` — it only knows about ONE example.
- `jax.vmap(scale)` on a `(5, 3)` batch gives `(5, 3)` back — the batch size `5` **slid onto the front**, no loop written.
- `in_axes=(0, None)` keeps the weight `w` **whole**; the output is `(5,)` — 5 predictions from 1 shared weight.
- Mapping `in_axes=1` over a `(3, 5)` batch gives `(5,)` — you pointed vmap at the column axis.
- **Nested** `vmap` over 4 rows and 3 columns gives a `(4, 3)` grid — every pair, no double loop by hand.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m07-thinking-in-jax/day-03-vmap/log.md`: (1) in one sentence, the vmap mindset — what you write versus what JAX runs; (2) what `in_axes` and `out_axes` each control, and when you'd set `in_axes=None`; (3) in one sentence, why `vmap` is faster than a hand-written Python for-loop over the batch.

@@@ fin
