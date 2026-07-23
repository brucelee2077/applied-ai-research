---
quest_id: w01-d02-prng
mode: concept
donor: v9-base.donor
page_title: "Week 1 · Day 2 — JAX Randomness: Explicit Keys"
module_label: "Thinking in JAX · Week 1 · Day 2"
title: "Randomness With Explicit Keys"
subtitle: "Carrying Your Own Source of Chance"
brand_sub: "Frontier Lab · Week 1 Day 2"
spine: "key"
nav_prev_href: "../day-01-jax-immutability/lesson.html"
nav_prev_label: "JAX Immutability"
nav_next_href: "../day-03-vmap/lesson.html"
nav_next_label: "vmap: Vectorization"
fin_title: "Week 1 · Day 2 complete! 🏆"
fin_body: "Nice — you've learned JAX's rule for randomness: there is no hidden dice cup. You carry an explicit <b>key</b> and hand it to every random call, and you <code>split</code> it to get fresh independent keys instead of reusing one (which would give the same numbers twice). That explicitness is what makes JAX randomness reproducible and safe to run in parallel.<br>Next up: <b>vmap: Vectorization</b>."
notebook_yardstick: null
---

@@@ hero
@lede Imagine a board game where the dice live in a little cup on the table, and everyone shares that one cup. You shake it, someone else shakes it, and nobody quite knows what number will come up next — the cup keeps its own secret memory of every shake. Now imagine a stricter, fairer game: instead of one shared cup, **you carry your own dice in your pocket**, and every time you want a random number you must take them out and show them. That is exactly how randomness works in **JAX**, the library behind many of the biggest AI models — including work at Google DeepMind. There is no hidden shared dice cup. You carry your own little token of chance, called a **key**, and you hand it to every random call in the open. It sounds like extra work, but here's the surprise: this one habit is what makes an AI experiment give the *exact same* "random" answer twice — the thing researchers call reproducibility — and what lets JAX roll thousands of dice at once without them stepping on each other.
@goal Together we'll meet the JAX <b>key</b> — the little token of chance you carry yourself — see why the old "hidden dice cup" way causes trouble, learn how a key produces random numbers, and master the one move (<code>split</code>) that hands you fresh keys so you never accidentally roll the same numbers twice. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="Your own dice" title="A key is your own little token of chance" gotit="Got the key"
Let's meet the star of the day. When you want a random number in JAX, you don't just ask for one out of thin air. First you make a small object called a [[key||A small piece of data (two whole numbers) that JAX uses as the "seed of chance" for making random numbers. You carry it yourself and hand it to every random call.]]. Think of it as a little token you keep in your pocket — it holds the "starting point" for all the randomness you're about to make. You create one from a plain whole number called a [[seed||A starting number you pick (like 0 or 42). Give the same seed, and you always get the exact same key — the same starting point for randomness.]], like this: `key = jax.random.PRNGKey(0)`.

Here's the mental picture. A key is like **your own pair of dice that you carry in your pocket**. It's a real object you can hold, look at, and hand to someone. It is not hidden away in some shared cup on the table — it's *yours*, right there in the open. And just like a physical object, it doesn't change on its own: looking at your dice doesn't magically reroll them.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A hand holding a small token labelled key, made by feeding the seed number 0 into a box labelled PRNGKey. The key is shown as two small numbers side by side. A label reads: you carry the key yourself, it is just data, not hidden state."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">A key = a little token of chance you carry yourself</text><text x="70" y="52" fill="#6B645E">seed you pick</text><g transform="translate(40,60)"><rect x="0" y="0" width="60" height="40" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="30" y="26" fill="#5E5191" font-size="16">0</text></g><text x="130" y="84" fill="#9A5A12" font-size="14">-&gt;</text><g transform="translate(150,58)"><rect x="0" y="0" width="120" height="44" rx="6" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="60" y="20" fill="#8A6D3B" font-size="10">PRNGKey(seed)</text><text x="60" y="36" fill="#9A938A" font-size="8">makes a key</text></g><text x="290" y="84" fill="#9A5A12" font-size="14">-&gt;</text><g transform="translate(320,58)"><rect x="0" y="0" width="110" height="44" rx="10" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="55" y="20" fill="#1a5c38" font-size="10">key</text><text x="55" y="37" fill="#1a5c38" font-size="12">[0, 0]</text></g><text x="375" y="120" fill="#276b45" font-size="9">just two numbers — data you hold</text><text x="130" y="150" fill="#6B645E" font-size="9">same seed in → same key out, every time</text></g></svg>
%%%

**What the pair-of-dice picture gets right:** the key is a real thing you hold and pass along, in the open, not hidden in a shared cup — and you always know exactly which "dice" you're using. **Where it breaks down:** real dice give a *fresh* surprise every roll, but a JAX key is stricter — the same key always gives the exact same "roll," which is a feature we'll lean on for reproducibility, not a bug.

#### What a key is really made of
A key is not magic — it's just a tiny [[array||A list of numbers in a fixed order. From Day 1: JAX's basic building block, and it is immutable — frozen once made.]] holding two whole numbers. Because it's a JAX array, it follows yesterday's rule: it is [[immutable||From Day 1: frozen after creation. A key never changes itself — you can't "use it up."]] — frozen, never changing on its own. Let's make one and peek inside:

%%% demo id=makekey label="make a key and look at it"
code: import jax;  key = jax.random.PRNGKey(0);  print(key)
out: [0 0]
take: <b>A key is just two numbers.</b> Here <code>PRNGKey(0)</code> gives the key <code>[0 0]</code>. It's plain data you can hold and pass around — not a hidden setting buried inside the library. Give the same seed and you'll always get this same key back.
%%%

That's the whole idea of a key: **randomness is data you carry, not a secret the library keeps.** Hold that thought — the next concept shows the *old* way, where randomness *was* a hidden secret, and why that caused so much trouble.

@@@ concept id=c2 tag="The old way" title="The old way: one shared dice cup on the table" gotit="Got the contrast"
To feel why carrying your own key matters, let's look at the tool it replaces. Many people first meet random numbers through [[NumPy||A popular Python number library. Its randomness lives in one hidden global place, unlike JAX's explicit keys.]]. In NumPy you write `np.random.seed(0)` once, and then every time you call `np.random.rand()` you get the next random number — no key in sight. Where does the randomness come from? From one **hidden shared cup** sitting inside the library. Every call reaches into that same cup, shakes it, and the cup quietly remembers where it left off. This hidden, self-remembering thing has a name: [[global state||Information stored in one shared, hidden place that every part of the program can reach and change. The "shared dice cup" — convenient, but easy to mess up.]].

Picture the whole room sharing **one dice cup in the middle of the table**. It's convenient — you never have to carry anything. But the cup keeps a private memory of every shake, and *anyone* can grab it. If two people reach for it, or someone shakes it in a different order than you expected, your "random" result changes — and you can't see why, because the cup's memory is hidden from you.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Left, NumPy: one shared dice cup in the middle of a table with three hands reaching in, and a hidden memory dial inside marked with a question mark, showing anyone can change the shared hidden state. Right, JAX: three people each holding their own key in their own pocket, separate and visible. A label contrasts shared hidden cup versus your own key."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Shared hidden cup (NumPy) vs your own key (JAX)</text><text x="130" y="38" fill="#5E5191">NumPy: one shared cup</text><g transform="translate(90,50)"><ellipse cx="45" cy="55" rx="60" ry="14" fill="#EDE9E0" stroke="#B8AEA2"/><path d="M15 20 h60 l-8 34 h-44 z" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"/><text x="45" y="42" fill="#8a3b3b" font-size="12">? ? ?</text><text x="45" y="14" fill="#9A938A" font-size="8">hidden memory</text></g><text x="55" y="118" fill="#C93B3B" font-size="9">hand</text><text x="130" y="118" fill="#C93B3B" font-size="9">hand</text><text x="205" y="118" fill="#C93B3B" font-size="9">hand</text><text x="130" y="138" fill="#C93B3B" font-size="9">anyone can shake it, order matters</text><text x="395" y="38" fill="#2D8B55">JAX: everyone holds a key</text><g transform="translate(320,54)"><rect x="0" y="10" width="42" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><rect x="55" y="10" width="42" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><rect x="110" y="10" width="42" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="21" y="27" fill="#1a5c38" font-size="9">key</text><text x="76" y="27" fill="#1a5c38" font-size="9">key</text><text x="131" y="27" fill="#1a5c38" font-size="9">key</text></g><text x="395" y="118" fill="#276b45" font-size="9">separate, visible, in the open</text><text x="395" y="138" fill="#276b45" font-size="9">order can't secretly bite you</text></g></svg>
%%%

**What the shared-cup picture gets right:** the old way really does keep one hidden memory that every call touches — handy when you're alone, risky when many things run at once. **Where it breaks down:** a real dice cup can't clone itself, but the deeper problem with hidden state is that when JAX later runs your code *very* fast, in a different order, or on many machines at once, a hidden shared cup gives wrong or unrepeatable answers — which is exactly why JAX threw the cup away.

#### Why hidden state breaks JAX's superpowers
Remember from Day 1 that JAX loves [[pure functions||From Day 1: same input always gives the same output, and nothing hidden gets changed. This is what lets JAX speed up and reorder your code.]] — same input, same output, no hidden surprises. A shared dice cup is the enemy of that. A function that secretly reaches into a hidden cup gives a *different* answer depending on who shook the cup before it — that's an impure surprise. And when JAX speeds up your code, it may reorder or duplicate calls; a hidden ordering-dependent cup would silently give the wrong numbers. So JAX made a firm choice: **no hidden cup. Carry your key in the open.** That single choice is why the rest of today exists.

@@@ concept id=c3 tag="A key makes numbers" title="Hand your key to a random call to get numbers" gotit="Got the draw"
Now the fun part — actually *using* the key to make random numbers. In JAX, every function that makes randomness asks you for a key as its first ingredient. You hand the key over, and out comes a [[draw||The random numbers a random call gives you — one number, or a whole array of them. Like the result of a dice roll.]]: some fresh random numbers. For example, `jax.random.normal(key, shape=(3,))` hands back three random numbers spread around zero. The word `normal` just names the *pattern* of the spread — you'll meet a few patterns in a moment.

Picture a **vending machine for random numbers**. The key is the coin you must insert. No coin, no snack — and JAX literally will not give you random numbers without a key. You push the coin in (`normal`, `uniform`, whatever you want), and out drops your handful of random numbers. Here's the twist that makes JAX special: **push in the same coin, and you get the exact same snack.** Same key in → same numbers out, every single time.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A vending machine for random numbers. A key shaped like a coin is inserted at the top. Below, three buttons labelled normal, uniform, randint. Out of the tray drops a handful of random numbers 0.9, minus 0.4, 1.3. A label reads: same key in gives the same numbers out, every time."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Insert your key (the coin) → out come random numbers</text><g transform="translate(150,30)"><rect x="0" y="0" width="220" height="130" rx="10" fill="#EDE9E0" stroke="#8A8172" stroke-width="2"/><rect x="70" y="10" width="80" height="16" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="110" y="22" fill="#1a5c38" font-size="9">insert key</text><rect x="20" y="36" width="55" height="20" rx="4" fill="#F4F1FB" stroke="#5E5191"/><rect x="82" y="36" width="55" height="20" rx="4" fill="#F4F1FB" stroke="#5E5191"/><rect x="144" y="36" width="56" height="20" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="47" y="50" fill="#5E5191" font-size="8">normal</text><text x="109" y="50" fill="#5E5191" font-size="8">uniform</text><text x="172" y="50" fill="#5E5191" font-size="8">randint</text><rect x="30" y="78" width="160" height="38" rx="4" fill="#fff" stroke="#B8AEA2"/><text x="110" y="102" fill="#3A342E" font-size="11">0.9  -0.4  1.3</text><text x="110" y="70" fill="#9A938A" font-size="8">out comes the draw</text></g><text x="260" y="176" fill="#276b45" font-size="9">same key → same draw, every time (this is the superpower)</text></g></svg>
%%%

**What the vending-machine picture gets right:** you must insert the coin (key) to get anything, and pressing the same button gives a predictable result — no coin, no numbers. **Where it breaks down:** a real vending machine gives you the *same* chocolate bar every time, which is boring; JAX gives you numbers that *look* perfectly random and varied, yet are perfectly repeatable if you use the same key — random-looking, but not truly unpredictable.

#### A few coins to press
You don't need every button today — just know the key goes in first, and the name picks the *pattern* of numbers you get out:

%%% table
:: Random call :: What it gives you
`random.normal(key, (n,))` :: `n` numbers spread around 0 (the classic "bell curve" spread)
`random.uniform(key, (n,))` :: `n` numbers evenly spread between 0 and 1
`random.randint(key, (n,), 0, 6)` :: `n` whole numbers from 0 up to 5 (like rolling a die)
`random.bernoulli(key, 0.5, (n,))` :: `n` coin flips: `True` or `False`
%%%

#### See "same key, same draw" with your own eyes
This is the property everything today rests on. Call `normal` twice with the **same** key — predict the result before you read the output:

%%% demo id=samekey label="predict: same key, twice"
code: key = jax.random.PRNGKey(0)
code: print(jax.random.normal(key, (3,)))
code: print(jax.random.normal(key, (3,)))
out: [ 1.81  -0.19  0.41]
out: [ 1.81  -0.19  0.41]
take: <b>Identical — on purpose.</b> Same key in means same numbers out, both times. This is what makes an AI experiment repeatable: rerun it tomorrow with the same key and you get the exact same "random" answer. That's the whole reason JAX makes you hold the key.
%%%

Feeling like that's a little strange — "random numbers that repeat"? That confusion is completely normal, and it's actually the point. But it also hides a trap: what if you *wanted* three different numbers and then three *more* different numbers? The next concept shows the mistake almost everyone makes here.

@@@ concept id=c4 tag="Trap: reused key" title="The trap: reusing one key gives the same roll twice" gotit="Got the trap"
Here's the mistake that catches nearly everyone on their first day with JAX keys. You learned that the same key gives the same draw. That's wonderful for repeating a *whole* experiment — but it's a disaster inside your program if you forget it. If you use one key for two different random calls, you don't get "more randomness." You get **the exact same numbers twice**, quietly, with no error to warn you.

Picture a **rubber stamp**. You press it once and get a shape on the paper. You press the *same* stamp again — you get the *identical* shape, not a new one. A key is like that stamp: pressing it (using it) doesn't wear it down or advance it. Reuse the same key and you re-stamp the same numbers. People expect a key to behave like the old dice cup that "remembers where it left off" and moves forward on its own — but a key does no such thing. It just sits there, always stamping the same picture.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A rubber stamp pressed twice with the same key produces two identical stamped patterns of the same three numbers, showing key reuse gives duplicate draws. A red warning label reads: same key twice equals same numbers twice, no error."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Reuse one key → the SAME numbers stamped twice (no error!)</text><g transform="translate(215,30)"><rect x="0" y="0" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="45" y="17" fill="#1a5c38" font-size="10">one key</text><path d="M45 26 v14" stroke="#8A8172" stroke-width="2"/></g><text x="130" y="80" fill="#C93B3B" font-size="9">stamp 1</text><g transform="translate(60,86)"><rect x="0" y="0" width="140" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="70" y="22" fill="#8a3b3b" font-size="11">0.9  -0.4  1.3</text></g><text x="390" y="80" fill="#C93B3B" font-size="9">stamp 2 (same key)</text><g transform="translate(320,86)"><rect x="0" y="0" width="140" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="70" y="22" fill="#8a3b3b" font-size="11">0.9  -0.4  1.3</text></g><text x="260" y="150" fill="#C93B3B" font-size="10">identical — the "random" numbers are secretly copies</text><text x="260" y="168" fill="#9A938A" font-size="9">a key does NOT advance on its own — it always stamps the same picture</text></g></svg>
%%%

**What the rubber-stamp picture gets right:** pressing the same stamp again gives the identical mark — reusing a key gives the identical draw, and the key never "moves forward" by itself. **Where it breaks down:** a real stamp slowly fades with ink; a JAX key never fades at all — it will stamp the exact same numbers forever, no matter how many times you reuse it.

#### Watch the trap happen
Say you want to make two *different* batches of random numbers. The natural (wrong) instinct is to reuse the same key. Predict what happens, then run it — watch for TWO printed lines, one per `print`:

%%% demo id=reuse label="the trap: reuse the key"
code: key = jax.random.PRNGKey(0)
code: batch_a = jax.random.normal(key, (3,))
code: batch_b = jax.random.normal(key, (3,))   # same key again!
code: print(batch_a)
code: print(batch_b)
out: [ 1.81  -0.19  0.41]
out: [ 1.81  -0.19  0.41]
take: <b>Two "random" batches that are secretly identical.</b> Two <code>print</code> calls, two lines — and they match number-for-number. You asked for two draws but reused one key, so you re-stamped the same numbers. No error appeared — this silent duplicate is the #1 key mistake. In a real model this means your two "random" things are accidentally twins.
%%%

!!! c-warn ⚠️
<b>Failure mode (silent):</b> using the <b>same key</b> for two random calls (or in every step of a loop) gives <b>identical</b> draws — no error, just secret duplicates. <b>Cause:</b> a key does not advance itself; reusing it re-stamps the same numbers. <b>Remedy:</b> never reuse a key — get a <b>fresh</b> key for each call by <b>splitting</b> (the next concept).
!!!

So the rule is simple and strict: **use a key once, then throw it away.** But if every key is single-use, where do all the fresh keys come from? That's the beautiful trick we learn next — one key can hatch many fresh keys.

@@@ concept id=c5 tag="Split for fresh keys" title="split: one key hatches many fresh keys" gotit="Got split"
Here's the move that fixes the reuse trap — the single most important habit with JAX keys. There's one function, `jax.random.split`, that takes one key and gives you back **several brand-new keys**, each different and independent. You use `split` to hatch fresh keys, hand one fresh key to each random call, and never reuse a key. That's the whole game.

Picture a key as a **parent that lays eggs**. You take your one parent key, call `split`, and out come new baby keys — each a complete, independent key of its own, ready to be used once. When you need randomness, you hatch a fresh egg, use it, and toss it. Run low? Split again for more. You never squeeze the same egg twice.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="One parent key on the left goes into a split function and hatches into several fresh baby keys on the right, each drawn separately and labelled subkey. A label reads: one key in, many fresh independent keys out."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">split(key, 3): one parent key → 3 fresh independent keys</text><text x="60" y="52" fill="#6B645E">parent key</text><g transform="translate(30,60)"><rect x="0" y="0" width="70" height="40" rx="10" fill="#EDE9E0" stroke="#8A8172" stroke-width="2"/><text x="35" y="26" fill="#3A342E" font-size="10">key</text></g><g transform="translate(130,58)"><rect x="0" y="2" width="90" height="44" rx="6" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="45" y="20" fill="#8A6D3B" font-size="10">split(_, 3)</text><text x="45" y="36" fill="#9A938A" font-size="8">hatch eggs</text></g><text x="235" y="82" fill="#9A5A12" font-size="14">-&gt;</text><g transform="translate(270,44)"><rect x="0" y="0" width="100" height="26" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><rect x="0" y="34" width="100" height="26" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><rect x="0" y="68" width="100" height="26" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="50" y="17" fill="#1a5c38" font-size="9">fresh key 0</text><text x="50" y="51" fill="#1a5c38" font-size="9">fresh key 1</text><text x="50" y="85" fill="#1a5c38" font-size="9">fresh key 2</text></g><text x="420" y="88" fill="#276b45" font-size="9">each one used ONCE</text><text x="260" y="168" fill="#9A938A" font-size="9">run out of fresh keys? split again for more</text></g></svg>
%%%

**What the parent-and-eggs picture gets right:** one key can produce many fresh, independent keys, and you use each fresh one exactly once. **Where it breaks down:** real eggs come from a parent that gets tired and can't lay forever, but a JAX key can be split as many times as you like — and, being deterministic, the *same* parent key always hatches the *same* set of babies (great for reproducibility).

#### The shape: split gives you a stack of keys
When you write `jax.random.split(key, 3)`, JAX hands back an array of **3 keys stacked together** — not 3 loose numbers. Each row is a full key (its own two numbers). The common habit is to peel off one to keep splitting from, and use the rest. Watch it:

%%% demo id=split label="split into fresh keys"
code: key = jax.random.PRNGKey(0)
code: keys = jax.random.split(key, 3)
code: print(keys.shape)                       # 3 keys, each 2 numbers
code: print(jax.random.normal(keys[0], (2,))) # different from...
code: print(jax.random.normal(keys[1], (2,))) # ...this one
out: (3, 2)
out: [-0.37  1.09]
out: [ 0.72  -1.44]
take: <b>Fresh keys → different numbers.</b> <code>split(key, 3)</code> gives shape <code>(3, 2)</code>: 3 keys, each holding 2 numbers. Feed two different keys to <code>normal</code> and you finally get two <b>different</b> draws — the reuse trap is gone.
%%%

#### The everyday pattern: key, subkey = split
So often you want "one to keep, one to use right now" that there's a standard two-line dance. You split into a `key` you keep for next time and a [[subkey||A fresh key you got from splitting. You use a subkey once for a random call, and keep the other key to split again next time.]] you use once and throw away. This "keep one, use one" habit is called **threading** the key through your program:

%%% formula
expr: key, subkey = jax.random.split(key)
note: keep `key` for next time, use `subkey` once right now — then never touch this subkey again
%%%

#### Threading a key through a loop
This shines in a loop. Say you want a *different* random number every step. If you reuse one key (left side below), every step draws the identical number — the trap. If you split a fresh subkey each step (right side), every step is genuinely different. Here's the before-and-after in one picture:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A before and after comparison of a loop. On the left, labelled wrong, the same key feeds every step and all three steps draw the identical number 0.9, 0.9, 0.9. On the right, labelled right, each step splits a fresh subkey from the carried key, so the three steps draw different numbers 0.9, minus 0.4, 1.3. Arrows show the carried key threading from step to step on the right side."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Loop: reuse one key (all same) vs split a fresh subkey each step (all different)</text><text x="130" y="36" fill="#C93B3B" font-size="10">WRONG: reuse the same key</text><g transform="translate(40,46)"><rect x="0" y="0" width="180" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="17" fill="#8a3b3b">step 1: normal(key) → 0.9</text><rect x="0" y="34" width="180" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="51" fill="#8a3b3b">step 2: normal(key) → 0.9</text><rect x="0" y="68" width="180" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="85" fill="#8a3b3b">step 3: normal(key) → 0.9</text></g><text x="130" y="158" fill="#C93B3B">all three identical ✗</text><text x="390" y="36" fill="#276b45" font-size="10">RIGHT: split a fresh subkey</text><g transform="translate(300,46)"><rect x="0" y="0" width="200" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="17" fill="#1a5c38">key,sub=split(key); normal(sub)→0.9</text><rect x="0" y="34" width="200" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="51" fill="#1a5c38">key,sub=split(key); normal(sub)→-0.4</text><rect x="0" y="68" width="200" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="85" fill="#1a5c38">key,sub=split(key); normal(sub)→1.3</text></g><path d="M400 72 v10" stroke="#2D8B55" stroke-width="1.5" marker-end="url(#a)"/><path d="M400 106 v10" stroke="#2D8B55" stroke-width="1.5" marker-end="url(#a)"/><defs><marker id="a" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#2D8B55"/></marker></defs><text x="400" y="158" fill="#276b45">all three different ✓</text><text x="400" y="172" fill="#9A938A">the carried key threads down the steps</text></g></svg>
%%%

You now hold the master habit of JAX randomness: **split for fresh keys, use each key once, thread the leftover key onward.** Take a victory lap — this is the exact pattern used to shuffle data and set up models in real training code. There's one handy cousin of `split` worth meeting, coming right up.

@@@ concept id=c6 tag="fold_in" title="fold_in: a fresh key labelled by a number" gotit="Got fold_in"
Sometimes you don't want a random stack of fresh keys — you want a fresh key that is *tied to a specific label*, like "the key for step number 7" or "the key for worker number 3." JAX gives you `jax.random.fold_in` for exactly this. You hand it a key and a whole number (the label), and it hands back a fresh key that belongs to that label. Same key plus same label always gives the same fresh key — so you can ask for "the key for step 7" any time and get the same one back.

Picture a big keyring with a **key-cutting machine** at a hardware store. You bring your master key and say a number — "cut me key number 7." The machine carves a fresh key stamped `7`. Say `7` again and you get an identical copy; say `8` and you get a different key. You don't have to cut keys 0 through 6 first — you can jump straight to the one you want.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A master key and a number 7 go into a key-cutting machine labelled fold_in, and out comes a fresh key stamped 7. A note shows that asking for number 8 gives a different key, and asking for 7 again gives the same key. A label reads: fold_in makes a fresh key labelled by a number, jump straight to the one you want."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">fold_in(key, 7): a fresh key labelled 7 — jump straight to it</text><text x="55" y="50" fill="#6B645E">master key</text><g transform="translate(25,58)"><rect x="0" y="0" width="66" height="34" rx="9" fill="#EDE9E0" stroke="#8A8172" stroke-width="2"/><text x="33" y="22" fill="#3A342E" font-size="9">key</text></g><text x="110" y="76" fill="#5E5191" font-size="12">+ 7</text><g transform="translate(150,52)"><rect x="0" y="0" width="110" height="46" rx="6" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="55" y="21" fill="#8A6D3B" font-size="10">fold_in</text><text x="55" y="37" fill="#9A938A" font-size="8">cut key #7</text></g><text x="278" y="76" fill="#9A5A12" font-size="14">-&gt;</text><g transform="translate(305,52)"><rect x="0" y="0" width="110" height="46" rx="10" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="55" y="21" fill="#1a5c38" font-size="10">fresh key</text><text x="55" y="38" fill="#1a5c38" font-size="11">stamped 7</text></g><text x="440" y="72" fill="#276b45" font-size="9">ask 7 again →</text><text x="440" y="86" fill="#276b45" font-size="9">same key</text><text x="260" y="158" fill="#9A938A" font-size="9">a different number (8) → a different fresh key; no need to cut 0..6 first</text></g></svg>
%%%

**What the key-cutting picture gets right:** you pick a number and get a specific fresh key for it, jumping straight to the one you want, and the same request always gives the same cut. **Where it breaks down:** a real cutting machine wears down the blade over time; `fold_in` never wears out and is perfectly repeatable — number 7 always cuts the identical key.

#### split vs fold_in — same goal, two flavors
Both give you fresh keys and both cure the reuse trap. The difference is *how you ask*:

%%% table
:: :: `split(key, n)` :: `fold_in(key, i)`
What you get :: a stack of `n` fresh keys :: one fresh key for label `i`
You pick :: how many keys :: which numbered key
Best for :: "give me a handful of fresh keys" :: "the key for step/worker number `i`"
%%%

%%% demo id=foldin label="fold_in by label"
code: key = jax.random.PRNGKey(0)
code: print(jax.random.fold_in(key, 7))   # the key for label 7
code: print(jax.random.fold_in(key, 7))   # ask again → same
code: print(jax.random.fold_in(key, 8))   # different label → different key
out: [ 4146024105  2718843009]
out: [ 4146024105  2718843009]
out: [ 1737047158  3901994897]
take: <b>Label picks the key.</b> <code>fold_in(key, 7)</code> gives the same fresh key every time you ask for 7, but a different one for 8. Handy when each step or worker has a natural number to key off — no threading needed, just fold in the label.
%%%

Both `split` and `fold_in` are your tools for the golden rule: **one key, one use.** Now for the honest part — every powerful idea has limits, and keys have two you should know by heart.

@@@ concept id=c7 tag="The honest limits" title="Two honest limits of keys (and the engine inside)" gotit="Got the limits"
Being clear about what a tool *can't* do is what separates a real engineer from someone who memorized a trick. A JAX key has two honest limits, and once you see them, the whole design finally clicks.

Picture a **music box**. Turn the crank with a given starting position and it plays the exact same tune every time. That's a key: fully determined, perfectly repeatable — but it is *not* a magic source of fresh, unpredictable surprise, and it does *not* wind itself forward on its own.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A music box with a crank. Setting the same starting position always plays the same tune, illustrating that a key is deterministic and repeatable, not truly unpredictable. A second note shows the crank does not turn itself, illustrating that a key does not advance on its own. Two labels read: limit one, not true randomness; limit two, does not advance itself."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A key is a music box: same start → same tune, and it never cranks itself</text><g transform="translate(190,34)"><rect x="0" y="20" width="140" height="70" rx="8" fill="#EDE9E0" stroke="#8A8172" stroke-width="2"/><circle cx="70" cy="12" r="10" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><path d="M70 12 h16" stroke="#C99A12" stroke-width="3"/><text x="70" y="58" fill="#3A342E" font-size="9">same start</text><text x="70" y="74" fill="#3A342E" font-size="9">→ same tune</text></g><g transform="translate(20,52)"><rect x="0" y="0" width="150" height="52" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="75" y="20" fill="#8a3b3b" font-size="9">limit 1: not TRUE random</text><text x="75" y="38" fill="#8a3b3b" font-size="8">fully set by the key</text></g><g transform="translate(350,52)"><rect x="0" y="0" width="150" height="52" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="75" y="20" fill="#8a3b3b" font-size="9">limit 2: won't advance itself</text><text x="75" y="38" fill="#8a3b3b" font-size="8">you must split for a new one</text></g><text x="260" y="150" fill="#9A938A" font-size="9">both limits come from the same honesty: the key fully decides the output</text></g></svg>
%%%

**What the music-box picture gets right:** same starting crank position gives the same tune, and the box never turns its own crank — both of a key's limits in one image. **Where it breaks down:** a music box plays a tune a human wrote on purpose, while a key's output *looks* like scattered random noise — but under the surface it is just as fixed and repeatable as that tune.

#### Limit 1 — a key is not TRUE randomness
The numbers a key produces only *look* random. Given the key, they are completely decided in advance — that's why the same key repeats. So a key is **not** a source of fresh, unguessable surprise the way a coin flip in the real world is. For learning and training AI, that's exactly what you want (repeatable experiments). For things like passwords or secret codes, you'd need real-world entropy from elsewhere — a job for a different tool.

#### Limit 2 — a key does not advance itself
This is the reuse trap from a higher view. Unlike the old dice cup that "remembered where it left off" and moved forward on every shake, a key just sits there. Using it does **not** bump it to a new value. If you want a new draw, *you* must make a new key with `split` or `fold_in`. The key will never do it for you — and now you know that's a feature, not a flaw.

#### The engine inside: threefry
So how does a key turn into random-looking numbers so reliably? Under the surface, JAX uses a named recipe called [[threefry||The counting-based math recipe JAX uses under the hood to turn a key into random-looking numbers. It's deterministic and safe to run in parallel.]] — a *counter-based* generator. The picture is simple: threefry is a locked formula that takes your **key** plus a plain **position number** and computes the output straight from those two — no hidden running memory in between. That's why it can jump straight to position 5 without doing 0 through 4 first, and why thousands of positions can be computed at the same time without stepping on each other:

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="The threefry engine drawn as a locked formula box. On the left, a key and three position numbers 0, 1, 2 each feed independently into their own copy of the threefry box, and each produces its own output number. There is no arrow connecting the boxes to each other, showing there is no shared running memory, so the outputs can be computed in any order or all at once."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">threefry: output = f(key, position) — no shared memory, so any order / all at once</text><text x="45" y="52" fill="#1a5c38" font-size="10">key</text><g transform="translate(20,58)"><rect x="0" y="0" width="52" height="80" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="26" y="44" fill="#1a5c38" font-size="10">key</text></g><g transform="translate(120,52)"><rect x="0" y="0" width="120" height="26" rx="5" fill="#FDF3E7" stroke="#C99A12"/><text x="60" y="17" fill="#8A6D3B">f(key, pos=0)</text><rect x="0" y="34" width="120" height="26" rx="5" fill="#FDF3E7" stroke="#C99A12"/><text x="60" y="51" fill="#8A6D3B">f(key, pos=1)</text><rect x="0" y="68" width="120" height="26" rx="5" fill="#FDF3E7" stroke="#C99A12"/><text x="60" y="85" fill="#8A6D3B">f(key, pos=2)</text></g><text x="278" y="70" fill="#9A5A12" font-size="12">-&gt;</text><text x="278" y="104" fill="#9A5A12" font-size="12">-&gt;</text><text x="278" y="138" fill="#9A5A12" font-size="12">-&gt;</text><g transform="translate(305,52)"><rect x="0" y="0" width="90" height="26" rx="5" fill="#fff" stroke="#B8AEA2"/><text x="45" y="17" fill="#3A342E">0.9</text><rect x="0" y="34" width="90" height="26" rx="5" fill="#fff" stroke="#B8AEA2"/><text x="45" y="51" fill="#3A342E">-0.4</text><rect x="0" y="68" width="90" height="26" rx="5" fill="#fff" stroke="#B8AEA2"/><text x="45" y="85" fill="#3A342E">1.3</text></g><text x="440" y="70" fill="#276b45">no arrows</text><text x="440" y="84" fill="#276b45">between rows</text><text x="440" y="104" fill="#9A938A" font-size="8">= no shared</text><text x="440" y="116" fill="#9A938A" font-size="8">running memory</text><text x="260" y="168" fill="#9A938A">this is why keys are reproducible AND safe to run in parallel</text></g></svg>
%%%

You don't need threefry's inner math today; you just need what it buys you.

!!! c-info 🔬
<b>Optional (skippable) — why threefry matters.</b> A counter-based generator like threefry works like a locked formula: give it the key and a position number, and it computes the output directly from those two, with no hidden running memory. Because there's no shared memory to trip over, JAX can compute draw number 5 without first computing 0 through 4, and can safely run thousands of draws at the same time on many machines — every one lands on the exact number it should. That parallel-safety is the deep reason JAX chose explicit keys plus threefry over a hidden dice cup. You'll feel the payoff on a later day when you split keys across a whole batch at once.
!!!

You've now met keys honestly — their power *and* their two limits *and* the engine that makes them tick. That balanced, no-hype view is exactly how a staff engineer thinks about a tool. Let's gather the whole day onto one page.

@@@ concept id=c8 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — you've learned the JAX way of randomness, one of the first "aha" moments every JAX newcomer earns. Before we gather the pieces, here's a fresh everyday picture for the whole day. Think of today like a **recipe card taped to the fridge**. A recipe lists its steps in a fixed order — get out an ingredient, do the step, then move to the next; skip a step or use the same ingredient twice and the dish comes out wrong. Our day was a recipe just like that: get a key, use it *once* for one dish, then reach for a *fresh* ingredient (split) before the next dish. **Where the recipe card breaks down:** a cook improvises and a pinch more salt is fine, but a JAX key recipe is exact — the same key always cooks the identical dish, and that strictness is the whole point (it's what makes the result repeatable). Below is that recipe, step by step, plus a cheat-sheet to keep on the fridge.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A five step flow. Step one: make a key from a seed with PRNGKey. Step two: hand the key to a random call to get a draw, and the same key gives the same draw. Step three: never reuse a key or you get identical numbers. Step four: split one key into fresh keys, use each once. Step five: honest limits, not true randomness and it does not advance itself. Each step points to the next."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The whole recipe: make key → draw → don't reuse → split for fresh → know the limits</text><g transform="translate(6,54)"><rect x="0" y="0" width="90" height="60" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="45" y="26" fill="#8A6D3B">1 make key</text><text x="45" y="42" fill="#9A938A" font-size="8">PRNGKey(seed)</text></g><text x="100" y="88" fill="#9A5A12" font-size="13">-&gt;</text><g transform="translate(116,54)"><rect x="0" y="0" width="90" height="60" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="45" y="26" fill="#1a5c38">2 draw</text><text x="45" y="42" fill="#276b45" font-size="8">normal(key,...)</text></g><text x="210" y="88" fill="#9A5A12" font-size="13">-&gt;</text><g transform="translate(226,54)"><rect x="0" y="0" width="90" height="60" rx="8" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="45" y="26" fill="#C93B3B">3 don't reuse</text><text x="45" y="42" fill="#8a3b3b" font-size="8">same key=same nums</text></g><text x="320" y="88" fill="#9A5A12" font-size="13">-&gt;</text><g transform="translate(336,54)"><rect x="0" y="0" width="90" height="60" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="45" y="26" fill="#1a5c38">4 split</text><text x="45" y="42" fill="#276b45" font-size="8">fresh keys, use once</text></g><text x="430" y="88" fill="#9A5A12" font-size="13">-&gt;</text><g transform="translate(446,54)"><rect x="0" y="0" width="72" height="60" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"/><text x="36" y="26" fill="#5E5191">5 limits</text><text x="36" y="42" fill="#5E5191" font-size="8">no auto-advance</text></g><text x="260" y="156" fill="#9A938A">no hidden cup → explicit key → reproducible AND safe to run in parallel</text></g></svg>
%%%

The day in six beats:
- **The key.** Randomness in JAX starts with a [[key||a small token of chance you carry — two whole numbers made from a seed]] you make yourself: `key = jax.random.PRNGKey(0)`.
- **The contrast.** The old way (NumPy) hid randomness in one shared dice cup — [[global state||hidden shared memory every call touches]] — which breaks when JAX reorders or parallelizes your code. JAX threw the cup away.
- **The draw.** You hand a key to a random call to get numbers: `random.normal(key, ...)`. Same key in → same numbers out (that's reproducibility).
- **The trap.** Reuse one key for two calls and you get the **identical** draw twice — silently, no error. Cause: a key doesn't advance itself.
- **The fix.** `split` hatches fresh independent keys (`key, subkey = jax.random.split(key)`); use each key **once**, thread the leftover onward. `fold_in(key, i)` is the label-picking cousin.
- **The honest limits.** A key is *not* true randomness (fully decided by the key) and does *not* advance itself. Under the hood, the counter-based recipe **threefry** makes it reproducible and parallel-safe.

#### Cheat-sheet · the moves at a glance
%%% table
:: You want to… :: Do this :: Note
Make a key :: `key = jax.random.PRNGKey(0)` :: same seed → same key
Get random numbers :: `jax.random.normal(key, (n,))` :: same key → same numbers
Get fresh keys :: `key, subkey = jax.random.split(key)` :: use `subkey` once, keep `key`
Key for a label :: `jax.random.fold_in(key, i)` :: same label → same key
Two different draws :: split first, then draw :: never reuse one key
%%%

#### Cheat-sheet · the words you met today
%%% jargon
key | a small token of chance (two whole numbers) you carry and hand to every random call
seed | the plain number you pick to make a key; same seed → same key
draw | the random numbers a random call gives you back
global state | hidden shared memory (the old "dice cup") that every call touches — what JAX avoids
split | jax.random.split: turn one key into several fresh, independent keys
subkey | a fresh key from split; use it once, then throw it away
fold_in | jax.random.fold_in: make a fresh key labelled by a number
threefry | the counter-based recipe under JAX keys — deterministic and parallel-safe
reproducible | rerun with the same key and get the exact same "random" answer
%%%

That's the whole day. You now think in explicit keys, not a hidden dice cup — the exact mindset that makes JAX experiments repeatable and ready to run fast and in parallel. Next you'll meet **vmap**, the tool that runs one function across a whole batch at once — and you'll see keys split neatly across that batch.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: In JAX, where does the randomness for a random call come from? | a:2 | A hidden global setting inside the library | A random number the computer picks for you | A key you make and hand to the call yourself | The current time | fb: JAX has no hidden dice cup. You make a key with jax.random.PRNGKey(seed) and pass it explicitly to every random call — randomness is data you carry, not a secret the library keeps.
q: You call random.normal(key, (3,)) twice with the SAME key. What do you get? | a:1 | Two different sets of random numbers | The exact same three numbers both times | An error about reusing a key | Only zeros | fb: Same key in means same numbers out — no error, just silent duplicates. This is the key-reuse trap: a key does not advance itself, so you must split to get fresh keys for a second draw.
q: You need two DIFFERENT batches of random numbers. What's the right move? | a:2 | Reuse the same key twice | Call PRNGKey again with the same seed | Split the key: key, subkey = jax.random.split(key), and use fresh keys | Add 1 to the key by hand | fb: split turns one key into fresh, independent keys. Use each key once and thread the leftover key onward — that's the master habit of JAX randomness.
q: Which statement about a JAX key is TRUE? | a:1 | It automatically advances to a new value each time you use it | The same key always produces the same draw (it does not advance itself) | It is a source of true, unguessable randomness | It secretly changes when you look at it | fb: A key is deterministic and does not advance on its own — same key, same draw. That's what makes experiments reproducible, and it's why you must split for fresh keys instead of reusing one.
%%%

@@@ produce id=produce tag="Produce" title="Feel the key with your own hands" gotit="Done"
Time to feel today's rule for yourself. You'll make a key, draw numbers, watch the reuse trap bite, and fix it with `split`. **Predict first:** if you draw with the same key twice, will the two draws be the *same* or *different*? Then run it and **watch** which happens — and watch it flip once you split. Pick one path.

#### Option A · write it yourself
Create `sessions/m07-thinking-in-jax/day-02-prng-keys/experiment.py`. Start with `import jax` and `import jax.numpy as jnp`. **Print every result so you can see it happen.** (1) Make `key = jax.random.PRNGKey(0)` and **print** it — notice it's just two numbers. (2) Call `jax.random.normal(key, (3,))` twice with the **same** key, print both, and **observe** they are identical — that's the "same key, same draw" rule. (3) Show the trap: make two "batches" reusing one key and **notice** they're secretly the same. (4) Fix it: `key, subkey = jax.random.split(key)`, draw with `subkey`, then split again for another draw, and **observe** the two draws are now different. (5) Try `jax.random.fold_in(key, 7)` twice and **notice** it gives the same key both times, but `fold_in(key, 8)` gives a different one. Run with `python3 sessions/m07-thinking-in-jax/day-02-prng-keys/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Week 1 Day 2 (JAX PRNG keys) artifact.

Create sessions/m07-thinking-in-jax/day-02-prng-keys/experiment.py that, with a comment on each step and printing every result:
1. import jax and jax.numpy as jnp; make key = jax.random.PRNGKey(0) and PRINT it (it should be two integers).
2. Call jax.random.normal(key, (3,)) TWICE with the same key; print both. They must be IDENTICAL — this proves "same key, same draw" (reproducibility).
3. Show the key-reuse TRAP: make batch_a and batch_b both by calling jax.random.normal(key, (3,)) with the same key; print both and note they are secretly identical (no error).
4. FIX it with split: key, subkey = jax.random.split(key); draw d1 = jax.random.normal(subkey, (3,)); then key, subkey = jax.random.split(key) again; draw d2 = jax.random.normal(subkey, (3,)); print d1 and d2 and show they are now DIFFERENT.
5. Also print jax.random.split(key, 3).shape to show it returns a stack of 3 keys, shape (3, 2).
6. Show fold_in: print jax.random.fold_in(key, 7) twice (identical) and jax.random.fold_in(key, 8) once (different) — a fresh key labelled by a number.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (same key repeats, split makes it fresh)
- The key `jax.random.PRNGKey(0)` prints as **two integers** — plain data you hold.
- Drawing with the **same key twice** gives the **identical** numbers — proof of "same key, same draw."
- Reusing one key for two batches makes them **secret twins** — the silent trap, with no error.
- After `split`, the two draws are finally **different** — fresh keys cure the trap.
- `fold_in(key, 7)` gives the **same** key both times, but `fold_in(key, 8)` gives a **different** one — a fresh key picked by its label.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m07-thinking-in-jax/day-02-prng-keys/log.md`: (1) in one sentence, why JAX makes you carry a key instead of using a hidden dice cup; (2) what goes wrong if you reuse the same key, and the one move that fixes it; (3) in one sentence, why "the same key always gives the same numbers" is a feature, not a bug.

@@@ fin
