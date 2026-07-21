---
quest_id: wf6-d01-residual
mode: concept
donor: v9-base.donor
page_title: "Module 5a · Day 1 — Residual Connections"
module_label: "Module 05a · Represent · Day 1"
title: "Residual Connections"
subtitle: "The Shortcut That Lets Depth Work"
brand_sub: "Foundations · M5a Day 1"
spine: "shortcut"
nav_prev_href: "../../m04-first-model-mlp/review.html"
nav_prev_label: "M4 · Review Gate"
nav_next_href: "../day-02-layer-norm/lesson.html"
nav_next_label: "Layer Normalization"
fin_title: "Module 5a · Day 1 complete! 🏆"
fin_body: "Nice work — you've unlocked <b>Residual Connections</b>: the simple <b>shortcut</b> that adds a layer's input straight back onto its output, so the signal (and the training gradient) can skip past the layer whenever it needs to. That one trick is what lets a transformer stack dozens of layers deep without the signal fading away to nothing.<br>Next up: <b>Layer Normalization</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: residual connection output F(x)+x, keywords: [output = f(x) + x, add the original input back, keep the draft, add the edits]}
  - {topic: residual is the change F(x)=0 is identity, keywords: [residual, learn only the change, f(x) = 0, do nothing, only has to learn the]}
  - {topic: identity shortcut path carries x forward, keywords: [identity, carried straight through, passed through, +x, express lane]}
  - {topic: additive merge same shape element-wise, keywords: [element-wise, same shape, slot by slot, matching, line up]}
  - {topic: gradient highway signal skips blocks, keywords: [gradient highway, learning signal, skips the layer, doesn't shrink, never fades]}
  - {topic: transformer wraps residuals residual stream, keywords: [residual stream, attention, feed-forward, wraps, runs the whole model]}
---

@@@ hero
@lede Picture climbing a tall tower by walking up floor after floor — but every floor also has an **elevator door** that drops you straight through to the next one, unchanged, whenever the stairs get you lost. That little "skip straight through" door is today's whole idea: the **shortcut** connection. Yesterday you finished the MLP; today you learn the one small trick that lets a model be *deep* — dozens of layers stacked — without the signal getting muddled to mush on the way up. And here's the surprise: this exact shortcut is inside every layer of ChatGPT, and for years nobody knew how to train deep networks *without* it. Sounds like a lot for one "+ x"? Good — hold that; by the end it'll feel obvious, and a little bit magic.
@goal Together we'll meet the shortcut one step at a time — first the puzzle it solves (deep stacks that mysteriously get *worse*), then the tiny fix (add the input back), why that turns a layer into a "learn only the change" helper, how it opens a clear highway for the learning signal, and finally why a transformer wraps this same shortcut around every one of its parts. Every failure we meet is a little puzzle with a neat answer — and every time a formula shows up, we say it out loud in plain words first.

@@@ concept id=c1 tag="The puzzle" title="Why more layers made things worse" gotit="Got the puzzle"
Let's start with a real mystery that stumped researchers for years. You'd think a taller building is always better — more floors, more room, more everything. So a deeper network (more layers stacked up) should always be *smarter*, right?

Here's the puzzle: it wasn't. When people stacked plain layers deeper and deeper, the deep network sometimes trained *worse* than a shallow one — even on the training data it had already seen. Not overfitting, not a bug: just plain worse. Engineers gave this a name — the [[degradation problem||Deeper plain networks training worse than shallow ones, even on data they have already seen. Not overfitting — the extra layers simply fail to help.]] — and it made no sense at first.

Think of a **long line of kids playing telephone**. The first kid whispers a clear message. Each kid passes it to the next, but nobody hears perfectly, so a little gets garbled each time. By the twentieth kid, the message is mush — not because any one kid was bad, but because the message got squeezed through *so many* imperfect hand-offs. A deep plain network does the same to a signal: each layer reshapes it a little, and after enough layers the original meaning is buried.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A line of children playing telephone. The first child speaks a clear message, and as it passes down the line each child hears less, until the last child hears only a faint garbled mumble."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Telephone down a long line — the message turns to mush</text><circle cx="60" cy="100" r="20" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="60" y="106" text-anchor="middle" font-size="16">🗣️</text><rect x="30" y="56" width="60" height="22" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="60" y="72" text-anchor="middle" fill="#276b45" font-size="10">clear!</text><text x="98" y="105" fill="#B8AEA2">→</text><circle cx="160" cy="100" r="20" fill="#F1F7F3" stroke="#7FB79A"/><text x="160" y="106" text-anchor="middle" font-size="14">👂</text><rect x="134" y="62" width="52" height="18" rx="6" fill="#F1F7F3" stroke="#7FB79A"/><text x="160" y="75" text-anchor="middle" fill="#4a8f6a" font-size="9">a bit off</text><text x="212" y="105" fill="#B8AEA2">→</text><circle cx="280" cy="100" r="20" fill="#FBF6EE" stroke="#C7B486"/><text x="280" y="106" text-anchor="middle" font-size="14">👂</text><rect x="256" y="66" width="48" height="16" rx="6" fill="#FBF6EE" stroke="#C7B486"/><text x="280" y="78" text-anchor="middle" fill="#9A7A10" font-size="9">fuzzy</text><text x="332" y="105" fill="#B8AEA2">→ … →</text><circle cx="440" cy="100" r="20" fill="#FDECEC" stroke="#C93B3B"/><text x="440" y="106" text-anchor="middle" font-size="14">😕</text><text x="440" y="72" text-anchor="middle" fill="#C93B3B" font-size="9">"…mumble…"</text><text x="260" y="152" text-anchor="middle" fill="#C93B3B" font-size="10">each hand-off garbles a little → after many layers the signal is buried</text></g></svg>
%%%

**What the telephone picture gets right:** many imperfect hand-offs in a row can bury a clear starting message — that's exactly the deep-stack trouble. **Where it breaks down:** in telephone the kids *try* to copy the message, while a network layer is *supposed* to transform its input — the trap is that even a layer meant to "change almost nothing" struggles to pass the input through cleanly.

#### The heart of the trap — a layer can't easily "do nothing"
Here's the part that surprised everyone. Sometimes the best thing an extra layer can do is *nothing at all* — just pass its input straight through, untouched. We call passing something through unchanged the [[identity||The "do-nothing" mapping: output equals input, unchanged. Passing a signal through without altering it.]] mapping. You'd think "do nothing" is the easiest job in the world. But a plain layer computes a *transformation* — it multiplies and reshapes — so to leave its input perfectly untouched it must learn a very specific, fiddly set of weights that exactly cancel out to "no change." That is surprisingly hard, so it usually doesn't quite manage it, and the small errors pile up down the stack.

Watch the fade build up. Each layer keeps only a fraction of the signal it received, and after a few layers there's almost nothing left:

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Five bars getting shorter left to right. The first bar is full height labeled 1.0, then each bar is a fraction of the last: 0.7, 0.5, 0.34, 0.24, showing the signal shrinking as it passes through plain layers."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#C93B3B" font-weight="bold">Plain deep stack — signal shrinks layer by layer</text><line x1="50" y1="140" x2="490" y2="140" stroke="#E5DFD6" stroke-width="1.5"/><rect x="70"  y="50"  width="52" height="90" fill="#C93B3B" opacity="0.85"/><text x="96"  y="156" fill="#6B645E">layer 1</text><text x="96"  y="44"  fill="#6B645E">1.0</text><rect x="160" y="77"  width="52" height="63" fill="#C93B3B" opacity="0.7"/><text x="186" y="156" fill="#6B645E">layer 2</text><text x="186" y="71"  fill="#6B645E">0.7</text><rect x="250" y="95"  width="52" height="45" fill="#C93B3B" opacity="0.55"/><text x="276" y="156" fill="#6B645E">layer 3</text><text x="276" y="89"  fill="#6B645E">0.5</text><rect x="340" y="110" width="52" height="30" fill="#C93B3B" opacity="0.4"/><text x="366" y="156" fill="#6B645E">layer 4</text><text x="366" y="104" fill="#6B645E">0.34</text><rect x="430" y="119" width="52" height="21" fill="#C93B3B" opacity="0.3"/><text x="456" y="156" fill="#6B645E">layer 5</text><text x="456" y="113" fill="#6B645E">0.24</text><text x="260" y="172" fill="#C93B3B" font-size="10">the deeper you go, the less of the original survives</text></g></svg>
%%%

If that feels strange — "wait, *more* layers should help, not hurt" — you're in exactly the right place. Almost everyone finds the degradation problem baffling the first time. The fix, coming next, is so simple you'll wonder why it took years to find.

@@@ concept id=c2 tag="The fix" title="Add the input back — the shortcut" gotit="Got the shortcut"
So how do we fix a layer that can't easily "do nothing"? The trick is delightfully lazy: **let the layer compute its change, then add the original input back on at the end.**

Think of **editing a friend's essay with a red pen**. You don't rewrite the whole essay from scratch — that would be exhausting and risky. You keep their original words and just scribble a few edits in the margin. The finished essay is *their draft plus your edits*. If your edits are good, great; if you have nothing to add, you write *nothing* and their draft passes through untouched. That "keep the original, add only your changes" move is the **shortcut** connection — the star of today.

%%% svg
<svg viewBox="0 0 520 178" role="img" aria-label="An essay page with the original typed text kept, plus a few red-pen edits scribbled in the margin, and an arrow to the final essay which equals the original draft plus the edits."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Editing an essay = keep the draft, ADD your edits</text><rect x="40" y="34" width="150" height="118" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="56" y1="54" x2="174" y2="54" stroke="#C7BEDF" stroke-width="6"/><line x1="56" y1="70" x2="174" y2="70" stroke="#DDD6EA" stroke-width="5"/><line x1="56" y1="84" x2="150" y2="84" stroke="#DDD6EA" stroke-width="5"/><line x1="56" y1="98" x2="174" y2="98" stroke="#DDD6EA" stroke-width="5"/><path d="M120 62 q14 -8 24 2" fill="none" stroke="#C93B3B" stroke-width="1.6"/><text x="150" y="120" fill="#C93B3B" font-size="10">↑ red-pen edits</text><text x="115" y="144" text-anchor="middle" fill="#6B645E" font-size="10">draft + margin notes</text><text x="205" y="96" fill="#B8AEA2" font-size="16">→</text><rect x="235" y="34" width="150" height="118" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><line x1="251" y1="54" x2="369" y2="54" stroke="#7FB79A" stroke-width="6"/><line x1="251" y1="70" x2="369" y2="70" stroke="#A9D3BC" stroke-width="5"/><line x1="251" y1="84" x2="345" y2="84" stroke="#A9D3BC" stroke-width="5"/><line x1="251" y1="98" x2="369" y2="98" stroke="#A9D3BC" stroke-width="5"/><text x="310" y="130" text-anchor="middle" fill="#276b45" font-size="10">final = draft + edits</text><text x="450" y="80" text-anchor="middle" fill="#5E5191" font-size="11">no edits?</text><text x="450" y="96" text-anchor="middle" fill="#5E5191" font-size="11">draft passes</text><text x="450" y="112" text-anchor="middle" fill="#5E5191" font-size="11">through!</text></g></svg>
%%%

**What the red-pen picture gets right:** you keep the original and layer edits on top, and "no edits" leaves the draft perfectly intact — that's the shortcut in a nutshell. **Where it breaks down:** a red pen edits words you can read, while a network's edits are numbers added to a list of numbers — same spirit, no handwriting.

#### The one formula — said out loud first
Let's name the pieces. Call the layer's input `x` (the draft). Call whatever the layer *computes* `F(x)` — that's the layer's suggested change (the red-pen edits). A plain layer would just hand back `F(x)` and throw away the draft. The shortcut instead hands back the draft **plus** the change:

%%% formula
expr: output = F(x) + x
note: the layer's change F(x), plus the original input x added straight back on
%%%

In words: "the block's output is the change it computed, *plus* the input it started with." That little `+ x` on the end is the entire shortcut. Read it as: keep the draft, add the edits.

Here's the magic that solves yesterday's puzzle. What if the best move really is to "do nothing"? Then the layer just has to make its change `F(x)` equal to **zero** — write no edits — and the output becomes `0 + x = x`, the input passed through perfectly. Learning to output *nothing* is far easier than learning to exactly copy an input. So the shortcut makes "do nothing" the easy default, and the degradation problem melts away.

Predict before you peek: if a block's change comes out as all zeros, what's its output? Then run it.

%%% demo id=identity label="run it"
code: x = [2.0, -1.0, 5.0];  F_x = [0.0, 0.0, 0.0];  out = [F_x[i] + x[i] for i in range(3)]
out: out = [2.0, -1.0, 5.0]   # exactly x — the draft passed straight through
take: <b>When F(x) = 0, output = 0 + x = x.</b> The block does nothing, and the shortcut hands the input back untouched. Learning "add zero" is easy — that's why the shortcut makes deep stacks behave.
%%%

Now watch the *change* itself. Below, the block computes a small edit and the shortcut adds it back onto the input, value by value — the "+ x" merge in one picture:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three columns of numbers. Left: the input x is 2, minus 1, 5. Middle: the block's change F of x is 0.3, 0.1, minus 0.4. Right: their element-wise sum output is 2.3, minus 0.9, 4.6. Arrows show each input value added to its matching change."><g font-family="monospace" font-size="12" text-anchor="middle"><text x="80" y="26" fill="#5E5191" font-weight="bold">input x</text><text x="260" y="26" fill="#9A5A12" font-weight="bold">change F(x)</text><text x="450" y="26" fill="#276b45" font-weight="bold">output = F(x)+x</text><g fill="#5E5191"><rect x="50" y="42" width="60" height="30" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="80" y="62">2.0</text><rect x="50" y="82" width="60" height="30" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="80" y="102">-1.0</text><rect x="50" y="122" width="60" height="30" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="80" y="142">5.0</text></g><g fill="#9A5A12"><rect x="230" y="42" width="60" height="30" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="260" y="62">+0.3</text><rect x="230" y="82" width="60" height="30" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="260" y="102">+0.1</text><rect x="230" y="122" width="60" height="30" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="260" y="142">-0.4</text></g><g fill="#276b45"><rect x="420" y="42" width="60" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="450" y="62">2.3</text><rect x="420" y="82" width="60" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="450" y="102">-0.9</text><rect x="420" y="122" width="60" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="450" y="142">4.6</text></g><text x="170" y="62" fill="#6B645E">+</text><text x="170" y="102" fill="#6B645E">+</text><text x="170" y="142" fill="#6B645E">+</text><text x="360" y="62" fill="#6B645E">=</text><text x="360" y="102" fill="#6B645E">=</text><text x="360" y="142" fill="#6B645E">=</text><text x="260" y="184" fill="#276b45" font-size="10">each input value + its matching change → the output</text></g></svg>
%%%

**You just unlocked the whole idea.** Everything else today is showing you *why* this one `+ x` is so powerful.

@@@ concept id=c3 tag="Same shape" title="Why the two must line up to add" gotit="Got the merge"
The shortcut adds two things together: the change `F(x)` and the input `x`. But adding has a rule you already know from everyday life — the two things have to *match up*.

Think of **two shopping lists you want to combine by adding quantities**. List one says "3 apples, 2 milks, 1 bread." List two says "+1 apple, +0 milks, +2 breads." To add them you line them up slot for slot — apples onto apples, milk onto milk — and sum each pair: "4 apples, 2 milks, 3 breads." It only works because both lists have the **same slots in the same order**. If one list had a slot the other didn't, you couldn't add them cleanly.

The input `x` and the change `F(x)` are just lists of numbers like that. Adding them is [[element-wise||Add two equal-length lists slot by slot: first plus first, second plus second, and so on. The two lists must be the same length.]] — first plus first, second plus second. So the block's output must come out the **same shape** as its input, or the "+ x" has nothing to line up against.

%%% svg
<svg viewBox="0 0 520 186" role="img" aria-label="Two shopping lists side by side with the same three slots — apples, milk, bread — added slot by slot to give a combined list. A note shows both lists must have the same slots to add."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Add two lists slot-by-slot — only if the slots line up</text><rect x="40" y="36" width="120" height="118" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="100" y="54" text-anchor="middle" fill="#5E5191" font-weight="bold">list x</text><text x="100" y="80" text-anchor="middle" fill="#5E5191">🍎 3</text><text x="100" y="106" text-anchor="middle" fill="#5E5191">🥛 2</text><text x="100" y="132" text-anchor="middle" fill="#5E5191">🍞 1</text><text x="185" y="80" fill="#6B645E">+</text><text x="185" y="106" fill="#6B645E">+</text><text x="185" y="132" fill="#6B645E">+</text><rect x="210" y="36" width="120" height="118" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="270" y="54" text-anchor="middle" fill="#9A7A10" font-weight="bold">list F(x)</text><text x="270" y="80" text-anchor="middle" fill="#9A7A10">🍎 +1</text><text x="270" y="106" text-anchor="middle" fill="#9A7A10">🥛 +0</text><text x="270" y="132" text-anchor="middle" fill="#9A7A10">🍞 +2</text><text x="352" y="96" fill="#6B645E">=</text><rect x="378" y="36" width="120" height="118" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="438" y="54" text-anchor="middle" fill="#276b45" font-weight="bold">combined</text><text x="438" y="80" text-anchor="middle" fill="#276b45">🍎 4</text><text x="438" y="106" text-anchor="middle" fill="#276b45">🥛 2</text><text x="438" y="132" text-anchor="middle" fill="#276b45">🍞 3</text><text x="260" y="176" text-anchor="middle" fill="#6B645E" font-size="10">same three slots, same order → clean add</text></g></svg>
%%%

**What the shopping-list picture gets right:** you can only add two lists when they share the same slots in the same order — that's the matching-shape rule exactly. **Where it breaks down:** groceries are whole numbers — you'd never get "2.3 apples" — but a network's slots hold any decimal — the *shape* rule is what carries over, not the counting.

#### The block becomes a "change helper"
Here's the lovely side effect. Because the output is `F(x) + x`, the block never has to reproduce the whole input — the `+ x` already carries the input along for free. So `F(x)` only has to learn the **change**: what to *add* or *adjust*, not the entire answer. We call `F(x)` the [[residual||The "leftover" change a block learns to add on top of its input, rather than the whole output. In output = F(x) + x, the residual is F(x).]] — the little leftover the block is responsible for. That's where today's title comes from: a *residual* connection.

#### The one thing that can go wrong — and its fix
So what happens if a block's output *doesn't* match the input's shape? Then the "+ x" has mismatched slots and can't line up — the add breaks. Here's the before-and-after in one picture: a mismatched merge fails, and the fix is to make the block's output the same shape as its input.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Top row: a 3-slot input and a 2-slot block output cannot be added — marked with a red X. Bottom row: the same input and a 3-slot block output add cleanly — marked with a green check."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#C93B3B" font-weight="bold">Mismatch breaks the add — matching shape fixes it</text><text x="55" y="52" fill="#5E5191">x: 3 slots</text><rect x="30" y="60" width="20" height="20" fill="#EFEAF7" stroke="#7C6DAA"/><rect x="52" y="60" width="20" height="20" fill="#EFEAF7" stroke="#7C6DAA"/><rect x="74" y="60" width="20" height="20" fill="#EFEAF7" stroke="#7C6DAA"/><text x="120" y="74" fill="#6B645E">+</text><text x="180" y="52" fill="#9A7A10">F(x): 2 slots</text><rect x="150" y="60" width="20" height="20" fill="#FCF3DC" stroke="#C99A12"/><rect x="172" y="60" width="20" height="20" fill="#FCF3DC" stroke="#C99A12"/><text x="250" y="74" fill="#C93B3B" font-size="18" font-weight="bold">✗</text><text x="330" y="74" fill="#C93B3B" text-anchor="start">3rd slot has nothing to add to → breaks</text><line x1="30" y1="98" x2="490" y2="98" stroke="#E5DFD6" stroke-dasharray="4,3"/><text x="55" y="130" fill="#5E5191">x: 3 slots</text><rect x="30" y="138" width="20" height="20" fill="#EFEAF7" stroke="#7C6DAA"/><rect x="52" y="138" width="20" height="20" fill="#EFEAF7" stroke="#7C6DAA"/><rect x="74" y="138" width="20" height="20" fill="#EFEAF7" stroke="#7C6DAA"/><text x="120" y="152" fill="#6B645E">+</text><text x="185" y="130" fill="#9A7A10">F(x): 3 slots</text><rect x="150" y="138" width="20" height="20" fill="#FCF3DC" stroke="#C99A12"/><rect x="172" y="138" width="20" height="20" fill="#FCF3DC" stroke="#C99A12"/><rect x="194" y="138" width="20" height="20" fill="#FCF3DC" stroke="#C99A12"/><text x="250" y="152" fill="#2D8B55" font-size="18" font-weight="bold">✓</text><text x="330" y="152" fill="#276b45" text-anchor="start">every slot lines up → clean add</text><text x="260" y="196" fill="#6B645E" font-size="10">the fix: design the block so its output shape = its input shape</text></g></svg>
%%%

**Good news:** in a transformer this rule is easy to honor. Every sub-block is built to keep the same width all the way through, so its output already matches its input, and the "+ x" just works. You'll see exactly which sub-blocks in a moment.

@@@ concept id=c4 tag="The highway" title="A clear lane for the learning signal" gotit="Got the highway"
Now for the part that makes the shortcut a superpower — not just at answering, but at *learning*. To improve, a network sends a "learning signal" backward through its layers, telling each one how to adjust. (You met this idea with activations: the signal gets multiplied down the line and can fade to a whisper.) The shortcut opens a clear lane for that signal.

Picture a **highway with an express lane running beside the exits**. Normal traffic weaves off at every exit, slows for lights, gets tangled — that's the signal squeezing through each layer. But the express lane runs straight past all of it. A car in the express lane reaches the far end at full speed, untouched by any of the local slowdowns. The shortcut's "+ x" is that express lane: because the input is added straight through, the signal has a route that skips the layer entirely whenever it needs to.

%%% svg
<svg viewBox="0 0 520 178" role="img" aria-label="A highway with a tangled local road that slows down at every exit and traffic light, and a straight express lane beside it that runs clear from start to finish untouched."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Express lane = the shortcut skips the slowdowns</text><path d="M40 120 C 110 120, 110 95, 180 95 C 250 95, 250 122, 320 122 C 390 122, 390 98, 470 98" fill="none" stroke="#C93B3B" stroke-width="7" opacity="0.35"/><circle cx="110" cy="108" r="4" fill="#C93B3B"/><circle cx="250" cy="122" r="4" fill="#C93B3B"/><circle cx="390" cy="110" r="4" fill="#C93B3B"/><text x="255" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">local road: slows at every exit (each layer)</text><line x1="40" y1="55" x2="470" y2="55" stroke="#2D8B55" stroke-width="5"/><polygon points="470,50 484,55 470,60" fill="#2D8B55"/><text x="255" y="44" text-anchor="middle" fill="#276b45" font-size="10">express lane (the "+ x"): straight through, full speed</text><circle cx="40" cy="55" r="5" fill="#2D8B55"/><text x="30" y="76" fill="#6B645E">start</text><text x="466" y="40" text-anchor="end" fill="#6B645E">end</text></g></svg>
%%%

**What the express-lane picture gets right:** a straight bypass reaches the end untouched while local traffic crawls — that's the learning signal skipping past the layer on the "+ x" route. **Where it breaks down:** real highways have one express lane, but a network has this bypass at *every* block, so the signal has a clear lane the whole way up and down the stack.

#### See the fade — and the rescue — in one picture
Remember the shrinking bars from the puzzle? Without a shortcut, the learning signal shrinks a little at each layer and can fade to almost nothing deep in the stack. *With* the shortcut, the "+ x" express lane keeps a full-strength copy flowing through every layer. Same stack, two very different fates:

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Two rows of bars. Top row, no shortcut: signal strength 1.0, 0.6, 0.36, 0.22, 0.13, shrinking down the stack. Bottom row, with shortcut: strength stays 1.0 at every layer, all bars full height."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="60" y="20" fill="#C93B3B" font-weight="bold" text-anchor="start">no shortcut · signal fades down the stack</text><rect x="60"  y="34" width="46" height="58" fill="#C93B3B"/><text x="83"  y="105" fill="#6B645E">1.0</text><rect x="140" y="57" width="46" height="35" fill="#C93B3B" opacity="0.8"/><text x="163" y="105" fill="#6B645E">0.6</text><rect x="220" y="70" width="46" height="22" fill="#C93B3B" opacity="0.6"/><text x="243" y="105" fill="#6B645E">0.36</text><rect x="300" y="79" width="46" height="13" fill="#C93B3B" opacity="0.45"/><text x="323" y="105" fill="#6B645E">0.22</text><rect x="380" y="84" width="46" height="8"  fill="#C93B3B" opacity="0.3"/><text x="403" y="105" fill="#C93B3B">0.13</text><text x="60" y="140" fill="#2D8B55" font-weight="bold" text-anchor="start">with shortcut · "+ x" keeps signal full-strength</text><rect x="60"  y="150" width="46" height="55" fill="#2D8B55"/><text x="83"  y="217" fill="#6B645E">1.0</text><rect x="140" y="150" width="46" height="55" fill="#2D8B55"/><text x="163" y="217" fill="#6B645E">1.0</text><rect x="220" y="150" width="46" height="55" fill="#2D8B55"/><text x="243" y="217" fill="#6B645E">1.0</text><rect x="300" y="150" width="46" height="55" fill="#2D8B55"/><text x="323" y="217" fill="#6B645E">1.0</text><rect x="380" y="150" width="46" height="55" fill="#2D8B55"/><text x="403" y="217" fill="#6B645E">1.0</text><text x="460" y="66" fill="#9A938A" text-anchor="start">layer 1…5 deep</text></g></svg>
%%%

The deep reason is a one-liner: because the output is `F(x) + x`, part of the signal travels the `+ x` route completely unchanged. No layer gets to squeeze *that* copy, so even if `F(x)`'s route fades, the shortcut copy arrives intact. This is why people call it a [[gradient highway||The straight "+ x" route the learning signal (the gradient) travels without shrinking, so very deep networks stay trainable.]] — the *gradient* is just the formal name for that learning signal.

!!! c-info 🧮
<b>Optional (skippable) — the "+ 1" peek.</b> If you like the algebra: because the output is <code>F(x) + x</code>, when the learning signal flows back through the block it gets multiplied by "(whatever <code>F</code> does) <b>plus 1</b>." That <b>+ 1</b> is the express lane written as math — even if the "whatever <code>F</code> does" part shrinks toward zero, the <code>+ 1</code> guarantees the signal never fully vanishes. You'll meet the exact backward-pass math that produces this <code>+ 1</code> in a later gradient-flow deep-dive; today the express-lane picture is all you need.
!!!

**You can now explain, in one sentence, why deep networks became trainable:** the shortcut gives the learning signal a lane that skips every layer, so it never fades to nothing. That's a real interview beat, and you have it.

@@@ concept id=c5 tag="In a transformer" title="The residual stream through the whole model" gotit="Got the stream"
Now let's put the shortcut where you'll actually meet it: inside a transformer, the model behind ChatGPT. A transformer is a tall stack of blocks, and each block has two working parts — an **attention** part and a **feed-forward** part. (You'll open up what's inside each on later days; today just treat them as two helpers.) The beautiful thing is that a shortcut wraps around *each* of them.

Picture a **conveyor belt carrying a box down a factory line**. The box rides straight along the belt from start to finish — that's your signal moving through the model. At each station, a worker leans over and *adds* something to the box (a label, a part, a note) without taking the box off the belt. The box keeps rolling, a little richer at every station, and the belt never stops. That belt is the [[residual stream||The running signal that flows straight through a transformer from bottom to top. Each sub-block reads it, computes a change, and adds that change back onto the stream — so the stream carries a running total of everything the model has added.]] — the shortcut, running the whole height of the model.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A conveyor belt running left to right carrying a box. Two workers at two stations each add something to the box as it passes, without removing it from the belt, so the box arrives carrying everything that was added."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Conveyor belt = the residual stream; workers ADD to the box</text><rect x="30" y="110" width="460" height="10" rx="5" fill="#D8D2C8"/><circle cx="55" cy="132" r="9" fill="#FDF9F3" stroke="#B8AEA2"/><circle cx="150" cy="132" r="9" fill="#FDF9F3" stroke="#B8AEA2"/><circle cx="260" cy="132" r="9" fill="#FDF9F3" stroke="#B8AEA2"/><circle cx="370" cy="132" r="9" fill="#FDF9F3" stroke="#B8AEA2"/><circle cx="465" cy="132" r="9" fill="#FDF9F3" stroke="#B8AEA2"/><rect x="40" y="86" width="30" height="24" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="55" y="80" text-anchor="middle" fill="#276b45" font-size="9">box in</text><rect x="160" y="50" width="90" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="205" y="69" text-anchor="middle" fill="#5E5191">attention</text><line x1="205" y1="80" x2="205" y2="104" stroke="#7C6DAA" stroke-width="1.5"/><polygon points="200,98 205,108 210,98" fill="#7C6DAA"/><text x="205" y="128" text-anchor="middle" fill="#5E5191" font-size="14">+</text><rect x="300" y="50" width="110" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="355" y="69" text-anchor="middle" fill="#9A7A10">feed-forward</text><line x1="355" y1="80" x2="355" y2="104" stroke="#C99A12" stroke-width="1.5"/><polygon points="350,98 355,108 360,98" fill="#C99A12"/><text x="355" y="128" text-anchor="middle" fill="#9A7A10" font-size="14">+</text><rect x="450" y="86" width="30" height="24" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="465" y="80" text-anchor="middle" fill="#276b45" font-size="9">box out</text><text x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="10">each part reads the box, computes a change, ADDS it back — box never leaves the belt</text></g></svg>
%%%

**What the conveyor picture gets right:** the box (the signal) rides straight through and each station only *adds* to it — never replaces it — exactly like `F(x) + x` at every sub-block. **Where it breaks down:** a factory worker adds a physical part, while a transformer sub-block adds a list of numbers onto the running list — same "add, don't replace," no bolts.

#### How the shortcut wraps each part
Zoom in on a single block. The signal comes in as `x`. Attention computes its change and the shortcut adds it back: `x + attention(x)`. That result flows into feed-forward, which computes *its* change and the shortcut adds it back too. So the same `+ x` move happens twice per block, and stacking many blocks builds one long stream that runs the full height of the model:

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="A vertical residual stream running bottom to top. The input enters, goes through an attention sub-block whose output is added back onto the stream, then through a feed-forward sub-block whose output is also added back, producing the block output. A straight arrow on the side shows the stream passing straight through both additions."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">One block: the stream + attention, then + feed-forward</text><line x1="150" y1="235" x2="150" y2="40" stroke="#2D8B55" stroke-width="4"/><polygon points="145,44 150,32 155,44" fill="#2D8B55"/><text x="150" y="248" fill="#276b45">input x</text><rect x="250" y="185" width="120" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="310" y="204" fill="#5E5191">attention F₁</text><path d="M150 200 L250 200" stroke="#7C6DAA" stroke-width="1.5" fill="none"/><path d="M370 200 C 420 200, 150 175, 150 160" stroke="#7C6DAA" stroke-width="1.5" fill="none"/><circle cx="150" cy="150" r="11" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="150" y="155" fill="#276b45" font-size="14">+</text><text x="205" y="140" fill="#6B645E" font-size="10">x + attention(x)</text><rect x="250" y="95" width="120" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="310" y="114" fill="#9A7A10">feed-fwd F₂</text><path d="M150 110 L250 110" stroke="#C99A12" stroke-width="1.5" fill="none"/><path d="M370 110 C 420 110, 150 80, 150 66" stroke="#C99A12" stroke-width="1.5" fill="none"/><circle cx="150" cy="56" r="11" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="150" y="61" fill="#276b45" font-size="14">+</text><text x="150" y="30" fill="#276b45">block output ↑</text><text x="440" y="140" fill="#276b45" font-size="10" text-anchor="middle">stream runs</text><text x="440" y="155" fill="#276b45" font-size="10" text-anchor="middle">straight up</text></g></svg>
%%%

#### One last puzzle — and where we solve it
Here's a fair worry: if every sub-block keeps *adding* to the stream, doesn't the stream get bigger and bigger — a running total that could balloon out of control after dozens of blocks? Yes, that's a genuine risk, and it has a tidy name and a tidy fix. The fix is a small step called **layer normalization** that gently re-scales the stream so it stays a sensible size no matter how many blocks add to it. That's tomorrow's whole lesson — so hold the worry; you've spotted the exact problem Day 2 solves.

**Victory lap:** you now know the shape of a transformer's spine — a stream that flows straight up, with attention and feed-forward each adding their change back via the shortcut. That mental picture will carry you through the rest of this module.

@@@ concept id=c6 tag="What it isn't" title="The shortcut helps you train, not think" gotit="Got the limit"
Before we wrap up, one honest caveat — the kind a good engineer always keeps in mind. It's tempting to think the shortcut makes the model *smarter*. It doesn't, and knowing that is a sign you really get it.

Think of **training wheels on a bike**. Training wheels don't make the bike go faster or steer better — the bike's real ability comes from its gears and your legs. What the training wheels do is let you *keep riding without falling over* long enough to actually get good. The shortcut is like that: it doesn't add any new thinking power on its own. All the real work — spotting patterns, mixing words — still happens inside `F(x)` (the attention and feed-forward parts). The shortcut just keeps the tall stack from "falling over" so it can be trained at all.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A child's bike with training wheels. A note points out that the training wheels only keep it upright, while the pedals and gears provide the actual speed and power."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Training wheels = keep it upright, not make it faster</text><circle cx="180" cy="110" r="34" fill="none" stroke="#5E5191" stroke-width="3"/><circle cx="320" cy="110" r="34" fill="none" stroke="#5E5191" stroke-width="3"/><line x1="180" y1="110" x2="250" y2="60" stroke="#5E5191" stroke-width="2.5"/><line x1="250" y1="60" x2="320" y2="110" stroke="#5E5191" stroke-width="2.5"/><line x1="180" y1="110" x2="250" y2="110" stroke="#5E5191" stroke-width="2.5"/><circle cx="250" cy="110" r="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><circle cx="345" cy="132" r="14" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="345" y="158" text-anchor="middle" fill="#276b45" font-size="9">training wheel</text><text x="435" y="120" text-anchor="middle" fill="#276b45" font-size="9">↖ keeps it</text><text x="435" y="133" text-anchor="middle" fill="#276b45" font-size="9">upright only</text><text x="250" y="90" text-anchor="middle" fill="#9A7A10" font-size="9">pedals+gears</text><text x="250" y="45" text-anchor="middle" fill="#9A7A10" font-size="9">= real power (F)</text></g></svg>
%%%

**What the training-wheels picture gets right:** they keep you upright so you can practice, but the speed comes from elsewhere — just as the shortcut keeps a deep stack trainable while `F(x)` does the real work. **Where it breaks down:** you eventually take training wheels *off* a bike, but the shortcut stays inside the model forever — deep networks never outgrow needing it.

So the shortcut has two humble rules to remember: it needs the block's output to be the **same shape** as the input (so the "+ x" can line up), and it adds **no new power** by itself. It's an optimization aid — a brilliant one — not a new kind of layer.

!!! c-ok 🎤
<b>Once it clicks, here's how you'd explain it (say, in an interview):</b> "A residual connection computes <code>output = F(x) + x</code> — the sub-block learns only the residual change <code>F(x)</code> and the input <code>x</code> is added straight back. That makes the identity easy to represent (just push <code>F(x)</code> toward zero), which fixes the degradation problem, and it opens a gradient highway: the backward signal flows through the <code>+ x</code> route unshrunk, so very deep stacks stay trainable. The one constraint is matching dimensions, since the merge is element-wise. In a transformer a residual wraps both the attention and feed-forward sub-blocks, forming a residual stream that runs the whole model — and we pair it with layer norm so the stream doesn't blow up as blocks keep adding to it." Notice you can already say every piece of that — that's how far you've come today.
!!!

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you did the whole thing. Before the cheat-sheets, one picture to hold the whole day.

Think of **stacking a sandwich on a tray as it slides down the counter**. You start with a slice of bread on the tray — that's your input `x`. At each station a helper *adds* one ingredient on top — cheese, tomato, lettuce — the change `F(x)`. The helper never throws the sandwich away and starts over; they only add. The tray slides straight through, so the bread you started with is still there at the end, now carrying everything that was added: bread + cheese + tomato + lettuce. That is the shortcut, `output = F(x) + x`, running the whole counter. **Where this breaks down:** you could lift a slice of tomato back off a real sandwich, but once a network adds `F(x)` onto the stream the sum is one blended list of numbers — you can't peel a single ingredient back out.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A sandwich being built on a tray sliding down a counter left to right. It starts as a single slice of bread, then stations add cheese, tomato, and lettuce on top, and the finished stack arrives on the tray carrying the original bread plus everything added."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Sandwich on a sliding tray: keep the bread, ADD each layer</text><line x1="30" y1="150" x2="490" y2="150" stroke="#D8D2C8" stroke-width="4"/><rect x="42" y="132" width="56" height="10" rx="3" fill="#E7C892" stroke="#C99A12"/><text x="70" y="126" fill="#9A7A10">bread = x</text><text x="70" y="168" fill="#6B645E">start</text><text x="120" y="120" fill="#B8AEA2" font-size="14">→</text><rect x="152" y="132" width="56" height="10" rx="3" fill="#E7C892" stroke="#C99A12"/><rect x="152" y="122" width="56" height="10" rx="3" fill="#F4D96B" stroke="#C99A12"/><text x="180" y="112" fill="#9A7A10">+ cheese</text><text x="230" y="120" fill="#B8AEA2" font-size="14">→</text><rect x="262" y="132" width="56" height="10" rx="3" fill="#E7C892" stroke="#C99A12"/><rect x="262" y="122" width="56" height="10" rx="3" fill="#F4D96B" stroke="#C99A12"/><rect x="262" y="112" width="56" height="10" rx="3" fill="#E58B76" stroke="#C93B3B"/><text x="290" y="102" fill="#C93B3B">+ tomato</text><text x="340" y="120" fill="#B8AEA2" font-size="14">→</text><rect x="378" y="132" width="66" height="10" rx="3" fill="#E7C892" stroke="#C99A12"/><rect x="378" y="122" width="66" height="10" rx="3" fill="#F4D96B" stroke="#C99A12"/><rect x="378" y="112" width="66" height="10" rx="3" fill="#E58B76" stroke="#C93B3B"/><rect x="378" y="102" width="66" height="10" rx="3" fill="#8FBF6B" stroke="#2D8B55"/><text x="411" y="92" fill="#276b45">+ lettuce</text><text x="411" y="168" fill="#276b45">= F(x)+x</text><text x="260" y="176" fill="#6B645E" font-size="10">the tray (stream) carries the bread straight through — helpers only add layers</text></g></svg>
%%%

**The day as five signposts, in order** — each one only makes sense because of the one before it:
- **1 · The puzzle.** Stacking plain layers deeper sometimes trained *worse* (the degradation problem), because a layer can't easily learn to "do nothing" and small errors pile up.
- **2 · The fix.** The **shortcut**: `output = F(x) + x` — compute the change, then add the original input back. Now "do nothing" is easy: just make `F(x) = 0`.
- **3 · The rule.** The add is **element-wise**, so the block's output must be the **same shape** as its input. `F(x)` learns only the **residual** (the change), not the whole answer.
- **4 · The highway.** Because `x` is added straight through, the learning signal (the gradient) has a lane that skips the layer and never fades — the **gradient highway** that made very deep nets trainable.
- **5 · In a transformer.** A shortcut wraps both the attention and feed-forward sub-blocks, forming a **residual stream** that runs the whole model. (Tomorrow: layer norm keeps that stream from ballooning.)

#### Cheat-sheet · the shortcut at a glance
%%% table
:: Piece :: In plain words :: Why it matters
`x` :: the input (the bread) :: carried straight through by the shortcut
`F(x)` :: the residual — the change the block adds :: only has to learn the *edit*, not the whole answer
`+ x` :: add the input back :: makes "do nothing" easy (F(x)=0 → output=x) and opens the gradient highway
same shape :: output shape = input shape :: the merge is element-wise, so the slots must line up
residual stream :: the running signal through a transformer :: attention & feed-forward each add their change onto it
%%%

#### Cheat-sheet · the words you met today
%%% jargon
shortcut (residual) connection | add a block's input back onto its output: output = F(x) + x
residual F(x) | the change a block learns to add, rather than the whole output
identity | the do-nothing mapping — output equals input, made easy by F(x) = 0
degradation problem | deeper plain networks training worse, because "do nothing" is hard to learn
element-wise | adding two equal-length lists slot by slot; the shapes must match
gradient highway | the "+ x" route the learning signal travels unshrunk, so deep nets stay trainable
residual stream | the signal that flows straight up a transformer, each sub-block adding onto it
layer normalization | tomorrow's fix that keeps the residual stream from ballooning as blocks add to it
%%%

That's the whole day. Next you'll learn the small step that keeps this stream from growing too large — **layer normalization**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does a residual (shortcut) connection compute? | a:2 | It replaces the input with the block's output | It multiplies the input by the block's output | output = F(x) + x — the block's change, plus the original input added back | It averages the input and the output | fb: The shortcut keeps the input and adds the block's change on top: output = F(x) + x. Keep the draft, add the edits.
q: Why does the shortcut fix the "degradation problem" (deep plain nets training worse)? | a:1 | It adds more parameters to each layer | It makes the identity easy: to "do nothing" the block just sets F(x)=0, so output = 0 + x = x | It removes layers from the network | It uses a faster kind of matmul | fb: A plain layer struggles to learn to copy its input; the shortcut makes copying the default — just push the change F(x) toward 0.
q: Why is the residual called a "gradient highway"? | a:2 | Because it speeds up matrix multiplication | Because it uses GPU highways | Because the input is added straight through, the learning signal has a route that skips the layer and doesn't shrink toward zero across many layers | Because it removes the need for an activation function | fb: The "+ x" carries a full-strength copy of the signal past each block, so even very deep stacks stay trainable.
q: For the "+ x" merge to work, what must be true of the block's output? | a:1 | It must be a single number | It must be the same shape as the input, because the add is element-wise (slot by slot) | It must be larger than the input | It must always be zero | fb: You can only add two lists slot by slot when they have the same slots — so the block's output shape must match its input shape.
%%%

@@@ produce id=produce tag="Produce" title="Watch a deep stack fade — then add the shortcut" gotit="Done"
Time to see today's big idea with your own eyes. You'll push a signal through many plain layers and watch it fade toward nothing — then add the shortcut and watch a full-strength copy survive all the way through. **Predict first:** after, say, 30 plain layers that each shrink the signal a little, will much of the original be left? Then run it and **watch** the difference the "+ x" makes. Pick one path.

#### Option A · write it yourself
Create `sessions/m05a-text-transformer/day-01-residual-connections/experiment.py`. Start a signal vector `x` (for example `np.ones(4)`). Define a simple block `F(x)` that shrinks and reshapes its input a little (for example `0.8 * (W @ x)` for a fixed small matrix `W`). **Notice** three things: (1) run `x` through 30 plain blocks (`x = F(x)` each time) and print how the size of `x` shrinks toward 0; (2) run it through 30 *shortcut* blocks (`x = F(x) + x` each time) and print how the size stays healthy; (3) confirm the identity trick — when `F(x)` returns all zeros, `F(x) + x` returns exactly `x` (use `np.allclose`). Also assert that the shapes of `F(x)` and `x` match so the `+ x` is valid. Run with `python3 sessions/m05a-text-transformer/day-01-residual-connections/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5a Day 1 artifact.

Create sessions/m05a-text-transformer/day-01-residual-connections/experiment.py that, with a comment on each step:
1. Sets up a fixed small block F(x) = 0.8 * (W @ x) for a fixed 4x4 matrix W and a starting vector x0 = np.ones(4). Asserts F(x0) and x0 have the same shape (so the "+ x" add is valid, element-wise).
2. Plain deep stack: loops 30 times doing x = F(x), and prints the vector norm (np.linalg.norm) every few steps to show the signal shrinking toward 0 (the degradation/fade problem).
3. Residual stack: resets x = x0, loops 30 times doing x = F(x) + x, and prints the norm every few steps to show the signal staying healthy (the gradient-highway effect).
4. Identity check: shows that when F returns all zeros, F(x) + x == x exactly, using np.allclose — the "do nothing = F(x)=0" trick.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the fade, then the rescue)
- The plain stack's signal norm shrinks toward `0` as it passes through 30 blocks — the deep-stack fade you met in the puzzle.
- The residual stack's signal norm stays healthy the whole way — the shortcut's "+ x" express lane keeping a full-strength copy alive.
- The moment to **watch**: when `F(x)` returns all zeros, `F(x) + x` prints *exactly* `x` — proof that the shortcut makes "do nothing" trivial, which is what fixes the degradation problem.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05a-text-transformer/day-01-residual-connections/log.md`: (1) what `output = F(x) + x` means and why it makes the identity easy; (2) what the "gradient highway" is, in one line; (3) where residuals sit in a transformer, and the one problem (a ballooning stream) that layer norm will fix tomorrow.
!!!

@@@ fin
