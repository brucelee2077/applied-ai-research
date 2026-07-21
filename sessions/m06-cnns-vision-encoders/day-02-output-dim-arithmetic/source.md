---
quest_id: wf8-d02-dims
mode: concept
donor: v9-base.donor
page_title: "Module 6 · Day 2 — Output-Size Arithmetic"
module_label: "Module 06 · Represent · Day 2"
title: "Output-Size Arithmetic"
subtitle: "Predicting the Shape a Convolution Produces"
brand_sub: "Foundations · M6 Day 2"
spine: "tape-measure"
nav_prev_href: "../day-01-convolution-pooling/lesson.html"
nav_prev_label: "Convolution & Pooling"
nav_next_href: "../day-03-cnn-classifier/lesson.html"
nav_next_label: "A CNN Classifier"
fin_title: "Module 6 · Day 2 complete! 🏆"
fin_body: "Great work — you can now <b>predict the exact shape</b> a convolution or pooling layer produces before you ever run it. The little <b>tape-measure</b> formula (N−K+2P)/S+1 turns kernel size, stride, and padding into the output grid — the skill that lets you design a whole stack without shape-mismatch errors.<br>Next up: <b>A CNN Classifier</b>."
notebook_yardstick: null
---

@@@ hero
@lede Have you ever measured a shelf with a tape-measure before buying a box to put on it — checking it fits *before* you carry it home, instead of lugging it back when it's too big? That little habit of measuring first saves the whole trip. Today you learn the exact same habit for machine vision. Yesterday you built a stencil that slides across a picture and spits out a smaller grid of scores. But *how much* smaller? If you stack ten of those layers, does your picture shrink to nothing? Today you get a tiny **tape-measure** — one short formula — that tells you the exact size of what comes out of a convolution *before you run a single line of code*. This is the mental math every engineer at every vision lab does in their head: it is how they design a whole network — the kind behind self-driving cars and photo search — without ever hitting a "shapes don't match" error. Predict the size, then run it and watch you were right.
@goal Together we'll turn four simple ingredients — the input size, the kernel size, the stride, and the padding — into the one **tape-measure** formula `(N − K + 2P)/S + 1` that predicts a layer's output height and width. You'll meet each ingredient with its own picture, learn the two padding styles (valid and same), see why we quietly round down, learn why height/width and channel-depth are measured by *separate* rules, and watch each output pixel come to "see" more of the picture as layers stack. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="Why measure first?" title="The habit: measure the output before you build" gotit="Got the why"
Before any formula, let's be clear about *why* this little skill is worth an entire day.

Yesterday you built a convolution: a tiny stencil of weights slides across a picture and leaves behind a smaller grid of scores called a [[feature map||The grid of scores a convolution or pooling layer produces. It is smaller than the input unless you protect the edges.]]. It came out smaller than the picture you started with. That is fine for one layer. But real vision networks stack *many* layers — conv, shrink, conv, shrink — one after another. So a fair worry pops up: **if every layer makes the picture smaller, does it eventually shrink to nothing?**

Picture packing a **nesting doll** — the wooden dolls that each hold a slightly smaller doll inside. Open one, a smaller one is inside; open that, an even smaller one. If you keep going you *run out* — the last doll is a solid nub with nothing inside. A deep network is the same: each layer can hand a smaller grid to the next, and if you are careless, a few layers down you have a `1 × 1` nub with no picture left to work on.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A nesting-doll analogy for stacked convolution layers. Four grids sit left to right, each smaller than the one before: 32 by 32, then 28 by 28, then 12 by 12, then a tiny 1 by 1 nub labelled nothing left. An arrow labelled each layer shrinks it runs across the top."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Careless stacking: each layer shrinks the grid until nothing is left</text><text x="260" y="34" fill="#9A5A12" font-size="9">layer → layer → layer → …</text><g transform="translate(30,50)"><rect x="0" y="0" width="96" height="96" fill="#FCF3DC" stroke="#C99A12"/><text x="48" y="120" fill="#6B645E">32×32</text></g><text x="140" y="100" fill="#9A5A12">→</text><g transform="translate(160,58)"><rect x="0" y="0" width="80" height="80" fill="#FCF3DC" stroke="#C99A12"/><text x="40" y="112" fill="#6B645E">28×28</text></g><text x="252" y="100" fill="#9A5A12">→</text><g transform="translate(272,82)"><rect x="0" y="0" width="34" height="34" fill="#FDF2E0" stroke="#C99A12"/><text x="17" y="88" fill="#6B645E">12×12</text></g><text x="322" y="100" fill="#9A5A12">→</text><g transform="translate(346,96)"><rect x="0" y="0" width="8" height="8" fill="#C93B3B" stroke="#8a3b3b"/><text x="4" y="72" fill="#C93B3B">1×1</text></g><text x="440" y="100" fill="#C93B3B" font-size="10">nothing left!</text></g></svg>
%%%

**What the nesting-doll picture gets right:** each step really can produce something smaller, and stacking enough steps really can run you out. **Where it breaks down:** a nesting doll *must* shrink — its sizes are fixed by the carver — but with a convolution *you* hold the dials. You can make a layer shrink fast, shrink slowly, or not shrink at all. The whole point of today is learning to read and set those dials on purpose.

So the skill is simple: before you build a layer, *measure* what it will output. Same as checking a shelf with a **tape-measure** before you buy the box — measure first, and the box always fits. Do that layer by layer and you can design a ten-layer network on paper, knowing every shape, before you run one line of code.

Four things decide the output size. We'll meet each with its own picture over the next concepts, then snap them together into one short formula:
- **N** — how big the picture is coming in.
- **K** — how big the sliding stencil is.
- **S** — how far the stencil jumps each step.
- **P** — how much border we glue on to protect the edges.

**You've earned the "why."** Hold on to the tape-measure image: by the end of today you'll measure any layer's output in your head.

@@@ concept id=c2 tag="Counting the stops" title="The base case: how many spots does the stencil visit?" gotit="Got N−K+1"
Let's start with the simplest possible sweep — one step at a time, no border added — and just *count how many spots the stencil lands on*. That count is the output size. Everything else today is a small twist on this one idea.

Think of walking a **row of stepping-stones** across a stream. You place your front foot on the first stone, then step, step, step — but you have to stop before your front foot would fall off the far bank. So the number of *places you can stand* is not the number of stones; it's a bit fewer, because your foot has a length and needs stones under all of it. The stencil is the same: it has a width, and it must sit *fully* on the picture — it can never hang off the edge.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A stepping-stones analogy. A row of 5 stones is drawn. A foot that is 3 stones long is placed on stones 1 to 3, labelled stop 1. Then it slides to stones 2 to 4, labelled stop 2, then stones 3 to 5, labelled stop 3. A caption says a 3-long foot on 5 stones fits in only 3 places: 5 minus 3 plus 1 equals 3."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A 3-wide stencil on a 5-wide row fits in only 5 − 3 + 1 = 3 places</text><g transform="translate(60,40)"><g fill="#FCF3DC" stroke="#C99A12"><rect x="0" y="0" width="76" height="34"/><rect x="80" y="0" width="76" height="34"/><rect x="160" y="0" width="76" height="34"/><rect x="240" y="0" width="76" height="34"/><rect x="320" y="0" width="76" height="34"/></g><text x="38" y="22" fill="#8A6D3B">1</text><text x="118" y="22" fill="#8A6D3B">2</text><text x="198" y="22" fill="#8A6D3B">3</text><text x="278" y="22" fill="#8A6D3B">4</text><text x="358" y="22" fill="#8A6D3B">5</text><rect x="-2" y="52" width="240" height="20" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="118" y="88" fill="#8a3b3b">stop 1 (stones 1–3)</text><rect x="78" y="100" width="240" height="20" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="198" y="136" fill="#276b45">stop 2 (stones 2–4)</text><rect x="158" y="148" width="240" height="20" fill="none" stroke="#5E5191" stroke-width="2"/><text x="278" y="184" fill="#5E5191">stop 3 (stones 3–5)</text></g></g></svg>
%%%

**What the stepping-stones picture gets right:** a wide foot fits in fewer places than there are stones, and the last valid spot is where the far edge of the foot lines up with the last stone. **Where it breaks down:** a foot can land anywhere it likes; our stencil only lands on neat whole-pixel spots, so we can just *count* them — no guessing.

#### The counting rule, in plain words
Line up the stencil at the very left. Its left edge sits at pixel 1, its right edge at pixel `K` (the [[kernel size||The side length of the sliding stencil, e.g. 3 means a 3×3 window. A bigger kernel covers more at once, so it fits in fewer places and shrinks the output more.]] — how wide the stencil is). Now slide it right, one pixel at a time, until its *right* edge touches the last pixel of the picture. Count the stops.

- The stencil's right edge starts at pixel `K` and ends at pixel `N` (the [[input size||The height or width of the incoming grid, measured one dimension at a time. We call it N. Do the whole formula once for height, once for width.]] — how wide the picture is).
- So the right edge visits pixels `K, K+1, K+2, … , N`.
- How many numbers is that? From `K` to `N` inclusive is `N − K + 1` of them.

That's the whole base case: with no border and single steps, the output side is **`N − K + 1`**. Let's watch the count for a real picture, and notice the pattern — a bigger stencil (bigger `K`) leaves *fewer* places to stand, so the output gets smaller:

%%% demo id=base label="predict, then reveal"
code: image side N = 7
   try three kernel sizes K = 3, 5, 7   (no border, one-pixel steps)
   output side = N − K + 1
out: K = 3  →  7 − 3 + 1 = 5      (a 3-wide stencil fits in 5 places)
   K = 5  →  7 − 5 + 1 = 3      (a wider stencil → fewer places → smaller output)
   K = 7  →  7 − 7 + 1 = 1      (a stencil as wide as the picture fits in just 1 place)
take: <b>Bigger kernel → smaller output.</b> A stencil the same width as the picture (K = N) fits in exactly one spot, giving a 1×1 output. This "N − K + 1" is the heart of the formula; the next two concepts just add the stride and padding dials on top.
%%%

**You now have the core count.** Notice we did this for *one* direction — the width. Height works exactly the same way with its own `N`. From here, two dials bend this rule: how far you step (stride), and whether you glue on a border (padding).

@@@ concept id=c3 tag="Dial 1 · big steps" title="Stride: skip pixels to shrink faster" gotit="Got stride"
The first dial changes *how far* the stencil jumps between stops. Take bigger jumps and you land on fewer spots — so the output comes out smaller, faster.

Think of climbing a **staircase**. Take the steps one at a time and you touch every stair. Take them two at a time and you touch only *half* the stairs — you get to the top in half the number of steps. You reach the same place; you just visit fewer stairs on the way. That "how many stairs I skip each time" is the dial, and it has a name: the [[stride||How many pixels the stencil jumps between stops. Stride 1 lands on every spot; stride 2 skips every other one, so the output comes out about half the size.]].

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A staircase analogy for stride. On the left a person takes every stair, labelled stride 1, touching all stops. On the right a person takes two stairs at a time, labelled stride 2, touching half the stairs. A caption says stride 2 lands on half as many spots so the output is about half the size."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Stride = stairs skipped per step. Bigger stride → fewer stops → smaller output</text><text x="130" y="38" fill="#276b45">stride 1 · touch every stair</text><g transform="translate(50,50)" stroke="#2D8B55" fill="#EAF5EE"><rect x="0" y="90" width="30" height="16"/><rect x="30" y="74" width="30" height="16"/><rect x="60" y="58" width="30" height="16"/><rect x="90" y="42" width="30" height="16"/><rect x="120" y="26" width="30" height="16"/></g><g transform="translate(50,50)" fill="#276b45"><circle cx="15" cy="90" r="4"/><circle cx="45" cy="74" r="4"/><circle cx="75" cy="58" r="4"/><circle cx="105" cy="42" r="4"/><circle cx="135" cy="26" r="4"/></g><text x="125" y="172" fill="#6B645E">5 stops</text><text x="395" y="38" fill="#5E5191">stride 2 · skip a stair</text><g transform="translate(320,50)" stroke="#5E5191" fill="#EAF0FA"><rect x="0" y="90" width="30" height="16"/><rect x="30" y="74" width="30" height="16"/><rect x="60" y="58" width="30" height="16"/><rect x="90" y="42" width="30" height="16"/><rect x="120" y="26" width="30" height="16"/></g><g transform="translate(320,50)" fill="#5E5191"><circle cx="15" cy="90" r="4"/><circle cx="75" cy="58" r="4"/><circle cx="135" cy="26" r="4"/></g><text x="395" y="172" fill="#6B645E">3 stops (half-ish)</text></g></svg>
%%%

**What the staircase picture gets right:** bigger jumps mean fewer stops, and you cover the same span with less effort. **Where it breaks down:** on a staircase you always land on a real stair; a big stride can make the stencil's last jump *overshoot* the edge and land in mid-air — which is exactly why the next concept (rounding down) exists.

#### How stride bends the count
Without stride, the stencil visited every one of the `N − K + 1` spots from the last concept. Stride `S` means it only lands on every `S`-th one. So take that span of valid starting spots and *divide it by the stride*:

- Stride 1: land on all of them → `N − K + 1` (unchanged, our base case).
- Stride 2: land on every other one → about half as many.
- Stride 3: land on every third one → about a third as many.

In plain words: **the stride divides the span.** Watch the same `7 × 7` picture and `3 × 3` stencil shrink faster as we turn up the stride:

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="Three bars showing the output side of a 7 by 7 image with a 3 by 3 kernel and no padding as stride grows. Stride 1 gives 5, the tallest bar. Stride 2 gives 3, a shorter bar. Stride 3 gives 2, the shortest bar. Bigger stride shrinks the output faster."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">7×7 image, 3×3 kernel, no padding — bigger stride → smaller output</text><line x1="60" y1="160" x2="470" y2="160" stroke="#E5DFD6" stroke-width="1.5"/><rect x="95" y="40" width="70" height="120" fill="#2D8B55"/><text x="130" y="32" fill="#276b45">5</text><text x="130" y="176" fill="#6B645E">stride 1</text><text x="130" y="120" fill="#fff" font-size="8">(7−3)/1+1</text><rect x="225" y="88" width="70" height="72" fill="#C99A12"/><text x="260" y="80" fill="#8A6D3B">3</text><text x="260" y="176" fill="#6B645E">stride 2</text><text x="260" y="132" fill="#fff" font-size="8">(7−3)/2+1</text><rect x="355" y="112" width="70" height="48" fill="#5E5191"/><text x="390" y="104" fill="#5E5191">2</text><text x="390" y="176" fill="#6B645E">stride 3</text><text x="390" y="142" fill="#fff" font-size="8">(7−3)/3+1</text></g></svg>
%%%

Read it left to right: stride 1 gives `(7−3)/1 + 1 = 5`; stride 2 gives `(7−3)/2 + 1 = 3`; stride 3 gives `(7−3)/3 + 1 = 2`. **Stride is your "shrink on purpose" dial** — labs use it to make a feature map smaller quickly, so deeper layers do less work. But you probably noticed `(7−3)/3 = 4/3`, which is not a whole number. What happens to that leftover third of a step? That is the next puzzle.

@@@ concept id=c4 tag="Rounding down" title="The floor: you can't take a fraction of a step" gotit="Got the floor"
Here's the little puzzle from last concept: `(7 − 3)/3 = 4/3 ≈ 1.33`. But you can't take *one and a third* steps — a step is a whole thing. So what do we do with the leftover 0.33 of a step? We **throw it away.** We keep only the whole steps and drop the fraction.

Think of pouring lemonade into **glasses**. You have `4` cups of lemonade and each glass holds `3` cups. How many *full* glasses can you make? One — and you're left with 1 cup that isn't enough to fill a second glass. You count *full* glasses only; the leftover splash doesn't count as a glass. Chopping off that leftover and keeping only the whole part has a name: the [[floor||"Round down to the nearest whole number." Written ⌊ ⌋. Floor of 1.33 is 1. It appears in the output-size formula because you can only take whole steps.]] (written `⌊ ⌋`, and just means "round down").

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A lemonade-glasses analogy for the floor operation. On the left a jug holds 4 cups. Each glass needs 3 cups. Two glasses are drawn: the first is full and green, the second is only one-third full and greyed out. A caption says 4 divided by 3 makes only 1 full glass, floor of 1.33 equals 1, with a leftover splash that does not count."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Floor = keep only WHOLE steps, drop the leftover splash</text><text x="80" y="42" fill="#6B645E">jug: 4 cups</text><path d="M50 55 h60 l-6 80 h-48 z" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="80" y="102" fill="#8A6D3B" font-size="11">4</text><text x="180" y="100" fill="#B8AEA2" font-size="13">→ pour, 3 per glass →</text><g transform="translate(300,55)"><rect x="0" y="0" width="46" height="80" fill="none" stroke="#2D8B55" stroke-width="1.5"/><rect x="2" y="4" width="42" height="72" fill="#7CE0A8"/><text x="23" y="102" fill="#276b45">glass 1: full ✓</text></g><g transform="translate(380,55)"><rect x="0" y="0" width="46" height="80" fill="none" stroke="#B8AEA2" stroke-width="1.5" stroke-dasharray="4,3"/><rect x="2" y="56" width="42" height="20" fill="#E5DFD6"/><text x="23" y="102" fill="#9A938A">glass 2: ⅓ — no ✗</text></g><text x="380" y="170" fill="#C93B3B" font-size="9">⌊1.33⌋ = 1 full glass</text></g></svg>
%%%

**What the lemonade picture gets right:** you count only whole glasses, and any leftover that can't fill a whole one is simply dropped. **Where it breaks down:** the lemonade splash is wasted forever; in a convolution the "dropped" part is a strip of *pixels* at the far edge that the stencil never quite reaches — they're skipped, not destroyed, but the layer still ignores them.

#### What the floor really drops: far-edge pixels
So the honest formula uses the floor: `⌊(N − K)/S⌋ + 1`. When the division isn't clean, rounding down means the stencil's last full jump lands *before* the far edge, and a thin strip of pixels on that edge never gets its own stop. The layer quietly ignores them. Let's see it — same `7 × 7`, `3 × 3` stencil, stride 3 — and watch which pixels get left out:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A 1-D strip of 7 pixels numbered 1 to 7. A 3-wide stencil at stride 3 lands on pixels 1 to 3 as stop 1 and pixels 4 to 6 as stop 2. The next jump would need pixels 7 to 9, but pixels 8 and 9 do not exist, so pixel 7 is left over and shaded grey. A caption says floor drops the far-edge pixel."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Stride 3 on 7 pixels: 2 full stops fit; pixel 7 is dropped by the floor</text><g transform="translate(60,42)"><g fill="#FCF3DC" stroke="#C99A12"><rect x="0" y="0" width="54" height="34"/><rect x="56" y="0" width="54" height="34"/><rect x="112" y="0" width="54" height="34"/><rect x="168" y="0" width="54" height="34"/><rect x="224" y="0" width="54" height="34"/><rect x="280" y="0" width="54" height="34"/></g><rect x="336" y="0" width="54" height="34" fill="#E5DFD6" stroke="#B8AEA2" stroke-dasharray="4,3"/><text x="27" y="22" fill="#8A6D3B">1</text><text x="83" y="22" fill="#8A6D3B">2</text><text x="139" y="22" fill="#8A6D3B">3</text><text x="195" y="22" fill="#8A6D3B">4</text><text x="251" y="22" fill="#8A6D3B">5</text><text x="307" y="22" fill="#8A6D3B">6</text><text x="363" y="22" fill="#9A938A">7</text><rect x="-2" y="46" width="168" height="18" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="82" y="80" fill="#276b45">stop 1 (1–3)</text><rect x="166" y="46" width="168" height="18" fill="none" stroke="#5E5191" stroke-width="2"/><text x="250" y="80" fill="#5E5191">stop 2 (4–6)</text><text x="363" y="112" fill="#C93B3B" font-size="9">pixel 7</text><text x="363" y="124" fill="#C93B3B" font-size="9">dropped</text></g></g></svg>
%%%

**What you just saw:** stride 3 fits two full stops (pixels 1–3 and 4–6); a third jump would need pixels 7, 8, 9 — but 8 and 9 don't exist, so pixel 7 is left with no stop and the floor rounds `1.33` down to `1`, giving `1 + 1 = 2`. **This is the one honest wrinkle in the formula.** It rarely bites — you usually pick sizes that divide cleanly — but now you know *why* the formula rounds down, and that a far-edge strip can be silently ignored when it doesn't.

@@@ concept id=c5 tag="Dial 2 · a border" title="Padding: glue on a border so the output grows" gotit="Got padding"
Every dial so far has *shrunk* the output. The second dial can *grow* it back — and it fixes a quiet unfairness at the edges of the picture.

Think of framing a **photo with a paper mat** — that cardboard border you slip around a picture before it goes in the frame. Two things happen. First, the framed thing gets *bigger* — the mat adds width all around. Second, the corners and edges of the photo, which used to sit right at the frame's lip barely visible, now have room to breathe and get looked at properly. Gluing that border of extra pixels around the picture is called [[padding||Adding a border of extra pixels (usually zeros) around the picture before sliding the stencil. It grows the output back and lets the stencil sit properly over the edge pixels.]], and we write the amount as `P` — the number of pixels added on *each* side.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A photo-mat analogy for padding. On the left a small photo sits alone, its corner pixel barely reachable. On the right the same photo has a dashed border, the mat, added all around it, making it bigger and letting the stencil centre on the corner. A caption says padding P adds P pixels each side, growing the output and protecting the edges."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Padding = a paper mat around the photo: bigger output + edges protected</text><text x="110" y="42" fill="#6B645E">no padding</text><g transform="translate(70,52)"><rect x="0" y="0" width="80" height="80" fill="#FCF3DC" stroke="#C99A12"/><rect x="0" y="0" width="26" height="26" fill="none" stroke="#C93B3B" stroke-width="2"/></g><text x="110" y="152" fill="#C93B3B" font-size="9">corner barely reached</text><text x="255" y="95" fill="#B8AEA2" font-size="13">→ add mat →</text><text x="405" y="42" fill="#6B645E">padding P = 1</text><g transform="translate(345,44)"><rect x="0" y="0" width="120" height="100" fill="none" stroke="#5E5191" stroke-width="1.5" stroke-dasharray="4,3"/><rect x="20" y="10" width="80" height="80" fill="#FCF3DC" stroke="#C99A12"/><rect x="7" y="-3" width="26" height="26" fill="none" stroke="#2D8B55" stroke-width="2"/></g><text x="405" y="162" fill="#276b45" font-size="9">stencil now centres on the corner</text></g></svg>
%%%

**What the photo-mat picture gets right:** the border makes the whole thing bigger and gives the edges room so nothing near the rim gets ignored. **Where it breaks down:** a real mat is decorative cardboard; our border is usually just a ring of **zeros** — fake pixels that add width and reach, but carry no real picture. That's fine: the stencil still gets to sit over the true edge pixels, which is the point.

#### How padding bends the count
Padding does the opposite of the kernel: the kernel *subtracted* width (the stencil can't hang off), and padding *adds* it back. Since you glue `P` pixels on the left *and* `P` on the right, the picture grows by `2P` in each direction. So wherever the base case had `N`, it now has `N + 2P`:

- No padding (`P = 0`): output `= ⌊(N − K)/S⌋ + 1` — shrinks, as before.
- Padding `P = 1`: it's as if the picture were `2` pixels wider, so the output grows by about `2` (at stride 1).

Two fixes fall out of this one dial. **Fix 1 — stop the shrinking:** add enough border and the output no longer nibbles smaller. **Fix 2 — protect the edges:** without padding, the stencil can never *center* on a corner pixel, so corner and edge pixels get looked at far less than middle ones — the model half-forgets the rim of every picture. That under-seen-edges problem is a real failure mode, and padding is its direct remedy. Watch the same `5 × 5` picture with a `3 × 3` stencil, first without a border, then with one:

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="Two bars comparing output size for a 5 by 5 image with a 3 by 3 kernel at stride 1. With no padding the output is 3, a shorter bar in red. With padding of 1 the output is 5, a taller bar in green, equal to the input. A caption says padding grows the output back to the input size."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">5×5 image, 3×3 kernel, stride 1 — padding grows the output back</text><line x1="70" y1="160" x2="450" y2="160" stroke="#E5DFD6" stroke-width="1.5"/><rect x="120" y="88" width="80" height="72" fill="#C93B3B"/><text x="160" y="80" fill="#C93B3B">3</text><text x="160" y="176" fill="#6B645E">P = 0 (no border)</text><text x="160" y="130" fill="#fff" font-size="8">(5−3+0)/1+1</text><rect x="300" y="40" width="80" height="120" fill="#2D8B55"/><text x="340" y="32" fill="#276b45">5</text><text x="340" y="176" fill="#6B645E">P = 1 (one-pixel border)</text><text x="340" y="120" fill="#fff" font-size="8">(5−3+2)/1+1</text></g></svg>
%%%

Read it left to right: with no border the `5 × 5` picture shrinks to `3 × 3`; add a one-pixel border and it stays a full `5 × 5`. **Padding is your "grow it back / protect the edges" dial.** Two friendly settings of this dial are so common they have names — that's the very next concept.

@@@ concept id=c6 tag="Two named settings" title="Valid vs same padding (and why kernels are odd)" gotit="Got valid & same"
You rarely pick the number of padding pixels by hand. Instead you pick one of two *named* settings, and the framework computes `P` for you. They match two everyday choices.

Think of the print dialog when you photocopy a page: you usually pick **"shrink to fit"** or **"keep original size."** *Shrink to fit* lets the copy come out a little smaller — that's [[valid padding||No border added (P = 0). The stencil stays fully inside the picture, so the output shrinks every layer. Also called "no padding."]] (`P = 0`, no border, output shrinks). *Keep original size* makes the copy come out exactly as big as the original — that's [[same padding||Choosing just enough border so the output has the SAME height and width as the input (when stride is 1). For an odd kernel K, P = (K − 1)/2.]] (add just enough border so the output equals the input). "Same" is named after what it protects: the *same* size in as out.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A photocopier print-setting analogy. On the left an original page. Two copies: one labelled valid, shrink to fit, comes out smaller in red. The other labelled same, keep original size, comes out the same size in green. A caption maps valid to P equals 0 output shrinks, and same to enough border so output equals input."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Two named settings, like a photocopier: shrink-to-fit vs keep-size</text><text x="70" y="42" fill="#6B645E">original</text><rect x="40" y="50" width="60" height="80" fill="#FCF3DC" stroke="#C99A12"/><text x="150" y="95" fill="#B8AEA2" font-size="12">→</text><g transform="translate(190,60)"><rect x="0" y="0" width="44" height="58" fill="#FDECEC" stroke="#C93B3B"/><text x="22" y="80" fill="#C93B3B">valid</text><text x="22" y="94" fill="#8a3b3b" font-size="9">P=0 · shrinks</text></g><g transform="translate(340,50)"><rect x="0" y="0" width="60" height="80" fill="#EAF5EE" stroke="#2D8B55"/><text x="30" y="102" fill="#276b45">same</text><text x="30" y="116" fill="#276b45" font-size="9">border added · same size</text></g></g></svg>
%%%

**What the photocopier picture gets right:** you pick the *goal* (smaller, or same size) and the machine works out the settings for you — you never count border pixels by hand. **Where it breaks down:** a copier can shrink to *any* fraction; convolution's "same" only truly holds the size when the stride is 1 (with big strides even "same" padding still shrinks, because striding is a separate dial).

#### How "same" picks its border — and why odd kernels win
For "same" padding at stride 1, you want the `−K` (the kernel's nibble) and the `+2P` (your border) to cancel out. Set them equal: `2P = K − 1`, so `P = (K − 1)/2`. Here's the neat part: for that to be a *whole* number of border pixels, `K − 1` must be even — which means `K` must be **odd**. That is the quiet reason kernels are almost always `3`, `5`, or `7`: an odd kernel has a true *center pixel*, so you can add the same border on both sides and keep the picture centered. A `4 × 4` kernel has no center and needs a lopsided border. **Predict first:** for kernels `3`, `5`, `7`, how much "same" padding does each need? Then reveal:

%%% demo id=same label="predict, then reveal"
code: same padding needs   P = (K − 1) / 2   (so output = input at stride 1)
   try the common odd kernels K = 3, 5, 7
out: K = 3  →  P = (3−1)/2 = 1     (a 1-pixel border — the everyday default)
   K = 5  →  P = (5−1)/2 = 2     (a 2-pixel border)
   K = 7  →  P = (7−1)/2 = 3     (a 3-pixel border)
   K = 4  →  P = (4−1)/2 = 1.5   (not whole! no clean center → even kernels are avoided)
take: <b>Odd kernels have a center pixel, so "same" padding splits evenly on both sides.</b> That is why you see 3×3, 5×5, 7×7 everywhere and almost never 4×4 — the odd size keeps the picture centered and the border a whole number.
%%%

**You now know both named settings.** *Valid* = no border, output shrinks (good when you *want* to downsize). *Same* = enough border to hold the size (good for deep stacks that must not shrink to a nub). And you know the little secret behind every `3 × 3` you'll ever see.

@@@ concept id=c7 tag="Snap it together" title="The whole tape-measure, one worked example" gotit="Got the formula"
Time to snap the four ingredients into one tape-measure. You already met every piece one at a time; now watch them click together — no new ideas, just the same story in one line.

Think of building a **sandwich** from a recipe. You don't invent the ingredients at the counter — bread, filling, a slice off the end, cut into pieces — you just follow the order. The formula is that recipe card: start with the picture, subtract the kernel's nibble, add your border, divide by your step, round down, add one. Same four ingredients you already know, in a fixed order.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="The output-size formula shown as an assembly line with five labelled stages left to right. Stage 1: input N. Stage 2: minus K, the kernel nibble. Stage 3: plus 2P, your border. Stage 4: divide by S then round down, the steps. Stage 5: plus 1, the first stop. The result is the output side O."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">O = ⌊(N − K + 2P) / S⌋ + 1 — one line, four ingredients you already know</text><line x1="20" y1="105" x2="500" y2="105" stroke="#E5DFD6" stroke-width="6"/><g transform="translate(24,66)"><rect x="0" y="0" width="66" height="42" fill="#FCF3DC" stroke="#C99A12"/><text x="33" y="26" fill="#8A6D3B" font-size="13">N</text><text x="33" y="128" fill="#6B645E">picture in</text></g><text x="98" y="92" fill="#9A5A12">→</text><g transform="translate(112,66)"><rect x="0" y="0" width="66" height="42" fill="#FDECEC" stroke="#C93B3B"/><text x="33" y="26" fill="#8a3b3b" font-size="12">− K</text><text x="33" y="128" fill="#6B645E">kernel nibble</text></g><text x="186" y="92" fill="#9A5A12">→</text><g transform="translate(200,66)"><rect x="0" y="0" width="66" height="42" fill="#EAF5EE" stroke="#2D8B55"/><text x="33" y="26" fill="#276b45" font-size="12">+ 2P</text><text x="33" y="128" fill="#6B645E">your border</text></g><text x="274" y="92" fill="#9A5A12">→</text><g transform="translate(288,66)"><rect x="0" y="0" width="76" height="42" fill="#EAF0FA" stroke="#5E5191"/><text x="38" y="26" fill="#5E5191" font-size="11">÷ S ⌊ ⌋</text><text x="38" y="128" fill="#6B645E">steps, round down</text></g><text x="372" y="92" fill="#9A5A12">→</text><g transform="translate(386,66)"><rect x="0" y="0" width="60" height="42" fill="#FDF2E0" stroke="#C99A12"/><text x="30" y="26" fill="#8A6D3B" font-size="12">+ 1</text><text x="30" y="128" fill="#6B645E">first stop</text></g><text x="456" y="92" fill="#9A5A12">→</text><g transform="translate(470,70)"><rect x="0" y="0" width="34" height="34" fill="#0d2b1a" stroke="#2D8B55"/><text x="17" y="128" fill="#276b45">O</text></g></g></svg>
%%%

**What the recipe picture gets right:** the ingredients are the ones you already prepped, and the order is fixed — you never have to remember anything new. **Where it breaks down:** a recipe usually forgives a wrong order; here the order truly matters — you must subtract the kernel and add the border *before* dividing by the stride, never after.

#### The one line, each symbol named
Here it is, the whole tape-measure, with every symbol you already met:

%%% formula
expr: O = ⌊(N − K + 2P) / S⌋ + 1
note: O = output side · N = input side · K = kernel size · P = padding per side · S = stride · ⌊ ⌋ = round down. Run it once for height, once for width.
%%%

#### Walk one example, step by step
Let's tape-measure a real layer: a `32 × 32` picture, a `5 × 5` kernel, `2` pixels of padding each side, stride `1`. Follow the recipe left to right, watching the number shrink and grow at each stage:

%%% mathladder
title: measure the output of a 32×32 picture through a 5×5 kernel, P=2, S=1
words: start with the picture, take off the kernel's nibble, add your border, divide by the step, round down, add one
formula: O = ⌊(N − K + 2P) / S⌋ + 1
numbers: N=32, K=5, P=2, S=1 → ⌊(32 − 5 + 4) / 1⌋ + 1 = ⌊31⌋ + 1 = 32
sanity: P=2 is exactly (5−1)/2, i.e. "same" padding — so the output equals the input (32), just as "same" promises
%%%

Read the ladder: `32 − 5 = 27` (kernel nibble), `+ 4` for the border → `31`, `÷ 1` → `31`, round down (already whole) → `31`, `+ 1` → **`32`**. The output is `32 × 32`, exactly the input — because `P = 2` was the "same" setting for a `5 × 5` kernel. **You just measured a layer without running any code.** Change any one dial and re-run the recipe: that is the whole skill. Let's prove it holds by predicting, then running, three different layers at once:

%%% demo id=full label="predict, then reveal"
code: O = ⌊(N − K + 2P)/S⌋ + 1   — three layers on a 28×28 picture:
   (a) K=3, P=0, S=1      (b) K=3, P=1, S=1      (c) K=3, P=1, S=2
out: (a)  ⌊(28−3+0)/1⌋+1 = 25+1 = 26     (valid: nibbles 28 → 26)
   (b)  ⌊(28−3+2)/1⌋+1 = 27+1 = 28     (same: holds it at 28)
   (c)  ⌊(28−3+2)/2⌋+1 = ⌊13.5⌋+1 = 13+1 = 14   (same+stride2: about half → 14)
take: <b>One formula, every case.</b> Valid shrinks a little, "same" holds the size, and adding stride roughly halves it — and notice (c) needed the floor to round 13.5 down to 13. Three dials, one tape-measure.
%%%

@@@ concept id=c8 tag="Size vs depth" title="Two rulers: spatial size and channel depth are separate" gotit="Got H,W vs C"
Here's a trap that catches almost everyone once. A feature map has *three* numbers: height, width, and **depth** — how many stacked grids there are. The tape-measure formula sizes only the height and width. The depth is set by a completely different rule. Mixing them up is the most common shape mistake beginners make.

Think of a **spreadsheet**. Each sheet has a number of *rows* and *columns* — that's its 2-D size on screen. But the workbook can also have many *tabs* along the bottom — sheet 1, sheet 2, sheet 3. The rows-and-columns are one thing; the number of tabs is a totally different thing. You'd never say "this workbook is 20 rows tall so it has 20 tabs" — the two are set independently. A feature map is the same: height and width are the rows and columns; the number of stacked grids is the tabs.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A spreadsheet analogy for spatial size versus depth. On the left a stack of sheets with tabs, each sheet a grid of rows and columns. A bracket around the rows and columns is labelled height and width, set by the tape-measure formula. A separate bracket across the tabs is labelled channels, the depth, set by the number of filters. A caption says two separate rulers."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A feature map has TWO separate rulers: (H × W) and depth (channels)</text><g transform="translate(140,45)"><rect x="40" y="20" width="90" height="70" fill="#EAF0FA" stroke="#5E5191"/><rect x="20" y="10" width="90" height="70" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="0" width="90" height="70" fill="#FCF3DC" stroke="#C99A12"/><path d="M30 0 V70 M60 0 V70 M0 23 H90 M0 46 H90" stroke="#C99A12" stroke-width="0.5"/></g><path d="M138 128 H230" stroke="#C99A12" stroke-width="1.5"/><text x="184" y="145" fill="#8A6D3B" font-size="9">H × W: the tape-measure formula</text><path d="M258 45 L280 30" stroke="#5E5191" stroke-width="1.5"/><text x="360" y="52" fill="#5E5191" font-size="9">depth = channels</text><text x="360" y="66" fill="#5E5191" font-size="9">= how many filters</text><text x="360" y="128" fill="#276b45" font-size="9">set the two sizes</text><text x="360" y="142" fill="#276b45" font-size="9">with two DIFFERENT rules</text></g></svg>
%%%

**What the spreadsheet picture gets right:** the on-screen size (rows × columns) and the number of tabs are truly independent — changing one never changes the other. **Where it breaks down:** spreadsheet tabs are just labels, but a feature map's depth is meaningful — each grid in the stack is one pattern-detector's report, so more depth means the layer is spotting more kinds of things.

#### The two rules, side by side
- **Height and width** come from the tape-measure: `⌊(N − K + 2P)/S⌋ + 1`. Kernel, stride, and padding decide these.
- **Depth** — the number of stacked grids, called the [[channel count||The number of stacked grids in a feature map, also called depth. Each output channel is one filter's feature map. Set purely by how many filters the layer has — NOT by the tape-measure formula.]] — comes from a different rule entirely: **it equals the number of filters (stencils) in the layer.** Twelve filters → twelve output grids → depth 12. The tape-measure never touches depth; kernel size and stride don't change it at all.

%%% table
:: What you're measuring :: Set by :: Formula / rule
Output height & width :: kernel, stride, padding :: `⌊(N − K + 2P)/S⌋ + 1`
Output depth (channels) :: number of filters in the layer :: depth = number of filters
%%%

#### One free win: pooling uses the very same tape-measure
Remember pooling from yesterday — the step that shrinks a map by summarizing little windows? Good news: it takes *no new formula*. A pooling window is just a stencil with a size and a stride, so its output size is the **same** `⌊(N − K + 2P)/S⌋ + 1`, with `K` = the window size. A classic `2 × 2` pool, stride `2`, no padding, on a `28 × 28` map gives `⌊(28 − 2)/2⌋ + 1 = 13 + 1 = 14` — halved, exactly as the formula says. One tape-measure covers both conv and pool.

!!! c-info 🧭
<b>Optional (skippable) — one small honesty note.</b> This tape-measure only sizes layers that keep the map the same size or make it *smaller* (ordinary conv and pool). Some special layers *grow* a map back — up-sampling for things like image segmentation and image generation — and those use their own, different formula. You'll meet them in a later module; today's formula is the everyday workhorse for the down-sizing layers that make up almost every vision backbone.
!!!

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A before-and-after figure contrasting today's down-sizing layers with the up-sizing layers of a later module. On the left a grid labelled today's formula shrinks a large grid to a smaller one. On the right an up-sampling layer, marked later module, grows a small grid into a larger one. A caption says today's tape-measure only covers the shrinking side."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Today's tape-measure sizes the SHRINK side; growing back is a later story</text><text x="130" y="38" fill="#276b45">today: conv / pool (shrinks)</text><g transform="translate(50,50)"><rect x="0" y="0" width="70" height="70" fill="#FCF3DC" stroke="#C99A12"/><text x="35" y="140" fill="#8A6D3B">big</text></g><text x="135" y="90" fill="#2D8B55" font-size="13">→</text><g transform="translate(155,68)"><rect x="0" y="0" width="34" height="34" fill="#EAF5EE" stroke="#2D8B55"/><text x="17" y="122" fill="#276b45">small</text></g><text x="395" y="38" fill="#5E5191">later module: up-sampling (grows)</text><g transform="translate(320,68)"><rect x="0" y="0" width="34" height="34" fill="#EAF0FA" stroke="#5E5191"/><text x="17" y="122" fill="#5E5191">small</text></g><text x="378" y="90" fill="#5E5191" font-size="13">→</text><g transform="translate(400,50)"><rect x="0" y="0" width="70" height="70" fill="none" stroke="#5E5191" stroke-width="1.5" stroke-dasharray="4,3"/><text x="35" y="140" fill="#5E5191">big again</text></g></g></svg>
%%%

**You've dodged the classic trap.** Height and width live on the tape-measure; depth is just "how many filters." Measure them with two separate rulers and your shapes will always line up.

@@@ concept id=c9 tag="Seeing more" title="Receptive field: each layer sees a bigger patch" gotit="Got receptive field"
Here's a lovely payoff of the shrinking you've been measuring all day. As you stack layers, each output pixel comes to "see" a *bigger* piece of the original picture — even though each stencil is still only `3 × 3`. This growing view is the whole reason deep networks work.

Think of a **rumor spreading through a class**. You whisper to your two neighbors. Each of them whispers to *their* neighbors. After one round, three people know something. After two rounds, the news has reached people you never spoke to directly — friends of friends. Each round of whispering, the "reach" of your original message grows. A stack of convolution layers whispers the same way: layer 1's output pixel hears from a small patch; layer 2's pixel hears from several layer-1 pixels, each of which already heard from its own patch — so layer 2 indirectly hears from a *much* wider patch of the original image.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A receptive-field growth diagram. Three rows stacked. Bottom row: the original image, a wide strip. Middle row: layer 1, where one pixel connects down to a small 3-wide patch of the image. Top row: layer 2, where one pixel connects down to 3 layer-1 pixels, which together reach back to a 5-wide patch of the original image. A caption says the reach grows every layer."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Stack layers → each pixel's "reach" into the original image grows</text><text x="80" y="40" fill="#5E5191">layer 2 (one pixel)</text><g transform="translate(240,32)"><rect x="0" y="0" width="24" height="16" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/></g><text x="80" y="105" fill="#2D8B55">layer 1</text><g transform="translate(180,96)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="24" height="16"/><rect x="36" y="0" width="24" height="16"/><rect x="72" y="0" width="24" height="16"/></g><text x="80" y="172" fill="#8A6D3B">original image</text><g transform="translate(120,162)" fill="#FCF3DC" stroke="#C99A12"><rect x="0" y="0" width="24" height="16"/><rect x="36" y="0" width="24" height="16"/><rect x="72" y="0" width="24" height="16"/><rect x="108" y="0" width="24" height="16"/><rect x="144" y="0" width="24" height="16"/></g><g stroke="#5E5191" stroke-width="0.8" opacity="0.8"><line x1="252" y1="48" x2="192" y2="96"/><line x1="252" y1="48" x2="228" y2="96"/><line x1="252" y1="48" x2="264" y2="96"/></g><g stroke="#2D8B55" stroke-width="0.6" opacity="0.7"><line x1="192" y1="112" x2="132" y2="162"/><line x1="192" y1="112" x2="168" y2="162"/><line x1="192" y1="112" x2="204" y2="162"/><line x1="264" y1="112" x2="240" y2="162"/><line x1="264" y1="112" x2="276" y2="162"/><line x1="264" y1="112" x2="312" y2="162"/></g><text x="410" y="105" fill="#276b45" font-size="9">1 layer: sees 3 pixels</text><text x="410" y="120" fill="#5E5191" font-size="9">2 layers: sees 5 pixels</text><text x="410" y="135" fill="#6B645E" font-size="9">…deeper: sees most of the image</text></g></svg>
%%%

**What the rumor picture gets right:** you only ever whisper to your immediate neighbors, yet the message reaches far-off people after a few rounds — reach grows with depth, not with how loud you whisper. **Where it breaks down:** a rumor can get garbled as it travels; a stack of layers keeps combining real numbers precisely, and (unlike gossip) you can *measure* exactly how far the reach has grown.

#### The reach has a name: the receptive field
The patch of the *original image* that one deep pixel can ultimately see is called its [[receptive field||The patch of the original input that one output pixel depends on. It starts as one kernel wide and grows larger with every stacked layer.]]. It starts at just the kernel size (a `3 × 3` first layer sees a `3 × 3` patch) and grows with every layer you stack. Two `3 × 3` layers give each top pixel a `5 × 5` view of the image; three give `7 × 7`; and adding stride or pooling makes it balloon even faster.

**Why this matters:** a single `3 × 3` stencil can only spot a tiny local thing — a short edge. But an edge plus an edge makes a corner; corners make a shape; shapes make an eye; eyes make a face. Because the receptive field grows with depth, deeper layers can react to bigger, more meaningful things, all while every individual stencil stays cheap and small. **That is the deep reason you stack layers at all** — and now you can *measure* the reach, not just hand-wave it.

@@@ concept id=c10 tag="Recap" title="Today in one page" gotit="Got the recap"
You earned a real superpower today: reading the output size of any conv or pool layer *before* you run it. Here's a fresh way to hold the whole day — picture the formula as a single **tape-measure** with four notches on it, one per ingredient. Slide the notches and the output number changes; you never need a different tool. **Where the tape-measure picture breaks down:** a real tape-measure only *reads* a size, but here you also *set* the size — you choose the notches (kernel, stride, padding) to make the output any shape you want.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A tape-measure summary of the day. A long horizontal tape is drawn with four labelled notches along it: N the input, minus K the kernel, plus 2P the padding, divided by S the stride, then round down and add 1. The reading at the end is O, the output side. A caption says four notches, one number out."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Today = one tape-measure: O = ⌊(N − K + 2P)/S⌋ + 1</text><rect x="30" y="70" width="460" height="30" fill="#FDF2E0" stroke="#C99A12" rx="4"/><g stroke="#8A6D3B" stroke-width="1"><line x1="90" y1="70" x2="90" y2="100"/><line x1="190" y1="70" x2="190" y2="100"/><line x1="290" y1="70" x2="290" y2="100"/><line x1="390" y1="70" x2="390" y2="100"/></g><text x="60" y="90" fill="#8A6D3B">N</text><text x="140" y="90" fill="#C93B3B">− K</text><text x="240" y="90" fill="#2D8B55">+ 2P</text><text x="340" y="90" fill="#5E5191">÷ S ⌊ ⌋</text><text x="445" y="90" fill="#276b45">+1 → O</text><text x="60" y="122" fill="#6B645E">input</text><text x="140" y="122" fill="#6B645E">kernel</text><text x="240" y="122" fill="#6B645E">padding</text><text x="340" y="122" fill="#6B645E">stride</text><text x="445" y="122" fill="#6B645E">output</text><text x="260" y="150" fill="#9A5A12" font-size="9">and depth is a SEPARATE ruler: channels = number of filters</text></g></svg>
%%%

The day in a few beats:
- **Why measure.** Stack layers carelessly and the picture shrinks to a `1 × 1` nub. Measure each layer first — like a tape-measure on a shelf — and design a whole network on paper.
- **The base count.** With no border and single steps, a `K`-wide stencil fits in `N − K + 1` places. Bigger kernel → smaller output.
- **Stride (`S`).** Bigger steps skip pixels, so divide the span by `S`. Your "shrink on purpose" dial.
- **The floor (`⌊ ⌋`).** You can't take a fraction of a step, so round down — which can quietly drop a strip of far-edge pixels.
- **Padding (`P`).** Glue on a border to grow the output back and protect the edge pixels. *Valid* = none (shrinks); *same* = enough to hold the size (`P = (K−1)/2` for odd `K`), which is why kernels are odd.
- **The whole tape-measure.** `O = ⌊(N − K + 2P)/S⌋ + 1`, run once for height, once for width.
- **Size vs depth.** Height and width come from the formula; **depth (channels) = number of filters**, a separate ruler. Pooling uses the *same* formula.
- **Receptive field.** Stack layers and each output pixel sees a bigger patch of the original image — how a network builds edges into shapes into faces.

#### Cheat-sheet · the tape-measure at a glance
%%% table
:: Ingredient :: Symbol :: Effect on output size
Input side :: `N` :: the starting size (per dimension)
Kernel size :: `K` :: bigger `K` → smaller output (nibbles more)
Stride :: `S` :: bigger `S` → smaller output (divides the span)
Padding :: `P` :: bigger `P` → larger output (`+2P`)
Round down :: `⌊ ⌋` :: drops fractional steps → may ignore far-edge pixels
Full formula :: — :: `O = ⌊(N − K + 2P)/S⌋ + 1`
Valid padding :: `P = 0` :: no border; output shrinks each layer
Same padding :: `P = (K−1)/2` :: output = input at stride 1 (needs odd `K`)
Channels (depth) :: `C_out` :: **separate rule:** = number of filters, not the formula
%%%

#### Cheat-sheet · the words you met today
%%% jargon
input size (N) | the height or width of the incoming grid, measured one dimension at a time
kernel size (K) | how wide the sliding stencil is; bigger K shrinks the output more
stride (S) | how many pixels the stencil jumps each step; bigger S shrinks faster
padding (P) | a border of extra pixels added each side; grows the output and protects edges
floor ⌊ ⌋ | round down to a whole number; you can't take a fraction of a step
valid padding | no border (P=0); the output shrinks every layer
same padding | just enough border to keep output = input at stride 1; P=(K−1)/2 for odd K
output size formula | O = ⌊(N − K + 2P)/S⌋ + 1, run once per dimension
channel count (depth) | how many stacked grids; equals the number of filters, a separate rule
receptive field | the patch of the original image one output pixel can ultimately see; grows with depth
%%%

That's the whole day. You can now measure any conv or pool layer's output — height, width, and depth — before you run it. Next you'll put many measured layers together into a working image classifier: **day-03, a CNN classifier**, where every shape you stack will line up because you can predict it.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: A 10×10 image, 3×3 kernel, no padding, stride 1 — what is the output side? | a:2 | 10, size never changes | 7, the kernel adds width | 8, from (10−3)/1 + 1 | 5, stride halves it | fb: Use O = ⌊(N−K+2P)/S⌋+1 = ⌊(10−3+0)/1⌋+1 = 7+1 = 8. With no padding the kernel nibbles the size down a little.
q: Why does "same" padding usually go with an ODD kernel size like 3, 5, or 7? | a:1 | Odd kernels are faster to compute | An odd kernel has a center pixel, so P=(K−1)/2 is a whole number and the border splits evenly on both sides | Even kernels are not allowed by the math | Odd kernels use fewer weights | fb: "Same" needs P=(K−1)/2. For K=4 that is 1.5 — not a whole number. Odd K gives a true center pixel and an even split, so 3×3, 5×5, 7×7 are the defaults.
q: A conv layer has 12 filters and takes a 32×32×3 input with "same" padding, stride 1. What is the OUTPUT shape? | a:2 | 32×32×3, depth never changes | 30×30×12, the kernel shrinks it | 32×32×12, "same" holds H,W and depth = number of filters | 12×12×32, filters set the size | fb: Height and width come from the tape-measure ("same", stride 1 → 32×32). Depth is a SEPARATE ruler: it equals the number of filters, so 12. Result: 32×32×12.
q: You stack more 3×3 conv layers. What happens to each output pixel's receptive field? | a:1 | It stays 3×3 forever | It grows — each pixel comes to see a bigger patch of the original image | It shrinks with depth | Only padding changes it | fb: Each layer's pixel reads several pixels from the layer below, each of which already saw its own patch. So the reach into the original image grows with depth — the reason stacking builds edges into shapes into faces.
%%%

@@@ produce id=produce tag="Produce" title="Measure a layer, then run it and check" gotit="Done"
Time to trust the tape-measure with your own eyes. You'll write a tiny function that computes `O = ⌊(N − K + 2P)/S⌋ + 1`, then feed a real array through a convolution and **check** the shape it actually produces matches your prediction — every time. **Predict first:** for a `28 × 28` image with a `3 × 3` kernel, `P = 1`, `S = 2`, what output side does the formula give? (Watch out for the floor.) Then run it and **observe** the real feature map comes out at exactly that size. Pick one path.

#### Option A · write it yourself
Create `sessions/m06-cnns-vision-encoders/day-02-output-dim-arithmetic/experiment.py`. Write `out_size(N, K, P, S)` that returns `(N - K + 2*P) // S + 1` (the `//` is Python's floor divide — that's the `⌊ ⌋`). **Print** its prediction for several settings: `(28,3,0,1)`, `(28,3,1,1)`, `(28,3,1,2)`, `(32,5,2,1)`. **Predict** each on paper first, then let the function confirm. Then make it real: build a `28 × 28` array (e.g. `img = np.zeros((28,28))`), write a small `conv2d(img, K, P, S)` that slides a `K×K` kernel with padding `P` (use `np.pad`) and stride `S`, and **print the actual feature-map shape at every step**. **Observe** the real shape equals your `out_size` prediction. Finally, add a channel check: make a fake "bank" of `12` filters and **notice** the output depth is `12` no matter what `K`, `P`, `S` are — depth is the separate ruler. Run with `python3 sessions/m06-cnns-vision-encoders/day-02-output-dim-arithmetic/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 6 Day 2 artifact.

Create sessions/m06-cnns-vision-encoders/day-02-output-dim-arithmetic/experiment.py that, with a comment on each step and printing values at every step:
1. Defines out_size(N, K, P, S) returning (N - K + 2*P) // S + 1 (floor divide = the floor in the formula). Add a docstring naming each argument.
2. Prints the predicted output side for these settings and labels each (valid / same / same+stride2 / same 5x5): (28,3,0,1)->26, (28,3,1,1)->28, (28,3,1,2)->14, (32,5,2,1)->32.
3. Builds a 28x28 float array img = np.zeros((28,28)). Defines conv2d(img, k_size, pad, stride) that pads with np.pad, slides a k_size x k_size window with the given stride computing a sum at each spot, and returns the feature map. Prints the ACTUAL feature-map shape for (K=3,P=1,S=2) and asserts it equals (out_size(28,3,1,2), out_size(28,3,1,2)) = (14,14).
4. Shows depth is separate: stack 12 filters (loop conv2d 12 times, np.stack) and print the output shape (expect (12,14,14)); point out the 12 comes ONLY from the filter count, not from K/P/S.
5. Prints a one-line takeaway: "predicted size == actual size, every time; depth = number of filters."
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (predict, then confirm)
- `out_size(28, 3, 1, 2)` returns `14`, because `⌊(28 − 3 + 2)/2⌋ + 1 = ⌊13.5⌋ + 1 = 13 + 1 = 14` — the floor rounds `13.5` down to `13`.
- The **actual** convolution output comes out `14 × 14` — the same number the formula predicted, with no surprises.
- Stacking `12` filters gives an output depth of `12`, and that `12` does **not** change when you change `K`, `P`, or `S` — proof that depth is a separate ruler from height and width.
- Every prediction matches the real shape — which is the whole promise: you can design a stack on paper and know it will fit before you run it.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m06-cnns-vision-encoders/day-02-output-dim-arithmetic/log.md`: (1) the output-size formula in your own words, naming each of the four ingredients; (2) what "same" padding does and why it needs an odd kernel; (3) why the output *depth* is not set by the tape-measure formula.
!!!

@@@ fin
