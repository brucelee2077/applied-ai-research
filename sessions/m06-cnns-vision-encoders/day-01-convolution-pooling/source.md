---
quest_id: wf8-d01-conv
mode: concept
donor: v9-base.donor
page_title: "Module 6 · Day 1 — Convolution & Pooling"
module_label: "Module 06 · Represent · Day 1"
title: "Convolution and Pooling"
subtitle: "A Tiny Filter That Slides Across an Image"
brand_sub: "Foundations · M6 Day 1"
spine: "stencil"
nav_prev_href: "../../m05b-vision-transformer/review.html"
nav_prev_label: "M5b · Review Gate"
nav_next_href: "../day-02-output-dim-arithmetic/lesson.html"
nav_next_label: "Output-Size Arithmetic"
fin_title: "Module 6 · Day 1 complete! 🏆"
fin_body: "Nice work — you've unlocked <b>Convolution &amp; Pooling</b>: a tiny <b>stencil</b> of weights that slides across the whole image, reusing the same numbers everywhere to spot one little pattern, then pooling that shrinks the result while keeping what matters. This weight-sharing trick is how a machine starts to <i>see</i> — the foundation under every vision model you have heard of.<br>Next up: <b>Output-Size Arithmetic</b>."
notebook_yardstick: null
---

@@@ hero
@lede Have you ever laid a stencil on a wall — that stiff sheet with a shape cut out — and slid it along, spray-painting the same little star over and over across the whole wall? You press it down, paint, lift it, move it a bit, paint again. One small shape, stamped everywhere. That is almost exactly how a computer starts to *see* a picture. It takes a tiny grid of numbers — a **stencil** of weights — and slides it across the whole image, checking "is my little pattern *here*? how about *here*?" at every spot. This one trick, called **convolution**, is the engine behind nearly every vision model you have heard of: the face-unlock on a phone, the filter that finds your dog in your photos, the self-driving car spotting a stop sign. Today you get to build that engine, one slide at a time.
@goal Together we'll teach a machine to spot a pattern in a picture. You'll meet the tiny **stencil** of weights (a kernel), slide it across an image to make a "found-it" map, see why reusing the same stencil everywhere is the whole magic trick, learn the two dials that control how it slides (stride and padding), stack several stencils to catch several patterns, then *shrink* the map with pooling while keeping the important bits. We'll also meet — gently, as little puzzles — the things one layer still can't do. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="Why not just wire it up?" title="The old way: connect every pixel to everything (and why it breaks)" gotit="Got the problem"
Before the clever trick, let's see the clumsy first attempt — because understanding *why it fails* is what makes convolution feel like a gift.

Here's the naive idea. We already know how to build a plain neural layer: take a list of numbers, and connect every input to every neuron with its own [[weight||A single number the model learns. It says how strongly one input pushes on one output.]] (a learned number). So why not just unroll the whole picture into one giant list of pixels and wire it straight into a layer?

Picture an old-fashioned **telephone switchboard**, where an operator must run one physical cord from *every* caller to *every* possible receiver. Two people? A few cords. But a whole city? The board explodes into a hopeless spaghetti of wires. That's what happens when you connect every pixel to every neuron — this all-to-all wiring is called a [[fully-connected layer||A layer where every input connects to every neuron with its own separate weight. Great for short lists, hopeless for images.]].

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A telephone switchboard analogy. On the left a small image is flattened into a tall column of pixel dots. Every pixel dot has a line running to every neuron on the right, making a dense tangle of wires labelled one weight per wire, millions of them."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Fully-connected = one wire (weight) from every pixel to every neuron</text><text x="70" y="44" text-anchor="middle" fill="#6B645E">flattened image</text><text x="70" y="58" text-anchor="middle" fill="#9A938A" font-size="9">(pixels in a column)</text><g fill="#C99A12"><circle cx="70" cy="78" r="5"/><circle cx="70" cy="100" r="5"/><circle cx="70" cy="122" r="5"/><circle cx="70" cy="144" r="5"/><circle cx="70" cy="166" r="5"/><circle cx="70" cy="188" r="5"/></g><g fill="#2D8B55"><circle cx="440" cy="100" r="7"/><circle cx="440" cy="140" r="7"/><circle cx="440" cy="180" r="7"/></g><text x="440" y="80" text-anchor="middle" fill="#276b45">neurons</text><g stroke="#C93B3B" stroke-width="0.5" opacity="0.55"><line x1="75" y1="78" x2="433" y2="100"/><line x1="75" y1="78" x2="433" y2="140"/><line x1="75" y1="78" x2="433" y2="180"/><line x1="75" y1="100" x2="433" y2="100"/><line x1="75" y1="100" x2="433" y2="140"/><line x1="75" y1="100" x2="433" y2="180"/><line x1="75" y1="122" x2="433" y2="100"/><line x1="75" y1="122" x2="433" y2="140"/><line x1="75" y1="122" x2="433" y2="180"/><line x1="75" y1="144" x2="433" y2="100"/><line x1="75" y1="144" x2="433" y2="140"/><line x1="75" y1="144" x2="433" y2="180"/><line x1="75" y1="166" x2="433" y2="100"/><line x1="75" y1="166" x2="433" y2="140"/><line x1="75" y1="166" x2="433" y2="180"/><line x1="75" y1="188" x2="433" y2="100"/><line x1="75" y1="188" x2="433" y2="140"/><line x1="75" y1="188" x2="433" y2="180"/></g><text x="255" y="205" text-anchor="middle" fill="#C93B3B" font-size="10">every wire is its own weight → the count explodes</text></g></svg>
%%%

**What the switchboard picture gets right:** every wire is a separate learned weight, and the total number of wires is what blows up — just like the operator drowning in cords. **Where it breaks down:** a switchboard operator can *choose* which cords to plug; a fully-connected layer must keep *all* of them, whether they help or not.

#### Puzzle 1 · the number of wires explodes
Let's count. A small `100 × 100` color photo has `100 × 100 × 3 = 30,000` pixels. Wire that into just `1,000` neurons and you need `30,000 × 1,000 = 30 million` weights — for *one* layer, on a *tiny* picture. Real photos are far bigger. The model would be enormous, slow, and starved for data. Watch the wire-count climb as the picture grows:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three growing bars showing how the number of weights in a fully-connected layer explodes with image size. A 28 by 28 image needs about 0.8 million weights, a short bar. A 100 by 100 color image needs 30 million, a taller bar. A 224 by 224 color image needs about 150 million, the tallest bar."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Fully-connected weights explode as the image grows (into 1000 neurons)</text><line x1="60" y1="160" x2="475" y2="160" stroke="#E5DFD6" stroke-width="1.5"/><rect x="95" y="150" width="66" height="10" fill="#2D8B55"/><text x="128" y="176" fill="#6B645E">28×28 gray</text><text x="128" y="142" fill="#276b45">~0.8M</text><rect x="228" y="98" width="66" height="62" fill="#C99A12"/><text x="261" y="176" fill="#6B645E">100×100×3</text><text x="261" y="90" fill="#8A6D3B">30M</text><rect x="361" y="42" width="66" height="118" fill="#C93B3B"/><text x="394" y="176" fill="#6B645E">224×224×3</text><text x="394" y="34" fill="#C93B3B">~150M</text></g></svg>
%%%

#### Puzzle 2 · it throws away the picture's shape
There's a second, sneakier problem. The moment you flatten the picture into one long list, you *scramble* it. Two pixels that sat right next to each other — say, two dots on the edge of a cat's ear — end up as two faraway entries in the list, no closer than a pixel from the opposite corner. The layer has no idea they were neighbors. But in pictures, *neighbors are everything*: edges, corners, and textures are all about nearby pixels agreeing or disagreeing. Flattening throws that away.

So we want something that (1) uses *far* fewer weights, and (2) *respects* which pixels are near each other. That "something" is convolution — and the fix is almost embarrassingly simple. **You've just earned the right to appreciate the trick:** hold on to these two complaints, because the next few concepts fix both at once.

@@@ concept id=c2 tag="The tiny stencil" title="The kernel: a tiny stencil that looks at one small patch" gotit="Got the kernel"
Here's the fix, and it starts with one small object. Instead of one giant layer that stares at the whole picture at once, we use a *tiny* grid of weights — just a few numbers, often `3 × 3` — and let it look at only a small square of the image at a time.

Think of that **stencil** from the very start: a stiff little sheet with a shape, small enough to hold in one hand. You don't press it over the whole wall at once; you place it on *one small spot*. That little grid of weights has a name: a [[kernel||A small grid of learned weights (e.g. 3×3) — the stencil convolution slides across an image. Also called a filter.]] (also called a [[filter||Another name for a kernel: the small grid of weights convolution slides across the image.]]). It is the one thing a convolution layer *learns*.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="On the right a large image grid of pixels. A small 3 by 3 stencil, the kernel, is laid over one small patch in the top-left corner. The patch it covers is highlighted. A label says the kernel only sees this small local patch, called its receptive field, not the whole image."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A kernel is a tiny stencil that covers only ONE small patch at a time</text><text x="95" y="44" text-anchor="middle" fill="#6B645E">the kernel (3×3 stencil)</text><g transform="translate(55,54)"><rect x="0" y="0" width="90" height="90" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><path d="M30 0 V90 M60 0 V90 M0 30 H90 M0 60 H90" stroke="#C93B3B" stroke-width="0.8"/><text x="15" y="20" text-anchor="middle" fill="#8a3b3b">w</text><text x="45" y="20" text-anchor="middle" fill="#8a3b3b">w</text><text x="75" y="20" text-anchor="middle" fill="#8a3b3b">w</text><text x="15" y="50" text-anchor="middle" fill="#8a3b3b">w</text><text x="45" y="50" text-anchor="middle" fill="#8a3b3b">w</text><text x="75" y="50" text-anchor="middle" fill="#8a3b3b">w</text><text x="15" y="80" text-anchor="middle" fill="#8a3b3b">w</text><text x="45" y="80" text-anchor="middle" fill="#8a3b3b">w</text><text x="75" y="80" text-anchor="middle" fill="#8a3b3b">w</text></g><text x="95" y="165" text-anchor="middle" fill="#9A938A" font-size="9">9 learned weights</text><text x="200" y="100" fill="#B8AEA2" font-size="16">→ laid on →</text><g transform="translate(310,45)"><rect x="0" y="0" width="180" height="120" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.2"/><path d="M30 0 V120 M60 0 V120 M90 0 V120 M120 0 V120 M150 0 V120 M0 30 H180 M0 60 H180 M0 90 H180" stroke="#C99A12" stroke-width="0.5"/><rect x="0" y="0" width="90" height="90" fill="none" stroke="#C93B3B" stroke-width="2.5"/></g><text x="400" y="182" text-anchor="middle" fill="#276b45" font-size="10">the small patch it sees = its receptive field</text></g></svg>
%%%

**What the stencil picture gets right:** the kernel is small, stiff (its weights don't change as it moves), and it only ever covers a *small* patch of the picture at one time. **Where it breaks down:** a real stencil just blocks paint; this stencil *does arithmetic* — it multiplies and adds the pixels under it (we'll see how in the next concept).

#### The small patch has a name: the receptive field
The little square of the image the kernel can see at one placement is called its [[receptive field||The small local patch of the input a single output value looks at. A 3×3 kernel has a 3×3 receptive field.]]. A `3 × 3` kernel has a `3 × 3` receptive field — it looks at 9 pixels, and *only* those 9. This is the direct fix for Puzzle 2 from before: the kernel looks at pixels that are actually *neighbors*, so it can notice things like "these three pixels in a row are all bright" — the start of seeing an edge. It never has to compare the top-left corner with the bottom-right corner, because those aren't neighbors and, for spotting a little pattern, they don't need to be.

And notice how *few* numbers we're carrying now: a `3 × 3` kernel is **9 weights**, no matter how huge the picture is. Compare that to the 30 million from Puzzle 1. **That is the first half of the magic** — a handful of weights instead of millions. The second half is coming: what happens when we *slide* it.

@@@ concept id=c3 tag="Slide and score" title="Convolution: slide the stencil and score every spot" gotit="Got convolution"
Now the fun part — we make the stencil *move*. We place the kernel at the top-left of the image, compute one number, slide it one step to the right, compute another number, and keep going until we've covered the whole picture. This sliding-and-scoring is the whole operation, and it has a name: [[convolution||Sliding a small kernel across an image and computing one number at each spot, producing a grid of scores called a feature map.]].

Remember stamping the stencil star across the wall? Each time you press it down you leave one mark. Convolution is the same rhythm — place, score, slide, score, slide — except each "mark" is a single *number* that answers: **"how strongly does my little pattern show up right here?"**

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A stencil kernel slides across an image in reading order. Three snapshots show it at the top-left, then one step right, then further along, each leaving one score behind. Together the scores fill a smaller output grid called the feature map."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Slide the stencil left-to-right, top-to-bottom — each stop leaves one score</text><g transform="translate(30,40)"><rect x="0" y="0" width="150" height="120" fill="#FCF3DC" stroke="#C99A12"/><path d="M30 0 V120 M60 0 V120 M90 0 V120 M120 0 V120 M0 30 H150 M0 60 H150 M0 90 H150" stroke="#C99A12" stroke-width="0.5"/><rect x="0" y="0" width="90" height="90" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text x="75" y="145" fill="#6B645E">stop 1: top-left</text></g><text x="205" y="100" fill="#9A5A12">→</text><g transform="translate(225,40)"><rect x="0" y="0" width="150" height="120" fill="#FCF3DC" stroke="#C99A12"/><path d="M30 0 V120 M60 0 V120 M90 0 V120 M120 0 V120 M0 30 H150 M0 60 H150 M0 90 H150" stroke="#C99A12" stroke-width="0.5"/><rect x="30" y="0" width="90" height="90" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text x="75" y="145" fill="#6B645E">stop 2: one step right</text></g><text x="400" y="100" fill="#9A5A12">→ …</text><g transform="translate(430,55)"><rect x="0" y="0" width="60" height="60" fill="#EAF5EE" stroke="#2D8B55"/><path d="M20 0 V60 M40 0 V60 M0 20 H60 M0 40 H60" stroke="#2D8B55" stroke-width="0.6"/><text x="30" y="90" fill="#276b45">feature map</text><text x="30" y="103" fill="#276b45" font-size="9">(the scores)</text></g></g></svg>
%%%

**What the stamping picture gets right:** the same stencil visits every spot in a fixed sweep, leaving one result behind at each stop. **Where it breaks down:** a paint stamp leaves the *same* mark everywhere; convolution's marks *differ* — a high score where the pattern matches, a low one where it doesn't.

#### What is the one number at each spot?
At each stop, we line up the kernel's 9 weights with the 9 pixels underneath, multiply each pair, and add them all up. That "multiply-matching-pairs-and-sum" is called a [[dot product||Multiply matching pairs of numbers, then add up all the products, to get one single number. It measures how well two grids of numbers agree.]]. One dot product → one number → one spot on the output. Let's do one by hand with tiny numbers — **predict first:** if the kernel is all `1`s (it just adds up the patch) and the patch under it is `[[1,0,1],[0,1,0],[1,0,1]]`, what's the sum? Then reveal:

%%% demo id=dot label="predict, then reveal"
code: patch=[[1,0,1],[0,1,0],[1,0,1]]  kernel=[[1,1,1],[1,1,1],[1,1,1]]  →  sum(patch * kernel)
out: (1·1)+(0·1)+(1·1) + (0·1)+(1·1)+(0·1) + (1·1)+(0·1)+(1·1) = 5
   → the output value at this one spot = 5
take: <b>One dot product = one number = one spot on the output.</b> Multiply each pixel by the weight sitting on top of it, add all 9 products, and you get the single score for this location. Slide the stencil one step and repeat to fill the next spot.
%%%

#### The grid of scores has a name: the feature map
Sweep the kernel over the whole image and you fill in a whole grid of these scores. That output grid is called a [[feature map||The grid of scores produced by sweeping one kernel over the whole input. Bright spots mark where the kernel's pattern was found.]] (or *activation map*). It's a map that says, spot by spot, "*here* is where my pattern shows up." Bright values mean "found it," dim values mean "not here." Watch a bright stripe in an image light up the matching cells of a feature map:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="On the left an input image with a bright vertical stripe down the middle. An edge-detecting kernel slides across it. On the right the resulting feature map is dark everywhere except two bright columns where the stripe's edges are, showing the map lights up where the pattern is found."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The feature map lights up where the kernel's pattern is found</text><text x="105" y="40" fill="#6B645E">input: a bright stripe</text><g transform="translate(50,50)"><rect x="0" y="0" width="120" height="120" fill="#2C2A28"/><rect x="45" y="0" width="30" height="120" fill="#FCF3DC"/><path d="M15 0 V120 M30 0 V120 M45 0 V120 M60 0 V120 M75 0 V120 M90 0 V120 M105 0 V120 M0 15 H120 M0 30 H120 M0 45 H120 M0 60 H120 M0 75 H120 M0 90 H120 M0 105 H120" stroke="#6B645E" stroke-width="0.4"/></g><text x="255" y="112" fill="#9A5A12" font-size="12">convolve →</text><text x="410" y="40" fill="#276b45">feature map: edges glow</text><g transform="translate(355,50)"><rect x="0" y="0" width="120" height="120" fill="#0d2b1a"/><rect x="30" y="0" width="15" height="120" fill="#7CE0A8"/><rect x="75" y="0" width="15" height="120" fill="#7CE0A8"/><path d="M15 0 V120 M30 0 V120 M45 0 V120 M60 0 V120 M75 0 V120 M90 0 V120 M105 0 V120 M0 15 H120 M0 30 H120 M0 45 H120 M0 60 H120 M0 75 H120 M0 90 H120 M0 105 H120" stroke="#276b45" stroke-width="0.4"/></g><text x="415" y="188" fill="#9A938A" font-size="9">bright = "edge found here"</text></g></svg>
%%%

**You've now built the core operation.** A tiny stencil, a slide, a dot product at each stop, and out comes a map of where a pattern lives. Everything else today is either *why this is so clever* or *knobs that change how it slides*.

@@@ concept id=c4 tag="One stencil, everywhere" title="Weight sharing: the same stencil at every spot" gotit="Got weight sharing"
Now the beautiful part — the idea that makes the whole thing worth it. When the stencil slides, its numbers *never change*. The same 9 weights are used at the top-left, in the middle, at the bottom-right — everywhere. This reusing of one set of weights at every position is called [[weight sharing||Reusing the same kernel weights at every position instead of learning separate weights per spot. It keeps the model tiny and lets it spot a pattern anywhere.]].

Think of a **cookie cutter**. You press the *same* star-shaped cutter into the dough here, then there, then over there. You don't carve a brand-new cutter for each cookie — one cutter makes stars all over the tray. The kernel is that one cookie cutter, pressed all over the image.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A cookie cutter analogy. One star-shaped cutter is shown once on the left. On the right a baking tray has the same star stamped in many positions, all identical, labelled one cutter, the same shape everywhere. This is weight sharing: one kernel reused at every spot."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">One cutter (kernel), pressed all over — the SAME weights everywhere</text><text x="80" y="42" fill="#6B645E">one cutter</text><path d="M80 55 l9 27 h28 l-22 17 l8 27 l-23 -17 l-23 17 l8 -27 l-22 -17 h28 z" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="80" y="150" fill="#9A938A" font-size="9">9 fixed weights</text><text x="175" y="105" fill="#B8AEA2" font-size="14">→ stamp everywhere →</text><g transform="translate(300,45)"><rect x="0" y="0" width="200" height="110" fill="#FCF3DC" stroke="#C99A12"/><g fill="#F6C744" stroke="#C99A12" stroke-width="1"><path d="M35 25 l5 15 h16 l-13 10 l5 16 l-13 -10 l-13 10 l5 -16 l-13 -10 h16 z"/><path d="M100 25 l5 15 h16 l-13 10 l5 16 l-13 -10 l-13 10 l5 -16 l-13 -10 h16 z"/><path d="M165 25 l5 15 h16 l-13 10 l5 16 l-13 -10 l-13 10 l5 -16 l-13 -10 h16 z"/><path d="M67 75 l5 15 h16 l-13 10 l5 16 l-13 -10 l-13 10 l5 -16 l-13 -10 h16 z"/><path d="M132 75 l5 15 h16 l-13 10 l5 16 l-13 -10 l-13 10 l5 -16 l-13 -10 h16 z"/></g></g><text x="400" y="172" fill="#276b45" font-size="9">every star is identical — one shared stencil</text></g></svg>
%%%

**What the cookie-cutter picture gets right:** one tool, reused everywhere, making the identical shape at every spot — that's exactly weight sharing. **Where it breaks down:** a cookie cutter is fixed forever, but this cutter *learns*: over training the 9 weights slowly adjust into a shape worth detecting. (How that learning happens is a later day — today the weights just sit still while we slide.)

#### Why sharing is such a big deal
Two gifts fall out of sharing one stencil:

- **Tiny models.** We keep only 9 weights, not one-per-position. This is the fix for Puzzle 1 — the parameter explosion is gone. A `3 × 3` kernel is 9 numbers whether the image is `28 × 28` or `4000 × 4000`.
- **Spot a pattern *anywhere*.** Because the same edge-detector is applied at every spot, an edge is found whether it's in the top-left or the bottom-right. You learn the pattern *once* and get it detected *everywhere* for free.

#### The salient property: translation equivariance
That second gift has a fancy name worth knowing. If you *shift* the whole picture — slide the cat two steps to the right — the feature map shifts two steps to the right too, but is otherwise the same. The output *moves with* the input. This matching-shift behavior is called [[translation equivariance||If the input shifts, the output shifts the same way. Because one shared kernel is applied everywhere, convolution has this property — which is exactly why it suits images.]]. It's the deep reason convolution suits pictures: a cat is a cat whether it's on the left or the right of the frame, and a shared stencil treats it the same way at both spots. Watch the input shift, and the feature map follow:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Two rows showing translation equivariance. Top row: an input with a dot on the left produces a feature map with a bright spot on the left. Bottom row: the same input with the dot shifted right produces the same feature map but with its bright spot shifted right by the same amount. The output shifts with the input."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Shift the input → the feature map shifts the SAME way (equivariance)</text><text x="55" y="40" fill="#6B645E">input</text><text x="200" y="40" fill="#276b45">feature map</text><g transform="translate(30,48)"><rect x="0" y="0" width="90" height="50" fill="#FCF3DC" stroke="#C99A12"/><circle cx="22" cy="25" r="9" fill="#3A342E"/></g><text x="140" y="76" fill="#9A5A12">→</text><g transform="translate(160,48)"><rect x="0" y="0" width="90" height="50" fill="#0d2b1a"/><circle cx="22" cy="25" r="9" fill="#7CE0A8"/></g><text x="300" y="76" fill="#9A938A" font-size="9">dot on the left → glow on the left</text><g transform="translate(30,120)"><rect x="0" y="0" width="90" height="50" fill="#FCF3DC" stroke="#C99A12"/><circle cx="66" cy="25" r="9" fill="#3A342E"/></g><text x="140" y="148" fill="#9A5A12">→</text><g transform="translate(160,120)"><rect x="0" y="0" width="90" height="50" fill="#0d2b1a"/><circle cx="66" cy="25" r="9" fill="#7CE0A8"/></g><text x="300" y="148" fill="#9A938A" font-size="9">dot shifted right → glow shifts right, same amount</text></g></svg>
%%%

**This is the payoff.** Weight sharing gives a model that is small *and* finds patterns anywhere they appear. That's why convolution, invented decades ago, is still inside the vision systems around you today.

@@@ concept id=c5 tag="What one stencil sees" title="An edge detector: a stencil you can read by hand" gotit="Got edge detection"
So far the kernel's 9 weights have been mystery numbers the model will learn. But here's a lovely surprise: you can *design a kernel by hand* and predict exactly what it finds. This makes the whole abstract idea click — because you get to *see* a stencil do a job.

Think of a **contrast dial** on an old TV. A picture is boring where everything is the same shade; it gets interesting at the *edges*, where light meets dark. An edge detector is a stencil tuned to shout at exactly those places: it stays quiet over flat, same-colored areas and lights up where brightness suddenly changes.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A hand-designed edge-detecting kernel. It is a 3 by 3 grid with a column of minus ones on the left, a column of zeros in the middle, and a column of plus ones on the right. A caption explains it subtracts the left side from the right, so it stays near zero on flat areas and spikes at a vertical edge."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A hand-made edge-detector kernel: "right side minus left side"</text><g transform="translate(190,36)"><rect x="0" y="0" width="120" height="120" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><path d="M40 0 V120 M80 0 V120 M0 40 H120 M0 80 H120" stroke="#C93B3B" stroke-width="0.7"/><text x="20" y="26" fill="#8a3b3b">-1</text><text x="60" y="26" fill="#8a3b3b">0</text><text x="100" y="26" fill="#8a3b3b">+1</text><text x="20" y="66" fill="#8a3b3b">-1</text><text x="60" y="66" fill="#8a3b3b">0</text><text x="100" y="66" fill="#8a3b3b">+1</text><text x="20" y="106" fill="#8a3b3b">-1</text><text x="60" y="106" fill="#8a3b3b">0</text><text x="100" y="106" fill="#8a3b3b">+1</text></g><text x="120" y="72" fill="#5E5191" font-size="9">left</text><text x="120" y="84" fill="#5E5191" font-size="9">(minus)</text><text x="400" y="72" fill="#276b45" font-size="9">right</text><text x="400" y="84" fill="#276b45" font-size="9">(plus)</text><text x="260" y="170" fill="#6B645E" font-size="10">flat area → near 0 · vertical edge → big number</text></g></svg>
%%%

**What the contrast-dial picture gets right:** the stencil ignores flat regions and reacts to *change* in brightness, just like turning up contrast makes edges pop. **Where it breaks down:** a contrast dial changes the whole picture at once; this stencil reports edges spot-by-spot into a feature map, and it only catches *vertical* edges — you'd need a rotated version for horizontal ones.

#### Read it by hand on two patches
The kernel is "`+1` on the right column, `-1` on the left, `0` in the middle." In plain words: **add up the right side, subtract the left side.** On a flat patch the two sides cancel to near zero. On an edge, one side is bright and the other dark, so they don't cancel — you get a big number. **Predict first:** which patch below scores near 0, and which scores high? Then reveal:

%%% demo id=edge label="predict, then reveal"
code: flat patch = [[5,5,5],[5,5,5],[5,5,5]]   (all one shade)
   edge patch = [[0,0,9],[0,0,9],[0,0,9]]   (dark left, bright right)
   → for each patch:  right column − left column
out: [flat]  (5+5+5) − (5+5+5) = 15 − 15 = 0      → "no edge here"
   [edge]  (9+9+9) − (0+0+0) = 27 − 0  = 27      → "strong edge here!"
take: <b>The stencil scores 0 on the flat patch and 27 on the edge.</b> Same kernel, two spots, two very different answers — that IS the feature map at work. Sweep this one hand-made kernel over a whole photo and its feature map traces every vertical edge in the image.
%%%

#### The whole point: this is what learning discovers
You just *designed* an edge detector and read its output by hand. When we later *train* a convolution layer, it discovers stencils like this one — edge detectors, corner detectors, color-blob detectors — all on its own, no hand-tuning. **Now you know what those learned numbers are *for*:** each kernel is a little pattern-spotter, and the feature map is its report card.

#### So how do the 9 numbers *find* a good shape?
You do not sit and tune the weights by hand — the model nudges them for you, a tiny bit at a time. Here is the whole idea in plain words, no math needed today.

Picture the **"you're getting warmer" game** you played as a kid: you hunt for a hidden object, and a friend says "warmer" when you step closer, "colder" when you step away. You do not need a map — you just keep stepping in whatever direction was warmer. Training a kernel works the same way. The model starts with 9 *random* weights (a stencil that spots nothing useful), makes a guess on some pictures, and gets told how wrong it was. Then it asks, weight by weight: "if I nudge *this* number up a hair, does the answer get warmer or colder?" It shifts every weight a little toward "warmer," looks at more pictures, and repeats — thousands of times. This nudge-toward-warmer routine is called [[gradient descent||A step-by-step "getting warmer" search: nudge each weight a little in whatever direction lowers the error, and repeat many times until the weights settle into a useful shape.]].

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A getting-warmer search for kernel weights, shown as three stencils in a row. Left: a stencil of random numbers labelled start, random, spots nothing, sitting high up a slope of error. Middle: after a few warmer nudges the numbers have shifted and it sits partway down the slope. Right: the numbers have settled into the minus one zero plus one edge-detector shape at the bottom of the slope, labelled learned edge detector."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Training = nudge the 9 weights "warmer" until a useful stencil appears</text><path d="M30 60 Q160 40 260 120 Q360 175 490 150" fill="none" stroke="#E5DFD6" stroke-width="6"/><text x="500" y="150" fill="#9A938A" font-size="8" text-anchor="end">less error →</text><g transform="translate(40,42)"><rect x="0" y="0" width="54" height="54" fill="#FDECEC" stroke="#C93B3B"/><path d="M18 0 V54 M36 0 V54 M0 18 H54 M0 36 H54" stroke="#C93B3B" stroke-width="0.5"/><text x="9" y="13" fill="#8a3b3b" font-size="8">7</text><text x="27" y="13" fill="#8a3b3b" font-size="8">2</text><text x="45" y="13" fill="#8a3b3b" font-size="8">4</text><text x="9" y="31" fill="#8a3b3b" font-size="8">1</text><text x="27" y="31" fill="#8a3b3b" font-size="8">9</text><text x="45" y="31" fill="#8a3b3b" font-size="8">3</text><text x="9" y="49" fill="#8a3b3b" font-size="8">6</text><text x="27" y="49" fill="#8a3b3b" font-size="8">0</text><text x="45" y="49" fill="#8a3b3b" font-size="8">5</text><text x="27" y="70" fill="#6B645E">start: random</text></g><text x="150" y="95" fill="#9A5A12" font-size="11">warmer →</text><g transform="translate(220,86)"><rect x="0" y="0" width="54" height="54" fill="#FDF2E0" stroke="#C99A12"/><path d="M18 0 V54 M36 0 V54 M0 18 H54 M0 36 H54" stroke="#C99A12" stroke-width="0.5"/><text x="9" y="13" fill="#8A6D3B" font-size="8">-1</text><text x="27" y="13" fill="#8A6D3B" font-size="8">1</text><text x="45" y="13" fill="#8A6D3B" font-size="8">2</text><text x="9" y="31" fill="#8A6D3B" font-size="8">0</text><text x="27" y="31" fill="#8A6D3B" font-size="8">1</text><text x="45" y="31" fill="#8A6D3B" font-size="8">1</text><text x="9" y="49" fill="#8A6D3B" font-size="8">-1</text><text x="27" y="49" fill="#8A6D3B" font-size="8">0</text><text x="45" y="49" fill="#8A6D3B" font-size="8">1</text><text x="27" y="70" fill="#6B645E">after nudges</text></g><text x="335" y="128" fill="#9A5A12" font-size="11">warmer →</text><g transform="translate(410,116)"><rect x="0" y="0" width="54" height="54" fill="#EAF5EE" stroke="#2D8B55"/><path d="M18 0 V54 M36 0 V54 M0 18 H54 M0 36 H54" stroke="#2D8B55" stroke-width="0.5"/><text x="9" y="13" fill="#276b45" font-size="8">-1</text><text x="27" y="13" fill="#276b45" font-size="8">0</text><text x="45" y="13" fill="#276b45" font-size="8">1</text><text x="9" y="31" fill="#276b45" font-size="8">-1</text><text x="27" y="31" fill="#276b45" font-size="8">0</text><text x="45" y="31" fill="#276b45" font-size="8">1</text><text x="9" y="49" fill="#276b45" font-size="8">-1</text><text x="27" y="49" fill="#276b45" font-size="8">0</text><text x="45" y="49" fill="#276b45" font-size="8">1</text><text x="27" y="70" fill="#276b45">learned edge detector</text></g></g></svg>
%%%

**What the warmer-game picture gets right:** you never need the full map — just "which little step lowers the error" — and repeating it enough times walks the weights into a useful shape, exactly like the random stencil settling into an edge detector. **Where it breaks down:** in the game *you* decide the step; in training the "warmer/colder" signal for every weight at once is computed by a rule called backpropagation.

!!! c-info 🧭
<b>Optional (skippable) — where the "warmer" signal comes from.</b> [[backpropagation||The rule that computes, for every weight at once, which small change would lower the error — the "warmer/colder" signal that gradient descent then follows. It works backward from the error through the network.]] is the actual arithmetic that measures which way is warmer for each of the 9 weights, all in one efficient backward pass through the network. It is a whole lesson of its own, so we don't unpack it today — you'll meet how CNNs are trained end-to-end in <b>day-03 (the CNN classifier)</b>, and you already saw the plain-words version of the idea in earlier modules. Today you only need the *feel*: random weights get nudged, over and over, toward stencils that spot real patterns. (Note: tomorrow, day-02, is the output-size arithmetic — a different, gentler drill.)
!!!

@@@ concept id=c6 tag="Two sliding dials" title="Stride and padding: the two knobs for how it slides" gotit="Got the dials"
The sliding stencil has two dials that decide *how* it sweeps and how big the output comes out. They're both simple, and they solve two real puzzles.

Think of mowing a **lawn** in straight passes. Dial one is your *step size*: after each pass, do you shift over by one strip, or skip ahead two strips? Bigger steps mow the lawn faster but miss thin bits. Dial two is the *edge problem*: the mower can't sit past the fence, so the strip right at the fence gets shortchanged.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A lawn-mowing analogy for the two convolution dials. Left panel shows stride: passes one strip apart versus two strips apart, with bigger steps covering fewer passes. Right panel shows the edge problem: the mower cannot go past the fence, so a border of the lawn is missed unless you add extra space."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Two dials for the sweep: step size (stride) and the edge fix (padding)</text><text x="130" y="38" fill="#5E5191">Dial 1 · step size (stride)</text><g transform="translate(40,48)"><rect x="0" y="0" width="80" height="110" fill="#EAF5EE" stroke="#2D8B55"/><g stroke="#2D8B55" stroke-width="2"><line x1="12" y1="8" x2="12" y2="102"/><line x1="28" y1="8" x2="28" y2="102"/><line x1="44" y1="8" x2="44" y2="102"/><line x1="60" y1="8" x2="60" y2="102"/><line x1="76" y1="8" x2="76" y2="102"/></g><text x="40" y="130" fill="#276b45" font-size="9">stride 1: every strip</text></g><g transform="translate(150,48)"><rect x="0" y="0" width="80" height="110" fill="#EAF5EE" stroke="#2D8B55"/><g stroke="#2D8B55" stroke-width="2"><line x1="12" y1="8" x2="12" y2="102"/><line x1="44" y1="8" x2="44" y2="102"/><line x1="76" y1="8" x2="76" y2="102"/></g><text x="40" y="130" fill="#276b45" font-size="9">stride 2: skip a strip</text></g><text x="390" y="38" fill="#9A5A12">Dial 2 · edge fix (padding)</text><g transform="translate(320,48)"><rect x="10" y="10" width="90" height="90" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><rect x="0" y="0" width="110" height="110" fill="none" stroke="#C93B3B" stroke-dasharray="4,3" stroke-width="1.5"/><text x="55" y="60" fill="#8A6D3B" font-size="9">image</text></g><text x="375" y="130" fill="#9A5A12" font-size="9">add a border so the</text><text x="375" y="143" fill="#9A5A12" font-size="9">stencil reaches the edge</text></g></svg>
%%%

**What the lawn picture gets right:** a bigger step covers the ground in fewer passes, and the mower really can't reach past the fence — so edges get missed. **Where it breaks down:** a mower overlaps its passes; a strided convolution can *skip* pixels entirely, and the "extra space" we add at the edge is fake (zeros), not real lawn.

#### Dial 1 · Stride — how far the stencil jumps
The [[stride||How many pixels the kernel jumps between stops. Stride 1 checks every position; stride 2 skips every other one, halving the output size.]] is how many pixels the kernel moves between stops. Stride `1` visits every position (the default). Stride `2` jumps two pixels each time, so it visits half as many spots across and half as many down — the feature map comes out about *half the size* in each direction. Bigger stride = smaller, cheaper output, at the cost of skipping detail.

#### Dial 2 · Padding — let the stencil reach the edges
Here's the edge puzzle. A `3 × 3` stencil can't be *centered* on a corner pixel — part of it would hang off the picture. So a plain "valid" sweep quietly shrinks the output and *drops the border pixels*. The fix is [[padding||Adding a border of extra pixels (usually zeros) around the image so the kernel can sit at the edges. It stops the output from shrinking and stops border pixels being ignored.]]: we glue a border of zeros around the image so the stencil *can* sit at every edge. Two common choices have names:

%%% table
:: Padding :: What it does :: Output size
"valid" (none) :: no border; stencil stays fully inside :: shrinks a little, drops the border
"same" (zeros) :: add a zero border so the stencil reaches every edge :: **same** height & width as the input
%%%

**What the padding fixes:** without it, every conv layer nibbles the image smaller and forgets its edges. That edge information loss is a real failure mode — and "same" padding is its simple remedy.

#### One tiny formula ties it together
How big is the output? It's just: start with the image, subtract the kernel (it can't hang off the edge), add back any padding you glued on, then divide by the stride (how many you skip). In symbols, with input side `N`, kernel `K`, padding `P`, stride `S`:

%%% formula
expr: output side = (N − K + 2P) / S  + 1
note: N = input height/width, K = kernel size, P = pixels of padding per side, S = stride. Do it once for height, once for width.
%%%

Let's turn each dial and *watch the output size change* — same `32 × 32` image, same `3 × 3` kernel, three settings:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three bars showing the output side length of a 32 by 32 image convolved with a 3 by 3 kernel under different settings. Valid padding stride 1 gives 30. Same padding stride 1 gives 32, the tallest. Same padding stride 2 gives 16, the shortest. This shows padding preserves size and stride shrinks it."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">32×32 image, 3×3 kernel — turning the dials changes the output side</text><line x1="55" y1="165" x2="480" y2="165" stroke="#E5DFD6" stroke-width="1.5"/><rect x="90" y="45" width="70" height="120" fill="#C99A12"/><text x="125" y="37" fill="#8A6D3B">30</text><text x="125" y="182" fill="#6B645E">valid, S=1</text><text x="125" y="128" fill="#fff" font-size="8">(32-3)/1+1</text><rect x="225" y="37" width="70" height="128" fill="#2D8B55"/><text x="260" y="29" fill="#276b45">32</text><text x="260" y="182" fill="#6B645E">same, S=1</text><text x="260" y="128" fill="#fff" font-size="8">(32-3+2)/1+1</text><rect x="360" y="101" width="70" height="64" fill="#5E5191"/><text x="395" y="93" fill="#5E5191">16</text><text x="395" y="182" fill="#6B645E">same, S=2</text><text x="395" y="140" fill="#fff" font-size="8">≈ half</text></g></svg>
%%%

Read it left to right: "valid" nibbles `32 → 30`; "same" padding holds it at `32`; adding stride `2` roughly *halves* it to `16`. **You now hold both sliding dials** — padding to protect the edges, stride to shrink on purpose. (Tomorrow's whole lesson is practicing this output-size arithmetic until it's automatic — today, just feel the two dials move.)

@@@ concept id=c7 tag="Depth and a bank" title="Channels and a bank of stencils" gotit="Got channels & filters"
Two quick upgrades turn our single stencil into a real convolution layer. Both are about *depth* — stacks of grids instead of one flat grid.

Think of a **stack of tracing-paper sheets**. A color photo isn't one sheet — it's three sheets stacked: one for how much *red* is at each spot, one for *green*, one for *blue*. Each sheet is called a [[channel||One layer of a stacked image or feature map. A color image has 3 channels (red, green, blue); a grayscale image has 1.]]. So a color image is `height × width × 3`, three tracing sheets lined up.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A color image shown as three stacked tracing-paper sheets, one red one green one blue, labelled 3 channels. Beside it, one kernel is shown as a matching stack of three small grids, spanning all three channels at once, producing a single feature map."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A color image = 3 stacked sheets (channels); the kernel spans all 3</text><text x="95" y="40" fill="#6B645E">image: 3 channels</text><g transform="translate(60,50)"><rect x="20" y="20" width="70" height="70" fill="#EAF0FA" stroke="#5E5191"/><rect x="10" y="10" width="70" height="70" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="0" width="70" height="70" fill="#FDECEC" stroke="#C93B3B"/><text x="35" y="40" fill="#C93B3B" font-size="9">R</text><text x="45" y="55" fill="#2D8B55" font-size="9">G</text><text x="55" y="70" fill="#5E5191" font-size="9">B</text></g><text x="200" y="90" fill="#B8AEA2" font-size="14">→</text><text x="300" y="40" fill="#8A6D3B">one kernel: also 3 deep</text><g transform="translate(270,50)"><rect x="20" y="20" width="50" height="50" fill="#FBE6E6" stroke="#5E5191"/><rect x="10" y="10" width="50" height="50" fill="#FBE6E6" stroke="#2D8B55"/><rect x="0" y="0" width="50" height="50" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><path d="M17 0 V50 M33 0 V50 M0 17 H50 M0 33 H50" stroke="#C93B3B" stroke-width="0.5"/></g><text x="380" y="90" fill="#B8AEA2" font-size="14">→</text><rect x="415" y="55" width="55" height="55" fill="#0d2b1a" stroke="#2D8B55"/><text x="442" y="128" fill="#276b45" font-size="9">1 feature map</text></g></svg>
%%%

**What the tracing-paper picture gets right:** the three color layers are stacked and lined up spot-for-spot, and one kernel reads through *all three* at once. **Where it breaks down:** tracing paper is see-through so you blend colors by eye; the kernel doesn't blend — it has its *own* little grid for each channel and sums all of them into one number.

#### Upgrade 1 · a kernel spans all the input channels
When the input has 3 channels, the kernel does too: a `3 × 3` kernel on a color image is really `3 × 3 × 3` — a little grid for red, one for green, one for blue. It multiplies each pixel by its matching weight across *all* channels and adds everything into **one** number. So no matter how many input channels go in, one kernel still produces **one** flat feature map.

#### Upgrade 2 · a layer has a whole bank of kernels
One stencil finds *one* pattern. But a picture has edges *and* corners *and* textures *and* colors. So a convolution layer doesn't use one kernel — it uses a whole [[filter bank||The set of many kernels in one conv layer. Each kernel detects a different pattern and produces its own feature map, so the layer outputs many channels.]] of them, each a different stencil hunting a different pattern. Each kernel makes its own feature map, and we stack those maps as the *output* channels. So a layer can take in 3 channels and pump out, say, 32 — one map per pattern-spotter.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A bank of many kernels applied to one input. On the left one input image. In the middle a stack of several different kernels, each labelled as a different pattern spotter such as vertical edge, horizontal edge, corner, blob. On the right each kernel produces its own feature map, stacked into many output channels."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A bank of stencils → many feature maps (many output channels)</text><rect x="30" y="60" width="70" height="70" fill="#FCF3DC" stroke="#C99A12"/><text x="65" y="148" fill="#6B645E">1 input</text><text x="130" y="95" fill="#B8AEA2" font-size="13">→</text><text x="230" y="42" fill="#8a3b3b">bank of kernels</text><g transform="translate(160,52)"><rect x="30" y="30" width="40" height="40" fill="#FDECEC" stroke="#C93B3B"/><rect x="15" y="15" width="40" height="40" fill="#FDECEC" stroke="#C93B3B"/><rect x="0" y="0" width="40" height="40" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="20" y="24" fill="#8a3b3b" font-size="12">▏</text></g><text x="265" y="95" fill="#B8AEA2" font-size="13">→</text><g transform="translate(320,45)"><rect x="45" y="45" width="50" height="50" fill="#0d2b1a" stroke="#2D8B55"/><rect x="25" y="25" width="50" height="50" fill="#0d2b1a" stroke="#2D8B55"/><rect x="0" y="0" width="50" height="50" fill="#0d2b1a" stroke="#2D8B55"/><rect x="12" y="0" width="4" height="50" fill="#7CE0A8"/></g><text x="405" y="112" fill="#276b45" font-size="9">many feature maps</text><text x="405" y="152" fill="#9A938A" font-size="9">stacked = output channels</text><text x="230" y="150" fill="#9A938A" font-size="9">edge · corner · blob · …</text></g></svg>
%%%

**Now the shape story is complete:** channels go *in* (however many), a bank of kernels reads all of them, and channels come *out* (one per kernel). Depth in, depth out — that stacking is exactly how CNNs build up from simple edges to rich shapes layer after layer. **You've just seen how one layer sees many things at once.**

@@@ concept id=c8 tag="Shrink it: pooling" title="Pooling: shrink the map, keep what matters" gotit="Got pooling"
Our feature maps can be big, and full of fine detail we don't always need. So after convolution we often *shrink* the map on purpose — keeping the important bits and throwing away the fussy ones. This shrinking step is called [[pooling||Downsampling a feature map by summarizing each small window into one value, making it smaller and a bit shift-tolerant.]].

Think of writing a **class photo caption**. You don't list every pixel of every face; you summarize each little group with one word — "smiling," "waving." Pooling does that to a feature map: it chops the map into small windows and replaces each window with *one* summary number.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A pooling summary analogy. On the left a 4 by 4 feature map is divided into four 2 by 2 windows. An arrow labelled summarize each window points to a 2 by 2 output where each cell holds one summary number for its window, so the map is shrunk by half in each direction."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Pooling = split into small windows, replace each window with one number</text><text x="110" y="40" fill="#6B645E">feature map (4×4)</text><g transform="translate(55,50)"><rect x="0" y="0" width="120" height="120" fill="#EAF5EE" stroke="#2D8B55"/><path d="M30 0 V120 M60 0 V120 M90 0 V120 M0 30 H120 M0 60 H120 M0 90 H120" stroke="#2D8B55" stroke-width="0.4"/><path d="M60 0 V120 M0 60 H120" stroke="#276b45" stroke-width="2.5"/><text x="15" y="20" fill="#1a5c38">1</text><text x="45" y="20" fill="#1a5c38">3</text><text x="15" y="50" fill="#1a5c38">2</text><text x="45" y="50" fill="#1a5c38">0</text></g><text x="215" y="112" fill="#9A5A12" font-size="11">summarize →</text><g transform="translate(320,80)"><rect x="0" y="0" width="60" height="60" fill="#0d2b1a" stroke="#2D8B55"/><path d="M30 0 V60 M0 30 H60" stroke="#276b45" stroke-width="1"/></g><text x="350" y="160" fill="#276b45" font-size="9">output (2×2): 1 number per window</text><text x="440" y="112" fill="#9A938A" font-size="9">half the size</text></g></svg>
%%%

**What the caption picture gets right:** each little group gets boiled down to one summary, so the whole thing gets shorter but keeps the gist. **Where it breaks down:** a caption uses judgment; pooling uses one dumb-but-fast rule — either "take the biggest" or "take the average" — applied to every window.

#### The two summaries: max vs average
There are two common ways to summarize a window:

- [[max pooling||Take the biggest value in each window. It keeps the strongest activation and gives a little tolerance to small shifts. The most common pooling.]] — keep only the **biggest** number in each window. Since a big feature-map value means "pattern strongly found here," max pooling keeps the *loudest* evidence and drops the rest. It's the popular choice.
- [[average pooling||Take the mean of each window instead of the max. It smooths the window into a gentle overall level rather than keeping the single strongest response.]] — take the **mean** of the window. This smooths things into a gentle overall level instead of keeping one loud spike.

**Predict first:** for the window `[[1,3],[2,0]]`, what does *max* pooling give, and what does *average* pooling give? Then reveal:

%%% demo id=pool label="predict, then reveal"
code: window = [[1, 3], [2, 0]]   (one 2×2 patch of a feature map)
out: max pooling      → max(1, 3, 2, 0) = 3      (keep the loudest)
   average pooling  → (1+3+2+0) / 4 = 1.5    (blend them all)
take: <b>Max keeps the strongest signal; average blends everything.</b> Both turn a 2×2 window into ONE number, shrinking the map by half in each direction. Max is the usual pick because "was the pattern strongly present?" matters more than "what was the average level?"
%%%

#### Two dials again: window and stride
Pooling has the same two dials as convolution. The [[pool size||The side of each pooling window, e.g. 2×2. It sets how many values get summarized into one.]] (window) is how big each group is — `2 × 2` is standard. The pool *stride* is how far the window jumps — usually equal to the window, so windows don't overlap. A `2 × 2` window with stride `2` shrinks the map to *half* in each direction, so it becomes a *quarter* of the number of values.

#### Why bother shrinking? Three quiet wins
Pooling isn't just tidying — it earns its place three ways:

1. **Cheaper compute.** A quarter as many values means the next layer does far less work.
2. **A wider view.** After pooling, one value now summarizes a bigger patch of the *original* image — so deeper layers effectively "see" more of the picture at once (a growing receptive field).
3. **A little shift-tolerance.** Here's the neat one: if the strongest response wiggles one pixel sideways *within* a window, max pooling still reports the same biggest number. So the output barely changes when the input jitters slightly. This is [[translation invariance||When a small shift of the input leaves the output unchanged. Max pooling gives a little of this: nudging the peak within a window keeps the same max.]] — and it's why a network can recognize a cat even if it's a few pixels off from where it trained.

!!! c-info 🧭
<b>Optional (skippable) — equivariance vs invariance.</b> These two cousins are easy to mix up. Convolution is *equivariant*: shift the input and the feature map shifts the same way (the response *moves*). Pooling adds a dash of *invariance*: after summarizing a window, a tiny shift leaves the output the *same* (the response *stays put*). Convolution says "the pattern moved *there*"; pooling says "the pattern is *around here* — close enough." You want both: know roughly where things are, but don't panic over a one-pixel wobble.
!!!

**That's the last building block.** Convolution finds patterns; pooling shrinks the map, widens the view, and shrugs off tiny wobbles. Stack these two — conv, pool, conv, pool — and you have the skeleton of every classic vision network.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
You built the engine of machine vision today. Here's a fresh way to hold the whole day: picture it as a **stencil-and-shrink pipeline**, where a raw image rolls in one end and a small, meaningful map rolls out the other. Each stop only makes sense because of the one before it. **Where the pipeline picture breaks down:** a real pipeline runs once and stops, but these steps repeat — conv, pool, conv, pool — stacking into a deep network you'll build in later days.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A pipeline with five stops in order. Stop one: an image which is a grid of pixels with channels. Stop two: slide a tiny kernel stencil across it. Stop three: get a feature map of scores. Stop four: a whole bank of kernels makes many feature maps. Stop five: pooling shrinks each map while keeping the strongest signal, ready to repeat."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Today = a stencil-and-shrink pipeline: image → small meaningful map</text><line x1="20" y1="118" x2="500" y2="118" stroke="#E5DFD6" stroke-width="6"/><g transform="translate(28,58)"><rect x="0" y="0" width="58" height="48" fill="#FCF3DC" stroke="#C99A12"/><path d="M19 0 V48 M38 0 V48 M0 16 H58 M0 32 H58" stroke="#C99A12" stroke-width="0.5"/><text x="29" y="68" fill="#6B645E">1 · image</text><text x="29" y="80" fill="#9A938A">pixels + channels</text></g><text x="98" y="86" fill="#9A5A12">→</text><g transform="translate(110,58)"><rect x="0" y="0" width="58" height="48" fill="#FCF3DC" stroke="#C99A12"/><rect x="0" y="0" width="24" height="24" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="29" y="68" fill="#6B645E">2 · slide kernel</text><text x="29" y="80" fill="#9A938A">the stencil</text></g><text x="180" y="86" fill="#9A5A12">→</text><g transform="translate(192,66)"><rect x="0" y="0" width="52" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="8" y="0" width="6" height="34" fill="#7CE0A8"/><text x="26" y="54" fill="#276b45">3 · feature map</text><text x="26" y="66" fill="#9A938A">scores</text></g><text x="256" y="86" fill="#9A5A12">→</text><g transform="translate(268,60)"><rect x="16" y="6" width="46" height="34" fill="#0d2b1a" stroke="#2D8B55"/><rect x="8" y="3" width="46" height="34" fill="#0d2b1a" stroke="#2D8B55"/><rect x="0" y="0" width="46" height="34" fill="#0d2b1a" stroke="#2D8B55"/><text x="31" y="60" fill="#276b45">4 · bank of kernels</text><text x="31" y="72" fill="#9A938A">many maps</text></g><text x="342" y="86" fill="#9A5A12">→</text><g transform="translate(360,70)"><rect x="0" y="0" width="30" height="26" fill="#0d2b1a" stroke="#2D8B55"/><path d="M15 0 V26 M0 13 H30" stroke="#276b45" stroke-width="1"/><text x="15" y="48" fill="#276b45">5 · pool</text><text x="15" y="60" fill="#9A938A">shrink, keep max</text></g><text x="418" y="86" fill="#9A5A12">↺</text><circle cx="470" cy="83" r="7" fill="#C93B3B"/><text x="470" y="70" fill="#6B645E">repeat 🏁</text></g></svg>
%%%

The five stops, in order:
- **Stop 1 · The image.** A grid of pixels, with 1 channel (gray) or 3 (color, R/G/B).
- **Stop 2 · Slide the stencil.** A tiny kernel of learned weights sweeps across, looking at one small local patch (its receptive field) at a time — the same weights everywhere (weight sharing). Those weights start random and get nudged "warmer" by gradient descent until they spot real patterns.
- **Stop 3 · The feature map.** Each stop is one dot product → one score. The grid of scores lights up where the kernel's pattern is found.
- **Stop 4 · A bank of kernels.** Many stencils, each spotting a different pattern, each spanning all input channels — so many feature maps stack up as output channels.
- **Stop 5 · Pool.** Shrink each map (usually max pooling, `2 × 2`), keeping the strongest signal — cheaper, wider view, a little shift-tolerance.
- **And the honest bits:** a plain "valid" sweep drops border pixels (fix: "same" padding), and a rigid grid still can't reason about the whole image at once — that's what stacking many layers is for.

#### Cheat-sheet · the pieces at a glance
%%% table
:: Piece :: What it is :: One-line reason
Kernel (filter) :: a small grid of learned weights, e.g. 3×3 :: the "stencil" — one pattern-spotter
Convolution :: slide the kernel, dot-product at each spot :: turns an image into a map of where a pattern is
Feature map :: the grid of scores from one kernel :: bright = pattern found here
Weight sharing :: same weights at every position :: tiny model + finds a pattern anywhere
Stride :: how far the kernel jumps :: bigger stride → smaller, cheaper output
Padding :: a zero border around the image :: "same" keeps the size and protects edges
Output size :: `(N − K + 2P)/S + 1` :: plug in image, kernel, padding, stride
Channels :: the stacked layers (3 for color) :: one kernel spans all input channels
Filter bank :: many kernels in one layer :: many patterns → many output channels
Pooling :: summarize each window into one value :: shrink + widen view + shift-tolerance
%%%

#### Cheat-sheet · the words you met today
%%% jargon
kernel (filter) | a small grid of learned weights — the stencil convolution slides
convolution | slide the kernel and dot-product at each spot to make a feature map
feature map | the grid of scores; bright where the kernel's pattern was found
receptive field | the small local patch one output value looks at
weight sharing | reusing the same kernel weights at every position
translation equivariance | shift the input, the feature map shifts the same way
gradient descent | nudge each weight "warmer" over and over until the stencil is useful
backpropagation | the rule that computes the "warmer/colder" signal for every weight (day-03)
stride | how many pixels the kernel jumps between stops
padding | a zero border so the kernel can sit at the edges ("same" vs "valid")
channel | one layer of a stacked image or map (3 for color: R,G,B)
filter bank | the set of many kernels in one conv layer
pooling | summarize each small window into one value to shrink the map
max pooling | keep the biggest value in each window (the common choice)
average pooling | take the mean of each window instead of the max
translation invariance | a small input shift leaves the output unchanged (from pooling)
%%%

That's the whole day. Next you'll drill the **output-size formula** you just met — `(N − K + 2P)/S + 1` — until output sizes are second nature. That's tomorrow's lesson, **day-02: output-size arithmetic**, the mental math every CNN designer does in their head.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: Why is a fully-connected layer a bad idea for a whole image? | a:2 | It is too slow to draw | It cannot use color | It needs a separate weight for every pixel-to-neuron pair, so the count explodes, and flattening throws away which pixels are neighbors | It can only see edges | fb: A 100×100×3 image into 1000 neurons is 30 million weights for ONE layer, and flattening scrambles neighbor pixels. Convolution fixes both: a tiny shared kernel that looks at local patches.
q: What is "weight sharing" in convolution, and why does it help? | a:1 | Two models share one GPU | The SAME small kernel weights are reused at every position — keeping the model tiny AND letting it spot a pattern anywhere in the image | The weights are split across layers | It shares weights between color channels only | fb: One stencil, slid everywhere. That's why a 3×3 kernel is just 9 weights no matter the image size, and why a learned edge-detector finds edges wherever they appear (translation equivariance).
q: A 32×32 image, 3×3 kernel, "same" padding, stride 2 — roughly what output side? | a:2 | 32, because "same" always keeps the size | 30, because the kernel nibbles it | About 16, because "same" padding holds the size but stride 2 roughly halves each side | 64, because padding doubles it | fb: Use (N−K+2P)/S + 1. "Same" padding cancels the kernel nibble, but stride 2 skips every other spot, so 32 → about 16 in each direction.
q: What does MAX pooling do, and what's one benefit? | a:1 | Averages every window to smooth noise | Keeps the biggest value in each small window — shrinking the map, widening the view, and giving a little tolerance to small shifts | Adds a zero border to the map | Detects vertical edges | fb: Max pooling keeps the loudest response per window. A 2×2 window with stride 2 quarters the values (cheaper + wider view), and a peak wobbling within a window still gives the same max — a dash of translation invariance.
%%%

@@@ produce id=produce tag="Produce" title="Slide a stencil with your own hands" gotit="Done"
Time to see today's big idea with your own eyes. You'll build a tiny fake "image," slide a hand-made edge-detecting kernel across it to make a feature map, then max-pool the map to shrink it — and *watch* the numbers confirm every idea from today. **Predict first:** if you slide a `3 × 3` kernel over a `6 × 6` image with no padding and stride 1, how big is the feature map? And after `2 × 2` max pooling, how big then? Then run it and **watch** the sizes shrink exactly as the formula says. Pick one path.

#### Option A · write it yourself
Create `sessions/m06-cnns-vision-encoders/day-01-convolution-pooling/experiment.py`. Make a small fake image with a bright vertical stripe, e.g. `img = np.zeros((6,6)); img[:, 3:] = 9` (left half dark, right half bright — a clean vertical edge). Define the edge kernel `k = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])` (right minus left). Write a `convolve(img, k)` that slides the kernel with stride 1 and no padding, computing the dot product `np.sum(patch * k)` at each spot, and **print the shape at every step**. **Observe** that the feature map is `4 × 4` (from `(6−3)/1 + 1 = 4`) and that it is near 0 over the flat halves but *spikes* along the edge column — the stencil found the edge. Then write a `max_pool(fmap, 2)` that takes the max of each non-overlapping `2 × 2` window and **notice** the map shrinks to `2 × 2`. Run with `python3 sessions/m06-cnns-vision-encoders/day-01-convolution-pooling/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 6 Day 1 artifact.

Create sessions/m06-cnns-vision-encoders/day-01-convolution-pooling/experiment.py that, with a comment on each step and printing shapes at every step:
1. Makes a 6×6 fake image with a vertical edge: img = np.zeros((6,6)); img[:, 3:] = 9. Prints its shape.
2. Defines an edge kernel k = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) (right column minus left column).
3. Defines convolve(img, k) that slides the 3×3 kernel with stride 1, no padding, computing np.sum(patch * k) at each position, and returns the feature map. Prints the feature map's shape (expect 4×4, from (6-3)/1 + 1 = 4) and prints the map itself — point out it is ~0 over the flat halves and spikes along the edge.
4. Defines max_pool(fmap, size=2) that takes the max of each non-overlapping 2×2 window. Prints the pooled shape (expect 2×2) and the pooled values.
5. Prints a one-line count: the 3×3 kernel is just 9 weights, versus a fully-connected layer on the flattened 36-pixel image into (say) 10 neurons = 360 weights — to show weight sharing shrinks the model.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (slide, then shrink)
- The feature map comes out `4 × 4` — exactly what the formula `(N − K + 2P)/S + 1 = (6 − 3 + 0)/1 + 1 = 4` predicts.
- The map is **near 0** over the flat dark and flat bright halves, but **spikes** along the column where dark meets bright — the hand-made stencil found the vertical edge, spot by spot. That IS a feature map.
- After `2 × 2` max pooling the map shrinks to `2 × 2` — a quarter of the values — while the biggest (edge) responses survive. That's the shrink-but-keep-the-loudest behavior.
- The kernel is just **9 weights** no matter the image size — weight sharing in action, next to a fully-connected count that grows with every pixel.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m06-cnns-vision-encoders/day-01-convolution-pooling/log.md`: (1) in one sentence, what convolution does to an image; (2) why weight sharing makes a CNN so much smaller than a fully-connected layer; (3) one thing max pooling gives you that convolution alone does not.
!!!

@@@ fin
