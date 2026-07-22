---
quest_id: w01-d06-vit
mode: concept
donor: v9-base.donor
page_title: "Week 1 · Weekend Lab — ViT Capstone: Build & Train From Scratch"
module_label: "Thinking in JAX · Week 1 · Weekend Lab"
title: "ViT Capstone"
subtitle: "Build and Train a Vision Transformer From Scratch"
brand_sub: "Frontier Lab · Week 1 Weekend Lab"
spine: "assembly"
nav_prev_href: "../day-05-flax-optax/lesson.html"
nav_prev_label: "Flax & Optax"
nav_next_href: "../../m08-transformer-math/day-01-transformer-arithmetic/lesson.html"
nav_next_label: "M8 · Transformer Math + Search"
fin_title: "Week 1 Weekend Lab complete! 🏆"
fin_body: "Incredible — you've finished the ViT Capstone and all of Week 1! This was the <b>assembly</b> step: you snapped together everything from the week — immutable arrays, explicit keys, vmap, jit, and Flax+Optax — to build and train a Vision Transformer from scratch and watch its loss fall. You now think in JAX: pure functions in, transformed functions out.<br>Next up: <b>Module 8 — Transformer Math + Search</b>."
notebook_yardstick: null
---

@@@ hero
@lede Think about building something from a **LEGO kit**. You don't invent new bricks — every brick already exists in the box. The magic is the *instruction booklet*: it tells you which brick to snap on next, in what order, until a pile of loose pieces becomes a spaceship. Today is exactly that kind of build. All week you collected bricks — arrays that never change, explicit dice for randomness, `vmap` to handle a whole batch, `jit` to go fast, and Flax+Optax to train. Now we open the booklet and do the **assembly**: we snap those bricks together into a real, working **Vision Transformer** — the same family of model that powers modern image AI, the thing that lets a computer look at a photo and say "that's a cat." You'll build it piece by piece, from raw pixels to a final guess, feed it a small pile of real little pictures, and then watch it *learn* — the loss falling and the accuracy climbing. By the end you won't just have used JAX — you'll have assembled a frontier-shaped model with your own hands and trained it end to end.
@goal Together we'll do the full <b>assembly</b> of a Vision Transformer (a <b>ViT</b>) in JAX, end to end: cut an image into square <b>patches</b>, turn each patch into a vector and tag it with its position, add one special summary slot, run the stack through transformer <b>encoder blocks</b> (borrowed whole from the text side), and read out a class guess from a tiny head. Then we <b>train it end to end on a small dataset</b> and read <i>both</i> the loss curve and the accuracy to confirm it learns. Alongside the model, we assemble the JAX machinery that builds and trains it — parameters that live outside the model as a plain tree of arrays that <i>never change in place</i>, explicit random keys you split, <code>vmap</code> to run over a batch, <code>jit</code> to compile it fast, and <code>grad</code> for one clean training step. Every brick gets named the moment we pick it up, and every trap that snags beginners gets a one-line fix.

@@@ concept id=c1 tag="The whole build" title="What a Vision Transformer is — the assembly line, top to bottom" gotit="Got the big picture"
Before we pick up any single brick, let's look at the finished instruction booklet — the whole **assembly** in one glance. Picture a **factory assembly line making a toy.** Raw material rolls in at one end. Each station does one small job — cut, shape, tag, stamp — and passes the piece to the next station. At the far end, a finished toy rolls out. A Vision Transformer is exactly this kind of line, and every station is a brick you already own from this week.

Here is the whole line. An [[image||a grid of pixels — for us a small square picture, like 28 rows by 28 columns of brightness values]] rolls in. Station one **cuts** it into small square [[patches||fixed-size square tiles cut from the image — the ViT's version of "words," each patch is one token]] — the ViT's stand-in for words. Station two turns each patch into a vector of numbers (a [[patch embedding||one learned step that maps each flattened patch to a fixed-size vector the model works in]]) and tags each with *where it sat* (a [[positional embedding||a small learned vector added to each patch so the model knows the patch's position in the grid]]). Station three slips in one extra **summary slot** (the [[class token||one extra learned vector added to the front of the patch sequence; its final state is the model's summary of the whole image]]). Station four is the big one: a stack of identical [[encoder blocks||the repeating transformer layer — attention plus a small feed-forward net — reused unchanged from text transformers]] — borrowed whole from the text transformer you may have met before. Station five is a tiny [[MLP head||a small feed-forward layer that reads the summary slot and outputs one score per class]] that reads the summary slot and outputs a score for each class. Out rolls the guess.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A factory assembly line for a Vision Transformer. An image rolls in at the left onto a conveyor belt. Station 1 labelled 'cut into patches'. Station 2 labelled 'embed + tag position'. Station 3 labelled 'add summary slot (class token)'. Station 4 labelled 'encoder blocks (stack)'. Station 5 labelled 'MLP head'. At the right end a label 'class guess: cat' rolls off. A caption reads: each station does one small job and passes the piece along.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The ViT assembly line: image in → five stations → class guess out</text>
<line x1="24" y1="150" x2="496" y2="150" stroke="#B8AEA2" stroke-width="3"/>
<g fill="#9A938A"><circle cx="60" cy="158" r="6"/><circle cx="160" cy="158" r="6"/><circle cx="260" cy="158" r="6"/><circle cx="360" cy="158" r="6"/><circle cx="460" cy="158" r="6"/></g>
<g transform="translate(28,44)"><rect x="0" y="0" width="60" height="52" rx="6" fill="#EDE9E0" stroke="#3A342E"/><text x="30" y="22" fill="#3A342E" font-size="8">image</text><text x="30" y="38" fill="#6B645E" font-size="7">pixels</text></g>
<text x="94" y="74" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(104,44)"><rect x="0" y="0" width="66" height="52" rx="6" fill="#F4F1FB" stroke="#5E5191"/><text x="33" y="20" fill="#5E5191" font-size="8">1 cut into</text><text x="33" y="34" fill="#5E5191" font-size="8">patches</text></g>
<text x="176" y="74" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(186,44)"><rect x="0" y="0" width="66" height="52" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="33" y="20" fill="#1a5c38" font-size="8">2 embed +</text><text x="33" y="34" fill="#1a5c38" font-size="8">tag position</text></g>
<text x="258" y="74" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(268,44)"><rect x="0" y="0" width="66" height="52" rx="6" fill="#FDF3E7" stroke="#C99A12"/><text x="33" y="20" fill="#8A6D3B" font-size="8">3 add</text><text x="33" y="34" fill="#8A6D3B" font-size="8">summary slot</text></g>
<text x="340" y="74" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(350,44)"><rect x="0" y="0" width="66" height="52" rx="6" fill="#F4F1FB" stroke="#5E5191"/><text x="33" y="20" fill="#5E5191" font-size="8">4 encoder</text><text x="33" y="34" fill="#5E5191" font-size="8">blocks ×N</text></g>
<text x="422" y="74" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(432,44)"><rect x="0" y="0" width="60" height="52" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="30" y="20" fill="#1a5c38" font-size="8">5 MLP</text><text x="30" y="34" fill="#1a5c38" font-size="8">head</text></g>
<g transform="translate(360,168)"><rect x="0" y="0" width="132" height="24" rx="5" fill="#FDF3E7" stroke="#C99A12"/><text x="66" y="16" fill="#8A6D3B" font-size="8">class guess: "cat"</text></g>
<text x="150" y="188" fill="#9A938A" font-size="9">each station does one small job, then passes the piece to the next</text>
</g></svg>
%%%

**What the assembly-line picture gets right:** each station does one job and hands the piece forward — you can understand any one station on its own, then see how they connect. **Where it breaks down:** on a real factory line the stations are fixed, but a ViT *learns* — during training, almost every station's numbers slowly change so the whole line gets better at guessing.

#### Why "vision *transformer*"? A short bit of history
For years, the go-to tool for images was a different machine, the [[CNN||convolutional neural network — the older image model that scans small windows across the picture, built to assume nearby pixels matter most]]. A CNN has a built-in belief baked into its bricks: *nearby pixels belong together* — a helpful head start called [[inductive bias||a built-in assumption a model is born with, before seeing any data — for a CNN, that nearby pixels are related]]. The ViT makes a bold move: it *throws that belief away* and just treats the image like a sequence of patch-"words," exactly like a text transformer treats a sentence. That's why the same encoder bricks work on both. It's a beautiful idea — but it comes with a cost we'll meet later (a ViT has no built-in "nearby matters" head start, so it needs more data to learn that on its own).

!!! c-info 💡
<b>The one sentence to hold onto:</b> a Vision Transformer turns a picture into a short "sentence" of patch-words, tags each with its position, adds a summary slot, runs the same transformer <b>encoder blocks</b> a text model uses, and reads the answer off the summary slot. Everything after this is just <i>assembling</i> those stations, one at a time.
!!!

You've seen the whole line. Now let's walk to station one and pick up the first brick: cutting the image into patches. That's the next concept.

@@@ concept id=c2 tag="Cut into patches" title="Patchify: turn a picture into a row of tiles" gotit="Got patchify"
Picture a **chocolate bar scored into little squares.** The whole bar is your image. You snap it along the lines into equal squares, then lay those squares out in a row on the table, left to right, top to bottom. That row of squares is what the model reads. Cutting an image into patches is exactly this snap-and-lay-out: chop the picture into equal square tiles, then line the tiles up in order.

Here's the idea in plain words. The image is a grid of pixels — say 28 tall by 28 wide. We pick a patch size, say 7 by 7. We slide a 7×7 window across with no overlap and no gaps, and each window becomes one **patch**. Then we *flatten* each patch — unroll its little square of pixels into one long list of numbers (a vector) — so a patch stops being a 2-D tile and becomes a single row, ready to be treated like a word. Do this for every tile and you get a *row of patch-vectors*: the image's "sentence."

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A chocolate-bar analogy for patchify. On the left, a 28 by 28 image drawn as a grid, with thick lines dividing it into a 4 by 4 arrangement of 7 by 7 patches, so 16 patches total. An arrow points right to a single row of 16 small squares laid out in order, labelled 'patches laid in a row, top-left to bottom-right'. Below, one patch is shown unrolling into a long thin strip labelled 'flatten: 7x7 = 49 numbers'. A caption reads: number of patches = (28/7) times (28/7) = 16.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Patchify: snap the image into equal tiles, lay them in a row</text>
<text x="86" y="40" fill="#5E5191">image 28×28</text>
<g transform="translate(30,48)"><rect x="0" y="0" width="112" height="112" fill="#F4F1FB" stroke="#5E5191" stroke-width="1"/>
<g stroke="#5E5191" stroke-width="2"><line x1="28" y1="0" x2="28" y2="112"/><line x1="56" y1="0" x2="56" y2="112"/><line x1="84" y1="0" x2="84" y2="112"/><line x1="0" y1="28" x2="112" y2="28"/><line x1="0" y1="56" x2="112" y2="56"/><line x1="0" y1="84" x2="112" y2="84"/></g>
<text x="14" y="20" fill="#8579b8" font-size="8">1</text><text x="42" y="20" fill="#8579b8" font-size="8">2</text><text x="98" y="106" fill="#8579b8" font-size="8">16</text></g>
<path d="M150 104 h26" stroke="#9A5A12" stroke-width="2" marker-end="url(#p2a)"/>
<defs><marker id="p2a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#9A5A12"/></marker></defs>
<text x="380" y="40" fill="#276b45">16 patches laid in a row (order matters)</text>
<g transform="translate(184,92)" fill="#EAF5EE" stroke="#2D8B55">
<rect x="0" y="0" width="18" height="24" rx="2"/><rect x="20" y="0" width="18" height="24" rx="2"/><rect x="40" y="0" width="18" height="24" rx="2"/><rect x="60" y="0" width="18" height="24" rx="2"/><rect x="80" y="0" width="18" height="24" rx="2"/><rect x="100" y="0" width="18" height="24" rx="2"/><rect x="120" y="0" width="18" height="24" rx="2"/><rect x="140" y="0" width="18" height="24" rx="2"/><rect x="160" y="0" width="18" height="24" rx="2"/><rect x="180" y="0" width="18" height="24" rx="2"/><rect x="200" y="0" width="18" height="24" rx="2"/><rect x="220" y="0" width="18" height="24" rx="2"/><rect x="240" y="0" width="18" height="24" rx="2"/><rect x="260" y="0" width="18" height="24" rx="2"/><rect x="280" y="0" width="18" height="24" rx="2"/><rect x="300" y="0" width="18" height="24" rx="2"/></g>
<text x="184" y="150" fill="#2D8B55" font-size="8">one 7×7 patch</text>
<g transform="translate(240,140)"><rect x="0" y="0" width="240" height="16" rx="3" fill="#FDF3E7" stroke="#C99A12"/><text x="120" y="12" fill="#8A6D3B" font-size="8">flatten → 7×7 = 49 numbers in a row</text></g>
<text x="260" y="198" fill="#9A938A" font-size="9">number of patches = (28/7) × (28/7) = 4 × 4 = 16</text>
</g></svg>
%%%

**What the chocolate-bar picture gets right:** you snap into equal squares along fixed lines with no leftovers, then lay them in a set order — just like non-overlapping patches read in a fixed sequence. **Where it breaks down:** you can eat the chocolate squares in any order, but patch *order matters* to the model — shuffle the row and you've scrambled the picture (we'll fix that with position tags in the next concept).

#### The one number to remember: how many patches?
There's a single rule for the patch count, and it falls right out of the picture. If the image is `H` tall and `W` wide and each patch is `P` on a side, then you get `(H/P)` rows of patches and `(W/P)` columns, so:

%%% formula
expr: number of patches = (H / P) × (W / P)
note: for a 28×28 image with 7×7 patches: (28/7) × (28/7) = 4 × 4 = 16 patches
%%%

That's the *only* arithmetic in this concept. Every patch has the same size, so this is just counting tiles.

#### The trap: a scrambled patch (the reshape that silently ruins your image)
Here's the puzzle that snags almost everyone. Cutting patches is done with a `reshape` — you rearrange the pixel numbers into the shape "patches × pixels-per-patch." But `reshape` will happily rearrange numbers in the *wrong order* if you get the axes mixed up. It won't crash. The shapes will even look right. But the pixels inside each patch will be scrambled, and your beautiful image quietly turns to noise — the model then learns from garbage. The fix is a habit, not a formula: **print the shape at every step, and check one patch by eye** before moving on.

!!! c-warn ⚠️
<b>Failure mode (silent):</b> flattening patches with the reshape axes in the wrong order — the tensor's <i>shape</i> comes out right, but the pixels inside each patch are scrambled, so the "image" is now noise and the model can never learn it. <b>Cause:</b> <code>reshape</code> reorders numbers by raw memory order; mixing up the height/width/patch axes silently shuffles pixels. <b>Remedy:</b> reshape <i>deliberately</i> and <b>print the shape after every step</b> (shape discipline); reconstruct one patch and look at it, or use a named-axis rearrange (einops-style) so the axes can't get crossed.

Watch the shape march below — that print-at-every-step habit is what catches a scramble before it costs you a training run.
!!!

%%% demo id=patchify label="watch the shapes as an image becomes a row of patches"
code: import jax.numpy as jnp
code: img = jnp.ones((28, 28))                 # one grayscale image: 28 tall, 28 wide
code: P = 7                                     # patch size: 7×7
code: # cut into a grid of patches, then flatten each — PRINT the shape each step
code: patches = img.reshape(28 // P, P, 28 // P, P)   # (4, 7, 4, 7): rows, in-row, cols, in-col
code: print("after split :", patches.shape)
code: patches = patches.transpose(0, 2, 1, 3)         # bring the two patch axes together
code: print("after reorder:", patches.shape)
code: patches = patches.reshape(-1, P * P)            # (16, 49): 16 patches, 49 pixels each
code: print("final row    :", patches.shape, "→ 16 patch-vectors of length 49")
out: after split : (4, 7, 4, 7)
     after reorder: (4, 4, 7, 7)
     final row    : (16, 49) → 16 patch-vectors of length 49
take: <b>16 patches, 49 numbers each — and we could SEE every shape on the way.</b> The picture (28×28) became a row of 16 patch-vectors, matching <code>(H/P)×(W/P) = 16</code>. Printing the shape at each line is the whole defense against a silent scramble: if any line prints an unexpected shape, you catch it here — not after a wasted training run.
%%%

You've turned a picture into a row of patch-vectors. But those are just raw pixels, and the model has no idea which tile came from where. Two more bricks fix both problems at once: a learned embedding, a position tag, and a summary slot — that's the next concept.

@@@ concept id=c3 tag="Embed, tag, summarize" title="Patch embedding, position tags, and the summary slot" gotit="Got the three tags"
Three small bricks snap on here, and one everyday picture ties them together: **name tags and a folder at a group meeting.** Imagine strangers arriving at a meeting. First, each person gets translated into a common language everyone in the room understands — that's the **patch embedding**. Second, each gets a sticker saying *where they were sitting* — seat 1, seat 2 — so the room isn't just a jumble of people; that's the **positional embedding**. Third, someone drops one empty **folder** on the table whose only job is to collect notes from everyone and end up as the meeting summary — that's the **class token**. Three little bricks: translate, tag the seat, add the summary folder.

Let's take them one at a time.

- **Patch embedding — one shared translator.** Each patch is still a raw list of pixel numbers (length 49 in our example). We pass every patch through *one* learned linear step — a single `Dense` layer — that maps it to the model's working size `d` (say 32). The same translator is used for *every* patch, so it learns one good way to read any tile. This turns "49 raw pixels" into "a 32-number meaning vector."
- **Positional embedding — a seat sticker.** After translating, we *add* a small learned vector to each patch that depends only on *its position in the row* — position 1 gets one sticker, position 2 gets another. Now identical-looking patches in different spots are no longer identical: the model can tell top-left from bottom-right.
- **Class token — the summary folder.** We prepend one extra learned vector to the front of the row. It carries no pixels; it's an empty folder. As the encoder blocks run, this slot gathers information from every patch, and at the very end we read *only this slot* to make the class guess.

%%% svg
<svg viewBox="0 0 520 230" role="img" aria-label="A meeting analogy for the three embedding bricks. A row of raw patch vectors on the left passes through a single shared box labelled 'patch embedding: one translator for all', producing meaning vectors of size d. Then small colored 'seat sticker' vectors labelled 'positional embedding' are added on top of each. An extra folder-shaped slot labelled 'class token (summary folder)' is placed at the front of the row. A caption reads: translate every patch the same way, tag each with its seat, prepend one summary slot.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Three bricks: translate every patch · tag its seat · add a summary folder</text>
<text x="70" y="40" fill="#9A938A" font-size="8">raw patches (49 pixels each)</text>
<g transform="translate(24,48)" fill="#EDE9E0" stroke="#9a8f80"><rect x="0" y="0" width="22" height="22" rx="2"/><rect x="26" y="0" width="22" height="22" rx="2"/><rect x="52" y="0" width="22" height="22" rx="2"/><rect x="78" y="0" width="22" height="22" rx="2"/></g>
<path d="M126 59 h20" stroke="#5E5191" stroke-width="2" marker-end="url(#p3a)"/>
<defs><marker id="p3a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5E5191"/></marker></defs>
<g transform="translate(150,44)"><rect x="0" y="0" width="70" height="32" rx="6" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="35" y="15" fill="#5E5191" font-size="8">patch embed</text><text x="35" y="27" fill="#5E5191" font-size="7">one Dense, shared</text></g>
<path d="M224 59 h20" stroke="#2D8B55" stroke-width="2" marker-end="url(#p3b)"/>
<defs><marker id="p3b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<text x="360" y="40" fill="#276b45" font-size="8">meaning vectors (size d), now with seat stickers + summary folder</text>
<g transform="translate(250,48)"><rect x="0" y="0" width="22" height="22" rx="2" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="11" y="15" fill="#8A6D3B" font-size="8">S</text></g>
<g transform="translate(280,48)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="22" height="22" rx="2"/><rect x="30" y="0" width="22" height="22" rx="2"/><rect x="60" y="0" width="22" height="22" rx="2"/><rect x="90" y="0" width="22" height="22" rx="2"/></g>
<g transform="translate(280,74)" fill="#8579b8" stroke="#5E5191"><rect x="0" y="0" width="22" height="8" rx="2"/><rect x="30" y="0" width="22" height="8" rx="2"/><rect x="60" y="0" width="22" height="8" rx="2"/><rect x="90" y="0" width="22" height="8" rx="2"/></g>
<text x="326" y="98" fill="#5E5191" font-size="7">+ seat stickers (positional)</text>
<text x="261" y="98" fill="#8A6D3B" font-size="7">↑ folder</text>
<text x="150" y="150" fill="#9A5A12" font-size="9">S = the class token: one empty summary folder placed at the FRONT</text>
<text x="150" y="168" fill="#9A938A" font-size="9">every patch goes through the SAME translator; each gets its own seat tag</text>
<text x="150" y="200" fill="#276b45" font-size="9">at the end, we read only the folder (S) to make the class guess</text>
</g></svg>
%%%

**What the meeting picture gets right:** one translator makes everyone speak the same language, seat stickers keep track of who sat where, and a single folder ends up holding the summary — three clean, separate jobs. **Where it breaks down:** in a real meeting the summary folder is filled by a person, but here the class token fills itself automatically through attention inside the encoder blocks — nobody writes in it by hand.

#### The puzzle that makes positional embeddings *necessary*
Here's a surprising fact that makes the seat stickers non-optional. The encoder's attention step is [[permutation-invariant||it treats a sequence and any reshuffle of that sequence the same way — it has no built-in sense of order]]: it looks at all patches at once and has *no built-in sense of order*. If you shuffled the patch row, attention alone would produce the *same* result — it literally cannot tell a cat from the same tiles scrambled into noise. That's the puzzle. The fix is exactly the seat sticker: by *adding* a unique position vector to each patch *before* the blocks, two identical patches in different seats become different inputs, so order finally matters. Watch the before→after: same patches, but position tags break the tie.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A before-and-after figure showing why positional embeddings are needed. On the left, labelled 'WITHOUT position tags', two rows of the same four patches in different orders both collapse to the same output, with an equals sign between them and a red note 'attention can't tell them apart'. On the right, labelled 'WITH position tags', the same two rows now have small colored seat stickers added, and a not-equals sign between them with a green note 'now order matters'.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Why seat stickers matter: position tags break the tie</text>
<text x="130" y="40" fill="#C93B3B">WITHOUT position tags</text>
<g transform="translate(40,50)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="18" height="18" rx="2"/><rect x="22" y="0" width="18" height="18" rx="2"/><rect x="44" y="0" width="18" height="18" rx="2"/><rect x="66" y="0" width="18" height="18" rx="2"/></g>
<g transform="translate(150,50)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="18" height="18" rx="2"/><rect x="22" y="0" width="18" height="18" rx="2"/><rect x="44" y="0" width="18" height="18" rx="2"/><rect x="66" y="0" width="18" height="18" rx="2"/></g>
<text x="131" y="64" fill="#C93B3B" font-size="14">=</text>
<text x="130" y="94" fill="#C93B3B" font-size="8">shuffled, but attention gives the SAME output</text>
<line x1="260" y1="30" x2="260" y2="120" stroke="#B8AEA2" stroke-dasharray="4 3"/>
<text x="390" y="40" fill="#276b45">WITH position tags</text>
<g transform="translate(300,50)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="18" height="18" rx="2"/><rect x="22" y="0" width="18" height="18" rx="2"/><rect x="44" y="0" width="18" height="18" rx="2"/><rect x="66" y="0" width="18" height="18" rx="2"/></g>
<g transform="translate(300,72)" fill="#8579b8" stroke="#5E5191"><rect x="0" y="0" width="18" height="6"/><rect x="22" y="0" width="18" height="6"/><rect x="44" y="0" width="18" height="6"/><rect x="66" y="0" width="18" height="6"/></g>
<g transform="translate(410,50)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="18" height="18" rx="2"/><rect x="22" y="0" width="18" height="18" rx="2"/><rect x="44" y="0" width="18" height="18" rx="2"/><rect x="66" y="0" width="18" height="18" rx="2"/></g>
<g transform="translate(410,72)" fill="#8579b8" stroke="#5E5191"><rect x="0" y="0" width="18" height="6"/><rect x="22" y="0" width="18" height="6"/><rect x="44" y="0" width="18" height="6"/><rect x="66" y="0" width="18" height="6"/></g>
<text x="391" y="64" fill="#276b45" font-size="14">≠</text>
<text x="390" y="104" fill="#276b45" font-size="8">seat stickers added → order now changes the output</text>
<text x="260" y="150" fill="#9A5A12" font-size="9">positional embeddings are ADDED to patch embeddings, before the blocks</text>
<text x="260" y="172" fill="#9A938A" font-size="9">without them, a shuffled image looks identical to the model — a cat = noise</text>
</g></svg>
%%%

!!! c-info 💡
<b>The tie-breaker, in one line:</b> attention has no sense of order, so we <b>add</b> a learned position vector to each patch before the blocks — that's the positional embedding, and it's what lets the model know top-left from bottom-right.
!!!

Three tags are on. The row is now a sequence of meaning-vectors, each tagged with its seat, plus a summary folder at the front — ready for the heart of the model. That heart is a stack of encoder blocks, and it's the same block a text transformer uses. That's the next concept.

@@@ concept id=c4 tag="The engine + the head" title="Encoder blocks (borrowed whole) and the tiny MLP head" gotit="Got the body"
Picture a **conveyor of identical polishing stations.** A rough stone goes in. It passes the *same* polishing station over and over — each pass makes it a little smoother and clearer, until a dull rock comes out shining. The heart of a ViT is exactly this: the row of patch-vectors passes through a stack of *identical* refining stations called **encoder blocks**, each one letting the patches share information and sharpen their meaning, until the summary folder is polished enough to read off a class.

The wonderful thing — and the whole reason it's called a "transformer" — is that this block is *not new*. It's the **exact same encoder block a text transformer uses**, snapped in unchanged. You already own this brick from the transformer modules; here you just feed it patches instead of words. Inside each block are two sub-steps you've met before (we won't re-derive them today — they're borrowed as finished bricks):

- **An attention sub-layer** — lets every patch look at every other patch and mix in what's relevant. (How attention's queries/keys/values work is a whole topic of its own; here it's a black-box brick.)
- **A small feed-forward (MLP) sub-layer** — lets each patch think on its own for a moment.

Each sub-step is wrapped with a *skip connection* and *normalization* — two more finished bricks that keep deep stacks trainable. Stack `N` of these identical blocks (say 4 or 6) and you have the ViT's engine.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A polishing-conveyor analogy for encoder blocks. A rough stone enters on the left and passes through three identical stations each labelled 'encoder block (attention + MLP)', getting visibly smoother, until a polished stone exits on the right into a small box labelled 'MLP head → class score'. A caption reads: the same block, repeated N times, then a tiny head reads the summary slot.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Encoder blocks: the same refining station, stacked N times</text>
<line x1="24" y1="120" x2="470" y2="120" stroke="#B8AEA2" stroke-width="3"/>
<g transform="translate(24,44)"><circle cx="26" cy="30" r="22" fill="#EDE9E0" stroke="#6B645E" stroke-width="2"/><text x="26" y="76" fill="#6B645E" font-size="8">rough (patches)</text></g>
<g transform="translate(96,44)"><rect x="0" y="0" width="82" height="60" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="41" y="24" fill="#5E5191" font-size="8">encoder block</text><text x="41" y="40" fill="#5E5191" font-size="7">attention + MLP</text><text x="41" y="52" fill="#8579b8" font-size="7">×1</text></g>
<text x="184" y="78" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(196,44)"><rect x="0" y="0" width="82" height="60" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="41" y="24" fill="#5E5191" font-size="8">encoder block</text><text x="41" y="40" fill="#5E5191" font-size="7">(identical)</text><text x="41" y="52" fill="#8579b8" font-size="7">×2</text></g>
<text x="284" y="78" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(296,44)"><rect x="0" y="0" width="82" height="60" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="41" y="24" fill="#5E5191" font-size="8">encoder block</text><text x="41" y="40" fill="#5E5191" font-size="7">(identical)</text><text x="41" y="52" fill="#8579b8" font-size="7">×N</text></g>
<text x="384" y="78" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(396,52)"><circle cx="20" cy="22" r="18" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="20" y="60" fill="#276b45" font-size="8">polished</text></g>
<g transform="translate(300,150)"><rect x="0" y="0" width="130" height="30" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="65" y="14" fill="#1a5c38" font-size="8">MLP head</text><text x="65" y="26" fill="#276b45" font-size="7">reads summary → class score</text></g>
<text x="150" y="196" fill="#9A938A" font-size="9">same block reused N times, then a tiny head reads only the summary folder</text>
</g></svg>
%%%

**What the polishing picture gets right:** the same station is applied again and again, and the piece gets steadily better each pass — just like identical encoder blocks stacked in a row. **Where it breaks down:** a polishing wheel does the same physical thing every pass, but each encoder block has its *own* learned numbers, so block 1 and block 4 refine in different, learned ways.

#### The last brick: a tiny head that reads the summary
After the last block, the summary folder (class token) is full — it has quietly gathered a read on the whole image. The final brick is the **MLP head**: a small feed-forward layer that takes *just that one summary vector* and outputs a list of scores, one per class. The biggest score is the model's guess. Notice how little work the head does — the heavy lifting happened in the blocks; the head just reads the answer off the folder.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="The MLP head reading the class token. On the left, the final row of patch states with the summary folder highlighted at the front. An arrow labelled 'take ONLY the summary slot' points to a small box labelled 'MLP head'. Out of it comes a bar chart of three class scores labelled cat, dog, bird, with 'cat' the tallest and marked as the guess.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The head reads ONLY the summary folder → one score per class</text>
<g transform="translate(24,44)"><rect x="0" y="0" width="24" height="24" rx="3" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="12" y="16" fill="#8A6D3B" font-size="9">S</text></g>
<g transform="translate(52,44)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="24" height="24" rx="3"/><rect x="28" y="0" width="24" height="24" rx="3"/><rect x="56" y="0" width="24" height="24" rx="3"/><rect x="84" y="0" width="24" height="24" rx="3"/></g>
<text x="42" y="86" fill="#8A6D3B" font-size="8">summary folder (S)</text>
<path d="M52 56 C 90 90, 150 90, 180 70" stroke="#C99A12" stroke-width="2" fill="none" marker-end="url(#p4a)"/>
<defs><marker id="p4a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C99A12"/></marker></defs>
<text x="150" y="110" fill="#9A5A12" font-size="7">take ONLY the summary slot</text>
<g transform="translate(184,48)"><rect x="0" y="0" width="70" height="36" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="35" y="22" fill="#1a5c38" font-size="8">MLP head</text></g>
<path d="M256 66 h16" stroke="#2D8B55" stroke-width="2" marker-end="url(#p4b)"/>
<defs><marker id="p4b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<g transform="translate(300,40)"><line x1="0" y1="60" x2="150" y2="60" stroke="#B8AEA2"/><rect x="14" y="14" width="30" height="46" fill="#8579b8"/><rect x="60" y="40" width="30" height="20" fill="#c9c1e8"/><rect x="106" y="46" width="30" height="14" fill="#c9c1e8"/><text x="29" y="74" fill="#6B645E" font-size="8">cat</text><text x="75" y="74" fill="#6B645E" font-size="8">dog</text><text x="121" y="74" fill="#6B645E" font-size="8">bird</text><text x="29" y="10" fill="#5E5191" font-size="7">guess ✓</text></g>
<text x="150" y="138" fill="#9A938A" font-size="9">the biggest score is the model's class guess</text>
</g></svg>
%%%

The model is now fully assembled on paper: patches → embed+tag+summary → encoder blocks → head → class scores. But so far it's just a *drawing*. To make it a thing you can run and train, we need JAX. The next few concepts pick up the JAX bricks that build these numbers, run the model fast, and teach it. First: where do all these learned numbers *live*? That's the next concept.

@@@ concept id=c5 tag="Numbers live outside" title="The JAX way: a pure function + a tree of numbers that never change in place" gotit="Got functional style"
Picture a **coffee machine and a jar of beans on the counter.** The machine knows the *steps* — grind, brew, pour — but keeps *no beans inside it*. The beans sit in a jar you hold, outside the machine, and you pour them in each time you brew. Same beans in, same coffee out. Building a model the JAX way is exactly this split: the model is the machine (just the steps), and all its learned numbers sit in a jar *you* hold, outside — you pour them in every time you run it.

This is the big JAX habit from Day 5, now applied to the whole ViT. Instead of a model-object that *hides* its own weights, JAX likes the model to be a plain **pure function**: `logits = forward(params, image)`. The word [[pure function||a function whose output depends only on its inputs, with no hidden memory and no side effects — same inputs always give the same output]] means it has no hidden memory: give it the same `params` and the same image, and it *always* returns the same answer. Nothing is stored inside; everything comes in through the front door.

And where do all those numbers live? In one tidy jar called the [[params||the single tree of all the model's learned numbers — patch-embed weights, position tags, class token, every block's weights, the head — held outside the model and passed in each run]]. In JAX this jar is a [[pytree||a nested container — dicts inside dicts inside lists — of plain arrays, which JAX can walk over all at once with one command]]: just nested dictionaries of arrays. Every learned number in the whole ViT — the patch-embed translator, the seat stickers, the summary folder, all `N` blocks, the head — sits as a leaf in this one tree.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A coffee-machine-and-beans analogy for functional JAX. On the left, a coffee machine labelled 'forward(): the steps only, no numbers inside'. On the right, a jar labelled 'params: a pytree of ALL learned numbers, held by you'. Inside the jar, a nested tree shows patch_embed, pos_embed, cls_token, block_0, block_1, head as leaves. An arrow from the jar into the machine labelled 'you pour the numbers in each run'. A caption reads: logits = forward(params, image) — a pure function.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The model is a pure function; every number lives in one params tree</text>
<text x="100" y="40" fill="#5E5191">forward() — the steps</text>
<g transform="translate(30,48)"><rect x="0" y="0" width="140" height="120" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="70" y="26" fill="#5E5191" font-size="9">forward(params, image)</text><line x1="16" y1="36" x2="124" y2="36" stroke="#8579b8" stroke-dasharray="3 3"/><text x="70" y="54" fill="#5E5191" font-size="8">patchify → embed</text><text x="70" y="72" fill="#5E5191" font-size="8">→ +position, +cls</text><text x="70" y="90" fill="#5E5191" font-size="8">→ blocks → head</text><text x="70" y="110" fill="#9A5A12" font-size="8">(no numbers stored here)</text></g>
<path d="M350 108 C 300 108, 260 100, 175 100" stroke="#2D8B55" stroke-width="2" fill="none" marker-end="url(#p5a)"/>
<defs><marker id="p5a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<text x="410" y="40" fill="#276b45">params (the jar)</text>
<g transform="translate(346,48)"><rect x="0" y="0" width="150" height="120" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><g fill="#1a5c38" font-size="8" text-anchor="start"><text x="12" y="24">params:</text><text x="24" y="40">patch_embed …</text><text x="24" y="56">pos_embed …</text><text x="24" y="72">cls_token …</text><text x="24" y="88">block_0, block_1 …</text><text x="24" y="104">head …</text></g></g>
<text x="260" y="196" fill="#9A5A12" font-size="9">you pour the numbers (params) into the steps (forward) each run</text>
<text x="260" y="212" fill="#9A938A" font-size="9">logits = forward(params, image) — same in → same out (a pure function)</text>
</g></svg>
%%%

**What the coffee picture gets right:** the machine holds the steps and the jar holds the beans, kept separate, and you pour beans in each brew — just as the model holds the steps and the params tree holds the numbers, poured in each run. **Where it breaks down:** a coffee jar just stores beans, but the params pytree can be walked, mapped, saved, and handed to `grad` — and its beans (numbers) *improve* every training step, which no real jar does.

#### The rule under it all: a JAX array never changes in place
Here's the quiet rule from Day 1 that makes this whole design work — and it's the one bit of the week we lean on hardest today. A JAX array is [[immutable||once made, a JAX array can never be edited in place; any "change" builds a brand-new array and leaves the old one untouched]]: once you make it, you can *never* edit it in place. In plain NumPy you might write `arr[0] = 5` to poke a new value into a slot. In JAX that *fails* — the array is carved in stone. To "change" a number you don't edit the old array; you build a **new** array with `arr.at[0].set(5)`, and the old one stays exactly as it was.

That sounds like a restriction, but it's the very thing that keeps the training loop honest. Every training step does *not* reach into the params jar and scribble new values over the old. Instead it **returns a whole new params tree** — the old numbers untouched, a fresh tree with the improved numbers handed back. Same idea, one level up from a single array: nothing mutates; you always get a new copy. That is exactly why `jit`, `vmap`, and `grad` can trust your function — there is no hidden slot quietly changing under their feet.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A before-and-after figure of immutability. On the left, labelled 'edit in place (NumPy) — NOT allowed in JAX', an array box has an arrow poking a new value 5 into slot 0, marked with a red X and 'arr[0]=5 fails'. On the right, labelled 'functional update (JAX)', the original array box stays unchanged with a green check, and a fresh new array box is built beside it with slot 0 now 5, labelled 'arr.at[0].set(5) → NEW array; old one untouched'. A caption reads: nothing changes in place; you always get a new copy — that is what training returns each step.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Immutable arrays: you don't edit — you build a NEW copy</text>
<text x="128" y="40" fill="#C93B3B">edit in place (NumPy) — banned in JAX</text>
<g transform="translate(52,52)" fill="#FBEDED" stroke="#C93B3B"><rect x="0" y="0" width="30" height="26" rx="3"/><rect x="32" y="0" width="30" height="26" rx="3"/><rect x="64" y="0" width="30" height="26" rx="3"/><rect x="96" y="0" width="30" height="26" rx="3"/></g>
<g fill="#a12b2b" font-size="8"><text x="67" y="69">1</text><text x="99" y="69">2</text><text x="131" y="69">3</text><text x="163" y="69">4</text></g>
<path d="M67 96 v-14" stroke="#C93B3B" stroke-width="2" marker-end="url(#p5b)"/>
<defs><marker id="p5b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C93B3B"/></marker></defs>
<text x="67" y="112" fill="#C93B3B" font-size="9">arr[0]=5</text>
<text x="130" y="112" fill="#C93B3B" font-size="16">✗</text>
<text x="128" y="132" fill="#C93B3B" font-size="8">JAX refuses — arrays are carved in stone</text>
<line x1="260" y1="30" x2="260" y2="150" stroke="#B8AEA2" stroke-dasharray="4 3"/>
<text x="392" y="40" fill="#276b45">functional update (JAX)</text>
<g transform="translate(300,52)" fill="#EDE9E0" stroke="#9a8f80"><rect x="0" y="0" width="30" height="26" rx="3"/><rect x="32" y="0" width="30" height="26" rx="3"/><rect x="64" y="0" width="30" height="26" rx="3"/><rect x="96" y="0" width="30" height="26" rx="3"/></g>
<g fill="#6B645E" font-size="8"><text x="315" y="69">1</text><text x="347" y="69">2</text><text x="379" y="69">3</text><text x="411" y="69">4</text></g>
<text x="450" y="69" fill="#276b45" font-size="12">✓</text>
<text x="360" y="88" fill="#9A938A" font-size="7">old array: untouched</text>
<g transform="translate(300,100)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="30" height="26" rx="3"/><rect x="32" y="0" width="30" height="26" rx="3"/><rect x="64" y="0" width="30" height="26" rx="3"/><rect x="96" y="0" width="30" height="26" rx="3"/></g>
<g fill="#1a5c38" font-size="8"><text x="315" y="117">5</text><text x="347" y="117">2</text><text x="379" y="117">3</text><text x="411" y="117">4</text></g>
<text x="392" y="140" fill="#276b45" font-size="7">arr.at[0].set(5) → a NEW array</text>
<text x="260" y="176" fill="#9A5A12" font-size="9">nothing changes in place — each train step RETURNS a new params tree, same idea</text>
</g></svg>
%%%

%%% demo id=immutable label="try to edit a JAX array in place (it fails), then update it the functional way"
code: import jax.numpy as jnp
code: arr = jnp.array([1, 2, 3, 4])
code: # 1) try to edit in place, the NumPy way — JAX arrays are IMMUTABLE
code: try:
code:     arr[0] = 5                         # poke a new value into slot 0
code: except TypeError as e:
code:     print("in-place edit failed:", str(e)[:52])
code: # 2) the functional update: build a NEW array; the old one is untouched
code: new_arr = arr.at[0].set(5)
code: print("old array (unchanged):", arr)
code: print("new array (a copy)   :", new_arr)
code: print("same object?", arr is new_arr)
out: in-place edit failed: JAX arrays are immutable and do not support in-place
     old array (unchanged): [1 2 3 4]
     new array (a copy)   : [5 2 3 4]
     same object? False
take: <b>You cannot poke a value into a JAX array — you build a new one.</b> The in-place edit <code>arr[0]=5</code> raised an error, and <code>arr.at[0].set(5)</code> returned a fresh array while the original stayed <code>[1 2 3 4]</code>. This is the mechanism the whole training loop rides on: a step never scribbles over the old params — it <b>returns a new params tree</b>, old numbers untouched. That "always a new copy" promise is exactly what lets <code>jit</code>, <code>vmap</code>, and <code>grad</code> trust your code.
%%%

#### Why one tree? Because you can update every number with one command
Keeping all numbers in a single pytree has a lovely payoff: JAX can walk the *whole* tree at once. `jax.tree_util.tree_map(f, params)` applies `f` to *every* leaf array — so "make a new tree with the update added to every weight" is one line, no matter how deep the tree, and (thanks to immutability) it hands back a *fresh* tree rather than editing the old one. You don't loop over layers by hand. Let's build a toy params tree for a mini-ViT and see its shape.

%%% demo id=pytree label="build a toy params tree and walk it in one line"
code: import jax, jax.numpy as jnp
code: # a tiny stand-in for a ViT's params: nested dict of arrays (a pytree)
code: params = {
code:   "patch_embed": {"w": jnp.zeros((49, 32)), "b": jnp.zeros((32,))},  # translator
code:   "pos_embed":   jnp.zeros((17, 32)),        # 16 patches + 1 summary slot
code:   "cls_token":   jnp.zeros((1, 32)),         # the summary folder
code:   "head":        {"w": jnp.zeros((32, 3)),  "b": jnp.zeros((3,))},   # 3 classes
code: }
code: # walk the WHOLE tree at once — print every leaf's shape
code: shapes = jax.tree_util.tree_map(lambda a: a.shape, params)
code: print(shapes)
code: n_leaves = len(jax.tree_util.tree_leaves(params))
code: print("number of arrays (leaves) in the tree:", n_leaves)
out: {'cls_token': (1, 32), 'head': {'b': (3,), 'w': (32, 3)}, 'patch_embed': {'b': (32,), 'w': (49, 32)}, 'pos_embed': (17, 32)}
     number of arrays (leaves) in the tree: 6
take: <b>Every number lives in one tree — and one command touched them all.</b> The <code>tree_map</code> call reached every leaf array and reported its shape, without a single hand-written loop. That's why "update every weight" later is just one more <code>tree_map</code> that returns a fresh tree. Note <code>pos_embed</code> has 17 rows: 16 patches plus the 1 summary slot.
%%%

#### The trap: a hidden side-effect breaks the "pure" promise
The whole JAX toolbox — `jit`, `vmap`, `grad` — trusts that `forward` is *pure*. If your function secretly reaches out and reads a changing global, or mutates something, or prints as part of its "real" work, it's no longer pure — and JAX can compute a *wrong* answer without ever complaining. The fix is a discipline: pass **every** input in through the arguments, keep no hidden state, and don't rely on side effects for anything the output depends on.

!!! c-warn ⚠️
<b>Failure mode (silent, wrong answers):</b> a "forward" function that reads a mutable global, changes state, or depends on a side effect — it looks fine but under <code>jit</code>/<code>grad</code> JAX traces it once and can silently return stale or wrong results. <b>Cause:</b> JAX transforms assume purity; hidden state is invisible to the tracer. <b>Remedy:</b> write pure functions — <b>all</b> inputs (params, image, and any config) passed in explicitly, no hidden reads or writes, output a function of the inputs alone.
!!!

Your model is a pure function and its numbers live in a pytree that never changes in place. But those numbers have to *start* somewhere — they begin as random draws. And in JAX, randomness is never hidden. That's the next brick: explicit random keys.

@@@ concept id=c6 tag="Explicit dice" title="PRNGKeys: JAX randomness you hold and split" gotit="Got keys"
Picture handing out **fresh lottery tickets.** Each ticket, once scratched, gives one random result. The golden rule: *never scratch the same ticket twice* — a scratched ticket always shows the same number, so reusing it gives you the same "random" result again, which isn't random at all. If you need many independent draws, you tear off a fresh ticket for each. JAX randomness works exactly like this: a random **key** is one ticket, and you must tear it into fresh keys — never reuse one — for independent random draws.

This is the Day 2 habit, and it matters here because *all* of our ViT's numbers start as random draws — the patch translator, the position tags, the summary folder, every block. In most languages randomness comes from a hidden global "coin" the computer shakes for you. JAX refuses to hide it. Instead you hold an explicit [[PRNGKey||JAX's explicit "seed of randomness" — a small piece of data you pass into any random operation; the same key always gives the same random numbers]]: a little piece of data you pass into every random operation. Same key in → same random numbers out, every time. That makes your whole model *reproducible*: run it twice with the same starting key and you get the identical model.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A lottery-ticket analogy for PRNG keys. On the left a single ticket labelled 'key = PRNGKey(0)'. An arrow to jax.random.split shows it tearing into three fresh tickets labelled k1, k2, k3. Each fresh ticket feeds a different part of the model: k1 to patch_embed, k2 to blocks, k3 to head. A red note warns: reuse the SAME ticket twice and you get the SAME numbers — not independent.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">One key → split into fresh keys → one per random draw</text>
<g transform="translate(30,60)"><rect x="0" y="0" width="96" height="40" rx="6" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="48" y="18" fill="#8A6D3B" font-size="8">key</text><text x="48" y="32" fill="#9A5A12" font-size="8">PRNGKey(0)</text></g>
<path d="M130 80 h34" stroke="#5E5191" stroke-width="2" marker-end="url(#p6a)"/>
<defs><marker id="p6a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5E5191"/></marker></defs>
<text x="147" y="72" fill="#5E5191" font-size="7">split</text>
<g transform="translate(168,56)"><rect x="0" y="0" width="90" height="48" rx="6" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="45" y="20" fill="#5E5191" font-size="8">jax.random</text><text x="45" y="34" fill="#5E5191" font-size="8">.split(key, 3)</text></g>
<path d="M260 66 C 300 50, 320 46, 350 46" stroke="#2D8B55" stroke-width="1.5" fill="none" marker-end="url(#p6b)"/>
<path d="M260 80 C 300 80, 320 80, 350 80" stroke="#2D8B55" stroke-width="1.5" fill="none" marker-end="url(#p6b)"/>
<path d="M260 94 C 300 110, 320 114, 350 114" stroke="#2D8B55" stroke-width="1.5" fill="none" marker-end="url(#p6b)"/>
<defs><marker id="p6b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<g transform="translate(354,36)" fill="#EAF5EE" stroke="#2D8B55" font-size="8"><rect x="0" y="0" width="140" height="20" rx="3"/><text x="70" y="14" fill="#1a5c38">k1 → patch_embed</text><rect x="0" y="24" width="140" height="20" rx="3"/><text x="70" y="38" fill="#1a5c38">k2 → blocks</text><rect x="0" y="48" width="140" height="20" rx="3"/><text x="70" y="62" fill="#1a5c38">k3 → head</text></g>
<text x="260" y="150" fill="#C93B3B" font-size="9">reuse the SAME key twice → the SAME "random" numbers (not independent!)</text>
<text x="260" y="170" fill="#9A938A" font-size="9">same starting key → identical model every run (reproducible)</text>
</g></svg>
%%%

**What the lottery-ticket picture gets right:** a scratched ticket always shows the same number, so you must tear off fresh ones for independent draws — just as a key always gives the same numbers, so you split it for each independent random init. **Where it breaks down:** lottery tickets are truly random, but a JAX key is *deterministic* — the same key always gives the same numbers, which is a feature (reproducibility), not a flaw.

#### The trap: reusing a key gives identical "random" draws
Here's the classic beginner surprise. If you pass the *same* key to two random draws — say, to init the patch-embed weights *and* the block weights — they come out **identical**, not independent. Two parts of your model start as clones. The fix is `jax.random.split`, which tears one key into several fresh, independent keys. Watch it: reuse the same key and the two arrays match; split first and they differ.

%%% demo id=keys label="reuse a key (bad) vs split it (good)"
code: from jax import random
code: import jax.numpy as jnp
code: key = random.PRNGKey(0)
code: # BAD: same key twice → identical "random" draws
code: a = random.normal(key, (3,))
code: b = random.normal(key, (3,))
code: print("same key twice → identical?", bool(jnp.allclose(a, b)))
code: # GOOD: split into fresh keys → independent draws
code: k1, k2 = random.split(key)
code: c = random.normal(k1, (3,))
code: d = random.normal(k2, (3,))
code: print("split keys     → identical?", bool(jnp.allclose(c, d)))
out: same key twice → identical? True
     split keys     → identical? False
take: <b>Same ticket, same number; fresh tickets, fresh numbers.</b> Reusing <code>key</code> gave two <i>identical</i> arrays — two model parts would start as clones. After <code>random.split</code>, the two draws differ — genuinely independent. Rule: <b>split before each independent random use</b>. For a whole ViT you split once into as many keys as you have random pieces.
%%%

Your model's numbers can now start as *independent* random draws. Next: how do we run this pure function fast, and over a whole batch of images at once, without slow Python loops? Two bricks — `jit` and `vmap` — and one recompilation trap. That's the next concept.

@@@ concept id=c7 tag="Fast + batched" title="jit and vmap: compile once, run over a whole batch" gotit="Got jit + vmap"
Two bricks make our pure `forward` fast and batch-friendly, and each has a homely picture.

For speed, picture a **cook who reads the whole recipe once, plans it, then cooks fast forever after.** The first time is slow — they study every step. But once planned, every future dish flies. That's [[jit||"just-in-time" compile: jax.jit(f) traces f once on the first call, compiles a fast version, and reuses it for later calls with the same input shapes]]: `jax.jit(forward)` *traces* your function once on the first call — walks through it to see the whole plan — compiles a fast version, and reuses that compiled version on every later call. First call slow, all the rest fast.

For batching, picture a **stamp that prints one photo — now bolt a whole row of them onto a press so one pull stamps a hundred photos at once.** You designed the single stamp; the press just repeats it across the batch. That's [[vmap||"vectorized map": jax.vmap(f) takes a function written for ONE example and automatically makes it run over a whole batch, adding a batch axis, with no Python loop]]: you write `forward` for *one* image, and `jax.vmap(forward)` automatically runs it across a whole batch of images — no slow Python `for` loop. You think about one image; JAX handles the crowd.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two pictures. On the left, jit shown as a cook: a slow first pass labelled 'trace + compile (slow, once)' then many fast passes labelled 'reuse compiled (fast)'. On the right, vmap shown as a stamp press: a single stamp labelled 'forward: one image' bolted into a press that prints a whole row labelled 'vmap: whole batch at once, no loop'. A caption reads: jit compiles once and caches; vmap adds the batch axis for you.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">jit = compile once, reuse fast · vmap = one stamp → whole batch</text>
<text x="130" y="40" fill="#5E5191">jit (the planning cook)</text>
<g transform="translate(30,50)"><rect x="0" y="0" width="90" height="30" rx="5" fill="#FBEDED" stroke="#C93B3B"/><text x="45" y="14" fill="#a12b2b" font-size="7">1st call: trace</text><text x="45" y="25" fill="#a12b2b" font-size="7">+ compile (slow)</text></g>
<path d="M124 65 h18" stroke="#9A938A" stroke-width="1.5" marker-end="url(#p7a)"/>
<defs><marker id="p7a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#9A938A"/></marker></defs>
<g transform="translate(146,50)" fill="#EAF5EE" stroke="#2D8B55" font-size="7"><rect x="0" y="0" width="26" height="30" rx="4"/><text x="13" y="18" fill="#1a5c38">fast</text><rect x="30" y="0" width="26" height="30" rx="4"/><text x="43" y="18" fill="#1a5c38">fast</text><rect x="60" y="0" width="26" height="30" rx="4"/><text x="73" y="18" fill="#1a5c38">fast</text></g>
<text x="180" y="98" fill="#276b45" font-size="7">later calls: reuse (fast)</text>
<line x1="260" y1="30" x2="260" y2="130" stroke="#B8AEA2" stroke-dasharray="4 3"/>
<text x="390" y="40" fill="#276b45">vmap (the stamp press)</text>
<g transform="translate(300,50)"><rect x="0" y="0" width="52" height="30" rx="5" fill="#F4F1FB" stroke="#5E5191"/><text x="26" y="14" fill="#5E5191" font-size="7">forward</text><text x="26" y="26" fill="#5E5191" font-size="7">one image</text></g>
<path d="M356 65 h16" stroke="#5E5191" stroke-width="2" marker-end="url(#p7b)"/>
<defs><marker id="p7b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5E5191"/></marker></defs>
<g transform="translate(376,48)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="16" height="34" rx="2"/><rect x="20" y="0" width="16" height="34" rx="2"/><rect x="40" y="0" width="16" height="34" rx="2"/><rect x="60" y="0" width="16" height="34" rx="2"/><rect x="80" y="0" width="16" height="34" rx="2"/></g>
<text x="424" y="98" fill="#276b45" font-size="7">one pull → whole batch, no loop</text>
<text x="260" y="160" fill="#9A5A12" font-size="9">write forward for ONE image; vmap it for the batch; jit it for speed</text>
<text x="260" y="180" fill="#9A938A" font-size="9">jit caches the compiled version keyed by input SHAPE and dtype</text>
</g></svg>
%%%

**What these pictures get right:** the cook plans once then flies, and the press repeats one stamp across many — matching jit's compile-once-reuse and vmap's write-one-run-many. **Where they break down:** the cook re-plans if the *recipe* changes; jit re-compiles if the input *shape* changes — which is exactly the trap below.

#### The trap: silent recompilation blowup
Here's the puzzle. `jit` caches its compiled version keyed by the input **shape and dtype**. So if you feed it a *different* shape each time — batch of 32, then 31, then 30 — it *re-compiles* from scratch every single call. Compilation is the slow part, so your "fast" model becomes *slower* than no `jit` at all, and nothing errors. The fix has two moves: **keep shapes consistent** (pad batches to a fixed size so the shape never changes), and for a value that's a true constant but changes the code path (like "how many classes"), mark it static with `static_argnums` so a *change* in it triggers exactly one intended recompile, not an accidental one on every call.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A before-and-after figure of jit recompilation. On the left, labelled 'changing shapes (BAD)', three calls with batch sizes 32, 31, 30 each trigger a red 'recompile!' box, with a note 'slow every call'. On the right, labelled 'consistent shape (GOOD)', three calls all batch 32: the first is a slow compile, the next two reuse a green 'cached' box, with a note 'fast after the first'.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Keep the shape fixed, or jit recompiles on every call</text>
<text x="130" y="40" fill="#C93B3B">changing shapes (BAD)</text>
<g font-size="7">
<g transform="translate(28,50)"><rect x="0" y="0" width="60" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="30" y="15" fill="#5E5191">batch 32</text></g>
<g transform="translate(28,80)"><rect x="0" y="0" width="60" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="30" y="15" fill="#5E5191">batch 31</text></g>
<g transform="translate(28,110)"><rect x="0" y="0" width="60" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="30" y="15" fill="#5E5191">batch 30</text></g>
<g transform="translate(120,50)" fill="#FBEDED" stroke="#C93B3B"><rect x="0" y="0" width="80" height="24" rx="4"/><text x="40" y="15" fill="#a12b2b">recompile!</text></g>
<g transform="translate(120,80)" fill="#FBEDED" stroke="#C93B3B"><rect x="0" y="0" width="80" height="24" rx="4"/><text x="40" y="15" fill="#a12b2b">recompile!</text></g>
<g transform="translate(120,110)" fill="#FBEDED" stroke="#C93B3B"><rect x="0" y="0" width="80" height="24" rx="4"/><text x="40" y="15" fill="#a12b2b">recompile!</text></g>
</g>
<text x="130" y="152" fill="#C93B3B" font-size="8">slow on EVERY call</text>
<line x1="260" y1="30" x2="260" y2="150" stroke="#B8AEA2" stroke-dasharray="4 3"/>
<text x="390" y="40" fill="#276b45">consistent shape (GOOD)</text>
<g font-size="7">
<g transform="translate(300,50)"><rect x="0" y="0" width="60" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="30" y="15" fill="#5E5191">batch 32</text></g>
<g transform="translate(300,80)"><rect x="0" y="0" width="60" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="30" y="15" fill="#5E5191">batch 32</text></g>
<g transform="translate(300,110)"><rect x="0" y="0" width="60" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="30" y="15" fill="#5E5191">batch 32</text></g>
<g transform="translate(392,50)" fill="#FDF3E7" stroke="#C99A12"><rect x="0" y="0" width="90" height="24" rx="4"/><text x="45" y="15" fill="#8A6D3B">compile once</text></g>
<g transform="translate(392,80)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="90" height="24" rx="4"/><text x="45" y="15" fill="#1a5c38">cached (fast)</text></g>
<g transform="translate(392,110)" fill="#EAF5EE" stroke="#2D8B55"><rect x="0" y="0" width="90" height="24" rx="4"/><text x="45" y="15" fill="#1a5c38">cached (fast)</text></g>
</g>
<text x="390" y="152" fill="#276b45" font-size="8">fast after the first call</text>
</g></svg>
%%%

!!! c-warn ⚠️
<b>Failure mode (slow, no error):</b> feeding a <code>jit</code>ted function a <b>different input shape</b> (or a varying Python value that changes the traced code) on each call — JAX recompiles every time, so the model runs slower than un-jitted and nothing warns you. <b>Cause:</b> <code>jit</code> caches by input shape/dtype; a new shape is a cache miss → a fresh compile. <b>Remedy:</b> keep shapes fixed (pad batches to a constant size); mark true constants static with <code>jax.jit(f, static_argnums=...)</code> so only a genuine change in them recompiles.

<b>Optional (skippable):</b> a value marked <code>static_argnums</code> is baked into the compiled code, so each distinct static value gets its own compiled version — great for a fixed "num_classes", wasteful if the value changes often.
!!!

You can now run the model fast (`jit`) and over a batch (`vmap`). The last brick teaches it: get the gradient with `grad`, and assemble one training step that runs over a small pile of real images. That's the next concept.

@@@ concept id=c8 tag="Train it end to end" title="grad + one training step: train on real little images, watch loss fall and accuracy climb" gotit="Got the train step"
Picture **learning to throw darts with your eyes on the board.** You throw, see how far you missed and in which direction, and adjust your next throw a little that way. Miss high-left? Aim a touch low-right. Do that over and over and your throws creep toward the bullseye. Training the ViT is this exact loop: guess, measure how wrong, find *which way* to nudge every number, take a small step, repeat — over a whole pile of images, not just one.

The brick that finds "which way to nudge" is [[grad||jax.grad(loss_fn) returns a new function that computes the gradient — the direction and amount to change each parameter to lower the loss — with respect to the params]]. You write a loss function (a single "how wrong?" number — big when the guesses are bad, small when they're good), and `jax.grad(loss_fn)` hands you back the **gradient**: a pytree, shaped exactly like `params`, saying how to nudge *every* number to lower the loss. Crucially, `grad` only works because `forward` is **pure** — same non-negotiable rule from before. A hidden side effect would make the gradient meaningless.

One **training step** snaps four bricks together: (1) run `forward` (via `vmap`) over the whole batch of images to get logits, (2) measure the loss, (3) `grad` for the nudge direction, (4) take an Optax step to update the params — which, thanks to immutability, hands back a *new* params tree. Because everything is pure and the numbers live outside, the whole step is one function: boxes in, improved boxes out — and you wrap it in `jit` so it flies.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A darts analogy for one training step, drawn as a four-station loop. Station 1 'forward → logits'. Station 2 'loss: how wrong?'. Station 3 'grad: which way to nudge'. Station 4 'Optax: take a small step → NEW params'. An arrow loops from station 4 back to station 1, labelled 'repeat'. Below, a dartboard shows throws creeping toward the bullseye. A caption reads: guess, measure, nudge, step — the whole thing wrapped in jit.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">One training step: forward → loss → grad → Optax step → repeat</text>
<g transform="translate(20,40)"><rect x="0" y="0" width="96" height="40" rx="6" fill="#F4F1FB" stroke="#5E5191"/><text x="48" y="18" fill="#5E5191" font-size="8">1 forward</text><text x="48" y="32" fill="#5E5191" font-size="7">→ logits</text></g>
<text x="120" y="64" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(132,40)"><rect x="0" y="0" width="96" height="40" rx="6" fill="#FDF3E7" stroke="#C99A12"/><text x="48" y="18" fill="#8A6D3B" font-size="8">2 loss</text><text x="48" y="32" fill="#9A5A12" font-size="7">how wrong?</text></g>
<text x="232" y="64" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(244,40)"><rect x="0" y="0" width="96" height="40" rx="6" fill="#F4F1FB" stroke="#5E5191"/><text x="48" y="18" fill="#5E5191" font-size="8">3 grad</text><text x="48" y="32" fill="#5E5191" font-size="7">which way?</text></g>
<text x="344" y="64" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(356,40)"><rect x="0" y="0" width="110" height="40" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="55" y="18" fill="#1a5c38" font-size="8">4 Optax step</text><text x="55" y="32" fill="#276b45" font-size="7">→ NEW params</text></g>
<path d="M411 82 C 411 120, 68 120, 68 84" stroke="#5E5191" stroke-width="1.5" fill="none" marker-end="url(#p8a)"/>
<defs><marker id="p8a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5E5191"/></marker></defs>
<text x="240" y="116" fill="#9A5A12" font-size="8">repeat: thread the new params forward — the whole step wrapped in jit</text>
<g transform="translate(430,120)"><circle cx="30" cy="30" r="26" fill="#FBF8F2" stroke="#C99A12"/><circle cx="30" cy="30" r="16" fill="none" stroke="#C99A12"/><circle cx="30" cy="30" r="5" fill="#C93B3B"/><circle cx="22" cy="20" r="2" fill="#5E5191"/><circle cx="34" cy="26" r="2" fill="#5E5191"/></g>
<text x="240" y="140" fill="#9A938A" font-size="8">each step, the throws creep toward the bullseye (lower loss)</text>
</g></svg>
%%%

**What the darts picture gets right:** you measure the miss, nudge the next throw that way, and slowly close in — just as the loss's gradient tells you how to nudge every weight to guess better next time. **Where it breaks down:** a dart player adjusts one aim, but `grad` nudges *thousands* of numbers at once, each by its own amount, all in one step.

#### Build a genuinely small image dataset — and train the real ViT on it
Talk is cheap; let's actually train. We'll make a **tiny image dataset** we can learn: 32 little 8×8 pictures in two classes — class 0 is *bright on top*, class 1 is *bright on the bottom*, each with a little random noise so no two are identical. That's a real (if small) learnable pattern. Then we run the **real ViT forward** — patchify → patch-embed → prepend the class token → add position tags → one real attention **encoder block** → read the summary slot → head — over the whole batch with `vmap`, and train it with `grad` + Optax. Predict first: if each step nudges the params toward a lower loss, the printed **loss** should slide *down* and the **accuracy** (share of images guessed right) should climb toward 1.00. Watch both.

%%% demo id=train label="build a tiny image dataset, run the real ViT, and train end to end"
code: import jax, jax.numpy as jnp, optax
code: from jax import random
code: # ---- a genuinely small IMAGE dataset: 32 pictures, 8×8, two classes ----
code: def make_dataset(key):                       # class 0 = bright TOP, class 1 = bright BOTTOM
code:     k0, k1 = random.split(key)
code:     top = jnp.zeros((8, 8)).at[:4, :].set(1.0)   # a functional update → a NEW array
code:     bot = jnp.zeros((8, 8)).at[4:, :].set(1.0)
code:     imgs0 = top + 0.1 * random.normal(k0, (16, 8, 8))   # 16 noisy "top" images
code:     imgs1 = bot + 0.1 * random.normal(k1, (16, 8, 8))   # 16 noisy "bottom" images
code:     X = jnp.concatenate([imgs0, imgs1], 0)              # (32, 8, 8)
code:     y = jnp.concatenate([jnp.zeros(16, int), jnp.ones(16, int)])  # labels 0/1
code:     return X, y
code: P, D, C = 4, 16, 2                            # patch 4×4 → 4 patches; model dim 16; 2 classes
code: def patchify(img):                            # (8,8) → (4, 16): 4 patches of 16 pixels
code:     return img.reshape(8//P, P, 8//P, P).transpose(0,2,1,3).reshape(-1, P*P)
code: def forward(params, img):                     # the REAL ViT forward, for ONE image → logits (2,)
code:     emb = patchify(img) @ params["patch_embed"]["w"] + params["patch_embed"]["b"]  # (4,D)
code:     seq = jnp.concatenate([params["cls_token"], emb], 0) + params["pos_embed"]      # (5,D): +cls +pos
code:     a = params["attn"]                        # ONE self-attention encoder block:
code:     q, k, v = seq @ a["wq"], seq @ a["wk"], seq @ a["wv"]
code:     seq = seq + jax.nn.softmax((q @ k.T) / jnp.sqrt(D), -1) @ v   # attention + residual
code:     summary = seq[0]                          # read ONLY the class token
code:     return summary @ params["head"]["w"] + params["head"]["b"]   # → (2,) logits
code: batched_forward = jax.vmap(forward, in_axes=(None, 0))   # vmap: run over the whole batch
code: def loss_fn(params, X, y):                    # cross-entropy: one "how wrong?" number
code:     logp = jax.nn.log_softmax(batched_forward(params, X))
code:     return -jnp.mean(jnp.sum(jax.nn.one_hot(y, C) * logp, -1))
code: def accuracy(params, X, y):                   # share of images guessed correctly
code:     return jnp.mean(jnp.argmax(batched_forward(params, X), -1) == y)
code: # ---- init every array from a SPLIT key, run, and train ----
code: kd, ki = random.split(random.PRNGKey(0))
code: X, y = make_dataset(kd); print("dataset:", X.shape, "labels:", y.shape, "classes:", C)
code: ks = random.split(ki, 7); s = 0.3           # a fresh key per random piece — never reuse one
code: params = {"patch_embed": {"w": s*random.normal(ks[0], (P*P, D)), "b": jnp.zeros((D,))},
code:           "cls_token": s*random.normal(ks[1], (1, D)),
code:           "pos_embed": s*random.normal(ks[2], (5, D)),
code:           "attn": {"wq": s*random.normal(ks[3],(D,D)), "wk": s*random.normal(ks[4],(D,D)), "wv": s*random.normal(ks[5],(D,D))},
code:           "head": {"w": s*random.normal(ks[6], (D, C)), "b": jnp.zeros((C,))}}
code: print("one image → logits", forward(params, X[0]).shape, "| whole batch →", batched_forward(params, X).shape)
code: opt = optax.sgd(0.3); opt_state = opt.init(params)   # fixed shapes → jit compiles once
code: @jax.jit
code: def train_step(params, opt_state, X, y):
code:     loss, grads = jax.value_and_grad(loss_fn)(params, X, y)  # loss + which-way nudge
code:     updates, opt_state = opt.update(grads, opt_state)
code:     params = optax.apply_updates(params, updates)            # → a NEW params tree (nothing mutated)
code:     return params, opt_state, loss
code: for epoch in range(13):                       # train end to end over the whole dataset
code:     params, opt_state, loss = train_step(params, opt_state, X, y)  # thread the boxes forward
code:     acc = accuracy(params, X, y)
code:     print(f"epoch {epoch:2d}: loss = {float(loss):.4f}   accuracy = {float(acc):.2f}")
out: dataset: (32, 8, 8) labels: (32,) classes: 2
     one image → logits (2,) | whole batch → (32, 2)
     epoch  0: loss = 0.7195   accuracy = 0.50
     epoch  1: loss = 0.6956   accuracy = 0.50
     epoch  2: loss = 0.6742   accuracy = 0.56
     epoch  3: loss = 0.6563   accuracy = 0.88
     epoch  4: loss = 0.6378   accuracy = 0.94
     epoch  5: loss = 0.6132   accuracy = 1.00
     epoch  6: loss = 0.5780   accuracy = 1.00
     epoch  7: loss = 0.5294   accuracy = 1.00
     epoch  8: loss = 0.4674   accuracy = 1.00
     epoch  9: loss = 0.3959   accuracy = 1.00
     epoch 10: loss = 0.3215   accuracy = 1.00
     epoch 11: loss = 0.2514   accuracy = 1.00
     epoch 12: loss = 0.1912   accuracy = 1.00
take: <b>The loss slides down AND the accuracy climbs to 1.00 — the ViT really learned.</b> This is a full end-to-end train on a small image dataset through the <i>real</i> ViT forward (patchify → embed → +cls +pos → a genuine attention encoder block → head), not a stand-in. Read <b>both</b> numbers: the falling loss says the guesses are getting less wrong; the rising accuracy (0.50 → 1.00) says it now labels every little picture correctly. Every step ran the pure <code>forward</code> over the batch with <code>vmap</code>, used <code>grad</code> for the nudge, took an Optax step that returned a <i>new</i> params tree, and threaded it forward — all inside one <code>jit</code>-compiled step.
%%%

#### Read the learning curve: loss down, accuracy up
Numbers in a table are hard to feel. Here is that exact run drawn as a picture — loss bars shrinking on the left, the accuracy line climbing to the top on the right — so you can *see* the model learn. This is how you confirm a train worked: **both** curves move the right way.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two side-by-side learning curves from the training run. On the left, a bar chart of the loss over epochs 0 to 12: bars shrink from 0.72 down to 0.19. On the right, a line of accuracy over the same epochs climbing from 0.50 up to 1.00 and staying flat at the top. A caption reads: a real train shows BOTH — loss going down and accuracy going up.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Read BOTH curves: loss falls (left), accuracy climbs (right)</text>
<text x="128" y="36" fill="#5E5191" font-size="9">loss over epochs (down = less wrong)</text>
<line x1="40" y1="170" x2="250" y2="170" stroke="#B8AEA2"/>
<line x1="40" y1="48" x2="40" y2="170" stroke="#B8AEA2"/>
<text x="26" y="54" fill="#9A938A" font-size="7">loss</text>
<g fill="#8579b8" stroke="#5E5191">
<rect x="48" y="52" width="13" height="118" rx="1"/>
<rect x="64" y="60" width="13" height="110" rx="1"/>
<rect x="80" y="68" width="13" height="102" rx="1"/>
<rect x="96" y="74" width="13" height="96" rx="1"/>
<rect x="112" y="80" width="13" height="90" rx="1"/>
<rect x="128" y="88" width="13" height="82" rx="1"/>
<rect x="144" y="99" width="13" height="71" rx="1"/>
<rect x="160" y="115" width="13" height="55" rx="1"/>
<rect x="176" y="135" width="13" height="35" rx="1"/>
<rect x="192" y="147" width="13" height="23" rx="1"/>
<rect x="208" y="157" width="13" height="13" rx="1"/>
<rect x="224" y="164" width="13" height="6" rx="1"/>
<rect x="240" y="166" width="13" height="4" rx="1"/>
</g>
<text x="54" y="48" fill="#5E5191" font-size="7">0.72</text>
<text x="246" y="162" fill="#276b45" font-size="7">0.19</text>
<text x="145" y="184" fill="#6B645E" font-size="7">epoch 0 → 12</text>
<text x="390" y="36" fill="#276b45" font-size="9">accuracy over epochs (up = more right)</text>
<line x1="288" y1="170" x2="498" y2="170" stroke="#B8AEA2"/>
<line x1="288" y1="48" x2="288" y2="170" stroke="#B8AEA2"/>
<text x="278" y="54" fill="#9A938A" font-size="7">1.0</text>
<text x="278" y="170" fill="#9A938A" font-size="7">0.5</text>
<polyline points="296,110 312,110 328,100 344,52 360,50 376,48 392,48 408,48 424,48 440,48 456,48 472,48 488,48" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<circle cx="296" cy="110" r="3" fill="#2D8B55"/><circle cx="344" cy="52" r="3" fill="#2D8B55"/><circle cx="488" cy="48" r="3" fill="#2D8B55"/>
<text x="300" y="126" fill="#C93B3B" font-size="7">0.50</text>
<text x="470" y="42" fill="#276b45" font-size="7">1.00</text>
<text x="393" y="184" fill="#6B645E" font-size="7">epoch 0 → 12</text>
<text x="260" y="202" fill="#9A938A" font-size="9">a real train shows BOTH: loss ↓ and accuracy ↑</text>
</g></svg>
%%%

#### The honest limits — what this build can and can't do
Before the recap, a clear-eyed look at the edges of today's build — knowing them *is* understanding it.

- **A plain ViT is data-hungry.** Remember the history from concept 1: the ViT threw away the CNN's "nearby pixels matter" head start. Our toy learned fast *because* the pattern was simple and clean; on a *real* small dataset a plain ViT often *loses* to a CNN — it has to learn locality from scratch. The remedy in practice is more data or starting from pretrained weights (a topic for later).
- **`jit` can't trace data-dependent Python control flow.** If your `forward` branches on the *value* of an array (`if x > 0: ...` on a traced array), `jit` can't handle it — it only sees shapes, not values. JAX has special tools (`lax.cond`, `lax.scan`) for that, met later.
- **This is one device, for understanding.** We build and train on a single machine to *learn the assembly*. Spreading training across many chips, serving it in production, and fancy variants (Swin, DeiT) are separate topics — deliberately out of scope so the assembly stays clear.

You've assembled the whole thing and trained it end to end — loss down, accuracy up. Time to gather every brick onto one page.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — you assembled a Vision Transformer *and* the JAX machinery to train it, from raw pixels to a falling loss and a climbing accuracy. One last everyday picture for the whole day: a **model-airplane kit.** The box (this week) held all the parts. The instruction booklet (this lesson) is the **assembly** order: glue the wings (patches → embeddings), tag each part (position tags + summary slot), snap in the engine (encoder blocks), add the propeller (the head) — and then a little motor (grad + Optax) that makes it actually *fly* better each time you wind it. **Where the kit picture breaks down:** a finished model plane never changes, but your ViT keeps *learning* — its numbers improve every step, which is the whole point. Below is the day station by station, plus two cheat-sheets to keep on your desk.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A one-page summary flow of the whole day in two rows. Top row, the model: patchify, then embed plus position plus class token, then encoder blocks, then MLP head, ending in class scores. Bottom row, the JAX machinery: pure function plus immutable pytree params, PRNGKey split, vmap for batch, jit for speed, grad plus Optax for the train step. A caption reads: the model is WHAT you build; the JAX bricks are HOW you build and train it.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The whole assembly: the model (top) built and trained by JAX (bottom)</text>
<text x="60" y="38" fill="#5E5191" font-size="8">the MODEL</text>
<g transform="translate(20,44)"><rect x="0" y="0" width="80" height="34" rx="6" fill="#F4F1FB" stroke="#5E5191"/><text x="40" y="15" fill="#5E5191" font-size="7">patchify</text><text x="40" y="27" fill="#8579b8" font-size="6">image → tiles</text></g>
<text x="102" y="65" fill="#9A5A12">→</text>
<g transform="translate(114,44)"><rect x="0" y="0" width="94" height="34" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="47" y="15" fill="#1a5c38" font-size="7">embed + pos + cls</text><text x="47" y="27" fill="#276b45" font-size="6">translate, tag, folder</text></g>
<text x="210" y="65" fill="#9A5A12">→</text>
<g transform="translate(222,44)"><rect x="0" y="0" width="84" height="34" rx="6" fill="#F4F1FB" stroke="#5E5191"/><text x="42" y="15" fill="#5E5191" font-size="7">encoder ×N</text><text x="42" y="27" fill="#8579b8" font-size="6">attention + MLP</text></g>
<text x="308" y="65" fill="#9A5A12">→</text>
<g transform="translate(320,44)"><rect x="0" y="0" width="70" height="34" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="35" y="15" fill="#1a5c38" font-size="7">MLP head</text><text x="35" y="27" fill="#276b45" font-size="6">read summary</text></g>
<text x="392" y="65" fill="#9A5A12">→</text>
<g transform="translate(404,44)"><rect x="0" y="0" width="88" height="34" rx="6" fill="#FDF3E7" stroke="#C99A12"/><text x="44" y="15" fill="#8A6D3B" font-size="7">class scores</text><text x="44" y="27" fill="#9A5A12" font-size="6">biggest = guess</text></g>
<text x="60" y="112" fill="#5E5191" font-size="8">the JAX machinery</text>
<g transform="translate(20,118)" font-size="7">
<rect x="0" y="0" width="90" height="34" rx="6" fill="#FBF8F2" stroke="#9A8A5A"/><text x="45" y="15" fill="#6b5e2a">pure fn + pytree</text><text x="45" y="27" fill="#9A8A5A" font-size="6">immutable, OUTSIDE</text>
<text x="98" y="22" fill="#9A5A12">→</text>
<rect x="110" y="0" width="80" height="34" rx="6" fill="#FBF8F2" stroke="#9A8A5A"/><text x="150" y="15" fill="#6b5e2a">PRNGKey split</text><text x="150" y="27" fill="#9A8A5A" font-size="6">explicit dice</text>
<text x="194" y="22" fill="#9A5A12">→</text>
<rect x="206" y="0" width="70" height="34" rx="6" fill="#FBF8F2" stroke="#9A8A5A"/><text x="241" y="15" fill="#6b5e2a">vmap</text><text x="241" y="27" fill="#9A8A5A" font-size="6">batch, no loop</text>
<text x="280" y="22" fill="#9A5A12">→</text>
<rect x="292" y="0" width="60" height="34" rx="6" fill="#FBF8F2" stroke="#9A8A5A"/><text x="322" y="15" fill="#6b5e2a">jit</text><text x="322" y="27" fill="#9A8A5A" font-size="6">compile fast</text>
<text x="356" y="22" fill="#9A5A12">→</text>
<rect x="368" y="0" width="124" height="34" rx="6" fill="#FBF8F2" stroke="#9A8A5A"/><text x="430" y="15" fill="#6b5e2a">grad + Optax step</text><text x="430" y="27" fill="#9A8A5A" font-size="6">measure → nudge → repeat</text>
</g>
<text x="260" y="184" fill="#9A938A" font-size="9">the model is WHAT you build; the JAX bricks are HOW you build and train it</text>
</g></svg>
%%%

The day in a few beats:
- **The line.** A ViT is an assembly line: [[patches||square tiles cut from the image; the ViT's "words"]] → [[patch embedding||one shared Dense that maps each patch to size d]] + [[positional embedding||a learned per-seat tag added so order matters]] + a [[class token||one summary folder read out at the end]] → [[encoder blocks||the same transformer layer reused, stacked N times]] → an [[MLP head||a tiny layer that turns the summary into class scores]].
- **Why position tags.** Attention is order-blind, so a shuffled image looks identical — *adding* a position tag to each patch breaks that tie.
- **Numbers live outside, and never change in place.** The model is a [[pure function||same inputs → same output, no hidden state]]; all learned numbers sit in one [[params||the model's tree of numbers, held outside and passed in]] pytree you hold. JAX arrays are [[immutable||you never edit in place; a "change" builds a new array]] — so a train step *returns a new params tree* instead of scribbling over the old one. That purity + immutability is exactly what makes `jit`/`vmap`/`grad` safe.
- **Explicit randomness.** All numbers start as random draws from a [[PRNGKey||JAX's explicit seed you pass in]]; **split** it for independent draws — never reuse a key.
- **Fast + batched.** [[jit||compile once, reuse]] traces once and caches by shape (keep shapes fixed or it recompiles); [[vmap||write for one image, run over the batch]] adds the batch axis for you.
- **Train end to end, read both curves.** [[grad||the which-way-to-nudge tool]] gives the gradient, Optax turns it into a step, you thread the improved params forward, all wrapped in `jit` — then confirm it learned by watching the **loss fall** *and* the **accuracy climb**.

The one rule to memorize: **assemble the model as a pure function whose numbers live in an immutable pytree, start them from a split PRNGKey, run it with vmap+jit, train it end to end with grad+Optax — one step in, a new improved params tree out — and confirm learning by both loss and accuracy.**

#### Cheat-sheet · the assembly, station by station
%%% table
:: Station / brick :: What it does :: Watch out for
Patchify :: cut image into square tiles, flatten each :: reshape can scramble pixels — print shapes
Patch embedding :: one shared Dense → each patch becomes size d :: same translator for every patch
Positional embedding :: add a learned per-position tag :: without it, attention is order-blind
Class token :: one summary slot read out at the end :: prepended to the front of the row
Encoder blocks ×N :: attention + MLP, reused from text transformers :: each block has its own numbers
MLP head :: read the summary slot → one score per class :: biggest score = the guess
params (immutable pytree) :: all numbers, held outside the model :: never edited in place — a step returns a NEW tree
PRNGKey / split :: explicit randomness for init :: never reuse a key — split first
jit :: compile once, cache by shape :: changing shape → silent recompile
vmap :: run the one-image fn over a batch :: write for one example, vmap the batch
grad + Optax :: gradient → update → new params :: thread the new boxes; confirm loss ↓ and accuracy ↑
%%%

#### Cheat-sheet · the words you met today
%%% jargon
ViT | Vision Transformer: treats an image as a sequence of patch-"words" and runs transformer encoder blocks on it
patch | a fixed-size square tile cut from the image; the ViT's version of a token
patchify | cutting the image into non-overlapping patches, then flattening each into a vector
patch embedding | one shared learned Dense layer mapping each flattened patch to the model size d
positional embedding | a learned per-position vector added to each patch so the model knows patch order
class token | one extra learned vector prepended to the sequence; its final state is read out for the class guess
encoder block | the repeating transformer layer (attention + feed-forward, with skip + norm), reused unchanged for vision
MLP head | a small feed-forward layer on the class-token state that outputs one score per class
permutation-invariant | attention has no built-in sense of order; a shuffled sequence gives the same result without position tags
pure function | output depends only on inputs — no hidden state; required by jit, vmap, and grad
immutable | a JAX array can never be edited in place; any change builds a new array, so a train step returns a new params tree
params (pytree) | the nested tree of all learned numbers, held OUTSIDE the model and passed in each run
PRNGKey / split | JAX's explicit randomness seed; split it for fresh independent draws, never reuse
jit | just-in-time compile: trace once, cache by input shape/dtype, reuse fast (a new shape recompiles)
static_argnums | mark a jit argument as a true constant so only a real change in it recompiles
vmap | write a function for one example; vmap runs it over a whole batch, no Python loop
grad | jax.grad(loss_fn): the gradient — how to nudge every parameter to lower the loss
accuracy | the share of examples the model guesses correctly; read alongside the loss to confirm it learns
%%%

#### Where these lead next
You met several ideas by name that get their own lessons — so you're not expected to know their guts yet:
- **Attention internals** (queries/keys/values, scores, **softmax**, **multi-head**) and **layer normalization** and **skip connections** — the guts of the encoder block — are covered in the transformer/attention modules.
- **Optimizer internals** (SGD vs Adam, learning-rate schedules) live in the optimization module.
- **Multi-device training** (`pmap`, sharding), **mixed precision**, compiled loops (`lax.scan`), **pretrained ViT weights / transfer learning**, and **data augmentation** are all later, richer topics — a whole landscape ahead of you.

That's the whole day, and the whole week. You now think in JAX: pure functions in, transformed (fast, batched, differentiable) functions out — and you've assembled a real Vision Transformer with your own hands and trained it end to end. Next you'll dive into the *math* of transformers and search in **Module 8**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete the week. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: In a Vision Transformer, what plays the role that "words" play in a text transformer? | a:2 | single pixels | color channels | fixed-size square patches of the image | the class scores | fb: A ViT cuts the image into non-overlapping square patches, flattens each, and treats the row of patches like a sentence of "words" — then runs the same encoder blocks a text transformer uses.
q: Why does a ViT need positional embeddings added to its patches? | a:1 | to make training faster | because attention is permutation-invariant, so without position tags a shuffled image looks identical | to reduce the number of parameters | to convert pixels to numbers | fb: Attention looks at all patches at once with no built-in sense of order. Adding a learned position vector to each patch is what lets the model tell top-left from bottom-right — otherwise a scrambled image gives the same result.
q: After training the mini-ViT on the small image dataset, how do you CONFIRM it actually learned? | a:2 | check that jit compiled without error | count the number of params | read BOTH the loss (should fall) and the accuracy (should climb toward 1.00) | make sure the class token is at the front | fb: Loss falling says the guesses are getting less wrong; accuracy rising says it labels more images correctly. You read BOTH — a falling loss with flat accuracy would be a warning, so the two together are the real confirmation of learning.
q: Where do a JAX/ViT model's learned numbers live, and why does it matter? | a:3 | hidden inside the model object, for convenience | in a global variable JAX manages | nowhere until training ends | in an immutable pytree (params) held OUTSIDE the model and passed in, so the model stays a pure function jit/vmap/grad can transform | fb: Keeping all numbers in an external params pytree makes forward a pure function of (params, image). Because JAX arrays are immutable, a train step returns a NEW params tree instead of editing in place — and that purity is exactly what jit, vmap, and grad require.
%%%

@@@ produce id=produce tag="Produce" title="Assemble and train a mini-ViT with your own hands" gotit="Done"
Time to feel the whole assembly for yourself. You'll build a *small* ViT pipeline — a tiny two-class image dataset, patchify, embed the patches, add a class token and position tags, run one real attention **encoder block**, read the summary slot through a head, then train it end to end with `grad` + Optax — printing shapes, the loss, and the accuracy as you go. **Predict first:** when you run the training loop, what should the printed **loss** do — go up, stay flat, or slide down — and what should the **accuracy** do? And when you feed `jit` a *different* batch shape each call, will it get faster or slower? Then run it and **watch** the numbers. Pick one path.

#### Option A · write it yourself
Create `sessions/m07-thinking-in-jax/day-06-vit-capstone/experiment.py`. Start with `import jax`, `import jax.numpy as jnp`, `import optax`, and `from jax import random`. **Print what you observe at each step.** (1) Build a tiny two-class image dataset: 16 "bright-top" 8×8 images (label 0) and 16 "bright-bottom" images (label 1), each with a little noise — use `jnp.zeros((8,8)).at[:4,:].set(1.0)` (a functional update — a *new* array, since JAX arrays are immutable) for the top pattern. `print` the dataset shape `(32, 8, 8)` and labels shape `(32,)`. (2) Write `patchify(img)` for `P = 4` (reshape `(2,4,2,4)` → transpose `(0,2,1,3)` → reshape `(4,16)`) — **print the shape** and confirm `(4, 16)`. (3) Init a params pytree with `jax.random.split` giving each piece a **fresh key** (prove reuse gives identical arrays but split keys differ): `patch_embed w (16,D)/b (D,)`, `cls_token (1,D)`, `pos_embed (5,D)`, an attention block `wq/wk/wv (D,D)`, a head `w (D,2)/b (2,)`, with `D = 16`. (4) Write a **pure** `forward(params, img)`: embed patches, prepend the class token, add `pos_embed`, run *one real attention step* (`q,k,v = seq@wq, seq@wk, seq@wv`; `seq = seq + jax.nn.softmax((q@k.T)/jnp.sqrt(D)) @ v`), take `seq[0]` (the class token), run the head — **print the single-image logits shape `(2,)`**. `vmap` it over the batch and print `(32, 2)`. (5) Write `loss_fn` (cross-entropy via `jax.nn.log_softmax` + `jax.nn.one_hot`) and an `accuracy` function (`argmax == y`), then a `@jax.jit` `train_step` using `jax.value_and_grad`, `optax.sgd(0.3)`, and `optax.apply_updates` (which returns a *new* params tree). Loop ~13 epochs threading the boxes and **print loss AND accuracy each epoch** — watch the loss fall and the accuracy climb toward 1.00. (6) On purpose, trigger the recompile trap: call the jitted step with a *different* batch shape and **notice** it's slow; then keep the shape fixed and notice it's fast. Run with `python3 sessions/m07-thinking-in-jax/day-06-vit-capstone/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Week 1 Weekend Lab (ViT Capstone) artifact.

Create sessions/m07-thinking-in-jax/day-06-vit-capstone/experiment.py that, with a comment on each step and printing the shapes/values it observes, builds and TRAINS a mini Vision Transformer end to end on a small image dataset:
1. import jax, jax.numpy as jnp, optax, and from jax import random. Build a tiny 2-class dataset: top = jnp.zeros((8,8)).at[:4,:].set(1.0) (note: .at[].set() returns a NEW array because JAX arrays are immutable), bot = jnp.zeros((8,8)).at[4:,:].set(1.0); make 16 noisy copies of each (top+0.1*random.normal(...)), stack into X (32,8,8) with labels y (32,) = sixteen 0s then sixteen 1s. Print X.shape and y.shape.
2. patchify(img) with P=4: img.reshape(2,4,2,4).transpose(0,2,1,3).reshape(4,16); print the shape and confirm (4,16).
3. Split a PRNGKey with jax.random.split into fresh keys, one per random piece. Build a params pytree (D=16): patch_embed {'w':(16,16),'b':(16,)}, cls_token (1,16), pos_embed (5,16), attn {'wq','wk','wv' each (16,16)}, head {'w':(16,2),'b':(2,)}. Also demonstrate that reusing the SAME key gives identical arrays while split keys give different ones.
4. Define a PURE forward(params, img): emb = patchify(img) @ patch_embed w + b; seq = concat([cls_token, emb]) + pos_embed (shape (5,16)); one real attention block: q,k,v = seq@wq, seq@wk, seq@wv; seq = seq + jax.nn.softmax((q@k.T)/jnp.sqrt(16), -1) @ v; summary = seq[0]; logits = summary @ head w + b; print single-image logits shape (2,). vmap it over the batch and print (32,2).
5. loss_fn = cross-entropy (jax.nn.log_softmax + jax.nn.one_hot(y,2)); accuracy = mean(argmax(logits)==y). opt = optax.sgd(0.3); opt_state = opt.init(params). Define @jax.jit train_step(params, opt_state, X, y): loss, grads = jax.value_and_grad(loss_fn)(...); updates, opt_state = opt.update(grads, opt_state); params = optax.apply_updates(params, updates) (returns a NEW params tree); return params, opt_state, loss. Loop 13 epochs threading params + opt_state, printing BOTH the loss (should slide down) and the accuracy (should climb toward 1.00) each epoch.
6. Show the jit recompilation trap: time the jitted step on the fixed shape vs a changed batch shape, and print which is slower and why (a new shape triggers a recompile).
Then run it and paste the output at the bottom as a comment. Explain in one line why forward must be a pure function whose immutable params live outside it.
%%%

#### What you should see (dataset → patchify → real forward → train — loss down, accuracy up)
- The dataset prints shapes `X (32, 8, 8)` and `y (32,)` — 32 little images in two classes.
- Patchify prints `(4, 16)` — 4 patches of 16 pixels, matching `(H/P)×(W/P) = (8/4)×(8/4) = 4`.
- Reusing the same **PRNGKey** gives *identical* arrays; `random.split` gives *independent* ones.
- The pure `forward(params, img)` returns **single-image logits of shape `(2,)`**; `vmap` over the batch gives `(32, 2)`.
- The training loop's **loss slides down** (about `0.72` → `0.19`) *and* the **accuracy climbs** to `1.00` — read both to confirm the mini-ViT learned.
- Feeding the jitted step a **different batch shape** each call makes it **recompile (slow)**; a fixed shape stays **fast**.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m07-thinking-in-jax/day-06-vit-capstone/log.md`: (1) in one sentence, name the five stations of the ViT assembly line, in order; (2) after training, which two numbers did you read to confirm the mini-ViT learned, and which way did each move? (3) why the model must be a pure function whose *immutable* params live outside it, and one JAX trap you hit (a reused key, a scrambled reshape, or a jit recompile) and how you fixed it.
!!!

@@@ fin
