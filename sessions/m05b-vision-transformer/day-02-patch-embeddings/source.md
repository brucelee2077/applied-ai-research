---
quest_id: wf7-d02-patchembed
mode: concept
donor: v9-base.donor
page_title: "Module 5b · Day 2 — Patch Embeddings"
module_label: "Module 05b · Represent · Day 2"
title: "Patch Embeddings"
subtitle: "Turning Each Tile Into a Vector"
brand_sub: "Foundations · M5b Day 2"
spine: "passport"
nav_prev_href: "../day-01-images-as-patches/lesson.html"
nav_prev_label: "Images as Patches"
nav_next_href: "../day-03-transformer-for-vision/lesson.html"
nav_next_label: "Transformer for Vision"
fin_title: "Module 5b · Day 2 complete! 🏆"
fin_body: "Nice work — you've unlocked <b>Patch Embeddings</b>: each tile gets a <b>passport</b> — flatten its pixels, run them through one shared linear projection to get a vector ID, then stamp in a position so the model knows where the tile sat. Add a special [CLS] token to lead the sequence, and the picture is finally ready for a Transformer.<br>Next up: <b>A Transformer for Vision</b>."
notebook_yardstick: null
---

@@@ hero
@lede You know how, at the airport, every traveler needs a **passport** — one small booklet that turns a person into something the border computer can check: a name, a number, a photo, a country stamp? Yesterday we chopped a picture into a grid of little square tiles. But a raw tile is just a messy pile of pixel numbers — the model can't do much with that. Today we hand each tile its own **passport**: we run it through one shared little machine that turns those messy pixels into a clean, fixed-length ID card of numbers, then we stamp on *where the tile sat* in the picture. This little step — the **patch embedding** — is the exact front door of the Vision Transformer, the same family of model behind image search and the "eyes" of tools like GPT-4. Once every tile carries a passport, the picture is finally ready to walk through the border and be read.
@goal Together we'll give every tile a **passport**. You'll see how one shared multiply turns a pile of pixels into a tidy vector, learn the surprising fact that this "stamp" is secretly a convolution, line the passports up into the exact sequence a Transformer eats, add one special leader token at the front, and stamp each passport with *where its tile sat* — then meet, as friendly little puzzles, the things this front door still can't do and exactly how each gets fixed later. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="The passport machine" title="One shared multiply turns a tile into a tidy vector" gotit="Got the projection"
Let's pick up right where yesterday left off. You already know how to take one square tile and **flatten** it — [[flatten||To unroll a small 2-D grid of numbers into one long single-row list, by reading it row by row.]] just means unrolling its little grid of pixels into one long strip of numbers. But that strip is an awkward, messy length, and its numbers are raw pixel brightnesses — not something a Transformer knows how to read. So we do one clean-up step: we push every tile through the **same** little machine that turns its messy strip into a tidy, fixed-length vector. That machine is the **patch embedding**, and its output is the tile's *passport*.

Picture the passport desk at an airport. A traveler walks up — different height, different clothes, different bag every time — and the clerk fills out the **same** kind of ID booklet for each one: same fields, same size, every time. The machine doesn't care who walks up; it always prints an ID card of the same tidy shape. Our tile machine works the same way: any tile in, one fixed-length passport out.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A passport desk. On the left, several different messy tiles of pixel numbers walk up. In the middle, one shared machine labelled the patch embedding matrix W. On the right, each tile leaves as a tidy fixed-length passport vector of length D. The same machine is used for every tile."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">The passport desk: any tile in → one tidy fixed-length vector out</text><text x="70" y="42" fill="#6B645E">messy tiles</text><rect x="35" y="52" width="70" height="22" fill="#F6EFCF" stroke="#C99A12"/><text x="70" y="67" fill="#8A6D3B" font-size="9">7 2 9 1 5 3 …</text><rect x="35" y="82" width="70" height="22" fill="#EAF0FA" stroke="#5E5191"/><text x="70" y="97" fill="#5E5191" font-size="9">0 8 4 6 2 1 …</text><rect x="35" y="112" width="70" height="22" fill="#FDECEC" stroke="#C93B3B"/><text x="70" y="127" fill="#C93B3B" font-size="9">3 3 9 0 7 5 …</text><path d="M112 63 H150 M112 93 H150 M112 123 H150" stroke="#B8AEA2"/><rect x="160" y="60" width="110" height="66" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="215" y="88" fill="#C93B3B">passport</text><text x="215" y="104" fill="#C93B3B">machine W</text><text x="215" y="145" fill="#9A938A" font-size="9">same for every tile</text><path d="M275 72 H315 M275 93 H315 M275 114 H315" stroke="#B8AEA2"/><rect x="325" y="66" width="150" height="14" fill="#EAF5EE" stroke="#2D8B55"/><rect x="325" y="86" width="150" height="14" fill="#EAF5EE" stroke="#2D8B55"/><rect x="325" y="106" width="150" height="14" fill="#EAF5EE" stroke="#2D8B55"/><text x="400" y="145" fill="#276b45">tidy passports (each length D)</text><text x="490" y="76" fill="#276b45" font-size="9" text-anchor="end">D long</text></g></svg>
%%%

**What the passport-desk picture gets right:** one shared desk, same booklet for everyone, out comes an ID of a fixed tidy shape — that is exactly "one shared machine turns any tile into a fixed-length vector." **Where it breaks down:** a passport clerk follows fixed rules, but our machine *learns* — during training its numbers slowly adjust so the passport it prints is actually useful for seeing.

#### What the machine actually does: one multiply
The machine is astonishingly simple. It is one [[linear projection||Multiplying a vector by one shared table of numbers (a weight matrix) to turn it into a new fixed-length vector.]] — in plain words, "multiply the tile's strip of numbers by one shared table of numbers." That shared table is the **patch-embedding matrix**, usually written `W`. The tidy result is the tile's [[patch embedding||The fixed-length vector, of length D, that a tile becomes after the shared linear projection. This is what the Transformer actually reads.]], a vector of a length we pick, called `D` (short for the *embedding dimension* — how many numbers are in each passport).

The whole thing is one gentle line: **passport = tile's flattened pixels × the shared matrix W.** One multiply, and a messy strip becomes a clean length-`D` vector.

#### Watch the shape change, step by step
Let's watch the numbers shrink into shape. Take a color tile that is `16 × 16` pixels with 3 color layers ([[channels||The layers of an image: 1 for grayscale, 3 for color — red, green, blue. A color pixel is 3 numbers.]]). Flattened, that's `16 × 16 × 3 = 768` raw numbers. We usually pick `D = 768` too, but the point is the machine maps that long messy strip to a tidy fixed length no matter what:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A three-step build-up of the shape change. Step one: a 16 by 16 by 3 color tile. Step two: flattened into a long strip of 768 raw pixel numbers. Step three: multiplied by the shared matrix W into a tidy passport of length D, shown as a short neat bar."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">One tile's journey: pixels → flat strip → tidy passport (length D)</text><g transform="translate(30,48)"><rect x="8" y="8" width="60" height="60" fill="#FDECEC" stroke="#C93B3B"/><rect x="4" y="4" width="60" height="60" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="0" width="60" height="60" fill="#EAF0FA" stroke="#5E5191"/><path d="M20 0 V60 M40 0 V60 M0 20 H60 M0 40 H60" stroke="#5E5191" stroke-width="0.5"/><text x="34" y="90" fill="#6B645E">1 · tile</text><text x="34" y="104" fill="#9A938A">16×16×3</text></g><text x="140" y="80" fill="#9A5A12" font-size="14">→</text><g transform="translate(160,66)"><rect x="0" y="0" width="150" height="16" fill="#F6EFCF" stroke="#C99A12"/><path d="M18 0 V16 M36 0 V16 M54 0 V16 M72 0 V16 M90 0 V16 M108 0 V16 M126 0 V16" stroke="#C99A12" stroke-width="0.5"/><text x="75" y="42" fill="#8A6D3B">2 · flatten</text><text x="75" y="56" fill="#9A938A">768 raw numbers</text></g><text x="330" y="80" fill="#9A5A12" font-size="14">→</text><g transform="translate(350,60)"><rect x="0" y="0" width="70" height="42" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="35" y="26" fill="#C93B3B" font-size="9">× W</text><text x="35" y="60" fill="#9A938A">3 · multiply</text></g><g transform="translate(350,120)"><rect x="0" y="0" width="90" height="14" fill="#EAF5EE" stroke="#2D8B55"/><text x="45" y="34" fill="#276b45">tidy passport</text><text x="45" y="48" fill="#276b45">length D</text></g><text x="480" y="128" fill="#276b45" font-size="9" text-anchor="end">D long</text></g></svg>
%%%

The magic is in that middle machine: *the same* matrix `W` is used for every single tile. This is called [[weight sharing||Using the same learned weights for every tile instead of a separate set per position — it treats every tile fairly and keeps the model small.]]. Why share? Because a useful little clue — an edge, a splash of color — is worth spotting *wherever* it shows up in the picture. One shared machine treats every tile fairly and keeps the model small. And now every tile speaks the same tidy language: a length-`D` vector, ready for the next steps.

@@@ concept id=c2 tag="Secretly a stamp roll" title="The same passport machine is really a convolution" gotit="Got the conv view"
Here's a delightful secret that makes engineers smile. Cutting the picture into tiles, flattening each one, and multiplying every tile by the same matrix `W` — all of that, done in one smooth motion, is *exactly* the same as a tool you may have met before: a **convolution**. You don't need to know the details of convolutions today; the everyday picture is all you need.

Think of one of those **date stamps** that offices roll across a stack of papers — the kind on a little wheel. You press it down, it prints a mark; you slide it over by exactly its own width, press again, print another mark; slide, press, slide, press. It never overlaps and never leaves a gap, because it moves over by exactly one stamp-width each time. That rolling stamp *is* our patch embedding: each "press" reads one `P × P` tile and prints one passport, and it steps over by exactly `P` pixels so the tiles line up edge to edge — no gaps, no overlaps.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A rolling date stamp moving across an image. The stamp is a P by P window. It presses at the top-left, prints one passport, then slides right by exactly P pixels and presses again. Because the step equals the stamp width, the presses tile the image with no gaps and no overlaps."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A rolling stamp: press, slide by exactly P, press again — no gaps</text><g transform="translate(70,40)"><rect x="0" y="0" width="180" height="120" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><path d="M45 0 V120 M90 0 V120 M135 0 V120 M0 30 H180 M0 60 H180 M0 90 H180" stroke="#9A5A12" stroke-width="0.7"/><rect x="0" y="0" width="45" height="30" fill="none" stroke="#C93B3B" stroke-width="3"/><text x="22" y="20" fill="#C93B3B" font-size="9">press</text><path d="M45 15 h34" stroke="#5E5191" stroke-width="1.4" marker-end="url(#ar)"/><text x="62" y="12" fill="#5E5191" font-size="8">slide P</text></g><defs><marker id="ar" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#5E5191"/></marker></defs><text x="160" y="180" fill="#6B645E">step = stamp width = P → tiles never gap or overlap</text><text x="360" y="55" fill="#9A5A12" font-size="14">→</text><g transform="translate(390,40)"><rect x="0" y="0" width="16" height="14" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="20" width="16" height="14" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="40" width="16" height="14" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="60" width="16" height="14" fill="#EAF5EE" stroke="#2D8B55"/><text x="8" y="98" fill="#276b45">one passport</text><text x="8" y="111" fill="#276b45">per press</text></g></g></svg>
%%%

**What the rolling-stamp picture gets right:** the stamp reads one square window, prints one result, and steps over by its own width — so the windows perfectly tile the page, exactly like our patches. **Where it breaks down:** a real date stamp always prints the *same* mark, but this stamp's "ink" is the learned matrix `W`, so what it prints depends on the tile's pixels and improves during training.

#### Two names, one operation
So why does anyone care that it's a convolution? Two reasons, and both are gifts. First, it explains a phrase you'll see everywhere: people describe the patch embedding as **"a convolution with kernel size `P` and stride `P`."** In plain words: the stamp is `P` pixels wide (kernel `= P`) and it slides `P` pixels each step (stride `= P`). Kernel equals stride equals patch size — that's the whole recipe. Second, it links the shiny new Vision Transformer straight back to the older world of [[CNN||Convolutional Neural Network — the classic image model that slides small learned filters across a picture. You met these in the earlier vision module.]]s you met earlier: the very first layer of a ViT is, quietly, just one convolution.

%%% table
:: The word :: What it means for our tile :: Everyday picture
kernel size = P :: the stamp reads a P×P square at a time :: the stamp's own width
stride = P :: it slides P pixels before the next press :: step over by exactly one stamp
output channels = D :: each press prints a length-D passport :: the ID card it stamps out
%%%

So you can hold two pictures of the same thing in your head: a **passport desk** (multiply each flattened tile by `W`) and a **rolling stamp** (one convolution with kernel `= stride = P`). They give the identical result — pick whichever helps you think. Either way, every tile now has its length-`D` passport.

@@@ concept id=c3 tag="A line of passports" title="Line the passports up into the sequence a Transformer eats" gotit="Got the sequence"
Now we have a whole stack of passports — one per tile. What does the Transformer actually want to receive? Not a grid, and not a stack. It wants a plain **ordered line**: passport 1, then passport 2, then passport 3, and so on. So we line the tiles' passports up in reading order — left to right along the top row, then the next row down, just like reading a page.

Picture a **boarding line at the gate**. All the travelers already have their passports (their tidy length-`D` vectors from the machine). To board, they don't crowd around in a grid — they form one single-file line, and the gate agent scans them one after another. That single-file line of passport-holders is precisely what we hand the Transformer: an ordered list of tile-vectors, called a [[sequence||An ordered list of items, one after another — the exact input a Transformer expects. Here, the tiles' passports laid out in reading order.]].

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A grid of passports on the left, numbered in reading order. An arrow shows them forming a single-file boarding line on the right: passport 1, passport 2, passport 3, and so on, an ordered sequence of length-D vectors."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Passports board single-file → the ordered line a Transformer scans</text><g transform="translate(35,42)"><rect x="0" y="0" width="110" height="110" fill="#EAF5EE" stroke="#2D8B55"/><path d="M27.5 0 V110 M55 0 V110 M82.5 0 V110 M0 27.5 H110 M0 55 H110 M0 82.5 H110" stroke="#2D8B55" stroke-width="0.8"/><text x="14" y="18" fill="#1a5c38">1</text><text x="41" y="18" fill="#1a5c38">2</text><text x="69" y="18" fill="#1a5c38">3</text><text x="96" y="18" fill="#1a5c38">4</text><text x="14" y="45" fill="#1a5c38">5</text><text x="41" y="45" fill="#1a5c38">6</text><text x="69" y="45" fill="#1a5c38">7</text><text x="96" y="45" fill="#1a5c38">8</text><text x="14" y="100" fill="#1a5c38">…</text><text x="55" y="130" fill="#6B645E">grid of passports</text></g><text x="175" y="98" fill="#9A5A12" font-size="16">→</text><g transform="translate(205,80)"><rect x="0" y="0" width="290" height="24" fill="#FCF3DC" stroke="#C99A12"/><path d="M29 0 V24 M58 0 V24 M87 0 V24 M116 0 V24 M145 0 V24 M174 0 V24 M203 0 V24 M232 0 V24 M261 0 V24" stroke="#C99A12" stroke-width="0.6"/><text x="14" y="16" fill="#8A6D3B">1</text><text x="43" y="16" fill="#8A6D3B">2</text><text x="72" y="16" fill="#8A6D3B">3</text><text x="145" y="16" fill="#8A6D3B">…</text><text x="275" y="16" fill="#8A6D3B">N</text></g><text x="350" y="130" fill="#276b45">a sequence of N passports (each length D)</text></g></svg>
%%%

**What the boarding-line picture gets right:** the travelers form one ordered single-file line and get scanned one after another — exactly how a Transformer reads a sequence, one token at a time in order. **Where it breaks down:** in a real boarding line the person can see who's ahead and behind them; our passports, once in line, don't yet know their neighbors at all (we'll fix the sense of *position* in a couple of concepts).

#### The shape of the thing, and one word: token
Each item in this line is called a [[token||One item in the sequence a Transformer reads — one vector. In text it's a word-piece; in vision it's one tile's passport.]]. In a text Transformer a token is a word-piece; in our Vision Transformer, **one token = one tile's passport**. That is the beautiful bridge: the model doesn't know or care that these tokens came from an image. To it, a picture is just a sentence of `N` tokens, each a length-`D` vector.

How many tokens is `N`? You met the count yesterday — it's just how many tiles fit across times how many fit down. With image height `H`, width `W`, and patch size `P`:

%%% formula
expr: N = (H / P) × (W / P)
note: N = number of patch tokens. H = image height in pixels, W = width, P = patch size. Each fraction is "how many P-sized tiles fit along that edge."
%%%

So a `224 × 224` image with `P = 16` gives `N = 14 × 14 = 196` tokens: the Transformer reads a sentence of 196 tile-passports, each a length-`D` vector. **You now have the full input format** a Transformer expects — an ordered line of `N` vectors — built entirely out of picture tiles.

@@@ concept id=c4 tag="The team captain" title="Add one special leader token: [CLS]" gotit="Got the CLS token"
Before the line boards, we slip in one extra passport-holder at the very front — and this one is *not* a tile. It's a single special vector we add ourselves, called the [[class token||An extra learnable vector added to the front of the patch sequence, written [CLS] or CLS. Later it gathers information from all the tiles to summarize the whole image.]], usually written `[CLS]` (short for *classification*).

Think of a **team captain** wearing an armband at the front of the line. The captain isn't a normal player — they don't come from the picture at all. Their whole job is to *listen to every teammate* and, at the end, speak for the whole team. When someone asks "what is this a picture of?", they don't poll all 196 tiles — they just ask the captain, who has gathered the summary. The `[CLS]` token is that captain: an empty leader we pin to the front, whose job (later, once the tiles start sharing information) is to collect a summary of the *whole* image in one place.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A single special CLS token, drawn with a captain armband at the very front, is prepended to the line of N tile passports. The sequence length becomes N plus 1. The CLS token starts blank and its job is to later summarize the whole image."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Pin one leader token at the front → sequence length N becomes N+1</text><g transform="translate(35,60)"><rect x="0" y="0" width="46" height="34" rx="4" fill="#EFEAF7" stroke="#5E5191" stroke-width="2"/><text x="23" y="15" fill="#5E5191" font-size="9">[CLS]</text><text x="23" y="28" fill="#7C6DAA" font-size="8">🎗 captain</text></g><text x="98" y="80" fill="#9A5A12" font-size="14">+</text><g transform="translate(115,60)"><rect x="0" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="46" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="92" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="138" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="184" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><text x="20" y="22" fill="#1a5c38">1</text><text x="66" y="22" fill="#1a5c38">2</text><text x="112" y="22" fill="#1a5c38">3</text><text x="158" y="22" fill="#1a5c38">…</text><text x="204" y="22" fill="#1a5c38">N</text></g><text x="175" y="118" fill="#6B645E">N tile passports</text><text x="55" y="118" fill="#5E5191">1 leader</text><text x="260" y="150" fill="#276b45" font-size="11">total sequence length = N + 1</text></g></svg>
%%%

**What the team-captain picture gets right:** the captain leads the line, isn't a normal player, and exists to gather and speak for the whole team — exactly the `[CLS]` token's role of summarizing the image. **Where it breaks down:** a captain already knows the game from experience, but our `[CLS]` token starts completely blank; it only fills up with a summary *after* the tiles share information later (that's attention's job, tomorrow).

#### One tiny change to the count
Adding the captain does one small, tidy thing to our sequence: it grows by one. So the line the Transformer reads is no longer `N` tokens — it's `N + 1`. For our `224 × 224`, `P = 16` picture that's `196 + 1 = 197` tokens. Like the tile passports, the `[CLS]` token is also a length-`D` vector, and it too is *learned* during training — the model figures out what makes a good "blank leader" to start from. Small change, big payoff later: it gives the model one clean place to read off "what is this a picture of?"

@@@ concept id=c5 tag="Stamp the origin" title="Puzzle: the passports forgot where their tile sat" gotit="Got positions"
Time for a puzzle hidden inside what we just built. When we lined the passports up into a plain sequence, we quietly threw something away: **where each tile used to sit** in the picture. The line just says "passport, passport, passport…" — nothing on the passport says which one was the top-left sky and which was the bottom-right grass.

Here's why that's a real problem. A passport with **no country-of-origin stamp** is useless for knowing where its holder came from. If you shuffled a stack of blank-origin passports, you could not sort them back — the passport itself carries no "from here" mark. Our tile passports have exactly this hole: shuffle the line and the model literally cannot tell the difference, because a plain vector carries no sense of position.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="On the left, a row of passports with a blank origin field and question marks, so you cannot tell where each tile sat. An arrow points to the right, where each passport now has an origin stamp like R1C1, R1C2, R2C1, so the model can recover top, bottom, left, and right."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A passport with no origin stamp can't say where its tile sat</text><text x="120" y="38" fill="#C93B3B">no origin → where did this tile sit?</text><g transform="translate(35,50)"><rect x="0" y="0" width="42" height="34" fill="#EFEAF0" stroke="#9A938A"/><rect x="52" y="0" width="42" height="34" fill="#EFEAF0" stroke="#9A938A"/><rect x="104" y="0" width="42" height="34" fill="#EFEAF0" stroke="#9A938A"/><rect x="156" y="0" width="42" height="34" fill="#EFEAF0" stroke="#9A938A"/><text x="21" y="22" fill="#C93B3B">?</text><text x="73" y="22" fill="#C93B3B">?</text><text x="125" y="22" fill="#C93B3B">?</text><text x="177" y="22" fill="#C93B3B">?</text></g><text x="258" y="70" fill="#9A5A12" font-size="13">→ stamp origin →</text><text x="405" y="38" fill="#276b45">each passport stamped with its spot</text><g transform="translate(322,50)"><rect x="0" y="0" width="42" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="52" y="0" width="42" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="104" y="0" width="42" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="156" y="0" width="42" height="34" fill="#EAF5EE" stroke="#2D8B55"/><text x="21" y="22" fill="#1a5c38" font-size="9">R1C1</text><text x="73" y="22" fill="#1a5c38" font-size="9">R1C2</text><text x="125" y="22" fill="#1a5c38" font-size="9">R2C1</text><text x="177" y="22" fill="#1a5c38" font-size="9">R2C2</text></g><text x="260" y="140" fill="#6B645E">passport + origin stamp = "this tile, and it sat HERE"</text></g></svg>
%%%

**What the origin-stamp picture gets right:** a passport with no origin field can't tell you where its holder came from, and no shuffling recovers it — exactly our tile problem. **Where it breaks down:** a real passport's origin is written by a country; our tiles' "origin" has to be *learned* by the model, because there's no clerk to write it in.

#### The word for this: permutation-invariance
This "shuffle and nothing changes" property has a name: the token sequence is [[permutation-invariant||A model is permutation-invariant if shuffling the order of its inputs gives the exact same result. A raw Transformer treats its tokens as an unordered bag until you add position information.]]. In plain words, a raw Transformer treats its tokens as an *unordered bag* — it can't tell top-left from bottom-right, because nothing in the numbers says which came first. For text that would scramble a sentence; for our image it means the model can't tell up from down. That's the puzzle we have to solve.

#### The fix: add a learned position stamp to every passport
The cure is simple and elegant. To each tile's passport we **add** a small extra vector that means "you sat *here*." That added stamp is a [[positional embedding||A position vector added to each patch passport so the model can recover where each tile sat. In a vanilla Vision Transformer it is learned during training — one vector per slot.]]. Because the model *learns* these stamps during training, we call them *learnable* positional embeddings — the model works out on its own what "top-left" versus "bottom-right" should feel like.

The main-flow rule is one gentle line: **final input for a tile = its patch embedding + its positional embedding.** Both are length-`D` vectors, so they add cleanly, number by number — we *add* the stamp, we don't glue it on the end. Now predict, then reveal: two tiles with the *identical* pixels but in *different* spots — after we add positions, do they leave the model looking the same or different?

%%% demo id=posadd label="predict, then reveal"
code: same tile pixels, two different slots:  passport + pos[slot 3]   vs   passport + pos[slot 40]
out: slot 3:  [0.4,-0.2,0.9,…] + [0.1, 0.0,-0.3,…] = [0.5,-0.2, 0.6,…]
     slot 40: [0.4,-0.2,0.9,…] + [-0.2,0.3, 0.1,…] = [0.2, 0.1, 1.0,…]
take: <b>Same pixels, now clearly DIFFERENT inputs.</b> Because each slot adds a different learned position stamp, two identical-looking tiles in different spots no longer look the same to the model — so it can finally tell top from bottom, left from right. That's permutation-invariance, cured by one added vector.
%%%

That one addition is the whole remedy for the lost-position puzzle — and it's why ViT can tell a face from an upside-down face. **You just closed the biggest hole in the front door.**

@@@ concept id=c6 tag="The one big dial" title="Puzzle: how big should each tile be?" gotit="Got the P trade-off"
There's one number we've been treating as given, and it's secretly the most important dial in the whole front door: the **patch size** `P` — how many pixels wide each tile is. Should tiles be big or small? It turns out both extremes hurt, in opposite ways, and finding the sweet spot is a real design choice.

Think of cutting a pizza to share. Cut it into a **few huge slices** and each slice is easy to hand out (little work) — but each slice mashes together toppings from a big area, so you lose the fine detail of what's where. Cut it into **tiny slivers** and you can see every topping precisely — but now there are *so many* slivers that serving them all takes forever. Tile size is that same trade-off: big tiles are cheap but blurry, tiny tiles are sharp but expensive.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Two pizzas showing the trade-off. On the left, a pizza cut into a few huge slices, labelled big tiles: cheap but you lose fine detail. On the right, the same pizza cut into many tiny slivers, labelled small tiles: sharp detail but far more slices to serve, more work."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Slicing the pizza: few big slices vs many tiny slivers</text><g transform="translate(120,95)"><circle r="55" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><path d="M0 0 L0 -55 M0 0 L52 18 M0 0 L-32 45 M0 0 L-44 -33" stroke="#9A5A12" stroke-width="1.5"/><circle cx="20" cy="-20" r="5" fill="#C93B3B"/><circle cx="-15" cy="18" r="5" fill="#C93B3B"/></g><text x="120" y="165" fill="#8A6D3B">big tiles (large P)</text><text x="120" y="34" fill="#6B645E" font-size="9">cheap, but blurry detail</text><g transform="translate(390,95)"><circle r="55" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><g stroke="#9A5A12" stroke-width="0.8"><path d="M0 0 L0 -55"/><path d="M0 0 L28 -47"/><path d="M0 0 L48 -27"/><path d="M0 0 L55 0"/><path d="M0 0 L48 27"/><path d="M0 0 L28 47"/><path d="M0 0 L0 55"/><path d="M0 0 L-28 47"/><path d="M0 0 L-48 27"/><path d="M0 0 L-55 0"/><path d="M0 0 L-48 -27"/><path d="M0 0 L-28 -47"/></g></g><text x="390" y="165" fill="#276b45">small tiles (small P)</text><text x="390" y="34" fill="#6B645E" font-size="9">sharp, but many slices = more work</text></g></svg>
%%%

**What the pizza picture gets right:** bigger pieces are less work but blur what's inside; smaller pieces show detail but there are far more to handle — exactly the patch-size trade-off. **Where it breaks down:** you can eat a pizza slice of any size, but our tiles must be equal squares on a fixed grid, and (for math to stay tidy) `P` should divide the image evenly.

#### Watch the cost explode as tiles shrink
Here's the sharp edge of the puzzle. Shrinking `P` doesn't just add a *few* tokens — it makes the sequence length `N` blow up, and a Transformer's work grows roughly with `N × N` (each token compares with every other). So halving `P` doesn't double the cost; it can *quadruple the tokens* and hit the work far harder. Predict first: on a `224 × 224` image, going from `P = 16` down to `P = 8` — how many more tokens?

%%% demo id=ptrade label="predict, then reveal"
code: N = (224/P)² tokens:   P=32   vs   P=16   vs   P=8
out: P=32 → 7×7   = 49 tokens
     P=16 → 14×14 = 196 tokens
     P=8  → 28×28 = 784 tokens
take: <b>Smaller tiles → far MORE tokens.</b> Halving P from 16 to 8 does NOT double the count — it roughly QUADRUPLES it (196 → 784), because you gain tokens both across AND down. And since a Transformer's work grows ~N², 4× more tokens costs vastly more than 4× the compute. That's why tiny patches get expensive fast.
%%%

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three bars showing token count growing as patch size shrinks on a 224 by 224 image. Patch size 32 gives 49 tokens, a short bar. Patch size 16 gives 196 tokens, a medium bar. Patch size 8 gives 784 tokens, the tallest bar. Smaller tiles mean far more tokens and much more compute."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Same 224×224 image — smaller P → far more tokens N</text><line x1="60" y1="165" x2="470" y2="165" stroke="#E5DFD6" stroke-width="1.5"/><rect x="90" y="152" width="70" height="13" fill="#2D8B55"/><text x="125" y="182" fill="#6B645E">P=32</text><text x="125" y="144" fill="#276b45">49</text><rect x="225" y="108" width="70" height="57" fill="#7C6DAA"/><text x="260" y="182" fill="#6B645E">P=16</text><text x="260" y="100" fill="#5E5191">196</text><rect x="360" y="47" width="70" height="118" fill="#C99A12"/><text x="395" y="182" fill="#6B645E">P=8</text><text x="395" y="39" fill="#8A6D3B">784</text><text x="480" y="118" fill="#C93B3B" font-size="9" text-anchor="end">more tokens</text><text x="480" y="131" fill="#C93B3B" font-size="9" text-anchor="end">≈ N² work</text></g></svg>
%%%

#### The remedy: pick P as a tuned trade-off (usually 16)
So what do we do? We don't max out either end — we **choose `P` as a tuned trade-off**: small enough to catch the detail we care about, big enough that the sequence stays affordable. In practice the classic Vision Transformer lands on `P = 16` as a sweet spot (that's the "16" in the paper's nickname, "ViT-B/16"). Some models go finer (`P = 8`) when detail matters and they can pay for it, or coarser (`P = 32`) when they need speed. There's no single right answer — `P` is the first dial you reach for, and now you know exactly what it trades.

@@@ concept id=c7 tag="What the front door can't do" title="What patch embedding still can't do — and the fixes" gotit="Got the limits"
Being honest about limits is what makes an engineer trustworthy — and each one here is a little puzzle with a satisfying answer already waiting. Picture our passport-holders as **travelers who have just cleared the border but haven't left the arrivals hall**. Everyone's stamped and in line; nobody has talked to anyone or gone anywhere yet. That one picture explains the whole list of limits at once.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="Three panels of limits. One: two passport-holders side by side with no arrow between them, labelled no talking yet. Two: an object shape split awkwardly across a hard grid line into two tiles. Three: a small grid of learned position stamps that does not stretch to cover a larger image grid."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Three honest limits of the patch-embedding front door</text><g transform="translate(20,36)"><rect x="0" y="0" width="46" height="46" fill="#EAF5EE" stroke="#2D8B55"/><rect x="60" y="0" width="46" height="46" fill="#EAF5EE" stroke="#2D8B55"/><text x="23" y="30" fill="#1a5c38" font-size="16">🛂</text><text x="83" y="30" fill="#1a5c38" font-size="16">🛂</text><text x="53" y="28" fill="#C93B3B" font-size="14">✕</text><text x="53" y="72" fill="#C93B3B">no talking yet</text></g><g transform="translate(200,36)"><rect x="0" y="0" width="106" height="46" fill="#FCF3DC" stroke="#C99A12"/><line x1="53" y1="0" x2="53" y2="46" stroke="#C93B3B" stroke-width="2"/><circle cx="53" cy="23" r="20" fill="none" stroke="#5E5191" stroke-width="2.5"/><text x="53" y="72" fill="#C93B3B">object split by a hard edge</text></g><g transform="translate(392,36)"><rect x="0" y="0" width="40" height="40" fill="none" stroke="#5E5191" stroke-dasharray="3,2"/><path d="M13 0 V40 M26 0 V40 M0 13 H40 M0 26 H40" stroke="#5E5191" stroke-width="0.6"/><rect x="0" y="0" width="70" height="70" fill="none" stroke="#C93B3B" stroke-width="1.5"/><text x="35" y="86" fill="#C93B3B">stamps don't stretch</text></g><text x="260" y="155" fill="#6B645E">everyone's cleared the border — but nobody has talked or moved yet</text><text x="260" y="172" fill="#9A938A">talking = attention, which is tomorrow's job</text></g></svg>
%%%

**What the arrivals-hall picture gets right:** everyone is stamped and lined up, but nothing useful happens until they *talk* and *move* — the front door seats the travelers; it doesn't start the conversation. **Where it breaks down:** real travelers can wander and chat freely, but our tiles can only "talk" through one specific mechanism (attention) that we haven't switched on yet.

#### Limit 1 · Each passport is made alone (no talking yet)
The patch embedding builds every passport **independently** — one tile's pixels, times `W`, and that's it. Nothing has mixed information *between* tiles, so no tile knows what the rest of the picture looks like. A tile of blue sky and a tile of blue water get near-identical passports, because neither has seen its neighbors. That between-tile mixing — letting passports compare notes — is exactly the job of [[self-attention||The mechanism that lets tokens share information with each other. It is what turns a bag of independent tile passports into an understood picture. Tomorrow's lesson.]], and it's day 3's whole story. Today we've only *stamped and seated* the travelers.

#### Limit 2 · A square grid can split an object
The stamp always presses on the same fixed lines, no matter what's in the picture. So an object that doesn't line up with the grid — a face, a wheel — can get **sliced across two tiles**, with half its shape in one passport and half in the next. A rigid square grid simply can't bend to fit whatever shape happens to be there. The known softening: use an **overlapping or convolutional patch stem** — let the stamp windows overlap a little, or run a few gentle conv layers first, so object edges aren't chopped so harshly. (You'll meet the deep form of this later; today just know the fix exists.)

#### Limit 3 · Position stamps are tied to the training size
We *learned* one position stamp per slot — but only as many slots as the training images had. Feed the model a **bigger** picture (more tiles, more slots) and there simply aren't enough learned stamps to go around; a smaller picture leaves some unused. So the learned stamps don't automatically stretch or shrink to a new image size. The known fix: **interpolate the positional embeddings** — gently stretch the grid of learned stamps to the new number of slots, like resizing a photo, so a model trained at one resolution can be used at another. (Deep dive on this comes in a later ViT day.)

!!! c-info 🧭
<b>Optional (skippable) — the whole map of limits and fixes.</b> None of these are dead ends. Tiles start "talking" via <b>self-attention</b> (day 3). A rigid grid that splits objects is softened by an <b>overlapping / convolutional patch stem</b>. And the fixed-size position stamps are stretched to a new resolution by <b>interpolating the positional embeddings</b>. There's even a way to train the patch embedding by hiding tiles and asking the model to fill them back in (<b>masked-patch pretraining</b>, like MAE) — a later day. Today's job was just to get the picture <i>through the front door</i>; making the tiles smart is what comes next.
!!!

Here's the same box as a map — each limit on the left, the fix waiting for it on the right:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three rows mapping each limit to its fix. Row one: passports are made alone with no talking, fixed by self-attention on day 3. Row two: a rigid grid can split an object, softened by an overlapping or convolutional patch stem. Row three: position stamps are tied to the training size, fixed by interpolating the stamps to the new resolution."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Every limit → its known fix (all coming up)</text><text x="120" y="36" text-anchor="middle" fill="#C93B3B">the limit today</text><text x="400" y="36" text-anchor="middle" fill="#276b45">the fix, later</text><g transform="translate(30,46)"><rect x="0" y="0" width="180" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="21" text-anchor="middle" fill="#8a3b3b">1 · passports made alone</text></g><text x="222" y="67" fill="#9A5A12" font-size="13">→</text><g transform="translate(250,46)"><rect x="0" y="0" width="240" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="120" y="21" text-anchor="middle" fill="#1a5c38">self-attention (day 3)</text></g><g transform="translate(30,96)"><rect x="0" y="0" width="180" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="21" text-anchor="middle" fill="#8a3b3b">2 · grid splits an object</text></g><text x="222" y="117" fill="#9A5A12" font-size="13">→</text><g transform="translate(250,96)"><rect x="0" y="0" width="240" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="120" y="21" text-anchor="middle" fill="#1a5c38">overlapping / conv patch stem</text></g><g transform="translate(30,146)"><rect x="0" y="0" width="180" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="21" text-anchor="middle" fill="#8a3b3b">3 · stamps tied to image size</text></g><text x="222" y="167" fill="#9A5A12" font-size="13">→</text><g transform="translate(250,146)"><rect x="0" y="0" width="240" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="120" y="21" text-anchor="middle" fill="#1a5c38">interpolate the position stamps</text></g></g></svg>
%%%

These aren't flaws to fear — they're the map of what you'll learn next. **You can now say exactly what the patch embedding does and, just as importantly, what it doesn't** — that honesty is a real staff-level habit.

@@@ concept id=c8 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — every tile now carries a full **passport** and the picture is ready to be read. Here's one clean way to hold the whole day: picture a short **airport line**, where a raw tile walks up at one end and a fully-stamped, position-marked passport leaves at the other, ready to board. Each station only makes sense because of the one before it — that's exactly how today stacked up. **Where the airport picture breaks down:** an airport processes you once and you're gone, but these steps are a recipe you'll revisit and tune (the patch size `P`, the length `D`) for years. Read the stations, then keep the cheat-sheets as your map.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="An airport line with five stations in order. Station one a raw tile. Station two the passport machine multiplies by W. Station three the passport is a length-D vector. Station four passports line up into a sequence with a CLS leader token at the front. Station five each passport gets a position stamp added, ready for the Transformer."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Today = a 5-station line: raw tile → boardable passport</text><line x1="20" y1="120" x2="500" y2="120" stroke="#E5DFD6" stroke-width="6"/><g transform="translate(30,62)"><rect x="0" y="0" width="56" height="46" fill="#FCF3DC" stroke="#C99A12"/><path d="M18 0 V46 M37 0 V46 M0 15 H56 M0 31 H56" stroke="#9A5A12" stroke-width="0.6"/><text x="28" y="70" fill="#6B645E">1 · raw tile</text></g><text x="100" y="88" fill="#9A5A12">→</text><g transform="translate(114,62)"><rect x="0" y="0" width="56" height="46" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="28" y="28" fill="#C93B3B">× W</text><text x="28" y="70" fill="#6B645E">2 · multiply</text></g><text x="184" y="88" fill="#9A5A12">→</text><g transform="translate(198,78)"><rect x="0" y="0" width="56" height="14" fill="#EAF5EE" stroke="#2D8B55"/><text x="28" y="52" fill="#276b45">3 · passport</text><text x="28" y="63" fill="#276b45">length D</text></g><text x="268" y="88" fill="#9A5A12">→</text><g transform="translate(282,76)"><rect x="0" y="0" width="12" height="16" fill="#EFEAF7" stroke="#5E5191"/><rect x="14" y="0" width="14" height="16" fill="#FCF3DC" stroke="#C99A12"/><rect x="30" y="0" width="14" height="16" fill="#FCF3DC" stroke="#C99A12"/><rect x="46" y="0" width="14" height="16" fill="#FCF3DC" stroke="#C99A12"/><text x="30" y="52" fill="#8A6D3B">4 · sequence</text><text x="30" y="63" fill="#5E5191">+ [CLS]</text></g><text x="352" y="88" fill="#9A5A12">→</text><g transform="translate(366,70)"><rect x="0" y="0" width="56" height="14" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="20" width="56" height="10" fill="#EFEAF7" stroke="#7C6DAA"/><text x="28" y="58" fill="#276b45">5 · add</text><text x="28" y="69" fill="#5E5191">position</text></g><circle cx="480" cy="84" r="7" fill="#C93B3B"/><text x="480" y="70" fill="#6B645E">→ Transformer 🏁</text></g></svg>
%%%

The five stations, in order:
- **Station 1 · A raw tile.** One `P × P` square of pixels from yesterday's patchify (`P × P × C` numbers).
- **Station 2 · The passport machine.** Multiply each flattened tile by one shared learned matrix `W` — a *linear projection*. This is exactly a convolution with kernel `= stride = P`.
- **Station 3 · The passport.** Out comes the tile's *patch embedding*: a tidy fixed-length vector of length `D`. Same machine (weight sharing) for every tile.
- **Station 4 · Line them up + add the captain.** Lay the passports in reading order into the *sequence* a Transformer eats (`N` tokens), and pin one learnable `[CLS]` leader token at the front → `N + 1` tokens.
- **Station 5 · Stamp the origin.** *Add* a learned *positional embedding* to each passport so the model knows where each tile sat — the cure for permutation-invariance.
- **And the honest bit:** passports are made alone (no tile-to-tile talking yet → attention, day 3), a rigid grid can split objects (→ overlapping/conv stem), and position stamps are tied to the training size (→ interpolate them). And `P` is the one big dial: small = sharp but expensive, big = cheap but blurry.

#### Cheat-sheet · the pieces at a glance
%%% table
:: Piece :: What it is :: One-line reason
Patch embedding :: flattened tile × one shared matrix W → length D :: gives each tile a tidy fixed-length "passport"
Conv view :: the same op = convolution, kernel = stride = P :: links the patch embedding back to CNNs
Sequence :: the passports in reading order, N tokens :: the exact input format a Transformer expects
[CLS] token :: extra learnable vector at the front (length D) :: makes it N+1; later summarizes the whole image
Positional embedding :: a learned vector ADDED to each passport :: cures permutation-invariance; says where each tile sat
Patch size P :: side length of each tile (typically 16) :: the dial: small = sharp+costly, big = cheap+blurry
%%%

#### Cheat-sheet · the words you met today
%%% jargon
patch embedding | the length-D vector a tile becomes after the shared linear projection
linear projection | multiply a vector by one shared matrix W to make a fixed-length vector
embedding dimension (D) | how many numbers are in each token's passport
weight sharing | using the same learned matrix W for every tile, not one per position
convolution (kernel=stride=P) | the equivalent view of patch embedding — a stamp that reads P×P and steps P
token | one item in the sequence — here, one tile's passport
sequence | the ordered list of tokens a Transformer reads
class token / [CLS] | an extra learnable leader vector at the front that will later summarize the image
positional embedding | a learned vector added to each token so it remembers where its tile sat
permutation-invariant | shuffling the tokens changes nothing — the reason we must add positions
patch size (P) | the side length of each tile; the dial for detail-vs-compute
overlapping / conv patch stem | a softer patch scheme so a rigid grid doesn't chop objects
interpolate positional embeddings | stretch the learned position stamps to a new image resolution
%%%

That's the whole day. Next you'll take these embedded, position-stamped passports and finally let them *talk* — the **self-attention** that turns a line of independent tiles into an understood picture.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is a "patch embedding," in one line? | a:2 | The grid of raw pixels in a tile | A blurred version of the tile | The tidy fixed-length vector a tile becomes after multiplying by one shared learned matrix W | The position of the tile in the image | fb: The patch embedding is one linear projection: flatten a tile, multiply by the shared matrix W, and get a fixed-length "passport" vector of length D. The same W is used for every tile (weight sharing).
q: People say the patch embedding is "a convolution with kernel = stride = P." What does that mean? | a:1 | It blurs the image before cutting it | A stamp reads a P×P window and slides P pixels each step, so windows tile the image with no gaps or overlaps | It uses a different matrix for each tile | It only works on grayscale images | fb: Kernel = P means the stamp reads a P×P square; stride = P means it steps over by exactly P pixels — so the presses tile the image edge to edge. Doing patchify + flatten + multiply-by-W in one motion IS that convolution.
q: After lining the passports up into a plain sequence, what problem appears and how is it fixed? | a:1 | The passports lose their color, fixed by adding channels | The sequence is permutation-invariant (no sense of WHERE each tile sat), fixed by ADDING a learned positional embedding to each passport | The passports get too long, fixed by flattening again | The model runs out of memory, fixed by the [CLS] token | fb: A raw Transformer treats its tokens as an unordered bag — shuffle them and nothing changes. We ADD a learned positional embedding (a length-D vector) to each passport so it remembers top/bottom, left/right.
q: On a 224×224 image, why is choosing patch size P a real trade-off? | a:2 | Bigger P always wins because it's simpler | Smaller P is always best because more tokens is always better | Smaller P catches finer detail but roughly quadruples the token count N (196→784 from P=16 to P=8) and blows up ~N² compute; bigger P is cheap but blurry — so P (typically 16) is tuned | P has no effect on speed | fb: Halving P from 16 to 8 gives ~4× the tokens (196→784) because you gain tokens across AND down, and Transformer work grows ~N². So small P = sharp but costly, big P = cheap but blurry. Vanilla ViT tunes P to a sweet spot, often 16.
%%%

@@@ produce id=produce tag="Produce" title="Build a patch embedding with your own hands" gotit="Done"
Time to see today's big idea with your own eyes. You'll take a tiny fake "image" (just a grid of numbers), cut it into tiles, run each tile through a shared matrix `W` to get its length-`D` passport, stack them into a sequence, prepend a `[CLS]` token, and add positional embeddings — printing the **shape at every step** so you can watch a picture turn into `N + 1` token-vectors. **Predict first:** for a `4 × 4` image with `P = 2` and `D = 8`, how many patch tokens `N` do you get, and what's the sequence length *after* adding the `[CLS]` token? Then run it and **watch** the shapes confirm it. Pick one path.

#### Option A · write it yourself
Create `sessions/m05b-vision-transformer/day-02-patch-embeddings/experiment.py`. Make a fake grayscale image with `img = np.arange(4*4).reshape(4, 4)` (`H = W = 4`). Cut it into `P × P` tiles with `P = 2` and **flatten** each tile to a length-`P*P` vector (here `4`). Make a shared random matrix `W` of shape `(P*P, D)` with `D = 8` and multiply every flattened tile by the **same** `W` to get its length-`D` patch embedding — print the shape at each step. Stack the passports into a `(N, D)` sequence and print `N` from the formula `(H//P)*(W//P)`. Now **prepend** a random `[CLS]` vector of length `D` (sequence becomes `(N+1, D)`) and **add** a random positional-embedding array of shape `(N+1, D)` to the whole sequence. **Observe** that `N = (4/2)*(4/2) = 4` and the final sequence shape is `(5, 8)` — five tokens, each length 8. Run with `python3 sessions/m05b-vision-transformer/day-02-patch-embeddings/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 5b Day 2 artifact.

Create sessions/m05b-vision-transformer/day-02-patch-embeddings/experiment.py that, with a comment on each step and printing shapes at every step:
1. Makes a fake grayscale image img = np.arange(4*4).reshape(4, 4) and prints its shape.
2. Cuts it into non-overlapping P×P tiles with P=2 and flattens each tile to length P*P (=4). Prints num_patches N from the formula (H//P)*(W//P) and confirms it matches the number of tiles produced.
3. Builds ONE shared random matrix W of shape (P*P, D) with D=8, and multiplies every flattened tile by the SAME W to get a (N, D) array of patch embeddings — prints the shape and notes this is weight sharing (a linear projection).
4. Prepends a random [CLS] vector of length D, making the sequence (N+1, D), and prints the shape (expect (5, 8)).
5. Adds a random positional-embedding array of shape (N+1, D) to the whole sequence (elementwise add, same shape out) and prints the final shape.
6. As a sanity check, shows that two tiles with identical pixels but different slots become DIFFERENT rows after adding positional embeddings (permutation-invariance cured).
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (a picture becoming tokens)
- The formula and the actual tiles **agree**: `N = (H/P) × (W/P) = 2 × 2 = 4` patch tokens.
- Each flattened tile (length `4`) becomes a length-`D = 8` passport after multiplying by the **same** shared `W` — that's the linear projection, printed shape `(4, 8)`.
- The moment to **watch**: prepend the `[CLS]` token and the sequence jumps from `(4, 8)` to `(5, 8)` — `N` becomes `N + 1`.
- After **adding** the positional embeddings, two tiles with identical pixels in different slots come out as **different** rows — the permutation-invariance cure you learned, seen in the numbers.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05b-vision-transformer/day-02-patch-embeddings/log.md`: (1) in one sentence, what a patch embedding does to one tile; (2) why we ADD a positional embedding instead of just lining the tiles up; (3) one thing the patch embedding alone CANNOT do yet, and where it gets fixed.

@@@ fin
