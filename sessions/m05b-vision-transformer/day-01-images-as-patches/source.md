---
quest_id: wf7-d01-patches
mode: concept
donor: v9-base.donor
page_title: "Module 5b · Day 1 — Images as Patches"
module_label: "Module 05b · Represent · Day 1"
title: "Images as Patches"
subtitle: "How a Transformer Sees a Picture"
brand_sub: "Foundations · M5b Day 1"
spine: "tile"
nav_prev_href: "../../m05a-text-transformer/review.html"
nav_prev_label: "M5a · Review Gate"
nav_next_href: "../day-02-patch-embeddings/lesson.html"
nav_next_label: "Patch Embeddings"
fin_title: "Module 5b · Day 1 complete! 🏆"
fin_body: "Nice work — you've seen how a Vision Transformer starts: chop a picture into a grid of small square <b>tile</b>s (patches), then line those tiles up into a sequence — exactly like the tokens a text transformer reads. A picture becomes a sentence of tiles.<br>Next up: <b>Patch Embeddings</b>."
notebook_yardstick: null
---

@@@ hero
@lede Grab a chocolate bar and snap it along the grooves — you don't eat it in one giant bite, you break it into neat little squares first. A Vision Transformer does the exact same trick with a picture: before it "reads" an image, it snaps the picture into a grid of small square **tile**s. And here's the surprise — once a picture is a grid of tiles, the model treats each tile just like a *word*, and reads the whole image like a sentence. That's the very same kind of model behind image tools you've heard of, from photo search to the vision side of GPT-4. Today you'll see the first move that makes all of it possible: turning a picture into a row of tiles.
@goal Together we'll turn a picture into something a Transformer can read. You'll see an image as a grid of dots, snap it into square **tile**s, flatten each tile into a list of numbers so it acts like a word, line the tiles up into a sentence, count them with one tiny formula, stamp each tile into a tidy vector, and tag every tile with *where it sat*. We'll also meet — gently, as little puzzles — the things this first step still can't do. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="A picture is a grid" title="A picture is really a grid of tiny dots" gotit="Got the grid"
Before we can chop a picture up, let's see what a picture *is* to a computer. Hold a magnifying glass over a phone screen and you'll spot it: the picture is built from thousands of tiny colored dots, packed into a neat grid of rows and columns. Each dot is called a [[pixel||One tiny colored dot in an image. A picture is a grid of pixels — rows and columns of them.]]. A picture is nothing more than a grid of pixels.

Think of it like a huge sheet of **graph paper**, where you've filled in each little square with one color. Step back and the squares blur into a photo; lean in and you see the grid again.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A photo on the left blurs into a smiley face. Zooming in on the right shows it is really a grid of colored squares, each square labelled as one pixel."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A picture = a grid of tiny colored dots (pixels)</text><circle cx="95" cy="105" r="52" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><circle cx="78" cy="92" r="6" fill="#3A342E"/><circle cx="112" cy="92" r="6" fill="#3A342E"/><path d="M74 118 Q95 138 116 118" fill="none" stroke="#3A342E" stroke-width="3"/><text x="95" y="176" text-anchor="middle" fill="#6B645E">step back → a photo</text><text x="185" y="108" fill="#B8AEA2" font-size="18">→</text><g transform="translate(230,50)"><rect x="0" y="0" width="200" height="120" fill="none" stroke="#B8AEA2"/><rect x="0" y="0" width="40" height="24" fill="#F6EFCF"/><rect x="40" y="0" width="40" height="24" fill="#FCF3DC"/><rect x="80" y="0" width="40" height="24" fill="#3A342E"/><rect x="120" y="0" width="40" height="24" fill="#3A342E"/><rect x="160" y="0" width="40" height="24" fill="#FCF3DC"/><rect x="0" y="24" width="40" height="24" fill="#FCF3DC"/><rect x="80" y="24" width="40" height="24" fill="#F6EFCF"/><rect x="0" y="48" width="200" height="24" fill="#FCF3DC" opacity="0.5"/><rect x="0" y="72" width="40" height="24" fill="#3A342E"/><rect x="160" y="72" width="40" height="24" fill="#3A342E"/><rect x="40" y="96" width="120" height="24" fill="#3A342E"/><path d="M0 0 H200 M0 24 H200 M0 48 H200 M0 72 H200 M0 96 H200 M0 120 H200 M0 0 V120 M40 0 V120 M80 0 V120 M120 0 V120 M160 0 V120 M200 0 V120" stroke="#B8AEA2" stroke-width="0.6" fill="none"/></g><text x="330" y="185" text-anchor="middle" fill="#6B645E">lean in → a grid of squares</text></g></svg>
%%%

**What the graph-paper picture gets right:** a photo really is a fixed grid of squares, each holding one color, exactly like filled-in graph paper. **Where it breaks down:** on graph paper you pick the colors by hand, but in a real photo a camera measures them — and each square isn't one color-word, it's a few *numbers* (we'll unpack that in a moment).

#### How big is the grid?
We describe an image by its **height** and **width** in pixels — say `224 × 224`, which means 224 rows of dots and 224 columns. That's about fifty thousand pixels in one small square photo. Feeding those to a model one dot at a time would be like reading a book one *letter* at a time — painfully slow, and you'd lose the shapes. So a Vision Transformer does **not** hand pixels to the model one by one. It first groups them into bigger, more meaningful chunks — the **tile**s we're about to make. Hold that thought: the whole day is about turning this giant grid of dots into a short, tidy row of tiles.

@@@ concept id=c2 tag="Snap into tiles" title="Patchify — snap the picture into square tiles" gotit="Got patchify"
Here's today's star move. We take that big grid of pixels and cut it into a smaller grid of **square tile**s. Remember the chocolate bar from the start? You snap it along the grooves into equal little squares. We do exactly that to the image: slice it into equal, non-overlapping squares. Each square is called a [[patch||A small square tile cut from an image, e.g. 16×16 pixels. A Vision Transformer breaks an image into a grid of these before reading it.]] — a **tile** of the picture. The whole step has a name: [[patchify||To cut an image into a grid of equal, non-overlapping square patches (tiles). The first step of a Vision Transformer.]].

Picture a bathroom floor covered in square **tile**s laid edge to edge — no gaps, no overlaps, each tile the same size. That's a patchified image.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="On the left, a plain image square. An arrow labelled patchify points right to the same square now sliced into a four by four grid of sixteen equal tiles, laid edge to edge with no gaps or overlaps, like floor tiles."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Patchify = snap the image into equal square tiles</text><rect x="40" y="45" width="120" height="120" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><circle cx="100" cy="100" r="34" fill="#F6EFCF" stroke="#C99A12"/><circle cx="88" cy="92" r="4" fill="#3A342E"/><circle cx="112" cy="92" r="4" fill="#3A342E"/><path d="M86 112 Q100 124 114 112" fill="none" stroke="#3A342E" stroke-width="2"/><text x="100" y="185" text-anchor="middle" fill="#6B645E">one image</text><text x="190" y="108" text-anchor="middle" fill="#9A5A12" font-size="11">patchify →</text><g transform="translate(280,45)"><rect x="0" y="0" width="120" height="120" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><circle cx="60" cy="55" r="34" fill="#F6EFCF" stroke="#C99A12"/><circle cx="48" cy="47" r="4" fill="#3A342E"/><circle cx="72" cy="47" r="4" fill="#3A342E"/><path d="M46 67 Q60 79 74 67" fill="none" stroke="#3A342E" stroke-width="2"/><path d="M30 0 V120 M60 0 V120 M90 0 V120 M0 30 H120 M0 60 H120 M0 90 H120" stroke="#9A5A12" stroke-width="1.4" fill="none"/></g><text x="340" y="185" text-anchor="middle" fill="#276b45">a grid of 16 tiles</text><text x="452" y="80" fill="#9A5A12">4 × 4</text><text x="452" y="96" fill="#9A5A12">tiles</text></g></svg>
%%%

**What the floor-tile picture gets right:** the tiles are equal squares, laid edge to edge with no gaps and no overlaps — a perfect grid, just like patches. **Where it breaks down:** floor tiles are decoration you can arrange freely, but image patches must line up on a fixed grid that always starts at the top-left corner. No fancy patterns; just a plain grid.

#### The one number that controls it all: the patch size
The size of each tile has a name: the **patch size**, written `P`. If `P = 16`, every tile is a `16 × 16` block of pixels. Cut a `224 × 224` image into `16 × 16` tiles and you get a tidy `14 × 14` grid of tiles across the picture. Watch the cut happen, tile by tile:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A build-up in three steps. Step one: the full image with only the outer border. Step two: the vertical cut lines are added, splitting it into columns. Step three: the horizontal cut lines are added too, finishing a four by four grid of sixteen tiles."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Cutting a 224×224 image with patch size P — added one set of cuts at a time</text><g transform="translate(30,40)"><rect x="0" y="0" width="120" height="120" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="60" y="145" fill="#6B645E">1 · whole image</text><text x="60" y="162" fill="#9A938A">no cuts yet</text></g><text x="180" y="105" fill="#B8AEA2" font-size="16">→</text><g transform="translate(200,40)"><rect x="0" y="0" width="120" height="120" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><path d="M30 0 V120 M60 0 V120 M90 0 V120" stroke="#9A5A12" stroke-width="1.4"/><text x="60" y="145" fill="#6B645E">2 · slice columns</text><text x="60" y="162" fill="#9A938A">every P pixels across</text></g><text x="350" y="105" fill="#B8AEA2" font-size="16">→</text><g transform="translate(370,40)"><rect x="0" y="0" width="120" height="120" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><path d="M30 0 V120 M60 0 V120 M90 0 V120 M0 30 H120 M0 60 H120 M0 90 H120" stroke="#2D8B55" stroke-width="1.4"/><text x="60" y="145" fill="#276b45">3 · slice rows too</text><text x="60" y="162" fill="#276b45">16 finished tiles</text></g></g></svg>
%%%

You now have the single most important object in the whole lesson: a **grid of tiles**. Everything from here is about lining these tiles up and getting them ready for the model. Next we ask the fun question — *how does a tile become something the Transformer can actually read?*

@@@ concept id=c3 tag="A tile becomes a word" title="Flatten each tile into one long list of numbers" gotit="Got the flatten"
Here's the clever idea that makes Vision Transformers work at all. A text Transformer reads *words*. So to reuse that same machine on pictures, we make each **tile** *act like a word*. How? We turn the tile's little grid of pixels into one long list of numbers — a single row.

Think of a tile like a small **spreadsheet** of numbers. To store it in one line, you read it left-to-right, top-to-bottom — first row, then second row, and so on — and paste all the numbers into one long strip. That "unroll into a single row" step is called [[flatten||To turn a small 2-D grid of numbers into one long single-row list, by reading it row by row. It lets a square tile act like one word-vector.]]. The flattened strip is the tile's [[vector||A plain list of numbers in a fixed order. Here, one flattened tile — the image world's version of a word.]] — its word.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="On the left a small grid of numbers like a spreadsheet, arrows show reading it row by row, and on the right the numbers are pasted into one long single-row strip labelled one flattened tile equals one word-vector."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Flatten = read the tile row by row into one long strip</text><text x="70" y="42" text-anchor="middle" fill="#6B645E">a tile (grid)</text><g transform="translate(40,50)"><rect x="0" y="0" width="90" height="90" fill="#F6EFCF" stroke="#C99A12"/><path d="M30 0 V90 M60 0 V90 M0 30 H90 M0 60 H90" stroke="#C99A12" stroke-width="0.8"/><text x="15" y="20" text-anchor="middle" fill="#3A342E">7</text><text x="45" y="20" text-anchor="middle" fill="#3A342E">2</text><text x="75" y="20" text-anchor="middle" fill="#3A342E">9</text><text x="15" y="50" text-anchor="middle" fill="#3A342E">1</text><text x="45" y="50" text-anchor="middle" fill="#3A342E">5</text><text x="75" y="50" text-anchor="middle" fill="#3A342E">3</text><text x="15" y="80" text-anchor="middle" fill="#3A342E">4</text><text x="45" y="80" text-anchor="middle" fill="#3A342E">8</text><text x="75" y="80" text-anchor="middle" fill="#3A342E">6</text><path d="M85 15 H100 M85 45 H100 M85 75 H100" stroke="#9A5A12" stroke-width="0.8"/></g><text x="175" y="98" fill="#9A5A12" font-size="11">flatten →</text><g transform="translate(235,78)"><rect x="0" y="0" width="240" height="26" fill="#EAF5EE" stroke="#2D8B55"/><path d="M26.6 0 V26 M53.3 0 V26 M80 0 V26 M106.6 0 V26 M133.3 0 V26 M160 0 V26 M186.6 0 V26 M213.3 0 V26" stroke="#2D8B55" stroke-width="0.7"/><text x="13" y="18" text-anchor="middle" fill="#1a5c38">7</text><text x="40" y="18" text-anchor="middle" fill="#1a5c38">2</text><text x="66" y="18" text-anchor="middle" fill="#1a5c38">9</text><text x="93" y="18" text-anchor="middle" fill="#1a5c38">1</text><text x="120" y="18" text-anchor="middle" fill="#1a5c38">5</text><text x="146" y="18" text-anchor="middle" fill="#1a5c38">3</text><text x="173" y="18" text-anchor="middle" fill="#1a5c38">4</text><text x="200" y="18" text-anchor="middle" fill="#1a5c38">8</text><text x="226" y="18" text-anchor="middle" fill="#1a5c38">6</text></g><text x="355" y="126" text-anchor="middle" fill="#276b45">one flattened tile = one "word-vector"</text><text x="355" y="60" text-anchor="middle" fill="#6B645E" font-size="10">row 1 · row 2 · row 3, pasted end to end</text></g></svg>
%%%

**What the spreadsheet picture gets right:** a tile really is a little grid of numbers, and flattening is just reading it in a fixed order and writing it as one row. **Where it breaks down:** a spreadsheet row is meant for humans to read; this strip is meant for a model to crunch — the numbers themselves are pixel brightnesses, not tidy labels.

#### Why this is the whole trick
A word in a sentence is already just one vector (a list of numbers) to a Transformer. Now a **tile** is *also* just one vector. So a tile can slot into the exact same machinery that reads words — no new model needed. That's the beautiful part: **a picture-tile plays the same role a word plays.** The Transformer never has to know it's looking at an image; to it, every tile is simply a "word" in a strange new language made of squares.

@@@ concept id=c4 tag="A sentence of tiles" title="Line the tiles up into a sequence" gotit="Got the sequence"
We have a grid of tiles, and each tile is now a word-vector. The last step is to put them *in order*. A Transformer doesn't read a 2-D grid — it reads a **list**, one item after another. So we lay the tiles out in reading order: left-to-right along the top row, then the next row down, just like reading a page.

Picture a **comic strip**. The artist draws lots of separate panels, but to tell the story you read them in a fixed order — top row first, left to right, then the next row. Line our tiles up the same way and the picture becomes a story the model can read panel by panel. That ordered list has a name: a [[sequence||An ordered list of items, one after another — the input a Transformer expects. Here, the image's tiles laid out in reading order.]].

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A four by four grid of tiles on the left, numbered one through sixteen in reading order. Arrows show them being unrolled into a single horizontal row on the right, a sequence of tiles like a comic strip read panel by panel."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Read the grid top-to-bottom, left-to-right → one ordered row</text><g transform="translate(35,40)"><rect x="0" y="0" width="120" height="120" fill="#EAF5EE" stroke="#2D8B55"/><path d="M30 0 V120 M60 0 V120 M90 0 V120 M0 30 H120 M0 60 H120 M0 90 H120" stroke="#2D8B55" stroke-width="1"/><text x="15" y="20" fill="#1a5c38">1</text><text x="45" y="20" fill="#1a5c38">2</text><text x="75" y="20" fill="#1a5c38">3</text><text x="105" y="20" fill="#1a5c38">4</text><text x="15" y="50" fill="#1a5c38">5</text><text x="45" y="50" fill="#1a5c38">6</text><text x="75" y="50" fill="#1a5c38">7</text><text x="105" y="50" fill="#1a5c38">8</text><text x="15" y="80" fill="#1a5c38">9</text><text x="45" y="80" fill="#1a5c38">10</text><text x="75" y="80" fill="#1a5c38">11</text><text x="105" y="80" fill="#1a5c38">12</text><text x="15" y="110" fill="#1a5c38">13</text><text x="45" y="110" fill="#1a5c38">14</text><text x="75" y="110" fill="#1a5c38">15</text><text x="105" y="110" fill="#1a5c38">16</text><text x="60" y="140" fill="#6B645E">grid of tiles</text></g><text x="180" y="100" fill="#9A5A12" font-size="16">→</text><g transform="translate(205,75)"><rect x="0" y="0" width="300" height="26" fill="#FCF3DC" stroke="#C99A12"/><path d="M18.75 0 V26 M37.5 0 V26 M56.25 0 V26 M75 0 V26 M93.75 0 V26 M112.5 0 V26 M131.25 0 V26 M150 0 V26 M168.75 0 V26 M187.5 0 V26 M206.25 0 V26 M225 0 V26 M243.75 0 V26 M262.5 0 V26 M281.25 0 V26" stroke="#C99A12" stroke-width="0.6"/><text x="9" y="17" fill="#8A6D3B">1</text><text x="28" y="17" fill="#8A6D3B">2</text><text x="47" y="17" fill="#8A6D3B">3</text><text x="150" y="17" fill="#8A6D3B">…</text><text x="272" y="17" fill="#8A6D3B">16</text></g><text x="355" y="128" fill="#276b45">a sequence of 16 tiles</text><text x="355" y="145" fill="#9A938A">read like comic panels →</text></g></svg>
%%%

**What the comic-strip picture gets right:** separate panels, read in a fixed order, together tell one story — exactly how ordered tiles together make one image. **Where it breaks down:** a comic reader knows a panel on the second row is "below" the first, but our plain list has thrown that away — the model just sees tile 1, tile 2, tile 3… with no built-in sense of up or down. (Keep that in your pocket — it becomes an important puzzle later today.)

#### The finish line of the first move
So the full first move is: **image → grid of pixels → grid of tiles → one ordered sequence of tile-vectors.** That sequence is precisely the shape a Transformer expects for its input. You've just translated a picture into the Transformer's native language. Take a small victory lap — the hardest idea of the day is already behind you.

@@@ concept id=c5 tag="Count the tiles" title="How many tiles? One tiny formula" gotit="Got the count"
Quick, useful question: how many tiles do we get? There's a delightfully simple way to know. Think about tiling a **rectangular floor** with square tiles: you count how many fit across, count how many fit down, and multiply. Same tiles, same picture.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A rectangular floor tiled with square tiles. Four tiles fit across the top edge and four fit down the side. A label shows four across times four down equals sixteen tiles."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Count across, count down, multiply — like tiling a floor</text><g transform="translate(150,32)"><rect x="0" y="0" width="120" height="120" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><path d="M30 0 V120 M60 0 V120 M90 0 V120 M0 30 H120 M0 60 H120 M0 90 H120" stroke="#9A5A12" stroke-width="1"/></g><path d="M150 22 H270" stroke="#5E5191" stroke-width="1"/><text x="210" y="14" fill="#5E5191">4 across (W/P)</text><path d="M140 32 V152" stroke="#2D8B55" stroke-width="1"/><text x="90" y="96" fill="#276b45">4 down</text><text x="90" y="110" fill="#276b45">(H/P)</text><text x="360" y="88" fill="#C93B3B" font-size="13">4 × 4 = 16 tiles</text></g></svg>
%%%

**What the floor picture gets right:** the number of square tiles really is just "how many across" times "how many down." **Where it breaks down:** on a real floor you might trim edge tiles to fit, but here we choose sizes so the tiles divide the image evenly — no trimming.

#### The formula, in plain words then symbols
In plain words: **number of tiles = (how many fit down) × (how many fit across).** In symbols, with height `H`, width `W`, and patch size `P`:

%%% formula
expr: number of patches = (H / P) × (W / P)
note: H = image height in pixels, W = width, P = patch size. Each fraction is "how many P-sized tiles fit along that edge."
%%%

Worked example: a `224 × 224` image with `P = 16` gives `(224/16) × (224/16) = 14 × 14 = 196` tiles. So the model reads a sentence of **196** tile-words. Now for the fun part — **predict before you run:** if we shrink the tiles to `P = 8`, do we get *more* tiles or *fewer*? Then reveal:

%%% demo id=count label="predict, then reveal"
code: patches(H=224, W=224, P=16)   vs   patches(H=224, W=224, P=8)
out: P=16 → 14×14 = 196 tiles    |    P=8 → 28×28 = 784 tiles
take: <b>Smaller tiles → many MORE tiles.</b> Halving P from 16 to 8 does NOT double the count — it roughly QUADRUPLES it (196 → 784), because you get more tiles both across AND down. More tiles means finer detail, but a longer sequence for the model to chew through — so more compute.
%%%

Watch the count jump as the tiles get smaller — each step down in `P` stacks up a lot more tiles:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three bars showing how the tile count grows as patch size shrinks. Patch size 32 gives 49 tiles, a short bar. Patch size 16 gives 196 tiles, a taller bar. Patch size 8 gives 784 tiles, the tallest bar. Smaller tiles mean far more tiles and more compute."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Same 224×224 image — smaller P snaps into far more tiles</text><line x1="60" y1="165" x2="470" y2="165" stroke="#E5DFD6" stroke-width="1.5"/><rect x="90" y="150" width="70" height="15" fill="#2D8B55"/><text x="125" y="182" fill="#6B645E">P=32</text><text x="125" y="142" fill="#276b45">49</text><rect x="225" y="105" width="70" height="60" fill="#7C6DAA"/><text x="260" y="182" fill="#6B645E">P=16</text><text x="260" y="97" fill="#5E5191">196</text><rect x="360" y="45" width="70" height="120" fill="#C99A12"/><text x="395" y="182" fill="#6B645E">P=8</text><text x="395" y="37" fill="#8A6D3B">784</text><text x="480" y="120" fill="#C93B3B" font-size="10" text-anchor="end">more tiles</text><text x="480" y="134" fill="#C93B3B" font-size="10" text-anchor="end">= more compute</text></g></svg>
%%%

So `P` is a dial with a real trade-off: **small tiles catch fine detail but cost more compute; big tiles are cheap but blur detail.** That single knob is one of the first choices anyone tuning a Vision Transformer reaches for.

@@@ concept id=c6 tag="One shared stamp" title="Turn each tile into a tidy embedding — with one shared stamp" gotit="Got the embedding"
Our flattened tiles come in an awkward length, and a color tile is even longer than you'd guess. So the model does one more tidy-up step: it passes every tile through the **same** little converter to produce a clean, fixed-length vector. Think of a **rubber stamp**: you press the *same* stamp onto every page, and each page comes out marked in the identical way. The model presses one shared stamp onto every tile.

That stamp is a [[linear projection||Multiplying a vector by one shared weight matrix to turn it into a new fixed-length vector. Here it turns each flattened tile into its embedding.]] — in plain words, "multiply the tile's numbers by one shared table of weights." The tidy result is the tile's [[patch embedding||The fixed-length vector (length D) a patch becomes after the shared linear projection. This is what the Transformer actually reads.]], a vector of a chosen length `D`.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Several different tiles each pass through the same rubber stamp, labelled one shared weight matrix, and each comes out as a tidy fixed-length embedding bar of length D. The same stamp is used for every tile."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">One shared "stamp" presses every tile into a tidy vector</text><rect x="30" y="45" width="34" height="34" fill="#F6EFCF" stroke="#C99A12"/><rect x="30" y="95" width="34" height="34" fill="#EAF0FA" stroke="#5E5191"/><text x="47" y="150" fill="#6B645E">tiles</text><path d="M70 62 H110 M70 112 H110" stroke="#B8AEA2"/><rect x="120" y="55" width="90" height="64" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="165" y="82" fill="#C93B3B">shared</text><text x="165" y="98" fill="#C93B3B">stamp W</text><text x="165" y="138" fill="#9A938A" font-size="9">same for all tiles</text><path d="M215 70 H255 M215 105 H255" stroke="#B8AEA2"/><rect x="265" y="58" width="150" height="16" fill="#EAF5EE" stroke="#2D8B55"/><rect x="265" y="100" width="150" height="16" fill="#EAF5EE" stroke="#2D8B55"/><text x="340" y="140" fill="#276b45">tidy embeddings (length D)</text><text x="455" y="70" fill="#276b45" font-size="10">D long</text></g></svg>
%%%

**What the rubber-stamp picture gets right:** one identical stamp, pressed the same way onto every tile — that's exactly what "one shared weight matrix for all patches" means. **Where it breaks down:** a rubber stamp always prints the same mark, but this stamp *learns* — during training its weights adjust so the mark it prints is actually useful.

#### Why ONE shared stamp for every tile?
Here's a lovely design choice. The model could have used a different converter for each tile position — top-left tile gets its own, top-right its own, and so on. Instead it uses the **same** weights for every tile. This is called [[weight sharing||Using the same learned weights for every patch instead of a separate set per position. It treats all tiles uniformly and keeps the model small.]]. Why? Because a useful little feature — an edge, a splash of color — is worth spotting *wherever* it appears in the picture. One shared stamp treats every tile fairly and keeps the model small. (A separate stamp per position would balloon the model and force it to re-learn the same edge dozens of times.)

#### Color tiles are longer than you'd think
One surprise about the length. A **grayscale** tile stores just one brightness number per pixel. But a **color** tile stacks three numbers per pixel — one each for red, green, and blue. These three are the tile's [[channels||The separate layers of an image. A grayscale image has 1 channel; a color image has 3 (red, green, blue). A color pixel is 3 numbers.]]. So flattening a color tile means stacking all three layers. Watch a `16 × 16` color tile grow into its true length:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A build-up. A sixteen by sixteen tile has 256 pixels. For a grayscale tile that flattens to 256 numbers. For a color tile it has three stacked channels red green and blue, so 256 times 3 equals 768 numbers. Bars show 256 versus 768."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A 16×16 tile: grayscale = 256 numbers, color = 256 × 3 = 768</text><text x="115" y="40" fill="#6B645E">16 × 16 = 256 pixels</text><rect x="60" y="52" width="24" height="24" fill="#EFEAF0" stroke="#9A938A"/><text x="115" y="70" fill="#6B645E" font-size="10">1 channel (gray)</text><line x1="200" y1="64" x2="470" y2="64" stroke="#E5DFD6"/><rect x="200" y="56" width="90" height="16" fill="#9A938A"/><text x="245" y="90" fill="#6B645E">256 numbers</text><g transform="translate(60,100)"><rect x="8" y="8" width="24" height="24" fill="#FDECEC" stroke="#C93B3B"/><rect x="4" y="4" width="24" height="24" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="0" width="24" height="24" fill="#EAF0FA" stroke="#5E5191"/></g><text x="120" y="135" fill="#6B645E" font-size="10">3 channels (R,G,B)</text><line x1="200" y1="128" x2="470" y2="128" stroke="#E5DFD6"/><rect x="200" y="120" width="270" height="16" fill="#C99A12"/><text x="335" y="155" fill="#8A6D3B">768 numbers = 256 × 3</text></g></svg>
%%%

So a `16 × 16 × 3` color tile flattens to `256 × 3 = 768` numbers, and the shared stamp turns *that* into a tidy length-`D` embedding. **You now know the exact recipe** the model uses to turn any tile — gray or color — into the neat vector it reads: flatten, then press with one shared, learned stamp.

@@@ concept id=c7 tag="Tag where each tile sat" title="Puzzle: the tiles forgot where they were" gotit="Got positions"
Here's the puzzle we tucked away earlier. When we lined the tiles up into a plain list, we quietly threw something away: **where each tile used to sit** in the picture. The list just says "tile, tile, tile…" — nothing marks which one was the top-left sky and which was the bottom-right grass.

Imagine a **jigsaw puzzle** dumped out of the box into one long row on the table. Every piece is there, but the row itself doesn't tell you which piece belongs in the top-left corner. Shuffle the row and you can't tell — the *order alone* doesn't carry the map. That's exactly our problem: a bare list of tile-embeddings has no built-in sense of position.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="On the left, jigsaw pieces laid in a plain row with question marks, since the row does not say where each piece belongs. On the right, each piece now carries a small position tag like R1C1, so the map is recovered."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A row of tiles alone loses the map — a position tag restores it</text><text x="120" y="40" fill="#C93B3B">no position → where does each go?</text><g transform="translate(40,52)"><rect x="0" y="0" width="40" height="34" fill="#EFEAF0" stroke="#9A938A"/><rect x="50" y="0" width="40" height="34" fill="#EFEAF0" stroke="#9A938A"/><rect x="100" y="0" width="40" height="34" fill="#EFEAF0" stroke="#9A938A"/><rect x="150" y="0" width="40" height="34" fill="#EFEAF0" stroke="#9A938A"/><text x="20" y="22" fill="#C93B3B">?</text><text x="70" y="22" fill="#C93B3B">?</text><text x="120" y="22" fill="#C93B3B">?</text><text x="170" y="22" fill="#C93B3B">?</text></g><text x="255" y="72" fill="#9A5A12" font-size="14">→ add tags →</text><text x="400" y="40" fill="#276b45">each tile tagged with its spot</text><g transform="translate(320,52)"><rect x="0" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="50" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="100" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><rect x="150" y="0" width="40" height="34" fill="#EAF5EE" stroke="#2D8B55"/><text x="20" y="22" fill="#1a5c38" font-size="9">R1C1</text><text x="70" y="22" fill="#1a5c38" font-size="9">R1C2</text><text x="120" y="22" fill="#1a5c38" font-size="9">R2C1</text><text x="170" y="22" fill="#1a5c38" font-size="9">R2C2</text></g><text x="260" y="140" fill="#6B645E" font-size="10">tile embedding + position tag = "this tile, and it sat HERE"</text></g></svg>
%%%

**What the jigsaw picture gets right:** the pieces are all there, but a plain row doesn't say where each belongs — just like our tile list. **Where it breaks down:** with a real jigsaw *you* can eyeball the shapes and figure it out; the model can't peek — it only sees the numbers we hand it, so *we* must add the position back in.

#### The fix: give every tile a position tag
The cure is simple and elegant: to each tile's embedding we **add** a small extra vector that says "you sat *here*." That added tag is a [[positional embedding||A position-tagged vector added to each patch embedding so the model can recover where each tile sat (top/bottom, left/right). Here it is learned during training.]], and because the model *learns* these tags during training, we call it a *learned* positional embedding. Now "top-left tile" and "bottom-right tile" carry different tags, so the model can tell them apart. This is the direct remedy for the lost-position puzzle.

The main-flow rule is one gentle line: **for each tile, final input = patch embedding + its positional embedding.** (Both are length-`D` vectors, so they add cleanly, number by number.)

%%% demo id=posadd label="see the tag added"
code: tile_3_embedding + position_tag_for_slot_3
out: patch:[0.4, -0.2, 0.9, ...]  +  pos:[0.1, 0.0, -0.3, ...]  =  input:[0.5, -0.2, 0.6, ...]
take: <b>Same tile, now stamped with WHERE it sat.</b> The position tag is just another length-D vector added on top. Two identical-looking tiles in different spots now leave the model with different inputs — so it can finally tell top from bottom, left from right.
%%%

#### One extra guest: the [class] token
There's one more vector we slip into the sequence — but it's *not* a tile. We add a single extra learnable vector at the very front, called the [[class token||An extra learnable vector (also written [class] or CLS) added to the front of the patch sequence. Later it gathers information from all the tiles to summarize the whole image.]] (often written `[class]` or **CLS**). Think of it as an empty **name-tag** you pin to the front of the row: it starts blank, and its whole job — later, once the tiles start sharing information — is to collect a summary of the *entire* picture in one place. Today we just note that it's there, riding at the front of the sequence. **You've now solved the position puzzle and met the mystery guest** who will one day speak for the whole image.

@@@ concept id=c8 tag="What tiles can't do yet" title="Three things this first step still can't do" gotit="Got the limits"
Being honest about limits is what makes an engineer trustworthy — and each one here is a little puzzle with a satisfying "aha." Picture our tiles as **guests who just arrived at a party** but haven't started talking yet. Everyone's in the room; nobody's compared notes. That mental picture explains all three limits at once.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Three small panels of limits. One: two tiles side by side with no arrow between them, labelled no talking yet. Two: an object shape split awkwardly across a hard grid line into two tiles. Three: a small grid of position tags that does not stretch to cover a larger image grid."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Three honest limits of patchifying alone</text><g transform="translate(20,34)"><rect x="0" y="0" width="46" height="46" fill="#EAF5EE" stroke="#2D8B55"/><rect x="60" y="0" width="46" height="46" fill="#EAF5EE" stroke="#2D8B55"/><text x="23" y="28" fill="#1a5c38">🙂</text><text x="83" y="28" fill="#1a5c38">🙂</text><text x="53" y="28" fill="#C93B3B" font-size="14">✕</text><text x="53" y="70" fill="#C93B3B">no talking yet</text></g><g transform="translate(200,34)"><rect x="0" y="0" width="106" height="46" fill="#FCF3DC" stroke="#C99A12"/><line x1="53" y1="0" x2="53" y2="46" stroke="#C93B3B" stroke-width="2"/><circle cx="53" cy="23" r="20" fill="none" stroke="#5E5191" stroke-width="2.5"/><text x="53" y="70" fill="#C93B3B">object split by a hard edge</text></g><g transform="translate(390,34)"><rect x="0" y="0" width="40" height="40" fill="none" stroke="#5E5191" stroke-dasharray="3,2"/><path d="M13 0 V40 M26 0 V40 M0 13 H40 M0 26 H40" stroke="#5E5191" stroke-width="0.6"/><rect x="0" y="0" width="70" height="70" fill="none" stroke="#C93B3B" stroke-width="1.5"/><text x="35" y="86" fill="#C93B3B">tags don't stretch</text></g><text x="260" y="150" fill="#6B645E">guests are all in the room — but nobody has compared notes yet</text><text x="260" y="167" fill="#9A938A">talking = attention, which is tomorrow's job</text></g></svg>
%%%

**What the party picture gets right:** everyone has arrived, but nothing useful happens until they *talk* — patchifying seats the guests; it doesn't start the conversation. **Where it breaks down:** party guests can wander and chat freely, but our tiles can only "talk" through a specific mechanism (attention) that we haven't switched on yet.

#### Limit 1 · Tiles don't see each other (yet)
A lone tile-embedding knows only *its own* pixels. Nothing has mixed information *between* tiles, so no tile knows what the rest of the picture looks like. That mixing — letting tiles compare notes — is exactly the job of **attention**, and it's tomorrow's lesson (day 2). Today we've only *seated* the guests.

#### Limit 2 · A square grid can split an object
The grid always cuts on the same fixed lines, no matter what's in the picture. So an object that doesn't line up with the grid — a face, a wheel — can get **sliced across two tiles**, with half its shape in one tile and half in the next. A rigid square grid simply can't bend to fit whatever shape happens to be there.

#### Limit 3 · Position tags are tied to the training size
We *learned* one position tag per slot — but only as many slots as the training images had. Feed the model a **bigger** picture (more tiles, more slots) and there simply aren't enough learned tags to go around; a smaller picture leaves some unused. So the learned tags don't automatically stretch or shrink to a new image size.

!!! c-info 🧭
<b>Optional (skippable) — where each limit gets fixed.</b> None of these are dead ends; each has a known answer you'll meet soon. Tiles start "talking" via <b>self-attention</b> (day 2). The split-object and rigid-grid trade-off is softened by fancier patch schemes you'll hear about later. And the fixed-size tags can be gently stretched to a new resolution by <b>interpolating</b> the positional embeddings — a trick for fine-tuning on bigger images (a later ViT day). Today's job was just to get the picture *into* the model; making the tiles smart is what comes next.
!!!

Here's the same box as a map — each limit on the left, and the fix waiting for it on the right:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three rows mapping each limit to its later fix. Row one: tiles can't see each other, fixed by self-attention on day 2. Row two: a rigid grid can split an object, softened by fancier patch schemes later. Row three: position tags are tied to the training size, fixed by interpolating the tags for a new resolution later."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Every limit → its known fix (all coming up)</text><text x="120" y="36" text-anchor="middle" fill="#C93B3B">the limit today</text><text x="400" y="36" text-anchor="middle" fill="#276b45">the fix, later</text><g transform="translate(30,46)"><rect x="0" y="0" width="180" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="21" text-anchor="middle" fill="#8a3b3b">1 · tiles can't see each other</text></g><text x="222" y="67" fill="#9A5A12" font-size="13">→</text><g transform="translate(250,46)"><rect x="0" y="0" width="240" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="120" y="21" text-anchor="middle" fill="#1a5c38">self-attention (day 2)</text></g><g transform="translate(30,96)"><rect x="0" y="0" width="180" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="21" text-anchor="middle" fill="#8a3b3b">2 · grid splits an object</text></g><text x="222" y="117" fill="#9A5A12" font-size="13">→</text><g transform="translate(250,96)"><rect x="0" y="0" width="240" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="120" y="21" text-anchor="middle" fill="#1a5c38">fancier patch schemes (later)</text></g><g transform="translate(30,146)"><rect x="0" y="0" width="180" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="90" y="21" text-anchor="middle" fill="#8a3b3b">3 · tags tied to image size</text></g><text x="222" y="167" fill="#9A5A12" font-size="13">→</text><g transform="translate(250,146)"><rect x="0" y="0" width="240" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="120" y="21" text-anchor="middle" fill="#1a5c38">interpolate the tags (later)</text></g></g></svg>
%%%

These aren't flaws to fear — they're the map of what you'll learn next. **You can now say exactly what patchifying does and, just as importantly, what it doesn't** — that honesty is a real staff-level habit.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — a whole picture turned into something a Transformer can read. Here's a fresh way to hold the whole day: picture it as a short **assembly line**, where a raw photo rolls in one end and a tidy sequence of tagged **tile**s rolls out the other. Each station only makes sense because of the one before it — that's exactly how today stacked up. **Where the assembly-line picture breaks down:** a factory line runs once and ships the product, but these steps are a recipe you'll revisit and tune (the patch size `P`, the length `D`) for years. Read the stations, then keep the cheat-sheets as your map.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="An assembly line with five stations in order. Station one a photo. Station two a grid of tiles from patchify. Station three each tile flattened to a vector. Station four the tiles lined into a sequence. Station five each tile stamped into an embedding and tagged with its position, ready for the Transformer."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Today = a 5-station line: photo → readable sequence of tiles</text><line x1="20" y1="120" x2="500" y2="120" stroke="#E5DFD6" stroke-width="6"/><g transform="translate(30,60)"><rect x="0" y="0" width="60" height="48" fill="#FCF3DC" stroke="#C99A12"/><circle cx="30" cy="24" r="16" fill="#F6EFCF" stroke="#C99A12"/><text x="30" y="70" fill="#6B645E">1 · photo</text></g><text x="105" y="88" fill="#9A5A12">→</text><g transform="translate(118,60)"><rect x="0" y="0" width="60" height="48" fill="#FCF3DC" stroke="#C99A12"/><path d="M20 0 V48 M40 0 V48 M0 16 H60 M0 32 H60" stroke="#9A5A12" stroke-width="0.7"/><text x="30" y="70" fill="#6B645E">2 · patchify</text></g><text x="193" y="88" fill="#9A5A12">→</text><g transform="translate(206,76)"><rect x="0" y="0" width="60" height="16" fill="#EAF5EE" stroke="#2D8B55"/><path d="M15 0 V16 M30 0 V16 M45 0 V16" stroke="#2D8B55" stroke-width="0.7"/><text x="30" y="54" fill="#276b45">3 · flatten</text></g><text x="281" y="88" fill="#9A5A12">→</text><g transform="translate(294,76)"><rect x="0" y="0" width="60" height="16" fill="#FCF3DC" stroke="#C99A12"/><path d="M7.5 0 V16 M15 0 V16 M22.5 0 V16 M30 0 V16 M37.5 0 V16 M45 0 V16 M52.5 0 V16" stroke="#C99A12" stroke-width="0.6"/><text x="30" y="54" fill="#8A6D3B">4 · sequence</text></g><text x="369" y="88" fill="#9A5A12">→</text><g transform="translate(382,70)"><rect x="0" y="0" width="60" height="14" fill="#EAF5EE" stroke="#2D8B55"/><rect x="0" y="20" width="60" height="10" fill="#EFEAF7" stroke="#7C6DAA"/><text x="30" y="60" fill="#276b45">5 · embed</text><text x="30" y="72" fill="#5E5191">+ position</text></g><circle cx="480" cy="84" r="7" fill="#C93B3B"/><text x="480" y="72" fill="#6B645E">→ Transformer 🏁</text></g></svg>
%%%

The five stations, in order:
- **Station 1 · The picture.** An image is just a grid of pixels (height × width, and 3 channels if it's color).
- **Station 2 · Patchify.** Snap that grid into equal square **tile**s (patches) of size `P` — the star move.
- **Station 3 · Flatten.** Unroll each tile into one long vector so it acts like a word.
- **Station 4 · Sequence.** Lay the tile-vectors in reading order — the list a Transformer eats.
- **Station 5 · Embed + tag.** Press every tile with one shared learned stamp (the patch embedding), add a learned position tag so tiles remember where they sat, and pin a CLS name-tag at the front.
- **And the honest bit:** patchifying alone does no mixing between tiles, a rigid grid can split objects, and the position tags are tied to the training size — all fixed later.

#### Cheat-sheet · the pieces at a glance
%%% table
:: Piece :: What it is :: One-line reason
Patch (tile) :: a P×P square cut from the image :: turns a picture into word-sized chunks
Patch count :: `(H/P) × (W/P)` :: smaller P → many more tiles → more detail, more compute
Flatten :: unroll a tile into one vector :: lets a tile act like a word
Patch embedding :: tile × one shared weight matrix → length D :: same learned "stamp" for every tile (weight sharing)
Positional embedding :: a learned tag added to each tile :: restores where each tile sat
CLS token :: an extra learnable vector at the front :: will later summarize the whole image
%%%

#### Cheat-sheet · the words you met today
%%% jargon
pixel | one tiny colored dot; a picture is a grid of pixels
channels | the layers of an image — 1 for gray, 3 (R,G,B) for color
patch (tile) | a small square block of pixels cut from the image, e.g. 16×16
patchify | to cut an image into a grid of equal, non-overlapping square patches
flatten | to unroll a tile's grid of numbers into one long single-row vector
vector | a plain ordered list of numbers — the image world's "word"
sequence | an ordered list of items, the input a Transformer expects
patch size (P) | the side length of each tile; the dial for detail-vs-compute
linear projection | multiply by one shared weight matrix to make a fixed-length vector
patch embedding | the length-D vector a tile becomes after the shared stamp
weight sharing | using the same learned weights for every tile, not one per position
positional embedding | a learned tag added to each tile so it remembers where it sat
class token (CLS) | an extra learnable vector at the front that will later summarize the image
%%%

That's the whole day. Next you'll take these embedded, position-tagged tiles and let them finally *talk* — the **patch embeddings** and attention that turn a bag of tiles into an understood picture.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is the first thing a Vision Transformer does to an image? | a:2 | Feeds pixels one at a time | Blurs the whole image | Cuts it into a grid of equal square tiles (patches) | Turns it black and white | fb: The star move is "patchify": snap the picture into a grid of equal, non-overlapping square tiles, then read those instead of raw pixels.
q: A 224×224 image with patch size P=16 gives how many tiles, and what happens if you SHRINK P to 8? | a:2 | 16 tiles, and shrinking P gives fewer | 196 tiles, and shrinking P has no effect | 196 tiles, and shrinking P to 8 gives ~4× more tiles (784) — finer detail but more compute | 448 tiles, and shrinking P halves it | fb: (224/16)×(224/16)=14×14=196. Halving P to 8 gives 28×28=784 — roughly quadruple, because you get more tiles both across AND down. More detail, more compute.
q: After we line the tiles up into a plain list, what problem appears, and how is it fixed? | a:1 | The tiles lose their color, fixed by grayscale | The list has no built-in sense of WHERE each tile sat, fixed by ADDING a learned positional embedding to each tile | The tiles become too long, fixed by flattening | The model runs out of memory, fixed by attention | fb: A bare list is just "tile, tile, tile…" with no map. We add a learned position tag (positional embedding) to each tile's embedding so it remembers top/bottom, left/right.
q: Why does the model use ONE shared weight matrix (stamp) for every patch instead of a separate one per position? | a:1 | To make the model bigger | A useful feature (an edge, a color) is worth spotting anywhere in the image, and sharing keeps the model small — this is weight sharing | Because color tiles need it | So tiles can talk to each other | fb: Weight sharing: the same learned "stamp" is pressed onto every tile, so a good feature is detected wherever it appears, and the model stays small. Tile-to-tile mixing is attention's job, not the stamp's.
%%%

@@@ produce id=produce tag="Produce" title="Patchify a picture with your own hands" gotit="Done"
Time to see today's big idea with your own eyes. You'll take a small fake "image" (just a grid of numbers), snap it into tiles, flatten each tile, and count them with the formula — then watch the count jump when you shrink the patch size. **Predict first:** an `8 × 8` image cut with `P = 4` gives how many tiles? Now cut the SAME image with `P = 2` — more tiles or fewer, and by how much? Then run it and **watch** the numbers confirm it. Pick one path.

#### Option A · write it yourself
Create `sessions/m05b-vision-transformer/day-01-images-as-patches/experiment.py`. Make a fake image with `img = np.arange(8*8).reshape(8, 8)` (an `8 × 8` grid, so `H = W = 8`). Write a function that cuts it into `P × P` tiles and returns them, and **print the shape at every step** so you can see it happen. For `P = 4`: print the number of tiles from the formula `(H/P) × (W/P)`, print one tile as its `4 × 4` grid, then `flatten` that tile and print the resulting length (`16`). **Observe** that the tile count is `(8/4) × (8/4) = 2 × 2 = 4`. Now re-run with `P = 2` and **notice** the count jumps to `(8/2) × (8/2) = 4 × 4 = 16` — smaller tiles, far more of them. Finally, for a pretend color tile of shape `P × P × 3`, print that its flattened length is `P*P*3`. Run with `python3 sessions/m05b-vision-transformer/day-01-images-as-patches/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5b Day 1 artifact.

Create sessions/m05b-vision-transformer/day-01-images-as-patches/experiment.py that, with a comment on each step and printing shapes at every step:
1. Makes a fake image img = np.arange(8*8).reshape(8, 8) and prints its shape.
2. Defines patchify(img, P) that cuts the H×W image into non-overlapping P×P tiles (assume P divides H and W) and returns them as an array of shape (num_patches, P, P). Prints num_patches computed both from the formula (H//P)*(W//P) and from the returned array — they must match.
3. For P=4: prints num_patches (expect 4), prints one tile as its 4×4 grid, then flattens that tile with tile.reshape(-1) and prints its length (expect 16).
4. Re-runs patchify with P=2 and prints num_patches (expect 16) — show that halving P roughly quadruples the tile count (4 → 16), i.e. smaller tiles = far more tiles = more compute.
5. For a pretend color tile of shape (P, P, 3), prints that flattening gives length P*P*3 (e.g. 16*16*3 = 768 for P=16).
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (patchify, then the count jump)
- The formula and the actual returned tiles **agree** on the number of patches: `(H/P) × (W/P)`.
- With `P = 4` on the `8 × 8` image you get `4` tiles, and flattening one `4 × 4` tile gives a length-`16` vector — a tile becoming a word.
- The moment to **watch**: shrink `P` from `4` to `2` and the tile count jumps from `4` to `16` — roughly four times as many, not twice — because you gain tiles both across and down. That's the detail-vs-compute dial you met today.
- A pretend `16 × 16 × 3` color tile flattens to `768` numbers — the surprise length from stacking 3 color channels.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05b-vision-transformer/day-01-images-as-patches/log.md`: (1) in one sentence, how a Vision Transformer turns a picture into something it can read; (2) why smaller patches mean more compute; (3) one thing patchifying alone CANNOT do yet, and where it gets fixed.

@@@ fin
