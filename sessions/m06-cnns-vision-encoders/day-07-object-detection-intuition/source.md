---
quest_id: wf8-d07-detection
mode: concept
donor: v9-base.donor
page_title: "Module 6 · Day 7 — Object Detection Intuition"
module_label: "Module 06 · Represent · Day 7"
title: "Object Detection"
subtitle: "From 'What' to 'What and Where'"
brand_sub: "Foundations · M6 Day 7"
spine: "spotlight"
nav_prev_href: "../day-06-transfer-learning-embeddings/lesson.html"
nav_prev_label: "Transfer Learning & Embeddings"
nav_next_href: "../day-08-clip-contrastive/lesson.html"
nav_next_label: "CLIP: Contrastive Image-Text"
fin_title: "Module 6 · Day 7 complete! 🏆"
fin_body: "Nice — you've built the intuition for <b>Object Detection</b>: not just <i>what</i> is in a picture but <i>what and where</i>. You shine a <b>spotlight</b> over many spots, draw a box around each object with a confidence score, and clean up the duplicates so each thing gets one tidy box. This is the idea behind self-driving perception and photo tagging.<br>Next up: <b>CLIP: Contrastive Image-Text</b>."
notebook_yardstick: null
---

@@@ hero
@lede You know that game where someone hands you a busy picture and asks you to find every hidden animal — and for each one you draw a little **spotlight** circle around it and say what it is? "There's a cat, top-left. A dog, over here. Another cat, in the corner." You're not just naming *one* thing for the whole page — you're saying *what* each thing is **and** *where* it sits. That "what **and** where, for everything at once" is exactly what a computer does in **object detection**. Up to now your networks could look at a photo and say one word — "cat." Today they learn to point: to shine a spotlight on each object, draw a box around it, and label every one. This is the idea behind a self-driving car spotting pedestrians, your phone tagging faces in a photo, and a warehouse robot finding a box on a shelf. Today you learn to go from "what" to "what and where."
@goal Together we'll build the detection idea one piece at a time. You'll see why plain "name the picture" isn't enough, what a box really is (just four numbers), how a box carries a confidence score, the slow-but-honest first idea of sliding a spotlight everywhere, the two clever ways to speed that up (guess good spots, or use a grid of ready-made boxes), how we score a box as "right," the tidy-up step that kills duplicate boxes, and why detection must handle a surprise number of objects. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="What vs. where" title="Naming the picture isn't enough — we need what AND where" gotit="Got the task"
Let's start with the gap that detection is here to fill. Up to now, your network could look at a whole photo and shout one word: "cat!" That skill is called [[classification||Naming the single most likely thing in a whole image — one label for the entire picture, with no position.]] — it names the picture, but it never tells you *where* the cat is, and it can only name *one* thing.

Think of two ways a friend can answer "what's in this photo?" A friend who only does **classification** glances and says "a dog" — one word for the whole page. A friend who does **detection** points a finger and says "a dog, *right here*, and another dog *over there*, and a ball *down here*." The second friend gives you **what** *and* **where**, for *every* thing. That "point at each one and name it" is [[object detection||Finding every object in an image and, for each one, saying both what it is (a label) and where it is (a box around it).]] — and adding the "where" is called [[localization||Saying where an object is in the image — drawing the box that pins down its position — as opposed to just naming it.]].

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two friends answering what is in a photo. On the left, labelled classification, a friend gives one word cat for the whole picture with no position. On the right, labelled detection, a friend points at each object and gives a label plus a box for each: two cats and a dog, each in its own box. A caption says classification gives one what for the whole image, detection gives what and where for every object.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Classification = one word for the page · Detection = point at each thing</text>
<g transform="translate(28,34)"><rect x="0" y="0" width="215" height="150" fill="#EAF0FA" stroke="#5E5191"/><text x="107" y="18" fill="#5E5191">CLASSIFICATION (what only)</text><rect x="40" y="34" width="135" height="86" fill="#F0EBE3" stroke="#B8AEA2"/><text x="72" y="72" font-size="18">🐱</text><text x="120" y="94" font-size="18">🐕</text><text x="107" y="138" fill="#5E5191" font-size="10">answer: "cat" — one label, no place</text></g>
<g transform="translate(277,34)"><rect x="0" y="0" width="215" height="150" fill="#EAF5EE" stroke="#2D8B55"/><text x="107" y="18" fill="#276b45">DETECTION (what AND where)</text><rect x="40" y="34" width="135" height="86" fill="#F0EBE3" stroke="#B8AEA2"/><rect x="50" y="52" width="34" height="34" fill="none" stroke="#C93B3B" stroke-width="1.5"/><text x="67" y="76" font-size="14">🐱</text><rect x="112" y="70" width="34" height="34" fill="none" stroke="#C93B3B" stroke-width="1.5"/><text x="129" y="94" font-size="14">🐕</text><rect x="96" y="40" width="26" height="22" fill="none" stroke="#C93B3B" stroke-width="1.5"/><text x="109" y="56" font-size="9">🐱</text><text x="107" y="138" fill="#276b45" font-size="10">answer: cat + cat + dog, each in a box</text></g></g></svg>
%%%

**What the two-friends picture gets right:** one friend gives a single summary word, the other points at each object and names it — exactly the jump from classification to detection. **Where it breaks down:** a person points with a fuzzy finger, but a computer must give an *exact* box, described by numbers — you'll see those numbers in the very next concept.

#### Why "what and where, for many things" is the real task
Plain classification has two hard limits, and detection removes both:
- It gives **one** label for the whole image — but a real photo has *many* objects (three cats, a dog, a ball).
- It gives **no position** — it can't tell a self-driving car that the pedestrian is *dead ahead* versus *far to the side*, which is the whole point.

So detection = for **each** object, a **label** (what) **plus** a **box** (where). A [[spotlight||Our running image for detection: for each object you shine a small light on it, draw a box around what the light lands on, and say what it is.]] on each thing, one at a time, until every object is found and named.

!!! c-info 💡
<b>Where you've heard of this:</b> every time a self-driving car draws boxes around cars and people, your phone puts a square on each face before it focuses, or a store camera counts items on a shelf — that's object detection. It is one of the most-used vision skills in the real world.
!!!

**You now know the task:** what *and* where, for every object. Next, the smallest possible question — how do you write down "where" with just a handful of numbers?

@@@ concept id=c2 tag="The box" title="A box is just four numbers" gotit="Got the box"
"Where is it?" sounds hard — a shape can be anywhere, any size. Here's the friendly surprise: we don't trace the object's outline. We just wrap a plain rectangle around it, and a rectangle needs only **four numbers**. That rectangle has a name: a [[bounding box||The rectangle that says where an object is. It takes only four numbers — for example the left, top, width, and height — and it's the "where" part of a detection.]].

Think of **framing a photo on a wall.** You don't cut the frame to hug the person's curves — you pick a rectangle big enough to hold them and hang it. To describe that frame to someone on the phone, you only need to say four things: how far from the left, how far from the top, how wide, and how tall. Four numbers, and they know exactly where the frame goes. A bounding box is that frame around an object.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A picture-frame analogy for a bounding box. A cat sits inside an image. A rectangle is drawn around the cat. Arrows and labels mark four numbers: x, the distance from the left edge to the box's left side; y, the distance from the top edge to the box's top; w, the box width; and h, the box height. A caption says a bounding box is just four numbers: left, top, width, height.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A bounding box = a rectangle around the object = just 4 numbers</text>
<rect x="120" y="34" width="280" height="150" fill="#FCFBF8" stroke="#E4DED4"/>
<rect x="205" y="78" width="110" height="86" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="258" y="128" font-size="26">🐱</text>
<line x1="120" y1="60" x2="205" y2="60" stroke="#5E5191" stroke-dasharray="3,2"/><text x="162" y="54" fill="#5E5191" font-size="8">x = from left</text>
<line x1="150" y1="34" x2="150" y2="78" stroke="#2D8B55" stroke-dasharray="3,2"/><text x="150" y="30" fill="#276b45" font-size="8">y = from top</text>
<line x1="205" y1="172" x2="315" y2="172" stroke="#C99A12"/><text x="260" y="182" fill="#8A6D3B" font-size="8">w = width</text>
<line x1="325" y1="78" x2="325" y2="164" stroke="#C99A12"/><text x="352" y="124" fill="#8A6D3B" font-size="8">h = height</text></g></svg>
%%%

**What the picture-frame picture gets right:** you pin down a location with a plain rectangle and four simple measurements, not by tracing the shape — exactly how a bounding box works. **Where it breaks down:** a real frame is a physical object you hang once; a detector *predicts* these four numbers from the pixels, and it can be a little off, so the box is a *guess* at the true rectangle, not a perfect fit.

#### The four numbers, and a second way to write them
There are two common ways to write the same box — you'll see both, so meet them once:
- **left, top, width, height** — the corner and the size (the picture-frame way above).
- **two corners** — the (left, top) corner and the (right, bottom) corner. Same rectangle, written by its two opposite corners.

Both describe the *identical* rectangle; they're just two spellings. Let's read one box out loud, number by number:

%%% demo id=box label="read a bounding box's four numbers"
code: box = {"x": 205, "y": 78, "w": 110, "h": 86, "label": "cat"}
out: left edge  x = 205 px   # box starts 205 pixels from the left
     top edge   y = 78 px    # box starts 78 pixels from the top
     width      w = 110 px   # box is 110 pixels wide
     height     h = 86 px    # box is 86 pixels tall
     label = "cat"           # WHAT is in the box
     (same box as two corners: top-left (205, 78), bottom-right (315, 164))
take: <b>Four numbers place the rectangle; one label says what's inside.</b> That's a complete detection: a box (where) plus a class (what). The two-corner spelling is the same rectangle — just add width to x and height to y to get the far corner.
%%%

**You can now write "where" as four numbers.** But a detector doesn't just draw one box — it draws many, and it needs to tell you *how sure* it is about each. That "how sure" number is next.

@@@ concept id=c3 tag="How sure?" title="Every box carries a confidence score" gotit="Got confidence"
A detector doesn't know the truth — it *guesses* boxes from pixels. Some guesses are rock-solid ("that's clearly a cat"); others are shaky ("...maybe a cat? hard to tell in the shadow"). So along with each box, the model reports a number saying **how sure** it is. That number is the box's [[confidence score||A number from 0 to 1 attached to each predicted box, saying how sure the model is that a real object of that class is really there. Higher = more sure.]].

Think of a **weather forecast.** The forecaster doesn't just say "rain" — they say "**80%** chance of rain." That percentage lets *you* decide: 80% means grab an umbrella; 10% means don't bother. A confidence score does the same for a box: "I'm 92% sure there's a cat here" versus "I'm 30% sure." You then keep the confident boxes and throw away the shaky ones by setting a cut-off — a [[threshold||A cut-off number you choose. Keep every box whose confidence is above it (say 0.5) and drop the rest. Higher threshold = fewer but surer boxes.]].

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A weather-forecast analogy for confidence scores. An image has three predicted boxes with confidence scores: a cat box at 0.95, a dog box at 0.88, and a shaky box at 0.22. A dashed threshold line at 0.5 separates them. The cat and dog boxes, above the threshold, are kept; the 0.22 box, below the threshold, is dropped. A caption says each box carries a confidence, and a threshold keeps the sure ones and drops the shaky ones.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Each box says how sure it is · a threshold keeps the confident ones</text>
<rect x="24" y="34" width="200" height="150" fill="#FCFBF8" stroke="#E4DED4"/>
<rect x="48" y="60" width="60" height="50" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="78" y="94" font-size="16">🐱</text><text x="78" y="52" fill="#276b45" font-size="8">cat 0.95</text>
<rect x="130" y="96" width="66" height="56" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="163" y="132" font-size="16">🐕</text><text x="163" y="90" fill="#276b45" font-size="8">dog 0.88</text>
<rect x="70" y="128" width="40" height="30" fill="none" stroke="#C93B3B" stroke-width="1" stroke-dasharray="3,2"/><text x="90" y="172" fill="#8a3b3b" font-size="8">?? 0.22</text>
<g transform="translate(256,44)"><text x="120" y="0" fill="#2C2A28" font-size="9">confidence, high → low</text><rect x="16" y="12" width="200" height="18" fill="#EAF5EE" stroke="#2D8B55"/><text x="116" y="25" fill="#276b45" font-size="8">cat 0.95  → KEEP ✓</text><rect x="16" y="34" width="200" height="18" fill="#EAF5EE" stroke="#2D8B55"/><text x="116" y="47" fill="#276b45" font-size="8">dog 0.88  → KEEP ✓</text><line x1="16" y1="60" x2="216" y2="60" stroke="#5E5191" stroke-dasharray="4,3"/><text x="116" y="70" fill="#5E5191" font-size="8">threshold = 0.50</text><rect x="16" y="76" width="200" height="18" fill="#FDECEC" stroke="#C93B3B"/><text x="116" y="89" fill="#8a3b3b" font-size="8">?? 0.22  → DROP ✗</text><text x="116" y="112" fill="#6B645E" font-size="7.5">raise the threshold → fewer, surer boxes</text></g></g></svg>
%%%

**What the weather-forecast picture gets right:** a single number tells you how much to trust a claim, and you act on it with a cut-off — exactly how confidence and a threshold work. **Where it breaks down:** a weather percent is a careful long-run probability, but a detector's confidence is just the model's own hunch — a very sure-sounding 0.95 can still be wrong, so the number is a strong hint, not a promise.

#### Why the threshold is a knob you get to turn
The threshold is a dial *you* choose, and it trades two mistakes against each other:
- **High threshold** (say 0.9): you keep only the surest boxes. Fewer false alarms, but you might *miss* a real, half-hidden object.
- **Low threshold** (say 0.2): you keep almost everything. You catch more real objects, but you also keep junk boxes on empty background.

%%% demo id=threshold label="turn the threshold dial and watch boxes appear or vanish"
code: boxes = [("cat",0.95), ("dog",0.88), ("cat",0.41), ("blob",0.22)]  # keep if score > threshold
out: threshold = 0.90 → keep: [cat 0.95]                       # very picky: 1 box, misses the 0.88 dog!
     threshold = 0.50 → keep: [cat 0.95, dog 0.88]             # balanced: the two real objects
     threshold = 0.20 → keep: [cat 0.95, dog 0.88, cat 0.41, blob 0.22]  # greedy: also keeps a junk "blob"
take: <b>The threshold is a trade-off dial.</b> Turn it up and you get fewer, surer boxes (but risk missing things); turn it down and you catch more (but let junk through). There's no single "right" value — a self-driving car might keep it low to never miss a person; a photo tagger might keep it high to avoid silly labels.
%%%

**You now know a detection is a box, a label, and a confidence.** But *how* does the model find candidate boxes in the first place? The oldest, most honest idea is beautifully simple — and a little slow. Let's meet it.

@@@ concept id=c4 tag="Slide a spotlight" title="The first idea: slide a spotlight and classify every crop" gotit="Got the sliding idea"
Here's the most natural first idea anyone has — and it uses a skill you already own: classification. You already have a network that can look at a small picture and say "cat" or "not cat." So to find cats *anywhere*, just check *everywhere*: slide a little window across the image, and at each stop, cut out that patch and ask your classifier "is there a cat here?"

Think of sweeping a **flashlight across a dark room** looking for your keys. You don't magically know where the keys are, so you move the beam a little at a time — left to right, top to bottom — and at each spot you ask "keys here? no. here? no. here — yes!" Object detection's first idea is exactly this: shine a **spotlight** window over the image, step by step, and classify what each spot contains. This is called the [[sliding window||Moving a small box step by step across the whole image and classifying the patch inside it at every position — the simplest, oldest way to turn a classifier into a detector.]] approach, or "crop-and-classify."

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A flashlight-in-a-dark-room analogy for the sliding window. An image is covered by a grid of positions. A bright spotlight box sits at one position; faded boxes show the earlier positions the spotlight already visited across rows. At the spotlight's current spot, an arrow points to a small classifier that outputs cat question mark yes or no. A caption says slide the spotlight over every position and classify each crop.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Sliding window = sweep a spotlight over the image, classify each crop</text>
<rect x="24" y="32" width="270" height="160" fill="#0f0d0c" stroke="#3a352f"/>
<g stroke="#3a352f" fill="none"><line x1="24" y1="72" x2="294" y2="72"/><line x1="24" y1="112" x2="294" y2="112"/><line x1="24" y1="152" x2="294" y2="152"/><line x1="78" y1="32" x2="78" y2="192"/><line x1="132" y1="32" x2="132" y2="192"/><line x1="186" y1="32" x2="186" y2="192"/><line x1="240" y1="32" x2="240" y2="192"/></g>
<g fill="#2a2620"><rect x="24" y="32" width="54" height="40"/><rect x="78" y="32" width="54" height="40"/><rect x="132" y="32" width="54" height="40"/></g>
<rect x="186" y="32" width="54" height="40" fill="#FDF2E0" stroke="#C99A12" stroke-width="2"/><text x="213" y="57" font-size="16">🐱</text>
<text x="150" y="112" fill="#8a7f70" font-size="8">…visited spots…</text>
<g transform="translate(320,64)"><text x="90" y="-16" fill="#B8AEA2" font-size="14">→</text><rect x="20" y="0" width="140" height="30" fill="#EAF0FA" stroke="#5E5191"/><text x="90" y="19" fill="#5E5191" font-size="8">classifier: "cat here?"</text><rect x="20" y="40" width="66" height="24" fill="#EAF5EE" stroke="#2D8B55"/><text x="53" y="56" fill="#276b45" font-size="8">yes ✓</text><rect x="94" y="40" width="66" height="24" fill="#FDECEC" stroke="#C93B3B"/><text x="127" y="56" fill="#8a3b3b" font-size="8">no ✗</text><text x="90" y="86" fill="#6B645E" font-size="7.5">ask at every single position</text></g></g></svg>
%%%

**What the flashlight picture gets right:** you find something by systematically checking every spot with a small beam, not by knowing the answer in advance — exactly the sliding-window sweep. **Where it breaks down:** a flashlight is one size, but real objects come in many sizes, so a detector must slide windows of *many* sizes too — which, as you're about to see, blows up the amount of work.

#### The lovely part — and the painful part
This idea reuses the [[CNN||Convolutional Neural Network — the image-reading network from earlier in this module that turns a picture into feature maps describing edges, textures, and parts.]] classifier you already built, which is wonderful. But it hides a nasty cost. Count the work: you must classify a window at *every position*, and at *every size*, and *every shape* (tall, wide, square). That's a huge number of classifier runs for one image.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A before figure showing how the number of windows explodes. Three bars grow left to right: one window size gives thousands of positions, adding several sizes multiplies it, adding several shapes multiplies it again, ending at hundreds of thousands of classifier runs for a single image. A caption says trying every position, size, and shape is far too slow.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Why the naive sweep is too slow: windows explode</text>
<g><rect x="40" y="110" width="90" height="26" fill="#EAF5EE" stroke="#2D8B55"/><text x="85" y="127" fill="#276b45" font-size="8">1 size</text><text x="85" y="150" fill="#6B645E" font-size="7.5">~thousands</text></g>
<text x="150" y="126" fill="#B8AEA2">×</text>
<g><rect x="170" y="80" width="90" height="56" fill="#FDF2E0" stroke="#C99A12"/><text x="215" y="112" fill="#8A6D3B" font-size="8">× many sizes</text><text x="215" y="150" fill="#6B645E" font-size="7.5">~tens of thousands</text></g>
<text x="280" y="126" fill="#B8AEA2">×</text>
<g><rect x="300" y="40" width="180" height="96" fill="#FDECEC" stroke="#C93B3B"/><text x="390" y="90" fill="#8a3b3b" font-size="8">× many shapes</text><text x="390" y="106" fill="#8a3b3b" font-size="9">= HUNDREDS of</text><text x="390" y="120" fill="#8a3b3b" font-size="9">THOUSANDS of runs</text><text x="390" y="150" fill="#8a3b3b" font-size="7.5">for ONE image ✗ too slow</text></g></g></svg>
%%%

So the honest first idea *works* but is far too slow to be practical. Every smarter detector you'll hear about is really just an answer to one question: **how do we avoid checking every possible window?** That question drives the next two concepts.

@@@ concept id=c5 tag="Reuse the eye" title="Good news: the CNN's feature map already knows roughly where things are" gotit="Got feature reuse"
Before we speed things up, here's a gift that makes everything cheaper. You don't need a fresh classifier at every spot. The [[CNN||Convolutional Neural Network — the image-reading network from earlier in this module that turns a picture into a grid of features.]] you already have produces a [[feature map||A grid of numbers a CNN produces from an image. Each cell of the grid summarizes what the network "sees" in the matching patch of the original picture — and because it's a grid, it keeps rough position.]] — a grid of features — and, crucially, that grid keeps *rough position*: the top-left cell describes the top-left of the photo, the bottom-right cell the bottom-right. So the features that recognize *what* also quietly carry *where*.

Think of a **map of a shopping mall.** The map is smaller than the real mall, but it keeps the *layout*: the food court really is near the top, the cinema really is on the right. You can point to a spot on the small map and know where it is in the big mall. A CNN's feature map is a shrunk "map" of the photo: much smaller, but each spot on it still lines up with a region of the real image — so "there's fur in *this* cell" tells you fur is in *that part* of the picture.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A mall-map analogy for a CNN feature map keeping position. On the left, a large image of a cat and a dog. An arrow labelled CNN points to a smaller grid on the right, the feature map. The cell in the feature map that lines up with the cat's location lights up with a cat marker, and the cell over the dog lights up with a dog marker, showing the shrunk grid keeps rough position. A caption says the feature map is a shrunk map of the photo that still keeps where things are.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A feature map = a shrunk map of the photo that still keeps rough position</text>
<g transform="translate(24,36)"><rect x="0" y="0" width="180" height="140" fill="#FCFBF8" stroke="#E4DED4"/><text x="90" y="-6" fill="#6B645E" font-size="8">full image (big)</text><text x="46" y="56" font-size="26">🐱</text><text x="120" y="112" font-size="26">🐕</text></g>
<g transform="translate(214,90)"><text x="30" y="-6" fill="#276b45" font-size="8">CNN</text><text x="24" y="16" fill="#B8AEA2" font-size="16">→</text></g>
<g transform="translate(288,44)"><text x="104" y="-8" fill="#6B645E" font-size="8">feature map (small grid, keeps layout)</text><g stroke="#B8AEA2" fill="#FCFBF8"><rect x="0" y="0" width="52" height="40"/><rect x="52" y="0" width="52" height="40"/><rect x="104" y="0" width="52" height="40"/><rect x="156" y="0" width="52" height="40"/><rect x="0" y="40" width="52" height="40"/><rect x="52" y="40" width="52" height="40"/><rect x="104" y="40" width="52" height="40"/><rect x="156" y="40" width="52" height="40"/><rect x="0" y="80" width="52" height="40"/><rect x="52" y="80" width="52" height="40"/><rect x="104" y="80" width="52" height="40"/><rect x="156" y="80" width="52" height="40"/></g><rect x="0" y="0" width="52" height="40" fill="#FDF2E0" stroke="#C99A12" stroke-width="2"/><text x="26" y="26" font-size="14">🐱</text><rect x="104" y="80" width="52" height="40" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="130" y="106" font-size="14">🐕</text><text x="104" y="136" fill="#6B645E" font-size="7.5">cat-cell top-left · dog-cell bottom-mid</text></g></g></svg>
%%%

**What the mall-map picture gets right:** a smaller map keeps the layout, so a spot on the map tells you a place in the real world — exactly how a feature-map cell lines up with an image region. **Where it breaks down:** a mall map is drawn *by a person* to be readable, but a CNN's feature map is a pile of learned numbers with no pretty labels — the "where" is real but you read it out with math, not by looking.

#### Why this changes everything
Because the feature map already carries position, a detector can run the expensive CNN **once** for the whole image, then read *many* boxes straight off that single grid — instead of re-running a classifier on thousands of separate crops. This one idea — *compute features once, reuse them for every box* — is what makes modern detection fast enough to run on live video.

%%% demo id=featuremap label="one CNN pass gives a grid that keeps position"
code: fmap = cnn(image)      # run the eye ONCE for the whole photo
out: input image:  shape (3, 512, 512)   # 512x512 pixels, 3 colors
     feature map:  shape (256, 16, 16)    # a 16x16 grid, 256 features per cell
     cell (2, 3)  ~ covers image region around x≈64..96, y≈96..128
     → a strong "cat" feature in cell (2,3) means: cat near that region
take: <b>One CNN pass → a 16×16 grid that still knows layout.</b> Each cell maps back to a patch of the original image, so a feature that fires in a cell tells you an object is near *that* spot. Compute once, place many boxes — no thousands of separate crops.
%%%

**You now hold the key trick: features once, position kept.** So the smarter detectors don't check every window — they use this grid to only look at *promising* spots. There are two famous ways to do that, and they're next.

@@@ concept id=c6 tag="Guess good spots" title="Two-stage: first guess likely spots, then classify only those" gotit="Got region proposals"
The naive sweep failed because it checked *every* window. The first clever fix is obvious once you say it: don't check everywhere — first make a *short list* of spots that *look* like they might hold an object, and only classify those. That short list is a set of [[region proposals||A small set of candidate boxes — a few hundred or thousand — that a first step guesses are likely to contain some object, so the classifier only has to check those instead of every possible window.]].

Think of an **Easter egg hunt with a helpful older sibling.** Instead of you searching every square inch of the whole yard, your sibling walks ahead and says "check *near the bushes*, and *by the fence*, and *under the slide* — I have a feeling eggs are around there." You now only look carefully at a handful of good spots instead of the entire yard. Region proposals are that helpful sibling: a first step quickly points at "objecty-looking" regions, and only *those* get the careful classify-and-box step. Because it works in two steps — propose, then classify — this is called a [[two-stage detector||A detector that works in two steps: first propose a small set of likely-object regions, then classify each proposed region and refine its box. Slower but often very accurate.]].

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="An Easter-egg-hunt analogy for region proposals. Stage one on the left: from a full image, a propose step draws a handful of candidate boxes only around objecty-looking spots, ignoring empty background. An arrow labelled only these to stage two on the right, where each proposed box is classified and given a label plus confidence: cat 0.95, dog 0.90, and one proposal rejected as background. A caption says first guess a few good spots, then carefully classify only those.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Two-stage: ① propose a few good spots  →  ② classify only those</text>
<g transform="translate(24,36)"><rect x="0" y="0" width="210" height="150" fill="#FCFBF8" stroke="#E4DED4"/><text x="105" y="-6" fill="#8A6D3B" font-size="8">① PROPOSE likely spots</text><text x="50" y="60" font-size="20">🐱</text><text x="150" y="120" font-size="20">🐕</text><rect x="34" y="38" width="52" height="44" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="3,2"/><rect x="132" y="96" width="56" height="48" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="3,2"/><rect x="20" y="110" width="34" height="28" fill="none" stroke="#C99A12" stroke-width="1" stroke-dasharray="2,2"/><text x="105" y="176" fill="#8A6D3B" font-size="7.5">a few candidate boxes, not every window</text></g>
<g transform="translate(240,100)"><text x="20" y="-6" fill="#5E5191" font-size="7.5">only these</text><text x="18" y="14" fill="#B8AEA2" font-size="16">→</text></g>
<g transform="translate(296,36)"><rect x="0" y="0" width="200" height="150" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="-6" fill="#276b45" font-size="8">② CLASSIFY each proposal</text><rect x="16" y="20" width="168" height="28" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="38" fill="#276b45" font-size="8">proposal 1 → cat 0.95 ✓</text><rect x="16" y="56" width="168" height="28" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="74" fill="#276b45" font-size="8">proposal 2 → dog 0.90 ✓</text><rect x="16" y="92" width="168" height="28" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="110" fill="#8a3b3b" font-size="8">proposal 3 → background ✗</text><text x="100" y="140" fill="#276b45" font-size="7.5">careful work on a short list</text></g></g></svg>
%%%

**What the egg-hunt picture gets right:** a quick first pass narrows a huge search down to a few promising spots, so the careful work is tiny — exactly the two-stage propose-then-classify idea. **Where it breaks down:** the older sibling *knows* where eggs tend to be from memory, but the proposal step *learns* what "objecty" looks like from training data, and it can still miss an object it wasn't taught to expect.

#### Why two stages, and what it costs
Two-stage detectors trade a little speed for a lot of accuracy:
- **The win:** instead of hundreds of thousands of windows, you classify a few hundred proposals — far less work, and each gets careful attention, so accuracy is usually excellent.
- **The cost:** it's *two* passes (propose, then classify), so it's slower than doing everything in one look. For live video, that extra pass can be too slow — which motivates the one-look idea coming next.

!!! c-info 💡
<b>Where you've heard of this:</b> the famous <b>R-CNN</b> family (R-CNN, Fast R-CNN, Faster R-CNN) are all two-stage detectors. You don't need their details today — just the shape of the idea: <i>propose good spots, then classify them.</i> The details wait for a later detection-architectures lesson.
!!!

**You now have the two-stage idea: propose, then classify.** But what if you skip the "propose" step and instead scatter ready-made boxes everywhere in advance? That's the other big idea — anchors on a grid — and it's next.

@@@ concept id=c7 tag="Ready-made boxes" title="Anchors on a grid: scatter guess-boxes, then nudge them to fit" gotit="Got anchors & grid"
Here's the second clever idea, and it powers the fastest detectors. Instead of *proposing* spots one image at a time, you decide *ahead of time* on a fixed set of starter boxes — different sizes and shapes — placed at every cell of that feature-map grid. Then the model's only job is to *nudge* each starter box to hug the real object and say what's inside. Those pre-placed starter boxes are called [[anchor boxes||Pre-set candidate boxes of a few fixed sizes and shapes, placed at every grid cell before the model runs. The model nudges each anchor to fit a real object instead of inventing a box from nothing.]] (also called default boxes).

Think of laying a **sheet of pre-printed sticky-note outlines** over a photo. Before you even look at the picture, the sheet already has empty rectangle outlines everywhere — some tall, some wide, some square. Your job is just to grab the outline that's *closest* to each real object and slide/stretch it a little so it fits snugly, then write the object's name on it. You never draw a box from scratch — you *adjust* a ready-made one. Anchor boxes are those pre-printed outlines: the model picks the nearest anchor to each object and nudges it into place.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A pre-printed-sticky-note-outline analogy for anchor boxes. On the left, a grid cell shows several pre-set anchor outlines of different shapes stacked at one location: a wide one, a tall one, and a square one, all before seeing the object. On the right, the model has nudged the best-matching anchor to snugly fit a real cat and labelled it cat 0.95, while the unused anchors fade away. A caption says start from ready-made boxes of several shapes, then nudge the best one to fit.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Anchors = pre-printed box outlines you nudge to fit (never draw from scratch)</text>
<g transform="translate(24,36)"><rect x="0" y="0" width="210" height="150" fill="#FCFBF8" stroke="#E4DED4"/><text x="105" y="-6" fill="#8A6D3B" font-size="8">① pre-set anchors at a grid cell</text><g stroke="#B8AEA2" fill="none" stroke-dasharray="2,2"><line x1="70" y1="0" x2="70" y2="150"/><line x1="140" y1="0" x2="140" y2="150"/><line x1="0" y1="50" x2="210" y2="50"/><line x1="0" y1="100" x2="210" y2="100"/></g><circle cx="105" cy="75" r="3" fill="#2C2A28"/><rect x="70" y="60" width="70" height="30" fill="none" stroke="#C99A12" stroke-width="1.5"/><rect x="90" y="40" width="30" height="70" fill="none" stroke="#5E5191" stroke-width="1.5"/><rect x="80" y="50" width="50" height="50" fill="none" stroke="#2D8B55" stroke-width="1.5"/><text x="105" y="176" fill="#8A6D3B" font-size="7.5">wide · tall · square — all ready before looking</text></g>
<g transform="translate(240,100)"><text x="18" y="-4" fill="#5E5191" font-size="7.5">nudge best</text><text x="18" y="14" fill="#B8AEA2" font-size="16">→</text></g>
<g transform="translate(296,36)"><rect x="0" y="0" width="200" height="150" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="-6" fill="#276b45" font-size="8">② nudge the best anchor to fit</text><text x="88" y="96" font-size="30">🐱</text><rect x="60" y="52" width="80" height="66" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="100" y="46" fill="#276b45" font-size="8">cat 0.95</text><text x="100" y="140" fill="#276b45" font-size="7.5">picked the square anchor, stretched to fit</text></g></g></svg>
%%%

**What the sticky-note picture gets right:** starting from a ready-made outline and just adjusting it is far easier than drawing from nothing, and having several shapes ready means one usually fits — exactly how anchors work. **Where it breaks down:** you can *see* the outline that fits best, but the model has to *learn* which anchor to pick and how far to nudge it — and if no anchor shape is close to a weirdly-shaped object, even the best nudge fits poorly.

#### The one-look idea: a grid that predicts in a single pass
Now put anchors on that feature grid and let *every cell predict at once*, in a single pass through the network. Divide the image into a grid; each cell is responsible for objects near it and predicts a few boxes (from its anchors), each with a label and confidence — all in **one look**. This is the [[single-shot detector||A detector that predicts all boxes for the whole image in one pass through the network — no separate propose step. Divides the image into a grid and lets each cell predict boxes at once. Fast enough for live video.]] idea, famous under the name [[YOLO||"You Only Look Once" — a family of single-shot detectors that predict every box in one grid-based pass. Fast enough to run on live video.]] — "You Only Look Once."

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A grid-based single-shot detection figure. An image is divided into a grid of cells. Each cell is marked as predicting a few boxes at once. The cells overlapping the cat and the dog output boxes with labels cat 0.95 and dog 0.90 in a single pass, while empty cells predict nothing. A caption says divide into a grid and let every cell predict boxes in one look — the YOLO idea.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Single-shot (YOLO): divide into a grid, every cell predicts boxes in ONE pass</text>
<rect x="130" y="30" width="260" height="150" fill="#FCFBF8" stroke="#E4DED4"/>
<g stroke="#D8D0C4" fill="none"><line x1="195" y1="30" x2="195" y2="180"/><line x1="260" y1="30" x2="260" y2="180"/><line x1="325" y1="30" x2="325" y2="180"/><line x1="130" y1="80" x2="390" y2="80"/><line x1="130" y1="130" x2="390" y2="130"/></g>
<g font-size="6.5" fill="#C6BCAE"><text x="162" y="58">cell</text><text x="227" y="58">cell</text><text x="292" y="58">cell</text><text x="357" y="58">cell</text><text x="162" y="108">cell</text><text x="357" y="108">cell</text><text x="162" y="158">cell</text><text x="292" y="158">cell</text></g>
<text x="165" y="72" font-size="18">🐱</text><rect x="145" y="42" width="40" height="34" fill="none" stroke="#C93B3B" stroke-width="2"/><text x="165" y="40" fill="#276b45" font-size="7">cat .95</text>
<text x="290" y="122" font-size="18">🐕</text><rect x="268" y="92" width="46" height="38" fill="none" stroke="#5E5191" stroke-width="2"/><text x="291" y="90" fill="#5E5191" font-size="7">dog .90</text>
<text x="60" y="100" fill="#276b45" font-size="9">one pass →</text><text x="60" y="116" fill="#6B645E" font-size="7.5">all boxes at once</text></g></svg>
%%%

#### Two-stage vs. single-shot, side by side
%%% table
:: :: Two-stage (propose then classify) :: Single-shot / grid (one look)
How it works :: guess good spots, then classify each :: grid + anchors predict all boxes in one pass
Passes :: two :: one
Speed :: slower :: fast (can run on live video)
Accuracy :: usually higher :: very good, sometimes a touch lower
Famous name :: R-CNN family :: YOLO, SSD
%%%

**You now have both fast ideas: propose good spots, or scatter anchors on a grid.** But both spit out *many* boxes — often several piled on the same object. Two clean-up questions remain: how do we *score* a box as "right," and how do we *remove* the duplicates? Those two puzzles are next.

@@@ concept id=c8 tag="Right or wrong?" title="IoU: measuring how well a box overlaps the truth" gotit="Got IoU"
Here's a puzzle. The model predicts a box. The true box (the correct one, drawn by a human) is *almost* the same but shifted a few pixels. Is the prediction "right"? A box will *never* match the truth pixel-perfectly, so "exactly equal" is useless. We need a way to measure *how much two boxes overlap*, as a single fair number.

Think of two **sheets of paper laid on a table, overlapping.** How much do they cover the *same* spot? Look at two areas: the part where they **overlap** (both papers cover it), and the total area they cover **together** (either paper). If you divide "overlap" by "together," you get a fraction from 0 to 1: 0 means they don't touch at all, 1 means they sit *perfectly* on top of each other. That fraction is called [[Intersection over Union||A number from 0 to 1 measuring how much two boxes overlap: the area they share divided by the total area they cover together. 0 = no overlap, 1 = a perfect match. Short name: IoU.]] — **IoU** for short — "overlap area over combined area."

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A two-overlapping-papers analogy for IoU. Two rectangles overlap: a green true box and a red predicted box. The shaded lens where they overlap is labelled intersection, the shared area. The whole region covered by either box is labelled union, the combined area. A formula shows IoU equals intersection divided by union. Three small examples on the right show IoU near 0 for barely touching boxes, IoU around 0.5 for partly overlapping boxes, and IoU near 1 for nearly identical boxes. A caption says IoU is overlap divided by combined area.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">IoU = (area they SHARE) ÷ (area they cover TOGETHER)</text>
<g transform="translate(30,34)"><rect x="20" y="20" width="120" height="90" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="40" y="16" fill="#276b45" font-size="8">true box</text><rect x="70" y="50" width="120" height="90" fill="#FDECEC" fill-opacity="0.5" stroke="#C93B3B" stroke-width="2"/><text x="170" y="150" fill="#8a3b3b" font-size="8">predicted box</text><rect x="70" y="50" width="70" height="60" fill="#C99A12" fill-opacity="0.45" stroke="none"/><text x="105" y="84" fill="#7a5f10" font-size="7">overlap</text></g>
<g transform="translate(250,54)"><rect x="0" y="0" width="90" height="26" fill="#FDF9EF" stroke="#C99A12"/><text x="45" y="17" fill="#8A6D3B" font-size="8">intersection</text><text x="110" y="17" fill="#6B645E" font-size="10">÷</text><rect x="130" y="0" width="90" height="26" fill="#FDF9EF" stroke="#C99A12"/><text x="175" y="17" fill="#8A6D3B" font-size="8">union</text><text x="110" y="42" fill="#2C2A28" font-size="9">= IoU (0 → 1)</text></g>
<g transform="translate(250,108)" font-size="7.5"><rect x="0" y="0" width="30" height="24" fill="none" stroke="#2D8B55"/><rect x="22" y="14" width="30" height="24" fill="none" stroke="#C93B3B"/><text x="26" y="52" fill="#8a3b3b">IoU≈0.05 ✗ miss</text><rect x="90" y="0" width="30" height="24" fill="none" stroke="#2D8B55"/><rect x="104" y="8" width="30" height="24" fill="none" stroke="#C93B3B"/><text x="116" y="52" fill="#8A6D3B">IoU≈0.5 ok</text><rect x="180" y="0" width="30" height="24" fill="none" stroke="#2D8B55"/><rect x="182" y="2" width="30" height="24" fill="none" stroke="#C93B3B"/><text x="205" y="52" fill="#276b45">IoU≈0.95 ✓ match</text></g></g></svg>
%%%

**What the two-papers picture gets right:** you fairly measure "how much do these two agree" by comparing shared area to total area, giving one clean number — exactly IoU. **Where it breaks down:** papers are always the same rectangles you laid down, but here one box is the *human truth* and the other is the *model's guess*, so IoU is really scoring how close the guess came to the answer.

#### Why IoU is the referee
IoU is the referee that turns "sort of right" into a number you can act on. Pick a cut-off (often 0.5): if a predicted box overlaps a true box with IoU above it, count the prediction as a **hit**; below it, a **miss**. Let's compute one by hand so the fraction feels real:

%%% mathladder
title: IoU for one predicted box vs. the truth
words: divide the area the two boxes share by the total area they cover together
formula: IoU = (overlap area) ÷ (area of box A + area of box B − overlap area)
numbers: true box area = 100·90 = 9000 · predicted area = 120·90 = 10800 · overlap = 70·60 = 4200 → union = 9000 + 10800 − 4200 = 15600 → IoU = 4200 ÷ 15600 ≈ 0.27
sanity: 0.27 is below a 0.5 cut-off → this box is too loose; count it a MISS. Nudge it to overlap more and IoU climbs toward 1.
%%%

!!! c-info 💡
<b>Optional (skippable) — why subtract the overlap in "union":</b> if you just added both box areas, you'd count the shared middle <i>twice</i>. Subtracting one copy of the overlap fixes the double-count, so "union" is the true total area covered by <i>either</i> box. That's the only arithmetic subtlety — the idea stays "shared ÷ total."
!!!

**You can now score a box as hit or miss with IoU.** IoU also quietly solves our *other* problem: when several boxes pile onto one object, they *overlap each other* a lot — high IoU — which is exactly the clue we need to spot and remove duplicates. That clean-up step is next.

@@@ concept id=c9 tag="Kill duplicates" title="Non-Maximum Suppression: one tidy box per object" gotit="Got NMS"
Here's the mess we have to clean up. Because many windows, proposals, or grid-cells all fire on the *same* cat, the model hands you five slightly-different boxes all hugging one cat. You don't want five boxes on one cat — you want **one**. So you need a tidy-up step that keeps the best box for each object and deletes the rest.

Think of **a group photo where everyone shouts the same answer.** Five friends all yell "it's a cat!" — but you only need to write the answer down *once*. So you listen to the **loudest, most confident** voice, write that one down, and tell everyone who was saying the *same thing* to hush. Then you move to the next different answer. That's exactly the clean-up: keep the **most confident** box, then silence every other box that overlaps it a lot (same object), and repeat. This tidy-up is called [[Non-Maximum Suppression||The clean-up step that removes duplicate boxes: keep the highest-confidence box, delete every other box that overlaps it heavily (high IoU, so it's the same object), then repeat with the next box. Short name: NMS.]] — **NMS** — "suppress the non-maximum ones," meaning silence every box that isn't the top one for its object.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A group-photo-everyone-shouting analogy for Non-Maximum Suppression. On the left, labelled before NMS, one cat has five overlapping boxes piled on it with different confidence scores. In the middle, an arrow labelled keep the most confident, suppress overlapping duplicates. On the right, labelled after NMS, the cat has a single clean box, the highest-confidence 0.95 one, and the duplicates are gone. A caption says NMS keeps the best box per object and deletes overlapping copies.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">NMS = keep the loudest box, hush the overlapping duplicates</text>
<g transform="translate(24,34)"><rect x="0" y="0" width="200" height="140" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="16" fill="#8a3b3b" font-size="8">BEFORE: 5 boxes on 1 cat ✗</text><text x="88" y="98" font-size="30">🐱</text><rect x="56" y="46" width="82" height="66" fill="none" stroke="#C93B3B" stroke-width="1.5"/><rect x="52" y="50" width="86" height="66" fill="none" stroke="#C99A12" stroke-width="1"/><rect x="60" y="44" width="80" height="62" fill="none" stroke="#5E5191" stroke-width="1"/><rect x="50" y="52" width="90" height="70" fill="none" stroke="#B8AEA2" stroke-width="1"/><rect x="58" y="48" width="82" height="64" fill="none" stroke="#2D8B55" stroke-width="1"/><text x="100" y="132" fill="#8a3b3b" font-size="7.5">scores: .95 .88 .81 .77 .60</text></g>
<g transform="translate(232,90)"><text x="24" y="-8" fill="#5E5191" font-size="7">keep top,</text><text x="24" y="4" fill="#5E5191" font-size="7">hush rest</text><text x="22" y="24" fill="#B8AEA2" font-size="16">→</text></g>
<g transform="translate(296,34)"><rect x="0" y="0" width="200" height="140" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="16" fill="#276b45" font-size="8">AFTER: 1 clean box ✓</text><text x="88" y="98" font-size="30">🐱</text><rect x="56" y="46" width="82" height="66" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text x="97" y="42" fill="#276b45" font-size="8">cat 0.95</text><text x="100" y="132" fill="#276b45" font-size="7.5">kept the most confident, dropped the rest</text></g></g></svg>
%%%

**What the shouting-friends picture gets right:** you record the single loudest voice and hush everyone repeating it, so one answer survives per thing — exactly NMS keeping the top box and suppressing overlapping copies. **Where it breaks down:** friends who agree are saying the *same* words, but NMS decides "same object" by *box overlap* (high IoU), so if two *different* cats stand very close and their boxes overlap, NMS can wrongly hush a real one — a known trade-off you tune with the IoU cut-off.

#### The recipe, step by step — watch it run
NMS is a tiny loop, and IoU (last concept) is the tool it uses to decide "same object":
1. Take the box with the **highest confidence**. Keep it.
2. Delete every remaining box that **overlaps it a lot** (IoU above a cut-off) — those are duplicates of the same object.
3. Repeat with the highest-confidence box that's left, until none remain.

Predict how many boxes survive, then run it:

%%% demo id=nms label="run NMS on five boxes piled on one cat"
code: boxes = [(.95,"cat"), (.88,"cat"), (.81,"cat"), (.77,"cat"), (.60,"cat")]  # all high-IoU overlap
out: step 1: pick highest = 0.95 → KEEP it
     step 2: 0.88, 0.81, 0.77, 0.60 all overlap 0.95 heavily (IoU > 0.5) → SUPPRESS all four
     step 3: no boxes left → stop
     final: [cat 0.95]      # five became one
take: <b>Five overlapping boxes collapsed into one.</b> NMS kept the most confident box and used IoU to spot the duplicates and silence them. Every good detector runs NMS at the end so each object gets exactly one tidy box.
%%%

**You now have the full clean-up: IoU to measure overlap, NMS to keep one box per object.** One last idea ties the whole day together — the reason detection needs *all* this machinery that plain classification never did.

@@@ concept id=c10 tag="How many?" title="Detection must handle a surprise number of objects" gotit="Got variable count"
Step back and notice something strange about detection that classification never faced. When you classify, the answer is always **one** label — the shape of the answer is fixed. But detection? One photo might have **zero** objects (an empty road), the next might have **one** (a single cat), the next **forty** (a crowd). The model can't know the count in advance — it must handle **any** number.

Think of **setting a dinner table without knowing how many guests are coming.** If you always laid *exactly four* plates, you'd be wrong the moment three people show up, or seven. A good host stays *flexible*: lay a plate for each guest as they arrive, however many that turns out to be. Detection is that flexible host: it outputs a box for each object it finds — zero, one, or many — never a fixed-size answer. This is why detection needs boxes-plus-confidence-plus-NMS instead of one tidy label: it must produce a [[variable number of outputs||An answer whose size isn't fixed ahead of time. Classification always outputs one label; detection outputs one box per object, and the number of objects changes from image to image — zero, one, or many.]].

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A dinner-table analogy for a variable number of objects. Three images sit in a row, each producing a different number of boxes. The first, an empty road, produces zero boxes. The second, one cat, produces one box. The third, a crowd, produces many boxes. Below, classification is shown always producing exactly one label, while detection produces zero, one, or many boxes. A caption says detection must handle any number of objects, unlike classification's single fixed answer.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Detection outputs a surprise number of boxes: 0, 1, or many</text>
<g transform="translate(24,30)"><rect x="0" y="0" width="150" height="80" fill="#FCFBF8" stroke="#E4DED4"/><text x="75" y="44" font-size="14">🛣️ (empty)</text><text x="75" y="100" fill="#8a3b3b" font-size="8">→ 0 boxes</text></g>
<g transform="translate(186,30)"><rect x="0" y="0" width="150" height="80" fill="#FCFBF8" stroke="#E4DED4"/><rect x="52" y="26" width="46" height="38" fill="none" stroke="#C93B3B" stroke-width="1.5"/><text x="75" y="54" font-size="16">🐱</text><text x="75" y="100" fill="#8A6D3B" font-size="8">→ 1 box</text></g>
<g transform="translate(348,30)"><rect x="0" y="0" width="150" height="80" fill="#FCFBF8" stroke="#E4DED4"/><g stroke="#C93B3B" fill="none" stroke-width="1"><rect x="12" y="18" width="26" height="24"/><rect x="46" y="30" width="26" height="24"/><rect x="80" y="20" width="26" height="24"/><rect x="112" y="34" width="26" height="24"/><rect x="30" y="48" width="26" height="24"/><rect x="86" y="50" width="26" height="22"/></g><text x="66" y="46" font-size="12">👥</text><text x="75" y="100" fill="#276b45" font-size="8">→ many boxes</text></g>
<line x1="24" y1="128" x2="496" y2="128" stroke="#E4DED4"/>
<text x="150" y="150" fill="#5E5191" font-size="8">classification: ALWAYS 1 label (fixed)</text><text x="380" y="150" fill="#276b45" font-size="8">detection: 0 / 1 / many boxes (flexible)</text></g></svg>
%%%

**What the dinner-table picture gets right:** you can't lay a fixed number of plates when the guest count is unknown — you adapt to however many arrive, exactly like detection adapting to however many objects appear. **Where it breaks down:** a host *counts guests* as they come, but a detector doesn't count first — it proposes lots of boxes, scores them, thresholds, and NMS-cleans, and *whatever survives* is the count; the number falls out of the pipeline rather than being decided up front.

#### Why this is the deep reason detection is harder
This one fact explains the whole day's machinery:
- **Classification** returns a fixed answer (one label), so it's simple.
- **Detection** must return a *set* of any size, so it needs candidate boxes (windows / proposals / anchors), a confidence per box, a threshold to drop the junk, and NMS to remove duplicates. All of it exists to turn "many raw guesses" into "exactly the objects that are really there — however many that is."

**You now understand why detection needs everything you learned today.** Let's gather the whole day onto one page.

@@@ concept id=c11 tag="Recap" title="Today in one page" gotit="Got the recap"
You built the whole detection idea today. Before we file it away, let's lay every piece out on one page.

Think of finishing a **treasure hunt** and pulling out the **map** you filled in along the way. You didn't discover anything new by looking at the map — you *gathered* every clue you already found into one glance, in order, so you can see the whole journey. This recap is that map for object detection: every idea you met, side by side, in the order you met them.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A treasure-map analogy for the recap. A single map holds a path with numbered stops: what and where, the box, confidence, slide a spotlight, reuse the eye, guess spots or anchors, IoU, NMS, and how many. Each stop is a small labelled marker along a dotted trail. A caption says one map gathers the whole day, each marker a hook back to a concept.">
<g font-family="monospace" font-size="8" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A recap is a treasure map: the whole day's trail in one glance</text>
<rect x="20" y="28" width="480" height="128" fill="#FDF9EF" stroke="#C99A12" stroke-width="1.5" rx="4"/>
<path d="M50 120 C120 60, 180 140, 250 90 S380 40, 470 100" fill="none" stroke="#C6BCAE" stroke-dasharray="4,4"/>
<g><circle cx="50" cy="120" r="12" fill="#EAF0FA" stroke="#5E5191"/><text x="50" y="123" fill="#5E5191">what+where</text></g>
<g><circle cx="118" cy="78" r="12" fill="#EAF5EE" stroke="#2D8B55"/><text x="118" y="81" fill="#276b45">box</text></g>
<g><circle cx="180" cy="128" r="12" fill="#FDF2E0" stroke="#C99A12"/><text x="180" y="131" fill="#8A6D3B">confidence</text></g>
<g><circle cx="250" cy="90" r="12" fill="#2a2620" stroke="#C99A12"/><text x="250" y="93" fill="#E7C972">spotlight</text></g>
<g><circle cx="310" cy="62" r="12" fill="#EAF5EE" stroke="#2D8B55"/><text x="310" y="65" fill="#276b45">reuse eye</text></g>
<g><circle cx="360" cy="112" r="12" fill="#FDF2E0" stroke="#C99A12"/><text x="360" y="115" fill="#8A6D3B">anchors</text></g>
<g><circle cx="410" cy="66" r="12" fill="#EAF0FA" stroke="#5E5191"/><text x="410" y="69" fill="#5E5191">IoU</text></g>
<g><circle cx="452" cy="108" r="12" fill="#FDECEC" stroke="#C93B3B"/><text x="452" y="111" fill="#8a3b3b">NMS</text></g>
<text x="260" y="150" fill="#6B645E" font-size="8.5">…and finally: handle 0 / 1 / many objects — each marker a hook back to a concept</text></g></svg>
%%%

**What the treasure-map picture gets right:** nothing new is created — you gather the clues you already found into one ordered trail. **Where it breaks down:** map markers just sit there, but these ideas *connect* into a working pipeline — a box carries a confidence, a spotlight or anchor proposes it, IoU scores it, NMS cleans it — so this "map" is really a recipe.

Here's the whole detection pipeline as one connected picture:

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A one-page detection pipeline. An image enters a CNN which makes a feature map that keeps position. From there, candidate boxes come from either proposals or grid anchors. Each candidate gets a label and a confidence score. A threshold drops low-confidence boxes, then NMS removes overlapping duplicates using IoU, leaving a variable number of final boxes: what and where for each object. A caption reads image to CNN to candidates to scores to threshold to NMS to final boxes.">
<g font-family="monospace" font-size="7.5" text-anchor="middle"><text x="260" y="14" fill="#2C2A28" font-size="12">The pipeline: image → CNN → candidates → score → clean up → boxes</text>
<g transform="translate(10,54)"><rect x="0" y="0" width="52" height="44" fill="#FDF9EF" stroke="#C99A12"/><text x="26" y="26" fill="#8A6D3B">image</text></g>
<text x="66" y="78" fill="#B8AEA2">→</text>
<g transform="translate(78,50)"><rect x="0" y="0" width="86" height="52" fill="#EAF5EE" stroke="#2D8B55"/><text x="43" y="22" fill="#276b45">CNN → feature</text><text x="43" y="34" fill="#276b45">map (keeps where)</text></g>
<text x="168" y="78" fill="#B8AEA2">→</text>
<g transform="translate(180,50)"><rect x="0" y="0" width="90" height="52" fill="#FDF2E0" stroke="#C99A12"/><text x="45" y="22" fill="#8A6D3B">candidates</text><text x="45" y="34" fill="#8A6D3B">(proposals/anchors)</text></g>
<text x="274" y="78" fill="#B8AEA2">→</text>
<g transform="translate(286,50)"><rect x="0" y="0" width="80" height="52" fill="#EAF0FA" stroke="#5E5191"/><text x="40" y="22" fill="#5E5191">label +</text><text x="40" y="34" fill="#5E5191">confidence</text></g>
<text x="370" y="78" fill="#B8AEA2">→</text>
<g transform="translate(382,50)"><rect x="0" y="0" width="60" height="52" fill="#FDECEC" stroke="#C93B3B"/><text x="30" y="20" fill="#8a3b3b">threshold</text><text x="30" y="34" fill="#8a3b3b">+ NMS</text><text x="30" y="46" fill="#8a3b3b">(uses IoU)</text></g>
<text x="446" y="78" fill="#B8AEA2">→</text>
<g transform="translate(456,50)"><rect x="0" y="0" width="54" height="52" fill="#EAF5EE" stroke="#2D8B55"/><text x="27" y="22" fill="#276b45">final</text><text x="27" y="34" fill="#276b45">boxes</text><text x="27" y="46" fill="#276b45">0/1/many</text></g>
<text x="260" y="140" fill="#6B645E" font-size="8">each surviving box = one object: WHAT (label) + WHERE (box)</text></g></svg>
%%%

The day in a few beats:
- **What vs. where.** Classification names one thing for the whole image; **detection** gives *what* **and** *where* for *every* object.
- **The box.** "Where" is a **bounding box** — just four numbers (left, top, width, height, or two corners).
- **Confidence.** Each box carries a **confidence score** (0–1); a **threshold** you pick keeps the sure boxes and drops the shaky ones.
- **Slide a spotlight.** The first idea: slide a window everywhere and classify each crop — honest but far too slow.
- **Reuse the eye.** A CNN's **feature map** is computed once and keeps rough position, so you place many boxes from one pass.
- **Guess spots or anchors.** **Two-stage** (region proposals: propose, then classify) or **single-shot** (grid + **anchor boxes**, all in one look, YOLO-style).
- **IoU.** **Intersection over Union** = overlap ÷ combined area; scores a box as hit or miss.
- **NMS.** **Non-Maximum Suppression** keeps the most confident box per object and deletes overlapping duplicates (using IoU).
- **How many.** Detection outputs a **variable number** of boxes — 0, 1, or many — which is why it needs all this machinery.

#### Cheat-sheet · the detection pipeline at a glance
%%% table
:: Step :: What it does :: Key word
Find candidates :: propose or scatter boxes to check :: proposals / anchors / grid
Score each box :: what it is + how sure :: label + confidence
Drop the junk :: keep boxes above a cut-off :: threshold
Measure overlap :: how well two boxes agree :: IoU
Remove duplicates :: one clean box per object :: NMS
%%%

#### Cheat-sheet · the words you met today
%%% jargon
object detection | finding every object and giving each a label (what) plus a box (where)
classification vs localization | classification names one thing; localization adds the box saying where
bounding box | the rectangle locating an object — four numbers (left, top, width, height)
confidence score | how sure the model is a real object is in a box (0 to 1)
threshold | the cut-off you pick to keep sure boxes and drop shaky ones
sliding window | the slow first idea: slide a box everywhere and classify each crop
feature map | the CNN's grid of features that keeps rough position — reused for every box
region proposals | a short list of likely-object spots (two-stage: propose, then classify)
anchor boxes | pre-set candidate boxes on a grid that the model nudges to fit objects
single-shot / YOLO | predict all boxes in one grid-based pass
IoU | Intersection over Union: overlap ÷ combined area (0 to 1)
NMS | Non-Maximum Suppression: keep the top box per object, delete overlapping duplicates
variable number of objects | detection outputs 0, 1, or many boxes — the count isn't fixed
%%%

That's the whole day. You can now explain the jump from "what" to "what and where," what a box and its confidence are, the slide-a-spotlight first idea and its two speed-ups (proposals and anchors), how IoU scores a box, how NMS keeps one box per object, and why detection must handle any number of objects. Next you'll learn a very different way to connect images and words — **day-08, CLIP: Contrastive Image-Text**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is the difference between classification and object detection? | a:2 | detection is just a faster version of classification | classification uses color and detection uses shape | classification gives one label for the whole image; detection gives a label AND a box (what and where) for every object | detection only works on one object at a time | fb: Classification names one thing for the whole picture. Detection adds "where" — a box around each object — and finds many objects at once. That's the "what AND where" jump.
q: A detector predicts a box that is close to the true box but shifted a bit. How do we score whether it's "right"? | a:1 | check if the two boxes are exactly equal, pixel for pixel | compute IoU = overlap area ÷ combined area, and count it correct if IoU is above a cut-off like 0.5 | count the pixels inside the predicted box | compare the two confidence scores | fb: Boxes never match perfectly, so "exactly equal" is useless. IoU (Intersection over Union) measures overlap as shared-area ÷ combined-area, from 0 to 1; above a cut-off (often 0.5) counts as a hit.
q: Your detector outputs five overlapping boxes on the SAME cat. What step fixes this, and how? | a:2 | raise the confidence threshold until only one box is left anywhere | run the CNN again on a smaller image | Non-Maximum Suppression (NMS): keep the highest-confidence box, then delete every other box that overlaps it heavily (high IoU), and repeat | lower the learning rate | fb: NMS is the clean-up step. It keeps the most confident box, uses IoU to spot the overlapping duplicates of the same object, and suppresses them — leaving one tidy box per object.
q: Why does object detection need candidate boxes, confidence scores, and NMS, when plain classification needs none of that? | a:1 | detection must output a VARIABLE number of results — zero, one, or many objects — so it proposes many boxes, scores them, thresholds, and cleans duplicates, while classification always returns one fixed label | detection images are always bigger | classification is an older technique that was never improved | detection cannot use a CNN | fb: Classification returns one fixed label. Detection must return a set of any size (0/1/many objects), so it needs candidates (proposals/anchors), a confidence per box, a threshold, and NMS to turn many raw guesses into exactly the objects that are really there.
%%%

@@@ produce id=produce tag="Produce" title="Compute IoU and run NMS with your own hands" gotit="Done"
Time to run the detection clean-up yourself and *watch* it work. You'll compute **IoU** between two boxes by hand-code, then run **Non-Maximum Suppression** on a little pile of overlapping boxes and see five collapse into one. **Predict first:** if a true box and a predicted box overlap in most of their area, will their IoU be closer to 0 or to 1? And if you have four boxes all hugging the same cat plus one box on a separate dog, how many boxes should survive NMS? Then run it and **observe** that heavy overlap gives high IoU, that NMS keeps the most-confident box per object and deletes the overlapping duplicates, and that the far-apart dog box survives on its own. Pick one path.

#### Option A · write it yourself
Create `sessions/m06-cnns-vision-encoders/day-07-object-detection-intuition/experiment.py` and **print values at every step**. (1) **A box as four numbers:** represent boxes as `[x1, y1, x2, y2]` (two corners) — write `cat_true = [50,50,150,140]` and `cat_pred = [60,55,158,150]` and a far-away `dog = [300,200,380,300]`; **print** them. (2) **IoU function:** write `iou(a,b)` that computes the intersection rectangle, its area, each box's area, the union `areaA + areaB - inter`, and returns `inter/union`; **print** `iou(cat_true, cat_pred)` (should be high, near 0.7–0.8) and `iou(cat_true, dog)` (should be `0.0`), and **notice** overlap → high IoU, no overlap → 0. (3) **A pile of boxes with scores:** make `boxes = [([55,52,152,142],0.95), ([58,50,150,140],0.88), ([52,55,156,145],0.81), ([60,48,148,138],0.77), ([300,200,380,300],0.90)]` — four cats + one dog. (4) **NMS:** sort by score (highest first); repeatedly take the top box, keep it, and drop every remaining box whose `iou` with it is above `0.5`; **print** the surviving boxes and **observe** the four overlapping cat-boxes collapse to *one* while the dog box survives — final count = 2. (5) **Threshold demo:** filter `boxes` by `score > 0.85` before NMS and **notice** which boxes disappear. Run with `python3 sessions/m06-cnns-vision-encoders/day-07-object-detection-intuition/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 6 Day 7 artifact.

Create sessions/m06-cnns-vision-encoders/day-07-object-detection-intuition/experiment.py that demonstrates IoU and Non-Maximum Suppression (NMS) for object detection using plain Python/NumPy (no real model needed), with a comment on each step and printing values at every step:
1. Represent boxes as [x1, y1, x2, y2] (top-left and bottom-right corners). Define cat_true=[50,50,150,140], cat_pred=[60,55,158,150], dog=[300,200,380,300]. Print all three.
2. Write iou(a,b): compute the overlap rectangle (max of the left/top corners, min of the right/bottom corners), clamp negative width/height to 0, get intersection area, each box area, union = areaA + areaB - inter, return inter/union. Print iou(cat_true,cat_pred) (should be high, ~0.7-0.8) and iou(cat_true,dog) (should be 0.0). Explain: heavy overlap = high IoU, no overlap = 0.
3. Define boxes = [([55,52,152,142],0.95),([58,50,150,140],0.88),([52,55,156,145],0.81),([60,48,148,138],0.77),([300,200,380,300],0.90)] — four overlapping cat boxes plus one separate dog box.
4. NMS: sort boxes by score descending; while any remain, pop the highest-score box, keep it, and remove every remaining box whose iou with the kept box > 0.5. Print the kept boxes and assert exactly 2 survive (one cat box + the dog box). Explain NMS keeps the most confident box per object and suppresses overlapping duplicates using IoU.
5. Threshold demo: before NMS, filter boxes to those with score > 0.85 and print which remain, showing the threshold is a separate dial from NMS.
6. Print a one-line takeaway: "detection outputs many boxes with confidence scores; IoU measures how well two boxes overlap; NMS keeps one tidy box per object; the number of final boxes is variable (0, 1, or many)."
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (predict, then confirm)
- `iou(cat_true, cat_pred)` is high (the boxes overlap in most of their area), while `iou(cat_true, dog)` is exactly `0.0` (they don't touch) — heavy overlap means high IoU.
- NMS turns the four overlapping cat boxes into **one** (the highest-confidence 0.95 box) and keeps the far-away dog box — final count is **2**, one clean box per object.
- Raising the threshold before NMS makes low-confidence boxes disappear — proof the threshold is a separate dial you can turn independently of NMS.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m06-cnns-vision-encoders/day-07-object-detection-intuition/log.md`: (1) the difference between classification and detection (what vs. what-and-where); (2) what IoU measures and how NMS uses it to remove duplicate boxes; (3) why detection must handle a variable number of objects while classification returns one fixed label.
!!!

@@@ fin

