---
quest_id: wf8-d08-clip
mode: concept
donor: v9-base.donor
page_title: "Module 6 · Day 8 — CLIP & Contrastive Learning"
module_label: "Module 06 · Represent · Day 8"
title: "CLIP and Contrastive Learning"
subtitle: "Teaching Images and Words to Share One Space"
brand_sub: "Foundations · M6 Day 8"
spine: "matchmaker"
nav_prev_href: "../day-07-object-detection-intuition/lesson.html"
nav_prev_label: "Object Detection (Intuition)"
nav_next_href: "../review.html"
nav_next_label: "Review Gate"
fin_title: "Module 6 · Day 8 complete! 🏆"
fin_body: "You did it — you've finished <b>CLIP & Contrastive Learning</b> and all of Module 6! CLIP is a <b>matchmaker</b>: it trains an image encoder and a text encoder together so a photo and its caption land at the same spot in one shared space — pulling true pairs together and pushing mismatches apart. That single idea powers zero-shot classification and the text-to-image models you've heard of. You've now gone from a single sliding stencil all the way to image-and-text understanding.<br>Next up: the <b>Module 6 Review Gate</b>."
notebook_yardstick: null
---

@@@ hero
@lede Picture a school dance where two crowds stand on opposite walls — one crowd of **photos**, one crowd of **words** — and a friendly **matchmaker** walks the floor pairing each photo with the caption that describes it: this dog photo with "a dog," this beach photo with "a beach at sunset." That job — deciding which picture goes with which sentence — is exactly what today's model learns to do. It's called **CLIP**, and it's the trick that lets you type "a photo of a corgi in sunglasses" and have a computer *find* it, or *make* it. Every image generator you've heard of leans on this idea: teach pictures and words to live in one shared space, so a photo and its true caption end up in the same spot. Today you become that matchmaker.
@goal Together we'll build CLIP one piece at a time. You'll see why a fixed list of labels can't keep up, how two separate "readers" (one for images, one for text) can be trained to agree, what it means for a picture and a caption to "land in the same place," how we score a match, how the whole thing learns by pulling true pairs together and pushing mismatches apart, the three knobs that make that learning actually work, and the party trick — recognizing things it was never directly taught, just by *describing* them. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="Why not just labels?" title="A fixed list of labels can't name new things" gotit="Got the gap"
Let's start with the way pictures were named *before* CLIP, and the wall it hits. The old way: pick a fixed list of, say, 1000 names — "cat," "dog," "car," "airplane" — and train a network to pick one name for each photo. This is [[classification||Naming the single most likely thing in a photo by choosing from a fixed, pre-decided list of labels. The list never grows on its own.]] with a fixed label list. It works great — as long as the thing in the photo is *on the list*.

Think of a **restaurant with a printed menu.** You can order anything on the menu, fast and easy. But the moment you want something *not printed* — "a peanut-butter-and-banana sandwich, please" — the kitchen just shrugs: it's not on the menu, so it can't be made. A fixed label list is that menu. Show the network a "corgi wearing sunglasses" and, if that exact name isn't on its 1000-item menu, it *cannot* say it. It can only ever answer with names someone typed in ahead of time.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A restaurant-menu analogy for a fixed label list. On the left, a printed menu titled fixed labels lists cat, dog, car, airplane. A photo of a cat is matched to the menu item cat with a green check. On the right, a photo of a corgi wearing sunglasses points at the same menu, but there is no matching item, marked with a red cross and the words not on the menu. A caption says a fixed label list can only name things someone typed in ahead of time.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A fixed label list is a printed menu — new things aren't on it</text>
<g transform="translate(180,34)"><rect x="0" y="0" width="160" height="120" fill="#FDF9EF" stroke="#C99A12"/><text x="80" y="18" fill="#8A6D3B" font-size="9">MENU · fixed labels</text><text x="80" y="42" fill="#6B645E" font-size="9">cat</text><text x="80" y="60" fill="#6B645E" font-size="9">dog</text><text x="80" y="78" fill="#6B645E" font-size="9">car</text><text x="80" y="96" fill="#6B645E" font-size="9">airplane</text><text x="80" y="114" fill="#B8AEA2" font-size="8">…1000 names, and no more</text></g>
<g transform="translate(24,54)"><rect x="0" y="0" width="120" height="76" fill="#FCFBF8" stroke="#E4DED4"/><text x="60" y="46" font-size="24">🐱</text><text x="60" y="70" fill="#276b45" font-size="8">"cat" ✓ on menu</text></g>
<line x1="146" y1="92" x2="178" y2="80" stroke="#2D8B55" stroke-width="1.5"/>
<g transform="translate(376,54)"><rect x="0" y="0" width="128" height="76" fill="#FCFBF8" stroke="#E4DED4"/><text x="64" y="44" font-size="20">🐶🕶️</text><text x="64" y="70" fill="#8a3b3b" font-size="8">"corgi in sunglasses"</text></g>
<line x1="376" y1="92" x2="342" y2="80" stroke="#C93B3B" stroke-width="1.5" stroke-dasharray="3,2"/><text x="360" y="180" fill="#8a3b3b" font-size="9" text-anchor="middle">✗ not on the menu — can't be named</text></g></svg>
%%%

**What the menu picture gets right:** a fixed menu is quick and reliable for anything printed on it, but useless for anything that isn't — exactly how a fixed label list works. **Where it breaks down:** a real menu is short on purpose, but the deeper problem is that *the world has endless things to name*, and no list, however long, can cover them all in advance.

#### The fix: let words themselves be the labels
Here's the escape. Instead of a short fixed menu, what if the "label" could be *any sentence at all*? The internet is full of pictures that already come with a caption — a photo of a beach next to the words "sunset over the ocean," a product photo next to "red running shoes." Those free caption pairs are called [[image-text pairs||A picture together with a sentence that describes it — for example a beach photo next to the caption "a sunset over the ocean." The internet has enormous numbers of them, already paired, for free.]], and there are huge numbers of them online, already matched, at no cost.

So the new plan is: don't hand-pick 1000 labels. Instead, learn from a giant pile of picture-and-caption pairs, and let *natural language* — any phrase a person might write — be the label. This is called [[natural-language supervision||Learning from free-form captions written by people, instead of from a short hand-made list of category names. The "correct answer" for each photo is its real caption, in plain words.]]. The menu just became endless: any sentence you can write is now a possible label.

!!! c-info 💡
<b>Where you've heard of this:</b> this switch — from a fixed label list to "learn from captions on the web" — is the whole reason a model can later recognize a "corgi in sunglasses" it was never explicitly trained to name. It's the seed of the <b>matchmaker</b> idea for the rest of today.
!!!

**You now see the gap CLIP fills:** a fixed menu can't name new things, so we let free captions be the labels. Next: how do we even *read* a picture and a sentence so we can compare them? We need two readers.


@@@ concept id=c2 tag="Two readers" title="Two encoders that map into one shared space" gotit="Got the two encoders"
To compare a picture with a sentence, we first have to turn *each* of them into the *same kind of thing* — a list of numbers. So CLIP uses two separate readers. One reads images and turns each photo into a list of numbers. The other reads text and turns each caption into a list of numbers. A reader like this is called an [[encoder||A network that reads one kind of input — a photo, or a sentence — and squeezes it down to a short list of numbers that captures its meaning.]]: the [[image encoder||The reader that turns a photo into a list of numbers. It's a vision network — a CNN or a Vision Transformer — like the ones you built earlier this module.]] reads photos, the [[text encoder||The reader that turns a sentence into a list of numbers. It reads the words of a caption and summarizes their meaning.]] reads captions. That short list of numbers standing for a photo or a caption is its [[embedding||The short list of numbers an encoder produces to stand for an input. Similar things get similar lists; you can think of it as a point in space.]].

Think of **two translators at the United Nations.** One only speaks French, one only speaks Japanese — totally different languages, they can't understand each other directly. But both are trained to translate into *one shared language*, say English. Once both have translated into English, you can finally *compare* what each person said. CLIP's two encoders are those translators: images speak "pixel," text speaks "words," but both get translated into one shared number-language — so a photo and a caption become comparable at last.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A two-translators analogy for dual encoders mapping into one shared space. On the top left a photo goes into a box labelled image encoder and comes out as a list of numbers. On the bottom left a caption a photo of a dog goes into a box labelled text encoder and comes out as a list of numbers. Both arrows point to the right into one shared box labelled shared embedding space, where an image dot and a text dot sit near each other. A caption says two separate readers translate a picture and a sentence into one shared number-space.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Two encoders → one shared space where photo and caption can be compared</text>
<g transform="translate(20,36)"><rect x="0" y="0" width="70" height="46" fill="#FCFBF8" stroke="#E4DED4"/><text x="35" y="30" font-size="20">🐶</text><text x="35" y="-6" fill="#6B645E">a photo</text></g>
<g transform="translate(110,36)"><rect x="0" y="0" width="96" height="46" fill="#EAF0FA" stroke="#5E5191"/><text x="48" y="22" fill="#5E5191">image encoder</text><text x="48" y="38" fill="#8a83b0" font-size="7">CNN / ViT</text></g>
<text x="214" y="63" fill="#B8AEA2" font-size="14">→</text>
<g transform="translate(20,124)"><rect x="0" y="0" width="70" height="46" fill="#FCFBF8" stroke="#E4DED4"/><text x="35" y="22" font-size="8">"a photo</text><text x="35" y="34" font-size="8">of a dog"</text><text x="35" y="-6" fill="#6B645E">a caption</text></g>
<g transform="translate(110,124)"><rect x="0" y="0" width="96" height="46" fill="#EAF5EE" stroke="#2D8B55"/><text x="48" y="22" fill="#276b45">text encoder</text><text x="48" y="38" fill="#5aa27a" font-size="7">reads words</text></g>
<text x="214" y="151" fill="#B8AEA2" font-size="14">→</text>
<g transform="translate(300,44)"><rect x="0" y="0" width="200" height="128" fill="#FDF9EF" stroke="#C99A12" rx="4"/><text x="100" y="18" fill="#8A6D3B">shared embedding space</text><circle cx="118" cy="66" r="7" fill="#5E5191"/><text x="118" y="52" fill="#5E5191" font-size="7">image dot</text><circle cx="132" cy="78" r="7" fill="#2D8B55"/><text x="150" y="94" fill="#276b45" font-size="7">text dot</text><text x="100" y="118" fill="#6B645E" font-size="7.5">a dog photo & "a dog" land near each other</text></g></g></svg>
%%%

**What the two-translators picture gets right:** two very different inputs each get turned into one shared language, so they finally become comparable — exactly what the two encoders do. **Where it breaks down:** human translators already *know* English before they start, but CLIP's two encoders have to *learn together* what the shared number-language means — and how they learn it is the heart of today's lesson.

#### One space, two doors in
Notice the shape of this design. There are *two* encoders — two different networks — but they aim at *one* destination: a single [[shared embedding space||One common "map" of number-lists that both encoders point into. A photo and its caption should land at nearby spots on this same map; a photo and a wrong caption should land far apart.]]. The image encoder is a vision network (a CNN or a Vision Transformer — the very things you built earlier this module). The text encoder reads the words of a caption. Different doors, same room.

%%% demo id=encoders label="see two inputs become two number-lists in one space"
code: img_vec  = image_encoder(dog_photo)      # read the photo
      text_vec = text_encoder("a photo of a dog")  # read the caption
out: img_vec  shape (512,)   # the photo as 512 numbers  → a point in space
     text_vec shape (512,)   # the caption as 512 numbers → a point in the SAME space
     both live in the same 512-dim space, so they can be compared
     goal of training: make these two points land near each other
take: <b>Two encoders, two lists of numbers, one shared space.</b> Because both lists have the same length and live in the same space, we can finally ask "how close are they?" — which is the next concept.
%%%

**You now have the machine's shape: two readers pointing into one shared space.** But "the same space" is only useful if we can *measure* how close two points are. And to measure fairly, we first tidy the points up — that's next.

@@@ concept id=c3 tag="Same length arrows" title="Normalize the arrows, then score the match by direction" gotit="Got cosine similarity"
Now that a photo and a caption are both points in one space, we want a single number for "how well do these two match?" There's a small trap first. If one arrow is long and another is short, comparing their raw numbers is unfair — the long one looks "bigger" even if it points the *same way*. So before comparing, CLIP makes every arrow the *same length*. Stretching or shrinking an arrow to length 1 is called [[L2 normalization||Rescaling a list of numbers so the arrow it draws has length exactly 1. It keeps the direction the same but throws away the length, so only direction matters.]] — it keeps the *direction* but throws away the length. After this, every point sits on a ball of radius 1, a [[unit sphere||The surface of a ball of radius 1. After normalization, every embedding sits on this ball, so two embeddings differ only by the direction they point — never by length.]].

Think of the **hands of two clocks.** You don't care that the hour hand is short and the minute hand is long — you only care *which way each one points*. Two clocks "agree" when their hands point the same direction, no matter the hand length. Normalizing is like trimming every clock hand to the same length so only the *angle* is left to compare. Then the match score is simply: how small is the angle between the two arrows? Point the same way → great match. Point at right angles → unrelated. Point opposite → total mismatch. This "how aligned are the two directions" score is called [[cosine similarity||A match score based only on the angle between two arrows: +1 if they point the same way, 0 if at right angles, −1 if opposite. After normalization it's just the arrows multiplied together, number by number, and summed.]].

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A clock-hands analogy for cosine similarity on a unit sphere. A circle of radius 1 is drawn. Two arrows from the center point in nearly the same direction with a small angle between them, labelled match, cosine near plus one. A second pair points at right angles, labelled unrelated, cosine near zero. A third pair points in opposite directions, labelled mismatch, cosine near minus one. A caption says after normalizing to the same length, the match score is just the angle: same direction is a match, opposite is a mismatch.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Trim every arrow to length 1 — then score by the ANGLE between them</text>
<g transform="translate(90,120)"><circle cx="0" cy="0" r="56" fill="#EAF5EE" stroke="#2D8B55" stroke-opacity="0.5"/><line x1="0" y1="0" x2="44" y2="-34" stroke="#5E5191" stroke-width="2.5"/><line x1="0" y1="0" x2="52" y2="-20" stroke="#2D8B55" stroke-width="2.5"/><text x="0" y="82" fill="#276b45" font-size="9">match</text><text x="0" y="96" fill="#6B645E" font-size="7.5">cos ≈ +1</text></g>
<g transform="translate(260,120)"><circle cx="0" cy="0" r="56" fill="#FDF9EF" stroke="#C99A12" stroke-opacity="0.5"/><line x1="0" y1="0" x2="0" y2="-52" stroke="#5E5191" stroke-width="2.5"/><line x1="0" y1="0" x2="52" y2="0" stroke="#2D8B55" stroke-width="2.5"/><text x="0" y="82" fill="#8A6D3B" font-size="9">unrelated</text><text x="0" y="96" fill="#6B645E" font-size="7.5">cos ≈ 0 (right angle)</text></g>
<g transform="translate(430,120)"><circle cx="0" cy="0" r="56" fill="#FDECEC" stroke="#C93B3B" stroke-opacity="0.5"/><line x1="0" y1="0" x2="48" y2="-24" stroke="#5E5191" stroke-width="2.5"/><line x1="0" y1="0" x2="-48" y2="24" stroke="#2D8B55" stroke-width="2.5"/><text x="0" y="82" fill="#8a3b3b" font-size="9">mismatch</text><text x="0" y="96" fill="#6B645E" font-size="7.5">cos ≈ −1 (opposite)</text></g></g></svg>
%%%

**What the clock-hands picture gets right:** you compare two things by which way they point, ignoring their length — exactly what normalization plus cosine similarity does. **Where it breaks down:** a clock has only two hands in a flat circle, but these arrows live in a space with hundreds of directions at once, so "the angle" is a math computation, not something you can eyeball.

#### One narrated line, then a worked number
Here's the whole score in one plain line: **after both arrows are trimmed to length 1, the cosine similarity is just their two number-lists multiplied together, position by position, and added up.** That single sum lands between −1 (opposite) and +1 (same direction). Let's watch it on tiny arrows so the number feels real — predict the sign before you peek:

%%% demo id=cosine label="normalize two arrows, then score the match by direction"
code: img  = [3.0, 4.0]                 # a short 2-number arrow (length 5)
      good = [6.0, 8.0]                 # points the SAME way (length 10)
      bad  = [-8.0, 6.0]                # points at a right angle
      # step 1 — trim each to length 1 (L2 normalize):
      img_n  = [0.6, 0.8]               # 3/5, 4/5
      good_n = [0.6, 0.8]               # 6/10, 8/10  → identical direction!
      bad_n  = [-0.8, 0.6]
out: cos(img, good) = 0.6*0.6 + 0.8*0.8 = 0.36 + 0.64 = 1.00   # same direction → perfect match
     cos(img, bad)  = 0.6*(-0.8) + 0.8*0.6 = -0.48 + 0.48 = 0.00  # right angle → unrelated
     raw lengths (5 vs 10) stopped mattering the moment we normalized
take: <b>Same direction scores +1; a right angle scores 0.</b> Normalizing killed the "length 5 vs 10" unfairness — only direction is left. This one number, cosine similarity, is how CLIP scores every photo-against-caption match.
%%%

!!! c-info 💡
<b>Optional (skippable) — why not just multiply the raw arrows?</b> Multiplying two raw arrows and summing is the <i>dot product</i>. On its own it grows when an arrow is simply longer, so a long-but-wrong arrow can beat a short-but-right one. Normalizing first (dividing by each arrow's length) cancels the length entirely, leaving only the angle — that's exactly why CLIP normalizes before it scores.
!!!

**You can now score any photo-vs-caption pair with one number in [−1, 1].** But CLIP never scores just one pair at a time — it scores a whole *batch* of photos against a whole batch of captions at once, in a grid. That grid is next.

@@@ concept id=c4 tag="The match grid" title="Score every photo against every caption: a grid of matches" gotit="Got the match grid"
Here's the clever move that makes CLIP a **matchmaker** and not just a scorer. During training, CLIP grabs a bunch of image-text pairs *at once* — say 8 photos, each with its own true caption — and scores *every* photo against *every* caption. That fills in a grid: 8 photos down the side, 8 captions across the top, and in each cell the cosine similarity of that photo with that caption. This grid is the [[similarity matrix||A grid of match scores: every photo in the batch scored against every caption in the batch. Photo i vs caption j sits in cell (i, j).]].

Think of a **round-robin sports tournament, where every team plays every other team.** You make a big grid: teams down the side, teams across the top, and each cell records the result of that matchup. CLIP's grid is the same — but instead of teams playing teams, it's *photos matched against captions*. And here's the key: the cells on the **diagonal** — where photo 1 meets caption 1, photo 2 meets caption 2 — are the *true* pairs (each photo with its own real caption). Every *off*-diagonal cell is a *wrong* pairing (photo 1 with someone else's caption). A true pair is a [[positive pair||A photo together with its own real caption — the matchup we WANT to score high. These sit on the diagonal of the grid.]]; a wrong pairing is a [[negative pair||A photo paired with a caption that is NOT its own — a matchup we want to score LOW. These are all the off-diagonal cells.]].

%%% svg
<svg viewBox="0 0 520 230" role="img" aria-label="A round-robin tournament grid analogy for the similarity matrix. A four-by-four grid has four photos down the left side and four captions across the top. Each cell holds a match score. The four diagonal cells, where photo one meets caption one and so on, are highlighted green and labelled positive pairs, the true matches we want high. All twelve off-diagonal cells are shaded red and labelled negative pairs, wrong matches we want low. A caption says the diagonal is the true pairs, everything off the diagonal is a mismatch.">
<g font-family="monospace" font-size="8" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Similarity matrix: every photo × every caption · diagonal = true pairs</text>
<text x="150" y="34" fill="#276b45" font-size="8">captions →</text>
<text x="34" y="130" fill="#5E5191" font-size="8" transform="rotate(-90 34 130)">photos ↓</text>
<g transform="translate(70,42)"><g font-size="7" fill="#276b45"><text x="40" y="8">cap1</text><text x="90" y="8">cap2</text><text x="140" y="8">cap3</text><text x="190" y="8">cap4</text></g><g font-size="7" fill="#5E5191"><text x="8" y="42">img1</text><text x="8" y="82">img2</text><text x="8" y="122">img3</text><text x="8" y="162">img4</text></g>
<g stroke="#E4DED4"><rect x="20" y="20" width="40" height="40" fill="#EAF5EE"/><rect x="70" y="20" width="40" height="40" fill="#FDECEC"/><rect x="120" y="20" width="40" height="40" fill="#FDECEC"/><rect x="170" y="20" width="40" height="40" fill="#FDECEC"/>
<rect x="20" y="60" width="40" height="40" fill="#FDECEC"/><rect x="70" y="60" width="40" height="40" fill="#EAF5EE"/><rect x="120" y="60" width="40" height="40" fill="#FDECEC"/><rect x="170" y="60" width="40" height="40" fill="#FDECEC"/>
<rect x="20" y="100" width="40" height="40" fill="#FDECEC"/><rect x="70" y="100" width="40" height="40" fill="#FDECEC"/><rect x="120" y="100" width="40" height="40" fill="#EAF5EE"/><rect x="170" y="100" width="40" height="40" fill="#FDECEC"/>
<rect x="20" y="140" width="40" height="40" fill="#FDECEC"/><rect x="70" y="140" width="40" height="40" fill="#FDECEC"/><rect x="120" y="140" width="40" height="40" fill="#FDECEC"/><rect x="170" y="140" width="40" height="40" fill="#EAF5EE"/></g>
<g stroke="#2D8B55" stroke-width="2" fill="none"><rect x="20" y="20" width="40" height="40"/><rect x="70" y="60" width="40" height="40"/><rect x="120" y="100" width="40" height="40"/><rect x="170" y="140" width="40" height="40"/></g>
<text x="40" y="44" fill="#276b45">.98</text><text x="90" y="84" fill="#276b45">.95</text><text x="140" y="124" fill="#276b45">.97</text><text x="190" y="164" fill="#276b45">.96</text></g>
<g transform="translate(300,60)"><rect x="0" y="0" width="14" height="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="112" y="11" fill="#276b45" font-size="8">diagonal = POSITIVE pairs (want HIGH)</text><rect x="0" y="26" width="14" height="14" fill="#FDECEC" stroke="#C93B3B"/><text x="120" y="37" fill="#8a3b3b" font-size="8">off-diagonal = NEGATIVE pairs (want LOW)</text><text x="100" y="66" fill="#6B645E" font-size="7.5">1 batch of 4 → 4 true pairs, 12 mismatches</text></g></g></svg>
%%%

**What the tournament-grid picture gets right:** you lay every contestant against every other in a grid, and each cell holds one matchup's score — exactly the similarity matrix. **Where it breaks down:** in a tournament every matchup is a *real* game you care about, but here we only *want* the diagonal to win — the off-diagonal matchups exist only to be pushed *down*, as you'll see next.

#### Free wrong answers, right there in the batch
Here's the sly part. To teach the matchmaker "this is *not* a match," you need examples of wrong pairings. Where do they come from? You get them for free: in a batch of 8 pairs, photo 1's *own* caption is the right answer, and the *other 7 captions* are ready-made wrong answers. Using the other pairs in the same batch as the wrong answers is called using [[in-batch negatives||Using the OTHER captions in the same batch as the wrong answers for each photo. No extra data needed — the batch hands you the mismatches for free.]]. So a batch of 8 gives, for each photo, 1 right caption and 7 wrong ones — no extra work.

%%% demo id=grid label="build the match grid for a tiny batch of 3 pairs"
code: batch = [ (dog_photo,  "a photo of a dog"),      # pair 0
                (beach_photo,"a sunset over the ocean"),# pair 1
                (car_photo,  "a red sports car") ]     # pair 2
      S = cosine(all_image_vecs, all_text_vecs)         # 3x3 grid of scores
out: similarity matrix S (rows = photos, cols = captions):
          "dog"   "sunset"  "car"
     dog [ 0.97     0.05     0.11 ]   ← row 0: want the DIAGONAL (0.97) to win
   beach [ 0.08     0.96     0.06 ]   ← row 1: diagonal 0.96 should beat the rest
     car [ 0.10     0.04     0.98 ]   ← row 2: diagonal 0.98 is the true match
     diagonal (0.97, 0.96, 0.98) = the 3 TRUE pairs → want these highest
     off-diagonal (6 cells) = mismatches → want these low
take: <b>The correct answer for each photo is on the diagonal; the batch hands you the wrong answers for free.</b> Training's whole job: make each row's diagonal cell the biggest in its row (and each column's too). How? That's the contrastive loss — next.
%%%

**You now have the grid: diagonal = true pairs to lift, off-diagonal = mismatches to push down.** But how do we actually *train* the encoders to make the diagonal win? That single rule — pull together, push apart — is the heart of CLIP, and it's next.

@@@ concept id=c5 tag="Pull & push" title="The contrastive loss: pull true pairs together, push mismatches apart" gotit="Got the contrastive idea"
This is the beating heart of CLIP. The training rule is beautifully simple: for each photo, **pull** its own caption *closer* and **push** every other caption *farther away*. Do that over and over, for millions of pairs, and the two encoders slowly learn to place each photo and its true caption at the same spot. A loss that trains by *contrasting* right answers against wrong ones is called a [[contrastive loss||A training rule that pulls each item toward its ONE correct partner and pushes it away from all the wrong ones at the same time. It learns by contrast: right vs. wrong.]].

Think of a **magnet game.** Give each photo a magnet. Its true caption's magnet is set to *attract* — they snap together. Every wrong caption's magnet is set to *repel* — they shove apart. Let go, and each photo drifts until it's hugging its true caption and far from all the others. The contrastive loss is that set of magnets: attraction on the diagonal (true pairs), repulsion everywhere off it (mismatches). And because we do it *both* ways — each photo picks its caption, *and* each caption picks its photo — it's a two-direction, or [[symmetric loss||Scoring the match from both sides at once: each photo must pick its true caption, AND each caption must pick its true photo. The two scores are averaged.]], rule.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A magnet game analogy for the contrastive loss. On the left, labelled before, a photo dot sits in the middle of scattered caption dots including its true caption, all mixed together. Green attract arrows pull the true caption toward the photo and red repel arrows push the wrong captions away. On the right, labelled after, the photo dot and its true caption dot sit snugly together while all the wrong caption dots have been pushed to the edges. A caption says pull the true pair together, push every mismatch apart.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Contrastive loss = magnets: attract the true caption, repel the rest</text>
<g transform="translate(24,30)"><rect x="0" y="0" width="210" height="160" fill="#FDF9EF" stroke="#C99A12"/><text x="105" y="16" fill="#8A6D3B">BEFORE: all mixed up</text><circle cx="105" cy="90" r="9" fill="#5E5191"/><text x="105" y="94" fill="#fff" font-size="7">img</text><circle cx="150" cy="70" r="7" fill="#2D8B55"/><text x="150" y="58" fill="#276b45" font-size="6.5">true cap</text><circle cx="60" cy="60" r="6" fill="#C93B3B"/><circle cx="70" cy="128" r="6" fill="#C93B3B"/><circle cx="160" cy="130" r="6" fill="#C93B3B"/>
<line x1="114" y1="86" x2="141" y2="73" stroke="#2D8B55" stroke-width="1.5"/><text x="130" y="88" fill="#276b45" font-size="6">attract</text>
<line x1="96" y1="86" x2="66" y2="64" stroke="#C93B3B" stroke-width="1" stroke-dasharray="2,2"/><line x1="100" y1="99" x2="72" y2="122" stroke="#C93B3B" stroke-width="1" stroke-dasharray="2,2"/><line x1="114" y1="96" x2="156" y2="124" stroke="#C93B3B" stroke-width="1" stroke-dasharray="2,2"/><text x="60" y="150" fill="#8a3b3b" font-size="6">repel wrong caps</text></g>
<text x="248" y="114" fill="#B8AEA2" font-size="16">→</text>
<g transform="translate(286,30)"><rect x="0" y="0" width="210" height="160" fill="#EAF5EE" stroke="#2D8B55"/><text x="105" y="16" fill="#276b45">AFTER: sorted</text><circle cx="103" cy="86" r="9" fill="#5E5191"/><circle cx="115" cy="90" r="7" fill="#2D8B55"/><text x="109" y="112" fill="#276b45" font-size="6.5">true pair snapped together</text><circle cx="24" cy="34" r="6" fill="#C93B3B"/><circle cx="186" cy="40" r="6" fill="#C93B3B"/><circle cx="190" cy="140" r="6" fill="#C93B3B"/><text x="105" y="150" fill="#8a3b3b" font-size="6.5">mismatches pushed to the edges</text></g></g></svg>
%%%

**What the magnet picture gets right:** one attraction and many repulsions, acting at once, sort each photo next to its true caption and away from the rest — exactly the contrastive loss. **Where it breaks down:** real magnets pull with a fixed strength forever, but here the "pull" and "push" *fade* as training succeeds — once a photo already hugs its caption, there's little left to correct.

#### Before → after: what happens with NO push
Why bother pushing mismatches apart at all — why not *only* pull true pairs together? Here's the trap, and it's a real failure mode. If you only ever say "these two should be close" and never "these two should be far," the encoders find a lazy cheat: **map every single input to the exact same point.** Then every pair is "close" and the pull-only rule is perfectly happy — but the space is useless, because everything is on top of everything. Every embedding becoming the same dull point is called [[representation collapse||The failure where the encoders map every input to the same point, so all embeddings are identical and useless. It happens when you only pull positives together and never push negatives apart.]].

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A before and after figure contrasting pull-only collapse with pull-and-push. On the left, labelled pull only, all dots — every image and every caption — have piled onto one single point in the center, labelled everything collapses to one spot, useless. On the right, labelled pull AND push, the dots are spread out in tidy image-caption pairs across the space, each true pair together but different pairs far apart, labelled a useful spread-out space. A caption says pushing mismatches apart is the cure for collapse.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Why we PUSH too: pull-only collapses everything to one point</text>
<g transform="translate(24,30)"><rect x="0" y="0" width="210" height="130" fill="#FDECEC" stroke="#C93B3B"/><text x="105" y="16" fill="#8a3b3b">PULL ONLY ✗</text><circle cx="105" cy="70" r="12" fill="#5E5191" fill-opacity="0.5"/><circle cx="105" cy="70" r="8" fill="#2D8B55" fill-opacity="0.6"/><circle cx="105" cy="70" r="4" fill="#C93B3B"/><text x="105" y="104" fill="#8a3b3b" font-size="7.5">everything piled on ONE spot</text><text x="105" y="118" fill="#8a3b3b" font-size="7.5">= collapse, useless</text></g>
<text x="248" y="98" fill="#B8AEA2" font-size="16">vs</text>
<g transform="translate(286,30)"><rect x="0" y="0" width="210" height="130" fill="#EAF5EE" stroke="#2D8B55"/><text x="105" y="16" fill="#276b45">PULL + PUSH ✓</text><g><circle cx="40" cy="44" r="6" fill="#5E5191"/><circle cx="50" cy="48" r="5" fill="#2D8B55"/><circle cx="150" cy="40" r="6" fill="#5E5191"/><circle cx="160" cy="46" r="5" fill="#2D8B55"/><circle cx="60" cy="104" r="6" fill="#5E5191"/><circle cx="70" cy="108" r="5" fill="#2D8B55"/><circle cx="165" cy="100" r="6" fill="#5E5191"/><circle cx="175" cy="104" r="5" fill="#2D8B55"/></g><text x="105" y="124" fill="#276b45" font-size="7.5">true pairs together, pairs spread apart</text></g></g></svg>
%%%

So pushing mismatches apart isn't a bonus — it's the *cure* that stops the encoders from cheating. The in-batch negatives you met last concept are exactly what supplies that push.

!!! c-warn ⚠️
<b>Failure mode → cause → cure:</b> <b>representation collapse</b> — every embedding becomes the same useless point. <i>Cause:</i> only pulling true pairs together, with no force pushing mismatches apart. <i>Cure:</i> the contrastive loss with in-batch negatives — the "push" that keeps the space spread out. This is why CLIP always trains with negatives, never with attraction alone.
!!!

**You now understand the core: pull true pairs together, push mismatches apart — and the push is what prevents collapse.** But this magnet game only works if a few knobs are set right. Three of them make or break training — that's the next concept.

@@@ concept id=c6 tag="Three knobs" title="Three knobs that make the matchmaker actually learn" gotit="Got the three knobs"
The pull-and-push idea is right, but it only *works* if three dials are set well. Each dial fixes one thing that would otherwise stall training. Meet all three at once with one everyday picture, then take them one at a time.

Think of **tuning a guitar before a show.** Three separate adjustments matter: the *volume* (too quiet and no one reacts, too loud and it's noise), how *many strings* you can compare against, and the *quality of the strings* themselves. Get any one wrong and the song falls flat. CLIP has three matching dials: a **temperature** (the volume of the match scores), the **batch size** (how many mismatches you compare against), and the **scale of the data** (how many, and how clean, the pairs are).

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A guitar-tuning analogy for CLIP's three training knobs, shown as three dials in a row. The first dial is labelled temperature, the volume of the scores, turning cramped scores into a sharp spread. The second dial is labelled batch size, how many mismatches to compare against, from few to many. The third dial is labelled data scale, how many and how clean the pairs are, from small and noisy to huge and robust. A caption says three dials — temperature, batch size, data scale — must all be set well for training to work.">
<g font-family="monospace" font-size="8" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Three dials to tune the matchmaker — set them all well</text>
<g transform="translate(90,110)"><circle cx="0" cy="0" r="40" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><line x1="0" y1="0" x2="24" y2="-24" stroke="#5E5191" stroke-width="3"/><circle cx="0" cy="0" r="4" fill="#5E5191"/><text x="0" y="66" fill="#5E5191" font-size="9">① temperature</text><text x="0" y="80" fill="#6B645E" font-size="7">"volume" of scores</text></g>
<g transform="translate(260,110)"><circle cx="0" cy="0" r="40" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><line x1="0" y1="0" x2="0" y2="-30" stroke="#2D8B55" stroke-width="3"/><circle cx="0" cy="0" r="4" fill="#2D8B55"/><text x="0" y="66" fill="#276b45" font-size="9">② batch size</text><text x="0" y="80" fill="#6B645E" font-size="7"># mismatches to beat</text></g>
<g transform="translate(430,110)"><circle cx="0" cy="0" r="40" fill="#FDF9EF" stroke="#C99A12" stroke-width="2"/><line x1="0" y1="0" x2="-24" y2="-24" stroke="#C99A12" stroke-width="3"/><circle cx="0" cy="0" r="4" fill="#C99A12"/><text x="0" y="66" fill="#8A6D3B" font-size="9">③ data scale</text><text x="0" y="80" fill="#6B645E" font-size="7">how many & how clean</text></g></g></svg>
%%%

**What the guitar picture gets right:** several independent adjustments each matter, and getting any one wrong ruins the result — exactly like these three dials. **Where it breaks down:** a guitarist tunes by ear in seconds, but CLIP's dials are set by careful experiments over huge training runs, not by listening.

#### Dial ① — temperature: turn up the volume on the scores
First puzzle. Cosine similarity is always trapped between −1 and +1. That's a tiny range — a true pair at 0.30 and a wrong pair at 0.25 are almost the same number, so the "pull vs push" signal is *faint* and the model barely learns. The fix: multiply every score by a factor before comparing, to *stretch* the gaps. That factor is the [[temperature||A single learned number that scales the match scores before the pull-push comparison. It stretches the cramped −1 to +1 range into a sharper spread, so the right answer clearly beats the wrong ones.]] (CLIP even *learns* the best value for it during training).

%%% demo id=temp label="watch temperature stretch cramped scores into a clear winner"
code: scores = [0.30, 0.25, 0.22, 0.20]   # true pair (0.30) vs 3 wrong — barely different!
      # multiply by a temperature factor to stretch the gaps:
out: temperature OFF (raw): softmax → [0.27, 0.26, 0.25, 0.24]   # nearly a tie → weak signal, model won't learn
     temperature = 20      : scores*20 → [6.0, 5.0, 4.4, 4.0]
                            softmax    → [0.62, 0.23, 0.10, 0.05]  # true pair clearly wins → strong signal ✓
     same scores, but stretching makes the right answer stand out
take: <b>Temperature stretches the cramped [−1, 1] range so the right answer clearly beats the rest.</b> Without it the pull-push signal is too faint to train. CLIP learns the exact temperature during training instead of guessing it.
%%%

!!! c-warn ⚠️
<b>Failure mode → cause → cure:</b> <b>the loss barely trains</b> — the model can't tell right from wrong pairs. <i>Cause:</i> raw cosine scores are cramped in [−1, 1], so true and wrong pairs look nearly equal. <i>Cure:</i> a learnable <b>temperature</b> that stretches the scores into a sharp spread.
!!!

#### Dial ② — batch size: more mismatches = a stronger push
Second puzzle. Each photo learns "not this" from the *other* captions in its batch (the in-batch negatives). If the batch has only 4 pairs, each photo pushes against just 3 wrong captions — a weak lesson. If the batch has *thousands* of pairs, each photo pushes against *thousands* of wrong captions — a rich, strong lesson. So a bigger [[batch size||How many image-text pairs CLIP looks at in one training step. A bigger batch gives each photo more wrong captions to push against, so the learning signal is stronger.]] means more negatives per step, which means a much stronger contrastive signal. This is why CLIP is famous for training with *very* large batches.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A before and after bar figure showing batch size and number of negatives. On the left a small batch of four pairs gives each photo only three wrong captions to push against, drawn as a short weak-signal bar. On the right a large batch of thousands of pairs gives each photo thousands of wrong captions, drawn as a tall strong-signal bar. A caption says a bigger batch supplies more negatives, so the push is stronger.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Bigger batch → more mismatches to push against → stronger signal</text>
<g><rect x="90" y="98" width="90" height="26" fill="#FDECEC" stroke="#C93B3B"/><text x="135" y="115" fill="#8a3b3b" font-size="8">batch = 4</text><text x="135" y="140" fill="#6B645E" font-size="7.5">3 negatives · weak push</text></g>
<text x="260" y="112" fill="#B8AEA2" font-size="14">→</text>
<g><rect x="320" y="44" width="150" height="80" fill="#EAF5EE" stroke="#2D8B55"/><text x="395" y="80" fill="#276b45" font-size="8">batch = thousands</text><text x="395" y="96" fill="#276b45" font-size="8">many negatives</text><text x="395" y="140" fill="#6B645E" font-size="7.5">strong push · learns fast</text></g></g></svg>
%%%

!!! c-warn ⚠️
<b>Failure mode → cause → cure:</b> <b>a weak learning signal</b> — training crawls and the space stays fuzzy. <i>Cause:</i> a small batch gives each photo only a handful of wrong captions to push against. <i>Cure:</i> a <b>large batch size</b>, so every step supplies many in-batch negatives.
!!!

#### Dial ③ — data scale: drown out the noisy captions
Third puzzle, and a subtle one. The web captions are *not* clean. Sometimes two photos in a batch genuinely match a caption ("a dog" fits *both* dog photos), so treating one as a "wrong answer" is actually a mistake — a [[false negative||A pair CLIP pushes apart even though it is genuinely a good match — for example calling a second real dog photo a "wrong" answer for the caption "a dog." Noisy web pairs cause these.]]. A single bad caption would poison a small dataset. The cure is *scale*: train on such a huge pile of pairs that the occasional bad one is a drop in the ocean — the rare mistakes average out, and the true pattern shines through. Learning from a giant collection of pairs is called using [[web-scale data||A training set of an enormous number of image-text pairs collected from the web. So big that noisy or wrong captions average out and stop mattering.]].

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A drop-in-the-ocean figure for data scale washing out noise. On the left a small dataset is shown as a few pairs where one red bad caption is a big fraction and poisons the set. On the right a web-scale dataset is shown as a vast field of green good pairs with a single tiny red bad one, labelled the noise averages out. A caption says at huge scale the occasional wrong caption stops mattering.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Data scale: at huge scale, a few noisy captions wash out</text>
<g transform="translate(24,30)"><rect x="0" y="0" width="200" height="106" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="16" fill="#8a3b3b">small data ✗</text><g><circle cx="50" cy="50" r="9" fill="#2D8B55"/><circle cx="90" cy="50" r="9" fill="#2D8B55"/><circle cx="130" cy="50" r="9" fill="#C93B3B"/><circle cx="70" cy="82" r="9" fill="#2D8B55"/><circle cx="110" cy="82" r="9" fill="#2D8B55"/></g><text x="100" y="102" fill="#8a3b3b" font-size="7">1 bad caption = big fraction → poisons it</text></g>
<text x="248" y="86" fill="#B8AEA2" font-size="16">vs</text>
<g transform="translate(286,30)"><rect x="0" y="0" width="210" height="106" fill="#EAF5EE" stroke="#2D8B55"/><text x="105" y="16" fill="#276b45">web-scale ✓</text><g fill="#2D8B55"><circle cx="20" cy="34" r="4"/><circle cx="40" cy="34" r="4"/><circle cx="60" cy="34" r="4"/><circle cx="80" cy="34" r="4"/><circle cx="100" cy="34" r="4"/><circle cx="120" cy="34" r="4"/><circle cx="140" cy="34" r="4"/><circle cx="160" cy="34" r="4"/><circle cx="180" cy="34" r="4"/><circle cx="20" cy="54" r="4"/><circle cx="40" cy="54" r="4"/><circle cx="60" cy="54" r="4"/><circle cx="80" cy="54" r="4"/><circle cx="120" cy="54" r="4"/><circle cx="140" cy="54" r="4"/><circle cx="160" cy="54" r="4"/><circle cx="180" cy="54" r="4"/><circle cx="20" cy="74" r="4"/><circle cx="40" cy="74" r="4"/><circle cx="60" cy="74" r="4"/><circle cx="80" cy="74" r="4"/><circle cx="100" cy="74" r="4"/><circle cx="120" cy="74" r="4"/><circle cx="140" cy="74" r="4"/><circle cx="160" cy="74" r="4"/><circle cx="180" cy="74" r="4"/></g><circle cx="100" cy="54" r="4" fill="#C93B3B"/><text x="105" y="102" fill="#276b45" font-size="7">1 bad among thousands → averages out</text></g></g></svg>
%%%

!!! c-warn ⚠️
<b>Failure mode → cause → cure:</b> <b>false negatives</b> — CLIP pushes apart a pair that is actually a fine match (a second real "dog" photo for the caption "a dog"). <i>Cause:</i> uncurated, noisy web captions and near-duplicates treated as wrong answers. <i>Cure (mitigation):</i> <b>web-scale data</b> — so much of it that the rare mistakes average out and the true pattern dominates.
!!!

**You now know the three dials: temperature (stretch the scores), batch size (more negatives), and data scale (wash out noise).** With all this, CLIP trains. Now the payoff — the party trick that made CLIP famous: naming things it was never directly taught. That's next.

@@@ concept id=c7 tag="The party trick" title="Zero-shot: name things it was never directly taught" gotit="Got zero-shot"
Here's the reward for all that training — the trick that made CLIP a sensation. Once the matchmaker knows how to place photos and captions in the same space, you can ask it to classify a photo *without ever training it on those specific labels*. You just *write the labels as sentences*, let the text encoder turn each into a point, embed the photo, and pick the caption-point closest to the photo. Naming a photo among classes the model never saw during training is called [[zero-shot classification||Classifying a photo into categories the model was never explicitly trained on, just by writing each category name as a sentence and picking the caption closest to the photo. "Zero-shot" = zero training examples of those exact classes.]] — "zero-shot" meaning it needed zero training examples of those exact classes.

Think of a **friend who has read the whole library.** You point at a strange plant and ask "what's this?" Your friend has never seen *that exact plant*, but from all their reading they can say "that looks like a Venus flytrap" — matching what they *see* to *descriptions* they've read. CLIP is that well-read friend: it never trained on your specific class list, but because it learned to match pictures to *any* description, you just hand it the candidate descriptions and it picks the closest.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A well-read-friend analogy for zero-shot classification. A photo of a cat on the left is turned by the image encoder into one dot. On the right, three brand-new class names — a photo of a cat, a photo of a dog, a photo of a car — are each written as a sentence and turned by the text encoder into three dots. Lines connect the image dot to each caption dot with a match score: 0.94 to cat, 0.11 to dog, 0.08 to car. The cat caption wins, marked with a green check, so the photo is labelled cat. A caption says write the labels as sentences and pick the closest — no training on these classes needed.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Zero-shot: write class names as sentences, pick the closest match</text>
<g transform="translate(20,80)"><rect x="0" y="0" width="70" height="56" fill="#FCFBF8" stroke="#E4DED4"/><text x="35" y="36" font-size="24">🐱</text><text x="35" y="-6" fill="#6B645E">new photo</text></g>
<g transform="translate(100,88)"><rect x="0" y="0" width="70" height="40" fill="#EAF0FA" stroke="#5E5191"/><text x="35" y="24" fill="#5E5191" font-size="7.5">image enc.</text></g>
<circle cx="200" cy="108" r="9" fill="#5E5191"/><text x="200" y="134" fill="#5E5191" font-size="7">img dot</text>
<g transform="translate(300,34)" font-size="7.5"><text x="90" y="0" fill="#276b45">candidate labels, written as sentences →</text>
<rect x="20" y="12" width="180" height="24" fill="#EAF5EE" stroke="#2D8B55"/><text x="110" y="28" fill="#276b45">"a photo of a cat"   → 0.94 ✓</text>
<rect x="20" y="44" width="180" height="24" fill="#FCFBF8" stroke="#E4DED4"/><text x="110" y="60" fill="#6B645E">"a photo of a dog"   → 0.11</text>
<rect x="20" y="76" width="180" height="24" fill="#FCFBF8" stroke="#E4DED4"/><text x="110" y="92" fill="#6B645E">"a photo of a car"   → 0.08</text>
<text x="110" y="122" fill="#276b45">closest caption wins → label = "cat"</text></g>
<line x1="209" y1="104" x2="318" y2="58" stroke="#2D8B55" stroke-width="2"/><line x1="209" y1="108" x2="318" y2="90" stroke="#B8AEA2" stroke-dasharray="2,2"/><line x1="209" y1="112" x2="318" y2="122" stroke="#B8AEA2" stroke-dasharray="2,2"/></g></svg>
%%%

**What the well-read-friend picture gets right:** you can name something you never studied directly by matching what you see to descriptions you know — exactly zero-shot classification. **Where it breaks down:** a friend can ask a follow-up question or reason about the plant, but CLIP only *matches* to the descriptions you give it — it can't think further or explain *why*.

#### Watch it classify a photo with a made-up label set
The magic is that you choose the label set *at test time* — no retraining. Predict which caption wins, then run it:

%%% demo id=zeroshot label="classify a photo against brand-new labels — no retraining"
code: photo = cat_photo
      labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]  # any words you like!
      img_vec  = image_encoder(photo)
      txt_vecs = text_encoder(labels)          # embed each label-sentence
      scores   = cosine(img_vec, txt_vecs)     # match photo to each label
out: "a photo of a cat" → 0.94   ← closest → predicted label
     "a photo of a dog" → 0.11
     "a photo of a car" → 0.08
     prediction = "a photo of a cat"   (CLIP was NEVER trained on this exact label set)
take: <b>Pick the labels at test time; CLIP names the photo without any retraining.</b> Swap in "a photo of a corgi in sunglasses" and it works the same way — that's zero-shot power.
%%%

#### What CLIP can't do — the honest limits
The matchmaker is powerful but *not* magic. Knowing where it stops is what separates hype from understanding:

%%% cards
🚫 | Can't generate | CLIP only *scores* and *matches* — it can't write a caption or draw an image. It's a matchmaker, not an artist or a writer.
🔢 | Weak at counting & layout | It struggles with "three cats" vs "two cats," exact positions, reading text in images (OCR), and abstract puzzles — it captures the gist, not fine detail.
📝 | Prompt-sensitive | The exact wording changes the answer: "a photo of a cat" often beats a bare "cat." Small phrasing tweaks can swing accuracy.
🌐 | Inherits web bias | It learned from uncurated web pairs, so it carries their gaps and biases — it knows what the internet showed it, warts and all.
%%%

!!! c-info 💡
<b>Where you've heard of this:</b> because CLIP's shared space lets you match <i>any</i> text to <i>any</i> image, it became the backbone of image search, content filtering, and — most famously — the text understanding inside text-to-image generators. When you type a prompt and a model paints it, a CLIP-style shared space is often what connects your words to the picture. Those generative models live in later genAI lessons; today you built the *matching* engine they stand on.
!!!

**You've now seen the payoff: zero-shot naming from plain descriptions, plus the honest limits.** Let's gather the whole day onto one page.

@@@ concept id=c8 tag="Recap" title="Today in one page" gotit="Got the recap"
You built the whole CLIP idea today — and finished the module. Before we file it away, let's lay every piece out on one page.

Think of finishing a **recipe** and reading the finished card top to bottom. You didn't invent anything new by reading it — you *gathered* every step you already did into one clear list, in order, so you can cook it again from memory. This recap is that recipe card for CLIP: every idea you met, in the order you met them, as one connected flow.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A recipe-card analogy for the recap, showing CLIP as one connected pipeline. An image goes into an image encoder and a caption goes into a text encoder; both produce embeddings that are normalized onto a unit sphere. Cosine similarity scores every photo against every caption in a batch, forming a similarity matrix whose diagonal is the true pairs. A contrastive loss with a temperature pulls the diagonal together and pushes off-diagonal mismatches apart. The trained shared space then powers zero-shot classification. A caption reads image and text encoders to shared space to contrastive loss to zero-shot.">
<g font-family="monospace" font-size="7" text-anchor="middle"><text x="260" y="14" fill="#2C2A28" font-size="12">The CLIP recipe: two encoders → shared space → pull/push → zero-shot</text>
<g transform="translate(8,44)"><rect x="0" y="0" width="60" height="30" fill="#EAF0FA" stroke="#5E5191"/><text x="30" y="14" fill="#5E5191">image enc.</text><text x="30" y="26" fill="#5E5191">🐶 photo</text></g>
<g transform="translate(8,86)"><rect x="0" y="0" width="60" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="30" y="14" fill="#276b45">text enc.</text><text x="30" y="26" fill="#276b45">"a dog"</text></g>
<text x="74" y="82" fill="#B8AEA2">→</text>
<g transform="translate(88,50)"><rect x="0" y="0" width="70" height="60" fill="#FDF9EF" stroke="#C99A12"/><text x="35" y="20" fill="#8A6D3B">normalize +</text><text x="35" y="32" fill="#8A6D3B">shared space</text><text x="35" y="48" fill="#6B645E" font-size="6">unit sphere</text></g>
<text x="164" y="82" fill="#B8AEA2">→</text>
<g transform="translate(178,50)"><rect x="0" y="0" width="78" height="60" fill="#EAF0FA" stroke="#5E5191"/><text x="39" y="18" fill="#5E5191">similarity</text><text x="39" y="30" fill="#5E5191">matrix</text><text x="39" y="46" fill="#6B645E" font-size="6">diagonal = true</text></g>
<text x="262" y="82" fill="#B8AEA2">→</text>
<g transform="translate(276,44)"><rect x="0" y="0" width="94" height="72" fill="#FDECEC" stroke="#C93B3B"/><text x="47" y="18" fill="#8a3b3b">contrastive loss</text><text x="47" y="32" fill="#276b45" font-size="6.5">pull diagonal ↑</text><text x="47" y="44" fill="#8a3b3b" font-size="6.5">push off-diag ↓</text><text x="47" y="60" fill="#6B645E" font-size="6">× temperature</text></g>
<text x="376" y="82" fill="#B8AEA2">→</text>
<g transform="translate(390,50)"><rect x="0" y="0" width="120" height="60" fill="#EAF5EE" stroke="#2D8B55"/><text x="60" y="20" fill="#276b45">trained shared space</text><text x="60" y="36" fill="#276b45">→ zero-shot naming</text><text x="60" y="52" fill="#6B645E" font-size="6">write labels as sentences</text></g>
<text x="260" y="150" fill="#6B645E" font-size="8">large batch + web-scale data keep the signal strong and the noise low</text><text x="260" y="166" fill="#8A6D3B" font-size="8">each step: two readers, one space, pull true pairs together, push mismatches apart</text></g></svg>
%%%

**What the recipe-card picture gets right:** nothing new is created — you gather the steps you already did into one ordered flow. **Where it breaks down:** recipe steps just sit in a list, but these steps *connect* into a working system — the encoders feed the space, the space feeds the loss, the trained space powers zero-shot — so this "card" is really a machine.

The day in a few beats:
- **Why not just labels.** A fixed label list is a printed menu — it can't name things that aren't on it. So CLIP learns from free **image-text pairs** on the web (**natural-language supervision**).
- **Two readers.** An **image encoder** and a **text encoder** turn a photo and a caption each into an **embedding** — a point in one **shared space**.
- **Same-length arrows.** **L2-normalize** every embedding onto the **unit sphere**, then score a match by direction: **cosine similarity**, from −1 to +1.
- **The match grid.** Score every photo against every caption → a **similarity matrix**. The **diagonal** = **positive pairs** (true); everything off it = **negative pairs**. The other captions in the batch are free **in-batch negatives**.
- **Pull & push.** The **contrastive loss** pulls each true pair together and pushes mismatches apart — **symmetric** (both directions). Pulling *only* causes **representation collapse**; the push is the cure.
- **Three knobs.** **Temperature** stretches the cramped scores; a **large batch** supplies more negatives; **web-scale data** washes out noisy captions and **false negatives**.
- **The party trick.** **Zero-shot classification**: write class names as sentences and pick the closest — no retraining. Limits: no generation, weak at counting/layout/OCR, prompt-sensitive, and it inherits web bias.

#### Cheat-sheet · the CLIP recipe at a glance
%%% table
:: Step :: What it does :: Key word
Read both sides :: photo & caption → number-lists :: image encoder / text encoder
Land in one space :: same length, compare by direction :: L2 normalize / cosine similarity
Score the batch :: every photo × every caption :: similarity matrix (diagonal = true)
Train :: pull true pairs, push mismatches :: contrastive loss (in-batch negatives)
Make it work :: sharpen scores · more negatives · clean via scale :: temperature / batch size / web-scale
Use it :: name new classes from descriptions :: zero-shot classification
%%%

#### Cheat-sheet · the words you met today
%%% jargon
image-text pair | a photo together with a caption that describes it — CLIP's free training data
natural-language supervision | using free-form captions as labels instead of a fixed category list
image encoder / text encoder | the two readers that turn a photo / a caption into a list of numbers
embedding | the short list of numbers standing for an input — a point in space
shared embedding space | one common map both encoders point into, so photo & caption are comparable
L2 normalization | rescaling an arrow to length 1 so only its direction matters (unit sphere)
cosine similarity | a match score from the angle between two arrows: +1 same way, 0 unrelated, −1 opposite
similarity matrix | the grid of every-photo-vs-every-caption scores; the diagonal is the true pairs
positive / negative pair | a true photo-caption match (diagonal) vs a mismatch (off-diagonal)
in-batch negatives | using the other captions in the batch as the wrong answers, for free
contrastive loss | pull true pairs together, push mismatches apart (symmetric, both directions)
representation collapse | the failure where everything maps to one point (from pulling with no push)
temperature | a learned factor that stretches the cramped scores so the right answer clearly wins
batch size | how many pairs per step — a bigger batch gives more negatives, a stronger push
web-scale data | an enormous pile of pairs, so noisy captions and false negatives average out
zero-shot classification | naming a photo among classes written as sentences, with no retraining
%%%

That's the whole day — and the whole module. You can now explain why a fixed label list falls short, how two encoders share one space, how normalization plus cosine similarity scores a match, how the similarity matrix sets up positives and negatives, how the contrastive loss pulls and pushes (and why the push prevents collapse), what temperature, batch size, and data scale each fix, and how zero-shot classification names things CLIP never directly trained on. Next: the **Module 6 Review Gate**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: How does CLIP's shared embedding space let it compare a photo with a caption? | a:2 | it converts the caption into pixels and overlays it on the photo | it counts matching colors in both | an image encoder and a text encoder each turn their input into a list of numbers in ONE shared space, so a photo and its caption can land near each other and be scored | it uses a fixed list of 1000 labels | fb: CLIP has two encoders — one for images, one for text — that both map into a single shared space. A photo and its caption become points there, so you can measure how close they are. That's the whole "two readers, one space" idea.
q: In CLIP's batch similarity matrix, which cells are the TRUE (positive) pairs, and what does the contrastive loss do? | a:1 | the off-diagonal cells; the loss makes them equal | the DIAGONAL cells (each photo with its own caption); the loss pulls those together and pushes every off-diagonal mismatch apart | the first row only; the loss ignores the rest | the brightest cell anywhere; the loss deletes the others | fb: The diagonal holds each photo paired with its own real caption — the positives. The contrastive loss pulls the diagonal together (attract) and pushes all off-diagonal mismatches apart (repel), both directions at once.
q: Why does CLIP use a learnable temperature to scale the scores? | a:2 | to make training slower on purpose | to add more class labels | raw cosine similarities are cramped in [−1, 1], so true and wrong pairs look nearly equal; temperature stretches the scores so the right answer clearly beats the rest | to turn images into text | fb: Cosine similarity is trapped between −1 and +1, a tiny range, so the pull-push signal is faint. A learnable temperature multiplies the scores to stretch the gaps, so the correct pair stands out and the model actually learns.
q: What is zero-shot classification with CLIP, and one thing CLIP CANNOT do? | a:1 | write each class name as a sentence, embed them, and pick the caption closest to the photo — no retraining on those classes; and CLIP cannot GENERATE images or captions, only match/score them | retrain the whole model on new labels each time; CLIP can generate any image | it memorizes every possible photo; CLIP can count objects perfectly | it only works on the original 1000 labels; CLIP can read any text in images flawlessly | fb: Zero-shot means you write the candidate labels as sentences, embed them, and pick the closest to the photo — no retraining. CLIP only scores/matches; it can't generate images or captions, and it's weak at counting, layout, and OCR.
%%%

@@@ produce id=produce tag="Produce" title="Build a tiny CLIP matchmaker with your own hands" gotit="Done"
Time to be the matchmaker yourself and *watch* the contrastive idea work. You'll make up a tiny batch of fake image and text embeddings, normalize them, build the similarity matrix, and check that the true pairs (the diagonal) win — then run a mini zero-shot classification. **Predict first:** after you normalize two arrows that point the *same* way, will their cosine similarity be closer to 0 or to 1? And in the similarity matrix, which cells should have the highest scores — the diagonal or the off-diagonal ones? Then run it and **observe** that normalization leaves only direction, that the diagonal (true pairs) scores highest in each row, and that scaling by a temperature makes the winner stand out more sharply. Pick one path.

#### Option A · write it yourself
Create `sessions/m06-cnns-vision-encoders/day-08-clip-contrastive/experiment.py` and **print values at every step**. (1) **Fake embeddings:** with NumPy, make `img = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])` (3 image vectors) and `txt = np.array([[0.9,0.1,0.],[0.05,0.95,0.1],[0.,0.1,0.9]])` (3 caption vectors, each roughly matching the image on its own row); **print** their shapes. (2) **L2 normalize:** write `normalize(x)` that divides each row by its length (`np.linalg.norm(x, axis=1, keepdims=True)`); normalize both and **print** that every row now has length ≈ 1. (3) **Similarity matrix:** compute `S = img_n @ txt_n.T` (cosine similarities, since rows are unit length); **print** the 3×3 matrix and **notice** the diagonal values are the biggest in each row — the true pairs win. (4) **Temperature:** multiply `S` by a temperature like `20`, apply a row-wise softmax, and **print** the result; **observe** the diagonal probabilities get much closer to 1 (the winner sharpens). (5) **Zero-shot:** take image row 0, compute its similarity to all 3 captions, and **print** `argmax` — check it picks caption 0. Print a one-line takeaway. Run with `python3 sessions/m06-cnns-vision-encoders/day-08-clip-contrastive/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 6 Day 8 artifact.

Create sessions/m06-cnns-vision-encoders/day-08-clip-contrastive/experiment.py that demonstrates the core of CLIP — a shared embedding space, cosine similarity, the batch similarity matrix, a temperature, and zero-shot classification — using plain NumPy (no real model), with a comment on each step and printing values at every step:
1. Make fake embeddings: img = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]) (3 image vectors) and txt = np.array([[0.9,0.1,0.],[0.05,0.95,0.1],[0.,0.1,0.9]]) (3 caption vectors, each roughly matching the image on its own row). Print both shapes.
2. Write normalize(x) that divides each row by its L2 norm (np.linalg.norm(x, axis=1, keepdims=True)). Normalize img and txt; print that each row now has length ~1.
3. Similarity matrix: S = img_n @ txt_n.T (this is cosine similarity because rows are unit length). Print the 3x3 matrix and point out the diagonal entries are the largest in each row = the true (positive) pairs.
4. Temperature: scale logits = S * 20, apply a row-wise softmax, print the probabilities, and explain that temperature sharpens the cramped scores so the correct pair clearly wins.
5. Zero-shot: for image row 0, compute cosine similarity to all 3 captions and print argmax; assert it equals 0 (picks its true caption). Explain that you can swap in any new caption sentences at test time with no retraining.
6. Print a one-line takeaway: "CLIP puts images and captions in one shared space; cosine similarity scores matches; the contrastive loss pulls the diagonal (true pairs) together and pushes mismatches apart; temperature sharpens the scores; zero-shot names new classes written as sentences."
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (predict, then confirm)
- After normalizing, every row has length ≈ 1, and two arrows pointing the same way score cosine ≈ 1 — normalization leaves only direction.
- In the similarity matrix, the **diagonal** cells (each image with its own caption) are the biggest in their row — the true pairs win, exactly what the contrastive loss is trained to make happen.
- Scaling by a temperature and taking a softmax makes the diagonal probabilities jump much closer to 1 — the winner sharpens, which is why the cramped scores needed stretching.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m06-cnns-vision-encoders/day-08-clip-contrastive/log.md`: (1) what the shared embedding space is and how cosine similarity scores a match; (2) what the contrastive loss does (pull true pairs together, push mismatches apart) and why pulling alone would collapse the space; (3) how zero-shot classification names a photo without retraining, plus one thing CLIP cannot do.

@@@ fin
