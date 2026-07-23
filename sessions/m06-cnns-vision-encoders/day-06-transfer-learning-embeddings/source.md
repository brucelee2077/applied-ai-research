---
quest_id: wf8-d06-transfer
mode: concept
donor: v9-base.donor
page_title: "Module 6 · Day 6 — Transfer Learning & Embeddings"
module_label: "Module 06 · Represent · Day 6"
title: "Transfer Learning and Embeddings"
subtitle: "Borrowing a Trained Network's Eyes"
brand_sub: "Foundations · M6 Day 6"
spine: "hand-me-down"
nav_prev_href: "../day-05-data-augmentation/lesson.html"
nav_prev_label: "Data Augmentation"
nav_next_href: "../day-07-object-detection-intuition/lesson.html"
nav_next_label: "Object Detection (Intuition)"
fin_title: "Module 6 · Day 6 complete! 🏆"
fin_body: "You've unlocked <b>Transfer Learning</b>: instead of training from zero, you take a network that already learned to see and pass it down — a <b>hand-me-down</b> — reusing its early layers as an eye and training just a small new head. Its feature vectors, called embeddings, become a reusable summary of any image. This is how good vision models get built with little data.<br>Next up: <b>Object Detection (Intuition)</b>."
notebook_yardstick: null
---

@@@ hero
@lede You know how a little brother or sister gets **hand-me-down** clothes? Someone older wore that jacket first, broke it in, and now it's passed down — you didn't have to buy a new one, you just get to *use* it. Here's a lovely idea: neural networks can do the same thing. Someone already spent months and a mountain of pictures training a network to *see* — to spot edges, colors, fur, wheels, faces. Instead of starting your own network from a blank slate, you take that trained network as a **hand-me-down**: you borrow its trained "eyes" and only add a small new piece of your own. This trick is called **transfer learning**, and it's the reason a small team with just a few hundred photos can build a vision model that actually works — the same core idea behind how models learn to recognize your face, sort medical scans, or power a "search by image" button. Today you learn to borrow a trained network's eyes.
@goal Together we'll build the hand-me-down idea one piece at a time. You'll see why starting from zero is so hard, what a pretrained "eye" really is, the beautiful reason early layers are reusable while late layers are not, how to swap in your own small head, the two ways to reuse (freeze vs. fine-tune), what a network's inner feeling about an image — its embedding — looks like, how those embeddings let you search images with no training at all, then the friendly gotchas each with its fix, and the honest limit of borrowed eyes. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="Starting from zero" title="Why training a vision model from scratch is so hard" gotit="Got the problem"
Let's start with the problem the hand-me-down trick is here to fix — because once you *feel* how painful starting from zero is, borrowing will feel like a gift.

Imagine a **brand-new baby learning to see.** At the very start, a baby's eyes take in light but the brain hasn't yet learned what any of it *means* — edges, faces, "that's Mom," "that's a cup." It takes months and *thousands and thousands* of glances before the baby reliably knows things by sight. A fresh neural network is in exactly that newborn state: its numbers start random, so it sees only noise. To teach it to recognize even a few kinds of things from scratch, you'd need a giant pile of labeled photos and a lot of time and computer power. Most people simply don't have that. This is the wall you hit: a [[neural network||A stack of math layers that turns an input, like an image, into an output, like a label. It starts with random numbers and must be trained on examples.]] that starts from random numbers — from scratch — needs *far* more data than you probably have.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A newborn-baby analogy for training from scratch. On the left, labelled from scratch, a network with random numbers needs a huge pile of labeled photos and lots of time and compute, and a small team only has a few hundred photos, so it overfits. On the right, labelled the wall, a note says a fresh network sees only noise and needs far more data than you have. A caption says starting from zero is expensive and usually fails on small data.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">From scratch = a newborn eye: it knows nothing until it sees a mountain of photos</text>
<g transform="translate(30,34)"><rect x="0" y="0" width="215" height="132" fill="#FDECEC" stroke="#C93B3B"/><text x="107" y="20" fill="#8a3b3b">start from zero (random numbers)</text><rect x="35" y="36" width="140" height="22" fill="#FDF2E0" stroke="#C99A12"/><text x="105" y="51" fill="#8a3b3b" font-size="8">needs: MILLIONS of labeled photos</text><rect x="35" y="66" width="140" height="22" fill="#FDF2E0" stroke="#C99A12"/><text x="105" y="81" fill="#8a3b3b" font-size="8">needs: lots of time + compute</text><text x="107" y="106" fill="#8a3b3b" font-size="8">you have: a few hundred photos</text><text x="107" y="122" fill="#8a3b3b">→ it overfits and fails ✗</text></g>
<text x="255" y="100" fill="#B8AEA2" font-size="13">→</text>
<g transform="translate(285,34)"><rect x="0" y="0" width="205" height="132" fill="#FDF9EF" stroke="#C99A12"/><text x="102" y="20" fill="#8A6D3B">the wall</text><circle cx="102" cy="66" r="30" fill="#F0EBE3" stroke="#B8AEA2"/><text x="102" y="62" fill="#6B645E" font-size="7.5">sees only</text><text x="102" y="74" fill="#6B645E" font-size="7.5">noise</text><text x="102" y="118" fill="#8a3b3b" font-size="8">a fresh eye knows nothing yet</text></g></g></svg>
%%%

**What the newborn picture gets right:** a fresh network, like a newborn, must learn *everything* about seeing from scratch, and that takes an enormous number of examples. **Where it breaks down:** a baby learns on its own from the world with no labels, while a from-scratch network needs *labeled* photos (someone must write "cat," "dog") — and it has no built-in instinct to help it start.

#### Why "too little data" is the real trap
When a big network meets a small pile of photos, it doesn't learn the *idea* of a cat — it just memorizes your exact photos and then fails on new ones. That trap is called [[overfitting||When a model learns the training photos so exactly that it fails on new, unseen photos — like a student who memorized the practice test but can't answer the real one.]] — you met it in the last lesson. Starting from scratch makes overfitting *much* worse, because there's so much to learn and so few examples to learn it from.

So the question becomes: what if the *hard part* — learning to see edges, textures, shapes — were already done for us, by someone with a mountain of data? Then we'd only have to learn our small, specific part. That "someone else already did the hard part" is exactly the hand-me-down we're about to unwrap.

@@@ concept id=c2 tag="The trained eye" title="A pretrained network is a hand-me-down eye you can borrow" gotit="Got the trained eye"
Here's the gift. Somewhere out there, big teams have already trained networks on *enormous* photo collections — millions of pictures of thousands of everyday things. The most famous such collection is called [[ImageNet||A very large public collection of about a million labeled photos across a thousand everyday categories. Many networks are trained on it first, then handed down for reuse.]]. A network trained on it has, over that mountain of pictures, learned to *see* — and we can borrow that trained eye instead of growing our own from a newborn.

Think of a **hand-me-down winter coat.** An older cousin wore it for years — broke it in, softened it, shaped it to real weather. When it's passed down to you, you don't start with a bolt of cloth and a sewing machine; you get a finished, road-tested coat for free, and maybe you just swap the buttons to make it yours. A [[pretrained||"Pre" means "before." A pretrained network was already trained by someone else, on a big dataset, before you ever touched it. You reuse its learned numbers.]] network is that coat: someone already did the long, expensive work of training it, and you inherit the finished result. The part you borrow — the stack of layers that turns a photo into useful features — is called the [[backbone||The main stack of trained layers you borrow from a pretrained network — the part that turns a raw photo into useful features. Also called the feature extractor.]], and when you use it just to *read* an image into features, it's playing the role of a [[feature extractor||A backbone used purely to turn an image into a list of useful numbers (features), without changing its own trained weights.]].

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A hand-me-down-coat analogy for a pretrained backbone. On the left, someone trains a network on a mountain of ImageNet photos, producing a finished trained backbone. An arrow labelled hand it down passes that same backbone to you on the right, where you reuse it as-is on your own few photos. A caption says you inherit the finished trained eye instead of sewing your own from scratch.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A pretrained backbone = a hand-me-down: someone trained it, you inherit it</text>
<g transform="translate(24,36)"><rect x="0" y="0" width="200" height="130" fill="#FDF9EF" stroke="#C99A12"/><text x="100" y="18" fill="#8A6D3B">big team, long ago</text><text x="100" y="38" fill="#8A6D3B" font-size="8">trained on ~1,000,000 photos</text><g fill="#C99A12"><circle cx="40" cy="58" r="4"/><circle cx="58" cy="58" r="4"/><circle cx="76" cy="58" r="4"/><circle cx="94" cy="58" r="4"/><circle cx="112" cy="58" r="4"/><circle cx="130" cy="58" r="4"/><circle cx="148" cy="58" r="4"/></g><rect x="35" y="76" width="130" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="95" fill="#276b45" font-size="8">finished, trained backbone</text><text x="100" y="122" fill="#8A6D3B" font-size="8">the "eye" is fully broken in</text></g>
<g transform="translate(228,90)"><text x="30" y="-6" fill="#5E5191" font-size="8">hand it down</text><text x="30" y="14" fill="#B8AEA2" font-size="16">→</text></g>
<g transform="translate(300,36)"><rect x="0" y="0" width="196" height="130" fill="#EAF5EE" stroke="#2D8B55"/><text x="98" y="18" fill="#276b45">you, today</text><text x="98" y="38" fill="#276b45" font-size="8">only a few hundred photos</text><rect x="33" y="52" width="130" height="30" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="98" y="71" fill="#276b45" font-size="8">SAME trained backbone, reused</text><text x="98" y="102" fill="#276b45" font-size="8">you run YOUR photo through it</text><text x="98" y="118" fill="#276b45" font-size="8">and keep what it "sees" ✓</text></g></g></svg>
%%%

**What the hand-me-down-coat picture gets right:** you inherit something already built and road-tested, saving all the hard work, and you only make small changes to fit you. **Where it breaks down:** a coat wears out with use, but a pretrained network's numbers don't degrade from being reused — copies are perfect, and you can hand the same backbone to a million people at once.

#### How you actually "read" an image with the borrowed eye
Using the backbone as a feature extractor is simple: you push your photo *in* one end and let it flow through the trained layers, and you keep what comes out. Those outputs — the numbers each layer produces as the image passes through — are called [[activations||The numbers a layer produces as an input flows through it. Early-layer activations describe simple things like edges; late-layer activations describe whole objects.]]. Deeper into the network, the activations stop describing raw pixels and start describing *meaning* — "there's fur here," "this looks like an ear." Watch one image flow through and see the shapes shrink while the meaning grows:

%%% demo id=extract label="run a photo through the borrowed backbone"
code: features = backbone(cat_photo)     # no training — just read the image
out: input photo:      shape (3, 224, 224)   # 3 colors, 224x224 pixels
     early activations: shape (64, 112, 112)  # many simple edge/color maps
     late activations:  shape (512, 7, 7)      # few, rich "meaning" maps
     kept features:      a list of 512 numbers  # a compact summary of the image
take: <b>The photo went in as raw pixels and came out as 512 meaning-numbers — with no training at all.</b> Notice the picture shrinks (224→7 across) while the channels grow (3→512): the network trades "where every pixel is" for "what's in the picture." That compact summary is the treasure we borrow.
%%%

**You now hold the core gift:** a trained eye you can borrow for free. But *why* does an eye trained on someone else's photos work on *yours*? The answer is the most beautiful idea in the whole lesson — and it's next.

@@@ concept id=c3 tag="Early vs. late" title="The magic: early layers are generic, late layers are specific" gotit="Got the layers"
Here's the reason the hand-me-down works at all — and it's genuinely lovely once you see it. A trained network doesn't learn everything in a jumble. It learns in *stages*, from simple to complicated, as the image flows deeper. And the early stages turn out to be useful for *almost any* photo task, not just the one the network was trained on.

Think of **learning to cook.** The very first skills you pick up are general: hold a knife safely, chop an onion, boil water, tell when oil is hot. Those basics are useful in *every* kitchen — Italian, Indian, dessert, anything. Only *later* do you learn the specific stuff: exactly how *this one restaurant* plates its signature dish. If you switch restaurants, you keep all the basic knife-and-heat skills and just relearn the final signature dish. A trained network is the same: its **early layers** learn general basics — edges, corners, colors, simple textures — that help with *any* image, while its **late layers** combine those basics into the *specific* things it was trained to name (this exact breed of dog, that exact model of car).

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A learning-to-cook analogy for generic early layers and specific late layers. A network is drawn as a left-to-right stack. Early layers on the left are labelled generic basics with tiny pictures of edges, corners, and colors, and marked reuse me. Late layers on the right are labelled specific to ImageNet with pictures of whole objects like a dog and a car, and marked replace me. A caption says early basics transfer to any task, late specifics do not.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Early layers = general knife-and-heat basics · Late layers = this restaurant's dish</text>
<rect x="24" y="40" width="472" height="120" fill="#FCFBF8" stroke="#E4DED4"/>
<g transform="translate(36,54)"><rect x="0" y="0" width="140" height="92" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="70" y="16" fill="#276b45" font-size="8">EARLY layers = generic</text><g stroke="#276b45" fill="none"><path d="M18 34 l22 0"/><path d="M52 44 l10 -18"/><path d="M78 26 l14 14 l-14 14"/></g><circle cx="30" cy="58" r="6" fill="#C93B3B"/><circle cx="50" cy="58" r="6" fill="#2D8B55"/><circle cx="70" cy="58" r="6" fill="#5E5191"/><text x="70" y="82" fill="#276b45" font-size="7.5">edges · corners · colors</text></g>
<g transform="translate(190,86)"><text x="20" y="0" fill="#B8AEA2" font-size="16">→</text><text x="20" y="18" fill="#6B645E" font-size="7.5">simple → rich</text></g>
<g transform="translate(240,54)"><rect x="0" y="0" width="120" height="92" fill="#FDF2E0" stroke="#C99A12" stroke-width="1.5"/><text x="60" y="16" fill="#8A6D3B" font-size="8">MIDDLE = textures/parts</text><text x="30" y="46" font-size="18">🐾</text><text x="72" y="46" font-size="18">👁</text><text x="60" y="82" fill="#8A6D3B" font-size="7.5">fur · ears · wheels</text></g>
<g transform="translate(374,54)"><rect x="0" y="0" width="110" height="92" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="55" y="16" fill="#8a3b3b" font-size="8">LATE = specific</text><text x="30" y="50" font-size="20">🐕</text><text x="72" y="50" font-size="20">🚗</text><text x="55" y="82" fill="#8a3b3b" font-size="7.5">exact ImageNet class</text></g>
<text x="130" y="184" fill="#276b45" font-size="9">✓ REUSE these — they help any photo task</text>
<text x="405" y="184" fill="#8a3b3b" font-size="9">✗ REPLACE this — it's tied to old classes</text></g></svg>
%%%

**What the learning-to-cook picture gets right:** basic skills carry over to every kitchen while the signature-dish knowledge is tied to one place — exactly why early layers transfer and late layers don't. **Where it breaks down:** a cook *knows* which skills are basic; a network doesn't label its own layers "generic" or "specific" — that pattern is something researchers *discovered* by looking at what each layer responds to.

#### Why this is the whole reason transfer works
Put the two facts together and the plan writes itself:

- **Early layers are generic.** Edges and textures look the same whether the photo is a cat, a chest X-ray, or a satellite view. So the borrowed early layers are useful for *your* task, even though they were trained on someone else's.
- **Late layers are specific.** The last layers were tuned to name ImageNet's exact 1,000 categories. Those are probably *not* your categories. So the late, specific part is the part you'll *replace* with your own.

!!! c-info 💡
<b>Optional (skippable) — the famous experiment:</b> researchers measured this directly. They took each layer of a trained network, froze everything up to it, and re-trained the rest on a new task. The finding: reusing the <i>early</i> layers barely hurt accuracy (they're general), but reusing the <i>late</i> layers hurt a lot (they're specialized to the original task). This "generic-early, specific-late" pattern is why we keep the front of the network and swap the back.
!!!

Let's watch, layer by layer, how well a borrowed layer transfers to a new task:

%%% demo id=transferability label="how well does each borrowed layer transfer?"
code: for layer in backbone.layers: measure_transfer(layer, new_task)
out: layer 1 (edges):     transfers great  ✓✓✓   (generic)
     layer 2 (textures):  transfers great  ✓✓✓   (generic)
     layer 3 (parts):     transfers well   ✓✓    (mostly generic)
     layer 4 (objects):   transfers so-so  ✓      (getting specific)
     final head (classes): transfers poorly ✗    (tied to old classes)
take: <b>The earlier the layer, the better it transfers.</b> The front of the network is a general-purpose eye you can keep; the very back is tied to the old task and should be swapped for your own. That single fact is the engine of the whole hand-me-down trick.
%%%

**You now understand the deep reason it works:** keep the generic front, replace the specific back. So let's actually replace that back — starting with the small new piece you add yourself.

@@@ concept id=c4 tag="Swap the head" title="Snap off the old ending, snap on your own small head" gotit="Got the head swap"
We keep the borrowed backbone (the generic eye) and throw away only the very last piece — the part that named the *old* categories. Then we snap on a small new piece sized to *your* categories. That last piece has a name: the [[head||The small final piece of a network that turns the backbone's features into your own answers. You replace the pretrained head with a new one sized to your classes.]] (also called the classifier head, because it does the final classifying).

Think of a **cordless drill with swappable bits.** The heavy part — the motor, the grip, the battery — is expensive and does the real work; you keep it. The little bit on the end is what actually meets the job: a drill bit for holes, a screwdriver bit for screws. To do a different job you don't buy a new drill — you *pop off* one bit and *snap on* another. The backbone is the drill motor (borrowed, kept); the head is the bit (cheap, swapped). ImageNet's head was a "1,000-category bit"; if your job is telling cats from dogs, you snap on a "2-category bit."

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="A swappable-drill-bit analogy for replacing the classifier head. A network is drawn as a borrowed backbone box followed by a detachable head. The old head labelled 1000 ImageNet classes is shown popping off with an X, and a new small head labelled your 2 classes cat dog snaps on in its place. A caption says keep the borrowed motor, swap only the tiny bit on the end.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Keep the drill (backbone); pop off the old bit and snap on YOUR head</text>
<g transform="translate(30,44)"><rect x="0" y="0" width="200" height="110" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="100" y="20" fill="#276b45">borrowed backbone (KEEP)</text><text x="100" y="38" fill="#276b45" font-size="8">the trained "eye" — the motor</text><g fill="#2D8B55"><rect x="30" y="52" width="18" height="30"/><rect x="58" y="52" width="18" height="30"/><rect x="86" y="52" width="18" height="30"/><rect x="114" y="52" width="18" height="30"/><rect x="142" y="52" width="18" height="30"/></g><text x="100" y="98" fill="#276b45" font-size="7.5">outputs a list of features</text></g>
<text x="238" y="102" fill="#B8AEA2" font-size="14">→</text>
<g transform="translate(262,36)"><rect x="0" y="0" width="110" height="58" fill="#FDECEC" stroke="#C93B3B" stroke-dasharray="4,3"/><text x="55" y="20" fill="#8a3b3b" font-size="8">OLD head (toss)</text><text x="55" y="38" fill="#8a3b3b" font-size="8">1000 ImageNet</text><text x="55" y="50" fill="#8a3b3b" font-size="8">classes ✗</text><text x="55" y="72" fill="#8a3b3b" font-size="18">✕ pop off</text></g>
<g transform="translate(262,116)"><rect x="0" y="0" width="110" height="58" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="55" y="20" fill="#5E5191" font-size="8">NEW head (yours)</text><text x="55" y="38" fill="#5E5191" font-size="8">2 classes:</text><text x="55" y="50" fill="#5E5191" font-size="8">cat · dog ✓</text><text x="55" y="72" fill="#5E5191" font-size="9">✓ snap on</text></g>
<text x="440" y="100" fill="#6B645E" font-size="8">tiny, quick</text><text x="440" y="114" fill="#6B645E" font-size="8">to train</text></g></svg>
%%%

**What the swappable-drill picture gets right:** the expensive motor stays and only the cheap end-piece changes to match the new job — exactly how backbone-plus-new-head works. **Where it breaks down:** a drill bit is a physical thing you buy ready-made, but your new head starts with *random* numbers and still has to be *trained* on your photos — it's the one small part you do teach.

#### Why the new head is small and cheap to train
The backbone already turned each photo into a short list of meaning-numbers (remember the 512 features). Your new head only has to learn a simple rule: *given these 512 numbers, which of my classes is it?* That's a tiny job compared to learning to see from scratch. So:

- The head is usually just **one small layer** that maps the backbone's features to your number of classes.
- Because it's small, it trains **fast** and needs **far fewer photos** than a whole network would.

%%% formula
expr: your_answer = new_head( backbone(image) )
note: run the image through the borrowed backbone to get features, then the tiny new head turns those features into YOUR classes. Only the new head starts out untrained.
%%%

**You've now swapped the head.** But there's a choice to make about the borrowed backbone: do you leave it exactly as inherited, or keep teaching it a little? That's the freeze-versus-fine-tune fork, and it's next.

@@@ concept id=c5 tag="Freeze or fine-tune" title="Two ways to reuse: freeze the eye, or keep teaching it a little" gotit="Got the two modes"
There are two ways to reuse the borrowed backbone, and picking between them is the main decision in transfer learning. Both keep the borrowed eye; they differ in whether you let it *keep learning* on your photos.

Think of the **hand-me-down coat again**, on a cold day. Option one: wear it exactly as it came — don't touch it. Option two: take it to a tailor for a *gentle* nip-and-tuck so it fits *you* a little better. Both start from the finished coat; the second just makes small, careful adjustments. In transfer learning, "wear it as-is" means you [[freeze||To freeze a layer is to lock its numbers so training leaves them unchanged — no updates at all. A frozen backbone is used exactly as inherited.]] the backbone — lock its numbers so they never change — and train only your new head. "Take it to the tailor" means you [[fine-tune||To fine-tune is to unlock some pretrained layers and keep training them gently on your own data, nudging the inherited numbers to fit your task better.]] — unlock some of the borrowed layers and let them keep learning, gently, on your data.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A wear-as-is versus tailor analogy for freeze versus fine-tune. On the left, feature extraction mode: the backbone is drawn with lock icons meaning frozen, and only the small new head is marked training. On the right, fine-tuning mode: the last few backbone layers are unlocked and marked training gently along with the head, while the earliest layers stay locked. A caption says freeze equals wear as-is and train only the head, fine-tune equals gently tailor the last layers too.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Two ways to reuse the coat: wear it as-is (freeze) or tailor it a little (fine-tune)</text>
<g transform="translate(28,34)"><rect x="0" y="0" width="216" height="140" fill="#EAF0FA" stroke="#5E5191"/><text x="108" y="18" fill="#5E5191">① feature extraction (FREEZE)</text><text x="108" y="34" fill="#5E5191" font-size="8">wear the coat exactly as-is</text><g><rect x="24" y="46" width="26" height="40" fill="#Dfe4f2" stroke="#5E5191"/><rect x="56" y="46" width="26" height="40" fill="#Dfe4f2" stroke="#5E5191"/><rect x="88" y="46" width="26" height="40" fill="#Dfe4f2" stroke="#5E5191"/><rect x="120" y="46" width="26" height="40" fill="#Dfe4f2" stroke="#5E5191"/></g><text x="85" y="102" fill="#5E5191" font-size="12">🔒 all frozen</text><rect x="152" y="46" width="42" height="40" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="173" y="70" fill="#276b45" font-size="7.5">head</text><text x="173" y="102" fill="#276b45" font-size="8">🔥 trains</text><text x="108" y="128" fill="#5E5191" font-size="7.5">fewest params to learn → safest on small data</text></g>
<g transform="translate(276,34)"><rect x="0" y="0" width="216" height="140" fill="#FDF2E0" stroke="#C99A12"/><text x="108" y="18" fill="#8A6D3B">② fine-tuning</text><text x="108" y="34" fill="#8A6D3B" font-size="8">gently tailor the last layers</text><g><rect x="24" y="46" width="26" height="40" fill="#Dfe4f2" stroke="#5E5191"/><rect x="56" y="46" width="26" height="40" fill="#Dfe4f2" stroke="#5E5191"/><rect x="88" y="46" width="26" height="40" fill="#FCE9C6" stroke="#C99A12" stroke-width="1.5"/><rect x="120" y="46" width="26" height="40" fill="#FCE9C6" stroke="#C99A12" stroke-width="1.5"/></g><text x="55" y="102" fill="#5E5191" font-size="10">🔒 early</text><text x="120" y="102" fill="#8A6D3B" font-size="10">🔥 late</text><rect x="152" y="46" width="42" height="40" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="173" y="70" fill="#276b45" font-size="7.5">head</text><text x="108" y="128" fill="#8A6D3B" font-size="7.5">more params to learn → needs care + more data</text></g></svg>
%%%

**What the coat-and-tailor picture gets right:** both start from a finished, road-tested coat, and the only question is whether you make small adjustments — just like both modes start from the same trained backbone. **Where it breaks down:** a tailor's changes are always safe and reversible, but fine-tuning can *harm* the borrowed features if done carelessly (you'll meet that gotcha soon) — so it demands a gentle touch.

#### The two modes, side by side
%%% table
:: :: Feature extraction (freeze) :: Fine-tuning
What's frozen :: the whole backbone :: early layers frozen, later ones unlocked
What trains :: only the new head :: the head + some backbone layers
Params to learn :: very few :: many more
Best when :: little data; task close to ImageNet :: more data; task a bit different
Risk :: might underfit if task differs a lot :: can wreck good features if careless
%%%

#### Why freezing is the safe default on small data
Here's the key link back to concept 1. Remember overfitting — a big network with too few photos memorizes them. When you **freeze** the backbone, its millions of numbers are *locked*, so they can't be memorized-into. You're only training the tiny head, which has *few* numbers. Fewer trainable numbers versus your few photos means far less room to overfit. So freezing isn't just cheaper — it's the direct **remedy for overfitting on a small dataset.**

!!! c-info 💡
<b>Optional (skippable) — and augmentation stacks on top.</b> The last lesson's trick still helps here: <b>data augmentation</b> (flips, crops, recolor) gives your few photos more variety, which shrinks overfitting even further. Freeze the backbone <i>and</i> augment your small dataset — the two remedies work together.
!!!

Watch how the two modes trade off, on a *small* target dataset:

%%% demo id=modes label="freeze vs. fine-tune on a small dataset"
code: freeze_and_train_head(small_data)   vs   fine_tune_carelessly(small_data)
out: FREEZE:   trainable params = 1,026        val accuracy = 0.91  ✓ (few knobs, safe)
     FINE-TUNE (big LR, no care): trainable params = 11,000,000   val accuracy = 0.63  ✗ overfit + wrecked features
take: <b>On a small dataset, freezing usually wins.</b> Freezing exposes only ~1,000 trainable numbers, so it can't overfit your few photos — while carelessly fine-tuning 11 million numbers on tiny data both overfits AND damages the borrowed eye. Start frozen; fine-tune only when you have the data and do it gently.
%%%

**You now know the two modes and when to reach for each.** So far we kept the backbone to *classify*. But the numbers it produces just before the head are a treasure all on their own — a compact "feeling" about the image. Meet the embedding.

@@@ concept id=c6 tag="The embedding" title="An image as a short list of numbers: the embedding" gotit="Got embeddings"
Remember those 512 meaning-numbers the backbone spits out *just before* the head? That short list is called the image's [[embedding||The short list of numbers a trained network produces to summarize a whole image — its features taken from just before the final head. Similar images get similar embeddings.]] — and it's one of the most useful ideas in all of modern AI. It's the network's compact summary of *what the whole image is about*, written as numbers.

Think of a **nutrition label** on a food package. The food itself is complicated — a whole pizza, a whole salad — but the label boils it down to a short row of numbers: calories, fat, sugar, protein. Two foods with *similar* labels are similar foods; a salad and a pizza have very different labels. An embedding is a nutrition label for an image: the backbone reads the whole rich picture and prints a short row of numbers that captures its "flavor." Two images that *look and mean* similar things get *similar* embeddings; two very different images get very different ones. The numbers come from the [[penultimate layer||"Penultimate" means second-to-last. It's the layer right before the final head — where the network holds its richest summary of the whole image, just before it commits to a class.]] — the layer right before the head, where the network holds its richest summary before committing to an answer.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A nutrition-label analogy for an image embedding. Two photos, a cat and a kitten, each pass through the backbone and become a short row of numbers, and their number rows look similar. A third photo, a car, becomes a very different row of numbers. A caption says an embedding is a short numeric summary, similar images get similar rows.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">An embedding = a "nutrition label" for an image (a short row of numbers)</text>
<g transform="translate(20,34)"><rect x="0" y="0" width="54" height="42" fill="#FDF9EF" stroke="#C99A12"/><text x="27" y="26" font-size="18">🐱</text></g>
<text x="86" y="58" fill="#B8AEA2">→</text><g transform="translate(104,40)"><rect x="0" y="0" width="30" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="15" y="19" fill="#276b45" font-size="7">eye</text></g>
<text x="140" y="58" fill="#B8AEA2">→</text><g transform="translate(158,42)"><rect x="0" y="0" width="150" height="26" fill="#EAF5EE" stroke="#2D8B55"/><text x="75" y="18" fill="#276b45" font-size="8">[0.8, 0.1, 0.7, 0.2, …]</text></g>
<g transform="translate(20,92)"><rect x="0" y="0" width="54" height="42" fill="#FDF9EF" stroke="#C99A12"/><text x="27" y="26" font-size="18">🐈</text></g>
<text x="86" y="116" fill="#B8AEA2">→</text><g transform="translate(104,98)"><rect x="0" y="0" width="30" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="15" y="19" fill="#276b45" font-size="7">eye</text></g>
<text x="140" y="116" fill="#B8AEA2">→</text><g transform="translate(158,100)"><rect x="0" y="0" width="150" height="26" fill="#EAF5EE" stroke="#2D8B55"/><text x="75" y="18" fill="#276b45" font-size="8">[0.7, 0.2, 0.8, 0.1, …]</text></g><text x="330" y="118" text-anchor="start" fill="#276b45" font-size="8">← almost the same row!</text>
<g transform="translate(20,150)"><rect x="0" y="0" width="54" height="42" fill="#FDF9EF" stroke="#C99A12"/><text x="27" y="26" font-size="18">🚗</text></g>
<text x="86" y="174" fill="#B8AEA2">→</text><g transform="translate(104,156)"><rect x="0" y="0" width="30" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="15" y="19" fill="#276b45" font-size="7">eye</text></g>
<text x="140" y="174" fill="#B8AEA2">→</text><g transform="translate(158,158)"><rect x="0" y="0" width="150" height="26" fill="#FDECEC" stroke="#C93B3B"/><text x="75" y="18" fill="#8a3b3b" font-size="8">[0.1, 0.9, 0.0, 0.8, …]</text></g><text x="330" y="174" text-anchor="start" fill="#8a3b3b" font-size="8">← a very different row</text></g></svg>
%%%

**What the nutrition-label picture gets right:** a short row of numbers can summarize something complicated well enough that "similar labels = similar things" holds — exactly how embeddings work. **Where it breaks down:** a nutrition label's numbers each mean one plain thing (grams of sugar), but an embedding's numbers have no tidy names — no single number means "cat"; the *whole pattern* together carries the meaning.

#### Why the embedding is so useful
The embedding is a *reusable* summary. Once you have it, you can do many things without ever touching the network again — feed it to your little head, compare it to other embeddings, store it in a database. It's a list of numbers, so it's tiny and fast to work with. One image → one short vector → endless downstream uses.

%%% demo id=embed label="turn three images into embeddings"
code: emb = [backbone_features(cat), backbone_features(kitten), backbone_features(car)]
out: cat    → [0.81, 0.09, 0.72, 0.20, ...]   (length 512)
     kitten → [0.74, 0.15, 0.79, 0.11, ...]   (length 512)   ← close to cat's numbers
     car    → [0.10, 0.88, 0.03, 0.77, ...]   (length 512)   ← far from both
take: <b>Each image collapsed into 512 numbers.</b> The cat and kitten rows look alike; the car's row looks nothing like them. The network never saw these exact photos and did no training — it just "read" them into comparable summaries.
%%%

**You now have the embedding — a short numeric summary of any image.** But we said similar images get *similar* embeddings. How do we *measure* "similar" between two lists of numbers? That's the next, and it unlocks something magical.

@@@ concept id=c7 tag="Search by distance" title="Measuring closeness — and searching images with zero training" gotit="Got embedding search"
Here's the payoff that makes people fall in love with embeddings. If similar images have similar number-lists, then to *find* similar images you just look for embeddings that sit *close together*. No training needed at all — you already have the borrowed eye. This is how a "search by image" button works.

Think of **pins on a map.** Every town is a dot on a map; towns that are near each other in real life sit close on the map, and far-apart towns sit far apart. To find the towns *nearest* to where you are, you just measure the distance on the map and pick the closest pins. Embeddings turn each image into a "pin" in a space of numbers: similar images land near each other, different images land far apart. To find images like a given one, you drop its pin and grab the [[nearest neighbors||The data points closest to a given one. For images, the nearest neighbors of a photo's embedding are the most visually similar photos.]] — the closest pins. This "grab the closest pins" search is called [[nearest-neighbor search||Finding the stored items whose embeddings are closest to a query's embedding — the core trick behind image search and retrieval.]], and it's the engine behind image [[retrieval||Fetching the most relevant stored items for a query — for images, returning the photos most similar to a query photo, ranked by embedding distance.]] (search-by-image).

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A pins-on-a-map analogy for nearest-neighbor image search in embedding space. A 2D space holds several image pins. Cat and kitten pins cluster close together on one side, dog pins nearby, and car and truck pins cluster far away on the other side. A query cat pin is dropped and arrows point to its two closest pins, the other cats. A caption says similar images sit close, so the nearest pins are the most similar images.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Embedding space = a map of images; the nearest pins are the most similar</text>
<rect x="30" y="28" width="460" height="150" fill="#FCFBF8" stroke="#E4DED4"/>
<g><ellipse cx="140" cy="80" rx="70" ry="42" fill="#EAF5EE" opacity="0.6"/><text x="140" y="46" fill="#276b45" font-size="8">cats &amp; kittens (close)</text><text x="118" y="82" font-size="15">🐱</text><text x="152" y="96" font-size="15">🐈</text><text x="140" y="112" font-size="15">🐱</text></g>
<g><ellipse cx="250" cy="130" rx="46" ry="30" fill="#FDF2E0" opacity="0.7"/><text x="250" y="122" fill="#8A6D3B" font-size="8">dogs (nearby)</text><text x="238" y="146" font-size="14">🐕</text></g>
<g><ellipse cx="400" cy="100" rx="66" ry="40" fill="#FDECEC" opacity="0.6"/><text x="400" y="66" fill="#8a3b3b" font-size="8">vehicles (far away)</text><text x="378" y="104" font-size="15">🚗</text><text x="416" y="116" font-size="15">🚚</text></g>
<g><text x="88" y="70" font-size="16">🔍🐱</text><text x="70" y="92" fill="#5E5191" font-size="7.5">query</text><line x1="102" y1="66" x2="118" y2="78" stroke="#5E5191" stroke-dasharray="3,2"/><line x1="102" y1="70" x2="140" y2="106" stroke="#5E5191" stroke-dasharray="3,2"/></g>
<text x="260" y="198" fill="#6B645E">drop the query pin, grab its nearest neighbors — no training required</text></g></svg>
%%%

**What the pins-on-a-map picture gets right:** turning things into positions lets "similar" become "close," so finding similar items is just finding nearby points — the whole idea of embedding search. **Where it breaks down:** a real map is 2 flat dimensions you can eyeball; an embedding space has *hundreds* of dimensions you can't draw, so the "distance" is computed by a formula, not read off with a ruler.

#### The two everyday rulers for "close"
To measure how close two embeddings are, you need a *ruler*. Two are standard, and you only need the plain idea of each:

- [[Euclidean distance||The straight-line distance between two points — for two number-lists, the length of the arrow from one to the other. Also written L2 distance. Small = close = similar.]] (also called **L2 distance**) — the plain straight-line distance between two pins, like measuring with a ruler on the map. **Small distance = similar.**
- [[cosine similarity||A measure of how much two number-lists point in the same direction, ignoring their length. 1 = same direction (very similar), 0 = unrelated. Popular for embeddings.]] — instead of the gap between pins, it asks whether the two number-lists *point the same way*. **Close to 1 = same direction = similar.**

!!! c-info 💡
<b>Optional (skippable) — the one-line formulas.</b> For two embeddings <code>a</code> and <code>b</code>: Euclidean distance is <code>‖a − b‖</code> (the length of the difference), and cosine similarity is <code>(a·b) / (‖a‖‖b‖)</code> (the dot product divided by both lengths). Cosine ignores how <i>long</i> the vectors are and looks only at their <i>direction</i>, which is why it's a favorite for comparing embeddings. You don't need the algebra today — just "small distance" or "cosine near 1" means "similar."
!!!

Watch the rulers agree that cat is near kitten and far from car. Predict which pair is closest, then run:

%%% demo id=distances label="measure closeness between three embeddings"
code: cosine(cat, kitten), cosine(cat, car);  L2(cat, kitten), L2(cat, car)
out: cosine(cat, kitten) = 0.94   ← near 1: very similar ✓
     cosine(cat, car)    = 0.11   ← near 0: unrelated ✗
     L2(cat, kitten)     = 0.32   ← small: close ✓
     L2(cat, car)        = 1.71   ← large: far apart ✗
take: <b>Both rulers agree: cat↔kitten are close, cat↔car are far.</b> High cosine (near 1) and small L2 (near 0) both mean "similar." So to find images like a query, you just rank every stored embedding by closeness and return the top few — search by image, with no training at all.
%%%

!!! c-info 💡
<b>Optional (skippable) — going fast at scale.</b> Comparing your query to <i>every</i> stored image is fine for thousands of photos, but slow for billions. Special "approximate nearest-neighbor" indexes (you'll meet names like FAISS and HNSW in a later ML-design lesson) find the closest pins almost as accurately, far faster. The idea today stays the same — closer embedding, more similar image.
!!!

**You've now unlocked embedding search:** turn images into pins, measure closeness, grab the nearest — real image search with zero training. That's the high point. Now let's meet the friendly gotchas so nothing surprises you, and the honest limit of borrowed eyes.

@@@ concept id=c8 tag="Gotchas & limits" title="Where borrowing trips you up, each with its fix — and what borrowed eyes can't do" gotit="Got the gotchas"
Transfer learning is wonderful, but a few classic slip-ups can quietly ruin it. Good news — each is a well-understood puzzle with a clean fix. Meet them now so none ever surprises you.

The scariest one first. Think of the hand-me-down coat one more time. A *gentle* tailor makes small, careful stitches and improves the fit. But a *reckless* tailor who yanks hard and cuts big can *destroy* a perfectly good coat in minutes. Fine-tuning is exactly this: if you nudge the borrowed layers *gently*, they improve; if you shove them hard, you wreck the very features you borrowed them for. That "shove too hard and ruin the good eye" disaster is called [[catastrophic forgetting||When fine-tuning with too big a learning rate shoves the pretrained weights far from their useful values, destroying the good features the network had learned.]] — the network forgets everything it knew. The shove that causes it is a too-big [[learning rate||A number that sets how big a step training takes each update. Too big shoves the weights far and can wreck good pretrained features; small steps keep them safe.]] (how big a step each update takes).

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A reckless-tailor analogy for catastrophic forgetting. On the left, labelled gentle small learning rate, small careful stitches improve the fit and the good features are kept, marked with a green check. On the right, labelled reckless big learning rate, big violent cuts destroy the coat and the pretrained features are wrecked, marked with a red cross. A caption says fine-tune with a small learning rate to keep the borrowed features safe.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Fine-tune GENTLY (small learning rate) — a big shove wrecks the borrowed eye</text>
<g transform="translate(30,32)"><rect x="0" y="0" width="205" height="118" fill="#EAF5EE" stroke="#2D8B55"/><text x="102" y="18" fill="#276b45">gentle: small learning rate</text><text x="102" y="40" fill="#276b45" font-size="8">tiny, careful nudges</text><text x="102" y="60" fill="#276b45" font-size="8">weights move a little →</text><text x="102" y="76" fill="#276b45" font-size="8">features improve for your task</text><text x="102" y="102" fill="#276b45">good eye kept ✓</text></g>
<text x="255" y="94" fill="#B8AEA2" font-size="13">vs</text>
<g transform="translate(285,32)"><rect x="0" y="0" width="205" height="118" fill="#FDECEC" stroke="#C93B3B"/><text x="102" y="18" fill="#8a3b3b">reckless: big learning rate</text><text x="102" y="40" fill="#8a3b3b" font-size="8">huge violent steps</text><text x="102" y="60" fill="#8a3b3b" font-size="8">weights fly far from useful →</text><text x="102" y="76" fill="#8a3b3b" font-size="8">the trained eye is destroyed</text><text x="102" y="102" fill="#8a3b3b">catastrophic forgetting ✗</text></g></g></svg>
%%%

**What the reckless-tailor picture gets right:** small careful adjustments help while violent ones destroy something valuable — the exact difference between a small and a huge learning rate when fine-tuning. **Where it breaks down:** a ruined coat is obvious at a glance, but wrecked features are *invisible* until your accuracy quietly drops — so you prevent it up front rather than spotting it after.

#### The gotchas, each with its fix
- **Gotcha 1 · Catastrophic forgetting** (cause: too big a learning rate during fine-tuning shoves the pretrained weights far from their useful values and destroys the good features). **Fix:** fine-tune with a **small learning rate**. Two named refinements make this even safer:
  - [[discriminative learning rates||Using a smaller learning rate for the early, generic layers and a larger one for the new head — so the precious early features barely move while the head learns freely.]] (also called layer-wise learning rates) — use a *tiny* rate for the precious early layers and a *bigger* one for the fresh head, so the general features barely move while the head learns fast.
  - [[gradual unfreezing||Training the new head first with the backbone frozen, then unfreezing deeper layers a few at a time — never shocking the whole network at once.]] — train the head first (backbone frozen), then unfreeze layers a few at a time, so you never shock the whole borrowed network at once.
- **Gotcha 2 · Overfitting on your small dataset** (cause: too many trainable numbers versus too few labeled photos — from concept 5). **Fix:** **freeze the backbone** and train only the head; and **augment** your small dataset for extra variety.
- **Gotcha 3 · Domain mismatch** (cause: the borrowed eye was trained on one kind of picture — the [[source domain||The kind of images the pretrained network was originally trained on — for ImageNet, everyday natural photos.]], everyday photos — but *your* pictures — the [[target domain||The kind of images YOUR task uses. If it differs a lot from the source domain, say medical scans or satellite views, more fine-tuning is needed.]] — are very different, like medical scans or satellite views). The generic edge-and-texture layers still help, but the higher borrowed features may not fit. **Fix:** don't freeze everything — **fine-tune more layers**, or do a full fine-tune, so the borrowed eye adapts to your world.
- **Gotcha 4 · Preprocessing mismatch** (cause: you feed the network a photo that isn't resized and normalized with the *exact same* recipe the pretrained model expects — from last lesson's normalization rule — so the numbers arrive on the wrong scale and the embedding is garbage). **Fix:** always apply the **pretrained model's exact preprocessing** — the same resize and the same per-color average and spread it was trained with.

!!! c-warn ⚠️
<b>The quiet trap:</b> all four of these usually run without any error — the code works, the pictures look fine. They show up only as *mysteriously worse* accuracy or *nonsense* search results. When results disappoint, check the boring things first: <i>Did I use the model's exact preprocessing? Is my learning rate small enough? Should I freeze more, or fine-tune more?</i>
!!!

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A before-and-after of preprocessing mismatch. On the left, labelled wrong recipe, a photo resized and normalized differently than the model expects gives a garbage embedding and wrong search results, marked with a red cross. On the right, labelled model's exact recipe, the same photo processed with the pretrained model's own resize and normalization gives a good embedding and correct results, marked with a green check. A caption says always match the pretrained model's exact preprocessing.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Feed the borrowed eye its OWN preprocessing recipe — or the embedding is junk</text>
<g transform="translate(30,30)"><rect x="0" y="0" width="205" height="100" fill="#FDECEC" stroke="#C93B3B"/><text x="102" y="18" fill="#8a3b3b">wrong recipe</text><text x="102" y="40" fill="#8a3b3b" font-size="8">different resize / different avg,spread</text><text x="102" y="58" fill="#8a3b3b" font-size="8">→ numbers on the wrong scale</text><text x="102" y="88" fill="#8a3b3b">garbage embedding ✗</text></g>
<text x="255" y="82" fill="#B8AEA2" font-size="13">→</text>
<g transform="translate(285,30)"><rect x="0" y="0" width="205" height="100" fill="#EAF5EE" stroke="#2D8B55"/><text x="102" y="18" fill="#276b45">model's exact recipe</text><text x="102" y="40" fill="#276b45" font-size="8">same resize + same avg,spread</text><text x="102" y="58" fill="#276b45" font-size="8">→ numbers on the right scale</text><text x="102" y="88" fill="#276b45">good embedding ✓</text></g></g></svg>
%%%

#### The honest limits: what borrowed eyes can't do
Two plain truths worth stating outright, so you never over-trust the trick:

- **A frozen ImageNet eye can only "see" what it already learned.** Its embeddings capture the everyday things it was trained on. Ask a frozen backbone to represent something *far* outside that world — a totally new kind of object it never saw — and its summary will be weak, because it has no features for it. **The way out:** don't freeze — *fine-tune* (or fully train) so the eye can learn the new thing.
- **Visual similarity is not the same as your labels.** Embeddings measure "these two look alike," which is *usually* helpful but *not* your task's truth. Two images can look nearly identical yet belong to *different* classes (a real banknote vs. a fake), and two images of the *same* class can look very different (a wolf photographed in snow vs. in a forest). So "close embedding" is a strong hint, not a guarantee.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A capability-limit picture. On the left, a frozen ImageNet eye can represent everyday things it saw but returns a weak summary for a brand-new out-of-distribution object, with a note that the fix is to fine-tune. On the right, two visually near-identical images marked different classes and two visually different images marked same class, showing visual similarity is not the same as the label. A caption says borrowed eyes see what they learned, and looking alike is not the same as being the same class.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Two honest limits of a borrowed, frozen eye</text>
<g transform="translate(24,30)"><rect x="0" y="0" width="230" height="122" fill="#FDF9EF" stroke="#C99A12"/><text x="115" y="18" fill="#8A6D3B" font-size="8">limit 1 · only sees what it learned</text><text x="60" y="44" font-size="16">🐱</text><text x="60" y="64" fill="#276b45" font-size="7.5">everyday → good ✓</text><text x="165" y="44" font-size="16">🛰️?</text><text x="165" y="64" fill="#8a3b3b" font-size="7.5">brand-new → weak ✗</text><text x="115" y="92" fill="#5E5191" font-size="8">fix: fine-tune so it can learn it</text><text x="115" y="110" fill="#8A6D3B" font-size="7.5">frozen = can't represent the truly unseen</text></g>
<g transform="translate(266,30)"><rect x="0" y="0" width="230" height="122" fill="#FDECEC" stroke="#C93B3B"/><text x="115" y="18" fill="#8a3b3b" font-size="8">limit 2 · looks alike ≠ same class</text><text x="55" y="46" font-size="14">💵</text><text x="90" y="46" font-size="14">💵</text><text x="55" y="64" fill="#8a3b3b" font-size="7">near-identical, DIFFERENT class</text><text x="165" y="46" font-size="14">🐺</text><text x="200" y="46" font-size="14">🐺</text><text x="182" y="64" fill="#8a3b3b" font-size="7">very different, SAME class</text><text x="115" y="100" fill="#8a3b3b" font-size="8">close embedding = a hint, not a promise</text></g></g></svg>
%%%

**You now hold every gotcha, its fix, and the honest limits.** Nothing here is scary once named. Let's gather the whole day onto one page.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
You built the whole hand-me-down idea today. Before we file the day away, let's lay every piece out on one page.

Think of the last day of a **school field trip**, when you sit down with your **scrapbook**. All day you collected little things — a ticket stub, a leaf, a photo. Now you lay them out on **one page**, in order, so a single glance replays the whole trip. You're not *doing* anything new; you're *gathering* what you already have so it fits in one look. This recap is your scrapbook page for transfer learning: every idea you met, side by side, in order.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A scrapbook-page analogy for the recap. A single page holds several taped-on keepsakes laid out in order: from scratch, borrow, early vs late, swap head, freeze or fine-tune, and embedding, each a small labelled card at a slight angle. A caption says one page gathers the whole day, each keepsake a hook back to a moment.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A recap is a scrapbook page: the whole day gathered in one glance</text>
<rect x="24" y="30" width="472" height="120" fill="#FDF9EF" stroke="#C99A12" stroke-width="1.5" rx="4"/>
<g transform="translate(40,50) rotate(-6)"><rect x="0" y="0" width="66" height="46" fill="#FDECEC" stroke="#C93B3B"/><text x="33" y="22" fill="#8a3b3b" font-size="8">from</text><text x="33" y="34" fill="#8a3b3b" font-size="8">scratch</text></g>
<g transform="translate(118,56) rotate(4)"><rect x="0" y="0" width="66" height="46" fill="#EAF5EE" stroke="#2D8B55"/><text x="33" y="27" fill="#276b45" font-size="8">borrow</text></g>
<g transform="translate(196,48) rotate(-3)"><rect x="0" y="0" width="72" height="46" fill="#FDF2E0" stroke="#C99A12"/><text x="36" y="22" fill="#8A6D3B" font-size="8">early vs</text><text x="36" y="34" fill="#8A6D3B" font-size="8">late</text></g>
<g transform="translate(280,56) rotate(5)"><rect x="0" y="0" width="66" height="46" fill="#EAF0FA" stroke="#5E5191"/><text x="33" y="22" fill="#5E5191" font-size="8">swap</text><text x="33" y="34" fill="#5E5191" font-size="8">head</text></g>
<g transform="translate(356,48) rotate(-4)"><rect x="0" y="0" width="72" height="46" fill="#FCF3DC" stroke="#C99A12"/><text x="36" y="22" fill="#8A6D3B" font-size="8">freeze /</text><text x="36" y="34" fill="#8A6D3B" font-size="8">fine-tune</text></g>
<g transform="translate(436,54) rotate(3)"><rect x="0" y="0" width="52" height="46" fill="#EAF5EE" stroke="#2D8B55"/><text x="26" y="20" fill="#276b45" font-size="7.5">embed</text><text x="26" y="34" fill="#276b45" font-size="7.5">+ search</text></g>
<text x="260" y="168" fill="#6B645E">laid out in order — each keepsake a hook back to a moment</text></g></svg>
%%%

**What the scrapbook picture gets right:** nothing new is created — you gather what you already collected into one ordered page. **Where it breaks down:** scrapbook keepsakes just sit there, but these ideas *connect* — you borrow an eye *because* scratch is too hard, keep its generic front, swap its specific head, choose freeze-or-fine-tune, and read out embeddings you can search — so this "page" is really a working recipe.

Here's the whole pipeline as one connected picture — a borrowed eye, a new head, and the embedding you can search:

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A one-page summary of transfer learning. An image enters a borrowed frozen backbone which outputs an embedding. From the embedding, two paths: one path to a small new head that outputs your classes, the other path to nearest-neighbor search that returns similar images with no training. A note says freeze for small data or fine-tune gently with a small learning rate, and always use the model's exact preprocessing.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">The pipeline: borrowed eye → embedding → (your head) or (search)</text>
<g transform="translate(16,60)"><rect x="0" y="0" width="58" height="40" fill="#FDF9EF" stroke="#C99A12"/><text x="29" y="18" fill="#8A6D3B">image</text><text x="29" y="32" fill="#8A6D3B" font-size="7.5">(preproc!)</text></g>
<text x="78" y="82" fill="#B8AEA2">→</text>
<g transform="translate(94,56)"><rect x="0" y="0" width="120" height="48" fill="#EAF5EE" stroke="#2D8B55"/><text x="60" y="18" fill="#276b45">borrowed backbone</text><text x="60" y="32" fill="#276b45" font-size="7.5">freeze OR fine-tune gently</text><text x="60" y="44" fill="#276b45" font-size="7.5">(small learning rate)</text></g>
<text x="218" y="82" fill="#B8AEA2">→</text>
<g transform="translate(234,58)"><rect x="0" y="0" width="90" height="44" fill="#EAF0FA" stroke="#5E5191"/><text x="45" y="20" fill="#5E5191">embedding</text><text x="45" y="34" fill="#5E5191" font-size="7.5">[512 numbers]</text></g>
<line x1="324" y1="72" x2="352" y2="44" stroke="#B8AEA2"/><line x1="324" y1="90" x2="352" y2="118" stroke="#B8AEA2"/>
<g transform="translate(356,24)"><rect x="0" y="0" width="150" height="40" fill="#EAF5EE" stroke="#2D8B55"/><text x="75" y="18" fill="#276b45">small NEW head → your classes</text><text x="75" y="32" fill="#276b45" font-size="7.5">(train this — fast, few photos)</text></g>
<g transform="translate(356,98)"><rect x="0" y="0" width="150" height="40" fill="#FDF2E0" stroke="#C99A12"/><text x="75" y="18" fill="#8A6D3B">nearest-neighbor search</text><text x="75" y="32" fill="#8A6D3B" font-size="7.5">(similar images, NO training)</text></g></g></svg>
%%%

The day in a few beats:
- **Starting from zero.** A fresh network is a newborn eye; it needs millions of labeled photos, so training from scratch overfits on small data.
- **The trained eye.** Borrow a **pretrained backbone** (trained on ImageNet) as a hand-me-down feature extractor — someone already did the hard part.
- **Early vs. late.** Early layers are **generic** (edges, textures — reuse them); late layers are **specific** to the old classes (replace them). That's why transfer works.
- **Swap the head.** Pop off the old classifier head, snap on a small new **head** sized to your classes — it trains fast on few photos.
- **Freeze or fine-tune.** **Freeze** (train only the head) is safest on small data; **fine-tune** (unlock some layers, gently) adapts more when you have data and a different task.
- **The embedding.** The penultimate-layer vector is a short numeric summary of the whole image — similar images get similar embeddings.
- **Search by distance.** Measure closeness with **cosine similarity** (near 1) or **Euclidean/L2 distance** (small); grab the **nearest neighbors** for image search — no training needed.
- **Gotchas.** Catastrophic forgetting (small learning rate; discriminative rates; gradual unfreezing); overfitting (freeze + augment); domain mismatch (fine-tune more layers); preprocessing mismatch (use the model's exact recipe).
- **The limits.** A frozen eye only sees what it learned; visual similarity is a hint, not your label.

#### Cheat-sheet · the two reuse modes at a glance
%%% table
:: Mode :: What trains :: Reach for it when :: Main risk & fix
Feature extraction (freeze) :: only the new head :: little data; task near ImageNet :: underfit if task differs → fine-tune more
Fine-tuning :: head + some backbone layers :: more data; task a bit different :: forgetting → small/discriminative LR, gradual unfreeze
Embeddings only (no head) :: nothing at all :: search / retrieval / grouping :: looks-alike ≠ same label → treat as a hint
%%%

#### Cheat-sheet · the words you met today
%%% jargon
transfer learning | reusing a network trained on a big dataset instead of training from scratch
pretrained backbone | the borrowed, already-trained stack of layers (the feature extractor / "eye")
generic vs specific | early layers transfer (edges/textures); late layers are tied to the old task
head | the small final piece you replace and train, sized to your own classes
freeze | lock a layer's numbers so training leaves them unchanged
fine-tune | unlock some pretrained layers and keep training them gently
embedding | a short list of numbers summarizing a whole image (the penultimate-layer features)
cosine / L2 | rulers for "close": cosine near 1, or small Euclidean distance = similar
nearest-neighbor search | find the closest embeddings = the most similar images (no training)
catastrophic forgetting | too-big learning rate wrecks the good pretrained features
domain mismatch | source images (photos) differ from your target images (scans/satellite)
preprocessing mismatch | not using the model's exact resize + normalize → garbage embeddings
%%%

That's the whole day. You can now explain why we borrow a trained eye, which layers to keep versus replace, freeze versus fine-tune, what an embedding is, how to search images by distance, and the gotchas with their fixes — plus the honest limits. Next you'll learn how a network finds *where* objects are, not just *what* they are, in **day-07, Object Detection (Intuition)**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is transfer learning, and why is it useful? | a:2 | training a bigger network so it learns faster | collecting more labeled photos to grow your dataset | reusing a network already trained on a big dataset (like ImageNet) instead of training from scratch, so you need far less data and time | deleting hard photos from the training set | fb: Transfer learning borrows a pretrained "eye" — a backbone trained on a mountain of photos — so you only train a small new part. That's why a small team with few photos can still build a working vision model.
q: Why can you reuse a pretrained network's EARLY layers but usually REPLACE its LAST layers? | a:1 | early layers are faster to compute than late ones | early layers learn generic basics (edges, textures) useful for any image, while late layers are specialized to the original task's exact classes | early layers use less memory | late layers have fewer numbers | fb: Early layers learn general edges/colors/textures that help any photo task, so they transfer. The last layers were tuned to name the ORIGINAL categories, which are probably not yours — so you swap them for a new head.
q: You fine-tune a pretrained backbone with a very LARGE learning rate on a small dataset and accuracy collapses. What happened, and what's the fix? | a:2 | the model ran out of memory; use a smaller batch | the learning rate was too small; make it bigger | a too-big learning rate shoved the weights far from their useful values (catastrophic forgetting), wrecking the borrowed features — fix with a small/discriminative learning rate and gradual unfreezing | you forgot to add more layers | fb: A big learning rate is a reckless tailor: it destroys the good pretrained features. Fine-tune gently — small learning rate, lower rates for early layers than the head (discriminative rates), and unfreeze layers gradually.
q: You built image search from embeddings, but results are nonsense. Your query photo was resized and normalized differently than the pretrained model expects. Why does that break it? | a:1 | the embedding is computed on wrong-scale numbers, so it no longer lands near similar images — you must apply the model's exact preprocessing (same resize + same mean/std) | embeddings always need training first | cosine similarity only works on training images | you must augment the query at test time | fb: A pretrained model expects its OWN preprocessing (specific resize + per-color average and spread). Feed it a differently-processed image and the numbers arrive on the wrong scale, so the embedding is garbage and lands nowhere near similar images. Always reuse the model's exact preprocessing.
%%%

@@@ produce id=produce tag="Produce" title="Turn images into embeddings, then search by distance" gotit="Done"
Time to run the hand-me-down idea yourself and *watch* it work. You'll stand in for a borrowed backbone with a tiny fake "feature extractor," turn a few images into embeddings, and **measure closeness** so you can see similar images land near each other. Then you'll show two failure puzzles from today with your own eyes. **Predict first:** if a "cat" and a "kitten" get similar number-lists and a "car" gets a very different one, which pair will have the *smaller* distance? And if you normalize a query with the *wrong* numbers, should it still land near the right images? Then run it and **observe** that cat and kitten sit close (high cosine, small L2) while the car sits far, that nearest-neighbor search returns the cats first, and that a preprocessing mismatch scrambles the ranking. Pick one path.

#### Option A · write it yourself
Create `sessions/m06-cnns-vision-encoders/day-06-transfer-learning-embeddings/experiment.py`. Simulate the borrowed eye and **print values at every step**. (1) **Fake embeddings:** make a stand-in feature extractor by writing fixed embeddings, e.g. `cat = np.array([0.8,0.1,0.7,0.2]); kitten = np.array([0.7,0.15,0.79,0.11]); car = np.array([0.1,0.9,0.0,0.8])` — **print** all three and **notice** cat and kitten look alike while car looks different. (2) **Cosine similarity:** write `cosine(a,b) = a@b / (np.linalg.norm(a)*np.linalg.norm(b))` — **print** `cosine(cat, kitten)` (near 1) and `cosine(cat, car)` (near 0), and **notice** near-1 means "same direction = similar." (3) **Euclidean/L2 distance:** write `l2(a,b) = np.linalg.norm(a-b)` — **print** `l2(cat, kitten)` (small) and `l2(cat, car)` (large), and **notice** small distance also means similar — both rulers agree. (4) **Nearest-neighbor search:** put `{"cat":cat,"kitten":kitten,"car":car}` in a small database, take a new query `query = np.array([0.78,0.12,0.74,0.15])`, rank the database by cosine to the query, and **print** the ranking — **observe** the two cats come first, the car last (image search with no training). (5) **Preprocessing-mismatch demo:** make `query_bad = (query - 5.0) / 0.1` (a *wrong* normalization) and rank again — **observe** the ranking is scrambled/nonsense, showing why you must use the model's exact preprocessing. Run with `python3 sessions/m06-cnns-vision-encoders/day-06-transfer-learning-embeddings/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 6 Day 6 artifact.

Create sessions/m06-cnns-vision-encoders/day-06-transfer-learning-embeddings/experiment.py that demonstrates image embeddings and nearest-neighbor search using tiny FAKE embeddings (no real model needed), with a comment on each step and printing values at every step:
1. Define fixed stand-in embeddings for three images: cat = np.array([0.8,0.1,0.7,0.2]); kitten = np.array([0.7,0.15,0.79,0.11]); car = np.array([0.1,0.9,0.0,0.8]). Print all three and point out cat and kitten look similar while car is different.
2. Cosine similarity: def cosine(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))). Print cosine(cat,kitten) (~near 1) and cosine(cat,car) (~near 0); explain near-1 = same direction = similar.
3. Euclidean/L2 distance: def l2(a,b): return float(np.linalg.norm(a-b)). Print l2(cat,kitten) (small) and l2(cat,car) (large); point out small distance also = similar, so both rulers agree.
4. Nearest-neighbor search: db = {"cat":cat,"kitten":kitten,"car":car}; query = np.array([0.78,0.12,0.74,0.15]); rank db keys by cosine(query, emb) descending; print the ranked list and assert the two cats rank above the car. Note this is image search with NO training.
5. Preprocessing-mismatch gotcha: query_bad = (query - 5.0)/0.1 (a wrong normalization the model never expected); rank db by cosine(query_bad, emb) and print it; show the ranking is scrambled compared to step 4, and warn that feeding a differently-preprocessed image gives garbage embeddings — the fix is to always apply the pretrained model's EXACT resize + normalization.
6. Print a one-line takeaway: "transfer learning borrows a trained backbone; its penultimate features are an embedding; similar images have close embeddings (high cosine / small L2), enabling nearest-neighbor search with no training — but only if you use the model's exact preprocessing."
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (predict, then confirm)
- Cat and kitten have a high cosine similarity (near `1`) and a small L2 distance, while cat and car have low cosine (near `0`) and a large L2 — both rulers agree that "close = similar."
- Nearest-neighbor search on the query returns the two cats *first* and the car *last* — real image search with no training at all.
- The wrongly-normalized `query_bad` scrambles the ranking — proof that a preprocessing mismatch makes embeddings garbage, so you must reuse the model's exact preprocessing.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m06-cnns-vision-encoders/day-06-transfer-learning-embeddings/log.md`: (1) what transfer learning is and why we keep early layers but replace late ones; (2) the difference between freezing and fine-tuning, and when you'd pick each; (3) what an embedding is and how you'd use one to search for similar images.
!!!

@@@ fin

