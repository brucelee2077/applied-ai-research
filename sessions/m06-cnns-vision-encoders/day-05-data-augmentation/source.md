---
quest_id: wf8-d05-augment
mode: concept
donor: v9-base.donor
page_title: "Module 6 · Day 5 — Data Augmentation"
module_label: "Module 06 · Represent · Day 5"
title: "Data Augmentation"
subtitle: "Teaching From the Same Photo, Many Ways"
brand_sub: "Foundations · M6 Day 5"
spine: "disguise"
nav_prev_href: "../day-04-batch-normalization/lesson.html"
nav_prev_label: "Batch Normalization"
nav_next_href: "../day-06-transfer-learning-embeddings/lesson.html"
nav_next_label: "Transfer Learning & Embeddings"
fin_title: "Module 6 · Day 5 complete! 🏆"
fin_body: "Great — you've unlocked <b>Data Augmentation</b>: showing the model the same photo in many <b>disguises</b> — flipped, cropped, rotated, recolored — so it learns the essence of a cat instead of memorizing one exact picture. It's the cheapest way to fight overfitting and squeeze more out of the data you already have.<br>Next up: <b>Transfer Learning & Embeddings</b>."
notebook_yardstick: null
---

@@@ hero
@lede Think about a friend you'd know anywhere — in a hat, in sunglasses, from behind, in bad light, laughing or frowning. You don't need to see the *exact* same photo twice to recognize them; you've learned the *essence* of their face, so any **disguise** falls right off. Now here's a puzzle. A fresh neural network is the opposite: show it one photo of a cat sitting perfectly centered in bright light, and it can end up memorizing *that one picture* — cat-in-the-middle, this exact pose, this exact color — instead of learning what a cat *is*. Show it the same cat slightly off to the side and it shrugs. Today you learn the cheapest, most beloved trick for fixing this: instead of hunting for a million new photos, you take the photos you already have and show them to the model in many **disguises** — flipped, cropped, tilted, recolored. Same cat, a hundred outfits. The model stops memorizing one picture and starts learning the real thing. This trick, called **data augmentation**, is baked into almost every image model you've ever heard of.
@goal Together we'll build the disguise kit one piece at a time. You'll see the memorizing trap it fixes, the big idea that a disguise must keep the label true, the everyday transforms (flip, crop, tilt, zoom, recolor), the always-on normalize step, the one iron rule about *when* to disguise, a look back at where the trick was born, and — as friendly puzzles — the ways it can trip you, each with its fix, plus the honest limit of what disguises can and can't do. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="The memorizing trap" title="Why a model overfits: it memorizes the photos instead of learning the thing" gotit="Got the trap"
Let's start with the problem the disguise trick is here to fix — because once you *feel* it, the fix will make instant sense.

Picture a student studying for a test the lazy way: instead of *understanding* the topic, they memorize the exact answers to last year's practice test, word for word. Give them that exact practice test again and they score 100%. But hand them the *real* test — same topic, different questions — and they fall apart, because they never learned the idea, only the specific answers. A neural network with too few pictures does exactly this. If it only ever sees one photo of a cat — centered, bright, this exact pose — it can just memorize *that photo* and score perfectly on it, without ever learning what makes a cat a cat. This "great on the pictures I studied, useless on new ones" trap is called [[overfitting||When a model learns the training pictures so exactly that it fails on new, unseen pictures — like a student who memorized the practice test but can't answer the real one.]] — and it is the number-one thing today's disguise trick exists to fight.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A crammer-student analogy for overfitting. On the left, labelled memorized the exact photos, a student scores 100 percent on the exact studied photos but 40 percent on new photos. On the right, labelled learned what a cat is, a student scores well on both. A caption says memorizing the exact photos gives great scores on those photos but fails on new ones.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Overfitting = the crammer: aces the studied photos, flunks new ones</text>
<g transform="translate(30,34)"><rect x="0" y="0" width="200" height="126" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="20" fill="#8a3b3b">memorized the exact photos</text><rect x="30" y="40" width="140" height="20" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="55" fill="#276b45" font-size="8">studied photos: 100% ✓</text><rect x="30" y="80" width="140" height="20" fill="#FDF2E0" stroke="#C99A12"/><text x="100" y="95" fill="#8a3b3b" font-size="8">NEW photos: 40% ✗</text><text x="100" y="118" fill="#8a3b3b" font-size="8">great on the studied set, lost on the rest</text></g>
<text x="255" y="98" fill="#B8AEA2" font-size="13">vs</text>
<g transform="translate(290,34)"><rect x="0" y="0" width="200" height="126" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="20" fill="#276b45">learned what a cat IS</text><rect x="30" y="40" width="140" height="20" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="55" fill="#276b45" font-size="8">studied photos: 96% ✓</text><rect x="30" y="80" width="140" height="20" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="95" fill="#276b45" font-size="8">NEW photos: 93% ✓</text><text x="100" y="118" fill="#276b45" font-size="8">solid on both — it learned the thing</text></g></g></svg>
%%%

**What the crammer picture gets right:** memorizing exact answers gives a fake sense of success — a perfect score on the studied set that vanishes on anything new. **Where it breaks down:** a student *chooses* to cram; a network overfits *by accident*, just because it has enough spare capacity and too few pictures, so it takes the easy path of memorizing.

#### Why "too few pictures" is the root cause
Overfitting gets worse when a big, powerful network meets a *small* pile of pictures. The network has plenty of room to memorize, and not enough variety to force it to generalize. So there are really two ways out: **get more pictures** (expensive — you'd have to go photograph a thousand more cats), or **make more variety out of the pictures you already have** (nearly free). The second path is today's whole subject.

You'll hear two more named tricks that also fight overfitting from a different angle — [[dropout||A regularizer that randomly switches off some of the network's units during training, so it can't lean too hard on any one of them. Covered in its own lesson.]] (randomly switch off some units so the network can't over-rely on any one) and [[weight decay||A gentle pull that keeps the network's weights small, discouraging it from fitting every tiny quirk of the training set. Covered in its own lesson.]] (gently keep the weights small). Those are *regularizers* — teammates of augmentation, each covered in its own lesson. Augmentation is the one that works by giving the network more *variety*. Let's see how a single big idea makes that variety safe.

@@@ concept id=c2 tag="Disguise the photo" title="The big idea: same photo, many disguises — but the label must stay true" gotit="Got the disguise idea"
Here's the heart of the whole day, and it's beautifully simple. To give the network more variety without new photos, we take each photo and show it wearing many **disguises** — a flip here, a crop there, a little tilt, a color change. One photo becomes a whole crowd of slightly different photos. But there's one golden rule that makes it all work.

Think about recognizing your best friend again. A hat, sunglasses, a new haircut — those are **disguises that don't change who they are**. You still say "that's them." But if a disguise went too far — a full face mask, a costume that hid every feature — you'd no longer be sure it was your friend at all. A good disguise hides the *unimportant* stuff (the outfit) while keeping the *important* stuff (the face) intact. Data augmentation follows exactly this rule: every disguise must keep the true answer — the [[label||The correct answer attached to a training photo, like "cat" or "dog." A disguise must never change what the label should be.]] — the same. A flipped cat is still a cat. So the iron rule is: **a disguise must be [[label-preserving||A transform that changes the photo but not its correct answer — a flipped cat is still labeled "cat." Disguises that break this rule teach the model wrong answers.]]** — it changes the picture, never the answer.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A friend-in-disguise analogy for label-preserving augmentation. In the center a plain face labelled cat. Around it, four disguised versions labelled flipped, cropped, tilted, and recolored, each still labelled cat with a green check. A caption says a good disguise changes the look but keeps the label true; go too far and the label breaks.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Disguise the photo — but the label ("cat") must survive</text>
<g transform="translate(215,60)"><rect x="0" y="0" width="90" height="70" fill="#FDF9EF" stroke="#C99A12" stroke-width="1.5"/><text x="45" y="30" fill="#8A6D3B" font-size="8">original</text><text x="45" y="52" fill="#276b45">"cat"</text></g>
<g fill="#EAF5EE" stroke="#2D8B55"><rect x="40" y="40" width="70" height="34"/><rect x="40" y="120" width="70" height="34"/><rect x="410" y="40" width="70" height="34"/><rect x="410" y="120" width="70" height="34"/></g>
<g font-size="7.5" fill="#276b45"><text x="75" y="54">flipped</text><text x="75" y="66">"cat" ✓</text><text x="75" y="134">cropped</text><text x="75" y="146">"cat" ✓</text><text x="445" y="54">tilted</text><text x="445" y="66">"cat" ✓</text><text x="445" y="134">recolored</text><text x="445" y="146">"cat" ✓</text></g>
<g stroke="#B8AEA2"><line x1="115" y1="66" x2="210" y2="80"/><line x1="115" y1="130" x2="210" y2="112"/><line x1="405" y1="66" x2="310" y2="80"/><line x1="405" y1="130" x2="310" y2="112"/></g>
<text x="260" y="188" fill="#8a3b3b" font-size="8">go too far (a full mask) and the label breaks — that's the one thing to avoid</text></g></svg>
%%%

**What the friend picture gets right:** you recognize a person through many surface changes because none of them touch the essence — and the disguise fails only when it hides that essence. **Where it breaks down:** with a friend *you* decide when a disguise went too far; with augmentation *you* must pick transforms that are safe *ahead of time*, because the model will trustingly believe whatever label you attach.

#### Why this teaches the model to ignore the right things
When the model sees the same cat flipped, shifted, and recolored — all still labeled "cat" — it slowly learns a wonderful lesson: *position, direction, and lighting don't decide the class.* Those are [[nuisance factors||Things about a photo that change how it looks but should NOT change its label — like where the object sits, which way it faces, or the lighting. Augmentation teaches the model to ignore them.]] — the stuff you *want* the model to ignore. Teaching a model to give the same answer no matter how a nuisance factor changes is called building [[invariance||The property of giving the same answer even when an unimportant thing about the input changes. A model invariant to flipping says "cat" whether the cat faces left or right.]]. A model *invariant* to flipping says "cat" whether the cat faces left or right.

Watch what one photo becomes when you apply five safe disguises:

%%% demo id=onephoto label="disguise one cat five ways"
code: augment(cat_photo, n=5)   # apply random safe disguises
out: 5 new images  →  [flip-LR, crop+zoom, tilt 8°, brighter, hue-shift]   labels: ["cat","cat","cat","cat","cat"]
take: <b>One photo became five training examples — and every label stayed "cat."</b> The model now sees a cat that faces both ways, sits off-center, tilts a little, and comes in different light. It stops keying on "cat is always dead-center and bright" and starts learning the cat itself.
%%%

**Notice:** one photo turned into five, and the label never budged. That's the whole game — free variety, same truth. Now let's open the disguise kit and look at each tool inside.

@@@ concept id=c3 tag="Geometry disguises" title="Moving the pixels around: flip, crop, tilt, zoom" gotit="Got the geometry kit"
Time to open the disguise kit. The first drawer holds the **geometry** disguises — the ones that move pixels *around* without changing their color. These are the cheapest and most common, and each one teaches the model to ignore one specific nuisance factor.

Think of the same toy on a table, and you looking at it from different spots. Step to the other side and it faces the other way (that's a **flip**). Lean in close so it fills your view and the edges drop out of sight (that's a **crop**). Tip your head a little and it looks tilted (that's a **rotation**). Walk backward and it looks smaller, or forward and it looks bigger (that's **zoom**). The toy never changed — *you* moved. Geometry disguises move the "camera" for the model, so it learns the toy is the same toy from every angle and distance.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Four geometry disguises drawn on the same simple shape. Panel one, horizontal flip: an arrow facing right and its mirror facing left. Panel two, random crop: a full frame and a smaller box zoomed into part of it. Panel three, rotation: a small square tilted by a small angle. Panel four, zoom: a small square and a larger square. Each panel is labelled and marked cat with a check.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">The geometry drawer: flip · crop · tilt · zoom — pixels move, label stays "cat"</text>
<g transform="translate(20,32)"><rect x="0" y="0" width="115" height="120" fill="#FDF9EF" stroke="#C99A12"/><text x="57" y="16" fill="#8A6D3B">① horizontal flip</text><path d="M30 70 l14 -14 l6 10 l6 -10 l14 14 z" fill="#C99A12" opacity="0.5"/><text x="40" y="90" fill="#8A6D3B">▶</text><text x="70" y="90" fill="#5E5191">◀</text><text x="57" y="112" fill="#276b45">left↔right · "cat" ✓</text></g>
<g transform="translate(147,32)"><rect x="0" y="0" width="115" height="120" fill="#FDF9EF" stroke="#C99A12"/><text x="57" y="16" fill="#8A6D3B">② random crop</text><rect x="20" y="28" width="76" height="56" fill="none" stroke="#B8AEA2" stroke-dasharray="3,2"/><rect x="42" y="40" width="34" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="57" y="112" fill="#276b45">a sub-region · "cat" ✓</text></g>
<g transform="translate(274,32)"><rect x="0" y="0" width="115" height="120" fill="#FDF9EF" stroke="#C99A12"/><text x="57" y="16" fill="#8A6D3B">③ small rotation</text><g transform="rotate(12 57 70)"><rect x="37" y="50" width="40" height="40" fill="#EAF0FA" stroke="#5E5191"/></g><text x="57" y="112" fill="#276b45">tilt a few ° · "cat" ✓</text></g>
<g transform="translate(401,32)"><rect x="0" y="0" width="115" height="120" fill="#FDF9EF" stroke="#C99A12"/><text x="57" y="16" fill="#8A6D3B">④ zoom in/out</text><rect x="30" y="55" width="20" height="20" fill="#EAF5EE" stroke="#2D8B55"/><rect x="62" y="42" width="44" height="44" fill="#EAF5EE" stroke="#2D8B55"/><text x="57" y="112" fill="#276b45">near/far · "cat" ✓</text></g>
<text x="260" y="192" fill="#6B645E">you moved the "camera"; the cat is still a cat</text></g></svg>
%%%

**What the moving-around-the-table picture gets right:** the object stays the same while your viewpoint changes, which is exactly the lesson we want the model to absorb — "same thing, new angle." **Where it breaks down:** in real life *you* move continuously; augmentation applies a *random* jump each time the photo is used, so the model sees a fresh, unpredictable viewpoint on every pass.

#### The four workhorses, one line each
- **Horizontal flip.** Mirror the photo left-to-right. For natural photos this is almost always safe (a cat facing left is still a cat) and dirt cheap — often the very first disguise people add.
- **Random crop** (and **random resized crop**). Cut out a random sub-region of the photo — and often zoom that patch back up to full size. This teaches the model the object need not be centered or fill the frame. It's the single most powerful geometry disguise for photo models.
- **Rotation.** Tilt the photo by a small random angle (say ±10°). Teaches tolerance to a slightly crooked camera.
- **Scaling / zoom.** Shrink or enlarge, so the object appears nearer or farther. Teaches the model that size shouldn't decide the class.

Here's the important habit these share: **the disguise is random and fresh every time.** The same photo is flipped on one pass, cropped on the next, tilted on a third — so across training the model effectively sees an endless stream of variations. Predict what you get from asking for many disguises of one image, then run it:

%%% demo id=randomcrop label="random crop, three times"
code: [random_resized_crop(cat_photo) for _ in range(3)]   # same photo, 3 fresh crops
out: crop A = top-left ear + face,   crop B = whole body zoomed,   crop C = off-center face
take: <b>Same photo in, three different sub-views out.</b> Because the crop box is chosen at random each time, one photo behaves like many. The model can never rely on "the cat is always centered" — it learns to find the cat wherever it sits.
%%%

**You now hold the geometry drawer.** But moving pixels around is only half the kit — the other half changes their *color*, which is the next disguise.

@@@ concept id=c4 tag="Color disguises" title="Changing the light: brightness, contrast, saturation, hue" gotit="Got color jitter"
The second drawer of the disguise kit doesn't move pixels — it recolors them. In the real world the *same* cat looks different in morning light, under a warm lamp, or through a cheap phone camera. We want the model to see through all that. So we teach it by nudging the colors of our photos on purpose.

Think of the **filters on a phone camera** — the little sliders that make a photo warmer, dimmer, punchier, or shift its tint. Slide "brightness" up and the whole scene glows; slide it down and it dims. Slide "contrast" and the difference between light and dark grows or shrinks. Slide "saturation" and colors go vivid or washed-out. Slide "hue" and everything tints a bit toward red or green. The subject of the photo is unchanged — you've only changed the *light* it's wearing. That's exactly what [[color jitter||A disguise that randomly nudges a photo's brightness, contrast, saturation, and hue, so the model learns the class doesn't depend on lighting or camera color.]] does: it randomly wiggles four color sliders so the model learns a cat is a cat in any light.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A phone-camera-sliders analogy for color jitter. Four labelled slider rows: brightness, contrast, saturation, hue, each shown as a track with a knob and a small before-after swatch pair going from dim to bright, flat to punchy, gray to vivid, and one tint to another. A caption says nudge the light sliders; the subject and its label stay the same.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Color jitter = wiggling four camera sliders — same cat, different light</text>
<g transform="translate(40,36)"><text x="0" y="10" text-anchor="start" fill="#8A6D3B" font-size="8">brightness</text><line x1="90" y1="7" x2="230" y2="7" stroke="#B8AEA2"/><circle cx="180" cy="7" r="5" fill="#C99A12"/><rect x="250" y="0" width="24" height="14" fill="#6B645E"/><rect x="280" y="0" width="24" height="14" fill="#FCF3DC"/><text x="330" y="10" text-anchor="start" fill="#6B645E" font-size="7.5">dim → bright</text></g>
<g transform="translate(40,72)"><text x="0" y="10" text-anchor="start" fill="#8A6D3B" font-size="8">contrast</text><line x1="90" y1="7" x2="230" y2="7" stroke="#B8AEA2"/><circle cx="140" cy="7" r="5" fill="#C99A12"/><rect x="250" y="0" width="24" height="14" fill="#B8AEA2"/><rect x="280" y="0" width="24" height="14" fill="#2C2A28"/><text x="330" y="10" text-anchor="start" fill="#6B645E" font-size="7.5">flat → punchy</text></g>
<g transform="translate(40,108)"><text x="0" y="10" text-anchor="start" fill="#8A6D3B" font-size="8">saturation</text><line x1="90" y1="7" x2="230" y2="7" stroke="#B8AEA2"/><circle cx="200" cy="7" r="5" fill="#C99A12"/><rect x="250" y="0" width="24" height="14" fill="#C9C4BC"/><rect x="280" y="0" width="24" height="14" fill="#2D8B55"/><text x="330" y="10" text-anchor="start" fill="#6B645E" font-size="7.5">gray → vivid</text></g>
<g transform="translate(40,144)"><text x="0" y="10" text-anchor="start" fill="#8A6D3B" font-size="8">hue</text><line x1="90" y1="7" x2="230" y2="7" stroke="#B8AEA2"/><circle cx="160" cy="7" r="5" fill="#C99A12"/><rect x="250" y="0" width="24" height="14" fill="#C93B3B"/><rect x="280" y="0" width="24" height="14" fill="#5E5191"/><text x="330" y="10" text-anchor="start" fill="#6B645E" font-size="7.5">tint shifts</text></g></g></svg>
%%%

**What the phone-sliders picture gets right:** each slider changes only the *look* of the light, not the thing being photographed — the same trick we want for teaching robustness to cameras and lighting. **Where it breaks down:** phone filters are for making a photo *prettier*; color jitter deliberately makes photos *messier and more varied*, because ugly variety is what forces the model to stop keying on exact colors.

#### The four color knobs, plainly
- **Brightness** — how light or dark the whole photo is.
- **Contrast** — how big the gap is between the darkest and lightest parts.
- **Saturation** — how vivid or washed-out the colors are.
- **Hue** — a shift in the overall tint (toward red, green, blue…).

A gentle nudge on each keeps the label safe. Watch a small color jitter turn one photo into three different "lightings," predict first, then run:

%%% demo id=jitter label="jitter the colors three ways"
code: [color_jitter(cat_photo, strength=0.2) for _ in range(3)]   # small random nudges
out: v1 = a touch brighter,   v2 = slightly warmer + more vivid,   v3 = dimmer + flatter
take: <b>Same cat, three lightings — every label still "cat."</b> A small strength (0.2) keeps the disguise honest. The model learns not to depend on the exact colors of the training photos, so it survives a new camera or a cloudy day.
%%%

**You've now got both drawers of the kit** — geometry (move the pixels) and color (recolor them). There's one more step that runs on *every* photo, disguised or not, and it's the quiet workhorse behind them all.

@@@ concept id=c5 tag="Always normalize" title="The always-on step: normalize every photo the same way" gotit="Got normalization"
The disguises are random and playful — but there's one step that is *not* a disguise and runs on *every* photo, no exceptions, at both train and test time. It's called **normalization**, and getting it right (and consistent) is quietly essential.

Think of a **recipe that lists everything in grams**. If one cook measures flour in cups, another in spoons, and a third in handfuls, their cakes come out wildly different. So the recipe says: *put everything on the same scale first — grams — then cook.* Normalizing a photo does the same to its pixel numbers. Raw pixel values live on an awkward scale (often 0 to 255). [[Normalization||The always-on step that puts every photo's pixel values on the same standard scale: subtract a fixed per-color average and divide by a fixed per-color spread. The SAME numbers must be used at train and test time.]] puts every photo on one standard scale: for each color channel, subtract a fixed average and divide by a fixed spread, so the numbers come out small and centered — a comfortable range the network trains on smoothly.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A same-scale-recipe analogy for normalization. On the left, three photos with wildly different raw pixel ranges shown as bars of different heights labelled 0 to 255. An arrow through a normalize box in the middle. On the right, the same three photos now all on one small centered scale around zero, bars of similar modest height. A caption says put every photo on the same standard scale, using the same fixed numbers every time.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Normalize = one standard scale for every photo (like grams in a recipe)</text>
<g transform="translate(30,40)"><text x="55" y="8" fill="#8a3b3b" font-size="8">raw pixels (0–255): all over the place</text><rect x="10" y="20" width="26" height="90" fill="#C99A12"/><rect x="46" y="50" width="26" height="60" fill="#C99A12"/><rect x="82" y="14" width="26" height="96" fill="#C99A12"/><line x1="0" y1="110" x2="118" y2="110" stroke="#B8AEA2"/></g>
<g transform="translate(200,55)"><rect x="0" y="10" width="120" height="56" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="60" y="34" fill="#5E5191" font-size="8">normalize:</text><text x="60" y="50" fill="#5E5191" font-size="8">(pixel − avg) ÷ spread</text></g>
<text x="342" y="86" fill="#B8AEA2" font-size="13">→</text>
<g transform="translate(370,40)"><text x="60" y="8" fill="#276b45" font-size="8">standardized: small &amp; centered</text><line x1="0" y1="70" x2="120" y2="70" stroke="#B8AEA2" stroke-dasharray="2,2"/><rect x="12" y="52" width="24" height="18" fill="#2D8B55"/><rect x="48" y="72" width="24" height="16" fill="#2D8B55"/><rect x="84" y="56" width="24" height="14" fill="#2D8B55"/><text x="60" y="108" fill="#276b45" font-size="7.5">around 0, same scale for all</text></g></g></svg>
%%%

**What the grams-recipe picture gets right:** putting everything on one shared scale before you cook makes the results comparable and repeatable — the same reason we standardize pixels before the network reads them. **Where it breaks down:** a recipe's "grams" is a universal unit, but normalization uses *dataset-specific* fixed numbers (a per-color average and spread computed once from the training data) — so those exact numbers have to travel with the model.

#### The one rule that matters most: the SAME numbers, always
Here's the subtle part, and it's the source of a nasty silent bug. The average and spread used to normalize are **fixed numbers picked once** (usually the training set's per-color averages). You must use those **same numbers** to normalize *every* photo — the ones you train on *and* the ones you test on later.

!!! c-warn ⚠️
<b>Failure mode (silent) — normalization mismatch:</b> if you normalize training photos one way and then forget to apply the *identical* average and spread at test time, every test photo arrives on the wrong scale. The model, trained to expect grams, is suddenly fed cups. It doesn't crash — it just quietly gets worse. <b>The fix:</b> store the normalization numbers with the model and reuse the *exact same* ones at inference.
!!!

Here's the before-and-after in one picture:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A before-and-after of normalization mismatch. On the left, labelled mismatch, a photo normalized with train numbers but tested with different numbers arrives on the wrong scale and the prediction is wrong, marked with a red cross. On the right, labelled matched, the same fixed numbers are used at train and test, the photo is on the right scale, and the prediction is correct, marked with a green check. A caption says reuse the identical normalization numbers at test time.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">Same fixed numbers at train AND test — or the model quietly breaks</text>
<g transform="translate(30,32)"><rect x="0" y="0" width="205" height="112" fill="#FDECEC" stroke="#C93B3B"/><text x="102" y="18" fill="#8a3b3b">mismatch</text><text x="102" y="42" fill="#8a3b3b" font-size="8">train: subtract avg = 120</text><text x="102" y="60" fill="#8a3b3b" font-size="8">test: subtract avg = 0 (forgot!)</text><text x="102" y="82" fill="#8a3b3b" font-size="8">→ wrong scale at test</text><text x="102" y="102" fill="#8a3b3b">prediction wrong ✗</text></g>
<text x="255" y="90" fill="#B8AEA2" font-size="13">→</text>
<g transform="translate(285,32)"><rect x="0" y="0" width="205" height="112" fill="#EAF5EE" stroke="#2D8B55"/><text x="102" y="18" fill="#276b45">matched</text><text x="102" y="42" fill="#276b45" font-size="8">train: subtract avg = 120</text><text x="102" y="60" fill="#276b45" font-size="8">test: subtract avg = 120 ✓</text><text x="102" y="82" fill="#276b45" font-size="8">→ same scale everywhere</text><text x="102" y="102" fill="#276b45">prediction right ✓</text></g></g></svg>
%%%

**You now know the always-on step:** normalize every photo with the *same* fixed per-color numbers, train and test alike. Notice something important, though — normalization is *not* a disguise: it's applied to *every* photo, always, while the random flips and crops are applied *only sometimes* and *only during training*. That difference — *when* each thing runs — is its own iron rule, and it's the next concept.

@@@ concept id=c6 tag="Disguise train only" title="The iron rule of when: disguise the training photos only" gotit="Got the train-only rule"
This is the single rule beginners get wrong most often, and it's worth burning in. You **disguise the training photos only.** At test time — when you're checking how good the model really is, or using it for real — you show each photo *plain*, with no random disguises at all.

Think of a student **practicing** for a driving test versus **taking** it. During practice, a good coach throws every curveball — rain, night, tight parking, a surprise detour — so the student gets ready for anything. But on the *actual exam*, the examiner gives everyone the *same, fair, standard* route. You don't randomly throw rain at one student and sunshine at another — that would make the scores meaningless and unrepeatable. Training is practice: pile on the disguises. Testing is the exam: keep it plain, fixed, and fair.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A practice-versus-exam analogy for the train-only rule. On the left, labelled training is practice, a stack of one cat photo shown in many random disguises with dice symbols meaning random. On the right, labelled testing is the exam, the same cat shown once, plain, with a fixed center-crop and normalize, marked deterministic. A caption says disguise the training photos only; keep test photos plain, fixed, and fair.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Disguise TRAINING only — testing gets the plain, fixed photo</text>
<g transform="translate(30,34)"><rect x="0" y="0" width="215" height="130" fill="#EAF5EE" stroke="#2D8B55"/><text x="107" y="18" fill="#276b45">TRAINING = practice 🎲 random</text><g fill="#FCF3DC" stroke="#C99A12"><rect x="24" y="34" width="40" height="34" transform="rotate(-6 44 51)"/><rect x="86" y="30" width="40" height="34" transform="rotate(5 106 47)"/><rect x="148" y="36" width="40" height="34" transform="rotate(-4 168 53)"/></g><text x="107" y="92" fill="#276b45" font-size="8">flip / crop / tilt / jitter —</text><text x="107" y="106" fill="#276b45" font-size="8">a fresh disguise every pass</text><text x="107" y="122" fill="#276b45" font-size="8">more variety → less overfit</text></g>
<text x="255" y="100" fill="#B8AEA2" font-size="13">→</text>
<g transform="translate(285,34)"><rect x="0" y="0" width="205" height="130" fill="#EAF0FA" stroke="#5E5191"/><text x="102" y="18" fill="#5E5191">TESTING = exam ⚙️ fixed</text><rect x="72" y="34" width="60" height="44" fill="#FDF9EF" stroke="#C99A12"/><text x="102" y="60" fill="#8A6D3B" font-size="8">plain photo</text><text x="102" y="98" fill="#5E5191" font-size="8">center-crop + normalize only</text><text x="102" y="114" fill="#5E5191" font-size="8">no randomness →</text><text x="102" y="128" fill="#5E5191" font-size="8">same answer every time</text></g></g></svg>
%%%

**What the practice-vs-exam picture gets right:** you *train* against messy variety but *measure* on one fair, fixed setup — mixing them up would make the score meaningless. **Where it breaks down:** a driving exam still has *some* real-world conditions; a test-time image pipeline is deliberately made as boring and repeatable as possible (a fixed center-crop, then normalize) so results are perfectly reproducible.

#### What the test-time pipeline looks like
At test time you strip out every random step and keep only the *deterministic* ones — the ones that give the same output every time:

%%% table
:: Step :: Training pipeline :: Test / validation pipeline
Geometry :: **random** flip, crop, tilt, zoom :: a **fixed center-crop** (no randomness)
Color :: **random** color jitter :: none
Normalize :: yes (fixed numbers) :: yes (the **same** fixed numbers)
Result :: endless variety → fights overfitting :: one plain, repeatable view → fair, stable score
%%%

!!! c-warn ⚠️
<b>Failure mode — augmenting the test set:</b> if you leave the random disguises on at test time, the *same* photo gets a *different* random crop and color each run, so its prediction jumps around and your accuracy number is noisy and impossible to reproduce. Worse, you're no longer measuring the model fairly. <b>The fix:</b> switch off all random transforms for validation and test; use only the deterministic center-crop + normalize.
!!!

**You now hold the iron rule:** disguise training photos only; keep test photos plain, fixed, and fair. That's the full working recipe. Before we push on to the gotchas, let's take a quick look back at where this whole trick came from — it has a famous origin.

@@@ concept id=c7 tag="Where it began" title="A look back: the disguise trick that helped win a famous contest" gotit="Got the history"
Data augmentation feels obvious now, but it wasn't always standard. It's worth a short story, because seeing *why* people reached for it makes the whole idea click.

Think of a coach with a very talented player but only a **tiny practice field**. The player is strong enough to master every inch of that small field — but that's the trap: master the *field*, not the *game*. A clever coach fixes it cheaply: same field, but change the drills constantly — cones here, then there, play left-footed, play in the rain. Suddenly the small field *feels* huge and varied, and the player learns the real game. Back around 2012, researchers had exactly this situation: a big, hungry network and a training set that, for its appetite, felt small. Their fix was the same coach's trick — keep the photos, but disguise them on the fly.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A small-practice-field analogy for the origin of augmentation. On the left, a small field with a strong player who only masters that exact field. An arrow to the right where the same small field now has changing drills, cones in new spots, and the player learns the real game. A caption says a big model plus a small dataset plus on-the-fly disguises equals the trick that helped a famous network win in 2012.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">2012: a big hungry network + a small-feeling dataset → disguise on the fly</text>
<g transform="translate(30,34)"><rect x="0" y="0" width="195" height="118" fill="#FDECEC" stroke="#C93B3B"/><text x="97" y="18" fill="#8a3b3b">plain small dataset</text><rect x="45" y="34" width="100" height="60" fill="#EAF5EE" stroke="#2D8B55"/><circle cx="95" cy="64" r="9" fill="#C99A12"/><text x="97" y="110" fill="#8a3b3b" font-size="8">big net memorizes it → overfits</text></g>
<text x="245" y="96" fill="#B8AEA2" font-size="13">→</text>
<g transform="translate(275,34)"><rect x="0" y="0" width="215" height="118" fill="#EAF5EE" stroke="#2D8B55"/><text x="107" y="18" fill="#276b45">same photos, disguised each pass</text><rect x="55" y="34" width="100" height="60" fill="#FDF9EF" stroke="#C99A12"/><circle cx="80" cy="52" r="7" fill="#C99A12"/><circle cx="128" cy="76" r="7" fill="#5E5191"/><circle cx="100" cy="66" r="7" fill="#2D8B55" opacity="0.6"/><text x="107" y="110" fill="#276b45" font-size="8">feels huge &amp; varied → generalizes</text></g></g></svg>
%%%

**What the practice-field picture gets right:** a small space plus constantly-changing drills teaches the real skill far better than drilling one fixed layout — the exact bargain augmentation strikes. **Where it breaks down:** a coach's drills are hand-designed each day; augmentation applies its disguises *automatically and randomly*, so the "drills" never repeat and cost the coach nothing.

#### The famous origin
The network people usually point to is **AlexNet (2012)** — the model that shocked everyone by winning a big image-recognition contest by a wide margin and kicked off the deep-learning boom in vision. A key part of its recipe was cheap, on-the-fly disguises: **random crops** and **horizontal flips** of every training photo, plus a color trick (a small random shift along the image's main color directions). None of it was fancy or costly. But it was a big reason a hungry network could *generalize* instead of just memorizing. Ever since, these disguises have been a default first step in almost every image model.

!!! c-info 💡
<b>Why it stuck:</b> augmentation is nearly free — it reuses data you already have, adds no new labeling cost, and needs no new photos. "Almost free, and it clearly helps" is exactly the kind of trick that becomes standard practice. That's why you'll find flip + crop + normalize in essentially every vision training recipe today.
!!!

**You've now seen where the trick was born** and why it became a default. But — like any tool — a disguise can be *misused*. The next concept turns the classic ways augmentation goes wrong into friendly puzzles, each with a clean fix.

@@@ concept id=c8 tag="Gotchas & limits" title="Where it trips you up: a disguise that lies, and what disguises can't do" gotit="Got the gotchas"
Augmentation is wonderful, but its whole power rests on one promise: *the disguise keeps the label true.* Break that promise, or ask the disguise to do more than it can, and things go wrong. Good news — each is a well-understood puzzle with a clean fix. Meet them now so none ever surprises you.

Think of dressing up for a costume party. A hat and a scarf? Still clearly *you* — a safe disguise. But turn your whole outfit *upside-down* and paint on a totally different face — now people honestly can't tell who you are, or worse, mistake you for someone else. A disguise that goes too far stops being a disguise and becomes a *lie*. The most famous example lives in handwritten digits: gently tilt a **6** and it's still a 6 — but turn it a full half-turn (a **180° rotation**, so the whole digit spins upside-down) and it becomes a **9**. Same pixels, opposite label. If you told the model "this rotated 6 is still a 6," you'd be *teaching it wrong on purpose*.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A label-destroying augmentation example with the digits 6 and 9. On the left, labelled safe, a 6 tilted slightly is still clearly a 6, labelled correctly. On the right, labelled label destroyed, a 6 rotated 180 degrees looks exactly like a 9, but is still labelled 6, marked with a red cross. A caption says a disguise that changes the true class is a lie, keep transforms label-preserving.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">A disguise too strong LIES: a 180°-rotated 6 is really a 9</text>
<g transform="translate(30,32)"><rect x="0" y="0" width="205" height="112" fill="#EAF5EE" stroke="#2D8B55"/><text x="102" y="18" fill="#276b45">safe: small tilt</text><text x="70" y="80" fill="#2D8B55" font-size="40" transform="rotate(10 70 66)">6</text><text x="140" y="66" fill="#276b45" font-size="8">still a 6</text><text x="140" y="80" fill="#276b45" font-size="8">label "6" ✓</text></g>
<text x="255" y="90" fill="#B8AEA2" font-size="13">vs</text>
<g transform="translate(285,32)"><rect x="0" y="0" width="205" height="112" fill="#FDECEC" stroke="#C93B3B"/><text x="102" y="18" fill="#8a3b3b">too strong: 180° rotation</text><text x="70" y="90" fill="#C93B3B" font-size="40" transform="rotate(180 70 66)">6</text><text x="150" y="60" fill="#8a3b3b" font-size="8">now looks like a 9</text><text x="150" y="74" fill="#8a3b3b" font-size="8">but labelled "6" ✗</text><text x="150" y="88" fill="#8a3b3b" font-size="8">teaches the wrong answer</text></g></g></svg>
%%%

**What the costume picture gets right:** a light disguise keeps your identity while a wild one destroys it — the exact line between a safe transform and a label-destroying one. **Where it breaks down:** at a party *people* judge when a costume went too far; with augmentation *you* must decide up front, per dataset, because the model has no way to notice its labels have been corrupted.

To *see* the lie happen, watch a tiny bitmap of a **6** get spun a half-turn. Predict what the pixel grid looks like after a 180° rotation, then run it:

%%% demo id=sixnine label="rotate a 6 by 180° and watch it become a 9"
code: six = np.array([[0,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,1],[1,1,1,1]]); np.rot90(six, 2)
out: six (loop at bottom)     rot90(six, 2) → the "9" (loop at top)
 .###      ####
 #...      #..#
 ####      ####
 #..#      ...#
 ####      ###.
take: <b>A 180° rotation genuinely maps a 6 → a 9 — the loop moves from bottom to top.</b> The pixel grid really changed (<code>np.array_equal(six, np.rot90(six, 2))</code> is <code>False</code>), so the picture now shows a different class. If you kept the label "6", you'd be teaching the model a lie. (Note: a plain vertical flip, <code>np.flipud</code>, does <i>not</i> give a clean 9 — only the half-turn does.)
%%%

#### The gotchas, each with its fix
- **Gotcha 1 · The disguise destroys the label** (cause: a transform too strong changes the true class — a 180° rotation of a 6, a huge tilt, or a crop that cuts the object out of frame). **Fix:** only use *label-preserving* transforms, and *tune the strength* — small tilts, gentle jitter, crops that keep the object in view. When unsure, look at a few augmented photos yourself.
- **Gotcha 2 · You augment the test set** (cause: leaving random transforms on at eval time makes predictions noisy and non-reproducible — from concept 6). **Fix:** train-only disguises; a fixed center-crop + normalize at test.
- **Gotcha 3 · Normalization mismatch** (cause: forgetting to reuse the same fixed average/spread at inference — from concept 5). **Fix:** ship the normalization numbers with the model and reuse the identical ones.

!!! c-warn ⚠️
<b>The quiet trap:</b> a too-strong or mismatched augmentation almost never throws an error — the code runs, the pictures look fine. It shows up only as *mysteriously worse* accuracy. When results disappoint, *look at your augmented images with your own eyes* first — you'll often spot the lying disguise instantly.
!!!

#### The honest limit: what a disguise can't do
Here's the plain truth worth stating outright. **Augmentation only recombines what you already have — it cannot invent genuinely new information.** Flips and crops of your cat photos give the model many *views* of those cats, but never a *new breed* it has never seen. So:

- It **can't rescue a dataset that's fundamentally too small or too narrow.** If you have ten cat photos, a thousand disguises of them still carry only ten cats' worth of real information. Sometimes you truly do need to *collect more, more varied* data.
- It **can't by itself cure class imbalance.** If you have 1,000 dog photos and 10 cat photos, disguising the 10 cats helps a little but doesn't manufacture the missing variety of real cats.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A capability limit picture. On the left, ten real cat photos feed a disguise box that outputs many disguised copies, but a note says still only ten cats' worth of real information. On the right, a note says augmentation cannot invent a new breed, fix a too-small dataset, or by itself cure class imbalance. A caption says disguises recombine what exists, they don't create new facts.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">A disguise recombines what's there — it can't create new facts</text>
<g transform="translate(30,34)"><rect x="0" y="0" width="70" height="100" fill="#FDF9EF" stroke="#C99A12"/><text x="35" y="18" fill="#8A6D3B" font-size="8">10 real</text><text x="35" y="30" fill="#8A6D3B" font-size="8">cats</text><g fill="#C99A12"><circle cx="20" cy="50" r="5"/><circle cx="50" cy="50" r="5"/><circle cx="35" cy="70" r="5"/><circle cx="20" cy="88" r="5"/><circle cx="50" cy="88" r="5"/></g></g>
<text x="118" y="88" fill="#B8AEA2" font-size="13">→</text>
<g transform="translate(140,44)"><rect x="0" y="20" width="90" height="44" fill="#EAF0FA" stroke="#5E5191"/><text x="45" y="46" fill="#5E5191" font-size="8">disguise ×100</text></g>
<text x="248" y="88" fill="#B8AEA2" font-size="13">→</text>
<g transform="translate(272,34)"><rect x="0" y="0" width="220" height="100" fill="#FDECEC" stroke="#C93B3B"/><text x="110" y="20" fill="#8a3b3b" font-size="8">many copies — but still only</text><text x="110" y="34" fill="#8a3b3b" font-size="8">10 cats' worth of real info</text><text x="110" y="56" fill="#8a3b3b" font-size="8">✗ can't invent a new breed</text><text x="110" y="72" fill="#8a3b3b" font-size="8">✗ can't fix a too-small dataset</text><text x="110" y="88" fill="#8a3b3b" font-size="8">✗ can't alone cure imbalance</text></g></g></svg>
%%%

#### A peek at the richer disguise kit ahead
The disguises today are the classic core. There's a whole landscape of cleverer ones you'll meet later: [[Cutout||A disguise that masks out a random square of the image, forcing the model not to rely on one spot. A later-lesson topic.]] (mask out a random square), [[Mixup||A disguise that blends two images and their labels together. A later-lesson topic.]] (blend two images and their labels), [[CutMix||A disguise that pastes a patch of one image into another and mixes their labels. A later-lesson topic.]] (paste a patch of one image into another), and *learned* policies like [[RandAugment||An automatic policy that picks and combines augmentations for you, instead of you tuning each by hand. A later-lesson topic.]] that pick the disguises for you. There's even [[test-time augmentation||Averaging the model's predictions over several disguised copies of a test image to squeeze out a little extra accuracy. A later-lesson topic.]] — the one time you *do* augment at test, on purpose, by averaging over disguises. All of that builds on today's foundation.

**You now hold every gotcha, its fix, and the honest limit** — plus a preview of the richer kit ahead. Nothing here is scary once named. Let's gather the whole day onto one page.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
You built a whole disguise kit today. Before we file the day away, let's lay every piece out on one page.

Think of the last day of a **school field trip**, when you sit down with your **scrapbook**. All day you collected little things — a ticket stub, a leaf, a photo, a postcard. Now you lay them out on **one page**, in order, so a single glance replays the whole trip. You're not *doing* anything new; you're *gathering* what you already have so it fits in one look. This recap is your scrapbook page for the disguise kit: every idea you met, side by side, in order.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A scrapbook-page analogy for the recap. A single page holds several taped-on keepsakes laid out in order: overfit, disguise, geometry, color, and train-only, each a small labelled card at a slight angle. A caption says one page gathers the whole day, each keepsake a hook back to a moment.">
<g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">A recap is a scrapbook page: the whole day gathered in one glance</text>
<rect x="30" y="30" width="460" height="120" fill="#FDF9EF" stroke="#C99A12" stroke-width="1.5" rx="4"/>
<g transform="translate(52,52) rotate(-6)"><rect x="0" y="0" width="70" height="50" fill="#FDECEC" stroke="#C93B3B"/><text x="35" y="29" fill="#8a3b3b">overfit</text></g>
<g transform="translate(150,58) rotate(4)"><rect x="0" y="0" width="70" height="50" fill="#FDF2E0" stroke="#C99A12"/><text x="35" y="29" fill="#8A6D3B">disguise</text></g>
<g transform="translate(248,50) rotate(-3)"><rect x="0" y="0" width="70" height="50" fill="#EAF5EE" stroke="#2D8B55"/><text x="35" y="29" fill="#276b45">geometry</text></g>
<g transform="translate(340,58) rotate(5)"><rect x="0" y="0" width="70" height="50" fill="#EAF0FA" stroke="#5E5191"/><text x="35" y="29" fill="#5E5191">color</text></g>
<g transform="translate(420,50) rotate(-4)"><rect x="0" y="0" width="70" height="50" fill="#FCF3DC" stroke="#C99A12"/><text x="35" y="24" fill="#8A6D3B" font-size="8">train-only</text><text x="35" y="38" fill="#8A6D3B" font-size="8">+ normalize</text></g>
<text x="260" y="168" fill="#6B645E">laid out in order — each keepsake a hook back to a moment</text></g></svg>
%%%

**What the scrapbook picture gets right:** nothing new is created — you gather what you already collected into one ordered page. **Where it breaks down:** scrapbook keepsakes just sit there, but these ideas *connect* — the overfit problem calls for disguises, which must preserve the label, applied to training only, atop an always-on normalize — so this "page" is really a working recipe.

Here's the whole pipeline as one connected picture — one photo in, disguised for training, plain for testing:

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A one-page summary of the augmentation pipeline. One photo enters and splits into two paths. The training path applies random flip, crop, tilt, and color jitter, then normalize, producing endless variety. The test path applies only a fixed center-crop then normalize, producing one plain repeatable view. A note says disguises are label-preserving and train-only, normalize uses the same fixed numbers on both paths.">
<g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#2C2A28" font-size="12">The pipeline: disguise for training, keep it plain for testing</text>
<g transform="translate(20,66)"><rect x="0" y="0" width="70" height="40" fill="#FDF9EF" stroke="#C99A12"/><text x="35" y="18" fill="#8A6D3B">one photo</text><text x="35" y="32" fill="#8A6D3B">label "cat"</text></g>
<line x1="90" y1="86" x2="118" y2="56" stroke="#B8AEA2"/><line x1="90" y1="86" x2="118" y2="120" stroke="#B8AEA2"/>
<g transform="translate(122,34)"><rect x="0" y="0" width="180" height="44" fill="#EAF5EE" stroke="#2D8B55"/><text x="90" y="18" fill="#276b45">TRAIN: random flip/crop/tilt/jitter</text><text x="90" y="34" fill="#276b45">then normalize (fixed #s)</text></g>
<text x="312" y="60" fill="#B8AEA2">→</text>
<g transform="translate(328,34)"><rect x="0" y="0" width="168" height="44" fill="#EAF5EE" stroke="#2D8B55"/><text x="84" y="26" fill="#276b45">endless variety → less overfit</text></g>
<g transform="translate(122,102)"><rect x="0" y="0" width="180" height="44" fill="#EAF0FA" stroke="#5E5191"/><text x="90" y="18" fill="#5E5191">TEST: fixed center-crop</text><text x="90" y="34" fill="#5E5191">then normalize (SAME fixed #s)</text></g>
<text x="312" y="128" fill="#B8AEA2">→</text>
<g transform="translate(328,102)"><rect x="0" y="0" width="168" height="44" fill="#EAF0FA" stroke="#5E5191"/><text x="84" y="26" fill="#5E5191">one plain, repeatable view</text></g></g></svg>
%%%

The day in a few beats:
- **The trap.** With too few photos a big network *overfits* — it memorizes the exact pictures and fails on new ones. Named teammates that also fight this: dropout and weight decay.
- **The big idea.** Show each photo in many **disguises**; every disguise must be *label-preserving* (a flipped cat is still a cat), which teaches *invariance* to nuisance factors like position, direction, and lighting.
- **Geometry disguises.** Flip, random crop (and random resized crop), rotation, zoom — move the pixels around.
- **Color disguises.** Color jitter — random nudges to brightness, contrast, saturation, hue.
- **Always normalize.** Put every photo on one standard scale with *fixed* per-color numbers — the *same* numbers at train and test.
- **The iron rule of when.** Disguise the *training* photos only; at test, use a plain, deterministic center-crop + normalize.
- **Where it began.** AlexNet (2012) popularized cheap on-the-fly crops + flips as the trick that let a hungry network generalize.
- **Gotchas.** Label-destroying disguise — e.g. a 180° rotation turns a 6 into a 9 (keep transforms safe, tune strength); augmenting the test set (train-only); normalization mismatch (reuse the same numbers).
- **The limit.** Disguises only recombine what you already have — they can't invent new data, fix a too-small dataset, or by themselves cure class imbalance.

#### Cheat-sheet · the disguise kit at a glance
%%% table
:: Disguise :: What it does :: Teaches the model to ignore
Horizontal flip :: mirror left↔right :: which way the object faces
Random (resized) crop :: cut out a random sub-region, maybe zoom :: where the object sits / how big it is
Rotation :: tilt a small angle :: a slightly crooked camera
Scale / zoom :: shrink or enlarge :: near vs. far
Color jitter :: nudge brightness/contrast/saturation/hue :: lighting and camera color
Normalize (not a disguise) :: fixed per-color (x−avg)/spread, always on :: — (train & test, same numbers)
%%%

#### Cheat-sheet · the words you met today
%%% jargon
data augmentation | showing the model the same photo in many disguises to add variety
overfitting | memorizing the training photos so you fail on new ones
label-preserving | a disguise that changes the look but not the correct answer
invariance | giving the same answer even when an unimportant thing changes
nuisance factors | position, direction, lighting — things the label should NOT depend on
color jitter | random nudges to brightness, contrast, saturation, hue
normalization | put every photo on one fixed scale (same numbers train & test)
train-only rule | disguise training photos only; test gets a plain center-crop + normalize
dropout / weight decay | teammate regularizers that also fight overfitting
%%%

That's the whole day. You can now explain what augmentation does, why it fights overfitting, the everyday disguises, the always-on normalize step, the one rule about when to disguise, and the ways it can trip you — plus its honest limit. Next you'll reuse a model someone else already trained, with **day-06, Transfer Learning & Embeddings**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is data augmentation, and what problem does it mainly fight? | a:2 | collecting brand-new photos to grow the dataset | making the model bigger so it learns faster | applying label-preserving disguises (flip, crop, tilt, recolor) to the photos you already have, to add variety and fight overfitting | deleting hard photos from the training set | fb: Augmentation reuses the photos you have, showing each in many disguises. That extra variety fights overfitting — the model stops memorizing exact pictures and learns the real thing. It does NOT collect new photos or change the model's size.
q: You add a "180° rotation" disguise to a dataset of handwritten digits and accuracy gets worse. Why? | a:1 | 180° rotation is too slow to compute | rotating a 6 a full half-turn turns it into a 9, so the disguise destroys the label and teaches the wrong answer | 180° rotation uses too much memory | the model needs a bigger learning rate | fb: A disguise must be label-preserving. A 180° rotation spins a 6 upside-down into a 9 — the pixels now show a different class, but the label still says "6", so the model is taught wrong. Keep transforms label-preserving and tune their strength.
q: When should you apply the random disguises (flip, crop, jitter)? | a:2 | to every photo at train and test time | never — they only slow things down | to the training photos only; at test time use a fixed center-crop + normalize | only to the test photos, to make the score harder | fb: Disguise training photos only. At test time, random transforms make the same photo get different, non-reproducible predictions — so you use a plain, deterministic pipeline (fixed center-crop + normalize) for a fair, stable score.
q: A friend says "I'll just use augmentation instead of collecting more data — it makes infinite new examples." What's the honest limit? | a:1 | augmentation only recombines the information you already have; it can't invent genuinely new data, fix a fundamentally too-small dataset, or by itself cure class imbalance | augmentation always beats collecting real data | augmentation replaces the need for a validation set | augmentation removes the need to normalize | fb: Disguises give many VIEWS of your existing photos, not new facts. Ten cats disguised a thousand ways still carry ten cats' worth of real information — so augmentation can't rescue a too-small or imbalanced dataset on its own; sometimes you truly need more, more varied real data.
%%%

@@@ produce id=produce tag="Produce" title="Disguise one image, then watch the label survive" gotit="Done"
Time to run the disguise kit yourself and *watch* it work. You'll take one small image, apply a few disguises, and **print the shape and a summary of the pixels before and after each one** so you can see the picture change while the label stays put. Then you'll show the two failure puzzles from today with your own eyes. **Predict first:** if you horizontally flip a 4×4 grid of numbers, which column ends up where? And if you normalize by subtracting the average and dividing by the spread, what should the new average be? Then run it and **observe** the flip mirrors the columns, the normalized average lands near `0`, and a too-strong transform (a 180° rotation of a 6) visibly changes what the picture "is." Pick one path.

#### Option A · write it yourself
Create `sessions/m06-cnns-vision-encoders/day-05-data-augmentation/experiment.py`. Build a tiny disguise kit on a small fake image and **print values at every step**. Start with a fake 4×4 grayscale image, `img = np.arange(16).reshape(4, 4).astype(float)`, and **print** it plus its shape. (1) **Horizontal flip:** `flipped = img[:, ::-1]` — **print** it and **notice** the columns are mirrored while the set of values is unchanged (same "label"). (2) **Normalize:** `norm = (img - img.mean()) / img.std()` — **print** `norm.mean()` (about `0`) and `norm.std()` (about `1`), and **notice** this is the always-on standardizing step. (3) **Train-only demo:** write a `pipeline(img, train=True)` that applies a *random* flip only when `train=True`, then always normalizes; call it twice with `train=True` (**observe** the two outputs can differ) and twice with `train=False` (**observe** they are identical — the plain, repeatable test view). (4) **Label-destroying demo:** make a recognizable little `six = np.array([[0,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,1],[1,1,1,1]])` (its loop sits at the bottom), then apply a **180° rotation** with `nine = np.rot90(six, 2)` — **print** both grids (the loop moves to the top, so it now reads as a `9`), check `np.array_equal(six, nine)` is `False`, and note that if you kept the label "6" you'd be teaching the model wrong. Run with `python3 sessions/m06-cnns-vision-encoders/day-05-data-augmentation/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 6 Day 5 artifact.

Create sessions/m06-cnns-vision-encoders/day-05-data-augmentation/experiment.py that demonstrates data augmentation on a tiny fake image, with a comment on each step and printing values at every step:
1. Makes a fake 4x4 grayscale image img = np.arange(16).reshape(4,4).astype(float) and prints it and its shape.
2. Horizontal flip: flipped = img[:, ::-1]; print it and point out the columns are mirrored but the set of pixel values (the "label") is unchanged — a label-preserving disguise.
3. Normalize: norm = (img - img.mean()) / img.std(); print norm.mean() (~0) and norm.std() (~1); assert np.isclose(norm.mean(), 0, atol=1e-6). Point out this is the always-on standardizing step, applied at train AND test with the SAME numbers.
4. Train-only rule: define pipeline(img, rng, train): if train, apply a random horizontal flip (rng.random() < 0.5), then always normalize; return it. Call it twice with train=True (show the two results can differ because of randomness) and twice with train=False (show the results are identical — a plain, deterministic test view). Print a note that random disguises are TRAINING ONLY.
5. Label-destroying gotcha: make a recognizable "6" bitmap six = np.array([[0,1,1,1],[1,0,0,0],[1,1,1,1],[1,0,0,1],[1,1,1,1]]) (its loop is at the BOTTOM); apply a 180-degree rotation nine = np.rot90(six, 2) (equivalently np.flipud(np.fliplr(six))) so the loop moves to the TOP and it now reads as a "9"; print both grids, assert np.array_equal(six, nine) is False, and warn that keeping the label "6" would teach the model wrong; the fix is to only use label-preserving transforms. (Note in a comment: a plain vertical flip np.flipud alone does NOT produce a clean 9 — only the 180-degree rotation does.)
6. Prints a one-line takeaway: "augmentation = label-preserving disguises applied to TRAINING photos only; normalize every photo with the same fixed numbers; a too-strong disguise (e.g. a 180-degree rotation turning a 6 into a 9) destroys the label."
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (predict, then confirm)
- The horizontal flip mirrors the columns of the image, but the collection of pixel values is unchanged — the picture looks different, yet it's still the same "class."
- After normalizing, the new average is about `0.0` and the new spread about `1.0` — every photo lands on the same standard scale.
- With `train=True` the pipeline can give two *different* outputs (a fresh random disguise each call); with `train=False` it gives the *same* output every time — that's the train-only rule in action.
- The 180° rotation of the crude "6" moves the loop from the bottom to the top, so it now reads as a "9" (and `np.array_equal(six, nine)` is `False`) — proof that a too-strong disguise destroys the label, so you must keep transforms label-preserving.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m06-cnns-vision-encoders/day-05-data-augmentation/log.md`: (1) what augmentation is and the one problem it mainly fights; (2) the one rule every disguise must obey, and one disguise that breaks it; (3) why you disguise training photos only, and what the test-time pipeline looks like instead.
!!!

@@@ fin
