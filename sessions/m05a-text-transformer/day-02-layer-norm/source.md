---
quest_id: wf6-d02-layernorm
mode: concept
donor: v9-base.donor
page_title: "Module 5a · Day 2 — Layer Normalization"
module_label: "Module 05a · Represent · Day 2"
title: "Layer Normalization"
subtitle: "Keeping Every Layer on an Even Keel"
brand_sub: "Foundations · M5a Day 2"
spine: "level"
nav_prev_href: "../day-01-residual-connections/lesson.html"
nav_prev_label: "Residual Connections"
nav_next_href: "../day-03-feed-forward/lesson.html"
nav_next_label: "The Feed-Forward Sublayer"
fin_title: "Module 5a · Day 2 complete! 🏆"
fin_body: "Nice work — you've unlocked <b>Layer Normalization</b>: the step that re-<b>level</b>s every token's numbers to a steady scale (mean 0, spread 1) before the next sublayer reads them, so nothing blows up or fades as the residual stream grows deeper. It's the quiet stabilizer tucked inside every transformer block.<br>Next up: <b>The Feed-Forward Sublayer</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: LayerNorm normalizes per token across features to mean 0 spread 1, keywords: [layer normalization, mean 0, unit variance, per token, across the features, re-level]}
  - {topic: independent of batch size same at train and inference, keywords: [batch size, at train and at inference, one token at a time, does not depend on the batch]}
  - {topic: learnable gain gamma and bias beta rescale or undo, keywords: [gain, gamma, bias, beta, learnable, can undo the normalization, volume knob]}
  - {topic: where LayerNorm sits wraps attention and feed-forward, keywords: [inside every transformer block, wraps the attention, feed-forward, before the sublayer]}
  - {topic: BatchNorm ancestor normalizes across the batch and why it breaks, keywords: [batch normalization, across the batch, variable-length, small batch, motivated normalizing over features]}
  - {topic: internal covariate shift drifting activation scale remedy is LayerNorm, keywords: [drift, shifting scale, internal covariate shift, each layer, remedy: layer norm]}
  - {topic: unstable training exploding vanishing signal remedy residual plus LayerNorm and Pre-LN, keywords: [exploding, vanishing, unstable, compound, residual connection, pre-ln]}
  - {topic: Pre-LN vs Post-LN placement affects stability, keywords: [pre-ln, post-ln, before the sublayer, after the residual, placement, more stable]}
  - {topic: epsilon prevents division by zero when variance near zero, keywords: [epsilon, division by zero, variance is near zero, inside the square root, tiny number]}
  - {topic: RMSNorm drops mean-centering rescales by root mean square, keywords: [rmsnorm, root mean square, drops the mean, no re-centering, simpler, cheaper]}
  - {topic: capability limit only stabilizes does not add power or mix tokens, keywords: [does not add power, cannot fix a bad architecture, per token, does not mix information across tokens]}
---

@@@ hero
@lede Imagine your teacher hands back a test and, instead of raw marks, gives everyone a **re-scaled** score so the class always lands around the same steady middle — nobody's number runs off to a thousand, nobody's shrinks to almost nothing. That "everybody back to a sensible scale" move is today's whole idea: **layer normalization**. Yesterday you built the residual stream — a signal that flows straight up a transformer while every block *adds* to it. You even spotted the danger at the end: keep adding, and the stream could balloon out of control. Today you meet the tidy fix that keeps it on an even keel, the same quiet step tucked inside every layer of ChatGPT. Sounds like just "divide by something"? Good — hold that; by the end you'll see why this little re-**level** is what lets a transformer be dozens of layers deep without falling apart.
@goal Together we'll meet the fix one step at a time — first the puzzle (a stream whose numbers drift and balloon as blocks pile on), then the move that re-**level**s each token to a steady scale, why we do it per token and not across the batch, the two little knobs that let the network take back control, the tiny safety number that stops a divide-by-zero, where the step sits in a block and why *before* beats *after*, a leaner cousin called RMSNorm, and finally the honest limit — what this step can and can't do. Every failure we meet is a small puzzle with a neat answer, and every formula gets said out loud in plain words first.

@@@ concept id=c1 tag="The puzzle" title="A stream that drifts off scale" gotit="Got the puzzle"
Let's pick up exactly where yesterday left off. You learned that a transformer is a tall stack of blocks, and each block *adds* its change onto a running signal — the residual stream. That "keep adding" trick is what lets the model be deep. But it hides a sneaky problem, and spotting it is today's starting point.

Think of a **backpack you keep tossing books into as you walk to school**. At the first stop a friend hands you one book — light, fine. At the next stop, another. And another. Each book on its own is small, but nobody ever takes a book *out*. By the tenth stop the backpack is so heavy it's crushing your shoulders and you can barely walk. The residual stream is like that backpack: every block adds its change, nothing gets removed, so the numbers inside can grow bigger and bigger the deeper you go.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A person walking left to right with a backpack. At each stop a book is added and the backpack grows heavier and larger, until the last one is huge and the walker is bent over under the weight."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Backpack down the block: books added, none removed → too heavy</text><line x1="30" y1="158" x2="490" y2="158" stroke="#D8D2C8" stroke-width="3"/><rect x="52" y="120" width="26" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="65" y="112" text-anchor="middle" fill="#276b45" font-size="9">stop 1</text><text x="65" y="176" text-anchor="middle" fill="#6B645E" font-size="9">light</text><rect x="158" y="108" width="34" height="42" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="175" y="100" text-anchor="middle" fill="#9A7A10" font-size="9">stop 4</text><rect x="268" y="88" width="46" height="62" rx="4" fill="#FBE6C9" stroke="#C99A12"/><text x="291" y="80" text-anchor="middle" fill="#9A7A10" font-size="9">stop 7</text><rect x="388" y="52" width="70" height="98" rx="5" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="423" y="44" text-anchor="middle" fill="#C93B3B" font-size="9">stop 10</text><text x="423" y="176" text-anchor="middle" fill="#C93B3B" font-size="9">crushing!</text><text x="118" y="140" fill="#B8AEA2">→</text><text x="228" y="140" fill="#B8AEA2">→</text><text x="348" y="140" fill="#B8AEA2">→</text></g></svg>
%%%

**What the backpack picture gets right:** many small additions with nothing taken out can pile into something far too big to carry — that's the stream's numbers drifting off scale. **Where it breaks down:** a backpack only gets *heavier*, but a stream's numbers can also drift the other way and *shrink* toward nothing, or lean lopsided (mostly big positives, or mostly negatives) — the trouble isn't only "too big," it's "drifted away from a steady middle."

#### Why a drifting scale hurts the *next* block
Here's the part that makes this a real problem, not just a curiosity. Each block was trained expecting its input to arrive at a *sensible* size. When the numbers coming in keep changing scale — bigger here, tinier there — every block is chasing a moving target. Engineers gave this a name: [[internal covariate shift||The input distribution to each layer keeps shifting during training as earlier layers change, so every layer is forced to chase a moving target.]] — the fancy way of saying "the scale the next block sees keeps sliding around." When the scale slides too far, training gets shaky: numbers can [[explode||Grow so large the computer can no longer represent them, turning values into "infinity" and breaking training.]] toward huge values or [[vanish||Shrink so close to zero that the signal effectively disappears and the layer stops learning.]] toward zero, and the model stops learning cleanly.

Watch the drift build up. Below, the *typical size* of the stream's numbers is plotted block by block — with nothing to hold it steady, it wanders far from the sensible middle:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A line chart. A dashed horizontal line marks the sensible middle scale. A red line labeled no normalization starts near the middle then climbs steeply upward across blocks, drifting far above the sensible scale."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#C93B3B" font-weight="bold">No re-leveling — the stream's scale drifts away, block by block</text><line x1="60" y1="40" x2="60" y2="160" stroke="#B8AEA2"/><line x1="60" y1="160" x2="490" y2="160" stroke="#B8AEA2"/><text x="30" y="46" fill="#6B645E" font-size="9">big</text><text x="30" y="164" fill="#6B645E" font-size="9">0</text><line x1="60" y1="128" x2="490" y2="128" stroke="#2D8B55" stroke-width="1.5" stroke-dasharray="5,4"/><text x="470" y="122" fill="#276b45" font-size="9" text-anchor="end">sensible middle</text><polyline points="60,126 130,120 200,104 270,82 340,58 410,44 480,40" fill="none" stroke="#C93B3B" stroke-width="2.5"/><circle cx="60" cy="126" r="3" fill="#C93B3B"/><circle cx="200" cy="104" r="3" fill="#C93B3B"/><circle cx="340" cy="58" r="3" fill="#C93B3B"/><circle cx="480" cy="40" r="3" fill="#C93B3B"/><text x="275" y="185" fill="#6B645E" font-size="10">block 1 → deeper → block N</text><text x="450" y="58" fill="#C93B3B" font-size="9" text-anchor="end">off the chart!</text></g></svg>
%%%

If that feels worrying — "wait, the residual stream we just loved has a leak?" — you're exactly where you should be. It's a genuine puzzle, and the fix, coming next, is beautifully simple. We're going to grab that drifting stream and gently pull it back to a steady scale at every block.

@@@ concept id=c2 tag="The fix" title="Re-level to a steady scale" gotit="Got the fix"
So how do we stop the stream from drifting off scale? The move is delightfully simple: **before the next block reads the numbers, grab them and re-level them to a steady scale** — pull them back to a familiar middle, every single time.

Think of **standardizing test scores in a class**. One teacher marks out of 100, another out of 20, another gives wild bonus points. Raw, the numbers aren't comparable — one kid has 95, another has 18, and it's chaos. So the school does a tidy trick: for each set of scores, it slides them so the *average* sits at zero, and stretches or squeezes them so the *spread* is always the same. Now every set of scores speaks the same language, no matter how the teacher marked. Layer normalization does exactly this to a token's numbers: it slides them so their average is **0**, and rescales them so their spread is always **1**.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two rows of dots on a number line. Top row before: dots scattered far to the right around a high average, wide spread, drifted off center. Bottom row after: the same dots re-centered around zero with a tidy even spread."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Re-leveling scores: slide the average to 0, set the spread to 1</text><text x="60" y="52" fill="#C93B3B" text-anchor="start" font-weight="bold">before · drifted, uneven</text><line x1="60" y1="78" x2="490" y2="78" stroke="#B8AEA2"/><line x1="120" y1="72" x2="120" y2="84" stroke="#B8AEA2"/><text x="120" y="98" fill="#6B645E" font-size="9">0</text><circle cx="300" cy="78" r="5" fill="#C93B3B"/><circle cx="360" cy="78" r="5" fill="#C93B3B"/><circle cx="410" cy="78" r="5" fill="#C93B3B"/><circle cx="470" cy="78" r="5" fill="#C93B3B"/><path d="M300 62 L470 62" stroke="#C93B3B" stroke-width="1.2" stroke-dasharray="3,2"/><text x="385" y="56" fill="#C93B3B" font-size="9">average way off to the right</text><line x1="60" y1="120" x2="490" y2="120" stroke="#E5DFD6" stroke-dasharray="4,3"/><text x="60" y="150" fill="#276b45" text-anchor="start" font-weight="bold">after · centered on 0, spread = 1</text><line x1="60" y1="176" x2="490" y2="176" stroke="#B8AEA2"/><line x1="275" y1="170" x2="275" y2="182" stroke="#2D8B55" stroke-width="1.5"/><text x="275" y="196" fill="#276b45" font-size="9">0</text><circle cx="215" cy="176" r="5" fill="#2D8B55"/><circle cx="255" cy="176" r="5" fill="#2D8B55"/><circle cx="295" cy="176" r="5" fill="#2D8B55"/><circle cx="335" cy="176" r="5" fill="#2D8B55"/><text x="385" y="180" fill="#276b45" font-size="9" text-anchor="start">tidy, even, centered</text></g></svg>
%%%

**What the test-score picture gets right:** re-centering to zero and fixing the spread makes wildly different sets comparable and calm — that's exactly the re-level. **Where it breaks down:** a school standardizes *one score per student across the class*, but layer norm re-levels the *whole list of numbers inside a single token* — same math, different thing being lined up (you'll pin down "which numbers" in the very next concept).

#### The move in three plain steps
Let's name the pieces gently. A token arrives as a list of numbers — call the list `x`. Re-leveling is three tiny steps:

1. **Find the average** of the list. Call it the [[mean||The average of a list of numbers: add them all up, divide by how many there are.]] (written `μ`, the Greek letter "mu"). This is the middle the numbers are drifting around.
2. **Find the spread** — how far the numbers typically sit from that average. Call it the [[standard deviation||A measure of spread: roughly the typical distance of the numbers from their average. Small = numbers bunched together, large = numbers spread out.]] (written `σ`, "sigma").
3. **Slide and squeeze:** subtract the average from every number (now the middle is 0), then divide by the spread (now the spread is 1).

%%% formula
expr: normalized = (x − μ) / σ
note: subtract the mean μ to center on 0, then divide by the spread σ so the spread becomes 1
%%%

Say it out loud: "take each number, step it back by the average, then shrink it by how spread out they were." That's the entire re-level. Let's watch it happen on real numbers — predict first: if the input is `[8, 10, 12]`, the average is `10`, so after centering the middle number becomes `0`. Then run it.

%%% demo id=leveldemo label="run it"
code: x = [8, 10, 12];  mu = 10;  sigma = 1.63;  norm = [round((v - mu)/sigma, 2) for v in x]
out: mu = 10.0   sigma = 1.63   norm = [-1.23, 0.0, 1.23]
take: <b>The average slid to 0 and the spread became 1.</b> The middle value (10) landed exactly on 0, and the two others sit one "spread" away on each side. Same shape of numbers, calm steady scale — that's the re-level.
%%%

Now watch the transformation itself. Below, the same three numbers are drawn as bars before and after — drifted-and-spread on the left, re-centered-and-tidied on the right — in one before→after picture:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two bar groups. Left before: three tall bars at 8, 10, 12 all sitting high above a zero baseline. Right after: three bars at minus 1.23, 0, and plus 1.23, centered around a zero line, one below and one above."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Same token, before → after the re-level</text><text x="130" y="40" fill="#C93B3B" font-weight="bold">before: 8, 10, 12</text><line x1="40" y1="150" x2="230" y2="150" stroke="#B8AEA2"/><rect x="60" y="86" width="34" height="64" fill="#C93B3B" opacity="0.8"/><text x="77" y="80" fill="#6B645E" font-size="9">8</text><rect x="115" y="70" width="34" height="80" fill="#C93B3B" opacity="0.8"/><text x="132" y="64" fill="#6B645E" font-size="9">10</text><rect x="170" y="54" width="34" height="96" fill="#C93B3B" opacity="0.8"/><text x="187" y="48" fill="#6B645E" font-size="9">12</text><text x="135" y="168" fill="#C93B3B" font-size="9">all high, off-center</text><text x="270" y="8" fill="#B8AEA2">→</text><text x="390" y="40" fill="#276b45" font-weight="bold">after: −1.23, 0, +1.23</text><line x1="300" y1="110" x2="490" y2="110" stroke="#2D8B55" stroke-width="1.5"/><text x="296" y="114" fill="#276b45" font-size="9" text-anchor="end">0</text><rect x="320" y="110" width="34" height="40" fill="#2D8B55" opacity="0.8"/><text x="337" y="164" fill="#6B645E" font-size="9">−1.23</text><rect x="375" y="106" width="34" height="8" fill="#2D8B55" opacity="0.5"/><text x="392" y="100" fill="#6B645E" font-size="9">0</text><rect x="430" y="70" width="34" height="40" fill="#2D8B55" opacity="0.8"/><text x="447" y="64" fill="#6B645E" font-size="9">+1.23</text><text x="395" y="188" fill="#276b45" font-size="9">centered on 0, even spread</text></g></svg>
%%%

**You just met the core move.** Everything else today is refining this one idea: *which* numbers we level, two knobs to control it, a safety catch, where it sits, and a leaner cousin.

@@@ concept id=c3 tag="Per token" title="Level each token on its own" gotit="Got the axis"
There's one question the last concept left open: *which* numbers do we average together to re-level? This choice is the heart of why it's called **layer** normalization, and it's a lovely little idea.

Think of **grading each student on their own report card**. A fair teacher looks at *one student's* subjects — their maths, their reading, their art — and levels *those* against each other. They do **not** mix your maths mark into someone else's report card to decide your grade. Each report card is judged on its own. Layer norm does the same: it takes *one token* — one word's own list of numbers — and levels those numbers against each other, all on their own. The token next to it in the sentence, and every token in every other example, is levelled completely separately.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A grid of report cards. Layer norm draws a green box around one single student's own row of subject marks, showing it is levelled on its own. Batch norm draws a red box down a whole column across many students, showing it mixes across different students."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Grade each report card on its OWN row (that's Layer Norm)</text><text x="90" y="44" fill="#6B645E" font-size="9">maths</text><text x="160" y="44" fill="#6B645E" font-size="9">reading</text><text x="230" y="44" fill="#6B645E" font-size="9">art</text><text x="35" y="70" fill="#6B645E" font-size="9" text-anchor="start">token A</text><text x="35" y="102" fill="#6B645E" font-size="9" text-anchor="start">token B</text><text x="35" y="134" fill="#6B645E" font-size="9" text-anchor="start">token C</text><g fill="#2C2A28"><text x="90" y="70">7</text><text x="160" y="70">9</text><text x="230" y="70">5</text><text x="90" y="102">3</text><text x="160" y="102">8</text><text x="230" y="102">6</text><text x="90" y="134">9</text><text x="160" y="134">4</text><text x="230" y="134">7</text></g><rect x="66" y="56" width="188" height="24" rx="6" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text x="300" y="72" fill="#276b45" font-size="9" text-anchor="start">← LayerNorm: level one token's own row</text><rect x="74" y="52" width="34" height="94" rx="6" fill="none" stroke="#C93B3B" stroke-width="2" stroke-dasharray="4,3"/><text x="300" y="140" fill="#C93B3B" font-size="9" text-anchor="start">↑ BatchNorm levels a column across tokens</text><text x="260" y="196" fill="#6B645E" font-size="10">Layer Norm = across a token's own features · Batch Norm = across the batch</text></g></svg>
%%%

**What the report-card picture gets right:** judging each card on its own row means no student depends on who else happens to be in the class — that's layer norm's independence, exactly. **Where it breaks down:** report cards have a handful of subjects, but a token's list is hundreds or thousands of numbers wide — the "level your own row" idea is what carries over, not the number of columns.

#### The older cousin: BatchNorm, and why it stumbled
Layer norm didn't appear from nowhere. The older, cruder trick is [[Batch Normalization||An earlier normalization that levels each feature across a whole batch of examples, using the batch's average and spread. Standard in vision networks.]] — "BatchNorm" for short. BatchNorm levels the *other way*: it takes one feature (say "maths") and levels it *across the whole batch of students*, using the batch's average. That works fine for a fixed pile of images. But for a language model it stumbles, and the reasons are a neat little puzzle:

!!! c-warn ⚠️
<b>Why BatchNorm breaks for text (three reasons):</b> (1) <b>Sentences have different lengths</b> — one has 5 words, the next has 50 — so there's no clean "column across the batch" to average. (2) <b>At inference you often run one sentence at a time</b> — a batch of *one* has no other examples to average against, so the whole idea falls apart. (3) <b>Small or uneven batches</b> give a noisy, unreliable average. Layer norm sidesteps all three by never looking at the batch at all — it levels each token using only that token's own numbers.
!!!

That last point is the quiet superpower, so let's make it loud:

!!! c-ok ✅
<b>Batch-independent.</b> Because layer norm only ever uses one token's own numbers, it does the <b>exact same thing whether you train on a batch of 512 or serve one token at a time.</b> No "batch statistics" to store, no train-vs-serve mismatch. That reliability is a big reason transformers use layer norm, not batch norm.
!!!

**Victory lap:** you can now answer a classic interview opener — "why LayerNorm and not BatchNorm in a transformer?" — in one breath: text has variable lengths and single-example inference, so a per-token level is stable while a per-batch level is not.

@@@ concept id=c4 tag="Two knobs" title="Give the network its scale back" gotit="Got the knobs"
Here's a fair worry about everything so far. We just *forced* every token to have average 0 and spread 1. But what if the network actually *wanted* some numbers bigger, or centered somewhere else? Did we just throw away useful information by squashing everyone into the same mold? A great question — and layer norm has a lovely answer: two little knobs.

Think of the **volume and bass knobs on a speaker**. When a song comes in, it arrives at some fixed, standardized recording level — that's the re-leveled numbers, all tidy at spread 1. But you're not stuck with that level. You turn the **volume** knob to make it louder or softer, and the **tone** knob to shift where it sits. The knobs let *you* decide the final loudness — the standardized signal is just the clean starting point. Layer norm adds two knobs exactly like this, and crucially the network gets to *learn* how to set them.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A tidy standardized signal enters two knobs. A gain knob labeled gamma stretches or shrinks the spread, and a shift knob labeled beta slides it up or down, producing the final output the network wants."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Two learned knobs: stretch (γ) and slide (β) the tidy signal</text><rect x="30" y="80" width="90" height="40" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="75" y="98" fill="#276b45" font-size="9">tidy signal</text><text x="75" y="112" fill="#276b45" font-size="9">mean 0, spread 1</text><text x="132" y="104" fill="#B8AEA2" font-size="15">→</text><circle cx="205" cy="100" r="30" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><line x1="205" y1="100" x2="223" y2="82" stroke="#5E5191" stroke-width="2.5"/><text x="205" y="150" fill="#5E5191" font-size="9">gain γ (gamma)</text><text x="205" y="163" fill="#6B645E" font-size="9">stretch the spread</text><text x="252" y="104" fill="#B8AEA2" font-size="15">→</text><circle cx="325" cy="100" r="30" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><line x1="325" y1="100" x2="343" y2="118" stroke="#9A7A10" stroke-width="2.5"/><text x="325" y="150" fill="#9A7A10" font-size="9">bias β (beta)</text><text x="325" y="163" fill="#6B645E" font-size="9">slide the center</text><text x="372" y="104" fill="#B8AEA2" font-size="15">→</text><rect x="400" y="80" width="96" height="40" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="448" y="104" fill="#276b45" font-size="9">final output</text></g></svg>
%%%

**What the knobs picture gets right:** the signal is standardized *first*, then two controls let you set the final loudness and center — that's the re-level plus two learned adjustments, exactly. **Where it breaks down:** you turn a speaker's knobs by hand, but the network *learns* its knob settings during training — it discovers, on its own, how much to stretch and slide each number.

#### Naming the knobs
The stretch knob has a name: [[gain||A learnable number layer norm multiplies the standardized value by, to stretch or shrink its spread. Written γ (gamma). Starts at 1.]] — written `γ` (gamma) — it multiplies the standardized value to set its spread. The slide knob is the [[bias||A learnable number layer norm adds to the standardized value, to shift its center up or down. Written β (beta). Starts at 0.]] — written `β` (beta) — it gets added on to shift the center. So the full move, said in plain words, is: "level the numbers, then stretch by the gain and slide by the bias."

%%% formula
expr: output = γ · normalized + β
note: γ (gain) stretches the spread, β (bias) shifts the center — both are learned during training
%%%

Here's the delightful part that answers our worry. If the network decides the re-leveling *lost* something it needed, it can simply learn `γ` and `β` to **undo** it — dial the knobs back to exactly recover any scale it wants. So layer norm never *traps* the network; it hands it a clean, steady starting point *and* the freedom to depart from it. Watch the same tidy signal reshaped by three different knob settings:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three small bar charts of the same standardized signal reshaped by different knob settings. Left: gamma 1 beta 0, the neutral tidy bars. Middle: gamma 2 beta 0, the bars stretched taller. Right: gamma 1 beta 3, the same bars all shifted upward."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Same signal, three learned knob settings</text><text x="90" y="42" fill="#276b45" font-size="10">γ=1, β=0 (neutral)</text><line x1="40" y1="110" x2="150" y2="110" stroke="#2D8B55"/><rect x="55" y="110" width="18" height="30" fill="#2D8B55" opacity="0.8"/><rect x="85" y="104" width="18" height="6" fill="#2D8B55" opacity="0.5"/><rect x="115" y="80" width="18" height="30" fill="#2D8B55" opacity="0.8"/><text x="260" y="42" fill="#5E5191" font-size="10">γ=2, β=0 (louder)</text><line x1="205" y1="120" x2="315" y2="120" stroke="#7C6DAA"/><rect x="220" y="120" width="18" height="52" fill="#7C6DAA" opacity="0.8"/><rect x="250" y="112" width="18" height="8" fill="#7C6DAA" opacity="0.5"/><rect x="280" y="68" width="18" height="52" fill="#7C6DAA" opacity="0.8"/><text x="430" y="42" fill="#9A7A10" font-size="10">γ=1, β=3 (shifted up)</text><line x1="375" y1="140" x2="485" y2="140" stroke="#C99A12"/><rect x="390" y="80" width="18" height="30" fill="#C99A12" opacity="0.8"/><rect x="420" y="74" width="18" height="6" fill="#C99A12" opacity="0.5"/><rect x="450" y="50" width="18" height="30" fill="#C99A12" opacity="0.8"/><text x="260" y="192" fill="#6B645E" font-size="10">the network learns the knobs → it can stretch, shift, or fully undo the level</text></g></svg>
%%%

**Victory lap:** you now know layer norm is really *two* moves — a fixed re-level, then a *learned* rescale — and that the learned part is what keeps the network in control.

@@@ concept id=c5 tag="Safety catch" title="The tiny number that saves the divide" gotit="Got epsilon"
Time for a small puzzle hidden inside the re-level. Remember the move divides by the spread `σ`. Dividing is usually fine — but what happens if the spread is **zero**?

Think of **splitting a pizza among friends**. If there are 4 friends, each gets a quarter — easy. If there's 1 friend, they get the whole thing. But what if there are *zero* friends? "Split the pizza among nobody" is a nonsense question — the maths just breaks. Dividing by zero is exactly that kind of nonsense: the computer throws up its hands and produces "infinity" or an error. And a token *can* have zero spread: if all its numbers happen to be equal — say every number is `5` — then they don't spread out at all, so `σ = 0`, and the divide blows up.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Two pizza-splitting scenes. Left: a pizza split among four friends, each gets a clean slice. Right: a pizza with a question mark and zero friends, marked with a red X and the word undefined, showing divide by zero breaks."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Split among 4 friends: fine. Split among 0: nonsense.</text><circle cx="130" cy="100" r="46" fill="#FBE6C9" stroke="#C99A12" stroke-width="2"/><line x1="130" y1="54" x2="130" y2="146" stroke="#C99A12"/><line x1="84" y1="100" x2="176" y2="100" stroke="#C99A12"/><text x="130" y="162" fill="#276b45" font-size="10">÷ 4 friends → clean slices ✓</text><circle cx="390" cy="100" r="46" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="390" y="107" fill="#C93B3B" font-size="26">?</text><text x="390" y="162" fill="#C93B3B" font-size="10">÷ 0 → undefined ✗</text><text x="260" y="100" fill="#6B645E" font-size="14">vs</text></g></svg>
%%%

**What the pizza picture gets right:** dividing by zero is a genuinely broken question, not a small rounding issue — the whole calculation falls apart. **Where it breaks down:** you rarely meet a pizza with zero friends, but a token with near-equal numbers (tiny spread) really does show up in a big model — so this isn't a rare edge case we can ignore.

#### The fix: nudge the bottom away from zero
The remedy is charmingly simple. We add a **tiny** number — call it [[epsilon||A very small constant (like 0.00001) added inside the square root during normalization, so the divisor is never exactly zero and the divide never blows up.]] (written `ε`, "epsilon") — to the spread before dividing. Epsilon is tiny, like `0.00001`, so it barely changes anything when the spread is normal. But when the spread is near zero, that little epsilon keeps the bottom of the fraction safely away from zero, and the divide stays sane.

%%% formula
expr: normalized = (x − μ) / √(σ² + ε)
note: ε is a tiny constant added inside the square root so the divisor is never exactly zero
%%%

Say it plainly: "before dividing by the spread, add a crumb so the bottom is never truly zero." Watch the rescue in one before→after picture — the same near-zero-spread token, blowing up without epsilon and staying calm with it:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Top row without epsilon: a division by a spread of zero produces infinity, marked in red. Bottom row with epsilon: the same division now divides by a tiny safe number and produces a normal finite result, marked in green."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Near-zero spread: without ε it blows up, with ε it's safe</text><text x="60" y="52" fill="#C93B3B" text-anchor="start" font-weight="bold">without ε</text><rect x="60" y="62" width="120" height="34" rx="5" fill="#FDF3F3" stroke="#C93B3B"/><text x="120" y="84" fill="#2C2A28" font-size="11">(x − μ) ÷ 0</text><text x="200" y="84" fill="#B8AEA2" font-size="15">→</text><rect x="240" y="62" width="150" height="34" rx="5" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="315" y="84" fill="#C93B3B" font-size="13">∞ / error 💥</text><line x1="60" y1="118" x2="460" y2="118" stroke="#E5DFD6" stroke-dasharray="4,3"/><text x="60" y="150" fill="#276b45" text-anchor="start" font-weight="bold">with ε</text><rect x="60" y="160" width="150" height="34" rx="5" fill="#F1F8F3" stroke="#2D8B55"/><text x="135" y="182" fill="#2C2A28" font-size="11">(x − μ) ÷ √(0 + ε)</text><text x="230" y="182" fill="#B8AEA2" font-size="15">→</text><rect x="270" y="160" width="150" height="34" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="345" y="182" fill="#276b45" font-size="12">a normal number ✓</text></g></svg>
%%%

**You just closed a real production bug before it happened.** Every layer-norm implementation you'll ever read carries that little `ε` inside the square root — now you know exactly why it's there.

@@@ concept id=c6 tag="Where it sits" title="Before the sublayer, not after" gotit="Got placement"
Now let's put the re-level where you'll actually meet it, and answer a question that turns out to matter a surprising amount: *when* in a block do we do it? Yesterday you learned a block has two working parts — attention and feed-forward — each wrapped by a residual shortcut. Layer norm slips in right next to each of those parts. But there are two places it could go, and the choice changes how smoothly the model trains.

Think of **spell-checking an email**. You can check it two ways. **Way one:** write the email, hit send, *then* run spell-check on the sent copy — too late, the messy version already went out and each reply builds on the mess. **Way two:** run spell-check *before* you send, so every reader gets a clean version and the conversation stays tidy from the start. Placing layer norm *before* each sublayer is "way two": clean the numbers first, so the sublayer always reads something tidy. Placing it *after* the residual add is "way one": the block does its work on messy numbers and cleans up late.

%%% svg
<svg viewBox="0 0 520 230" role="img" aria-label="Two block diagrams side by side. Left labeled Post-LN: input goes into the sublayer, then the residual add, then layer norm at the end. Right labeled Pre-LN: input first goes through layer norm, then the sublayer, then the residual add, with the clean stream flowing straight up."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="130" y="20" fill="#C93B3B" font-weight="bold">Post-LN (original) · clean up LATE</text><rect x="80" y="180" width="100" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="130" y="197" fill="#276b45">input x</text><rect x="80" y="132" width="100" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="130" y="149" fill="#5E5191">sublayer</text><rect x="80" y="88" width="100" height="26" rx="4" fill="#F1F1F1" stroke="#9A938A"/><text x="130" y="105" fill="#6B645E">+ residual</text><rect x="80" y="44" width="100" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="130" y="61" fill="#9A7A10">layer norm</text><line x1="130" y1="180" x2="130" y2="158" stroke="#B8AEA2"/><line x1="130" y1="132" x2="130" y2="114" stroke="#B8AEA2"/><line x1="130" y1="88" x2="130" y2="70" stroke="#B8AEA2"/><line x1="390" y1="20" x2="390" y2="20" stroke="none"/><text x="390" y="20" fill="#276b45" font-weight="bold">Pre-LN (modern) · clean up FIRST</text><rect x="340" y="180" width="100" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="390" y="197" fill="#276b45">input x</text><rect x="340" y="136" width="100" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="390" y="153" fill="#9A7A10">layer norm</text><rect x="340" y="92" width="100" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="390" y="109" fill="#5E5191">sublayer</text><rect x="340" y="48" width="100" height="26" rx="4" fill="#F1F1F1" stroke="#9A938A"/><text x="390" y="65" fill="#6B645E">+ residual</text><line x1="390" y1="180" x2="390" y2="162" stroke="#B8AEA2"/><line x1="390" y1="136" x2="390" y2="118" stroke="#B8AEA2"/><line x1="390" y1="92" x2="390" y2="74" stroke="#B8AEA2"/><path d="M470 193 L470 61" stroke="#2D8B55" stroke-width="2.5" stroke-dasharray="5,3"/><text x="485" y="130" fill="#276b45" font-size="8" text-anchor="start">clean</text><text x="485" y="142" fill="#276b45" font-size="8" text-anchor="start">stream</text></g></svg>
%%%

**What the spell-check picture gets right:** cleaning *before* the work means every step reads a tidy version, while cleaning *after* lets the mess spread first — that's the placement choice exactly. **Where it breaks down:** a spell-check fixes typos, but layer norm doesn't fix "mistakes" — it just rescales — so it's the *timing*, not the *correction*, that the analogy is really about.

#### The two placements have names
Putting the norm *after* the sublayer and residual add is [[Post-LN||"Post-LayerNorm": the original transformer design, where layer norm comes after the sublayer and residual add. Can be unstable to train very deep without warm-up tricks.]] — the original 2017 transformer did this. Putting the norm *before* the sublayer is [[Pre-LN||"Pre-LayerNorm": the modern design (GPT-style), where layer norm comes before the sublayer so the residual stream flows through unchanged. Much more stable to train deep.]] — the design nearly every modern model (GPT-style) uses today.

Why did the field switch? This is where **yesterday's shortcut and today's re-level team up**. Remember the residual stream should flow straight up, a clean express lane. In Pre-LN, the norm sits *off to the side* — it cleans a copy that feeds the sublayer, while the main stream sails through untouched and only gets the sublayer's change *added* on. That keeps the express lane pristine, so very deep stacks train smoothly. In Post-LN, the norm sits *right on* the main stream, nudging it at every block, which can make the signal unstable to train when you stack dozens of layers.

!!! c-ok ✅
<b>The team-up, in one line:</b> the <b>residual connection</b> (yesterday) keeps a full-strength copy of the signal flowing; <b>layer norm</b> (today) keeps that signal's scale steady. Together — especially in the <b>Pre-LN</b> arrangement — they're what make it possible to train transformers dozens of layers deep without the signal exploding or vanishing. Neither trick alone is enough; the pair is the magic.
!!!

**Victory lap:** you can now sketch a transformer sublayer *and* explain why modern models put the norm first. That "Pre-LN vs Post-LN" question is a real staff-interview beat, and you've got it.

@@@ concept id=c7 tag="Leaner cousin" title="RMSNorm — skip the centering" gotit="Got RMSNorm"
Here's a curious afterthought that turned into the modern default. Layer norm does two things: it *centers* (slides the average to 0) and it *rescales* (sets the spread to 1). Someone asked a cheeky question — do we really need the centering step? And it turns out, mostly, no.

Think of **tidying your desk before studying**. The full tidy is two jobs: (1) *shove everything to the center* so it's lined up, then (2) *shrink the piles* down to a neat size. But honestly, the shrinking is what makes the desk usable — the centering is extra fuss. If you're in a hurry, you skip the centering and just shrink the piles; the desk is *almost* as good, and you saved half the work. RMSNorm is that shortcut: it drops the "slide to center" step and only does the "shrink to a neat size" step.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two recipes side by side. LayerNorm has two steps: center then rescale. RMSNorm has one step: rescale only, with the center step crossed out, labeled cheaper and faster."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Same goal, one fewer step</text><text x="130" y="46" fill="#5E5191" font-weight="bold">LayerNorm</text><rect x="55" y="60" width="150" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="130" y="79" fill="#5E5191">1 · center (− mean)</text><line x1="130" y1="90" x2="130" y2="104" stroke="#B8AEA2"/><rect x="55" y="104" width="150" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="130" y="123" fill="#5E5191">2 · rescale (÷ spread)</text><text x="130" y="158" fill="#6B645E" font-size="9">two steps</text><text x="390" y="46" fill="#276b45" font-weight="bold">RMSNorm</text><rect x="315" y="60" width="150" height="30" rx="5" fill="#F1F1F1" stroke="#B8AEA2" stroke-dasharray="4,3"/><text x="390" y="79" fill="#B8AEA2">1 · center (skip!)</text><line x1="322" y1="62" x2="458" y2="88" stroke="#C93B3B" stroke-width="1.5"/><rect x="315" y="104" width="150" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="390" y="123" fill="#276b45">rescale by RMS only</text><text x="390" y="158" fill="#276b45" font-size="9">one step → cheaper, faster</text><text x="260" y="192" fill="#6B645E" font-size="10">skipping the centering barely hurts quality and saves compute</text></g></svg>
%%%

**What the desk picture gets right:** two tidy-up jobs where the second does most of the good — drop the first and you're nearly as tidy for half the effort — that's RMSNorm's trade exactly. **Where it breaks down:** a desk is one physical mess, but here we're skipping a step across billions of numbers per layer, so the "half the work saved" adds up to real speed on a huge model.

#### What the "shrink" step measures
Instead of the spread (which needs the average first), RMSNorm rescales by the [[root mean square||A measure of the typical magnitude of a list of numbers: square them, take the average, take the square root. Unlike standard deviation it does NOT subtract the mean first.]] — "RMS" for short. In plain words: square the numbers, average them, take the square root. It measures the *typical size* of the numbers without ever caring where their center is — which is exactly why it can skip the centering step.

%%% mathladder
title: RMSNorm vs LayerNorm — one fewer step
words: LayerNorm centers then rescales; RMSNorm skips centering and just rescales by the typical size (RMS)
formula: LayerNorm: (x − μ)/σ · γ + β    →    RMSNorm: x / RMS(x) · γ
numbers: x=[3,4]: LayerNorm subtracts mean 3.5 first; RMSNorm uses RMS=√((9+16)/2)=√12.5≈3.54 straight away
sanity: RMSNorm has no β and never computes μ — fewer operations, so it runs faster on giant models
%%%

Because it does less arithmetic — no averaging, no subtracting, one fewer knob — RMSNorm is faster, and on big models it works about as well as full layer norm. That's why many of today's largest language models (the LLaMA family, for example) use RMSNorm instead of the original layer norm.

**Victory lap:** you now know not just *the* norm but the *family* — a full layer norm and its leaner cousin — and exactly what the cousin trades away to run faster.

@@@ concept id=c8 tag="What it isn't" title="It steadies, it doesn't think" gotit="Got the limit"
Before the recap, one honest caveat — the kind a good engineer always keeps in mind. It's tempting to think a step this important must make the model *smarter*. It doesn't, and knowing that is a sign you truly get it.

Think of a **camera tripod**. A tripod holds your camera perfectly steady so your photos aren't blurry. But the tripod doesn't take a *better picture* — it doesn't choose the subject, set the focus, or frame the shot. That's all the camera and you. All the tripod does is keep things steady enough that the real work can happen cleanly. Layer norm is that tripod: it keeps the numbers steady so the actual thinking parts — attention and feed-forward — can do their job. It adds no thinking power of its own.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A camera on a tripod. A note points to the tripod saying it only keeps the camera steady, and a note points to the camera lens saying the real picture comes from here, not the tripod."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Tripod = keep it steady, NOT take a better photo</text><rect x="220" y="50" width="80" height="46" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><circle cx="260" cy="73" r="15" fill="#FDF9F3" stroke="#5E5191" stroke-width="2"/><circle cx="260" cy="73" r="7" fill="#5E5191"/><line x1="260" y1="96" x2="230" y2="150" stroke="#9A938A" stroke-width="2.5"/><line x1="260" y1="96" x2="290" y2="150" stroke="#9A938A" stroke-width="2.5"/><line x1="260" y1="96" x2="260" y2="150" stroke="#9A938A" stroke-width="2.5"/><text x="130" y="140" fill="#276b45" font-size="9">tripod → steady only</text><path d="M175 140 L235 135" stroke="#2D8B55" stroke-width="1.2"/><text x="405" y="66" fill="#5E5191" font-size="9">camera → real picture</text><path d="M345 68 L302 72" stroke="#7C6DAA" stroke-width="1.2"/></g></svg>
%%%

**What the tripod picture gets right:** it keeps things steady so the real work is clean, but it captures nothing itself — just like layer norm steadies the numbers while attention and feed-forward do the thinking. **Where it breaks down:** you can take a photo *without* a tripod (just shakier), but a deep transformer without normalization often won't train at all — so it's closer to "essential support" than "nice-to-have."

#### The three honest limits
Let's be precise about what layer norm *cannot* do — this is exactly the kind of "know the edges" answer interviewers love:

%%% cards
🧠 | No new brainpower | It only rescales numbers — it adds no capacity to represent new patterns. All the real learning lives in attention and feed-forward.
🏗️ | Can't fix a bad design | It won't rescue a poorly-designed architecture or bad weight setup — a wobbly model stays wobbly; norm just steadies the numbers.
🔀 | Doesn't mix tokens | It levels each token *on its own*, so it never shares information *between* words. Mixing across words is attention's job, not the norm's.
%%%

That last one is worth holding onto. Yesterday's residual carried each token straight up; today's layer norm re-levels each token *on its own*. Neither one lets tokens *talk to each other* — that's the special power of **attention**, which you already met and will keep coming back to. Layer norm is a per-token helper, full stop.

!!! c-ok 🎤
<b>Once it clicks, here's how you'd explain it (say, in an interview):</b> "Layer norm standardizes each token's feature vector to zero mean and unit variance, then rescales with a learned gain γ and bias β. It's per-token — independent of the batch, so it behaves identically at train and inference — which is why transformers use it over batch norm, given variable-length sequences and single-example serving. An ε inside the square root guards against zero variance. Modern nets place it *before* each sublayer (Pre-LN) to keep the residual stream clean and stable at depth, and many large models swap in RMSNorm to drop the mean-centering for speed. Crucially, it only stabilizes — it adds no representational capacity and doesn't mix information across tokens; that's attention's job." Notice you can already say every piece of that — that's how far you've come today.
!!!

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you did the whole thing. Before the cheat-sheets, one picture to hold the whole day.

Think of a **water level (a spirit level) on a shelf you're hanging**. You add a bracket, and the shelf tilts a little. You add another, it tilts more. Left alone, the shelf drifts further off true with each thing you hang. So after every bracket you check the little bubble and gently re-**level** it back to center. The residual stream is the shelf, each block is a bracket that nudges it off, and layer norm is the spirit level you apply after each one — a quick re-center so the shelf stays true no matter how much you hang on it. **Where this breaks down:** a spirit level only checks *tilt*, but layer norm re-levels a whole list of numbers at once — center *and* spread — and it hands the network two knobs to tilt things back on purpose if it wants.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A shelf on a wall with a spirit level bubble. Brackets are added one by one tilting the shelf, and after each one the level re-centers the bubble, keeping the shelf true from left to right."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">Spirit level: each bracket tilts the shelf, re-level it back to true</text><line x1="40" y1="70" x2="150" y2="82" stroke="#C93B3B" stroke-width="4"/><rect x="80" y="60" width="30" height="12" rx="6" fill="#FDECEC" stroke="#C93B3B"/><circle cx="88" cy="66" r="3" fill="#C93B3B"/><text x="95" y="104" fill="#C93B3B" font-size="9">tilted (drifted)</text><text x="180" y="80" fill="#B8AEA2" font-size="14">→</text><line x1="220" y1="76" x2="330" y2="76" stroke="#2D8B55" stroke-width="4"/><rect x="260" y="66" width="30" height="12" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><circle cx="275" cy="72" r="3" fill="#2D8B55"/><text x="275" y="104" fill="#276b45" font-size="9">re-leveled ✓</text><text x="360" y="80" fill="#B8AEA2" font-size="14">→</text><line x1="400" y1="76" x2="490" y2="76" stroke="#2D8B55" stroke-width="4"/><rect x="430" y="66" width="30" height="12" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><circle cx="445" cy="72" r="3" fill="#2D8B55"/><text x="445" y="104" fill="#276b45" font-size="9">stays true</text><text x="260" y="150" fill="#6B645E" font-size="10">shelf = residual stream · bracket = a block · spirit level = layer norm</text><text x="260" y="170" fill="#6B645E" font-size="10">check and re-center after every step → the shelf never drifts off true</text></g></svg>
%%%

**The day as seven signposts, in order** — each one only makes sense because of the one before it:
- **1 · The puzzle.** The residual stream keeps *adding*, so its numbers drift off scale (bigger, tinier, lopsided) — that's *internal covariate shift*, and it makes training shaky.
- **2 · The fix.** Re-**level** each token: slide its average to `0` and set its spread to `1`. That's `(x − μ) / σ`.
- **3 · Per token.** Level a token against its *own* numbers (across features), not across the *batch* — unlike the older *BatchNorm*, which stumbles on variable-length text and single-example serving.
- **4 · Two knobs.** A learned gain `γ` and bias `β` let the network stretch and shift the tidy signal — even fully undo the level if it wants: `output = γ · normalized + β`.
- **5 · Safety catch.** A tiny `ε` inside the square root stops a divide-by-zero when a token's spread is near zero.
- **6 · Where it sits.** Modern models put the norm *before* each sublayer (**Pre-LN**) to keep the residual stream clean and stable at depth — with residuals, this pair is what makes deep transformers trainable.
- **7 · Leaner cousin.** **RMSNorm** drops the centering and rescales by the *root mean square* only — cheaper and faster, used by many big models.
- **The limit.** Layer norm only *steadies*; it adds no thinking power and never mixes tokens (that's attention's job).

#### Cheat-sheet · layer norm at a glance
%%% table
:: Piece :: In plain words :: Why it matters
`μ`, `σ` :: a token's average and spread :: measured across that token's own numbers, on its own
`(x − μ) / σ` :: center to 0, rescale spread to 1 :: the re-level — a steady scale for the next block
`γ`, `β` :: learned gain and bias knobs :: let the network stretch/shift or fully undo the level
`ε` :: tiny number inside the √ :: stops a divide-by-zero when the spread is near 0
Pre-LN :: norm *before* the sublayer :: keeps the residual stream clean → stable deep training
RMSNorm :: rescale by RMS, no centering :: cheaper, faster; used by many large models
%%%

#### Cheat-sheet · the words you met today
%%% jargon
layer normalization | re-level one token's numbers to average 0 and spread 1, then rescale with learned knobs
mean (μ) | the average of a token's numbers — the center they drift around
standard deviation (σ) | the typical spread of the numbers around their average
gain (γ) | a learned knob that stretches the spread of the standardized numbers
bias (β) | a learned knob that shifts the center of the standardized numbers
epsilon (ε) | a tiny constant inside the √ so the divide never hits zero
internal covariate shift | the input scale to each layer drifting during training — the problem norm fixes
BatchNorm | the older norm that levels across the batch; stumbles on text (variable lengths, single-example serving)
Pre-LN / Post-LN | norm placed before the sublayer (modern, stable) vs after the residual add (original)
RMSNorm | a leaner norm that skips centering and rescales by the root mean square only
%%%

That's the whole day. Next you'll open up one of the parts the norm wraps — **the feed-forward sublayer**, where each token gets its own little burst of thinking.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does layer normalization do to a token's numbers? | a:2 | It multiplies them by the batch average | It removes the smallest numbers | It re-levels them so their average is 0 and their spread is 1, then rescales with learned knobs | It sorts them from largest to smallest | fb: Layer norm centers each token's numbers to mean 0 and spread 1, then stretches/shifts with the learned gain γ and bias β.
q: Why do transformers use LayerNorm instead of BatchNorm? | a:1 | Because BatchNorm is slower to compute | Because layer norm levels each token on its own, so it doesn't depend on batch size and behaves the same at train and inference — while text has variable lengths and single-example serving | Because BatchNorm can't run on a GPU | Because layer norm adds more parameters | fb: BatchNorm needs stable batch statistics, which break for variable-length sequences and one-at-a-time inference; layer norm uses only the token's own numbers.
q: What is the tiny ε inside the square root for? | a:2 | To make training faster | To add extra learnable parameters | To keep the divisor from being exactly zero when a token's spread is near zero, avoiding a divide-by-zero blow-up | To shift the average to zero | fb: If all a token's numbers are nearly equal, the spread is near 0; the tiny ε keeps the bottom of the fraction safely non-zero.
q: Which is TRUE of layer normalization? | a:3 | It mixes information across all the tokens in the sentence | It replaces the need for attention | It makes the model smarter by adding representational power | It only stabilizes and rescales — it adds no thinking power and never mixes tokens (that's attention's job) | fb: Layer norm is a per-token steadier: it keeps numbers on an even keel but adds no new capacity and never shares information between tokens.
%%%

@@@ produce id=produce tag="Produce" title="Watch a drifting stream get re-leveled" gotit="Done"
Time to see today's big idea with your own eyes. You'll take a token whose numbers have drifted off scale, re-level it, and **watch** the average snap to 0 and the spread snap to 1 — then confirm the two knobs can put any scale back. **Predict first:** if a token's numbers are `[10, 20, 30]`, what will their average be *after* you subtract the mean? Then run it and **observe** the re-level. Pick one path.

#### Option A · write it yourself
Create `sessions/m05a-text-transformer/day-02-layer-norm/experiment.py`. Start with a drifted token, for example `x = np.array([10.0, 20.0, 30.0])`. **Notice** four things: (1) compute the mean `μ` and standard deviation `σ`, subtract the mean and divide by the spread, and print that the result's mean is ~`0` and its spread is ~`1`; (2) add a tiny `eps = 1e-5` inside the square root and confirm the answer barely changes for normal spreads; (3) make a *degenerate* token where every value is equal (for example `np.full(4, 7.0)`) and **observe** that *without* `eps` you get NaN/inf, but *with* `eps` you get a finite result; (4) apply learned knobs `gamma` and `beta` (`out = gamma * norm + beta`) and show that picking `gamma = σ` and `beta = μ` recovers the original `x` — the "undo" trick. Assert the normalized mean is close to 0 with `np.allclose`. Run with `python3 sessions/m05a-text-transformer/day-02-layer-norm/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 5a Day 2 artifact.

Create sessions/m05a-text-transformer/day-02-layer-norm/experiment.py that, with a comment on each step:
1. Defines layer_norm(x, gamma, beta, eps=1e-5) that computes mean and variance across the last axis, normalizes as (x - mean)/sqrt(var + eps), then returns gamma * norm + beta.
2. Takes a drifted token x = np.array([10., 20., 30.]), applies layer_norm with gamma=1, beta=0, and prints that the output mean is ~0 and std is ~1 (use np.allclose to assert mean ~ 0).
3. Makes a degenerate token np.full(4, 7.0) (zero variance) and shows that WITHOUT eps the normalization gives nan/inf, but WITH eps it gives a finite result — demonstrating why eps is needed.
4. Shows the "undo" trick: with gamma = original std and beta = original mean, layer_norm recovers the original x (np.allclose).
5. Also computes RMSNorm(x) = x / sqrt(mean(x**2) + eps) * gamma (no mean-centering) and prints it beside the LayerNorm output so the difference is visible.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the re-level, then the knobs)
- The drifted token `[10, 20, 30]` comes out with mean `~0` and spread `~1` — the re-level snapping everything to a steady scale.
- The degenerate token (all equal) gives `nan`/`inf` *without* `eps` and a clean finite number *with* it — proof of why that tiny safety number lives inside the square root.
- The moment to **watch**: setting `gamma` to the original spread and `beta` to the original mean returns you *exactly* to the original numbers — showing the knobs can fully *undo* the level, so the network never loses control.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05a-text-transformer/day-02-layer-norm/log.md`: (1) what `(x − μ) / σ` does and why we do it *per token* not per batch; (2) what the gain `γ` and bias `β` are for, and how they can undo the level; (3) why modern models place the norm *before* the sublayer (Pre-LN) and what RMSNorm drops to run faster.
!!!

@@@ fin
