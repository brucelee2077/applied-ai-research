---
quest_id: wf5-d03-minibatch
mode: concept
donor: v9-base.donor
page_title: "Module 5 · Day 3 — The Mini-Batch Training Loop"
module_label: "Module 5 · Train · Day 3"
title: "The Mini-Batch Loop"
subtitle: "Where the Model Finally Learns"
brand_sub: "Foundations · M5 Day 3"
spine: "practice: a machine learns by practicing on small handfuls and being corrected, over and over"
nav_prev_href: "../day-02-backward-pass/lesson.html"
nav_prev_label: "The Backward Pass, Wired by Hand"
nav_next_href: "../day-04-training-loss-dropout/lesson.html"
nav_next_label: "Full Training: Loss Curve & Dropout"
fin_title: "Module 5 · Day 3 complete! 🏆"
fin_body: "You did it! You finished <b>The Mini-Batch Loop</b>. Look at what you just built: you wrapped yesterday's single correction inside a loop that runs forward → loss → backward → update, over batch after batch, epoch after epoch — and you watched the training loss actually fall. You can now use the real words (epoch, iteration, batch, batch_size), you know why a small handful beats both one example and the whole set, why you shuffle every epoch, why you average instead of add, and — the big one — that a falling training loss proves the loop works, not that the model is good.<br>Take a breath: that is the whole heartbeat of how every neural network on Earth learns. Next up: <b>Full Training: Loss Curve & Dropout</b> — where you split off some held-out data and finally get to <i>measure</i> whether your model truly learned."
notebook_yardstick: 00-neural-networks/fundamentals/08_training_loop.ipynb
coverage_topics:
  - {topic: the training loop as the central mechanism, keywords: [training loop, forward pass, compute loss, backward pass, update, repeat over]}
  - {topic: epoch is one full pass over the dataset, keywords: [epoch, full pass, entire training, many epochs]}
  - {topic: iteration or step is one update on one batch, keywords: [iteration, step, one parameter update, distinct from an epoch]}
  - {topic: batch and batch_size, keywords: [batch, minibatch, mini-batch, batch_size, small group, processed together]}
  - {topic: minibatch gradient descent averages the gradient over the batch, keywords: [minibatch gradient descent, average the gradient, one step, modern default]}
  - {topic: full-batch gradient descent ancestor, keywords: [full-batch, all n examples, exact gradient, one update per pass, slow, memory-heavy]}
  - {topic: stochastic gradient descent single-example extreme, keywords: [stochastic gradient descent, single example, batch of 1, frequent, noisy updates]}
  - {topic: the gradient-noise trade-off dial, keywords: [gradient-noise trade-off, noisier gradient, more updates, generalization, smoother, more memory, batch_size is the dial]}
  - {topic: shuffling the training data each epoch, keywords: [shuffle, fixed order, learn the sequence, correlated batches, shuffle indices, every epoch]}
  - {topic: iterating the dataset in batches including the ragged final batch, keywords: [slice, chunks of batch_size, final batch, ragged, partial batch, not divisible]}
  - {topic: averaging mean rather than summing the loss and gradient, keywords: [mean, average, rather than summing, effective step size, does not scale with batch_size]}
  - {topic: tracking training loss to confirm learning, keywords: [track, print, training loss, per epoch, trending down, actually learning]}
  - {topic: learning rate as the per-update step size coupled to batch_size, keywords: [learning rate, step-size, once per batch, coupling, mean vs sum, effective update]}
  - {topic: capability limit low training loss does not prove a good model, keywords: [capability limit, optimization procedure, not a correctness guarantee, overfit, generalize poorly, low training loss does not prove]}
require_artifact: true
---

@@@ hero
@lede Yesterday you built the corrector: one forward pass that guesses, one backward pass that says exactly how to fix each weight. But that was *one* correction on *one* handful of images. Here's the puzzle. Nobody learns 60,000 flashcards by studying them all at once. And nobody learns them one lonely card at a time either. You practice in small handfuls, correct yourself, grab the next handful, and go around again. Today you wrap yesterday's single nudge inside that loop — and for the first time, you get to watch the loss actually fall.
@goal Here's what we'll figure out together today, one small idea at a time:
- the **training loop** — the four steps you already know, now put on repeat;
- the three words for its rhythm — **epoch**, **iteration**, **batch** (and `batch_size`);
- why a small **handful** beats both one example and the whole dataset;
- why you **shuffle** every time around, and why you **average** instead of add;
- and the one honest thing a falling training loss does *not* prove.

@@@ concept id=c1 tag="The loop" title="The training loop: the same four steps, on repeat" gotit="Got the loop"
Practice is never a single act. It is a *loop*. Yesterday you did one correction. Today you put that correction inside a repeat — and that is the whole trick.

First, let's put plain words on every scary term you'll meet today, so nothing sneaks up on you:

%%% jargon
training loop | doing the same four steps over and over on the data until the model learns
epoch | one full pass over the whole training set
iteration (step) | one weight update, on one small batch of data
batch (mini-batch) | a small handful of examples the model looks at together before updating once
batch_size | how many examples are in that handful
shuffle | mixing up the order of the data before you use it
learning rate | how big a step you take each time you update the weights
%%%

Now the loop itself. The [[training loop||Repeating four steps over the data — forward pass, compute the loss, backward pass, update the weights — until the model has learned. It is what turns a static forward+backward network into a model that actually learns.]] is just those same four steps you already built, run again and again:

- **forward pass** — the model guesses;
- **compute the loss** — how wrong the guess was;
- **backward pass** — which way to fix each weight;
- **update** — take one small step in that direction.

Then you do it again on the next slice of data. That repeat is the whole thing that turns a frozen network into one that learns.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A circular loop of four boxes: forward pass, compute loss, backward pass, update, with an arrow returning from update back to forward pass to show the loop repeats over the data"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="30" y="60" width="90" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="75" y="78" fill="#1a5c38" font-weight="bold">forward</text><text x="75" y="92" fill="#6B645E" font-size="9">guess</text><text x="128" y="84" fill="#6B645E">→</text><rect x="146" y="60" width="90" height="40" rx="6" fill="#FDF3E8" stroke="#C99A12" stroke-width="2"/><text x="191" y="78" fill="#8A6D3B" font-weight="bold">loss</text><text x="191" y="92" fill="#6B645E" font-size="9">how wrong</text><text x="244" y="84" fill="#6B645E">→</text><rect x="262" y="60" width="90" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="307" y="78" fill="#C93B3B" font-weight="bold">backward</text><text x="307" y="92" fill="#6B645E" font-size="9">which way</text><text x="360" y="84" fill="#6B645E">→</text><rect x="378" y="60" width="90" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="423" y="78" fill="#5E5191" font-weight="bold">update</text><text x="423" y="92" fill="#6B645E" font-size="9">take step</text><path d="M423 100 L423 132 L75 132 L75 100" fill="none" stroke="#7C6DAA" stroke-width="1.5" stroke-dasharray="4 3"/><text x="249" y="148" fill="#6B645E" font-size="10">repeat on the next slice of data — that loop is what learns</text></g></svg>
%%%

**Think of practicing free throws.** You shoot (guess). You see how far you missed (loss). You figure out what to change — too much wrist, aim a bit left (backward). You adjust your form (update). Then you shoot again. One shot teaches you almost nothing. A thousand shots, each one corrected, teaches you the game.

**What the free-throw picture gets right:** you get better from the *loop* — the same four beats, repeated shot after shot — not from any single shot.

**Where it breaks down:** a shooter fixes their form after every single shot. Our model, as you'll see next, waits for a small *handful* of examples first, because a handful gives a steadier read on what to change.

#### The loop is the mechanism — everything else is a dial on it
Practice with correction is the spine of this whole module. Day 1 gave you the guess. Day 2 gave you the correction. Today's new idea is small but it is the whole game: *put it in a loop over the data.* Nothing about the forward or backward math changes. Everything else in this lesson — batches, shuffling, the learning rate — is just a knob you turn on this one loop.

And that is your first win of the day. You just learned the training loop — the single idea that turns a still, frozen network into one that learns. That is the heart of everything that follows. Everything from here is detail hung on this one loop.

@@@ concept id=c2 tag="The words" title="Epoch, iteration, batch: three words for three sizes of a step" gotit="Got the vocabulary"
The loop runs a *lot* of times. People use three precise words for its rhythm: **epoch**, **iteration**, and **batch**.

Don't worry if these three words blur together at first. Everyone mixes them up in the beginning — it is one of the most normal stumbles there is. Let's untangle them slowly, together, and by the end of this section they'll click into place.

- A [[batch||A small group of examples processed together before one weight update. The number of examples in it is the batch_size.]] (also called a **mini-batch**) is a small handful of examples the model looks at together before it updates once.
- An [[iteration||Also called a step: one parameter update on one batch. Distinct from an epoch — there are many iterations in one epoch.]] — also called a **step** — is *one* weight update, on *one* batch.
- An [[epoch||One full pass over the entire training dataset. The loop runs for many epochs.]] is one full pass over the *entire* dataset.

The key link: one epoch contains many iterations.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A dataset bar split into many small batch chunks. One chunk is labeled batch of batch_size examples. One arrow from a chunk to an update is labeled iteration or step. The whole bar swept once is labeled one epoch."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="22" fill="#6B645E">the full training set (N examples)</text><g>
<rect x="20" y="34" width="44" height="34" rx="3" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/>
<rect x="66" y="34" width="44" height="34" rx="3" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/>
<rect x="112" y="34" width="44" height="34" rx="3" fill="#FDF3E8" stroke="#C99A12" stroke-width="2.5"/>
<rect x="158" y="34" width="44" height="34" rx="3" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/>
<rect x="204" y="34" width="44" height="34" rx="3" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/>
<text x="290" y="55" fill="#6B645E" font-size="10">… many more batches …</text>
<rect x="456" y="34" width="44" height="34" rx="3" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/>
</g><text x="134" y="86" fill="#8A6D3B" font-weight="bold" font-size="10">one batch</text><text x="134" y="99" fill="#6B645E" font-size="9">= batch_size examples</text><path d="M134 102 L134 120" fill="none" stroke="#5E5191" stroke-width="1.5"/><rect x="96" y="122" width="76" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="134" y="139" fill="#5E5191" font-weight="bold" font-size="10">1 update</text><text x="134" y="160" fill="#6B645E" font-size="9">= one iteration / step</text><path d="M20 76 L500 76" stroke="#2D8B55" stroke-width="1" stroke-dasharray="3 2"/><text x="400" y="140" fill="#1a5c38" font-size="10">sweeping the whole bar once = one epoch</text></g></svg>
%%%

**Think of reading a textbook to study for an exam.** A *batch* is a small group of pages you read in one sitting. Finishing that sitting and jotting a note — adjusting your understanding — is one *step*. Reading the *whole book* cover to cover, once, is one *epoch*. And you'll read the whole book several times before the exam.

**What the textbook picture gets right:** the three words are just three sizes of progress — a sitting, a note per sitting, a full read-through — and a full read-through is made of many sittings.

**Where it breaks down:** a reader remembers pages between sittings. Our model only ever "remembers" through its weights. So each step's only lasting effect is the small nudge it makes to those numbers.

#### One worked count, so the words stick
Say you have `N = 60,000` training images. Say you choose `batch_size = 32`.

- Iterations (steps) in one epoch: `60,000 / 32 = 1,875`. So **one epoch = 1,875 weight updates.**
- Run `10` epochs, and the loop takes `10 × 1,875 = 18,750` steps in total.

Keep the three words distinct and you're set:

- **epoch** = one full pass;
- **iteration/step** = one update on one batch;
- **batch** = the handful, `batch_size` = how many are in it.

See? Not so bad once they sit still. You just picked up the exact vocabulary the whole field uses to describe training. From here on, when someone says "we trained for 10 epochs," you'll know exactly what they mean.

@@@ concept id=c3 tag="Two extremes" title="The two ancestors: the whole set, or one example at a time" gotit="Got the two poles"
Why a *handful*? Why not the whole dataset, or just one example? Great question — and the best way to answer it is to meet the two extremes that a handful sits between. These two are the historical ancestors. They are what people tried first, and their weaknesses are exactly why the modern default is a handful.

- At one pole is [[full-batch gradient descent||Compute the gradient over all N examples, then take a single update. The exact, smooth gradient every step — but only one update per full pass, and it is slow and memory-heavy on large data.]]: use *all* `N` examples to work out one exact gradient, then take a single step.
- At the other pole is single-example [[stochastic gradient descent||The extreme where batch_size = 1: update the weights after every single example. Very frequent updates, but each gradient is a noisy one-example estimate.]] (the original SGD): update after *every single* example, one at a time.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="Three panels of a U-shaped loss valley. Left, full-batch: one long smooth arrow straight to the bottom, labeled exact but one step per pass. Middle, mini-batch: a few short mostly-downhill steps, labeled the sweet spot. Right, single-example SGD: many tiny jittery zig-zag steps, labeled noisy but frequent."><g font-family="monospace" font-size="9" text-anchor="middle"><path d="M14 26 Q60 130 106 26" fill="none" stroke="#B8B0C8" stroke-width="2"/><path d="M40 74 Q56 112 60 118" fill="none" stroke="#2D8B55" stroke-width="2"/><circle cx="60" cy="118" r="4" fill="#1a5c38"/><text x="60" y="150" fill="#1a5c38" font-weight="bold">full-batch</text><text x="60" y="162" fill="#6B645E">exact · 1 step/pass · slow</text><path d="M194 26 Q240 130 286 26" fill="none" stroke="#B8B0C8" stroke-width="2"/><path d="M214 66 L226 92 L232 106 L240 116" fill="none" stroke="#2D8B55" stroke-width="1.8"/><circle cx="214" cy="66" r="3" fill="#2D8B55"/><circle cx="226" cy="92" r="3" fill="#2D8B55"/><circle cx="240" cy="116" r="4" fill="#1a5c38"/><text x="240" y="150" fill="#1a5c38" font-weight="bold">mini-batch</text><text x="240" y="162" fill="#6B645E">steady + frequent · the default</text><path d="M374 26 Q420 130 466 26" fill="none" stroke="#B8B0C8" stroke-width="2"/><path d="M392 58 L406 70 L400 84 L418 90 L410 104 L424 110 L420 118" fill="none" stroke="#C99A12" stroke-width="1.5"/><circle cx="420" cy="118" r="4" fill="#C99A12"/><text x="420" y="150" fill="#8A6D3B" font-weight="bold">single-example SGD</text><text x="420" y="162" fill="#6B645E">noisy · many steps</text></g></svg>
%%%

**Think of adjusting a soup you're cooking for a big crowd.** Full-batch is tasting the *entire* pot before every single change — the most accurate read you can get, but you can only adjust once per full stir, and stirring a giant pot is slow. Single-example is tasting *one spoonful* and re-seasoning off just that — fast and constant, but one spoonful can fool you (maybe it hit a salty spot).

**What the cooking picture gets right:** taste more at once and your read is more accurate but slower; taste less and it's faster but jumpier.

**Where it breaks down:** you re-season a real pot on your tongue's hunch. The model re-seasons using an exact, averaged gradient over whatever it tasted — so the "noise" here is a precise, measurable thing, not a guess.

#### Why each extreme hurts
Let's take them one at a time.

**Full-batch** gives you a *smooth, exact* gradient. That sounds ideal. The catch: you get only **one update per full pass** over the data. On 60,000 images, that is one single step per epoch. And you have to hold all `N` examples' computations in memory at once, which is heavy. Great in theory, painfully slow to actually train.

**Single-example** gives you `N` updates per epoch — very frequent. The catch: each gradient comes from just *one* example, so it is **noisy** and jumps all over the place. Progress is jittery.

So neither pole wins. Full-batch is too slow; single-example is too jumpy. The answer is to sit *between* them — and that is exactly the next idea.

@@@ concept id=c4 tag="The sweet spot" title="Mini-batch: average a handful, then take one step" gotit="Got the mini-batch"
Here's the happy middle. [[Minibatch gradient descent||Average the gradient over a small batch of examples, then take one step. The modern default that sits between full-batch (exact but slow) and single-example (frequent but noisy).]] is the compromise between the two poles, and it is the modern default.

Here's the recipe. You take a small handful — say 32 examples. You run the forward and backward pass on all of them. You **average** their gradients into one. Then you take a single step with that average. Then you grab the next handful.

The payoff: you get *many* updates per epoch (like single-example), but each update rests on 32 examples of evidence instead of 1 (much steadier). You quietly get the best of both poles.

%%% viz src=../../viz/gradient-descent.html title="stepping downhill on the loss surface — each step follows the (averaged) gradient" caption="interactive — the arrow is the gradient (uphill); each update steps the opposite way, downhill. A mini-batch averages a handful of these arrows into one steadier step."
%%%

**Think of a group of friends guessing the temperature of a pool.** Ask one friend and you get a fast but shaky answer ("feels cold!"). Ask all 500 people at the pool and you get a rock-solid number — but polling everyone takes forever. Ask a handful of 32 and average their guesses, and you land on an answer almost as steady as polling everyone, for a tiny fraction of the effort.

**What the pool picture gets right:** averaging a small sample kills most of the noise of a single opinion while staying cheap. That is exactly what averaging 32 gradients does.

**Where it breaks down:** pool-guessers are estimating one number. Each of your 32 examples produces a full gradient over about 100,000 weights, and you average all of those, weight by weight, into one step.

Nice — you just met the workhorse of modern training. Almost every network you'll ever hear about is trained this way. Now let's look at the one knob that controls it.

#### The batch_size dial: how shaky vs. how frequent your steps are
The one knob that sets where you sit between the two poles is `batch_size`. Turning it trades two things against each other: **how shaky** each step is versus **how frequent** your steps are.

%%% table
:: batch_size :: gradient :: updates per epoch :: memory :: note
small (e.g. 8) :: shakier estimate :: many :: light :: the shakiness can even *help* the model generalize
large (e.g. 512) :: smoother, more accurate :: few :: heavy :: fewer, steadier steps
%%%

That balance has a name — the [[gradient-noise trade-off||Small batch = a shakier (noisier) gradient estimate but more updates and often better generalization; large batch = a smoother, more accurate gradient but fewer updates and more memory. batch_size is the dial: how shaky versus how frequent your steps are.]], which is just a fancy way of saying "how shaky vs. how frequent your steps are." Let's unpack it slowly.

A **small** batch gives a shakier gradient — but more frequent updates. Strangely, that shakiness often *helps* the model generalize. The little jiggles knock the weights out of bad spots.

A **large** batch gives a smoother, more accurate gradient — but fewer updates, and it eats more memory.

So `batch_size` is the dial you turn to trade one against the other. In practice, sizes of 32 to 256 are the common sweet spot.

@@@ concept id=c5 tag="Slice it up" title="Iterating in batches — and the ragged last one" gotit="Got the slicing"
Now the hands-on mechanics. How do you actually feed the loop one batch at a time? You **slice** the data. You take the training set, cut it into consecutive chunks of `batch_size`, and hand the loop one chunk per iteration.

There's one small wrinkle worth naming right now, because it trips up almost everyone the first time. When `N` doesn't divide evenly by `batch_size`, the *last* chunk comes out smaller. That leftover chunk has a name: a [[ragged final batch||The last batch of an epoch when N is not divisible by batch_size: it holds the leftover examples, so it is smaller than a full batch. Your loop must handle it, not assume every batch is exactly batch_size.]] — it holds just the leftovers.

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="A bar of 100 examples sliced into chunks of 32: three full chunks of 32 shown in green, then a short final chunk of 4 shown in amber and labeled ragged final batch, four examples, smaller than batch_size."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="24" fill="#6B645E">N = 100 examples, batch_size = 32</text><rect x="20" y="40" width="140" height="38" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="90" y="64" fill="#1a5c38">batch 1 · 32</text><rect x="166" y="40" width="140" height="38" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="236" y="64" fill="#1a5c38">batch 2 · 32</text><rect x="312" y="40" width="140" height="38" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="382" y="64" fill="#1a5c38">batch 3 · 32</text><rect x="458" y="40" width="42" height="38" rx="4" fill="#FDF3E8" stroke="#C99A12" stroke-width="2.5"/><text x="479" y="64" fill="#8A6D3B" font-weight="bold">4</text><text x="479" y="104" fill="#8A6D3B" font-size="9">ragged</text><text x="479" y="115" fill="#8A6D3B" font-size="9">final batch</text><text x="236" y="128" fill="#6B645E" font-size="10">32 + 32 + 32 + 4 = 100 — the leftovers still get to train</text></g></svg>
%%%

**Think of packing eggs into cartons of a dozen.** You have 100 eggs and cartons of 12. You fill eight cartons full. The ninth carton has just 4 eggs in it. You don't throw those 4 eggs away. And you don't pretend the carton is full.

**What the egg-carton picture gets right:** the leftover group is real. It is smaller than a full carton. And it still counts — your loop has to handle it just as it is.

**Where it breaks down:** eggs are interchangeable, so a short carton is harmless. In code it's less forgiving. If your code *assumes* every batch is exactly `batch_size` — say, you hard-code the number 32 in a reshape — then on that last short batch it will either crash or quietly produce wrong numbers. That's the whole reason we name the ragged batch out loud now: so you write your loop to expect it.

#### Shuffle first, every time around
Before you slice, you **shuffle**. Here is the exact failure that forces you to.

**The cause.** Imagine the data sits in a fixed order. Many datasets are stored sorted by label — all the 0s first, then all the 1s, and so on. If you slice that without shuffling, each batch is almost entirely one digit. Consecutive batches are then *correlated*: one batch pulls the weights hard toward one class, then the next batch yanks them back the other way. The model can even start to learn the *order* of the data instead of the digits themselves.

**The remedy.** You [[shuffle||Randomly reorder the training examples (shuffle their indices) at the start of every epoch, so each batch is a fresh, well-mixed sample of classes and the model never sees the same batch composition or ordering twice.]] the data at the start of *every* epoch. Then each batch is a fair, freshly mixed sample of all the classes.

!!! c-warn ⚠️
<b>Failure mode (silent): forgetting to shuffle sorted data.</b> Nothing crashes. The loss just drifts down in a jerky way, and test accuracy stays poor — because each batch yanks the weights back and forth toward one class at a time.
!!!

!!! c-info 🛠️
<b>How a senior engineer reads this:</b> a suspiciously jerky loss on data that happens to be sorted is a <b>data-ordering smell</b>, not a learning-rate problem. The fix is to shuffle every epoch — not to fiddle with the step size.
!!!

@@@ concept id=c6 tag="Mean not sum" title="Average, don't add — so batch_size isn't a hidden step-size" gotit="Got mean vs sum"
Here is a subtle bug that averaging quietly prevents. When you combine the loss and the gradient over a batch, you **average** them (take the mean) rather than **adding** them up.

Why does one little choice matter so much? Because of how it connects to the [[learning rate||A small positive number that scales each update: param = param − learning_rate × gradient. It is applied once per batch update and sets the step size. Its "right" value is coupled to whether you mean or sum the batch.]] — the step-size knob you apply once per update.

%%% svg
<svg viewBox="0 0 520 155" role="img" aria-label="Two boxes. Left, SUM: eight example gradients add up to a big total gradient, and the update step is huge and scales with batch size — bad. Right, MEAN: the same eight gradients average to a normal-sized gradient, and the update step stays the same size no matter the batch size — good."><g font-family="monospace" font-size="11"><rect x="18" y="20" width="230" height="118" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.8"/><text x="30" y="42" fill="#C93B3B" font-weight="bold">SUM the 8 gradients</text><text x="30" y="62" fill="#6B645E" font-size="10">total = g1+g2+…+g8  (≈ 8× one)</text><text x="30" y="80" fill="#6B645E" font-size="10">step = lr × (big total)</text><text x="30" y="102" fill="#C93B3B" font-size="10">→ step grows with batch_size</text><text x="30" y="120" fill="#C93B3B" font-size="10">→ change batch → step size jumps</text><rect x="272" y="20" width="230" height="118" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="284" y="42" fill="#1a5c38" font-weight="bold">MEAN the 8 gradients</text><text x="284" y="62" fill="#6B645E" font-size="10">avg = (g1+…+g8) / 8  (normal size)</text><text x="284" y="80" fill="#6B645E" font-size="10">step = lr × (average)</text><text x="284" y="102" fill="#1a5c38" font-size="10">→ step size independent of batch</text><text x="284" y="120" fill="#1a5c38" font-size="10">→ change batch → lr still works</text></g></svg>
%%%

**Think of grading a stack of tests to decide how hard tomorrow's lesson should be.** Suppose you *add up* everyone's error. Then a class of 40 looks 40 times "more wrong" than a class of 1 — even if each student did equally well — and you'd wildly over-correct for the big class. Now suppose you take the *average* error instead. Class size no longer distorts your read, so the same teaching adjustment works for a class of 1 or a class of 40.

**What the grading picture gets right:** averaging strips the *size* of the group out of the signal, so your reaction depends on *how wrong*, not *how many*.

**Where it breaks down:** a teacher naturally thinks in averages. In code, though, the sum is often the *easier* thing to type. So this is a bug you have to choose to avoid — it does not avoid itself.

#### Why this connects to the learning rate
Let's see the connection slowly.

If you **add up** the gradients, the total grows with `batch_size`. So your real step becomes `learning_rate × (about batch_size × one gradient)`. The step now secretly scales with the batch. Double the batch, and you've silently doubled your step size — and a learning rate that felt fine before now overshoots.

If you **average** instead, the gradient stays a normal size no matter the `batch_size`. So the same learning rate keeps working even when you change the batch.

Here's the rule worth keeping: **take the mean of the per-example loss and gradient over the batch.** Do that, and the learning rate stays a clean, separate knob you can tune on its own.

@@@ concept id=c7 tag="Watch it fall" title="Track the loss — the proof the loop is learning" gotit="Got the loss curve"
So how do you actually *know* the loop is working? You **watch the training loss**.

After each step — or each epoch — you print the loss and look at the trend. If the loop is wired right and the learning rate is sane, the training loss should **head downward**. It'll be bumpy from batch to batch (that's the mini-batch shakiness we talked about), but overall it trends down.

This moment is a big deal. It is your first live proof that all the machinery you built across Days 1 and 2 actually connects and learns.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A loss-versus-step line chart. The line starts high on the left, is jagged from batch noise, and trends clearly downward toward a low flat region on the right, showing the training loss falling as steps accumulate."><g font-family="monospace" font-size="10"><line x1="50" y1="20" x2="50" y2="128" stroke="#6B645E" stroke-width="1"/><line x1="50" y1="128" x2="500" y2="128" stroke="#6B645E" stroke-width="1"/><text x="16" y="28" fill="#6B645E">loss</text><text x="470" y="146" fill="#6B645E">step →</text><polyline points="55,32 75,40 95,36 115,55 135,50 160,68 185,64 210,80 240,78 270,92 300,90 330,102 365,100 400,110 440,108 490,114" fill="none" stroke="#2D8B55" stroke-width="2"/><circle cx="55" cy="32" r="3" fill="#C93B3B"/><text x="90" y="24" fill="#C93B3B">high at start (untrained)</text><circle cx="490" cy="114" r="3" fill="#1a5c38"/><text x="360" y="152" fill="#1a5c38">trending down = it's learning</text></g></svg>
%%%

**Think of watching your resting heart rate drop as you get fitter over weeks.** Day to day it bounces around — you slept badly, you had coffee. But plot it over weeks and the trend is clearly down. You don't judge your fitness from one morning's reading. You judge it from the trend.

**What the heart-rate picture gets right:** a single noisy reading (one batch's loss) tells you almost nothing, but the *trend* across many readings is a signal you can trust.

**Where it breaks down:** your heart rate can only drop so far, and then you're genuinely healthy. A training loss can keep dropping toward zero even while the model is quietly getting *worse* at its real job. That twist is exactly what the last concept is about.

#### What "down" and "not down" each tell you
Two quick readings you'll do again and again:

- **Loss trending down** — the loop is connected and learning. Good sign.
- **Loss flat or climbing** — something is wrong. Often it's the learning rate (too big and it climbs or blows up; far too small and it barely budges). Sometimes it's a wiring bug, or forgotten shuffling.

The loss curve is the very first place an experienced engineer looks when training misbehaves. You now know how to read it too.

@@@ concept id=c8 tag="The ceiling" title="A falling loss is not a good model" gotit="Got the ceiling"
We'll end on the most honest — and most important — idea in the lesson. So slow down here.

You now have a loop whose training loss falls beautifully. It is so tempting to declare victory. Please don't, not yet. The mini-batch loop is an **optimization procedure** — a machine for driving the *training* loss down. And that is *all* it is. It is **not** a promise that your model is any good. A perfectly-wired loop with a low training loss can still be a bad model.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="Two loss curves over epochs. The training loss (green) falls steadily toward zero. The unseen-data loss (red, dashed) falls at first, then turns and rises — the gap between them widening, showing that a low training loss does not mean the model is good on new data."><g font-family="monospace" font-size="10"><line x1="50" y1="18" x2="50" y2="126" stroke="#6B645E" stroke-width="1"/><line x1="50" y1="126" x2="500" y2="126" stroke="#6B645E" stroke-width="1"/><text x="16" y="26" fill="#6B645E">loss</text><text x="450" y="144" fill="#6B645E">epochs →</text><polyline points="55,40 110,60 170,78 240,92 320,104 410,112 490,118" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="330" y="112" fill="#1a5c38">training loss ↓ (looks great)</text><polyline points="55,44 110,58 170,66 240,64 320,74 410,92 490,108" fill="none" stroke="#C93B3B" stroke-width="2" stroke-dasharray="5 3"/><text x="200" y="52" fill="#C93B3B">loss on unseen data ↑ (the truth)</text><text x="150" y="158" fill="#6B645E">low training loss ≠ good model — the gap is what matters</text></g></svg>
%%%

**Think of a student who memorizes last year's exact exam answers.** On last year's paper they score 100% — their "training loss" is zero. But hand them *this* year's paper and they flounder, because they learned the answer sheet, not the subject.

**What the memorizing-student picture gets right:** scoring perfectly on the material you *practiced* says nothing about material you haven't seen — and "haven't seen" is what actually matters.

**Where it breaks down:** a student at least knows they're cramming. A model has no idea whether it's memorizing or truly understanding. The *only* way to tell is to test it on data it never trained on — and that's a tool you don't have yet. You'll build it tomorrow.

#### What this limit points to next
Let's be very clear about the takeaway, because it's the one to carry forward. A low training loss proves your loop *runs*. It does not prove your model is *good*. The model may be **overfitting** — memorizing the training set instead of learning the general pattern.

To even *measure* whether that's happening, you need data the model never trained on (a held-out split). And then you need techniques to fix it when it does happen. That is exactly where you go next.

!!! c-info 🔗
<b>Forward pointer:</b> Day 4 (<b>Full Training: Loss Curve & Dropout</b>) introduces the train/validation split — data the model never trains on — so you can finally *see* the gap in the picture above, plus dropout and weight decay to shrink it. Day 5 swaps your hand-written loop for PyTorch's <code>DataLoader</code> (automatic shuffling + batching) and <code>torch.optim.SGD</code> (the framework's version of your manual update). The loop you built today is the very thing all of that automates.
!!!

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each. If one trips you up, that's just a cue to scroll back up — not a grade, and definitely not a failure. Getting one wrong is how the ideas actually stick.
%%% quiz
q: You have N = 60,000 images and batch_size = 32. How many weight updates (iterations) happen in ONE epoch? | a:2 | 60,000 | 32 | 1,875 | 10 | fb: One epoch is one full pass, and each update uses one batch: 60,000 / 32 = 1,875 iterations per epoch. An epoch is many iterations, not one.
q: Why is mini-batch the default instead of full-batch (all N) or single-example (batch of 1)? | a:1 | It changes the forward/backward math | It averages a handful into a steadier gradient than one example, while updating far more often than full-batch | It needs the most memory | It removes the need to shuffle | fb: Full-batch is exact but gives one slow update per pass; single-example updates constantly but shakily. A mini-batch averages about 32 examples: steadier than one, far more frequent than the whole set — best of both, and the math is unchanged.
q: Your data file is sorted by label (all 0s, then all 1s…). You slice into batches but forget to shuffle. What happens? | a:2 | The code crashes right away | Nothing — order never matters | Each batch is almost one class, so training is jerky and generalizes poorly, with no error thrown | The loss goes negative | fb: Fixed sorted order makes consecutive batches correlated — each is one digit and yanks the weights back and forth. It's a silent bug: no crash, just poor learning. The remedy is to shuffle the indices every epoch.
q: Your training loss has fallen almost to zero. What has this PROVEN? | a:2 | The model will do great on new data | Training is complete and correct | Nothing about new data — only that the loop optimizes the training loss; the model may be overfitting | The learning rate is perfect | fb: The loop is an optimizer, not a promise of correctness. A low TRAINING loss can hide a model that just memorized the training set. You can only tell by testing on held-out data — which is Day 4.
%%%

@@@ produce id=produce tag="Produce" title="Predict the loss trend, then run the loop and watch it fall" gotit="Done"
Now it's your turn to build the loop with your own hands. Wrapping yesterday's single update inside a real loop is what makes it truly stick.

**Before you write any code, make three predictions.** Guessing first is what turns "I watched it" into "I understand it," so don't skip this:

1. With `N = 60,000` and `batch_size = 32`, how many iterations will one epoch print?
2. Will the per-batch training loss fall *smoothly* or *bumpily* as the loop runs?
3. If you train on data sorted by label and *skip* shuffling, will the loss look smoother or jerkier than the shuffled run?

Write your three guesses down. Then run it and **watch** whether you were right. Pick one path.

#### Option A · write it yourself
Create `sessions/m04-first-model-mlp/day-03-minibatch-loop/experiment.py`. Wrap your Day 1–2 MLP in a mini-batch loop. Each epoch: **shuffle** the training indices, **slice** into batches of 32 (and handle the ragged final batch), and per batch run forward → loss → backward → update — **averaging** (mean, not sum) the loss and gradient over the batch. Print the number of iterations per epoch and the average loss per epoch, and confirm it trends down. Then, as a contrast, run one epoch on label-sorted data **without** shuffling and watch the jerkier loss. Run with `python3 sessions/m04-first-model-mlp/day-03-minibatch-loop/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5 Day 3 artifact.

Create sessions/m04-first-model-mlp/day-03-minibatch-loop/experiment.py that, with a comment on each step, wraps the Day 1–2 MNIST MLP (NumPy) in a mini-batch training loop:
1. Print the vocabulary as a sanity check: N_train, batch_size, and iterations_per_epoch = N_train // batch_size (and mention the ragged final batch when N is not divisible).
2. Each epoch: shuffle the training indices, slice into batches of 32 INCLUDING a final ragged batch, and per batch run forward -> compute loss -> backward -> update. AVERAGE (mean, not sum) the loss and gradient over the batch so the step size does not scale with batch_size.
3. Track and print the average training loss per epoch for several epochs; confirm it trends downward.
4. Contrast run: on label-sorted data, run one epoch WITHOUT shuffling and show the loss is jerkier than the shuffled run.
5. End with a printed reminder that a low TRAINING loss does not prove the model generalizes — that needs held-out data (Day 4).
Then run it and paste the per-epoch losses (shuffled vs un-shuffled) at the bottom as a comment.
%%%

#### What you should see (check your prediction)
Here's what to look for when it runs:

- **Iterations per epoch:** `60,000 / 32 = 1,875`. Did your count match?
- **The shape of the loss:** the per-batch loss is **bumpy** (that's the mini-batch shakiness), but the per-epoch average **trends down** — your live proof the loop learns. Did you predict bumpy-but-downward?
- **The un-shuffled run:** on label-sorted data with no shuffle, the loss is visibly **jerkier** and learns worse than the shuffled run. Did you guess which way it would go?
- **The thing worth sitting with:** the loss fell, and that only proves the *loop* works. Whether the *model* is any good is a question this loop simply cannot answer. That's what held-out data in Day 4 is for.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m04-first-model-mlp/day-03-minibatch-loop/log.md`: (1) epoch vs iteration vs batch, one sentence each; (2) why mini-batch beats both full-batch and single-example; (3) the one thing a falling training loss does NOT prove.
!!!

@@@ fin
