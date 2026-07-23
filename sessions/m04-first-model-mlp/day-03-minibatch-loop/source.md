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
nav_next_label: "Full Training: Loss Curve &amp; Dropout"
fin_title: "Module 5 · Day 3 complete! 🏆"
fin_body: "You did it! You finished <b>The Mini-Batch Loop</b>. Look at what you just built: you wrapped yesterday's single correction inside a loop that runs forward → loss → backward → update, over batch after batch, epoch after epoch — and you watched the training loss actually fall. You can now use the real words (epoch, iteration, batch, batch_size), you know why a small handful beats both one example and the whole set, why you shuffle every epoch, why you average instead of add, and — the big one — that a falling training loss proves the loop works, not that the model is good.<br>Take a breath: that is the whole heartbeat of how every neural network on Earth learns. Next up: <b>Full Training: Loss Curve &amp; Dropout</b> — where you split off some held-out data and finally get to <i>measure</i> whether your model truly learned."
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
  - {topic: learning rate as the per-update step size, keywords: [learning rate, step size, once per update, param minus learning_rate times gradient, mean vs sum]}
  - {topic: learning rate too large diverges, keywords: [too large, overshoot, oscillates, diverges, nan, lower the learning rate]}
  - {topic: learning rate too small crawls, keywords: [too small, barely moves, crawls, raise the learning rate]}
  - {topic: gradient noise makes the per-batch loss jumpy, keywords: [bumpy, jagged, shakiness, noisy, trend down, larger batch]}
  - {topic: forgetting to zero or reset gradients, keywords: [zero the gradients, reset every, pile up, accumulate, every iteration, zero_grad]}
  - {topic: capability limit low training loss does not prove a good model, keywords: [capability limit, optimization procedure, not a correctness guarantee, overfit, held-out, low training loss does not prove]}
require_artifact: true
---

@@@ hero
@lede Picture learning to shoot a **basketball free-throw**. You don't fix your shot by staring at the hoop for an hour and then taking one perfect shot. You grab a small **handful** of balls, shoot them, look at where they landed — "a little left" — and nudge your aim a hair to the right. Grab the next handful, shoot, look, nudge. Handful after handful, your aim creeps toward the center. That exact rhythm — *try a small handful, measure the miss, nudge, repeat* — is the whole secret of how a neural network learns. It has a name: the **mini-batch training loop**, and it is the engine humming under every model you've heard of, from the tiny network you'll train today all the way up to ChatGPT — every one of them learned by practicing on small handfuls, over and over. Yesterday you built the two pieces: a forward pass that guesses, a backward pass that says exactly how to fix each weight. Today those pieces snap into one repeating cycle, and for the first time you get to press "go" and watch a single number — the loss — actually fall.
@goal Here's what we'll figure out together today, one small idea at a time:
- the **training loop** — the four steps you already know, now put on repeat;
- the three words for its rhythm — **epoch**, **iteration**, **batch** (and `batch_size`);
- why a small **handful** beats both one example and the whole dataset;
- why you **shuffle** every time around, and why you **average** instead of add;
- and the one honest thing a falling training loss does *not* prove.

@@@ concept id=c1 tag="The loop" title="The training loop: the same four steps, on repeat" gotit="Got the loop"
Practice is never a single act. It is a *loop*. Yesterday you did one correction. Today you put that correction inside a repeat — and that is the whole trick.

**Back to the free-throw court.** You shoot (that's a *guess*). You see how far you missed (that's the *loss*). You figure out what to change — too much wrist, aim a bit left (that's *backward*, which way to fix it). You adjust your form (that's the *update*). Then you shoot again, and again. One shot teaches you almost nothing. A thousand shots, each one corrected, teaches you the game. Here is that practice loop drawn out:

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A basketball free-throw practice loop drawn as four stations in a ring: a player shooting, a ball landing left of the hoop, a thought bubble aim a bit right, and the player adjusting their form, with a return arrow labeled shoot again"><g font-family="monospace" font-size="10" text-anchor="middle"><circle cx="120" cy="55" r="30" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="120" y="52">🏀</text><text x="120" y="66" font-size="9" fill="#1a5c38">shoot</text><text x="200" y="52" fill="#6B645E">→</text><circle cx="290" cy="55" r="30" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.8"/><text x="290" y="50" font-size="9" fill="#8A6D3B">missed</text><text x="290" y="63" font-size="9" fill="#8A6D3B">left!</text><text x="360" y="52" fill="#6B645E">→</text><circle cx="430" cy="55" r="30" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.8"/><text x="430" y="50" font-size="9" fill="#C93B3B">aim a</text><text x="430" y="63" font-size="9" fill="#C93B3B">bit right</text><circle cx="290" cy="130" r="30" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="290" y="127" font-size="9" fill="#5E5191">adjust</text><text x="290" y="140" font-size="9" fill="#5E5191">form</text><path d="M430 82 Q400 130 322 130" fill="none" stroke="#7C6DAA" stroke-width="1.5"/><path d="M258 130 Q160 130 132 82" fill="none" stroke="#7C6DAA" stroke-width="1.5" stroke-dasharray="4 3"/><text x="180" y="120" fill="#6B645E" font-size="9">shoot again ↺</text><text x="260" y="172" fill="#6B645E">get good from the loop — not from any one shot</text></g></svg>
%%%

**What the free-throw picture gets right:** you get better from the *loop* — the same four beats, repeated shot after shot — not from any single shot. **Where it breaks down:** a shooter fixes their form after every single shot; our model, as you'll see next, waits for a small *handful* of shots first, because a handful gives a steadier read on what to change.

#### The same loop, in network words
Now the loop itself, with the real names. The [[training loop||Repeating four steps over the data — forward pass, compute the loss, backward pass, update the weights — until the model has learned. It is what turns a static forward+backward network into a model that actually learns.]] is those same four steps you already built, run again and again:

- **forward pass** — the model guesses;
- **compute the loss** — how wrong the guess was;
- **backward pass** — which way to fix each weight;
- **update** — take one small step in that direction.

Then you do it again on the next slice of data. That repeat is the whole thing that turns a frozen network into one that learns.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A circular loop of four boxes: forward pass, compute loss, backward pass, update, with an arrow returning from update back to forward pass to show the loop repeats over the data"><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="30" y="60" width="90" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="75" y="78" fill="#1a5c38" font-weight="bold">forward</text><text x="75" y="92" fill="#6B645E" font-size="9">guess</text><text x="128" y="84" fill="#6B645E">→</text><rect x="146" y="60" width="90" height="40" rx="6" fill="#FDF3E8" stroke="#C99A12" stroke-width="2"/><text x="191" y="78" fill="#8A6D3B" font-weight="bold">loss</text><text x="191" y="92" fill="#6B645E" font-size="9">how wrong</text><text x="244" y="84" fill="#6B645E">→</text><rect x="262" y="60" width="90" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="307" y="78" fill="#C93B3B" font-weight="bold">backward</text><text x="307" y="92" fill="#6B645E" font-size="9">which way</text><text x="360" y="84" fill="#6B645E">→</text><rect x="378" y="60" width="90" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="423" y="78" fill="#5E5191" font-weight="bold">update</text><text x="423" y="92" fill="#6B645E" font-size="9">take step</text><path d="M423 100 L423 132 L75 132 L75 100" fill="none" stroke="#7C6DAA" stroke-width="1.5" stroke-dasharray="4 3"/><text x="249" y="148" fill="#6B645E" font-size="10">repeat on the next slice of data — that loop is what learns</text></g></svg>
%%%

#### The loop is the mechanism — everything else is a dial on it
Practice with correction is the spine of this whole module. Day 1 gave you the guess. Day 2 gave you the correction. Today's new idea is small but it is the whole game: *put it in a loop over the data.* Nothing about the forward or backward math changes. Everything else in this lesson — batches, shuffling, the learning rate — is just a knob you turn on this one loop.

And that is your first win of the day. You just learned the training loop — the single idea that turns a still, frozen network into one that learns. That is the heart of everything that follows. Everything from here is detail hung on this one loop.

@@@ concept id=c2 tag="The words" title="Epoch, iteration, batch: three words for three sizes of a step" gotit="Got the vocabulary"
The loop runs a *lot* of times, so the field has three precise words for its rhythm. Before we name them, feel them.

**Think of reading a textbook to study for a big exam.** You read a small group of pages in one *sitting*. When you finish that sitting, you jot a note and adjust your understanding a little — that's one *nudge*. Reading the *whole book* cover to cover, once, is one full *read-through*. And of course you'll read the whole book several times before the exam. Three natural sizes: a sitting, a nudge per sitting, a full read-through — and a full read-through is just many sittings stacked up.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A textbook study analogy drawn as a stack of pages. A small group of pages is bracketed and labeled one sitting equals a batch. A pencil making one note after the sitting is labeled one nudge equals an iteration. A bracket around the whole book is labeled cover-to-cover once equals an epoch."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="18" fill="#6B645E" font-size="11">📖 one textbook = the whole training set</text><rect x="40" y="34" width="26" height="40" rx="2" fill="#FDF3E8" stroke="#C99A12" stroke-width="2"/><rect x="70" y="34" width="26" height="40" rx="2" fill="#FDF3E8" stroke="#C99A12" stroke-width="2"/><rect x="100" y="34" width="26" height="40" rx="2" fill="#FDF3E8" stroke="#C99A12" stroke-width="2"/><path d="M40 80 L126 80" stroke="#8A6D3B" stroke-width="1.5"/><text x="83" y="94" fill="#8A6D3B" font-weight="bold" font-size="9">one sitting</text><text x="83" y="106" fill="#8A6D3B" font-size="9">(a few pages)</text><text x="150" y="58" fill="#5E5191">✏️</text><path d="M166 54 L192 54" stroke="#5E5191" stroke-width="1.4"/><text x="210" y="50" fill="#5E5191" font-weight="bold" font-size="9" text-anchor="start">one note after the sitting</text><text x="210" y="63" fill="#6B645E" font-size="9" text-anchor="start">= one nudge to what you know</text><rect x="300" y="34" width="20" height="40" rx="2" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="322" y="34" width="20" height="40" rx="2" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="344" y="34" width="20" height="40" rx="2" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="366" y="34" width="20" height="40" rx="2" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="388" y="34" width="20" height="40" rx="2" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="410" y="34" width="20" height="40" rx="2" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="432" y="34" width="20" height="40" rx="2" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="454" y="34" width="20" height="40" rx="2" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><path d="M300 80 L474 80" stroke="#2D8B55" stroke-width="1.5"/><text x="387" y="94" fill="#1a5c38" font-weight="bold" font-size="9">whole book, once</text><text x="387" y="106" fill="#1a5c38" font-size="9">= one read-through</text><text x="260" y="132" fill="#6B645E">a read-through is just many sittings stacked up</text><text x="260" y="152" fill="#6B645E" font-size="9">and you read the whole book several times before the exam</text></g></svg>
%%%

**What the textbook picture gets right:** the three sizes are just three sizes of progress — a sitting, a note per sitting, a full read-through — and a full read-through is made of many sittings.

**Where it breaks down:** a reader remembers pages between sittings. Our model only ever "remembers" through its weights, so each sitting's only lasting effect is the small nudge it makes to those numbers.

#### Now the real names
Here are the three words the whole field uses. Match each one to the picture you just built:

- A [[batch||A small group of examples processed together before one weight update. The number of examples in it is the batch_size.]] (also called a **mini-batch**) is the small handful of examples the model looks at together before it updates once — the *sitting*.
- An [[iteration||Also called a step: one parameter update on one batch. Distinct from an epoch — there are many iterations in one epoch.]] — also called a **step** — is *one* weight update on *one* batch — the *note* after a sitting.
- An [[epoch||One full pass over the entire training dataset. The loop runs for many epochs.]] is one full pass over the *entire* dataset — the *cover-to-cover read-through*.

Don't worry if these blur together at first — everyone mixes them up in the beginning. The one link that keeps them straight: **one epoch contains many iterations.**

#### One worked count, so the words stick
Say you have `N = 60,000` training images and you pick `batch_size = 32`. Before you read on, *predict*: how many weight updates happen in one epoch? Then run it and see:

%%% demo id=count label="run it — count the steps"
code: N = 60_000; batch_size = 32
code: iters_per_epoch = N // batch_size
code: print('iterations per epoch:', iters_per_epoch)
code: print('total steps over 10 epochs:', 10 * iters_per_epoch)
out: iterations per epoch: 1875
out: total steps over 10 epochs: 18750
take: <b>One epoch = 1,875 weight updates</b> (60,000 ÷ 32). Ten epochs = 18,750 tiny nudges. So "we trained for 10 epochs" quietly means <i>nudge every weight 18,750 times</i> — that's the sheer repetition that makes learning happen.
%%%

Keep the three words distinct and you're set:

- **epoch** = one full pass;
- **iteration/step** = one update on one batch;
- **batch** = the handful, `batch_size` = how many are in it.

See? Not so bad once they sit still. You just picked up the exact vocabulary the whole field uses to describe training. From here on, when someone says "we trained for 10 epochs," you'll know exactly what they mean.

@@@ concept id=c3 tag="Two extremes" title="The two ancestors: the whole set, or one example at a time" gotit="Got the two poles"
Why a *handful*? Why not the whole dataset, or just one example? Great question — and the best way to answer it is to feel the two extremes first.

**Think of adjusting a big pot of soup you're cooking for a crowd.** One way: taste the *entire* pot — stir it all, take one big careful spoonful of the whole thing — before you change a single thing. That gives you the most accurate read possible of how the soup really tastes. But you can only adjust once per full stir, and stirring a giant pot is slow. The other way: taste *one* little spoonful and re-season off just that. That's fast and constant — but one spoonful can fool you (maybe you hit a salty spot near the edge).

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A soup-tasting analogy. Left, a big pot with a big ladle tasting the whole pot: labeled taste everything, most accurate, but slow, one adjustment per stir. Right, a tiny spoon lifting one drop: labeled taste one spoonful, fast and constant, but one spoonful can fool you."><g font-family="monospace" font-size="10" text-anchor="middle"><path d="M60 60 Q60 118 130 118 Q200 118 200 60 Z" fill="#FDF3E8" stroke="#C99A12" stroke-width="2"/><ellipse cx="130" cy="60" rx="70" ry="14" fill="#F6E4C4" stroke="#C99A12" stroke-width="2"/><path d="M130 40 L130 60" stroke="#8A6D3B" stroke-width="3"/><ellipse cx="130" cy="40" rx="12" ry="5" fill="#8A6D3B"/><text x="130" y="140" fill="#8A6D3B" font-weight="bold">taste the WHOLE pot</text><text x="130" y="155" fill="#6B645E" font-size="9">most accurate · but slow</text><text x="130" y="167" fill="#6B645E" font-size="9">one adjust per full stir</text><path d="M370 66 Q370 116 410 116 Q450 116 450 66 Z" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.6"/><ellipse cx="410" cy="66" rx="40" ry="9" fill="#F6E4C4" stroke="#C99A12" stroke-width="1.6"/><path d="M410 50 L410 66" stroke="#6B645E" stroke-width="1.6"/><circle cx="410" cy="48" r="4" fill="#2A7B9B"/><text x="410" y="140" fill="#8A6D3B" font-weight="bold">taste ONE spoonful</text><text x="410" y="155" fill="#6B645E" font-size="9">fast + constant · but jumpy</text><text x="410" y="167" fill="#6B645E" font-size="9">one spoonful can fool you</text><text x="285" y="90" fill="#6B645E" font-size="14">vs</text></g></svg>
%%%

**What the cooking picture gets right:** taste more at once and your read is more accurate but slower; taste less and it's faster but jumpier. That trade-off is exactly the one the two ancestors of modern training are stuck between.

**Where it breaks down:** you re-season a real pot on your tongue's hunch. The model re-seasons using an exact, averaged gradient over whatever it tasted — so the "noise" here is a precise, measurable thing, not a guess.

#### The two ancestors, by name
Those two ways of tasting are the two historical ancestors of how a network learns. They are what people tried first, and their weaknesses are exactly why the modern default is a *handful*.

- The "taste everything" way is [[full-batch gradient descent||Compute the gradient over all N examples, then take a single update. The exact, smooth gradient every step — but only one update per full pass, and it is slow and memory-heavy on large data.]]: use *all* `N` examples to work out one exact gradient, then take a single step.
- The "taste one spoonful" way is single-example [[stochastic gradient descent||The extreme where batch_size = 1: update the weights after every single example. Very frequent updates, but each gradient is a noisy one-example estimate.]] (the original SGD): update after *every single* example, one at a time.

#### Why each extreme hurts — see the two kinds of step
Now let's *see* what each pole costs. Both are walking downhill into the same valley of loss — but they walk very differently. Full-batch takes one big, precise, expensive step. Single-example takes a swarm of tiny, cheap, jittery steps that wander on the way down:

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="Two panels of the same U-shaped loss valley. Left, full-batch: one single long precise arrow straight to the bottom, labeled one exact step but slow and only one per pass. Right, single-example SGD: many tiny jittery zig-zag steps wandering down to the bottom, labeled many cheap steps but noisy and wandering."><g font-family="monospace" font-size="9" text-anchor="middle"><path d="M24 28 Q135 150 246 28" fill="none" stroke="#B8B0C8" stroke-width="2"/><circle cx="135" cy="122" r="4" fill="#1a5c38"/><text x="135" y="118" fill="#1a5c38" font-size="8">min</text><path d="M52 44 L131 118" fill="none" stroke="#2D8B55" stroke-width="3" marker-end="url(#ar1)"/><circle cx="52" cy="44" r="4" fill="#2D8B55"/><defs><marker id="ar1" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#2D8B55"/></marker><marker id="ar2" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#C99A12"/></marker></defs><text x="135" y="166" fill="#1a5c38" font-weight="bold">full-batch</text><text x="135" y="180" fill="#6B645E">ONE exact step · slow · 1 per pass</text><path d="M274 28 Q385 150 496 28" fill="none" stroke="#B8B0C8" stroke-width="2"/><circle cx="385" cy="122" r="4" fill="#1a5c38"/><text x="385" y="118" fill="#1a5c38" font-size="8">min</text><path d="M300 48 L322 66 L312 80 L340 84 L328 98 L356 100 L346 112 L372 114 L363 120 L385 122" fill="none" stroke="#C99A12" stroke-width="1.6" marker-end="url(#ar2)"/><circle cx="300" cy="48" r="3.5" fill="#C99A12"/><text x="385" y="166" fill="#8A6D3B" font-weight="bold">single-example SGD</text><text x="385" y="180" fill="#6B645E">MANY cheap steps · noisy · wandering</text></g></svg>
%%%

Read the two pictures side by side:

- **Full-batch** (left) gives you a *smooth, exact* gradient — one long arrow pointing straight at the bottom. That sounds ideal. The catch: you get only **one update per full pass** over the data. On 60,000 images, that is one single step per epoch. And you must hold all `N` examples' computations in memory at once, which is heavy. Great in theory, painfully slow to actually train.
- **Single-example** (right) gives you `N` updates per epoch — a whole swarm of tiny steps, very frequent. The catch: each gradient comes from just *one* example, so it is **noisy** and zig-zags all over. Progress is jittery, and it can wander a long way sideways on its way down.

So neither pole wins. Full-batch is too slow; single-example is too jumpy. The answer is to sit *between* them — and that is exactly the next idea.

@@@ concept id=c4 tag="The sweet spot" title="Mini-batch: average a handful, then take one step" gotit="Got the mini-batch"
So how do you get most of full-batch's accuracy *and* most of single-example's speed? You feel it first.

**Think of trying to find out how warm a swimming pool is.** Ask *one* friend and you get a fast but shaky answer — "feels cold!" — but that friend might have just stepped in from the cold air. Ask *all 500* people at the pool and average them, and you get a rock-solid number — but polling 500 people takes forever. Now the trick: ask a *handful* of about 32 people, average their answers, and act once. You land on a number almost as steady as polling everyone, for a tiny fraction of the effort.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A pool-temperature analogy. Left, ask one swimmer: a fast but shaky single guess. Middle, ask a handful of about 32 and average: a steady answer for little effort, highlighted as the sweet spot. Right, ask all 500 and average: rock-solid but very slow."><g font-family="monospace" font-size="10" text-anchor="middle"><rect x="20" y="34" width="150" height="96" rx="8" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.6"/><text x="95" y="58" font-size="20">🏊</text><text x="95" y="86" fill="#8A6D3B" font-weight="bold" font-size="9">ask 1 swimmer</text><text x="95" y="102" fill="#6B645E" font-size="9">fast · but shaky</text><text x="95" y="116" fill="#6B645E" font-size="9">"feels cold!" (maybe)</text><rect x="185" y="28" width="150" height="108" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.6"/><text x="260" y="54" font-size="16">🏊🏊🏊</text><text x="260" y="82" fill="#1a5c38" font-weight="bold" font-size="9">ask a handful (~32)</text><text x="260" y="98" fill="#1a5c38" font-weight="bold" font-size="9">and AVERAGE</text><text x="260" y="114" fill="#6B645E" font-size="9">steady · cheap</text><text x="260" y="127" fill="#2D8B55" font-weight="bold" font-size="9">the sweet spot ★</text><rect x="350" y="34" width="150" height="96" rx="8" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.6"/><text x="425" y="58" font-size="14">🏊🏊🏊🏊🏊</text><text x="425" y="86" fill="#5E5191" font-weight="bold" font-size="9">ask all 500 · average</text><text x="425" y="102" fill="#6B645E" font-size="9">rock-solid</text><text x="425" y="116" fill="#6B645E" font-size="9">but very slow</text></g></svg>
%%%

**What the pool picture gets right:** averaging a small sample kills most of the noise of a single opinion while staying cheap. That is exactly what averaging 32 gradients does.

**Where it breaks down:** pool-guessers are estimating one number. Each of your 32 examples produces a full gradient over about 100,000 weights, and you average all of those, weight by weight, into one step.

#### The name, and the recipe
That "ask a handful and average" middle is [[Minibatch gradient descent||Average the gradient over a small batch of examples, then take one step. The modern default that sits between full-batch (exact but slow) and single-example (frequent but noisy).]] — the compromise between the two poles, and the modern default. Here is the recipe:

- take a small handful — say 32 examples;
- run the forward and backward pass on all of them;
- **average** their gradients into one;
- take a single step with that average;
- grab the next handful and repeat.

The payoff: you get *many* updates per epoch (like single-example), but each update rests on 32 examples of evidence instead of 1 (much steadier). You quietly get the best of both poles. Drag the slider below and watch each step follow the (averaged) gradient downhill:

%%% svg
<svg id="mb-svg" viewBox="0 0 520 210" role="img" aria-label="Interactive mini-batch averaging. Each training example gives its own noisy gradient arrow, all fanning out from a common start. Drag the batch-size slider to average more of them into one bold green step; with a bigger batch the wobble cancels and the averaged step lines up with the true downhill direction (dashed)."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Each example gives a noisy arrow → average a batch into one steady step</text><line id="mb-true" x1="150" y1="100" x2="202" y2="133" stroke="#B8AEA2" stroke-width="1.5" stroke-dasharray="5,4"/><text id="mb-truelbl" x="206" y="140" fill="#9A938A" font-size="9">true downhill</text><line id="mb-l0" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l1" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l2" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l3" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l4" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l5" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l6" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l7" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l8" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l9" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l10" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><line id="mb-l11" x1="150" y1="100" x2="150" y2="100" stroke="#D6CCB6" stroke-width="1.5"/><circle id="mb-d0" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d1" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d2" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d3" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d4" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d5" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d6" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d7" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d8" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d9" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d10" cx="150" cy="100" r="2" fill="#C9BEA8"/><circle id="mb-d11" cx="150" cy="100" r="2" fill="#C9BEA8"/><line id="mb-avg" x1="150" y1="100" x2="202" y2="133" stroke="#2D8B55" stroke-width="4"/><circle id="mb-avgtip" cx="202" cy="133" r="4" fill="#2D8B55"/><circle cx="150" cy="100" r="4" fill="#5E5191"/><text x="150" y="92" text-anchor="middle" fill="#5E5191" font-size="9">start</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label>batch size <input id="mb-k" type="range" min="1" max="12" step="1" value="4" style="width:52%;accent-color:#2D8B55;vertical-align:middle"></label>
<div id="mb-out" style="margin-top:6px;color:#2C2A28">batch = <b>4</b> examples → average 4 noisy arrows into <b>one</b> step. More examples → the wobble cancels more; the step steadies.</div>
</div>
<script>(function(){
  var sl=document.getElementById('mb-k');if(!sl)return;
  var out=document.getElementById('mb-out');
  var TRUE=[72,46];
  var J=[[40,25],[-40,-25],[35,-30],[-35,30],[15,40],[-15,-40],[45,-15],[-45,15],[25,38],[-25,-38],[42,-8],[-42,8]];
  var V=J.map(function(j){return [TRUE[0]+j[0],TRUE[1]+j[1]];});
  var OX=150,OY=100,S=0.72;
  for(var i=0;i<12;i++){
    var ex=OX+V[i][0]*S, ey=OY+V[i][1]*S;
    var ln=document.getElementById('mb-l'+i), dt=document.getElementById('mb-d'+i);
    if(ln){ln.setAttribute('x2',ex.toFixed(1));ln.setAttribute('y2',ey.toFixed(1));}
    if(dt){dt.setAttribute('cx',ex.toFixed(1));dt.setAttribute('cy',ey.toFixed(1));}
  }
  var avg=document.getElementById('mb-avg'), tip=document.getElementById('mb-avgtip');
  function paint(){
    var k=+sl.value, ax=0, ay=0;
    for(var i=0;i<12;i++){
      var ln=document.getElementById('mb-l'+i), dt=document.getElementById('mb-d'+i), on=i<k;
      if(ln)ln.setAttribute('opacity',on?'0.9':'0');
      if(dt)dt.setAttribute('opacity',on?'0.9':'0');
      if(on){ax+=V[i][0];ay+=V[i][1];}
    }
    ax/=k; ay/=k;
    var mx=OX+ax*S, my=OY+ay*S;
    avg.setAttribute('x2',mx.toFixed(1)); avg.setAttribute('y2',my.toFixed(1));
    tip.setAttribute('cx',mx.toFixed(1)); tip.setAttribute('cy',my.toFixed(1));
    var msg = k===1 ? 'One example is a wild guess — its arrow can point well off the true downhill.'
      : (k>=10 ? 'Big batch → the wobble cancels out; the averaged step lines up with the true downhill.'
      : 'More examples → the wobble cancels more; the averaged step steadies.');
    out.innerHTML='batch = <b>'+k+'</b> example'+(k>1?'s':'')+' → average '+k+' noisy arrow'+(k>1?'s':'')+' into <b>one</b> step. '+msg;
  }
  sl.addEventListener('input',paint);paint();
})();</script>
%%%

Nice — you just met the workhorse of modern training. Almost every network you'll ever hear about is trained this way. Now let's look at the one knob that controls it.

#### The batch_size dial: how shaky vs. how frequent your steps are
The one knob that sets where you sit between the two poles is `batch_size`. Turning it trades two things against each other: **how shaky** each step is versus **how frequent** your steps are.

%%% table
:: batch_size :: gradient :: updates per epoch :: memory :: note
small (e.g. 8) :: shakier estimate :: many :: light :: the shakiness can even *help* the model generalize
large (e.g. 512) :: smoother, more accurate :: few :: heavy :: fewer, steadier steps
%%%

That balance has a name — the [[gradient-noise trade-off||Small batch = a shakier (noisier) gradient estimate but more updates and often better generalization; large batch = a smoother, more accurate gradient but fewer updates and more memory. batch_size is the dial: how shaky versus how frequent your steps are.]], which is just a fancy way of saying "how shaky vs. how frequent your steps are."

A **small** batch gives a shakier gradient — but more frequent updates. Strangely, that shakiness often *helps* the model generalize: the little jiggles knock the weights out of bad spots. A **large** batch gives a smoother, more accurate gradient — but fewer updates, and it eats more memory. So `batch_size` is the dial you turn to trade one against the other. In practice, sizes of 32 to 256 are the common sweet spot.

@@@ concept id=c5 tag="Slice it up" title="Iterating in batches — and the ragged last one" gotit="Got the slicing"
Now the hands-on mechanics: how do you actually hand the loop one batch at a time? Picture it first.

**Think of packing eggs into cartons that hold a dozen.** You have a big tray of eggs and you fill one carton, then the next, then the next — each holds exactly twelve. But suppose you have 100 eggs. You fill eight full cartons (96 eggs), and the ninth carton has just 4 eggs rattling around in it. You don't throw those 4 eggs away, and you don't pretend the carton is full. You keep the short carton, exactly as it is.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="An egg-carton analogy. Several full cartons each holding a dozen eggs, then a short final carton holding only 4 eggs, labeled leftover carton, still kept, not thrown away."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="18" fill="#6B645E">100 eggs · cartons of a dozen</text><g fill="#F6E4C4" stroke="#C99A12"><rect x="30" y="34" width="96" height="34" rx="5" stroke-width="1.6"/><rect x="134" y="34" width="96" height="34" rx="5" stroke-width="1.6"/><rect x="238" y="34" width="96" height="34" rx="5" stroke-width="1.6"/></g><text x="78" y="55" fill="#8A6D3B">🥚×12</text><text x="182" y="55" fill="#8A6D3B">🥚×12</text><text x="286" y="55" fill="#8A6D3B">🥚×12</text><text x="360" y="55" fill="#6B645E" font-size="9">… more full cartons …</text><rect x="446" y="34" width="46" height="34" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2.4"/><text x="469" y="55" fill="#C93B3B" font-weight="bold">×4</text><text x="469" y="90" fill="#C93B3B" font-size="9">leftover</text><text x="469" y="101" fill="#C93B3B" font-size="9">carton</text><text x="240" y="118" fill="#6B645E">keep the short carton — the 4 eggs still count</text></g></svg>
%%%

**What the egg-carton picture gets right:** the leftover group is real, it is smaller than a full carton, and it still counts — you handle it just as it is.

**Where it breaks down:** eggs are interchangeable, so a short carton is harmless. In code it's less forgiving — a short final batch can crash a loop that assumed every batch was full. More on that in a second.

#### Slicing, and the ragged last batch
Now the real mechanics. You **slice** the data: take the training set, cut it into consecutive chunks of `batch_size`, and hand the loop one chunk per iteration. When `N` doesn't divide evenly by `batch_size`, the *last* chunk comes out smaller — that leftover chunk (the short carton) has a name: a [[ragged final batch||The last batch of an epoch when N is not divisible by batch_size: it holds the leftover examples, so it is smaller than a full batch. Your loop must handle it, not assume every batch is exactly batch_size.]].

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="A bar of 100 examples sliced into chunks of 32: three full chunks of 32 shown in green, then a short final chunk of 4 shown in amber and labeled ragged final batch, four examples, smaller than batch_size."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="24" fill="#6B645E">N = 100 examples, batch_size = 32</text><rect x="20" y="40" width="140" height="38" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="90" y="64" fill="#1a5c38">batch 1 · 32</text><rect x="166" y="40" width="140" height="38" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="236" y="64" fill="#1a5c38">batch 2 · 32</text><rect x="312" y="40" width="140" height="38" rx="4" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="382" y="64" fill="#1a5c38">batch 3 · 32</text><rect x="458" y="40" width="42" height="38" rx="4" fill="#FDF3E8" stroke="#C99A12" stroke-width="2.5"/><text x="479" y="64" fill="#8A6D3B" font-weight="bold">4</text><text x="479" y="104" fill="#8A6D3B" font-size="9">ragged</text><text x="479" y="115" fill="#8A6D3B" font-size="9">final batch</text><text x="236" y="128" fill="#6B645E" font-size="10">32 + 32 + 32 + 4 = 100 — the leftovers still get to train</text></g></svg>
%%%

Why name this out loud? Because if your code *assumes* every batch is exactly `batch_size` — say, you hard-code the number 32 in a reshape — then on that last short batch it will either crash or quietly produce wrong numbers. Naming the ragged batch now means you'll write your loop to expect it.

#### Shuffle first, every time around
Before you slice, you **shuffle**. Here is the exact puzzle that forces you to.

**The trap.** Imagine the data sits in a fixed order. Many datasets are stored sorted by label — all the 0s first, then all the 1s, and so on. Slice that *without* shuffling and each batch is almost entirely one digit. Consecutive batches are then *correlated*: one batch pulls the weights hard toward one class, the next yanks them back the other way. The model can even start to learn the *order* of the data instead of the digits themselves.

**The fix.** You [[shuffle||Randomly reorder the training examples (shuffle their indices) at the start of every epoch, so each batch is a fresh, well-mixed sample of classes and the model never sees the same batch composition or ordering twice.]] the data at the start of *every* epoch. Then each batch is a fair, freshly mixed sample of all the classes.

!!! c-warn ⚠️
<b>Failure mode (silent): forgetting to shuffle sorted data.</b> Nothing crashes. The loss just drifts down in a jerky way, and test accuracy stays poor — because each batch yanks the weights back and forth toward one class at a time.
!!!

!!! c-info 🛠️
<b>How a senior engineer reads this:</b> a suspiciously jerky loss on data that happens to be sorted is a <b>data-ordering smell</b>, not a learning-rate problem. The fix is to shuffle every epoch — not to fiddle with the step size.
!!!

@@@ concept id=c6 tag="Mean not sum" title="Average, don't add — so batch_size isn't a hidden step-size" gotit="Got mean vs sum"
When you combine the loss and gradient over a batch, do you *add them up* or *average them*? It feels like it shouldn't matter. Feel the difference first.

**Think of a teacher grading a stack of tests to decide how hard tomorrow's lesson should be.** Suppose the teacher looks at the *total* number of mistakes. A class of 40 students racks up 40 times as many mistakes as a class of 1 — even if every single student did equally well! Judge by the total and the big class looks like a disaster, so the teacher wildly over-corrects. Now suppose the teacher looks at the *average* mistakes per student instead. The class *average* is the same whether there's 1 student or 40 — so the same, sensible teaching adjustment works for either class.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A grading analogy. Left, SUM the mistakes: a class of 1 versus a class of 40, where the total for 40 is 40 times bigger even at equal skill, so the reaction is distorted by class size. Right, AVERAGE the mistakes: the class average is the same for 1 or 40 students, so class size does not distort the reaction."><g font-family="monospace" font-size="10"><rect x="16" y="18" width="232" height="130" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.8"/><text x="28" y="40" fill="#C93B3B" font-weight="bold">SUM the mistakes</text><text x="28" y="62" fill="#6B645E" font-size="9.5">class of 1  → total = 2</text><text x="28" y="80" fill="#6B645E" font-size="9.5">class of 40 → total = 80</text><text x="28" y="100" fill="#6B645E" font-size="9.5">(same skill each!)</text><text x="28" y="122" fill="#C93B3B" font-size="9.5">big class looks 40× worse</text><text x="28" y="138" fill="#C93B3B" font-size="9.5">→ you over-correct</text><rect x="272" y="18" width="232" height="130" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="284" y="40" fill="#1a5c38" font-weight="bold">AVERAGE the mistakes</text><text x="284" y="62" fill="#6B645E" font-size="9.5">class of 1  → avg = 2</text><text x="284" y="80" fill="#6B645E" font-size="9.5">class of 40 → avg = 2</text><text x="284" y="100" fill="#6B645E" font-size="9.5">(class size drops out)</text><text x="284" y="122" fill="#1a5c38" font-size="9.5">same read either way</text><text x="284" y="138" fill="#1a5c38" font-size="9.5">→ one sane adjustment</text></g></svg>
%%%

**What the grading picture gets right:** averaging strips the *size* of the group out of the signal, so your reaction depends on *how wrong*, not *how many*.

**Where it breaks down:** a teacher naturally thinks in averages. In code, though, the sum is often the *easier* thing to type — so this is a bug you have to choose to avoid; it does not avoid itself.

#### The bug this quietly prevents
Now the twist: adding instead of averaging secretly turns `batch_size` into a hidden step-size. To see why, meet the [[learning rate||A small positive number that scales each update: param = param − learning_rate × gradient. It is applied once per batch update and sets the step size. Its "right" value is coupled to whether you mean or sum the batch.]] — the step-size knob you apply once per update. Watch what the two choices do to it:

%%% svg
<svg viewBox="0 0 520 155" role="img" aria-label="Two boxes. Left, SUM: eight example gradients add up to a big total gradient, and the update step is huge and scales with batch size — bad. Right, MEAN: the same eight gradients average to a normal-sized gradient, and the update step stays the same size no matter the batch size — good."><g font-family="monospace" font-size="11"><rect x="18" y="20" width="230" height="118" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.8"/><text x="30" y="42" fill="#C93B3B" font-weight="bold">SUM the 8 gradients</text><text x="30" y="62" fill="#6B645E" font-size="10">total = g1+g2+…+g8  (≈ 8× one)</text><text x="30" y="80" fill="#6B645E" font-size="10">step = lr × (big total)</text><text x="30" y="102" fill="#C93B3B" font-size="10">→ step grows with batch_size</text><text x="30" y="120" fill="#C93B3B" font-size="10">→ change batch → step size jumps</text><rect x="272" y="20" width="230" height="118" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="284" y="42" fill="#1a5c38" font-weight="bold">MEAN the 8 gradients</text><text x="284" y="62" fill="#6B645E" font-size="10">avg = (g1+…+g8) / 8  (normal size)</text><text x="284" y="80" fill="#6B645E" font-size="10">step = lr × (average)</text><text x="284" y="102" fill="#1a5c38" font-size="10">→ step size independent of batch</text><text x="284" y="120" fill="#1a5c38" font-size="10">→ change batch → lr still works</text></g></svg>
%%%

Read it slowly. If you **add up** the gradients, the total grows with `batch_size`, so your real step becomes `learning_rate × (about batch_size × one gradient)`. The step now secretly scales with the batch — double the batch and you've silently doubled your step size, and a learning rate that felt fine before now overshoots. If you **average** instead, the gradient stays a normal size no matter the `batch_size`, so the same learning rate keeps working even when you change the batch.

Here's the rule worth keeping: **take the mean of the per-example loss and gradient over the batch.** Do that, and the learning rate stays a clean, separate knob you can tune on its own.

@@@ concept id=c7 tag="Watch it fall" title="Track the loss — the proof the loop is learning" gotit="Got the loss curve"
How do you *know* the loop is actually working? Feel the idea before the mechanics.

**Think of tracking your resting heart rate as you get fitter over weeks.** Day to day it bounces around — you slept badly, you had coffee this morning, you were nervous. One morning's reading tells you almost nothing. But plot it over many weeks and the *trend* is unmistakably down: you're getting fitter. You never judge your fitness from one morning. You judge it from the trend across many readings.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A resting-heart-rate chart over weeks. The daily readings are jagged and bounce up and down, but the overall trend clearly slopes downward, with a dashed trend line showing the downward direction across many weeks."><g font-family="monospace" font-size="10"><line x1="50" y1="18" x2="50" y2="126" stroke="#6B645E" stroke-width="1"/><line x1="50" y1="126" x2="500" y2="126" stroke="#6B645E" stroke-width="1"/><text x="14" y="26" fill="#6B645E">bpm</text><text x="440" y="146" fill="#6B645E">weeks →</text><polyline points="56,40 78,54 100,44 124,62 148,52 174,70 200,60 228,78 258,70 290,86 322,80 356,96 392,90 430,104 470,98 494,108" fill="none" stroke="#C93B3B" stroke-width="1.6"/><path d="M56 46 L494 104" stroke="#2D8B55" stroke-width="2" stroke-dasharray="6 4"/><text x="300" y="60" fill="#1a5c38">trend = down (getting fitter)</text><text x="120" y="122" fill="#6B645E" font-size="9">bumpy day to day · but the trend is what counts</text></g></svg>
%%%

**What the heart-rate picture gets right:** a single noisy reading tells you almost nothing, but the *trend* across many readings is a signal you can trust.

**Where it breaks down:** your heart rate can only drop so far, and then you're genuinely healthy. A training loss can keep dropping toward zero even while the model is quietly getting *worse* at its real job — that twist is exactly what the last concept is about.

#### Now watch the training loss
Here's the mechanic: after each step — or each epoch — you print the **training loss** and look at the trend. If the loop is wired right and the learning rate is sane, the training loss should **head downward**. It'll be bumpy from batch to batch (that's the mini-batch shakiness from earlier), but overall it trends down — just like your heart rate. This moment is a big deal: it's your first live proof that all the machinery you built across Days 1 and 2 actually connects and learns.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A loss-versus-step line chart. The line starts high on the left, is jagged from batch noise, and trends clearly downward toward a low flat region on the right, showing the training loss falling as steps accumulate."><g font-family="monospace" font-size="10"><line x1="50" y1="20" x2="50" y2="128" stroke="#6B645E" stroke-width="1"/><line x1="50" y1="128" x2="500" y2="128" stroke="#6B645E" stroke-width="1"/><text x="16" y="28" fill="#6B645E">loss</text><text x="470" y="146" fill="#6B645E">step →</text><polyline points="55,32 75,40 95,36 115,55 135,50 160,68 185,64 210,80 240,78 270,92 300,90 330,102 365,100 400,110 440,108 490,114" fill="none" stroke="#2D8B55" stroke-width="2"/><circle cx="55" cy="32" r="3" fill="#C93B3B"/><text x="90" y="24" fill="#C93B3B">high at start (untrained)</text><circle cx="490" cy="114" r="3" fill="#1a5c38"/><text x="360" y="152" fill="#1a5c38">trending down = it's learning</text></g></svg>
%%%

#### The shape of the curve names the bug
The *shape* of the loss curve is a diagnosis. Here are the three you'll meet most, side by side — the healthy fall, and the two ways the **learning rate** (your step-size knob) goes wrong:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Three loss curves over steps. Green just-right curve falls and flattens near the bottom. Blue too-small curve slopes down barely, staying high. Red too-large curve zig-zags upward and diverges off the top toward NaN."><g font-family="monospace" font-size="9.5"><line x1="46" y1="20" x2="46" y2="140" stroke="#6B645E" stroke-width="1"/><line x1="46" y1="140" x2="500" y2="140" stroke="#6B645E" stroke-width="1"/><text x="20" y="30" fill="#6B645E">loss</text><text x="470" y="158" fill="#6B645E">steps →</text><path d="M52 34 C 140 120, 300 136, 494 138" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text x="330" y="130" fill="#1a5c38">just right ✓ (falls, flattens)</text><path d="M52 40 C 200 54, 360 66, 494 76" fill="none" stroke="#2A7B9B" stroke-width="2.2"/><text x="350" y="62" fill="#1F6280">too small (crawls, still high)</text><path d="M52 110 L 96 54 L 140 118 L 184 40 L 228 124 L 272 26 L 310 138" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text x="120" y="30" fill="#C93B3B">too large (zig-zags → diverges → NaN)</text></g></svg>
%%%

**Puzzle 1 — the learning rate too *large*.** Each update overshoots the bottom of the valley, so the next one overshoots the other way, harder. The loss **oscillates**, then **diverges**, and can blow up to `NaN` ("not a number" — what you get when the math runs off to infinity). **Cause:** the step is bigger than the distance to the bottom. **Remedy:** *lower the learning rate.*

**Puzzle 2 — the learning rate too *small*.** Every update is a grain of sand. The loss *does* fall, but so slowly training **crawls** — hours for what a sane rate does in seconds. **Cause:** steps too tiny to make headway. **Remedy:** *raise the learning rate* until the loss falls briskly without zig-zagging.

#### One silent trap: forgetting to reset the gradients
There's a third bug, and this one throws *no error at all*. Most training tools *add* each step's gradient onto whatever was already there (a default that helps in fancier setups). So if you don't clear it, step 2's gradient piles on top of step 1's, step 3's on top of that — the nudge grows wronger every step. **Cause:** gradients accumulate by default. **Remedy:** [[zero the gradients||reset every weight's gradient to 0 at the start of each iteration, before the backward pass, so each update uses only this step's signal — often literally a zero_grad() call]] at the start of *every* iteration. Predict what "used gradient" prints with and without a reset, then run it:

%%% demo id=zerograd label="run it — reset vs pile-up"
code: g = 2   # this step's gradient, same every step
code: acc = 0
code: for i in range(3): acc = acc + g; print('no reset -> used', acc)
code: for i in range(3): used = g; print('with reset -> used', used)
out: no reset -> used 2
out: no reset -> used 4
out: no reset -> used 6
out: with reset -> used 2
out: with reset -> used 2
out: with reset -> used 2
take: <b>Forget to reset and the gradient piles up: 2 → 4 → 6</b>, so each nudge is more wrong than the last. Reset every iteration and it stays a clean 2. One tiny line — zero the gradients — saves the whole run.
%%%

!!! c-info 🛠️
<b>How a senior engineer reads the curve:</b> a loss that <i>climbs or NaNs</i> screams learning rate too high (lower it). A loss that <i>barely moves</i> whispers learning rate too low (raise it). A loss that <i>explodes for no reason</i> on a loop you just wrote is very often forgotten gradient-resetting. The curve is the first place experienced engineers look — and now you can read it too.
!!!

@@@ concept id=c8 tag="The ceiling" title="A falling loss is not a good model" gotit="Got the ceiling"
We'll end on the most honest — and most important — idea in the lesson. So slow down here, and feel it before you read the rule.

**Think of a student who memorizes last year's exact exam answers, word for word.** Hand them *last year's* paper and they score a perfect 100% — their "training loss" is zero, they look like a genius. But hand them *this* year's paper, with fresh questions, and they flounder. They learned the answer sheet, not the subject. Their spotless score on the material they *practiced* said nothing about material they'd never seen.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A memorizing-student analogy. Left, last year's exam scored 100 percent because the student memorized it. Right, this year's new exam scored poorly because the student never learned the subject. An arrow shows that a perfect score on practiced material does not carry over to new material."><g font-family="monospace" font-size="10" text-anchor="middle"><rect x="30" y="26" width="180" height="110" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="120" y="50" fill="#1a5c38" font-weight="bold">last year's exam</text><text x="120" y="82" font-size="26" fill="#1a5c38">100%</text><text x="120" y="112" fill="#6B645E" font-size="9">(memorized the answers)</text><text x="120" y="126" fill="#6B645E" font-size="9">= "training loss" ≈ 0</text><text x="260" y="84" fill="#6B645E" font-size="16">→</text><rect x="310" y="26" width="180" height="110" rx="8" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.8"/><text x="400" y="50" fill="#C93B3B" font-weight="bold">THIS year's new exam</text><text x="400" y="82" font-size="26" fill="#C93B3B">42%</text><text x="400" y="112" fill="#6B645E" font-size="9">(never learned the subject)</text><text x="400" y="126" fill="#6B645E" font-size="9">= the score that matters</text></g></svg>
%%%

**What the memorizing-student picture gets right:** scoring perfectly on material you *practiced* says nothing about material you haven't seen — and "haven't seen" is what actually matters.

**Where it breaks down:** a student at least knows they're cramming. A model has no idea whether it's memorizing or truly understanding. The *only* way to tell is to test it on data it never trained on — a tool you don't have yet. You'll build it tomorrow.

#### The rule, and the curve that shows it
Now the rule, plainly. The mini-batch loop is an **optimization procedure** — a machine for driving the *training* loss down. That is *all* it is. It is **not** a promise that your model is any good. A perfectly-wired loop with a low training loss can still be a bad model.

Here's the picture that gives the game away. The green line is the loss on the data the model trains on — it keeps falling, looking great. The red dashed line is the loss on data the model has *never seen* — and watch it: it dips at first, then **turns and climbs back up**. The gap between the two widens as training goes on. That widening gap is the memorizing student, caught in the act:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Two loss curves over epochs. The training loss (green) falls steadily toward the bottom. The unseen-data loss (red, dashed) falls at first, then turns and rises back up toward the top — the gap between them widening — so the unseen-data loss ends far higher than the training loss, showing that a low training loss does not mean the model is good on new data."><g font-family="monospace" font-size="10"><line x1="50" y1="18" x2="50" y2="136" stroke="#6B645E" stroke-width="1"/><line x1="50" y1="136" x2="500" y2="136" stroke="#6B645E" stroke-width="1"/><text x="16" y="26" fill="#6B645E">loss</text><text x="450" y="154" fill="#6B645E">epochs →</text><polyline points="55,44 110,64 170,82 240,96 320,108 410,116 490,122" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="300" y="130" fill="#1a5c38">training loss ↓ (looks great)</text><polyline points="55,48 110,62 170,70 240,72 320,64 410,54 490,44" fill="none" stroke="#C93B3B" stroke-width="2" stroke-dasharray="5 3"/><text x="150" y="40" fill="#C93B3B">loss on unseen data ↑ (the truth)</text><path d="M498 122 L498 44" stroke="#8A6D3B" stroke-width="1" stroke-dasharray="2 2"/><text x="470" y="90" fill="#8A6D3B" font-size="9" text-anchor="middle">gap</text><text x="150" y="164" fill="#6B645E">low training loss ≠ good model — the widening gap is what matters</text></g></svg>
%%%

Read the two lines: they start together, but as the loop keeps optimizing, the training loss keeps sinking while the unseen-data loss bottoms out and *rises*. The model is getting better at the answer sheet and worse at the subject.

#### What this limit points to next
The takeaway to carry forward: a low training loss proves your loop *runs*. It does not prove your model is *good*. The model may be **overfitting** — memorizing the training set instead of learning the general pattern. To even *measure* whether that's happening, you need data the model never trained on (a held-out split). And then you need techniques to fix it when it does. That is exactly where you go next.

!!! c-info 🔗
<b>Forward pointer:</b> Day 4 (<b>Full Training: Loss Curve &amp; Dropout</b>) introduces the train/validation split — data the model never trains on — so you can finally *see* the widening gap in the picture above as it happens, plus dropout and weight decay to shrink it. Day 5 swaps your hand-written loop for PyTorch's <code>DataLoader</code> (automatic shuffling + batching) and <code>torch.optim.SGD</code> (the framework's version of your manual update). The loop you built today is the very thing all of that automates.
!!!

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you just built the engine that makes every neural network learn. Here's the whole day held in one picture: the **free-throw drill** you started with, now labelled with the real training words. Read it as one lap of practice — grab a handful, shoot, measure, nudge — repeated batch after batch, epoch after epoch, while you watch the loss fall.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A one-page recap. Top: a ring of four steps forward, loss, backward, update with a repeat arrow, fed by a handful of examples labeled one batch. Middle: a falling jagged loss curve. Bottom: the batch_size dial between full-batch and single-example, plus the three key habits shuffle, average, and check held-out."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#6B645E" font-size="11">grab a handful → forward → loss → backward → update → repeat</text><rect x="70" y="28" width="70" height="30" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="105" y="47" fill="#1a5c38">forward</text><text x="146" y="47" fill="#6B645E">→</text><rect x="158" y="28" width="60" height="30" rx="5" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.5"/><text x="188" y="47" fill="#8A6D3B">loss</text><text x="224" y="47" fill="#6B645E">→</text><rect x="236" y="28" width="76" height="30" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.5"/><text x="274" y="47" fill="#C93B3B">backward</text><text x="318" y="47" fill="#6B645E">→</text><rect x="330" y="28" width="64" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.5"/><text x="362" y="47" fill="#5E5191">update</text><path d="M362 58 L362 72 L105 72 L105 58" fill="none" stroke="#7C6DAA" stroke-width="1.2" stroke-dasharray="3 2"/><text x="234" y="86" fill="#6B645E">↺ one iteration per batch · one epoch per full pass</text><line x1="50" y1="100" x2="50" y2="160" stroke="#6B645E" stroke-width="1"/><line x1="50" y1="160" x2="250" y2="160" stroke="#6B645E" stroke-width="1"/><text x="24" y="108" fill="#6B645E">loss</text><polyline points="55,106 80,120 105,116 135,132 165,130 200,144 235,142 248,150" fill="none" stroke="#2D8B55" stroke-width="1.6"/><text x="150" y="176" fill="#1a5c38">bumpy but trending down = learning</text><text x="380" y="108" fill="#6B645E">batch_size = the dial:</text><text x="380" y="124" fill="#8A6D3B">small → shaky + frequent</text><text x="380" y="138" fill="#5E5191">large → smooth + rare + heavy</text><text x="380" y="156" fill="#2D8B55">32–256 = sweet spot</text><line x1="20" y1="184" x2="500" y2="184" stroke="#E5DDD0" stroke-width="1"/><text x="90" y="200" fill="#C93B3B">shuffle every epoch</text><text x="255" y="200" fill="#C93B3B">average, don't add</text><text x="425" y="200" fill="#C93B3B">low loss ≠ good model</text></g></svg>
%%%

**Where the free-throw picture finally breaks down:** a player fixes one thing per handful of shots, but the loop nudges *every* one of its ~100,000 weights at once, thousands of times — and that scale is the only reason it can learn things a person never could. Keep the two cheat-sheets below as your map back into today.

#### Cheat-sheet · the loop, its dials, and its traps
%%% table
:: Piece / trap :: In one line :: Remember
forward → loss → backward → update :: one lap of the loop :: repeat over batches and epochs
epoch vs iteration vs batch :: full pass vs one update vs one handful :: many iterations make one epoch
full-batch (all N) :: exact gradient, one slow step per pass :: accurate but slow + memory-heavy
single-example SGD (batch=1) :: constant updates, noisy gradient :: fast but jumpy
mini-batch (the default) :: average a handful, then one step :: best of both — steadier and frequent
batch_size (the dial) :: small = shaky+frequent, large = smooth+rare :: sweet spot ≈ 32–256
ragged final batch :: leftover when N not divisible by batch_size :: don't assume every batch is full
forgot to shuffle :: sorted data → correlated batches, jerky learning :: cure: shuffle every epoch
sum instead of mean :: step secretly scales with batch_size :: cure: average the loss + gradient
learning rate too high :: loss oscillates / diverges / NaN :: cure: lower the learning rate
learning rate too low :: loss barely moves, crawls :: cure: raise the learning rate
forgot to reset gradients :: they pile up across steps :: cure: zero the gradients each iteration
low training loss :: only proves the loop optimizes :: it does NOT prove the model is good
%%%

#### Cheat-sheet · the words you met today
%%% jargon
training loop | the four-step cycle — forward, loss, backward, update — repeated over the data until the model learns
epoch | one full pass over the entire training set
iteration (step) | one weight update, on one batch
batch (mini-batch) | a small handful of examples processed together before one update
batch_size | how many examples are in the handful — the dial between shaky+frequent and smooth+rare
full-batch gradient descent | use all N examples for one exact gradient, then one step — accurate but slow
stochastic gradient descent (SGD) | update after every single example (batch of 1) — frequent but noisy
mini-batch gradient descent | average a handful of gradients, then step — the modern default
shuffle | randomly reorder the data at the start of every epoch so batches are well-mixed
ragged final batch | the smaller last batch when N is not divisible by batch_size
learning rate | the step size each update: param = param − learning_rate × gradient
zero the gradients | reset every gradient to 0 before each backward pass so they don't accumulate
overfitting | memorizing the training data instead of the pattern — great on seen data, bad on new
%%%

That's the whole day. Tomorrow you split off some data the model never trains on, so you can finally *measure* whether it truly learned — and meet dropout, a trick that fights the memorizing you just learned to fear.

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
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 5 Day 3 artifact.

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
