---
quest_id: wf5-d04-training
mode: concept
donor: v9-base.donor
page_title: "Module 5 · Day 4 — Full Training: The Loss &amp; Dropout"
module_label: "Module 5 · Train · Day 4"
title: "Full Training"
subtitle: "The Loss Score &amp; Dropout"
brand_sub: "Foundations · M5 Day 4"
spine: "practice: a machine learns to read by practicing on examples and being corrected"
nav_prev_href: "../day-03-minibatch-loop/lesson.html"
nav_prev_label: "The Mini-Batch Loop"
nav_next_href: "../day-05-pytorch-version/lesson.html"
nav_next_label: "The Same Model in PyTorch"
fin_title: "Module 5 · Day 4 complete! 🏆"
fin_body: "You did it! You finished <b>Full Training: The Loss Score &amp; Dropout</b>. Look at what you can now do: you can say exactly what a <b>loss</b> is — one number for how wrong a guess was — and why <b>cross-entropy</b> punishes a confident-and-wrong answer so hard. You know training is nothing more than pushing the <i>average</i> loss down, and that the number that actually matters is the <b>held-out</b> loss, not the training one. You can name the failure — <b>overfitting</b> — say what causes it, and reach for its famous cure: <b>dropout</b>. You understand why dropout is on in practice but off at test time, and the exact <b>1/(1−p)</b> scaling that makes those two settings agree. And you know the one bug that silently wrecks predictions: leaving dropout on at test time.<br>Take a breath — that is the honesty and the safety net of every real training run. Next up: <b>The Same Model in PyTorch</b>."
notebook_yardstick: 00-neural-networks/fundamentals/08_training_loop.ipynb
coverage_topics:
  - {topic: loss function is one number for how wrong a guess was, keywords: [loss, one number, how wrong, how far off, single score]}
  - {topic: mean squared error MSE for regression, keywords: [mean squared error, mse, regression, squared, distance]}
  - {topic: cross-entropy for classification paired with softmax/sigmoid, keywords: [cross-entropy, cross entropy, classification, softmax, sigmoid, probability, confident and wrong]}
  - {topic: training is pushing the average loss down, keywords: [average loss, average the loss, push the loss down, shrink the loss, lower loss]}
  - {topic: watching the loss curve fall across epochs, keywords: [loss curve, per epoch, trending down, dashboard, learning is happening]}
  - {topic: train and validation split with held-out data, keywords: [train/validation split, validation, held-out, holding out, generalize, never trains on]}
  - {topic: overfitting is memorizing training-set noise, keywords: [overfitting, memoriz, training loss keeps dropping while validation loss rises, diverge, noise]}
  - {topic: dropout headline regularizer randomly switches off units at training time, keywords: [dropout, randomly switch off, drop units, dropout rate, regularizer, remedy]}
  - {topic: dropout as a cheap approximation of averaging many networks ensemble, keywords: [ensemble, averaging many networks, cheap approximation, many networks, no single unit]}
  - {topic: dropout on at train time off at inference with scaling, keywords: [only at training, off at inference, off at test, keep all units, scale, 1/(1-p)]}
  - {topic: train vs eval mode switch model.train model.eval, keywords: [train mode, eval mode, model.train, model.eval, switch, turns dropout off, silent bug]}
  - {topic: early stopping as a second remedy for overfitting, keywords: [early stopping, stop training, validation loss stops improving, halt]}
  - {topic: underfitting is the opposite failure model too small or trained too little, keywords: [underfitting, too small, trained too little, loss stays high, both train and validation, dropout does not fix]}
  - {topic: loss not decreasing stuck learning rate too small, keywords: [not decreasing, stuck, learning rate too small, poor initialization, raise the learning rate]}
  - {topic: loss diverging to NaN learning rate too large, keywords: [diverging, nan, blows up, learning rate too large, lower the learning rate]}
  - {topic: capability limit low loss alone does not prove a good model, keywords: [low loss alone, does not prove, held-out, dropout is not free, slows convergence, can hurt]}
require_artifact: true
---

@@@ hero
@lede Think about a round of **mini-golf**. You take a swing, the ball stops somewhere, and you measure one thing: *how far is it from the hole?* Two feet is a good shot. Twenty feet is a bad one. That single distance is the only number you need to know whether to be happy or to try again. A neural network learns the exact same way. After it makes a guess, we measure one number — how far the guess landed from the right answer — and call it the **loss**. The whole of "training" is just this: take a swing, read the distance, nudge your aim, swing again, until the ball keeps landing close. It is the same rhythm you built yesterday, but today you finally get to *read the scorecard* — and to catch the sneakiest way a model can fool you: it can ace the practice course and then fall apart on a course it has never seen. That trap has a name, **overfitting**, and it has a famous, almost silly-sounding cure called **dropout** — the same kind of trick that quietly steadies the big models behind tools like ChatGPT. By the end of today you will read a model's honesty the way a coach reads a scorecard.
@goal Here's what we'll figure out together today, one small idea at a time:
- what a **loss** really is — one number for how wrong a guess was — and its two everyday flavors (**MSE** and **cross-entropy**);
- why training is nothing more than pushing the *average* loss **down**, and how to read the falling **loss curve**;
- the honest test — **held-out** data — and the failure it exposes: **overfitting**;
- the headline cure — **dropout** — why it works, and why it's *on* in practice but *off* at test time;
- and the small bugs that quietly wreck a run: a wrong learning rate, and forgetting to flip one switch.

@@@ concept id=c1 tag="The loss" title="The loss: one number for how wrong a guess was" gotit="Got the loss"
Yesterday you ran the loop — guess, correct, repeat. But to *correct* a guess, you first have to say how wrong it was. Not "a bit wrong" or "pretty good" — an actual **number**. That number is the whole idea of today, so let's meet it gently.

**Back to the mini-golf hole.** You want the ball in the cup. It lands two feet short. How bad is that? You measure the gap: two feet. Land it twenty feet away and the gap is twenty. The gap *is* your score — small gap, good shot; big gap, bad shot. A neural network does the same. It makes a guess (say, "this pixel-blob is a 7"), we know the true answer, and we measure the gap between them. That gap, boiled down to one number, is the **[[loss||one number that says how far the model's guess landed from the right answer — smaller is better]]**.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A mini-golf green. A hole (the cup) sits on the right. A ball landed short of it, and a measuring tape shows the distance from ball to cup. A label says this distance is the loss: small distance is a good shot, big distance is a bad shot."><g font-family="monospace" font-size="10" text-anchor="middle"><rect x="20" y="30" width="480" height="110" rx="12" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><circle cx="430" cy="85" r="14" fill="#2b2b2b" stroke="#1a5c38" stroke-width="1.5"/><text x="430" y="60" font-size="9" fill="#1a5c38">the cup</text><text x="430" y="72" font-size="9" fill="#1a5c38">(right answer)</text><circle cx="150" cy="85" r="8" fill="#ffffff" stroke="#C93B3B" stroke-width="1.8"/><text x="150" y="112" font-size="9" fill="#C93B3B">the guess</text><line x1="158" y1="85" x2="416" y2="85" stroke="#C99A12" stroke-width="1.6" stroke-dasharray="5 3"/><text x="287" y="78" font-size="11" fill="#8A6D3B">gap = the loss</text><text x="120" y="132" font-size="9" fill="#2D8B55">small gap → good</text><text x="400" y="132" font-size="9" fill="#C93B3B">big gap → bad</text><text x="260" y="22" font-size="10" fill="#6B645E">one number = how far off the guess is</text></g></svg>
%%%

**What the golf picture gets right:** wrongness becomes a single distance you can shrink, so "get better" simply means "make that number smaller." **Where it breaks down:** on a golf green the gap is a plain distance you can eyeball; a model's guess is a list of numbers, so we need a formula to turn "how far off" into one clean score — and there is more than one way to do it.

#### The simplest flavor: measure the distance and square it (MSE)
When the answer is a plain number — say the model predicts a house price, or tomorrow's temperature — we use the most natural score of all: take the gap, and square it. Averaged over many guesses, that's the **[[mean squared error||MSE: average of the squared gaps between guesses and answers — the go-to loss when the answer is a number]]**, or **MSE**. Squaring does two friendly things: it makes every gap positive (a miss is a miss, left or right), and it makes *big* misses count for a lot more than small ones. This is the loss you reach for in **regression** — any time the answer is a quantity, not a category.

Let's *watch* the squaring turn misses into a score. Predict which guess gets punished hardest, then reveal it:

%%% demo id=mse label="see the squared gaps"
code: gaps = [0.5, 1.0, 2.0, 4.0];  squared = [g*g for g in gaps]
out: gaps      = [0.5, 1.0, 2.0, 4.0]
out: squared   = [0.25, 1.0, 4.0, 16.0]   # each gap × itself
out: MSE = mean(squared) = 5.31
take: <b>Squaring is not fair — on purpose.</b> A gap of 4 isn't 8× worse than a gap of 0.5, it's 64× worse (16 vs 0.25). Big misses scream; the model rushes to fix them first.
%%%

So a loss is just a rule for turning "how wrong" into one number. MSE is the rule when the answer is a quantity. But your MNIST model isn't guessing a price — it's picking a *category* (which digit, 0–9). Categories need a different, cleverer score, and that's next.

@@@ concept id=c2 tag="Cross-entropy" title="Cross-entropy: the loss that hates being confidently wrong" gotit="Got cross-entropy"
Your MNIST model doesn't guess a number on a number line — it picks a *category*. So it doesn't say "7.0", it says something more honest: "I'm 80% sure it's a 7, 15% a 1, 5% a 9." Those percentages are a set of **[[probabilities||numbers between 0 and 1 that say how sure the model is of each choice, and add up to 1]]**. To score a guess like that, we need a loss built for confidence — and the everyday feel of it is the surprise of a weather forecaster.

**Picture a TV weather forecaster.** On Monday they say "99% chance of sun" — and it pours rain. Everyone is furious: they were *so sure* and *so wrong*. On Tuesday they hedge, "50% sun, 50% rain" — and it rains. Nobody minds much; they never promised. The punishment fits the confidence: being sure *and* wrong is embarrassing; being unsure and wrong is forgivable. **[[Cross-entropy||the standard loss for classification: it grows gently when the model hedges and is wrong, and explodes when the model is confident and wrong]]** — the standard loss when the answer is a category — scores a model the exact same way.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Two weather forecasters. Left: one says 99 percent sun with a big confident face, but it rains, so a huge penalty arrow points up. Right: one says 50-50, it rains, and only a small penalty arrow. The caption: cross-entropy punishes confident-and-wrong far harder than unsure-and-wrong."><g font-family="monospace" font-size="10" text-anchor="middle"><rect x="20" y="24" width="230" height="130" rx="10" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.6"/><text x="135" y="44" fill="#C93B3B" font-size="11">"99% SUN!"  ☀️</text><text x="135" y="62" fill="#6B645E" font-size="9">…it rains 🌧️</text><line x1="135" y1="140" x2="135" y2="78" stroke="#C93B3B" stroke-width="6"/><path d="M135 74 l-7 12 l14 0 z" fill="#C93B3B"/><text x="135" y="132" fill="#C93B3B" font-size="10">HUGE penalty</text><rect x="270" y="24" width="230" height="130" rx="10" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.6"/><text x="385" y="44" fill="#8A6D3B" font-size="11">"50% sun, 50% rain"</text><text x="385" y="62" fill="#6B645E" font-size="9">…it rains 🌧️</text><line x1="385" y1="140" x2="385" y2="118" stroke="#C99A12" stroke-width="6"/><path d="M385 114 l-7 12 l14 0 z" fill="#C99A12"/><text x="385" y="132" fill="#8A6D3B" font-size="10">small penalty</text><text x="260" y="172" fill="#6B645E">confident + wrong = punished hard · unsure + wrong = forgiven</text></g></svg>
%%%

**What the forecaster picture gets right:** the penalty depends on *how sure you were*, not just on being wrong — so the model learns to be honest about its confidence, not just to pick the top answer. **Where it breaks down:** a forecaster feels social embarrassment; cross-entropy is a cold formula — it just gets very large when the probability you put on the *true* answer gets close to zero.

#### Where the probabilities come from: softmax
Your network's last layer spits out raw scores, not tidy percentages. A small function called **[[softmax||turns a row of raw scores into probabilities that are all positive and add up to 1]]** squeezes those scores into probabilities that are all positive and sum to 1. Then cross-entropy reads off the one probability the model gave to the *true* class and scores it. This is a rule worth remembering: a probability-style loss (cross-entropy) is almost always paired with a softmax output for many classes, or its cousin the sigmoid for a yes/no answer. Full softmax math is a later day — today, just hold the picture: **scores → softmax → probabilities → cross-entropy reads the true class's probability.**

Now let's *feel* why "confident and wrong" is so much worse. Predict which row hurts most, then reveal:

%%% demo id=ce label="watch confidence get punished"
code: # true answer is "7". loss = -log(probability the model gave to 7)
code: cross_entropy(prob_on_true=0.90)   # confident AND right
code: cross_entropy(prob_on_true=0.50)   # hedging
code: cross_entropy(prob_on_true=0.01)   # confident AND wrong
out: prob 0.90 → loss 0.11    # barely punished
out: prob 0.50 → loss 0.69    # mild
out: prob 0.01 → loss 4.61    # OUCH
take: <b>The pain explodes as the true-class probability heads to zero.</b> Confident-and-right ≈ 0.11; confident-and-wrong ≈ 4.61 — about 40× the pain. That steep punishment is a strong, clear signal that yanks the model away from cocky mistakes.
%%%

!!! c-info 🧮
<b>Optional (skippable) — the one-line formula.</b> For a single example, cross-entropy is `loss = −log(p)`, where `p` is the probability the model assigned to the *correct* class. When `p` is near 1 (confident and right), `−log(p)` is near 0. When `p` is near 0 (confident and wrong), `−log(p)` shoots toward infinity. That single `log` is the whole "explode when confidently wrong" behavior. You never have to derive it today — just know the shape.
!!!

@@@ concept id=c3 tag="Loss curve" title="Training = pushing the average loss down (and watching it fall)" gotit="Got the loss curve"
Here's the quietly huge idea: once wrongness is a single number, "learn" stops being mysterious. To learn is simply to make that number **smaller**. Practice, over and over, until the loss keeps dropping. That's it — that's the whole job.

**Think of a heart-rate monitor at the gym.** One beep tells you almost nothing. But watch the line over ten minutes and the *trend* tells you everything — going down means you're recovering, flat means you've stalled, spiking means something's wrong. Training has the same dashboard. Each swing gives one loss number (jumpy, like one heartbeat), but we average the loss over each **[[epoch||one full pass through all the training data — from yesterday's loop]]** and plot it. A line that slides **down** across epochs is your live proof that learning is happening.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A dashboard-style chart. The x-axis is epochs, the y-axis is average loss. A green curve starts high on the left and slides down toward the right, flattening near the bottom. A dashed target line sits near the floor. Labels point out the steep early drop as fast learning and the flat tail as converging."><g font-family="monospace" font-size="10" text-anchor="middle"><line x1="55" y1="20" x2="55" y2="150" stroke="#6B645E" stroke-width="1.2"/><line x1="55" y1="150" x2="490" y2="150" stroke="#6B645E" stroke-width="1.2"/><text x="30" y="30" fill="#6B645E" font-size="9">loss</text><text x="30" y="90" fill="#6B645E" font-size="9">high</text><text x="30" y="148" fill="#6B645E" font-size="9">low</text><text x="270" y="176" fill="#6B645E" font-size="9">epochs (full passes) →</text><polyline points="60,34 90,52 120,74 150,96 185,112 225,124 270,132 320,138 380,142 470,144" fill="none" stroke="#2D8B55" stroke-width="2.2"/><circle cx="60" cy="34" r="3" fill="#2D8B55"/><circle cx="150" cy="96" r="3" fill="#2D8B55"/><circle cx="470" cy="144" r="3" fill="#2D8B55"/><line x1="55" y1="150" x2="490" y2="150" stroke="#C99A12" stroke-width="1" stroke-dasharray="4 3"/><text x="130" y="52" fill="#1a5c38" font-size="9">steep drop = fast learning</text><text x="400" y="132" fill="#8A6D3B" font-size="9">flattening = converging</text></g></svg>
%%%

**What the heart-rate picture gets right:** you judge progress by the *trend* over time, not by any single reading. **Where it breaks down:** a heart rate has a healthy target you know in advance; with a loss you rarely know the "right" floor — you mostly watch whether it's still dropping, and (as you'll see) whether it's dropping for the *right* reasons.

#### It's the *average* loss, not any single one
One swing's loss bounces around — that's the mini-batch shakiness you met yesterday. So we don't chase single numbers. Training minimizes the **average** loss over the training data: add up the loss on every example, divide by how many there are, and push *that* down. Below, three noisy per-batch losses smooth into one honest epoch average — the number we actually plot:

%%% demo id=avg label="smooth the noise into one number"
code: batch_losses = [0.71, 0.44, 0.63]     # three jumpy per-batch reads
code: epoch_avg = sum(batch_losses) / len(batch_losses)
out: batch_losses = [0.71, 0.44, 0.63]      # bouncy
out: epoch_avg    = 0.60                     # the steady number we track
take: <b>Average, then track.</b> Any single batch is noisy; the epoch average is the calm signal. Training's whole aim is to drag that average down, epoch after epoch.
%%%

So far, so good: pick a loss, average it, push it down, watch the curve fall. It feels like winning. But here's the plot twist that the rest of the day is about — a falling training loss can be **lying to you**. Let's find out how.

@@@ concept id=c4 tag="Held-out data" title="The honest test: hold some data back" gotit="Got held-out data"
Here is the twist. A falling training loss only proves the model got good at the exact examples it practiced on. But we don't care about those — we already know their answers! We care whether it can handle *new* pictures it has never seen. So we run an honest test: hide some data from it, and check the loss there.

**Think of studying for an exam with a stack of practice questions.** The smart move is to *not* study all of them. You set a handful aside, unopened, and only try those the night before the real test. If you ace the ones you studied but bomb the set you saved, you learned the answers, not the subject. If you do well on the saved set too, you actually *get* it. Neural networks do exactly this. We split the data: most of it becomes the **[[training set||the examples the model learns from — it sees these over and over]]** the model practices on, and a slice becomes the **[[validation set||held-out examples the model never trains on — used only to check if it truly learned, not memorized]]** — held-out examples it *never* trains on, used only to check honest progress. That check is the **held-out loss**, and it is the number that actually matters.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A stack of practice questions being split into two piles. A large left pile labeled training set with an open-book icon, and a small right pile labeled validation set, sealed, labeled do not study. An arrow shows the sealed pile is only opened at test time to give the honest score."><g font-family="monospace" font-size="10" text-anchor="middle"><rect x="30" y="30" width="250" height="110" rx="10" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><text x="155" y="52" fill="#1a5c38" font-size="11">TRAINING SET (~80%)</text><text x="155" y="72" fill="#6B645E" font-size="9">📖 practice on these, over and over</text><text x="155" y="94" fill="#6B645E" font-size="9">the loss here always falls —</text><text x="155" y="108" fill="#6B645E" font-size="9">it just means "learned these"</text><rect x="300" y="30" width="190" height="110" rx="10" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.6" stroke-dasharray="6 4"/><text x="395" y="52" fill="#8A6D3B" font-size="11">VALIDATION (~20%)</text><text x="395" y="72" fill="#C93B3B" font-size="9">🔒 never trained on</text><text x="395" y="94" fill="#6B645E" font-size="9">the honest score:</text><text x="395" y="108" fill="#6B645E" font-size="9">"does it work on NEW data?"</text><text x="260" y="164" fill="#6B645E">split once, keep the sealed pile sealed</text></g></svg>
%%%

**What the practice-exam picture gets right:** the only fair test uses questions you kept sealed away — the score on studied questions can be faked by memorizing. **Where it breaks down:** a student might accidentally peek; a model literally cannot see the validation set during training, so the wall between the two piles is airtight — as long as you split *once* and never train on the held-out slice.

#### Now you watch *two* curves, not one
From here on, the dashboard plots two lines: the training loss and the validation loss. When both slide down together, you're genuinely learning. Below, an early snapshot where both are still falling — the happy case. Watch how close they stay:

%%% demo id=twoloss label="read both curves"
code: # after a few epochs of healthy training
code: train_loss      = [0.9, 0.6, 0.45, 0.38]
code: validation_loss = [0.95, 0.65, 0.50, 0.44]
out: epoch:            1     2     3     4
out: train_loss:      0.90  0.60  0.45  0.38   ↓ falling
out: validation_loss: 0.95  0.65  0.50  0.44   ↓ falling too
take: <b>Both curves falling together = real learning.</b> The small gap between them is normal and fine. The moment to worry is when they stop moving together — which is exactly the failure we meet next.
%%%

That gap between the two lines is about to become the single most important thing on the whole dashboard. When it starts to *grow*, your model has caught the disease every practitioner fears.

@@@ concept id=c5 tag="Overfitting" title="Overfitting: when the model memorizes instead of learns" gotit="Got overfitting"
This is the trap the whole day has been walking toward. It's sneaky because it *looks* like success — the training loss keeps dropping, and you feel great — right up until you check the held-out data and the floor falls out.

**Think of a student who memorizes the answer key.** Give them the practice test again and they score 100% — flawless, instant. But it's a trick: they didn't learn the *subject*, they memorized *which letter goes with which question number*. Hand them the real exam with new questions and they crash. That's **[[overfitting||the model memorizes the exact training examples, even their random noise, instead of learning the general pattern — so it fails on new data]]**: the model has so much room that it starts memorizing the training examples — even their meaningless quirks and noise — instead of the real pattern. It aces what it has seen and fails what it hasn't.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A loss dashboard showing overfitting. Two curves start together on the left and both fall. Partway across, the green training curve keeps falling toward zero, but the amber validation curve bottoms out and turns back UP. The widening gap between them is shaded and labeled overfitting, with a marker at the point where validation starts rising labeled best model was here."><g font-family="monospace" font-size="10" text-anchor="middle"><line x1="55" y1="18" x2="55" y2="150" stroke="#6B645E" stroke-width="1.2"/><line x1="55" y1="150" x2="495" y2="150" stroke="#6B645E" stroke-width="1.2"/><text x="30" y="28" fill="#6B645E" font-size="9">loss</text><text x="275" y="176" fill="#6B645E" font-size="9">epochs →</text><path d="M60 40 L120 70 L180 92 L240 108 L300 118 L360 126 L420 132 L485 137" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text x="450" y="128" fill="#1a5c38" font-size="9">train ↓</text><path d="M60 46 L120 76 L180 96 L235 104 L270 106 L320 118 L380 130 L485 143" fill="none" stroke="#C99A12" stroke-width="2.2"/><text x="450" y="150" fill="#8A6D3B" font-size="9">validation ↑</text><line x1="235" y1="104" x2="235" y2="150" stroke="#C93B3B" stroke-width="1" stroke-dasharray="3 3"/><circle cx="235" cy="104" r="4" fill="#C93B3B"/><text x="235" y="96" fill="#C93B3B" font-size="9">best model was HERE</text><path d="M320 118 L320 128 M380 130 L380 143" stroke="#C93B3B" stroke-width="0.8"/><text x="405" y="112" fill="#C93B3B" font-size="9">growing gap = overfitting</text></g></svg>
%%%

**What the answer-key picture gets right:** a perfect score on seen material can be pure memorization, worthless on anything new — so a low *training* loss is not the goal. **Where it breaks down:** a student chooses to cheat; a model doesn't choose — it slides into memorizing automatically whenever it has more capacity (more weights) than the data can pin down.

#### How to *spot* it: the two curves split apart
You never have to guess whether you're overfitting — the dashboard tells you. The tell-tale sign: **training loss keeps dropping while validation loss stops and turns back up.** The moment those two lines split and the gap grows, memorizing has begun. Below, six full epochs, numbered plainly 1 to 6. Read the validation row and find where it stops falling:

%%% demo id=overfit label="catch the split"
code: # six epochs, numbered 1..6 — watch where the two curves diverge
code: epoch:            1     2     3     4     5     6
code: train_loss:      0.60  0.40  0.25  0.14  0.07  0.03
code: validation_loss: 0.62  0.44  0.35  0.36  0.42  0.51
out: train keeps falling:  0.60 → 0.03 across epochs 1→6  (looks amazing!)
out: validation LOWEST:    0.35 at epoch 3  ← the bottom
out: validation RISES:     epoch 4: 0.36 → epoch 5: 0.42 → epoch 6: 0.51   ⚠️ overfitting
take: <b>The split is the alarm.</b> Validation is lowest at <b>epoch 3</b> (0.35). From <b>epoch 4</b> on it turns UP (0.36 → 0.42 → 0.51) while training keeps diving. Everything after epoch 3 is the model memorizing training noise — so the best model was back at <b>epoch 3</b>, the bottom of the validation curve, not at the end.
%%%

!!! c-warn ⚠️
<b>The failure — overfitting (cause):</b> the model has more capacity (weights) than the data constrains, so it fits the training examples' random noise, not the true pattern. <b>The symptom:</b> training loss falls, validation loss rises — the gap between the two curves grows. <b>Coming up:</b> two named cures — the headline one, dropout, and a simple one, early stopping.
!!!

Now for the good part — how do we *fight* it? There's a famous, almost silly-sounding trick that beat overfitting so well it's now in nearly every big model. Let's meet dropout.

@@@ concept id=c6 tag="Dropout" title="Dropout: the headline cure for overfitting" gotit="Got dropout"
This is today's star. **Dropout** sounds almost too simple to work: while the network is practicing, at every single step you randomly *switch off* a fraction of its neurons — you set their outputs to zero, as if they weren't there. Different neurons each step, chosen at random. It feels like sabotage. It is actually one of the most beloved cures for overfitting ever invented (Srivastava & Hinton, 2014).

**Picture a group project where the teacher randomly sends one teammate home every day.** If your group always leans on the one genius kid, you're in trouble the day *they* stay home. So the teacher rolls a die each morning and benches a random member. Now *everyone* has to learn a bit of everything — nobody can coast, and no single person is a make-or-break crutch. The team gets robust: it works no matter who shows up. **[[Dropout||during training, randomly switch off a fraction of neurons each step so no single neuron becomes a crutch — a cheap way to fake averaging many networks]]** does this to neurons. By randomly benching some each step, it stops the network from leaning too hard on any one neuron. No neuron gets to be the irreplaceable genius, so the network spreads its knowledge out — and a spread-out network generalizes instead of memorizes.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="Two side-by-side network snapshots for dropout at training time. Left: a small network with all four hidden neurons active and connected. Right: the same network on a different training step with two neurons crossed out and greyed, their connections faded. A caption says each step randomly benches some neurons so no single one becomes a crutch."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="130" y="20" fill="#6B645E" font-size="10">step A: bench neurons 2 &amp; 4</text><text x="390" y="20" fill="#6B645E" font-size="10">step B: bench neurons 1 &amp; 3</text><g stroke="#cfc8bd" stroke-width="0.8"><line x1="55" y1="100" x2="115" y2="45"/><line x1="55" y1="100" x2="115" y2="80"/><line x1="55" y1="100" x2="115" y2="115"/><line x1="55" y1="100" x2="115" y2="150"/><line x1="115" y1="45" x2="185" y2="100"/><line x1="115" y1="80" x2="185" y2="100"/><line x1="115" y1="115" x2="185" y2="100"/><line x1="115" y1="150" x2="185" y2="100"/></g><circle cx="55" cy="100" r="9" fill="#E8EEF6" stroke="#4a6fa5"/><circle cx="115" cy="45" r="10" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><circle cx="115" cy="80" r="10" fill="#eee" stroke="#bbb" stroke-dasharray="3 2"/><text x="115" y="83" fill="#C93B3B">✕</text><circle cx="115" cy="115" r="10" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><circle cx="115" cy="150" r="10" fill="#eee" stroke="#bbb" stroke-dasharray="3 2"/><text x="115" y="153" fill="#C93B3B">✕</text><circle cx="185" cy="100" r="9" fill="#F6EEF6" stroke="#a54a8f"/><g stroke="#cfc8bd" stroke-width="0.8"><line x1="315" y1="100" x2="375" y2="45"/><line x1="315" y1="100" x2="375" y2="80"/><line x1="315" y1="100" x2="375" y2="115"/><line x1="315" y1="100" x2="375" y2="150"/><line x1="375" y1="45" x2="445" y2="100"/><line x1="375" y1="80" x2="445" y2="100"/><line x1="375" y1="115" x2="445" y2="100"/><line x1="375" y1="150" x2="445" y2="100"/></g><circle cx="315" cy="100" r="9" fill="#E8EEF6" stroke="#4a6fa5"/><circle cx="375" cy="45" r="10" fill="#eee" stroke="#bbb" stroke-dasharray="3 2"/><text x="375" y="48" fill="#C93B3B">✕</text><circle cx="375" cy="80" r="10" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><circle cx="375" cy="115" r="10" fill="#eee" stroke="#bbb" stroke-dasharray="3 2"/><text x="375" y="118" fill="#C93B3B">✕</text><circle cx="375" cy="150" r="10" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><circle cx="445" cy="100" r="9" fill="#F6EEF6" stroke="#a54a8f"/><text x="260" y="178" fill="#6B645E">different neurons benched each step → nobody becomes a crutch</text></g></svg>
%%%

**What the group-project picture gets right:** benching a random member each day forces every member to be useful, so the team never depends on one person — exactly what makes the network robust. **Where it breaks down:** in a real group the benched kid genuinely goes home; in dropout the "benched" neuron is only zeroed for that one step, and it's fully back — with its learning intact — on the very next step.

#### Why it really works: it's a cheap way to average many networks
Here's the deeper reason, and it's beautiful. Every step, dropout switches off a *different* random set of neurons — so it's really training a slightly *different* network each time. Over thousands of steps, you've quietly trained a huge crowd of overlapping networks that all share the same weights. The old, expensive way to fight overfitting was to train *many* separate networks and average their answers — a so-called **[[ensemble||a crowd of models whose answers are averaged; averaging cancels out each model's private mistakes]]**. Averaging cancels each one's private mistakes, like asking a big crowd to guess and taking the average. Dropout gives you that ensemble magic for almost free, inside one network. That ancestry is *why* dropout works.

#### The one knob, and the catch that makes it correct
Dropout has a single dial: the **dropout rate `p`**, the fraction you switch off each step (a common choice is `p = 0.5` — bench half). Bigger `p` means stronger medicine. But there's a catch you have to get right, and it's where beginners trip. During training, if you drop half the neurons, only half the signal flows through — the total is *quieter* than it will be at test time when everyone's back. If you did nothing about that, the network would see one signal strength while practicing and a *different, louder* one when predicting, and it would misbehave. The fix is a small scaling. Watch the numbers line up:

%%% mathladder
title: why we scale by 1/(1−p) at training time
words: dropping neurons makes the signal quieter; scale the survivors up so the average signal matches what test time will see
formula: kept-signal is scaled by 1 / (1 − p)
numbers: p = 0.5 → survivors ×(1/0.5) = ×2; so 4 neurons each worth 1, drop 2, keep 2 → 2×(1×2) = 4 — same total as all 4 kept
sanity: at test time you keep ALL neurons and do NOT scale; because training already scaled up, the two settings now produce the same average signal
%%%

So the rule is a pair: **train time — randomly drop, and scale the survivors up by 1/(1−p); test time — keep everyone, no scaling.** Get that pair right and the model behaves identically whether dropout is on or off. Which raises the obvious question we tackle next: how does the code know *which* mode it's in?

@@@ concept id=c7 tag="Train vs eval" title="The switch: dropout on for practice, off for real" gotit="Got the switch"
Dropout must be **on** while the model practices and **off** when it makes real predictions. So the code needs a switch — one flip that says "we're practicing" versus "this is for real." Getting this switch wrong is the single most common dropout bug, and it's silent: nothing crashes, the predictions just quietly go wrong.

**Think of a light dimmer with two labelled positions.** There's a "rehearsal" setting where the stage lights flicker on and off at random (that's practice — dropout on), and a "showtime" setting where every light is steady and full (that's the real performance — dropout off). Flick it to *rehearsal* and run the actual show, and the audience sees a flickering, random mess. **[[Train mode||model.train() — the setting where dropout is ON: neurons are randomly switched off each step]]** and **[[eval mode||model.eval() — the setting where dropout is OFF: every neuron is kept so predictions are steady and repeatable]]** are exactly those two positions. You call `model.train()` before practicing (dropout on) and `model.eval()` before predicting (dropout off). Flip the wrong one and your show flickers.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A two-position switch drawn as a dimmer. Left position labeled train mode / rehearsal shows a network with some neurons flickering off and the words dropout ON. Right position labeled eval mode / showtime shows all neurons steady and lit with the words dropout OFF. A red warning shows the bug: switch stuck on rehearsal during the real show gives random, flickering predictions."><g font-family="monospace" font-size="9" text-anchor="middle"><rect x="25" y="28" width="220" height="115" rx="10" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.6"/><text x="135" y="48" fill="#8A6D3B" font-size="11">model.train()  — rehearsal</text><circle cx="90" cy="88" r="9" fill="#E8F5EE" stroke="#2D8B55"/><circle cx="135" cy="88" r="9" fill="#eee" stroke="#bbb" stroke-dasharray="3 2"/><text x="135" y="91" fill="#C93B3B">✕</text><circle cx="180" cy="88" r="9" fill="#eee" stroke="#bbb" stroke-dasharray="3 2"/><text x="180" y="91" fill="#C93B3B">✕</text><text x="135" y="122" fill="#8A6D3B" font-size="10">dropout ON (flickering)</text><rect x="275" y="28" width="220" height="115" rx="10" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.6"/><text x="385" y="48" fill="#1a5c38" font-size="11">model.eval()  — showtime</text><circle cx="340" cy="88" r="9" fill="#E8F5EE" stroke="#2D8B55"/><circle cx="385" cy="88" r="9" fill="#E8F5EE" stroke="#2D8B55"/><circle cx="430" cy="88" r="9" fill="#E8F5EE" stroke="#2D8B55"/><text x="385" y="122" fill="#1a5c38" font-size="10">dropout OFF (all steady)</text><text x="260" y="164" fill="#C93B3B">bug: run the real show in rehearsal mode → flickering, random predictions</text></g></svg>
%%%

**What the dimmer picture gets right:** it's one switch with two clear settings, and using the rehearsal setting for the real show gives a flickering, unreliable result. **Where it breaks down:** a stage dimmer only changes brightness; the mode switch changes *behavior* — it decides whether neurons get randomly zeroed at all, and (from the last concept) whether that 1/(1−p) scaling is applied.

#### The silent bug, seen side by side
Forgetting `model.eval()` doesn't throw an error. The model still runs — it just keeps randomly zeroing neurons while predicting, so the *same input* gives a *different* answer each time. Watch what "on" versus "off" does to one fixed prediction:

%%% demo id=evalbug label="predict on vs off"
code: x = same_image           # feed the identical picture three times
code: # --- WRONG: forgot model.eval(), dropout still ON ---
code: model.train(); [model(x) for _ in range(3)]
code: # --- RIGHT: model.eval(), dropout OFF ---
code: model.eval();  [model(x) for _ in range(3)]
out: dropout ON  → [ "7", "1", "7" ]    # same image, different answers ⚠️
out: dropout OFF → [ "7", "7", "7" ]    # steady, repeatable ✓
take: <b>Same picture, different answers = the eval-mode bug.</b> With dropout left on, prediction is random and worse. `model.eval()` flips it off so predictions are steady and correct. Always switch to eval before you measure or deploy.
%%%

!!! c-warn ⚠️
<b>The bug (silent):</b> leaving the model in train mode at prediction time keeps dropout ON, so predictions are noisy and worse — and nothing errors out. <b>The remedy:</b> call `model.eval()` before validating, testing, or deploying, and `model.train()` again before you resume practicing. Flip the switch every time you change what you're doing.
!!!

@@@ concept id=c8 tag="Reading the curves" title="Reading the curves like a doctor: four diagnoses" gotit="Got the diagnoses"
You now have a dashboard and know its worst disease. The last skill is reading it like a doctor reads a chart — a handful of tell-tale patterns, each with a cause and a cure. This is the part that turns you from "ran a training loop" into "can fix a training loop." We'll add the readings one at a time, so each shape sinks in before the next.

**Think of a nurse reading a patient's vital-signs monitor.** A steady falling fever means recovery. A line that won't budge means the medicine isn't working. A wild spike off the chart means an emergency. And "feels fine on paper but can't walk" means the chart wasn't telling the whole story. Your loss curves have the same readings — let's collect them one shape at a time, then see all four side by side.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="A nurse's vital-signs monitor with a heartbeat line, and a caption saying a loss curve is read the same way, by shape."><g font-family="monospace" font-size="9" text-anchor="middle"><rect x="20" y="18" width="480" height="70" rx="10" fill="#0f1720" stroke="#2D8B55" stroke-width="1.4"/><polyline points="35,60 90,60 105,35 120,80 140,52 200,52 215,30 230,78 250,55 320,55 340,25 360,82 380,58 470,58" fill="none" stroke="#39d98a" stroke-width="1.6"/><text x="260" y="106" fill="#6B645E" font-size="9">a loss curve is read like a vital sign — by its SHAPE, not one reading</text></g></svg>
%%%

**What the vital-signs picture gets right:** each pattern has one likely cause and one first move, so you diagnose by *shape* instead of guessing. **Where it breaks down:** a nurse has textbook thresholds; loss curves rarely give clean numbers — you read the *trend and the gap*, not an exact value.

#### Reading 1 — healthy: both curves fall together
The good news first. When training loss and validation loss slide **down together**, staying close, the model is genuinely learning. This is the shape you're aiming for — no action needed, keep going.

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Healthy diagnosis: a green training curve and a blue dashed validation curve both fall together from top-left toward the bottom-right, staying close. Labeled healthy, both fall together, no action needed."><g font-family="monospace" font-size="9" text-anchor="middle"><rect x="20" y="14" width="480" height="92" rx="8" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="260" y="30" fill="#1a5c38" font-size="10">✓ HEALTHY — both curves fall together</text><polyline points="45,50 140,66 250,78 470,86" fill="none" stroke="#2D8B55" stroke-width="2"/><polyline points="45,54 140,70 250,82 470,90" fill="none" stroke="#4a6fa5" stroke-width="1.5" stroke-dasharray="3 2"/><text x="360" y="66" fill="#2D8B55" font-size="9">train ↓</text><text x="360" y="102" fill="#4a6fa5" font-size="9">validation ↓ (close behind)</text></g></svg>
%%%

That's the target. Now the three things that can go wrong — one reading at a time.

#### Reading 2 — stuck and high: underfitting → raise the learning rate
Sometimes both curves just **sit flat and high** and barely move. The model isn't improving even on the data it *can* see. That's **[[underfitting||the opposite of overfitting: the model is too weak or trained too little, so it does poorly even on the training data]]** — the model is too small, trained too little, or the **learning rate is too small** so each step barely nudges the weights. Here's the twist that trips people up: **dropout does NOT fix this.** Dropout fights *memorizing*; if the model isn't even memorizing yet, adding dropout only makes a too-weak model weaker. Cure: raise the learning rate, train longer, or grow the model. Watch a stuck run refuse to budge:

%%% demo id=underfit label="a stuck run"
code: # both losses barely move, epoch after epoch
code: epoch:            1     2     3     4     5
code: train_loss:      2.29  2.27  2.26  2.26  2.25
code: validation_loss: 2.30  2.29  2.28  2.28  2.27
out: train stays HIGH:      2.29 → 2.25   (flat)
out: validation stays HIGH: 2.30 → 2.27   (flat)
out: diagnosis: UNDERFITTING — dropout would NOT help
take: <b>Flat and high on BOTH = underfitting.</b> The model barely learns even the training data. Cure: raise the learning rate, train longer, or grow the model — NOT dropout.
%%%

#### Reading 3 — shooting off the chart: diverging to NaN → lower the learning rate
The opposite of stuck: the loss **blows up**, climbing fast and often landing on `NaN` (not-a-number, meaning the math overflowed). The cause is the mirror image of Reading 2 — the **learning rate is too large**, so each step overshoots so hard the numbers explode. Cure: **lower the learning rate.** Watch a run detonate:

%%% demo id=diverge label="a diverging run"
code: # learning rate far too big — each step overshoots harder
code: epoch:        1      2       3        4       5
code: train_loss:  2.3    9.8     140.0    8.7e6   NaN
out: loss climbs:   2.3 → 9.8 → 140 → 8,700,000
out: then overflows: → NaN  💥
out: diagnosis: DIVERGING — learning rate too large
take: <b>Loss shooting UP toward NaN = the learning rate is too big.</b> Each step overshoots and the numbers explode. Cure: lower the learning rate (a common first move: cut it by 10×).
%%%

#### Reading 4 — the overfitting bottom: early stopping
Back to the disease you already know — overfitting, where training loss keeps falling but validation turns back up. Besides dropout, it has a second, dead-simple cure: **[[early stopping||just stop training at the point where validation loss stopped improving — the model was at its best there]]**. Literally stop at the epoch where the validation curve hit its lowest point, before the gap grew, and keep the model from *that* moment. Below, the run from before — the star marks where you should have stopped:

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Overfitting with an early-stopping marker. A green training curve keeps falling to the bottom-right. A blue validation curve falls, reaches a lowest point marked with a star labeled stop HERE, then turns back up. Everything after the star is shaded and labeled wasted / memorizing."><g font-family="monospace" font-size="9" text-anchor="middle"><rect x="20" y="14" width="480" height="102" rx="8" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.4"/><text x="260" y="30" fill="#8A6D3B" font-size="10">⚠ OVERFIT — early stopping: keep the model at the validation bottom</text><path d="M45 46 L140 62 L240 80 L470 96" fill="none" stroke="#2D8B55" stroke-width="2"/><path d="M45 50 L140 66 L235 78 L320 84 L470 104" fill="none" stroke="#4a6fa5" stroke-width="1.8"/><rect x="235" y="40" width="235" height="70" fill="#C93B3B" opacity="0.06"/><line x1="235" y1="40" x2="235" y2="110" stroke="#C93B3B" stroke-width="0.8" stroke-dasharray="3 3"/><text x="235" y="76" fill="#C93B3B" font-size="13">★</text><text x="205" y="102" fill="#C93B3B" font-size="9">stop HERE</text><text x="380" y="60" fill="#C93B3B" font-size="9">everything after = wasted (memorizing)</text></g></svg>
%%%

#### All four readings, side by side
Now that you've met each shape on its own, here they are on one chart — this is the picture to keep. Glance at the shape, name the diagnosis, reach for the cure:

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A 2 by 2 chart of four loss-curve diagnoses. Top-left healthy: both curves fall together. Top-right overfitting: train falls but validation rises. Bottom-left stuck/underfitting: both curves stay flat and high. Bottom-right diverging: the curve shoots up off the chart to NaN."><g font-family="monospace" font-size="8.5" text-anchor="middle"><rect x="20" y="18" width="235" height="85" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.3"/><text x="137" y="32" fill="#1a5c38" font-size="9">✓ HEALTHY — both fall together</text><polyline points="35,48 90,62 150,74 240,82" fill="none" stroke="#2D8B55" stroke-width="1.8"/><polyline points="35,52 90,66 150,79 240,88" fill="none" stroke="#4a6fa5" stroke-width="1.4" stroke-dasharray="3 2"/><rect x="265" y="18" width="235" height="85" rx="6" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.3"/><text x="382" y="32" fill="#8A6D3B" font-size="9">⚠ OVERFIT — cure: dropout / stop early</text><polyline points="280,48 330,62 400,78 490,92" fill="none" stroke="#2D8B55" stroke-width="1.8"/><polyline points="280,52 330,64 380,66 440,80 490,94" fill="none" stroke="#C99A12" stroke-width="1.6"/><rect x="20" y="113" width="235" height="85" rx="6" fill="#F0EFEC" stroke="#8a857c" stroke-width="1.3"/><text x="137" y="127" fill="#5c584f" font-size="9">— STUCK / UNDERFIT — raise LR / grow model</text><polyline points="35,150 100,149 170,151 240,150" fill="none" stroke="#8a857c" stroke-width="1.8"/><text x="137" y="185" fill="#5c584f">flat &amp; high on BOTH</text><rect x="265" y="113" width="235" height="85" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.3"/><text x="382" y="127" fill="#C93B3B" font-size="9">✕ DIVERGING (NaN) — lower LR</text><polyline points="280,190 330,180 370,150 400,110 415,120" fill="none" stroke="#C93B3B" stroke-width="1.8"/><text x="450" y="150" fill="#C93B3B">→ ∞</text></g></svg>
%%%

#### The honest limits — hold these two
Two truths to carry out of today, because they keep you humble:

!!! c-info 🎯
<b>1. A low loss number alone does NOT prove a good model.</b> The loss only measures the examples you scored. A tiny *training* loss can hide a memorizer; you can only trust a model after checking its *held-out* loss — and even that is a proxy for real-world quality, not a guarantee. <br><b>2. Dropout is not free.</b> Because it hides neurons each step, the network learns more slowly (it needs more epochs), and if a model already generalizes fine, adding dropout can *hurt* — you'd be treating a disease it doesn't have. Reach for dropout when you actually see the overfitting gap, not by reflex.
!!!

That's the full toolkit: measure honestly, read the curves, and apply the right cure. Let's gather the whole day onto one page.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you just learned to read a model's honesty. Here's the whole day in one picture. Follow the arc left to right: a guess becomes one **loss** number, training pushes the *average* loss **down**, but the number that matters is the **held-out** loss — because a model can **overfit** (memorize), which **dropout** cures at train time and **eval mode** turns off at test time.

%%% svg
<svg viewBox="0 0 520 235" role="img" aria-label="A one-page recap of the day. Top row: a guess becomes a loss number (a golf gap), MSE for numbers and cross-entropy for categories. Middle: a two-curve dashboard where training loss falls but validation is the honest score, and when they split it is overfitting. Bottom: dropout benches random neurons at train time to cure it, the train/eval switch turns it off for real predictions, and a reminder that low loss alone is not proof."><g font-family="monospace" font-size="8.5" text-anchor="middle"><text x="260" y="15" fill="#6B645E" font-size="10">guess → measure the gap → one LOSS number → push its AVERAGE down</text><rect x="35" y="26" width="210" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.3"/><text x="140" y="41" fill="#1a5c38">MSE — for numbers</text><text x="140" y="54" fill="#6B645E">square the gap (regression)</text><rect x="275" y="26" width="210" height="34" rx="5" fill="#FDF3E8" stroke="#C99A12" stroke-width="1.3"/><text x="380" y="41" fill="#8A6D3B">cross-entropy — for categories</text><text x="380" y="54" fill="#6B645E">−log(p); hates confident+wrong</text><line x1="55" y1="78" x2="55" y2="140" stroke="#6B645E" stroke-width="1"/><line x1="55" y1="140" x2="270" y2="140" stroke="#6B645E" stroke-width="1"/><polyline points="60,84 100,104 150,120 210,132 265,136" fill="none" stroke="#2D8B55" stroke-width="1.6"/><polyline points="60,88 100,108 150,116 200,124 240,132 265,138" fill="none" stroke="#C99A12" stroke-width="1.4"/><text x="150" y="156" fill="#6B645E">train ↓ vs VALIDATION (the honest score)</text><text x="150" y="168" fill="#C93B3B">curves SPLIT = overfitting</text><rect x="290" y="78" width="200" height="62" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.3"/><text x="390" y="94" fill="#5E5191">CURE · dropout</text><text x="390" y="108" fill="#6B645E">bench random neurons each step</text><text x="390" y="121" fill="#6B645E">= cheap ensemble; scale ×1/(1−p)</text><text x="390" y="134" fill="#6B645E">also: early stopping</text><line x1="20" y1="182" x2="500" y2="182" stroke="#E5DDD0" stroke-width="1"/><text x="105" y="200" fill="#2D8B55">train() = dropout ON</text><text x="260" y="200" fill="#1a5c38">eval() = dropout OFF (real)</text><text x="425" y="200" fill="#C93B3B">low loss ≠ good model</text><text x="150" y="220" fill="#8a857c">flat+high = underfit → raise LR/grow</text><text x="390" y="220" fill="#C93B3B">→ NaN = LR too big → lower LR</text></g></svg>
%%%

**Where the golf/exam pictures finally break down:** a golfer reads one gap and a student takes one exam, but a model juggles all of this at once across thousands of tiny nudges — pushing a loss down, watching two curves, benching random neurons, flipping a mode switch — and that machinery running together is what lets it learn things no person hand-tuned. Keep the two cheat-sheets below as your map back into today.

#### Cheat-sheet · the ideas, failures, and cures
%%% table
:: Piece / failure :: In one line :: Remember
loss :: one number for how wrong a guess is :: smaller = better; it's what training shrinks
MSE :: average of squared gaps :: use for numbers (regression); big misses hurt most
cross-entropy :: −log(p) on the true class :: use for categories; explodes on confident+wrong
average loss :: mean over the data, not one batch :: the calm number you plot per epoch
validation (held-out) loss :: score on data never trained on :: THE number that matters, not train loss
overfitting :: memorizes training noise (too much capacity) :: symptom = train ↓ while validation ↑
dropout :: bench random neurons each step :: headline cure; cheap ensemble; only at train time
1/(1−p) scaling :: scale survivors up while training :: so train and test see the same signal
train() vs eval() :: switch: dropout ON vs OFF :: eval() before predicting — or predictions flicker
early stopping :: stop where validation bottomed :: second, simplest cure for overfitting
underfitting :: flat+high on BOTH curves :: too small / too little; dropout does NOT fix it
loss stuck :: barely moves :: cause: LR too small → raise it (or grow model)
loss → NaN :: shoots off the chart :: cause: LR too large → lower it
low loss alone :: proves nothing about new data :: check held-out; dropout isn't free
%%%

#### Cheat-sheet · the words you met today
%%% jargon
loss | one number for how far a guess landed from the right answer — smaller is better
MSE (mean squared error) | average of the squared gaps — the loss for number answers (regression)
cross-entropy | −log(p) on the true class — the loss for category answers; punishes confident-and-wrong hardest
softmax | turns raw scores into probabilities that are positive and add up to 1 (paired with cross-entropy)
average loss | the mean loss over the data — the steady number you push down each epoch
epoch | one full pass through all the training data
training set | the examples the model practices on
validation (held-out) set | examples the model never trains on — the honest test of real learning
overfitting | memorizing the training examples (even their noise) instead of the pattern
dropout | randomly switch off a fraction of neurons each training step so none becomes a crutch
dropout rate p | the fraction switched off each step — bigger = stronger regularization
ensemble | a crowd of models whose answers are averaged; dropout fakes this cheaply
train mode / model.train() | dropout ON — the practice setting
eval mode / model.eval() | dropout OFF — the real-prediction setting
early stopping | halt training at the epoch where validation loss stopped improving
underfitting | model too weak or trained too little — does poorly even on the training data
%%%

That's the whole day. Tomorrow you rebuild this exact model in **PyTorch**, where `loss`, `model.train()`, `model.eval()`, and `Dropout` are all one-line tools you now understand from the inside.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each. If one trips you up, that's just a cue to scroll back up — not a grade, and definitely not a failure. Getting one wrong is how the ideas actually stick.
%%% quiz
q: Your model predicts a digit and puts 99% on "1" — but the true answer is "7". Which loss punishes this the HARDEST, and why? | a:1 | MSE, because it squares the gap | Cross-entropy, because it explodes when the model is confident and wrong | Neither — 99% is a great guess | Both punish it equally | fb: Cross-entropy is −log(p) on the TRUE class. Here p(true)=p(7) is tiny, so the loss is huge. That steep punishment for being confident-and-wrong is exactly why cross-entropy is the go-to loss for categories.
q: Over many epochs your TRAINING loss keeps falling toward zero, but your VALIDATION loss bottomed out and is now rising. What is happening? | a:2 | The model is underfitting | The learning rate is too large | The model is overfitting — memorizing training noise instead of learning the pattern | Training is finished and the model is great | fb: Train ↓ while validation ↑ is the classic overfitting signature. The growing gap means it's memorizing the training set. The best model was at the bottom of the validation curve — cures: dropout, or early stopping there.
q: Why is dropout switched ON during training but OFF (with all neurons kept) when the model makes real predictions? | a:2 | It's a bug — dropout should always be on | Dropout is only for speed | Dropping neurons forces the network not to rely on any one unit while learning; at prediction time you want the full, steady network, and the 1/(1−p) train-time scaling makes the two agree | Turning it off saves memory | fb: On during training = the ensemble/anti-crutch effect. Off at test = steady, repeatable predictions. Training already scales survivors by 1/(1−p), so keeping everyone at test time gives the same average signal. Flip to eval mode with model.eval().
q: Your loss sits flat and HIGH on BOTH the training and validation curves and barely moves. Will adding dropout fix it? | a:1 | Yes — dropout always helps | No — this is underfitting (too small / LR too low / trained too little); dropout fights memorizing, so it would only make a too-weak model weaker | Yes, if you set p = 0.9 | No, you must switch to MSE | fb: Flat-and-high on BOTH curves is underfitting, the opposite of overfitting. Dropout cures memorizing; there's no memorizing here. The real fixes are raising the learning rate, training longer, or growing the model.
%%%

@@@ produce id=produce tag="Produce" title="Predict the two curves, then run it and watch them split" gotit="Done"
Now it's your turn to *see* overfitting happen and then cure it with your own hands. Watching the validation curve peel away from the training curve — and then watching dropout pull it back — is what makes today truly stick.

**Before you write any code, make three predictions.** Guessing first is what turns "I watched it" into "I understand it," so don't skip this:

1. If you train a big model on a *small* dataset for many epochs with NO dropout, what will the training loss and the validation loss each do?
2. When the two curves split, which one keeps falling and which one turns back up?
3. After you add dropout, will the *gap* between the two curves get bigger or smaller?

Write your three guesses down. Then run it and **watch** whether you were right. Pick one path.

#### Option A · write it yourself
Create `sessions/m04-first-model-mlp/day-04-training-loss-dropout/experiment.py`. Take your Day 3 mini-batch loop and split the data into a training set and a held-out **validation** set. Each epoch, record BOTH the average training loss and the average validation loss (remember to switch to eval mode — dropout off — before scoring validation). Train a deliberately over-powered model on a small slice with **no dropout** and watch the two curves **split** — training loss diving while validation loss turns back up (that's overfitting). Then add a dropout layer (`p = 0.5`, on only in train mode, off in eval mode) and re-run: the validation curve should stay lower for longer. Run with `python3 sessions/m04-first-model-mlp/day-04-training-loss-dropout/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Module 5 Day 4 artifact.

Create sessions/m04-first-model-mlp/day-04-training-loss-dropout/experiment.py that, with a comment on each step, demonstrates the loss, overfitting, and dropout:
1. Split the data into a TRAINING set and a held-out VALIDATION set (e.g. 80/20). Use a deliberately small training slice so overfitting shows up fast.
2. Use cross-entropy as the classification loss. Each epoch, record the AVERAGE training loss AND the average validation loss. Before scoring validation, switch to eval mode (dropout OFF); switch back to train mode before the next epoch.
3. Run A — NO dropout, an over-powered model, many epochs: print both losses per epoch and point out the epoch where the validation loss stops falling and turns UP while the training loss keeps diving (this is overfitting).
4. Run B — add a dropout layer (p = 0.5, applied only in train mode, and remember the 1/(1-p) scaling so train and test signals match): re-run and show the validation loss stays lower for longer / the train-vs-validation gap is smaller.
5. Add a tiny sanity demo: feed the SAME input through the model twice with dropout left ON vs with eval mode ON, and show the ON version gives different answers each time while eval mode is steady — the model.eval() bug.
6. End with a printed reminder: a low TRAINING loss alone does not prove a good model — the held-out loss is the honest number, and dropout is not free (slower to train, can hurt an already-generalizing model).
Then run it and paste the per-epoch train/validation losses (Run A vs Run B) at the bottom as a comment.
%%%

#### What you should see (check your prediction)
Here's what to look for when it runs:

- **Run A (no dropout):** the **training loss dives toward zero** while the **validation loss bottoms out and turns back up** — the two curves visibly split. Did you predict which one rose?
- **The gap:** the growing distance between the curves *is* overfitting, drawn live. The best model was at the bottom of the validation curve, not at the end.
- **Run B (with dropout):** the validation curve stays lower for longer and the gap shrinks — dropout working. Did you guess the gap would get smaller?
- **The eval-mode demo:** the same picture gives different answers with dropout left on, and steady answers in eval mode. That's the silent bug you now know to avoid.
- **The thing worth sitting with:** a tiny training loss felt like winning in Run A — and it was hiding a model that memorized. The *held-out* loss is the only honest scorecard. That honesty is the whole point of today.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m04-first-model-mlp/day-04-training-loss-dropout/log.md`: (1) what a loss is, and how MSE and cross-entropy differ, one sentence each; (2) how you SPOT overfitting on the two curves, and its two cures; (3) why dropout is on at train time but off (with model.eval()) at test time.
!!!

@@@ fin
