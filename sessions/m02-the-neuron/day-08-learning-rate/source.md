---
quest_id: wf3-d05-lr
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 8 — Learning-Rate Intuition"
module_label: "Module 2 · Train · Day 8"
title: "Learning-Rate Intuition"
subtitle: "The One Knob That Makes or Breaks Training"
brand_sub: "Foundations · M2 Day 8"
spine: "stepping stone"
nav_prev_href: "../day-07-optimizers/lesson.html"
nav_prev_label: "Optimizers (SGD, Momentum, Adam)"
nav_next_href: "../day-09-train-val-test/lesson.html"
nav_next_label: "Train / Val / Test & Overfitting"
fin_title: "Module 2 · Day 8 complete! 🏆"
fin_body: "Nice work — you've built real intuition for the <b>learning rate</b>: too small and training crawls, too big and it overshoots (even to NaN), and the sweet spot in between — plus warmup and decay schedules — is where training actually works.<br>Next up: <b>Train / Val / Test</b> — did the model really learn, or just memorize?"
notebook_yardstick: null
coverage_topics:
  - {topic: learning rate scalar, keywords: [learning rate, step size, how big a step, scales the gradient, one knob]}
  - {topic: update rule, keywords: [new weight = old weight, w = w - lr, minus the gradient, gradient gives direction]}
  - {topic: lr controls step downhill, keywords: [how big a step you take downhill, step size on the loss surface, how far you move]}
  - {topic: lr too small failure, keywords: [too small, crawls, barely moves, painfully slow, far more steps]}
  - {topic: lr too small remedy, keywords: [increase the learning rate, raise the rate, larger learning rate]}
  - {topic: lr too large failure, keywords: [too large, overshoots, bounces around, jumps past the bottom]}
  - {topic: lr too large remedy, keywords: [decrease the learning rate, lower the rate, smaller learning rate]}
  - {topic: divergence exploding loss, keywords: [diverges, blows up, infinity, NaN, exploding loss, grows each update]}
  - {topic: divergence remedy, keywords: [lower the learning rate, clip]}
  - {topic: find lr by watching loss curve, keywords: [try a range of values, watch the loss curve, curve shape, smoothly down, flat, spiky]}
  - {topic: order of magnitude sweep, keywords: [1.0, 0.1, 0.01, 0.001, order of magnitude, powers of ten, sweep]}
  - {topic: read loss vs step curves, keywords: [loss-vs-step, loss curve, too small looks flat, just right, too large spiky, diagnose]}
  - {topic: lr schedule decay, keywords: [learning-rate schedule, decay, shrink it over time, big early, small late, warmup]}
  - {topic: connection to single neuron loop, keywords: [single neuron, gradient-descent loop, weights and bias, one knob multiplying the gradient]}
---

@@@ hero
@lede Picture crossing a stream by hopping on **stepping stones**. Take teeny-tiny hops and you will get across — but so slowly, one careful little shuffle at a time. Leap way too far and you fly right past the next stone and splash into the water. There is a *just-right* hop that gets you across quickly and safely. Here is the surprising part: teaching a neural network is exactly this hop-across-the-stream, and the size of each hop has a name — the **learning rate**. It is the single most important knob in all of machine learning. Set it well and a model like ChatGPT learns in hours; set it wrong and training either crawls for a week or explodes into nonsense on the first few steps. Today you get to feel that knob with your own hands.
@goal Together we will build a real gut feel for the learning rate: what it *is* (the size of each downhill hop), what goes wrong when it is too small (crawling) or too big (overshooting, even blowing up to NaN), how to *find* a good one by watching the loss curve, and one clever trick — shrinking the hop over time — that gives fast progress early and a gentle landing at the end. Every idea comes with a picture first and a slider you can drag. No heavy math — just intuition you can steer.

@@@ concept id=c1 tag="What it is" title="The learning rate — the size of each downhill hop" gotit="Got what it is"
Let us start with the one thing the learning rate actually does. Back on the stream, every **stepping stone** you hop to is a little lower than the last — you are working your way *down* toward the far bank. Two separate things decide your next hop. First, *which way* is downhill — you look at the ground and it tips one way. Second, *how far* you actually hop in that direction. The learning rate is only that second thing: **how far you hop each time.** It does not pick the direction; it just scales how big a move you make once the direction is chosen. In training, "downhill" means toward less [[loss||the model's "how wrong am I" score — training tries to make it smaller]], the direction comes from the [[gradient||the slope of the loss: which way the loss goes up, and how steeply. Training moves the OPPOSITE way, downhill]], and the learning rate is the one number that says how big a hop to take down that slope. In one line: the learning rate is **how big a step you take downhill** on the loss surface each update — it sets how far you move, not which way.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A person crossing a stream on stepping stones that step down toward the far bank. An arrow labelled direction (from the gradient) points down the slope. A curly brace labelled learning rate marks how far one hop reaches. A caption reads gradient picks the way, learning rate picks how far."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Gradient picks the WAY · learning rate picks HOW FAR</text><rect x="20" y="150" width="480" height="40" fill="#DCEBF0"/><text x="470" y="176" text-anchor="end" fill="#3A7C93" font-size="8.5">stream (the water = high loss)</text><ellipse cx="70" cy="70" rx="26" ry="9" fill="#B8AEA2"/><ellipse cx="180" cy="98" rx="26" ry="9" fill="#B8AEA2"/><ellipse cx="300" cy="126" rx="26" ry="9" fill="#B8AEA2"/><ellipse cx="430" cy="150" rx="26" ry="9" fill="#B8AEA2"/><text x="70" y="55" text-anchor="middle" fill="#6B645E" font-size="8">high</text><text x="430" y="135" text-anchor="middle" fill="#6B645E" font-size="8">low (far bank)</text><circle cx="70" cy="62" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text x="70" y="44" text-anchor="middle" fill="#1F6280" font-size="8.5">you are here</text><path d="M92 70 L162 96" stroke="#C93B3B" stroke-width="2"/><polygon points="162,96 151,92 152,101" fill="#C93B3B"/><text x="128" y="112" text-anchor="middle" fill="#C93B3B" font-size="8.5">direction = downhill (gradient)</text><path d="M92 82 q 44 12 90 24" fill="none" stroke="#9A7208" stroke-width="1.4"/><text x="140" y="132" text-anchor="middle" fill="#9A7208" font-size="8.5">↑ how far you hop = learning rate</text></g></svg>
%%%

**What the stepping-stone picture gets right:** the direction and the hop size really are two separate choices, and the learning rate is only the hop size — exactly like training. **Where it breaks down:** on a real stream the stones are already placed, so you cannot choose *any* hop length; in training you truly can dial the hop to any size you like — which is exactly why picking it well matters so much.

#### The one-line update rule
Here is the whole move in one plain sentence: **new weight = old weight − learning rate × gradient.** The gradient hands you the direction and steepness; the learning rate scales it; the minus sign turns you to face *downhill*. That is the same update you built in the training loop on earlier days — the learning rate is just the one number multiplying the gradient in it.

%%% formula
expr: new_weight = old_weight − learning_rate × gradient
note: gradient = the direction (which way is downhill, and how steep). learning_rate = how far you hop. The MINUS sign walks downhill. This is the exact same step that trains your single neuron's weights and bias.
%%%

Let us watch it with tiny real numbers so the sentence turns concrete. Say a weight is `old_weight = 2.0` and the gradient (the slope) is `4.0`. Change *only* the learning rate and watch how far the weight moves in one hop:

%%% demo id=onestep label="run it — one hop with lr = 0.1"
code: 2.0 - 0.1 * 4.0        # new_weight = old_weight - learning_rate * gradient
out: 1.6
take: <b>Only the learning rate scales the hop.</b> With `old_weight = 2.0` and `gradient = 4.0`: at lr = 0.1 the hop is 0.1×4.0 = 0.4, so the weight moves to 1.6. Try it in your head at other rates: lr = 0.01 gives a tiny 0.04 hop (→ 1.96, barely nudged), and lr = 0.5 gives a giant 2.0 hop (→ 0.0). Same weight, same gradient — the learning rate alone decides how far each hop reaches. This is the whole knob; the rest of today is what happens when it is too small, too big, or just right.
%%%

Here is the same one-hop story as a picture — one weight, one gradient, three learning rates, three hop lengths:

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A number line from 0 to 2. A dot starts at weight equals 2.0. Three arrows of different lengths hop left toward zero: a tiny arrow for learning rate 0.01 landing at 1.96, a medium arrow for learning rate 0.1 landing at 1.6, and a long arrow for learning rate 0.5 landing all the way at 0.0. Same weight and gradient, only the hop size changes."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One weight, one gradient — three learning rates, three hop sizes</text><line x1="60" y1="120" x2="460" y2="120" stroke="#B8AEA2" stroke-width="2"/><line x1="460" y1="114" x2="460" y2="126" stroke="#6B645E"/><text x="460" y="140" text-anchor="middle" fill="#6B645E" font-size="8">weight 2.0 (start)</text><line x1="60" y1="114" x2="60" y2="126" stroke="#6B645E"/><text x="60" y="140" text-anchor="middle" fill="#6B645E" font-size="8">0.0 (bottom)</text><path d="M456 96 L432 96" stroke="#C99A12" stroke-width="2.2"/><polygon points="432,96 440,92 440,100" fill="#C99A12"/><text x="452" y="86" text-anchor="end" fill="#9A7208" font-size="8">lr 0.01 → 1.96 🐢</text><path d="M456 72 L376 72" stroke="#2D8B55" stroke-width="2.2"/><polygon points="376,72 384,68 384,76" fill="#2D8B55"/><text x="452" y="62" text-anchor="end" fill="#1a5c38" font-size="8">lr 0.1 → 1.6 ✓</text><path d="M456 48 L60 48" stroke="#C93B3B" stroke-width="2.2"/><polygon points="60,48 68,44 68,52" fill="#C93B3B"/><text x="452" y="38" text-anchor="end" fill="#C93B3B" font-size="8">lr 0.5 → 0.0 (big hop)</text></g></svg>
%%%

So the learning rate is one small number that scales every hop. Small stories and big stories follow from that one choice — and the first story is what happens when you make the hops *too small*.

@@@ concept id=c2 tag="Too small = crawl" title="Too small — training crawls" gotit="Got the crawl"
Now the first thing that goes wrong. Imagine crossing that stream by only ever shuffling your foot forward *one centimeter* per hop. You will get to the far bank eventually — nothing bad happens, you never fall in — but it takes *forever*. That is a learning rate set **too small**. Each hop barely moves the weight, so the loss does drop, just achingly slowly. You might need thousands or millions of extra steps to reach the bottom that a bigger hop would have reached quickly.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A big U-shaped valley with a ball high on the left slope taking many tiny closely-spaced steps that have barely moved it downward. A label reads tiny hops, barely moving, and another reads still far from the bottom after many steps."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Too small: many tiny hops, barely moving</text><path d="M40 40 Q 160 176 260 162 Q 360 176 480 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="8.5" dy="12">bottom = smallest loss</text><g fill="#C99A12"><circle cx="86" cy="78" r="3"/><circle cx="92" cy="82" r="3"/><circle cx="98" cy="86" r="3"/><circle cx="104" cy="90" r="3"/><circle cx="110" cy="94" r="3"/><circle cx="116" cy="98" r="3"/></g><circle cx="116" cy="98" r="7" fill="#C99A12" stroke="#fff" stroke-width="1.4"/><text x="150" y="72" fill="#9A7208" font-size="9">tiny hops 🐢</text><text x="300" y="96" text-anchor="middle" fill="#6B645E" font-size="8.5">still far from the bottom after many steps</text></g></svg>
%%%

**What the shuffle picture gets right:** small hops are *safe* — you always head the right way and never overshoot — they are just painfully slow. **Where it breaks down:** on the stream you get tired; a computer does not get tired, it just wastes real time and money taking millions of near-useless steps.

#### Watch it crawl
Let us make "crawls" something you can *see*. Here is a tiny valley — loss `= weight²`, so the bottom is at `weight = 0` — started at `weight = 2.0`. With a too-small learning rate of `0.01`, one step moves the weight only a hair. **Predict:** will one step get anywhere near the bottom? Then run it:

%%% demo id=crawl label="run it — one step at a too-small lr = 0.01"
code: 2.0 - 0.01 * (2 * 2.0)      # one step downhill on loss = weight**2
out: 1.96
take: <b>One step crept from 2.00 to 1.96</b> — the loss barely dipped from 4.0 to 3.84. Five steps only reach ~1.81; it would take well over a hundred steps to reach the bottom at 0. Nothing is broken — it is just crawling. The fix is simple: raise the learning rate.
%%%

Now compare the crawl step-by-step against a healthy drop, so you can *see* how little ground a too-small rate covers. Each bar is the weight after one more step, starting at 2.0 and heading for the bottom at 0:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Two rows of bars showing the weight after each of five steps, starting at 2.0 and heading toward 0. The top gold row, too small at learning rate 0.01, barely shrinks: 2.0, 1.96, 1.92, 1.88, 1.84, 1.81. The bottom green row, a good rate at learning rate 0.4, plunges: 2.0, 0.4, 0.08, 0.016, and essentially 0. A caption contrasts barely moving with nearly there."><g font-family="monospace" font-size="9"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Weight after each step (start 2.0 → bottom 0)</text><text x="16" y="42" fill="#9A7208" font-size="9">🐢 lr 0.01</text><g fill="#C99A12"><rect x="80" y="34" width="118" height="12"/><rect x="80" y="50" width="116" height="12"/><rect x="80" y="66" width="113" height="12"/><rect x="80" y="82" width="111" height="12"/><rect x="80" y="98" width="109" height="12"/></g><text x="210" y="70" fill="#9A7208" font-size="8.5">…still ≈1.81 after 5 steps (barely moving)</text><text x="16" y="140" fill="#1a5c38" font-size="9">✓ lr 0.4</text><g fill="#2D8B55"><rect x="80" y="132" width="118" height="12"/><rect x="80" y="148" width="24" height="12"/><rect x="80" y="164" width="6" height="12"/></g><text x="210" y="156" fill="#1a5c38" font-size="8.5">≈0 after 3 steps (nearly there)</text><line x1="80" y1="28" x2="80" y2="182" stroke="#B8AEA2" stroke-dasharray="2,2"/><text x="80" y="196" text-anchor="middle" fill="#6B645E" font-size="8">bottom = 0</text><text x="198" y="196" text-anchor="middle" fill="#6B645E" font-size="8">start = 2.0</text></g></svg>
%%%

And the *shape* of the whole loss curve gives it away at a glance. A too-small rate makes a curve that slopes down so gently it looks almost **flat**:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two loss-versus-step curves. A gold curve labelled too small slopes down very gently and stays high, almost flat. A green curve labelled good rate drops steeply and flattens near the bottom. The x-axis is steps, the y-axis is loss."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The loss curve of a too-small rate looks almost FLAT</text><line x1="48" y1="30" x2="48" y2="152" stroke="#B8AEA2"/><line x1="48" y1="152" x2="500" y2="152" stroke="#B8AEA2"/><text x="22" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 22 94)">loss</text><text x="272" y="174" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><path d="M54 46 C 200 60, 360 74, 496 86" fill="none" stroke="#C99A12" stroke-width="2.4"/><text x="360" y="70" fill="#9A7208" font-size="9">too small → barely drops (looks flat) 🐢</text><path d="M54 46 C 130 130, 300 148, 496 150" fill="none" stroke="#2D8B55" stroke-width="2.4"/><text x="300" y="134" fill="#2D8B55" font-size="9">good rate → drops fast, then settles ✓</text></g></svg>
%%%

So: **too small → the loss barely drops, the curve looks flat. Cause:** each hop is a grain of sand. **Remedy:** increase the learning rate. But raise it too far and you meet the *opposite* problem — which is where it gets exciting.

@@@ concept id=c3 tag="Too big = overshoot" title="Too big — overshoot, bounce, and blow up" gotit="Got the overshoot"
Now the opposite failure, and it is the dramatic one. Back on the stream, imagine hopping so hard that you sail clean *over* the next **stepping stone** and land in the water past it — then you leap back the other way even harder and overshoot again, splashing back and forth and getting *wetter* each time instead of crossing. That is a learning rate set **too big**. Each hop is longer than the distance to the bottom of the valley, so you jump *past* the lowest point and land higher up the far wall. The loss, instead of going down, bounces around — and if the hops keep growing, it shoots up to infinity. The word for a number that has grown past infinity in the computer is [[NaN||"not a number" — what a computer prints when a calculation blew up past any real value, like infinity times infinity. Seeing NaN in your loss means training exploded]], and a loss of `NaN` is the classic sign of a learning rate set way too high.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A U-shaped valley. A ball starts on the left slope and takes huge hops that overshoot the bottom, landing higher on the far wall, then bounces back even higher on the left, the arrows growing longer each time, escaping upward out of the valley. A label reads jumps PAST the bottom, and another reads bounces higher each time then blows up."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Too big: each hop jumps PAST the bottom and bounces higher</text><path d="M40 34 Q 160 180 260 168 Q 360 180 480 34" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5" dy="12">bottom</text><path d="M120 118 Q 200 60 336 120" fill="none" stroke="#C93B3B" stroke-width="1.8"/><polygon points="336,120 325,113 330,124" fill="#C93B3B"/><path d="M336 120 Q 240 40 92 82" fill="none" stroke="#C93B3B" stroke-width="1.8"/><polygon points="92,82 103,80 100,90" fill="#C93B3B"/><path d="M92 82 Q 240 16 470 54" fill="none" stroke="#C93B3B" stroke-width="1.8"/><polygon points="470,54 459,52 463,62" fill="#C93B3B"/><circle cx="120" cy="118" r="7" fill="#C93B3B" stroke="#fff" stroke-width="1.4"/><text x="230" y="58" text-anchor="middle" fill="#C93B3B" font-size="9">bounces higher each time → blows up (NaN) 💥</text></g></svg>
%%%

**What the over-leaping picture gets right:** a hop longer than the valley overshoots the bottom every single time, and repeated overshoots grow instead of settling — that really is how divergence happens. **Where it breaks down:** a real person would notice they are soaked and *stop*; the computer keeps blindly applying the rule until the numbers overflow into NaN.

#### Watch it explode
Same tiny valley as before — loss `= weight²`, bottom at `weight = 0`, start at `weight = 2.0`. Now crank the learning rate up to `1.1` (too big). **Predict:** does the weight get closer to 0, or fling itself *past* the bottom? Then run one step:

%%% demo id=diverge label="run it — one step at a too-big lr = 1.1"
code: 2.0 - 1.1 * (2 * 2.0)      # one step downhill, lr way too big
out: -2.4
take: <b>The weight flew clean past the bottom.</b> It started at +2.0, aimed downhill toward 0, and overshot all the way to −2.4 — now *farther* from the bottom than it started (loss went 4.0 → 5.76, up not down). The next step flings it to +2.9, then −3.5, then +4.1… bouncing bigger and bigger until it overflows to <b>NaN</b>. This is <b>divergence</b> — and the fix is simply to lower the learning rate.
%%%

Here is the whole runaway as a picture — each bar is the weight after one more step, flipping sign and growing every time instead of shrinking toward the bottom:

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="A bar chart of the weight after each of six steps at a too-big learning rate. The bars alternate left and right of a center line at zero and grow taller each step: plus 2.0, minus 2.4, plus 2.9, minus 3.5, plus 4.1, minus 5.0. A label reads flips sign and grows every step, then blows up to NaN."><g font-family="monospace" font-size="9"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Weight after each step at lr = 1.1 (too big)</text><line x1="260" y1="28" x2="260" y2="176" stroke="#B8AEA2" stroke-dasharray="2,2"/><text x="260" y="190" text-anchor="middle" fill="#6B645E" font-size="8">bottom = 0</text><g fill="#C93B3B"><rect x="200" y="34" width="60" height="14"/><rect x="260" y="54" width="72" height="14"/><rect x="173" y="74" width="87" height="14"/><rect x="260" y="94" width="104" height="14"/><rect x="136" y="114" width="124" height="14"/><rect x="260" y="134" width="149" height="14"/></g><text x="340" y="46" fill="#C93B3B" font-size="8">+2.0</text><text x="340" y="66" fill="#C93B3B" font-size="8">−2.4</text><text x="140" y="86" fill="#C93B3B" font-size="8" text-anchor="end">+2.9</text><text x="372" y="106" fill="#C93B3B" font-size="8">−3.5</text><text x="128" y="126" fill="#C93B3B" font-size="8" text-anchor="end">+4.1</text><text x="416" y="146" fill="#C93B3B" font-size="8">−5.0 → 💥 NaN</text></g></svg>
%%%

#### The whole picture — one knob, three outcomes
Now you can *feel* both failures and the sweet spot between them in one place. This is a real valley (loss `= weight²`) with a **learning rate slider**. Drag it *small* and watch the ball inch down in tiny crawling steps. Drag it *big* and watch each step overshoot the bottom and climb the far wall. Somewhere in between it slides down and settles. **Predict** what happens at each end *before* you drag — then check yourself:

%%% svg
<svg id="gd-svg" viewBox="0 0 520 214" role="img" aria-label="Interactive gradient descent. A U-shaped loss valley with a ball on the left slope. Drag the learning-rate slider: a small rate makes the ball inch down in tiny steps, a good rate settles it at the bottom, and a too-big rate makes each step overshoot and bounce higher up the far wall."><g font-family="monospace" font-size="10"><rect x="40" y="26" width="440" height="160" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><path id="gd-bowl" d="M60 40 L260 180 L460 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><polyline id="gd-traj" points="" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.8"/><circle id="gd-b0" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b1" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b2" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b3" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b4" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b5" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b6" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b7" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b8" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b9" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-ball" cx="80" cy="120" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text x="260" y="200" text-anchor="middle" fill="#9A938A" font-size="9">bottom = smallest loss</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label>learning rate <input id="gd-lr" type="range" min="0.02" max="1.2" step="0.02" value="0.3" style="width:56%;accent-color:#2A7B9B;vertical-align:middle"> <b id="gd-lrv">0.30</b></label>
<div id="gd-out" style="margin-top:6px;color:#2C2A28">lr = 0.30 → the ball rolls down and <b>settles at the bottom</b> 🎯</div>
</div>
<script>(function(){
  var lr=document.getElementById('gd-lr');if(!lr)return;
  var bowl=document.getElementById('gd-bowl'),traj=document.getElementById('gd-traj'),ball=document.getElementById('gd-ball'),
      out=document.getElementById('gd-out'),lrv=document.getElementById('gd-lrv');
  var WLO=-1.2,WHI=1.2,X0=40,X1=480,Y0=180,YT=30,LMAX=1.44,W0=-0.95,N=10;
  function px(w){return X0+(w-WLO)/(WHI-WLO)*(X1-X0);}
  function py(loss){return Y0-Math.min(loss,LMAX)/LMAX*(Y0-YT);}
  var bd='';for(var i=0;i<=60;i++){var w=WLO+(WHI-WLO)*i/60;bd+=(i?'L':'M')+px(w).toFixed(1)+' '+py(w*w).toFixed(1)+' ';}
  bowl.setAttribute('d',bd.trim());
  function paint(){
    var a=+lr.value;lrv.textContent=a.toFixed(2);
    var w=W0,pts=[];
    for(var i=0;i<=N;i++){pts.push(w);w=w-a*2*w;}
    var poly='';
    for(var j=0;j<=N;j++){var wj=Math.max(WLO,Math.min(WHI,pts[j]));var X=px(wj),Y=py(pts[j]*pts[j]);poly+=X.toFixed(1)+','+Y.toFixed(1)+' ';
      if(j<N){var d=document.getElementById('gd-b'+j);if(d){d.setAttribute('cx',X.toFixed(1));d.setAttribute('cy',Y.toFixed(1));d.setAttribute('opacity','0.55');}}}
    traj.setAttribute('points',poly.trim());
    var wf=Math.max(WLO,Math.min(WHI,pts[N]));ball.setAttribute('cx',px(wf).toFixed(1));ball.setAttribute('cy',py(pts[N]*pts[N]).toFixed(1));
    var fin=Math.abs(pts[N]),msg,col;
    if(fin>Math.abs(W0)*1.2){msg='each step <b>overshoots</b> — the ball bounces higher up the far wall and diverges 💥';col='#C93B3B';}
    else if(fin>0.4){msg='tiny steps — the ball is still <b>crawling</b>, barely down after 10 steps 🐢';col='#C99A12';}
    else if(fin<0.03){msg='the ball rolls down and <b>settles at the bottom</b> 🎯';col='#2D8B55';}
    else {msg='the ball <b>converges</b> smoothly toward the bottom ✓';col='#2D8B55';}
    ball.setAttribute('fill',col==='#C93B3B'?'#C93B3B':'#2A7B9B');
    out.innerHTML='lr = '+a.toFixed(2)+' → '+msg;
  }
  lr.addEventListener('input',paint);paint();
})();</script>
%%%

So **too big → the loss overshoots, bounces, and can blow up to NaN. Cause:** each hop is longer than the distance to the bottom. **Remedy:** lower the learning rate (and in the wildest cases, *clip* the gradient so no single hop can be enormous). You have now seen both walls — too small crawls, too big explodes. Next: how do you actually *find* the good rate in the middle?

@@@ concept id=c4 tag="Finding a good one" title="Finding a good rate — sweep and read the curve" gotit="Got the search"
So there is a crawling wall on one side and an exploding wall on the other, with a good rate somewhere between. How do you find it? You do *not* solve an equation — you *try a few and look.* Think of tuning an old radio dial to find a station. You do not know the exact number, so you sweep the dial and listen: too far one way is silence, too far the other is static, and somewhere in between the music comes in clear. Finding a learning rate is that dial-sweep. You try a handful of rates spread far apart, run a little training with each, and *watch the loss curve* — its shape tells you instantly which regime you are in.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A radio dial marked with learning-rate values 0.0001, 0.001, 0.01, 0.1, 1.0 spread across it. The low end is labelled silence (too small, crawls), the high end is labelled static (too big, blows up), and the middle mark 0.1 glows and is labelled clear signal (good rate). A caption reads sweep the dial and listen to the loss curve."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Sweep the dial — the loss curve tells you where you are</text><line x1="60" y1="96" x2="460" y2="96" stroke="#B8AEA2" stroke-width="3"/><g text-anchor="middle"><line x1="80" y1="88" x2="80" y2="104" stroke="#6B645E"/><text x="80" y="122" fill="#6B645E" font-size="8">0.0001</text><line x1="170" y1="88" x2="170" y2="104" stroke="#6B645E"/><text x="170" y="122" fill="#6B645E" font-size="8">0.001</text><line x1="260" y1="88" x2="260" y2="104" stroke="#6B645E"/><text x="260" y="122" fill="#6B645E" font-size="8">0.01</text><line x1="350" y1="86" x2="350" y2="106" stroke="#2D8B55" stroke-width="2"/><text x="350" y="122" fill="#1a5c38" font-size="8.5">0.1</text><line x1="440" y1="88" x2="440" y2="104" stroke="#6B645E"/><text x="440" y="122" fill="#6B645E" font-size="8">1.0</text></g><circle cx="350" cy="96" r="10" fill="#2D8B55" opacity="0.25"/><circle cx="350" cy="96" r="5" fill="#2D8B55"/><text x="100" y="150" text-anchor="middle" fill="#9A7208" font-size="8.5">🔇 silence: too small, crawls</text><text x="350" y="150" text-anchor="middle" fill="#1a5c38" font-size="8.5">🎵 clear: good rate</text><text x="440" y="150" text-anchor="middle" fill="#C93B3B" font-size="8.5">📡 static</text></g></svg>
%%%

**What the radio-dial picture gets right:** you *sweep* across widely-spaced settings and *listen* (watch the loss) rather than calculating the perfect number — that is exactly the real workflow. **Where it breaks down:** a radio has one true station; a network can have a whole *range* of decent learning rates, and the best one can even drift as training goes on (that is the schedule trick coming next).

#### The order-of-magnitude sweep
Because good rates can be anywhere from tiny to almost one, you do not try `0.011, 0.012, 0.013…` — that would take forever and cover almost no ground. Instead you jump by *powers of ten*: **1.0, 0.1, 0.01, 0.001, 0.0001.** Each is ten times smaller than the last, so five tries cover an enormous range. This is the practical search almost everyone starts with. Here is the winner from that sweep — the final loss after 40 steps at `lr = 0.1`. **Predict:** near the bottom (≈0), or still stuck up high? Then run it:

%%% demo id=sweep label="run it — final loss after 40 steps at lr = 0.1"
code: train(lr=0.1)      # loss = weight**2 after 40 downhill steps from weight = 2.0
out: 7.1e-08
take: <b>lr = 0.1 lands essentially at the bottom</b> — final loss ≈ 0.00000007, basically 0. Run the whole sweep and the winner jumps right out: lr = 1.0 never settles (stuck bouncing at loss 4), lr = 0.1 nails it (≈0), lr = 0.01 crawls only partway (≈0.79), and lr = 0.001 (≈3.4) and 0.0001 (≈3.9) barely budge off the starting loss of 4. No equation needed — the sweep points straight at 0.1.
%%%

Here is the whole sweep as a picture — final loss after 40 steps for each rate. See how only the middle rate reaches the floor, while both ends stay stuck high:

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="A bar chart of final loss after 40 steps for five learning rates. Learning rate 1.0 is a tall bar at loss 4 (bouncing, never settles). 0.1 is essentially zero (reaches the bottom). 0.01 is a short bar at loss 0.79 (partway). 0.001 at loss 3.4 and 0.0001 at loss 3.9 are tall (barely moved). Only the middle rate reaches the floor."><g font-family="monospace" font-size="9"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Final loss after 40 steps (lower = better; floor = 0)</text><line x1="70" y1="150" x2="470" y2="150" stroke="#B8AEA2"/><text x="40" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 40 94)">final loss</text><g><rect x="88" y="30" width="46" height="120" fill="#C93B3B"/><text x="111" y="166" text-anchor="middle" fill="#C93B3B" font-size="8">1.0</text><text x="111" y="24" text-anchor="middle" fill="#C93B3B" font-size="7.5">4.0 bounces 💥</text></g><g><rect x="164" y="147" width="46" height="3" fill="#2D8B55"/><text x="187" y="166" text-anchor="middle" fill="#1a5c38" font-size="8">0.1</text><text x="187" y="140" text-anchor="middle" fill="#1a5c38" font-size="7.5">≈0 ✓</text></g><g><rect x="240" y="126" width="46" height="24" fill="#C99A12"/><text x="263" y="166" text-anchor="middle" fill="#9A7208" font-size="8">0.01</text><text x="263" y="120" text-anchor="middle" fill="#9A7208" font-size="7.5">0.79 partway</text></g><g><rect x="316" y="48" width="46" height="102" fill="#C99A12"/><text x="339" y="166" text-anchor="middle" fill="#9A7208" font-size="8">0.001</text><text x="339" y="42" text-anchor="middle" fill="#9A7208" font-size="7.5">3.4 🐢</text></g><g><rect x="392" y="32" width="46" height="118" fill="#C99A12"/><text x="415" y="166" text-anchor="middle" fill="#9A7208" font-size="8">0.0001</text><text x="415" y="26" text-anchor="middle" fill="#9A7208" font-size="7.5">3.9 🐢</text></g></g></svg>
%%%

#### Read the curve shape to diagnose
When you plot loss-versus-step for each rate, the *shape* tells you the regime at a glance. Keep these three silhouettes in your head — they are how every practitioner eyeballs a learning rate:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three small loss-versus-step charts side by side. Left, too small: an almost-flat gently-sloping line staying high. Middle, just right: a curve that drops steeply then flattens near the bottom. Right, too large: a spiky line that jumps up and down and trends upward. Each is labelled with its diagnosis."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Three loss-curve shapes — learn them on sight</text><g><rect x="16" y="30" width="152" height="128" rx="6" fill="#FDF3D6" stroke="#C99A12"/><path d="M28 48 C 80 54, 130 62, 156 70" fill="none" stroke="#C99A12" stroke-width="2.2"/><text x="92" y="176" text-anchor="middle" fill="#9A7208" font-size="9">too small → flat 🐢</text></g><g><rect x="184" y="30" width="152" height="128" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><path d="M196 44 C 232 140, 300 150, 324 150" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text x="260" y="176" text-anchor="middle" fill="#1a5c38" font-size="9">just right → smooth drop ✓</text></g><g><rect x="352" y="30" width="152" height="128" rx="6" fill="#FDECEC" stroke="#C93B3B"/><path d="M364 120 L 388 60 L 412 130 L 436 44 L 460 118 L 484 34" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text x="428" y="176" text-anchor="middle" fill="#C93B3B" font-size="9">too large → spiky, rising 💥</text></g></g></svg>
%%%

So the recipe is: **sweep rates by powers of ten, plot loss-versus-step, and pick the one whose curve drops smoothly.** Flat means raise it; spiky or rising means lower it; a clean steep-then-settle drop means you found it. There is one more trick that beats picking any single fixed rate — and it is the reason real training does not just set one number and walk away.

@@@ concept id=c5 tag="Shrink it over time" title="Schedules — big hops early, tiny hops late" gotit="Got the schedule"
Here is a tension you may have already felt. A *big* hop is great early on — you are far from the bottom and want to cover ground fast. But a big hop is *terrible* near the end — you are almost at the bottom and a big hop just overshoots it and bounces. A *small* hop is the reverse: gentle and precise at the end, but hopelessly slow at the start. So a single fixed hop size is always a compromise. The clever fix is to *not keep it fixed*: **start big, then shrink it as you go — but not all the way to nothing; ease it down to a small, steady size.** Picture walking home in the dark. Far from the house you take big confident strides. As you get close you slow to tiny careful shuffles so you do not walk straight into the door.

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A figure walking from left to right toward a house on the right. The footsteps start as long strides on the left and get shorter and shorter approaching the door, then hold steady as tiny shuffles at the door. A label reads big strides far away and another reads tiny steady shuffles at the door."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Big strides far from home · tiny steady shuffles at the door</text><line x1="30" y1="130" x2="500" y2="130" stroke="#B8AEA2" stroke-width="2"/><rect x="452" y="86" width="44" height="44" fill="#E4D9C8" stroke="#B8AEA2"/><polygon points="452,86 474,64 496,86" fill="#C9B79C" stroke="#B8AEA2"/><rect x="468" y="104" width="12" height="26" fill="#9A7208"/><text x="474" y="150" text-anchor="middle" fill="#6B645E" font-size="8">home (bottom)</text><g fill="#2A7B9B"><ellipse cx="60" cy="130" rx="6" ry="3"/><ellipse cx="120" cy="130" rx="6" ry="3"/><ellipse cx="180" cy="130" rx="5" ry="3"/><ellipse cx="235" cy="130" rx="5" ry="3"/><ellipse cx="285" cy="130" rx="4" ry="2.5"/><ellipse cx="330" cy="130" rx="3" ry="2"/><ellipse cx="368" cy="130" rx="2.5" ry="1.8"/><ellipse cx="400" cy="130" rx="2.5" ry="1.8"/><ellipse cx="426" cy="130" rx="2.5" ry="1.8"/><ellipse cx="445" cy="130" rx="2.5" ry="1.8"/></g><text x="100" y="112" fill="#1F6280" font-size="8.5">big strides →</text><text x="405" y="112" text-anchor="middle" fill="#1F6280" font-size="8.5">tiny steady steps</text></g></svg>
%%%

**What the walking-home picture gets right:** you *want* big steps when far away and small, careful steps when close, and shrinking the step as you approach gives both — exactly what a schedule does. **Where it breaks down:** you can *see* the house, so you know when to slow down; training cannot see the bottom, so the schedule shrinks the rate on a *pre-set plan* (for example, halve it every so often, then hold a small floor) rather than by watching distance.

#### The remedy has a name — a learning-rate schedule
Deliberately shrinking the learning rate as training goes on is called a [[learning-rate schedule||a plan that changes the learning rate over training — usually big early for speed, small late to settle. Shrinking it is also called decay]] (and the shrinking part is called **decay**). Big hops early cover ground fast; small hops late let you settle precisely into the bottom instead of bouncing over it. A common, simple plan: start at `0.8`, **halve the rate** each time — `0.8 → 0.4 → 0.2 → 0.1` — but hold a small floor of `0.1` so it eases into a gentle, steady size instead of vanishing to nothing. **Predict:** after several halvings from 0.8 with a 0.1 floor, what rate are we left with? Then run it:

%%% demo id=decay label="run it — a halving schedule that floors at 0.1"
code: max(0.1, 0.8 * 0.5**5)      # start 0.8, halve each step, never below the 0.1 floor
out: 0.1
take: <b>The rate starts big at 0.8, halves toward zero, and settles at the 0.1 floor.</b> The sequence is `0.8 → 0.4 → 0.2 → 0.1 → 0.1 → 0.1 …` — big fast hops early, then it eases down and holds a small, gentle 0.1 for a precise landing. That plan is a learning-rate schedule (decay) — one of the most common tricks in real training.
%%%

Here is the schedule as a picture — each bar is the learning rate after another halving, big early and easing down to the steady `0.1` floor:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A bar chart of the learning rate over successive halvings starting at 0.8: bars at 0.8, 0.4, 0.2, then 0.1, 0.1, 0.1 held steady at a floor. A dashed line marks the 0.1 floor. Labels read start big and settle at the 0.1 floor."><g font-family="monospace" font-size="9"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Learning rate over halvings: big early → steady 0.1 floor</text><line x1="60" y1="150" x2="470" y2="150" stroke="#B8AEA2"/><text x="34" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 34 94)">learning rate</text><line x1="60" y1="135" x2="470" y2="135" stroke="#2D8B55" stroke-dasharray="3,3"/><text x="474" y="138" fill="#1a5c38" font-size="8">0.1 floor</text><g fill="#2A7B9B"><rect x="80" y="30" width="44" height="120"/><rect x="146" y="90" width="44" height="60"/><rect x="212" y="120" width="44" height="30"/><rect x="278" y="135" width="44" height="15"/><rect x="344" y="135" width="44" height="15"/><rect x="410" y="135" width="44" height="15"/></g><g text-anchor="middle" fill="#6B645E" font-size="8"><text x="102" y="164">0.8</text><text x="168" y="164">0.4</text><text x="234" y="164">0.2</text><text x="300" y="164">0.1</text><text x="366" y="164">0.1</text><text x="432" y="164">0.1</text></g><text x="102" y="24" text-anchor="middle" fill="#1F6280" font-size="8">start big</text><text x="380" y="128" text-anchor="middle" fill="#1a5c38" font-size="8">settle at floor →</text></g></svg>
%%%

Real training often adds one more flourish at the very start: instead of jumping straight to the big rate on step one, it ramps *up* to it over the first few steps — a **warmup**, so training does not lurch on the first update — then decays. Here is that full shape, warmup up then decay down:

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A learning-rate-versus-step curve. It rises briefly from zero at the start, labelled warmup, reaches a peak, then curves smoothly downward and levels off at a small value, labelled decay. The x-axis is steps, the y-axis is learning rate."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A common schedule: brief warmup up, then decay down</text><line x1="50" y1="30" x2="50" y2="148" stroke="#B8AEA2"/><line x1="50" y1="148" x2="500" y2="148" stroke="#B8AEA2"/><text x="24" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 24 94)">learning rate</text><text x="275" y="170" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><path d="M50 148 L 120 46 C 240 46, 340 118, 496 122" fill="none" stroke="#2A7B9B" stroke-width="2.6"/><line x1="120" y1="46" x2="120" y2="148" stroke="#E5DFD6" stroke-dasharray="3,3"/><text x="86" y="40" fill="#1F6280" font-size="8.5">warmup ↑</text><text x="330" y="84" fill="#1F6280" font-size="8.5">decay ↓ (settle gently)</text></g></svg>
%%%

So the schedule solves the fast-early-versus-precise-late tension: **start big for speed, shrink it (decay) for a gentle landing.** (There is a whole zoo of schedule shapes — step, exponential, cosine — plus warmup and adaptive optimizers like Adam that adjust the rate for you; those are for later lessons. Today you just need the core idea: the best hop size is not one number, it is a *plan*.) Let us gather the whole day onto one page.

@@@ concept id=c6 tag="Recap" title="The whole day on one page" gotit="Got the recap"
Here is a fresh way to hold the whole day in your head. Think of the **hot-water knob** in a shower. Turn it too far toward cold and the water is freezing — nothing happens fast, you just stand there shivering (that is a rate too *small*, the crawl). Turn it too far toward hot and you leap out scalded (that is a rate too *big*, the overshoot). You find the comfortable middle by nudging the knob a little each way and *feeling* the result — that is the sweep. And as the shower runs, you keep easing the knob to hold it just right — that is the schedule. One knob, felt by its result, adjusted over time. **Where the shower knob breaks down:** the shower has one comfy setting that barely changes; a network has a whole *range* of decent rates, and the best one really does drift as training goes, which is why decay matters more than any single perfect number.

%%% svg
<svg viewBox="0 0 520 178" role="img" aria-label="A shower temperature dial. The far-left cold zone is labelled too small, freezing, crawls. The far-right hot zone is labelled too big, scalding, overshoots. The middle green zone is labelled just right, comfortable. A hand nudges the dial toward the middle. A caption reads one knob, felt by its result."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One knob, felt by its result — the shower dial</text><path d="M110 140 A 150 150 0 0 1 410 140" fill="none" stroke="#B8AEA2" stroke-width="14"/><path d="M110 140 A 150 150 0 0 1 194 44" fill="none" stroke="#3A7C93" stroke-width="14"/><path d="M326 44 A 150 150 0 0 1 410 140" fill="none" stroke="#C93B3B" stroke-width="14"/><path d="M212 34 A 150 150 0 0 1 308 34" fill="none" stroke="#2D8B55" stroke-width="14"/><line x1="260" y1="140" x2="260" y2="52" stroke="#2C2A28" stroke-width="3"/><circle cx="260" cy="140" r="7" fill="#2C2A28"/><text x="120" y="164" text-anchor="middle" fill="#3A7C93" font-size="8.5">🥶 too small (crawl)</text><text x="260" y="30" text-anchor="middle" fill="#1a5c38" font-size="8.5">✓ just right</text><text x="404" y="164" text-anchor="middle" fill="#C93B3B" font-size="8.5">🔥 too big (overshoot)</text></g></svg>
%%%

Now the whole arc in a few beats — the learning rate is the size of each downhill hop across the stream on its **stepping stones**:

- **What it is.** In *new weight = old weight − learning rate × gradient*, the gradient picks the *direction* (downhill) and the learning rate picks *how far you hop*. It is the one number scaling the gradient in the same loop that trains your neuron's weights and bias.
- **Too small → crawls.** Each hop is a grain of sand; the loss barely drops and the curve looks flat. **Remedy:** raise the learning rate.
- **Too big → overshoots, bounces, blows up.** Hops longer than the valley jump past the bottom and grow each step, ending in `NaN`. **Remedy:** lower the learning rate (or clip).
- **Find a good one by looking.** Sweep rates by powers of ten (1.0, 0.1, 0.01, 0.001), plot loss-versus-step, and pick the curve that drops smoothly. Flat → raise it; spiky/rising → lower it.
- **Beat a fixed rate with a schedule.** Start big for fast early progress, then *decay* it toward a small steady rate for a gentle, precise landing near the bottom.

One picture to keep — the same knob, three outcomes:

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Three loss-versus-step curves overlaid. A gold flat line labelled too small, a green steep-then-flat curve labelled just right, and a red spiky rising line labelled too large. A caption reads one knob, three outcomes."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One knob — three outcomes</text><line x1="48" y1="28" x2="48" y2="146" stroke="#B8AEA2"/><line x1="48" y1="146" x2="500" y2="146" stroke="#B8AEA2"/><text x="22" y="90" fill="#6B645E" font-size="9" transform="rotate(-90 22 90)">loss</text><text x="274" y="168" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><path d="M54 44 C 200 56, 360 68, 496 78" fill="none" stroke="#C99A12" stroke-width="2.2"/><text x="360" y="62" fill="#9A7208" font-size="8.5">too small 🐢</text><path d="M54 44 C 130 128, 300 142, 496 143" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text x="300" y="128" fill="#2D8B55" font-size="8.5">just right ✓</text><path d="M54 90 L 96 42 L 138 96 L 180 34 L 222 100 L 264 30" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text x="150" y="28" fill="#C93B3B" font-size="8.5">too large 💥</text></g></svg>
%%%

The quick diagnosis table — read the curve, know the fix:

%%% table
:: Learning rate :: What the loss curve does :: The fix
too small :: barely drops — looks flat 🐢 :: raise the rate
just right :: drops fast, then settles ✓ :: leave it (or add a schedule)
too large :: spiky, bounces, rises to NaN 💥 :: lower the rate (or clip)
%%%

And the cheat-sheet — every word from today in one place:

%%% jargon
learning rate (lr) | the size of each downhill hop: how far a weight moves per step
gradient | the slope of the loss — which way is uphill and how steep; we step the opposite way
update rule | new weight = old weight − learning rate × gradient
diverge | the loss grows instead of shrinking, from a too-big rate — ends in NaN
NaN | "not a number": what the computer prints when a value blew up past any real number
order-of-magnitude sweep | trying rates spaced by powers of ten (1.0, 0.1, 0.01, 0.001) to find a good one
loss curve | loss plotted against step — its shape tells you if the rate is too small, right, or too big
learning-rate schedule (decay) | a plan that shrinks the rate over training: big hops early, tiny steady hops late
%%%

@@@ quiz id=quiz tag="Quiz" title="Four quick checks" gotit="answer all first"
%%% quiz
q: In "new weight = old weight − learning rate × gradient", what does the learning rate control? | a:1 | which way is downhill | how far you hop each step | the number of weights | the value of the loss | fb: The learning rate is the step size — how far the weight moves each hop. The gradient picks the direction.
q: Your loss barely drops — the curve looks almost flat. What is likely wrong, and the fix? | a:2 | rate too big → lower it | model is broken → restart | rate too small → raise it | you are done → stop | fb: A near-flat loss means the hops are tiny — the rate is too small. Raise it.
q: Your loss bounces around and shoots up to NaN. What happened? | a:0 | the rate is too big — steps overshoot and diverge | the rate is too small — it crawls | the gradient is zero | the schedule finished | fb: Too-big steps jump past the bottom and grow each update until the numbers blow up to NaN. Lower the rate.
q: What does a learning-rate schedule (decay) do? | a:3 | uses more data per step | changes the loss function | fixes a bad model | starts with a big rate and shrinks it over time | fb: Decay: big hops early for fast progress, tiny hops late for a gentle, precise landing near the bottom.
%%%

@@@ produce id=produce tag="Produce" title="Feel the three regimes with your own sweep" gotit="Done"
Time to *feel* the whole spine in one run: crawl, settle, explode — all from turning one knob. You will train the same tiny model at several learning rates and watch each loss curve tell its story.

The setup is the simplest possible valley: loss `= weight**2` (so the bottom is at `weight = 0`), the gradient is `2*weight`, and every run starts at `weight = 2.0` for `40` steps. The *only* thing you change between runs is the learning rate. Sweep it by powers of ten: `1.0, 0.1, 0.01, 0.001`.

**Predict first, before you run it:** for each of the four rates, guess whether the loss will (a) barely move (crawl), (b) drop smoothly to near zero (just right), or (c) bounce and blow up (diverge). Write your four guesses down.

**What you should see** when you run it:
- **lr = 1.0** → the weight flips sign and the loss **bounces/blows up** — too big (final loss stuck at 4).
- **lr = 0.1** → the loss **drops smoothly to almost zero** (about 0.00000007) — just right.
- **lr = 0.01** → the loss drops a lot but **only partway** — down to about 0.79, still short of the bottom; starting to crawl.
- **lr = 0.001** → the loss **barely moves** (about 3.4, only a little below its start of 4.0) — far too small, a near-flat curve.
- Check the order matches your prediction: as the rate shrinks by tens, you walk from *explode* → *just right* → *partway crawl* → *near-frozen*. That is the whole knob in four lines.

Write it in `experiment.py` and run it. Then poke it: nudge `1.0` up to `1.1` and watch it reach NaN within a few steps; add a tiny schedule that *halves* the rate (starting from `0.8`) every few steps, holding a `0.1` floor, and watch a big starting rate settle at 0.1 instead of bouncing.

**Option A — write it yourself.** Fill in `experiment.py` with a `train(lr)` loop that steps `weight -= lr * (2*weight)` for 40 steps and prints the final loss for each rate in `[1.0, 0.1, 0.01, 0.001]`.

**Option B — hand it to the lab.** Paste this into the experiment lab and let it build `experiment.py` for you:

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my artifact.
Create experiment.py for Day 8 (learning-rate intuition). Use the toy valley
loss = weight**2 with gradient = 2*weight, starting at weight = 2.0, for 40 steps.
Write train(lr) that runs the update weight = weight - lr * (2*weight) for 40 steps
and returns the final loss (weight**2). Sweep lr over [1.0, 0.1, 0.01, 0.001] and
print, for each, the final loss AND print the per-step loss for lr=1.0 and lr=0.1 so
the too-big divergence and the just-right smooth drop are both visible.
Expected: lr=1.0 stays stuck at loss 4 (bounces), lr=0.1 drops to ~0, lr=0.01 crawls
to ~0.79 partway, lr=0.001 barely moves to ~3.4. Then add a bonus run with a halving
schedule that starts lr at 0.8 and does lr = max(0.1, lr*0.5) each step (a 0.1 floor),
and show the rate settles at 0.1 and the loss lands gently instead of bouncing.
%%%

@@@ fin
