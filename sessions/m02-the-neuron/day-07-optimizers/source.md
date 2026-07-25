---
quest_id: wf3-d04-optimizers
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 7 — Optimizers: SGD → Momentum → Adam"
module_label: "Module 2 · Train · Day 7"
title: "Optimizers: SGD → Momentum → Adam"
subtitle: "Smarter Ways to Step Downhill"
brand_sub: "Foundations · M2 Day 7"
spine: "rolling ball"
nav_prev_href: "../day-06-training-loop/lesson.html"
nav_prev_label: "The Training Loop"
nav_next_href: "../day-08-learning-rate/lesson.html"
nav_next_label: "Learning-Rate Intuition"
fin_title: "Module 2 · Day 7 complete! 🏆"
fin_body: "Nice work — you've met the <b>optimizers</b>: plain SGD takes the raw downhill step, Momentum builds speed like a ball rolling downhill, and Adam adapts the step size for every weight. Same gradient, smarter steps.<br>Next up: <b>Learning Rate</b> — the single knob that sets how big every step is."
notebook_yardstick: null
coverage_topics:
  - {topic: learning rate scalar, keywords: [learning rate, step size, how big a step, scales the gradient]}
  - {topic: vanilla update rule, keywords: [new weight = old weight, w = w - lr, minus the gradient, plain step downhill]}
  - {topic: batch gradient descent, keywords: [whole dataset, full batch, all the data per step, stable but slow]}
  - {topic: stochastic gradient descent, keywords: [one example at a time, stochastic, sgd, fast and noisy, single example]}
  - {topic: mini-batch gradient descent, keywords: [mini-batch, small batch, handful of examples, middle ground]}
  - {topic: sgd motivation, keywords: [too slow to use the whole dataset, sample, why we moved from full-batch, waiting]}
  - {topic: momentum, keywords: [momentum, running velocity, build up speed, past gradients, rolling ball]}
  - {topic: adaptive per-parameter rates, keywords: [per-parameter, its own step size, each weight its own, adaptive]}
  - {topic: rmsprop, keywords: [rmsprop, squared gradients, running average of, scale each weight]}
  - {topic: adam, keywords: [adam, momentum plus, per-parameter scaling, default optimizer]}
  - {topic: lr too large failure, keywords: [too large, overshoot, oscillates, diverges, blows up]}
  - {topic: lr schedule decay remedy, keywords: [learning-rate schedule, decay, shrink the learning rate, smaller steps later]}
  - {topic: lr too small failure, keywords: [too small, barely moves, painfully slow, crawls]}
  - {topic: lr too small remedy, keywords: [larger learning rate, use momentum, adaptive optimizer, raise the rate]}
  - {topic: sgd zig-zag failure, keywords: [zig-zag, narrow valley, side to side, noisy, ravine]}
  - {topic: zig-zag remedy momentum, keywords: [momentum smooths, damps the zig-zag, cancels the side-to-side]}
  - {topic: ill-scaled features failure, keywords: [ill-scaled, sparse, one global learning rate, cannot suit every weight]}
  - {topic: ill-scaled remedy adaptive, keywords: [adaptive per-parameter rates, rmsprop, adam, its own step size]}
  - {topic: capability limit follows gradient, keywords: [only follows the gradient, cannot fix a bad loss, non-separable, global minimum, non-convex]}
---

@@@ hero
@lede Put a marble on a tilted tray and let go. You do not have to tell it anything — it just **rolls** the way the tray tips, quick where the tilt is steep, slow where it flattens, and it can even coast over a small dent because it picked up speed. Here is the surprising bit: training a neural network is *almost exactly* that. The chatbot you ask questions to, the face-unlock on a phone, the AI that plays a video game — every one of them learned by letting a **rolling ball** find the bottom of a hill made of "how wrong am I." The rule that decides *how* that ball rolls — plainly, or with built-up speed, or cleverly for every single weight — is called an **optimizer**. It is one of the big reasons those models train in hours instead of months. Yesterday you built the loop that takes one downhill step. Today you upgrade that step three times.
@goal Together we will make the "nudge" part of the loop smarter, three times over. You will start with plain **gradient descent** (the raw downhill step) and its three flavors — using **all** the data, **one** example, or a small **batch** per step. Then you add **Momentum**, a running speed that smooths a jittery path just like a real marble. Then **RMSProp** and **Adam**, which hand *every* weight its *own* step size. Along the way you will poke the ways stepping goes wrong — too big, too small, zig-zagging — each one a small puzzle with a neat fix. Every formula arrives in plain words first, and heavy math sits in a skippable box you may ignore.

%%% warmup
q: From yesterday: what are the four steps of the training loop, in order? | a:2 | loss → forward → update → backward | update → backward → loss → forward | forward → loss → backward → update | backward → forward → update → loss | concept: training-loop | fb: One lap is forward (take the shot) → loss (measure the miss) → backward (which way off?) → update (nudge the aim), then repeat.
q: From yesterday: in "w ← w − lr × gradient", what does the minus sign do? | a:1 | it deletes bad weights | the gradient points uphill, so minus steps the weight downhill toward less loss | it just marks the notation, no effect | it counts the epochs | concept: update-rule | fb: The gradient points uphill toward more loss, so subtracting it walks the weight downhill.
q: From yesterday: what does the learning rate control, and what if it is too big? | a:0 | the step size each loop — too big overshoots and the loss can blow up | the number of layers — too big adds neurons | which way is downhill — too big flips the sign | the loss value itself — too big means zero loss | concept: learning-rate | fb: The learning rate is the step size; too big and each nudge overshoots the bottom, so the loss oscillates and diverges.
%%%

@@@ concept id=c1 tag="The plain step" title="Gradient descent — the ball's raw downhill step" gotit="Got the plain step"
Let us start with the plainest way to roll the marble — the one already sitting inside yesterday's loop. Crouch down next to the ball and look at the ground right under it. Which way does it tip? *That* is the way the ball wants to go: downhill, toward less "how wrong am I." The plainest optimizer does that and nothing else. Feel the tilt under the ball, slide the ball a little way downhill, repeat. How far it slides each time comes from one number — the [[learning rate||the step size: how far the ball moves each step. Written "lr". A bigger lr takes bigger slides, a smaller lr takes smaller slides]] — which simply scales the slope before it moves the weight. This roll-with-the-tilt move is called **gradient descent**.

%%% svg
<svg viewBox="0 0 520 202" role="img" aria-label="A marble resting on a tilted tray drawn as a U-shaped valley. An arrow under the marble follows the ground tipping downhill to the right. A ghosted marble sits further along the same curve, showing where one slide lands. A label reads step size equals learning rate times tilt."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Feel the tilt under the marble → slide it one step downhill</text><path d="M40 40 Q 160 190 260 176 Q 360 190 480 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text x="300" y="194" text-anchor="middle" fill="#6B645E" font-size="8.5">bottom = smallest loss</text><circle cx="118" cy="122" r="9" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text x="112" y="104" text-anchor="middle" fill="#1F6280" font-size="9">the marble (current weight)</text><path d="M130 132 l46 26" stroke="#C93B3B" stroke-width="2"/><polygon points="178,159 166,157 170,148" fill="#C93B3B"/><text x="210" y="146" fill="#C93B3B" font-size="9">the tray tips this way →</text><circle cx="186" cy="165" r="8" fill="#2A7B9B" opacity="0.4" stroke="#1F6280" stroke-dasharray="2,2"/><text x="212" y="180" fill="#2D8B55" font-size="9">↑ one slide lands here</text><text x="260" y="42" text-anchor="middle" fill="#9A7208" font-size="9">step size = learning rate × tilt</text></g></svg>
%%%

**What the marble-on-a-tray picture gets right:** the ball moves *downhill* all by itself, and a steeper tilt gives a bigger push. That is exactly the step.

**Where it breaks down:** a real marble has weight and keeps rolling on its own. Plain gradient descent has *no memory at all* — each step feels only the tilt right where it stands and forgets the last one instantly. (Giving it memory is the very next upgrade.)

#### The whole move, one rung at a time
The rule is one short sentence in plain words: **new weight = old weight − learning rate × slope.** Let us walk it slowly, one small move per rung, so nothing sneaks past you.

%%% steps
step: feel the tilt
why: the slope (the *gradient*) under the ball tells you which way the ground rises
step: turn around — that's the minus sign
why: the slope points *uphill* toward MORE loss, therefore subtracting it walks you downhill
step: decide how far — multiply by the learning rate
why: this is the step size; it scales the gradient into an actual distance to travel
step: new weight = old weight − learning rate × slope
why: that's the entire optimizer, and every clever one today is this line plus one twist
%%%

So far: three tiny decisions — which way, turn around, how far — and you have a working optimizer. Here it is as one line you can point at:

%%% formula
expr: w ← w − lr × (slope of the loss at w)
note: "w" is a weight, "lr" is the learning rate (the step size). The MINUS sign walks downhill — opposite the uphill slope. Same rule you built in yesterday's loop; today we make it smarter.
%%%

%%% insight
Notice which knob is doing the scary work here: not the slope (the maths hands you that for free) but the **learning rate**. Pick it well and the marble glides to the bottom. Pick it badly and the same perfectly-correct gradient will throw your model off a cliff. That's why one number gets a whole day of its own tomorrow.
%%%

Now *you* drive the marble. Below is a real valley with a **learning rate** slider. Before you touch it, guess: what happens at the tiny end, and what happens when you push it all the way up? Then drag it and find out — small first, then big.

%%% svg
<svg id="gd-svg" viewBox="0 0 520 214" role="img" aria-label="Interactive gradient descent. A U-shaped loss valley with a ball on the left slope. Drag the learning-rate slider: a small rate makes the ball inch down in tiny steps, a good rate settles it at the bottom, a rate above one half makes it hop across the bottom to the other side each step, and a rate above one makes each step overshoot and climb the far wall."><g font-family="monospace" font-size="10"><rect x="40" y="26" width="440" height="160" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><path id="gd-bowl" d="M60 40 L260 180 L460 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><polyline id="gd-traj" points="" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.8"/><circle id="gd-b0" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b1" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b2" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b3" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b4" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b5" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b6" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b7" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b8" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b9" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-ball" cx="80" cy="120" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text x="260" y="200" text-anchor="middle" fill="#9A938A" font-size="9">bottom = smallest loss</text></g></svg>
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
  // static bowl (loss = w^2)
  var bd='';for(var i=0;i<=60;i++){var w=WLO+(WHI-WLO)*i/60;bd+=(i?'L':'M')+px(w).toFixed(1)+' '+py(w*w).toFixed(1)+' ';}
  bowl.setAttribute('d',bd.trim());
  function paint(){
    var a=+lr.value;lrv.textContent=a.toFixed(2);
    var w=W0,pts=[];
    for(var i=0;i<=N;i++){pts.push(w);w=w-a*2*w;} // step: w -= lr*d(w^2)/dw
    // trajectory polyline + step dots
    var poly='';
    for(var j=0;j<=N;j++){var wj=Math.max(WLO,Math.min(WHI,pts[j]));var X=px(wj),Y=py(pts[j]*pts[j]);poly+=X.toFixed(1)+','+Y.toFixed(1)+' ';
      if(j<N){var d=document.getElementById('gd-b'+j);if(d){d.setAttribute('cx',X.toFixed(1));d.setAttribute('cy',Y.toFixed(1));d.setAttribute('opacity','0.55');}}}
    traj.setAttribute('points',poly.trim());
    var wf=Math.max(WLO,Math.min(WHI,pts[N]));ball.setAttribute('cx',px(wf).toFixed(1));ball.setAttribute('cy',py(pts[N]*pts[N]).toFixed(1));
    // each step multiplies w by (1 - 2*lr), so |1-2*lr| decides the story — NOT |w| alone
    var ratio=Math.abs(1-2*a),fin=Math.abs(pts[N]),msg,col;
    if(ratio>1.001){msg='each step <b>overshoots</b> — the ball lands higher up the far wall and <b>diverges</b> 💥';col='#C93B3B';}
    else if(ratio>0.999){msg='the ball <b>bounces between the same two points forever</b> — it never settles and never gets lower 🔁';col='#C93B3B';}
    else if(a>0.5){msg='it <b>hops over the bottom</b> to the other side every step — still shrinking, but wastefully ↔️';col='#C99A12';}
    else if(fin>0.4){msg='tiny steps — the ball is still <b>crawling</b>, barely down after 10 steps 🐢';col='#C99A12';}
    else if(fin<0.03){msg='the ball rolls down and <b>settles at the bottom</b> 🎯';col='#2D8B55';}
    else {msg='the ball <b>converges</b> smoothly toward the bottom ✓';col='#2D8B55';}
    ball.setAttribute('fill',col==='#C93B3B'?'#C93B3B':'#2A7B9B');
    out.innerHTML='lr = '+a.toFixed(2)+' → '+msg;
  }
  lr.addEventListener('input',paint);paint();
})();</script>
%%%

Did the big end surprise you? Most people expect "bigger step = faster." Instead the marble leaps clean over the bottom and climbs the far wall. Notice the in-between zone too, just past the middle of the slider: the ball keeps landing on the *opposite* side each step — still getting lower, but wasting half its motion. Hold onto that feeling; we come back to it near the end of the day.

That plain step — feel the tilt, slide downhill — is the foundation, and you now own it. Everything fancier today is *this move plus one twist*. But first, one honest question we skipped: the tilt of *what*? All your data? One example? That choice has three famous flavors.

@@@ concept id=c2 tag="How much data per step" title="Batch, stochastic, mini-batch — the three flavors" gotit="Got the three flavors"
Before we make the ball smarter, that honest question: the tilt of *what*? Picture a **class deciding which way to walk** on a school trip. You could ask *every single* student before you take one step — very accurate, but slow, and painfully slow if the class has a million kids. Or you could ask *one* random student and walk immediately — super quick, though that one voice may point you slightly wrong. Or the sensible middle: ask a small *handful* and go with what most of them say. Those three choices have names, and picking one is the first real decision any optimizer makes.

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="Three ways a class decides which way to walk. Left: ask the whole class, all student icons highlighted, labelled batch, accurate but slow. Middle: ask one random student, one icon highlighted, labelled stochastic, fast but noisy. Right: ask a small handful, a few icons highlighted, labelled mini-batch, the practical middle."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">How many classmates do you ask before each step?</text><rect x="14" y="30" width="158" height="150" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="93" y="50" text-anchor="middle" fill="#C93B3B">ask the WHOLE class</text><g fill="#C93B3B"><circle cx="46" cy="78" r="7"/><circle cx="72" cy="78" r="7"/><circle cx="98" cy="78" r="7"/><circle cx="124" cy="78" r="7"/><circle cx="59" cy="102" r="7"/><circle cx="85" cy="102" r="7"/><circle cx="111" cy="102" r="7"/><circle cx="46" cy="126" r="7"/><circle cx="72" cy="126" r="7"/><circle cx="98" cy="126" r="7"/><circle cx="124" cy="126" r="7"/></g><text x="93" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">batch: accurate, but slow</text><rect x="182" y="30" width="156" height="150" rx="6" fill="#FDF3D6" stroke="#C99A12"/><text x="260" y="50" text-anchor="middle" fill="#9A7208">ask ONE student</text><g fill="#D8CFBE"><circle cx="216" cy="78" r="7"/><circle cx="242" cy="78" r="7"/><circle cx="294" cy="78" r="7"/><circle cx="229" cy="102" r="7"/><circle cx="281" cy="102" r="7"/><circle cx="216" cy="126" r="7"/><circle cx="294" cy="126" r="7"/></g><circle cx="268" cy="78" r="9" fill="#C99A12"/><text x="260" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">stochastic: fast, but noisy</text><rect x="348" y="30" width="158" height="150" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="427" y="50" text-anchor="middle" fill="#1a5c38">ask a HANDFUL</text><g fill="#C7DDCB"><circle cx="382" cy="78" r="7"/><circle cx="460" cy="78" r="7"/><circle cx="395" cy="102" r="7"/><circle cx="382" cy="126" r="7"/><circle cx="460" cy="126" r="7"/></g><g fill="#2D8B55"><circle cx="408" cy="78" r="7"/><circle cx="434" cy="78" r="7"/><circle cx="447" cy="102" r="7"/></g><text x="427" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">mini-batch: the sweet spot</text></g></svg>
%%%

**What the class-vote picture gets right:** more voices means a truer direction but a slower decision, and one voice is quick but jumpy — that trade is real. **Where it breaks down:** in a real class everyone can shout at once; a network's "asking" means doing maths on each example, so asking more genuinely costs more time and memory. That cost is exactly why the middle option won.

#### The three flavors, in network words
Each flavor is only a choice of how many examples you measure the tilt on before you take one step. Same class, three sizes of vote:

%%% steps
step: Batch gradient descent — ask the whole class
why: measure the slope over the [[whole dataset||every training example is used to compute one averaged slope before a single step is taken]], then take ONE step; the direction is stable but slow, because you touch all the data per step just to move once
step: Stochastic gradient descent (SGD) — ask one student
why: measure the slope on a [[single example||one randomly chosen training example gives the slope for this step — quick to compute, but jumpy]] and step right away, so it is fast and noisy; one example at a time points a bit off, and the path jitters
step: Mini-batch gradient descent — ask a handful
why: measure the slope on a small batch of examples (often 32 or 64), then step; that is the middle ground — fast like SGD, and the little average smooths most of the jitter
step: and the winner is… mini-batch
why: therefore almost everyone uses it; it keeps the many-steps speed while the handful of examples keeps each step from wandering
%%%

So far: three sizes of vote, one clear favorite. But *why* did the world walk away from the accurate option?

%%% insight
This is where beginners get suspicious, and rightly so: we just chose the *less accurate* answer on purpose. That is the surprising heart of modern training — a rough direction taken NOW beats a perfect direction taken after a very long wait. Speed of stepping matters more than purity of stepping.
%%%

#### Why we left full-batch behind
Early on, people used full batch: measure on *all* the data, then move once. On a small dataset that is perfectly fine.

But real datasets hold millions of examples. Which means it is too slow to use the whole dataset before every single step — you crawl, waiting and waiting for one move.

The fix was almost cheeky: **do not wait — sample.** Look at a few examples, step *now*, repeat. Guess how lopsided that trade is before you reveal it:

%%% demo id=flavors label="reveal — steps taken in the same time budget"
predict: with the SAME budget of "examples touched", how many steps does full-batch buy you compared to mini-batch? Twice as many? Ten times? More?
code: n = 1_000_000  # dataset size; cost ≈ examples touched per step
out: batch      :         1 step  per full pass over the data
     mini-batch :    15,625 steps per full pass over the data
     sgd        : 1,000,000 steps per full pass over the data
take: <b>Same data budget, wildly different number of steps.</b> Full-batch buys ONE step per pass; mini-batch (64 at a time) buys ~15,625; pure SGD buys a million. That is why we sample — many rough steps almost always learn faster than one perfect step. Mini-batch is the sweet spot: plenty of steps, and the small average keeps each one fairly steady.
%%%

If the size of that gap startled you, good — it startles most people, and it is the single fact that explains why nobody trains on the whole dataset per step any more.

So the modern default is *mini-batch* SGD: sample a small batch, take the plain downhill step, repeat. You now know **what** tilt you are stepping on. Next we make the *step itself* smarter, by giving our marble some real rolling speed.

@@@ concept id=c3 tag="Give it speed" title="Momentum — a ball that builds up speed" gotit="Got Momentum"
Now the first real upgrade, and it is the most fun one. Our plain marble has *no memory*: it forgets each step the instant it takes it. But a real **ball rolling** down a long slope does not forget — it builds up **speed**. It goes faster and faster down a steady hill. It coasts straight over a small dent because it carried speed into it. And if the ground tugs it left, then right, then left, those side tugs partly *cancel* and the ball mostly keeps going the way it was already going. **Momentum** the optimizer copies that exactly: instead of stepping by the raw tilt each time, it keeps a running **velocity** — a memory of recent tilts — and steps by the velocity.

%%% svg
<svg viewBox="0 0 520 188" role="img" aria-label="A ball rolling down a long slope, drawn three times with speed streaks growing longer as it descends, showing it building up speed. It coasts over a small bump partway down without stopping. Labels read slow start, coasts over it, and fast because it built up speed."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A real rolling ball builds speed — and coasts over little bumps</text><path d="M30 44 C 140 70, 210 118, 280 128 Q 300 116 320 128 C 400 138, 460 150, 500 158" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="70" cy="52" r="6" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="58" y1="52" x2="70" y2="52" stroke="#2A7B9B" stroke-width="1.4"/><circle cx="180" cy="104" r="7" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="158" y1="102" x2="180" y2="104" stroke="#2A7B9B" stroke-width="1.6"/><circle cx="410" cy="141" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="372" y1="137" x2="410" y2="141" stroke="#2A7B9B" stroke-width="1.8"/><text x="120" y="46" fill="#1F6280" font-size="8.5">slow start</text><text x="440" y="132" fill="#1F6280" font-size="8.5">fast — built up speed</text><text x="300" y="106" text-anchor="middle" fill="#9A7208" font-size="8.5">bump</text><text x="300" y="152" text-anchor="middle" fill="#2D8B55" font-size="8.5">coasts over it ✓</text></g></svg>
%%%

**What the rolling-ball-with-speed picture gets right:** speed really does build along a steady downhill, and it really can carry the ball over a small bump. Both are true of the optimizer.

**Where it breaks down:** a real ball can pick up *too much* speed and shoot past the valley. The optimizer's velocity fades a little each step on purpose — we "leak" some speed — so it does not run away forever like a frictionless ball would.

#### The puzzle Momentum was invented to solve
Many loss valleys are not round bowls. They are long, narrow **ravines** — steep across the skinny direction, barely tilted along the long one. Here is the trap our plain ball walks into:

%%% steps
step: the shape of the trap
why: the slope is huge ACROSS the narrow direction and tiny ALONG the long one, so the two directions pull with wildly different force
step: the crawl
why: therefore each step barely creeps forward along the gentle direction — the ball hardly makes progress toward the bottom
step: the zig-zag
why: meanwhile the big across-slope flings it wall to wall, so it bounces side to side and wastes almost all its motion (this is the noisy zig-zag failure of plain SGD in a narrow valley)
%%%

Here is the cost of that, drawn — the same ravine walked two ways, before and after:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Before and after of a path down a narrow valley. Left, plain SGD: the path zig-zags sharply from one wall to the other while creeping slowly toward the bottom. Right, with momentum: the side-to-side bounces cancel and the path glides much more smoothly down toward the bottom."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Narrow valley: plain SGD zig-zags · Momentum glides through</text><text x="130" y="34" text-anchor="middle" fill="#C93B3B" font-size="10">plain SGD</text><path d="M60 44 Q 130 120 200 44" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><path d="M60 60 Q 130 130 200 60" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><polyline points="80,52 182,66 90,82 176,96 100,110 168,122 118,134 150,142" fill="none" stroke="#C93B3B" stroke-width="2"/><circle cx="150" cy="142" r="4" fill="#C93B3B"/><text x="130" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5">bounces wall to wall,</text><text x="130" y="188" text-anchor="middle" fill="#6B645E" font-size="8.5">creeps down slowly</text><line x1="260" y1="30" x2="260" y2="160" stroke="#E5DFD6" stroke-dasharray="4,3"/><text x="390" y="34" text-anchor="middle" fill="#2D8B55" font-size="10">with Momentum</text><path d="M320 44 Q 390 120 460 44" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><path d="M320 60 Q 390 130 460 60" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><polyline points="360,52 392,74 388,96 391,116 390,140" fill="none" stroke="#2D8B55" stroke-width="2"/><circle cx="390" cy="140" r="4" fill="#2D8B55"/><text x="390" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5">side tugs cancel →</text><text x="390" y="188" text-anchor="middle" fill="#6B645E" font-size="8.5">glides down, much smoother</text></g></svg>
%%%

Look at the right-hand path: that is the *whole* repair, and it comes from two things happening at once.

%%% steps
step: the fix — add up the agreements
why: along the gentle direction every push points the SAME way, which means momentum lets them add up into real speed
step: the fix — cancel the disagreements
why: across the narrow direction the pushes point OPPOSITE on each bounce, so the running velocity cancels the side-to-side motion — momentum smooths and damps the zig-zag
%%%

%%% insight
Here is why this bites in real life: a zig-zagging run does not *look* broken. The loss goes down, slowly, and nothing crashes. You could burn hours thinking "my model is just hard to train" when the real story is a ball bouncing off two walls. Momentum is often the one-line change that turns those hours into minutes — and that is not a toy claim. The bot that learned to play chess, the model that picks the next song for you: they got trained overnight instead of over a month largely because their ball rolled with memory.
%%%

So far: same ball, same hill, and only *memory* added — yet the path is a different animal.

#### A second thing that memory fixes
That same running velocity cures the opposite complaint: the learning rate set **too small**, where each step is a grain of sand and the loss barely moves, painfully slow.

**Cause:** every single step is far too short for the distance to the bottom, so the ball crawls.

**Remedy:** momentum lets many tiny same-direction steps *add up* into real speed. So **use momentum** — or simply raise the rate to a larger learning rate, or reach for an adaptive optimizer — whenever your run creeps.

#### The rule, in plain words
Momentum is two short lines. First update the velocity: **new velocity = a fraction of the old velocity + this step's tilt.** Then step by the velocity. That fraction is how much past speed you keep — commonly about 0.9, so 90% of yesterday's speed carries over. Take a guess before you reveal the numbers:

%%% demo id=momentum label="reveal — velocity building up (steady slope = 1)"
predict: the slope stays a steady 1.0 every step. Plain SGD would step 1.0 each time — after 5 steps, how big do you think Momentum's step has grown? 1.5? 2? more?
code: v = 0.0; keep = 0.9   # each step: v = keep*v + slope, with slope = 1.0
out: step 1: velocity = 1.000   (plain SGD would step 1.000)
     step 2: velocity = 1.900   (plain SGD would step 1.000)
     step 3: velocity = 2.710   (plain SGD would step 1.000)
     step 4: velocity = 3.439   (plain SGD would step 1.000)
     step 5: velocity = 4.095   (plain SGD would step 1.000)
take: <b>Steady downhill → velocity grows 1.0 → 1.9 → 2.71 → 3.44 → 4.10.</b> Plain SGD keeps taking the same 1.0 step forever; Momentum accelerates along a consistent slope, so it covers ground faster and faster. Now flip the slope's sign every step instead: the terms mostly cancel and the velocity stays near zero — that is exactly how it kills the zig-zag. (Same <b>v = keep·v + slope</b> rule you will use in today's experiment.)
%%%

If you guessed "about 2" you were in good company, and the real 4.1 is the whole point: a ball that remembers gets *four times* the push of a ball that does not. Nice — you have just understood the upgrade that most real training runs still rely on today.

!!! c-info 🔬
<b>Optional (skippable) — Momentum in symbols.</b> Skip this box freely; the words above are the whole idea.<br><br>

With velocity `v`, slope `g`, momentum coefficient `β` (Greek "beta", how much past speed you keep) and learning rate `η`: **`v ← β·v + g`**, then `w ← w − η·v`.<br><br>

Set `β = 0` and you get plain SGD back. This "keep some, add the whole new slope" form is the one this lesson uses everywhere — the demo above and today's experiment both use exactly `v = β·v + g`. One honest warning you can test yourself later: `β` is a *knob*, not a magic constant. 0.9 is the common starting point, but on a very steep valley 0.9 can build so much speed that the ball overshoots and finishes *slower* than plain SGD. Today's experiment uses a gentler 0.6 for exactly that reason, and invites you to try 0.9 and watch it happen.<br><br>

Heads-up for the next section: **Adam** writes its velocity in slightly different bookkeeping — `β·v + (1−β)·g`, which just rescales the same idea. "Keep some past speed" is identical, and you never have to choose: `SGD(momentum=0.9)` does it for you.
!!!

Momentum made the ball *steadier* by remembering direction, so it stops bouncing across a valley. The next upgrade fixes a different unfairness: what if one weight needs big steps while another needs tiny ones?

@@@ concept id=c4 tag="A stride per weight" title="RMSProp & Adam — every weight gets its own step size" gotit="Got adaptive rates"
Here is the last upgrade, and it fixes a sneaky unfairness. So far *one* learning rate — one stride length — has set the step for *every* weight at once. Picture a group hike where everyone is forced to take the **exact same size step**. The person on a gentle path is fine. The person scrambling down a steep rocky slope takes that same big step and tumbles. The person on a nearly flat stretch takes that same tiny step and barely moves. One stride cannot fit everybody. The fix is obvious once you see it: let *each hiker* choose their *own* stride for *their* piece of ground. Adaptive optimizers do exactly that — they give **every weight its own learning rate**, automatically.

%%% svg
<svg viewBox="0 0 520 198" role="img" aria-label="Two hikers forced to use the same stride. Left panel, one global stride: a hiker on a steep cliff takes a huge stride and stumbles, while a hiker on flat ground takes a tiny stride and barely moves. Right panel, per-hiker stride: each hiker takes a stride suited to their own terrain and both make steady progress."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One stride for everyone vs a stride tuned per hiker</text><rect x="14" y="28" width="232" height="158" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="130" y="46" text-anchor="middle" fill="#C93B3B">same stride for both</text><path d="M30 70 L90 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="52" cy="99" r="6" fill="#C93B3B"/><path d="M52 99 l40 30" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="3,2"/><text x="70" y="150" fill="#C93B3B" font-size="8">steep → giant step, stumbles</text><path d="M150 150 L236 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="170" cy="144" r="6" fill="#C93B3B"/><path d="M170 144 l10 0" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="3,2"/><text x="196" y="140" fill="#C93B3B" font-size="8">flat → tiny step, stuck</text><rect x="256" y="28" width="250" height="158" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="381" y="46" text-anchor="middle" fill="#1a5c38">each picks its own stride</text><path d="M272 70 L332 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="294" cy="99" r="6" fill="#2D8B55"/><path d="M294 99 l14 10" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="3,2"/><text x="312" y="150" fill="#1a5c38" font-size="8">steep → smaller step ✓</text><path d="M400 150 L496 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="418" cy="144" r="6" fill="#2D8B55"/><path d="M418 144 l40 0" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="3,2"/><text x="448" y="138" fill="#1a5c38" font-size="8">flat → bigger step ✓</text></g></svg>
%%%

**What the hiking-stride picture gets right:** forcing one stride on everyone helps nobody, and letting each hiker pick their own fits the ground. That *is* per-weight learning rates.

**Where it breaks down:** hikers choose a stride by *looking* around. The optimizer cannot see the ground ahead, so it *estimates* each weight's stride from that weight's recent slopes — big recent slopes mean "steep here, step smaller."

#### The puzzle — one global rate cannot fit every weight
Suppose one input feature ranges 0–1 and another ranges 0–10,000. They are **ill-scaled**: wildly different sizes. Or one feature is **sparse** — mostly zero, so its weight almost never gets a slope at all.

%%% steps
step: the mismatch
why: the big-scale weight gets huge slopes while the sparse weight gets almost none, so their step needs differ by a factor of thousands
step: why one rate cannot win
why: a rate big enough for the sleepy weight makes the loud weight explode, therefore one global learning rate cannot suit every weight — someone always stumbles or stalls
%%%

And the repair is two moves, straight out of the hiking picture:

%%% steps
step: the fix — let each weight measure its own ground
why: keep, per weight, a running average of how big its own recent slopes are; that number IS "how steep is it here for me"
step: the fix — divide by it
why: dividing that weight's step by its own slope size gives adaptive per-parameter rates, so a steep weight steps smaller and a sleepy weight steps bigger — each weight gets its own step size
%%%

%%% insight
Feeling like this one is heavier than momentum? It is, and that reaction is completely normal — you are meeting the idea that took the field from 2011 to 2014 to settle. Just hold on to the hiking picture: *everyone picks their own stride*. The maths only ever measures "how steep is my own patch."
%%%

%%% hint
t1: Do not worry about the formula yet — just ask one question about a single weight: has its slope been big or small lately?
t2: Take a weight with a big slope, say 100. Divide its step by its own slope size: 100 ÷ 100 = 1. Now a tiny-slope weight, 0.01 ÷ 0.01 = 1 too. Both end up a sensible size.
t3: Each weight divides its own step by how big its own recent slopes are, so a steep weight steps smaller and a flat weight steps bigger — every weight gets a stride that fits its own terrain.
%%%

#### The two names you will hear
Two optimizers do this, and the second is the one you will meet absolutely everywhere.

**[[RMSProp||"root mean square propagation": it keeps, for each weight, a running average of that weight's recent squared slopes, and divides that weight's step by the square root of it — so noisy-big-slope weights get smaller steps, quiet weights get bigger ones]]** keeps a running average of each weight's recent **squared slopes**. Squaring just makes every slope count as a positive size.

Then it divides that weight's step by the square root of that average. Which means a loud, big-slope weight gets a *smaller* stride, and a quiet weight gets a *bigger* one. Every weight is auto-scaled to its own patch of ground.

**[[Adam||"adaptive moment estimation": it combines Momentum's running velocity with RMSProp's per-weight scaling, so each weight gets both built-up speed AND its own stride — the common default optimizer]]** is simply **Momentum plus RMSProp in one.** It carries Momentum's running velocity *and* RMSProp's per-parameter scaling, so every weight gets built-up speed *and* its own stride.

That is why Adam is the default optimizer most people reach for first. When a paper says "we trained with Adam," this is it — and so did the chatbot on your phone and the recommender picking your next video. Adam is quietly one of the most-used lines of code in the world.

Here is the whole family on one line each, so you can see every member is the plain step plus one twist:

%%% table
:: Optimizer :: The one twist it adds :: In one line
plain SGD :: nothing — raw slope × one rate :: step straight downhill, no memory
Momentum :: a running velocity (memory of direction) :: builds speed, smooths the zig-zag
RMSProp :: a per-weight stride from recent squared slopes :: each weight scaled to its own terrain
Adam :: Momentum + RMSProp together :: built-up speed AND a per-weight stride — the default
%%%

Now *feel* the per-weight fix on two weights with very different slope sizes. Make your guess first — the two numbers on the right are the punchline:

%%% demo id=adaptive label="reveal — one rate vs a per-weight stride"
predict: one weight has slope 100, the other 0.01, and both share lr = 0.1. Guess each one's step size. Then guess what the SAME two weights do once each divides by its own slope size.
code: lr = 0.1   # one-rate step = lr*slope ; adaptive step = lr*slope/sqrt(slope**2)
out: big-slope  w:   one-rate step = 10.000    adaptive step = 0.100
     tiny-slope w:   one-rate step =  0.001    adaptive step = 0.100
take: <b>One global rate makes the big-slope weight leap 10.0 while the tiny-slope weight crawls 0.001</b> — a ten-thousand-fold unfairness from one shared number. Divide each by its own slope size (RMSProp's trick) and BOTH take a sensible 0.1 step. That per-weight rescaling is why Adam trains ill-scaled and sparse problems smoothly where plain SGD stalls.
%%%

That is the whole idea, and you just watched it work: same lr, same slopes, one small division, and the unfairness disappears. Every hiker got the stride their own ground deserved.

!!! c-info 🔬
<b>Optional (skippable) — Adam in symbols.</b> Safe to skip; the hiking picture already told you everything that matters.<br><br>

Adam tracks a velocity `m ← β₁·m + (1−β₁)·g` — that is Momentum's idea, in "keep-some, mix-in-some" bookkeeping.<br><br>

It also tracks a squared-slope average `v ← β₂·v + (1−β₂)·g²` — that is RMSProp's per-weight measurement.<br><br>

Then it steps `w ← w − η·m / (√v + ε)`. The tiny `ε` (about 1e-8) only stops a divide-by-zero. A small "bias-correction" tweak nudges the first few steps; its algebra lives in the interview deep-dive.<br><br>

You never compute any of this by hand: `Adam(lr=0.001)` does all of it for you.
!!!

You now have the whole family — plain, speedy, and per-weight-smart. Before the recap, one honest truth: even the cleverest optimizer has a wall it cannot climb.

@@@ concept id=c5 tag="Steps too big + the wall" title="When the step overshoots — and the wall no optimizer climbs" gotit="Got the limit"
Two last things before we gather it all up: the one failure that bites *every* optimizer, and one honest limit that *no* optimizer can fix. Start with the failure, and go back to our marble. Imagine the **ball rolling** into a valley — except now it cannot roll smoothly, it can only *hop* a fixed distance each time, like a kid jumping between paving stones. If the hop is wider than the valley, the ball sails clean over the bottom and lands *higher* on the far wall. Then it hops back, even higher. Up and out, instead of settling. That is a step size (learning rate) set **too large**.

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="A child hopping between paving stones across a dip. The hop is wider than the dip, so instead of landing at the bottom the hopper lands higher up the far side, and the next hop lands higher still on the near side. Each ball rests on the ground curve. Labels read hop one sails over the bottom and hop two lands higher still."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A hop wider than the dip lands you HIGHER on the far side</text><path d="M30 60 C 120 78, 190 150, 260 154 C 330 150, 400 78, 490 60" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text x="260" y="172" text-anchor="middle" fill="#6B645E" font-size="8.5">the bottom you wanted</text><circle cx="112" cy="90" r="7" fill="#2A7B9B" stroke="#fff" stroke-width="1.4"/><text x="112" y="110" text-anchor="middle" fill="#1F6280" font-size="8.5">start</text><path d="M118 84 Q 262 20 412 78" fill="none" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="4,3"/><polygon points="416,81 405,74 409,70" fill="#C93B3B"/><circle cx="420" cy="85" r="7" fill="#C93B3B" stroke="#fff" stroke-width="1.4"/><text x="262" y="36" text-anchor="middle" fill="#C93B3B" font-size="9">hop 1: sails over the bottom, lands higher</text><path d="M414 92 Q 244 152 68 73" fill="none" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="4,3"/><polygon points="64,69 75,73 71,79" fill="#C93B3B"/><circle cx="60" cy="68" r="7" fill="#C93B3B" stroke="#fff" stroke-width="1.4"/><text x="250" y="136" text-anchor="middle" fill="#C93B3B" font-size="9">hop 2: lands higher still ✗</text></g></svg>
%%%

**What the paving-stone hop gets right:** a jump wider than the dip overshoots the bottom every single time and climbs the far side, so the loss goes *up*, not down.

**Where it breaks down:** a real child can see the dip and shorten the jump. The optimizer is blindfolded — it only ever feels the tilt where it stands, which is exactly why an oversized jump can skip the bottom without ever noticing.

#### The puzzle — a rate too *large*
Let us name exactly what you would see on screen, and why:

%%% steps
step: the symptom
why: the loss **oscillates** — up, down, up higher — and then **diverges** and blows up, often ending as `NaN` ("not a number," what maths gives you when a value runs to infinity)
step: the cause
why: the step is simply longer than the distance to the bottom, therefore every jump lands on the opposite wall instead of in the dip
%%%

Good news: this is one of the easiest failures in all of machine learning to fix. Two moves, and the second one has a name worth knowing — a [[learning-rate schedule||a plan that shrinks the learning rate as training goes on — big steps early for speed, small steps late to settle; also called decay]], also called **decay**.

%%% steps
step: remedy 1 — shrink the step
why: lower the learning rate (dividing it by 3 or 10 is the usual first move) so the hop fits inside the valley
step: remedy 2 — shrink it *over time*
why: start big for speed, then shrink the learning rate as training goes on — big early hops cover ground, and smaller steps later let the ball settle gently instead of bouncing
%%%

Here is the same run drawn three ways — the sensible rate, the too-big rate, and the too-big rate rescued by a schedule:

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three loss curves over steps. A green curve with a sensible learning rate slides down and flattens near the bottom. A red curve with too-large a learning rate zig-zags upward and diverges off the top of the chart, labelled overshoots then blows up. A dashed amber curve shows a learning-rate schedule shrinking the steps so the run settles."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Same model, three learning-rate stories</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><path d="M52 44 C 150 120, 320 144, 494 148" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text x="360" y="140" fill="#2D8B55" font-size="9">sensible rate ✓ (falls, settles)</text><path d="M52 120 L 96 66 L 140 128 L 184 44 L 228 136 L 272 30" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text x="120" y="30" fill="#C93B3B" font-size="9">too big → overshoots, blows up (NaN)</text><path d="M272 30 C 320 60, 400 96, 494 110" fill="none" stroke="#C99A12" stroke-width="1.8" stroke-dasharray="5,3"/><text x="380" y="104" fill="#9A7208" font-size="8.5">schedule shrinks the step → settles</text></g></svg>
%%%

%%% insight
Read those three curves once more, because this is a superpower you now own: the *shape* of a loss curve tells you what to change. Jagged and climbing means the step is too big. Flat and barely moving means it is too small. Falling then flattening means you got it right. Engineers training the models you use every day stare at exactly this picture all day long — and you can now read it too.
%%%

Want to check that you can? Predict the verdict for a run before you look at it:

%%% demo id=readcurve label="reveal — read the curve, name the fix"
predict: a run prints loss 2.30, then 41.7, then 980.4, then nan. What went wrong, and what is your first move?
code: for step, loss in enumerate(history): print(f'step {step}: loss = {loss}')
out: step 0: loss = 2.30
     step 1: loss = 41.7
     step 2: loss = 980.4
     step 3: loss = nan
take: <b>The loss climbing and then hitting NaN is the too-large learning rate, every time.</b> First move: divide the lr by 10 and rerun. Second move: add a schedule so the rate decays as training goes on. If instead the numbers had barely moved (2.30, 2.29, 2.29…), the diagnosis flips: the rate is too small — raise it, add momentum, or switch to an adaptive optimizer.
%%%

So far: you can spot a bad step size and fix it. Now the harder truth — the thing a *perfect* step size still cannot buy you.

#### The honest wall
An optimizer is *only* a way of walking downhill on the ground it is handed. It cannot invent a better path than the ground allows. So three things it simply cannot do:

%%% steps
step: it cannot fix a bad loss
why: if the "how wrong am I" score measures the wrong thing, the optimizer will faithfully roll to the bottom of the WRONG hill — and look perfectly healthy doing it
step: it cannot make a non-separable problem separable
why: remember XOR — one neuron draws one straight line, and no straight line splits XOR; swapping SGD for Adam changes the walk, not the model, so the ceiling stays exactly where it was
step: it cannot promise the lowest point of all
why: on a bumpy ([[non-convex||a landscape with more than one dip, so the lowest nearby point may not be the lowest point of all]]) landscape it feels only the tilt where it stands, therefore it can settle in a shallow dip and never learn that a deeper one existed — no optimizer guarantees the global minimum
%%%

The common thread: an optimizer **only follows the gradient** it is given. That third one is easiest to just *see*:

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A bumpy loss landscape with two dips. A ball has come to rest at the lowest point of the shallower left dip, with uphill on both sides, labelled the optimizer stops here. The deeper right dip is lower but unreached, labelled the real best, out of reach. A caption reads the optimizer only follows the slope it is given."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">An optimizer only follows the slope it is given</text><path d="M30 60 C 90 130, 150 120, 190 96 C 230 72, 270 150, 340 152 C 400 154, 440 90, 500 60" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="129" cy="113" r="7" fill="#C99A12" stroke="#fff" stroke-width="1.5"/><text x="129" y="95" text-anchor="middle" fill="#9A7208" font-size="9">🛑 stops here</text><text x="112" y="134" text-anchor="middle" fill="#9A7208" font-size="8.5">shallow dip</text><circle cx="340" cy="152" r="6" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="384" y="152" fill="#1a5c38" font-size="9">← the real best</text><text x="330" y="170" text-anchor="middle" fill="#6B645E" font-size="8.5">deeper, out of reach from the left dip</text></g></svg>
%%%

**The honest takeaway:** a flat, settled loss curve is *good news*, but it is not a promise you found the lowest possible loss. You found the bottom of whatever bowl your ball happened to roll into. (Cheerful footnote for later: inside the huge networks behind real models these dips are usually shallow and rarely a real problem.)

A better optimizer takes a *smarter path down the same hill*; it never builds a new hill. Knowing that line is what separates someone who uses optimizers from someone who understands them — and you now know it.

@@@ concept id=c6 tag="Recap" title="The whole day on one page" gotit="Got the recap"
You just turned one plain downhill step into a small family of smart ones — and every one of them is still just a **ball rolling** down the "how wrong am I" hill, only rolling more cleverly. Here is the whole arc in five beats:

- **The plain step.** Feel the tilt under the ball, slide a little downhill: *new weight = old weight − learning rate × slope.* The learning rate is the step size.
- **Three flavors.** *Batch* asks all the data (accurate, slow), *SGD* asks one example (fast, noisy), *mini-batch* asks a small handful (the everyday sweet spot). We sample because many rough steps beat one perfect step.
- **Momentum.** Keep a running velocity, so a steady downhill *builds up speed* and side-to-side bounces *cancel* — it damps the zig-zag and cures the too-slow crawl.
- **RMSProp & Adam.** Give every weight its own stride, measured from its own recent slopes. **Adam = Momentum + RMSProp**, the default behind most models you have heard of.
- **The failures and the wall.** Too-big rate → overshoots and blows up (remedy: smaller rate, or a *schedule* that decays it). Too-small → crawls (remedy: bigger rate, momentum, or an adaptive optimizer). And the ceiling: an optimizer *only follows the gradient it is handed* — it cannot fix a bad loss, split XOR, or promise the global lowest point.

One picture to keep: same hill, smarter ways down it.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Three paths down the same valley, side by side. Plain SGD takes many small even steps and arrives last. Momentum builds up speed and arrives sooner. Adam takes a well-sized step per weight and arrives soonest. A caption reads same hill, smarter steps."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Same hill — smarter steps get down sooner</text><path d="M30 40 Q 165 150 300 150 Q 400 150 500 46" fill="none" stroke="#B8AEA2" stroke-width="2"/><text x="300" y="163" text-anchor="middle" fill="#6B645E" font-size="8">bottom</text><g fill="#C93B3B"><circle cx="70" cy="70" r="3"/><circle cx="100" cy="90" r="3"/><circle cx="130" cy="106" r="3"/><circle cx="160" cy="120" r="3"/><circle cx="192" cy="132" r="3"/><circle cx="228" cy="142" r="3"/></g><text x="86" y="60" fill="#C93B3B" font-size="8.5">plain SGD: many small steps</text><g fill="#2A7B9B"><circle cx="84" cy="80" r="3.4"/><circle cx="132" cy="107" r="3.4"/><circle cx="204" cy="136" r="3.4"/><circle cx="268" cy="148" r="3.4"/></g><text x="150" y="92" fill="#1F6280" font-size="8.5">Momentum: builds speed</text><g fill="#2D8B55"><circle cx="108" cy="94" r="3.8"/><circle cx="224" cy="141" r="3.8"/><circle cx="292" cy="150" r="3.8"/></g><text x="250" y="118" fill="#1a5c38" font-size="8.5">Adam: right-sized per-weight step</text></g></svg>
%%%

And the one-glance fix table, for the day you are staring at a bad loss curve:

%%% table
:: What you see :: What it means :: What to try first
loss climbs, then NaN :: the step is too large — it overshoots the bottom :: divide the lr by 10; then add a decay schedule
loss barely moves :: the step is too small — it crawls :: raise the rate, add momentum, or switch to Adam
loss zig-zags down slowly :: a narrow ravine — bouncing wall to wall :: momentum (and adaptive rates if weights are ill-scaled)
loss falls, then flattens :: it is working — you found a bottom :: keep going; remember it may not be the deepest bowl
%%%

Keep this cheat-sheet nearby — it holds every word from today in one place:

%%% jargon
optimizer | the rule that turns a slope into a weight step — how the ball rolls
learning rate (lr) | the step size: how far the ball moves each step
gradient descent | the plain step: new weight = old weight − lr × slope
batch gradient descent | one step from the slope over the whole dataset — accurate but slow
stochastic gradient descent (SGD) | one step from one example — fast but noisy
mini-batch gradient descent | one step from a small handful (e.g. 32–64) — the everyday default
momentum | keep a running velocity of past slopes so steady downhill builds speed and zig-zags cancel
RMSProp | scale each weight's step by the square root of its own recent squared slopes
Adam | Momentum + RMSProp: built-up speed AND a per-weight stride — the common default
learning-rate schedule (decay) | shrink the learning rate over training: big early, small late
non-convex | a landscape with more than one dip, so a settled loss is not a promise of the lowest point
%%%

@@@ quiz id=quiz tag="Quiz" title="Four quick checks" gotit="answer all first"
%%% quiz
q: In "w ← w − lr × slope", what does the learning rate (lr) control? | a:1 | which way is downhill | how big each step is | how many examples you use | the number of weights | fb: The lr is the step size — how far the ball slides each step.
q: What is the everyday default flavor of gradient descent? | a:2 | full-batch (all data per step) | one giant step total | mini-batch (a small handful per step) | no steps at all | fb: Mini-batch — many steps, and the small average keeps each fairly steady.
q: What does Momentum add to the plain step? | a:1 | a bigger dataset | a running velocity (memory of recent slopes) | a second loss | a random restart | fb: A running velocity, so steady downhill builds speed and side-to-side bounces cancel.
q: Adam is best described as… | a:2 | plain SGD with a bigger lr | full-batch only | Momentum + RMSProp (per-weight scaling) together | a way to change the loss | fb: Adam = Momentum's velocity + RMSProp's per-weight stride — the common default.
%%%

@@@ produce id=produce tag="Produce" title="Race three optimizers down the same valley" gotit="Done"
Time to *see* the whole day in one run: plain, then speedy, then per-weight — each reaching the bottom sooner. You will drop three optimizers onto the *same* long, narrow valley and count how fast each one gets there.

The valley is `loss = 0.5 * (4*wx**2 + 0.2*wy**2)`, so its slopes are `gx = 4*wx` (steep across `wx`) and `gy = 0.2*wy` (gentle along `wy`). That is exactly the ravine from the Momentum section.

All three start at `wx = wy = 1.5` and share the **same** learning rate `lr = 0.3` for a fair race, for `30` steps. Momentum uses the same `v = keep*v + slope` rule as the demo, with `keep = 0.6`. The adaptive one divides each weight's step by the square root of a running average of that weight's own squared slopes (RMSProp's trick).

One honest note before you run it: the demo used the common `keep = 0.9`, and here we use a gentler `0.6`. That is deliberate, not a typo — this valley is *very* steep across `wx`, and 0.9 builds so much speed that the ball overshoots. Try it and watch Momentum finish *slower* than plain SGD. That is a real failure mode, and finding it yourself is worth more than being told.

**Predict first, before you run it:** which optimizer reaches `loss < 0.01` *first* — plain SGD, Momentum, or the adaptive one? And roughly how many steps apart do you think they will be?

**What you should see** when you run it:
- **Plain SGD is slowest.** It crawls down the gentle `wy` direction one tiny step at a time and only reaches `loss < 0.01` around **step 26**.
- **Momentum is much faster** — it builds speed down that same gentle direction and gets there around **step 13**. Same learning rate, same valley: the running velocity is the whole difference, roughly halving the trip.
- **The adaptive (RMSProp-style) optimizer is fastest of all** — around **step 4**. By rescaling each weight to its own slope size, it takes a well-sized step in *both* directions from the very start instead of crawling in one of them.
- Read the printed table and check the order against your guess: **plain → speedy → per-weight, each reaching the bottom sooner.** That is today's whole story in three lines of output.

Write it in `experiment.py` and run it. Then poke it — the script prints your result either way, so nothing breaks:
- Set Momentum's `keep` to `0.0` and watch it turn back into plain SGD (same step 26 — that *is* the proof that momentum is the only difference).
- Set `keep` to `0.9` and watch too much speed *hurt* on this steep valley.
- Nudge the shared `lr` up to `0.6` and watch plain SGD overshoot and blow up — the too-big-step failure from the last concept, live in your terminal (the adaptive one and Momentum survive it, which is a lesson in itself).
- Move the start to `3.0` and see the gaps stretch; the adaptive one's fast start shrinks, because its very first step is always about the same size no matter how steep the ground is.

**Option A — write it yourself.** Fill in `experiment.py` with the three loops (one plain, one with a velocity, one with per-weight scaling) and print, for each, the first step where `loss < 0.01`.

**Option B — hand it to the lab.** Paste this into the experiment lab and let it build `experiment.py` for you:

%%% prompt id=pp label="ask Claude to build it"
Help me build my artifact.
Create experiment.py for Day 7 (optimizers). On the SAME valley
loss = 0.5*(4*wx**2 + 0.2*wy**2)  (so gx = 4*wx, gy = 0.2*wy), run three
optimizers, all starting at wx = wy = 1.5, all with learning rate lr = 0.3,
for 30 steps:
  1) plain SGD:      wx -= lr*gx ; wy -= lr*gy
  2) Momentum:       vx = 0.6*vx + gx ; vy = 0.6*vy + gy ; wx -= lr*vx ; wy -= lr*vy
  3) Adaptive (RMSProp-style): keep sqx = 0.9*sqx + 0.1*gx*gx (same for y),
     then wx -= lr*gx/(sqrt(sqx)+1e-8)  (same for y)
For each optimizer, print the FIRST step where loss < 0.01 (print "never" if it
never gets there) and the loss at step 30.
Expected with those defaults: plain SGD first crosses 0.01 near step 26,
Momentum near step 13, and the adaptive one near step 4 — plain -> speedy ->
per-weight, each sooner. Make the constants LR, START and KEEP easy to edit at
the top, and only run the "did I get the expected numbers" self-check when they
are still at their defaults, so experimenting never fails an assertion.
%%%

@@@ fin
