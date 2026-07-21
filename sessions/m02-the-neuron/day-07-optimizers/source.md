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
@lede Picture a **ball rolling** down into a valley. Set it on the slope and it does the smart thing all by itself — it rolls the way the ground tips, faster where the hill is steep, slower where it flattens, and it can even coast through a little bump because it built up speed. Now here is the surprising part: teaching a neural network is *almost exactly* this — a ball finding the bottom of a hill. Every model you have heard of, from a tiny neuron up to ChatGPT, learns by rolling a ball downhill on a landscape of "how wrong am I." The thing that decides *how* that ball rolls — plain, or with built-up speed, or clever about every single step — is called an **optimizer**, and it is one of the big reasons modern models train in hours instead of weeks. Yesterday you built the loop that takes one downhill step. Today you will meet three ways to take that step, each smarter than the last.
@goal Together we will upgrade the "nudge" step of the loop three times. You will start with plain **gradient descent** (the raw downhill step) and its three flavors: using **all** the data, **one** example, or a small **batch** per step. Then you will add **Momentum** — a running speed that smooths a jittery path, just like a real rolling ball. Then **RMSProp** and **Adam**, which give *every* weight its *own* step size. Along the way you will poke the ways stepping goes wrong — steps too big, too small, or zig-zagging — each a little puzzle with a neat fix. Every formula shows up in plain words first; any heavy math sits in a skippable box.

@@@ concept id=c1 tag="The plain step" title="Gradient descent — the ball's raw downhill step" gotit="Got the plain step"
Let us start with the plainest way to roll the ball, the one you already met in yesterday's loop. Stand at the ball and look straight down at the ground under your feet. Which way does it tip? *That* is the direction the ball wants to roll — downhill, toward less "how wrong am I." The plainest optimizer does exactly that and nothing more: read the slope right under the ball, then slide the ball a little way in the downhill direction. How far it slides each time is set by one number — the [[learning rate||the step size: how far the ball moves each step. Written "lr". A bigger lr takes bigger slides, a smaller lr takes smaller slides]] — which just scales the slope before it moves the weight. This plain roll-with-the-slope move is called **gradient descent**.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A U-shaped valley drawn as a curve. A ball sits partway up the left slope. An arrow under the ball shows the slope tipping downhill to the right. A second, ghosted ball shows where one step lands, a little way down the slope toward the bottom. A label reads step size = learning rate times slope."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Read the slope under the ball → slide it one step downhill</text><path d="M40 40 Q 160 190 260 176 Q 360 190 480 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text x="260" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5" dy="14">bottom = smallest loss</text><circle cx="118" cy="118" r="9" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text x="118" y="100" text-anchor="middle" fill="#1F6280" font-size="9">the ball (current weight)</text><path d="M132 132 l52 26" stroke="#C93B3B" stroke-width="2"/><polygon points="184,158 173,155 176,146" fill="#C93B3B"/><text x="205" y="150" fill="#C93B3B" font-size="9">slope tips downhill →</text><circle cx="186" cy="150" r="8" fill="#2A7B9B" opacity="0.35" stroke="#1F6280" stroke-dasharray="2,2"/><text x="230" y="176" fill="#2D8B55" font-size="9">↑ one step lands here</text><text x="260" y="40" text-anchor="middle" fill="#9A7208" font-size="9">step size = learning rate × slope</text></g></svg>
%%%

**What the rolling-ball picture gets right:** the ball moves *downhill* on its own, and a steeper slope means a bigger push — exactly how the step works. **Where it breaks down:** a real ball has weight and keeps rolling on its own; plain gradient descent has *no memory at all* — each step reads only the slope right where it stands and instantly forgets the last one. (Giving it memory is the very next upgrade — that is Momentum.)

#### The one-line update rule
The whole move is one short sentence: **new weight = old weight − learning rate × slope.** The slope (the *gradient*) points *uphill* toward more loss, so the **minus sign** turns the ball around to roll *downhill*. Here it is as one line:

%%% formula
expr: w ← w − lr × (slope of the loss at w)
note: "w" is a weight, "lr" is the learning rate (the step size). The MINUS sign walks downhill — opposite the uphill slope. Same rule you built in yesterday's loop; today we make it smarter.
%%%

Now let *you* drive the ball. Below is a real valley with a **learning rate** slider. Drag it small and watch the ball inch down in tiny steps. Drag it big and watch the step overshoot the bottom and land on the far wall. Predict what happens at each end *before* you drag — then see if you were right.

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
  // static bowl (loss = w^2)
  var bd='';for(var i=0;i<=60;i++){var w=WLO+(WHI-WLO)*i/60;bd+=(i?'L':'M')+px(w).toFixed(1)+' '+py(w*w).toFixed(1)+' ';}
  bowl.setAttribute('d',bd.trim());
  function paint(){
    var a=+lr.value;lrv.textContent=a.toFixed(2);
    var w=W0,pts=[],maxAbs=0;
    for(var i=0;i<=N;i++){pts.push(w);maxAbs=Math.max(maxAbs,Math.abs(w));w=w-a*2*w;} // step: w -= lr*d(w^2)/dw
    // trajectory polyline + step dots
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

That plain step — read the slope, slide downhill — is the foundation. Every fancier optimizer today is *this move plus one clever twist*. But before the twists, one basic question: the slope of *what*? All your data? One example? That choice has three famous flavors, coming up next.

@@@ concept id=c2 tag="How much data per step" title="Batch, stochastic, mini-batch — the three flavors" gotit="Got the three flavors"
Before we make the ball smarter, one honest question about that slope: the slope of *what*? Think of it like a **class taking a vote** on which way to go. You could ask *every* student before you take a step — very accurate, but slow, especially if the class has a million students. Or you could ask *one* random student and step right away — super fast, but that one voice might point you a bit wrong. Or the sensible middle: ask a small *handful* and go with their average. Those three choices have names, and picking one is the first real decision every optimizer makes.

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="Three ways to decide which way to step, drawn as polling a class. Left: ask the whole class, all icons highlighted, labelled batch, accurate but slow. Middle: ask one random student, one icon highlighted, labelled stochastic, fast but noisy. Right: ask a small handful, a few icons highlighted, labelled mini-batch, the practical middle."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">How many examples do you ask before each step?</text><rect x="14" y="30" width="158" height="150" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="93" y="50" text-anchor="middle" fill="#C93B3B">ask the WHOLE class</text><g fill="#C93B3B"><circle cx="46" cy="78" r="7"/><circle cx="72" cy="78" r="7"/><circle cx="98" cy="78" r="7"/><circle cx="124" cy="78" r="7"/><circle cx="59" cy="102" r="7"/><circle cx="85" cy="102" r="7"/><circle cx="111" cy="102" r="7"/><circle cx="46" cy="126" r="7"/><circle cx="72" cy="126" r="7"/><circle cx="98" cy="126" r="7"/><circle cx="124" cy="126" r="7"/></g><text x="93" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">batch: accurate, but slow</text><rect x="182" y="30" width="156" height="150" rx="6" fill="#FDF3D6" stroke="#C99A12"/><text x="260" y="50" text-anchor="middle" fill="#9A7208">ask ONE student</text><g fill="#D8CFBE"><circle cx="216" cy="78" r="7"/><circle cx="242" cy="78" r="7"/><circle cx="294" cy="78" r="7"/><circle cx="229" cy="102" r="7"/><circle cx="281" cy="102" r="7"/><circle cx="216" cy="126" r="7"/><circle cx="294" cy="126" r="7"/></g><circle cx="268" cy="78" r="9" fill="#C99A12"/><text x="260" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">stochastic: fast, but noisy</text><rect x="348" y="30" width="158" height="150" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="427" y="50" text-anchor="middle" fill="#1a5c38">ask a HANDFUL</text><g fill="#C7DDCB"><circle cx="382" cy="78" r="7"/><circle cx="460" cy="78" r="7"/><circle cx="395" cy="102" r="7"/><circle cx="382" cy="126" r="7"/><circle cx="460" cy="126" r="7"/></g><g fill="#2D8B55"><circle cx="408" cy="78" r="7"/><circle cx="434" cy="78" r="7"/><circle cx="447" cy="102" r="7"/></g><text x="427" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">mini-batch: the sweet spot</text></g></svg>
%%%

**What the class-vote picture gets right:** more voices means a truer direction but a slower decision, and one voice is quick but jumpy — that trade is exactly real. **Where it breaks down:** in a real vote everyone answers at once; a network's "asking" is doing math on each example, so asking more genuinely costs more time and memory, which is why the middle option won.

#### The three flavors, in network words
Each flavor is just a choice of how many examples you measure the slope on before you take one step:

- **Batch gradient descent** uses the [[whole dataset||every training example is used to compute one averaged slope before a single step is taken]] to compute one slope, then takes one step. The direction is *stable and accurate*, but one step can take forever on a big dataset — you compute over *all* the data just to move once.
- **Stochastic gradient descent (SGD)** goes to the other extreme: measure the slope on [[one example||a single randomly chosen training example gives the slope for this step — quick to compute, but jumpy]] and step *immediately*. Steps are *fast and cheap*, but *noisy* — each single example points a bit off, so the path jitters.
- **Mini-batch gradient descent** is the practical middle: measure the slope on a small [[batch||a small handful of examples, often 32 or 64, averaged for each step — fast like SGD, steadier than one example]] — often 32 or 64 examples — then step. Fast like SGD, but the average smooths out most of the jitter. This is what almost everyone actually uses.

#### Why we left full-batch behind (the SGD motivation)
Here is the whole reason SGD was born. Early on, people used full-batch — measure on *all* the data, then move once. On a small dataset that is fine. But real datasets have millions of examples, so waiting to look at *every* one before taking a *single* step means you crawl. The fix was almost cheeky: **don't wait — sample.** Look at just a few examples, take a step *now*, and repeat. You take *many* rough steps in the time one perfect step would cost, and many rough steps beat one perfect step. See it as steps-per-pass:

%%% demo id=flavors label="run it — steps taken in the same time budget"
code: n_examples = 1_000_000   # the whole dataset
code: # cost is roughly "how many examples you touch per step"
code: for name, per_step in [('batch', n_examples), ('mini-batch', 64), ('sgd', 1)]:
code:     steps = n_examples // per_step   # steps you can afford in one full-data budget
code:     print(f'{name:11s}: {steps:>9,} steps per full pass')
out: batch      :         1 steps per full pass
out: mini-batch :    15,625 steps per full pass
out: sgd        : 1,000,000 steps per full pass
take: <b>Same data budget, wildly different number of steps.</b> Full-batch buys ONE step per pass; mini-batch buys ~15,625; pure SGD buys a million. More (rougher) steps almost always learn faster than one (perfect) step — that is why we sample. Mini-batch is the sweet spot: many steps, and the small average keeps each one fairly steady.
%%%

So the modern default is *mini-batch* SGD: sample a small batch, take the plain downhill step, repeat. Now that we know *what* slope we are stepping on, let us make the *step itself* smarter — starting with giving our ball some real rolling speed.

@@@ concept id=c3 tag="Give it speed" title="Momentum — a ball that builds up speed" gotit="Got Momentum"
Now the first real upgrade, and it is the most fun. Remember our plain ball has *no memory* — it forgets every step the instant it takes it. But a *real* **ball rolling** downhill does not forget: it builds up **speed**. Roll down a long straight slope and it goes faster and faster. Hit a little bump and it coasts right over because it carried momentum into the bump. And if the ground keeps tugging it left, then right, then left — the left and right tugs partly *cancel*, and the ball mostly keeps heading the way it was already going. **Momentum** the optimizer copies exactly this: instead of stepping by the raw slope each time, it keeps a running **velocity** — a memory of recent slopes — and steps by the velocity.

%%% svg
<svg viewBox="0 0 520 188" role="img" aria-label="A ball rolling down a long slope, drawn with speed streaks growing longer as it descends, showing it building up speed. It coasts over a small bump partway down without stopping. Labels read builds up speed and coasts over the bump."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A real rolling ball builds speed — and coasts over little bumps</text><path d="M30 44 C 140 70, 210 118, 280 128 Q 300 116 320 128 C 400 138, 460 150, 500 158" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="70" cy="52" r="6" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="58" y1="52" x2="70" y2="52" stroke="#2A7B9B" stroke-width="1.4"/><circle cx="180" cy="104" r="7" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="158" y1="102" x2="180" y2="104" stroke="#2A7B9B" stroke-width="1.6"/><circle cx="410" cy="141" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="372" y1="137" x2="410" y2="141" stroke="#2A7B9B" stroke-width="1.8"/><text x="120" y="46" fill="#1F6280" font-size="8.5">slow start</text><text x="440" y="132" fill="#1F6280" font-size="8.5">fast — built up speed</text><text x="300" y="106" text-anchor="middle" fill="#9A7208" font-size="8.5">bump</text><text x="300" y="150" text-anchor="middle" fill="#2D8B55" font-size="8.5">coasts over it ✓</text></g></svg>
%%%

**What the rolling-ball-with-speed picture gets right:** momentum builds up along a steady downhill and can carry the ball over a small bump — both are real behaviors of the optimizer. **Where it breaks down:** a real ball can pick up *too much* speed and overshoot the valley; the optimizer's velocity also fades a little each step (we "leak" some speed on purpose), so it does not run away forever like a frictionless ball would.

#### Puzzle — plain SGD crawls, then zig-zags, in a narrow valley
Here is the exact problem Momentum was invented to fix. Many loss valleys are not round bowls — they are long, *narrow ravines*, steep across the narrow direction and only gently sloped along the long direction. Plain SGD reads a slope each step, and that slope is big across the narrow direction but tiny along the long one. So the ball takes *tiny* steps *along* the valley (it barely creeps toward the bottom) while the big across-slope makes it bounce from wall to wall — **zig-zagging** side to side and hardly moving forward. **Cause:** one slope is huge, the other tiny, so the ball wastes its motion on the steep side-to-side direction and crawls along the gentle one. **Remedy:** *Momentum.* Along the gentle direction every push points the *same* way, so momentum lets them **add up** into real speed; across the narrow direction the pushes point *opposite* on each bounce, so **Momentum smooths** them out — its running velocity **cancels the side-to-side** motion. In short, Momentum **damps the zig-zag** and builds speed down the valley, so it reaches the bottom far sooner. Watch the before and after:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Before and after of a path down a narrow valley. Left, plain SGD: the path zig-zags sharply from one wall to the other while creeping slowly toward the bottom. Right, with momentum: the side-to-side bounces cancel and the path glides much more smoothly down toward the bottom."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Narrow valley: plain SGD zig-zags · Momentum glides through</text><text x="130" y="34" text-anchor="middle" fill="#C93B3B" font-size="10">plain SGD</text><path d="M60 44 Q 130 120 200 44" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><path d="M60 60 Q 130 130 200 60" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><polyline points="80,52 182,66 90,82 176,96 100,110 168,122 118,134 150,142" fill="none" stroke="#C93B3B" stroke-width="2"/><circle cx="150" cy="142" r="4" fill="#C93B3B"/><text x="130" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5">bounces wall to wall,</text><text x="130" y="188" text-anchor="middle" fill="#6B645E" font-size="8.5">creeps down slowly</text><line x1="260" y1="30" x2="260" y2="160" stroke="#E5DFD6" stroke-dasharray="4,3"/><text x="390" y="34" text-anchor="middle" fill="#2D8B55" font-size="10">with Momentum</text><path d="M320 44 Q 390 120 460 44" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><path d="M320 60 Q 390 130 460 60" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><polyline points="360,52 392,74 388,96 391,116 390,140" fill="none" stroke="#2D8B55" stroke-width="2"/><circle cx="390" cy="140" r="4" fill="#2D8B55"/><text x="390" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5">side tugs cancel →</text><text x="390" y="188" text-anchor="middle" fill="#6B645E" font-size="8.5">glides down, much smoother</text></g></svg>
%%%

That same running velocity also cures the **too-small learning rate** trap from yesterday: if every raw step is a tiny grain of sand, momentum lets many tiny steps in the *same* direction *add up* into real speed — so **using momentum** (or raising the learning rate) is the remedy when plain steps crawl.

#### The rule, in plain words
Momentum is two short lines: first update the velocity — **new velocity = a fraction of the old velocity + this step's slope** — then step by the velocity. That "fraction of the old velocity" is how much past speed you keep (often about 0.9, so you keep 90% each step). Watch the velocity build up over five steps when the slope stays a steady `1`:

%%% demo id=momentum label="run it — velocity building up (steady slope = 1)"
code: v = 0.0; slope = 1.0; keep = 0.9   # keep 90% of past velocity
code: for step in range(5):
code:     v = keep*v + slope        # new velocity = fraction of old + this slope
code:     print(f'step {step+1}: velocity = {v:.3f}  (plain SGD would just use {slope})')
out: step 1: velocity = 1.000  (plain SGD would just use 1.0)
out: step 2: velocity = 1.900  (plain SGD would just use 1.0)
out: step 3: velocity = 2.710  (plain SGD would just use 1.0)
out: step 4: velocity = 3.439  (plain SGD would just use 1.0)
out: step 5: velocity = 4.095  (plain SGD would just use 1.0)
take: <b>Steady downhill → velocity grows 1.0 → 1.9 → 2.71 → 3.44 → 4.10.</b> Plain SGD would take the same size step (1.0) every time; Momentum speeds up along a consistent slope, so it covers ground faster. Flip the slope's sign each step instead and the velocity would mostly cancel — that is exactly how it kills the zig-zag. (This is the same <b>v = keep·v + slope</b> rule you will use in today's experiment.)
%%%

!!! c-info 🔬
<b>Optional (skippable) — Momentum in symbols.</b> With velocity `v`, slope `g`, momentum coefficient `β` (Greek "beta", how much past speed you keep) and learning rate `η`: **`v ← β·v + g`**, then `w ← w − η·v`. Setting `β = 0` gives back plain SGD. This "keep some, add the whole new slope" form is the one this lesson uses everywhere — the c3 demo above and today's experiment both use exactly `v = β·v + g`. (Heads-up for the next section: **Adam** writes its velocity in a slightly different bookkeeping — `β·v + (1−β)·g`, which just rescales the same idea — but "keep some past speed" is identical, and you never have to choose: `SGD(momentum=0.9)` does it for you.)
!!!

Momentum made the ball *steadier* by remembering direction, so it stops bouncing across a valley. The next upgrade fixes a different problem: what if one weight needs big steps while another needs tiny ones? For that we give *every* weight its *own* step size.

@@@ concept id=c4 tag="A stride per weight" title="RMSProp & Adam — every weight gets its own step size" gotit="Got adaptive rates"
Here is the last upgrade, and it fixes a sneaky unfairness. So far *one* learning rate — one stride length — has set the step for *every* weight at once. But imagine a group hike where everyone must take the **exact same stride length**. The person on a gentle path is fine, but the person scrambling down a steep cliff takes that same big stride and tumbles, while the person on a nearly flat stretch takes that same tiny stride and barely moves. One stride simply cannot fit everyone. The fix is obvious once you see it: let *each hiker* pick their *own* stride for *their* terrain. Adaptive optimizers do exactly that — give **every weight its own learning rate**, automatically.

%%% svg
<svg viewBox="0 0 520 198" role="img" aria-label="Two hikers forced to use the same stride. Left panel, one global stride: a hiker on a steep cliff takes a huge stride and stumbles, while a hiker on flat ground takes a tiny stride and barely moves. Right panel, per-hiker stride: each hiker takes a stride suited to their own terrain and both make steady progress."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One stride for everyone vs a stride tuned per hiker</text><rect x="14" y="28" width="232" height="158" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="130" y="46" text-anchor="middle" fill="#C93B3B">same stride for both</text><path d="M30 70 L90 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="52" cy="99" r="6" fill="#C93B3B"/><path d="M52 99 l40 30" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="3,2"/><text x="70" y="150" fill="#C93B3B" font-size="8">steep → giant step, stumbles</text><path d="M150 150 L236 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="170" cy="144" r="6" fill="#C93B3B"/><path d="M170 144 l10 0" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="3,2"/><text x="196" y="140" fill="#C93B3B" font-size="8">flat → tiny step, stuck</text><rect x="256" y="28" width="250" height="158" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="381" y="46" text-anchor="middle" fill="#1a5c38">each picks its own stride</text><path d="M272 70 L332 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="294" cy="99" r="6" fill="#2D8B55"/><path d="M294 99 l14 10" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="3,2"/><text x="312" y="150" fill="#1a5c38" font-size="8">steep → smaller step ✓</text><path d="M400 150 L496 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="418" cy="144" r="6" fill="#2D8B55"/><path d="M418 144 l40 0" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="3,2"/><text x="448" y="138" fill="#1a5c38" font-size="8">flat → bigger step ✓</text></g></svg>
%%%

**What the hiking-stride picture gets right:** forcing one stride on everyone helps no one, and letting each pick their own fits the terrain — that is exactly per-weight learning rates. **Where it breaks down:** hikers *choose* their stride by looking around; the optimizer cannot see the terrain, so it *estimates* the right stride for each weight from that weight's recent slopes — big recent slopes mean "steep here, step smaller."

#### Puzzle — one global rate can't fit ill-scaled or sparse weights
This is the failure adaptive rates were made for. Suppose one input feature ranges 0–1 and another ranges 0–10,000 (they are **ill-scaled** — wildly different sizes), or one feature is **sparse** (mostly zero, so its weight rarely gets a slope at all). With a single learning rate, the rate that suits the big-slope weight is far too big for the tiny-slope weight, and vice versa — someone always stumbles or stalls. **Cause:** one global learning rate cannot suit weights whose slopes differ by huge factors. **Remedy:** *adaptive per-parameter rates* — give each weight a stride scaled to *its own* recent slopes.

#### The two names you will hear — RMSProp and Adam
Two optimizers do this, and Adam is the one you will meet everywhere:

- [[RMSProp||"root mean square propagation": it keeps, for each weight, a running average of that weight's recent squared slopes, and divides that weight's step by the square root of it — so noisy-big-slope weights get smaller steps, quiet weights get bigger ones]] keeps a running average of *each weight's* recent **squared slopes** (squaring makes every slope count as a positive size), then divides that weight's step by the square root of it. A weight with big, noisy slopes gets a *smaller* stride; a weight with tiny slopes gets a *bigger* stride. Each weight is auto-scaled to its own terrain.
- [[Adam||"adaptive moment estimation": it combines Momentum's running velocity with RMSProp's per-weight scaling, so each weight gets both built-up speed AND its own stride — the common default optimizer]] is simply **Momentum + RMSProp in one**: it keeps Momentum's running velocity *and* RMSProp's per-weight scaling. So every weight gets both built-up speed *and* its own stride. Adam is the optimizer most people reach for first — when a paper or tutorial says "we trained with Adam," this is it.

Here is the family, so you can see each is the plain step plus one twist:

%%% table
:: Optimizer :: The one twist it adds :: In one line
plain SGD :: nothing — raw slope × one rate :: step straight downhill, no memory
Momentum :: a running velocity (memory of direction) :: builds speed, smooths the zig-zag
RMSProp :: a per-weight stride from recent squared slopes :: each weight scaled to its own terrain
Adam :: Momentum + RMSProp together :: built-up speed AND a per-weight stride — the default
%%%

And to *feel* the per-weight fix, watch two weights with very different slope sizes. A single rate serves one badly; RMSProp-style scaling divides each by its own slope size, so both take a *sensible* step:

%%% demo id=adaptive label="run it — one rate vs a per-weight stride"
code: import math
code: lr = 0.1
code: for name, slope in [('big-slope w', 100.0), ('tiny-slope w', 0.01)]:
code:     one_rate_step = lr * slope                       # same lr for both
code:     adaptive_step = lr * slope / (math.sqrt(slope*slope) + 1e-8)  # RMSProp-style scale
code:     print(f'{name:12s}: one-rate step={one_rate_step:8.3f}  adaptive step={adaptive_step:.3f}')
out: big-slope w : one-rate step=  10.000  adaptive step=0.100
out: tiny-slope w: one-rate step=   0.001  adaptive step=0.100
take: <b>One global rate makes the big-slope weight leap 10.0 while the tiny-slope weight crawls 0.001.</b> Dividing by each weight's own slope size (RMSProp's trick) gives BOTH a sensible 0.1 step. That per-weight rescaling is why Adam trains ill-scaled, sparse problems smoothly where plain SGD stalls.
%%%

!!! c-info 🔬
<b>Optional (skippable) — Adam in symbols.</b> Adam tracks a velocity `m ← β₁·m + (1−β₁)·g` (Momentum's idea, written in "keep-some, mix-in-some" bookkeeping) and a squared-slope average `v ← β₂·v + (1−β₂)·g²` (RMSProp), then steps `w ← w − η·m / (√v + ε)`. The tiny `ε` (about 1e-8) just avoids dividing by zero. There is also a small "bias-correction" tweak for the first few steps — we skip its algebra here; it lives in the interview deep-dive. You never compute this by hand: `Adam(lr=0.001)` does it all.
!!!

You now have the whole family — plain, speedy, and per-weight-smart. Before the recap, one honest truth: even the cleverest optimizer has a ceiling it cannot climb past.

@@@ concept id=c5 tag="Steps too big + the wall" title="When the step overshoots — and the wall no optimizer climbs" gotit="Got the limit"
Two last things before we gather it all up: the one failure that bites *every* optimizer, and one honest limit that *no* optimizer can fix. First the failure. Think again of the **ball rolling** into a valley, but now imagine it can only teleport a *fixed distance* each step. If that distance is bigger than the width of the valley, the ball leaps clean over the bottom and lands *higher* on the far wall — then leaps back even higher — bouncing up and out instead of settling. That is a step size (learning rate) set **too large**.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two loss curves over steps. A green curve with a sensible learning rate slides down and flattens near the bottom. A red curve with too-large a learning rate zig-zags upward and diverges off the top of the chart, labelled overshoots then blows up. A dashed line shows a learning-rate schedule shrinking the red steps so it settles."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Step too big → the loss overshoots and blows up</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><path d="M52 44 C 150 120, 320 144, 494 148" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text x="360" y="140" fill="#2D8B55" font-size="9">sensible rate ✓ (falls, settles)</text><path d="M52 120 L 96 66 L 140 128 L 184 44 L 228 136 L 272 30" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text x="120" y="30" fill="#C93B3B" font-size="9">too big → overshoots, blows up (NaN)</text><path d="M272 30 C 320 60, 400 96, 494 110" fill="none" stroke="#C99A12" stroke-width="1.8" stroke-dasharray="5,3"/><text x="380" y="104" fill="#9A7208" font-size="8.5">schedule shrinks the step → settles</text></g></svg>
%%%

**What the leaping-ball picture gets right:** a step wider than the valley overshoots the bottom every time and climbs the far wall — the loss goes *up*. **Where it breaks down:** a real ball rolls smoothly and cannot teleport; the optimizer moves in discrete jumps, which is exactly why an oversized jump can skip the minimum entirely.

#### Puzzle — the learning rate too *large*, and its remedy
Set the rate too big and each step overshoots the bottom, so the loss **oscillates** and then **diverges** — it can blow up to `NaN` ("not a number," what you get from math gone infinite). **Cause:** the step is larger than the distance to the bottom. **Remedy 1:** simply *lower the learning rate.* **Remedy 2 (the smart one):** start with a big rate for speed, then *shrink it as training goes on* so the ball settles gently near the bottom instead of bouncing. Deliberately shrinking the rate over time is a [[learning-rate schedule||a plan that shrinks the learning rate as training goes on — big steps early for speed, small steps late to settle; also called decay]] (also called **decay**). Big early steps to cover ground, small late steps to settle — the best of both.

#### The honest wall — an optimizer only follows the gradient it is handed
Here is the truth that separates someone who *uses* optimizers from someone who *understands* them. An optimizer is *only* a way of walking downhill on the slope it is given. It cannot invent a better path than the ground allows. So three things it simply *cannot* do, no matter how clever:

- It **cannot fix a bad loss.** If the "how wrong am I" score measures the wrong thing, the optimizer will faithfully roll to the bottom of the *wrong* hill.
- It **cannot make a non-separable problem separable.** Remember XOR from earlier — a single neuron can only draw one straight line, and no line splits XOR. Swap SGD for Adam and it still cannot: the ceiling is the *model*, not the step.
- It **cannot guarantee the lowest point overall.** On a bumpy ([[non-convex||a landscape with more than one dip, so the lowest nearby point may not be the lowest point of all]]) landscape it feels only the slope right where it stands, so it can settle into a shallow dip and think it is done.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A bumpy loss landscape with two dips. A ball has rolled into the shallower left dip and stopped, with uphill on both sides, labelled the optimizer stops here. The deeper right dip is lower but unreached, labelled the real best, out of reach. A caption reads the optimizer only follows the slope it is given."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">An optimizer only follows the slope it is given</text><path d="M30 60 C 90 130, 150 120, 190 96 C 230 72, 270 150, 340 152 C 400 154, 440 90, 500 60" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="170" cy="108" r="7" fill="#C99A12" stroke="#fff" stroke-width="1.5"/><text x="170" y="88" text-anchor="middle" fill="#9A7208" font-size="9">🛑 stops here</text><text x="150" y="130" text-anchor="middle" fill="#9A7208" font-size="8.5">shallow dip</text><circle cx="320" cy="150" r="6" fill="none" stroke="#2D8B55" stroke-width="2"/><text x="362" y="150" fill="#1a5c38" font-size="9">← the real best</text><text x="330" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">deeper, out of reach from the left dip</text></g></svg>
%%%

**The honest takeaway:** a flat, settled loss curve is *good news* but not a *guarantee* you found the lowest possible loss — you found the bottom of *whatever bowl the ball happened to roll into*. (Good news for later: in the huge networks behind real models these dips are usually shallow and rarely a real problem — but it is a limit worth knowing.) A better optimizer takes a *smarter path down the same hill*; it does not build a new hill. Now let us gather the whole day onto one page.

@@@ concept id=c6 tag="Recap" title="The whole day on one page" gotit="Got the recap"
You just turned one plain downhill step into a small family of *smart* ones — every one of them still just a **ball rolling** down the "how wrong am I" hill, only rolling more cleverly. Here is the whole arc in a few beats:

- **The plain step.** Read the slope under the ball, slide a little downhill: *new weight = old weight − learning rate × slope.* The learning rate is the step size.
- **Three flavors.** *Batch* asks all the data (accurate, slow), *SGD* asks one example (fast, noisy), *mini-batch* asks a small handful (the everyday sweet spot). We sample because many rough steps beat one perfect step.
- **Momentum.** Keep a running velocity so steady downhill *builds up speed* and side-to-side bounces *cancel* — it damps the zig-zag and cures the too-slow crawl.
- **RMSProp & Adam.** Give *every weight its own stride* from its own recent slopes. **Adam = Momentum + RMSProp**, and it is the default you will see named everywhere.
- **The failures & the wall.** Too-big rate → overshoots and blows up (remedy: smaller rate, or a *schedule* that decays it). Too-small → crawls (remedy: bigger rate, momentum, or an adaptive optimizer). And the honest ceiling: an optimizer *only follows the gradient it is handed* — it cannot fix a bad loss, separate XOR, or promise the global lowest point.

One picture to keep: same hill, smarter ways down it.

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Three paths down the same valley, side by side. Plain SGD takes many small even steps and arrives last. Momentum builds up speed and arrives sooner. Adam takes a well-sized step per weight and arrives soonest. A caption reads same hill, smarter steps."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Same hill — smarter steps get down sooner</text><path d="M30 40 Q 165 150 300 150 Q 400 150 500 46" fill="none" stroke="#B8AEA2" stroke-width="2"/><text x="300" y="150" text-anchor="middle" fill="#6B645E" font-size="8" dy="13">bottom</text><g fill="#C93B3B"><circle cx="70" cy="66" r="3"/><circle cx="100" cy="86" r="3"/><circle cx="130" cy="104" r="3"/><circle cx="160" cy="118" r="3"/><circle cx="192" cy="130" r="3"/><circle cx="228" cy="140" r="3"/></g><text x="86" y="60" fill="#C93B3B" font-size="8.5">plain SGD: many small steps</text><g fill="#2A7B9B"><circle cx="80" cy="72" r="3.4"/><circle cx="128" cy="102" r="3.4"/><circle cx="196" cy="132" r="3.4"/><circle cx="262" cy="149" r="3.4"/></g><text x="150" y="92" fill="#1F6280" font-size="8.5">Momentum: builds speed</text><g fill="#2D8B55"><circle cx="96" cy="84" r="3.8"/><circle cx="196" cy="134" r="3.8"/><circle cx="284" cy="150" r="3.8"/></g><text x="250" y="118" fill="#1a5c38" font-size="8.5">Adam: right-sized per-weight step</text></g></svg>
%%%

Keep this cheat-sheet nearby — it is every word from today in one place:

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

@@@ quiz id=quiz tag="Quiz" title="Four quick checks" gotit="answer all first"
%%% quiz
q: In "w ← w − lr × slope", what does the learning rate (lr) control? | a:1 | which way is downhill | how big each step is | how many examples you use | the number of weights | fb: The lr is the step size — how far the ball slides each step.
q: What is the everyday default flavor of gradient descent? | a:2 | full-batch (all data per step) | one giant step total | mini-batch (a small handful per step) | no steps at all | fb: Mini-batch — many steps, and the small average keeps each fairly steady.
q: What does Momentum add to the plain step? | a:1 | a bigger dataset | a running velocity (memory of recent slopes) | a second loss | a random restart | fb: A running velocity, so steady downhill builds speed and side-to-side bounces cancel.
q: Adam is best described as… | a:2 | plain SGD with a bigger lr | full-batch only | Momentum + RMSProp (per-weight scaling) together | a way to change the loss | fb: Adam = Momentum's velocity + RMSProp's per-weight stride — the common default.
%%%

@@@ produce id=produce tag="Produce" title="Race three optimizers down the same valley" gotit="Done"
Time to *see* the whole spine in one run: plain, then speedy, then per-weight — each reaching the bottom sooner. You will drop three optimizers onto the *same* long, narrow valley and count how fast each one gets to the bottom.

The valley is `loss = 0.5 * (4*wx**2 + 0.2*wy**2)`, so its slopes are `gx = 4*wx` (steep across `wx`) and `gy = 0.2*wy` (gentle along `wy`). All three start at `wx = wy = 1.0` and use the **same** learning rate `lr = 0.3` for a fair race, for `30` steps. Momentum uses the same rule as the demo above — `v = 0.6*v + slope` (keep 60% of past speed, add this slope) — then steps by `v`. The adaptive one divides each weight's step by the square root of a running average of that weight's own squared slopes (RMSProp's trick).

**Predict first, before you run it:** which optimizer reaches `loss < 0.01` *first* — plain SGD, Momentum, or the adaptive one? Order your guess.

**What you should see** when you run it:
- **Plain SGD is slowest.** It crawls down the gentle `wy` direction one tiny step at a time and only reaches `loss < 0.01` around **step 19**.
- **Momentum is much faster** — it builds up speed down that same gentle direction, so it gets to `loss < 0.01` around **step 8**. Same learning rate, same valley: the running velocity is the whole difference.
- **The adaptive (RMSProp-style) optimizer is fastest of all** — by rescaling each weight to its own slope size it takes a well-sized step in *both* directions from the start and is already at `loss < 0.01` by **step 1**.
- Read off the printed table and check the order matches your prediction: **plain → speedy → per-weight, each reaching the bottom sooner.** That is today's spine in three lines of output.

Write it in `experiment.py` and run it. Then poke it: set Momentum's `0.6` down to `0.0` and watch it turn back into plain SGD; nudge the shared `lr` up toward `0.5` and watch plain SGD start to overshoot and blow up — the too-big-step failure from the last concept, live.

**Option A — write it yourself.** Fill in `experiment.py` with the three loops (one plain, one with a velocity, one with per-weight scaling) and print, for each, the first step where `loss < 0.01`.

**Option B — hand it to the lab.** Paste this into the experiment lab and let it build `experiment.py` for you:

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my artifact.
Create experiment.py for Day 7 (optimizers). On the SAME valley
loss = 0.5*(4*wx**2 + 0.2*wy**2)  (so gx = 4*wx, gy = 0.2*wy), run three
optimizers, all starting at wx = wy = 1.0, all with learning rate lr = 0.3,
for 30 steps:
  1) plain SGD:      wx -= lr*gx ; wy -= lr*gy
  2) Momentum:       vx = 0.6*vx + gx ; vy = 0.6*vy + gy ; wx -= lr*vx ; wy -= lr*vy
  3) Adaptive (RMSProp-style): keep sqx = 0.9*sqx + 0.1*gx*gx (same for y),
     then wx -= lr*gx/(sqrt(sqx)+1e-8)  (same for y)
For each optimizer, print the FIRST step where loss < 0.01 and the loss at step 30.
Expected: plain SGD first crosses 0.01 near step 19, Momentum near step 8,
and the adaptive one by step 1 — plain -> speedy -> per-weight, each sooner.
%%%

@@@ fin
