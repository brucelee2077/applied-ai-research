---
quest_id: wf3-d04-optimizers
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 7 — Optimizers: SGD → Momentum → Adam"
module_label: "Module 2 · Train · Day 7"
zh_module_label: "第 2 模块 · 训练 · 第 7 天"
title: "Optimizers: SGD → Momentum → Adam"
zh_title: "优化器：SGD → Momentum → Adam"
subtitle: "Smarter Ways to Step Downhill"
zh_subtitle: "更聪明的下坡走法"
brand_sub: "Foundations · M2 Day 7"
spine: "rolling ball"
zh_spine: "滚动的球"
nav_prev_href: "../day-06-training-loop/lesson.html"
nav_prev_label: "The Training Loop"
zh_nav_prev_label: "训练循环"
nav_next_href: "../day-08-learning-rate/lesson.html"
nav_next_label: "Learning-Rate Intuition"
zh_nav_next_label: "学习率的直觉"
fin_title: "Module 2 · Day 7 complete! 🏆"
zh_fin_title: "第 2 模块 · 第 7 天 完成！🏆"
fin_body: "Nice work — you've met the <b>optimizers</b>: plain SGD takes the raw downhill step, Momentum builds speed like a ball rolling downhill, and Adam adapts the step size for every weight. Same gradient, smarter steps.<br>Next up: <b>Learning Rate</b> — the single knob that sets how big every step is."
zh_fin_body: "做得好 —— 你认识了这些 <b>optimizer（优化器）</b>：plain SGD 走原始的下坡一步，Momentum 像滚下坡的球那样攒速度，Adam 给每一个 weight 单独调步长。同样的梯度，更聪明的步子。<br>下一站：<b>学习率</b> —— 决定每一步多大的那一个旋钮。"
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
@zh_lede 把一颗弹珠放在倾斜的托盘上，然后松手。你不用告诉它任何事 —— 它就自己顺着托盘倾斜的方向**滚**下去，坡陡的地方滚得快，坡平的地方滚得慢，甚至因为攒了速度还能滑过一个小坑。让人意外的地方来了：训练一个神经网络*几乎就是*这件事。你提问的那个聊天机器人、手机上的人脸解锁、会玩电子游戏的 AI —— 它们每一个都是靠让一颗**滚动的球**去找一座由「我错得多厉害」堆成的山的谷底，才学会本领的。决定那颗球*怎么*滚的规则 —— 朴素地滚、带着攒起来的速度滚、还是为每一个 weight 都聪明地滚 —— 叫做 **optimizer（优化器）**。它是那些模型能在几小时而不是几个月里训练出来的一个重要原因。昨天你搭出了走一步下坡的那个循环。今天你把那一步升级三次。
@zh_goal 我们会一起把这个循环里「轻推」那部分变聪明，一共三次。你会从朴素的 **gradient descent（梯度下降）**（最原始的下坡一步）和它的三种口味开始 —— 每一步用**全部**数据、**一个**样本，还是一小**批**。然后你加上 **Momentum**，一个会累积的速度，像真弹珠那样把颠簸的路径抹平。再来 **RMSProp** 和 **Adam**，它们给*每一个* weight 自己的步长。路上你会去戳一戳走步会出的那些错 —— 太大、太小、走 Z 字形 —— 每一个都是一个小谜题，配一个漂亮的解法。每个公式都先用大白话讲一遍，重的数学放在一个你可以完全跳过的盒子里。

~~~zh
%%% warmup
q: 接着昨天：训练循环的四步按顺序是什么？ | a:2 | loss → forward → update → backward | update → backward → loss → forward | forward → loss → backward → update | backward → forward → update → loss | concept: training-loop | fb: 一圈是 forward（出手）→ loss（量偏了多少）→ backward（往哪边偏）→ update（调准星），然后重复。
q: 接着昨天：在「w ← w − lr × gradient」里，那个减号在做什么？ | a:1 | 它删掉坏的 weight | gradient 指向上坡，所以减号把 weight 往下坡推、走向更小的 loss | 它只是个记号，没有作用 | 它在数 epoch | concept: update-rule | fb: gradient 指向 loss 更大的上坡方向，所以减掉它就是让 weight 往下坡走。
q: 接着昨天：learning rate 控制什么，太大了会怎样？ | a:0 | 每一圈的步长 —— 太大就会迈过头，loss 可能炸掉 | 层数 —— 太大就加神经元 | 哪边是下坡 —— 太大就把符号翻过来 | loss 的值本身 —— 太大就是零 loss | concept: learning-rate | fb: learning rate 就是步长；太大的话每一次轻推都迈过谷底，于是 loss 来回震荡然后发散。
%%%
~~~

@@@ concept id=c1 zh_tag="朴素的一步" tag="The plain step" zh_title="梯度下降 —— 球最原始的下坡一步" title="Gradient descent — the ball's raw downhill step" zh_gotit="懂了朴素的一步" gotit="Got the plain step"
Let us start with the plainest way to roll the marble — the one already sitting inside yesterday's loop. Crouch down next to the ball and look at the ground right under it. Which way does it tip? *That* is the way the ball wants to go: downhill, toward less "how wrong am I." The plainest optimizer does that and nothing else. Feel the tilt under the ball, slide the ball a little way downhill, repeat. How far it slides each time comes from one number — the [[learning rate||the step size: how far the ball moves each step. Written "lr". A bigger lr takes bigger slides, a smaller lr takes smaller slides]] — which simply scales the slope before it moves the weight. This roll-with-the-tilt move is called **gradient descent**.
~~~zh
我们先从让弹珠滚起来最朴素的办法开始 —— 也就是昨天那个循环里已经装着的那个。蹲到球旁边，看看它正下方的地面。它朝哪边倾？*那*就是球想去的方向：下坡，走向更小的「我错得多厉害」。最朴素的优化器就做这件事，别的都不做。感受球下面的倾斜，把球往下坡挪一小段，重复。它每次挪多远，来自一个数字 —— [[learning rate||步长：球每一步走多远。写作 "lr"。lr 越大每次挪得越远，越小挪得越近]] —— 它只是在移动 weight 之前把斜率缩放一下。这个「顺着倾斜滚」的动作叫做 **gradient descent（梯度下降）**。
~~~

%%% svg
<svg viewBox="0 0 520 202" role="img" aria-label="A marble resting on a tilted tray drawn as a U-shaped valley. An arrow under the marble follows the ground tipping downhill to the right. A ghosted marble sits further along the same curve, showing where one slide lands. A label reads step size equals learning rate times tilt."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Feel the tilt under the marble → slide it one step downhill</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">感受弹珠下面的倾斜 → 把它往下坡挪一步</text><path d="M40 40 Q 160 190 260 176 Q 360 190 480 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text class="lang-en" x="300" y="194" text-anchor="middle" fill="#6B645E" font-size="8.5">bottom = smallest loss</text><text class="lang-zh" x="300" y="194" text-anchor="middle" fill="#6B645E" font-size="8.5">谷底 = 最小的 loss</text><circle cx="118" cy="122" r="9" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text class="lang-en" x="112" y="104" text-anchor="middle" fill="#1F6280" font-size="9">the marble (current weight)</text><text class="lang-zh" x="112" y="104" text-anchor="middle" fill="#1F6280" font-size="9">弹珠（当前的 weight）</text><path d="M130 132 l46 26" stroke="#C93B3B" stroke-width="2"/><polygon points="178,159 166,157 170,148" fill="#C93B3B"/><text class="lang-en" x="210" y="146" fill="#C93B3B" font-size="9">the tray tips this way →</text><text class="lang-zh" x="210" y="146" fill="#C93B3B" font-size="9">托盘朝这边倾 →</text><circle cx="186" cy="165" r="8" fill="#2A7B9B" opacity="0.4" stroke="#1F6280" stroke-dasharray="2,2"/><text class="lang-en" x="212" y="180" fill="#2D8B55" font-size="9">↑ one slide lands here</text><text class="lang-zh" x="212" y="180" fill="#2D8B55" font-size="9">↑ 挪一步落在这里</text><text class="lang-en" x="260" y="42" text-anchor="middle" fill="#9A7208" font-size="9">step size = learning rate × tilt</text><text class="lang-zh" x="260" y="42" text-anchor="middle" fill="#9A7208" font-size="9">步长 = learning rate × 倾斜</text></g></svg>
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
~~~zh
**托盘上的弹珠这个比喻对在哪里：** 球完全靠自己往*下坡*走，而且越陡推力越大。那正是这一步。

**它在哪里不成立：** 真弹珠有重量，会自己继续滚。朴素的 gradient descent *完全没有记忆* —— 每一步只感受它站的地方的倾斜，上一步转身就忘。（给它记忆正好就是下一个升级。）

#### 整个动作，一级一级来
这条规则用大白话讲就是一句短话：**新 weight = 老 weight − learning rate × 斜率。** 我们慢慢走，一级只挪一小步，别让任何东西溜过去。

%%% steps
step: 感受倾斜
why: 球下面的斜率（也就是 *gradient*）告诉你地面朝哪边升高
step: 转过身
why: 我们想去更低的地方，所以往相反方向走 —— 那个减号就是干这个的
step: 走一段，长短由 lr 定
why: 斜率给出方向和陡度，learning rate 决定你顺着它走多远
step: 新 weight = 老 weight − learning rate × 斜率
why: 这就是整个优化器，今天所有聪明的做法都是这一行加一个小花样
%%%

到这里为止：三个小决定 —— 哪边、转身、多远 —— 你就有了一个能用的优化器。写成一行，你可以指着它看：

%%% formula
expr: w ← w − lr × (loss 在 w 处的斜率)
note: 「w」是一个 weight，「lr」是 learning rate（步长）。那个减号让你往下坡走 —— 和上坡的斜率反着来。和你昨天在循环里搭的是同一条规则；今天我们让它变聪明。
%%%

%%% insight
注意这里真正干着吓人活儿的是哪个旋钮：不是斜率（数学免费送给你的），而是 **learning rate**。挑得好，弹珠滑到谷底。挑得糟，同一个完全正确的 gradient 会把你的模型甩下悬崖。这就是为什么明天有一整天专门讲这一个数字。
%%%

现在换*你*来开这颗弹珠。下面是一个真实的山谷，带一个 **learning rate** 滑块。在你动手之前先猜：拉到最小端会发生什么，推到最大端又会发生什么？然后拖一拖，看看到底 —— 先小的，再大的。
~~~

%%% svg
<svg id="gd-svg" viewBox="0 0 520 214" role="img" aria-label="Interactive gradient descent. A U-shaped loss valley with a ball on the left slope. Drag the learning-rate slider: a small rate makes the ball inch down in tiny steps, a good rate settles it at the bottom, a rate above one half makes it hop across the bottom to the other side each step, and a rate above one makes each step overshoot and climb the far wall."><g font-family="monospace" font-size="10"><rect x="40" y="26" width="440" height="160" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><path id="gd-bowl" d="M60 40 L260 180 L460 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><polyline id="gd-traj" points="" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.8"/><circle id="gd-b0" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b1" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b2" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b3" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b4" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b5" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b6" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b7" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b8" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b9" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-ball" cx="80" cy="120" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text class="lang-en" x="260" y="200" text-anchor="middle" fill="#9A938A" font-size="9">bottom = smallest loss</text><text class="lang-zh" x="260" y="200" text-anchor="middle" fill="#9A938A" font-size="9">谷底 = 最小的 loss</text></g></svg>
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
~~~zh
大的那一端让你意外了吗？大多数人以为「步子更大 = 更快」。结果弹珠干净地跳过了谷底，爬上了对面的墙。也注意一下中间那个区域，滑块过了中点一点：球每一步都落在*对面*那一侧 —— 还在往低处走，但一半的运动都浪费了。记住这个感觉；今天快结束的时候我们会回来。

那个朴素的一步 —— 感受倾斜、往下坡挪 —— 是地基，而它现在归你了。今天所有更花哨的东西都是*这个动作加一个小花样*。但先回答一个我们跳过去的诚实问题：*什么东西*的倾斜？你全部的数据？一个样本？这个选择有三种有名的口味。
~~~

@@@ concept id=c2 zh_tag="每一步用多少数据" tag="How much data per step" zh_title="batch、stochastic、mini-batch —— 三种口味" title="Batch, stochastic, mini-batch — the three flavors" zh_gotit="懂了三种口味" gotit="Got the three flavors"
Before we make the ball smarter, that honest question: the tilt of *what*? Picture a **class deciding which way to walk** on a school trip. You could ask *every single* student before you take one step — very accurate, but slow, and painfully slow if the class has a million kids. Or you could ask *one* random student and walk immediately — super quick, though that one voice may point you slightly wrong. Or the sensible middle: ask a small *handful* and go with what most of them say. Those three choices have names, and picking one is the first real decision any optimizer makes.
~~~zh
在让球变聪明之前，先回答那个诚实的问题：*什么东西*的倾斜？想象**一个班在决定往哪边走**，正在春游。你可以在走出一步之前问*每一个*学生 —— 很准，但慢，如果班上有一百万个孩子就慢得难受。或者你可以问*一个*随便挑的学生然后立刻就走 —— 超快，不过那一个声音可能指得稍微偏。或者取个明智的中间值：问一小**把**人，按大多数人说的走。这三个选择都有名字，而挑哪一个是任何优化器要做的第一个真决定。
~~~

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="Three ways a class decides which way to walk. Left: ask the whole class, all student icons highlighted, labelled batch, accurate but slow. Middle: ask one random student, one icon highlighted, labelled stochastic, fast but noisy. Right: ask a small handful, a few icons highlighted, labelled mini-batch, the practical middle."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">How many classmates do you ask before each step?</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">每走一步之前你问几个同学？</text><rect x="14" y="30" width="158" height="150" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="93" y="50" text-anchor="middle" fill="#C93B3B">ask the WHOLE class</text><text class="lang-zh" x="93" y="50" text-anchor="middle" fill="#C93B3B">问全班</text><g fill="#C93B3B"><circle cx="46" cy="78" r="7"/><circle cx="72" cy="78" r="7"/><circle cx="98" cy="78" r="7"/><circle cx="124" cy="78" r="7"/><circle cx="59" cy="102" r="7"/><circle cx="85" cy="102" r="7"/><circle cx="111" cy="102" r="7"/><circle cx="46" cy="126" r="7"/><circle cx="72" cy="126" r="7"/><circle cx="98" cy="126" r="7"/><circle cx="124" cy="126" r="7"/></g><text class="lang-en" x="93" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">batch: accurate, but slow</text><text class="lang-zh" x="93" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">batch：准，但慢</text><rect x="182" y="30" width="156" height="150" rx="6" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="260" y="50" text-anchor="middle" fill="#9A7208">ask ONE student</text><text class="lang-zh" x="260" y="50" text-anchor="middle" fill="#9A7208">问一个学生</text><g fill="#D8CFBE"><circle cx="216" cy="78" r="7"/><circle cx="242" cy="78" r="7"/><circle cx="294" cy="78" r="7"/><circle cx="229" cy="102" r="7"/><circle cx="281" cy="102" r="7"/><circle cx="216" cy="126" r="7"/><circle cx="294" cy="126" r="7"/></g><circle cx="268" cy="78" r="9" fill="#C99A12"/><text class="lang-en" x="260" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">stochastic: fast, but noisy</text><text class="lang-zh" x="260" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">stochastic：快，但吵</text><rect x="348" y="30" width="158" height="150" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="427" y="50" text-anchor="middle" fill="#1a5c38">ask a HANDFUL</text><text class="lang-zh" x="427" y="50" text-anchor="middle" fill="#1a5c38">问一小把</text><g fill="#C7DDCB"><circle cx="382" cy="78" r="7"/><circle cx="460" cy="78" r="7"/><circle cx="395" cy="102" r="7"/><circle cx="382" cy="126" r="7"/><circle cx="460" cy="126" r="7"/></g><g fill="#2D8B55"><circle cx="408" cy="78" r="7"/><circle cx="434" cy="78" r="7"/><circle cx="447" cy="102" r="7"/></g><text class="lang-en" x="427" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">mini-batch: the sweet spot</text><text class="lang-zh" x="427" y="158" text-anchor="middle" fill="#6B645E" font-size="8.5">mini-batch：那个甜点</text></g></svg>
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
~~~zh
**班级投票这个比喻对在哪里：** 声音越多方向越真但决定越慢，一个声音快但跳 —— 这个取舍是真的。**它在哪里不成立：** 真班里所有人可以同时喊；而网络的「问」意味着对每个样本做数学，所以问得越多真的越费时间和内存。正是这个成本让中间那个选项赢了。

#### 三种口味，用网络的话说
每种口味只是在决定「走一步之前在多少个样本上测倾斜」。同一个班，三种投票规模：

%%% steps
step: Batch gradient descent —— 问全班
why: 在[[整个数据集||每一个训练样本都参与算出一个平均斜率，然后才走一步]]上测斜率，然后走一步；方向很稳但慢，因为为了动一次你要摸遍所有数据
step: Stochastic gradient descent（SGD）—— 问一个学生
why: 在[[一个样本||随机挑一个训练样本给出这一步的斜率 —— 算起来快，但跳]]上测斜率然后马上走，所以又快又吵；一次一个样本指得有点偏，路径会抖
step: Mini-batch gradient descent —— 问一小把
why: 在一小批样本（常见 32 或 64）上测斜率，然后走；这是中间路线 —— 快得像 SGD，而那一小把的平均把大部分抖动抹平了
step: 赢家是…… mini-batch
why: 因此几乎所有人都用它；它保留了「步数多」的速度，而那一把样本又让每一步不至于乱跑
%%%

到这里为止：三种投票规模，一个明显的最爱。但世界*为什么*要抛弃那个准确的选项？

%%% insight
这里是初学者会起疑心的地方，而且起得有道理：我们刚刚故意选了*不那么准*的答案。这就是现代训练那个令人意外的核心 —— *现在*就走一个粗糙的方向，胜过等很久之后才走一个完美的方向。走步的速度比走步的纯度更重要。
%%%

#### 我们为什么把 full-batch 留在了身后
早期人们用 full batch：在*所有*数据上测，然后动一次。数据集小的时候这完全没问题。

但真实数据集装着几百万个样本。也就是说，在每走一步之前都用完整个数据集太慢了 —— 你会爬着走，为了动一次等啊等。

解决办法几乎有点无赖：**别等 —— 抽样。** 看几个样本，*现在*就走，重复。在展开之前先猜这个取舍有多不对等：

%%% demo id=flavzh label="先猜，再展开"
predict: 在「摸过的样本数」预算相同的情况下，full-batch 能给你几步，和 mini-batch 比？两倍？十倍？还是更多？
code: n = 1_000_000  # 数据集大小；成本 ≈ 每步摸过的样本数
out: batch      :         1 步  每过一遍数据
     mini-batch :    15,625 步 每过一遍数据
     sgd        : 1,000,000 步 每过一遍数据
take: <b>同样的数据预算，步数差得离谱。</b>Full-batch 每过一遍只买到一步；mini-batch（一次 64 个）买到约 15,625 步；纯 SGD 买到一百万步。这就是我们抽样的原因 —— 很多粗糙的步子几乎总比一个完美的步子学得快。Mini-batch 是那个甜点：步数多，而那一小把的平均让每一步还算稳。
%%%

如果那个差距的大小让你吓了一跳，很好 —— 它让大多数人都吓一跳，而它就是那个唯一的事实，解释了为什么现在没人再在每一步上用整个数据集。

所以现代的默认做法是 *mini-batch* SGD：抽一小批，走朴素的下坡一步，重复。你现在知道你踩着**什么**倾斜在走了。接下来我们让*这一步本身*变聪明，办法是给我们的弹珠一点真正的滚动速度。
~~~

@@@ concept id=c3 zh_tag="给它速度" tag="Give it speed" zh_title="Momentum —— 一个会攒速度的球" title="Momentum — a ball that builds up speed" zh_gotit="懂了 Momentum" gotit="Got Momentum"
Now the first real upgrade, and it is the most fun one. Our plain marble has *no memory*: it forgets each step the instant it takes it. But a real **ball rolling** down a long slope does not forget — it builds up **speed**. It goes faster and faster down a steady hill. It coasts straight over a small dent because it carried speed into it. And if the ground tugs it left, then right, then left, those side tugs partly *cancel* and the ball mostly keeps going the way it was already going. **Momentum** the optimizer copies that exactly: instead of stepping by the raw tilt each time, it keeps a running **velocity** — a memory of recent tilts — and steps by the velocity.
~~~zh
现在是第一个真正的升级，也是最好玩的一个。我们朴素的弹珠*没有记忆*：它走完一步的那一瞬间就忘了。但一颗真的**滚动的球**沿着长坡往下滚时不会忘 —— 它会攒起**速度**。在稳定的坡上它越滚越快。它因为带着速度，能直接滑过一个小坑。而如果地面把它往左拽、再往右拽、再往左拽，那些侧向的拽力会部分*互相抵消*，球基本还是照着原来的方向走。**Momentum** 这个优化器就是照抄这件事：它不是每次都按原始的倾斜走一步，而是保留一个持续的**速度** —— 一份最近倾斜的记忆 —— 然后按这个速度走。
~~~

%%% svg
<svg viewBox="0 0 520 188" role="img" aria-label="A ball rolling down a long slope, drawn three times with speed streaks growing longer as it descends, showing it building up speed. It coasts over a small bump partway down without stopping. Labels read slow start, coasts over it, and fast because it built up speed."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A real rolling ball builds speed — and coasts over little bumps</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">真的滚动的球会攒速度 —— 还能滑过小凸起</text><path d="M30 44 C 140 70, 210 118, 280 128 Q 300 116 320 128 C 400 138, 460 150, 500 158" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="70" cy="52" r="6" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="58" y1="52" x2="70" y2="52" stroke="#2A7B9B" stroke-width="1.4"/><circle cx="180" cy="104" r="7" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="158" y1="102" x2="180" y2="104" stroke="#2A7B9B" stroke-width="1.6"/><circle cx="410" cy="141" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.2"/><line x1="372" y1="137" x2="410" y2="141" stroke="#2A7B9B" stroke-width="1.8"/><text class="lang-en" x="120" y="46" fill="#1F6280" font-size="8.5">slow start</text><text class="lang-zh" x="120" y="46" fill="#1F6280" font-size="8.5">起步慢</text><text class="lang-en" x="440" y="132" fill="#1F6280" font-size="8.5">fast — built up speed</text><text class="lang-zh" x="440" y="132" fill="#1F6280" font-size="8.5">快 —— 攒起了速度</text><text class="lang-en" x="300" y="106" text-anchor="middle" fill="#9A7208" font-size="8.5">bump</text><text class="lang-zh" x="300" y="106" text-anchor="middle" fill="#9A7208" font-size="8.5">凸起</text><text class="lang-en" x="300" y="152" text-anchor="middle" fill="#2D8B55" font-size="8.5">coasts over it ✓</text><text class="lang-zh" x="300" y="152" text-anchor="middle" fill="#2D8B55" font-size="8.5">滑过去了 ✓</text></g></svg>
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
~~~zh
**「会攒速度的滚动的球」这个比喻对在哪里：** 速度在稳定的下坡上确实会累积，而且确实能把球带过一个小凸起。这两件事对优化器都成立。

**它在哪里不成立：** 真的球可能攒*太多*速度，冲过山谷。优化器的速度是故意每一步都衰减一点的 —— 我们「漏掉」一点速度 —— 这样它就不会像一颗无摩擦的球那样永远跑掉。

#### Momentum 被发明出来解决的那个谜题
很多 loss 山谷不是圆碗。它们是又长又窄的**沟**：在窄的那个方向很陡，在长的那个方向几乎没有倾斜。我们那颗朴素的球会走进下面这个陷阱：

%%% steps
step: 陷阱的形状
why: 斜率在窄的方向上巨大，在长的方向上极小，所以两个方向的拉力差得离谱
step: 爬行
why: 因此每一步在平缓的那个方向上只能勉强蠕动一点点 —— 球几乎没有朝谷底前进
step: Z 字形
why: 同时那个巨大的横向斜率把它从一面墙甩到另一面墙，于是它左右弹跳，几乎浪费了全部运动（这就是 plain SGD 在窄谷里那个吵闹的 Z 字形失败）
%%%

这个代价画出来是这样 —— 同一条沟走两遍，前后对比：
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Before and after of a path down a narrow valley. Left, plain SGD: the path zig-zags sharply from one wall to the other while creeping slowly toward the bottom. Right, with momentum: the side-to-side bounces cancel and the path glides much more smoothly down toward the bottom."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Narrow valley: plain SGD zig-zags · Momentum glides through</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">窄山谷：plain SGD 走 Z 字形 · Momentum 一路滑过</text><text class="lang-en" x="130" y="34" text-anchor="middle" fill="#C93B3B" font-size="10">plain SGD</text><text class="lang-zh" x="130" y="34" text-anchor="middle" fill="#C93B3B" font-size="10">plain SGD</text><path d="M60 44 Q 130 120 200 44" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><path d="M60 60 Q 130 130 200 60" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><polyline points="80,52 182,66 90,82 176,96 100,110 168,122 118,134 150,142" fill="none" stroke="#C93B3B" stroke-width="2"/><circle cx="150" cy="142" r="4" fill="#C93B3B"/><text class="lang-en" x="130" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5">bounces wall to wall,</text><text class="lang-zh" x="130" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5">在两面墙之间弹跳，</text><text class="lang-en" x="130" y="188" text-anchor="middle" fill="#6B645E" font-size="8.5">creeps down slowly</text><text class="lang-zh" x="130" y="188" text-anchor="middle" fill="#6B645E" font-size="8.5">慢慢往下爬</text><line x1="260" y1="30" x2="260" y2="160" stroke="#E5DFD6" stroke-dasharray="4,3"/><text class="lang-en" x="390" y="34" text-anchor="middle" fill="#2D8B55" font-size="10">with Momentum</text><text class="lang-zh" x="390" y="34" text-anchor="middle" fill="#2D8B55" font-size="10">用 Momentum</text><path d="M320 44 Q 390 120 460 44" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><path d="M320 60 Q 390 130 460 60" fill="none" stroke="#E5DFD6" stroke-width="1.6"/><polyline points="360,52 392,74 388,96 391,116 390,140" fill="none" stroke="#2D8B55" stroke-width="2"/><circle cx="390" cy="140" r="4" fill="#2D8B55"/><text class="lang-en" x="390" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5">side tugs cancel →</text><text class="lang-zh" x="390" y="176" text-anchor="middle" fill="#6B645E" font-size="8.5">侧向的拽力互相抵消 →</text><text class="lang-en" x="390" y="188" text-anchor="middle" fill="#6B645E" font-size="8.5">glides down, much smoother</text><text class="lang-zh" x="390" y="188" text-anchor="middle" fill="#6B645E" font-size="8.5">一路下滑，平顺得多</text></g></svg>
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
~~~zh
看右边那条路：那就是*整个*修复，而它同时来自两件事。

%%% steps
step: 修法 —— 把一致的加起来
why: 在平缓的方向上每一次推力都指向同一边，也就是说 momentum 让它们累加成真正的速度
step: 修法 —— 把不一致的抵消掉
why: 在窄的方向上每次弹跳的推力方向相反，所以那个持续的速度把左右运动抵消掉了 —— momentum 把 Z 字形抹平并且压住
%%%

%%% insight
这在真实生活里为什么咬人：一次走 Z 字形的训练*看起来*并不像坏了。loss 在下降，慢慢地，什么都不崩。你可能烧掉几个小时想着「我这个模型就是难训」，而真相是一颗球在两面墙之间弹来弹去。Momentum 常常就是那个把几小时变成几分钟的一行改动 —— 而这不是玩笑话。那个学会下棋的机器人、那个替你挑下一首歌的模型：它们能一夜之间训完而不是训一个月，很大程度上就是因为它们的球滚的时候带着记忆。
%%%

到这里为止：同一颗球，同一座山，只加了*记忆* —— 路径就完全是另一种生物了。

#### 记忆还顺手修好的第二件事
同一个持续的速度也治好了反过来的那个抱怨：learning rate 设得**太小**，每一步都像一粒沙子，loss 几乎不动，慢得难受。

**原因：** 每一步对到谷底的距离来说都太短了，所以球在爬。

**解药：** momentum 让很多同方向的小步子*累加*成真正的速度。所以当你的训练在爬行时，**用 momentum** —— 或者干脆把速率提到一个更大的 learning rate，或者伸手拿一个自适应的优化器。

#### 规则，用大白话讲
Momentum 是两行短句。先更新速度：**新速度 = 老速度的一部分 + 这一步的倾斜。** 然后按速度走一步。那个「一部分」是你留下多少过去的速度 —— 常见大约 0.9，也就是昨天的速度有 90% 带过来。在展开数字之前先猜一下：

%%% demo id=momzh label="先猜，再展开"
predict: 斜率每一步都稳定是 1.0。Plain SGD 每次都走 1.0 —— 5 步之后，你觉得 Momentum 的步子长到多大了？1.5？2？还是更多？
code: v = 0.0; keep = 0.9   # 每一步：v = keep*v + slope，其中 slope = 1.0
out: 第 1 步: velocity = 1.000   (plain SGD 会走 1.000)
     第 2 步: velocity = 1.900   (plain SGD 会走 1.000)
     第 3 步: velocity = 2.710   (plain SGD 会走 1.000)
     第 4 步: velocity = 3.439   (plain SGD 会走 1.000)
     第 5 步: velocity = 4.095   (plain SGD 会走 1.000)
take: <b>稳定下坡 → 速度从 1.0 → 1.9 → 2.71 → 3.44 → 4.10。</b>Plain SGD 永远走同样的 1.0；Momentum 在一致的坡上会加速，所以它覆盖地面越来越快。现在换成每一步都把斜率的符号翻过来：那些项大部分互相抵消，速度停在接近零 —— 这正是它杀掉 Z 字形的方式。（和你今天实验里要用的 <b>v = keep·v + slope</b> 是同一条规则。）
%%%

如果你猜的是「大概 2」，你不孤单，而真实的 4.1 就是全部重点：一颗会记事的球，推力是不会记事那颗的*四倍*。很好 —— 你刚刚搞懂了今天大多数真实训练仍然靠着的那个升级。

!!! c-info 🪜
<b>选读（可跳过）—— Momentum 的符号写法。</b>这个盒子随便跳过；上面那些话就是全部的想法。<br><br>
用速度 `v`、斜率 `g`、momentum 系数 `β`（希腊字母「beta」，你留下多少过去的速度）和 learning rate `η`：**`v ← β·v + g`**，然后 `w ← w − η·v`。<br><br>
把 `β = 0` 代进去你就拿回了 plain SGD。这个「留一些，加上整个新斜率」的形式是这节课到处在用的那个。一个诚实的提醒：`β` 是一个*旋钮*，不是什么神奇常数。0.9 是常见的起点，但在一个非常陡的山谷里，0.9 可能攒出太多速度，让球迈过头、最后比 plain SGD 还*慢*。今天的实验用了更温和的 0.6，就是因为这个，并且请你去试试 0.9、亲眼看它发生。<br><br>
给下一节打个招呼：**Adam** 把它的速度写成稍微不同的记账方式 —— `β·v + (1−β)·g`，那只是把同一个想法重新缩放了一下。「留一些过去的速度」是完全一样的，而你永远不用选：`SGD(momentum=0.9)` 会替你做好。
!!!

Momentum 靠记住方向让球更*稳*，于是它不再横穿山谷弹跳。下一个升级修的是另一种不公平：如果一个 weight 需要大步子，而另一个只需要极小的步子呢？
~~~

@@@ concept id=c4 zh_tag="每个 weight 一个步幅" tag="A stride per weight" zh_title="RMSProp 和 Adam —— 每个 weight 有自己的步长" title="RMSProp & Adam — every weight gets its own step size" zh_gotit="懂了每个 weight 的步幅" gotit="Got adaptive rates"
Here is the last upgrade, and it fixes a sneaky unfairness. So far *one* learning rate — one stride length — has set the step for *every* weight at once. Picture a group hike where everyone is forced to take the **exact same size step**. The person on a gentle path is fine. The person scrambling down a steep rocky slope takes that same big step and tumbles. The person on a nearly flat stretch takes that same tiny step and barely moves. One stride cannot fit everybody. The fix is obvious once you see it: let *each hiker* choose their *own* stride for *their* piece of ground. Adaptive optimizers do exactly that — they give **every weight its own learning rate**, automatically.
~~~zh
这是最后一个升级，它修的是一个偷偷的不公平。到目前为止，*一个* learning rate —— 一个步幅 —— 同时决定了*每一个* weight 的步子。想象一群人徒步，所有人被要求迈**完全一样大的步子**。走平缓小路的人没事。在陡峭碎石坡上下行的人迈同样的大步，摔了。走在几乎平地上的人迈同样的小步，几乎没动。一个步幅装不下所有人。看清之后解法就很明显：让*每个人*为*自己脚下的地面*挑*自己的*步幅。自适应优化器就是干这个的 —— 它们自动给**每一个 weight 自己的 learning rate**。
~~~

%%% svg
<svg viewBox="0 0 520 198" role="img" aria-label="Two hikers forced to use the same stride. Left panel, one global stride: a hiker on a steep cliff takes a huge stride and stumbles, while a hiker on flat ground takes a tiny stride and barely moves. Right panel, per-hiker stride: each hiker takes a stride suited to their own terrain and both make steady progress."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One stride for everyone vs a stride tuned per hiker</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">所有人一个步幅 vs 每个人调自己的步幅</text><rect x="14" y="28" width="232" height="158" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="130" y="46" text-anchor="middle" fill="#C93B3B">same stride for both</text><text class="lang-zh" x="130" y="46" text-anchor="middle" fill="#C93B3B">两个人同一个步幅</text><path d="M30 70 L90 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="52" cy="99" r="6" fill="#C93B3B"/><path d="M52 99 l40 30" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="3,2"/><text class="lang-en" x="70" y="150" fill="#C93B3B" font-size="8">steep → giant step, stumbles</text><text class="lang-zh" x="70" y="150" fill="#C93B3B" font-size="8">陡 → 巨大的一步，摔了</text><path d="M150 150 L236 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="170" cy="144" r="6" fill="#C93B3B"/><path d="M170 144 l10 0" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="3,2"/><text class="lang-en" x="196" y="140" fill="#C93B3B" font-size="8">flat → tiny step, stuck</text><text class="lang-zh" x="196" y="140" fill="#C93B3B" font-size="8">平 → 极小的一步，卡住</text><rect x="256" y="28" width="250" height="158" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="381" y="46" text-anchor="middle" fill="#1a5c38">each picks its own stride</text><text class="lang-zh" x="381" y="46" text-anchor="middle" fill="#1a5c38">各挑自己的步幅</text><path d="M272 70 L332 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="294" cy="99" r="6" fill="#2D8B55"/><path d="M294 99 l14 10" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="3,2"/><text class="lang-en" x="312" y="150" fill="#1a5c38" font-size="8">steep → smaller step ✓</text><text class="lang-zh" x="312" y="150" fill="#1a5c38" font-size="8">陡 → 步子更小 ✓</text><path d="M400 150 L496 150" stroke="#B8AEA2" stroke-width="1.8"/><circle cx="418" cy="144" r="6" fill="#2D8B55"/><path d="M418 144 l40 0" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="3,2"/><text class="lang-en" x="448" y="138" fill="#1a5c38" font-size="8">flat → bigger step ✓</text><text class="lang-zh" x="448" y="138" fill="#1a5c38" font-size="8">平 → 步子更大 ✓</text></g></svg>
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
~~~zh
**徒步步幅这个比喻对在哪里：** 强迫所有人一个步幅对谁都没好处，而让每个人自己挑就贴合地面。那*就是*每个 weight 各自的 learning rate。

**它在哪里不成立：** 徒步的人是*看着*四周挑步幅的。优化器看不见前面的地面，所以它从那个 weight 最近的斜率去*估计*它的步幅 —— 最近斜率大就意味着「这里陡，步子小一点」。

#### 谜题 —— 一个全局速率装不下每一个 weight
假设一个输入特征的范围是 0–1，另一个是 0–10,000。它们是**尺度失衡的**：大小差得离谱。或者某个特征是**稀疏的** —— 大部分时候是零，所以它的 weight 几乎从来拿不到斜率。

%%% steps
step: 不匹配
why: 大尺度的 weight 拿到巨大的斜率，而稀疏的 weight 几乎拿不到，所以它们需要的步长差上几千倍
step: 为什么一个速率赢不了
why: 大到能推动那个睡着的 weight 的速率，会让那个吵闹的 weight 炸掉，因此一个全局的 learning rate 没法适合每一个 weight —— 总有人要么摔跤要么卡住
%%%

而修法是两步，直接从徒步那张图里来：

%%% steps
step: 修法 —— 让每个 weight 量自己脚下的地
why: 为每个 weight 保留一个「它自己最近的斜率有多大」的滑动平均；那个数字*就是*「对我来说这里有多陡」
step: 修法 —— 除掉它
why: 把这个 weight 的步子除以它自己的斜率大小，就得到了每个参数自适应的速率，于是陡的 weight 步子更小、睡着的 weight 步子更大 —— 每个 weight 拿到自己的步长
%%%

%%% insight
觉得这个比 momentum 更重？确实是，而这个反应完全正常 —— 你正在遇到那个让这个领域从 2011 年折腾到 2014 年才定下来的想法。抓住徒步那张图就行：*每个人挑自己的步幅*。数学从头到尾只是在量「我自己这块地有多陡」。
%%%

%%% hint
t1: 先别管公式 —— 只对某一个 weight 问一个问题：它最近的斜率是大还是小？
t2: 拿一个斜率很大的 weight，比如 100。把它的步子除以它自己的斜率大小：100 ÷ 100 = 1。再拿一个斜率极小的，0.01 ÷ 0.01 也是 1。两个最后都是一个合理的大小。
t3: 每个 weight 把自己的步子除以自己最近斜率的大小，于是陡的 weight 步子小、平的 weight 步子大 —— 每个 weight 拿到贴合自己地形的步幅。
%%%

#### 你会听到的两个名字
有两个优化器做这件事，而第二个是你到处都会遇到的那个。

**[[RMSProp||「root mean square propagation」：它为每个 weight 保留那个 weight 最近平方斜率的滑动平均，并把这个 weight 的步子除以它的平方根 —— 于是斜率又吵又大的 weight 步子更小，安静的 weight 步子更大]]** 保留每个 weight 最近**平方斜率**的滑动平均。平方只是让每个斜率都按一个正的大小来算。

然后它把这个 weight 的步子除以那个平均值的平方根。也就是说，一个吵闹的、斜率大的 weight 拿到*更小*的步幅，而一个安静的 weight 拿到*更大*的。每个 weight 都被自动缩放到它自己那块地上。

**[[Adam||「adaptive moment estimation」：它把 Momentum 那个持续的速度和 RMSProp 每个 weight 的缩放合在一起，所以每个 weight 既有攒起来的速度，也有自己的步幅 —— 常见的默认优化器]]** 就是**Momentum 加 RMSProp 合成一个**。它同时带着 Momentum 那个持续的速度*和* RMSProp 每个参数的缩放，所以每个 weight 既有攒起来的速度*又*有自己的步幅。

这就是为什么 Adam 是大多数人第一个伸手拿的默认优化器。当一篇论文说「我们用 Adam 训练」，说的就是它 —— 你手机上那个聊天机器人和挑你下一个视频的推荐系统也是。Adam 悄悄地是全世界被用得最多的代码行之一。

下面是这一整家人，每个一行，让你看清每个成员都是那个朴素的一步加一个小花样：

%%% table
:: 优化器 :: 它加的那一个花样 :: 一行说完
plain SGD :: 什么都没加 —— 原始斜率 × 一个速率 :: 直接往下坡走，没有记忆
Momentum :: 一个持续的速度（方向的记忆） :: 攒速度，抹平 Z 字形
RMSProp :: 从最近平方斜率算出的每 weight 步幅 :: 每个 weight 缩放到它自己的地形
Adam :: Momentum + RMSProp 一起 :: 既有攒起来的速度又有每 weight 的步幅 —— 默认选择
%%%

现在在两个斜率大小差很多的 weight 上*感受一下*这个每 weight 的修法。先猜 —— 右边那两个数字是笑点：

%%% demo id=adazh label="先猜，再展开"
predict: 一个 weight 斜率是 100，另一个是 0.01，两个共用 lr = 0.1。猜猜各自的步长。然后猜同样这两个 weight 在各自除以自己的斜率大小之后会怎样。
code: lr = 0.1   # 单速率步长 = lr*slope ；自适应步长 = lr*slope/sqrt(slope**2)
out: 大斜率  w:   单速率步长 = 10.000    自适应步长 = 0.100
     小斜率  w:   单速率步长 =  0.001    自适应步长 = 0.100
take: <b>一个全局速率让大斜率的 weight 跳了 10.0，而小斜率的 weight 爬了 0.001</b> —— 一个共用的数字造成了一万倍的不公平。让每个都除以自己的斜率大小（RMSProp 的把戏），两个都走一个合理的 0.1。正是这个每 weight 的重新缩放，让 Adam 能在尺度失衡和稀疏的问题上平顺训练，而 plain SGD 在那里会卡住。
%%%

这就是全部的想法，而你刚刚看着它起作用：同样的 lr、同样的斜率，一个小小的除法，不公平就消失了。每个徒步的人都拿到了自己脚下地面该有的那个步幅。

!!! c-info 🪜
<b>选读（可跳过）—— Adam 的符号写法。</b>可以放心跳过；徒步那张图已经告诉你所有要紧的事了。<br><br>
Adam 追踪一个速度 `m ← β₁·m + (1−β₁)·g` —— 那是 Momentum 的想法，换成「留一些、混一些」的记账方式。<br><br>
它同时追踪一个平方斜率的平均 `v ← β₂·v + (1−β₂)·g²` —— 那是 RMSProp 每个 weight 的测量。<br><br>
然后它走一步 `w ← w − η·m / (√v + ε)`。那个极小的 `ε`（大约 1e-8）只是防止除以零。一个小小的「bias-correction」修正会微调最前面几步；它的代数放在面试深挖里。<br><br>
这些你永远不用手算：`Adam(lr=0.001)` 全都替你做了。
!!!

你现在有了一整家人 —— 朴素的、带速度的、和每 weight 聪明的。在回顾之前，一个诚实的事实：就算最聪明的优化器，也有一道它翻不过去的墙。
~~~

@@@ concept id=c5 zh_tag="步子太大 + 那道墙" tag="Steps too big + the wall" zh_title="步子迈过头 —— 以及没有优化器能翻过的那道墙" title="When the step overshoots — and the wall no optimizer climbs" zh_gotit="懂了那道墙" gotit="Got the limit"
Two last things before we gather it all up: the one failure that bites *every* optimizer, and one honest limit that *no* optimizer can fix. Start with the failure, and go back to our marble. Imagine the **ball rolling** into a valley — except now it cannot roll smoothly, it can only *hop* a fixed distance each time, like a kid jumping between paving stones. If the hop is wider than the valley, the ball sails clean over the bottom and lands *higher* on the far wall. Then it hops back, even higher. Up and out, instead of settling. That is a step size (learning rate) set **too large**.
~~~zh
在把所有东西收拢之前还有两件事：那个咬*每一个*优化器的失败，以及一个*没有*优化器能修的诚实限制。先说失败，回到我们的弹珠。想象那颗**滚动的球**滚进一个山谷 —— 只不过现在它不能平滑地滚，只能每次*跳*一段固定的距离，像个在铺路石之间跳的小孩。如果那一跳比山谷还宽，球就干净地飞过谷底，落在对面墙上*更高*的地方。然后它跳回来，更高。往上、往外，而不是安顿下来。那就是步长（learning rate）设得**太大**。
~~~

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="A child hopping between paving stones across a dip. The hop is wider than the dip, so instead of landing at the bottom the hopper lands higher up the far side, and the next hop lands higher still on the near side. Each ball rests on the ground curve. Labels read hop one sails over the bottom and hop two lands higher still."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A hop wider than the dip lands you HIGHER on the far side</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">一跳比坑还宽，会让你落在对面更高的地方</text><path d="M30 60 C 120 78, 190 150, 260 154 C 330 150, 400 78, 490 60" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text class="lang-en" x="260" y="172" text-anchor="middle" fill="#6B645E" font-size="8.5">the bottom you wanted</text><text class="lang-zh" x="260" y="172" text-anchor="middle" fill="#6B645E" font-size="8.5">你想去的谷底</text><circle cx="112" cy="90" r="7" fill="#2A7B9B" stroke="#fff" stroke-width="1.4"/><text class="lang-en" x="112" y="110" text-anchor="middle" fill="#1F6280" font-size="8.5">start</text><text class="lang-zh" x="112" y="110" text-anchor="middle" fill="#1F6280" font-size="8.5">起点</text><path d="M118 84 Q 262 20 412 78" fill="none" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="4,3"/><polygon points="416,81 405,74 409,70" fill="#C93B3B"/><circle cx="420" cy="85" r="7" fill="#C93B3B" stroke="#fff" stroke-width="1.4"/><text class="lang-en" x="262" y="36" text-anchor="middle" fill="#C93B3B" font-size="9">hop 1: sails over the bottom, lands higher</text><text class="lang-zh" x="262" y="36" text-anchor="middle" fill="#C93B3B" font-size="9">第 1 跳：飞过谷底，落得更高</text><path d="M414 92 Q 244 152 68 73" fill="none" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="4,3"/><polygon points="64,69 75,73 71,79" fill="#C93B3B"/><circle cx="60" cy="68" r="7" fill="#C93B3B" stroke="#fff" stroke-width="1.4"/><text class="lang-en" x="250" y="136" text-anchor="middle" fill="#C93B3B" font-size="9">hop 2: lands higher still ✗</text><text class="lang-zh" x="250" y="136" text-anchor="middle" fill="#C93B3B" font-size="9">第 2 跳：落得更高 ✗</text></g></svg>
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
~~~zh
**铺路石那一跳这个比喻对在哪里：** 一跳比坑还宽，就每一次都迈过谷底并爬上对面，于是 loss *往上*走，不是往下。

**它在哪里不成立：** 真的小孩能看见那个坑并把跳缩短。优化器是被蒙着眼的 —— 它只感受它站的地方的倾斜，这正是为什么一个过大的跳可以跳过谷底而完全不知道。

#### 谜题 —— 速率*太大*
我们把你在屏幕上会看到什么、以及为什么，都点清楚：

%%% steps
step: 症状
why: loss **震荡** —— 上、下、更高 —— 然后**发散**并炸掉，常常以 `NaN`（「not a number」，数值跑到无穷时数学给你的东西）收场
step: 原因
why: 步子就是比到谷底的距离更长，因此每一跳都落在对面的墙上而不是坑里
%%%

好消息：这是整个机器学习里最容易修的失败之一。两个动作，第二个有个值得知道的名字 —— [[learning-rate schedule||一个随训练推进缩小 learning rate 的计划 —— 早期大步求快，后期小步求稳；也叫 decay]]，也叫 **decay（衰减）**。

%%% steps
step: 解药 1 —— 把步子缩小
why: 降低 learning rate（除以 3 或 10 是通常的第一招），让那一跳装进山谷里
step: 解药 2 —— 让它*随时间*缩小
why: 一开始大以求快，然后随训练推进缩小 learning rate —— 早期的大跳覆盖地面，后期的小步让球轻轻安顿下来而不是弹跳
%%%

同一次训练画成三种样子 —— 合理的速率、太大的速率、以及被 schedule 救回来的太大速率：
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three loss curves over steps. A green curve with a sensible learning rate slides down and flattens near the bottom. A red curve with too-large a learning rate zig-zags upward and diverges off the top of the chart, labelled overshoots then blows up. A dashed amber curve shows a learning-rate schedule shrinking the steps so the run settles."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Same model, three learning-rate stories</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">同一个模型，三个 learning rate 的故事</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text class="lang-en" x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text class="lang-zh" x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text class="lang-en" x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><text class="lang-zh" x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">步数 →</text><path d="M52 44 C 150 120, 320 144, 494 148" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text class="lang-en" x="360" y="140" fill="#2D8B55" font-size="9">sensible rate ✓ (falls, settles)</text><text class="lang-zh" x="360" y="140" fill="#2D8B55" font-size="9">合理的速率 ✓（下降、安顿）</text><path d="M52 120 L 96 66 L 140 128 L 184 44 L 228 136 L 272 30" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text class="lang-en" x="120" y="30" fill="#C93B3B" font-size="9">too big → overshoots, blows up (NaN)</text><text class="lang-zh" x="120" y="30" fill="#C93B3B" font-size="9">太大 → 迈过头，炸掉（NaN）</text><path d="M272 30 C 320 60, 400 96, 494 110" fill="none" stroke="#C99A12" stroke-width="1.8" stroke-dasharray="5,3"/><text class="lang-en" x="380" y="104" fill="#9A7208" font-size="8.5">schedule shrinks the step → settles</text><text class="lang-zh" x="380" y="104" fill="#9A7208" font-size="8.5">schedule 缩小步子 → 安顿下来</text></g></svg>
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
~~~zh
%%% insight
把那三条曲线再读一遍，因为这是你现在拥有的一个超能力：loss 曲线的*形状*告诉你该改什么。锯齿状而且在爬 = 步子太大。平坦而且几乎不动 = 步子太小。先掉然后走平 = 你调对了。训练你每天在用的那些模型的工程师，整天盯着的就是这张图 —— 而你现在也能读它了。
%%%

想验证一下你真会读了吗？在看之前先给一次训练下判决：

%%% demo id=lrzh label="先猜，再展开"
predict: 一次训练打印出 loss 2.30，然后 41.7，然后 980.4，然后 nan。哪里出错了，你的第一个动作是什么？
code: for step, loss in enumerate(history): print(f'step {step}: loss = {loss}')
out: step 0: loss = 2.30
     step 1: loss = 41.7
     step 2: loss = 980.4
     step 3: loss = nan
take: <b>loss 往上爬然后撞到 NaN，每一次都是 learning rate 太大。</b>第一个动作：把 lr 除以 10 重跑。第二个动作：加一个 schedule 让速率随训练衰减。如果反过来，数字几乎没动（2.30、2.29、2.29……），诊断就翻过来：速率太小 —— 提高它、加 momentum，或者换成自适应优化器。
%%%

到这里为止：你能认出一个糟糕的步长并把它修好。现在说更难的那个真相 —— 就算一个*完美*的步长也买不到的东西。

#### 那道诚实的墙
一个优化器*只是*在别人交给它的地面上往下坡走的一种方式。它没法发明出比地面允许的更好的路。所以有三件事它就是做不到：

%%% steps
step: 它没法修一个糟糕的 loss
why: 如果那个「我错得多厉害」的分数量错了东西，优化器会忠实地滚到那座*错误*的山的谷底 —— 而且一路上看起来完全健康
step: 它没法把一个不可分的问题变成可分的
why: 还记得 XOR —— 一个神经元画一条直线，而没有直线能分开 XOR；把 SGD 换成 Adam 改的是走法，不是模型，所以天花板还在原来的地方
step: 它没法承诺给你所有点里最低的那一个
why: 在一个坑坑洼洼（[[non-convex||一片有不止一个坑的地形，所以最近的最低点可能不是所有点里最低的]]）的地形上，它只感受站着的地方的倾斜，因此它可能安顿在一个浅坑里，永远不知道还有一个更深的 —— 没有优化器保证全局最小值
%%%

共同的那根线：一个优化器**只跟着它被给的那个 gradient 走**。第三条最容易直接*看*出来：
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A bumpy loss landscape with two dips. A ball has come to rest at the lowest point of the shallower left dip, with uphill on both sides, labelled the optimizer stops here. The deeper right dip is lower but unreached, labelled the real best, out of reach. A caption reads the optimizer only follows the slope it is given."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">An optimizer only follows the slope it is given</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">优化器只跟着它被给的斜率走</text><path d="M30 60 C 90 130, 150 120, 190 96 C 230 72, 270 150, 340 152 C 400 154, 440 90, 500 60" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="129" cy="113" r="7" fill="#C99A12" stroke="#fff" stroke-width="1.5"/><text class="lang-en" x="129" y="95" text-anchor="middle" fill="#9A7208" font-size="9">🛑 stops here</text><text class="lang-zh" x="129" y="95" text-anchor="middle" fill="#9A7208" font-size="9">🛑 停在这里</text><text class="lang-en" x="112" y="134" text-anchor="middle" fill="#9A7208" font-size="8.5">shallow dip</text><text class="lang-zh" x="112" y="134" text-anchor="middle" fill="#9A7208" font-size="8.5">浅坑</text><circle cx="340" cy="152" r="6" fill="none" stroke="#2D8B55" stroke-width="2"/><text class="lang-en" x="384" y="152" fill="#1a5c38" font-size="9">← the real best</text><text class="lang-zh" x="384" y="152" fill="#1a5c38" font-size="9">← 真正最好的</text><text class="lang-en" x="330" y="170" text-anchor="middle" fill="#6B645E" font-size="8.5">deeper, out of reach from the left dip</text><text class="lang-zh" x="330" y="170" text-anchor="middle" fill="#6B645E" font-size="8.5">更深，从左边那个坑到不了</text></g></svg>
%%%

**The honest takeaway:** a flat, settled loss curve is *good news*, but it is not a promise you found the lowest possible loss. You found the bottom of whatever bowl your ball happened to roll into. (Cheerful footnote for later: inside the huge networks behind real models these dips are usually shallow and rarely a real problem.)

A better optimizer takes a *smarter path down the same hill*; it never builds a new hill. Knowing that line is what separates someone who uses optimizers from someone who understands them — and you now know it.
~~~zh
**诚实的结论：** 一条平坦、安顿下来的 loss 曲线是*好消息*，但它不是一个「你找到了可能的最低 loss」的承诺。你找到的是你的球正好滚进的那个碗的底。（一个给以后的乐观脚注：在真实模型背后那些巨大的网络里，这些坑通常很浅，很少真成为问题。）

一个更好的优化器是在*同一座山上走一条更聪明的路*；它从不造一座新的山。知道这句话，就是「会用优化器的人」和「理解优化器的人」之间的分界 —— 而你现在知道了。
~~~

@@@ concept id=c6 zh_tag="回顾" tag="Recap" zh_title="一页看完今天" title="The whole day on one page" zh_gotit="回顾好了" gotit="Got the recap"
You just turned one plain downhill step into a small family of smart ones — and every one of them is still just a **ball rolling** down the "how wrong am I" hill, only rolling more cleverly. Here is the whole arc in five beats:

- **The plain step.** Feel the tilt under the ball, slide a little downhill: *new weight = old weight − learning rate × slope.* The learning rate is the step size.
- **Three flavors.** *Batch* asks all the data (accurate, slow), *SGD* asks one example (fast, noisy), *mini-batch* asks a small handful (the everyday sweet spot). We sample because many rough steps beat one perfect step.
- **Momentum.** Keep a running velocity, so a steady downhill *builds up speed* and side-to-side bounces *cancel* — it damps the zig-zag and cures the too-slow crawl.
- **RMSProp & Adam.** Give every weight its own stride, measured from its own recent slopes. **Adam = Momentum + RMSProp**, the default behind most models you have heard of.
- **The failures and the wall.** Too-big rate → overshoots and blows up (remedy: smaller rate, or a *schedule* that decays it). Too-small → crawls (remedy: bigger rate, momentum, or an adaptive optimizer). And the ceiling: an optimizer *only follows the gradient it is handed* — it cannot fix a bad loss, split XOR, or promise the global lowest point.

One picture to keep: same hill, smarter ways down it.
~~~zh
你刚刚把一个朴素的下坡一步，变成了一小家子聪明的步子 —— 而它们每一个仍然只是一颗**滚动的球**在「我错得多厉害」这座山上往下滚，只是滚得更聪明。整条线索，五个拍子：

- **朴素的一步。** 感受球下面的倾斜，往下坡挪一点：*新 weight = 老 weight − learning rate × 斜率。* learning rate 就是步长。
- **三种口味。** *Batch* 问全部数据（准、慢），*SGD* 问一个样本（快、吵），*mini-batch* 问一小把（日常的甜点）。我们抽样，因为很多粗糙的步子胜过一个完美的步子。
- **Momentum。** 保留一个持续的速度，于是稳定的下坡会*攒速度*，左右的弹跳会*抵消* —— 它压住 Z 字形，也治好走得太慢的爬行。
- **RMSProp 和 Adam。** 给每个 weight 自己的步幅，从它自己最近的斜率量出来。**Adam = Momentum + RMSProp**，你听过的大多数模型背后的默认选择。
- **那些失败，和那道墙。** 速率太大 → 迈过头并炸掉（解药：更小的速率，或者一个让它衰减的 *schedule*）。太小 → 爬行（解药：更大的速率、momentum，或者一个自适应优化器）。以及那个天花板：优化器*只跟着它被给的 gradient 走* —— 它没法修一个糟糕的 loss、没法分开 XOR、也没法承诺全局最低点。

一张要留住的图：同一座山，更聪明的下山方式。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Three paths down the same valley, side by side. Plain SGD takes many small even steps and arrives last. Momentum builds up speed and arrives sooner. Adam takes a well-sized step per weight and arrives soonest. A caption reads same hill, smarter steps."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Same hill — smarter steps get down sooner</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">同一座山 —— 更聪明的步子更早到底</text><path d="M30 40 Q 165 150 300 150 Q 400 150 500 46" fill="none" stroke="#B8AEA2" stroke-width="2"/><text class="lang-en" x="300" y="163" text-anchor="middle" fill="#6B645E" font-size="8">bottom</text><text class="lang-zh" x="300" y="163" text-anchor="middle" fill="#6B645E" font-size="8">谷底</text><g fill="#C93B3B"><circle cx="70" cy="70" r="3"/><circle cx="100" cy="90" r="3"/><circle cx="130" cy="106" r="3"/><circle cx="160" cy="120" r="3"/><circle cx="192" cy="132" r="3"/><circle cx="228" cy="142" r="3"/></g><text class="lang-en" x="86" y="60" fill="#C93B3B" font-size="8.5">plain SGD: many small steps</text><text class="lang-zh" x="86" y="60" fill="#C93B3B" font-size="8.5">plain SGD：很多小步</text><g fill="#2A7B9B"><circle cx="84" cy="80" r="3.4"/><circle cx="132" cy="107" r="3.4"/><circle cx="204" cy="136" r="3.4"/><circle cx="268" cy="148" r="3.4"/></g><text class="lang-en" x="150" y="92" fill="#1F6280" font-size="8.5">Momentum: builds speed</text><text class="lang-zh" x="150" y="92" fill="#1F6280" font-size="8.5">Momentum：攒速度</text><g fill="#2D8B55"><circle cx="108" cy="94" r="3.8"/><circle cx="224" cy="141" r="3.8"/><circle cx="292" cy="150" r="3.8"/></g><text class="lang-en" x="250" y="118" fill="#1a5c38" font-size="8.5">Adam: right-sized per-weight step</text><text class="lang-zh" x="250" y="118" fill="#1a5c38" font-size="8.5">Adam：每 weight 大小合适的步子</text></g></svg>
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
~~~zh
还有那张一眼看完的修法表，留给你盯着一条糟糕 loss 曲线的那天：

%%% table
:: 你看到什么 :: 它意味着什么 :: 先试什么
loss 往上爬，然后 NaN :: 步子太大 —— 它迈过了谷底 :: 把 lr 除以 10；然后加一个 decay schedule
loss 几乎不动 :: 步子太小 —— 它在爬 :: 提高速率、加 momentum，或者换成 Adam
loss 走 Z 字形慢慢下降 :: 一条窄沟 —— 在两面墙之间弹跳 :: momentum（如果 weight 尺度失衡就再加自适应速率）
loss 先掉然后走平 :: 它在起作用 —— 你找到了一个谷底 :: 继续走；记住它可能不是最深的那个碗
%%%

把这张小抄放在手边 —— 它把今天每一个词都放在一个地方了：

%%% jargon
optimizer | 把斜率变成 weight 步子的那条规则 —— 球怎么滚
learning rate (lr) | 步长：球每一步走多远
gradient descent | 朴素的一步：新 weight = 老 weight − lr × 斜率
batch gradient descent | 用整个数据集的斜率走一步 —— 准但慢
stochastic gradient descent (SGD) | 用一个样本走一步 —— 快但吵
mini-batch gradient descent | 用一小把（比如 32–64）走一步 —— 日常默认
momentum | 保留一个过去斜率的持续速度，让稳定下坡攒速度、Z 字形抵消
RMSProp | 用每个 weight 自己最近平方斜率的平方根去缩放它的步子
Adam | Momentum + RMSProp：既有攒起来的速度又有每 weight 的步幅 —— 常见默认
learning-rate schedule (decay) | 在训练过程中缩小 learning rate：早期大，后期小
non-convex | 一片有不止一个坑的地形，所以一条安顿下来的 loss 不是最低点的承诺
%%%
~~~

@@@ quiz id=quiz zh_tag="小测" tag="Quiz" zh_title="四道快速检查" title="Four quick checks" zh_gotit="先全部答完" gotit="answer all first"
%%% quiz
q: In "w ← w − lr × slope", what does the learning rate (lr) control? | a:1 | which way is downhill | how big each step is | how many examples you use | the number of weights | fb: The lr is the step size — how far the ball slides each step.
q: What is the everyday default flavor of gradient descent? | a:2 | full-batch (all data per step) | one giant step total | mini-batch (a small handful per step) | no steps at all | fb: Mini-batch — many steps, and the small average keeps each fairly steady.
q: What does Momentum add to the plain step? | a:1 | a bigger dataset | a running velocity (memory of recent slopes) | a second loss | a random restart | fb: A running velocity, so steady downhill builds speed and side-to-side bounces cancel.
q: Adam is best described as… | a:2 | plain SGD with a bigger lr | full-batch only | Momentum + RMSProp (per-weight scaling) together | a way to change the loss | fb: Adam = Momentum's velocity + RMSProp's per-weight stride — the common default.
%%%
~~~zh
四道快问，每题都有即时反馈。

%%% quiz
q: 在「w ← w − lr × slope」里，learning rate（lr）控制什么？ | a:1 | 哪边是下坡 | 每一步有多大 | 你用多少个样本 | weight 的数量 | fb: lr 就是步长 —— 球每一步滑多远。
q: gradient descent 日常默认的那种口味是哪个？ | a:2 | full-batch（每步用全部数据） | 总共只走一大步 | mini-batch（每步一小把） | 完全不走步 | fb: Mini-batch —— 步数多，而那一小把的平均让每一步还算稳。
q: Momentum 给朴素的一步加了什么？ | a:1 | 一个更大的数据集 | 一个持续的速度（最近斜率的记忆） | 第二个 loss | 一次随机重启 | fb: 一个持续的速度，于是稳定下坡攒速度、左右弹跳互相抵消。
q: Adam 最好被描述成…… | a:2 | plain SGD 配一个更大的 lr | 只能 full-batch | Momentum + RMSProp（每 weight 缩放）合在一起 | 一种改 loss 的办法 | fb: Adam = Momentum 的速度 + RMSProp 的每 weight 步幅 —— 常见的默认。
%%%
~~~

@@@ produce id=produce zh_tag="动手做" tag="Produce" zh_title="让三个优化器在同一个山谷里赛跑" title="Race three optimizers down the same valley" zh_gotit="做完了" gotit="Done"
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
~~~zh
是时候在一次运行里*看见*今天的全部：朴素的、带速度的、每 weight 的 —— 每一个都更早到达谷底。你会把三个优化器丢到*同一条*又长又窄的山谷里，数一数各自多快到。

这条山谷是 `loss = 0.5 * (4*wx**2 + 0.2*wy**2)`，所以它的斜率是 `gx = 4*wx`（在 `wx` 方向很陡）和 `gy = 0.2*wy`（在 `wy` 方向平缓）。这正是 Momentum 那一节里的那条沟。

三个都从 `wx = wy = 1.5` 出发，共用**同一个** learning rate `lr = 0.3` 以求公平，跑 `30` 步。Momentum 用和 demo 一样的 `v = keep*v + slope` 规则，`keep = 0.6`。自适应那个把每个 weight 的步子除以「那个 weight 自己平方斜率的滑动平均」的平方根（RMSProp 的把戏）。

跑之前一个诚实的说明：demo 用的是常见的 `keep = 0.9`，这里用更温和的 `0.6`。那是故意的，不是笔误 —— 这条山谷在 `wx` 方向*非常*陡，0.9 会攒出太多速度让球迈过头。试试看，看着 Momentum 跑得比 plain SGD 还*慢*。那是一个真实的失败模式，而自己找到它比被告知更值钱。

**先预测，再运行：** 哪个优化器*最先*到达 `loss < 0.01` —— plain SGD、Momentum，还是自适应那个？而且你觉得它们之间大概差几步？

**你应该看到什么：**
- **Plain SGD 最慢。** 它在平缓的 `wy` 方向上一小步一小步地爬，到 `loss < 0.01` 大约要到**第 26 步**。
- **Momentum 快得多** —— 它在同一个平缓方向上攒速度，大约**第 13 步**就到了。同样的 learning rate、同样的山谷：那个持续的速度就是全部差别，大致把路程砍掉一半。
- **自适应（RMSProp 风格）那个最快** —— 大约**第 4 步**。靠着把每个 weight 缩放到它自己的斜率大小，它从一开始就在*两个*方向上都走一个大小合适的步子，而不是在其中一个方向上爬。
- 读那张打印出来的表，把顺序和你的猜测对一下：**朴素 → 带速度 → 每 weight，一个比一个更早到底。** 那就是今天整个故事，三行输出。

把它写在 `experiment.py` 里然后跑。然后去戳它 —— 脚本不管怎样都会打印你的结果，所以不会弄坏什么：
- 把 Momentum 的 `keep` 设成 `0.0`，看它变回 plain SGD（同样是第 26 步 —— 那*就是*「momentum 是唯一差别」的证明）。
- 把 `keep` 设成 `0.9`，看太多速度在这条陡山谷上*帮了倒忙*。
- 把共用的 `lr` 提到 `0.6`，看 plain SGD 迈过头然后炸掉 —— 上一个概念里那个步子太大的失败，活生生出现在你的终端里（自适应那个和 Momentum 能活下来，这本身也是一课）。
- 把起点挪到 `3.0`，看差距被拉大；自适应那个的快起步会缩小，因为不管地面多陡，它的第一步大小总是差不多。

**路线 A —— 自己写。** 在 `experiment.py` 里填三个循环（一个朴素、一个带速度、一个带每 weight 缩放），并为每一个打印出 `loss < 0.01` 的第一步。

**路线 B —— 交给实验室。** 把下面这段贴进 experiment lab，让它替你搭出 `experiment.py`。（prompt 本身保持英文，它是给工具看的指令。）
~~~

@@@ fin
