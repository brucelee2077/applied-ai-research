---
quest_id: wf3-d05-lr
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 8 — Learning-Rate Intuition"
module_label: "Module 2 · Train · Day 8"
zh_module_label: "第 2 模块 · 训练 · 第 8 天"
title: "Learning-Rate Intuition"
zh_title: "学习率的直觉"
subtitle: "The One Knob That Makes or Breaks Training"
zh_subtitle: "每一步该迈多大"
brand_sub: "Foundations · M2 Day 8"
spine: "stepping stone"
zh_spine: "垫脚石"
nav_prev_href: "../day-07-optimizers/lesson.html"
nav_prev_label: "Optimizers (SGD, Momentum, Adam)"
zh_nav_prev_label: "优化器"
nav_next_href: "../day-09-train-val-test/lesson.html"
nav_next_label: "Train / Val / Test & Overfitting"
zh_nav_next_label: "训练 / 验证 / 测试"
fin_title: "Module 2 · Day 8 complete! 🏆"
zh_fin_title: "第 2 模块 · 第 8 天 完成！🏆"
fin_body: "Nice work — you've built real intuition for the <b>learning rate</b>: too small and training crawls, too big and it overshoots (even to NaN), and the sweet spot in between — plus warmup and decay schedules — is where training actually works.<br>Next up: <b>Train / Val / Test</b> — did the model really learn, or just memorize?"
zh_fin_body: "做得好 —— 你现在能凭手感调**学习率**了：太小就爬，太大就迈过头炸成 NaN，扫一遍幂次找到中间那个，再用一个 schedule 早期大步、后期小步稳稳落地。<br>下一站：<b>训练 / 验证 / 测试</b>。"
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

%%% warmup
q: Yesterday you met Momentum. What does Momentum add to the plain downhill step? | a:1 | a bigger training dataset | a running velocity — a memory of recent slopes | a second loss to minimize | a random restart when stuck | concept: momentum | fb: Momentum keeps a running velocity, so steady downhill builds up speed and side-to-side bounces cancel.
q: Of batch, stochastic (SGD), and mini-batch, which is the everyday default? | a:2 | full-batch — all the data before each step | one giant step over the whole run | mini-batch — a small handful of examples per step | no steps at all, just one look | concept: mini-batch | fb: Mini-batch is the sweet spot: many fast steps, and the small average keeps each one fairly steady.
q: What is Adam, the optimizer people reach for first? | a:3 | plain SGD with a much bigger learning rate | a way to change the loss function | a full-batch-only method | Momentum + RMSProp together — speed plus a per-weight step | concept: adam | fb: Adam combines Momentum's running velocity with RMSProp's per-weight stride, so each weight gets both.
%%%
@zh_lede 想象你踩着**垫脚石**过一条小溪。每次只挪一丁点，你能过去 —— 但慢得要命，一小步一小步地磨。用力跳得太远，你会直接飞过下一块石头，噗通掉进水里。存在一个*刚刚好*的跨度，让你又快又安全地过去。让人意外的地方来了：教一个神经网络就是这个「踩石头过溪」，而每一步的大小有个名字 —— **learning rate（学习率）**。它是整个机器学习里最重要的那一个旋钮。调好了，像 ChatGPT 那样的模型几小时就学会；调错了，训练要么爬一个星期，要么在最前面几步就炸成一堆乱码。今天你可以亲手感受这个旋钮。
@zh_goal 我们会一起建立对学习率真正的手感：它*是*什么（每一步下坡跨多大）、太小会怎样（爬）、太大会怎样（迈过头，甚至炸成 NaN）、怎么靠看 loss 曲线*找到*一个好的，以及一个聪明的花招 —— 让跨度随时间缩小 —— 它让你早期进展快、末尾轻轻落地。每个想法都先给一张图和一个你能拖的滑块。没有重数学 —— 只有你能拿来掌舵的直觉。

~~~zh
%%% warmup
q: 昨天你见了 Momentum。它给朴素的下坡一步加了什么？ | a:1 | 一个更大的训练数据集 | 一个持续的速度 —— 最近斜率的记忆 | 第二个要最小化的 loss | 卡住时随机重启 | concept: momentum | fb: Momentum 保留一个持续的速度，于是稳定下坡攒速度，左右弹跳互相抵消。
q: batch、stochastic（SGD）、mini-batch 里，日常默认是哪个？ | a:2 | full-batch —— 每步之前用全部数据 | 整个训练只走一大步 | mini-batch —— 每步一小把样本 | 完全不走步，只看一眼 | concept: mini-batch | fb: Mini-batch 是那个最佳点：步数多，而那一小把的平均让每一步还算稳。
q: Adam 是什么，为什么人们第一个伸手拿它？ | a:3 | plain SGD 配一个大得多的学习率 | 一种改 loss 函数的办法 | 一个只能 full-batch 的方法 | Momentum + RMSProp 合在一起 —— 速度加上每 weight 的步子 | concept: adam | fb: Adam 把 Momentum 的持续速度和 RMSProp 的每 weight 步幅合起来，所以每个 weight 两样都有。
%%%
~~~

@@@ concept id=c1 zh_tag="它是什么" tag="What it is" zh_title="学习率 —— 每一步下坡跨多大" title="The learning rate — the size of each downhill hop" zh_gotit="懂了它是什么" gotit="Got what it is"
Let us start with the one thing the learning rate actually does. Back on the stream, every **stepping stone** you hop to is a little lower than the last — you are working your way *down* toward the far bank. Two separate things decide your next hop. First, *which way* is downhill — you look at the ground and it tips one way. Second, *how far* you actually hop in that direction. The learning rate is only that second thing: **how far you hop each time.** It does not pick the direction; it just scales how big a move you make once the direction is chosen. In training, "downhill" means toward less [[loss||the model's "how wrong am I" score — training tries to make it smaller]], the direction comes from the [[gradient||the slope of the loss: which way the loss goes up, and how steeply. Training moves the OPPOSITE way, downhill]], and the learning rate is the one number that says how big a hop to take down that slope. In one line: the learning rate is **how big a step you take downhill** on the loss surface each update — it sets how far you move, not which way.
~~~zh
我们先说清楚学习率到底做什么。回到小溪上，你跳到的每一块**垫脚石**都比上一块低一点 —— 你在一路*往下*走向对岸。有两件分开的事决定你下一跳。第一，*哪边*是下坡 —— 你看看地面，它朝一边倾。第二，你在那个方向上实际跳*多远*。学习率只是第二件事：**你每次跳多远。** 它不挑方向；方向定了之后，它只缩放你这一下动多大。在训练里，「下坡」意味着走向更小的 [[loss||模型的「我错得多厉害」分数 —— 训练想让它更小]]，方向来自 [[gradient||loss 的斜率：loss 朝哪边升高、有多陡。训练往*相反*方向、也就是下坡走]]，而学习率就是那一个数字，说明顺着这个坡跨多大一步。一句话：学习率就是**你在 loss 表面上每次更新往下坡走多大一步** —— 它决定你走多远，不决定往哪边。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A person crossing a stream on stepping stones that step down toward the far bank. An arrow labelled direction (from the gradient) points down the slope. A curly brace labelled learning rate marks how far one hop reaches. A caption reads gradient picks the way, learning rate picks how far."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Gradient picks the WAY · learning rate picks HOW FAR</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">gradient 挑*哪边* · learning rate 挑*多远*</text><rect x="20" y="150" width="480" height="40" fill="#DCEBF0"/><text class="lang-en" x="470" y="176" text-anchor="end" fill="#3A7C93" font-size="8.5">stream (the water = high loss)</text><text class="lang-zh" x="470" y="176" text-anchor="end" fill="#3A7C93" font-size="8.5">小溪（水面 = 高 loss）</text><ellipse cx="70" cy="70" rx="26" ry="9" fill="#B8AEA2"/><ellipse cx="180" cy="98" rx="26" ry="9" fill="#B8AEA2"/><ellipse cx="300" cy="126" rx="26" ry="9" fill="#B8AEA2"/><ellipse cx="430" cy="150" rx="26" ry="9" fill="#B8AEA2"/><text class="lang-en" x="70" y="55" text-anchor="middle" fill="#6B645E" font-size="8">high</text><text class="lang-zh" x="70" y="55" text-anchor="middle" fill="#6B645E" font-size="8">高</text><text class="lang-en" x="430" y="135" text-anchor="middle" fill="#6B645E" font-size="8">low (far bank)</text><text class="lang-zh" x="430" y="135" text-anchor="middle" fill="#6B645E" font-size="8">低（对岸）</text><circle cx="70" cy="62" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text class="lang-en" x="70" y="44" text-anchor="middle" fill="#1F6280" font-size="8.5">you are here</text><text class="lang-zh" x="70" y="44" text-anchor="middle" fill="#1F6280" font-size="8.5">你在这里</text><path d="M92 70 L162 96" stroke="#C93B3B" stroke-width="2"/><polygon points="162,96 151,92 152,101" fill="#C93B3B"/><text class="lang-en" x="128" y="112" text-anchor="middle" fill="#C93B3B" font-size="8.5">direction = downhill (gradient)</text><text class="lang-zh" x="128" y="112" text-anchor="middle" fill="#C93B3B" font-size="8.5">方向 = 下坡（gradient）</text><path d="M92 82 q 44 12 90 24" fill="none" stroke="#9A7208" stroke-width="1.4"/><text class="lang-en" x="140" y="132" text-anchor="middle" fill="#9A7208" font-size="8.5">↑ how far you hop = learning rate</text><text class="lang-zh" x="140" y="132" text-anchor="middle" fill="#9A7208" font-size="8.5">↑ 你跨多远 = learning rate</text></g></svg>
%%%

**What the stepping-stone picture gets right:** the direction and the hop size really are two separate choices, and the learning rate is only the hop size — exactly like training. **Where it breaks down:** on a real stream the stones are already placed, so you cannot choose *any* hop length; in training you truly can dial the hop to any size you like — which is exactly why picking it well matters so much.

#### The one-line update rule
Here is the whole move in one plain sentence: **new weight = old weight − learning rate × gradient.** The gradient hands you the direction and steepness; the learning rate scales it; the minus sign turns you to face *downhill*. That is the same update you built in the training loop on earlier days — the learning rate is just the one number multiplying the gradient in it.

Let us build that line one rung at a time, so you can see which part is *yours* to turn:

%%% steps
step: the gradient hands you a direction — and how steep it is
why: it points the way the loss goes UP, so the way *down* is the exact opposite way
step: the minus sign turns you around, to face downhill
why: you want *less* loss, therefore the rule subtracts instead of adds
step: the learning rate turns that direction into an actual distance
why: this is the only place a hop *size* lives — nothing else in the line decides how far you travel
step: new weight = old weight − learning rate × gradient
why: three tiny decisions, one line — and the only one you tune today is the third
%%%

%%% formula
expr: new_weight = old_weight − learning_rate × gradient
note: gradient = the direction (which way is downhill, and how steep). learning_rate = how far you hop. The MINUS sign walks downhill. This is the exact same step that trains your single neuron's weights and bias.
%%%

Let us watch it with tiny real numbers so the sentence turns concrete. Say a weight is `old_weight = 2.0` and the gradient (the slope) is `4.0`. Change *only* the learning rate and watch how far the weight moves in one hop:

%%% demo id=onestep label="run it — one hop with lr = 0.1"
predict: old weight 2.0, gradient 4.0, learning rate 0.1 — how far does the weight move in this one hop: 0.04, 0.4, or 4.0? Guess before you reveal.
code: 2.0 - 0.1 * 4.0        # new_weight = old_weight - learning_rate * gradient
out: 1.6
take: <b>Only the learning rate scales the hop.</b> With `old_weight = 2.0` and `gradient = 4.0`: at lr = 0.1 the hop is 0.1×4.0 = 0.4, so the weight moves to 1.6. Try it in your head at other rates: lr = 0.01 gives a tiny 0.04 hop (→ 1.96, barely nudged), and lr = 0.5 gives a giant 2.0 hop (→ 0.0). Same weight, same gradient — the learning rate alone decides how far each hop reaches. This is the whole knob; the rest of today is what happens when it is too small, too big, or just right.
%%%

%%% insight
Why does one small number get this much attention? Because it is the *same* hop size for every weight in the network, on every single step, for the whole run. Two people can share the same model, the same data and the same code, change only this one number, and end up with "it learned beautifully" versus "it never learned at all". That is why this is the first thing an experienced engineer checks when training misbehaves — and why it is worth the day.
%%%

Here is the same one-hop story as a picture — one weight, one gradient, three learning rates, three hop lengths:
~~~zh
**垫脚石这个比喻对在哪里：** 方向和跨度真的是两个分开的选择，而学习率只是跨度 —— 和训练完全一样。**它在哪里不成立：** 真的小溪里石头已经摆好了，所以你不能选*任意*跨度；而训练里你真的可以把跨度调成任何大小 —— 这正是为什么挑好它这么要紧。

#### 一行的更新规则
整个动作一句大白话：**新 weight = 老 weight − learning rate × gradient。** gradient 给你方向和陡度；learning rate 缩放它；那个减号让你转身朝*下坡*。这和你前几天在训练循环里搭的是同一个更新 —— 学习率只是里面乘着 gradient 的那一个数字。

我们把那一行一级一级搭起来，让你看清哪一部分是*你*要转的：

%%% steps
step: gradient 给你一个方向 —— 以及它有多陡
why: 它指向 loss 升高的那边，所以*下*的方向正好是反的那边
step: 那个减号把你转过来，面朝下坡
why: 你想要*更小*的 loss，因此规则用减而不是加
step: learning rate 把那个方向变成一个实际的距离
why: 这是整行里唯一装着跨度*大小*的地方 —— 别的东西都不决定你走多远
step: 新 weight = 老 weight − learning rate × gradient
why: 三个小决定，一行 —— 而今天你要调的只是第三个
%%%

%%% formula
expr: new_weight = old_weight − learning_rate × gradient
note: gradient = 方向（哪边是下坡、有多陡）。learning_rate = 你跨多远。那个减号让你往下坡走。这和训练你那个单个神经元的 weight 与 bias 是完全同一个步子。
%%%

我们用很小的真实数字看一遍，让这句话变具体。假设一个 weight 是 `old_weight = 2.0`，gradient（梯度）是 `4.0`。*只*改学习率，看这一跳把 weight 挪多远：

%%% demo id=lrhopzh label="先猜，再展开"
predict: 老 weight 2.0，gradient 4.0，learning rate 0.1 —— 这一跳把 weight 挪多远：0.04、0.4，还是 4.0？先猜再看。
code: 2.0 - 0.1 * 4.0        # new_weight = old_weight - learning_rate * gradient
out: 1.6
take: <b>只有学习率在缩放这一跳。</b>`old_weight = 2.0`、`gradient = 4.0`：lr = 0.1 时跨度是 0.1×4.0 = 0.4，所以 weight 挪到 1.6。在心里换别的速率试试：lr = 0.01 只有 0.04 的一小步（→ 1.96，几乎没动），lr = 0.5 有 2.0 的一大跳（→ 0.0）。同一个 weight、同一个 gradient —— 光是学习率就决定了每一跳伸多远。这就是整个旋钮；今天剩下的部分是它太小、太大、和刚刚好时会发生什么。
%%%

%%% insight
为什么这一个小数字值得这么多注意？因为它是网络里*每一个* weight、*每一步*、整个训练过程用的**同一个**跨度。两个人可以共用同一个模型、同一份数据、同一份代码，只改这一个数字。结果一个是「学得漂亮」，另一个是「根本没学」。这就是为什么训练出问题时，有经验的工程师第一个检查的就是它 —— 也是为什么它值一整天。
%%%

同样这个「一跳」的故事画成图 —— 一个 weight、一个 gradient、三个学习率、三种跨度：
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A number line from 0 to 2. A dot starts at weight equals 2.0. Three arrows of different lengths hop left toward zero: a tiny arrow for learning rate 0.01 landing at 1.96, a medium arrow for learning rate 0.1 landing at 1.6, and a long arrow for learning rate 0.5 landing all the way at 0.0. Same weight and gradient, only the hop size changes."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One weight, one gradient — three learning rates, three hop sizes</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">一个 weight、一个 gradient —— 三个学习率，三种跨度</text><line x1="60" y1="120" x2="460" y2="120" stroke="#B8AEA2" stroke-width="2"/><line x1="460" y1="114" x2="460" y2="126" stroke="#6B645E"/><text class="lang-en" x="460" y="140" text-anchor="middle" fill="#6B645E" font-size="8">weight 2.0 (start)</text><text class="lang-zh" x="460" y="140" text-anchor="middle" fill="#6B645E" font-size="8">weight 2.0（起点）</text><line x1="60" y1="114" x2="60" y2="126" stroke="#6B645E"/><text class="lang-en" x="60" y="140" text-anchor="middle" fill="#6B645E" font-size="8">0.0 (bottom)</text><text class="lang-zh" x="60" y="140" text-anchor="middle" fill="#6B645E" font-size="8">0.0（谷底）</text><path d="M456 96 L432 96" stroke="#C99A12" stroke-width="2.2"/><polygon points="432,96 440,92 440,100" fill="#C99A12"/><text x="452" y="86" text-anchor="end" fill="#9A7208" font-size="8">lr 0.01 → 1.96 🐢</text><path d="M456 72 L376 72" stroke="#2D8B55" stroke-width="2.2"/><polygon points="376,72 384,68 384,76" fill="#2D8B55"/><text x="452" y="62" text-anchor="end" fill="#1a5c38" font-size="8">lr 0.1 → 1.6 ✓</text><path d="M456 48 L60 48" stroke="#C93B3B" stroke-width="2.2"/><polygon points="60,48 68,44 68,52" fill="#C93B3B"/><text class="lang-en" x="452" y="38" text-anchor="end" fill="#C93B3B" font-size="8">lr 0.5 → 0.0 (big hop)</text><text class="lang-zh" x="452" y="38" text-anchor="end" fill="#C93B3B" font-size="8">lr 0.5 → 0.0（大跨度）</text></g></svg>
%%%

So the learning rate is one small number that scales every hop. Small stories and big stories follow from that one choice — and the first story is what happens when you make the hops *too small*.
~~~zh
所以学习率是一个缩放每一跳的小数字。小故事和大故事都从这一个选择里长出来 —— 而第一个故事是：把跨度弄得*太小*会怎样。
~~~

@@@ concept id=c2 zh_tag="太小 = 爬" tag="Too small = crawl" zh_title="太小 —— 训练在爬" title="Too small — training crawls" zh_gotit="懂了这个爬" gotit="Got the crawl"
Now the first thing that goes wrong. Imagine crossing that stream by only ever shuffling your foot forward *one centimeter* per hop. You will get to the far bank eventually — nothing bad happens, you never fall in — but it takes *forever*. That is a learning rate set **too small**. Each hop barely moves the weight, so the loss does drop, just achingly slowly. You might need thousands or millions of extra steps to reach the bottom that a bigger hop would have reached quickly.
~~~zh
现在说第一件会出错的事。想象你过那条小溪，每次只把脚往前挪*一厘米*。你最后能到对岸 —— 什么坏事都没发生，你从没掉进水里 —— 但花的时间*长得没边*。那就是学习率设得**太小**。每一跳几乎不挪动 weight，所以 loss 确实在降，只是慢到难受。一个更大的跨度本来很快就到的底，你可能要多走几千甚至几百万步。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A big U-shaped valley with a ball high on the left slope taking many tiny closely-spaced steps that have barely moved it downward. A label reads tiny hops, barely moving, and another reads still far from the bottom after many steps."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Too small: many tiny hops, barely moving</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">太小：很多小跨度，几乎不动</text><path d="M40 40 Q 160 176 260 162 Q 360 176 480 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text class="lang-en" x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="8.5" dy="12">bottom = smallest loss</text><text class="lang-zh" x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="8.5" dy="12">谷底 = 最小的 loss</text><g fill="#C99A12"><circle cx="86" cy="78" r="3"/><circle cx="92" cy="82" r="3"/><circle cx="98" cy="86" r="3"/><circle cx="104" cy="90" r="3"/><circle cx="110" cy="94" r="3"/><circle cx="116" cy="98" r="3"/></g><circle cx="116" cy="98" r="7" fill="#C99A12" stroke="#fff" stroke-width="1.4"/><text class="lang-en" x="150" y="72" fill="#9A7208" font-size="9">tiny hops 🐢</text><text class="lang-zh" x="150" y="72" fill="#9A7208" font-size="9">小跨度 🐢</text><text class="lang-en" x="300" y="96" text-anchor="middle" fill="#6B645E" font-size="8.5">still far from the bottom after many steps</text><text class="lang-zh" x="300" y="96" text-anchor="middle" fill="#6B645E" font-size="8.5">走了很多步之后还离谷底很远</text></g></svg>
%%%

**What the shuffle picture gets right:** small hops are *safe* — you always head the right way and never overshoot — they are just painfully slow. **Where it breaks down:** on the stream you get tired; a computer does not get tired, it just wastes real time and money taking millions of near-useless steps.

#### Watch it crawl
Let us make "crawls" something you can *see*. Here is a tiny valley — loss `= weight²`, so the bottom is at `weight = 0` — started at `weight = 2.0`. With a too-small learning rate of `0.01`, one step moves the weight only a hair. **Predict:** will one step get anywhere near the bottom? Then run it:

%%% demo id=crawl label="run it — one step at a too-small lr = 0.01"
predict: write your guess for the new weight down first — does it still start with 1.9…, or does it get down near 1.0?
code: 2.0 - 0.01 * (2 * 2.0)      # one step downhill on loss = weight**2
out: 1.96
take: <b>One step crept from 2.00 to 1.96</b> — the loss barely dipped from 4.0 to 3.84. Five steps only reach ~1.81; it would take well over a hundred steps to reach the bottom at 0. Nothing is broken — it is just crawling. The fix is simple: raise the learning rate.
%%%

And it does not get better on its own — watch the crawl rung by rung:

%%% steps
step: step 1 — 2.00 − 0.01 × (2 × 2.00) = 1.96
why: the hop is 0.04 wide and the gap to the bottom is 2.0, so this hop covers one fiftieth of the trip
step: step 2 — 1.96 − 0.01 × (2 × 1.96) = 1.92
why: the slope shrinks as the weight shrinks, which means each hop is even *smaller* than the one before it
step: step 5 — still about 1.81, loss only down from 4.00 to 3.27
why: five hops moved you less than a tenth of the way — and that is with everything working correctly
step: about 260 steps — finally within 0.01 of the bottom
why: therefore the whole cost of this failure is time; a healthier rate makes the same trip in a handful of steps
%%%

%%% insight
Here is the part that bites: a crawl does not *look* like a mistake. No crash, no error message, no `NaN` — the loss really is going down, and the graph really does slope the right way. So you wait. You leave it running overnight, come back, and it has learned almost nothing. This failure costs you time and money rather than correctness, which is exactly why it is the easy one to miss.
%%%

Now compare the crawl step-by-step against a healthy drop, so you can *see* how little ground a too-small rate covers. Each bar is the weight after one more step, starting at 2.0 and heading for the bottom at 0:
~~~zh
**挪脚这个比喻对在哪里：** 小跨度是*安全*的 —— 你总是朝对的方向走、从不迈过头 —— 只是慢得难受。**它在哪里不成立：** 在小溪上你会累；计算机不会累，它只是白白烧掉真实的时间和钱，走几百万步几乎没用的步子。

#### 看着它爬
我们把「爬」变成你能*看见*的东西。这是一个很小的山谷 —— loss `= weight²`，所以谷底在 `weight = 0` —— 从 `weight = 2.0` 出发。用一个太小的学习率 `0.01`，一步只把 weight 挪一根头发。**先预测：** 一步能到接近谷底吗？然后跑：

%%% demo id=crawlzh label="先猜，再展开"
predict: 先把你猜的新 weight 写下来 —— 它还是 1.9 开头，还是掉到 1.0 附近了？
code: 2.0 - 0.01 * (2 * 2.0)      # 在 loss = weight**2 上走一步下坡
out: 1.96
take: <b>一步从 2.00 蠕动到 1.96</b> —— loss 只从 4.0 掉到 3.84。五步也只到大约 1.81；要到谷底 0 得走一百多步。什么都没坏 —— 它只是在爬。解法很简单：把学习率提上去。
%%%

而且它不会自己好起来 —— 一级一级看这个爬：

%%% steps
step: 第 1 步 —— 2.00 − 0.01 × (2 × 2.00) = 1.96
why: 跨度是 0.04 宽，而到谷底还有 2.0，所以这一跳走了全程的五十分之一
step: 第 2 步 —— 1.96 − 0.01 × (2 × 1.96) = 1.92
why: weight 变小，斜率也变小，也就是说每一跳都比前一跳*更小*
step: 第 5 步 —— 还在 1.81 左右，loss 只从 4.00 降到 3.27
why: 五跳走了不到全程的十分之一 —— 而这还是一切都正常工作的情况
step: 大约 260 步 —— 终于进到谷底 0.01 以内
why: 因此这个失败的全部代价是时间；一个健康的速率走同样一趟只要几步
%%%

%%% insight
真正咬人的地方在这里：一次爬行*看起来*不像个错误。不崩溃、不报错、没有 `NaN` —— loss 真的在往下走，图真的朝对的方向倾。所以你就等。你让它跑一整夜，回来一看，它几乎什么都没学到。这个失败花的是你的时间和钱，而不是正确性，这正是它容易被漏掉的原因。
%%%

现在把这个爬和一次健康的下降一步一步对比，让你*看见*一个太小的速率覆盖了多么少的地面。每一根条形是又走一步之后的 weight，从 2.0 出发，朝谷底 0 去：
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Two rows of bars showing the weight after each of five steps, starting at 2.0 and heading toward 0. The top gold row, too small at learning rate 0.01, barely shrinks: 2.0, 1.96, 1.92, 1.88, 1.84, 1.81. The bottom green row, a good rate at learning rate 0.4, plunges: 2.0, 0.4, 0.08, 0.016, and essentially 0. A caption contrasts barely moving with nearly there."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Weight after each step (start 2.0 → bottom 0)</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">每一步之后的 weight（起点 2.0 → 谷底 0）</text><text x="16" y="42" fill="#9A7208" font-size="9">🐢 lr 0.01</text><g fill="#C99A12"><rect x="80" y="34" width="118" height="12"/><rect x="80" y="50" width="116" height="12"/><rect x="80" y="66" width="113" height="12"/><rect x="80" y="82" width="111" height="12"/><rect x="80" y="98" width="109" height="12"/></g><text class="lang-en" x="210" y="70" fill="#9A7208" font-size="8.5">…still ≈1.81 after 5 steps (barely moving)</text><text class="lang-zh" x="210" y="70" fill="#9A7208" font-size="8.5">…5 步之后还在 ≈1.81（几乎没动）</text><text x="16" y="140" fill="#1a5c38" font-size="9">✓ lr 0.4</text><g fill="#2D8B55"><rect x="80" y="132" width="118" height="12"/><rect x="80" y="148" width="24" height="12"/><rect x="80" y="164" width="6" height="12"/></g><text class="lang-en" x="210" y="156" fill="#1a5c38" font-size="8.5">≈0 after 3 steps (nearly there)</text><text class="lang-zh" x="210" y="156" fill="#1a5c38" font-size="8.5">3 步之后 ≈0（快到了）</text><line x1="80" y1="28" x2="80" y2="182" stroke="#B8AEA2" stroke-dasharray="2,2"/><text class="lang-en" x="80" y="196" text-anchor="middle" fill="#6B645E" font-size="8">bottom = 0</text><text class="lang-zh" x="80" y="196" text-anchor="middle" fill="#6B645E" font-size="8">谷底 = 0</text><text class="lang-en" x="198" y="196" text-anchor="middle" fill="#6B645E" font-size="8">start = 2.0</text><text class="lang-zh" x="198" y="196" text-anchor="middle" fill="#6B645E" font-size="8">起点 = 2.0</text></g></svg>
%%%

And the *shape* of the whole loss curve gives it away at a glance. A too-small rate makes a curve that slopes down so gently it looks almost **flat**:
~~~zh
而整条 loss 曲线的*形状*一眼就露馅了。一个太小的速率画出来的曲线，倾斜得如此轻微，看起来几乎是**平的**：
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two loss-versus-step curves. A gold curve labelled too small slopes down very gently and stays high, almost flat. A green curve labelled good rate drops steeply and flattens near the bottom. The x-axis is steps, the y-axis is loss."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The loss curve of a too-small rate looks almost FLAT</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">一个太小的速率，loss 曲线看起来几乎是平的</text><line x1="48" y1="30" x2="48" y2="152" stroke="#B8AEA2"/><line x1="48" y1="152" x2="500" y2="152" stroke="#B8AEA2"/><text class="lang-en" x="22" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 22 94)">loss</text><text class="lang-zh" x="22" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 22 94)">loss</text><text class="lang-en" x="272" y="174" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><text class="lang-zh" x="272" y="174" text-anchor="middle" fill="#6B645E" font-size="9">步数 →</text><path d="M54 46 C 200 60, 360 74, 496 86" fill="none" stroke="#C99A12" stroke-width="2.4"/><text class="lang-en" x="360" y="70" fill="#9A7208" font-size="9">too small → barely drops (looks flat) 🐢</text><text class="lang-zh" x="360" y="70" fill="#9A7208" font-size="9">太小 → 几乎不降（看起来平）🐢</text><path d="M54 46 C 130 130, 300 148, 496 150" fill="none" stroke="#2D8B55" stroke-width="2.4"/><text class="lang-en" x="300" y="134" fill="#2D8B55" font-size="9">good rate → drops fast, then settles ✓</text><text class="lang-zh" x="300" y="134" fill="#2D8B55" font-size="9">好速率 → 快速下降，然后安顿 ✓</text></g></svg>
%%%

So: **too small → the loss barely drops, the curve looks flat. Cause:** each hop is a grain of sand. **Remedy:** increase the learning rate. But raise it too far and you meet the *opposite* problem — which is where it gets exciting.
~~~zh
所以：**太小 → loss 几乎不降，曲线看起来是平的。原因：** 每一跳都是一粒沙子。**解药：** 提高学习率。但提得太过，你就会遇到*反过来*的问题 —— 那就有意思了。
~~~

@@@ concept id=c3 zh_tag="太大 = 迈过头" tag="Too big = overshoot" zh_title="太大 —— 迈过头、弹跳、然后炸掉" title="Too big — overshoot, bounce, and blow up" zh_gotit="懂了迈过头" gotit="Got the overshoot"
Now the opposite failure, and it is the dramatic one. Back on the stream, imagine hopping so hard that you sail clean *over* the next **stepping stone** and land in the water past it — then you leap back the other way even harder and overshoot again, splashing back and forth and getting *wetter* each time instead of crossing. That is a learning rate set **too big**. Each hop is longer than the distance to the bottom of the valley, so you jump *past* the lowest point and land higher up the far wall. The loss, instead of going down, bounces around — and if the hops keep growing, it shoots up to infinity. The word for a number that has grown past infinity in the computer is [[NaN||"not a number" — what a computer prints when a calculation blew up past any real value, like infinity times infinity. Seeing NaN in your loss means training exploded]], and a loss of `NaN` is the classic sign of a learning rate set way too high.
~~~zh
现在说反过来那个失败，而它是戏剧性的那个。回到小溪上，想象你跳得那么用力，干净地飞*过*了下一块**垫脚石**，落在它后面的水里 —— 然后你更用力地往回跳。又跳过头，来回噗通，一次比一次*更湿*，而不是过河。那就是学习率设得**太大**。每一跳都比到谷底的距离更长，所以你跳*过*了最低点，落在对面墙上更高的地方。loss 不往下走，反而在乱弹 —— 而如果跨度一直变大，它会冲到无穷。计算机里一个长过无穷的数字有个词叫 [[NaN||「not a number」—— 一个计算炸过任何实数值时计算机打印的东西，比如无穷乘无穷。在你的 loss 里看到 NaN 就意味着训练爆了]]，而一个 `NaN` 的 loss 是学习率设得远远太高的经典标志。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A U-shaped valley. A ball starts on the left slope and takes huge hops that overshoot the bottom, landing higher on the far wall, then bounces back even higher on the left, the arrows growing longer each time, escaping upward out of the valley. A label reads jumps PAST the bottom, and another reads bounces higher each time then blows up."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Too big: each hop jumps PAST the bottom and bounces higher</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">太大：每一跳都跳*过*谷底，弹得更高</text><path d="M40 34 Q 160 180 260 168 Q 360 180 480 34" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><text class="lang-en" x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5" dy="12">bottom</text><text class="lang-zh" x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5" dy="12">谷底</text><path d="M120 118 Q 200 60 336 120" fill="none" stroke="#C93B3B" stroke-width="1.8"/><polygon points="336,120 325,113 330,124" fill="#C93B3B"/><path d="M336 120 Q 240 40 92 82" fill="none" stroke="#C93B3B" stroke-width="1.8"/><polygon points="92,82 103,80 100,90" fill="#C93B3B"/><path d="M92 82 Q 240 16 470 54" fill="none" stroke="#C93B3B" stroke-width="1.8"/><polygon points="470,54 459,52 463,62" fill="#C93B3B"/><circle cx="120" cy="118" r="7" fill="#C93B3B" stroke="#fff" stroke-width="1.4"/><text class="lang-en" x="230" y="58" text-anchor="middle" fill="#C93B3B" font-size="9">bounces higher each time → blows up (NaN) 💥</text><text class="lang-zh" x="230" y="58" text-anchor="middle" fill="#C93B3B" font-size="9">每次弹得更高 → 炸掉（NaN）💥</text></g></svg>
%%%

%%% hint
t1: Do not worry about the exact numbers yet. Just ask one thing: after a hop, is the ball closer to the bottom (0) or farther away than it started? If a hop takes you farther, the next hop will be even bigger — that is the whole runaway.
t2: Try the tiny valley step by hand. Start at weight 2.0 with a too-big rate of 1.1: new weight = 2.0 − 1.1 × (2 × 2.0) = 2.0 − 4.4 = −2.4. You started 2.0 from the bottom and landed 2.4 away — farther. Now do it again from −2.4 and you get about +2.9, farther still. Each hop is bigger than the last.
t3: A rate that is too big makes each hop longer than the distance to the bottom, so you overshoot, land farther out, and every hop grows until the numbers explode to NaN — the fix is simply to lower the rate.
%%%

**What the over-leaping picture gets right:** a hop longer than the valley overshoots the bottom every single time, and repeated overshoots grow instead of settling — that really is how divergence happens. **Where it breaks down:** a real person would notice they are soaked and *stop*; the computer keeps blindly applying the rule until the numbers overflow into NaN.

#### Watch it explode
Same tiny valley as before — loss `= weight²`, bottom at `weight = 0`, start at `weight = 2.0`. Now crank the learning rate up to `1.1` (too big). **Predict:** does the weight get closer to 0, or fling itself *past* the bottom? Then run one step:

%%% demo id=diverge label="run it — one step at a too-big lr = 1.1"
predict: the hop measures 1.1 × 4.0 = 4.4, and the bottom is only 2.0 away — so guess the SIGN of the new weight before you reveal it.
code: 2.0 - 1.1 * (2 * 2.0)      # one step downhill, lr way too big
out: -2.4
take: <b>The weight flew clean past the bottom.</b> It started at +2.0, aimed downhill toward 0, and overshot all the way to −2.4 — now *farther* from the bottom than it started (loss went 4.0 → 5.76, up not down). The next step flings it to +2.9, then −3.5, then +4.1… bouncing bigger and bigger until it overflows to <b>NaN</b>. This is <b>divergence</b> — and the fix is simply to lower the learning rate.
%%%

Why must it *grow*? Walk the runaway one rung at a time — the trap is in rung three:

%%% steps
step: compare the hop to the trip — hop = 1.1 × 4.0 = 4.4, distance to the bottom = 2.0
why: the hop is more than twice as long as the trip, therefore landing *on* the bottom is not even possible
step: so you overshoot: 2.0 − 4.4 = −2.4, past the bottom and up the far wall
why: you are now 2.4 from the bottom where you began 2.0 away, which means the loss went UP: 4.00 → 5.76
step: farther out means a steeper slope, and a steeper slope means a longer hop
why: this is the trap — the overshoot itself makes the *next* overshoot bigger, with no outside help
step: every bounce multiplies your distance by 1.2 — 2.4 → 2.88 → 3.46 → 4.15 → 4.98 …
why: nothing in the rule ever stops it, so the numbers keep doubling up until they overflow into `NaN`
%%%

%%% insight
This is the failure you will actually meet in person, and being able to name it in one second is a real skill. A loss that prints `nan` almost always means one thing: the steps were too big. People lose whole afternoons hunting for a bug in their data or their model when the fix was to divide the learning rate by ten. You now know the first thing to try — that alone is worth today.
%%%

Here is the whole runaway as a picture — each bar is the weight after one more step, flipping sign and growing every time instead of shrinking toward the bottom:
~~~zh
%%% hint
t1: 先别管确切数字。只问一件事：跳完一下之后，球离谷底（0）更近了还是比出发时更远？如果一跳让你更远，下一跳就会更大 —— 这就是整个失控。
t2: 用那个小山谷手算一步。从 weight 2.0 出发，用太大的速率 1.1：新 weight = 2.0 − 1.1 × (2 × 2.0) = 2.0 − 4.4 = −2.4。你出发时离谷底 2.0，落下来离 2.4 —— 更远了。再从 −2.4 做一次，你会得到大约 +2.9，更远。每一跳都比上一跳大。
t3: 一个太大的速率让每一跳都比到谷底的距离更长，所以你迈过头、落得更远，然后每一跳都变大，直到数字炸成 NaN —— 解法就是把速率降下来。
%%%

**跳过头这个比喻对在哪里：** 一跳比山谷还长，就每一次都迈过谷底，而反复迈过头会越来越大而不是安顿下来 —— 发散真的就是这样发生的。**它在哪里不成立：** 真人会发现自己湿透了然后*停下*；计算机会一直盲目地套用那条规则，直到数字溢出成 NaN。

#### 看着它炸
和刚才同一个小山谷 —— loss `= weight²`，谷底在 `weight = 0`，从 `weight = 2.0` 出发。现在把学习率拉到 `1.1`（太大）。**先预测：** weight 会更靠近 0，还是把自己甩*过*谷底？然后跑一步：

%%% demo id=boomzh label="先猜，再展开"
predict: 这一跳量出来是 1.1 × 4.0 = 4.4，而谷底只有 2.0 远 —— 所以在展开之前先猜新 weight 的正负号。
code: 2.0 - 1.1 * (2 * 2.0)      # 走一步下坡，lr 大得离谱
out: -2.4
take: <b>weight 干净地飞过了谷底。</b>它从 +2.0 出发，朝 0 的下坡走，结果一路迈到了 −2.4 —— 现在比出发时*更远*离谷底（loss 从 4.0 到 5.76，是升不是降）。下一步把它甩到 +2.9，然后 −3.5，然后 +4.1…… 越弹越大，直到溢出成 <b>NaN</b>。这就是 <b>divergence（发散）</b> —— 而解法就是把学习率降下来。
%%%

它为什么*必须*变大？把这个失控一级一级走一遍 —— 陷阱在第三级：

%%% steps
step: 把跨度和路程比一比 —— 跨度 = 1.1 × 4.0 = 4.4，到谷底的距离 = 2.0
why: 跨度比路程长了一倍多，因此正好落*在*谷底根本不可能
step: 于是你迈过头：2.0 − 4.4 = −2.4，过了谷底、上了对面的墙
why: 你现在离谷底 2.4，而出发时是 2.0，也就是说 loss 升了：4.00 → 5.76
step: 更远意味着更陡的斜率，而更陡的斜率意味着更长的一跳
why: 这就是陷阱 —— 迈过头这件事本身让*下一次*迈得更过，不需要外面帮忙
step: 每一次弹跳都把你的距离乘 1.2 —— 2.4 → 2.88 → 3.46 → 4.15 → 4.98 …
why: 规则里没有任何东西会拦住它，所以数字一路翻倍上去，直到溢出成 `NaN`
%%%

%%% insight
这是你真的会亲身遇到的那个失败，而能在一秒钟里叫出它的名字是一项真本事。一个打印出 `nan` 的 loss 几乎总意味着一件事：步子太大了。人们会花掉整个下午在数据或模型里找 bug，而解法是把学习率除以十。你现在知道第一个要试的动作 —— 光这一点就值今天。
%%%

整个失控画成图 —— 每一根条形是又走一步之后的 weight，每次都翻符号而且变大，而不是朝谷底缩小：
~~~

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="A bar chart of the weight after each of six steps at a too-big learning rate. The bars alternate left and right of a center line at zero and grow taller each step: plus 2.0, minus 2.4, plus 2.9, minus 3.5, plus 4.1, minus 5.0. A label reads flips sign and grows every step, then blows up to NaN."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Weight after each step at lr = 1.1 (too big)</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">lr = 1.1（太大）时每一步之后的 weight</text><line x1="260" y1="28" x2="260" y2="176" stroke="#B8AEA2" stroke-dasharray="2,2"/><text class="lang-en" x="260" y="190" text-anchor="middle" fill="#6B645E" font-size="8">bottom = 0</text><text class="lang-zh" x="260" y="190" text-anchor="middle" fill="#6B645E" font-size="8">谷底 = 0</text><g fill="#C93B3B"><rect x="200" y="34" width="60" height="14"/><rect x="260" y="54" width="72" height="14"/><rect x="173" y="74" width="87" height="14"/><rect x="260" y="94" width="104" height="14"/><rect x="136" y="114" width="124" height="14"/><rect x="260" y="134" width="149" height="14"/></g><text x="340" y="46" fill="#C93B3B" font-size="8">+2.0</text><text x="340" y="66" fill="#C93B3B" font-size="8">−2.4</text><text x="140" y="86" fill="#C93B3B" font-size="8" text-anchor="end">+2.9</text><text x="372" y="106" fill="#C93B3B" font-size="8">−3.5</text><text x="128" y="126" fill="#C93B3B" font-size="8" text-anchor="end">+4.1</text><text class="lang-en" x="416" y="146" fill="#C93B3B" font-size="8">−5.0 → 💥 NaN</text><text class="lang-zh" x="416" y="146" fill="#C93B3B" font-size="8">−5.0 → 💥 NaN</text></g></svg>
%%%

#### The whole picture — one knob, three outcomes
Now you can *feel* both failures and the sweet spot between them in one place. This is a real valley (loss `= weight²`) with a **learning rate slider**. Drag it *small* and watch the ball inch down in tiny crawling steps. Drag it *big* and watch each step overshoot the bottom and climb the far wall. Somewhere in between it slides down and settles. **Predict** what happens at each end *before* you drag — then check yourself:
~~~zh
#### 整张图 —— 一个旋钮，三种结果
现在你可以在一个地方*感受*两个失败和它们中间那个最佳点。这是一个真实的山谷（loss `= weight²`），带一个 **learning rate 滑块**。往*小*拖，看球一小步一小步地爬下去。往*大*拖，看每一步都迈过谷底、爬上对面的墙。在中间某处它会滑下去并安顿。在你拖之前**先预测**两端会发生什么 —— 然后检查你自己：
~~~

%%% svg
<svg id="gd-svg" viewBox="0 0 520 214" role="img" aria-label="Interactive gradient descent. A U-shaped loss valley with a ball on the left slope. Drag the learning-rate slider: a small rate makes the ball inch down in tiny steps, a good rate settles it at the bottom, and a too-big rate makes each step overshoot and bounce higher up the far wall."><g font-family="monospace" font-size="10"><rect x="40" y="26" width="440" height="160" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><path id="gd-bowl" d="M60 40 L260 180 L460 40" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><polyline id="gd-traj" points="" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="3,3" opacity="0.8"/><circle id="gd-b0" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b1" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b2" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b3" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b4" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b5" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b6" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b7" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b8" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-b9" cx="0" cy="0" r="4" fill="#C99A12" opacity="0"/><circle id="gd-ball" cx="80" cy="120" r="8" fill="#2A7B9B" stroke="#fff" stroke-width="1.5"/><text class="lang-en" x="260" y="200" text-anchor="middle" fill="#9A938A" font-size="9">bottom = smallest loss</text><text class="lang-zh" x="260" y="200" text-anchor="middle" fill="#9A938A" font-size="9">谷底 = 最小的 loss</text></g></svg>
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
~~~zh
所以 **太大 → loss 迈过头、弹跳，还可能炸成 NaN。原因：** 每一跳都比到谷底的距离更长。**解药：** 降低学习率（最疯的情况下还可以*裁剪* gradient，让任何单独一跳都不能太大）。你现在见过两面墙了 —— 太小会爬，太大会炸。接下来：你到底怎么*找到*中间那个好速率？
~~~

@@@ concept id=c4 zh_tag="怎么找一个好的" tag="Finding a good one" zh_title="找一个好速率 —— 扫一遍，读曲线" title="Finding a good rate — sweep and read the curve" zh_gotit="懂了怎么找" gotit="Got the search"
So there is a crawling wall on one side and an exploding wall on the other, with a good rate somewhere between. How do you find it? You do *not* solve an equation — you *try a few and look.* Think of tuning an old radio dial to find a station. You do not know the exact number, so you sweep the dial and listen: too far one way is silence, too far the other is static, and somewhere in between the music comes in clear. Finding a learning rate is that dial-sweep. You try a handful of rates spread far apart, run a little training with each, and *watch the loss curve* — its shape tells you instantly which regime you are in.
~~~zh
所以一边是爬行的墙，另一边是爆炸的墙，好速率在中间某处。怎么找？你*不*去解方程 —— 你*试几个，然后看*。想象你在调一台老收音机的旋钮找电台。你不知道确切的数字，所以你扫过旋钮然后听：往一边太多是安静，往另一边太多是杂音，中间某处音乐清清楚楚出来了。找学习率就是这个扫旋钮。你试一小把彼此隔得很开的速率，各训练一小段，然后*看 loss 曲线* —— 它的形状立刻告诉你你在哪个区间。
~~~

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A radio dial marked with learning-rate values 0.0001, 0.001, 0.01, 0.1, 1.0 spread across it. The low end is labelled silence (too small, crawls), the high end is labelled static (too big, blows up), and the middle mark 0.1 glows and is labelled clear signal (good rate). A caption reads sweep the dial and listen to the loss curve."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Sweep the dial — the loss curve tells you where you are</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">扫一遍旋钮 —— loss 曲线告诉你你在哪里</text><line x1="60" y1="96" x2="460" y2="96" stroke="#B8AEA2" stroke-width="3"/><g text-anchor="middle"><line x1="80" y1="88" x2="80" y2="104" stroke="#6B645E"/><text x="80" y="122" fill="#6B645E" font-size="8">0.0001</text><line x1="170" y1="88" x2="170" y2="104" stroke="#6B645E"/><text x="170" y="122" fill="#6B645E" font-size="8">0.001</text><line x1="260" y1="88" x2="260" y2="104" stroke="#6B645E"/><text x="260" y="122" fill="#6B645E" font-size="8">0.01</text><line x1="350" y1="86" x2="350" y2="106" stroke="#2D8B55" stroke-width="2"/><text x="350" y="122" fill="#1a5c38" font-size="8.5">0.1</text><line x1="440" y1="88" x2="440" y2="104" stroke="#6B645E"/><text x="440" y="122" fill="#6B645E" font-size="8">1.0</text></g><circle cx="350" cy="96" r="10" fill="#2D8B55" opacity="0.25"/><circle cx="350" cy="96" r="5" fill="#2D8B55"/><text class="lang-en" x="100" y="150" text-anchor="middle" fill="#9A7208" font-size="8.5">🔇 silence: too small, crawls</text><text class="lang-zh" x="100" y="150" text-anchor="middle" fill="#9A7208" font-size="8.5">🔇 安静：太小，在爬</text><text class="lang-en" x="350" y="150" text-anchor="middle" fill="#1a5c38" font-size="8.5">🎵 clear: good rate</text><text class="lang-zh" x="350" y="150" text-anchor="middle" fill="#1a5c38" font-size="8.5">🎵 清楚：好速率</text><text class="lang-en" x="440" y="150" text-anchor="middle" fill="#C93B3B" font-size="8.5">📡 static</text><text class="lang-zh" x="440" y="150" text-anchor="middle" fill="#C93B3B" font-size="8.5">📡 杂音</text></g></svg>
%%%

**What the radio-dial picture gets right:** you *sweep* across widely-spaced settings and *listen* (watch the loss) rather than calculating the perfect number — that is exactly the real workflow. **Where it breaks down:** a radio has one true station; a network can have a whole *range* of decent learning rates, and the best one can even drift as training goes on (that is the schedule trick coming next).

#### The order-of-magnitude sweep
Because good rates can be anywhere from tiny to almost one, you do not try `0.011, 0.012, 0.013…` — that would take forever and cover almost no ground. Instead you jump by *powers of ten*: **1.0, 0.1, 0.01, 0.001, 0.0001.** Each is ten times smaller than the last, so five tries cover an enormous range. This is the practical search almost everyone starts with.

Here is the whole sweep as four rungs you could run this afternoon:

%%% steps
step: pick rates a power of ten apart — 1.0, 0.1, 0.01, 0.001, 0.0001
why: a workable rate can live anywhere from tiny to nearly one, so jumping by tens covers that whole range in five tries
step: run a SHORT training with each — you do not need to finish any of them
why: a bad rate shows its shape in the first handful of steps, which means the sweep costs you almost nothing
step: plot loss-versus-step for each run and look at the SHAPE, not just the final number
why: the shape names the regime — flat means too small, spiky or rising means too big
step: keep the rate whose curve drops smoothly, then narrow around it if you want more
why: therefore you never solved an equation; you looked at pictures and picked the best one
%%%

Here is the winner from that sweep — the final loss after 40 steps at `lr = 0.1`. **Predict:** near the bottom (≈0), or still stuck up high? Then run it:

%%% demo id=sweep label="run it — final loss after 40 steps at lr = 0.1"
predict: at lr = 0.1 each step shrinks the weight to 0.8 of what it was — after 40 of those, is the final loss nearer 1, nearer 0.1, or almost nothing at all?
code: train(lr=0.1)      # loss = weight**2 after 40 downhill steps from weight = 2.0
out: 7.1e-08
take: <b>lr = 0.1 lands essentially at the bottom</b> — final loss ≈ 0.00000007, basically 0. Run the whole sweep and the winner jumps right out: lr = 1.0 never settles (stuck bouncing at loss 4), lr = 0.1 nails it (≈0), lr = 0.01 crawls only partway (≈0.79), and lr = 0.001 (≈3.4) and 0.0001 (≈3.9) barely budge off the starting loss of 4. No equation needed — the sweep points straight at 0.1.
%%%

Here is the whole sweep as a picture — final loss after 40 steps for each rate. See how only the middle rate reaches the floor, while both ends stay stuck high:
~~~zh
**收音机旋钮这个比喻对在哪里：** 你在间隔很大的设置之间*扫*，然后*听*（看 loss），而不是去算那个完美的数字 —— 这正是真实的工作流。**它在哪里不成立：** 收音机只有一个真的电台；而一个网络可以有一整*段*不错的学习率，而且最好的那个还会随训练推进漂移（那就是接下来那个 schedule 花招）。

#### 按数量级扫一遍
因为好速率可以在从极小到接近一的任何地方，你不会去试 `0.011、0.012、0.013……` —— 那会花掉很久而且几乎没覆盖到什么。你按*十的幂次*跳：**1.0、0.1、0.01、0.001、0.0001。** 每一个是前一个的十分之一，所以五次尝试覆盖了一个巨大的范围。这是几乎所有人都从这里开始的实用搜索。

整个扫法四级，你今天下午就能跑：

%%% steps
step: 挑相隔一个数量级的速率 —— 1.0、0.1、0.01、0.001、0.0001
why: 一个能用的速率可以住在从极小到接近一的任何地方，所以按十跳能用五次覆盖整个范围
step: 每个都跑一段**短**训练 —— 你不需要跑完任何一个
why: 一个糟糕的速率在最前面几步就露出它的形状，也就是说这个扫法几乎不花你什么
step: 为每次运行画 loss-对-步数，然后看**形状**，不只看最后那个数字
why: 形状说出区间的名字 —— 平的意味着太小，带尖刺或者在升意味着太大
step: 留下曲线平滑下降的那个速率，想更细的话再围着它收窄
why: 因此你从来没解方程；你看了图，挑了最好的那个
%%%

下面是那次扫描的赢家 —— 在 `lr = 0.1` 下 40 步之后的最终 loss。**先预测：** 接近谷底（≈0），还是还卡在高处？然后跑：

%%% demo id=sweepzh label="先猜，再展开"
predict: 在 lr = 0.1 下每一步把 weight 缩到它原来的 0.8 倍 —— 走 40 步之后，最终 loss 更接近 1、更接近 0.1，还是几乎什么都不剩？
code: train(lr=0.1)      # loss = weight**2，从 weight = 2.0 走 40 步下坡之后
out: 7.1e-08
take: <b>lr = 0.1 基本落在谷底上</b> —— 最终 loss ≈ 0.00000007，基本就是 0。把整个扫法跑一遍，赢家会自己跳出来：lr = 1.0 从不安顿（卡在 loss 4 上弹），lr = 0.1 一击命中（≈0），lr = 0.01 只爬了一部分（≈0.79），而 lr = 0.001（≈3.4）和 0.0001（≈3.9）几乎没从起始 loss 4 上动过。不需要方程 —— 扫法直接指向 0.1。
%%%

整个扫法画成图 —— 每个速率在 40 步之后的最终 loss。看清只有中间那个速率到了地板，而两端都还卡在高处：
~~~

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="A bar chart of final loss after 40 steps for five learning rates. Learning rate 1.0 is a tall bar at loss 4 (bouncing, never settles). 0.1 is essentially zero (reaches the bottom). 0.01 is a short bar at loss 0.79 (partway). 0.001 at loss 3.4 and 0.0001 at loss 3.9 are tall (barely moved). Only the middle rate reaches the floor."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Final loss after 40 steps (lower = better; floor = 0)</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">40 步之后的最终 loss（越低越好；地板 = 0）</text><line x1="70" y1="150" x2="470" y2="150" stroke="#B8AEA2"/><text class="lang-en" x="40" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 40 94)">final loss</text><text class="lang-zh" x="40" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 40 94)">最终 loss</text><g><rect x="88" y="30" width="46" height="120" fill="#C93B3B"/><text x="111" y="166" text-anchor="middle" fill="#C93B3B" font-size="8">1.0</text><text class="lang-en" x="111" y="24" text-anchor="middle" fill="#C93B3B" font-size="7.5">4.0 bounces 💥</text><text class="lang-zh" x="111" y="24" text-anchor="middle" fill="#C93B3B" font-size="7.5">4.0 在弹 💥</text></g><g><rect x="164" y="147" width="46" height="3" fill="#2D8B55"/><text x="187" y="166" text-anchor="middle" fill="#1a5c38" font-size="8">0.1</text><text x="187" y="140" text-anchor="middle" fill="#1a5c38" font-size="7.5">≈0 ✓</text></g><g><rect x="240" y="126" width="46" height="24" fill="#C99A12"/><text x="263" y="166" text-anchor="middle" fill="#9A7208" font-size="8">0.01</text><text class="lang-en" x="263" y="120" text-anchor="middle" fill="#9A7208" font-size="7.5">0.79 partway</text><text class="lang-zh" x="263" y="120" text-anchor="middle" fill="#9A7208" font-size="7.5">0.79 半路</text></g><g><rect x="316" y="48" width="46" height="102" fill="#C99A12"/><text x="339" y="166" text-anchor="middle" fill="#9A7208" font-size="8">0.001</text><text x="339" y="42" text-anchor="middle" fill="#9A7208" font-size="7.5">3.4 🐢</text></g><g><rect x="392" y="32" width="46" height="118" fill="#C99A12"/><text x="415" y="166" text-anchor="middle" fill="#9A7208" font-size="8">0.0001</text><text x="415" y="26" text-anchor="middle" fill="#9A7208" font-size="7.5">3.9 🐢</text></g></g></svg>
%%%

%%% insight
Notice what you did *not* need there: no derivation, no closed form, no theory of the loss surface. You tried five numbers and looked at five pictures. That really is how the rate gets picked in working labs — sweep, look, keep the best — and it is one of the friendliest facts in all of machine learning: the single most important knob is tuned by *looking*, and you can already look.
%%%

#### Read the curve shape to diagnose
When you plot loss-versus-step for each rate, the *shape* tells you the regime at a glance. Keep these three silhouettes in your head — they are how every practitioner eyeballs a learning rate:
~~~zh
%%% insight
注意你在那里*不*需要什么：没有推导、没有闭式解、没有关于 loss 表面的理论。你试了五个数字，看了五张图。真实的实验室里速率就是这么挑的 —— 扫、看、留下最好的。而这是整个机器学习里最友好的事实之一：那个最重要的旋钮是靠*看*调出来的，而你已经会看了。
%%%

#### 读曲线形状来诊断
当你为每个速率画出 loss-对-步数，那个*形状*一眼就告诉你区间。把这三个剪影记在脑子里 —— 每个从业者就是这么目测学习率的：
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three small loss-versus-step charts side by side. Left, too small: an almost-flat gently-sloping line staying high. Middle, just right: a curve that drops steeply then flattens near the bottom. Right, too large: a spiky line that jumps up and down and trends upward. Each is labelled with its diagnosis."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Three loss-curve shapes — learn them on sight</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">三种 loss 曲线形状 —— 练到一眼认出</text><g><rect x="16" y="30" width="152" height="128" rx="6" fill="#FDF3D6" stroke="#C99A12"/><path d="M28 48 C 80 54, 130 62, 156 70" fill="none" stroke="#C99A12" stroke-width="2.2"/><text class="lang-en" x="92" y="176" text-anchor="middle" fill="#9A7208" font-size="9">too small → flat 🐢</text><text class="lang-zh" x="92" y="176" text-anchor="middle" fill="#9A7208" font-size="9">太小 → 平 🐢</text></g><g><rect x="184" y="30" width="152" height="128" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><path d="M196 44 C 232 140, 300 150, 324 150" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text class="lang-en" x="260" y="176" text-anchor="middle" fill="#1a5c38" font-size="9">just right → smooth drop ✓</text><text class="lang-zh" x="260" y="176" text-anchor="middle" fill="#1a5c38" font-size="9">刚刚好 → 平滑下降 ✓</text></g><g><rect x="352" y="30" width="152" height="128" rx="6" fill="#FDECEC" stroke="#C93B3B"/><path d="M364 120 L 388 60 L 412 130 L 436 44 L 460 118 L 484 34" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text class="lang-en" x="428" y="176" text-anchor="middle" fill="#C93B3B" font-size="9">too large → spiky, rising 💥</text><text class="lang-zh" x="428" y="176" text-anchor="middle" fill="#C93B3B" font-size="9">太大 → 带尖刺、在升 💥</text></g></g></svg>
%%%

So the recipe is: **sweep rates by powers of ten, plot loss-versus-step, and pick the one whose curve drops smoothly.** Flat means raise it; spiky or rising means lower it; a clean steep-then-settle drop means you found it. There is one more trick that beats picking any single fixed rate — and it is the reason real training does not just set one number and walk away.
~~~zh
所以配方是：**按十的幂次扫速率，画 loss-对-步数，挑曲线平滑下降的那个。** 平的就提高它；带尖刺或者在升就降低它；干净的「陡降然后安顿」意味着你找到了。还有一个花招比挑任何单一固定速率都好 —— 而它就是真实训练不会只设一个数字然后走开的原因。
~~~

@@@ concept id=c5 zh_tag="让它随时间缩小" tag="Shrink it over time" zh_title="Schedule —— 早期大步，后期小步" title="Schedules — big hops early, tiny hops late" zh_gotit="懂了 schedule" gotit="Got the schedule"
Here is a tension you may have already felt. A *big* hop is great early on — you are far from the bottom and want to cover ground fast. But a big hop is *terrible* near the end — you are almost at the bottom and a big hop just overshoots it and bounces. A *small* hop is the reverse: gentle and precise at the end, but hopelessly slow at the start. So a single fixed hop size is always a compromise. The clever fix is to *not keep it fixed*: **start big, then shrink it as you go — but not all the way to nothing; ease it down to a small, steady size.** Picture walking home in the dark. Far from the house you take big confident strides. As you get close you slow to tiny careful shuffles so you do not walk straight into the door.
~~~zh
这里有一个你可能已经感觉到的矛盾。*大*跨度在早期很棒 —— 你离谷底还远，想快点覆盖地面。但大跨度在末尾*很糟* —— 你已经快到谷底了，一大跳只会迈过它然后弹开。*小*跨度反过来：末尾温柔又精确，但开头慢得没救。所以一个固定的跨度永远是个妥协。聪明的解法是*别让它固定*：**从大开始，然后一路缩小 —— 但不要缩到没有；把它缓缓降到一个小而稳定的大小。** 想象你在黑暗里走回家。离房子远的时候你迈自信的大步。快到的时候你慢成小心的小碎步，免得一头撞在门上。
~~~

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A figure walking from left to right toward a house on the right. The footsteps start as long strides on the left and get shorter and shorter approaching the door, then hold steady as tiny shuffles at the door. A label reads big strides far away and another reads tiny steady shuffles at the door."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Big strides far from home · tiny steady shuffles at the door</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">离家远时大步 · 到门口时小而稳的碎步</text><line x1="30" y1="130" x2="500" y2="130" stroke="#B8AEA2" stroke-width="2"/><rect x="452" y="86" width="44" height="44" fill="#E4D9C8" stroke="#B8AEA2"/><polygon points="452,86 474,64 496,86" fill="#C9B79C" stroke="#B8AEA2"/><rect x="468" y="104" width="12" height="26" fill="#9A7208"/><text class="lang-en" x="474" y="150" text-anchor="middle" fill="#6B645E" font-size="8">home (bottom)</text><text class="lang-zh" x="474" y="150" text-anchor="middle" fill="#6B645E" font-size="8">家（谷底）</text><g fill="#2A7B9B"><ellipse cx="60" cy="130" rx="6" ry="3"/><ellipse cx="120" cy="130" rx="6" ry="3"/><ellipse cx="180" cy="130" rx="5" ry="3"/><ellipse cx="235" cy="130" rx="5" ry="3"/><ellipse cx="285" cy="130" rx="4" ry="2.5"/><ellipse cx="330" cy="130" rx="3" ry="2"/><ellipse cx="368" cy="130" rx="2.5" ry="1.8"/><ellipse cx="400" cy="130" rx="2.5" ry="1.8"/><ellipse cx="426" cy="130" rx="2.5" ry="1.8"/><ellipse cx="445" cy="130" rx="2.5" ry="1.8"/></g><text class="lang-en" x="100" y="112" fill="#1F6280" font-size="8.5">big strides →</text><text class="lang-zh" x="100" y="112" fill="#1F6280" font-size="8.5">大步 →</text><text class="lang-en" x="405" y="112" text-anchor="middle" fill="#1F6280" font-size="8.5">tiny steady steps</text><text class="lang-zh" x="405" y="112" text-anchor="middle" fill="#1F6280" font-size="8.5">小而稳的步子</text></g></svg>
%%%

**What the walking-home picture gets right:** you *want* big steps when far away and small, careful steps when close, and shrinking the step as you approach gives both — exactly what a schedule does. **Where it breaks down:** you can *see* the house, so you know when to slow down; training cannot see the bottom, so the schedule shrinks the rate on a *pre-set plan* (for example, halve it every so often, then hold a small floor) rather than by watching distance.

#### The remedy has a name — a learning-rate schedule
Deliberately shrinking the learning rate as training goes on is called a [[learning-rate schedule||a plan that changes the learning rate over training — usually big early for speed, small late to settle. Shrinking it is also called decay]] (and the shrinking part is called **decay**). Big hops early cover ground fast; small hops late let you settle precisely into the bottom instead of bouncing over it. A common, simple plan: start at `0.8`, **halve the rate** each time — `0.8 → 0.4 → 0.2 → 0.1` — but hold a small floor of `0.1` so it eases into a gentle, steady size instead of vanishing to nothing. **Predict:** after several halvings from 0.8 with a 0.1 floor, what rate are we left with? Then run it:

%%% demo id=decay label="run it — a halving schedule that floors at 0.1"
predict: 0.8 halved five times is 0.025 — but the plan also has a 0.1 floor. Which of those two numbers does the schedule hand back?
code: max(0.1, 0.8 * 0.5**5)      # start 0.8, halve each step, never below the 0.1 floor
out: 0.1
take: <b>The rate starts big at 0.8, halves toward zero, and settles at the 0.1 floor.</b> The sequence is `0.8 → 0.4 → 0.2 → 0.1 → 0.1 → 0.1 …` — big fast hops early, then it eases down and holds a small, gentle 0.1 for a precise landing. That plan is a learning-rate schedule (decay) — one of the most common tricks in real training.
%%%

Each halving is a decision, so here is the plan rung by rung — with the reason each rung exists:

%%% steps
step: start at 0.8 — long, confident strides
why: you are far from the bottom, so a big hop is pure profit; there is nothing nearby to overshoot yet
step: halve to 0.4, then halve again to 0.2
why: the loss has dropped, which means you are closer in now, and the hop that helped at the start would begin to fly past the bottom
step: halve once more to 0.1 — and stop halving there
why: 0.1 is the floor; keep halving forever and the hops shrink to nothing, so learning would quietly stall out
step: hold 0.1 for the rest of training
why: therefore you finish with small steady shuffles that settle *into* the bottom instead of hopping back and forth over it
%%%

%%% insight
There is a quiet lesson hiding in this trick: the best hop size is not a number, it is a *plan*. Nearly every model you have used today — the one that finishes your sentence, the one that picks your next song — was trained on a schedule rather than one fixed rate. Once you see training as "big steps first, small steps last", you are thinking about it the way the people who train these models do.
%%%

Here is the schedule as a picture — each bar is the learning rate after another halving, big early and easing down to the steady `0.1` floor:
~~~zh
**走回家这个比喻对在哪里：** 你*就是*想在远处大步、在近处小心地小步，而随着接近缩小步子能同时拿到两样 —— 这正是 schedule 做的事。**它在哪里不成立：** 你能*看见*房子，所以你知道什么时候该慢下来；训练看不见谷底，所以 schedule 是按一个*预设的计划*缩小速率（比如每隔一段就减半，然后守住一个小的下限），而不是靠看距离。

#### 这个解药有个名字 —— learning-rate schedule
故意让学习率随训练推进缩小，叫做 [[learning-rate schedule||一个在训练过程中改变学习率的计划 —— 通常早期大以求快，后期小以安顿。缩小那部分也叫 decay]]（而缩小那部分叫 **decay（衰减）**）。早期的大跨度快速覆盖地面；后期的小跨度让你精确地落进谷底，而不是从它上面弹过去。一个常见又简单的计划：从 `0.8` 开始，每次**把速率减半** —— `0.8 → 0.4 → 0.2 → 0.1` —— 但守住一个 `0.1` 的下限，让它缓缓降到一个温和稳定的大小，而不是消失成零。**先预测：** 从 0.8 减半几次、带一个 0.1 下限，我们最后剩下什么速率？然后跑：

%%% demo id=schedzh label="先猜，再展开"
predict: 0.8 减半五次是 0.025 —— 但这个计划还有一个 0.1 的下限。schedule 会还给你这两个数字里的哪一个？
code: max(0.1, 0.8 * 0.5**5)      # 从 0.8 开始，每次减半，永不低于 0.1 的下限
out: 0.1
take: <b>速率从大的 0.8 出发，一路减半朝零走，最后停在 0.1 的下限上。</b>序列是 `0.8 → 0.4 → 0.2 → 0.1 → 0.1 → 0.1 …` —— 早期又大又快的跨度，然后它缓缓降下来、守住一个温和的 0.1 做精确落地。那个计划就是一个 learning-rate schedule（decay）—— 真实训练里最常见的花招之一。
%%%

每一次减半都是一个决定，所以这个计划一级一级来 —— 附上每一级存在的理由：

%%% steps
step: 从 0.8 开始 —— 长而自信的大步
why: 你离谷底还远，所以大跨度是纯赚；附近还没有什么可以让你迈过头的
step: 减半到 0.4，再减半到 0.2
why: loss 已经降了，也就是说你现在离得更近了，而开头帮了你的那个跨度开始要飞过谷底了
step: 再减半一次到 0.1 —— 然后就在那里停止减半
why: 0.1 是下限；一直减半下去跨度会缩成零，于是学习会悄悄停摆
step: 剩下的训练一直守住 0.1
why: 因此你用小而稳的碎步收尾，稳稳落*进*谷底，而不是在它上面来回跳
%%%

%%% insight
这个花招里藏着一个安静的道理：最好的跨度不是一个数字，是一个*计划*。你今天用过的几乎每一个模型 —— 那个替你补完句子的、那个替你挑下一首歌的 —— 都是用 schedule 训练出来的，而不是一个固定速率。一旦你把训练看成「先大步，后小步」，你想问题的方式就和训练这些模型的人一样了。
%%%

这个 schedule 画成图 —— 每一根条形是又减半一次之后的学习率，早期大，然后缓缓降到稳定的 `0.1` 下限：
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A bar chart of the learning rate over successive halvings starting at 0.8: bars at 0.8, 0.4, 0.2, then 0.1, 0.1, 0.1 held steady at a floor. A dashed line marks the 0.1 floor. Labels read start big and settle at the 0.1 floor."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Learning rate over halvings: big early → steady 0.1 floor</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">每次减半后的学习率：早期大 → 稳定的 0.1 下限</text><line x1="60" y1="150" x2="470" y2="150" stroke="#B8AEA2"/><text class="lang-en" x="34" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 34 94)">learning rate</text><text class="lang-zh" x="34" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 34 94)">learning rate</text><line x1="60" y1="135" x2="470" y2="135" stroke="#2D8B55" stroke-dasharray="3,3"/><text class="lang-en" x="474" y="138" fill="#1a5c38" font-size="8">0.1 floor</text><text class="lang-zh" x="474" y="138" fill="#1a5c38" font-size="8">0.1 下限</text><g fill="#2A7B9B"><rect x="80" y="30" width="44" height="120"/><rect x="146" y="90" width="44" height="60"/><rect x="212" y="120" width="44" height="30"/><rect x="278" y="135" width="44" height="15"/><rect x="344" y="135" width="44" height="15"/><rect x="410" y="135" width="44" height="15"/></g><g text-anchor="middle" fill="#6B645E" font-size="8"><text x="102" y="164">0.8</text><text x="168" y="164">0.4</text><text x="234" y="164">0.2</text><text x="300" y="164">0.1</text><text x="366" y="164">0.1</text><text x="432" y="164">0.1</text></g><text class="lang-en" x="102" y="24" text-anchor="middle" fill="#1F6280" font-size="8">start big</text><text class="lang-zh" x="102" y="24" text-anchor="middle" fill="#1F6280" font-size="8">从大开始</text><text class="lang-en" x="380" y="128" text-anchor="middle" fill="#1a5c38" font-size="8">settle at floor →</text><text class="lang-zh" x="380" y="128" text-anchor="middle" fill="#1a5c38" font-size="8">停在下限上 →</text></g></svg>
%%%

Real training often adds one more flourish at the very start: instead of jumping straight to the big rate on step one, it ramps *up* to it over the first few steps — a **warmup**, so training does not lurch on the first update — then decays. Here is that full shape, warmup up then decay down:
~~~zh
真实训练常常在最开头再加一个花样：不是第一步就跳到那个大速率，而是在最前面几步*爬升*到它。这叫 **warmup（预热）**，让训练不会在第一次更新时猛地一顿 —— 然后再衰减。下面是那个完整的形状，先预热上去再衰减下来：
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A learning-rate-versus-step curve. It rises briefly from zero at the start, labelled warmup, reaches a peak, then curves smoothly downward and levels off at a small value, labelled decay. The x-axis is steps, the y-axis is learning rate."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A common schedule: brief warmup up, then decay down</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">一个常见的 schedule：短暂预热上去，然后衰减下来</text><line x1="50" y1="30" x2="50" y2="148" stroke="#B8AEA2"/><line x1="50" y1="148" x2="500" y2="148" stroke="#B8AEA2"/><text class="lang-en" x="24" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 24 94)">learning rate</text><text class="lang-zh" x="24" y="94" fill="#6B645E" font-size="9" transform="rotate(-90 24 94)">learning rate</text><text class="lang-en" x="275" y="170" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><text class="lang-zh" x="275" y="170" text-anchor="middle" fill="#6B645E" font-size="9">步数 →</text><path d="M50 148 L 120 46 C 240 46, 340 118, 496 122" fill="none" stroke="#2A7B9B" stroke-width="2.6"/><line x1="120" y1="46" x2="120" y2="148" stroke="#E5DFD6" stroke-dasharray="3,3"/><text class="lang-en" x="86" y="40" fill="#1F6280" font-size="8.5">warmup ↑</text><text class="lang-zh" x="86" y="40" fill="#1F6280" font-size="8.5">预热 ↑</text><text class="lang-en" x="330" y="84" fill="#1F6280" font-size="8.5">decay ↓ (settle gently)</text><text class="lang-zh" x="330" y="84" fill="#1F6280" font-size="8.5">衰减 ↓（温柔地安顿）</text></g></svg>
%%%

So the schedule solves the fast-early-versus-precise-late tension: **start big for speed, shrink it (decay) for a gentle landing.** (There is a whole zoo of schedule shapes — step, exponential, cosine — plus warmup and adaptive optimizers like Adam that adjust the rate for you; those are for later lessons. Today you just need the core idea: the best hop size is not one number, it is a *plan*.) Let us gather the whole day onto one page.
~~~zh
所以 schedule 解决了「早期要快 vs 末尾要准」这个矛盾：**从大开始求速度，缩小它（decay）求一个温柔的落地。** （schedule 的形状有一整个动物园 —— step、exponential、cosine —— 再加上 warmup 和像 Adam 那样替你调速率的自适应优化器；那些留给以后的课。今天你只需要那个核心想法：最好的跨度不是一个数字，是一个*计划*。）我们把一整天收到一页上。
~~~

@@@ concept id=c6 zh_tag="回顾" tag="Recap" zh_title="一页看完今天" title="The whole day on one page" zh_gotit="回顾好了" gotit="Got the recap"
Here is a fresh way to hold the whole day in your head. Think of the **hot-water knob** in a shower. Turn it too far toward cold and the water is freezing — nothing happens fast, you just stand there shivering (that is a rate too *small*, the crawl). Turn it too far toward hot and you leap out scalded (that is a rate too *big*, the overshoot). You find the comfortable middle by nudging the knob a little each way and *feeling* the result — that is the sweep. And as the shower runs, you keep easing the knob to hold it just right — that is the schedule. One knob, felt by its result, adjusted over time. **Where the shower knob breaks down:** the shower has one comfy setting that barely changes; a network has a whole *range* of decent rates, and the best one really does drift as training goes, which is why decay matters more than any single perfect number.
~~~zh
这里有一个把一整天装进脑子的新办法。想象淋浴间的**热水旋钮**。往冷的方向转太多，水冰凉 —— 什么都不会快，你只是站在那里发抖（那是速率*太小*，也就是爬）。往热的方向转太多，你被烫得跳出来（那是速率*太大*，也就是迈过头）。你靠着往两边各拧一点、*感受*结果来找到舒服的中间 —— 那就是扫描。而随着水一直流，你不停地把旋钮往回微调以保持刚好 —— 那就是 schedule。一个旋钮，靠结果去感受，随时间调整。**淋浴旋钮这个比喻在哪里不成立：** 淋浴有一个几乎不变的舒服位置；而一个网络有一整*段*不错的速率，而且最好的那个真的会随训练漂移，这就是为什么 decay 比任何单一的完美数字都更重要。
~~~

%%% svg
<svg viewBox="0 0 520 178" role="img" aria-label="A shower temperature dial. The far-left cold zone is labelled too small, freezing, crawls. The far-right hot zone is labelled too big, scalding, overshoots. The middle green zone is labelled just right, comfortable. A hand nudges the dial toward the middle. A caption reads one knob, felt by its result."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One knob, felt by its result — the shower dial</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">一个旋钮，靠结果去感受 —— 淋浴的旋钮</text><path d="M110 140 A 150 150 0 0 1 410 140" fill="none" stroke="#B8AEA2" stroke-width="14"/><path d="M110 140 A 150 150 0 0 1 194 44" fill="none" stroke="#3A7C93" stroke-width="14"/><path d="M326 44 A 150 150 0 0 1 410 140" fill="none" stroke="#C93B3B" stroke-width="14"/><path d="M212 34 A 150 150 0 0 1 308 34" fill="none" stroke="#2D8B55" stroke-width="14"/><line x1="260" y1="140" x2="260" y2="52" stroke="#2C2A28" stroke-width="3"/><circle cx="260" cy="140" r="7" fill="#2C2A28"/><text class="lang-en" x="120" y="164" text-anchor="middle" fill="#3A7C93" font-size="8.5">🥶 too small (crawl)</text><text class="lang-zh" x="120" y="164" text-anchor="middle" fill="#3A7C93" font-size="8.5">🥶 太小（爬）</text><text class="lang-en" x="260" y="30" text-anchor="middle" fill="#1a5c38" font-size="8.5">✓ just right</text><text class="lang-zh" x="260" y="30" text-anchor="middle" fill="#1a5c38" font-size="8.5">✓ 刚刚好</text><text class="lang-en" x="404" y="164" text-anchor="middle" fill="#C93B3B" font-size="8.5">🔥 too big (overshoot)</text><text class="lang-zh" x="404" y="164" text-anchor="middle" fill="#C93B3B" font-size="8.5">🔥 太大（迈过头）</text></g></svg>
%%%

Now the whole arc in a few beats — the learning rate is the size of each downhill hop across the stream on its **stepping stones**:

- **What it is.** In *new weight = old weight − learning rate × gradient*, the gradient picks the *direction* (downhill) and the learning rate picks *how far you hop*. It is the one number scaling the gradient in the same loop that trains your neuron's weights and bias.
- **Too small → crawls.** Each hop is a grain of sand; the loss barely drops and the curve looks flat. **Remedy:** raise the learning rate.
- **Too big → overshoots, bounces, blows up.** Hops longer than the valley jump past the bottom and grow each step, ending in `NaN`. **Remedy:** lower the learning rate (or clip).
- **Find a good one by looking.** Sweep rates by powers of ten (1.0, 0.1, 0.01, 0.001), plot loss-versus-step, and pick the curve that drops smoothly. Flat → raise it; spiky/rising → lower it.
- **Beat a fixed rate with a schedule.** Start big for fast early progress, then *decay* it toward a small steady rate for a gentle, precise landing near the bottom.

One picture to keep — the same knob, three outcomes:
~~~zh
现在整条线索几个拍子 —— 学习率就是你踩着**垫脚石**过溪时每一步下坡的跨度：

- **它是什么。** 在 *新 weight = 老 weight − learning rate × gradient* 里，gradient 挑*方向*（下坡），learning rate 挑*你跨多远*。它就是那个在同一个循环里乘着 gradient、训练你神经元 weight 和 bias 的数字。
- **太小 → 爬。** 每一跳是一粒沙子；loss 几乎不降，曲线看起来是平的。**解药：** 提高学习率。
- **太大 → 迈过头、弹跳、炸掉。** 比山谷更长的跨度会跳过谷底、每一步变大，最后以 `NaN` 收场。**解药：** 降低学习率（或者裁剪）。
- **靠看去找一个好的。** 按十的幂次扫速率（1.0、0.1、0.01、0.001）。画 loss-对-步数，挑那条平滑下降的曲线。平 → 提高；带尖刺/在升 → 降低。
- **用 schedule 打败固定速率。** 从大开始求早期速度，然后把它 *decay* 到一个小而稳的速率，在谷底附近做温柔精确的落地。

一张要留住的图 —— 同一个旋钮，三种结果：
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Three loss-versus-step curves overlaid. A gold flat line labelled too small, a green steep-then-flat curve labelled just right, and a red spiky rising line labelled too large. A caption reads one knob, three outcomes."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One knob — three outcomes</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">一个旋钮 —— 三种结果</text><line x1="48" y1="28" x2="48" y2="146" stroke="#B8AEA2"/><line x1="48" y1="146" x2="500" y2="146" stroke="#B8AEA2"/><text class="lang-en" x="22" y="90" fill="#6B645E" font-size="9" transform="rotate(-90 22 90)">loss</text><text class="lang-zh" x="22" y="90" fill="#6B645E" font-size="9" transform="rotate(-90 22 90)">loss</text><text class="lang-en" x="274" y="168" text-anchor="middle" fill="#6B645E" font-size="9">steps →</text><text class="lang-zh" x="274" y="168" text-anchor="middle" fill="#6B645E" font-size="9">步数 →</text><path d="M54 44 C 200 56, 360 68, 496 78" fill="none" stroke="#C99A12" stroke-width="2.2"/><text class="lang-en" x="360" y="62" fill="#9A7208" font-size="8.5">too small 🐢</text><text class="lang-zh" x="360" y="62" fill="#9A7208" font-size="8.5">太小 🐢</text><path d="M54 44 C 130 128, 300 142, 496 143" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text class="lang-en" x="300" y="128" fill="#2D8B55" font-size="8.5">just right ✓</text><text class="lang-zh" x="300" y="128" fill="#2D8B55" font-size="8.5">刚刚好 ✓</text><path d="M54 90 L 96 42 L 138 96 L 180 34 L 222 100 L 264 30" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text class="lang-en" x="150" y="28" fill="#C93B3B" font-size="8.5">too large 💥</text><text class="lang-zh" x="150" y="28" fill="#C93B3B" font-size="8.5">太大 💥</text></g></svg>
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
~~~zh
那张快速诊断表 —— 读曲线，知道怎么修：

%%% table
:: 学习率 :: loss 曲线在做什么 :: 怎么修
太小 :: 几乎不降 —— 看起来是平的 🐢 :: 提高速率
刚刚好 :: 快速下降，然后安顿 ✓ :: 别动（或者加一个 schedule）
太大 :: 带尖刺、弹跳、升到 NaN 💥 :: 降低速率（或者裁剪）
%%%

还有那张小抄 —— 今天每一个词都在一个地方：

%%% jargon
learning rate (lr) | 每一步下坡的跨度：一个 weight 每步挪多远
gradient | loss 的斜率 —— 哪边是上坡、有多陡；我们往相反方向走
update rule | 新 weight = 老 weight − learning rate × gradient
diverge | loss 变大而不是变小，来自一个太大的速率 —— 以 NaN 收场
NaN | 「not a number」：一个值炸过任何实数时计算机打印的东西
order-of-magnitude sweep | 按十的幂次（1.0、0.1、0.01、0.001）试速率来找一个好的
loss curve | loss 对步数画出来 —— 它的形状告诉你速率是太小、刚好、还是太大
learning-rate schedule (decay) | 一个在训练中缩小速率的计划：早期大跨度，后期小而稳的跨度
%%%
~~~

@@@ quiz id=quiz zh_tag="小测" tag="Quiz" zh_title="四道快速检查" title="Four quick checks" zh_gotit="先全部答完" gotit="answer all first"
%%% quiz
q: In "new weight = old weight − learning rate × gradient", what does the learning rate control? | a:1 | which way is downhill | how far you hop each step | the number of weights | the value of the loss | fb: The learning rate is the step size — how far the weight moves each hop. The gradient picks the direction.
q: Your loss barely drops — the curve looks almost flat. What is likely wrong, and the fix? | a:2 | rate too big → lower it | model is broken → restart | rate too small → raise it | you are done → stop | fb: A near-flat loss means the hops are tiny — the rate is too small. Raise it.
q: Your loss bounces around and shoots up to NaN. What happened? | a:0 | the rate is too big — steps overshoot and diverge | the rate is too small — it crawls | the gradient is zero | the schedule finished | fb: Too-big steps jump past the bottom and grow each update until the numbers blow up to NaN. Lower the rate.
q: What does a learning-rate schedule (decay) do? | a:3 | uses more data per step | changes the loss function | fixes a bad model | starts with a big rate and shrinks it over time | fb: Decay: big hops early for fast progress, tiny hops late for a gentle, precise landing near the bottom.
%%%
~~~zh
四道快问，每题都有即时反馈。

%%% quiz
q: 在「新 weight = 老 weight − learning rate × gradient」里，learning rate 控制什么？ | a:1 | 哪边是下坡 | 你每一步跨多远 | weight 的数量 | loss 的值 | fb: learning rate 就是步长 —— weight 每一跳挪多远。方向由 gradient 挑。
q: 你的 loss 几乎不降 —— 曲线看起来几乎是平的。可能哪里出错了，怎么修？ | a:2 | 速率太大 → 降低它 | 模型坏了 → 重来 | 速率太小 → 提高它 | 你练完了 → 停 | fb: 一条接近平的 loss 意味着跨度极小 —— 速率太小。提高它。
q: 你的 loss 在乱弹然后冲到 NaN。发生了什么？ | a:0 | 速率太大 —— 步子迈过头并发散 | 速率太小 —— 它在爬 | gradient 是零 | schedule 跑完了 | fb: 太大的步子跳过谷底并且每次更新都变大，直到数字炸成 NaN。降低速率。
q: 一个 learning-rate schedule（decay）做什么？ | a:3 | 每步用更多数据 | 改 loss 函数 | 修好一个糟糕的模型 | 从一个大速率开始，随时间把它缩小 | fb: Decay：早期大跨度求快，后期小跨度在谷底附近温柔精确地落地。
%%%
~~~

@@@ produce id=produce zh_tag="动手做" tag="Produce" zh_title="自己扫一遍，感受三种状态" title="Feel the three regimes with your own sweep" zh_gotit="做完了" gotit="Done"
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

%%% prompt id=pp label="ask Claude to build it"
Help me build my artifact.
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
~~~zh
是时候在一次运行里*感受*整条主线：爬、安顿、爆炸 —— 全都来自转动一个旋钮。你会在几个不同的学习率下训练同一个小模型，看每条 loss 曲线讲它自己的故事。

设置是最简单的那种山谷：loss `= weight**2`（所以谷底在 `weight = 0`），gradient 是 `2*weight`，每次运行都从 `weight = 2.0` 出发跑 `40` 步。运行之间你*唯一*改的就是学习率。按十的幂次扫：`1.0, 0.1, 0.01, 0.001`。

**先预测，再运行：** 对这四个速率，猜猜 loss 会 (a) 几乎不动（爬）、(b) 平滑降到接近零（刚刚好）、还是 (c) 弹跳并炸掉（发散）。把你的四个猜测写下来。

**你应该看到什么：**
- **lr = 1.0** → weight 翻符号，loss **弹跳/炸掉** —— 太大（最终 loss 卡在 4）。
- **lr = 0.1** → loss **平滑降到几乎零**（大约 0.00000007）—— 刚刚好。
- **lr = 0.01** → loss 降了不少但**只走了一部分** —— 到大约 0.79，还没到谷底；开始爬了。
- **lr = 0.001** → loss **几乎不动**（大约 3.4，只比起点 4.0 低一点）—— 太小得离谱，一条接近平的曲线。
- 对一下顺序和你的预测：速率每缩小十倍，你就从*爆炸* → *刚刚好* → *半路爬行* → *几乎冻住*。整个旋钮，四行讲完。

把它写在 `experiment.py` 里然后跑。然后去戳它：把 `1.0` 提到 `1.1`，看它几步之内就到 NaN；加一个小 schedule，从 `0.8` 开始每隔几步把速率*减半*、守住 `0.1` 下限，看一个大的起始速率停在 0.1 而不是弹跳。

**路线 A —— 自己写。** 在 `experiment.py` 里填一个 `train(lr)` 循环，做 40 步 `weight -= lr * (2*weight)`，并为 `[1.0, 0.1, 0.01, 0.001]` 里每个速率打印最终 loss。

**路线 B —— 交给实验室。** 把下面这段贴进 experiment lab，让它替你搭出 `experiment.py`。（prompt 本身保持英文，它是给工具看的指令。）
~~~

@@@ fin
