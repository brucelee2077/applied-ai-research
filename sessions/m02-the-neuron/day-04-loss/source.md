---
quest_id: wf3-d01-loss
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 4 — Loss Functions"
module_label: "Module 2 · Train · Day 4"
zh_module_label: "第 2 模块 · 训练 · 第 4 天"
title: "Loss Functions"
zh_title: "损失函数"
subtitle: "How the Network Measures Its Own Mistakes"
zh_subtitle: "网络怎么给自己的错误打分"
brand_sub: "Foundations · M2 Day 4"
spine: "golf score"
zh_spine: "高尔夫"
nav_prev_href: "../review-part-a.html"
nav_prev_label: "M2 · Review Gate"
zh_nav_prev_label: "第 2 模块 · 复习关"
nav_next_href: "../day-05-gradients-backprop/lesson.html"
nav_next_label: "Gradients & Backpropagation"
zh_nav_next_label: "梯度与反向传播"
fin_title: "Module 2 · Day 4 complete! 🏆"
zh_fin_title: "第 2 模块 · 第 4 天 完成！🏆"
fin_body: "Nice work — you've completed <b>Loss Functions</b>. You can now put a single number on how wrong a guess is — the score the whole network is about to chase down.<br>Next up: <b>Gradients & Backpropagation</b> — how the network figures out which way to improve."
zh_fin_body: "做得好 —— 你完成了<b>损失函数</b>。现在你能用一个数字说清一次猜测错得多厉害。这个分数就是整个网络接下来要一路追着往下压的东西。<br>下一站：<b>梯度与反向传播</b> —— 网络怎么弄清该往哪一边改。"
notebook_yardstick: 00-neural-networks/fundamentals/06_loss_functions.ipynb
coverage_topics:
  - {topic: what a loss function is, keywords: [one honest number, how wrong]}
  - {topic: prediction vs target pairing, keywords: [prediction, target, how wrong]}
  - {topic: mean squared error, keywords: [mean squared error, squared misses, "mse ="]}
  - {topic: mse squaring property, keywords: [square, big misses count more, always positive]}
  - {topic: mean absolute error, keywords: [mean absolute error, plain miss sizes, absolute value]}
  - {topic: cross-entropy log loss, keywords: [cross-entropy, log loss, prob of the correct]}
  - {topic: cross-entropy punishes confident wrong, keywords: [confidently wrong, "-log", shoots up]}
  - {topic: binary cross-entropy, keywords: [binary cross-entropy, yes/no, sigmoid]}
  - {topic: categorical cross-entropy softmax, keywords: [categorical cross-entropy, softmax, sum to 1]}
  - {topic: logits raw scores, keywords: [logits, raw output scores]}
  - {topic: loss averaged over dataset, keywords: [average, cost, whole batch]}
  - {topic: loss must be smooth differentiable, keywords: [smooth, slope, "0/1 loss", counting mistakes]}
  - {topic: mse+sigmoid learns slowly, keywords: [stall, flat sigmoid slope, confidently wrong]}
  - {topic: remedy bce for slow mse, keywords: [cross-entropy is the fix, push stays]}
  - {topic: outlier dominates mse, keywords: [outlier, hijack]}
  - {topic: remedy mae, keywords: [switch to mae, huber]}
  - {topic: cross-entropy log(0) blow-up, keywords: ["log(0)", "-log(0)", infinity, nan]}
  - {topic: remedy epsilon clipping, keywords: [epsilon clipping, clamp]}
  - {topic: interpreting the loss value, keywords: [lower is better, goal is to make loss go down]}
  - {topic: capability limit only scores, keywords: [only scores, does not change any weight, optimizer]}
---

@@@ hero
@lede Welcome back! Picture a round of **mini-golf**. At each hole you tap the ball, it stops somewhere, and you write **one number** on your card: how far off you were. Low number, great hole. High number, rough one. A neural network keeps a scorecard exactly like that — and today you'll build it. Here's the part that should make you lean in: that one little number is what every model on earth, from the spam filter in your inbox to the model behind ChatGPT, spends its entire life trying to shrink. Learn to read the **golf score**, and you finally see what the whole game is *for*.
@goal Together we'll build the scorecard from scratch — one score for guessing numbers, a calm cousin for messy data, one for yes/no calls, one for picking among many. You'll *feel* why a confident-but-wrong guess should sting most, learn to read a falling score like a pro, catch three sneaky ways the score gets fooled (each a quick puzzle with a one-line fix), and learn the one honest thing a score can't do alone. Every formula arrives in plain words first. Heavy math sits in a skippable box.
%%% warmup
q: From yesterday — which way does a forward pass flow through the network? | a:2 | backward, from answer to input | in a loop, over and over | one direction, from input straight to the prediction | it skips around between layers | concept: forward-pass | fb: The forward pass is a one-way trip: input → hidden layer → output → prediction. No going back — that backward trip is training, coming later.
q: From yesterday — what happens if you stack layers with NO bend (activation) between them? | a:1 | they run twice as fast | the whole stack collapses into a single layer, so all the depth is wasted | the shapes stop matching and it crashes | each layer learns a different pattern | concept: linear-collapse | fb: With no activation between them, the straight-line layers fold into one — 5 layers get the power of 1. The fix is a bend between every pair of layers.
q: From yesterday — a layer with 4 neurons, each reading a 3-number input. What shape is its weight grid W? | a:0 | [4, 3] — one row per neuron, one column per input | [3, 4] | [4, 4] | [3, 3] | concept: weight-matrix-shape | fb: Rows = neurons, columns = inputs. 4 neurons reading 3 inputs give a [4, 3] grid.
%%%
@zh_lede 欢迎回来！想象你在打一场**迷你高尔夫**。每一洞你轻轻推一下球，球停在某个地方，你就在卡片上写下**一个数字**：你差了多远。数字小，这一洞打得漂亮。数字大，这一洞很糟。神经网络也有一张完全一样的记分卡 —— 今天你要亲手把它做出来。下面这句话值得你坐直一点听：这世界上的每一个模型，从你邮箱里的垃圾邮件过滤器，到 ChatGPT 背后的那个，一辈子都在想办法把这个小数字变小。学会读这张**高尔夫记分卡**，你才真正看清整场游戏是*为了什么*。
@zh_goal 我们会从零开始搭这张记分卡。一个分数用来猜数字。它还有一个脾气更稳的表亲，专门对付乱糟糟的数据。再一个用来回答「是」或「否」。最后一个用来在很多选项里挑一个。你会*亲身感觉到*为什么「很自信、但答错了」最该被扎一下。你会学着像老手一样读一条正在往下走的分数。你还会抓住三种偷偷骗过分数的手法，每一种都是一道小谜题，配一行就能修好的办法。最后你会知道一件分数自己做不到的老实事。每个公式都先用大白话讲一遍。重的数学放在一个可以跳过的盒子里。

~~~zh
%%% warmup
q: 昨天讲的 —— forward pass（前向传播）在网络里往哪个方向流？ | a:2 | 往后，从答案回到输入 | 在一个循环里，一遍又一遍 | 一个方向，从输入一直到 prediction（预测值） | 它在层与层之间跳来跳去 | concept: forward-pass | fb: forward pass 是一趟单程：输入 → hidden layer → 输出 → prediction。不回头 —— 那趟往回走的路叫训练，后面才讲。
q: 昨天讲的 —— 如果你把很多 layer（层）叠起来，中间不放任何弯（也就是 activation（激活函数）），会发生什么？ | a:1 | 它们跑得快两倍 | 整叠塌成一个 layer，所有的深度都白费了 | 形状对不上，程序崩掉 | 每一层学到不一样的模式 | concept: linear-collapse | fb: 中间没有 activation，几条直线的 layer 会折成一条 —— 5 层只有 1 层的力气。修法是在每两层之间放一个弯。
q: 昨天讲的 —— 一个 layer 有 4 个 neuron（神经元），每个读一个 3 个数字的输入。它的 weight（权重）网格 W 是什么形状？ | a:0 | [4, 3] —— 每个 neuron 一行，每个输入一列 | [3, 4] | [4, 4] | [3, 3] | concept: weight-matrix-shape | fb: 行 = neuron，列 = 输入。4 个 neuron 读 3 个输入，得到一个 [4, 3] 的网格。
%%%
~~~

@@@ concept id=c1 zh_tag="记分卡" tag="The scorecard" zh_title="loss 是什么 —— 一个诚实的数字" title="What a loss is — one honest number" zh_gotit="懂了 loss 是什么" gotit="Got what a loss is"
Let's start with the picture, before any math. In mini-golf your **score** is how far off you landed, and lower is always better. A perfect tap is a low number. The whole game is shooting lower next time.

A network keeps the same card. After it guesses, we hand it one number for how far off that guess landed.

That number is the [[loss||One number that says how wrong a prediction is. Lower is better; 0 means the guess was perfect.]], and the little rule that computes it is the [[loss function||The rule that takes the prediction and the true answer and returns a single "how wrong" number.]].
~~~zh
我们先看图，还不碰数学。在迷你高尔夫里，你的**分数**就是你的球落点差了多远，而越小永远越好。完美的一推是一个很小的数字。整场游戏就是下一次打得更小。

网络也用同一张卡。它猜完之后，我们递给它一个数字，说这次猜测差了多远。

那个数字就是 [[loss（损失）||一个数字，说明一次 prediction 错得多厉害。越小越好；0 表示这次猜得完美。]]，而算出它的那条小规则叫 [[loss function||一条规则：拿走 prediction 和真答案，还给你一个「错得多厉害」的数字。]]。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A mini-golf scorecard on a clipboard. Hole 1 scored 1, a low number, marked great. Hole 2 scored 7, a high number, marked rough. A note reads lower is better and 0 is perfect — a network keeps the same card."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Your golf card: one number per hole — lower is better</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">你的高尔夫记分卡：每洞一个数字 —— 越小越好</text><rect x="150" y="28" width="220" height="118" rx="8" fill="#FDFCF9" stroke="#C9B77A" stroke-width="2"/><rect x="228" y="22" width="64" height="12" rx="4" fill="#B8AEA2"/><line x1="150" y1="58" x2="370" y2="58" stroke="#E5DFD6"/><text class="lang-en" x="166" y="50" fill="#6B645E">hole</text><text class="lang-zh" x="166" y="50" fill="#6B645E">洞</text><text class="lang-en" x="300" y="50" fill="#6B645E">score</text><text class="lang-zh" x="300" y="50" fill="#6B645E">分数</text><text x="166" y="82" fill="#2C2A28">1  ⛳</text><rect x="286" y="70" width="30" height="18" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="301" y="83" text-anchor="middle" fill="#2D8B55">1</text><text class="lang-en" x="324" y="83" fill="#2D8B55" font-size="10">great!</text><text class="lang-zh" x="324" y="83" fill="#2D8B55" font-size="10">漂亮！</text><text x="166" y="110" fill="#2C2A28">2  ⛳</text><rect x="286" y="98" width="30" height="18" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="301" y="111" text-anchor="middle" fill="#C93B3B">7</text><text class="lang-en" x="324" y="111" fill="#C93B3B" font-size="10">rough</text><text class="lang-zh" x="324" y="111" fill="#C93B3B" font-size="10">很糟</text><text class="lang-en" x="260" y="136" text-anchor="middle" fill="#6B645E" font-size="10">0 = perfect shot · a network keeps this exact card</text><text class="lang-zh" x="260" y="136" text-anchor="middle" fill="#6B645E" font-size="10">0 = 完美一击 · 网络用的就是这张卡</text><text x="40" y="92" font-size="30">⛳</text><text class="lang-en" x="55" y="120" text-anchor="middle" fill="#6B645E" font-size="10">how far</text><text class="lang-zh" x="55" y="120" text-anchor="middle" fill="#6B645E" font-size="10">差了</text><text class="lang-en" x="55" y="133" text-anchor="middle" fill="#6B645E" font-size="10">off?</text><text class="lang-zh" x="55" y="133" text-anchor="middle" fill="#6B645E" font-size="10">多远？</text></g></svg>
%%%

**What the golf card gets right:** one tidy number, lower is better, 0 is perfect. **Where it breaks down:** a golf score is a whole number you read at the end, but a loss is a *smooth* number the network reads constantly — and smooth is what lets us find *which way* to improve tomorrow.

#### Beat 1 · the two things a loss compares
Every loss looks at exactly two things, and nothing else:

- the **prediction** — the guess that rolled out of yesterday's forward pass;
- the **target** (also called the **label**) — the true answer the data came with.
~~~zh
**高尔夫记分卡对在哪里：** 一个干干净净的数字，越小越好，0 就是完美。**它在哪里不成立：** 高尔夫分数是你最后才读的一个整数。而 loss 是网络时时刻刻都在读的一个*平滑*的数字。正是这份平滑，让我们能找出明天该往*哪一边*改。

#### 第 1 拍 · loss 只比两样东西
每个 loss 只看两样东西，别的都不看：

- **prediction** —— 昨天那趟 forward pass 滚出来的猜测；
- **target（目标值）**，也叫 **label（标签）** —— 数据自带的真答案。
~~~

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="A guess lands near a target; the loss is that gap turned into one number, like a golf score"><g font-family="monospace" font-size="13" text-anchor="middle"><circle cx="360" cy="60" r="26" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text class="lang-en" x="360" y="64" fill="#1a5c38" font-size="11">target</text><text class="lang-zh" x="360" y="64" fill="#1a5c38" font-size="11">目标值</text><circle cx="150" cy="60" r="7" fill="#2A7B9B"/><text class="lang-en" x="150" y="90" fill="#1F6280" font-size="11">guess</text><text class="lang-zh" x="150" y="90" fill="#1F6280" font-size="11">猜测</text><line x1="160" y1="60" x2="332" y2="60" stroke="#C99A12" stroke-width="2" stroke-dasharray="5,4"/><text class="lang-en" x="248" y="48" fill="#9A7208" font-size="11">how far off?</text><text class="lang-zh" x="248" y="48" fill="#9A7208" font-size="11">差了多远？</text><text class="lang-en" x="248" y="115" fill="#6B645E" font-size="11">loss = that gap, as one number (lower is better)</text><text class="lang-zh" x="248" y="115" fill="#6B645E" font-size="11">loss = 这个差距，写成一个数字（越小越好）</text></g></svg>
%%%

Hand a loss function that `(prediction, target)` pair and it gives back one **how wrong** number. That's it.

%%% insight
Here's the sentence to tattoo on your brain: **all of training is just changing the weights to make that number smaller.** Every trick you will ever read about — every clever architecture, every giant GPU cluster — is in service of shrinking this one honest number. So it's worth ten minutes.
%%%

#### Beat 2 · one hole, or the whole round?
*So far* we've scored **one** guess — one hole. Real training has thousands of examples, so we play the whole round and average the card.

%%% steps
step: score hole 1 → loss on example 1
why: pair up (prediction, target), get one number for how wrong that single guess was
step: do it for every example in the batch
why: each one gets its own little score, exactly like each hole gets its own number
step: average them all → the **cost**
why: one number for the whole batch, so the network has a single thing to chase down
%%%

That averaged number has its own name: the [[cost||The loss averaged over all the training examples: the single number the whole network tries to shrink.]] (you'll also hear "objective"). Same idea as one hole — just spread over the whole book of examples instead of one page.

Notice we took the **average**, not the sum. That's deliberate: an average stays the same size whether your batch has 8 examples or 800, therefore your training settings don't secretly change when you change the batch size.

!!! c-info 🏭
<b>A real one to keep you hungry:</b> a spam filter, a photo tagger, a chatbot — every one of them is trained by shrinking the same kind of number you're about to build with your own hands. Same idea, wildly different scale.
!!!
~~~zh
把那一对 `(prediction, target)` 递给一个 loss function，它就还给你一个**错得多厉害**的数字。就这样。

%%% insight
这句话值得刻在脑子里：**整个训练就是改 weight，让那个数字变小。** 你以后会读到的每一个花招 —— 每一个聪明的架构、每一片巨大的 GPU 集群 —— 都是为了把这一个诚实的数字压小。所以它值得你花十分钟。
%%%

#### 第 2 拍 · 一个洞，还是一整轮？
*到现在为止*我们只给**一次**猜测打了分 —— 一个洞。真正的训练有几千个样本。所以我们打完一整轮，再把整张卡取平均。

%%% steps
step: 给第 1 洞打分 → 第 1 个样本上的 loss
why: 把 (prediction, target) 配成一对，得到一个数字，说明这一次猜测错得多厉害
step: 对 batch 里的每一个样本都做一遍
why: 每个样本拿到自己的小分数，就像每一洞拿到自己的数字
step: 把它们全部取平均 → 得到 **cost**
why: 整个 batch 只剩一个数字，网络就有了唯一一个要追着往下压的东西
%%%

那个平均后的数字有自己的名字：[[cost（代价）||loss 在所有训练样本上取的平均：整个网络想要压小的那一个数字。]]（你也会听到有人叫它 objective）。和一个洞是同一个想法 —— 只是摊到整本样本上，而不是摊在一页上。

注意我们取的是**平均**，不是求和。这是故意的：不管你的 batch 里装 8 个样本还是 800 个，平均值的大小都差不多。所以你换 batch size 的时候，训练的设置不会偷偷跟着变。

!!! c-info 🏭
<b>一个真实的例子，让你更想往下看：</b>垃圾邮件过滤器、照片自动打标签、聊天机器人 —— 它们每一个，都是靠压小你马上要亲手做出来的这种数字训练出来的。同一个想法，规模差得离谱。
!!!
~~~

@@@ concept id=c2 zh_tag="为什么要平滑" tag="Why smooth" zh_title="数错题给不出提示 —— 所以我们改用平滑的分数" title="Counting mistakes gives no hint — so we go smooth" zh_gotit="懂了平滑为什么赢" gotit="Got why smooth wins"
Before anything fancy, let's try the scorecard a kid would invent: just **count the mistakes**. Ten guesses, three wrong, score is 3. Simple. Honest. And completely useless for learning — which is a lovely little puzzle.

Picture a quiz graded with a rubber stamp: **✓ or ✗**, nothing in between. You missed by a hair? ✗. You guessed something wild? Also ✗. The stamp tells you *that* you were wrong, never *how* wrong.
~~~zh
在搞任何花哨东西之前，我们先试试一个小孩子会想出来的记分卡：**数一数错了几道**。十次猜测，错了三次，分数就是 3。简单。老实。而且对学习完全没用 —— 这是一道挺可爱的小谜题。

想象一份用橡皮图章批改的考卷：**✓ 或者 ✗**，中间什么都没有。你只差一根头发丝？✗。你瞎猜了个离谱的？也是 ✗。图章只告诉你*你错了*，从不告诉你*错得多厉害*。
~~~

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A quiz graded with a red rubber stamp of check or X. A near-miss answer and a wild-guess answer both get the same red X. The stamp says whether you were wrong, never how wrong, and gives no hint which way to lean."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The rubber stamp: only ✓ or ✗ — nothing in between</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">橡皮图章：只有 ✓ 或 ✗ —— 中间什么都没有</text><rect x="40" y="30" width="210" height="52" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><text class="lang-en" x="52" y="50" fill="#6B645E">so close — off by a hair</text><text class="lang-zh" x="52" y="50" fill="#6B645E">就差一点点 —— 差一根头发丝</text><text class="lang-en" x="52" y="70" fill="#2C2A28">answer: 41  (true: 42)</text><text class="lang-zh" x="52" y="70" fill="#2C2A28">答案：41（真答案：42）</text><text x="228" y="70" font-size="26" fill="#C93B3B">✗</text><rect x="40" y="96" width="210" height="52" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><text class="lang-en" x="52" y="116" fill="#6B645E">wild guess — miles off</text><text class="lang-zh" x="52" y="116" fill="#6B645E">瞎猜 —— 差了老远</text><text class="lang-en" x="52" y="136" fill="#2C2A28">answer: 900 (true: 42)</text><text class="lang-zh" x="52" y="136" fill="#2C2A28">答案：900（真答案：42）</text><text x="228" y="136" font-size="26" fill="#C93B3B">✗</text><rect x="282" y="46" width="78" height="78" rx="10" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="321" y="92" text-anchor="middle" font-size="34" fill="#C93B3B">✗</text><text class="lang-en" x="321" y="140" text-anchor="middle" fill="#6B645E" font-size="10">same stamp</text><text class="lang-zh" x="321" y="140" text-anchor="middle" fill="#6B645E" font-size="10">同一个图章</text><text class="lang-en" x="392" y="78" fill="#C93B3B">tells you THAT</text><text class="lang-zh" x="392" y="78" fill="#C93B3B">只告诉你</text><text class="lang-en" x="392" y="94" fill="#C93B3B">you were wrong,</text><text class="lang-zh" x="392" y="94" fill="#C93B3B">你错了，</text><text class="lang-en" x="392" y="110" fill="#6B645E">never HOW wrong —</text><text class="lang-zh" x="392" y="110" fill="#6B645E">从不说错得多厉害 ——</text><text class="lang-en" x="392" y="126" fill="#6B645E">no way to lean better</text><text class="lang-zh" x="392" y="126" fill="#6B645E">没法知道该往哪边靠</text></g></svg>
%%%

**What the rubber stamp gets right:** counting mistakes is what you care about *in the end* — how many did I get right? **Where it breaks down:** a stamp can't say "you were almost right, lean a little this way." A network learns *only* by following little hints toward better.

#### The freeze — watch the count refuse to move
This crude score even has a proper name, the **0/1 loss** (1 for wrong, 0 for right). Put it under a microscope and nudge one weight.

%%% steps
step: nudge a weight a hair — the guess creeps from 41 toward 42, the true answer
why: real progress! Now: does the score notice?
step: re-read the count → still 1. Still 1. Still 1 — then at one exact point it snaps to 0
why: flat, flat, flat, then a cliff. Flat means no "warmer, colder" — nothing to follow.
step: the fix — measure HOW FAR off, not whether
why: distance changes with every nudge, therefore there is always a hint to follow
%%%

Here's that story as one picture — a cliff you can't climb, next to a ramp you can always slide down:
~~~zh
**橡皮图章对在哪里：** 数错了几道，正是你*到最后*真正在意的事 —— 我做对了几道？**它在哪里不成立：** 图章说不出「你已经很接近了，往这边挪一点点」。而网络*只*靠跟着这种一点点的提示走，才会变好。

#### 冻住了 —— 看那个计数怎么拒绝动
这个粗糙的分数还有个正经名字，叫 **0/1 loss**（错是 1，对是 0）。把它放到显微镜下，然后轻轻推一个 weight。

%%% steps
step: 把一个 weight 推一根头发丝 —— 猜测从 41 慢慢挪向 42，也就是真答案
why: 这是真的进步！那么：分数注意到了吗？
step: 再读一次计数 → 还是 1。还是 1。还是 1 —— 然后在某一个点上，它一下跳到 0
why: 平、平、平，然后一道悬崖。平就意味着没有「更热了、更冷了」—— 没有东西可以跟着走。
step: 修法 —— 量「差多远」，而不是「有没有错」
why: 距离每推一下都会变，所以永远有一个提示可以跟着走
%%%

把那个故事画成一张图 —— 一道你爬不上去的悬崖，旁边是一道你随时能往下滑的斜坡：
~~~

%%% svg
<svg viewBox="0 0 520 155" role="img" aria-label="Counting mistakes is a flat staircase with a cliff, giving no slope to follow; a smooth loss is a gentle ramp you can always slide down"><g font-family="monospace" font-size="11"><text class="lang-en" x="130" y="22" fill="#3A342E" font-weight="bold">count mistakes — a cliff</text><text class="lang-zh" x="130" y="22" fill="#3A342E" font-weight="bold">数错题 —— 一道悬崖</text><line x1="40" y1="130" x2="250" y2="130" stroke="#E5DFD6" stroke-width="1.5"/><path d="M50 55 L143 55 L143 120 L245 120" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text class="lang-en" x="90" y="48" fill="#C93B3B">wrong = 1 (flat)</text><text class="lang-zh" x="90" y="48" fill="#C93B3B">错 = 1（平的）</text><text class="lang-en" x="200" y="113" fill="#2D8B55">right = 0</text><text class="lang-zh" x="200" y="113" fill="#2D8B55">对 = 0</text><text class="lang-en" x="145" y="148" fill="#6B645E" text-anchor="middle">no slope to follow — stuck</text><text class="lang-zh" x="145" y="148" fill="#6B645E" text-anchor="middle">没有斜率可跟 —— 卡住</text><text class="lang-en" x="400" y="22" fill="#3A342E" font-weight="bold">smooth loss — a ramp</text><text class="lang-zh" x="400" y="22" fill="#3A342E" font-weight="bold">平滑的 loss —— 一道斜坡</text><line x1="300" y1="130" x2="500" y2="130" stroke="#E5DFD6" stroke-width="1.5"/><path d="M312 50 C 360 95, 430 122, 495 128" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text class="lang-en" x="400" y="148" fill="#6B645E" text-anchor="middle">always a hint: nudge downhill</text><text class="lang-zh" x="400" y="148" fill="#6B645E" text-anchor="middle">总有提示：往下坡挪一点</text></g></svg>
%%%

%%% insight
The sneaky part, worth feeling properly: a model trained on the count doesn't crash or complain. It runs, reports a number, and **learns absolutely nothing** — because a network improves by reading the **slope** of the loss, and a flat staircase has no slope to read.
%%%

So every real loss is a **smooth** stand-in for the crude count: it measures *how far off* you were, which means the score wiggles a little with every nudge, which means there is always a downhill direction. That's the rule behind every score in the rest of today.

!!! c-info 🧭
<b>Keep these two jobs straight:</b> counting mistakes is a fine way to <em>report</em> a grade — that's basically "accuracy", the number you'd tell a person. It's a terrible thing to <em>train</em> on. So we train on a smooth loss and report accuracy separately. Two jobs, two tools.
!!!
~~~zh
%%% insight
偷偷坑人的地方，值得好好感受一下：一个拿计数来训练的模型不会崩，也不会抱怨。它照样跑，照样报出一个数字，而且**什么都学不到** —— 因为网络是靠读 loss 的**斜率**变好的，而一段平的台阶没有斜率可读。
%%%

所以每一个真正在用的 loss，都是那个粗糙计数的**平滑**替身：它量的是你*差多远*。这意味着分数会随着每一次轻推而抖动一点。这又意味着永远存在一个往下坡的方向。今天后面每一个分数，背后都是这条规矩。

!!! c-info 🧭
<b>把这两份活分清楚：</b>数错了几道，是一个很好的<em>汇报</em>成绩的方式 —— 那基本上就是 accuracy（准确率），你会告诉别人的那个数字。但它是个很糟的<em>训练</em>对象。所以我们用一个平滑的 loss 来训练，再单独汇报 accuracy。两份活，两样工具。
!!!
~~~

@@@ concept id=c3 zh_tag="MSE" tag="MSE" zh_title="MSE —— 猜数字时用的分数" title="MSE — the score for guessing numbers" zh_gotit="懂了 MSE" gotit="Got MSE"
Now the fun starts — your first real golf score. Say the network is guessing a *number*: a house price, tomorrow's temperature, someone's age.

That task is called [[regression||A task where the network predicts a plain number (a price, a temperature), not a category.]], and its go-to score is [[MSE||Mean Squared Error: measure how far each guess is off, square it, then average. The default loss for predicting numbers.]] — "mean squared error". Heavy name, playground idea.

Picture **cornhole**: tossing beanbags at a hole in a board. For each toss you measure how far the bag landed from the hole. "Short by a foot" and "long by a foot" are equally bad — you don't care about the *direction* of a miss, only its **size**. And a wild toss into the neighbour's yard should count for a *lot* more than one that barely missed.
~~~zh
好玩的部分开始了 —— 你的第一个真正的高尔夫分数。假设网络在猜一个*数字*：房子的价格、明天的气温、某个人的年龄。

这类任务叫 [[regression（回归）||一类任务：网络预测的是一个普通数字（价格、温度），不是一个类别。]]，它的首选分数是 [[MSE||Mean Squared Error：量出每次猜测差多远，平方，再取平均。预测数字时的默认 loss。]] —— 也就是「mean squared error」。名字很沉，想法很像玩游戏。

想象**沙包投洞**：把沙包扔向木板上的一个洞。每扔一次，你就量沙包落点离洞多远。「短了一尺」和「远了一尺」一样糟 —— 你不在乎差的*方向*，只在乎它的**大小**。而一个飞到邻居家院子里的野球，应该比一个只差一点点的算重*得多*。
~~~

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Cornhole misses: short and long by the same amount are equally bad; a wild toss should hurt far more"><g font-family="monospace" font-size="11" text-anchor="middle"><circle cx="260" cy="60" r="14" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text class="lang-en" x="260" y="64" fill="#1a5c38" font-size="10">hole</text><text class="lang-zh" x="260" y="64" fill="#1a5c38" font-size="10">洞</text><circle cx="210" cy="60" r="6" fill="#2A7B9B"/><text class="lang-en" x="200" y="88" fill="#1F6280">short 1ft</text><text class="lang-zh" x="200" y="88" fill="#1F6280">短 1 尺</text><circle cx="310" cy="60" r="6" fill="#2A7B9B"/><text class="lang-en" x="320" y="88" fill="#1F6280">long 1ft</text><text class="lang-zh" x="320" y="88" fill="#1F6280">长 1 尺</text><text class="lang-en" x="260" y="30" fill="#6B645E">both equally bad — size, not direction</text><text class="lang-zh" x="260" y="30" fill="#6B645E">两边一样糟 —— 看大小，不看方向</text><circle cx="470" cy="60" r="8" fill="#C93B3B"/><text class="lang-en" x="450" y="88" fill="#C93B3B">wild toss</text><text class="lang-zh" x="450" y="88" fill="#C93B3B">瞎扔的一个</text><text class="lang-en" x="450" y="104" fill="#C93B3B">should hurt a LOT more</text><text class="lang-zh" x="450" y="104" fill="#C93B3B">应该疼得多得多</text></g></svg>
%%%

**What cornhole gets right:** direction doesn't matter, size does, and a wild miss should hurt far more than a near one. **Where it breaks down:** your worst toss is capped by how hard you can throw — MSE has no cap at all. Remember that; it's the very next puzzle.

#### Watch one score get born
Say the network predicted `[2.5, 0.0, 2.0]` and the true targets were `[3.0, −0.5, 2.0]`. Three tosses. Let's turn them into one number, one move at a time.

%%% steps
step: the misses — subtract: pred − target = [−0.5, 0.5, 0.0]
why: how far off each toss landed. Notice the trap: add these up and the first two cancel to zero!
step: the squeeze — square each miss: [0.25, 0.25, 0.0]
why: squaring kills the minus signs, so misses are always positive and can never cancel out
step: the same squeeze also blows up the big ones
why: twice as wrong costs four times as much, therefore big misses count more than small ones
step: the average — (0.25 + 0.25 + 0) / 3 = 0.167
why: one tidy score for all three tosses. A perfect set of guesses gives exactly 0.
%%%

Same three moves, drawn left to right so you can *see* the numbers change shape:
~~~zh
**沙包投洞对在哪里：** 方向不重要，大小重要，而一个野球该比一个差一点点的疼得多得多。**它在哪里不成立：** 你最糟的一扔顶多差那么远，因为你的力气有限；而 MSE 根本没有上限。记住这一点，它就是下一道谜题。

#### 看一个分数怎么诞生
假设网络预测了 `[2.5, 0.0, 2.0]`，而真的 target 是 `[3.0, −0.5, 2.0]`。三次投掷。我们一步一步把它们变成一个数字。

%%% steps
step: 差多少 —— 相减：pred − target = [−0.5, 0.5, 0.0]
why: 每次投掷差了多远。注意这里的坑：把这些加起来，前两个正好抵消成零！
step: 挤一下 —— 每个差取平方：[0.25, 0.25, 0.0]
why: 平方把减号杀掉，所以每个差都是正的，永远不会互相抵消
step: 同一个挤压，也会把大的那些吹得更大
why: 错两倍要付四倍的代价，所以大的差比小的差算得更重
step: 取平均 —— (0.25 + 0.25 + 0) / 3 = 0.167
why: 三次投掷合成一个干净的分数。一组完美的猜测正好得 0。
%%%

同样三步，从左到右画出来，让你能*看见*数字怎么换形状：
~~~

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Three steps of MSE drawn as a flow: misses, then squared misses, then their average of 0.167"><g font-family="monospace" font-size="12" text-anchor="middle"><text class="lang-en" x="90" y="30" fill="#6B645E">1 · misses</text><text class="lang-zh" x="90" y="30" fill="#6B645E">1 · 差多少</text><rect x="30" y="42" width="120" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="90" y="62" fill="#9A7208">[−0.5, 0.5, 0]</text><text x="185" y="62" fill="#6B645E">→</text><text class="lang-en" x="290" y="30" fill="#6B645E">2 · square each</text><text class="lang-zh" x="290" y="30" fill="#6B645E">2 · 每个取平方</text><rect x="220" y="42" width="140" height="30" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="290" y="62" fill="#C93B3B">[0.25, 0.25, 0]</text><text x="390" y="62" fill="#6B645E">→</text><text class="lang-en" x="465" y="30" fill="#6B645E">3 · average</text><text class="lang-zh" x="465" y="30" fill="#6B645E">3 · 取平均</text><rect x="415" y="42" width="95" height="32" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="462" y="63" fill="#1a5c38" font-weight="bold">0.167</text><text class="lang-en" x="260" y="118" fill="#6B645E" font-size="11">square → direction gone, big misses blown up; average → one MSE number</text><text class="lang-zh" x="260" y="118" fill="#6B645E" font-size="11">平方 → 方向消失，大的差被吹大；平均 → 一个 MSE 数字</text></g></svg>
%%%

In one line of plain words: **MSE = the average of the squared misses.** That's the whole score for numeric guesses.

%%% demo id=mse label="reveal the score"
predict: three misses of −0.5, +0.5 and 0. Will the score come out negative, zero, or a small positive number? Guess before you click.
code: pred=np.array([2.5,0,2]); target=np.array([3,-0.5,2]); np.mean((pred-target)**2)
out: 0.16666666666666666
take: <b>MSE = mean of squared errors.</b> Misses [−0.5, 0.5, 0] → squares [0.25, 0.25, 0] → mean ≈ 0.167. Small, because the guess is close. A far-off guess prints a much bigger number — lower always means a better fit.
%%%

%%% insight
Did you feel the near-miss in step 1? Two real misses of −0.5 and +0.5 would have **added to exactly zero** — a score of "perfect" for a model that missed twice. That's why the squaring isn't decoration; it's the thing that keeps the scorecard honest.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the formula in symbols.</b> Only peek if symbols help you; the four moves above are all you need. <code>MSE = (1/N) Σᵢ (predᵢ − targetᵢ)²</code>. Read it right to left: take each miss <code>(predᵢ − targetᵢ)</code>, square it, add them all up (the <code>Σ</code>), divide by <code>N</code> examples to make it an average. Same moves, written tight.
!!!

*So far:* one loss, fully built. **MSE is the default whenever the answer is a number** — reach for it first, and only switch away for the reason you're about to meet.
~~~zh
用一句大白话说：**MSE = 平方之后的差的平均。** 猜数字时的分数，全部就是这个。

%%% demo id=msezh label="展开这个分数"
predict: 三个差分别是 −0.5、+0.5 和 0。分数会是负的、是零，还是一个很小的正数？先猜，再点开。
code: pred=np.array([2.5,0,2]); target=np.array([3,-0.5,2]); np.mean((pred-target)**2)
out: 0.16666666666666666
take: <b>MSE = 平方之后的差的平均。</b>差是 [−0.5, 0.5, 0] → 平方是 [0.25, 0.25, 0] → 平均 ≈ 0.167。很小，因为猜得很近。一个差得老远的猜测会打印出大得多的数字 —— 越小永远表示贴得越好。
%%%

%%% insight
第 1 步里那个「差一点点」你感觉到了吗？两个真实的差 −0.5 和 +0.5 会**加起来正好等于零** —— 一个错了两次的模型，却拿到「完美」的分数。所以那个平方不是装饰；它就是让记分卡保持诚实的东西。
%%%

!!! c-info 🪜
<b>可跳过 —— 用符号写的公式。</b>只有在符号对你有帮助时才看一眼；上面那四步已经够用了。<code>MSE = (1/N) Σᵢ (predᵢ − targetᵢ)²</code>。从右往左读：拿每一个差 <code>(predᵢ − targetᵢ)</code>，平方，全部加起来（那个 <code>Σ</code>），再除以 <code>N</code> 个样本变成平均。同样几步，写得更紧。
!!!

*到目前为止：*一个 loss，完整搭好了。**只要答案是一个数字，MSE 就是默认选择** —— 先伸手拿它，只有为了你马上要见到的那个理由才换掉它。
~~~

@@@ concept id=c4 zh_tag="离群点 → MAE" tag="Outliers → MAE" zh_title="谜题 —— 一个野点绑走了整个分数" title="Puzzle — when one wild point hijacks the score" zh_gotit="懂了 MAE" gotit="Got MAE"
First puzzle of the day, and you've felt this one at a lunch table. You want the **average weight** of ten friends. Nine of you are ordinary kids — and the tenth guest is a sumo wrestler. Add, divide by ten, and the "average" comes out heavier than any real kid at the table.

One giant value dragged the whole average up toward itself. Now remember what MSE does *before* averaging: it **squares**. Can you see the trap coming?
~~~zh
今天的第一道谜题，而你在饭桌上感受过它。你想知道十个朋友的**平均体重**。你们九个是普通小孩 —— 而第十位客人是一个相扑选手。加起来，除以十，算出来的「平均」比桌上任何一个真小孩都重。

一个巨大的数值，把整个平均往它自己那边拖过去了。现在回想一下 MSE 在取平均*之前*做了什么：它**取平方**。你看到那个坑要来了吗？
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A lunch table of nine ordinary kids and one huge sumo wrestler guest. The average-weight marker gets dragged far up toward the sumo, ending heavier than any real kid, so it no longer describes the nine kids."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One sumo guest hijacks the average of ten</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">一个相扑客人绑走了十个人的平均</text><text x="30" y="64" font-size="22">🧒🧒🧒</text><text x="30" y="92" font-size="22">🧒🧒🧒</text><text x="30" y="120" font-size="22">🧒🧒🧒</text><text class="lang-en" x="48" y="140" fill="#6B645E" font-size="10">nine ordinary kids</text><text class="lang-zh" x="48" y="140" fill="#6B645E" font-size="10">九个普通小孩</text><text x="150" y="104" font-size="48">🥋</text><text class="lang-en" x="168" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">one giant guest</text><text class="lang-zh" x="168" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">一个巨人客人</text><line x1="238" y1="40" x2="238" y2="150" stroke="#E5DFD6"/><text class="lang-en" x="258" y="52" fill="#6B645E" font-size="10">weight scale →</text><text class="lang-zh" x="258" y="52" fill="#6B645E" font-size="10">体重刻度 →</text><line x1="248" y1="116" x2="330" y2="116" stroke="#2D8B55" stroke-width="2" stroke-dasharray="4,3"/><text class="lang-en" x="338" y="120" fill="#2D8B55" font-size="10">where the 9 kids sit</text><text class="lang-zh" x="338" y="120" fill="#2D8B55" font-size="10">9 个小孩在这里</text><line x1="248" y1="66" x2="440" y2="66" stroke="#C93B3B" stroke-width="2.5"/><text class="lang-en" x="338" y="58" fill="#C93B3B" font-size="10">the "average" dragged UP here</text><text class="lang-zh" x="338" y="58" fill="#C93B3B" font-size="10">「平均」被拖到了这里</text><path d="M300 108 C 330 96, 360 82, 388 70" fill="none" stroke="#C93B3B" stroke-width="1.5" stroke-dasharray="3,3"/><polygon points="388,70 380,70 385,77" fill="#C93B3B"/><text class="lang-en" x="320" y="166" text-anchor="middle" fill="#6B645E" font-size="10">the number now fits nobody real — and MSE SQUARES it first</text><text class="lang-zh" x="320" y="166" text-anchor="middle" fill="#6B645E" font-size="10">这个数字谁都不像 —— 而 MSE 还先把它平方</text></g></svg>
%%%

**What the sumo picture gets right:** one value far bigger than the rest pulls the whole average toward itself.

**Where it breaks down:** the sumo is real and belongs at the table, so his weight *should* count. A data [[outlier||A single data point far away from all the rest. Because MSE squares errors, one outlier can dominate the whole loss.]] is often just a typo or a freak case — and MSE doesn't merely add it, it **squares** it first.

#### The hijack, in three moves
You're grading house-price guesses. Feel the score slip out of your hands.

%%% steps
step: the honest misses — most guesses are off by a few thousand dollars
why: exactly what you'd hope for. A reasonable model on reasonable houses.
step: one freak mansion, missed by a million — and MSE **squares** it first
why: a million squared dwarfs a few thousand squared, therefore that single row now outweighs the whole crowd
step: the **hijack** — after averaging, the score is basically "how did you do on that ONE house?"
why: your good guesses stopped mattering. One bad row is shouting over ten thousand good ones.
%%%

#### The neat fix — stop squaring
Don't square each miss. Just take its plain size — its [[absolute value||The size of a number, ignoring its sign: |−7| = 7. Used by MAE so a miss counts by its size, not its size-squared.]], meaning how far off it was with the sign ignored — and average those.

That's [[MAE||Mean Absolute Error: average of the plain miss sizes (no squaring). Each miss counts in proportion, so one outlier can't dominate.]], **mean absolute error**: the average of the **plain miss sizes**. A miss of a million now counts as a million, *not* a million squared, which means it can no longer explode past everybody else. So when outliers are hijacking your score, you **switch to MAE**.

Your turn to break it — and to watch the cure hold. Below are twenty ordinary misses of about 2, plus one wild one **you** control. Predict what each score does, then drag:
~~~zh
**相扑这张图对在哪里：** 一个比其他人大得多的数值，会把整个平均往它自己那边拉。

**它在哪里不成立：** 相扑选手是真人，他本来就坐在这张桌子上，所以他的体重*应该*算进去。而数据里的 [[outlier（离群点）||一个离所有其他点都很远的数据点。因为 MSE 会把误差平方，一个 outlier 就能压住整个 loss。]] 常常只是一个打错的字，或者一个极端的怪例 —— 而 MSE 不只是把它加进去，它还先把它**平方**。

#### 分数被抢走方向盘，三步讲完
你在给房价的猜测打分。感觉一下分数怎么从你手里滑走。

%%% steps
step: 老实的那些差 —— 大部分猜测差几千美元
why: 正是你会希望看到的样子。一个讲道理的模型，在一批讲道理的房子上。
step: 一栋怪物豪宅，差了一百万 —— 而 MSE 还先把它**平方**
why: 一百万的平方，比几千的平方大得离谱，所以这一行现在压过了整群房子
step: **方向盘被抢走** —— 取完平均之后，分数基本上变成了「你在那一栋房子上做得怎么样？」
why: 你那些好猜测不再算数了。一行糟糕的数据，在对着一万行好数据喊。
%%%

#### 干净的修法 —— 别再平方
不要把每个差取平方。只取它原本的大小 —— 也就是它的 [[absolute value（绝对值）||一个数字的大小，不看它的正负号：|−7| = 7。MAE 用它，让一个差按它的大小算，而不是按它的平方算。]]，意思是「不看正负号，差了多远」—— 然后把这些取平均。

那就是 [[MAE||Mean Absolute Error：把原本大小的差取平均（不平方）。每个差按比例算，所以一个 outlier 压不住全场。]]，也就是 **mean absolute error**：**原本大小的差的平均**。差一百万现在就算一百万，*不是*一百万的平方。这意味着它再也没法炸到所有人头上去。所以当 outlier 在绑走你的分数时，你就**换成 MAE**。

轮到你来把它弄坏 —— 再看那个药怎么撑住。下面是二十个大约 2 的普通差，加上一个由**你**控制的野的。先预测每个分数会怎么动，然后拖：
~~~

%%% svg
<svg id="olx-svg" viewBox="0 0 520 196" role="img" aria-label="Interactive outlier explorer. Twenty ordinary misses of about two sit next to one wild miss you drag from 1 up to 1000. Two score bars on a shared log scale show the squared score MSE running away while the plain score MAE grows only gently."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Drag the wild miss — watch which score runs away</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">拖动那个野的差 —— 看哪个分数会跑掉</text><line x1="26" y1="160" x2="250" y2="160" stroke="#E5DFD6" stroke-width="1.5"/><g id="olx-small" fill="#2D8B55"><rect x="30" y="150" width="4" height="10"/><rect x="38" y="146" width="4" height="14"/><rect x="46" y="155" width="4" height="5"/><rect x="54" y="150" width="4" height="10"/><rect x="62" y="146" width="4" height="14"/><rect x="70" y="155" width="4" height="5"/><rect x="78" y="150" width="4" height="10"/><rect x="86" y="146" width="4" height="14"/><rect x="94" y="155" width="4" height="5"/><rect x="102" y="150" width="4" height="10"/><rect x="110" y="146" width="4" height="14"/><rect x="118" y="155" width="4" height="5"/><rect x="126" y="150" width="4" height="10"/><rect x="134" y="146" width="4" height="14"/><rect x="142" y="155" width="4" height="5"/><rect x="150" y="150" width="4" height="10"/><rect x="158" y="146" width="4" height="14"/><rect x="166" y="155" width="4" height="5"/><rect x="174" y="150" width="4" height="10"/><rect x="182" y="146" width="4" height="14"/></g><text class="lang-en" x="106" y="176" text-anchor="middle" fill="#2D8B55" font-size="10">20 ordinary misses (~2 each)</text><text class="lang-zh" x="106" y="176" text-anchor="middle" fill="#2D8B55" font-size="10">20 个普通的差（各约 2）</text><rect id="olx-wild" x="212" y="120" width="18" height="40" fill="#C93B3B"/><text class="lang-en" x="221" y="176" text-anchor="middle" fill="#C93B3B" font-size="10">the wild one</text><text class="lang-zh" x="221" y="176" text-anchor="middle" fill="#C93B3B" font-size="10">那个野的</text><line x1="278" y1="160" x2="504" y2="160" stroke="#E5DFD6" stroke-width="1.5"/><rect id="olx-mse" x="300" y="140" width="44" height="20" fill="#C93B3B"/><text class="lang-en" x="322" y="176" text-anchor="middle" fill="#C93B3B" font-size="10">MSE (squared)</text><text class="lang-zh" x="322" y="176" text-anchor="middle" fill="#C93B3B" font-size="10">MSE（平方过）</text><text id="olx-msev" x="322" y="134" text-anchor="middle" fill="#C93B3B" font-size="10">4.8</text><rect id="olx-mae" x="420" y="146" width="44" height="14" fill="#2A7B9B"/><text class="lang-en" x="442" y="176" text-anchor="middle" fill="#1F6280" font-size="10">MAE (plain size)</text><text class="lang-zh" x="442" y="176" text-anchor="middle" fill="#1F6280" font-size="10">MAE（原本大小）</text><text id="olx-maev" x="442" y="140" text-anchor="middle" fill="#1F6280" font-size="10">2.0</text><text class="lang-en" x="391" y="34" text-anchor="middle" fill="#6B645E" font-size="9">both bars share one squashed scale</text><text class="lang-zh" x="391" y="34" text-anchor="middle" fill="#6B645E" font-size="9">两根柱子用同一个压过的刻度</text></g></svg>
<div style="margin-top:8px">
<div style="display:flex;justify-content:space-between;font-size:.9em;color:#5A544E"><span><span class="lang-en">One weird house: missed by </span><span class="lang-zh">一栋怪房子：差了 </span><b id="olx-w">2</b></span><span><span class="lang-en">drag me →</span><span class="lang-zh">拖我 →</span></span></div>
<input id="olx-r" type="range" min="1" max="1000" value="2" step="1" aria-label="size of the one wild miss" style="width:100%;margin:6px 0;accent-color:#C93B3B">
<div id="olx-out" style="text-align:center;font-size:1.02em;color:#2C2A28"><span class="lang-en">MSE <b>4.8</b> · MAE <b>2.0</b> &nbsp;<span style="color:#2D8B55">no outlier yet — both scores agree</span></span><span class="lang-zh">MSE <b>4.8</b> · MAE <b>2.0</b> &nbsp;<span style="color:#2D8B55">还没有 outlier —— 两个分数一致</span></span></div>
</div>
<script>(function(){
  var s=document.getElementById('olx-r');if(!s)return;
  var small=[2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3];
  var sum=0,sumsq=0;for(var i=0;i<small.length;i++){sum+=small[i];sumsq+=small[i]*small[i];}
  var baseM=sumsq/small.length, baseA=sum/small.length, N=small.length+1;
  var bM=document.getElementById('olx-mse'),bA=document.getElementById('olx-mae'),
      tM=document.getElementById('olx-msev'),tA=document.getElementById('olx-maev'),
      wild=document.getElementById('olx-wild'),wtxt=document.getElementById('olx-w'),
      out=document.getElementById('olx-out');
  var BASE=160,MAXH=118,LOGMAX=Math.log10(60000);
  function hOf(v){return Math.max(6,Math.min(MAXH,MAXH*Math.log10(Math.max(v,1.01))/LOGMAX));}
  function paint(){
    var w=Math.max(1,+s.value);
    var mse=(sumsq+w*w)/N, mae=(sum+w)/N;
    var hM=hOf(mse),hA=hOf(mae),hW=Math.max(6,Math.min(MAXH,MAXH*Math.log10(Math.max(w,1.01))/Math.log10(1000)));
    if(bM){bM.setAttribute('y',(BASE-hM).toFixed(1));bM.setAttribute('height',hM.toFixed(1));}
    if(bA){bA.setAttribute('y',(BASE-hA).toFixed(1));bA.setAttribute('height',hA.toFixed(1));}
    if(wild){wild.setAttribute('y',(BASE-hW).toFixed(1));wild.setAttribute('height',hW.toFixed(1));}
    if(tM){tM.setAttribute('y',(BASE-hM-6).toFixed(1));tM.textContent=mse<100?mse.toFixed(1):Math.round(mse);}
    if(tA){tA.setAttribute('y',(BASE-hA-6).toFixed(1));tA.textContent=mae.toFixed(1);}
    if(wtxt)wtxt.textContent=w;
    var gM=mse/baseM,gA=mae/baseA;
    var word=w<10?'still tame — both scores agree':(w<120?'MSE is pulling ahead already':'MSE has been hijacked — MAE is only nudged');
    var wordZh=w<10?'还很温和 —— 两个分数一致':(w<120?'MSE 已经开始领跑了':'MSE 被绑走了 —— MAE 只是被轻推了一下');
    var col=w<10?'#2D8B55':(w<120?'#C99A12':'#C93B3B');
    var mseTxt=(mse<100?mse.toFixed(1):Math.round(mse));
    if(out)out.innerHTML='<span class="lang-en">MSE <b>'+mseTxt+'</b> (×'+Math.round(gM)+' bigger) · MAE <b>'+mae.toFixed(1)+'</b> (×'+Math.round(gA)+') &nbsp;<span style="color:'+col+'">'+word+'</span></span>'+'<span class="lang-zh">MSE <b>'+mseTxt+'</b>（大了 ×'+Math.round(gM)+'）· MAE <b>'+mae.toFixed(1)+'</b>（×'+Math.round(gA)+'） &nbsp;<span style="color:'+col+'">'+wordZh+'</span></span>';
  }
  s.addEventListener('input',paint);paint();
})();</script>
%%%

%%% insight
Slide it to the far right and read the two multipliers. One wild row blows the squared score up by **thousands of times**; the plain score grows only about twenty-fold. Both scores notice the freak — but that gap is the whole difference between an outlier being *amplified* and an outlier being merely *felt*.
%%%

%%% demo id=mae label="reveal both scores"
predict: twenty ordinary misses of about 2, then one wild 1000 crashes the party. One score grows about 24×, the other about 10,000×. Which is which?
code: small=np.array([2,3,1]*6+[2,3]); big=np.append(small,1000)
code: print("no outlier   MSE",round(np.mean(small**2),1)," MAE",round(np.mean(np.abs(small)),1))
code: print("with outlier MSE",round(np.mean(big**2),1)," MAE",round(np.mean(np.abs(big)),1))
out: no outlier   MSE 4.8  MAE 2.0
out: with outlier MSE 47623.7  MAE 49.6
take: <b>MAE = mean of the plain miss sizes.</b> One wild row multiplies MSE by about <b>10,000×</b> (4.8 → 47,624) but MAE by only about <b>24×</b> (2.0 → 49.6). MAE still <em>feels</em> the outlier — it just never lets it be squared into a monster, so the honest misses keep their say.
%%%

There's a best-of-both cousin worth knowing by name: [[Huber loss||A hybrid loss that acts like MSE for small misses (smooth near zero) and like MAE for large misses (calm about outliers).]] acts like **MSE for small misses** and like **MAE for big ones** — smooth near the target, calm about outliers. It's what people reach for when they want both.

!!! c-info ⚖️
<b>The trade-off in one breath:</b> MSE punishes big misses hardest (great on clean data); MAE treats every miss in proportion (great when a few crazy points would hijack the score); Huber sits in between. Which one you pick quietly decides what "doing well" even <em>means</em> — a real design call, not a detail.
!!!

**Victory lap:** puzzle one, solved. You turned a failure into a named cure — outliers hijack MSE, MAE (or Huber) fixes it. That failure→remedy move is *exactly* how engineers think. Two more puzzles later today, and both are quicker than this one.
~~~zh
%%% insight
把它拖到最右边，然后读那两个倍数。一行野数据把平方过的分数吹大了**几千倍**；原本大小的那个分数只长了大约二十倍。两个分数都注意到了这个怪例 —— 但那个差距，就是「outlier 被放大」和「outlier 只是被感觉到」之间的全部区别。
%%%

%%% demo id=maezh label="展开两个分数"
predict: 二十个大约 2 的普通差，然后一个野的 1000 冲进场。一个分数长大约 24 倍，另一个长大约 10000 倍。哪个是哪个？
code: small=np.array([2,3,1]*6+[2,3]); big=np.append(small,1000)
code: print("no outlier   MSE",round(np.mean(small**2),1)," MAE",round(np.mean(np.abs(small)),1))
code: print("with outlier MSE",round(np.mean(big**2),1)," MAE",round(np.mean(np.abs(big)),1))
out: no outlier   MSE 4.8  MAE 2.0
out: with outlier MSE 47623.7  MAE 49.6
take: <b>MAE = 原本大小的差的平均。</b>一行野数据把 MSE 乘上大约 <b>10,000 倍</b>（4.8 → 47,624），却只把 MAE 乘上大约 <b>24 倍</b>（2.0 → 49.6）。MAE 仍然<em>感觉得到</em>这个 outlier —— 它只是从不让这个 outlier 被平方成一头怪物，所以那些老实的差还能说上话。
%%%

还有一个两边好处都想要的表亲，值得记住它的名字：[[Huber loss||一个混合的 loss：对小的差像 MSE（在零附近平滑），对大的差像 MAE（对 outlier 很冷静）。]] 对**小的差像 MSE**，对**大的差像 MAE** —— 在 target 附近平滑，对 outlier 很冷静。人们两样都想要的时候，就伸手拿它。

!!! c-info ⚖️
<b>一口气说完这个取舍：</b>MSE 罚大的差最狠（数据干净时很棒）；MAE 让每个差按比例算（有几个疯点会绑走分数时很棒）；Huber 坐在中间。你挑哪一个，其实悄悄决定了「做得好」到底<em>是什么意思</em> —— 这是一个真正的设计决定，不是细节。
!!!

**跑一圈庆祝：**第一道谜题，解决了。你把一个失败变成了一个有名字的药 —— outlier 绑走 MSE，MAE（或者 Huber）修好它。这个「失败 → 解法」的动作，*正是*工程师思考的方式。今天后面还有两道谜题，两道都比这道快。
~~~

@@@ concept id=c5 zh_tag="是/否 → BCE" tag="Yes/No → BCE" zh_title="Cross-entropy —— 给有多自信打分" title="Cross-entropy — the score for confident guesses" zh_gotit="懂了 cross-entropy" gotit="Got cross-entropy"
Switch games — and meet the score behind almost every classifier you have ever used. Instead of a number, the network makes a **yes/no** call: spam or not spam, cat or dog, sick or healthy. That's [[binary classification||A task where the network picks between exactly two options: yes/no, spam/not-spam.]].

Think of a **weather forecaster**. She says "90% chance of rain" and it rains — great, barely a ding. But she shouts "99% chance of sun!" and it *pours* — that's a disaster: loud, certain, and dead wrong. A good scorecard should barely tap the first and absolutely hammer the second.
~~~zh
换个游戏 —— 来见见你用过的几乎每一个分类器背后的那个分数。网络这次不猜数字，而是做一个**是/否**的判断：是不是垃圾邮件、是猫还是狗、生病还是健康。这叫 [[binary classification（二分类）||一类任务：网络在正好两个选项之间挑一个：是/否、垃圾邮件/不是垃圾邮件。]]。

想一个**天气预报员**。她说「90% 的机会下雨」，然后真的下雨了 —— 很好，几乎不用罚。但她大喊「99% 的机会是晴天！」，结果*大雨倾盆* —— 那就是灾难：又响、又肯定、又完全错。一张好的记分卡，应该只轻轻碰一下前面那个，而对后面那个狠狠砸下去。
~~~

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="Two weather forecasters. Top: one loudly bets 99 percent sun but it rains — a huge surprise, a huge loss. Bottom: one carefully bets 90 percent rain and it rains — tiny surprise, near-zero loss. Cross-entropy measures that surprise."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Cross-entropy = your SURPRISE when the truth appears</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Cross-entropy = 真相出现时你有多惊讶</text><rect x="12" y="34" width="176" height="40" rx="8" fill="#FDECEC" stroke="#E0A9A0"/><text class="lang-en" x="100" y="52" text-anchor="middle" fill="#8a3b3b">bet ☀️ 99% SUN</text><text class="lang-zh" x="100" y="52" text-anchor="middle" fill="#8a3b3b">押 ☀️ 99% 晴</text><text class="lang-en" x="100" y="67" text-anchor="middle" font-size="9" fill="#9a6b64">loud &amp; certain</text><text class="lang-zh" x="100" y="67" text-anchor="middle" font-size="9" fill="#9a6b64">又响又肯定</text><text x="198" y="58" fill="#B8AEA2">→</text><rect x="214" y="34" width="86" height="40" rx="8" fill="#EFEAF7" stroke="#B3A6D4"/><text class="lang-en" x="257" y="58" text-anchor="middle" fill="#5E5191">truth: ☔ RAIN</text><text class="lang-zh" x="257" y="58" text-anchor="middle" fill="#5E5191">真相：☔ 下雨</text><text x="308" y="58" fill="#B8AEA2">→</text><rect x="324" y="34" width="184" height="40" rx="8" fill="#FDE8E8" stroke="#C93B3B"/><text x="338" y="60" font-size="16">😱</text><text class="lang-en" x="364" y="52" fill="#C93B3B">HUGE surprise</text><text class="lang-zh" x="364" y="52" fill="#C93B3B">惊讶得要命</text><text class="lang-en" x="364" y="67" font-size="9" fill="#C93B3B">→ huge loss</text><text class="lang-zh" x="364" y="67" font-size="9" fill="#C93B3B">→ 巨大的 loss</text><rect x="12" y="108" width="176" height="40" rx="8" fill="#EAF5EE" stroke="#A7D2B8"/><text class="lang-en" x="100" y="126" text-anchor="middle" fill="#276b45">bet ☔ 90% RAIN</text><text class="lang-zh" x="100" y="126" text-anchor="middle" fill="#276b45">押 ☔ 90% 雨</text><text class="lang-en" x="100" y="141" text-anchor="middle" font-size="9" fill="#5a8a6e">careful &amp; humble</text><text class="lang-zh" x="100" y="141" text-anchor="middle" font-size="9" fill="#5a8a6e">小心又谦虚</text><text x="198" y="132" fill="#B8AEA2">→</text><rect x="214" y="108" width="86" height="40" rx="8" fill="#EFEAF7" stroke="#B3A6D4"/><text class="lang-en" x="257" y="132" text-anchor="middle" fill="#5E5191">truth: ☔ RAIN</text><text class="lang-zh" x="257" y="132" text-anchor="middle" fill="#5E5191">真相：☔ 下雨</text><text x="308" y="132" fill="#B8AEA2">→</text><rect x="324" y="108" width="184" height="40" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="338" y="134" font-size="16">😌</text><text class="lang-en" x="364" y="126" fill="#2D8B55">tiny surprise</text><text class="lang-zh" x="364" y="126" fill="#2D8B55">一点点惊讶</text><text class="lang-en" x="364" y="141" font-size="9" fill="#2D8B55">→ loss ≈ 0</text><text class="lang-zh" x="364" y="141" font-size="9" fill="#2D8B55">→ loss ≈ 0</text><text class="lang-en" x="260" y="182" text-anchor="middle" font-size="10" fill="#6B645E">Same for cats, spam, anything: bet big on the WRONG answer → big shock → big loss.</text><text class="lang-zh" x="260" y="182" text-anchor="middle" font-size="10" fill="#6B645E">猫、垃圾邮件，什么都一样：重押在错答案上 → 大震惊 → 大 loss。</text></g></svg>
%%%

**What the forecaster gets right:** the penalty depends on *confidence*, not just right-vs-wrong. **Where it breaks down:** a real forecaster feels embarrassment; the network feels nothing. The whole sting lives in the math.

#### How the answer comes out as a probability
Before we can score confidence, the network has to *express* it. Two quick moves:

%%% steps
step: the last layer spits out a raw, untidied number — the **logits**
why: these are just the raw output scores. They can be −4.2 or +17; they are not percentages yet.
step: a **sigmoid** squashes that into a number between 0 and 1
why: now you can read it as "I'm 90% sure this is spam" — a confidence you can actually score
%%%

Those raw scores are the [[logits||The raw output scores of the last layer, before a sigmoid or softmax turns them into probabilities.]], and the squash is the [[sigmoid||A function that squashes any number into a probability between 0 and 1 — the natural output for a yes/no call.]] you met on Day 2.

The matching score for that kind of output is [[binary cross-entropy||The loss for yes/no predictions made as a probability. It barely dings confident-and-right guesses and hammers confident-and-wrong ones. Also called log loss.]] — also called **log loss**.

#### The one-line rule, in plain words
Cross-entropy looks at exactly one thing: the **prob of the correct** answer. Then it scores `−log` of it.

%%% steps
step: find the probability the model gave the TRUE answer
why: everything else it said is irrelevant — only the truth's share counts
step: take −log of that probability
why: `log` is the shape you met in M1; the minus flips it so small probability → big number
step: near 1 → the score is near 0
why: confident and right, so barely any penalty. This is the "😌" row of the picture.
step: sliding toward 0 → the score shoots up
why: confidently wrong, and the punishment grows without any ceiling. The "😱" row.
%%%

`log` quietly turns "probability" into "surprise". Now stop reading and **drag the dial** — watch that surprise climb a cliff:
~~~zh
**预报员这张图对在哪里：** 罚多少取决于*有多自信*，不只取决于对还是错。**它在哪里不成立：** 真的预报员会觉得难为情；网络什么都感觉不到。全部的「扎痛」都住在数学里。

#### 答案怎么变成一个概率
在能给自信打分之前，网络得先把自信*说出来*。两个快动作：

%%% steps
step: 最后一层吐出一个原始的、没整理过的数字 —— 也就是 **logits**
why: 这些只是原始的输出分数。它们可以是 −4.2 或者 +17；它们还不是百分比。
step: 一个 **sigmoid** 把它挤进 0 和 1 之间
why: 现在你可以把它读成「我有 90% 确定这是垃圾邮件」—— 一个你真的能打分的信心
%%%

那些原始分数就是 [[logits||最后一层的原始输出分数，还没被 sigmoid 或 softmax 变成概率。]]，而那个挤压就是你在第 2 天见过的 [[sigmoid||一个函数，把任何数字挤成 0 到 1 之间的一个概率 —— 是/否判断最自然的输出。]]。

配这种输出的分数是 [[binary cross-entropy||给「用概率表示的是/否预测」用的 loss。它几乎不罚又自信又对的猜测，却狠砸又自信又错的。也叫 log loss。]] —— 也叫 **log loss**。

#### 一行的规则，用大白话说
Cross-entropy（交叉熵）只看一样东西：模型给**正确答案**的那个概率。然后它给这个概率算 `−log`。

%%% steps
step: 找出模型给真答案的那个概率
why: 它说的其他一切都不相干 —— 只有真相那一份算数
step: 对那个概率取 −log
why: `log` 就是你在 M1 里见过的那个形状；那个减号把它翻过来，于是概率小 → 数字大
step: 接近 1 → 分数接近 0
why: 又自信又对，所以几乎不罚。这就是图里那一行「😌」。
step: 往 0 那边滑 → 分数直冲上去
why: 又自信又错，而惩罚一路往上长，没有天花板。那一行「😱」。
%%%

`log` 悄悄把「概率」变成了「惊讶」。现在别读了，**去拖那个转盘** —— 看那份惊讶怎么爬上一道悬崖：
~~~

%%% svg
<svg id="cex-svg" viewBox="0 0 520 190" role="img" aria-label="Interactive cross-entropy: drag your model's confidence in the correct answer and watch the loss — near zero when confident and right, exploding upward when confident and wrong"><g font-family="monospace" font-size="11"><line x1="60" y1="150" x2="484" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><line x1="60" y1="24" x2="60" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><path id="cex-curve" d="M62 30 C 140 84, 230 130, 320 142 C 384 149, 440 150, 478 150" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><circle id="cex-dot" cx="308" cy="142" r="6.5" fill="#C99A12" stroke="#fff" stroke-width="2"/><text class="lang-en" x="272" y="174" fill="#6B645E" text-anchor="middle">your model's confidence in the CORRECT answer  →</text><text class="lang-zh" x="272" y="174" fill="#6B645E" text-anchor="middle">你的模型对正确答案的信心  →</text><text class="lang-en" x="26" y="92" fill="#6B645E" text-anchor="middle" transform="rotate(-90 26 92)">loss (surprise)</text><text class="lang-zh" x="26" y="92" fill="#6B645E" text-anchor="middle" transform="rotate(-90 26 92)">loss（惊讶）</text><text class="lang-en" x="478" y="144" fill="#2D8B55" text-anchor="end">sure &amp; right → ≈ 0</text><text class="lang-zh" x="478" y="144" fill="#2D8B55" text-anchor="end">又肯定又对 → ≈ 0</text><text class="lang-en" x="90" y="40" fill="#C93B3B">sure &amp; WRONG → 😱</text><text class="lang-zh" x="90" y="40" fill="#C93B3B">又肯定又错 → 😱</text></g></svg>
<div style="margin-top:8px">
<div style="display:flex;justify-content:space-between;font-size:.9em;color:#5A544E"><span><span class="lang-en">Truth: it really <b>is</b> a cat 🐱.</span><span class="lang-zh">真相：它确实<b>是</b>一只猫 🐱。</span></span><span><span class="lang-en">Your model was </span><span class="lang-zh">你的模型有 </span><b id="cex-pct">60%</b><span class="lang-en"> sure.</span><span class="lang-zh"> 的把握。</span></span></div>
<input id="cex-p" type="range" min="1" max="99" value="60" step="1" aria-label="model confidence in the correct answer" style="width:100%;margin:6px 0;accent-color:#7C6DAA">
<div id="cex-out" style="text-align:center;font-size:1.02em;color:#2C2A28"><span class="lang-en">loss = −log(0.60) = <b>0.51</b> 😐 <span style="color:#C99A12">hedging — a medium ding</span></span><span class="lang-zh">loss = −log(0.60) = <b>0.51</b> 😐 <span style="color:#C99A12">在打太极 —— 不痛不痒的一下</span></span></div>
</div>
<script>(function(){
  var s=document.getElementById('cex-p');if(!s)return;
  var dot=document.getElementById('cex-dot'),curve=document.getElementById('cex-curve'),
      pct=document.getElementById('cex-pct'),out=document.getElementById('cex-out');
  var X0=60,X1=478,Y0=150,YT=30,LMAX=5;
  function xOf(p){return X0+p*(X1-X0);}
  function yOf(L){return Y0-Math.min(L,LMAX)/LMAX*(Y0-YT);}
  var d='';for(var i=0;i<=100;i++){var q=0.01+(0.99-0.01)*i/100;d+=(i?'L':'M')+xOf(q).toFixed(1)+' '+yOf(-Math.log(q)).toFixed(1)+' ';}
  if(curve)curve.setAttribute('d',d.trim());
  function paint(){
    var p=Math.max(1,Math.min(99,+s.value))/100,L=-Math.log(p);
    var col=p>=0.8?'#2D8B55':(p>=0.4?'#C99A12':'#C93B3B');
    var face=p>=0.8?'😌':(p>=0.4?'😐':'😱');
    var word=p>=0.8?'confident and right — barely a tap':(p>=0.4?'hedging — a medium ding':'confident and WRONG — brutal!');
    var wordZh=p>=0.8?'又自信又对 —— 只轻轻碰一下':(p>=0.4?'在打太极 —— 不痛不痒的一下':'又自信又错 —— 狠死了！');
    if(dot){dot.setAttribute('cx',xOf(p).toFixed(1));dot.setAttribute('cy',yOf(L).toFixed(1));dot.setAttribute('fill',col);}
    if(pct)pct.textContent=Math.round(p*100)+'%';
    if(out)out.innerHTML='<span class="lang-en">loss = −log('+p.toFixed(2)+') = <b>'+L.toFixed(2)+'</b> '+face+' &nbsp;<span style="color:'+col+'">'+word+'</span></span>'+'<span class="lang-zh">loss = −log('+p.toFixed(2)+') = <b>'+L.toFixed(2)+'</b> '+face+' &nbsp;<span style="color:'+col+'">'+wordZh+'</span></span>';
  }
  s.addEventListener('input',paint);paint();
})();</script>
%%%

%%% insight
Slide it all the way left and look at the number. That cliff is not a bug — it's the *point*. A model that hedges at 50% is only mildly punished, but one that is loudly, **confidently wrong** gets hammered, which means cross-entropy teaches a network to be humble unless it's sure. That single design choice is why classifiers behave sensibly.
%%%

%%% demo id=ce label="reveal the three scores"
predict: the model's confidence in the right answer goes 0.3 → 0.6 → 0.9. Does the loss rise or fall — and does it fall evenly?
code: [round(-np.log(p),3) for p in [0.3, 0.6, 0.9]]
out: [1.204, 0.511, 0.105]
take: <b>Cross-entropy = −log(probability of the correct answer).</b> As confidence in the right answer grows (0.3 → 0.6 → 0.9), the loss falls (1.204 → 0.511 → 0.105). Get it right with full confidence and the score glides toward 0.
%%%

#### The quiet pairing rule
*So far, so good.* One thing trips up every beginner: **each loss expects the output in a matching shape.** You can't bolt any loss onto any output.

%%% table
:: The task :: What the output looks like :: The loss to use
Guess a number (regression) :: a raw number (linear output) :: MSE (or MAE / Huber)
Yes / no (binary) :: one probability, 0 to 1, from a sigmoid :: binary cross-entropy
%%%

Get the pairing wrong and the score still computes *a* number — just the wrong one. That's a sneaky bug, and you'll meet it soon. **You can now pick the right scorecard for a yes/no game.** But what if there are more than two answers?
~~~zh
%%% insight
把它一路拖到最左边，然后看那个数字。那道悬崖不是毛病 —— 它就是*重点*。一个在 50% 上打太极的模型只被轻轻罚，而一个又响、又**自信、又错**的模型会被狠砸。这意味着 cross-entropy 在教网络：没把握就谦虚一点。就这一个设计选择，让分类器的行为变得讲道理。
%%%

%%% demo id=cezh label="展开这三个分数"
predict: 模型对正确答案的信心从 0.3 → 0.6 → 0.9。loss 会升还是降 —— 而且它降得均匀吗？
code: [round(-np.log(p),3) for p in [0.3, 0.6, 0.9]]
out: [1.204, 0.511, 0.105]
take: <b>Cross-entropy = −log（正确答案的概率）。</b>对正确答案的信心变大（0.3 → 0.6 → 0.9）时，loss 下降（1.204 → 0.511 → 0.105）。用满满的信心答对，分数就滑向 0。
%%%

#### 悄悄存在的配对规则
*到这里都还顺。*有一件事绊倒每一个新手：**每个 loss 都要求输出是配得上它的形状。** 你不能把随便一个 loss 硬拧到随便一个输出上。

%%% table
:: 任务 :: 输出长什么样 :: 该用的 loss
猜一个数字（regression） :: 一个原始数字（线性输出） :: MSE（或 MAE / Huber）
是 / 否（二分类） :: 一个 0 到 1 的概率，来自 sigmoid :: binary cross-entropy
%%%

配错了，分数照样能算出*一个*数字 —— 只是算错了那一个。这是个很阴的毛病，你很快就会见到它。**你现在能为一个是/否的游戏挑对记分卡了。** 但如果答案不止两个呢？
~~~

@@@ concept id=c6 zh_tag="多个答案 → softmax" tag="Many answers → softmax" zh_title="不止两个选项 —— softmax 加 categorical cross-entropy" title="More than two choices — softmax + categorical cross-entropy" zh_gotit="懂了 softmax" gotit="Got softmax"
Real life rarely stops at yes/no. Is this photo a **cat, a dog, or a bird**? Is this digit a **0, 1, 2 … or 9**? Picking one answer from three or more choices is [[multi-class classification||Picking one answer out of three or more choices, like cat vs dog vs bird.]] — and here's the happy surprise: it's the *same* score you just learned, gently stretched.

Picture **one whole pizza that you slice up** among the choices. You have 100% of a pizza — your total confidence — and you cut it: a big slice for "cat", a smaller one for "dog", a sliver for "bird". The slices always add up to exactly one whole pizza. Never more, never less.
~~~zh
真实生活很少停在是/否。这张照片是**猫、狗，还是鸟**？这个数字是 **0、1、2 …… 还是 9**？从三个或更多选项里挑一个答案，叫 [[multi-class classification（多分类）||从三个或更多选项里挑一个答案，比如猫、狗、鸟。]] —— 而这里有个让人开心的意外：它用的是你刚刚学过的*同一个*分数，只是轻轻拉长了一点。

想象**一整张披萨，你把它切开**分给这些选项。你手上有 100% 的一张披萨 —— 那是你全部的信心 —— 你把它切开：给「猫」一大块，给「狗」小一点的一块，给「鸟」一条细的。这些块加起来永远正好是一整张披萨。不会更多，也不会更少。
~~~

%%% svg
<svg viewBox="0 0 520 182" role="img" aria-label="One whole pizza of confidence sliced among three choices. Cat gets a big slice covering two thirds of the pizza at 66 percent, dog gets a smaller slice at 24 percent, and bird gets a thin sliver at 10 percent. The slices always add up to exactly one whole pizza."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One pizza of confidence, sliced across every choice — always adds to 1</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">一整张信心披萨，切给每一个选项 —— 加起来永远是 1</text><circle cx="128" cy="104" r="58" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><path d="M128 104 L186 104 A58 58 0 1 1 96.9 55.0 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><path d="M128 104 L96.9 55.0 A58 58 0 0 1 174.9 69.9 Z" fill="#E7F0F5" stroke="#2A7B9B" stroke-width="1.5"/><path d="M128 104 L174.9 69.9 A58 58 0 0 1 186 104 Z" fill="#F3ECDB" stroke="#C99A12" stroke-width="1.5"/><text x="111" y="136" text-anchor="middle" fill="#276b45" font-size="11">🐱 .66</text><text x="134" y="73" text-anchor="middle" fill="#1F6280" font-size="9">.24</text><rect x="228" y="48" width="180" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="236" y="64" fill="#276b45">🐱 cat</text><text class="lang-zh" x="236" y="64" fill="#276b45">🐱 猫</text><text class="lang-en" x="402" y="64" text-anchor="end" fill="#276b45">.66 big slice</text><text class="lang-zh" x="402" y="64" text-anchor="end" fill="#276b45">.66 大块</text><rect x="228" y="78" width="65" height="22" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text x="234" y="94" fill="#1F6280">🐶</text><text class="lang-en" x="302" y="94" fill="#1F6280">dog .24</text><text class="lang-zh" x="302" y="94" fill="#1F6280">狗 .24</text><rect x="228" y="108" width="27" height="22" rx="4" fill="#F3ECDB" stroke="#C99A12"/><text x="232" y="124" font-size="10">🐦</text><text class="lang-en" x="264" y="124" fill="#9A7208">bird .10 — the sliver</text><text class="lang-zh" x="264" y="124" fill="#9A7208">鸟 .10 —— 那一条细的</text><text class="lang-en" x="228" y="150" fill="#6B645E" font-size="10">chip widths match the slice sizes</text><text class="lang-zh" x="228" y="150" fill="#6B645E" font-size="10">小条的宽度对上每块的大小</text><text class="lang-en" x="228" y="168" fill="#1a5c38" font-size="10">.66 + .24 + .10 = one whole pizza</text><text class="lang-zh" x="228" y="168" fill="#1a5c38" font-size="10">.66 + .24 + .10 = 一整张披萨</text></g></svg>
%%%

The step that cuts the pizza — turning the raw logits into slices that add to 1 — is called [[softmax||A step that turns a list of raw scores (logits) into probabilities that add up to 1 — the natural output for choosing among 3+ classes.]].

**What the pizza gets right:** a fixed whole to share out, so giving more to one choice *means* less for the others. **Where it breaks down:** you could cut real slices any size by hand, but softmax sets the sizes automatically from the raw scores — always positive, always adding to one whole.

#### From raw scores to slices to a score
Watch three raw numbers become a score, one move at a time.

%%% steps
step: the raw scores arrive — logits [2.0, 1.0, 0.1]
why: just the last layer's untidied output. Nothing here is a percentage yet.
step: **softmax slices the pizza** → [0.66, 0.24, 0.10]
why: bigger score gets a bigger slice, all slices positive, and together they sum to 1
step: read the slice you gave the CORRECT class — 0.66 for cat
why: same rule as yes/no — only the truth's share matters
step: score it: −log(0.66) ≈ 0.416
why: fat slice for the right answer → small loss. Thin sliver → big loss. Identical heart.
%%%
~~~zh
切披萨那一步 —— 把原始的 logits 变成加起来等于 1 的一块块 —— 叫做 [[softmax||一个步骤：把一串原始分数（logits）变成加起来等于 1 的概率 —— 从 3 个以上类别里挑一个时最自然的输出。]]。

**披萨这张图对在哪里：** 有一个固定的整体要分出去，所以给某个选项多一点，就*意味着*其他选项少一点。**它在哪里不成立：** 真披萨你可以随手切成任何大小，而 softmax 是根据原始分数自动定大小的 —— 永远是正的，也永远加起来等于一整张。

#### 从原始分数，到一块块，再到一个分数
看三个原始数字怎么一步一步变成一个分数。

%%% steps
step: 原始分数来了 —— logits [2.0, 1.0, 0.1]
why: 只是最后一层还没整理过的输出。这里还没有任何东西是百分比。
step: **softmax 切披萨** → [0.66, 0.24, 0.10]
why: 分数大的拿到大块，每一块都是正的，而且它们合起来等于 1
step: 读你给**正确**那一类的那一块 —— 猫是 0.66
why: 和是/否同一条规则 —— 只有真相那一份算数
step: 给它打分：−log(0.66) ≈ 0.416
why: 正确答案拿到厚厚一块 → loss 小。只拿到一条细的 → loss 大。心是同一颗。
%%%
~~~

%%% svg
<svg viewBox="0 0 520 152" role="img" aria-label="Softmax turns three raw scores into three probabilities that add up to one. The cat segment fills two thirds of the bar, the dog segment is a quarter of it, and the bird segment is a thin sliver."><g font-family="monospace" font-size="11"><text class="lang-en" x="20" y="30" fill="#6B645E">raw scores (logits)</text><text class="lang-zh" x="20" y="30" fill="#6B645E">原始分数（logits）</text><rect x="20" y="42" width="120" height="26" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="80" y="60" fill="#9A7208" text-anchor="middle">[2.0, 1.0, 0.1]</text><text class="lang-en" x="152" y="56" fill="#6B645E">— softmax →</text><text class="lang-zh" x="152" y="56" fill="#6B645E">— softmax →</text><text class="lang-en" x="160" y="72" fill="#9A938A" font-size="9">cuts the pizza</text><text class="lang-zh" x="160" y="72" fill="#9A938A" font-size="9">把披萨切开</text><text class="lang-en" x="280" y="30" fill="#6B645E">probabilities (they add to 1)</text><text class="lang-zh" x="280" y="30" fill="#6B645E">概率（加起来是 1）</text><rect x="280" y="42" width="145" height="26" fill="#2D8B55"/><rect x="425" y="42" width="53" height="26" fill="#2A7B9B"/><rect x="478" y="42" width="22" height="26" fill="#C99A12"/><text class="lang-en" x="352" y="60" fill="#fff" text-anchor="middle">cat .66</text><text class="lang-zh" x="352" y="60" fill="#fff" text-anchor="middle">猫 .66</text><text class="lang-en" x="451" y="60" fill="#fff" text-anchor="middle" font-size="10">dog .24</text><text class="lang-zh" x="451" y="60" fill="#fff" text-anchor="middle" font-size="10">狗 .24</text><line x1="489" y1="70" x2="489" y2="82" stroke="#9A7208"/><text class="lang-en" x="489" y="94" fill="#9A7208" text-anchor="middle" font-size="10">bird .10</text><text class="lang-zh" x="489" y="94" fill="#9A7208" text-anchor="middle" font-size="10">鸟 .10</text><text class="lang-en" x="20" y="120" fill="#6B645E" font-size="11">one whole "pizza" of confidence, sliced across every class → always sums to 1.0</text><text class="lang-zh" x="20" y="120" fill="#6B645E" font-size="11">一整张信心「披萨」，切给每一类 → 加起来永远是 1.0</text><text class="lang-en" x="20" y="142" fill="#1a5c38" font-size="11">loss = −log(the slice you gave the CORRECT class) — same rule as yes/no</text><text class="lang-zh" x="20" y="142" fill="#1a5c38" font-size="11">loss = −log（你给正确那一类的那块）—— 和是/否同一条规则</text></g></svg>
%%%

This multi-class version has a name: [[categorical cross-entropy||The loss for picking one of 3+ classes. It reads the softmax probability you gave the correct class and scores −log of it — the same rule as binary cross-entropy, spread across many classes.]]. Notice you learned nothing genuinely new — you stretched one idea over more slices.

%%% insight
The pizza rule has a consequence that's easy to miss: because the slices must **sum to 1**, the classes *compete*. Giving "cat" more confidence forces "dog" and "bird" to give some up. That's why softmax is right for "pick exactly one" — and why it's the wrong tool when several answers can be true at once.
%%%

%%% demo id=cce label="reveal the score"
predict: the model gave the correct class (cat) a 0.66 slice. Will the loss be closer to 0.4, or closer to 4?
code: probs=[0.66,0.24,0.10]; correct=0; round(-np.log(probs[correct]),3)  # 3 classes, true class = cat
out: 0.416
take: <b>Categorical cross-entropy = −log(the slice you gave the correct class).</b> Cat got 0.66, so the loss is −log(0.66) ≈ 0.416. Slice the correct class thinner and this climbs; hand it nearly the whole pizza and it drops toward 0.
%%%

**The whole classification toolkit in one breath:** yes/no → **sigmoid + binary cross-entropy**; one-of-many → **softmax + categorical cross-entropy**. Same `−log` heart, two shapes of output. That's what you'll carry into every model you build from here.
~~~zh
这个多类别的版本有个名字：[[categorical cross-entropy||从 3 个以上类别里挑一个时用的 loss。它读你给正确类别的那个 softmax 概率，再算它的 −log —— 和 binary cross-entropy 同一条规则，只是摊到很多类别上。]]。注意你其实没学到什么真正新的东西 —— 你只是把一个想法拉长到更多块上。

%%% insight
披萨这条规则有一个很容易漏掉的后果：因为这些块必须**加起来等于 1**，各个类别在*互相抢*。给「猫」多一点信心，就迫使「狗」和「鸟」让出一些。这就是为什么 softmax 适合「正好挑一个」—— 也是为什么好几个答案可以同时为真的时候，它是错的工具。
%%%

%%% demo id=ccezh label="展开这个分数"
predict: 模型给正确的那一类（猫）0.66 的一块。loss 会更接近 0.4，还是更接近 4？
code: probs=[0.66,0.24,0.10]; correct=0; round(-np.log(probs[correct]),3)  # 3 个类别，真的那一类 = 猫
out: 0.416
take: <b>Categorical cross-entropy = −log（你给正确类别的那一块）。</b>猫拿到 0.66，所以 loss 是 −log(0.66) ≈ 0.416。把正确类别那一块切得更薄，这个数就往上爬；几乎把整张披萨都给它，它就掉向 0。
%%%

**一口气说完整个分类工具箱：**是/否 → **sigmoid + binary cross-entropy**；多选一 → **softmax + categorical cross-entropy**。同一颗 `−log` 的心，两种输出形状。从这里往后，你搭的每一个模型都会带着它。
~~~

@@@ concept id=c7 zh_tag="MSE 变慢" tag="Slow MSE" zh_title="谜题 —— MSE 在是/否题上为什么卡住" title="Puzzle — why MSE stalls on yes/no calls" zh_gotit="懂了那个卡住" gotit="Got the stall"
Second puzzle, and it starts with a tempting shortcut. MSE is so simple — why not use it for yes/no too? Score "yes" as `1`, "no" as `0`, take the squared miss. It runs. It gives a perfectly valid number. And then the network learns **painfully slowly**, exactly when it's most wrong.

Picture shoving a heavy box toward home across a floor that turns to thick **mud** the farther out you are. Right when the box is far away and *should* be racing back, the mud is thickest — so each shove barely moves it.
~~~zh
第二道谜题，它从一条很诱人的捷径开始。MSE 这么简单 —— 为什么不也拿它来做是/否呢？把「是」记成 `1`，「否」记成 `0`，取平方后的差。它跑得起来。它给出一个完全合法的数字。然后网络学得**慢得让人难受**，而且正好在它错得最厉害的时候慢。

想象你在地板上把一个重箱子推回家，而你离家越远，地板就变成越稠的**烂泥**。正好在箱子离得很远、*本该*飞奔回来的时候，烂泥最稠 —— 所以你每推一下，它几乎不动。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A person shoving a heavy box toward home across a floor that turns to thick mud far from home. Where the box is farthest and most wrong the mud is thickest, so it barely budges per shove, while near home the floor is clean."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">MSE + sigmoid: the floor turns to mud right where you're most wrong</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">MSE + sigmoid：在你错得最厉害的地方，地面变成烂泥</text><rect x="20" y="96" width="230" height="38" fill="#EDE4D4"/><path d="M20 96 q 20 -8 40 0 q 20 -8 40 0 q 20 -8 40 0 q 20 -8 40 0 q 20 -8 40 0 q 15 -6 30 0" fill="none" stroke="#8a6d3b" stroke-width="2"/><text class="lang-en" x="120" y="124" text-anchor="middle" fill="#6b4f22" font-size="10">THICK mud</text><text class="lang-zh" x="120" y="124" text-anchor="middle" fill="#6b4f22" font-size="10">很稠的烂泥</text><rect x="250" y="96" width="250" height="38" fill="#F2F4F0"/><text class="lang-en" x="400" y="120" text-anchor="middle" fill="#6B645E" font-size="10">clean floor</text><text class="lang-zh" x="400" y="120" text-anchor="middle" fill="#6B645E" font-size="10">干净的地面</text><rect x="34" y="66" width="36" height="30" rx="3" fill="#C99A12" stroke="#9A7208"/><text class="lang-en" x="52" y="86" text-anchor="middle" fill="#fff" font-size="10">box</text><text class="lang-zh" x="52" y="86" text-anchor="middle" fill="#fff" font-size="10">箱子</text><text x="14" y="58" font-size="20">🧍</text><path d="M74 80 l22 0" stroke="#C93B3B" stroke-width="2"/><polygon points="96,80 88,76 88,84" fill="#C93B3B"/><text class="lang-en" x="70" y="152" fill="#C93B3B" font-size="10">far from home &amp; wrong → barely budges 😩</text><text class="lang-zh" x="70" y="152" fill="#C93B3B" font-size="10">离家远又错了 → 几乎不动 😩</text><text class="lang-en" x="320" y="152" fill="#6B645E" font-size="10">push should be strongest here — but it's weakest</text><text class="lang-zh" x="320" y="152" fill="#6B645E" font-size="10">这里的推力本该最强 —— 但它最弱</text></g></svg>
%%%

That's MSE on a yes/no call: the learning push fades to almost nothing at the worst possible moment. **Where the mud picture breaks down:** real mud slows a box *everywhere*, but this slowdown bites in one specific spot — when the model is **confidently wrong**.

Want to work it out yourself first? One rung at a time:

%%% hint
t1: No math — just the picture. On a yes/no call the sigmoid output flattens at the extreme ends. When the model is *confidently* wrong ("0.99 spam!" but it isn't), is it sitting on the flat part of that squash, or the steep part?
t2: MSE's push = the plain miss × the sigmoid's steepness. At p = 0.99 that steepness is only 0.99 × 0.01 ≈ 0.01 — so MSE's push is about a hundredth of the plain miss. Cross-entropy's push IS the plain miss.
t3: MSE + sigmoid multiplies the push by the sigmoid's near-zero slope exactly when the model is confidently wrong, so it barely moves — that's the mud. Cross-entropy cancels that slope so the push stays strong there.
%%%

#### The stall, in three moves
Back to the box in the mud. Here's what the mud actually *is*.

%%% steps
step: the box sits far out — the model says 0.99 "spam" and the truth is "not spam"
why: as wrong as a model can get, and exactly the moment you want the biggest correction
step: the **flat sigmoid slope** — out at 0.99 the squash has gone almost flat: 0.99 × 0.01 ≈ 0.01
why: MSE multiplies its push BY that flatness, therefore the push shrinks to about a hundredth of the plain miss — a whisper instead of a shove. That is the mud, and that is the **stall**: the most-wrong examples get the weakest correction.
step: the cure — cross-entropy is built so that flat factor CANCELS
why: its push is simply the plain miss (≈ 0.99 here), so the push stays loud precisely where MSE went quiet
%%%

Same two scorecards, measured at two moments — mildly wrong, then confidently wrong. Watch the red bar shrink while the green one grows:
~~~zh
这就是 MSE 用在是/否判断上的样子：在最糟的那个时刻，学习的推力淡到几乎没有。**烂泥这张图在哪里不成立：** 真的烂泥*到处*都让箱子变慢，而这个变慢只在一个特定的地方咬人 —— 就是模型**很自信地错了**的时候。

想先自己推一遍吗？一级一级来：

%%% hint
t1: 不用数学 —— 只看图。在一个是/否判断里，sigmoid 的输出在两个极端会变平。当模型*很自信*地错了（喊「0.99 是垃圾邮件！」，但它不是），它是坐在那个挤压的平的部分，还是陡的部分？
t2: MSE 的推力 = 原本的差 × sigmoid 的陡度。在 p = 0.99 处，那个陡度只有 0.99 × 0.01 ≈ 0.01 —— 所以 MSE 的推力大约只有原本的差的百分之一。而 cross-entropy 的推力，就是原本的差本身。
t3: MSE + sigmoid 正好在模型很自信地错了的时候，把推力乘上 sigmoid 那个接近零的斜率，于是它几乎不动 —— 那就是烂泥。Cross-entropy 让那个斜率抵消掉，所以推力在那里依然很强。
%%%

#### 卡住，三步讲完
回到烂泥里的箱子。烂泥到底*是*什么，看这里。

%%% steps
step: 箱子停在很远的地方 —— 模型说 0.99「是垃圾邮件」，而真相是「不是垃圾邮件」
why: 一个模型能错到的极限，而且正是你最想要一次大修正的时刻
step: **sigmoid 的斜率变平了** —— 在 0.99 那里，挤压几乎已经平了：0.99 × 0.01 ≈ 0.01
why: MSE 把它的推力乘上这个平度，所以推力缩到原本的差的大约百分之一 —— 是一声细语，不是一把推。那就是烂泥，也就是那个**卡住**：错得最厉害的样本拿到最弱的修正。
step: 解药 —— cross-entropy 被造成让那个平的因子正好**抵消**
why: 它的推力就是原本的差（这里 ≈ 0.99），所以在 MSE 变哑的地方，推力依然很响
%%%

同样两张记分卡，在两个时刻各量一次 —— 先是只错一点，再是很自信地错。看那根红柱子缩小，绿柱子长大：
~~~

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="Paired bars at two moments. When the model is mildly wrong at p equals 0.6, the MSE push is 0.14 and the cross-entropy push is 0.60. When the model is confidently wrong at p equals 0.99, the MSE push shrinks to about 0.01 while the cross-entropy push grows to 0.99. MSE fades exactly where cross-entropy gets stronger."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The learning push: same two moments, two scorecards</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">学习的推力：同样两个时刻，两张记分卡</text><line x1="62" y1="152" x2="500" y2="152" stroke="#E5DFD6" stroke-width="1.5"/><line x1="62" y1="34" x2="62" y2="152" stroke="#E5DFD6" stroke-width="1.5"/><text class="lang-en" x="30" y="100" text-anchor="middle" fill="#6B645E" font-size="10" transform="rotate(-90 30 100)">learning push</text><text class="lang-zh" x="30" y="100" text-anchor="middle" fill="#6B645E" font-size="10" transform="rotate(-90 30 100)">学习推力</text><rect x="90" y="36" width="12" height="10" fill="#C93B3B"/><text class="lang-en" x="108" y="45" fill="#C93B3B" font-size="10">MSE + sigmoid</text><text class="lang-zh" x="108" y="45" fill="#C93B3B" font-size="10">MSE + sigmoid</text><rect x="230" y="36" width="12" height="10" fill="#2D8B55"/><text class="lang-en" x="248" y="45" fill="#2D8B55" font-size="10">cross-entropy</text><text class="lang-zh" x="248" y="45" fill="#2D8B55" font-size="10">cross-entropy</text><rect x="140" y="138" width="26" height="14" fill="#C93B3B"/><text x="153" y="132" text-anchor="middle" fill="#C93B3B" font-size="10">0.14</text><rect x="176" y="92" width="26" height="60" fill="#2D8B55"/><text x="189" y="86" text-anchor="middle" fill="#2D8B55" font-size="10">0.60</text><text class="lang-en" x="171" y="169" text-anchor="middle" fill="#6B645E" font-size="10">mildly wrong (p=0.6)</text><text class="lang-zh" x="171" y="169" text-anchor="middle" fill="#6B645E" font-size="10">只错一点（p=0.6）</text><rect x="360" y="149" width="26" height="3" fill="#C93B3B"/><text x="373" y="143" text-anchor="middle" fill="#C93B3B" font-size="10">0.01</text><rect x="396" y="53" width="26" height="99" fill="#2D8B55"/><text x="409" y="47" text-anchor="middle" fill="#2D8B55" font-size="10">0.99</text><text class="lang-en" x="391" y="169" text-anchor="middle" fill="#6B645E" font-size="10">confidently WRONG (p=0.99)</text><text class="lang-zh" x="391" y="169" text-anchor="middle" fill="#6B645E" font-size="10">很自信地错了（p=0.99）</text><path d="M166 141 L358 150" fill="none" stroke="#C93B3B" stroke-width="1.2" stroke-dasharray="4,3"/><text class="lang-en" x="262" y="137" text-anchor="middle" fill="#C93B3B" font-size="10">MSE fades to a whisper ↘</text><text class="lang-zh" x="262" y="137" text-anchor="middle" fill="#C93B3B" font-size="10">MSE 弱成一声细语 ↘</text><path d="M202 90 L394 55" fill="none" stroke="#2D8B55" stroke-width="1.2" stroke-dasharray="4,3"/><text class="lang-en" x="298" y="64" text-anchor="middle" fill="#2D8B55" font-size="10">cross-entropy grows to a shout ↗</text><text class="lang-zh" x="298" y="64" text-anchor="middle" fill="#2D8B55" font-size="10">cross-entropy 长成一声大喊 ↗</text><text class="lang-en" x="260" y="188" text-anchor="middle" fill="#6B645E" font-size="10">right where the fix is needed most, one push dies and the other gets louder</text><text class="lang-zh" x="260" y="188" text-anchor="middle" fill="#6B645E" font-size="10">正是最需要修的地方，一个推力死掉，另一个变得更响</text></g></svg>
%%%

%%% insight
Read the two red bars again: **0.14 → 0.01**. MSE's push got *smaller* as the mistake got *worse*. The strange part: MSE + sigmoid is not broken — it computes a fine, correct number. It just whispers the correction when it should be shouting. If that took two reads, you're in good company.
%%%

%%% demo id=push label="reveal both pushes"
predict: the model says 0.99 but the truth is 0. One push is about 0.99, the other about 0.01 — which one belongs to MSE?
code: p=0.99; sig_slope=p*(1-p); mse_push=round((p-0)*sig_slope,4); ce_push=round(p-0,4); print("MSE push",mse_push," cross-entropy push",ce_push)
out: MSE push 0.0098  cross-entropy push 0.99
take: <b>MSE's push nearly vanishes; cross-entropy's stays loud.</b> Confidently wrong at p=0.99, the sigmoid's steepness is only ≈0.01, so MSE's push is the plain miss <em>times</em> that — roughly two orders of magnitude weaker than cross-entropy's push, which is the plain miss itself (0.99). Same mistake, wildly different correction.
%%%

**The neat fix:** swap in binary cross-entropy. Where MSE-plus-sigmoid whispers "meh, barely move", **cross-entropy is the fix** that shouts "you're loudly wrong — fix this *now*". That's the whole reason classifiers train on cross-entropy.

!!! c-info 🔎
<b>Optional (skippable) — one line of "why".</b> With a sigmoid output, the MSE learning signal carries a factor of the sigmoid's slope, which is ≈ 0 in the flat tails (Day 2) — so a confident-wrong unit gets almost no signal.<br>
<b>Why cross-entropy escapes it:</b> it is built so that factor cancels: its signal is simply proportional to the plain error <code>(prediction − target)</code>, which is <em>largest</em> when the model is most wrong.<br>
<b>One bookkeeping note:</b> the pushes above use the tidy <code>½·(pred − target)²</code> form of MSE, which is why no stray factor of 2 shows up; the shrinking-versus-growing story is identical either way.<br>
<b>Where this goes next:</b> you'll see the actual slopes multiplied out on Day 5.
!!!

**Victory lap:** two puzzles down, and both cures fit on a sticky note — *outliers? use MAE.* *Yes/no? never MSE, use cross-entropy.* Now let's take a breath and do something easier and rather satisfying: learn to *read* the score you've been building.
~~~zh
%%% insight
再读一次那两根红柱子：**0.14 → 0.01**。错误变得*更糟*，MSE 的推力反而*更小*。奇怪的地方是：MSE + sigmoid 并没有坏 —— 它算出来的是一个正常、正确的数字。它只是在该大喊的时候，把修正说成了细语。如果这一段你要读两遍，很多人都一样。
%%%

%%% demo id=pushzh label="展开两个推力"
predict: 模型说 0.99，而真相是 0。一个推力大约是 0.99，另一个大约是 0.01 —— 哪一个属于 MSE？
code: p=0.99; sig_slope=p*(1-p); mse_push=round((p-0)*sig_slope,4); ce_push=round(p-0,4); print("MSE push",mse_push," cross-entropy push",ce_push)
out: MSE push 0.0098  cross-entropy push 0.99
take: <b>MSE 的推力几乎消失；cross-entropy 的推力依然很响。</b>在 p=0.99 上很自信地错了，sigmoid 的陡度只有 ≈0.01，所以 MSE 的推力是原本的差<em>乘上</em>它 —— 大约比 cross-entropy 的推力弱一百倍，而后者就是原本的差本身（0.99）。同一个错误，修正差得离谱。
%%%

**干净的修法：**换成 binary cross-entropy。MSE 加 sigmoid 在那里细声说「算了，动一点点就好」，而 **cross-entropy 就是那个解法**，它大喊「你错得很响 —— *现在*就修」。这就是分类器用 cross-entropy 训练的全部理由。

!!! c-info 🔎
<b>可跳过 —— 一行「为什么」。</b>当输出是一个 sigmoid 时，MSE 的学习信号里带着一个 sigmoid 斜率的因子，而这个斜率在平坦的两端 ≈ 0（第 2 天讲过）—— 所以一个又自信又错的单元几乎拿不到信号。<br>
<b>cross-entropy 为什么躲开了它：</b>它被造成让那个因子正好抵消：它的信号就正比于原本的误差 <code>(prediction − target)</code>，而这个误差在模型错得最厉害时<em>最大</em>。<br>
<b>一个记账上的说明：</b>上面那些推力用的是 MSE 的整洁写法 <code>½·(pred − target)²</code>，所以没有多出来的因子 2；不管用哪种写法，「一个缩小、一个长大」的故事完全一样。<br>
<b>这条线接下来去哪：</b>第 5 天你会看到那些斜率被真的乘出来。
!!!

**跑一圈庆祝：**两道谜题拿下，两个解药都能写在一张小贴纸上 —— *有 outlier？用 MAE。* *是/否？永远别用 MSE，用 cross-entropy。* 现在我们喘口气，做点更轻松、也挺舒服的事：学会*读*你一直在搭的这个分数。
~~~

@@@ concept id=c8 zh_tag="读分数" tag="Read the score" zh_title="像老手一样读分数 —— 它真的在起作用吗？" title="Reading the score like a pro — is it actually working?" zh_gotit="懂了怎么读" gotit="Got how to read it"
Breather time — and a genuinely useful skill. You've built three scorecards. Now: how do you *look* at a loss number and know whether your model is doing well? This is the part that makes you feel like an insider, because there's a trick to it.

Picture a stack of **golf cards taped to the fridge** — one per week for a whole season. Nobody stares at a single card and asks "is 46 a good number?" You look at the *stack* and ask one thing: **is the line going down?**
~~~zh
喘口气的时间 —— 也是一个真正有用的技能。你已经搭了三张记分卡。现在的问题是：你怎么*看*一个 loss 数字，就知道你的模型做得好不好？这一段会让你觉得自己是个内行，因为它有一个门道。

想象**一叠贴在冰箱上的高尔夫记分卡** —— 一整个赛季，每周一张。没有人会盯着单独一张卡问「46 是个好数字吗？」你会看那**一整叠**，然后只问一件事：**这条线在往下走吗？**
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A fridge door with a season of golf cards taped to it, showing weekly scores of 46, 41, 37, 34 and 33. A line drawn through them slopes downward. A note says nobody asks whether one card is good; you ask whether the line is going down."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A season on the fridge — you read the trend, not one card</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">冰箱上的一整季 —— 你读的是趋势，不是某一张卡</text><rect x="26" y="28" width="300" height="128" rx="10" fill="#F4F5F3" stroke="#C4C9C2" stroke-width="2"/><rect x="34" y="60" width="10" height="34" rx="4" fill="#B8AEA2"/><g font-size="10"><rect x="58" y="44" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="80" y="58" text-anchor="middle" fill="#6B645E">wk 1</text><text x="80" y="72" text-anchor="middle" fill="#C93B3B">46</text><rect x="112" y="52" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="134" y="66" text-anchor="middle" fill="#6B645E">wk 2</text><text x="134" y="80" text-anchor="middle" fill="#C99A12">41</text><rect x="166" y="62" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="188" y="76" text-anchor="middle" fill="#6B645E">wk 3</text><text x="188" y="90" text-anchor="middle" fill="#C99A12">37</text><rect x="220" y="74" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="242" y="88" text-anchor="middle" fill="#6B645E">wk 4</text><text x="242" y="102" text-anchor="middle" fill="#2D8B55">34</text><rect x="266" y="82" width="44" height="34" rx="3" fill="#FDFCF9" stroke="#C9B77A"/><text x="288" y="96" text-anchor="middle" fill="#6B645E">wk 5</text><text x="288" y="110" text-anchor="middle" fill="#2D8B55">33</text></g><path d="M80 46 L134 56 L188 66 L242 78 L288 86" fill="none" stroke="#2D8B55" stroke-width="2" stroke-dasharray="5,3"/><text class="lang-en" x="176" y="136" text-anchor="middle" fill="#1a5c38" font-size="10">the line goes DOWN — you're getting better ⛳</text><text class="lang-zh" x="176" y="136" text-anchor="middle" fill="#1a5c38" font-size="10">这条线在往下 —— 你在变好 ⛳</text><text class="lang-en" x="344" y="60" fill="#6B645E">one card alone:</text><text class="lang-zh" x="344" y="60" fill="#6B645E">只看一张卡：</text><text class="lang-en" x="344" y="76" fill="#C93B3B">"is 46 good?" 🤷</text><text class="lang-zh" x="344" y="76" fill="#C93B3B">「46 算好吗？」🤷</text><text class="lang-en" x="344" y="104" fill="#6B645E">the stack:</text><text class="lang-zh" x="344" y="104" fill="#6B645E">一整叠：</text><text class="lang-en" x="344" y="120" fill="#2D8B55">"it's dropping!" 🎉</text><text class="lang-zh" x="344" y="120" fill="#2D8B55">「它在往下掉！」🎉</text></g></svg>
%%%

**What the fridge gets right:** the direction over time is the whole story, and one number on its own tells you almost nothing. **Where it breaks down:** in golf you can eventually shoot a perfect round, but a loss usually flattens out somewhere above zero — and a *perfectly* zero training loss is often bad news (a story for a later day).

%%% insight
This is why "is my loss good?" is the wrong question, and it trips up nearly everyone. `0.42` is fantastic on one task and terrible on another — the units and the number of classes change everything. **Lower is better**, always, but *lower than what?* You're two minutes from knowing exactly how to answer that.
%%%

#### Four curves — spot the healthy one
Every training run draws a line: the loss after each step. Four shapes, and you can name all four after this one picture.
~~~zh
**冰箱这张图对在哪里：** 随时间的方向就是整个故事，而一个数字单独摆着，几乎什么都说明不了。**它在哪里不成立：** 打高尔夫你最后真有可能打出完美的一轮。而 loss 通常会在零的上方某处变平。而且训练 loss *正好*等于零常常是坏消息（那是以后某一天的故事）。

%%% insight
这就是为什么「我的 loss 好不好？」是个错问题，而且它几乎绊倒所有人。`0.42` 在一个任务上很棒，在另一个任务上很糟 —— 单位和类别的数量会改变一切。**越小越好**，这永远成立，但是*比什么小？* 你离知道确切答案，只剩两分钟。
%%%

#### 四条曲线 —— 找出健康的那一条
每一次训练都会画一条线：每一步之后的 loss。四种形状，看完这一张图你就能把四种都叫出名字。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Four training-loss curves side by side. First: a curve that drops steeply then flattens, a healthy run. Second: a flat line that never moves, nothing is learning. Third: a jagged line that trends down but jumps around, the step size is too big. Fourth: a curve that falls and then shoots off the top into NaN."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Four loss curves — which one is a healthy run?</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">四条 loss 曲线 —— 哪一条是健康的训练？</text><rect x="18" y="30" width="112" height="84" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><line x1="24" y1="108" x2="124" y2="108" stroke="#E5DFD6"/><path d="M26 40 C 52 84, 82 102, 122 105" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text class="lang-en" x="74" y="128" text-anchor="middle" fill="#3A342E">drops → flattens</text><text class="lang-zh" x="74" y="128" text-anchor="middle" fill="#3A342E">先掉 → 后变平</text><text class="lang-en" x="74" y="142" text-anchor="middle" fill="#2D8B55">✓ healthy</text><text class="lang-zh" x="74" y="142" text-anchor="middle" fill="#2D8B55">✓ 健康</text><rect x="146" y="30" width="112" height="84" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><line x1="152" y1="108" x2="252" y2="108" stroke="#E5DFD6"/><path d="M152 64 L252 64" fill="none" stroke="#C93B3B" stroke-width="2.5"/><text class="lang-en" x="202" y="128" text-anchor="middle" fill="#3A342E">never moves</text><text class="lang-zh" x="202" y="128" text-anchor="middle" fill="#3A342E">一直不动</text><text class="lang-en" x="202" y="142" text-anchor="middle" fill="#C93B3B">✗ not learning</text><text class="lang-zh" x="202" y="142" text-anchor="middle" fill="#C93B3B">✗ 没在学</text><rect x="274" y="30" width="112" height="84" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><line x1="280" y1="108" x2="380" y2="108" stroke="#E5DFD6"/><path d="M280 42 L294 74 L306 50 L320 86 L334 58 L348 94 L362 70 L378 98" fill="none" stroke="#C99A12" stroke-width="2.5"/><text class="lang-en" x="330" y="128" text-anchor="middle" fill="#3A342E">down but jumpy</text><text class="lang-zh" x="330" y="128" text-anchor="middle" fill="#3A342E">往下但很跳</text><text class="lang-en" x="330" y="142" text-anchor="middle" fill="#9A7208">⚠ step too big</text><text class="lang-zh" x="330" y="142" text-anchor="middle" fill="#9A7208">⚠ 步子太大</text><rect x="402" y="30" width="112" height="84" rx="6" fill="#FDFCF9" stroke="#E5DFD6"/><line x1="408" y1="108" x2="508" y2="108" stroke="#E5DFD6"/><path d="M408 44 C 428 72, 442 90, 452 96" fill="none" stroke="#C93B3B" stroke-width="2.5"/><path d="M456 96 L456 36" fill="none" stroke="#C93B3B" stroke-width="2" stroke-dasharray="4,3"/><text x="482" y="44" fill="#C93B3B" font-size="9">nan 💥</text><text class="lang-en" x="458" y="128" text-anchor="middle" fill="#3A342E">falls, then 💥</text><text class="lang-zh" x="458" y="128" text-anchor="middle" fill="#3A342E">先掉，然后 💥</text><text class="lang-en" x="458" y="142" text-anchor="middle" fill="#C93B3B">✗ next puzzle!</text><text class="lang-zh" x="458" y="142" text-anchor="middle" fill="#C93B3B">✗ 下一道谜题！</text></g></svg>
%%%

%%% steps
step: the healthy shape — a steep drop, then a slow flattening
why: the easy mistakes get fixed first, therefore progress naturally slows down. That's the shape you want on your fridge.
step: dead flat from step one → nothing is learning
why: often the wrong loss for the output shape, or a step size so tiny nothing moves
step: jagged but trending down → alive, just clumsy
why: each step is overshooting a bit. Usually harmless; smaller steps smooth it out.
step: falls, then leaps to `nan` → the run has poisoned itself
why: that one is coming up next, and after the next unit you'll fix it in one line
%%%

#### The 5-second sanity check — what does pure guessing score?
Here's the insider trick, and it's *lovely*. Before trusting any run, ask: **what would a model that knows nothing score?** For cross-entropy that number is easy — a pure guesser spreads its pizza evenly, so its loss is `ln K` for `K` choices. Drag the dial and watch that bar move:
~~~zh
%%% steps
step: 健康的形状 —— 先陡陡地掉，然后慢慢变平
why: 简单的错误先被修好，所以进展自然会变慢。这就是你想贴在冰箱上的形状。
step: 从第一步就是死平的 → 什么都没在学
why: 常见原因是 loss 和输出的形状配错了，或者步子小到什么都动不了
step: 锯齿状但整体在往下 → 活着，只是笨手笨脚
why: 每一步都稍微迈过头了。通常没大害；把步子改小就平顺了。
step: 先往下掉，然后跳到 `nan` → 这次训练把自己毒死了
why: 那一条马上就讲，看完下一个单元你就能用一行把它修好
%%%

#### 5 秒的常识检查 —— 纯瞎猜能得多少分？
内行的门道来了，而且它*很可爱*。在相信任何一次训练之前，先问一句：**一个什么都不懂的模型会得多少分？** 对 cross-entropy 来说这个数字很好算 —— 一个纯瞎猜的模型会把披萨切得一样平均，所以在 `K` 个选项时它的 loss 是 `ln K`。拖那个转盘，看那根柱子怎么动：
~~~

%%% svg
<svg id="cl-svg" viewBox="0 0 520 156" role="img" aria-label="Interactive chance-level meter. Drag the number of classes and the marker moves along a loss axis from 0 to 5, showing the loss a pure guesser would score. Left of the marker is the learning zone; right of it means the model is still guessing."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The 5-second check: where does pure guessing land?</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">5 秒检查：纯瞎猜会落在哪里？</text><rect id="cl-good" x="70" y="58" width="55" height="28" fill="#EAF5EE" stroke="#2D8B55"/><rect id="cl-bad" x="125" y="58" width="345" height="28" fill="#FDECEC" stroke="#E0A9A0"/><line id="cl-mark" x1="125" y1="44" x2="125" y2="100" stroke="#C99A12" stroke-width="2.5"/><text id="cl-lab" x="125" y="38" text-anchor="middle" fill="#9A7208" font-size="10">0.69</text><line x1="70" y1="100" x2="470" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><text x="70" y="116" fill="#6B645E" font-size="10">0</text><text x="470" y="116" text-anchor="end" fill="#6B645E" font-size="10">5</text><text class="lang-en" x="150" y="134" text-anchor="middle" fill="#2D8B55" font-size="10">← real learning lives here</text><text class="lang-zh" x="150" y="134" text-anchor="middle" fill="#2D8B55" font-size="10">← 真的在学，在这一边</text><text class="lang-en" x="400" y="134" text-anchor="middle" fill="#C93B3B" font-size="10">still guessing →</text><text class="lang-zh" x="400" y="134" text-anchor="middle" fill="#C93B3B" font-size="10">还在瞎猜 →</text><text class="lang-en" x="270" y="150" text-anchor="middle" fill="#6B645E" font-size="10">cross-entropy loss value →</text><text class="lang-zh" x="270" y="150" text-anchor="middle" fill="#6B645E" font-size="10">cross-entropy 的 loss 值 →</text></g></svg>
<div style="margin-top:8px">
<div style="display:flex;justify-content:space-between;font-size:.9em;color:#5A544E"><span><span class="lang-en">How many choices? </span><span class="lang-zh">有几个选项？</span><b id="cl-k">2</b><span class="lang-en"> (yes/no → 2, digits → 10)</span><span class="lang-zh">（是/否 → 2，数字 → 10）</span></span><span><span class="lang-en">drag me →</span><span class="lang-zh">拖我 →</span></span></div>
<input id="cl-r" type="range" min="2" max="100" value="2" step="1" aria-label="number of classes" style="width:100%;margin:6px 0;accent-color:#C99A12">
<div id="cl-out" style="text-align:center;font-size:1.02em;color:#2C2A28"><span class="lang-en">A pure guesser scores <b>0.69</b> &nbsp;<span style="color:#2D8B55">so a yes/no model stuck at 0.69 has learned nothing yet</span></span><span class="lang-zh">一个纯瞎猜的模型得 <b>0.69</b> &nbsp;<span style="color:#2D8B55">所以一个卡在 0.69 的是/否模型还什么都没学到</span></span></div>
</div>
<script>(function(){
  var s=document.getElementById('cl-r');if(!s)return;
  var good=document.getElementById('cl-good'),bad=document.getElementById('cl-bad'),
      mark=document.getElementById('cl-mark'),lab=document.getElementById('cl-lab'),
      ktxt=document.getElementById('cl-k'),out=document.getElementById('cl-out');
  var X0=70,X1=470,LMAX=5;
  function paint(){
    var k=Math.max(2,+s.value), chance=Math.log(k);
    var x=X0+Math.min(chance,LMAX)/LMAX*(X1-X0);
    if(good)good.setAttribute('width',Math.max(1,x-X0).toFixed(1));
    if(bad){bad.setAttribute('x',x.toFixed(1));bad.setAttribute('width',Math.max(1,X1-x).toFixed(1));}
    if(mark){mark.setAttribute('x1',x.toFixed(1));mark.setAttribute('x2',x.toFixed(1));}
    if(lab){lab.setAttribute('x',Math.min(Math.max(x,84),456).toFixed(1));lab.textContent=chance.toFixed(2);}
    if(ktxt)ktxt.textContent=k;
    var word=k<=2?'a yes/no model stuck at 0.69 has learned nothing yet':(k<=12?'below this line, your model really is picking up the pattern':'with this many choices, even a good model starts high — compare to the line, not to 0');
    var wordZh=k<=2?'一个卡在 0.69 的是/否模型还什么都没学到':(k<=12?'掉到这条线下面，你的模型才真的在抓规律':'选项这么多时，连好模型一开始也很高 —— 跟这条线比，不是跟 0 比');
    if(out)out.innerHTML='<span class="lang-en">A pure guesser scores <b>'+chance.toFixed(2)+'</b> &nbsp;<span style="color:#2D8B55">so '+word+'</span></span>'+'<span class="lang-zh">一个纯瞎猜的模型得 <b>'+chance.toFixed(2)+'</b> &nbsp;<span style="color:#2D8B55">所以'+wordZh+'</span></span>';
  }
  s.addEventListener('input',paint);paint();
})();</script>
%%%

%%% demo id=chance label="reveal the guessing scores"
predict: a 10-class digit model that just guesses at random — is its cross-entropy about 0, about 0.5, or about 2.3?
code: [round(-np.log(1/k),3) for k in (2, 10, 1000)]   # a pure guesser gives every class 1/k
out: [0.693, 2.303, 6.908]
take: <b>Chance level = ln K.</b> Two choices → 0.693, ten → 2.303, a thousand → 6.908. So a 10-class run sitting at 2.3 has learned <em>nothing</em>, while 2.3 on a 1000-class problem is real progress. Numbers only mean something next to a baseline.
%%%

There's a matching baseline for numbers, too: if you always guess the average target, MSE lands on the **spread of your data**. Beat that and your model is genuinely adding something.

So the habit is short and sweet: **the goal is to make loss go down**, and you judge the number by comparing it to a baseline, never to zero.

**Victory lap:** you can now open somebody's training log, glance at the curve and the baseline, and say something true about it in five seconds. That's a real engineer's move. One puzzle left — and it's the one hiding in that fourth curve.
~~~zh
%%% demo id=chancezh label="展开瞎猜的分数"
predict: 一个 10 类的数字模型，纯随机瞎猜 —— 它的 cross-entropy 大约是 0、大约 0.5，还是大约 2.3？
code: [round(-np.log(1/k),3) for k in (2, 10, 1000)]   # 一个纯瞎猜的模型给每一类都是 1/k
out: [0.693, 2.303, 6.908]
take: <b>瞎猜的水平 = ln K。</b>两个选项 → 0.693，十个 → 2.303，一千个 → 6.908。所以一次 10 类的训练停在 2.3，说明它<em>什么都没学到</em>；而在 1000 类的问题上，2.3 是真的进展。数字只有放在一个基准旁边才有意义。
%%%

猜数字也有配套的基准：如果你每次都猜 target 的平均值，MSE 会落在**你的数据散开得有多宽**上。比这个更好，你的模型才真的加了点东西。

所以这个习惯很短也很好记：**目标就是让 loss 往下走**，而你判断这个数字，是拿它和一个基准比，永远不是和零比。

**跑一圈庆祝：**你现在能打开别人的训练日志，扫一眼曲线和基准，在五秒钟里说出一句真话。这是真工程师的动作。还剩一道谜题 —— 它就藏在第四条曲线里。
~~~

@@@ concept id=c9 zh_tag="log(0) 炸掉" tag="log(0) blow-up" zh_title="谜题 —— cross-entropy 什么时候炸到无穷大" title="Puzzle — when cross-entropy explodes to infinity" zh_gotit="懂了那个裁剪" gotit="Got the clip"
Last puzzle, and it's the sharp edge of cross-entropy — the one behind that fourth curve. What if the model is *absolutely certain* and wrong: it declares "0% chance this is spam" and it **is** spam?

Picture a wall you're told never to touch, with a meter that charges more the closer your bumper creeps. A foot away: a little. Six inches: double. Three inches: double again. The charge climbs with **no ceiling** — and if you actually kiss the wall, the meter refuses to show a number at all.
~~~zh
最后一道谜题，它是 cross-entropy 的那道锋利的边 —— 就是第四条曲线背后的那个。如果模型*绝对肯定*，而且错了，会怎样呢：它宣布「这封邮件是垃圾邮件的机会是 0%」，而它**就是**垃圾邮件。

想象一面别人叫你千万别碰的墙，墙上装了一个计费表，你的保险杠越靠近，它收的钱就越多。离一尺：收一点。离六寸：翻倍。离三寸：再翻倍。这个收费一路往上爬，**没有上限** —— 而如果你真的贴上了那面墙，计费表干脆拒绝显示任何数字。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A car bumper creeping toward a wall you must never touch, with a charge meter. A foot away costs a little, six inches double, three inches double again, climbing with no ceiling. Kissing the wall shows no number at all. A tiny epsilon margin keeps the bumper off the wall."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The closer to the wall, the bigger the charge — with no ceiling</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">离墙越近，收费越大 —— 而且没有上限</text><rect x="468" y="30" width="22" height="120" fill="#D9CFC0" stroke="#8a7d68"/><text class="lang-en" x="479" y="96" text-anchor="middle" fill="#6b4f22" font-size="10" transform="rotate(90 479 96)">the WALL — never touch</text><text class="lang-zh" x="479" y="96" text-anchor="middle" fill="#6b4f22" font-size="10" transform="rotate(90 479 96)">那面墙 —— 千万别碰</text><rect x="60" y="70" width="56" height="30" rx="4" fill="#2A7B9B"/><text x="88" y="90" text-anchor="middle" fill="#fff" font-size="10">🚗</text><path d="M116 85 l40 0" stroke="#6B645E" stroke-width="1.5" stroke-dasharray="4,3"/><polygon points="156,85 148,81 148,89" fill="#6B645E"/><rect x="110" y="120" width="36" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="128" y="133" text-anchor="middle" fill="#2D8B55">$1</text><rect x="250" y="120" width="36" height="18" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="268" y="133" text-anchor="middle" fill="#9A7208">$2</text><rect x="370" y="120" width="40" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text x="390" y="133" text-anchor="middle" fill="#C93B3B">$4…💥</text><text class="lang-en" x="128" y="156" text-anchor="middle" fill="#6B645E" font-size="9">a foot away</text><text class="lang-zh" x="128" y="156" text-anchor="middle" fill="#6B645E" font-size="9">离一尺</text><text class="lang-en" x="268" y="156" text-anchor="middle" fill="#6B645E" font-size="9">6 inches</text><text class="lang-zh" x="268" y="156" text-anchor="middle" fill="#6B645E" font-size="9">离六寸</text><text class="lang-en" x="390" y="156" text-anchor="middle" fill="#6B645E" font-size="9">3 inches — no ceiling</text><text class="lang-zh" x="390" y="156" text-anchor="middle" fill="#6B645E" font-size="9">离三寸 —— 没有上限</text><line x1="456" y1="30" x2="456" y2="150" stroke="#2D8B55" stroke-width="1.5" stroke-dasharray="4,3"/><text class="lang-en" x="448" y="52" text-anchor="end" fill="#2D8B55" font-size="9">stop here (ε)</text><text class="lang-zh" x="448" y="52" text-anchor="end" fill="#2D8B55" font-size="9">停在这里（ε）</text><text class="lang-en" x="448" y="64" text-anchor="end" fill="#2D8B55" font-size="9">tiny safe margin</text><text class="lang-zh" x="448" y="64" text-anchor="end" fill="#2D8B55" font-size="9">很小的安全余量</text></g></svg>
%%%

That runaway bill is exactly `−log` as the probability on the true answer slides toward `0`. **Where the meter breaks down:** a real meter has a top price it can't pass. The math has no cap — right at the wall the penalty is a true `+∞`.

#### One certain mistake, three quick beats

%%% steps
step: the model kisses the wall — it gives the true answer a probability of exactly 0.0
why: not "very small" — actually 0.0, because floating-point rounding flattened a tiny number
step: the blow-up — the score becomes `-log(0)`, which is `+∞`, and infinity soon turns into a **NaN**
why: the log's climb has no ceiling, so this really is infinite
step: the spread — that one NaN leaks into every step after it
why: nothing crashes, no error appears. The numbers just quietly turn to garbage.
%%%

Here's the cliff, plus the little green line that saves you from it:
~~~zh
那张停不下来的账单，正是真答案上的概率滑向 `0` 时的 `−log`。**计费表这张图在哪里不成立：** 真的计费表有一个它超不过的最高价。而数学没有上限 —— 正好贴在墙上的时候，惩罚是一个真正的 `+∞`。

#### 一个笃定的错误，三个快节拍

%%% steps
step: 模型贴上了墙 —— 它给真答案的概率正好是 0.0
why: 不是「非常小」—— 是真的 0.0，因为浮点数的四舍五入把一个极小的数压平了
step: 炸掉 —— 分数变成 `-log(0)`，也就是 `+∞`，而无穷很快会变成一个 **NaN**
why: log 的上爬没有天花板，所以这真的是无穷
step: 扩散 —— 那一个 NaN 会漏进它后面的每一步
why: 什么都不会崩，也不会跳出报错。那些数字只是悄悄变成垃圾。
%%%

那道悬崖在这里，还有那条救你的小绿线：
~~~

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="The minus-log curve shoots to infinity as probability approaches zero; clamping the probability just above zero with a tiny epsilon keeps the loss finite"><g font-family="monospace" font-size="11"><line x1="70" y1="130" x2="470" y2="130" stroke="#E5DFD6" stroke-width="1.5"/><line x1="90" y1="20" x2="90" y2="130" stroke="#E5DFD6" stroke-width="1"/><path d="M96 24 C 120 60, 170 112, 260 124 C 350 130, 420 131, 460 131" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><text class="lang-en" x="150" y="30" fill="#C93B3B">p → 0 : −log(p) → +∞  (then 💥 NaN)</text><text class="lang-zh" x="150" y="30" fill="#C93B3B">p → 0 ：−log(p) → +∞（然后 💥 NaN）</text><line x1="112" y1="20" x2="112" y2="135" stroke="#2D8B55" stroke-width="1.5" stroke-dasharray="4,3"/><text class="lang-en" x="118" y="150" fill="#2D8B55">clamp at ε (tiny)</text><text class="lang-zh" x="118" y="150" fill="#2D8B55">在 ε 处夹住（很小）</text><text class="lang-en" x="330" y="150" fill="#6B645E">probability of correct answer →</text><text class="lang-zh" x="330" y="150" fill="#6B645E">正确答案的概率 →</text></g></svg>
%%%

A [[NaN||"Not a Number" — what a computer makes from an undefined operation like ∞ − ∞. Once one appears it spreads through every later step.]] is the villain, and it never announces itself.

%%% insight
Picture the morning after: you left a model training overnight, you come back, and the loss column reads `nan, nan, nan…` all the way down. No error message. Everyone hits this once — and after today you'll recognise it in five seconds.
%%%

#### The neat fix — keep the bumper off the wall
Before taking the log, nudge every probability into a tiny safe margin — say `[0.0000001, 0.9999999]`.

That margin is [[epsilon||A tiny number (like 1e-7) used as a safety margin, so you never take log(0).]] (`ε`), and squeezing the number into it — also called a **clamp** — is [[epsilon clipping||Clamping every predicted probability into a tiny range like [ε, 1−ε] before the log, so cross-entropy can never blow up to infinity.]]. Now `−log` sees a big *finite* number instead of infinity: a fair punishment instead of a poisoned run.

%%% demo id=clip label="reveal the clamped score"
predict: a certain-and-wrong p = 0 would score infinity. Clamped to 0.0000001, does it become something like 2, 16, or a million?
code: p=0.0; print("clamped",round(-np.log(max(p,1e-7)),3))  # clamp p into [1e-7, 1-1e-7]
out: clamped 16.118
take: <b>Epsilon clipping keeps the loss finite.</b> A confident-wrong <code>p=0</code> gives <code>−log(0)=+∞</code>, which later leaks out as a NaN. Clamped to <code>1e-7</code> it becomes <code>≈16.1</code> — a large, honest penalty training can actually use. Crisis averted with one <code>max</code>.
%%%

!!! c-info 🔎
<b>Optional (skippable):</b> real libraries go one better — they compute the loss straight from the raw logits (the "from-logits" trick) so <code>log(0)</code> never happens at all. The idea you now hold — keep the probability off the cliff edge — is exactly its spirit.
!!!

**Victory lap:** that's **three puzzles, three cures**, all yours — and you can say each cure in one breath. Outliers → MAE. Yes/no → cross-entropy. `p = 0` → clamp with ε. One honest thing left: what a score simply *can't* do.
~~~zh
[[NaN||「Not a Number」，意思是「不是一个数字」—— 计算机遇到像 ∞ − ∞ 这种没有定义的运算时造出来的东西。一旦出现一个，它就会扩散到后面的每一步。]] 就是那个坏蛋，而它从不自报家门。

%%% insight
想象第二天早上：你让一个模型练了一整夜，回来一看，loss 那一列从上到下写着 `nan, nan, nan……`。没有任何报错。每个人都会撞上这个一次 —— 而学完今天，你五秒钟就能认出它。
%%%

#### 干净的修法 —— 让保险杠别贴墙
在取 log 之前，把每一个概率推进一个很小的安全余量里 —— 比如 `[0.0000001, 0.9999999]`。

那个余量叫 [[epsilon||一个很小的数字（比如 1e-7），用来当安全余量，好让你永远不会去取 log(0)。]]（`ε`）。把数字挤进这个余量里，也叫做一次 **clamp（夹住）**。这一步就是 [[epsilon clipping||在取 log 之前，把每一个预测概率夹进一个很小的范围。比如 [ε, 1−ε]。这样 cross-entropy 永远炸不到无穷。]]。现在 `−log` 看到的是一个很大但*有限*的数字，而不是无穷：一个公平的惩罚，而不是一次被毒死的训练。

%%% demo id=clipzh label="展开被夹住的分数"
predict: 一个又笃定又错的 p = 0 会得到无穷。夹到 0.0000001 之后，它会变成大约 2、大约 16，还是一百万？
code: p=0.0; print("clamped",round(-np.log(max(p,1e-7)),3))  # 把 p 夹进 [1e-7, 1-1e-7]
out: clamped 16.118
take: <b>Epsilon clipping 让 loss 保持有限。</b>一个又自信又错的 <code>p=0</code> 给出 <code>−log(0)=+∞</code>，之后它会以 NaN 的形式漏出来。夹到 <code>1e-7</code> 之后，它变成 <code>≈16.1</code> —— 一个很大但很老实的惩罚，训练真的用得上。一个 <code>max</code> 就化解了危机。
%%%

!!! c-info 🔎
<b>可跳过：</b>真实的库还更进一步 —— 它们直接从原始的 logits 算 loss（那个「from-logits」的做法），所以 <code>log(0)</code> 根本不会发生。你现在手上这个想法 —— 让概率别贴到悬崖边 —— 正是它的精神。
!!!

**跑一圈庆祝：三道谜题，三个解药**，全都是你的了 —— 而且每一个解药你都能一口气说完。有 outlier → 用 MAE。是/否 → 用 cross-entropy。`p = 0` → 用 ε 夹住。还剩一件老实事：一个分数根本*做不到*什么。
~~~

@@@ concept id=c10 zh_tag="它的边界" tag="The limit" zh_title="分数只负责评判 —— 它不负责修（那是另一份工作）" title="A score judges — it doesn't fix (that's a separate job)" zh_gotit="懂了它的边界" gotit="Got the limit"
Let's end on the honest truth about what today's tool *can't* do — because knowing the edge of a tool is what separates a beginner from someone who really gets it.

Picture a **bathroom scale**. It shows a number, and a lower number might be your goal. But the scale itself has never once made anyone lighter. It only *reports*. The eating and the running do the changing.
~~~zh
我们用一句老实话来收尾：今天这个工具*做不到*什么 —— 因为知道一个工具的边界，正是新手和真懂的人之间的分界线。

想象一台**体重秤**。它显示一个数字，而更小的数字可能就是你的目标。但这台秤本身，从来没有让任何一个人变轻过。它只负责*报数*。真正带来改变的是吃饭和跑步。
~~~

%%% svg
<svg viewBox="0 0 520 152" role="img" aria-label="A bathroom scale showing a weight number with a goal arrow pointing down. A note says the scale only reports the number and has never once made anyone lighter; the eating and running do the changing."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The scale reports the number — it never made anyone lighter</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">体重秤只报数字 —— 它从没让任何人变轻</text><rect x="70" y="38" width="150" height="90" rx="14" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><rect x="96" y="56" width="98" height="40" rx="6" fill="#FDFCF9" stroke="#B3A6D4"/><text x="145" y="84" text-anchor="middle" fill="#5E5191" font-size="18">0.42</text><text class="lang-en" x="145" y="116" text-anchor="middle" fill="#6B645E" font-size="10">bathroom scale = the loss</text><text class="lang-zh" x="145" y="116" text-anchor="middle" fill="#6B645E" font-size="10">体重秤 = loss</text><path d="M250 70 l40 0" stroke="#6B645E" stroke-width="1.5"/><polygon points="290,70 282,66 282,74" fill="#6B645E"/><text class="lang-en" x="270" y="62" text-anchor="middle" fill="#2D8B55" font-size="10">goal: ↓ lower</text><text class="lang-zh" x="270" y="62" text-anchor="middle" fill="#2D8B55" font-size="10">目标：↓ 更小</text><text class="lang-en" x="330" y="58" fill="#6B645E">it only REPORTS.</text><text class="lang-zh" x="330" y="58" fill="#6B645E">它只负责报数。</text><text class="lang-en" x="330" y="78" fill="#C93B3B">it never moves a weight.</text><text class="lang-zh" x="330" y="78" fill="#C93B3B">它从不动任何一个 weight。</text><text class="lang-en" x="330" y="104" fill="#2D8B55">the eating &amp; running</text><text class="lang-zh" x="330" y="104" fill="#2D8B55">吃饭和跑步</text><text class="lang-en" x="330" y="120" fill="#2D8B55">(a separate step)</text><text class="lang-zh" x="330" y="120" fill="#2D8B55">（另一个步骤）</text><text class="lang-en" x="330" y="136" fill="#2D8B55">do the changing</text><text class="lang-zh" x="330" y="136" fill="#2D8B55">才带来改变</text></g></svg>
%%%

**Where the scale breaks down:** a scale is silent about *how* to change, but a loss is a little more helpful. Because it's smooth, it also hands over a **slope** — a "which way is downhill" hint.

#### Step onto the scale — four beats
Let's walk the analogy all the way to the edge, one beat at a time. Keep that scale picture in your head: today's loss reads `0.42`.

%%% steps
step: **the reading** — the scale shows 68.4 kg; your loss shows 0.42
why: one honest number for "where you stand right now". That is today's entire job: (prediction, target) → 0.42.
step: **the tilt** — stand off-centre and the needle leans; a smooth loss leans too, and that lean is a slope
why: it whispers "downhill is *this* way" — a direction. A real bathroom scale can't even do this much, so the loss is already one step ahead of the analogy.
step: **the full stop** — the scale has never once made anyone lighter, and neither has a loss
why: it **only scores**, and it **does not change any weight**. Not one, not ever. Reading 0.42 all day long changes nothing.
step: **the mover** — a separate step reads that slope and actually edits the weights: the **optimizer**
why: the eating and the running of our analogy. Tomorrow you learn how that slope is computed (**gradients and backpropagation**); the step that spends it — the optimizer itself — gets its own day soon after.
%%%

So the day ends with a clean seam: today's tool judges, and a different tool fixes.
~~~zh
**体重秤这张图在哪里不成立：** 秤对*怎么*改变一句话都不说，而 loss 会稍微多帮一点。因为它是平滑的，它还顺手给出一个**斜率** —— 一个「哪边是下坡」的提示。

#### 站上体重秤 —— 四个节拍
我们把这个比喻一直走到它的边界，一拍一拍来。脑子里留着那台秤的画面：今天的 loss 读数是 `0.42`。

%%% steps
step: **读数** —— 秤显示 68.4 公斤；你的 loss 显示 0.42
why: 一个诚实的数字，说明「你现在站在哪里」。这就是今天全部的活：(prediction, target) → 0.42。
step: **倾斜** —— 你站偏一点，指针会歪；一个平滑的 loss 也会歪，而那个歪就是一个斜率
why: 它悄悄说「下坡在*这*一边」—— 一个方向。真的体重秤连这一点都做不到，所以 loss 已经比这个比喻多走了一步。
step: **彻底停住** —— 秤从没让任何人变轻，loss 也一样
why: 它**只负责打分**，它**不改任何一个 weight**。一个都不改，永远不改。你读一整天 0.42，什么都不会变。
step: **真正动手的那个** —— 另一个步骤会读那个斜率，然后真的去改 weight：那就是 **optimizer**
why: 它就是我们比喻里的吃饭和跑步。明天你学那个斜率是怎么算出来的（**gradient（梯度）和 backpropagation（反向传播）**）；而花掉这个斜率的那一步 —— optimizer 本身 —— 很快也会有属于它自己的一天。
%%%

所以今天在一条干净的缝上结束：今天这个工具负责评判，另一个工具负责修。
~~~

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="A loss scores a prediction and hands over a slope, but a separate optimizer step is what actually changes the weights; the slope itself is computed on day five"><g font-family="monospace" font-size="12" text-anchor="middle"><rect x="30" y="40" width="130" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="95" y="58" fill="#5E5191">loss (today)</text><text class="lang-zh" x="95" y="58" fill="#5E5191">loss（今天）</text><text class="lang-en" x="95" y="73" fill="#6B645E" font-size="10">reads how wrong</text><text class="lang-zh" x="95" y="73" fill="#6B645E" font-size="10">读出错得多厉害</text><text class="lang-en" x="196" y="58" fill="#6B645E">→ slope →</text><text class="lang-zh" x="196" y="58" fill="#6B645E">→ 斜率 →</text><text class="lang-en" x="196" y="74" fill="#9A938A" font-size="9">Day 5: how it's found</text><text class="lang-zh" x="196" y="74" fill="#9A938A" font-size="9">第 5 天：它怎么被算出来</text><rect x="255" y="40" width="150" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text class="lang-en" x="330" y="58" fill="#1a5c38">optimizer</text><text class="lang-zh" x="330" y="58" fill="#1a5c38">optimizer</text><text class="lang-en" x="330" y="73" fill="#6B645E" font-size="10">actually moves weights</text><text class="lang-zh" x="330" y="73" fill="#6B645E" font-size="10">真正去挪 weight</text><text class="lang-en" x="460" y="58" fill="#9A938A" font-size="10">judge,</text><text class="lang-zh" x="460" y="58" fill="#9A938A" font-size="10">先评判，</text><text class="lang-en" x="460" y="73" fill="#9A938A" font-size="10">then fix</text><text class="lang-zh" x="460" y="73" fill="#9A938A" font-size="10">再去修</text></g></svg>
%%%

That separate step is the [[optimizer||The separate step that reads the loss's slope and actually changes the weights to make the loss smaller. It gets its own lesson shortly after gradients.]] — the eating and the running of our analogy.

%%% insight
This split is *good news*, not a limitation to mourn. Because scoring and fixing are separate jobs, you can swap either one without touching the other — keep your loss, try a smarter optimizer; keep your optimizer, change what "wrong" means. That clean seam is why the same handful of losses shows up in every model ever built.
%%%

One more honest edge, and it's worth ten seconds: a loss only measures agreement with the labels **you** handed it. Hand it wrong labels and it will cheerfully, precisely march your model toward the wrong answer. The score is a mirror, not a judge of your data.

**You now know exactly where a loss stops.** It's the judge, not the fix — and tomorrow you start building the fix. One page to gather it all, and you're done.
~~~zh
那个分开的步骤就是 [[optimizer（优化器）||分开的那个步骤：它读 loss 的斜率，然后真的去改 weight，让 loss 变小。讲完 gradient 之后不久，它会有属于自己的一课。]] —— 也就是我们比喻里的吃饭和跑步。

%%% insight
这个分工是*好消息*，不是一个要哀叹的限制。因为打分和修改是两份活，你可以换掉其中任何一个，而不碰另一个 —— 留着你的 loss，试一个更聪明的 optimizer；留着你的 optimizer，改一改「错」的意思。正是这条干净的缝，让同样那几个 loss 出现在人类造过的每一个模型里。
%%%

还有一条老实的边界，值得花十秒：一个 loss 只量它和**你**递给它的那些 label 合不合。你递给它错的 label，它会开开心心、精精准准地把你的模型带向错的答案。这个分数是一面镜子，不是你数据的裁判。

**你现在确切知道一个 loss 停在哪里了。** 它是评判的那个，不是修的那个 —— 而明天你就开始搭修的那个。再来一页把全部收拢，你就完成了。
~~~

@@@ concept id=c11 zh_tag="回顾" tag="Recap" zh_title="今天，一页讲完" title="Today in one page" zh_gotit="懂了回顾" gotit="Got the recap"
Take a breath — you built the whole scorecard. Here's a way to hold the day at once: picture it as a single **round of golf**, with the beats below as the holes you played, in order. Each hole only made sense because of the one before it.

**Where the golf-round picture breaks down:** a real round ends when you sink the last putt, but this scorecard is one you'll come back to for years.

The holes you played, in order:
- **Hole 1 · The score.** A **loss** turns "how wrong was that guess?" into one number (lower is better), comparing a **prediction** to a **target**. The **cost** is that loss averaged over all examples.
- **Hole 2 · Why smooth.** Counting mistakes (the **0/1 loss**) is flat, so it gives no direction. Every real loss is **smooth** — it measures how far off. Report accuracy separately.
- **Hole 3 · Guessing numbers.** **MSE** = average of the squared misses; squaring makes misses always positive and big misses count more. Soft spot: one **outlier** hijacks it → **MAE** or **Huber**.
- **Hole 4 · Yes/no and one-of-many.** **Cross-entropy** = `−log(prob of the correct answer)`. Pair **sigmoid → binary cross-entropy** for yes/no, **softmax → categorical cross-entropy** for one-of-many.
- **Hole 5 · Reading the score.** Watch the *trend*, not one number: a healthy curve drops then flattens. Compare against a baseline — a pure guesser scores `ln K` (0.69 for yes/no, 2.30 for ten classes).
- **Hole 6 · Three traps, three cures.** Outliers → MAE. MSE **stalls** on confident-wrong yes/no calls → cross-entropy. A `p=0` sends cross-entropy to `+∞`/NaN → **epsilon clipping**.
- **Hole 7 · The honest limit.** A loss only scores and hands over a slope. It never moves a weight — that's the **optimizer**, which arrives shortly after tomorrow's gradients.
~~~zh
喘口气 —— 你把整张记分卡都搭出来了。有一个办法能让你一次抓住整天：把它想成**一轮高尔夫**，下面这些节拍就是你按顺序打过的那些洞。每一洞之所以讲得通，都是因为它前面那一洞。

**高尔夫这一轮的图在哪里不成立：** 真的一轮在你推进最后一杆时就结束了，而这张记分卡是你未来好几年都会回来看的。

你打过的那些洞，按顺序：
- **第 1 洞 · 那个分数。** 一个 **loss** 把「那次猜测错得多厉害？」变成一个数字（越小越好），它拿一个 **prediction** 和一个 **target** 比。**cost** 就是这个 loss 在所有样本上取的平均。
- **第 2 洞 · 为什么要平滑。** 数错了几道（也就是 **0/1 loss**）是平的，所以它给不出方向。每一个真正在用的 loss 都是**平滑**的 —— 它量的是差多远。accuracy 单独汇报。
- **第 3 洞 · 猜数字。** **MSE** = 平方之后的差的平均；平方让每个差都是正的，也让大的差算得更重。它的软肋：一个 **outlier** 就能绑走它 → 换 **MAE** 或 **Huber**。
- **第 4 洞 · 是/否，以及多选一。** **Cross-entropy** = `−log（正确答案的概率）`。是/否配 **sigmoid → binary cross-entropy**，多选一配 **softmax → categorical cross-entropy**。
- **第 5 洞 · 读这个分数。** 看*趋势*，不看单个数字：健康的曲线先掉再变平。要和一个基准比 —— 一个纯瞎猜的模型得 `ln K`（是/否是 0.69，十类是 2.30）。
- **第 6 洞 · 三个坑，三个解药。** 有 outlier → MAE。MSE 在又自信又错的是/否判断上**卡住** → cross-entropy。`p=0` 会把 cross-entropy 送到 `+∞` 或 NaN → **epsilon clipping**。
- **第 7 洞 · 老实的边界。** 一个 loss 只打分，再交出一个斜率。它从不动任何一个 weight —— 那是 **optimizer** 的活，它在明天的 gradient 之后不久就到。
~~~

%%% svg
<svg viewBox="0 0 520 216" role="img" aria-label="A mini-golf scorecard for today's round listing the seven holes played in order: the score, why smooth, guessing numbers, yes-no and one-of-many, reading the score, three traps three cures, and the honest limit. Each hole builds on the one before."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Today's round — the holes you played, in order ⛳</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">今天这一轮 —— 你按顺序打过的那些洞 ⛳</text><rect x="30" y="28" width="460" height="176" rx="8" fill="#FDFCF9" stroke="#C9B77A" stroke-width="2"/><text x="48" y="52" fill="#9A7208">1 ·</text><text class="lang-en" x="78" y="52" fill="#2C2A28">The score — one number, lower is better</text><text class="lang-zh" x="78" y="52" fill="#2C2A28">那个分数 —— 一个数字，越小越好</text><text x="48" y="76" fill="#9A7208">2 ·</text><text class="lang-en" x="78" y="76" fill="#2C2A28">Why smooth — a stamp gives no hint</text><text class="lang-zh" x="78" y="76" fill="#2C2A28">为什么要平滑 —— 图章给不出提示</text><text x="48" y="100" fill="#9A7208">3 ·</text><text class="lang-en" x="78" y="100" fill="#2C2A28">Guessing numbers — MSE (outlier → MAE)</text><text class="lang-zh" x="78" y="100" fill="#2C2A28">猜数字 —— MSE（有 outlier → MAE）</text><text x="48" y="124" fill="#9A7208">4 ·</text><text class="lang-en" x="78" y="124" fill="#2C2A28">Yes/no &amp; one-of-many — cross-entropy</text><text class="lang-zh" x="78" y="124" fill="#2C2A28">是/否 和 多选一 —— cross-entropy</text><text x="48" y="148" fill="#9A7208">5 ·</text><text class="lang-en" x="78" y="148" fill="#2C2A28">Reading it — watch the trend vs a baseline</text><text class="lang-zh" x="78" y="148" fill="#2C2A28">读它 —— 看趋势，和基准比</text><text x="48" y="172" fill="#9A7208">6 ·</text><text class="lang-en" x="78" y="172" fill="#2C2A28">Three traps, three cures — outlier, stall, log(0)</text><text class="lang-zh" x="78" y="172" fill="#2C2A28">三个坑，三个解药 —— outlier、卡住、log(0)</text><text x="48" y="196" fill="#9A7208">7 ·</text><text class="lang-en" x="78" y="196" fill="#2C2A28">The honest limit — it scores, doesn't fix</text><text class="lang-zh" x="78" y="196" fill="#2C2A28">老实的边界 —— 它打分，不负责修</text><path d="M40 58 C 24 92, 24 124, 40 158" fill="none" stroke="#B8AEA2" stroke-width="1.5" stroke-dasharray="3,3"/><text class="lang-en" x="18" y="118" fill="#6B645E" font-size="9" transform="rotate(-90 18 118)">each builds on the last</text><text class="lang-zh" x="18" y="118" fill="#6B645E" font-size="9" transform="rotate(-90 18 118)">每一洞都踩着上一洞</text></g></svg>
%%%

And the one picture to keep — the four workhorse scores at a glance:
~~~zh
还有那张要留住的图 —— 四个主力分数一眼看完：
~~~

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Recap: MSE and MAE score numeric guesses; binary and categorical cross-entropy score classification; every one is a smooth how-wrong number"><g font-family="monospace" font-size="12" text-anchor="middle"><text class="lang-en" x="140" y="26" fill="#3A342E" font-weight="bold">guess a NUMBER</text><text class="lang-zh" x="140" y="26" fill="#3A342E" font-weight="bold">猜一个数字</text><rect x="40" y="40" width="90" height="34" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text class="lang-en" x="85" y="61" fill="#9A7208">MSE</text><text class="lang-zh" x="85" y="61" fill="#9A7208">MSE</text><rect x="150" y="40" width="90" height="34" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text class="lang-en" x="195" y="61" fill="#9A7208">MAE</text><text class="lang-zh" x="195" y="61" fill="#9A7208">MAE</text><text class="lang-en" x="140" y="92" fill="#6B645E" font-size="10">→ how far off, averaged</text><text class="lang-zh" x="140" y="92" fill="#6B645E" font-size="10">→ 差多远，取平均</text><text class="lang-en" x="390" y="26" fill="#3A342E" font-weight="bold">pick a CLASS</text><text class="lang-zh" x="390" y="26" fill="#3A342E" font-weight="bold">挑一个类别</text><rect x="290" y="40" width="100" height="34" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="340" y="61" fill="#5E5191" font-size="11">BCE (yes/no)</text><text class="lang-zh" x="340" y="61" fill="#5E5191" font-size="11">BCE（是/否）</text><rect x="400" y="40" width="100" height="34" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="450" y="61" fill="#5E5191" font-size="11">CCE (many)</text><text class="lang-zh" x="450" y="61" fill="#5E5191" font-size="11">CCE（多类）</text><text class="lang-en" x="390" y="92" fill="#6B645E" font-size="10">→ −log(prob of correct)</text><text class="lang-zh" x="390" y="92" fill="#6B645E" font-size="10">→ −log（正确的概率）</text><text class="lang-en" x="260" y="120" fill="#1a5c38" font-size="11">every one: a smooth "how wrong" number the network shrinks</text><text class="lang-zh" x="260" y="120" fill="#1a5c38" font-size="11">每一个都是：一个平滑的「错得多厉害」数字，网络去压小它</text></g></svg>
%%%

#### Cheat-sheet · which score to reach for
%%% table
:: Your task :: Output shape :: Loss :: Watch out for
Guess a number :: raw number (linear) :: MSE :: one outlier hijacks it → switch to MAE / Huber
Yes / no :: one probability (sigmoid) :: binary cross-entropy :: never use MSE here — it stalls
One of 3+ classes :: probabilities that sum to 1 (softmax) :: categorical cross-entropy :: `log(0)` → clamp with ε
%%%

#### Cheat-sheet · the words you met today
%%% jargon
loss | one number for how wrong ONE prediction is; lower is better, 0 is perfect
cost | the loss averaged over the whole batch — the single number training shrinks
prediction | the guess the forward pass produced
target | the true answer (also called the label) the loss compares against
0/1 loss | just counting mistakes; flat, so it can't be trained on
MSE | mean squared error — average of the squared misses; punishes big misses hardest
MAE | mean absolute error — average of the plain miss sizes; calm about outliers
Huber loss | MSE for small misses, MAE for big ones — the best-of-both middle ground
logits | the raw output scores of the last layer, before sigmoid/softmax make probabilities
sigmoid | squashes a number into one probability (0 to 1) — for yes/no
softmax | turns raw scores into probabilities that sum to 1 — for 3+ classes
binary cross-entropy | −log(prob of the correct answer) for yes/no; hammers confident-and-wrong
categorical cross-entropy | the same −log(prob of correct), spread over 3+ classes
chance level | the loss a pure guesser gets: ln K for K classes (0.69 for yes/no) — your baseline
epsilon clipping | clamp probabilities to [ε, 1−ε] so −log can never blow up to infinity
optimizer | the separate step that reads the loss's slope and actually changes the weights (its own lesson, just after gradients)
%%%

That's the whole day. Next you'll see how the network *finds* this score's slope — **gradients and backpropagation**.
~~~zh
#### 小抄 · 该伸手拿哪个分数

%%% table
:: 你的任务 :: 输出形状 :: Loss :: 要当心什么
猜一个数字 :: 原始数字（线性） :: MSE :: 一个 outlier 就能绑走它 → 换成 MAE / Huber
是 / 否 :: 一个概率（sigmoid） :: binary cross-entropy :: 这里永远别用 MSE —— 它会卡住
3 类以上里挑一个 :: 加起来等于 1 的概率（softmax） :: categorical cross-entropy :: `log(0)` → 用 ε 夹住
%%%

#### 小抄 · 今天你遇到的那些词

%%% jargon
loss | 一个数字，说明一次 prediction 错得多厉害；越小越好，0 是完美
cost | loss 在整个 batch 上取的平均 —— 训练要压小的那一个数字
prediction | forward pass 产生出来的那个猜测
target | 数据自带的真答案（也叫 label），loss 拿它来比
0/1 loss | 只数错了几道；它是平的，所以没法拿来训练
MSE | mean squared error —— 平方之后的差的平均；罚大的差最狠
MAE | mean absolute error —— 原本大小的差的平均；对 outlier 很冷静
Huber loss | 小的差用 MSE，大的差用 MAE —— 两边好处都要的中间路线
logits | 最后一层的原始输出分数，还没被 sigmoid 或 softmax 变成概率
sigmoid | 把一个数字挤成一个概率（0 到 1）—— 给是/否用
softmax | 把原始分数变成加起来等于 1 的概率 —— 给 3 类以上用
binary cross-entropy | 是/否用的 −log（正确答案的概率）；狠砸又自信又错的
categorical cross-entropy | 同一个 −log（正确答案的概率），摊到 3 类以上
chance level | 一个纯瞎猜的模型拿到的 loss：K 类时是 ln K（是/否是 0.69）—— 你的基准
epsilon clipping | 把概率夹到 [ε, 1−ε]，让 −log 永远炸不到无穷
optimizer | 分开的那个步骤：它读 loss 的斜率，然后真的去改 weight（讲完 gradient 之后不久，它有自己的一课）
%%%

今天就是这些。接下来你会看到网络怎么*找到*这个分数的斜率 —— **gradient 和 backpropagation**。
~~~

@@@ quiz id=quiz zh_tag="小测" tag="Quiz" zh_title="点一个答案 —— 每题都有即时反馈" title="Click an answer — instant feedback on each" zh_gotit="先把四题答完" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is a loss function, in one line? | a:1 | The step that updates the weights | One number saying how wrong a prediction is, comparing it to the target (lower is better) | The network's number of layers | The learning rate | fb: A loss takes (prediction, target) and turns "how far off was that guess?" into one number. Lower is better; 0 is a perfect guess. The step that updates weights is the optimizer.
q: You're predicting house prices and a few freak mansions are wild outliers. Which loss keeps one crazy point from hijacking the score? | a:2 | MSE, because it squares errors | Cross-entropy | MAE (or Huber), because it doesn't square the miss | counting mistakes | fb: MSE squares each miss, so one giant error explodes and dominates. MAE takes the plain miss size, so an outlier counts in proportion — Huber is the smooth middle ground.
q: For "cat vs dog vs bird" (three classes), which output-and-loss pairing is right? | a:2 | linear output + MSE | sigmoid + binary cross-entropy | softmax + categorical cross-entropy | no loss is needed | fb: Softmax turns the raw scores (logits) into probabilities that sum to 1 — one whole pizza sliced across the classes — and categorical cross-entropy scores −log of the slice you gave the correct class. Sigmoid + binary cross-entropy is for yes/no; MSE is for numbers.
q: Why does cross-entropy beat MSE for a yes/no classifier? | a:2 | Cross-entropy uses less memory | It counts mistakes directly | Its learning push stays strong when the model is confidently wrong, while MSE+sigmoid goes nearly flat there and stalls | It never needs a probability | fb: With a sigmoid output, MSE's push shrinks toward 0 exactly when the model is confidently wrong. Cross-entropy keeps a strong push right there, so it fixes loud mistakes fast.
%%%
~~~zh
四道快问，每一题都有即时反馈 —— 四题都答完，今天就完成了。（如果有一题绊住你，那是叫你往上翻回去看，不是给你打不及格。）

%%% quiz
q: 一句话说，loss function 是什么？ | a:1 | 更新 weight 的那一步 | 一个数字，说明一次 prediction 错得多厉害，它拿 prediction 和 target 比（越小越好） | 网络有几层 | 学习率 | fb: 一个 loss 拿走 (prediction, target)，把「那次猜测差了多远？」变成一个数字。越小越好；0 是完美的猜测。更新 weight 的那一步叫 optimizer。
q: 你在预测房价，而有几栋怪物豪宅是很野的 outlier。哪个 loss 能不让一个疯点绑走整个分数？ | a:2 | MSE，因为它把误差平方 | Cross-entropy | MAE（或者 Huber），因为它不把差平方 | 数一数错了几道 | fb: MSE 把每个差平方，所以一个巨大的误差会炸开并压住全场。MAE 取原本大小的差，所以一个 outlier 按比例算 —— Huber 是那条平滑的中间路线。
q: 对「猫 vs 狗 vs 鸟」（三个类别），哪一种「输出 + loss」的配对是对的？ | a:2 | 线性输出 + MSE | sigmoid + binary cross-entropy | softmax + categorical cross-entropy | 不需要 loss | fb: Softmax 把原始分数（logits）变成加起来等于 1 的概率 —— 一整张披萨切给各个类别 —— 而 categorical cross-entropy 算你给正确类别那一块的 −log。Sigmoid + binary cross-entropy 是给是/否用的；MSE 是给数字用的。
q: 做一个是/否的分类器时，cross-entropy 为什么比 MSE 强？ | a:2 | Cross-entropy 用的内存更少 | 它直接数错了几道 | 模型很自信地错了的时候，它的学习推力依然很强，而 MSE+sigmoid 在那里几乎变平、卡住了 | 它完全不需要概率 | fb: 输出是 sigmoid 时，MSE 的推力正好在模型很自信地错了的时候缩向 0。Cross-entropy 在那里保持很强的推力，所以它能很快修好那些大声犯的错。
%%%
~~~

@@@ produce id=produce zh_tag="动手做" tag="Produce" zh_title="看 loss 奖励自信 —— 也惩罚大声犯错" title="Watch a loss reward confidence — and punish being loudly wrong" zh_gotit="做完了" gotit="Done"
Time to see today's big idea with your own eyes. You'll compute MSE and cross-entropy on tiny examples and **watch** how each reacts as a guess gets better or worse. **Predict first:** as a classifier's probability on the *correct* answer climbs from 0.1 to 0.99, does cross-entropy go up or down — and where does it change fastest: in the middle of that climb, or right down at the confident-wrong end? Then run it and **notice** whether your guess held. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-04-loss/experiment.py`. (1) Write `mse(pred, target)` and `mae(pred, target)`, and print both for `pred=[2.5, 0, 2]`, `target=[3, −0.5, 2]` — **notice** MSE prints `≈ 0.167`. (2) Now stage the hijack on a crowd: twenty houses each missed by `0.5`, then bolt on **one** mansion missed by `100`, and print both scores before and after — **watch** MSE leap from `0.25` to about `476` while MAE only walks from `0.5` to about `5.2`. (3) Write `cross_entropy(p)` as `−log(p)` and print it for `p = [0.99, 0.6, 0.1, 1e-7]`, plus the two step-to-step drops (`0.1→0.6` and `0.6→0.99`) — **notice** it falls toward 0 as `p` nears 1, shoots up as `p` nears 0, and that clamping `p` with `max(p, 1e-7)` keeps a `p=0` case finite (`≈16.1`) instead of `+∞`. Run with `python3 sessions/m02-the-neuron/day-04-loss/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 4 artifact.

Create sessions/m02-the-neuron/day-04-loss/experiment.py that, with a comment on each step:
1. Defines mse(pred,target)=np.mean((pred-target)**2) and mae(pred,target)=np.mean(np.abs(pred-target)); prints both for pred=[2.5,0,2], target=[3,-0.5,2] (expect MSE~0.167, MAE~0.333).
2. Stages the outlier hijack on a CROWD: 20 examples each missed by 0.5, then the same 20 plus ONE mansion missed by 100. Print MSE and MAE for both batches (expect MSE 0.25 -> ~476.4, MAE 0.5 -> ~5.2) and print each growth factor, so the numbers show MSE is amplified about 1900x while MAE only grows about 10x — the outlier is FELT by MAE, not amplified.
3. Defines cross_entropy(p)=-np.log(np.clip(p,1e-7,1-1e-7)) and prints it for p in [0.99,0.6,0.1,1e-7]; show it falls toward 0 as p->1 and shoots up as p->0, and that the clip keeps p=0 finite (~16.1) instead of +infinity.
4. Prints WHERE cross-entropy changes fastest: the drop from p=0.1 to 0.6 versus the drop from p=0.6 to 0.99 (expect ~1.79 vs ~0.50), so the numbers show it changes fastest near p->0 — the confident-wrong end — and flattens as p->1.
5. Prints a one-line reminder that a loss only SCORES and hands over a slope — it does not change any weight (that is the optimizer, a later lesson).
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the scorecard, reacting)
- **MSE** on the close guess prints `≈ 0.167` (MAE `≈ 0.333`) — a small score, because the guesses are close.
- The crowd test: adding **one** mansion missed by `100` sends MSE from `0.25` to `≈ 476` — about **1,900× bigger** — while MAE only goes `0.5` → `≈ 5.2`, about **10×**. Same single bad row: MSE *amplified* it, MAE merely *felt* it. That's the hijack, live, and the reason MAE is the calmer scorecard.
- **Cross-entropy** falls toward `0` as the probability on the correct answer climbs to `0.99`, and shoots up as it slides toward `0.1` — the confident-and-wrong sting, live.
- **Where it changes fastest** (your prediction from the top): the drop from `p=0.1` to `p=0.6` is about `1.79`, while `p=0.6` to `p=0.99` is only about `0.50`. So cross-entropy moves fastest down at the **confident-wrong end** (`p → 0`) and flattens out as `p → 1`. If you guessed "in the middle", that's a great mistake to have made once — the steepness lives at the bad end, which is exactly what makes the loss push hardest there.
- The `p=1e-7` case prints a big but **finite** number (`≈16.1`) instead of `+∞` — that's epsilon clipping saving you from a NaN.

!!! c-info 📓
<b>5-minute research log:</b> before you close the tab, write three lines in `sessions/m02-the-neuron/day-04-loss/log.md`: (1) what a loss is, and what the "cost" adds on top; (2) which loss you'd pick for house prices vs. spam-or-not vs. cat/dog/bird, and why; (3) one trap from today (outlier, stall, or log(0)) and its one-line cure.
!!!
~~~zh
是时候亲眼看看今天这个大想法了。你会在很小的例子上算出 MSE 和 cross-entropy，然后**看**每一个在猜测变好或变坏时怎么反应。**先预测：** 当一个分类器给*正确*答案的概率从 0.1 爬到 0.99，cross-entropy 是往上走还是往下走？而它在哪里变得最快：在这段爬升的中间，还是在「很自信地错了」那一头？然后跑一遍，**注意**你的猜测有没有站住。挑一条路走。

#### 路线 A · 自己写
建一个 `sessions/m02-the-neuron/day-04-loss/experiment.py`。(1) 写出 `mse(pred, target)` 和 `mae(pred, target)`，为 `pred=[2.5, 0, 2]`、`target=[3, −0.5, 2]` 打印两个值 —— **注意** MSE 打印出 `≈ 0.167`。(2) 现在在一群房子上摆出那场绑架：二十栋房子每栋差 `0.5`，然后加上**一**栋差 `100` 的豪宅。打印加之前和加之后的两个分数 —— **看** MSE 从 `0.25` 跳到大约 `476`，而 MAE 只从 `0.5` 走到大约 `5.2`。(3) 把 `cross_entropy(p)` 写成 `−log(p)`，为 `p = [0.99, 0.6, 0.1, 1e-7]` 打印它，再打印相邻两段的下降（`0.1→0.6` 和 `0.6→0.99`）。**注意**：`p` 靠近 1 时它掉向 0。`p` 靠近 0 时它冲上去。而用 `max(p, 1e-7)` 夹住 `p`，能让 `p=0` 的情况保持有限（`≈16.1`），不变成 `+∞`。用 `python3 sessions/m02-the-neuron/day-04-loss/experiment.py` 跑起来。

#### 路线 B · 让 Claude 帮你搭，然后你读它
把下面那段 prompt 复制回 Claude Code。它会让 Claude Code 建好文件、写好代码，并且替你跑一遍。（prompt 本身保持英文，它是给工具看的指令。）

#### 你应该看到什么（记分卡，正在反应）
- 那个很近的猜测上，**MSE** 打印 `≈ 0.167`（MAE `≈ 0.333`）—— 一个很小的分数，因为猜得很近。
- 那个人群测试：加上**一**栋差 `100` 的豪宅，把 MSE 从 `0.25` 送到 `≈ 476` —— 大约**大了 1,900 倍** —— 而 MAE 只从 `0.5` 走到 `≈ 5.2`，大约**10 倍**。同样一行糟糕的数据：MSE 把它*放大*了，MAE 只是*感觉到*它。这就是那场绑架的现场，也是 MAE 更冷静的理由。
- 当正确答案上的概率爬到 `0.99` 时，**cross-entropy** 掉向 `0`；当它滑向 `0.1` 时，cross-entropy 冲上去 —— 「又自信又错」那一扎的现场。
- **它在哪里变得最快**（你在开头做的那个预测）：从 `p=0.1` 到 `p=0.6` 的下降大约是 `1.79`，而 `p=0.6` 到 `p=0.99` 只有大约 `0.50`。所以 cross-entropy 在「很自信地错了」那一头（`p → 0`）动得最快，而在 `p → 1` 时变平。如果你猜的是「在中间」，那是个很值得犯一次的错 —— 陡的地方在坏的那一头，正是这一点让 loss 在那里推得最狠。
- `p=1e-7` 那一项打印出一个很大但**有限**的数字（`≈16.1`），而不是 `+∞` —— 那就是 epsilon clipping 把你从一个 NaN 手里救了回来。

!!! c-info 📓
<b>5 分钟研究日志：</b>关掉页面之前，在 `sessions/m02-the-neuron/day-04-loss/log.md` 里写三行：(1) loss 是什么，而「cost」在它之上多加了什么；(2) 房价、是不是垃圾邮件、猫/狗/鸟，这三件事你各会挑哪个 loss，为什么；(3) 今天的一个坑（outlier、卡住，或者 log(0)），以及它那一行的解药。
!!!
~~~

@@@ fin
