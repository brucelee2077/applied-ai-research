---
quest_id: wf3-d02-backprop
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 5 — Gradients & Backpropagation"
module_label: "Module 2 · Train · Day 5"
zh_module_label: "第 2 模块 · 训练 · 第 5 天"
title: "Gradients & Backpropagation"
zh_title: "梯度与反向传播"
subtitle: "Which Way Is Downhill?"
zh_subtitle: "哪边是下坡？"
brand_sub: "Foundations · M2 Day 5"
spine: "hiker in fog"
zh_spine: "雾里的登山者"
nav_prev_href: "../day-04-loss/lesson.html"
nav_prev_label: "Loss Functions"
zh_nav_prev_label: "损失函数"
nav_next_href: "../day-06-training-loop/lesson.html"
nav_next_label: "The Training Loop"
zh_nav_next_label: "训练循环"
fin_title: "Module 2 · Day 5 complete! 🏆"
zh_fin_title: "第 2 模块 · 第 5 天 完成！🏆"
fin_body: "Nice work — you've met the <b>gradient</b>, the arrow that points along the steepest slope <em>uphill</em> on the loss surface, which is exactly why we step the opposite way — downhill — and <b>backpropagation</b>, the chain-rule trick that hands every weight its own share of the blame. Now the network knows which way to step.<br>Next up: <b>The Training Loop</b> — where these steps repeat until the network learns."
zh_fin_body: "做得好 —— 你认识了 <b>gradient（梯度）</b>：它是在 loss 表面上指向最陡<em>上坡</em>的那支箭。正因为这样，我们才往相反的方向走，也就是下坡。你也认识了 <b>backpropagation（反向传播）</b>：它用 chain rule 把责任分给每一个 weight。现在网络知道该往哪边迈步了。<br>下一站：<b>训练循环</b> —— 这些步子在那里一圈一圈重复，直到网络学会。"
notebook_yardstick: 00-neural-networks/fundamentals/07_backpropagation.ipynb
coverage_topics:
  - {topic: loss/error signal, keywords: [how wrong, one number, error signal]}
  - {topic: gradient, keywords: [gradient, which way is downhill, steepest]}
  - {topic: step opposite the gradient, keywords: [opposite, downhill, minus]}
  - {topic: forward pass, keywords: [forward pass, weighted sum, save the intermediate]}
  - {topic: computation graph, keywords: [small steps, chain of operations, computation graph]}
  - {topic: local derivative, keywords: [local slope, one small step, immediately downstream]}
  - {topic: chain rule, keywords: [chain rule, multiply the slopes, chain of effects]}
  - {topic: upstream downstream gradient, keywords: [incoming gradient, upstream, pass it further back]}
  - {topic: backward pass backprop, keywords: [backward pass, backpropagation, from the loss back]}
  - {topic: backprop reuse efficiency, keywords: [one sweep, reuse, instead of re-deriving]}
  - {topic: gradient of the weighted sum, keywords: [input value times, big inputs, weight's gradient]}
  - {topic: activation derivative in chain, keywords: [activation's slope multiplies, sigmoid slope, relu slope]}
  - {topic: gradient with respect to bias, keywords: [bias, passes straight through, slope 1]}
  - {topic: vanishing gradient, keywords: [vanishing gradient, multiply toward 0, saturating, signal fades]}
  - {topic: relu remedy for vanishing, keywords: [relu slope stays 1, switch to relu]}
  - {topic: dead relu zero gradient, keywords: [dead relu, permanently negative, slope 0, never learns]}
  - {topic: leaky relu remedy, keywords: [leaky relu, small negative slope, sensible init]}
  - {topic: exploding gradient, keywords: [exploding gradient, huge values, overshoot]}
  - {topic: gradient clipping remedy, keywords: [gradient clipping, clip, careful init]}
  - {topic: zero-init symmetry, keywords: [zero-initialization, identical gradients, symmetry, stay identical]}
  - {topic: small random init remedy, keywords: [small random, break the symmetry]}
  - {topic: numerical gradient check, keywords: [numerical gradient check, finite difference, nudge and measure]}
---

@@@ hero
@lede Imagine you're standing on a hillside in thick **fog**. You can't see the valley, but you want to walk down to the lowest point. So you do the one thing you still can: you feel the ground right under your boots, notice which way it tips *down*, and take a step that way. Then you feel again, and step again. That simple move — *feel the slope, step downhill* — is the whole secret behind how every neural network learns: the face-unlock model on your phone, the thing that picks your next recommended video, and the giant one behind ChatGPT. Yesterday you built the network's **score** — one number for how wrong its guess was. Today you learn the magic that reads that score and whispers to every single knob inside the network: *"nudge this way to make the score a little lower."* That whisper has a name — the **gradient** — and the clever trick that computes it for thousands of knobs at once is called **backpropagation**. By the end you'll know exactly which way is downhill.
@goal Together we'll turn a foggy hillside into a precise recipe. You'll meet the **gradient** (the arrow that points along the slope), watch the network **save its work** on the way forward so it can trace blame on the way back, learn the one multiplying trick — the **chain rule** — that hands every knob its fair share of the blame, and see the whole **backward pass** sweep through in a single pass. You'll get sliders to make the downhill signal fade, die, or explode with your own hand, a 30-second trick to prove your gradients are right, and a few sneaky traps with neat one-line cures. Every formula shows up in plain words first; any heavy math sits in a skippable box.

%%% warmup
q: Yesterday's loss — what is it, in one line? | a:1 | The number of layers in the network | One number saying how wrong a guess is; lower is better, 0 is perfect | The step that changes the weights | The speed the network trains at | concept: loss-one-number | fb: A loss turns "how far off was that guess?" into one honest number. Lower is better — today's gradient is what reads that number and finds which way to shrink it.
q: You're guessing house prices and a few freak mansions are wild outliers. Which loss keeps one crazy point from hijacking the score? | a:2 | MSE, because it squares each miss | Cross-entropy | MAE (or Huber), because it takes the plain miss size instead of squaring it | Counting mistakes | concept: mse-vs-mae-outlier | fb: MSE squares every miss, so one giant error explodes and drowns out the rest. MAE uses the plain miss size, so an outlier counts in proportion — Huber is the smooth middle ground.
q: Yesterday you learned a loss only judges — it does not fix. What actually changes the weights? | a:2 | The loss function itself | The activation function | A separate step called the optimizer, which reads the loss's slope | The number of training examples | concept: loss-scores-not-fix | fb: A loss only scores and hands over a slope; it never touches a weight. The optimizer reads that slope and steps the weights — and today's gradient is exactly the slope it reads.
%%%
@zh_lede 想象你站在山坡上，周围是很浓的**雾**。你看不见山谷，但你想走到最低的地方。于是你做那件你还能做的事：你摸一摸靴子底下的地面，注意它朝哪边*往下*倾，然后朝那边迈一步。然后再摸一次，再迈一步。这个简单的动作 —— *摸清坡度，往下坡走* —— 就是每一个神经网络学习的全部秘密。你手机上的刷脸解锁模型、替你挑下一个推荐视频的东西、还有 ChatGPT 背后那个巨大的家伙，学的时候都在做这件事。昨天你搭好了网络的**分数** —— 一个数字，说它猜得有多错。今天你要学的是读这个分数的魔法。它会对网络里每一个旋钮小声说：*「朝这边挪一点，分数就会低一点。」* 这句耳语有个名字 —— **gradient（梯度）**。而一次替成千上万个旋钮算出它的那个聪明花招，叫 **backpropagation（反向传播）**。到今天结束，你会确切知道哪边是下坡。
@zh_goal 我们会一起把一片有雾的山坡变成一份精确的配方。你会认识 **gradient**，也就是顺着坡指的那支箭。你会看网络在往前走的时候**把过程记下来**，好让它往回走的时候能追查责任。你会学到那一个乘法花招 —— **chain rule（链式法则）** —— 它把责任公平地分给每一个旋钮。你还会看整个 **backward pass（反向传播）**一次扫完全程。你会拿到几个滑块，亲手让下坡信号变弱、死掉、或者炸开。你会拿到一个三十秒的小技巧，用来证明你的 gradient 是对的。你还会认识几个偷偷藏着的陷阱，每个都配一行干净的解药。每个公式都先用大白话说一遍。重一点的数学都放在可以跳过的盒子里。

~~~zh
%%% warmup
q: 昨天的 loss（损失）—— 一句话说，它是什么？ | a:1 | 网络里 layer（层）的数量 | 一个数字，说一次猜测有多错；越低越好，0 是完美 | 改变 weight（权重）的那一步 | 网络训练的速度 | concept: loss-one-number | fb: loss 把「那一猜偏了多远？」变成一个诚实的数字。越低越好 —— 今天的 gradient 就是读这个数字、再找出往哪边能让它变小的东西。
q: 你在猜房价，有几栋离谱的大豪宅是 outlier（离群点）。哪个 loss 能让一个疯狂的点抢不走整个分数？ | a:2 | MSE，因为它把每次偏差平方 | cross-entropy（交叉熵） | MAE（或者 Huber），因为它用偏差本来的大小，不平方 | 数一数错了几个 | concept: mse-vs-mae-outlier | fb: MSE 把每次偏差都平方，所以一个巨大的误差会炸开，把别的都盖住。MAE 用偏差本来的大小，所以一个 outlier 只按比例算进去 —— Huber 是中间那条平滑的路。
q: 昨天你学到 loss 只负责评分 —— 它不负责修。真正改变 weight 的是什么？ | a:2 | loss 函数本身 | activation（激活函数） | 一个单独的步骤，叫 optimizer（优化器），它读 loss 的斜率 | 训练样本的数量 | concept: loss-scores-not-fix | fb: loss 只打分，然后交出一个斜率；它从不去碰一个 weight。optimizer 读那个斜率，再挪动 weight —— 今天的 gradient 正好就是它读的那个斜率。
%%%
~~~

@@@ concept id=c1 zh_tag="下坡的箭" tag="The downhill arrow" zh_title="gradient —— 哪边是下坡？" title="The gradient — which way is downhill?" zh_gotit="懂了 gradient" gotit="Got the gradient"
Let's start on that foggy hillside, before any math. You're a **hiker in fog**, and the ground under your boots is the network's mistake: high ground means a very wrong guess, low ground means a good one. Yesterday's [[loss||one number saying how wrong the network's guess was; lower is better, 0 is perfect]] is your *height* on this hill — one number for how wrong you are right now. You can't see the whole valley through the fog. But you *can* feel the tilt of the ground right where you stand, and step the way it tips down. Do that again and again and you'll reach the bottom — the lowest mistake.
~~~zh
我们先回到那片有雾的山坡，还不碰数学。你是一个**雾里的登山者**，靴子底下的地面就是网络犯的错：地面高意味着猜得很错，地面低意味着猜得不错。昨天的 [[loss||一个数字，说网络猜得有多错；越低越好，0 是完美]] 就是你在这座山上的*高度* —— 一个数字，说你现在错得有多厉害。隔着雾你看不见整个山谷。但你*能*摸到脚下这块地朝哪边倾，然后朝它往下倾的那一边迈一步。一遍一遍这样做，你就会走到谷底 —— 错得最少的地方。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A hiker standing in fog on a hillside that is high on the left and low on the right. Under the hiker's boots a small dashed patch of ground is visible. A red arrow points up the slope to the left, labelled the gradient, steepest way UP, more mistake. A green arrow points down the slope to the right, labelled our step, downhill, equals minus the gradient. Height is labelled how wrong you are."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A hiker in fog: feel the tilt underfoot, step downhill</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">雾里的登山者：摸脚下的坡，往下坡走</text><path d="M20 62 C 140 62, 260 152, 500 162" fill="none" stroke="#B8AEA2" stroke-width="2"/><rect x="20" y="24" width="480" height="32" fill="#EDEAE4" opacity="0.85"/><text class="lang-en" x="260" y="44" text-anchor="middle" fill="#9A938A" font-size="10">— fog · you can't see the whole hill —</text><text class="lang-zh" x="260" y="44" text-anchor="middle" fill="#9A938A" font-size="10">—— 雾 · 你看不见整座山 ——</text><text class="lang-en" x="24" y="72" fill="#6B645E" font-size="10">height = how wrong</text><text class="lang-zh" x="24" y="72" fill="#6B645E" font-size="10">高度 = 有多错</text><text x="148" y="74" font-size="20">🥾</text><circle cx="160" cy="92" r="26" fill="none" stroke="#7C6DAA" stroke-width="1.3" stroke-dasharray="3,3"/><path d="M144 88 l-40 -20" stroke="#C93B3B" stroke-width="2.5"/><polygon points="104,68 116,69 112,77" fill="#C93B3B"/><text class="lang-en" x="72" y="98" text-anchor="middle" fill="#C93B3B" font-size="10">the gradient →</text><text class="lang-zh" x="72" y="98" text-anchor="middle" fill="#C93B3B" font-size="10">gradient 这支箭 →</text><text class="lang-en" x="72" y="110" text-anchor="middle" fill="#C93B3B" font-size="10">steepest way UP</text><text class="lang-zh" x="72" y="110" text-anchor="middle" fill="#C93B3B" font-size="10">最陡的上坡方向</text><text class="lang-en" x="72" y="122" text-anchor="middle" fill="#C93B3B" font-size="10">(more mistake)</text><text class="lang-zh" x="72" y="122" text-anchor="middle" fill="#C93B3B" font-size="10">（错误更多）</text><path d="M176 100 l56 24" stroke="#2D8B55" stroke-width="2.5"/><polygon points="232,124 220,124 224,116" fill="#2D8B55"/><text class="lang-en" x="240" y="118" fill="#2D8B55" font-size="10">our step → DOWNHILL</text><text class="lang-zh" x="240" y="118" fill="#2D8B55" font-size="10">我们的一步 → 下坡</text><text class="lang-en" x="240" y="131" fill="#2D8B55" font-size="10">= minus the gradient</text><text class="lang-zh" x="240" y="131" fill="#2D8B55" font-size="10">= 减去 gradient</text><text class="lang-en" x="200" y="152" text-anchor="middle" fill="#7C6DAA" font-size="9">the patch under your boots</text><text class="lang-zh" x="200" y="152" text-anchor="middle" fill="#7C6DAA" font-size="9">靴子底下那一块地</text><text class="lang-en" x="60" y="150" text-anchor="middle" fill="#6B645E" font-size="10">high = very wrong</text><text class="lang-zh" x="60" y="150" text-anchor="middle" fill="#6B645E" font-size="10">高 = 很错</text><text class="lang-en" x="430" y="184" text-anchor="middle" fill="#2D8B55" font-size="10">low = good guess</text><text class="lang-zh" x="430" y="184" text-anchor="middle" fill="#2D8B55" font-size="10">低 = 猜得不错</text></g></svg>
%%%

**What the hiker-in-fog picture gets right:** you only ever need the *local* tilt — the ground right under your boots — to know which way to step, and you reach the bottom one small step at a time. **Where it breaks down:** a real hill has just two directions to lean (north–south, east–west), but a real network has *thousands* of knobs, so its "hill" tips in thousands of directions at once.

#### From boots to knobs
Let's walk the hiker into the network, one small idea at a time. Each knob inside the neuron is a [[weight||a number the neuron multiplies its input by; learning means finding good values for the weights]] (or a bias). Here's the move: instead of feeling the ground with *boots*, we ask the same tilt question about *each knob*.

%%% steps
step: the test lean — nudge ONE knob up by a hair
why: that's the hiker leaning one foot one way; a tiny test move, not a real step yet
step: the reading — did the mistake go up or down, a lot or a little?
why: that answer is this knob's own slope, and it usefully splits into two parts
step: the two parts — the SIGN says which way, the SIZE says how much this knob matters right now
why: therefore a steep knob deserves a big correction, and a nearly-flat one barely matters yet
step: the whole arrow — collect that reading for every knob at once
why: that stack of slopes IS the gradient, and it points the steepest way UPHILL (more mistake)
%%%

So the [[gradient||the arrow that says, for every weight at once, which way to nudge it to change the loss the fastest — and how steeply]] is not mysterious — it's that four-rung ladder, run for every knob. Which means it answers today's title question directly: *which way is downhill?* It's the way the arrow **isn't** pointing.
~~~zh
**雾里登山者这个比喻对在哪里：** 你只需要*脚下这一小块*的倾斜，就知道该往哪边迈，而你是一小步一小步走到谷底的。**它在哪里不成立：** 真的山坡只有两个方向可以倾：南北，东西。而真的网络有*成千上万*个旋钮。所以它那座「山」同时朝成千上万个方向倾。

#### 从靴子到旋钮
我们把这个登山者一点一点带进网络里。neuron（神经元）里面的每一个旋钮，都是一个 [[weight||神经元用来乘输入的一个数字；学习就是给这些 weight 找到好的取值]]，或者一个 bias（偏置）。动作是这样：不用*靴子*去摸地面，而是对*每一个旋钮*问同一个倾斜问题。

%%% steps
step: 试探性地倾一下 —— 把一个旋钮往上挪一根头发
why: 这就是登山者把一只脚往一边倾；一个很小的试探动作，还不是真正的一步
step: 读数 —— 错误是变大了还是变小了，变了多少？
why: 这个答案就是这个旋钮自己的斜率，而它可以很有用地拆成两部分
step: 两部分 —— 正负号说往哪边，大小说这个旋钮现在有多重要
why: 因此一个陡的旋钮该得到一个大的修正，一个几乎平的旋钮现在几乎不用管
step: 整支箭 —— 一次把每一个旋钮的读数都收集起来
why: 那一叠斜率就是 gradient，而它指向最陡的*上坡*方向，也就是错误更多的那边
%%%

所以 [[gradient||一支箭，一次告诉你每一个 weight 该往哪边挪、才能最快改变 loss —— 还有它有多陡]] 并不神秘 —— 它就是那个四级的梯子，对每一个旋钮各跑一遍。这也就直接回答了今天标题里的问题：*哪边是下坡？* 就是那支箭**没有**指的那一边。
~~~

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A single dip-shaped loss curve. A dot sits on the right side. A red arrow points up-and-right along the slope labelled gradient points uphill, more error. A green arrow points the opposite way, down toward the bottom of the dip, labelled we step the opposite way, downhill."><g font-family="monospace" font-size="11"><path d="M40 30 C 160 150, 360 150, 480 30" fill="none" stroke="#B8AEA2" stroke-width="2.5"/><circle cx="410" cy="92" r="6" fill="#C99A12" stroke="#fff" stroke-width="2"/><path d="M416 88 l38 -28" stroke="#C93B3B" stroke-width="2.5"/><polygon points="454,60 443,61 448,70" fill="#C93B3B"/><text class="lang-en" x="440" y="52" fill="#C93B3B" text-anchor="middle" font-size="10">gradient → uphill (more error)</text><text class="lang-zh" x="440" y="52" fill="#C93B3B" text-anchor="middle" font-size="10">gradient → 上坡（错误更多）</text><path d="M404 96 l-40 26" stroke="#2D8B55" stroke-width="2.5"/><polygon points="364,122 375,120 370,111" fill="#2D8B55"/><text class="lang-en" x="300" y="140" fill="#2D8B55" font-size="10">we step the OPPOSITE way → downhill</text><text class="lang-zh" x="300" y="140" fill="#2D8B55" font-size="10">我们走相反方向 → 下坡</text><text class="lang-en" x="258" y="120" text-anchor="middle" fill="#6B645E" font-size="10">the bottom = smallest mistake</text><text class="lang-zh" x="258" y="120" text-anchor="middle" fill="#6B645E" font-size="10">谷底 = 错误最小</text><text class="lang-en" x="40" y="24" fill="#6B645E" font-size="10">loss (height) ↑</text><text class="lang-zh" x="40" y="24" fill="#6B645E" font-size="10">loss（高度）↑</text></g></svg>
%%%

The one line to remember: **the gradient points uphill, and to learn we step the opposite way — downhill.** That's why the update you meet tomorrow carries a **minus** sign: subtract the gradient and you walk *down* the hill instead of up it.

%%% insight
Notice what we never needed: a map of the valley. The hiker never sees the bottom — and neither does ChatGPT while it trains. Every network you've heard of learned by feeling the ground under its feet, thousands of knobs at a time, and stepping the opposite way. That's the whole idea, and you already have it.
%%%

So far: we know what we *want* — one slope per knob. The rest of today is one question: *how does the network actually feel that slope for every knob, cheaply?* That's backpropagation, and it starts with something the neuron did yesterday.
~~~zh
要记住的一句话：**gradient 指向上坡，而学习是往相反的方向走 —— 下坡。** 这就是你明天会见到的那个 update（更新）为什么带一个**减号**：把 gradient 减掉，你就是往山*下*走，而不是往上走。

%%% insight
注意我们从来不需要什么：一张山谷的地图。登山者从没见过谷底 —— ChatGPT 训练的时候也没见过。你听说过的每一个网络，都是靠摸脚下的地面学会的，一次几千个旋钮，然后朝相反的方向迈步。这就是全部的想法，而你已经拿到手了。
%%%

到这里：我们知道自己*想要*什么 —— 每个旋钮一个斜率。今天剩下的部分只有一个问题：*网络到底怎么便宜地替每一个旋钮摸到那个斜率？* 那就是 backpropagation，而它从神经元昨天做过的一件事开始。
~~~

@@@ concept id=c2 zh_tag="留好笔记" tag="Save your work" zh_title="forward pass —— 它为什么要留笔记" title="The forward pass — and why it saves its notes" zh_gotit="懂了 forward pass" gotit="Got the forward pass"
Before a network can trace its mistake backward, it has to make a guess *forward* first — and, quietly, keep a few notes along the way. Think about **baking a cake**. You add flour, then sugar, then eggs, then bake — one step feeds the next. If the cake comes out wrong, you don't throw out the kitchen; you look back at your notes — *"how much sugar did I add? how long did it bake?"* — to figure out which step to change. If you wrote nothing down, you'd be guessing blind.
~~~zh
网络要想往回追查自己的错，得先*往前*猜一次 —— 而且要不声不响地在路上留几张笔记。想想**烤蛋糕**。你先加面粉，再加糖，再加蛋，然后烤 —— 一步喂下一步。如果蛋糕出来不对，你不会把整个厨房扔掉。你会回头看你的笔记 —— *「我加了多少糖？烤了多久？」* —— 好想清楚该改哪一步。要是你什么都没记，你就只能瞎猜。
~~~

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A baking assembly line: flour then sugar then eggs then bake then taste. Under each step a small sticky note records what was used. A caption says keep the notes so if the cake is wrong you know which step to fix."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Baking forward — and keeping a note at every step</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">往前烤 —— 每一步都留一张笔记</text><rect x="24" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="61" y="59" text-anchor="middle" fill="#9A7208">flour</text><text class="lang-zh" x="61" y="59" text-anchor="middle" fill="#9A7208">面粉</text><text x="104" y="59" fill="#B8AEA2">→</text><rect x="120" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="157" y="59" text-anchor="middle" fill="#9A7208">sugar</text><text class="lang-zh" x="157" y="59" text-anchor="middle" fill="#9A7208">糖</text><text x="200" y="59" fill="#B8AEA2">→</text><rect x="216" y="40" width="74" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="253" y="59" text-anchor="middle" fill="#9A7208">eggs</text><text class="lang-zh" x="253" y="59" text-anchor="middle" fill="#9A7208">蛋</text><text x="296" y="59" fill="#B8AEA2">→</text><rect x="312" y="40" width="74" height="30" rx="5" fill="#F3ECDB" stroke="#C99A12"/><text class="lang-en" x="349" y="59" text-anchor="middle" fill="#9A7208">bake</text><text class="lang-zh" x="349" y="59" text-anchor="middle" fill="#9A7208">烤</text><text x="392" y="59" fill="#B8AEA2">→</text><rect x="408" y="40" width="88" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="452" y="59" text-anchor="middle" fill="#1a5c38">taste 😋</text><text class="lang-zh" x="452" y="59" text-anchor="middle" fill="#1a5c38">尝味道 😋</text><g><rect x="30" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text class="lang-en" x="61" y="107" text-anchor="middle" fill="#5E5191" font-size="9">2 cups</text><text class="lang-zh" x="61" y="107" text-anchor="middle" fill="#5E5191" font-size="9">2 杯</text><rect x="126" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text class="lang-en" x="157" y="107" text-anchor="middle" fill="#5E5191" font-size="9">1 cup</text><text class="lang-zh" x="157" y="107" text-anchor="middle" fill="#5E5191" font-size="9">1 杯</text><rect x="222" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text class="lang-en" x="253" y="107" text-anchor="middle" fill="#5E5191" font-size="9">3 eggs</text><text class="lang-zh" x="253" y="107" text-anchor="middle" fill="#5E5191" font-size="9">3 个蛋</text><rect x="318" y="90" width="62" height="26" rx="3" fill="#EFEAF7" stroke="#B3A6D4"/><text class="lang-en" x="349" y="107" text-anchor="middle" fill="#5E5191" font-size="9">30 min</text><text class="lang-zh" x="349" y="107" text-anchor="middle" fill="#5E5191" font-size="9">30 分钟</text></g><text class="lang-en" x="260" y="140" text-anchor="middle" fill="#6B645E">the sticky notes = the saved values — so a wrong cake tells you which step to fix</text><text class="lang-zh" x="260" y="140" text-anchor="middle" fill="#6B645E">便利贴 = 存下来的值 —— 蛋糕错了就知道该改哪一步</text></g></svg>
%%%

**What the baking picture gets right:** the steps run in order, each feeding the next, and *keeping notes* is what lets you trace a bad result back to its cause. **Where it breaks down:** a baker keeps notes for a person to read later; the network keeps its notes for *itself* — it re-uses them a fraction of a second later, on the way back.

#### The recipe, one step at a time
The [[forward pass||running the input through the neuron to get a guess — and saving the in-between values for the trip back]] is just the neuron making its guess, the same recipe you built earlier this module. Here it is as a recipe card, exactly like the cake:

%%% steps
step: the mix — multiply each input by its weight, add the bias
why: this in-between number is the [[pre-activation||the weighted sum plus bias, before the activation bends it; often written z]] `z` — the weighted sum, our "batter"
step: the bake — bend `z` with the activation (sigmoid or ReLU from Day 2) to get the output `a`
why: that bend is what makes a network more than a straight line; `a` is the neuron's guess
step: the taste — compare `a` to the true answer to get today's mistake number
why: that's yesterday's loss, the height of our hill
step: the notes — save the intermediate values `x`, `z` and `a` as each one is made
why: these are the sticky notes on the cake tin, and the backward trip re-uses every single one
%%%

That last rung is the quiet, crucial one. Save the intermediate values now, and the trip backward becomes almost free. Skip them, and you'd have to re-bake the whole cake for every question you ask.
~~~zh
**烤蛋糕这个比喻对在哪里：** 步骤是按顺序跑的，每一步喂下一步，而*留下笔记*正是让你能把一个坏结果追回到原因的东西。**它在哪里不成立：** 面包师留笔记是给人晚点读的；网络留笔记是给*它自己*用的 —— 不到一秒之后，它在往回走的路上就会重新读这些笔记。

#### 配方，一步一步来
[[forward pass||把输入送过神经元得到一个猜测 —— 同时把中间值存下来，好给回程用]]（前向传播）就是神经元做出它的猜测，和你这个模块前面搭过的是同一份配方。把它写成一张配方卡，和蛋糕一模一样：

%%% steps
step: 搅拌 —— 把每个输入乘上它的 weight，再加上 bias
why: 这个中间的数字就是 [[pre-activation||加权和加上 bias，在 activation 把它弯折之前；常写成 z]] `z` —— 也就是 weighted sum（加权和），我们的「面糊」
step: 烘烤 —— 用 activation（第 2 天的 sigmoid 或 ReLU）把 `z` 弯一下，得到输出 `a`
why: 那个弯正是让网络不只是一条直线的东西；`a` 就是神经元的猜测
step: 尝味道 —— 把 `a` 和真正的答案比一比，得到今天这个错误数字
why: 那就是昨天的 loss，我们这座山的高度
step: 笔记 —— 每做出一个中间值就把 `x`、`z`、`a` 存下来
why: 这些就是贴在蛋糕模上的便利贴，回程会一张一张全都用到
%%%

最后那一级是安静但要命的一级。现在把中间值存下来，回程就几乎不要钱。跳过它们，你问的每一个问题都得把整个蛋糕重烤一次。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="The neuron's forward pass as a conveyor: input x flows into weighted-sum-plus-bias giving z, then activation giving a, then loss. Under z and a, saved-note tags show the values are stored for the backward trip."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Forward pass → guess, saving z and a as it goes</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">forward pass → 猜测，一路存下 z 和 a</text><rect x="16" y="42" width="72" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="52" y="61" text-anchor="middle" fill="#1F6280">input x</text><text class="lang-zh" x="52" y="61" text-anchor="middle" fill="#1F6280">输入 x</text><text x="94" y="61" fill="#B8AEA2">→</text><rect x="112" y="42" width="118" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="171" y="57" text-anchor="middle" fill="#9A7208" font-size="9.5">weighted sum + bias</text><text class="lang-zh" x="171" y="57" text-anchor="middle" fill="#9A7208" font-size="9.5">加权和 + bias</text><text x="171" y="68" text-anchor="middle" fill="#9A7208" font-size="9">= z</text><text x="236" y="61" fill="#B8AEA2">→</text><rect x="254" y="42" width="96" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text class="lang-en" x="302" y="57" text-anchor="middle" fill="#5E5191" font-size="9.5">activation</text><text class="lang-zh" x="302" y="57" text-anchor="middle" fill="#5E5191" font-size="9.5">activation</text><text class="lang-en" x="302" y="68" text-anchor="middle" fill="#5E5191" font-size="9">= a (guess)</text><text class="lang-zh" x="302" y="68" text-anchor="middle" fill="#5E5191" font-size="9">= a（猜测）</text><text x="356" y="61" fill="#B8AEA2">→</text><rect x="374" y="42" width="126" height="30" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="437" y="61" text-anchor="middle" fill="#C93B3B" font-size="9.5">loss (how wrong)</text><text class="lang-zh" x="437" y="61" text-anchor="middle" fill="#C93B3B" font-size="9.5">loss（有多错）</text><rect x="130" y="96" width="82" height="24" rx="3" fill="#F3ECDB" stroke="#C99A12" stroke-dasharray="3,2"/><text class="lang-en" x="171" y="112" text-anchor="middle" fill="#9A7208" font-size="9">📌 saved z</text><text class="lang-zh" x="171" y="112" text-anchor="middle" fill="#9A7208" font-size="9">📌 存下 z</text><rect x="266" y="96" width="72" height="24" rx="3" fill="#EFEAF7" stroke="#B3A6D4" stroke-dasharray="3,2"/><text class="lang-en" x="302" y="112" text-anchor="middle" fill="#5E5191" font-size="9">📌 saved a</text><text class="lang-zh" x="302" y="112" text-anchor="middle" fill="#5E5191" font-size="9">📌 存下 a</text><text class="lang-en" x="260" y="146" text-anchor="middle" fill="#6B645E">these saved notes get re-used on the trip back — nothing recomputed</text><text class="lang-zh" x="260" y="146" text-anchor="middle" fill="#6B645E">这些笔记在回程被重新读一遍 —— 什么都不用重算</text></g></svg>
%%%

%%% insight
Here's why this bites in real life: those saved notes are also the reason training a big model eats so much memory. The network is holding on to every in-between value it made, for every example in the batch, just so the backward trip can read them. When you hear "we ran out of GPU memory," this is often the culprit — sticky notes, by the billion.
%%%

There's a second gift hiding in that recipe card. Written as a *chain of operations* — multiply, add, bend, score — the neuron becomes a [[computation graph||a picture of the math as a chain of tiny operations, so each little step has a clear, simple slope of its own]]: an assembly line where each box does one tiny thing.

Why care? Because each of those **small steps** has a *simple* slope of its own — and stitching simple slopes together is the whole backward trick. On to the stitch.
~~~zh
%%% insight
这件事在现实里为什么会咬人：那些存下来的笔记，也是训练一个大模型这么吃内存的原因。网络把它做出来的每一个中间值都攥着不放，batch（批）里每一个样本的都攥着，就为了让回程能读到它们。当你听到「我们把 GPU 内存跑爆了」，问题常常就出在这里 —— 几十亿张便利贴。
%%%

那张配方卡里还藏着第二份礼物。把神经元写成一*串操作* —— 乘、加、弯、打分 —— 它就变成了一张 [[computation graph||把这些数学画成一串很小的操作，于是每一小步都有自己清楚又简单的斜率]]（计算图）：一条流水线，每个盒子只做一件很小的事。

为什么要在意？因为那些**小步子**每一个都有自己*简单*的斜率 —— 而把简单的斜率缝起来，就是整个回程花招。我们去看这一针。
~~~

@@@ concept id=c3 zh_tag="一串影响" tag="Chain of effects" zh_title="局部斜率与 chain rule" title="Local slopes and the chain rule" zh_gotit="懂了 chain rule" gotit="Got the chain rule"
Here's the one idea that makes the whole thing tick, and you already understand it from planning a **road trip**. Push the gas pedal a little harder → your *speed* goes up a little. More speed → you cover more *distance* each hour. More distance → you pay more *money* at the pump. Now ask: *"how much does one extra push on the pedal cost me?"* You don't need a mega-formula. You follow the **chain of effects** and **multiply** the little hops.
~~~zh
让整件事转起来的想法只有一个，而你从计划一次**开车旅行**里已经懂了它。油门踩重一点 → 你的*速度*就上去一点。速度更快 → 你每小时跑的*路程*更多。路程更多 → 你在加油站付的*钱*更多。现在问一句：*「多踩一下油门，要花我多少钱？」* 你不需要一个超级大公式。你顺着这**一串影响**走，把这些小跳**乘起来**。
~~~

%%% svg
<svg viewBox="0 0 520 148" role="img" aria-label="A road-trip chain with three hops: pedal affects speed, speed affects distance, distance affects money. Each of the three arrows carries one small local effect, written as times 2, times 1.5 and times 0.5. A caption says multiply all three little effects to get how the pedal affects the money."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Road trip: three little effects, multiplied</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">开车旅行：三个小影响，乘起来</text><rect x="30" y="48" width="84" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="72" y="67" text-anchor="middle" fill="#1F6280">🦶 pedal</text><text class="lang-zh" x="72" y="67" text-anchor="middle" fill="#1F6280">🦶 油门</text><rect x="166" y="48" width="84" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="208" y="67" text-anchor="middle" fill="#9A7208">speed</text><text class="lang-zh" x="208" y="67" text-anchor="middle" fill="#9A7208">速度</text><rect x="302" y="48" width="92" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="348" y="67" text-anchor="middle" fill="#9A7208">distance</text><text class="lang-zh" x="348" y="67" text-anchor="middle" fill="#9A7208">路程</text><rect x="446" y="48" width="64" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="478" y="67" text-anchor="middle" fill="#1a5c38">💰 money</text><text class="lang-zh" x="478" y="67" text-anchor="middle" fill="#1a5c38">💰 钱</text><g stroke="#C93B3B" stroke-width="2" fill="#C93B3B"><path d="M116 63 l40 0"/><polygon points="164,63 156,59 156,67"/><path d="M252 63 l40 0"/><polygon points="300,63 292,59 292,67"/><path d="M396 63 l40 0"/><polygon points="444,63 436,59 436,67"/></g><text x="140" y="94" text-anchor="middle" fill="#C93B3B">× 2</text><text class="lang-en" x="140" y="106" text-anchor="middle" fill="#6B645E" font-size="9">mph per push</text><text class="lang-zh" x="140" y="106" text-anchor="middle" fill="#6B645E" font-size="9">每踩一下的 mph</text><text x="276" y="94" text-anchor="middle" fill="#C93B3B">× 1.5</text><text class="lang-en" x="276" y="106" text-anchor="middle" fill="#6B645E" font-size="9">miles per mph</text><text class="lang-zh" x="276" y="106" text-anchor="middle" fill="#6B645E" font-size="9">每 mph 的英里</text><text x="420" y="94" text-anchor="middle" fill="#C93B3B">× 0.5</text><text class="lang-en" x="420" y="106" text-anchor="middle" fill="#6B645E" font-size="9">dollars per mile</text><text class="lang-zh" x="420" y="106" text-anchor="middle" fill="#6B645E" font-size="9">每英里的美元</text><text class="lang-en" x="260" y="132" text-anchor="middle" fill="#6B645E">three small "local" effects, multiplied → the pedal's effect on money</text><text class="lang-zh" x="260" y="132" text-anchor="middle" fill="#6B645E">三个小小的「局部」影响，乘起来 → 油门对钱的影响</text></g></svg>
%%%

**What the road-trip picture gets right:** you never need one giant formula for "pedal → money." You break it into easy one-hop effects and multiply. **Where it breaks down:** on a road trip the effects mostly go one way (more pedal → more money), but inside a network a slope can be negative — nudging a weight *up* can push the loss *down*. The multiplying still works; the sign just tells you which way to lean.

#### Two words, then we drive
Each little arrow above is a [[local derivative||the slope of just one tiny step — how much its OWN output moves when its OWN input moves a hair, ignoring the rest of the chain]] — a fancy name for one easy question: *"if I nudge this box's input a hair, how much does its output move?"* Just that **one small step** — this box, and nothing else. The rest of the chain can wait its turn.

The rule for gluing those local slopes together is the [[chain rule||to get how the far-away loss depends on an early quantity, multiply the local slopes all along the path between them]]: **multiply the slopes** all along the path. That's the whole rule. Let's drive it with the three numbers from the picture.

%%% steps
step: pedal → speed: one more push adds `2` mph — local slope = `2`
why: the first hop only knows about the pedal and the speedometer; nothing else exists yet
step: speed → distance: each mph adds `1.5` miles — running product `2 × 1.5 = 3`
why: therefore one push now buys 3 extra miles; we are carrying the effect forward by multiplying
step: distance → money: each mile costs `$0.5` at the pump — running product `3 × 0.5 = 1.5`
why: that's the answer — one extra push on the pedal costs about **$1.50**
step: notice what we never wrote: a single "pedal → money" formula
why: three easy one-hop slopes, multiplied. That's why this scales to a network of thousands of steps
%%%

Watch that running product grow, hop by hop:
~~~zh
**开车旅行这个比喻对在哪里：** 你从来不需要一个「油门 → 钱」的巨大公式。你把它拆成好算的一跳一跳，然后相乘。**它在哪里不成立：** 开车的时候影响基本上只往一个方向走：多踩油门，就多花钱。但在网络里一个斜率可以是负的。把一个 weight 往*上*挪，可能把 loss 往*下*推。乘法照样成立；正负号只是告诉你该往哪边倾。

#### 两个词，然后就上路
上面每一支小箭头都是一个 [[local derivative||只有一小步的斜率 —— 当它*自己*的输入挪一根头发时，它*自己*的输出挪多少，不管链条上其他部分]]（局部导数）—— 这个花哨的名字问的是一个好回答的问题：*「我把这个盒子的输入挪一根头发，它的输出挪多少？」* 就只有那**一小步** —— 这个盒子，别的都不算。链条上剩下的可以排队等着。

把这些局部斜率粘起来的规则就是 [[chain rule||想知道远处的 loss 怎么依赖一个早期的量，就把它们之间这条路上的局部斜率全部乘起来]]：**把斜率乘起来**，沿着整条路乘。这就是全部的规则。我们用图里那三个数字来开一遍。

%%% steps
step: 油门 → 速度：多踩一下加 `2` mph —— 局部斜率 = `2`
why: 第一跳只知道油门和速度表；别的东西还不存在
step: 速度 → 路程：每 1 mph 多加 `1.5` 英里 —— 累积乘积 `2 × 1.5 = 3`
why: 因此现在多踩一下能买到 3 英里；我们靠相乘把影响往前带
step: 路程 → 钱：加油站每英里花 `$0.5` —— 累积乘积 `3 × 0.5 = 1.5`
why: 这就是答案 —— 油门多踩一下大约花 **$1.50**
step: 注意我们从来没写过什么：一个「油门 → 钱」的公式
why: 三个好算的一跳斜率，乘起来。这就是它为什么能撑起一个几千步的网络
%%%

看着那个累积乘积一跳一跳长起来：
~~~

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A running-product ladder for the chain rule. Three multiply steps: after pedal to speed the product is 2, after speed to distance it is 3, after distance to money it is 1.5. Bars show the running product changing at each hop."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Running product: multiply one local slope at a time</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">累积乘积：一次乘一个局部斜率</text><line x1="40" y1="118" x2="500" y2="118" stroke="#E5DFD6" stroke-width="1.5"/><rect x="70" y="58" width="66" height="60" fill="#2A7B9B"/><text x="103" y="52" text-anchor="middle" fill="#1F6280">2.0</text><text class="lang-en" x="103" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 2 (pedal→speed)</text><text class="lang-zh" x="103" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 2（油门→速度）</text><rect x="230" y="28" width="66" height="90" fill="#C99A12"/><text x="263" y="22" text-anchor="middle" fill="#9A7208">3.0</text><text class="lang-en" x="263" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 1.5 (speed→dist)</text><text class="lang-zh" x="263" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 1.5（速度→路程）</text><rect x="390" y="73" width="66" height="45" fill="#2D8B55"/><text x="423" y="67" text-anchor="middle" fill="#1a5c38">1.5</text><text class="lang-en" x="423" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 0.5 (dist→money)</text><text class="lang-zh" x="423" y="132" text-anchor="middle" fill="#6B645E" font-size="9">× 0.5（路程→钱）</text><text x="163" y="90" text-anchor="middle" fill="#C93B3B" font-size="13">×</text><text x="343" y="90" text-anchor="middle" fill="#C93B3B" font-size="13">×</text></g></svg>
%%%

%%% demo id=chain label="run it — multiply the little slopes"
predict: before you look — do the three slopes 2, 1.5 and 0.5 get ADDED into 4, or MULTIPLIED into 1.5? Guess, then reveal.
code: pedal_to_speed=2; speed_to_dist=1.5; dist_to_money=0.5; pedal_to_speed*speed_to_dist*dist_to_money
out: 1.5
take: <b>Chain rule = multiply the local slopes.</b> 2 × 1.5 × 0.5 = 1.5 — one extra push on the pedal costs about $1.50. Break a scary "how does the start affect the end?" question into easy one-hop slopes and multiply. That single move is the engine of backprop.
%%%

#### One last word, and it's the one that flows
As we work back along the chain, the slope we've built up *so far* — the running product arriving at a box from the loss side — is called the [[upstream gradient||the gradient that has already arrived at a step from the loss side; multiply it by the step's own local slope to pass it further back]]. It was handed over by the box **immediately downstream** of you — the next box along toward the loss. (When engineers say **error signal**, this travelling number is what they mean: not the loss itself, but the blame flowing backward.) The rhythm at every box is a two-beat move:

%%% steps
step: take the incoming gradient (what arrived from the loss side)
why: you don't recompute anything — the box downstream already did that work for you
step: multiply it by THIS box's own local slope, then pass it further back
why: the box you hand it to now sees YOUR number as its incoming gradient — that's how the slope flows
%%%

%%% insight
Stop and appreciate this, because it's the reason deep networks are possible at all: each box only needs to know two things — what arrived from downstream, and its own one-step slope. It never needs to know the whole network. That locality is why the same trick works for 2 layers or 200.
%%%

If the road-trip multiply felt fiddly, that's normal — it is the one genuinely new idea today, and everything else is this idea reused. You just met the engine of backprop. Next we start it up.
~~~zh
%%% demo id=chainzh label="先猜，再展开"
predict: 先别看 —— 2、1.5、0.5 这三个斜率是**加**成 4，还是**乘**成 1.5？先猜，再展开。
code: pedal_to_speed=2; speed_to_dist=1.5; dist_to_money=0.5; pedal_to_speed*speed_to_dist*dist_to_money
out: 1.5
take: <b>chain rule = 把局部斜率乘起来。</b>2 × 1.5 × 0.5 = 1.5 —— 油门多踩一下大约花 $1.50。把一个吓人的「开头怎么影响结尾？」问题拆成好算的一跳斜率，再相乘。这一个动作就是 backprop 的引擎。
%%%

#### 最后一个词，而它是会流动的那个
我们沿着链条往回走的时候，*到目前为止*攒起来的那个斜率 —— 从 loss 那一侧到达某个盒子的累积乘积 —— 叫做 [[upstream gradient||已经从 loss 那一侧到达某一步的 gradient；把它乘上这一步自己的局部斜率，就能继续往回传]]（上游 gradient）。它是**紧挨着你下游**的那个盒子交过来的，也就是朝 loss 方向的下一个盒子。（工程师说 **error signal（误差信号）** 的时候，指的就是这个在旅行的数字：不是 loss 本身，而是往回流的责任。）在每一个盒子上，节奏都是两拍：

%%% steps
step: 接住送进来的 gradient（从 loss 那一侧到的那个）
why: 你什么都不用重算 —— 下游那个盒子已经替你做完了那份活
step: 把它乘上**这个**盒子自己的局部斜率，然后继续往回传
why: 你交给的那个盒子现在把**你的**数字当成它送进来的 gradient —— 斜率就是这样流动的
%%%

%%% insight
停一下好好欣赏这件事，因为它正是深层网络能存在的原因：每个盒子只需要知道两件事 —— 从下游到的是什么，还有它自己一步的斜率。它从来不需要知道整个网络。正是这种「只看身边」的性质，让同一个花招在 2 层和 200 层上都好用。
%%%

如果那个开车旅行的乘法让你觉得有点绕，这很正常 —— 它是今天唯一一个真正新的想法，剩下的全都是这个想法在重复使用。你刚刚认识了 backprop 的引擎。接下来我们把它点着。
~~~

@@@ concept id=c4 zh_tag="往回追责任" tag="Trace the blame" zh_title="backward pass —— 一次扫过，分完责任" title="The backward pass — one sweep, all the blame" zh_gotit="懂了 backprop" gotit="Got backprop"
Now the trick clicks into place. Imagine a student who just got a test back with a low score. The lazy way to improve: re-take the *entire* test after changing one study habit, then re-take it again after changing the next — a fresh full test per habit. Exhausting, and most of the work is repeated. The smart student does something better: she starts from the wrong answers and traces the blame *backward* — *"this mistake came from that step, which came from this habit"* — giving each habit its fair share in **one careful pass** over the graded test.
~~~zh
现在这个花招要「咔」地一声归位了。想象一个学生，刚拿回一份分数很低的考卷。想提高的笨办法是：改一个学习习惯，然后把*整张*考卷重考一遍，再改下一个习惯，再重考一遍 —— 每个习惯配一整张新考卷。累人，而且绝大部分活都是重复的。聪明的学生会做得更好。她从做错的题目出发，把责任*往回*追 —— *「这个错来自那一步，那一步来自这个习惯」*。在批好的考卷上她**只仔细走一遍**，就把每个习惯该担的那份都分清楚了。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A student tracing blame backward from a low test score to specific study habits, in one pass. On the left a slow way redoes the whole test once per habit; on the right the smart way traces blame from the score back through each step to each habit in a single sweep."><g font-family="monospace" font-size="10"><text class="lang-en" x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">Slow: redo the whole test per habit</text><text class="lang-zh" x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">笨办法：每个习惯都把整张卷重考一遍</text><rect x="30" y="30" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="60" y="45" text-anchor="middle" fill="#C93B3B">test #1</text><text class="lang-zh" x="60" y="45" text-anchor="middle" fill="#C93B3B">第 1 张卷</text><rect x="30" y="58" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="60" y="73" text-anchor="middle" fill="#C93B3B">test #2</text><text class="lang-zh" x="60" y="73" text-anchor="middle" fill="#C93B3B">第 2 张卷</text><rect x="30" y="86" width="60" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="60" y="101" text-anchor="middle" fill="#C93B3B">test #3</text><text class="lang-zh" x="60" y="101" text-anchor="middle" fill="#C93B3B">第 3 张卷</text><text class="lang-en" x="130" y="73" fill="#6B645E">…once per habit 😩</text><text class="lang-zh" x="130" y="73" fill="#6B645E">…每个习惯一张 😩</text><line x1="258" y1="24" x2="258" y2="150" stroke="#E5DFD6"/><text class="lang-en" x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">Smart: trace blame back ONCE</text><text class="lang-zh" x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">聪明办法：往回追责任，只追一次</text><rect x="452" y="44" width="56" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="480" y="61" text-anchor="middle" fill="#C93B3B">score</text><text class="lang-zh" x="480" y="61" text-anchor="middle" fill="#C93B3B">分数</text><rect x="356" y="44" width="56" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text class="lang-en" x="384" y="61" text-anchor="middle" fill="#5E5191">step</text><text class="lang-zh" x="384" y="61" text-anchor="middle" fill="#5E5191">步骤</text><rect x="278" y="44" width="52" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="304" y="61" text-anchor="middle" fill="#1F6280">habit</text><text class="lang-zh" x="304" y="61" text-anchor="middle" fill="#1F6280">习惯</text><g stroke="#2D8B55" stroke-width="2.5" fill="#2D8B55"><path d="M452 57 l-40 0"/><polygon points="412,57 420,53 420,61"/><path d="M356 57 l-26 0"/><polygon points="330,57 338,53 338,61"/></g><text class="lang-en" x="390" y="96" text-anchor="middle" fill="#2D8B55">one backward sweep 🎯</text><text class="lang-zh" x="390" y="96" text-anchor="middle" fill="#2D8B55">往回扫一趟 🎯</text><text class="lang-en" x="390" y="112" text-anchor="middle" fill="#6B645E">every habit gets its blame at once</text><text class="lang-zh" x="390" y="112" text-anchor="middle" fill="#6B645E">每个习惯一次就拿到自己那份责任</text></g></svg>
%%%

**What the student picture gets right:** blame flows backward from the final score to each cause, and one traced pass beats redoing everything once per cause. **Where it breaks down:** a student judges her habits with fuzzy hunches; the network's blame is exact — each step's share is precisely the incoming gradient times that step's local slope, the chain rule you just met.

#### The sweep, one move at a time
The [[backward pass||backpropagation: starting at the loss, walk backward through the saved steps, multiplying local slopes, to get every weight's gradient in one sweep]] — **backpropagation** — is that smart student, made exact. Watch her walk backward through our cake-recipe graph:

%%% steps
step: start at the score — begin from the loss back, where the mistake is measured
why: this is the graded test in her hand; the very first slope is "how does the guess change the loss?"
step: one hop back — multiply by the activation's own local slope
why: same two-beat move as the road trip: incoming gradient × this box's local slope
step: read the sticky note — the activation's slope needs the `z` the forward pass saved
why: that's why we kept notes; nothing gets recomputed, it's just looked up
step: arrive at a knob — the number sitting there IS that knob's gradient
why: its share of the blame, exact, no debate
step: keep going — hand the number further back and repeat
why: one walk hands EVERY weight its gradient, because they all share the same running product
%%%
~~~zh
**这个学生的比喻对在哪里：** 责任是从最后那个分数往回流到每一个原因的，而追着走一遍比每个原因都重做一次要好。**它在哪里不成立：** 学生判断自己的习惯靠的是模模糊糊的感觉。网络分的责任是精确的。每一步该担的份额，正好等于送进来的 gradient 乘上这一步的局部斜率 —— 也就是你刚认识的 chain rule。

#### 这一趟扫过，一个动作一个动作看
[[backward pass||backpropagation：从 loss 出发，沿着存下来的步子往回走，一路乘局部斜率，一趟就拿到每一个 weight 的 gradient]] —— 也就是 **backpropagation** —— 就是那个聪明学生，只不过做到了精确。看着她在我们那张蛋糕配方图上往回走：

%%% steps
step: 从分数出发 —— 从 loss 这头往回开始，错误就是在这里量出来的
why: 这就是她手里那份批好的考卷；最开头那个斜率是「猜测怎么改变 loss？」
step: 往回跳一步 —— 乘上 activation 自己的局部斜率
why: 和开车旅行一样的两拍：送进来的 gradient × 这个盒子的局部斜率
step: 读便利贴 —— activation 的斜率需要 forward pass 存下来的那个 `z`
why: 这就是我们留笔记的原因；什么都不用重算，查一下就有
step: 到达一个旋钮 —— 停在那里的那个数字就是这个旋钮的 gradient
why: 它该担的那份责任，精确，没什么可争的
step: 继续走 —— 把这个数字再往回交，重复一次
why: 走一趟就把 gradient 发给了**每一个** weight，因为它们共用同一个累积乘积
%%%
~~~

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="The neuron's graph with green gradient arrows sweeping backward from the loss to the activation, then to the weighted sum, then to the weights. Each arrow carries a numbered marker, and a legend below names them: one, multiply by the loss slope which uses the saved a; two, multiply by the activation's own slope which uses the saved z; three, multiply by the input x, which is each weight's gradient."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Backward pass: one sweep from loss → every weight's gradient</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">backward pass：从 loss 扫一趟 → 每个 weight 的 gradient</text><rect x="16" y="42" width="64" height="30" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="48" y="61" text-anchor="middle" fill="#1F6280">weights</text><text class="lang-zh" x="48" y="61" text-anchor="middle" fill="#1F6280">weight</text><rect x="110" y="42" width="108" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="164" y="61" text-anchor="middle" fill="#9A7208" font-size="9">weighted sum → z</text><text class="lang-zh" x="164" y="61" text-anchor="middle" fill="#9A7208" font-size="9">加权和 → z</text><rect x="248" y="42" width="90" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text class="lang-en" x="293" y="61" text-anchor="middle" fill="#5E5191" font-size="9">activation → a</text><text class="lang-zh" x="293" y="61" text-anchor="middle" fill="#5E5191" font-size="9">activation → a</text><rect x="368" y="42" width="132" height="30" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="434" y="61" text-anchor="middle" fill="#C93B3B" font-size="9">loss (start here)</text><text class="lang-zh" x="434" y="61" text-anchor="middle" fill="#C93B3B" font-size="9">loss（从这里开始）</text><g stroke="#2D8B55" stroke-width="2.5" fill="#2D8B55"><path d="M368 90 l-22 0"/><polygon points="338,90 346,86 346,94"/><path d="M248 90 l-22 0"/><polygon points="218,90 226,86 226,94"/><path d="M110 90 l-22 0"/><polygon points="80,90 88,86 88,94"/></g><g fill="#2D8B55"><circle cx="353" cy="76" r="8"/><circle cx="233" cy="76" r="8"/><circle cx="95" cy="76" r="8"/></g><g fill="#ffffff" text-anchor="middle" font-size="10"><text x="353" y="80">1</text><text x="233" y="80">2</text><text x="95" y="80">3</text></g><text class="lang-en" x="44" y="110" text-anchor="middle" fill="#1a5c38" font-size="9">gradient! 🎯</text><text class="lang-zh" x="44" y="110" text-anchor="middle" fill="#1a5c38" font-size="9">gradient！🎯</text><line x1="30" y1="120" x2="490" y2="120" stroke="#E5DFD6"/><g fill="#2D8B55"><circle cx="44" cy="136" r="7"/><circle cx="44" cy="156" r="7"/><circle cx="44" cy="176" r="7"/></g><g fill="#ffffff" text-anchor="middle" font-size="9"><text x="44" y="139">1</text><text x="44" y="159">2</text><text x="44" y="179">3</text></g><g fill="#5A544E" font-size="9.5"><text class="lang-en" x="58" y="139">× the loss slope — uses the saved a 📌</text><text class="lang-zh" x="58" y="139">× loss 的斜率 —— 用存下的 a 📌</text><text class="lang-en" x="58" y="159">× the activation's own slope — uses the saved z 📌</text><text class="lang-zh" x="58" y="159">× activation 自己的斜率 —— 用存下的 z 📌</text><text class="lang-en" x="58" y="179">× the input x — and that IS each weight's gradient</text><text class="lang-zh" x="58" y="179">× 输入 x —— 这就是每个 weight 的 gradient</text></g></g></svg>
%%%

So far: forward to make a guess and leave notes, backward to spread the blame. Two sweeps, and the network knows how to improve.

#### Why this is such a big deal
Here's the part worth getting excited about. The slow way is re-deriving each weight's effect from scratch — one full run of the whole network per weight — which repeats almost all of the same work every single time. A real network has *millions* of weights, so that road is hopeless.

%%% insight
Backprop's win is [[reuse||computing every weight's gradient in a single backward sweep that shares its work, instead of redoing the chain separately for each weight]]: **one sweep** gives you every weight's gradient — **instead of re-deriving** each weight's effect separately — because they all ride the same running product. This single fact is why training a large model costs roughly *two* passes instead of a million. No backprop, no ChatGPT — the arithmetic simply would not fit in a human lifetime.
%%%

You've now got the engine and the sweep. Next: what blame actually lands on each part of the neuron.
~~~zh
到这里：往前走做出一个猜测并留下笔记，往回走把责任分下去。两趟扫过，网络就知道该怎么变好了。

#### 这为什么是件大事
这一段值得你兴奋一下。笨办法是从头重新推导每一个 weight 的影响 —— 每一个 weight 都要把整个网络完整跑一遍 —— 而每一次都在重复几乎一样的活。真的网络有*几百万*个 weight，所以那条路根本走不通。

%%% insight
backprop 赢在 [[reuse||在一趟共享工作的 backward pass 里算出每一个 weight 的 gradient，而不是为每个 weight 单独把链条重做一遍]]（复用）：**一趟扫过**就给你每一个 weight 的 gradient —— **而不是**把每个 weight 的影响**分开重新推导** —— 因为它们全都骑在同一个累积乘积上。就是这一件事，让训练一个大模型的代价大约是*两*趟，而不是一百万趟。没有 backprop 就没有 ChatGPT —— 那些算术根本塞不进一个人的一生。
%%%

现在引擎和这一趟扫过你都有了。接下来：责任到底落在神经元的哪一部分上、各是多少。
~~~

@@@ concept id=c5 zh_tag="各自的份额" tag="Each part's share" zh_title="每个旋钮拿到的 gradient —— 三条规则，再加一条上路用的" title="The gradient at each knob — three rules, plus one for the road" zh_gotit="懂了这几条规则" gotit="Got the rules"
Time to open the neuron and see who gets blamed for what. Picture a **group project** where everyone shares one grade. It only feels fair if your share matches how much you actually *did*: the teammate who wrote most of the report carries most of the responsibility, and the one who just showed up gets a light touch. The neuron splits its blame the exact same way — and it comes down to a few tiny rules, one per part.
~~~zh
该打开神经元，看看谁为了什么被怪。想象一个**小组作业**，所有人共用一个成绩。只有当你担的那份和你真正*做*的量对得上，才让人觉得公平。写了报告大部分内容的队友要担最多的责任。只是来露了个脸的那个，只被轻轻碰一下。神经元分责任的方式一模一样 —— 而它归结成几条很小的规则，每个部分一条。
~~~

%%% svg
<svg viewBox="0 0 520 158" role="img" aria-label="A group project sharing one grade. Three teammates get different-sized slices of blame: the big contributor gets a big slice, the medium contributor a medium slice, the tag-along a tiny slice. Caption: fair blame matches how much each part actually did."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Group project: your share of blame = how much you did</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">小组作业：你担的那份责任 = 你做了多少</text><rect x="40" y="40" width="120" height="80" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="100" y="34" text-anchor="middle" fill="#6B645E" font-size="9">did a lot 📝</text><text class="lang-zh" x="100" y="34" text-anchor="middle" fill="#6B645E" font-size="9">做了很多 📝</text><text class="lang-en" x="100" y="86" text-anchor="middle" fill="#C93B3B" font-size="13">big blame</text><text class="lang-zh" x="100" y="86" text-anchor="middle" fill="#C93B3B" font-size="13">责任大</text><rect x="200" y="60" width="120" height="60" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="260" y="54" text-anchor="middle" fill="#6B645E" font-size="9">did some</text><text class="lang-zh" x="260" y="54" text-anchor="middle" fill="#6B645E" font-size="9">做了一些</text><text class="lang-en" x="260" y="96" text-anchor="middle" fill="#9A7208" font-size="12">medium</text><text class="lang-zh" x="260" y="96" text-anchor="middle" fill="#9A7208" font-size="12">中等</text><rect x="360" y="98" width="120" height="22" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="420" y="92" text-anchor="middle" fill="#6B645E" font-size="9">just showed up</text><text class="lang-zh" x="420" y="92" text-anchor="middle" fill="#6B645E" font-size="9">只是来露个脸</text><text class="lang-en" x="420" y="113" text-anchor="middle" fill="#1a5c38" font-size="10">tiny blame</text><text class="lang-zh" x="420" y="113" text-anchor="middle" fill="#1a5c38" font-size="10">责任很小</text><text class="lang-en" x="260" y="142" text-anchor="middle" fill="#6B645E">the neuron splits its blame the same fair way — a few simple rules</text><text class="lang-zh" x="260" y="142" text-anchor="middle" fill="#6B645E">神经元用同样公平的方式分责任 —— 几条简单的规则</text></g></svg>
%%%

**What the group-project picture gets right:** blame is shared in proportion to contribution — do more, own more of the result. **Where it breaks down:** teammates argue about who did what; the neuron's split is exact and instant, just the chain-rule multiply.

Before the rules, one behavior to feel in your own hands. When the gradient reaches the activation it gets *multiplied* by the activation's slope — and that slope changes a lot with the input. **Predict first:** where do you think sigmoid's slope is biggest — out in the tails, or in the middle? Now drag `z` and find out:
~~~zh
**小组作业这个比喻对在哪里：** 责任是按贡献比例分的 —— 做得多，就拥有结果里更多的一份。**它在哪里不成立：** 队友会争谁做了什么；神经元的分法是精确的、也是瞬间的，就是 chain rule 那一个乘法。

在讲规则之前，先亲手感受一个行为。当 gradient 到达 activation 的时候，它会被 activation 的斜率*乘*一下 —— 而那个斜率会随输入变化很大。**先预测：** 你觉得 sigmoid 的斜率在哪里最大 —— 在两侧的尾巴上，还是在中间？现在拖动 `z` 看看：
~~~

%%% svg
<svg id="sl-svg" viewBox="0 0 520 214" role="img" aria-label="Interactive activation slope. Pick sigmoid, tanh, or ReLU and drag the input z. A dot rides the activation curve and a short tangent line shows the slope there — steep in the middle, flat in the tails. A horizontal reference line marks where the output is zero, and it moves to the right place for each activation, so tanh's negative half is visible below it. The slope is the exact number that multiplies the gradient passing back through."><g font-family="monospace" font-size="11"><rect x="60" y="30" width="400" height="160" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line id="sl-axis" x1="60" y1="190" x2="460" y2="190" stroke="#D8D2C8" stroke-width="1" stroke-dasharray="4,3"/><text id="sl-axlab" x="64" y="185" fill="#9A938A" font-size="9">output = 0</text><line x1="260" y1="30" x2="260" y2="190" stroke="#EFE9DF" stroke-width="1"/><path id="sl-curve" d="M60 190 L260 110 L460 40" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><line id="sl-tan" x1="234" y1="150" x2="286" y2="70" stroke="#C99A12" stroke-width="3"/><circle id="sl-dot" cx="260" cy="115" r="6" fill="#C99A12" stroke="#fff" stroke-width="2"/><text class="lang-en" x="466" y="40" fill="#9A938A" font-size="9">high</text><text class="lang-zh" x="466" y="40" fill="#9A938A" font-size="9">高</text><text class="lang-en" x="466" y="188" fill="#9A938A" font-size="9">low</text><text class="lang-zh" x="466" y="188" fill="#9A938A" font-size="9">低</text><text x="264" y="204" fill="#9A938A" font-size="9">z = 0</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px">
<span>activation:</span>
<button id="sl-sig" type="button" style="cursor:pointer;border:2px solid #7C6DAA;background:#EDE9F8;color:#5E5191;border-radius:5px;padding:2px 8px;font-family:monospace">sigmoid</button>
<button id="sl-tanh" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">tanh</button>
<button id="sl-relu" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">ReLU</button>
</div>
<label>drag z <input id="sl-z" type="range" min="-6" max="6" step="0.2" value="0" style="width:60%;accent-color:#C99A12;vertical-align:middle"></label>
<div id="sl-out" style="margin-top:6px;color:#2C2A28">at z = 0.0, sigmoid slope = <b>0.25</b> → a gradient of 1.00 passes back as <b>0.25</b> <span style="color:#C99A12">(shrinks to a quarter)</span></div>
<div style="margin-top:4px;color:#6B645E;font-size:.92em">Switch to <b>tanh</b> and watch the dashed "output = 0" line jump to the middle of the box — tanh is the one activation here that can go <i>negative</i>, and its curve dips below that line. Note on ReLU at exactly <code>z = 0</code>: the curve has a sharp corner there, so it has no single slope — the slope is 0 just to the left and 1 just to the right. Everyone simply picks one by convention (we show 1); nudge <code>z</code> either side and you'll see the honest values.</div>
</div>
<script>(function(){
  var zs=document.getElementById('sl-z');if(!zs)return;
  var curve=document.getElementById('sl-curve'),dot=document.getElementById('sl-dot'),tan=document.getElementById('sl-tan'),
      out=document.getElementById('sl-out'),axis=document.getElementById('sl-axis'),axlab=document.getElementById('sl-axlab'),
      bSig=document.getElementById('sl-sig'),bTanh=document.getElementById('sl-tanh'),bRelu=document.getElementById('sl-relu');
  var act='sigmoid';
  var X0=60,X1=460,ZLO=-6,ZHI=6;
  function xpx(z){return X0+(z-ZLO)/(ZHI-ZLO)*(X1-X0);}
  var xscale=(X1-X0)/(ZHI-ZLO);
  var defs={
    sigmoid:{f:function(z){return 1/(1+Math.exp(-z));},d:function(z){var s=1/(1+Math.exp(-z));return s*(1-s);},ypx:function(f){return 190-f*150;},yscale:150},
    tanh:{f:function(z){return Math.tanh(z);},d:function(z){var t=Math.tanh(z);return 1-t*t;},ypx:function(f){return 110-f*70;},yscale:70},
    relu:{f:function(z){return Math.max(0,z);},d:function(z){return z>=0?1:0;},ypx:function(f){return 190-f/6*150;},yscale:25}
  };
  function setBtns(){[[bSig,'sigmoid'],[bTanh,'tanh'],[bRelu,'relu']].forEach(function(p){
    var sel=(p[1]===act);
    p[0].style.borderColor=sel?'#7C6DAA':'#E5DFD6';p[0].style.background=sel?'#EDE9F8':'#FDF9F3';p[0].style.color=sel?'#5E5191':'#6B645E';});}
  function paint(){
    var D=defs[act],z=+zs.value;
    // put the "output = 0" reference line where THIS activation's zero actually is
    var zeroY=D.ypx(0);
    axis.setAttribute('y1',zeroY.toFixed(1));axis.setAttribute('y2',zeroY.toFixed(1));
    axlab.setAttribute('y',(zeroY-5).toFixed(1));
    var d='';for(var i=0;i<=80;i++){var zz=ZLO+(ZHI-ZLO)*i/80;d+=(i?'L':'M')+xpx(zz).toFixed(1)+' '+D.ypx(D.f(zz)).toFixed(1)+' ';}
    curve.setAttribute('d',d.trim());
    var dotx=xpx(z),doty=D.ypx(D.f(z)),sl=D.d(z);
    dot.setAttribute('cx',dotx.toFixed(1));dot.setAttribute('cy',doty.toFixed(1));
    var pxslope=sl*D.yscale/xscale,L=28;
    tan.setAttribute('x1',(dotx-L).toFixed(1));tan.setAttribute('y1',(doty+L*pxslope).toFixed(1));
    tan.setAttribute('x2',(dotx+L).toFixed(1));tan.setAttribute('y2',(doty-L*pxslope).toFixed(1));
    var col=sl>=0.5?'#2D8B55':(sl>=0.1?'#C99A12':'#C93B3B');
    var word=sl>=0.5?'passes strongly':(sl>=0.1?'shrinks':(sl>0?'nearly dies (flat tail)':'a full stop — nothing gets through'));
    var kink=(act==='relu'&&Math.abs(z)<0.001)?' <span style="color:#6B645E">— right at the corner, so this is the convention, not a real slope</span>':'';
    tan.setAttribute('stroke',col);dot.setAttribute('fill',col);
    out.innerHTML='at z = '+z.toFixed(1)+', '+act+' slope = <b>'+sl.toFixed(2)+'</b> → a gradient of 1.00 passes back as <b>'+sl.toFixed(2)+'</b> <span style="color:'+col+'">('+word+')</span>'+kink;
  }
  zs.addEventListener('input',paint);
  bSig.addEventListener('click',function(){act='sigmoid';setBtns();paint();});
  bTanh.addEventListener('click',function(){act='tanh';setBtns();paint();});
  bRelu.addEventListener('click',function(){act='relu';setBtns();paint();});
  setBtns();paint();
})();</script>
%%%

**What that slider is telling you:** the gradient does not pass through the activation untouched — the **activation's slope multiplies** it, using the `z` your forward pass saved. Biggest in the middle (and even at its peak, sigmoid slope only reaches about `0.25`), nearly flat in the tails. Keep that number `0.25` in your pocket; it's the villain of the next unit.

Now the rules, in the order the backward walk actually meets them: first the activation takes its cut, then whatever survives gets split between the weight and the bias — and finally one extra rule for travelling further back.

#### Rule 1 · the activation charges a toll
Whatever gradient arrives at the neuron from the loss side is first scaled by the activation's slope at the saved `z` — the thing you just felt on the slider. Call the number that gets *through* the toll `δ` ("delta"); that is the gradient that reaches the knobs.

- **[[sigmoid||an S-shaped activation that squashes numbers into 0–1; its slope is σ(z)(1−σ(z)), which peaks at only 0.25]]** — the **sigmoid slope** `σ(z)(1−σ(z))` tops out near `0.25` in the middle, and goes tiny in the tails. An expensive toll.
- **[[ReLU||an activation that passes positives and zeros negatives; its slope is a clean 1 for positive inputs, 0 for negative]]** — the **ReLU slope** is a clean `1` (positive `z`) or `0` (negative `z`). Free passage, or a full stop.

#### Rule 2 · the weight's share = its input × the gradient that got through the toll
A weight only touches the answer *through* the input it multiplies. So take `δ` — the gradient that survived the toll — and multiply it by that weight's own **input value**. Nothing else joins in. That one multiply is the **weight's gradient**, the whole story.

%%% steps
step: the loud teammate — a weight fed a BIG input gets a big gradient
why: it had more say in the answer, therefore it owns more of the blame
step: the quiet teammate — a weight fed a tiny input barely moves
why: it hardly affected the guess, so correcting it would hardly help
step: the rule of thumb — **big inputs** cause big weight updates
why: that's also why we scale our data later; a runaway input makes a runaway update
%%%

#### Rule 3 · the bias travels free
The [[bias||the neuron's constant offset, added after the weighted sum; nudging it moves z one-for-one]] has **slope 1** — nudge it by a hair and `z` moves by exactly that hair. So `δ` **passes straight through** to the bias, unchanged. No second toll at all.

#### Rule 4 · one for the road — going further back, multiply by the weight
Rules 1–3 finish this neuron's own knobs. But in a real network the input `x` was itself the *output of the neuron behind it*, so the sweep has to keep travelling. The weighted-sum box has one more local slope, pointing back the way we came: nudge the **input** by a hair and `z` moves by `w` — the weight it gets multiplied by. Therefore the gradient handed to the previous layer is `δ × w`.
~~~zh
**那个滑块在告诉你什么：** gradient 不会原样穿过 activation —— **activation 的斜率会乘**它一下，用的就是你 forward pass 存下来的那个 `z`。中间最大（而且就算在最高点，sigmoid 的斜率也只到大约 `0.25`），两侧的尾巴上几乎是平的。把 `0.25` 这个数字揣在兜里；它是下一个单元里的坏人。

现在讲规则。顺序就是往回走的时候真正遇到它们的顺序：先是 activation 收它那一份。然后活下来的部分在 weight 和 bias 之间分。最后再加一条继续往回走用的规则。

#### 规则 1 · activation 收过路费
从 loss 那一侧到达神经元的 gradient，先被 activation 在存下的 `z` 处的斜率缩放一下 —— 就是你刚在滑块上摸到的那个东西。把*通过*收费站的那个数字叫 `δ`（读作 delta）；那就是真正到达旋钮的 gradient。

- **[[sigmoid||一个 S 形的 activation，把数字压进 0–1；它的斜率是 σ(z)(1−σ(z))，最高只到 0.25]]** —— **sigmoid 的斜率** `σ(z)(1−σ(z))` 在中间最高也就 `0.25`，在尾巴上小到不行。一个很贵的过路费。
- **[[ReLU||一个 activation，正数原样通过，负数变成零；它对正输入的斜率是干净的 1，对负输入是 0]]** —— **ReLU 的斜率**是干净的 `1`（`z` 为正）或者 `0`（`z` 为负）。要么免费通行，要么彻底停住。

#### 规则 2 · weight 该担的 = 它的输入 × 通过收费站的那个 gradient
一个 weight 只能*通过*它乘的那个输入去碰答案。所以拿 `δ` —— 活着通过收费站的那个 gradient —— 乘上这个 weight 自己的**输入值**。别的什么都不参与。那一个乘法就是**这个 weight 的 gradient**，全部故事。

%%% steps
step: 嗓门大的队友 —— 喂进一个**大**输入的 weight 拿到一个大 gradient
why: 它对答案的话语权更多，因此它拥有更多的责任
step: 安静的队友 —— 喂进一个很小输入的 weight 几乎不动
why: 它几乎没影响那个猜测，所以修它也几乎没什么帮助
step: 经验法则 —— **大输入**造成大的 weight update
why: 这也是我们以后要给数据做缩放的原因；一个失控的输入会造出一个失控的 update
%%%

#### 规则 3 · bias 免费通行
[[bias||神经元那个固定的偏移量，加在加权和之后；把它挪一点，z 就一比一地挪同样多]] 的**斜率是 1** —— 把它挪一根头发，`z` 就正好挪那么一根头发。所以 `δ` 原样不变地**直接穿过**到 bias。完全没有第二个收费站。

#### 规则 4 · 上路用的那一条 —— 要继续往回走，乘上 weight
规则 1 到 3 把这个神经元自己的旋钮处理完了。但在真的网络里，输入 `x` 本身就是*它后面那个神经元的输出*，所以这一趟还得继续走。加权和那个盒子还有一个局部斜率，指着我们来的方向：把**输入**挪一根头发，`z` 就挪 `w` 那么多 —— 也就是它被乘上的那个 weight。因此交给前一层的 gradient 是 `δ × w`。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="The weighted-sum box with three exits for the gradient. Delta arrives from the activation. Exit one goes down to the weight and is delta times the input x. Exit two goes down to the bias and is delta times one. Exit three goes left, further back to the previous layer, and is delta times the weight w. A caption says the weight matters twice: once as a knob to fix, once as the toll on the road back."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Three exits out of the weighted-sum box</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">从加权和盒子出去的三个出口</text><rect x="196" y="34" width="132" height="34" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><text x="262" y="55" text-anchor="middle" fill="#9A7208">z = w·x + b</text><text class="lang-en" x="400" y="55" fill="#5A544E" font-size="9.5">δ arrives here</text><text class="lang-zh" x="400" y="55" fill="#5A544E" font-size="9.5">δ 从这里进来</text><g stroke="#2D8B55" stroke-width="2.2" fill="#2D8B55"><path d="M392 51 l-56 0"/><polygon points="336,51 344,47 344,55"/><path d="M240 68 l0 34"/><polygon points="240,102 236,94 244,94"/><path d="M290 68 l0 34"/><polygon points="290,102 286,94 294,94"/><path d="M196 51 l-56 0"/><polygon points="140,51 148,47 148,55"/></g><rect x="188" y="106" width="104" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="240" y="123" text-anchor="middle" fill="#1F6280" font-size="9">weight: δ × x</text><text class="lang-zh" x="240" y="123" text-anchor="middle" fill="#1F6280" font-size="9">weight：δ × x</text><rect x="300" y="106" width="96" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="348" y="123" text-anchor="middle" fill="#1F6280" font-size="9">bias: δ × 1</text><text class="lang-zh" x="348" y="123" text-anchor="middle" fill="#1F6280" font-size="9">bias：δ × 1</text><rect x="18" y="38" width="118" height="28" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text class="lang-en" x="77" y="56" text-anchor="middle" fill="#5E5191" font-size="9">back a layer: δ × w</text><text class="lang-zh" x="77" y="56" text-anchor="middle" fill="#5E5191" font-size="9">往回一层：δ × w</text><text class="lang-en" x="77" y="80" text-anchor="middle" fill="#6B645E" font-size="9">← keeps the sweep going</text><text class="lang-zh" x="77" y="80" text-anchor="middle" fill="#6B645E" font-size="9">← 让这一趟继续走</text><line x1="30" y1="146" x2="490" y2="146" stroke="#E5DFD6"/><text class="lang-en" x="260" y="166" text-anchor="middle" fill="#5A544E" font-size="9.5">the weight matters twice: as a knob to fix (δ × x), and as the toll on the road back (δ × w)</text><text class="lang-zh" x="260" y="166" text-anchor="middle" fill="#5A544E" font-size="9.5">weight 起两次作用：一次是要修的旋钮（δ × x），一次是回程路上的过路费（δ × w）</text></g></svg>
%%%

That little `× w` looks harmless. Hold on to it — it is the reason the next unit's dial can go both *below* and *above* 1, and the reason "size your starting weights sensibly" is a real cure and not just advice.

%%% demo id=rules label="run it — split the blame three ways"
predict: the input is `x = 2.0` and the same gradient `0.3` arrives at the neuron. Will the weight's gradient come out BIGGER than the bias's, smaller, or exactly equal? By what factor?
code: x=2.0; w=0.5; grad_into_neuron=0.3; relu_slope=1.0  # input, weight, gradient arriving from the loss side, ReLU slope for positive z
code: delta=grad_into_neuron*relu_slope           # Rule 1: pay the activation's toll
code: w_grad=delta*x; b_grad=delta*1.0            # Rule 2: × its input · Rule 3: bias travels free
code: back=delta*w                                # Rule 4: what gets handed to the previous layer
code: print("delta",delta," weight grad",w_grad," bias grad",b_grad," passed back",back)
out: delta 0.3  weight grad 0.6  bias grad 0.3  passed back 0.15
take: <b>Big input → big weight gradient.</b> The toll was free here (ReLU slope 1), so δ = 0.3. The weight's blame is δ × its input = 0.3 × 2.0 = 0.6 — exactly twice the bias's 0.3, because its input was 2×. And what travels one layer further back is δ × w = 0.3 × 0.5 = 0.15: this weight was smaller than 1, so the signal got quieter on the way. Same chain-rule multiply, four exits.
%%%

%%% insight
Did you catch the shape of all four rules? They are the *same* move — incoming gradient × one local slope — wearing four different hats. That's the whole reason a beginner can hold backprop in their head: there is only ever one rule, applied at whatever box you happen to be standing on.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the rules in symbols.</b> Peek only if you like notation; the plain lines above are all you need. With loss slope <code>g = ∂L/∂a</code> arriving at the neuron and activation slope <code>a'(z)</code>: Rule 1 is <code>δ = g · a'(z)</code>; Rule 2 is <code>∂L/∂wᵢ = δ · xᵢ</code>; Rule 3 is <code>∂L/∂b = δ · 1 = δ</code>; Rule 4 is <code>∂L/∂xᵢ = δ · wᵢ</code>. Every symbol is just "incoming gradient × one local slope."
!!!

Those rules *are* the neuron's backward pass — and notice each one is safe and boring on its own: a multiply by a slope, by an input, by a plain `1`, or by a weight. Nice work; that's the mechanism, done. The drama only starts when you **chain many of them together**, which is exactly the next puzzle.
~~~zh
那个小小的 `× w` 看起来没什么杀伤力。抓紧它。它就是下一个单元那个旋钮既能转到 1 *以下*、也能转到 1 *以上*的原因。它也是为什么「把起始 weight 的大小定得合理一点」是一个真解药，而不只是一句建议。

%%% demo id=ruleszh label="先猜，再展开"
predict: 输入是 `x = 2.0`，同一个 gradient `0.3` 到达神经元。weight 的 gradient 会比 bias 的**更大**、更小，还是正好相等？差几倍？
code: x=2.0; w=0.5; grad_into_neuron=0.3; relu_slope=1.0  # 输入、weight、从 loss 那侧到达的 gradient、z 为正时的 ReLU 斜率
code: delta=grad_into_neuron*relu_slope           # 规则 1：交 activation 的过路费
code: w_grad=delta*x; b_grad=delta*1.0            # 规则 2：× 它的输入 · 规则 3：bias 免费通行
code: back=delta*w                                # 规则 4：交给前一层的东西
code: print("delta",delta," weight grad",w_grad," bias grad",b_grad," passed back",back)
out: delta 0.3  weight grad 0.6  bias grad 0.3  passed back 0.15
take: <b>大输入 → 大的 weight gradient。</b>这里的过路费是免费的（ReLU 斜率 1），所以 δ = 0.3。weight 该担的是 δ × 它的输入 = 0.3 × 2.0 = 0.6 —— 正好是 bias 那个 0.3 的两倍，因为它的输入是 2 倍。而再往回走一层的是 δ × w = 0.3 × 0.5 = 0.15：这个 weight 比 1 小，所以信号在路上变轻了。同一个 chain rule 乘法，四个出口。
%%%

%%% insight
你注意到这四条规则的形状了吗？它们是*同一个*动作 —— 送进来的 gradient × 一个局部斜率 —— 只是戴了四顶不同的帽子。这就是一个初学者能把 backprop 装在脑子里的全部原因：永远只有一条规则，在你正好站着的那个盒子上用一次。
%%%

!!! c-info 🪜
<b>可跳过的选修 —— 用符号写这几条规则。</b>只有你喜欢记号法才用看；上面那些大白话就够了。设到达神经元的 loss 斜率为 <code>g = ∂L/∂a</code>，activation 斜率为 <code>a'(z)</code>：规则 1 是 <code>δ = g · a'(z)</code>；规则 2 是 <code>∂L/∂wᵢ = δ · xᵢ</code>；规则 3 是 <code>∂L/∂b = δ · 1 = δ</code>；规则 4 是 <code>∂L/∂xᵢ = δ · wᵢ</code>。每一个符号都只是「送进来的 gradient × 一个局部斜率」。
!!!

这几条规则*就是*神经元的 backward pass —— 而注意每一条单独看都安全又无聊：乘一个斜率、乘一个输入、乘一个普通的 `1`、或者乘一个 weight。做得好；机制到这里讲完了。只有当你**把很多条串起来**的时候，戏才开始 —— 而那正好就是下一个谜题。
~~~

@@@ concept id=c6 zh_tag="变弱的信号" tag="The fading signal" zh_title="谜题 —— 下坡信号弱到几乎没有" title="Puzzle — when the downhill signal fades to nothing" zh_gotit="懂了 vanishing gradient" gotit="Got vanishing gradients"
Here's your first puzzle, and it's the most famous one in deep learning. Play the **telephone game**: a whisper passed down a long line of kids, each repeating it a little quieter, until the last kid hears nothing at all. Now hold that picture next to what you just learned — travelling back one layer means multiplying by *two* small things (the activation's slope, then the weight), and you saw sigmoid's slope is at most `0.25`. So what reaches the *first* layer after four of those multiplies?
~~~zh
这是你的第一个谜题，而它是深度学习里最有名的那一个。玩一次**传话游戏**：一句耳语在一长排小孩之间传下去，每个人复述的时候都轻一点，直到最后那个小孩什么都听不见。现在把这幅画和你刚学到的东西并排放着看 —— 往回走一层意味着乘上*两*个小东西（先是 activation 的斜率，然后是 weight），而你已经看到 sigmoid 的斜率最多只有 `0.25`。那么这样乘四次之后，到达*第一*层的还剩多少？
~~~

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A line of kids playing telephone. A whisper starts loud on the right and gets quieter at each kid to the left, shrinking to nothing. Caption: multiply by a small amount at every layer and the gradient fades to zero."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Telephone: each layer multiplies by a small amount → the whisper fades</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">传话游戏：每一层都乘一个很小的量 → 耳语变弱</text><text x="470" y="70" font-size="24">🧒</text><text class="lang-en" x="470" y="98" text-anchor="middle" fill="#C93B3B" font-size="10">loud</text><text class="lang-zh" x="470" y="98" text-anchor="middle" fill="#C93B3B" font-size="10">响</text><text x="368" y="70" font-size="22">🧒</text><text x="378" y="98" text-anchor="middle" fill="#C99A12" font-size="10">×0.25</text><text x="270" y="70" font-size="19">🧒</text><text x="280" y="98" text-anchor="middle" fill="#C99A12" font-size="10">×0.25</text><text x="176" y="70" font-size="16">🧒</text><text x="184" y="98" text-anchor="middle" fill="#9A938A" font-size="10">×0.25</text><text x="92" y="70" font-size="12">🧒</text><text x="96" y="98" text-anchor="middle" fill="#B8AEA2" font-size="10">×0.25</text><text x="40" y="70" font-size="9" fill="#B8AEA2">·</text><g stroke="#B8AEA2" stroke-width="1.5" fill="#B8AEA2"><path d="M460 60 l-70 0"/><polygon points="390,60 398,56 398,64"/><path d="M360 60 l-70 0"/><polygon points="290,60 298,56 298,64"/><path d="M262 60 l-70 0"/><polygon points="192,60 200,56 200,64"/><path d="M168 60 l-64 0"/><polygon points="104,60 112,56 112,64"/></g><text class="lang-en" x="60" y="130" fill="#9A938A" font-size="10">last kid hears ≈ nothing 😴</text><text class="lang-zh" x="60" y="130" fill="#9A938A" font-size="10">最后一个小孩听到的 ≈ 没有 😴</text><text class="lang-en" x="470" y="130" text-anchor="end" fill="#C93B3B" font-size="10">loss end (loud) →</text><text class="lang-zh" x="470" y="130" text-anchor="end" fill="#C93B3B" font-size="10">loss 那一端（响）→</text></g></svg>
%%%

%%% hint
t1: You already know one sigmoid layer multiplies the gradient by a slope that is at most 0.25. What happens to that number when a SECOND layer multiplies by 0.25 again?
t2: Multiply it out one layer at a time: 0.25, then 0.25 × 0.25 = 0.0625, then × 0.25 again = 0.0156. Each layer makes it about four times smaller.
t3: Stacking many layers means multiplying many numbers below 1, so the gradient shrinks toward 0 and the early layers stop learning — that's the vanishing gradient.
%%%

**What the telephone picture gets right:** repeatedly shrinking a signal, step after step, drives it to almost nothing — and the *far* end (the early layers) suffers most. **Where it breaks down:** in telephone the message also gets garbled; here the gradient stays perfectly "correct," it just gets so tiny that the early weights barely move.

#### Walk down the line of kids
This is the [[vanishing gradient||when many small per-layer multiplies stack up, the gradient shrinks toward 0, so early layers stop learning]] problem. To keep the numbers friendly, say each layer's weights are sized so the `× w` part is about `1`, leaving the sigmoid's `0.25` to do the shrinking. Let's read the volume at each kid:

%%% steps
step: kid 1 (the last layer) — gradient `1` × `0.25` = `0.25`
why: one sigmoid toll booth already took three quarters of the signal
step: kid 2 — `× 0.25` again = `0.0625`
why: the shrink compounds; each layer makes the whisper about four times quieter
step: kid 3 — `0.0156`
why: we are still only three layers deep and the signal is already 1.5% of what it was
step: kid 4 (the FIRST layer) — under `0.004`
why: that's the freeze — the early layers get almost no downhill signal, so they stop learning
%%%

Now put your own hand on the dial. It sets **what each layer multiplies the gradient by** — the activation's slope times the weight, the two tolls from Rule 1 and Rule 4 rolled into one number. **Predict first:** to keep the gradient alive across 8 layers, does that number need to be *below* 1, *about* 1, or *above* 1? Drag and see — and while you're there, push it past 1 and watch what happens the other way:
~~~zh
%%% hint
t1: 你已经知道一个 sigmoid 层会把 gradient 乘上一个最多 0.25 的斜率。当**第二**个层又乘一次 0.25，那个数字会变成什么样？
t2: 一层一层乘出来看：0.25，然后 0.25 × 0.25 = 0.0625，再 × 0.25 一次 = 0.0156。每一层都让它小大约四倍。
t3: 叠很多层就意味着乘很多个小于 1 的数字，于是 gradient 一路缩向 0，早期的层停止学习 —— 那就是 vanishing gradient。
%%%

**传话游戏这个比喻对在哪里：** 一个信号被反复缩小，一步接一步，最后就几乎什么都不剩 —— 而*远*的那一端（也就是早期的层）受害最重。**它在哪里不成立：** 传话游戏里消息还会被传歪；这里 gradient 一直完全「正确」，它只是小到早期的 weight 几乎不动。

#### 沿着这排小孩走一遍
这就是 [[vanishing gradient||当很多个每层的小乘数叠起来，gradient 一路缩向 0，于是早期的层停止学习]]（梯度消失）问题。为了让数字友好一点，就说每一层的 weight 大小定得刚好让 `× w` 那部分差不多是 `1`，把缩小的活全留给 sigmoid 那个 `0.25`。我们来读一读每个小孩那里的音量：

%%% steps
step: 第 1 个小孩（最后一层）—— gradient `1` × `0.25` = `0.25`
why: 一个 sigmoid 收费站就已经拿走了信号的四分之三
step: 第 2 个小孩 —— 再 `× 0.25` = `0.0625`
why: 缩小会叠加；每一层让这句耳语大约轻四倍
step: 第 3 个小孩 —— `0.0156`
why: 我们还只深了三层，信号就已经只剩原来的 1.5% 了
step: 第 4 个小孩（也就是**第一**层）—— 不到 `0.004`
why: 这就是冻住 —— 早期的层几乎拿不到下坡信号，所以它们停止学习
%%%

现在把你自己的手放到那个旋钮上。它设定的是**每一层把 gradient 乘上多少** —— 也就是 activation 的斜率乘上 weight，把规则 1 和规则 4 那两个收费站合成了一个数字。**先预测：** 要让 gradient 活着穿过 8 层，那个数字需要*小于* 1、*差不多* 1，还是*大于* 1？拖一下看看 —— 顺便把它推过 1，看看反方向会发生什么：
~~~

%%% svg
<svg id="vg-svg" viewBox="0 0 520 200" role="img" aria-label="Interactive gradient-through-depth ladder. A slider sets what each layer multiplies the gradient by, which is the activation slope times the weight, and eight bars show the running product of the gradient after it has travelled back one, two, up to eight layers. Bar heights are on a logarithmic scale, so an equal drop in height means an equal multiplying cut. At 0.25 per layer the bars step down evenly and the printed values collapse toward zero, which is the vanishing gradient. At 1 the bars stay level. Above 1 the bars run off the top, which is the exploding gradient."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The gradient after each layer — you set the per-layer multiply</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">每一层之后的 gradient —— 每层乘多少由你定</text><text class="lang-en" x="34" y="31" fill="#9A938A" font-size="9">bar height = LOG scale · each equal drop is an equal ×cut — read the printed number for true size</text><text class="lang-zh" x="34" y="31" fill="#9A938A" font-size="9">条形高度 = 对数刻度 · 每次等高的下降 = 同样倍数的削减 —— 真实大小请看打印的数字</text><line x1="34" y1="150" x2="500" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><line x1="34" y1="67.5" x2="500" y2="67.5" stroke="#B8AEA2" stroke-width="1" stroke-dasharray="4,3"/><text class="lang-en" x="500" y="63" text-anchor="end" fill="#9A938A" font-size="9">= 1.0 · signal unchanged</text><text class="lang-zh" x="500" y="63" text-anchor="end" fill="#9A938A" font-size="9">= 1.0 · 信号不变</text><g id="vg-bars"><rect id="vg-b0" x="46" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b1" x="100" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b2" x="154" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b3" x="208" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b4" x="262" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b5" x="316" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b6" x="370" y="149" width="40" height="1" fill="#C99A12"/><rect id="vg-b7" x="424" y="149" width="40" height="1" fill="#C99A12"/></g><g font-size="8.5" text-anchor="middle" fill="#6B645E"><text id="vg-v0" x="66" y="146">·</text><text id="vg-v1" x="120" y="146">·</text><text id="vg-v2" x="174" y="146">·</text><text id="vg-v3" x="228" y="146">·</text><text id="vg-v4" x="282" y="146">·</text><text id="vg-v5" x="336" y="146">·</text><text id="vg-v6" x="390" y="146">·</text><text id="vg-v7" x="444" y="146">·</text></g><g font-size="9" text-anchor="middle" fill="#9A938A"><text class="lang-en" x="66" y="166">1 back</text><text class="lang-zh" x="66" y="166">往回 1 层</text><text x="120" y="166">2</text><text x="174" y="166">3</text><text x="228" y="166">4</text><text x="282" y="166">5</text><text x="336" y="166">6</text><text x="390" y="166">7</text><text x="444" y="166">8</text></g><text class="lang-en" x="260" y="188" text-anchor="middle" fill="#6B645E" font-size="9.5">how many layers the gradient has travelled back · 8 = the earliest layer</text><text class="lang-zh" x="260" y="188" text-anchor="middle" fill="#6B645E" font-size="9.5">gradient 往回走了几层 · 8 = 最早的那一层</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:6px">
<span>try:</span>
<button id="vg-sig" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">sigmoid, tidy weights → 0.25</button>
<button id="vg-relu" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">ReLU, tidy weights → 1.0</button>
<button id="vg-big" type="button" style="cursor:pointer;border:2px solid #E5DFD6;background:#FDF9F3;color:#6B645E;border-radius:5px;padding:2px 8px;font-family:monospace">ReLU, big weights → 1.6</button>
</div>
<label>each layer multiplies by <input id="vg-s" type="range" min="0.05" max="2" step="0.05" value="0.25" style="width:50%;accent-color:#C99A12;vertical-align:middle"></label>
<div id="vg-out" style="margin-top:6px;color:#2C2A28">multiply <b>0.25</b> per layer → after 8 layers the gradient is <b>×1.5e-5</b> <span style="color:#C93B3B">(vanished — the early layers get nothing)</span></div>
<div style="margin-top:4px;color:#6B645E;font-size:.92em">This one number is the activation's slope <b>times</b> the weight. Sigmoid alone caps its half at 0.25; a weight bigger than 1 can push the product above 1 — which is why the dial reaches past 1 at all.</div>
</div>
<script>(function(){
  var sl=document.getElementById('vg-s');if(!sl)return;
  var out=document.getElementById('vg-out'),BASE=150,H=110;
  var bars=[],vals=[];
  for(var i=0;i<8;i++){bars.push(document.getElementById('vg-b'+i));vals.push(document.getElementById('vg-v'+i));}
  function fmt(p){
    if(p>=1000||p<0.001)return '×'+p.toExponential(1);
    if(p>=10)return '×'+p.toFixed(1);
    return '×'+p.toFixed(p<0.1?4:3);
  }
  function hgt(p){
    var h=H*(Math.log(p)/Math.LN10+6)/8;
    return Math.max(1,Math.min(H,h));
  }
  function paint(){
    var s=+sl.value,p=1,col,word;
    for(var i=0;i<8;i++){
      p=p*s;
      var h=hgt(p);
      col=(p>3)?'#C93B3B':(p<0.05?'#C93B3B':(p<0.5?'#C99A12':'#2D8B55'));
      bars[i].setAttribute('y',(BASE-h).toFixed(1));
      bars[i].setAttribute('height',h.toFixed(1));
      bars[i].setAttribute('fill',col);
      vals[i].setAttribute('y',(BASE-h-4).toFixed(1));
      vals[i].textContent=fmt(p);
      vals[i].setAttribute('fill',col);
    }
    word=(p>3)?'<span style="color:#C93B3B">(exploded — one step overshoots the whole valley)</span>'
        :(p<0.05?'<span style="color:#C93B3B">(vanished — the early layers get nothing)</span>'
        :(p<0.5?'<span style="color:#C99A12">(fading — the early layers learn slowly)</span>'
        :'<span style="color:#2D8B55">(healthy — the signal survives all 8 layers)</span>'));
    out.innerHTML='multiply <b>'+s.toFixed(2)+'</b> per layer → after 8 layers the gradient is <b>'+fmt(p)+'</b> '+word;
  }
  sl.addEventListener('input',paint);
  document.getElementById('vg-sig').addEventListener('click',function(){sl.value=0.25;paint();});
  document.getElementById('vg-relu').addEventListener('click',function(){sl.value=1;paint();});
  document.getElementById('vg-big').addEventListener('click',function(){sl.value=1.6;paint();});
  paint();
})();</script>
%%%

Slide to `0.25` and each bar drops by the *same* height as the one before — one equal step shorter, every time. But those equal steps are a **log scale**, and that is the sneaky bit: each equal drop means *four times smaller*.

Do it eight times and you have `0.25⁸` — a `65,536×` cut, which is exactly why the printed number ends at about `×1.5e-5` (that notation just means 15 millionths) while the bars only look a few times shorter. Trust the numbers over the picture here — small multiplies **multiply toward 0** far faster than any height can show.

So far we have shown one setting of the dial. Try the other two and today's whole cast of traps fits on three rungs:

%%% steps
step: the fade — leave the dial at `0.25`
why: every bar is one equal height shorter than the last, and the printed numbers collapse toward zero
step: the flat line — slide to `1.0`
why: the bars sit level on the dashed line all the way back to the earliest layer, so the signal survives
step: the runaway — push past `1`
why: they climb instead of drop, and near the top of the dial they run clean off the picture — hold that thought, you'll meet that trap by name shortly
%%%

%%% demo id=vanish label="run it — the fade in five numbers"
predict: after 5 sigmoid layers, is the gradient about 0.1, about 0.01, or about 0.001? Commit to a guess, then reveal.
code: per_layer=0.25; [round(per_layer**n,4) for n in range(1,6)]  # gradient after 1..5 sigmoid layers
out: [0.25, 0.0625, 0.0156, 0.0039, 0.001]
take: <b>Small multiplies drive the gradient toward zero.</b> After 5 sigmoid layers the gradient is ×0.001 — the early layers barely move. This is why deep sigmoid networks were so hard to train for years.
%%%

It's our **hiker in fog** again, but the ground feels flatter and flatter the deeper the layer, until she genuinely cannot tell which way is down. And the main cause has a name: sigmoid is a **saturating** activation — out in its flat tails it barely reacts at all, so the **signal fades**.

%%% insight
This is the sneaky part, and it's worth a moment: the network does not crash. It trains happily, prints a loss, uses your GPU — while its first layers secretly learn almost nothing. For about two decades this quiet fade is what kept neural networks shallow. You are looking at the bug that held back a whole field.
%%%

#### The neat fix — a slope that refuses to shrink
So the cure writes itself: stop multiplying by tiny numbers. **Switch to ReLU**, whose slope for any positive input is a clean `1` (press the "ReLU, tidy weights" button above and watch the bars go level). The **ReLU slope stays 1** however many layers you stack, so the *activation* stops eating the signal — multiply by `1` a hundred times and the whisper is still exactly as loud.

That leaves only the other half of the per-layer multiply: the `× w` from Rule 4. Size the starting weights sensibly and that half also lands near `1`, so the product neither fades nor grows. Sizing the starting weights well is exactly what people mean by **careful initialization** — there are named recipes for it, and you'll meet them when you build your first multi-layer network.
~~~zh
滑到 `0.25`，每一根条形都比前一根矮*同样*的高度 —— 每次都矮掉相等的一小截。但那些相等的一小截是**对数刻度**，而狡猾的地方就在这里：每一次等高的下降都意味着*小四倍*。

做八次你就得到 `0.25⁸` —— 一个 `65,536×` 的削减。这正是为什么打印出来的数字最后停在大约 `×1.5e-5`（这个记法就是十五个百万分之一），而条形看起来只矮了几倍。在这里请相信数字，不要相信图 —— 小的乘数**一路乘向 0** 的速度，比任何高度画得出来的都快得多。

到目前为止我们只看了旋钮的一个位置。试试另外两个，今天的整套陷阱三级就装完了：

%%% steps
step: 变弱 —— 把旋钮留在 `0.25`
why: 每一根条形都比上一根矮相等的一截，而打印出来的数字一路塌向零
step: 平线 —— 滑到 `1.0`
why: 条形一路平平地贴在那条虚线上，直到最早的那一层，所以信号活了下来
step: 失控 —— 推过 `1`
why: 它们开始往上爬而不是往下掉，靠近旋钮顶端时干脆跑出画面外 —— 记住这一幕，你很快就会知道那个陷阱的名字
%%%

%%% demo id=vanishzh label="先猜，再展开"
predict: 过了 5 个 sigmoid 层之后，gradient 大约是 0.1、0.01，还是 0.001？先定一个猜测，再展开。
code: per_layer=0.25; [round(per_layer**n,4) for n in range(1,6)]  # 经过 1..5 个 sigmoid 层之后的 gradient
out: [0.25, 0.0625, 0.0156, 0.0039, 0.001]
take: <b>小的乘数把 gradient 一路推向零。</b>过了 5 个 sigmoid 层之后 gradient 是 ×0.001 —— 早期的层几乎不动。这就是为什么很深的 sigmoid 网络多年来那么难训练。
%%%

这又是我们那个**雾里的登山者**，只不过层越深，地面摸起来越平，直到她真的分不出哪边是下面。而主要原因有个名字：sigmoid 是一个 **saturating（饱和）**的 activation —— 在它平平的尾巴上它几乎没有反应，于是**信号变弱**。

%%% insight
狡猾的地方在这里，值得停一下：网络不会崩。它开心地训练、打印 loss、用你的 GPU —— 而它最前面的层偷偷地几乎什么都没学到。大约有二十年，就是这种安静的变弱让神经网络一直很浅。你正看着那个把一整个领域拖住的 bug。
%%%

#### 干净的修法 —— 一个拒绝缩小的斜率
所以解药自己就写出来了：别再乘很小的数字。**换成 ReLU**，它对任何正输入的斜率都是干净的 `1`（按一下上面那个「ReLU, tidy weights」按钮，看着条形变平）。不管你叠多少层，**ReLU 的斜率一直是 1**，所以 *activation* 不再吃掉信号 —— 乘 `1` 一百次，这句耳语还是一样响。

这样就只剩每层乘数的另一半了：规则 4 里那个 `× w`。把起始 weight 的大小定得合理，这一半也会落在 `1` 附近，于是乘积既不变弱也不变大。把起始 weight 的大小定好，正是人们说 **careful initialization（谨慎初始化）** 时的意思 —— 它有几个有名字的配方，等你搭第一个多层网络的时候就会见到。
~~~

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Before and after: with sigmoid the per-layer multiply is about 0.25 and the gradient fades to near zero across four layers; with ReLU and sensibly sized weights the per-layer multiply is about 1 and the gradient stays full height. Two rows of bars compared."><g font-family="monospace" font-size="10"><text class="lang-en" x="140" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">sigmoid → ×0.25 per layer → fades</text><text class="lang-zh" x="140" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">sigmoid → 每层 ×0.25 → 变弱</text><rect x="40" y="34" width="30" height="60" fill="#C99A12"/><rect x="86" y="79" width="30" height="15" fill="#C99A12"/><rect x="132" y="90" width="30" height="4" fill="#C93B3B"/><rect x="178" y="93" width="30" height="1" fill="#C93B3B"/><line x1="34" y1="94" x2="216" y2="94" stroke="#E5DFD6"/><text class="lang-en" x="120" y="112" text-anchor="middle" fill="#6B645E" font-size="9">early layers freeze 💀</text><text class="lang-zh" x="120" y="112" text-anchor="middle" fill="#6B645E" font-size="9">早期的层冻住 💀</text><line x1="258" y1="24" x2="258" y2="104" stroke="#E5DFD6"/><text class="lang-en" x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">ReLU + sane weights → ×≈1 → stays loud</text><text class="lang-zh" x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">ReLU + 合理的 weight → ×≈1 → 一直很响</text><rect x="300" y="34" width="30" height="60" fill="#2D8B55"/><rect x="346" y="34" width="30" height="60" fill="#2D8B55"/><rect x="392" y="34" width="30" height="60" fill="#2D8B55"/><rect x="438" y="34" width="30" height="60" fill="#2D8B55"/><line x1="294" y1="94" x2="476" y2="94" stroke="#E5DFD6"/><text class="lang-en" x="384" y="112" text-anchor="middle" fill="#1a5c38" font-size="9">signal survives every layer 🎉</text><text class="lang-zh" x="384" y="112" text-anchor="middle" fill="#1a5c38" font-size="9">信号活过每一层 🎉</text></g></svg>
%%%

**One puzzle solved, and it's the big one.** You just learned the cause (small multiplies stacked) *and* the cure (ReLU's slope of 1, plus sanely sized weights), which is more than most people can explain about deep learning. Let's cash that in on something you can use today.
~~~zh
**一个谜题解开了，而它是最大的那个。** 你刚学到了原因（小的乘数叠起来）*还有*解药（ReLU 那个 1 的斜率，加上大小合理的 weight），这比大多数人能讲清楚的深度学习都多。我们把这份收获换成一件今天就能用的东西。
~~~

@@@ concept id=c7 zh_tag="检查你的活" tag="Check your work" zh_title="gradient check —— 用两种办法量同一个斜率" title="The gradient check — measure it two ways" zh_gotit="懂了 gradient check" gotit="Got the gradient check"
Breather time, and this one is a gift you get to keep for the rest of your life in this field. You can now compute the downhill slope with the chain rule — but how do you know you didn't slip a sign or fat-finger a symbol? Think of **weighing flour two ways**: once by reading the recipe's math, and once by putting the bowl on a kitchen scale. If the two numbers agree, you trust your recipe. If they disagree, you know a step is wrong — *before* you bake a ruined cake. That is the whole trick, and it takes about thirty seconds.
~~~zh
喘口气，而这一个是你在这个领域里可以留一辈子的礼物。你现在会用 chain rule 算出下坡的斜率了 —— 但你怎么知道自己没有把正负号写错、没有手滑打错一个符号？想想**用两种办法称面粉**：一次是照配方里的算术算出来，一次是把碗放到厨房秤上。如果两个数字一致，你就信自己的配方。如果它们不一致，你就知道有一步错了 —— 而且是在你烤出一个废蛋糕*之前*。这就是全部的花招，而它大约只要三十秒。
~~~

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Two ways to get the same slope compared on a balance scale. On the left, backprop's computed gradient. On the right, a measured slope from nudging the weight up and down. When they match, the check passes; a mismatch flags a bug."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two ways to the same slope — do they match?</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">两条路通向同一个斜率 —— 它们对得上吗？</text><rect x="34" y="46" width="150" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="109" y="63" text-anchor="middle" fill="#5E5191">backprop's gradient</text><text class="lang-zh" x="109" y="63" text-anchor="middle" fill="#5E5191">backprop 算出的 gradient</text><text class="lang-en" x="109" y="78" text-anchor="middle" fill="#6B645E" font-size="9">(the chain rule)</text><text class="lang-zh" x="109" y="78" text-anchor="middle" fill="#6B645E" font-size="9">（用 chain rule）</text><rect x="336" y="46" width="150" height="40" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text class="lang-en" x="411" y="63" text-anchor="middle" fill="#1a5c38">measured slope</text><text class="lang-zh" x="411" y="63" text-anchor="middle" fill="#1a5c38">量出来的斜率</text><text class="lang-en" x="411" y="78" text-anchor="middle" fill="#6B645E" font-size="9">(nudge &amp; re-check the loss)</text><text class="lang-zh" x="411" y="78" text-anchor="middle" fill="#6B645E" font-size="9">（挪一下，再读一次 loss）</text><text x="260" y="60" text-anchor="middle" font-size="20">⚖️</text><text class="lang-en" x="260" y="80" text-anchor="middle" fill="#6B645E" font-size="10">≈ equal?</text><text class="lang-zh" x="260" y="80" text-anchor="middle" fill="#6B645E" font-size="10">≈ 相等吗？</text><text class="lang-en" x="150" y="122" fill="#2D8B55" font-size="10">match → trust your code ✓</text><text class="lang-zh" x="150" y="122" fill="#2D8B55" font-size="10">对得上 → 相信你的代码 ✓</text><text class="lang-en" x="360" y="122" fill="#C93B3B" font-size="10" text-anchor="end">mismatch → hunt the bug 🐛</text><text class="lang-zh" x="360" y="122" fill="#C93B3B" font-size="10" text-anchor="end">对不上 → 去抓那个 bug 🐛</text></g></svg>
%%%

**What the two-weighings picture gets right:** computing a number and *measuring* it independently is the surest way to catch a mistake — agreement builds trust, disagreement flags a bug. **Where it breaks down:** a kitchen scale is exact, but our measured slope is only an *estimate* (it uses a tiny nudge, not a truly zero-size step), so you check that the two are *close*, not identical to the last digit.

#### Put the weight on the scale
A slope is just "how much the output moves when you nudge the input." So let's **nudge and measure** it, with no calculus at all — this is our **hiker** stamping her boot to double-check the tilt she thinks she feels:

%%% steps
step: nudge up — add a tiny `ε` to one weight and read the loss
why: that's putting a pinch more flour in and re-weighing; one forward pass, nothing clever
step: nudge down — subtract the same `ε` and read the loss again
why: measuring both sides (a **finite difference**) cancels most of the error, therefore a sharper reading
step: divide — `(loss(w+ε) − loss(w−ε)) / (2ε)`
why: change in height ÷ distance moved — the plain definition of a slope
step: compare — does it match what backprop computed?
why: if yes, your backward pass is right. If no, you have a bug, and you found it in seconds
%%%

That four-rung recipe is the **numerical gradient check**. Two loss readings and a divide — that's it.

So far we have shown the recipe. Now watch it agree with backprop on a case where you already know the right answer:

%%% demo id=gcheck label="run it — two ways, one slope"
predict: for the toy loss `f(w) = w²` at `w = 3`, the true slope is `2w = 6`. Will the nudge-and-measure version land on 6.0, or drift noticeably off?
code: f=lambda w: w**2; w=3.0; eps=1e-5  # toy loss f(w)=w^2, whose true slope at w is 2w=6
code: measured=(f(w+eps)-f(w-eps))/(2*eps); backprop=2*w; print("measured",round(measured,4)," backprop",backprop)
out: measured 6.0  backprop 6.0
take: <b>They match → the gradient is right.</b> The measured slope (nudge up and down, divide by 2ε) is 6.0, and backprop says 6.0 too. When these disagree, you've got a bug in your backward pass — this cheap check catches it before you waste a training run.
%%%

%%% insight
Why this is a habit and not a chore: a wrong gradient does not throw an error. It trains, it prints numbers, it just quietly learns the wrong thing for six hours. Thirty seconds of gradient-checking on a toy example is the cheapest confidence you will ever buy in this field.
%%%

!!! c-warn ⚠️
<b>Use it to check, not to train.</b> Measuring the slope needs <i>two</i> full forward passes <i>per weight</i> — one for <code>w+ε</code> and one for <code>w−ε</code> — which is hopelessly slow for millions of weights. Backprop gets them all in one sweep. So you gradient-check on a tiny toy to prove the code is right, then let fast backprop do the real work.
!!!

**Victory lap: you can now *prove* your own gradients.** That is a genuinely professional habit, it fits on one page, and you picked it up in about five minutes — engineers at every lab run this exact check before trusting a new model.

Keep it in your pocket, because the next three beats are the classic ways learning goes wrong even when your math is perfect. With the check in hand you'll be able to tell "my code is buggy" apart from "my code is fine, my *setup* is bad" — and that distinction is most of debugging.

First up, the sneakiest one, and it hides in the very first line of any training script.
~~~zh
**称两次这个比喻对在哪里：** 把一个数字算出来，再独立地*量*一次，是抓错最靠得住的办法 —— 一致就建立信任，不一致就举旗报 bug。**它在哪里不成立：** 厨房秤是精确的，而我们量出来的斜率只是一个*估计*。它用的是一个很小的挪动，不是真正零大小的一步。所以你检查的是两者*接近*，而不是每一位数字都一样。

#### 把 weight 放上秤
一个斜率就是「你挪动输入时输出挪多少」。所以我们来**挪一下再量一下**，完全不用微积分 —— 这就是我们那位**登山者**用脚跺一下地面，再确认一次她以为自己摸到的那个倾斜：

%%% steps
step: 往上挪 —— 给一个 weight 加上一个很小的 `ε`，然后读 loss
why: 这就是多放一小撮面粉再称一次；一次 forward pass，没什么花巧
step: 往下挪 —— 减掉同样的 `ε`，再读一次 loss
why: 两边都量（这叫 **finite difference（有限差分）**）会把大部分误差抵消掉，因此读数更准
step: 相除 —— `(loss(w+ε) − loss(w−ε)) / (2ε)`
why: 高度的变化 ÷ 挪动的距离 —— 这就是斜率最朴素的定义
step: 比一比 —— 它和 backprop 算出来的对得上吗？
why: 对得上，你的 backward pass 就是对的。对不上，你就有个 bug，而你在几秒钟里找到了它
%%%

那个四级的配方就是 **numerical gradient check（数值梯度检查）**。读两次 loss，做一次除法 —— 就这样。

到目前为止我们只给出了配方。现在看它在一个你已经知道正确答案的例子上和 backprop 达成一致：

%%% demo id=gcheckzh label="先猜，再展开"
predict: 对玩具 loss `f(w) = w²`，在 `w = 3` 处真正的斜率是 `2w = 6`。挪一下再量一下的那个版本会落在 6.0 上，还是会明显跑偏？
code: f=lambda w: w**2; w=3.0; eps=1e-5  # 玩具 loss f(w)=w^2，它在 w 处的真斜率是 2w=6
code: measured=(f(w+eps)-f(w-eps))/(2*eps); backprop=2*w; print("measured",round(measured,4)," backprop",backprop)
out: measured 6.0  backprop 6.0
take: <b>它们一致 → 这个 gradient 是对的。</b>量出来的斜率（上下各挪一下，再除以 2ε）是 6.0，而 backprop 说的也是 6.0。当这两个不一致的时候，你的 backward pass 里就有 bug —— 这个便宜的检查在你浪费一次训练之前就抓住它。
%%%

%%% insight
为什么这是一个习惯而不是一件苦活：一个算错的 gradient 不会报错。它照样训练、照样打印数字，只是安安静静地学错的东西学六个小时。在一个玩具例子上花三十秒做 gradient check，是你在这个领域里能买到的最便宜的安心。
%%%

!!! c-warn ⚠️
<b>用它来检查，不要用它来训练。</b>量一个斜率<i>每个 weight</i> 都需要<i>两次</i>完整的 forward pass —— 一次给 <code>w+ε</code>，一次给 <code>w−ε</code> —— 对几百万个 weight 来说这慢得没救。backprop 一趟就全拿到。所以你在一个很小的玩具上做 gradient check 来证明代码是对的，然后让快的 backprop 去干真活。
!!!

**庆祝一下：你现在能*证明*自己的 gradient 了。** 这是一个真正专业的习惯，它一页就写得完，而你大约五分钟就学会了 —— 每个实验室的工程师在信任一个新模型之前，跑的都是这个检查。

把它揣在兜里，因为接下来三拍讲的是：就算你的数学完全正确，学习还是会出错的那几种经典方式。手里有这个检查，你就能把「我的代码有 bug」和「我的代码没问题，是我的*设置*不好」分开 —— 而这个区分就是调试的大半。

第一个上场的是最狡猾的那个，而它藏在任何训练脚本的第一行里。
~~~

@@@ concept id=c8 zh_tag="双胞胎陷阱" tag="The twin trap" zh_title="谜题 —— 每个神经元都变成同一个神经元" title="Puzzle — when every neuron becomes the same neuron" zh_gotit="懂了打破对称" gotit="Got symmetry breaking"
One quick puzzle, and it's the sneakiest of the day because nothing about it looks broken. Picture **identical twins** who copy each other's every move: same breakfast, same route to school, same answer on every quiz. Two people, one behavior. Now think about how you might start a fresh network. Setting every weight to the same tidy value — all `0`, say — feels neat and fair. It is also the twins.
~~~zh
一个很快的谜题，而它是今天最狡猾的那个，因为它哪里都不像坏了。想象一对**同卵双胞胎**，互相照抄对方的每一个动作：同样的早饭、同样的上学路线、每道小测都写同样的答案。两个人，一种行为。现在想想你会怎么开始一个全新的网络。把每一个 weight 都设成同一个整齐的值 —— 比如全都设 `0` —— 感觉又干净又公平。它也正是那对双胞胎。
~~~

%%% svg
<svg viewBox="0 0 520 152" role="img" aria-label="Identical twins in lockstep. Two children with the same clothes take the same step at the same moment, with a caption saying same start, same move, forever. Beside them a label says a whole layer of neurons started at the same value behaves like one neuron, and the cure is a small random start."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Identical twins: same start → same move → forever</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">同卵双胞胎：同样的起点 → 同样的动作 → 一直如此</text><text x="90" y="72" font-size="34">👯</text><rect x="150" y="42" width="150" height="34" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="225" y="63" text-anchor="middle" fill="#C93B3B">same step, always</text><text class="lang-zh" x="225" y="63" text-anchor="middle" fill="#C93B3B">总是同一步</text><g stroke="#C93B3B" stroke-width="2" fill="#C93B3B"><path d="M306 59 l38 0"/><polygon points="344,59 336,55 336,63"/></g><rect x="350" y="42" width="156" height="34" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="428" y="57" text-anchor="middle" fill="#C93B3B" font-size="9.5">a whole layer acting</text><text class="lang-zh" x="428" y="57" text-anchor="middle" fill="#C93B3B" font-size="9.5">一整层的表现</text><text class="lang-en" x="428" y="70" text-anchor="middle" fill="#C93B3B" font-size="9.5">like ONE neuron</text><text class="lang-zh" x="428" y="70" text-anchor="middle" fill="#C93B3B" font-size="9.5">像一个神经元</text><line x1="30" y1="96" x2="490" y2="96" stroke="#E5DFD6"/><text class="lang-en" x="260" y="116" text-anchor="middle" fill="#2D8B55">cure: give each neuron a small RANDOM starting value 🎲</text><text class="lang-zh" x="260" y="116" text-anchor="middle" fill="#2D8B55">解药：给每个神经元一个很小的随机起始值 🎲</text><text class="lang-en" x="260" y="136" text-anchor="middle" fill="#6B645E" font-size="9.5">a slightly different start is all it takes for them to grow into different jobs</text><text class="lang-zh" x="260" y="136" text-anchor="middle" fill="#6B645E" font-size="9.5">只要起点稍微不一样，它们就能长成不同的工作</text></g></svg>
%%%

**What the twins picture gets right:** the sameness is *self-sustaining* — identical starts plus identical experiences keep them identical, however long you wait. **Where it breaks down:** real twins eventually differ because life hands them different days; a network gets exactly the same inputs and the same rules for every clone, so nothing ever breaks the tie on its own.

#### Why the clones never drift apart
Follow the blame and you'll see the trap close:

%%% steps
step: the innocent choice — start every weight in a layer at the same value
why: this is **zero-initialization** (or any all-same start), and it feels tidy and fair
step: the collapse — every neuron computes the same thing, so backprop hands them **identical gradients**
why: same inputs, same local slopes, and the same outgoing wire carrying blame back — therefore the same share arrives at each unit, to the last digit
step: the trap — they update identically and **stay identical** forever
why: a hundred neurons behave like one; you paid for a layer and got a single unit
step: the fix — start at **small random** values to **break the symmetry**
why: the layer is no longer symmetric, so the units stop being interchangeable and each is free to grow into its own detector
%%%
~~~zh
**双胞胎这个比喻对在哪里：** 这种「一样」是*自己维持自己*的 —— 一样的起点加上一样的经历，会让他们一直一样，你等多久都一样。**它在哪里不成立：** 真的双胞胎最后会不一样，因为生活给他们不同的日子。而网络给每一个克隆体的输入和规则完全相同。所以没有任何东西会自己打破这个平局。

#### 这些克隆体为什么永远不会分开
跟着责任走一遍，你就会看到陷阱合上：

%%% steps
step: 那个看起来无害的选择 —— 把一层里每个 weight 都从同一个值开始
why: 这就是 **zero-initialization（零初始化）**，或者任何「全都一样」的起点，而它感觉又整齐又公平
step: 塌陷 —— 每个神经元算出同一个东西，于是 backprop 发给它们**完全相同的 gradient**
why: 一样的输入、一样的局部斜率、还有同一根往回带责任的出线 —— 因此到每个单元手里的份额一样，一位数字都不差
step: 陷阱 —— 它们更新得一模一样，然后**永远保持一样**
why: 一百个神经元表现得像一个；你付了一层的钱，拿到一个单元
step: 修法 —— 从**很小的随机**值开始，来**打破对称**
why: 这一层不再对称，于是这些单元不再可以互换，每一个都可以自由地长成自己的探测器
%%%
~~~

%%% svg
<svg viewBox="0 0 520 208" role="img" aria-label="Two panels comparing starting weights. Left panel: three tanh neurons whose incoming weights all start at 0.30 and whose wires out are also equal, so their local slopes match and backprop hands all three the identical gradient minus 0.45; they stay clones forever and the whole layer acts like one neuron. Right panel: three tanh neurons whose incoming weights start at three different small values, minus 0.10, 0.45 and 0.65, so their local slopes differ and the gradients coming back differ too, minus 0.59, minus 0.40 and minus 0.26; the three units grow into three different detectors. Every wire out is 0.5 in both panels, so the weight in is the only thing that changed."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">❌ all weights start at the SAME value</text><text class="lang-zh" x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">❌ 所有 weight 都从同一个值开始</text><text class="lang-en" x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">✅ small random start</text><text class="lang-zh" x="390" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">✅ 很小的随机起点</text><rect x="14" y="24" width="232" height="136" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="274" y="24" width="232" height="136" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><text class="lang-en" x="130" y="39" text-anchor="middle" fill="#5A544E" font-size="9">same weight in → same steepness → same blame</text><text class="lang-zh" x="130" y="39" text-anchor="middle" fill="#5A544E" font-size="9">进来的 weight 一样 → 陡度一样 → 责任一样</text><text class="lang-en" x="390" y="39" text-anchor="middle" fill="#5A544E" font-size="9">own weight in → own steepness → own blame</text><text class="lang-zh" x="390" y="39" text-anchor="middle" fill="#5A544E" font-size="9">各有自己的 weight → 各有自己的陡度 → 各有自己的责任</text><g stroke="#C93B3B" fill="#F7E4E4"><circle cx="60" cy="66" r="15"/><circle cx="60" cy="102" r="15"/><circle cx="60" cy="138" r="15"/></g><g fill="#C93B3B" text-anchor="middle" font-size="8.5"><text x="60" y="69">.30</text><text x="60" y="105">.30</text><text x="60" y="141">.30</text></g><g stroke="#C93B3B" stroke-width="1.8"><line x1="75" y1="66" x2="143" y2="66"/><line x1="75" y1="102" x2="143" y2="102"/><line x1="75" y1="138" x2="143" y2="138"/></g><g fill="#C93B3B" font-size="9.5"><text x="147" y="69">grad −0.45</text><text x="147" y="105">grad −0.45</text><text x="147" y="141">grad −0.45</text></g><text class="lang-en" x="130" y="156" text-anchor="middle" fill="#C93B3B" font-size="9.5">3 neurons behaving as 1</text><text class="lang-zh" x="130" y="156" text-anchor="middle" fill="#C93B3B" font-size="9.5">3 个神经元像 1 个在动</text><g stroke="#2D8B55" fill="#E6F2EA"><circle cx="320" cy="66" r="15"/><circle cx="320" cy="102" r="15"/><circle cx="320" cy="138" r="15"/></g><g fill="#2D8B55" text-anchor="middle" font-size="8.5"><text x="320" y="69">−.10</text><text x="320" y="105">.45</text><text x="320" y="141">.65</text></g><g stroke="#2D8B55" stroke-width="1.8"><line x1="335" y1="66" x2="403" y2="66"/><line x1="335" y1="102" x2="403" y2="102"/><line x1="335" y1="138" x2="403" y2="138"/></g><g fill="#2D8B55" font-size="9.5"><text x="407" y="69">grad −0.59</text><text x="407" y="105">grad −0.40</text><text x="407" y="141">grad −0.26</text></g><text class="lang-en" x="390" y="156" text-anchor="middle" fill="#2D8B55" font-size="9.5">3 neurons doing 3 things</text><text class="lang-zh" x="390" y="156" text-anchor="middle" fill="#2D8B55" font-size="9.5">3 个神经元做 3 件事</text><text class="lang-en" x="260" y="177" text-anchor="middle" fill="#5A544E" font-size="8.5">every wire out is .5 in both panels — the weight IN is the only thing that changed (tanh units)</text><text class="lang-zh" x="260" y="177" text-anchor="middle" fill="#5A544E" font-size="8.5">两边出去的线都是 .5 —— 只有进来的 weight 变了（tanh 单元）</text><text class="lang-en" x="260" y="191" text-anchor="middle" fill="#5A544E" font-size="9">symmetric all through — same weights in AND same wires out → same forever</text><text class="lang-zh" x="260" y="191" text-anchor="middle" fill="#5A544E" font-size="9">全程对称 —— 进来的 weight 一样、出去的线也一样 → 永远一样</text><text class="lang-en" x="260" y="204" text-anchor="middle" fill="#5A544E" font-size="9">randomness is not sloppiness — it is what lets a layer specialize</text><text class="lang-zh" x="260" y="204" text-anchor="middle" fill="#5A544E" font-size="9">随机不是马虎 —— 它正是让一层能分工的东西</text></g></svg>
%%%

So far we have shown the trap as a picture. Now let's settle it with arithmetic. We'll use **tanh** units here — the smooth S-curve from Day 2 — because a tanh unit's local slope changes as its `z` changes, so a hair of difference in the weight *going in* has somewhere to show up. Let's find out whether that is enough to break the tie.

%%% demo id=twins label="run it — does a hair of difference break the tie?"
predict: three hidden units start with the same weight `0.3` and see the same input. You nudge just ONE of those starting weights to `0.31`. Does that unit's share of the blame change — or do all three stay identical?
code: import math; x=1.5; t=1.0                                        # one input, one true answer
code: def hid(w):    return [math.tanh(wi*x) for wi in w]              # 3 hidden units, each bends its OWN weighted input
code: def guess(w,v): return sum(vi*hi for vi,hi in zip(v,hid(w)))     # the single output adds up what it hears
code: def blame(w,v): return [round(2*(guess(w,v)-t)*vi*(1-hi*hi)*x,4)+0.0 for vi,hi in zip(v,hid(w))]
code: #   ↑ one unit's share = 2·(guess−t) · its wire out v · tanh's slope at ITS OWN z · the input x
code: print("A  all three start the same  ", blame([0.30,0.30,0.30], [0.5,0.5,0.5]))
code: print("B  one weight nudged to 0.31 ", blame([0.31,0.30,0.30], [0.5,0.5,0.5]))
code: print("C  every weight its own value", blame([-0.10,0.45,0.65], [0.5,0.5,0.5]))
code: print("D  every weight starts at 0  ", blame([0.00,0.00,0.00], [0.0,0.0,0.0]))
out: A  all three start the same   [-0.4527, -0.4527, -0.4527]
out: B  one weight nudged to 0.31  [-0.4395, -0.4451, -0.4451]
out: C  every weight its own value [-0.5938, -0.3971, -0.2649]
out: D  every weight starts at 0   [0.0, 0.0, 0.0]
take: <b>Start a whole layer the same and it gets one shared answer back; give the units different weights and they stop being interchangeable.</b> Row A is the twins: identical weights bend the input identically, so all three units get the very same `-0.4527` and move as one forever. Row B breaks the tie with a hair — the nudged unit now sits at a different place on the tanh curve, where the curve has a different steepness, so its share is `-0.4395`, while the two you left alone still match each other at `-0.4451` (they still share a weight). Row C does the cure properly: give every weight its own small random value and all three shares differ. Row D is the tidiest-looking start and the worst of the four — every share is exactly `0`.
%%%

!!! c-info 🔬
<b>Optional (skippable) — what if the units use ReLU or sigmoid instead of tanh?</b> The one rule that holds for every activation is the trap itself: start a layer completely symmetric — same weights in <i>and</i> same wires out — and it stays symmetric forever. How fast the CURE bites is what varies.<br>
<b>Sigmoid behaves like tanh:</b> its steepness also changes with `z`, so a nudged weight gives that unit a different share on the very first step.<br>
<b>ReLU is slower:</b> its slope is a flat `1` for every positive `z`, so the nudge moves all three shares together — from `-0.4875` to `-0.4763` — changing them without making them differ.<br>
<b>What breaks it anyway:</b> a unit sitting at a positive `z` still <i>outputs</i> something different, so the wire leading out of it picks up its own gradient on the first step, and the layer stops being a set of clones.<br>
<b>Careful with "different is always enough":</b> it isn't. Shove a ReLU unit's `z` negative and it outputs `0` whatever its weight is — which is exactly the dead unit you meet next.
!!!

#### One extra beat — why the all-zero start is the worst of all
Row D deserves its own sentence. Rule 4 said the blame reaching a hidden unit travels through the wire leading *out* of it. Start every weight at `0` and every one of those wires is `0` too, so on the first step every weight in the hidden layer gets a gradient of exactly `0`. For a `tanh` or `ReLU` unit — whose output is `0` when `z` is `0` — it stays stuck there: the hidden layer never learns anything at all.

The only thing left that can still move is the output's own bias (Rule 3), so the network quietly settles on predicting one constant number for every input. And note which trap this is: it is the twins in their most extreme form, so the cure is a **small random start** — not Leaky ReLU. Nothing here is a dead unit; the whole layer is simply mute. A small random start fixes both sides at once: each unit gets its own weight in *and* its own wire out.

%%% insight
Why this one bites hardest: a same-start network trains without a single error message. The loss even goes down a little — one neuron's worth (and from an all-zero start with `tanh` or `ReLU` units the hidden layer learns nothing at all, yet the loss usually *still* creeps down, because that one output bias is drifting toward the average answer). You would stare at that code for hours. Knowing this trap by name is worth an entire afternoon of debugging, today.
%%%

**One line of code, one whole class of bug avoided.** Every framework already hands you small random starting weights by default, and now you know *why* that default exists — you'd have invented it yourself. Two more cousins to meet, and they are the loud ones.
~~~zh
到目前为止我们把陷阱画成了图。现在用算术把它敲定。这里我们用 **tanh** 单元 —— 第 2 天那条平滑的 S 曲线 —— 因为 tanh 单元的局部斜率会随它的 `z` 变化，所以*进来*的 weight 上一根头发的差别，有地方可以显现出来。我们来看看这够不够打破那个平局。

%%% demo id=twinszh label="先猜，再展开"
predict: 三个隐藏单元都从同一个 weight `0.3` 开始，看到同一个输入。你只把其中**一个**起始 weight 挪到 `0.31`。那个单元该担的责任会变吗 —— 还是三个还是一模一样？
code: import math; x=1.5; t=1.0                                        # 一个输入，一个真答案
code: def hid(w):    return [math.tanh(wi*x) for wi in w]              # 3 个隐藏单元，每个弯自己那份加权输入
code: def guess(w,v): return sum(vi*hi for vi,hi in zip(v,hid(w)))     # 单个输出把它听到的加起来
code: def blame(w,v): return [round(2*(guess(w,v)-t)*vi*(1-hi*hi)*x,4)+0.0 for vi,hi in zip(v,hid(w))]
code: #   ↑ 一个单元的份额 = 2·(guess−t) · 它的出线 v · tanh 在它自己 z 处的斜率 · 输入 x
code: print("A  all three start the same  ", blame([0.30,0.30,0.30], [0.5,0.5,0.5]))
code: print("B  one weight nudged to 0.31 ", blame([0.31,0.30,0.30], [0.5,0.5,0.5]))
code: print("C  every weight its own value", blame([-0.10,0.45,0.65], [0.5,0.5,0.5]))
code: print("D  every weight starts at 0  ", blame([0.00,0.00,0.00], [0.0,0.0,0.0]))
out: A  all three start the same   [-0.4527, -0.4527, -0.4527]
out: B  one weight nudged to 0.31  [-0.4395, -0.4451, -0.4451]
out: C  every weight its own value [-0.5938, -0.3971, -0.2649]
out: D  every weight starts at 0   [0.0, 0.0, 0.0]
take: <b>让一整层从同一个值开始，它拿回来的就是一个共用的答案；给这些单元不同的 weight，它们就不再可以互换。</b>A 行就是那对双胞胎：一样的 weight 把输入弯得一样，所以三个单元拿到的都是同一个 `-0.4527`，然后永远一起动。B 行用一根头发打破了平局。被挪过的那个单元现在坐在 tanh 曲线上另一个位置，那里曲线的陡度不一样。所以它的份额是 `-0.4395`。而你没动的那两个还是彼此一样，都是 `-0.4451`，因为它们还共用一个 weight。C 行才是把解药用对了：给每个 weight 自己那个很小的随机值，三个份额就都不一样。D 行是看起来最整齐的起点，也是四行里最糟的 —— 每个份额都正好是 `0`。
%%%

!!! c-info 🔬
<b>可跳过的选修 —— 如果这些单元用的是 ReLU 或者 sigmoid，而不是 tanh？</b>对每一个 activation 都成立的那一条规则，就是这个陷阱本身：让一层完全对称 —— 进来的 weight 一样<i>而且</i>出去的线也一样 —— 它就会永远保持对称。变的是那个**解药**多快起作用。<br>
<b>Sigmoid 和 tanh 一样：</b>它的陡度也随 `z` 变化，所以一个被挪过的 weight 会在第一步就给那个单元一个不同的份额。<br>
<b>ReLU 慢一些：</b>它对每一个正的 `z` 斜率都是平的 `1`，所以那一挪会让三个份额一起动 —— 从 `-0.4875` 到 `-0.4763` —— 变了，但没有变得彼此不同。<br>
<b>那是什么最后打破了它：</b>一个坐在正 `z` 上的单元毕竟<i>输出</i>了不一样的东西。所以从它出去的那根线在第一步就拿到自己的 gradient。于是这一层不再是一堆克隆体。<br>
<b>小心「不一样就一定够」这句话：</b>它不对。把一个 ReLU 单元的 `z` 推成负的，它不管 weight 是多少都输出 `0` —— 而那正是你接下来会见到的那个死掉的单元。
!!!

#### 多讲一拍 —— 为什么全零起点是最糟的
D 行值得单独一句话。规则 4 说过，到达一个隐藏单元的责任是从它*出去*的那根线上走过来的。让每个 weight 都从 `0` 开始，那些线也全都是 `0`，所以第一步的时候隐藏层里每个 weight 拿到的 gradient 正好是 `0`。对一个 `tanh` 或 `ReLU` 单元来说 —— 它们在 `z` 是 `0` 时输出也是 `0` —— 它就卡在那里了：隐藏层什么都学不到。

剩下唯一还能动的，是输出自己的 bias（规则 3），所以网络就悄悄地安顿成：对任何输入都预测同一个常数。而请注意这是哪一个陷阱：它是双胞胎最极端的形态，所以解药是**很小的随机起点** —— 不是 Leaky ReLU。这里没有任何死掉的单元；整层只是哑了。一个很小的随机起点一次修好两边：每个单元既有自己进来的 weight，也有自己出去的线。

%%% insight
这一个为什么咬得最狠：一个「同一起点」的网络训练起来一条错误信息都不报。loss 甚至还会降一点 —— 降的是一个神经元那么多。而在 `tanh` 或 `ReLU` 单元的全零起点下，隐藏层什么都学不到。可 loss 通常*还是*会慢慢往下爬，因为那一个输出 bias 正朝着平均答案漂移。你会盯着那段代码看好几个小时。今天知道这个陷阱的名字，值一整个下午的调试时间。
%%%

**一行代码，避开一整类 bug。** 每一个框架默认就已经给你很小的随机起始 weight 了，而你现在知道那个默认*为什么*存在 —— 你自己也会把它发明出来。还有两个表亲要见，而它们是吵闹的那种。
~~~

@@@ concept id=c9 zh_tag="卡住与尖叫" tag="Stuck &amp; squealing" zh_title="谜题 —— 卡住的开关和话筒的尖叫" title="Puzzle — the stuck switch and the mic squeal" zh_gotit="懂了这两个解药" gotit="Got the two cures"
Two last puzzles, and this time you get to break things on purpose. Picture two everyday annoyances. First, a **light switch jammed OFF** — you flick it and nothing happens, because the thing that would move it is the very thing that's broken. Second, a **microphone held too close to its own speaker**: the sound feeds itself, louder and louder, until the whole room covers its ears. One trap goes silent, the other goes deafening — and both are the same per-layer multiply, just at the two extremes: `×0` and `×big`.
~~~zh
最后两个谜题，而这一次你可以故意把东西弄坏。想象两件日常的烦心事。第一件，一个**卡在关的位置上的电灯开关** —— 你去拨它，什么都没发生，因为能让它动的那个东西正是坏掉的那个东西。第二件，一个**离自己的音箱太近的话筒**：声音自己喂自己，越来越响，直到整屋子的人都捂上耳朵。一个陷阱变得没声音，另一个变得吵到受不了 —— 而两个都是同一个每层乘数，只是站在两个极端上：`×0` 和 `×很大`。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Two everyday things drawn side by side. On the left a wall light switch jammed in the off position with a hand flicking it and nothing happening, labelled goes silent, multiply by zero. On the right a microphone pointed at a loudspeaker with sound waves feeding back and growing, labelled goes deafening, multiply by something big."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two loud cousins: one goes silent, one goes deafening</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">两个吵闹的表亲：一个变没声，一个变震耳</text><rect x="18" y="26" width="228" height="126" rx="8" fill="#FDFCF9" stroke="#E5DFD6"/><rect x="88" y="44" width="46" height="70" rx="6" fill="#EDEAE4" stroke="#9A938A"/><rect x="96" y="84" width="30" height="24" rx="3" fill="#C93B3B"/><text class="lang-en" x="111" y="130" text-anchor="middle" fill="#C93B3B">OFF · jammed</text><text class="lang-zh" x="111" y="130" text-anchor="middle" fill="#C93B3B">关 · 卡住了</text><text x="152" y="76" font-size="20">👆</text><text class="lang-en" x="176" y="96" text-anchor="middle" fill="#6B645E" font-size="9">flick…</text><text class="lang-zh" x="176" y="96" text-anchor="middle" fill="#6B645E" font-size="9">拨一下…</text><text class="lang-en" x="176" y="108" text-anchor="middle" fill="#6B645E" font-size="9">nothing</text><text class="lang-zh" x="176" y="108" text-anchor="middle" fill="#6B645E" font-size="9">没反应</text><text class="lang-en" x="132" y="146" text-anchor="middle" fill="#C93B3B" font-size="10.5">goes SILENT · ×0</text><text class="lang-zh" x="132" y="146" text-anchor="middle" fill="#C93B3B" font-size="10.5">变没声 · ×0</text><rect x="274" y="26" width="228" height="126" rx="8" fill="#FDFCF9" stroke="#E5DFD6"/><text x="308" y="84" font-size="24">🎤</text><rect x="428" y="56" width="40" height="52" rx="4" fill="#EDEAE4" stroke="#9A938A"/><circle cx="448" cy="82" r="12" fill="none" stroke="#9A938A"/><g fill="none" stroke="#C93B3B" stroke-width="1.6"><path d="M350 78 q10 -10 0 -20"/><path d="M362 84 q18 -16 0 -34"/><path d="M374 90 q26 -22 0 -48"/></g><g fill="none" stroke="#C93B3B" stroke-width="1.6"><path d="M418 82 q-10 8 0 16"/><path d="M406 82 q-16 12 0 26"/></g><text class="lang-en" x="388" y="130" text-anchor="middle" fill="#C93B3B">sound feeds itself → squeal</text><text class="lang-zh" x="388" y="130" text-anchor="middle" fill="#C93B3B">声音自己喂自己 → 尖叫</text><text class="lang-en" x="388" y="146" text-anchor="middle" fill="#C93B3B" font-size="10.5">goes DEAFENING · ×big</text><text class="lang-zh" x="388" y="146" text-anchor="middle" fill="#C93B3B" font-size="10.5">变震耳 · ×很大</text></g></svg>
%%%

**What the two pictures get right:** each is *self-sustaining* — the jammed switch has nothing left to move it, and the squeal is its own cause. **Where they break down:** a real switch can be fixed with a screwdriver from outside, but a network only ever has its own gradients to work with, so a unit at `×0` truly cannot rescue itself. That is exactly why the cures change the *rules of the road* rather than the unit.

#### The stuck switch — a **dead ReLU**
ReLU's **slope 0** for negatives is what makes this one possible. Follow the jam:

%%% steps
step: the shove — one oversized update pushes a unit's `z` **permanently negative**
why: ReLU outputs 0 there, so the unit says nothing at all
step: the jam — its slope is 0 too, therefore its gradient is 0
why: zero gradient means zero update, so `z` stays negative — the switch cannot flip itself back
step: the verdict — this unit **never learns** again; it is a **dead ReLU**
why: nothing in normal training can rescue it, because a rescue would need a gradient
step: the fix — **Leaky ReLU** keeps a **small negative slope** (about `0.01`, gently rising, instead of flat `0`)
why: a trickle of gradient always gets through, so a shoved unit can climb back out
%%%

Now go break both of them yourself. The left dial shoves a unit's `z` around (tick **Leaky** to change the rules of the road); the right dial cranks the weight size that every layer multiplies by (tick **clip** to put a ceiling on it). **Predict first:** on the left, what gradient does the unit get once `z` goes negative? On the right, what does a weight of `3` do to the gradient after six layers?
~~~zh
**这两幅图对在哪里：** 每一个都*自己维持自己* —— 卡住的开关已经没有东西能让它动了，而那声尖叫就是它自己的原因。**它们在哪里不成立：** 真的开关可以从外面用螺丝刀修好，但网络手里只有它自己的 gradient 可用，所以一个停在 `×0` 上的单元真的救不了自己。这正是为什么这些解药改的是*路上的规则*，而不是那个单元本身。

#### 卡住的开关 —— 一个 **dead ReLU（死掉的 ReLU）**
ReLU 对负数的**斜率 0**，正是让这件事可能发生的东西。跟着这个「卡住」走一遍：

%%% steps
step: 那一推 —— 一次过大的 update 把某个单元的 `z` 推成**永久为负**
why: ReLU 在那里输出 0，所以这个单元什么都不说了
step: 卡住 —— 它的斜率也是 0，因此它的 gradient 是 0
why: gradient 为零就意味着 update 为零，所以 `z` 一直是负的 —— 这个开关没法自己拨回去
step: 判决 —— 这个单元再也**学不到东西**了；它是一个 **dead ReLU**
why: 正常训练里没有任何东西能救它，因为救它需要一个 gradient
step: 修法 —— **Leaky ReLU** 保留一个**很小的负侧斜率**（大约 `0.01`，缓缓上升，而不是平平的 `0`）
why: 总有一丝 gradient 能漏过去，所以一个被推下去的单元能爬回来
%%%

现在你自己去把这两个都弄坏。左边的旋钮把一个单元的 `z` 推来推去（勾上 **Leaky** 就改变路上的规则）；右边的旋钮把每一层要乘的 weight 大小往上拉（勾上 **clip** 就给它加一个上限）。**先预测：** 左边，一旦 `z` 变成负的，这个单元拿到的 gradient 是多少？右边，一个 `3` 的 weight 在六层之后会把 gradient 变成什么样？
~~~

%%% svg
<svg id="tc-svg" viewBox="0 0 520 236" role="img" aria-label="Two interactive panels. The left panel is a unit dashboard: a light switch drawing shows on or off, and readouts show the unit's z, its output, its slope, and a bar for the gradient reaching its weight. When z is negative with plain ReLU the bar is empty and the verdict reads dead. Ticking Leaky ReLU leaves a small non-zero bar and the verdict reads it can recover. The right panel shows six bars for the gradient after travelling back one to six layers as the weight size changes. Small weights keep the bars low, a weight of three makes every bar taller than the last and the verdict reads exploded. Ticking clip holds any bar that reaches the ceiling at the ceiling line, leaving the smaller earlier bars alone, so the held bars end up level with each other, and the verdict then reports the size the gradient would have reached and the smaller size clipping passed on instead."><g font-family="monospace" font-size="10"><text class="lang-en" x="132" y="20" text-anchor="middle" fill="#2C2A28" font-size="11.5">🔌 the stuck switch</text><text class="lang-zh" x="132" y="20" text-anchor="middle" fill="#2C2A28" font-size="11.5">🔌 卡住的开关</text><text class="lang-en" x="388" y="20" text-anchor="middle" fill="#2C2A28" font-size="11.5">🎤 the squeal</text><text class="lang-zh" x="388" y="20" text-anchor="middle" fill="#2C2A28" font-size="11.5">🎤 尖叫</text><rect x="14" y="28" width="236" height="178" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="270" y="28" width="236" height="178" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><rect x="34" y="44" width="40" height="60" rx="6" fill="#EDEAE4" stroke="#9A938A"/><rect id="tc-knob" x="40" y="74" width="28" height="24" rx="3" fill="#C93B3B"/><text class="lang-en" id="tc-state" x="54" y="118" text-anchor="middle" fill="#C93B3B" font-size="9">OFF</text><text class="lang-zh" id="tc-statezh" x="54" y="118" text-anchor="middle" fill="#C93B3B" font-size="9">关</text><text id="tc-z" x="92" y="58" fill="#5A544E">z = -1.0</text><text id="tc-out" x="92" y="76" fill="#5A544E">output = 0.00</text><text id="tc-slope" x="92" y="94" fill="#5A544E">slope = 0.00</text><text class="lang-en" x="34" y="140" fill="#6B645E" font-size="9">gradient reaching its weight (bar stretched, so a sliver still shows)</text><text class="lang-zh" x="34" y="140" fill="#6B645E" font-size="9">到达它 weight 的 gradient（条形被拉长过，所以一丝也看得见）</text><rect x="34" y="146" width="196" height="14" rx="3" fill="#EFE9DF"/><rect id="tc-bar" x="34" y="146" width="2" height="14" rx="3" fill="#C93B3B"/><text class="lang-en" id="tc-verdict" x="132" y="180" text-anchor="middle" fill="#C93B3B" font-size="10">dead — zero gradient, forever</text><text class="lang-zh" id="tc-verdictzh" x="132" y="180" text-anchor="middle" fill="#C93B3B" font-size="10">死了 —— gradient 为零，永远如此</text><text class="lang-en" id="tc-verdict2" x="132" y="196" text-anchor="middle" fill="#6B645E" font-size="9">a rescue would need a gradient it cannot get</text><text class="lang-zh" id="tc-verdict2zh" x="132" y="196" text-anchor="middle" fill="#6B645E" font-size="9">救它需要一个它拿不到的 gradient</text><line x1="284" y1="176" x2="496" y2="176" stroke="#E5DFD6" stroke-width="1.5"/><g id="tc-bars"><rect id="tc-e0" x="292" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e1" x="326" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e2" x="360" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e3" x="394" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e4" x="428" y="174" width="26" height="2" fill="#2D8B55"/><rect id="tc-e5" x="462" y="174" width="26" height="2" fill="#2D8B55"/></g><line id="tc-clipline" x1="284" y1="130.7" x2="496" y2="130.7" stroke="#1a5c38" stroke-width="1.6" stroke-dasharray="4,3" stroke-opacity="0"/><text class="lang-en" id="tc-cliplab" x="496" y="126.7" text-anchor="end" fill="#1a5c38" font-size="8.5" opacity="0">clip ceiling</text><text class="lang-zh" id="tc-cliplabzh" x="496" y="126.7" text-anchor="end" fill="#1a5c38" font-size="8.5" opacity="0">clip 的上限</text><g font-size="7.5" text-anchor="middle" fill="#6B645E" paint-order="stroke" stroke="#FDF9F3" stroke-width="1.5" stroke-linejoin="round"><text id="tc-ev0" x="305" y="170">·</text><text id="tc-ev1" x="339" y="170">·</text><text id="tc-ev2" x="373" y="170">·</text><text id="tc-ev3" x="407" y="170">·</text><text id="tc-ev4" x="441" y="170">·</text><text id="tc-ev5" x="475" y="170">·</text></g><text class="lang-en" x="388" y="188" text-anchor="middle" fill="#9A938A" font-size="8.5">layers travelled back: 1 · 2 · 3 · 4 · 5 · 6</text><text class="lang-zh" x="388" y="188" text-anchor="middle" fill="#9A938A" font-size="8.5">往回走了几层：1 · 2 · 3 · 4 · 5 · 6</text><text class="lang-en" id="tc-everdict" x="388" y="202" text-anchor="middle" fill="#2D8B55" font-size="10">healthy — the signal survives</text><text class="lang-zh" id="tc-everdictzh" x="388" y="202" text-anchor="middle" fill="#2D8B55" font-size="10">健康 —— 信号活了下来</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">
<label>shove z <input id="tc-zs" type="range" min="-4" max="4" step="0.2" value="-1" style="width:150px;accent-color:#C99A12;vertical-align:middle"></label>
<label><input id="tc-leaky" type="checkbox"> use <b>Leaky</b> ReLU</label>
</div>
<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:6px">
<label>weight size <input id="tc-ws" type="range" min="0.5" max="3" step="0.1" value="1" style="width:150px;accent-color:#C99A12;vertical-align:middle"></label>
<label><input id="tc-clip" type="checkbox"> <b>clip</b> the gradient at 5</label>
</div>
<div style="margin-top:6px;color:#6B645E;font-size:.92em">Left: drag `z` below 0 with plain ReLU and the gradient bar empties — that unit is finished. Tick Leaky and the same unit keeps a sliver. Right: the bars are the gradient after 1…6 layers when each layer multiplies by the weight size; the printed numbers are the true values. With clip ticked, only the bars that actually reach the ceiling are held there — the earlier, smaller ones are left alone — and the numbers then show the clipped size the optimizer would really use, not the raw one.</div>
</div>
<script>(function(){
  var zs=document.getElementById('tc-zs');if(!zs)return;
  var leaky=document.getElementById('tc-leaky'),ws=document.getElementById('tc-ws'),clip=document.getElementById('tc-clip'),
      knob=document.getElementById('tc-knob'),state=document.getElementById('tc-state'),
      tz=document.getElementById('tc-z'),tout=document.getElementById('tc-out'),tsl=document.getElementById('tc-slope'),
      bar=document.getElementById('tc-bar'),ver=document.getElementById('tc-verdict'),ver2=document.getElementById('tc-verdict2'),
      cline=document.getElementById('tc-clipline'),clab=document.getElementById('tc-cliplab'),ever=document.getElementById('tc-everdict');
  // The bilingual twin of every JS-written label needs its OWN id. Two nodes sharing
  // one id means getElementById only ever finds the English one, so a Chinese reader
  // watches a FROZEN verdict while the prose tells them to watch it change.
  var statez=document.getElementById('tc-statezh'),verz=document.getElementById('tc-verdictzh'),
      ver2z=document.getElementById('tc-verdict2zh'),clabz=document.getElementById('tc-cliplabzh'),
      everz=document.getElementById('tc-everdictzh');
  function put(en,zh,tEn,tZh,col){                 // write one readout in BOTH languages
    if(en){en.textContent=tEn;if(col)en.setAttribute('fill',col);}
    if(zh){zh.textContent=tZh;if(col)zh.setAttribute('fill',col);}
  }
  var eb=[],ev=[];
  for(var i=0;i<6;i++){eb.push(document.getElementById('tc-e'+i));ev.push(document.getElementById('tc-ev'+i));}
  var LEAK=0.01, INCOMING=1.0, BASE=176, H=120, CLIP=5;
  function hgt(p){var h=H*(Math.log(p)/Math.LN10+1)/4.5;return Math.max(2,Math.min(H,h));}
  function paintUnit(){
    var z=+zs.value, on=(z>=0), lk=leaky.checked;   /* c5 shows slope 1 at z=0 — match it */
    var out=on?z:(lk?LEAK*z:0), slope=on?1:(lk?LEAK:0), grad=INCOMING*slope;
    knob.setAttribute('y',on?'50':'74');knob.setAttribute('fill',on?'#2D8B55':'#C93B3B');
    put(state,statez,on?'ON':'OFF',on?'开':'关',on?'#2D8B55':'#C93B3B');
    tz.textContent='z = '+z.toFixed(1);
    tout.textContent='output = '+out.toFixed(3);
    tsl.textContent='slope = '+slope.toFixed(2);
    var w=(grad<=0)?0:Math.max(8,Math.sqrt(grad)*196);
    bar.setAttribute('width',w.toFixed(1));
    if(on){bar.setAttribute('fill','#2D8B55');
      put(ver,verz,'alive — full gradient gets through','活着 —— 完整的 gradient 都过去了','#2D8B55');
      put(ver2,ver2z,'this unit is learning normally','这个单元在正常学习');}
    else if(lk){bar.setAttribute('fill','#C99A12');
      put(ver,verz,'shoved, but NOT dead — gradient '+grad.toFixed(2),'被推下去了，但没死 —— gradient '+grad.toFixed(2),'#C99A12');
      put(ver2,ver2z,'the trickle can push z back above 0 · that is Leaky ReLU','这一丝能把 z 推回 0 以上 · 这就是 Leaky ReLU');}
    else {bar.setAttribute('fill','#C93B3B');
      put(ver,verz,'dead — zero gradient, forever','死了 —— gradient 为零，永远如此','#C93B3B');
      put(ver2,ver2z,'a rescue would need a gradient it cannot get','救它需要一个它拿不到的 gradient');}
  }
  function paintChain(){
    var w=+ws.value, doClip=clip.checked, p=1, raw=1, last=1, didClip=false;
    cline.setAttribute('stroke-opacity',doClip?'1':'0');clab.setAttribute('opacity',doClip?'1':'0');
    if(clabz)clabz.setAttribute('opacity',doClip?'1':'0');   // the Chinese ceiling label hides/shows too
    for(var i=0;i<6;i++){
      p=p*w; raw=raw*w;
      if(doClip&&p>CLIP){p=CLIP;didClip=true;}
      last=p;
      // Colour by the SAME bands as the verdict below and as the c6 ladder, on the value the
      // bar actually shows. No special case for a clipped bar: clipping bounds the step, it
      // does not turn a big gradient into a healthy one, so a bar pinned at the ceiling stays
      // in the band its number falls in. Otherwise a clipped bar reads green while the
      // verdict quoting that same number reads red.
      var h=hgt(p),col=(p>3||p<0.05)?'#C93B3B':(p<0.5?'#C99A12':'#2D8B55');
      eb[i].setAttribute('y',(BASE-h).toFixed(1));eb[i].setAttribute('height',h.toFixed(1));
      eb[i].setAttribute('fill',col);
      // With the ceiling showing, a tall bar's value label would land on the same line as
      // the "clip ceiling" text (both at y ~ 127), so print it INSIDE the bar. Use dark ink,
      // not white: white on the amber fill (#C99A12) is only ~2.6:1 at this size. The group
      // carries a light halo (paint-order stroke) so dark ink stays legible on green, amber
      // and red alike.
      var crowded=doClip&&(BASE-h-3)<135;
      ev[i].setAttribute('y',(crowded?(BASE-h+10):(BASE-h-3)).toFixed(1));
      ev[i].textContent=(p>=100?p.toExponential(1):p.toFixed(p<0.1?3:(p<10?2:1)));   /* 3dp under 0.1 so 0.046656 prints 0.047, not a "0.05" that straddles the band edge */
      ev[i].setAttribute('fill',crowded?'#1F1B16':col);
    }
    // Bands are the c6 ladder exactly (>3 exploded, <0.05 vanished, <0.5 fading), so the same
    // product cannot read "exploded" in one widget and "healthy" in the other.
    // Clipping is NOT its own band. The ceiling (5) sits above the exploded line (3), so a
    // "clipped, all sane now" verdict would contradict the red bars it is sitting under.
    // Clipping bounds the STEP; it does not make the gradient healthy. So report the raw
    // product that WOULD have arrived, and say what clipping did to it.
    if(didClip){
      put(ever,everz,'would have hit '+(raw>=100?raw.toExponential(1):raw.toFixed(1))
        +' → clipped to '+CLIP+' before the step',
        '本来会冲到 '+(raw>=100?raw.toExponential(1):raw.toFixed(1))
        +' → 迈步之前被裁到 '+CLIP,'#C93B3B');}
    else if(last>3){
      put(ever,everz,'exploded → '+(last>=100?last.toExponential(1):last.toFixed(1))+' · one step overshoots the valley',
        '爆炸了 → '+(last>=100?last.toExponential(1):last.toFixed(1))+' · 一步就迈过整个山谷','#C93B3B');}
    else if(last<0.05){
      put(ever,everz,'vanished → '+last.toExponential(1)+' · the early layers get nothing',
        '消失了 → '+last.toExponential(1)+' · 早期的层什么都拿不到','#C93B3B');}
    else if(last<0.5){
      put(ever,everz,'fading → '+last.toFixed(2)+' · the early layers learn slowly',
        '在变弱 → '+last.toFixed(2)+' · 早期的层学得很慢','#C99A12');}
    else {put(ever,everz,'healthy — the signal survives','健康 —— 信号活了下来','#2D8B55');}
  }
  function paint(){paintUnit();paintChain();}
  zs.addEventListener('input',paint);ws.addEventListener('input',paint);
  leaky.addEventListener('change',paint);clip.addEventListener('change',paint);
  paint();
})();</script>
%%%

**What you just made happen:** on the left, a negative `z` with plain ReLU empties the bar to exactly `0` — and once it's `0`, nothing can refill it. Tick **Leaky** and the same shoved unit keeps a sliver of gradient, enough to climb back. **Sensible init** helps too: start the weights at reasonable sizes and units don't get shoved off the cliff on day one.

#### The squeal — the **exploding gradient**
Now the right-hand dial. Crank the weight size up and the running product from Rule 4 stops shrinking and starts snowballing:

%%% steps
step: the squeal — the numbers each layer multiplies by are big instead of tiny, so the running product grows
why: same chain of multiplies as the fade, just pointed the other way
step: the blow-up — a few layers of that and you get **huge values**: this is the **exploding gradient**
why: the gradient is still "correct" — it is just enormous, and one giant step **overshoots** the whole valley, so the loss often comes back as `inf` or `NaN`
step: the fix — **gradient clipping**: set a ceiling and **clip** anything above it before you step
why: tick the clip box and every bar that reaches the ceiling stops right there — one monster number can never launch you off the hill again
%%%

%%% demo id=explode label="run it — the squeal in six numbers"
predict: multiply the gradient by 3 at each of six layers. Does it reach roughly 18, roughly 100, or over 700?
code: g=1.0; big=3.0; [round(g:=g*big,1) for _ in range(6)]  # gradient multiplied by a big amount each layer
out: [3.0, 9.0, 27.0, 81.0, 243.0, 729.0]
take: <b>Big multiplies drive the gradient toward infinity.</b> ×3 per layer and the gradient rockets from 3 to 729 in six steps — a step that size overshoots wildly, and the loss often prints as inf or NaN. Clip it, or size the starting weights so the per-layer multiply sits near 1.
%%%

And **careful init** is the other half of the cure: sensible starting weights keep the per-layer multiplies near `1` in the first place, so the product neither fades nor explodes.

%%% insight
Notice the pattern across both dials you just drove: nobody had to change the *neuron*. Both cures change the rules the gradient travels under — leave a trickle open, or put a ceiling on. That's how almost every training fix in this field works, right up to the models you use every day.
%%%

#### One dial, four ways to go wrong
Here's the payoff for sitting through the puzzles. They are not four things to memorise — they are one number behaving badly. Every layer the gradient passes through multiplies it by something (the activation's slope times the weight); the whole story is *what that something is*:
~~~zh
**你刚刚让什么发生了：** 左边，一个负的 `z` 配上朴素的 ReLU，会把那根条形抽空到正好 `0` —— 而一旦它是 `0`，就没有任何东西能把它填回去。勾上 **Leaky**，同一个被推下去的单元还留着一丝 gradient，够它爬回来。**合理的初始化**也有帮助：把 weight 的起始大小定得合理，单元就不会在第一天就被推下悬崖。

#### 尖叫 —— **exploding gradient（梯度爆炸）**
现在看右边那个旋钮。把 weight 的大小往上拉，规则 4 那个累积乘积就不再缩小，而是开始滚雪球：

%%% steps
step: 尖叫 —— 每一层要乘的数字是大的而不是小的，于是累积乘积越来越大
why: 和变弱是同一串乘法，只是指向了另一边
step: 炸开 —— 这样几层下来你就得到**巨大的值**：这就是 **exploding gradient**
why: 这个 gradient 还是「正确」的 —— 它只是大得离谱，而一步这么大会**迈过**整个山谷，所以 loss 常常回来就是 `inf` 或者 `NaN`
step: 修法 —— **gradient clipping（梯度裁剪）**：设一个上限，迈步之前把超过它的都**裁掉**
why: 勾上 clip 那个框，每一根到达上限的条形就停在那里 —— 一个怪物级的数字再也不能把你从山上射出去
%%%

%%% demo id=explodezh label="先猜，再展开"
predict: 在六层里每一层都把 gradient 乘 3。它会到大约 18、大约 100，还是超过 700？
code: g=1.0; big=3.0; [round(g:=g*big,1) for _ in range(6)]  # 每一层都把 gradient 乘上一个很大的量
out: [3.0, 9.0, 27.0, 81.0, 243.0, 729.0]
take: <b>大的乘数把 gradient 一路推向无穷。</b>每层 ×3，gradient 六步之内就从 3 蹿到 729 —— 一步这么大会疯狂地迈过头，而 loss 常常打印成 inf 或者 NaN。裁剪它，或者把起始 weight 的大小定得让每层乘数落在 1 附近。
%%%

而**谨慎初始化**是解药的另一半：合理的起始 weight 从一开始就让每层乘数留在 `1` 附近，于是乘积既不变弱也不爆炸。

%%% insight
注意你刚开过的这两个旋钮共有的模式：没有人需要去改那个*神经元*。两个解药改的都是 gradient 旅行时要遵守的规则 —— 留一条细缝让它漏过去，或者给它加一个天花板。这个领域里几乎每一个训练上的修法都是这样工作的，一直到你每天在用的那些模型。
%%%

#### 一个旋钮，四种出错的方式
这就是你坐着看完这些谜题的回报。它们不是四件要背的东西 —— 它们是一个数字在闹脾气。gradient 每穿过一层，就被乘上某个东西（activation 的斜率乘上 weight）；整个故事就是*那个东西是多少*：
~~~

%%% svg
<svg viewBox="0 0 520 186" role="img" aria-label="One dial showing the per-layer multiplier from 0 to above 1. At exactly 0 the label reads dead unit, cured by Leaky ReLU. Below 1 the label reads the fade, cured by ReLU. At about 1 the label reads healthy, the signal survives. Above 1 the label reads explode, cured by clipping. A separate note below shows that if every unit multiplies by the same value, they become twins, cured by small random init."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The same multiply, four ways — what does each layer multiply by?</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">同一个乘法，四种走向 —— 每一层乘的是多少？</text><line x1="40" y1="76" x2="480" y2="76" stroke="#B8AEA2" stroke-width="2"/><g stroke="#B8AEA2"><line x1="40" y1="70" x2="40" y2="82"/><line x1="200" y1="70" x2="200" y2="82"/><line x1="330" y1="70" x2="330" y2="82"/><line x1="480" y1="70" x2="480" y2="82"/></g><text x="40" y="96" text-anchor="middle" fill="#6B645E" font-size="9">0</text><text class="lang-en" x="200" y="96" text-anchor="middle" fill="#6B645E" font-size="9">under 1</text><text class="lang-zh" x="200" y="96" text-anchor="middle" fill="#6B645E" font-size="9">小于 1</text><text x="330" y="96" text-anchor="middle" fill="#6B645E" font-size="9">≈ 1</text><text class="lang-en" x="480" y="96" text-anchor="middle" fill="#6B645E" font-size="9">over 1</text><text class="lang-zh" x="480" y="96" text-anchor="middle" fill="#6B645E" font-size="9">大于 1</text><circle cx="40" cy="76" r="7" fill="#C93B3B"/><text class="lang-en" x="40" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">dead unit 🔌</text><text class="lang-zh" x="40" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">死掉的单元 🔌</text><text x="40" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">→ Leaky ReLU</text><circle cx="200" cy="76" r="7" fill="#C99A12"/><text class="lang-en" x="200" y="46" text-anchor="middle" fill="#C99A12" font-size="10">the fade 📞</text><text class="lang-zh" x="200" y="46" text-anchor="middle" fill="#C99A12" font-size="10">变弱 📞</text><text class="lang-en" x="200" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">→ ReLU (slope 1)</text><text class="lang-zh" x="200" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">→ ReLU（斜率 1）</text><circle cx="330" cy="76" r="7" fill="#2D8B55"/><text class="lang-en" x="330" y="46" text-anchor="middle" fill="#2D8B55" font-size="10">healthy 🎉</text><text class="lang-zh" x="330" y="46" text-anchor="middle" fill="#2D8B55" font-size="10">健康 🎉</text><text class="lang-en" x="330" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">signal survives</text><text class="lang-zh" x="330" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">信号活下来</text><circle cx="480" cy="76" r="7" fill="#C93B3B"/><text class="lang-en" x="470" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">explode 🎤</text><text class="lang-zh" x="470" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">爆炸 🎤</text><text class="lang-en" x="470" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">→ clip it</text><text class="lang-zh" x="470" y="58" text-anchor="middle" fill="#2D8B55" font-size="9">→ 把它裁掉</text><line x1="40" y1="122" x2="480" y2="122" stroke="#E5DFD6"/><text class="lang-en" x="46" y="144" fill="#C93B3B" font-size="10">👯 and if EVERY unit multiplies by the very same thing → twins</text><text class="lang-zh" x="46" y="144" fill="#C93B3B" font-size="10">👯 而如果每一个单元都乘同一个东西 → 双胞胎</text><text class="lang-en" x="46" y="160" fill="#2D8B55" font-size="10">→ small random starting weights, so the units differ</text><text class="lang-zh" x="46" y="160" fill="#2D8B55" font-size="10">→ 很小的随机起始 weight，让这些单元各不相同</text><text class="lang-en" x="260" y="180" text-anchor="middle" fill="#6B645E" font-size="9.5">aim for multiplies near 1, plus a little healthy variety between units</text><text class="lang-zh" x="260" y="180" text-anchor="middle" fill="#6B645E" font-size="9.5">目标是乘数在 1 附近，再加上单元之间一点健康的差异</text></g></svg>
%%%

!!! c-info 🧭
<b>Keep it straight:</b> all four traps are the same math seen from different angles. <b>Fade</b> = multiplying by numbers under 1. <b>Explode</b> = multiplying by numbers over 1. <b>Dead unit</b> = multiplying by exactly 0. <b>Twins</b> = every unit multiplying by the <i>same</i> thing. The cures all aim for the same sweet spot: per-layer multiplies near 1, and a little healthy variety between units.
!!!

**Four traps, four quick cures — and that was the hardest stretch of the day, now behind you.** One page to tie it all together.
~~~zh
!!! c-info 🧭
<b>把它理清楚：</b>这四个陷阱是同一份数学从不同角度看过去。<b>变弱</b> = 乘上小于 1 的数字。<b>爆炸</b> = 乘上大于 1 的数字。<b>死掉的单元</b> = 乘上正好 0。<b>双胞胎</b> = 每个单元都乘上<i>同一个</i>东西。这些解药瞄准的都是同一个最佳点：每层乘数在 1 附近，再加上单元之间一点健康的差异。
!!!

**四个陷阱，四个快速解药 —— 而那是今天最难的一段，现在已经在你身后了。** 用一页把所有东西串起来。
~~~

@@@ concept id=c10 zh_tag="回顾" tag="Recap" zh_title="一页看完今天" title="Today in one page" zh_gotit="回顾好了" gotit="Got the recap"
Take a breath — you just learned how a network figures out which way to improve, which is genuinely the core idea of modern AI. Here's a way to hold the whole day at once: picture our **hiker in fog** one more time, and read the beats below as the steps of a single descent, in order. Each step only made sense because of the one before it — you can't trace blame backward until you've saved your work going forward. **Where the hiker picture breaks down:** a hiker takes the downhill step herself, but today you only learned how to *find* the downhill direction; actually taking the step is tomorrow's job (the optimizer).

**The trail you walked, in order:**

- **Step 1 · The downhill arrow.** The **gradient** answers "which way is downhill, for every weight?" It points **uphill** (more error), so to learn we step the **opposite** way, downhill — that's the **minus** sign in tomorrow's update.
- **Step 2 · Save your work.** The **forward pass** makes a guess and quietly saves its in-between values (`z`, `a`) — the sticky notes the backward trip re-uses.
- **Step 3 · Chain of effects.** A **local slope** is one tiny step's effect; the **chain rule** multiplies the local slopes along the path to get the whole effect.
- **Step 4 · Trace the blame.** **Backpropagation** starts at the loss and sweeps backward, multiplying local slopes, handing every weight its gradient in **one** pass — the reason big networks are trainable at all.
- **Step 5 · Each part's share.** Four tiny rules, in backward order: the **activation charges a toll** (multiply by its slope at the saved `z`, giving `δ`); the **weight's share = its input × δ** (big inputs → big updates); the **bias travels free** (slope 1, so the gradient passes straight through); and to travel **one layer further back**, multiply by the **weight** (`δ × w`).
- **Step 6 · The fading signal.** **Vanishing**: per-layer multiplies below 1 (sigmoid's slope is ≤ 0.25) stack up and drive the gradient toward 0, so early layers freeze — cure: **ReLU**, whose slope stays 1, plus sanely sized starting weights.
- **Step 7 · Check your work.** The **gradient check** measures the same slope a second way (nudge ±ε, divide) and compares — 30 seconds that catch a silent bug.
- **Step 8 · The twin trap.** Start every weight at the same value and every neuron gets **identical gradients**, so they **stay identical** forever — cure: **small random** starting weights to break the symmetry.
- **Step 9 · Stuck and squealing.** **Dead ReLU** (unit shoved permanently negative, slope 0, never learns → **Leaky ReLU**, a small negative slope, plus sensible init) and **exploding** (big multiplies snowball to huge values → **clip** the gradient + careful init).

Here's the one picture to keep — the whole flow, forward then back.
~~~zh
喘一口气 —— 你刚学会了一个网络怎么弄清该往哪个方向变好，而这真的就是现代 AI 的核心想法。有一个办法可以一次抓住一整天：再想一次我们那位**雾里的登山者**，然后把下面这些拍子当成一次下坡的一步一步，按顺序读。每一步之所以讲得通，都是因为它前面那一步 —— 你没有在往前走的时候存好笔记，就没法往回追责任。**登山者这个比喻在哪里不成立：** 登山者自己会迈出下坡那一步，而今天你只学了怎么*找到*下坡的方向；真的迈出那一步是明天的活（optimizer）。

**你走过的那条路，按顺序：**

- **第 1 步 · 下坡的箭。** **gradient** 回答的是「对每一个 weight 来说，哪边是下坡？」它指向**上坡**（错误更多），所以学习就是往**相反**的方向走、往下坡走 —— 那就是明天那个 update 里的**减号**。
- **第 2 步 · 留好笔记。** **forward pass** 做出一个猜测，同时不声不响地把中间值（`z`、`a`）存下来 —— 那些便利贴，回程会重新读一遍。
- **第 3 步 · 一串影响。** 一个**局部斜率**是一小步的影响；**chain rule** 把路上的局部斜率乘起来，得到整个影响。
- **第 4 步 · 往回追责任。** **backpropagation** 从 loss 出发往回扫，一路乘局部斜率，**一趟**就把 gradient 发给每一个 weight —— 这正是大网络能训练得起来的原因。
- **第 5 步 · 各自的份额。** 四条小规则，按往回走的顺序：**activation 收过路费**（乘上它在存下的 `z` 处的斜率，得到 `δ`）；**weight 该担的 = 它的输入 × δ**（大输入 → 大 update）；**bias 免费通行**（斜率 1，所以 gradient 原样穿过）；而要**再往回走一层**，就乘上那个 **weight**（`δ × w`）。
- **第 6 步 · 变弱的信号。** **vanishing**：每层小于 1 的乘数（sigmoid 的斜率 ≤ 0.25）叠起来，把 gradient 一路推向 0，于是早期的层冻住 —— 解药：**ReLU**，它的斜率一直是 1，再加上大小合理的起始 weight。
- **第 7 步 · 检查你的活。** **gradient check** 用第二种办法量同一个斜率（±ε 挪一下，再相除）然后比一比 —— 三十秒抓住一个不出声的 bug。
- **第 8 步 · 双胞胎陷阱。** 让每个 weight 都从同一个值开始，每个神经元就拿到**完全相同的 gradient**，于是它们**永远保持一样** —— 解药：**很小的随机**起始 weight，用来打破对称。
- **第 9 步 · 卡住与尖叫。** **dead ReLU**：单元被推成永久为负，斜率 0，再也学不到。解药是 **Leaky ReLU**，一个很小的负侧斜率，再加上合理的 init。**exploding**：大的乘数滚雪球滚成巨大的值。解药是把 gradient **裁掉**，再加上谨慎的 init。

这是要留住的那一张图 —— 整个流程，先往前再往回。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Recap flow: forward pass goes right from input through weighted sum to activation to loss, saving z and a; backward pass returns left multiplying local slopes to give each weight its gradient. Below, four traps and their cures are listed."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The whole day: forward saves, backward traces blame</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">一整天：往前存笔记，往回追责任</text><rect x="16" y="30" width="60" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="46" y="47" text-anchor="middle" fill="#1F6280">input</text><text class="lang-zh" x="46" y="47" text-anchor="middle" fill="#1F6280">输入</text><rect x="120" y="30" width="86" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="163" y="47" text-anchor="middle" fill="#9A7208">sum → z 📌</text><text class="lang-zh" x="163" y="47" text-anchor="middle" fill="#9A7208">加权和 → z 📌</text><rect x="250" y="30" width="86" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text class="lang-en" x="293" y="47" text-anchor="middle" fill="#5E5191">act → a 📌</text><text class="lang-zh" x="293" y="47" text-anchor="middle" fill="#5E5191">激活 → a 📌</text><rect x="380" y="30" width="86" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="423" y="47" text-anchor="middle" fill="#C93B3B">loss</text><g stroke="#2A7B9B" stroke-width="1.8" fill="#2A7B9B"><path d="M76 43 l40 0"/><polygon points="116,43 108,39 108,47"/><path d="M206 43 l40 0"/><polygon points="246,43 238,39 238,47"/><path d="M336 43 l40 0"/><polygon points="376,43 368,39 368,47"/></g><text class="lang-en" x="250" y="24" text-anchor="middle" fill="#2A7B9B" font-size="9">forward →</text><text class="lang-zh" x="250" y="24" text-anchor="middle" fill="#2A7B9B" font-size="9">往前 →</text><g stroke="#2D8B55" stroke-width="1.8" fill="#2D8B55"><path d="M380 70 l-40 0"/><polygon points="340,70 348,66 348,74"/><path d="M250 70 l-40 0"/><polygon points="210,70 218,66 218,74"/><path d="M120 70 l-40 0"/><polygon points="80,70 88,66 88,74"/></g><text class="lang-en" x="250" y="86" text-anchor="middle" fill="#2D8B55" font-size="9">← backward: × local slopes → each weight's gradient</text><text class="lang-zh" x="250" y="86" text-anchor="middle" fill="#2D8B55" font-size="9">← 往回：× 局部斜率 → 每个 weight 的 gradient</text><line x1="30" y1="100" x2="490" y2="100" stroke="#E5DFD6"/><text class="lang-en" x="30" y="120" fill="#C93B3B">fade → ReLU</text><text class="lang-zh" x="30" y="120" fill="#C93B3B">变弱 → ReLU</text><text class="lang-en" x="160" y="120" fill="#C93B3B">dead → Leaky ReLU</text><text class="lang-zh" x="160" y="120" fill="#C93B3B">死掉 → Leaky ReLU</text><text class="lang-en" x="330" y="120" fill="#C93B3B">explode → clip</text><text class="lang-zh" x="330" y="120" fill="#C93B3B">爆炸 → clip</text><text class="lang-en" x="30" y="140" fill="#C93B3B">twins → small random init</text><text class="lang-zh" x="30" y="140" fill="#C93B3B">双胞胎 → 很小的随机 init</text><text class="lang-en" x="240" y="140" fill="#2D8B55">check → measure the slope two ways</text><text class="lang-zh" x="240" y="140" fill="#2D8B55">检查 → 用两种办法量斜率</text><text class="lang-en" x="260" y="164" text-anchor="middle" fill="#6B645E">aim for per-layer multiplies near 1, a little variety between units</text><text class="lang-zh" x="260" y="164" text-anchor="middle" fill="#6B645E">目标是每层乘数在 1 附近，单元之间有一点差异</text></g></svg>
%%%

#### Cheat-sheet · each part's gradient, and the traps

%%% table
:: Part / trap :: In one line :: Remember
activation :: multiplies the gradient by its slope at the saved `z`, giving `δ` :: sigmoid ≤ 0.25 (tiny in tails), ReLU is 1 or 0
weight gradient :: its input × `δ` (the gradient after the activation's toll) :: big input → big update
bias gradient :: `δ`, unchanged :: slope 1 → passes straight through
one layer further back :: `δ × w` — multiply by the weight you came through :: this is why weight SIZE decides fade vs explode
vanishing :: per-layer multiplies under 1 stack up toward 0 :: cure: ReLU (slope stays 1) + sane weight sizes
dead ReLU :: unit stuck negative, slope 0 forever :: cure: Leaky ReLU (small negative slope) + sensible init
exploding :: per-layer multiplies over 1 snowball toward ∞ :: cure: gradient clipping + careful init
same-value init (twins) :: identical weights get identical gradients :: cure: small random init
gradient check :: measure the slope a second way and compare :: use it to check, never to train
%%%

#### Cheat-sheet · the words you met today

%%% jargon
gradient | the arrow saying which way to nudge every weight to change the loss fastest; it points uphill, so we step the opposite way
forward pass | running the input through the neuron to a guess, saving z and a for the trip back
computation graph | the neuron's math seen as a chain of tiny steps, each with an easy local slope
local derivative | the slope of just one tiny step — how much its own output moves when its own input moves a hair
chain rule | multiply the local slopes along the path to get how the loss depends on an early weight
upstream gradient | the gradient arriving at a step from the loss side (the "error signal"); × the step's local slope to pass it further back
delta (δ) | the gradient that got through the activation's toll — what the weight and bias then share
backpropagation | one backward sweep from the loss that multiplies local slopes to give every weight its gradient
vanishing gradient | per-layer multiplies below 1 (sigmoid's slope ≤ 0.25) stacked over many layers shrink the gradient toward 0, so early layers stop learning
dead ReLU | a unit stuck outputting 0 with slope 0, so it never learns again → cure: Leaky ReLU
exploding gradient | per-layer multiplies above 1 blow the gradient up huge → cure: gradient clipping
gradient check | measure a slope by nudging a weight ±ε and compare it to backprop, to catch bugs
%%%

That's the whole day. Tomorrow you take the downhill step for real — the **training loop**, where these gradients actually change the weights, over and over, until the network learns.
~~~zh
#### 小抄 · 每一部分的 gradient，还有那些陷阱

%%% table
:: 部分 / 陷阱 :: 一句话 :: 记住
activation :: 把 gradient 乘上它在存下的 `z` 处的斜率，得到 `δ` :: sigmoid ≤ 0.25（尾巴上极小），ReLU 是 1 或者 0
weight gradient :: 它的输入 × `δ`（交过 activation 过路费之后的 gradient） :: 大输入 → 大 update
bias gradient :: `δ`，原样不变 :: 斜率 1 → 直接穿过
再往回一层 :: `δ × w` —— 乘上你走过来的那个 weight :: 这就是为什么 weight 的**大小**决定变弱还是爆炸
vanishing :: 每层小于 1 的乘数叠起来一路趋向 0 :: 解药：ReLU（斜率一直是 1）+ 合理的 weight 大小
dead ReLU :: 单元卡在负的一侧，斜率永远是 0 :: 解药：Leaky ReLU（很小的负侧斜率）+ 合理的 init
exploding :: 每层大于 1 的乘数滚雪球趋向 ∞ :: 解药：gradient clipping + 谨慎的 init
同值 init（双胞胎） :: 一样的 weight 拿到一样的 gradient :: 解药：很小的随机 init
gradient check :: 用第二种办法量斜率再比一比 :: 用它来检查，绝不用它来训练
%%%

#### 小抄 · 你今天见到的那些词

%%% jargon
gradient | 那支箭，说每一个 weight 该往哪边挪才能最快改变 loss；它指向上坡，所以我们往相反方向走
forward pass | 把输入送过神经元得到一个猜测，同时存下 z 和 a 给回程用
computation graph | 把神经元的数学看成一串很小的步子，每一步都有一个好算的局部斜率
local derivative | 只有一小步的斜率 —— 当它自己的输入挪一根头发时，它自己的输出挪多少
chain rule | 把路上的局部斜率乘起来，得到 loss 怎么依赖一个早期的 weight
upstream gradient | 从 loss 那一侧到达某一步的 gradient（也就是「error signal」）；× 这一步的局部斜率就能继续往回传
delta (δ) | 交过 activation 过路费之后剩下的 gradient —— weight 和 bias 接着分的就是它
backpropagation | 从 loss 出发往回扫的一趟，一路乘局部斜率，给每一个 weight 它的 gradient
vanishing gradient | 每层小于 1 的乘数（sigmoid 斜率 ≤ 0.25）叠过很多层，把 gradient 缩向 0，于是早期的层停止学习
dead ReLU | 一个卡在输出 0、斜率也是 0 的单元，所以它再也学不到东西 → 解药：Leaky ReLU
exploding gradient | 每层大于 1 的乘数把 gradient 吹得巨大 → 解药：gradient clipping
gradient check | 把一个 weight 挪 ±ε 来量出斜率，再和 backprop 比一比，用来抓 bug
%%%

今天就是这些。明天你真的要迈出下坡那一步了 —— **训练循环**，那些 gradient 在那里真正改变 weight，一遍又一遍，直到网络学会。
~~~

@@@ quiz id=quiz zh_tag="小测" tag="Quiz" zh_title="点一个答案 —— 每题都有即时反馈" title="Click an answer — instant feedback on each" zh_gotit="先全部答完" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)

%%% quiz
q: What does the gradient tell the network, and which way do we step? | a:2 | The final loss value; we step toward it | How many layers to use; we step randomly | Which way is uphill (more error) for every weight; we step the opposite way, downhill | The learning rate; we don't step | fb: The gradient points UPHILL — the direction that makes the loss bigger. So learning means stepping the opposite way, downhill, which is exactly why tomorrow's update carries a minus sign. (A very common mix-up is to say it "points downhill" — it doesn't; we flip it.)
q: What is the chain rule, in one line? | a:1 | Add up the loss of every layer | Multiply the local slopes along the path to get how an early weight affects the far-away loss | Pick the layer with the biggest weight | Run the network backward with new random weights | fb: Multiply, don't add. Pedal → speed → distance → money: three easy one-hop slopes multiplied give the whole effect, and that single move is the engine of backprop.
q: Why do deep sigmoid networks suffer from vanishing gradients, and what's the common cure? | a:2 | Sigmoid uses too much memory; cure: bigger batches | The loss is negative; cure: flip its sign | The sigmoid's slope is at most 0.25, so multiplying it in at every layer shrinks the gradient toward 0; cure: ReLU, whose slope stays 1 for positive inputs | Sigmoid outputs are too large; cure: clip them | fb: Every sigmoid layer multiplies the gradient by at most 0.25, so the whisper fades about 4× per layer. ReLU's slope is a clean 1 for positive inputs, so the activation stops eating the signal — the weights still have to be sized sensibly for the rest.
q: A ReLU unit is stuck outputting 0 with slope 0 and never learns. What is this, and how do you fix it? | a:1 | Exploding gradient; fix by lowering the learning rate | A dead ReLU; fix with Leaky ReLU, which keeps a small nonzero slope on the negative side, plus sensible weight initialization | Vanishing gradient; fix with a bigger network | Zero-init symmetry; fix by clipping | fb: Slope 0 means gradient 0, so the unit can never climb back — a dead ReLU. Leaky ReLU keeps a small rising slope (about 0.01) on the negative side, so a trickle of gradient always gets through.
%%%
~~~zh
四道快问，每题都有即时反馈 —— 四道全答完就算完成今天。（如果有一道把你绊住了，那是让你往回翻的提示，不是一个不及格的分数。）

%%% quiz
q: gradient 告诉网络什么，而我们往哪边迈步？ | a:2 | 最终那个 loss 的值；我们朝它走 | 该用多少层；我们随机迈步 | 对每一个 weight 来说哪边是上坡（错误更多）；我们往相反方向走，也就是下坡 | learning rate（学习率）；我们不迈步 | fb: gradient 指向「上坡」—— 让 loss 变大的那个方向。所以学习就是往相反方向走、往下坡走，这正是明天那个 update 带一个减号的原因。（一个很常见的搞混是说它「指向下坡」—— 它不是；是我们把它翻过来。）
q: chain rule 是什么，一句话说？ | a:1 | 把每一层的 loss 加起来 | 把路上的局部斜率乘起来，得到一个早期的 weight 怎么影响远处的 loss | 挑 weight 最大的那一层 | 用新的随机 weight 把网络倒着跑一遍 | fb: 是乘，不是加。油门 → 速度 → 路程 → 钱：三个好算的一跳斜率乘起来就给出整个影响，而这一个动作就是 backprop 的引擎。
q: 很深的 sigmoid 网络为什么会有 vanishing gradient，常见的解药是什么？ | a:2 | sigmoid 太吃内存；解药：更大的 batch | loss 是负的；解药：把它的符号翻过来 | sigmoid 的斜率最多只有 0.25，所以每一层都乘它一次会把 gradient 缩向 0；解药：ReLU，它对正输入的斜率一直是 1 | sigmoid 的输出太大；解药：把它们裁掉 | fb: 每一个 sigmoid 层最多把 gradient 乘 0.25，所以这句耳语每层大约轻四倍。ReLU 对正输入的斜率是干净的 1，所以 activation 不再吃掉信号 —— 剩下那一半还是得靠把 weight 的大小定得合理。
q: 一个 ReLU 单元卡在输出 0、斜率也是 0，再也学不到东西。这是什么，你怎么修？ | a:1 | exploding gradient；靠降低 learning rate 来修 | 一个 dead ReLU；用 Leaky ReLU 来修，它在负的一侧保留一个很小的非零斜率，再加上合理的 weight 初始化 | vanishing gradient；用一个更大的网络来修 | 零初始化造成的对称；靠裁剪来修 | fb: 斜率 0 就意味着 gradient 0，所以这个单元永远爬不回来 —— 一个 dead ReLU。Leaky ReLU 在负的一侧保留一个很小的上升斜率（大约 0.01），所以总有一丝 gradient 能漏过去。
%%%
~~~

@@@ produce id=produce zh_tag="动手做" tag="Produce" zh_title="看 gradient 往回流 —— 再用两种办法检查它" title="Watch a gradient flow backward — and check it two ways" zh_gotit="做完了" gotit="Done"
Time to see today's big idea with your own eyes. You'll run a tiny neuron forward, trace the gradient backward through it by hand with the rules, and then *check your work* by measuring the same slope a totally different way. **Predict first:** for a neuron with input `x = 2` and a positive `z`, will the *weight's* gradient be bigger or smaller than the *bias's* gradient — and by what factor? Then run it and **watch** the weight's gradient come out exactly `x` times the bias's, because the weight's blame is scaled by its input while the bias just passes `δ` through. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py`. (1) Do a forward pass for one neuron: `z = w*x + b` then `a = relu(z)`, and a squared-error loss `L = (a - target)**2`; *save* `x`, `z`, `a` as you go. (2) Backprop by hand: the gradient arriving from the loss is `incoming = 2*(a-target)`, the ReLU slope is `1` if `z>0` else `0`, so Rule 1 gives `delta = incoming * relu_slope`, then Rule 2 gives `w_grad = delta * x` and Rule 3 gives `b_grad = delta` — **notice** `w_grad` is `x` times `b_grad`. (3) **Check it:** write `loss_of_w(w)` and compute the measured slope `(loss_of_w(w+1e-5) - loss_of_w(w-1e-5)) / (2e-5)`, then print it next to your `w_grad` and **watch** them match. Run with `python3 sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 5 artifact.
Create sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py that, with a comment on each step:
1. Runs a forward pass for one neuron: given x=2.0, w=0.5, b=0.1, target=1.0, compute z=w*x+b, a=relu(z) (relu = max(0,z)), and loss L=(a-target)**2; save x, z, a.
2. Backprops by hand with the lesson's rules: incoming = 2*(a-target); relu_slope = 1.0 if z>0 else 0.0; Rule 1 delta = incoming*relu_slope; Rule 2 w_grad = delta*x; Rule 3 b_grad = delta. Print all three and show w_grad == x * b_grad (big input -> big weight update).
3. Gradient-checks w_grad: define loss_of_w(w) that redoes the forward pass and returns L, then measured = (loss_of_w(w+1e-5) - loss_of_w(w-1e-5))/(2e-5); print measured next to w_grad and show they match to ~4 decimals.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the gradient, traced and verified)
- The **weight's gradient is exactly `x` times the bias's** — with `x = 2`, twice as big — because a weight's blame is scaled by its input while the bias passes `δ` straight through.
- Flip `z` negative (try a big negative `b`) and **both gradients become `0`** — the ReLU slope is `0` there, so that unit gets no signal at all. That's a dead ReLU, live in front of you.
- The **measured slope matches your hand-computed `w_grad`** to about four decimals — proof your backward pass is correct, exactly the check a careful engineer runs before trusting the code.

!!! c-info 📓
<b>5-minute research log:</b> before you close the tab, write three lines in <code>sessions/m02-the-neuron/day-05-gradients-backprop/log.md</code>: (1) what the gradient is, and which way we step; (2) why the forward pass saves its in-between values; (3) one trap from today (vanishing, dead ReLU, exploding, or twins) and its one-line cure.
!!!
~~~zh
是时候用你自己的眼睛看今天这个大想法了。你会让一个很小的神经元往前跑一次，用那几条规则亲手把 gradient 往回追一遍，然后用一个完全不同的办法量同一个斜率来*检查你的活*。**先预测：** 对一个输入 `x = 2`、`z` 为正的神经元，*weight* 的 gradient 会比 *bias* 的更大还是更小 —— 差几倍？然后跑一遍，**看着** weight 的 gradient 正好是 bias 的 `x` 倍，因为 weight 该担的责任被它的输入缩放过，而 bias 只是把 `δ` 原样放过去。挑一条路走。

#### 路线 A · 自己写
建一个 `sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py`。（1）给一个神经元做一次 forward pass：`z = w*x + b`，然后 `a = relu(z)`，再算平方误差 loss `L = (a - target)**2`；一路把 `x`、`z`、`a` *存下来*。（2）亲手做 backprop。从 loss 到达的 gradient 是 `incoming = 2*(a-target)`。ReLU 斜率在 `z>0` 时是 `1`，否则是 `0`。所以规则 1 给出 `delta = incoming * relu_slope`。接着规则 2 给出 `w_grad = delta * x`，规则 3 给出 `b_grad = delta` —— **注意** `w_grad` 是 `b_grad` 的 `x` 倍。（3）**检查它：** 写一个 `loss_of_w(w)`，算出量出来的斜率 `(loss_of_w(w+1e-5) - loss_of_w(w-1e-5)) / (2e-5)`，然后把它打印在你的 `w_grad` 旁边，**看着**它们对上。用 `python3 sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py` 跑。

#### 路线 B · 让 Claude 搭出来，然后读它
把下面那段 prompt 复制回 Claude Code。它会让 Claude Code 建好文件、写好代码，并且替你跑一遍。（prompt 本身保持英文，它是给工具看的指令。）

%%% prompt id=ppzh label="让 Claude 帮你搭出来"
Help me build my Module 2 Day 5 artifact.
Create sessions/m02-the-neuron/day-05-gradients-backprop/experiment.py that, with a comment on each step:
1. Runs a forward pass for one neuron: given x=2.0, w=0.5, b=0.1, target=1.0, compute z=w*x+b, a=relu(z) (relu = max(0,z)), and loss L=(a-target)**2; save x, z, a.
2. Backprops by hand with the lesson's rules: incoming = 2*(a-target); relu_slope = 1.0 if z>0 else 0.0; Rule 1 delta = incoming*relu_slope; Rule 2 w_grad = delta*x; Rule 3 b_grad = delta. Print all three and show w_grad == x * b_grad (big input -> big weight update).
3. Gradient-checks w_grad: define loss_of_w(w) that redoes the forward pass and returns L, then measured = (loss_of_w(w+1e-5) - loss_of_w(w-1e-5))/(2e-5); print measured next to w_grad and show they match to ~4 decimals.
Then run it and paste the output at the bottom as a comment.
%%%

#### 你应该看到什么（gradient 被追出来，也被验证了）
- **weight 的 gradient 正好是 bias 的 `x` 倍** —— `x = 2` 的时候是两倍 —— 因为一个 weight 该担的责任被它的输入缩放过，而 bias 把 `δ` 原样放过去。
- 把 `z` 翻成负的（试一个很负的 `b`），**两个 gradient 都变成 `0`** —— ReLU 的斜率在那里是 `0`，所以那个单元完全拿不到信号。那就是一个 dead ReLU，活生生摆在你面前。
- **量出来的斜率和你手算的 `w_grad` 对得上**，大约到小数点后四位 —— 这证明你的 backward pass 是对的，正是一个仔细的工程师在信任代码之前跑的那个检查。

!!! c-info 📓
<b>五分钟研究日志：</b>关掉这个页面之前，在 <code>sessions/m02-the-neuron/day-05-gradients-backprop/log.md</code> 里写三行：（1）gradient 是什么，我们往哪边迈步；（2）forward pass 为什么要存下它的中间值；（3）今天的一个陷阱（vanishing、dead ReLU、exploding，或者双胞胎）以及它那一行解药。
!!!
~~~

@@@ fin

