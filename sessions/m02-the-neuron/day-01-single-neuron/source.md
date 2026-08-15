---
quest_id: wf2-d01-neuron
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 1 — A Single Neuron"
module_label: "Module 2 · Train · Day 1"
zh_module_label: "第 2 模块 · 训练 · 第 1 天"
title: "A Single Neuron"
zh_title: "一个神经元"
subtitle: "The Smallest Piece of a Learning Machine"
zh_subtitle: "会学习的机器里最小的一块"
brand_sub: "Foundations · M2 Day 1"
spine: "brain cell: weigh → add → decide"
zh_spine: "脑细胞"
nav_prev_href: "../../m01-shape-of-data/review.html"
nav_prev_label: "M1 · Review Gate"
zh_nav_prev_label: "M1 · 复习关卡"
nav_next_href: "../day-02-activations/lesson.html"
nav_next_label: "Activation Functions"
zh_nav_next_label: "激活函数"
fin_title: "Module 2 · Day 1 complete! 🏆"
zh_fin_title: "第 2 模块 · 第 1 天 完成！🏆"
fin_body: "Nice work — you've met the <b>single neuron</b>: it weighs its inputs, adds them up with a bias, and decides. That tiny weigh → add → decide is the atom every big model is built from.<br>Next up: <b>Activation Functions</b> — the little bend that lets neurons stack into something deep."
zh_fin_body: "做得好 —— 你见过<b>单个 neuron</b> 了：它给每个输入称一称，连着 bias 一起加起来，然后做决定。这个小小的「称一称 → 加起来 → 做决定」，就是每一个大模型的原子。<br>下一站：<b>激活函数</b> —— 那个小小的弯，让 neuron 能叠成真正有深度的东西。"
notebook_yardstick: 00-neural-networks/fundamentals/01_what_is_a_neural_network.ipynb
coverage_topics:
  - {topic: single neuron, keywords: [artificial neuron, a neuron, one neuron]}
  - {topic: inputs, keywords: [inputs, raw number]}
  - {topic: weights, keywords: [weight, trust dial]}
  - {topic: bias, keywords: [bias, starting mood, head start]}
  - {topic: weighted sum, keywords: [weighted sum, pre-activation, running total]}
  - {topic: activation function, keywords: [activation function, decide step, the activation]}
  - {topic: step function, keywords: [step function, mcculloch, perceptron, height bar]}
  - {topic: sigmoid, keywords: [sigmoid, dimmer, s-curve, s-shaped]}
  - {topic: decision boundary, keywords: [decision boundary, straight line, chalk line]}
  - {topic: biological neuron inspiration, keywords: [brain cell, inspired by]}
  - {topic: capability limit XOR, keywords: [xor, linearly separable, one line can]}
  - {topic: layers and networks fix, keywords: [layer, hidden layer, neural network, deep learning]}
  - {topic: forward propagation, keywords: [forward propagation, forward pass, conveyor]}
  - {topic: real-world applications, keywords: [image recognition, speech recognition, language translation, game playing, recommendation, self-driving]}
  - {topic: how networks learn, keywords: [guess, check error, adjust, repeat, training is the search]}
  - {topic: common misconceptions, keywords: [myth, magic black box, bigger is always, think like a human, pattern]}
  - {topic: optional further reading, keywords: [further reading, michael nielsen, 3blue1brown]}
---

@@@ hero
@lede Picture asking three friends whether to see a movie tonight. One friend has amazing taste, so you *lean on* what they say. One is right about half the time, so you only half-listen. One always hates everything, so you barely count their vote. You quietly weigh each friend, add it all up in your head, and land on a decision: yes or no. That little "weigh → add → decide" is *exactly* what one **neuron** does — a tiny math copy of a **brain cell**. And here's the wild part: connect a few million of these little vote-counters into a network and you get the thing behind ChatGPT finishing your sentence, your phone recognizing your face, or an app translating a language. Today you meet that single piece, all on its own. It's simpler than you fear, and by the end you'll honestly be able to say, "I know what the smallest part of an AI actually does."
@goal Together we'll build one neuron from nothing: its inputs, its **weights** (how much it trusts each input), its **bias** (its starting mood), the little sum it adds up, and the "decide" step at the end. Then you'll drag a real neuron's dials and watch it draw a line, catch the one pattern it *can't* learn, and meet the neat fix — the **layers** and **networks** that the word *deep* in "deep learning" points at. Every new word gets a plain-English meaning the moment it shows up. One promise about the math: anything in a box marked **Optional (skippable)** really is optional — skip every one and you'll still follow the whole day.

%%% warmup
q: This is Day 1 — no neural-network knowledge needed yet. Quick readiness check: what does "multiply" do to two numbers like 0.5 and 2? | a:2 | it adds them to get 2.5 | it picks the bigger one | it scales one by the other to get 1.0 | it makes them both zero | concept: prereq-multiply | fb: Multiplying scales one number by the other: 0.5 × 2 = 1.0. Today a neuron does exactly this to each input.
q: A neuron will hand you a plain list of numbers to work with, like [2, 3]. What is that — a single number, or a few numbers grouped together? | a:1 | a single number | a few numbers grouped in a list | a word | a picture | concept: prereq-list-of-numbers | fb: It's a small list (a group) of numbers. You met data as lists of numbers in M1 — today those become a neuron's inputs.
%%%
@zh_lede 想象你在问三个朋友，今天晚上要不要去看电影。一个朋友品味特别好，所以你很*看重*他说的话。一个朋友大概只有一半时候说对，所以你只听一半。还有一个朋友什么都讨厌，所以你几乎不算他的票。你悄悄给每个朋友称了称分量，在心里加起来，然后落到一个决定上：去，或者不去。这个小小的「称一称 → 加起来 → 做决定」，*正好*就是一个 **neuron（神经元）**做的事 —— 一份用数学做的**脑细胞**复制品。让人吃惊的地方在这里：几百万个这样的小小计票员连成一个网络，就成了别的东西。ChatGPT 替你把句子补完、手机认出你的脸、一个 app 把中文翻成英文 —— 背后都是它。今天你只见这一个零件，单独的一个。它比你担心的简单得多。到了最后你能真心说一句：「我知道一个 AI 里最小的那块到底在干什么。」
@zh_goal 我们会一起从零搭出一个 neuron：它的输入、它的 **weight（权重）**（它有多相信每个输入）、它的 **bias（偏置）**（它出门时的心情）、它加出来的那个小小的和，以及最后那一步「做决定」。然后你会亲手拖一个真 neuron 的旋钮，看它画出一条线。你会抓到它*学不会*的那一个花样，也会见到那个漂亮的解法 —— 「deep learning（深度学习）」里那个 *deep* 指的东西，也就是 **layer（层）**和**网络**。每一个新词第一次出现，就当场给你一句大白话的意思。关于数学有一个承诺：任何标了**选读（可以跳过）**的方框，真的都可以跳过。全部跳掉，你照样能跟完一整天。

~~~zh
%%% warmup
q: 今天是第 1 天 —— 还不需要任何神经网络知识。先热个身：把 0.5 和 2 这两个数「相乘」，是在做什么？ | a:2 | 把它们相加，得到 2.5 | 挑出更大的那个 | 用一个去缩放另一个，得到 1.0 | 让它们都变成零 | concept: prereq-multiply | fb: 相乘就是用一个数去缩放另一个：0.5 × 2 = 1.0。今天一个 neuron 对每个输入做的正是这件事。
q: 一个 neuron 会给你一串普通的数字来用，比如 [2, 3]。那是什么 —— 一个数字，还是几个数字凑在一起？ | a:1 | 一个数字 | 几个数字凑成的一串 | 一个词 | 一张图 | concept: prereq-list-of-numbers | fb: 那是一小串（一组）数字。你在 M1 里见过数据就是一串串数字 —— 今天它们变成一个 neuron 的输入。
%%%
~~~

@@@ concept id=c1 zh_tag="它是什么" tag="What it is" zh_title="一个 neuron：称一称 → 加起来 → 做决定" title="A neuron: weigh → add → decide" zh_gotit="懂了这个想法" gotit="Got the idea"
Let's start where the name comes from. Deep inside your head sit tiny cells called **brain cells** — the wiring that lets you read this sentence.

Each brain cell has little arms that catch signals from other cells. Some arms it trusts a lot; some it barely listens to. It gathers all those signals, sizes them up, and when it hears *enough*, it "fires" and passes a signal along. One small yes-or-no call.

An artificial [[neuron||A tiny math function: it takes a few input numbers, scales each by how much it matters, adds them up, and produces one output number.]] copies that idea with plain arithmetic — and that's the whole machine. Really.
~~~zh
我们先从这个名字的来处说起。你的脑袋深处住着一些很小的细胞，叫做**脑细胞** —— 就是让你能读懂这句话的那套线路。

每个脑细胞都有一些小手臂，用来接住别的细胞传来的信号。有些手臂它很相信；有些它几乎不听。它把所有这些信号收起来，估一估分量。等它听到*够多*的时候，就「发射」一下，把一个信号传下去。一次小小的「是」或者「不是」。

一个人造的 [[neuron||一个很小的数学函数：它拿几个输入数字，按每个有多重要缩放一下，加起来，给出一个输出数字。]] 用普通的算术照抄了这个想法 —— 而这就是整台机器了。真的。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A brain cell drawn like a blob with three input arms on the left catching signals, and one output arm on the right. Beneath it, the same idea as three labelled boxes: weigh the inputs, add them up, then decide."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A brain cell → a neuron: catch signals, weigh them, fire</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">脑细胞 → neuron：接住信号、称一称、发射</text><line x1="40" y1="55" x2="150" y2="80" stroke="#B8AEA2" stroke-width="2"/><line x1="40" y1="80" x2="150" y2="85" stroke="#B8AEA2" stroke-width="2"/><line x1="40" y1="108" x2="150" y2="92" stroke="#B8AEA2" stroke-width="2"/><circle cx="40" cy="55" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="40" cy="80" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="40" cy="108" r="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="22" y="59" text-anchor="end" fill="#6B645E">in</text><text x="22" y="84" text-anchor="end" fill="#6B645E">in</text><text x="22" y="112" text-anchor="end" fill="#6B645E">in</text><path d="M150 60 Q 195 40 235 70 Q 260 100 225 118 Q 175 128 152 100 Q 140 78 150 60 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="192" y="94" text-anchor="middle" fill="#276b45">neuron</text><line x1="238" y1="86" x2="320" y2="86" stroke="#2D8B55" stroke-width="2.5"/><polygon points="320,80 334,86 320,92" fill="#2D8B55"/><text class="lang-en" x="345" y="90" fill="#276b45">one number out →</text><text class="lang-zh" x="345" y="90" fill="#276b45">一个数字出去 →</text><rect x="60" y="150" width="90" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="105" y="169" text-anchor="middle" fill="#8A6D3B">1 · weigh</text><text class="lang-zh" x="105" y="169" text-anchor="middle" fill="#8A6D3B">1 · 称一称</text><text x="163" y="169" fill="#B8AEA2">→</text><rect x="185" y="150" width="90" height="30" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text class="lang-en" x="230" y="169" text-anchor="middle" fill="#5E5191">2 · add</text><text class="lang-zh" x="230" y="169" text-anchor="middle" fill="#5E5191">2 · 加起来</text><text x="288" y="169" fill="#B8AEA2">→</text><rect x="310" y="150" width="90" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="355" y="169" text-anchor="middle" fill="#276b45">3 · decide</text><text class="lang-zh" x="355" y="169" text-anchor="middle" fill="#276b45">3 · 做决定</text><text class="lang-en" x="260" y="196" text-anchor="middle" fill="#6B645E" font-size="10">every neuron, big model or small, is just these three steps</text><text class="lang-zh" x="260" y="196" text-anchor="middle" fill="#6B645E" font-size="10">不管模型大小，每个 neuron 都只是这三步</text></g></svg>
%%%

**The picture is honest about the shape** — many signals in, each trusted differently, summed, one signal out. **It stops being true about the biology:** a real brain cell is alive and wildly complicated, while an artificial neuron is a pinch of multiply-and-add. So: *inspired by*, never "the same as." (First myth busted; we'll collect a few more at the end.)

#### The three beats, one line each
These three names are the skeleton of the entire day, so let's walk them slowly.

%%% steps
step: weigh
why: each incoming number gets multiplied by a number that says how much it matters — that multiplier is a *weight*, the arm the brain cell trusts (or doesn't)
step: add
why: pile all the weighed numbers into one running total, therefore many opinions become one score — plus a little starting nudge called the *bias*
step: decide
why: pass that one total through a final rule that turns it into the neuron's answer, which is the moment the cell "fires" or stays quiet
%%%

%%% insight
Here's why those three little words are worth memorising: they never change. Not for a bigger neuron, not for a deeper network, not for the model writing your emails. Learn **weigh → add → decide** today and you have the shape of *every* neuron you will ever meet — the rest of this module just fills in each beat with detail.
%%%

So far: a neuron is a tiny **brain cell** that likes to *weigh → add → decide*. That's the shape to hold. Now let's meet each beat as its own unit — starting with the trust dials.
~~~zh
**这张图对形状是诚实的** —— 很多信号进来，每个被相信的程度不一样，加在一起，一个信号出去。**它对生物学就不成立了：** 真的脑细胞是活的，复杂得离谱，而一个人造 neuron 只是一小撮乘法和加法。所以：*受它启发*，绝不是「和它一样」。（第一个迷思拆掉了；结尾我们再收几个。）

#### 三个拍子，一句话一个
这三个名字是一整天的骨架，所以我们慢慢走一遍。

%%% steps
step: 称一称
why: 每个进来的数字都要乘上一个数，那个数说它有多重要 —— 那个乘数就是一个 *weight*，也就是脑细胞相信（或者不相信）的那条手臂
step: 加起来
why: 把所有称过的数字堆成一个总数，于是很多意见变成一个分数 —— 再加上一个小小的起跑推力，叫 *bias*
step: 做决定
why: 把那一个总数送进最后一条规则，变成 neuron 的答案，这就是细胞「发射」还是保持安静的那一刻
%%%

%%% insight
这三个小词值得背下来，原因在这里：它们从不改变。换成更大的 neuron 不变，换成更深的网络不变，换成替你写邮件的那个模型也不变。今天学会**称一称 → 加起来 → 做决定**，你就拿到了你以后会遇到的*每一个* neuron 的形状 —— 这个模块剩下的部分，只是给每个拍子填上细节。
%%%

到这里：一个 neuron 就是一个小小的**脑细胞**，喜欢*称一称 → 加起来 → 做决定*。这就是要记住的形状。现在我们把每个拍子当成单独一个单元来见 —— 先从那几个信任旋钮开始。
~~~

@@@ concept id=c2 zh_tag="Weight" tag="Weights" zh_title="Weight —— 你有多相信每个输入" title="Weights — how much you trust each input" zh_gotit="懂了 weight" gotit="Got weights"
Time for the first beat: **weigh**. Back to the movie night.

Your friend with amazing taste? You turn their "trust dial" way up. The friend who hates everything? You turn theirs way down — maybe even into the negatives, so their thumbs-up actually *pushes you away*. Each friend gets their own dial, and it decides how loudly that friend's opinion lands in your head.

A neuron does the same. Every input arrives with its own [[weight||One number per input that scales how much that input matters. Big weight = "listen closely"; near-zero = "ignore"; negative = "count this against."]] — one number that says how much to trust it.
~~~zh
第一个拍子来了：**称一称**。回到电影之夜。

那个品味特别好的朋友？你把他的「信任旋钮」拧得很高。那个什么都讨厌的朋友？你把他的拧得很低 —— 甚至拧进负数里，于是他点赞反而*把你推开*。每个朋友都有自己的旋钮，它决定这个朋友的意见在你脑子里砸下来有多响。

一个 neuron 做的一样。每个输入进来时都自带一个 [[weight||每个输入一个数字，缩放这个输入有多重要。weight 大 = 「仔细听」；接近零 = 「别管它」；负的 = 「把这个算成反对」。]] —— 一个数字，说明该有多相信它。
~~~

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="Three trust dials, one per friend. Each dial is a gauge: the needle points straight up at zero, leans right for a positive weight and left for a negative one. Friend one's needle leans well to the right, weight 0.9, strong trust. Friend two's leans slightly right, weight 0.5, half trust. Friend three's leans left onto the negative side, weight minus 0.6, counting against."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A weight = a trust dial, one per input (straight up = 0)</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一个 weight = 一个信任旋钮，每个输入一个（指正上方 = 0）</text><g><circle cx="110" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><path d="M76 80 A 34 34 0 0 1 110 46" fill="none" stroke="#C93B3B" stroke-width="3" opacity="0.55"/><path d="M110 46 A 34 34 0 0 1 144 80" fill="none" stroke="#2D8B55" stroke-width="3" opacity="0.55"/><text x="110" y="38" text-anchor="middle" fill="#9A938A" font-size="9">0</text><text x="64" y="84" text-anchor="middle" fill="#C93B3B" font-size="10">−</text><text x="156" y="84" text-anchor="middle" fill="#276b45" font-size="10">+</text><line x1="110" y1="80" x2="128" y2="59" stroke="#2D8B55" stroke-width="3"/><circle cx="110" cy="80" r="4" fill="#2D8B55"/><text class="lang-en" x="110" y="132" text-anchor="middle" fill="#6B645E">good-taste friend</text><text class="lang-zh" x="110" y="132" text-anchor="middle" fill="#6B645E">品味好的朋友</text><text class="lang-en" x="110" y="148" text-anchor="middle" fill="#276b45" font-weight="bold">weight = 0.9 (high)</text><text class="lang-zh" x="110" y="148" text-anchor="middle" fill="#276b45" font-weight="bold">weight = 0.9（高）</text></g><g><circle cx="260" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><path d="M226 80 A 34 34 0 0 1 260 46" fill="none" stroke="#C93B3B" stroke-width="3" opacity="0.55"/><path d="M260 46 A 34 34 0 0 1 294 80" fill="none" stroke="#2D8B55" stroke-width="3" opacity="0.55"/><text x="260" y="38" text-anchor="middle" fill="#9A938A" font-size="9">0</text><text x="214" y="84" text-anchor="middle" fill="#C93B3B" font-size="10">−</text><text x="306" y="84" text-anchor="middle" fill="#276b45" font-size="10">+</text><line x1="260" y1="80" x2="270" y2="54" stroke="#7C6DAA" stroke-width="3"/><circle cx="260" cy="80" r="4" fill="#7C6DAA"/><text class="lang-en" x="260" y="132" text-anchor="middle" fill="#6B645E">half-listened friend</text><text class="lang-zh" x="260" y="132" text-anchor="middle" fill="#6B645E">只听一半的朋友</text><text class="lang-en" x="260" y="148" text-anchor="middle" fill="#5E5191" font-weight="bold">weight = 0.5 (half)</text><text class="lang-zh" x="260" y="148" text-anchor="middle" fill="#5E5191" font-weight="bold">weight = 0.5（一半）</text></g><g><circle cx="410" cy="80" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><path d="M376 80 A 34 34 0 0 1 410 46" fill="none" stroke="#C93B3B" stroke-width="3" opacity="0.55"/><path d="M410 46 A 34 34 0 0 1 444 80" fill="none" stroke="#2D8B55" stroke-width="3" opacity="0.55"/><text x="410" y="38" text-anchor="middle" fill="#9A938A" font-size="9">0</text><text x="364" y="84" text-anchor="middle" fill="#C93B3B" font-size="10">−</text><text x="456" y="84" text-anchor="middle" fill="#276b45" font-size="10">+</text><line x1="410" y1="80" x2="397" y2="55" stroke="#C93B3B" stroke-width="3"/><circle cx="410" cy="80" r="4" fill="#C93B3B"/><text class="lang-en" x="410" y="132" text-anchor="middle" fill="#6B645E">hates-everything friend</text><text class="lang-zh" x="410" y="132" text-anchor="middle" fill="#6B645E">什么都讨厌的朋友</text><text class="lang-en" x="410" y="148" text-anchor="middle" fill="#C93B3B" font-weight="bold">weight = −0.6 (against)</text><text class="lang-zh" x="410" y="148" text-anchor="middle" fill="#C93B3B" font-weight="bold">weight = −0.6（反对）</text></g><text class="lang-en" x="260" y="180" text-anchor="middle" fill="#6B645E" font-size="10">input × weight = how loudly that input lands</text><text class="lang-zh" x="260" y="180" text-anchor="middle" fill="#6B645E" font-size="10">输入 × weight = 这个输入砸下来有多响</text></g></svg>
%%%

**The dial gets it right:** each input has its own setting that scales its say, exactly like turning a knob. **One catch:** you set friend-dials once and forget them, but a neuron's weights are *meant* to be tuned over and over until it stops making mistakes — that tuning is called **training**, and it's what Days 5–6 are for. Today the dials just sit at numbers we hand them.

#### Reading a dial: three settings worth knowing
Stay with the dials for a second, because their whole vocabulary is three positions.

%%% steps
step: dial turned up (weight 0.9)
why: "listen closely" — this input shouts, therefore it dominates the total
step: dial near zero (weight 0.05)
why: "basically ignore you" — the input is still there, but it barely moves the score
step: dial into the negative (weight −0.6)
why: "count this *against*" — which means a bigger input now pushes the answer toward NO, the hates-everything friend in numbers
%%%

%%% insight
This is the small part that turns out to be very powerful. A weight is not just "importance" — its **sign** flips the *meaning* of an input. Same input, positive dial: evidence for. Same input, negative dial: evidence against. One number, two jobs: *how much*, and *which way*.

You already lean on this daily: your spam filter turns its dial on "the words FREE MONEY appear" way up, and its dial on "the sender is in your contacts" *negative* — that input is evidence **against** spam, so it drags the score back down. If "negative weight" still feels slippery, that's normal; it clicks the moment you watch it multiply, which is right now.
%%%

#### See a weight do its one job
A weight has exactly one job: multiply. Let's watch it, with your guess first.

%%% demo id=weigh label="predict, then run"
predict: input 2 with weight 0.5, and input 3 with weight −1.0 — which one comes out negative, and how big?
code: inputs = np.array([2, 3]); weights = np.array([0.5, -1.0]); inputs * weights
out: array([ 1., -3.])
take: <b>Each input is scaled by its own weight.</b> The first input <code>2</code> with weight <code>0.5</code> becomes <code>1</code> — halved, "half-trusted." The second input <code>3</code> with weight <code>−1.0</code> becomes <code>−3</code> — flipped negative, "counted against." That's the whole "weigh" beat: one multiply per input.
%%%

So a weight is a trust dial made of arithmetic. Turn it up, the input shouts; toward zero, it whispers; negative, it argues the other way.

**You just learned the first of a neuron's two sets of numbers.** The second is one lonely number called the bias, and it's next.
~~~zh
**这个旋钮比得对：** 每个输入都有自己的一档设置，缩放它说话的分量，就像拧一个旋钮。**有一个地方要注意：** 朋友的旋钮你拧一次就忘了。而一个 neuron 的 weight *本来就是*要被反复调的，一直调到它不再犯错。这个调的过程叫做 **training（训练）**，是第 5 到第 6 天的事。今天这些旋钮就停在我们递给它的数字上。

#### 读一个旋钮：三档值得知道
先在旋钮上多待一会儿，因为它全部的词汇就是三个位置。

%%% steps
step: 旋钮拧上去（weight 0.9）
why: 「仔细听」—— 这个输入在喊，因此它主导了总数
step: 旋钮接近零（weight 0.05）
why: 「基本上不管你」—— 输入还在那里，但它几乎推不动这个分数
step: 旋钮拧进负数（weight −0.6）
why: 「把这个算成*反对*」—— 也就是说，输入越大，答案越被推向「不是」，那个什么都讨厌的朋友变成了数字
%%%

%%% insight
这一小块，最后会非常有力量。一个 weight 不只是「有多重要」—— 它的**正负号**会翻转一个输入的*意思*。同一个输入，旋钮是正的：支持的证据。同一个输入，旋钮是负的：反对的证据。一个数字，两份工作：*多少*，和*哪个方向*。

你每天已经在用这件事了。你的垃圾邮件过滤器把「出现『免费送钱』这几个字」这个旋钮拧得很高。它把「寄件人在你的联系人里」这个旋钮拧成*负的*。那个输入是**反对**垃圾邮件的证据，所以它把分数往下拽。如果「负的 weight」听起来还是滑溜溜抓不住，那很正常。你看它乘一次就懂了 —— 而那就是现在。
%%%

#### 看一个 weight 做它唯一的工作
一个 weight 只有一个工作：乘。我们来看它做，你先猜。

%%% demo id=weighzh label="先猜，再展开"
predict: 输入 2 配 weight 0.5，输入 3 配 weight −1.0 —— 哪一个出来是负的，有多大？
code: inputs = np.array([2, 3]); weights = np.array([0.5, -1.0]); inputs * weights
out: array([ 1., -3.])
take: <b>每个输入都被自己的 weight 缩放。</b>第一个输入 <code>2</code> 配 weight <code>0.5</code> 变成 <code>1</code> —— 砍掉一半，「只信一半」。第二个输入 <code>3</code> 配 weight <code>−1.0</code> 变成 <code>−3</code> —— 翻成负的，「算成反对」。这就是整个「称一称」拍子：每个输入乘一次。
%%%

所以一个 weight 就是用算术做的信任旋钮。拧上去，输入在喊；拧向零，它在小声说；拧成负的，它站到另一边去争。

**你刚学完 neuron 两套数字里的第一套。** 第二套是一个孤零零的数字，叫 bias，就在下面。
~~~

@@@ concept id=c3 zh_tag="Bias" tag="Bias" zh_title="Bias —— neuron 出门时的心情" title="Bias — the neuron's starting mood" zh_gotit="懂了 bias" gotit="Got bias"
Here's a feeling you know well. Some mornings you wake up already cheerful — it takes almost nothing to tip you into a "yes." Other mornings you wake up grumpy, and even good news barely moves you.

That built-in starting mood, before *anything* happens today, is the everyday picture of a [[bias||A single extra number added to the neuron's sum. It shifts the neuron's starting point — how easily it leans toward a "yes" before any input arrives.]]. A neuron carries one — and only one.
~~~zh
有一种感觉你很熟。有些早上你醒过来就已经很开心 —— 几乎不用什么，你就能倒向一个「好」。有些早上你醒过来就烦，连好消息都几乎推不动你。

这种自带的、今天*什么都还没发生*就有的起始心情，就是 [[bias||加到 neuron 那个和上面的一个额外数字。它挪动 neuron 的起点 —— 在任何输入到来之前，它有多容易倒向「是」。]] 的日常样子。一个 neuron 带着一个 —— 而且只有一个。
~~~

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Two runners at the start of a race. One has a head start ahead of the line (a positive bias, an easy yes). One starts behind the line (a negative bias, a hard yes). The bias just shifts where the neuron begins before any input."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Bias = a head start (or a handicap) before the race begins</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Bias = 比赛开始前提前一点（或者吃亏一点）</text><line x1="260" y1="40" x2="260" y2="150" stroke="#B8AEA2" stroke-width="2" stroke-dasharray="4,4"/><text class="lang-en" x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">start line (0)</text><text class="lang-zh" x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">起跑线（0）</text><circle cx="360" cy="70" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="360" y="75" text-anchor="middle" font-size="13">🏃</text><line x1="274" y1="70" x2="346" y2="70" stroke="#2D8B55" stroke-width="2"/><polygon points="346,64 360,70 346,76" fill="#2D8B55"/><text class="lang-en" x="410" y="74" fill="#276b45">+bias: a head start</text><text class="lang-zh" x="410" y="74" fill="#276b45">+bias：提前起跑</text><text class="lang-en" x="410" y="88" fill="#276b45" font-size="10">easy "yes"</text><text class="lang-zh" x="410" y="88" fill="#276b45" font-size="10">「是」很容易</text><circle cx="150" cy="120" r="14" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="150" y="125" text-anchor="middle" font-size="13">🏃</text><line x1="246" y1="120" x2="174" y2="120" stroke="#C93B3B" stroke-width="2"/><polygon points="174,114 160,120 174,126" fill="#C93B3B"/><text class="lang-en" x="110" y="102" text-anchor="middle" fill="#C93B3B">−bias: starts behind</text><text class="lang-zh" x="110" y="102" text-anchor="middle" fill="#C93B3B">−bias：从线后面起跑</text><text class="lang-en" x="110" y="140" text-anchor="middle" fill="#C93B3B" font-size="10">hard "yes"</text><text class="lang-zh" x="110" y="140" text-anchor="middle" fill="#C93B3B" font-size="10">「是」很难</text></g></svg>
%%%

**The head start gets it right:** the bias really does move where the neuron *begins* — how far it leans before hearing a single input. **One catch:** your mood drifts by itself all day; a neuron's bias is a fixed number that only moves during training (Days 5–6).

#### One bias, not one per input
Keep the runner in mind and this all stays simple.

%%% steps
step: the head start is added *once*, at the end
why: the neuron weighs and adds all its inputs first, then drops the bias on top — one nudge for the whole race, not one per runner
step: a big positive bias = the cheerful morning
why: the neuron already leans "yes," therefore even weak input tips it over
step: a big negative bias = the grumpy morning
why: the neuron starts behind the line, which means the inputs must work hard to win it over
%%%

%%% insight
Here's the freedom this one extra number buys: the neuron gets to hold an **opinion before the evidence arrives**. It can wake up already leaning yes, or already leaning no, and make the inputs argue it out of that lean. Your spam filter uses exactly that — it leans **NO** before reading a word, because most of your mail isn't spam.

Take the bias away and that freedom goes: the neuron is pinned to "start at zero," so it can only ever say yes when the **weighed** inputs add up past zero. (Weighed, not raw — remember the friend whose dial is negative can drag a big cheerful input *down*.) Real questions are rarely balanced on zero like that. Small number, big freedom.
%%%

#### Watch the bias shift the total
The weighed inputs from last unit summed to `1 + (−3) = −2`. Now add a bias of `1`.

%%% demo id=bias label="predict, then run"
predict: total was −2, and the head start is +1 — does the answer land at −1, or at −3?
code: weighed = np.array([1.0, -3.0]); bias = 1.0; weighed.sum() + bias
out: -1.0
take: <b>Bias is added once, on top of the summed inputs.</b> The inputs summed to <code>−2</code>; the bias <code>+1</code> nudged the total up to <code>−1</code>. A bigger bias would push it toward "yes"; a negative bias would drag it toward "no." One number, added last.
%%%

So far: weights say *how much each input matters*; the bias says *where the neuron starts from*. Two kinds of number, and that's the neuron's entire memory. Now watch them team up into the neuron's one real calculation.
~~~zh
**提前起跑这个比喻对在哪里：** bias 真的挪动了 neuron *开始*的地方 —— 它在听到任何一个输入之前，已经倒向多少。**有一个地方要注意：** 你的心情一整天自己会飘；而一个 neuron 的 bias 是一个固定的数字，只在 training 的时候才动（第 5 到第 6 天）。

#### 一个 bias，不是每个输入一个
把那个跑步的人放在心里，这一切就都简单。

%%% steps
step: 提前的那一点是*一次性*加上的，加在最后
why: neuron 先把所有输入称一称、加起来，然后才把 bias 放在上面 —— 整场比赛推一下，不是每个跑者推一下
step: 一个大的正 bias = 那个开心的早上
why: neuron 已经倒向「是」，因此就算输入很弱也能把它推倒
step: 一个大的负 bias = 那个烦躁的早上
why: neuron 从线后面起跑，也就是说输入必须很努力才能说服它
%%%

%%% insight
这一个额外数字买来的自由在这里：neuron 可以**在证据来之前就先有一个看法**。它可以醒过来就已经倒向「是」，或者已经倒向「不是」，然后让输入把它从这个倒向里争出来。你的垃圾邮件过滤器用的正是这个 —— 它一个字都还没读就先倒向**不是**，因为你的邮件大部分不是垃圾邮件。

把 bias 拿掉，这份自由就没了：neuron 被钉在「从零开始」上，于是它只能在**称过的**输入加起来超过零的时候说是。（是称过的，不是原始的 —— 记得那个旋钮是负的朋友，能把一个很大的开心输入往*下*拽。）真实的问题很少这么正好平衡在零上。小小的数字，大大的自由。
%%%

#### 看 bias 把总数挪一挪
上一个单元里称过的输入加起来是 `1 + (−3) = −2`。现在加上一个 `1` 的 bias。

%%% demo id=biaszh label="先猜，再展开"
predict: 总数是 −2，提前的那一点是 +1 —— 答案会落在 −1，还是 −3？
code: weighed = np.array([1.0, -3.0]); bias = 1.0; weighed.sum() + bias
out: -1.0
take: <b>Bias 只加一次，加在输入的和上面。</b>输入加起来是 <code>−2</code>；bias <code>+1</code> 把总数推上去到 <code>−1</code>。更大的 bias 会把它推向「是」；一个负的 bias 会把它往「不是」拽。一个数字，最后加上。
%%%

到这里：weight 说的是*每个输入有多重要*；bias 说的是 *neuron 从哪里开始*。两种数字，而这就是 neuron 全部的记忆。现在看它们联手，做出 neuron 唯一那一次真正的计算。
~~~

@@@ concept id=c4 zh_tag="那个和" tag="The sum" zh_title="全都加起来 —— neuron 的那一个数字" title="Adding it all up — the neuron's one number" zh_gotit="懂了那个和" gotit="Got the sum"
Now the "add" beat, where weights and bias finally work together.

Picture a **shopping cart at the till**. Each item's price gets scanned, the register keeps a running total, and at the end the cashier adds one flat service fee on top. One final number on the receipt: your total.

A neuron builds its answer the exact same way, and that number has a name: the [[weighted sum||The running total: every input times its weight, all added up, plus the bias. Also called the pre-activation, because it comes just before the final "decide" step. Written z.]] — often called the **pre-activation**, because it comes right *before* the final "decide" step. Engineers give it a short nickname: `z`.
~~~zh
现在是「加起来」这个拍子，weight 和 bias 终于一起干活了。

想象**收银台前的一车东西**。每样东西的价格被扫一下，收银机保持一个累计的总数，最后收银员在上面加一笔固定的服务费。收据上最后那一个数字：你要付的总额。

一个 neuron 搭出答案的方式完全一样，而那个数字有个名字：[[weighted sum（加权和）||那个累计的总数：每个输入乘它的 weight，全部加起来，再加上 bias。也叫 pre-activation，因为它就在最后那一步「做决定」之前。写成 z。]] —— 常常被叫做 **pre-activation**，因为它就在最后那一步「做决定」*之前*。工程师给它一个短外号：`z`。
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A shopping-cart receipt. Two scanned items each show price times a weight, they add into a running total, then a flat service fee (the bias) is added on the bottom line to give the final total z."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The weighted sum z = ring up each item, add the fee</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Weighted sum z = 每样都扫一遍，再加那笔费</text><rect x="150" y="30" width="220" height="150" rx="4" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="150" y1="52" x2="370" y2="52" stroke="#E5DFD6"/><text class="lang-en" x="260" y="47" text-anchor="middle" fill="#6B645E">— RECEIPT —</text><text class="lang-zh" x="260" y="47" text-anchor="middle" fill="#6B645E">—— 收据 ——</text><text x="166" y="74" fill="#5E5191">input₁ × w₁</text><text x="354" y="74" text-anchor="end" fill="#3A342E">1.0</text><text x="166" y="98" fill="#5E5191">input₂ × w₂</text><text x="354" y="98" text-anchor="end" fill="#3A342E">−3.0</text><line x1="166" y1="110" x2="354" y2="110" stroke="#B8AEA2"/><text class="lang-en" x="166" y="128" fill="#6B645E">running total</text><text class="lang-zh" x="166" y="128" fill="#6B645E">累计</text><text x="354" y="128" text-anchor="end" fill="#3A342E">−2.0</text><text class="lang-en" x="166" y="150" fill="#8A6D3B">+ bias (fee)</text><text class="lang-zh" x="166" y="150" fill="#8A6D3B">+ bias（那笔费）</text><text x="354" y="150" text-anchor="end" fill="#8A6D3B">+1.0</text><line x1="166" y1="160" x2="354" y2="160" stroke="#3A342E" stroke-width="1.5"/><text class="lang-en" x="166" y="176" fill="#2D8B55" font-weight="bold">TOTAL  z</text><text class="lang-zh" x="166" y="176" fill="#2D8B55" font-weight="bold">总计  z</text><text x="354" y="176" text-anchor="end" fill="#2D8B55" font-weight="bold">−1.0</text></g></svg>
%%%

**The receipt gets it right:** scan each item, keep a running total, add one fee at the end — that's the neuron's exact order. **One catch:** on a receipt every price is positive, but a weighed input can be *negative*, so an item can *subtract* from the total, which no real till ever does.

#### The one line of math, said out loud
%%% formula
expr: z = w₁·x₁ + w₂·x₂ + … + b
note: z is the total; each x is an input; each w is that input's weight; b is the one bias. In words: "weigh every input, add them up, then add the bias."
%%%

Read it left to right: `w₁·x₁` is "input one times its weight," you add on `w₂·x₂` for input two, keep going for every input, and finally add `b`, the bias. That's every symbol. Nothing hidden.

%%% insight
Notice how *small* that line is. Two multiplies and two adds — a calculator could do it. And yet: this exact line, copied across millions of tiny neurons, is what writes poetry and spots tumours. The power was never in the formula's cleverness; it's in how many copies get stacked and tuned. That's a genuinely surprising fact about AI, and you now know it first-hand.
%%%

#### Ring up the receipt, one line at a time
Let's build `z` the way the till does — one item, then the next, then the fee. Watch the running total change at each rung.

%%% steps
step: start the receipt at 0
why: nothing scanned yet, therefore the running total is empty — this is the till before you unload the cart
step: scan item 1 → 0.5 × 2 = 1.0 → total is 1.0
why: the first input times its own trust dial; a positive dial, so it *adds* to the total
step: scan item 2 → −1.0 × 3 = −3.0 → total drops to −2.0
why: the negative dial means this item *subtracts* — which is why the total goes down even though the input was a cheerful 3
step: add the fee → −2.0 + 1.0 = −1.0
why: the bias lands once, at the end; the head start lifts the total from −2.0 to −1.0
step: read the bottom line → z = −1.0
why: that single number is everything the neuron has to say, right before it decides
%%%

Same ladder, drawn to scale as bars — because a picture of a running total is far faster to read than a line of arithmetic. Every `1.0` of value is the same bar height, so you can read the sizes straight off:
~~~zh
**收据这个比喻对在哪里：** 每样东西扫一下，保持一个累计的总数，最后加一笔费 —— 这就是 neuron 的确切顺序。**有一个地方要注意：** 收据上每个价格都是正的。但一个称过的输入可以是*负的*，所以一样东西能从总数里*减掉*，真的收银机从来不这样。

#### 那一行数学，念出来

%%% formula
expr: z = w₁·x₁ + w₂·x₂ + … + b
note: z 是总数；每个 x 是一个输入；每个 w 是那个输入的 weight；b 是唯一那个 bias。用话说：「每个输入称一称，加起来，再加上 bias。」
%%%

从左往右读：`w₁·x₁` 是「第一个输入乘它的 weight」。再加上第二个输入的 `w₂·x₂`，每个输入都这么继续下去。最后加上 `b`，也就是 bias。每一个符号都在这里了。什么都没藏。

%%% insight
注意那一行有多*小*。两次乘法、两次加法 —— 一个计算器就能做。可是：正是这一行，被抄进几百万个小小的 neuron 里，写出了诗，也看出了肿瘤。力量从来不在这个公式有多聪明；它在于被叠起来、被调过的副本有多少。这是关于 AI 一件真正让人意外的事，而你现在亲手知道了。
%%%

#### 一行一行把收据打出来
我们像收银机那样搭出 `z` —— 一样东西，然后下一样，然后那笔费。看每一级上累计的总数怎么变。

%%% steps
step: 收据从 0 开始
why: 还什么都没扫，因此累计的总数是空的 —— 这是你把车里东西拿出来之前的收银机
step: 扫第 1 样 → 0.5 × 2 = 1.0 → 总数是 1.0
why: 第一个输入乘它自己的信任旋钮；旋钮是正的，所以它给总数*加*上
step: 扫第 2 样 → −1.0 × 3 = −3.0 → 总数掉到 −2.0
why: 旋钮是负的，意味着这一样在*减* —— 所以就算输入是个开心的 3，总数还是往下走
step: 加上那笔费 → −2.0 + 1.0 = −1.0
why: bias 只落一次，落在最后；那点提前把总数从 −2.0 抬到 −1.0
step: 读最下面那一行 → z = −1.0
why: 那一个数字就是 neuron 要说的全部，正好在它做决定之前
%%%

同一个梯子，按真实比例画成条形 —— 因为一张累计总数的图，读起来比一行算术快得多。每 `1.0` 的值都是同样的条形高度，所以你可以直接把大小读出来：
~~~

%%% svg
<svg viewBox="0 0 520 264" role="img" aria-label="A running-total bar chart drawn to scale, thirty pixels per unit, against a zero line with tick marks at plus one, zero, minus one, minus two and minus three. The first bar is plus one for the first weighed input, running total one. The second bar is minus three, three times as tall as the first and pointing down, dropping the running total to minus two. The third bar is plus one for the bias, lifting the total to minus one. The final short downward bar is z equals minus one."><g font-family="monospace" font-size="11" text-anchor="middle"><text class="lang-en" x="260" y="18" fill="#2C2A28" font-size="12">The total z builds up piece by piece (30 px = 1.0, everywhere)</text><text class="lang-zh" x="260" y="18" fill="#2C2A28" font-size="12">总数 z 一块一块搭起来（哪里都是 30 px = 1.0）</text><line x1="50" y1="90" x2="480" y2="90" stroke="#EFE9DF" stroke-width="1" stroke-dasharray="3,3"/><line x1="50" y1="150" x2="480" y2="150" stroke="#EFE9DF" stroke-width="1" stroke-dasharray="3,3"/><line x1="50" y1="180" x2="480" y2="180" stroke="#EFE9DF" stroke-width="1" stroke-dasharray="3,3"/><line x1="50" y1="210" x2="480" y2="210" stroke="#EFE9DF" stroke-width="1" stroke-dasharray="3,3"/><line x1="50" y1="120" x2="480" y2="120" stroke="#B8AEA2" stroke-width="1.5"/><text x="44" y="94" text-anchor="end" fill="#9A938A" font-size="9">+1</text><text x="44" y="124" text-anchor="end" fill="#9A938A" font-size="9">0</text><text x="44" y="154" text-anchor="end" fill="#9A938A" font-size="9">−1</text><text x="44" y="184" text-anchor="end" fill="#9A938A" font-size="9">−2</text><text x="44" y="214" text-anchor="end" fill="#9A938A" font-size="9">−3</text><rect x="70" y="90" width="70" height="30" fill="#7C6DAA" opacity="0.85"/><text x="105" y="110" fill="#fff">+1.0</text><text x="105" y="230" fill="#6B645E" font-size="10">w₁·x₁</text><text class="lang-en" x="105" y="244" fill="#5E5191" font-size="10">total: 1.0</text><text class="lang-zh" x="105" y="244" fill="#5E5191" font-size="10">累计：1.0</text><rect x="175" y="120" width="70" height="90" fill="#C93B3B" opacity="0.85"/><text x="210" y="172" fill="#fff">−3.0</text><text x="210" y="230" fill="#6B645E" font-size="10">w₂·x₂</text><text class="lang-en" x="210" y="244" fill="#C93B3B" font-size="10">total: −2.0</text><text class="lang-zh" x="210" y="244" fill="#C93B3B" font-size="10">累计：−2.0</text><rect x="280" y="90" width="70" height="30" fill="#C99A12" opacity="0.85"/><text x="315" y="110" fill="#fff">+1.0</text><text x="315" y="230" fill="#6B645E" font-size="10">bias b</text><text class="lang-en" x="315" y="244" fill="#8A6D3B" font-size="10">total: −1.0</text><text class="lang-zh" x="315" y="244" fill="#8A6D3B" font-size="10">累计：−1.0</text><rect x="390" y="120" width="70" height="30" fill="#2D8B55"/><text x="425" y="140" fill="#fff">z</text><text x="425" y="230" fill="#276b45" font-size="11" font-weight="bold">z = −1.0</text><text class="lang-en" x="260" y="260" fill="#9A938A" font-size="9">the −3.0 bar really is three times the +1.0 bar — that's why the total dips below zero</text><text class="lang-zh" x="260" y="260" fill="#9A938A" font-size="9">那根 −3.0 的条形真的是 +1.0 的三倍 —— 所以总数会掉到零以下</text></g></svg>
%%%

%%% demo id=zsum label="predict, then run"
predict: you rang it up by hand above — does the code agree that z lands at −1.0?
code: x = np.array([2, 3]); w = np.array([0.5, -1.0]); b = 1.0; z = (w * x).sum() + b; z
out: -1.0
take: <b>z = (w·x summed) + b = −1.0.</b> Two inputs weighed to <code>1.0</code> and <code>−3.0</code>, summed to <code>−2.0</code>, plus the bias <code>1.0</code>, lands at <code>z = −1.0</code>. Exactly the receipt you built by hand.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the same sum in one breath.</b> <code>0.5 × 2 = 1.0</code>; <code>−1.0 × 3 = −3.0</code>; <code>1.0 + (−3.0) = −2.0</code>; <code>−2.0 + 1.0 = −1.0</code>. Two multiplies, two adds. You'll type it yourself in <b>Produce</b>.
!!!

**You just did the only real arithmetic a neuron ever does.** If the bars made sense, you are past the hardest arithmetic of the day — everything left is what *happens* to that one number `z`.
~~~zh
%%% demo id=zsumzh label="先猜，再展开"
predict: 你上面已经手算打过一遍收据了 —— 代码同意 z 落在 −1.0 吗？
code: x = np.array([2, 3]); w = np.array([0.5, -1.0]); b = 1.0; z = (w * x).sum() + b; z
out: -1.0
take: <b>z =（w·x 加起来）+ b = −1.0。</b>两个输入称成 <code>1.0</code> 和 <code>−3.0</code>，加起来是 <code>−2.0</code>，加上 bias <code>1.0</code>，落在 <code>z = −1.0</code>。正是你手算搭出来的那张收据。
%%%

!!! c-info 🪜
<b>选读（可以跳过）—— 同一个和，一口气算完。</b> <code>0.5 × 2 = 1.0</code>；<code>−1.0 × 3 = −3.0</code>；<code>1.0 + (−3.0) = −2.0</code>；<code>−2.0 + 1.0 = −1.0</code>。两次乘法，两次加法。你会在 <b>动手做</b> 里自己敲一遍。
!!!

**你刚做完了 neuron 唯一真正会做的那点算术。** 如果那些条形讲得通，你已经过了今天最难的算术 —— 剩下的全部，都是那一个数字 `z` *身上发生*了什么。
~~~

@@@ concept id=c5 zh_tag="做决定" tag="Decide" zh_title="做决定这一步 —— 把 z 变成一个答案" title="The decide step — turning z into an answer" zh_gotit="懂了做决定" gotit="Got the decide step"
The neuron has its one number, `z`. But a raw score isn't yet an *answer*.

Picture a **judge at the end of a talent show**. The scores are in and totalled — that's `z` — and now the judge has to turn that total into a verdict the audience understands: "you're through" or "you're out," or maybe a confidence like "87%." Turning a total into a usable answer is the last beat, **decide**.

The step that does it is the [[activation function||The last step of a neuron: it takes the weighted sum z and turns it into the neuron's output. There's a whole family of them — a step, a smooth S-curve, and more.]]. Carry away this shape: a neuron is always the **linear part** (the weigh-and-add that gives `z`) *then* the **activation** (the decide step on top of `z`).
~~~zh
neuron 有了它那一个数字 `z`。但一个原始的分数还不是一个*答案*。

想象**才艺比赛最后的那位裁判**。分数都到了、也加完了 —— 那就是 `z`。现在裁判得把这个总数变成观众听得懂的判决：「你过了」或者「你出局」，也可能是一个信心值，比如「87%」。把一个总数变成一个能用的答案，就是最后那个拍子，**做决定**。

做这件事的那一步，叫 [[activation function（激活函数）||一个 neuron 的最后一步：它拿加权和 z，把它变成 neuron 的输出。它是一整个家族 —— 一个硬开关、一条平滑的 S 形曲线，还有更多。]]。把这个形状带走：一个 neuron 永远是**线性的那部分**（称一称、加起来，给出 `z`），*然后*才是 **activation**（在 `z` 上面做决定那一步）。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A talent-show judge. On the left, a scoreboard shows the total score z. An arrow leads to a judge who turns that score into a verdict: a yes or no, or a confidence percentage. The judge is the activation function."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The activation = a judge turning the score z into a verdict</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Activation = 一个把分数 z 变成判决的裁判</text><rect x="40" y="60" width="120" height="60" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="100" y="82" text-anchor="middle" fill="#5E5191" font-size="10">weighted sum</text><text class="lang-zh" x="100" y="82" text-anchor="middle" fill="#5E5191" font-size="10">加权和</text><text x="100" y="105" text-anchor="middle" fill="#3A342E" font-size="16" font-weight="bold">z = −1.0</text><text class="lang-en" x="100" y="140" text-anchor="middle" fill="#6B645E" font-size="10">the linear part</text><text class="lang-zh" x="100" y="140" text-anchor="middle" fill="#6B645E" font-size="10">线性的那部分</text><line x1="164" y1="90" x2="230" y2="90" stroke="#B8AEA2" stroke-width="2.5"/><polygon points="230,84 244,90 230,96" fill="#B8AEA2"/><circle cx="300" cy="90" r="34" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="300" y="86" text-anchor="middle" font-size="18">⚖️</text><text class="lang-en" x="300" y="104" text-anchor="middle" fill="#276b45" font-size="9">judge</text><text class="lang-zh" x="300" y="104" text-anchor="middle" fill="#276b45" font-size="9">裁判</text><text class="lang-en" x="300" y="150" text-anchor="middle" fill="#276b45" font-size="10">the activation</text><text class="lang-zh" x="300" y="150" text-anchor="middle" fill="#276b45" font-size="10">activation 这一步</text><line x1="336" y1="90" x2="400" y2="90" stroke="#B8AEA2" stroke-width="2.5"/><polygon points="400,84 414,90 400,96" fill="#B8AEA2"/><rect x="418" y="66" width="90" height="48" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text class="lang-en" x="463" y="86" text-anchor="middle" fill="#8A6D3B" font-size="10">verdict</text><text class="lang-zh" x="463" y="86" text-anchor="middle" fill="#8A6D3B" font-size="10">判决</text><text class="lang-en" x="463" y="104" text-anchor="middle" fill="#8A6D3B" font-size="10">yes / no / 87%</text><text class="lang-zh" x="463" y="104" text-anchor="middle" fill="#8A6D3B" font-size="10">是 / 不是 / 87%</text></g></svg>
%%%

**The judge gets it right:** a raw total is not a verdict, and something must *convert* one into the other — that conversion is the activation's whole job. **One catch:** a human judge has hunches; a neuron's activation is one fixed rule applied the same way every time.

#### Watch the judge build a verdict
Don't take "it turns `z` into an answer" on faith — let's watch the judge work, using the simplest possible rule (a hard yes/no at zero).

%%% steps
step: the score arrives — z = −1.0
why: straight off the receipt you built last unit; the judge invents nothing, it only reads what the sum handed over
step: the judge asks its one question — "is z ≥ 0?"
why: that's the entire rule, therefore the judge needs no opinion of its own
step: the answer here is *no*, because −1.0 sits below 0
why: this is the fork — flip z positive and this same question answers "yes" instead
step: read off the verdict → 0, meaning "off"
why: the no-answer forces a 0; a yes-answer would have forced a 1 ("on")
%%%
~~~zh
**裁判这个比喻对在哪里：** 一个原始的总数不是判决，必须有个东西把前者*变成*后者 —— 那个转换就是 activation 的全部工作。**有一个地方要注意：** 人类裁判有直觉；一个 neuron 的 activation 是一条固定的规则，每次都用同样的方式套上去。

#### 看裁判把一个判决搭出来
别光信「它把 `z` 变成一个答案」这句话 —— 我们看裁判干活，用最简单的那条规则（在零这里硬给是或不是）。

%%% steps
step: 分数来了 —— z = −1.0
why: 直接来自你上一个单元搭的那张收据；裁判不发明任何东西，它只读那个和递过来的数
step: 裁判问它唯一那个问题 —— 「z ≥ 0 吗？」
why: 这就是全部的规则，因此裁判不需要自己的看法
step: 这里的答案是*不*，因为 −1.0 坐在 0 下面
why: 这是那个分岔口 —— 把 z 翻成正的，同一个问题就答「是」了
step: 读出判决 → 0，意思是「关」
why: 「不」这个答案逼出一个 0；「是」会逼出一个 1（「开」）
%%%
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Three steps that build the verdict from a raw score. Step one: the raw score z equals minus one. Step two: ask is z at or above zero? Here the answer is no. Step three: the verdict is zero, meaning off. Each step feeds the next."><g font-family="monospace" font-size="11" text-anchor="middle"><text class="lang-en" x="260" y="18" fill="#2C2A28" font-size="12">z → ask one question → the verdict</text><text class="lang-zh" x="260" y="18" fill="#2C2A28" font-size="12">z → 问一个问题 → 判决</text><rect x="30" y="55" width="130" height="66" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text class="lang-en" x="95" y="76" fill="#6B645E" font-size="10">1 · the raw score</text><text class="lang-zh" x="95" y="76" fill="#6B645E" font-size="10">1 · 原始分数</text><text x="95" y="100" fill="#3A342E" font-size="16" font-weight="bold">z = −1.0</text><text class="lang-en" x="95" y="116" fill="#9A938A" font-size="9">(from weigh + add)</text><text class="lang-zh" x="95" y="116" fill="#9A938A" font-size="9">（来自称一称 + 加起来）</text><line x1="162" y1="88" x2="192" y2="88" stroke="#B8AEA2" stroke-width="2"/><polygon points="192,83 202,88 192,93" fill="#B8AEA2"/><rect x="205" y="55" width="130" height="66" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text class="lang-en" x="270" y="76" fill="#8A6D3B" font-size="10">2 · ask one thing</text><text class="lang-zh" x="270" y="76" fill="#8A6D3B" font-size="10">2 · 只问一件事</text><text x="270" y="98" fill="#3A342E" font-size="12">is z ≥ 0 ?</text><text x="270" y="116" fill="#C93B3B" font-size="11" font-weight="bold">no (−1.0 &lt; 0)</text><line x1="337" y1="88" x2="367" y2="88" stroke="#B8AEA2" stroke-width="2"/><polygon points="367,83 377,88 367,93" fill="#B8AEA2"/><rect x="380" y="55" width="120" height="66" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text class="lang-en" x="440" y="76" fill="#6B645E" font-size="10">3 · the verdict</text><text class="lang-zh" x="440" y="76" fill="#6B645E" font-size="10">3 · 判决</text><text class="lang-en" x="440" y="100" fill="#C93B3B" font-size="16" font-weight="bold">0 (off)</text><text class="lang-zh" x="440" y="100" fill="#C93B3B" font-size="16" font-weight="bold">0（关）</text><text class="lang-en" x="440" y="116" fill="#9A938A" font-size="9">the neuron's answer</text><text class="lang-zh" x="440" y="116" fill="#9A938A" font-size="9">neuron 的答案</text><text class="lang-en" x="260" y="150" fill="#6B645E" font-size="10">raw score → one question → answer: that's the whole "decide" beat</text><text class="lang-zh" x="260" y="150" fill="#6B645E" font-size="10">原始分数 → 一个问题 → 答案：这就是整个「做决定」拍子</text><text class="lang-en" x="260" y="166" fill="#9A938A" font-size="9">a positive z would flip step 2 to "yes" and the verdict to 1 (on)</text><text class="lang-zh" x="260" y="166" fill="#9A938A" font-size="9">一个正的 z 会把第 2 步翻成「是」，判决变成 1（开）</text></g></svg>
%%%

%%% insight
This little three-box picture is your defence against the biggest myth in AI. People say neural nets are "magic black boxes." But look: the answer was *assembled* — a score, one question, a verdict. No magic anywhere. Every answer ChatGPT gives you is built from millions of steps exactly this plain. Knowing that is genuinely a superpower in a room full of people who don't.
%%%

#### It's a whole family, not one rule
Here's the fun part: there isn't *one* activation. There's a family, and choosing between them is a real design decision.

- A blunt on/off "yes or no" — the one you just watched the judge use.
- A smooth "how confident, 0 to 100%."
- Several more you'll meet tomorrow.

So far: every neuron is *linear part, then activation*. Next up, the very first activation anyone ever tried — and the puzzle it left behind.
~~~zh
%%% insight
这张小小的三格图，是你对 AI 里最大那个迷思的防线。人们说神经网络是「有魔法的黑盒子」。但你看：那个答案是被*一步步搭出来*的 —— 一个分数、一个问题、一个判决。哪里都没有魔法。ChatGPT 给你的每一个答案，都是由几百万个正好这么朴素的步骤搭起来的。在一屋子不知道这件事的人里面，知道它真的是一种超能力。
%%%

#### 它是一整个家族，不是一条规则
好玩的地方来了：activation 不只*一个*。它是一个家族，而在它们之间挑选，是一个真实的设计决定。

- 一个直白的开/关「是或不是」—— 你刚看裁判用的那个。
- 一个平滑的「有多确定，0 到 100%」。
- 还有几个，明天见。

到这里：每个 neuron 都是*线性的那部分，然后 activation*。接下来是有人做出来的第一个 activation —— 以及它留下的那个难题。
~~~

@@@ concept id=c6 zh_tag="第一个 activation" tag="First activation" zh_title="Step function —— 一个硬邦邦的开关" title="The step function — a hard on/off switch" zh_gotit="见过 step 了" gotit="Met the step"
Let's meet the very first activation anyone ever built. It's the great-grandparent of the whole family, from the 1940s, and its idea is beautifully simple.

Picture the **height bar at a theme-park ride**: "you must be *this* tall to ride." A tall-enough rider clears the bar and gets waved on. A too-short rider is turned away at the gate. There's no "you're 80% on the ride" — a hard yes or no, decided by one fixed bar.

That is the [[step function||The original activation: if the weighted sum z is at or above a threshold (zero), output 1 (fire); otherwise output 0. A hard on/off switch, with nothing in between.]] — the "decide" rule inside the earliest artificial neurons (people called them the **McCulloch–Pitts neuron**, and later the **Perceptron**).
~~~zh
我们来见有人做出来的第一个 activation。它是整个家族的曾祖父，来自 1940 年代，而它的想法漂亮地简单。

想象**游乐园项目门口那根身高杆**：「你必须有*这么*高才能玩。」够高的乘客过了杆，被挥手放进去。太矮的乘客在门口被劝回。这里没有「你 80% 上了这个项目」—— 硬的是或者不是，由一根固定的杆决定。

那就是 [[step function||最早的那个 activation：如果加权和 z 到了或者超过一个门槛（零），就输出 1（发射）；否则输出 0。一个硬的开/关开关，中间什么都没有。]] —— 最早那些人造 neuron 里面「做决定」的规则（人们叫它 **McCulloch–Pitts neuron**，后来叫 **Perceptron（感知机）**）。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A theme-park height bar. On the left a short person is below the bar and is turned away (output 0, off). On the right a tall person clears the bar and is let on (output 1, on). It is a hard cutoff with nothing in between."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Step = a height bar: tall enough (on) or not (off)</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Step = 一根身高杆：够高（开）或者不够（关）</text><line x1="260" y1="45" x2="260" y2="150" stroke="#C99A12" stroke-width="3"/><text class="lang-en" x="260" y="42" text-anchor="middle" fill="#8A6D3B" font-size="10">the bar (z = 0)</text><text class="lang-zh" x="260" y="42" text-anchor="middle" fill="#8A6D3B" font-size="10">那根杆（z = 0）</text><rect x="120" y="105" width="16" height="45" fill="#C93B3B" opacity="0.8"/><circle cx="128" cy="98" r="9" fill="#C93B3B" opacity="0.8"/><text class="lang-en" x="128" y="164" text-anchor="middle" fill="#C93B3B" font-size="10">below the bar</text><text class="lang-zh" x="128" y="164" text-anchor="middle" fill="#C93B3B" font-size="10">在杆下面</text><text class="lang-en" x="60" y="90" fill="#C93B3B" font-weight="bold">z &lt; 0 → 0 (off)</text><text class="lang-zh" x="60" y="90" fill="#C93B3B" font-weight="bold">z &lt; 0 → 0（关）</text><rect x="384" y="70" width="16" height="80" fill="#2D8B55"/><circle cx="392" cy="62" r="9" fill="#2D8B55"/><text class="lang-en" x="392" y="164" text-anchor="middle" fill="#276b45" font-size="10">clears the bar</text><text class="lang-zh" x="392" y="164" text-anchor="middle" fill="#276b45" font-size="10">过了那根杆</text><text class="lang-en" x="420" y="90" text-anchor="end" fill="#276b45" font-weight="bold">z ≥ 0 → 1 (on)</text><text class="lang-zh" x="420" y="90" text-anchor="end" fill="#276b45" font-weight="bold">z ≥ 0 → 1（开）</text></g></svg>
%%%

**The height bar gets it right:** one fixed bar, two outcomes, no middle ground. **One catch:** at a real ride *you* stand at the bar, while here the neuron's own number `z` is what gets measured — and the bar never tells you *how far* over or under you were.

#### Line four riders up at the bar
Let's build the step one rider at a time. Each rider is a score `z`; the attendant only asks "did you reach the bar (z = 0)?"

%%% steps
step: rider z = −1.2 → waved away, 0
why: well below the bar, therefore an easy "no"
step: rider z = −0.3 → waved away, 0
why: closer, but still short — and the attendant gives no credit for being close
step: rider z = 0.1 → waved on, 1
why: *just* clears the bar, which means the verdict flips the instant the bar is reached
step: rider z = 2.5 → waved on, 1
why: towers over the bar, yet gets the *identical* answer as 0.1 — the bar never asks HOW tall
%%%
~~~zh
**身高杆这个比喻对在哪里：** 一根固定的杆，两种结果，没有中间地带。**有一个地方要注意：** 真的项目门口是*你*站在杆边，而这里被量的是 neuron 自己那个数字 `z` —— 而且那根杆从不告诉你，你超过或者差了*多少*。

#### 让四位乘客到杆前排队
我们一位乘客一位乘客地把 step 搭出来。每位乘客是一个分数 `z`；工作人员只问「你够到杆了吗（z = 0）？」

%%% steps
step: 乘客 z = −1.2 → 被劝回，0
why: 离杆还差很远，因此这是个轻松的「不」
step: 乘客 z = −0.3 → 被劝回，0
why: 近一些了，但还是差 —— 而工作人员不会因为你差得少就通融
step: 乘客 z = 0.1 → 放进去，1
why: *刚刚*过了杆，也就是说判决在够到杆的那一瞬间就翻了
step: 乘客 z = 2.5 → 放进去，1
why: 高出杆一大截，可是拿到和 0.1 *完全一样*的答案 —— 那根杆从不问有多高
%%%
~~~

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Four riders lined up by their score z at a theme-park height bar. The bar sits at z equals zero. A rider with z of minus one point two is below the bar and turned away, output zero. A rider with z of minus zero point three is still below the bar, output zero. A rider with z of zero point one just clears the bar, output one. A rider with z of two point five clears it easily, output one. The verdict flips from zero to one the moment a rider reaches the bar."><g font-family="monospace" font-size="11" text-anchor="middle"><text class="lang-en" x="260" y="16" fill="#2C2A28" font-size="12">Build the step: each rider meets the bar, gets waved on (1) or away (0)</text><text class="lang-zh" x="260" y="16" fill="#2C2A28" font-size="12">一步步搭出 step：每位乘客对一下杆，被放进（1）或者赶走（0）</text><line x1="270" y1="34" x2="270" y2="150" stroke="#C99A12" stroke-width="3"/><text class="lang-en" x="270" y="30" fill="#8A6D3B" font-size="9">the bar (z = 0)</text><text class="lang-zh" x="270" y="30" fill="#8A6D3B" font-size="9">那根杆（z = 0）</text><text class="lang-en" x="120" y="30" fill="#C93B3B" font-size="9">below → away (0)</text><text class="lang-zh" x="120" y="30" fill="#C93B3B" font-size="9">在下面 → 赶走（0）</text><text class="lang-en" x="420" y="30" fill="#276b45" font-size="9">at/above → on (1)</text><text class="lang-zh" x="420" y="30" fill="#276b45" font-size="9">到了/更高 → 放进（1）</text><line x1="40" y1="150" x2="500" y2="150" stroke="#E5DFD6" stroke-width="1.5"/><g><circle cx="90" cy="128" r="8" fill="#C93B3B" opacity="0.85"/><rect x="86" y="136" width="8" height="14" fill="#C93B3B" opacity="0.85"/><text x="90" y="172" fill="#6B645E" font-size="9">z = −1.2</text><text x="90" y="186" fill="#C93B3B" font-size="10" font-weight="bold">→ 0</text></g><g><circle cx="200" cy="128" r="8" fill="#C93B3B" opacity="0.85"/><rect x="196" y="136" width="8" height="14" fill="#C93B3B" opacity="0.85"/><text x="200" y="172" fill="#6B645E" font-size="9">z = −0.3</text><text x="200" y="186" fill="#C93B3B" font-size="10" font-weight="bold">→ 0</text></g><g><circle cx="330" cy="120" r="8" fill="#2D8B55"/><rect x="326" y="128" width="8" height="22" fill="#2D8B55"/><text x="330" y="172" fill="#6B645E" font-size="9">z = 0.1</text><text x="330" y="186" fill="#276b45" font-size="10" font-weight="bold">→ 1</text></g><g><circle cx="450" cy="112" r="8" fill="#2D8B55"/><rect x="446" y="120" width="8" height="30" fill="#2D8B55"/><text x="450" y="172" fill="#6B645E" font-size="9">z = 2.5</text><text x="450" y="186" fill="#276b45" font-size="10" font-weight="bold">→ 1</text></g><text class="lang-en" x="260" y="204" fill="#9A938A" font-size="9">notice: 0.1 and 2.5 both just say "on" — the bar never asks HOW tall, only "tall enough?"</text><text class="lang-zh" x="260" y="204" fill="#9A938A" font-size="9">注意：0.1 和 2.5 都只说「开」—— 那根杆从不问有多高，只问「够不够高」</text></g></svg>
%%%

Draw that snap as a curve and you get the step's famous shape — flat at `0`, a sudden jump at the bar, flat at `1`:
~~~zh
把那个突变画成一条曲线，你就得到 step 那个有名的形状 —— 在 `0` 上是平的，在杆那里突然一跳，在 `1` 上又是平的：
~~~

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Step function curve: output 0 for negative inputs, a sudden jump up to 1 at zero, then flat at 1 for positive inputs."><g font-family="monospace"><line x1="60" y1="100" x2="340" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><line x1="200" y1="30" x2="200" y2="118" stroke="#E5DFD6" stroke-width="1"/><path d="M70 100 L200 100 L200 48 L320 48" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="200" cy="48" r="3.5" fill="#8A6D3B"/><text class="lang-en" x="130" y="117" font-size="11" fill="#6B645E" text-anchor="middle">output 0 (away)</text><text class="lang-zh" x="130" y="117" font-size="11" fill="#6B645E" text-anchor="middle">输出 0（赶走）</text><text class="lang-en" x="265" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">output 1 (on)</text><text class="lang-zh" x="265" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">输出 1（放进）</text><text class="lang-en" x="200" y="133" font-size="10" fill="#9A938A" text-anchor="middle">the same snap, drawn: a cliff at the bar — no in-between</text><text class="lang-zh" x="200" y="133" font-size="10" fill="#9A938A" text-anchor="middle">同一个突变，画出来：杆那里是一道悬崖 —— 没有中间</text><text x="420" y="76" font-size="12" fill="#8A6D3B">step → {0, 1}</text></g></svg>
%%%

%%% demo id=step label="predict, then run"
predict: same four riders. Will 0.1 and 2.5 get *different* answers, or the same one?
code: step(np.array([-1.2, -0.3, 0.1, 2.5]))
out: array([0., 0., 1., 1.])
take: <b>Step = 1 if z ≥ 0, else 0.</b> The two below the bar become <code>0</code>, the two at or above it become <code>1</code>. Notice <code>0.1</code> and <code>2.5</code> give the identical answer — the bar never asks "how tall," only "tall enough?"
%%%

%%% insight
And *that* is the puzzle the step leaves behind. Improving a neuron means nudging its dials a hair and checking whether the answer got a little better. But the height bar is a **cliff**: nudge a rider by a millimetre and nothing budges — until they tumble over the edge and it flips completely. No small change, no hint, no direction to improve in.

For *one* neuron there was a clever way around it: Rosenblatt's **Perceptron** (1957) touched its dials only when it got an example *wrong*, nudging them toward the right answer. It worked — the first neuron that learned its own weights. But stack neurons (the thing that makes networks powerful) and "which way do I nudge a dial in the *middle* of the stack?" has no answer at all, because a cliff hands back no direction. Swapping the cliff for a gentle slope is what makes a whole stack improvable — the machinery of Day 5.
%%%

You just diagnosed a real design problem, on Day 1. **The fix is beautifully simple:** swap the cliff for a ramp — and that's next.
~~~zh
%%% demo id=stepzh label="先猜，再展开"
predict: 还是那四位乘客。0.1 和 2.5 会拿到*不同*的答案，还是同一个？
code: step(np.array([-1.2, -0.3, 0.1, 2.5]))
out: array([0., 0., 1., 1.])
take: <b>Step = z ≥ 0 就给 1，否则给 0。</b>杆下面那两位变成 <code>0</code>，到了或者超过杆的那两位变成 <code>1</code>。注意 <code>0.1</code> 和 <code>2.5</code> 给出完全一样的答案 —— 那根杆从不问「有多高」，只问「够不够高」。
%%%

%%% insight
而*那个*就是 step 留下的难题。要让一个 neuron 变好，意思是把它的旋钮拧一丝，再看答案有没有好一点。但那根身高杆是一道**悬崖**：把一位乘客挪一毫米，什么都不动 —— 直到他翻过边缘，然后整个翻转。没有小变化，没有提示，没有可以改进的方向。

对*一个* neuron，当年有一个聪明的绕法：Rosenblatt 的 **Perceptron**（1957）只在答错一个样本的时候才碰它的旋钮，把它们往正确答案那边推一点。它管用 —— 第一个自己学出 weight 的 neuron。但把 neuron 叠起来（正是这件事让网络有力量），「叠在*中间*的那个旋钮该往哪边拧？」就完全没有答案了，因为一道悬崖不会给你回一个方向。把悬崖换成一道缓坡，才让一整叠都能被改进 —— 那是第 5 天的机器。
%%%

你刚在第 1 天诊断出了一个真实的设计问题。**解法漂亮地简单：** 把悬崖换成一道斜坡 —— 就在下面。
~~~

@@@ concept id=c7 zh_tag="平滑的版本" tag="Smooth version" zh_title="Sigmoid —— 一个平滑的调光器，不是悬崖" title="Sigmoid — a smooth dimmer, not a cliff" zh_gotit="见过 sigmoid 了" gotit="Met sigmoid"
So we want the "decide" step to change *gradually*, so a small nudge to the dials makes a small, visible change in the answer.

Swap the light *switch* for a **dimmer**. A dimmer slides smoothly from off to fully on, passing through every level of brightness in between — dim, medium, bright. Nudge it a hair and the light changes a hair.

That smooth dimmer is the [[sigmoid||A smooth S-shaped activation that squashes any number into the range 0 to 1. It replaces the step's hard cliff with a gentle slide, so a small change in z makes a small change in the output.]] activation: it takes any score `z` and squashes it into the range 0 to 1 — read it as "how confident, from 0% to 100%."
~~~zh
所以我们想让「做决定」这一步*一点一点*地变，这样把旋钮拧一丝，答案就有一个小小的、看得见的变化。

把电灯**开关**换成一个**调光器**。调光器从关平滑地滑到全开，中间每一档亮度都经过 —— 暗、中等、亮。轻轻拧一丝，灯就变一丝。

那个平滑的调光器，就是 [[sigmoid||一个平滑的 S 形 activation，把任何数字压进 0 到 1 这个范围。它把 step 那道硬悬崖换成一道温柔的滑坡，于是 z 一个小变化就带来输出一个小变化。]] 这个 activation：它拿任何一个分数 `z`，把它压进 0 到 1 的范围 —— 读成「有多确定，从 0% 到 100%」。
~~~

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A dimmer switch shown as a horizontal slider. On the left the slider is low and the bulb is dark, near zero. In the middle it is half bright at one half. On the right it is full and the bulb is fully bright, near one. It slides smoothly through every level in between."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid = a dimmer that slides smoothly off → on</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid = 一个从关平滑滑到开的调光器</text><rect x="60" y="78" width="400" height="14" rx="7" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="110" cy="85" r="12" fill="#5E5191"/><text class="lang-en" x="110" y="114" text-anchor="middle" fill="#6B645E">low</text><text class="lang-zh" x="110" y="114" text-anchor="middle" fill="#6B645E">低</text><text class="lang-en" x="110" y="128" text-anchor="middle" fill="#9A938A" font-size="10">→ near 0</text><text class="lang-zh" x="110" y="128" text-anchor="middle" fill="#9A938A" font-size="10">→ 接近 0</text><circle cx="260" cy="58" r="16" fill="#EFEAF0" stroke="#9A938A"/><text x="260" y="64" text-anchor="middle" font-size="14">🌗</text><circle cx="350" cy="54" r="18" fill="#F6EFCF" stroke="#C99A12"/><text x="350" y="60" text-anchor="middle" font-size="14">🔆</text><circle cx="430" cy="50" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="430" y="57" text-anchor="middle" font-size="16">💡</text><text class="lang-en" x="260" y="114" text-anchor="middle" fill="#5E5191">middle</text><text class="lang-zh" x="260" y="114" text-anchor="middle" fill="#5E5191">中间</text><text x="260" y="128" text-anchor="middle" fill="#9A938A" font-size="10">= 0.5</text><text class="lang-en" x="430" y="114" text-anchor="middle" fill="#8A6D3B">full</text><text class="lang-zh" x="430" y="114" text-anchor="middle" fill="#8A6D3B">满</text><text class="lang-en" x="430" y="128" text-anchor="middle" fill="#9A938A" font-size="10">→ near 1</text><text class="lang-zh" x="430" y="128" text-anchor="middle" fill="#9A938A" font-size="10">→ 接近 1</text><text class="lang-en" x="260" y="158" text-anchor="middle" fill="#6B645E" font-size="10">no hard jump — every level in between is a real answer in (0, 1)</text><text class="lang-zh" x="260" y="158" text-anchor="middle" fill="#6B645E" font-size="10">没有硬跳 —— 中间每一档都是 (0, 1) 里一个真的答案</text></g></svg>
%%%

**The dimmer gets it right:** a smooth, gradual slide from off to on, every brightness in between a real setting. **One catch:** a real dimmer is evenly graded end to end, but sigmoid *flattens out* at the far ends — push the score very high or very low and the light barely changes. (That flattening hides a puzzle you'll crack tomorrow.)

#### Slide the dimmer through three positions
Stay on the dimmer knob and read off what the light does.

%%% steps
step: slide it far left (z = −4) → about 0.02
why: the bulb is nearly dark, therefore the neuron is saying "almost certainly no" — but not a flat zero, it still leaves a sliver
step: park it dead centre (z = 0) → exactly 0.5
why: perfectly undecided, which is the honest answer when the evidence cancels out
step: slide it far right (z = +4) → about 0.98
why: nearly full brightness — "almost certainly yes," and again it never quite reaches 1
step: nudge it anywhere by a hair → the light changes by a hair
why: that's the whole win over the cliff; a small change in z always makes a small, readable change in the answer
%%%

Here is the point of this unit in one figure — the step's cliff (left) versus sigmoid's smooth ramp (right). Look for the dot: at `z = 0` the ramp sits exactly halfway up, at `0.5`.
~~~zh
**调光器这个比喻对在哪里：** 一段平滑、渐进的滑动，从关到开，中间每一个亮度都是一档真的设置。**有一个地方要注意：** 真的调光器从头到尾刻度是均匀的。但 sigmoid 在两个远端*变平*了 —— 把分数推得非常高或者非常低，灯几乎不变。（那个变平藏着一个难题，你明天会把它破掉。）

#### 把调光器滑过三个位置
待在调光器旋钮上，读出灯在做什么。

%%% steps
step: 往最左边滑（z = −4）→ 大约 0.02
why: 灯泡差不多是暗的，因此 neuron 在说「几乎肯定不是」—— 但不是一个干脆的零，它还留了一条缝
step: 停在正中间（z = 0）→ 正好 0.5
why: 完全没主意，而在证据互相抵消的时候，这是最诚实的答案
step: 往最右边滑（z = +4）→ 大约 0.98
why: 差不多全亮 —— 「几乎肯定是」，而它又一次没有真的到 1
step: 在任何地方轻轻拧一丝 → 灯就变一丝
why: 这就是它比悬崖强的全部；z 一个小变化，总是带来答案一个小的、读得出来的变化
%%%

这个单元的重点，一张图讲完 —— step 那道悬崖（左边）对上 sigmoid 那道平滑的斜坡（右边）。找那个小点：在 `z = 0` 的地方，斜坡正好在半高，也就是 `0.5`。
~~~

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="Left panel: the step function as a sharp cliff jumping from 0 to 1. Right panel: the sigmoid as a smooth S-shaped ramp that passes exactly halfway up, at 0.5, where z equals zero, then keeps rising toward 1. The cliff gives no in-between; the ramp does."><g font-family="monospace" font-size="11"><text class="lang-en" x="130" y="20" text-anchor="middle" fill="#C93B3B" font-size="11">step: a cliff (no in-between)</text><text class="lang-zh" x="130" y="20" text-anchor="middle" fill="#C93B3B" font-size="11">step：一道悬崖（没有中间）</text><line x1="40" y1="120" x2="230" y2="120" stroke="#E5DFD6" stroke-width="1.5"/><line x1="135" y1="45" x2="135" y2="132" stroke="#E5DFD6" stroke-width="1"/><path d="M50 120 L135 120 L135 60 L225 60" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="135" cy="60" r="3" fill="#8A6D3B"/><text class="lang-en" x="130" y="150" text-anchor="middle" fill="#9A938A" font-size="10">tiny nudge → nothing moves</text><text class="lang-zh" x="130" y="150" text-anchor="middle" fill="#9A938A" font-size="10">轻轻一拧 → 什么都不动</text><text class="lang-en" x="390" y="20" text-anchor="middle" fill="#276b45" font-size="11">sigmoid: a smooth ramp</text><text class="lang-zh" x="390" y="20" text-anchor="middle" fill="#276b45" font-size="11">sigmoid：一道平滑的斜坡</text><line x1="295" y1="118" x2="490" y2="118" stroke="#E5DFD6" stroke-width="1.5"/><line x1="392" y1="45" x2="392" y2="132" stroke="#E5DFD6" stroke-width="1"/><line x1="295" y1="88" x2="490" y2="88" stroke="#E5DFD6" stroke-width="1" stroke-dasharray="3,3"/><path d="M300 117 C 342 117, 372 110, 392 88 C 412 66, 442 59, 485 58" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><circle cx="392" cy="88" r="4" fill="#5E5191"/><text x="404" y="84" fill="#5E5191" font-size="9">0.5 at z = 0</text><text class="lang-en" x="300" y="132" fill="#9A938A" font-size="9">near 0</text><text class="lang-zh" x="300" y="132" fill="#9A938A" font-size="9">接近 0</text><text class="lang-en" x="482" y="52" text-anchor="end" fill="#9A938A" font-size="9">near 1</text><text class="lang-zh" x="482" y="52" text-anchor="end" fill="#9A938A" font-size="9">接近 1</text><text class="lang-en" x="390" y="150" text-anchor="middle" fill="#2D8B55" font-size="10">tiny nudge → answer moves a little</text><text class="lang-zh" x="390" y="150" text-anchor="middle" fill="#2D8B55" font-size="10">轻轻一拧 → 答案动一点</text></g></svg>
%%%

%%% demo id=sigmoid label="predict, then run"
predict: five scores from −4 to +4 go into the dimmer. Which one comes out at exactly 0.5?
code: sigmoid(np.array([-4.0, -1.0, 0.0, 1.0, 4.0]))
out: array([0.018, 0.269, 0.5  , 0.731, 0.982])
take: <b>Sigmoid squashes any score into (0, 1).</b> A smooth off → on: <code>sigmoid(0) = 0.5</code> sits dead centre, big negatives land near <code>0</code>, big positives near <code>1</code> (values rounded). Because every answer lands between 0 and 1, it reads like a confidence — perfect when you want "how sure, 0 to 100%."
%%%

%%% insight
Feel the difference in one sentence: with the step, a small nudge changed *nothing* (or flipped everything). With the dimmer, a small nudge makes a small, honest change you can *follow* — and "follow the change" is literally how every model on earth learns. That swap, cliff → ramp, is what lets us improve a whole *stack* of neurons at once instead of just one.
%%%

**You now know both bookends of the activation family:** the original hard step, and the smooth sigmoid that replaced it. Tomorrow you'll meet the rest of the family. Today, let's watch the *whole* neuron run start to finish.
~~~zh
%%% demo id=sigmoidzh label="先猜，再展开"
predict: 从 −4 到 +4 的五个分数进调光器。哪一个出来正好是 0.5？
code: sigmoid(np.array([-4.0, -1.0, 0.0, 1.0, 4.0]))
out: array([0.018, 0.269, 0.5  , 0.731, 0.982])
take: <b>Sigmoid 把任何分数压进 (0, 1)。</b>一段平滑的关 → 开：<code>sigmoid(0) = 0.5</code> 坐在正中间，很大的负数落在 <code>0</code> 附近，很大的正数落在 <code>1</code> 附近（数值做了四舍五入）。因为每个答案都落在 0 和 1 之间，它读起来像一个信心值 —— 当你想要「有多确定，0 到 100%」的时候正合适。
%%%

%%% insight
一句话感受这个差别：用 step，一个小拧动什么都*没变*（或者整个翻转）。用调光器，一个小拧动带来一个小的、诚实的变化，是你能*跟着走*的 —— 而「跟着变化走」就是地球上每一个模型学习的方式，字面意思。悬崖 → 斜坡这一换，让我们能一次改进一整*叠* neuron，而不只是一个。
%%%

**你现在认识 activation 家族的两头了：** 最早那个硬邦邦的 step，和取代它的平滑 sigmoid。明天你会见到家族其余的成员。今天，我们看*整个* neuron 从头跑到尾。
~~~

@@@ concept id=c8 zh_tag="前向传播" tag="Forward pass" zh_title="Forward propagation —— 把整个 neuron 跑一遍" title="Forward propagation — run the whole neuron" zh_gotit="懂了 forward pass" gotit="Got the forward pass"
Let's run the whole brain cell end to end: **weigh → add → decide**, in one clean sweep.

Picture a **factory conveyor belt**. Raw parts go on at one end, move steadily to the right through station after station, and a finished product rolls off the far end. Nothing goes backward; each station does its one job and passes the work along.

A neuron computes its answer exactly like that, and the sweep has a name: [[forward propagation||Pushing the inputs left-to-right through the neuron — first the weighted sum, then the activation — to produce the output. Also called the forward pass.]], or the **forward pass**.
~~~zh
我们把整个脑细胞从头跑到尾：**称一称 → 加起来 → 做决定**，一趟干净地走完。

想象一条**工厂的传送带**。原料从一头放上去，稳稳地往右走，一个工位接一个工位，最后成品从另一头滚下来。没有东西往回走；每个工位做它那一件事，然后把活传下去。

一个 neuron 算出答案的方式正好就是这样，而这一趟有个名字：[[forward propagation（前向传播）||把输入从左到右推过 neuron —— 先是加权和，然后是 activation —— 产出输出。也叫 forward pass。]]，也就是 **forward pass**。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A factory conveyor belt moving left to right. Inputs enter on the left, pass a weigh-and-add station that produces the score z, then a decide station holding the activation, and a finished output rolls off the right end. Everything flows one way."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Forward pass = a conveyor: inputs in, one answer out</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Forward pass = 一条传送带：输入进去，一个答案出来</text><rect x="30" y="120" width="460" height="10" rx="5" fill="#D8D2C8"/><circle cx="45" cy="135" r="8" fill="#B8AEA2"/><circle cx="120" cy="135" r="8" fill="#B8AEA2"/><circle cx="260" cy="135" r="8" fill="#B8AEA2"/><circle cx="400" cy="135" r="8" fill="#B8AEA2"/><circle cx="475" cy="135" r="8" fill="#B8AEA2"/><rect x="40" y="55" width="70" height="45" rx="5" fill="#FDF9F3" stroke="#B8AEA2"/><text class="lang-en" x="75" y="74" text-anchor="middle" fill="#6B645E" font-size="10">inputs</text><text class="lang-zh" x="75" y="74" text-anchor="middle" fill="#6B645E" font-size="10">输入</text><text x="75" y="90" text-anchor="middle" fill="#3A342E" font-size="10">x = [2, 3]</text><text x="122" y="82" fill="#B8AEA2">→</text><rect x="145" y="55" width="110" height="45" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="200" y="74" text-anchor="middle" fill="#5E5191" font-size="10">weigh &amp; add</text><text class="lang-zh" x="200" y="74" text-anchor="middle" fill="#5E5191" font-size="10">称一称 &amp; 加起来</text><text x="200" y="90" text-anchor="middle" fill="#3A342E" font-size="10">z = −1.0</text><text x="267" y="82" fill="#B8AEA2">→</text><rect x="290" y="55" width="110" height="45" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text class="lang-en" x="345" y="74" text-anchor="middle" fill="#276b45" font-size="10">decide (sigmoid)</text><text class="lang-zh" x="345" y="74" text-anchor="middle" fill="#276b45" font-size="10">做决定（sigmoid）</text><text x="345" y="90" text-anchor="middle" fill="#3A342E" font-size="10">→ 0.27</text><text x="412" y="82" fill="#B8AEA2">→</text><rect x="430" y="55" width="60" height="45" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="460" y="74" text-anchor="middle" fill="#8A6D3B" font-size="10">output</text><text x="460" y="90" text-anchor="middle" fill="#8A6D3B" font-size="10">0.27</text><text class="lang-en" x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="10">one direction only — left to right, no going back</text><text class="lang-zh" x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="10">只有一个方向 —— 从左到右，不回头</text></g></svg>
%%%

**The conveyor gets it right:** work moves one way, each station does its job once, a finished result comes off the end. **One catch:** a factory belt never reverses, but a neuron *does* have a backward trip — that's how it learns (backprop, a later day). Today we only ride the belt forward.

#### Ride the belt, station by station
Nothing new here — just your three beats, bolted onto the conveyor in order.

%%% steps
step: station 0 · the parts arrive → x = [2, 3]
why: two raw input numbers get placed on the belt; the neuron never sees anything else about the world
step: station 1 · weigh & add → z = −1.0
why: each input meets its trust dial and the bias lands on top, therefore many numbers become one score
step: station 2 · decide (sigmoid) → 0.269
why: the dimmer turns that score into a readable confidence, which means the answer is now something a human can use
step: off the end · output 0.269
why: a soft "probably no, about 27% yes" — one number, from one pass, in one direction
%%%

%%% demo id=forward label="predict, then run"
predict: z will be −1.0 again. Since sigmoid(0) = 0.5, will the final answer sit above or below 0.5?
code: x = np.array([2, 3]); w = np.array([0.5, -1.0]); b = 1.0; z = (w*x).sum() + b; out = sigmoid(z); (z, round(out, 3))
out: (-1.0, 0.269)
take: <b>That's a full forward pass.</b> The inputs flowed through weigh-and-add to <code>z = −1.0</code>, then the sigmoid "decide" step turned that into <code>0.269</code> — a soft "probably no, about 27% yes." Weigh → add → decide, start to finish, in one direction.
%%%

%%% insight
Pause on this, because it's a bigger moment than it looks: you just ran the *same computation* the largest models on earth run. Not a simplified cartoon of it — the actual operation. When a giant model answers you, millions of these little forward passes fire at once, on the same conveyor, in the same order. The scale is different. The step is identical.
%%%

So far: a complete neuron, computed end to end. Now for the twist — this tidy little machine has one thing it flat-out *cannot* do.
~~~zh
**传送带这个比喻对在哪里：** 活只往一个方向走，每个工位做一次它的事，成品从末端出来。**有一个地方要注意：** 工厂的带子从不倒转，但一个 neuron *真的*有一趟往回的行程 —— 它就是这样学的（**backpropagation（反向传播）**，以后的某一天）。今天我们只往前坐这条带子。

#### 一个工位一个工位地坐这条带子
这里没有新东西 —— 就是你那三个拍子，按顺序装在传送带上。

%%% steps
step: 工位 0 · 原料到了 → x = [2, 3]
why: 两个原始的输入数字被放到带子上；关于这个世界，neuron 别的什么都看不到
step: 工位 1 · 称一称 & 加起来 → z = −1.0
why: 每个输入遇上它的信任旋钮，bias 落在上面，因此很多数字变成一个分数
step: 工位 2 · 做决定（sigmoid）→ 0.269
why: 调光器把那个分数变成一个读得懂的信心值，也就是说答案现在是人能用的东西了
step: 从末端出来 · 输出 0.269
why: 一个软软的「大概不是，大约 27% 是」—— 一个数字，一趟走完，一个方向
%%%

%%% demo id=forwardzh label="先猜，再展开"
predict: z 又会是 −1.0。既然 sigmoid(0) = 0.5，最后的答案会坐在 0.5 上面还是下面？
code: x = np.array([2, 3]); w = np.array([0.5, -1.0]); b = 1.0; z = (w*x).sum() + b; out = sigmoid(z); (z, round(out, 3))
out: (-1.0, 0.269)
take: <b>这就是一次完整的 forward pass。</b>输入流过称一称、加起来，到 <code>z = −1.0</code>，然后 sigmoid 这一步「做决定」把它变成 <code>0.269</code> —— 一个软软的「大概不是，大约 27% 是」。称一称 → 加起来 → 做决定，从头到尾，一个方向。
%%%

%%% insight
在这里停一下，因为这一刻比它看起来更大：你刚跑了地球上最大那些模型跑的*同一个计算*。不是它的简化卡通版 —— 是真正那个运算。当一个巨大的模型回答你，几百万个这样的小 forward pass 同时点火，在同一条传送带上，按同一个顺序。规模不一样。步骤一模一样。
%%%

到这里：一个完整的 neuron，从头算到尾。现在是那个转折 —— 这台整整齐齐的小机器，有一件事它彻底*做不到*。
~~~

@@@ concept id=c9 zh_tag="那条线" tag="The line" zh_title="一个 neuron 只画一条直线" title="A neuron draws one straight line" zh_gotit="懂了那条线" gotit="Got the line"
Here's a beautiful way to *see* what a neuron really does.

Imagine a **playground with kids scattered around**, and you're picking two teams. You grab a piece of chalk and draw one straight line across the ground: everyone on this side, team A; everyone on that side, team B. One straight chalk line, and the whole yard is split in two.

A neuron makes exactly that kind of call. Picture every possible pair of inputs as a dot on a flat map — the neuron's weights and bias together draw **one straight line** across it, a [[decision boundary||The straight line (or flat plane) a neuron draws to split its inputs into two sides — "yes" on one side, "no" on the other. Its position and tilt come from the weights and bias.]].
~~~zh
这里有一个漂亮的办法，让你*看见*一个 neuron 到底在做什么。

想象一个**操场上散着一堆小孩**，你要分两队。你抓起一根粉笔，在地上画一条直线：这一边的都是 A 队，那一边的都是 B 队。一条直的粉笔线，整个院子就一分为二。

一个 neuron 做的正是这种判断。把每一对可能的输入想成一张平面地图上的一个点 —— neuron 的 weight 和 bias 一起在地图上画**一条直线**，也就是一条 [[decision boundary（决策边界）||一个 neuron 画出来、把它的输入分成两边的那条直线（或者平面）—— 一边是「是」，另一边是「不是」。它的位置和倾斜来自 weight 和 bias。]]。
~~~

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A playground map with scattered dots. One straight chalk line splits them into two teams: green dots on one side get a yes, red dots on the other get a no. The line's tilt is set by the weights and its position by the bias."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A neuron = one straight chalk line splitting the yard</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一个 neuron = 一条把院子切开的直粉笔线</text><rect x="150" y="32" width="220" height="130" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="160" y1="150" x2="360" y2="44" stroke="#9A5A12" stroke-width="3"/><text class="lang-en" x="378" y="46" fill="#9A5A12" font-size="10">the line</text><text class="lang-zh" x="378" y="46" fill="#9A5A12" font-size="10">那条线</text><circle cx="195" cy="70" r="7" fill="#2D8B55"/><circle cx="230" cy="58" r="7" fill="#2D8B55"/><circle cx="215" cy="95" r="7" fill="#2D8B55"/><circle cx="260" cy="72" r="7" fill="#2D8B55"/><text class="lang-en" x="205" y="120" text-anchor="middle" fill="#276b45" font-size="10">"yes" side</text><text class="lang-zh" x="205" y="120" text-anchor="middle" fill="#276b45" font-size="10">「是」那一边</text><circle cx="300" cy="120" r="7" fill="#C93B3B"/><circle cx="330" cy="135" r="7" fill="#C93B3B"/><circle cx="315" cy="105" r="7" fill="#C93B3B"/><circle cx="345" cy="118" r="7" fill="#C93B3B"/><text class="lang-en" x="330" y="90" text-anchor="middle" fill="#C93B3B" font-size="10">"no" side</text><text class="lang-zh" x="330" y="90" text-anchor="middle" fill="#C93B3B" font-size="10">「不是」那一边</text><text x="70" y="80" text-anchor="middle" fill="#6B645E" font-size="10">weights</text><text class="lang-en" x="70" y="94" text-anchor="middle" fill="#6B645E" font-size="10">tilt it</text><text class="lang-zh" x="70" y="94" text-anchor="middle" fill="#6B645E" font-size="10">让它倾斜</text><text x="70" y="126" text-anchor="middle" fill="#6B645E" font-size="10">bias</text><text class="lang-en" x="70" y="140" text-anchor="middle" fill="#6B645E" font-size="10">slides it</text><text class="lang-zh" x="70" y="140" text-anchor="middle" fill="#6B645E" font-size="10">让它平移</text></g></svg>
%%%

**The chalk line gets it right:** the neuron carves its world into two sides with a single straight cut, and where that cut sits is *all* it controls. **One catch:** with more than two inputs the "line" becomes a flat sheet in a space too big to sketch — same idea, no drawing.

#### Grab the dials and move the line yourself
Forget my word for it — **play.** Three things to try, and predict each before you let go:

- Drag **w₁** or **w₂** and watch the line **tilt**. Weights set which direction counts as "more yes."
- Drag **b** and watch the same-angled line **slide** across the yard. Nothing tilts; it just shifts.
- Now try to make it **bend**. Any combination you like. (Watch the **yes ✓** label too — drag both weights to −2 and it jumps to the opposite corner.)
~~~zh
**粉笔线这个比喻对在哪里：** neuron 用一条直的切口把它的世界切成两边，而那条切口落在哪里，就是它能控制的*全部*。**有一个地方要注意：** 输入超过两个的时候，那条「线」会在一个大得画不出来的空间里变成一张平面 —— 想法一样，画不出来。

#### 抓住旋钮，自己把线挪一挪
别听我说 —— **玩。** 三件事可以试，每一件在松手之前先预测：

- 拖 **w₁** 或者 **w₂**，看那条线**倾斜**。weight 定了哪个方向算「更是」。
- 拖 **b**，看角度不变的那条线在院子里**平移**。什么都不倾斜；它只是挪了位置。
- 现在试着让它**弯**。你喜欢的任何组合都行。（也盯着 **是 ✓** 那个标签 —— 把两个 weight 都拖到 −2，它会跳到对面的角上。）
~~~

%%% svg
<svg id="db-svg" viewBox="0 0 520 232" role="img" aria-label="Interactive decision boundary. Drag the weight and bias sliders and watch the neuron's straight line tilt and slide across a map of dots, recoloring each dot green for yes or red for no. The yes and no labels move to whichever corner is actually on that side."><g font-family="monospace" font-size="11"><rect x="60" y="30" width="400" height="180" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="120" x2="460" y2="120" stroke="#EFE9DF" stroke-width="1"/><line x1="260" y1="30" x2="260" y2="210" stroke="#EFE9DF" stroke-width="1"/><line id="db-line" x1="60" y1="30" x2="460" y2="210" stroke="#9A5A12" stroke-width="3"/><circle id="db-p0" cx="110" cy="66" r="7" fill="#C93B3B"/><circle id="db-p1" cx="180" cy="102" r="7" fill="#C93B3B"/><circle id="db-p2" cx="310" cy="53" r="7" fill="#2D8B55"/><circle id="db-p3" cx="380" cy="84" r="7" fill="#2D8B55"/><circle id="db-p4" cx="140" cy="161" r="7" fill="#C93B3B"/><circle id="db-p5" cx="290" cy="183" r="7" fill="#C93B3B"/><circle id="db-p6" cx="400" cy="147" r="7" fill="#2D8B55"/><circle id="db-p7" cx="220" cy="143" r="7" fill="#C93B3B"/><text class="lang-en" id="db-lt" x="435" y="45" text-anchor="middle" fill="#276b45" font-size="10">yes ✓</text><text class="lang-zh" id="db-lt-zh" x="435" y="45" text-anchor="middle" fill="#276b45" font-size="10">是 ✓</text><text class="lang-en" id="db-lb" x="85" y="199" text-anchor="middle" fill="#C93B3B" font-size="10">no ✗</text><text class="lang-zh" id="db-lb-zh" x="85" y="199" text-anchor="middle" fill="#C93B3B" font-size="10">不是 ✗</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
<label>w₁ <input id="db-w1" type="range" min="-2" max="2" step="0.1" value="1" style="accent-color:#9A5A12;vertical-align:middle" aria-label="weight one"> <b id="db-w1v">1.0</b></label>
<label>w₂ <input id="db-w2" type="range" min="-2" max="2" step="0.1" value="1" style="accent-color:#9A5A12;vertical-align:middle" aria-label="weight two"> <b id="db-w2v">1.0</b></label>
<label>b <input id="db-b" type="range" min="-2" max="2" step="0.1" value="0" style="accent-color:#9A5A12;vertical-align:middle" aria-label="bias"> <b id="db-bv">0.0</b></label>
</div>
<div id="db-out" style="margin-top:6px;color:#2C2A28"><span class="lang-en">boundary: <b>1.0·x₁ + 1.0·x₂ + 0.0 = 0</b> — the weights tilt this line, the bias slides it.</span><span class="lang-zh">边界：<b>1.0·x₁ + 1.0·x₂ + 0.0 = 0</b> —— weight 让这条线倾斜，bias 让它平移。</span></div>
</div>
<script>(function(){
  var w1=document.getElementById('db-w1');if(!w1)return;
  var w2=document.getElementById('db-w2'),b=document.getElementById('db-b'),
      line=document.getElementById('db-line'),out=document.getElementById('db-out'),
      w1v=document.getElementById('db-w1v'),w2v=document.getElementById('db-w2v'),bv=document.getElementById('db-bv'),
      lt=document.getElementById('db-lt'),lb=document.getElementById('db-lb'),
      ltz=document.getElementById('db-lt-zh'),lbz=document.getElementById('db-lb-zh');
  var pts=[[-1.5,1.2],[-0.8,0.4],[0.5,1.5],[1.2,0.8],[-1.2,-0.9],[0.3,-1.4],[1.4,-0.6],[-0.4,-0.5]];
  function px(x){return 60+(x+2)/4*400;}
  function py(y){return 210-(y+2)/4*180;}
  // where the boundary meets the edges of the -2..2 box. DE-DUPLICATED: when the line
  // passes exactly through a corner, the x1-scan and the x2-scan find the SAME point, and
  // pushing it twice used to draw a zero-length (invisible) line while the dots stayed split.
  function ends(a,c,d){var lo=-2,hi=2,e=[];
    function add(x1,x2){
      for(var i=0;i<e.length;i++){if(Math.abs(e[i][0]-x1)<1e-6&&Math.abs(e[i][1]-x2)<1e-6)return;}
      e.push([x1,x2]);}
    if(Math.abs(c)>1e-9){[lo,hi].forEach(function(x1){var x2=-(a*x1+d)/c;if(x2>=lo-1e-9&&x2<=hi+1e-9)add(x1,x2);});}
    if(Math.abs(a)>1e-9){[lo,hi].forEach(function(x2){var x1=-(c*x2+d)/a;if(x1>=lo-1e-9&&x1<=hi+1e-9)add(x1,x2);});}
    return e;}
  function sgn(n){return (n<0?'− ':'+ ')+Math.abs(n).toFixed(1);}
  // the yes/no labels are NOT fixed: each one MOVES to a corner that is really on that side
  var corners=[[-1.75,1.75],[1.75,1.75],[-1.75,-1.75],[1.75,-1.75]];
  function place(el,txt,col,p,show){if(!el)return;
    el.textContent=txt;el.setAttribute('fill',col);
    el.setAttribute('x',px(p[0]).toFixed(1));el.setAttribute('y',py(p[1]).toFixed(1));
    el.setAttribute('opacity',show?'1':'0');}
  function labels(a,c,d){
    var best=null,worst=null;
    corners.forEach(function(p){var z=a*p[0]+c*p[1]+d;
      if(!best||z>best.z)best={p:p,z:z};
      if(!worst||z<worst.z)worst={p:p,z:z};});
    place(lt,'yes ✓','#276b45',best.p,best.z>=0);
    place(lb,'no ✗','#C93B3B',worst.p,worst.z<0);
    // the Chinese twins are SEPARATE elements with their own ids. Sharing id="db-lt"
    // would let getElementById move only the English one, so a Chinese reader would
    // watch a frozen label while the prose tells them to watch it jump corners.
    place(ltz,'是 ✓','#276b45',best.p,best.z>=0);
    place(lbz,'不是 ✗','#C93B3B',worst.p,worst.z<0);
  }
  function paint(){
    var a=+w1.value,c=+w2.value,d=+b.value;
    w1v.textContent=a.toFixed(1);w2v.textContent=c.toFixed(1);bv.textContent=d.toFixed(1);
    var yes=0;
    for(var i=0;i<pts.length;i++){var z=a*pts[i][0]+c*pts[i][1]+d;var el=document.getElementById('db-p'+i);
      if(el)el.setAttribute('fill',z>=0?'#2D8B55':'#C93B3B');if(z>=0)yes++;}
    labels(a,c,d);
    var e=ends(a,c,d);
    if(e.length>=2){line.setAttribute('x1',px(e[0][0]).toFixed(1));line.setAttribute('y1',py(e[0][1]).toFixed(1));
      line.setAttribute('x2',px(e[1][0]).toFixed(1));line.setAttribute('y2',py(e[1][1]).toFixed(1));line.setAttribute('opacity','1');}
    else line.setAttribute('opacity','0');
    out.innerHTML='<span class="lang-en">boundary: <b>'+a.toFixed(1)+'·x₁ '+sgn(c)+'·x₂ '+sgn(d)+' = 0</b> — '+yes+' of 8 dots land on the “yes” side.</span>'
                 +'<span class="lang-zh">边界：<b>'+a.toFixed(1)+'·x₁ '+sgn(c)+'·x₂ '+sgn(d)+' = 0</b> —— 8 个点里有 '+yes+' 个落在「是」那一边。</span>';
  }
  [w1,w2,b].forEach(function(el){el.addEventListener('input',paint);});paint();
})();</script>
%%%

%%% insight
Did you try to bend it? Everyone does. Tilt, slide, flip which side is "yes" — but no combination of those sliders will ever give you a curve or an L-shape. That stubborn straightness is not a limitation of the sliders; it *is* the neuron. And in about two minutes it's going to cost us something real.
%%%

**You just saw, with your own hands, the single most important fact about one neuron: it draws exactly one straight line.** Hold that thought — it's about to become the neuron's one true limit.
~~~zh
%%% insight
你试着让它弯了吗？每个人都会试。倾斜、平移、翻转哪一边是「是」—— 但这些滑块的任何组合，都不会给你一条曲线或者一个 L 形。那种顽固的直，不是滑块的限制；它*就是* neuron。而大约再过两分钟，它就要让我们付出真实的代价。
%%%

**你刚亲手看到了关于一个 neuron 最重要的那一个事实：它只画一条直线。** 记住这个念头 —— 它马上就要变成 neuron 唯一真正的上限。
~~~

@@@ concept id=c10 zh_tag="那个上限" tag="The limit" zh_title="一条线做不到 XOR" title="One line can't do XOR" zh_gotit="懂了这个上限" gotit="Got the limit"
Now the famous puzzle that stumped this whole field for years — and it starts as a party trick you can picture in your head.

Picture a **room with four chairs**, one in each corner, seen from above like a map. Two red chairs sit **top-left and bottom-right**; two blue chairs sit **top-right and bottom-left** — so each colour sits on its own **diagonal**. I hand you one straight **rope** pulled tight across the floor: put both reds on one side and both blues on the other.

Go on, try to picture it. You can't. Slide the rope, angle it any way you like — one straight rope always leaves one chair on the wrong side.

That room is the [[XOR||"On when exactly one of two inputs is on." Its two answer-groups sit on opposite diagonals, so no single straight line can separate them.]] pattern: "yes when exactly one of two inputs is on." And a single neuron, you just saw, draws exactly **one straight line** — your one straight rope. XOR is not [[linearly separable||A pattern is linearly separable if one straight line can put every "yes" on one side and every "no" on the other. XOR is the classic pattern that is NOT — no single line works.]].
~~~zh
现在是那个把整个领域卡住好几年的有名难题 —— 而它一开始像一个你能在脑子里想出来的派对小把戏。

想象一个**放着四把椅子的房间**，每个角一把，从上面看像一张地图。两把红椅子坐在**左上和右下**；两把蓝椅子坐在**右上和左下** —— 所以每个颜色各自坐在自己的**对角线**上。我给你一根拉紧的直**绳子**横在地板上：把两把红的放到一边，两把蓝的放到另一边。

来，试着在脑子里摆一摆。你摆不出来。把绳子挪一挪，随便怎么转角度 —— 一根直绳子总会留下一把椅子在错的那一边。

那个房间就是 [[XOR||「两个输入里正好有一个开的时候就开。」它的两组答案坐在相对的对角线上，所以没有任何一条直线能把它们分开。]] 这个花样：「两个输入里正好有一个开的时候就是。」而一个 neuron，你刚看到，只画**一条直线** —— 就是你那一根直绳子。XOR 不是 [[linearly separable||如果一条直线能把每个「是」放到一边、每个「不是」放到另一边，这个花样就是 linearly separable。XOR 是那个经典的*不是* —— 没有一条线管用。]] 的。
~~~

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="A room floor seen from above with four chairs at the corners. Two red chairs sit on one diagonal (top-left and bottom-right); two blue chairs sit on the other (top-right and bottom-left). One straight rope crosses the floor so that only the bottom-right red chair is below it, while the top-left red chair, the top-right blue chair and the bottom-left blue chair are all above it. The top-left red chair is therefore stranded on the blues' side, so this rope gets three of the four chairs right, and no angle does better."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One straight rope can't split two diagonal pairs</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一根直绳子分不开两条对角线上的两对</text><rect x="150" y="30" width="220" height="138" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><line x1="160" y1="166" x2="365" y2="71" stroke="#9A5A12" stroke-width="3"/><text class="lang-en" x="376" y="66" fill="#9A5A12" font-size="10">straight rope</text><text class="lang-zh" x="376" y="66" fill="#9A5A12" font-size="10">直绳子</text><rect x="172" y="45" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="185" y="64" text-anchor="middle" fill="#C93B3B">🔴</text><circle cx="185" cy="58" r="21" fill="none" stroke="#C99A12" stroke-width="2" stroke-dasharray="3,3"/><rect x="322" y="45" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="335" y="64" text-anchor="middle" fill="#5E5191">🔵</text><rect x="172" y="119" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="185" y="138" text-anchor="middle" fill="#5E5191">🔵</text><rect x="322" y="119" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="335" y="138" text-anchor="middle" fill="#C93B3B">🔴</text><text class="lang-en" x="70" y="88" text-anchor="middle" fill="#6B645E" font-size="10">reds: top-left +</text><text class="lang-zh" x="70" y="88" text-anchor="middle" fill="#6B645E" font-size="10">红椅子：左上 +</text><text class="lang-en" x="70" y="102" text-anchor="middle" fill="#6B645E" font-size="10">bottom-right;</text><text class="lang-zh" x="70" y="102" text-anchor="middle" fill="#6B645E" font-size="10">右下；</text><text class="lang-en" x="70" y="116" text-anchor="middle" fill="#6B645E" font-size="10">blues the other</text><text class="lang-zh" x="70" y="116" text-anchor="middle" fill="#6B645E" font-size="10">蓝的在另一条</text><text class="lang-en" x="440" y="120" text-anchor="middle" fill="#6B645E" font-size="10">above the rope:</text><text class="lang-zh" x="440" y="120" text-anchor="middle" fill="#6B645E" font-size="10">绳子上面：</text><text class="lang-en" x="440" y="134" text-anchor="middle" fill="#5E5191" font-size="10">2 blue + 1 red</text><text class="lang-zh" x="440" y="134" text-anchor="middle" fill="#5E5191" font-size="10">2 蓝 + 1 红</text><text class="lang-en" x="440" y="152" text-anchor="middle" fill="#6B645E" font-size="10">below the rope:</text><text class="lang-zh" x="440" y="152" text-anchor="middle" fill="#6B645E" font-size="10">绳子下面：</text><text class="lang-en" x="440" y="166" text-anchor="middle" fill="#C93B3B" font-size="10">1 red</text><text class="lang-zh" x="440" y="166" text-anchor="middle" fill="#C93B3B" font-size="10">1 红</text><text class="lang-en" x="260" y="186" text-anchor="middle" fill="#C99A12" font-size="10">the ringed red chair is stranded on the blues' side</text><text class="lang-zh" x="260" y="186" text-anchor="middle" fill="#C99A12" font-size="10">被圈出来的那把红椅子被困在蓝的那一边</text><text class="lang-en" x="260" y="200" text-anchor="middle" fill="#C93B3B" font-size="10">any angle → best you get is 3 of 4</text><text class="lang-zh" x="260" y="200" text-anchor="middle" fill="#C93B3B" font-size="10">任何角度 → 最好也只有 4 把里的 3 把</text></g></svg>
%%%

**The chairs get it right:** the diagonal layout *is* the trap — two groups arranged so no single straight cut can separate them. **One catch:** with a real rope you're stuck for good, but a network isn't limited to one rope (that's the next unit's rescue).

If you're mid-shrug going "wait, surely *some* angle works" — good. That's the exact feeling everybody has here, including the researchers who argued about it in print. Want to wrestle it yourself first? Here's a ladder of nudges.

%%% hint
t1: Don't worry about the math yet — just look at the four chairs. Same-colored chairs sit on opposite corners (a diagonal), not next to each other. Ask yourself: could one straight rope ever keep two things that sit on opposite corners on the same side?
t2: Try it slowly. Put the rope so it fences off the top-left chair alone — good, that one's handled. Now its same-color partner is way down at the bottom-right, on the far side of your rope. To grab it too you'd have to swing the rope over there — but then you let one of the OTHER color slip onto your side. You keep trading one mistake for another, which is why the ceiling is 3 of 4.
t3: One neuron draws exactly one straight line, and XOR's two groups sit on crossing diagonals — so no single straight line can ever put all the "yes"s on one side and all the "no"s on the other. That's the whole limit.
%%%

#### Try to beat it yourself — you can't
Drag the slider to slide the one straight rope across the floor. The chair that escapes gets a ring around it. (Curious about the fix? Tick **add a hidden layer** to drop in a second rope — that's the next unit, previewed.)
~~~zh
**四把椅子这个比喻对在哪里：** 对角线这个摆法*就是*那个陷阱 —— 两组东西摆成这样，就没有一条直的切口能分开它们。**有一个地方要注意：** 用一根真绳子你就永远卡住了，但一个网络不是只能有一根绳子（那就是下一个单元的救援）。

如果你正在半耸着肩想「等等，肯定*有个*角度行吧」—— 很好。这里每个人的感觉都正好是这样，包括当年在论文里为它吵架的研究者。想先自己较量一下吗？这里有一梯子的小提示。

%%% hint
t1: 先别管数学 —— 就看那四把椅子。同色的椅子坐在相对的角上（一条对角线），不是挨着的。问自己一句：一根直绳子有可能把两个坐在相对角上的东西留在同一边吗？
t2: 慢慢试。把绳子摆成只把左上那把围起来 —— 好，这一把搞定了。现在它的同色伙伴远远在右下，在你绳子的另一边。要把它也抓过来，你得把绳子甩到那边去 —— 但那样你就让*另一个*颜色的一把溜进了你这边。你一直在用一个错换另一个错，所以上限是 4 把里的 3 把。
t3: 一个 neuron 只画一条直线，而 XOR 的两组坐在交叉的对角线上 —— 所以没有任何一条直线能把所有「是」放到一边、所有「不是」放到另一边。这就是整个上限。
%%%

#### 自己试着打破它 —— 你打不破
拖动滑块，让那一根直绳子在地板上挪。跑掉的那把椅子会被圈起来。（好奇解法？勾上 **加一个 hidden layer** 放进第二根绳子 —— 那是下一个单元，先给你预览。）
~~~

%%% svg
<svg id="xor-svg" viewBox="0 0 520 236" role="img" aria-label="Interactive XOR. Slide a single straight rope across four corner chairs and watch it catch at most three of four, with a ring around the one that escapes. Tick the hidden-layer box to add a second rope that forms a band around the two red chairs and catches all four."><g font-family="monospace" font-size="11"><rect x="150" y="30" width="220" height="180" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line id="xor-l2" x1="0" y1="0" x2="0" y2="0" stroke="#2D8B55" stroke-width="3" stroke-dasharray="6,4" opacity="0"/><line id="xor-l1" x1="190" y1="106" x2="274" y2="190" stroke="#9A5A12" stroke-width="3"/><circle id="xor-miss" cx="330" cy="50" r="15" fill="none" stroke="#C99A12" stroke-width="2.5" stroke-dasharray="3,3" opacity="1"/><circle cx="190" cy="190" r="9" fill="#5E5191"/><circle cx="190" cy="50" r="9" fill="#C93B3B"/><circle cx="330" cy="190" r="9" fill="#C93B3B"/><circle cx="330" cy="50" r="9" fill="#5E5191"/><text class="lang-en" x="150" y="224" fill="#5E5191" font-size="9">🔵 off = both/neither on</text><text class="lang-zh" x="150" y="224" fill="#5E5191" font-size="9">🔵 关 = 两个都开或都不开</text><text class="lang-en" x="300" y="224" fill="#C93B3B" font-size="9">🔴 on = exactly one</text><text class="lang-zh" x="300" y="224" fill="#C93B3B" font-size="9">🔴 开 = 正好一个开</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center">
<label><span class="lang-en">slide the rope</span><span class="lang-zh">拖动绳子</span> <input id="xor-c" type="range" min="0.3" max="1.7" step="0.05" value="0.6" style="accent-color:#9A5A12;vertical-align:middle" aria-label="rope position"></label>
<label style="color:#276b45"><input id="xor-hl" type="checkbox" style="vertical-align:middle"> <span class="lang-en">add a hidden layer (a 2nd rope)</span><span class="lang-zh">加一个 hidden layer（第 2 根绳子）</span></label>
</div>
<div id="xor-out" style="margin-top:6px;color:#2C2A28"><span class="lang-en">one straight rope catches <b>3 of 4</b> — one chair always escapes (ringed).</span><span class="lang-zh">一根直绳子只抓住 <b>4 把里的 3 把</b> —— 总有一把椅子跑掉（被圈出来的那把）。</span></div>
</div>
<script>(function(){
  var cs=document.getElementById('xor-c');if(!cs)return;
  var hl=document.getElementById('xor-hl'),l1=document.getElementById('xor-l1'),l2=document.getElementById('xor-l2'),
      miss=document.getElementById('xor-miss'),out=document.getElementById('xor-out');
  function px(x){return 190+x*140;}
  function py(y){return 190-y*140;}
  // edge hits, de-duplicated (a rope through a corner is found twice; drawing e[0]->e[1]
  // would then be a zero-length, invisible line)
  function ends(c){var e=[];
    function add(x1,x2){
      for(var i=0;i<e.length;i++){if(Math.abs(e[i][0]-x1)<1e-6&&Math.abs(e[i][1]-x2)<1e-6)return;}
      e.push([x1,x2]);}
    [0,1].forEach(function(x1){var x2=c-x1;if(x2>=0-1e-9&&x2<=1+1e-9)add(x1,x2);});
    [0,1].forEach(function(x2){var x1=c-x2;if(x1>=0-1e-9&&x1<=1+1e-9)add(x1,x2);});return e;}
  function setLine(el,c){var e=ends(c);if(e.length>=2){el.setAttribute('x1',px(e[0][0]).toFixed(1));el.setAttribute('y1',py(e[0][1]).toFixed(1));
    el.setAttribute('x2',px(e[1][0]).toFixed(1));el.setAttribute('y2',py(e[1][1]).toFixed(1));el.setAttribute('opacity','1');}}
  var P=[[0,0,0],[0,1,1],[1,0,1],[1,1,0]]; // x1,x2,trueClass
  function paint(){
    if(hl.checked){
      setLine(l1,0.5);setLine(l2,1.5);miss.setAttribute('opacity','0');cs.disabled=true;
      out.innerHTML='<span class="lang-en">two ropes make a <b>band</b> around the two red chairs — inside = “on”, outside = “off”. All <b>4 of 4</b> split ✓ — that’s a hidden layer.</span>'
                   +'<span class="lang-zh">两根绳子在两把红椅子外面围成一条<b>带子</b> —— 带子里面 = 「开」，外面 = 「关」。<b>4 把全部</b>分开了 ✓ —— 这就是一个 hidden layer。</span>';
      return;
    }
    cs.disabled=false;l2.setAttribute('opacity','0');
    var c=+cs.value;setLine(l1,c);
    var A=[],B=[];P.forEach(function(p){(p[0]+p[1]>=c?A:B).push(p);});
    function maj(arr){var o=arr.filter(function(p){return p[2]===1;}).length;return o*2>=arr.length?1:0;}
    var la=maj(A),lb=maj(B),wrong=[];
    A.forEach(function(p){if(p[2]!==la)wrong.push(p);});B.forEach(function(p){if(p[2]!==lb)wrong.push(p);});
    if(wrong.length){miss.setAttribute('cx',px(wrong[0][0]));miss.setAttribute('cy',py(wrong[0][1]));miss.setAttribute('opacity','1');}
    else miss.setAttribute('opacity','0');
    out.innerHTML='<span class="lang-en">one straight rope catches <b>'+(4-wrong.length)+' of 4</b> — one chair always escapes (ringed). Slide it anywhere: never 4.</span>'
                 +'<span class="lang-zh">一根直绳子只抓住 <b>4 把里的 '+(4-wrong.length)+' 把</b> —— 总有一把椅子跑掉（被圈出来的那把）。随便怎么拖：都到不了 4 把。</span>';
  }
  cs.addEventListener('input',paint);hl.addEventListener('change',paint);paint();
})();</script>
%%%

%%% insight
This is not you being bad at sliding a rope. It's a *proof*, and it once cost the field years of funding: one neuron draws one straight line, and some patterns can't be split by any straight line. So whenever you hear "neural nets can solve any problem" — you now have the four-chair counterexample. Knowing exactly where a tool stops is what separates someone who *uses* AI from someone who *understands* it, and you just did that on Day 1.
%%%

**You've found the exact edge of a single neuron's power.** The fix is waiting in the very next unit, and it's almost embarrassingly simple.
~~~zh
%%% insight
这不是你不会挪绳子。这是一个*证明*，而它当年让这个领域损失了好几年的经费：一个 neuron 画一条直线，而有些花样任何直线都分不开。所以以后你听到「神经网络能解决任何问题」—— 你现在手里有那个四把椅子的反例。清楚知道一个工具在哪里停下来，就是*用* AI 的人和*懂* AI 的人之间的分界，而你在第 1 天就做到了。
%%%

**你找到了一个 neuron 力量的确切边界。** 解法就在下一个单元等着，而它简单得几乎让人不好意思。
~~~

@@@ concept id=c11 zh_tag="Layer 与网络" tag="Layers & networks" zh_title="把 neuron 连成网络，线就能弯了" title="Join neurons into a network and the line can bend" zh_gotit="懂了 layer" gotit="Got layers"
So one rope can't split the four chairs. The fix is almost silly once you see it: **use more than one rope.**

Give yourself a *second* straight rope and you can fence off a strip of floor — one rope on each side of a diagonal band. Together they wrap two chairs inside the strip while the other two sit outside. Two simple straight pieces, combining into something bent.

That's exactly what happens when you put several neurons side by side. A [[layer||A row of neurons working side by side, all reading the SAME inputs. Each neuron draws its own straight line; together the layer draws several lines at once.]] is a **row of neurons** that all read the same inputs at the same time — a panel of judges, each scoring the same act with their own opinion.
~~~zh
所以一根绳子分不开那四把椅子。解法一看就有点傻：**用不止一根绳子。**

给自己*第二*根直绳子，你就能围出地板上的一条带 —— 一条对角带子的两边各放一根绳子。它们一起把两把椅子裹在带子里面，另外两把坐在外面。两段简单的直东西，合起来变成了弯的东西。

把几个 neuron 并排放在一起，发生的正是这件事。一个 [[layer||一排并排干活的 neuron，全都读同样的输入。每个 neuron 画自己那条直线；合在一起，这个 layer 一次画好几条线。]] 就是**一排 neuron**，它们同时读同样的输入 —— 一组评委，每个人用自己的看法给同一个节目打分。
~~~

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A layer drawn as a row of two neurons side by side. The same two inputs on the left fan out to both neurons. Each neuron draws its own straight line, so the layer produces two lines from the same inputs. This is like a panel of judges each scoring the same act."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A layer = a row of neurons, all reading the SAME inputs</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一个 layer = 一排 neuron，全都读同样的输入</text><circle cx="55" cy="70" r="9" fill="#EFEAF7" stroke="#7C6DAA"/><text x="55" y="74" text-anchor="middle" fill="#5E5191" font-size="9">x₁</text><circle cx="55" cy="120" r="9" fill="#EFEAF7" stroke="#7C6DAA"/><text x="55" y="124" text-anchor="middle" fill="#5E5191" font-size="9">x₂</text><line x1="64" y1="70" x2="180" y2="65" stroke="#B8AEA2"/><line x1="64" y1="120" x2="180" y2="75" stroke="#B8AEA2"/><line x1="64" y1="70" x2="180" y2="130" stroke="#B8AEA2"/><line x1="64" y1="120" x2="180" y2="140" stroke="#B8AEA2"/><rect x="180" y="52" width="120" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="240" y="70" text-anchor="middle" fill="#276b45" font-size="10">neuron A</text><text class="lang-en" x="240" y="86" text-anchor="middle" fill="#6B645E" font-size="9">draws line 1</text><text class="lang-zh" x="240" y="86" text-anchor="middle" fill="#6B645E" font-size="9">画线 1</text><rect x="180" y="112" width="120" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="240" y="130" text-anchor="middle" fill="#276b45" font-size="10">neuron B</text><text class="lang-en" x="240" y="146" text-anchor="middle" fill="#6B645E" font-size="9">draws line 2</text><text class="lang-zh" x="240" y="146" text-anchor="middle" fill="#6B645E" font-size="9">画线 2</text><text class="lang-en" x="120" y="44" text-anchor="middle" fill="#6B645E" font-size="10">same inputs</text><text class="lang-zh" x="120" y="44" text-anchor="middle" fill="#6B645E" font-size="10">同样的输入</text><text class="lang-en" x="120" y="170" text-anchor="middle" fill="#9A938A" font-size="9">fan out to every neuron</text><text class="lang-zh" x="120" y="170" text-anchor="middle" fill="#9A938A" font-size="9">分给每一个 neuron</text><path d="M312 74 Q 340 90 312 134" fill="none" stroke="#B8AEA2" stroke-width="1.5"/><text class="lang-en" x="380" y="98" text-anchor="middle" fill="#276b45" font-size="10">2 neurons</text><text class="lang-zh" x="380" y="98" text-anchor="middle" fill="#276b45" font-size="10">2 个 neuron</text><text class="lang-en" x="380" y="114" text-anchor="middle" fill="#276b45" font-size="10">= 2 lines at once</text><text class="lang-zh" x="380" y="114" text-anchor="middle" fill="#276b45" font-size="10">= 一次 2 条线</text></g></svg>
%%%

**The panel of judges gets it right:** every neuron in a layer sees the same inputs and forms its own opinion (its own line). **One catch:** real judges influence each other; neurons in a layer never talk — they work in parallel, and only their outputs get combined, by the *next* layer.

#### From one rope to a bendy boundary
Here is the whole rescue, one rung at a time. Keep the ropes in mind — each rung is one more rope, or one more knot.

%%% steps
step: one neuron → one rope
why: exactly what you played with; a single straight cut, therefore stuck at 3 of 4
step: a layer of two neurons → two ropes at once
why: both read the same inputs but hold different dials, which means two independent straight cuts across the same floor
step: feed those outputs into a *second* layer → the knot
why: the next neuron reads "which side of rope 1?" and "which side of rope 2?" — so it can demand *inside both*, and that combination is a band, not a line
step: the band catches all four chairs → 4 of 4
why: the straightness never went away; it got *combined*, therefore the boundary bends without any single neuron bending
%%%

That stack of layers has a name: a [[neural network||Several layers of neurons connected in a chain: each layer's outputs become the next layer's inputs. The network as a whole can bend and fold boundaries a single neuron never could.]] — neurons wired in a chain, each layer feeding the next.

%%% insight
Sit with how strange and lovely this is. Nobody taught any neuron to draw a curve. Every single one is still doing the same dumb straight cut you built in ten minutes. The bending is an *emergent* property of stacking — the limit didn't get fixed, it got **outgrown by teamwork**. That is genuinely the central trick of all deep learning, and you just watched it happen with two ropes.
%%%

#### The three kinds of layer, by where they sit
Layers get names from their position in the chain. Three names, one line each.

- The [[input layer||The raw numbers you feed in — the very first values that enter the network. Not neurons that compute, just the inputs handed to the first real layer.]] — the raw inputs you hand in (`x₁`, `x₂`, …). No computing; just the numbers at the very front.
- A [[hidden layer||A layer of neurons in the MIDDLE of a network — between the inputs and the final answer. It's "hidden" because you never read its outputs directly; they only feed the next layer. Hidden layers are what let the boundary bend.]] — any layer of neurons in the *middle*. "Hidden" because you never read its numbers; they exist only to feed the next layer. This is the layer that split the four chairs.
- The [[output layer||The LAST layer, whose result is the network's final answer — the number (or numbers) you actually read and use.]] — the very last layer, whose result is the answer you actually use.
~~~zh
**评委席这个比喻对在哪里：** 一个 layer 里每个 neuron 都看到同样的输入，各自形成自己的看法（自己那条线）。**有一个地方要注意：** 真的评委会互相影响；一个 layer 里的 neuron 从不说话 —— 它们并行地干活，只有它们的输出会被合起来，而合它们的是*下一个* layer。

#### 从一根绳子到一条会弯的边界
整个救援在这里，一级一级来。把绳子放在心里 —— 每一级就是多一根绳子，或者多一个结。

%%% steps
step: 一个 neuron → 一根绳子
why: 正是你刚玩的那个；一条直的切口，因此卡在 4 把里的 3 把
step: 一个装两个 neuron 的 layer → 一次两根绳子
why: 两个都读同样的输入，但握着不同的旋钮，也就是说同一块地板上有两条独立的直切口
step: 把那些输出喂进*第二个* layer → 那个结
why: 下一个 neuron 读的是「在绳子 1 的哪一边？」和「在绳子 2 的哪一边？」—— 所以它可以要求*两个都在里面*，而那个组合是一条带子，不是一条线
step: 那条带子抓住全部四把椅子 → 4 把全对
why: 那份直从来没消失；它被*合起来*了，因此边界弯了，而没有任何单个 neuron 在弯
%%%

那一叠 layer 有个名字：一个 [[neural network（神经网络）||好几层 neuron 连成一条链：每一层的输出变成下一层的输入。整个网络能弯、能折出单个 neuron 永远做不到的边界。]] —— neuron 接成一条链，每一层喂下一层。

%%% insight
坐一会儿，感受这件事有多奇怪、多可爱。没有人教过任何一个 neuron 画曲线。每一个都还在做你十分钟就搭出来的那个笨笨的直切口。弯是叠起来以后*长出来*的性质 —— 那个上限没有被修好，它是**被团队合作长过去了**。这真的是整个 deep learning 的中心花招，而你刚用两根绳子看它发生了一次。
%%%

#### 三种 layer，按它们坐在哪里分
Layer 的名字来自它在链上的位置。三个名字，一句话一个。

- [[input layer（输入层）||你喂进去的原始数字 —— 进入网络的最前面那些值。它们不是会计算的 neuron，只是递给第一个真 layer 的输入。]] —— 你递进去的原始输入（`x₁`、`x₂`、……）。不计算；就是最前面那些数字。
- 一个 [[hidden layer（隐藏层）||一个在网络*中间*的 neuron 层 —— 在输入和最后的答案之间。它「隐藏」是因为你从不直接读它的输出；它们只喂给下一层。hidden layer 就是让边界能弯的东西。]] —— 任何在*中间*的 neuron 层。叫「隐藏」是因为你从不读它的数字；它们存在只是为了喂下一层。就是这一层分开了那四把椅子。
- [[output layer（输出层）||最后那一层，它的结果就是网络最终的答案 —— 你真正读到、真正会用的那个（或者那些）数字。]] —— 最后那一层，它的结果就是你真正会用的答案。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A small neural network in three named columns. On the left, the input layer is two raw input values. In the middle, a hidden layer of two neurons, each connected to both inputs. On the right, an output layer of one neuron producing the final answer. Arrows flow left to right from inputs through the hidden layer to the output."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A network: input layer → hidden layer → output layer</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一个网络：input layer → hidden layer → output layer</text><circle cx="70" cy="75" r="12" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="70" y="79" text-anchor="middle" fill="#6B645E" font-size="9">x₁</text><circle cx="70" cy="135" r="12" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="70" y="139" text-anchor="middle" fill="#6B645E" font-size="9">x₂</text><text x="70" y="172" text-anchor="middle" fill="#6B645E" font-size="10">input layer</text><text class="lang-en" x="70" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(raw inputs)</text><text class="lang-zh" x="70" y="186" text-anchor="middle" fill="#9A938A" font-size="9">（原始输入）</text><line x1="82" y1="75" x2="248" y2="75" stroke="#B8AEA2"/><line x1="82" y1="75" x2="248" y2="135" stroke="#B8AEA2"/><line x1="82" y1="135" x2="248" y2="75" stroke="#B8AEA2"/><line x1="82" y1="135" x2="248" y2="135" stroke="#B8AEA2"/><circle cx="260" cy="75" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><circle cx="260" cy="135" r="14" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="260" y="172" text-anchor="middle" fill="#276b45" font-size="10">hidden layer</text><text class="lang-en" x="260" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(the bend lives here)</text><text class="lang-zh" x="260" y="186" text-anchor="middle" fill="#9A938A" font-size="9">（弯就住在这里）</text><line x1="274" y1="75" x2="426" y2="105" stroke="#B8AEA2"/><line x1="274" y1="135" x2="426" y2="105" stroke="#B8AEA2"/><circle cx="440" cy="105" r="14" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="440" y="172" text-anchor="middle" fill="#8A6D3B" font-size="10">output layer</text><text class="lang-en" x="440" y="186" text-anchor="middle" fill="#9A938A" font-size="9">(the answer)</text><text class="lang-zh" x="440" y="186" text-anchor="middle" fill="#9A938A" font-size="9">（答案）</text><line x1="454" y1="105" x2="500" y2="105" stroke="#C99A12" stroke-width="2"/><polygon points="500,100 512,105 500,110" fill="#C99A12"/></g></svg>
%%%

Real networks stack many hidden layers with thousands of neurons each — far too many to draw — but the shape is always this same input → hidden → output chain.

#### That's what "deep" means
Now the word you've heard a hundred times finally has a plain meaning.

Stack **many hidden layers** one after another and we call the network **deep** — training it is [[deep learning||Training a neural network that has many hidden layers stacked in a chain. "Deep" literally means "many layers deep"; each layer bends the boundary a little more.]]. That's the whole secret behind the term: "deep" just means "many layers deep."

Each extra hidden layer bends the boundary a little more, so a deep stack carves shapes wildly more intricate than any single line. ChatGPT is a very deep network — a tall stack built out of exactly these neurons, plus a few extra parts you'll meet in later modules.

!!! c-warn ⚠️
<b>Myth to dodge — "bigger is always better."</b> It's tempting to think "more layers, more neurons, always smarter." Not so: a network that's too big for its data just <i>memorizes</i> the examples instead of learning the real pattern (you'll meet this as <b>overfitting</b> later), and it costs far more to train and run. The right size fits the problem — bigger is a tool, not a magic wand.
!!!

#### See the fix: one line fails, a hidden layer bends it
Same four chairs, split by one line (fails) versus two lines from a hidden layer (works). The band wraps the two **red** chairs — the same pair the live widget wraps. Either diagonal pair can be the one inside the band; what matters is that a band is a shape no single line could ever make.
~~~zh
真实的网络会叠很多 hidden layer，每层几千个 neuron —— 多到根本画不出来 —— 但形状永远是同一条 input → hidden → output 的链。

#### 「deep」就是这个意思
现在那个你听过一百遍的词，终于有了一句大白话的意思。

把**很多 hidden layer** 一个接一个叠起来，我们就叫这个网络**深**（deep）—— 训练它就是 [[deep learning||训练一个有很多 hidden layer 叠成一条链的神经网络。「deep」字面就是「叠了很多层」；每一层把边界多弯一点。]]。这个词背后全部的秘密就是这个：「deep」只是「叠了很多层」。

每多一个 hidden layer，边界就多弯一点，所以一叠很深的网络能刻出比任何单条线都复杂得离谱的形状。ChatGPT 是一个非常深的网络 —— 一座高高的叠，正是用这些 neuron 搭起来的，外加几个你会在后面模块里见到的额外零件。

!!! c-warn ⚠️
<b>要躲开的迷思 —— 「越大总是越好」。</b> 很容易觉得「层更多、neuron 更多，就总是更聪明」。不是这样。一个对它的数据来说太大的网络，只会把样本<i>背下来</i>，而不是学到真正的花样（你以后会以 <b>overfitting（过拟合）</b> 的名字见到它）。而且它训练和运行都贵得多。合适的大小是配得上这个问题的大小 —— 大是一个工具，不是魔法棒。
!!!

#### 看那个解法：一条线失败，一个 hidden layer 把它弯过来
还是那四把椅子，一条线来分（失败）对上一个 hidden layer 给的两条线（成功）。带子裹住的是那两把**红**椅子 —— 和上面那个活的小工具裹的是同一对。哪一条对角线上的一对在带子里面都可以；要紧的是，带子是任何单条线永远做不出来的形状。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Left: one straight line tries to separate two diagonal pairs of dots and fails. It leaves the top-left red dot on the same side as both blue dots, so only three of the four are right. Right: two parallel straight lines from a hidden layer form a band along the other diagonal. Both red dots (top-left and bottom-right) fall inside the band; both blue dots (top-right and bottom-left) fall outside it. The band cleanly separates the two classes, four of four."><g font-family="monospace" font-size="11"><text class="lang-en" x="128" y="20" text-anchor="middle" fill="#C93B3B" font-size="11">one neuron: one line (fails)</text><text class="lang-zh" x="128" y="20" text-anchor="middle" fill="#C93B3B" font-size="11">一个 neuron：一条线（失败）</text><rect x="33" y="30" width="190" height="132" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="43" y1="152" x2="213" y2="67" stroke="#9A5A12" stroke-width="2.5"/><circle cx="68" cy="60" r="6" fill="#C93B3B"/><circle cx="188" cy="128" r="6" fill="#C93B3B"/><circle cx="188" cy="60" r="6" fill="#5E5191"/><circle cx="68" cy="128" r="6" fill="#5E5191"/><circle cx="68" cy="60" r="14" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="3,3"/><text class="lang-en" x="128" y="180" text-anchor="middle" fill="#C93B3B" font-size="10">stuck at 3 of 4 (ringed red is wrong)</text><text class="lang-zh" x="128" y="180" text-anchor="middle" fill="#C93B3B" font-size="10">卡在 4 把里的 3 把（被圈的红是错的）</text><text class="lang-en" x="392" y="20" text-anchor="middle" fill="#276b45" font-size="11">hidden layer: two lines bend (works)</text><text class="lang-zh" x="392" y="20" text-anchor="middle" fill="#276b45" font-size="11">hidden layer：两条线弯起来（成功）</text><rect x="297" y="30" width="190" height="132" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="325" y1="30" x2="470" y2="112" stroke="#2D8B55" stroke-width="2.5"/><line x1="315" y1="76" x2="462" y2="159" stroke="#2D8B55" stroke-width="2.5"/><circle cx="332" cy="60" r="6" fill="#C93B3B"/><circle cx="452" cy="128" r="6" fill="#C93B3B"/><circle cx="452" cy="60" r="6" fill="#5E5191"/><circle cx="332" cy="128" r="6" fill="#5E5191"/><text class="lang-en" x="392" y="180" text-anchor="middle" fill="#276b45" font-size="10">band holds both reds → 4 of 4</text><text class="lang-zh" x="392" y="180" text-anchor="middle" fill="#276b45" font-size="10">带子装住两把红的 → 4 把全对</text></g></svg>
%%%

Joining simple straight pieces really can fence off a shape neither piece could handle alone. With real ropes *you* tie the knot; in a network the "knot" is a tiny math step called the activation (tomorrow's whole lesson), and the network finds the arrangement by **training** (Days 5–6). Want to watch it happen? Flip on the **"add a hidden layer"** toggle back in the XOR widget above: the second rope appears, forms exactly this band, and all four chairs finally split.

<strong>You now know a single neuron <em>and</em> how neurons join into layers and networks to outgrow its limit.</strong> That door — stacking, plus the bend that makes stacking worth anything — is exactly where Day 2 begins.
~~~zh
把简单的直东西接起来，真的能围出一个哪一段自己都搞不定的形状。用真绳子的时候，那个结是*你*打的。在一个网络里，那个「结」是一个叫 activation 的小小数学步骤（明天一整课都讲它）。而那个摆法，是网络靠 **training** 自己找到的（第 5 到第 6 天）。想看它发生吗？回到上面那个 XOR 小工具，打开 **「加一个 hidden layer」** 那个开关：第二根绳子出现，围成正好这条带子，四把椅子终于全分开了。

<strong>你现在既懂一个 neuron，<em>也</em>懂 neuron 怎么连成 layer 和网络、长过它自己的上限。</strong>那扇门 —— 叠起来，加上那个让叠起来有意义的弯 —— 正是第 2 天开始的地方。
~~~

@@@ concept id=c12 zh_tag="它撑起了什么" tag="What it powers" zh_title="这些小 neuron 长大以后去了哪里" title="Where these little neurons grow up" zh_gotit="看到回报了" gotit="Saw the payoff"
Before we wrap, let's zoom *all* the way out — because the tiny weigh → add → decide you just learned is the exact seed inside the tech you use every day.

Picture a single **LEGO brick**. On its own it's almost nothing — a bump of plastic. Snap a few thousand together and you can build a whole castle. One neuron is that brick.
~~~zh
收尾之前，我们把镜头拉到*最*远 —— 因为你刚学的那个小小的「称一称 → 加起来 → 做决定」，正是你每天在用的技术里面那颗种子。

想象一块**乐高积木**。它自己几乎什么都不是 —— 一小块塑料凸点。把几千块拼起来，你能盖出一整座城堡。一个 neuron 就是那块积木。
~~~

%%% svg
<svg viewBox="0 0 520 206" role="img" aria-label="Six labelled tiles showing real-world uses of neural networks: image recognition (a camera), speech recognition (a person speaking), language translation (a globe), game playing (a game controller), recommendation systems (a thumbs-up), and self-driving cars (a car). In the middle sits one small neuron brick — the same brick all six are built from."><g font-family="monospace" font-size="10" text-anchor="middle"><text class="lang-en" x="260" y="16" fill="#2C2A28" font-size="12">One neuron (the brick) → the things people build with millions of them</text><text class="lang-zh" x="260" y="16" fill="#2C2A28" font-size="12">一个 neuron（那块积木）→ 人们用几百万块搭出来的东西</text><rect x="30" y="30" width="140" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="100" y="50" font-size="15">📸</text><text class="lang-en" x="100" y="68" fill="#276b45">image recognition</text><text class="lang-zh" x="100" y="68" fill="#276b45">图像识别</text><rect x="190" y="30" width="140" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="260" y="50" font-size="15">🗣️</text><text class="lang-en" x="260" y="68" fill="#5E5191">speech recognition</text><text class="lang-zh" x="260" y="68" fill="#5E5191">语音识别</text><rect x="350" y="30" width="140" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="420" y="50" font-size="15">🌐</text><text class="lang-en" x="420" y="68" fill="#8A6D3B">language translation</text><text class="lang-zh" x="420" y="68" fill="#8A6D3B">语言翻译</text><rect x="205" y="84" width="110" height="34" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="260" y="99" font-size="13">🧱</text><text class="lang-en" x="260" y="113" fill="#6B645E">one neuron</text><text class="lang-zh" x="260" y="113" fill="#6B645E">一个 neuron</text><rect x="30" y="126" width="140" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text x="100" y="146" font-size="15">🎮</text><text class="lang-en" x="100" y="164" fill="#8A6D3B">game playing</text><text class="lang-zh" x="100" y="164" fill="#8A6D3B">下棋打游戏</text><rect x="190" y="126" width="140" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text x="260" y="146" font-size="15">👍</text><text class="lang-en" x="260" y="164" fill="#5E5191">recommendations</text><text class="lang-zh" x="260" y="164" fill="#5E5191">推荐</text><rect x="350" y="126" width="140" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="420" y="146" font-size="15">🚗</text><text class="lang-en" x="420" y="164" fill="#276b45">self-driving cars</text><text class="lang-zh" x="420" y="164" fill="#276b45">自动驾驶汽车</text><text class="lang-en" x="260" y="192" fill="#9A938A" font-size="9">same brick, different castle</text><text class="lang-zh" x="260" y="192" fill="#9A938A" font-size="9">同一块积木，不同的城堡</text></g></svg>
%%%

**The LEGO brick gets it right:** one simple unit, repeated and connected in huge numbers, builds wildly different things. **One catch:** bricks are snapped together *by hand* into a fixed shape, while a network tunes itself to the job by **training** (Day 6, the training loop).

#### Six castles built from your brick
One line each. Every one of these is stacks of the neuron you now understand.

- **📸 Image recognition** — your phone unlocking at your face: early layers spot edges, later layers combine them into eyes, then whole faces.
- **🗣️ Speech recognition** — your voice assistant turning sound into words: layers pick out sound-pieces, then syllables, then words.
- **🌐 Language translation** — an app turning English into French: the network reads one sentence's meaning and writes it out in another language.
- **🎮 Game playing** — the AI that beat the world's best Go players: it looks at the board and scores which move is most likely to win.
- **👍 Recommendation systems** — the "you might also like" on a streaming app: it weighs what you've watched to guess what you'll enjoy next.
- **🚗 Self-driving cars** — a car reading the road: networks find lanes, cars and pedestrians in the camera feed and decide when to brake.

%%% insight
Here's the part that surprises everyone: not one of those six is a *different kind* of machine. Same brick, same weigh → add → decide, different stack and different training data. When someone tells you "AI for medicine" and "AI for translation" are worlds apart — they're mostly the same bricks in a different castle. You are now in on that.
%%%

#### How does it *get good* at any of this?
Every castle starts out **useless**. A fresh network's weights are random numbers, so its first guesses are basically coin-flips. Think of learning to bake: taste a too-sweet cake, use less sugar next time, taste again — repeat until it's right. A network improves in exactly that loop.
~~~zh
**乐高积木这个比喻对在哪里：** 一个简单的单元，重复很多很多次、连起来，就能搭出差别巨大的东西。**有一个地方要注意：** 积木是*用手*拼成一个固定形状的，而一个网络靠 **training** 把自己调到适合这份活（第 6 天，训练循环）。

#### 用你那块积木盖出来的六座城堡
一句话一个。这里每一个都是你现在懂的那个 neuron 叠起来的。

- **📸 图像识别** —— 你手机看到你的脸就解锁：靠前的 layer 找边缘，靠后的 layer 把它们合成眼睛，再合成整张脸。
- **🗣️ 语音识别** —— 你的语音助手把声音变成字：layer 先挑出声音的碎片，然后是音节，然后是词。
- **🌐 语言翻译** —— 一个 app 把英文变成法文：网络读出一句话的意思，再用另一种语言写出来。
- **🎮 下棋打游戏** —— 打赢了世界最强围棋选手的那个 AI：它看棋盘，给每一步「最可能赢」打分。
- **👍 推荐系统** —— 视频 app 上那个「你可能也喜欢」：它把你看过的东西称一称，猜你接下来会喜欢什么。
- **🚗 自动驾驶汽车** —— 一辆车在读路面：网络在摄像头画面里找出车道、车和行人，再决定什么时候刹车。

%%% insight
让每个人都意外的地方在这里：那六个里面，没有一个是*不同种类*的机器。同一块积木，同一个「称一称 → 加起来 → 做决定」，只是叠法不同、训练数据不同。以后有人跟你说「医疗的 AI」和「翻译的 AI」是两个世界 —— 它们大体上是同样的积木，摆成了不同的城堡。这件事你现在知道了。
%%%

#### 它是怎么*变好*的？
每一座城堡刚开始都是**没用的**。一个新网络的 weight 是随机数字，所以它最早的猜基本上等于抛硬币。想想学做蛋糕：尝一口太甜的蛋糕，下次少放糖，再尝一口 —— 重复到它对了为止。一个网络变好的过程，正好就是这个循环。
~~~

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A four-step learning loop drawn as a cycle. Step one: guess with the current weights. Step two: measure the error, how wrong the guess was. Step three: adjust the weights a little to reduce the error. Step four: repeat. An arrow loops from step four back to step one."><g font-family="monospace" font-size="10" text-anchor="middle"><text class="lang-en" x="260" y="16" fill="#2C2A28" font-size="12">How a network learns: guess → check error → adjust → repeat</text><text class="lang-zh" x="260" y="16" fill="#2C2A28" font-size="12">一个网络怎么学：猜 → 查错 → 调整 → 重复</text><rect x="30" y="55" width="90" height="44" rx="6" fill="#EFEAF7" stroke="#7C6DAA"/><text class="lang-en" x="75" y="74" fill="#5E5191">1 · guess</text><text class="lang-zh" x="75" y="74" fill="#5E5191">1 · 猜</text><text class="lang-en" x="75" y="90" fill="#9A938A" font-size="9">(current weights)</text><text class="lang-zh" x="75" y="90" fill="#9A938A" font-size="9">（现在的 weight）</text><line x1="120" y1="77" x2="160" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="160,72 170,77 160,82" fill="#B8AEA2"/><rect x="172" y="55" width="96" height="44" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="220" y="74" fill="#C93B3B">2 · check error</text><text class="lang-zh" x="220" y="74" fill="#C93B3B">2 · 查错</text><text class="lang-en" x="220" y="90" fill="#9A938A" font-size="9">how wrong?</text><text class="lang-zh" x="220" y="90" fill="#9A938A" font-size="9">错得多厉害？</text><line x1="268" y1="77" x2="308" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="308,72 318,77 308,82" fill="#B8AEA2"/><rect x="320" y="55" width="96" height="44" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="368" y="74" fill="#276b45">3 · adjust</text><text class="lang-zh" x="368" y="74" fill="#276b45">3 · 调整</text><text class="lang-en" x="368" y="90" fill="#9A938A" font-size="9">nudge weights</text><text class="lang-zh" x="368" y="90" fill="#9A938A" font-size="9">轻推 weight</text><line x1="416" y1="77" x2="452" y2="77" stroke="#B8AEA2" stroke-width="2"/><polygon points="452,72 462,77 452,82" fill="#B8AEA2"/><rect x="464" y="55" width="46" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="487" y="80" fill="#8A6D3B">4 · repeat</text><text class="lang-zh" x="487" y="80" fill="#8A6D3B">4 · 重复</text><path d="M487 99 Q 487 130 260 130 Q 75 130 75 101" fill="none" stroke="#C99A12" stroke-width="1.5" stroke-dasharray="4,3"/><polygon points="70,105 75,95 80,105" fill="#C99A12"/><text class="lang-en" x="260" y="145" fill="#9A938A" font-size="9">do it thousands of times and the guesses get good — that's training</text><text class="lang-zh" x="260" y="145" fill="#9A938A" font-size="9">做几千次，猜就会变好 —— 那就是 training</text></g></svg>
%%%

%%% steps
step: guess
why: run a forward pass with whatever weights it has now — exactly the conveyor you rode, mistakes and all
step: check the error
why: compare the guess to the right answer, therefore the network gets a number for "how wrong was I?"
step: adjust
why: nudge each weight and the bias a little in the direction that shrinks that error — the taste-less-sugar move
step: repeat, thousands of times
why: each pass is a tiny correction, which means the guesses creep from coin-flip to genuinely good
%%%

In one sentence: **training is the search for the weights and biases that make the network wrong as little as possible.**

The machinery behind each rung has its own day ahead:
- *How wrong was the guess?* → a **loss function**, **Day 4**.
- *Which way to nudge each weight?* → **gradient descent** and **backpropagation**, **Day 5** and beyond.
- *The loop that runs it all thousands of times?* → **the training loop**, **Day 6**.

Today you built the thing that does the guessing. **Next you'll sharpen its "decide" step, then teach it to teach itself.** That's the arc of this whole module — and you're standing right at the start of it.
~~~zh
%%% steps
step: 猜
why: 用它现在手上的 weight 跑一次 forward pass —— 正是你坐过的那条传送带，错也一起带上
step: 查错
why: 把这个猜和正确答案比一比，因此网络拿到一个数字，回答「我错得多厉害？」
step: 调整
why: 把每个 weight 和 bias 往让那个错变小的方向轻推一点 —— 就是那个少放糖的动作
step: 重复，几千次
why: 每一趟都是一个小小的修正，也就是说猜会从抛硬币慢慢爬到真的很好
%%%

一句话：**training 就是在找那些让网络错得尽量少的 weight 和 bias。**

每一级背后的机器，各自都有以后的一天：
- *这个猜错得多厉害？* → 一个 **loss function（损失函数）**，**第 4 天**。
- *每个 weight 该往哪边推？* → **gradient descent（梯度下降）**和 **backpropagation**，**第 5 天**往后。
- *那个把这一切跑几千遍的循环？* → **training loop（训练循环）**，**第 6 天**。

今天你搭出了那个负责猜的东西。**接下来你会把它的「做决定」磨得更利，然后教它自己教自己。** 那是整个模块的弧线 —— 而你正站在它的起点上。
~~~

@@@ concept id=c13 zh_tag="回顾" tag="Recap" zh_title="一页看完今天" title="Today in one page" zh_gotit="回顾好了" gotit="Got the recap"
Take a breath — that was a real journey.

Here's a nice way to hold it: picture yourself at the end of a hike, laying out the **postcards you collected at each stop**. At every viewpoint you picked up one small picture, and now, spread on the table, they tell the whole climb in order. (Postcards are souvenirs you put away; these are working parts you'll pick back up every day from here.)

The postcards from today's trail, in order:

- **Stop 1 · The brain cell.** A neuron is a tiny copy of a brain cell: **weigh → add → decide**, and that's the whole machine.
- **Stop 2 · Inputs & weights.** The raw numbers you feed in are the **inputs**. Each gets one **weight** — a trust dial: big = "listen," near-zero = "ignore," negative = "count against."
- **Stop 3 · Bias.** One extra number, the neuron's starting mood — a head start toward "yes," or a handicap.
- **Stop 4 · The weighted sum z.** Weigh every input, add them up, add the bias → one number `z` (the pre-activation).
- **Stop 5 · The activation (decide).** A step turns `z` into the answer. You met the hard **step** (a cliff — no direction to nudge in, so a *stack* of them can't be improved) and the smooth **sigmoid** (a dimmer, 0→1, so it can). Always *linear part, then activation.*
- **Stop 6 · The line, and its limit.** Weights + bias draw **one straight line**. That's the ceiling: one line can't do **XOR**.
- **Stop 7 · Layers, networks, depth.** A **layer** is a row of neurons on the same inputs; chain layers into a **network** (**input layer** → **hidden layer** → **output layer**) and the lines combine and bend. Many hidden layers = **deep** (**deep learning**).
- **Stop 8 · The payoff.** Millions of these bricks power image and speech recognition, translation, game playing, recommendations and self-driving cars — and they get good by **guess → check error → adjust → repeat** (training, Days 4–6).

Here's the one picture to keep — the whole neuron on one line, and where it grows into a network.
~~~zh
喘口气 —— 这真的是一趟旅程。

这里有一个好办法把它抓住：想象你在一次徒步的终点，把**每一站收集的明信片**摊开。每个观景点你都捡了一张小图，现在它们摊在桌上，按顺序讲出了整段爬升。（明信片是纪念品，你会收起来；这些是从今天起你每天都会再拿起来用的零件。）

今天这条路上的明信片，按顺序：

- **第 1 站 · 脑细胞。** 一个 neuron 是脑细胞的一份小复制品：**称一称 → 加起来 → 做决定**，而这就是整台机器。
- **第 2 站 · 输入和 weight。** 你喂进去的原始数字就是**输入**。每个输入拿到一个 **weight** —— 一个信任旋钮：大 = 「仔细听」，接近零 = 「别管它」，负的 = 「算成反对」。
- **第 3 站 · Bias。** 一个额外的数字，neuron 出门时的心情 —— 一点朝「是」的提前，或者一点吃亏。
- **第 4 站 · 加权和 z。** 每个输入称一称，加起来，再加上 bias → 一个数字 `z`（也就是 pre-activation）。
- **第 5 站 · Activation（做决定）。** 一步把 `z` 变成答案。你见了硬邦邦的 **step**（一道悬崖 —— 没有可以推的方向，所以一*叠* step 没法被改进），也见了平滑的 **sigmoid**（一个调光器，0→1，所以可以）。永远是*线性的那部分，然后 activation*。
- **第 6 站 · 那条线，和它的上限。** weight + bias 画出**一条直线**。那就是天花板：一条线做不到 **XOR**。
- **第 7 站 · Layer、网络、深度。** 一个 **layer** 是读同样输入的一排 neuron；把 layer 接成一个**网络**（**input layer** → **hidden layer** → **output layer**），线就会合起来、弯起来。很多 hidden layer = **deep**（**deep learning**）。
- **第 8 站 · 那份回报。** 几百万块这样的积木撑起了图像和语音识别、翻译、下棋打游戏、推荐和自动驾驶汽车 —— 而它们靠**猜 → 查错 → 调整 → 重复**变好（training，第 4 到第 6 天）。

这是要留住的那一张图 —— 整个 neuron 排在一行上，以及它长成一个网络的地方。
~~~

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="The complete neuron on one line: two inputs each times a weight, summed with a bias into z, then an activation turns z into the output. Below, a note that weights and bias draw one straight line, which cannot do XOR alone, and that stacking neurons into a hidden layer bends the boundary."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The whole neuron: weigh → add → decide</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">整个 neuron：称一称 → 加起来 → 做决定</text><circle cx="40" cy="55" r="7" fill="#EFEAF7" stroke="#7C6DAA"/><text x="40" y="59" text-anchor="middle" fill="#5E5191" font-size="9">x₁</text><circle cx="40" cy="100" r="7" fill="#EFEAF7" stroke="#7C6DAA"/><text x="40" y="104" text-anchor="middle" fill="#5E5191" font-size="9">x₂</text><line x1="47" y1="57" x2="150" y2="72" stroke="#B8AEA2" stroke-width="1.5"/><text x="95" y="58" fill="#6B645E" font-size="9">×w₁</text><line x1="47" y1="98" x2="150" y2="80" stroke="#B8AEA2" stroke-width="1.5"/><text x="95" y="105" fill="#6B645E" font-size="9">×w₂</text><rect x="150" y="58" width="80" height="36" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="190" y="80" text-anchor="middle" fill="#5E5191" font-size="10">Σ + b → z</text><line x1="230" y1="76" x2="290" y2="76" stroke="#B8AEA2" stroke-width="2"/><polygon points="290,70 302,76 290,82" fill="#B8AEA2"/><rect x="305" y="58" width="90" height="36" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="350" y="80" text-anchor="middle" fill="#276b45" font-size="10">activation</text><line x1="395" y1="76" x2="450" y2="76" stroke="#B8AEA2" stroke-width="2"/><polygon points="450,70 462,76 450,82" fill="#B8AEA2"/><rect x="465" y="58" width="45" height="36" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="487" y="80" text-anchor="middle" fill="#8A6D3B" font-size="10">out</text><text class="lang-zh" x="487" y="80" text-anchor="middle" fill="#8A6D3B" font-size="10">输出</text><text class="lang-en" x="260" y="132" text-anchor="middle" fill="#6B645E" font-size="10">weights + bias draw ONE straight line…</text><text class="lang-zh" x="260" y="132" text-anchor="middle" fill="#6B645E" font-size="10">weight + bias 画出一条直线……</text><text class="lang-en" x="260" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">…can't do XOR alone → stack into a hidden layer to bend it</text><text class="lang-zh" x="260" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">……单独做不到 XOR → 叠成一个 hidden layer 才能把它弯过来</text><text class="lang-en" x="260" y="168" text-anchor="middle" fill="#9A938A" font-size="9">many hidden layers = "deep" (deep learning)</text><text class="lang-zh" x="260" y="168" text-anchor="middle" fill="#9A938A" font-size="9">很多 hidden layer = 「deep」（deep learning）</text></g></svg>
%%%

#### Three myths to leave behind
- **"Neural nets think like a human brain."** No — a real brain cell is a living thing; an artificial neuron is only multiply-and-add. *Inspired by*, not the same as.
- **"They're magic black boxes."** No — every answer is *assembled* from plain steps (weigh, add, decide), exactly the ones you ran today. What they're genuinely great at is **finding patterns** in data.
- **"Bigger is always better" / "they can solve any problem."** No — one neuron provably can't even do XOR, and a network too big for its data just memorizes. The right *size* and *shape* beats raw size.

#### Cheat-sheet · the neuron's parts at a glance
One table, worth bookmarking. Every word you met today is tucked into the optional list underneath it.

%%% table
:: Part :: Everyday picture :: What it does
Input :: a raw number you hand in :: the value the neuron reads
Weight :: a trust dial (one per input) :: scales how much an input matters (negative = counts against)
Bias :: a starting mood / head start :: shifts where the neuron leans before any input
Weighted sum z :: a shopping receipt total :: every input × its weight, added, plus bias (the pre-activation)
Activation :: a judge's verdict :: turns z into the answer — step (hard 0/1) or sigmoid (smooth 0→1)
Decision boundary :: one chalk line on a playground :: the straight cut weights + bias draw
The limit :: one rope, four chairs :: one line can't do XOR (not linearly separable)
Layer & network :: a panel of judges, chained :: input → hidden → output; many hidden layers = deep
%%%

~~~html
<details style="margin:14px 0;border:1px solid #E5DFD6;border-radius:8px;padding:10px 14px;background:#FDF9F3">
<summary style="cursor:pointer;font-weight:600;color:#5A544E">Optional (skippable) · every word you met today, in one list</summary>
<div style="font-family:monospace;font-size:.86em;line-height:1.95;color:#3A342E;margin-top:8px">
<b>neuron</b> — a tiny math function that weighs its inputs, adds them up, and decides<br>
<b>input</b> — a raw number handed to the neuron; the input layer is all the raw inputs together<br>
<b>weight</b> — one number per input: how much that input matters (negative = counts against)<br>
<b>bias</b> — one extra number added to the sum: the neuron's starting lean toward "yes"<br>
<b>weighted sum (z)</b> — every input times its weight, summed, plus the bias; also called the pre-activation<br>
<b>activation function</b> — the last step that turns z into the neuron's output<br>
<b>step function</b> — the first activation: fires 1 when z reaches zero or higher, else 0<br>
<b>sigmoid</b> — a smooth S-curve that squashes z into (0, 1) — the "how confident" answer<br>
<b>decision boundary</b> — the one straight line a neuron draws to split its inputs into two sides<br>
<b>forward propagation</b> — pushing inputs left-to-right through the neuron to get the output<br>
<b>XOR</b> — "on when exactly one input is on" — the pattern one straight line can't split<br>
<b>linearly separable</b> — when one straight line can cleanly split every class; XOR is not<br>
<b>layer</b> — a row of neurons all reading the same inputs<br>
<b>hidden layer</b> — a middle layer, between inputs and answer — where the boundary bends<br>
<b>output layer</b> — the last layer, whose result is the network's final answer<br>
<b>neural network</b> — layers of neurons chained so each layer feeds the next<br>
<b>deep learning</b> — training a network with many hidden layers stacked ("deep" = many layers)<br>
<b>training</b> — slowly adjusting the weights and bias so the network makes fewer mistakes (Days 5–6)
</div>
</details>
~~~
~~~zh
#### 三个要留在身后的迷思
- **「神经网络像人脑一样思考。」** 不 —— 真的脑细胞是活的东西；一个人造 neuron 只是乘和加。是*受它启发*，不是和它一样。
- **「它们是有魔法的黑盒子。」** 不 —— 每一个答案都是用朴素的步骤（称一称、加起来、做决定）*拼出来*的，正是你今天跑过的那些。它们真正厉害的地方，是在数据里**找出花样**。
- **「越大总是越好」／「它们能解决任何问题」。** 不 —— 一个 neuron 被证明连 XOR 都做不到，而一个对它的数据来说太大的网络只会背答案。合适的*大小*和*形状*，胜过单纯的大。

#### 小抄 · 一眼看完 neuron 的零件
一张表，值得收藏。今天你见过的每一个词，都收在它下面那张选读的单子里。

%%% table
:: 零件 :: 日常的样子 :: 它做什么
输入 :: 你递进去的一个原始数字 :: neuron 读到的那个值
Weight :: 一个信任旋钮（每个输入一个） :: 缩放一个输入有多重要（负的 = 算成反对）
Bias :: 出门时的心情／提前一点 :: 挪动 neuron 在任何输入之前倒向哪里
加权和 z :: 购物收据的总额 :: 每个输入 × 它的 weight，加起来，再加 bias（也就是 pre-activation）
Activation :: 裁判的判决 :: 把 z 变成答案 —— step（硬的 0/1）或者 sigmoid（平滑的 0→1）
Decision boundary :: 操场上一条粉笔线 :: weight + bias 画出的那条直切口
那个上限 :: 一根绳子，四把椅子 :: 一条线做不到 XOR（不是 linearly separable）
Layer 与网络 :: 一排评委，接成一条链 :: input → hidden → output；很多 hidden layer = deep
%%%

#### 选读（可以跳过）· 今天你见过的每一个词，一张单子
- **neuron** —— 一个很小的数学函数：给输入称一称，加起来，然后做决定
- **input** —— 递给 neuron 的一个原始数字；input layer 就是所有原始输入合在一起
- **weight** —— 每个输入一个数字：这个输入有多重要（负的 = 算成反对）
- **bias** —— 加到那个和上面的一个额外数字：neuron 一开始朝「是」倒多少
- **weighted sum（z）** —— 每个输入乘它的 weight，加起来，再加 bias；也叫 pre-activation
- **activation function** —— 最后那一步，把 z 变成 neuron 的输出
- **step function** —— 第一个 activation：z 到零或者更高就发出 1，否则 0
- **sigmoid** —— 一条平滑的 S 形曲线，把 z 压进 (0, 1) —— 那个「有多确定」的答案
- **decision boundary** —— 一个 neuron 画出来、把输入分成两边的那条直线
- **forward propagation** —— 把输入从左到右推过 neuron，拿到输出
- **XOR** —— 「正好一个输入开的时候就开」—— 一条直线分不开的那个花样
- **linearly separable** —— 一条直线能干净地分开每一类的时候；XOR 不是
- **layer** —— 一排读同样输入的 neuron
- **hidden layer** —— 中间的一层，在输入和答案之间 —— 边界就在这里弯
- **output layer** —— 最后一层，它的结果就是网络最终的答案
- **neural network** —— 一层层 neuron 接成链，每一层喂下一层
- **deep learning** —— 训练一个叠了很多 hidden layer 的网络（「deep」= 很多层）
- **training** —— 慢慢调 weight 和 bias，让网络犯更少的错（第 5 到第 6 天）
~~~

!!! c-info 📚
<b>Want more before Day 2? (all optional further reading)</b> Nothing here is required. If today lit a spark, the friendliest next read is the free online book <b>"Neural Networks and Deep Learning"</b> by Michael Nielsen (chapter 1 covers this exact neuron), or <b>3Blue1Brown</b>'s short video series <b>"Neural Networks"</b> on YouTube for a beautiful visual tour. Skip them with a clear conscience — Day 2 assumes only what you learned right here.
!!!

That's the whole day. Next you'll zoom in on that "decide" step — the **activation** — and see the little bend that lets neurons stack into something genuinely deep.
~~~zh
!!! c-info 📚
<b>第 2 天之前想再看点什么？（全都是选读）</b> 这里没有一样是必须的。如果今天点着了你，最友好的下一本读物是免费的在线书 <b>《Neural Networks and Deep Learning》</b>，作者是 Michael Nielsen（第 1 章讲的正是这个 neuron）。另一个选择是 YouTube 上 <b>3Blue1Brown</b> 的短视频系列 <b>《Neural Networks》</b>，那是一趟很漂亮的视觉之旅。跳过它们你可以毫无负担 —— 第 2 天只用到你在这里学的东西。
!!!

这就是一整天。接下来你会放大那个「做决定」的步骤 —— **activation** —— 去看那个让 neuron 能叠成真正有深度的东西的小小的弯。
~~~

@@@ quiz id=quiz zh_tag="小测" tag="Quiz" zh_title="点一个答案 —— 每题都有即时反馈" title="Click an answer — instant feedback on each" zh_gotit="先把四题都答完" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a nudge to scroll back, not a failing grade.)
%%% quiz
q: What are the three beats a single neuron always does, in order? | a:2 | add → weigh → decide | decide → add → weigh | weigh → add → decide | weigh → decide → add | fb: A neuron scales each input by its weight (weigh), sums them with the bias (add), then turns that total into an answer (decide).
q: A neuron has two inputs. How many weights and how many biases does it have? | a:1 | 1 weight and 2 biases | 2 weights and 1 bias | 2 weights and 2 biases | 1 weight and 1 bias | fb: One weight per input (so 2), and exactly one bias for the whole neuron — the bias is a single starting-lean number, not one per input. Training adjusts all three of those numbers.
q: Why did people move from the hard step function to a smooth activation like sigmoid? | a:2 | The step used too much memory | The step outputs negative numbers | The step is a cliff, so a tiny nudge to the weights changes the output by nothing (or flips it entirely) — a smooth curve lets a small nudge make a small, followable change, which is what you need to improve a whole stack of neurons | Sigmoid is faster to compute | fb: A cliff hands back no direction to nudge in. That is survivable for one neuron (the Perceptron's mistake rule managed it), but a stack of neurons needs the smooth version — you will see exactly why on Day 5.
q: One neuron can't get all four XOR points right. What is the fix, and what do we call a network with many such stacked layers? | a:1 | Raise the learning rate | Stack neurons into a hidden layer so their lines combine into a bent boundary; a network with many hidden layers is called "deep" (deep learning) | Give the neuron more weights | Use a negative bias | fb: One neuron = one straight line, and XOR is not linearly separable. A hidden layer draws several lines that combine into a band to split all four. Stacking many hidden layers is what "deep" learning means.
%%%
~~~zh
四道快问，每题都有即时反馈 —— 四题都答完，今天就完成了。（如果有一题把你绊住了，那是提醒你往上翻，不是给你打不及格。）

%%% quiz
q: 一个 neuron 永远做的三个拍子，按顺序是什么？ | a:2 | 加起来 → 称一称 → 做决定 | 做决定 → 加起来 → 称一称 | 称一称 → 加起来 → 做决定 | 称一称 → 做决定 → 加起来 | fb: 一个 neuron 先用 weight 缩放每个输入（称一称），连着 bias 加起来（加起来），再把那个总数变成一个答案（做决定）。
q: 一个 neuron 有两个输入。它有几个 weight、几个 bias？ | a:1 | 1 个 weight 和 2 个 bias | 2 个 weight 和 1 个 bias | 2 个 weight 和 2 个 bias | 1 个 weight 和 1 个 bias | fb: 每个输入一个 weight（所以是 2 个），整个 neuron 正好一个 bias —— bias 是一个「一开始倒向哪边」的数字，不是每个输入一个。Training 会调这三个数字。
q: 人们为什么从硬邦邦的 step function 换到像 sigmoid 这样平滑的 activation？ | a:2 | step 太占内存 | step 会输出负数 | step 是一道悬崖，所以把 weight 拧一丝，输出要么完全不变、要么整个翻转。一条平滑的曲线让小拧动带来一个小的、跟得住的变化，而那正是改进一整叠 neuron 需要的东西 | sigmoid 算起来更快 | fb: 一道悬崖不会回给你任何可以推的方向。对一个 neuron 还能活（Perceptron 的「答错才改」规则撑住了），但一叠 neuron 需要平滑的版本 —— 第 5 天你会看到确切的原因。
q: 一个 neuron 没办法把四个 XOR 的点全做对。解法是什么，而一个叠了很多这种层的网络叫什么？ | a:1 | 提高 learning rate | 把 neuron 叠成一个 hidden layer，让它们的线合成一条弯的边界；一个有很多 hidden layer 的网络叫「deep」（deep learning） | 给这个 neuron 更多 weight | 用一个负的 bias | fb: 一个 neuron = 一条直线，而 XOR 不是 linearly separable。一个 hidden layer 画好几条线，合成一条带子，把四个全分开。叠很多 hidden layer，就是「deep」learning 的意思。
%%%
~~~

@@@ produce id=produce zh_tag="动手做" tag="Produce" zh_title="做一个 neuron，看它做决定" title="Build a neuron and watch it decide" zh_gotit="做完了" gotit="Done"
Time to see today with your own eyes. You'll code one neuron — weights, bias, weighted sum, activation — run a full forward pass, and then watch it *fail* on XOR so the limit is real, not just a story.

**Predict first:** for inputs `x = [2, 3]`, weights `w = [0.5, −1.0]`, and bias `b = 1.0`, what's `z`? And will `sigmoid(z)` land above or below `0.5`? Then run it and **watch.** Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-01-single-neuron/experiment.py`. Write `sigmoid(z)` and `step(z)`, then a `neuron(x, w, b)` that returns `sigmoid((w*x).sum() + b)`.

**Notice** that for `x=[2,3]`, `w=[0.5,−1.0]`, `b=1.0` you get `z = −1.0` and `sigmoid(z) ≈ 0.269` — a soft "probably no." Print `z` and the output so you can see both.

Then set up the four XOR points `[0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0` and try to find a single `w, b` that a step-neuron gets all four right — **observe** that you can never beat 3 of 4, because one neuron draws one straight line. Run with `python3 sessions/m02-the-neuron/day-01-single-neuron/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 1 artifact.

Create sessions/m02-the-neuron/day-01-single-neuron/experiment.py that, with a comment on each step:
1. Defines sigmoid(z)=1/(1+np.exp(-z)) and step(z)=(z>=0).astype(float).
2. Defines neuron(x, w, b) that computes z=(w*x).sum()+b and returns (z, sigmoid(z)). For x=[2,3], w=[0.5,-1.0], b=1.0 it should print z=-1.0 and sigmoid(z)≈0.269 (a soft "probably no").
3. Shows the single-neuron limit on XOR: the four points [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0. Loop over a grid of candidate (w1, w2, b), classify each XOR point with a step-neuron (fire 1 if z>=0), and print the BEST accuracy any single line achieves — it should top out at 3 of 4 (0.75), never 4 of 4, because one neuron draws one straight line and XOR is not linearly separable.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (a neuron deciding, then hitting its limit)
- For `x=[2,3]`, `w=[0.5,−1.0]`, `b=1.0` you print `z = −1.0` and `sigmoid(z) ≈ 0.269` — a full forward pass, ending in a soft "about 27% yes."
- Flipping the bias more positive pushes the output above `0.5`; more negative pushes it down — the bias really is the starting lean.
- The moment to **watch**: no single `(w, b)` gets all four XOR points right — the best any straight line manages is **3 of 4**. That ceiling is the whole point: one neuron = one line, and stacking neurons into a hidden layer is the way out.

!!! c-info 📓
<b>5-minute research log:</b> before you close the tab, write three lines in `sessions/m02-the-neuron/day-01-single-neuron/log.md`: (1) the three beats a neuron does and what a weight vs. a bias each controls; (2) why we prefer a smooth sigmoid over the hard step; (3) why one neuron can't do XOR, and the one-line fix (a hidden layer).
!!!
~~~zh
是时候用自己的眼睛看今天了。你会把一个 neuron 写成代码：weight、bias、加权和、activation。然后跑一次完整的 forward pass。最后看它在 XOR 上*失败*，让那个上限变成真的，不只是一个故事。

**先预测：** 对输入 `x = [2, 3]`、weight `w = [0.5, −1.0]`、bias `b = 1.0`，`z` 是多少？而 `sigmoid(z)` 会落在 `0.5` 上面还是下面？然后跑它，**看着。** 选一条路。

#### 路线 A · 自己写
建一个 `sessions/m02-the-neuron/day-01-single-neuron/experiment.py`。写 `sigmoid(z)` 和 `step(z)`，再写一个 `neuron(x, w, b)`，返回 `sigmoid((w*x).sum() + b)`。

**注意**：对 `x=[2,3]`、`w=[0.5,−1.0]`、`b=1.0`，你会得到 `z = −1.0` 和 `sigmoid(z) ≈ 0.269` —— 一个软软的「大概不是」。把 `z` 和输出都打印出来，这样两个你都看得到。

然后摆出四个 XOR 的点 `[0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0`，试着找一组 `w, b` 让一个 step-neuron 把四个全做对 —— **看着**你永远赢不过 4 个里的 3 个，因为一个 neuron 只画一条直线。用 `python3 sessions/m02-the-neuron/day-01-single-neuron/experiment.py` 跑。

#### 路线 B · 让 Claude 搭，你再读
把下面这段 prompt 复制回 Claude Code。它会让 Claude Code 建文件、写代码，并且替你跑一遍。（prompt 本身保持英文，它是给工具看的指令。）

#### 你应该看到什么（一个 neuron 在做决定，然后撞上它的上限）
- 对 `x=[2,3]`、`w=[0.5,−1.0]`、`b=1.0`，你打印出 `z = −1.0` 和 `sigmoid(z) ≈ 0.269` —— 一次完整的 forward pass，收在一个软软的「大约 27% 是」上。
- 把 bias 往正的方向翻，输出会被推到 `0.5` 以上；往负的方向翻，它会被压下去 —— bias 真的就是那个「一开始倒向哪边」。
- 要**看**的那一刻：没有任何一组 `(w, b)` 能把四个 XOR 的点全做对 —— 任何直线最好也只做到 **4 个里的 3 个**。那个天花板就是全部的重点：一个 neuron = 一条线，而把 neuron 叠成一个 hidden layer 就是出路。

!!! c-info 📓
<b>5 分钟研究笔记：</b>关掉页面之前，在 `sessions/m02-the-neuron/day-01-single-neuron/log.md` 里写三行。(1) 一个 neuron 做的三个拍子，以及 weight 和 bias 各自控制什么。(2) 为什么我们更喜欢平滑的 sigmoid 而不是硬的 step。(3) 为什么一个 neuron 做不到 XOR，以及那个一行的解法（一个 hidden layer）。
!!!
~~~

@@@ fin
