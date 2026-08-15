---
quest_id: wf2-d03-forward
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 3 — Layers & the Forward Pass"
module_label: "Module 2 · Train · Day 3"
zh_module_label: "第 2 模块 · 训练 · 第 3 天"
title: "Layers & the Forward Pass"
zh_title: "层与前向传播"
subtitle: "How Neurons Team Up to Make One Guess"
zh_subtitle: "一排工人，合起来给出一个答案"
brand_sub: "Foundations · M2 Day 3"
spine: "assembly line"
zh_spine: "流水线"
nav_prev_href: "../day-02-activations/lesson.html"
nav_prev_label: "Activation Functions"
zh_nav_prev_label: "激活函数"
nav_next_href: "../review-part-a.html"
nav_next_label: "Review Gate"
zh_nav_next_label: "复习关卡"
fin_title: "Module 2 · Day 3 complete! 🏆"
zh_fin_title: "第 2 模块 · 第 3 天 完成！🏆"
fin_body: "Nice work — you've completed <b>Layers & the Forward Pass</b>. You can now send numbers through a whole network, one layer at a time, and read a prediction off the end.<br>Next up: the <b>Review Gate</b> — a quick checkpoint on everything so far."
zh_fin_body: "做得好 —— 你完成了<b>层与前向传播</b>。现在你能把一串数字送进整个网络，一层一层往前推，最后从末端读出一个 prediction。<br>下一站：<b>复习关卡</b>，把到这里为止的东西快速过一遍。"
notebook_yardstick: 00-neural-networks/fundamentals/05_forward_propagation.ipynb
coverage_topics:
  - {topic: "Layer as a stack of parallel neurons reading the same input", keywords: ["row of neurons that all read the same input", "read the same input"]}
  - {topic: "Weight matrix W: one row per neuron, one column per input feature", keywords: ["weight matrix", "rows = neurons, columns = inputs", "row per neuron"]}
  - {topic: "Bias vector b: one bias per neuron, added after the weighted sum", keywords: ["bias vector", "one bias number per neuron", "one bias per neuron"]}
  - {topic: "Layer pre-activation z = W x + b as one matrix-vector multiply-plus-add", keywords: ["matrix-vector multiply", "z = W·x + b", "every neuron's weighted sum at once"]}
  - {topic: "Elementwise activation a = f(z): the day-02 bend applied to each entry of z", keywords: ["elementwise", "a = f(z)", "each entry of"]}
  - {topic: "Layer types by role: input, hidden, output layer", keywords: ["input layer", "hidden layer", "output layer"]}
  - {topic: "Stacking layers: output of one layer becomes input of the next; sizes must chain", keywords: ["output list of one layer becomes the input list of the next", "sizes must match", "columns = the previous layer"]}
  - {topic: "Forward pass: the ordered, one-direction chain input to prediction", keywords: ["forward pass", "one-direction trip", "one direction, fixed order"]}
  - {topic: "Dimension / shape bookkeeping: rows = neurons, columns = incoming features", keywords: ["shape bookkeeping", "rows = neurons, columns = inputs", "columns = the previous layer"]}
  - {topic: "Why a nonlinearity must sit between layers (else linear collapse); remedy = insert the day-02 activation", keywords: ["linear collapse", "bend between the layers", "put a bend"]}
  - {topic: "Depth as the source of representational power: multiple hidden layers", keywords: ["depth", "several hidden layers", "richer patterns from the last"]}
  - {topic: "Reading the network as a pipeline / composition of functions", keywords: ["composition", "functions nested inside functions", "W₂·f₁"]}
---

@@@ hero
@lede Picture a **car assembly line** in a factory. A bare metal frame rolls in one end. It stops at the first station, where a team bolts on the engine. It rolls to the next station, where another team adds the doors. Station after station, each team does one small job and passes the car along — and a finished car rolls out the far end. A neural network works exactly like this line: your numbers roll in one end, each **layer** is a station full of workers, and a finished **guess** rolls out the other end. That single trip down the line — input to answer — is called the **forward pass**, and it's the exact thing happening every time ChatGPT reads your message and begins to reply. Yesterday you met one worker (a neuron) and the little bend it adds. Today you watch a whole line of them team up.
@goal Together we'll build the assembly line one piece at a time: line up many neurons into a **layer**, learn the neat trick that runs a whole layer's math in one step, then chain layers into a network and push a real input all the way down to a prediction. You'll see what depth actually buys you, catch the two classic ways the line jams — with the simple fix for each — and by the end you'll trace a number from input to answer with your finger. Every formula shows up in plain words first. No symbol left unexplained.
%%% warmup
q: Yesterday's puzzle: you stack two straight-line layers with NO bend between them. What do you get? | a:1 | An even more powerful two-layer network | Still just one straight-line layer — the depth folds flat | A curved boundary for free | A network that runs twice as fast | concept: linear-collapse | fb: Straight ∘ straight is still straight. With no activation between them, the two layers' weights fold into one — depth buys nothing until you add a bend.
q: What does ReLU do to a number? | a:2 | Squashes it into the range 0 to 1 | Turns every number into 0 or 1 | Passes positives straight through, turns negatives into 0 (max(0, z)) | Adds a small bias to it | concept: relu | fb: ReLU = max(0, z), a one-way valve: positives flow through unchanged, negatives become 0. It's the default hidden-layer bend.
q: Why did deep sigmoid networks stall for years, while ReLU rescued them? | a:0 | Sigmoid's slope tops out near 0.25, so multiplying it through many layers shrinks the learning signal toward 0 (vanishing gradient); ReLU's positive slope is 1, so the signal survives | Sigmoid uses too much memory | ReLU has more parameters | Sigmoid can only handle small numbers | concept: vanishing-gradient | fb: The backward signal is multiplied by each layer's slope. With ≈0.25 per layer it fades fast; ReLU's slope of 1 keeps it full strength — that's why swapping to ReLU let people train deep nets.
%%%
@zh_lede 想象工厂里的一条**汽车流水线**。一个光秃秃的车架从一头滚进来。它在第一个工位停下，一队人给它装上发动机。它滚到下一个工位，另一队人装上车门。一个工位接一个工位，每队人只做一件小事，然后把车往前推 —— 一辆完整的车就从另一头开出来。神经网络就是这么一条线：你的数字从一头滚进去，每一个 **layer（层）** 是一个站满工人的工位，一个做好的**猜测**从另一头滚出来。这一趟从输入走到答案的路，叫做 **forward pass（前向传播）**。每次 ChatGPT 读你的消息、开始回你话，做的就是这件事。昨天你认识了一个工人，也就是一个 neuron（神经元），还有它加的那个小弯。今天你看一整排工人合起来干活。
@zh_goal 我们会把这条流水线一块一块搭起来：先把很多 neuron 排成一个 layer，再学一个漂亮的花招，让一整层的算术一步做完。然后把 layer 串成一个网络，把一个真实的输入一路推到 prediction（预测值）。你会看到 depth（深度）到底给你带来了什么。你还会抓到这条线卡壳的两种经典方式，每种都有一个简单的修法。到最后，你能用手指从输入一路指到答案。每个公式都先用大白话说一遍。没有一个符号不解释。

~~~zh
%%% warmup
q: 昨天的谜题：你把两个直线 layer 叠起来，中间**不**加弯。你会得到什么？ | a:1 | 一个更强的两层网络 | 还是一个直线 layer —— 深度被压平了 | 白送一条弯的边界 | 跑得快一倍的网络 | concept: linear-collapse | fb: 直线接直线还是直线。中间没有 activation（激活函数），两层的 weight（权重）会折成一层。不加弯，深度什么都换不到。
q: ReLU 对一个数字做什么？ | a:2 | 把它压到 0 和 1 之间 | 把每个数字变成 0 或 1 | 正数原样通过，负数变成 0（max(0, z)） | 给它加一点 bias（偏置） | concept: relu | fb: ReLU = max(0, z)，一个单向阀门：正数原样流过，负数变成 0。它是 hidden layer（隐藏层）默认的弯。
q: 为什么用 sigmoid 的深网络卡了好多年，而 ReLU 把它们救了回来？ | a:0 | sigmoid 的斜率最大只有 0.25 左右，一层层乘下去，学习信号会缩到接近 0，这叫 vanishing gradient（梯度消失）；ReLU 正半边的斜率是 1，信号活得下来 | sigmoid 太吃内存 | ReLU 的 parameter 更多 | sigmoid 只能处理小数字 | concept: vanishing-gradient | fb: 反向的信号会被每一层的斜率乘一次。一层大约 0.25，很快就淡掉了。ReLU 的斜率是 1，信号保持满格。这就是换成 ReLU 之后大家终于能训练深网络的原因。
%%%
~~~

@@@ concept id=c1 zh_tag="一队工人" tag="A team of workers" zh_title="一个 layer 就是一排干同样活的 neuron" title="A layer is a row of neurons doing the same job" zh_gotit="懂了 layer" gotit="Got the layer"
Let's walk up to the first station on our assembly line.

Yesterday you built one worker — one neuron. It looks at the input, does a weighted sum, adds a bias, applies its little bend. One worker, one job. Nice, but lonely.

A real station doesn't have one worker. It has a **whole row of them, side by side.**

Picture a row of graders at the front of a classroom. Every grader gets a photocopy of the *same* test. One hunts for spelling, one checks the math, one looks at the handwriting — and each hands back their own score. Same test in, many different scores out.

A [[layer||A row of neurons that all read the same input at the same time. Each neuron has its own weights and bias, so each one looks for a different pattern and produces its own number.]] is exactly that: a row of neurons that all read the same input, where each neuron has its **own** weights and its **own** bias — so each one hunts for a different pattern.
~~~zh
我们走到流水线的第一个工位前面。

昨天你搭了一个工人，也就是一个 neuron。它看一眼输入，做一次 weighted sum（加权和），加上 bias，再抹上自己那个小弯。一个工人，一件活。挺好，但是有点孤单。

真的工位不会只有一个工人。它有**一整排工人，肩并肩站着。**

想象教室前面站着一排批卷人。每个人都拿到*同一份*试卷的复印件。一个人专挑拼写，一个人核算术，还有一个人看字写得整不整齐。每个人交回自己给的分数。同一份试卷进去，很多个不同的分数出来。

一个 [[layer||一排 neuron，它们同时读同一个输入。每个 neuron 有自己的 weight 和 bias，所以每个都在找不同的模式，给出自己的数字。]] 就是这样：一排 neuron 都读同一个输入，而每个 neuron 有它**自己的** weight 和**自己的** bias。所以每个都在找不同的模式。
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A stack of photocopies of the same test is handed to a row of four graders. Each grader checks a different thing — spelling, math, neatness, grammar — and each writes back their own score."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same test → a row of graders → each returns its own score</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">同一份试卷 → 一排批卷人 → 各给自己的分</text><rect x="18" y="70" width="58" height="70" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><rect x="24" y="64" width="58" height="70" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><rect x="30" y="58" width="58" height="70" rx="4" fill="#FFFFFF" stroke="#5E5191" stroke-width="2"/><text class="lang-en" x="59" y="80" text-anchor="middle" fill="#5E5191" font-size="10">the</text><text class="lang-zh" x="59" y="80" text-anchor="middle" fill="#5E5191" font-size="10">同</text><text class="lang-en" x="59" y="96" text-anchor="middle" fill="#5E5191" font-size="10">same</text><text class="lang-zh" x="59" y="96" text-anchor="middle" fill="#5E5191" font-size="10">一份</text><text class="lang-en" x="59" y="112" text-anchor="middle" fill="#5E5191" font-size="10">test 📄</text><text class="lang-zh" x="59" y="112" text-anchor="middle" fill="#5E5191" font-size="10">试卷 📄</text><text class="lang-en" x="59" y="152" text-anchor="middle" fill="#6B645E" font-size="9">photocopies</text><text class="lang-zh" x="59" y="152" text-anchor="middle" fill="#6B645E" font-size="9">复印件</text><line x1="90" y1="55" x2="128" y2="40" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="80" x2="128" y2="78" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="105" x2="128" y2="116" stroke="#B8AEDA" stroke-width="1.3"/><line x1="90" y1="120" x2="128" y2="154" stroke="#B8AEDA" stroke-width="1.3"/><rect x="130" y="28" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="140" y="45" fill="#276b45">🧑 grader: spelling</text><text class="lang-zh" x="140" y="45" fill="#276b45">🧑 批卷人：拼写</text><text x="286" y="45" fill="#B8AEA2">→</text><text class="lang-en" x="300" y="45" fill="#6B645E">score a₁</text><text class="lang-zh" x="300" y="45" fill="#6B645E">分数 a₁</text><rect x="130" y="66" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="140" y="83" fill="#276b45">🧑 grader: math</text><text class="lang-zh" x="140" y="83" fill="#276b45">🧑 批卷人：算术</text><text x="286" y="83" fill="#B8AEA2">→</text><text class="lang-en" x="300" y="83" fill="#6B645E">score a₂</text><text class="lang-zh" x="300" y="83" fill="#6B645E">分数 a₂</text><rect x="130" y="104" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="140" y="121" fill="#276b45">🧑 grader: neatness</text><text class="lang-zh" x="140" y="121" fill="#276b45">🧑 批卷人：整洁</text><text x="286" y="121" fill="#B8AEA2">→</text><text class="lang-en" x="300" y="121" fill="#6B645E">score a₃</text><text class="lang-zh" x="300" y="121" fill="#6B645E">分数 a₃</text><rect x="130" y="142" width="150" height="26" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="140" y="159" fill="#276b45">🧑 grader: grammar</text><text class="lang-zh" x="140" y="159" fill="#276b45">🧑 批卷人：语法</text><text x="286" y="159" fill="#B8AEA2">→</text><text class="lang-en" x="300" y="159" fill="#6B645E">score a₄</text><text class="lang-zh" x="300" y="159" fill="#6B645E">分数 a₄</text><text class="lang-en" x="430" y="96" text-anchor="middle" fill="#5E5191" font-size="11">one layer =</text><text class="lang-zh" x="430" y="96" text-anchor="middle" fill="#5E5191" font-size="11">一个 layer =</text><text class="lang-en" x="430" y="112" text-anchor="middle" fill="#5E5191" font-size="11">the whole row</text><text class="lang-zh" x="430" y="112" text-anchor="middle" fill="#5E5191" font-size="11">整整一排</text></g></svg>
%%%

**What the row-of-graders picture gets right:** every grader reads the same test, yet each returns a different score, because each cares about a different thing. **Where it breaks down:** graders talk and think, while a neuron only does a fixed sum-then-bend — and graders take turns, while the neurons in a layer all fire at the *same* moment.

%%% insight
Here's why the row matters. One neuron can ask the input exactly **one** question ("does this look tall?"). A row of four asks four questions at once — and that little committee of answers is what the next station gets to work with. Every big model you've heard of is this trick, repeated: rows of workers, each asking their own question.
%%%

#### One input in, four answers out
So what actually happens when the input arrives at the station? Let's watch it, one move at a time.

%%% steps
step: hand the same input list to every neuron in the row
why: no neuron gets a special copy — this is the "photocopy of the same test" move
step: each neuron uses its OWN weights and its OWN bias
why: that's why four neurons reading identical numbers can return four different answers
step: nobody inside the row talks to anybody else
why: therefore all four can be computed at the very same time — which is exactly what makes layers fast
step: collect the four answers into one small list
why: that list is the station's finished part, ready to roll to the next station
%%%

**So far:** a layer takes in a small list of numbers and hands back a small list of numbers — one number per neuron. Four neurons in, four numbers out.

That "list of numbers" has a name: a **vector**. It is how every number travels down our assembly line, station to station.

If that felt like a lot of new words for one screen, that's normal — you only need one idea from this unit: *a layer is a row, not a worker.* Next we find a beautifully tidy place to keep all those workers' settings.
~~~zh
**批卷人这个画面对在哪里：** 每个人读的是同一份试卷，可是每个人给的分不一样，因为在意的东西不同。**它在哪里不成立：** 批卷人会说话、会思考，而一个 neuron 只做固定的「先求和、再抹弯」。批卷人是轮着来的，而一个 layer 里的 neuron 在*同一个*瞬间一起动。

%%% insight
这一排为什么重要。一个 neuron 只能问输入**一个**问题（「这个看起来高吗？」）。一排四个就同时问四个问题。这一小组答案，就是下一个工位能拿来用的东西。你听过的每个大模型都是这个花招重复很多次：一排排工人，每个问自己的那个问题。
%%%

#### 一个输入进去，四个答案出来
输入到了工位以后，到底发生了什么？我们一步一步看。

%%% steps
step: 把同一串输入交给这一排里的每一个 neuron
why: 没有哪个 neuron 拿到特别的那一份 —— 这就是「同一份试卷的复印件」那一下
step: 每个 neuron 用它自己的 weight 和自己的 bias
why: 所以四个 neuron 读一模一样的数字，也能交回四个不同的答案
step: 这一排里面谁也不跟谁说话
why: 因此四个可以在完全同一个时刻算出来，这正是 layer 快的原因
step: 把四个答案收进一小串里
why: 这一串就是这个工位做好的零件，可以往下一个工位滚了
%%%

**到这里：** 一个 layer 接进来一小串数字，交回一小串数字 —— 每个 neuron 一个数字。四个 neuron 进去，四个数字出来。

那个「一串数字」有个名字：一个 **vector（向量）**。我们流水线上的数字就是这样从一个工位走到下一个工位的。

如果你觉得一屏里的新词有点多，那很正常。这一单元你只需要带走一个想法：*一个 layer 是一排，不是一个工人。* 接下来我们找一个特别整齐的地方，把所有工人的设置都放进去。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="One input vector fans out to four neurons in a layer; each neuron reads the same input and produces its own output number"><g font-family="monospace" font-size="12">
<rect x="20" y="80" width="70" height="40" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/>
<text class="lang-en" x="55" y="98" text-anchor="middle" fill="#5E5191">input</text><text class="lang-zh" x="55" y="98" text-anchor="middle" fill="#5E5191">输入</text>
<text x="55" y="114" text-anchor="middle" fill="#5E5191">x</text>
<line x1="90" y1="100" x2="150" y2="40" stroke="#B8AEDA" stroke-width="1.5"/>
<line x1="90" y1="100" x2="150" y2="80" stroke="#B8AEDA" stroke-width="1.5"/>
<line x1="90" y1="100" x2="150" y2="120" stroke="#B8AEDA" stroke-width="1.5"/>
<line x1="90" y1="100" x2="150" y2="160" stroke="#B8AEDA" stroke-width="1.5"/>
<circle cx="180" cy="40" r="20" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="180" y="45" text-anchor="middle" fill="#1a5c38">N₁</text>
<circle cx="180" cy="80" r="20" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="180" y="85" text-anchor="middle" fill="#1a5c38">N₂</text>
<circle cx="180" cy="120" r="20" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="180" y="125" text-anchor="middle" fill="#1a5c38">N₃</text>
<circle cx="180" cy="160" r="20" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="180" y="165" text-anchor="middle" fill="#1a5c38">N₄</text>
<text class="lang-en" x="180" y="192" text-anchor="middle" fill="#6B645E" font-size="11">the layer: 4 neurons, same input each</text><text class="lang-zh" x="180" y="192" text-anchor="middle" fill="#6B645E" font-size="11">这个 layer：4 个 neuron，各读同一个输入</text>
<line x1="200" y1="40" x2="300" y2="40" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="200" y1="80" x2="300" y2="80" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="200" y1="120" x2="300" y2="120" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="200" y1="160" x2="300" y2="160" stroke="#8FBFA3" stroke-width="1.5"/>
<text class="lang-en" x="330" y="45" fill="#6B645E">→ own score a₁</text><text class="lang-zh" x="330" y="45" fill="#6B645E">→ 自己的分 a₁</text>
<text class="lang-en" x="330" y="85" fill="#6B645E">→ own score a₂</text><text class="lang-zh" x="330" y="85" fill="#6B645E">→ 自己的分 a₂</text>
<text class="lang-en" x="330" y="125" fill="#6B645E">→ own score a₃</text><text class="lang-zh" x="330" y="125" fill="#6B645E">→ 自己的分 a₃</text>
<text class="lang-en" x="330" y="165" fill="#6B645E">→ own score a₄</text><text class="lang-zh" x="330" y="165" fill="#6B645E">→ 自己的分 a₄</text>
</g></svg>
%%%

@@@ concept id=c2 zh_tag="设置表" tag="The settings sheet" zh_title="一张网格装下每个 neuron 的 weight" title="One grid holds every neuron's weights" zh_gotit="懂了 weight matrix" gotit="Got the weight matrix"
Our station has four workers, and each one carries its own little list of weights. That's four lists floating around — and a real station has hundreds. Where do we keep them all?

Think of a **school timetable pinned to the wall.** One grid. Each **row** is one student's schedule; each **column** is a time slot. Everyone's plan lives in one place, and you find any single class by looking up its row and its column.

We do exactly that for a layer. We stack every neuron's weights into one grid called the [[weight matrix||A grid of numbers holding every neuron's weights for a layer. One row per neuron, one column per input feature. Written W.]], written **W**.
~~~zh
我们的工位有四个工人，每个人都带着自己的一小串 weight。这就有四串到处飘着。真的工位有好几百个工人。这些都放在哪里？

想一想**钉在墙上的学校课程表。** 就一张网格。每一**行**是一个学生的课表，每一**列**是一个时间段。所有人的安排都在同一个地方。你要查任何一节课，只要找它的行和它的列。

我们对一个 layer 就这么做。我们把每个 neuron 的 weight 叠进一张网格。这张网格叫 [[weight matrix||一张数字网格，装着一个 layer 里所有 neuron 的 weight。一个 neuron 一行，一个输入特征一列。写成 W。]]，写成 **W**。
~~~

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A school timetable pinned to a wall as a grid. Each row is one student's schedule, each column is a time slot, and any single class is found by its row and column."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A timetable: one grid holds everyone's plan 📌</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">课程表：一张网格装下每个人的安排 📌</text><circle cx="140" cy="36" r="4" fill="#C99A12" stroke="#9A5A12"/><circle cx="380" cy="36" r="4" fill="#C99A12" stroke="#9A5A12"/><rect x="110" y="44" width="300" height="120" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="155" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">9am</text><text x="225" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">10am</text><text x="295" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">11am</text><text x="365" y="60" text-anchor="middle" fill="#9A5A12" font-size="9">12pm</text><line x1="190" y1="66" x2="190" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="260" y1="66" x2="260" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="330" y1="66" x2="330" y2="164" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="90" x2="410" y2="90" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="115" x2="410" y2="115" stroke="#E6D08A" stroke-width="1"/><line x1="110" y1="140" x2="410" y2="140" stroke="#E6D08A" stroke-width="1"/><text class="lang-en" x="104" y="82" text-anchor="end" fill="#5E5191" font-size="9">Ann →</text><text class="lang-zh" x="104" y="82" text-anchor="end" fill="#5E5191" font-size="9">小安 →</text><text class="lang-en" x="104" y="107" text-anchor="end" fill="#5E5191" font-size="9">Ben →</text><text class="lang-zh" x="104" y="107" text-anchor="end" fill="#5E5191" font-size="9">小本 →</text><text class="lang-en" x="104" y="132" text-anchor="end" fill="#5E5191" font-size="9">Cam →</text><text class="lang-zh" x="104" y="132" text-anchor="end" fill="#5E5191" font-size="9">小凯 →</text><text class="lang-en" x="104" y="157" text-anchor="end" fill="#5E5191" font-size="9">Dan →</text><text class="lang-zh" x="104" y="157" text-anchor="end" fill="#5E5191" font-size="9">小丹 →</text><text class="lang-en" x="155" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text class="lang-zh" x="155" y="82" text-anchor="middle" fill="#6B645E" font-size="9">数学</text><text class="lang-en" x="225" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text class="lang-zh" x="225" y="82" text-anchor="middle" fill="#6B645E" font-size="9">美术</text><text x="295" y="82" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text class="lang-en" x="365" y="82" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text class="lang-zh" x="365" y="82" text-anchor="middle" fill="#6B645E" font-size="9">实验</text><text class="lang-en" x="155" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text class="lang-zh" x="155" y="107" text-anchor="middle" fill="#6B645E" font-size="9">美术</text><text class="lang-en" x="225" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text class="lang-zh" x="225" y="107" text-anchor="middle" fill="#6B645E" font-size="9">数学</text><text class="lang-en" x="295" y="107" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text class="lang-zh" x="295" y="107" text-anchor="middle" fill="#6B645E" font-size="9">实验</text><text x="365" y="107" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text x="155" y="132" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text class="lang-en" x="225" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text class="lang-zh" x="225" y="132" text-anchor="middle" fill="#6B645E" font-size="9">实验</text><text class="lang-en" x="295" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text class="lang-zh" x="295" y="132" text-anchor="middle" fill="#6B645E" font-size="9">数学</text><text class="lang-en" x="365" y="132" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text class="lang-zh" x="365" y="132" text-anchor="middle" fill="#6B645E" font-size="9">美术</text><text class="lang-en" x="155" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Lab</text><text class="lang-zh" x="155" y="157" text-anchor="middle" fill="#6B645E" font-size="9">实验</text><text x="225" y="157" text-anchor="middle" fill="#6B645E" font-size="9">PE</text><text class="lang-en" x="295" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Art</text><text class="lang-zh" x="295" y="157" text-anchor="middle" fill="#6B645E" font-size="9">美术</text><text class="lang-en" x="365" y="157" text-anchor="middle" fill="#6B645E" font-size="9">Math</text><text class="lang-zh" x="365" y="157" text-anchor="middle" fill="#6B645E" font-size="9">数学</text><text class="lang-en" x="260" y="182" text-anchor="middle" fill="#6B645E" font-size="10">one row per person = one row per neuron in W</text><text class="lang-zh" x="260" y="182" text-anchor="middle" fill="#6B645E" font-size="10">一人一行 = W 里一个 neuron 一行</text></g></svg>
%%%

**What the timetable picture gets right:** one grid stores everyone's settings tidily, and each row belongs to one person. **Where it breaks down:** a timetable is only for reading, but W sits inside a live calculation — and unlike a fixed schedule, its numbers get nudged and improved during training (a later day).

#### The shape rule: rows = neurons, columns = inputs
Keep the timetable in your head and just swap the labels. Two moves, and then the payoff they hand you:

%%% steps
step: give the grid one row per neuron
why: a row IS a worker's settings, exactly like a row was one student's whole schedule
step: give the grid one column per input number
why: a column lines up with one incoming number, exactly like a column was one time slot
%%%

**So far:** those two moves fix the size of the grid. A layer of 4 neurons reading 3 numbers gets a grid 4 tall and 3 wide — written as shape `[4, 3]`, read out loud as *"four neurons, three inputs."* Every weight in the whole station now lives at one address: row for the worker, column for the input.

%%% insight
Notice what just happened: the layer stopped being a crowd of neurons and became **two objects** — one grid and one short list. That is genuinely all a layer is. When you hear that a model "has 175 billion parameters", those parameters are the numbers sitting in grids exactly like this one.
%%%

#### The bias comes along too
Each neuron also has one **bias** — a single number it adds after its weighted sum, nudging its result up or down. Stack them, and you get a short list called the [[bias vector||A short list holding one bias number for each neuron in the layer. Written b. It has the same length as the number of neurons.]], written **b**.

There is one bias per neuron, so a 4-neuron layer has exactly 4 of them — one bias number per neuron, no more, no less. Look at the tall thin box on the right of the picture below and you can count them.
~~~zh
**课程表这个画面对在哪里：** 一张网格把所有人的设置放得整整齐齐，而且每一行属于一个人。**它在哪里不成立：** 课程表只是拿来看的，而 W 坐在一个正在跑的计算里面。课程表是固定的，可是 W 里的数字会在训练时被一点点推着变好（那是后面的一天）。

#### 形状规则：行 = neuron，列 = 输入
把课程表留在脑子里，只把标签换一下。两个动作，然后是它们给你的好处：

%%% steps
step: 让这张网格一个 neuron 占一行
why: 一行就是一个工人的设置，正像一行本来是一个学生的整张课表
step: 让这张网格一个输入数字占一列
why: 一列对上一个进来的数字，正像一列本来是一个时间段
%%%

**到这里：** 这两个动作把网格的大小定下来了。一个 layer 有 4 个 neuron、读 3 个数字，它的网格就是 4 高 3 宽 —— 写成形状 `[4, 3]`，念出来是*「四个 neuron，三个输入」*。整个工位里的每一个 weight 现在都有一个地址：行是工人，列是输入。

%%% insight
注意刚刚发生了什么：这个 layer 不再是一群 neuron 了。它变成了**两个东西** —— 一张网格和一小串。一个 layer 真的就只有这些。当你听到某个模型「有 1750 亿个 parameter」，那些 parameter 就是坐在这种网格里的数字。
%%%

#### bias 也跟着来
每个 neuron 还有一个 **bias** —— 一个数字，在它的加权和之后加上去，把结果往上或往下推一点。把它们叠起来，你得到一小串，叫 [[bias vector||一小串数字，一个 layer 里每个 neuron 各占一个 bias。写成 b。它的长度和 neuron 的个数一样。]]，写成 **b**。

一个 neuron 一个 bias，所以一个 4 个 neuron 的 layer 正好有 4 个 —— 每个 neuron 一个 bias，不多也不少。看看下面这张图右边那个又高又窄的框，你可以数一数。
~~~

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A weight matrix W drawn as a 4-row, 3-column grid. Each row is one neuron's weights, each column lines up with one input feature. A bias vector b sits beside it with one bias per row."><g font-family="monospace" font-size="12">
<text class="lang-en" x="40" y="24" fill="#3A342E" font-weight="bold">W — one row per neuron, one column per input</text><text class="lang-zh" x="40" y="24" fill="#3A342E" font-weight="bold">W —— 一个 neuron 一行，一个输入一列</text>
<text x="150" y="46" text-anchor="middle" fill="#9A938A" font-size="10">x₁</text>
<text x="200" y="46" text-anchor="middle" fill="#9A938A" font-size="10">x₂</text>
<text x="250" y="46" text-anchor="middle" fill="#9A938A" font-size="10">x₃</text>
<rect x="125" y="55" width="150" height="120" rx="4" fill="none" stroke="#7C6DAA" stroke-width="2"/>
<line x1="175" y1="55" x2="175" y2="175" stroke="#D8CFEC" stroke-width="1"/>
<line x1="225" y1="55" x2="225" y2="175" stroke="#D8CFEC" stroke-width="1"/>
<line x1="125" y1="85" x2="275" y2="85" stroke="#D8CFEC" stroke-width="1"/>
<line x1="125" y1="115" x2="275" y2="115" stroke="#D8CFEC" stroke-width="1"/>
<line x1="125" y1="145" x2="275" y2="145" stroke="#D8CFEC" stroke-width="1"/>
<text x="118" y="74" text-anchor="end" fill="#2D8B55" font-size="10">N₁ →</text>
<text x="118" y="104" text-anchor="end" fill="#2D8B55" font-size="10">N₂ →</text>
<text x="118" y="134" text-anchor="end" fill="#2D8B55" font-size="10">N₃ →</text>
<text x="118" y="164" text-anchor="end" fill="#2D8B55" font-size="10">N₄ →</text>
<text x="150" y="74" text-anchor="middle" fill="#5E5191">.2</text><text x="200" y="74" text-anchor="middle" fill="#5E5191">-.1</text><text x="250" y="74" text-anchor="middle" fill="#5E5191">.5</text>
<text x="150" y="104" text-anchor="middle" fill="#5E5191">.0</text><text x="200" y="104" text-anchor="middle" fill="#5E5191">.3</text><text x="250" y="104" text-anchor="middle" fill="#5E5191">-.1</text>
<text x="150" y="134" text-anchor="middle" fill="#5E5191">.4</text><text x="200" y="134" text-anchor="middle" fill="#5E5191">.1</text><text x="250" y="134" text-anchor="middle" fill="#5E5191">.0</text>
<text x="150" y="164" text-anchor="middle" fill="#5E5191">-.3</text><text x="200" y="164" text-anchor="middle" fill="#5E5191">.2</text><text x="250" y="164" text-anchor="middle" fill="#5E5191">.6</text>
<text class="lang-en" x="200" y="200" text-anchor="middle" fill="#6B645E" font-size="11">shape [4 neurons, 3 inputs]</text><text class="lang-zh" x="200" y="200" text-anchor="middle" fill="#6B645E" font-size="11">形状 [4 个 neuron, 3 个输入]</text>
<rect x="340" y="55" width="60" height="120" rx="4" fill="#FDF6EC" stroke="#9A5A12" stroke-width="2"/>
<text x="370" y="46" text-anchor="middle" fill="#9A5A12" font-size="11">b</text>
<text x="370" y="74" text-anchor="middle" fill="#9A5A12">0.0</text><text x="370" y="104" text-anchor="middle" fill="#9A5A12">-0.5</text><text x="370" y="134" text-anchor="middle" fill="#9A5A12">0.0</text><text x="370" y="164" text-anchor="middle" fill="#9A5A12">-2.0</text>
<text class="lang-en" x="412" y="118" fill="#6B645E" font-size="11">one bias</text><text class="lang-zh" x="412" y="118" fill="#6B645E" font-size="11">每个 neuron</text><text class="lang-en" x="412" y="134" fill="#6B645E" font-size="11">per neuron</text><text class="lang-zh" x="412" y="134" fill="#6B645E" font-size="11">一个 bias</text>
</g></svg>
%%%

Those exact numbers are our running example for the whole day — one grid `W`, one bias list `b`. We'll feed them the same input `x = [1, 2, 3]` in every unit from here on, so you can always check the page against your own arithmetic.

Hold onto that shape rule — *rows = neurons, columns = inputs* — because later today it becomes the safety check that keeps the whole line from jamming.
~~~zh
这些数字就是我们今天一整天要用的例子 —— 一张网格 `W`，一串 bias `b`。从这里开始，每一单元我们都喂给它们同一个输入 `x = [1, 2, 3]`。这样你随时可以拿自己算的结果跟页面对一下。

把那条形状规则记住 —— *行 = neuron，列 = 输入*。今天晚一点，它会变成让整条线不卡壳的安全检查。
~~~

@@@ concept id=c3 zh_tag="一步，不是很多步" tag="One step, not many" zh_title="z = W·x + b —— 整个工位一步做完" title="z = W·x + b — the whole station in one move" zh_gotit="懂了矩阵那一步" gotit="Got the matrix step"
This is the moment the assembly line gets its superpower.

On Day 1, each neuron did its own weighted sum, one at a time: worker 1 multiplies and adds, then worker 2, then worker 3. Fine for three workers. A real layer can have *thousands*, and one-at-a-time would crawl.

Picture a **self-checkout scanner** versus adding up prices in your head. In your head you go "$2… plus $3… plus $1…" item by item. The scanner reads the whole cart and gives you one total in a single beep.

Math has that same "one beep" move for our grid. It is called a [[matrix-vector multiply||A single operation that multiplies a grid of numbers (the weight matrix) by a list of numbers (the input) to produce a new list — computing every neuron's weighted sum at once.]]: you hand it the whole grid **W** and the whole input list **x**, and in one move it gives you every neuron's weighted sum at once. Then you add the bias list **b**. The result is a list we call **z**.

In one line: **z = W·x + b**. Out loud, in plain words: *"multiply the weight grid by the input list, then add the bias list."*
~~~zh
流水线的超能力就在这一刻出现。

第 1 天，每个 neuron 各自做加权和，一个接一个：工人 1 乘完再加，然后工人 2，然后工人 3。三个工人这样还行。真的 layer 可以有*好几千*个工人，一个一个来会慢得像爬。

想象**超市的自助扫码机**，再对比你在心里加价钱。在心里你是「$2…加 $3…加 $1…」一件一件来。扫码机把整车读完，一声「哔」就给你一个总数。

数学对我们的网格也有这个「一声哔」的动作。它叫 [[matrix-vector multiply||一个单独的运算，把一张数字网格（weight matrix）乘上一串数字（输入），得到一串新数字 —— 一次算出每个 neuron 的加权和。]]：你把整张网格 **W** 和整串输入 **x** 交给它，它一步就给你每个 neuron 的加权和。然后你加上 bias 那一串 **b**。结果是一串，我们叫它 **z**。

写成一行：**z = W·x + b**。念出来的大白话是：*「把 weight 网格乘上输入那一串，然后加上 bias 那一串。」*
~~~

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A self-checkout scanner reads a whole shopping cart in one beep and prints one total, next to a slow person adding item prices one at a time in their head."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Scan the whole cart in one BEEP → one total</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">整车一声「哔」扫完 → 一个总数</text><rect x="18" y="34" width="230" height="58" rx="8" fill="#FDECEC" stroke="#E0A9A0"/><text class="lang-en" x="133" y="52" text-anchor="middle" fill="#8a3b3b">slow way 😓</text><text class="lang-zh" x="133" y="52" text-anchor="middle" fill="#8a3b3b">慢办法 😓</text><text class="lang-en" x="133" y="70" text-anchor="middle" fill="#9a6b64" font-size="10">add prices one item at a time</text><text class="lang-zh" x="133" y="70" text-anchor="middle" fill="#9a6b64" font-size="10">一件一件加价钱</text><text class="lang-en" x="133" y="84" text-anchor="middle" fill="#9a6b64" font-size="10">$2 … +$3 … +$1 … in your head</text><text class="lang-zh" x="133" y="84" text-anchor="middle" fill="#9a6b64" font-size="10">心里算 $2 … +$3 … +$1 …</text><rect x="18" y="104" width="230" height="64" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="64" y="124" text-anchor="middle" fill="#276b45">🛒 cart</text><text class="lang-zh" x="64" y="124" text-anchor="middle" fill="#276b45">🛒 购物车</text><rect x="110" y="114" width="46" height="44" rx="6" fill="#FFFFFF" stroke="#2D8B55" stroke-width="2"/><text x="133" y="133" text-anchor="middle" font-size="14">📷</text><text class="lang-en" x="133" y="150" text-anchor="middle" fill="#276b45" font-size="9">scan</text><text class="lang-zh" x="133" y="150" text-anchor="middle" fill="#276b45" font-size="9">扫</text><text class="lang-en" x="176" y="140" fill="#2D8B55" font-size="14">BEEP!</text><text class="lang-zh" x="176" y="140" fill="#2D8B55" font-size="14">哔！</text><text class="lang-en" x="133" y="166" text-anchor="middle" fill="#276b45" font-size="9">the fast way</text><text class="lang-zh" x="133" y="166" text-anchor="middle" fill="#276b45" font-size="9">快办法</text><text x="268" y="110" fill="#B8AEA2" font-size="16">→</text><rect x="292" y="104" width="210" height="64" rx="8" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="397" y="128" text-anchor="middle" fill="#9A5A12">🧾 one TOTAL</text><text class="lang-zh" x="397" y="128" text-anchor="middle" fill="#9A5A12">🧾 一个总数</text><text class="lang-en" x="397" y="148" text-anchor="middle" fill="#9A5A12" font-size="10">every neuron's sum</text><text class="lang-zh" x="397" y="148" text-anchor="middle" fill="#9A5A12" font-size="10">每个 neuron 的和</text><text class="lang-en" x="397" y="162" text-anchor="middle" fill="#9A5A12" font-size="10">at once = z = W·x + b</text><text class="lang-zh" x="397" y="162" text-anchor="middle" fill="#9A5A12" font-size="10">一次算完 = z = W·x + b</text></g></svg>
%%%

**What the self-checkout picture gets right:** one action handles the whole cart at once, just as one multiply handles every neuron at once. **Where it breaks down:** the scanner only sums prices, while the multiply first pairs each input with each neuron's weight and *then* sums — a weighted total, not a plain one.

Now go play with it, because this widget lets you look *inside* the beep. Two things to try, in this order:

1. **Hover or tap any green cell** on the right-hand grid. The row of `A` and the column of `B` that build that one cell light up — that is a single neuron's weighted sum, assembling in front of you.
2. **Drag the `n` slider down to 1.** Now `B` is a single column — exactly our input list `x` — and the whole right-hand grid becomes one column of totals: one number per neuron. That is `W·x`, live.
~~~zh
**扫码机这个画面对在哪里：** 一个动作一次搞定整车，就像一次乘法一次搞定每个 neuron。**它在哪里不成立：** 扫码机只是把价钱加起来，而这个乘法先把每个输入和每个 neuron 的 weight 配对，*然后*才求和 —— 是加权的总数，不是简单的总数。

现在去玩一下。这个小工具让你看到那声「哔」*里面*发生了什么。按这个顺序试两件事：

1. **把鼠标停在右边网格里任何一个绿格上，或者点它。** 拼出这一格的那一行 `A` 和那一列 `B` 会亮起来 —— 那就是一个 neuron 的加权和，在你眼前拼出来。
2. **把 `n` 滑块拖到 1。** 现在 `B` 只剩一列 —— 正是我们的输入 `x` —— 右边整张网格变成一列总数：每个 neuron 一个数字。那就是 `W·x`，活的。
~~~

%%% viz src=../../viz/matmul.html title="a matrix times a vector, cell by cell" caption="interactive — hover or tap any green cell to light up the row and column that build it; drag the n slider to 1 so B is a single column, our input x, and each green cell is one neuron's weighted sum"
%%%

#### Watch one number get born
Before we trust the beep, let's do one neuron by hand — with the real grid from the last unit. Our input is `x = [1, 2, 3]`, and neuron 1 is the **first row** of `W`: `[.2, -.1, .5]`.

%%% steps
step: pair each input with its matching weight
why: 1 goes with .2, 2 goes with −.1, and 3 goes with .5 — the row lines up with the input, slot by slot
step: multiply each pair: 1 × .2 = .2, then 2 × (−.1) = −.2, then 3 × .5 = 1.5
why: a big weight means "this input matters a lot to me"; a negative weight means "this counts against me"
step: add the pieces up: .2 + (−.2) + 1.5 = 1.5
why: that single number is neuron 1's weighted sum — one worker's whole opinion, in one number
step: now do that for EVERY row of the grid, in the same instant
why: that's the beep — the same little sum as before, one per neuron, all at once
%%%

**So far:** nothing new has been invented. The matrix multiply is the Day-1 weighted sum, done for every row of W at the same time.

%%% insight
Why does anyone care that it's *one* move instead of a loop? Because "one move" is a shape that graphics chips (GPUs) were built to eat. The same numbers, hundreds of times faster. That is the real reason big models are trained as stacks of matrix multiplies and not as loops over neurons.
%%%

#### Predict, then run it
Here's the full `[4, 3]` grid meeting `x = [1, 2, 3]` in one beep. This is `W·x` alone — the pure multiply, **no bias added yet** — so you can check every number straight from the grid you just saw. You already hand-computed the first one.

%%% demo id=wx label="run W·x (the multiply, no bias yet)"
predict: you got 1.5 for neuron 1 by hand. How MANY numbers come out of the beep in total — and will your 1.5 be one of them?
code: W = np.array([[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]])   # 4 neurons, 3 inputs
code: x = np.array([1., 2., 3.])
code: W @ x
out: array([1.5, 0.3, 0.6, 1.9])
take: <b>One multiply, four weighted sums born.</b> Your hand-computed 1.5 is sitting right there in slot 1. The whole `[4, 3]` grid met the length-3 input and produced one weighted sum per neuron — a length-4 list, in a single move. Add the bias list `b` on top and you have the full `z = W·x + b`.
%%%

!!! c-info 🪜
<b>Optional (skippable) — reading the shapes.</b> The multiply only works when the shapes line up. A <code>[4, 3]</code> weight grid (4 neurons, 3 inputs) times a length-3 input list gives a length-4 result — one number per neuron. In shorthand: <code>[4, 3] · [3] → [4]</code>. The middle "3" is shared and cancels out; the outer numbers tell you the size going in and the size coming out. This becomes a real safety check later today.
!!!

That list is still raw, and the bias hasn't even landed yet. Remember yesterday's bend? Both are about to arrive — one per worker.
~~~zh
#### 看一个数字诞生
在相信那声「哔」之前，我们先手算一个 neuron —— 用上一单元那张真的网格。我们的输入是 `x = [1, 2, 3]`，neuron 1 是 `W` 的**第一行**：`[.2, -.1, .5]`。

%%% steps
step: 把每个输入和它对应的 weight 配成一对
why: 1 配 .2，2 配 −.1，3 配 .5 —— 这一行和输入一格一格对上
step: 每一对相乘：1 × .2 = .2，然后 2 × (−.1) = −.2，然后 3 × .5 = 1.5
why: weight 大表示「这个输入对我很重要」；weight 是负的表示「这个对我是减分的」
step: 把这几块加起来：.2 + (−.2) + 1.5 = 1.5
why: 这一个数字就是 neuron 1 的加权和 —— 一个工人的全部意见，缩成一个数字
step: 现在对网格的每一行都这么做，在同一个瞬间
why: 那就是那声「哔」—— 还是刚才那个小小的求和，每个 neuron 一次，全部同时
%%%

**到这里：** 没有发明任何新东西。矩阵乘法就是第 1 天的加权和，只是对 W 的每一行同时做。

%%% insight
为什么大家在意它是*一*个动作，而不是一个循环？因为「一个动作」正好是显卡（GPU）生来就爱吃的形状。同样的数字，快上几百倍。这就是大模型被写成一堆矩阵乘法、而不是对 neuron 做循环的真正原因。
%%%

#### 先猜，再运行
下面是完整的 `[4, 3]` 网格和 `x = [1, 2, 3]` 在一声「哔」里相遇。这里只有 `W·x` —— 纯粹的乘法，**还没有加 bias** —— 所以每个数字你都可以直接对着刚看过的网格核一遍。第一个你已经手算过了。

%%% demo id=wxzh label="只有 W·x，还没加 bias —— 先猜，再展开"
predict: 你手算 neuron 1 得到 1.5。那声「哔」一共出来几个数字 —— 你的 1.5 会在里面吗？
code: W = np.array([[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]])   # 4 个 neuron，3 个输入
code: x = np.array([1., 2., 3.])
code: W @ x
out: array([1.5, 0.3, 0.6, 1.9])
take: <b>一次乘法，四个加权和诞生。</b>你手算的 1.5 就坐在第 1 格。整张 `[4, 3]` 网格遇上长度 3 的输入，产出每个 neuron 一个加权和 —— 一串长度 4，一步完成。再把 bias 那一串 `b` 加上去，你就有了完整的 `z = W·x + b`。
%%%

!!! c-info 🪜
<b>选读（可以跳过）—— 怎么读形状。</b>只有形状对上，这个乘法才成立。一张 <code>[4, 3]</code> 的 weight 网格（4 个 neuron，3 个输入）乘一串长度 3 的输入，得到一个长度 4 的结果 —— 每个 neuron 一个数字。简写成：<code>[4, 3] · [3] → [4]</code>。中间那个「3」是共用的，会抵消掉。外面那两个数字告诉你进去多大、出来多大。今天晚一点，这会变成一个真正的安全检查。
!!!

这一串还是生的，而 bias 甚至还没落下来。还记得昨天那个弯吗？两样马上都要到了 —— 每个工人一份。
~~~

@@@ concept id=c4 zh_tag="每个工人自己的弯" tag="The bend, per worker" zh_title="a = f(z) —— 每个工人加上自己的弯" title="a = f(z) — each worker adds their bend" zh_gotit="懂了 activation 那一步" gotit="Got the activation step"
The beep gave us four raw weighted sums. Two small things still have to happen before the part can roll on: each worker adds its own bias, and then each worker adds its own bend.

Picture a row of **rubber stamps.** The multiply laid out a row of raw forms (the numbers in `z`). Now each form gets stamped — the same stamp, pressed onto each form separately. The stamp never mixes two forms together. It just touches one, then the next.

That "same rule, applied to each entry on its own" is what we mean by [[elementwise||Applied to each number in a list separately, one at a time, with no mixing between them. The bend f runs on each entry of z independently.]]. We take the bend `f` — yesterday's ReLU, or sigmoid, or tanh — and run it on each entry of `z` by itself. The result is the layer's real output list, which we call **a**.

In one line: **a = f(z)**. In plain words: *"apply the bend to every number in z, one at a time."*
~~~zh
那声「哔」给了我们四个生的加权和。零件往下滚之前，还有两件小事要做：每个工人加上自己的 bias，然后每个工人抹上自己的弯。

想象一排**橡皮印章。** 乘法把一排生的表格摊开了，就是 `z` 里那些数字。现在每张表格都要被盖一下 —— 同一个印章，一张一张分开盖。印章从来不会把两张表格搅在一起。它只是碰一张，再碰下一张。

「同一条规则，分别用在每一格上」，我们就叫它 [[elementwise||分别用在一串里的每个数字上，一次一个，彼此不混。弯 f 独立地作用在 z 的每一格上。]]。我们拿那个弯 `f` —— 昨天的 ReLU，或者 sigmoid，或者 tanh —— 让它单独作用在 `z` 的每一格上。结果就是这个 layer 真正的输出那一串，我们叫它 **a**。

写成一行：**a = f(z)**。大白话：*「把弯用在 z 里每个数字上，一次一个。」*
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A row of four raw paper forms, each getting the same rubber stamp pressed onto it separately. One stamp per form, no form touches another — the same rule applied to each on its own."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same stamp, pressed on each form on its own</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">同一个印章，一张一张单独盖</text><text class="lang-en" x="55" y="38" text-anchor="middle" fill="#6B645E" font-size="10">raw forms (z)</text><text class="lang-zh" x="55" y="38" text-anchor="middle" fill="#6B645E" font-size="10">生的表格（z）</text><rect x="30" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="98" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="166" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><rect x="234" y="46" width="52" height="66" rx="3" fill="#FFFFFF" stroke="#7C6DAA" stroke-width="1.6"/><line x1="38" y1="58" x2="74" y2="58" stroke="#D8CFEC"/><line x1="38" y1="70" x2="74" y2="70" stroke="#D8CFEC"/><line x1="106" y1="58" x2="142" y2="58" stroke="#D8CFEC"/><line x1="106" y1="70" x2="142" y2="70" stroke="#D8CFEC"/><line x1="174" y1="58" x2="210" y2="58" stroke="#D8CFEC"/><line x1="174" y1="70" x2="210" y2="70" stroke="#D8CFEC"/><line x1="242" y1="58" x2="278" y2="58" stroke="#D8CFEC"/><line x1="242" y1="70" x2="278" y2="70" stroke="#D8CFEC"/><rect x="38" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="56" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="106" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="124" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="174" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="192" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><rect x="242" y="84" width="36" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text x="260" y="99" text-anchor="middle" fill="#2D8B55" font-size="12">✓</text><text x="56" y="128" text-anchor="middle" font-size="18">🔨</text><text x="124" y="128" text-anchor="middle" font-size="18">🔨</text><text x="192" y="128" text-anchor="middle" font-size="18">🔨</text><text x="260" y="128" text-anchor="middle" font-size="18">🔨</text><text class="lang-en" x="158" y="148" text-anchor="middle" fill="#6B645E" font-size="9">the SAME stamp f, one per form — no form touches another</text><text class="lang-zh" x="158" y="148" text-anchor="middle" fill="#6B645E" font-size="9">同一个印章 f，一张一个 —— 表格之间互不相碰</text><rect x="330" y="56" width="172" height="56" rx="8" fill="#EFEAF7" stroke="#7C6DAA"/><text class="lang-en" x="416" y="80" text-anchor="middle" fill="#5E5191">elementwise:</text><text class="lang-zh" x="416" y="80" text-anchor="middle" fill="#5E5191">逐元素：</text><text class="lang-en" x="416" y="98" text-anchor="middle" fill="#5E5191" font-size="10">a = f(z), each entry alone</text><text class="lang-zh" x="416" y="98" text-anchor="middle" fill="#5E5191" font-size="10">a = f(z)，每一格单独来</text></g></svg>
%%%

**What the row-of-stamps picture gets right:** the same stamp touches each form separately and changes none of the others. **Where it breaks down:** a stamp always leaves the same mark, but the bend's *result* depends on the number underneath — a `−1` becomes `0`, a `2.4` stays `2.4`. The rule is fixed; the outcome is not.

#### Finish the station with our own numbers
We left the beep holding `W·x = [1.5, 0.3, 0.6, 1.9]`. Now the bias lands, and then the stamp. Watch what the bias quietly does to two of the workers.

%%% steps
step: add the bias list — z = [1.5, 0.3, 0.6, 1.9] + [0.0, −0.5, 0.0, −2.0] = [1.5, −0.2, 0.6, −0.1]
why: each worker gets ITS own bias, and worker 4's big −2.0 just dragged a healthy 1.9 below zero
step: entry 1 is 1.5 → ReLU gives 1.5
why: ReLU is max(0, z), so a positive passes straight through, untouched — worker 1 is shouting and gets heard
step: entry 2 is −0.2 → ReLU gives 0.0
why: therefore a negative gets flattened to zero — worker 2 stays quiet on this input
step: entry 3 is 0.6 → ReLU gives 0.6
why: notice worker 1 being loud did NOT rescue worker 2; each entry of z is judged completely alone
step: entry 4 is −0.1 → ReLU gives 0.0
why: four entries in, four entries out — the bend never changes the length of the list
%%%

**So far:** the station's finished part is `a = [1.5, 0.0, 0.6, 0.0]`. Here is that exact ladder, drawn — raw list on the left, bent list on the right.
~~~zh
**印章这个画面对在哪里：** 同一个印章分别碰每一张表格，不改动其他任何一张。**它在哪里不成立：** 印章留下的印记永远一样，可是这个弯的*结果*取决于下面那个数字 —— 一个 `−1` 变成 `0`，一个 `2.4` 还是 `2.4`。规则是固定的，结果不是。

#### 用我们自己的数字把工位做完
那声「哔」留给我们的是 `W·x = [1.5, 0.3, 0.6, 1.9]`。现在 bias 落下来，然后印章盖上去。看 bias 悄悄对其中两个工人做了什么。

%%% steps
step: 加上 bias 那一串 —— z = [1.5, 0.3, 0.6, 1.9] + [0.0, −0.5, 0.0, −2.0] = [1.5, −0.2, 0.6, −0.1]
why: 每个工人拿到「他自己的」bias，工人 4 那个大大的 −2.0 把健康的 1.9 拽到了零以下
step: 第 1 格是 1.5 → ReLU 给 1.5
why: ReLU 是 max(0, z)，所以正数原样直接通过 —— 工人 1 在喊，而且被听见了
step: 第 2 格是 −0.2 → ReLU 给 0.0
why: 因此负数被压平成零 —— 工人 2 对这个输入保持沉默
step: 第 3 格是 0.6 → ReLU 给 0.6
why: 注意工人 1 喊得响，并没有救到工人 2；z 的每一格都是完全单独判的
step: 第 4 格是 −0.1 → ReLU 给 0.0
why: 四格进去，四格出来 —— 这个弯从不改变这一串的长度
%%%

**到这里：** 这个工位做好的零件是 `a = [1.5, 0.0, 0.6, 0.0]`。下面就是这道台阶画出来的样子 —— 左边是生的那一串，右边是抹过弯的那一串。
~~~

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A list z of four numbers 1.5, -0.2, 0.6, -0.1, each passed separately through a ReLU bend, producing the output list a of 1.5, 0, 0.6, 0. Negatives become zero, positives pass through unchanged."><g font-family="monospace" font-size="12">
<text class="lang-en" x="55" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold">z (raw)</text><text class="lang-zh" x="55" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold">z（生的）</text>
<rect x="20" y="40" width="70" height="110" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/>
<text x="55" y="66" text-anchor="middle" fill="#5E5191">1.5</text>
<text x="55" y="92" text-anchor="middle" fill="#5E5191">-0.2</text>
<text x="55" y="118" text-anchor="middle" fill="#5E5191">0.6</text>
<text x="55" y="144" text-anchor="middle" fill="#5E5191">-0.1</text>
<text class="lang-en" x="200" y="26" text-anchor="middle" fill="#1a5c38" font-weight="bold">apply bend f, each on its own</text><text class="lang-zh" x="200" y="26" text-anchor="middle" fill="#1a5c38" font-weight="bold">抹上弯 f，每一格单独来</text>
<line x1="95" y1="60" x2="300" y2="60" stroke="#8FBFA3" stroke-width="1.5"/><text x="200" y="55" text-anchor="middle" fill="#2D8B55" font-size="10">ReLU: max(0, z)</text>
<line x1="95" y1="86" x2="300" y2="86" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="95" y1="112" x2="300" y2="112" stroke="#8FBFA3" stroke-width="1.5"/>
<line x1="95" y1="138" x2="300" y2="138" stroke="#8FBFA3" stroke-width="1.5"/>
<text class="lang-en" x="380" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold">a (output)</text><text class="lang-zh" x="380" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold">a（输出）</text>
<rect x="345" y="40" width="70" height="110" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text x="380" y="66" text-anchor="middle" fill="#1a5c38">1.5</text>
<text x="380" y="92" text-anchor="middle" fill="#1a5c38">0.0</text>
<text x="380" y="118" text-anchor="middle" fill="#1a5c38">0.6</text>
<text x="380" y="144" text-anchor="middle" fill="#1a5c38">0.0</text>
<text class="lang-en" x="440" y="92" fill="#C93B3B" font-size="10">← flattened</text><text class="lang-zh" x="440" y="92" fill="#C93B3B" font-size="10">← 被压平</text>
<text class="lang-en" x="440" y="144" fill="#C93B3B" font-size="10">← flattened</text><text class="lang-zh" x="440" y="144" fill="#C93B3B" font-size="10">← 被压平</text>
<text class="lang-en" x="260" y="172" text-anchor="middle" fill="#6B645E" font-size="10">negatives → 0, positives pass — one entry at a time</text><text class="lang-zh" x="260" y="172" text-anchor="middle" fill="#6B645E" font-size="10">负数 → 0，正数通过 —— 一格一格来</text>
</g></svg>
%%%

%%% insight
Here's the clean split worth remembering. The multiply is where neurons **mix** the input together. The bend does the opposite — it touches each score **alone**, mixing nothing. That's why the two-move recipe `z = W·x + b`, then `a = f(z)`, describes *every* dense layer you will ever build, in any framework, for the rest of your life. Learn it once, reuse it forever.
%%%

Two of our four workers just went silent on this input, and that is completely normal — a layer is a committee where only the relevant members speak up. (If a worker went quiet on *every* input, that would be a real problem with a real name, and you'll meet it on a later day.)

You now own one complete station: mix, then bend. Next we snap stations together into a full line.
~~~zh
%%% insight
这里有一个值得记住的干净分工。乘法是 neuron 把输入**搅在一起**的地方。弯正好相反 —— 它**单独**碰每一个分数，什么都不搅。所以这两步配方 `z = W·x + b`，然后 `a = f(z)`，描述了你今后会搭的*每一个* dense layer，在任何框架里，一辈子都是这样。学一次，用一辈子。
%%%

我们四个工人里有两个对这个输入哑了，这完全正常 —— 一个 layer 是一个委员会，只有相关的成员才发言。（如果一个工人对*每一个*输入都哑，那才是真的问题。它有一个真的名字，你会在后面的一天见到它。）

你现在拥有一个完整的工位：先搅，再抹弯。接下来我们把工位拼成一整条线。
~~~

@@@ concept id=c5 zh_tag="把它们拼起来" tag="Snap them together" zh_title="input、hidden、output —— 它们怎么接上" title="Input, hidden, output — and how they chain" zh_gotit="懂了堆叠规则" gotit="Got the stacking rule"
One station is nice. A factory has *many* — and this is where our assembly line finally comes alive.

Picture the part rolling off station 1 straight onto station 2's bench, then onto station 3, all the way down. Each station takes whatever the last one handed it, does its own work, and passes it forward. Nothing skips ahead. Nothing waits.

Feel that flow first, because *the flow is the whole idea.* All we do now is give the stations their names.
~~~zh
一个工位挺好。一座工厂有*很多*个 —— 我们的流水线终于要活起来了。

想象零件从工位 1 滚下来，直接滚到工位 2 的台子上，再滚到工位 3，一路往下。每个工位接过上一个交给它的东西，做自己的活，然后往前传。没有谁抢先跳过去。也没有谁在等着。

先感受这个流动，因为*这个流动就是整个想法。* 我们现在做的只是给这些工位起名字。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A car assembly line of three stations connected by a conveyor belt. A bare frame rolls in, station one adds a part, passes it along the belt to station two, then station three, and a finished car rolls off the end."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Parts roll bench → bench, one way, station to station</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">零件从一张台子滚到下一张，一个方向，工位接工位</text><rect x="20" y="90" width="470" height="18" rx="9" fill="#ECE6DC" stroke="#C0B8AC"/><circle cx="45" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="120" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="200" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="290" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="380" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><circle cx="465" cy="99" r="6" fill="#FFF" stroke="#9A938A"/><text x="36" y="66" text-anchor="middle" font-size="16">🔩</text><text class="lang-en" x="36" y="84" text-anchor="middle" fill="#5E5191" font-size="9">frame in</text><text class="lang-zh" x="36" y="84" text-anchor="middle" fill="#5E5191" font-size="9">车架进</text><rect x="96" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text class="lang-en" x="135" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 1</text><text class="lang-zh" x="135" y="60" text-anchor="middle" fill="#276b45" font-size="9">工位 1</text><text class="lang-en" x="135" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ engine</text><text class="lang-zh" x="135" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ 发动机</text><rect x="196" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text class="lang-en" x="235" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 2</text><text class="lang-zh" x="235" y="60" text-anchor="middle" fill="#276b45" font-size="9">工位 2</text><text class="lang-en" x="235" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ doors</text><text class="lang-zh" x="235" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ 车门</text><rect x="296" y="44" width="78" height="40" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text class="lang-en" x="335" y="60" text-anchor="middle" fill="#276b45" font-size="9">station 3</text><text class="lang-zh" x="335" y="60" text-anchor="middle" fill="#276b45" font-size="9">工位 3</text><text class="lang-en" x="335" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ wheels</text><text class="lang-zh" x="335" y="75" text-anchor="middle" fill="#276b45" font-size="9">+ 轮子</text><text x="430" y="62" text-anchor="middle" font-size="18">🚗</text><text class="lang-en" x="430" y="82" text-anchor="middle" fill="#9A5A12" font-size="9">car out</text><text class="lang-zh" x="430" y="82" text-anchor="middle" fill="#9A5A12" font-size="9">车出</text><text x="82" y="130" fill="#7C6DAA" font-size="9">input layer</text><text x="235" y="130" text-anchor="middle" fill="#2D8B55" font-size="9">hidden layers</text><text x="430" y="130" text-anchor="middle" fill="#9A5A12" font-size="9">output layer</text><text class="lang-en" x="260" y="152" text-anchor="middle" fill="#6B645E" font-size="10">each station hands its part to the next — output of one = input of the next</text><text class="lang-zh" x="260" y="152" text-anchor="middle" fill="#6B645E" font-size="10">每个工位把零件交给下一个 —— 前一个的输出 = 下一个的输入</text></g></svg>
%%%

**What the assembly-line picture gets right:** parts flow one way, station to station, each handing its work to the next. **Where it breaks down:** a real line can pass a part *backward* for rework, but the forward pass only ever flows forward. (Going backward is learning — a later day.)

#### Name the three stations by where they sit
- The frame rolling on is the [[input layer||The raw features you feed in — the numbers describing one example, like the pixels of an image. It's not a computing layer; it's just the starting list.]]: the numbers describing one example. It computes nothing; it's just the starting list.
- The middle station is a [[hidden layer||A middle layer between input and output. It builds intermediate patterns from the previous layer's output. A network can have several stacked hidden layers.]]: it turns the previous list into a richer list of patterns. A network can have several of these in a row.
- The last station is the [[output layer||The final layer. Its output list is the network's answer — one number for a yes/no guess, or several for a multi-way choice.]]: its output list *is* the network's answer.

**So far:** three names, one flow. Now the rule that makes the flow legal.

%%% insight
Why "hidden"? Because nobody ever tells that middle station what to produce. There's no correct answer for it to copy — it *invents* its own useful features from whatever the layer below hands it. That freedom is where a network's cleverness actually lives, and it's why we care so much about what happens between the input and the answer.
%%%

#### The one rule that makes them chain
The flow has a precise version: **the output list of one layer becomes the input list of the next.** The hidden station's finished part rolls straight onto the output station's bench. So the sizes must match — and here's how you check it, in three moves you can do with your finger.

%%% steps
step: count the neurons in the layer you just built — our hidden layer has 4
why: 4 neurons means it hands forward a list of exactly 4 numbers, no more, no less
step: build the NEXT layer's grid with 4 columns
why: columns = the previous layer's neuron count, because each column must catch one incoming number
step: walk the whole line once, layer by layer, asking "columns here = neurons before?"
why: therefore a size clash is caught before you ever hit Run — sizes must match at every joint
%%%

Read the diagram below as that finger-walk, drawn: 3 numbers in, our 4-neuron hidden layer, then a 2-neuron output layer. See how the "4 out" and the "4 in" meet in the middle.
~~~zh
**流水线这个画面对在哪里：** 零件往一个方向流，工位接工位，每个把自己的活交给下一个。**它在哪里不成立：** 真的流水线可以把零件*往回*送去返工，可是 forward pass 只往前流。（往回走是学习，那是后面的一天。）

#### 按它们坐的位置给三个工位起名
- 滚进来的车架是 [[input layer||你喂进去的原始特征 —— 描述一个样本的那些数字，比如一张图的像素。它不是一个会算东西的 layer；它只是起点那一串。]]：描述一个样本的那些数字。它不做计算，只是起点那一串。
- 中间那个工位是一个 [[hidden layer||在 input 和 output 之间的中间层。它从上一层的输出搭出中间的模式。一个网络可以有好几个 hidden layer 叠在一起。]]：它把上一串变成一串更丰富的模式。一个网络可以有好几个这样的层排在一起。
- 最后那个工位是 [[output layer||最后一层。它输出的那一串就是网络的答案 —— 一个数字用来给是或否的猜测，几个数字用来在多个选项里挑。]]：它输出的那一串*就是*网络的答案。

**到这里：** 三个名字，一个流动。接下来是让这个流动合法的那条规则。

%%% insight
为什么叫「hidden（隐藏）」？因为从来没有人告诉中间那个工位该产出什么。没有正确答案给它照抄 —— 它从下面那层交给它的东西里，*自己发明*有用的特征。这份自由就是一个网络的聪明真正待的地方。这也是我们这么在意输入和答案之间发生了什么的原因。
%%%

#### 让它们串起来的那一条规则
这个流动有一个精确的说法：**前一层输出的那一串，变成下一层输入的那一串。** hidden 工位做好的零件，直接滚到 output 工位的台子上。所以大小必须对上。下面是你怎么检查它，三个动作，用手指就能做。

%%% steps
step: 数一数你刚搭好那层里有几个 neuron —— 我们的 hidden layer 有 4 个
why: 4 个 neuron 意味着它往前交出的正好是 4 个数字，不多也不少
step: 把「下一层」的网格建成 4 列
why: 列 = 上一层的 neuron 个数，因为每一列都要接住一个进来的数字
step: 把整条线走一遍，一层一层问「这里的列 = 前面的 neuron 数吗？」
why: 因此大小冲突会在你按下运行之前就被抓到 —— 每个接口的大小都必须对上
%%%

把下面这张图当成那次手指走查画出来的样子：3 个数字进来，我们 4 个 neuron 的 hidden layer，然后一个 2 个 neuron 的 output layer。看看「4 出」和「4 进」怎么在中间碰上。
~~~

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="An input list flows into a hidden layer of 4 neurons, whose output flows into an output layer of 2 neurons, producing the final answer. Sizes must chain: the hidden layer's 4 outputs must match the output layer's 4 inputs."><g font-family="monospace" font-size="11">
<rect x="15" y="65" width="60" height="50" rx="6" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/>
<text x="45" y="86" text-anchor="middle" fill="#5E5191">input</text><text class="lang-en" x="45" y="102" text-anchor="middle" fill="#5E5191">3 nums</text><text class="lang-zh" x="45" y="102" text-anchor="middle" fill="#5E5191">3 个数</text>
<line x1="75" y1="90" x2="140" y2="90" stroke="#B8AEDA" stroke-width="1.5"/><text x="107" y="82" text-anchor="middle" fill="#9A938A" font-size="9">→</text>
<rect x="140" y="45" width="90" height="90" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text x="185" y="30" text-anchor="middle" fill="#1a5c38" font-weight="bold">hidden</text>
<text class="lang-en" x="185" y="88" text-anchor="middle" fill="#1a5c38">4 neurons</text><text class="lang-zh" x="185" y="88" text-anchor="middle" fill="#1a5c38">4 个 neuron</text><text class="lang-en" x="185" y="104" text-anchor="middle" fill="#1a5c38">→ 4 nums</text><text class="lang-zh" x="185" y="104" text-anchor="middle" fill="#1a5c38">→ 4 个数</text>
<line x1="230" y1="90" x2="300" y2="90" stroke="#8FBFA3" stroke-width="1.5"/><text class="lang-en" x="265" y="82" text-anchor="middle" fill="#2D8B55" font-size="9">4 out = 4 in</text><text class="lang-zh" x="265" y="82" text-anchor="middle" fill="#2D8B55" font-size="9">4 出 = 4 进</text>
<rect x="300" y="55" width="90" height="70" rx="6" fill="#FDF6EC" stroke="#9A5A12" stroke-width="2"/>
<text x="345" y="40" text-anchor="middle" fill="#9A5A12" font-weight="bold">output</text>
<text class="lang-en" x="345" y="90" text-anchor="middle" fill="#9A5A12">2 neurons</text><text class="lang-zh" x="345" y="90" text-anchor="middle" fill="#9A5A12">2 个 neuron</text><text class="lang-en" x="345" y="106" text-anchor="middle" fill="#9A5A12">→ 2 nums</text><text class="lang-zh" x="345" y="106" text-anchor="middle" fill="#9A5A12">→ 2 个数</text>
<line x1="390" y1="90" x2="455" y2="90" stroke="#E0B87A" stroke-width="1.5"/>
<text class="lang-en" x="470" y="94" text-anchor="middle" fill="#6B645E">answer</text><text class="lang-zh" x="470" y="94" text-anchor="middle" fill="#6B645E">答案</text>
<text class="lang-en" x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="10">each layer's output length must equal the next layer's input length</text><text class="lang-zh" x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="10">每一层输出的长度，必须等于下一层输入的长度</text>
</g></svg>
%%%

Nice — you can now read any network diagram in the world: count the boxes, check the joints. Hold that finger-walk; it is about to save you from the first jam.
~~~zh
很好 —— 现在世界上任何一张网络图你都能读了：数框，查接口。把那次手指走查记住。它马上就要救你躲过第一个卡壳。
~~~

@@@ concept id=c6 zh_tag="完整的一趟" tag="The full trip" zh_title="forward pass —— 顺着流水线走一趟" title="The forward pass — one trip down the line" zh_gotit="懂了 forward pass" gotit="Got the forward pass"
Now we run the whole thing, end to end. This is today's star.

Picture **dominoes standing in a row.** You tip the first one. It knocks the next, which knocks the next, all the way to the end. One direction, in a fixed order, no skipping and no going back.

Feed one input in, let each station do its two-move job, hand the result along, and read the answer off the end. That single front-to-back trip is the [[forward pass||The ordered, one-direction trip that turns an input into a prediction: input → layer 1 → bend → layer 2 → … → output. Also called forward propagation, or inference on a trained network.]] — a one-direction trip from numbers to a guess. The input tips the hidden layer, the hidden layer tips the output layer, and the last domino's fall *is* the prediction.
~~~zh
现在我们把整个东西从头跑到尾。这是今天的主角。

想象**一排立着的多米诺骨牌。** 你推倒第一张。它撞倒下一张，下一张再撞下一张，一直到最后。一个方向，固定的顺序，不跳过，也不回头。

喂进一个输入，让每个工位做它那两步的活，把结果往下传，然后从末端读出答案。这一趟从头到尾的路程，就是 [[forward pass||有顺序、单方向的一趟，把一个输入变成一个 prediction：输入 → 第 1 层 → 弯 → 第 2 层 → … → 输出。也叫 forward propagation，在训练好的网络上跑就叫 inference。]] —— 一趟从数字走到猜测的单向旅程。输入推倒 hidden layer，hidden layer 推倒 output layer，最后一张骨牌倒下的那一下*就是* prediction。
~~~

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A row of standing dominoes tipping over in sequence. A finger tips the first domino, which knocks the second, then the third, all falling one direction in a fixed order until the last one falls — that fall is the prediction."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Tip the first → the whole row topples, one way, in order</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">推倒第一张 → 整排依次倒下，一个方向</text><text x="30" y="64" font-size="18">👆</text><g transform="rotate(-32 78 118)"><rect x="66" y="78" width="24" height="56" rx="3" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.8"/></g><text x="78" y="152" text-anchor="middle" fill="#5E5191" font-size="9">input</text><g transform="rotate(-16 150 110)"><rect x="138" y="70" width="24" height="64" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/></g><text x="150" y="152" text-anchor="middle" fill="#276b45" font-size="9">layer 1</text><rect x="218" y="66" width="24" height="68" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.8"/><text class="lang-en" x="230" y="152" text-anchor="middle" fill="#276b45" font-size="9">bend</text><text class="lang-zh" x="230" y="152" text-anchor="middle" fill="#276b45" font-size="9">弯</text><rect x="296" y="66" width="24" height="68" rx="3" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.8"/><text x="308" y="152" text-anchor="middle" fill="#9A5A12" font-size="9">layer 2</text><rect x="374" y="66" width="24" height="68" rx="3" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.8"/><text x="386" y="152" text-anchor="middle" fill="#9A5A12" font-size="9">output</text><g transform="rotate(66 452 118)"><rect x="440" y="84" width="24" height="56" rx="3" fill="#FDE8E8" stroke="#C93B3B" stroke-width="1.8"/></g><text x="478" y="110" text-anchor="middle" fill="#C93B3B" font-size="14">💥</text><text class="lang-en" x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">no skipping, no going back — the last fall IS the prediction</text><text class="lang-zh" x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">不跳过，不回头 —— 最后倒下的那一下就是 prediction</text></g></svg>
%%%

**What the dominoes picture gets right:** the strict one-way order — each step only happens after the one before it, and the last fall is the result. **Where it breaks down:** dominoes are identical and just fall over, while each layer does real, different math. And you can't run this backward for free — the backward trip is a separate, harder journey you'll meet on the training days.

%%% insight
This little trip is not a toy. Every time you send ChatGPT a message, your words become numbers and take exactly this ride — in one direction, layer after layer, until numbers come out the far end and get turned back into words. Same shape of journey you're about to trace with your finger. Just a much longer line.
%%%

#### Take the trip, one station at a time
Our network, with the numbers we've been building all day: 3 numbers in, our 4-neuron hidden layer with a ReLU bend, then a 2-neuron output layer whose grid `W₂` is `[[.5, .5, 0, .5], [−.5, 0, 1, .5]]` with both biases 0.

%%% steps
step: the part rolls on — x = [1, 2, 3]
why: this is the input layer, just the raw numbers describing one example
step: station 1 mixes — z₁ = W₁·x + b₁ = [1.5, −0.2, 0.6, −0.1]
why: one beep gives all 4 hidden neurons their weighted sums, so the list is now length 4
step: station 1 bends — a₁ = f(z₁) = [1.5, 0.0, 0.6, 0.0]
why: the stamp hits each of the 4 entries alone; this is the step that stops the collapse we meet later
step: the hand-off — a₁ rolls onto station 2's bench
why: the hidden layer's output IS the output layer's input; that's the chaining rule, live
step: station 2 mixes — ŷ = W₂·a₁ + b₂, so ŷ₁ = .5(1.5) + .5(0) + 0(0.6) + .5(0) = 0.75
why: 2 output neurons, so the list shrinks from 4 to 2 — and today we stop here, raw
step: read ŷ off the end of the line
why: those 2 numbers are the network's guess; the trip is over, one direction, fixed order
%%%

**So far:** six moves, and every single one you already learned. The whole trip drawn as one line looks like this.
~~~zh
**多米诺这个画面对在哪里：** 那个严格的单向顺序 —— 每一步只在前一步之后发生，最后倒下的那一下就是结果。**它在哪里不成立：** 多米诺骨牌长得一样，也只是倒下去，而每一层做的是真的、各不相同的数学。而且你没办法白白地把这个倒着跑 —— 往回的那一趟是另一段更难的旅程，你会在训练那几天遇到它。

%%% insight
这一小趟不是玩具。每次你给 ChatGPT 发一条消息，你的字会变成数字，然后坐的就是这趟车。一个方向，一层接一层，直到数字从另一头出来，再被变回文字。和你马上要用手指走的那一趟，形状一模一样。只是线长得多。
%%%

#### 走一趟，一个工位一个工位来
我们的网络，用今天一直在搭的那些数字：3 个数字进来，我们 4 个 neuron 的 hidden layer 带一个 ReLU 的弯，然后一个 2 个 neuron 的 output layer。它的网格 `W₂` 是 `[[.5, .5, 0, .5], [−.5, 0, 1, .5]]`，两个 bias 都是 0。

%%% steps
step: 零件滚上来 —— x = [1, 2, 3]
why: 这就是 input layer，只是描述一个样本的原始数字
step: 工位 1 搅拌 —— z₁ = W₁·x + b₁ = [1.5, −0.2, 0.6, −0.1]
why: 一声「哔」给了 4 个 hidden neuron 各自的加权和，所以这一串现在长度是 4
step: 工位 1 抹弯 —— a₁ = f(z₁) = [1.5, 0.0, 0.6, 0.0]
why: 印章单独盖在这 4 格上；这一步正是挡住我们后面要见的那个坍缩的一步
step: 交接 —— a₁ 滚到工位 2 的台子上
why: hidden layer 的输出「就是」output layer 的输入；这就是串联规则，活的
step: 工位 2 搅拌 —— ŷ = W₂·a₁ + b₂，所以 ŷ₁ = .5(1.5) + .5(0) + 0(0.6) + .5(0) = 0.75
why: 2 个 output neuron，所以这一串从 4 缩到 2 —— 今天我们停在这里，保持生的
step: 从线的末端读出 ŷ
why: 这 2 个数字就是网络的猜测；旅程结束，一个方向，固定顺序
%%%

**到这里：** 六个动作，每一个你都已经学过了。整趟路画成一条线，长这样。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="The forward pass as an ordered chain: input, then layer 1 multiply, then bend, then layer 2 multiply, arriving at the output prediction. Arrows point one direction only."><g font-family="monospace" font-size="10">
<rect x="10" y="70" width="52" height="40" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="36" y="88" text-anchor="middle" fill="#5E5191">x</text><text x="36" y="102" text-anchor="middle" fill="#5E5191">[1,2,3]</text>
<text x="72" y="94" fill="#9A938A">→</text>
<rect x="86" y="70" width="66" height="40" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="119" y="88" text-anchor="middle" fill="#1a5c38">W₁·x+b₁</text><text x="119" y="102" text-anchor="middle" fill="#1a5c38">= z₁</text>
<text x="160" y="94" fill="#9A938A">→</text>
<rect x="174" y="70" width="54" height="40" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="1.8"/><text x="201" y="94" text-anchor="middle" fill="#9A5A12">f(z₁)=a₁</text>
<text x="236" y="94" fill="#9A938A">→</text>
<rect x="250" y="70" width="70" height="40" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="285" y="88" text-anchor="middle" fill="#1a5c38">W₂·a₁+b₂</text><text x="285" y="102" text-anchor="middle" fill="#1a5c38">= ŷ</text>
<text x="328" y="94" fill="#9A938A">→</text>
<rect x="346" y="65" width="140" height="50" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="416" y="86" text-anchor="middle" fill="#C93B3B" font-weight="bold">prediction ŷ</text><text class="lang-en" x="416" y="102" text-anchor="middle" fill="#C93B3B">[0.75, -0.15] (raw)</text><text class="lang-zh" x="416" y="102" text-anchor="middle" fill="#C93B3B">[0.75, -0.15]（生的）</text>
<text class="lang-en" x="260" y="40" text-anchor="middle" fill="#3A342E" font-weight="bold" font-size="12">one direction, fixed order — the forward pass</text><text class="lang-zh" x="260" y="40" text-anchor="middle" fill="#3A342E" font-weight="bold" font-size="12">一个方向，固定顺序 —— forward pass</text>
<text class="lang-en" x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">length 3 → 4 → 2: multiply, bend, multiply → read the answer</text><text class="lang-zh" x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">长度 3 → 4 → 2：乘、弯、乘 → 读出答案</text>
</g></svg>
%%%

#### The whole network in one honest line
Read that chain right to left, as boxes inside boxes, and the six moves squeeze into one line:

`ŷ = W₂·f₁(W₁·x + b₁) + b₂`

In plain words: *"take the first station's total, bend it, then run that through the second station's multiply-plus-bias."*

The inner `f₁(W₁·x + b₁)` is the hidden layer's output `a₁`. The outer `W₂·f₁(…) + b₂` is the raw answer. Reading a network as boxes inside boxes like this — functions nested inside functions — is called seeing it as a [[composition||A function made by feeding one function's output into the next. A network is a chain of layers composed together: the output of each becomes the input of the next.]].

It looks busy on the page. It is exactly the six steps you just walked. If your eyes slid off it the first time, that's the normal reaction to nested notation — trace the boxes in the picture instead, and the line reads itself.

%%% demo id=fwd label="run the full forward pass"
predict: the input has 3 numbers. How many come out the far end — 3, 4, or 2? And what happened to the list in between?
code: W1 = np.array([[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]]); b1 = np.array([0., -.5, 0., -2.])
code: W2 = np.array([[.5,.5,0.,.5],[-.5,0.,1.,.5]]);                    b2 = np.array([0., 0.])
code: x  = np.array([1., 2., 3.])
code: a1 = relu(W1 @ x + b1); yhat = W2 @ a1 + b2; (a1.shape, yhat)
out: ((4,), array([ 0.75, -0.15]))
take: <b>One trip down the line, one prediction out.</b> Length 3 in → length 4 across the hidden layer (bent by ReLU into `[1.5, 0, 0.6, 0]`) → length 2 at the output. Check the first answer yourself: `.5(1.5) + .5(0) + 0(0.6) + .5(0) = 0.75`. That is the whole nested line `ŷ = W₂·f₁(W₁·x + b₁) + b₂`, run start to finish, in one line of code.
%%%

!!! c-info 🪜
<b>Optional (skippable) — where's the output bend, and why is one answer negative?</b> Notice there's no <code>f₂</code> wrapped around the outside: today the output layer does its multiply-plus-bias and stops, handing back a <i>raw</i> list — which is why <code>-0.15</code> is allowed to be negative. Some networks do add a final bend (a sigmoid or a softmax) to turn that raw list into a clean probability, but that belongs with the <b>loss</b> lesson next, so we leave the output raw for now.
!!!

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it:</b> "The forward pass is the ordered chain input → (W·x + b) → activation → next layer → … → output. Each hidden dense layer is one matrix-vector multiply plus a bias, then an elementwise activation. The whole network is a composition of those layers — the output of one is the input of the next — and it runs strictly in one direction to produce a prediction." You can already say every piece of that.
!!!

**You just ran a whole network in your head.** Next: the first way this line jams — and it takes one line to fix.
~~~zh
#### 一整个网络，老实写成一行
把那条链子从右往左读，当成一层套一层的盒子，六个动作就挤成一行：

`ŷ = W₂·f₁(W₁·x + b₁) + b₂`

大白话：*「拿第一个工位的总数，把它抹弯，然后把那个结果送进第二个工位的乘法加 bias。」*

里面那个 `f₁(W₁·x + b₁)` 是 hidden layer 的输出 `a₁`。外面那个 `W₂·f₁(…) + b₂` 是生的答案。像这样把一个网络读成一层套一层的盒子 —— 函数套在函数里面 —— 叫做把它看成一个 [[composition||把一个函数的输出喂给下一个函数，这样做出来的函数。一个网络就是一串层复合在一起：每一层的输出变成下一层的输入。]]。

它在纸上看起来很挤。它其实正是你刚走过的那六步。如果你第一眼看它眼睛就滑走了，那是面对套起来的记号时的正常反应 —— 去看图里那些盒子，这一行自己就读得通了。

%%% demo id=fwdzh label="完整的 forward pass —— 先猜，再展开"
predict: 输入有 3 个数字。末端出来几个 —— 3、4，还是 2？中间那一串又发生了什么？
code: W1 = np.array([[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]]); b1 = np.array([0., -.5, 0., -2.])
code: W2 = np.array([[.5,.5,0.,.5],[-.5,0.,1.,.5]]);                    b2 = np.array([0., 0.])
code: x  = np.array([1., 2., 3.])
code: a1 = relu(W1 @ x + b1); yhat = W2 @ a1 + b2; (a1.shape, yhat)
out: ((4,), array([ 0.75, -0.15]))
take: <b>顺着线走一趟，出来一个 prediction。</b>长度 3 进去 → 穿过 hidden layer 变成长度 4（被 ReLU 抹弯成 `[1.5, 0, 0.6, 0]`）→ 到 output 变成长度 2。第一个答案你自己核一下：`.5(1.5) + .5(0) + 0(0.6) + .5(0) = 0.75`。那就是整条套起来的式子 `ŷ = W₂·f₁(W₁·x + b₁) + b₂`，从头跑到尾，一行代码。
%%%

!!! c-info 🪜
<b>选读（可以跳过）—— output 的弯去哪了，为什么有一个答案是负的？</b>注意外面并没有套一个 <code>f₂</code>：今天 output layer 做完乘法加 bias 就停了，交回一串<i>生的</i>数字 —— 这就是 <code>-0.15</code> 可以是负数的原因。有些网络确实会在最后加一个弯（一个 sigmoid 或者一个 softmax），把这串生的数字变成干净的概率。可是那属于下一课的 <b>loss</b>，所以现在我们让 output 保持生的。
!!!

!!! c-ok 🎤
<b>一旦想通了，你会这样说出来：</b>「forward pass 是这条有顺序的链子：输入 → (W·x + b) → activation → 下一层 → … → 输出。每一个 hidden dense layer 就是一次 matrix-vector multiply 加一个 bias，然后一个 elementwise 的 activation。整个网络是这些层的 composition —— 前一个的输出就是下一个的输入 —— 而且它严格只往一个方向跑，产出一个 prediction。」这里面每一块你现在都说得出来了。
!!!

**你刚刚在脑子里跑完了一整个网络。** 接下来：这条线卡壳的第一种方式 —— 修它只要一行。
~~~

@@@ concept id=c7 zh_tag="卡壳 #1：接不上" tag="Jam #1: bad fit" zh_title="零件对不上的时候（shape mismatch）" title="When the parts don't fit (shape mismatch)" zh_gotit="懂了形状怎么修" gotit="Got the shape fix"
Here's a puzzle every single beginner runs into, usually within their first hour of writing a network. The good news: it's loud, honest, and takes one line to fix.

Picture holding a **4-prong plug in front of a 3-hole socket.** You don't need a manual. The counts don't match, so it simply won't seat, and you know instantly.

That's a [[shape mismatch||When one layer outputs a list of one length but the next layer was built to read a different length, so the matrix multiply can't line up. The program stops with a dimension error.]]: layer 1 hands forward a list of 4 numbers, but layer 2's grid was built with only 3 columns — so its multiply has nowhere to put the fourth number. The program stops and prints a "dimension mismatch" error.
~~~zh
下面这个难题，每一个新手都会碰上，通常就在他写网络的第一个小时里。好消息是：它很吵、很老实，修它只要一行。

想象你举着一个**四脚插头，对着一个三孔插座。** 你不需要说明书。数量对不上，它就是插不进去，你一下就知道。

那就是一个 [[shape mismatch||一层输出的那一串是某个长度，可是下一层是按另一个长度建的，于是矩阵乘法对不上。程序会停下来，报一个维度错误。]]：第 1 层往前交出 4 个数字，可是第 2 层的网格只建了 3 列 —— 它的乘法没有地方放第四个数字。程序停下来，打印一个「dimension mismatch（维度不匹配）」的错误。
~~~

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A four-prong electrical plug held in front of a socket that has only three holes. The prong count and the hole count do not match, so the plug cannot seat — a visible, instant mismatch."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">4 prongs into a 3-hole socket — won't seat</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">4 根插脚插 3 个孔 —— 插不进去</text><rect x="40" y="46" width="70" height="72" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text class="lang-en" x="75" y="40" text-anchor="middle" fill="#276b45" font-size="9">layer-1 plug</text><text class="lang-zh" x="75" y="40" text-anchor="middle" fill="#276b45" font-size="9">第 1 层的插头</text><line x1="110" y1="58" x2="150" y2="58" stroke="#2D8B55" stroke-width="5"/><line x1="110" y1="74" x2="150" y2="74" stroke="#2D8B55" stroke-width="5"/><line x1="110" y1="90" x2="150" y2="90" stroke="#2D8B55" stroke-width="5"/><line x1="110" y1="106" x2="150" y2="106" stroke="#2D8B55" stroke-width="5"/><text class="lang-en" x="75" y="86" text-anchor="middle" fill="#276b45" font-weight="bold">4 prongs</text><text class="lang-zh" x="75" y="86" text-anchor="middle" fill="#276b45" font-weight="bold">4 根插脚</text><text x="178" y="86" text-anchor="middle" fill="#C93B3B" font-size="22">✗</text><rect x="210" y="50" width="84" height="64" rx="8" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="252" y="42" text-anchor="middle" fill="#C93B3B" font-size="9">layer-2 socket</text><text class="lang-zh" x="252" y="42" text-anchor="middle" fill="#C93B3B" font-size="9">第 2 层的插座</text><circle cx="252" cy="66" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><circle cx="252" cy="82" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><circle cx="252" cy="98" r="5" fill="#FFF" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="278" y="86" fill="#C93B3B" font-weight="bold">3 holes</text><text class="lang-zh" x="278" y="86" fill="#C93B3B" font-weight="bold">3 个孔</text><text class="lang-en" x="167" y="140" text-anchor="middle" fill="#C93B3B" font-size="10">4 ≠ 3 → you notice instantly (loud error)</text><text class="lang-zh" x="167" y="140" text-anchor="middle" fill="#C93B3B" font-size="10">4 ≠ 3 → 你立刻就发现（大声报错）</text><rect x="330" y="52" width="172" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="416" y="74" text-anchor="middle" fill="#276b45">fix: match the counts</text><text class="lang-zh" x="416" y="74" text-anchor="middle" fill="#276b45">修法：让数量对上</text><text class="lang-en" x="416" y="92" text-anchor="middle" fill="#276b45" font-size="10">columns = previous layer's neurons</text><text class="lang-zh" x="416" y="92" text-anchor="middle" fill="#276b45" font-size="10">列 = 上一层的 neuron 数</text></g></svg>
%%%

**What the plug-and-socket picture gets right:** mismatched counts can't connect, and you find out right away. **Where it breaks down:** a plug is a physical thing you can't force, while this mismatch is a rule the computer checks — same idea, but it fails as a printed message, not a physical block.

%%% insight
Be glad this one is loud. A crash with a line number is the friendliest kind of bug there is: it tells you *exactly* where the parts don't fit. The other jam waiting for you today says nothing at all — and quietly wastes your afternoon. Loud beats silent, every time.
%%%

#### The habit that never lets it happen
The fix is the shape rule from unit 2, used as a checklist. Engineers call this **shape bookkeeping**, and it's genuinely the first thing a professional does when a network won't run.

%%% steps
step: write the sizes down BEFORE writing any code — 3 → 4 → 2
why: that little arrow chain is your map; every grid in the network has to agree with it
step: for each layer, set rows = the number of neurons in THIS layer
why: rows are the workers, so rows decide how many numbers roll out
step: set columns = the previous layer's neuron count
why: columns are the sockets, so columns must equal the number of prongs arriving
step: print the shape after every layer while the code runs
why: therefore a mismatch shows up at the exact station where it starts, not three layers later
%%%

**So far:** rows are what a layer *produces*, columns are what it *accepts*. Get those two right at every joint and the parts always fit — as the picture below shows, the whole fix is "give layer 2 a fourth column".
~~~zh
**插头插座这个画面对在哪里：** 数量不对就接不上，而且你马上就知道。**它在哪里不成立：** 插头是实物，你硬塞也塞不进去。这个不匹配是电脑检查的一条规则。同一个意思，可是它以一条打印出来的消息失败，不是被实物挡住。

%%% insight
为这个卡壳很吵而高兴吧。一个带行号的崩溃是世上最友好的那种错误：它*正好*告诉你零件在哪里对不上。今天还有另一个卡壳在等你，它一个字都不说 —— 悄悄浪费掉你一个下午。吵永远好过安静。
%%%

#### 让它永远不会发生的那个习惯
修法就是第 2 单元那条形状规则，拿来当检查清单用。工程师把这个叫 **shape bookkeeping（形状记账）**。当一个网络跑不起来时，这真的是专业的人做的第一件事。

%%% steps
step: 在写任何代码「之前」先把大小写下来 —— 3 → 4 → 2
why: 那条小小的箭头链就是你的地图；网络里每一张网格都得跟它一致
step: 对每一层，把行设成「这一层」里 neuron 的个数
why: 行是工人，所以行决定有多少个数字滚出来
step: 把列设成上一层的 neuron 个数
why: 列是插孔，所以列必须等于到达的插脚数量
step: 代码跑的时候，每过一层就打印一次形状
why: 因此不匹配会在它开始的那个工位就冒出来，而不是三层之后
%%%

**到这里：** 行是一层*产出*的东西，列是它*接受*的东西。在每个接口把这两个弄对，零件就永远合得上 —— 就像下面这张图说的，整个修法就是「给第 2 层加上第四列」。
~~~

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A layer outputs a length-4 list, but the next layer's weight grid has only 3 columns, so they cannot connect. The fix: make columns equal the previous layer's neuron count, giving layer 2 four columns."><g font-family="monospace" font-size="11">
<rect x="20" y="55" width="70" height="70" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text class="lang-en" x="55" y="40" text-anchor="middle" fill="#1a5c38">layer 1 out</text><text class="lang-zh" x="55" y="40" text-anchor="middle" fill="#1a5c38">第 1 层出来</text>
<text class="lang-en" x="55" y="80" text-anchor="middle" fill="#1a5c38">length</text><text class="lang-zh" x="55" y="80" text-anchor="middle" fill="#1a5c38">长度</text><text x="55" y="98" text-anchor="middle" fill="#1a5c38" font-weight="bold">4</text>
<text x="120" y="94" fill="#C93B3B" font-size="20">✗</text>
<rect x="155" y="55" width="120" height="70" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/>
<text class="lang-en" x="215" y="40" text-anchor="middle" fill="#C93B3B">layer 2's W</text><text class="lang-zh" x="215" y="40" text-anchor="middle" fill="#C93B3B">第 2 层的 W</text>
<text class="lang-en" x="215" y="82" text-anchor="middle" fill="#C93B3B">only 3</text><text class="lang-zh" x="215" y="82" text-anchor="middle" fill="#C93B3B">只有 3</text><text class="lang-en" x="215" y="100" text-anchor="middle" fill="#C93B3B">columns</text><text class="lang-zh" x="215" y="100" text-anchor="middle" fill="#C93B3B">列</text>
<text class="lang-en" x="215" y="145" text-anchor="middle" fill="#C93B3B" font-size="10">4 ≠ 3 → won't connect</text><text class="lang-zh" x="215" y="145" text-anchor="middle" fill="#C93B3B" font-size="10">4 ≠ 3 → 接不上</text>
<text class="lang-en" x="300" y="70" fill="#6B645E">fix:</text><text class="lang-zh" x="300" y="70" fill="#6B645E">修法：</text>
<rect x="330" y="55" width="170" height="50" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/>
<text class="lang-en" x="415" y="76" text-anchor="middle" fill="#1a5c38">give layer 2's W</text><text class="lang-zh" x="415" y="76" text-anchor="middle" fill="#1a5c38">给第 2 层的 W</text><text class="lang-en" x="415" y="94" text-anchor="middle" fill="#1a5c38" font-weight="bold">4 columns</text><text class="lang-zh" x="415" y="94" text-anchor="middle" fill="#1a5c38" font-weight="bold">4 列</text>
<text class="lang-en" x="415" y="130" text-anchor="middle" fill="#2D8B55" font-size="10">columns = previous layer's neurons</text><text class="lang-zh" x="415" y="130" text-anchor="middle" fill="#2D8B55" font-size="10">列 = 上一层的 neuron 数</text>
</g></svg>
%%%

One jam solved, and you fixed it with a rule you already had — that's your first professional debugging habit, earned on day 3. Now for the fun part: what all this stacking actually *buys* you.
~~~zh
一个卡壳解决了，而你是用一条本来就有的规则修的 —— 这是你的第一个专业调试习惯，在第 3 天挣到的。现在来玩有趣的部分：这样一层层叠起来，到底*换来*了什么。
~~~

@@@ concept id=c8 zh_tag="为什么深的更强" tag="Why depth wins" zh_title="一层层叠起来，想法越来越大" title="Stacked layers build bigger ideas" zh_gotit="懂了 depth" gotit="Got depth"
Time for the payoff. Why bother with a second station at all — why not one enormous row of workers?

Think about building with **LEGO.** You never build a house straight out of loose studs. You snap a few studs into a **brick**. You line bricks up into a **wall**. You stand walls together into a **room**, and rooms into a **house**. Each stage builds from the pieces the stage before it made — and that's the only reason a kid can build something as complicated as a house at all.

Layers do exactly that. This is [[depth||Having several hidden layers in a row. With an activation between them, each layer builds richer patterns from the last — this is where a network's power comes from.]]: several hidden layers in a row, each one drawing richer patterns from the last.
~~~zh
该收获了。为什么非要第二个工位 —— 为什么不来一排超大的工人？

想一想拼**乐高。** 你不会直接用散颗粒盖一栋房子。你先把几颗粒拼成一块**砖**。你把砖排成一面**墙**。你把墙立起来围成一个**房间**，房间再合成一栋**房子**。每个阶段都用前一个阶段做出来的零件。这也是一个小孩能盖出房子这么复杂的东西的唯一原因。

层做的正是这件事。这就是 [[depth||好几个 hidden layer 排在一起。中间有 activation 的话，每一层都从上一层搭出更丰富的模式 —— 一个网络的力量就是从这里来的。]]：好几个 hidden layer 排在一起，每一个都从上一层画出更丰富的模式。
~~~

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A LEGO build-up in four stages: loose studs snap into a single brick, bricks line up into a wall, walls stand together into a room, and rooms make a house. Each stage is built from the pieces the previous stage made."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">studs → brick → wall → house: each stage uses the last stage's pieces 🧱</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">小颗粒 → 砖 → 墙 → 房子：每一阶段都用上一阶段的零件 🧱</text><circle cx="30" cy="70" r="5" fill="#C99A12" stroke="#9A5A12"/><circle cx="46" cy="70" r="5" fill="#C99A12" stroke="#9A5A12"/><circle cx="38" cy="86" r="5" fill="#C99A12" stroke="#9A5A12"/><text class="lang-en" x="38" y="120" text-anchor="middle" fill="#9A5A12">studs</text><text class="lang-zh" x="38" y="120" text-anchor="middle" fill="#9A5A12">小颗粒</text><text class="lang-en" x="38" y="134" text-anchor="middle" fill="#9A938A" font-size="9">the input</text><text class="lang-zh" x="38" y="134" text-anchor="middle" fill="#9A938A" font-size="9">输入</text><text x="72" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="94" y="62" width="60" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><circle cx="106" cy="60" r="4" fill="#FCF3DC" stroke="#C99A12"/><circle cx="124" cy="60" r="4" fill="#FCF3DC" stroke="#C99A12"/><circle cx="142" cy="60" r="4" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="124" y="120" text-anchor="middle" fill="#9A5A12">one brick</text><text class="lang-zh" x="124" y="120" text-anchor="middle" fill="#9A5A12">一块砖</text><text x="124" y="134" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 1</text><text x="168" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="190" y="48" width="100" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><rect x="190" y="68" width="100" height="18" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><text class="lang-en" x="240" y="120" text-anchor="middle" fill="#276b45">a wall</text><text class="lang-zh" x="240" y="120" text-anchor="middle" fill="#276b45">一面墙</text><text x="240" y="134" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 2</text><text x="304" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="336" y="62" width="86" height="30" rx="3" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.8"/><path d="M330 62 L379 34 L428 62 Z" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.8"/><rect x="368" y="74" width="22" height="18" fill="#FFFFFF" stroke="#7C6DAA"/><text class="lang-en" x="379" y="120" text-anchor="middle" fill="#5E5191">a house 🏠</text><text class="lang-zh" x="379" y="120" text-anchor="middle" fill="#5E5191">一栋房子 🏠</text><text class="lang-en" x="379" y="134" text-anchor="middle" fill="#9A938A" font-size="9">the answer</text><text class="lang-zh" x="379" y="134" text-anchor="middle" fill="#9A938A" font-size="9">答案</text><text class="lang-en" x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="10">no stage skips ahead — each one is only possible because of the one before it</text><text class="lang-zh" x="260" y="162" text-anchor="middle" fill="#6B645E" font-size="10">没有一个阶段能跳过 —— 每一个都靠前一个才可能</text></g></svg>
%%%

**What the LEGO picture gets right:** big things are built from middle-sized things, which are built from small things — no stage can skip ahead. **Where it breaks down:** *you* decide what a LEGO wall should look like, but nobody tells a hidden layer what to build. It invents its own bricks from the data, which is stranger and much more powerful.

%%% insight
This is why a photo app can look at a grid of raw pixels and say "cat". No single row of workers could ever answer that from raw dots. But a stack can: dots → edges → an eye, an ear, a whisker → cat. Each layer re-describes what the layer below handed it, in slightly bigger words. That re-description *is* what "deep" actually buys you.
%%%

#### Watch the ideas grow, stage by stage
Take a photo of a face and follow the assembly line up.

%%% steps
step: hidden layer 1 finds tiny edges — a light-dark border here, a corner there
why: it can only see raw pixels, so tiny local contrasts are all it CAN find
step: hidden layer 2 combines those edges into parts — a curve of an eyelid, the line of a nose
why: therefore layer 2 never touches a pixel; it works on layer 1's edge-words, which is why its ideas are bigger
step: hidden layer 3 combines parts into a whole face
why: two eyes above a nose above a mouth is a pattern in PART-language, unsayable in pixel-language
step: the output layer reads that description and answers "this is Sam"
why: the answer was easy — because three layers of re-describing made it easy
%%%

**So far:** each layer speaks a slightly bigger language than the one below. Here is that growth drawn in one figure.
~~~zh
**乐高这个画面对在哪里：** 大东西是由中等大的东西搭起来的，中等大的又是由小东西搭起来的 —— 没有一个阶段能跳过去。**它在哪里不成立：** 一面乐高墙长什么样是*你*定的，可是没有人告诉一个 hidden layer 该搭什么。它自己从数据里发明自己的砖，这更奇怪，也强大得多。

%%% insight
这就是为什么一个照片应用能看着一格一格的原始像素说出「猫」。单独一排工人永远没办法从一堆点里回答这个。可是一叠可以：点 → 边 → 一只眼睛、一只耳朵、一根胡须 → 猫。每一层都把下面那层交给它的东西，用大一点的词重新描述一遍。这个重新描述*就是*「深」真正给你换来的东西。
%%%

#### 看想法一阶一阶长大
拿一张人脸照片，跟着流水线往上走。

%%% steps
step: hidden layer 1 找到很小的边 —— 这里一条明暗的分界，那里一个角
why: 它只能看到原始像素，所以很小的局部反差是它「唯一能」找到的东西
step: hidden layer 2 把这些边拼成部件 —— 一条眼皮的弧，一条鼻子的线
why: 因此第 2 层根本不碰像素；它在第 1 层的「边词」上干活，这就是它的想法更大的原因
step: hidden layer 3 把部件拼成一整张脸
why: 两只眼睛在鼻子上面、鼻子在嘴巴上面，这是「部件语言」里的模式，用像素语言说不出来
step: output layer 读这段描述，回答「这是 Sam」
why: 这个答案很容易 —— 因为三层重新描述把它变容易了
%%%

**到这里：** 每一层说的语言，都比下面那层大一点。下面这张图把这个长大的过程画在一起。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A feature hierarchy in four stages. Raw pixels become small edges, edges become face parts like an eye and a nose, parts become a whole face, and the output names the person. Each stage is drawn bigger and more meaningful than the last."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Each layer re-describes the last in slightly bigger words</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">每一层用大一点的词，重新描述上一层</text><rect x="18" y="42" width="70" height="70" rx="4" fill="#F3F0EA" stroke="#C0B8AC"/><g fill="#B8AEA2"><rect x="24" y="48" width="12" height="12"/><rect x="48" y="48" width="12" height="12"/><rect x="36" y="66" width="12" height="12"/><rect x="60" y="78" width="12" height="12"/><rect x="24" y="90" width="12" height="12"/></g><text class="lang-en" x="53" y="128" text-anchor="middle" fill="#6B645E">raw pixels</text><text class="lang-zh" x="53" y="128" text-anchor="middle" fill="#6B645E">原始像素</text><text class="lang-en" x="53" y="142" text-anchor="middle" fill="#9A938A" font-size="9">the input x</text><text class="lang-zh" x="53" y="142" text-anchor="middle" fill="#9A938A" font-size="9">输入 x</text><text x="98" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="122" y="42" width="70" height="70" rx="4" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.6"/><path d="M130 100 L160 55" stroke="#5E5191" stroke-width="2" fill="none"/><path d="M150 100 L184 70" stroke="#5E5191" stroke-width="2" fill="none"/><path d="M132 62 L176 62" stroke="#5E5191" stroke-width="2" fill="none"/><text class="lang-en" x="157" y="128" text-anchor="middle" fill="#5E5191">edges</text><text class="lang-zh" x="157" y="128" text-anchor="middle" fill="#5E5191">边</text><text x="157" y="142" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 1</text><text x="202" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="226" y="42" width="70" height="70" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><ellipse cx="252" cy="66" rx="14" ry="8" fill="#FFF" stroke="#2D8B55" stroke-width="1.6"/><circle cx="252" cy="66" r="3" fill="#2D8B55"/><path d="M272 82 L272 100 L282 100" stroke="#2D8B55" stroke-width="1.8" fill="none"/><text class="lang-en" x="261" y="128" text-anchor="middle" fill="#276b45">parts</text><text class="lang-zh" x="261" y="128" text-anchor="middle" fill="#276b45">部件</text><text x="261" y="142" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 2</text><text x="306" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="330" y="42" width="70" height="70" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><circle cx="365" cy="77" r="26" fill="#FFF" stroke="#C99A12" stroke-width="1.6"/><circle cx="356" cy="70" r="3" fill="#9A5A12"/><circle cx="374" cy="70" r="3" fill="#9A5A12"/><path d="M356 88 Q365 94 374 88" stroke="#9A5A12" stroke-width="1.6" fill="none"/><text class="lang-en" x="365" y="128" text-anchor="middle" fill="#9A5A12">a face</text><text class="lang-zh" x="365" y="128" text-anchor="middle" fill="#9A5A12">一张脸</text><text x="365" y="142" text-anchor="middle" fill="#9A938A" font-size="9">hidden layer 3</text><text x="410" y="80" fill="#B8AEA2" font-size="14">→</text><rect x="430" y="60" width="76" height="34" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="468" y="82" text-anchor="middle" fill="#C93B3B" font-weight="bold">"it's Sam"</text><text class="lang-zh" x="468" y="82" text-anchor="middle" fill="#C93B3B" font-weight="bold">「是 Sam」</text><text x="468" y="128" text-anchor="middle" fill="#9A938A" font-size="9">output layer</text><text class="lang-en" x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">pixels → edges → parts → a face → a name</text><text class="lang-zh" x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">像素 → 边 → 部件 → 一张脸 → 一个名字</text></g></svg>
%%%

#### Your turn — try to beat one layer, then give it a second
Here's the classic 60-second experiment that made the whole field want depth. Four dots. Two classes, sitting on opposite diagonals (this pattern has a name: **XOR**).

- **Drag the three sliders** and try to place ONE straight line so both blue dots are on one side and both orange dots are on the other. Really try. The best anyone ever manages is 3 out of 4.
- Now **tick "add a hidden layer"** — a second line appears, and the two lines together fence off exactly the right region. Four out of four.
~~~zh
#### 换你来 —— 先试着用一层赢，再给它第二层
下面是那个经典的 60 秒实验，它让整个领域都想要深度。四个点。两个类别，坐在对角线的两头（这个图案有个名字：**XOR**）。

- **拖那三个滑块**，试着放「一条」直线，让两个蓝点在一边，两个橙点在另一边。真的去试。谁都最多只能做到 4 个里对 3 个。
- 现在**勾上「add a hidden layer」** —— 第二条线出现了，两条线一起正好把该围的区域围出来。4 个里对 4 个。
~~~

%%% viz src=../../viz/xor-limit.html title="one line versus a hidden layer" caption="interactive — drag w1, w2, b to move your single dividing line (you cannot get all four dots right), then tick add a hidden layer and watch a second line appear"
%%%

That little failure is real history. In 1969, Marvin Minsky and Seymour Papert showed that a single layer of neurons cannot express XOR — and interest in neural networks cooled for years. The way out was the thing you just clicked: add a hidden layer.

**Victory lap.** You just re-ran, in one minute, the experiment that explains why every modern network is deep. Several hidden layers, each drawing richer patterns from the last, can build almost anything, while one layer alone only draws simple shapes. Depth is the whole game — with one condition attached, which is the next unit.
~~~zh
这个小小的失败是真的历史。1969 年，Marvin Minsky 和 Seymour Papert 证明了一层 neuron 表达不了 XOR。之后好多年，大家对神经网络的兴趣冷了下去。走出来的办法就是你刚点的那个：加一个 hidden layer。

**庆祝一下。** 你刚刚用一分钟重跑了那个实验，它解释了为什么每一个现代网络都是深的。好几个 hidden layer，每一个都从上一层画出更丰富的模式，几乎什么都能搭出来。而单独一层只画得出简单的形状。深度就是全部的关键 —— 但它带着一个条件，那就是下一单元。
~~~

@@@ concept id=c9 zh_tag="卡壳 #2：不出声的那个" tag="Jam #2: the quiet one" zh_title="中间不加弯，多层会折成一层" title="Layers with no bend fold into one" zh_gotit="懂了坍缩怎么修" gotit="Got the collapse fix"
So depth is the superpower. Here's the one condition attached to it — and it's a single line of code, which is a wonderful price for that much power. It's yesterday's idea, now seen on the whole assembly line.

Picture a line where **every station is just a conveyor belt that stretches the part.** Station 1 doubles its length. Station 2 triples it. Run a part through both and it comes out six times longer.

But hold on — you didn't need two stations. **One** belt set to "×6" does the identical job. Stack a hundred such belts and you still only get one stretch.

That's the [[linear collapse||When you stack layers with no activation between them, the whole stack behaves like a single layer. All the depth is wasted, because straight-line steps combine into one straight-line step.]]: layers with no bend between them fold into a single layer, and all your hard-won depth vanishes.
~~~zh
所以深度是那个超能力。下面是它带的那一个条件 —— 它只是一行代码，换这么大的力量，这个价钱好得很。这是昨天的想法，现在放到整条流水线上看。

想象一条线，**每个工位都只是一条会把零件拉长的传送带。** 工位 1 把它的长度变成两倍。工位 2 变成三倍。让一个零件走完两个工位，它出来时长了六倍。

可是等一下 —— 你并不需要两个工位。**一**条设成「×6」的传送带做的是一模一样的活。把一百条这样的带子叠起来，你得到的还是一次拉长。

那就是 [[linear collapse||当你把层叠起来、中间没有 activation 时，整叠的行为就跟单独一层一样。所有深度都浪费了，因为直线的步子合起来还是一个直线的步子。]]：中间不加弯的层会折成一层，你辛苦挣来的深度全部消失。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Two conveyor belts in a row that only scale a bar's length. The first stretches it two times, the second three times. A dashed arrow shows one single belt set to six times does the exact same job — two belts collapse into one."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two scaling belts (×2 then ×3) = one belt set to ×6</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">两条缩放带（先 ×2 再 ×3）= 一条设成 ×6 的带子</text><rect x="20" y="56" width="22" height="10" rx="2" fill="#7C6DAA"/><text class="lang-en" x="31" y="48" text-anchor="middle" fill="#5E5191" font-size="9">part</text><text class="lang-zh" x="31" y="48" text-anchor="middle" fill="#5E5191" font-size="9">零件</text><rect x="52" y="48" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><text class="lang-en" x="97" y="65" text-anchor="middle" fill="#276b45">belt ×2</text><text class="lang-zh" x="97" y="65" text-anchor="middle" fill="#276b45">带子 ×2</text><rect x="150" y="56" width="44" height="10" rx="2" fill="#7C6DAA"/><rect x="200" y="48" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.6"/><text class="lang-en" x="245" y="65" text-anchor="middle" fill="#276b45">belt ×3</text><text class="lang-zh" x="245" y="65" text-anchor="middle" fill="#276b45">带子 ×3</text><rect x="298" y="56" width="132" height="10" rx="2" fill="#7C6DAA"/><text class="lang-en" x="364" y="48" text-anchor="middle" fill="#5E5191" font-size="9">6× longer</text><text class="lang-zh" x="364" y="48" text-anchor="middle" fill="#5E5191" font-size="9">长了 6 倍</text><text x="445" y="64" fill="#C93B3B">=</text><rect x="20" y="104" width="22" height="10" rx="2" fill="#7C6DAA"/><rect x="52" y="96" width="238" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.8"/><text class="lang-en" x="171" y="113" text-anchor="middle" fill="#C93B3B" font-weight="bold">one belt ×6 — same result</text><text class="lang-zh" x="171" y="113" text-anchor="middle" fill="#C93B3B" font-weight="bold">一条 ×6 的带子 —— 一样的结果</text><rect x="298" y="104" width="132" height="10" rx="2" fill="#7C6DAA"/><text class="lang-en" x="210" y="144" text-anchor="middle" fill="#C93B3B" font-size="10">no bend → two stations do the work of one (depth wasted)</text><text class="lang-zh" x="210" y="144" text-anchor="middle" fill="#C93B3B" font-size="10">没有弯 → 两个工位做一个的活（深度浪费了）</text><text class="lang-en" x="260" y="164" text-anchor="middle" fill="#2D8B55" font-size="10">a bend between them → they stay two real layers</text><text class="lang-zh" x="260" y="164" text-anchor="middle" fill="#2D8B55" font-size="10">中间放一个弯 → 它们保持成两个真的层</text></g></svg>
%%%

**What the conveyor-belt picture gets right:** plain scaling steps really do multiply into a single scaling, so extra stations add nothing. **Where it breaks down:** a belt only stretches, while a layer also *mixes and shifts* the numbers — but the punchline survives: with no bend, all that mixing still folds into one single mixing step.

%%% insight
Here's why this one bites so hard. Nothing breaks. The code runs, the shapes fit, a number comes out, the loss even goes down a little. Your 5-layer network is quietly doing the work of one layer, and nothing on your screen says so. That's why we're going to *see* the collapse with our own eyes instead of trusting it.
%%%

#### Catch the collapse with real numbers
Let's shrink the line to a size you can check on paper. Two tiny stations — call their grids **A** and **B** (same roles as `W₁` and `W₂`, just smaller) — and one 2-number input `u`, with biases left at 0 to keep it clean:

`u = [−3, 1]`, `A = [[1, 2], [0, 1]]`, `B = [[1, 0], [3, 1]]`

%%% steps
step: run station A — A·u = [1(−3) + 2(1), 0(−3) + 1(1)] = [−1, 1]
why: ordinary weighted sums, exactly like unit 3; no bend applied, because that's the mistake we're studying
step: run station B — B·[−1, 1] = [1(−1) + 0(1), 3(−1) + 1(1)] = [−1, −2]
why: two full stations done, and [−1, −2] is this little network's answer
step: now the shortcut — multiply the two grids FIRST: B·A = [[1, 2], [3, 7]]
why: one grid that claims to do both stations' work in a single move
step: run that single grid once — [[1,2],[3,7]]·[−3, 1] = [1(−3)+2(1), 3(−3)+7(1)] = [−1, −2]
why: the same answer, therefore the two stations really were one station in disguise
step: name it — this is the collapse
why: two layers of effort, one layer of power, and not one error message to warn you
step: now the fix — bend the middle list before passing it on: relu([−1, 1]) = [0, 1], then B·[0, 1] = [0, 1]
why: [0, 1] is nothing like [−1, −2], so with a bend the two stations are finally doing two different jobs
%%%

**So far:** without a bend, two paths land on the same answer; with a bend, they part company. Here is that whole story in one figure — collapse on top, the fix underneath.
~~~zh
**传送带这个画面对在哪里：** 单纯的缩放步子真的会乘成一次缩放，所以多出来的工位什么都没加。**它在哪里不成立：** 一条带子只会拉长，而一层还会*搅拌和平移*数字 —— 可是结论还是站得住：没有弯，那些搅拌照样折成一次搅拌。

%%% insight
这个卡壳为什么咬得这么疼。什么都没坏。代码跑得动，形状对得上，数字出来了，loss 甚至还降了一点。你那个 5 层的网络在悄悄做着一层的活，而屏幕上没有一个字告诉你。所以我们要用自己的眼睛*看见*这个坍缩，而不是相信它。
%%%

#### 用真的数字抓住这个坍缩
我们把这条线缩到你能在纸上核对的大小。两个很小的工位 —— 它们的网格叫 **A** 和 **B**（和 `W₁`、`W₂` 是同样的角色，只是更小）—— 还有一个 2 个数字的输入 `u`，bias 都留成 0，好让它干净：

`u = [−3, 1]`，`A = [[1, 2], [0, 1]]`，`B = [[1, 0], [3, 1]]`

%%% steps
step: 跑工位 A —— A·u = [1(−3) + 2(1), 0(−3) + 1(1)] = [−1, 1]
why: 就是普通的加权和，和第 3 单元一样；这里不抹弯，因为那正是我们要研究的错误
step: 跑工位 B —— B·[−1, 1] = [1(−1) + 0(1), 3(−1) + 1(1)] = [−1, −2]
why: 两个完整的工位做完了，[−1, −2] 就是这个小网络的答案
step: 现在走近路 —— 先把两张网格乘起来：B·A = [[1, 2], [3, 7]]
why: 一张网格，声称一步就能做完两个工位的活
step: 把这一张网格跑一次 —— [[1,2],[3,7]]·[−3, 1] = [1(−3)+2(1), 3(−3)+7(1)] = [−1, −2]
why: 同一个答案，因此那两个工位真的只是一个工位在伪装
step: 给它起名 —— 这就是坍缩
why: 两层的功夫，一层的力量，而且一条错误消息都没有来提醒你
step: 现在是修法 —— 把中间那一串抹弯再往下传：relu([−1, 1]) = [0, 1]，然后 B·[0, 1] = [0, 1]
why: [0, 1] 和 [−1, −2] 完全不像，所以加了弯，这两个工位终于在做两件不同的活
%%%

**到这里：** 没有弯，两条路落在同一个答案上；有弯，它们就分开了。下面这张图把整个故事画在一起 —— 上面是坍缩，下面是修法。
~~~

%%% svg
<svg viewBox="0 0 520 260" role="img" aria-label="Top: input u equals minus three, one, times A gives minus one, one, times B gives minus one, minus two; and the combined grid B times A also gives minus one, minus two, so the two paths match and the layers collapse. Bottom: with a ReLU bend the middle list becomes zero, one, and the answer becomes zero, one, which no longer matches the combined grid."><g font-family="monospace" font-size="11">
<text class="lang-en" x="18" y="20" fill="#C93B3B" font-weight="bold" font-size="12">NO bend — the two paths match (collapse)</text><text class="lang-zh" x="18" y="20" fill="#C93B3B" font-weight="bold" font-size="12">不加弯 —— 两条路一样（坍缩）</text>
<rect x="18" y="30" width="58" height="32" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="47" y="50" text-anchor="middle" fill="#5E5191">u=[-3,1]</text>
<text x="82" y="50" fill="#9A938A">→</text>
<rect x="98" y="30" width="74" height="32" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="135" y="44" text-anchor="middle" fill="#1a5c38">A·u</text><text x="135" y="57" text-anchor="middle" fill="#1a5c38">=[-1,1]</text>
<text x="178" y="50" fill="#9A938A">→</text>
<rect x="194" y="30" width="92" height="32" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="240" y="44" text-anchor="middle" fill="#1a5c38">B·[-1,1]</text><text x="240" y="57" text-anchor="middle" fill="#1a5c38">=[-1,-2]</text>
<text x="292" y="50" fill="#9A938A">→</text>
<rect x="308" y="26" width="96" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="356" y="50" text-anchor="middle" fill="#C93B3B" font-weight="bold">[-1, -2]</text>
<rect x="98" y="76" width="188" height="30" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="1.8"/><text class="lang-en" x="192" y="95" text-anchor="middle" fill="#9A5A12">one grid B·A = [[1,2],[3,7]]</text><text class="lang-zh" x="192" y="95" text-anchor="middle" fill="#9A5A12">一张网格 B·A = [[1,2],[3,7]]</text>
<text x="292" y="95" fill="#9A938A">→</text>
<rect x="308" y="76" width="96" height="30" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="356" y="96" text-anchor="middle" fill="#C93B3B" font-weight="bold">[-1, -2]</text>
<text class="lang-en" x="416" y="70" fill="#C93B3B" font-weight="bold">same!</text><text class="lang-zh" x="416" y="70" fill="#C93B3B" font-weight="bold">一样！</text><text class="lang-en" x="416" y="86" fill="#6B645E" font-size="10">two layers</text><text class="lang-zh" x="416" y="86" fill="#6B645E" font-size="10">两层</text><text class="lang-en" x="416" y="100" fill="#6B645E" font-size="10">= one layer</text><text class="lang-zh" x="416" y="100" fill="#6B645E" font-size="10">= 一层</text>
<line x1="18" y1="126" x2="502" y2="126" stroke="#E4DED5" stroke-width="1"/>
<text class="lang-en" x="18" y="150" fill="#2D8B55" font-weight="bold" font-size="12">WITH a bend — the match breaks (depth is real)</text><text class="lang-zh" x="18" y="150" fill="#2D8B55" font-weight="bold" font-size="12">加了弯 —— 相等破了（深度是真的）</text>
<rect x="18" y="164" width="58" height="32" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="47" y="184" text-anchor="middle" fill="#5E5191">u=[-3,1]</text>
<text x="82" y="184" fill="#9A938A">→</text>
<rect x="98" y="164" width="74" height="32" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="135" y="178" text-anchor="middle" fill="#1a5c38">A·u</text><text x="135" y="191" text-anchor="middle" fill="#1a5c38">=[-1,1]</text>
<text x="178" y="184" fill="#9A938A">→</text>
<rect x="194" y="164" width="92" height="32" rx="5" fill="#FDF6EC" stroke="#9A5A12" stroke-width="2"/><text class="lang-en" x="240" y="178" text-anchor="middle" fill="#9A5A12">relu → [0,1]</text><text class="lang-zh" x="240" y="178" text-anchor="middle" fill="#9A5A12">relu → [0,1]</text><text class="lang-en" x="240" y="191" text-anchor="middle" fill="#9A5A12" font-size="9">the -1 is flattened</text><text class="lang-zh" x="240" y="191" text-anchor="middle" fill="#9A5A12" font-size="9">那个 -1 被压平了</text>
<text x="292" y="184" fill="#9A938A">→</text>
<rect x="308" y="160" width="96" height="40" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="356" y="184" text-anchor="middle" fill="#1a5c38" font-weight="bold">[0, 1]</text>
<text class="lang-en" x="416" y="178" fill="#2D8B55" font-weight="bold">different!</text><text class="lang-zh" x="416" y="178" fill="#2D8B55" font-weight="bold">不一样！</text><text class="lang-en" x="416" y="194" fill="#6B645E" font-size="10">no single grid</text><text class="lang-zh" x="416" y="194" fill="#6B645E" font-size="10">没有单张网格</text><text class="lang-en" x="416" y="208" fill="#6B645E" font-size="10">can do this</text><text class="lang-zh" x="416" y="208" fill="#6B645E" font-size="10">做得出这个</text>
<text class="lang-en" x="260" y="238" text-anchor="middle" fill="#6B645E" font-size="10">the ONE thing that changed: a bend on the middle list</text><text class="lang-zh" x="260" y="238" text-anchor="middle" fill="#6B645E" font-size="10">只改了一件事：给中间那一串加了一个弯</text>
</g></svg>
%%%

%%% mathladder
title: why two layers with no bend fold into one
words: doing station A and then station B is the same as doing one station whose grid is B·A
formula: B·(A·u) = (B·A)·u
numbers: u = [-3, 1], A = [[1,2],[0,1]], B = [[1,0],[3,1]] → both paths give [-1, -2]
sanity: put a bend in the middle and the match breaks — B·relu(A·u) = [0, 1]
%%%

%%% hint
t1: Feeling lost on how two grids become one? Start smaller. Forget the whole grid — just track ONE number through both stations and ask whether a single station could have done the same job.
t2: Take u = [-3, 1]. Station A: A·u = [1(-3) + 2(1), 0(-3) + 1(1)] = [-1, 1]. Station B: B·[-1, 1] = [-1, -2]. Now the shortcut: multiply the grids first, B·A = [[1, 2], [3, 7]], and run that once on u — you get [-1, -2] too. Same answer, one station.
t3: With no bend, doing A then B is the exact same math as one combined grid B·A — so the two layers are secretly one, and all the depth is wasted.
%%%

Don't take the picture's word for it. **Guess first, then reveal.**

%%% demo id=collapse label="run: two linear layers vs one combined grid vs one bend"
predict: three outputs come back. Which two will be identical — and will the ReLU one join them or break away?
code: A = np.array([[1,2],[0,1]]); B = np.array([[1,0],[3,1]]); u = np.array([-3,1])
code: no_bend = B @ (A @ u); combined = (B @ A) @ u; with_bend = B @ relu(A @ u)
code: (B @ A).tolist(), no_bend.tolist(), combined.tolist(), with_bend.tolist()
out: ([[1, 2], [3, 7]], [-1, -2], [-1, -2], [0, 1])
take: <b>The first two match; the bend breaks away.</b> With no bend, `A` then `B` is exactly the single grid `B·A = [[1,2],[3,7]]` — both give `[-1, -2]`, so the two layers really did fold into one. Add the bend and you get `[0, 1]`: a different answer that no single grid can produce. One line of code, and your depth is real again.
%%%

!!! c-info 🪜
<b>Optional (skippable) — when the bend looks like it does nothing.</b> Try the same demo with <code>u = [1, 2]</code>. Now <code>A·u = [5, 2]</code> — both entries are already positive, so ReLU has nothing to flatten and the bent answer matches the collapsed one exactly. The bend only changes the result once some middle number goes <b>negative</b>. So when you test for the collapse, pick an input that drives a hidden total below zero — otherwise your test passes for the wrong reason.
!!!

#### The fix you already own
The remedy is yesterday's star: **put a bend** — the activation `f` — between the layers. That elementwise `a = f(z)` step from unit 4 is not decoration; it is the thing that stops the collapse. With a bend between the layers, `A` and `B` can no longer be multiplied into one grid, so the two stations stay two genuinely different transforms.

!!! c-warn 🤫
<b>Failure mode (silent):</b> stack linear layers with no activation and nothing breaks — the code runs, the shapes fit, a number comes out. But your 5-layer network secretly has the power of <i>one</i> layer, so it learns almost nothing and you can't tell why. The tell is a model that stays stubbornly weak no matter how many layers you add. The fix is one line: an activation between every pair of layers.
!!!

Depth is an illusion *without* the bend, and a superpower *with* it. That single trade is the whole reason "deep" learning is called deep — and you just proved it with two tiny grids and one multiply.
~~~zh
%%% mathladder
title: 为什么中间不加弯的两层会折成一层
words: 先做工位 A 再做工位 B，和做一个网格是 B·A 的单个工位，是同一件事
formula: B·(A·u) = (B·A)·u
numbers: u = [-3, 1], A = [[1,2],[0,1]], B = [[1,0],[3,1]] → 两条路都给出 [-1, -2]
sanity: 在中间放一个弯，这个相等就破了 —— B·relu(A·u) = [0, 1]
%%%

%%% hint
t1: 想不通两张网格怎么变成一张？从更小的开始。先别管整张网格 —— 只追一个数字走完两个工位，然后问：一个工位能不能做同样的活？
t2: 拿 u = [-3, 1]。工位 A：A·u = [1(-3) + 2(1), 0(-3) + 1(1)] = [-1, 1]。工位 B：B·[-1, 1] = [-1, -2]。现在走近路：先把网格乘起来，B·A = [[1, 2], [3, 7]]，再把它对 u 跑一次 —— 你也得到 [-1, -2]。同一个答案，一个工位。
t3: 没有弯，先 A 再 B 和一张合起来的网格 B·A 是完全一样的数学 —— 所以这两层其实是一层，所有深度都浪费了。
%%%

别只听这张图说。**先猜，再展开。**

%%% demo id=collapsezh label="两个 linear 层、一张合起来的网格、一个弯 —— 先猜，再展开"
predict: 会回来三个输出。哪两个会一模一样 —— 那个用了 ReLU 的会跟它们一样，还是跑开？
code: A = np.array([[1,2],[0,1]]); B = np.array([[1,0],[3,1]]); u = np.array([-3,1])
code: no_bend = B @ (A @ u); combined = (B @ A) @ u; with_bend = B @ relu(A @ u)
code: (B @ A).tolist(), no_bend.tolist(), combined.tolist(), with_bend.tolist()
out: ([[1, 2], [3, 7]], [-1, -2], [-1, -2], [0, 1])
take: <b>前两个一样；有弯的那个跑开了。</b>没有弯的时候，先 `A` 再 `B` 正好就是那一张网格 `B·A = [[1,2],[3,7]]` —— 两边都给 `[-1, -2]`，所以这两层真的折成了一层。加上弯，你得到 `[0, 1]`：一个不同的答案，没有任何单张网格做得出来。一行代码，你的深度又是真的了。
%%%

!!! c-info 🪜
<b>选读（可以跳过）—— 弯看起来什么都没做的时候。</b>把同一个 demo 换成 <code>u = [1, 2]</code> 试试。现在 <code>A·u = [5, 2]</code> —— 两格本来就都是正的，所以 ReLU 没有东西可以压平，抹过弯的答案和坍缩的那个正好一样。只有当中间某个数字变成<b>负数</b>，这个弯才会改变结果。所以你测坍缩的时候，要挑一个能把中间的总数压到零以下的输入。不然你的测试是因为错的理由才通过的。
!!!

#### 你早就有的那个修法
解药就是昨天的主角：**放一个弯** —— 也就是 activation `f` —— 放在层之间。第 4 单元那个 elementwise 的 `a = f(z)` 不是装饰；它就是挡住坍缩的那个东西。层之间有了弯，`A` 和 `B` 就没办法再乘成一张网格，所以这两个工位保持成两个真正不同的变换。

!!! c-warn 🤫
<b>失效模式（安静的那种）：</b>把 linear 的层叠起来、中间不放 activation，什么都不会坏 —— 代码跑得动，形状对得上，数字出来了。可是你那个 5 层的网络暗地里只有<i>一</i>层的力量，所以它几乎学不到东西，而你看不出为什么。信号是：一个模型死活很弱，你加多少层都一样。修法只有一行：在每两层之间放一个 activation。
!!!

*没有*弯，深度是幻觉；*有*弯，深度是超能力。就这一笔交易，是 deep learning（深度学习）被叫做「深」的全部原因 —— 而你刚刚用两张小网格和一次乘法证明了它。
~~~

@@@ concept id=c10 zh_tag="老实说的边界" tag="The honest edge" zh_title="forward pass 只是猜 —— 训练在后面" title="A forward pass guesses — training comes next" zh_gotit="懂了这条边界" gotit="Got the edge"
One honest thing to say about today's machine, and it's the most exciting sentence in the lesson — because it's the door into everything next.

Picture a brand-new **vending machine that's plugged in but never stocked.** Press a button and it *will* dispense something — the mechanism is flawless. But nobody loaded the right snacks, so whatever drops out is basically random.

Today's forward pass is that machine. It runs perfectly and always produces an answer. But the weights `W` and biases `b` are still whatever they started as, so the answer is **just a guess.** The forward pass *uses* the weights; it never *changes* them.
~~~zh
关于今天这台机器，有一件老实话要说。它是这一课里最让人兴奋的一句 —— 因为它是通往后面所有东西的门。

想象一台崭新的**自动售货机，插上电了，可是从来没有装过货。** 你按一个按钮，它*确实*会掉出东西来 —— 机械部分完美无缺。可是没有人把对的零食装进去，所以掉出来的基本上是随机的。

今天的 forward pass 就是这台机器。它跑得完美，而且总会产出一个答案。可是 weight `W` 和 bias `b` 还是它们一开始的样子，所以这个答案**只是一个猜测。** forward pass *用*这些 weight，它从不*改动*它们。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A vending machine plugged into the wall with its light on. A hand presses a button and an item drops into the tray, but the shelves are empty or randomly filled, so whatever drops out is a random guess."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Plugged in, works — but never stocked → random drop</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">插上电，能动 —— 可是从没装货 → 随便掉一个</text><rect x="150" y="30" width="120" height="128" rx="8" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text class="lang-en" x="210" y="46" text-anchor="middle" fill="#5E5191" font-size="9">💡 powered on</text><text class="lang-zh" x="210" y="46" text-anchor="middle" fill="#5E5191" font-size="9">💡 通电了</text><rect x="160" y="52" width="70" height="58" rx="3" fill="#FFFFFF" stroke="#B8AEDA"/><line x1="160" y1="71" x2="230" y2="71" stroke="#E5DFD6"/><line x1="160" y1="90" x2="230" y2="90" stroke="#E5DFD6"/><line x1="195" y1="52" x2="195" y2="110" stroke="#E5DFD6"/><text class="lang-en" x="177" y="66" text-anchor="middle" fill="#B8AEA2" font-size="9">empty</text><text class="lang-zh" x="177" y="66" text-anchor="middle" fill="#B8AEA2" font-size="9">空的</text><text x="213" y="85" text-anchor="middle" fill="#B8AEA2" font-size="9">?</text><text x="177" y="104" text-anchor="middle" fill="#B8AEA2" font-size="9">?</text><circle cx="248" cy="66" r="5" fill="#FCF3DC" stroke="#C99A12"/><circle cx="248" cy="82" r="5" fill="#FCF3DC" stroke="#C99A12"/><text x="245" y="98" fill="#9A5A12" font-size="9">👆</text><rect x="160" y="128" width="70" height="22" rx="3" fill="#FDE8E8" stroke="#C93B3B"/><text class="lang-en" x="195" y="143" text-anchor="middle" fill="#C93B3B" font-size="9">random drop 🎲</text><text class="lang-zh" x="195" y="143" text-anchor="middle" fill="#C93B3B" font-size="9">随便掉一个 🎲</text><text class="lang-en" x="110" y="70" text-anchor="middle" fill="#6B645E" font-size="10">press →</text><text class="lang-zh" x="110" y="70" text-anchor="middle" fill="#6B645E" font-size="10">按下 →</text><text class="lang-en" x="300" y="70" fill="#6B645E" font-size="10">it runs...</text><text class="lang-zh" x="300" y="70" fill="#6B645E" font-size="10">它跑起来了…</text><text class="lang-en" x="300" y="88" fill="#C93B3B" font-size="10">...but the answer</text><text class="lang-zh" x="300" y="88" fill="#C93B3B" font-size="10">…可是这个答案</text><text class="lang-en" x="300" y="104" fill="#C93B3B" font-size="10">is just a guess</text><text class="lang-zh" x="300" y="104" fill="#C93B3B" font-size="10">只是一个猜测</text><text class="lang-en" x="300" y="124" fill="#9A938A" font-size="9">restocking =</text><text class="lang-zh" x="300" y="124" fill="#9A938A" font-size="9">重新装货 =</text><text class="lang-en" x="300" y="138" fill="#9A938A" font-size="9">training (later)</text><text class="lang-zh" x="300" y="138" fill="#9A938A" font-size="9">训练（以后）</text></g></svg>
%%%

**What the vending-machine picture gets right:** the machine works whether or not it's stocked well, just as the forward pass runs whether or not the weights are any good. **Where it breaks down:** you restock a vending machine by hand, but a network restocks *itself* through training — a clever backward trip we haven't taken yet.

%%% insight
Read this as a preview, not a complaint. Every piece of machinery you built today is exactly what training needs in order to work: it *scores* the guess you just produced, then walks back down the same line nudging every `W` and `b` a little. You didn't build half a network today — you built the half that has to exist first.
%%%

#### The missing arrow, drawn as a ghost
Getting *good* weights means nudging `W` and `b` toward better answers, which needs a trip in the opposite direction — **gradient descent and backpropagation**, coming on the training days. Look at the dashed arrow below: it's drawn as a ghost, because today it doesn't exist yet. Filling it in is the rest of this module.
~~~zh
**自动售货机这个画面对在哪里：** 不管货装得好不好，机器都能动，就像不管 weight 好不好，forward pass 都跑得动。**它在哪里不成立：** 自动售货机是人用手去补货的，而一个网络通过训练*自己*补货 —— 那是一趟聪明的、往回走的路，我们还没走过。

%%% insight
把这段当成预告，不是抱怨。你今天搭的每一件机械，正好就是训练跑起来所需要的东西。训练先给你刚产出的那个猜测*打分*。然后它沿着同一条线往回走，把每一个 `W` 和 `b` 都推一点点。你今天不是搭了半个网络 —— 你搭的是必须先存在的那一半。
%%%

#### 那根缺掉的箭头，画成一个影子
想得到*好的* weight，就要把 `W` 和 `b` 往更好的答案推。这需要一趟反方向的旅程 —— **gradient descent（梯度下降）和 backpropagation（反向传播）**，它们在训练那几天来。看下面那根虚线箭头：它画成一个影子，因为今天它还不存在。把它填满，就是这个模块剩下的部分。
~~~

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="The forward pass produces a prediction from fixed weights, but there is no arrow going back to update the weights. That backward, weight-updating trip is training, coming later."><g font-family="monospace" font-size="11">
<rect x="20" y="55" width="60" height="40" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="50" y="79" text-anchor="middle" fill="#5E5191">input</text>
<text x="90" y="79" fill="#2D8B55">→</text>
<rect x="108" y="50" width="120" height="50" rx="6" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="168" y="70" text-anchor="middle" fill="#1a5c38">forward pass</text><text class="lang-en" x="168" y="87" text-anchor="middle" fill="#1a5c38" font-size="10">(fixed W, b)</text><text class="lang-zh" x="168" y="87" text-anchor="middle" fill="#1a5c38" font-size="10">（W、b 固定）</text>
<text x="236" y="79" fill="#2D8B55">→</text>
<rect x="254" y="55" width="90" height="40" rx="5" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="299" y="79" text-anchor="middle" fill="#C93B3B">a guess</text><text class="lang-zh" x="299" y="79" text-anchor="middle" fill="#C93B3B">一个猜测</text>
<path d="M299 100 C 299 130, 168 130, 168 102" fill="none" stroke="#C0B8AC" stroke-width="1.6" stroke-dasharray="5,4"/><text class="lang-en" x="235" y="145" text-anchor="middle" fill="#9A938A" font-size="10">updating W back this way = training (starts next lesson)</text><text class="lang-zh" x="235" y="145" text-anchor="middle" fill="#9A938A" font-size="10">沿这条路回来更新 W = 训练（下一课开始）</text>
</g></svg>
%%%

Knowing exactly where the forward pass stops and training begins *is* the win — that's the map for every day ahead. One page to gather it all, and you're done.
~~~zh
清楚知道 forward pass 在哪里停、训练从哪里开始，这*本身*就是收获 —— 那是接下来每一天的地图。一页把全部收拢起来，你今天就完成了。
~~~

@@@ concept id=c11 zh_tag="回顾" tag="Recap" zh_title="一页装下今天" title="Today in one page" zh_gotit="懂了回顾" gotit="Got the recap"
Take a breath — you built a whole working machine today.

Here's a fresh way to hold it all at once: think of the day like the **layers of a sandwich you just made.** The bottom slice of bread is your input. Each filling you add — cheese, then tomato, then lettuce — is one layer's work stacked on the last. The top slice you press down is the final prediction. To describe your sandwich to a friend you name the fillings in order, bottom to top — exactly how you read a network, input to answer.

**Where the sandwich picture breaks down:** fillings just sit there, while each layer actively *reshapes* what the layer below handed it (and later even retunes itself during training).
~~~zh
喘口气 —— 你今天搭出了一台完整能用的机器。

这里有一个新的办法，把它们一次抱住：把今天想成**你刚做好的那个三明治的一层层。** 最下面那片面包是你的输入。你加的每一样夹心 —— 先芝士，再番茄，再生菜 —— 是一层的活叠在上一层上面。你压下来的那片顶面包，就是最后的 prediction。你要跟朋友描述你的三明治，会从下往上按顺序念夹心 —— 这正是你读一个网络的方式，从输入念到答案。

**三明治这个画面在哪里不成立：** 夹心就那样待着，而每一层会主动*重塑*下面那层交给它的东西（后面在训练时，它甚至还会重新调自己）。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A stacked sandwich seen from the side. Bottom slice of bread is the input, then a cheese layer, a tomato layer, a lettuce layer stacked in order, and the top slice pressed down is the final prediction — read bottom to top."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Read a network like a sandwich — bottom to top 🥪</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">像读三明治一样读一个网络 —— 从下往上 🥪</text><rect x="120" y="40" width="180" height="18" rx="9" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><text class="lang-en" x="330" y="53" fill="#9A5A12" font-size="10">← top slice = prediction ŷ</text><text class="lang-zh" x="330" y="53" fill="#9A5A12" font-size="10">← 顶面包 = prediction ŷ</text><rect x="128" y="58" width="164" height="16" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.4"/><text class="lang-en" x="330" y="70" fill="#276b45" font-size="10">← lettuce = output layer</text><text class="lang-zh" x="330" y="70" fill="#276b45" font-size="10">← 生菜 = output layer</text><rect x="128" y="74" width="164" height="16" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.4"/><text class="lang-en" x="330" y="86" fill="#C93B3B" font-size="10">← tomato = hidden layer 2</text><text class="lang-zh" x="330" y="86" fill="#C93B3B" font-size="10">← 番茄 = hidden layer 2</text><rect x="128" y="90" width="164" height="16" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.4"/><text class="lang-en" x="330" y="102" fill="#9A5A12" font-size="10">← cheese = hidden layer 1</text><text class="lang-zh" x="330" y="102" fill="#9A5A12" font-size="10">← 芝士 = hidden layer 1</text><rect x="120" y="106" width="180" height="18" rx="9" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.6"/><text class="lang-en" x="330" y="119" fill="#5E5191" font-size="10">← bottom slice = input x</text><text class="lang-zh" x="330" y="119" fill="#5E5191" font-size="10">← 底面包 = 输入 x</text><text class="lang-en" x="210" y="146" text-anchor="middle" fill="#6B645E" font-size="10">name the fillings in order = read layers input → answer</text><text class="lang-zh" x="210" y="146" text-anchor="middle" fill="#6B645E" font-size="10">按顺序念夹心 = 从输入往答案读各层</text></g></svg>
%%%

#### The seven beats of today, in order
- **1 · The layer.** A row of neurons that all read the same input, each with its own weights and bias — graders marking one test for different things.
- **2 · The settings.** All the weights live in one grid **W** (rows = neurons, columns = inputs); the biases in one list **b** (one bias per neuron).
- **3 · The multiply.** One move — `z = W·x + b` — gives every neuron's weighted sum at once. Fast, and built for GPUs.
- **4 · The bend.** Apply the activation to each entry on its own: `a = f(z)`. That's one complete hidden layer, in two moves.
- **5 · The chain.** Each layer's output feeds the next; input → hidden → output. Sizes must match: columns = the previous layer's neuron count.
- **6 · The full trip.** The **forward pass** runs the chain one direction, start to finish. Our network, output left raw: `ŷ = W₂·f₁(W₁·x + b₁) + b₂`, which took `[1, 2, 3]` to `[0.75, -0.15]`.
- **7 · The payoff.** **Depth** — several hidden layers, each drawing richer patterns from the last — is what lets pixels become "a face". One layer alone can't even split XOR.

#### The two jams, and the honest edge
- **Shape mismatch** (loud — it crashes) → set columns = the previous layer's neurons, and print shapes as you go.
- **Linear collapse** (silent — it just wastes depth) → put a bend between every pair of layers. Test it with an input that drives a middle number negative, or the bend will look like it does nothing.
- **The edge:** the forward pass *guesses*, it doesn't *learn*. That's training, and it starts next.

Here's the one picture to keep.
~~~zh
#### 今天的七个节拍，按顺序
- **1 · 那个 layer。** 一排 neuron 都读同一个输入，每个有自己的 weight 和 bias —— 一群批卷人对同一份试卷看不同的东西。
- **2 · 那些设置。** 所有 weight 住在一张网格 **W** 里（行 = neuron，列 = 输入）；bias 住在一串 **b** 里（每个 neuron 一个 bias）。
- **3 · 那次乘法。** 一个动作 —— `z = W·x + b` —— 一次给出每个 neuron 的加权和。快，而且是为 GPU 生的。
- **4 · 那个弯。** 把 activation 单独用在每一格上：`a = f(z)`。这就是一个完整的 hidden layer，两个动作。
- **5 · 那条链。** 每一层的输出喂给下一层；input → hidden → output。大小必须对上：列 = 上一层的 neuron 个数。
- **6 · 完整的一趟。** **forward pass** 把这条链往一个方向从头跑到尾。我们的网络，output 留成生的：`ŷ = W₂·f₁(W₁·x + b₁) + b₂`，它把 `[1, 2, 3]` 变成了 `[0.75, -0.15]`。
- **7 · 那份收获。** **depth** —— 好几个 hidden layer，每一个都从上一层画出更丰富的模式 —— 就是让像素变成「一张脸」的东西。单独一层连 XOR 都分不开。

#### 两个卡壳，和那条老实的边界
- **shape mismatch**（吵 —— 它会崩）→ 把列设成上一层的 neuron 数，边跑边打印形状。
- **linear collapse**（安静 —— 它只是浪费深度）→ 在每两层之间放一个弯。测的时候要用一个能把中间某个数字压成负数的输入，不然这个弯看起来像什么都没做。
- **那条边界：** forward pass 只*猜*，它不*学*。学是训练，训练下一课就开始。

下面是最该留住的那一张图。
~~~

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="Recap: the forward pass as a one-direction chain — input, hidden layer multiply and bend, output layer multiply, raw prediction."><g font-family="monospace" font-size="10">
<rect x="12" y="52" width="46" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="1.8"/><text x="35" y="73" text-anchor="middle" fill="#5E5191">x</text>
<text x="64" y="72" fill="#9A938A">→</text>
<rect x="80" y="52" width="94" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="127" y="66" text-anchor="middle" fill="#1a5c38">W₁·x+b₁</text><text class="lang-en" x="127" y="80" text-anchor="middle" fill="#9A5A12">→ bend</text><text class="lang-zh" x="127" y="80" text-anchor="middle" fill="#9A5A12">→ 弯</text>
<text x="180" y="72" fill="#9A938A">→</text>
<rect x="196" y="52" width="94" height="34" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="1.8"/><text x="243" y="66" text-anchor="middle" fill="#1a5c38">W₂·a₁+b₂</text><text class="lang-en" x="243" y="80" text-anchor="middle" fill="#6B645E">(raw)</text><text class="lang-zh" x="243" y="80" text-anchor="middle" fill="#6B645E">（生的）</text>
<text x="296" y="72" fill="#9A938A">→</text>
<rect x="312" y="48" width="96" height="42" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="360" y="73" text-anchor="middle" fill="#C93B3B" font-weight="bold">prediction ŷ</text>
<text class="lang-en" x="260" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold" font-size="12">the forward pass — one trip down the line</text><text class="lang-zh" x="260" y="26" text-anchor="middle" fill="#3A342E" font-weight="bold" font-size="12">forward pass —— 顺着线走一趟</text>
<text class="lang-en" x="260" y="116" text-anchor="middle" fill="#6B645E" font-size="10">multiply → bend → multiply → read the answer</text><text class="lang-zh" x="260" y="116" text-anchor="middle" fill="#6B645E" font-size="10">乘 → 弯 → 乘 → 读出答案</text>
</g></svg>
%%%

#### Cheat-sheet · the two moves of every layer
%%% table
:: :: what it does :: in one line :: shape story
The multiply :: mixes the input — every neuron's weighted sum at once :: `z = W·x + b` :: `[neurons, inputs]·[inputs] → [neurons]`
The bend :: adds the non-linearity, each entry on its own :: `a = f(z)` :: same length in and out
%%%

#### Cheat-sheet · the words you met today
%%% jargon
layer | a row of neurons that all read the same input, each with its own weights and bias
weight matrix W | a grid of all a layer's weights — rows = neurons, columns = inputs
bias vector b | a short list, one bias number per neuron in the layer
matrix-vector multiply | one operation that computes every neuron's weighted sum at once
elementwise | applied to each number in a list on its own, with no mixing between them
input / hidden / output layer | the raw features / the middle pattern-builders / the final answer
forward pass | the one-direction trip input → layers → output that produces a prediction
composition | a network read as functions nested inside functions, output of one feeding the next
depth | several hidden layers with bends between them — where a network's power comes from
XOR | the four-dot pattern no single layer can split; the classic reason hidden layers exist
shape bookkeeping | writing the sizes down and checking every joint before you run
shape mismatch | the loud jam: a layer's input length doesn't match the previous layer's output
linear collapse | the silent jam: layers with no bend fold into one, wasting depth
%%%

That's the whole day. Next you'll learn how to *score* the network's guess and start teaching it to improve — the **loss** function, and the road to training.
~~~zh
#### 小抄 · 每一层的两个动作

%%% table
:: :: 它做什么 :: 写成一行 :: 形状的故事
那次乘法 :: 把输入搅在一起 —— 一次算出每个 neuron 的加权和 :: `z = W·x + b` :: `[neuron 数, 输入数]·[输入数] → [neuron 数]`
那个弯 :: 加上非线性，每一格单独来 :: `a = f(z)` :: 进去和出来长度一样
%%%

#### 小抄 · 你今天遇到的那些词

%%% jargon
layer | 一排 neuron，都读同一个输入，每个有自己的 weight 和 bias
weight matrix W | 一张网格，装着一层所有的 weight —— 行 = neuron，列 = 输入
bias vector b | 一小串，一层里每个 neuron 一个 bias
matrix-vector multiply | 一个运算，一次算出每个 neuron 的加权和
elementwise | 单独用在一串里的每个数字上，彼此不混
input / hidden / output layer | 原始特征 / 中间搭模式的那些层 / 最后的答案
forward pass | 单方向的一趟：输入 → 各层 → 输出，产出一个 prediction
composition | 把一个网络读成函数套函数，前一个的输出喂给下一个
depth | 好几个 hidden layer，中间都有弯 —— 一个网络的力量就是从这里来的
XOR | 那个四点图案，单独一层分不开它；hidden layer 存在的经典理由
shape bookkeeping | 把大小写下来，跑之前检查每一个接口
shape mismatch | 吵的那个卡壳：一层的输入长度和上一层的输出对不上
linear collapse | 安静的那个卡壳：中间不加弯的层折成一层，浪费深度
%%%

今天就是这些。接下来你会学怎么给网络的猜测*打分*，并开始教它变好 —— 那就是 **loss** 函数，和通往训练的路。
~~~

@@@ quiz id=quiz zh_tag="小测" tag="Quiz" zh_title="点一个答案 —— 每题都马上给反馈" title="Click an answer — instant feedback on each" zh_gotit="先答完四题" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: A layer has 5 neurons and each neuron reads a 3-number input. What is the shape of its weight matrix W? | a:1 | [3, 5] | [5, 3] | [5, 5] | [3, 3] | fb: Rows = neurons, columns = inputs. 5 neurons and 3 inputs give a [5, 3] grid — one row per neuron, one column per input feature.
q: In one dense layer, the correct order of the two moves is: | a:0 | multiply first (z = W·x + b), then bend (a = f(z)) | bend first, then multiply | bend, then bend again | multiply, then multiply again | fb: A layer first computes z = W·x + b (mixing the inputs), then applies the activation elementwise: a = f(z).
q: You stack 6 layers but put NO activation between them. The network runs fine and gives numbers, yet it barely learns anything. Why? | a:2 | The shapes don't match, so it silently skips layers | It ran out of GPU memory | With no bend between them, the linear layers collapse into a single layer — so 6 layers have the power of 1, and depth is wasted; the fix is an activation between each pair | Six layers is always too many | fb: No non-linearity means W₁…W₆ multiply into one grid — the whole stack acts like one layer. Insert an activation between layers and they stay distinct.
q: After a forward pass, why is the network's output usually a bad guess at first? | a:1 | The forward pass has a bug | The weights W and biases b start untrained — the forward pass USES them but never updates them, so the answer is basically random until training (gradient descent) tunes them | The activation is missing | The input was the wrong shape | fb: The forward pass only runs the numbers through fixed weights. Making those weights good is training (gradient descent + backpropagation), which comes later.
%%%
~~~zh
四道快问，每题都马上给反馈 —— 四题都答完，今天就完成了。（如果有一题把你难住了，那是提示你往上翻，不是不及格。）

%%% quiz
q: 一个 layer 有 5 个 neuron，每个 neuron 读一个 3 个数字的输入。它的 weight matrix W 形状是什么？ | a:1 | [3, 5] | [5, 3] | [5, 5] | [3, 3] | fb: 行 = neuron，列 = 输入。5 个 neuron、3 个输入给出一张 [5, 3] 的网格 —— 一个 neuron 一行，一个输入特征一列。
q: 在一个 dense layer 里面，两个动作的正确顺序是： | a:0 | 先乘（z = W·x + b），再抹弯（a = f(z)） | 先抹弯，再乘 | 抹弯，然后再抹一次弯 | 先乘，然后再乘一次 | fb: 一层先算 z = W·x + b，把输入搅在一起，然后 elementwise 地用 activation：a = f(z)。
q: 你叠了 6 层，可是中间「不」放 activation。网络跑得好好的，也给出数字，可是它几乎学不到东西。为什么？ | a:2 | 形状对不上，所以它悄悄跳过了几层 | GPU 内存不够了 | 中间没有弯，这些 linear 的层坍缩成一层 —— 所以 6 层只有 1 层的力量，深度浪费了；修法是在每两层之间放一个 activation | 六层永远都太多 | fb: 没有非线性，W₁…W₆ 会乘成一张网格 —— 整叠的行为就像一层。在层之间插一个 activation，它们就保持各不相同。
q: 一次 forward pass 之后，为什么网络的输出一开始通常是个糟糕的猜测？ | a:1 | forward pass 有毛病 | weight W 和 bias b 一开始没被训练过 —— forward pass「用」它们，可是从不更新它们，所以在训练（gradient descent）把它们调好之前，答案基本上是随机的 | activation 不见了 | 输入的形状不对 | fb: forward pass 只是把数字送过固定的 weight。把那些 weight 变好是训练（gradient descent + backpropagation），那在后面。
%%%
~~~

@@@ produce id=produce zh_tag="动手做" tag="Produce" zh_title="把一个数字推过整条流水线，再看没有弯时它怎么坍缩" title="Push a number down the whole line — and watch it collapse without a bend" zh_gotit="做完了" gotit="Done"
Time to see today's assembly line run with your own eyes. You'll build the exact 2-layer network from this lesson and push `x = [1, 2, 3]` all the way down to a prediction — then run the sneaky experiment: rip the bend out and watch two layers collapse into one. **Predict first:** with NO bend between them, will a 2-layer network give the *same* answer as a single cleverly-chosen layer, or a different one? Then run it and **watch.** Seeing them match without the bend, and part ways the moment you add it, is the whole lesson in a few lines of output. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py`. Build a forward pass by hand:
1. Make a layer function `layer(x, W, b, f)` that returns `f(W @ x + b)` — one multiply-plus-bias, then an elementwise bend. Print the shape of `W @ x + b` inside it so you can **watch** the vector length change from station to station.
2. Set up today's network: `x = [1, 2, 3]` → hidden layer of 4 neurons (`W1 = [[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]]`, `b1 = [0, -.5, 0, -2]`, bend = `relu`) → output layer of 2 neurons (`W2 = [[.5,.5,0,.5],[-.5,0,1,.5]]`, `b2 = [0, 0]`, no activation on the output — the raw prediction, as in the lesson). **Notice** that the hidden layer's 4 outputs must equal the output layer's 4 inputs — that's the chaining rule.
3. Run the full forward pass and print `a1` and the final length-2 prediction. You should see `a1 = [1.5, 0, 0.6, 0]` and `yhat = [0.75, -0.15]` — check `0.75` by hand and confirm the page wasn't lying to you.
4. The collapse demo, on the tiny pair from the lesson: `A = [[1,2],[0,1]]`, `B = [[1,0],[3,1]]`, `u = [-3, 1]`. Show that `B @ (A @ u)` equals `(B @ A) @ u` exactly — two linear layers really are one. Then put `relu` in the middle and show `B @ relu(A @ u)` is **different**. One trap to avoid: your input must drive a middle number **negative** (that's why `u = [-3, 1]` and not `[1, 2]`), otherwise ReLU has nothing to flatten and your test would pass for the wrong reason. Run with `python3 sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 3 artifact.

Create sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py that, with a comment on each step:
1. Defines relu(z)=np.maximum(0,z), identity(z)=z, and a layer(x, W, b, f) that returns f(W @ x + b); prints the shape of W @ x + b at each call.
2. Builds this exact 2-layer forward pass: x = [1., 2., 3.]; hidden W1 = [[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]], b1 = [0., -.5, 0., -2.], activation relu; output W2 = [[.5,.5,0.,.5],[-.5,0.,1.,.5]], b2 = [0., 0.], no activation (raw prediction). Print the lengths 3 -> 4 -> 2, print a1 and yhat, and assert np.allclose(a1, [1.5, 0., 0.6, 0.]) and np.allclose(yhat, [0.75, -0.15]).
3. Shows the linear collapse on a tiny pair: A = [[1,2],[0,1]], B = [[1,0],[3,1]], u = [-3, 1]. Print B @ A, then assert np.array_equal(B @ (A @ u), (B @ A) @ u) — two linear layers equal one layer whose grid is B @ A.
4. Then put relu between them and show the collapse breaks: assert not np.array_equal(B @ relu(A @ u), (B @ A) @ u). IMPORTANT: the test input must make A @ u contain a NEGATIVE entry (u = [-3, 1] gives A @ u = [-1, 1]), otherwise relu changes nothing and this assert would pass vacuously. Also print what happens with u = [1, 2] (A @ u = [5, 2], both positive) to show the bend changing nothing there, and say in a comment why.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the trip, then the collapse)
- The vector length changes as it flows: length **3** in → **4** after the hidden layer → **2** at the output. Watching the length change is watching each station reshape the part.
- The hidden bend bites: `z1 = [1.5, -0.2, 0.6, -0.1]` becomes `a1 = [1.5, 0, 0.6, 0]`, and the final raw prediction is `[0.75, -0.15]`.
- With **no** bend, the two-layer output `[-1, -2]` matches the single combined grid `B @ A = [[1,2],[3,7]]` exactly — two linear layers really did collapse into one.
- The moment to **watch**: drop `relu` into the middle and the answer jumps to `[0, 1]`, which no single grid can produce. That break is depth earning its keep — the same lesson as Day 2, now on a full network.

!!! c-info 📓
<b>5-minute research log:</b> before you close the tab, write three lines in `sessions/m02-the-neuron/day-03-layers-forward-pass/log.md`: (1) the two moves inside one layer, in order, and the shape rule for W; (2) what the forward pass is, in one sentence, and why its first guess is usually bad; (3) the two ways the line jams (shape mismatch, linear collapse) and the one-line fix for each.
!!!
~~~zh
是时候用自己的眼睛看今天这条流水线跑起来。你会搭出这一课里那个一模一样的 2 层网络，把 `x = [1, 2, 3]` 一路推到一个 prediction —— 然后做那个偷偷的实验：把弯拆掉，看两层坍缩成一层。**先预测：** 中间「不」加弯的话，一个 2 层网络会给出和一个精心挑过的单层*一样*的答案，还是不一样的？然后运行它，**看着。** 看到没有弯时它们一样、加上弯的那一刻它们就分开，这就是整课的内容，只要几行输出。挑一条路。

#### 路线 A · 自己写
创建 `sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py`。用手搭一个 forward pass：
1. 写一个 layer 函数 `layer(x, W, b, f)`，返回 `f(W @ x + b)` —— 一次乘法加 bias，然后一个 elementwise 的弯。在它里面打印 `W @ x + b` 的形状，这样你能**看着** vector 的长度从一个工位变到下一个工位。
2. 把今天的网络搭好：`x = [1, 2, 3]` → 4 个 neuron 的 hidden layer → 2 个 neuron 的 output layer。hidden 那层用 `W1 = [[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]]`；`b1 = [0, -.5, 0, -2]`；弯是 `relu`。output 那层用 `W2 = [[.5,.5,0,.5],[-.5,0,1,.5]]`；`b2 = [0, 0]`；output 上不加 activation —— 生的 prediction，和课上一样。**注意** hidden layer 的 4 个输出必须等于 output layer 的 4 个输入 —— 那就是串联规则。
3. 跑完整的 forward pass，打印 `a1` 和最后那个长度 2 的 prediction。你应该看到 `a1 = [1.5, 0, 0.6, 0]` 和 `yhat = [0.75, -0.15]` —— 手算一下 `0.75`，确认这一页没有骗你。
4. 坍缩演示，用课上那对小网格：`A = [[1,2],[0,1]]`，`B = [[1,0],[3,1]]`，`u = [-3, 1]`。证明 `B @ (A @ u)` 正好等于 `(B @ A) @ u` —— 两个 linear 的层真的就是一层。然后在中间放 `relu`，证明 `B @ relu(A @ u)` **不一样**。有一个坑要躲开：你的输入必须把中间某个数字压成**负数**。这就是为什么用 `u = [-3, 1]`，而不是 `[1, 2]`。不然 ReLU 没有东西可以压平。那样你的测试会因为错的理由通过。用 `python3 sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py` 运行。

#### 路线 B · 让 Claude 搭，然后你读它
把下面这段提示词复制回 Claude Code。它会让 Claude Code 创建文件、写代码，并替你运行。

%%% prompt id=ppzh label="让 Claude 帮你搭"
帮我搭 Module 2 Day 3 的产出。

创建 sessions/m02-the-neuron/day-03-layers-forward-pass/experiment.py，每一步都写注释：
1. 定义 relu(z)=np.maximum(0,z)、identity(z)=z，以及一个 layer(x, W, b, f)，返回 f(W @ x + b)；每次调用都打印 W @ x + b 的形状。
2. 搭出这个一模一样的 2 层 forward pass：x = [1., 2., 3.]；hidden W1 = [[.2,-.1,.5],[.0,.3,-.1],[.4,.1,.0],[-.3,.2,.6]], b1 = [0., -.5, 0., -2.], activation 用 relu；output W2 = [[.5,.5,0.,.5],[-.5,0.,1.,.5]], b2 = [0., 0.]，不加 activation（生的 prediction）。打印长度 3 -> 4 -> 2，打印 a1 和 yhat，并且 assert np.allclose(a1, [1.5, 0., 0.6, 0.]) 和 assert np.allclose(yhat, [0.75, -0.15])。
3. 用一对小网格展示 linear collapse：A = [[1,2],[0,1]], B = [[1,0],[3,1]], u = [-3, 1]。打印 B @ A，然后 assert np.array_equal(B @ (A @ u), (B @ A) @ u) —— 两个 linear 的层等于一个网格是 B @ A 的层。
4. 然后在它们中间放 relu，展示坍缩被打破：assert not np.array_equal(B @ relu(A @ u), (B @ A) @ u)。重要：测试输入必须让 A @ u 里出现一个负数（u = [-3, 1] 给出 A @ u = [-1, 1]），不然 relu 什么都没改，这个 assert 会空洞地通过。另外打印 u = [1, 2] 时会发生什么（A @ u = [5, 2]，两个都是正的），展示那里的弯什么都没改，并在注释里说明为什么。
然后运行它，把输出作为注释贴在文件底部。
%%%

#### 你应该看到什么（先是那一趟，然后是坍缩）
- vector 的长度会随着流动改变：长度 **3** 进去 → 过了 hidden layer 变 **4** → 到 output 变 **2**。看着长度改变，就是在看每个工位重塑这个零件。
- hidden 的弯咬下去了：`z1 = [1.5, -0.2, 0.6, -0.1]` 变成 `a1 = [1.5, 0, 0.6, 0]`，最后那个生的 prediction 是 `[0.75, -0.15]`。
- **没有**弯的时候，两层的输出 `[-1, -2]` 和那张合起来的网格 `B @ A = [[1,2],[3,7]]` 正好一样 —— 两个 linear 的层真的坍缩成了一层。
- 该**看**的那一刻：把 `relu` 丢进中间，答案跳到 `[0, 1]`，没有任何单张网格做得出来。那个断开就是深度在挣它的饭钱 —— 和第 2 天同一课，现在放在一整个网络上。

!!! c-info 📓
<b>5 分钟研究日志：</b>关掉这个页面之前，在 `sessions/m02-the-neuron/day-03-layers-forward-pass/log.md` 里写三行。(1) 一层里面那两个动作，按顺序；还有 W 的形状规则。(2) forward pass 是什么，一句话；以及为什么它第一次的猜测通常很糟。(3) 这条线卡壳的两种方式（shape mismatch、linear collapse），还有各自那一行的修法。
!!!
~~~

@@@ fin
