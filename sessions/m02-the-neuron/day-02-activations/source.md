---
quest_id: wf2-d02-activations
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 2 — Activation Functions"
module_label: "Module 2 · Train · Day 2"
zh_module_label: "第 2 模块 · 训练 · 第 2 天"
title: "Activation Functions"
zh_title: "激活函数"
subtitle: "The Bend That Makes Depth Matter"
zh_subtitle: "让深度真正有意义的那个弯"
brand_sub: "Foundations · M2 Day 2"
spine: "bend"
zh_spine: "弯"
nav_prev_href: "../day-01-single-neuron/lesson.html"
nav_prev_label: "A Single Neuron"
zh_nav_prev_label: "一个神经元"
nav_next_href: "../day-03-layers-forward-pass/lesson.html"
nav_next_label: "Layers & the Forward Pass"
zh_nav_next_label: "层与前向传播"
fin_title: "Module 2 · Day 2 complete! 🏆"
zh_fin_title: "第 2 模块 · 第 2 天 完成！🏆"
fin_body: "Nice work — you've completed <b>Activation Functions</b>. You just unlocked the one small idea that makes deep learning possible: the bend.<br>Next up: <b>Layers &amp; the Forward Pass</b>."
zh_fin_body: "做得好 —— 你学完了<b>激活函数</b>。你刚刚拿到了让深度学习成为可能的那个小小的想法：那个<b>弯</b>。<br>下一站：<b>层与前向传播</b>。"
notebook_yardstick: 00-neural-networks/fundamentals/03_activation_functions.ipynb
---

@@@ hero
@lede Welcome back! Try this in your head: draw a smiley face using only a **ruler**. You can't — a ruler makes straight lines, and a face is all curves. To draw anything real, your hand has to *bend*. A neural network hits that exact same wall, and the fix has a name: the **activation** — the "bend." Yesterday you met the neuron's third beat, *decide*, and its two oldest bends: the hard on/off switch and the smooth dimmer. Today we ask the bigger question nobody answered yesterday — *why does a bend have to be there at all?* — and then meet the rest of the family. And here's how far this one little idea reaches: it's what lets your phone decide in a blink that this face is yours, what shapes the feed that picks your next video, and what lets something like ChatGPT curve its way to an answer. Sounds strange that one bend does all that? Good — hold that feeling; by the end it'll feel obvious, and a little bit magic.
@zh_lede 欢迎回来！先在脑子里试一件事：只用一把**尺子**，画一个笑脸。你画不出来 —— 尺子只能画直线，
而一张脸全是曲线。要画出任何真实的东西，你的手必须**弯**一下。神经网络撞上的是完全一样的墙，而解决办法
有个名字：**activation（激活函数）** —— 也就是那个「弯」。昨天你认识了神经元的第三拍，*decide（决定）*，
以及两个最老的弯：硬开关和平滑的调光旋钮。今天我们问一个昨天没人回答的、更大的问题 —— *为什么一定要有
一个弯？* —— 然后认识这一家子里剩下的成员。这个小小的想法能伸得多远？它让你的手机一眨眼就认出这张脸是
你，它塑造了那个替你挑下一个视频的信息流，它也让 ChatGPT 这样的东西能拐着弯给出一个答案。一个弯就能做
到这些，听起来很奇怪？很好 —— 记住这种感觉；到今天结束时它会变得显而易见，而且还有点神奇。
@goal We'll line the bends up side by side — yesterday's on/off switch and dimmer, plus the two you haven't met: **ReLU** and **tanh**. You'll see why depth without a bend is a mirage, catch a bend quietly going to sleep, and pocket the tiny fixes that wake it up. Every failure is a little puzzle with a neat answer, and every formula gets said out loud in plain words first. No symbol left unexplained.
@zh_goal 我们会把这些弯并排摆出来 —— 昨天的硬开关和调光旋钮，加上你还没见过的两个：**ReLU** 和 **tanh**。
你会看到为什么「没有弯的深度」只是一场幻觉，会亲手抓到一个悄悄睡着的弯，还会把叫醒它的那些小办法装进
口袋。每一个失败都是一个小谜题，而且都有一个漂亮的答案；每一个公式都会先用大白话念出来。不会留下任何
一个没解释的符号。

%%% warmup
q: Yesterday you built a single neuron. What three beats does it always do, in order? | a:2 | add → weigh → decide | decide → weigh → add | weigh → add → decide | weigh → decide → add | concept: single-neuron | fb: A neuron scales each input by its weight (weigh), sums them with the bias (add), then turns that total into an answer (decide).
q: Yesterday's "decide" step came in two flavours: a hard on/off switch, and a smooth dimmer. Which one lets a *tiny* nudge to the score make a *tiny* change in the answer? | a:1 | the hard on/off switch | the smooth dimmer (sigmoid) | both, equally | neither of them | concept: sigmoid | fb: The switch jumps from 0 to 1 and is flat everywhere else, so a small nudge changes nothing at all. The dimmer slides, so a small nudge makes a small change. Today you'll see why that one difference decides almost everything.
q: A single neuron draws one straight line to split its inputs. Which pattern could it NOT learn on its own? | a:1 | a single blob of "yes" points | XOR (two classes on opposite diagonals) | points all on one side | an empty dataset | concept: xor-limit | fb: XOR sits on two diagonals, so no single straight line separates it — that's the neuron's hard ceiling, and layers are the fix.
q: In a neuron, how many weights and how many biases are there for two inputs? | a:1 | 2 weights and 2 biases | 2 weights and 1 bias | 1 weight and 1 bias | 1 weight and 2 biases | concept: weights-bias | fb: One weight per input (so two), and exactly one bias for the whole neuron — the bias is a single starting-lean number.
%%%
~~~zh
%%% warmup
q: 昨天你搭出了一个 neuron（神经元）。它总是按顺序做哪三拍？ | a:2 | 加 → 称重 → 决定 | 决定 → 称重 → 加 | 称重 → 加 → 决定 | 称重 → 决定 → 加 | concept: single-neuron | fb: 神经元先把每个输入乘上它的 weight（称重），再和 bias 一起加起来（加），最后把这个总分变成一个答案（决定）。
q: 昨天的「决定」这一拍有两种口味：一个硬的开关，和一个平滑的调光旋钮。哪一个能让分数上*很小*的一点变化，带来答案上*很小*的一点变化？ | a:1 | 硬的开关 | 平滑的调光旋钮（sigmoid） | 两个都一样 | 两个都不行 | concept: sigmoid | fb: 开关从 0 直接跳到 1，其他地方都是平的，所以一点小变化什么都改变不了。调光旋钮是滑动的，所以一点小变化就带来一点小改变。今天你会看到，就这一个区别几乎决定了一切。
q: 一个神经元只画一条直线来分开它的输入。下面哪种图案是它自己学不会的？ | a:1 | 一团「是」的点 | XOR（两类点分别在两条对角线上） | 全部点都在一侧 | 一个空数据集 | concept: xor-limit | fb: XOR 的点落在两条对角线上，没有任何一条直线能把它们分开 —— 这就是单个神经元的硬天花板，而层就是解决办法。
q: 一个神经元有两个输入时，有几个 weight、几个 bias？ | a:1 | 2 个 weight、2 个 bias | 2 个 weight、1 个 bias | 1 个 weight、1 个 bias | 1 个 weight、2 个 bias | concept: weights-bias | fb: 每个输入一个 weight（所以是两个），整个神经元只有一个 bias —— bias 是一个「起步偏向」的单个数字。
%%%
~~~

@@@ concept id=c1 tag="The trap" zh_tag="陷阱" title="Straight + straight is still straight" zh_title="直的加直的，还是直的" gotit="Got the trap" zh_gotit="看懂这个陷阱了"
Let's start with a puzzle you can hold in your hands. A neuron does a weighted sum and adds a bias — in plain words, a straight-line rule. Picture a perfectly straight **ruler**.

Now the puzzle: **what do you get if you stack two straight rulers?**
~~~zh
先从一个你能拿在手里的小谜题开始。一个 neuron（神经元）做的事情是：把每个输入乘上它的
weight（权重）加起来，再加一个 bias（偏置）。说白了，这就是一条**直线规则**。你可以想象一把
完全笔直的**尺子**。

那么谜题来了：**把两把直尺叠在一起，你会得到什么？**
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="Two straight rulers laid one on top of the other still make a single straight edge. The everyday picture for two straight-line layers folding into one."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Two straight rulers stacked = still one straight edge</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">两把直尺叠起来 = 还是一条直线</text><rect x="40" y="42" width="260" height="20" rx="3" fill="#FCF3DC" stroke="#C99A12"/><line x1="60" y1="52" x2="280" y2="52" stroke="#9A7A10" stroke-width="1"/><line x1="70" y1="46" x2="70" y2="58" stroke="#9A7A10"/><line x1="110" y1="46" x2="110" y2="58" stroke="#9A7A10"/><line x1="150" y1="46" x2="150" y2="58" stroke="#9A7A10"/><line x1="190" y1="46" x2="190" y2="58" stroke="#9A7A10"/><line x1="230" y1="46" x2="230" y2="58" stroke="#9A7A10"/><text class="lang-en" x="312" y="56" fill="#6B645E">ruler 1</text><text class="lang-zh" x="312" y="56" fill="#6B645E">尺子 1</text><rect x="40" y="66" width="260" height="20" rx="3" fill="#EFEAF7" stroke="#7C6DAA"/><line x1="60" y1="76" x2="280" y2="76" stroke="#5E5191" stroke-width="1"/><line x1="70" y1="70" x2="70" y2="82" stroke="#5E5191"/><line x1="110" y1="70" x2="110" y2="82" stroke="#5E5191"/><line x1="150" y1="70" x2="150" y2="82" stroke="#5E5191"/><line x1="190" y1="70" x2="190" y2="82" stroke="#5E5191"/><line x1="230" y1="70" x2="230" y2="82" stroke="#5E5191"/><text class="lang-en" x="312" y="80" fill="#6B645E">ruler 2</text><text class="lang-zh" x="312" y="80" fill="#6B645E">尺子 2</text><text class="lang-en" x="260" y="110" text-anchor="middle" fill="#6B645E">↓ stack them ↓</text><text class="lang-zh" x="260" y="110" text-anchor="middle" fill="#6B645E">↓ 叠起来 ↓</text><rect x="40" y="120" width="260" height="22" rx="3" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><line x1="60" y1="131" x2="280" y2="131" stroke="#C93B3B" stroke-width="1.4"/><text class="lang-en" x="316" y="135" fill="#C93B3B">still ONE straight edge</text><text class="lang-zh" x="316" y="135" fill="#C93B3B">还是一条直线</text><text class="lang-en" x="170" y="158" text-anchor="middle" fill="#C93B3B" font-size="10">no bend → depth adds nothing</text><text class="lang-zh" x="170" y="158" text-anchor="middle" fill="#C93B3B" font-size="10">没有弯 → 深度白搭</text></g></svg>
%%%

Answer: still one straight edge. Two neuron-layers with no bend between them fold back into one layer. Engineers call that fold-back the **linear collapse**.

**What the ruler gets right:** each layer really is a straight-line rule. **Where it breaks down:** a ruler is one fixed line, while a layer can tune many weights — but however it tunes them, the *shape* stays straight.
~~~zh
答案是：还是一条直线。两层神经元之间如果没有一个**弯**，它们会折回成一层。工程师给这个折回起了
个名字，叫 **linear collapse（线性坍缩）**。

**尺子这个比喻对在哪里：** 每一层确实就是一条直线规则。**它在哪里不成立：** 一把尺子只有一条
固定的线，而一层神经元可以调很多个 weight —— 但不管怎么调，形状始终是直的。
~~~

%%% insight
This is the sneaky part: a network with no bend still runs, still trains, still prints a loss that drops. Nothing on your screen ever says "your depth is fake." That's exactly why we go looking for the collapse with our own eyes instead of trusting the loss.
%%%
~~~zh
%%% insight
这里最阴的一点是：一个没有弯的网络照样能跑、照样能训练、照样打印出一个在下降的 loss（损失）。
你的屏幕上永远不会有一行字告诉你「你的深度是假的」。这正是为什么我们要用自己的眼睛去找这个坍缩，
而不是相信 loss。
%%%
~~~

#### Watch two rulers fold into one
Back to the ruler bench — with numbers this time, so you can check every rung yourself. Take the input `4` and give it two straight-line moves in a row.
~~~zh
#### 看两把尺子折成一把
回到那张放尺子的桌子 —— 这次带上数字，这样每一级你都能自己验算。拿输入 `4`，连着给它两个直线动作。
~~~

%%% steps
step: **Move 1 — "triple it."**  4 → 12
why: one straight-line layer: multiply, that's all
step: **Move 2 — "double that, add one."**  12 → 25
why: a second straight-line layer stacked on the first
step: **The fold.** One single rule — "six times it, plus one" — sends 4 → 25 too
why: the two moves quietly merged into ONE straight-line move; this is the linear collapse
step: **The bad news.** So two layers = one layer
why: a third, or a hundredth, folds in the same way — depth alone buys nothing
%%%
~~~zh
%%% steps
step: **第 1 个动作 —— 「乘三倍」。**  4 → 12
why: 一层直线规则：乘一下，就这样
step: **第 2 个动作 —— 「再乘二，加一」。**  12 → 25
why: 第二层直线规则，叠在第一层上面
step: **折叠。** 一条单独的规则 —— 「乘六倍，再加一」 —— 也把 4 送到 25
why: 两个动作悄悄合并成了一个直线动作；这就是 linear collapse
step: **坏消息。** 所以两层 = 一层
why: 第三层，或者第一百层，都会同样折进去 —— 光加深度什么也买不到
%%%
~~~

Don't take my word for it. Below, the two stacked moves are drawn as a thick line and the single combined move as a dashed line — and the dashed one lands right on top of the thick one, because they are the very same line:
~~~zh
别只听我说。下面这张图里，两个叠起来的动作画成一条粗线，那一个合并后的动作画成一条虚线 ——
虚线正好落在粗线上面，因为它们本来就是同一条线：
~~~

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="Two straight-line steps, triple-it then double-and-add-one, applied to 4 give 25, exactly matching the single combined rule six-times-plus-one. Below, both rules are drawn as one and the same straight line: the thick line for the two stacked moves lies exactly on top of the dashed line for the single move."><g font-family="monospace" font-size="12" text-anchor="middle">
<text x="45" y="35" fill="#3A342E" font-weight="bold">4</text>
<rect x="70" y="20" width="88" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="114" y="37" fill="#5E5191">×3 → 12</text>
<rect x="178" y="20" width="120" height="26" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="238" y="37" fill="#5E5191">×2 +1 → 25</text>
<text x="330" y="37" fill="#6B645E">=</text>
<rect x="352" y="18" width="140" height="30" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text x="422" y="38" fill="#C93B3B" font-weight="bold">6×4 +1 = 25</text>
<line x1="96" y1="66" x2="128" y2="66" stroke="#7C6DAA" stroke-width="7" opacity="0.4"/>
<text class="lang-en" x="136" y="70" fill="#5E5191" font-size="10" text-anchor="start">thick = the two stacked moves</text><text class="lang-zh" x="136" y="70" fill="#5E5191" font-size="10" text-anchor="start">粗线 = 两个叠起来的动作</text>
<line x1="96" y1="86" x2="128" y2="86" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="5,4"/>
<text class="lang-en" x="136" y="90" fill="#C93B3B" font-size="10" text-anchor="start">dashed = the one combined move — it lands on exactly the same line</text><text class="lang-zh" x="136" y="90" fill="#C93B3B" font-size="10" text-anchor="start">虚线 = 合并后的那一个动作 —— 它落在完全同一条线上</text>
<line x1="60" y1="195" x2="470" y2="195" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="80" y1="100" x2="80" y2="201" stroke="#E5DFD6" stroke-width="1"/>
<path d="M80 190 L440 110" fill="none" stroke="#7C6DAA" stroke-width="7" opacity="0.4"/>
<path d="M80 190 L440 110" fill="none" stroke="#C93B3B" stroke-width="1.8" stroke-dasharray="5,4"/>
<text class="lang-en" x="272" y="178" fill="#C93B3B" font-size="10" text-anchor="middle">one straight line, drawn twice</text><text class="lang-zh" x="272" y="178" fill="#C93B3B" font-size="10" text-anchor="middle">同一条直线，画了两遍</text>
<text class="lang-en" x="486" y="199" fill="#9A938A" font-size="10" text-anchor="middle">input</text><text class="lang-zh" x="486" y="199" fill="#9A938A" font-size="10" text-anchor="middle">输入</text>
</g></svg>
%%%

**So far:** two straight moves are always secretly one straight move.
~~~zh
**到这里为止：** 两个直线动作，背地里永远都是一个直线动作。
~~~

#### The same story with real layers
A layer's weights live in a grid of numbers called a matrix, written `W`. Stack two and the grid-math folds them together exactly like `×3` then `×2` folded into `×6`:
~~~zh
#### 换成真的层，故事一样
一层的 weight 住在一个数字网格里，这个网格叫 matrix（矩阵），写作 `W`。叠两层，网格的算法会把
它们折在一起，和 `×3` 接着 `×2` 折成 `×6` 完全一样：
~~~

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Two linear layers W1 then W2 collapse into a single matrix W1 times W2"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="40" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="70" y="65" fill="#5E5191">W₁</text><text class="lang-en" x="115" y="65" fill="#6B645E">then</text><text class="lang-zh" x="115" y="65" fill="#6B645E">接着</text><rect x="150" y="45" width="60" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="180" y="65" fill="#5E5191">W₂</text><text x="250" y="65" fill="#6B645E">=</text><rect x="290" y="40" width="140" height="40" rx="6" fill="#FDE8E8" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="360" y="65" fill="#C93B3B" font-weight="bold">one matrix W₁@W₂</text><text class="lang-zh" x="360" y="65" fill="#C93B3B" font-weight="bold">一个矩阵 W₁@W₂</text><text class="lang-en" x="260" y="105" fill="#C93B3B" font-size="11">no activation → two layers become one</text><text class="lang-zh" x="260" y="105" fill="#C93B3B" font-size="11">没有激活函数 → 两层变成一层</text></g></svg>
%%%

You'll prove that yourself in **Produce** — a couple of lines of output, one big idea.
~~~zh
这一点你会在 **Produce（动手做）** 里自己证明出来 —— 几行输出，一个大想法。
~~~

!!! c-info 🪜
<b>Optional peek (skippable).</b> The one-line version of the fold: two straight-line layers back to back are <code>(x·W₁)·W₂</code>, and grid-math lets you re-bracket that as <code>x·(W₁·W₂)</code> — a single set of weights. The biases fold too: <code>b₁·W₂ + b₂</code> is just one new bias. So the whole stack is still one straight-line-plus-offset rule. Nothing was gained; the second layer never had a chance to add anything new.
!!!
~~~zh
!!! c-info 🪜
<b>可跳过的一眼（选读）。</b>把折叠写成一行：两层背靠背的直线规则是 <code>(x·W₁)·W₂</code>，
而网格的算法允许你把括号换个位置，写成 <code>x·(W₁·W₂)</code> —— 也就是一套 weight。bias 也会
折进去：<code>b₁·W₂ + b₂</code> 就是一个新的 bias。所以整个叠起来的东西仍然只是一条「直线加偏移」
的规则。什么都没多出来；第二层从来没有机会加进任何新东西。
!!!
~~~

!!! c-warn 😕
<b>This step feels strange to almost everyone at first — that's completely normal.</b> The activation looks like an optional extra tacked on at the end, so it's easy to shrug off. But without it, 100 layers behave like <b>one</b>. Sit with that for a moment — it's the whole reason today matters. And you already understand it, which is a great place to be four minutes in.
!!!
~~~zh
!!! c-warn 😕
<b>这一步很多人一开始都会觉得奇怪，很正常。</b>激活函数看起来像是最后随手加上去的一个可选零件，
所以很容易被一带而过。但没有它，100 层的表现就和<b>一层</b>一样。停在这里想一会儿 —— 这就是今天
之所以重要的全部原因。而你已经看懂了，才过了四分钟就到这一步，很不错。
!!!
~~~

@@@ concept id=c2 tag="The fix" zh_tag="解决办法" title="A bend between the layers" zh_title="在层与层之间加一个弯" gotit="Got the fix" zh_gotit="懂了这个解决办法"
So how do we stop the collapse? We slip a **bend** between the layers.

Picture a sheet of **paper**. Flat paper on flat paper stays flat. Fold one crease into it and suddenly it stands up in shapes a flat sheet could never make. That crease is the whole idea.
~~~zh
那么怎么才能不让它坍缩？我们在层与层之间塞进一个**弯**。

想象一张**纸**。平的纸叠在平的纸上，还是平的。但只要折出一道折痕，它突然就能立起来，变成一张平
纸永远做不出的形状。这道折痕就是全部的想法。
~~~

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="Left: a flat sheet of paper stays flat. Right: the same sheet with one folded crease stands up into a shape. A single crease unlocks shapes flat paper cannot make."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One crease unlocks shapes flat paper can't make</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一道折痕就能做出平纸做不出的形状</text><polygon points="40,120 200,120 220,90 60,90" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="50" y1="110" x2="200" y2="110" stroke="#E5DFD6"/><text class="lang-en" x="128" y="145" text-anchor="middle" fill="#6B645E">flat paper</text><text class="lang-zh" x="128" y="145" text-anchor="middle" fill="#6B645E">平的纸</text><text class="lang-en" x="128" y="160" text-anchor="middle" fill="#9A938A" font-size="10">stays flat (no shape)</text><text class="lang-zh" x="128" y="160" text-anchor="middle" fill="#9A938A" font-size="10">还是平的（没有形状）</text><text x="255" y="105" fill="#B8AEA2" font-size="16">→</text><path d="M300 130 L390 130 L410 100 L360 60 L340 95 Z" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><line x1="360" y1="60" x2="340" y2="95" stroke="#5E5191" stroke-width="2.5"/><line x1="340" y1="95" x2="390" y2="130" stroke="#5E5191" stroke-width="1"/><text class="lang-en" x="420" y="92" fill="#5E5191">the crease</text><text class="lang-zh" x="420" y="92" fill="#5E5191">这道折痕</text><text class="lang-en" x="420" y="106" fill="#5E5191">= the bend</text><text class="lang-zh" x="420" y="106" fill="#5E5191">= 那个弯</text><text class="lang-en" x="350" y="152" text-anchor="middle" fill="#2D8B55">creased paper stands up</text><text class="lang-zh" x="350" y="152" text-anchor="middle" fill="#2D8B55">折过的纸能立起来</text><text class="lang-en" x="350" y="166" text-anchor="middle" fill="#9A938A" font-size="10">can't be squashed flat</text><text class="lang-zh" x="350" y="166" text-anchor="middle" fill="#9A938A" font-size="10">压不回平的</text></g></svg>
%%%

That bend has a name: the [[activation function||A simple function applied at the end of a neuron, after the weighted sum + bias. It adds non-linearity so stacked layers can learn complex shapes.]] — a small [[non-linearity||Not a straight line. A non-linear step between layers lets a deep network bend and combine features to learn complex patterns.]] applied at the very end of a neuron. Yesterday you called it the "decide" step; today you get to see what it is really *for*.

**What the crease gets right:** one fold unlocks shapes flat paper can't make. **Where it breaks down:** paper only creases where *you* fold it, while an activation bends the signal at *every* neuron on *every* example — far more folds than your hands could ever make.
~~~zh
这个弯有个名字：[[activation function||加在一个神经元最后的一个简单函数，在加权求和加 bias 之后。它带来非线性，让叠起来的层能学到复杂的形状。]]（激活函数）—— 一个加在神经元最末端的小小
[[non-linearity||不是直线。层与层之间有一个非线性的步骤，深层网络才能弯、才能组合特征去学复杂的图案。]]（非线性）。昨天你把它叫做「决定」这一拍；今天你能看到它到底是*为了什么*。

**折痕这个比喻对在哪里：** 一道折就能做出平纸做不出的形状。**它在哪里不成立：** 纸只在*你*折的
地方有折痕，而激活函数在*每一个*样本上、*每一个*神经元处都把信号弯一下 —— 远比你的手能折出的
次数多得多。
~~~

%%% insight
Notice how little this costs. You are not adding layers, parameters, or data — you're adding one tiny bend per neuron, and that alone is what turns "deep" from a marketing word into real extra power.
%%%
~~~zh
%%% insight
注意这件事有多便宜。你没有加层、没有加 parameter、也没有加数据 —— 你只是给每个神经元加了一个小
小的弯，而光是这一点，就把「深」从一个营销词变成了真正多出来的能力。
%%%
~~~

#### Where the bend goes, in three beats
Let's follow one number through a layer, so you know exactly which step is the bend.
~~~zh
#### 这个弯放在哪里，三拍讲完
我们跟着一个数字走过一层，这样你就清楚到底哪一步才是那个弯。
~~~

%%% steps
step: **Mix.** The weights combine the inputs into one running total, `z`
why: this is the straight-line part — mixing is the weights' job, and only theirs
step: **Bend.** The activation reshapes that single number, one number at a time
why: it takes one number in and gives one number out — that's why we call it the bend
step: **Pass on.** The bent value becomes the input of the next layer
why: therefore the next layer starts from something curved, and can't be folded back into the first
%%%
~~~zh
%%% steps
step: **混合。** weight 把所有输入合成一个总分 `z`
why: 这是直线的那部分 —— 混合是 weight 的活，而且只是它的活
step: **弯。** 激活函数把这一个数字重新塑形，一次只处理一个数字
why: 进来一个数字，出去一个数字 —— 所以我们叫它「弯」
step: **传下去。** 弯过的值成为下一层的输入
why: 因此下一层是从一个已经弯了的东西开始的，没法再折回第一层里去
%%%
~~~

%%% svg
<svg viewBox="0 0 520 120" role="img" aria-label="Inserting ReLU between the two layers stops the collapse"><g font-family="monospace" font-size="13" text-anchor="middle"><rect x="30" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="57" y="65" fill="#5E5191">W₁</text><rect x="100" y="45" width="70" height="30" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="135" y="65" fill="#1a5c38" font-weight="bold">ReLU</text><rect x="185" y="45" width="55" height="30" rx="5" fill="#EDE9F8" stroke="#7C6DAA" stroke-width="2"/><text x="212" y="65" fill="#5E5191">W₂</text><text x="275" y="65" fill="#6B645E">≠</text><rect x="305" y="45" width="150" height="30" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="2" stroke-dasharray="4,3"/><text class="lang-en" x="380" y="65" fill="#6B645E">any single W</text><text class="lang-zh" x="380" y="65" fill="#6B645E">任何单独一个 W</text><text class="lang-en" x="250" y="105" fill="#2D8B55" font-size="11">a bend in the middle → cannot be folded flat</text><text class="lang-zh" x="250" y="105" fill="#2D8B55" font-size="11">中间有一个弯 → 压不回平的</text></g></svg>
%%%

**So far:** weights mix, the bend curves, and that one curve is what keeps the layers separate. One honest limit while we're here: every bend you'll meet today works on **one number at a time**, so it can't build new combinations of features — combining inputs stays the weights' job. (There is exactly one exception, a bend called *softmax* that sits at the very end and looks at a whole row at once; that's Day 4's story.)

And that's the whole secret. **You just unlocked why deep learning is even possible.** The rest of today is meeting the bends themselves — starting with the one you already met yesterday.
~~~zh
**到这里为止：** weight 负责混合，弯负责弯曲，而正是这一个弯让各层保持分开。顺便说一个诚实的
限制：今天你会见到的每一个弯都**一次只处理一个数字**，所以它没法造出特征的新组合 —— 组合输入
始终是 weight 的活儿（只有一个例外，一个叫 *softmax* 的弯，它坐在最末端，一次看一整行；那是
第 4 天的故事。）

这就是全部的秘密。**你刚刚搞懂了深度学习为什么可能。** 今天剩下的部分就是去认识这些弯本身 ——
从你昨天已经见过的那一个开始。
~~~

@@@ concept id=c3 tag="The first bend" zh_tag="第一个弯" title="The step function — a hard on/off switch" zh_title="step function —— 一个硬的开关" gotit="Met the step" zh_gotit="见过 step 了"
Here's the on/off switch from yesterday again — but this time we put it on trial, because today you know what a bend is *for*.

Picture the light switch on your wall. Flip it and the light is fully **on** or fully **off** — nothing in between.
~~~zh
昨天那个开关又回来了 —— 但这次我们要审它一审，因为今天你已经知道一个弯是*为了什么*。

想象你家墙上的电灯开关。一按，灯要么**全亮**，要么**全灭** —— 中间没有任何状态。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A wall light switch in two states. On the left the switch is down, the bulb is dark and off. On the right the switch is up, the bulb is bright and on. Nothing in between."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A wall switch: fully OFF or fully ON — no in-between</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">墙上的开关：要么全灭，要么全亮 —— 没有中间</text><rect x="60" y="40" width="70" height="100" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><rect x="84" y="92" width="22" height="40" rx="4" fill="#D8D2C8" stroke="#8f8a80"/><text class="lang-en" x="95" y="156" text-anchor="middle" fill="#6B645E">switch down</text><text class="lang-zh" x="95" y="156" text-anchor="middle" fill="#6B645E">开关向下</text><circle cx="200" cy="70" r="20" fill="#EFEAF0" stroke="#9A938A" stroke-width="1.5"/><text x="200" y="76" text-anchor="middle" font-size="16">🌑</text><text class="lang-en" x="200" y="120" text-anchor="middle" fill="#6B645E">input &lt; 0</text><text class="lang-zh" x="200" y="120" text-anchor="middle" fill="#6B645E">输入 &lt; 0</text><text class="lang-en" x="200" y="136" text-anchor="middle" fill="#6B645E" font-weight="bold">output 0 (off)</text><text class="lang-zh" x="200" y="136" text-anchor="middle" fill="#6B645E" font-weight="bold">输出 0（灭）</text><rect x="330" y="40" width="70" height="100" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><rect x="354" y="48" width="22" height="40" rx="4" fill="#F2D97A" stroke="#9A7A10"/><text class="lang-en" x="365" y="156" text-anchor="middle" fill="#6B645E">switch up</text><text class="lang-zh" x="365" y="156" text-anchor="middle" fill="#6B645E">开关向上</text><circle cx="470" cy="70" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="470" y="76" text-anchor="middle" font-size="16">💡</text><text class="lang-en" x="470" y="120" text-anchor="middle" fill="#8A6D3B">input ≥ 0</text><text class="lang-zh" x="470" y="120" text-anchor="middle" fill="#8A6D3B">输入 ≥ 0</text><text class="lang-en" x="470" y="136" text-anchor="middle" fill="#8A6D3B" font-weight="bold">output 1 (on)</text><text class="lang-zh" x="470" y="136" text-anchor="middle" fill="#8A6D3B" font-weight="bold">输出 1（亮）</text></g></svg>
%%%

That is the [[step function||The original activation: output 1 if the input is at or above zero, else 0. A hard on/off switch with no in-between — and slope 0 everywhere, so a network can't learn from it.]], the earliest of all these bends — the one yesterday's theme-park height bar was describing. In words: input at or above zero → output `1` (the neuron "fires"); below zero → output `0`.

**What the switch gets right:** all-or-nothing, no middle ground. **Where it breaks down:** a wall switch is flipped by *you*; here the *input* flips it — and there's no dimming, which turns out to be fatal.
~~~zh
这就是 [[step function||最早的激活函数：输入大于或等于零就输出 1，否则输出 0。一个没有中间状态的硬开关 —— 而且它处处斜率为 0，所以网络没法从它身上学到东西。]]，所有这些弯里最早的
一个 —— 昨天游乐园那根身高杆说的就是它。用大白话讲：输入大于或等于零 → 输出 `1`（神经元「发火」）；
小于零 → 输出 `0`。

**开关这个比喻对在哪里：** 全有或全无，没有中间地带。**它在哪里不成立：** 墙上的开关是*你*去按的；
这里是*输入*在按它 —— 而且没有调光这回事，这一点后来被证明是致命的。
~~~

%%% demo id=step label="predict, then run"
predict: what does the step function give for the inputs −2, −1, 0, 1, 2? Guess the five outputs before you peek — where exactly does it flip?
code: step(np.array([-2, -1, 0, 1, 2]))
out: array([0., 0., 1., 1., 1.])
take: <b>Step = 1 if z ≥ 0 else 0.</b> Everything below zero is 0, everything at or above zero is 1. (They print as `0.` and `1.` because the code returns decimals.) Notice what's missing: any sense of "how much." Only yes or no.
%%%
~~~zh
%%% demo id=stepzh label="先猜，再展开"
predict: step function 对输入 −2、−1、0、1、2 分别给出什么？先猜这五个输出，再看答案 —— 它究竟在哪里翻转？
code: step(np.array([-2, -1, 0, 1, 2]))
out: array([0., 0., 1., 1., 1.])
take: <b>z ≥ 0 就是 1，否则是 0。</b>零以下全是 0，零和零以上全是 1。（打印成 `0.` 和 `1.` 是因为代码返回的是小数。）注意少了什么：完全没有「多少」这个概念。只有是或不是。
%%%
~~~

#### The switch's fatal flaw
Here's the puzzle it leaves us: the step function looks tidy, so why does nobody use it inside a network today? Look at its shape.
~~~zh
#### 这个开关的致命缺陷
它留给我们一个谜题：step function 看起来很整齐，那今天为什么没有人在网络内部用它？看看它的形状。
~~~

%%% svg
<svg viewBox="0 0 520 140" role="img" aria-label="Step function: output 0 for negative inputs, jumps to 1 at zero, then flat at 1 for positive inputs"><g font-family="monospace"><line x1="60" y1="100" x2="330" y2="100" stroke="#E5DFD6" stroke-width="1.5"/><line x1="195" y1="30" x2="195" y2="118" stroke="#E5DFD6" stroke-width="1"/><path d="M70 100 L195 100 L195 48 L305 48" fill="none" stroke="#8A6D3B" stroke-width="2.5"/><circle cx="195" cy="48" r="3.5" fill="#8A6D3B"/><text class="lang-en" x="125" y="117" font-size="11" fill="#6B645E" text-anchor="middle">output 0 (off)</text><text class="lang-zh" x="125" y="117" font-size="11" fill="#6B645E" text-anchor="middle">输出 0（灭）</text><text class="lang-en" x="255" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">output 1 (on)</text><text class="lang-zh" x="255" y="41" font-size="11" fill="#8A6D3B" text-anchor="middle">输出 1（亮）</text><text class="lang-en" x="415" y="72" font-size="12" fill="#8A6D3B">step → {0, 1}</text><text class="lang-zh" x="415" y="72" font-size="12" fill="#8A6D3B">step → {0, 1}</text><text class="lang-en" x="195" y="133" font-size="10" fill="#9A938A" text-anchor="middle">jump at 0 — no in-between, and flat (slope 0) everywhere else</text><text class="lang-zh" x="195" y="133" font-size="10" fill="#9A938A" text-anchor="middle">在 0 处跳一下 —— 没有中间，其他地方全是平的（斜率 0）</text></g></svg>
%%%

Flat, flat, jump, flat. And "flat" is exactly the wrong thing to be:
~~~zh
平、平、跳一下、平。而「平」恰恰是最不该是的样子：
~~~

%%% steps
step: **The nudge.** Learning improves a network by nudging weights in whichever direction lowers the error
why: to nudge, it needs a hint about which way is "downhill"
step: **The flat stretch.** Wiggle the input of a step function and the output doesn't budge at all
why: therefore there is no hint — flat means no direction, no signal
step: **The escape.** Keep the S-shape, but make it *smooth* so it always has some tilt
why: which is exactly what sigmoid and tanh do — that swap is why yesterday's dimmer exists
%%%
~~~zh
%%% steps
step: **轻推。** 学习的方式是：往能把误差降下去的方向，轻轻推一下 weight
why: 要推，它需要一个提示，告诉它哪边是「下坡」
step: **那段平的地方。** 你把 step function 的输入晃一下，输出一点都不动
why: 所以根本没有提示 —— 平就意味着没有方向、没有信号
step: **出路。** 保留这个 S 形，但让它*平滑*，这样它处处都有一点倾斜
why: 这正是 sigmoid 和 tanh 做的事 —— 昨天那个调光旋钮就是为了这次替换才存在的
%%%
~~~

%%% insight
Feel the shape of today's whole story here: the enemy is never the curve you can see — it's **flatness**. Every failure you'll meet from now on is some part of a bend going flat, and every cure puts the tilt back.
%%%
~~~zh
%%% insight
在这里先感受一下今天整个故事的形状：敌人从来不是你看得见的那条曲线 —— 而是**平**。从现在开始你
遇到的每一个失败，都是某个弯的某一段变平了；而每一个解药，都是把那点倾斜放回去。
%%%
~~~

**So far:** the step switch works, but it can't teach. And now you can explain why the very first neural networks stalled for years — nice.
~~~zh
**到这里为止：** step 这个开关能用，但它教不了东西。而现在你能解释为什么最早的那批神经网络卡了
好多年 —— 厉害。
~~~

@@@ concept id=c4 tag="Meet ReLU" zh_tag="认识 ReLU" title="ReLU — a one-way valve" zh_title="ReLU —— 一个单向阀门" gotit="Met ReLU" zh_gotit="见过 ReLU 了"
Now the first bend that's brand new to you — and it's the one that actually runs the modern world. It is almost as simple as the switch, but it keeps a tilt.

Picture a **one-way valve** in a water pipe: push from the front and water flows straight through; push from the back and it's blocked.
~~~zh
现在是第一个对你来说全新的弯 —— 而且它是真正在支撑现代世界的那一个。它几乎和开关一样简单，但它
保留了一点倾斜。

想象水管里的一个**单向阀门**：从正面推，水直接流过去；从背面推，就被挡住。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve in a pipe. Top: a positive signal pushes forward and water flows straight through. Bottom: a negative signal is blocked and nothing passes, output zero."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">ReLU = a one-way valve</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">ReLU = 一个单向阀门</text><rect x="40" y="38" width="440" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><polygon points="250,42 262,55 250,68" fill="#2D8B55"/><line x1="250" y1="38" x2="250" y2="72" stroke="#2D8B55" stroke-width="1"/><text class="lang-en" x="90" y="59" fill="#276b45">positive →</text><text class="lang-zh" x="90" y="59" fill="#276b45">正数 →</text><text class="lang-en" x="330" y="59" fill="#276b45" font-weight="bold">flows through 💧</text><text class="lang-zh" x="330" y="59" fill="#276b45" font-weight="bold">流过去 💧</text><text x="470" y="59" text-anchor="end" fill="#2D8B55">→</text><rect x="40" y="96" width="440" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><line x1="250" y1="92" x2="250" y2="134" stroke="#C93B3B" stroke-width="3"/><rect x="244" y="100" width="12" height="26" fill="#C93B3B"/><text class="lang-en" x="95" y="117" fill="#8a3b3b">negative ←</text><text class="lang-zh" x="95" y="117" fill="#8a3b3b">负数 ←</text><text class="lang-en" x="330" y="117" fill="#C93B3B" font-weight="bold">BLOCKED → 0</text><text class="lang-zh" x="330" y="117" fill="#C93B3B" font-weight="bold">被挡住 → 0</text><text class="lang-en" x="260" y="154" text-anchor="middle" fill="#6B645E" font-size="10">flows one way, blocked the other — that's max(0, z)</text><text class="lang-zh" x="260" y="154" text-anchor="middle" fill="#6B645E" font-size="10">一个方向流过，另一个方向挡住 —— 这就是 max(0, z)</text></g></svg>
%%%

That is [[ReLU||A one-way valve: it lets positive numbers pass and turns negatives into 0. The modern default activation.]]: positives pass through untouched, negatives are shut off to `0`. One line of plain words — and it is the bend inside almost every network you use without noticing: the one that unlocks your phone with your face, and the one that picks the next video in your feed. The big language models use a close cousin of it — a smoothed ReLU called **GELU**, which you'll meet in a moment.

**What the valve gets right:** flow one way, blocked the other. **Where it breaks down:** a real valve is open or shut, but ReLU passes positives at their *full* size — 5 stays 5, 100 stays 100.
~~~zh
这就是 [[ReLU||一个单向阀门：让正数通过，把负数变成 0。现代的默认激活函数。]]：正数原样通过，负数被关成 `0`。一行大白话 —— 而它就是你天天在用却没注意到的
几乎每个网络里的那个弯：用你的脸解锁手机的是它，替你挑下一个视频的也是它。大语言模型用的是它的
一个近亲 —— 一个被磨平了的 ReLU，叫 **GELU**，等一下你就会见到。

**阀门这个比喻对在哪里：** 一个方向通，另一个方向堵。**它在哪里不成立：** 真的阀门只有开和关，
而 ReLU 让正数以*原本的大小*通过 —— 5 还是 5，100 还是 100。
~~~

%%% demo id=relu label="predict, then run"
predict: before you look — what does ReLU do to −3, −1, 0, 2, 5? Which numbers survive, and do the survivors change size?
code: relu(np.array([-3, -1, 0, 2, 5]))
out: array([0, 0, 0, 2, 5])
take: <b>ReLU = max(0, z)</b> — "take the bigger of 0 and z." Negatives become 0; positives come out completely unchanged.
%%%
~~~zh
%%% demo id=reluzh label="先猜，再展开"
predict: 先别看 —— ReLU 对 −3、−1、0、2、5 会做什么？哪些数字活下来了，活下来的大小变了吗？
code: relu(np.array([-3, -1, 0, 2, 5]))
out: array([0, 0, 0, 2, 5])
take: <b>ReLU = max(0, z)</b> —— 「取 0 和 z 里大的那个」。负数变成 0；正数原封不动地出来。
%%%
~~~

Said out loud, the formula is friendly: `max(0, z)` means "keep whichever is bigger, zero or z." If `z` is positive, `z` wins. If `z` is negative, `0` wins. That's it — no `e`, no exponent, one comparison.

#### Three gifts from one valve
Back at the pipe: the valve's shape hands us three separate wins. Watch each one appear.
~~~zh
念出来的话，这个公式很友好：`max(0, z)` 就是「0 和 z 里，留下大的那个」。如果 `z` 是正的，`z` 赢。
如果 `z` 是负的，`0` 赢。就这样 —— 没有 `e`，没有指数，一次比较。

#### 一个阀门送来的三份礼物
回到水管边：阀门的形状一下给了我们三个独立的好处。看它们一个个出现。
~~~

%%% svg
<svg viewBox="0 0 520 130" role="img" aria-label="ReLU curve: flat at zero for negatives, a straight rising line for positives"><g font-family="monospace"><line x1="60" y1="95" x2="320" y2="95" stroke="#E5DFD6" stroke-width="1.5"/><line x1="190" y1="25" x2="190" y2="110" stroke="#E5DFD6" stroke-width="1"/><path d="M70 95 L190 95 L300 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text class="lang-en" x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">negatives → 0</text><text class="lang-zh" x="120" y="112" font-size="11" fill="#6B645E" text-anchor="middle">负数 → 0</text><text class="lang-en" x="260" y="30" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text class="lang-zh" x="260" y="30" font-size="11" fill="#2D8B55" text-anchor="middle">正数通过</text><text x="400" y="70" font-size="13" fill="#1a5c38" font-family="monospace">ReLU(x)=max(0,x)</text></g></svg>
%%%

%%% steps
step: **Gift 1 — a real tilt.** On the positive side the line rises at a steady slope of `1`
why: unlike the flat step switch, learning gets a clear "this way" signal there
step: **Gift 2 — it's dirt cheap.** One comparison per number, no exponent to compute
why: which means you can afford millions of them in every layer of a big model
step: **Gift 3 — free [[sparsity||Many neurons output exactly 0 at the same time, so only part of the network is active per example — cheaper, and "only the units that matter speak up."]].** Roughly half the units get a negative input, so they output exactly `0`
why: therefore only the units that matter speak up on any one example — quieter and cheaper
%%%
~~~zh
%%% steps
step: **礼物 1 —— 真正的倾斜。** 在正的那一侧，这条线以稳定的斜率 `1` 往上走
why: 和平坦的 step 开关不一样，学习在那里能拿到一个清楚的「往这边」的信号
step: **礼物 2 —— 便宜到不要钱。** 每个数字一次比较，没有指数要算
why: 也就是说，大模型的每一层里放几百万个你都负担得起
step: **礼物 3 —— 白送的 [[sparsity||很多神经元同时输出正好 0，所以每个样本只用到网络的一部分 —— 更便宜，而且「只有真正重要的单元才说话」。]]（稀疏性）。** 大约一半的单元拿到的是负输入，于是输出正好 `0`
why: 因此在任何一个样本上，只有真正重要的那些单元会说话 —— 更安静也更便宜
%%%
~~~

Its range is `[0, ∞)` — zero and up, with no ceiling. Push the input higher and the output keeps climbing; ReLU never flattens on the positive side, which is the property the next few units all lean on.
~~~zh
它的取值范围是 `[0, ∞)` —— 从零往上，没有上限。把输入推得更高，输出就一直往上爬；ReLU 在正的
那一侧永远不会变平，接下来几个单元靠的都是这个性质。
~~~

%%% insight
One question people love to ask: what's the slope at exactly `z = 0`, where the two straight pieces meet in a kink? Honest answer: there isn't one. Everyone just agrees to call it `0` and moves on — landing *exactly* on zero in floating-point numbers basically never happens.
%%%
~~~zh
%%% insight
有一个问题大家特别爱问：在正好 `z = 0`、两段直线交出一个折角的地方，斜率是多少？诚实的答案是：
没有。大家就一致约定把它叫做 `0`，然后继续往下走 —— 在浮点数里*正好*落在零上，基本上不会发生。
%%%
~~~

**So far:** cheap, tilted, sparse — that trio is why ReLU is the default bend almost everywhere.
~~~zh
**到这里为止：** 便宜、有倾斜、稀疏 —— 就是这三样，让 ReLU 几乎在所有地方都成了默认的那个弯。
~~~

@@@ concept id=c5 tag="Meet sigmoid" zh_tag="认识 sigmoid" title="Sigmoid — a soft dimmer" zh_title="sigmoid —— 一个柔和的调光旋钮" gotit="Met sigmoid" zh_gotit="见过 sigmoid 了"
Back to the gentle bend you already slid yesterday — the one behind every "how confident are you, 0 to 100%?" answer a model ever gives. Today we look at its two flat ends, and at one quirk that made people invent a whole other bend.

Picture that **dimmer switch** instead of an on/off switch. Slide it and the room brightens smoothly through every level in between.
~~~zh
回到你昨天已经滑过的那个温和的弯 —— 模型每次回答「你有多确定，0 到 100%？」背后都是它。今天我们
看它的两个平坦的末端，以及一个让人们发明出另一个弯的小毛病。

把那个开关换成**调光旋钮**。滑动它，房间会平滑地经过中间每一个亮度慢慢变亮。
~~~

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="A dimmer switch as a slider track. On the left, the slider is low and the bulb is dark near zero. In the middle it is half bright at one half. On the right it is full and the bulb is fully bright near one. It slides smoothly through every level."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid = a dimmer that slides smoothly off → on</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid = 一个从灭平滑滑到亮的旋钮</text><rect x="60" y="70" width="400" height="14" rx="7" fill="#EFEAF7" stroke="#7C6DAA"/><circle cx="110" cy="77" r="12" fill="#5E5191"/><text class="lang-en" x="110" y="106" text-anchor="middle" fill="#6B645E">low</text><text class="lang-zh" x="110" y="106" text-anchor="middle" fill="#6B645E">低</text><text class="lang-en" x="110" y="120" text-anchor="middle" fill="#9A938A" font-size="10">→ near 0</text><text class="lang-zh" x="110" y="120" text-anchor="middle" fill="#9A938A" font-size="10">→ 接近 0</text><circle cx="260" cy="52" r="16" fill="#EFEAF0" stroke="#9A938A"/><text x="260" y="58" text-anchor="middle" font-size="14">🌑</text><circle cx="350" cy="48" r="18" fill="#F6EFCF" stroke="#C99A12"/><text x="350" y="54" text-anchor="middle" font-size="14">🔆</text><circle cx="430" cy="44" r="20" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="430" y="51" text-anchor="middle" font-size="16">💡</text><text class="lang-en" x="260" y="106" text-anchor="middle" fill="#5E5191">middle</text><text class="lang-zh" x="260" y="106" text-anchor="middle" fill="#5E5191">中间</text><text x="260" y="120" text-anchor="middle" fill="#9A938A" font-size="10">= 0.5</text><text class="lang-en" x="430" y="106" text-anchor="middle" fill="#8A6D3B">full</text><text class="lang-zh" x="430" y="106" text-anchor="middle" fill="#8A6D3B">满</text><text class="lang-en" x="430" y="120" text-anchor="middle" fill="#9A938A" font-size="10">→ near 1</text><text class="lang-zh" x="430" y="120" text-anchor="middle" fill="#9A938A" font-size="10">→ 接近 1</text><text class="lang-en" x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">no hard jump — every brightness in between lives in (0, 1)</text><text class="lang-zh" x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">没有硬跳跃 —— 中间每一个亮度都住在 (0, 1) 里</text></g></svg>
%%%

That smooth slide is [[sigmoid||A soft dimmer switch that gently squashes any number into the range 0 to 1.]]. Hand it any number at all and it hands back a brightness between `0` and `1` — never below, never above.

**What the dimmer gets right:** the gradual slide, no hard jump. **Where it breaks down:** a real dimmer is evenly graded all the way along, but sigmoid *flattens* at both ends — slide past a point and the bulb barely changes. Hold onto that; it's a puzzle we crack shortly.
~~~zh
这个平滑的滑动就是 [[sigmoid||一个柔和的调光旋钮，把任何数字轻轻挤进 0 到 1 之间。]]。给它任何一个数字，它都还给你一个介于 `0` 和 `1` 之间的亮度 ——
不会更低，也不会更高。

**旋钮这个比喻对在哪里：** 渐进的滑动，没有硬跳跃。**它在哪里不成立：** 真的旋钮从头到尾刻度是
均匀的，而 sigmoid 在两头都会*变平* —— 滑过某个点之后，灯泡几乎不再变化。记住这一点；这是我们
很快要破的一个谜题。
~~~

%%% demo id=sigmoid label="predict, then run"
predict: sigmoid turns any number into a brightness in (0, 1). Guess its answer for −2, 0 and 2 — and especially: what do you think sigmoid(0) is?
code: sigmoid(np.array([-2, 0, 2]))
out: array([0.119, 0.5  , 0.881])
take: <b>Sigmoid squashes into (0, 1).</b> Dead centre, sigmoid(0) = 0.5 — half brightness. Big negatives land near 0, big positives near 1 (values rounded; sigmoid(2) ≈ 0.8808).
%%%
~~~zh
%%% demo id=sigmoidzh label="先猜，再展开"
predict: sigmoid 把任何数字变成 (0, 1) 里的一个亮度。猜一猜它对 −2、0、2 的答案 —— 特别是：你觉得 sigmoid(0) 是多少？
code: sigmoid(np.array([-2, 0, 2]))
out: array([0.119, 0.5  , 0.881])
take: <b>Sigmoid 把数字挤进 (0, 1)。</b>正中间，sigmoid(0) = 0.5 —— 半亮。很大的负数落在 0 附近，很大的正数落在 1 附近（数值四舍五入过；sigmoid(2) ≈ 0.8808）。
%%%
~~~

#### Reading its shape, one feature at a time
Same dimmer, now drawn as a curve on real axes — the floor of the picture is `0`, the dashed ceiling is `1`, and the dot marks half brightness right at `z = 0`:
~~~zh
#### 一次读它形状里的一个特征
同一个旋钮，现在画成真实坐标轴上的一条曲线 —— 图的地板是 `0`，虚线的天花板是 `1`，那个点标出
正好在 `z = 0` 处的半亮：
~~~

%%% svg
<svg viewBox="0 0 520 145" role="img" aria-label="Sigmoid curve on labelled axes. The horizontal axis is the output value zero. A dashed ceiling line marks one. The S-shaped curve rises from just above zero on the left, passes exactly through one half at z equals zero, and levels off just under one on the right, so the whole curve stays inside zero to one."><g font-family="monospace"><line x1="60" y1="110" x2="330" y2="110" stroke="#E5DFD6" stroke-width="1.5"/><line x1="60" y1="50" x2="330" y2="50" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/><text x="56" y="54" font-size="10" fill="#7C6DAA" text-anchor="end">1.0</text><text x="56" y="114" font-size="10" fill="#9A938A" text-anchor="end">0.0</text><line x1="190" y1="40" x2="190" y2="122" stroke="#E5DFD6" stroke-width="1"/><path d="M70 109 C 130 109, 162 82, 190 80 C 218 78, 250 51, 310 50" fill="none" stroke="#7C6DAA" stroke-width="2.5"/><circle cx="190" cy="80" r="3.5" fill="#C93B3B"/><text class="lang-en" x="152" y="74" font-size="10" fill="#C93B3B" text-anchor="end">0.5 at z=0</text><text class="lang-zh" x="152" y="74" font-size="10" fill="#C93B3B" text-anchor="end">z=0 时是 0.5</text><text class="lang-en" x="115" y="126" font-size="11" fill="#6B645E" text-anchor="middle">→ near 0</text><text class="lang-zh" x="115" y="126" font-size="11" fill="#6B645E" text-anchor="middle">→ 接近 0</text><text class="lang-en" x="272" y="44" font-size="11" fill="#5E5191" text-anchor="middle">→ near 1</text><text class="lang-zh" x="272" y="44" font-size="11" fill="#5E5191" text-anchor="middle">→ 接近 1</text><text x="398" y="86" font-size="12" fill="#5E5191" font-family="monospace">sigmoid → (0,1)</text><text class="lang-en" x="190" y="138" font-size="10" fill="#9A938A" text-anchor="middle">the curve never leaves the band between 0 and 1</text><text class="lang-zh" x="190" y="138" font-size="10" fill="#9A938A" text-anchor="middle">这条曲线从不离开 0 和 1 之间的带子</text></g></svg>
%%%

%%% steps
step: **The range.** Every output lands strictly between `0` and `1`
why: which is why sigmoid reads like a confidence dial — perfect for "how sure are you?"
step: **The one-way climb.** Bigger input always means brighter output, never dimmer
why: it only ever goes up, so the answer is never ambiguous
step: **The two flat ends.** Far out on either side the curve lies almost flat
why: therefore a very confident unit stops reacting — remember flat means no learning signal
%%%
~~~zh
%%% steps
step: **取值范围。** 每个输出都严格落在 `0` 和 `1` 之间
why: 所以 sigmoid 读起来像一个信心刻度盘 —— 特别适合回答「你有多确定？」
step: **只往一个方向爬。** 输入越大，输出一定越亮，绝不会更暗
why: 它只会往上走，所以答案永远不会含糊
step: **两个平坦的末端。** 在两侧远处，曲线几乎是平的
why: 因此一个非常确定的单元就不再反应了 —— 记住，平就意味着没有学习信号
%%%
~~~

%%% insight
This is the bend that made deep learning *possible* in the 1990s and then quietly held it back for a decade. Same curve, two opposite verdicts — the reason is hiding in those flat ends, and you're a couple of units away from seeing it.
%%%
~~~zh
%%% insight
就是这个弯让深度学习在 1990 年代*成为可能*，然后又悄悄把它拖了十年。同一条曲线，两个相反的判决 ——
原因就藏在那两个平坦的末端里，而你再过两个单元就会看到它。
%%%
~~~

In symbols it's `sigmoid(z) = 1 / (1 + e^(−z))`: "one, divided by one plus e-to-the-minus-z." You never compute that by hand.
~~~zh
写成符号是 `sigmoid(z) = 1 / (1 + e^(−z))`：「一，除以一加 e 的负 z 次方」。你永远不需要手算它。
~~~

!!! c-info 🧪
<b>Optional (skippable) — one practical wrinkle.</b> If you ever hand-write that formula, a very negative <code>z</code> makes <code>e^(−z)</code> enormous and your computer can overflow to <code>inf</code>. The fix is free: call the library version (<code>torch.sigmoid</code>, <code>jax.nn.sigmoid</code>), which handles the two sides separately and never blows up.
!!!
~~~zh
!!! c-info 🧪
<b>选读（可跳过）—— 一个实际中的小坑。</b>如果你真的手写这个公式，一个很负的 <code>z</code> 会让
<code>e^(−z)</code> 变得极大，你的计算机可能溢出成 <code>inf</code>。解决办法不要钱：调用库里的
版本（<code>torch.sigmoid</code>、<code>jax.nn.sigmoid</code>），它会把两边分开处理，永远不会炸。
!!!
~~~

#### One more quirk: the dimmer only ever pushes one way
Here's a small, gentler mystery — and it's the reason the *next* bend exists. Every sigmoid output is a positive brightness. Nothing in `(0, 1)` is ever negative. Sounds harmless, right?

Picture pushing a **shopping cart with one stuck wheel** that always pulls a little to the left. You still reach the milk. You just zig-zag the whole way there.
~~~zh
#### 还有一个小毛病：这个旋钮只会往一个方向推
这里还有一个小一点、温和一点的谜题 —— 而它正是*下一个*弯存在的理由。sigmoid 的每一个输出都是一个
正的亮度。`(0, 1)` 里没有任何东西是负的。听起来没什么害处，对吧？

想象你推一辆**有一个轮子卡住的购物车**，那个轮子总是往左边带一点。你还是能拿到牛奶。你只是一路
走成了 Z 字形。
~~~

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A shopping cart with one wheel stuck turned to the side so it always pulls left. Its path zig-zags across the floor toward the goal instead of rolling straight, so it takes longer."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid's one-sided pull = a cart with one stuck wheel</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Sigmoid 的单向拉力 = 一个轮子卡住的购物车</text><rect x="48" y="48" width="70" height="40" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><line x1="56" y1="56" x2="56" y2="88" stroke="#7C6DAA"/><line x1="110" y1="48" x2="122" y2="36" stroke="#7C6DAA" stroke-width="1.5"/><circle cx="122" cy="34" r="4" fill="#7C6DAA"/><circle cx="62" cy="98" r="9" fill="#FDF9F3" stroke="#6B645E" stroke-width="1.5"/><circle cx="104" cy="98" r="9" fill="#FDF9F3" stroke="#C93B3B" stroke-width="2.5"/><line x1="98" y1="92" x2="110" y2="104" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="104" y="124" text-anchor="middle" fill="#C93B3B" font-size="9">stuck wheel</text><text class="lang-zh" x="104" y="124" text-anchor="middle" fill="#C93B3B" font-size="9">卡住的轮子</text><path d="M150 100 L200 66 L245 100 L295 62 L340 100 L390 60 L435 88" fill="none" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="300" y="122" text-anchor="middle" fill="#C93B3B" font-size="10">always-positive nudges → zig-zag, slow</text><text class="lang-zh" x="300" y="122" text-anchor="middle" fill="#C93B3B" font-size="10">永远为正的推力 → Z 字形，慢</text><path d="M150 100 L435 60" fill="none" stroke="#2D8B55" stroke-width="1.6" stroke-dasharray="5,4"/><text class="lang-en" x="320" y="72" fill="#2D8B55" font-size="10">straight (zero-centered)</text><text class="lang-zh" x="320" y="72" fill="#2D8B55" font-size="10">直线（以零为中心）</text><circle cx="440" cy="58" r="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="440" y="46" text-anchor="middle" fill="#6B645E" font-size="10">goal</text><text class="lang-zh" x="440" y="46" text-anchor="middle" fill="#6B645E" font-size="10">目标</text></g></svg>
%%%

**What the stuck wheel gets right:** a constant sideways pull really does force a zig-zag. **Where it breaks down:** a stuck cart may never arrive, while a zig-zagging network usually does — it simply takes more steps.
~~~zh
**卡住的轮子这个比喻对在哪里：** 一个恒定的侧向拉力真的会逼出 Z 字形。**它在哪里不成立：** 卡住的
车可能永远到不了，而走 Z 字形的网络通常还是能到 —— 只是要多走几步。
~~~

%%% steps
step: **The one-sided output.** Sigmoid can only ever hand the next layer *positive* numbers
why: everything it produces lives in `(0, 1)`, so a negative value is impossible for it
step: **The shared shove.** So the nudges arriving at a downstream weight all point the same way at once
why: which means the network can't correct one direction without dragging the others along — that's the stuck wheel
step: **The fix, in one word: balance.** We want a bend whose outputs sit evenly above *and* below `0`
why: therefore nudges come in both signs, the sideways pull cancels, and the path straightens
%%%
~~~zh
%%% steps
step: **单侧的输出。** sigmoid 只能给下一层*正的*数字
why: 它产出的一切都住在 `(0, 1)` 里，所以对它来说负值根本不可能
step: **共享的那一推。** 于是到达下游某个 weight 的那些推力，会同时指向同一个方向
why: 也就是说，网络没法只纠正一个方向而不把其他方向一起拖着走 —— 这就是那个卡住的轮子
step: **解决办法，一个词：平衡。** 我们想要一个弯，它的输出均匀地分布在 `0` 的上面*和*下面
why: 因此推力会有两种符号，侧向的拉力互相抵消，路径就被拉直了
%%%
~~~

%%% insight
Nothing breaks here — no crash, no error. The network still arrives; it just wanders on the way. Keep that in your pocket: most of today's failures are like this. They don't break your code, they quietly tax it.
%%%
~~~zh
%%% insight
这里什么都没坏 —— 不崩溃，不报错。网络照样能到；它只是在路上绕了一圈。把这句话放进口袋：今天的
大多数失败都是这个样子的。它们不会弄坏你的代码，它们只是悄悄地向你收税。
%%%
~~~

**So far:** sigmoid is a smooth confidence dial with two suspiciously flat ends and a one-sided pull. The bend that fixes the pull is next — and it has the best analogy of the day.
~~~zh
**到这里为止：** sigmoid 是一个平滑的信心刻度盘，带着两个可疑的平坦末端和一个单向的拉力。修掉这个
拉力的那个弯就在下一节 —— 而它有今天最好的一个比喻。
~~~

@@@ concept id=c6 tag="Meet tanh" zh_tag="认识 tanh" title="tanh — the balanced see-saw" zh_title="tanh —— 平衡的跷跷板" gotit="Met tanh" zh_gotit="见过 tanh 了"
So we want a smooth bend that pushes both ways. Someone had exactly that idea, and the everyday picture for it is the best one today — plus there's a live widget at the end so you can feel all these bends under your own finger.

Picture a playground **see-saw**: a plank resting on a bar in the middle. Nobody on it → perfectly level. Push the left side down hard → left end hits the ground. Push the right side → right end goes all the way up.
~~~zh
所以我们想要一个平滑的、能往两边推的弯。有人正好想到了这个，而它的日常比喻是今天最好的一个 ——
而且最后还有一个可以动的小控件，让你能亲手感受这些弯。

想象操场上的**跷跷板**：一块板子搁在中间的横杆上。没人坐 → 完全水平。把左边狠狠往下压 → 左端
碰到地面。压右边 → 右端一路升到最高。
~~~

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A playground see-saw. A plank rests on a central triangular pivot, tilted so its left end drops to minus one and its right end rises to plus one, resting level at zero in the middle."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">tanh = a balanced see-saw: −1 … 0 … +1</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">tanh = 平衡的跷跷板：−1 … 0 … +1</text><line x1="90" y1="120" x2="430" y2="70" stroke="#9A5A12" stroke-width="6" stroke-linecap="round"/><polygon points="260,95 240,140 280,140" fill="#C99A12" stroke="#9A7A10"/><line x1="170" y1="140" x2="350" y2="140" stroke="#B8AEA2" stroke-width="2"/><circle cx="90" cy="120" r="7" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="90" y="150" text-anchor="middle" fill="#C93B3B">push left</text><text class="lang-zh" x="90" y="150" text-anchor="middle" fill="#C93B3B">压左边</text><text x="90" y="164" text-anchor="middle" fill="#C93B3B" font-weight="bold">−1</text><circle cx="260" cy="95" r="5" fill="#6B645E"/><text class="lang-en" x="290" y="90" fill="#6B645E">rests level</text><text class="lang-zh" x="290" y="90" fill="#6B645E">静止时水平</text><text x="260" y="64" text-anchor="middle" fill="#2D8B55" font-weight="bold">0</text><circle cx="430" cy="70" r="7" fill="#EAF5EE" stroke="#2D8B55"/><text x="430" y="58" text-anchor="middle" fill="#276b45" font-weight="bold">+1</text><text class="lang-en" x="430" y="150" text-anchor="middle" fill="#276b45">push right</text><text class="lang-zh" x="430" y="150" text-anchor="middle" fill="#276b45">压右边</text></g></svg>
%%%

Put numbers on the plank: lowest is `−1`, highest is `+1`, resting is exactly `0`. That balanced swing is [[tanh||Squashes any number into (−1, 1), and passes through the origin: tanh(0)=0. It is zero-centered, so its outputs are balanced above and below zero.]] — and because its outputs sit evenly above and below zero, we call it **zero-centered**. That is the cure for the stuck wheel you just met.

**What the see-saw gets right:** the balanced tilt around a level middle. **Where it breaks down:** a real see-saw keeps tilting the harder you push, but tanh flattens near both ends — the same flattening as sigmoid.
~~~zh
在板子上标上数字：最低是 `−1`，最高是 `+1`，静止时正好是 `0`。这个平衡的摆动就是
[[tanh||把任何数字挤进 (−1, 1)，而且经过原点：tanh(0)=0。它以零为中心，所以输出在零的上下是平衡的。]] —— 因为它的输出均匀地分布在零的上面和下面，我们说它是**以零为中心的**
（zero-centered）。这正是你刚刚见到的那个卡住的轮子的解药。

**跷跷板这个比喻对在哪里：** 围绕一个水平中点的平衡倾斜。**它在哪里不成立：** 真的跷跷板你越用力
压它就越倾斜，而 tanh 在两端附近会变平 —— 和 sigmoid 一样的变平。
~~~

%%% demo id=tanh label="predict, then run"
predict: tanh is the see-saw. What does it give for −2, 0 and 2? The middle one is the giveaway — what should a see-saw with nobody on it read?
code: tanh(np.array([-2, 0, 2]))
out: array([-0.964,  0.   ,  0.964])
take: <b>tanh squashes into (−1, 1) and rests at 0.</b> Input 0 leaves it level, a big negative tips it toward −1, a big positive toward +1 — balanced above and below zero. That's "zero-centered," so the cart rolls straight.
%%%
~~~zh
%%% demo id=tanhzh label="先猜，再展开"
predict: tanh 就是那个跷跷板。它对 −2、0、2 给出什么？中间那个是关键 —— 一块没人坐的跷跷板应该读出多少？
code: tanh(np.array([-2, 0, 2]))
out: array([-0.964,  0.   ,  0.964])
take: <b>tanh 把数字挤进 (−1, 1)，静止在 0。</b>输入 0 让它保持水平，一个大负数把它压向 −1，一个大正数压向 +1 —— 在零的上下是平衡的。这就是「以零为中心」，所以购物车能走直线。
%%%
~~~

#### One word to carry forward: slope
tanh and sigmoid are close cousins — same smooth S, different plank height and resting point. Sigmoid runs `0 → 1` and rests at `0.5`; tanh runs `−1 → +1` and rests at `0`. To see *why* that difference matters, the rest of the day turns on a single word, so let's make it concrete. The **slope** of a bend at a point is *how steep the curve is right there* — how much the output moves when you nudge the input a tiny bit.
~~~zh
#### 要带走的一个词：斜率
tanh 和 sigmoid 是近亲 —— 同样平滑的 S 形，只是板子的高度和静止点不同。Sigmoid 从 `0 → 1`，静止在
`0.5`；tanh 从 `−1 → +1`，静止在 `0`。要看清这个差别*为什么*重要，今天剩下的部分都围绕一个词转，
所以我们把它讲具体。一个弯在某一点的**斜率（slope）**就是*那里的曲线有多陡* —— 你把输入轻轻推
一点点，输出会动多少。
~~~

%%% steps
step: **Steep slope** → the neuron reacts strongly to small changes
why: lots of tilt means a clear "which way to nudge" signal, so learning moves fast
step: **Slope near 0 (flat)** → the neuron barely reacts at all
why: no tilt, no direction — learning there crawls to a stop, exactly the step switch's problem
step: **So each bend has a slope story, not just a shape**
why: that's why the picture below draws the curves on top and their slopes underneath
%%%
~~~zh
%%% steps
step: **斜率大（陡）** → 神经元对小变化反应强烈
why: 倾斜大意味着「往哪边推」的信号很清楚，所以学得快
step: **斜率接近 0（平）** → 神经元几乎完全不反应
why: 没有倾斜就没有方向 —— 学习在那里慢到停下，正是 step 开关的那个问题
step: **所以每个弯都有一个斜率的故事，不只是一个形状**
why: 这就是为什么下面那张图上半部分画曲线，下半部分画它们的斜率
%%%
~~~

The bottom half of this picture is the one to look at slowly — it's the whole next unit in advance:
~~~zh
这张图的下半部分值得你慢慢看 —— 它提前把下一个单元整个讲完了：
~~~

%%% svg
<svg viewBox="0 0 520 380" role="img" aria-label="Top panel: ReLU, sigmoid and tanh drawn on shared labelled axes with gridlines at minus one, zero and plus one. Sigmoid runs from zero up through one half at z equals zero to one; tanh runs from minus one through zero up to plus one; ReLU is flat at zero for negatives then rises straight past one. Bottom panel: their slopes, with sigmoid's slope peaking at about 0.25 and tanh's at about 1.0, both at z equals zero, and ReLU's slope a flat 0 then a flat 1."><g font-family="monospace" font-size="12">
<text class="lang-en" x="60" y="22" font-size="13" fill="#3A342E" font-weight="bold">The bend  f(z)</text><text class="lang-zh" x="60" y="22" font-size="13" fill="#3A342E" font-weight="bold">这个弯  f(z)</text>
<rect x="292" y="13" width="10" height="10" fill="#2D8B55"/><text x="307" y="22" fill="#2D8B55" font-size="11" text-anchor="start">ReLU</text>
<rect x="352" y="13" width="10" height="10" fill="#7C6DAA"/><text x="367" y="22" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid</text>
<rect x="432" y="13" width="10" height="10" fill="#9A5A12"/><text x="447" y="22" fill="#9A5A12" font-size="11" text-anchor="start">tanh</text>
<line x1="60" y1="65" x2="470" y2="65" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<line x1="60" y1="110" x2="470" y2="110" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="60" y1="155" x2="470" y2="155" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="69" fill="#9A938A" font-size="10" text-anchor="end">+1</text>
<text x="56" y="114" fill="#9A938A" font-size="10" text-anchor="end">0</text>
<text x="56" y="159" fill="#9A938A" font-size="10" text-anchor="end">−1</text>
<line x1="265" y1="36" x2="265" y2="162" stroke="#E5DFD6" stroke-width="1"/>
<text x="477" y="114" fill="#9A938A" font-size="10" text-anchor="start">z</text>
<path d="M70 156 C 190 154, 228 113, 265 110 C 300 107, 330 66, 460 64" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<path d="M70 108 C 195 107, 238 90, 265 87.5 C 292 85, 350 68, 460 66" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<path d="M70 110 L265 110 L440 46" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<circle cx="265" cy="87.5" r="3.5" fill="#C93B3B"/>
<text class="lang-en" x="256" y="80" fill="#C93B3B" font-size="10" text-anchor="end">sigmoid(0)=0.5</text><text class="lang-zh" x="256" y="80" fill="#C93B3B" font-size="10" text-anchor="end">sigmoid(0)=0.5</text>
<text class="lang-en" x="60" y="212" font-size="13" fill="#3A342E" font-weight="bold">The slope  f′(z)</text><text class="lang-zh" x="60" y="212" font-size="13" fill="#3A342E" font-weight="bold">斜率  f′(z)</text>
<line x1="60" y1="340" x2="470" y2="340" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="265" y1="230" x2="265" y2="352" stroke="#E5DFD6" stroke-width="1"/>
<text x="477" y="344" fill="#9A938A" font-size="10" text-anchor="start">z</text>
<line x1="60" y1="248" x2="470" y2="248" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="252" fill="#9A5A12" font-size="10" text-anchor="end">1.0</text>
<line x1="60" y1="317" x2="470" y2="317" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="321" fill="#7C6DAA" font-size="10" text-anchor="end">0.25</text>
<path d="M70 340 L265 340 L265 248 L460 248" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<path d="M70 339 C 200 338, 235 252, 265 248 C 295 252, 330 338, 460 339" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<path d="M70 340 C 205 339, 240 319, 265 317 C 290 319, 325 339, 460 340" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<text class="lang-en" x="272" y="244" fill="#9A5A12" font-size="11" text-anchor="start">tanh′ peaks ≈ 1.0 at z=0</text><text class="lang-zh" x="272" y="244" fill="#9A5A12" font-size="11" text-anchor="start">tanh′ 在 z=0 处最高 ≈ 1.0</text>
<text class="lang-en" x="272" y="313" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid′ peaks ≈ 0.25 at z=0</text><text class="lang-zh" x="272" y="313" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid′ 在 z=0 处最高 ≈ 0.25</text>
<text class="lang-en" x="450" y="240" fill="#2D8B55" text-anchor="end" font-weight="bold">ReLU′ = 1 (z&gt;0)</text><text class="lang-zh" x="450" y="240" fill="#2D8B55" text-anchor="end" font-weight="bold">ReLU′ = 1（z&gt;0）</text>
<text class="lang-en" x="130" y="333" fill="#2D8B55" font-size="10" text-anchor="middle">ReLU′ = 0 (z&lt;0)</text><text class="lang-zh" x="130" y="333" fill="#2D8B55" font-size="10" text-anchor="middle">ReLU′ = 0（z&lt;0）</text>
</g></svg>
%%%

Three numbers off that bottom panel, and they explain the next hour of your life:

- **sigmoid** — slope never gets above about `0.25`
- **tanh** — slope reaches about `1.0`
- **ReLU** — slope is either a flat `0` (negatives) or a plain `1` (positives)

#### Now go bend them yourself
Enough reading — this one is yours to play with. Pick an activation, drag the marker along `z`, and watch the slope reading change. Try to **call the number before you let go**: drag into either tail of sigmoid or tanh, or into ReLU's flat left side, and see how close to `0` you can drive it.
~~~zh
从那个下半部分读出三个数字，它们能解释你接下来一个小时的人生：

- **sigmoid** —— 斜率永远不超过大约 `0.25`
- **tanh** —— 斜率能到大约 `1.0`
- **ReLU** —— 斜率要么是平的 `0`（负数），要么就是干脆的 `1`（正数）

#### 现在轮到你亲手去弯它们
读够了 —— 这个是给你玩的。挑一个激活函数，把标记沿着 `z` 拖动，看斜率的读数怎么变。试着
**在松手之前先说出那个数字**：拖进 sigmoid 或 tanh 任意一侧的尾巴，或者拖进 ReLU 左边那段平的
地方，看你能把它逼到多接近 `0`。
~~~

%%% viz src=../../viz/activation-derivatives.html title="activations and their slopes" caption="interactive — pick an activation, drag z; watch the slope flatten"
%%%

%%% insight
That flattening you just felt with your finger is the split that organises every activation ever invented. **Bounded and saturating** (sigmoid, tanh): they squash into a fixed range and go flat at the ends. **Unbounded and non-saturating** (the ReLU family): no ceiling, never flat on the positive side. Almost every choice you'll make comes down to which side of that line you want.
%%%
~~~zh
%%% insight
你刚刚用手指感受到的那个变平，就是把所有被发明出来的激活函数分成两边的那道分界线。**有界且会饱和**
（sigmoid、tanh）：它们把数字挤进一个固定范围，并在两端变平。**无界且不饱和**（ReLU 家族）：没有
上限，在正的那一侧永远不平。你以后做的几乎每一个选择，都归结为你想要这条线的哪一边。
%%%
~~~

(The exact recipe for a slope is called the *derivative*, and you'll do that math on Day 5. Today the picture is genuinely enough.)
~~~zh
（求斜率的确切配方叫做 *derivative（导数）*，那道数学你会在第 5 天做。今天这张图真的就够了。）
~~~

#### Victory lap — you've met the whole family
Breathe. You now know the four bends every deep-learning engineer can name — three you'll actually use, and one you'll know why nobody uses:

- **step** — the crude switch: flat, can't learn (the museum piece)
- **ReLU** — the valve: cheap, keeps a tilt for positives
- **sigmoid** — the dimmer: smooth `0 → 1`, one-sided
- **tanh** — the see-saw: smooth `−1 → +1`, balanced at `0`

One honest footnote so you're never caught out: **ReLU is one-sided too** — its outputs are all `≥ 0`, so it has the same drift as sigmoid. The field accepts that trade because ReLU's slope of `1` pays for it, and normalization layers (a later day) clean up the rest.

**So far:** four bends, and one number each — their slope. Next we find out what happens when that number goes flat. Nothing to memorise; a side-by-side table is waiting for you in the recap.
~~~zh
#### 庆功圈 —— 你已经认识了一整家人
喘口气。你现在知道每个深度学习工程师都能说出名字的那四个弯 —— 三个你真的会用，一个你会知道为什么
没人用：

- **step** —— 粗糙的开关：平的，学不了东西（博物馆展品）
- **ReLU** —— 阀门：便宜，对正数保留倾斜
- **sigmoid** —— 调光旋钮：平滑的 `0 → 1`，单侧
- **tanh** —— 跷跷板：平滑的 `−1 → +1`，在 `0` 处平衡

一个诚实的脚注，免得你以后被问住：**ReLU 也是单侧的** —— 它的输出全都 `≥ 0`，所以它有和 sigmoid
一样的漂移。这个领域接受这个交换，因为 ReLU 的斜率 `1` 值这个价，而 normalization 层（后面某一天）
会把剩下的问题收拾干净。

**到这里为止：** 四个弯，每个配一个数字 —— 它们的斜率。接下来我们看看那个数字变平时会发生什么。
没什么要背的；recap 里有一张并排的表在等你。
~~~

@@@ concept id=c7 tag="The whisper" zh_tag="传话" title="The whisper down the line" zh_title="一句话传下去，越传越小" gotit="Got the whisper" zh_gotit="懂了这个传话"
Here's the first real mystery of the day, and it's a lovely one: a perfectly good smooth bend can quietly *stop learning*.

Picture a line of friends passing a **whisper** from the back of a room to the front. Each person hears a bit less than the one before. By the fourth person it's a mumble; by the tenth, silence.
~~~zh
今天第一个真正的谜题来了，而且是个漂亮的谜题：一个好端端的平滑的弯，可以悄悄地*停止学习*。

想象一排朋友从房间后面往前**传一句耳语**。每个人听到的都比前一个人少一点。传到第四个人已经是含糊
的咕哝；传到第十个，就是一片安静。
~~~

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="Five people standing in a line passing a whisper forward. The first speaks at full strength 1.0, the next hears about 0.25, the next about 0.06, the next about 0.016, and the last hears only about 0.004 — a faint mumble. Labels under each person count how many layers back they are, from zero back to four back."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A whisper down a line — the learning signal fades</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一句话往下传 —— 学习信号在衰减</text>
<circle cx="58" cy="104" r="19" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="2"/><text x="58" y="110" text-anchor="middle" font-size="16">🗣️</text><rect x="34" y="58" width="48" height="22" rx="7" fill="#EFEAF7" stroke="#7C6DAA"/><text x="58" y="74" text-anchor="middle" fill="#5E5191">1.0</text>
<text x="100" y="110" fill="#B8AEA2">→</text>
<circle cx="155" cy="104" r="19" fill="#EFEAF7" stroke="#9A93C0"/><text x="155" y="110" text-anchor="middle" font-size="14">👂</text><rect x="132" y="60" width="46" height="20" rx="6" fill="#EFEAF7" stroke="#9A93C0"/><text x="155" y="74" text-anchor="middle" fill="#7C6DAA" font-size="10">≈0.25</text>
<text x="197" y="110" fill="#B8AEA2">→</text>
<circle cx="253" cy="104" r="19" fill="#F4F1FA" stroke="#C7BEDF"/><text x="253" y="110" text-anchor="middle" font-size="14">👂</text><rect x="231" y="62" width="44" height="18" rx="6" fill="#F4F1FA" stroke="#C7BEDF"/><text x="253" y="75" text-anchor="middle" fill="#9A93C0" font-size="9">≈0.06</text>
<text x="295" y="110" fill="#B8AEA2">→</text>
<circle cx="351" cy="104" r="19" fill="#F8F6FC" stroke="#D9D3EC"/><text x="351" y="110" text-anchor="middle" font-size="14">👂</text><rect x="328" y="63" width="48" height="17" rx="6" fill="#F8F6FC" stroke="#D9D3EC"/><text x="352" y="76" text-anchor="middle" fill="#9A93C0" font-size="9">≈0.016</text>
<text x="393" y="110" fill="#B8AEA2">→</text>
<circle cx="449" cy="104" r="19" fill="#FDECEC" stroke="#C93B3B"/><text x="449" y="110" text-anchor="middle" font-size="14">😕</text><text x="449" y="76" text-anchor="middle" fill="#C93B3B" font-size="9">≈0.004</text>
<text class="lang-en" x="58" y="138" text-anchor="middle" fill="#9A938A" font-size="9">0 back</text><text class="lang-zh" x="58" y="138" text-anchor="middle" fill="#9A938A" font-size="9">往回 0 层</text><text class="lang-en" x="155" y="138" text-anchor="middle" fill="#9A938A" font-size="9">1 back</text><text class="lang-zh" x="155" y="138" text-anchor="middle" fill="#9A938A" font-size="9">往回 1 层</text><text class="lang-en" x="253" y="138" text-anchor="middle" fill="#9A938A" font-size="9">2 back</text><text class="lang-zh" x="253" y="138" text-anchor="middle" fill="#9A938A" font-size="9">往回 2 层</text><text class="lang-en" x="351" y="138" text-anchor="middle" fill="#9A938A" font-size="9">3 back</text><text class="lang-zh" x="351" y="138" text-anchor="middle" fill="#9A938A" font-size="9">往回 3 层</text><text class="lang-en" x="449" y="138" text-anchor="middle" fill="#C93B3B" font-size="9">4 back</text><text class="lang-zh" x="449" y="138" text-anchor="middle" fill="#C93B3B" font-size="9">往回 4 层</text>
<text class="lang-en" x="260" y="166" text-anchor="middle" fill="#C93B3B" font-size="10">×0.25 at every hand-off → four layers back it is ≈0.004 → the vanishing gradient</text><text class="lang-zh" x="260" y="166" text-anchor="middle" fill="#C93B3B" font-size="10">每次交接 ×0.25 → 往回四层就只剩 ≈0.004 → 梯度消失</text></g></svg>
%%%

That whisper is the **learning signal** travelling backward through the layers, telling each early layer how to improve. Each layer it passes gets multiplied by that layer's slope — so a small slope means a quieter whisper. ("0 layers back" is where the signal starts, at the end of the network.)

**What the whisper gets right:** each hand-off loses a fixed fraction, so the loss compounds. **Where it breaks down:** real whispers get *garbled*; here the message stays correct, it just gets too faint to act on.
~~~zh
那句耳语就是**学习信号**，它往回穿过各层，告诉每一个靠前的层该怎么改进。它每经过一层就被乘上那层
的斜率 —— 所以斜率小就意味着耳语更轻。（「往回 0 层」是信号出发的地方，在网络的末端。）

**耳语这个比喻对在哪里：** 每一次交接都损失固定的一部分，所以损失会累积。**它在哪里不成立：** 真的
耳语会*传错*；这里消息始终是对的，只是变得太轻，没法照着做。
~~~

#### First, why one neuron goes quiet
You already felt this with your own finger a minute ago. When you dragged `z` far out into sigmoid's tail, the slope reading fell toward `0` — the output was pinned near the top and stopped reacting. That is [[saturation||When an activation's input is far from zero, its output is pinned near a limit and its slope is ≈ 0. A saturated neuron barely responds to changes in its input.]], and it has an easy cure: keep `z` small so units stay in the steep middle, or use a bend that never flattens.

A saturated neuron has a slope of about `0`, and you already know what that means: no tilt, no nudge, no learning. One sleepy neuron. Annoying, but survivable.
~~~zh
#### 先说，为什么单个神经元会安静下来
一分钟前你已经用自己的手指感受过这个了。当你把 `z` 拖到 sigmoid 尾巴的远处时，斜率的读数掉向 `0` ——
输出被顶在了上限附近，不再反应。这就是 [[saturation||当一个激活函数的输入离零很远时，它的输出被顶在某个极限附近，斜率 ≈ 0。一个饱和的神经元几乎不响应输入的变化。]]（饱和），
它有一个很简单的解药：让 `z` 保持小，这样单元待在陡的中段；或者用一个永远不变平的弯。

一个饱和的神经元斜率大约是 `0`，而你已经知道那意味着什么：没有倾斜，没有推力，没有学习。一个睡着
的神经元。烦人，但还活得下去。
~~~

%%% insight
Here's why it bites much harder than "one sleepy neuron." Nothing in your code errors. Nothing prints a warning. The loss just settles a bit higher than it should and sits there — for hours. This failure's whole trick is looking like a bad idea rather than a broken network.
%%%
~~~zh
%%% insight
这就是它为什么比「一个睡着的神经元」咬人得多。你的代码里什么都不报错。什么警告都不打印。loss 只是
停在比它该停的地方稍高一点，然后就一直坐在那里 —— 坐好几个小时。这个失败的全部伎俩，就是让它看起来
像是一个糟糕的想法，而不是一个坏掉的网络。
%%%
~~~

#### Then, why depth makes it brutal
Back to the whisper line — and now put sigmoid's numbers on it. Its steepest slope is only about `0.25`, so at *best* each layer passes back a quarter of what it heard. **Guess before you look:** four layers back, is the whisper down to a half? A tenth? Hold a number in your head, then read the bars.

Each bar below is how loud the learning signal still is, that many layers back. Both rows are drawn to the same height scale, so you can compare them straight across — and no arithmetic is needed, the picture does it for you:
~~~zh
#### 再说，为什么深度让它变得残酷
回到那排传话的人 —— 现在把 sigmoid 的数字放上去。它最陡的斜率也只有大约 `0.25`，所以*最好的情况下*
每层往回传的也只是它听到的四分之一。**先猜再看：** 往回四层，那句耳语还剩一半？还剩十分之一？在
心里定一个数字，然后再读那些条形。

下面每一根条形代表在往回那么多层的地方，学习信号还有多大声。两行用的是同一个高度比例，所以你可以
直接横着比 —— 而且不需要算术，图会替你算：
~~~

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="Two rows of bars drawn to the same height scale. Top row, sigmoid: the signal is a full bar at 1.0 zero layers back, then about 0.25, about 0.06, about 0.016 and about 0.004 — shrinking to almost nothing. Bottom row, ReLU: every bar is full height at 1.0, from zero layers back to four layers back."><g font-family="monospace" font-size="11" text-anchor="middle">
<text class="lang-en" x="52" y="18" fill="#5E5191" font-weight="bold" text-anchor="start">sigmoid · ×0.25 per hand-off → the signal fades</text><text class="lang-zh" x="52" y="18" fill="#5E5191" font-weight="bold" text-anchor="start">sigmoid · 每次交接 ×0.25 → 信号衰减</text>
<text class="lang-en" x="470" y="18" fill="#9A938A" font-size="9" text-anchor="end">same scale in both rows: 1.0 = 60 px</text><text class="lang-zh" x="470" y="18" fill="#9A938A" font-size="9" text-anchor="end">两行同一比例：1.0 = 60 像素</text>
<line x1="52" y1="92" x2="440" y2="92" stroke="#E5DFD6" stroke-width="1.5"/>
<rect x="60"  y="32" width="46" height="60" fill="#7C6DAA"/><text x="83"  y="106" fill="#6B645E">1.0</text><text class="lang-en" x="83"  y="120" fill="#9A938A" font-size="9">0 back</text><text class="lang-zh" x="83"  y="120" fill="#9A938A" font-size="9">往回 0 层</text>
<rect x="140" y="77" width="46" height="15" fill="#7C6DAA"/><text x="163" y="106" fill="#6B645E">≈0.25</text><text class="lang-en" x="163" y="120" fill="#9A938A" font-size="9">1 back</text><text class="lang-zh" x="163" y="120" fill="#9A938A" font-size="9">往回 1 层</text>
<rect x="220" y="88" width="46" height="4"  fill="#7C6DAA"/><text x="243" y="106" fill="#6B645E">≈0.06</text><text class="lang-en" x="243" y="120" fill="#9A938A" font-size="9">2 back</text><text class="lang-zh" x="243" y="120" fill="#9A938A" font-size="9">往回 2 层</text>
<rect x="300" y="91" width="46" height="1"  fill="#9A93C0"/><text x="323" y="106" fill="#6B645E">≈0.016</text><text class="lang-en" x="323" y="120" fill="#9A938A" font-size="9">3 back</text><text class="lang-zh" x="323" y="120" fill="#9A938A" font-size="9">往回 3 层</text>
<rect x="380" y="91" width="46" height="1"  fill="#D9D3EC"/><text x="403" y="106" fill="#C93B3B">≈0.004</text><text class="lang-en" x="403" y="120" fill="#C93B3B" font-size="9">4 back</text><text class="lang-zh" x="403" y="120" fill="#C93B3B" font-size="9">往回 4 层</text>
<text class="lang-en" x="52" y="152" fill="#2D8B55" font-weight="bold" text-anchor="start">ReLU · ×1 per hand-off → the signal survives</text><text class="lang-zh" x="52" y="152" fill="#2D8B55" font-weight="bold" text-anchor="start">ReLU · 每次交接 ×1 → 信号活了下来</text>
<line x1="52" y1="224" x2="440" y2="224" stroke="#E5DFD6" stroke-width="1.5"/>
<rect x="60"  y="164" width="46" height="60" fill="#2D8B55"/><text x="83"  y="238" fill="#6B645E">1.0</text>
<rect x="140" y="164" width="46" height="60" fill="#2D8B55"/><text x="163" y="238" fill="#6B645E">1.0</text>
<rect x="220" y="164" width="46" height="60" fill="#2D8B55"/><text x="243" y="238" fill="#6B645E">1.0</text>
<rect x="300" y="164" width="46" height="60" fill="#2D8B55"/><text x="323" y="238" fill="#6B645E">1.0</text>
<rect x="380" y="164" width="46" height="60" fill="#2D8B55"/><text x="403" y="238" fill="#6B645E">1.0</text>
<text class="lang-en" x="470" y="238" fill="#9A938A" font-size="9" text-anchor="end">same 0…4 layers back</text><text class="lang-zh" x="470" y="238" fill="#9A938A" font-size="9" text-anchor="end">同样是往回 0…4 层</text>
</g></svg>
%%%

If the top row surprised you, good — almost everyone guesses "about a half." Three short beats and it's yours:
~~~zh
如果上面那一行让你意外，很好 —— 几乎所有人都猜「大概一半」。三个短拍子，它就归你了：
~~~

%%% steps
step: **The fade.** Every hand-off multiplies the whisper by sigmoid's best slope, `0.25`
why: four hand-offs later it is a rounding error, `≈ 0.004` — that top row of bars
step: **The name.** Early layers hear only a mumble, so they barely change: the [[vanishing gradient||A learning signal (gradient) that shrinks toward zero as it is multiplied by each layer's small slope on the way back through a deep network — so the earliest layers stop learning. Saturating activations like sigmoid/tanh cause it; ReLU largely avoids it.]]
why: therefore a deep sigmoid net trains its last layers and quietly abandons its first ones
step: **The rescue.** Swap in ReLU, whose positive-side slope is exactly `1`
why: multiplying by 1 changes nothing, so the whisper arrives at full volume — that bottom row of bars
%%%
~~~zh
%%% steps
step: **衰减。** 每一次交接都把那句耳语乘上 sigmoid 最好的斜率 `0.25`
why: 四次交接之后它就是一个舍入误差了，`≈ 0.004` —— 就是上面那一行条形
step: **名字。** 靠前的层只听到咕哝，所以几乎不改变：这就是 [[vanishing gradient||学习信号（梯度）在往回穿过深层网络时，被每一层小小的斜率乘来乘去，越来越接近零 —— 于是最靠前的层停止学习。sigmoid/tanh 这类会饱和的激活函数会引起它；ReLU 基本上避开了它。]]（梯度消失）
why: 因此一个很深的 sigmoid 网络会训练它最后几层，然后悄悄放弃最前面那几层
step: **救援。** 换上 ReLU，它在正的那一侧斜率正好是 `1`
why: 乘 1 什么都不改变，所以那句耳语以全音量到达 —— 就是下面那一行条形
%%%
~~~

Struggling to hold it all? That's the normal reaction, not a bad sign — the ladder here is one idea repeated, not four new ones. If you'd like a nudge, open the hints below one at a time.
~~~zh
觉得抓不住全部？那是正常反应，不是坏兆头 —— 这里的阶梯是同一个想法重复了几遍，不是四个新想法。
如果你想要一点提示，下面的提示一条一条打开。
~~~

%%% hint
t1: Don't try to hold four layers in your head at once. Look at ONE hand-off: a person hears "1.0" and passes on "0.25." What single number did they multiply by?
t2: They multiplied by 0.25 — sigmoid's steepest slope. Doing that four times shrinks 1.0 to about 0.004, which is why the fourth bar is invisible.
t3: Each layer multiplies the learning signal by a small slope (≈0.25 for sigmoid), so many layers shrink it toward zero and the earliest layers stop learning. That fade is the vanishing gradient, and ReLU's slope of 1 is what stops it.
%%%
~~~zh
%%% hint
t1: 不要试着一次把四层都装进脑子里。只看一次交接：一个人听到「1.0」，传出去「0.25」。他乘了哪一个数字？
t2: 他乘了 0.25 —— sigmoid 最陡的斜率。做四次，1.0 就缩到大约 0.004，这就是第四根条形看不见的原因。
t3: 每一层都把学习信号乘上一个小小的斜率（sigmoid 约 0.25），所以层数一多就把它压向零，最前面的层就停止学习了。这个衰减就是 vanishing gradient，而 ReLU 的斜率 1 就是拦住它的东西。
%%%
~~~

That single swap — saturating sigmoids out, ReLU in — is what let people finally train genuinely deep networks. **You can now explain in one sentence why sigmoid stalled deep nets and ReLU rescued them.** That is a real interview answer, and it's yours.
~~~zh
就是这一次替换 —— 把会饱和的 sigmoid 换掉，换上 ReLU —— 让人们终于能训练真正深的网络。
**你现在能用一句话解释为什么 sigmoid 卡住了深层网络，而 ReLU 救了它们。** 这是一个真正的面试答案，
而它属于你了。
~~~

!!! c-ok 🧰
<b>Three more cures, so you know the names.</b> Keep <code>z</code> small so units stay in the steep middle. Prefer <code>tanh</code> over <code>sigmoid</code> when you need an S-curve (slope up to 1.0, not 0.25). And two standard tools you'll meet properly later: <b>He</b> / <b>Xavier initialization</b> (recipes for sensible starting weights) and <b>normalization layers</b> (they re-centre each layer's numbers as training runs).
!!!
~~~zh
!!! c-ok 🧰
<b>另外三个解药，让你知道它们的名字。</b>让 <code>z</code> 保持小，这样单元待在陡的中段。需要一条
S 形曲线时，优先用 <code>tanh</code> 而不是 <code>sigmoid</code>（斜率最高到 1.0，不是 0.25）。还有
两个标准工具，你以后会正式见到：<b>He</b> / <b>Xavier initialization</b>（挑一组合理起始 weight 的
配方），以及 <b>normalization 层</b>（在训练过程中把每一层的数字重新居中）。
!!!
~~~

**So far:** flat tails + depth = a whisper that dies. ReLU's tilt of 1 is the fix. (The backward-pass math that actually multiplies these slopes is Day 5's job.)
~~~zh
**到这里为止：** 平坦的尾巴 + 深度 = 一句会死掉的耳语。ReLU 那个 1 的倾斜就是解决办法。（真正把这些
斜率乘起来的反向传播数学，是第 5 天的活。）
~~~

@@@ concept id=c8 tag="Which bend?" zh_tag="用哪个弯？" title="Inside the factory vs. the shipping door" zh_title="工厂内部 vs. 出货口" gotit="Got the two decisions" zh_gotit="懂了这两个决定"
Time to cash in what you've learned — this is the fun part, and it's a break from hunting failures. You've met the whole family; now you get to *choose*.

Picture a **factory**. Inside, along the conveyor belt, every station uses the same cheap stamping tool over and over, because cheap and reliable is exactly what you want a hundred times a second. Then, at the **shipping door**, you pick a box that fits the thing you're actually sending: a bottle needs a bottle-shaped box, a book needs a flat one.
~~~zh
是时候把学到的东西兑现了 —— 这是好玩的部分，也是从「找失败」里歇一口气。你已经认识了一整家人；
现在轮到你来*挑*。

想象一座**工厂**。里面，沿着传送带，每一个工位都反复使用同一个便宜的冲压工具，因为一秒钟要做一百次
的时候，便宜又可靠正是你想要的。然后，在**出货口**，你按照真正要寄的东西挑一个合适的箱子：瓶子要
瓶子形的箱子，书要一个扁的。
~~~

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A factory picture in two halves. On the left, a conveyor belt with three identical ReLU stamping stations above it, labelled as the hidden layers where you pick one cheap bend and repeat it. On the right, a shipping door holding three different shaped boxes labelled yes-no, which-one, and number, labelled as the output layer where you match the box to the answer's shape."><g font-family="monospace" font-size="10">
<text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Inside the factory: the same cheap tool. At the shipping door: the right box.</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">工厂内部：同一个便宜工具。出货口：对的那个箱子。</text>
<rect x="28" y="96" width="270" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/>
<circle cx="48" cy="109" r="7" fill="#FDF9F3" stroke="#7C6DAA"/><circle cx="163" cy="109" r="7" fill="#FDF9F3" stroke="#7C6DAA"/><circle cx="278" cy="109" r="7" fill="#FDF9F3" stroke="#7C6DAA"/>
<rect x="60" y="62" width="52" height="26" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="86" y="79" text-anchor="middle" fill="#1a5c38">ReLU</text>
<rect x="137" y="62" width="52" height="26" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="163" y="79" text-anchor="middle" fill="#1a5c38">ReLU</text>
<rect x="214" y="62" width="52" height="26" rx="5" fill="#E8F5EE" stroke="#2D8B55" stroke-width="2"/><text x="240" y="79" text-anchor="middle" fill="#1a5c38">ReLU</text>
<text class="lang-en" x="163" y="142" text-anchor="middle" fill="#5E5191">the hidden layers — one cheap bend, repeated</text><text class="lang-zh" x="163" y="142" text-anchor="middle" fill="#5E5191">hidden 层 —— 一个便宜的弯，重复用</text>
<text class="lang-en" x="163" y="158" text-anchor="middle" fill="#9A938A">decision 1: pick ONE default and move on</text><text class="lang-zh" x="163" y="158" text-anchor="middle" fill="#9A938A">决定 1：挑一个默认值，然后往下走</text>
<text x="313" y="114" fill="#B8AEA2" font-size="16">→</text>
<rect x="340" y="50" width="150" height="78" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/>
<text class="lang-en" x="415" y="66" text-anchor="middle" fill="#8A6D3B">🚪 shipping door</text><text class="lang-zh" x="415" y="66" text-anchor="middle" fill="#8A6D3B">🚪 出货口</text>
<rect x="348" y="74" width="44" height="20" rx="3" fill="#FFFDF6" stroke="#9A7A10"/><text class="lang-en" x="370" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">yes/no</text><text class="lang-zh" x="370" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">是/否</text>
<rect x="396" y="74" width="44" height="20" rx="3" fill="#FFFDF6" stroke="#9A7A10"/><text class="lang-en" x="418" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">which?</text><text class="lang-zh" x="418" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">哪一个？</text>
<rect x="444" y="74" width="42" height="20" rx="3" fill="#FFFDF6" stroke="#9A7A10"/><text class="lang-en" x="465" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">number</text><text class="lang-zh" x="465" y="88" text-anchor="middle" fill="#8A6D3B" font-size="9">一个数</text>
<text class="lang-en" x="415" y="110" text-anchor="middle" fill="#8A6D3B" font-size="9">a different box for a different job</text><text class="lang-zh" x="415" y="110" text-anchor="middle" fill="#8A6D3B" font-size="9">不同的活，配不同的箱子</text>
<text class="lang-en" x="415" y="142" text-anchor="middle" fill="#8A6D3B">the output layer</text><text class="lang-zh" x="415" y="142" text-anchor="middle" fill="#8A6D3B">output 层</text>
<text class="lang-en" x="415" y="158" text-anchor="middle" fill="#9A938A">decision 2: match the answer's shape</text><text class="lang-zh" x="415" y="158" text-anchor="middle" fill="#9A938A">决定 2：对上答案的形状</text>
</g></svg>
%%%

That's the one thing to carry out of today about *choosing*: a network makes **two separate activation decisions**, and people mix them up constantly. Inside the [[hidden layers||Every layer between the input and the final output layer. Their job is to build useful internal features, not to produce the answer.]] you want a cheap, reliable bend. At the [[output layer||The last layer, whose numbers are the model's actual answer. Its activation has to match the SHAPE of the answer you want.]] you want the answer's shape.

**What the factory gets right:** one tool inside, a fitted box at the door. **Where it breaks down:** a factory's stations are physically different machines, while every hidden layer here runs the exact same little bend — the sameness is the point.
~~~zh
这就是今天关于*怎么挑*要带走的那一件事：一个网络会做**两个分开的激活函数决定**，而人们总是把它们
搞混。在 [[hidden layers||输入层和最后的 output 层之间的每一层。它们的活是造出有用的内部特征，不是给出答案。]]（隐藏层）里面，你想要一个便宜、可靠的弯。在
[[output layer||最后一层，它的数字就是模型真正的答案。它的激活函数必须对上你想要的答案的形状。]]（输出层），你想要的是答案的形状。

**工厂这个比喻对在哪里：** 里面一个工具，门口一个量身的箱子。**它在哪里不成立：** 工厂的工位是
物理上不同的机器，而这里每一个 hidden 层跑的是完全同一个小小的弯 —— 这个「一样」才是重点。
~~~

You've met all three doors already without noticing. Your phone's face unlock ends in one **yes/no** — is this you? A feed that ranks videos ends in a **which-one-of-many**. A model that guesses a house price or tomorrow's temperature ends in **a plain number**. Same conveyor inside; three different doors.
~~~zh
这三个门你其实都已经见过了，只是没注意。你手机的人脸解锁最后是一个**是/否** —— 这是你吗？一个给
视频排序的信息流最后是一个**从很多里挑一个**。一个猜房价或者猜明天气温的模型最后是**一个普普通通
的数字**。里面是同一条传送带；门有三种。
~~~

#### Decision 1 is easy: take the default
**Use `ReLU` in the hidden layers unless you have a specific reason not to.** That's it. Reach for `GELU` in a transformer, `Leaky ReLU` if units go silent (that's the very next unit), `tanh` if you truly need a bounded, balanced signal. Otherwise: ReLU, move on, spend your thinking somewhere it pays.
~~~zh
#### 决定 1 很简单：用默认值
**hidden 层里就用 `ReLU`，除非你有一个具体的理由不用。** 就这样。在 transformer 里伸手拿 `GELU`；
单元开始沉默时拿 `Leaky ReLU`（就是下一个单元的事）；真的需要一个有界、平衡的信号时拿 `tanh`。
其他情况：ReLU，往下走，把脑力花在能回本的地方。
~~~

#### Decision 2 is the one people get wrong
Now the interesting mistake — and it's easy to make, because it feels like being consistent. Below are **two different jobs**, and each door fails a different one. Job A: say a house price of `350,000`. Job B: say a temperature change of `−20`. **Predict which door survives each job before you read the picture.**
~~~zh
#### 决定 2 才是人们会做错的那个
现在说那个有意思的错误 —— 而且它很容易犯，因为它感觉像是在「保持一致」。下面是**两件不同的活**，
每个门各败在其中一件上。活 A：说出一个房价 `350,000`。活 B：说出一个温度变化 `−20`。
**在读这张图之前，先猜哪个门能挺过哪件活。**
~~~

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Two different jobs tested on three output doors. Job A is to say the house price 350,000: with sigmoid at the output the answer is squashed to about 1.0, so it can never say 350,000, while with no activation at all the number passes straight through unchanged and arrives intact. Job B is to say a negative answer, minus 20 degrees: ReLU clips it to 0, because ReLU passes a large positive like 350,000 unchanged but can never output a negative number. Each door fails a different job."><g font-family="monospace" font-size="11">
<text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two jobs. Which door can even SAY the answer?</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">两件活。哪个门连答案都说不出来？</text>
<text class="lang-en" x="26" y="38" fill="#8A6D3B" font-size="10" text-anchor="start">JOB A · say the house price: 350,000</text><text class="lang-zh" x="26" y="38" fill="#8A6D3B" font-size="10" text-anchor="start">活 A · 说出房价：350,000</text>
<text x="122" y="63" text-anchor="end" fill="#5E5191">sigmoid</text>
<rect x="130" y="46" width="96" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text class="lang-en" x="178" y="63" text-anchor="middle" fill="#6B645E" font-size="10">350000 in</text><text class="lang-zh" x="178" y="63" text-anchor="middle" fill="#6B645E" font-size="10">进 350000</text>
<text x="234" y="63" fill="#B8AEA2">→</text>
<rect x="252" y="46" width="96" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="300" y="63" text-anchor="middle" fill="#C93B3B" font-size="10">≈ 1.0 out</text><text class="lang-zh" x="300" y="63" text-anchor="middle" fill="#C93B3B" font-size="10">出 ≈ 1.0</text>
<text class="lang-en" x="358" y="61" fill="#C93B3B" font-size="9" text-anchor="start">❌ squashed — can't even say 5</text><text class="lang-zh" x="358" y="61" fill="#C93B3B" font-size="9" text-anchor="start">❌ 被挤扁了 —— 连 5 都说不出</text>
<text class="lang-en" x="122" y="99" text-anchor="end" fill="#9A5A12">no bend (linear)</text><text class="lang-zh" x="122" y="99" text-anchor="end" fill="#9A5A12">不加弯（线性）</text>
<rect x="130" y="82" width="96" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text class="lang-en" x="178" y="99" text-anchor="middle" fill="#6B645E" font-size="10">350000 in</text><text class="lang-zh" x="178" y="99" text-anchor="middle" fill="#6B645E" font-size="10">进 350000</text>
<text x="234" y="99" fill="#B8AEA2">→</text>
<rect x="252" y="82" width="96" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text class="lang-en" x="300" y="99" text-anchor="middle" fill="#276b45" font-size="10">350000 out</text><text class="lang-zh" x="300" y="99" text-anchor="middle" fill="#276b45" font-size="10">出 350000</text>
<text class="lang-en" x="358" y="97" fill="#2D8B55" font-size="9" text-anchor="start">✅ arrives intact</text><text class="lang-zh" x="358" y="97" fill="#2D8B55" font-size="9" text-anchor="start">✅ 完好到达</text>
<line x1="26" y1="120" x2="494" y2="120" stroke="#E5DFD6" stroke-width="1"/>
<text class="lang-en" x="26" y="140" fill="#8A6D3B" font-size="10" text-anchor="start">JOB B · say a temperature change: −20</text><text class="lang-zh" x="26" y="140" fill="#8A6D3B" font-size="10" text-anchor="start">活 B · 说出温度变化：−20</text>
<text x="122" y="165" text-anchor="end" fill="#2D8B55">ReLU</text>
<rect x="130" y="148" width="96" height="24" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text class="lang-en" x="178" y="165" text-anchor="middle" fill="#6B645E" font-size="10">−20 in</text><text class="lang-zh" x="178" y="165" text-anchor="middle" fill="#6B645E" font-size="10">进 −20</text>
<text x="234" y="165" fill="#B8AEA2">→</text>
<rect x="252" y="148" width="96" height="24" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="300" y="165" text-anchor="middle" fill="#C93B3B" font-size="10">0 out</text><text class="lang-zh" x="300" y="165" text-anchor="middle" fill="#C93B3B" font-size="10">出 0</text>
<text class="lang-en" x="358" y="163" fill="#C93B3B" font-size="9" text-anchor="start">❌ clipped — no negatives</text><text class="lang-zh" x="358" y="163" fill="#C93B3B" font-size="9" text-anchor="start">❌ 被截断 —— 出不了负数</text>
<text class="lang-en" x="260" y="196" text-anchor="middle" fill="#6B645E" font-size="10">ReLU would pass 350,000 through fine — it just can never say −20</text><text class="lang-zh" x="260" y="196" text-anchor="middle" fill="#6B645E" font-size="10">ReLU 让 350,000 顺利通过 —— 它只是永远说不出 −20</text>
<text class="lang-en" x="260" y="212" text-anchor="middle" fill="#6B645E" font-size="10">so match the door to the SHAPE of the answer, job by job</text><text class="lang-zh" x="260" y="212" text-anchor="middle" fill="#6B645E" font-size="10">所以要按答案的形状选门，一件活一件活地选</text>
</g></svg>
%%%

%%% steps
step: **The mistake.** Habit says "activations go at the end of layers," so a `sigmoid` or `ReLU` gets stuck on the answer too
why: it *looks* tidy and consistent, which is exactly why this one slips through code review
step: **The damage.** `sigmoid` can never output `5`, and `tanh` can never output `100` — every large answer is squashed to the ceiling
why: therefore your model is not wrong by a little; it is physically unable to say the right number
step: **The fix, by job.** No bend at all (plain linear) when the answer is any number; a `sigmoid` when the answer is one yes/no
why: match the door to the shape of the answer, and the model can at least *express* the truth
%%%
~~~zh
%%% steps
step: **那个错误。** 习惯会说「激活函数放在每层末尾」，于是 `sigmoid` 或 `ReLU` 也被贴到了答案上
why: 它*看起来*整齐又一致，这正是它能通过代码评审的原因
step: **造成的损害。** `sigmoid` 永远输出不了 `5`，`tanh` 永远输出不了 `100` —— 每个大答案都被挤到天花板上
why: 因此你的模型不是错了一点点；它是物理上说不出那个正确的数字
step: **按活来修。** 答案是任意数字时，完全不加弯（普通线性）；答案是一个是/否时，用 `sigmoid`
why: 让门对上答案的形状，模型才至少能*说出*真相
%%%
~~~

%%% insight
This is the cleanest capability limit in the whole day, and it's worth saying out loud: **a bounded bend cannot express a big answer.** Not "does it badly" — *cannot*. The ceiling isn't a training problem you can fix with more data or more layers; it's built into the shape of the curve.
%%%
~~~zh
%%% insight
这是今天最干净的一个能力上限，值得念出来：**一个有界的弯说不出一个大答案。** 不是「说得不好」——
是*说不出来*。这个天花板不是一个你能用更多数据或更多层修好的训练问题；它长在曲线的形状里。
%%%
~~~

Which box goes with which job — the which-one-of-many door (called **softmax**), and the loss function that pairs with each — is Day 4's whole story. Today you only need the split: **inside, one cheap bend; at the door, match the answer.**

**So far:** two decisions, not one. Hidden layers: take ReLU and move on. Output layer: think, every time. **You can now answer "which activation should I use?" like someone who has shipped a model.**
~~~zh
哪个箱子配哪件活 —— 那个「从很多里挑一个」的门（叫 **softmax**），以及和每个门配对的 loss 函数 ——
是第 4 天一整天的故事。今天你只需要记住这个分工：**里面用一个便宜的弯；门口对上答案。**

**到这里为止：** 是两个决定，不是一个。hidden 层：拿 ReLU，往下走。output 层：每次都要想。
**你现在能像一个真正上线过模型的人那样回答「我该用哪个激活函数？」了。**
~~~

@@@ concept id=c9 tag="The jammed valve" zh_tag="卡住的阀门" title="When a ReLU quietly jams shut" zh_title="一个 ReLU 悄悄卡死的时候" gotit="Got the jammed valve" zh_gotit="懂了卡住的阀门"
ReLU saved us from the fading whisper — and then brought a puzzle of its own, one that's genuinely fun to solve because it is so *silent*: nothing crashes, no error prints, the loss just stops dropping.

Picture that one-way valve again, but this time it has **jammed shut**. Water piles up behind it; nothing gets through, ever.
~~~zh
ReLU 把我们从衰减的耳语里救了出来 —— 然后带来了它自己的一个谜题，一个解起来很有意思的谜题，因为
它太*安静*了：什么都不崩溃，什么错误都不打印，loss 就是停止下降了。

再想一次那个单向阀门，但这次它**卡死了**。水在它后面堆起来；什么都过不去，永远过不去。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve jammed fully shut. Water piles up behind the closed gate and nothing passes through. The pipe on the far side is empty. The valve is frozen and cannot reopen on its own."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A dead ReLU = the valve jammed SHUT</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一个死掉的 ReLU = 阀门卡死了</text><rect x="40" y="56" width="200" height="46" rx="4" fill="#DCE9F5" stroke="#7C6DAA" stroke-width="1.5"/><text class="lang-en" x="120" y="84" text-anchor="middle" fill="#5E5191">💧 water piles up 💧</text><text class="lang-zh" x="120" y="84" text-anchor="middle" fill="#5E5191">💧 水堆起来 💧</text><line x1="250" y1="46" x2="250" y2="112" stroke="#C93B3B" stroke-width="5"/><rect x="242" y="56" width="16" height="46" fill="#C93B3B"/><text x="250" y="40" text-anchor="middle" fill="#C93B3B" font-size="14">🔒</text><rect x="262" y="56" width="216" height="46" rx="4" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5" stroke-dasharray="4,3"/><text class="lang-en" x="370" y="84" text-anchor="middle" fill="#9A938A">nothing gets through</text><text class="lang-zh" x="370" y="84" text-anchor="middle" fill="#9A938A">什么都过不去</text><text class="lang-en" x="120" y="126" text-anchor="middle" fill="#6B645E" font-size="10">input stays negative</text><text class="lang-zh" x="120" y="126" text-anchor="middle" fill="#6B645E" font-size="10">输入一直是负的</text><text class="lang-en" x="370" y="126" text-anchor="middle" fill="#C93B3B" font-size="10">output frozen at 0</text><text class="lang-zh" x="370" y="126" text-anchor="middle" fill="#C93B3B" font-size="10">输出冻结在 0</text><text class="lang-en" x="260" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">no hand to pry it open — slope 0, so training can't grip it</text><text class="lang-zh" x="260" y="150" text-anchor="middle" fill="#C93B3B" font-size="10">没有手能把它掰开 —— 斜率 0，训练抓不住它</text></g></svg>
%%%

That is a [[dead ReLU||A ReLU whose input has become negative for every example, so it outputs a flat 0 forever. Its slope there is 0, so training gives it no signal, and it never recovers on its own.]]: a unit whose input has drifted negative for *every* example, so it outputs a flat `0` and never opens again.

**What the jammed valve gets right:** nothing passes; the output is frozen. **Where it breaks down:** a real valve you can pry open by hand — a dead ReLU has no handle, because its slope out there is exactly `0`. That's what makes this one permanent.
~~~zh
这就是一个 [[dead ReLU||一个 ReLU 的输入在每个样本上都变成了负的，于是它永远输出一个平的 0。它在那里的斜率是 0，所以训练给不了它任何信号，它自己也永远恢复不了。]]（死掉的 ReLU）：一个单元的输入在*每一个*样本上都漂到了负的那边，
于是它输出一个平的 `0`，再也不会打开。

**卡死的阀门这个比喻对在哪里：** 什么都不通；输出被冻住。**它在哪里不成立：** 真的阀门你可以用手
掰开 —— 一个死掉的 ReLU 没有把手，因为它在那外面的斜率正好是 `0`。这就是它为什么是永久性的。
~~~

%%% insight
Read that again slowly: the unit is not *wrong*, it's *gone*. Let 100 units die in a layer of 300 and you are paying for 300 neurons while only 200 of them learn anything — and the loss curve will never tell you.
%%%
~~~zh
%%% insight
慢慢再读一遍：这个单元不是*错了*，是*没了*。让一层 300 个单元里死掉 100 个，你就是在为 300 个
神经元付钱，而其中只有 200 个在学东西 —— 而 loss 曲线永远不会告诉你这件事。
%%%
~~~

#### What jams the valve, and how you catch it
Two crisp beats, then you're armed.
~~~zh
#### 什么让阀门卡住，以及你怎么抓到它
两个干脆的拍子，然后你就有武器了。
~~~

%%% steps
step: **The shove.** One oversized training step pushes a unit's input permanently negative
why: the learning rate is how big a step training takes — a real knob you'll meet on Day 5 when we walk downhill; too big, and a unit can be flung past no return
step: **The freeze.** Out there the slope is exactly 0, so no future signal can move it back
why: therefore the unit is stuck at 0 forever — dead, not merely quiet
step: **The tell.** Log the *fraction of ReLUs sitting at exactly 0*, not just the loss
why: when that fraction creeps up while the loss plateaus, you've found your culprit in one glance
%%%
~~~zh
%%% steps
step: **那一推。** 一次过大的训练步子，把一个单元的输入永久推到了负的那边
why: learning rate（学习率）就是训练一步走多大 —— 一个你在第 5 天下坡时会真正见到的旋钮；太大了，一个单元就可能被甩过再也回不来的地方
step: **冻结。** 在那外面斜率正好是 0，所以以后任何信号都推不动它
why: 因此这个单元永远卡在 0 —— 是死了，不只是安静
step: **线索。** 记录*正好等于 0 的 ReLU 占多少比例*，不要只看 loss
why: 当这个比例在 loss 停滞的同时慢慢爬升，你一眼就找到了凶手
%%%
~~~

Now play detective with the numbers themselves. Two things get printed each epoch — the loss, and the fraction of hidden ReLUs sitting at exactly `0`:
~~~zh
现在拿数字本身来当侦探。每一轮会打印两个东西 —— loss，以及正好等于 `0` 的 hidden ReLU 的比例：
~~~

%%% demo id=deadtell label="predict, then run"
predict: the loss stops improving after epoch 40. Which of the two numbers below tells you *why* — and what do you expect the fraction of zeros to do?
code: h = relu(X @ W1 + b1)            # one batch through the hidden layer
code: print(f"epoch {epoch} · loss {loss:.2f} · zeros {(h == 0).mean():.2f}")
out: epoch 10 · loss 0.42 · zeros 0.51
out: epoch 40 · loss 0.31 · zeros 0.58
out: epoch 70 · loss 0.31 · zeros 0.79
out: epoch 100 · loss 0.31 · zeros 0.80
take: <b>The loss says "stuck"; the zeros say "why."</b> About half the units at 0 is healthy ReLU sparsity. But watch the third column climb to `0.79`, then `0.80`, and *park* there while the loss flatlines at `0.31` — that is units dying, one shove at a time. One extra printed number turns a mystery into a diagnosis.
%%%
~~~zh
%%% demo id=deadtellzh label="先猜，再展开"
predict: loss 在第 40 轮之后就不再改善了。下面两个数字里，哪一个告诉你*为什么* —— 而且你觉得零的比例会怎么走？
code: h = relu(X @ W1 + b1)            # 一个 batch 走过 hidden 层
code: print(f"epoch {epoch} · loss {loss:.2f} · zeros {(h == 0).mean():.2f}")
out: epoch 10 · loss 0.42 · zeros 0.51
out: epoch 40 · loss 0.31 · zeros 0.58
out: epoch 70 · loss 0.31 · zeros 0.79
out: epoch 100 · loss 0.31 · zeros 0.80
take: <b>loss 说「卡住了」；零的比例说「为什么」。</b>大约一半单元在 0 是健康的 ReLU 稀疏性。但看第三列爬到 `0.79`、然后 `0.80`，接着*停在那里*，而 loss 在 `0.31` 上走平 —— 那就是单元在一个个死掉。多打印一个数字，就把一个谜题变成了一个诊断。
%%%
~~~

!!! c-warn ⚠️
<b>And the opposite failure, in one line.</b> ReLU has no ceiling, so starting weights that are too <em>large</em> make values grow layer after layer instead of dying — activations explode toward huge numbers. Same two cures: sensible starting weights (<b>He initialization</b>) and normalization layers.
!!!
~~~zh
!!! c-warn ⚠️
<b>还有反过来的那个失败，一行说完。</b>ReLU 没有上限，所以起始 weight 太<em>大</em>的时候，数值会
一层一层地长上去而不是死掉 —— 激活值朝着巨大的数字爆炸。解药还是那两个：合理的起始 weight
（<b>He initialization</b>）和 normalization 层。
!!!
~~~

**So far:** ReLU can die silently, and activation stats are how you spot it. **You can now diagnose a stuck network from its activations instead of guessing** — and the good news is that the cure is one line of code, waiting in the very next unit.
~~~zh
**到这里为止：** ReLU 会悄无声息地死掉，而激活值的统计就是你发现它的方法。**你现在能从激活值
诊断一个卡住的网络，而不是靠猜** —— 好消息是解药只有一行代码，就在下一个单元等着。
~~~

@@@ concept id=c10 tag="The rescue" zh_tag="救援" title="Leaky ReLU — one small leak brings it back" zh_title="Leaky ReLU —— 一个小小的漏就把它救回来" gotit="Got the rescue" zh_gotit="懂了这个救援"
Here's the delightful part: that permanent death has a fix you can picture in five seconds and type in one line.

Take the same one-way valve — and leave it with a **tiny permanent leak**. Positives still flow through as before. Negatives, instead of being fully shut off, seep past as a small trickle.
~~~zh
现在是让人高兴的部分：那个永久性的死亡，有一个你五秒钟就能想象出来、一行就能敲出来的解法。

拿同一个单向阀门 —— 让它带一个**很小的、永久的漏**。正数照旧直接流过。负数不再被完全关掉，而是
以一点细流渗过去。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="A one-way valve with a tiny leak. Top: positive flow passes straight through as before. Bottom: on the blocked side a small trickle still seeps past the gate instead of being fully stopped, keeping the valve alive."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Leaky ReLU = a valve with a tiny built-in leak</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Leaky ReLU = 一个自带小漏的阀门</text><rect x="40" y="38" width="440" height="32" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><polygon points="250,42 262,54 250,66" fill="#2D8B55"/><text class="lang-en" x="95" y="58" fill="#276b45">positive →</text><text class="lang-zh" x="95" y="58" fill="#276b45">正数 →</text><text class="lang-en" x="330" y="58" fill="#276b45" font-weight="bold">flows through 💧</text><text class="lang-zh" x="330" y="58" fill="#276b45" font-weight="bold">流过去 💧</text><rect x="40" y="92" width="440" height="36" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><line x1="250" y1="88" x2="250" y2="132" stroke="#C99A12" stroke-width="3"/><rect x="244" y="96" width="12" height="20" fill="#C99A12"/><text x="259" y="124" fill="#9A7A10" font-size="9">·····</text><text class="lang-en" x="95" y="114" fill="#8A6D3B">negative ←</text><text class="lang-zh" x="95" y="114" fill="#8A6D3B">负数 ←</text><text class="lang-en" x="320" y="114" fill="#8A6D3B" font-weight="bold">small trickle (slope α)</text><text class="lang-zh" x="320" y="114" fill="#8A6D3B" font-weight="bold">一点细流（斜率 α）</text><text class="lang-en" x="260" y="150" text-anchor="middle" fill="#2D8B55" font-size="10">the leak = a tiny non-zero slope → the unit can't fully die</text><text class="lang-zh" x="260" y="150" text-anchor="middle" fill="#2D8B55" font-size="10">这个漏 = 一点不为零的斜率 → 单元没法完全死掉</text></g></svg>
%%%

That's [[Leaky ReLU||Like ReLU, but negatives get a small slope (α·z, e.g. α=0.01) instead of a flat 0 — so the unit keeps a tiny slope on the negative side and can't fully die.]]. In plain words: negatives aren't zeroed, they're shrunk.

**What the leaky valve gets right:** the trickle is exactly the small slope that keeps the unit alive. **Where it breaks down:** a real leak is a fault you'd fix — here the leak is the *feature*, deliberately built in.
~~~zh
这就是 [[Leaky ReLU||和 ReLU 一样，只是负数拿到一个很小的斜率（α·z，比如 α=0.01），而不是一个平的 0 —— 所以单元在负的那一侧保留了一点斜率，没法完全死掉。]]。用大白话讲：负数不是被归零，而是被缩小。

**带漏的阀门这个比喻对在哪里：** 那点细流正好就是让单元活着的那点小斜率。**它在哪里不成立：**
真的漏是你会去修的故障 —— 这里的漏是*特性*，故意造进去的。
~~~

#### Bring the dead unit back, live
Take the jammed unit from the last section — its input is stuck out at `z = −3`. Watch what each bend does as that input drifts from `−5` to `−3`, because *movement* is exactly what a slope is. **Predict Leaky ReLU's numbers before you reveal them.**
~~~zh
#### 当场把死掉的单元救回来
拿上一节那个卡死的单元 —— 它的输入卡在 `z = −3`。看这个输入从 `−5` 漂到 `−3` 时，每个弯分别做了
什么，因为*动了多少*正是斜率的意思。**在展开之前先猜 Leaky ReLU 的数字。**
~~~

%%% demo id=leaky label="swap the bend — predict, then run"
predict: ReLU gives 0 at z = −5, −4 and −3 — that row never moves, and the stillness IS the death. What does Leaky ReLU (α = 0.01) give for the same three? And what do you think it does to the positives, 2 and 5?
code: relu(np.array([-5., -4., -3.]))
code: leaky_relu(np.array([-5., -4., -3.]), alpha=0.01)
code: leaky_relu(np.array([-5., -2., 0., 2., 5.]), alpha=0.01)
out: array([0., 0., 0.])
out: array([-0.05, -0.04, -0.03])
out: array([-0.05, -0.02,  0.  ,  2.  ,  5.  ])
take: <b>The whole rescue, in three rows.</b> ReLU's row never budges as `z` climbs — no movement means no slope, so nothing can pull the unit back. Leaky ReLU's row rises by `0.01` at every step: tiny, but not zero, so a real nudge finally reaches the unit and it can walk back to life. The third row shows what that costs you on the positive side: nothing at all — `2` and `5` come out exactly as ReLU would leave them.
%%%
~~~zh
%%% demo id=leakyzh label="换一个弯 —— 先猜，再展开"
predict: ReLU 在 z = −5、−4、−3 都给 0 —— 那一行永远不动，而这个「不动」就是死亡本身。Leaky ReLU（α = 0.01）对同样三个数给什么？你觉得它对正数 2 和 5 做了什么？
code: relu(np.array([-5., -4., -3.]))
code: leaky_relu(np.array([-5., -4., -3.]), alpha=0.01)
code: leaky_relu(np.array([-5., -2., 0., 2., 5.]), alpha=0.01)
out: array([0., 0., 0.])
out: array([-0.05, -0.04, -0.03])
out: array([-0.05, -0.02,  0.  ,  2.  ,  5.  ])
take: <b>整个救援，三行讲完。</b>`z` 往上爬时，ReLU 那一行一动不动 —— 不动就意味着没有斜率，所以没有任何东西能把这个单元拉回来。Leaky ReLU 那一行每一步都上升 `0.01`：很小，但不是零，所以一个真正的推力终于能到达这个单元，它可以慢慢走回来活着。第三行告诉你这在正的那一侧要付什么代价：完全不用付 —— `2` 和 `5` 出来的样子和 ReLU 留下的一模一样。
%%%
~~~

#### Why one small leak undoes a permanent death
Walk it through with the valve in mind. Here is the same fix drawn as a shape — the dashed red line is where plain ReLU sat flat, and the green line is the tilt we gave it instead:
~~~zh
#### 为什么一个小漏就能解开一个永久的死亡
带着阀门的画面走一遍。这里是同一个解法画成形状 —— 红色虚线是普通 ReLU 平坐着的地方，绿线是我们
改给它的那点倾斜：
~~~

%%% svg
<svg viewBox="0 0 520 155" role="img" aria-label="Leaky ReLU drawn against plain ReLU. For negative inputs, plain ReLU is a flat dashed line at zero while leaky ReLU slopes gently downward, so it keeps a small non-zero slope there. For positive inputs both rise as the same straight line. A note says the leak is drawn larger than real so it is visible."><g font-family="monospace"><line x1="60" y1="92" x2="330" y2="92" stroke="#E5DFD6" stroke-width="1.5"/><line x1="195" y1="24" x2="195" y2="116" stroke="#E5DFD6" stroke-width="1"/><path d="M70 92 L195 92" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><path d="M70 104 L195 92 L305 40" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text class="lang-en" x="120" y="120" font-size="11" fill="#2D8B55" text-anchor="middle">tiny leak (slope α)</text><text class="lang-zh" x="120" y="120" font-size="11" fill="#2D8B55" text-anchor="middle">小小的漏（斜率 α）</text><text class="lang-en" x="118" y="84" font-size="10" fill="#C93B3B" text-anchor="middle">ReLU: flat 0 (can die)</text><text class="lang-zh" x="118" y="84" font-size="10" fill="#C93B3B" text-anchor="middle">ReLU：平的 0（会死）</text><text class="lang-en" x="258" y="34" font-size="11" fill="#2D8B55" text-anchor="middle">positives pass</text><text class="lang-zh" x="258" y="34" font-size="11" fill="#2D8B55" text-anchor="middle">正数通过</text><text x="415" y="70" font-size="12" fill="#1a5c38">leaky(z)=max(αz, z)</text><text class="lang-en" x="195" y="140" font-size="9" fill="#9A938A" text-anchor="middle">the leak is drawn much bigger than the real α = 0.01, so you can see it</text><text class="lang-zh" x="195" y="140" font-size="9" fill="#9A938A" text-anchor="middle">图上的漏画得比真实的 α = 0.01 大很多，这样你才看得见</text></g></svg>
%%%

%%% steps
step: **The old dead end.** Plain ReLU's negative side is flat, slope exactly `0`
why: a unit pushed out there gets no signal, so it can never come back — the jammed valve
step: **The leak.** Give that side a small tilt instead — `α·z`, with `α` (the Greek letter "alpha") around `0.01`
why: small enough that ReLU's behaviour barely changes, big enough that the slope is no longer zero
step: **The way back.** A non-zero slope means a real nudge arrives
why: therefore the unit can climb back toward positive and start working again — no death, no rescue mission
%%%
~~~zh
%%% steps
step: **原来的死胡同。** 普通 ReLU 负的那一侧是平的，斜率正好 `0`
why: 被推到那外面的单元拿不到任何信号，所以永远回不来 —— 就是那个卡死的阀门
step: **那个漏。** 给那一侧一点小小的倾斜 —— `α·z`，其中 `α`（希腊字母「alpha」）大约是 `0.01`
why: 小到 ReLU 的行为几乎没变，又大到斜率不再是零
step: **回来的路。** 斜率不为零就意味着有一个真正的推力能到
why: 因此这个单元能往正的方向爬回来，重新开始工作 —— 不用死，也不用救援队
%%%
~~~

%%% insight
Notice the *shape* of this fix, because you'll see it again and again in machine learning: the problem was a hard zero, and the cure was not a clever algorithm — just refusing to let a number be exactly zero. Most of the modern activations after ReLU are variations on that one move.
%%%
~~~zh
%%% insight
注意这个解法的*形状*，因为你会在机器学习里反复见到它：问题是一个硬的零，而解药不是什么聪明的算法 ——
只是拒绝让一个数字正好等于零。ReLU 之后现代那些激活函数，大多都是这一招的变体。
%%%
~~~

**Victory lap.** Stop and look at what you're now holding: two silent failures, both understood, both with a named cure you could type today. That failure-plus-cure pairing *is* how engineers think about design — and you built it yourself, in about twenty minutes.
~~~zh
**庆功圈。** 停下来看看你现在手里拿着什么：两个安静的失败，都被你搞懂了，而且每一个都配了一个
有名字、你今天就能敲出来的解药。这种「失败 + 解药」的配对*就是*工程师思考设计的方式 —— 而这是
你自己在大约二十分钟里搭起来的。
~~~

!!! c-ok 🧰
<b>Three relatives you'll hear named.</b>
<br><b>· GELU</b> and <b>ELU</b> curve smoothly near zero instead of using a straight leak, so they keep a non-zero slope around the origin — where dying happens. (Far out on the negative side they do flatten too; <b>Leaky ReLU</b> is the one with a slope everywhere.) GELU is the bend inside modern transformer models.
<br><b>· He (Kaiming) initialization</b> heads off dead units <em>before</em> training starts: it sizes the random starting weights so activations begin in a healthy range. The default for a ReLU network, set with one line.
<br><b>· Xavier / Glorot initialization</b> is the sibling recipe for <code>tanh</code>/<code>sigmoid</code> layers.
!!!
~~~zh
!!! c-ok 🧰
<b>三个你会听到名字的亲戚。</b>
<br><b>· GELU</b> 和 <b>ELU</b> 在零附近是平滑地弯，而不是用一段直的漏，所以它们在原点周围保留了
不为零的斜率 —— 死亡就发生在那里。（在负的那一侧远处它们也会变平；<b>Leaky ReLU</b> 才是处处都有
斜率的那一个。）GELU 就是现代 transformer 模型里面那个弯。
<br><b>· He（Kaiming）initialization</b> 在训练开始<em>之前</em>就挡掉死单元：它把随机起始 weight
的大小调好，让激活值一开始就落在健康的范围里。ReLU 网络的默认选择，一行就能设。
<br><b>· Xavier / Glorot initialization</b> 是给 <code>tanh</code>/<code>sigmoid</code> 层的姊妹配方。
!!!
~~~

!!! c-info ⚖️
<b>Trade-off: ReLU's speed and sparsity vs. its dead-unit risk.</b> ReLU buys a dirt-cheap bend, healthy gradients in deep stacks, and lots of exact zeros (<b>sparsity</b>). What you give up is safety at zero. Softer bends that are never exactly zero near the origin (Leaky ReLU, ELU, GELU) trade a little extra math and less sparsity for units that don't die.
!!!
~~~zh
!!! c-info ⚖️
<b>取舍：ReLU 的速度和稀疏性 vs. 它的死单元风险。</b>ReLU 买到的是一个便宜到不要钱的弯、深层堆叠里
健康的梯度，以及大量正好为零的值（<b>sparsity</b>）。你交出去的是零点处的安全。那些在原点附近永远
不正好为零的更柔和的弯（Leaky ReLU、ELU、GELU），用一点额外的数学和更少的稀疏性，换来不会死掉的
单元。
!!!
~~~

@@@ concept id=c11 tag="Two ropes" zh_tag="两根绳子" title="One rope can't do XOR — two can" zh_title="一根绳子做不了 XOR —— 两根可以" gotit="Got the limit" zh_gotit="懂了这个上限"
Let's finish with the puzzle that started this entire field — and see the bend from the other side.

Picture a **room with four chairs**, one in each corner. Two red chairs sit on one diagonal; two blue chairs sit on the other. I hand you a single straight **rope**, pulled tight across the floor, and ask you to put both reds on one side and both blues on the other.
~~~zh
我们用启动了整个领域的那个谜题来收尾 —— 顺便从另一面再看一次这个弯。

想象一个**放着四把椅子的房间**，每个角落一把。两把红椅子在一条对角线上；两把蓝椅子在另一条上。
我给你一根拉直的**绳子**，横在地板上，请你把两把红的放在一边，两把蓝的放在另一边。
~~~

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A room floor with four chairs at the corners. Two red chairs sit on one diagonal, two blue chairs on the other. A single straight rope drawn across the floor separates only the bottom-left blue chair, so the top-right blue chair ends up on the same side as both reds — three of the four chairs are right and one is wrong, and no angle does better."><g font-family="monospace" font-size="11"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One straight rope can't split two diagonal pairs</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一根直绳子分不开两组对角的点</text><rect x="150" y="32" width="220" height="120" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><rect x="172" y="52" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="185" y="70" text-anchor="middle" fill="#C93B3B">🔴</text><rect x="322" y="106" width="26" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="335" y="124" text-anchor="middle" fill="#C93B3B">🔴</text><rect x="322" y="52" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="335" y="70" text-anchor="middle" fill="#5E5191">🔵</text><rect x="172" y="106" width="26" height="26" rx="4" fill="#EAF0FA" stroke="#5E5191" stroke-width="2"/><text x="185" y="124" text-anchor="middle" fill="#5E5191">🔵</text><line x1="140" y1="36" x2="276" y2="162" stroke="#9A5A12" stroke-width="3"/><text class="lang-en" x="284" y="152" fill="#9A5A12" font-size="10" text-anchor="start">straight rope</text><text class="lang-zh" x="284" y="152" fill="#9A5A12" font-size="10" text-anchor="start">直的绳子</text><circle cx="335" cy="65" r="20" fill="none" stroke="#C93B3B" stroke-width="1.5" stroke-dasharray="4,3"/><text class="lang-en" x="376" y="62" fill="#C93B3B" font-size="10" text-anchor="start">this 🔵 ends up</text><text class="lang-zh" x="376" y="62" fill="#C93B3B" font-size="10" text-anchor="start">这个 🔵 最后</text><text class="lang-en" x="376" y="76" fill="#C93B3B" font-size="10" text-anchor="start">with the reds</text><text class="lang-zh" x="376" y="76" fill="#C93B3B" font-size="10" text-anchor="start">和红的在一起</text><text class="lang-en" x="72" y="96" text-anchor="middle" fill="#6B645E" font-size="10">reds on one</text><text class="lang-zh" x="72" y="96" text-anchor="middle" fill="#6B645E" font-size="10">红的在一条</text><text class="lang-en" x="72" y="110" text-anchor="middle" fill="#6B645E" font-size="10">diagonal,</text><text class="lang-zh" x="72" y="110" text-anchor="middle" fill="#6B645E" font-size="10">对角线上，</text><text class="lang-en" x="72" y="124" text-anchor="middle" fill="#6B645E" font-size="10">blues the other</text><text class="lang-zh" x="72" y="124" text-anchor="middle" fill="#6B645E" font-size="10">蓝的在另一条</text><text class="lang-en" x="260" y="186" text-anchor="middle" fill="#C93B3B" font-size="10">any angle → best you get is 3 of 4 (you need a bend + a 2nd rope)</text><text class="lang-zh" x="260" y="186" text-anchor="middle" fill="#C93B3B" font-size="10">任何角度 → 最好也只有 4 中 3（你需要一个弯 + 第二根绳子）</text></g></svg>
%%%

Go on, try it. Any angle you like — then flip on the hidden layer and watch the second rope appear.
~~~zh
来，试试看。任意角度都行 —— 然后打开 hidden 层，看第二根绳子出现。
~~~

%%% viz src=../../viz/xor-limit.html title="why one neuron can't do XOR" caption="interactive — slide the single line (stuck at 3/4), then switch on the hidden layer"
%%%

**What the room of chairs gets right:** the diagonal layout is the entire trap. **Where it breaks down:** with a real rope you're stuck for good — a network can lay down a *second* rope, which changes everything.
~~~zh
**这个椅子房间这个比喻对在哪里：** 对角的摆法就是全部的陷阱。**它在哪里不成立：** 真的绳子你就
永远被困住了 —— 而一个网络可以再放*第二根*绳子，那就完全不一样了。
~~~

#### From "impossible" to "easy" in three moves

%%% steps
step: **The failure you just felt.** Slide the rope anywhere: three chairs land right, the fourth is always wrong
why: one straight cut can't separate two diagonal pairs — the best score is 3 of 4, forever
step: **The name for it.** That room is the [[XOR||XOR: output 1 when exactly one of two inputs is on, else 0. Its two classes sit on opposite diagonals, so no single straight line separates them.]] problem — "on when exactly one input is on"
why: a single neuron draws exactly one straight line (that was Day 1), so XOR is not [[linearly separable||A dataset is linearly separable if one straight line (or flat plane) can put every class on its own side. XOR is the classic example that is NOT.]] — and no activation changes that, because one neuron is still one boundary
step: **The rescue: two ropes, laid side by side.** Switch on the hidden layer in the demo and *two* cuts appear at once
why: therefore one cut slices off the (0,0) corner and the other slices off the (1,1) corner, leaving both reds in the stripe between them — the two cuts are drawn side by side by two hidden units, and the bend is what stops the output layer's combination of them from folding back into one straight line
%%%
~~~zh
#### 从「不可能」到「容易」，三步走

%%% steps
step: **你刚刚亲手感受到的那个失败。** 绳子随便滑到哪里：三把椅子对了，第四把总是错的
why: 一条直的切割分不开两组对角的点 —— 最好的成绩永远是 4 中 3
step: **它的名字。** 那个房间就是 [[XOR||XOR：两个输入里正好有一个是开的时候输出 1，否则输出 0。它的两类点落在两条相反的对角线上，所以没有任何一条直线能分开它们。]] 问题 —— 「正好一个输入是开的时候才开」
why: 一个神经元只画一条直线（那是第 1 天的事），所以 XOR 不是 [[linearly separable||一个数据集是线性可分的，指一条直线（或一个平面）就能把每一类放到自己一边。XOR 是那个经典的「不能」的例子。]]（线性可分）—— 而且换任何激活函数都改不了这一点，因为一个神经元还是一条边界
step: **救援：两根绳子，并排放。** 在演示里打开 hidden 层，*两*条切割一下就出现了
why: 因此一条切掉 (0,0) 那个角，另一条切掉 (1,1) 那个角，两把红的留在它们之间那条带子里 —— 这两条切割是由两个 hidden 单元并排画出来的，而那个弯就是让 output 层对它们的组合没法折回成一条直线的东西
%%%
~~~

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="Left panel: one straight rope across the four chairs cuts off only the top-left red chair, so the bottom-right red chair ends up on the wrong side — three of four correct. Right panel: two roughly parallel cuts, each drawn by its own hidden unit from the same input, leave a middle stripe that contains both red points, while each blue corner is cut off outside the stripe — four of four correct."><g font-family="monospace" font-size="10"><text class="lang-en" x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">one rope → 3 of 4</text><text class="lang-zh" x="130" y="16" text-anchor="middle" fill="#C93B3B" font-size="11">一根绳子 → 4 中 3</text><rect x="50" y="26" width="160" height="100" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><circle cx="80" cy="52" r="7" fill="#C93B3B"/><circle cx="180" cy="100" r="7" fill="#C93B3B"/><circle cx="180" cy="52" r="7" fill="#5E5191"/><circle cx="80" cy="100" r="7" fill="#5E5191"/><line x1="46" y1="89" x2="150" y2="24" stroke="#9A5A12" stroke-width="2.5"/><circle cx="180" cy="100" r="13" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="3,3"/><text class="lang-en" x="130" y="142" text-anchor="middle" fill="#C93B3B">this 🔴 is on the wrong side</text><text class="lang-zh" x="130" y="142" text-anchor="middle" fill="#C93B3B">这个 🔴 在错的一边</text><text x="255" y="80" fill="#B8AEA2" font-size="18">→</text><text class="lang-en" x="395" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">two cuts, side by side → 4 of 4</text><text class="lang-zh" x="395" y="16" text-anchor="middle" fill="#2D8B55" font-size="11">两条切割并排 → 4 中 4</text><rect x="315" y="26" width="160" height="100" rx="5" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><polygon points="315,62 315,26 341,26 475,90 475,126 449,126" fill="#EAF5EE" opacity="0.85"/><path d="M315 62 L449 126" fill="none" stroke="#2D8B55" stroke-width="2.5"/><path d="M341 26 L475 90" fill="none" stroke="#2D8B55" stroke-width="2.5"/><text class="lang-en" x="322" y="80" fill="#276b45" font-size="8" text-anchor="start">unit A</text><text class="lang-zh" x="322" y="80" fill="#276b45" font-size="8" text-anchor="start">单元 A</text><text class="lang-en" x="430" y="44" fill="#276b45" font-size="8" text-anchor="start">unit B</text><text class="lang-zh" x="430" y="44" fill="#276b45" font-size="8" text-anchor="start">单元 B</text><circle cx="345" cy="52" r="7" fill="#C93B3B"/><circle cx="445" cy="100" r="7" fill="#C93B3B"/><circle cx="445" cy="52" r="7" fill="#5E5191"/><circle cx="345" cy="100" r="7" fill="#5E5191"/><text class="lang-en" x="395" y="142" text-anchor="middle" fill="#2D8B55">both 🔴 inside the stripe, each 🔵 cut off outside</text><text class="lang-zh" x="395" y="142" text-anchor="middle" fill="#2D8B55">两个 🔴 都在带子里，每个 🔵 都被切在外面</text><text class="lang-en" x="395" y="156" text-anchor="middle" fill="#9A938A" font-size="9">two hidden units cut in parallel; the output layer combines them</text><text class="lang-zh" x="395" y="156" text-anchor="middle" fill="#9A938A" font-size="9">两个 hidden 单元并行地切；output 层把它们组合起来</text></g></svg>
%%%

%%% insight
Look at that stripe: it isn't a smooth curve — it is built from **two straight cuts**, each drawn by its own hidden unit from the same input, and then combined by the output layer. The bend is what stops that combination from folding back into one straight line (remember the rulers). That is what a ReLU network always builds: flat pieces joined at kinks. Stack enough of them and you can trace any shape you like, the way a hundred short straight fence panels can follow the edge of a round pond. That is also its honest limit — a ReLU net *approximates* a smooth curve with many small kinks; it never matches one exactly.
%%%
~~~zh
%%% insight
看那条带子：它不是一条平滑的曲线 —— 它是由**两条直的切割**拼出来的，每一条由同一个输入喂养的一个
hidden 单元画出，然后被 output 层组合起来。那个弯就是让这个组合没法折回成一条直线的东西（还记得
那两把尺子）。这就是一个 ReLU 网络永远在造的东西：一段段平的片，在折角处接起来。堆得够多，你就能
描出任何你想要的形状，就像一百块短短的直栅栏能沿着一个圆水塘的边围一圈。这也是它诚实的上限 ——
一个 ReLU 网络是用很多小折角去*逼近*一条平滑曲线；它永远不会正好等于那条曲线。
%%%
~~~

**So far:** the bend doesn't give one neuron new powers — it lets *many* simple pieces combine into something no straight line could reach. (And no bend rescues a network that's too small or fed useless inputs; the activation changes the *shape* of the response, never what the layer can see.)

~~~zh
**到这里为止：** 这个弯并没有给单个神经元新的能力 —— 它让*许多*简单的片能组合成一条直线永远到不了
的东西。（而且没有任何弯能救一个太小、或者被喂了没用输入的网络；激活函数改变的是响应的*形状*，
从来不是这一层能看到什么。）
~~~

#### Why the big labs care
`ReLU` is the one you'll meet most often — dirt cheap, and it keeps the backward whisper loud. That's why the **ReLU family** sits inside the model that reads a photo, the one that ranks your feed, and the one that answers your questions.

"Family" is the important word there. Different labs pick different cousins for exactly the reason you'd pick Leaky ReLU — never let the bend flatten where it matters. Modern transformer models reach for **GELU** (a smoothed ReLU), and some newer ones use a gated cousin called **SwiGLU**. Sigmoid and its cousin [[softmax||Turns a vector of scores into probabilities that sum to 1. Used at the output of a classifier.]] mostly live at the shipping door, where you want probabilities — that's Day 4's story.

The bend really is the quiet reason "deep" learning works at all, and now you know why.
~~~zh
#### 为什么大实验室在意这件事
`ReLU` 是你最常会遇到的那一个 —— 便宜到不要钱，而且它让往回传的那句耳语保持大声。这就是为什么
**ReLU 家族**坐在那个读照片的模型里、那个给你的信息流排序的模型里，以及那个回答你问题的模型里。

「家族」这个词很重要。不同的实验室挑不同的表亲，理由和你会挑 Leaky ReLU 完全一样 —— 别让这个弯在
要紧的地方变平。现代 transformer 模型伸手拿 **GELU**（一个被磨平的 ReLU），有些更新的用一个带门的
表亲叫 **SwiGLU**。Sigmoid 和它的表亲 [[softmax||把一串分数变成加起来等于 1 的概率。用在分类器的输出端。]] 大多住在出货口，那里你想要的是概率 —— 那是第 4 天的故事。

这个弯真的就是「深度」学习之所以能成立的那个安静的原因，而现在你知道为什么了。
~~~

@@@ concept id=c12 tag="Recap" zh_tag="回顾" title="Today in one page" zh_title="一页看完今天" gotit="Got the recap" zh_gotit="回顾好了"
Take a breath — you did the whole thing. Here's a way to hold it all at once: today was one **road trip**, and these were the five signposts you drove past, in order. Each one only makes sense because of the one before it.
~~~zh
喘口气 —— 你整个走完了。这里有一个把它们一次全握住的办法：今天是一趟**公路旅行**，而下面是你按
顺序开过的五个路牌。每一个之所以说得通，都是因为它前面那一个。
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A straight road from start to finish with five numbered signposts standing along it in order: one the trap, two the bend, three the family of bends, four the quiet failures, five the limits and the door."><g font-family="monospace" font-size="9">
<text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Today = one road trip, five signposts in order</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">今天 = 一趟公路旅行，五个路牌按顺序</text>
<rect x="20" y="88" width="480" height="16" rx="3" fill="#E5DFD6"/>
<line x1="26" y1="96" x2="494" y2="96" stroke="#FDF9F3" stroke-width="2" stroke-dasharray="10,8"/>
<circle cx="24" cy="96" r="6" fill="#2D8B55"/><text class="lang-en" x="24" y="118" text-anchor="middle" fill="#276b45">start</text><text class="lang-zh" x="24" y="118" text-anchor="middle" fill="#276b45">起点</text>
<line x1="86" y1="68" x2="86" y2="88" stroke="#9A5A12" stroke-width="2"/><rect x="60" y="48" width="52" height="20" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="86" y="62" text-anchor="middle" fill="#8A6D3B">1 trap</text><text class="lang-zh" x="86" y="62" text-anchor="middle" fill="#8A6D3B">1 陷阱</text>
<line x1="172" y1="104" x2="172" y2="124" stroke="#9A5A12" stroke-width="2"/><rect x="146" y="124" width="52" height="20" rx="3" fill="#EFEAF7" stroke="#7C6DAA"/><text class="lang-en" x="172" y="138" text-anchor="middle" fill="#5E5191">2 bend</text><text class="lang-zh" x="172" y="138" text-anchor="middle" fill="#5E5191">2 弯</text>
<line x1="258" y1="68" x2="258" y2="88" stroke="#9A5A12" stroke-width="2"/><rect x="228" y="48" width="60" height="20" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="258" y="62" text-anchor="middle" fill="#276b45">3 family</text><text class="lang-zh" x="258" y="62" text-anchor="middle" fill="#276b45">3 家族</text>
<line x1="344" y1="104" x2="344" y2="124" stroke="#9A5A12" stroke-width="2"/><rect x="312" y="124" width="64" height="20" rx="3" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="344" y="138" text-anchor="middle" fill="#C93B3B">4 failures</text><text class="lang-zh" x="344" y="138" text-anchor="middle" fill="#C93B3B">4 失败</text>
<line x1="430" y1="68" x2="430" y2="88" stroke="#9A5A12" stroke-width="2"/><rect x="400" y="48" width="60" height="20" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text class="lang-en" x="430" y="62" text-anchor="middle" fill="#8A6D3B">5 limits</text><text class="lang-zh" x="430" y="62" text-anchor="middle" fill="#8A6D3B">5 上限</text>
<circle cx="496" cy="96" r="6" fill="#C93B3B"/><text class="lang-en" x="488" y="118" text-anchor="middle" fill="#6B645E">finish 🏁</text><text class="lang-zh" x="488" y="118" text-anchor="middle" fill="#6B645E">终点 🏁</text>
</g></svg>
%%%

**Where the road-trip picture breaks down:** a real trip is a straight road you drive once — these signposts are a loop you'll circle back to for years, and today's whole point (the bend!) is the opposite of straight.

#### Now rebuild the road yourself, one "therefore" at a time
Don't re-read the day — *re-derive* it. Each link below is forced by the one above it, so if you can say the first line out loud, the rest follow.
~~~zh
**公路旅行这个比喻在哪里不成立：** 真的旅行是一条你只开一次的直路 —— 而这些路牌是一个你以后好几年
都会绕回来的圈，而且今天的全部重点（那个弯！）恰恰是「直」的反面。

#### 现在你自己把这条路重建一遍，一次一个「所以」
不要重读今天的内容 —— 去*重新推导*它。下面每一环都被它上面那一环逼出来，所以只要你能把第一句念
出来，后面的就自己跟着来了。
~~~

%%% steps
step: **Link 1 — the trap.** Stack straight-line layers and they fold back into one straight line
why: therefore depth on its own buys you nothing — that was the pair of rulers
step: **Link 2 — so insert a bend.** One small non-linearity between layers and they can't fold flat
why: weights mix, the bend curves; therefore we now need to pick *which* bend
step: **Link 3 — the danger is flatness.** The hard on/off switch has slope 0 everywhere, and the smooth S-curves keep a tilt only in the middle
why: small slopes multiply layer after layer, so the backward whisper fades — the vanishing gradient
step: **Link 4 — ReLU keeps a tilt of exactly `1` on the positive side.** Cheap, sparse, and the signal survives depth
why: therefore hidden layers default to ReLU (or GELU) — but its flat `0` side can jam a unit shut, so we refuse the hard zero with a leak (Leaky ReLU) plus He initialization
step: **Link 5 — the door, and the ceiling.** The output layer must match the *shape* of the answer, and one neuron is still one straight cut
why: therefore a bend lets many simple pieces combine, but XOR still needs layers — which is exactly where tomorrow starts
%%%
~~~zh
%%% steps
step: **第 1 环 —— 陷阱。** 把直线的层叠起来，它们会折回成一条直线
why: 因此光有深度什么都买不到 —— 那就是那两把尺子
step: **第 2 环 —— 所以塞进一个弯。** 层与层之间一个小小的非线性，它们就压不平了
why: weight 负责混合，弯负责弯曲；因此我们现在需要挑*哪一个*弯
step: **第 3 环 —— 危险的是平。** 硬开关处处斜率为 0，而平滑的 S 形曲线只在中间保留倾斜
why: 小斜率一层一层地乘起来，所以往回传的耳语会衰减 —— 就是 vanishing gradient
step: **第 4 环 —— ReLU 在正的那一侧保留正好 `1` 的倾斜。** 便宜、稀疏，而且信号能活过深度
why: 因此 hidden 层默认用 ReLU（或 GELU）—— 但它平的 `0` 那一侧会把单元卡死，所以我们用一个漏（Leaky ReLU）加上 He initialization 来拒绝那个硬零
step: **第 5 环 —— 门，和天花板。** output 层必须对上答案的*形状*，而一个神经元还是一条直的切割
why: 因此一个弯让许多简单的片能组合起来，但 XOR 仍然需要层 —— 而那正是明天开始的地方
%%%
~~~

Here are those same five links drawn as a staircase, each one resting on the one above it. This is the picture to rebuild in your head tomorrow morning:
~~~zh
下面是同样这五环画成一段楼梯，每一级都搁在上一级上。这就是明天早上你要在脑子里重建的那张图：
~~~

%%% svg
<svg viewBox="0 0 520 235" role="img" aria-label="The day rebuilt as one chain of consequences, drawn as a staircase where each step rests on the one above it. Link one: straight plus straight is still straight. Therefore link two: we insert a bend. Therefore link three: flatness is the danger, because a slope near zero gives no learning signal. Therefore link four: ReLU keeps a slope of exactly one on the positive side, though its flat zero side can jam, so we add a small leak. Therefore link five: match the output door to the shape of the answer, and remember that XOR still needs layers."><g font-family="monospace" font-size="10">
<text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The day as one chain — each link forced by the one above</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">一整天串成一条链 —— 每一环都被上一环逼出来</text>
<rect x="24" y="28" width="320" height="26" rx="5" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text class="lang-en" x="34" y="45" fill="#C93B3B" text-anchor="start">1 · straight + straight = still straight</text><text class="lang-zh" x="34" y="45" fill="#C93B3B" text-anchor="start">1 · 直的 + 直的 = 还是直的</text>
<line x1="184" y1="54" x2="184" y2="68" stroke="#B8AEA2" stroke-width="1.5"/><text class="lang-en" x="192" y="66" fill="#9A938A" font-size="8" text-anchor="start">therefore</text><text class="lang-zh" x="192" y="66" fill="#9A938A" font-size="8" text-anchor="start">所以</text>
<rect x="44" y="68" width="320" height="26" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text class="lang-en" x="54" y="85" fill="#5E5191" text-anchor="start">2 · so we insert a bend</text><text class="lang-zh" x="54" y="85" fill="#5E5191" text-anchor="start">2 · 所以我们塞进一个弯</text>
<line x1="204" y1="94" x2="204" y2="108" stroke="#B8AEA2" stroke-width="1.5"/><text class="lang-en" x="212" y="106" fill="#9A938A" font-size="8" text-anchor="start">therefore</text><text class="lang-zh" x="212" y="106" fill="#9A938A" font-size="8" text-anchor="start">所以</text>
<rect x="64" y="108" width="320" height="26" rx="5" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text class="lang-en" x="74" y="125" fill="#8A6D3B" text-anchor="start">3 · flat slope = no learning signal</text><text class="lang-zh" x="74" y="125" fill="#8A6D3B" text-anchor="start">3 · 斜率平 = 没有学习信号</text>
<line x1="224" y1="134" x2="224" y2="148" stroke="#B8AEA2" stroke-width="1.5"/><text class="lang-en" x="232" y="146" fill="#9A938A" font-size="8" text-anchor="start">therefore</text><text class="lang-zh" x="232" y="146" fill="#9A938A" font-size="8" text-anchor="start">所以</text>
<rect x="84" y="148" width="320" height="26" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text class="lang-en" x="94" y="165" fill="#276b45" text-anchor="start">4 · ReLU: slope 1 — but add a leak</text><text class="lang-zh" x="94" y="165" fill="#276b45" text-anchor="start">4 · ReLU：斜率 1 —— 但要加一个漏</text>
<line x1="244" y1="174" x2="244" y2="188" stroke="#B8AEA2" stroke-width="1.5"/><text class="lang-en" x="252" y="186" fill="#9A938A" font-size="8" text-anchor="start">therefore</text><text class="lang-zh" x="252" y="186" fill="#9A938A" font-size="8" text-anchor="start">所以</text>
<rect x="104" y="188" width="320" height="26" rx="5" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text class="lang-en" x="114" y="205" fill="#5E5191" text-anchor="start">5 · match the door; XOR needs layers</text><text class="lang-zh" x="114" y="205" fill="#5E5191" text-anchor="start">5 · 对上那个门；XOR 需要层</text>
</g></svg>
%%%

#### Cheat-sheet · the one picture to keep
The three working bends side by side, drawn on the same axes so you can read each range straight off:
~~~zh
#### 小抄 · 要留住的那一张图
三个能用的弯并排放，画在同一组坐标轴上，这样你能直接读出各自的取值范围：
~~~

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Recap figure: ReLU, sigmoid and tanh on one set of labelled axes with gridlines at minus one, zero and plus one. ReLU is flat at zero for negatives then rises straight past plus one. Sigmoid rises from just above zero, passes exactly through one half at z equals zero, and heads toward plus one. Tanh passes exactly through zero at z equals zero and is the same distance below the axis on the left as above it on the right, running from minus one to plus one."><g font-family="monospace" font-size="12">
<rect x="292" y="13" width="10" height="10" fill="#2D8B55"/><text x="307" y="22" fill="#2D8B55" font-size="11" text-anchor="start">ReLU</text>
<rect x="352" y="13" width="10" height="10" fill="#7C6DAA"/><text x="367" y="22" fill="#7C6DAA" font-size="11" text-anchor="start">sigmoid</text>
<rect x="432" y="13" width="10" height="10" fill="#9A5A12"/><text x="447" y="22" fill="#9A5A12" font-size="11" text-anchor="start">tanh</text>
<line x1="60" y1="45" x2="470" y2="45" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<line x1="60" y1="95" x2="470" y2="95" stroke="#E5DFD6" stroke-width="1.5"/>
<line x1="60" y1="145" x2="470" y2="145" stroke="#E9E3D9" stroke-width="1" stroke-dasharray="3,3"/>
<text x="56" y="49" fill="#9A938A" font-size="10" text-anchor="end">+1</text>
<text x="56" y="99" fill="#9A938A" font-size="10" text-anchor="end">0</text>
<text x="56" y="149" fill="#9A938A" font-size="10" text-anchor="end">−1</text>
<line x1="265" y1="30" x2="265" y2="155" stroke="#E5DFD6" stroke-width="1"/>
<path d="M70 145 C 160 145, 235 99, 265 95 C 295 91, 370 49, 460 45" fill="none" stroke="#9A5A12" stroke-width="2.5"/>
<path d="M70 93 C 160 93, 235 73, 265 70 C 295 67, 370 55, 460 53" fill="none" stroke="#7C6DAA" stroke-width="2.5"/>
<path d="M70 95 L265 95 L455 25" fill="none" stroke="#2D8B55" stroke-width="2.5"/>
<circle cx="265" cy="70" r="3.5" fill="#C93B3B"/><text class="lang-en" x="256" y="64" fill="#C93B3B" font-size="10" text-anchor="end">sigmoid(0)=0.5</text><text class="lang-zh" x="256" y="64" fill="#C93B3B" font-size="10" text-anchor="end">sigmoid(0)=0.5</text>
<circle cx="265" cy="95" r="3.5" fill="#C93B3B"/><text class="lang-en" x="276" y="112" fill="#C93B3B" font-size="10" text-anchor="start">tanh(0)=0</text><text class="lang-zh" x="276" y="112" fill="#C93B3B" font-size="10" text-anchor="start">tanh(0)=0</text>
<text class="lang-en" x="130" y="88" fill="#9A938A" font-size="10" text-anchor="middle">negatives</text><text class="lang-zh" x="130" y="88" fill="#9A938A" font-size="10" text-anchor="middle">负数</text>
<text class="lang-en" x="380" y="140" fill="#9A938A" font-size="10" text-anchor="middle">positives</text><text class="lang-zh" x="380" y="140" fill="#9A938A" font-size="10" text-anchor="middle">正数</text>
<text class="lang-en" x="260" y="168" fill="#6B645E" font-size="10" text-anchor="middle">ReLU keeps climbing past 1 · sigmoid stays inside (0,1) · tanh stays inside (−1,1)</text><text class="lang-zh" x="260" y="168" fill="#6B645E" font-size="10" text-anchor="middle">ReLU 会一直爬过 1 · sigmoid 待在 (0,1) 里 · tanh 待在 (−1,1) 里</text>
</g></svg>
%%%

#### Cheat-sheet · the three bends at a glance
~~~zh
#### 小抄 · 三个弯一眼看完
~~~

%%% table
:: :: ReLU :: sigmoid :: tanh
Shape :: one-way valve :: soft dimmer :: balanced see-saw
Range :: `[0, ∞)` :: `(0, 1)` :: `(−1, 1)`
Slope :: `1` for positives, `0` for negatives :: peaks ≈ `0.25` at z=0 :: peaks ≈ `1.0` at z=0
Watch out :: can go **dead** at 0; also **not zero-centered** :: **saturates** at both tails; **not zero-centered** :: **saturates** at both tails
Reach for it :: hidden layers (the default) :: an output probability :: a balanced, bounded signal
%%%
~~~zh
%%% table
:: :: ReLU :: sigmoid :: tanh
形状 :: 单向阀门 :: 柔和的调光旋钮 :: 平衡的跷跷板
取值范围 :: `[0, ∞)` :: `(0, 1)` :: `(−1, 1)`
斜率 :: 正数为 `1`，负数为 `0` :: 在 z=0 处最高 ≈ `0.25` :: 在 z=0 处最高 ≈ `1.0`
要小心 :: 会在 0 处**死掉**；而且**不以零为中心** :: 两端都会**饱和**；**不以零为中心** :: 两端都会**饱和**
什么时候用 :: hidden 层（默认选择） :: 一个输出概率 :: 一个平衡、有界的信号
%%%
~~~

#### Cheat-sheet · the words you met today
~~~zh
#### 小抄 · 今天遇到的那些词
~~~

%%% jargon
activation | the little bend added at the end of a neuron, after the weighted sum
non-linearity | not a straight line — the property a bend adds so depth matters
linear collapse | stacked straight-line layers folding back into one layer
step function | the original hard on/off bend; flat everywhere, so it can't learn
ReLU | a one-way valve: max(0, z) — passes positives, zeroes negatives
sigmoid | a soft dimmer that squashes any number into (0, 1)
tanh | a see-saw that squashes into (−1, 1) and rests at 0
zero-centered | outputs sit evenly above and below 0 (tanh yes; sigmoid and ReLU no)
slope | how steep the curve is at a point — a slope near 0 means learning stalls there
saturation | the curve has gone flat, so the slope ≈ 0 and the unit barely responds
vanishing gradient | the learning signal shrinks toward 0 through a deep stack of small slopes
dead ReLU | a ReLU stuck outputting a flat 0 forever, with no slope to recover
sparsity | many units output exactly 0 at once, so only the ones that matter speak up
Leaky ReLU | ReLU with a tiny negative slope (α·z) so the unit can't fully die — a slope everywhere
GELU | a smooth ReLU-like bend (used in modern transformers); never exactly zero near the origin, though it does flatten far out on the negative side
He initialization | the sensible starting-weight recipe for ReLU nets, so fewer units start dead
Xavier initialization | the sibling recipe for tanh/sigmoid layers
hidden layer | any layer between the input and the final output layer
output layer | the last layer — its bend must match the shape of the answer you want
XOR | "on when exactly one input is on" — the classic pattern one straight line can't split
%%%
~~~zh
%%% jargon
activation | 加在神经元末端、在加权求和之后的那个小小的弯
non-linearity | 不是直线 —— 一个弯带来的、让深度有意义的那个性质
linear collapse | 叠起来的直线层折回成一层
step function | 最早的那个硬开关；处处是平的，所以学不了东西
ReLU | 一个单向阀门：max(0, z) —— 让正数通过，把负数归零
sigmoid | 一个柔和的调光旋钮，把任何数字挤进 (0, 1)
tanh | 一个跷跷板，把数字挤进 (−1, 1)，静止在 0
zero-centered | 输出均匀分布在 0 的上下（tanh 是；sigmoid 和 ReLU 不是）
slope | 曲线在某一点有多陡 —— 斜率接近 0 意味着学习在那里停滞
saturation | 曲线变平了，所以斜率 ≈ 0，单元几乎不反应
vanishing gradient | 学习信号穿过一叠小斜率时越缩越接近 0
dead ReLU | 一个永远输出平的 0 的 ReLU，没有斜率能让它恢复
sparsity | 很多单元同时正好输出 0，所以只有重要的那些才说话
Leaky ReLU | 带一点负斜率（α·z）的 ReLU，让单元没法完全死掉 —— 处处都有斜率
GELU | 一个平滑的、像 ReLU 的弯（现代 transformer 在用）；在原点附近永远不正好为零，不过在负的那一侧远处它也会变平
He initialization | 给 ReLU 网络挑起始 weight 的合理配方，让更少的单元一开始就是死的
Xavier initialization | 给 tanh/sigmoid 层的姊妹配方
hidden layer | 输入层和最后的 output 层之间的任何一层
output layer | 最后一层 —— 它的弯必须对上你想要的答案的形状
XOR | 「正好一个输入是开的时候才开」—— 一条直线分不开的那个经典图案
%%%
~~~

!!! c-ok 🎤
<b>If someone asks you what an activation is, here's the whole day in four sentences:</b>
<br><b>· Why it exists:</b> "It's the per-layer bend. Without it the stack collapses — <code>(x W₁) W₂ = x (W₁ W₂)</code>, and the two biases fold into one new bias as well, so depth adds nothing at all."
<br><b>· Why ReLU is the default:</b> "ReLU = max(0, z) is the cheap default and keeps the backward signal alive, because its positive-side slope of 1 doesn't shrink it the way sigmoid's ≈0.25 does."
<br><b>· What goes wrong, and what you reach for:</b> "The failures are silent — saturated units and dead ReLUs — so I watch activation statistics, not just the loss, and I reach for Leaky ReLU or GELU with He initialization."
<br>Every piece of that is something you can already explain in your own words.
!!!
~~~zh
!!! c-ok 🎤
<b>如果有人问你什么是激活函数，这里是一整天，四句话：</b>
<br><b>· 它为什么存在：</b>「它是每层的那个弯。没有它，整叠会坍缩 ——
<code>(x W₁) W₂ = x (W₁ W₂)</code>，两个 bias 也折成一个新的 bias，所以深度完全不加东西。」
<br><b>· 为什么 ReLU 是默认：</b>「ReLU = max(0, z) 是那个便宜的默认选择，而且它让往回传的信号活着，
因为它正侧的斜率 1 不会像 sigmoid 的 ≈0.25 那样把信号缩小。」
<br><b>· 什么会出错，以及你会伸手拿什么：</b>「这些失败都是安静的 —— 饱和的单元和死掉的 ReLU ——
所以我会看激活值的统计，不只看 loss，而且我会拿 Leaky ReLU 或 GELU 配 He initialization。」
<br>这里面每一块，都是你已经能用自己的话讲出来的东西。
!!!
~~~

That's the whole day. Next you'll stack these neurons into layers and push data through them — the **forward pass**.
~~~zh
今天就是这些。接下来你会把这些神经元堆成层，把数据推过去 —— 那就是**前向传播（forward pass）**。
~~~

@@@ quiz id=quiz tag="Quiz" zh_tag="小测" title="Click an answer — instant feedback on each" zh_title="点一个答案 —— 每题都有即时反馈" gotit="answer all four first" zh_gotit="先答完四题"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does an activation function add to a network? | a:1 | More parameters | Non-linearity (a bend) | Faster matrix multiplies (matmuls) | A bias term | fb: The activation is the non-linear bend; without it, layers stay linear and depth is wasted.
q: With NO activation, ten stacked linear layers behave like: | a:1 | a much more powerful model | a single linear layer | a random function | a sigmoid | fb: (x W₁)W₂…W₁₀ = x (W₁…W₁₀) — one combined matrix, and the ten biases fold into one new bias too. Depth buys nothing without a bend.
q: Why does deep sigmoid suffer the "vanishing gradient" while deep ReLU largely doesn't? | a:2 | Sigmoid uses more memory | ReLU is written in C | Sigmoid's slope tops out near 0.25, so multiplying it through many layers shrinks the backward signal toward 0; ReLU's positive slope is 1, so it doesn't shrink | Sigmoid outputs negative numbers | fb: The backward signal is multiplied by each layer's slope. With ≈0.25 per layer it fades fast (0.25⁴ ≈ 0.004); ReLU's slope of 1 keeps it full-strength.
q: Your network trains without error, but the loss stops dropping early and stays high. You log the hidden ReLU outputs and find most are exactly 0 for every input in the batch. Most likely explanation — and a fix? | a:1 | Sigmoid at the output is always a bug | Many ReLU units are "dead": their input stays negative for all inputs, so max(0, z) is always 0 and they can no longer learn — switching those units to Leaky ReLU keeps a small negative slope so they can recover | The GPU ran out of memory, which always makes ReLU output zero | A high loss with zero activations means the network has already converged | fb: A ReLU that always sees a negative input outputs a flat 0 for every example — a dead unit. Many dead units mean much of the network does nothing, so the loss plateaus. You spot it from activation statistics (fraction of zeros), and Leaky ReLU (a small negative slope) is the standard cure.
%%%
~~~zh
四道快问，每题都有即时反馈 —— 四题都答完就算完成今天。（如果有一题把你绊住了，那是提示你往回翻，
不是给你打了不及格。）

%%% quiz
q: 一个激活函数给网络加了什么？ | a:1 | 更多 parameter | 非线性（一个弯） | 更快的矩阵乘法 | 一个 bias 项 | fb: 激活函数就是那个非线性的弯；没有它，各层还是线性的，深度就浪费了。
q: 在**没有**激活函数的情况下，十层叠起来的线性层表现得像： | a:1 | 一个强大得多的模型 | 一个单独的线性层 | 一个随机函数 | 一个 sigmoid | fb: (x W₁)W₂…W₁₀ = x (W₁…W₁₀) —— 一个合并后的矩阵，而那十个 bias 也折成一个新的 bias。没有弯，深度什么都买不到。
q: 为什么很深的 sigmoid 会得「梯度消失」，而很深的 ReLU 基本上不会？ | a:2 | Sigmoid 用更多内存 | ReLU 是用 C 写的 | Sigmoid 的斜率最高只到 0.25 左右，穿过很多层乘下来就把往回传的信号压向 0；ReLU 正侧的斜率是 1，所以不会缩小 | Sigmoid 会输出负数 | fb: 往回传的信号会被每一层的斜率乘一次。每层 ≈0.25 的话它衰减得很快（0.25⁴ ≈ 0.004）；ReLU 那个 1 的斜率让它保持全强度。
q: 你的网络训练时不报错，但 loss 很早就不再下降，一直很高。你把 hidden 层的 ReLU 输出打出来，发现对 batch 里每个输入它们大多正好是 0。最可能的解释 —— 以及一个解法？ | a:1 | 在输出端用 sigmoid 永远是个 bug | 很多 ReLU 单元「死了」：它们的输入对所有输入都保持为负，所以 max(0, z) 永远是 0，它们再也学不了 —— 把那些单元换成 Leaky ReLU，保留一点负斜率，它们就能恢复 | 显卡内存不够了，这总会让 ReLU 输出零 | 激活值为零而 loss 很高，说明网络已经收敛了 | fb: 一个永远看到负输入的 ReLU 对每个样本都输出一个平的 0 —— 一个死掉的单元。很多死单元意味着网络的很大一部分什么都不做，所以 loss 走平。你从激活值的统计（零的比例）看出它，而 Leaky ReLU（一点负斜率）是标准解药。
%%%
~~~

@@@ produce id=produce tag="Produce" zh_tag="动手做" title="Watch two layers collapse — then break the collapse" zh_title="看两层坍缩 —— 然后打破这个坍缩" gotit="Done" zh_gotit="做完了"
Time to see today's big idea with your own eyes. You'll build two straight-line layers with no bend between them and watch them fold into a single matrix — then drop a `ReLU` in the middle and watch the fold stop working. **Predict first:** will `(x@W1)@W2` and `x@(W1@W2)` print the same numbers, or different ones? Then run it and **watch.** Pick one path.
~~~zh
是时候用你自己的眼睛看看今天这个大想法了。你会搭两层之间没有弯的直线层，看它们折成一个矩阵 ——
然后在中间丢一个 `ReLU`，看那个折叠不再成立。**先预测：** `(x@W1)@W2` 和 `x@(W1@W2)` 会打印出
一样的数字，还是不一样的？然后跑一下，**仔细看**。挑一条路走。
~~~

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-02-activations/experiment.py`. Write `step`, `relu`, `leaky_relu`, `sigmoid`, and `tanh`, and print all five over the grid `np.linspace(-6, 6, 13)`. **Notice** that the step is a flat 0/1 jump, ReLU zeros negatives, `leaky_relu` lets a small trickle through, sigmoid stays in `(0,1)`, and tanh stays in `(−1,1)` and is 0 at 0. Also print `sigmoid'(0) ≈ 0.25` and note that ReLU's slope is `0` for every negative input while `leaky_relu`'s is a small `alpha` — the seed of both the dead-ReLU failure and its cure.

Then use the anchor pair `W1=[[1,2],[0,1]]`, `W2=[[1,0],[3,1]]` with `x_a=[[1.0, -3.0]]` and `x_b=[[-1.0, 3.0]]` (they add up to all zeros), and run the two paths side by side:

- **the collapse:** `(x_a@W1)@W2` equals `x_a@(W1@W2)` — print `W1@W2` (expect `[[7,2],[3,1]]`), proving two straight-line layers = one matrix.
- **the break:** a straight-line stack always *adds up* — `g(x_a) + g(x_b)` equals `g(x_a + x_b)` exactly. **Watch** the bent path fail that test: `f(x) = relu(x@W1)@W2` gives `f(x_a) + f(x_b) = [[4, 1]]` but `f(x_a + x_b) = [[0, 0]]`. Since every single matrix adds up, no single matrix can be doing this — the bend really did buy you something new.

Run with `python3 sessions/m02-the-neuron/day-02-activations/experiment.py`.
~~~zh
#### 路线 A · 自己写
创建 `sessions/m02-the-neuron/day-02-activations/experiment.py`。写出 `step`、`relu`、`leaky_relu`、
`sigmoid` 和 `tanh`，在网格 `np.linspace(-6, 6, 13)` 上把五个都打印出来。**注意看**：step 是一个平的
0/1 跳跃，ReLU 把负数归零，`leaky_relu` 让一点细流通过，sigmoid 待在 `(0,1)` 里，tanh 待在 `(−1,1)`
里而且在 0 处是 0。另外打印 `sigmoid'(0) ≈ 0.25`，并记下 ReLU 对每一个负输入斜率都是 `0`，而
`leaky_relu` 是一个小小的 `alpha` —— 这既是死 ReLU 那个失败的种子，也是它解药的种子。

然后用这一对锚定值 `W1=[[1,2],[0,1]]`、`W2=[[1,0],[3,1]]`，配 `x_a=[[1.0, -3.0]]` 和
`x_b=[[-1.0, 3.0]]`（它们加起来全是零），把两条路并排跑：

- **坍缩：** `(x_a@W1)@W2` 等于 `x_a@(W1@W2)` —— 打印 `W1@W2`（应该是 `[[7,2],[3,1]]`），
  证明两层直线层 = 一个矩阵。
- **打破：** 一个直线的叠加总是*可加的* —— `g(x_a) + g(x_b)` 正好等于 `g(x_a + x_b)`。**看着**
  带弯的那条路没通过这个检验：`f(x) = relu(x@W1)@W2` 给出 `f(x_a) + f(x_b) = [[4, 1]]`，
  而 `f(x_a + x_b) = [[0, 0]]`。既然任何单个矩阵都是可加的，就没有任何单个矩阵能做出这件事 ——
  那个弯真的给你买到了新东西。

用 `python3 sessions/m02-the-neuron/day-02-activations/experiment.py` 跑。
~~~

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.
~~~zh
#### 路线 B · 让 Claude 搭，然后你读
把下面的 prompt 复制回 Claude Code。它会让 Claude Code 创建文件、写代码，并替你跑一遍。
（prompt 本身保持英文，因为它是给工具看的指令，不是给你读的课文。）
~~~

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 2 artifact.

Create sessions/m02-the-neuron/day-02-activations/experiment.py that, with a comment on each step:
1. Defines step(z)=(z>=0).astype(float), relu(z)=np.maximum(0,z), leaky_relu(z, alpha=0.01)=np.where(z>0, z, alpha*z), sigmoid(z)=1/(1+np.exp(-z)), and tanh(z)=np.tanh(z); prints all five as a table over grid = np.linspace(-6,6,13) (step is a 0/1 jump printed as floats, ReLU zeros negatives, leaky_relu lets a small trickle through, sigmoid in (0,1), tanh in (-1,1) and 0 at 0). Also prints sigmoid'(0)=0.25 and notes ReLU's slope is 0 for negative inputs while leaky_relu's is the small alpha.
2. Shows the collapse: with W1=[[1,2],[0,1]], W2=[[1,0],[3,1]] and x_a=[[1.0, -3.0]], asserts np.allclose((x_a@W1)@W2, x_a@(W1@W2)) and prints W1@W2 (expect [[7,2],[3,1]]) — two linear layers equal one.
3. Then proves the bend breaks that, the honest way, with x_b=[[-1.0, 3.0]] so x_a+x_b is all zeros. For the straight path g(x)=x@W1@W2, check g(x_a)+g(x_b) == g(x_a+x_b) (a linear map always adds up). For the bent path f(x)=relu(x@W1)@W2, print f(x_a), f(x_b), f(x_a)+f(x_b) and f(x_a+x_b), and assert they are NOT equal (expect [[4,1]] vs [[0,0]]). Add a comment saying that because every linear map adds up, failing this additivity test proves no single matrix can reproduce f.
Keep the keyword name alpha (leaky_relu(z, alpha=0.01)) so the calls in the lesson work unchanged, and keep relu as np.maximum(0, z) so integer inputs print as integers.
Then run it and paste the output at the bottom as a comment.
%%%

~~~zh
#### 你应该看到什么（先坍缩，再解药）
- 你的五个弯在网格上打印出合理的值：step 是一个平的 0/1 跳跃，ReLU 把负数归零，`leaky_relu` 让一点
  细流通过，sigmoid 待在 `(0, 1)` 里，tanh 待在 `(−1, 1)` 里而且在 0 处是 0。
- 你打印出 `sigmoid'(0) ≈ 0.25`，并记下 ReLU 对负输入斜率是 `0`（死单元的种子），而 `leaky_relu`
  在那里保留一个小斜率 `alpha` —— 解药的种子。
- 第一个要**看着**的时刻：`(x_a@W1)@W2` 打印出和 `x_a@(W1@W2)` <em>一样</em>的数字 —— 两层直线层
  真的坍缩成了一层。
- 第二个要**看着**的时刻：直线那条路完美可加（`g(x_a) + g(x_b) = g(x_a + x_b) = [[0, 0]]`），而
  带弯的那条不可加（`[[4, 1]]` 对 `[[0, 0]]`）。任何单个矩阵都是可加的，所以这个不匹配就是那个弯
  压不平的证明。那次「打破」就是这个弯挣到了它的饭钱。
~~~

#### What you should see (the collapse, then the cure)
- Your five bends print sensible values over the grid: the step is a flat 0/1 jump, ReLU zeros negatives, `leaky_relu` lets a small trickle through, sigmoid stays inside `(0, 1)`, tanh stays inside `(−1, 1)` and is 0 at 0.
- You print `sigmoid'(0) ≈ 0.25` and note ReLU's slope is `0` for negative inputs (the seed of dead units) while `leaky_relu` keeps a small slope `alpha` there — the seed of the cure.
- The first moment to **watch**: `(x_a@W1)@W2` prints the <em>same</em> numbers as `x_a@(W1@W2)` — two straight-line layers really did collapse into one.
- The second moment to **watch**: the straight path adds up perfectly (`g(x_a) + g(x_b) = g(x_a + x_b) = [[0, 0]]`), while the bent path does not (`[[4, 1]]` versus `[[0, 0]]`). Every single matrix adds up, so that mismatch is proof the bend cannot be folded flat. That break is the bend earning its keep.

!!! c-info 📓
<b>5-minute research log:</b> before you close the tab, write three lines in `sessions/m02-the-neuron/day-02-activations/log.md`: (1) why a 100-layer network with no activations is no better than one layer; (2) ReLU vs sigmoid in one line, mentioning the vanishing gradient; (3) what a "dead" or "saturated" unit is, how you'd detect it, and the one-line fix.
!!!
~~~zh
!!! c-info 📓
<b>5 分钟研究笔记：</b>关掉页面前，用自己的话在
`sessions/m02-the-neuron/day-02-activations/log.md` 里写三行：(1) 为什么一个没有激活函数的 100 层
网络不比一层好；(2) 一句话说清 ReLU 和 sigmoid 的区别，要提到 vanishing gradient；(3) 什么是
「死掉的」或者「饱和的」单元，你会怎么发现它，以及那一行解药。
!!!
~~~

@@@ fin
