---
quest_id: wf3-d03-training-loop
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 6 — The Training Loop"
zh_page_title: "Module 2 · 第 6 天 —— 训练循环"
module_label: "Module 2 · Train · Day 6"
zh_module_label: "Module 2 · 训练 · 第 6 天"
title: "The Training Loop"
zh_title: "训练循环"
subtitle: "Where the Network Actually Learns"
zh_subtitle: "网络真正学会东西的地方"
brand_sub: "Foundations · M2 Day 6"
zh_brand_sub: "基础 · M2 第 6 天"
spine: "free-throw"
zh_spine: "罚球"
nav_prev_href: "../day-05-gradients-backprop/lesson.html"
nav_prev_label: "Gradients & Backpropagation"
zh_nav_prev_label: "梯度与 backpropagation（反向传播）"
nav_next_href: "../day-07-optimizers/lesson.html"
nav_next_label: "Optimizers (SGD, Momentum, Adam)"
zh_nav_next_label: "Optimizer（优化器）：SGD、Momentum、Adam"
fin_title: "Module 2 · Day 6 complete! 🏆"
zh_fin_title: "Module 2 · 第 6 天完成！🏆"
fin_body: "Nice work — you've assembled the <b>training loop</b>: forward pass → loss → backward pass → update, repeated over batches and epochs until the loss falls. That loop is where a network actually learns.<br>Next up: <b>Optimizers</b> — smarter ways to take each downhill step."
zh_fin_body: "干得好 —— 你把 <b>training loop（训练循环）</b>拼起来了：forward pass（前向传播）→ loss（损失）→ backward pass（反向传播）→ update（更新），一圈一圈重复，直到 loss 降下来。网络就是在这个循环里真正学会东西的。<br>下一站：<b>Optimizer（优化器）</b> —— 更聪明的下坡走法。"
notebook_yardstick: 00-neural-networks/fundamentals/08_training_loop.ipynb
coverage_topics:
  - {topic: the four-phase loop, keywords: [four steps, forward pass, loss, backward pass, update, over and over]}
  - {topic: forward pass phase, keywords: [make a guess, weighted sum, prediction]}
  - {topic: loss phase, keywords: [how wrong, one number, loss]}
  - {topic: backward pass phase, keywords: [gradient, which way, backpropagation, blame]}
  - {topic: update phase, keywords: [the update, nudge the aim, shift each weight, new weight, small step in the gradient]}
  - {topic: minus sign goes downhill, keywords: [minus sign, opposite way, subtract]}
  - {topic: learning rate step size, keywords: [learning rate, step size, how big a step]}
  - {topic: epoch vs iteration, keywords: [epoch, one full pass, iteration, one step, one update]}
  - {topic: loss curve, keywords: [loss curve, plot the loss, goes down, watch it fall]}
  - {topic: convergence, keywords: [convergence, flattens out, stops improving, levels off]}
  - {topic: perceptron ancestor, keywords: [perceptron, rosenblatt, only on mistakes, fixed rule, crude ancestor]}
  - {topic: lr too high failure, keywords: [too high, overshoot, diverges, blows up, nan, oscillates]}
  - {topic: lr scheduling decay remedy, keywords: [learning-rate schedule, decay, shrink the learning rate, smaller steps later]}
  - {topic: lr too low failure, keywords: [too low, crawls, barely moves, too tiny]}
  - {topic: zero grad failure, keywords: [reset the gradients, gradients pile up, accumulate, forget to reset]}
  - {topic: zero grad remedy, keywords: [zero the gradients, reset the gradients each step, clear the gradients]}
  - {topic: shuffle failure, keywords: [same fixed order, fixed order, biased by that order, never shuffling]}
  - {topic: shuffle remedy, keywords: [shuffle the data, reshuffle each epoch, mix up the order]}
  - {topic: too few epochs underfit, keywords: [too few epochs, stopping too early, still high, underfit, train longer]}
  - {topic: overfitting from over-training, keywords: [memorizes, overfitting, train too long, validation loss rises]}
  - {topic: early stopping remedy, keywords: [early stopping, stop when validation stops improving, held-out]}
  - {topic: capability limit xor, keywords: [xor, non-linearly separable, more iterations never, plateaus above zero, single neuron cannot]}
  - {topic: local minimum plateau, keywords: [local minimum, plateau, non-convex, not the global best]}
---

@@@ hero
@lede Think about learning to sink a **free-throw**. You step up, you shoot — clank, off the left of the rim. So you aim a hair to the right and shoot again. A little closer. Clank, adjust, shoot. Nobody sinks it clean on the first try. You get good by *repeating one small fix, over and over*, until the ball drops through. That rhythm — *try, measure the miss, nudge, repeat* — is the whole secret of how a neural network learns. It has a name: the **training loop**. It is the engine humming underneath every model you've heard of: the tiny neuron you'll train today, the face-unlock on a phone, the thing that picks your next video, all the way up to ChatGPT. You already built each separate piece — the guess, the loss that scores the miss, the gradient that says which way to nudge. Today they click together into one repeating cycle. Press go, and watch one number fall, shot after shot, until the network can sink it.
@goal You'll snap the pieces into one loop: **forward** (take the shot) → **loss** (measure the miss) → **backward** (which way was I off?) → **update** (nudge the aim), then go again. You'll meet the **learning rate** — how bold each nudge is — and count practice in **epochs** and **iterations**. You'll read a falling **loss curve** the way an engineer does. Then we'll poke the sneaky ways practice goes wrong, each a small puzzle with a one-line fix. Plain words come first every time; any heavy math sits in a skippable box.
%%% warmup
q: From yesterday: the gradient points which way, and which way do we step to learn? | a:1 | It points downhill, and we step the same way | It points uphill (toward more loss), and we step the OPPOSITE way, downhill | It points sideways, and we don't move | It points at the answer, and we jump straight there | concept: gradient-downhill | fb: The gradient points UPHILL toward more loss, so we step the opposite way — downhill toward less loss. That minus sign is exactly today's update.
q: From yesterday: what does backpropagation do in one backward sweep? | a:2 | Redoes the whole forward pass once per weight | Picks the biggest weight and deletes it | Multiplies local slopes backward to hand EVERY weight its gradient at once | Guesses each weight's blame with a hunch | concept: backprop-sweep | fb: Backprop starts at the loss and sweeps backward, multiplying local slopes, so every weight gets its gradient in ONE pass. Today that gradient is what the update actually uses.
q: From day 4: what is the loss, in one line? | a:0 | One number saying how wrong the guess is; lower is better | The number of layers in the network | The speed of training | A random starting weight | concept: loss-signal | fb: The loss is a single number scoring how wrong the guess is — lower is better, 0 is perfect. Today we watch it fall, loop after loop.
%%%
@zh_lede 想一想你是怎么学会投**罚球**的。你走到线上，出手 —— 当，砸在篮框左边。于是你把准星往右挪一点点，再投一次。近了一些。当，再调，再投。没有人第一次就能空心命中。你是靠*把一个小小的修正重复无数次*，才让球落进网里的。这个节奏 —— *试一次、量一下偏了多少、挪一点、再来* —— 就是 neural network（神经网络）学习的全部秘密。它有个名字：**training loop（训练循环）**。你听说过的每个模型底下都是它在转：今天你要训练的那个小 neuron，手机的刷脸解锁，给你推下一个视频的东西，一直到 ChatGPT。零件你已经一个一个造好了 —— 那个猜测，给偏差打分的 loss，还有告诉你该往哪边挪的 gradient。今天它们咔一声合成一个不停重复的圈。按下开始，看着一个数字一投一投地往下掉，直到网络能把球投进去。
@zh_goal 你会把零件扣进一个循环：**forward（出手）** → **loss（量偏差）** → **backward（我往哪边偏了？）** → **update（挪准星）**，然后再来一圈。你会认识 **learning rate（学习率）** —— 每次挪多大胆，还会用 **epoch（轮次）** 和 **iteration（迭代）** 来数练了多少。你会像工程师一样读一条往下掉的 **loss curve（损失曲线）**。然后我们去戳一戳练习出岔子的几种阴招，每个都是一个小谜题，配一行就能改好的答案。每次都先讲人话；重一点的数学都放在可以跳过的框里。
~~~zh
%%% warmup
q: 接昨天：gradient 指向哪边，我们又该往哪边走才能学到东西？ | a:1 | 它指下坡，我们就往同一边走 | 它指上坡（loss 更大的方向），我们要往**反**方向走，也就是下坡 | 它指旁边，我们不动 | 它直接指着答案，我们一步跳过去 | concept: gradient-downhill | fb: gradient 指的是**上坡**，也就是 loss 更大的方向，所以我们往反方向走 —— 下坡，走向更小的 loss。那个减号就是今天的 update。
q: 接昨天：backpropagation 在一次反向扫描里做了什么？ | a:2 | 每个 weight 都把整个前向过程重做一遍 | 挑出最大的那个 weight 删掉 | 把局部斜率一路往后乘，一次就把 gradient 发给**每一个** weight | 凭感觉猜每个 weight 该背多少责任 | fb: backpropagation 从 loss 出发往后扫，一路乘局部斜率，所以**一次**就让每个 weight 拿到自己的 gradient。今天的 update 用的就是这个 gradient。
q: 接第 4 天：一句话说清 loss 是什么？ | a:0 | 一个数字，说这次猜得有多错；越小越好 | 网络有多少层 | 训练有多快 | 一个随机的起始 weight | fb: loss 就是给「猜得有多错」打分的**一个**数字 —— 越小越好，0 就是完美。今天我们看着它一圈一圈往下掉。
%%%
~~~

@@@ concept id=c1 zh_tag="练习的循环" tag="The practice loop" zh_title="training loop（训练循环）—— 四步，一圈一圈重复" title="The training loop — four steps, on repeat" zh_gotit="看懂这个循环了" gotit="Got the loop"
Let's stay on the basketball court, because you already know this loop by heart. Picture one **free-throw** practice cycle. **(1)** You shoot. **(2)** You *look* at where it landed — a foot left of the rim. **(3)** You work out *which way* you were off: "I leaned left." **(4)** You *nudge* your aim a little to the right. Then you do the whole thing again. Four little steps, over and over — and every single one is something you already built this module.
~~~zh
我们还留在篮球场上，因为这个循环你早就会了。想一想一次**罚球**练习。**(1)** 你出手。**(2)** 你*看*球落在哪 —— 篮框左边一尺。**(3)** 你想清楚自己*往哪边*偏了：「我往左倾了。」**(4)** 你把准星往右*挪一点*。然后整套再来一遍。四个小步骤，一遍一遍 —— 而每一步都是你这个 module 里已经造好的东西。
~~~

%%% svg
<svg viewBox="0 0 520 232" role="img" aria-label="A basketball free-throw practice loop drawn as four steps arranged in a circle: shoot the ball, see where it landed, figure out which way you were off, nudge your aim. Arrows connect them in a ring, showing the cycle repeats."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One free-throw practice cycle — repeated until it drops clean</text><text class="lang-zh" x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">一次罚球练习 —— 一直重复到球干净落网</text><circle cx="140" cy="70" r="40" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="140" y="64" text-anchor="middle" fill="#1F6280">🏀 shoot</text><text class="lang-zh" x="140" y="64" text-anchor="middle" fill="#1F6280">🏀 出手</text><text class="lang-en" x="140" y="78" text-anchor="middle" fill="#1F6280" font-size="9">(take the shot)</text><text class="lang-zh" x="140" y="78" text-anchor="middle" fill="#1F6280" font-size="9">（投一次）</text><circle cx="380" cy="70" r="40" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="380" y="64" text-anchor="middle" fill="#C93B3B">📏 measure</text><text class="lang-zh" x="380" y="64" text-anchor="middle" fill="#C93B3B">📏 量一下</text><text class="lang-en" x="380" y="78" text-anchor="middle" fill="#C93B3B" font-size="9">(how far off?)</text><text class="lang-zh" x="380" y="78" text-anchor="middle" fill="#C93B3B" font-size="9">（偏了多远？）</text><circle cx="380" cy="180" r="40" fill="#EDE9F8" stroke="#7C6DAA"/><text class="lang-en" x="380" y="174" text-anchor="middle" fill="#5E5191">🧭 which way?</text><text class="lang-zh" x="380" y="174" text-anchor="middle" fill="#5E5191">🧭 往哪边偏？</text><text class="lang-en" x="380" y="188" text-anchor="middle" fill="#5E5191" font-size="9">(I leaned left)</text><text class="lang-zh" x="380" y="188" text-anchor="middle" fill="#5E5191" font-size="9">（我往左倾了）</text><circle cx="140" cy="180" r="40" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="140" y="174" text-anchor="middle" fill="#1a5c38">🎯 nudge aim</text><text class="lang-zh" x="140" y="174" text-anchor="middle" fill="#1a5c38">🎯 挪准星</text><text class="lang-en" x="140" y="188" text-anchor="middle" fill="#1a5c38" font-size="9">(a little right)</text><text class="lang-zh" x="140" y="188" text-anchor="middle" fill="#1a5c38" font-size="9">（往右一点）</text><g stroke="#B8AEA2" stroke-width="2" fill="#B8AEA2"><path d="M180 62 l160 0"/><polygon points="340,62 331,57 331,67"/><path d="M380 110 l0 30"/><polygon points="380,140 375,131 385,131"/><path d="M340 188 l-160 0"/><polygon points="180,188 189,183 189,193"/><path d="M140 140 l0 -30"/><polygon points="140,110 135,119 145,119"/></g><text class="lang-en" x="260" y="130" text-anchor="middle" fill="#6B645E" font-size="9">round and round — each loop, a little closer</text><text class="lang-zh" x="260" y="130" text-anchor="middle" fill="#6B645E" font-size="9">一圈又一圈 —— 每一圈都近一点</text></g></svg>
%%%

**What the picture gets right:** you improve by *repeating* one small correction — no single shot fixes everything, but hundreds of tiny fixes add up.

**Where it breaks down:** a player fixes one thing at a time (aim, then arc, then power). A network nudges *every* knob at once, on every lap.

#### Now the same four steps, in network words
Same rhythm, new vocabulary. Read one rung at a time — the left side is the move, the right side is why it's there.

%%% steps
step: 1 · the shot — the **forward pass**
why: run an input through the neuron to **make a guess**. It's the **weighted sum** from day 1 plus its bend from day 2 — and the guess it spits out is called the **prediction**.
step: 2 · the miss — the **loss**
why: **one number** saying **how wrong** that guess is. Lower is better, 0 is perfect. This is the "ball landed a foot left" measurement.
step: 3 · the blame — the **backward pass**
why: **backpropagation** sweeps backward from the loss and hands every weight its **gradient** — **which way** that weight would have to move to make the miss bigger. That's the "I leaned left" **blame** report.
step: 4 · the nudge — **the update**
why: **shift each weight** a small step **against** the gradient — the direction that *lowers* the loss. This is the only step that actually changes the network.
step: ↻ then jump back to step 1
why: one lap barely helps. Hundreds of laps is what learning *is*.
%%%

%%% insight
Here's the part that surprises everyone: nothing in this loop is clever. There is no "understanding" step, no moment where the network figures anything out. It just repeats four boring moves — guess, measure, blame, nudge — a few hundred thousand times. That stubborn repetition, and nothing else, is what turns a pile of random numbers into something that can finish your sentences.
%%%

So far: four steps, in a ring, forever. Now here's the same lap drawn flat, so you can watch one lap feed straight into the next.
~~~zh
**这个比喻对的地方：**你是靠*重复*一个小修正变好的 —— 没有哪一投能一次修好全部，但几百次小修正会累起来。

**它不对的地方：**球员一次只修一件事：先准星，再弧线，再力气。网络每一圈把*每一个*旋钮都挪一下。

#### 现在把同样的四步换成网络的说法
节奏一样，词换了。一格一格读 —— 左边是动作，右边是它为什么在这儿。

%%% steps
step: 1 · 出手 —— **forward pass（前向传播）**
why: 把一个输入送过 neuron，**做出一个猜测**。就是第 1 天的**加权和**，加上第 2 天给它的那个弯 —— 它吐出来的那个猜测叫 **prediction（预测值）**。
step: 2 · 偏差 —— **loss（损失）**
why: **一个数字**，说那个猜测**有多错**。越小越好，0 就是完美。这就是「球落在左边一尺」这个测量。
step: 3 · 追责 —— **backward pass（反向传播）**
why: **backpropagation** 从 loss 往后扫，把 **gradient（梯度）** 发给每一个 weight —— 也就是这个 weight 要往**哪边**动才会让偏差变得更大。这就是那份「我往左倾了」的**追责**报告。
step: 4 · 挪一下 —— **update（更新）**
why: 让每个 weight **逆着** gradient 挪一小步 —— 那个方向会让 loss *变小*。这是整个循环里唯一真正改动网络的一步。
step: ↻ 然后跳回第 1 步
why: 一圈几乎没用。几百圈才叫学习。
%%%

%% insight
这一段是最让人意外的：这个循环里没有一处是聪明的。没有「理解」这一步，也没有哪个瞬间网络想明白了什么。它只是把四个无聊的动作 —— 猜、量、追责、挪 —— 重复几十万次。就是这份笨笨的重复，别的什么都没有，把一堆随机数字变成了能帮你把句子写完的东西。
%%

到这里：四个步骤，串成一圈，一直转。现在把同一圈画平，你就能看到一圈怎么直接喂给下一圈。
~~~

%%% svg
<svg viewBox="0 0 520 168" role="img" aria-label="The four phases of the training loop drawn as a horizontal conveyor: forward pass gives a prediction, loss measures how wrong, backward pass gives gradients, update nudges the weights. A curved arrow loops from update back to forward, labelled repeat."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One lap of the training loop — then it repeats</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">training loop 的一圈 —— 然后重复</text><rect x="14" y="44" width="108" height="34" rx="5" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="68" y="59" text-anchor="middle" fill="#1F6280" font-size="9.5">1 · forward pass</text><text class="lang-zh" x="68" y="59" text-anchor="middle" fill="#1F6280" font-size="9.5">1 · forward pass</text><text class="lang-en" x="68" y="71" text-anchor="middle" fill="#1F6280" font-size="9">→ prediction</text><text class="lang-zh" x="68" y="71" text-anchor="middle" fill="#1F6280" font-size="9">→ prediction</text><text x="128" y="63" fill="#B8AEA2">→</text><rect x="142" y="44" width="96" height="34" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="190" y="59" text-anchor="middle" fill="#C93B3B" font-size="9.5">2 · loss</text><text class="lang-zh" x="190" y="59" text-anchor="middle" fill="#C93B3B" font-size="9.5">2 · loss</text><text class="lang-en" x="190" y="71" text-anchor="middle" fill="#C93B3B" font-size="9">→ how wrong</text><text class="lang-zh" x="190" y="71" text-anchor="middle" fill="#C93B3B" font-size="9">→ 有多错</text><text x="244" y="63" fill="#B8AEA2">→</text><rect x="258" y="44" width="112" height="34" rx="5" fill="#EDE9F8" stroke="#7C6DAA"/><text class="lang-en" x="314" y="59" text-anchor="middle" fill="#5E5191" font-size="9.5">3 · backward pass</text><text class="lang-zh" x="314" y="59" text-anchor="middle" fill="#5E5191" font-size="9.5">3 · backward pass</text><text class="lang-en" x="314" y="71" text-anchor="middle" fill="#5E5191" font-size="9">→ gradients</text><text class="lang-zh" x="314" y="71" text-anchor="middle" fill="#5E5191" font-size="9">→ gradient</text><text x="376" y="63" fill="#B8AEA2">→</text><rect x="390" y="44" width="112" height="34" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="446" y="59" text-anchor="middle" fill="#1a5c38" font-size="9.5">4 · update</text><text class="lang-zh" x="446" y="59" text-anchor="middle" fill="#1a5c38" font-size="9.5">4 · update</text><text class="lang-en" x="446" y="71" text-anchor="middle" fill="#1a5c38" font-size="9">→ nudge weights</text><text class="lang-zh" x="446" y="71" text-anchor="middle" fill="#1a5c38" font-size="9">→ 挪 weight</text><path d="M446 82 C 446 130, 68 130, 68 84" fill="none" stroke="#C99A12" stroke-width="2" stroke-dasharray="4,3"/><polygon points="68,84 63,94 73,94" fill="#C99A12"/><text class="lang-en" x="260" y="126" text-anchor="middle" fill="#9A7208" font-size="10">↻ repeat — every lap, the loss drops a little</text><text class="lang-zh" x="260" y="126" text-anchor="middle" fill="#9A7208" font-size="10">↻ 重复 —— 每一圈 loss 都掉一点</text></g></svg>
%%%

The one line to keep: **forward → loss → backward → update, on repeat.**

Steps 1–3 you already own. Step 4 is today's new magic — the single moment the network *changes*. Let's zoom in.
~~~zh
要记住的一句话：**forward → loss → backward → update，一圈一圈重复。**

第 1 到 3 步你已经会了。第 4 步是今天的新魔法 —— 网络*发生改变*的那唯一一个瞬间。我们放大看看。
~~~

@@@ concept id=c2 zh_tag="把准星挪一点" tag="Nudge the aim" zh_title="update（更新）—— 唯一真正改动 weight 的一步" title="The update — the one step that changes the weights" zh_gotit="看懂 update 了" gotit="Got the update"
This is the heart of the loop, so let's slow right down. Your last **free-throw** clanked off the *left* of the rim. Your body already knows the fix: aim a little to the *right*. Not all the way across the gym — you'd overcorrect and miss right — just a *small nudge*, the **opposite way** from the miss. The network does exactly this with each weight, and it never has to guess: yesterday's gradient already told it "pushing this weight up makes the miss *worse*."
~~~zh
这是整个循环的心脏，所以我们慢下来。你上一次**罚球**砸在篮框*左*边。你的身体已经知道该怎么改：准星往*右*挪一点。不是挪到球场那头 —— 那样会改过头，偏到右边去 —— 只是*挪一小点*，朝着偏差的**反方向**。网络对每个 weight 做的就是这件事，而且它从来不用猜：昨天的 gradient 已经告诉它「把这个 weight 往上推会让偏差*更糟*」。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A basketball hoop seen from above. The last shot landed left of the rim, marked with an X. A red arrow shows the miss pointing left. A green arrow shows the aim being nudged the opposite way, a small step to the right, toward the center of the rim."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Miss went left → nudge the aim the OPPOSITE way, a small step right</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">球偏左 → 把准星往反方向挪一小步，向右</text><circle cx="300" cy="96" r="46" fill="none" stroke="#C99A12" stroke-width="3"/><text class="lang-en" x="300" y="152" text-anchor="middle" fill="#9A7208" font-size="9">the rim (target)</text><text class="lang-zh" x="300" y="152" text-anchor="middle" fill="#9A7208" font-size="9">篮框（目标）</text><text x="188" y="100" text-anchor="middle" fill="#C93B3B" font-size="16">✕</text><text class="lang-en" x="188" y="120" text-anchor="middle" fill="#C93B3B" font-size="9">last shot landed here</text><text class="lang-zh" x="188" y="120" text-anchor="middle" fill="#C93B3B" font-size="9">上一投落在这</text><path d="M270 96 l-70 0" stroke="#C93B3B" stroke-width="2.5"/><polygon points="200,96 210,91 210,101" fill="#C93B3B"/><text class="lang-en" x="235" y="86" text-anchor="middle" fill="#C93B3B" font-size="9">the miss →</text><text class="lang-zh" x="235" y="86" text-anchor="middle" fill="#C93B3B" font-size="9">偏差 →</text><path d="M300 70 l44 -20" stroke="#2D8B55" stroke-width="2.5"/><polygon points="344,50 333,50 338,60" fill="#2D8B55"/><text class="lang-en" x="392" y="50" text-anchor="middle" fill="#2D8B55" font-size="9">nudge aim: small step</text><text class="lang-zh" x="392" y="50" text-anchor="middle" fill="#2D8B55" font-size="9">挪准星：一小步</text><text class="lang-en" x="392" y="63" text-anchor="middle" fill="#2D8B55" font-size="9">the opposite way</text><text class="lang-zh" x="392" y="63" text-anchor="middle" fill="#2D8B55" font-size="9">反方向</text></g></svg>
%%%

**What the picture gets right:** the correction is *small* and points *opposite* to the miss.

**Where it breaks down:** a shooter *feels* the fix. The network doesn't feel anything — the gradient handed it the exact opposite direction, for every weight at once.

#### Building the update rule, one piece at a time
Five short rungs and you'll have the whole thing. Notice we're just writing down what your arm already did.

%%% steps
step: start from the aim you have now — the old weight
why: nobody starts over from scratch each shot. You keep your stance and adjust it.
step: ask the gradient which way is *worse*
why: the gradient points **uphill**, toward MORE loss. It's the direction of the miss.
step: turn around — **subtract** it
why: that little **minus sign** is the whole trick. Going the **opposite way** from "worse" means going toward *better*, downhill.
step: take only a *fraction* of that direction — multiply it by the **learning rate**
why: a full-size jump would fly past the rim. The learning rate decides **how big a step** you take for a given slope; you'll meet it properly in concept 4, right after we say hello to the loop's 1958 ancestor.
step: that gives you the **new weight**, and every weight in the network gets the same treatment on the same lap
why: this is **the update**. Do it to `w`, do it to `b`, do it to all million of them — one lap, one nudge each.
%%%

Put those five rungs on one line and you get the rule people actually write down:

%%% formula
expr: w ← w − lr × (gradient of the loss for w)
note: "w" is a weight, "lr" is the learning rate (also called the step size). The MINUS sign walks downhill — opposite the uphill gradient. The bias updates the exact same way: b ← b − lr × (its gradient).
%%%

%%% insight
Why *small* steps, when we know the direction? Because the gradient is only honest *right where you stand*. It's the slope under your feet, not a map of the whole hill. Trust it for a short step and you go down; trust it for a giant leap and you'll land somewhere it never promised anything about. That single fact explains almost every training disaster in this lesson.
%%%

So far: subtract a small piece of the gradient. Now let's watch a real weight do it. It starts at `0.5` and its ideal home is `3.0` — like standing one big step left of the rim.

%%% demo id=update label="run it — one weight, five nudges"
predict: With a learning rate of 0.25, does the weight jump straight to 3.0 on the first nudge, creep up in shrinking hops, or overshoot past it? Pick one, then reveal.
code: w=0.5; target=3.0; lr=0.25
code: for i in range(5): grad = 2*(w-target); w = w - lr*grad; print(round(w,3))
out: 1.75
out: 2.375
out: 2.688
out: 2.844
out: 2.922
take: <b>Shrinking hops.</b> 0.5 → 1.75 → 2.375 → 2.69 → 2.84 → 2.92. Each nudge closes a chunk of the remaining gap, so as the gap shrinks the gradient shrinks — and the hops shrink with it, all on their own. That patient repeated nudge IS the free-throw drill. Run it longer and it settles right on 3.0.
%%%

And here is that same climb as bars, so you can *see* the gap closing instead of reading numbers.
~~~zh
**这个比喻对的地方：**修正是*小*的，而且指向偏差的*反*方向。

**它不对的地方：**投手是*感觉*到该怎么改的。网络什么都感觉不到 —— 是 gradient 把准确的反方向递给了它，而且是所有 weight 一起。

#### 一块一块把 update 规则搭起来
五个短格子，你就拿到完整的规则了。注意我们只是把你的手臂刚才做的事写下来。

%%% steps
step: 从你现在的准星开始 —— 也就是旧的 weight
why: 没有人每投一次都从头开始。你保留自己的站姿，然后调整它。
step: 问 gradient 哪边是*更糟*的
why: gradient 指向**上坡**，也就是 loss 更大的方向。它是偏差的方向。
step: 转过身 —— **减掉**它
why: 那个小小的**减号**就是全部的门道。走「更糟」的**反方向**，就是走向更好，也就是下坡。
step: 只取那个方向的一*小部分* —— 乘上 **learning rate（学习率）**
why: 一整步跳出去会飞过篮框。learning rate 决定给定一个斜率时你**跨多大一步**；我们先去跟这个循环 1958 年的祖先打个招呼，第 4 个 concept 再正式认识它。
step: 这就得到**新的 weight**，而且网络里每个 weight 在同一圈里都这么处理一次
why: 这就是 **update**。对 `w` 做一次，对 `b` 做一次，对那一百万个全做一次 —— 一圈，每个挪一下。
%%%

把这五格写成一行，就得到大家真正写下来的那条规则：

%%% formula
expr: w ← w − lr × (loss 对 w 的 gradient)
note: 「w」是一个 weight，「lr」是 learning rate（也叫 step size，步长）。那个减号让你往下坡走 —— 和指上坡的 gradient 反着来。bias 的更新方式一模一样：b ← b − lr ×（它自己的 gradient）。
%%%

%% insight
方向都知道了，为什么还只走*小*步？因为 gradient 只在*你站的那一点*上是诚实的。它是你脚底下的斜率，不是整座山的地图。走一小步就信它，你会往下走；一大步就信它，你会落到它从没许诺过任何事的地方。这一个事实几乎能解释这节课里所有的训练灾难。
%%

到这里：减掉 gradient 的一小块。现在看一个真的 weight 这么做。它从 `0.5` 出发，理想的家在 `3.0` —— 就像站在篮框左边一大步的地方。

%%% demo
predict: learning rate 是 0.25 的时候，这个 weight 第一次挪就直接跳到 3.0，还是一小步一小步越挪越小地爬上去，还是冲过头？先选一个，再揭晓。
code: w=0.5; target=3.0; lr=0.25
code: for i in range(5): grad = 2*(w-target); w = w - lr*grad; print(round(w,3))
out: 1.75
out: 2.375
out: 2.688
out: 2.844
out: 2.922
take: <b>越挪越小的小步。</b>0.5 → 1.75 → 2.375 → 2.69 → 2.84 → 2.92。每次挪都吃掉剩下差距的一块，所以差距变小，gradient 就变小 —— 每一步也就自己跟着变小。这份耐心的重复挪动**就是**那个罚球练习。跑久一点，它会稳稳停在 3.0 上。
%%%

同样这段爬升再画成柱子，这样你能*看见*差距在合上，而不是读数字。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A bar chart of one weight climbing over five loops from 0.5 toward its target of 3.0. Bars rise 0.5, 1.75, 2.375, 2.69, 2.84, 2.92, each closer to a dashed line at 3.0. The gap between the bar top and the target shrinks every loop."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One weight nudging toward its target (3.0) — gap shrinks each loop</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">一个 weight 挪向它的目标（3.0）—— 每圈差距变小</text><line x1="46" y1="140" x2="500" y2="140" stroke="#E5DFD6" stroke-width="1.5"/><line x1="46" y1="42" x2="500" y2="42" stroke="#2D8B55" stroke-width="1.3" stroke-dasharray="5,3"/><text class="lang-en" x="502" y="45" fill="#2D8B55" font-size="9">target 3.0</text><text class="lang-zh" x="502" y="45" fill="#2D8B55" font-size="9">目标 3.0</text><g fill="#2A7B9B"><rect x="60" y="123" width="44" height="17"/><rect x="132" y="83" width="44" height="57"/><rect x="204" y="63" width="44" height="77"/><rect x="276" y="53" width="44" height="87"/><rect x="348" y="48" width="44" height="92"/><rect x="420" y="45" width="44" height="95"/></g><g fill="#1F6280" text-anchor="middle"><text x="82" y="118">0.5</text><text x="154" y="78">1.75</text><text x="226" y="58">2.38</text><text x="298" y="48">2.69</text><text x="370" y="43">2.84</text><text x="442" y="40">2.92</text></g><g fill="#6B645E" text-anchor="middle" font-size="9"><text class="lang-en" x="82" y="154">start</text><text class="lang-zh" x="82" y="154">起点</text><text class="lang-en" x="154" y="154">loop 1</text><text class="lang-zh" x="154" y="154">第 1 圈</text><text class="lang-en" x="226" y="154">loop 2</text><text class="lang-zh" x="226" y="154">第 2 圈</text><text class="lang-en" x="298" y="154">loop 3</text><text class="lang-zh" x="298" y="154">第 3 圈</text><text class="lang-en" x="370" y="154">loop 4</text><text class="lang-zh" x="370" y="154">第 4 圈</text><text class="lang-en" x="442" y="154">loop 5</text><text class="lang-zh" x="442" y="154">第 5 圈</text></g></g></svg>
%%%

That one move — subtract a small multiple of the gradient — has a name: **gradient descent**. It is Step 4 of every loop ever run.

It is also the great-grandchild of a much cruder idea from 1958. Meeting the ancestor makes today's version feel almost luxurious.
~~~zh
这一个动作 —— 减掉 gradient 的一个小倍数 —— 有个名字：**gradient descent（梯度下降）**。它是每一个循环的第 4 步。

它也是 1958 年一个粗糙得多的想法的曾孙。见过祖先之后，今天这个版本会显得相当奢侈。
~~~

@@@ concept id=c3 zh_tag="老办法" tag="The old way" zh_title="perceptron（感知机）—— 这个循环粗糙的祖先" title="The perceptron — the crude ancestor of the loop" zh_gotit="看懂这个祖先了" gotit="Got the ancestor"
Imagine an old-school **free-throw** coach with exactly one rule. While you're sinking shots he says *nothing* — no praise, no tips. The instant you *miss*, he blows a whistle and shoves your elbow — the same shove no matter **how badly** you missed. Airball off the backboard? One shove. Rattled out by a fingernail? The same shove. That coach is a real machine: the [[perceptron||Frank Rosenblatt's 1958 learning machine: it corrected the weights only when it got an example wrong, and its correction carried no measure of how wrong]], built by Frank **Rosenblatt** in 1958 — the first thing anyone ever called a learning machine, and the crude ancestor of everything you did in the last concept.
~~~zh
想象一个老派的**罚球**教练，他只有一条规则。你投进的时候他*什么都不说* —— 不夸你，也不给建议。你一*没投中*，他就吹哨，然后推你的胳膊肘 —— 不管你偏得**多厉害**，都是同一下推。球砸在篮板上飞了？推一下。被框沿刮了一指甲盖弹出来？同样推一下。这个教练是一台真实存在的机器：[[perceptron（感知机）||Frank Rosenblatt 在 1958 年造的学习机器：只有在做错的时候才修 weight，而且它的修正里不带「错了多少」这个信息]]，由 Frank **Rosenblatt** 在 1958 年造出来，是人类第一次称之为学习机器的东西，也是你在上一个 concept 里做的一切的粗糙祖先。
~~~

%%% svg
<svg viewBox="0 0 520 184" role="img" aria-label="An old-fashioned coach with a whistle. On the left, a made shot: the coach is silent, no correction. On the right, a missed shot: the coach blows the whistle and gives an elbow shove whose size carries no information about how big the miss was."><g font-family="monospace" font-size="10.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The perceptron coach: silent on hits, one blind shove on any miss</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">perceptron 教练：投进就不说话，没投中就来一下没脑子的推</text><line x1="260" y1="30" x2="260" y2="170" stroke="#E5DFD6" stroke-dasharray="4,3"/><rect x="20" y="34" width="222" height="130" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="131" y="54" text-anchor="middle" fill="#1a5c38">shot was GOOD</text><text class="lang-zh" x="131" y="54" text-anchor="middle" fill="#1a5c38">这一投投进了</text><text x="131" y="92" text-anchor="middle" font-size="24">🏀 ✓</text><text class="lang-en" x="131" y="126" text-anchor="middle" fill="#1a5c38" font-size="9.5">coach: silent 🤐</text><text class="lang-zh" x="131" y="126" text-anchor="middle" fill="#1a5c38" font-size="9.5">教练：不说话 🤐</text><text class="lang-en" x="131" y="144" text-anchor="middle" fill="#6B645E" font-size="9">no nudge at all</text><text class="lang-zh" x="131" y="144" text-anchor="middle" fill="#6B645E" font-size="9">完全不挪</text><rect x="278" y="34" width="222" height="130" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="389" y="54" text-anchor="middle" fill="#C93B3B">shot MISSED</text><text class="lang-zh" x="389" y="54" text-anchor="middle" fill="#C93B3B">这一投没中</text><text x="389" y="92" text-anchor="middle" font-size="24">🏀 ✕</text><text class="lang-en" x="389" y="126" text-anchor="middle" fill="#C93B3B" font-size="9.5">whistle 📣 + a shove</text><text class="lang-zh" x="389" y="126" text-anchor="middle" fill="#C93B3B" font-size="9.5">哨子 📣 加一下推</text><text class="lang-en" x="389" y="144" text-anchor="middle" fill="#6B645E" font-size="9">no measure of HOW wrong</text><text class="lang-zh" x="389" y="144" text-anchor="middle" fill="#6B645E" font-size="9">不衡量错了多少</text></g></svg>
%%%

**What the picture gets right:** it reacts **only on mistakes**, and the correction carries no information about the size of the miss — a **fixed rule**, not a measured one.

**Where it breaks down:** a real coach can at least *see* how badly you missed, even if he ignores it. The perceptron had no "how wrong" number at all — no loss, nothing to plot.

#### Two rough edges — and the upgrade that fixed both
This is a short story with a happy ending, and it explains why today's loop looks the way it does.

%%% steps
step: rough edge 1 — silence on the shots that barely squeaked in
why: a ball that rattled around the rim and *dropped* counted as a hit, so it got zero feedback. Therefore the barely-right cases never got polished, and the machine stopped improving the moment it was merely correct.
step: rough edge 2 — the shove ignores how wrong you were
why: a hair-thin miss and a wild airball earn the same-size correction, because the perceptron has no number for **how wrong** it was. That means the machine could not aim its effort where the error actually was.
step: the upgrade — a smooth loss, and its gradient
why: today's loop scores *every* example with a number, then reads the slope of that number. Which means a big miss earns a big nudge, a tiny miss earns a tiny one, and even a barely-right shot gets polished — a **graded** correction for everyone, on every lap.
step: the bonus you get for free
why: because there's now a number, you can *plot* it. The perceptron had nothing to watch; you get a curve that tells you at a glance whether the run is healthy.
%%%

%%% insight
Notice the shape of that story, because it repeats through all of machine learning: someone replaces a hard yes/no rule with a smooth number you can measure, and suddenly you can take derivatives, take small steps, and watch progress. "Make it smooth so you can nudge it" is arguably the single most reused idea in the field.
%%%

So far: a blind shove versus a graded nudge. Here's what that difference looks like across six practice shots.
~~~zh
**这个比喻对的地方：**它**只在做错的时候**才反应，而且这个修正不带任何关于偏差大小的信息 —— 是一条**固定的规则**，不是量出来的。

**它不对的地方：**真的教练至少*看得见*你偏了多少，就算他不理。perceptron 连一个「有多错」的数字都没有 —— 没有 loss，什么都画不出来。

#### 两处毛边 —— 以及一次把两个都修好的升级
这是个短故事，结局是好的，而且它解释了今天的循环为什么长这样。

%%% steps
step: 毛边 1 —— 对那些勉强擦进去的球一声不出
why: 一个在框沿上转了一圈然后*掉进去*的球算命中，所以它拿到的反馈是零。于是那些勉强算对的情况永远得不到打磨，机器一旦只是「对」，就停止进步了。
step: 毛边 2 —— 那一下推不管你错得多厉害
why: 差一根头发的偏差和飞出场外的抛投，拿到的修正一样大，因为 perceptron 没有一个数字表示自己**有多错**。这意味着机器没法把力气用在误差真正在的地方。
step: 升级 —— 一个平滑的 loss，加上它的 gradient
why: 今天的循环给*每一个*例子都打一个数字的分，然后读这个数字的斜率。于是大偏差换来大挪动，小偏差换来小挪动，连勉强算对的那一投也会被打磨 —— 每个例子每一圈都拿到**分级**的修正。
step: 顺手白拿的好处
why: 因为现在有了一个数字，你就可以*画*它。perceptron 什么都没有可看；你拿到一条曲线，一眼就能看出这次训练是否健康。
%%%

%% insight
注意这个故事的形状，因为它在整个机器学习里反复出现：有人把一条硬邦邦的是/否规则换成一个你能测量的平滑数字，然后你突然就能求导、能走小步、能看见进展。「把它变平滑，这样你就能挪它」大概是这个领域里被复用得最多的一个想法。
%%

到这里：一下没脑子的推，对比一次分级的挪动。下面是这个差别在六次练习投篮上长什么样。
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Before-and-after comparison. Top row, the perceptron: correct examples get no bar, wrong examples all get the same-height bar because the correction ignores how big the miss was. Bottom row, gradient descent: every example gets a bar whose height matches how big its miss was, small misses give small bars, big misses give big bars."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Size of correction per example — blind vs graded</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">每个例子拿到的修正大小 —— 没脑子的 对比 分级的</text><text class="lang-en" x="16" y="42" fill="#C93B3B" font-size="10">perceptron (only on a miss, size ignores how wrong):</text><text class="lang-zh" x="16" y="42" fill="#C93B3B" font-size="10">perceptron（只在没中时，大小不管错多少）：</text><line x1="30" y1="86" x2="360" y2="86" stroke="#E5DFD6"/><g fill="#C93B3B"><rect x="60" y="58" width="26" height="28"/><rect x="150" y="58" width="26" height="28"/><rect x="285" y="58" width="26" height="28"/></g><g fill="#6B645E" text-anchor="middle" font-size="8"><text class="lang-en" x="73" y="98">miss</text><text class="lang-zh" x="73" y="98">没中</text><text class="lang-en" x="118" y="98">✓ hit: 0</text><text class="lang-zh" x="118" y="98">✓ 投进：0</text><text class="lang-en" x="163" y="98">miss</text><text class="lang-zh" x="163" y="98">没中</text><text class="lang-en" x="208" y="98">✓ hit: 0</text><text class="lang-zh" x="208" y="98">✓ 投进：0</text><text class="lang-en" x="253" y="98">✓ hit: 0</text><text class="lang-zh" x="253" y="98">✓ 投进：0</text><text class="lang-en" x="298" y="98">miss</text><text class="lang-zh" x="298" y="98">没中</text></g><text class="lang-en" x="16" y="128" fill="#2D8B55" font-size="10">gradient descent (graded / every example):</text><text class="lang-zh" x="16" y="128" fill="#2D8B55" font-size="10">gradient descent（分级 / 每个例子都有）：</text><line x1="30" y1="172" x2="360" y2="172" stroke="#E5DFD6"/><g fill="#2D8B55"><rect x="60" y="150" width="26" height="22"/><rect x="105" y="166" width="26" height="6"/><rect x="150" y="140" width="26" height="32"/><rect x="195" y="163" width="26" height="9"/><rect x="240" y="158" width="26" height="14"/><rect x="285" y="132" width="26" height="40"/></g><text class="lang-en" x="200" y="186" text-anchor="middle" fill="#6B645E" font-size="8">bar height = how big the miss — everyone gets a fair, graded nudge</text><text class="lang-zh" x="200" y="186" text-anchor="middle" fill="#6B645E" font-size="8">柱子高度 = 偏差有多大 —— 每个例子都拿到公平的、分级的挪动</text></g></svg>
%%%

You just placed today's loop in history — 1958's blind shove became a graded, measurable nudge. Nice.

Now back to your loop, and its single most important dial.
~~~zh
你刚把今天的循环放进了历史里 —— 1958 年那一下没脑子的推，变成了一次分级的、可测量的挪动。不错。

现在回到你的循环，以及它最重要的那一个旋钮。
~~~

@@@ concept id=c4 zh_tag="一次挪多少" tag="How big a nudge" zh_title="learning rate（学习率）—— 以及它咬人的两种方式" title="The learning rate — and two ways it bites" zh_gotit="看懂 learning rate 了" gotit="Got the learning rate"
Picture walking down a hill into a valley, in the dark. You can't see the bottom, but you can *feel* the tilt of the ground under your boots. Before you set off you choose one rule — how **boldly** you turn that tilt into a step — and you are not allowed to change your mind halfway down. Choose timid and every step is a shuffle. Choose bold and steep ground launches you a long way. Here's the lovely part: with *one* setting, your steps are naturally long high on the steep slope and short down where the ground flattens. The steps shrink by themselves. What stays the same all walk long is the boldness.
~~~zh
想象你在黑夜里走下一座山，往山谷里走。你看不见谷底，但你*感觉*得到靴子底下地面的倾斜。出发之前你选定一条规则 —— 你要多**大胆**地把这个倾斜变成一步 —— 而且走到一半不许改主意。选得胆小，每一步都是挪一挪。选得大胆，陡的地面会把你甩出很远。妙的地方在这里：只用*一个*设置，你在高处陡坡上的步子自然就长，走到地面变平的地方步子自然就短。步子会自己变小。整段路里没变的是那份胆量。
~~~

%%% svg
<svg viewBox="0 0 520 206" role="img" aria-label="Three walkers going down the same valley with three different boldness settings. Left: a timid setting, steps stay tiny and the walker barely descends. Middle: a good setting, the steps start long on the steep slope and get shorter on their own as the ground flattens, landing at the bottom. Right: a too-bold setting, each leap flies clean across the valley and lands higher up the far side."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">One boldness setting for the whole walk — the STEP still changes with the ground</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">整段路只有一个胆量设置 —— 但每一步的大小还是跟着地面变</text><path d="M18 46 Q 95 152 166 46" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#2A7B9B" stroke-width="1.8" fill="none"><path d="M30 60 l9 9"/><path d="M39 69 l8 8"/><path d="M47 77 l8 7"/><path d="M55 84 l7 7"/></g><circle cx="62" cy="91" r="4" fill="#2A7B9B"/><text class="lang-en" x="92" y="172" text-anchor="middle" fill="#1F6280" font-size="9.5">timid setting</text><text class="lang-zh" x="92" y="172" text-anchor="middle" fill="#1F6280" font-size="9.5">胆小的设置</text><text class="lang-en" x="92" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">tiny steps → still up the slope</text><text class="lang-zh" x="92" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">小小的步子 → 还在坡上</text><path d="M186 46 Q 263 152 334 46" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#2D8B55" stroke-width="2" fill="none"><path d="M196 58 l26 38"/><path d="M222 96 l21 22"/><path d="M243 118 l13 10"/></g><circle cx="262" cy="131" r="4.5" fill="#2D8B55"/><text class="lang-en" x="260" y="172" text-anchor="middle" fill="#1a5c38" font-size="9.5">good setting</text><text class="lang-zh" x="260" y="172" text-anchor="middle" fill="#1a5c38" font-size="9.5">合适的设置</text><text class="lang-en" x="260" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">steps shrink on their own → lands</text><text class="lang-zh" x="260" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">步子自己变小 → 落下来了</text><path d="M354 46 Q 431 152 502 46" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><g stroke="#C93B3B" stroke-width="2" fill="none"><path d="M370 64 l106 34"/><path d="M476 98 l-116 -50"/></g><circle cx="360" cy="48" r="4" fill="#C93B3B"/><text class="lang-en" x="428" y="172" text-anchor="middle" fill="#C93B3B" font-size="9.5">too bold</text><text class="lang-zh" x="428" y="172" text-anchor="middle" fill="#C93B3B" font-size="9.5">太大胆</text><text class="lang-en" x="428" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">each leap lands higher up</text><text class="lang-zh" x="428" y="186" text-anchor="middle" fill="#6B645E" font-size="8.5">每一跃都落得更高</text></g></svg>
%%%

That boldness setting is the [[learning rate||the multiplier on every nudge: step = learning rate × gradient. Also called the step size. Too bold flings you past the bottom; too timid crawls]] — written `lr` — and it multiplies every single nudge, so it decides **how big a step** you take on any given slope. (You'll also hear it called the **step size**.) It is the single most fiddled-with number in the whole field: the team training the model that ranks your video feed, and the one that unlocks a phone with a face, are tuning this exact dial today.

**What the picture gets right:** one setting decides between crawling, arriving, and flying past the bottom forever.

**Where it breaks down:** the loop's real step is `lr × gradient`, so it *already* shrinks by itself as you near the bottom — those are the shrinking hops you just watched in concept 2. What stays fixed is `lr`, the multiplier. That's exactly why a bold `lr` can still fling you past the bottom even from close up: the slope down there is small, but a big enough multiplier turns a small slope into a big step.

#### When the setting is too bold — the loss climbs
This one is worth savouring, because it feels backwards: a *bolder* step makes the loss go **up**. Three rungs and you'll have the whole runaway.

%%% steps
step: the leap — a bold multiplier can carry you *past* the bottom
why: crossing the bottom is fine — you're still closer than you were, so the loss still falls. It only turns bad when the step carries you **more than twice** the distance you had left: then you land *farther* out than you started.
step: the runaway — farther out means steeper ground, which means a longer leap still
why: same multiplier, bigger slope, therefore a bigger jump — back across the bottom and farther out again. So the **weight bounces back and forth across the bottom while the loss climbs every lap**. It **diverges**, and it **blows up** until the number is too big for the computer (it prints `inf`) — and then the arithmetic stops making sense at all: `NaN`, "not a number".
step: the fix — shrink the multiplier
why: **Cause:** the learning rate was **too high**, so each step **overshoot**s by more than twice the distance left. **Remedy:** lower it until the curve slopes down instead of climbing.
%%%

So the whole story turns on *how far* past the bottom you land. Here are the two cases side by side, with the distance-to-the-bottom written on each hop.
~~~zh
那个胆量设置就是 [[learning rate（学习率）||每一次挪动的乘数：步长 = learning rate × gradient。也叫 step size（步长）。太大胆会把你甩过谷底；太胆小就只能爬]] —— 写成 `lr` —— 它乘在每一次挪动上，所以它决定了在任何给定的斜率上你**跨多大一步**。（你也会听到它被叫做 **step size**。）它是整个领域里被调得最多的一个数字：训练那个给你的视频排序的模型的团队，还有做刷脸解锁的团队，今天都在调这同一个旋钮。

**这个比喻对的地方：**一个设置就决定了你是在爬、能到达，还是永远飞过谷底。

**它不对的地方：**循环真正的一步是 `lr × gradient`，所以它*本来*就会在你接近谷底时自己变小 —— 就是你刚在 concept 2 里看到的那些越来越小的小步。固定不变的是 `lr`，那个乘数。这也正好解释了为什么一个大胆的 `lr` 就算离得很近也还能把你甩过谷底：那底下的斜率很小，但乘数够大，小斜率也会变成一大步。

#### 设置太大胆的时候 —— loss 会往上爬
这一段值得细品，因为它感觉是反着的：一步走得*更大胆*，反而让 loss **变大**。三格，你就拿到整个失控过程了。

%%% steps
step: 那一跃 —— 一个大胆的乘数会把你带到谷底*那一边*
why: 越过谷底本身没问题 —— 你还是比原来近，所以 loss 还是在降。只有当这一步把你带过去**超过**你原来剩下距离的**两倍**时，事情才变坏：那时你落的地方比出发时*更远*。
step: 失控 —— 落得更远意味着地面更陡，于是下一跃更长
why: 同样的乘数，更大的斜率，所以跳得更远 —— 又跨回谷底那边，而且更远。于是 **weight 在谷底两侧来回弹，loss 每一圈都在往上爬**。它 **diverge（发散）**了，然后一路 **blow up（炸开）**到数字大得电脑装不下（它会打印 `inf`）—— 再往后算术就彻底没有意义了：`NaN`，也就是「不是一个数字」。
step: 修法 —— 把乘数改小
why: **原因：**learning rate **太高**了，所以每一步 **overshoot（冲过头）**的距离超过剩余距离的两倍。**处方：**把它降下来，直到曲线是往下走而不是往上爬。
%%%

所以整个故事的关键在于你落在谷底*那一边多远*。下面把两种情况并排放，每一跳上都标了离谷底还有多远。
~~~

%%% svg
<svg viewBox="0 0 520 214" role="img" aria-label="Two bowls compared. Left bowl: a hop overshoots the bottom a little, distance left goes from 1.0 to 0.4 to 0.16, so the ball crosses back and forth over the bottom but still closes in and the loss falls. Right bowl: each hop overshoots by more than twice the distance left, so distance goes 1.0 to 1.5 to 2.25, landing farther out every lap and the loss climbs."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">How far past the bottom you land decides everything</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">你落在谷底那一边多远，决定了一切</text><text class="lang-en" x="95" y="34" text-anchor="middle" fill="#2D8B55" font-size="10">overshoot a LITTLE (under 2×)</text><text class="lang-zh" x="95" y="34" text-anchor="middle" fill="#2D8B55" font-size="10">只冲过头一点（不到 2 倍）</text><path d="M24 56 Q 95 162 166 56" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><line x1="95" y1="60" x2="95" y2="118" stroke="#E5DFD6" stroke-dasharray="3,3"/><text class="lang-en" x="95" y="130" text-anchor="middle" fill="#6B645E" font-size="8">bottom</text><text class="lang-zh" x="95" y="130" text-anchor="middle" fill="#6B645E" font-size="8">谷底</text><circle cx="155" cy="71" r="4.5" fill="#2D8B55"/><text x="172" y="70" fill="#1a5c38" font-size="8.5">1.0</text><circle cx="71" cy="103" r="4.5" fill="#2D8B55"/><text x="52" y="100" fill="#1a5c38" font-size="8.5">0.4</text><circle cx="105" cy="108" r="4.5" fill="#2D8B55"/><text x="112" y="118" fill="#1a5c38" font-size="8.5">0.16</text><path d="M152 76 C 120 100, 90 104, 74 100" fill="none" stroke="#2D8B55" stroke-width="1.4" stroke-dasharray="4,3"/><path d="M75 107 C 88 114, 96 112, 102 110" fill="none" stroke="#2D8B55" stroke-width="1.4" stroke-dasharray="4,3"/><text class="lang-en" x="95" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">ball crosses the bottom — but still closes in ✓</text><text class="lang-zh" x="95" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">球越过谷底 —— 但还是在靠近 ✓</text><text class="lang-en" x="95" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">distance left: 1.0 → 0.4 → 0.16 → …</text><text class="lang-zh" x="95" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">剩余距离：1.0 → 0.4 → 0.16 → …</text><text class="lang-en" x="385" y="34" text-anchor="middle" fill="#C93B3B" font-size="10">overshoot MORE than 2×</text><text class="lang-zh" x="385" y="34" text-anchor="middle" fill="#C93B3B" font-size="10">冲过头超过 2 倍</text><path d="M314 56 Q 385 162 456 56" fill="none" stroke="#B8AEA2" stroke-width="1.8"/><line x1="385" y1="60" x2="385" y2="118" stroke="#E5DFD6" stroke-dasharray="3,3"/><text class="lang-en" x="385" y="130" text-anchor="middle" fill="#6B645E" font-size="8">bottom</text><text class="lang-zh" x="385" y="130" text-anchor="middle" fill="#6B645E" font-size="8">谷底</text><circle cx="415" cy="100" r="4.5" fill="#C93B3B"/><text x="424" y="110" fill="#C93B3B" font-size="8.5">1.0</text><circle cx="340" cy="88" r="4.5" fill="#C93B3B"/><text x="322" y="84" fill="#C93B3B" font-size="8.5">1.5</text><circle cx="453" cy="61" r="4.5" fill="#C93B3B"/><text x="462" y="58" fill="#C93B3B" font-size="8.5">2.25</text><path d="M411 96 C 390 84, 366 80, 344 84" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><path d="M344 84 C 390 70, 430 62, 449 60" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><text class="lang-en" x="385" y="150" text-anchor="middle" fill="#C93B3B" font-size="9">lands FARTHER out every lap → the loss climbs ✕</text><text class="lang-zh" x="385" y="150" text-anchor="middle" fill="#C93B3B" font-size="9">每一圈都落得更远 → loss 往上爬 ✕</text><text class="lang-en" x="385" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">distance left: 1.0 → 1.5 → 2.25 → …</text><text class="lang-zh" x="385" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">剩余距离：1.0 → 1.5 → 2.25 → …</text><line x1="20" y1="178" x2="500" y2="178" stroke="#E5DFD6"/><text class="lang-en" x="260" y="196" text-anchor="middle" fill="#2C2A28" font-size="9.5">Overshooting is not the enemy. Overshooting by more than DOUBLE is.</text><text class="lang-zh" x="260" y="196" text-anchor="middle" fill="#2C2A28" font-size="9.5">冲过头不是敌人。冲过头超过一倍才是。</text><text class="lang-en" x="260" y="208" text-anchor="middle" fill="#6B645E" font-size="8.5">left: messy but it lands · right: runaway</text><text class="lang-zh" x="260" y="208" text-anchor="middle" fill="#6B645E" font-size="8.5">左：乱但落得下来 · 右：失控</text></g></svg>
%%%

%%% insight
This is the failure you will actually hit, over and over, for the rest of your ML life — and the loss curve tells you instantly. A curve that **climbs** instead of falling is almost never a "hard problem"; it's almost always a learning rate that's **too high**. That one reflex will save you hours. And notice how narrow the honest rule is: a hop that crosses the bottom is *fine*, even good. It's only the hop that lands farther out than it started that eats the run.
%%%

Now stop reading and **play with it.** Drag the learning-rate slider below. Predict before each drag: at a tiny `lr`, does the ball reach the bottom? Then crank it up and watch three different behaviours appear — a smooth slide, a ball that hops across the bottom and still lands, and finally a ball that never settles at all.
~~~zh
%%% insight
这个失败你在往后的 ML 生涯里会一次又一次真的撞上 —— 而 loss curve 会立刻告诉你。一条**往上爬**而不是往下掉的曲线，几乎从来不是「这个问题很难」；它几乎总是 learning rate **太高**。这一个条件反射能帮你省下好几个小时。也注意这条诚实的规则有多窄：跨过谷底的那一跳是*没问题*的，甚至是好事。只有落得比出发时更远的那一跳才会吃掉整次训练。
%%%

现在别读了，去**玩它**。拖下面那个 learning rate 滑块。每次拖之前先预测：`lr` 很小的时候，球能到谷底吗？然后把它调大，看三种不同的行为出现 —— 一次平滑的下滑，一个跨过谷底但还是落下来的球，最后一个永远停不下来的球。
~~~

%%% viz src=../../viz/gradient-descent.html title="Gradient descent playground — drag the learning rate" caption="Drag the learning-rate slider and step the ball down the bowl. Small setting = many tiny hops. Middle settings (try around 0.5–0.9) make the BALL hop back and forth ACROSS the bottom while still closing in — messy but it lands. Push past about 1.0 and it stops settling at all: every hop lands farther out and the loss climbs. Find that flip point."
%%%

#### When the setting is too timid — the loss crawls
Same dial, opposite bite, much less drama. Each nudge is a grain of sand: the loss *does* fall, but it **crawls** — after a thousand laps it **barely moves**. **Cause:** a learning rate **too low**, so the steps are **too tiny** to make headway. **Remedy:** raise it until the loss falls briskly *without* climbing.

So far: **too high** climbs, **too low** crawls. And you can tell which one you're in from the *shape* of the curve alone.
~~~zh
#### 设置太胆小的时候 —— loss 只能爬
同一个旋钮，反方向咬人，戏剧性小得多。每次挪动都只是一粒沙：loss *确实*在降，但它在**爬** —— 跑了一千圈之后它**几乎没动**。**原因：**learning rate **太低**，所以每一步**太小**，走不动。**处方：**把它调高，直到 loss 明快地下降，*而且*不往上爬。

到这里：**太高**会往上爬，**太低**只能爬。而且你光看曲线的*形状*就能分辨自己在哪一种里。
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Three loss curves over loops. Green just-right curve slides smoothly down and flattens near zero. Blue too-low curve slopes down very gently, barely dropping, still high at the end. Red too-high curve climbs steeply lap after lap and runs off the top of the chart."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Read the loss curve → diagnose the learning rate</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">读 loss curve → 诊断 learning rate</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text class="lang-en" x="20" y="90" fill="#6B645E" font-size="9" transform="rotate(-90 20 90)">loss</text><text class="lang-zh" x="20" y="90" fill="#6B645E" font-size="9" transform="rotate(-90 20 90)">loss</text><text class="lang-en" x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">loops →</text><text class="lang-zh" x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">圈数 →</text><path d="M52 40 C 140 130, 300 146, 490 148" fill="none" stroke="#2D8B55" stroke-width="2.2"/><text class="lang-en" x="392" y="140" fill="#2D8B55" font-size="9">just right ✓ (falls, flattens)</text><text class="lang-zh" x="392" y="140" fill="#2D8B55" font-size="9">刚好 ✓（下降、变平）</text><path d="M52 60 C 200 74, 360 86, 490 96" fill="none" stroke="#2A7B9B" stroke-width="2.2"/><text class="lang-en" x="352" y="82" fill="#1F6280" font-size="9">too low (crawls, still high)</text><text class="lang-zh" x="352" y="82" fill="#1F6280" font-size="9">太低（在爬，还很高）</text><path d="M52 60 C 90 58, 120 50, 146 40 C 164 34, 176 30, 184 27" fill="none" stroke="#C93B3B" stroke-width="2.4"/><polygon points="184,26 175,33 182,36" fill="#C93B3B"/><text class="lang-en" x="196" y="36" fill="#C93B3B" font-size="9">too high (climbs → diverges → NaN)</text><text class="lang-zh" x="196" y="36" fill="#C93B3B" font-size="9">太高（往上爬 → diverge → NaN）</text><text class="lang-en" x="196" y="48" fill="#6B645E" font-size="8">runs off the top of the chart</text><text class="lang-zh" x="196" y="48" fill="#6B645E" font-size="8">冲出图的顶上去了</text></g></svg>
%%%

One honest footnote on that red line: in a small loop like today's the climb is a clean, steady rise, every lap bigger than the last. In a real run with millions of weights the same runaway usually looks *jagged* — different weights blow up at different moments — so people often describe it as "spiky". Either way the trend is the same, and the diagnosis is the same: up and away means lower the learning rate.

#### The neat trick: start bold, then soften
A bold setting is great *far* from the bottom and terrible *near* it. So do the obvious thing: start bold, then **shrink the learning rate** as training goes on. **Smaller steps later** means you cover ground fast early and settle gently at the end. A plan that does this is a [[learning-rate schedule||a plan that shrinks the learning rate over time — bold early steps for speed, gentle late steps to settle; also called decay]], also called **decay**, and essentially every real model you've heard of is trained with one.

!!! c-ok 🎯
<b>Coming soon — Day 8 is a whole day on this one dial.</b> Today you only need what the loop needs: `lr` is the multiplier, too bold climbs, too timid crawls, and a schedule softens it over time. <i>How</i> to hunt for a good value (sweeping it by factors of ten and reading each curve) and <i>how</i> to design the shrink (the shapes a schedule can take) are Day 8's whole job. Park them there and enjoy the ride.
!!!

%%% hint
t1: Confused why a BIG step makes the loss go UP instead of just learning faster? Picture the walker in the valley again — think about *how far past* the bottom one leap can carry you.
t2: Say the bottom is 1 step to your right, and your leap lands you 3 steps to the right. You started 1 away; now you are 2 away — farther than before. Next lap the ground out there is steeper, so the leap is bigger still, and you land farther out again.
t3: Crossing the bottom is fine; landing farther out than you started is not. That happens when the step is more than **twice** the distance you had left — so the loss grows instead of falling. Lower the learning rate (the multiplier) and every step drops back under that line.
%%%

!!! c-info 🔬
<b>Optional (skippable) — the same thing in symbols.</b><br>
<b>The rule:</b> For weight `w` with gradient `g` and learning rate `η` (the Greek letter "eta", the usual symbol): the update is `w ← w − η·g`.<br>
<b>Why the step shrinks on its own:</b> The step *length* is `η × |g|`, and on a bowl-shaped loss `|g|` grows with your distance from the bottom — which is why the step shrinks all by itself as you approach, with `η` untouched.<br>
<b>Where it runs away:</b> The run only runs away when `η` is bigger than roughly `2 / (curvature of the loss)`; that inequality is exactly the "more than twice the distance left" rule, written in symbols.<br>
<b>What a schedule changes:</b> A schedule replaces the single `η` with a shrinking sequence `η₁ > η₂ > η₃ > …`.<br>
<b>And you can skip all of it:</b> You never need any of this to run a loop — the curve tells you everything by eye.
!!!

Dial sorted — and that's **two of the six traps** already in your kit. Next: how do we *count* all this practice, and how do we know when to stop?
~~~zh
关于那条红线，有一句诚实的补充：在今天这种小循环里，往上爬是一条干净稳定的上升线，每一圈都比上一圈高。在真实的、有几百万个 weight 的训练里，同样的失控通常看起来是*锯齿状*的 —— 不同的 weight 在不同的时刻炸开 —— 所以大家常说它「很尖」。不管哪种，趋势是一样的，诊断也是一样的：一路往上冲，就把 learning rate 降下来。

#### 一个漂亮的小把戏：先大胆，再变温柔
一个大胆的设置在*离*谷底远的时候很棒，在*靠近*谷底的时候很糟。那就做那件显而易见的事：先大胆，然后随着训练进行**把 learning rate 缩小**。**后期步子更小**意味着你早期跑得快，最后又能轻轻停下。做这件事的计划叫 [[learning-rate schedule（学习率调度）||一个让 learning rate 随时间缩小的计划 —— 早期大胆走得快，后期温柔好停下；也叫 decay（衰减）]]，也叫 **decay（衰减）**，而你听说过的几乎每一个真实模型都是用这个训练出来的。

!!! c-ok 🎯
<b>马上就来 —— 第 8 天整整一天都在讲这一个旋钮。</b>今天你只需要循环需要的那部分：`lr` 是乘数，太大胆会往上爬，太胆小只能爬，而 schedule 会让它随时间变温柔。<i>怎么</i>找一个好的值（按十倍一档扫过去，然后读每一条曲线），还有<i>怎么</i>设计这个缩小（schedule 可以长成什么形状），是第 8 天的全部工作。把它们停在那儿，先享受这一段。
!!!

%%% hint
t1: 不明白为什么一**大**步会让 loss **变大**，而不是学得更快？再想一想山谷里那个走路的人 —— 想想一跃能把你带到谷底*那一边多远*。
t2: 假设谷底在你右边 1 步的地方，而你一跃落在右边 3 步。你出发时离谷底 1；现在离 2 —— 比之前更远。下一圈那边的地面更陡，所以这一跃更大，你又落得更远。
t3: 越过谷底没问题；落得比出发时更远就有问题。当这一步超过你剩下距离的**两倍**时就会这样 —— 于是 loss 变大而不是变小。把 learning rate（那个乘数）降下来，每一步就都回到这条线以下了。
%%%

!!! c-info 🔬
<b>可选（可以跳过）—— 同一件事，用符号写。</b><br>
<b>规则：</b>对一个 weight `w`，它的 gradient 是 `g`，learning rate 是 `η`（希腊字母「eta」，惯用的符号）：update 就是 `w ← w − η·g`。<br>
<b>为什么步长会自己变小：</b>步*长*是 `η × |g|`，而在一个碗形的 loss 上，`|g|` 随着你离谷底的距离变大 —— 所以你越靠近，步长就自己越小，`η` 一个字都不用动。<br>
<b>它在哪里失控：</b>只有当 `η` 大过大约 `2 /（loss 的曲率）`时，训练才会失控；那个不等式正好就是「超过剩余距离两倍」这条规则，用符号写出来。<br>
<b>schedule 改的是什么：</b>schedule 把单一的 `η` 换成一串越来越小的 `η₁ > η₂ > η₃ > …`。<br>
<b>而且你完全可以跳过这一整段：</b>要跑一个循环，你从来不需要这里的任何东西 —— 曲线用眼睛就把一切告诉你了。
!!!

旋钮搞定 —— 这样**六个陷阱里已经有两个**进了你的工具包。接下来：我们怎么*数*这些练习，又怎么知道什么时候该停？
~~~

@@@ concept id=c5 zh_tag="数一数练了多少" tag="Counting practice" zh_title="epoch（轮次）、iteration（迭代），还有 loss curve（损失曲线）" title="Epochs, iterations, and the loss curve" zh_gotit="看懂 epoch 和曲线了" gotit="Got epochs & the curve"
Picture studying with a **deck of flashcards** before a test. Flipping *one* card and fixing what you got wrong is one small round of practice. Flipping through the *whole deck*, front to back, is one full pass. And you'd go through the deck many times before test day. Training counts practice in exactly those two units — it just gives them fancier names. Once you have the names you unlock the single most useful picture in all of machine learning: a plot of your own progress.
~~~zh
想象考试前用**一叠单词卡**复习。翻*一张*卡，把答错的地方改过来，这是一小轮练习。把*整叠*从头翻到尾，是一次完整的过。而考试前你会把这叠卡过很多遍。训练数练习量用的正好是这两个单位 —— 只是名字花哨一点。有了名字，你就解锁了整个机器学习里最有用的一张图：你自己进度的曲线。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A deck of flashcards. One highlighted card is labelled iteration, one update. A bracket around the whole row of cards is labelled epoch, one full pass through all cards. Below, three copies of the deck labelled epoch 1, epoch 2, epoch 3 show the deck being repeated."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Flashcards: one card = an iteration, one full deck = an epoch</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">单词卡：一张卡 = 一个 iteration，整叠一遍 = 一个 epoch</text><g><rect x="40" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="92" y="40" width="40" height="52" rx="4" fill="#E7F0F5" stroke="#2A7B9B" stroke-width="2"/><rect x="144" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="196" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/><rect x="248" y="40" width="40" height="52" rx="4" fill="#FDF3D6" stroke="#C99A12"/></g><path d="M112 96 l0 12" stroke="#2A7B9B"/><text class="lang-en" x="112" y="122" text-anchor="middle" fill="#1F6280" font-size="9">1 iteration</text><text class="lang-zh" x="112" y="122" text-anchor="middle" fill="#1F6280" font-size="9">1 个 iteration</text><text class="lang-en" x="112" y="134" text-anchor="middle" fill="#6B645E" font-size="8">(one update)</text><text class="lang-zh" x="112" y="134" text-anchor="middle" fill="#6B645E" font-size="8">（一次 update）</text><path d="M40 34 l248 0" stroke="#7C6DAA" stroke-width="1.6"/><text class="lang-en" x="164" y="30" text-anchor="middle" fill="#5E5191" font-size="9">1 epoch = whole deck once</text><text class="lang-zh" x="164" y="30" text-anchor="middle" fill="#5E5191" font-size="9">1 个 epoch = 整叠过一遍</text><g font-size="8" fill="#6B645E"><rect x="330" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text class="lang-en" x="347" y="102" text-anchor="middle">epoch 1</text><text class="lang-zh" x="347" y="102" text-anchor="middle">第 1 个 epoch</text><rect x="392" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text class="lang-en" x="409" y="102" text-anchor="middle">epoch 2</text><text class="lang-zh" x="409" y="102" text-anchor="middle">第 2 个 epoch</text><rect x="454" y="44" width="34" height="44" rx="3" fill="#F3ECDB" stroke="#C99A12"/><text class="lang-en" x="471" y="102" text-anchor="middle">epoch 3</text><text class="lang-zh" x="471" y="102" text-anchor="middle">第 3 个 epoch</text></g><text class="lang-en" x="409" y="130" text-anchor="middle" fill="#6B645E" font-size="8">repeat the deck many times</text><text class="lang-zh" x="409" y="130" text-anchor="middle" fill="#6B645E" font-size="8">把这叠卡重复很多遍</text></g></svg>
%%%

**What the picture gets right:** one card is one small update, the whole deck is one full pass, and you repeat the deck many times.

**Where it breaks down:** with flashcards you flip one card at a time. A network often grabs a small *handful* of cards per update, and that handful has a name: a [[batch||the handful of examples used for ONE update. Today our batch is a single example; how big to make the handful is a knob you'll tune in a later module]]. Today's batch is exactly one example, which keeps the counting simple — the handful size is a knob for a later module.

#### The two counts, and then the plot
Three short rungs. The third one is the payoff.

%%% steps
step: one card, one fix — an **iteration**
why: an [[iteration||one loop of forward → loss → backward → update, i.e. one weight update]] (also called **one step**) is a single trip round the four-step loop. So an iteration is exactly **one update** — nothing more.
step: the whole deck once — an **epoch**
why: an [[epoch||one full pass over all of your training data — many iterations back to back]] is **one full pass** over all your training data. Therefore 100 examples with one update each = 100 iterations in one epoch. Epochs are just iterations, bundled.
step: write down the loss after every lap, then **plot the loss**
why: that plot is your progress report. Because the loss is one number per lap, you can literally **watch it fall** — the network's whole education on one line.
%%%

Quick play before we look at the plot — guess the numbers first, they're friendlier than they look.

%%% demo id=count label="predict, then run — how many nudges hide inside '3 epochs'?"
predict: You have 200 training examples and you nudge once per example. How many iterations is ONE epoch? And after 3 epochs, how many nudges has the network taken — 3, 200, or something much bigger?
code: examples = 200                 # rows in the training set
code: iters_per_epoch = examples     # one nudge per example today (batch of 1)
code: for epoch in range(1, 4): print('after epoch', epoch, '-> total iterations:', epoch*iters_per_epoch)
out: after epoch 1 -> total iterations: 200
out: after epoch 2 -> total iterations: 400
out: after epoch 3 -> total iterations: 600
take: <b>200 iterations per epoch, 600 nudges after only 3 epochs.</b> That's why "just 3 epochs" already means hundreds of tiny corrections — and why nobody counts iterations out loud. Epochs are the human-sized unit.
%%%

%%% insight
This little plot you're about to meet is the instrument panel every ML engineer stares at all day — the same graph is up on a screen somewhere right now for the model that picks your next song. And you can read it already: it's how you tell "my model is learning" from "my model is broken" without knowing a single thing about the data. That's a genuinely professional skill, and you got it for the price of one graph.
%%%

So far: iterations, epochs, and a number to plot. Here's what a *healthy* run looks like.
~~~zh
**这个比喻对的地方：**一张卡是一次小的 update，整叠是一次完整的过，而你会把整叠重复很多遍。

**它不对的地方：**用单词卡你一次翻一张。网络常常一次 update 抓*一小把*卡，这一小把有个名字：[[batch（批）||一次 update 用到的那一小把例子。今天我们的 batch 就是一个例子；这一把该多大是你在后面 module 里要调的旋钮]]。今天的 batch 正好是一个例子，这让数起来简单 —— 那一把有多大是后面 module 的旋钮。

#### 两个计数单位，然后是那张图
三个短格子。第三个是回报。

%%% steps
step: 一张卡，一次修正 —— 一个 **iteration（迭代）**
why: 一个 [[iteration（迭代）||forward → loss → backward → update 的一圈，也就是一次 weight 更新]]（也叫 **one step**）就是四步循环走一圈。所以一个 iteration 正好是**一次 update** —— 不多不少。
step: 整叠过一遍 —— 一个 **epoch（轮次）**
why: 一个 [[epoch（轮次）||把你全部训练数据完整过一遍 —— 很多个 iteration 连起来]] 就是把你所有训练数据**完整过一遍**。所以 100 个例子、每个 update 一次 = 一个 epoch 里 100 个 iteration。epoch 就是打成一捆的 iteration。
step: 每一圈之后把 loss 记下来，然后**把 loss 画出来**
why: 那张图就是你的进度报告。因为 loss 是每圈一个数字，你真的可以**看着它往下掉** —— 网络的全部教育写在一条线上。
%%%

看图之前先玩一下 —— 先猜数字，它们比看起来友好。

%%% demo
predict: 你有 200 个训练例子，每个例子挪一次。一个 epoch 是多少个 iteration？3 个 epoch 之后，网络一共挪了多少次 —— 3 次、200 次，还是大得多的数？
code: examples = 200                 # 训练集里有多少行
code: iters_per_epoch = examples     # 今天每个例子挪一次（batch 为 1）
code: for epoch in range(1, 4): print('after epoch', epoch, '-> total iterations:', epoch*iters_per_epoch)
out: after epoch 1 -> total iterations: 200
out: after epoch 2 -> total iterations: 400
out: after epoch 3 -> total iterations: 600
take: <b>一个 epoch 200 个 iteration，才 3 个 epoch 就挪了 600 次。</b>这就是为什么「只有 3 个 epoch」已经意味着好几百次小修正 —— 也是为什么没人把 iteration 数出来讲。epoch 才是人用得顺的单位。
%%%

%%% insight
你马上要认识的这张小图，是每个 ML 工程师整天盯着的仪表盘 —— 此刻某处的屏幕上就有一张一样的图，属于那个挑你下一首歌的模型。而你已经能读它了：不用知道数据的任何一件事，你就能分辨「我的模型在学」和「我的模型坏了」。这是一项真正专业的技能，而你只花了一张图的价钱。
%%%

到这里：iteration、epoch，还有一个可以画出来的数字。下面是一次*健康*的训练长什么样。
~~~

%%% svg
<svg viewBox="0 0 520 186" role="img" aria-label="A healthy loss curve over epochs. It starts high, drops steeply in the first few epochs, then bends and glides down more gently, then flattens into a nearly horizontal line near the bottom, where a label reads converged: loss stops dropping."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A healthy loss curve: steep drop, gentle glide, then flat</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">健康的 loss curve：陡降、缓滑，然后变平</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text class="lang-en" x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text class="lang-zh" x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text class="lang-en" x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">epochs →</text><text class="lang-zh" x="270" y="172" text-anchor="middle" fill="#6B645E" font-size="9">epoch →</text><path d="M52 36 C 110 120, 180 140, 260 146 C 340 149, 420 150, 494 150" fill="none" stroke="#2D8B55" stroke-width="2.4"/><circle cx="70" cy="52" r="3.5" fill="#2D8B55"/><text class="lang-en" x="110" y="52" fill="#1a5c38" font-size="9">steep drop (fast learning)</text><text class="lang-zh" x="110" y="52" fill="#1a5c38" font-size="9">陡降（学得快）</text><circle cx="200" cy="141" r="3.5" fill="#2D8B55"/><text class="lang-en" x="240" y="128" fill="#1a5c38" font-size="9">gentle glide</text><text class="lang-zh" x="240" y="128" fill="#1a5c38" font-size="9">缓缓下滑</text><circle cx="470" cy="150" r="3.5" fill="#C99A12"/><text class="lang-en" x="360" y="140" fill="#9A7208" font-size="9">flat = converged</text><text class="lang-zh" x="360" y="140" fill="#9A7208" font-size="9">平了 = 收敛</text></g></svg>
%%%

Read it left to right in three beats. A steep drop, while there's plenty to learn. Then a gentle glide, because the nudges get smaller as the gap closes. Then the curve **flattens out**.

That last beat has a name. So what is convergence, exactly? When the loss **levels off** and **stops improving**, we say the run reached [[convergence||when the loss stops dropping much and levels off — the loop has mostly finished learning]] — the loop has mostly finished. One more little guess before you move on:

%%% demo id=flatten label="predict, then run — does the loss ever hit exactly 0?"
predict: Here are the losses from a healthy run, printed every 10 laps. Before you look: does the last number land exactly on 0, or just get very close and then stall?
code: losses = [(0,1.0),(10,0.1216),(20,0.0148),(30,0.0018),(40,0.00022),(50,0.00003)]
code: for lap, loss in losses: print(f'lap {lap:>2}: loss {loss:.5f}')
out: lap  0: loss 1.00000
out: lap 10: loss 0.12160
out: lap 20: loss 0.01480
out: lap 30: loss 0.00180
out: lap 40: loss 0.00022
out: lap 50: loss 0.00003
take: <b>Close, never exactly 0.</b> Watch the row labels — each printed line is <i>ten</i> laps apart, and each of those ten-lap jumps cuts the loss to about an eighth. One single lap trims it only to about 0.81 of what it was, so the per-lap drops get microscopic near the end. The curve <b>goes down</b> and then looks flat even though it is still creeping. That's convergence: not "finished", just "no longer worth another lap".
%%%

So: a [[loss curve||a plot of the loss over loops; a falling curve is proof the network is learning]] that **goes down** and then flattens is the shape you're hunting for. Anything else is a clue — and reading those clues is exactly what the next sections hand you.
~~~zh
从左到右按三拍读。先是陡降，因为还有很多东西要学。然后是缓缓下滑，因为差距合上时挪动也变小了。然后曲线**变平**。

最后那一拍有个名字。那么 convergence 到底是什么？当 loss **走平**并且**不再变好**时，我们说这次训练到了 [[convergence（收敛）||loss 不再明显下降、走平了 —— 循环基本学完了]] —— 循环基本结束了。往下走之前再猜一个小问题：

%%% demo
predict: 下面是一次健康训练的 loss，每 10 圈打印一次。看之前先想：最后那个数字会正好落在 0 上，还是只是非常接近然后停住？
code: losses = [(0,1.0),(10,0.1216),(20,0.0148),(30,0.0018),(40,0.00022),(50,0.00003)]
code: for lap, loss in losses: print(f'lap {lap:>2}: loss {loss:.5f}')
out: lap  0: loss 1.00000
out: lap 10: loss 0.12160
out: lap 20: loss 0.01480
out: lap 30: loss 0.00180
out: lap 40: loss 0.00022
out: lap 50: loss 0.00003
take: <b>很接近，但从不正好是 0。</b>看行号 —— 每一行之间差<i>十</i>圈，而每一次这样的十圈跳跃把 loss 砍到大约八分之一。单独一圈只把它削到原来的 0.81 左右，所以到后面每一圈的降幅小到看不见。曲线<b>往下走</b>，然后看起来平了，其实还在慢慢挪。这就是 convergence：不是「结束了」，只是「再跑一圈不值得了」。
%%%

所以：一条**往下走**然后变平的 [[loss curve（损失曲线）||把 loss 随圈数画出来的图；一条下降的曲线就是网络在学的证据]] 就是你要找的形状。别的任何形状都是线索 —— 而读这些线索正是接下来几节要交给你的。
~~~

@@@ concept id=c6 zh_tag="两个白拿的好处" tag="Two free wins" zh_title="两行代码的习惯，让每次训练都可信" title="Two one-line habits that make every run trustworthy" zh_gotit="看懂清零和打乱了" gotit="Got reset & shuffle"
Good news first: this section is two **free wins**. Two habits, one line of code each, and once you've met them you will never lose an afternoon to either again — which is more than most people can say, because these two bite *everybody* exactly once. Think of an **Etch A Sketch**: the red toy where two knobs draw a line and you shake it to erase. Start a new drawing without shaking the old one off and your fresh clean line lands on top of yesterday's scribble. In the loop, the "old drawing" is last lap's gradient.
~~~zh
先说好消息：这一节是两个**白拿的好处**。两个习惯，各一行代码，认识了它们你就再也不会为其中任何一个赔上一个下午 —— 这已经比大多数人强了，因为这两个坑*每个人*都正好踩一次。想一想**画画板**：那种红色玩具，两个旋钮画出一条线，摇一摇就擦干净。不摇掉旧的画就开始新的，你干净的新线会落在昨天的乱涂上面。在循环里，那个「旧的画」就是上一圈的 gradient。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Two Etch A Sketch toys side by side. Left: not shaken clear first, so a new clean line is drawn on top of an old scribble, making a mess, labelled forgot to reset: gradients pile up. Right: shaken clear first, so only the new clean line shows, labelled reset first: clean gradient each loop."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Clear the sketch before you draw — reset the gradient each loop</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">画之前先擦干净画板 —— 每一圈把 gradient 清零</text><rect x="24" y="34" width="212" height="120" rx="10" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><path d="M44 70 q30 30 60 -10 q30 40 60 0 q20 -20 50 20" fill="none" stroke="#B8AEA2" stroke-width="1.4"/><path d="M50 110 L110 60 L170 118 L216 74" fill="none" stroke="#C93B3B" stroke-width="2"/><text class="lang-en" x="130" y="140" text-anchor="middle" fill="#C93B3B" font-size="9">forgot to reset → old + new pile up</text><text class="lang-zh" x="130" y="140" text-anchor="middle" fill="#C93B3B" font-size="9">忘了清零 → 旧的加新的堆起来</text><rect x="284" y="34" width="212" height="120" rx="10" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><path d="M310 110 L370 60 L430 118 L476 74" fill="none" stroke="#2D8B55" stroke-width="2"/><text class="lang-en" x="390" y="140" text-anchor="middle" fill="#1a5c38" font-size="9">reset first → clean gradient this loop</text><text class="lang-zh" x="390" y="140" text-anchor="middle" fill="#1a5c38" font-size="9">先清零 → 这一圈的 gradient 是干净的</text></g></svg>
%%%

**What the picture gets right:** you must clear the old marks before each new drawing, or they add together.

**Where it breaks down:** you shake an Etch A Sketch when it *looks* messy. In the loop the piling-up is silent and automatic — nothing looks wrong, so you have to reset on purpose, every lap.

#### Win 1 · clear the sketch — reset the gradients
Three rungs, and the middle one is why this is nasty rather than merely annoying.

%%% steps
step: the default — the tools **accumulate**
why: most training tools *add* each new gradient onto whatever was already sitting there. It's a deliberate default that helps in fancier setups, therefore nobody warns you about it.
step: the pile-up — **gradients pile up** lap after lap
why: if you **forget to reset**, lap 2 uses lap 1 + lap 2, lap 3 uses all three, and so on. Which means your effective step grows every lap — you accidentally re-created the "too bold" runaway from the last section, with the learning rate untouched.
step: the fix — one line, at the top of every lap
why: [[zero the gradients||reset every weight's gradient to 0 at the start of each loop, before the backward pass — so each nudge uses only this loop's signal]] — literally `zero_grad()` in most frameworks. **Remedy:** **reset the gradients each step** — **clear the gradients** so each nudge carries only today's signal.
%%%

%%% insight
Why this one earns a callout: there is no error message. No crash, no warning, no red text — the loop runs happily and the numbers quietly rot. Silent bugs are the expensive kind, and "did I clear the gradients?" belongs on your reflex list forever.
%%%

Same gradient of `2` arriving every lap. Guess what the loop *uses* in each case.

%%% demo id=zerograd label="predict, then run — reset vs pile-up"
predict: If the same gradient (2) shows up on all three laps and you never clear the gradients, is the gradient the loop uses on lap 3 still 2, or something bigger? How much bigger?
code: g = 2   # this loop's gradient, same every loop
code: acc = 0
code: for i in range(3): acc = acc + g; print('no reset -> used gradient', acc)
code: for i in range(3): used = g; print('with reset -> used gradient', used)
out: no reset -> used gradient 2
out: no reset -> used gradient 4
out: no reset -> used gradient 6
out: with reset -> used gradient 2
out: with reset -> used gradient 2
out: with reset -> used gradient 2
take: <b>2 → 4 → 6.</b> Forget to reset and by lap 3 the nudge is three times too big — and it keeps growing. Clear the gradients and it stays a clean 2 forever. One tiny line saves the whole run.
%%%

#### Win 2 · don't practise from one spot — shuffle the data
The second win is about *order*, and it's a one-liner too. Practising **free-throw** after **free-throw** from the exact same spot on the floor makes you great at that one spot and useless everywhere else. Feed your examples in the **same fixed order** every epoch and the nudges get **biased by that order** — and **never shuffling** really bites when your data happens to arrive sorted (all the class-0 rows, then all the class-1 rows), because each stretch of laps drags the weights one way and the next stretch drags them back.

The remedy is delightfully cheap: [[shuffle the data||mix the order of your training examples before each epoch, so the network can't lock onto the order itself]] — **reshuffle each epoch**. **Mix up the order** and the problem is simply gone.
~~~zh
**这个比喻对的地方：**每次开始新的画之前你必须清掉旧的痕迹，不然它们会加在一起。

**它不对的地方：**画画板是你*看着*乱了才摇。在循环里，这种堆积是无声的、自动的 —— 什么都看不出不对，所以你必须故意去清，每一圈都清。

#### 好处 1 · 擦干净画板 —— 把 gradient 清零
三格，中间那一格解释了为什么这件事讨厌，而不只是有点烦。

%%% steps
step: 默认行为 —— 工具会**累加**
why: 大多数训练工具会把每个新的 gradient *加*到原来已经放在那里的东西上。这是一个刻意的默认设定，在更花哨的场景里有用，所以没人会来警告你。
why: 如果你**忘了清零**，第 2 圈用的是第 1 圈加第 2 圈，第 3 圈用的是三圈之和，以此类推。这意味着你实际的步长每一圈都在变大 —— 你在 learning rate 一个字没动的情况下，意外重造了上一节那个「太大胆」的失控。
step: 堆积 —— **gradient 一圈一圈堆起来**
step: 修法 —— 一行代码，放在每一圈的开头
why: [[把 gradient 清零||在每一圈开始、backward pass 之前，把每个 weight 的 gradient 重置为 0 —— 这样每次挪动只用这一圈的信号]] —— 在大多数框架里就是 `zero_grad()`。**处方：****每一步都把 gradient 清零** —— **清掉 gradient**，让每次挪动只带今天的信号。
%%%

%%% insight
为什么这一条值得单独拉出来说：它没有报错信息。不崩溃，不警告，没有红字 —— 循环跑得很开心，而数字在悄悄腐烂。无声的 bug 是最贵的那一种，而「我清 gradient 了吗？」这句话应该永远留在你的条件反射清单上。
%%%

每一圈都来同一个 gradient `2`。猜猜两种情况下循环各自*用*的是什么。

%%% demo
predict: 如果三圈都来同一个 gradient（2），而你从不清 gradient，第 3 圈循环用的 gradient 还是 2，还是更大的数？大多少？
code: g = 2   # 这一圈的 gradient，每圈都一样
code: acc = 0
code: for i in range(3): acc = acc + g; print('no reset -> used gradient', acc)
code: for i in range(3): used = g; print('with reset -> used gradient', used)
out: no reset -> used gradient 2
out: no reset -> used gradient 4
out: no reset -> used gradient 6
out: with reset -> used gradient 2
out: with reset -> used gradient 2
out: with reset -> used gradient 2
take: <b>2 → 4 → 6。</b>忘了清零，到第 3 圈这次挪动就大了三倍 —— 而且还会继续变大。清掉 gradient，它就永远是干干净净的 2。一行小代码救下整次训练。
%%%

#### 好处 2 · 别只在一个点上练 —— 把数据打乱
第二个好处讲的是*顺序*，也同样是一行。在地板上同一个点上一次又一次投**罚球**，会让你在那一个点上很厉害，在别的地方一无是处。每个 epoch 都按**同样固定的顺序**喂例子，挪动就会**被这个顺序带偏** —— 而**从不打乱**在你的数据刚好是排好序的时候（先是全部 class-0 的行，再是全部 class-1 的行）咬得最狠，因为一段圈把 weight 往一边拽，下一段又把它拽回来。

处方便宜得让人开心：[[把数据打乱||每个 epoch 之前把训练例子的顺序混一混，这样网络锁不住顺序本身]] —— **每个 epoch 重新打乱一次**。**把顺序混开**，问题就直接没了。
~~~

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="Top row: cards arriving in a fixed sorted order, all class zero cards first then all class one cards, with an arrow showing the weights being dragged one way then yanked back the other way. Bottom row: the same cards shuffled into a mixed order, with a straight steady arrow showing balanced nudges."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Same data, two orders — sorted drags, shuffled steadies</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">同样的数据，两种顺序 —— 排好序会拽，打乱了会稳</text><text class="lang-en" x="14" y="42" fill="#C93B3B" font-size="10">never shuffling (sorted):</text><text class="lang-zh" x="14" y="42" fill="#C93B3B" font-size="10">从不打乱（排好序）：</text><g><rect x="150" y="30" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="178" y="30" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="206" y="30" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="234" y="30" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="266" y="30" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="294" y="30" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="322" y="30" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="350" y="30" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/></g><path d="M150 66 L258 92" stroke="#C93B3B" stroke-width="2"/><path d="M258 92 L374 62" stroke="#C93B3B" stroke-width="2"/><polygon points="374,62 364,62 368,71" fill="#C93B3B"/><text class="lang-en" x="400" y="82" fill="#C93B3B" font-size="9">weights get yanked</text><text class="lang-zh" x="400" y="82" fill="#C93B3B" font-size="9">weight 被猛拽</text><text class="lang-en" x="400" y="94" fill="#6B645E" font-size="8">one way, then back</text><text class="lang-zh" x="400" y="94" fill="#6B645E" font-size="8">往一边，然后又拽回来</text><line x1="14" y1="106" x2="504" y2="106" stroke="#E5DFD6"/><text class="lang-en" x="14" y="126" fill="#2D8B55" font-size="10">reshuffle each epoch:</text><text class="lang-zh" x="14" y="126" fill="#2D8B55" font-size="10">每个 epoch 重新打乱：</text><g><rect x="150" y="118" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="178" y="118" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="206" y="118" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="234" y="118" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="266" y="118" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/><rect x="294" y="118" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="322" y="118" width="24" height="18" rx="3" fill="#E7F0F5" stroke="#2A7B9B"/><rect x="350" y="118" width="24" height="18" rx="3" fill="#FDECEC" stroke="#C93B3B"/></g><path d="M150 152 L374 152" stroke="#2D8B55" stroke-width="2"/><polygon points="374,152 364,148 364,156" fill="#2D8B55"/><text class="lang-en" x="400" y="150" fill="#2D8B55" font-size="9">steady, balanced</text><text class="lang-zh" x="400" y="150" fill="#2D8B55" font-size="9">平稳、均衡</text><text class="lang-en" x="400" y="162" fill="#6B645E" font-size="8">no order bias</text><text class="lang-zh" x="400" y="162" fill="#6B645E" font-size="8">没有顺序带来的偏差</text></g></svg>
%%%

!!! c-warn ⚠️
<b>Both wins are silent if you skip them.</b> No error, no crash — the loop runs and the numbers just come out wrong. So: reset the gradients every lap, reshuffle the data every epoch. Two one-line habits that separate a loop that learns from one that only looks like it does.
!!!

🎉 **Two wins banked** — that's **four of the six traps** down, and they cost you a paragraph each. Now let's put the traps down for a minute and go *play*, because there's something rather lovely to see.
~~~zh
!!! c-warn ⚠️
<b>这两个好处你不做的话都是无声的。</b>不报错，不崩溃 —— 循环照跑，只是数字出来是错的。所以：每一圈清零 gradient，每个 epoch 重新打乱数据。两个一行的习惯，分开了「真的在学的循环」和「只是看起来像在学的循环」。
!!!

🎉 **两个好处入袋** —— 这样**六个陷阱里去掉了四个**，而它们各只花了你一段话。现在先把陷阱放下一会儿，去*玩*，因为有个相当可爱的东西要看。
~~~

@@@ concept id=c7 zh_tag="看它动起来" tag="Watch it work" zh_title="玩一下 —— 这个循环到底在挪什么" title="Fun break — what the loop is actually moving" zh_gotit="玩过了" gotit="Had a play"
Numbers falling in a column are fine. But what is the loop *actually doing* out there in the world? Here's a picture you can hold. You're on a gym floor with basketballs on one side of the room and footballs on the other, all jumbled together. You lay a long **rope** on the floor and try to split them: balls on this side, footballs on that side. The rope is too far left? Slide it right. Tilted the wrong way? Spin it a little. Every nudge from the loop is one of those slides or spins — and after enough of them the rope sits exactly where it should. That's it. That's what training *looks* like from outside.
~~~zh
一列数字往下掉当然很好。但这个循环在真实世界里到底*在做什么*？这里有一幅你可以拿住的画。你站在体育馆的地板上，篮球在房间一边，橄榄球在另一边，全都混在一起。你在地上放一条长**绳子**，想把它们分开：篮球在这边，橄榄球在那边。绳子太靠左？往右挪。歪错方向了？转一点。循环的每一次挪动就是这样的一次平移或一次旋转 —— 挪够多次之后，绳子就正好停在该在的位置。就这样。这就是从外面看训练*长什么样*。
~~~

%%% svg
<svg viewBox="0 0 520 196" role="img" aria-label="A gym floor seen from above with two kinds of balls scattered on it. Left panel: a rope laid on the floor in the wrong place, so some balls are on the wrong side, with arrows showing it can be slid and spun. Right panel: after many nudges the rope sits in the right place and every ball is on its correct side."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A rope on the gym floor: slide it, spin it, until the two kinds are split</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">体育馆地板上的一条绳子：平移它、旋转它，直到两种球被分开</text><rect x="14" y="28" width="232" height="150" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><text class="lang-en" x="130" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">rope in the wrong place</text><text class="lang-zh" x="130" y="46" text-anchor="middle" fill="#C93B3B" font-size="10">绳子放错位置了</text><g font-size="13"><text x="46" y="84">🏀</text><text x="76" y="112">🏀</text><text x="52" y="140">🏀</text><text x="104" y="76">🏀</text><text x="150" y="94">🏈</text><text x="186" y="70">🏈</text><text x="196" y="118">🏈</text><text x="156" y="146">🏈</text></g><line x1="60" y1="60" x2="96" y2="166" stroke="#9A7208" stroke-width="3"/><text class="lang-en" x="30" y="60" fill="#9A7208" font-size="8">rope</text><text class="lang-zh" x="30" y="60" fill="#9A7208" font-size="8">绳子</text><g stroke="#2D8B55" stroke-width="1.8" fill="#2D8B55"><path d="M108 150 l30 0"/><polygon points="138,150 130,146 130,154"/></g><text class="lang-en" x="164" y="166" fill="#1a5c38" font-size="8">slide →</text><text class="lang-zh" x="164" y="166" fill="#1a5c38" font-size="8">平移 →</text><text class="lang-en" x="122" y="56" fill="#1a5c38" font-size="8">↻ spin</text><text class="lang-zh" x="122" y="56" fill="#1a5c38" font-size="8">↻ 旋转</text><rect x="272" y="28" width="232" height="150" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="388" y="46" text-anchor="middle" fill="#1a5c38" font-size="10">after many nudges ✓</text><text class="lang-zh" x="388" y="46" text-anchor="middle" fill="#1a5c38" font-size="10">挪动很多次之后 ✓</text><g font-size="13"><text x="292" y="84">🏀</text><text x="320" y="112">🏀</text><text x="298" y="140">🏀</text><text x="330" y="76">🏀</text><text x="418" y="94">🏈</text><text x="446" y="70">🏈</text><text x="456" y="118">🏈</text><text x="422" y="146">🏈</text></g><line x1="378" y1="56" x2="392" y2="170" stroke="#2D8B55" stroke-width="3"/><text class="lang-en" x="388" y="190" text-anchor="middle" fill="#6B645E" font-size="8.5">every ball on its own side — the loop moved the rope there, one nudge at a time</text><text class="lang-zh" x="388" y="190" text-anchor="middle" fill="#6B645E" font-size="8.5">每个球都在自己那一边 —— 是这个循环一次一挪把绳子挪过去的</text></g></svg>
%%%

**What the picture gets right:** the loop is not doing anything mysterious — it is *sliding and spinning one line* until the two groups fall on opposite sides.

**Where it breaks down:** a rope is one line on a flat floor. A real network has many lines in many more directions than you can draw — but every one of them gets nudged by exactly the loop you just built.

#### Your turn — grab the rope
This one is pure play. There is nothing to get wrong here, so just drag things and enjoy it. **Predict first:** if you slide the line without spinning it, can you still split the two groups? Then find out.
~~~zh
**这个比喻对的地方：**这个循环没在做任何神秘的事 —— 它就是在*平移和旋转一条线*，直到两群球落在两边。

**它不对的地方：**绳子是平地上的一条线。真的网络有很多条线，方向多得你画不出来 —— 但每一条都是被你刚造好的这个循环挪动的。

#### 换你来 —— 抓住绳子
这一段纯粹是玩。这里没有什么会做错，所以就拖着玩，享受它。**先预测：**如果你只平移这条线、不旋转它，还能把两群分开吗？然后去看答案。
~~~

%%% viz src=../../viz/neuron-boundary.html title="Drag the weights and move the line yourself" caption="You are now the training loop. Drag w₁, w₂ and b, and watch the line slide and spin across the dots. Try to split the two colours by hand — then think about how many tiny nudges the loop would need to find that same line on its own."
%%%

%%% insight
Here is the fun part, and it is genuinely true: the loop you built today is the *same* loop used to train the models that write essays and draw pictures. Nobody invented a fancier engine for them. They have more weights, more data, a smarter version of Step 4, and enormous computers — but the cycle is still forward → loss → backward → update. You have just learned the engine, at the size where you can watch every part of it move. Everything after this is *scale*.
%%%

Let's count that scale for a laugh, because the numbers are silly.

%%% demo id=scale label="predict, then run — how many nudges is 'a big training run'?"
predict: Your neuron takes 50 nudges in the experiment at the end of this lesson. A big model's training run takes... a thousand times more? A million times more? Guess an order of magnitude, then reveal.
code: mine  = 50                       # nudges in today's little experiment
code: theirs = 500_000                 # updates in a serious training run
code: print('my nudges   :', mine)
code: print('their nudges:', theirs)
code: print('same four steps, run', theirs // mine, 'times more often')
out: my nudges   : 50
out: their nudges: 500000
out: same four steps, run 10000 times more often
take: <b>About ten thousand times more laps — of the exact same four steps.</b> No new idea, just an enormous amount of patience (and electricity). Which means the thing you understood today is not a toy version of how learning works. It <i>is</i> how learning works.
%%%

Break over — and you've earned it. Now the one real thinking question of the day: how *long* should the loop run?
~~~zh
%%% insight
有意思的地方在这里，而且这是真的：你今天造的这个循环，和训练那些会写文章、会画画的模型用的是*同一个*循环。没人为它们发明更花哨的引擎。它们有更多 weight、更多数据、第 4 步的一个更聪明的版本，还有巨大的计算机 —— 但那个循环还是 forward → loss → backward → update。你刚学到的就是那个引擎，而且是在你能看清每个零件怎么动的尺度上学的。这之后的一切都是*规模*。
%%%

我们把这个规模数出来乐一下，因为这些数字挺傻的。

%%% demo
predict: 在这节课最后的实验里，你的 neuron 会被挪 50 次。一个大模型的训练要挪…… 多一千倍？多一百万倍？先猜一个数量级，再揭晓。
code: mine  = 50                       # 今天这个小实验里挪了几次
code: theirs = 500_000                 # 一次正经训练里的 update 次数
code: print('my nudges   :', mine)
code: print('their nudges:', theirs)
code: print('same four steps, run', theirs // mine, 'times more often')
out: my nudges   : 50
out: their nudges: 500000
out: same four steps, run 10000 times more often
take: <b>大约一万倍的圈数 —— 完全同样的四个步骤。</b>没有新想法，只是极其大量的耐心（和电）。这意味着你今天懂的这个东西不是学习原理的玩具版。它<i>就是</i>学习的原理。
%%%

休息结束 —— 你也确实该歇一下。现在是今天唯一一个真正需要动脑的问题：这个循环该跑*多久*？
~~~

@@@ concept id=c8 zh_tag="要练多久" tag="How long to practise" zh_title="要练多久 —— 以及为什么练多了反而更差" title="How long to practise — and why more can be worse" zh_gotit="看懂早停和练过头了" gotit="Got early/late stopping"
This section hides a surprising idea: *more practice can make a model worse.*

You already know both halves from school. Study **too little** and you walk into the exam unprepared. Study **too much of the wrong kind** — memorizing the practice sheet answers word-for-word — and you ace the practice sheet but freeze on the real exam, because it asks the same ideas with different numbers.

Good practice lives in between. The loop has exactly the same Goldilocks problem.
~~~zh
这一节藏着一个让人意外的想法：*练得更多，模型可能变得更差。*

这两半你在学校里都经历过。学得**太少**，你会毫无准备地走进考场。学了**太多错的那种** —— 把练习卷的答案一字一句背下来 —— 你在练习卷上满分，真考试却懵了，因为它用不同的数字问同样的道理。

好的练习在两者之间。这个循环有的正是同一个「刚刚好」的问题。
~~~

%%% svg
<svg viewBox="0 0 520 188" role="img" aria-label="A student studying for a test, shown three ways. Left: barely studied, sad face, labelled too little, underfit. Middle: studied just enough, happy face, labelled just right. Right: memorized every practice answer word for word, confused on the real exam, labelled too much, memorized."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Studying for a test: too little, just right, too much</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">为考试复习：太少、刚刚好、太多</text><rect x="16" y="34" width="150" height="132" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="91" y="70" text-anchor="middle" font-size="26">😟</text><text class="lang-en" x="91" y="104" text-anchor="middle" fill="#C93B3B">too little</text><text class="lang-zh" x="91" y="104" text-anchor="middle" fill="#C93B3B">太少</text><text class="lang-en" x="91" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">barely practised</text><text class="lang-zh" x="91" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">几乎没练</text><text class="lang-en" x="91" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ bombs the test</text><text class="lang-zh" x="91" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ 考砸了</text><text class="lang-en" x="91" y="156" text-anchor="middle" fill="#C93B3B" font-size="9">(underfit)</text><text class="lang-zh" x="91" y="156" text-anchor="middle" fill="#C93B3B" font-size="9">（underfit）</text><rect x="185" y="34" width="150" height="132" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="260" y="70" text-anchor="middle" font-size="26">🙂</text><text class="lang-en" x="260" y="104" text-anchor="middle" fill="#1a5c38">just right</text><text class="lang-zh" x="260" y="104" text-anchor="middle" fill="#1a5c38">刚刚好</text><text class="lang-en" x="260" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">learned the pattern</text><text class="lang-zh" x="260" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">学会了规律</text><text class="lang-en" x="260" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ handles new questions</text><text class="lang-zh" x="260" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ 能应付新题</text><rect x="354" y="34" width="150" height="132" rx="6" fill="#FDF3D6" stroke="#C99A12"/><text x="429" y="70" text-anchor="middle" font-size="26">😵</text><text class="lang-en" x="429" y="104" text-anchor="middle" fill="#9A7208">too much</text><text class="lang-zh" x="429" y="104" text-anchor="middle" fill="#9A7208">太多</text><text class="lang-en" x="429" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">memorized the answers</text><text class="lang-zh" x="429" y="124" text-anchor="middle" fill="#6B645E" font-size="8.5">把答案背下来了</text><text class="lang-en" x="429" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ freezes on new ones</text><text class="lang-zh" x="429" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">→ 碰到新题就懵</text><text class="lang-en" x="429" y="156" text-anchor="middle" fill="#9A7208" font-size="9">(overfit)</text><text class="lang-zh" x="429" y="156" text-anchor="middle" fill="#9A7208" font-size="9">（overfit）</text></g></svg>
%%%

**What the picture gets right:** too little and too much both hurt, in opposite ways — one from not enough practice, one from the wrong kind.

**Where it breaks down:** a student *knows* when they're only memorizing. A network doesn't — its practice score keeps looking great, so you need a separate check to catch it.

#### Quitting too early — the easy half
One line and you're done with this one. Cut the loop short and the loss is **still high** — the network simply hadn't finished walking downhill yet. **Cause: too few epochs** (**stopping too early**). **Remedy:** **train longer** and watch the curve until it flattens. That not-finished-yet state has a name: **underfit** (you will also see it written **underfitting**).

#### Drilling too long — the sneaky half (training too long)
Now the good one, and the most important idea of the day. Follow the beats — and notice that the second one is completely invisible if you only look at the training loss.

%%% steps
step: the honest phase — real pattern, still plenty to learn
why: both the practice score and the real-exam score improve together. Every lap is genuine progress.
step: the turn — the real signal runs out
why: there's nothing general left to learn, therefore the loop starts fitting the little accidents that exist *only* in your training rows. That is **overfitting**: the network **memorizes** instead of generalizing.
step: the trap — the training loss still looks great
why: memorizing lowers the training loss! Which means the number you were happily watching now *lies* to you. **Cause:** you **train too long** for the amount your data can teach.
step: the alarm — keep a slice of data back and score it too
why: on a [[validation set||a slice of data held back from training, used only to check whether the network handles data it has never practised on]] the network has never practised, so memorizing doesn't help there. Therefore the moment memorizing starts, the **validation loss rises** while training loss keeps falling. Two curves fork — and the fork is your alarm bell.
step: the fix — **early stopping**
why: [[early stopping||halt training when the validation loss stops improving and starts rising — right before memorizing takes over]] means you halt at the fork: **stop when validation stops improving**, and keep the version that scored best on **held-out** data rather than the one that memorized best.
%%%

%%% insight
Sit with this one, because it's the first time the number you're optimizing becomes your enemy. The loop is doing *exactly* what you asked — drive the training loss down — and that is precisely how it goes wrong. Almost all of "real" machine learning is this lesson on repeat: what you can measure is never quite what you actually want.
%%%

Enough words — let's catch the fork in the act. Six epochs of a real-ish run, both scores printed side by side.

%%% demo id=fork label="predict, then run — which epoch would you keep?"
predict: Training loss falls all six epochs. Validation loss does *not*. Which epoch has the BEST (lowest) validation score — the one early stopping would keep? Guess an epoch number, then reveal.
code: train = [0.90, 0.42, 0.21, 0.12, 0.07, 0.04]   # score on rows it practises on
code: val   = [0.95, 0.50, 0.31, 0.28, 0.33, 0.41]   # score on held-out rows
code: for e, (t, v) in enumerate(zip(train, val), 1): print(f'epoch {e}: train {t:.2f} | val {v:.2f}')
out: epoch 1: train 0.90 | val 0.95
out: epoch 2: train 0.42 | val 0.50
out: epoch 3: train 0.21 | val 0.31
out: epoch 4: train 0.12 | val 0.28
out: epoch 5: train 0.07 | val 0.33
out: epoch 6: train 0.04 | val 0.41
take: <b>Epoch 4 — that's the one to keep.</b> Validation loss turns upward at epoch 5 (0.28 → 0.33), so epoch 4 was its best score. After that, train keeps improving (0.12 → 0.04, looks fantastic!) while val gets steadily worse. Epochs 5 and 6 made the model <i>worse at its actual job</i>. Early stopping notices the rise at epoch 5, keeps the epoch-4 weights, and walks away.
%%%

Here's the same story as the picture you'll actually stare at.
~~~zh
**这个比喻对的地方：**太少和太多都会伤人，而且方向相反 —— 一个是练得不够，一个是练错了种类。

**它不对的地方：**学生*知道*自己只是在背。网络不知道 —— 它的练习分数一直很好看，所以你需要一个单独的检查才能抓到它。

#### 停得太早 —— 容易的那一半
一句话就讲完了。把循环截短，loss **还很高** —— 网络只是还没走完下坡。**原因：epoch 太少**（**停得太早**）。**处方：****多训练一会儿**，看着曲线直到它变平。那个「还没完成」的状态有个名字：**underfit（欠拟合）**（你也会看到写成 **underfitting**）。

#### 练得太久 —— 阴的那一半（训练时间过长）
现在是好东西，也是今天最重要的想法。跟着这几拍走 —— 注意第二拍如果你只看训练 loss 的话是完全看不见的。

%%% steps
step: 诚实的阶段 —— 真的规律，还有很多要学
why: 练习分数和真考试分数一起变好。每一圈都是真实的进步。
step: 转折 —— 真的信号用完了
why: 已经没有可以推广的东西可学了，所以循环开始去拟合那些*只*存在于你训练数据行里的小意外。这就是 **overfitting（过拟合）**：网络在**背答案**，而不是在学会推广。
step: 陷阱 —— 训练 loss 看起来还是很棒
why: 背答案会让训练 loss 变小！这意味着你一直开心地盯着的那个数字现在在*骗*你。**原因：**相对于你的数据能教的量，你**训练得太久**了。
step: 警报 —— 留一片数据出来，也给它打分
why: 在 [[validation set（验证集）||从训练里留出来的一片数据，只用来检查网络对它从没练过的数据表现如何]] 上网络从没练过，所以背答案在那里帮不上忙。因此背答案一开始，**validation loss 就会上升**，而训练 loss 还在下降。两条曲线分叉 —— 这个分叉就是你的警铃。
step: 修法 —— **early stopping（早停）**
why: [[early stopping（早停）||当 validation loss 不再变好、开始上升时就停下训练 —— 正好在背答案接管之前]] 意思是你在分叉处停手：**validation 不再变好就停**，然后保留在**留出**数据上分数最好的那个版本，而不是背得最好的那个。
%%%

%%% insight
在这一条上多坐一会儿，因为这是第一次你在优化的那个数字变成了你的敌人。循环做的*正好*是你要它做的事 —— 把训练 loss 压下去 —— 而这恰恰就是它出错的方式。「真实」机器学习里几乎所有事都是这一课的重复：你能测量的东西，从来都不完全是你真正想要的东西。
%%%

话说够了 —— 我们当场抓住这个分叉。一次接近真实的训练跑了六个 epoch，两个分数并排打出来。

%%% demo
predict: 训练 loss 六个 epoch 一直在降。validation loss *不是*。哪个 epoch 的 validation 分数最好（最低）—— 也就是 early stopping 会留下的那个？先猜一个 epoch 编号，再揭晓。
code: train = [0.90, 0.42, 0.21, 0.12, 0.07, 0.04]   # 在它练过的那些行上的分数
code: val   = [0.95, 0.50, 0.31, 0.28, 0.33, 0.41]   # 在留出的那些行上的分数
code: for e, (t, v) in enumerate(zip(train, val), 1): print(f'epoch {e}: train {t:.2f} | val {v:.2f}')
out: epoch 1: train 0.90 | val 0.95
out: epoch 2: train 0.42 | val 0.50
out: epoch 3: train 0.21 | val 0.31
out: epoch 4: train 0.12 | val 0.28
out: epoch 5: train 0.07 | val 0.33
out: epoch 6: train 0.04 | val 0.41
take: <b>第 4 个 epoch —— 就是要留下的那个。</b>validation loss 在第 5 个 epoch 转头往上（0.28 → 0.33），所以第 4 个 epoch 是它最好的分数。之后训练分数继续变好（0.12 → 0.04，看着真棒！），而 validation 一路变差。第 5 和第 6 个 epoch 让模型<i>在它真正的工作上变差了</i>。early stopping 注意到第 5 个 epoch 的上升，留下第 4 个 epoch 的 weight，然后收工。
%%%

同样的故事，换成你真正会盯着看的那张图。
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two loss curves over epochs. The blue training-loss curve falls steadily and keeps dropping toward zero. The red validation-loss curve falls at first alongside it, reaches a lowest point, then turns back upward. A vertical dashed line at that lowest point is labelled stop here, early stopping. The region after it is shaded and labelled overfitting."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Training loss vs validation loss — stop at the fork</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">训练 loss 对 validation loss —— 在分叉处停下</text><line x1="46" y1="26" x2="46" y2="150" stroke="#B8AEA2"/><line x1="46" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text class="lang-en" x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text class="lang-zh" x="20" y="92" fill="#6B645E" font-size="9" transform="rotate(-90 20 92)">loss</text><text class="lang-en" x="270" y="174" text-anchor="middle" fill="#6B645E" font-size="9">epochs →</text><text class="lang-zh" x="270" y="174" text-anchor="middle" fill="#6B645E" font-size="9">epoch →</text><rect x="300" y="26" width="200" height="124" fill="#FDF3D6" opacity="0.55"/><text class="lang-en" x="400" y="42" text-anchor="middle" fill="#9A7208" font-size="9">overfitting zone</text><text class="lang-zh" x="400" y="42" text-anchor="middle" fill="#9A7208" font-size="9">overfitting 区</text><path d="M52 40 C 160 110, 320 138, 494 148" fill="none" stroke="#2A7B9B" stroke-width="2.2"/><text class="lang-en" x="410" y="140" fill="#1F6280" font-size="9">training loss (keeps falling)</text><text class="lang-zh" x="410" y="140" fill="#1F6280" font-size="9">训练 loss（一直在降）</text><path d="M52 46 C 150 96, 280 108, 300 108 C 380 108, 440 90, 494 70" fill="none" stroke="#C93B3B" stroke-width="2.2"/><text class="lang-en" x="360" y="66" fill="#C93B3B" font-size="9">validation loss (turns up!)</text><text class="lang-zh" x="360" y="66" fill="#C93B3B" font-size="9">validation loss（转头往上了！）</text><line x1="300" y1="26" x2="300" y2="150" stroke="#2D8B55" stroke-width="1.8" stroke-dasharray="5,3"/><circle cx="300" cy="108" r="4" fill="#2D8B55"/><text class="lang-en" x="300" y="166" text-anchor="middle" fill="#1a5c38" font-size="9">⬆ stop here (early stopping)</text><text class="lang-zh" x="300" y="166" text-anchor="middle" fill="#1a5c38" font-size="9">⬆ 在这里停（early stopping）</text></g></svg>
%%%

That's the Goldilocks rule in one image: run *enough* to converge, stop *before* the red line turns up.

🎉 **Victory lap.** You can now read a loss curve and name what's wrong with a run — too bold, too timid, no reset, fixed order, quit too early, drilled too long. **All six traps, with a one-line cure each**, and honestly that's most of what a working engineer does on a bad day. One honest truth left, and it's the interesting kind: the thing the loop *cannot* do, no matter how perfectly you tune it.
~~~zh
这就是一张图里的「刚刚好」规则：跑*够久*让它收敛，在红线转头向上*之前*停下。

🎉 **胜利一圈。**你现在能读一条 loss curve，并说出这次训练哪里不对 —— 太大胆、太胆小、没清零、顺序固定、停得太早、练得太久。**六个陷阱，每个配一行的解法**，而说实话，这就是一个工程师在糟糕的一天里做的大部分事情。还剩一个诚实的事实，而且是有意思的那一种：无论你调得多完美，这个循环*做不到*的事。
~~~

@@@ concept id=c9 zh_tag="它撞墙的地方" tag="Where it hits a wall" zh_title="诚实的边界 —— 什么时候调参救不了你" title="The honest limit — when tuning cannot help" zh_gotit="看懂这些边界了" gotit="Got the limits"
This last idea is the one that separates someone who *uses* a tool from someone who *understands* it — and it's a satisfying one, because it hands you a diagnosis nobody taught you before. Try this: draw a perfect **circle using only a straight ruler**. Practise all year. Every stroke you make will be a straight line, so you'll never get a circle — not because you practised badly, but because the tool cannot make that shape. More repetition can't add a power a tool never had.
~~~zh
最后这个想法，把*会用*工具的人和*懂*工具的人分开了 —— 而且它让人满意，因为它给了你一个从来没人教过你的诊断。试试这个：**只用一把直尺**画一个完美的**圆**。练一整年。你画的每一笔都会是直线，所以你永远画不出圆 —— 不是因为你练得不好，而是因为这个工具做不出那个形状。重复再多次，也没法给一个工具加上它从来没有的能力。
~~~

%%% svg
<svg viewBox="0 0 520 184" role="img" aria-label="On the left, a hand holding a straight ruler drawing straight strokes that try and fail to form a circle: the target circle is dashed and the ruler strokes form a jagged polygon that never matches it. On the right, a compass draws a clean circle in one sweep. Caption: no amount of practice makes a ruler draw a circle; you need a different tool."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A ruler can only make straight strokes — practice never gives it a curve</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">直尺只能画直的笔画 —— 练习永远不会给它一个弯</text><rect x="14" y="28" width="240" height="142" rx="6" fill="#FDECEC" stroke="#C93B3B"/><circle cx="130" cy="96" r="48" fill="none" stroke="#B8AEA2" stroke-width="1.4" stroke-dasharray="4,4"/><path d="M96 62 L164 62 L178 96 L164 130 L96 130 L82 96 Z" fill="none" stroke="#C93B3B" stroke-width="2"/><rect x="60" y="146" width="84" height="10" rx="2" fill="#E5DFD6" stroke="#9A938A"/><text class="lang-en" x="102" y="155" text-anchor="middle" fill="#6B645E" font-size="7">ruler</text><text class="lang-zh" x="102" y="155" text-anchor="middle" fill="#6B645E" font-size="7">直尺</text><text class="lang-en" x="196" y="152" fill="#C93B3B" font-size="9">still not</text><text class="lang-zh" x="196" y="152" fill="#C93B3B" font-size="9">还是不行</text><text class="lang-en" x="196" y="163" fill="#C93B3B" font-size="9">a circle ✕</text><text class="lang-zh" x="196" y="163" fill="#C93B3B" font-size="9">一个圆 ✕</text><rect x="272" y="28" width="234" height="142" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><circle cx="380" cy="96" r="48" fill="none" stroke="#2D8B55" stroke-width="2.4"/><path d="M380 96 L416 62" stroke="#9A938A" stroke-width="2"/><circle cx="380" cy="96" r="3" fill="#9A938A"/><text class="lang-en" x="452" y="100" fill="#1a5c38" font-size="9">compass ✓</text><text class="lang-zh" x="452" y="100" fill="#1a5c38" font-size="9">圆规 ✓</text><text class="lang-en" x="452" y="112" fill="#6B645E" font-size="8">a different tool</text><text class="lang-zh" x="452" y="112" fill="#6B645E" font-size="8">换一个工具</text><text class="lang-en" x="389" y="160" text-anchor="middle" fill="#1a5c38" font-size="9">add the right tool, not more practice</text><text class="lang-zh" x="389" y="160" text-anchor="middle" fill="#1a5c38" font-size="9">加对的工具，而不是加更多练习</text></g></svg>
%%%

**What the picture gets right:** some jobs need a power the tool simply doesn't have, and repetition cannot conjure it.

**Where it breaks down:** you *can* get a circle — by adding a compass. Likewise a network gets curved boundaries by adding more neurons in a hidden layer. The wall belongs to the *single* neuron, not to networks in general.

#### The ruler wall — one neuron and [[XOR||"exclusive or": output 1 when the two inputs differ, 0 when they match — the four points can't be split by any single straight line]]
XOR is the classic. Four points; the two that *match* belong to one class, the two that *differ* to the other. A single neuron is a ruler: it can only cut the space with **one straight line** — the same rope you were dragging two sections ago.

%%% steps
step: the shape — XOR is **non-linearly separable**
why: no single straight line can put both matching points on one side and both differing points on the other. Try it in your head; then try it for real in the widget below.
step: the run — train it anyway, and the loss **plateaus above zero**
why: the loop still works perfectly! It walks downhill as far as downhill goes, then sits on a **plateau** — flat, but stuck at a bad score, because a **single neuron cannot** represent the answer.
step: the trap — reaching for more epochs
why: **more iterations never** fix this. A flat-but-high curve after a long, healthy run is your hint that the *model* is wrong, not the *knobs*.
step: the fix — change the tool, not the practice
why: add neurons (a hidden layer) and the pair of them carve two lines, which together solve XOR. That's the compass — and it's the whole reason the next module stacks layers.
%%%

%%% insight
This is the most useful diagnosis in the whole day: is my run failing because the **knobs** are wrong (too bold, too few epochs, forgot to shuffle) or because the **model** cannot express the answer? Everything before this section fixes knobs. Nothing before this section can fix a ruler being asked for a circle. Telling those two apart is the skill — and now you have it.
%%%

Don't take my word for it — go **try** to split XOR with one line. Drag the three sliders. Then tick "add a hidden layer" and watch the wall disappear.
~~~zh
**这个比喻对的地方：**有些工作需要一种工具根本没有的能力，而重复变不出这种能力。

**它不对的地方：**你*确实*能得到圆 —— 加一个圆规就行。同样，网络靠在 hidden layer（隐藏层）里加更多 neuron 来得到弯的边界。这道墙属于*单个* neuron，不属于网络这一类东西。

#### 直尺的墙 —— 一个 neuron 和 [[XOR||「异或」：两个输入不一样时输出 1，一样时输出 0 —— 这四个点没法被任何一条直线分开]]
XOR 是那个经典例子。四个点；两个*一样*的属于一类，两个*不一样*的属于另一类。单个 neuron 就是一把直尺：它只能用**一条直线**去切这个空间 —— 就是你两节前拖过的那条绳子。

%%% steps
step: 形状 —— XOR 是 **non-linearly separable（线性不可分）**的
why: 没有哪一条直线能把两个「一样」的点放在一边、两个「不一样」的点放在另一边。在脑子里试试；然后去下面的控件里真的试一次。
step: 训练 —— 照样训它，loss 会**停在零以上的一个平台上**
why: 循环还是工作得很好！它一路往下走到下坡走完，然后停在一个 **plateau（平台）**上 —— 平了，但卡在一个糟糕的分数上，因为**单个 neuron 没法**表示这个答案。
step: 陷阱 —— 伸手去加更多 epoch
why: **更多 iteration 永远**修不好这个。一次长时间、健康的训练之后曲线又平又高，这是在提示你*模型*错了，不是*旋钮*错了。
step: 修法 —— 换工具，而不是换练法
why: 加 neuron（一个 hidden layer），它们两个一起切出两条线，合起来就解决了 XOR。那就是圆规 —— 也正是下一个 module 要把层叠起来的全部理由。
%%%

%%% insight
这是今天最有用的诊断：我这次训练失败，是因为**旋钮**不对（太大胆、epoch 太少、忘了打乱），还是因为**模型**表达不出这个答案？这一节之前的所有内容都是在修旋钮。这一节之前的任何东西都修不好「拿直尺去画圆」。把这两件事分清楚就是那项技能 —— 现在你有了。
%%%

别光听我说 —— 去**试**着用一条线把 XOR 分开。拖那三个滑块。然后勾上「加一个 hidden layer」，看那道墙消失。
~~~

%%% viz src=../../viz/xor-limit.html title="One line, four points, no solution" caption="Drag w₁, w₂ and b to move the single straight boundary. Predict first: can you find ANY setting that sorts all four points? Then tick 'add a hidden layer' — two lines, and suddenly it's solvable."
%%%

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A grid of four points showing the XOR pattern: bottom-left and top-right are one class, top-left and bottom-right are the other. A dashed straight line tries to separate them but always leaves one point on the wrong side. Beside it, a loss curve falls then flattens on a plateau well above zero."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A straight cut can't split XOR → the loss flattens above zero</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">一条直的切线分不开 XOR → loss 平在零以上</text><line x1="90" y1="40" x2="90" y2="150" stroke="#B8AEA2"/><line x1="90" y1="150" x2="240" y2="150" stroke="#B8AEA2"/><circle cx="110" cy="130" r="9" fill="#2A7B9B"/><circle cx="220" cy="60" r="9" fill="#2A7B9B"/><circle cx="110" cy="60" r="9" fill="#C93B3B"/><circle cx="220" cy="130" r="9" fill="#C93B3B"/><line x1="80" y1="150" x2="240" y2="55" stroke="#9A938A" stroke-width="1.8" stroke-dasharray="5,3"/><text class="lang-en" x="165" y="172" text-anchor="middle" fill="#6B645E" font-size="8.5">a straight cut always mis-sorts one dot</text><text class="lang-zh" x="165" y="172" text-anchor="middle" fill="#6B645E" font-size="8.5">一条直的切线总会分错一个点</text><line x1="300" y1="40" x2="300" y2="150" stroke="#B8AEA2"/><line x1="300" y1="150" x2="500" y2="150" stroke="#B8AEA2"/><text class="lang-en" x="285" y="94" fill="#6B645E" font-size="8" transform="rotate(-90 285 94)">loss</text><text class="lang-zh" x="285" y="94" fill="#6B645E" font-size="8" transform="rotate(-90 285 94)">loss</text><text class="lang-en" x="400" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">loops →</text><text class="lang-zh" x="400" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">圈数 →</text><path d="M306 50 C 340 96, 380 104, 494 104" fill="none" stroke="#C93B3B" stroke-width="2.2"/><line x1="300" y1="104" x2="494" y2="104" stroke="#C99A12" stroke-width="1" stroke-dasharray="3,3"/><text class="lang-en" x="410" y="98" fill="#9A7208" font-size="8.5">stuck above 0 forever</text><text class="lang-zh" x="410" y="98" fill="#9A7208" font-size="8.5">永远卡在 0 以上</text></g></svg>
%%%

#### One small footnote — the shallow dip
One more honest line, and it's a short one. Our valley picture assumed a single clean bowl; a lumpy landscape with several dips is called [[non-convex||a bumpy landscape with more than one dip, so the lowest nearby point may not be the lowest point overall]]. Because the gradient only feels the ground under your boots, the loop can roll into a small dip, find uphill in every direction, and stop there — a [[local minimum||a spot where every direction is uphill nearby, so the loop stops, even though a deeper valley exists elsewhere]], like a hiker who settles in a roadside ditch and announces "this is the bottom!" The curve flattens, so it *looks* converged: you did find the bottom of *this* dip, and that is **not the global best**.
~~~zh
#### 一个小小的脚注 —— 那个浅坑
还有一句诚实的话，很短。我们那幅山谷的画假设了只有一个干净的碗；一片有好几个坑的坑坑洼洼的地形叫 [[non-convex（非凸）||坑坑洼洼、有不止一个坑的地形，所以附近最低的那一点不一定是整体最低的点]]。因为 gradient 只感觉得到你靴子底下的地面，循环可能滚进一个小坑里，发现每个方向都是上坡，然后就停在那儿 —— 这叫 [[local minimum（局部最小值）||一个附近每个方向都是上坡的点，所以循环会停下，尽管别处还有更深的谷]]，就像一个徒步的人在路边的沟里安顿下来，然后宣布「这就是底了！」曲线变平了，所以它*看起来*收敛了：你确实找到了*这个*坑的底，而那**不是全局最好**。
~~~

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A bumpy loss landscape with two dips. A ball has rolled into the shallower left dip and is sitting exactly at its lowest point, a local minimum, with uphill ground on both sides. The deeper right dip, the global minimum, is lower but unreached."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A bumpy landscape: the loop can stop in a shallow dip</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">坑坑洼洼的地形：循环可能停在一个浅坑里</text><path d="M24 44 C 60 44, 84 120, 120 120 C 156 120, 172 80, 208 80 C 244 80, 264 152, 300 152 C 336 152, 460 44, 500 44" fill="none" stroke="#B8AEA2" stroke-width="2.2"/><circle cx="120" cy="113" r="7" fill="#C99A12" stroke="#fff" stroke-width="1.5"/><text class="lang-en" x="120" y="100" text-anchor="middle" fill="#9A7208" font-size="8.5">🛑 stuck here (local minimum)</text><text class="lang-zh" x="120" y="100" text-anchor="middle" fill="#9A7208" font-size="8.5">🛑 卡在这（local minimum）</text><circle cx="300" cy="145" r="6" fill="none" stroke="#2D8B55" stroke-width="2"/><text class="lang-en" x="316" y="148" fill="#1a5c38" font-size="9">← global minimum</text><text class="lang-zh" x="316" y="148" fill="#1a5c38" font-size="9">← global minimum（全局最小值）</text><text class="lang-en" x="360" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">the real best (deeper), unreached</text><text class="lang-zh" x="360" y="164" text-anchor="middle" fill="#6B645E" font-size="8.5">真正最好的（更深），没走到</text><text class="lang-en" x="120" y="140" text-anchor="middle" fill="#6B645E" font-size="8">uphill both ways → the loop stops</text><text class="lang-zh" x="120" y="140" text-anchor="middle" fill="#6B645E" font-size="8">两边都是上坡 → 循环停下</text></g></svg>
%%%

Cheerful footnote to the footnote: in the enormous networks behind real models these dips turn out to be mostly shallow and rarely a practical problem. It's an honest limit, not a reason to worry — and now you know it exists.

Now let's gather the whole day onto one page.
~~~zh
给脚注再加个开心的脚注：在真实模型背后那些巨大的网络里，这些坑基本都很浅，很少成为实际问题。这是一个诚实的边界，不是需要担心的事 —— 而现在你知道它存在了。

现在把今天的全部内容收进一页。
~~~

@@@ concept id=c10 zh_tag="回顾" tag="Recap" zh_title="今天的全部内容，一页看完" title="Today in one page" zh_gotit="看完回顾了" gotit="Got the recap"
Take a breath — you just built the engine that makes every neural network learn, and you can now diagnose six ways it goes wrong from a single glance at a graph. Here's the whole day on one page: our **free-throw** drill, wearing its real training names. **Where the free-throw picture finally breaks down:** a player fixes one thing per shot; the loop nudges every knob at once, thousands of times — and that scale is the only reason it can learn things no person could.

The loop you built, in order:
- **Step 1 · Forward** — take the shot: run the input to a prediction.
- **Step 2 · Loss** — measure the miss: one number for how wrong the guess is.
- **Step 3 · Backward** — which way off: backprop hands every weight its gradient.
- **Step 4 · Update** — nudge the aim: `w ← w − lr × gradient`; the **minus** walks downhill.
- **Repeat** over **iterations** (one update, on one batch — today a batch of one example) and **epochs** (one full pass), watching the **loss curve** fall and flatten (**converge**).
- The **learning rate** is the multiplier on every nudge; the **perceptron** (1958) was the crude ancestor that corrected only on mistakes.

Here's the one picture to keep — the four steps, the dial, the curve, and where each trap strikes.
~~~zh
喘口气 —— 你刚造好了让每个 neural network 学会东西的那个引擎，而且现在你看一眼图就能诊断出它出错的六种方式。这里是今天的全部内容，写在一页上：我们的**罚球**练习，穿上它真正的训练术语。**罚球这幅画最终不成立的地方：**球员每投一次修一件事；这个循环一次挪动每一个旋钮，挪几千次 —— 而正是这个规模，让它能学会没有人学得会的东西。

你造好的这个循环，按顺序：

- **第 1 步 · forward** —— 出手：把输入跑成一个 prediction。
- **第 2 步 · loss** —— 量偏差：用一个数字说这次猜得有多错。
- **第 3 步 · backward** —— 往哪边偏了：backpropagation 把 gradient 发给每一个 weight。
- **第 4 步 · update** —— 挪准星：`w ← w − lr × gradient`；那个**减号**让你往下坡走。
- 在 **iteration（迭代）**（一次 update，在一个 batch 上 —— 今天的 batch 是一个例子）和 **epoch（轮次）**（完整过一遍）上**重复**，看着 **loss curve** 下降并变平（**converge，收敛**）。
- **learning rate（学习率）**是乘在每次挪动上的乘数；**perceptron（感知机）**（1958）是那个只在做错时才修正的粗糙祖先。

这是要留住的那一张图 —— 四个步骤、那个旋钮、那条曲线，以及每个陷阱打在哪里。
~~~

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Recap diagram. Top: the four-step loop forward, loss, backward, update in a ring with a repeat arrow, and a learning-rate dial feeding the update step. Middle: a loss curve falling and flattening at convergence. Bottom: a row listing the six traps and their one-line cures."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">The whole day: the loop, the dial, the curve, the traps</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">今天的全部：循环、旋钮、曲线、陷阱</text><rect x="16" y="26" width="86" height="26" rx="4" fill="#E7F0F5" stroke="#2A7B9B"/><text class="lang-en" x="59" y="43" text-anchor="middle" fill="#1F6280">1 forward</text><text class="lang-zh" x="59" y="43" text-anchor="middle" fill="#1F6280">1 forward</text><rect x="120" y="26" width="70" height="26" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="155" y="43" text-anchor="middle" fill="#C93B3B">2 loss</text><text class="lang-zh" x="155" y="43" text-anchor="middle" fill="#C93B3B">2 loss</text><rect x="208" y="26" width="94" height="26" rx="4" fill="#EDE9F8" stroke="#7C6DAA"/><text class="lang-en" x="255" y="43" text-anchor="middle" fill="#5E5191">3 backward</text><text class="lang-zh" x="255" y="43" text-anchor="middle" fill="#5E5191">3 backward</text><rect x="320" y="26" width="82" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="361" y="43" text-anchor="middle" fill="#1a5c38">4 update</text><text class="lang-zh" x="361" y="43" text-anchor="middle" fill="#1a5c38">4 update</text><g stroke="#B8AEA2" stroke-width="1.6" fill="#B8AEA2"><path d="M102 39 l16 0"/><polygon points="118,39 111,35 111,43"/><path d="M190 39 l16 0"/><polygon points="206,39 199,35 199,43"/><path d="M302 39 l16 0"/><polygon points="318,39 311,35 311,43"/></g><path d="M361 52 C 361 70, 59 70, 59 54" fill="none" stroke="#C99A12" stroke-width="1.6" stroke-dasharray="3,2"/><polygon points="59,54 55,62 63,62" fill="#C99A12"/><text class="lang-en" x="210" y="66" text-anchor="middle" fill="#9A7208">↻ repeat</text><text class="lang-zh" x="210" y="66" text-anchor="middle" fill="#9A7208">↻ 重复</text><rect x="418" y="26" width="86" height="26" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="461" y="40" text-anchor="middle" fill="#9A7208">learning rate</text><text class="lang-zh" x="461" y="40" text-anchor="middle" fill="#9A7208">learning rate</text><text class="lang-en" x="461" y="49" text-anchor="middle" fill="#9A7208" font-size="7.5">(× every nudge)</text><text class="lang-zh" x="461" y="49" text-anchor="middle" fill="#9A7208" font-size="7.5">（× 每一次挪动）</text><line x1="46" y1="86" x2="46" y2="140" stroke="#B8AEA2"/><line x1="46" y1="140" x2="250" y2="140" stroke="#B8AEA2"/><path d="M50 92 C 90 132, 160 138, 246 139" fill="none" stroke="#2D8B55" stroke-width="2"/><text class="lang-en" x="150" y="100" fill="#1a5c38">loss curve → converges (flat)</text><text class="lang-zh" x="150" y="100" fill="#1a5c38">loss curve → 收敛（变平）</text><text class="lang-en" x="280" y="100" fill="#6B645E">read the curve:</text><text class="lang-zh" x="280" y="100" fill="#6B645E">读这条曲线：</text><text class="lang-en" x="280" y="114" fill="#C93B3B">climbs every lap → lr too high</text><text class="lang-zh" x="280" y="114" fill="#C93B3B">每圈都爬 → lr 太高</text><text class="lang-en" x="280" y="128" fill="#2A7B9B">barely moves → lr too low</text><text class="lang-zh" x="280" y="128" fill="#2A7B9B">几乎不动 → lr 太低</text><text class="lang-en" x="280" y="142" fill="#9A7208">val loss turns up → overfit</text><text class="lang-zh" x="280" y="142" fill="#9A7208">val loss 转头向上 → overfit</text><line x1="16" y1="152" x2="504" y2="152" stroke="#E5DFD6"/><text class="lang-en" x="16" y="168" fill="#C93B3B">lr high → lower it / schedule</text><text class="lang-zh" x="16" y="168" fill="#C93B3B">lr 高 → 降低它 / 用 schedule</text><text class="lang-en" x="270" y="168" fill="#C93B3B">lr low → raise it</text><text class="lang-zh" x="270" y="168" fill="#C93B3B">lr 低 → 提高它</text><text class="lang-en" x="16" y="182" fill="#C93B3B">no reset → zero the gradients</text><text class="lang-zh" x="16" y="182" fill="#C93B3B">没清零 → 把 gradient 清零</text><text class="lang-en" x="270" y="182" fill="#C93B3B">fixed order → shuffle each epoch</text><text class="lang-zh" x="270" y="182" fill="#C93B3B">顺序固定 → 每个 epoch 打乱</text><text class="lang-en" x="16" y="196" fill="#C93B3B">too few epochs → train longer</text><text class="lang-zh" x="16" y="196" fill="#C93B3B">epoch 太少 → 多训练一会儿</text><text class="lang-en" x="270" y="196" fill="#C93B3B">too many → early stopping</text><text class="lang-zh" x="270" y="196" fill="#C93B3B">太多 → early stopping</text><text class="lang-en" x="16" y="212" fill="#6B645E">limit: one neuron can't learn XOR · GD can stop in a local dip</text><text class="lang-zh" x="16" y="212" fill="#6B645E">边界：一个 neuron 学不了 XOR · GD 会停在浅坑里</text></g></svg>
%%%

#### One cheat-sheet · the words, the knobs, and the six traps
%%% table
:: Word / trap :: In plain English :: Remember
training loop :: forward → loss → backward → update, on repeat :: repeat until the loss flattens
forward pass :: run the input through the neuron to get a prediction :: the shot
loss :: one number saying how wrong the guess is :: lower is better, 0 is perfect
backward pass :: backpropagation hands every weight its gradient :: which way was I off?
update :: `w ← w − lr × gradient` :: the MINUS walks downhill
learning rate (lr) :: the multiplier on each nudge (the step size) :: too bold overshoots, too timid crawls
schedule (decay) :: shrink the learning rate over time :: bold steps early, gentle steps late (Day 8 goes deep)
iteration vs epoch :: one update vs one full pass over the data :: many iterations make one epoch
batch :: the handful of examples behind one update :: today, one example per update
loss curve · convergence :: the loss plotted per lap; flat = mostly finished :: the instrument panel
trap · lr too high :: a step lands farther out than it started → the loss climbs every lap, then NaN :: cure: lower lr / use a schedule
trap · lr too low :: loss barely moves :: cure: raise lr
trap · forgot to reset :: gradients pile up silently, so the step grows :: cure: zero the gradients each loop
trap · fixed data order :: not shuffling data means the updates get biased by the order :: cure: shuffle each epoch
trap · too few epochs :: underfit — loss still high :: cure: train longer, watch it flatten
trap · too many epochs :: overfit — memorizes; validation loss rises :: cure: early stopping on held-out data
limit · one neuron :: can't learn XOR; plateaus above zero :: more neurons, not more epochs
limit · local minimum :: settles in a shallow dip of a non-convex landscape :: converged ≠ globally best
%%%

%%% jargon
training loop | the four-step cycle that repeats until the loss stops falling
forward pass | run the input through the neuron to get a prediction
loss | one number scoring how wrong the prediction is
backward pass | backpropagation, which hands every weight its gradient
the update | `w ← w − lr × gradient` — the one step that changes the network
learning rate (lr) | how bold each nudge is; also called the step size
schedule / decay | a plan that shrinks the learning rate as training goes on
iteration | one trip round the loop — exactly one weight update
epoch | one full pass over all the training data
batch | the handful of examples behind a single update
loss curve | the loss plotted lap by lap — your instrument panel
convergence | the loss levels off and stops improving
underfit | stopped too soon; the loss is still high
overfitting | the network memorizes the training rows instead of the pattern
early stopping | halt at the point where validation loss stops improving
local minimum | a shallow dip the loop can settle in, on a non-convex landscape
%%%

That's the whole day. Tomorrow you keep the loop and swap Step 4 for something smarter than a plain nudge — **optimizers** like Momentum and Adam, which take this same downhill idea and make it faster and steadier.
~~~zh
#### 一张速查表 · 词、旋钮，还有六个陷阱

%%% table
:: 词 / 陷阱 :: 用人话说 :: 记住
training loop（训练循环） :: forward → loss → backward → update，一直重复 :: 重复到 loss 变平
forward pass（前向传播） :: 把输入送过 neuron 得到一个 prediction :: 出手
loss（损失） :: 一个数字，说这次猜得有多错 :: 越小越好，0 是完美
backward pass（反向传播） :: backpropagation 把 gradient 发给每一个 weight :: 我往哪边偏了？
update（更新） :: `w ← w − lr × gradient` :: 那个减号让你往下坡走
learning rate（学习率，lr） :: 乘在每次挪动上的乘数（步长） :: 太大胆会冲过头，太胆小只能爬
schedule（调度 / decay 衰减） :: 让 learning rate 随时间缩小 :: 早期大胆，后期温柔（第 8 天细讲）
iteration 对 epoch :: 一次 update 对 完整过一遍数据 :: 很多个 iteration 组成一个 epoch
batch（批） :: 一次 update 背后的那一小把例子 :: 今天每次 update 用一个例子
loss curve · convergence :: 每圈画一个 loss；平了就是基本结束 :: 你的仪表盘
陷阱 · lr 太高 :: 一步落得比出发时更远 → loss 每圈都爬，最后变 NaN :: 解法：降低 lr / 用 schedule
陷阱 · lr 太低 :: loss 几乎不动 :: 解法：提高 lr
陷阱 · 忘了清零 :: gradient 无声地堆起来，于是步子越来越大 :: 解法：每一圈把 gradient 清零
陷阱 · 数据顺序固定 :: 不打乱数据，update 就会被顺序带偏 :: 解法：每个 epoch 打乱一次
陷阱 · epoch 太少 :: underfit —— loss 还很高 :: 解法：多训练一会儿，看着它变平
陷阱 · epoch 太多 :: overfit —— 在背答案；validation loss 上升 :: 解法：在留出数据上做 early stopping
边界 · 只有一个 neuron :: 学不了 XOR；停在零以上的平台 :: 加 neuron，而不是加 epoch
边界 · local minimum :: 在非凸地形的一个浅坑里安顿下来 :: 收敛 ≠ 全局最好
%%%

%%% jargon
training loop（训练循环） | 那个四步的循环，一直重复到 loss 不再下降
forward pass（前向传播） | 把输入送过 neuron 得到一个 prediction
loss（损失） | 一个数字，给「猜得有多错」打分
backward pass（反向传播） | 也就是 backpropagation，它把 gradient 发给每一个 weight
update（更新） | `w ← w − lr × gradient` —— 唯一真正改动网络的那一步
learning rate（学习率，lr） | 每次挪动有多大胆；也叫 step size（步长）
schedule / decay（调度 / 衰减） | 一个让 learning rate 随训练进行而缩小的计划
iteration（迭代） | 循环走一圈 —— 正好一次 weight 更新
epoch（轮次） | 把所有训练数据完整过一遍
batch（批） | 一次 update 背后的那一小把例子
loss curve（损失曲线） | 一圈一圈画出来的 loss —— 你的仪表盘
convergence（收敛） | loss 走平，不再变好
underfit（欠拟合） | 停得太早；loss 还很高
overfitting（过拟合） | 网络在背训练数据的行，而不是在学规律
early stopping（早停） | 在 validation loss 不再变好的那一点停下
local minimum（局部最小值） | 非凸地形上循环可能安顿下来的一个浅坑
%%%

今天就是这些。明天你保留这个循环，把第 4 步换成比普通挪动更聪明的东西 —— **optimizer（优化器）**，比如 Momentum 和 Adam，它们把同样这个下坡的想法做得更快更稳。
~~~

@@@ quiz id=quiz zh_tag="小测" tag="Quiz" zh_title="点一个答案 —— 每题都会立刻给你反馈" title="Click an answer — instant feedback on each" zh_gotit="先把四题都答完" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What are the four steps of the training loop, in order? | a:2 | loss → forward → update → backward | update → backward → loss → forward | forward pass → loss → backward pass → update | backward → update → forward → loss | fb: One lap is forward (take the shot) → loss (measure the miss) → backward (which way off?) → update (nudge the aim), then repeat. It's the free-throw drill: try, measure, correct, again.
q: In the update rule w ← w − lr × gradient, what does the MINUS sign do, and what does lr control? | a:1 | Minus makes training faster; lr sets how many layers | The gradient points uphill, so minus steps the weight downhill (toward less loss); lr is the multiplier on that step — how bold each nudge is | Minus deletes bad weights; lr is the number of epochs | Minus is just notation with no effect; lr is the loss value | fb: The gradient points UPHILL (toward more loss), so subtracting it walks the weight downhill toward less loss. The learning rate multiplies that step: too bold and it overshoots until the loss diverges, too timid and training crawls.
q: Your loss climbs higher on every single lap and finally blows up to NaN. What's wrong, and what's the fix? | a:2 | The learning rate is too low; fix: lower it more | You have too few epochs; fix: shuffle the data | The learning rate is too high — each step overshoots by more than twice the distance left, so the weight lands farther out every lap and the loss climbs; fix: lower the learning rate (or use a schedule that shrinks it over time) | The gradients weren't reset; fix: train for more epochs | fb: A loss that grows lap after lap means each step carries the weight past the bottom by MORE than it started away from it, so the weight bounces across the bottom and lands farther out every time. (Crossing the bottom by a little is harmless — the ball zig-zags but still settles.) That's the classic too-high learning rate: lower it, or use a schedule (decay) so early steps are bold for speed and late steps gentle to settle.
q: Training loss keeps falling but validation loss starts rising. What is happening, and what's the remedy? | a:1 | The learning rate is too low; remedy: raise it | The network is overfitting — memorizing the training data instead of the pattern; remedy: early stopping (halt when validation loss stops improving) | The gradients are piling up; remedy: zero them each loop | The neuron can't represent the target; remedy: shuffle the data | fb: When the two curves fork — training loss down, validation loss up — the network has stopped learning the pattern and started memorizing the training examples. That's overfitting. Early stopping halts right at the fork, keeping the version that generalizes best.
%%%
~~~zh
四个快问题，每题都会立刻给你反馈 —— 四题都答完就算今天完成。（如果有一题卡住你，那是一个往回滚的提示，不是不及格。）

%%% quiz
q: training loop 的四个步骤，按顺序是什么？ | a:2 | loss → forward → update → backward | update → backward → loss → forward | forward pass → loss → backward pass → update | backward → update → forward → loss | fb: 一圈是 forward（出手）→ loss（量偏差）→ backward（往哪边偏了？）→ update（挪准星），然后重复。就是那个罚球练习：试、量、改、再来。
q: 在 update 规则 w ← w − lr × gradient 里，那个减号做什么，lr 又控制什么？ | a:1 | 减号让训练更快；lr 决定有几层 | gradient 指上坡，所以减号让 weight 往下坡走（走向更小的 loss）；lr 是乘在这一步上的乘数 —— 每次挪动有多大胆 | 减号删掉不好的 weight；lr 是 epoch 的个数 | 减号只是写法，没有作用；lr 就是 loss 的值 | fb: gradient 指的是**上坡**（loss 更大的方向），所以减掉它就让 weight 往下坡走，走向更小的 loss。learning rate 乘在这一步上：太大胆会冲过头直到 loss 发散，太胆小训练就只能爬。
q: 你的 loss 每一圈都爬得更高，最后炸成 NaN。哪里出错了，怎么修？ | a:2 | learning rate 太低；解法：再降低一些 | epoch 太少；解法：把数据打乱 | learning rate 太高 —— 每一步冲过头的距离超过剩余距离的两倍，所以 weight 每圈都落得更远，loss 就往上爬；解法：降低 learning rate（或者用一个随时间缩小它的 schedule） | gradient 没清零；解法：训练更多 epoch | fb: loss 一圈一圈变大，意味着每一步把 weight 带过谷底的距离**超过**它原来离谷底的距离，所以 weight 在谷底两侧弹，而且每次落得更远。（稍微越过谷底是无害的 —— 球会来回折，但还是会停下。）这就是那个经典的 learning rate 太高：把它降下来，或者用一个 schedule（decay），让早期的步子大胆求快、后期的步子温柔好停。
q: 训练 loss 一直在降，但 validation loss 开始上升。这是在发生什么，处方是什么？ | a:1 | learning rate 太低；处方：提高它 | 网络在 overfitting —— 在背训练数据而不是学规律；处方：early stopping（validation loss 不再变好就停） | gradient 堆起来了；处方：每一圈清零 | 这个 neuron 表示不了目标；处方：把数据打乱 | fb: 两条曲线分叉的时候 —— 训练 loss 往下，validation loss 往上 —— 网络已经不在学规律，开始背训练例子了。这就是 overfitting。early stopping 正好在分叉处停手，留下推广得最好的那个版本。
%%%
~~~

@@@ produce id=produce zh_tag="动手做" tag="Produce" zh_title="跑一个真的 training loop，看着 loss 掉下来" title="Run a real training loop and watch the loss fall" zh_gotit="做完了" gotit="Done"
Time to run the engine yourself. You'll build the full four-step loop for one neuron and **watch** the loss drop, lap after lap, then poke the learning rate to *feel* the too-bold and too-timid settings. **Predict first:** with a sensible learning rate, will the loss fall smoothly toward zero, or jump around? Then set the learning rate way too high and predict again — fall, or blow up? Run it and **watch** which prediction was right. Pick one path.

#### Option A · write it yourself
Create `sessions/m02-the-neuron/day-06-training-loop/experiment.py`. (1) Set up one neuron learning a simple target: `x = 2.0`, `target = 1.0`, start `w = 0.0` and `b = 0.0`, learning rate `lr = 0.01`. (2) Write the loop for 50 iterations: **forward** `pred = w*x + b`; **loss** `L = (pred - target)**2`; **backward** `grad_w = 2*(pred-target)*x` and `grad_b = 2*(pred-target)`; **update** `w = w - lr*grad_w` and `b = b - lr*grad_b`. Print the loss every 10 laps and **watch** it fall. (3) Now rerun with `lr = 1.5` and **observe** the prediction flip sign every lap while the loss explodes, then with `lr = 0.00005` and **observe** it barely move. Run with `python3 sessions/m02-the-neuron/day-06-training-loop/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 2 Day 6 artifact.

Create sessions/m02-the-neuron/day-06-training-loop/experiment.py that, with a comment on each step:
1. Trains ONE neuron with a full training loop. Setup: x=2.0, target=1.0, w=0.0, b=0.0, lr=0.01.
2. Loops 50 iterations doing the four steps: forward pred=w*x+b; loss L=(pred-target)**2; backward grad_w=2*(pred-target)*x and grad_b=2*(pred-target); update w=w-lr*grad_w, b=b-lr*grad_b. Store the loss each loop and print it every 10 loops so we can watch it fall: it should start at 1.0 and slide down to roughly 0.12, 0.015, 0.0018, 0.0002 — a steep drop then a gentle glide, flattening near zero (convergence).
3. Re-runs the same loop twice more to show the learning-rate failures: once with lr=1.5 (print the prediction as well as the loss, and watch the PREDICTION flip sign every loop — 0, 15, -195, 2745 — while the LOSS climbs to an astronomical number: divergence) and once with lr=0.00005 (watch the loss go from 1.000 to only about 0.95 in 50 loops — crawling). Print the final loss for each lr and a one-line comment naming the failure.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (the loop, learning)
- With `lr = 0.01` the loss starts at `1.000000` and the every-10-laps prints read roughly `0.1216 → 0.0148 → 0.0018 → 0.00022`, ending near `0.00003`. That's the healthy shape from concept 5: a **steep drop, a gentle glide, then flat** near zero — converging. **Watch** how the drop per lap shrinks as the gap shrinks, all on its own, with `lr` untouched.
- With `lr = 1.5` the **prediction flips sign every lap while the loss explodes** — the guess swings `0 → 15 → -195 → 2745 → …` past the target and back, and because the loss is that miss squared it only ever grows: by the last print it is an astronomical number (around `10¹¹²`), and if you keep looping it becomes `inf` and then `NaN`. That's the too-bold setting, live: each step lands farther from the bottom than it started. The cure is a smaller `lr`.
- With `lr = 0.00005` the loss goes from `1.000` to only about `0.952` in 50 laps — it **barely moves**. That's the too-timid setting; the cure is a bigger `lr`. Seeing all three back to back is exactly how engineers pick a learning rate by eye.

!!! c-info 📓
<b>5-minute research log:</b> before you close the tab, write three lines in `sessions/m02-the-neuron/day-06-training-loop/log.md`: (1) the four steps of the loop, in order; (2) what the learning rate multiplies, and what happens if it's too high or too low; (3) one trap from today (reset, shuffle, underfit, overfit, or a capability limit) and its one-line cure.
!!!
~~~zh
该你自己跑这个引擎了。你会为一个 neuron 造出完整的四步循环，**看着** loss 一圈一圈掉下来，然后去戳 learning rate，*感受*太大胆和太胆小这两个设置。**先预测：**用一个合理的 learning rate，loss 会平滑地降向零，还是跳来跳去？然后把 learning rate 设得高得离谱，再预测一次 —— 下降，还是炸开？跑一下，**看**哪个预测对了。选一条路。

#### 路线 A · 自己写
新建 `sessions/m02-the-neuron/day-06-training-loop/experiment.py`。(1) 让一个 neuron 学一个简单目标：`x = 2.0`，`target = 1.0`，起始 `w = 0.0`、`b = 0.0`，learning rate `lr = 0.01`。(2) 写一个 50 次 iteration 的循环：**forward** `pred = w*x + b`；**loss** `L = (pred - target)**2`；**backward** `grad_w = 2*(pred-target)*x` 和 `grad_b = 2*(pred-target)`；**update** `w = w - lr*grad_w` 和 `b = b - lr*grad_b`。每 10 圈打印一次 loss，**看着**它下降。(3) 现在用 `lr = 1.5` 再跑一次，**观察** prediction 每圈变一次正负号，同时 loss 炸开；再用 `lr = 0.00005` 跑一次，**观察**它几乎不动。用 `python3 sessions/m02-the-neuron/day-06-training-loop/experiment.py` 运行。

#### 路线 B · 让 Claude 写，然后你读
把下面这段提示词复制回 Claude Code。它会让 Claude Code 新建文件、写代码，并且帮你跑起来。

%%% prompt
帮我做 Module 2 第 6 天的产出。

新建 sessions/m02-the-neuron/day-06-training-loop/experiment.py，每一步都写注释：
1. 用一个完整的 training loop 训练一个 neuron。设置：x=2.0, target=1.0, w=0.0, b=0.0, lr=0.01。
2. 循环 50 次 iteration，做四个步骤：forward pred=w*x+b；loss L=(pred-target)**2；backward grad_w=2*(pred-target)*x 和 grad_b=2*(pred-target)；update w=w-lr*grad_w, b=b-lr*grad_b。每圈存下 loss，每 10 圈打印一次，让我们看着它下降：它应该从 1.0 开始，一路滑到大约 0.12、0.015、0.0018、0.0002 —— 先陡降，再缓滑，在接近零处变平（convergence）。
3. 再把同样的循环跑两遍，展示 learning rate 的两种失败：一次用 lr=1.5（把 prediction 和 loss 都打印出来，看 PREDICTION 每圈变一次正负号 —— 0, 15, -195, 2745 —— 而 LOSS 爬到一个天文数字：发散），一次用 lr=0.00005（看 loss 在 50 圈里只从 1.000 走到大约 0.95 —— 在爬）。每个 lr 都打印最终的 loss，并用一行注释说出这是哪种失败。
然后运行它，把输出粘在文件底部当注释。
%%%

#### 你应该看到什么（这个循环，在学）
- 用 `lr = 0.01`，loss 从 `1.000000` 开始，每 10 圈的打印大致读作 `0.1216 → 0.0148 → 0.0018 → 0.00022`，最后停在 `0.00003` 附近。这就是 concept 5 里那个健康的形状：**陡降、缓滑，然后**在接近零处**变平** —— 收敛了。**注意看**每圈的降幅怎么随着差距变小而自己变小，而 `lr` 一个字都没动。
- 用 `lr = 1.5`，**prediction 每圈变一次正负号，同时 loss 炸开** —— 猜测在 `0 → 15 → -195 → 2745 → …` 之间越过目标又弹回来，而因为 loss 是这个偏差的平方，它只会一直变大：到最后一次打印时它是一个天文数字（大约 `10¹¹²`），如果你继续循环，它会变成 `inf`，然后变成 `NaN`。这就是太大胆的设置，活的：每一步落得都比出发时离谷底更远。解药是更小的 `lr`。
- 用 `lr = 0.00005`，loss 在 50 圈里只从 `1.000` 走到大约 `0.952` —— 它**几乎不动**。这是太胆小的设置；解药是更大的 `lr`。把这三种连着看一遍，正是工程师用眼睛挑 learning rate 的方法。

!!! c-info 📓
<b>5 分钟研究笔记：</b>关掉页面前，用自己的话在 `sessions/m02-the-neuron/day-06-training-loop/log.md` 里写三行：(1) 循环的四个步骤，按顺序；(2) learning rate 乘的是什么，太高或太低会发生什么；(3) 今天的一个陷阱（清零、打乱、underfit、overfit，或者一个能力边界）以及它一行的解法。
!!!
~~~

@@@ fin
