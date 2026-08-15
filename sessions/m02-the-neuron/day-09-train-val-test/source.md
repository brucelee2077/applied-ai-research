---
quest_id: wf3-d06-split
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 9 — Train / Val / Test & Overfitting"
module_label: "Module 2 · Train · Day 9"
zh_module_label: "第 2 模块 · 训练 · 第 9 天"
title: "Train / Val / Test & Overfitting"
zh_title: "训练 / 验证 / 测试与 overfitting"
subtitle: "Did the Model Learn, or Just Memorize?"
zh_subtitle: "模型是真学会了，还是只是背下来了？"
brand_sub: "Foundations · M2 Day 9"
spine: "exam vs answer-key"
zh_spine: "考卷和答案本"
nav_prev_href: "../day-08-learning-rate/lesson.html"
nav_prev_label: "Learning-Rate Intuition"
zh_nav_prev_label: "学习率的直觉"
nav_next_href: "../review.html"
nav_next_label: "Review Gate"
zh_nav_next_label: "复习关卡"
fin_title: "Module 2 · Day 9 complete! 🏆"
zh_fin_title: "第 2 模块 · 第 9 天 完成！🏆"
fin_body: "Nice work — you can now tell <b>learning</b> from <b>memorizing</b>: train on one set, watch a held-out validation set to catch overfitting (the gap between train and val loss), and keep the test set sealed like a locked answer key. Early stopping and regularization keep the model honest.<br>That completes the days of Module 2 — up next is the <b>Review Gate</b>."
zh_fin_body: "做得好 —— 你现在能分清<b>学会</b>和<b>背下来</b>了：在 training set 上学，用留出来的 validation set 盯住 overfitting，也就是 train 和 val 的 loss 之间那道缝。test set 像锁起来的答案本一样封着，只开一次。early stopping 和 regularization 让模型保持诚实。<br>第 2 模块的每一天到这里全部走完 —— 下一站是<b>复习关卡</b>。"
notebook_yardstick: null
coverage_topics:
  - {topic: why split at all, keywords: [judge on data it has not seen, judged on data, generalize, generalization, not just memorize, dishonest]}
  - {topic: training set job, keywords: [training set, learns on, fits its weights, the split the model learns, practice questions]}
  - {topic: validation set job, keywords: [validation set, during development, check progress, pick, which settings, without touching the test]}
  - {topic: test set job, keywords: [test set, touched once, locked away, honest final score, sealed, only at the end]}
  - {topic: three-way roles, keywords: [only train updates, validation guides, test is a one-time, three splits, three jobs]}
  - {topic: two-way ancestor, keywords: [train and test only, two-way split, tuning against the test, contaminates, motivates a separate validation]}
  - {topic: overfitting cause and symptom, keywords: [overfitting, memorizes the training data, memorized, high train, low validation, aces practice but fails]}
  - {topic: train val gap diagnostic, keywords: [gap between train and validation, train-validation gap, gap widens, watch the gap, see overfitting]}
  - {topic: test leakage peeking, keywords: [peeking, looking at the test score, tune on the test, keep the test untouched, use it only once]}
  - {topic: data leakage across splits, keywords: [same rows in more than one, duplicate rows, split first, preprocess after splitting, leaks across splits]}
  - {topic: shuffle before splitting, keywords: [shuffle before splitting, ordered data, sorted, unrepresentative slice]}
  - {topic: cross-validation remedy, keywords: [cross-validation, k-fold, unlucky split, rotate which slice, average the scores]}
  - {topic: split proportions, keywords: [70/15/15, 80/10/10, no ratio is sacred, bigger data, smaller fraction]}
  - {topic: limit diagnoses not creates, keywords: [diagnoses generalization, it does not create it, does not cause it, only reveals, reveals whether learning transferred]}
  - {topic: limit same distribution, keywords: [same distribution, honest only for data like, says nothing about a shifted future, distribution shift]}
---

@@@ hero
@lede Think about the practice sheet your teacher hands out before a school test. It has the answers printed on the back, so you can learn from it. The *real* exam has different questions — and the answer key stays locked in a drawer until grading day. Why not just let you practise on the real exam itself? Because then you would not learn the subject at all. You would memorize those exact questions, and your grade would be a lie. Here is the surprising part: a neural network falls into exactly the same trap. Grade it on the very examples it studied and you cannot tell whether it *learned the pattern* or just *memorized the answers*. So we split our data into a practice pile, a check-up pile, and a sealed exam. That one habit stands behind every model you have ever used — the face-unlock that knows your face, the app that guesses which video you want next, ChatGPT answering a question nobody ever typed before. Today you get to run that split with your own hands and catch a model in the act of memorizing.
@goal Together we will build a real gut feel for splitting data: why we never grade a model on what it studied, what each of the three piles (train, validation, test) is actually *for*, how the gap between two of them lets you *see* overfitting happen live on a slider, and the sneaky ways a split can lie to you — a peek at the sealed exam, the same row landing in two piles, an unlucky slice — each with its one-move fix. Every idea arrives as a picture first, and there are three things here you can drag with your own finger. No heavy math, just intuition you can steer.
%%% warmup
q: Yesterday you met the learning rate. In "new weight = old weight − learning rate × gradient", what does the learning rate control? | a:1 | which way is downhill | how far you hop each step | how many weights there are | the value of the loss itself | concept: learning-rate | fb: The learning rate is the step size — how far the weight moves each hop. The gradient picks the direction.
q: You set the learning rate too big and the loss bounced around, then shot up to NaN. What was going on? | a:0 | each hop overshoots the bottom and grows, so training diverges | the hops were too tiny, so it just crawled | the model finished training early | the gradient was exactly zero | concept: lr-too-big | fb: Too-big steps jump past the bottom and grow each update until the numbers blow up to NaN. The fix is to lower the rate.
%%%
@zh_lede 想想老师在考试前发给你的那张练习卷。背面印着答案，所以你能照着学。*真正的*考卷题目不一样。答案本锁在抽屉里，一直到批卷那天才拿出来。为什么不干脆让你直接在真考卷上练呢？因为那样你根本学不到这门课。你会把那几道题一个字一个字背下来，你的分数就是假的。让人意外的地方来了：神经网络会掉进完全一样的坑。拿它学过的那些例子给它打分，你分不清它是*学到了规律*，还是只是*背下了答案*。所以我们把数据分成三堆：一堆用来练，一堆用来查，还有一堆是封起来的考卷。这一个习惯，站在你用过的每一个模型背后。认得你的脸的手机解锁、猜你下一条想看什么视频的软件、回答从来没有人打过的问题的 ChatGPT，全都靠它。今天你可以亲手分一次，还能当场抓住一个模型在背答案。
@zh_goal 我们会一起把「分数据」这件事的手感建起来：为什么绝不用模型学过的题给它打分，三堆（train、validation、test）各自到底是*干什么*的，其中两堆之间的那道缝怎么让你在滑块上*看见* overfitting 正在发生，还有 split 悄悄骗你的那些方式 —— 偷看封好的考卷、同一行数据落进两堆、抽到运气差的一块 —— 每一个都配上一招就能解决的办法。每个想法都先给一张图，这里有三样东西你能用手指拖。没有重数学，只有你能拿来掌舵的直觉。

~~~zh
%%% warmup
q: 昨天你见了 learning rate。在「新 weight = 老 weight − learning rate × gradient」里，learning rate 管的是什么？ | a:1 | 哪边是下坡 | 每一步跨多远 | 一共有多少个 weight | loss 本身的数值 | concept: learning-rate | fb: learning rate 就是步子的大小 —— 每一跳 weight 挪多远。方向是 gradient 挑的。
q: 你把 learning rate 设得太大，loss 先是乱弹，然后一路冲上 NaN。当时发生了什么？ | a:0 | 每一跳都迈过谷底、而且越迈越大，训练就炸了 | 跨度太小，所以只是在爬 | 模型提前训练完了 | gradient 刚好是零 | concept: lr-too-big | fb: 太大的步子会跳过谷底，每次更新还变得更大，最后数字炸成 NaN。办法是把速率调小。
%%%
~~~

@@@ concept id=c1 zh_tag="为什么要分开" tag="Why split?" zh_title="为什么绝不用模型学过的题给它打分" title="Why we never grade a model on what it studied" zh_gotit="懂了为什么要分" gotit="Got why we split"
Let us start with the one question this whole day answers. Imagine a friend who "studied" for a history exam by memorizing the answers to *last year's exact paper*. On that old paper they score 100%. Hand them *this year's* paper — fresh questions, same topic — and they fall apart.

Did they learn history? No. They learned *those answers*. A neural network can do exactly the same thing: score perfectly on the examples it trained on, and be useless on anything new. The word for doing well on *fresh, unseen* examples is [[generalization||doing well on fresh examples the model never saw during training — the real goal, as opposed to just memorizing the ones it studied]], and how well a model can generalize is the only thing we actually care about.
~~~zh
我们先从这一整天要回答的那一个问题开始。想象一个朋友「复习」历史考试的办法是：把*去年那张原题卷*的答案背下来。在那张旧卷子上，他考 100 分。把*今年*的卷子给他 —— 新题目，同一门课 —— 他就垮了。

他学到历史了吗？没有。他学到的是*那些答案*。神经网络能做出完全一样的事：在它训练过的那些例子上分数满满，碰到新东西就毫无用处。在*没见过的新*例子上做得好，这件事有个名字，叫 [[generalization（泛化）||在模型训练时从没见过的新例子上也做得好 —— 这才是真正的目标，而不是把学过的那些背下来]]。一个模型 generalization 的能力，是我们真正唯一在意的东西。
~~~

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two students side by side. On the left, a student holds last year's exam paper and scores 100 percent by memorizing it, but scores badly on a fresh new paper. On the right, a student who learned the topic scores well on both the old and the new paper. A caption reads memorizing the paper is not learning the subject."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Memorizing the paper is NOT learning the subject</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">把卷子背下来，不等于学会这门课</text><rect x="20" y="34" width="230" height="158" rx="8" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="135" y="52" text-anchor="middle" fill="#C93B3B" font-size="10">🙈 memorized last year's paper</text><text class="lang-zh" x="135" y="52" text-anchor="middle" fill="#C93B3B" font-size="10">🙈 背下了去年的卷子</text><rect x="46" y="66" width="70" height="90" rx="4" fill="#fff" stroke="#C93B3B"/><text class="lang-en" x="81" y="112" text-anchor="middle" fill="#C93B3B" font-size="8">old paper</text><text class="lang-zh" x="81" y="112" text-anchor="middle" fill="#C93B3B" font-size="8">旧卷子</text><text x="81" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">100% ✓</text><rect x="150" y="66" width="70" height="90" rx="4" fill="#fff" stroke="#C93B3B"/><text class="lang-en" x="185" y="112" text-anchor="middle" fill="#C93B3B" font-size="8">new paper</text><text class="lang-zh" x="185" y="112" text-anchor="middle" fill="#C93B3B" font-size="8">新卷子</text><text x="185" y="150" text-anchor="middle" fill="#C93B3B" font-size="9">42% ✗</text><rect x="270" y="34" width="230" height="158" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="385" y="52" text-anchor="middle" fill="#1a5c38" font-size="10">🧠 learned the topic</text><text class="lang-zh" x="385" y="52" text-anchor="middle" fill="#1a5c38" font-size="10">🧠 学会了这门课</text><rect x="296" y="66" width="70" height="90" rx="4" fill="#fff" stroke="#2D8B55"/><text class="lang-en" x="331" y="112" text-anchor="middle" fill="#1a5c38" font-size="8">old paper</text><text class="lang-zh" x="331" y="112" text-anchor="middle" fill="#1a5c38" font-size="8">旧卷子</text><text x="331" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">96% ✓</text><rect x="400" y="66" width="70" height="90" rx="4" fill="#fff" stroke="#2D8B55"/><text class="lang-en" x="435" y="112" text-anchor="middle" fill="#1a5c38" font-size="8">new paper</text><text class="lang-zh" x="435" y="112" text-anchor="middle" fill="#1a5c38" font-size="8">新卷子</text><text x="435" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">93% ✓</text></g></svg>
%%%

**What the exam picture gets right:** grading someone on the paper they memorized tells you nothing — only *fresh* questions reveal the truth.

**Where it breaks down:** a student *knows* when they are cramming trivia and could choose to stop. A model has no such awareness. It will happily memorize every example unless we design the check that catches it.

#### So here is the rule that fixes everything
**A model must be judged on data it has NOT seen, or the judgement is dishonest.** That single sentence is the *exam-vs-answer-key* idea at the heart of today — and the reason we hide some examples away instead of letting the model study all of them. We want a model that can generalize, not just memorize.

Let us walk the trick and the cure, one rung at a time — it takes three moves.

%%% steps
step: the memorizer's shortcut
why: storing "this exact question → this exact answer" is *easier* for a flexible model than learning the real pattern, so it takes the shortcut whenever we let it
step: the fake grade
why: therefore a score on the studied examples measures *storage*, not understanding — it is like a chef tasting a dish they cooked themselves, of course they love it
step: the honest check
why: that's why we keep a few examples hidden during learning and grade only on those — a plate the chef has never tasted, which is the only score worth trusting
%%%

%%% insight
Here is why this bites, and it is not a school-exercise worry. Almost every embarrassing "the demo was amazing, production was terrible" story in machine learning starts right here — a beautiful number measured on data the model had already seen. Your phone's face-unlock, your spam filter, the video recommendations you scroll past: every one of them had to prove itself on faces, emails and clicks it had never met. Learning to distrust the other kind of number is a genuine superpower, and you are three minutes into owning it.
%%%

So far: one rule, one reason. Now let us *prove* it with the smallest possible cheater — a model that does nothing but memorize.

%%% demo id=lookup label="reveal — the world's laziest cheating model"
predict: this "model" is just a lookup table: it stores every training question with its answer and repeats it back. It never learns any pattern at all. Guess its two scores — on the 8 questions it stored, and on 4 fresh ones it has never seen.
code: table = {q: a for q, a in train_pairs}      # 8 stored pairs — zero learning
code: score(table, train_pairs)                   # graded on what it studied
code: score(table, fresh_pairs)                   # graded on 4 unseen questions
out: score on the 8 STUDIED questions : 8/8  = 100%   🏆
out: score on the 4 FRESH   questions : 1/4  =  25%   (1 lucky guess)
take: <b>A perfect training score proves nothing.</b> This thing has no idea what the pattern is, yet it looks flawless on the studied pile — so a training score can never tell learning apart from memorizing. Only the fresh pile talks. That is the entire reason for everything else today.
%%%

Victory lap: you just met the single habit that separates people whose models work from people whose models only *look* like they work. Everything ahead is about carving out those hidden examples wisely — and spotting every sneaky way it can go wrong.
~~~zh
**考卷这张图对在哪里：** 拿一个人背下来的那张卷子给他打分，什么也说明不了 —— 只有*新*题目才露出真相。

**它在哪里不成立：** 学生*知道*自己在死记零碎的东西，他可以选择停下来。模型没有这种自觉。除非我们设计出抓它的那道检查，它会开心地把每一个例子都背下来。

#### 于是有了那条能修好一切的规则
**一个模型必须用它没见过的数据来评判，不然这个评判就是不诚实的。** 这一句话，就是今天核心的那个*考卷和答案本*的想法 —— 也是我们为什么要藏起一部分例子，而不是让模型把它们全学一遍。我们要的是一个能 generalization 的模型，不是一个会背的模型。

我们把这个花招和它的解药一级一级走一遍 —— 一共三步。

%%% steps
step: 背题的人走的捷径
why: 把「这道题 → 这个答案」原样存下来，对一个很灵活的模型来说比学真正的规律*更省事*，所以只要我们允许，它就走这条捷径
step: 假的分数
why: 因此，学过的例子上的分数量的是*存储*，不是理解 —— 就像厨师尝自己做的菜，他当然觉得好吃
step: 诚实的检查
why: 这就是为什么我们在学的时候藏起几个例子，只用这几个来打分 —— 一盘厨师从没尝过的菜，那才是唯一值得相信的分数
%%%

%%% insight
说说这件事为什么会咬人。它不是课堂练习里的小担心。机器学习里几乎每一个尴尬的故事都从这里开始 —— 「演示的时候特别棒，上线之后特别糟」。根子是一个漂亮的数字，量在模型早就见过的数据上。你手机的人脸解锁、你的垃圾邮件过滤、你划过去的那些视频推荐，每一个都必须在它从没见过的脸、邮件和点击上证明自己。学会不相信另一种数字，是一个真正的超能力。你才花了三分钟，就已经开始拥有它。
%%%

到这里：一条规则，一个理由。现在我们用最小的一个作弊者来*证明*它 —— 一个除了背什么都不做的模型。

%%% demo id=lookupzh label="揭晓 —— 世界上最懒的作弊模型"
predict: 这个「模型」只是一张查找表：它把每一道训练题连着答案存起来，再原样念回去。它完全不学任何规律。猜猜它的两个分数 —— 在它存过的 8 道题上，和在 4 道它从没见过的新题上。
code: table = {q: a for q, a in train_pairs}      # 存下来的 8 对 —— 零学习
code: score(table, train_pairs)                   # 用它学过的东西打分
code: score(table, fresh_pairs)                   # 用 4 道没见过的题打分
out: score on the 8 STUDIED questions : 8/8  = 100%   🏆
out: score on the 4 FRESH   questions : 1/4  =  25%   (1 lucky guess)
take: <b>满分的训练分数什么都证明不了。</b>这个东西根本不知道规律是什么，可它在学过的那一堆上一个错都没有 —— 所以训练分数永远分不出「学会」和「背下来」。只有那堆新题会说话。今天剩下的所有内容，理由都在这里。
%%%

胜利一圈：你刚刚见到的这一个习惯，把「模型真的能用」的人和「模型只是*看起来*能用」的人分开了。后面的一切，都是在讲怎么聪明地划出那些藏起来的例子 —— 以及怎么认出它悄悄出错的每一种方式。
~~~

@@@ concept id=c2 zh_tag="三堆牌" tag="Three piles" zh_title="三个 split —— 每一堆是干什么的" title="The three splits — and what each one is for" zh_gotit="懂了这三堆" gotit="Got the three piles"
So we hide some examples. *How many* hidden piles, and what is each one's job? The clean answer is **three**, and your own school year already taught you all three.

Picture three stacks of question cards on a desk. The **practice sheet** — practice questions with the answers on the back, the ones you study from. The **weekly quiz** — small, taken during the week, to see whether your studying is working and to decide what to change. The **final exam** — sealed in a drawer, opened once at the very end for your real grade.

Those three stacks have names: the [[training set||the pile the model actually learns on — it adjusts, or fits, its weights to these examples]], the [[validation set||a held-out pile checked DURING development to track progress and pick settings, without ever touching the final exam]], and the [[test set||a pile sealed away and scored only ONCE at the very end, to get one honest final number]]. **Three splits, three jobs.**
~~~zh
所以我们藏起一部分例子。藏起来的堆要*几*堆，每一堆的活是什么？干净的答案是**三**堆。你自己的一个学年，早就把这三堆都教过你了。

想象桌上有三叠题卡。**练习卷** —— 背面印着答案的练习题，你就是照着它学的。**周测** —— 小小的一张，在这一周里做，看看你的学法有没有用，再决定要改什么。**期末考** —— 封在抽屉里，最后才拆开一次，给你真正的成绩。

这三叠有名字：[[training set（训练集）||模型真正在上面学的那一堆 —— 它照着这些例子调整自己的 weight]]、[[validation set（验证集）||留出来的一堆，在开发*过程中*拿来看进展、挑设置，从不碰那张期末考卷]]，还有 [[test set（测试集）||封起来的一堆，只在最后打分*一次*，拿到一个诚实的最终数字]]。**三个 split，三份活。**
~~~

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Three stacks of cards side by side. The first, labelled training set, is the practice sheet the model studies with answers shown. The second, labelled validation set, is a weekly quiz used to check progress and pick settings during the week. The third, labelled test set, is a sealed final exam with a lock, opened only once at the end. Arrows show train comes first, then validation during study, then test at the end."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Three piles of cards · three different jobs</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">三叠题卡 · 三份不一样的活</text><rect x="18" y="36" width="150" height="150" rx="8" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="93" y="56" text-anchor="middle" fill="#1F6280" font-size="10">📚 TRAINING</text><text class="lang-zh" x="93" y="56" text-anchor="middle" fill="#1F6280" font-size="10">📚 训练 TRAINING</text><rect x="44" y="70" width="98" height="20" rx="3" fill="#fff" stroke="#2A7B9B"/><rect x="50" y="82" width="98" height="20" rx="3" fill="#fff" stroke="#2A7B9B"/><rect x="56" y="94" width="98" height="20" rx="3" fill="#fff" stroke="#2A7B9B"/><text class="lang-en" x="93" y="140" text-anchor="middle" fill="#1F6280" font-size="8">practice sheet</text><text class="lang-zh" x="93" y="140" text-anchor="middle" fill="#1F6280" font-size="8">练习卷</text><text class="lang-en" x="93" y="156" text-anchor="middle" fill="#6B645E" font-size="8">model LEARNS here</text><text class="lang-zh" x="93" y="156" text-anchor="middle" fill="#6B645E" font-size="8">模型在这里学</text><text class="lang-en" x="93" y="172" text-anchor="middle" fill="#6B645E" font-size="8">(answers shown)</text><text class="lang-zh" x="93" y="172" text-anchor="middle" fill="#6B645E" font-size="8">（答案是给你看的）</text><rect x="186" y="36" width="148" height="150" rx="8" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="260" y="56" text-anchor="middle" fill="#9A7208" font-size="10">📝 VALIDATION</text><text class="lang-zh" x="260" y="56" text-anchor="middle" fill="#9A7208" font-size="10">📝 验证 VALIDATION</text><rect x="216" y="76" width="88" height="22" rx="3" fill="#fff" stroke="#C99A12"/><text class="lang-en" x="260" y="91" text-anchor="middle" fill="#9A7208" font-size="8">weekly quiz</text><text class="lang-zh" x="260" y="91" text-anchor="middle" fill="#9A7208" font-size="8">周测</text><text class="lang-en" x="260" y="128" text-anchor="middle" fill="#9A7208" font-size="8">CHECK progress,</text><text class="lang-zh" x="260" y="128" text-anchor="middle" fill="#9A7208" font-size="8">看进展、</text><text class="lang-en" x="260" y="144" text-anchor="middle" fill="#9A7208" font-size="8">PICK settings</text><text class="lang-zh" x="260" y="144" text-anchor="middle" fill="#9A7208" font-size="8">挑设置</text><text class="lang-en" x="260" y="164" text-anchor="middle" fill="#6B645E" font-size="8">during study</text><text class="lang-zh" x="260" y="164" text-anchor="middle" fill="#6B645E" font-size="8">在学习过程中</text><rect x="352" y="36" width="150" height="150" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="427" y="56" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 TEST</text><text class="lang-zh" x="427" y="56" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 测试 TEST</text><rect x="388" y="74" width="78" height="52" rx="3" fill="#fff" stroke="#2D8B55"/><text x="427" y="96" text-anchor="middle" font-size="16">🔒</text><text class="lang-en" x="427" y="116" text-anchor="middle" fill="#1a5c38" font-size="8">sealed exam</text><text class="lang-zh" x="427" y="116" text-anchor="middle" fill="#1a5c38" font-size="8">封好的考卷</text><text class="lang-en" x="427" y="146" text-anchor="middle" fill="#1a5c38" font-size="8">opened ONCE</text><text class="lang-zh" x="427" y="146" text-anchor="middle" fill="#1a5c38" font-size="8">只拆一次</text><text class="lang-en" x="427" y="162" text-anchor="middle" fill="#6B645E" font-size="8">at the very end</text><text class="lang-zh" x="427" y="162" text-anchor="middle" fill="#6B645E" font-size="8">在最后</text><path d="M168 200 L 186 200" stroke="#6B645E" stroke-width="1.4"/><polygon points="186,200 178,196 178,204" fill="#6B645E"/><path d="M334 200 L 352 200" stroke="#6B645E" stroke-width="1.4"/><polygon points="352,200 344,196 344,204" fill="#6B645E"/><text class="lang-en" x="260" y="204" text-anchor="middle" fill="#6B645E" font-size="8">learn → check → final grade</text><text class="lang-zh" x="260" y="204" text-anchor="middle" fill="#6B645E" font-size="8">学 → 查 → 最终成绩</text></g></svg>
%%%

**What the three-stacks picture gets right:** the piles really do have three distinct jobs — study, check, grade — and only one of them is ever opened for the real score.

**Where it breaks down:** real cards are physical, so you *cannot* accidentally study the exam. With data it is dangerously easy to let the exam pile leak into studying — which is exactly the set of traps waiting for us later today.

#### The one rule to burn in: who is allowed to touch what
Back to the three stacks, and this time watch *who changes what*. This is the property everything else protects.

%%% steps
step: the only pile that changes the model
why: only train updates the weights — the model learns on the training set and fits its weights to those practice questions, and nowhere else, exactly like you only ever study from the practice sheet
step: the pile that steers *you*
why: validation guides your choices during development — is it still improving? should I stop? which settings win? — so it changes YOUR decisions, never the weights directly, and it lets you pick without touching the test pile
step: the pile nobody touches
why: test is a one-time honest verdict, locked away like the answer key in the teacher's drawer — it is touched once, only at the end, and you let that honest final score change nothing
%%%

%%% insight
Mixing up those three jobs is not a rookie slip you grow out of — it is how whole teams ship a model that scores beautifully in the report and then falls over in front of real users. The three roles *are* the discipline. Keep them straight and you already sound like someone who has trained real models.
%%%

So far: three piles, three jobs, one rule about who touches what. Let us check that a real split obeys it.

%%% demo id=piles label="reveal — 1000 rows dealt into three piles"
predict: 1000 rows, shuffled, then cut 80/10/10. You can already guess the three sizes. Now the sharper question: how many rows should appear in *two* piles at once?
code: rows = shuffle(all_rows)                       # 1000 rows, mixed up first
code: train, val, test = rows[:800], rows[800:900], rows[900:]
code: len(train), len(val), len(test)
code: set(train) & set(val), set(train) & set(test)  # the honest check
out: sizes           : train 800 | val 100 | test 100
out: train ∩ val     : set()   ← empty ✓
out: train ∩ test    : set()   ← empty ✓
take: <b>Zero. Every row lives in exactly ONE pile.</b> That emptiness is the whole guarantee: the 100 test rows were never studied, so their score is honest. When we meet leakage later, this is precisely the check that would have caught it — two lines of code, and you can run it on any split you ever make.
%%%

Two small habits worth stealing while the piles are fresh. First, a word you will read on every chart from here on: one **epoch** means one full pass over the *training* split. Second, before you celebrate any number, score the laziest possible model — "always guess the most common answer" — on the *same* piles, so you know what your model actually beat. That baseline-first habit costs one line and saves a lot of embarrassment. Nice: three piles, clear jobs. Next, why *two* piles were not enough, and where the middle one came from.
~~~zh
**三叠题卡这张图对在哪里：** 这三堆真的有三份不一样的活 —— 学、查、评分 —— 而且只有其中一堆会为了真成绩被拆开。

**它在哪里不成立：** 真的题卡是实物，所以你*不可能*不小心拿期末考卷来学。数据就危险多了：让考卷那堆漏进学习里，太容易了 —— 而这正是今天后面等着我们的那一串陷阱。

#### 要刻进脑子的那条规则：谁有资格碰什么
回到这三叠卡，这次盯住*谁改了什么*。后面所有东西保护的，就是这个性质。

%%% steps
step: 唯一会改动模型的那一堆
why: 只有 train 会更新 weight —— 模型在 training set 上学，照着那些练习题调自己的 weight，别的地方一概不学，就像你只照着练习卷复习
step: 掌舵*你*的那一堆
why: validation 在开发过程中指引你的选择 —— 还在变好吗？该停了吗？哪套设置赢？—— 所以它改的是*你*的决定，从不直接改 weight，而且它让你在不碰 test 那堆的情况下做出选择
step: 谁都不碰的那一堆
why: test 是一次性的诚实判决，像老师抽屉里的答案本一样锁着 —— 它只在最后被碰一次，而且你让那个诚实的最终分数什么都不改
%%%

%%% insight
把这三份活搞混，不是那种你长大就不会犯的新手小错。整个团队走的就是这条路：交付一个模型，报告里分数漂亮，真用户面前就趴下。这三个角色*本身*就是纪律。把它们分清楚，你说话就已经像一个真的训练过模型的人了。
%%%

到这里：三堆、三份活，还有一条谁能碰什么的规则。我们来检查一个真的 split 有没有守规则。

%%% demo id=pileszh label="揭晓 —— 1000 行发成三堆"
predict: 1000 行，洗过，然后按 80/10/10 切开。三堆各多大你已经能猜到。现在问一个更尖的：有多少行应该同时出现在*两*堆里？
code: rows = shuffle(all_rows)                       # 1000 行，先打乱
code: train, val, test = rows[:800], rows[800:900], rows[900:]
code: len(train), len(val), len(test)
code: set(train) & set(val), set(train) & set(test)  # 那道诚实的检查
out: sizes           : train 800 | val 100 | test 100
out: train ∩ val     : set()   ← empty ✓
out: train ∩ test    : set()   ← empty ✓
take: <b>零。每一行都只住在一堆里。</b>那份「空」就是全部的保证：那 100 行 test 从没被学过，所以它们的分数是诚实的。等我们后面碰到 leakage，抓住它的正是这道检查 —— 两行代码，你以后做的任何一次 split 都能跑一遍。
%%%

趁这三堆还新鲜，有两个小习惯值得偷走。第一，一个你从现在起在每张图上都会读到的词：一个 **epoch（轮次）**就是把*训练*那一份完整走一遍。第二，在你为任何数字高兴之前，先让最懒的模型在*同样*的几堆上打一次分。最懒的模型就是「永远猜最常见的那个答案」。这样你才知道你的模型到底赢了什么。这个「先看基线」的习惯只花一行，能省下很多难为情。不错：三堆，活也清楚了。接下来讲为什么*两*堆不够，以及中间那一堆是从哪来的。
~~~

@@@ concept id=c3 zh_tag="为什么是三堆" tag="Why three, not two" zh_title="中间那一堆是怎么来的" title="Where the middle pile came from" zh_gotit="懂了这段来历" gotit="Got the history"
Fair question: why not just *two* piles — one to study, one to grade? That is exactly how people did it at first, and it looks perfectly reasonable. This is the **two-way split**: train and test only, no middle pile. (The even older habit was to report the score on the training data itself — grading the student on the exact problems they studied — which is the fake grade you just retired.)

Picture a student with only a practice sheet and one final exam, no weekly quiz. They study, sit the exam, score 70%. They change how they study, sit *the same exam again*, 80%. Change again, sit it again, 88%. Somewhere in there the grade quietly stopped being a grade.
~~~zh
一个很合理的问题：为什么不就*两*堆 —— 一堆用来学，一堆用来评分？人们最早就是这么做的，看起来完全说得通。这叫**两分 split**：只有 train 和 test，没有中间那一堆。（更老的习惯是直接报训练数据上的分数 —— 拿学生练过的原题给他打分 —— 那正是你刚刚淘汰掉的假分数。）

想象一个学生只有一张练习卷和一次期末考，没有周测。他复习，考试，考了 70%。他改了复习的办法，*再考同一张卷子*，80%。再改一次，再考一次，88%。在这中间的某个地方，这个成绩悄悄地不再是成绩了。
~~~

%%% svg
<svg viewBox="0 0 520 214" role="img" aria-label="Top row shows the old two-way setup: a student repeatedly takes the same final exam and tweaks their studying based on the score, so the exam becomes contaminated. Bottom row shows the three-way fix: tuning now happens against a separate weekly quiz (validation), and the final exam stays sealed and honest."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Tuning against the exam quietly contaminates it</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">对着考卷调，会悄悄污染它</text><text class="lang-en" x="16" y="44" fill="#C93B3B" font-size="9.5">OLD · two piles</text><text class="lang-zh" x="16" y="44" fill="#C93B3B" font-size="9.5">旧 · 两堆</text><rect x="120" y="32" width="86" height="34" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="163" y="53" text-anchor="middle" fill="#1F6280" font-size="8">train</text><text class="lang-zh" x="163" y="53" text-anchor="middle" fill="#1F6280" font-size="8">train 训练</text><rect x="230" y="32" width="120" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="290" y="47" text-anchor="middle" fill="#C93B3B" font-size="8">final exam</text><text class="lang-zh" x="290" y="47" text-anchor="middle" fill="#C93B3B" font-size="8">期末考卷</text><text class="lang-en" x="290" y="59" text-anchor="middle" fill="#C93B3B" font-size="7.5">also used to TUNE</text><text class="lang-zh" x="290" y="59" text-anchor="middle" fill="#C93B3B" font-size="7.5">同时还用来调</text><path d="M350 40 C 400 40, 400 58, 350 58" fill="none" stroke="#C93B3B" stroke-width="1.4"/><polygon points="352,58 360,54 360,62" fill="#C93B3B"/><text class="lang-en" x="410" y="52" fill="#C93B3B" font-size="8">tweak, re-take,</text><text class="lang-zh" x="410" y="52" fill="#C93B3B" font-size="8">改一改、再考一次、</text><text class="lang-en" x="410" y="64" fill="#C93B3B" font-size="8">repeat 💥 contaminated</text><text class="lang-zh" x="410" y="64" fill="#C93B3B" font-size="8">一直转 💥 被污染了</text><line x1="16" y1="88" x2="504" y2="88" stroke="#E5DFD6"/><text class="lang-en" x="16" y="120" fill="#1a5c38" font-size="9.5">NEW · three piles</text><text class="lang-zh" x="16" y="120" fill="#1a5c38" font-size="9.5">新 · 三堆</text><rect x="120" y="108" width="80" height="34" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="160" y="129" text-anchor="middle" fill="#1F6280" font-size="8">train</text><text class="lang-zh" x="160" y="129" text-anchor="middle" fill="#1F6280" font-size="8">train 训练</text><rect x="212" y="108" width="96" height="34" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="260" y="123" text-anchor="middle" fill="#9A7208" font-size="8">validation</text><text class="lang-zh" x="260" y="123" text-anchor="middle" fill="#9A7208" font-size="8">validation 验证</text><text class="lang-en" x="260" y="135" text-anchor="middle" fill="#9A7208" font-size="7.5">TUNE here</text><text class="lang-zh" x="260" y="135" text-anchor="middle" fill="#9A7208" font-size="7.5">在这里调</text><rect x="320" y="108" width="110" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="375" y="123" text-anchor="middle" fill="#1a5c38" font-size="8">🔒 test</text><text class="lang-zh" x="375" y="123" text-anchor="middle" fill="#1a5c38" font-size="8">🔒 test 测试</text><text class="lang-en" x="375" y="135" text-anchor="middle" fill="#1a5c38" font-size="7.5">stays sealed ✓</text><text class="lang-zh" x="375" y="135" text-anchor="middle" fill="#1a5c38" font-size="7.5">一直封着 ✓</text><text class="lang-en" x="260" y="170" text-anchor="middle" fill="#6B645E" font-size="9">All your tweaking hits the validation pile — so the real exam stays an honest grade.</text><text class="lang-zh" x="260" y="170" text-anchor="middle" fill="#6B645E" font-size="9">你所有的调整都打在 validation 那一堆上，真正的考卷就还是个诚实的成绩。</text><text class="lang-en" x="260" y="192" text-anchor="middle" fill="#1a5c38" font-size="9">The middle pile exists to PROTECT the final one.</text><text class="lang-zh" x="260" y="192" text-anchor="middle" fill="#1a5c38" font-size="9">中间那一堆存在，就是为了保护最后那一堆。</text></g></svg>
%%%

**What the re-taking picture gets right:** the moment a score starts steering your decisions, it stops being an unbiased grade. Repeated tuning leaks the exam's contents into your choices, quietly and for free.

**Where it breaks down:** a person contaminates an exam slowly and might notice. A model can do it instantly and invisibly, inside one automated tuning run that finishes while you get coffee.

#### Watch the grade turn into fiction, one round at a time
Follow the student through three rounds. Nobody cheats. Nothing dramatic happens. And the number still ends up a lie.

%%% steps
step: round 1 — an honest grade
why: the first time they meet the exam it really is unseen, so 70% is a true reading of what they know
step: round 2 — the harmless-looking tweak
why: they change their studying *because of* what the exam asked, which means a little of the exam has now moved into their preparation
step: round 3 — the drift into fiction
why: after enough rounds the 88% describes their fit to *that one exam*, not the subject — tuning against the test contaminates the test, and the grade is no longer honest
%%%

%%% insight
This is the sneakiest failure of the day, because there is no moment where you did something obviously wrong. You just… looked. Every time you look at the test score and change something because of it, you hand the model a little more of the answer key. It is also why public leaderboards slowly rot: hundreds of teams tuning against one hidden set eventually turn it into a practice sheet.
%%%

So far: two piles collapse because you cannot help tuning against the only grade you have. Let us put numbers on it.

%%% demo id=retake label="reveal — the score climbs while the student does not"
predict: the same student tunes against the same exam three times, and their reported score goes 70% → 80% → 88%. Meanwhile, how much do you think their *true* ability on brand-new questions has improved?
code: for round in 1, 2, 3:  tweak_studying_using(exam_score)
code: print(exam_score, hidden_fresh_score)   # the second number nobody measured
out: round 1 : exam 70%   fresh 69%    ← still honest
out: round 2 : exam 80%   fresh 71%    ← gap opening
out: round 3 : exam 88%   fresh 70%    ← pure fiction
take: <b>+18 points on the exam. +1 point of real ability.</b> The exam score climbed because the studying was shaped to that exam, not because the student got better. Two piles cannot catch this, because the number you would need in order to check it <i>is</i> the number you already spent.
%%%

#### The fix: sacrifice a pile
So people carved out a **third** pile. That splits one job into two: *choosing* between options (which settings, which epoch, which model size) and *reporting* an honest final number are different jobs, so they get different piles.

That need to choose things is precisely what motivates a separate validation pile — the pile you are *allowed* to tune against, spent on purpose so the sealed exam stays pristine. The middle pile exists to protect the final one. Now for the sneaky failure this whole setup was built to catch.
~~~zh
**反复重考这张图对在哪里：** 一个分数一旦开始掌舵你的决定，它就不再是一个不偏不倚的成绩。反复调整会把考卷的内容漏进你的选择里，安安静静，不花一分钱。

**它在哪里不成立：** 人污染一张考卷是慢慢来的，还可能自己察觉。模型能瞬间做完，而且看不见 —— 就在一次自动调参里，你去倒杯咖啡它就跑完了。

#### 一轮一轮地看这个成绩怎么变成假的
跟着这个学生走三轮。没有人作弊。没有什么戏剧性的事发生。可这个数字最后还是成了谎话。

%%% steps
step: 第 1 轮 —— 一个诚实的成绩
why: 他第一次碰到这张考卷，它真的是没见过的，所以 70% 是他会多少的真实读数
step: 第 2 轮 —— 那个看起来无害的小改动
why: 他*因为*考卷问了什么而改了自己的复习，这意味着考卷的一小块已经搬进了他的准备里
step: 第 3 轮 —— 一路飘成假的
why: 转够几轮之后，那个 88% 描述的是他对*那一张考卷*的贴合，不是这门课 —— 拿 test 来调就污染了 test，这个成绩不再诚实
%%%

%%% insight
这是今天最难察觉的一次失败，因为没有哪一刻你做了明显不对的事。你只是……看了一眼。每一次你看了 test 分数，又因为它改了什么东西，你就把答案本再多交给模型一点。这也是为什么公开榜单会慢慢烂掉：上百个团队对着同一份藏起来的数据调，最后把它调成了一张练习卷。
%%%

到这里：两堆会塌，因为你手上只有那一个成绩，忍不住就要对着它调。我们给它配上数字。

%%% demo id=retakezh label="揭晓 —— 分数在爬，学生没有"
predict: 同一个学生对着同一张考卷调了三次，他报出来的分数走 70% → 80% → 88%。同时，你觉得他在全新题目上的*真*本事提高了多少？
code: for round in 1, 2, 3:  tweak_studying_using(exam_score)
code: print(exam_score, hidden_fresh_score)   # 第二个数字，没有人量过
out: round 1 : exam 70%   fresh 69%    ← still honest
out: round 2 : exam 80%   fresh 71%    ← gap opening
out: round 3 : exam 88%   fresh 70%    ← pure fiction
take: <b>考卷上 +18 分。真本事 +1 分。</b>考卷分数爬上去，是因为复习被塑造成贴合那张考卷，不是因为学生变强了。两堆抓不住这件事，因为你要用来检查它的那个数字，<i>就是</i>你已经花掉的那个数字。
%%%

#### 办法：牺牲一堆
所以人们又划出**第三**堆。这把一份活拆成两份。一份是在几个选项里*挑*一个：哪套设置、哪个 epoch、多大的模型。另一份是*报出*一个诚实的最终数字。这是两份不同的活，所以它们各拿一堆。

正是这份「要挑东西」的需要，逼出了一个单独的 validation 堆 —— 那一堆是你*被允许*对着调的，故意花掉它，好让封起来的考卷保持干净。中间那一堆存在的意义，就是保护最后那一堆。现在来看这整套安排最初就是为了抓住的那个悄悄发生的失败。
~~~

@@@ concept id=c4 zh_tag="Overfitting" tag="Overfitting" zh_title="Overfitting —— 模型在背答案，不是在学" title="Overfitting — when the model memorizes instead of learns" zh_gotit="懂了 overfitting" gotit="Got overfitting"
This is the failure the whole *exam-vs-answer-key* setup exists to catch, so it deserves a name and a clear picture. Picture a student who, instead of understanding the topic, memorizes their practice sheet down to the tiniest useless detail — every exact word, the order the answers came in, even the little **coffee stain** on question 4. On that one sheet they are flawless. But coffee stains and answer-order are *noise* — they say nothing about the subject — so on a fresh exam, where none of that junk appears, the student falls apart. When a model does this, we call it [[overfitting||when a model memorizes the exact training examples (including their random noise) instead of learning the general pattern — so it scores high on training data but poorly on new data]].
~~~zh
整套*考卷和答案本*的安排，就是为了抓住这一次失败，所以它值得一个名字和一张清楚的图。想象一个学生。他不去弄懂这门课，而是把自己那张练习卷背到最没用的细节。每一个字、答案出现的顺序，连第 4 题上那一小块**咖啡渍**，他都背下来。在那一张卷子上，他一个错都没有。可是咖啡渍和答案顺序都是*噪声* —— 它们对这门课什么都不说 —— 所以换一张新考卷，那些杂物一个都不出现，这个学生就垮了。模型做这件事的时候，我们叫它 [[overfitting（过拟合）||模型把训练样本原样背下来，连它们的随机噪声一起背，而不是学会背后那条普遍的规律 —— 于是它在训练数据上分数很高，在新数据上很差]]。
~~~

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A student hunched over a single practice sheet, memorizing every useless detail: the exact words, the order of the answers, and a coffee stain on question four. Boxes beside the student show the memorized junk. A caption notes that memorizing the noise is not learning the subject."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Memorizing every useless detail is NOT learning</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">把每个没用的细节都背下来，不是学会</text><rect x="60" y="34" width="180" height="150" rx="8" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="150" y="52" text-anchor="middle" fill="#C93B3B" font-size="9.5">🙈 the memorizing student</text><text class="lang-zh" x="150" y="52" text-anchor="middle" fill="#C93B3B" font-size="9.5">🙈 那个背书的学生</text><rect x="86" y="66" width="128" height="102" rx="4" fill="#fff" stroke="#C93B3B"/><text class="lang-en" x="150" y="84" text-anchor="middle" fill="#6B645E" font-size="8">the ONE practice sheet</text><text class="lang-zh" x="150" y="84" text-anchor="middle" fill="#6B645E" font-size="8">唯一那张练习卷</text><line x1="98" y1="96" x2="202" y2="96" stroke="#D9CFC2"/><line x1="98" y1="110" x2="202" y2="110" stroke="#D9CFC2"/><line x1="98" y1="124" x2="202" y2="124" stroke="#D9CFC2"/><line x1="98" y1="138" x2="180" y2="138" stroke="#D9CFC2"/><circle cx="176" cy="150" r="9" fill="#B98A3E" opacity="0.55"/><text x="176" y="153" text-anchor="middle" fill="#7A5A22" font-size="7">☕</text><text class="lang-en" x="150" y="180" text-anchor="middle" fill="#1a5c38" font-size="8.5">score on THIS sheet: 100% ✓</text><text class="lang-zh" x="150" y="180" text-anchor="middle" fill="#1a5c38" font-size="8.5">这张卷子上的分数：100% ✓</text><rect x="286" y="34" width="214" height="150" rx="8" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="393" y="52" text-anchor="middle" fill="#9A7208" font-size="9">what got memorized (all noise):</text><text class="lang-zh" x="393" y="52" text-anchor="middle" fill="#9A7208" font-size="9">被背下来的东西（全是噪声）：</text><rect x="300" y="66" width="186" height="22" rx="4" fill="#fff" stroke="#C99A12"/><text class="lang-en" x="393" y="81" text-anchor="middle" fill="#9A7208" font-size="8">🔤 the exact wording</text><text class="lang-zh" x="393" y="81" text-anchor="middle" fill="#9A7208" font-size="8">🔤 原来的每一个字</text><rect x="300" y="94" width="186" height="22" rx="4" fill="#fff" stroke="#C99A12"/><text class="lang-en" x="393" y="109" text-anchor="middle" fill="#9A7208" font-size="8">🔢 the ORDER of the answers</text><text class="lang-zh" x="393" y="109" text-anchor="middle" fill="#9A7208" font-size="8">🔢 答案出现的顺序</text><rect x="300" y="122" width="186" height="22" rx="4" fill="#fff" stroke="#C99A12"/><text class="lang-en" x="393" y="137" text-anchor="middle" fill="#9A7208" font-size="8">☕ the coffee stain on Q4</text><text class="lang-zh" x="393" y="137" text-anchor="middle" fill="#9A7208" font-size="8">☕ 第 4 题上的咖啡渍</text><text class="lang-en" x="393" y="166" text-anchor="middle" fill="#C93B3B" font-size="8">none of this is on the real exam →</text><text class="lang-zh" x="393" y="166" text-anchor="middle" fill="#C93B3B" font-size="8">这些在真考卷上一个都没有 →</text><text class="lang-en" x="393" y="178" text-anchor="middle" fill="#C93B3B" font-size="8">fresh exam: crashes ✗</text><text class="lang-zh" x="393" y="178" text-anchor="middle" fill="#C93B3B" font-size="8">新考卷：崩了 ✗</text></g></svg>
%%%

**What the memorizing-student picture gets right:** clinging to noise — the exact words, the order, the coffee stain — feels like knowledge on the one sheet you studied, but it is worthless anywhere else, so a fresh exam exposes it.

**Where it breaks down:** a student could, in principle, *notice* they are cramming trivia and stop. A model cannot — it will absorb every stray detail, because for a big flexible model storing the junk is easier than finding the pattern underneath.

#### From "memorizing the noise" to a wiggly curve
Now watch *what that memorizing does to the model's shape* — this is the heart of overfitting, so we take it one step at a time. Say your training data are seven dots that follow a gentle rising trend, jiggled up and down a little by random noise. A model that **learns the pattern** draws a *smooth* line through the middle of the cloud, ignoring the jiggle. A model that **memorizes** does the opposite: it bends and detours to pass *exactly* through every single dot, chasing each bit of noise — which forces the line to whip up and down between them. That whipping shape is what people mean by a **wiggly** curve. Same seven dots, twice, so you can *see* memorizing happen:
~~~zh
**背书学生这张图对在哪里：** 抱着噪声不放 —— 每个字、顺序、那块咖啡渍。在你学过的那一张卷子上，这些感觉像是知识。可它们在别的任何地方一点用都没有，所以一张新考卷就把它们揭穿了。

**它在哪里不成立：** 学生原则上*能察觉*自己在死记零碎的东西，然后停下来。模型不能。它会把每一个散落的细节都吸进去，因为对一个又大又灵活的模型来说，存杂物比找到下面那条规律更省事。

#### 从「把噪声背下来」到一条扭来扭去的曲线
现在看*这种背法把模型的形状变成了什么样* —— 这是 overfitting 的核心，所以我们一步一步来。假设你的训练数据是七个点，它们跟着一条缓缓上升的趋势，被随机噪声上下抖了一点。一个**学到规律**的模型会画一条*平滑*的线，穿过这团点的中间，不理那些抖动。一个**背下来**的模型正好反过来：它拐弯绕路，*精确*穿过每一个点，追着每一小块噪声 —— 这就逼得那条线在点与点之间上下猛甩。那个猛甩的形状，就是人们说的**扭来扭去**的曲线。同样这七个点，画两遍，好让你*看见*背答案发生：
~~~

%%% svg
<svg viewBox="0 0 520 224" role="img" aria-label="A staged before-and-after of the exact same seven data dots. Step one on the left: the seven raw noisy dots with no curve, following a gentle rising trend with random jiggle. Step two top-right: a smooth line drawn through the middle of the very same seven dots, labelled learned the pattern, ignoring the jiggle, and passing close to a new point marked at the same input position. Step three bottom-right: a wild wiggly line bending to touch every single one of those identical seven dots exactly, labelled memorized every dot, which whips up and down and, at that same new input position, plunges far away from the new point so it misses it badly."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The SAME seven dots → smooth vs. wiggly</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">同样这七个点 → 平滑 还是 扭来扭去</text><g><rect x="12" y="30" width="180" height="184" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text class="lang-en" x="102" y="48" text-anchor="middle" fill="#6B645E" font-size="9">① seven raw noisy dots</text><text class="lang-zh" x="102" y="48" text-anchor="middle" fill="#6B645E" font-size="9">① 七个带噪声的原始点</text><g fill="#2C2A28"><circle cx="40" cy="178" r="3.2"/><circle cx="62" cy="150" r="3.2"/><circle cx="84" cy="164" r="3.2"/><circle cx="106" cy="120" r="3.2"/><circle cx="128" cy="140" r="3.2"/><circle cx="150" cy="94" r="3.2"/><circle cx="176" cy="100" r="3.2"/></g><text class="lang-en" x="102" y="204" text-anchor="middle" fill="#6B645E" font-size="7.5">a gentle rise + jiggle</text><text class="lang-zh" x="102" y="204" text-anchor="middle" fill="#6B645E" font-size="7.5">缓缓上升 + 抖动</text></g><g><rect x="204" y="30" width="150" height="184" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="279" y="48" text-anchor="middle" fill="#1a5c38" font-size="9">② learned: smooth</text><text class="lang-zh" x="279" y="48" text-anchor="middle" fill="#1a5c38" font-size="9">② 学到了：平滑</text><path d="M224 172 C 262 150, 300 118, 344 102" fill="none" stroke="#2D8B55" stroke-width="2.6"/><g fill="#2C2A28"><circle cx="224" cy="178" r="2.6"/><circle cx="244" cy="150" r="2.6"/><circle cx="264" cy="164" r="2.6"/><circle cx="286" cy="120" r="2.6"/><circle cx="306" cy="140" r="2.6"/><circle cx="326" cy="94" r="2.6"/><circle cx="344" cy="100" r="2.6"/></g><circle cx="290" cy="118" r="4" fill="none" stroke="#1F6280" stroke-width="1.6"/><text class="lang-en" x="279" y="196" text-anchor="middle" fill="#1F6280" font-size="7.5">new point → close ✓</text><text class="lang-zh" x="279" y="196" text-anchor="middle" fill="#1F6280" font-size="7.5">新的点 → 很近 ✓</text></g><g><rect x="366" y="30" width="142" height="184" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="437" y="48" text-anchor="middle" fill="#C93B3B" font-size="9">③ memorized: wiggly</text><text class="lang-zh" x="437" y="48" text-anchor="middle" fill="#C93B3B" font-size="9">③ 背下来了：扭来扭去</text><path d="M380 178 L 389 132 L 398 150 L 407 184 L 416 164 L 425 108 L 434 120 L 443 188 L 452 140 L 461 96 L 470 94 L 482 150 L 494 100" fill="none" stroke="#C93B3B" stroke-width="2"/><g fill="#2C2A28"><circle cx="380" cy="178" r="2.6"/><circle cx="398" cy="150" r="2.6"/><circle cx="416" cy="164" r="2.6"/><circle cx="434" cy="120" r="2.6"/><circle cx="452" cy="140" r="2.6"/><circle cx="470" cy="94" r="2.6"/><circle cx="494" cy="100" r="2.6"/></g><circle cx="443" cy="118" r="4" fill="none" stroke="#C93B3B" stroke-width="1.6"/><text class="lang-en" x="437" y="196" text-anchor="middle" fill="#C93B3B" font-size="7.5">new point → way off ✗</text><text class="lang-zh" x="437" y="196" text-anchor="middle" fill="#C93B3B" font-size="7.5">新的点 → 差很远 ✗</text></g></g></svg>
%%%

**What this build-up shows:** the *exact same seven dots* give a calm, useful curve when the model ignores the jiggle (②), and a frantic, useless curve when it insists on touching every dot (③). Panel ③ is memorizing made visible. **Where the drawing breaks down:** here you see one tidy curve in 2-D, but a real model has thousands of weights doing this in many dimensions at once — you cannot eyeball the wiggle, which is exactly why the next concept gives you a *number*.

#### Your turn: drag the flexibility dial and make it wiggle
You do not have to take my word for any of that — you can *cause* it. The dial below is the model's **flexibility**: how bendy the curve is allowed to get. In a real model that dial is its [[capacity||how much a model can bend or store — more neurons, more layers, a higher-degree curve. High capacity is exactly what makes memorizing possible]]. Predict first: at the far LEFT the curve can barely bend at all — do you think it fits the dots well? Now drag.
~~~zh
**这一步搭出来的东西说明：** *完全同样的七个点*，模型不理那些抖动时给出一条安静、有用的曲线（②）；它非要碰到每一个点时，给出一条抓狂、没用的曲线（③）。③ 号格子就是背答案被画出来的样子。**这张图在哪里不成立：** 这里你看到的是二维里一条清楚的曲线。可真的模型有好几千个 weight，它们在很多个维度里同时干这件事。你没法用眼睛看那些扭动 —— 而这正是下一个概念要给你一个*数字*的原因。

#### 换你来：拖那个「灵活度」旋钮，把它拖到扭起来
上面这些你不用信我 —— 你可以自己*造出来*。下面这个旋钮是模型的**灵活度**：这条曲线被允许弯到多厉害。在真的模型里，那个旋钮就是它的 [[capacity（容量）||一个模型能弯多厉害、能存多少 —— 更多 neuron、更多 layer、次数更高的曲线。高 capacity 正是让背答案变得可能的东西]]。先预测：在最**左**边，曲线几乎完全弯不了 —— 你觉得它贴合这些点吗？现在拖吧。
~~~

%%% svg
<svg id="flex-svg" viewBox="0 0 520 244" role="img" aria-label="An interactive fit explorer. Seven noisy training dots follow a gentle rising trend. A single curve is drawn through them and changes as you drag a flexibility dial from one to ten. At flexibility one the curve is a straight stiff line that misses most dots, so both the training error and the new-data error are high: underfitting. Around flexibility five the curve is smooth and passes near every dot, and the new-data error is at its lowest: the sweet spot. At flexibility ten the curve whips up and down to touch every single dot exactly, so the training error is almost zero while the new-data error explodes: overfitting."><g font-family="monospace" font-size="10"><text class="lang-en" x="260" y="17" text-anchor="middle" fill="#2C2A28" font-size="12">Drag the flexibility dial — watch the curve start to wiggle</text><text class="lang-zh" x="260" y="17" text-anchor="middle" fill="#2C2A28" font-size="12">拖那个灵活度旋钮 —— 看曲线开始扭</text><rect x="60" y="28" width="418" height="174" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="202" x2="478" y2="202" stroke="#B8AEA2"/><line x1="60" y1="28" x2="60" y2="202" stroke="#B8AEA2"/><text class="lang-en" x="44" y="118" fill="#6B645E" font-size="9" transform="rotate(-90 44 118)">output</text><text class="lang-zh" x="44" y="118" fill="#6B645E" font-size="9" transform="rotate(-90 44 118)">输出</text><text class="lang-en" x="269" y="220" text-anchor="middle" fill="#6B645E" font-size="9">input →</text><text class="lang-zh" x="269" y="220" text-anchor="middle" fill="#6B645E" font-size="9">输入 →</text><path id="flex-curve" d="" fill="none" stroke="#C93B3B" stroke-width="2.4"/><g fill="#2C2A28"><circle cx="70" cy="175" r="3.4"/><circle cx="135" cy="148" r="3.4"/><circle cx="200" cy="160" r="3.4"/><circle cx="265" cy="120" r="3.4"/><circle cx="330" cy="138" r="3.4"/><circle cx="395" cy="96" r="3.4"/><circle cx="460" cy="92" r="3.4"/></g><text class="lang-en" x="470" y="196" text-anchor="end" fill="#6B645E" font-size="8">• = the 7 training dots (noisy)</text><text class="lang-zh" x="470" y="196" text-anchor="end" fill="#6B645E" font-size="8">• = 那 7 个训练点（带噪声）</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label><span class="lang-en">model flexibility</span><span class="lang-zh">模型灵活度</span> <input id="flex-r" type="range" min="1" max="10" step="1" value="1" style="width:52%;accent-color:#C93B3B;vertical-align:middle"> <b id="flex-v">1</b> / 10</label>
<div id="flex-out" class="lang-en" style="margin-top:6px;color:#2C2A28">flexibility 1 → a stiff straight line: it misses most dots — <b>underfitting</b></div>
<div id="flex-out-zh" class="lang-zh" style="margin-top:6px;color:#2C2A28">灵活度 1 → 一条僵硬的直线：它错过了大部分点 —— <b>underfitting</b></div>
<div id="flex-err" class="lang-en" style="margin-top:4px;color:#6B645E">error on the training dots: 0.31 · error on NEW data: 0.34</div>
<div id="flex-err-zh" class="lang-zh" style="margin-top:4px;color:#6B645E">训练点上的误差：0.31 · 新数据上的误差：0.34</div>
</div>
<script>(function(){
  var r=document.getElementById('flex-r');if(!r)return;
  var cur=document.getElementById('flex-curve'),out=document.getElementById('flex-out'),
      ev=document.getElementById('flex-v'),er=document.getElementById('flex-err'),
      outz=document.getElementById('flex-out-zh'),erz=document.getElementById('flex-err-zh');
  var xs=[70,135,200,265,330,395,460], ys=[175,148,160,120,138,96,92], STEP=65;
  // least-squares straight line through the 7 dots = the stiffest possible model
  var n=xs.length,sx=0,sy=0,sxy=0,sxx=0;
  for(var i=0;i<n;i++){sx+=xs[i];sy+=ys[i];sxy+=xs[i]*ys[i];sxx+=xs[i]*xs[i];}
  var slope=(n*sxy-sx*sy)/(n*sxx-sx*sx), icpt=(sy-slope*sx)/n;
  function line(x){return slope*x+icpt;}
  var res=[];for(i=0;i<n;i++)res.push(ys[i]-line(xs[i]));
  // smooth (cosine) interpolation of the residuals: at full flexibility the curve hits every dot
  function residAt(x){
    if(x<=xs[0])return res[0];if(x>=xs[n-1])return res[n-1];
    var k=Math.floor((x-xs[0])/STEP);if(k>n-2)k=n-2;
    var t=(x-xs[k])/STEP, s=(1-Math.cos(Math.PI*t))/2;
    return res[k]*(1-s)+res[k+1]*s;
  }
  function paint(){
    var f=+r.value, w=(f-1)/9;                       // w = 0 stiff … 1 fully flexible
    var d='';
    for(var x=70;x<=460.5;x+=2){
      // stiff line + (flexibility × the wobble needed to reach each dot) + whip between the dots
      var y=line(x)+w*residAt(x)+Math.pow(w,3)*22*Math.sin(Math.PI*(x-70)/STEP);
      d+=(d?'L':'M')+x.toFixed(1)+' '+y.toFixed(1)+' ';
    }
    cur.setAttribute('d',d.trim());
    var trainE=0.30*Math.pow(1-w,1.6)+0.01, valE=trainE+0.03+0.55*Math.pow(w,3);
    ev.textContent=f;
    var msg,msgz,col;
    if(f<=3){msg='too stiff to follow the trend: it misses most dots, and it is just as bad on new data — <b>underfitting</b>';msgz='太僵硬，跟不上这条趋势：它错过大部分点，在新数据上也一样差 —— <b>underfitting</b>';col='#1F6280';}
    else if(f<=6){msg='smooth, close to every dot, ignoring the jiggle — the <b>sweet spot</b> ⭐ (new-data error bottoms out around here)';msgz='平滑，离每个点都很近，不理那些抖动 —— 这就是<b>那个最佳点</b> ⭐（新数据的误差大概在这里到达最低）';col='#2D8B55';}
    else{msg='now it whips up and down to touch every dot exactly — it is memorizing the noise: <b>overfitting</b> 💥';msgz='现在它上下猛甩，好精确碰到每一个点 —— 它在把噪声背下来：<b>overfitting</b> 💥';col='#C93B3B';}
    out.innerHTML='flexibility '+f+' → '+msg;out.style.color=col;
    outz.innerHTML='灵活度 '+f+' → '+msgz;outz.style.color=col;
    er.innerHTML='error on the training dots: <b>'+trainE.toFixed(2)+'</b> · error on NEW data: <b>'+valE.toFixed(2)+'</b>';
    erz.innerHTML='训练点上的误差：<b>'+trainE.toFixed(2)+'</b> · 新数据上的误差：<b>'+valE.toFixed(2)+'</b>';
  }
  r.addEventListener('input',paint);paint();
})();</script>
%%%

%%% insight
Did you notice the cruel part? As you drag right, the error on the training dots keeps getting *better* — all the way down to almost zero — while the error on new data quietly turns around and explodes. If the training number were the only thing you looked at, the far right would look like your best model ever. That mismatch is the whole reason today exists.
%%%

#### Cause and symptom, side by side
Two names to keep, now that you have felt both ends of the dial:

- **Overfitting.** *Cause:* the model **memorizes the training data** — it has enough capacity to store the examples and their random noise instead of learning the pattern beneath them, and a finite training set always contains some noise to store. *Symptom:* a **high train** score with a **low validation** score. It aces the practice sheet and fails the fresh exam, and the bigger that difference, the worse the overfitting.
- **Underfitting** (the far left of the dial). *Cause:* too little capacity, too few useful features, or too few training steps. *Symptom:* both scores are bad and stop improving. *Remedy:* the opposite moves — more capacity, better features, train longer.

And the three honest remedies for overfitting, all of which you can already picture: turn the flexibility dial **down** (capacity control — fewer neurons, a simpler model), **stop training earlier** (next concept shows you exactly when), or **get more training data** — because memorizing a thousand examples is much harder than memorizing seven. There is a whole family of extra tricks with names like weight decay and dropout, waiting for you in the training lesson ahead.

The word "overfitting" literally means *fitting too much* — bending past the real pattern into the noise. Good news: because the symptom is a *difference between two scores*, we can measure it directly. That measurement is the single most useful diagnostic in all of training, and it is next.
~~~zh
%%% insight
你注意到最残忍的那部分了吗？你往右拖的时候，训练点上的误差一直在*变好* —— 一路降到接近零 —— 而新数据上的误差悄悄转了向，然后炸开。如果你只看训练那个数字，最右边看起来会像你史上最好的模型。这个不一致，就是今天存在的全部理由。
%%%

#### 原因和症状，摆在一起
现在你两端都摸过了，有两个名字要记住：

- **Overfitting。** *原因：* 模型**把训练数据背下来** —— 它的 capacity 足够存下这些样本和它们的随机噪声，而不去学下面那条规律，而一个有限的 training set 里总有一些噪声可以存。*症状：* **train 分数高**，**validation 分数低**。它在练习卷上满分，在新考卷上不及格；这两者差得越大，overfitting 越严重。
- **Underfitting（欠拟合）**（旋钮的最左边）。*原因：* capacity 太少、有用的特征太少，或者训练步数太少。*症状：* 两个分数都很差，而且不再变好。*解药：* 反方向的动作 —— 更多 capacity、更好的特征、训练更久。

再说 overfitting 的三个诚实解药。每一个你现在都已经能想出画面。第一，把灵活度旋钮**往下**转，也就是 capacity 控制：更少 neuron、更简单的模型。第二，**更早停下训练**，下一个概念会告诉你到底在哪一步停。第三，**拿到更多训练数据**，因为背一千个样本比背七个难太多了。还有一整家族的额外花招，名字叫 weight decay、dropout 之类，在后面的训练课里等着你。

「Overfitting」这个词字面上就是*拟合得太多* —— 弯过了真正的规律，弯进了噪声里。好消息：因为它的症状是*两个分数之间的差*，我们可以直接把它量出来。这个量法是整个训练里最有用的一个诊断，而它就在下一节。
~~~

@@@ concept id=c5 zh_tag="那道缝" tag="The gap" zh_title="train–val 的那道缝 —— 你能亲眼看见 overfitting" title="The train–val gap — how you SEE overfitting happen" zh_gotit="懂了那道缝" gotit="Got the gap"
Here is the beautiful part: you do not have to *guess* whether a model is memorizing — you can *watch* it on a chart. Picture two runners on the same track. One runner is the score on the training pile, the other is the score on the validation pile. Early on they run shoulder to shoulder, both getting better together — and that closeness means the model is learning real patterns that work on *both* piles. Then, if the model starts memorizing, they come apart: the training runner keeps sprinting while the validation runner slows and stops. **The space that opens between the two runners is overfitting, made visible.** We call it the [[train–val gap||the difference between the training score and the validation score — a small, steady gap is healthy; a gap that keeps widening is overfitting you can see]].
~~~zh
好玩的地方来了：你不用*猜*一个模型是不是在背答案 —— 你可以在一张图上*看着*它。想象同一条跑道上的两个跑者。一个跑者是 training 那一堆上的分数，另一个是 validation 那一堆上的分数。开头他们肩并肩地跑，一起变好 —— 这份贴得很近，意味着模型学到的是在*两*堆上都管用的真规律。接着，如果模型开始背答案，他们就分开了：training 那个跑者继续冲刺，validation 那个慢下来，然后停住。**在两个跑者之间张开的那块空隙，就是被看见的 overfitting。** 我们叫它 [[train–val 的那道缝||training 分数和 validation 分数之间的差 —— 小而稳的一道缝是健康的；一直变宽的缝，就是你能看见的 overfitting]]。
~~~

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two runners on a two-lane track as an analogy for the train and validation scores. In the top lane the training runner has sprinted far ahead toward the finish. In the bottom lane the validation runner has slowed and stopped much earlier on the track. A dashed bracket between the two runners is labelled the gap, and a caption says the space between the runners is overfitting made visible."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two runners, one track — mind the space between them</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">两个跑者，一条跑道 —— 注意他们之间那块空隙</text><rect x="40" y="30" width="446" height="44" rx="6" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="40" y="84" width="446" height="44" rx="6" fill="#FDF3D6" stroke="#C99A12"/><g stroke="#B8AEA2" stroke-dasharray="6,8"><line x1="40" y1="52" x2="486" y2="52"/><line x1="40" y1="106" x2="486" y2="106"/></g><line x1="52" y1="24" x2="52" y2="134" stroke="#6B645E" stroke-width="1.4"/><text class="lang-en" x="52" y="146" text-anchor="middle" fill="#6B645E" font-size="8">start</text><text class="lang-zh" x="52" y="146" text-anchor="middle" fill="#6B645E" font-size="8">起点</text><line x1="474" y1="24" x2="474" y2="134" stroke="#2D8B55" stroke-width="1.6"/><text class="lang-en" x="474" y="146" text-anchor="middle" fill="#1a5c38" font-size="8">finish</text><text class="lang-zh" x="474" y="146" text-anchor="middle" fill="#1a5c38" font-size="8">终点</text><circle cx="404" cy="52" r="13" fill="#2A7B9B"/><text x="404" y="57" text-anchor="middle" font-size="13">🏃</text><text class="lang-en" x="404" y="26" text-anchor="middle" fill="#1F6280" font-size="8.5">training score — still sprinting</text><text class="lang-zh" x="404" y="26" text-anchor="middle" fill="#1F6280" font-size="8.5">training 分数 —— 还在冲刺</text><circle cx="226" cy="106" r="13" fill="#C99A12"/><text x="226" y="111" text-anchor="middle" font-size="13">🚶</text><text class="lang-en" x="226" y="170" text-anchor="middle" fill="#9A7208" font-size="8.5">validation score — slowed, then stopped</text><text class="lang-zh" x="226" y="170" text-anchor="middle" fill="#9A7208" font-size="8.5">validation 分数 —— 慢下来，然后停住</text><path d="M226 132 L 226 152 L 404 152 L 404 70" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><text class="lang-en" x="315" y="148" text-anchor="middle" fill="#C93B3B" font-size="9">the GAP</text><text class="lang-zh" x="315" y="148" text-anchor="middle" fill="#C93B3B" font-size="9">那道缝</text></g></svg>
%%%

**What the two-runners picture gets right:** while the model is learning real patterns, both scores improve together and stay close; the moment it starts memorizing, only the training score keeps improving, and the space that opens between them is the overfitting.

**Where it breaks down:** a runner who slows down is still moving toward the finish, and keeps every metre already covered. A validation score does something no runner can do — it turns around and goes *backwards*, because the model is actively getting *worse* on new data even as it looks better on practice.

#### Now put it on a chart you can drag
Both runners' progress is drawn as a **loss** curve over epochs (loss is the "how wrong am I" number from Day 4, and *lower is better*; an epoch is one full pass over the training split). So "running well" means the curve going *down*. Two curves on one chart, and the whole story of a training run is in the space between them.

Predict before you drag: at which point on the chart do you think the *best* model lives — at the far right where training loss is smallest, or somewhere earlier? Drag the slider from left to right and find out.
~~~zh
**两个跑者这张图对在哪里：** 模型在学真规律的时候，两个分数一起变好、贴得很近；它一开始背答案，就只有 training 分数继续变好，而它们之间张开的那块空隙就是 overfitting。

**它在哪里不成立：** 一个慢下来的跑者仍然在往终点走，已经跑过的每一米都还留着。validation 分数会做一件没有跑者做得到的事 —— 它转身往*后*走，因为模型在新数据上正在变得*更差*，尽管它在练习卷上看起来更好了。

#### 现在把它放到一张你能拖的图上
两个跑者的进展都画成一条 **loss（损失）**曲线，横轴是 epoch（loss 就是第 4 天那个「我错得多厉害」的数字，*越低越好*；一个 epoch 是把训练那一份完整走一遍）。所以「跑得好」意思是曲线在*往下*。一张图上两条曲线，一次训练的整个故事，都在它们之间那块空隙里。

在你拖之前先预测：你觉得*最好*的模型住在图上哪个位置 —— 在最右边、training loss 最小的地方，还是更早的某处？把滑块从左往右拖，看看答案。
~~~

%%% svg
<svg id="gap-svg" viewBox="0 0 520 250" role="img" aria-label="Interactive loss curves over training epochs. A blue training-loss curve keeps dropping. An amber validation-loss curve drops with it at first, reaches its lowest point around epoch ten, then rises. Drag the epoch slider: early on the two curves are close and both dropping together with a small steady gap (healthy). At the lowest validation point the gap is about to open (the best model, where you would stop). Just after it, the validation curve turns the corner and the gap starts to open. Later the gap is wide as validation climbs while training keeps falling (overfitting). A marker slides along both curves and the readout names the regime."><g font-family="monospace" font-size="10"><rect x="46" y="26" width="440" height="150" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><text class="lang-en" x="266" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The gap between the two runners IS overfitting</text><text class="lang-zh" x="266" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">两个跑者之间那道缝，就是 overfitting</text><line x1="46" y1="176" x2="486" y2="176" stroke="#B8AEA2"/><line x1="46" y1="26" x2="46" y2="176" stroke="#B8AEA2"/><text class="lang-en" x="30" y="104" fill="#6B645E" font-size="9" transform="rotate(-90 30 104)">loss</text><text class="lang-zh" x="30" y="104" fill="#6B645E" font-size="9" transform="rotate(-90 30 104)">loss</text><text class="lang-en" x="266" y="196" text-anchor="middle" fill="#6B645E" font-size="9">training epochs →</text><text class="lang-zh" x="266" y="196" text-anchor="middle" fill="#6B645E" font-size="9">训练 epoch →</text><path id="gap-train" d="" fill="none" stroke="#2A7B9B" stroke-width="2.6"/><path id="gap-val" d="" fill="none" stroke="#C99A12" stroke-width="2.6"/><line id="gap-line" x1="0" y1="0" x2="0" y2="0" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="3,3" opacity="0"/><circle id="gap-dt" cx="0" cy="0" r="4.5" fill="#2A7B9B" opacity="0"/><circle id="gap-dv" cx="0" cy="0" r="4.5" fill="#C99A12" opacity="0"/><text class="lang-en" x="470" y="150" text-anchor="end" fill="#1F6280" font-size="8.5">train loss ↓ (keeps dropping)</text><text class="lang-zh" x="470" y="150" text-anchor="end" fill="#1F6280" font-size="8.5">train loss ↓（一直在掉）</text><text class="lang-en" x="470" y="46" text-anchor="end" fill="#9A7208" font-size="8.5">val loss ↓ then ↑</text><text class="lang-zh" x="470" y="46" text-anchor="end" fill="#9A7208" font-size="8.5">val loss 先 ↓ 后 ↑</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label><span class="lang-en">training epoch</span><span class="lang-zh">训练 epoch</span> <input id="gap-ep" type="range" min="1" max="30" step="1" value="6" style="width:56%;accent-color:#2A7B9B;vertical-align:middle"> <b id="gap-epv">6</b></label>
<div id="gap-out" class="lang-en" style="margin-top:6px;color:#2C2A28">epoch 6 → both curves dropping together, small steady gap — <b>healthy learning</b> ✓</div>
<div id="gap-out-zh" class="lang-zh" style="margin-top:6px;color:#2C2A28">epoch 6 → 两条曲线一起往下掉，缝小而稳 —— <b>健康的学习</b> ✓</div>
</div>
<script>(function(){
  var ep=document.getElementById('gap-ep');if(!ep)return;
  var tp=document.getElementById('gap-train'),vp=document.getElementById('gap-val'),
      dt=document.getElementById('gap-dt'),dv=document.getElementById('gap-dv'),
      gl=document.getElementById('gap-line'),out=document.getElementById('gap-out'),epv=document.getElementById('gap-epv'),
      outz=document.getElementById('gap-out-zh');
  var X0=46,X1=486,Y0=176,YT=30,N=30,BOTTOM=10;
  function px(e){return X0+(e-1)/(N-1)*(X1-X0);}
  function py(l){return Y0-Math.min(l,1)/1*(Y0-YT);}
  // train loss: smooth exponential decay toward ~0.05
  function trainL(e){return 0.9*Math.exp(-e/4.5)+0.05;}
  // val loss: drops with train, reaches its TRUE minimum at epoch 10, then climbs (overfitting)
  function valL(e){var base=0.9*Math.exp(-e/4.5)+0.08;var over=0.035*Math.max(0,e-BOTTOM);return base+over;}
  var td='',vd='';
  for(var i=1;i<=N;i++){td+=(i>1?'L':'M')+px(i).toFixed(1)+' '+py(trainL(i)).toFixed(1)+' ';vd+=(i>1?'L':'M')+px(i).toFixed(1)+' '+py(valL(i)).toFixed(1)+' ';}
  tp.setAttribute('d',td.trim());vp.setAttribute('d',vd.trim());
  function paint(){
    var e=+ep.value;epv.textContent=e;
    var xt=px(e),yt=py(trainL(e)),yv=py(valL(e));
    dt.setAttribute('cx',xt);dt.setAttribute('cy',yt);dt.setAttribute('opacity','1');
    dv.setAttribute('cx',xt);dv.setAttribute('cy',yv);dv.setAttribute('opacity','1');
    gl.setAttribute('x1',xt);gl.setAttribute('x2',xt);gl.setAttribute('y1',yt);gl.setAttribute('y2',yv);gl.setAttribute('opacity','1');
    var gap=valL(e)-trainL(e),msg,msgz,col;
    if(e<BOTTOM){msg='both curves still dropping together, small steady gap — <b>healthy learning</b> ✓';msgz='两条曲线还在一起往下掉，缝小而稳 —— <b>健康的学习</b> ✓';col='#2D8B55';}
    else if(e===BOTTOM){msg='validation is at its <b>lowest point</b> — the best model, and where you would stop ⭐';msgz='validation 到了它的<b>最低点</b> —— 最好的模型，也是你该停的地方 ⭐';col='#C99A12';}
    else if(e<=13){msg='validation has just <b>turned the corner</b> and started climbing — the gap is beginning to open (you are past the best point)';msgz='validation 刚刚<b>拐了弯</b>，开始往上爬 —— 缝开始张开了（你已经走过最好的那一点）';col='#B9761A';}
    else{msg='validation is climbing while training keeps falling — the gap is now <b>wide open: overfitting</b> 💥';msgz='validation 在往上爬，training 还在往下掉 —— 缝现在<b>大开：overfitting</b> 💥';col='#C93B3B';}
    out.innerHTML='epoch '+e+' → '+msg+'  <span style="color:#6B645E">(gap = '+gap.toFixed(2)+')</span>';
    out.style.color=col;
    outz.innerHTML='epoch '+e+' → '+msgz+'  <span style="color:#6B645E">（缝 = '+gap.toFixed(2)+'）</span>';
    outz.style.color=col;
  }
  ep.addEventListener('input',paint);paint();
})();</script>
%%%

%%% hint
t1: Do not try to read both curves at once. Just watch ONE thing: the space between the two lines as you drag the slider to the right.
t2: Early on, both lines drop together and the space stays small — that is healthy. Keep dragging: the training line keeps falling, but the validation line bottoms out and turns back up. The moment it turns up, the space starts growing.
t3: The train-validation gap IS overfitting made visible, and the best model is exactly where the validation line is lowest — right before the gap opens.
%%%

#### Reading the gap — the practitioner's habit
Three moves, and this is genuinely all of it. This is the chart you will squint at for the rest of your machine-learning life, so it is worth going slowly here.

%%% steps
step: read the gap, not the curves
why: the gap between train and validation loss is one number — validation minus training — which means you only have to watch one thing instead of two
step: small and steady = learning
why: while both curves fall together the gap barely changes, therefore the model is finding patterns that work on data it never studied — this is the part you want to last as long as possible
step: the gap widens = memorizing
why: when validation turns up while training keeps falling, the extra skill is going into the training pile only — that's why a widening gap means memorizing, and why watching the gap is how you see overfitting without any extra tools
step: stop at the lowest validation point
why: that epoch is the sweet spot — the best model on new data — so you save that checkpoint and stop, which has a name: early stopping
%%%

%%% insight
Notice how cheap this superpower is. No new math, no new library — just plot the second curve. Most people who "cannot get their model to work" have simply never looked at the validation curve, so they either stop far too early or train long past their best model and never know it. You will not be one of them.
%%%

So far we have shown one chart and one habit: watch the space between the two lines, and keep the epoch where validation is lowest. Real training curves add one practical wrinkle.

Real curves are bumpy, so a single epoch can dip or bump by pure luck. That is why you do not stop at the very first uptick. You give it a **patience** window — "if validation has not improved for, say, 5 epochs, stop and go back to the best checkpoint."

Early stopping is one member of a family of anti-memorizing tricks called [[regularization||techniques that discourage a model from memorizing — early stopping, weight decay, dropout, data augmentation. Today you learn to SEE overfitting; the training lesson ahead builds these fixes]], and the rest of that family is waiting in the training lesson.

Victory lap: you can now look at two curves and say what a model is doing — learning, memorizing, or stuck. Next: the sneaky ways a split itself can lie to you, even when the curves look lovely.
~~~zh
%%% hint
t1: 不要想着同时读两条曲线。只盯住一件事：你把滑块往右拖的时候，两条线之间那块空隙。
t2: 开头两条线一起往下掉，空隙一直很小 —— 那是健康的。继续拖：training 那条线继续掉，可 validation 那条到了最低点就往回抬。它一往上抬，空隙就开始变大。
t3: train–val 的那道缝**就是**被看见的 overfitting，而最好的模型正好在 validation 那条线最低的地方 —— 就在缝张开之前。
%%%

#### 读那道缝 —— 老手的习惯
三个动作，真的就这些。这张图你在往后的机器学习生涯里会一直盯着看，所以在这里慢一点是值得的。

%%% steps
step: 读那道缝，不是读两条曲线
why: train 和 validation 的 loss 之间那道缝是一个数字 —— validation 减 training —— 这意味着你只要看一件事，不用看两件
step: 小而稳 = 在学
why: 两条曲线一起往下掉的时候，那道缝几乎不变，因此模型正在找到那些在它从没学过的数据上也管用的规律 —— 这一段你会希望它持续得越久越好
step: 缝变宽 = 在背
why: 当 validation 抬头而 training 继续往下掉，多出来的本事全进了 training 那一堆 —— 这就是为什么变宽的缝意味着背答案，也是为什么盯着那道缝就能不靠任何额外工具看见 overfitting
step: 在 validation 最低的那一点停下
why: 那个 epoch 就是最佳点 —— 新数据上最好的模型 —— 所以你把那个 checkpoint 存下来然后停，这件事有个名字：early stopping（早停）
%%%

%%% insight
注意这个超能力有多便宜。没有新数学，没有新库 —— 只是把第二条曲线也画出来。大多数「我的模型就是跑不起来」的人，只是从来没看过 validation 那条曲线。所以他们要么停得太早，要么在最好的模型之后又练了很久。而且他们自己完全不知道。你不会是他们中的一个。
%%%

到这里我们给出了一张图和一个习惯：盯住两条线之间那块空隙，留住 validation 最低的那个 epoch。真实的训练曲线还多一个实用的小皱褶。

真实曲线是坑坑洼洼的，所以单独一个 epoch 可能纯靠运气往下掉或者往上跳。这就是为什么你不在第一次抬头就停。你给它一个**耐心窗口** —— 「如果 validation 连着比如 5 个 epoch 都没变好，就停下来，回到最好的那个 checkpoint」。

Early stopping 是一家族反背答案花招里的一员，这家族叫 [[regularization（正则化）||一些不鼓励模型去背答案的技术 —— early stopping、weight decay、dropout、数据增强。今天你学的是*看见* overfitting；后面的训练课会把这些解法搭起来]]，这家族剩下的成员在训练课里等着。

胜利一圈：你现在能看着两条曲线，说出一个模型在干什么 —— 在学、在背，还是卡住了。接下来：split 本身悄悄骗你的那些方式，就算曲线看起来很漂亮也一样。
~~~

@@@ concept id=c6 zh_tag="偷看和洗牌" tag="Peek & shuffle" zh_title="split 骗你的两种方式：偷看，和排好序的那一刀" title="Two ways a split lies: the peek and the sorted slice" zh_gotit="懂了这两种" gotit="Got these two"
Even with three piles, a split can quietly betray you — and the two traps in this unit are the ones that catch careful people. Think about dealing cards for a game with your friends. Two house rules make the deal fair: you **shuffle** the deck first, and nobody **peeks** at the pile you set aside. Break either one and the game is rigged, even if everybody smiles politely. A data split has exactly the same two house rules.
~~~zh
就算有了三堆，一次 split 还是可能悄悄背叛你 —— 而这一节里的两个陷阱，专门抓那些很仔细的人。想想你和朋友玩牌的时候发牌。有两条桌上的规矩让这次发牌是公平的：你先**洗牌**，而且没有人**偷看**你放在一边的那一叠。破了任何一条，这局就是做局，哪怕大家都还礼貌地笑着。数据的 split 有完全一样的两条规矩。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A card game as an analogy for a fair split. Two house rules are drawn: on the left a shuffled deck so no hand is stacked, and on the right a sealed pile with a padlock that nobody peeks at. A caption says in data these two rules become shuffle before you slice and never peek at the test pile."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A fair deal has two house rules — break one, the game is rigged</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">公平的一次发牌有两条规矩 —— 破一条，这局就是做局</text><g><rect x="30" y="30" width="210" height="120" rx="8" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="135" y="52" text-anchor="middle" fill="#1F6280" font-size="10">🃏 shuffle the deck first</text><text class="lang-zh" x="135" y="52" text-anchor="middle" fill="#1F6280" font-size="10">🃏 先把牌洗一遍</text><g fill="#2A7B9B"><rect x="66" y="70" width="20" height="30" rx="3" transform="rotate(-8 76 85)"/><rect x="96" y="76" width="20" height="30" rx="3" transform="rotate(5 106 91)"/><rect x="126" y="68" width="20" height="30" rx="3" transform="rotate(-3 136 83)"/><rect x="156" y="74" width="20" height="30" rx="3" transform="rotate(9 166 89)"/></g><text class="lang-en" x="135" y="132" text-anchor="middle" fill="#6B645E" font-size="8.5">so every hand is a fair mix</text><text class="lang-zh" x="135" y="132" text-anchor="middle" fill="#6B645E" font-size="8.5">这样每一手牌都是公平的混合</text></g><g><rect x="280" y="30" width="210" height="120" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="385" y="52" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 nobody peeks at the set-aside pile</text><text class="lang-zh" x="385" y="52" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 没有人偷看放在一边的那一叠</text><rect x="356" y="64" width="58" height="42" rx="3" fill="#fff" stroke="#2D8B55"/><text x="385" y="92" text-anchor="middle" font-size="17">🔒</text><text class="lang-en" x="385" y="132" text-anchor="middle" fill="#6B645E" font-size="8.5">it is opened once, at the end</text><text class="lang-zh" x="385" y="132" text-anchor="middle" fill="#6B645E" font-size="8.5">它只在最后被拆开一次</text></g><text class="lang-en" x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">In data these become: shuffle before you slice · never peek at the test pile.</text><text class="lang-zh" x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">在数据里这两条变成：切之前先洗牌 · 永远不偷看 test 那一堆。</text></g></svg>
%%%

**What the card-deal picture gets right:** fairness comes from two habits you apply *before* play starts — shuffling, and leaving the set-aside pile alone. Exactly like a split.

**Where it breaks down:** at a card table you can *see* a cheat. In data these two leaks leave no mark at all — a peek and a sorted slice both produce perfectly normal-looking code, so you prevent them by discipline, not by spotting them later.

#### Trap 1 — the peek
Back to the sealed pile. Your test set is honest for exactly one reason: it has never influenced anything. **Peeking** breaks that in the gentlest imaginable way — you run the test set, look at the number, and think "hmm, let me try a slightly bigger model." Nothing was copied. Nobody cheated. But a little of the answer key just moved into your head, and from there into your next decision.

- *Cause:* you make decisions by **looking at the test score** — even once. Tuning on the test, choosing the best epoch on the test, picking a threshold on the test: all the same leak.
- *Fix:* **keep the test untouched** and **use it only once**, right at the end, as a report. Every choice gets made on the validation pile instead. If you truly must keep iterating after you have opened it, the honest move is to collect a *fresh* test set.

%%% insight
Here is the uncomfortable question that makes this real: how would you ever *know* it happened? There is no error message for peeking. The score just looks a bit nicer than the model deserves, and every graph in your report agrees with you. That is why this one is a *habit*, not a check — you protect the sealed pile because you decided to, before you ever ran the model.
%%%

#### Trap 2 — the sorted slice
Now the other house rule: the shuffle. Files arrive in an order — sorted by label, by date, by customer id, by whatever the export happened to do. If you slice **ordered data** straight down the middle, each pile gets a *lopsided* slice of the world instead of a fair sample. Let us see how bad that gets, with a spam-filter file that came out of the database **sorted** by label.

%%% demo id=sorted label="predict, then reveal — slicing a sorted file"
predict: 500 emails, sorted: the 400 ordinary ones first, then the 100 spam ones. You take the last 20% as your test pile — no shuffle. What do you think lands in that test pile, and what will the test score be?
code: rows = load_csv("emails.csv")   # 500 rows, SORTED by label
code: train, test = rows[:400], rows[400:]     # no shuffle — just cut the end off
code: Counter(r.label for r in train), Counter(r.label for r in test)
code: accuracy(model_trained_on(train), test)
out: train : Counter({'ordinary': 400})     ← not one single spam email
out: test  : Counter({'spam': 100})         ← nothing BUT spam
out: test accuracy : 0.00
take: <b>A perfect 0%.</b> The model never met a spam email while learning, then its exam was 100% spam — an <b>unrepresentative slice</b> at both ends. Notice it is not the model's fault at all: the split destroyed the experiment before training even started. (And on the *whole* file, the lazy "always ordinary" baseline would have scored 80% — which is why we always check the baseline on the same piles.)
%%%

The fix is one word: **shuffle before splitting**, so every pile is a fair sample of the whole. One extra move when a class is rare: shuffle in a way that keeps the label *mix* the same in every pile — that is called [[stratified splitting||shuffling and cutting so each pile keeps the same mix of classes as the whole dataset — the fix for a rare class that would otherwise miss a pile entirely]], and it is the fix for "my rare class vanished from validation." Read the two traps as before → after:
~~~zh
**发牌这张图对在哪里：** 公平来自两个你在开局*之前*就做好的习惯 —— 洗牌，和不去动放在一边那一叠。和 split 完全一样。

**它在哪里不成立：** 在牌桌上你*看得见*有人出手作弊。在数据里这两种泄漏一点痕迹都不留 —— 偷看和排好序的那一刀，写出来的代码都完全正常，所以你靠纪律防住它们，不是靠事后看出来。

#### 陷阱 1 —— 偷看
回到封起来的那一叠。你的 test set 之所以诚实，理由只有一个：它从来没影响过任何东西。**偷看**用你能想到的最温柔的方式破了这一点 —— 你跑了一遍 test set，看了那个数字，然后想「嗯，我试个稍微大一点的模型吧」。什么都没被复制。没有人作弊。可答案本的一小块刚刚搬进了你的脑子，再从那里搬进了你的下一个决定。

- *原因：* 你**看着 test 分数**做决定 —— 哪怕只看一次。在 test 上调参、在 test 上挑最好的 epoch、在 test 上选一个阈值：全是同一种泄漏。
- *办法：* **别碰 test**，而且**只用它一次**，就在最后，当成一份报告。每一个选择都改到 validation 那一堆上做。如果你真的在开过它之后还必须继续迭代，诚实的做法是再去收一份*新的* test set。

%%% insight
让这件事变得真实的是这个让人不舒服的问题：你到底怎么*知道*它发生了？偷看没有报错信息。分数只是看起来比模型该得的好那么一点，而你报告里的每一张图都跟你一个意见。这就是为什么这一条是一个*习惯*，不是一道检查 —— 你保护那封起来的一堆，是因为你在跑模型之前就决定要保护它。
%%%

#### 陷阱 2 —— 排好序的那一刀
现在说另一条规矩：洗牌。文件到你手上的时候是有顺序的 —— 按 label 排、按日期排、按客户编号排，或者按导出的时候恰好怎么弄就怎么排。如果你把**有顺序的数据**直接从中间一刀切开，每一堆拿到的是这个世界*偏了*的一片，不是一份公平的样本。我们来看这能坏到什么程度，用一个从数据库里按 label **排好序**导出来的垃圾邮件过滤文件。

%%% demo id=sortedzh label="先猜，再揭晓 —— 切一个排好序的文件"
predict: 500 封邮件，排好序：先是 400 封普通的，然后是 100 封垃圾的。你拿最后 20% 当 test 那一堆 —— 不洗牌。你觉得落进那一堆的是什么，test 分数会是多少？
code: rows = load_csv("emails.csv")   # 500 行，按 label 排好序
code: train, test = rows[:400], rows[400:]     # 不洗牌 —— 就把尾巴切下来
code: Counter(r.label for r in train), Counter(r.label for r in test)
code: accuracy(model_trained_on(train), test)
out: train : Counter({'ordinary': 400})     ← not one single spam email
out: test  : Counter({'spam': 100})         ← nothing BUT spam
out: test accuracy : 0.00
take: <b>完美的 0%。</b>模型在学的时候一封垃圾邮件都没见过，然后它的考卷是 100% 垃圾邮件 —— 两头都是**不具代表性的一片**。注意这完全不是模型的错：这次 split 在训练开始之前就把这个实验毁了。（而在*整个*文件上，那个懒惰的「永远说普通」的基线能拿 80% —— 这就是为什么我们总要在同样几堆上看一下基线。）
%%%

办法只有一个词：**先洗牌再切**，这样每一堆都是整体的一份公平样本。当某一类很少见的时候还要多一个动作：洗牌的方式要让每一堆里 label 的*比例*保持一样 —— 这叫 [[stratified splitting（分层划分）||洗牌和切开的时候让每一堆保持和整个数据集一样的类别比例 —— 专门解决那种少见的类别，不然它可能整堆都没有]]，它就是「我那个少见的类别从 validation 里消失了」的解法。把这两个陷阱按 之前 → 之后 读一遍：
~~~

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Two traps drawn as before-and-after pairs. Trap one, peeking: before, an arrow loops from the sealed test pile back into your tuning decisions, so the grade is contaminated; after, the test pile stays sealed and is used only once, so the grade is honest. Trap two, unshuffled: before, sorted data sliced straight sends all of one label into train and all of the other into test; after, shuffling before slicing gives every pile a fair mix of both labels."><g font-family="monospace" font-size="8.5"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Each trap: before (rigged) → after (fixed)</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">每个陷阱：之前（做局）→ 之后（修好）</text><text class="lang-en" x="150" y="32" text-anchor="middle" fill="#C93B3B" font-size="9">BEFORE — rigged</text><text class="lang-zh" x="150" y="32" text-anchor="middle" fill="#C93B3B" font-size="9">之前 —— 做局</text><text class="lang-en" x="392" y="32" text-anchor="middle" fill="#1a5c38" font-size="9">AFTER — fixed</text><text class="lang-zh" x="392" y="32" text-anchor="middle" fill="#1a5c38" font-size="9">之后 —— 修好</text><g><text class="lang-en" x="14" y="72" fill="#6B645E" font-size="8.5">① PEEKING</text><text class="lang-zh" x="14" y="72" fill="#6B645E" font-size="8.5">① 偷看</text><rect x="70" y="44" width="160" height="58" rx="6" fill="#FDECEC" stroke="#C93B3B"/><rect x="86" y="58" width="42" height="30" rx="3" fill="#fff" stroke="#C93B3B"/><text class="lang-en" x="107" y="77" text-anchor="middle" fill="#C93B3B" font-size="7.5">🔒 test</text><text class="lang-zh" x="107" y="77" text-anchor="middle" fill="#C93B3B" font-size="7.5">🔒 test</text><path d="M130 64 C 178 52, 200 82, 168 86" fill="none" stroke="#C93B3B" stroke-width="1.4"/><polygon points="168,86 176,82 176,90" fill="#C93B3B"/><text class="lang-en" x="196" y="66" fill="#C93B3B" font-size="7">you tune</text><text class="lang-zh" x="196" y="66" fill="#C93B3B" font-size="7">你对着它的</text><text class="lang-en" x="196" y="78" fill="#C93B3B" font-size="7">on its score 💥</text><text class="lang-zh" x="196" y="78" fill="#C93B3B" font-size="7">分数调参 💥</text><rect x="312" y="44" width="160" height="58" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><rect x="328" y="58" width="42" height="30" rx="3" fill="#fff" stroke="#2D8B55"/><text class="lang-en" x="349" y="77" text-anchor="middle" fill="#1a5c38" font-size="7.5">🔒 test</text><text class="lang-zh" x="349" y="77" text-anchor="middle" fill="#1a5c38" font-size="7.5">🔒 test</text><text class="lang-en" x="392" y="68" fill="#1a5c38" font-size="7">sealed — opened</text><text class="lang-zh" x="392" y="68" fill="#1a5c38" font-size="7">封着 —— 只在最后</text><text class="lang-en" x="392" y="80" fill="#1a5c38" font-size="7">ONCE, at the end ✓</text><text class="lang-zh" x="392" y="80" fill="#1a5c38" font-size="7">拆开一次 ✓</text></g><g><text class="lang-en" x="14" y="166" fill="#6B645E" font-size="8.5">② SORTED SLICE</text><text class="lang-zh" x="14" y="166" fill="#6B645E" font-size="8.5">② 排好序的那一刀</text><rect x="70" y="128" width="160" height="62" rx="6" fill="#FDECEC" stroke="#C93B3B"/><g fill="#2A7B9B"><rect x="86" y="144" width="12" height="18"/><rect x="100" y="144" width="12" height="18"/><rect x="114" y="144" width="12" height="18"/><rect x="128" y="144" width="12" height="18"/></g><g fill="#C99A12"><rect x="176" y="144" width="12" height="18"/><rect x="190" y="144" width="12" height="18"/><rect x="204" y="144" width="12" height="18"/></g><text class="lang-en" x="110" y="176" text-anchor="middle" fill="#1F6280" font-size="7">all ordinary → train</text><text class="lang-zh" x="110" y="176" text-anchor="middle" fill="#1F6280" font-size="7">普通的全进 train</text><text class="lang-en" x="192" y="176" text-anchor="middle" fill="#9A7208" font-size="7">all spam → test 💥</text><text class="lang-zh" x="192" y="176" text-anchor="middle" fill="#9A7208" font-size="7">垃圾的全进 test 💥</text><rect x="312" y="128" width="160" height="62" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><g><rect x="328" y="144" width="12" height="18" fill="#2A7B9B"/><rect x="342" y="144" width="12" height="18" fill="#C99A12"/><rect x="356" y="144" width="12" height="18" fill="#2A7B9B"/><rect x="370" y="144" width="12" height="18" fill="#2A7B9B"/><rect x="416" y="144" width="12" height="18" fill="#C99A12"/><rect x="430" y="144" width="12" height="18" fill="#2A7B9B"/><rect x="444" y="144" width="12" height="18" fill="#2A7B9B"/></g><text class="lang-en" x="392" y="176" text-anchor="middle" fill="#1a5c38" font-size="7">shuffle first → every pile a fair mix ✓</text><text class="lang-zh" x="392" y="176" text-anchor="middle" fill="#1a5c38" font-size="7">先洗牌 → 每一堆都是公平的混合 ✓</text></g><text class="lang-en" x="260" y="210" text-anchor="middle" fill="#6B645E" font-size="8.5">Same shape both times: information ends up in the wrong pile. Same cure: one habit, applied up front.</text><text class="lang-zh" x="260" y="210" text-anchor="middle" fill="#6B645E" font-size="8.5">两次的形状一样：信息进了错的那一堆。药也一样：一个习惯，提前用上。</text></g></svg>
%%%

Two traps down, and look how cheap the cures are: seal the test pile, shuffle before you slice. You have just learned to survive the two mistakes that quietly ruin more beginner projects than any modelling error. The next one is different, though — it hides *inside* code you wrote yourself, and it looks completely innocent.
~~~zh
两个陷阱搞定，看看这两副药有多便宜：把 test 那一堆封好，切之前先洗牌。你刚刚学会了怎么活过这两个错误 —— 它们悄悄毁掉的新手项目，比任何建模错误都多。不过下一个不一样 —— 它藏在你自己写的代码*里面*，而且看起来完全无害。
~~~

@@@ concept id=c7 zh_tag="藏起来的泄漏" tag="The hidden leak" zh_title="藏在你自己那几行好心代码里的泄漏" title="The leak inside your own innocent code" zh_gotit="懂了 leakage" gotit="Got leakage"
This one is my favourite, because the code that causes it looks *helpful*. Imagine a game where three friends hide behind a curtain and you must guess their heights. To guess well you first work out the **average height of everyone in the room** — and, without thinking, you include the three hidden friends in that average. Now your measuring stick secretly carries information about the very people you were supposed to guess. You did not look behind the curtain. You did not cheat. But your ruler did.

That is [[leakage||when information from the validation or test piles sneaks into training — through a shared calculation, a duplicate row, or anything else the model should not have known. It always shows up as a suspiciously good score]], and it is the trap that catches people who have already learned the other two.
~~~zh
这一个是我最喜欢的，因为造成它的代码看起来很*好心*。想象一个游戏：三个朋友躲在帘子后面，你要猜他们的身高。为了猜得准，你先算出**房间里所有人的平均身高** —— 而且你没多想，就把那三个躲起来的朋友也算进了这个平均里。现在你的量尺上偷偷带着关于你本该去猜的那几个人的信息。你没有看帘子后面。你没有作弊。可你的尺子看了。

那就是 [[leakage（泄漏）||validation 或 test 那几堆的信息偷偷进了训练里 —— 通过一个共用的计算、一行重复的数据，或者别的任何模型本不该知道的东西。它总是表现成一个好得可疑的分数]]。这个陷阱专门抓那些已经学会另外两个的人。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A guessing game as an analogy for preprocessing leakage. On the left, six visible friends stand in the open, labelled the training pile. On the right, three friends hide behind a curtain, labelled the sealed pile you must guess. Arrows run from all nine friends, including the three hidden ones, into a box labelled measuring stick built from the average of everyone. A red caption warns the stick now secretly knows about the hidden friends."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Your ruler peeked, even though you did not</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">你的尺子偷看了，尽管你没有</text><g stroke="#2A7B9B" fill="#2A7B9B"><g><circle cx="40" cy="52" r="6"/><line x1="40" y1="58" x2="40" y2="76" stroke-width="2"/></g><g><circle cx="66" cy="46" r="6"/><line x1="66" y1="52" x2="66" y2="76" stroke-width="2"/></g><g><circle cx="92" cy="56" r="6"/><line x1="92" y1="62" x2="92" y2="76" stroke-width="2"/></g><g><circle cx="118" cy="48" r="6"/><line x1="118" y1="54" x2="118" y2="76" stroke-width="2"/></g><g><circle cx="144" cy="54" r="6"/><line x1="144" y1="60" x2="144" y2="76" stroke-width="2"/></g><g><circle cx="170" cy="44" r="6"/><line x1="170" y1="50" x2="170" y2="76" stroke-width="2"/></g></g><line x1="26" y1="78" x2="184" y2="78" stroke="#B8AEA2"/><text class="lang-en" x="105" y="92" text-anchor="middle" fill="#1F6280" font-size="8.5">the 6 you can see = training pile</text><text class="lang-zh" x="105" y="92" text-anchor="middle" fill="#1F6280" font-size="8.5">你看得见的那 6 个 = 训练那一堆</text><rect x="300" y="30" width="150" height="50" rx="3" fill="#E7DCEB" stroke="#7A5A8E"/><text class="lang-en" x="375" y="50" text-anchor="middle" fill="#5C4270" font-size="9">🎭 curtain</text><text class="lang-zh" x="375" y="50" text-anchor="middle" fill="#5C4270" font-size="9">🎭 帘子</text><text class="lang-en" x="375" y="66" text-anchor="middle" fill="#5C4270" font-size="8">3 hidden friends</text><text class="lang-zh" x="375" y="66" text-anchor="middle" fill="#5C4270" font-size="8">3 个躲起来的朋友</text><text class="lang-en" x="375" y="92" text-anchor="middle" fill="#5C4270" font-size="8.5">the sealed pile you must guess</text><text class="lang-zh" x="375" y="92" text-anchor="middle" fill="#5C4270" font-size="8.5">你必须去猜的、封起来的那一堆</text><rect x="150" y="118" width="220" height="34" rx="6" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="260" y="139" text-anchor="middle" fill="#9A7208" font-size="9">📏 measuring stick = average of EVERYONE</text><text class="lang-zh" x="260" y="139" text-anchor="middle" fill="#9A7208" font-size="9">📏 量尺 = 所有人的平均</text><g stroke="#C99A12" stroke-width="1.3" fill="none"><path d="M105 96 L 200 116"/><path d="M375 96 L 330 116"/></g><polygon points="200,116 192,110 195,119" fill="#C99A12"/><polygon points="330,116 336,109 338,118" fill="#C93B3B"/><text class="lang-en" x="392" y="112" fill="#C93B3B" font-size="8">this arrow is the leak 💥</text><text class="lang-zh" x="392" y="112" fill="#C93B3B" font-size="8">这个箭头就是泄漏 💥</text><text class="lang-en" x="260" y="172" text-anchor="middle" fill="#C93B3B" font-size="9">The stick now secretly knows about the hidden three —</text><text class="lang-zh" x="260" y="172" text-anchor="middle" fill="#C93B3B" font-size="9">这把尺子现在偷偷知道藏起来那三个人的事 ——</text><text class="lang-en" x="260" y="188" text-anchor="middle" fill="#C93B3B" font-size="9">so your guesses look better than your skill deserves.</text><text class="lang-zh" x="260" y="188" text-anchor="middle" fill="#C93B3B" font-size="9">所以你的猜测看起来比你的本事该得的更好。</text></g></svg>
%%%

**What the curtain picture gets right:** you can leak information through a *shared calculation*, without ever looking at the hidden data yourself. The ruler is the leak.

**Where it breaks down:** in the game you know exactly who is behind the curtain. In real code the "average of everyone" is one innocent line that ran before you split — nothing in the output looks wrong, which is why this trap survives so long.

#### Costume 1 — the scaler that saw everything
In real pipelines that measuring stick is a **preprocessing** step: scaling numbers by the average and spread, filling in blanks with the column mean, building a vocabulary, picking the "best" features. Every one of those has to *look at data* to decide its numbers. Do it on the whole file first and you have handed the model a whisper of the eval piles. So the order matters more than the code:

%%% steps
step: split FIRST
why: before any transformation runs, cut the data into train / validation / test — therefore no later step can possibly mix them
step: fit the transform on TRAIN only
why: the average, the spread, the vocabulary all get computed from training rows alone, which means your ruler is built from data the model is allowed to know
step: then apply (never re-fit) to validation and test
why: the eval piles get *transformed* using the training numbers — that's why they stay honest: they influenced nothing
step: wrap the whole thing in a pipeline
why: a pipeline makes the order un-breakable, so nobody on your team can accidentally preprocess after splitting in the wrong order six months from now
%%%

Short version to keep in your pocket: **split first, preprocess after splitting** — and fit on train only.

#### Costume 2 — the row that shows up twice
Now the other everyday costume, and it is even simpler: the **same rows in more than one** pile. Exports get joined twice, logs get re-uploaded, a customer gets saved on two dates. Suddenly a row is sitting in train *and* in test — like finding the 7♠ in two different hands. When the exam arrives, the model does not predict at all. It *recites*.

%%% demo id=dupes label="predict, then reveal — the export with duplicate rows"
predict: 500 rows in your export, but a botched join copied 125 of them, so those 125 rows each appear twice. You shuffle and cut 80/10/10 (so train 400, val 50, test 50). Guess: how many of your 50 test rows also sit in the training pile — and will the score look better or worse than the truth?
code: rows = load_csv("export.csv")               # 500 rows · 125 of them are exact copies
code: train, val, test = split(shuffle(rows))     # 400 / 50 / 50
code: len([r for r in test if r in train])        # the two-line check from earlier
code: accuracy(model, test)                       # the score you would have reported
out: rows in BOTH train and test : 20   of 50
out: reported test accuracy      : 0.89   🎉
out: after de-duplicating first  : 0.81
take: <b>20 of your 50 exam rows were rows the model had already studied.</b> It recited those and genuinely guessed the other 30 — so the reported 0.89 is the honest 0.81 with <b>8 points of pure fiction</b> stirred in. And notice the direction: leakage never makes your score look bad. It always arrives disguised as good news.
%%%

#### Your turn: drag the leak
Numbers land better when you cause them. The dial below is how many of the 50 test rows are duplicates the model already studied. Predict first: to gain 8 fake points, how many leaked rows do you think it takes? Then drag and watch the reported score float away from the truth.
~~~zh
**帘子这张图对在哪里：** 你可以通过一个*共用的计算*漏出信息，自己完全没有去看那些藏起来的数据。尺子就是那个泄漏。

**它在哪里不成立：** 游戏里你清楚地知道帘子后面是谁。真代码里，「所有人的平均」是在你 split 之前跑过的一行无害的代码 —— 输出里没有任何东西看起来不对，这就是为什么这个陷阱能活这么久。

#### 第 1 副行头 —— 那个什么都看过的 scaler
在真的流程里，那把量尺是一步**预处理**：按平均值和离散程度缩放数字、用列的均值填空、建一份词表、挑「最好」的特征。这里每一个都必须*看数据*才能定出自己的数字。先在整个文件上做，你就已经把评估那几堆的一点耳语递给了模型。所以顺序比代码更要紧：

%%% steps
step: 先 split
why: 在任何变换跑起来之前，把数据切成 train / validation / test —— 因此后面任何一步都不可能把它们混起来
step: 只在 train 上拟合这个变换
why: 平均值、离散程度、词表全部只从训练那些行算出来，这意味着你的尺子是用模型被允许知道的数据造的
step: 然后对 validation 和 test 只做应用，绝不重新拟合
why: 评估那几堆是用训练那边的数字被*变换*的 —— 这就是它们保持诚实的原因：它们什么都没影响
step: 把整件事包进一个 pipeline
why: pipeline 让这个顺序变得不可能被打破，所以半年之后，你团队里没有人会不小心把预处理和 split 的顺序做反
%%%

放在口袋里的短版：**先 split，split 之后再预处理** —— 而且只在 train 上拟合。

#### 第 2 副行头 —— 出现了两次的那一行
现在说另一副日常行头，它更简单：**同一行数据出现在不止一堆里**。导出被连接了两次、日志被重新上传、一个客户在两个日期各存了一次。突然之间，一行数据同时坐在 train *和* test 里 —— 就像在两手不同的牌里都找到了黑桃 7。考卷来的时候，模型根本不是在预测。它在*背诵*。

%%% demo id=dupeszh label="先猜，再揭晓 —— 带重复行的那份导出"
predict: 你的导出里有 500 行，可一次写坏的连接复制了其中 125 行，所以那 125 行各出现两次。你洗牌后切 80/10/10（也就是 train 400、val 50、test 50）。猜猜：你那 50 行 test 里，有多少行同时也坐在训练那一堆里 —— 分数会看起来比真相更好还是更差？
code: rows = load_csv("export.csv")               # 500 行 · 其中 125 行是完全一样的副本
code: train, val, test = split(shuffle(rows))     # 400 / 50 / 50
code: len([r for r in test if r in train])        # 前面那道两行的检查
code: accuracy(model, test)                       # 你本来会报出去的那个分数
out: rows in BOTH train and test : 20   of 50
out: reported test accuracy      : 0.89   🎉
out: after de-duplicating first  : 0.81
take: <b>你那 50 行考题里，有 20 行是模型早就学过的。</b>它把那些背了出来，另外 30 行是真的在猜 —— 所以报出来的 0.89 是诚实的 0.81 里搅进了<b>纯粹白拿的 8 分</b>。还要注意方向：leakage 从来不会让你的分数看起来差。它总是打扮成好消息来找你。
%%%

#### 换你来：拖那个泄漏
数字是你自己造出来的时候，才真的落进心里。下面这个旋钮是那 50 行 test 里有多少行是模型已经学过的重复行。先预测：要白拿 8 分假分数，你觉得需要几行泄漏？然后拖，看着报出来的分数从真相那边飘走。
~~~

%%% svg
<svg id="dup-svg" viewBox="0 0 520 176" role="img" aria-label="An interactive bar showing how duplicate rows inflate a reported score. A horizontal bar grows as you drag the number of leaked duplicate rows from zero to twenty-five. The part of the bar up to the honest score of zero point eight one is drawn in green, and any part beyond it is drawn in red and labelled fiction. A dashed vertical marker shows the honest score. With zero leaked rows the bar stops exactly at the marker; with twenty leaked rows it reaches zero point eight nine, well past the marker."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Leaked duplicates only ever push the score UP</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">漏过来的重复行只会把分数往上推</text><line x1="70" y1="112" x2="470" y2="112" stroke="#B8AEA2"/><text x="70" y="126" text-anchor="middle" fill="#6B645E" font-size="8">0.0</text><text x="470" y="126" text-anchor="middle" fill="#6B645E" font-size="8">1.0</text><rect id="dup-ok" x="70" y="60" width="324" height="34" rx="3" fill="#2D8B55"/><rect id="dup-lie" x="394" y="60" width="0" height="34" rx="3" fill="#C93B3B"/><line x1="394" y1="46" x2="394" y2="108" stroke="#2C2A28" stroke-width="1.4" stroke-dasharray="4,3"/><text class="lang-en" x="394" y="42" text-anchor="middle" fill="#2C2A28" font-size="8.5">honest score 0.81</text><text class="lang-zh" x="394" y="42" text-anchor="middle" fill="#2C2A28" font-size="8.5">诚实的分数 0.81</text><text class="lang-en" id="dup-lbl" x="86" y="82" fill="#fff" font-size="9">reported 0.81</text><text class="lang-zh" id="dup-lbl-zh" x="86" y="82" fill="#fff" font-size="9">报出 0.81</text><text class="lang-en" x="260" y="152" text-anchor="middle" fill="#6B645E" font-size="8.5">green = skill the model really has · red = rows it simply recited</text><text class="lang-zh" x="260" y="152" text-anchor="middle" fill="#6B645E" font-size="8.5">绿 = 模型真有的本事 · 红 = 它只是背出来的行</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label><span class="lang-en">leaked duplicate rows in the 50-row test pile</span><span class="lang-zh">50 行 test 那一堆里漏过来的重复行</span> <input id="dup-r" type="range" min="0" max="25" step="1" value="0" style="width:38%;accent-color:#C93B3B;vertical-align:middle"> <b id="dup-v">0</b></label>
<div id="dup-out" class="lang-en" style="margin-top:6px;color:#2C2A28">0 leaked rows → reported <b>0.81</b>, which is the truth. This is what a clean split looks like ✓</div>
<div id="dup-out-zh" class="lang-zh" style="margin-top:6px;color:#2C2A28">0 行漏过来 → 报出 <b>0.81</b>，这就是真相。干净的 split 就是这个样子 ✓</div>
</div>
<script>(function(){
  var r=document.getElementById('dup-r');if(!r)return;
  var ok=document.getElementById('dup-ok'),lie=document.getElementById('dup-lie'),
      lbl=document.getElementById('dup-lbl'),out=document.getElementById('dup-out'),v=document.getElementById('dup-v'),
      lblz=document.getElementById('dup-lbl-zh'),outz=document.getElementById('dup-out-zh');
  var X0=70,W=400,HONEST=0.81,NTEST=50;
  function paint(){
    var k=+r.value;
    // leaked rows are recited (counted correct); the rest score at the honest rate
    var s=(HONEST*(NTEST-k)+k)/NTEST, fake=Math.round((s-HONEST)*100);
    ok.setAttribute('width',(HONEST*W).toFixed(1));
    lie.setAttribute('x',(X0+HONEST*W).toFixed(1));
    lie.setAttribute('width',((s-HONEST)*W).toFixed(1));
    lbl.textContent='reported '+s.toFixed(2);
    lblz.textContent='报出 '+s.toFixed(2);
    v.textContent=k;
    if(k===0){out.innerHTML='0 leaked rows → reported <b>0.81</b>, which is the truth. This is what a clean split looks like ✓';out.style.color='#1a5c38';
              outz.innerHTML='0 行漏过来 → 报出 <b>0.81</b>，这就是真相。干净的 split 就是这个样子 ✓';outz.style.color='#1a5c38';}
    else{out.innerHTML=k+' leaked row'+(k===1?'':'s')+' → reported <b>'+s.toFixed(2)+'</b> = honest 0.81 + <b>'+fake+' point'+(fake===1?'':'s')+' of fiction</b> 💥 (the model recited '+k+' of the 50 exam rows)';out.style.color='#C93B3B';
         outz.innerHTML=k+' 行漏过来 → 报出 <b>'+s.toFixed(2)+'</b> = 诚实的 0.81 + <b>白拿的 '+fake+' 分</b> 💥（模型把 50 行考题里的 '+k+' 行背了出来）';outz.style.color='#C93B3B';}
  }
  r.addEventListener('input',paint);paint();
})();</script>
%%%

%%% insight
Notice what the red block means for your working life: leakage never announces itself with a bad number. It hands you your best result yet. So the day your score suddenly jumps and you feel delighted is exactly the day to run the two-line overlap check from earlier and ask "what did I just accidentally tell the model?" Suspicion of good news is a real professional instinct, and you now have it.
%%%

The fix is the same shape as before: **split first**, de-duplicate across the piles so no row leaks across splits, and fit every transform on train only.

#### The same leak in three more costumes
Leakage is one idea wearing different clothes. Three you will genuinely meet:

%%% table
:: The costume :: What leaks :: The one-move fix
the same person in two piles :: rows are not independent — one user's or patient's records land in train and test :: split by **group id** (group-aware splitting), never row by row
predicting the past from the future :: a random shuffle of time-ordered data lets the model study tomorrow to answer yesterday :: split by **time** (train on the past, validate on the future)
the rare class that vanished :: a class so rare it misses a pile — and "always say no" scores 99% on a 99%-no dataset :: **stratified splitting** + always read the per-class counts, never accuracy alone
%%%

Two smaller habits, one line each: **pin a random seed and save your split indices**, so the piles do not silently change between runs (drifting splits leak everything into training eventually, and make two runs impossible to compare). And if you score *hundreds* of settings against one validation pile, the winner is partly luck — rotate the validation pile (next concept) or, when the tuning itself must be graded, use **nested cross-validation**, a later-lesson tool.

That is the whole leakage family, and every single member is beaten by habits you now know. One failure left — and this one happens even when you do everything right.
~~~zh
%%% insight
注意那块红色对你以后的工作意味着什么：leakage 从来不会用一个难看的数字宣布自己。它递给你的是你到目前为止最好的结果。所以你分数突然跳上去、你很高兴的那一天，正是该跑一下前面那道两行重叠检查的那一天，然后问自己「我刚刚不小心告诉了模型什么？」对好消息保持怀疑，是一个真正的专业本能，而你现在有了。
%%%

办法和之前是同一个形状：**先 split**，跨着几堆去重，别让任何一行漏过 split，还有每一个变换都只在 train 上拟合。

#### 同一个泄漏，还有三副行头
Leakage 是一个想法穿着不同的衣服。有三副你真的会碰到：

%%% table
:: 这副行头 :: 漏的是什么 :: 一招就能解决
同一个人出现在两堆里 :: 行与行不是独立的 —— 同一个用户或病人的记录落进了 train 和 test :: 按**分组编号**切（group-aware splitting），不要一行一行地切
拿未来去预测过去 :: 把按时间排列的数据随机洗牌，会让模型学明天去答昨天 :: 按**时间**切（在过去上训练，在未来上验证）
那个消失了的少见类别 :: 一个类别少到整堆都没有。在一个 99% 都是「否」的数据集上，「永远说否」能拿 99% :: **stratified splitting** + 永远去看每个类别各自的计数，不要只看 accuracy
%%%

还有两个小一点的习惯，各一行。第一，**固定一个随机种子，并把你的 split 下标存下来**。这样两次运行之间那几堆不会悄悄变。会漂移的 split 最后会把一切都漏进训练里，还让两次运行没法比较。第二，如果你拿*上百*套设置去对着同一堆 validation 打分，赢的那一套里有一部分是运气。办法是换着轮流做 validation 那一堆，也就是下一个概念。当调参本身也需要被评分时，就用 **nested cross-validation**，那是后面课里的工具。

这就是整个 leakage 家族，而它的每一个成员都被你现在已经知道的习惯打败了。还剩一次失败 —— 而这一次，就算你什么都做对了，它也会发生。
~~~

@@@ concept id=c8 zh_tag="运气差的那一块" tag="Unlucky slice" zh_title="运气差的那一块 —— 和 cross-validation" title="The unlucky slice — and cross-validation" zh_gotit="懂了 cross-validation" gotit="Got cross-validation"
Do everything right — shuffle, seal, split first — and a small split can still be simply **unlucky**. Picture judging a whole cake by tasting *one* slice. If that slice happens to contain the only raisin, you will announce "this cake is raisin-heavy!" and be completely wrong. Not dishonest. Just unlucky.

A small validation pile does the same thing to your model. One random slice can flatter it (an easy slice — the score looks great) or punish it (a hard slice — the score looks terrible), and neither number is the truth. The **cause** is that a small pile is *high-variance*: change which examples landed in it and the score jumps around.
~~~zh
什么都做对了 —— 洗牌、封好、先 split —— 一个小的 split 还是可能纯粹**运气差**。想象你只尝*一*块，就要评价一整个蛋糕。如果那一块刚好含着唯一那颗葡萄干，你就会宣布「这个蛋糕葡萄干好多！」，而你完全错了。不是不诚实。只是运气差。

一个小的 validation 堆对你的模型做的就是这件事。随机的一块可能捧它（一块简单的 —— 分数看着很棒），也可能整它（一块难的 —— 分数看着很糟），而这两个数字都不是真相。**原因**是小的一堆*方差大*：换一批样本落进去，分数就跳来跳去。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A cake cut into five slices as an analogy for an unlucky validation split. The whole cake is drawn on the left with one raisin in a single slice. That one slice sits on a plate on the right, with the raisin in it, and a taster declares the whole cake is raisin-heavy. A caption notes that one slice can badly mislead you about the whole cake."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Judging a whole cake from one slice</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">只尝一块，就评价整个蛋糕</text><g><circle cx="120" cy="106" r="62" fill="#FDF3D6" stroke="#C99A12" stroke-width="1.6"/><g stroke="#C99A12" stroke-width="1.2"><line x1="120" y1="106" x2="182" y2="106"/><line x1="120" y1="106" x2="139" y2="47"/><line x1="120" y1="106" x2="70" y2="70"/><line x1="120" y1="106" x2="70" y2="142"/><line x1="120" y1="106" x2="139" y2="165"/></g><circle cx="150" cy="86" r="4.6" fill="#5C4270"/><text class="lang-en" x="120" y="184" text-anchor="middle" fill="#6B645E" font-size="8.5">the whole cake · exactly ONE raisin</text><text class="lang-zh" x="120" y="184" text-anchor="middle" fill="#6B645E" font-size="8.5">整个蛋糕 · 只有一颗葡萄干</text></g><path d="M300 60 L 372 92 L 300 124 Z" fill="#FDF3D6" stroke="#C99A12" stroke-width="1.6"/><circle cx="330" cy="92" r="4.6" fill="#5C4270"/><text class="lang-en" x="336" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">the one slice you tasted</text><text class="lang-zh" x="336" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">你尝的那一块</text><path d="M240 92 L 292 92" stroke="#6B645E" stroke-width="1.4"/><polygon points="292,92 284,88 284,96" fill="#6B645E"/><rect x="386" y="66" width="122" height="52" rx="8" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="447" y="88" text-anchor="middle" fill="#C93B3B" font-size="9">"this cake is</text><text class="lang-zh" x="447" y="88" text-anchor="middle" fill="#C93B3B" font-size="9">「这个蛋糕</text><text class="lang-en" x="447" y="104" text-anchor="middle" fill="#C93B3B" font-size="9">raisin-heavy!" 💥</text><text class="lang-zh" x="447" y="104" text-anchor="middle" fill="#C93B3B" font-size="9">葡萄干好多！」💥</text><text class="lang-en" x="260" y="176" text-anchor="middle" fill="#6B645E" font-size="9">One slice can be unusual — so one small validation pile can lie by pure luck.</text><text class="lang-zh" x="260" y="176" text-anchor="middle" fill="#6B645E" font-size="9">一块可能很特别 —— 所以小小一堆 validation 会纯靠运气骗你。</text></g></svg>
%%%

**What the cake picture gets right:** a single small sample can be unrepresentative through no fault of yours, and you would never know from the taste alone.

**Where it breaks down:** cake slices are the same size but data slices differ in *content*, and you can re-taste a cake for free — re-scoring a model on a new slice costs you a whole training run.

#### The remedy: taste every slice, then average
The fix has a name: [[cross-validation||a way to beat an unlucky split — rotate which slice is the validation pile across several rounds and average the scores, so no single lucky or unlucky slice decides the verdict]], and its everyday form is **k-fold**. Cut the data into k equal slices, then let each slice take its turn as the validation pile while the others train.

%%% steps
step: cut into k slices (k = 5 is a fine default)
why: each slice is a possible validation pile, which means no slice gets to be special
step: rotate which slice is held out, and train k times
why: every row is validated exactly once and trained on the other times, therefore a single unlucky slice cannot dominate the verdict
step: average the scores
why: the average of 5 wobbly numbers is far steadier than any one of them, which is why cross-validation is the standard move on small data
step: the price
why: it costs k full training runs — that's why on very large datasets people happily keep one big split instead
%%%
~~~zh
**蛋糕这张图对在哪里：** 一份小的样本可能不具代表性，而这完全不是你的错，你光靠尝也永远不会知道。

**它在哪里不成立：** 蛋糕的每一块大小一样，可数据的每一块*内容*不一样；而且蛋糕你可以免费再尝一次 —— 换一块重新给模型打分，要花掉你一整次训练。

#### 解药：每一块都尝，然后取平均
这个办法有名字：[[cross-validation（交叉验证）||打败运气差的 split 的一个办法 —— 分几轮，轮流让不同的一块当 validation 那一堆，再把分数平均起来，这样没有哪一块的好运或坏运能决定判决]]，它的日常形式叫 **k-fold**。把数据切成 k 块一样大的，然后让每一块轮流当 validation 那一堆，其余的用来训练。

%%% steps
step: 切成 k 块（k = 5 是个不错的默认值）
why: 每一块都是一个可能的 validation 堆，这意味着没有哪一块能当特殊的那个
step: 轮换留出哪一块，训练 k 次
why: 每一行都正好被验证一次、其余几次被训练，因此单独一块运气差的不可能主导判决
step: 把分数平均起来
why: 5 个晃动的数字的平均，比其中任何一个都稳得多，这就是为什么在小数据上 cross-validation 是标准动作
step: 代价
why: 它要花 k 次完整的训练 —— 这就是为什么在很大的数据集上，人们乐意换成保留一个大的 split
%%%
~~~

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Five rounds of k-fold cross-validation shown as five rows. The data is split into five equal slices. In each row a different slice is highlighted as the validation pile while the other four are training. The five validation scores are averaged at the bottom into one steady number, so no single unlucky slice decides the result."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">k-fold: rotate the validation slice, then average</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">k-fold：轮换 validation 那一块，然后取平均</text><g font-size="8.5"><g><text class="lang-en" x="40" y="46" fill="#6B645E">round 1</text><text class="lang-zh" x="40" y="46" fill="#6B645E">第 1 轮</text><rect x="88" y="34" width="70" height="16" fill="#C99A12"/><rect x="160" y="34" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="232" y="34" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="304" y="34" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="376" y="34" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/></g><g><text class="lang-en" x="40" y="70" fill="#6B645E">round 2</text><text class="lang-zh" x="40" y="70" fill="#6B645E">第 2 轮</text><rect x="88" y="58" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="160" y="58" width="70" height="16" fill="#C99A12"/><rect x="232" y="58" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="304" y="58" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="376" y="58" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/></g><g><text class="lang-en" x="40" y="94" fill="#6B645E">round 3</text><text class="lang-zh" x="40" y="94" fill="#6B645E">第 3 轮</text><rect x="88" y="82" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="160" y="82" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="232" y="82" width="70" height="16" fill="#C99A12"/><rect x="304" y="82" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="376" y="82" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/></g><g><text class="lang-en" x="40" y="118" fill="#6B645E">round 4</text><text class="lang-zh" x="40" y="118" fill="#6B645E">第 4 轮</text><rect x="88" y="106" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="160" y="106" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="232" y="106" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="304" y="106" width="70" height="16" fill="#C99A12"/><rect x="376" y="106" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/></g><g><text class="lang-en" x="40" y="142" fill="#6B645E">round 5</text><text class="lang-zh" x="40" y="142" fill="#6B645E">第 5 轮</text><rect x="88" y="130" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="160" y="130" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="232" y="130" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="304" y="130" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="376" y="130" width="70" height="16" fill="#C99A12"/></g></g><rect x="88" y="156" width="18" height="12" fill="#C99A12"/><text class="lang-en" x="112" y="166" fill="#9A7208" font-size="7.5">= validation this round</text><text class="lang-zh" x="112" y="166" fill="#9A7208" font-size="7.5">= 这一轮的 validation</text><line x1="88" y1="176" x2="446" y2="176" stroke="#B8AEA2" stroke-dasharray="2,2"/><text class="lang-en" x="260" y="192" text-anchor="middle" fill="#1a5c38" font-size="9">average the 5 scores → one steady, trustworthy number</text><text class="lang-zh" x="260" y="192" text-anchor="middle" fill="#1a5c38" font-size="9">把 5 个分数平均 → 一个稳的、可信的数字</text><text class="lang-en" x="260" y="206" text-anchor="middle" fill="#6B645E" font-size="8">no single lucky or unlucky slice decides the verdict</text><text class="lang-zh" x="260" y="206" text-anchor="middle" fill="#6B645E" font-size="8">没有哪一块的好运或坏运能决定判决</text></g></svg>
%%%

%%% insight
There is a lovely extreme version: set k to the number of rows, so each round holds out exactly *one* example. It is called leave-one-out, it squeezes the most out of tiny data — and it costs one full training run *per row*, which is why almost nobody runs it on anything big. Worth knowing it exists, mostly so you can smile at the price tag.
%%%

Two companion habits for the same wobble: if a holdout is so small that flipping one example moves the score by several points, **make it bigger** — and when you report a number, report the *spread* too (the same experiment across a few seeds or folds), so nobody mistakes noise for progress.

#### So how big should each pile be?
The question everyone asks, and the answer is friendlier than you expect. Common starting points are **70/15/15** or **80/10/10** — train / validation / test. But **no ratio is sacred**. A validation or test pile has just one requirement: be big enough to be a fair sample. With a tiny dataset you give eval a bigger share (or use k-fold instead). With **bigger data** you can give it a much **smaller fraction** — ten million rows and a 1% validation split is still 100,000 examples, which is plenty, so you keep almost everything for training. What matters is the *count*, not the percentage.
~~~zh
%%% insight
有一个很可爱的极端版本：把 k 设成行数，这样每一轮正好留出*一个*样本。它叫 leave-one-out，能把很少的数据榨到最尽 —— 而它的代价是*每一行*一次完整训练，所以几乎没有人在大一点的东西上跑它。值得知道它存在，主要是好让你对那个价签笑一笑。
%%%

对付同一种晃动，还有两个搭配的习惯：如果一个留出堆小到翻掉一个样本就能让分数动好几分，就**把它做大**；而当你报一个数字的时候，也**把散布报出来**（同一个实验换几个种子或者几折跑一遍），这样没有人会把噪声当成进展。

#### 那么每一堆到底该多大？
这是所有人都会问的问题，而答案比你想的友好。常见的起点是 **70/15/15** 或者 **80/10/10** —— train / validation / test。但**没有哪个比例是神圣的**。validation 或 test 那一堆只有一个要求：大到能算一份公平的样本。数据集很小的时候，你给评估更大的一份（或者干脆用 k-fold）。**数据更大**的时候，你可以只给它**小得多的一个比例** —— 一千万行，1% 的 validation 也还是 100,000 个样本，够多了，所以你几乎把全部留给训练。要紧的是那个*个数*，不是百分比。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Two horizontal bars showing common split ratios. The top bar is 70 percent train, 15 percent validation, 15 percent test. The bottom bar is 80 percent train, 10 percent validation, 10 percent test. A caption notes that with very large datasets even a 1 percent eval slice is plenty, so no ratio is sacred."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Common ratios — but no ratio is sacred</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">常见比例 —— 但没有哪个比例是神圣的</text><g><text x="16" y="58" fill="#6B645E" font-size="9">70/15/15</text><rect x="90" y="44" width="280" height="24" fill="#2A7B9B"/><rect x="370" y="44" width="60" height="24" fill="#C99A12"/><rect x="430" y="44" width="60" height="24" fill="#2D8B55"/><text class="lang-en" x="230" y="61" text-anchor="middle" fill="#fff" font-size="8.5">train 70%</text><text class="lang-zh" x="230" y="61" text-anchor="middle" fill="#fff" font-size="8.5">train 70%</text><text class="lang-en" x="400" y="61" text-anchor="middle" fill="#fff" font-size="7.5">val 15</text><text class="lang-zh" x="400" y="61" text-anchor="middle" fill="#fff" font-size="7.5">val 15</text><text class="lang-en" x="460" y="61" text-anchor="middle" fill="#fff" font-size="7.5">test 15</text><text class="lang-zh" x="460" y="61" text-anchor="middle" fill="#fff" font-size="7.5">test 15</text></g><g><text x="16" y="106" fill="#6B645E" font-size="9">80/10/10</text><rect x="90" y="92" width="320" height="24" fill="#2A7B9B"/><rect x="410" y="92" width="40" height="24" fill="#C99A12"/><rect x="450" y="92" width="40" height="24" fill="#2D8B55"/><text class="lang-en" x="250" y="109" text-anchor="middle" fill="#fff" font-size="8.5">train 80%</text><text class="lang-zh" x="250" y="109" text-anchor="middle" fill="#fff" font-size="8.5">train 80%</text><text class="lang-en" x="430" y="109" text-anchor="middle" fill="#fff" font-size="7">val</text><text class="lang-zh" x="430" y="109" text-anchor="middle" fill="#fff" font-size="7">val</text><text class="lang-en" x="470" y="109" text-anchor="middle" fill="#fff" font-size="7">test</text><text class="lang-zh" x="470" y="109" text-anchor="middle" fill="#fff" font-size="7">test</text></g><text class="lang-en" x="260" y="150" text-anchor="middle" fill="#1a5c38" font-size="8.5">Bigger data → a smaller fraction still makes a fair eval slice (even ~1%).</text><text class="lang-zh" x="260" y="150" text-anchor="middle" fill="#1a5c38" font-size="8.5">数据更大 → 更小的一个比例也够做一份公平的评估样本（连 1% 都行）。</text><text class="lang-en" x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="8.5">Keep most for study; hold enough back to grade honestly.</text><text class="lang-zh" x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="8.5">大部分留给学习；留住够多的，好诚实地打分。</text></g></svg>
%%%

So: an **unlucky split** is beaten by **cross-validation** — rotate which slice is validation and average the scores — and the exact ratio is a judgement call, not a law. One thing left, and it is the most grown-up idea of the day: what even a perfect split cannot do.
~~~zh
所以：**运气差的 split** 被 **cross-validation** 打败 —— 轮换哪一块当 validation，再把分数平均 —— 而具体的比例是一个判断，不是一条法律。还剩一件事，它是今天最成熟的那个想法：连一个完美的 split 也做不到什么。
~~~

@@@ concept id=c9 zh_tag="它做不到什么" tag="What it can't do" zh_title="一个好的 split 帮不了你什么" title="What a good split cannot do for you" zh_gotit="懂了这些边界" gotit="Got the limits"
A great split is honest — but honesty has limits, and knowing them is what stops you expecting magic. Think of a **thermometer**. It will tell you truthfully that you have a fever. It will not *make* you well. And it only reads correctly if you use it on the right thing — press it to a warm radiator and it happily reports 40°C about the wrong body entirely.

A data split is exactly that kind of instrument: it *reveals* the truth about your model, it does not *improve* the model, and its reading is trustworthy only under one condition.
~~~zh
一个很棒的 split 是诚实的 —— 可诚实有它的边界，知道这些边界才让你不去期待魔法。想想一支**体温计**。它会老老实实告诉你你在发烧。它不会*把*你治好。而且只有你量对了东西，它读出来的才对 —— 把它贴在一个温热的暖气片上，它会开心地报出 40°C，说的却是完全另一个身体。

数据的 split 正是这一类仪器：它*揭示*关于你的模型的真相，它不*改进*模型，而且它的读数只在一个条件下值得信。
~~~

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A thermometer as an analogy for a data split. The thermometer is drawn reading a high fever. Three labels point out that it tells you the honest truth, that it does not cure you, and that it only reads correctly when used on the right body. This mirrors a data split, which reveals whether a model generalizes but does not improve it."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A split is a thermometer: it MEASURES, it does not cure</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">split 是一支体温计：它量，它不治</text><rect x="120" y="36" width="16" height="96" rx="8" fill="#fff" stroke="#C93B3B" stroke-width="1.6"/><circle cx="128" cy="140" r="15" fill="#C93B3B"/><rect x="124" y="86" width="8" height="54" fill="#C93B3B"/><g stroke="#B8AEA2"><line x1="136" y1="52" x2="144" y2="52"/><line x1="136" y1="68" x2="144" y2="68"/><line x1="136" y1="84" x2="144" y2="84"/><line x1="136" y1="100" x2="144" y2="100"/></g><text class="lang-en" x="128" y="164" text-anchor="middle" fill="#C93B3B" font-size="8">reads: FEVER!</text><text class="lang-zh" x="128" y="164" text-anchor="middle" fill="#C93B3B" font-size="8">读数：发烧！</text><text class="lang-en" x="200" y="60" fill="#1a5c38" font-size="9">✓ tells you the honest truth</text><text class="lang-zh" x="200" y="60" fill="#1a5c38" font-size="9">✓ 告诉你诚实的真相</text><text class="lang-en" x="200" y="88" fill="#C93B3B" font-size="9">✗ but it does not CURE you</text><text class="lang-zh" x="200" y="88" fill="#C93B3B" font-size="9">✗ 但它不会治好你</text><text class="lang-en" x="200" y="116" fill="#C93B3B" font-size="9">✗ and only if used on the right body</text><text class="lang-zh" x="200" y="116" fill="#C93B3B" font-size="9">✗ 而且只有量对身体才算</text><text class="lang-en" x="200" y="144" fill="#6B645E" font-size="8">(a split is the same kind of instrument)</text><text class="lang-zh" x="200" y="144" fill="#6B645E" font-size="8">（split 是同一类仪器）</text></g></svg>
%%%

**What the thermometer picture gets right:** it hands you an honest reading you can act on — and the reading itself changes nothing.

**Where it breaks down:** a thermometer measures one simple thing. A split measures something subtler — *generalization to data like your splits* — so the "used on the right body" condition is much easier to forget.

#### Limit 1 — it diagnoses, it does not cure
Follow the thermometer one step further. It reads your fever perfectly, and reading is not healing; you still need medicine. A split is the same: it only reveals whether learning transferred to new data.

If your model is too simple to capture the pattern — trying to draw a curve with a straight ruler — a flawless three-way split will faithfully report "this does not generalize" and then stop, having done its one job. **The split diagnoses generalization; it does not create it.**

No clever splitting will *make* a weak model fit; that needs a better model, better features, or more data. The split just tells you honestly that you need them.

So far we have shown limit one: the thermometer reads the truth, and reading is not fixing. Limit two is about *which body* you press it to.

#### Limit 2 — a thermometer on the wrong body lies
Now the radiator. An instrument is only trustworthy when it measures the thing you care about, and a split's grade is honest **only for data like your splits** — drawn from the same world, the **same distribution**. If tomorrow's real data looks different (new users, a new season, a new camera, a new kind of question), your glowing test score was measuring the old world and **says nothing about a shifted future**.
~~~zh
**体温计这张图对在哪里：** 它给你一个你可以照着行动的诚实读数 —— 而这个读数本身什么都不改变。

**它在哪里不成立：** 体温计量的是一件简单的事。split 量的东西更微妙 —— *在和你那几堆同类的数据上的 generalization* —— 所以「量对了身体」这个条件更容易被忘掉。

#### 边界 1 —— 它诊断，它不治病
把体温计再往前跟一步。它把你的烧读得完美，而读数不等于治好；你还是需要药。split 一样：它只揭示学到的东西有没有迁移到新数据上。

如果你的模型太简单，抓不住那条规律，那就好比拿一把直尺去画曲线。这时一个挑不出毛病的三分 split 会老老实实报出「这个不 generalization」，然后就停在那里。它那一份活已经干完了。**split 诊断 generalization；它不创造 generalization。**

再聪明的切法也不会*让*一个弱模型贴合上去；那需要更好的模型、更好的特征，或者更多数据。split 只是诚实地告诉你，你需要它们。

到这里我们给出了边界一：体温计读出真相，而读数不等于修好。边界二讲的是你把它贴在*哪个身体*上。

#### 边界 2 —— 体温计贴错身体就会骗人
现在说那个暖气片。一件仪器只有在量的是你在意的那个东西时，才值得信。而一个 split 的成绩是诚实的，**只对和你那几堆同类的数据**而言。那些数据是从同一个世界、**同一个分布**里舀出来的。如果明天真实的数据看起来不一样了呢？比如新用户、新季节、新相机、新的一类问题。那时你那个亮眼的 test 分数量的是旧的那个世界，**对一个已经偏移的未来什么都没说**。
~~~

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A two-world diagram. On the left, World A: the pool your train, validation, and test splits are all drawn from, showing round blue shapes, with a caption that the test score is honest here. A large arrow crosses a dashed border to World B on the right: the shifted world the model actually meets after deployment, showing different triangular red shapes, with a caption that the honest test score says nothing here. The border is labelled distribution shift."><g font-family="monospace" font-size="9"><text class="lang-en" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A split only grades the world it was drawn from</text><text class="lang-zh" x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">split 只给它被舀出来的那个世界打分</text><g><rect x="18" y="30" width="200" height="150" rx="10" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="118" y="50" text-anchor="middle" fill="#1F6280" font-size="10">🌍 WORLD A</text><text class="lang-zh" x="118" y="50" text-anchor="middle" fill="#1F6280" font-size="10">🌍 世界 A</text><text class="lang-en" x="118" y="66" text-anchor="middle" fill="#6B645E" font-size="8">your train / val / test all come from here</text><text class="lang-zh" x="118" y="66" text-anchor="middle" fill="#6B645E" font-size="8">你的 train / val / test 全从这里来</text><g fill="#2A7B9B"><circle cx="60" cy="96" r="9"/><circle cx="96" cy="110" r="9"/><circle cx="140" cy="92" r="9"/><circle cx="170" cy="116" r="9"/><circle cx="82" cy="140" r="9"/><circle cx="132" cy="140" r="9"/></g><text class="lang-en" x="118" y="172" text-anchor="middle" fill="#1a5c38" font-size="8">test score is HONEST here ✓</text><text class="lang-zh" x="118" y="172" text-anchor="middle" fill="#1a5c38" font-size="8">test 分数在这里是诚实的 ✓</text></g><line x1="238" y1="26" x2="238" y2="184" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="4,4"/><text class="lang-en" x="238" y="196" text-anchor="middle" fill="#C93B3B" font-size="8">distribution shift</text><text class="lang-zh" x="238" y="196" text-anchor="middle" fill="#C93B3B" font-size="8">distribution shift</text><path d="M244 105 L 296 105" stroke="#6B645E" stroke-width="1.8"/><polygon points="296,105 286,100 286,110" fill="#6B645E"/><g><rect x="302" y="30" width="200" height="150" rx="10" fill="#FDECEC" stroke="#C93B3B"/><text class="lang-en" x="402" y="50" text-anchor="middle" fill="#C93B3B" font-size="10">🌍 WORLD B</text><text class="lang-zh" x="402" y="50" text-anchor="middle" fill="#C93B3B" font-size="10">🌍 世界 B</text><text class="lang-en" x="402" y="66" text-anchor="middle" fill="#6B645E" font-size="8">the shifted world after deployment</text><text class="lang-zh" x="402" y="66" text-anchor="middle" fill="#6B645E" font-size="8">上线之后那个偏移了的世界</text><g fill="#C93B3B"><polygon points="344,104 354,86 364,104"/><polygon points="384,116 394,98 404,116"/><polygon points="424,100 434,82 444,100"/><polygon points="364,146 374,128 384,146"/><polygon points="414,148 424,130 434,148"/></g><text class="lang-en" x="402" y="172" text-anchor="middle" fill="#C93B3B" font-size="8">the honest score says NOTHING here ✗</text><text class="lang-zh" x="402" y="172" text-anchor="middle" fill="#C93B3B" font-size="8">那个诚实的分数在这里什么都没说 ✗</text></g></g></svg>
%%%

**What the two-world drawing shows:** every pile you hold out is still scooped from the *same* World A, so it can only promise how you do on World-A-like data. The moment reality becomes World B, the promise expires.

That mismatch has a name: [[distribution shift||when the data a model meets in the real world drifts away from the data it was trained and tested on, so a great test score no longer predicts real performance — the subject of a later monitoring lesson]].

**Where the drawing breaks down:** real shift is usually gradual and partial — World B bleeds in slowly — which is why you keep watching in production instead of trusting one old test score forever. And one thing you *can* do up front: build your validation and test piles to match the users you will actually serve, on purpose, rather than whatever was easiest to collect.

%%% insight
This is the grown-up version of today's habit, and it is the sentence that separates a beginner from someone you would trust with a real system: "my test score is honest for the world I sampled." Not for every world. Saying that out loud — instead of quoting one number — is what confidence actually sounds like in this field.
%%%

#### Four more honest limits, quickly
None of these are reasons to stop splitting. They are the fine print, and knowing it is a flex.

%%% cards
🎲 | One number is not a guarantee | A holdout score is a point estimate with sampling noise. A different valid split gives a slightly different number, so treat it as an estimate with a wobble, not a promise.
🪞 | Splitting creates no information | If your labels are wrong or your collection was biased, all three piles are wrong and biased in the *same* direction. The split will cheerfully confirm a broken dataset.
🕵️ | A split cannot detect its own leakage | A leaked split does not look suspicious — it looks *excellent*. Leakage is caught by process and inspection (the overlap check, the pipeline order), never by the score.
🍰 | Held-out data is data you cannot learn from | Every example you hold out is one the model never gets to study. That is the small-data tension cross-validation exists to soften.
%%%

Two last honest limits worth saying plainly. A strong offline test score does **not** guarantee the product will improve, nor that the model works equally well for every group of users — that is what online tests and per-group scores are for, later. And a split tells you **that** the model is wrong, never **why** or **on which examples**; finding that out means looking at your errors one by one and scoring each slice of your data separately.

Victory lap: you now know what your instrument measures *and* where it stops working. Time to gather the whole day onto one page.
~~~zh
**两个世界这张图说明的是：** 你留出来的每一堆，都还是从*同一个*世界 A 里舀出来的，所以它只能承诺你在「像世界 A」的数据上做得怎么样。现实一旦变成世界 B，这个承诺就过期了。

这个不匹配有名字：[[distribution shift（分布偏移）||模型在真实世界里碰到的数据慢慢漂离了它训练和测试时用的数据，于是一个很好的 test 分数不再能预测真实表现 —— 这是后面一节监控课的主题]]。

**这张图在哪里不成立：** 真实的偏移通常是慢慢的、局部的。世界 B 是一点点渗进来的。所以你要在上线之后一直看着，而不是永远相信一个旧的 test 分数。还有一件你*提前*就能做的事：故意把你的 validation 和 test 那几堆做成和你真正要服务的用户一样，而不是拿最容易收到的东西凑。

%%% insight
这是今天这个习惯的成熟版本，也是把一个新手和一个你愿意托付真系统的人分开的那句话：「我的 test 分数，对我采样的那个世界是诚实的。」不是对每一个世界。把这句话说出口 —— 而不是只报一个数字 —— 这才是这个行业里真正的自信听起来的样子。
%%%

#### 再快快说四个诚实的边界
这些都不是停止分数据的理由。它们是小字条款，而知道它们是一件能拿出来的本事。

%%% cards
🎲 | 一个数字不是保证 | 一个留出堆上的分数是一个带采样噪声的点估计。换一个同样合理的 split 会给出稍微不同的数字，所以把它当成一个带晃动的估计，不是一个承诺。
🪞 | 分数据不会创造信息 | 如果你的 label 是错的，或者你的采集有偏，那么三堆全都错，而且偏向*同一个*方向。split 会开开心心地为一个坏掉的数据集背书。
🕵️ | split 检测不出自己的 leakage | 一个漏了的 split 看起来一点都不可疑 —— 它看起来*非常出色*。leakage 是靠流程和检查抓出来的（那道重叠检查、pipeline 的顺序），永远不是靠分数。
🍰 | 留出去的数据，是你学不到的数据 | 你留出去的每一个样本，都是模型永远学不到的一个。这就是 cross-validation 存在去缓解的那份小数据的紧张。
%%%

最后还有两个值得直说的诚实边界。一个很强的离线 test 分数**不**保证产品会变好，也不保证模型对每一群用户都一样好用 —— 那是后面线上实验和分群打分要做的事。还有，split 告诉你模型**错了**，从不告诉你**为什么**错、**在哪些样本上**错；要弄清这些，得一条一条看你的错误，再把数据的每一片单独打分。

胜利一圈：你现在既知道你的仪器量的是什么，*也*知道它在哪里就不管用了。该把今天的一切收到一页上了。
~~~

@@@ concept id=c10 zh_tag="回顾" tag="Recap" zh_title="一页装下今天全部" title="The whole day on one page" zh_gotit="懂了这个回顾" gotit="Got the recap"
Here is a fresh way to hold the whole day in your head: a **driving test**. You *practise* for weeks in your own neighbourhood (the training set). Now and then a friend rides along and quizzes you on the tricky moves so you can fix them before the big day (the validation set). Finally an examiner you have never met takes you on a route you have never driven, once, for your real licence (the test set). If you had "practised" by memorizing that examiner's exact route, you would pass and still be a danger on every other road — that is overfitting, and the difference between your smooth practice runs and your shaky new-route runs is how anyone would *see* it. **Where the driving-test picture breaks down:** an examiner can change the route on the spot, but your test set is frozen the moment you seal it — so you have to design it fairly up front, because you only get to use it once.
~~~zh
再给你一个把今天整个装进脑子的新办法：**考驾照**。你在自己家附近*练*好几个星期（training set）。隔三差五有个朋友坐上来，专挑难的动作考你，好让你在大日子之前改掉（validation set）。最后一个你从没见过的考官带你走一条你从没开过的路线，一次，为了你真正的驾照（test set）。如果你的「练习」是把那位考官的路线原样背下来呢？你会通过，然后在别的每一条路上都是个危险。那就是 overfitting。你顺畅的练习和你手抖的新路线之间那个差别，就是任何人*看见*它的方式。**考驾照这张图在哪里不成立：** 考官可以当场换路线。可你的 test set 在你封上它的那一刻就冻住了。所以你得提前把它设计得公平，因为你只能用它一次。
~~~

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A driving-test analogy in three steps. First, practice in your own neighbourhood, labelled training. Second, a friend quizzes you on tricky moves to fix, labelled validation. Third, an examiner tests you once on a new route for your licence, labelled test. A caption ties it to learning versus memorizing the route."><g font-family="monospace" font-size="9.5"><text class="lang-en" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The driving test — practise, quiz, then the real examiner</text><text class="lang-zh" x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">考驾照 —— 练、被考、然后是真考官</text><rect x="20" y="34" width="150" height="96" rx="8" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="95" y="56" text-anchor="middle" fill="#1F6280" font-size="10">🚗 practise</text><text class="lang-zh" x="95" y="56" text-anchor="middle" fill="#1F6280" font-size="10">🚗 练车</text><text class="lang-en" x="95" y="82" text-anchor="middle" fill="#6B645E" font-size="8">own neighbourhood</text><text class="lang-zh" x="95" y="82" text-anchor="middle" fill="#6B645E" font-size="8">自己家附近</text><text class="lang-en" x="95" y="102" text-anchor="middle" fill="#1F6280" font-size="8.5">= TRAINING</text><text class="lang-zh" x="95" y="102" text-anchor="middle" fill="#1F6280" font-size="8.5">= 训练 TRAINING</text><text class="lang-en" x="95" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">learn here</text><text class="lang-zh" x="95" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">在这里学</text><rect x="185" y="34" width="150" height="96" rx="8" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="260" y="56" text-anchor="middle" fill="#9A7208" font-size="10">🧑‍🏫 friend quizzes</text><text class="lang-zh" x="260" y="56" text-anchor="middle" fill="#9A7208" font-size="10">🧑‍🏫 朋友考你</text><text class="lang-en" x="260" y="82" text-anchor="middle" fill="#6B645E" font-size="8">fix the tricky moves</text><text class="lang-zh" x="260" y="82" text-anchor="middle" fill="#6B645E" font-size="8">改掉难的动作</text><text class="lang-en" x="260" y="102" text-anchor="middle" fill="#9A7208" font-size="8.5">= VALIDATION</text><text class="lang-zh" x="260" y="102" text-anchor="middle" fill="#9A7208" font-size="8.5">= 验证 VALIDATION</text><text class="lang-en" x="260" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">check + adjust</text><text class="lang-zh" x="260" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">查 + 调</text><rect x="350" y="34" width="150" height="96" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="425" y="56" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 examiner, new route</text><text class="lang-zh" x="425" y="56" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 考官，新路线</text><text class="lang-en" x="425" y="82" text-anchor="middle" fill="#6B645E" font-size="8">once, for your licence</text><text class="lang-zh" x="425" y="82" text-anchor="middle" fill="#6B645E" font-size="8">一次，为了你的驾照</text><text class="lang-en" x="425" y="102" text-anchor="middle" fill="#1a5c38" font-size="8.5">= TEST</text><text class="lang-zh" x="425" y="102" text-anchor="middle" fill="#1a5c38" font-size="8.5">= 测试 TEST</text><text class="lang-en" x="425" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">honest final grade</text><text class="lang-zh" x="425" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">诚实的最终成绩</text><path d="M170 82 L 185 82" stroke="#6B645E" stroke-width="1.4"/><polygon points="185,82 177,78 177,86" fill="#6B645E"/><path d="M335 82 L 350 82" stroke="#6B645E" stroke-width="1.4"/><polygon points="350,82 342,78 342,86" fill="#6B645E"/><text class="lang-en" x="260" y="152" text-anchor="middle" fill="#1a5c38" font-size="9">Memorizing the route passes the exam and fails every other road = overfitting.</text><text class="lang-zh" x="260" y="152" text-anchor="middle" fill="#1a5c38" font-size="9">把路线背下来能过考试，在别的每条路上都不行 = overfitting。</text><text class="lang-en" x="260" y="170" text-anchor="middle" fill="#6B645E" font-size="8.5">The exam-vs-answer-key idea, start to finish.</text><text class="lang-zh" x="260" y="170" text-anchor="middle" fill="#6B645E" font-size="8.5">考卷和答案本这个想法，从头到尾。</text></g></svg>
%%%

#### The day, rebuilt one panel at a time
Read the picture below left to right: first only the study pile, then the check pile clicks into place, then the sealed grade pile, and finally the *gap* you watch between study and check. One new idea per panel — the same order you learned them in.
~~~zh
#### 今天这件事，一格一格重搭一遍
下面这张图从左往右读。先只有学习那一堆，然后检查那一堆卡进来。接着是封起来打分的那一堆。最后是你在学习和检查之间盯着的那道*缝*。一格一个新想法 —— 和你学它们的顺序一样。
~~~

%%% svg
<svg viewBox="0 0 520 208" role="img" aria-label="A four-panel progressive assembly of the day's idea. Panel one: only the training pile, where the model learns. Panel two adds the validation pile beside it, used to check progress during study. Panel three adds the sealed test pile, opened once for the honest grade. Panel four keeps all three piles and draws a widening gap between the training and validation curves, labelled overfitting is the gap you watch. Each panel adds one piece on top of the previous."><g font-family="monospace" font-size="8"><text class="lang-en" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Rebuild the day: one piece per panel →</text><text class="lang-zh" x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">重搭今天：一格加一块 →</text><g><rect x="10" y="26" width="118" height="168" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text class="lang-en" x="69" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">① just study</text><text class="lang-zh" x="69" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">① 只有学习</text><rect x="26" y="58" width="86" height="30" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="69" y="77" text-anchor="middle" fill="#1F6280">📚 train</text><text class="lang-zh" x="69" y="77" text-anchor="middle" fill="#1F6280">📚 train</text><text class="lang-en" x="69" y="112" text-anchor="middle" fill="#6B645E">model LEARNS</text><text class="lang-zh" x="69" y="112" text-anchor="middle" fill="#6B645E">模型只在这里</text><text class="lang-en" x="69" y="126" text-anchor="middle" fill="#6B645E">here only</text><text class="lang-zh" x="69" y="126" text-anchor="middle" fill="#6B645E">学习</text></g><g><rect x="134" y="26" width="118" height="168" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text class="lang-en" x="193" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">② + check</text><text class="lang-zh" x="193" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">② + 检查</text><rect x="150" y="58" width="86" height="26" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="193" y="75" text-anchor="middle" fill="#1F6280">📚 train</text><text class="lang-zh" x="193" y="75" text-anchor="middle" fill="#1F6280">📚 train</text><rect x="150" y="92" width="86" height="26" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="193" y="109" text-anchor="middle" fill="#9A7208">📝 validation</text><text class="lang-zh" x="193" y="109" text-anchor="middle" fill="#9A7208">📝 validation</text><text class="lang-en" x="193" y="140" text-anchor="middle" fill="#9A7208">check progress,</text><text class="lang-zh" x="193" y="140" text-anchor="middle" fill="#9A7208">看进展、</text><text class="lang-en" x="193" y="152" text-anchor="middle" fill="#9A7208">tune settings</text><text class="lang-zh" x="193" y="152" text-anchor="middle" fill="#9A7208">调设置</text></g><g><rect x="258" y="26" width="118" height="168" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text class="lang-en" x="317" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">③ + sealed grade</text><text class="lang-zh" x="317" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">③ + 封起来的成绩</text><rect x="274" y="54" width="86" height="24" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text class="lang-en" x="317" y="70" text-anchor="middle" fill="#1F6280">📚 train</text><text class="lang-zh" x="317" y="70" text-anchor="middle" fill="#1F6280">📚 train</text><rect x="274" y="84" width="86" height="24" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text class="lang-en" x="317" y="100" text-anchor="middle" fill="#9A7208">📝 validation</text><text class="lang-zh" x="317" y="100" text-anchor="middle" fill="#9A7208">📝 validation</text><rect x="274" y="114" width="86" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text class="lang-en" x="317" y="130" text-anchor="middle" fill="#1a5c38">🔒 test</text><text class="lang-zh" x="317" y="130" text-anchor="middle" fill="#1a5c38">🔒 test</text><text class="lang-en" x="317" y="156" text-anchor="middle" fill="#1a5c38">opened ONCE</text><text class="lang-zh" x="317" y="156" text-anchor="middle" fill="#1a5c38">只拆一次</text><text class="lang-en" x="317" y="168" text-anchor="middle" fill="#6B645E">honest grade</text><text class="lang-zh" x="317" y="168" text-anchor="middle" fill="#6B645E">诚实的成绩</text></g><g><rect x="382" y="26" width="128" height="168" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text class="lang-en" x="446" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">④ + watch the gap</text><text class="lang-zh" x="446" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">④ + 盯那道缝</text><path d="M398 150 C 420 120, 440 96, 500 62" fill="none" stroke="#2A7B9B" stroke-width="2"/><path d="M398 150 C 420 128, 440 128, 500 118" fill="none" stroke="#C99A12" stroke-width="2"/><line x1="500" y1="62" x2="500" y2="118" stroke="#C93B3B" stroke-width="1.2" stroke-dasharray="3,3"/><text class="lang-en" x="470" y="76" fill="#1F6280" font-size="7">train ↓</text><text class="lang-zh" x="470" y="76" fill="#1F6280" font-size="7">train ↓</text><text class="lang-en" x="466" y="132" fill="#9A7208" font-size="7">val turns up</text><text class="lang-zh" x="466" y="132" fill="#9A7208" font-size="7">val 抬头</text><text class="lang-en" x="446" y="182" text-anchor="middle" fill="#C93B3B" font-size="7.5">the widening gap = overfitting</text><text class="lang-zh" x="446" y="182" text-anchor="middle" fill="#C93B3B" font-size="7.5">越来越宽的缝 = overfitting</text></g></g></svg>
%%%

That is the whole arc in four clicks — study, check, grade, and watch the gap. The same story in words:

- **Why split.** A model must be judged on data it has NOT seen, or the grade is a lie. We want generalization, not memorizing.
- **Three piles.** *Train* — the only pile the model learns on. *Validation* — held out, checked during development to track progress and pick settings. *Test* — sealed, opened ONCE for an honest final score.
- **Why three, not two.** Tuning against the test pile contaminates it, so the validation pile exists to protect the test pile.
- **Overfitting.** *Cause:* the model memorizes the training data and its noise — the wiggly curve you dialled up yourself. *Symptom:* high train score, low validation score. Remedies: less capacity, stop earlier, more data.
- **See it with the gap.** Both curves falling together and close is healthy; validation turning upward while training falls is overfitting. The **lowest validation point is the best model** — stopping there (with a patience window) is early stopping.
- **Peek & shuffle.** *Peeking* → keep the test untouched, use it only once. *Sorted slice* → shuffle before splitting (stratify if a class is rare).
- **The hidden leak.** *Cause:* a shared calculation or duplicate rows. *Fix:* split first, fit transforms on train only, de-duplicate — and split by group or by time when rows are not independent.
- **Unlucky slice.** A small pile is high-variance; **cross-validation (k-fold)** rotates the slice and averages. Ratios ~70/15/15 or ~80/10/10, but no ratio is sacred.
- **Honest limits.** A split *diagnoses* generalization, it does not create it; its verdict holds only for data from the same distribution; and it can never detect its own leakage.

The quick diagnosis table — spot the trouble, know the fix:

%%% table
:: What you see :: What it means :: The fix
train loss low, val loss rising :: overfitting (memorizing) :: stop at the lowest val point (early stopping); less capacity; more data
you tuned using the test score :: test contaminated (peeking) :: keep test sealed; use it once; report, never decide
duplicate rows in two piles :: leakage across splits :: split FIRST, de-dup, fit preprocessing on train only
one slice easy, another hard :: unlucky, high-variance split :: cross-validation (k-fold), average the scores
great test score, bad in the real world :: distribution shift (a different world) :: match val/test to real users; monitor after launch
%%%

And the cheat-sheet — every word from today in one place:

%%% jargon
generalization | doing well on fresh, unseen examples — the real goal, not memorizing the ones you studied
training set | the pile the model actually learns on (it fits its weights to these)
validation set | a held-out pile you check progress on and pick settings with, during development
test set | a sealed pile scored ONCE at the very end for an honest final score
epoch | one full pass over the training split
overfitting | memorizing the training examples and their noise instead of the pattern — high train score, low validation score
underfitting | too little capacity or too few steps — both scores stay bad
capacity | how much a model can bend or store (more neurons, more layers, a bendier curve)
train-validation gap | the difference between the two scores; a widening gap where validation rises is overfitting you can see
early stopping | keep the checkpoint at the lowest validation loss and stop, with a patience window for bumpy curves
peeking | making any decision by looking at the test score — it quietly turns test into training data
leakage | information from an eval pile reaching training: duplicate rows, a transform fitted before splitting, a group or a date spanning piles
stratified splitting | cutting so each pile keeps the same class mix — the fix for a rare class
cross-validation (k-fold) | rotate which slice is validation across k rounds and average the scores, to beat an unlucky split
distribution shift | when real data drifts away from your splits, so a good test score stops predicting performance
%%%
~~~zh
整条弧线就是这四下 —— 学、查、打分、盯那道缝。同一个故事换成文字：

- **为什么要分。** 一个模型必须用它没见过的数据来评判，不然这个成绩就是谎话。我们要的是 generalization，不是背答案。
- **三堆。** *Train* —— 唯一那堆模型在上面学的。*Validation* —— 留出来的，在开发过程中查进展、挑设置。*Test* —— 封着，只拆开一次，拿一个诚实的最终分数。
- **为什么是三堆不是两堆。** 对着 test 那一堆调会污染它，所以 validation 那一堆存在的意义就是保护 test 那一堆。
- **Overfitting。** *原因：* 模型把训练数据和它的噪声背下来 —— 就是你自己拖出来的那条扭来扭去的曲线。*症状：* train 分数高，validation 分数低。解药：更少 capacity、更早停、更多数据。
- **用那道缝看见它。** 两条曲线一起往下掉、贴得很近，是健康的；validation 抬头而 training 还在掉，是 overfitting。**validation 最低的那一点就是最好的模型** —— 在那里停下（配一个耐心窗口）就是 early stopping。
- **偷看和洗牌。** *偷看* → 别碰 test，只用它一次。*排好序的那一刀* → 切之前先洗牌（某一类少见的时候用 stratified）。
- **藏起来的泄漏。** *原因：* 一个共用的计算，或者重复的行。*办法：* 先 split、只在 train 上拟合变换、去重 —— 而当行与行不独立时，按分组或按时间切。
- **运气差的那一块。** 小的一堆方差大；**cross-validation（k-fold）** 轮换那一块并取平均。比例大约 70/15/15 或者 80/10/10，但没有哪个比例是神圣的。
- **诚实的边界。** split *诊断* generalization，它不创造它；它的判决只对同一个分布里的数据成立；而且它永远检测不出自己的 leakage。

快速诊断表 —— 看出毛病，知道办法：

%%% table
:: 你看到什么 :: 它意味着什么 :: 办法
train loss 低，val loss 在升 :: overfitting（在背答案） :: 在 val 最低的那一点停（early stopping）；更少 capacity；更多数据
你用 test 分数来调了 :: test 被污染了（偷看） :: 把 test 封好；只用一次；用来报告，绝不用来决定
两堆里有重复的行 :: 跨 split 的 leakage :: 先 split、去重、预处理只在 train 上拟合
一块简单，另一块难 :: 运气差、方差大的 split :: cross-validation（k-fold），把分数平均
test 分数很好，真实世界里很差 :: distribution shift（换了一个世界） :: 让 val/test 对上真实用户；上线之后持续监控
%%%

还有那张小抄 —— 今天每一个词都在这里：

%%% jargon
generalization | 在没见过的新样本上做得好 —— 这才是真正的目标，不是把学过的背下来
training set | 模型真正在上面学的那一堆（它照着这些调自己的 weight）
validation set | 留出来的一堆，你在开发过程中用它看进展、挑设置
test set | 封起来的一堆，只在最后打分一次，拿一个诚实的最终分数
epoch | 把训练那一份完整走一遍
overfitting | 把训练样本和它们的噪声背下来，而不是学那条规律 —— train 分数高，validation 分数低
underfitting | capacity 太少或者步数太少 —— 两个分数都一直很差
capacity | 一个模型能弯多厉害、能存多少（更多 neuron、更多 layer、更能弯的曲线）
train–val 的那道缝 | 两个分数之间的差；缝一直变宽、validation 在升，就是你能看见的 overfitting
early stopping | 留住 validation loss 最低的那个 checkpoint 然后停，曲线坑洼时配一个耐心窗口
偷看（peeking） | 任何靠看 test 分数做出的决定 —— 它悄悄把 test 变成了训练数据
leakage | 评估那几堆的信息进到了训练里：重复的行、split 之前就拟合好的变换、跨着几堆的一个分组或一个日期
stratified splitting | 切的时候让每一堆保持一样的类别比例 —— 专治少见的类别
cross-validation（k-fold） | 分 k 轮轮换哪一块当 validation，再把分数平均，用来打败运气差的 split
distribution shift | 真实数据漂离了你那几堆，于是一个好的 test 分数不再能预测表现
%%%
~~~

@@@ quiz id=quiz zh_tag="小测" tag="Quiz" zh_title="四个快速检查" title="Four quick checks" zh_gotit="先全部作答" gotit="answer all first"
%%% quiz
q: Why do we grade a model on a held-out pile instead of on the data it trained on? | a:2 | held-out data trains faster | to save memory | a score on studied data cannot tell learning from memorizing | held-out data is more accurate | fb: The whole point: a model can ace data it memorized. Only unseen data shows whether it truly generalizes.
q: Training loss keeps dropping but validation loss reaches its lowest point and starts rising. What is happening, and where is the best model? | a:1 | the learning rate is too small | overfitting — the best model is at the lowest validation point | the test set leaked | training finished successfully | fb: A widening train-validation gap with validation rising is overfitting. The best model on new data is exactly at the lowest validation point — that is where early stopping stops.
q: Which pile do you open exactly ONCE, at the very end, and never tune against? | a:3 | the training set | the validation set | whichever is largest | the test set | fb: The test set is the sealed final exam — touch it once for an honest grade. Tuning against it (peeking) contaminates it.
q: You scale your features using the mean of the WHOLE dataset, then split into train/val/test. What went wrong, and what is the fix? | a:0 | leakage — split first, then fit the scaler on train only and apply it to val/test | nothing, scaling is always safe | the learning rate is now too big | you must scale twice for balance | fb: Fitting the scaler on everything lets eval data influence training — leakage. Split first, fit on train only, then transform val and test with those training numbers.
%%%
~~~zh
%%% quiz
q: 我们为什么用留出来的一堆给模型打分，而不用它训练过的数据？ | a:2 | 留出来的数据训练更快 | 为了省内存 | 学过的数据上的分数分不出「学会」和「背下来」 | 留出来的数据更准 | fb: 关键就在这里：一个模型能在它背下来的数据上考满分。只有没见过的数据才显出它是不是真的 generalization。
q: Training loss 一直在掉，可 validation loss 到了最低点就开始往上升。发生了什么，最好的模型在哪里？ | a:1 | learning rate 太小了 | overfitting —— 最好的模型在 validation 最低的那一点 | test set 漏了 | 训练顺利结束了 | fb: train–val 的缝一直变宽、validation 在升，就是 overfitting。新数据上最好的模型正好在 validation 最低的那一点 —— early stopping 就停在那里。
q: 哪一堆你只在最后拆开一次，而且永远不对着它调参？ | a:3 | training set | validation set | 哪一堆最大就是哪一堆 | test set | fb: test set 是那张封好的期末考卷 —— 只碰一次，拿一个诚实的成绩。对着它调（偷看）会污染它。
q: 你用**整个**数据集的均值去缩放特征，然后才切成 train/val/test。哪里出错了，办法是什么？ | a:0 | leakage —— 先 split，然后只在 train 上拟合 scaler，再把它应用到 val/test | 没问题，缩放永远是安全的 | learning rate 现在太大了 | 你必须缩放两次才平衡 | fb: 在全部数据上拟合 scaler，会让评估数据影响训练 —— 这就是 leakage。先 split，只在 train 上拟合，再用那些训练的数字去变换 val 和 test。
%%%
~~~

@@@ produce id=produce zh_tag="动手做" tag="Produce" zh_title="亲眼抓住一个模型在 overfitting" title="Catch a model overfitting with your own eyes" zh_gotit="做完了" gotit="Done"
Time to *feel* the whole thing in one run: train a model, watch both losses, and catch the exact moment memorizing beats learning. This is where the *exam-vs-answer-key* idea turns into two curves and a gap on your own screen.

The setup is the simplest honest experiment there is: make a small dataset with a real pattern plus a little random noise, **shuffle it, then split it** into a training pile and a validation pile (say 80/20). Train a flexible little model for many epochs. After each epoch, record two numbers — the loss on the training pile and the loss on the validation pile.

**Predict first, before you run it:** for the *early* epochs, will the two losses be close or far apart? And late on, what will the validation loss do while the training loss keeps dropping? Write your two guesses down, then check them.

**What you should see** when you plot both losses against epoch:
- **Early epochs** → both losses drop **together and stay close** — real pattern that works on both piles (healthy).
- **The lowest validation point** → validation reaches its **minimum**, then turns around. This epoch is the *best* model, and exactly where early stopping would stop.
- **Late epochs** → training loss keeps falling toward zero while **validation loss rises** — the gap widens. That gap *is* overfitting, now visible on your screen.
- Note the epoch of lowest validation loss, and how far past it you would have trained if you had only watched the training curve.

Write it in `experiment.py` and run it. Then poke it, because poking is where the intuition sticks: skip the shuffle on sorted data and watch the split break; shrink the dataset and watch the validation curve get noisy (the unlucky-slice effect); make the model smaller and watch the gap shrink, because a simpler model has less room to memorize.

**Option A — write it yourself.** In `experiment.py`, build a noisy dataset, shuffle and split 80/20, train a flexible model for many epochs, and after each epoch append `train_loss` and `val_loss` to two lists. Print (or plot) both, and mark the epoch where `val_loss` is lowest.

**Option B — hand it to the lab.** Paste this and let it build `experiment.py` for you:

%%% prompt id=pp label="ask Claude to build it"
Help me build my artifact.
Create experiment.py for Day 9 (train/val/test & overfitting). Make a small 1-D dataset
from a smooth true function (e.g. y = sin(x)) plus random noise, about 40 points.
SHUFFLE it, then split 80/20 into train and validation. Fit a deliberately flexible model
(e.g. a high-degree polynomial, or a tiny MLP trained for many epochs) and, after each epoch
(or each fit step), record train_loss (MSE on the train split) and val_loss (MSE on the val split).
Print both lists and report the epoch with the LOWEST val_loss (the early-stopping point).
Expected: train_loss keeps dropping toward ~0, while val_loss drops at first, reaches a
minimum, then RISES as the model overfits — so the train-validation gap widens after that minimum.
Then add three short bonus runs: (1) a SMALLER/simpler model, showing a smaller gap;
(2) no shuffle on sorted data, showing an unrepresentative split;
(3) duplicate some rows before splitting, showing the score get fake-good.
Make the overfitting gap clearly visible in the printed output.
%%%
~~~zh
该在一次运行里*感受*整件事了：训练一个模型、看着两条 loss、抓住背答案压过学习的那一刻。就在这里，*考卷和答案本*这个想法变成你自己屏幕上的两条曲线和一道缝。

准备工作是最简单的一个诚实实验。造一个小数据集，里面有一条真的规律加上一点随机噪声。然后**先洗牌，再把它切开**成一个训练堆和一个 validation 堆，比如 80/20。训练一个灵活的小模型，跑很多个 epoch。每个 epoch 之后记两个数字 —— 训练堆上的 loss 和 validation 堆上的 loss。

**在你运行之前先预测：** 在*早期*的那些 epoch 里，两条 loss 是贴得很近还是离得很远？到了后期，training loss 一直往下掉的时候，validation loss 会做什么？把你的两个猜测写下来，然后去检查。

**你应该看到什么**，当你把两条 loss 对着 epoch 画出来：
- **早期 epoch** → 两条 loss **一起往下掉、一直贴得很近** —— 在两堆上都管用的真规律（健康）。
- **validation 最低的那一点** → validation 到达它的**最小值**，然后转向。这个 epoch 就是*最好*的模型，也正是 early stopping 会停下的地方。
- **后期 epoch** → training loss 继续往零掉，而 **validation loss 在升** —— 缝变宽了。那道缝*就是* overfitting，现在在你屏幕上看得见了。
- 记下 validation loss 最低的那个 epoch，还有：如果你只看训练那条曲线，你会多练过头多少。

把它写进 `experiment.py` 然后跑起来。接着去戳它，因为直觉是靠戳才粘住的：在排好序的数据上跳过洗牌，看这个 split 怎么坏掉；把数据集缩小，看 validation 曲线变得多噪（那就是运气差的一块的效果）；把模型做小，看那道缝缩小，因为更简单的模型没那么多空间去背。

**选项 A —— 自己写。** 在 `experiment.py` 里造一个带噪声的数据集，洗牌后切 80/20，训练一个灵活的模型跑很多个 epoch，每个 epoch 之后把 `train_loss` 和 `val_loss` 各追加到一个列表里。把两个都打印（或者画）出来，并标出 `val_loss` 最低的那个 epoch。

**选项 B —— 交给 lab。** 把下面这段贴过去，让它帮你把 `experiment.py` 搭出来：

%%% prompt id=ppzh label="让 Claude 帮你搭"
帮我做我的产出。

新建 Module 2 第 9 天（train/val/test 与 overfitting）的 experiment.py。造一个小的一维数据集：
一条平滑的真函数（比如 y = sin(x)）加上随机噪声，大约 40 个点。
先**洗牌**，再按 80/20 切成 train 和 validation。拟合一个故意做得很灵活的模型
（比如一个高次多项式，或者一个训练很多个 epoch 的小 MLP）。每个 epoch（或者每个拟合步）之后，
记下 train_loss（train 那一份上的 MSE）和 val_loss（val 那一份上的 MSE）。
把两个列表都打印出来，并报出 val_loss 最低的那个 epoch，也就是 early stopping 的那一点。
预期：train_loss 一路往 0 掉，而 val_loss 先降、到达一个最小值、然后随着模型 overfitting 而**上升** ——
所以过了那个最小值之后，train–val 的缝会变宽。
再加三段短的额外运行：(1) 一个更小/更简单的模型，显示缝更小；
(2) 在排好序的数据上不洗牌，显示一个不具代表性的 split；
(3) 切之前先复制一些行，显示分数变成假的好。
让 overfitting 的那道缝在打印出来的输出里清清楚楚看得见。
%%%
~~~

@@@ fin
