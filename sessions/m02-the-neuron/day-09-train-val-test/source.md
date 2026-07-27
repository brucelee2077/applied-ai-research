---
quest_id: wf3-d06-split
mode: concept
donor: v9-base.donor
page_title: "Module 2 · Day 9 — Train / Val / Test & Overfitting"
module_label: "Module 2 · Train · Day 9"
title: "Train / Val / Test & Overfitting"
subtitle: "Did the Model Learn, or Just Memorize?"
brand_sub: "Foundations · M2 Day 9"
spine: "exam vs answer-key"
nav_prev_href: "../day-08-learning-rate/lesson.html"
nav_prev_label: "Learning-Rate Intuition"
nav_next_href: "../review.html"
nav_next_label: "Review Gate"
fin_title: "Module 2 · Day 9 complete! 🏆"
fin_body: "Nice work — you can now tell <b>learning</b> from <b>memorizing</b>: train on one set, watch a held-out validation set to catch overfitting (the gap between train and val loss), and keep the test set sealed like a locked answer key. Early stopping and regularization keep the model honest.<br>That completes the days of Module 2 — up next is the <b>Review Gate</b>."
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

@@@ concept id=c1 tag="Why split?" title="Why we never grade a model on what it studied" gotit="Got why we split"
Let us start with the one question this whole day answers. Imagine a friend who "studied" for a history exam by memorizing the answers to *last year's exact paper*. On that old paper they score 100%. Hand them *this year's* paper — fresh questions, same topic — and they fall apart.

Did they learn history? No. They learned *those answers*. A neural network can do exactly the same thing: score perfectly on the examples it trained on, and be useless on anything new. The word for doing well on *fresh, unseen* examples is [[generalization||doing well on fresh examples the model never saw during training — the real goal, as opposed to just memorizing the ones it studied]], and how well a model can generalize is the only thing we actually care about.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two students side by side. On the left, a student holds last year's exam paper and scores 100 percent by memorizing it, but scores badly on a fresh new paper. On the right, a student who learned the topic scores well on both the old and the new paper. A caption reads memorizing the paper is not learning the subject."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Memorizing the paper is NOT learning the subject</text><rect x="20" y="34" width="230" height="158" rx="8" fill="#FDECEC" stroke="#C93B3B"/><text x="135" y="52" text-anchor="middle" fill="#C93B3B" font-size="10">🙈 memorized last year's paper</text><rect x="46" y="66" width="70" height="90" rx="4" fill="#fff" stroke="#C93B3B"/><text x="81" y="112" text-anchor="middle" fill="#C93B3B" font-size="8">old paper</text><text x="81" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">100% ✓</text><rect x="150" y="66" width="70" height="90" rx="4" fill="#fff" stroke="#C93B3B"/><text x="185" y="112" text-anchor="middle" fill="#C93B3B" font-size="8">new paper</text><text x="185" y="150" text-anchor="middle" fill="#C93B3B" font-size="9">42% ✗</text><rect x="270" y="34" width="230" height="158" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="385" y="52" text-anchor="middle" fill="#1a5c38" font-size="10">🧠 learned the topic</text><rect x="296" y="66" width="70" height="90" rx="4" fill="#fff" stroke="#2D8B55"/><text x="331" y="112" text-anchor="middle" fill="#1a5c38" font-size="8">old paper</text><text x="331" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">96% ✓</text><rect x="400" y="66" width="70" height="90" rx="4" fill="#fff" stroke="#2D8B55"/><text x="435" y="112" text-anchor="middle" fill="#1a5c38" font-size="8">new paper</text><text x="435" y="150" text-anchor="middle" fill="#1a5c38" font-size="9">93% ✓</text></g></svg>
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

@@@ concept id=c2 tag="Three piles" title="The three splits — and what each one is for" gotit="Got the three piles"
So we hide some examples. *How many* hidden piles, and what is each one's job? The clean answer is **three**, and your own school year already taught you all three.

Picture three stacks of question cards on a desk. The **practice sheet** — practice questions with the answers on the back, the ones you study from. The **weekly quiz** — small, taken during the week, to see whether your studying is working and to decide what to change. The **final exam** — sealed in a drawer, opened once at the very end for your real grade.

Those three stacks have names: the [[training set||the pile the model actually learns on — it adjusts, or fits, its weights to these examples]], the [[validation set||a held-out pile checked DURING development to track progress and pick settings, without ever touching the final exam]], and the [[test set||a pile sealed away and scored only ONCE at the very end, to get one honest final number]]. **Three splits, three jobs.**

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Three stacks of cards side by side. The first, labelled training set, is the practice sheet the model studies with answers shown. The second, labelled validation set, is a weekly quiz used to check progress and pick settings during the week. The third, labelled test set, is a sealed final exam with a lock, opened only once at the end. Arrows show train comes first, then validation during study, then test at the end."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Three piles of cards · three different jobs</text><rect x="18" y="36" width="150" height="150" rx="8" fill="#EAF3F7" stroke="#2A7B9B"/><text x="93" y="56" text-anchor="middle" fill="#1F6280" font-size="10">📚 TRAINING</text><rect x="44" y="70" width="98" height="20" rx="3" fill="#fff" stroke="#2A7B9B"/><rect x="50" y="82" width="98" height="20" rx="3" fill="#fff" stroke="#2A7B9B"/><rect x="56" y="94" width="98" height="20" rx="3" fill="#fff" stroke="#2A7B9B"/><text x="93" y="140" text-anchor="middle" fill="#1F6280" font-size="8">practice sheet</text><text x="93" y="156" text-anchor="middle" fill="#6B645E" font-size="8">model LEARNS here</text><text x="93" y="172" text-anchor="middle" fill="#6B645E" font-size="8">(answers shown)</text><rect x="186" y="36" width="148" height="150" rx="8" fill="#FDF3D6" stroke="#C99A12"/><text x="260" y="56" text-anchor="middle" fill="#9A7208" font-size="10">📝 VALIDATION</text><rect x="216" y="76" width="88" height="22" rx="3" fill="#fff" stroke="#C99A12"/><text x="260" y="91" text-anchor="middle" fill="#9A7208" font-size="8">weekly quiz</text><text x="260" y="128" text-anchor="middle" fill="#9A7208" font-size="8">CHECK progress,</text><text x="260" y="144" text-anchor="middle" fill="#9A7208" font-size="8">PICK settings</text><text x="260" y="164" text-anchor="middle" fill="#6B645E" font-size="8">during study</text><rect x="352" y="36" width="150" height="150" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="427" y="56" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 TEST</text><rect x="388" y="74" width="78" height="52" rx="3" fill="#fff" stroke="#2D8B55"/><text x="427" y="96" text-anchor="middle" font-size="16">🔒</text><text x="427" y="116" text-anchor="middle" fill="#1a5c38" font-size="8">sealed exam</text><text x="427" y="146" text-anchor="middle" fill="#1a5c38" font-size="8">opened ONCE</text><text x="427" y="162" text-anchor="middle" fill="#6B645E" font-size="8">at the very end</text><path d="M168 200 L 186 200" stroke="#6B645E" stroke-width="1.4"/><polygon points="186,200 178,196 178,204" fill="#6B645E"/><path d="M334 200 L 352 200" stroke="#6B645E" stroke-width="1.4"/><polygon points="352,200 344,196 344,204" fill="#6B645E"/><text x="260" y="204" text-anchor="middle" fill="#6B645E" font-size="8">learn → check → final grade</text></g></svg>
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

@@@ concept id=c3 tag="Why three, not two" title="Where the middle pile came from" gotit="Got the history"
Fair question: why not just *two* piles — one to study, one to grade? That is exactly how people did it at first, and it looks perfectly reasonable. This is the **two-way split**: train and test only, no middle pile. (The even older habit was to report the score on the training data itself — grading the student on the exact problems they studied — which is the fake grade you just retired.)

Picture a student with only a practice sheet and one final exam, no weekly quiz. They study, sit the exam, score 70%. They change how they study, sit *the same exam again*, 80%. Change again, sit it again, 88%. Somewhere in there the grade quietly stopped being a grade.

%%% svg
<svg viewBox="0 0 520 214" role="img" aria-label="Top row shows the old two-way setup: a student repeatedly takes the same final exam and tweaks their studying based on the score, so the exam becomes contaminated. Bottom row shows the three-way fix: tuning now happens against a separate weekly quiz (validation), and the final exam stays sealed and honest."><g font-family="monospace" font-size="9.5"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Tuning against the exam quietly contaminates it</text><text x="16" y="44" fill="#C93B3B" font-size="9.5">OLD · two piles</text><rect x="120" y="32" width="86" height="34" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text x="163" y="53" text-anchor="middle" fill="#1F6280" font-size="8">train</text><rect x="230" y="32" width="120" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="290" y="47" text-anchor="middle" fill="#C93B3B" font-size="8">final exam</text><text x="290" y="59" text-anchor="middle" fill="#C93B3B" font-size="7.5">also used to TUNE</text><path d="M350 40 C 400 40, 400 58, 350 58" fill="none" stroke="#C93B3B" stroke-width="1.4"/><polygon points="352,58 360,54 360,62" fill="#C93B3B"/><text x="410" y="52" fill="#C93B3B" font-size="8">tweak, re-take,</text><text x="410" y="64" fill="#C93B3B" font-size="8">repeat 💥 contaminated</text><line x1="16" y1="88" x2="504" y2="88" stroke="#E5DFD6"/><text x="16" y="120" fill="#1a5c38" font-size="9.5">NEW · three piles</text><rect x="120" y="108" width="80" height="34" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text x="160" y="129" text-anchor="middle" fill="#1F6280" font-size="8">train</text><rect x="212" y="108" width="96" height="34" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text x="260" y="123" text-anchor="middle" fill="#9A7208" font-size="8">validation</text><text x="260" y="135" text-anchor="middle" fill="#9A7208" font-size="7.5">TUNE here</text><rect x="320" y="108" width="110" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="375" y="123" text-anchor="middle" fill="#1a5c38" font-size="8">🔒 test</text><text x="375" y="135" text-anchor="middle" fill="#1a5c38" font-size="7.5">stays sealed ✓</text><text x="260" y="170" text-anchor="middle" fill="#6B645E" font-size="9">All your tweaking hits the validation pile — so the real exam stays an honest grade.</text><text x="260" y="192" text-anchor="middle" fill="#1a5c38" font-size="9">The middle pile exists to PROTECT the final one.</text></g></svg>
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

@@@ concept id=c4 tag="Overfitting" title="Overfitting — when the model memorizes instead of learns" gotit="Got overfitting"
This is the failure the whole *exam-vs-answer-key* setup exists to catch, so it deserves a name and a clear picture. Picture a student who, instead of understanding the topic, memorizes their practice sheet down to the tiniest useless detail — every exact word, the order the answers came in, even the little **coffee stain** on question 4. On that one sheet they are flawless. But coffee stains and answer-order are *noise* — they say nothing about the subject — so on a fresh exam, where none of that junk appears, the student falls apart. When a model does this, we call it [[overfitting||when a model memorizes the exact training examples (including their random noise) instead of learning the general pattern — so it scores high on training data but poorly on new data]].

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A student hunched over a single practice sheet, memorizing every useless detail: the exact words, the order of the answers, and a coffee stain on question four. Boxes beside the student show the memorized junk. A caption notes that memorizing the noise is not learning the subject."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Memorizing every useless detail is NOT learning</text><rect x="60" y="34" width="180" height="150" rx="8" fill="#FDECEC" stroke="#C93B3B"/><text x="150" y="52" text-anchor="middle" fill="#C93B3B" font-size="9.5">🙈 the memorizing student</text><rect x="86" y="66" width="128" height="102" rx="4" fill="#fff" stroke="#C93B3B"/><text x="150" y="84" text-anchor="middle" fill="#6B645E" font-size="8">the ONE practice sheet</text><line x1="98" y1="96" x2="202" y2="96" stroke="#D9CFC2"/><line x1="98" y1="110" x2="202" y2="110" stroke="#D9CFC2"/><line x1="98" y1="124" x2="202" y2="124" stroke="#D9CFC2"/><line x1="98" y1="138" x2="180" y2="138" stroke="#D9CFC2"/><circle cx="176" cy="150" r="9" fill="#B98A3E" opacity="0.55"/><text x="176" y="153" text-anchor="middle" fill="#7A5A22" font-size="7">☕</text><text x="150" y="180" text-anchor="middle" fill="#1a5c38" font-size="8.5">score on THIS sheet: 100% ✓</text><rect x="286" y="34" width="214" height="150" rx="8" fill="#FDF3D6" stroke="#C99A12"/><text x="393" y="52" text-anchor="middle" fill="#9A7208" font-size="9">what got memorized (all noise):</text><rect x="300" y="66" width="186" height="22" rx="4" fill="#fff" stroke="#C99A12"/><text x="393" y="81" text-anchor="middle" fill="#9A7208" font-size="8">🔤 the exact wording</text><rect x="300" y="94" width="186" height="22" rx="4" fill="#fff" stroke="#C99A12"/><text x="393" y="109" text-anchor="middle" fill="#9A7208" font-size="8">🔢 the ORDER of the answers</text><rect x="300" y="122" width="186" height="22" rx="4" fill="#fff" stroke="#C99A12"/><text x="393" y="137" text-anchor="middle" fill="#9A7208" font-size="8">☕ the coffee stain on Q4</text><text x="393" y="166" text-anchor="middle" fill="#C93B3B" font-size="8">none of this is on the real exam →</text><text x="393" y="178" text-anchor="middle" fill="#C93B3B" font-size="8">fresh exam: crashes ✗</text></g></svg>
%%%

**What the memorizing-student picture gets right:** clinging to noise — the exact words, the order, the coffee stain — feels like knowledge on the one sheet you studied, but it is worthless anywhere else, so a fresh exam exposes it.

**Where it breaks down:** a student could, in principle, *notice* they are cramming trivia and stop. A model cannot — it will absorb every stray detail, because for a big flexible model storing the junk is easier than finding the pattern underneath.

#### From "memorizing the noise" to a wiggly curve
Now watch *what that memorizing does to the model's shape* — this is the heart of overfitting, so we take it one step at a time. Say your training data are seven dots that follow a gentle rising trend, jiggled up and down a little by random noise. A model that **learns the pattern** draws a *smooth* line through the middle of the cloud, ignoring the jiggle. A model that **memorizes** does the opposite: it bends and detours to pass *exactly* through every single dot, chasing each bit of noise — which forces the line to whip up and down between them. That whipping shape is what people mean by a **wiggly** curve. Same seven dots, twice, so you can *see* memorizing happen:

%%% svg
<svg viewBox="0 0 520 224" role="img" aria-label="A staged before-and-after of the exact same seven data dots. Step one on the left: the seven raw noisy dots with no curve, following a gentle rising trend with random jiggle. Step two top-right: a smooth line drawn through the middle of the very same seven dots, labelled learned the pattern, ignoring the jiggle, and passing close to a new point marked at the same input position. Step three bottom-right: a wild wiggly line bending to touch every single one of those identical seven dots exactly, labelled memorized every dot, which whips up and down and, at that same new input position, plunges far away from the new point so it misses it badly."><g font-family="monospace" font-size="9"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The SAME seven dots → smooth vs. wiggly</text><g><rect x="12" y="30" width="180" height="184" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text x="102" y="48" text-anchor="middle" fill="#6B645E" font-size="9">① seven raw noisy dots</text><g fill="#2C2A28"><circle cx="40" cy="178" r="3.2"/><circle cx="62" cy="150" r="3.2"/><circle cx="84" cy="164" r="3.2"/><circle cx="106" cy="120" r="3.2"/><circle cx="128" cy="140" r="3.2"/><circle cx="150" cy="94" r="3.2"/><circle cx="176" cy="100" r="3.2"/></g><text x="102" y="204" text-anchor="middle" fill="#6B645E" font-size="7.5">a gentle rise + jiggle</text></g><g><rect x="204" y="30" width="150" height="184" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><text x="279" y="48" text-anchor="middle" fill="#1a5c38" font-size="9">② learned: smooth</text><path d="M224 172 C 262 150, 300 118, 344 102" fill="none" stroke="#2D8B55" stroke-width="2.6"/><g fill="#2C2A28"><circle cx="224" cy="178" r="2.6"/><circle cx="244" cy="150" r="2.6"/><circle cx="264" cy="164" r="2.6"/><circle cx="286" cy="120" r="2.6"/><circle cx="306" cy="140" r="2.6"/><circle cx="326" cy="94" r="2.6"/><circle cx="344" cy="100" r="2.6"/></g><circle cx="290" cy="118" r="4" fill="none" stroke="#1F6280" stroke-width="1.6"/><text x="279" y="196" text-anchor="middle" fill="#1F6280" font-size="7.5">new point → close ✓</text></g><g><rect x="366" y="30" width="142" height="184" rx="6" fill="#FDECEC" stroke="#C93B3B"/><text x="437" y="48" text-anchor="middle" fill="#C93B3B" font-size="9">③ memorized: wiggly</text><path d="M380 178 L 389 132 L 398 150 L 407 184 L 416 164 L 425 108 L 434 120 L 443 188 L 452 140 L 461 96 L 470 94 L 482 150 L 494 100" fill="none" stroke="#C93B3B" stroke-width="2"/><g fill="#2C2A28"><circle cx="380" cy="178" r="2.6"/><circle cx="398" cy="150" r="2.6"/><circle cx="416" cy="164" r="2.6"/><circle cx="434" cy="120" r="2.6"/><circle cx="452" cy="140" r="2.6"/><circle cx="470" cy="94" r="2.6"/><circle cx="494" cy="100" r="2.6"/></g><circle cx="443" cy="118" r="4" fill="none" stroke="#C93B3B" stroke-width="1.6"/><text x="437" y="196" text-anchor="middle" fill="#C93B3B" font-size="7.5">new point → way off ✗</text></g></g></svg>
%%%

**What this build-up shows:** the *exact same seven dots* give a calm, useful curve when the model ignores the jiggle (②), and a frantic, useless curve when it insists on touching every dot (③). Panel ③ is memorizing made visible. **Where the drawing breaks down:** here you see one tidy curve in 2-D, but a real model has thousands of weights doing this in many dimensions at once — you cannot eyeball the wiggle, which is exactly why the next concept gives you a *number*.

#### Your turn: drag the flexibility dial and make it wiggle
You do not have to take my word for any of that — you can *cause* it. The dial below is the model's **flexibility**: how bendy the curve is allowed to get. In a real model that dial is its [[capacity||how much a model can bend or store — more neurons, more layers, a higher-degree curve. High capacity is exactly what makes memorizing possible]]. Predict first: at the far LEFT the curve can barely bend at all — do you think it fits the dots well? Now drag.

%%% svg
<svg id="flex-svg" viewBox="0 0 520 244" role="img" aria-label="An interactive fit explorer. Seven noisy training dots follow a gentle rising trend. A single curve is drawn through them and changes as you drag a flexibility dial from one to ten. At flexibility one the curve is a straight stiff line that misses most dots, so both the training error and the new-data error are high: underfitting. Around flexibility five the curve is smooth and passes near every dot, and the new-data error is at its lowest: the sweet spot. At flexibility ten the curve whips up and down to touch every single dot exactly, so the training error is almost zero while the new-data error explodes: overfitting."><g font-family="monospace" font-size="10"><text x="260" y="17" text-anchor="middle" fill="#2C2A28" font-size="12">Drag the flexibility dial — watch the curve start to wiggle</text><rect x="60" y="28" width="418" height="174" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="202" x2="478" y2="202" stroke="#B8AEA2"/><line x1="60" y1="28" x2="60" y2="202" stroke="#B8AEA2"/><text x="44" y="118" fill="#6B645E" font-size="9" transform="rotate(-90 44 118)">output</text><text x="269" y="220" text-anchor="middle" fill="#6B645E" font-size="9">input →</text><path id="flex-curve" d="" fill="none" stroke="#C93B3B" stroke-width="2.4"/><g fill="#2C2A28"><circle cx="70" cy="175" r="3.4"/><circle cx="135" cy="148" r="3.4"/><circle cx="200" cy="160" r="3.4"/><circle cx="265" cy="120" r="3.4"/><circle cx="330" cy="138" r="3.4"/><circle cx="395" cy="96" r="3.4"/><circle cx="460" cy="92" r="3.4"/></g><text x="470" y="196" text-anchor="end" fill="#6B645E" font-size="8">• = the 7 training dots (noisy)</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label>model flexibility <input id="flex-r" type="range" min="1" max="10" step="1" value="1" style="width:52%;accent-color:#C93B3B;vertical-align:middle"> <b id="flex-v">1</b> / 10</label>
<div id="flex-out" style="margin-top:6px;color:#2C2A28">flexibility 1 → a stiff straight line: it misses most dots — <b>underfitting</b></div>
<div id="flex-err" style="margin-top:4px;color:#6B645E">error on the training dots: 0.31 · error on NEW data: 0.34</div>
</div>
<script>(function(){
  var r=document.getElementById('flex-r');if(!r)return;
  var cur=document.getElementById('flex-curve'),out=document.getElementById('flex-out'),
      ev=document.getElementById('flex-v'),er=document.getElementById('flex-err');
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
    var msg,col;
    if(f<=3){msg='too stiff to follow the trend: it misses most dots, and it is just as bad on new data — <b>underfitting</b>';col='#1F6280';}
    else if(f<=6){msg='smooth, close to every dot, ignoring the jiggle — the <b>sweet spot</b> ⭐ (new-data error bottoms out around here)';col='#2D8B55';}
    else{msg='now it whips up and down to touch every dot exactly — it is memorizing the noise: <b>overfitting</b> 💥';col='#C93B3B';}
    out.innerHTML='flexibility '+f+' → '+msg;out.style.color=col;
    er.innerHTML='error on the training dots: <b>'+trainE.toFixed(2)+'</b> · error on NEW data: <b>'+valE.toFixed(2)+'</b>';
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

@@@ concept id=c5 tag="The gap" title="The train–val gap — how you SEE overfitting happen" gotit="Got the gap"
Here is the beautiful part: you do not have to *guess* whether a model is memorizing — you can *watch* it on a chart. Picture two runners on the same track. One runner is the score on the training pile, the other is the score on the validation pile. Early on they run shoulder to shoulder, both getting better together — and that closeness means the model is learning real patterns that work on *both* piles. Then, if the model starts memorizing, they come apart: the training runner keeps sprinting while the validation runner slows and stops. **The space that opens between the two runners is overfitting, made visible.** We call it the [[train–val gap||the difference between the training score and the validation score — a small, steady gap is healthy; a gap that keeps widening is overfitting you can see]].

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two runners on a two-lane track as an analogy for the train and validation scores. In the top lane the training runner has sprinted far ahead toward the finish. In the bottom lane the validation runner has slowed and stopped much earlier on the track. A dashed bracket between the two runners is labelled the gap, and a caption says the space between the runners is overfitting made visible."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two runners, one track — mind the space between them</text><rect x="40" y="30" width="446" height="44" rx="6" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="40" y="84" width="446" height="44" rx="6" fill="#FDF3D6" stroke="#C99A12"/><g stroke="#B8AEA2" stroke-dasharray="6,8"><line x1="40" y1="52" x2="486" y2="52"/><line x1="40" y1="106" x2="486" y2="106"/></g><line x1="52" y1="24" x2="52" y2="134" stroke="#6B645E" stroke-width="1.4"/><text x="52" y="146" text-anchor="middle" fill="#6B645E" font-size="8">start</text><line x1="474" y1="24" x2="474" y2="134" stroke="#2D8B55" stroke-width="1.6"/><text x="474" y="146" text-anchor="middle" fill="#1a5c38" font-size="8">finish</text><circle cx="404" cy="52" r="13" fill="#2A7B9B"/><text x="404" y="57" text-anchor="middle" font-size="13">🏃</text><text x="404" y="26" text-anchor="middle" fill="#1F6280" font-size="8.5">training score — still sprinting</text><circle cx="226" cy="106" r="13" fill="#C99A12"/><text x="226" y="111" text-anchor="middle" font-size="13">🚶</text><text x="226" y="170" text-anchor="middle" fill="#9A7208" font-size="8.5">validation score — slowed, then stopped</text><path d="M226 132 L 226 152 L 404 152 L 404 70" fill="none" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="4,3"/><text x="315" y="148" text-anchor="middle" fill="#C93B3B" font-size="9">the GAP</text></g></svg>
%%%

**What the two-runners picture gets right:** while the model is learning real patterns, both scores improve together and stay close; the moment it starts memorizing, only the training score keeps improving, and the space that opens between them is the overfitting.

**Where it breaks down:** a runner who slows down is still moving toward the finish, and keeps every metre already covered. A validation score does something no runner can do — it turns around and goes *backwards*, because the model is actively getting *worse* on new data even as it looks better on practice.

#### Now put it on a chart you can drag
Both runners' progress is drawn as a **loss** curve over epochs (loss is the "how wrong am I" number from Day 4, and *lower is better*; an epoch is one full pass over the training split). So "running well" means the curve going *down*. Two curves on one chart, and the whole story of a training run is in the space between them.

Predict before you drag: at which point on the chart do you think the *best* model lives — at the far right where training loss is smallest, or somewhere earlier? Drag the slider from left to right and find out.

%%% svg
<svg id="gap-svg" viewBox="0 0 520 250" role="img" aria-label="Interactive loss curves over training epochs. A blue training-loss curve keeps dropping. An amber validation-loss curve drops with it at first, reaches its lowest point around epoch ten, then rises. Drag the epoch slider: early on the two curves are close and both dropping together with a small steady gap (healthy). At the lowest validation point the gap is about to open (the best model, where you would stop). Just after it, the validation curve turns the corner and the gap starts to open. Later the gap is wide as validation climbs while training keeps falling (overfitting). A marker slides along both curves and the readout names the regime."><g font-family="monospace" font-size="10"><rect x="46" y="26" width="440" height="150" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><text x="266" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The gap between the two runners IS overfitting</text><line x1="46" y1="176" x2="486" y2="176" stroke="#B8AEA2"/><line x1="46" y1="26" x2="46" y2="176" stroke="#B8AEA2"/><text x="30" y="104" fill="#6B645E" font-size="9" transform="rotate(-90 30 104)">loss</text><text x="266" y="196" text-anchor="middle" fill="#6B645E" font-size="9">training epochs →</text><path id="gap-train" d="" fill="none" stroke="#2A7B9B" stroke-width="2.6"/><path id="gap-val" d="" fill="none" stroke="#C99A12" stroke-width="2.6"/><line id="gap-line" x1="0" y1="0" x2="0" y2="0" stroke="#C93B3B" stroke-width="1.4" stroke-dasharray="3,3" opacity="0"/><circle id="gap-dt" cx="0" cy="0" r="4.5" fill="#2A7B9B" opacity="0"/><circle id="gap-dv" cx="0" cy="0" r="4.5" fill="#C99A12" opacity="0"/><text x="470" y="150" text-anchor="end" fill="#1F6280" font-size="8.5">train loss ↓ (keeps dropping)</text><text x="470" y="46" text-anchor="end" fill="#9A7208" font-size="8.5">val loss ↓ then ↑</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label>training epoch <input id="gap-ep" type="range" min="1" max="30" step="1" value="6" style="width:56%;accent-color:#2A7B9B;vertical-align:middle"> <b id="gap-epv">6</b></label>
<div id="gap-out" style="margin-top:6px;color:#2C2A28">epoch 6 → both curves dropping together, small steady gap — <b>healthy learning</b> ✓</div>
</div>
<script>(function(){
  var ep=document.getElementById('gap-ep');if(!ep)return;
  var tp=document.getElementById('gap-train'),vp=document.getElementById('gap-val'),
      dt=document.getElementById('gap-dt'),dv=document.getElementById('gap-dv'),
      gl=document.getElementById('gap-line'),out=document.getElementById('gap-out'),epv=document.getElementById('gap-epv');
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
    var gap=valL(e)-trainL(e),msg,col;
    if(e<BOTTOM){msg='both curves still dropping together, small steady gap — <b>healthy learning</b> ✓';col='#2D8B55';}
    else if(e===BOTTOM){msg='validation is at its <b>lowest point</b> — the best model, and where you would stop ⭐';col='#C99A12';}
    else if(e<=13){msg='validation has just <b>turned the corner</b> and started climbing — the gap is beginning to open (you are past the best point)';col='#B9761A';}
    else{msg='validation is climbing while training keeps falling — the gap is now <b>wide open: overfitting</b> 💥';col='#C93B3B';}
    out.innerHTML='epoch '+e+' → '+msg+'  <span style="color:#6B645E">(gap = '+gap.toFixed(2)+')</span>';
    out.style.color=col;
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

@@@ concept id=c6 tag="Peek & shuffle" title="Two ways a split lies: the peek and the sorted slice" gotit="Got these two"
Even with three piles, a split can quietly betray you — and the two traps in this unit are the ones that catch careful people. Think about dealing cards for a game with your friends. Two house rules make the deal fair: you **shuffle** the deck first, and nobody **peeks** at the pile you set aside. Break either one and the game is rigged, even if everybody smiles politely. A data split has exactly the same two house rules.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A card game as an analogy for a fair split. Two house rules are drawn: on the left a shuffled deck so no hand is stacked, and on the right a sealed pile with a padlock that nobody peeks at. A caption says in data these two rules become shuffle before you slice and never peek at the test pile."><g font-family="monospace" font-size="9"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A fair deal has two house rules — break one, the game is rigged</text><g><rect x="30" y="30" width="210" height="120" rx="8" fill="#EAF3F7" stroke="#2A7B9B"/><text x="135" y="52" text-anchor="middle" fill="#1F6280" font-size="10">🃏 shuffle the deck first</text><g fill="#2A7B9B"><rect x="66" y="70" width="20" height="30" rx="3" transform="rotate(-8 76 85)"/><rect x="96" y="76" width="20" height="30" rx="3" transform="rotate(5 106 91)"/><rect x="126" y="68" width="20" height="30" rx="3" transform="rotate(-3 136 83)"/><rect x="156" y="74" width="20" height="30" rx="3" transform="rotate(9 166 89)"/></g><text x="135" y="132" text-anchor="middle" fill="#6B645E" font-size="8.5">so every hand is a fair mix</text></g><g><rect x="280" y="30" width="210" height="120" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="385" y="52" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 nobody peeks at the set-aside pile</text><rect x="356" y="64" width="58" height="42" rx="3" fill="#fff" stroke="#2D8B55"/><text x="385" y="92" text-anchor="middle" font-size="17">🔒</text><text x="385" y="132" text-anchor="middle" fill="#6B645E" font-size="8.5">it is opened once, at the end</text></g><text x="260" y="168" text-anchor="middle" fill="#6B645E" font-size="8.5">In data these become: shuffle before you slice · never peek at the test pile.</text></g></svg>
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

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="Two traps drawn as before-and-after pairs. Trap one, peeking: before, an arrow loops from the sealed test pile back into your tuning decisions, so the grade is contaminated; after, the test pile stays sealed and is used only once, so the grade is honest. Trap two, unshuffled: before, sorted data sliced straight sends all of one label into train and all of the other into test; after, shuffling before slicing gives every pile a fair mix of both labels."><g font-family="monospace" font-size="8.5"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Each trap: before (rigged) → after (fixed)</text><text x="150" y="32" text-anchor="middle" fill="#C93B3B" font-size="9">BEFORE — rigged</text><text x="392" y="32" text-anchor="middle" fill="#1a5c38" font-size="9">AFTER — fixed</text><g><text x="14" y="72" fill="#6B645E" font-size="8.5">① PEEKING</text><rect x="70" y="44" width="160" height="58" rx="6" fill="#FDECEC" stroke="#C93B3B"/><rect x="86" y="58" width="42" height="30" rx="3" fill="#fff" stroke="#C93B3B"/><text x="107" y="77" text-anchor="middle" fill="#C93B3B" font-size="7.5">🔒 test</text><path d="M130 64 C 178 52, 200 82, 168 86" fill="none" stroke="#C93B3B" stroke-width="1.4"/><polygon points="168,86 176,82 176,90" fill="#C93B3B"/><text x="196" y="66" fill="#C93B3B" font-size="7">you tune</text><text x="196" y="78" fill="#C93B3B" font-size="7">on its score 💥</text><rect x="312" y="44" width="160" height="58" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><rect x="328" y="58" width="42" height="30" rx="3" fill="#fff" stroke="#2D8B55"/><text x="349" y="77" text-anchor="middle" fill="#1a5c38" font-size="7.5">🔒 test</text><text x="392" y="68" fill="#1a5c38" font-size="7">sealed — opened</text><text x="392" y="80" fill="#1a5c38" font-size="7">ONCE, at the end ✓</text></g><g><text x="14" y="166" fill="#6B645E" font-size="8.5">② SORTED SLICE</text><rect x="70" y="128" width="160" height="62" rx="6" fill="#FDECEC" stroke="#C93B3B"/><g fill="#2A7B9B"><rect x="86" y="144" width="12" height="18"/><rect x="100" y="144" width="12" height="18"/><rect x="114" y="144" width="12" height="18"/><rect x="128" y="144" width="12" height="18"/></g><g fill="#C99A12"><rect x="176" y="144" width="12" height="18"/><rect x="190" y="144" width="12" height="18"/><rect x="204" y="144" width="12" height="18"/></g><text x="110" y="176" text-anchor="middle" fill="#1F6280" font-size="7">all ordinary → train</text><text x="192" y="176" text-anchor="middle" fill="#9A7208" font-size="7">all spam → test 💥</text><rect x="312" y="128" width="160" height="62" rx="6" fill="#EAF5EE" stroke="#2D8B55"/><g><rect x="328" y="144" width="12" height="18" fill="#2A7B9B"/><rect x="342" y="144" width="12" height="18" fill="#C99A12"/><rect x="356" y="144" width="12" height="18" fill="#2A7B9B"/><rect x="370" y="144" width="12" height="18" fill="#2A7B9B"/><rect x="416" y="144" width="12" height="18" fill="#C99A12"/><rect x="430" y="144" width="12" height="18" fill="#2A7B9B"/><rect x="444" y="144" width="12" height="18" fill="#2A7B9B"/></g><text x="392" y="176" text-anchor="middle" fill="#1a5c38" font-size="7">shuffle first → every pile a fair mix ✓</text></g><text x="260" y="210" text-anchor="middle" fill="#6B645E" font-size="8.5">Same shape both times: information ends up in the wrong pile. Same cure: one habit, applied up front.</text></g></svg>
%%%

Two traps down, and look how cheap the cures are: seal the test pile, shuffle before you slice. You have just learned to survive the two mistakes that quietly ruin more beginner projects than any modelling error. The next one is different, though — it hides *inside* code you wrote yourself, and it looks completely innocent.

@@@ concept id=c7 tag="The hidden leak" title="The leak inside your own innocent code" gotit="Got leakage"
This one is my favourite, because the code that causes it looks *helpful*. Imagine a game where three friends hide behind a curtain and you must guess their heights. To guess well you first work out the **average height of everyone in the room** — and, without thinking, you include the three hidden friends in that average. Now your measuring stick secretly carries information about the very people you were supposed to guess. You did not look behind the curtain. You did not cheat. But your ruler did.

That is [[leakage||when information from the validation or test piles sneaks into training — through a shared calculation, a duplicate row, or anything else the model should not have known. It always shows up as a suspiciously good score]], and it is the trap that catches people who have already learned the other two.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A guessing game as an analogy for preprocessing leakage. On the left, six visible friends stand in the open, labelled the training pile. On the right, three friends hide behind a curtain, labelled the sealed pile you must guess. Arrows run from all nine friends, including the three hidden ones, into a box labelled measuring stick built from the average of everyone. A red caption warns the stick now secretly knows about the hidden friends."><g font-family="monospace" font-size="9"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">Your ruler peeked, even though you did not</text><g stroke="#2A7B9B" fill="#2A7B9B"><g><circle cx="40" cy="52" r="6"/><line x1="40" y1="58" x2="40" y2="76" stroke-width="2"/></g><g><circle cx="66" cy="46" r="6"/><line x1="66" y1="52" x2="66" y2="76" stroke-width="2"/></g><g><circle cx="92" cy="56" r="6"/><line x1="92" y1="62" x2="92" y2="76" stroke-width="2"/></g><g><circle cx="118" cy="48" r="6"/><line x1="118" y1="54" x2="118" y2="76" stroke-width="2"/></g><g><circle cx="144" cy="54" r="6"/><line x1="144" y1="60" x2="144" y2="76" stroke-width="2"/></g><g><circle cx="170" cy="44" r="6"/><line x1="170" y1="50" x2="170" y2="76" stroke-width="2"/></g></g><line x1="26" y1="78" x2="184" y2="78" stroke="#B8AEA2"/><text x="105" y="92" text-anchor="middle" fill="#1F6280" font-size="8.5">the 6 you can see = training pile</text><rect x="300" y="30" width="150" height="50" rx="3" fill="#E7DCEB" stroke="#7A5A8E"/><text x="375" y="50" text-anchor="middle" fill="#5C4270" font-size="9">🎭 curtain</text><text x="375" y="66" text-anchor="middle" fill="#5C4270" font-size="8">3 hidden friends</text><text x="375" y="92" text-anchor="middle" fill="#5C4270" font-size="8.5">the sealed pile you must guess</text><rect x="150" y="118" width="220" height="34" rx="6" fill="#FDF3D6" stroke="#C99A12"/><text x="260" y="139" text-anchor="middle" fill="#9A7208" font-size="9">📏 measuring stick = average of EVERYONE</text><g stroke="#C99A12" stroke-width="1.3" fill="none"><path d="M105 96 L 200 116"/><path d="M375 96 L 330 116"/></g><polygon points="200,116 192,110 195,119" fill="#C99A12"/><polygon points="330,116 336,109 338,118" fill="#C93B3B"/><text x="392" y="112" fill="#C93B3B" font-size="8">this arrow is the leak 💥</text><text x="260" y="172" text-anchor="middle" fill="#C93B3B" font-size="9">The stick now secretly knows about the hidden three —</text><text x="260" y="188" text-anchor="middle" fill="#C93B3B" font-size="9">so your guesses look better than your skill deserves.</text></g></svg>
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

%%% svg
<svg id="dup-svg" viewBox="0 0 520 176" role="img" aria-label="An interactive bar showing how duplicate rows inflate a reported score. A horizontal bar grows as you drag the number of leaked duplicate rows from zero to twenty-five. The part of the bar up to the honest score of zero point eight one is drawn in green, and any part beyond it is drawn in red and labelled fiction. A dashed vertical marker shows the honest score. With zero leaked rows the bar stops exactly at the marker; with twenty leaked rows it reaches zero point eight nine, well past the marker."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Leaked duplicates only ever push the score UP</text><line x1="70" y1="112" x2="470" y2="112" stroke="#B8AEA2"/><text x="70" y="126" text-anchor="middle" fill="#6B645E" font-size="8">0.0</text><text x="470" y="126" text-anchor="middle" fill="#6B645E" font-size="8">1.0</text><rect id="dup-ok" x="70" y="60" width="324" height="34" rx="3" fill="#2D8B55"/><rect id="dup-lie" x="394" y="60" width="0" height="34" rx="3" fill="#C93B3B"/><line x1="394" y1="46" x2="394" y2="108" stroke="#2C2A28" stroke-width="1.4" stroke-dasharray="4,3"/><text x="394" y="42" text-anchor="middle" fill="#2C2A28" font-size="8.5">honest score 0.81</text><text id="dup-lbl" x="86" y="82" fill="#fff" font-size="9">reported 0.81</text><text x="260" y="152" text-anchor="middle" fill="#6B645E" font-size="8.5">green = skill the model really has · red = rows it simply recited</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<label>leaked duplicate rows in the 50-row test pile <input id="dup-r" type="range" min="0" max="25" step="1" value="0" style="width:38%;accent-color:#C93B3B;vertical-align:middle"> <b id="dup-v">0</b></label>
<div id="dup-out" style="margin-top:6px;color:#2C2A28">0 leaked rows → reported <b>0.81</b>, which is the truth. This is what a clean split looks like ✓</div>
</div>
<script>(function(){
  var r=document.getElementById('dup-r');if(!r)return;
  var ok=document.getElementById('dup-ok'),lie=document.getElementById('dup-lie'),
      lbl=document.getElementById('dup-lbl'),out=document.getElementById('dup-out'),v=document.getElementById('dup-v');
  var X0=70,W=400,HONEST=0.81,NTEST=50;
  function paint(){
    var k=+r.value;
    // leaked rows are recited (counted correct); the rest score at the honest rate
    var s=(HONEST*(NTEST-k)+k)/NTEST, fake=Math.round((s-HONEST)*100);
    ok.setAttribute('width',(HONEST*W).toFixed(1));
    lie.setAttribute('x',(X0+HONEST*W).toFixed(1));
    lie.setAttribute('width',((s-HONEST)*W).toFixed(1));
    lbl.textContent='reported '+s.toFixed(2);
    v.textContent=k;
    if(k===0){out.innerHTML='0 leaked rows → reported <b>0.81</b>, which is the truth. This is what a clean split looks like ✓';out.style.color='#1a5c38';}
    else{out.innerHTML=k+' leaked row'+(k===1?'':'s')+' → reported <b>'+s.toFixed(2)+'</b> = honest 0.81 + <b>'+fake+' point'+(fake===1?'':'s')+' of fiction</b> 💥 (the model recited '+k+' of the 50 exam rows)';out.style.color='#C93B3B';}
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

@@@ concept id=c8 tag="Unlucky slice" title="The unlucky slice — and cross-validation" gotit="Got cross-validation"
Do everything right — shuffle, seal, split first — and a small split can still be simply **unlucky**. Picture judging a whole cake by tasting *one* slice. If that slice happens to contain the only raisin, you will announce "this cake is raisin-heavy!" and be completely wrong. Not dishonest. Just unlucky.

A small validation pile does the same thing to your model. One random slice can flatter it (an easy slice — the score looks great) or punish it (a hard slice — the score looks terrible), and neither number is the truth. The **cause** is that a small pile is *high-variance*: change which examples landed in it and the score jumps around.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A cake cut into five slices as an analogy for an unlucky validation split. The whole cake is drawn on the left with one raisin in a single slice. That one slice sits on a plate on the right, with the raisin in it, and a taster declares the whole cake is raisin-heavy. A caption notes that one slice can badly mislead you about the whole cake."><g font-family="monospace" font-size="9"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Judging a whole cake from one slice</text><g><circle cx="120" cy="106" r="62" fill="#FDF3D6" stroke="#C99A12" stroke-width="1.6"/><g stroke="#C99A12" stroke-width="1.2"><line x1="120" y1="106" x2="182" y2="106"/><line x1="120" y1="106" x2="139" y2="47"/><line x1="120" y1="106" x2="70" y2="70"/><line x1="120" y1="106" x2="70" y2="142"/><line x1="120" y1="106" x2="139" y2="165"/></g><circle cx="150" cy="86" r="4.6" fill="#5C4270"/><text x="120" y="184" text-anchor="middle" fill="#6B645E" font-size="8.5">the whole cake · exactly ONE raisin</text></g><path d="M300 60 L 372 92 L 300 124 Z" fill="#FDF3D6" stroke="#C99A12" stroke-width="1.6"/><circle cx="330" cy="92" r="4.6" fill="#5C4270"/><text x="336" y="140" text-anchor="middle" fill="#6B645E" font-size="8.5">the one slice you tasted</text><path d="M240 92 L 292 92" stroke="#6B645E" stroke-width="1.4"/><polygon points="292,92 284,88 284,96" fill="#6B645E"/><rect x="386" y="66" width="122" height="52" rx="8" fill="#FDECEC" stroke="#C93B3B"/><text x="447" y="88" text-anchor="middle" fill="#C93B3B" font-size="9">"this cake is</text><text x="447" y="104" text-anchor="middle" fill="#C93B3B" font-size="9">raisin-heavy!" 💥</text><text x="260" y="176" text-anchor="middle" fill="#6B645E" font-size="9">One slice can be unusual — so one small validation pile can lie by pure luck.</text></g></svg>
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

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Five rounds of k-fold cross-validation shown as five rows. The data is split into five equal slices. In each row a different slice is highlighted as the validation pile while the other four are training. The five validation scores are averaged at the bottom into one steady number, so no single unlucky slice decides the result."><g font-family="monospace" font-size="9"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">k-fold: rotate the validation slice, then average</text><g font-size="8.5"><g><text x="40" y="46" fill="#6B645E">round 1</text><rect x="88" y="34" width="70" height="16" fill="#C99A12"/><rect x="160" y="34" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="232" y="34" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="304" y="34" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="376" y="34" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/></g><g><text x="40" y="70" fill="#6B645E">round 2</text><rect x="88" y="58" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="160" y="58" width="70" height="16" fill="#C99A12"/><rect x="232" y="58" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="304" y="58" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="376" y="58" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/></g><g><text x="40" y="94" fill="#6B645E">round 3</text><rect x="88" y="82" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="160" y="82" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="232" y="82" width="70" height="16" fill="#C99A12"/><rect x="304" y="82" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="376" y="82" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/></g><g><text x="40" y="118" fill="#6B645E">round 4</text><rect x="88" y="106" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="160" y="106" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="232" y="106" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="304" y="106" width="70" height="16" fill="#C99A12"/><rect x="376" y="106" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/></g><g><text x="40" y="142" fill="#6B645E">round 5</text><rect x="88" y="130" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="160" y="130" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="232" y="130" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="304" y="130" width="70" height="16" fill="#EAF3F7" stroke="#2A7B9B"/><rect x="376" y="130" width="70" height="16" fill="#C99A12"/></g></g><rect x="88" y="156" width="18" height="12" fill="#C99A12"/><text x="112" y="166" fill="#9A7208" font-size="7.5">= validation this round</text><line x1="88" y1="176" x2="446" y2="176" stroke="#B8AEA2" stroke-dasharray="2,2"/><text x="260" y="192" text-anchor="middle" fill="#1a5c38" font-size="9">average the 5 scores → one steady, trustworthy number</text><text x="260" y="206" text-anchor="middle" fill="#6B645E" font-size="8">no single lucky or unlucky slice decides the verdict</text></g></svg>
%%%

%%% insight
There is a lovely extreme version: set k to the number of rows, so each round holds out exactly *one* example. It is called leave-one-out, it squeezes the most out of tiny data — and it costs one full training run *per row*, which is why almost nobody runs it on anything big. Worth knowing it exists, mostly so you can smile at the price tag.
%%%

Two companion habits for the same wobble: if a holdout is so small that flipping one example moves the score by several points, **make it bigger** — and when you report a number, report the *spread* too (the same experiment across a few seeds or folds), so nobody mistakes noise for progress.

#### So how big should each pile be?
The question everyone asks, and the answer is friendlier than you expect. Common starting points are **70/15/15** or **80/10/10** — train / validation / test. But **no ratio is sacred**. A validation or test pile has just one requirement: be big enough to be a fair sample. With a tiny dataset you give eval a bigger share (or use k-fold instead). With **bigger data** you can give it a much **smaller fraction** — ten million rows and a 1% validation split is still 100,000 examples, which is plenty, so you keep almost everything for training. What matters is the *count*, not the percentage.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Two horizontal bars showing common split ratios. The top bar is 70 percent train, 15 percent validation, 15 percent test. The bottom bar is 80 percent train, 10 percent validation, 10 percent test. A caption notes that with very large datasets even a 1 percent eval slice is plenty, so no ratio is sacred."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Common ratios — but no ratio is sacred</text><g><text x="16" y="58" fill="#6B645E" font-size="9">70/15/15</text><rect x="90" y="44" width="280" height="24" fill="#2A7B9B"/><rect x="370" y="44" width="60" height="24" fill="#C99A12"/><rect x="430" y="44" width="60" height="24" fill="#2D8B55"/><text x="230" y="61" text-anchor="middle" fill="#fff" font-size="8.5">train 70%</text><text x="400" y="61" text-anchor="middle" fill="#fff" font-size="7.5">val 15</text><text x="460" y="61" text-anchor="middle" fill="#fff" font-size="7.5">test 15</text></g><g><text x="16" y="106" fill="#6B645E" font-size="9">80/10/10</text><rect x="90" y="92" width="320" height="24" fill="#2A7B9B"/><rect x="410" y="92" width="40" height="24" fill="#C99A12"/><rect x="450" y="92" width="40" height="24" fill="#2D8B55"/><text x="250" y="109" text-anchor="middle" fill="#fff" font-size="8.5">train 80%</text><text x="430" y="109" text-anchor="middle" fill="#fff" font-size="7">val</text><text x="470" y="109" text-anchor="middle" fill="#fff" font-size="7">test</text></g><text x="260" y="150" text-anchor="middle" fill="#1a5c38" font-size="8.5">Bigger data → a smaller fraction still makes a fair eval slice (even ~1%).</text><text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="8.5">Keep most for study; hold enough back to grade honestly.</text></g></svg>
%%%

So: an **unlucky split** is beaten by **cross-validation** — rotate which slice is validation and average the scores — and the exact ratio is a judgement call, not a law. One thing left, and it is the most grown-up idea of the day: what even a perfect split cannot do.

@@@ concept id=c9 tag="What it can't do" title="What a good split cannot do for you" gotit="Got the limits"
A great split is honest — but honesty has limits, and knowing them is what stops you expecting magic. Think of a **thermometer**. It will tell you truthfully that you have a fever. It will not *make* you well. And it only reads correctly if you use it on the right thing — press it to a warm radiator and it happily reports 40°C about the wrong body entirely.

A data split is exactly that kind of instrument: it *reveals* the truth about your model, it does not *improve* the model, and its reading is trustworthy only under one condition.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A thermometer as an analogy for a data split. The thermometer is drawn reading a high fever. Three labels point out that it tells you the honest truth, that it does not cure you, and that it only reads correctly when used on the right body. This mirrors a data split, which reveals whether a model generalizes but does not improve it."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">A split is a thermometer: it MEASURES, it does not cure</text><rect x="120" y="36" width="16" height="96" rx="8" fill="#fff" stroke="#C93B3B" stroke-width="1.6"/><circle cx="128" cy="140" r="15" fill="#C93B3B"/><rect x="124" y="86" width="8" height="54" fill="#C93B3B"/><g stroke="#B8AEA2"><line x1="136" y1="52" x2="144" y2="52"/><line x1="136" y1="68" x2="144" y2="68"/><line x1="136" y1="84" x2="144" y2="84"/><line x1="136" y1="100" x2="144" y2="100"/></g><text x="128" y="164" text-anchor="middle" fill="#C93B3B" font-size="8">reads: FEVER!</text><text x="200" y="60" fill="#1a5c38" font-size="9">✓ tells you the honest truth</text><text x="200" y="88" fill="#C93B3B" font-size="9">✗ but it does not CURE you</text><text x="200" y="116" fill="#C93B3B" font-size="9">✗ and only if used on the right body</text><text x="200" y="144" fill="#6B645E" font-size="8">(a split is the same kind of instrument)</text></g></svg>
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

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A two-world diagram. On the left, World A: the pool your train, validation, and test splits are all drawn from, showing round blue shapes, with a caption that the test score is honest here. A large arrow crosses a dashed border to World B on the right: the shifted world the model actually meets after deployment, showing different triangular red shapes, with a caption that the honest test score says nothing here. The border is labelled distribution shift."><g font-family="monospace" font-size="9"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">A split only grades the world it was drawn from</text><g><rect x="18" y="30" width="200" height="150" rx="10" fill="#EAF3F7" stroke="#2A7B9B"/><text x="118" y="50" text-anchor="middle" fill="#1F6280" font-size="10">🌍 WORLD A</text><text x="118" y="66" text-anchor="middle" fill="#6B645E" font-size="8">your train / val / test all come from here</text><g fill="#2A7B9B"><circle cx="60" cy="96" r="9"/><circle cx="96" cy="110" r="9"/><circle cx="140" cy="92" r="9"/><circle cx="170" cy="116" r="9"/><circle cx="82" cy="140" r="9"/><circle cx="132" cy="140" r="9"/></g><text x="118" y="172" text-anchor="middle" fill="#1a5c38" font-size="8">test score is HONEST here ✓</text></g><line x1="238" y1="26" x2="238" y2="184" stroke="#C93B3B" stroke-width="1.6" stroke-dasharray="4,4"/><text x="238" y="196" text-anchor="middle" fill="#C93B3B" font-size="8">distribution shift</text><path d="M244 105 L 296 105" stroke="#6B645E" stroke-width="1.8"/><polygon points="296,105 286,100 286,110" fill="#6B645E"/><g><rect x="302" y="30" width="200" height="150" rx="10" fill="#FDECEC" stroke="#C93B3B"/><text x="402" y="50" text-anchor="middle" fill="#C93B3B" font-size="10">🌍 WORLD B</text><text x="402" y="66" text-anchor="middle" fill="#6B645E" font-size="8">the shifted world after deployment</text><g fill="#C93B3B"><polygon points="344,104 354,86 364,104"/><polygon points="384,116 394,98 404,116"/><polygon points="424,100 434,82 444,100"/><polygon points="364,146 374,128 384,146"/><polygon points="414,148 424,130 434,148"/></g><text x="402" y="172" text-anchor="middle" fill="#C93B3B" font-size="8">the honest score says NOTHING here ✗</text></g></g></svg>
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

@@@ concept id=c10 tag="Recap" title="The whole day on one page" gotit="Got the recap"
Here is a fresh way to hold the whole day in your head: a **driving test**. You *practise* for weeks in your own neighbourhood (the training set). Now and then a friend rides along and quizzes you on the tricky moves so you can fix them before the big day (the validation set). Finally an examiner you have never met takes you on a route you have never driven, once, for your real licence (the test set). If you had "practised" by memorizing that examiner's exact route, you would pass and still be a danger on every other road — that is overfitting, and the difference between your smooth practice runs and your shaky new-route runs is how anyone would *see* it. **Where the driving-test picture breaks down:** an examiner can change the route on the spot, but your test set is frozen the moment you seal it — so you have to design it fairly up front, because you only get to use it once.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="A driving-test analogy in three steps. First, practice in your own neighbourhood, labelled training. Second, a friend quizzes you on tricky moves to fix, labelled validation. Third, an examiner tests you once on a new route for your licence, labelled test. A caption ties it to learning versus memorizing the route."><g font-family="monospace" font-size="9.5"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The driving test — practise, quiz, then the real examiner</text><rect x="20" y="34" width="150" height="96" rx="8" fill="#EAF3F7" stroke="#2A7B9B"/><text x="95" y="56" text-anchor="middle" fill="#1F6280" font-size="10">🚗 practise</text><text x="95" y="82" text-anchor="middle" fill="#6B645E" font-size="8">own neighbourhood</text><text x="95" y="102" text-anchor="middle" fill="#1F6280" font-size="8.5">= TRAINING</text><text x="95" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">learn here</text><rect x="185" y="34" width="150" height="96" rx="8" fill="#FDF3D6" stroke="#C99A12"/><text x="260" y="56" text-anchor="middle" fill="#9A7208" font-size="10">🧑‍🏫 friend quizzes</text><text x="260" y="82" text-anchor="middle" fill="#6B645E" font-size="8">fix the tricky moves</text><text x="260" y="102" text-anchor="middle" fill="#9A7208" font-size="8.5">= VALIDATION</text><text x="260" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">check + adjust</text><rect x="350" y="34" width="150" height="96" rx="8" fill="#EAF5EE" stroke="#2D8B55"/><text x="425" y="56" text-anchor="middle" fill="#1a5c38" font-size="10">🔒 examiner, new route</text><text x="425" y="82" text-anchor="middle" fill="#6B645E" font-size="8">once, for your licence</text><text x="425" y="102" text-anchor="middle" fill="#1a5c38" font-size="8.5">= TEST</text><text x="425" y="120" text-anchor="middle" fill="#6B645E" font-size="7.5">honest final grade</text><path d="M170 82 L 185 82" stroke="#6B645E" stroke-width="1.4"/><polygon points="185,82 177,78 177,86" fill="#6B645E"/><path d="M335 82 L 350 82" stroke="#6B645E" stroke-width="1.4"/><polygon points="350,82 342,78 342,86" fill="#6B645E"/><text x="260" y="152" text-anchor="middle" fill="#1a5c38" font-size="9">Memorizing the route passes the exam and fails every other road = overfitting.</text><text x="260" y="170" text-anchor="middle" fill="#6B645E" font-size="8.5">The exam-vs-answer-key idea, start to finish.</text></g></svg>
%%%

#### The day, rebuilt one panel at a time
Read the picture below left to right: first only the study pile, then the check pile clicks into place, then the sealed grade pile, and finally the *gap* you watch between study and check. One new idea per panel — the same order you learned them in.

%%% svg
<svg viewBox="0 0 520 208" role="img" aria-label="A four-panel progressive assembly of the day's idea. Panel one: only the training pile, where the model learns. Panel two adds the validation pile beside it, used to check progress during study. Panel three adds the sealed test pile, opened once for the honest grade. Panel four keeps all three piles and draws a widening gap between the training and validation curves, labelled overfitting is the gap you watch. Each panel adds one piece on top of the previous."><g font-family="monospace" font-size="8"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">Rebuild the day: one piece per panel →</text><g><rect x="10" y="26" width="118" height="168" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text x="69" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">① just study</text><rect x="26" y="58" width="86" height="30" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text x="69" y="77" text-anchor="middle" fill="#1F6280">📚 train</text><text x="69" y="112" text-anchor="middle" fill="#6B645E">model LEARNS</text><text x="69" y="126" text-anchor="middle" fill="#6B645E">here only</text></g><g><rect x="134" y="26" width="118" height="168" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text x="193" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">② + check</text><rect x="150" y="58" width="86" height="26" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text x="193" y="75" text-anchor="middle" fill="#1F6280">📚 train</text><rect x="150" y="92" width="86" height="26" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text x="193" y="109" text-anchor="middle" fill="#9A7208">📝 validation</text><text x="193" y="140" text-anchor="middle" fill="#9A7208">check progress,</text><text x="193" y="152" text-anchor="middle" fill="#9A7208">tune settings</text></g><g><rect x="258" y="26" width="118" height="168" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text x="317" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">③ + sealed grade</text><rect x="274" y="54" width="86" height="24" rx="4" fill="#EAF3F7" stroke="#2A7B9B"/><text x="317" y="70" text-anchor="middle" fill="#1F6280">📚 train</text><rect x="274" y="84" width="86" height="24" rx="4" fill="#FDF3D6" stroke="#C99A12"/><text x="317" y="100" text-anchor="middle" fill="#9A7208">📝 validation</text><rect x="274" y="114" width="86" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="317" y="130" text-anchor="middle" fill="#1a5c38">🔒 test</text><text x="317" y="156" text-anchor="middle" fill="#1a5c38">opened ONCE</text><text x="317" y="168" text-anchor="middle" fill="#6B645E">honest grade</text></g><g><rect x="382" y="26" width="128" height="168" rx="6" fill="#FBF7F0" stroke="#E5DFD6"/><text x="446" y="42" text-anchor="middle" fill="#6B645E" font-size="8.5">④ + watch the gap</text><path d="M398 150 C 420 120, 440 96, 500 62" fill="none" stroke="#2A7B9B" stroke-width="2"/><path d="M398 150 C 420 128, 440 128, 500 118" fill="none" stroke="#C99A12" stroke-width="2"/><line x1="500" y1="62" x2="500" y2="118" stroke="#C93B3B" stroke-width="1.2" stroke-dasharray="3,3"/><text x="470" y="76" fill="#1F6280" font-size="7">train ↓</text><text x="466" y="132" fill="#9A7208" font-size="7">val turns up</text><text x="446" y="182" text-anchor="middle" fill="#C93B3B" font-size="7.5">the widening gap = overfitting</text></g></g></svg>
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

@@@ quiz id=quiz tag="Quiz" title="Four quick checks" gotit="answer all first"
%%% quiz
q: Why do we grade a model on a held-out pile instead of on the data it trained on? | a:2 | held-out data trains faster | to save memory | a score on studied data cannot tell learning from memorizing | held-out data is more accurate | fb: The whole point: a model can ace data it memorized. Only unseen data shows whether it truly generalizes.
q: Training loss keeps dropping but validation loss reaches its lowest point and starts rising. What is happening, and where is the best model? | a:1 | the learning rate is too small | overfitting — the best model is at the lowest validation point | the test set leaked | training finished successfully | fb: A widening train-validation gap with validation rising is overfitting. The best model on new data is exactly at the lowest validation point — that is where early stopping stops.
q: Which pile do you open exactly ONCE, at the very end, and never tune against? | a:3 | the training set | the validation set | whichever is largest | the test set | fb: The test set is the sealed final exam — touch it once for an honest grade. Tuning against it (peeking) contaminates it.
q: You scale your features using the mean of the WHOLE dataset, then split into train/val/test. What went wrong, and what is the fix? | a:0 | leakage — split first, then fit the scaler on train only and apply it to val/test | nothing, scaling is always safe | the learning rate is now too big | you must scale twice for balance | fb: Fitting the scaler on everything lets eval data influence training — leakage. Split first, fit on train only, then transform val and test with those training numbers.
%%%

@@@ produce id=produce tag="Produce" title="Catch a model overfitting with your own eyes" gotit="Done"
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

@@@ fin
