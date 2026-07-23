---
quest_id: wf7-d04-classifier
mode: concept
donor: v9-base.donor
page_title: "Module 5b · Day 4 — The ViT Classifier"
module_label: "Module 05b · Represent · Day 4"
title: "The ViT Classifier"
subtitle: "Reading the Answer From the [CLS] Token"
brand_sub: "Foundations · M5b Day 4"
spine: "verdict"
nav_prev_href: "../day-03-transformer-for-vision/lesson.html"
nav_prev_label: "Transformer for Vision"
nav_next_href: "../review.html"
nav_next_label: "Review Gate"
fin_title: "Module 5b · Day 4 complete! 🏆"
fin_body: "You did it — you've finished <b>The ViT Classifier</b> and all of Module 5b! The special [CLS] token rides through every block gathering evidence from all the tiles, and a small head reads its final vector to deliver a <b>verdict</b> — the predicted class. You've now built a Vision Transformer end to end: image → tiles → embeddings → transformer → answer.<br>Next up: the <b>Module 5b Review Gate</b>."
notebook_yardstick: null
---

@@@ hero
@lede Think about a courtroom at the very end of a trial. The jury has listened to every witness, all week long. Then the foreperson stands up and says one word — "guilty" or "not guilty." That one word is the **verdict**: a whole week of listening, boiled down to a single answer. Today we build the exact same moment for a Vision Transformer. All those blocks you stacked yesterday have let the tiles listen to each other. Now we need to stand one of them up and say the single word — "cat" — and this last step is the difference between a model that *thinks* about a picture and one that actually *answers* the question "what is this?" It's the same final readout used inside the language models you've heard of, like the [CLS]-style summary token that BERT made famous.
@goal Together we'll build the whole answer-reading path, end to end. You'll meet the special leader token that spends the whole trip gathering evidence, watch its final vector get tidied and turned into one score per class, see those scores become clean percentages that pick a winner, and learn how the model is *taught* to give the right verdict. Then we'll meet — as friendly puzzles — the ways this readout can go wrong, and the exact, named fix waiting for each. Every new word gets explained the first moment it appears.

@@@ concept id=c1 tag="From tiles to one answer" title="The whole path: a sequence of tiles becomes one label" gotit="Got the big picture"
Let's start by looking at the whole journey from a bird's-eye view, so you always know where you are. Yesterday a picture became a line of tile-passports, and the Transformer let every tile listen to every other tile. But at the end you don't want a *line* of things — you want a **verdict**: one word for the whole picture. The job of today is the [[classification head||The small final piece of a Vision Transformer that turns the model's summary of the picture into one predicted label, like "cat" or "dog".]] — the small last piece that takes everything the model learned and delivers that single answer.

Think about a **school science fair**. All day, judges walk around, look at every project, and take notes. But at the end, the crowd doesn't want a stack of notes — they want the **one name** read out: "the winner is…". The classification head is that final announcement. All the looking already happened inside the Transformer; the head's only job is to turn the gathered notes into one clear verdict. The analogy breaks down in a nice way: a fair has one winner, but our head gives a *score to every possible answer* at once, and the top score is the one it reads out loud.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="The whole path shown left to right as five stages. Stage one: a picture. Stage two: a line of tile tokens with a special CLS token in front. Stage three: the Transformer stack. Stage four: only the CLS token's final vector is pulled out. Stage five: the classification head reads it and announces one label, cat. Each stage is a box connected by an arrow to the next.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The whole path: picture → tiles → transformer → read [CLS] → one verdict</text>
<g transform="translate(20,58)"><rect x="0" y="0" width="60" height="60" rx="6" fill="#FBF7EF" stroke="#C99A12"/><text x="30" y="34" fill="#8A6D3B" font-size="16">🖼</text><text x="30" y="78" fill="#6B645E" font-size="8">picture</text></g>
<text x="92" y="92" fill="#9A5A12" font-size="14">→</text>
<g transform="translate(110,66)"><rect x="0" y="0" width="26" height="44" rx="3" fill="#EFEAF7" stroke="#5E5191" stroke-width="1.6"/><text x="13" y="26" fill="#5E5191" font-size="7">[CLS]</text><rect x="30" y="0" width="18" height="44" fill="#EAF5EE" stroke="#2D8B55"/><rect x="50" y="0" width="18" height="44" fill="#EAF5EE" stroke="#2D8B55"/><rect x="70" y="0" width="18" height="44" fill="#EAF5EE" stroke="#2D8B55"/><text x="44" y="58" fill="#1a5c38" font-size="8">tiles + leader</text></g>
<text x="210" y="92" fill="#9A5A12" font-size="14">→</text>
<g transform="translate(228,60)"><rect x="0" y="0" width="70" height="56" rx="6" fill="#EAF0FA" stroke="#5E5191"/><text x="35" y="26" fill="#5E5191" font-size="9">Transformer</text><text x="35" y="40" fill="#7C6DAA" font-size="8">(the stack)</text></g>
<text x="308" y="92" fill="#9A5A12" font-size="14">→</text>
<g transform="translate(326,66)"><rect x="0" y="0" width="26" height="44" rx="3" fill="#EFEAF7" stroke="#5E5191" stroke-width="2"/><text x="13" y="26" fill="#5E5191" font-size="7">[CLS]</text><text x="13" y="58" fill="#5E5191" font-size="8">pull out only this</text></g>
<text x="368" y="92" fill="#9A5A12" font-size="14">→</text>
<g transform="translate(386,64)"><rect x="0" y="0" width="60" height="48" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="30" y="22" fill="#C93B3B" font-size="8">classify</text><text x="30" y="36" fill="#8a3b3b" font-size="8">head</text></g>
<text x="458" y="92" fill="#276b45" font-size="12">"cat"🐱</text>
<text x="260" y="150" fill="#6B645E" font-size="9">the transformer already did the looking — the head just reads out the answer</text>
<text x="260" y="180" fill="#9A938A" font-size="9">everything today lives in the last two boxes: pull out [CLS], then classify it</text>
</g></svg>
%%%

**What the science-fair picture gets right:** all the careful looking happens first, and the final step is just a short announcement of the winner. **Where it breaks down:** the head doesn't only name one winner — it quietly scores *every* class, and the highest score is what gets announced.

#### Where today sits in the whole build
You've actually built almost all of this already. The tiles (Day 1), their embeddings and position stamps (Day 2), and the Transformer stack that lets them listen to each other (Day 3) are done. Today is the *last two boxes* of that picture: **pull out one special token, then read a verdict off it.** That's the entire scope. By the end you'll be able to trace a picture all the way to a word with your finger — a complete Vision Transformer.

@@@ concept id=c2 tag="The [CLS] leader token" title="The extra token whose whole job is to gather" gotit="Got the [CLS] token"
So which token do we read the verdict off? Not any of the tile tokens — a single tile only saw one little corner of the picture. Instead we add a brand-new token that belongs to *no* tile, pin it to the front of the line, and give it one job: **listen to everybody**. This is the [[class token||An extra learnable token added to the front of the tile sequence. It belongs to no image region; its only job is to gather a summary of the whole picture. Written [CLS], short for "classification".]], written `[CLS]` (short for "classification").

Think of a **class monitor** whom the teacher appoints on the first day. The monitor isn't assigned to one desk or one subject — their job is to walk around, listen to the whole class, and report back a summary. At the start of the year they know nothing; by talking to everyone all year, they build up a picture of how the class is doing. Our `[CLS]` token is exactly that monitor: it starts as a blank, [[learnable||A set of numbers the model adjusts during training to get better. The [CLS] token starts as random numbers and is slowly tuned, just like the model's other weights.]] vector — not copied from any tile — and it gathers a summary of the whole picture as it rides through the blocks. The analogy breaks down here: a real monitor has their own personality, but the `[CLS]` token starts totally blank and holds *only* what it collected from the tiles.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A line of tokens. At the front sits the CLS token, drawn with a monitor's badge, colored differently from the rest. Behind it sit the tile tokens numbered one through four. A label points to the CLS token saying: not a tile, added on purpose, starts blank and learnable. A label under the tiles says: one per image region.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="18" fill="#2C2A28" font-size="12">One extra token, pinned to the front — the class monitor</text>
<g transform="translate(40,54)"><rect x="0" y="0" width="52" height="52" rx="6" fill="#EFEAF7" stroke="#5E5191" stroke-width="2.5"/><text x="26" y="26" fill="#5E5191" font-size="9">[CLS]</text><text x="26" y="42" fill="#7C6DAA" font-size="14">🎖</text></g>
<g transform="translate(120,54)"><rect x="0" y="0" width="52" height="52" fill="#EAF5EE" stroke="#2D8B55"/><text x="26" y="31" fill="#1a5c38">tile 1</text></g>
<g transform="translate(180,54)"><rect x="0" y="0" width="52" height="52" fill="#EAF5EE" stroke="#2D8B55"/><text x="26" y="31" fill="#1a5c38">tile 2</text></g>
<g transform="translate(240,54)"><rect x="0" y="0" width="52" height="52" fill="#EAF5EE" stroke="#2D8B55"/><text x="26" y="31" fill="#1a5c38">tile 3</text></g>
<g transform="translate(300,54)"><rect x="0" y="0" width="52" height="52" fill="#EAF5EE" stroke="#2D8B55"/><text x="26" y="31" fill="#1a5c38">tile 4</text></g>
<path d="M66 48 Q66 34 120 34 Q200 34 430 40" fill="none" stroke="#5E5191" stroke-dasharray="3,2" stroke-width="0.8"/>
<text x="430" y="34" fill="#5E5191" font-size="8" text-anchor="start">not a tile — added on</text>
<text x="430" y="46" fill="#7C6DAA" font-size="8" text-anchor="start">purpose; starts blank</text>
<text x="430" y="58" fill="#7C6DAA" font-size="8" text-anchor="start">and learnable</text>
<text x="240" y="128" fill="#6B645E" font-size="9">tiles: one token per image region · [CLS]: belongs to no region</text>
</g></svg>
%%%

**What the class-monitor picture gets right:** one appointed member whose whole role is to listen around and carry a summary — not tied to any one desk. **Where it breaks down:** the monitor already knows things; the `[CLS]` token starts blank and its "personality" is *learned* from training, then filled in fresh from the tiles on every new picture.

#### Build-up: watch it fill up, block by block
Here's the satisfying part. The `[CLS]` token starts blank, but remember the "dinner table" from yesterday — in every block, *every* token (including `[CLS]`) gets to look at every other token. So block by block, the `[CLS]` token pulls in a little more of the picture, until by the last block it holds a rich summary of the *whole* image. Watch the bar of "how much of the picture it has gathered" grow as it walks the stack.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A build-up bar chart. The CLS token starts blank at block zero, an empty bar. After block one it has gathered a little, a short filled bar. After block two, more. After block three, more still. By the last block the bar is nearly full, meaning it holds a summary of the whole picture. Arrows show information flowing from all tiles into the CLS bar at each block.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">[CLS] starts blank and fills with picture-summary, block by block</text>
<line x1="50" y1="150" x2="480" y2="150" stroke="#B8AEA2"/>
<text x="80" y="168" fill="#6B645E">start</text><rect x="62" y="146" width="36" height="4" fill="#D9D2C7"/><text x="80" y="140" fill="#9A938A" font-size="8">blank</text>
<text x="170" y="168" fill="#6B645E">after block 1</text><rect x="152" y="118" width="36" height="32" fill="#B7A7DC"/>
<text x="260" y="168" fill="#6B645E">after block 2</text><rect x="242" y="92" width="36" height="58" fill="#8E79C4"/>
<text x="350" y="168" fill="#6B645E">after block 3</text><rect x="332" y="70" width="36" height="80" fill="#6E58AE"/>
<text x="440" y="168" fill="#6B645E">last block</text><rect x="422" y="44" width="36" height="106" fill="#5E5191"/><text x="440" y="38" fill="#5E5191" font-size="8">full summary</text>
<text x="260" y="186" fill="#9A938A" font-size="8">at each block, attention lets [CLS] pull in a little more from every tile</text>
</g></svg>
%%%

#### Where the idea came from: BERT's [CLS] token
This trick wasn't invented for pictures. It came from a famous *text* model called [[BERT||A landmark language model that reads a whole sentence at once. It put a special [CLS] token at the front and read the sentence's answer off that token — the exact idea ViT borrowed.]]. BERT put a `[CLS]` token at the front of a *sentence* and read the sentence's answer — like "is this review positive?" — off that one token. ViT simply borrowed the same idea and pointed it at a picture instead of a sentence. So when you use this token, you're using a bridge straight from the language models you've heard of to the vision models — one more sign that it really is "one recipe" under the surface.

@@@ concept id=c3 tag="Read it, then tidy it" title="Pull out the [CLS] vector — and smooth it before the head" gotit="Got the readout"
The tiles have finished the whole relay. Out the other end comes a full line of tokens again — but we only want one. Here's the move: we **throw away every tile token and keep only the `[CLS]` token's final vector.** That one vector, a list of `D` numbers, is the model's summary of the whole picture. This step has a plain name: the [[CLS readout||Taking the [CLS] token's final-layer output vector as the single, image-level summary and ignoring all the tile tokens for the final answer.]] — the verdict is read from `[CLS]` alone.

Think about a **long group trip** where one friend keeps the shared travel diary. At the end of the trip, everyone has photos on their own phones — but to remember the trip, you just read the *one* diary. You don't scroll through five people's camera rolls. Reading the `[CLS]` vector is reading that one diary: the tile tokens still exist, but for the final answer we only open the summary that `[CLS]` kept. The analogy breaks down gently: a diary is written in words you can read directly, but the `[CLS]` vector is still raw numbers — we need one more small step to turn it into a label.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="After the transformer, a row of output tokens. Only the CLS token at the front is circled and pulled downward with an arrow, labelled keep this one vector. The tile tokens are greyed out and labelled ignored for the final answer. The pulled-out CLS vector is shown as a small column of numbers labelled the picture summary, length D.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Keep only the [CLS] vector; the tile outputs are set aside</text>
<g transform="translate(40,44)"><rect x="0" y="0" width="44" height="40" rx="4" fill="#EFEAF7" stroke="#5E5191" stroke-width="2.5"/><text x="22" y="24" fill="#5E5191" font-size="8">[CLS]</text></g>
<g opacity="0.4"><rect x="100" y="44" width="40" height="40" fill="#EAF5EE" stroke="#2D8B55"/><rect x="146" y="44" width="40" height="40" fill="#EAF5EE" stroke="#2D8B55"/><rect x="192" y="44" width="40" height="40" fill="#EAF5EE" stroke="#2D8B55"/><rect x="238" y="44" width="40" height="40" fill="#EAF5EE" stroke="#2D8B55"/></g>
<text x="189" y="102" fill="#9A938A" font-size="8">tile outputs — ignored for the final answer</text>
<path d="M62 86 Q62 118 120 120" fill="none" stroke="#5E5191" stroke-width="1.6" marker-end="url(#a3a)"/>
<defs><marker id="a3a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#5E5191"/></marker></defs>
<g transform="translate(128,104)"><rect x="0" y="0" width="30" height="52" rx="3" fill="#EFEAF7" stroke="#5E5191"/><text x="15" y="14" fill="#5E5191" font-size="7">0.4</text><text x="15" y="28" fill="#5E5191" font-size="7">-1.2</text><text x="15" y="42" fill="#5E5191" font-size="7">0.9</text></g>
<text x="230" y="134" fill="#5E5191" font-size="9" text-anchor="start">the picture summary — a vector of length D</text>
</g></svg>
%%%

**What the travel-diary picture gets right:** many people gathered material, but you read the trip off the *one* shared summary, not everyone's raw footage. **Where it breaks down:** a diary is already words; the `[CLS]` vector is still raw numbers that need one small conversion into a label.

#### One tidy-up step first: the final LayerNorm
Before we hand that summary vector to the answer-reader, we run it through one gentle cleanup. The numbers in the `[CLS]` vector can come out at all sorts of scales — some big, some tiny — and a wildly-scaled input makes the final step wobbly. So we pass it through a [[LayerNorm||A gentle rescaling step that re-centers and re-scales a vector so its numbers sit in a stable, healthy range. It doesn't change what the vector "means," just its scale.]] (short for "layer normalization"): it re-centers and re-scales the numbers so they sit in a calm, standard range — without changing what the summary *means*.

Think of **plating a dish before serving**. The cooking is done; you just wipe the rim and arrange it neatly so the person tasting it isn't distracted by a messy plate. LayerNorm plates the `[CLS]` vector: same food, tidy presentation, so the next step judges the *content*, not the mess.

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="A before-and-after of LayerNorm. On the left, a bar chart labelled before LayerNorm with bars of wildly different heights, some very tall, some tiny, some negative. An arrow labelled LayerNorm points right. On the right, the same bars but re-centered and re-scaled into a calm, similar-sized range around zero, labelled after LayerNorm, stable scale.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">LayerNorm: same meaning, tidy scale — like plating a dish</text>
<text x="115" y="34" fill="#C93B3B">before: wild scales</text>
<line x1="40" y1="110" x2="200" y2="110" stroke="#B8AEA2"/>
<rect x="52" y="48" width="16" height="62" fill="#C93B3B"/><rect x="76" y="100" width="16" height="10" fill="#C93B3B"/><rect x="100" y="70" width="16" height="40" fill="#C93B3B"/><rect x="124" y="110" width="16" height="26" fill="#C93B3B"/><rect x="148" y="58" width="16" height="52" fill="#C93B3B"/>
<text x="240" y="86" fill="#9A5A12" font-size="12">LayerNorm →</text>
<text x="405" y="34" fill="#276b45">after: calm, standard</text>
<line x1="330" y1="90" x2="480" y2="90" stroke="#B8AEA2"/>
<rect x="340" y="66" width="16" height="24" fill="#2D8B55"/><rect x="362" y="90" width="16" height="18" fill="#2D8B55"/><rect x="384" y="72" width="16" height="18" fill="#2D8B55"/><rect x="406" y="90" width="16" height="22" fill="#2D8B55"/><rect x="428" y="68" width="16" height="22" fill="#2D8B55"/>
<text x="405" y="132" fill="#9A938A" font-size="8">re-centered around zero, similar sizes</text>
</g></svg>
%%%

So the readout path so far is two calm steps: **keep the `[CLS]` vector, then LayerNorm it.** Now that summary is clean and ready for the piece that actually turns numbers into a verdict — which is next.

@@@ concept id=c4 tag="One score per class" title="The linear head: a scorecard for every possible answer" gotit="Got the head"
Now the fun part: turning that clean summary vector into actual answers. The [[linear classification head||A single dense (fully-connected) layer that maps the summary vector to one raw score per class. "Linear" means it just multiplies and adds — no bends. It has one output number per possible label.]] is one small layer that reads the summary and gives **one score for every possible class**. If the model can pick from 1,000 classes, the head hands back 1,000 scores — one per class. The biggest score is the model's favorite answer. Each raw score is called a [[logit||A raw, un-normalized score the head gives to one class. Bigger means "more likely" in the model's opinion, but logits are not yet percentages — that conversion comes next.]] (say it "LOW-jit") — just a fancy word for "raw score before we tidy it into a percentage."

Think of a **panel of judges filling in a scorecard**. Each judge writes a number next to every contestant — a 7 here, a 2 there, a 9 for the favorite. The scorecard doesn't crown a winner yet; it just lists a number per contestant. The linear head is that scorecard: one raw number (a logit) per class. "Linear" means the head does the simplest possible thing — it multiplies the summary by a set of learned weights and adds them up, no fancy bends. The analogy breaks down a little: real judges each write independently, but the head computes all its scores in one quick multiply from the *same* summary vector.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A scorecard picture. On the left, the CLS summary vector as a small column of numbers. An arrow labelled linear head, multiply and add points to the right, into a scorecard listing four classes: cat with a tall bar and score 4.1, dog 2.2, bird 0.8, car 1.4. The cat row is highlighted as the highest score. A note says these raw scores are called logits.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The linear head fills a scorecard: one raw score (logit) per class</text>
<g transform="translate(30,58)"><rect x="0" y="0" width="34" height="70" rx="3" fill="#EFEAF7" stroke="#5E5191"/><text x="17" y="16" fill="#5E5191" font-size="7">0.4</text><text x="17" y="32" fill="#5E5191" font-size="7">-1.2</text><text x="17" y="48" fill="#5E5191" font-size="7">0.9</text><text x="17" y="64" fill="#5E5191" font-size="7">…</text><text x="17" y="84" fill="#7C6DAA" font-size="7">summary</text></g>
<text x="112" y="96" fill="#9A5A12" font-size="9">linear head</text><text x="112" y="108" fill="#9A938A" font-size="8">(multiply + add)</text><path d="M70 92 H150" stroke="#B8AEA2" marker-end="url(#a4a)"/><defs><marker id="a4a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 z" fill="#B8AEA2"/></marker></defs>
<g transform="translate(170,40)"><rect x="0" y="0" width="300" height="110" rx="6" fill="#FBF7EF" stroke="#C99A12"/><text x="150" y="16" fill="#8A6D3B" font-size="9">scorecard (logits)</text>
<text x="20" y="36" text-anchor="start" fill="#276b45" font-size="9">cat</text><rect x="70" y="28" width="170" height="10" fill="#2D8B55"/><text x="255" y="37" fill="#276b45" font-size="9">4.1</text>
<text x="20" y="56" text-anchor="start" fill="#6B645E" font-size="9">dog</text><rect x="70" y="48" width="92" height="10" fill="#B8AEA2"/><text x="255" y="57" fill="#6B645E" font-size="9">2.2</text>
<text x="20" y="76" text-anchor="start" fill="#6B645E" font-size="9">bird</text><rect x="70" y="68" width="34" height="10" fill="#B8AEA2"/><text x="255" y="77" fill="#6B645E" font-size="9">0.8</text>
<text x="20" y="96" text-anchor="start" fill="#6B645E" font-size="9">car</text><rect x="70" y="88" width="58" height="10" fill="#B8AEA2"/><text x="255" y="97" fill="#6B645E" font-size="9">1.4</text>
</g>
<text x="320" y="172" fill="#276b45" font-size="9">highest score wins → "cat"</text>
</g></svg>
%%%

**What the scorecard picture gets right:** every contestant gets a number, and the top number wins — that's exactly one logit per class, and picking the biggest score (called **argmax** — "the argument that gives the max," meaning *which* class has the highest score) is the label. **Where it breaks down:** judges score independently and by feel; the head computes all its logits in one clean multiply-and-add from the same summary vector.

#### Build-up: watch a summary become a scorecard
The head has one setting you'll hear about: its number of output units equals the number of classes. Give it a summary and it fires back that many logits. Predict the shape, then run it and *watch* a length-`D` summary turn into one score per class — and watch **argmax** point at the biggest.

%%% demo id=head label="predict, then run it"
code: summary = np.random.randn(8)          # the [CLS] vector, D=8
      W_head   = np.random.randn(8, 4)      # head: D=8 in, 4 classes out
      logits   = summary @ W_head           # one raw score per class
      print(logits.shape, logits)
      print("argmax =", logits.argmax())    # WHICH class scored highest
out: (4,)  [ 4.12  2.22  0.84  1.35 ]        # 4 logits — one per class
     argmax = 0  →  predicted class: "cat"   # the biggest logit's index
take: <b>The head maps a length-D summary to one logit per class in a single multiply, and argmax names the winner.</b> Here D=8 in, 4 classes out, so the output is length 4. The biggest logit (4.12, index 0) is the model's pick — that "index of the biggest score" is exactly what <b>argmax</b> returns. Change 4 to 1000 and you'd get 1000 logits — the head's output size IS the number of classes.
%%%

#### Optional (skippable): the one line of math
!!! c-info 🧮
<b>Optional (skippable) — the head in one line.</b> The head is a plain linear layer: `logits = LayerNorm(cls) · W + b`. In words: take the tidied summary vector `cls` (length `D`), multiply by a learned weight matrix `W` of shape `D × C` (where `C` = number of classes), add a learned bias `b` (length `C`), and out come `C` logits. That's it — one matrix multiply and an add. No bends, no extra layers. Everything clever already happened inside the Transformer; the head is deliberately the simplest possible reader.
!!!

So now we have a scorecard of raw logits — one per class. But raw scores are hard to read: is a `4.1` "very sure" or "barely sure"? We can't tell yet. The next step turns these raw scores into clean percentages.

@@@ concept id=c5 tag="Scores into percentages" title="Softmax: raw scores become a clean set of odds" gotit="Got softmax"
Raw logits are lumpy and hard to read. We want the verdict spoken in plain odds: "I'm 80% sure it's a cat, 12% dog, and a little bit of everything else." The step that does this is called [[softmax||A step that turns a list of raw scores into positive percentages that add up to 100%. Bigger scores get bigger shares; the biggest score gets the biggest share.]]. It takes the scorecard of logits and turns it into a set of percentages that are all positive and **add up to 100%** (that is, to 1). The class with the biggest logit gets the biggest share — so the model's pick doesn't change, but now we can *read* how confident it is.

Think of **cutting one pizza into slices, one slice per class**. There's exactly one whole pizza — that's your 100%. A class with a big logit gets a big slice; a class with a small logit gets a sliver. The slices always add up to the whole pizza, no more, no less. Softmax cuts the confidence-pizza: the winner gets the biggest slice, and every class gets *some* slice. The analogy breaks down slightly: softmax doesn't cut by simple proportion — it first stretches the scores (bigger scores pull *extra* far ahead) before slicing, so a clear winner can grab a very large slice.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="On the left, four raw logit bars of different heights for cat, dog, bird, car, labelled raw scores, hard to read. An arrow labelled softmax points to the right. On the right, a pizza divided into four slices: a very large cat slice covering most of the circle labelled 80 percent, a small dog slice 12 percent, and two tiny slices for bird 3 percent and car 5 percent, all adding up to 100 percent.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Softmax slices one confidence-pizza — the slices add up to 100%</text>
<text x="90" y="40" fill="#6B645E">raw scores (logits)</text>
<line x1="30" y1="150" x2="170" y2="150" stroke="#B8AEA2"/>
<rect x="42" y="78" width="20" height="72" fill="#2D8B55"/><text x="52" y="164" fill="#276b45">cat</text>
<rect x="72" y="118" width="20" height="32" fill="#B8AEA2"/><text x="82" y="164" fill="#6B645E">dog</text>
<rect x="102" y="138" width="20" height="12" fill="#D9B0B0"/><text x="112" y="164" fill="#6B645E">bird</text>
<rect x="132" y="130" width="20" height="20" fill="#B8AEA2"/><text x="142" y="164" fill="#6B645E">car</text>
<text x="215" y="110" fill="#9A5A12" font-size="11">softmax →</text>
<g transform="translate(385,108)">
<path d="M0 0 L0 -60 A60 60 0 1 1 -57.06 -18.54 Z" fill="#2D8B55"/>
<path d="M0 0 L-57.06 -18.54 A60 60 0 0 1 -28.91 -52.58 Z" fill="#9DB8A9"/>
<path d="M0 0 L-28.91 -52.58 A60 60 0 0 1 -18.54 -57.06 Z" fill="#D9B0B0"/>
<path d="M0 0 L-18.54 -57.06 A60 60 0 0 1 0 -60 Z" fill="#C9BFAF"/>
<text x="14" y="-4" fill="#fff" font-size="12">cat</text><text x="12" y="12" fill="#fff" font-size="10">80%</text>
<text x="-40" y="-40" fill="#6B645E" font-size="8">dog 12%</text>
</g>
<text x="385" y="192" fill="#276b45">cat 80% · dog 12% · bird 3% · car 5%</text>
<text x="385" y="205" fill="#9A938A" font-size="8">one whole pizza = 100%</text>
</g></svg>
%%%

**What the pizza picture gets right:** there's exactly one whole pizza (100%), and bigger-scoring classes get bigger slices — so it reads as clean odds. **Where it breaks down:** softmax stretches the scores before slicing, so a strong winner can grab an outsized slice compared to a plain proportional cut.

#### Build-up: watch logits become percentages
Let's turn the exact scorecard from the last concept into odds. Predict which class wins (the biggest logit did — `cat`), then run it and *watch* the four logits become four percentages that add up to 1.00.

%%% demo id=softmax label="predict, then run it"
code: logits = np.array([4.12, 2.22, 0.84, 1.35])    # cat, dog, bird, car
      probs  = np.exp(logits) / np.exp(logits).sum()  # softmax
      print(probs.round(2), " sum =", probs.sum().round(2))
out: [0.80  0.12  0.03  0.05]   sum = 1.0    # cat 80%, dog 12%, bird 3%, car 5%
     argmax = 0  →  predicted class: "cat", ~80% sure
take: <b>Softmax turns raw logits into percentages that sum to 1.</b> The biggest logit (cat, 4.12) becomes the biggest share, and every class gets some. The model's pick (argmax) is unchanged — but now the number is readable: it says "cat, ~80% sure." Notice the shares always total 100%.
%%%

The verdict path is now complete for *using* a trained model: **[CLS] → LayerNorm → linear head → logits → softmax → the predicted class and its confidence.** But we skipped one big question — how does the model *learn* to put the big score on the right class in the first place? That's next, and it's short.

@@@ concept id=c6 tag="Teaching the verdict" title="Cross-entropy: the nudge that says 'that's the right one'" gotit="Got the training signal"
So far we've *read* the answer, but a fresh model would just guess. How does it learn to put the big score on the *correct* class? During training every picture comes with a true label — "this really is a cat." We compare the model's percentages to that truth and give it a gentle push to raise the percentage on the correct class. The measure of "how wrong were you?" is called [[cross-entropy loss||A training signal that is small when the model gives a high percentage to the correct class, and large when it gives the correct class a low percentage. Training nudges the model to make this number small.]].

Think of a **weather forecaster** who says "10% chance of rain" — and then it pours. Everyone groans: they were confidently wrong, so they "lose" a lot of points. If instead they'd said "90% chance of rain," they'd lose almost nothing. Cross-entropy scores the model like that forecaster: if it gave the true class a *high* percentage, the loss is tiny; if it gave the true class a *low* percentage, the loss is big. Training just keeps nudging the model to lower that loss — to be confident *and right*. The analogy breaks down a little: a forecaster only bets on one event, but cross-entropy only ever looks at the percentage the model gave the *one true class* for this picture.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A weather forecaster picture. On the left, a forecaster confidently says 10 percent rain but it pours, with a big penalty arrow labelled high loss, confidently wrong. On the right, a forecaster says 90 percent rain and it pours, with a tiny penalty arrow labelled low loss, confidently right. Below: cross-entropy only looks at the percentage given to the true class.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Cross-entropy = the forecaster's penalty for the TRUE outcome</text>
<g transform="translate(30,34)"><rect x="0" y="0" width="210" height="100" rx="8" fill="#FDECEC" stroke="#C93B3B" stroke-dasharray="4,3"/><text x="105" y="20" fill="#C93B3B">said "10% cat" — it WAS a cat</text><text x="105" y="46" fill="#8a3b3b" font-size="16">☔</text><text x="105" y="72" fill="#C93B3B" font-size="10">big penalty ↑</text><text x="105" y="88" fill="#9A938A" font-size="8">confidently wrong = high loss</text></g>
<g transform="translate(280,34)"><rect x="0" y="0" width="210" height="100" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="4,3"/><text x="105" y="20" fill="#276b45">said "90% cat" — it WAS a cat</text><text x="105" y="46" fill="#1a5c38" font-size="16">😊</text><text x="105" y="72" fill="#276b45" font-size="10">tiny penalty ↓</text><text x="105" y="88" fill="#9A938A" font-size="8">confidently right = low loss</text></g>
<text x="260" y="160" fill="#6B645E" font-size="9">training only asks: what % did you give the TRUE class? higher → lower loss</text>
<text x="260" y="176" fill="#9A938A" font-size="8">the push flows back and adjusts the head AND the whole transformer</text>
</g></svg>
%%%

**What the forecaster picture gets right:** being confidently wrong hurts a lot, being confidently right barely hurts — so the model is pushed toward confident, correct answers. **Where it breaks down:** cross-entropy doesn't reward being right about the *other* classes; it only cares about the percentage placed on the single true class.

#### Build-up: watch the penalty shrink as the model gets it right
Predict which case hurts more — giving the true class 10% or 90%? Then run it and *watch* the loss number fall as the model puts more of its confidence on the correct class.

%%% demo id=ce label="predict, then run it"
code: import numpy as np
      for p_true in [0.10, 0.50, 0.90]:      # % the model gave the TRUE class
          loss = -np.log(p_true)             # cross-entropy for that picture
          print(f"gave true class {p_true:.2f}  →  loss = {loss:.2f}")
out: gave true class 0.10  →  loss = 2.30    # confidently wrong: big
     gave true class 0.50  →  loss = 0.69    # unsure: medium
     gave true class 0.90  →  loss = 0.11    # confidently right: tiny
take: <b>Loss falls fast as the true-class percentage rises.</b> 10% → 2.30, but 90% → 0.11. Training keeps nudging the model to raise the % on the correct class, which pulls this number down. That push travels back through the head AND the whole transformer, so everything learns together.
%%%

You now have the *complete* verdict machine: read `[CLS]`, tidy it, score every class, soften into odds, and — during training — nudge those odds toward the truth with cross-entropy. That's a working ViT classifier. Now for the fun part: the honest puzzles about how this readout can go wrong, and the named fix for each.

@@@ concept id=c7 tag="Puzzle: one token, big job" title="The [CLS] bottleneck — and the GAP fix" gotit="Got GAP"
Here's the first honest puzzle, and it comes straight from a choice we made. We asked *one* token — `[CLS]` — to summarize the *whole* picture, and we throw away all the tile outputs for the final answer. But what if one little vector isn't enough to hold everything important? Then `[CLS]` becomes a [[bottleneck||A single narrow point that everything must squeeze through. If the [CLS] token can't hold all the useful information, some of it gets lost before the verdict.]]: useful information the tiles found never reaches the head, because it couldn't all fit through that one token.

Think about **asking a single delegate to represent a huge meeting**. If the meeting was small, one delegate speaks for everyone fine. But if it was a giant meeting with many strong opinions, one person carrying the whole summary will inevitably drop some of it. The fix is obvious once you see it: instead of trusting one delegate, **take a quick poll of the whole room and average the answers.** That averaging is exactly the alternative readout, and it has a name: [[global average pooling||A readout that skips the [CLS] token entirely and instead averages all the tile tokens' final vectors into one summary vector. Written GAP for short.]] (GAP for short). The analogy breaks down gently: a real poll counts discrete votes, but GAP just adds up the tile vectors and divides — it needs no special extra token at all.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Two readout styles side by side. On the left, CLS readout: many tile tokens squeeze into one narrow CLS token, drawn as a funnel with a warning that some info can't fit, labelled bottleneck. On the right, GAP readout: all tile tokens flow into an average box that combines them equally into one summary, labelled every tile contributes.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Two ways to read out: squeeze through [CLS], or average every tile (GAP)</text>
<g transform="translate(20,36)"><rect x="0" y="0" width="230" height="150" rx="8" fill="#FDECEC" stroke="#C93B3B" stroke-dasharray="4,3"/><text x="115" y="20" fill="#C93B3B">[CLS] readout — one narrow token</text>
<rect x="24" y="36" width="24" height="16" fill="#EAF5EE" stroke="#2D8B55"/><rect x="24" y="58" width="24" height="16" fill="#EAF5EE" stroke="#2D8B55"/><rect x="24" y="80" width="24" height="16" fill="#EAF5EE" stroke="#2D8B55"/><rect x="24" y="102" width="24" height="16" fill="#EAF5EE" stroke="#2D8B55"/>
<path d="M52 44 L150 82 M52 66 L150 86 M52 88 L150 90 M52 110 L150 94" stroke="#C93B3B" stroke-width="0.7"/>
<rect x="150" y="78" width="30" height="24" rx="3" fill="#EFEAF7" stroke="#5E5191" stroke-width="2"/><text x="165" y="93" fill="#5E5191" font-size="7">[CLS]</text>
<text x="115" y="132" fill="#8a3b3b" font-size="8">everything squeezes through one token</text><text x="115" y="144" fill="#C93B3B" font-size="8">→ can bottleneck</text></g>
<g transform="translate(270,36)"><rect x="0" y="0" width="230" height="150" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="4,3"/><text x="115" y="20" fill="#276b45">GAP readout — average every tile</text>
<rect x="24" y="36" width="24" height="16" fill="#EAF5EE" stroke="#2D8B55"/><rect x="24" y="58" width="24" height="16" fill="#EAF5EE" stroke="#2D8B55"/><rect x="24" y="80" width="24" height="16" fill="#EAF5EE" stroke="#2D8B55"/><rect x="24" y="102" width="24" height="16" fill="#EAF5EE" stroke="#2D8B55"/>
<path d="M52 44 L150 80 M52 66 L150 84 M52 88 L150 88 M52 110 L150 92" stroke="#2D8B55" stroke-width="0.7"/>
<rect x="150" y="72" width="34" height="34" rx="3" fill="#DDF0E5" stroke="#2D8B55" stroke-width="1.6"/><text x="167" y="92" fill="#1a5c38" font-size="8">avg</text>
<text x="115" y="132" fill="#1a5c38" font-size="8">every tile contributes equally</text><text x="115" y="144" fill="#276b45" font-size="8">→ no single bottleneck</text></g>
</g></svg>
%%%

**What the meeting-delegate picture gets right:** trusting one summarizer can lose detail when there's a lot to carry; polling the whole room keeps more of it. **Where it breaks down:** a real poll counts votes, but GAP just *averages* the tile vectors equally — simple, fast, and it needs no special extra token at all.

#### Build-up: watch the GAP summary form from the tiles
`[CLS]` is *one* delegate carrying a summary. GAP builds its summary a different way: line up **every** tile's final vector, add them together position-by-position, then divide by how many there are. Predict the averaged value of the first slot, then run it and *watch* four tile vectors collapse into one pooled summary — no special token needed.

%%% demo id=gap label="predict, then run it"
code: tiles = np.array([[ 2.,  0.,  4.],    # tile 1's final vector
                        [ 0.,  2.,  2.],    # tile 2
                        [ 4.,  4.,  0.],    # tile 3
                        [ 2.,  2.,  2.]])   # tile 4   (4 tiles, each length 3)
      gap = tiles.mean(axis=0)             # average DOWN the tiles → one summary
      print("pooled GAP summary:", gap, " shape:", gap.shape)
out: pooled GAP summary: [2. 2. 2.]   shape: (3,)
     # slot 0: (2+0+4+2)/4 = 2 · slot 1: (0+2+4+2)/4 = 2 · slot 2: (4+2+0+2)/4 = 2
take: <b>GAP = average all tile vectors into one summary — no [CLS] needed.</b> Four length-3 tiles become one length-3 summary by averaging each slot. Every tile contributes equally, so nothing has to squeeze through a single token. Feed this summary to the same LayerNorm → head → softmax and you get a verdict, exactly like the [CLS] path.
%%%

The build-up shows the whole trade-off in one figure: `[CLS]` funnels everything through one learned token (it can *learn how* to summarize, but can strain); GAP pools every tile equally (nothing to learn, no bottleneck). Neither is "correct" — they're a choice. Many strong ViT-style models use GAP for exactly this reason. Keep both in your pocket: **`[CLS]` = a learned summarizer; GAP = average everything.** When the single token feels like it's straining, GAP is the standard escape hatch.

@@@ concept id=c8 tag="More honest puzzles" title="Three more rough edges — and the named fix for each" gotit="Got the limits"
Being honest about limits is what makes an engineer trustworthy — and each rough edge here comes with a satisfying, named fix. Let's meet three as friendly puzzles. Each one gets its own everyday picture, because each is a genuinely different kind of rough edge.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Three panels of rough edges. Panel one: a softmax bar at 99 percent labelled over-confident even when unsure. Panel two: a whole image labelled cat with a red X over a bounding box, meaning it cannot say where. Panel three: a small-data pile where a CNN beats a ViT, labelled from scratch on little data the CNN wins.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Three honest rough edges of the plain ViT classifier</text>
<g transform="translate(20,36)"><text x="75" y="6" fill="#C93B3B">1 · over-confident</text><rect x="30" y="14" width="90" height="16" fill="#EDE7F5" stroke="#5E5191"/><rect x="30" y="14" width="88" height="16" fill="#5E5191"/><text x="75" y="26" fill="#fff" font-size="8">99%</text><text x="75" y="52" fill="#9A938A" font-size="8">sure even when it shouldn't be</text></g>
<g transform="translate(190,36)"><text x="70" y="6" fill="#C93B3B">2 · whole-image label only</text><rect x="30" y="14" width="80" height="42" fill="#FBF7EF" stroke="#C99A12"/><text x="70" y="40" fill="#276b45" font-size="9">"cat"🐱</text><rect x="48" y="22" width="26" height="24" fill="none" stroke="#9A938A" stroke-dasharray="3,2"/><text x="61" y="16" fill="#C93B3B" font-size="11">✕</text><text x="70" y="70" fill="#9A938A" font-size="8">can't say WHERE</text></g>
<g transform="translate(360,36)"><text x="70" y="6" fill="#C93B3B">3 · data-hungry from scratch</text><rect x="34" y="30" width="34" height="26" fill="#FDECEC" stroke="#C93B3B"/><text x="51" y="47" fill="#C93B3B" font-size="8">CNN</text><text x="78" y="47" fill="#9A5A12">&gt;</text><rect x="90" y="30" width="34" height="26" fill="#EFEAF7" stroke="#5E5191"/><text x="107" y="47" fill="#5E5191" font-size="8">ViT</text><text x="70" y="70" fill="#9A938A" font-size="8">little data → CNN wins</text></g>
</g></svg>
%%%

#### Puzzle 1 · The verdict is too cocky — fix: label smoothing
Think of a **student who fills in every answer in heavy pen, pressing so hard it can't be erased** — even on questions they only half-know. If they're wrong, they look foolish; they can't walk it back. Cross-entropy quietly encourages exactly this: to make the loss tiny, the model keeps pushing the winning logit *higher and higher*, so softmax spits out "99.9% cat" even on a blurry, hard picture. A model whose confidence doesn't match how often it's actually right is called [[miscalibrated||When a model's stated confidence doesn't match reality — e.g. it says "99% sure" but is right far less than 99% of the time.]]. The fix is charming: teach the model to *pencil lightly* instead of pressing in pen. Instead of telling it the truth is "100% cat, 0% everything else," we soften the target a hair — "aim for ~90% cat, and leave a little for the rest." That gentle softening is [[label smoothing||A fix for over-confidence: soften the training target from a hard 100%/0% to something like 90%/spread-the-rest, so cross-entropy stops pushing logits to extremes.]]. It stops cross-entropy from chasing extreme logits, so the model stays a little humble. **Where the pen-vs-pencil picture breaks down:** a student chooses their pressure; here it's the loss target we soften, and the change is the same tiny nudge on *every* answer, not a case-by-case choice.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="A before-and-after of label smoothing. On the left, a hard target: one bar at 100 percent for cat, zero for all others, labelled pushes logits to extremes. An arrow labelled label smoothing points right. On the right, a soft target: cat at 90 percent and small equal slivers for the other classes, labelled stays humble.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Label smoothing softens the target so the verdict stops getting cocky</text>
<text x="110" y="34" fill="#C93B3B">hard target (1 / 0) — "pen"</text>
<line x1="40" y1="120" x2="200" y2="120" stroke="#B8AEA2"/>
<rect x="52" y="46" width="22" height="74" fill="#C93B3B"/><text x="63" y="134" fill="#6B645E">cat</text>
<rect x="84" y="118" width="22" height="2" fill="#C93B3B"/><text x="95" y="134" fill="#6B645E">dog</text>
<rect x="116" y="118" width="22" height="2" fill="#C93B3B"/><text x="127" y="134" fill="#6B645E">bird</text>
<rect x="148" y="118" width="22" height="2" fill="#C93B3B"/><text x="159" y="134" fill="#6B645E">car</text>
<text x="240" y="86" fill="#9A5A12" font-size="10">smooth →</text>
<text x="410" y="34" fill="#276b45">soft target (0.9 / spread) — "pencil"</text>
<line x1="330" y1="120" x2="490" y2="120" stroke="#B8AEA2"/>
<rect x="340" y="54" width="22" height="66" fill="#2D8B55"/><text x="351" y="134" fill="#6B645E">cat</text>
<rect x="372" y="112" width="22" height="8" fill="#9DB8A9"/><text x="383" y="134" fill="#6B645E">dog</text>
<rect x="404" y="112" width="22" height="8" fill="#9DB8A9"/><text x="415" y="134" fill="#6B645E">bird</text>
<rect x="436" y="112" width="22" height="8" fill="#9DB8A9"/><text x="447" y="134" fill="#6B645E">car</text>
</g></svg>
%%%

#### Puzzle 2 · It only labels the whole picture — fix: detection & segmentation heads
Think of the difference between **writing a one-line caption for a photo** and **drawing circles around each thing in it and tracing their outlines with a marker**. Our classifier only writes the caption: it reads *one* verdict off `[CLS]` — "this whole picture is a cat." That single class vector genuinely **cannot tell you where** the cat is or trace its exact shape. The two richer jobs have names. [[object detection||Drawing a box around each object and saying what it is — not just one label for the whole picture.]] draws a box around each object; [[segmentation||Labelling every pixel — outlining exactly which pixels belong to the cat. Much finer than one whole-image label.]] colors in every pixel that belongs to the cat.

Here's the mechanism, in one breath — and it's a friendly one, because you *keep* almost everything you built. The ViT still produces its line of tile tokens exactly as before. The difference is which tokens you read and what small head you bolt on:
- **Classification (today):** read the *one* `[CLS]` summary → a linear head → one label for the whole picture.
- **Detection:** read the *tile* tokens (they still carry *where*-info) → a small head that, per object, predicts a *box* (four numbers: x, y, width, height) *plus* a class.
- **Segmentation:** read the *tile* tokens → a small head that predicts a class label for *every pixel*, producing a colored-in map.

So the same ViT backbone feeds all three; only the little head on top changes — a caption-writer, a box-drawer, or a pixel-painter. **Where the caption-vs-marker picture breaks down:** a marker draws on the image itself, but these heads output *numbers* (box coordinates or per-pixel labels) that a viewer then paints. The full training of detection and segmentation heads is a later topic; today you just need to know *why* the plain classifier can't do it and *what* replaces the head when you want it to.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A build-up of three heads on the same ViT features. Left: the same tile tokens flow up. Then three branches. Branch one, classification: read only CLS, output the caption cat. Branch two, detection: read the tile tokens, output a box around the cat plus its label. Branch three, segmentation: read the tile tokens, output a pixel mask that colors in the cat.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="14" fill="#2C2A28" font-size="12">Same ViT features → swap the little head for a richer task</text>
<text x="60" y="34" fill="#6B645E" font-size="8">the ViT's tile tokens</text>
<rect x="30" y="40" width="14" height="118" fill="#EFEAF7" stroke="#5E5191"/><text x="37" y="52" fill="#5E5191" font-size="6">CLS</text>
<rect x="46" y="40" width="14" height="118" fill="#EAF5EE" stroke="#2D8B55"/><rect x="62" y="40" width="14" height="118" fill="#EAF5EE" stroke="#2D8B55"/><rect x="78" y="40" width="14" height="118" fill="#EAF5EE" stroke="#2D8B55"/>
<path d="M92 60 H150" stroke="#B8AEA2"/><path d="M92 100 H150" stroke="#B8AEA2"/><path d="M92 140 H150" stroke="#B8AEA2"/>
<g transform="translate(152,44)"><rect x="0" y="0" width="150" height="34" rx="4" fill="#FBF7EF" stroke="#C99A12"/><text x="75" y="14" fill="#8A6D3B" font-size="8">classify: read [CLS]</text><text x="75" y="27" fill="#276b45" font-size="9">→ "cat" (one caption)</text></g>
<g transform="translate(152,84)"><rect x="0" y="0" width="150" height="34" rx="4" fill="#EAF0FA" stroke="#5E5191"/><text x="75" y="14" fill="#5E5191" font-size="8">detect: read tiles</text><text x="75" y="27" fill="#5E5191" font-size="9">→ box(x,y,w,h) + class</text></g>
<g transform="translate(152,124)"><rect x="0" y="0" width="150" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="75" y="14" fill="#1a5c38" font-size="8">segment: read tiles</text><text x="75" y="27" fill="#1a5c38" font-size="9">→ label EVERY pixel</text></g>
<g transform="translate(330,48)"><rect x="0" y="0" width="60" height="34" fill="#FBF7EF" stroke="#C99A12"/><text x="30" y="22" font-size="10">🐱 cat</text></g>
<g transform="translate(330,88)"><rect x="0" y="0" width="60" height="34" fill="#FBF7EF" stroke="#C99A12"/><rect x="14" y="6" width="30" height="22" fill="none" stroke="#5E5191" stroke-width="1.4"/><text x="29" y="20" font-size="9">🐱</text></g>
<g transform="translate(330,128)"><rect x="0" y="0" width="60" height="34" fill="#FBF7EF" stroke="#C99A12"/><path d="M12 26 Q20 6 30 14 Q44 4 48 26 Z" fill="#2D8B55" opacity="0.55"/></g>
<text x="440" y="70" fill="#9A938A" font-size="8" text-anchor="middle">one label</text>
<text x="440" y="110" fill="#9A938A" font-size="8" text-anchor="middle">where (box)</text>
<text x="440" y="150" fill="#9A938A" font-size="8" text-anchor="middle">exact pixels</text>
</g></svg>
%%%

#### Puzzle 3 · From scratch on little data, a CNN wins — fix: pretrain, then fine-tune
Think of **two new students on their first day of a jigsaw-puzzle club**. One kid already knows two handy rules of thumb: *pieces that touch usually belong together* (look at neighbors), and *a corner piece is a corner piece no matter where the box is on the table* (the same shape means the same thing anywhere). The second kid knows *no* rules and must figure everything out from scratch, one puzzle at a time. Give them both only a handful of puzzles and the rule-knowing kid wins easily. Those two built-in rules of thumb have a name: [[inductive bias||Helpful assumptions baked into a model before training — like "nearby pixels matter" (locality) and "a pattern means the same anywhere" (translation-equivariance). A CNN has these; a plain ViT barely does.]] — for images the two big ones are **locality** ("nearby pixels matter, so look at small neighborhoods first") and **translation-equivariance** ("a cat-ear pattern means the same whether it's top-left or bottom-right"). A CNN has both baked in, so it needs *fewer* examples to learn. A plain ViT starts with almost none of these rules — it must *learn* even "nearby pixels matter" from the data itself, which is why, trained from scratch on a *small* dataset, it often **loses** to a same-size CNN.

The standard fix is the two-step recipe from yesterday. First, **[[pretraining||Train the model first on a huge general image set so it learns broad visual habits, before adapting it to your task.]]**: train the ViT on a *huge* general image set, so it *learns* those visual habits from sheer volume of examples — the volume stands in for the missing built-in rules. Then, **[[fine-tuning||Take the pretrained model and train it a little more on your smaller task — this is where you (re)train the classification head.]]**: take that pretrained model and train it a little more on your smaller task — this is exactly when you (re)train the classification head. Big pretraining feeds the data-hunger; fine-tuning aims the head at your problem. **Where the puzzle-club picture breaks down:** the rule-knowing kid was *born* knowing the rules, but a pretrained ViT *earned* its habits from millions of examples first — the rules aren't free, they're paid for up front with data.

%%% svg
<svg viewBox="0 0 520 195" role="img" aria-label="A build-up showing why a ViT is data-hungry and how pretraining fixes it. Top row: on little data, the CNN with built-in rules beats the plain ViT. Bottom row: the fix, a huge pretraining dataset feeds the ViT broad visual habits, then a small fine-tune step aims the head at your task, and now the ViT matches or beats the CNN.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<text x="260" y="14" fill="#2C2A28" font-size="12">Why data-hungry — and how pretrain → fine-tune feeds it</text>
<g transform="translate(20,26)"><rect x="0" y="0" width="230" height="70" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-dasharray="4,3"/><text x="115" y="16" fill="#C93B3B">from scratch, little data</text>
<rect x="18" y="26" width="26" height="30" fill="#FDECEC" stroke="#C93B3B"/><text x="31" y="45" fill="#C93B3B" font-size="8">CNN</text><text x="31" y="66" fill="#9A938A" font-size="7">has rules</text>
<text x="60" y="45" fill="#276b45" font-size="12">wins &gt;</text>
<rect x="100" y="26" width="26" height="30" fill="#EFEAF7" stroke="#5E5191"/><text x="113" y="45" fill="#5E5191" font-size="8">ViT</text><text x="113" y="66" fill="#9A938A" font-size="7">no rules yet</text>
<text x="180" y="45" fill="#8a3b3b" font-size="8">few examples</text><text x="180" y="57" fill="#8a3b3b" font-size="8">→ ViT loses</text></g>
<g transform="translate(20,110)"><rect x="0" y="0" width="480" height="76" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="4,3"/><text x="240" y="16" fill="#276b45">the fix: pretrain on a lot, then fine-tune on a little</text>
<g transform="translate(16,26)"><rect x="0" y="0" width="70" height="34" fill="#DDF0E5" stroke="#2D8B55"/><text x="35" y="15" fill="#1a5c38" font-size="8">HUGE set</text><text x="35" y="28" fill="#1a5c38" font-size="8">pretrain</text></g>
<text x="98" y="48" fill="#9A5A12" font-size="11">→</text>
<g transform="translate(118,26)"><rect x="0" y="0" width="70" height="34" fill="#EFEAF7" stroke="#5E5191"/><text x="35" y="15" fill="#5E5191" font-size="8">your small</text><text x="35" y="28" fill="#5E5191" font-size="8">set: fine-tune</text></g>
<text x="200" y="48" fill="#9A5A12" font-size="11">→</text>
<g transform="translate(220,26)"><rect x="0" y="0" width="90" height="34" fill="#FBF7EF" stroke="#C99A12"/><text x="45" y="15" fill="#8A6D3B" font-size="8">(re)train the</text><text x="45" y="28" fill="#8A6D3B" font-size="8">head on task</text></g>
<text x="322" y="48" fill="#9A5A12" font-size="11">→</text>
<text x="410" y="44" fill="#276b45" font-size="10">ViT now matches/beats CNN</text>
<text x="410" y="58" fill="#9A938A" font-size="7">volume replaced the missing rules</text></g>
</g></svg>
%%%

Here's the whole map — each rough edge on the left, its named fix on the right:

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Four rows mapping each rough edge to its fix. Row one: over-confident softmax, fixed by label smoothing. Row two: the CLS token can bottleneck, fixed by a GAP head. Row three: whole-image label only, fixed by detection or segmentation heads. Row four: data-hungry from scratch, fixed by large-scale pretraining then fine-tuning.">
<g font-family="monospace" font-size="9">
<text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Every rough edge → its named fix</text>
<text x="115" y="34" text-anchor="middle" fill="#C93B3B">the rough edge</text><text x="395" y="34" text-anchor="middle" fill="#276b45">the fix</text>
<g transform="translate(20,42)"><rect x="0" y="0" width="200" height="28" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="18" text-anchor="middle" fill="#8a3b3b">over-confident softmax</text></g>
<text x="230" y="60" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(256,42)"><rect x="0" y="0" width="244" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="122" y="18" text-anchor="middle" fill="#1a5c38">label smoothing</text></g>
<g transform="translate(20,80)"><rect x="0" y="0" width="200" height="28" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="18" text-anchor="middle" fill="#8a3b3b">one [CLS] token can bottleneck</text></g>
<text x="230" y="98" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(256,80)"><rect x="0" y="0" width="244" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="122" y="18" text-anchor="middle" fill="#1a5c38">global average pooling (GAP) head</text></g>
<g transform="translate(20,118)"><rect x="0" y="0" width="200" height="28" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="18" text-anchor="middle" fill="#8a3b3b">whole-image label only</text></g>
<text x="230" y="136" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(256,118)"><rect x="0" y="0" width="244" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="122" y="18" text-anchor="middle" fill="#1a5c38">detection / segmentation heads</text></g>
<g transform="translate(20,156)"><rect x="0" y="0" width="200" height="28" rx="4" fill="#FDECEC" stroke="#C93B3B"/><text x="100" y="18" text-anchor="middle" fill="#8a3b3b">data-hungry from scratch</text></g>
<text x="230" y="174" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(256,156)"><rect x="0" y="0" width="244" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="122" y="18" text-anchor="middle" fill="#1a5c38">large-scale pretrain, then fine-tune</text></g>
</g></svg>
%%%

These aren't flaws to fear — they're your map of what's next. You can now say exactly what a ViT classifier does *and*, just as honestly, what it doesn't — that honesty is a real staff-level habit.

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — and with today you've finished a *whole* Vision Transformer, from raw picture to spoken **verdict**. Here's one clean way to hold the day: picture a courtroom. The jury (all the tiles) listened for the whole trial. The foreperson (`[CLS]`) gathered everyone's evidence. Then a short, calm reading turns that into one word — the verdict. **Where the courtroom picture breaks down:** a verdict is final, but during *training* the model hears "you were wrong/right" (cross-entropy) after every case and slowly gets better at the reading. Read the steps, then keep the cheat-sheets as your map.

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="A courtroom-style summary card listing today's readout steps in order. Step 1: keep only the CLS token's final vector. Step 2: LayerNorm to tidy it. Step 3: the linear head gives one logit per class. Step 4: softmax turns logits into percentages summing to 100. Step 5: the top percentage is the verdict; cross-entropy trained it. Below, notes on the GAP alternative and the honest limits.">
<g font-family="monospace" font-size="9" text-anchor="middle">
<rect x="60" y="26" width="400" height="170" rx="10" fill="#FBF7EF" stroke="#C99A12" stroke-width="1.5"/>
<text x="260" y="44" fill="#8A6D3B" font-size="12">⚖️ Today's readout — [CLS] in, one verdict out</text>
<text x="82" y="66" text-anchor="start" fill="#5E5191">1 · keep ONLY the [CLS] token's final vector (the summary)</text>
<text x="82" y="86" text-anchor="start" fill="#5E5191">2 · LayerNorm — tidy the scale before the head</text>
<text x="82" y="106" text-anchor="start" fill="#5E5191">3 · linear head → one raw score (logit) per class</text>
<text x="82" y="126" text-anchor="start" fill="#5E5191">4 · softmax → percentages that add up to 100%</text>
<text x="82" y="146" text-anchor="start" fill="#5E5191">5 · top % is the verdict 🐱 · cross-entropy taught it</text>
<line x1="82" y1="158" x2="438" y2="158" stroke="#E5DFD6"/>
<text x="82" y="176" text-anchor="start" fill="#276b45" font-size="8">alt readout: GAP averages all tiles instead of using [CLS]</text>
<text x="82" y="188" text-anchor="start" fill="#9A938A" font-size="8">limits: over-confident (→ smoothing) · label-only (→ detect/segment) · data-hungry (→ pretrain+fine-tune)</text>
</g></svg>
%%%

The steps of the readout, in order:
- **Step 1 · Keep the `[CLS]` vector.** After the Transformer, throw away the tile outputs and keep the one leader token's final vector — the whole-image summary. (Or use **GAP**: average all tile vectors instead.)
- **Step 2 · LayerNorm.** Tidy the summary's scale so the head sees a stable input — same meaning, calmer numbers.
- **Step 3 · Linear head.** One dense layer maps the summary to **one logit (raw score) per class** — its output size *is* the number of classes; **argmax** names the biggest.
- **Step 4 · Softmax.** Turn the logits into **percentages that add up to 100%** — now you can read confidence, and the top one is the pick.
- **Step 5 · Cross-entropy (training only).** Nudge the percentage on the *true* class up; the push flows back through the head and the whole Transformer.
- **The ancestors:** the `[CLS]` idea came from **BERT** (a text model), and this readout replaced the CNN's **global-average-pooling + fully-connected head**.
- **The honest bits:** over-confident softmax (→ **label smoothing**), one-token bottleneck (→ **GAP**), whole-image label only (→ **detection/segmentation heads**, read from the tile tokens), and data-hunger from scratch (→ **large-scale pretrain, then fine-tune**; a CNN's built-in locality + translation-equivariance is what a plain ViT lacks).

#### Cheat-sheet · the readout at a glance
%%% table
:: Step :: What it is :: One-line reason
Keep [CLS] :: take only the leader token's final vector :: it gathered a summary of the whole picture
LayerNorm :: gently re-scale the summary vector :: give the head a stable-scale input
Linear head :: one dense layer, D in → C logits out :: one raw score per class; output size = #classes
Softmax :: logits → percentages summing to 100% :: readable odds; the top one (argmax) is the verdict
Cross-entropy :: penalty for the % on the true class :: small when right & confident; trains everything
GAP (alternative) :: average all tile vectors, skip [CLS] :: no single-token bottleneck; needs no special token
%%%

#### Cheat-sheet · the words you met today
%%% jargon
classification head | the small final piece that turns the picture-summary into one predicted label
class token / [CLS] | an extra learnable token added to the front; its job is to gather a whole-image summary
CLS readout | using only the [CLS] token's final vector as the image summary for the answer
LayerNorm | a gentle re-scaling that keeps the summary vector's numbers in a stable range
linear head | one dense layer mapping the summary to one raw score per class
logit | a raw, un-normalized class score from the head (not yet a percentage)
argmax | the index of the biggest score — i.e. which class the model picks
softmax | turns logits into positive percentages that add up to 100%
cross-entropy | the training penalty for the percentage given to the TRUE class
global average pooling (GAP) | average all tile vectors into a summary instead of using [CLS]
label smoothing | soften the training target so softmax stops getting over-confident
detection / segmentation head | swap the head to read the tile tokens → boxes (detection) or per-pixel labels (segmentation)
pretrain then fine-tune | train on a huge set first, then specialize — feeds a plain ViT's data-hunger
inductive bias | helpful image rules baked into a CNN (locality, translation-equivariance) that a plain ViT lacks
BERT | the text model whose [CLS]-token readout idea ViT borrowed
%%%

That's the whole day — and the whole module. You can now trace a picture end to end: **tiles → embeddings → transformer → keep `[CLS]` → LayerNorm → head → softmax → a named verdict.** Take the victory lap: you built a Vision Transformer.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does a ViT read its final class answer off, by default? | a:2 | The average of the raw pixels | The last tile token only | The [CLS] token's final vector — a learnable leader token that gathered a whole-image summary | The biggest tile in the picture | fb: The [CLS] token is an extra learnable token pinned to the front. Over the blocks it gathers a summary of the whole picture, and we read the verdict off its final vector — the tile outputs are set aside for the answer.
q: What is a logit, and what does softmax do to a whole set of them? | a:1 | A trained tile; softmax deletes the small ones | A raw, un-normalized class score; softmax turns the set of logits into percentages that add up to 100% | A kind of image filter; softmax sharpens it | The number of classes; softmax counts them | fb: The linear head outputs one raw score (logit) per class. Softmax turns those raw scores into positive percentages that sum to 1, so the biggest becomes a readable confidence — and its class is the pick (argmax).
q: The single [CLS] token can become an information bottleneck. What's the standard alternative readout? | a:2 | Use a bigger image | Delete the [CLS] token and stop classifying | Global average pooling (GAP) — average ALL the tile vectors into the summary instead of using [CLS] | Add more softmax layers | fb: If one token can't hold everything useful, GAP sidesteps it: average every tile's final vector into one summary. It needs no special token and removes the single-token bottleneck — many strong ViT models use it.
q: Trained from scratch on a SMALL dataset, a plain ViT classifier often loses to a CNN. Why, and what's the fix? | a:1 | ViTs are just worse; there is no fix | A plain ViT lacks a CNN's built-in image rules (inductive bias — locality + translation-equivariance), so it must learn everything from data — the fix is large-scale pretraining, then fine-tuning | The images are the wrong color; convert to grayscale | Softmax is too slow on small data | fb: A CNN bakes in locality + translation-equivariance, which help when examples are scarce. A plain ViT learns structure from data, so it's data-hungry. Remedy: pretrain on a huge general set, then fine-tune the head on your task.
%%%

@@@ produce id=produce tag="Produce" title="Read a verdict off [CLS] with your own hands" gotit="Done"
Time to watch the verdict get read with your own eyes. You'll take a tiny fake `[CLS]` summary vector, tidy it, push it through a linear head to get one logit per class, soften those into percentages with softmax, and read out the winner — printing the **shape and values at every step** so you can watch a summary become a decision. **Predict first:** if the `[CLS]` summary has length `D = 8` and there are `C = 4` classes, what shape does the head's output (the logits) have, and what should the four softmax percentages add up to? Then run it and **watch** the numbers confirm it — and **notice** that the percentages always total `1.0`. Pick one path.

#### Option A · write it yourself
Create `sessions/m05b-vision-transformer/day-04-vit-classifier/experiment.py`. Make a fake `[CLS]` summary `cls = np.random.randn(8)` (length `D = 8`) and print its shape. Tidy it with a simple LayerNorm-style step (subtract the mean, divide by the standard deviation) and print the result. Make a random head matrix `W = np.random.randn(8, 4)` (so `C = 4` classes) and compute `logits = cls_tidied @ W`; print the logits and their shape `(4,)` — **one score per class**. Apply softmax: `probs = np.exp(logits) / np.exp(logits).sum()`; print `probs`, print `probs.sum()` and **observe** it is `1.0`, and print `np.argmax(probs)` as the predicted class index. Finally, as a one-line note, print what the *true-class* cross-entropy loss would be with `-np.log(probs[true_class])` for some chosen `true_class`, and **notice** the loss is small when the true class already got a big percentage. Run with `python3 sessions/m05b-vision-transformer/day-04-vit-classifier/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 5b Day 4 artifact.

Create sessions/m05b-vision-transformer/day-04-vit-classifier/experiment.py that, with a comment on each step and printing shapes and values at every step:
1. Makes a fake [CLS] summary vector cls = np.random.randn(8) (length D=8) and prints its shape — note this stands in for the [CLS] token's final vector.
2. Tidies it with a simple LayerNorm-style step (subtract the mean, divide by the standard deviation) and prints the tidied vector; comment that this keeps the scale stable for the head.
3. Makes a random classification-head matrix W = np.random.randn(8, 4) so there are C=4 classes, computes logits = cls_tidied @ W, and prints the logits with shape (4,); comment that these are raw scores (logits), one per class.
4. Applies softmax probs = np.exp(logits) / np.exp(logits).sum(), prints probs, prints probs.sum() and asserts it is ~1.0, and prints np.argmax(probs) as the predicted class index; comment that softmax turns logits into percentages that sum to 1 and argmax names the winner.
5. Picks a true_class (say 0) and prints the cross-entropy loss -np.log(probs[true_class]); comment that the loss is small when the true class already has a high percentage and large when it doesn't.
6. Also computes a GAP summary from a fake tiles = np.random.randn(4, 8) with gap = tiles.mean(axis=0), prints its shape (8,), and notes it could replace the [CLS] summary as the head's input — same head, softmax, and loss after that.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (a summary becoming a verdict)
- The head's output (logits) has shape `(4,)` — **one raw score per class**, because the head's output size equals the number of classes.
- After softmax, the four percentages **add up to `1.0`** — that's the "one whole pizza" rule; the biggest slice (argmax) is the predicted class.
- The cross-entropy loss is **small when the true class already got a big percentage** and large when it didn't — that's the nudge training uses.
- **Notice:** the whole-image verdict came from *one* vector (the `[CLS]` summary). Swapping in a GAP head would just average all the tile vectors first — same head, softmax, and loss after that.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m05b-vision-transformer/day-04-vit-classifier/log.md`: (1) in one sentence, why we read the answer off the `[CLS]` token and not the tile tokens; (2) what softmax gives us that raw logits don't; (3) one honest limit of a plain ViT classifier and the name of its fix.

@@@ fin
