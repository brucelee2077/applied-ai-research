---
quest_id: w03-d01-arith
mode: concept
donor: v9-base.donor
page_title: "Week 3 · Day 1 — Transformer Arithmetic: C ≈ 6ND"
module_label: "Transformer Math · Week 3 · Day 1"
title: "Transformer Arithmetic"
subtitle: "C ≈ 6ND — What a Training Run Costs"
brand_sub: "Frontier Lab · Week 3 Day 1"
spine: "handshake"
nav_prev_href: "../../m07-thinking-in-jax/day-06-vit-capstone/lesson.html"
nav_prev_label: "ViT Capstone"
nav_next_href: "../day-02-qkv-matrices/lesson.html"
nav_next_label: "Q, K, V Matrices"
fin_title: "Week 3 · Day 1 complete! 🏆"
fin_body: "You can now price a training run with a pen and nothing else. One <b>handshake</b> = one knob inside the model meeting one token of text. The price is about six steps of arithmetic per handshake. So the whole bill is <b>C ≈ 6 × N × D</b> — six, times knobs, times tokens. Divide that bill by the steps a chip really does each second and you get seconds. Divide the many-chip seconds by 3 600 for hours of waiting. Share the work across many chips and the waiting divides too. You also know the four places where the six is not six. And you know the two questions this arithmetic can never answer.<br>Next up: <b>Q, K, V Matrices</b> — where the knob count N itself comes from."
notebook_yardstick: null
---

@@@ hero
@lede Have you ever seen someone stand at a doorway with a little metal clicker? They press it once for every person who walks in. At the end of the night they cannot describe a single face. They just read one number off the clicker. And that number is enough to say whether the room was busy. Today you become that person — except you click for **arithmetic**. One press for every multiply. One press for every add. It sounds so simple that you may wonder why it takes a whole day to learn. It is also the whole idea. The training cost of a chatbot you have actually used — ChatGPT, for example — comes from one line of multiplication, and by the end of this page you will do that line with a pen while someone else is still opening their laptop. The model does about **six** presses for every **handshake**. A handshake is one knob inside the model meeting one token of its reading. Where the clicker picture breaks down: nobody really sits there pressing anything. We never count the presses one at a time. We find the pattern and multiply, and the whole sum takes a few seconds.
@goal Together we will build the one line every training run is priced with: <b>C ≈ 6 × N × D</b>. First you meet the two numbers you need — knobs and tokens. Then you see why one knob meeting one token costs two steps forward and about four steps back. Then you turn that bill into hours on real chips. Last, you meet the four places where the six is not six. You also meet the two questions this arithmetic can never answer for you. Every new word gets explained the moment it appears.

@@@ concept id=c1 tag="One click of arithmetic" title="A step of arithmetic is one multiply or one add" gotit="Got the click"
Before we can price anything, we need something to count. This is what we count, and it is smaller than you expect: **one multiply, or one add.** That is it. Not a whole sum. Not a line of code. One little operation on two numbers.

The people who build these models have a name for it: a [[FLOP||One step of arithmetic on decimal numbers — one multiply, or one add. Short for "floating-point operation". Say it "flop".]]. Say it out loud: *flop*. It is short for "floating-point operation". "Floating point" just means a number with a decimal point in it, like 3.7 or −0.002. You do not need the long name. You need the picture: a doorway clicker. You press it once per multiply and once per add.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A hand-held doorway clicker used as the picture for one step of arithmetic. The clicker has a plunger button on top labelled one press per operation, a round display window reading 4, and a reset knob on the side. To the right, four small tiles show what was pressed: multiply, multiply, add, add. A caption reads: one press equals one multiply or one add, and that press is called a FLOP.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="17" fill="#2C2A28" font-size="12">One press = one multiply OR one add. The press has a name: a FLOP.</text>
<g transform="translate(40,40)">
  <rect x="0" y="22" width="120" height="112" rx="12" fill="#EDE9E0" stroke="#3A342E" stroke-width="2"/>
  <rect x="44" y="4" width="32" height="22" rx="5" fill="#C93B3B" stroke="#8a3b3b" stroke-width="2"/>
  <circle cx="60" cy="66" r="30" fill="#FDFCFA" stroke="#3A342E" stroke-width="2"/>
  <text x="60" y="73" fill="#2C2A28" font-size="24">4</text>
  <rect x="112" y="72" width="18" height="12" rx="3" fill="#B8AEA2" stroke="#3A342E"/>
  <text x="60" y="122" fill="#6B645E" font-size="8">steps counted</text>
</g>
<text x="100" y="26" fill="#8a3b3b" font-size="9">one press per step</text>
<text x="190" y="112" fill="#6B645E" font-size="8">reset dial</text>
<text x="330" y="46" fill="#9A5A12" font-size="10">what the 4 presses were:</text>
<g transform="translate(232,56)">
  <g fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"><rect x="0" y="0" width="90" height="30" rx="5"/><rect x="100" y="0" width="90" height="30" rx="5"/><rect x="0" y="40" width="90" height="30" rx="5"/><rect x="100" y="40" width="90" height="30" rx="5"/></g>
  <g fill="#8A6D3B" font-size="10"><text x="45" y="20">5 × 3</text><text x="145" y="20">2 × 4</text><text x="45" y="60">15 + 8</text><text x="145" y="60">23 + 1</text></g>
  <g fill="#9A5A12" font-size="7"><text x="45" y="29">multiply</text><text x="145" y="29">multiply</text><text x="45" y="69">add</text><text x="145" y="69">add</text></g>
</g>
<text x="260" y="178" fill="#9A938A" font-size="9">a big model does a great many of these presses — the whole day is about counting them without pressing</text>
</g></svg>
%%%

**What the clicker picture gets right:** every press is the same size. So a total of presses is a fair measure of how much work happened, whatever the work was for. **What the picture gets wrong:** a real chip does not press one at a time. It does thousands of these steps side by side in the same tick. That is exactly why we will later divide our count by a *rate*.

#### Try the count on something tiny
Look at this little sum: `5 × 3 + 2 × 4 + 1`. Do not reach for the answer. Count the *presses* instead. Every multiply is one press. Every add is one press.

%%% steps
step: 5 × 3 → 15
why: one multiply. That is press number 1.
step: 2 × 4 → 8
why: another multiply. Press number 2.
step: 15 + 8 → 23
why: one add. Press number 3.
step: 23 + 1 → 24
why: one more add. Press number 4.
%%%

So that sum cost **4 steps** of arithmetic. The answer, 24, is not what we care about today. The *4* is.

%%% insight
This is why such a tiny idea is worth a whole day. Every single thing a big model does is made of these presses and nothing else. Reading your question. Choosing the next word. Learning from a mistake. So if you can count presses, you can price *anything* it does. You will never press a clicker a hundred billion times, though. You will find the pattern and multiply.
%%%

%%% demo id=c1count label="guess the press count, then reveal it"
predict: how many steps of arithmetic are in 2 × 6 + 3 × 6 + 4 × 6? Count multiplies and adds separately before you look.
code: 2 × 6  →  12        (multiply: 1 step)
code: 3 × 6  →  18        (multiply: 1 step)
code: 4 × 6  →  24        (multiply: 1 step)
code: 12 + 18 →  30       (add: 1 step)
code: 30 + 24 →  54       (add: 1 step)
out: 3 multiplies + 2 adds = 5 steps of arithmetic
take: <b>5 steps.</b> Three multiplies, two adds. Notice the shape of it. Three multiplies needed two adds to join them up. So multiplies and adds arrive in almost equal numbers. Remember that. In the next two units it becomes the reason a certain 2 shows up everywhere.
%%%

You have your unit of work. Next you need to know how many presses a whole model does. The surprise is that only **two** numbers about the model decide it.

@@@ concept id=c2 tag="The two numbers" title="Only two numbers about a model decide the bill" gotit="Got the two numbers"
Imagine you buy a second-hand bicycle. The seller hands you a small card. On the card there are only two lines: *number of gears* and *kilometres ridden*. Two lines only — and yet you can already guess a great deal about that bicycle.

A model comes with a card like that too. It is called a **model card**. To price a training run you only need two lines off it. Line one: how many [[parameter||One of the numbers inside the model that it is allowed to change — a knob it can turn. In m02 these were called weights. The count of them is written N.]] knobs the model has, written **N**. Line two: how many [[token||One chunk of text — often a whole word, sometimes a piece of one. A model reads and writes tokens, never letters and never words exactly.]] the model read while training, written **D**. That is the entire shopping list. Everything else on the card is for another day.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A model card drawn like a second-hand bicycle card. The card has two big filled-in lines: knobs, written N, equals 7 billion; and tokens read, written D, equals 1.4 trillion. Below them two greyed-out crossed-through lines: how long each stretch of text was, and how many stretches at a time. A caption says: only the top two lines are needed to price a run.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="17" fill="#2C2A28" font-size="12">The model card: two lines matter today, the rest can wait</text>
<g transform="translate(70,32)">
  <rect x="0" y="0" width="380" height="140" rx="10" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="2"/>
  <rect x="0" y="0" width="380" height="26" rx="10" fill="#EDE9E0" stroke="#B8AEA2"/>
  <text x="190" y="18" fill="#3A342E" font-size="10">MODEL CARD</text>
  <g transform="translate(16,34)">
    <rect x="0" y="0" width="348" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
    <text x="8" y="20" fill="#1a5c38" font-size="11" text-anchor="start">knobs (parameters)   N = 7 000 000 000</text>
    <text x="330" y="20" fill="#2D8B55" font-size="12" text-anchor="end">✔</text>
  </g>
  <g transform="translate(16,70)">
    <rect x="0" y="0" width="348" height="30" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
    <text x="8" y="20" fill="#1a5c38" font-size="11" text-anchor="start">tokens read       D = 1 400 000 000 000</text>
    <text x="330" y="20" fill="#2D8B55" font-size="12" text-anchor="end">✔</text>
  </g>
  <g transform="translate(16,106)" font-size="9" fill="#9A938A" text-anchor="start">
    <text x="8" y="10">how long each stretch of text was — not needed today</text>
    <text x="8" y="22">how many stretches at a time — not needed today</text>
    <line x1="4" y1="7" x2="300" y2="7" stroke="#B8AEA2"/>
    <line x1="4" y1="19" x2="290" y2="19" stroke="#B8AEA2"/>
  </g>
</g>
<text x="260" y="188" fill="#9A938A" font-size="9">N counts the knobs inside the model · D counts the text that went past them</text>
</g></svg>
%%%

**What the bicycle-card picture gets right:** two well-chosen lines carry most of the information. So you can decide something real without reading the whole history. **Where the picture stops being true:** gears and kilometres are things you can see and touch. N is a count of changeable numbers hidden inside a file. D is a count of text chunks that have long since scrolled past.

#### What each of the two numbers really is
Let's slow down on both, because the rest of the day multiplies them.

%%% steps
step: N — the number of knobs
why: every knob is one number the model owns and is allowed to change. m02 called these weights; "knob" is the same thing in plain words. A model with N = 7 000 000 000 has seven billion little dials inside it.
step: N does not care how long your text is
why: the knobs are built into the model, like the gears on the bicycle. Reading a longer [[stretch of text||One continuous piece of text the model reads in one go. Code often calls it a sequence. It is made of many tokens.]] does not add knobs. So N is a fact about the MODEL, and that is all.
step: D — the number of tokens read during training
why: a token is one chunk of text. Training means text flowing past the knobs, and D is the total count of chunks that flowed. D = 1 400 000 000 000 means one point four trillion chunks went by.
step: one knob meeting one token is a handshake
why: this is the word we will use all day. N knobs times D tokens gives the number of handshakes in the whole run. The bill is then just a fixed price per handshake.
%%%

#### Touch the two numbers before we price them
You have not learned the price yet — that is the next three units. But you can already play with the two numbers, and playing first makes the arithmetic land better. Below is a little explorer page with a slider for N and a slider for D. **Predict first:** which slider do you think changes the big number at the top more — the knobs one, or the tokens one? Then try changing the sliders yourself and find out.

%%% viz src=../../viz/transformer-flops.html title="first play: drag N and D, watch the big number move" caption="Part 1 has one slider for N (knobs) and one for D (tokens). Ignore the formula for now. Just drag each slider on its own and notice that BOTH of them move the big number, and they move it by the same amount. That single observation is what unit 7 will explain."
%%%

Where does N itself come from? You do not have to trust the card. You can work it out from the model's shape, and that is exactly what **day 2** does. Today N arrives as a given, the way a price arrives on a label.

#### Two runs, drawn side by side
Here are two real-sized runs. Look at the two blocks below before you read any numbers. Each block is as wide as its knob count and as tall as its token count. The handshakes are the **space inside** the block.

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="Two blocks drawn side by side to compare two training runs. Run A is narrow and tall: 70 billion knobs wide, 1.4 trillion tokens high, and its area is labelled 9.8 times ten to the 22 handshakes. Run B is wide and short: 175 billion knobs wide, 300 billion tokens high, and its area is labelled 5.25 times ten to the 22 handshakes. A caption says the narrow tall block holds more space, so the smaller model had the bigger bill.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Wide = more knobs · tall = more tokens · the SPACE inside = handshakes</text>
<g transform="translate(60,30)">
  <rect x="0" y="0" width="50" height="168" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/>
  <text x="25" y="90" fill="#5E5191" font-size="10">RUN A</text>
  <text x="25" y="184" fill="#5E5191" font-size="9">N = 70 billion</text>
  <text x="-84" y="84" fill="#5E5191" font-size="9" transform="rotate(-90 -84 84)">D = 1.4 trillion</text>
  <text x="25" y="204" fill="#2C2A28" font-size="10">9.8e22</text>
  <text x="25" y="218" fill="#6B645E" font-size="8">handshakes</text>
</g>
<g transform="translate(230,162)">
  <rect x="0" y="0" width="125" height="36" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="62" y="24" fill="#1a5c38" font-size="10">RUN B</text>
  <text x="62" y="52" fill="#276b45" font-size="9">N = 175 billion</text>
  <text x="-18" y="18" fill="#276b45" font-size="9" transform="rotate(-90 -18 18)">D = 300 bn</text>
  <text x="62" y="72" fill="#2C2A28" font-size="10">5.25e22</text>
  <text x="62" y="86" fill="#6B645E" font-size="8">handshakes</text>
</g>
<g transform="translate(400,60)">
  <rect x="0" y="0" width="112" height="86" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/>
  <text x="56" y="20" fill="#8A6D3B" font-size="9">A is 2.5× narrower</text>
  <text x="56" y="36" fill="#8A6D3B" font-size="9">but 4.7× taller</text>
  <text x="56" y="58" fill="#9A5A12" font-size="9">4.7 beats 2.5</text>
  <text x="56" y="76" fill="#8a3b3b" font-size="9">so A holds MORE</text>
</g>
<text x="260" y="240" fill="#9A938A" font-size="9">both sides pull on the bill with the same strength — unit 7 draws this idea again as a tiled floor</text>
</g></svg>
%%%

Run A's model is much smaller, and its block still holds more space. Read the little yellow note in the picture: A is 2.5 times narrower, but 4.7 times taller, and 4.7 beats 2.5. That is the whole comparison, done by eye.

%%% insight
The two-numbers idea is genuinely strange the first time. Two models can differ in every visible way. Different depth, different width, different language. If their N and D match, their training bills match too. That is what makes this arithmetic worth remembering. It sees past all the design detail to the one thing that costs money.
%%%

%%% demo id=c2which label="predict which one cost more, then reveal"
predict: run A has 70 billion knobs and read 1.4 trillion tokens. Run B has 175 billion knobs and read 300 billion tokens. B has a much bigger model — so B cost more, surely?
code: run A handshakes = 70 000 000 000 × 1 400 000 000 000
code: run B handshakes = 175 000 000 000 ×   300 000 000 000
out: run A = 9.8e22 handshakes
out: run B = 5.25e22 handshakes
out: A is about 1.9 times bigger than B
take: <b>The smaller model had the bigger bill.</b> The numbers agree with the picture exactly. Both sides pull on the bill with equal strength. That is the single most useful thing to notice today, and unit 7 draws it one more time.
%%%

Two numbers in hand. Next we work out the price of ONE handshake. For that we visit a shop till.

@@@ concept id=c3 tag="The till" title="Using one knob costs two steps: multiply, then add" gotit="Got the two steps"
Stand at a shop till — a cash register — and watch what happens for one item. The cashier scans a bag of apples. Price 2 per bag, three bags, so **2 × 3 = 6**. That is a multiply. Then the till **adds** that 6 into the running total on the little screen. Scan, add. Scan, add. Two actions per item, every single time.

That is exactly what a knob does inside a model. It is the second key fact of the day. A knob never sits alone. It multiplies the number arriving at it, and the result is added into a running total. Two steps. The engineers call the pair a [[multiply-accumulate||One multiply followed by one add into a running total — the two-step move a knob makes every time it is used. Often shortened to MAC. It costs 2 steps of arithmetic.]]. You can hear the whole thing in the name: multiply, then **accumulate** — and to accumulate simply means to add your result into a total that keeps growing.

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="A shop till and its receipt as the picture for multiply-accumulate. On the left, the till has a scanner labelled step one multiply price times count, a plus key labelled step two add into the running total, and a screen showing the running total 23. On the right, a receipt lists three lines: apples 2 times 3 equals 6, bread 5 times 1 equals 5, milk 4 times 3 equals 12, and a total of 23. A caption says: two presses per item, always.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">One item at the till = one multiply + one add. Two steps, always.</text>
<g transform="translate(24,34)">
  <rect x="0" y="20" width="180" height="120" rx="10" fill="#EDE9E0" stroke="#3A342E" stroke-width="2"/>
  <rect x="18" y="34" width="144" height="34" rx="4" fill="#2C2A28"/>
  <text x="90" y="57" fill="#7FE3A6" font-size="16">running total 23</text>
  <g fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"><rect x="18" y="82" width="66" height="42" rx="5"/><rect x="96" y="82" width="66" height="42" rx="5"/></g>
  <text x="51" y="100" fill="#8A6D3B" font-size="9">scan</text>
  <text x="51" y="114" fill="#9A5A12" font-size="8">2 × 3</text>
  <text x="129" y="102" fill="#8A6D3B" font-size="16">+</text>
  <text x="129" y="116" fill="#9A5A12" font-size="8">add it in</text>
  <text x="51" y="16" fill="#8A6D3B" font-size="8">step 1: multiply</text>
  <text x="129" y="16" fill="#8A6D3B" font-size="8">step 2: add to the total</text>
</g>
<g transform="translate(300,30)">
  <path d="M0 0 h190 v150 l-9 -8 l-9 8 l-9 -8 l-9 8 l-9 -8 l-9 8 l-9 -8 l-9 8 l-9 -8 l-9 8 l-9 -8 l-9 8 l-9 -8 l-9 8 l-9 -8 l-9 8 l-9 -8 l-9 8 z" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="1.5"/>
  <text x="95" y="20" fill="#3A342E" font-size="9">RECEIPT</text>
  <g font-size="9" fill="#6B645E" text-anchor="start">
    <text x="14" y="42">apples   2 × 3  =   6</text>
    <text x="14" y="62">bread    5 × 1  =   5</text>
    <text x="14" y="82">milk     4 × 3  =  12</text>
    <line x1="14" y1="92" x2="176" y2="92" stroke="#B8AEA2"/>
    <text x="14" y="110" fill="#2C2A28">TOTAL          23</text>
  </g>
  <text x="95" y="136" fill="#9A5A12" font-size="8">3 items → 3 multiplies + 3 adds = 6 steps</text>
</g>
<text x="260" y="196" fill="#9A938A" font-size="9">a knob is an item at the till: it multiplies what arrives, then the result joins the running total</text>
</g></svg>
%%%

**What the till picture gets right:** the two actions always travel together. So if you know how many items were scanned, you instantly know how many actions happened — double the items. **What the picture gets wrong:** a cashier does them one after another. A chip does many multiplies at the same instant. The *count* is identical; only the *waiting* differs.

#### Count the till, and a 2 appears
Say a knob-row has three knobs, and three numbers arrive. Every knob multiplies its own arriving number, and each result joins the running total. Count the presses with me.

%%% steps
step: knob 1: 0.5 × 4 = 2.0
why: one multiply. Press 1.
step: total ← 0 + 2.0
why: one add into the running total. Press 2. That is knob 1 fully paid for: two presses.
step: knob 2: 2.0 × 3 = 6.0, then total ← 2.0 + 6.0
why: another multiply and another add. Presses 3 and 4.
step: knob 3: 1.0 × 7 = 7.0, then total ← 8.0 + 7.0
why: presses 5 and 6. Final total 15.0.
step: 3 knobs used → 6 steps of arithmetic
why: exactly 2 steps per knob used. This little 2 is the first half of our six.
step: wait — didn't the last unit say three terms need only TWO adds?
why: it did, and you were right to notice. Back there we added up a sum that was already written down, so joining three numbers took two adds. Here we count the very first add as well — the one that puts 2.0 into a total that is still empty. We count it on purpose, because the chip really does perform that add: its running-total register starts at zero and every product gets added in, including the first. That one extra add is the whole difference, and it is what makes the price land on **exactly 2 per knob** instead of "a bit under 2". Every published FLOP count does it this way, which is why the rule reads 2 steps per knob and not 1.7.
%%%

!!! c-warn ⚠️
**The convention, stated once so it never bites you.** From here on, "2 steps per knob" means *one multiply plus one add, counting the opening add into the empty total*. That is the convention behind every published number in this lesson. If you ever count a small sum by hand and land one add short, this is why — and both answers are "right", they are just answering with different rules. State your rule, then stay inside it.
!!!

There it is: **2 steps for every knob, every time that knob gets used.** Not two steps per model. Not two steps per day. Two steps per *use*.

%%% insight
This is the moment the day stops being about arithmetic and starts being about money. You no longer have to know what a model does inside. If you can say how many times its knobs get used, you multiply by 2 and you know the work. Everything left today is one happy question: how many times do the knobs get used?
%%%

#### The same count, drawn for a whole grid of knobs
Knobs do not sit in a single row. They sit in a rectangle, and the arriving numbers form another rectangle. The picture below draws the small case so you can see the same 2 appear again. Watch the one shaded answer-cell: it is just a little till receipt.

%%% svg
<svg viewBox="0 0 520 240" role="img" aria-label="A grid multiplication drawn as tills. A two by two block of arriving numbers meets a two by two block of knobs and produces a two by two block of answers. One answer cell is shaded and blown up to the right, showing two multiply ticks and two add ticks, four steps in all. Underneath, a line reads four answer cells times four steps equals sixteen steps, which is two times m times k times n.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">One answer cell = one little receipt. Count one, then multiply.</text>
<g transform="translate(20,40)">
  <g fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"><rect x="0" y="0" width="30" height="30"/><rect x="30" y="0" width="30" height="30"/><rect x="0" y="30" width="30" height="30"/><rect x="30" y="30" width="30" height="30"/></g>
  <text x="30" y="76" fill="#5E5191" font-size="8">arriving numbers</text>
  <text x="30" y="88" fill="#5E5191" font-size="8">m × k = 2 × 2</text>
</g>
<text x="94" y="70" fill="#9A5A12" font-size="14">×</text>
<g transform="translate(112,40)">
  <g fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"><rect x="0" y="0" width="30" height="30"/><rect x="30" y="0" width="30" height="30"/><rect x="0" y="30" width="30" height="30"/><rect x="30" y="30" width="30" height="30"/></g>
  <text x="30" y="76" fill="#276b45" font-size="8">knobs</text>
  <text x="30" y="88" fill="#276b45" font-size="8">k × n = 2 × 2</text>
</g>
<text x="188" y="70" fill="#9A5A12" font-size="14">=</text>
<g transform="translate(208,40)">
  <g fill="#FDFCFA" stroke="#3A342E" stroke-width="1.5"><rect x="0" y="0" width="30" height="30"/><rect x="30" y="0" width="30" height="30"/><rect x="0" y="30" width="30" height="30"/><rect x="30" y="30" width="30" height="30"/></g>
  <rect x="0" y="0" width="30" height="30" fill="#FDF3E7" stroke="#C93B3B" stroke-width="2.5"/>
  <text x="30" y="76" fill="#3A342E" font-size="8">answers</text>
  <text x="30" y="88" fill="#3A342E" font-size="8">m × n = 4 cells</text>
</g>
<path d="M244 44 L306 40" stroke="#C93B3B" stroke-width="1.5"/>
<path d="M244 70 L306 116" stroke="#C93B3B" stroke-width="1.5"/>
<g transform="translate(306,30)">
  <rect x="0" y="0" width="196" height="96" rx="8" fill="#FDFCFA" stroke="#C93B3B" stroke-width="2"/>
  <text x="98" y="18" fill="#8a3b3b" font-size="9">inside ONE shaded cell</text>
  <g fill="#9A5A12" font-size="9" text-anchor="start">
    <text x="14" y="38">× tick 1   ✓</text><text x="112" y="38">+ tick 1   ✓</text>
    <text x="14" y="56">× tick 2   ✓</text><text x="112" y="56">+ tick 2   ✓</text>
  </g>
  <line x1="14" y1="66" x2="182" y2="66" stroke="#B8AEA2"/>
  <text x="98" y="84" fill="#2C2A28" font-size="10">k multiplies + k adds = 4 steps</text>
</g>
<g transform="translate(96,152)">
  <rect x="0" y="0" width="328" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="164" y="22" fill="#1a5c38" font-size="11">4 answer cells × 4 steps = 16 steps</text>
  <text x="164" y="42" fill="#276b45" font-size="10">= 2 × m × k × n = 2 × 2 × 2 × 2 ✔</text>
</g>
<text x="260" y="230" fill="#9A938A" font-size="9">the 2 in that formula is the same 2 as the till's multiply-and-add — nothing new is hiding in there</text>
</g></svg>
%%%

Look at what that picture says. Each answer-cell costs `2 × k` steps. There are `m × n` answer-cells. Multiply those together and you get `2 × m × k × n`. The 2 came from the till, and it never leaves.

!!! c-info 🔬
<b>Optional (skippable) — the grid count in words.</b><br>
<b>The name:</b> multiplying two rectangles of numbers is called a matrix multiply.<br>
<b>The shapes:</b> a rectangle of size <code>m × k</code> times one of size <code>k × n</code> gives an answer with <code>m × n</code> cells.<br>
<b>The count:</b> each answer-cell is a receipt with <code>k</code> items on it, so it costs <code>2 × k</code> steps.<br>
<b>All together:</b> steps = <code>2 × m × k × n</code>.<br>
<b>Why we care:</b> this is the long way. The short way, coming in unit 4, gives the same answer from N alone.
!!!

%%% demo id=c3grid label="predict the grid count, then reveal it"
predict: two small grids, each 2 rows by 2 columns, multiplied together. The answer has 4 cells. How many steps of arithmetic in total? Try it before you look.
code: answer cells        = 2 × 2            = 4 cells
code: items per cell      = 2                 (k = 2)
code: steps per cell      = 2 multiplies + 2 adds = 4
code: total steps         = 4 cells × 4 steps
out: 16 steps of arithmetic
out: check with the formula: 2 × m × k × n = 2 × 2 × 2 × 2 = 16 ✔
take: <b>16 — and the formula agrees.</b> You just counted a matrix multiply by hand. That is the honest, slow way people did it for years. Remember how it felt. The shortcut we are about to meet is not magic. It is this same count, arranged so you never have to do it again.
%%%

Two steps per knob-use. Now only one question is left for the reading direction: **how many times does each knob get used?**

@@@ concept id=c4 tag="Reading: two per handshake" title="The forward pass — every knob greets every token once" gotit="Got the forward pass"
Picture a wedding receiving line. Ten hosts stand in a row. Each guest walks along the line and shakes every host's hand exactly once. No skipping, no doubling back. If forty guests come, the number of handshakes is simply 40 × 10 = 400. Nobody has to watch. The multiplication does the counting for you.

A model reading text works the same way. The hosts are the knobs. The guests are the tokens. When the model does a [[forward pass||The model reading its input and producing a prediction — the "answering" direction. No learning happens here.]] it reads text and predicts what comes next. During that pass, **every knob gets its turn with every token, once.** So the number of handshakes is knobs × tokens. And every handshake costs the two steps you just counted at the till.

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="A receiving line as the picture for the forward pass. On the left a queue of four token guests walks in. On the right a row of five knob hosts stands in a line. Every one of the four tokens is joined by a line to every one of the five knobs, twenty lines in all, and one of those lines is highlighted and labelled one handshake equals one multiply plus one add equals two steps. A caption says twenty lines equals four tokens times five knobs, so handshakes equal tokens times knobs.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Every token greets every knob exactly once — count the lines</text>
<text x="62" y="38" fill="#5E5191" font-size="9">tokens (guests)</text>
<g transform="translate(20,46)" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5">
  <rect x="0" y="0" width="84" height="26" rx="5"/><rect x="0" y="34" width="84" height="26" rx="5"/><rect x="0" y="68" width="84" height="26" rx="5"/><rect x="0" y="102" width="84" height="26" rx="5"/>
</g>
<g fill="#5E5191" font-size="9"><text x="62" y="63">"the"</text><text x="62" y="97">"cat"</text><text x="62" y="131">"sat"</text><text x="62" y="165">"down"</text></g>
<text x="404" y="38" fill="#276b45" font-size="9">knobs (hosts in a row)</text>
<g transform="translate(340,46)" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5">
  <rect x="0" y="0" width="128" height="22" rx="4"/><rect x="0" y="28" width="128" height="22" rx="4"/><rect x="0" y="56" width="128" height="22" rx="4"/><rect x="0" y="84" width="128" height="22" rx="4"/><rect x="0" y="112" width="128" height="22" rx="4"/>
</g>
<g fill="#1a5c38" font-size="8"><text x="404" y="61">knob 1</text><text x="404" y="89">knob 2</text><text x="404" y="117">knob 3</text><text x="404" y="145">knob 4</text><text x="404" y="173">knob 5</text></g>
<g stroke="#B8AEA2" stroke-width="0.9">
  <path d="M106 60 L338 57"/><path d="M106 60 L338 85"/><path d="M106 60 L338 113"/><path d="M106 60 L338 141"/><path d="M106 60 L338 169"/>
  <path d="M106 94 L338 57"/><path d="M106 94 L338 85"/><path d="M106 94 L338 113"/><path d="M106 94 L338 141"/>
  <path d="M106 128 L338 57"/><path d="M106 128 L338 85"/><path d="M106 128 L338 113"/><path d="M106 128 L338 141"/><path d="M106 128 L338 169"/>
  <path d="M106 162 L338 57"/><path d="M106 162 L338 85"/><path d="M106 162 L338 113"/><path d="M106 162 L338 141"/><path d="M106 162 L338 169"/>
</g>
<path d="M106 94 L338 169" stroke="#C93B3B" stroke-width="2.5"/>
<g transform="translate(150,180)"><rect x="0" y="0" width="230" height="26" rx="5" fill="#FDF3E7" stroke="#C93B3B" stroke-width="1.5"/><text x="115" y="17" fill="#8a3b3b" font-size="9">1 handshake = 1 multiply + 1 add = 2 steps</text></g>
<text x="260" y="212" fill="#9A938A" font-size="8">20 lines = 4 tokens × 5 knobs · nobody skipped, nobody greeted twice</text>
</g></svg>
%%%

**What the receiving line gets right:** the total is a plain multiplication, because nobody is skipped and nobody is greeted twice. Count the twenty faint lines in the picture if you like — they are all drawn, and 4 × 5 = 20. That is exactly the bookkeeping we need. **Where the picture stops being true:** guests file past one at a time, while a chip greets whole crowds at once. Also, a few knobs inside a real model get used a little more or less than exactly once. That is why the rule says *about*.

#### Build it up from the tiny case
Start absurdly small, so you can check every number in your head.

%%% steps
step: a model with N = 2 knobs reads D = 3 tokens
why: two hosts, three guests. Nothing hidden.
step: handshakes = 2 × 3 = 6
why: each of the 3 tokens greets each of the 2 knobs once.
step: steps of arithmetic = 6 handshakes × 2 = 12
why: every handshake is that multiply-then-add pair from the till.
step: so reading ONE token costs 2 × N steps
why: one guest greeting all N hosts. With N = 2 that is 4 steps per token — and 4 × 3 tokens = 12. The two ways agree.
%%%

Now swap in a real size. Nothing changes except the digits. A model with 7 billion knobs spends about 14 billion steps of arithmetic to read **one single token**. Fourteen billion, for one chunk of text. That is a good moment to stop and notice how big this number is.

%%% mathladder
title: the forward pass, in one line
words: reading costs two steps for every knob, for every token
formula: forward steps ≈ 2 × N × D
numbers: N = 7 000 000 000 knobs and D = 1 token → 2 × 7e9 × 1 = 1.4e10 steps (14 billion)
sanity: set N = 0 and the cost is 0 — a model with no knobs does no arithmetic. Set D = 0 and the cost is 0 too: read nothing, pay nothing.
%%%

%%% insight
Notice what just happened. You never asked what the model *is*. Not how deep it is, not how wide, not what it was reading about. The receiving-line count did not need any of that. That is why one line of multiplication can replace a whole table of per-layer formulas. It is also why this rule outlives every particular model design.
%%%

%%% demo id=c4fwd label="predict the reading bill, then reveal it"
predict: a model with 7 billion knobs reads a 1 000-token article. How many steps of arithmetic? Guess the power of ten before you look.
code: steps per token   = 2 × 7 000 000 000        = 1.4e10
code: tokens            = 1 000
code: total             = 1.4e10 × 1 000
out: 1.4e13 steps of arithmetic  (14 trillion)
out: on a chip doing 4e14 steps per second: about 0.035 seconds
take: <b>14 trillion steps to read one article — and about three hundredths of a second to do it.</b> Both halves of that sentence are true. Holding them together is the actual skill. The count is enormous. The chip is enormous too. Unit 9 turns counts into seconds properly.
%%%

Reading is paid for. But training is not only reading. The model also has to *learn* from what it read, and learning costs more. That is the other half of the six.

@@@ concept id=c5 tag="Learning: about four" title="The backward pass — two notes of blame, not one" gotit="Got the backward pass"
Think about a relay race where the team loses by two seconds. The message travels back down the line of runners. Each runner has to work out **two** separate things. First: *how much of those two seconds was my own running?* Second: *how many seconds do I hand back to the runner before me?* Two questions per runner, not one. So the walk back down the line takes about twice as long as the trip forward.

Training a model does the same walk. After the forward pass, the model compares its guess with the true next token. It gets a **blame** — the number that says how wrong each knob was. That return trip is the [[backward pass||The trip back through the model that works out the blame for every knob. It does two jobs at each step, so it costs about twice the forward pass.]]. And here is the good news: you do not have to take the "about twice" on trust. In a minute you will count it, press by press, on one single knob.

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="A relay line as the picture for the backward pass. A green arrow across the top goes forward through three runners and costs two steps per handshake. Below, a red arrow returns through the same three runners, and each runner holds two notes: note one, the blame that is mine, and note two, the blame I pass back. A caption says two notes per knob, so about four steps per handshake.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Forward: one job. Backward: two jobs — so about twice the cost.</text>
<path d="M30 44 H480" stroke="#2D8B55" stroke-width="2.5" marker-end="url(#fw)"/>
<defs>
  <marker id="fw" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker>
  <marker id="bk" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C93B3B"/></marker>
</defs>
<text x="255" y="36" fill="#276b45" font-size="9">forward: read and guess · 2 steps per handshake</text>
<g transform="translate(60,58)" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5">
  <rect x="0" y="0" width="100" height="34" rx="6"/><rect x="150" y="0" width="100" height="34" rx="6"/><rect x="300" y="0" width="100" height="34" rx="6"/>
</g>
<g fill="#1a5c38" font-size="9"><text x="110" y="80">knob row 1</text><text x="260" y="80">knob row 2</text><text x="410" y="80">knob row 3</text></g>
<path d="M480 106 H30" stroke="#C93B3B" stroke-width="2.5" marker-end="url(#bk)"/>
<text x="255" y="100" fill="#8a3b3b" font-size="9">backward: carry the blame home · about 4 steps per handshake</text>
<g font-size="8">
  <g transform="translate(60,120)"><rect x="0" y="0" width="100" height="24" rx="4" fill="#FDF3E7" stroke="#C99A12"/><text x="50" y="16" fill="#8A6D3B">note 1: my own blame</text></g>
  <g transform="translate(60,150)"><rect x="0" y="0" width="100" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="50" y="16" fill="#5E5191">note 2: pass it back</text></g>
  <g transform="translate(210,120)"><rect x="0" y="0" width="100" height="24" rx="4" fill="#FDF3E7" stroke="#C99A12"/><text x="50" y="16" fill="#8A6D3B">note 1: my own blame</text></g>
  <g transform="translate(210,150)"><rect x="0" y="0" width="100" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="50" y="16" fill="#5E5191">note 2: pass it back</text></g>
  <g transform="translate(360,120)"><rect x="0" y="0" width="100" height="24" rx="4" fill="#FDF3E7" stroke="#C99A12"/><text x="50" y="16" fill="#8A6D3B">note 1: my own blame</text></g>
  <g transform="translate(360,150)"><rect x="0" y="0" width="100" height="24" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="50" y="16" fill="#5E5191">note 2: pass it back</text></g>
</g>
<text x="260" y="196" fill="#9A938A" font-size="9">two notes per knob per token — one to fix itself, one for whoever came earlier</text>
<text x="260" y="209" fill="#9A938A" font-size="8">the forward trip made one note per knob, so the return trip is about double</text>
</g></svg>
%%%

**What the relay picture gets right:** the return trip really does carry two separate errands per runner. Two errands take about twice as long as one. **Where the picture stops working:** a runner's two questions take the same effort each. In a model the two jobs are only *roughly* equal in size. That is exactly why we say "about four" and not "exactly four".

#### Count one knob's notes, the same way you counted the till
Let's go back to the relay and slow it right down — one runner, one knob, three little numbers. Nothing here is harder than the till.

- the knob's own value is **0.5**
- the number that arrived at it on the way in was **4**
- the blame arriving from ahead on the way back is **0.2**

%%% steps
step: forward — presses 1 and 2
why: the knob multiplies, 0.5 × 4 = 2.0 (press 1), and that 2.0 is added into the running total (press 2). This is the till, unchanged. Two presses, and you already owned this pair.
step: note 1, my own blame — presses 3 and 4
why: my blame = the blame that arrived × the number I saw = 0.2 × 4 = 0.8. That is one multiply (press 3). Then the 0.8 is added into the little store where this knob keeps its blame (press 4). Two presses.
step: note 2, the blame I hand back — presses 5 and 6
why: what I pass back = the blame that arrived × my own value = 0.2 × 0.5 = 0.1. One multiply (press 5). Then it is added into the blame arriving at the knob before me (press 6). Two more presses.
step: putting it together: 2 forward, 4 backward
why: therefore the trip back cost 4 presses against the trip forward's 2 — because two notes means two multiply-and-add pairs instead of one. And 2 + 4 = 6 presses for one handshake. You did not accept the six. You counted it.
%%%

Here are those six presses drawn as one picture. Three little panels, two presses each, and a tally bar underneath.

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="The six presses of one handshake drawn as three panels and a tally bar. Panel one, forward, shows a multiply tick zero point five times four equals two point zero and an add tick into the running total, two presses. Panel two, note one, shows a multiply tick zero point two times four equals zero point eight and an add tick into my own blame store, two presses. Panel three, note two, shows a multiply tick zero point two times zero point five equals zero point one and an add tick into the earlier knob's blame, two presses. Under them a bar is split into a green part two sixths wide labelled forward two presses and a red part four sixths wide labelled backward four presses, with the total six per handshake.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">One knob, one token: 2 presses out, 4 presses back — six in all</text>
<g transform="translate(6,26)">
  <rect x="0" y="0" width="160" height="96" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="80" y="17" fill="#1a5c38" font-size="9">FORWARD · reading</text>
  <g text-anchor="start" font-size="8" fill="#276b45">
    <text x="12" y="38">✓ ×  0.5 × 4 = 2.0</text>
    <text x="12" y="56">✓ +  2.0 into the total</text>
  </g>
  <line x1="12" y1="64" x2="148" y2="64" stroke="#B8AEA2"/>
  <text x="80" y="82" fill="#1a5c38" font-size="11">2 presses</text>
</g>
<g transform="translate(180,26)">
  <rect x="0" y="0" width="160" height="96" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/>
  <text x="80" y="17" fill="#8A6D3B" font-size="9">NOTE 1 · my own blame</text>
  <g text-anchor="start" font-size="8" fill="#9A5A12">
    <text x="12" y="38">✓ ×  0.2 × 4 = 0.8</text>
    <text x="12" y="56">✓ +  0.8 into my store</text>
  </g>
  <line x1="12" y1="64" x2="148" y2="64" stroke="#B8AEA2"/>
  <text x="80" y="82" fill="#8A6D3B" font-size="11">2 presses</text>
</g>
<g transform="translate(354,26)">
  <rect x="0" y="0" width="160" height="96" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/>
  <text x="80" y="17" fill="#5E5191" font-size="9">NOTE 2 · blame handed back</text>
  <g text-anchor="start" font-size="8" fill="#5E5191">
    <text x="12" y="38">✓ ×  0.2 × 0.5 = 0.1</text>
    <text x="12" y="56">✓ +  0.1 to the knob before</text>
  </g>
  <line x1="12" y1="64" x2="148" y2="64" stroke="#B8AEA2"/>
  <text x="80" y="82" fill="#5E5191" font-size="11">2 presses</text>
</g>
<text x="260" y="142" fill="#6B645E" font-size="9">now stack the three panels into one bar — each press the same width</text>
<g transform="translate(60,152)">
  <rect x="0" y="0" width="133" height="30" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <rect x="133" y="0" width="267" height="30" fill="#FDF3E7" stroke="#C93B3B" stroke-width="2"/>
  <text x="66" y="20" fill="#1a5c38" font-size="9">forward: 2</text>
  <text x="266" y="20" fill="#8a3b3b" font-size="9">backward: 2 + 2 = 4</text>
  <text x="66" y="46" fill="#276b45" font-size="8">one third</text>
  <text x="266" y="46" fill="#8a3b3b" font-size="8">two thirds</text>
</g>
<g transform="translate(170,208)">
  <rect x="0" y="0" width="180" height="30" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="90" y="20" fill="#1a5c38" font-size="13">6 presses per handshake</text>
</g>
<text x="260" y="248" fill="#9A938A" font-size="8">the backward bar is twice the forward bar because it carries two notes, not one</text>
</g></svg>
%%%

%%% mathladder
title: the backward pass, counted on one knob
words: one note out, two notes back — and every note is a multiply plus an add
formula: forward ≈ 2 × N × D and backward ≈ 4 × N × D
numbers: one handshake with knob 0.5, input 4, arriving blame 0.2 → forward 2 presses, note 1 two presses, note 2 two presses, so 2 + 4 = 6
sanity: take note 2 away (which is what happens at the very first knob row — there is nobody earlier to hand blame to) and the backward cost drops from 4 towards 2. The count moves when the work moves, which is what a real count should do.
%%%

!!! c-warn ⚠️
<b>Say the assumption out loud.</b> You counted 2 presses for each of the two backward notes, so 4 in all. In a real model the two notes are only <i>roughly</i> the same size — at the first knob row there is nobody earlier, so note 2 is skipped, and a few steps in the middle are cheaper than a plain multiply-and-add.<br>
<b>So the honest sentence is:</b> backward is <i>about</i> twice forward. That is measured, standard, and what the published rules use — it is not exact.<br>
When you show someone an estimate built on it, name it: <i>I assumed backward is about twice forward.</i> An estimate with its assumptions named is a professional estimate. One without them is only a guess that looks careful.
!!!

!!! c-info 🔬
<b>Optional (skippable) — where the two notes come from.</b><br>
<b>Note 1 exists</b> because the knob needs to know how to change itself, and that depends on the number it saw on the way in.<br>
<b>Note 2 exists</b> because the knobs *before* it also need blame, and the only route to them is through this knob's own value.<br>
<b>Why two and not three:</b> those are the only two things anyone downstream needs. Nothing else has to be worked out.<br>
<b>Want every step?</b> m04 day 2 works the whole return trip out in full. Today's count is complete on its own — that day is extra depth, not the source of the 4.
!!!

%%% insight
This is the part that surprises people: learning is the expensive half. Look at the bar again — two-thirds of a training bill is the walk *back*, not the reading. So when somebody says "we spent nine days training", most of those days were the model carrying notes of blame backwards.
%%%

%%% demo id=c5bwd label="predict the learning bill, then reveal it"
predict: a model with 7 billion knobs reads one token AND learns from it. Forward was 1.4e10 steps. What is the backward part — and what is the total?
code: forward  = 2 × 7e9   = 1.4e10 steps
code: backward = 4 × 7e9   = 2.8e10 steps
code: total    = 6 × 7e9
out: forward  1.4e10  (one third of the work)
out: backward 2.8e10  (two thirds of the work)
out: total    4.2e10 steps for one token, read AND learned from
take: <b>Six steps per handshake, and there is our number.</b> 2 for reading + 4 for learning = 6. You have now built the whole rule out of two pictures and one honest count: a till, a relay line, and six presses on one knob. The next unit just writes it down.
%%%

Two plus four. Let's put them together and meet the line this whole day is named after.

@@@ concept id=c6 tag="Six per handshake" title="Add them up: C ≈ 6 × N × D" gotit="Got the rule"
Go back to the shop till one last time. This time ask for the *bill* rather than one item. The receipt has two lines on it. Reading: 2 steps per handshake. Learning: about 4 steps per handshake. Total per handshake: **6**. Then the till does what tills do. It multiplies the price per item by the number of items.

The number of items is the number of handshakes, and you already know how to count those: N knobs × D tokens. So the whole training bill is six, times knobs, times tokens. People write it with a **C** for compute, meaning the total steps of arithmetic in the run: **C ≈ 6 × N × D**. The wavy equals sign means *about*. By the end of today you will know exactly which honest wobbles live inside that word.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A receipt as the picture for the training bill. The receipt has two priced lines: reading, 2 steps per handshake, and learning, 4 steps per handshake, totalling 6 steps per handshake. Below, a multiplication line reads: times handshakes, which equal N knobs times D tokens, giving the total C equals 6 N D. A caption says: a fixed price per handshake times the number of handshakes.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">The whole bill: a fixed price per handshake × how many handshakes</text>
<g transform="translate(120,28)">
  <path d="M0 0 h280 v128 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 l-10 -7 l-10 7 z" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="1.5"/>
  <text x="140" y="20" fill="#3A342E" font-size="10">TRAINING BILL · per handshake</text>
  <g font-size="10" text-anchor="start" fill="#6B645E">
    <text x="24" y="46">reading  (forward)</text><text x="238" y="46" fill="#2D8B55" text-anchor="end">2</text>
    <text x="24" y="68">learning (backward)</text><text x="238" y="68" fill="#C93B3B" text-anchor="end">4</text>
    <line x1="24" y1="78" x2="240" y2="78" stroke="#B8AEA2"/>
    <text x="24" y="98" fill="#2C2A28">per handshake</text><text x="238" y="98" fill="#2C2A28" text-anchor="end">6</text>
  </g>
  <text x="140" y="118" fill="#9A5A12" font-size="9">× handshakes = N × D</text>
</g>
<g transform="translate(130,166)">
  <rect x="0" y="0" width="260" height="34" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="130" y="23" fill="#1a5c38" font-size="15">C ≈ 6 × N × D</text>
</g>
<text x="260" y="214" fill="#9A938A" font-size="8">C = total steps of arithmetic · N = knobs · D = tokens read while training</text>
</g></svg>
%%%

**What the receipt picture gets right:** a bill really is a price-per-item times a number of items. Both of ours are honest counts. **What the picture gets wrong:** a shop receipt is exact to the penny. Our 6 is an *about*. The small print at the bottom of this receipt is units 11, 13 and 14.

#### Now price a famous run, by hand
This is the moment where the whole day pays off. Take the well-known 2020 run with **175 billion knobs**, trained on **300 billion tokens**. Here is the arithmetic. You can do every line with a pen.

%%% steps
step: N = 175 000 000 000 = 1.75e11 knobs
why: 1.75e11 is shorthand. It means 1.75 with the decimal point moved 11 places right. Same number, fewer zeros to lose.
step: D = 300 000 000 000 = 3.0e11 tokens
why: three hundred billion chunks of text went past those knobs.
step: multiply the front numbers: 6 × 1.75 × 3.0 = 31.5
why: plain school multiplication. 6 × 1.75 = 10.5, and 10.5 × 3 = 31.5.
step: add the powers of ten: 11 + 11 = 22
why: this is the only trick you need. When you multiply, the powers of ten add up.
step: C ≈ 31.5e22 = 3.15e23 steps of arithmetic
why: 31.5 with 22 zeros is the same as 3.15 with 23 zeros. Slide the point one place, raise the power by one.
%%%

#### The same five lines, drawn
That ladder is the heaviest arithmetic on the page, so here it is again as a picture. Read it left to right and watch the answer assemble.

%%% svg
<svg viewBox="0 0 520 260" role="img" aria-label="The worked example drawn in three panels. Panel one shows three boxes: the price six, the knobs one point seven five times ten to the eleven, and the tokens three times ten to the eleven. Panel two splits them into two rows: the front numbers six times one point seven five times three equal thirty one point five, and the powers of ten eleven plus eleven equal twenty two. Panel three slides the decimal point from thirty one point five times ten to the twenty two to three point one five times ten to the twenty three, and sets it beside the published figure three point one four times ten to the twenty three with a tick.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">Panel 1: what you have · Panel 2: two easy jobs · Panel 3: tidy it up</text>
<g transform="translate(10,26)">
  <rect x="0" y="0" width="150" height="86" rx="8" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="1.5"/>
  <text x="75" y="16" fill="#6B645E" font-size="8">1 · the three numbers</text>
  <g transform="translate(10,24)">
    <rect x="0" y="0" width="130" height="18" rx="4" fill="#FDF3E7" stroke="#C99A12"/><text x="65" y="13" fill="#9A5A12" font-size="9">price = 6</text>
    <rect x="0" y="22" width="130" height="18" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="65" y="35" fill="#1a5c38" font-size="9">N = 1.75e11</text>
    <rect x="0" y="44" width="130" height="18" rx="4" fill="#F4F1FB" stroke="#5E5191"/><text x="65" y="57" fill="#5E5191" font-size="9">D = 3.0e11</text>
  </g>
</g>
<text x="172" y="72" fill="#9A5A12" font-size="14">→</text>
<g transform="translate(186,26)">
  <rect x="0" y="0" width="160" height="86" rx="8" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="1.5"/>
  <text x="80" y="16" fill="#6B645E" font-size="8">2 · fronts multiply, powers add</text>
  <g transform="translate(8,24)">
    <rect x="0" y="0" width="144" height="26" rx="4" fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"/>
    <text x="72" y="12" fill="#9A5A12" font-size="7">fronts (× them)</text>
    <text x="72" y="22" fill="#8A6D3B" font-size="9">6 × 1.75 × 3 = 31.5</text>
    <rect x="0" y="32" width="144" height="26" rx="4" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"/>
    <text x="72" y="44" fill="#5E5191" font-size="7">powers of ten (+ them)</text>
    <text x="72" y="54" fill="#5E5191" font-size="9">11 + 11 = 22</text>
  </g>
</g>
<text x="358" y="72" fill="#9A5A12" font-size="14">→</text>
<g transform="translate(372,26)">
  <rect x="0" y="0" width="140" height="86" rx="8" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="1.5"/>
  <text x="70" y="16" fill="#6B645E" font-size="8">3 · slide the point</text>
  <text x="70" y="38" fill="#8A6D3B" font-size="11">31.5e22</text>
  <path d="M40 44 q 30 12 60 0" fill="none" stroke="#C93B3B" stroke-width="1.5" marker-end="url(#sl)"/>
  <defs><marker id="sl" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C93B3B"/></marker></defs>
  <text x="70" y="74" fill="#1a5c38" font-size="13">3.15e23</text>
</g>
<g transform="translate(120,132)">
  <rect x="0" y="0" width="280" height="52" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="140" y="20" fill="#1a5c38" font-size="10">your answer 3.15e23</text>
  <text x="140" y="40" fill="#276b45" font-size="10">the published figure 3.14e23  ✔</text>
</g>
<text x="260" y="204" fill="#8a3b3b" font-size="10">one line of multiplication landed on a real number from a real paper</text>
<text x="260" y="226" fill="#9A938A" font-size="9">the point slid one place left, so the power went one up — 31.5e22 and 3.15e23 are the same number</text>
<text x="260" y="246" fill="#9A938A" font-size="8">every step in this picture is school arithmetic: three multiplies, one add, one slide</text>
</g></svg>
%%%

Three hundred and fifteen thousand million million million steps. And here is the good part. **That run's own published total is 3.14e23.** You landed on a real number from a real paper. You used nothing but the two ideas from earlier on this page: read two lines off a card, then price each handshake.

%%% insight
Stop and enjoy this, because it is earned. People assume a number like that comes from a spreadsheet somebody in a lab owns. It comes from two numbers and one multiplication. You can now do it by hand, with only a pen, while someone else is still opening their laptop.
%%%

!!! c-info 📚
<b>Where this rule is written down.</b> The forward cost of about 2 steps per knob per token, and the training cost of about 6, come from Kaplan and colleagues, <i>Scaling Laws for Neural Language Models</i> (2020, arXiv 2001.08361). The counting is set out in its Table 1.<br>
<b>The rule in use:</b> Hoffmann and colleagues, <i>Training Compute-Optimal Large Language Models</i> (2022, arXiv 2203.15556 — the "Chinchilla" paper), prices every one of its runs with exactly this C ≈ 6ND.<br>
<b>One bookkeeping note:</b> the N in this rule can be counted with or without the word tables at the two ends of the model. Kaplan counts N <i>without</i> those tables; Chinchilla prices its runs with the <i>total</i>. Day 2 shows you both counts and when each one is right.
!!!

#### What about counting it the long way?
You could refuse the shortcut. You could take every knob rectangle in the model, apply the `2 × m × k × n` count from unit 3 to each one, and add up hundreds of lines. People do this. It takes a long time and it does work, and the honest result is worth knowing. **The item-by-item count usually lands within about a tenth of the one-line answer.**

A tenth. Ten percent. Our number spans twenty-three powers of ten, so being within a tenth is as good as being right. It is also a hundred times more precise than the decisions this number gets used for.

%%% demo id=c6long label="predict the gap, then reveal it"
predict: one line of multiplication, against a careful count of every rectangle in the model, one at a time. How far apart do you think they land — 2 times? 10 times? Or closer than you would expect?
code: one line:      C ≈ 6 × N × D
code: the long way:  add up 2 × m × k × n for every knob rectangle, every block
out: the two agree within roughly 10 percent for ordinary model shapes
out: the leftovers are the small parts: the bends, the re-centring steps, the biases
take: <b>Within about a tenth — and that is why nobody does the long way twice.</b> The leftovers are real but tiny. The professional move is not to chase them. It is to say "about 3e23, and my error bar is ten percent" out loud, so everybody knows what the number is worth.
%%%

#### Your turn: drag the two numbers
You met this explorer in unit 2, before you had the price. Now you have it, so the big number will make sense. **Predict first:** if you double N, what happens to the bill? Then try changing the sliders yourself and see whether the number agrees with you.

%%% viz src=../../viz/transformer-flops.html title="C ≈ 6ND explorer: drag N and D, watch the bill" caption="Part 1 is today's rule. Drag N (knobs) and D (tokens) and watch C ≈ 6ND move. Try doubling N, then try doubling D instead — the bill does the same thing both times. Part 2 is the memory bill, which we meet in unit 14."
%%%

You have the rule and you have felt it move. Now let's make sure you hold it the right way round. The most common mistake with it is not arithmetic — it is a picture.

@@@ concept id=c7 tag="The bill is an area" title="Six is per handshake — so the bill is an area, not a side" gotit="Got the area"
Imagine tiling a bathroom floor. Tiles cost 6 each. Somebody asks what the floor will cost, and you answer "six". That is not an answer, is it? You need *both* sides of the room. Six is the price of one tile. The bill is the price times the **area**.

This is the single most common slip with today's rule. The 6 is not the price of a model. It is not the price of a token. It is the price of one **handshake** — one knob meeting one token. N is one side of the floor. D is the other side. The bill is the whole rectangle. Get that picture right and you will never mix the numbers up again.

%%% svg
<svg viewBox="0 0 520 235" role="img" aria-label="A tiled floor as the picture for the training bill. On the left a rectangle four tiles wide by three tiles tall, each tile stamped with a 6, labelled N knobs along the bottom and D tokens up the side, with a total of twelve tiles times six equals seventy two. On the right the same floor with the width doubled to eight tiles, giving twenty four tiles and a total of one hundred and forty four. A caption says: double either side and the bill doubles.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Each tile costs 6. The bill is the AREA — both sides matter equally.</text>
<g transform="translate(28,34)">
  <g fill="#FDF3E7" stroke="#C99A12" stroke-width="1.2">
    <rect x="0" y="0" width="34" height="34"/><rect x="34" y="0" width="34" height="34"/><rect x="68" y="0" width="34" height="34"/><rect x="102" y="0" width="34" height="34"/>
    <rect x="0" y="34" width="34" height="34"/><rect x="34" y="34" width="34" height="34"/><rect x="68" y="34" width="34" height="34"/><rect x="102" y="34" width="34" height="34"/>
    <rect x="0" y="68" width="34" height="34"/><rect x="34" y="68" width="34" height="34"/><rect x="68" y="68" width="34" height="34"/><rect x="102" y="68" width="34" height="34"/>
  </g>
  <g fill="#9A5A12" font-size="12"><text x="17" y="23">6</text><text x="51" y="23">6</text><text x="85" y="23">6</text><text x="119" y="23">6</text><text x="17" y="57">6</text><text x="51" y="57">6</text><text x="85" y="57">6</text><text x="119" y="57">6</text><text x="17" y="91">6</text><text x="51" y="91">6</text><text x="85" y="91">6</text><text x="119" y="91">6</text></g>
  <text x="68" y="120" fill="#276b45" font-size="9">N = 4 knobs</text>
  <text x="-52" y="52" fill="#5E5191" font-size="9" transform="rotate(-90 -52 52)">D = 3 tokens</text>
  <text x="68" y="140" fill="#2C2A28" font-size="10">12 handshakes × 6 = 72</text>
</g>
<g transform="translate(232,34)">
  <g fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.2">
    <rect x="0" y="0" width="34" height="34"/><rect x="34" y="0" width="34" height="34"/><rect x="68" y="0" width="34" height="34"/><rect x="102" y="0" width="34" height="34"/><rect x="136" y="0" width="34" height="34"/><rect x="170" y="0" width="34" height="34"/><rect x="204" y="0" width="34" height="34"/><rect x="238" y="0" width="34" height="34"/>
    <rect x="0" y="34" width="34" height="34"/><rect x="34" y="34" width="34" height="34"/><rect x="68" y="34" width="34" height="34"/><rect x="102" y="34" width="34" height="34"/><rect x="136" y="34" width="34" height="34"/><rect x="170" y="34" width="34" height="34"/><rect x="204" y="34" width="34" height="34"/><rect x="238" y="34" width="34" height="34"/>
    <rect x="0" y="68" width="34" height="34"/><rect x="34" y="68" width="34" height="34"/><rect x="68" y="68" width="34" height="34"/><rect x="102" y="68" width="34" height="34"/><rect x="136" y="68" width="34" height="34"/><rect x="170" y="68" width="34" height="34"/><rect x="204" y="68" width="34" height="34"/><rect x="238" y="68" width="34" height="34"/>
  </g>
  <text x="136" y="120" fill="#276b45" font-size="9">N doubled = 8 knobs</text>
  <text x="136" y="140" fill="#2C2A28" font-size="10">24 handshakes × 6 = 144</text>
</g>
<text x="260" y="196" fill="#8a3b3b" font-size="10">double ONE side → the bill doubles · double BOTH → four times the bill</text>
<text x="260" y="214" fill="#9A938A" font-size="9">"6" alone answers nothing, the same way "tiles cost 6" does not tell you what your floor costs</text>
<text x="260" y="228" fill="#9A938A" font-size="8">(day 5 meets this same rectangle again, with its area held fixed instead of growing)</text>
</g></svg>
%%%

**What the tiled-floor picture gets right:** the two sides multiply, so they matter equally. Nothing about the price per tile tells you the total. **Where the picture stops being true:** a real floor has a fixed size. A lab *chooses* both sides. Choosing them well is a whole art, and you will meet it on day 5.

#### Halve one side, double the other, and watch the area
Here is the swap that matters most. Halve N and double D, and the bill does not change at all. Not "roughly" — exactly. Half of one side times twice the other gives the same area back: `6 × (N ÷ 2) × (2 × D) = 6 × N × D`. The two halves of the change cancel, the same way cutting a floor's width in half and doubling its length leaves the same number of tiles.

%%% steps
step: start at N = 7 billion knobs, D = 1.4 trillion tokens
why: this is where the sliders begin. The bill comes out near 5.9e22 steps.
step: double N, leave D alone → the bill doubles
why: twice as many tiles across the floor. Nothing else changed.
step: double D, leave N alone → the bill doubles too
why: as far as the bill is concerned, the two sides swap freely. Text is exactly as expensive as knobs.
step: halve N and double D → the bill is unchanged
why: 6 × (N ÷ 2) × (2 × D) = 6 × N × D. The ÷2 and the ×2 cancel exactly. This is the trade every lab argues about, and day 5 is the argument.
%%%

#### Now drag the sides yourself
Reading about a rectangle is fine. Pulling its sides around is better. Go back to the explorer, and this time try that swap. **Predict first:** if you halve N and double D, what happens to the bill?

%%% viz src=../../viz/transformer-flops.html title="the same explorer: halve N, double D, watch the bill" caption="Halve the N slider and double the D slider. Half the width, twice the length: the same floor area, so the same bill. The sliders move in fixed steps, so the number on screen may wobble by a percent or two — that is the slider, not the rule."
%%%

%%% insight
This is why "how big is the model?" is only ever half a question. A small model that reads an enormous amount of text can cost more than a large model that reads a little. You saw that yourself back in unit 2, in the picture and in the numbers. Whenever somebody quotes only the model size, the honest reply is: *and how many tokens?*
%%%

Your rule is solid and you can feel its shape. Time for the first reward. The same rule prices the *serving* of a model, and serving turns out to be cheap.

@@@ concept id=c8 tag="Serving is cheaper" title="Answering costs a third of learning" gotit="Got the third"
Think about learning to spell a hard word. While practising, you do three things. You write it. Someone checks it. Then you fix what you got wrong. Later you write that word in a birthday card, and you do **one** of those three things. You just write it. Same word, a third of the effort.

A model answering your question is in exactly that position. It only does the forward pass — read, guess, done. No blame comes back, because nothing is being learned. The proper name for this is [[inference||A model answering, rather than learning: the forward pass only. Sometimes called serving. It costs about 2 steps per handshake instead of about 6.]]. People also just say *serving*. The price per handshake drops from about 6 to about **2**. Two out of six is one third.

%%% svg
<svg viewBox="0 0 520 205" role="img" aria-label="Two receipts side by side. The training receipt lists reading 2 and learning 4, total 6 per handshake. The serving receipt lists reading 2, with the learning line crossed out, total 2 per handshake. Between them a label reads: serving is about one third of the arithmetic. A caption notes that this is about arithmetic only.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Same knobs, same tokens — one of the two receipts is much shorter</text>
<g transform="translate(24,30)">
  <rect x="0" y="0" width="190" height="120" rx="8" fill="#FDFCFA" stroke="#C93B3B" stroke-width="2"/>
  <text x="95" y="20" fill="#8a3b3b" font-size="10">TRAINING (learning)</text>
  <g font-size="10" text-anchor="start" fill="#6B645E">
    <text x="16" y="44">reading</text><text x="170" y="44" text-anchor="end" fill="#2D8B55">2</text>
    <text x="16" y="66">learning</text><text x="170" y="66" text-anchor="end" fill="#C93B3B">4</text>
    <line x1="16" y1="76" x2="172" y2="76" stroke="#B8AEA2"/>
    <text x="16" y="96" fill="#2C2A28">per handshake</text><text x="170" y="96" text-anchor="end" fill="#2C2A28" font-size="12">6</text>
  </g>
</g>
<g transform="translate(306,30)">
  <rect x="0" y="0" width="190" height="120" rx="8" fill="#FDFCFA" stroke="#2D8B55" stroke-width="2"/>
  <text x="95" y="20" fill="#1a5c38" font-size="10">SERVING (answering)</text>
  <g font-size="10" text-anchor="start" fill="#6B645E">
    <text x="16" y="44">reading</text><text x="170" y="44" text-anchor="end" fill="#2D8B55">2</text>
    <text x="16" y="66" fill="#9A938A">learning</text><text x="170" y="66" text-anchor="end" fill="#9A938A">0</text>
    <line x1="14" y1="62" x2="174" y2="62" stroke="#C93B3B" stroke-width="1.5"/>
    <line x1="16" y1="76" x2="172" y2="76" stroke="#B8AEA2"/>
    <text x="16" y="96" fill="#2C2A28">per handshake</text><text x="170" y="96" text-anchor="end" fill="#2C2A28" font-size="12">2</text>
  </g>
</g>
<text x="260" y="96" fill="#9A5A12" font-size="11">→ ⅓ →</text>
<text x="260" y="172" fill="#276b45" font-size="10">2 out of 6 = one third of the arithmetic</text>
<text x="260" y="192" fill="#9A938A" font-size="8">careful: this compares ARITHMETIC only — day 3 shows what serving really waits on</text>
</g></svg>
%%%

**What the spelling picture gets right:** dropping the checking and the correcting really does leave one job out of three. That is the same ratio here. **What the picture gets wrong:** when you write a birthday card, the pen is the slow part. For a serving model the slow part is often not the arithmetic at all. It is fetching all those knobs out of memory, one token at a time. Day 3 is entirely about that surprise.

#### Price an answer
Same two numbers, one smaller price tag.

%%% steps
step: serving cost ≈ 2 × N × (tokens handled)
why: the 6 becomes a 2, because four of the six steps were the learning you are no longer doing.
step: a 7-billion-knob model writes a 500-token reply
why: 2 × 7e9 × 500. Front numbers: 2 × 7 × 5 = 70. Powers of ten: 9 + 2 = 11.
step: ≈ 7e12 steps of arithmetic for that whole reply
why: seven million million steps to write half a page. At 4e14 steps a second, that is under a twentieth of a second of pure arithmetic.
%%%

Those three lines are worth a picture, because the two bills are so far apart that words hide the gap. The left half draws the reply's little receipt. The right half puts the reply's bill next to the training bill, with **one fixed width for every digit**, so you can compare them by eye.

%%% svg
<svg viewBox="0 0 520 235" role="img" aria-label="Two panels comparing one reply with a whole training run. On the left, a small receipt builds the reply bill: two steps per handshake times seven billion knobs times five hundred tokens, fronts two times seven times five equal seventy, powers nine plus two equal eleven, giving seven times ten to the twelve. On the right, two bars drawn at ten pixels per digit from the same starting line: one reply at seven times ten to the twelve has thirteen digits and a bar one hundred and thirty long, and the whole training run at five point nine times ten to the twenty two has twenty three digits and a bar two hundred and thirty long, with a note that ten more digits means about ten thousand million times bigger.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">One reply's receipt, and how small it is next to the training run</text>
<g transform="translate(14,26)">
  <rect x="0" y="0" width="192" height="120" rx="8" fill="#FDFCFA" stroke="#2D8B55" stroke-width="2"/>
  <text x="96" y="18" fill="#1a5c38" font-size="9">ONE REPLY · 500 tokens</text>
  <g font-size="9" text-anchor="start" fill="#6B645E">
    <text x="14" y="40">price per handshake</text><text x="178" y="40" text-anchor="end" fill="#2D8B55">2</text>
    <text x="14" y="58">knobs N</text><text x="178" y="58" text-anchor="end">7e9</text>
    <text x="14" y="76">tokens in the reply</text><text x="178" y="76" text-anchor="end">500</text>
    <line x1="14" y1="86" x2="178" y2="86" stroke="#B8AEA2"/>
  </g>
  <text x="96" y="102" fill="#9A5A12" font-size="8">fronts 2 × 7 × 5 = 70 · powers 9 + 2 = 11</text>
  <text x="96" y="116" fill="#1a5c38" font-size="12">= 7e12 steps</text>
</g>
<g transform="translate(240,34)">
  <text x="130" y="0" fill="#6B645E" font-size="8">one fixed width per digit: 10 px each, same starting line</text>
  <line x1="0" y1="10" x2="0" y2="92" stroke="#3A342E" stroke-width="1.5"/>
  <rect x="0" y="14" width="130" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="65" y="31" fill="#1a5c38" font-size="8">one reply 7e12</text>
  <text x="150" y="31" fill="#276b45" font-size="8" text-anchor="start">13 digits</text>
  <rect x="0" y="56" width="230" height="26" rx="4" fill="#FDF3E7" stroke="#C93B3B" stroke-width="2"/>
  <text x="115" y="73" fill="#8a3b3b" font-size="8">the whole training run 5.9e22</text>
  <text x="238" y="73" fill="#8a3b3b" font-size="8" text-anchor="start">23</text>
  <text x="115" y="104" fill="#9A5A12" font-size="9">23 − 13 = ten more digits =</text>
  <text x="115" y="118" fill="#9A5A12" font-size="9">about ten thousand million times bigger</text>
</g>
<text x="260" y="176" fill="#276b45" font-size="10">so one answer is cheap · but the answers never stop coming</text>
<text x="260" y="198" fill="#9A938A" font-size="9">if a million people each ask one question, that reply bar has to be drawn a million times</text>
<text x="260" y="220" fill="#9A938A" font-size="8">the training bar is drawn once, ever — this is the argument day 5 settles</text>
</g></svg>
%%%

%%% insight
Now hold two facts side by side. Training that model happened once and cost about 5.9e22 steps. Answering one question costs about 7e12. That is ten thousand million times smaller. And yet the serving bill is the one that arrives *every single day, forever*. Day 5 is where those two facts have their argument.
%%%

%%% demo id=c8third label="predict the ratio, then reveal it"
predict: reading is 2 steps per handshake and training is 6. If a lab spends one day of arithmetic training a model, how much arithmetic would the same handshakes cost if it were only answering?
code: training per handshake = 6
code: serving  per handshake = 2
code: ratio = 2 ÷ 6
out: 1/3 — about a third of the arithmetic
out: said the other way: training is about 3 times the arithmetic of the same reading done once
take: <b>A third — and no design detail could change it.</b> The ratio is 2 to 6 for every model that trains this way, whatever its shape. This kind of fact survives new model designs, which is why it is worth memorising.
%%%

You can now price arithmetic in both directions. But nobody asks "how many steps?" They ask **"how long will it take?"** Let's turn steps into hours.

@@@ concept id=c9 tag="Steps into hours" title="Divide by the rate — and let many chips share the wait" gotit="Got the hours"
You are filling a bath. The bath holds 200 litres. The tap gives 10 litres a minute. How long? You divide: 20 minutes. Nobody taught you a special formula for baths. Amount divided by rate gives time, and that is all there is to it.

Steps of arithmetic work the same way. Your bath is the count C. Your tap is the **chip**, and its rate is how many steps it really gets through in one second. Divide, and you have seconds. And if you open several taps at once, the bath fills sooner. That is exactly why training runs use many chips instead of one.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="A bath filling from taps as the picture for turning steps into time. On the left one tap labelled four times ten to the fourteen steps per second fills a bath labelled three point one five times ten to the twenty three steps, with a stopwatch reading twenty five years. On the right, one thousand taps fill the same bath and the stopwatch reads about nine days. A caption says: amount divided by rate gives time, and more taps divide the waiting.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">amount ÷ rate = time · more taps → less waiting</text>
<g transform="translate(20,32)">
  <rect x="26" y="0" width="14" height="26" fill="#B8AEA2" stroke="#3A342E"/>
  <path d="M33 26 L33 40" stroke="#5E5191" stroke-width="3"/>
  <text x="33" y="-4" fill="#5E5191" font-size="8">1 tap</text>
  <rect x="0" y="44" width="140" height="70" rx="6" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/>
  <text x="70" y="70" fill="#5E5191" font-size="9">bath = 3.15e23</text>
  <text x="70" y="84" fill="#5E5191" font-size="8">steps of arithmetic</text>
  <text x="70" y="102" fill="#9A5A12" font-size="8">tap = 4e14 steps/sec</text>
  <g transform="translate(48,120)"><circle cx="22" cy="22" r="20" fill="#FDFCFA" stroke="#3A342E" stroke-width="2"/><path d="M22 22 L22 8" stroke="#C93B3B" stroke-width="2"/><path d="M22 22 L33 28" stroke="#3A342E" stroke-width="2"/></g>
  <text x="70" y="180" fill="#8a3b3b" font-size="10">≈ 25 years</text>
</g>
<text x="185" y="112" fill="#9A5A12" font-size="14">→</text>
<g transform="translate(240,32)">
  <g stroke="#3A342E" fill="#B8AEA2">
    <rect x="20" y="0" width="10" height="20"/><rect x="42" y="0" width="10" height="20"/><rect x="64" y="0" width="10" height="20"/><rect x="86" y="0" width="10" height="20"/><rect x="108" y="0" width="10" height="20"/><rect x="130" y="0" width="10" height="20"/><rect x="152" y="0" width="10" height="20"/><rect x="174" y="0" width="10" height="20"/><rect x="196" y="0" width="10" height="20"/><rect x="218" y="0" width="10" height="20"/>
  </g>
  <g stroke="#5E5191" stroke-width="2"><path d="M25 20 v20"/><path d="M47 20 v20"/><path d="M69 20 v20"/><path d="M91 20 v20"/><path d="M113 20 v20"/><path d="M135 20 v20"/><path d="M157 20 v20"/><path d="M179 20 v20"/><path d="M201 20 v20"/><path d="M223 20 v20"/></g>
  <text x="124" y="-4" fill="#5E5191" font-size="8">1 000 chips, all pouring at once</text>
  <rect x="0" y="44" width="248" height="70" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="124" y="70" fill="#1a5c38" font-size="9">same bath = 3.15e23 steps</text>
  <text x="124" y="88" fill="#276b45" font-size="8">combined rate = 1000 × 4e14 = 4e17 steps/sec</text>
  <g transform="translate(102,120)"><circle cx="22" cy="22" r="20" fill="#FDFCFA" stroke="#3A342E" stroke-width="2"/><path d="M22 22 L30 10" stroke="#2D8B55" stroke-width="2"/><path d="M22 22 L33 28" stroke="#3A342E" stroke-width="2"/></g>
  <text x="124" y="180" fill="#276b45" font-size="10">≈ 9 days</text>
</g>
<text x="260" y="212" fill="#9A938A" font-size="9">the bath never gets smaller — you only ever open more taps</text>
</g></svg>
%%%

**What the bath picture gets right:** dividing an amount by a rate really does give a time. Taps really do add up when you open several. **Where the picture stops being true:** taps do not have to talk to each other. Chips do. They stop to share their notes of blame, so a thousand chips give a little less than a thousand taps' worth of speed. Day 5 and m09c measure that loss. Today we take the clean division.

#### One chip, then a thousand
Borrow one number for a moment. A modern training chip really gets through about **4e14 steps a second**. The next unit is entirely about where that number comes from, and why it is not the number printed on the box.

%%% steps
step: seconds on ONE chip = 3.15e23 ÷ 4e14
why: front numbers 3.15 ÷ 4 = 0.79. Powers of ten: 23 − 14 = 9. When you divide, the powers of ten subtract.
step: = 0.79e9 = 7.9e8 seconds
why: seven hundred and ninety million seconds. A year is about 3.15e7 seconds, so divide again: 7.9e8 ÷ 3.15e7 ≈ 25.
step: ≈ 25 years on one chip
why: which is why nobody trains a model on one chip. This number exists to make you reach for more taps.
step: on 1 000 chips: 7.9e8 ÷ 1 000 = 7.9e5 seconds
why: divide by the number of chips. An hour is 3 600 seconds and a day is 86 400.
step: 7.9e5 ÷ 86 400 ≈ 9 days of waiting
why: on today's chips that famous run would take about nine days. The real 2020 run used much slower chips, so it actually took longer — but the arithmetic you just did is the arithmetic everyone uses.
%%%

#### The word for renting time: chip-hours
Labs rarely say "9 days on 1 000 chips". They say **chip-hours**: how many hours of one chip's attention the whole run needed. Careful here — this one starts from the **one-chip** seconds, not from the many-chip seconds.

%%% steps
step: 7.9e8 seconds ON ONE CHIP ÷ 3 600 seconds per hour
why: 7.9e8 ÷ 3.6e3 → front numbers 7.9 ÷ 3.6 ≈ 2.2, powers of ten 8 − 3 = 5.
step: ≈ 2.2e5 chip-hours, that is about 220 000 chip-hours
why: the same bath, measured in hours-of-one-tap instead of days-of-many-taps. Both describe one run.
step: the waiting is a different number: 7.9e5 seconds ÷ 3 600 ≈ 219 hours
why: 219 hours is about 9 days — the time you actually wait. Chip-hours is 1 000 times bigger, because 1 000 chips were each busy for those 219 hours.
%%%

#### The five divisions, drawn as one chain
That is the busiest arithmetic on the page, so here it is as a picture. Every arrow carries the number you divide by, and every box says whether it holds a **count**, a **rate** or a **time**.

%%% svg
<svg viewBox="0 0 520 300" role="img" aria-label="The whole unit conversion drawn as a branching chain. At the top a box holds the bill three point one five times ten to the twenty three steps, marked a count. An arrow labelled divide by four times ten to the fourteen steps per second, marked a rate, leads down to a box holding seven point nine times ten to the eight seconds on one chip. From that box three arrows branch out. The left arrow, divide by three point one five times ten to the seven seconds per year, reaches about twenty five years on one chip. The middle arrow, divide by one thousand chips, reaches seven point nine times ten to the five seconds, and a further arrow, divide by eighty six thousand four hundred seconds per day, reaches about nine days of waiting. The right arrow, divide by three thousand six hundred seconds per hour, reaches about two point two times ten to the five chip hours. A footer says a count divided by a rate gives seconds, and a count divided by a count gives nonsense.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="14" fill="#2C2A28" font-size="12">One chain, five divisions — watch the units change at every arrow</text>
<g transform="translate(150,22)">
  <rect x="0" y="0" width="220" height="34" rx="6" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/>
  <text x="110" y="16" fill="#5E5191" font-size="10">the bill  C = 3.15e23 steps</text>
  <text x="110" y="28" fill="#5E5191" font-size="8">a COUNT of steps</text>
</g>
<path d="M260 58 V 78" stroke="#C99A12" stroke-width="2" marker-end="url(#c9a)"/>
<defs><marker id="c9a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C99A12"/></marker></defs>
<text x="268" y="72" fill="#9A5A12" font-size="8" text-anchor="start">÷ 4e14 steps per second — a RATE</text>
<g transform="translate(150,82)">
  <rect x="0" y="0" width="220" height="34" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="110" y="16" fill="#1a5c38" font-size="10">7.9e8 seconds on ONE chip</text>
  <text x="110" y="28" fill="#276b45" font-size="8">a TIME — every branch below starts here</text>
</g>
<path d="M180 116 Q 110 130 88 158" fill="none" stroke="#5E5191" stroke-width="1.8" marker-end="url(#c9b)"/>
<defs><marker id="c9b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5E5191"/></marker></defs>
<text x="6" y="140" fill="#5E5191" font-size="8" text-anchor="start">÷ 3.15e7 sec per year</text>
<g transform="translate(8,160)">
  <rect x="0" y="0" width="150" height="38" rx="6" fill="#FDFCFA" stroke="#5E5191" stroke-width="2"/>
  <text x="75" y="16" fill="#5E5191" font-size="10">≈ 25 years</text>
  <text x="75" y="30" fill="#6B645E" font-size="8">one chip alone</text>
</g>
<path d="M260 116 V 158" stroke="#2D8B55" stroke-width="1.8" marker-end="url(#c9c)"/>
<defs><marker id="c9c" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<text x="266" y="140" fill="#276b45" font-size="8" text-anchor="start">÷ 1 000 chips</text>
<g transform="translate(185,160)">
  <rect x="0" y="0" width="150" height="34" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="75" y="15" fill="#1a5c38" font-size="10">7.9e5 seconds</text>
  <text x="75" y="28" fill="#276b45" font-size="8">the waiting, in seconds</text>
</g>
<path d="M260 194 V 226" stroke="#2D8B55" stroke-width="1.8" marker-end="url(#c9d)"/>
<defs><marker id="c9d" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#2D8B55"/></marker></defs>
<text x="266" y="214" fill="#276b45" font-size="8" text-anchor="start">÷ 86 400 sec per day</text>
<g transform="translate(185,228)">
  <rect x="0" y="0" width="150" height="38" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2.5"/>
  <text x="75" y="16" fill="#1a5c38" font-size="11">≈ 9 days</text>
  <text x="75" y="30" fill="#276b45" font-size="8">what you actually wait</text>
</g>
<path d="M340 116 Q 420 130 440 158" fill="none" stroke="#C93B3B" stroke-width="1.8" marker-end="url(#c9e)"/>
<defs><marker id="c9e" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C93B3B"/></marker></defs>
<text x="356" y="140" fill="#8a3b3b" font-size="8" text-anchor="start">÷ 3 600 sec per hour</text>
<g transform="translate(362,160)">
  <rect x="0" y="0" width="152" height="38" rx="6" fill="#FDF3E7" stroke="#C93B3B" stroke-width="2"/>
  <text x="76" y="16" fill="#8a3b3b" font-size="10">≈ 2.2e5 chip-hours</text>
  <text x="76" y="30" fill="#8a3b3b" font-size="8">from the ONE-chip seconds</text>
</g>
<g transform="translate(96,274)">
  <rect x="0" y="0" width="328" height="22" rx="6" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="1.5"/>
  <text x="164" y="15" fill="#3A342E" font-size="9">count ÷ rate = seconds ✔ · count ÷ count = nonsense ✘</text>
</g>
</g></svg>
%%%

!!! c-info 🔬
<b>Optional (skippable) — the two words that sound the same.</b><br>
<b>A count:</b> steps of arithmetic, written FLOPs with a small s for plural. That is the size of the bath. 3.15e23 FLOPs.<br>
<b>A rate:</b> steps per second, written FLOP/s or FLOPS. That is the tap. 4e14 FLOP/s.<br>
<b>Why it matters:</b> a count divided by a rate gives seconds. A count divided by a count gives nonsense.<br>
<b>And it is silent:</b> no error message appears, just a wrong answer. Say the unit out loud with every number and you will never do it.
!!!

%%% insight
Both jobs are now in your hands. Price the arithmetic, then turn it into a wait. Only one thing stands between you and a real estimate: the tap's rate. And that number is where nearly everybody's estimate goes wrong, in a very specific and very avoidable way.
%%%

%%% demo id=c9days label="predict the days, then reveal them"
predict: the 7-billion-knob run needs about 5.9e22 steps. On 1 000 chips at 4e14 steps a second each — days or months?
code: combined rate = 1 000 × 4e14        = 4e17 steps/sec
code: seconds       = 5.9e22 ÷ 4e17
code: days          = seconds ÷ 86 400
out: seconds ≈ 1.5e5
out: days    ≈ 1.7
out: about a day and a half
take: <b>Under two days.</b> You now do in three lines what people imagine needs a big planning tool. And notice the shape of the arithmetic: divide by the rate, divide by the chips, divide by 86 400. Three divisions, all done left to right by hand.
%%%

Now the best puzzle of the day. Careful people divide correctly and still come out two to five times too fast. Why?

@@@ concept id=c10 tag="Sticker vs stopwatch" title="The speed on the box is not the speed you get" gotit="Got the two haircuts"
Every printer box says something like "prints 20 pages a minute". Time it with a stopwatch and you get eight. The box was not exactly lying. It measured one page of plain text, on one paper size, with the printer already warm. Your actual pages had pictures on them. Sticker speed and stopwatch speed are two different things, and the gap is not small.

Chips come with the same sticker, and learning to read it is the most valuable ten minutes of the day. The number in the maker's fact sheet is a **peak**. It is what the arithmetic units could manage if every single tick were perfect. A real training run never gets there. So we take the sticker number and give it two **haircuts** — two cuts that make it smaller — before dividing by it.

%%% svg
<svg viewBox="0 0 520 225" role="img" aria-label="A boxed chip with a sticker next to a speedometer. The sticker says 2000 trillion steps per second with a star, and the small print says star, shown with a special trick, halve it for ordinary work. An arrow leads to a speedometer whose needle sits two fifths of the way round, labelled MFU zero point four, the share of top speed a real run reaches. The final figure is 400 trillion steps per second. A caption says: haircut one removes the trick, haircut two is the share you actually reach.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Two haircuts turn the sticker number into a number you can divide by</text>
<g transform="translate(16,34)">
  <rect x="0" y="0" width="150" height="96" rx="8" fill="#EDE9E0" stroke="#3A342E" stroke-width="2"/>
  <rect x="12" y="12" width="126" height="46" rx="4" fill="#FDF3E7" stroke="#C93B3B" stroke-width="2"/>
  <text x="75" y="30" fill="#8a3b3b" font-size="10">2 000 trillion</text>
  <text x="75" y="44" fill="#8a3b3b" font-size="9">steps / second *</text>
  <text x="75" y="74" fill="#6B645E" font-size="8">the sticker on the box</text>
  <text x="75" y="88" fill="#9A938A" font-size="7">* shown with a special trick</text>
</g>
<text x="182" y="82" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(196,34)">
  <rect x="0" y="0" width="130" height="96" rx="8" fill="#FDFCFA" stroke="#C99A12" stroke-width="2"/>
  <text x="65" y="22" fill="#9A5A12" font-size="9">haircut 1: halve it</text>
  <text x="65" y="48" fill="#8A6D3B" font-size="11">1 000 trillion</text>
  <text x="65" y="62" fill="#8A6D3B" font-size="9">steps / second</text>
  <text x="65" y="84" fill="#6B645E" font-size="7">honest top speed for ordinary work</text>
</g>
<text x="340" y="82" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(356,30)">
  <path d="M10 78 A 55 55 0 0 1 120 78" fill="none" stroke="#B8AEA2" stroke-width="8"/>
  <path d="M10 78 A 55 55 0 0 1 43 29" fill="none" stroke="#2D8B55" stroke-width="8"/>
  <circle cx="65" cy="78" r="5" fill="#3A342E"/>
  <path d="M65 78 L40 34" stroke="#C93B3B" stroke-width="2.5"/>
  <text x="65" y="98" fill="#1a5c38" font-size="9">MFU = 0.4</text>
  <text x="65" y="112" fill="#276b45" font-size="8">the share you reach</text>
  <text x="65" y="18" fill="#9A938A" font-size="7">haircut 2</text>
</g>
<g transform="translate(150,150)">
  <rect x="0" y="0" width="220" height="34" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="110" y="22" fill="#1a5c38" font-size="13">400 trillion steps / second</text>
</g>
<text x="260" y="204" fill="#9A938A" font-size="9">4e14 — the number we borrowed in the last unit, now earned</text>
<text x="260" y="218" fill="#9A938A" font-size="8">divide by THIS, never by the sticker</text>
</g></svg>
%%%

**What the printer picture gets right:** the sticker measures an ideal case nobody meets. Timing it yourself gives a different and truer number. **What the picture gets wrong:** a printer's gap is mostly the paper. A chip's gap has several causes at once. And one of those causes is a footnote rather than a fact of life.

#### Haircut one: read the small print
The first haircut is not about reality at all. It is about advertising. The headline figure on a modern chip's page often carries a little star. The star says *shown with [[sparsity||A special trick that only counts when half of your numbers are zeros in a set pattern. Ordinary training does not do that, so halve any figure that mentions it.]]* — and sparsity just means "lots of zeros". That trick only counts if half your numbers are zeros in a set pattern. Ordinary training does not do that. So the honest top speed for ordinary work is **half the headline**.

%%% steps
step: the fact sheet headline: about 2 000 trillion steps per second
why: this is what a modern training chip's page advertises for the number format training uses.
step: find the star, read the small print: "shown with sparsity"
why: it means the figure assumes a trick that ordinary training does not use. The same page usually prints the plain figure right beside it, at half the size.
step: honest top speed ≈ 1 000 trillion = 1e15 steps per second
why: haircut one, done. You have not touched reality yet. You have only stopped quoting an advertisement.
%%%

#### Haircut two: the share you actually reach
Now reality. Even at its best, a run does not keep every arithmetic unit busy every tick. It waits for text to arrive. It stops so the chips can share their notes of blame. Its inner programs are good but not perfect. The share of top speed a run actually reaches has one name, and you should use that name: [[MFU (Model FLOPs Utilization)||Model FLOPs Utilization — the share of a chip's honest top speed that a real training run actually reaches. Measured, not assumed. Real runs land around 0.3 to 0.55.]]. People say the three letters out loud: em-eff-you.

%%% steps
step: MFU = the steps you really got, divided by the top speed
why: a share, so it is a number between 0 and 1. Nothing exotic: it is the printer's stopwatch reading divided by its sticker.
step: real runs land between about 0.3 and 0.55
why: measured on real clusters — a cluster is just the whole group of chips a run uses, wired together. We use 0.4 all day, which is a fair middle.
step: 1e15 × 0.4 = 4e14 steps per second
why: haircut two, done. This is the rate you divide by, and it is the number we borrowed in unit 9, now earned.
%%%

Two small cuts, and the borrowed number from unit 9 is now a number you built yourself. That is the last missing piece of a real estimate — from here on you are only checking your own work.

!!! c-info 🔬
<b>Optional (skippable) — real chips, real numbers, all in one number format.</b><br>
<b>An older workhorse chip:</b> about 312 trillion honest steps a second for training-sized numbers — and about 624 trillion if you let it count the sparsity trick.<br>
<b>Its replacement, which is the chip in this unit:</b> about 989 trillion honest, and about 1 979 trillion with the sparsity trick. Round those two and you get the 1 000 trillion and 2 000 trillion we have been working with. Same chip, both figures.<br>
<b>Keep one format:</b> a chip also prints smaller figures for other kinds of number. Check you are reading the format your run trains in before you compare two chips, or you will halve something that was already halved.<br>
<b>What to do with this:</b> nothing to memorise. Look up your own chip, find the star, halve it if it is there.
!!!

%%% demo id=c10sticker label="predict how wrong the sticker is, then reveal it"
predict: you price the 3.15e23-step run on 1 000 chips, but you divide by the sticker's 2 000 trillion. Your answer will be too fast — but by how much? Two times? Five times?
code: with the sticker: 3.15e23 ÷ (1 000 × 2e15) = 1.6e5 sec
code: after 2 haircuts: 3.15e23 ÷ (1 000 × 4e14) = 7.9e5 sec
code: ratio = 2e15 ÷ 4e14
out: sticker answer:  about 1.8 days
out: honest answer:   about 9 days
out: 5 times too fast
take: <b>Five times too fast — and every hour of that gap is real money.</b> Notice how the two haircuts multiply: a half times a 0.4 gives a fifth. This one habit, applied before you speak, is worth more than any other detail today.
%%%

You just found a five-times error with one halving and one multiplication. That is the whole trick, and it is now yours. Here it is written up as the failure to watch for.

!!! c-warn ⚠️
<b>Failure mode (no error message, just a wrong answer):</b> you divide by the number printed on the box, and promise 2 days for a run that takes 9.<br>
<b>Cause:</b> the sticker may include the sparsity trick, and it always assumes every tick is used perfectly. Neither is true for your run.<br>
<b>Remedy 1:</b> halve the headline if it carries the sparsity star.<br>
<b>Remedy 2:</b> multiply by an MFU you have <i>measured</i> on the actual group of chips — day 4 reads it straight off a training log.<br>
<b>Remedy 3:</b> quote a <b>range</b>. With MFU between 0.3 and 0.55, our run lands between about 7 and 12 days, so "somewhere between 7 and 12 days" is the honest shape of the answer.
!!!

%%% insight
This is a story about reading, not about arithmetic. The division was never the hard part. The hard part is noticing a star in tiny print and asking "what does this footnote assume?" That habit is most of what separates an estimate people trust from one they quietly re-do themselves.
%%%

Two haircuts on the rate, handled. Now a correction that goes the other way — something to *add* to the count, because there is one kind of work the six quietly leaves out.

@@@ concept id=c11 tag="The width test" title="What the six leaves out: tokens comparing each other" gotit="Got the width test"
On the first day of school your teacher hands out a getting-to-know-you grid. Every child's name runs down the side, and the same names run across the top. You fill in one box for every name, including your own. With 4 children in the room that is 4 × 4 = 16 boxes. Add four more children — twice as many — and the grid becomes 8 × 8 = 64 boxes. The children doubled. The boxes went up **four times**.

A model fills in a grid like that. Inside it, every token in the stretch of text looks at every other token to decide who matters. That looking-at-each-other work uses **no knobs at all** — it is token against token. So our six-per-handshake bill never counted it. And it grows like the grid: double the length of the text, and roughly four times as much comparing. The good part is that one small division tells you whether you can safely ignore it.

%%% svg
<svg viewBox="0 0 520 235" role="img" aria-label="Two getting-to-know-you grids. On the left a four by four grid with all sixteen boxes filled, labelled four tokens, sixteen look-ups. On the right an eight by eight grid, labelled eight tokens, sixty four look-ups, four times as many boxes. A banner underneath gives the width test: divide seq_len by twelve times d_model, and if the share is under a tenth the comparing work is small enough to ignore.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Every token fills in a box for every token — one whole grid</text>
<g transform="translate(46,32)">
  <g fill="#F4F1FB" stroke="#5E5191" stroke-width="1">
    <rect x="0" y="0" width="26" height="26"/><rect x="26" y="0" width="26" height="26"/><rect x="52" y="0" width="26" height="26"/><rect x="78" y="0" width="26" height="26"/>
    <rect x="0" y="26" width="26" height="26"/><rect x="26" y="26" width="26" height="26"/><rect x="52" y="26" width="26" height="26"/><rect x="78" y="26" width="26" height="26"/>
    <rect x="0" y="52" width="26" height="26"/><rect x="26" y="52" width="26" height="26"/><rect x="52" y="52" width="26" height="26"/><rect x="78" y="52" width="26" height="26"/>
    <rect x="0" y="78" width="26" height="26"/><rect x="26" y="78" width="26" height="26"/><rect x="52" y="78" width="26" height="26"/><rect x="78" y="78" width="26" height="26"/>
  </g>
  <rect x="0" y="0" width="52" height="52" fill="none" stroke="#5E5191" stroke-width="1.5"/>
  <g fill="#5E5191" opacity="0.55"><rect x="1" y="1" width="11" height="11"/><rect x="14" y="1" width="11" height="11"/><rect x="27" y="1" width="11" height="11"/><rect x="40" y="1" width="11" height="11"/><rect x="1" y="14" width="11" height="11"/><rect x="14" y="14" width="11" height="11"/><rect x="27" y="14" width="11" height="11"/><rect x="40" y="14" width="11" height="11"/><rect x="1" y="27" width="11" height="11"/><rect x="14" y="27" width="11" height="11"/><rect x="27" y="27" width="11" height="11"/><rect x="40" y="27" width="11" height="11"/><rect x="1" y="40" width="11" height="11"/><rect x="14" y="40" width="11" height="11"/><rect x="27" y="40" width="11" height="11"/><rect x="40" y="40" width="11" height="11"/></g>
  <g fill="none" stroke="#FDFCFA" stroke-width="0.8"><line x1="13" y1="0" x2="13" y2="52"/><line x1="26" y1="0" x2="26" y2="52"/><line x1="39" y1="0" x2="39" y2="52"/><line x1="0" y1="13" x2="52" y2="13"/><line x1="0" y1="26" x2="52" y2="26"/><line x1="0" y1="39" x2="52" y2="39"/></g>
  <text x="26" y="70" fill="#5E5191" font-size="9">4 tokens → 4 × 4 = 16 look-ups</text>
</g>
<text x="185" y="86" fill="#9A5A12" font-size="12">→</text>
<g transform="translate(240,32)">
  <rect x="0" y="0" width="104" height="104" fill="#5E5191" opacity="0.55"/>
  <g fill="none" stroke="#FDFCFA" stroke-width="0.8">
    <line x1="13" y1="0" x2="13" y2="104"/><line x1="26" y1="0" x2="26" y2="104"/><line x1="39" y1="0" x2="39" y2="104"/><line x1="52" y1="0" x2="52" y2="104"/><line x1="65" y1="0" x2="65" y2="104"/><line x1="78" y1="0" x2="78" y2="104"/><line x1="91" y1="0" x2="91" y2="104"/>
    <line x1="0" y1="13" x2="104" y2="13"/><line x1="0" y1="26" x2="104" y2="26"/><line x1="0" y1="39" x2="104" y2="39"/><line x1="0" y1="52" x2="104" y2="52"/><line x1="0" y1="65" x2="104" y2="65"/><line x1="0" y1="78" x2="104" y2="78"/><line x1="0" y1="91" x2="104" y2="91"/>
  </g>
  <rect x="0" y="0" width="104" height="104" fill="none" stroke="#5E5191" stroke-width="1.5"/>
  <text x="52" y="122" fill="#5E5191" font-size="9">8 tokens → 8 × 8 = 64 look-ups</text>
</g>
<text x="260" y="176" fill="#8a3b3b" font-size="10">twice the tokens → four times the comparing</text>
<g transform="translate(76,190)">
  <rect x="0" y="0" width="368" height="34" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/>
  <text x="184" y="15" fill="#8A6D3B" font-size="9">THE WIDTH TEST: share = seq_len ÷ (12 × d_model)</text>
  <text x="184" y="28" fill="#9A5A12" font-size="8">share under a tenth → small enough to ignore · over a tenth → say 6ND is too low</text>
</g>
</g></svg>
%%%

**What the class-grid picture gets right:** the number of boxes grows far faster than the number of children. That is exactly the shape of this work. **Where the picture stops being true:** the class grid is the whole morning's task. Inside a model this comparing sits *alongside* the knob work. It only sometimes matters. So the useful question is not "how big is it?" but "is it big enough to care?"

#### The width test: one division, one decision
You do not need the exact size of this extra work today — day 3 measures it properly. You need one check. It uses two numbers you can read off any model's description.

%%% steps
step: seq_len — how many tokens the model handles in one stretch
why: often 2 048, 4 096 or 8 192. It is the length of the text the model sees at once.
step: d_model — the width of one token inside the model
why: the number of values used to describe one token. m03 met this as the token's width; a common value is 4 096.
step: work out 12 × d_model
why: with d_model = 4 096 that is 49 152. One multiplication. This is the length at which the comparing work would equal ALL the knob work.
step: now divide: share = seq_len ÷ (12 × d_model)
why: with seq_len = 2 048 that is 2 048 ÷ 49 152 ≈ 0.04, so about 4 percent.
step: share under a tenth? → ignore the comparing work
why: 4 percent sits inside the ten percent wobble the six already has. So 6ND is fine, and you can say why.
step: share over a tenth? → 6ND is undercounting, and you must say so
why: at seq_len = 8 192 the share is about 17 percent — too big to ignore. At the full 49 152 the comparing equals all the knob work, so your estimate would be roughly half of the truth.
%%%

!!! c-info 📚
<b>Where the twelve comes from.</b> It is not invented. Kaplan and colleagues (2020, arXiv 2001.08361) write their forward cost as the knob term plus a comparing term. They then state plainly that they drop the comparing term, because for their models the width is far bigger than the text length divided by twelve. The width test is that sentence, turned into one division you can do by hand.
!!!

%%% demo id=c11len label="predict the growth, then reveal it"
predict: you keep the model the same and double the length of the text it reads at once, from 2 048 to 4 096 tokens. The knob work per token stays the same. What happens to the comparing work?
code: comparing work grows with seq_len × seq_len
code: 2 048 × 2 048 = 4.2e6 look-ups   (the whole grid, every token against every token)
code: 4 096 × 4 096 = 1.7e7 look-ups
out: about 4 times as many look-ups
out: share of the knob work: 4% → 8%
take: <b>Four times the comparing, from one doubling.</b> Watch the share climb: 4 percent, then 8, then 17. Each doubling doubles the share. So a term you were right to ignore becomes the main event after a few doublings. That is why the width test is a check you re-run, not a fact you memorise.
%%%

One division, and you now know whether you are allowed to ignore a whole piece of a model. That is a small, cheap superpower. Here is the failure it protects you from.

!!! c-warn ⚠️
<b>Failure mode:</b> you price a long-text run with plain 6ND and come in far too low.<br>
<b>Cause:</b> the six counts knob work only, so the token-against-token comparing is invisible to it. And that comparing grows with the <i>square</i> of the text length, so it stops being small quickly.<br>
<b>Remedy:</b> run the width test <i>before</i> you trust 6ND. If the share is over a tenth, say out loud "6ND is too low here — a smallest-possible answer, not an estimate", then go and get the comparing term.
!!!

%%% insight
Notice what the width test really is: permission to ignore something, granted with a reason. That is a different thing from not knowing about it. "I checked and it is under five percent" is a professional sentence. "I forgot about attention" is not. And both can produce the same number on the same day.
%%%

!!! c-info 🔬
<b>Optional (skippable) — where to read those two numbers.</b><br>
<b>In a model's description file</b> (often called <code>config.json</code>) the width is usually the field <code>hidden_size</code>, and the stretch length is <code>max_position_embeddings</code>.<br>
<b>Different words, same two numbers.</b> Papers say "d_model" and "context length"; code often says "hidden size" and "sequence length".<br>
<b>One more counting habit:</b> in a real model each token only looks at the tokens BEFORE it, so only about half the grid gets filled. The published counts usually take the whole grid anyway, which makes them a little generous. Day 3 opens this up.
!!!

Want to see long text cost you twice? In the explorer's Part 2, drag the sequence-length slider and watch the memory bar climb. Long text charges you on **both** bills, and unit 14 is about the second one.

Two traps met and beaten. Now a reward, and it is the most useful habit of the whole day: a way to catch your own mistakes before anybody else does.

@@@ concept id=c12 tag="Check the size" title="Count the zeros before you trust the answer" gotit="Got the size check"
You are in a shop and a banana has a price tag reading **£340.00**. You do not need to know what bananas cost to know that tag is wrong. Something slipped — a decimal point, most likely. You caught it in half a second. You used nothing but a rough sense of *how many digits a banana ought to have*.

That is the most valuable habit of this whole day, and professionals use it every single time. A multiplication never complains. Type the wrong number of zeros and it hands you a confident, tidy, completely wrong answer with no warning at all. The only defence is a rough size you already carry in your head.

%%% svg
<svg viewBox="0 0 520 230" role="img" aria-label="Two price tags on a banana as the picture for the size check. The first tag reads three pounds forty and is ticked. The second reads three hundred and forty pounds and is crossed out with the note wrong number of digits, caught in half a second. Below, a shelf holds two anchor items whose prices you know by heart: a run with 175 billion knobs at 3e23 steps, and a run with 7 billion knobs at 6e22 steps. A caption says: you judge a new number by how far it sits from an anchor you already know.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">You spot a wrong price by its number of digits, not by knowing the right price</text>
<g transform="translate(40,30)">
  <path d="M20 60 q 20 -40 54 -44 q -6 16 -14 24 q 12 22 -6 36 q -20 14 -34 -16 z" fill="#F5D76E" stroke="#9A5A12" stroke-width="1.5"/>
  <g transform="translate(84,16)"><rect x="0" y="0" width="86" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="43" y="20" fill="#1a5c38" font-size="12">£3.40 ✔</text></g>
  <g transform="translate(84,56)"><rect x="0" y="0" width="86" height="30" rx="4" fill="#FDF3E7" stroke="#C93B3B" stroke-width="2"/><text x="43" y="20" fill="#8a3b3b" font-size="12">£340.00</text><line x1="4" y1="15" x2="82" y2="15" stroke="#C93B3B" stroke-width="2"/></g>
  <text x="127" y="102" fill="#8a3b3b" font-size="8">wrong number of digits</text>
</g>
<text x="260" y="132" fill="#9A5A12" font-size="10">the same trick, for training runs: two anchors you keep in your head</text>
<g transform="translate(34,142)">
  <rect x="0" y="0" width="212" height="56" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/>
  <text x="106" y="18" fill="#5E5191" font-size="9">ANCHOR 1 · 175 billion knobs</text>
  <text x="106" y="34" fill="#5E5191" font-size="9">300 billion tokens → 3e23 steps</text>
  <text x="106" y="48" fill="#5E5191" font-size="8">on today's chips ≈ 9 days on 1 000</text>
</g>
<g transform="translate(274,142)">
  <rect x="0" y="0" width="212" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/>
  <text x="106" y="18" fill="#1a5c38" font-size="9">ANCHOR 2 · 7 billion knobs</text>
  <text x="106" y="34" fill="#1a5c38" font-size="9">1.4 trillion tokens → 6e22 steps</text>
  <text x="106" y="48" fill="#276b45" font-size="8">on today's chips ≈ 1.7 days on 1 000</text>
</g>
<text x="260" y="220" fill="#9A938A" font-size="9">a new answer far from BOTH anchors is not exciting — it is a slipped decimal point</text>
</g></svg>
%%%

**What the price-tag picture gets right:** a rough sense of scale catches the big errors instantly, and the big errors are the ones that actually hurt. **What the picture gets wrong:** you have handled money since you were small, so banana-sized instincts came free. Model-run instincts have to be planted on purpose. That is what those two anchor cards are for.

#### The whole method, drawn as a ladder
Every number today has been written in the `3.15e23` style, for one reason. It turns frightening arithmetic into two small jobs you can do in your head. Here is the ladder those numbers live on, with our own division drawn on it.

%%% svg
<svg viewBox="0 0 520 275" role="img" aria-label="A powers of ten ladder drawn as a vertical ruler. Rungs are marked at ten to the nine giga a billion, ten to the twelve tera a trillion, ten to the fifteen peta, ten to the eighteen exa, ten to the twenty one, and ten to the twenty four, three powers apart. A band marks where training bills live, from ten to the twenty two to ten to the twenty six. The bill three point one five times ten to the twenty three is marked high up, the chip rate four times ten to the fourteen is marked in the middle, and an arrow runs down to the answer seven point nine times ten to the eight seconds, labelled with the subtraction twenty three minus fourteen equals nine, giving zero point seven nine times ten to the nine. Two side panels show the rules: when you multiply, multiply the fronts and add the powers; when you divide, divide the fronts and subtract the powers; and sliding the point one place left raises the power by one.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">One division = subtract the powers: 23 − 14 = 9</text>
<line x1="70" y1="36" x2="70" y2="256" stroke="#3A342E" stroke-width="2"/>
<g stroke="#3A342E" stroke-width="1.5">
  <line x1="62" y1="40" x2="78" y2="40"/><line x1="62" y1="75" x2="78" y2="75"/><line x1="62" y1="110" x2="78" y2="110"/><line x1="62" y1="145" x2="78" y2="145"/><line x1="62" y1="180" x2="78" y2="180"/><line x1="62" y1="215" x2="78" y2="215"/><line x1="62" y1="250" x2="78" y2="250"/>
</g>
<g fill="#3A342E" font-size="9" text-anchor="end">
  <text x="58" y="43">1e24</text><text x="58" y="78">1e21</text><text x="58" y="113">1e18</text><text x="58" y="148">1e15</text><text x="58" y="183">1e12</text><text x="58" y="218">1e9</text><text x="58" y="253">1e6</text>
</g>
<g fill="#6B645E" font-size="8" text-anchor="start">
  <text x="84" y="113">exa</text><text x="84" y="148">peta</text><text x="84" y="183">tera (a trillion)</text><text x="84" y="218">giga (a billion)</text>
</g>
<rect x="120" y="40" width="86" height="34" rx="4" fill="#F4F1FB" stroke="#5E5191" opacity="0.9"/>
<text x="163" y="54" fill="#5E5191" font-size="8">training bills</text>
<text x="163" y="66" fill="#5E5191" font-size="8">1e22 – 1e26</text>
<circle cx="70" cy="52" r="4" fill="#C93B3B"/>
<text x="216" y="55" fill="#8a3b3b" font-size="9" text-anchor="start">← the bill 3.15e23</text>
<circle cx="70" cy="157" r="4" fill="#2D8B55"/>
<text x="216" y="160" fill="#276b45" font-size="9" text-anchor="start">← the tap 4e14 steps/sec</text>
<circle cx="70" cy="216" r="4" fill="#9A5A12"/>
<text x="216" y="219" fill="#9A5A12" font-size="9" text-anchor="start">← the answer 7.9e8 seconds</text>
<path d="M100 56 L100 212" stroke="#C99A12" stroke-width="2" marker-end="url(#dn)"/>
<defs><marker id="dn" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C99A12"/></marker></defs>
<text x="108" y="126" fill="#9A5A12" font-size="8" text-anchor="start">powers: 23 − 14 = 9</text>
<text x="108" y="137" fill="#9A5A12" font-size="8" text-anchor="start">fronts: 3.15 ÷ 4 = 0.79</text>
<text x="108" y="148" fill="#9A5A12" font-size="8" text-anchor="start">→ 0.79e9 = 7.9e8</text>
<g transform="translate(330,36)">
  <rect x="0" y="0" width="176" height="60" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="1.5"/>
  <text x="88" y="16" fill="#8A6D3B" font-size="9">MULTIPLY</text>
  <text x="88" y="32" fill="#9A5A12" font-size="8">fronts: multiply them</text>
  <text x="88" y="46" fill="#9A5A12" font-size="8">powers: ADD them (11 + 11 = 22)</text>
</g>
<g transform="translate(330,106)">
  <rect x="0" y="0" width="176" height="60" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/>
  <text x="88" y="16" fill="#1a5c38" font-size="9">DIVIDE</text>
  <text x="88" y="32" fill="#276b45" font-size="8">fronts: 3.15 ÷ 4 = 0.79</text>
  <text x="88" y="46" fill="#276b45" font-size="8">powers: SUBTRACT (23 − 14 = 9)</text>
</g>
<g transform="translate(330,176)">
  <rect x="0" y="0" width="176" height="70" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"/>
  <text x="88" y="16" fill="#5E5191" font-size="9">SLIDE THE POINT</text>
  <text x="60" y="36" fill="#5E5191" font-size="10">31.5e22</text>
  <path d="M92 32 h 22" stroke="#5E5191" stroke-width="1.5" marker-end="url(#sp)"/>
  <defs><marker id="sp" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#5E5191"/></marker></defs>
  <text x="140" y="36" fill="#5E5191" font-size="10">3.15e23</text>
  <text x="88" y="54" fill="#5E5191" font-size="8">point one place LEFT →</text>
  <text x="88" y="65" fill="#5E5191" font-size="8">power one HIGHER. Same number.</text>
</g>
<text x="260" y="270" fill="#9A938A" font-size="9">each rung on this ruler is 3 powers of ten · the yellow arrow drops about 15 powers</text>
</g></svg>
%%%

Read the ladder once more and say the two rules out loud. When you multiply: multiply the fronts, add the powers. When you divide: divide the fronts, subtract the powers. Sliding the point one place left raises the power by one, and the number does not change at all.

%%% steps
step: multiply → multiply the fronts, ADD the powers
why: 6 × (1.75e11) × (3e11): fronts 6 × 1.75 × 3 = 31.5, powers 11 + 11 = 22 → 31.5e22 = 3.15e23.
step: divide → divide the fronts, SUBTRACT the powers
why: 3.15e23 ÷ 4e14: fronts 3.15 ÷ 4 = 0.79, powers 23 − 14 = 9 → 0.79e9 = 7.9e8.
step: tidy up by sliding the point
why: 31.5e22 and 3.15e23 are the same number. Move the point one place left, raise the power by one. Move it right, lower the power.
step: the useful ladder to memorise
why: 1e9 is a billion (giga), 1e12 a trillion (tera), 1e15 a thousand trillion (peta), 1e18 a million trillion (exa). Training bills live around 1e22 to 1e26.
%%%

#### Now use the anchors
Keep those two runs in your head. Every new estimate gets compared with them *before* you say it out loud.

%%% steps
step: a new estimate arrives: 2e23 steps for a run
why: it sits between anchor 2 (6e22) and anchor 1 (3e23). Believable. Carry on.
step: a new estimate arrives: 4e17 steps for a big run
why: that is more than a hundred thousand times under anchor 2. Something is wrong, and this is not a small mistake. Start looking for it.
step: usual culprits, in the order they happen
why: the 6 was dropped (answer 6 times too small); tokens got confused with stretches of text (out by hundreds or thousands); a rate was used where a count belonged; the powers of ten were added when they should have been subtracted.
%%%

%%% insight
This is the difference between using a rule and owning it. Anybody can multiply three numbers. Owning it means the answer arrives *and you already have an opinion about whether it is believable.* That opinion is trainable, and two anchors are enough to start it.
%%%

%%% demo id=c12check label="predict whether it is believable, then reveal"
predict: somebody tells you their 13-billion-knob model trained on 2 trillion tokens used about 3e18 steps of arithmetic. Believable, or has something slipped?
code: check it yourself: 6 × 1.3e10 × 2e12
code: fronts:  6 × 1.3 × 2 = 15.6
code: powers:  10 + 12 = 22
out: about 1.6e23 steps — not 3e18
out: their number is about 50 000 times too small
out: 1.6e23 sits happily between the two anchors ✔
take: <b>Something slipped, badly.</b> A gap that big is never a small modelling detail. It is a units mistake. Most likely they counted stretches of text instead of tokens, or they read a rate as a count. And you found it with one multiplication and a glance at your two anchors.
%%%

You can now build an estimate and catch a bad one. Two honest complications left, and then the day is finished.

@@@ concept id=c13 tag="When six is not six" title="Two honest reasons the six changes" gotit="Got the two constants"
Picture doing a long division on the last clean page of your notebook. Your working fills the page. You need space for the next question, so you rub the working out. Then the teacher says: "show me how you got that." You have to do the whole division **again**. Rubbing out saved paper and cost you time.

A training run gets to make that same choice on purpose, and it is a good trick to have. It buys back memory when a run would not otherwise fit — and the price is only two extra steps per handshake. Knowing why it exists also explains a whole class of arguments between two people who can both multiply. Here is how it works. During the forward pass the model produces piles of half-finished working-out, and it must keep that working, because the backward pass needs it. Memory fills fast. So a lab can flip a switch: **throw the working away, and redo it during the backward pass.** The clean six becomes about **eight**.

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="A notebook page as the picture for recomputation. On the left the page is full of working, labelled keep the working, memory fills up, six steps per handshake. An eraser rubs it out in the middle. On the right the page is blank and a redo arrow shows the working being written again, labelled rub it out and redo it later, memory saved, about eight steps per handshake. A caption says: same answer, different constant, so say which one you used.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Keep your working, or rub it out and redo it later — you must pick one</text>
<g transform="translate(24,30)">
  <rect x="0" y="0" width="150" height="120" rx="4" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="2"/>
  <g stroke="#DCD6CC"><line x1="10" y1="26" x2="140" y2="26"/><line x1="10" y1="46" x2="140" y2="46"/><line x1="10" y1="66" x2="140" y2="66"/><line x1="10" y1="86" x2="140" y2="86"/><line x1="10" y1="106" x2="140" y2="106"/></g>
  <g fill="#5E5191" font-size="9" text-anchor="start"><text x="14" y="22">4 8 2 1 . 6</text><text x="14" y="42">  3 6</text><text x="14" y="62">  1 2 2</text><text x="14" y="82">  1 0 8</text><text x="14" y="102">    1 4 1</text></g>
  <text x="75" y="136" fill="#276b45" font-size="9">keep the working</text>
  <text x="75" y="150" fill="#276b45" font-size="8">memory fills · 6 per handshake</text>
</g>
<g transform="translate(196,58)">
  <rect x="0" y="0" width="52" height="26" rx="4" fill="#F5A9A9" stroke="#8a3b3b" stroke-width="1.5"/>
  <rect x="52" y="0" width="26" height="26" rx="3" fill="#EDE9E0" stroke="#8a3b3b" stroke-width="1.5"/>
  <text x="39" y="45" fill="#8a3b3b" font-size="9">rub it out</text>
  <path d="M10 60 h 60" stroke="#C93B3B" stroke-width="2" marker-end="url(#er)"/>
  <defs><marker id="er" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C93B3B"/></marker></defs>
</g>
<g transform="translate(300,30)">
  <rect x="0" y="0" width="150" height="120" rx="4" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="2"/>
  <g stroke="#DCD6CC"><line x1="10" y1="26" x2="140" y2="26"/><line x1="10" y1="46" x2="140" y2="46"/><line x1="10" y1="66" x2="140" y2="66"/><line x1="10" y1="86" x2="140" y2="86"/><line x1="10" y1="106" x2="140" y2="106"/></g>
  <text x="75" y="56" fill="#9A938A" font-size="9">(page cleared)</text>
  <path d="M40 76 q 35 24 70 0" fill="none" stroke="#C99A12" stroke-width="2" marker-end="url(#rd)"/>
  <defs><marker id="rd" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#C99A12"/></marker></defs>
  <text x="75" y="102" fill="#9A5A12" font-size="8">do the working AGAIN</text>
  <text x="75" y="136" fill="#8a3b3b" font-size="9">rub out and redo</text>
  <text x="75" y="150" fill="#8a3b3b" font-size="8">memory saved · about 8 per handshake</text>
</g>
<text x="260" y="186" fill="#9A5A12" font-size="10">6 → 8 is about a third more arithmetic, bought with memory</text>
<text x="260" y="204" fill="#9A938A" font-size="9">both constants are correct — the professional move is to say which one you used</text>
</g></svg>
%%%

**What the notebook picture gets right:** you really are trading paper for repeated effort, and the repeat is one extra pass over the same work. **Where the picture stops working:** you would rub out the whole page. A run usually keeps a few saved pages and only redoes the stretches between them. So the extra cost is close to one extra forward pass, not two.

#### Reason one: the working got rubbed out
The proper name for this switch is [[activation recomputation||Throwing away the forward pass's saved working-out and doing it again during the backward pass. It saves memory and costs about one extra forward pass. Also called gradient checkpointing.]]. You will also hear *gradient checkpointing*, which is the same thing.

%%% steps
step: normal training: 2 forward + 4 backward = 6 per handshake
why: the working from the forward pass was kept, so the backward pass just used it.
step: with recomputation: 2 forward + 2 redone forward + 4 backward = 8
why: one extra forward pass is 2 more steps per handshake. That is the whole price.
step: 8 ÷ 6 ≈ 1.33 → about a third more arithmetic
why: so a 9-day run becomes about a 12-day run. Nothing broke; the lab chose it.
step: why would anybody choose that? because the run would not fit otherwise
why: memory is a separate budget, and unit 14 is about exactly that. Day 4 shows you the saved working-out itself, and m09a puts numbers on the memory it frees.
%%%

#### See for yourself why anybody would pay a third more
Nobody spends three extra days for fun. Open the explorer again and scroll to **Part 2**, the memory half. **Predict first:** if you raise the batch or the sequence length, does the arithmetic bill in Part 1 move? Then drag those sliders and watch the memory bar climb towards the chip's line while the Part 1 bill sits completely still. That gap you are looking at is the whole reason the rub-out switch exists.

%%% viz src=../../viz/transformer-flops.html title="why recomputation exists: drag the Part 2 memory sliders" caption="Part 2 draws the training memory next to one chip's memory line. Raise the batch or the sequence length and the memory bar grows while the Part 1 arithmetic bill does not move. When that bar goes over the line, a lab reaches for the rub-out switch and pays about 8 instead of 6."
%%%

#### Reason two: some knobs were switched off
Here is the second reason, and it is a picture rather than a formula. Our whole day assumed that **every knob works on every token**. That was the receiving line where nobody gets skipped. Models built that way are called [[dense||A model where every knob is used for every token — the receiving line with nobody skipped. Today's rule assumes a dense model.]]. Some newer models break that rule on purpose. They hold a great many knobs but switch most of them off for each token, like a building where only two rooms have the lights on.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A wall of light switches as the picture for a sparse model. Eight switch panels are drawn: two are on and glowing, six are off and grey. A label says: you pay for the lights that are on. Below, two lines: total knobs 8 rooms, but active knobs per token 2 rooms, so use the active count in the rule.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Most rooms dark, two rooms lit — you pay for the lit ones</text>
<g transform="translate(36,30)">
  <g fill="#EDE9E0" stroke="#B8AEA2" stroke-width="1.5">
    <rect x="0" y="0" width="46" height="58" rx="5"/><rect x="56" y="0" width="46" height="58" rx="5"/><rect x="112" y="0" width="46" height="58" rx="5"/><rect x="168" y="0" width="46" height="58" rx="5"/><rect x="224" y="0" width="46" height="58" rx="5"/><rect x="280" y="0" width="46" height="58" rx="5"/>
  </g>
  <g fill="#FDF3E7" stroke="#C99A12" stroke-width="2"><rect x="336" y="0" width="46" height="58" rx="5"/><rect x="392" y="0" width="46" height="58" rx="5"/></g>
  <g fill="#9A938A" font-size="8"><text x="23" y="34">off</text><text x="79" y="34">off</text><text x="135" y="34">off</text><text x="191" y="34">off</text><text x="247" y="34">off</text><text x="303" y="34">off</text></g>
  <g fill="#9A5A12" font-size="8"><text x="359" y="30">ON</text><text x="415" y="30">ON</text></g>
  <g fill="#C99A12" font-size="12"><text x="359" y="46">💡</text><text x="415" y="46">💡</text></g>
</g>
<text x="260" y="118" fill="#5E5191" font-size="10">total knobs = all 8 rooms · knobs used for THIS token = 2 rooms</text>
<text x="260" y="140" fill="#8a3b3b" font-size="10">so put the ACTIVE count into the rule, never the total</text>
<text x="260" y="164" fill="#9A938A" font-size="8">m14a is the module that opens this door — today you only need to notice the door exists</text>
</g></svg>
%%%

For a model like that, N in our rule means the **active** knob count. That is the knobs which actually work on each token, not the headline total. Put the total in and you will over-price the run by a large factor.

!!! c-warn ⚠️
<b>Failure mode:</b> two people price the same run and disagree by a third, or by several times. Each is sure the other cannot multiply.<br>
<b>Cause:</b> one used 6 and the other 8 (recomputation off or on). Or one used the total knob count for a model that switches most knobs off per token.<br>
<b>Remedy:</b> quote your constant along with your answer. "6ND, no recomputation, dense" is a complete sentence, and it ends the argument before it starts.
!!!

Look at what you just picked up: one short sentence that settles an argument which can cost two engineers an afternoon. That is a good trade for two minutes of reading.

%%% insight
Both of these are the same lesson wearing two coats. The number 6 is a *setting*, not a law of nature. It came from choices: keep the working, use every knob. Change a choice and the constant changes with it. Knowing which choices are baked into your number is the whole difference between quoting a rule and understanding one.
%%%

%%% demo id=c13eight label="predict the extra days, then reveal them"
predict: your run was 9 days with the six. Memory is tight, so the team turns on recomputation. How many days now?
code: constant goes 6 → 8
code: 8 ÷ 6 = 1.333...
code: 9 days × 1.333
out: about 12 days
out: 3 extra days of arithmetic, bought so the run fits in memory at all
take: <b>Twelve days, and it was a good trade.</b> A run that fits and takes 12 days beats a run that does not fit and takes none. That sentence is really about a second budget we have not measured yet — and it is the last thing standing between you and the end of the day.
%%%

One last thing, and it is the calm one. There are two questions this arithmetic simply cannot answer, and knowing them is what makes the rest of it trustworthy.

@@@ concept id=c14 tag="What it cannot tell you" title="Two questions this arithmetic will never answer" gotit="Got the two limits"
Look at a café menu. Two dishes, both £9. The menu is completely honest about the price and completely silent about which one you will enjoy. Price and pleasure are different questions. No amount of staring at the menu turns one into the other.

Our rule is that menu, and by now it is a very good menu: it prices any plan you hand it, quickly and reliably. So here is the last thing you gain today, and it is the thing that makes people trust you with the rest: you get to name the exact edges of your own tool. There are two, both easy to hold in your head. It cannot tell you which plan is *good*. And it cannot tell you whether the plan will even fit in the chips.

%%% svg
<svg viewBox="0 0 520 215" role="img" aria-label="A café menu as the picture for the first limit. The menu lists plan A, a small model reading a great deal of text, at nine pounds, and plan B, a large model reading little text, also at nine pounds. A question mark sits between them with the note: the menu prices both the same and cannot say which is better. A caption points at day five for the answer.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Limit 1: the same price, two different meals</text>
<g transform="translate(90,28)">
  <rect x="0" y="0" width="340" height="126" rx="8" fill="#FDFCFA" stroke="#B8AEA2" stroke-width="2"/>
  <text x="170" y="22" fill="#3A342E" font-size="10">TODAY'S PLANS — both priced by 6ND</text>
  <line x1="20" y1="30" x2="320" y2="30" stroke="#DCD6CC"/>
  <g text-anchor="start" font-size="9">
    <text x="24" y="52" fill="#276b45">PLAN A · small model, a great deal of text</text>
    <text x="24" y="66" fill="#6B645E">N small, D large</text>
    <text x="296" y="58" fill="#2C2A28" text-anchor="end" font-size="12">3e23</text>
    <text x="24" y="92" fill="#5E5191">PLAN B · large model, little text</text>
    <text x="24" y="106" fill="#6B645E">N large, D small</text>
    <text x="296" y="98" fill="#2C2A28" text-anchor="end" font-size="12">3e23</text>
  </g>
  <text x="170" y="122" fill="#8a3b3b" font-size="9">identical bills · very different models · the menu says nothing</text>
</g>
<g transform="translate(20,70)"><text x="0" y="0" fill="#C93B3B" font-size="34">?</text></g>
<text x="260" y="176" fill="#9A5A12" font-size="10">the rule prices a plan; choosing between plans is day 5's job</text>
<text x="260" y="198" fill="#9A938A" font-size="9">(there is a measured best split — about twenty tokens per knob — and it is not in this arithmetic)</text>
</g></svg>
%%%

**What the menu picture gets right:** a price list is genuinely useful and genuinely silent about quality. Pretending otherwise is how people get disappointed. **Where the picture stops being true:** with dinner you find out in ten minutes. With a training run you find out in nine days. That is exactly why the choosing question deserves its own day.

#### Limit one: it prices a plan, it never chooses one
You can hand the rule a hundred plans and it will price all hundred without complaining. Every plan with the same N × D gets the same number. Somewhere in that list is the plan that produces the best model, and the arithmetic cannot point at it.

%%% steps
step: same bill, different results — this really happens, it is not a small detail
why: the 2022 measurement that surprised everyone found several famous models had been given too little text for their size. Same money, better model available, missed.
step: there IS an answer, and it was measured
why: roughly twenty tokens of reading per knob is the balance that gets the best model out of a fixed bill (Hoffmann and colleagues, 2022). Day 5 is where you meet it properly.
step: so today's honest sentence is: "this costs about 3e23 — whether it is the right plan is a different question"
why: keeping those two claims apart is a good habit. It is not a way of dodging the question.
%%%

Here is that idea as one picture, and it is the clearest way to see what the rule is blind to. The green hill is how good the model turns out. The flat grey line is our bill. Same money everywhere along the bottom.

%%% svg
<svg viewBox="0 0 520 240" role="img" aria-label="A hill of model quality drawn above a flat line of cost. The bottom axis runs from a tiny model reading a great deal of text on the left, through about twenty tokens per knob in the middle, to a huge model reading very little text on the right. A green curve of how good the model turns out rises to a peak at the twenty tokens per knob mark. A flat grey line labelled the bill, 6ND, three times ten to the twenty three, runs straight across at the same height everywhere. Plan A sits left of the peak and plan B sits right of it, both on the flat cost line. A caption says the rule is flat, so it cannot see the hill.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="15" fill="#2C2A28" font-size="12">Same bill everywhere along the bottom — but the model is not the same</text>
<path d="M60 150 Q 160 40 260 46 Q 360 52 460 150" fill="none" stroke="#2D8B55" stroke-width="3"/>
<text x="152" y="42" fill="#1a5c38" font-size="9">how good the model turns out</text>
<line x1="60" y1="112" x2="460" y2="112" stroke="#9A938A" stroke-width="2.5" stroke-dasharray="6 4"/>
<text x="404" y="104" fill="#6B645E" font-size="9">the bill: 6ND = 3e23</text>
<text x="404" y="126" fill="#9A938A" font-size="8">flat — it never moves</text>
<circle cx="112" cy="112" r="5" fill="#5E5191"/>
<text x="112" y="132" fill="#5E5191" font-size="9">PLAN A</text>
<circle cx="408" cy="112" r="5" fill="#C93B3B"/>
<text x="408" y="132" fill="#8a3b3b" font-size="9">PLAN B</text>
<circle cx="260" cy="46" r="5" fill="#C99A12"/>
<text x="260" y="34" fill="#9A5A12" font-size="9">best split ≈ 20 tokens per knob</text>
<line x1="60" y1="168" x2="460" y2="168" stroke="#3A342E" stroke-width="1.5"/>
<g fill="#6B645E" font-size="8">
  <text x="96" y="182">tiny model,</text><text x="96" y="192">a great deal of text</text>
  <text x="260" y="182">balanced</text>
  <text x="424" y="182">huge model,</text><text x="424" y="192">very little text</text>
</g>
<text x="260" y="214" fill="#8a3b3b" font-size="10">the hill has a top · the bill is flat · so the bill cannot point at the top</text>
<text x="260" y="232" fill="#9A938A" font-size="8">the hill was measured in 2022 (Hoffmann and colleagues) — day 5 walks up it properly</text>
</g></svg>
%%%

#### Limit two: money and memory are two different budgets
Now the second blind spot, and it is worth knowing because it has cancelled more runs than any arithmetic error. Picture a sofa you can afford, delivered to a doorway it does not fit through. The money question said yes. The doorway question said no. They were never the same question.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A sofa and a narrow doorway as the picture for the second limit. On the left a wide sofa with a price tag reading affordable, ticked. On the right a narrow doorway the sofa cannot pass through, labelled will not fit, crossed. A caption says: arithmetic is one budget, memory is another, and a plan must pass both.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">Limit 2: affordable and impossible are not opposites</text>
<g transform="translate(30,40)">
  <rect x="0" y="40" width="190" height="46" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/>
  <rect x="0" y="18" width="190" height="26" rx="8" fill="#EDE9E0" stroke="#5E5191" stroke-width="2"/>
  <rect x="0" y="86" width="20" height="16" fill="#5E5191"/><rect x="170" y="86" width="20" height="16" fill="#5E5191"/>
  <g transform="translate(60,-6)"><rect x="0" y="0" width="80" height="24" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="40" y="16" fill="#1a5c38" font-size="9">affordable ✔</text></g>
  <text x="95" y="120" fill="#5E5191" font-size="9">the arithmetic budget said YES</text>
</g>
<g transform="translate(320,34)">
  <rect x="0" y="0" width="120" height="112" fill="#EDE9E0" stroke="#3A342E" stroke-width="2"/>
  <rect x="42" y="18" width="36" height="94" fill="#FDFCFA" stroke="#3A342E" stroke-width="2"/>
  <text x="60" y="66" fill="#8a3b3b" font-size="9">too</text>
  <text x="60" y="80" fill="#8a3b3b" font-size="9">narrow</text>
  <g transform="translate(18,-6)"><rect x="0" y="0" width="84" height="24" rx="4" fill="#FDF3E7" stroke="#C93B3B" stroke-width="2"/><text x="42" y="16" fill="#8a3b3b" font-size="9">will not fit ✘</text></g>
  <text x="60" y="126" fill="#8a3b3b" font-size="9">the memory budget said NO</text>
</g>
<text x="260" y="180" fill="#9A5A12" font-size="10">a plan has to pass BOTH doors — and 6ND only knows about one of them</text>
<text x="260" y="196" fill="#9A938A" font-size="8">the model's knobs, their blames, the updater's extra numbers and the saved working all want room</text>
</g></svg>
%%%

Nothing in `6 × N × D` mentions space. A run can be perfectly affordable in arithmetic and still unable to start. The knobs, their blames, the updater's extra numbers and the saved working-out all have to sit in a chip's memory at the same time. That is a whole second bill, and it has its own arithmetic. Day 2 starts it, day 4 shows what fills memory during a real loop, and m09a counts every byte.

#### Go and see the second budget move
The explorer you used earlier has a second half you have not touched yet. Scroll it down to **Part 2**. Then drag the memory sliders and **watch** the memory bar cross the chip's memory line. Same model, same arithmetic bill, and suddenly the answer is "it does not fit".

%%% viz src=../../viz/transformer-flops.html title="the second budget: drag the memory sliders in Part 2" caption="Part 2 draws the training memory against one chip's memory line. Try raising the sequence length or the batch and watch the bar cross the line while the Part 1 bill does not move at all. Careful with the layers slider: here it only feeds the memory, but in a real model more layers means more knobs, so the arithmetic bill would grow too."
%%%

%%% insight
Notice that the two limits are opposite kinds of humility. The first says: I can price it but I cannot judge it. The second says: I can price it but I cannot promise it will run. Both are things this rule is *not*. Stating what a tool is not is how you keep people trusting the parts that work.
%%%

%%% demo id=c14fit label="predict which plan the rule prefers, then reveal"
predict: plan A is a small model reading a great deal of text. Plan B is a large model reading little text. Both come to 3e23 steps. Which one does 6ND recommend?
code: plan A: N small, D large  → 6ND = 3e23
code: plan B: N large, D small  → 6ND = 3e23
out: the rule recommends NEITHER — it prices both at 3e23 and stops
out: plan B may also fail the memory door that plan A walks straight through
take: <b>Neither, and that is the right answer.</b> The rule is a price list, not an adviser. Ask it "how much?" and it is excellent. Ask it "which one?" or "will it fit?" and you need day 5 and the memory arithmetic. Knowing which question you are asking is the skill.
%%%

That is the whole day. Let's gather it into one page you can keep.

@@@ concept id=c15 tag="Recap" title="Today in one page" gotit="Got the recap"
You started the day with a doorway clicker. You are ending it able to price a training run out loud, with your assumptions named. That is a genuinely useful thing to be able to do, and most people cannot do it.

Here is one last everyday picture for the whole chain: a **taxi fare**. There is a price per kilometre, which is our six. There is a distance, which is the handshakes, N × D. And then there is a separate question about how long the journey takes, which depends on the traffic — that is the chip's real rate. Fare and journey time are two calculations, in that order. **Where the taxi picture breaks down:** a taxi's distance is given to you. A lab *chooses* both N and D, and choosing well is day 5.

%%% svg
<svg viewBox="0 0 520 232" role="img" aria-label="The whole day as a five step chain. Step one: read two numbers, knobs N and tokens D. Step two: six steps of arithmetic per handshake. Step three: the bill C equals six N D. Step four: divide by the chip's real rate, the honest peak times MFU. Step five: seconds, then days, then chip-hours. Below, four notes: serving is two not six, the width test for long text, about eight with recomputation, and use active knobs for a switched-off model. At the bottom, the two anchors to keep: 175 billion knobs on 300 billion tokens gives about three times ten to the twenty three, and 7 billion knobs on 1.4 trillion tokens gives about six times ten to the twenty two.">
<g font-family="monospace" font-size="10" text-anchor="middle">
<text x="260" y="16" fill="#2C2A28" font-size="12">two numbers → a price per handshake → a bill → a rate → a wait</text>
<g transform="translate(6,30)"><rect x="0" y="0" width="96" height="52" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="2"/><text x="48" y="22" fill="#5E5191" font-size="9">1 · two numbers</text><text x="48" y="38" fill="#5E5191" font-size="8">N knobs, D tokens</text></g>
<text x="108" y="60" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(120,30)"><rect x="0" y="0" width="96" height="52" rx="8" fill="#FDF3E7" stroke="#C99A12" stroke-width="2"/><text x="48" y="22" fill="#8A6D3B" font-size="9">2 · price</text><text x="48" y="38" fill="#9A5A12" font-size="8">6 per handshake</text></g>
<text x="222" y="60" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(234,30)"><rect x="0" y="0" width="96" height="52" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="48" y="22" fill="#1a5c38" font-size="9">3 · the bill</text><text x="48" y="38" fill="#276b45" font-size="9">C ≈ 6ND</text></g>
<text x="336" y="60" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(348,30)"><rect x="0" y="0" width="80" height="52" rx="8" fill="#EDE9E0" stroke="#3A342E" stroke-width="2"/><text x="40" y="22" fill="#3A342E" font-size="9">4 · ÷ rate</text><text x="40" y="38" fill="#6B645E" font-size="7">peak × MFU</text></g>
<text x="434" y="60" fill="#9A5A12" font-size="13">→</text>
<g transform="translate(444,30)"><rect x="0" y="0" width="70" height="52" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="35" y="22" fill="#1a5c38" font-size="9">5 · time</text><text x="35" y="38" fill="#276b45" font-size="7">days · chip-hours</text></g>
<line x1="20" y1="98" x2="500" y2="98" stroke="#DCD6CC"/>
<text x="260" y="116" fill="#3A342E" font-size="10">the four places the six is not six</text>
<g font-size="8">
  <g transform="translate(14,124)"><rect x="0" y="0" width="118" height="42" rx="6" fill="#FDFCFA" stroke="#B8AEA2"/><text x="59" y="17" fill="#276b45">serving: 2, not 6</text><text x="59" y="32" fill="#6B645E">no backward pass</text></g>
  <g transform="translate(140,124)"><rect x="0" y="0" width="118" height="42" rx="6" fill="#FDFCFA" stroke="#B8AEA2"/><text x="59" y="17" fill="#9A5A12">long text: 6 undercounts</text><text x="59" y="32" fill="#6B645E">share over a tenth?</text></g>
  <g transform="translate(266,124)"><rect x="0" y="0" width="118" height="42" rx="6" fill="#FDFCFA" stroke="#B8AEA2"/><text x="59" y="17" fill="#8a3b3b">recompute: about 8</text><text x="59" y="32" fill="#6B645E">memory bought with time</text></g>
  <g transform="translate(392,124)"><rect x="0" y="0" width="118" height="42" rx="6" fill="#FDFCFA" stroke="#B8AEA2"/><text x="59" y="17" fill="#5E5191">switched-off knobs</text><text x="59" y="32" fill="#6B645E">use the ACTIVE count</text></g>
</g>
<text x="260" y="186" fill="#8a3b3b" font-size="9">and two things it never knows: which plan is better · whether it fits in memory</text>
<text x="260" y="206" fill="#9A938A" font-size="9">anchors to keep: 175 billion knobs on 300 billion tokens → C ≈ 3e23</text>
<text x="260" y="220" fill="#9A938A" font-size="9">and 7 billion knobs on 1.4 trillion tokens → C ≈ 6e22</text>
</g></svg>
%%%

The day in a few beats:

- **A step of arithmetic is one multiply or one add** — a [[FLOP||one step of arithmetic: one multiply or one add]]. That is the only unit of work in this whole subject.
- **Two numbers price a run:** N knobs (parameters) and D tokens read. Nothing else. And N never depends on how long the text is.
- **One knob used costs two steps** — a multiply and an add into a running total, the till's [[multiply-accumulate||one multiply plus one add into a running total; 2 steps]].
- **Reading costs 2 per handshake, learning about 4.** You counted both on one knob: one note out, two notes back. Two plus four is six.
- **So C ≈ 6 × N × D**, and the bill is an *area*: double either side and the bill doubles. Halve one side and double the other and it does not change at all.
- **Serving is 2, not 6** — about a third of the arithmetic, because nothing is being learned.
- **Seconds = C ÷ rate ÷ chips.** Divide *those* seconds by 3 600 for the hours you wait, or by 86 400 for days. Use the honest peak — halve any figure carrying the sparsity star — times a measured [[MFU||the share of a chip's honest top speed a real run reaches, about 0.3 to 0.55]], never the sticker.
- **Chip-hours is a different number:** take the seconds on ONE chip and divide by 3 600. It comes out as the hours you wait times the number of chips.
- **Check the size before you speak:** powers of ten add when you multiply and subtract when you divide. Every answer gets compared with an anchor you keep in your head.
- **Four places the six changes:** serving (2), long text (6 undercounts — run the width test), recomputation (about 8), switched-off knobs (use the active count).
- **Two things it never knows:** which plan is better (day 5), and whether the plan fits in memory (day 2, day 4, m09a).

The one line to carry: **C ≈ 6ND, priced per handshake, then divided by a rate you did not read off an advertisement.**

#### Cheat-sheet · the moves at a glance
%%% table
You want… :: Do this :: Watch out for
Training arithmetic :: `C ≈ 6 × N × D` :: N is knobs, D is tokens — not stretches of text
Reading only (forward) :: `2 × N × D` :: this is one third of training
Serving one reply :: `2 × N × (tokens in the reply)` :: real serving often waits on memory (day 3)
Seconds on one chip :: `C ÷ (honest peak × MFU)` :: never divide by the sticker number
Seconds on many chips :: `÷ number of chips` :: slightly optimistic — chips must talk (day 5)
Hours you actually wait :: `(seconds on MANY chips) ÷ 3 600` :: this is wall-clock time, not chip-hours
Chip-hours :: `(seconds on ONE chip) ÷ 3 600` :: same as waiting-hours × number of chips
Is the comparing work ignorable? :: `seq_len ÷ (12 × d_model)` — under a tenth? :: re-run the check when the text gets longer
Recomputation is on :: use `8` instead of `6` :: say which constant you used
A switched-off model :: use the ACTIVE knob count :: the headline total over-prices it
Sanity-check an answer :: compare with 3e23 or 6e22 :: a multiplication never warns you
%%%

#### Cheat-sheet · the words you met today
%%% jargon
FLOP | one step of arithmetic: one multiply or one add. FLOPs (with an s) is a count of them
FLOP/s | steps of arithmetic per second — a rate, not a count. Divide a count by this to get seconds
parameter | one number inside the model that it is allowed to change. Plain handle: a knob. m02 called these weights
N | the number of knobs (parameters) in the model
token | one chunk of text. D counts these, never letters and never words exactly
D | the number of tokens read during the whole training run
handshake | one knob meeting one token. The bill is six steps per handshake
multiply-accumulate | one multiply plus one add into a running total: the 2 steps a knob costs per use
accumulate | to add your result into a total that keeps growing
forward pass | reading and guessing. About 2 steps per handshake
blame | how wrong a knob was, worked out on the trip back. m02 and m04 use this word throughout
backward pass | the trip back that works out the blame. Two notes per knob, so about 4 steps per handshake
C ≈ 6ND | the training bill: six steps of arithmetic per handshake, times knobs, times tokens
inference | a model answering rather than learning. Forward only, so about 2 per handshake
sparsity | a chip trick that only counts when half your numbers are zeros in a set pattern. Halve any figure that mentions it
MFU | Model FLOPs Utilization: the share of a chip's honest top speed a real run reaches (about 0.3–0.55)
chip-hours | how many hours of ONE chip's attention a run needed. Seconds on one chip, divided by 3 600
seq_len | how many tokens the model handles in one stretch of text
d_model | the width of one token inside the model — how many values describe it (met in m03)
the width test | divide seq_len by 12 × d_model. Share under a tenth → the token-comparing work is small enough to ignore
activation recomputation | rubbing out the saved working and redoing it: saves memory, turns the 6 into about 8
dense | every knob is used for every token. Today's rule assumes this
active parameters | the knobs a switched-off-most-of-the-time model actually uses per token
%%%

One more thing, and it is the point of the whole day. **The arithmetic is not the hard part.** The hard part is naming your assumptions out loud with your number: 6 or 8, dense or switched-off, this MFU, that constant. Do that and people will trust your estimates. That is a strange and excellent kind of power for something you can do with a pen. Tomorrow, in **Q, K, V Matrices**, you go and find out where the N itself comes from.

@@@ quiz id=quiz tag="Quiz" title="Four questions — instant feedback on each" gotit="answer all four first"
Four quick questions, with feedback on each one. If a question trips you up, that is a signal to scroll back up, not a mark against you. Everybody has to re-read the MFU part.
%%% quiz
q: A model has N knobs and read D tokens while training. About how many steps of arithmetic did the whole run take? | a:2 | 6 × N | 2 × N × D | 6 × N × D | 6 × D | fb: Six steps per handshake, and there are N × D handshakes. The 2ND answer is the reading half only — the forward pass. 6 × N alone forgets the text completely, which is the "the bill is an area" slip.
q: Why does the backward pass cost about twice the forward pass? | a:1 | It runs the model twice for safety | Each knob does two jobs on the way back — its own blame, plus the blame to hand backwards | Because chips are slower going backwards | Because the text is read a second time | fb: You counted this on one knob: 2 presses forward, then 2 presses for note 1 (my own blame) and 2 presses for note 2 (the blame handed back). Two notes instead of one, so about double — which is where the 4 in 2 + 4 = 6 comes from.
q: A chip's fact sheet advertises 2 000 trillion steps a second, with a star saying "shown with sparsity". Which rate should you divide your step count by? | a:3 | 2 000 trillion — it is the official figure | 2 000 trillion, then add a safety margin at the end | 1 000 trillion, because the star only affects marketing | About 400 trillion: halve it for the sparsity star, then multiply by a measured MFU of about 0.4 | fb: Two haircuts, and they multiply. Halve the headline because the star assumes a trick your run does not use, then take the share a real run reaches (MFU, about 0.3–0.55). Skip both and your estimate is about five times too fast.
q: You price a run that reads very long stretches of text with plain 6ND, and your answer comes out far too low. What did the six leave out? | a:0 | The token-against-token comparing work, which grows with the square of the text length | The cost of loading the model from disk | The electricity bill | The optimizer's own arithmetic | fb: Tokens comparing each other uses no knobs, so N never sees it — and it grows with the square of the length. Run the width test: divide seq_len by 12 × d_model, and if that share is over a tenth, 6ND is too low — a smallest-possible answer rather than an estimate.
%%%

@@@ produce id=produce tag="Produce" title="Price a real run with your own hands" gotit="Done"
Time to do it for yourself. A rule you have only read is a rule you will not trust at three in the afternoon in front of other people. You will price a real published run, then time it on real chips, then break the rule on purpose so you can feel each of its edges.

**Predict first, before you write a single line.** The well-known 2020 run had 175 billion knobs and read 300 billion tokens. Write down your guess for the total steps of arithmetic — just the power of ten is fine. Then write down your guess for how many days it takes on 1 000 of today's chips. Keep both guesses where you can see them, and **notice** how close you were once the numbers land.

#### Option A · write it yourself
Create `sessions/m08-transformer-math/day-01-transformer-arithmetic/experiment.py`. No libraries are needed beyond plain Python. This is multiplication, and the point is that you can see every step. **Print what you observe at each stage.**

1. Define `N = 175e9` and `D = 300e9`, then print `C = 6 * N * D`. **Observe** that it lands near `3.15e23`, and print how far that is from the published `3.14e23` as a percentage.
2. Print the forward-only bill (`2 * N * D`) and its ratio to the training bill. **Notice** that it is one third.
3. Turn the bill into time. Set `peak = 1e15` (the honest top speed, already halved for the sparsity star) and `mfu = 0.4`. Then print seconds on one chip, years on one chip, days on 1 000 chips, and chip-hours (the ONE-chip seconds divided by 3 600). **Observe** that the days figure lands near 9.
4. Now the sticker mistake. Work the days out again using `2e15` with no MFU at all, and print how many times too fast that answer is. **Observe** the factor of 5.
5. Run the width test. With `d_model = 4096`, print `12 * d_model`. Then for `seq_len` in `[2048, 8192, 49152]` print the share `seq_len / (12 * d_model)` and whether 6ND is safe to use. **Notice** where the answer flips from "safe" to "no".
6. Print the recomputation version (`8 * N * D`) and the extra days it costs. Then print the second anchor's bill (`N = 7e9`, `D = 1.4e12`) and check it lands near `6e22`.

Finish with a self-check. Work out each claim as a boolean, print `✅ you got it` or `❌ not yet — expected …`, and assert each one. Run it with `python3 sessions/m08-transformer-math/day-01-transformer-arithmetic/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It will create the file, write the code, and run it for you. Then read it line by line and check each number against what you predicted.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Week 3 Day 1 (Transformer Arithmetic, C ≈ 6ND) artifact.

Create sessions/m08-transformer-math/day-01-transformer-arithmetic/experiment.py using plain Python only (no numpy needed), with a comment on every step and printing what it observes:
1. N = 175e9, D = 300e9. Print C = 6*N*D in scientific notation and the percentage gap from the published 3.14e23.
2. Print the forward-only cost 2*N*D and the ratio (2*N*D)/(6*N*D) — show it is 1/3.
3. Wall-clock: peak = 1e15 FLOP/s (dense, already halved from the 2e15 sparsity headline), mfu = 0.4, effective = peak*mfu. Print seconds on one chip, years on one chip, days on 1000 chips, wall-clock hours (the many-chip seconds / 3600) and chip-hours (the one-chip seconds / 3600) — and print that chip-hours equals wall-clock hours times 1000.
4. The sticker mistake: work out days on 1000 chips again using 2e15 with no MFU, and print how many times too fast it is (expect 5).
5. The width test: d_model = 4096; print 12*d_model; then for seq_len in [2048, 8192, 49152] print share = seq_len/(12*d_model) as a percentage and a verdict string "6ND is fine" when the share is under 0.10, else "6ND undercounts, add the attention term".
6. Recomputation: print 8*N*D and the extra days versus the 6ND figure (expect about a third more). Then the second anchor: N=7e9, D=1.4e12 → print C and confirm it is near 6e22, plus days on 1000 chips (expect about 1.7).
Then a self-check section: one boolean per claim (C within 1% of 3.15e23; forward ratio within 0.001 of 1/3; days on 1000 chips between 8 and 10; sticker factor within 0.01 of 5; chip-hours equals wall-clock hours times 1000 within 0.1%; the width-test verdict flips between seq_len 2048 and 8192; recompute ratio within 0.001 of 8/6), printing "✅ you got it" or "❌ not yet — expected ..." for each, then assert each with a message.
Run it and paste the output at the bottom as a comment. In one line, explain why the sticker number gives an answer five times too fast.
%%%

#### What you should see
- `C ≈ 3.15e23` steps, within about half a percent of the published `3.14e23` — your own arithmetic matched a paper.
- The forward-only bill is exactly one third of the training bill, for any N and D you try.
- About **9 days** on 1 000 of today's chips at an effective `4e14` steps a second — and about **25 years** on one. (The real 2020 run used slower chips and took longer.)
- Chip-hours near `2.2e5`, which is the same as those 219 waiting-hours multiplied by 1 000 chips.
- The sticker version claims about 1.8 days: **five times too fast**, from two haircuts you skipped.
- The width test passes at `seq_len = 2048` (share about 4 percent), fails the tenth test at `8192` (about 17 percent), and at `49152` the share reaches 100 percent, where the comparing work equals all the knob work.
- Recomputation costs about a third more: roughly 12 days instead of 9.
- The second anchor lands near `6e22` and about 1.7 days — both anchors now checked by your own code.

!!! c-info 📓
<b>5-minute research log:</b> before you close the tab, write three lines in `sessions/m08-transformer-math/day-01-transformer-arithmetic/log.md`: (1) the rule in one sentence, in your own words, including what a handshake is; (2) the two haircuts you give a chip's advertised speed, and why each one exists; (3) one question you would ask before trusting somebody else's 6ND estimate.

@@@ fin
