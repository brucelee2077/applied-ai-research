---
quest_id: wf4-d01-embeddings
mode: concept
donor: v9-base.donor
page_title: "Module 4 · Day 1 — Words to Numbers (Embeddings)"
module_label: "Module 4 · Represent · Day 1"
title: "Words to Numbers (Embeddings)"
subtitle: "How a Model Turns Words Into Places on a Map"
brand_sub: "Foundations · M4 Day 1"
spine: "map"
nav_prev_href: "../../m02-the-neuron/review.html"
nav_prev_label: "M2 · Review Gate"
nav_next_href: "../day-02-qkv/lesson.html"
nav_next_label: "Query, Key, Value"
fin_title: "Module 4 · Day 1 complete! 🏆"
fin_body: "Nice work — you've seen how a model turns each word into a point on a giant <b>map of meaning</b>, where words that mean similar things sit close together. That map is what everything else in a transformer reads from.<br>Next up: <b>Query, Key, Value</b> — how words look each other up on that map."
notebook_yardstick: null
---

@@@ hero
@lede Think about how you tidy a bookshelf. Cookbooks in one stack, comics in another, school books somewhere else. Your hand just *knows* that similar books belong close together — you never write down a rule. Now here's the little mystery: a computer cannot read the word `cat` the way you can. To a computer, letters are just shapes, not meaning. So before a model like ChatGPT can do anything clever with language, it has to pull off that same bookshelf trick — give every word a *spot*, and put words with similar meanings close together. Today you build that spot-giving trick yourself. It's called an **embedding**, and it is the very first thing that happens inside every language model. Get it right and "cat" and "dog" become neighbors, while "cat" and "car" sit rooms apart. Too simple to be the foundation of modern AI? Good — hold that feeling. By the end it will feel obvious, and a little bit magic.
@goal Together we'll turn a word into a list of numbers, arrange those numbers so meaning becomes *distance* on a map, and measure "how alike are these two words?" with one friendly little score — you'll get to drag a slider and watch that score move. You'll also meet the puzzles this simple map can't crack alone (a word it has never seen, a word with two meanings, and who-bit-whom) and see the clever fixes waiting ahead. Every new word gets said in plain English the moment it shows up.

%%% warmup
q: Deep down, what is the only kind of thing a computer really understands? | a:2 | pictures | letters and words | numbers | sounds | concept: computer-uses-numbers | fb: A computer is basically a calculator — it works with numbers, not letters. That's why a word must become numbers first.
q: In Module 2 you saw a model get better by slowly adjusting its inner numbers as it learns. What do we call those adjustable inner numbers? | a:0 | weights | letters | pictures | passwords | concept: weights-are-learned | fb: Weights are the tunable numbers a model nudges while it learns — the same idea that will shape today's word-numbers.
%%%

@@@ concept id=c1 tag="Word to numbers" title="A word becomes a little list of numbers" gotit="Got what an embedding is"
Let's start where every model starts. A computer is a calculator — it only understands numbers. So job number one is to turn a word into numbers.

Think of a **school report card**. Yours is a short list of scores: maybe math 9, reading 7, sports 4. Those numbers are not *you* — but together they paint a little picture of you.

Someone with a similar card is probably a similar kind of student. A model does exactly this to a word: it gives each word a short list of numbers, a kind of *report card for the word*.

That list is called an [[embedding||A word's report card: a fixed-length list of numbers that stands in for the word's meaning. Similar words get similar lists.]]. Because it is just a list of numbers, we also call it a [[vector||A plain list of numbers. An embedding is a vector.]] — "vector" is simply the math word for "a list of numbers."

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A report card for the word cat. The card has four rows labelled is-animal, is-furry, is-vehicle, is-fast, each with a small number score. Below, the same four scores are written as a plain list of numbers, the embedding vector."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A word gets a report card of scores</text><rect x="150" y="30" width="220" height="96" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="260" y="48" text-anchor="middle" fill="#8A6D3B" font-weight="bold">word: "cat"</text><line x1="164" y1="56" x2="356" y2="56" stroke="#E1C674"/><text x="172" y="74" fill="#6B645E">is-animal</text><text x="348" y="74" text-anchor="end" fill="#2C2A28" font-weight="bold">0.9</text><text x="172" y="92" fill="#6B645E">is-furry</text><text x="348" y="92" text-anchor="end" fill="#2C2A28" font-weight="bold">0.8</text><text x="172" y="110" fill="#6B645E">is-vehicle</text><text x="348" y="110" text-anchor="end" fill="#2C2A28" font-weight="bold">0.0</text><text x="172" y="124" fill="#6B645E">is-fast</text><text x="348" y="124" text-anchor="end" fill="#2C2A28" font-weight="bold">0.3</text><text x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">↓ read the scores straight down as one list ↓</text><text x="260" y="166" text-anchor="middle" fill="#5E5191" font-weight="bold">[ 0.9, 0.8, 0.0, 0.3 ]  ← the embedding</text></g></svg>
%%%

**Where the report card breaks down:** a real card has labels you can read ("math", "sports"), but a model invents its own hidden scores and we mostly can't name them. Our four friendly labels are a teaching crutch — I'll keep using them all day because they make the numbers easy to follow.

#### Back to the card: how a word actually becomes a list
Let's walk the card idea one rung at a time. Same three moves for every word in the language.

%%% steps
step: pick the questions the card will score
why: like "is-animal", "is-furry" — but a model invents its own hidden questions instead of ones we name
step: give the word a score on each question
why: cat is high on "is-animal", zero on "is-vehicle" — the scores ARE the word's report card
step: read the scores straight down as one list → [0.9, 0.8, 0.0, 0.3]
why: that list is the embedding — the only form of "cat" the rest of the model ever sees
%%%

**So far:** a word is now a short list of numbers, and every word's list has the *same* length.

#### How long is the list?
Every word gets the same length — 4 numbers in our toy table all day today, a few hundred (or a few thousand) in a real model. That length has a name: [[d_model||The length of every word's list. Bigger means more room to keep words apart, and more numbers to store.]]. A longer card can tell two students apart more finely — same for words.

%%% insight
Here is the part worth pausing on: the model never sees the letters `c`, `a`, `t` again. From this moment on, "cat" *is* that little list of numbers. Everything clever a language model does — every answer, every joke — is arithmetic on lists like this one.
%%%

Now let's make it real with three words. **Predict first:** cat and dog are both furry animals, car is a machine. Which two lists will look most alike? Guess, then reveal.

%%% demo id=lookup label="show three word-cards"
predict: which two of cat / dog / car will come back with nearly the SAME numbers?
code: embed("cat"), embed("dog"), embed("car")
out: cat -> [0.9, 0.8, 0.0, 0.3]
out: dog -> [0.8, 0.9, 0.0, 0.4]
out: car -> [0.0, 0.0, 0.9, 0.3]
take: <b>Three words, three lists of the same length (4 numbers).</b> Look down the columns: cat and dog are high in the same two places, while car puts its big score where the animals have 0. The numbers already whisper "cat and dog are alike; car is the odd one out" — before we do any math at all.
%%%

That is the whole starting idea, and you now own it. Next we turn these lists into places you can actually *see*.

@@@ concept id=c2 tag="Meaning as a map" title="Turning the lists into a map of meaning" gotit="Got the map"
Here is where it gets beautiful. We have lists of numbers. Now let's *see* them.

Think of a **map of your town**. A map turns a place into two numbers: how far east, how far north. Your school might be "3 blocks east, 5 blocks north."

Once every place is a pair of numbers, you can *see* which places are close — the bakery right by the school, the park far across town. Nobody has to explain it to you. You just look.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A simple street map of a town with a grid of streets. The school is marked at three blocks east and five blocks north. A bakery sits right next to the school. A park sits far away across town. A note says two numbers give every place a spot."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A town map: two numbers give every place a spot</text><rect x="46" y="30" width="428" height="150" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><g stroke="#EEEBE5" stroke-width="8"><line x1="46" y1="66" x2="474" y2="66"/><line x1="46" y1="108" x2="474" y2="108"/><line x1="46" y1="150" x2="474" y2="150"/><line x1="130" y1="30" x2="130" y2="180"/><line x1="230" y1="30" x2="230" y2="180"/><line x1="330" y1="30" x2="330" y2="180"/><line x1="416" y1="30" x2="416" y2="180"/></g><circle cx="230" cy="66" r="7" fill="#2D8B55"/><text x="230" y="54" text-anchor="middle" fill="#276b45" font-weight="bold">school</text><text x="230" y="86" text-anchor="middle" fill="#6B645E" font-size="9">3 east, 5 north</text><circle cx="262" cy="78" r="5" fill="#2D8B55"/><text x="286" y="82" fill="#276b45" font-size="9">bakery — next door</text><circle cx="416" cy="150" r="6" fill="#7C6DAA"/><text x="416" y="168" text-anchor="middle" fill="#5E5191" font-size="9">park — across town</text><line x1="262" y1="78" x2="416" y2="150" stroke="#C93B3B" stroke-dasharray="3,3"/><text x="340" y="112" fill="#C93B3B" font-size="9">far apart</text><text x="60" y="46" fill="#9A938A" font-size="9">↑ north</text><text x="60" y="174" fill="#9A938A" font-size="9">east →</text><text x="260" y="198" text-anchor="middle" fill="#6B645E" font-size="10">you never do math — you just LOOK and see what is near what</text></g></svg>
%%%

A word's embedding does the same thing. Treat its list of numbers as coordinates, and drop the word onto a map. Now you can literally see which words are neighbors. This is the big idea of the day, so let's name it: **meaning becomes distance.**

**Where the town map breaks down:** a town map has two directions (east, north); a word's map has *hundreds* at once. To draw a flat picture I pick **two** of the numbers and ignore the rest — all day, slot 1 (**is-animal**) goes across and slot 3 (**is-vehicle**) goes up.

#### Walk the four words onto the map
Keep the town map in your head and follow them on, one at a time.

%%% steps
step: take cat's list, keep the two map slots — is-animal 0.9, is-vehicle 0.0
why: those two numbers are the word's "east" and "north"; the other two slots are still there, just not drawn
step: read it as a place — 0.9 across, 0.0 up
why: exactly like "3 blocks east, 5 blocks north" — a list has become a spot on the map
step: drop dog on the same map — is-animal 0.8, is-vehicle 0.0
why: it lands right beside cat, therefore the map itself now says "these two are alike"
step: drop car — is-animal 0.0, is-vehicle 0.9
why: it lands way up the far side, which is exactly what "unrelated" looks like as a picture
step: drop "the" — is-animal 0.1, is-vehicle 0.1
why: it parks right by the middle of the map, because every one of its four numbers is tiny (its whole arrow is 0.17 long, while cat's is 1.24)
%%%

**So far:** lists became places, and places have neighborhoods.

%%% svg
<svg viewBox="0 0 520 306" role="img" aria-label="A map of meaning drawn to equal scale on both axes. The horizontal axis is the is-animal score and the vertical axis is the is-vehicle score. Cat at 0.9 across and 0.0 up sits beside dog at 0.8 across and 0.0 up in an animals neighborhood at the bottom right. Car at 0.0 across and 0.9 up sits alone at the top left in a vehicles neighborhood. The word the sits near the middle corner, and two equal-length dashed lines show that the is exactly the same distance from cat as it is from car."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The same four words, now as places (equal scale both ways)</text><rect x="128" y="30" width="304" height="248" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><line x1="160" y1="250" x2="404" y2="250" stroke="#B8AEA2" stroke-width="1.5"/><line x1="160" y1="40" x2="160" y2="250" stroke="#B8AEA2" stroke-width="1.5"/><text x="284" y="296" text-anchor="middle" fill="#6B645E" font-size="10">is-animal score (slot 1) →</text><text x="146" y="150" text-anchor="middle" fill="#6B645E" font-size="10" transform="rotate(-90 146 150)">is-vehicle (slot 3) ↑</text><ellipse cx="347" cy="250" rx="40" ry="20" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="4,3"/><circle cx="358" cy="250" r="6" fill="#2D8B55"/><text x="368" y="238" fill="#276b45">cat</text><circle cx="336" cy="250" r="6" fill="#2D8B55"/><text x="316" y="270" fill="#276b45">dog</text><text x="352" y="212" text-anchor="middle" fill="#276b45" font-size="10">the "animals" corner</text><ellipse cx="168" cy="52" rx="34" ry="20" fill="#EFEAF7" stroke="#7C6DAA" stroke-dasharray="4,3"/><circle cx="160" cy="52" r="6" fill="#7C6DAA"/><text x="180" y="56" fill="#5E5191">car</text><text x="240" y="46" fill="#5E5191" font-size="10">the "vehicles" corner</text><circle cx="182" cy="228" r="5" fill="#B8AEA2"/><text x="196" y="232" fill="#9A938A" font-size="10">"the"</text><line x1="182" y1="228" x2="336" y2="250" stroke="#8A6D3B" stroke-dasharray="3,3"/><line x1="182" y1="228" x2="160" y2="52" stroke="#8A6D3B" stroke-dasharray="3,3"/><text x="446" y="120" text-anchor="middle" fill="#8A6D3B" font-size="9">the two brown</text><text x="446" y="132" text-anchor="middle" fill="#8A6D3B" font-size="9">lines are the</text><text x="446" y="144" text-anchor="middle" fill="#8A6D3B" font-size="9">SAME length —</text><text x="446" y="156" text-anchor="middle" fill="#8A6D3B" font-size="9">"the" sits equally</text><text x="446" y="168" text-anchor="middle" fill="#8A6D3B" font-size="9">far from cat</text><text x="446" y="180" text-anchor="middle" fill="#8A6D3B" font-size="9">and from car</text><line x1="336" y1="250" x2="160" y2="52" stroke="#C93B3B" stroke-width="1" stroke-dasharray="3,3"/><text x="268" y="134" text-anchor="middle" fill="#C93B3B" font-size="9">far = unrelated</text><text x="284" y="184" text-anchor="middle" fill="#2D8B55" font-size="9">cat + dog: close = similar</text></g></svg>
%%%

%%% insight
Why care about a picture? Because this map is the *only* thing the rest of a transformer ever reads. Every "words looking at each other" trick you build over the coming days works purely with these positions — it never sees the letters. The map is the seating chart for the whole show.
%%%

!!! c-info 🔍
<b>Optional (skippable) — how much does a flat picture bend the truth?</b><br>A little, and you can check it. On this 2-slot map, cat (0.9 across, 0.0 up) and dog (0.8 across, 0.0 up) point in the <i>exact</i> same direction, so a 2-slot score reads a perfect <code>1.00</code>. Score them on all four numbers and you get <code>0.99</code> — the two slots I dropped add a small wobble.<br>• <b>Cause:</b> squashing many directions down to 2 has to distort <i>some</i> distances, and it hides whatever the dropped slots were saying.<br>• <b>Remedy:</b> read every 2-D embedding picture (mine included) as an <i>illustration</i>, and when it matters check closeness with a number in the full dimension — which is exactly what we do next.<br>• One more honest catch: a word parked near the middle, like <code>"the"</code>, has a very short arrow (0.17 long). Short arrows swing direction easily, so where it points is not worth reading much into.
!!!

!!! c-info 🗺️
<b>The chain to remember:</b> word → a list of numbers (embedding) → a point on the map of meaning. That is the whole journey of a single word into a model, and it is step 1 of every language model on Earth.
!!!

@@@ concept id=c3 tag="Measuring closeness" title="How close are two words? Point and compare" gotit="Got closeness"
We can *see* closeness on the map. But a computer can't squint at a picture. It needs a number that says "these two words are 80% alike."

Think of two friends standing in a field, each **pointing at something**. If they point nearly the *same direction*, they probably care about the same thing.

If one points north and the other points behind them, they're interested in totally different things. That's the whole test — and notice you don't care how far their arms stretch, only which way they aim.

%%% svg
<svg viewBox="0 0 520 216" role="img" aria-label="Three pictures of two friends standing in a field pointing. In the first both point almost the same way, score near 1, very alike. In the second they point at right angles, score near 0, unrelated. In the third they point away from each other, score near minus 1, the far end of the scale."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Two friends in a field: do they point the same way?</text><line x1="20" y1="150" x2="500" y2="150" stroke="#E5DFD6" stroke-width="2"/><g><circle cx="58" cy="112" r="9" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="58" y1="121" x2="58" y2="150" stroke="#8A6D3B" stroke-width="2"/><line x1="58" y1="126" x2="96" y2="104" stroke="#2D8B55" stroke-width="3" stroke-linecap="round"/><circle cx="82" cy="112" r="9" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="82" y1="121" x2="82" y2="150" stroke="#8A6D3B" stroke-width="2"/><line x1="82" y1="128" x2="126" y2="112" stroke="#2D8B55" stroke-width="3" stroke-linecap="round"/><text x="86" y="172" text-anchor="middle" fill="#276b45" font-size="10">both point the same way</text><text x="86" y="188" text-anchor="middle" fill="#276b45" font-weight="bold" font-size="11">score ≈ 1</text><text x="86" y="202" text-anchor="middle" fill="#9A938A">very alike</text></g><g><circle cx="238" cy="112" r="9" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="238" y1="121" x2="238" y2="150" stroke="#8A6D3B" stroke-width="2"/><line x1="238" y1="126" x2="238" y2="82" stroke="#C99A12" stroke-width="3" stroke-linecap="round"/><circle cx="262" cy="112" r="9" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="262" y1="121" x2="262" y2="150" stroke="#8A6D3B" stroke-width="2"/><line x1="262" y1="128" x2="308" y2="128" stroke="#C99A12" stroke-width="3" stroke-linecap="round"/><text x="266" y="172" text-anchor="middle" fill="#8A6D3B" font-size="10">a right angle apart</text><text x="266" y="188" text-anchor="middle" fill="#8A6D3B" font-weight="bold" font-size="11">score ≈ 0</text><text x="266" y="202" text-anchor="middle" fill="#9A938A">unrelated</text></g><g><circle cx="418" cy="112" r="9" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="418" y1="121" x2="418" y2="150" stroke="#8A6D3B" stroke-width="2"/><line x1="418" y1="126" x2="380" y2="104" stroke="#C93B3B" stroke-width="3" stroke-linecap="round"/><circle cx="442" cy="112" r="9" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="442" y1="121" x2="442" y2="150" stroke="#8A6D3B" stroke-width="2"/><line x1="442" y1="128" x2="486" y2="146" stroke="#C93B3B" stroke-width="3" stroke-linecap="round"/><text x="440" y="172" text-anchor="middle" fill="#8a3b3b" font-size="10">pointing away from each other</text><text x="440" y="188" text-anchor="middle" fill="#C93B3B" font-weight="bold" font-size="11">score ≈ −1</text><text x="440" y="202" text-anchor="middle" fill="#9A938A">as unlike as this score can read</text></g></g></svg>
%%%

A word's embedding is like an arrow pointing out from the middle of the map. To measure how alike two words are, we ask one question: **do their arrows point the same way?** The score for that has a name — [[cosine similarity||A score for how much two arrows point the same way: about 1 = same direction (very alike), 0 = at right angles (unrelated), −1 = pointing back the other way.]]. Fancy name, simple idea.

**Where the pointing friends break down:** two friends point in the flat world you can see, while word-arrows point in a space with hundreds of directions. You can't picture that — but "same direction = alike" works exactly the same there.

#### Your turn — grab the dials
Reading about arrows is fine. Moving one is better. Below, the green arrow is **cat**, parked where our table put it. The purple one is yours. Spin it, stretch it, and hunt for two things: *which* dial changes the score — and which one the score flat-out ignores. Make a guess before you drag.

%%% viz src=../../viz/embedding-similarity.html title="Point the arrow — cosine similarity, live" caption="Drag the direction dial and the score swings. Now drag the LENGTH dial — one of the two numbers refuses to move. Which one, and why do you think that is?"
%%%

#### Score cat and dog by hand — three moves, then just read it
Back to the two friends. To turn "same direction?" into a number we do three small things, and you can do them with all four numbers on a scrap of paper.

%%% steps
step: move 1 · pair them up — cat [0.9, 0.8, 0.0, 0.3] with dog [0.8, 0.9, 0.0, 0.4], multiply slot by slot
why: 0.9×0.8 = 0.72, 0.8×0.9 = 0.72, 0.0×0.0 = 0, 0.3×0.4 = 0.12 — each pair asks "do these two agree in this slot?"
step: move 2 · add the pieces up — 0.72 + 0.72 + 0 + 0.12 = 1.56
why: agreement in every slot piles up, therefore a big positive total means "same direction"
step: move 3 · divide out how long each arrow is — cat's length is 1.24, dog's is 1.27
why: that's the fairness step you just found with the length dial — now only DIRECTION counts, and the score always lands between −1 and +1
step: now just read it — 1.56 ÷ (1.24 × 1.27) = 0.99
why: one small number now carries what your eyes saw on the map: near 1 alike, near 0 unrelated, negative means pointing back
%%%

**So far:** multiply, add, divide by the lengths. That's the whole score.

%%% insight
Remember this number — you're going to meet it again on Day 2 with a different hat on. When words start *looking at each other* inside attention, "how much should I listen to you?" is answered by this exact same multiply-and-add. Today's little score is tomorrow's engine.
%%%

%%% hint
t1: Don't try to picture hundreds of directions. Take TWO lists and ask one question: do they rise and fall together, slot by slot?
t2: cat [0.9, 0.8, 0.0, 0.3] and dog [0.8, 0.9, 0.0, 0.4] → 0.72, 0.72, 0, 0.12 → total 1.56. Big and positive means "same direction". Now try cat against a made-up opposite [-0.9, -0.8, 0.0, -0.3] and watch the total go negative.
t3: Cosine similarity = "multiply the two lists slot-by-slot, add it up, divide by their lengths so only direction counts" — high means alike, near zero unrelated, negative means pointing back the other way.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the exact recipe, for the curious.</b><br>• The "multiply slot-by-slot and add" step has a name: the <b>dot product</b>, written <code>a·b = Σᵢ aᵢbᵢ</code> — sum, over every slot <code>i</code>, of the two lists' <code>i</code>-th numbers multiplied. For cat and dog, <code>a·b = 1.56</code>.<br>• Cosine similarity is that dot product divided by the two arrows' lengths: <code>cos(a, b) = (a·b) / (‖a‖ · ‖b‖)</code>, where <code>‖a‖</code> means "length of arrow a."<br>• Skip it freely — the four rungs above say the same thing in words.
!!!

Now watch the score work on a pair that should *not* match. **Predict first:** we score cat-vs-dog and cat-vs-car. Which is higher — and could one land near *zero*?

%%% demo id=cos label="score the pairs"
predict: cat vs dog, and cat vs car — which score is higher, and how close to 0 does the loser get?
code: cos(embed("cat"), embed("dog")),  cos(embed("cat"), embed("car"))
out: cos(cat, dog) = 0.99
out: cos(cat, car) = 0.08
take: <b>The animals score 0.99 (arrows almost the same way); cat-vs-car scores 0.08 (arrows nearly at a right angle).</b> Where did that leftover 0.08 even come from? One slot: cat and car both score 0.3 on <i>is-fast</i>, and 0.3×0.3 = 0.09 is the only agreement they have. Everything else cancels to nothing. And notice: no spelling was used anywhere. Meaning lived entirely in <i>where the arrows pointed</i>, and one little score read it off.
%%%

#### The payoff of that length dial
When you dragged the length dial, the raw multiply-and-add moved but the cosine didn't. That's not a party trick — it's protection, and here is the cleanest way to see it. Take cat's row and simply **double every number**. Same word, same meaning, arrow twice as long. **Predict first:** what happens to each of the two scores?

%%% demo id=lengthdial label="double the arrow — which score cares?"
predict: double cat's row. Does the raw multiply-and-add change? Does the cosine?
code: dot(cat, cat),      cos(cat, cat)
code: dot(cat, 2 * cat),  cos(cat, 2 * cat)
out: dot = 1.54    cos = 1.00
out: dot = 3.08    cos = 1.00
take: <b>The raw total doubled; the cosine didn't budge.</b> Nothing about the word changed — only the arrow's size — so a ranking built on the raw total would have called the doubled copy "twice as similar". The raw total is really answering two questions at once: "do these two agree?" AND "how big are these two rows?" Cosine divides the size out and asks only the first one.
%%%

That matters in a real table, where rows genuinely do differ in size: a row that got a lot of training grows, while a rarely-seen row stays short and can never win a raw-score ranking however well it points. So cosine similarity is the safe default when you care about meaning, not size.

%%% cards
📏 | Cosine, or plain distance? | Straight-line (Euclidean) distance reacts to arrow SIZE too, so a long arrow and a short one aiming the same way look "far apart" to it. Cosine reads only direction. Care about direction → cosine; care about actual position → distance.
🔥 | "Close" means used alike | "hot" and "cold" land CLOSE together, because they turn up in the same sentences ("the water is ___"). So a high score means "these two behave alike", NOT "same meaning" — real antonyms score high. When it matters, look at a word's nearest neighbours instead of trusting one number.
%%%

You just turned meaning into a single number you can compare. That is the quiet engine under search, recommendations, and the attention you'll build next.

@@@ concept id=c4 tag="The lookup table" title="Where the numbers live — the big lookup table" gotit="Got the lookup table"
Every word has a list of numbers. Fair question: where are all those lists *kept*? And how does the model grab the right one fast?

Think of a **coat-check counter** at a theater. You hand over your coat and get a numbered ticket — say ticket 27.

Later you give back ticket 27, and the attendant walks straight to hook 27 and returns your exact coat. No searching, no guessing. Just "give me the number, get the thing."

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A coat-check counter. A person hands over a coat and receives ticket number 27. Behind the counter is a rail of numbered hooks, and hook 27 holds that coat. An arrow shows the ticket going straight to the matching hook."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A coat-check: hand over a number, get your exact thing back</text><circle cx="56" cy="86" r="11" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="56" y1="97" x2="56" y2="132" stroke="#8A6D3B" stroke-width="2"/><line x1="56" y1="104" x2="86" y2="96" stroke="#8A6D3B" stroke-width="2"/><rect x="90" y="86" width="42" height="22" rx="4" fill="#FFFFFF" stroke="#C99A12"/><text x="111" y="101" text-anchor="middle" fill="#8A6D3B" font-size="9">ticket 27</text><rect x="146" y="60" width="26" height="86" rx="3" fill="#EEEBE5" stroke="#B8AEA2"/><text x="159" y="156" text-anchor="middle" fill="#9A938A" font-size="9">counter</text><line x1="196" y1="52" x2="500" y2="52" stroke="#B8AEA2" stroke-width="3"/><g><g><line x1="228" y1="52" x2="228" y2="62" stroke="#B8AEA2" stroke-width="2"/><rect x="212" y="62" width="32" height="42" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="228" y="118" text-anchor="middle" fill="#9A938A" font-size="9">25</text></g><g><line x1="288" y1="52" x2="288" y2="62" stroke="#B8AEA2" stroke-width="2"/><rect x="272" y="62" width="32" height="42" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="288" y="118" text-anchor="middle" fill="#9A938A" font-size="9">26</text></g><g><line x1="348" y1="52" x2="348" y2="62" stroke="#2D8B55" stroke-width="2"/><rect x="330" y="62" width="36" height="46" rx="5" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2.5"/><text x="348" y="122" text-anchor="middle" fill="#276b45" font-size="10" font-weight="bold">27</text></g><g><line x1="408" y1="52" x2="408" y2="62" stroke="#B8AEA2" stroke-width="2"/><rect x="392" y="62" width="32" height="42" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="408" y="118" text-anchor="middle" fill="#9A938A" font-size="9">28</text></g><g><line x1="466" y1="52" x2="466" y2="62" stroke="#B8AEA2" stroke-width="2"/><rect x="450" y="62" width="32" height="42" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="466" y="118" text-anchor="middle" fill="#9A938A" font-size="9">29</text></g></g><path d="M132 92 Q 240 150 340 116" fill="none" stroke="#2D8B55" stroke-width="1.5" stroke-dasharray="4,3"/><text x="252" y="148" text-anchor="middle" fill="#276b45" font-size="9">the number walks you straight to the right hook</text><text x="260" y="180" text-anchor="middle" fill="#6B645E" font-size="10">no searching — a number IS the address</text></g></svg>
%%%

A model keeps every word's embedding on one wall of numbered hooks. That wall is the [[embedding matrix||One big table of numbers: one row per word the model knows. Row number i holds the embedding for the word whose id is i.]] — a fancy phrase for "a big table where each row is one word's list of numbers."

Every word first gets its ticket number, called a [[token id||The integer ticket-number a word is given. The model uses it to grab that word's row from the table.]]. (The model calls the word itself a [[token||One word, or a word-piece — the basic chunk a model reads text in.]].) To embed a word, the model just goes to that row. That's the whole "lookup."

**Where the coat-check breaks down:** a coat-check hands back the coat *you* brought in, but the model's rows were never put there by you — they were **learned** (that's the next concept). And one word always maps to exactly one row, which turns out to be a real limitation we'll meet later today.

#### The wall, drawn out
Here is our whole toy model's wall — seven words, seven hooks. Real models have tens of thousands, but the idea does not change one bit.

%%% svg
<svg viewBox="0 0 520 268" role="img" aria-label="The toy embedding table drawn as seven numbered rows. Row 0 cat is highlighted and holds 0.9 0.8 0.0 0.3. Row 1 is dog, row 2 car, row 3 the, row 4 bank, row 5 bites, row 6 man. A label says the ticket number is just the row number."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Give the ticket number → get that word's row</text><rect x="18" y="86" width="96" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="66" y="106" text-anchor="middle" fill="#8A6D3B">"cat"</text><text x="66" y="122" text-anchor="middle" fill="#6B645E" font-size="10">ticket 0</text><text x="124" y="112" fill="#B8AEA2" font-size="16">→</text><rect x="152" y="30" width="300" height="222" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="302" y="48" text-anchor="middle" fill="#6B645E" font-size="10">the embedding table — one row per known word</text><rect x="162" y="56" width="280" height="26" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="174" y="73" fill="#276b45">row 0  cat</text><text x="434" y="73" text-anchor="end" fill="#2C2A28" font-weight="bold">[0.9, 0.8, 0.0, 0.3]</text><rect x="162" y="84" width="280" height="24" rx="3" fill="#FFFFFF" stroke="#E5DFD6"/><text x="174" y="100" fill="#6B645E">row 1  dog</text><text x="434" y="100" text-anchor="end" fill="#9A938A">[0.8, 0.9, 0.0, 0.4]</text><rect x="162" y="110" width="280" height="24" rx="3" fill="#FFFFFF" stroke="#E5DFD6"/><text x="174" y="126" fill="#6B645E">row 2  car</text><text x="434" y="126" text-anchor="end" fill="#9A938A">[0.0, 0.0, 0.9, 0.3]</text><rect x="162" y="136" width="280" height="24" rx="3" fill="#FFFFFF" stroke="#E5DFD6"/><text x="174" y="152" fill="#6B645E">row 3  the</text><text x="434" y="152" text-anchor="end" fill="#9A938A">[0.1, 0.0, 0.1, 0.1]</text><rect x="162" y="162" width="280" height="24" rx="3" fill="#FFFFFF" stroke="#E5DFD6"/><text x="174" y="178" fill="#6B645E">row 4  bank</text><text x="434" y="178" text-anchor="end" fill="#9A938A">[0.4, 0.4, 0.4, 0.4]</text><rect x="162" y="188" width="280" height="24" rx="3" fill="#FFFFFF" stroke="#E5DFD6"/><text x="174" y="204" fill="#6B645E">row 5  bites</text><text x="434" y="204" text-anchor="end" fill="#9A938A">[0.2, 0.1, 0.0, 0.6]</text><rect x="162" y="214" width="280" height="24" rx="3" fill="#FFFFFF" stroke="#E5DFD6"/><text x="174" y="230" fill="#6B645E">row 6  man</text><text x="434" y="230" text-anchor="end" fill="#9A938A">[0.3, 0.1, 0.0, 0.2]</text><text x="302" y="248" text-anchor="middle" fill="#9A938A" font-size="9">7 rows × 4 numbers — the ticket number IS the row number</text><text x="462" y="106" fill="#B8AEA2" font-size="16">→</text><text x="492" y="102" text-anchor="middle" fill="#276b45" font-size="10">that</text><text x="492" y="114" text-anchor="middle" fill="#276b45" font-size="10">row</text></g></svg>
%%%

Small honest aside while you have the wall in front of you: `bites` is a verb, so our four noun-ish labels barely fit it — the only one that says anything is *is-fast* (a bite is a quick action), and the rest sit near zero. That awkwardness is precisely why a real model invents its own hidden questions instead of borrowing ours.

#### A whole sentence through the counter
Follow "the cat" through, one move at a time.

%%% steps
step: cut the text into tokens — "the cat" → ["the", "cat"]
why: the model reads chunks, not letters; each chunk must be something the wall has a hook for
step: swap each token for its ticket number — ["the", "cat"] → [3, 0]
why: this list of ids is literally ALL the model receives; the words themselves are gone from here on
step: fetch row 3, then row 0
why: two lookups, no math — therefore embedding a sentence is the cheapest step in the whole model
step: keep the two rows in the order they arrived
why: that stack (one row per token, top to bottom in sentence order) is the object every attention step from Day 2 onward eats
%%%

**So far:** text → ticket numbers → rows → a stack of rows.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="Two token ids, 3 and 0, each pull one row out of the embedding table, and the two rows stack into a small block with two rows and four columns, kept in sentence order."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">2 token ids in → a block of 2 rows out</text><rect x="18" y="52" width="70" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="53" y="69" text-anchor="middle" fill="#8A6D3B">id 3 "the"</text><rect x="18" y="94" width="70" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="53" y="111" text-anchor="middle" fill="#8A6D3B">id 0 "cat"</text><text x="98" y="90" fill="#B8AEA2" font-size="15">→</text><rect x="124" y="40" width="150" height="92" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="199" y="56" text-anchor="middle" fill="#6B645E" font-size="9">the table (every word)</text><rect x="132" y="62" width="134" height="16" rx="2" fill="#FCF3DC" stroke="#C99A12"/><text x="199" y="74" text-anchor="middle" fill="#8A6D3B" font-size="9">row 3 → [0.1, 0.0, 0.1, 0.1]</text><rect x="132" y="82" width="134" height="16" rx="2" fill="#EAF5EE" stroke="#2D8B55"/><text x="199" y="94" text-anchor="middle" fill="#276b45" font-size="9">row 0 → [0.9, 0.8, 0.0, 0.3]</text><rect x="132" y="102" width="134" height="16" rx="2" fill="#FFFFFF" stroke="#E5DFD6"/><text x="199" y="114" text-anchor="middle" fill="#9A938A" font-size="9">… all the other rows …</text><text x="284" y="90" fill="#B8AEA2" font-size="15">→</text><rect x="312" y="52" width="180" height="56" rx="6" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="402" y="70" text-anchor="middle" fill="#5E5191" font-size="9">[0.1, 0.0, 0.1, 0.1]  ← "the"</text><text x="402" y="88" text-anchor="middle" fill="#5E5191" font-size="9">[0.9, 0.8, 0.0, 0.3]  ← "cat"</text><text x="402" y="102" text-anchor="middle" fill="#9A938A" font-size="9">2 rows × 4 numbers</text><text x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">one row per token, kept top-to-bottom in sentence order — this block is what attention reads</text><text x="260" y="166" text-anchor="middle" fill="#9A938A" font-size="9">(train on many sentences at once and you just stack these blocks: sentences × tokens × numbers)</text></g></svg>
%%%

%%% insight
You already lean on this counter every day. When your phone offers the next word before you've typed it, the first thing it did was swap your words for ticket numbers and fetch their rows. That's it. The flashiest text tool you've used and the humble coat-check share a first step.
%%%

#### How big is the wall of hooks?
One row per word the model knows. That full set of known words is its [[vocabulary||The complete set of words (tokens) a model knows — one table row for each.]], and real models know tens of thousands. So the table is "words × list-length" numbers, which makes it one of the biggest single blocks in the whole model.

%%% insight
Let's put a number on "biggest". 50,000 words × a 512-long list ≈ **25.6 million numbers** — just to spell words, before the model has thought about anything. Three ordinary fixes keep that in check: keep the word-list modest (word-*pieces* instead of whole words, which we get to later today), pick a shorter list length, or reuse this one table at the model's output too — a trick called *weight tying*. Big tables are a design choice, not an accident.
%%%

Now watch it run. **Predict first:** "the cat" becomes two ticket numbers, then "cat" fetches its row. What row do you think `cat` returns?

%%% demo id=idlookup label="turn a sentence into row numbers"
predict: guess the two ticket numbers for "the cat" — and which row 'cat' pulls back
code: ids = [vocab[w] for w in "the cat".split()];  ids, E[ids[1]]
out: token ids: [3, 0]
out: row for 'cat' (id 0): [0.9, 0.8, 0.0, 0.3]
take: <b>A sentence became a list of ticket numbers, and each number just grabs its row.</b> "the" is id 3, "cat" is id 0, so embedding "cat" means "go to row 0" — no math, just fetch. This tiny step (text → ids → rows) runs at the very start of every model, on every sentence you ever type.
%%%

!!! c-info 🪜
<b>Optional (skippable) — the same lookup, written as multiplication.</b><br>• Put a 1 in slot <i>i</i> of a long all-zeros row (that's the one-hot vector from the next concept) and multiply it by the table <code>E</code>: out comes row <i>i</i>. So <code>onehot(i) × E = E[i]</code>.<br>• Why bother? Because it shows the table is just an ordinary learnable layer — the same kind of thing you trained in Module 2 — which is exactly why gradient descent can improve it.<br>• Skip this and you lose nothing today.
!!!

@@@ concept id=c5 tag="They are learned" title="Nobody typed those numbers — they were learned" gotit="Got that they're learned"
Now the part that surprises almost everyone. Who *chose* 0.9, 0.8, 0.0? Did an engineer type a report card for every word in the language? No. The numbers are **learned**.

Think of how *you* learned what "dog" means. Nobody handed you a definition sheet as a baby.

You just heard the word over and over, sitting next to words like "bark", "walk", "puppy", "leash". Slowly your brain filed "dog" next to those. Company taught you meaning.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A child hears the word dog again and again, always surrounded by the words bark, walk, puppy and leash. A thought bubble shows the child filing dog next to those words. A caption says company teaches meaning."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">How a kid learns "dog": by the company it keeps</text><circle cx="76" cy="96" r="18" fill="#FCF3DC" stroke="#8A6D3B" stroke-width="1.5"/><circle cx="70" cy="92" r="2" fill="#8A6D3B"/><circle cx="83" cy="92" r="2" fill="#8A6D3B"/><path d="M69 103 Q76 109 84 103" fill="none" stroke="#8A6D3B" stroke-width="1.5"/><line x1="76" y1="114" x2="76" y2="152" stroke="#8A6D3B" stroke-width="2"/><text x="76" y="172" text-anchor="middle" fill="#6B645E" font-size="9">a kid, listening</text><g fill="#EAF0FA" stroke="#5E5191"><rect x="126" y="42" width="60" height="22" rx="10"/><rect x="204" y="34" width="62" height="22" rx="10"/><rect x="132" y="126" width="66" height="22" rx="10"/><rect x="212" y="140" width="60" height="22" rx="10"/></g><g fill="#5E5191" font-size="10" text-anchor="middle"><text x="156" y="57">bark</text><text x="235" y="49">leash</text><text x="165" y="141">walk</text><text x="242" y="155">puppy</text></g><rect x="176" y="80" width="82" height="30" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><text x="217" y="100" text-anchor="middle" fill="#8A6D3B" font-size="12" font-weight="bold">"dog"</text><g stroke="#B8AEA2" stroke-dasharray="3,2"><line x1="186" y1="64" x2="204" y2="82"/><line x1="235" y1="56" x2="240" y2="80"/><line x1="190" y1="126" x2="205" y2="110"/><line x1="242" y1="140" x2="240" y2="110"/></g><path d="M300 96 q 14 -10 28 0 q 14 -10 28 0" fill="none" stroke="#B8AEA2"/><text x="330" y="86" text-anchor="middle" fill="#9A938A" font-size="9">so the kid files it…</text><rect x="368" y="56" width="136" height="86" rx="10" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="436" y="76" text-anchor="middle" fill="#276b45" font-size="9">the "things about dogs"</text><text x="436" y="90" text-anchor="middle" fill="#276b45" font-size="9">drawer</text><text x="436" y="112" text-anchor="middle" fill="#276b45" font-size="11" font-weight="bold">dog · bark · leash</text><text x="436" y="128" text-anchor="middle" fill="#276b45" font-size="11" font-weight="bold">walk · puppy</text><text x="260" y="192" text-anchor="middle" fill="#6B645E" font-size="10">nobody gave a definition — the company around the word did all the teaching</text></g></svg>
%%%

A model does exactly this. It reads mountains of text and notices which words keep showing up in the *same company*. Words with similar surroundings get nudged closer together on the map, a tiny bit at a time. Do that for long enough and the map organizes itself: animals in one neighborhood, foods in another, colors in another. Nobody placed them — the reading did.

**Where the growing-up analogy breaks down:** a baby also touches, tastes and sees the world, while a model only ever sees text — so it learns which words *go together* without ever meeting a real dog. Its map is built from word-company, never from truth.

#### How a row actually moves
Same training loop you met in Module 2, pointed at the word table.

%%% steps
step: the random start — every row is filled with meaningless noise
why: at this moment "cat" and "dog" are strangers; a fresh (or frozen-random) table carries no meaning at all
step: the model reads a sentence and guesses badly
why: a bad guess produces a signal that says "these numbers were wrong, and in which direction"
step: the nudge — gradient descent tugs the rows that were involved
why: this is the SAME update that trained your neuron in Module 2, just applied to word rows
step: mostly the words in that sentence move
why: rows for words in the sentence get the big tug, and absent words get little or none — so the table learns in small patches (if a team reuses the table at the output with weight tying, absent rows do feel a small pull too)
step: repeat across a mountain of text
why: therefore words that keep the same company drift together, and the neighborhoods appear on their own
%%%

**So far:** noise in, a mountain of reading, and the map emerges without anyone placing a dot.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Before and after. On the left, at the start of training, four word dots are scattered randomly with no pattern. An arrow labelled after reading lots of text points right. On the right, the same dots have organized themselves: cat and dog cluster together, car and bus cluster together, apart from each other."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Training organizes the map — nobody places the dots by hand</text><rect x="30" y="36" width="180" height="150" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="120" y="54" text-anchor="middle" fill="#9A938A" font-size="10">at the start: random noise</text><circle cx="70" cy="90" r="5" fill="#B8AEA2"/><text x="70" y="82" text-anchor="middle" fill="#9A938A" font-size="9">cat</text><circle cx="170" cy="150" r="5" fill="#B8AEA2"/><text x="170" y="166" text-anchor="middle" fill="#9A938A" font-size="9">dog</text><circle cx="90" cy="160" r="5" fill="#B8AEA2"/><text x="90" y="176" text-anchor="middle" fill="#9A938A" font-size="9">car</text><circle cx="160" cy="80" r="5" fill="#B8AEA2"/><text x="160" y="72" text-anchor="middle" fill="#9A938A" font-size="9">bus</text><g><text x="240" y="106" text-anchor="middle" fill="#8A6D3B" font-size="16">→</text><text x="240" y="126" text-anchor="middle" fill="#8A6D3B" font-size="9">reads</text><text x="240" y="138" text-anchor="middle" fill="#8A6D3B" font-size="9">lots of</text><text x="240" y="150" text-anchor="middle" fill="#8A6D3B" font-size="9">text</text></g><rect x="270" y="36" width="220" height="150" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="380" y="54" text-anchor="middle" fill="#276b45" font-size="10">after training: organized</text><ellipse cx="330" cy="90" rx="46" ry="30" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="3,2"/><circle cx="316" cy="88" r="5" fill="#2D8B55"/><text x="316" y="80" text-anchor="middle" fill="#276b45" font-size="9">cat</text><circle cx="346" cy="96" r="5" fill="#2D8B55"/><text x="346" y="112" text-anchor="middle" fill="#276b45" font-size="9">dog</text><ellipse cx="430" cy="150" rx="46" ry="28" fill="#EFEAF7" stroke="#7C6DAA" stroke-dasharray="3,2"/><circle cx="416" cy="148" r="5" fill="#7C6DAA"/><text x="416" y="140" text-anchor="middle" fill="#5E5191" font-size="9">car</text><circle cx="446" cy="156" r="5" fill="#7C6DAA"/><text x="446" y="172" text-anchor="middle" fill="#5E5191" font-size="9">bus</text></g></svg>
%%%

%%% insight
Notice what this quietly means: a *rare* word barely gets tugged, so its row stays close to its random start — noisy and untrustworthy, even in a great model. That's a real, sneaky failure. The fix arrives later today: let a rare word borrow rows from common word-*pieces* it shares, so it rides on well-trained numbers instead of its own lonely ones.
%%%

Let's watch a row learn — this time on a *real* trained table, not our seven-row toy. **Predict first:** at the very start, cat-vs-dog scores about 0.0. After training on lots of text, what do you think it becomes?

%%% demo id=learned label="score cat vs dog before and after training"
predict: cat vs dog at random start ≈ 0.0 — what does it become after training?
code: # illustration from a REAL training run: a 512-number table reading real text
code: # (not today's 4-number toy wall — you can't reproduce these seven rows)
code: cos(cat, dog) at step 0  →  after 10k sentences  →  after 1M sentences
out: step 0:            -0.02
out: after 10k:          0.41
out: after 1M:           0.97
take: <b>The same two rows, three moments in training.</b> At step 0 the rows are pure noise, and in a long 512-number list two random arrows land almost exactly at a right angle — hence a score near 0. Nobody edited them by hand afterwards; reading did it. The score climbs because "cat" and "dog" keep turning up in the same company ("the ___ barked", "my ___ is hungry"), so every nudge pulls them a little closer on the map.
%%%

#### The party trick this buys you
Because the map is organized, *directions* on it start to mean things too. The famous example: take the row for "king", subtract "man", add "woman", and the spot you land on is close to "queen". You have probably used a cousin of this in your pocket — a translation app leans on the same idea, matching a word to the spot a word in another language occupies. Two honest notes: this is an *observed habit* of trained tables, not a guarantee, and the individual slots are still not human-readable. Only the relative geometry carries the information.

!!! c-warn ⚖️
<b>The uncomfortable side of "it learns from us".</b><br>• If human text treats some group unfairly, those patterns land in the geometry too, and stereotype pairs end up suspiciously close.<br>• <b>Cause:</b> the map is built from who-appears-near-whom in text written by people. The vectors are a mirror, and mirrors copy flaws.<br>• <b>Remedy:</b> real teams curate the training text and run explicit bias probes on the finished table *before* it ships.
!!!

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it:</b> "An embedding is a learned lookup table — token id <code>i</code> returns row <code>i</code>, and because training pulls together words that appear in similar contexts, two rows point the same way exactly when their words are used alike. Nobody hand-writes the numbers; the reading writes them." You can already say every piece of that.
!!!

@@@ concept id=c6 tag="The old way" title="The clumsy first idea — one-hot" gotit="Got why one-hot lost"
Before this neat map existed, people tried a much cruder way to turn a word into numbers. Seeing *why it flopped* makes the map feel like the clever fix it is. Fun bit of history — let's enjoy it.

Imagine a **classroom where, to name one kid, exactly one kid raises a hand** and everyone else keeps theirs down.

It works perfectly for naming. But look at that room for a second: every kid is exactly as far from every other kid. There are no friend groups, no "these two sit together". Just a row of identical desks.

%%% svg
<svg viewBox="0 0 520 186" role="img" aria-label="A classroom row of five children. One child has a hand raised and the others keep hands down, which names that one child. A caption points out that every child is exactly the same distance from every other, so there are no friend groups."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">"Which kid?" — exactly one hand goes up</text><g><g><circle cx="70" cy="74" r="13" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="70" y1="87" x2="70" y2="118" stroke="#8A6D3B" stroke-width="2"/><line x1="70" y1="94" x2="56" y2="110" stroke="#8A6D3B" stroke-width="2"/><text x="70" y="136" text-anchor="middle" fill="#9A938A">down</text></g><g><circle cx="170" cy="74" r="13" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2.5"/><line x1="170" y1="87" x2="170" y2="118" stroke="#2D8B55" stroke-width="2.5"/><line x1="170" y1="92" x2="186" y2="52" stroke="#2D8B55" stroke-width="3"/><text x="170" y="136" text-anchor="middle" fill="#276b45" font-weight="bold">HAND UP</text><text x="170" y="150" text-anchor="middle" fill="#276b45">= "cat"</text></g><g><circle cx="270" cy="74" r="13" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="270" y1="87" x2="270" y2="118" stroke="#8A6D3B" stroke-width="2"/><line x1="270" y1="94" x2="256" y2="110" stroke="#8A6D3B" stroke-width="2"/><text x="270" y="136" text-anchor="middle" fill="#9A938A">down</text></g><g><circle cx="370" cy="74" r="13" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="370" y1="87" x2="370" y2="118" stroke="#8A6D3B" stroke-width="2"/><line x1="370" y1="94" x2="356" y2="110" stroke="#8A6D3B" stroke-width="2"/><text x="370" y="136" text-anchor="middle" fill="#9A938A">down</text></g><g><circle cx="460" cy="74" r="13" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="460" y1="87" x2="460" y2="118" stroke="#8A6D3B" stroke-width="2"/><line x1="460" y1="94" x2="446" y2="110" stroke="#8A6D3B" stroke-width="2"/><text x="460" y="136" text-anchor="middle" fill="#9A938A">down</text></g></g><g stroke="#C93B3B" stroke-dasharray="2,2"><line x1="83" y1="60" x2="157" y2="60"/><line x1="183" y1="60" x2="257" y2="60"/><line x1="283" y1="60" x2="357" y2="60"/><line x1="383" y1="60" x2="447" y2="60"/></g><text x="260" y="178" text-anchor="middle" fill="#C93B3B" font-size="10">every desk is the same distance from every other — no friend groups anywhere</text></g></svg>
%%%

To say "this word is *cat*," you make a huge row of switches — one switch per word the model knows — and flip *only cat's switch on*. All the rest stay off. That's [[one-hot||The old way to turn a word into numbers: a giant list that is all zeros except a single 1 in the slot for that one word.]] encoding: a giant list of zeros with a single `1` in that word's slot. "One-hot" = only one slot is hot.

**Where the raised hands break down:** pointing at one kid is unmistakable, so one-hot is a perfectly good *name tag* — it is a hopeless *map*, and meaning is what we came for.

#### Watch the flaw appear, one rung at a time
Use the closeness score you built earlier today on two one-hot rows. Our wall knows seven words, so each one-hot row is seven slots wide.

%%% steps
step: line up cat = [1, 0, 0, 0, 0, 0, 0] and dog = [0, 1, 0, 0, 0, 0, 0], then multiply slot by slot
why: 1×0, 0×1, and five 0×0 pairs — every single pair has a zero in it
step: add it up → 0
why: they never share a hot slot, therefore the score is exactly 0 no matter which two words you pick
step: now try cat and car = [0, 0, 1, 0, 0, 0, 0] — also 0
why: this is the collapse: cat-and-dog scores the SAME as cat-and-car, so "alike" became unmeasurable
step: count the width — seven slots here, but a real model knows 50,000 words, so each list is 50,000 long
why: 49,999 of those numbers are zero, which means you pay for a stadium to seat one person
%%%

**So far:** one-hot can name a word perfectly and say nothing at all about meaning.

%%% svg
<svg viewBox="0 0 520 206" role="img" aria-label="One-hot vectors for cat, dog and car, each drawn as a row of seven boxes: all zeros with a single 1 in a different position. Below, a note that every pair of these is exactly the same distance apart, so none is more similar to any other."><g font-family="monospace" font-size="12"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One-hot: seven slots because our wall knows seven words — one switch on</text><text x="24" y="56" fill="#6B645E">cat</text><g fill="#FFFFFF" stroke="#E5DFD6"><rect x="70" y="41" width="26" height="20"/><rect x="98" y="41" width="26" height="20"/><rect x="126" y="41" width="26" height="20"/><rect x="154" y="41" width="26" height="20"/><rect x="182" y="41" width="26" height="20"/><rect x="210" y="41" width="26" height="20"/><rect x="238" y="41" width="26" height="20"/></g><rect x="70" y="41" width="26" height="20" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="83" y="56" text-anchor="middle" fill="#276b45" font-weight="bold">1</text><text x="111" y="56" text-anchor="middle" fill="#9A938A">0</text><text x="139" y="56" text-anchor="middle" fill="#9A938A">0</text><text x="167" y="56" text-anchor="middle" fill="#9A938A">0</text><text x="195" y="56" text-anchor="middle" fill="#9A938A">0</text><text x="223" y="56" text-anchor="middle" fill="#9A938A">0</text><text x="251" y="56" text-anchor="middle" fill="#9A938A">0</text><text x="278" y="56" fill="#9A938A" font-size="10">… 50,000 slots in a real model</text><text x="24" y="96" fill="#6B645E">dog</text><g fill="#FFFFFF" stroke="#E5DFD6"><rect x="70" y="81" width="26" height="20"/><rect x="98" y="81" width="26" height="20"/><rect x="126" y="81" width="26" height="20"/><rect x="154" y="81" width="26" height="20"/><rect x="182" y="81" width="26" height="20"/><rect x="210" y="81" width="26" height="20"/><rect x="238" y="81" width="26" height="20"/></g><rect x="98" y="81" width="26" height="20" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="83" y="96" text-anchor="middle" fill="#9A938A">0</text><text x="111" y="96" text-anchor="middle" fill="#276b45" font-weight="bold">1</text><text x="139" y="96" text-anchor="middle" fill="#9A938A">0</text><text x="167" y="96" text-anchor="middle" fill="#9A938A">0</text><text x="195" y="96" text-anchor="middle" fill="#9A938A">0</text><text x="223" y="96" text-anchor="middle" fill="#9A938A">0</text><text x="251" y="96" text-anchor="middle" fill="#9A938A">0</text><text x="24" y="136" fill="#6B645E">car</text><g fill="#FFFFFF" stroke="#E5DFD6"><rect x="70" y="121" width="26" height="20"/><rect x="98" y="121" width="26" height="20"/><rect x="126" y="121" width="26" height="20"/><rect x="154" y="121" width="26" height="20"/><rect x="182" y="121" width="26" height="20"/><rect x="210" y="121" width="26" height="20"/><rect x="238" y="121" width="26" height="20"/></g><rect x="126" y="121" width="26" height="20" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="83" y="136" text-anchor="middle" fill="#9A938A">0</text><text x="111" y="136" text-anchor="middle" fill="#9A938A">0</text><text x="139" y="136" text-anchor="middle" fill="#276b45" font-weight="bold">1</text><text x="167" y="136" text-anchor="middle" fill="#9A938A">0</text><text x="195" y="136" text-anchor="middle" fill="#9A938A">0</text><text x="223" y="136" text-anchor="middle" fill="#9A938A">0</text><text x="251" y="136" text-anchor="middle" fill="#9A938A">0</text><text x="260" y="174" text-anchor="middle" fill="#C93B3B" font-size="11">cat, dog, car are ALL the exact same distance apart</text><text x="260" y="192" text-anchor="middle" fill="#C93B3B" font-size="10">→ the numbers say NOTHING about which words are alike</text></g></svg>
%%%

%%% insight
Feel the size of that second problem: the input width is *glued* to the number of words you know. Learn one more word and every word's list gets longer. The embedding table cuts that cord — a few hundred numbers per word no matter how big the word-list grows, and every one of those numbers doing real work. That is why the field switched and never looked back.
%%%

#### The fix, in one move
Use a short list of a few hundred numbers instead of 50,000 slots — that's what makes it a [[dense||"Dense" = a short list where every slot carries real information, the opposite of one-hot's long list of mostly zeros.]] vector — and *let training arrange them* so similar words land close. Short instead of huge. Meaningful instead of blind.

You met the first famous versions of this idea under the names **word2vec** and **GloVe**: one vector per word, learned from "similar contexts give similar vectors." They are the ancestors of today's table — and the reason "one fixed vector per word" is a limit worth naming in a moment.

Head-to-head time. **Predict first:** we score cat-vs-dog and cat-vs-car *twice*, once with one-hot and once with embeddings. Which method notices that the animals are alike?

%%% demo id=onehot label="one-hot vs embedding — measure closeness"
predict: with one-hot, will cat-vs-dog beat cat-vs-car? and what does the embedding say?
code: cos(onehot("cat"), onehot("dog")),  cos(onehot("cat"), onehot("car")),  cos(embed("cat"), embed("dog"))
out: ONE-HOT   cos(cat, dog) = 0.0
out: ONE-HOT   cos(cat, car) = 0.0
out: EMBEDDING cos(cat, dog) = 0.99
take: <b>One-hot gives 0.0 for BOTH pairs — it literally cannot tell cat-and-dog apart from cat-and-car.</b> The embedding gives the animals 0.99. Same words, but only the embedding knows they are alike. That gap is why every modern model uses embeddings.
%%%

Little side note for the curious: the ancestor of the ancestor was **bag-of-words** — count how often each word appears in a document and ignore order entirely. Great for "is this email spam?", hopeless for "who bit whom", for exactly the reason you just saw.

@@@ concept id=c7 tag="Unknown words" title="A puzzle: a word the model has never seen" gotit="Got the OOV fix"
Here's a puzzle to chew on. The table has one row per known word. So what happens when a word arrives that *isn't in the table*? New slang, a rare name, a typo. There's no hook for it on the wall.

Back to the **coat-check**, which now has a rule: it only has hooks for coat styles it has seen before.

A guest arrives with a style the attendant has never seen. Every hook is labelled for something else. There is no ticket to hand over. What now?

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A coat-check rail with four labelled hooks, all taken by known coat styles. A guest arrives holding an unfamiliar style and there is no hook labelled for it, so the attendant has nothing to hand over."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">The wall of hooks has no hook for something new</text><line x1="150" y1="46" x2="500" y2="46" stroke="#B8AEA2" stroke-width="3"/><g><g><line x1="190" y1="46" x2="190" y2="56" stroke="#B8AEA2" stroke-width="2"/><rect x="174" y="56" width="32" height="40" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="190" y="110" text-anchor="middle" fill="#9A938A">known</text></g><g><line x1="262" y1="46" x2="262" y2="56" stroke="#B8AEA2" stroke-width="2"/><rect x="246" y="56" width="32" height="40" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="262" y="110" text-anchor="middle" fill="#9A938A">known</text></g><g><line x1="334" y1="46" x2="334" y2="56" stroke="#B8AEA2" stroke-width="2"/><rect x="318" y="56" width="32" height="40" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="334" y="110" text-anchor="middle" fill="#9A938A">known</text></g><g><line x1="406" y1="46" x2="406" y2="56" stroke="#B8AEA2" stroke-width="2"/><rect x="390" y="56" width="32" height="40" rx="5" fill="#EFEAF7" stroke="#7C6DAA"/><text x="406" y="110" text-anchor="middle" fill="#9A938A">known</text></g><g><line x1="472" y1="46" x2="472" y2="56" stroke="#C93B3B" stroke-width="2" stroke-dasharray="3,2"/><rect x="456" y="56" width="32" height="40" rx="5" fill="#FFFFFF" stroke="#C93B3B" stroke-dasharray="3,2"/><text x="472" y="79" text-anchor="middle" fill="#C93B3B" font-size="14">?</text><text x="472" y="110" text-anchor="middle" fill="#C93B3B">no hook</text></g></g><circle cx="56" cy="70" r="12" fill="#FCF3DC" stroke="#8A6D3B"/><line x1="56" y1="82" x2="56" y2="116" stroke="#8A6D3B" stroke-width="2"/><line x1="56" y1="90" x2="88" y2="84" stroke="#8A6D3B" stroke-width="2"/><rect x="88" y="70" width="44" height="28" rx="5" fill="#FDECEC" stroke="#C93B3B"/><text x="110" y="88" text-anchor="middle" fill="#C93B3B" font-size="9">a style</text><text x="110" y="112" text-anchor="middle" fill="#C93B3B" font-size="9">never seen</text><text x="260" y="146" text-anchor="middle" fill="#6B645E" font-size="10">the attendant has no ticket to give — the wall was built before this guest arrived</text><text x="260" y="164" text-anchor="middle" fill="#9A938A" font-size="9">same story for a word the model never met in training</text></g></svg>
%%%

That's the [[out-of-vocabulary||"OOV" for short — a word that isn't in the model's known-word list, so it has no row to look up.]] problem — people say "OOV". The cause is simple: the word-list is fixed before training, so a word that never appeared has no row. Don't worry, there's a genuinely clever escape, and it's fun to work out.

**Where the coat-check breaks down:** a coat can't be cut into halves that each fit a smaller hook. A *word* can — and that is exactly the clever fix.

#### The fix: break the word into pieces it already knows
Follow "tokenization" through the door.

%%% steps
step: the miss — look up "tokenization", find no row
why: it never appeared in training as a whole word, so the wall has no hook for it
step: the split — chop it into "token" + "ization"
why: the model keeps rows for common word-PIECES, and both of these pieces are on the wall
step: the two lookups — fetch a row for each piece
why: therefore an unseen word still arrives as real, well-trained numbers instead of a shrug
step: the win — meaning survives
why: "token" carries most of the sense, and "ization" says "this is a process" — nothing was thrown away
%%%

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="The word tokenization is not in the table. Two fixes are shown. Fix one, the crude fallback, maps it all to a single shared unknown row. Fix two, the smart fix, breaks it into known pieces token and ization, each of which has its own row."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Unknown word "tokenization" → two ways to handle it</text><rect x="180" y="30" width="160" height="26" rx="5" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="260" y="47" text-anchor="middle" fill="#C93B3B">"tokenization" — no row!</text><line x1="200" y1="56" x2="120" y2="82" stroke="#B8AEA2"/><line x1="320" y1="56" x2="400" y2="82" stroke="#B8AEA2"/><text x="120" y="78" text-anchor="middle" fill="#8A6D3B" font-size="10">crude fallback</text><rect x="60" y="86" width="130" height="26" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="125" y="103" text-anchor="middle" fill="#8A6D3B">→ [UNK] (one shared row)</text><text x="125" y="128" text-anchor="middle" fill="#9A938A" font-size="9">every unknown word</text><text x="125" y="140" text-anchor="middle" fill="#9A938A" font-size="9">looks identical</text><text x="400" y="78" text-anchor="middle" fill="#276b45" font-size="10">smart fix: split into pieces</text><rect x="330" y="86" width="66" height="26" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="363" y="103" text-anchor="middle" fill="#276b45">"token"</text><rect x="402" y="86" width="72" height="26" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="438" y="103" text-anchor="middle" fill="#276b45">"ization"</text><text x="400" y="128" text-anchor="middle" fill="#276b45" font-size="9">each piece IS known,</text><text x="400" y="140" text-anchor="middle" fill="#276b45" font-size="9">so each gets a real row</text><text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">splitting into known pieces keeps the meaning; one shared [UNK] throws it away</text></g></svg>
%%%

This is [[subword tokenization||The fix for unknown words: chop a rare word into smaller pieces that ARE in the vocabulary, so each piece gets its own row.]], and you'll hear three brand names for it: **BPE**, **WordPiece**, and **SentencePiece**. Different recipes, same idea.

**So far:** no whole-word row needed — pieces are enough. The blunt old fallback was one shared [[UNK token||A single "unknown" row that every unseen word gets mapped to — cheap, but it makes all unknown words look identical.]] instead: a single "unknown" row that *every* unseen word maps to. It never crashes, but "tokenization", "flibbertigibbet" and a rare surname all collapse into the same vector. Splitting is far kinder, which is why it won.

%%% insight
Bonus prize: this same fix quietly repairs the rare-word problem from the "they are learned" concept. A rare word split into common pieces now rides on rows that got *millions* of nudges instead of three. One idea, two failures fixed — that's the sign of a good idea.
%%%

Watch the split happen. **Predict first:** "cat" is known; "tokenization" was never seen whole. How many pieces does each become?

%%% demo id=oov label="known word vs unknown word"
predict: how many pieces for "cat"? and for "tokenization"?
code: tokenize("cat"),  tokenize("tokenization")
out: known:   ['cat']
out: unknown: ['token', 'ization']
take: <b>"cat" is one known token — one row.</b> "tokenization" splits into "token" + "ization", two pieces the model DOES know. No crash, and the meaning of the pieces survives. That's why subword tokenization beats a blunt [UNK] — and why it's genuinely hard to stump a modern model with a new word.
%%%

!!! c-warn 🧷
<b>The practical trap that bites real projects.</b><br>• A token is matched as *text*, character for character. So <code>"Dog"</code>, <code>"dog"</code> and <code>" dog"</code> (with a leading space) can be three different tokens with three different rows.<br>• <b>Remedy:</b> pick one text-cleaning rule and never change it, and always ship a model with the exact tokenizer it was trained with. Use a different splitter at test time than at training time and your beautiful map turns to mush.
!!!

@@@ concept id=c8 tag="One word, two meanings" title="The blind spot that opens the rest of this module" gotit="Got the context-free limit"
Time for the puzzle that sets up the *entire rest of this module*. Our map is lovely, but it has a blind spot — and spotting it is the doorway to everything ahead.

Think of a **single name-tag sticker** you wear to a party. You write "Alex" on it in the hallway, and that's it for the night.

In the kitchen you'd love it to say "Alex, who cooks". By the pool, "Alex, who swims". It can't. One sticker, one message, every room.

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="One name-tag sticker reading Alex is worn into two different rooms, a kitchen and a pool. In both rooms the sticker says exactly the same thing, because a sticker written once cannot change to fit the room."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One sticker, written once, worn into every room</text><rect x="200" y="30" width="120" height="42" rx="6" fill="#FFFFFF" stroke="#C99A12" stroke-width="2"/><text x="260" y="48" text-anchor="middle" fill="#8A6D3B" font-size="9">HELLO my name is</text><text x="260" y="64" text-anchor="middle" fill="#2C2A28" font-size="13" font-weight="bold">Alex</text><line x1="222" y1="72" x2="140" y2="100" stroke="#B8AEA2"/><line x1="298" y1="72" x2="380" y2="100" stroke="#B8AEA2"/><rect x="40" y="102" width="200" height="52" rx="8" fill="#EAF0FA" stroke="#5E5191"/><text x="140" y="120" text-anchor="middle" fill="#5E5191" font-size="10">in the kitchen</text><text x="140" y="136" text-anchor="middle" fill="#5E5191" font-size="9">wants to say "Alex, who cooks"</text><text x="140" y="149" text-anchor="middle" fill="#C93B3B" font-size="9">still just says "Alex"</text><rect x="280" y="102" width="200" height="52" rx="8" fill="#FCF3DC" stroke="#C99A12"/><text x="380" y="120" text-anchor="middle" fill="#8A6D3B" font-size="10">at the pool</text><text x="380" y="136" text-anchor="middle" fill="#8A6D3B" font-size="9">wants to say "Alex, who swims"</text><text x="380" y="149" text-anchor="middle" fill="#C93B3B" font-size="9">still just says "Alex"</text><text x="260" y="174" text-anchor="middle" fill="#6B645E" font-size="10">the room changes; the sticker cannot</text></g></svg>
%%%

Our table has the same problem: **one word gets exactly one row, one fixed list of numbers, forever.** But some words mean different things in different sentences. Take "bank" — row 4 on our wall:

- "I sat on the **bank** of the river." → the edge of a river
- "I put my money in the **bank**." → a place for money

Same spelling, totally different meaning. That's called [[polysemy||When one spelling has two or more meanings, like river "bank" vs money "bank".]] — and the table has only *one* row for "bank".

**Where the name-tag breaks down:** a person can at least *say* "actually, I'm the cook here." A static row can't say anything — it never looks around at its sentence at all.

#### Let's prove it, not just claim it
Claims are cheap, so let's measure this one. Here is the picture of what happens to both sentences.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="The word bank appears in two very different sentences, a river sentence and a money sentence, but both lookups go to the same row 4 of the fixed table and return the exact same four numbers, so the table cannot tell the two meanings apart."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Two meanings, one row — the sticker problem in numbers</text><rect x="20" y="40" width="204" height="30" rx="5" fill="#EAF0FA" stroke="#5E5191"/><text x="122" y="59" text-anchor="middle" fill="#5E5191" font-size="10">"…the bank of the river"</text><rect x="20" y="120" width="204" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="122" y="139" text-anchor="middle" fill="#8A6D3B" font-size="10">"…money in the bank"</text><text x="122" y="94" text-anchor="middle" fill="#9A938A" font-size="9">both ask for token "bank" → id 4</text><line x1="224" y1="55" x2="288" y2="88" stroke="#B8AEA2"/><line x1="224" y1="135" x2="288" y2="102" stroke="#B8AEA2"/><rect x="290" y="72" width="196" height="46" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="388" y="90" text-anchor="middle" fill="#C93B3B" font-size="10">row 4 of the table — "bank"</text><text x="388" y="108" text-anchor="middle" fill="#C93B3B" font-weight="bold">[0.4, 0.4, 0.4, 0.4]</text><text x="388" y="138" text-anchor="middle" fill="#C93B3B" font-size="10">the SAME four numbers both times</text><text x="388" y="152" text-anchor="middle" fill="#9A938A" font-size="9">can't tell river from money</text><text x="260" y="180" text-anchor="middle" fill="#6B645E" font-size="10">the sentence around the word never enters the lookup at all</text></g></svg>
%%%

**Predict first:** we fetch "bank" for the river sentence and "bank" for the money sentence, then score how alike they are. Two different meanings — what score would you *hope* for? Now guess what the frozen table will actually give.

%%% demo id=polysemy label="score river-bank vs money-bank"
predict: two totally different meanings — what cosine score do you expect the frozen table to give?
code: cos(E[vocab["bank"]], E[vocab["bank"]])   # the river one and the money one
out: bank(river) -> row 4 -> [0.4, 0.4, 0.4, 0.4]
out: bank(money) -> row 4 -> [0.4, 0.4, 0.4, 0.4]
out: cos(bank_river, bank_money) = 1.00
take: <b>Both lookups return the identical row, so the score is a perfect 1.00 — the map insists the two "bank"s are the same word with the same meaning.</b> You'd hope for something low. The table simply cannot give one, because it only ever fetches one frozen row per spelling. This limit has a name: the table is <b>context-free</b>.
%%%

%%% insight
That `1.00` is the whole reason the next lesson exists. And notice how sneaky the failure is: nothing crashed, no error appeared, the model happily carried on with a *blurred average* of two meanings. Silent wrongness is the hardest kind to catch — which is why we make it show itself as a number.
%%%

#### Your reward: you found the doorway
You do **not** have to solve this today, and you don't need to know *how* the fix works yet. What you've done is spot exactly *why* it will be needed — which is the hard part. Carry today's `1.00` with you; it is the problem the very next lesson opens on.

%%% cards
🔮 | Coming next lesson | The fix for the frozen-"bank" limit is called **attention** — you'll build it from scratch on Day 2. Nothing to learn about it today.
🧩 | The thing it produces | Its output has a name too: **contextual embeddings** — a word's vector shaped by its actual sentence. Just a name to file away.
%%%

@@@ concept id=c9 tag="Who bit whom" title="The quieter blind spot: nothing marks a word's place" gotit="Got the position limit"
One more blind spot, and this one is quieter — so quiet that it is easy to state wrongly. Let's get it exactly right, because it is fun once you see it.

Play a game with three friends. You write **DOG**, **BITES**, **MAN** on three cards, hand one to each friend, and ask them to stand in a line. Read the line: the story is obvious.

Now turn your back. They shuffle. Read again — *MAN BITES DOG*, the opposite story. The cards themselves never changed. Nothing is written on any card that says "I was standing first."

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Three friends each hold a word card. In the first line they stand in the order dog, bites, man, which tells the story dog bites man. After they shuffle they stand in the order man, bites, dog, which tells the opposite story. A caption points out that no card says which place its holder stood in."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Three friends holding word-cards — then they shuffle</text><g><circle cx="52" cy="48" r="11" fill="#EAF5EE" stroke="#2D8B55"/><line x1="52" y1="59" x2="52" y2="82" stroke="#2D8B55" stroke-width="2"/><rect x="26" y="84" width="52" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="52" y="99" text-anchor="middle" fill="#276b45">DOG</text><circle cx="122" cy="48" r="11" fill="#FCF3DC" stroke="#C99A12"/><line x1="122" y1="59" x2="122" y2="82" stroke="#C99A12" stroke-width="2"/><rect x="94" y="84" width="56" height="22" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="122" y="99" text-anchor="middle" fill="#8A6D3B">BITES</text><circle cx="192" cy="48" r="11" fill="#EFEAF7" stroke="#7C6DAA"/><line x1="192" y1="59" x2="192" y2="82" stroke="#7C6DAA" stroke-width="2"/><rect x="166" y="84" width="52" height="22" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="192" y="99" text-anchor="middle" fill="#5E5191">MAN</text><text x="122" y="126" text-anchor="middle" fill="#6B645E" font-size="10">the story: the dog did the biting</text></g><g><text x="250" y="72" text-anchor="middle" fill="#8A6D3B" font-size="14">→</text><text x="250" y="92" text-anchor="middle" fill="#8A6D3B" font-size="9">they</text><text x="250" y="104" text-anchor="middle" fill="#8A6D3B" font-size="9">shuffle</text></g><g><circle cx="322" cy="48" r="11" fill="#EFEAF7" stroke="#7C6DAA"/><line x1="322" y1="59" x2="322" y2="82" stroke="#7C6DAA" stroke-width="2"/><rect x="296" y="84" width="52" height="22" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="322" y="99" text-anchor="middle" fill="#5E5191">MAN</text><circle cx="392" cy="48" r="11" fill="#FCF3DC" stroke="#C99A12"/><line x1="392" y1="59" x2="392" y2="82" stroke="#C99A12" stroke-width="2"/><rect x="364" y="84" width="56" height="22" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="392" y="99" text-anchor="middle" fill="#8A6D3B">BITES</text><circle cx="462" cy="48" r="11" fill="#EAF5EE" stroke="#2D8B55"/><line x1="462" y1="59" x2="462" y2="82" stroke="#2D8B55" stroke-width="2"/><rect x="436" y="84" width="52" height="22" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="462" y="99" text-anchor="middle" fill="#276b45">DOG</text><text x="392" y="126" text-anchor="middle" fill="#C93B3B" font-size="10">opposite story — same three cards</text></g><text x="260" y="164" text-anchor="middle" fill="#C93B3B" font-size="11">no card says "I was standing first"</text><text x="260" y="184" text-anchor="middle" fill="#6B645E" font-size="10">the story lived in the LINE, not in the cards — so if the line is ignored, the story is gone</text><text x="260" y="202" text-anchor="middle" fill="#9A938A" font-size="9">a word's row is exactly like one of those cards</text></g></svg>
%%%

Word rows are those cards. "dog bites man" and "man bites dog" use the same three words — same three rows off the wall — and the whole difference in meaning lives in the *order*.

**Where the card game breaks down:** your friends actually remember where they stood, and could tell you. A list of four numbers has no memory of anything.

#### Careful — say this one precisely
The rows *do* come back in sentence order. That part is fine. The gap is subtler, and worth watching assemble.

%%% svg
<svg viewBox="0 0 520 302" role="img" aria-label="The sentence dog bites man becomes rows 1, 5 and 6 stacked in that order. The sentence man bites dog becomes rows 6, 5 and 1 stacked in that order. The two stacks differ only in row order, and no row contains any number saying which slot it came from, so a stage that reads the rows all at once sees the same three rows both times."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The rows arrive in order — but nothing inside a row records its slot</text><text x="18" y="38" fill="#5E5191" font-weight="bold" font-size="11">"dog bites man"</text><g><rect x="18" y="44" width="90" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="63" y="57" text-anchor="middle" fill="#276b45" font-size="9">dog · row 1</text><text x="63" y="69" text-anchor="middle" fill="#276b45" font-size="8">[0.8, 0.9, …]</text><rect x="112" y="44" width="90" height="30" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="157" y="57" text-anchor="middle" fill="#8A6D3B" font-size="9">bites · row 5</text><text x="157" y="69" text-anchor="middle" fill="#8A6D3B" font-size="8">[0.2, 0.1, …]</text><rect x="206" y="44" width="90" height="30" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="251" y="57" text-anchor="middle" fill="#5E5191" font-size="9">man · row 6</text><text x="251" y="69" text-anchor="middle" fill="#5E5191" font-size="8">[0.3, 0.1, …]</text></g><text x="304" y="63" fill="#B8AEA2" font-size="13">→</text><rect x="326" y="38" width="176" height="44" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="414" y="53" text-anchor="middle" fill="#6B645E" font-size="9">stack A (top to bottom)</text><text x="414" y="68" text-anchor="middle" fill="#2C2A28" font-size="10">row 1 · row 5 · row 6</text><text x="18" y="112" fill="#C99A12" font-weight="bold" font-size="11">"man bites dog"</text><g><rect x="18" y="118" width="90" height="30" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="63" y="131" text-anchor="middle" fill="#5E5191" font-size="9">man · row 6</text><text x="63" y="143" text-anchor="middle" fill="#5E5191" font-size="8">[0.3, 0.1, …]</text><rect x="112" y="118" width="90" height="30" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="157" y="131" text-anchor="middle" fill="#8A6D3B" font-size="9">bites · row 5</text><text x="157" y="143" text-anchor="middle" fill="#8A6D3B" font-size="8">[0.2, 0.1, …]</text><rect x="206" y="118" width="90" height="30" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="251" y="131" text-anchor="middle" fill="#276b45" font-size="9">dog · row 1</text><text x="251" y="143" text-anchor="middle" fill="#276b45" font-size="8">[0.8, 0.9, …]</text></g><text x="304" y="137" fill="#B8AEA2" font-size="13">→</text><rect x="326" y="112" width="176" height="44" rx="6" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="414" y="127" text-anchor="middle" fill="#6B645E" font-size="9">stack B (top to bottom)</text><text x="414" y="142" text-anchor="middle" fill="#2C2A28" font-size="10">row 6 · row 5 · row 1</text><text x="260" y="176" text-anchor="middle" fill="#9A938A" font-size="9">(first 2 of each row's 4 numbers shown, to fit)</text><rect x="96" y="188" width="328" height="66" rx="8" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="260" y="206" text-anchor="middle" fill="#C93B3B" font-size="10">what the next stage sees if we ignore position:</text><text x="260" y="226" text-anchor="middle" fill="#C93B3B" font-size="11" font-weight="bold">{ row 1 , row 5 , row 6 }</text><text x="260" y="245" text-anchor="middle" fill="#C93B3B" font-size="9">the same three rows both times — and no number inside them says "I was first"</text><text x="260" y="276" text-anchor="middle" fill="#6B645E" font-size="10">who-bit-whom lives in the ORDER, so the order has to be written INTO the numbers</text><text x="260" y="294" text-anchor="middle" fill="#9A938A" font-size="9">that writing-in step has a name (positional encoding) and its own lesson later</text></g></svg>
%%%

Walk the figure with me, one rung at a time.

%%% steps
step: embed "dog bites man" — fetch row 1, row 5, row 6, and keep them in that order
why: the stack genuinely does arrive in sentence order; the ordering is sitting right there in how the rows are stacked
step: embed "man bites dog" — fetch row 6, row 5, row 1
why: the same three rows come back, just stacked the other way round
step: now look INSIDE one row — nothing in [0.8, 0.9, 0.0, 0.4] says "I was word number one"
why: a row only ever describes its word's meaning, therefore position was never written into the numbers
step: the catch — the next stage reads all the rows at once, as a set
why: shuffle the stack and that stage sees the same three rows, which means who-bit-whom is lost even though the order was technically there
%%%

**Predict first:** the two stacks *do* differ in order. So if a stage gathers them up as an unordered set, will the two sentences look different or identical to it?

%%% demo id=order label="shuffle the sentence, compare the bag of rows"
predict: same three words, opposite story — does the SET of rows differ between the two sentences?
code: rows("dog bites man"), rows("man bites dog")
code: set(rows_A) == set(rows_B)
out: A: [row 1, row 5, row 6]
out: B: [row 6, row 5, row 1]
out: same SET of rows? True
take: <b>Identical set, opposite story.</b> The lookup handed over exactly the same three rows both times, so any stage that treats them as a bag of rows cannot possibly tell the dog from the man. The order was in the stacking, and nothing in the numbers protected it — that is the position limit, in one `True`.
%%%

%%% insight
This is why "just stack them in order" is not enough, and it is a beautiful bit of design: rather than forcing every later stage to remember a sequence, the fix simply *writes the seat number into the numbers themselves*. Weird, cheap, and it works. You don't need it today — you just needed to see the hole it fills.
%%%

#### Your reward: two doorways found, both proven
Two blind spots, and you didn't take either on faith — you watched a `1.00` and a `True` prove them. That is exactly how a researcher spots the next thing to build.

%%% cards
🔢 | A few days out | The missing-position gap gets its own patch, called **positional encoding** — it literally adds "which slot am I" into the numbers. The future fix for the puzzle you just drew.
🌶️ | One more honest limit | A lookup table can't build phrase meaning either — "hot dog" is not "hot" plus "dog", and no single row fixes that. Composition is a job for the layers stacked on top.
%%%

!!! c-ok 😌
<b>为什么这不是设计失误 · Why this isn't a mistake:</b> the plain embedding is doing its one job perfectly — give every word a good <i>starting</i> point on the map. Deciding "which meaning?" and "what order?" is a different job with its own beautiful machinery. That machinery is the rest of this module, and you've just found the reason it has to exist.
!!!

@@@ concept id=c10 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built the whole first floor today. Here's one fresh picture to hold it all at once.

Think of a **package moving through a parcel-sorting facility.** A package arrives with words on the label ("cat"). At the door it gets a barcode — a number the machines can actually read.

A scanner reads that barcode and sends the package to its own shelf slot. And the shelves aren't random: similar parcels end up in the same aisle, so the whole warehouse is laid out by what's inside. One barcode, one slot, one aisle — fast and tidy. That's your day: **token id → lookup → map of meaning.**

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A package moving through a sorting facility in three steps: the word cat on a label gets a barcode ticket, a scanner sends it to a shelf slot, and similar parcels end up in the same aisle on a map. A note says the shelf never changes, but a word sometimes should."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A word through the sorting facility</text><rect x="20" y="44" width="120" height="56" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="80" y="66" text-anchor="middle" fill="#8A6D3B">label: "cat"</text><rect x="44" y="74" width="72" height="16" rx="2" fill="#2C2A28"/><text x="80" y="86" text-anchor="middle" fill="#FDF9F3" font-size="8">barcode = id 0</text><text x="150" y="76" fill="#B8AEA2" font-size="16">→</text><rect x="176" y="44" width="120" height="56" rx="8" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="236" y="64" text-anchor="middle" fill="#5E5191">scanner reads it</text><text x="236" y="82" text-anchor="middle" fill="#5E5191">→ shelf slot 0</text><text x="236" y="96" text-anchor="middle" fill="#9A938A" font-size="9">(the lookup)</text><text x="306" y="76" fill="#B8AEA2" font-size="16">→</text><rect x="332" y="44" width="164" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="414" y="62" text-anchor="middle" fill="#276b45">similar parcels,</text><text x="414" y="76" text-anchor="middle" fill="#276b45">same aisle</text><text x="414" y="92" text-anchor="middle" fill="#9A938A" font-size="9">(the map of meaning)</text><text x="260" y="128" text-anchor="middle" fill="#6B645E" font-size="9">label → barcode (token id) → shelf slot (lookup) → aisle of like things (map)</text><text x="260" y="146" text-anchor="middle" fill="#C93B3B" font-size="9">breaks down: the shelf never changes — but "bank" sometimes should. That's next lesson's job.</text></g></svg>
%%%

**Where the parcel picture breaks down:** a parcel's shelf never changes once it's stored, but a word sometimes should move depending on its sentence ("bank" by a river vs "bank" with money). That moving-to-fit job belongs to the *next* lessons.

#### The journey of one word, assembling in front of you
Watch the chain grow one link per line — by the last line you are looking at everything today built.

%%% svg
<svg viewBox="0 0 520 268" role="img" aria-label="A cumulative diagram with five lines. Line one shows only the word cat. Line two adds an arrow to its token id 0. Line three adds an arrow to row 0 of the table, the four numbers. Line four adds an arrow to a spot on the map of meaning. Line five adds an arrow to the cosine score against dog, 0.99. Each line repeats the previous links and adds exactly one more."><g font-family="monospace" font-size="9"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The same word, assembled one link at a time</text><g><text x="6" y="52" fill="#9A938A" font-size="8">stop 1</text><rect x="38" y="34" width="54" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="65" y="52" text-anchor="middle" fill="#8A6D3B">"cat"</text></g><g><text x="6" y="96" fill="#9A938A" font-size="8">stop 2</text><rect x="38" y="78" width="54" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="65" y="96" text-anchor="middle" fill="#8A6D3B">"cat"</text><text x="96" y="97" fill="#B8AEA2" font-size="11">→</text><rect x="106" y="78" width="48" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="130" y="96" text-anchor="middle" fill="#8A6D3B">id 0</text></g><g><text x="6" y="140" fill="#9A938A" font-size="8">stop 3</text><rect x="38" y="122" width="54" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="65" y="140" text-anchor="middle" fill="#8A6D3B">"cat"</text><text x="96" y="141" fill="#B8AEA2" font-size="11">→</text><rect x="106" y="122" width="48" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="130" y="140" text-anchor="middle" fill="#8A6D3B">id 0</text><text x="158" y="141" fill="#B8AEA2" font-size="11">→</text><rect x="168" y="122" width="116" height="28" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="226" y="140" text-anchor="middle" fill="#5E5191">[0.9, 0.8, 0.0, 0.3]</text></g><g><text x="6" y="184" fill="#9A938A" font-size="8">stop 4</text><rect x="38" y="166" width="54" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="65" y="184" text-anchor="middle" fill="#8A6D3B">"cat"</text><text x="96" y="185" fill="#B8AEA2" font-size="11">→</text><rect x="106" y="166" width="48" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="130" y="184" text-anchor="middle" fill="#8A6D3B">id 0</text><text x="158" y="185" fill="#B8AEA2" font-size="11">→</text><rect x="168" y="166" width="116" height="28" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="226" y="184" text-anchor="middle" fill="#5E5191">[0.9, 0.8, 0.0, 0.3]</text><text x="288" y="185" fill="#B8AEA2" font-size="11">→</text><rect x="298" y="166" width="88" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="342" y="179" text-anchor="middle" fill="#276b45" font-size="8">a spot on the</text><text x="342" y="189" text-anchor="middle" fill="#276b45" font-size="8">map of meaning</text></g><g><text x="6" y="228" fill="#9A938A" font-size="8">stop 5</text><rect x="38" y="210" width="54" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="65" y="228" text-anchor="middle" fill="#8A6D3B">"cat"</text><text x="96" y="229" fill="#B8AEA2" font-size="11">→</text><rect x="106" y="210" width="48" height="28" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="130" y="228" text-anchor="middle" fill="#8A6D3B">id 0</text><text x="158" y="229" fill="#B8AEA2" font-size="11">→</text><rect x="168" y="210" width="116" height="28" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="226" y="228" text-anchor="middle" fill="#5E5191">[0.9, 0.8, 0.0, 0.3]</text><text x="288" y="229" fill="#B8AEA2" font-size="11">→</text><rect x="298" y="210" width="88" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="342" y="223" text-anchor="middle" fill="#276b45" font-size="8">a spot on the</text><text x="342" y="233" text-anchor="middle" fill="#276b45" font-size="8">map of meaning</text><text x="390" y="229" fill="#B8AEA2" font-size="11">→</text><rect x="400" y="210" width="112" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="456" y="223" text-anchor="middle" fill="#276b45" font-size="8">cos(cat, dog)</text><text x="456" y="233" text-anchor="middle" fill="#276b45" font-size="9" font-weight="bold">= 0.99</text></g><text x="260" y="258" text-anchor="middle" fill="#6B645E" font-size="10">one link per line — and the last line is the whole of today</text></g></svg>
%%%

%%% steps
step: Stop 1 · a word becomes numbers
why: each word gets a short, fixed-length list — its embedding, its report card
step: Stop 2 · the word gets a ticket number
why: a token id is just the row number, and that list of ids is all the model ever receives
step: Stop 3 · the numbers live in one big table
why: one row per word in the vocabulary, so fetching a word's meaning is pure row-grabbing, no math
step: Stop 4 · numbers become a place
why: read the list as coordinates and the word lands on a map where close = similar; nobody typed those numbers — training nudged words that keep the same company closer, which is why this beat the old one-hot way (huge, and blind to similarity)
step: Stop 5 · one score reads the closeness
why: cosine similarity ≈1 alike, ≈0 unrelated, ≈−1 pointing back — and it is deliberately blind to arrow size
step: The three things it still can't do
why: an unseen or rare word needs subword tokenization (split into known pieces, better than a blunt [UNK]); one frozen row per word makes it context-free (river-"bank" vs money-"bank" scored 1.00); and nothing inside a row marks its slot, so the position has to be written in later
%%%

#### Cheat-sheet · every word you met today
%%% jargon
token | one word, or a word-piece — the basic chunk a model reads
token id | the ticket-number (row number) a token is given
embedding | a token's report card: a fixed-length list of numbers that stands for its meaning (the slots have no human-readable labels)
vector | just a list of numbers; an embedding is a vector
embedding matrix | the big lookup table — one row per word in the vocabulary (words × list-length is a huge block of numbers)
vocabulary | the full set of words (tokens) a model knows
d_model | how long each word's list is — more room to separate meanings, more numbers to store
map of meaning | reading each list as coordinates, so close = similar (real maps have hundreds of directions, not 2)
cosine similarity | a "same-direction" score for two arrows: ≈1 alike, ≈0 unrelated, ≈−1 pointing back (careful: it reads direction not size, and real antonyms like hot/cold usually score HIGH because they appear in the same sentences)
dot product | the multiply-slot-by-slot-and-add step inside that score
Euclidean distance | plain straight-line distance — unlike cosine it also reacts to arrow size
dense | a short list where every slot carries real info (opposite of one-hot's mostly-zeros)
one-hot | the old way: all zeros with a single 1 — huge, and every pair of words scores exactly the same
word2vec / GloVe | the first famous learned word vectors — one fixed vector per word
out-of-vocabulary (OOV) | a word with no row because it was never seen in training
subword tokenization | the fix for OOV and rare words: chop into smaller known pieces (BPE, WordPiece, SentencePiece)
UNK token | the crude fallback: one shared "unknown" row for every unseen word
polysemy | one spelling, two meanings (river "bank" vs money "bank")
context-free | the same word gets the same row no matter the surrounding words — the 1.00 you measured
positional encoding | the coming fix that writes "which slot am I" into the numbers
weight tying | reusing this one table at the model's output too, so it isn't paid for twice
%%%

That's the whole day. Next you'll teach these frozen word-points to *look at each other* — the start of **attention**, with Query, Key, and Value.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to finish today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is an embedding, in one line? | a:2 | A rule for spelling a word | A picture of the letters | A fixed-length list of numbers that stands for a word's meaning | A dictionary definition | fb: An embedding is a word's report card — a list of numbers, arranged so similar words get similar lists.
q: On the "map of meaning", what does it mean when two words sit CLOSE together? | a:1 | They are spelled alike | They are used in similar ways, so we read them as similar in meaning | They are the same length | They appear on the same page | fb: Close = used alike. The map turns "alike in use" into "near in distance" — that's the whole idea.
q: Why did models drop one-hot encoding in favor of embeddings? | a:2 | One-hot is slower to type | One-hot uses the wrong font | One-hot makes every pair of words equally far apart (no similarity) and is huge and mostly zeros; embeddings are short and place similar words close | One-hot can't store numbers | fb: One-hot can't say cat is closer to dog than to car — every pair scores 0 — and it wastes a slot per word. Embeddings are dense and meaningful.
q: A static lookup returns the SAME row for "bank" in "river bank" and "money bank", scoring them 1.00 alike. What is this limit called? | a:1 | A spelling bug fixed by a bigger table | The context-free limit — one frozen row per word, no matter the sentence | A font problem fixed by one-hot | A speed problem fixed by a faster computer | fb: One frozen row per spelling means the table is context-free. Fixing it is the job of a coming lesson.
%%%

@@@ produce id=produce tag="Produce" title="Predict which pair scores higher, then run it" gotit="Done"
Here's the fun part: you get to watch today's big idea prove itself in a few lines of output.

**Before you write any code, predict four things** and jot them down:

- `cosine(cat, dog)` vs `cosine(cat, car)` — which is higher, and does the loser land near zero?
- `cosine(bank, bank)` — both fetched from row 4 of the same frozen table. What can a table with one row per word possibly say?
- `cosine(onehot cat, onehot dog)` — can one-hot see that the animals are alike?
- "dog bites man" vs "man bites dog" — will the *set* of rows differ?

Then build the tiny table and **watch** it happen. That "I was right / oh!" moment is what makes the day stick. Pick one path.

#### Option A · write it yourself
Create `experiment.py` in this day's folder — `sessions/m03-attention/day-01-embeddings/experiment.py`. Six small pieces:

- a `vocab` dict with `cat` 0, `dog` 1, `car` 2, `the` 3, `bank` 4, `bites` 5, `man` 6, and the sentence "the cat" turned into token ids
- a table `E` with one hand-set row per word: cat `[0.9, 0.8, 0.0, 0.3]`, dog `[0.8, 0.9, 0.0, 0.4]`, car `[0.0, 0.0, 0.9, 0.3]`, the `[0.1, 0.0, 0.1, 0.1]`, bank `[0.4, 0.4, 0.4, 0.4]`, bites `[0.2, 0.1, 0.0, 0.6]`, man `[0.3, 0.1, 0.0, 0.2]`
- a lookup: fetch `E[vocab["cat"]]` and print it — a lookup really is just grabbing a row
- a `cosine(a, b)` function, then print cat-vs-dog against cat-vs-car
- the two contrasts: `cosine(E[vocab["bank"]], E[vocab["bank"]])` for the river and money cases (the same row twice), and one-hot cat-vs-dog (each one-hot row is 7 slots wide, one per known word)
- the order check: turn "dog bites man" and "man bites dog" into id lists, print both, and print whether the two *sets* of ids are equal

Print every step, then run it with `python3 sessions/m03-attention/day-01-embeddings/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 4 Day 1 artifact.

Create sessions/m03-attention/day-01-embeddings/experiment.py that, with a comment on each step:
1. Defines vocab = {"cat":0,"dog":1,"car":2,"the":3,"bank":4,"bites":5,"man":6} and maps the sentence "the cat" to token ids.
2. Builds an embedding table E (shape 7 x 4) with hand-set rows: cat=[0.9,0.8,0.0,0.3], dog=[0.8,0.9,0.0,0.4], car=[0.0,0.0,0.9,0.3], the=[0.1,0.0,0.1,0.1], bank=[0.4,0.4,0.4,0.4], bites=[0.2,0.1,0.0,0.6], man=[0.3,0.1,0.0,0.2], so cat and dog are close and car is far.
3. Looks up E[vocab["cat"]] and prints it — show that a lookup is just grabbing a row by its id.
4. Defines cosine(a, b) and prints cosine(cat, dog) vs cosine(cat, car), confirming the animals score much higher (about 0.99 vs about 0.08).
5. Shows the context-free limit: fetch E[vocab["bank"]] once for a "river bank" case and once for a "money bank" case, and print cosine(bank_river, bank_money) — show it is 1.00, i.e. the static table can't tell the two meanings apart.
6. As a contrast, builds one-hot vectors of length len(vocab) for cat/dog and prints cosine(onehot cat, onehot dog) — show it is 0.0, i.e. one-hot can't tell similar words apart.
7. Shows the position limit: turn "dog bites man" and "man bites dog" into id lists, print both, and print whether set(ids_a) == set(ids_b) — show it is True, i.e. the same rows come back for opposite stories.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- Looking up "cat" (id 0) returns row 0 of `E` — a short list of numbers. That's the whole lookup.
- `cosine(cat, dog)` lands about `0.99`; `cosine(cat, car)` lands near zero, about `0.08`. The animals win.
- `cosine(bank_river, bank_money)` prints a perfect `1.00` — the context-free limit, proven. That single number is why the coming lessons exist.
- The one-hot contrast prints `0.0` — one-hot is blind to similarity while the embedding *sees* it.
- The two sentences print different id lists but the same *set* — `True` — the position limit, proven.
- The thing worth **noticing**: nothing about spelling was used anywhere. Meaning lived entirely in where the rows sat, and one little score read it off.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-01-embeddings/log.md`: (1) why it is useful that similar words end up with nearby vectors; (2) the similarity number that surprised you (the 0.08? the 1.00?); (3) one thing you're still curious about.
!!!

@@@ fin
