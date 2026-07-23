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
@lede Think about how you tidy a bookshelf. You put the cookbooks in one stack, the comic books in another, the school books somewhere else. Your hand just *knows* that similar books belong close together — you never write down a rule. Here's a little mystery: a computer cannot read the word `cat` the way you can. To a computer, letters are just shapes, not meaning. So before a model like ChatGPT can do anything clever with language, it has to pull off that same bookshelf trick — give every word a *spot*, and place words with similar meanings close together. Today you build that spot-giving trick from scratch. It's called an **embedding**, and it's the very first thing that happens inside every language model. Get it right and "cat" and "dog" become neighbors, while "cat" and "car" sit rooms apart. Sounds almost too simple to be the foundation of modern AI? Good — hold that feeling. By the end it'll feel obvious, and a little bit magic.
@goal Together we'll turn a word into a list of numbers, arrange those numbers so meaning becomes *distance* on a map, and measure "how alike are these two words?" with one friendly little score. You'll also meet the two puzzles this simple map can't crack on its own — a word it has never seen, and a word with two meanings — and see the clever fixes waiting ahead. Every new word gets said in plain English the moment it shows up. No symbol left unexplained.

%%% warmup
q: Deep down, what is the only kind of thing a computer really understands? | a:2 | pictures | letters and words | numbers | sounds | concept: computer-uses-numbers | fb: A computer is basically a calculator — it works with numbers, not letters. That's why a word must become numbers first.
q: In Module 2 you saw a model get better by slowly adjusting its inner numbers as it learns. What do we call those adjustable inner numbers? | a:0 | weights | letters | pictures | passwords | concept: weights-are-learned | fb: Weights are the tunable numbers a model nudges while it learns — the same idea that will shape today's word-numbers.
%%%

@@@ concept id=c1 tag="Word to numbers" title="A word becomes a little list of numbers" gotit="Got what an embedding is"
Let's start where every model starts. A computer is a calculator — it only understands numbers. So the very first job is to turn a word into numbers.

Think of a **school report card**. Your report card is a short list of scores: maybe math 9, reading 7, sports 4. Those numbers are not *you* — but together they paint a little picture of you. Someone with a similar card is probably a similar kind of student. A model does the exact same thing to a word: it gives each word a short list of numbers — a kind of *report card for the word*. That list is called an [[embedding||A word's report card: a fixed-length list of numbers that stands in for the word's meaning. Similar words get similar lists.]]. Because it is just a list of numbers, we also call it a [[vector||A plain list of numbers. An embedding is a vector.]] — "vector" is simply the math word for "a list of numbers."

%%% svg
<svg viewBox="0 0 520 172" role="img" aria-label="A report card for the word cat. The card has four rows labelled is-animal, is-furry, is-vehicle, is-fast, each with a small number score. Below, the same four scores are written as a plain list of numbers, the embedding vector."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A word gets a report card of scores</text><rect x="150" y="30" width="220" height="96" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="260" y="48" text-anchor="middle" fill="#8A6D3B" font-weight="bold">word: "cat"</text><line x1="164" y1="56" x2="356" y2="56" stroke="#E1C674"/><text x="172" y="74" fill="#6B645E">is-animal</text><text x="348" y="74" text-anchor="end" fill="#2C2A28" font-weight="bold">0.9</text><text x="172" y="92" fill="#6B645E">is-furry</text><text x="348" y="92" text-anchor="end" fill="#2C2A28" font-weight="bold">0.8</text><text x="172" y="110" fill="#6B645E">is-vehicle</text><text x="348" y="110" text-anchor="end" fill="#2C2A28" font-weight="bold">0.0</text><text x="172" y="124" fill="#6B645E">is-fast</text><text x="348" y="124" text-anchor="end" fill="#2C2A28" font-weight="bold">0.3</text><text x="260" y="150" text-anchor="middle" fill="#6B645E" font-size="10">↓ read the scores straight down as one list ↓</text><text x="260" y="166" text-anchor="middle" fill="#5E5191" font-weight="bold">[ 0.9, 0.8, 0.0, 0.3 ]  ← the embedding</text></g></svg>
%%%

**What the report-card picture gets right:** a few numbers, read together, stand in for something rich — a student, or a word's meaning. **Where it breaks down:** a real report card has labels you can read ("math", "sports"), but a model's numbers usually have *no human label at all* — the model invents its own hidden "scores," and we mostly can't name what each one means. It still works; we just can't always read the columns.

#### How long is the list?
A model uses the *same* length for every word — maybe 4 numbers in our toy examples, but a few hundred (or a few thousand) in a real model. Every word's list is exactly this long. Longer lists can hold more subtle differences between words, the way a longer report card can tell two students apart more finely.

Let's make it real. Below, we look up three words and see their little number-lists. **Predict first:** cat and dog are both furry animals, and car is a machine. Which two lists do you think will look the *most* alike? Then reveal it.

%%% demo id=lookup label="show three word-cards"
code: embed("cat"), embed("dog"), embed("car")
out: cat -> [0.9, 0.8, 0.0, 0.3]   dog -> [0.8, 0.9, 0.0, 0.4]   car -> [0.0, 0.0, 0.9, 0.3]
take: <b>Each word became a list of the same length (4 numbers).</b> Look closely: cat and dog have almost the same numbers (both high on the first two scores), while car is completely different (its big score sits on <i>is-vehicle</i>, where cat and dog are 0). The numbers already whisper "cat and dog are alike; car is the odd one out" — before we do any math at all.
%%%

That is the whole starting idea: a word is now a list of numbers, and similar words got similar lists. Next we turn those lists into places you can actually *see*.

@@@ concept id=c2 tag="Meaning as a map" title="Turning the lists into a map of meaning" gotit="Got the map"
Here's where it gets beautiful. We have lists of numbers — now let's *see* them.

Think of a **map of your town**. A map turns a place into two numbers: how far east, how far north. Your school might be at "3 blocks east, 5 blocks north." Once every place is a pair of numbers, you can *see* which places are close: the bakery near the school, the park far across town. A word's embedding does the same thing. Treat each word's list of numbers as a set of *coordinates*, and drop the word onto a map. Now you can literally *see* which words are neighbors. This is the big idea of the whole day, so let's give it a name: **meaning becomes distance.** Two words that mean similar things land close together on the map; two unrelated words land far apart.

%%% svg
<svg viewBox="0 0 520 240" role="img" aria-label="A 2-D map of meaning. Cat and dog sit close together in the upper-left, labelled the animals neighborhood. Car and truck sit close together in the lower-right, labelled the vehicles neighborhood. The two neighborhoods are far apart."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Meaning becomes distance on a map</text><rect x="40" y="30" width="440" height="190" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><line x1="40" y1="220" x2="480" y2="220" stroke="#E5DFD6" stroke-width="1.5"/><line x1="40" y1="30" x2="40" y2="220" stroke="#E5DFD6" stroke-width="1.5"/><ellipse cx="130" cy="80" rx="72" ry="42" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="4,3"/><text x="130" y="46" text-anchor="middle" fill="#276b45" font-size="10">the "animals" neighborhood</text><circle cx="108" cy="82" r="6" fill="#2D8B55"/><text x="108" y="102" text-anchor="middle" fill="#276b45">cat</text><circle cx="152" cy="76" r="6" fill="#2D8B55"/><text x="152" y="70" text-anchor="middle" fill="#276b45">dog</text><ellipse cx="380" cy="168" rx="78" ry="42" fill="#EFEAF7" stroke="#7C6DAA" stroke-dasharray="4,3"/><text x="380" y="204" text-anchor="middle" fill="#5E5191" font-size="10">the "vehicles" neighborhood</text><circle cx="356" cy="160" r="6" fill="#7C6DAA"/><text x="356" y="152" text-anchor="middle" fill="#5E5191">car</text><circle cx="404" cy="172" r="6" fill="#7C6DAA"/><text x="404" y="190" text-anchor="middle" fill="#5E5191">truck</text><line x1="108" y1="82" x2="152" y2="76" stroke="#2D8B55" stroke-width="1" stroke-dasharray="2,2"/><text x="130" y="126" text-anchor="middle" fill="#2D8B55" font-size="9">close = similar</text><line x1="152" y1="76" x2="356" y2="160" stroke="#C93B3B" stroke-width="1" stroke-dasharray="2,2"/><text x="256" y="112" text-anchor="middle" fill="#C93B3B" font-size="9">far = unrelated</text></g></svg>
%%%

**What the town-map picture gets right:** turning things into coordinates lets you *see* closeness at a glance — near means alike, far means different. **Where it breaks down:** a town map has just two directions (east, north), but a word's map has *hundreds* of directions at once. We draw a flat 2-D picture so you can see the idea, but the model really lives in a space with far too many directions to sketch. The rule "close means similar" still holds; you just can't draw all the axes.

#### The one thing to hold onto
Everything else in a transformer — the whole "words looking at each other" machinery you'll build over the next days — only ever reads *where each word sits on this map*. It never looks at the letters. So the map is the seating chart for the entire show. Get the seating right and every later step has something meaningful to work with.

!!! c-info 🗺️
<b>The chain to remember:</b> word → a list of numbers (embedding) → a point on the map of meaning. That is the whole journey of a single word into a model, and it is step 1 of every language model on Earth.
!!!

@@@ concept id=c3 tag="Measuring closeness" title="How close are two words? Point and compare" gotit="Got closeness"
We can *see* closeness on the map. But a computer can't squint at a picture — it needs a number that says "these two words are 80% alike." So how do we measure closeness on the map?

Think of two friends standing in a field, each **pointing at something**. If they point in nearly the *same direction*, they probably care about the same thing. If one points north and the other points south — opposite directions — they're interested in totally different things. A word's embedding is like an arrow pointing out from the center of the map. To measure how alike two words are, we check: **do their arrows point the same way?** The name for this "same-direction score" is [[cosine similarity||A score for how much two arrows point the same way: about 1 = same direction (very alike), 0 = at right angles (unrelated), −1 = opposite directions (opposite meaning).]]. Don't let the fancy name scare you — it's just "how much do these two arrows agree on a direction?"

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three pairs of arrows from a center point. Left pair points almost the same way, labelled score near 1 very alike. Middle pair points at right angles, labelled score near 0 unrelated. Right pair points in opposite directions, labelled score near minus 1 opposite."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Do the two arrows point the same way?</text><g><circle cx="90" cy="110" r="4" fill="#6B645E"/><line x1="90" y1="110" x2="140" y2="60" stroke="#2D8B55" stroke-width="2.5"/><line x1="90" y1="110" x2="150" y2="72" stroke="#2D8B55" stroke-width="2.5"/><text x="100" y="150" text-anchor="middle" fill="#276b45" font-size="10">same way</text><text x="100" y="166" text-anchor="middle" fill="#276b45" font-weight="bold">score ≈ 1</text><text x="100" y="180" text-anchor="middle" fill="#9A938A" font-size="9">very alike</text></g><g><circle cx="260" cy="110" r="4" fill="#6B645E"/><line x1="260" y1="110" x2="260" y2="56" stroke="#C99A12" stroke-width="2.5"/><line x1="260" y1="110" x2="316" y2="110" stroke="#C99A12" stroke-width="2.5"/><text x="270" y="150" text-anchor="middle" fill="#8A6D3B" font-size="10">right angle</text><text x="270" y="166" text-anchor="middle" fill="#8A6D3B" font-weight="bold">score ≈ 0</text><text x="270" y="180" text-anchor="middle" fill="#9A938A" font-size="9">unrelated</text></g><g><circle cx="440" cy="110" r="4" fill="#6B645E"/><line x1="440" y1="110" x2="400" y2="66" stroke="#C93B3B" stroke-width="2.5"/><line x1="440" y1="110" x2="480" y2="154" stroke="#C93B3B" stroke-width="2.5"/><text x="440" y="150" text-anchor="middle" fill="#8a3b3b" font-size="10">opposite</text><text x="440" y="166" text-anchor="middle" fill="#C93B3B" font-weight="bold">score ≈ −1</text><text x="440" y="180" text-anchor="middle" fill="#9A938A" font-size="9">opposite meaning</text></g></g></svg>
%%%

**What the pointing-arrows picture gets right:** agreement on a *direction* is exactly what we want — two words are alike when their arrows point the same way, whatever their length. **Where it breaks down:** two friends only point in the flat world you can see, but word-arrows point in a space with hundreds of directions. You can't picture it, but the "same direction = alike" rule works exactly the same there.

%%% hint
t1: Don't try to picture hundreds of directions at once. Just take TWO short lists and ask one question: do they rise and fall together?
t2: Take cat [0.9, 0.8] and dog [0.8, 0.9]. Multiply position-by-position: 0.9×0.8 = 0.72, then 0.8×0.9 = 0.72. Add them: 0.72 + 0.72 = 1.44. A big positive total means "same direction." Now try cat vs a made-up opposite like [-0.9, -0.8] and watch the total go negative.
t3: Cosine similarity is just "multiply the two lists position-by-position, add it up, then divide by their lengths so only the direction counts" — high means alike, near zero means unrelated, negative means opposite.
%%%

#### The one-line rule (say it in plain words)
The actual score is one small calculation. In plain words: *multiply the two lists position-by-position, add it all up, then adjust for how long each arrow is.* That's it. A big positive score means "same direction, very alike." A score near zero means "unrelated." A negative score means "pointing opposite ways."

!!! c-info 🪜
<b>Optional (skippable) — the exact recipe, for the curious.</b> You do not need this to understand the day. The plain "multiply position-by-position and add" step has a name — the <b>dot product</b>, written <code>a·b = Σᵢ aᵢ·bᵢ</code> (sum, over every position <code>i</code>, of the two lists' <code>i</code>-th numbers multiplied). Cosine similarity is that dot product divided by the two arrows' lengths, so only the *direction* counts, not the size: <code>cos(a, b) = (a·b) / (‖a‖ · ‖b‖)</code>, where <code>‖a‖</code> means "the length of arrow a." That divide-by-length is why the score always lands neatly between −1 and +1.
!!!

Now let's *watch* the score do its job. **Predict first:** we'll score cat-vs-dog and cat-vs-car. Which pair comes out higher — and could one of them land near *zero*? Guess, then reveal.

%%% demo id=cos label="score the pairs"
code: cos(embed("cat"), embed("dog")),  cos(embed("cat"), embed("car"))
out: cat · dog = 0.99      cat · car = 0.08
take: <b>The animals score high (0.99 — arrows almost the same way); cat-vs-car scores near 0 (0.08 — arrows nearly at a right angle).</b> The number matches what your eyes saw on the map: cat and dog are neighbors, car is a stranger. And notice — nothing about spelling was used. Meaning lived entirely in *where the arrows pointed*, and one little score read it off.
%%%

You just turned "meaning" into a single number you can compare. That is the quiet engine under search, recommendations, and the attention you'll build next.

@@@ concept id=c4 tag="The lookup table" title="Where the numbers live — the big lookup table" gotit="Got the lookup table"
So every word has a list of numbers. Fair question: where are all those lists *kept*? And how does the model grab the right one fast?

Think of a **coat-check counter** at a theater. You hand over your coat and get a little numbered ticket — say ticket 27. Later you give back ticket 27, and the attendant walks straight to hook 27 and hands you your exact coat. No searching, no guessing — just "give me the number, get the thing." A model keeps every word's embedding in one giant list of rows, like a wall of numbered hooks. This wall is called the [[embedding matrix||One big table of numbers: one row per word in the vocabulary. Row number i holds the embedding for the word whose id is i.]] — a fancy phrase that just means "a big table where each row is one word's list of numbers." Every word first gets a ticket number, called its [[token id||The integer ticket-number a word is given. The model uses it to grab that word's row from the embedding table.]] (the model calls the word a [[token||One word, or a word-piece — the basic chunk a model reads text in.]]). To get a word's embedding, the model just goes to that row number. That's the whole "lookup."

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A coat-check style lookup table. On the left the word cat has ticket number 0. An arrow points to a table of rows numbered 0 to 3. Row 0 is highlighted and holds the numbers for cat. The output arrow returns that row."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Give the ticket number → get that word's row</text><rect x="24" y="70" width="96" height="44" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="72" y="90" text-anchor="middle" fill="#8A6D3B">"cat"</text><text x="72" y="106" text-anchor="middle" fill="#6B645E" font-size="10">ticket 0</text><text x="140" y="96" fill="#B8AEA2" font-size="16">→</text><rect x="170" y="36" width="240" height="150" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="290" y="54" text-anchor="middle" fill="#6B645E" font-size="10">the embedding table (one row per word)</text><rect x="182" y="64" width="216" height="26" rx="3" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="196" y="81" fill="#276b45">row 0  cat</text><text x="392" y="81" text-anchor="end" fill="#2C2A28">[0.9, 0.8, 0.0, 0.3]</text><rect x="182" y="94" width="216" height="26" rx="3" fill="#FDF9F3" stroke="#E5DFD6"/><text x="196" y="111" fill="#6B645E">row 1  dog</text><text x="392" y="111" text-anchor="end" fill="#9A938A">[0.8, 0.9, 0.0, 0.4]</text><rect x="182" y="124" width="216" height="26" rx="3" fill="#FDF9F3" stroke="#E5DFD6"/><text x="196" y="141" fill="#6B645E">row 2  car</text><text x="392" y="141" text-anchor="end" fill="#9A938A">[0.0, 0.0, 0.9, 0.3]</text><rect x="182" y="154" width="216" height="26" rx="3" fill="#FDF9F3" stroke="#E5DFD6"/><text x="196" y="171" fill="#6B645E">row 3  the</text><text x="392" y="171" text-anchor="end" fill="#9A938A">[0.1, 0.0, 0.1, 0.1]</text><text x="430" y="96" fill="#B8AEA2" font-size="16">→</text><text x="470" y="92" text-anchor="middle" fill="#276b45" font-size="10">that</text><text x="470" y="104" text-anchor="middle" fill="#276b45" font-size="10">row</text></g></svg>
%%%

**What the coat-check picture gets right:** you don't search — you hand over a number and instantly get the matching thing back. A lookup by row number is that fast and that simple. **Where it breaks down:** a coat-check gives back the coat *you* brought in, but the model's rows weren't put there by you — they were *learned* (that's the next concept), and one word always maps to exactly one row, which turns out to be a limitation we'll meet later.

#### How big is the wall of hooks?
The table has one row for every word the model knows. That full set of known words is the model's [[vocabulary||The complete set of words (tokens) a model knows — one table row for each.]]. Real models know a *lot* of words: their tables have tens of thousands of rows. So the table's shape is "number-of-words rows × length-of-each-list columns" — and that makes it one of the biggest single blocks of numbers in the whole model.

**Watch a sentence turn into row numbers.** **Predict first:** the sentence is "the cat". Each word will become a ticket number, then "cat" will fetch its row. Guess what row `cat` returns, then reveal.

%%% demo id=idlookup label="turn a sentence into row numbers"
code: ids = [vocab[w] for w in "the cat".split()];  ids, table[0]
out: token ids: [3, 0]      row for 'cat' (id 0): [0.9, 0.8, 0.0, 0.3]
take: <b>A sentence became a list of ticket numbers, and each number just grabs its row.</b> "the" is id 3, "cat" is id 0. To embed "cat", the model goes to row 0 — no math, just "fetch that row." This tiny step (text → ids → rows) runs at the very start of every model, on every sentence.
%%%

@@@ concept id=c5 tag="They are learned" title="Nobody typed those numbers — they were learned" gotit="Got that they're learned"
Now the part that surprises almost everyone. Who *chose* the numbers 0.9, 0.8, 0.0? Did an engineer sit down and type a report card for every word in the language? No. That would be impossible — and it would miss the magic. The numbers are **learned**.

Think of how *you* learned what the word "dog" means. Nobody handed you a definition sheet as a baby. You just *heard the word over and over*, near words like "bark", "walk", "puppy", "leash". Slowly your brain filed "dog" next to those. A model does the same thing. It reads mountains of text, and it notices which words keep showing up in the *same company*. Words that appear in similar surroundings get nudged closer together on the map, a tiny bit at a time. Do that across billions of sentences and, all on its own, the map organizes itself: animals drift into one neighborhood, foods into another, colors into another. Nobody placed them — the reading did.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Before and after. On the left, at the start of training, four word dots are scattered randomly with no pattern. An arrow labelled after reading lots of text points right. On the right, the same dots have organized themselves: cat and dog cluster together, car and bus cluster together, apart from each other."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Training organizes the map — nobody places the dots by hand</text><rect x="30" y="36" width="180" height="150" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="120" y="54" text-anchor="middle" fill="#9A938A" font-size="10">at the start: random</text><circle cx="70" cy="90" r="5" fill="#B8AEA2"/><text x="70" y="82" text-anchor="middle" fill="#9A938A" font-size="9">cat</text><circle cx="170" cy="150" r="5" fill="#B8AEA2"/><text x="170" y="166" text-anchor="middle" fill="#9A938A" font-size="9">dog</text><circle cx="90" cy="160" r="5" fill="#B8AEA2"/><text x="90" y="176" text-anchor="middle" fill="#9A938A" font-size="9">car</text><circle cx="160" cy="80" r="5" fill="#B8AEA2"/><text x="160" y="72" text-anchor="middle" fill="#9A938A" font-size="9">bus</text><g><text x="240" y="106" text-anchor="middle" fill="#8A6D3B" font-size="16">→</text><text x="240" y="126" text-anchor="middle" fill="#8A6D3B" font-size="9">reads</text><text x="240" y="138" text-anchor="middle" fill="#8A6D3B" font-size="9">lots of</text><text x="240" y="150" text-anchor="middle" fill="#8A6D3B" font-size="9">text</text></g><rect x="270" y="36" width="220" height="150" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="380" y="54" text-anchor="middle" fill="#276b45" font-size="10">after training: organized</text><ellipse cx="330" cy="90" rx="46" ry="30" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="3,2"/><circle cx="316" cy="88" r="5" fill="#2D8B55"/><text x="316" y="80" text-anchor="middle" fill="#276b45" font-size="9">cat</text><circle cx="346" cy="96" r="5" fill="#2D8B55"/><text x="346" y="112" text-anchor="middle" fill="#276b45" font-size="9">dog</text><ellipse cx="430" cy="150" rx="46" ry="28" fill="#EFEAF7" stroke="#7C6DAA" stroke-dasharray="3,2"/><circle cx="416" cy="148" r="5" fill="#7C6DAA"/><text x="416" y="140" text-anchor="middle" fill="#5E5191" font-size="9">car</text><circle cx="446" cy="156" r="5" fill="#7C6DAA"/><text x="446" y="172" text-anchor="middle" fill="#5E5191" font-size="9">bus</text></g></svg>
%%%

**What the "learning a word as a baby" picture gets right:** meaning comes from the *company a word keeps* — the words around it — not from a definition someone wrote down. That is exactly how a model builds its map. **Where it breaks down:** a baby also touches, tastes, and sees the world, but a model only ever sees text. So it learns which words *go together*, without ever meeting a real dog. Its map is built entirely from word-company.

#### Why this matters
You met the training loop in Module 2 — the same idea that tuned a neuron's weights also tunes these embedding numbers. The map isn't set once and frozen; it is *nudged into shape* by the very same learning that trains the rest of the model. So "meaning becomes geometry" isn't a lucky accident — it's what training is *for*, at the input.

!!! c-ok 🎤
<b>Once it clicks, here's how you'd say it:</b> "An embedding is a learned lookup table — token id <code>i</code> returns row <code>i</code> of a big table, and because training pulls together words that appear in similar contexts, two rows point the same way exactly when their words mean similar things. Nobody hand-writes the numbers; the reading writes them." You can already say every piece of that.
!!!

@@@ concept id=c6 tag="The old way" title="The clumsy first idea — one-hot" gotit="Got why one-hot lost"
Before we had this neat map, people tried a much cruder way to turn a word into numbers — and seeing *why it flopped* makes the map feel like the clever fix it is. This is a fun bit of history, so let's enjoy it.

Imagine a **classroom where, to name one kid, exactly one kid raises a hand and everyone else keeps theirs down.** To say "this word is *cat*," you make a huge row of switches — one switch for every word the model knows — and you flip *only cat's switch on*, all the rest off. That's it. This is called [[one-hot||The old way to turn a word into numbers: a giant list that is all zeros except a single 1 in the slot for that one word.]] encoding: a giant list of zeros with a single `1` in the slot for that word ("one-hot" = only one slot is hot).

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="One-hot vectors for cat, dog and car. Each is a long row of zeros with a single 1 in a different position. Below, a note that every pair of these is exactly the same distance apart, so none is more similar to any other."><g font-family="monospace" font-size="12"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One-hot: all zeros, a single 1 — one switch on</text><text x="30" y="52" fill="#6B645E">cat</text><g fill="#FDF9F3" stroke="#E5DFD6"><rect x="70" y="40" width="26" height="20"/><rect x="98" y="40" width="26" height="20"/><rect x="126" y="40" width="26" height="20"/><rect x="154" y="40" width="26" height="20"/></g><rect x="70" y="40" width="26" height="20" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="83" y="55" text-anchor="middle" fill="#276b45" font-weight="bold">1</text><text x="111" y="55" text-anchor="middle" fill="#9A938A">0</text><text x="139" y="55" text-anchor="middle" fill="#9A938A">0</text><text x="167" y="55" text-anchor="middle" fill="#9A938A">0</text><text x="210" y="55" fill="#9A938A" font-size="10">… (one slot per word)</text><text x="30" y="92" fill="#6B645E">dog</text><g fill="#FDF9F3" stroke="#E5DFD6"><rect x="70" y="80" width="26" height="20"/><rect x="98" y="80" width="26" height="20"/><rect x="126" y="80" width="26" height="20"/><rect x="154" y="80" width="26" height="20"/></g><rect x="98" y="80" width="26" height="20" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="83" y="95" text-anchor="middle" fill="#9A938A">0</text><text x="111" y="95" text-anchor="middle" fill="#276b45" font-weight="bold">1</text><text x="139" y="95" text-anchor="middle" fill="#9A938A">0</text><text x="167" y="95" text-anchor="middle" fill="#9A938A">0</text><text x="30" y="132" fill="#6B645E">car</text><g fill="#FDF9F3" stroke="#E5DFD6"><rect x="70" y="120" width="26" height="20"/><rect x="98" y="120" width="26" height="20"/><rect x="126" y="120" width="26" height="20"/><rect x="154" y="120" width="26" height="20"/></g><rect x="126" y="120" width="26" height="20" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="83" y="135" text-anchor="middle" fill="#9A938A">0</text><text x="111" y="135" text-anchor="middle" fill="#9A938A">0</text><text x="139" y="135" text-anchor="middle" fill="#276b45" font-weight="bold">1</text><text x="167" y="135" text-anchor="middle" fill="#9A938A">0</text><text x="260" y="168" text-anchor="middle" fill="#C93B3B" font-size="11">cat, dog, car are ALL the exact same distance apart</text><text x="260" y="184" text-anchor="middle" fill="#C93B3B" font-size="10">→ the numbers say NOTHING about which words are alike</text></g></svg>
%%%

**What the raised-hands picture gets right:** it's a dead-simple, unmistakable way to point at exactly one word — no confusion about which word you mean. **Where it breaks down:** in the classroom every kid is just as far from every other kid — there are no "friend groups." And that is exactly one-hot's fatal flaw for meaning.

#### The two problems that killed it
- **No sense of similarity.** In one-hot, cat, dog, and car are *all exactly the same distance apart*. The numbers carry zero information about which words are alike — "cat" is no closer to "dog" than to "car". You threw away the one thing you actually wanted.
- **Wildly wasteful.** If a model knows 50,000 words, every single word is a list of 50,000 numbers — 49,999 of them zero. Huge, and almost empty.

The **embedding** fixes both in one move: instead of 50,000 slots, use a short list of a few hundred numbers (that's why we call it a [[dense||"Dense" = a short list where every slot carries real information, the opposite of one-hot's long list of mostly zeros.]] vector — every slot does real work), and *let training arrange them* so similar words land close. Short instead of huge, and meaningful instead of blind. This is the whole reason embeddings won.

Now let's put them head-to-head. **Predict first:** we score cat-vs-dog and cat-vs-car *twice* — once with one-hot, once with embeddings. Which method will notice that cat and dog are alike? Guess, then reveal.

%%% demo id=onehot label="one-hot vs embedding — measure closeness"
code: cos(onehot("cat"), onehot("dog")),  cos(onehot("cat"), onehot("car")),  cos(embed("cat"), embed("dog"))
out: ONE-HOT cat·dog = 0.0    ONE-HOT cat·car = 0.0    EMBEDDING cat·dog = 0.99
take: <b>One-hot gives 0.0 for BOTH pairs — it literally can't tell cat-and-dog apart from cat-and-car.</b> The embedding gives cat-and-dog a high 0.99. Same words, but only the embedding knows the animals are alike. That gap is why every modern model uses embeddings, not one-hot.
%%%

@@@ concept id=c7 tag="Unknown words" title="A puzzle: a word the model has never seen" gotit="Got the OOV fix"
Here's a puzzle to chew on. The lookup table has one row per known word. So what happens when a word shows up that *isn't in the table*? A brand-new slang word, a rare name, a typo. There's no hook for it on the wall. Uh oh — but don't worry, there's a genuinely clever escape, and it's fun to work out.

Think of a **coat-check that only has hooks for coat styles it has seen before.** A guest arrives with a coat style the attendant has never seen — there's no hook number for it. What now? This is the [[out-of-vocabulary||"OOV" for short — a word that isn't in the model's known-word list, so it has no row to look up.]] problem (people say "OOV" — out-of-vocabulary). The *cause* is simple: the vocabulary is a fixed list made before training, and a word that never appeared has no row. Left alone, the model would stall on that word.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="The word tokenization is not in the table. Two fixes are shown. Fix one, the crude fallback, maps it all to a single shared unknown row. Fix two, the smart fix, breaks it into known pieces token and ization, each of which has its own row."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Unknown word "tokenization" → two ways to handle it</text><rect x="180" y="30" width="160" height="26" rx="5" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="260" y="47" text-anchor="middle" fill="#C93B3B">"tokenization" — no row!</text><line x1="200" y1="56" x2="120" y2="82" stroke="#B8AEA2"/><line x1="320" y1="56" x2="400" y2="82" stroke="#B8AEA2"/><text x="120" y="78" text-anchor="middle" fill="#8A6D3B" font-size="10">crude fallback</text><rect x="60" y="86" width="130" height="26" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="125" y="103" text-anchor="middle" fill="#8A6D3B">→ [UNK] (one shared row)</text><text x="125" y="128" text-anchor="middle" fill="#9A938A" font-size="9">every unknown word</text><text x="125" y="140" text-anchor="middle" fill="#9A938A" font-size="9">looks identical</text><text x="400" y="78" text-anchor="middle" fill="#276b45" font-size="10">smart fix: split into pieces</text><rect x="330" y="86" width="66" height="26" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="363" y="103" text-anchor="middle" fill="#276b45">"token"</text><rect x="402" y="86" width="72" height="26" rx="5" fill="#EAF5EE" stroke="#2D8B55"/><text x="438" y="103" text-anchor="middle" fill="#276b45">"ization"</text><text x="400" y="128" text-anchor="middle" fill="#276b45" font-size="9">each piece IS known,</text><text x="400" y="140" text-anchor="middle" fill="#276b45" font-size="9">so each gets a real row</text><text x="260" y="166" text-anchor="middle" fill="#6B645E" font-size="10">splitting into known pieces keeps the meaning; one shared [UNK] throws it away</text></g></svg>
%%%

**What the coat-check picture gets right:** a fixed set of hooks really can be caught out by something new — the same way a fixed word-list gets caught out by a new word. **Where it breaks down:** a coat can't be split into halves that each fit a smaller hook, but a *word* can — and that's exactly the clever fix.

#### The clever fix: break the word into known pieces
The trick almost every modern model uses is [[subword tokenization||The fix for unknown words: chop a rare word into smaller pieces that ARE in the vocabulary, so each piece gets its own row.]]. Instead of needing a row for every whole word, the model keeps rows for common *word-pieces*. A rare word like "tokenization" gets chopped into pieces it *does* know — "token" + "ization" — and each piece has its own row. So the model can handle a word it never saw as a whole, because it saw the *parts*. This is why you almost never truly stump a modern model with a new word.

#### The cruder backup plan
The old, blunt fallback is a single shared [[UNK token||A single "unknown" row that every unseen word gets mapped to — cheap, but it makes all unknown words look identical.]] — one "unknown" row that *every* unseen word maps to. It never crashes, but it's lossy: "tokenization", "flibbertigibbet", and a rare surname all collapse into the *same* vector, so the model can't tell them apart. Subword splitting is far kinder, which is why it won.

**Watch the split happen.** **Predict first:** "cat" is a known word; "tokenization" was never seen whole. How many pieces will each become? Guess, then reveal.

%%% demo id=oov label="known word vs unknown word"
code: tokenize("cat"),  tokenize("tokenization")
out: known:   ['cat']        unknown: ['token', 'ization']
take: <b>"cat" is one known token — one row.</b> "tokenization" was never seen whole, so it splits into "token" + "ization", two pieces the model DOES know. No crash, and the meaning of the pieces survives — that's why subword tokenization beats a blunt [UNK].
%%%

@@@ concept id=c8 tag="One word, two meanings" title="The one thing this map can't do — yet" gotit="Got the limit"
Time for the puzzle that sets up the *entire rest of this module*. Our map is lovely, but it has a blind spot, and spotting it is the doorway to what comes next. This isn't a bug to feel bad about — it's the exciting cliffhanger that everything ahead exists to solve.

Think of a **single name-tag sticker** you wear to a party. It says one thing — "Alex" — no matter which room you walk into. It can't say "Alex the cook" in the kitchen and "Alex the swimmer" at the pool. It's stuck on one message. Our embedding table has the same problem: **one word gets exactly one row, one fixed list of numbers, forever.** But some words mean *different* things in different sentences. Consider the word "bank":
- "I sat on the **bank** of the river." (the edge of a river)
- "I put my money in the **bank**." (a place for money)

Same spelling, totally different meaning — a mix-up called [[polysemy||When one spelling has two or more meanings, like river "bank" vs money "bank".]]. But the table only has *one* row for "bank". So it hands back the *same* vector both times, blending the two meanings into one muddy average. It literally cannot tell the river from the money.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="The word bank appears in two very different sentences, a river sentence and a money sentence, but the fixed lookup table returns the exact same single vector for both, so it cannot tell the two meanings apart."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One word, one fixed row — even when it has two meanings</text><rect x="24" y="40" width="200" height="30" rx="5" fill="#EAF0FA" stroke="#5E5191"/><text x="124" y="59" text-anchor="middle" fill="#5E5191" font-size="10">"…the bank of the river"</text><rect x="24" y="120" width="200" height="30" rx="5" fill="#FCF3DC" stroke="#C99A12"/><text x="124" y="139" text-anchor="middle" fill="#8A6D3B" font-size="10">"…money in the bank"</text><line x1="224" y1="55" x2="300" y2="90" stroke="#B8AEA2"/><line x1="224" y1="135" x2="300" y2="100" stroke="#B8AEA2"/><rect x="300" y="78" width="150" height="34" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="375" y="93" text-anchor="middle" fill="#C93B3B" font-size="10">row for "bank"</text><text x="375" y="106" text-anchor="middle" fill="#C93B3B">[0.4, 0.4, 0.4]</text><text x="375" y="132" text-anchor="middle" fill="#C93B3B" font-size="10">the SAME vector both times</text><text x="375" y="146" text-anchor="middle" fill="#9A938A" font-size="9">can't tell river from money</text></g></svg>
%%%

**What the name-tag picture gets right:** one fixed label really can't change to fit the room — exactly like one fixed row can't change to fit the sentence. **Where it breaks down:** a person can just *say* "actually I'm the cook here", but a static embedding has no way to look around at its sentence at all — it's frozen the moment it's looked up.

#### Limit 1 · context-free — see it in motion
Let's *prove* the blind spot instead of just claiming it. **Predict first:** we look up "bank" from the river sentence and "bank" from the money sentence, then score how alike the two are with cosine similarity. Two totally different meanings — what score would you *hope* for? Now guess what the frozen table will actually give. Reveal it.

%%% demo id=polysemy label="score river-bank vs money-bank"
code: cos(embed("bank" @ "river"), embed("bank" @ "money"))
out: bank(river) -> [0.4, 0.4, 0.4]    bank(money) -> [0.4, 0.4, 0.4]    cos = 1.00
take: <b>Both look-ups return the identical vector, so cosine similarity is a perfect 1.00 — the map says the two "bank"s are the *same word with the same meaning*.</b> That's the failure in one number: a static table has no idea the river edge and the money place are different, because it only ever fetches one frozen row per spelling. You'd *hope* for a low score; the table can't give one.
%%%

#### Limit 2 · no word order — draw it out
There's a second, sneakier gap. A plain pile of these vectors doesn't remember what *order* the words came in. **Predict first:** "dog bites man" and "man bites dog" use the *same three words* — will their bags of vectors look different? Watch the figure below assemble.

%%% svg
<svg viewBox="0 0 520 250" role="img" aria-label="Two sentences, dog bites man and man bites dog, are each turned into a bag of the same three word vectors. Because a plain lookup ignores order, the two bags of vectors are identical, so the model cannot tell the two very different sentences apart."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Same three words, opposite meaning — but the bag of vectors is identical</text><text x="20" y="46" fill="#5E5191" font-weight="bold">"dog bites man"</text><g><rect x="20" y="56" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="65" y="73" text-anchor="middle" fill="#276b45">dog [0.8,0.1]</text><rect x="118" y="56" width="96" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="166" y="73" text-anchor="middle" fill="#8A6D3B">bites [0.3,0.7]</text><rect x="222" y="56" width="96" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="270" y="73" text-anchor="middle" fill="#5E5191">man [0.2,0.9]</text></g><text x="340" y="74" fill="#B8AEA2" font-size="14">→ into a bag →</text><text x="20" y="118" fill="#C99A12" font-weight="bold">"man bites dog"</text><g><rect x="20" y="128" width="96" height="26" rx="4" fill="#EFEAF7" stroke="#7C6DAA"/><text x="68" y="145" text-anchor="middle" fill="#5E5191">man [0.2,0.9]</text><rect x="124" y="128" width="96" height="26" rx="4" fill="#FCF3DC" stroke="#C99A12"/><text x="172" y="145" text-anchor="middle" fill="#8A6D3B">bites [0.3,0.7]</text><rect x="228" y="128" width="90" height="26" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="273" y="145" text-anchor="middle" fill="#276b45">dog [0.8,0.1]</text></g><text x="340" y="146" fill="#B8AEA2" font-size="14">→ into a bag →</text><rect x="360" y="92" width="150" height="56" rx="8" fill="#FDF9F3" stroke="#E5DFD6" stroke-width="1.5"/><text x="435" y="112" text-anchor="middle" fill="#6B645E" font-size="10">the bag (order dropped)</text><text x="435" y="128" text-anchor="middle" fill="#2C2A28">{dog, bites, man}</text><text x="435" y="142" text-anchor="middle" fill="#9A938A" font-size="9">same set both times</text><text x="260" y="182" text-anchor="middle" fill="#C93B3B" font-size="11">Both sentences → the exact same bag of vectors</text><text x="260" y="200" text-anchor="middle" fill="#C93B3B" font-size="10">so the model can't tell "dog bites man" from "man bites dog"</text><text x="260" y="226" text-anchor="middle" fill="#6B645E" font-size="10">who-bit-whom lives in the ORDER — and the plain lookup never wrote the order down</text></g></svg>
%%%

**What this build-up shows:** the *same* three rows come out in both cases, so a plain bag of embeddings genuinely cannot tell a dog biting a man from a man biting a dog. The meaning that flipped lived entirely in the *order*, and the lookup never recorded it.

#### Two limits, said plainly
So the blind spot is really two facts about a plain lookup table:
- **It is context-free.** The same word gets the same vector no matter what surrounds it. "bank" by the river and "bank" with money are identical to it (that 1.00 you just saw).
- **It has no sense of order.** A bag of these vectors doesn't know that "dog bites man" and "man bites dog" are different — same words, same rows, and order was never recorded.

!!! c-warn 😕
<b>为什么这不是设计失误 · Why this isn't a mistake:</b> the plain embedding is doing its one job perfectly — give every word a good *starting* point on the map. Fixing "which meaning?" and "what order?" is a *different* job, with its own beautiful machinery. That machinery is exactly what the rest of this module is about — and you've just found the reason it has to exist.
!!!

#### Your reward for finishing today: you found the doorway
Here's the exciting part: you do **not** have to solve either limit today, and you don't need to learn *how* the fixes work yet — that's what the coming lessons are for. What you *have* done is spot exactly *why* they'll be needed. Carry today's `1.00` "bank" score and today's identical "dog bites man" / "man bites dog" bag with you: they are the two problems the next lessons open on. Today's map isn't a dead end — it's the trailhead for everything ahead.

!!! c-info 🔗
<b>Preview only — a peek at where the road goes next (names to recognize later, nothing to learn today):</b><br>• The fix for the frozen-"bank" limit arrives in the <b>next lesson (Day 2)</b>. Just know the fix has a name — <b>attention</b> — and you'll build it there from scratch. No need to understand how it works today.<br>• The result attention produces is called <b>contextual embeddings</b> — again, just a name to file away for next lesson, not today's material.<br>• The separate "no word order" gap gets its own patch a few days out, named <b>positional encoding</b>. You don't need to understand it today either — it's simply the future fix for the "dog bites man" puzzle you just drew.<br>These three are <i>coming attractions</i>, not part of today. Today you built the floor the whole building stands on.
!!!

@@@ concept id=c9 tag="Recap" title="Today in one page" gotit="Got the recap"
Take a breath — you built the whole first floor today. Here's a fresh way to hold it all at once.

Think of a **package moving through a parcel-sorting facility.** A package arrives with words on the label ("cat"). At the door it gets a barcode sticker — a number the machines can actually read (that's the **token id**). A scanner reads the barcode and sends the package to its own shelf slot (that's the **lookup** into the table). And every shelf isn't random: similar parcels end up in the same aisle, so the whole warehouse is laid out by *what's inside* (that's the **map of meaning**, learned over time as more parcels flow through). One barcode, one slot, one aisle — fast and tidy.

%%% svg
<svg viewBox="0 0 520 160" role="img" aria-label="A package moving through a sorting facility in three steps: the word cat on a label gets a barcode ticket, a scanner sends it to a shelf slot, and similar parcels end up in the same aisle on a map. A note says the shelf never changes, but a word sometimes should."><g font-family="monospace" font-size="10"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">A word through the sorting facility</text><rect x="20" y="44" width="120" height="56" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="80" y="66" text-anchor="middle" fill="#8A6D3B">label: "cat"</text><rect x="44" y="74" width="72" height="16" rx="2" fill="#2C2A28"/><text x="80" y="86" text-anchor="middle" fill="#FDF9F3" font-size="8">barcode = id 0</text><text x="150" y="76" fill="#B8AEA2" font-size="16">→</text><rect x="176" y="44" width="120" height="56" rx="8" fill="#EFEAF7" stroke="#7C6DAA" stroke-width="1.5"/><text x="236" y="64" text-anchor="middle" fill="#5E5191">scanner reads it</text><text x="236" y="82" text-anchor="middle" fill="#5E5191">→ shelf slot 0</text><text x="236" y="96" text-anchor="middle" fill="#9A938A" font-size="9">(the lookup)</text><text x="306" y="76" fill="#B8AEA2" font-size="16">→</text><rect x="332" y="44" width="164" height="56" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="414" y="62" text-anchor="middle" fill="#276b45">similar parcels,</text><text x="414" y="76" text-anchor="middle" fill="#276b45">same aisle</text><text x="414" y="92" text-anchor="middle" fill="#9A938A" font-size="9">(the map of meaning)</text><text x="260" y="128" text-anchor="middle" fill="#6B645E" font-size="9">label → barcode (token id) → shelf slot (lookup) → aisle of like things (map)</text><text x="260" y="146" text-anchor="middle" fill="#C93B3B" font-size="9">breaks down: the shelf never changes — but "bank" sometimes should. That's next lesson's job.</text></g></svg>
%%%

**What this picture gets right:** a messy human label becomes a clean number, that number fetches the right thing instantly, and similar things sit near each other. **Where it breaks down:** a parcel's shelf never changes once it's stored, but a real word sometimes needs to move depending on the sentence around it ("bank" by a river vs. "bank" with money) — and that moving-to-fit-the-sentence job is what the *next* lessons add, not today. Read the five stops below, then keep the cheat-sheets as your map.

The journey of one word, in order:
- **Stop 1 · A word becomes numbers.** Each word gets a short, fixed-length list of numbers — its **embedding** (its report card).
- **Stop 2 · Numbers become a place.** Treat that list as coordinates and drop the word on a **map of meaning**, where **close = similar**. To read closeness as a number, check whether two arrows point the same way — **cosine similarity** (≈1 alike, ≈0 unrelated, ≈−1 opposite).
- **Stop 3 · Where the numbers live.** All the lists sit in one big **lookup table** (the embedding matrix), one row per word; a word's **token id** is just the row number.
- **Stop 4 · Nobody typed them.** The numbers are **learned** from reading lots of text — words that keep the same company drift close together. This beat the old **one-hot** way, which was huge and couldn't tell any two words apart.
- **Stop 5 · What it still can't do.** A rare word with no row is handled by **subword tokenization** (split into known pieces) — better than a blunt **[UNK]**. And one word gets one frozen vector, so it's **context-free** (river-"bank" vs money-"bank" scored a perfect 1.00) and carries **no word order** ("dog bites man" = "man bites dog"). Fixing those two is the job of the *coming* lessons, not today.

#### Cheat-sheet · the ideas at a glance
%%% table
:: idea :: in one line :: watch out
Embedding :: a word's short list of numbers (its report card) :: the numbers have no human-readable labels
Map of meaning :: each list is a point; close = similar :: real maps have hundreds of directions, not 2
Cosine similarity :: do the two arrows point the same way? (≈1 alike, ≈0 unrelated) :: a big score means direction, not size
Lookup table :: one row per word; token id = row number :: often the biggest block of numbers in a small model
Learned :: training moves same-company words close together :: it only ever sees text, never the real world
One-hot (old way) :: all zeros with a single 1 :: huge, and can't tell any two words apart
Subword tokenization :: split a rare word into known pieces :: beats a single blunt [UNK] row
Context-free limit :: one word = one frozen vector :: can't split river-"bank" from money-"bank" — a coming lesson fixes it
No-order limit :: a bag of vectors forgets word order :: "dog bites man" = "man bites dog" — a coming lesson fixes it
%%%

#### Cheat-sheet · the words you met today
%%% jargon
token | one word, or a word-piece — the basic chunk a model reads
token id | the ticket-number (row number) a token is given
embedding | a token's report card: a fixed-length list of numbers that stands for its meaning
vector | just a list of numbers; an embedding is a vector
embedding matrix | the big lookup table — one row per word in the vocabulary
vocabulary | the full set of words (tokens) a model knows
cosine similarity | a "same-direction" score for two arrows: ≈1 alike, ≈0 unrelated, ≈−1 opposite
dense | a short list where every slot carries real info (opposite of one-hot's mostly-zeros)
one-hot | the old way: all zeros with a single 1 — big and blind to similarity
out-of-vocabulary (OOV) | a word with no row because it was never seen in training
subword tokenization | the fix for OOV: chop a rare word into smaller known pieces
UNK token | the crude fallback: one shared "unknown" row for every unseen word
polysemy | one spelling, two meanings (river "bank" vs money "bank")
context-free | the same word gets the same vector no matter the surrounding words
%%%

That's the whole day. Next you'll teach these frozen word-points to *look at each other* — the start of **attention**, with Query, Key, and Value.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What is an embedding, in one line? | a:2 | A rule for spelling a word | A picture of the letters | A fixed-length list of numbers that stands for a word's meaning | A dictionary definition | fb: An embedding is a word's report card — a list of numbers, arranged so similar words get similar lists.
q: On the "map of meaning", what does it mean when two words sit CLOSE together? | a:1 | They are spelled alike | They have similar meaning | They are the same length | They appear on the same page | fb: Close = similar meaning. The map turns "alike in meaning" into "near in distance" — that's the whole idea.
q: Why did models drop one-hot encoding in favor of embeddings? | a:2 | One-hot is slower to type | One-hot uses the wrong font | One-hot makes every pair of words equally far apart (no similarity) and is huge and mostly zeros; embeddings are short and place similar words close | One-hot can't store numbers | fb: One-hot can't say cat is closer to dog than to car — every pair is equally distant — and it wastes a slot per word. Embeddings are dense and meaningful.
q: A static lookup returns the SAME vector for "bank" in "river bank" and "money bank", scoring them 1.00 alike. What is this limit called? | a:1 | A spelling bug fixed by a bigger table | The context-free limit — one frozen vector per word, no matter the sentence | A font problem fixed by one-hot | A speed problem fixed by a faster computer | fb: One frozen row per spelling means the table is context-free — it can't tell the two "bank"s apart. Fixing it is the job of a coming lesson.
%%%

@@@ produce id=produce tag="Produce" title="Predict which pair scores higher, then run it" gotit="Done"
Here's the fun part — you get to see today's big idea prove itself in a few lines of output. **Before you write any code, predict:** you'll score "cat" against "dog" and "cat" against "car" with cosine similarity — which pair comes out higher, and will either score come out near *zero*? Then, for the twist, you'll look up "bank" twice and score it against *itself* — guess what a frozen table returns. Jot your guesses down, then build the tiny table and **watch** it happen. That little "I was right / oh!" moment is what makes the day stick. Pick one path.

#### Option A · write it yourself
Create `sessions/m03-attention/day-01-embeddings/experiment.py`. Make a tiny `vocab` dict (`cat`, `dog`, `car`, `the`), a small embedding table `E` with hand-set rows — use cat `[0.9, 0.8, 0.0, 0.3]`, dog `[0.8, 0.9, 0.0, 0.4]`, car `[0.0, 0.0, 0.9, 0.3]` so cat/dog are close and car is far — look up a word by its token id (that's just grabbing a row), and compute the cosine similarity of cat-and-dog versus cat-and-car. Then, to *observe* the context-free limit, look up "bank" for two different sentences from the same frozen table and print `cosine(bank, bank)` — watch it come out `1.00`. Print each step. Run with `python3 sessions/m03-attention/day-01-embeddings/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It has Claude Code create the file, write the code, and run it for you.

%%% prompt id=pp label="ask Claude to build it"
Help me build my Module 4 Day 1 artifact.

Create sessions/m03-attention/day-01-embeddings/experiment.py that, with a comment on each step:
1. Defines vocab = {"cat":0,"dog":1,"car":2,"the":3} and maps the sentence "the cat" to token ids.
2. Builds an embedding table E (shape 4 x 4) with hand-set rows: cat=[0.9,0.8,0.0,0.3], dog=[0.8,0.9,0.0,0.4], car=[0.0,0.0,0.9,0.3], the=[0.1,0.0,0.1,0.1], so cat and dog are close and car is far.
3. Looks up E[vocab["cat"]] and prints it — show that a lookup is just grabbing a row by its id.
4. Defines cosine(a, b) and prints cosine(cat, dog) vs cosine(cat, car), confirming the animals score much higher (about 0.99 vs about 0.08).
5. Shows the context-free limit: look up the SAME "bank" row for a "river bank" case and a "money bank" case, and print cosine(bank_river, bank_money) — show it is 1.00, i.e. the static table can't tell the two meanings apart.
6. As a contrast, builds one-hot vectors for cat/dog and prints cosine(onehot cat, onehot dog) — show it is 0.0, i.e. one-hot can't tell similar words apart.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (check your prediction)
- Looking up "cat" (id 0) just returns row 0 of `E` — a short list of numbers. That's the whole "lookup": grab a row by its ticket number.
- `cosine(cat, dog)` comes out clearly higher (about `0.99`) than `cosine(cat, car)` (near `0`, about `0.08`). The animals win. Did you predict how close to zero the unrelated pair would land?
- `cosine(bank_river, bank_money)` prints a perfect `1.00` — proof of the context-free limit: the frozen table hands back the identical row both times and can't tell the river from the money. That single number is exactly why the coming lessons exist.
- The one-hot contrast prints `0.0` — proof that one-hot is *blind* to similarity, while the embedding *sees* it.
- The thing worth **noticing**: nothing about spelling was used anywhere. Meaning lived entirely in *where the rows sat*, and one little score read it off.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m03-attention/day-01-embeddings/log.md`: (1) why it is useful that similar words end up with nearby vectors; (2) the similarity number that surprised you (the 0.08? the 1.00?); (3) one thing you're still curious about (why the numbers are *learned* and not typed in, or how "bank" ever gets two meanings, are both fair game).
!!!

@@@ fin
