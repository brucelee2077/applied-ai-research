---
quest_id: wf4-d05-position
mode: concept
donor: v9-base.donor
page_title: "Module 4 · Day 5 — Positional Encoding"
module_label: "Module 4 · Represent · Day 5"
title: "Positional Encoding"
subtitle: "&amp; the Shuffle Problem"
brand_sub: "Foundations · M4 Day 5"
spine: "seat"
nav_prev_href: "../day-04-multihead/lesson.html"
nav_prev_label: "Multi-Head Attention"
nav_next_href: "../review.html"
nav_next_label: "Review Gate"
fin_title: "Module 4 · Day 5 complete! 🏆"
fin_body: "Nice work — you've completed <b>Positional Encoding</b>. You fixed the shuffle problem: give every word a <b>seat</b> number so word order finally carries meaning.<br>Next up: the <b>Module 4 Review Gate</b>."
notebook_yardstick: null
coverage_topics:
  - {topic: order changes meaning, keywords: [word order, order matters, dog bites man, different order, same words]}
  - {topic: attention is order blind permutation invariant, keywords: [order-blind, unordered set, permutation, shuffle, bag of words, same answer]}
  - {topic: positional encoding add a position signal, keywords: [position signal, seat number, add a signal, position to each word, where it sits]}
  - {topic: sinusoidal positional encoding added not concatenated, keywords: [sine, cosine, wave pattern, added to the embedding, not concatenated, fixed formula]}
  - {topic: many frequencies fast and slow waves unique fingerprint, keywords: [many waves, different speeds, fast wave, slow wave, frequencies, unique fingerprint]}
  - {topic: encoding is fixed not learned, keywords: [fixed, not learned, no training, formula, computed]}
  - {topic: learned positional embeddings alternative, keywords: [learned position, learned embedding, learn a vector, alternative]}
  - {topic: single wave collides neighbors need multiple frequencies, keywords: [collide, neighbors, one slow wave, too slowly, adjacent, combine multiple]}
  - {topic: encoding only supplies signal network must learn to use it, keywords: [only tells, still has to learn, does not teach, supplies the signal]}
  - {topic: fixed absolute encodings extrapolate poorly motivates advanced methods, keywords: [longer sentences, extrapolate, does not automatically, trained on short, advanced position methods]}
---

@@@ hero
@lede Picture the seats in a movie theater, each with a printed number on it. When you buy a ticket it says "row F, seat 12" — so even in the dark, everyone finds exactly where they belong. Now imagine someone scrubbed the numbers off every chair: people could sit anywhere, and "the seat by the aisle" would stop meaning anything. Word order works a lot like those seat numbers. "Dog bites man" and "man bites dog" use the exact same three words — only the *order* is different, yet they mean opposite things. Here is the surprise that makes today fun: the attention you built all week is like a theater with no seat numbers. It can tell you *who* is in the room and how the words relate, but it has no idea *where* each word sits. Today you'll fix that with one clever trick — giving every word its own **seat** number — and it's the very trick humming quietly inside a model like ChatGPT so it can read a long sentence in the right order.
@goal Together we'll first feel the bug: attention treats a sentence as an unordered pile of words, so shuffling them changes nothing. Then we fix it by stamping each word with a position signal — a **seat** number — so the same word can mean different things depending on where it sits. We'll meet the clever wave-based recipe the first Transformer used, see why it needs many waves and not just one, meet the learned-seat alternative, and finish knowing exactly what this trick does and doesn't do. Every new word gets said in plain English the moment it shows up.

@@@ concept id=c1 tag="Order = meaning" title="Why word order carries meaning" gotit="Got why order matters"
Let's warm up on something you already know in your bones, no math at all. Take three words — *dog*, *bites*, *man* — and line them up two ways. "Dog bites man" is a boring Tuesday. "Man bites dog" is front-page news. Same three words, same letters, same everything — the *only* difference is the **seat** each word sits in, the order they come in. That tiny difference flips the whole meaning. So any machine that wants to understand language has to know not just *which* words are there, but *where* each one sits.

Think of the **seats around your dinner table**. Imagine the food is fixed — the same plates, the same people — but you shuffle who sits where. Put the toddler next to the birthday cake and you get a very different evening than if grandma sits there. The people didn't change; their *seats* did, and that changed the whole story. Words in a sentence are the people; their order is the seating chart. Move a word to a new seat and the meaning can flip completely.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="The same three word-cards dog, bites, man arranged in two different seat orders. Top row spells dog bites man, a calm everyday event. Bottom row spells man bites dog, a shocking news event. Same three cards, only the seating order changed, and the meaning flips."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same three words · two seat orders · opposite meanings</text><rect x="40" y="36" width="90" height="34" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="85" y="58" text-anchor="middle" fill="#276b45" font-size="11">dog</text><rect x="145" y="36" width="90" height="34" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="190" y="58" text-anchor="middle" fill="#6B645E" font-size="11">bites</text><rect x="250" y="36" width="90" height="34" rx="6" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="295" y="58" text-anchor="middle" fill="#5E5191" font-size="11">man</text><text x="415" y="58" fill="#276b45" font-size="10">a calm Tuesday</text><text x="30" y="112" fill="#9A938A" font-size="9">now swap seats 1 and 3 →</text><rect x="40" y="122" width="90" height="34" rx="6" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="85" y="144" text-anchor="middle" fill="#5E5191" font-size="11">man</text><rect x="145" y="122" width="90" height="34" rx="6" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="190" y="144" text-anchor="middle" fill="#6B645E" font-size="11">bites</text><rect x="250" y="122" width="90" height="34" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="295" y="144" text-anchor="middle" fill="#276b45" font-size="11">dog</text><text x="415" y="144" fill="#C0392B" font-size="10">front-page news!</text><text x="260" y="180" text-anchor="middle" fill="#9A938A" font-size="9">the cards never changed — only the seat order did</text></g></svg>
%%%

**What the dinner-table picture gets right:** same people, different seats, whole different evening — exactly how the same words in a different order can mean the opposite thing. **Where it breaks down:** at a dinner table a person keeps being the same person no matter where they sit, but for a model a word only becomes "the subject" or "the object" *because* of where it sits — the seat is part of the meaning, not just a place to put it.

#### Hold this one thought
A sentence is not a bag of words tossed together. It's an *ordered line* of words, and the order carries real meaning. Keep that in your hand — because in the very next concept we'll discover that the attention you built has no idea about any of it.

@@@ concept id=c2 tag="Attention is order-blind" title="Attention sees a pile, not a line" gotit="Got the shuffle bug"
Now the twist. All week you built attention as a meeting: each word looks around, rates how much it wants to listen to every other word, and blends them into one answer. It's brilliant at *who relates to whom*. But here's the thing it quietly does NOT do — it never checks what **seat** anyone is in. To attention, a sentence is a pile of words dumped on a table, not a line of words in order. Scramble the pile and it hands back the exact same answer. Fancy people call this being [[permutation-invariant||A big word meaning "order-blind": shuffle the inputs and the output does not change.]] — "order-blind." For most jobs that would be a nice feature. For reading language it's a bug, because order is meaning.

Think of **dumping a bag of alphabet fridge magnets onto the table**. The magnets are all there — every letter present and accounted for — but they landed in a jumble. From the pile alone you can see *which* letters you have, but not what *word* they spell, because a pile has no order. Attention, with no seat numbers, sees your sentence exactly like that pile: all the words, none of the order.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="On the left, three ordered word tiles spelling a sentence. An arrow points to the middle where they are shown as a jumbled pile of the same three tiles with no order. On the right, attention returns the same answer whether the words were ordered or piled — a checkmark labelled same answer either way."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Attention sees a pile of words, not an ordered line</text><text x="70" y="42" text-anchor="middle" fill="#6B645E" font-size="9">an ordered sentence</text><rect x="34" y="50" width="26" height="24" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="47" y="67" text-anchor="middle" fill="#276b45" font-size="9">the</text><rect x="62" y="50" width="26" height="24" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="75" y="67" text-anchor="middle" fill="#6B645E" font-size="9">cat</text><rect x="90" y="50" width="26" height="24" rx="3" fill="#EAF0FA" stroke="#5E5191"/><text x="103" y="67" text-anchor="middle" fill="#5E5191" font-size="9">sat</text><text x="150" y="67" fill="#8A6D3B" font-size="13">→</text><rect x="180" y="34" width="150" height="90" rx="10" fill="#F1ECE3" stroke="#B8AEA2" stroke-width="1.5" stroke-dasharray="4,3"/><text x="255" y="30" text-anchor="middle" fill="#9A938A" font-size="9">what attention sees: a pile</text><g transform="rotate(-18 220 70)"><rect x="205" y="58" width="26" height="24" rx="3" fill="#EAF0FA" stroke="#5E5191"/><text x="218" y="75" text-anchor="middle" fill="#5E5191" font-size="9">sat</text></g><g transform="rotate(22 260 90)"><rect x="247" y="80" width="26" height="24" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="260" y="97" text-anchor="middle" fill="#276b45" font-size="9">the</text></g><g transform="rotate(8 290 62)"><rect x="277" y="52" width="26" height="24" rx="3" fill="#FDF9F3" stroke="#B8AEA2"/><text x="290" y="69" text-anchor="middle" fill="#6B645E" font-size="9">cat</text></g><text x="360" y="67" fill="#8A6D3B" font-size="13">→</text><rect x="392" y="52" width="108" height="40" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="446" y="70" text-anchor="middle" fill="#8A6D3B" font-size="9">same answer</text><text x="446" y="84" text-anchor="middle" fill="#8A6D3B" font-size="9">either way ✓</text><text x="260" y="164" text-anchor="middle" fill="#C0392B" font-size="10">order-blind: shuffle the words → attention's answer does not change</text><text x="260" y="184" text-anchor="middle" fill="#9A938A" font-size="9">great for a set of things · a bug for a sentence, where order is meaning</text></g></svg>
%%%

**What the fridge-magnet picture gets right:** a pile shows you every letter but hides the word, exactly like attention seeing every word but not the order. **Where it breaks down:** you *can* still slowly reorder real magnets by hand, but attention has no hands — with no seat signal it has literally no way to recover the order, ever.

#### Why does this happen? One plain reason
Under the hood, attention builds each word's answer by taking a *weighted sum* of the other words. And a sum doesn't care what order you add things in: `3 + 5 + 2` is the same as `2 + 3 + 5`. That's the whole reason. Because the core step is "add up the words, weighted," swapping the order of the words being added lands on the identical total. Order isn't built in — it has to be *handed in* as an extra signal.

#### Watch the shuffle change nothing
Let's catch the bug red-handed. We'll run tiny attention on "the cat sat," then shuffle the word rows to "sat the cat" and run it again. **Predict first:** will each word's output vector change, or will attention hand back the same set of numbers? Guess, then reveal.

%%% demo id=shuffle label="run attention, then shuffle the words and run again"
code: attention(["the", "cat", "sat"])   vs   attention(["sat", "the", "cat"])   # same words, new seats
out: ordered  ["the","cat","sat"]  ->  the:[0.31,0.62]  cat:[0.55,0.40]  sat:[0.48,0.51]
     shuffled ["sat","the","cat"]  ->  the:[0.31,0.62]  cat:[0.55,0.40]  sat:[0.48,0.51]
     every word's output vector is IDENTICAL — only the row order moved with it
take: <b>Same numbers, both times — attention is order-blind.</b> Each word got the exact same answer no matter where it sat, because attention just re-weights and sums the other words, and a sum ignores order. So "the cat sat" and "sat the cat" look identical to it. That's the shuffle bug in one run — and the whole rest of the day is the fix.
%%%

You just proved the bug with your own eyes. That is a real victory lap — because now the fix, coming next, will feel obvious and satisfying.

@@@ concept id=c3 tag="Give each word a seat" title="Positional encoding — stamp a seat number on every word" gotit="Got the seat-number fix"
Here comes the fix, and it's beautifully simple. If attention can't see order, we hand it order — we give every word a **seat** number. Before the words ever walk into the attention meeting, we stamp each one with a little signal that says "you are in seat 0," "you are in seat 1," "you are in seat 2," and so on. This stamp is called a [[positional encoding||A little vector added to each word that tells the model which seat, or position, the word sits in.]] — the position signal. Now the same word carries different numbers depending on where it sits, so "cat" in seat 1 is no longer identical to "cat" in seat 3. Order becomes something attention can finally read.

Think of **numbered stickers on moving boxes**. On moving day every box looks the same brown cube. So you slap a sticker on each: "Box 1 — kitchen," "Box 2 — kitchen," "Box 3 — bedroom." The boxes didn't change, but now each one carries a little note about *where it belongs in the order*. A positional encoding is that sticker for a word: a small tag added on that says which seat it's in.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Three identical word vectors for the word cat shown as bars. On the left they are bare and identical. In the middle a small seat-number stamp is added to each — seat 0, seat 1, seat 2. On the right the three results are now different from each other. The seat stamp is added on top of the word, not glued beside it."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Add a seat stamp to each word → identical words become distinct</text><text x="60" y="40" text-anchor="middle" fill="#6B645E" font-size="9">word "cat"</text><rect x="30" y="48" width="60" height="20" fill="#EAF0FA" stroke="#5E5191"/><rect x="30" y="72" width="60" height="20" fill="#EAF0FA" stroke="#5E5191"/><rect x="30" y="96" width="60" height="20" fill="#EAF0FA" stroke="#5E5191"/><text x="60" y="132" text-anchor="middle" fill="#9A938A" font-size="8">same 3 times</text><text x="150" y="72" fill="#2D8B55" font-size="13">+</text><text x="205" y="40" text-anchor="middle" fill="#6B645E" font-size="9">seat stamp</text><rect x="175" y="48" width="60" height="20" fill="#EAF5EE" stroke="#2D8B55"/><text x="205" y="63" text-anchor="middle" fill="#276b45" font-size="8">seat 0</text><rect x="175" y="72" width="60" height="20" fill="#EAF5EE" stroke="#2D8B55"/><text x="205" y="87" text-anchor="middle" fill="#276b45" font-size="8">seat 1</text><rect x="175" y="96" width="60" height="20" fill="#EAF5EE" stroke="#2D8B55"/><text x="205" y="111" text-anchor="middle" fill="#276b45" font-size="8">seat 2</text><text x="295" y="72" fill="#8A6D3B" font-size="13">=</text><text x="380" y="40" text-anchor="middle" fill="#6B645E" font-size="9">now all different</text><rect x="330" y="48" width="100" height="20" fill="#FCF3DC" stroke="#C99A12"/><text x="380" y="63" text-anchor="middle" fill="#8A6D3B" font-size="8">cat @ seat 0</text><rect x="330" y="72" width="100" height="20" fill="#FDECEA" stroke="#C0392B"/><text x="380" y="87" text-anchor="middle" fill="#C0392B" font-size="8">cat @ seat 1</text><rect x="330" y="96" width="100" height="20" fill="#F4EBFB" stroke="#7c3aed"/><text x="380" y="111" text-anchor="middle" fill="#7c3aed" font-size="8">cat @ seat 2</text><text x="260" y="160" text-anchor="middle" fill="#2D8B55" font-size="10">the stamp is ADDED right on top of the word (same width in, same width out)</text><text x="260" y="182" text-anchor="middle" fill="#9A938A" font-size="9">not glued beside it — the word doesn't get any wider, it just carries its seat</text></g></svg>
%%%

**What the moving-box picture gets right:** an identical box plus a "which one in the order" sticker becomes a box you can place correctly — just like an identical word plus a seat stamp becomes a word attention can place in order. **Where it breaks down:** a box sticker sits *on the outside* and peels off cleanly, but a seat stamp is *added into* the word's own numbers — it blends in and becomes part of the word, it doesn't stay a separate label.

#### One quiet but important detail: we ADD it, not glue it on
There are two ways you might imagine attaching a seat number to a word. You could **glue it beside** the word, making the word longer (that's called concatenating). Or you could **add it on top** of the word's existing numbers, keeping the same width. The Transformer does the second: the seat signal is a vector the *same size* as the word, and we simply add the two together. Same width goes in, same width comes out — nothing gets bigger, the word just quietly carries its seat now.

#### Watch the same word take on two different seats
Let's make it real. Take the identical word "cat" and drop it into seat 1, then into seat 3. **Predict first:** after we add the seat stamp, will the two "cat"s carry the same numbers or different ones? Guess, then reveal.

%%% demo id=addseat label="the same word cat in two different seats"
code: cat = [0.20, 0.90]           # the plain word, identical both times
      seat_1 = [0.84, 0.54]        # stamp for seat 1
      seat_3 = [0.14, -0.99]       # stamp for seat 3
      cat_in_seat_1 = cat + seat_1     ;     cat_in_seat_3 = cat + seat_3
out: cat_in_seat_1 = [0.20+0.84, 0.90+0.54] = [1.04, 1.44]
     cat_in_seat_3 = [0.20+0.14, 0.90-0.99] = [0.34, -0.09]
     -> same word "cat", but two clearly DIFFERENT vectors now
take: <b>Same word, two seats, two different vectors.</b> Adding the seat stamp made "cat @ seat 1" and "cat @ seat 3" distinct — so attention can finally tell them apart and read order. And notice the width never changed: 2 numbers in, 2 numbers out. That is the entire core idea of positional encoding; the rest of the day is just *how we choose those seat stamps well*.
%%%

@@@ concept id=c4 tag="Seat stamps from waves" title="Building the seat stamps out of waves" gotit="Got the wave recipe"
So the fix is "add a seat stamp." But what *numbers* should the stamp contain? The very first Transformer picked a clever answer: build each seat stamp out of **waves** — smooth up-and-down ripples like a jump-rope or a water wave. This is called [[sinusoidal encoding||A recipe that builds each seat's stamp from smooth sine and cosine waves at different speeds.]] ("sinusoidal" just means "wave-shaped," from *sine*, the math name for a wave). Here's the neat part: instead of one wave, it uses *many* waves at different speeds — some wiggling fast, some slow. Each seat reads off one point from every wave, and that bundle of readings becomes its stamp. Because the waves run at different speeds, no two seats ever get the exact same bundle.

Think of a **clock face with an hour hand and a minute hand**. The minute hand sweeps fast; the hour hand crawls slow. At any single moment, the *pair* of hand positions names one exact time — 3:47 looks different from 3:48 (minute hand moved) and different from 4:47 (hour hand moved). Neither hand alone is enough — two identical-looking minutes an hour apart would clash — but *together*, fast plus slow, they pin down one exact moment with no repeats all day. Waves at different speeds do the same for seats: fast waves tell nearby seats apart, slow waves tell far-apart seats apart, and together they give every seat a stamp nothing else shares.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A clock face on the left with a fast minute hand and a slow hour hand, labelled together they name one exact time. On the right, two waves are drawn: a fast wiggling wave labelled fast wave tells nearby seats apart, and a slow gentle wave labelled slow wave tells far seats apart. Together they give each seat a unique reading."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Fast hand + slow hand = one exact time · fast wave + slow wave = one exact seat</text><circle cx="90" cy="105" r="52" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="90" y1="105" x2="90" y2="66" stroke="#C0392B" stroke-width="2.5"/><text x="118" y="60" fill="#C0392B" font-size="8">minute (fast)</text><line x1="90" y1="105" x2="118" y2="118" stroke="#5E5191" stroke-width="3"/><text x="120" y="132" fill="#5E5191" font-size="8">hour (slow)</text><text x="90" y="176" text-anchor="middle" fill="#9A938A" font-size="9">two speeds → one exact time</text><path d="M210 80 Q222 56 234 80 T258 80 T282 80 T306 80 T330 80 T354 80 T378 80 T402 80 T426 80 T450 80 T474 80" fill="none" stroke="#C0392B" stroke-width="1.6"/><text x="484" y="83" fill="#C0392B" font-size="8">fast</text><path d="M210 130 Q270 96 330 130 T450 130 T570 130" fill="none" stroke="#5E5191" stroke-width="1.8"/><text x="484" y="133" fill="#5E5191" font-size="8">slow</text><text x="330" y="162" text-anchor="middle" fill="#9A7208" font-size="9">fast wave separates NEARBY seats</text><text x="330" y="178" text-anchor="middle" fill="#5E5191" font-size="9">slow wave separates FAR-APART seats</text></g></svg>
%%%

**What the clock picture gets right:** two hands at different speeds together name one moment with no repeats, exactly how waves at different speeds together name one seat with no repeats. **Where it breaks down:** a clock has only two hands and *does* repeat every 12 hours, but the encoding uses many waves and spreads their speeds so widely that a real sentence's seats never come back around to clash.

#### Why one wave isn't enough — the neighbor puzzle
Here's a puzzle worth solving. What if we tried just *one* slow wave for the seat stamps? A slow wave barely moves from one seat to the next — seat 5 and seat 6 would read almost the same value and **collide**, looking like the same seat. And one fast wave alone? It wiggles so fast it repeats quickly, so seat 2 and seat 40 might land on the same value and collide too. The escape is to use *both, and more* — combine many wave speeds. Fast waves keep neighbors apart; slow waves keep distant seats apart; stack enough of them and every seat gets a fingerprint no other seat can match.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Three panels. Left: one slow wave alone, with two adjacent seats reading nearly the same height and marked collide. Middle: one fast wave alone, with two far-apart seats reading the same height and marked collide. Right: fast plus slow combined, where every seat has a distinct pair of readings, marked all unique."><g font-family="monospace" font-size="9"><text x="260" y="14" text-anchor="middle" fill="#2C2A28" font-size="12">One wave collides · many waves give every seat a unique fingerprint</text><text x="86" y="34" text-anchor="middle" fill="#C0392B" font-size="9">only a SLOW wave</text><path d="M20 70 Q60 46 100 70 T180 70" fill="none" stroke="#5E5191" stroke-width="1.6"/><circle cx="70" cy="61" r="3" fill="#C0392B"/><circle cx="84" cy="64" r="3" fill="#C0392B"/><text x="86" y="92" text-anchor="middle" fill="#C0392B" font-size="8">neighbors ≈ same → collide</text><text x="260" y="34" text-anchor="middle" fill="#C0392B" font-size="9">only a FAST wave</text><path d="M195 70 Q205 50 215 70 T235 70 T255 70 T275 70 T295 70 T315 70 T335 70" fill="none" stroke="#C99A12" stroke-width="1.6"/><circle cx="215" cy="70" r="3" fill="#C0392B"/><circle cx="315" cy="70" r="3" fill="#C0392B"/><text x="265" y="92" text-anchor="middle" fill="#C0392B" font-size="8">far seats repeat → collide</text><text x="440" y="34" text-anchor="middle" fill="#2D8B55" font-size="9">FAST + SLOW together</text><path d="M360 70 Q368 54 376 70 T392 70 T408 70 T424 70 T440 70 T456 70 T472 70 T488 70 T504 70" fill="none" stroke="#C99A12" stroke-width="1.3"/><path d="M360 74 Q400 52 440 74 T520 74" fill="none" stroke="#5E5191" stroke-width="1.5"/><text x="440" y="92" text-anchor="middle" fill="#2D8B55" font-size="8">every seat unique ✓</text><text x="30" y="126" fill="#6B645E" font-size="9">each seat's stamp = its reading from EVERY wave, bundled together:</text><rect x="30" y="136" width="70" height="22" rx="3" fill="#FCF3DC" stroke="#C99A12"/><text x="65" y="151" text-anchor="middle" fill="#8A6D3B" font-size="8">seat 0: [.., ..]</text><rect x="110" y="136" width="70" height="22" rx="3" fill="#FDECEA" stroke="#C0392B"/><text x="145" y="151" text-anchor="middle" fill="#C0392B" font-size="8">seat 1: [.., ..]</text><rect x="190" y="136" width="70" height="22" rx="3" fill="#EAF5EE" stroke="#2D8B55"/><text x="225" y="151" text-anchor="middle" fill="#276b45" font-size="8">seat 2: [.., ..]</text><rect x="270" y="136" width="70" height="22" rx="3" fill="#EAF0FA" stroke="#5E5191"/><text x="305" y="151" text-anchor="middle" fill="#5E5191" font-size="8">seat 3: [.., ..]</text><text x="260" y="182" text-anchor="middle" fill="#9A938A" font-size="9">fast wave keeps neighbors apart · slow wave keeps far seats apart · together = no collisions</text><text x="260" y="200" text-anchor="middle" fill="#276b45" font-size="9">that bundle of readings IS the seat stamp we add to the word</text></g></svg>
%%%

!!! c-info 🧮
<b>Optional (skippable) — the exact wave formula.</b> If you love the math: for seat `pos`, the stamp fills even slots with a sine and odd slots with a cosine — `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))` and `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`. Here `pos` is which seat, `i` is which slot-pair, `d_model` is the word width, and `10000` is the knob that spreads the wave speeds from fast (front slots) to slow (back slots). You never need this to *use* attention — it's here for the curious, and it gets the full worked-numbers treatment in the interview deep-dive.
!!!

#### Watch the fingerprints come out unique
Let's see the stamps for the first few seats and check they're all different. **Predict first:** we build stamps from a fast wave and a slow wave. Will seat 0, 1, and 2 come out identical, or each unique? Guess, then reveal.

%%% demo id=fingerprint label="build seat stamps from a fast + slow wave"
code: two waves — one fast, one slow.  stamp[seat] = [fast(seat), slow(seat)]
out: seat 0 -> [0.00, 0.00]     (both waves start at 0)
     seat 1 -> [0.84, 0.01]     (fast wave jumped a lot, slow wave barely moved)
     seat 2 -> [0.91, 0.02]     (fast wave near its top, slow wave still creeping)
     seat 3 -> [0.14, 0.03]     (fast wave swung back down, slow wave keeps creeping)
     -> all four stamps are DIFFERENT — a unique fingerprint per seat
take: <b>Every seat got its own fingerprint.</b> The fast wave does the heavy lifting for nearby seats (watch it swing 0.00 → 0.84 → 0.91 → 0.14), while the slow wave inches up (0.00 → 0.01 → 0.02 → 0.03) to keep far-apart seats from ever repeating. One wave alone would have collided; two speeds together never do. Add each of these stamps to its word and order is now baked in.
%%%

@@@ concept id=c5 tag="Fixed vs learned seats" title="Two ways to make seat numbers" gotit="Got both families"
Now a fork in the road — because there are two whole families of ways to make seat stamps, and it's good to know both exist. The wave recipe you just met is **fixed**: it comes straight from a formula, so nobody has to train it. Seat 5 always gets the same wave readings, on day one and forever, whether the model has trained for an hour or a year. The other family is **learned**: instead of a formula, you let the model *invent* its own stamp for each seat during training, the same way it learns everything else. Both plug in the exact same way — add the stamp to the word — they just differ in *where the numbers come from*: a formula, or practice.

Think of **two ways to number the seats at an event**. Way one: the seats come with numbers already printed on them from the factory — no work for you, they're just *there* (that's the fixed wave formula). Way two: you hand out blank chairs and let the crowd write their own labels over the evening, tuning them until the seating works best (that's the learned way). Either way every chair ends up with a number; you just chose whether it was printed for free or worked out by hand.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two families side by side. Left: a fixed seat stamp, a chair with a number pre-printed from the factory by a formula, needs no training, works at any length. Right: a learned seat stamp, a blank chair whose number is written and tuned during training, flexible but undefined past the longest trained seat."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two families of seat stamps — same slot, different origin</text><rect x="26" y="34" width="216" height="140" rx="10" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="134" y="54" text-anchor="middle" fill="#276b45" font-size="11" font-weight="bold">FIXED (the wave formula)</text><rect x="104" y="66" width="60" height="44" rx="4" fill="#fff" stroke="#2D8B55"/><text x="134" y="93" text-anchor="middle" fill="#276b45" font-size="12">#5</text><text x="134" y="126" text-anchor="middle" fill="#276b45" font-size="9">pre-printed by a formula</text><text x="134" y="142" text-anchor="middle" fill="#276b45" font-size="9">no training needed</text><text x="134" y="158" text-anchor="middle" fill="#276b45" font-size="9">works at any length</text><rect x="278" y="34" width="216" height="140" rx="10" fill="#F4EBFB" stroke="#7c3aed" stroke-width="1.5"/><text x="386" y="54" text-anchor="middle" fill="#7c3aed" font-size="11" font-weight="bold">LEARNED (model invents it)</text><rect x="356" y="66" width="60" height="44" rx="4" fill="#fff" stroke="#7c3aed" stroke-dasharray="3,2"/><text x="386" y="93" text-anchor="middle" fill="#7c3aed" font-size="12">#?</text><text x="386" y="126" text-anchor="middle" fill="#7c3aed" font-size="9">written during training</text><text x="386" y="142" text-anchor="middle" fill="#7c3aed" font-size="9">flexible, can fit the data</text><text x="386" y="158" text-anchor="middle" fill="#7c3aed" font-size="9">undefined past longest seat seen</text></g></svg>
%%%

**What the numbered-seats picture gets right:** printed-from-the-factory versus written-by-hand is exactly the fixed-formula versus learned-during-training split, and either way every seat ends up numbered. **Where it breaks down:** real printed and hand-written numbers both work for any seat you point at, but the *learned* stamps only exist for seats the model actually saw in training — ask for seat 5000 when it only ever practiced up to seat 512 and there's simply no label, a real weakness we'll name in a moment.

#### The two families, side by side
Here's the whole choice on one card. Neither is "the winner" — they trade off, and both show up in real models.

%%% table
:: Fixed (sinusoidal waves) :: Learned (model invents each seat)
Where the numbers come from :: a wave formula :: training, like any other weights
Needs training? :: no — ready on day one :: yes — tuned while the model learns
Extra parameters :: none :: one vector per seat to store and train
Seats past training length :: still defined by the formula :: undefined — no label exists
%%%

You now know both families exist and how they differ. That "learned stamps have no label past the longest trained seat" line is a genuine weakness — hold onto it, because it's exactly the crack the next concept opens up.

@@@ concept id=c6 tag="What it does & doesn't do" title="Two honest limits of the seat trick" gotit="Got the limits"
Time for some honesty — the kind that makes you sound sharp, not the kind that spoils the fun. The seat trick is a real fix, but it's not magic. Two limits are worth knowing, and each one is really an invitation to something cooler later.

**Limit one: the stamp only *tells* where — it doesn't *teach* how to use it.** Handing every word a seat number is like giving everyone a printed ticket that says their seat. It doesn't force anyone to actually *use* the seat wisely — the attention layers still have to *learn*, during training, how to make sense of the order. Positional encoding supplies the signal; the network has to figure out what to do with it.

**Limit two: fixed seat numbers don't stretch to seats they've never seen.** If a model only ever practiced on short sentences — seats 0 through 100 — a much longer sentence full of seats 500, 800, 2000 is new territory, and the model can handle it poorly. This is called failing to *extrapolate* to longer inputs. And this exact weakness is *why* smarter position methods were invented — you'll meet them on a later day.

Think of a **race with numbered lanes painted on the track**. Painting lane numbers (the seat stamp) tells each runner which lane is theirs — but it doesn't teach anyone *how to run*; that still takes practice (limit one). And if the painter only ever marked lanes 1 through 8, a race that suddenly needs lane 20 has no paint out there — the runner in "lane 20" is lost, because that stretch of track was never marked (limit two).

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A running track. The near section, seats 0 to 100, has clearly painted lane numbers and a runner labelled trained here, comfortable. Farther along, seats past 500, the paint fades to nothing and a runner labelled tested here, longer sentence looks lost because the lanes were never marked out there."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Painted lanes only where the model practiced — beyond that, no marks</text><rect x="20" y="40" width="220" height="120" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="130" y="60" text-anchor="middle" fill="#276b45" font-size="10" font-weight="bold">seats 0–100: trained here</text><line x1="40" y1="76" x2="230" y2="76" stroke="#2D8B55" stroke-dasharray="6,4"/><line x1="40" y1="100" x2="230" y2="100" stroke="#2D8B55" stroke-dasharray="6,4"/><line x1="40" y1="124" x2="230" y2="124" stroke="#2D8B55" stroke-dasharray="6,4"/><text x="130" y="150" text-anchor="middle" fill="#276b45" font-size="9">clear seat numbers → comfortable ✓</text><rect x="280" y="40" width="220" height="120" rx="6" fill="#FDECEA" stroke="#C0392B" stroke-width="1.5"/><text x="390" y="60" text-anchor="middle" fill="#C0392B" font-size="10" font-weight="bold">seats past 500: tested here</text><line x1="300" y1="76" x2="360" y2="76" stroke="#C0392B" stroke-dasharray="3,7" opacity="0.4"/><line x1="300" y1="100" x2="345" y2="100" stroke="#C0392B" stroke-dasharray="3,7" opacity="0.3"/><line x1="300" y1="124" x2="335" y2="124" stroke="#C0392B" stroke-dasharray="3,7" opacity="0.25"/><text x="420" y="102" text-anchor="middle" fill="#C0392B" font-size="9">paint fades…</text><text x="390" y="150" text-anchor="middle" fill="#C0392B" font-size="9">longer sentence → gets lost ✗</text><text x="260" y="182" text-anchor="middle" fill="#9A938A" font-size="9">the stamp says WHERE, never HOW to run · and only for seats it actually saw</text><text x="260" y="197" text-anchor="middle" fill="#5E5191" font-size="9">→ this is exactly what smarter methods (a later day) are built to fix</text></g></svg>
%%%

**What the painted-lanes picture gets right:** paint tells runners which lane is theirs but not how to run, and it runs out where nobody marked it — the two limits, exactly. **Where it breaks down:** a painter *could* just keep painting more lanes, but a model can't simply "paint more" trained seats without more training data at those lengths, which is what makes the long-sentence problem genuinely hard.

!!! c-info 🔭
<b>The exciting part:</b> that "gets lost on longer sentences" crack is not a dead end — it's the doorway to the methods behind today's long-context models that read whole books. Later days introduce smarter schemes (rotate each word by its seat instead of adding a stamp; or gently nudge attention toward nearby words) that stretch to lengths the model never trained on. You've earned the map; those days fill it in.
!!!

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
Let's gather the whole day onto one page you can come back to any time. Here's a fresh way to hold it, nothing to do with theaters — think of a **numbered relay race**. The runners are your words. Give each a lane number (the **seat** stamp) and the race makes sense in order; leave them unnumbered and it's just people running around (the order-blind bug). Build the numbers from a formula and they're free and endless; let runners write their own and they're flexible but only reach as far as the track was marked. Either way, the number tells them *where*, never *how* to run. **Where the relay breaks down:** runners obviously know their own order, but a model genuinely can't — order only enters through the stamp we add.

%%% svg
<svg viewBox="0 0 520 176" role="img" aria-label="A recap flow in four steps left to right. Step 1: order is meaning, dog bites man versus man bites dog. Step 2: plain attention is order-blind, it sees a pile. Step 3: add a seat stamp built from waves so every word carries its position. Step 4: honest limits — it only tells where, and fixed stamps do not stretch to unseen seats. Arrows connect the steps."><g font-family="monospace" font-size="9"><text x="260" y="15" text-anchor="middle" fill="#2C2A28" font-size="12">The day in one line: order = meaning → attention is order-blind → add a seat stamp → its limits</text><rect x="10" y="46" width="112" height="70" rx="8" fill="#F1ECE3" stroke="#B8AEA2" stroke-width="1.5"/><text x="66" y="72" text-anchor="middle" fill="#6B645E">order = meaning</text><text x="66" y="90" text-anchor="middle" fill="#9A938A" font-size="8">dog bites man ≠</text><text x="66" y="102" text-anchor="middle" fill="#9A938A" font-size="8">man bites dog</text><rect x="138" y="46" width="112" height="70" rx="8" fill="#FDECEA" stroke="#C0392B" stroke-width="1.5"/><text x="194" y="72" text-anchor="middle" fill="#C0392B">order-blind</text><text x="194" y="90" text-anchor="middle" fill="#C0392B" font-size="8">attention sees a pile</text><text x="194" y="102" text-anchor="middle" fill="#C0392B" font-size="8">shuffle = same answer</text><rect x="266" y="46" width="112" height="70" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="322" y="72" text-anchor="middle" fill="#276b45">add a seat stamp</text><text x="322" y="90" text-anchor="middle" fill="#276b45" font-size="8">waves, many speeds</text><text x="322" y="102" text-anchor="middle" fill="#276b45" font-size="8">unique fingerprint</text><rect x="394" y="46" width="116" height="70" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="452" y="72" text-anchor="middle" fill="#5E5191">honest limits</text><text x="452" y="90" text-anchor="middle" fill="#5E5191" font-size="8">tells where, not how</text><text x="452" y="102" text-anchor="middle" fill="#5E5191" font-size="8">weak past trained seats</text><line x1="122" y1="81" x2="138" y2="81" stroke="#8A6D3B" stroke-width="1.5"/><line x1="250" y1="81" x2="266" y2="81" stroke="#8A6D3B" stroke-width="1.5"/><line x1="378" y1="81" x2="394" y2="81" stroke="#8A6D3B" stroke-width="1.5"/><text x="260" y="150" text-anchor="middle" fill="#5E5191" font-size="10">stamp the seat before attention → order finally carries meaning</text><text x="260" y="168" text-anchor="middle" fill="#9A938A" font-size="9">full input side of a transformer: embeddings → Q/K/V → attention → many heads → + position</text></g></svg>
%%%

#### The day in five beats
1. **Order is meaning.** "Dog bites man" and "man bites dog" share every word — only the seat order differs, and it flips the meaning.
2. **Attention is order-blind.** It builds each answer from a weighted sum, and sums ignore order, so shuffling the words changes nothing. That's the bug.
3. **Give each word a seat stamp.** Add a position signal to each word *before* attention (added, not glued on) so the same word differs by where it sits.
4. **Build the stamps from waves.** Sinusoidal encoding uses many waves at different speeds — fast for neighbors, slow for far seats — so every seat gets a unique fingerprint and none collide. Fixed (from a formula) is one family; learned (invented in training) is the other.
5. **Know the limits.** The stamp only says *where*, not *how* to use it, and fixed stamps don't stretch to seats longer than training — which is exactly what smarter, later methods repair.

#### Cheat-sheet — every word from today
%%% jargon
seat / position | which slot a word sits in — 0, 1, 2, … — the order it comes in
permutation-invariant | order-blind: shuffle the inputs and the output does not change
positional encoding | a signal added to each word telling it which seat it sits in
sinusoidal encoding | seat stamps built from sine and cosine waves at many speeds
frequency | how fast a wave wiggles — fast waves split neighbors, slow waves split far seats
fingerprint | the unique bundle of wave readings each seat gets, so no two collide
fixed encoding | seat stamps that come from a formula — no training, works at any length
learned encoding | seat stamps the model invents during training — flexible but undefined past trained lengths
extrapolate | handle sentences longer than any seen in training — where fixed stamps struggle
%%%

You just finished the entire input side of a transformer: embeddings, Q/K/V, attention, many heads, and now position. Come back to this page whenever the shuffle problem feels foggy — the five beats are all of it. Next up: the Module 4 Review Gate, where you get to prove it all fits together.

@@@ quiz id=quiz tag="Quiz" title="Four quick checks" gotit="answer all first"
%%% quiz
q: Why is plain self-attention called "order-blind" (permutation-invariant)? | a:2 | it can only read short sentences | it forgets words over time | it builds each answer from a weighted sum, and sums ignore the order of what you add | it never looks at other words | fb: Attention re-weights and sums the words, and a sum ignores order — so shuffling the input changes nothing.
q: What does positional encoding actually do to fix the shuffle problem? | a:1 | it reorders the words alphabetically | it adds a per-position "seat" signal to each word's embedding before attention | it removes the softmax step | it makes the sentence shorter | fb: You add a seat-number signal to each word so the same word differs by where it sits — order becomes readable.
q: Why does sinusoidal encoding use many waves at different speeds instead of one? | a:3 | to make training faster | to save memory | one wave is prettier | one wave alone collides — fast waves split neighbors, slow waves split far seats, so together every seat is unique | fb: A single wave makes nearby or far seats collide; mixing speeds gives every seat a unique fingerprint.
q: Which is a real limit of fixed sinusoidal encoding? | a:0 | it can struggle on sentences much longer than any seen in training (poor extrapolation) | it forgets which words are present | it makes attention slower than a for-loop | it can only be used once per model | fb: Fixed absolute stamps don't automatically stretch to unseen longer lengths — which motivates the smarter methods introduced later.
%%%

@@@ produce id=produce tag="Produce" title="Catch the shuffle bug, then fix it" gotit="Done"
Time to prove it with your own hands. You'll first catch attention being order-blind, then watch the seat stamp fix it — in `experiment.py`.

**Predict before you write any code:** if you run a tiny self-attention on a sentence, then shuffle the word rows and run it again, will each word's output vector change, or come back the same? Write your guess down. Then predict again for *after* you add positional encoding.

Then build it, one step at a time:
1. Write a tiny `self_attention(x)` (reuse your Day-3 `softmax(Q @ Kᵀ / √d_k) @ V`). Run it on an input with `seq=3, d_model=4`. **Print** each word's output vector.
2. Shuffle the input rows and run `self_attention` again. **Observe** that the *set* of per-word outputs is unchanged — you just caught permutation-invariance red-handed.
3. Write `sinusoidal_encoding(max_len, d_model)` from the wave idea (even slots sine, odd slots cosine, waves at spreading speeds). **Print** a small table, say 6 seats × 8 slots, and **notice** every row is unique — seat 0 starts `[0, 1, 0, 1]`.
4. Add the encoding to the embeddings (`x + pe[:seq]`) and run attention on two *different* word orders. **Watch** the two outputs now come out different — the fix works.

What you should see by the end: without the seat stamp, shuffling changes nothing (the bug); with it, order finally changes the answer (the fix). Did your two predictions match what printed? Write it in `experiment.py` and run it.

@@@ fin
