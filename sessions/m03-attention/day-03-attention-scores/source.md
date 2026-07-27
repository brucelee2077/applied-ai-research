---
quest_id: wf4-d03-attention-scores
mode: concept
donor: v9-base.donor
page_title: "Module 4 · Day 3 — Attention Scores &amp; Softmax"
module_label: "Module 4 · Represent · Day 3"
title: "Attention Scores &amp; Softmax"
subtitle: "From Matches to a Weighted Blend"
brand_sub: "Foundations · M4 Day 3"
spine: "budget"
nav_prev_href: "../day-02-qkv/lesson.html"
nav_prev_label: "Query, Key, Value"
nav_next_href: "../day-04-multihead/lesson.html"
nav_next_label: "Multi-Head Attention"
fin_title: "Module 4 · Day 3 complete! 🏆"
fin_body: "Nice work — you've completed <b>Attention Scores &amp; Softmax</b>. You just wrote the heart of the transformer: score every pair, split a 100% attention <b>budget</b> with softmax, then blend the Values by that budget — <code>softmax(Q@Kᵀ/√d_k)@V</code>.<br>Next up: <b>Multi-Head Attention</b> — running several of these meetings at once."
notebook_yardstick: null
coverage_topics:
  - {topic: attention score one number, keywords: [attention score, one number, how much]}
  - {topic: query and key roles, keywords: [query, key, looking for, offers]}
  - {topic: dot product similarity, keywords: [dot product, slot by slot, bigger dot]}
  - {topic: score matrix every pair, keywords: [score matrix, every query, one row per]}
  - {topic: scaling by sqrt d_k, keywords: [sqrt, scale, d_k, divide]}
  - {topic: softmax weights sum to one, keywords: [softmax, sum to 1, positive]}
  - {topic: additive bahdanau attention, keywords: [additive, Bahdanau, small neural network]}
  - {topic: unscaled scores blow up, keywords: [scores swell, clips, gradients shrink, too loud]}
  - {topic: masking future and padding, keywords: [mask, negative infinity, padding, future]}
  - {topic: raw score is not a weight, keywords: [raw score, means nothing, until softmax]}
  - {topic: dot product has no position, keywords: [no word order, order-blind, word order]}
---

@@@ hero
@lede You know that moment at a busy dinner table when three people talk at once, and you quietly decide who to listen to? You lean toward the friend with the good story, give a little to the person beside you, almost none to the far end — and, without noticing, your listening always adds up to one whole you. That everyday balancing act — handing out shares of one fixed pot of attention — is exactly the move at the centre of every transformer. It is what lets ChatGPT work out which words in your sentence belong together, and it is also why your phone's keyboard can guess your next word, and how a translation app knows which foreign word to line up with which English one. Today you build it by hand: for one word, score how well it matches every other word, split a single 100% attention **budget** among them, then spend that budget. By the end you'll have written the one line at the heart of the transformer — and it'll feel as familiar as choosing who to listen to at dinner.
@goal Together we'll turn "how well do these two words match?" into a single number with a friendly multiply-and-add, lay every pair's number into a tidy grid, keep those numbers from getting too loud as words grow long, split one attention **budget** into shares with softmax, and finally spend that budget to blend what the words carry. We'll follow ONE tiny sentence — "the river bank" — from first multiply to final answer, so every number on this page is one you can check yourself. You'll also meet the clumsy older way of scoring (and see why the cheap way won), two lovely puzzles that raw scores hand us — each with a one-move fix — and the honest limits of a score. Every new word gets said in plain English the moment it shows up. No symbol left unexplained.

%%% warmup
q: Yesterday each word split into THREE versions of itself. What were those three roles called? | a:2 | past, present, future | noun, verb, adjective | Query, Key, Value | red, green, blue | concept: query-key-value | fb: A word plays three roles at once — Query (what it asks), Key (what it advertises), Value (what it shares). Today the Query and Key finally meet to make a score.
q: Yesterday a word made its three versions by passing its one embedding through three learned filter-grids. What were those grids called? | a:1 | Q, K, V | W_Q, W_K, W_V | d_k and √d_k | softmax | concept: projection-filters | fb: The three learned filters are W_Q, W_K, W_V — one per role. Today we use the Query and Key they produced.
%%%

@@@ concept id=c1 tag="One little number" title="A score is just 'how much should I listen to you?'" gotit="Got the score idea"
Here is the loveliest thing about today: the whole day hangs on one small idea, and you have been doing it your entire life without a name for it.

When a word tries to understand itself, it does not listen to every other word equally. It hands each neighbour a number — a little rating that says "how much should I listen to *you*?" That rating is called an [[attention score||One number that says how much a word should listen to another word. Bigger = listen more.]]. One number, for one pair of words. Bigger number, listen more.

Think of a **dinner table** where three people are talking at once. You never split your ears evenly.

In your head, without even noticing, you rate each speaker: the story about the trip is gripping (an 8), the weather chat is meh (a 2), the mumbling at the far end barely registers (a 1). Those ratings are your attention scores — and a word does exactly this to the words around it.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A dinner table scene. One listener in the middle rates three speakers with little score numbers: the storyteller gets an 8, the neighbor gets a 2, the far-end mumbler gets a 1. The listener leans toward the highest score."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">You rate each speaker: how much do I listen to you?</text><ellipse cx="260" cy="118" rx="120" ry="52" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="260" y="122" text-anchor="middle" fill="#8A6D3B" font-size="10">the table</text><circle cx="260" cy="70" r="16" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="260" y="74" text-anchor="middle" fill="#5E5191" font-size="10">you</text><circle cx="120" cy="118" r="16" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="120" y="122" text-anchor="middle" fill="#276b45" font-size="9">story</text><circle cx="400" cy="118" r="16" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="400" y="122" text-anchor="middle" fill="#9A938A" font-size="9">weather</text><circle cx="260" cy="176" r="15" fill="#FDF9F3" stroke="#B8AEA2"/><text x="260" y="180" text-anchor="middle" fill="#9A938A" font-size="8">far end</text><line x1="246" y1="80" x2="134" y2="110" stroke="#2D8B55" stroke-width="3"/><text x="188" y="86" text-anchor="middle" fill="#276b45" font-weight="bold">8</text><line x1="276" y1="78" x2="386" y2="110" stroke="#C99A12" stroke-width="1.5"/><text x="336" y="86" text-anchor="middle" fill="#8A6D3B">2</text><line x1="260" y1="86" x2="260" y2="161" stroke="#B8AEA2" stroke-dasharray="3,2"/><text x="272" y="130" fill="#9A938A">1</text></g></svg>
%%%

**What the dinner-table picture gets right:** you hand out one rating per speaker, and a bigger rating means you lean in more — that is exactly an attention score.

**Where it breaks down:** at a real table you can only really follow one voice at a time, but a word scores *every* other word at once and can blend many of them together.

#### First — the charmingly simple thing people tried before scoring
Before scoring existed, the trick was as simple as it sounds: just add all the word vectors together in perfectly equal shares. Engineers call that [[equal-share averaging||Also called average pooling: mix every word into the answer with exactly the same weight, so no word can matter more than another.]] — average pooling. It is one line of code, it needs nothing learned, and it is a genuinely reasonable first guess.

%%% insight
And here is the crack in it — the crack the rest of the day pours through. Take "we sat on the bank and watched the river go by." With equal shares, "bank" mixes in the same flat recipe every time: an equal spoonful of "we", of "sat", of "the", of "river". The recipe is fixed, so "bank" can *never* lean harder on "river" than on "the" — instead of "bank, the river kind" the model gets an even mush of the whole sentence. Scoring is what breaks that tie, and lets one word turn up the volume on the neighbour that actually decides its meaning. It is also why your phone's keyboard can finish "see you at the…" sensibly: something in there is leaning on the words that matter and ignoring the filler.
%%%

#### Now — the two sides of every score
A score always compares two things: what one word is *looking for*, and what another word *offers*. Day 2 already gave those two things names, so let's line them up.

%%% steps
step: the asking side — a word's [[Query||The "what am I looking for?" side of a word — the question it brings to the comparison.]]
why: it carries the question "what am I looking for right now?" — like you at the table, hunting for something interesting
step: the offering side — another word's [[Key||The "what do I offer?" side of a word — the label it advertises to be matched against.]]
why: it advertises "here's what I've got" — like a speaker announcing their topic
step: the score = how well this Query matches that Key
why: therefore one asks, one offers, and the score simply rates the fit. That's the whole idea
%%%

So far, one sentence holds it all: **a score rates one word's Query against another word's Key.** Everything left today is how we turn that fit into a number, keep the number tidy, split attention by it, and finally spend it.

!!! c-info 🗺️
<b>The one line to hold all day:</b> score every Query against every Key, split one attention <b>budget</b> among the words by those scores, then blend what they carry. Hold that sentence; the rest is slow motion.
!!!

@@@ concept id=c2 tag="The match number" title="The dot product — one number for 'do we point the same way?'" gotit="Got the dot product"
So a word needs a number for "how well does my Query match your Key?" Here comes the happy surprise of the day: the recipe is tiny, it is free, and you already do something like it in your head.

It is called the [[dot product||Multiply two lists of numbers slot by slot, then add it all up. One number that says how much the two lists point the same way.]] — multiply the two lists slot by slot, then add up the results. One number falls out, and that number *is* the score.

Think of **two friends filling in the same taste survey**, each rating pizza, hiking and horror movies from 0 to 5.

To see how alike their tastes are you'd go item by item: multiply their two pizza scores, then their two hiking scores, then their two movie scores — and add those up. Both love the same things? The matching high numbers multiply into a big total.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two taste surveys side by side. Friend A rates pizza 5, hiking 1, horror 0. Friend B rates pizza 4, hiking 2, horror 1. Arrows show multiplying each matching pair — 5 times 4, 1 times 2, 0 times 1 — and adding the products into one big match number, 22."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">Multiply matching items, add them up → one match number</text><rect x="24" y="40" width="150" height="118" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="99" y="60" text-anchor="middle" fill="#5E5191" font-weight="bold">Query (friend A)</text><text x="40" y="84" fill="#3A342E">🍕 pizza .... 5</text><text x="40" y="106" fill="#3A342E">🥾 hiking ... 1</text><text x="40" y="128" fill="#3A342E">🎬 horror ... 0</text><rect x="346" y="40" width="150" height="118" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="421" y="60" text-anchor="middle" fill="#8A6D3B" font-weight="bold">Key (friend B)</text><text x="362" y="84" fill="#3A342E">🍕 pizza .... 4</text><text x="362" y="106" fill="#3A342E">🥾 hiking ... 2</text><text x="362" y="128" fill="#3A342E">🎬 horror ... 1</text><text x="260" y="82" text-anchor="middle" fill="#276b45" font-size="10">5 × 4 = 20</text><text x="260" y="104" text-anchor="middle" fill="#C99A12" font-size="10">1 × 2 = 2</text><text x="260" y="126" text-anchor="middle" fill="#9A938A" font-size="10">0 × 1 = 0</text><line x1="174" y1="82" x2="346" y2="82" stroke="#B8AEA2" stroke-dasharray="2,2"/><line x1="174" y1="104" x2="346" y2="104" stroke="#B8AEA2" stroke-dasharray="2,2"/><line x1="174" y1="126" x2="346" y2="126" stroke="#B8AEA2" stroke-dasharray="2,2"/><rect x="180" y="168" width="160" height="30" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="260" y="188" text-anchor="middle" fill="#276b45" font-weight="bold">add them: 20 + 2 + 0 = 22</text></g></svg>
%%%

**What the taste-survey picture gets right:** matching high numbers multiply into a big total, so a big dot product really does mean "you two are alike."

**Where it breaks down:** a survey row has a label like "pizza," but a Query and Key are raw numbers with no labels — the model *learned* what each slot means during training.

#### The recipe in three little moves
Let's stay with the survey and just walk the arithmetic, one move at a time.

%%% steps
step: line the two lists up, slot 1 with slot 1, slot 2 with slot 2, …
why: pizza must meet pizza, never pizza meeting hiking — the slots are what make the comparison fair
step: multiply each matching pair
why: two big numbers in the same slot make a big product; a big and a small make a small one. That's the "do we agree here?" test, per slot
step: add all the products into one total — that total IS the score
why: therefore agreement in many slots piles up into a big score, and disagreement never gets to pile up at all
%%%

%%% formula
expr: score = Q · K = (q₁·k₁) + (q₂·k₂) + … + (q_d·k_d)
note: Q = one word's Query list, K = another word's Key list. The dot "·" means multiply slot-by-slot and add. d = how many slots each list has (we'll use d = 4 all day). One number comes out — the match score.
%%%

So far: three moves, one number. Now — what does that number actually *mean*?

%%% steps
step: the two lists point the same way → a big positive score
why: which means "your Key is exactly the kind of thing my Query was hunting for" — lean in
step: the two lists have nothing in common → a score near zero
why: no slot lights up in both lists, so nothing accumulates. Polite indifference
step: the two lists point opposite ways → a negative score
why: that's an active mismatch, not just a shrug — the score can go below zero, which softmax will handle later
%%%

%%% insight
Notice how *cheap* that was: a handful of multiplies and one add. No extra learned weights, nothing clever. That cheapness is exactly why attention scales up to models the size of ChatGPT — a chip can do a colossal number of these multiply-and-adds per second, and later today you'll meet the slower ancestor it replaced.
%%%

#### Play with it — drag "bank" and watch who it listens to
Meet our sentence for the day: **"the river bank."** The word **bank** is asking, and three Keys are on offer — the `[0,0,1,1]`, river `[2,0,1,1]`, bank `[0,2,1,1]`. Drag bank's four Query slots and watch all three scores redraw live.

Two things to try. First, make **"the"** the winner — the trick is that "the" only wins when slot 1 **and** slot 2 are *both* below zero, because those are the only slots river and bank use. Then push river's score all the way negative and see what an active mismatch looks like; the caption under the sliders tells you exactly how far to drag from wherever you are.

%%% svg
<svg id="d3dot-svg" viewBox="0 0 520 226" role="img" aria-label="Interactive dot product. Drag the four slots of bank's Query and watch three match scores update live as bars: the score against the Key for the, for river and for bank. Bars grow right from a zero line for positive scores and left for negative ones, with the multiply-and-add arithmetic written beside each."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Drag bank's Query — watch its three match scores redraw</text><text x="20" y="38" fill="#5E5191" font-size="10">Query (bank) =</text><text id="d3dot-q" x="118" y="38" fill="#3A342E" font-size="11">[2, 1, 1, 1]</text><line x1="210" y1="46" x2="210" y2="176" stroke="#B8AEA2"/><text x="210" y="190" text-anchor="middle" fill="#9A938A" font-size="8">0</text><text x="18" y="64" fill="#8A6D3B" font-size="9">Key the</text><text x="18" y="76" fill="#9A938A" font-size="8">[0,0,1,1]</text><rect id="d3dot-b0" x="210" y="52" width="22" height="18" fill="#B8AEA2"/><text id="d3dot-v0" x="440" y="66" fill="#6B645E" font-size="12" font-weight="bold">2</text><text id="d3dot-a0" x="240" y="84" fill="#9A938A" font-size="8">0+0+1+1 = 2</text><text x="18" y="106" fill="#8A6D3B" font-size="9">Key river</text><text x="18" y="118" fill="#9A938A" font-size="8">[2,0,1,1]</text><rect id="d3dot-b1" x="210" y="94" width="66" height="18" fill="#2D8B55"/><text id="d3dot-v1" x="440" y="108" fill="#276b45" font-size="12" font-weight="bold">6</text><text id="d3dot-a1" x="284" y="126" fill="#276b45" font-size="8">4+0+1+1 = 6</text><text x="18" y="148" fill="#8A6D3B" font-size="9">Key bank</text><text x="18" y="160" fill="#9A938A" font-size="8">[0,2,1,1]</text><rect id="d3dot-b2" x="210" y="136" width="44" height="18" fill="#8Cc4a3"/><text id="d3dot-v2" x="440" y="150" fill="#276b45" font-size="12" font-weight="bold">4</text><text id="d3dot-a2" x="262" y="168" fill="#276b45" font-size="8">0+2+1+1 = 4</text><rect x="60" y="198" width="400" height="24" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text id="d3dot-win" x="260" y="214" text-anchor="middle" fill="#276b45" font-size="10">river wins (6) — bank leans toward "river"</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center">
<label>slot 1 <input id="d3dot-s0" type="range" min="-3" max="5" step="1" value="2" style="accent-color:#5E5191;vertical-align:middle"></label>
<label>slot 2 <input id="d3dot-s1" type="range" min="-3" max="5" step="1" value="1" style="accent-color:#5E5191;vertical-align:middle"></label>
<label>slot 3 <input id="d3dot-s2" type="range" min="-3" max="5" step="1" value="1" style="accent-color:#5E5191;vertical-align:middle"></label>
<label>slot 4 <input id="d3dot-s3" type="range" min="-3" max="5" step="1" value="1" style="accent-color:#5E5191;vertical-align:middle"></label>
</div>
<div id="d3dot-out" style="margin-top:6px;color:#2C2A28">Query <b>[2, 1, 1, 1]</b> → the <b>2</b>, river <b>6</b>, bank <b>4</b>.</div>
</div>
<script>(function(){
  var s0=document.getElementById('d3dot-s0');if(!s0)return;
  var s1=document.getElementById('d3dot-s1'),s2=document.getElementById('d3dot-s2'),s3=document.getElementById('d3dot-s3');
  var KEYS=[[0,0,1,1],[2,0,1,1],[0,2,1,1]],NAMES=['the','river','bank'];
  var ZERO=210,PX=11;
  function tip(q,sc){
    if(sc[1]<0) return 'river’s score is below zero right now — that is an active mismatch, not just a shrug.';
    var s=q[2]+q[3];
    for(var t=5;t>=-3;t--){ if(2*t+s<0) return 'to push river below zero from here, drag slot 1 down to '+t+' or lower.'; }
    return 'slots 3 and 4 are so high that river cannot go negative yet — pull one of them down first.';
  }
  function paint(){
    var q=[+s0.value,+s1.value,+s2.value,+s3.value];
    document.getElementById('d3dot-q').textContent='['+q.join(', ')+']';
    var sc=KEYS.map(function(k){return q[0]*k[0]+q[1]*k[1]+q[2]*k[2]+q[3]*k[3];});
    for(var i=0;i<3;i++){
      var w=Math.max(2,Math.abs(sc[i])*PX),b=document.getElementById('d3dot-b'+i);
      b.setAttribute('width',w.toFixed(1));
      b.setAttribute('x',(sc[i]<0?ZERO-w:ZERO).toFixed(1));
      b.setAttribute('fill',sc[i]<0?'#C0392B':(i===1?'#2D8B55':(i===2?'#8Cc4a3':'#B8AEA2')));
      document.getElementById('d3dot-v'+i).textContent=sc[i];
      var k=KEYS[i];
      document.getElementById('d3dot-a'+i).textContent=(q[0]*k[0])+'+'+(q[1]*k[1])+'+'+(q[2]*k[2])+'+'+(q[3]*k[3])+' = '+sc[i];
      document.getElementById('d3dot-a'+i).setAttribute('x',(sc[i]<0?ZERO-w-70:ZERO+w+8).toFixed(1));
    }
    var best=0;for(var j=1;j<3;j++){if(sc[j]>sc[best])best=j;}
    var tied=sc.filter(function(v){return v===sc[best];}).length>1;
    document.getElementById('d3dot-win').textContent = tied
      ? 'a tie at '+sc[best]+' — bank has no favourite at all'
      : NAMES[best]+' wins ('+sc[best]+') — bank leans toward "'+NAMES[best]+'"';
    document.getElementById('d3dot-out').innerHTML='Query <b>['+q.join(', ')+']</b> → the <b>'+sc[0]+'</b>, river <b>'+sc[1]+'</b>, bank <b>'+sc[2]+'</b> — '+tip(q,sc);
  }
  [s0,s1,s2,s3].forEach(function(el){el.addEventListener('input',paint);});paint();
})();</script>
%%%

%%% demo id=dot label="score three Keys against one Query"
predict: leave the sliders at [2, 1, 1, 1]. Which Key wins — the, river, or bank? And how far ahead of "the" will the winner be: just barely, or three times over?
code: Q_bank = [2, 1, 1, 1]
code: dot(Q_bank, K_the=[0,0,1,1])    ->  2*0 + 1*0 + 1*1 + 1*1
code: dot(Q_bank, K_river=[2,0,1,1])  ->  2*2 + 1*0 + 1*1 + 1*1
code: dot(Q_bank, K_bank=[0,2,1,1])   ->  2*0 + 1*2 + 1*1 + 1*1
out: the   -> 0 + 0 + 1 + 1 = 2
out: river -> 4 + 0 + 1 + 1 = 6
out: bank  -> 0 + 2 + 1 + 1 = 4
take: <b>river 6, bank 4, the 2 — river wins by three times over.</b> Look at *where* the 6 came from: slot 1, where bank's Query held a 2 and river's Key held a 2, so their product alone was 4. "the" only ever matched on the last two slots, which every Key shares — so "the" got the polite minimum. Same recipe all three times, and the numbers cleanly rank who matters.
%%%

Take a small victory lap: you just turned "how well do these two words match?" into one number you can compute by hand, on paper, with no machine. That number is the atom of attention — and those three numbers `[2, 6, 4]` are going to follow us all the way to the end of the day.

@@@ concept id=c3 tag="Every pair, a grid" title="The score matrix — every word scored against every word" gotit="Got the grid"
One score is nice. But here's a picture you are about to be able to read for the rest of your life — the one that shows up in every attention paper, poster and blog post — and it is just *lots* of the number you already know how to make.

A sentence has many words, and *each* word wants to rate *all* the others — including itself. So we never compute one score. We compute a whole tidy grid of them, called the [[score matrix||A grid holding one score for every pair of words: one row per asking word, one column per offered word.]]: **one row per asking word**, one column per offered word.

Think of a **round-robin sports schedule** pinned to the wall, where every team plays every other team and each little box holds that one matchup.

Reading across the row marked "Lions" tells you how the Lions did against everybody. A score matrix works the same way — one row is one word's opinion of all the others.

%%% svg
<svg viewBox="0 0 520 230" role="img" aria-label="A 3 by 3 score grid for the words the, river, bank. Rows are asking words (queries), columns are offered words (keys). The row for the holds 2, 2, 2. The row for river holds 2, 4, 2. The row for bank holds 2, 6, 4, so bank listens most to river. A note points out the grid is not a mirror: river asking bank is 2 while bank asking river is 6."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Score matrix: row = who's asking, column = who's offered</text><text x="150" y="52" fill="#9A938A" font-size="9">the</text><text x="240" y="52" fill="#9A938A" font-size="9">river</text><text x="330" y="52" fill="#9A938A" font-size="9">bank</text><text x="245" y="40" text-anchor="middle" fill="#8A6D3B" font-size="9">offered words (Keys) →</text><text x="70" y="92" fill="#5E5191" font-size="9">the</text><text x="66" y="132" fill="#5E5191" font-size="9">river</text><text x="66" y="172" fill="#5E5191" font-size="9">bank</text><text x="30" y="126" fill="#5E5191" font-size="9" transform="rotate(-90 30 126)">asking (Q)</text><rect x="130" y="70" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="165" y="91" text-anchor="middle" fill="#9A938A">2</text><rect x="220" y="70" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="255" y="91" text-anchor="middle" fill="#9A938A">2</text><rect x="310" y="70" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="345" y="91" text-anchor="middle" fill="#9A938A">2</text><rect x="130" y="110" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="165" y="131" text-anchor="middle" fill="#9A938A">2</text><rect x="220" y="110" width="70" height="34" fill="#F2F8F4" stroke="#8Cc4a3"/><text x="255" y="131" text-anchor="middle" fill="#276b45">4</text><rect x="310" y="110" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="345" y="131" text-anchor="middle" fill="#9A938A">2</text><rect x="130" y="150" width="70" height="34" fill="#FDF9F3" stroke="#B8AEA2"/><text x="165" y="171" text-anchor="middle" fill="#9A938A">2</text><rect x="220" y="150" width="70" height="34" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="255" y="171" text-anchor="middle" fill="#276b45" font-weight="bold">6</text><rect x="310" y="150" width="70" height="34" fill="#F2F8F4" stroke="#8Cc4a3"/><text x="345" y="171" text-anchor="middle" fill="#276b45">4</text><text x="255" y="204" text-anchor="middle" fill="#276b45" font-size="10">bank's row [2, 6, 4]: it listens most to "river"</text><text x="255" y="222" text-anchor="middle" fill="#8A6D3B" font-size="9">not a mirror: river→bank is 2, bank→river is 6</text></g></svg>
%%%

**What the sports-grid picture gets right:** every player meets every other player, and each box is one matchup — so the whole grid captures all pairings at once.

**Where it breaks down:** in a schedule a team never plays itself, but in attention a word *does* score itself (the diagonal cell) — see how "river" gives itself a 4, because a word often finds its own meaning useful.

#### How to read a row — the skill you'll use forever
Stay with the wall schedule; reading the grid takes just a few moves.

%%% steps
step: pick a ROW — that's the word doing the asking
why: row 3 is "bank" wondering what to pay attention to; the row is one word's whole opinion
step: walk ACROSS the columns — each column is a word being offered
why: which means cell (row 3, column 2) answers exactly one question: how much should "bank" listen to "river"? That cell is the 6 you computed by hand a moment ago
step: find the brightest / biggest cell in the row
why: that's the winner — the word this one leans toward. In the picture, "bank" lights up on "river", not on "the"
step: a dark, blank cell means that pair was switched off
why: later today you'll switch cells off on purpose (masking), and you'll see exactly what that looks like drawn on the grid
%%%

%%% insight
Look at the top row for a second: "the" gives everybody a 2. A flat row like that is a word with no opinion — its attention will end up spread evenly, which is just an expensive average, exactly the equal-share averaging we left behind in unit 1. Freshly built models start out with *every* row looking this flat; learning is what makes rows spiky. Also notice the grid is not a mirror: river→bank is 2 while bank→river is 6, because each side brings a different list to the meeting.
%%%

So far: the grid is just every dot product from the last unit, parked in a tidy square. Let's fill one for real.

%%% demo id=grid label="fill the whole 3×3 grid in one move"
predict: you already know bank's row is [2, 6, 4]. Before revealing: what do you think "the" does across its whole row — pick a favourite, or shrug at everyone?
code: Q = [[0,0,1,1], [1,0,1,1], [2,1,1,1]]   # the, river, bank  (asking side)
code: K = [[0,0,1,1], [2,0,1,1], [0,2,1,1]]   # the, river, bank  (offering side)
code: Q @ K.T
out: [[2, 2, 2],     <- "the"   shrugs: every word gets a 2
out:  [2, 4, 2],     <- "river" likes itself most (4)
out:  [2, 6, 4]]     <- "bank"  leans hard on "river" (6)
take: <b>Nine scores, one line of code.</b> Every cell is a dot product you could do by hand — and the grid is exactly the picture above. "the" really does shrug at everyone (a flat row), while "bank" picks a clear winner. That difference between a flat row and a spiky row is what attention is <i>for</i>.
%%%

%%% insight
Here's the part nobody tells beginners: this grid is why long context costs so much. It has one cell per PAIR of words, so a sentence twice as long needs a grid FOUR times as big. Double the words, quadruple the grid — that single fact is behind most of the price tag on long-context models.
%%%

#### One tidy move fills the entire grid
You *could* fill the grid one dot product at a time, by hand. But there's a single step that does all of them at once, and it's the reason chips love attention.

%%% formula
expr: Scores = Q @ Kᵀ        (shape: n_words × n_words)
note: Q = all the Queries stacked as rows. Kᵀ = all the Keys turned to stand as columns. "@" = do every row-meets-column dot product in one step. Out comes the n×n grid — one score per pair.
%%%

Here is that one move drawn as a shape story — a tall stack of Queries meets a wide row of Keys, and out pops a perfect square.

%%% svg
<svg viewBox="0 0 520 170" role="img" aria-label="Shape transformation for the score matrix. A tall n by d block of Queries times a wide d by n block of Keys transposed gives a square n by n grid of scores. Boxes labelled with their shapes and an equals sign between them."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Queries (n×d) meet Keys (d×n) → a square n×n grid of scores</text><rect x="30" y="50" width="44" height="90" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="52" y="98" text-anchor="middle" fill="#5E5191" font-size="10">Q</text><text x="52" y="156" text-anchor="middle" fill="#5E5191" font-size="9">n × d</text><text x="95" y="100" text-anchor="middle" fill="#8A6D3B" font-size="16">@</text><rect x="118" y="72" width="120" height="46" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="178" y="100" text-anchor="middle" fill="#8A6D3B" font-size="10">Kᵀ</text><text x="178" y="134" text-anchor="middle" fill="#8A6D3B" font-size="9">d × n</text><text x="270" y="100" text-anchor="middle" fill="#8A6D3B" font-size="16">=</text><rect x="310" y="50" width="90" height="90" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><line x1="340" y1="50" x2="340" y2="140" stroke="#8Cc4a3"/><line x1="370" y1="50" x2="370" y2="140" stroke="#8Cc4a3"/><line x1="310" y1="80" x2="400" y2="80" stroke="#8Cc4a3"/><line x1="310" y1="110" x2="400" y2="110" stroke="#8Cc4a3"/><text x="355" y="46" text-anchor="middle" fill="#276b45" font-size="9">scores</text><text x="355" y="156" text-anchor="middle" fill="#276b45" font-size="9">n × n (a square!)</text><text x="460" y="90" text-anchor="middle" fill="#9A938A" font-size="9">one cell</text><text x="460" y="104" text-anchor="middle" fill="#9A938A" font-size="9">per pair</text></g></svg>
%%%

%%% steps
step: stack all n Queries as rows → a tall block, n rows by d slots
why: each row is one word's question, still just the list of numbers from before
step: stand all n Keys up as columns → a wide block, d slots by n words
why: turning them sideways is what makes "every row meet every column" possible in one move
step: multiply → every Query row meets every Key column, all at once
why: that's n × n dot products done in a single step instead of one at a time. This is the move chips are built for
step: the d in the middle disappears; an n × n square is left
why: therefore d gets "used up" by the multiply-and-add, and what survives is exactly one score per pair
%%%

!!! c-info 💡
<b>Optional (skippable) — the shape story in symbols.</b> If Q has shape n×d (n words, each a d-slot Query) and Kᵀ is d×n (each Key standing as a column), multiplying them gives n×n — a perfect square, one row and one column per word. Add a batch of sentences and it becomes (batch, n, n). If you ever forget the transpose you get a shape error instead of a grid — that's the machine being kind to you. You never have to track this to use attention; it's here for the curious.
!!!

Victory lap: you can now look at any attention heat map in any paper and say out loud what a single bright square means. That is a real skill, and you got it in one unit.

@@@ concept id=c4 tag="Turn it down" title="Scaling — keep the scores from getting too loud" gotit="Got the scaling"
Now for my favourite one-liner in all of deep learning. It is a single division, it costs nothing, and every transformer ever shipped has it — including the one answering your questions in a chat window. Let's earn it.

Here's the thing about our sums: the dot product adds one product per slot. Our lists have 4 slots, so four products get added and the totals stay gentle. Real Query and Key lists are *long* — often hundreds of slots — so the products pile up and the totals swell into big, shouty numbers.

Think of **a choir with one volume knob**. One singer is pleasant. Add sixty-three more all singing at once and the room becomes painfully loud.

Nobody sang louder — there are simply *more* voices stacking up. And the gentle cure isn't to silence anyone: you turn the room's volume knob down by an amount that matches the size of the choir.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A choir with a volume knob. On the left one singer makes a short loudness bar of height 14. In the middle a choir of 64 singers makes a bar 8 times taller, height 112, marked too loud. A volume knob labelled divide by the square root of d_k, which is 8, turns it down. On the right the same 64 singers give a bar back at the original short height."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">More slots → louder score. Divide by √d_k → back to the 1-slot size.</text><line x1="24" y1="170" x2="496" y2="170" stroke="#B8AEA2"/><line x1="24" y1="156" x2="496" y2="156" stroke="#8Cc4a3" stroke-dasharray="3,3"/><text x="404" y="150" fill="#8Cc4a3" font-size="8">the gentle line</text><circle cx="55" cy="44" r="7" fill="#EAF0FA" stroke="#5E5191"/><text x="55" y="62" text-anchor="middle" fill="#5E5191" font-size="8">1 singer</text><rect x="30" y="156" width="50" height="14" fill="#EAF5EE" stroke="#2D8B55"/><text x="55" y="186" text-anchor="middle" fill="#276b45" font-size="9">1 slot</text><text x="55" y="150" text-anchor="middle" fill="#276b45" font-size="8">gentle</text><circle cx="205" cy="40" r="6" fill="#FBE3E3" stroke="#C0392B"/><circle cx="222" cy="40" r="6" fill="#FBE3E3" stroke="#C0392B"/><circle cx="239" cy="40" r="6" fill="#FBE3E3" stroke="#C0392B"/><circle cx="213" cy="52" r="6" fill="#FBE3E3" stroke="#C0392B"/><circle cx="230" cy="52" r="6" fill="#FBE3E3" stroke="#C0392B"/><text x="222" y="32" text-anchor="middle" fill="#C0392B" font-size="8">64 singers</text><rect x="200" y="58" width="50" height="112" fill="#FBE3E3" stroke="#C0392B" stroke-width="2"/><text x="225" y="186" text-anchor="middle" fill="#C0392B" font-size="9">64 slots</text><text x="225" y="72" text-anchor="middle" fill="#C0392B" font-size="8">too loud!</text><text x="225" y="86" text-anchor="middle" fill="#C0392B" font-size="8">(8× taller)</text><circle cx="330" cy="100" r="24" fill="#FCF3DC" stroke="#C99A12" stroke-width="2"/><line x1="330" y1="100" x2="316" y2="82" stroke="#8A6D3B" stroke-width="2"/><text x="330" y="140" text-anchor="middle" fill="#8A6D3B" font-size="12">÷ √d_k</text><text x="330" y="154" text-anchor="middle" fill="#8A6D3B" font-size="9">(÷ 8)</text><path d="M362 112 L424 112" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#ar4)"/><defs><marker id="ar4" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#8A6D3B"/></marker></defs><rect x="440" y="156" width="50" height="14" fill="#EAF5EE" stroke="#2D8B55"/><text x="465" y="186" text-anchor="middle" fill="#276b45" font-size="9">64 slots</text><text x="465" y="150" text-anchor="middle" fill="#276b45" font-size="8">gentle again</text></g></svg>
%%%

**What the choir picture gets right:** the loudness grows just from adding more voices, and the cure is one knob turned down by an amount that matches the crowd — exactly what our fix does.

**Where it breaks down:** singers get literally louder in your ears, but scores make no sound; "too loud" here means "too big a number," which quietly changes the *next* step instead of hurting anyone.

#### The swell, and the knob
That volume knob has a name: we divide every score by [[√d_k||The square root of d_k, the number of slots in each Key/Query list. Dividing scores by it cancels the growth that comes from having more slots.]], where `d_k` is simply how many slots each Key and Query has. Let's watch the swell happen, then turn it down.

%%% steps
step: the swell — 1 slot means 1 product; 64 slots means 64 products added up
why: nothing got more important, there are just more voices in the choir, so the total climbs
step: the size of the swell — with 64 slots, typical scores come out about 8× bigger than with a single slot
why: because 8 is the square root of 64 — that's the pattern the totals actually follow as slots multiply
step: the knob — divide every score by √d_k, so with 64 slots divide by 8
why: therefore we cancel exactly the growth the slot count caused, and land back at the gentle 1-slot size
step: this is the famous "scaled dot-product attention"
why: that single division is the default in every transformer ever shipped. One character of arithmetic, enormous payoff
%%%

%%% insight
Why does loudness even matter? Because of what comes next. Feed huge scores into the sharing step and it slams the entire attention <b>budget</b> onto one word — and here's the sneaky part: the model keeps running perfectly happily, the loss just quietly stops improving. Nothing crashes. That's exactly why this one division is worth its own unit, and we'll catch it in the act later today.
%%%

So far: more slots → louder scores → divide by √d_k. Our lists have 4 slots, so our knob is √4 = 2 — let's turn it and watch what does (and does not) change.

%%% demo id=scale label="bank's row before and after dividing by √d_k"
predict: bank's raw row is [2, 6, 4] and we're about to divide it all by 2. The numbers shrink — but does the WINNER change? Guess yes or no, then reveal.
code: d_k = 4  ->  √d_k = 2
code: raw    = [2, 6, 4]        # bank vs the, river, bank
code: scaled = [2/2, 6/2, 4/2]
out: raw    -> [2, 6, 4]     river ahead of bank ahead of the
out: scaled -> [1, 3, 2]     river ahead of bank ahead of the   (same order!)
take: <b>The order never changed.</b> 3 is still bigger than 2 is still bigger than 1, so river still wins — dividing every score by the SAME number shrinks the loudness and leaves the ranking untouched. That's the whole trick: tame the size, keep the winner. With a real d_k of 64 the knob would be 8 instead of 2, and it would do the same quiet favour. Hold onto <b>[1, 3, 2]</b> — that's the row we spend in the next unit.
%%%

!!! c-ok ✅
<b>You now own the "/√d_k" in the famous formula.</b> One division, done before the sharing step, that keeps scores in a range the next step can actually work with. It looks like a detail; it is the difference between a transformer that learns and one that just hums.
!!!

@@@ concept id=c5 tag="Split the budget" title="Softmax — turn scores into shares that add up to one" gotit="Got softmax"
Here comes the star of the day — the step that decides where a model *looks*. A scaled score like 3 still doesn't tell a word *how much* of its attention to spend, and that is the gap this unit closes.

We want to hand attention out like slices of one pizza: every slice positive, and all the slices together making exactly one whole pizza — no more, no less. The step that does this is [[softmax||A step that turns a row of scores into positive shares that sum to 1 — the biggest score gets the biggest share.]].

Think of **splitting one pizza among friends by how hungry they are**.

The hungriest friend gets the biggest slice and the least hungry gets a sliver — but you only have *one* pizza, so all the slices still add up to exactly one whole. Nobody gets a negative slice. That one whole pizza is the attention **budget**.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="On the left, bank's three scaled scores as bars: the 1, river 3, bank 2. On the right, a pizza split into three slices by softmax: river gets a big 0.67 slice, bank a 0.24 slice, the a small 0.09 slice, and the three slices add up to one whole pizza."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Scores → one pizza of attention, split into shares that sum to 1</text><text x="90" y="40" text-anchor="middle" fill="#8A6D3B" font-size="10">bank's scaled row</text><rect x="40" y="60" width="30" height="22" fill="#FDF9F3" stroke="#B8AEA2"/><text x="80" y="76" fill="#9A938A" font-size="9">the 1</text><rect x="40" y="92" width="90" height="22" fill="#EAF5EE" stroke="#2D8B55"/><text x="140" y="108" fill="#276b45" font-size="9">river 3</text><rect x="40" y="124" width="60" height="22" fill="#FCF3DC" stroke="#C99A12"/><text x="110" y="140" fill="#8A6D3B" font-size="9">bank 2</text><text x="235" y="110" fill="#5E5191" font-size="16">softmax →</text><circle cx="410" cy="120" r="70" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><path d="M410 120 L410 50 A70 70 0 1 1 348.7 153.7 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><path d="M410 120 L348.7 153.7 A70 70 0 0 1 372.5 60.9 Z" fill="#FDF7EC" stroke="#C99A12" stroke-width="1.5"/><path d="M410 120 L372.5 60.9 A70 70 0 0 1 410 50 Z" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="438" y="144" fill="#276b45" font-size="9" font-weight="bold">river 0.67</text><text x="352" y="113" fill="#8A6D3B" font-size="9">bank 0.24</text><text x="384" y="80" fill="#9A938A" font-size="8">the 0.09</text><text x="410" y="205" text-anchor="middle" fill="#276b45" font-size="9">slices add to 1 whole pizza (0.09 + 0.67 + 0.24 = 1.00)</text></g></svg>
%%%

**What the pizza picture gets right:** shares are all positive and always add to exactly one whole, and the hungriest friend (highest score) gets the biggest slice.

**Where it breaks down:** a pizza gets shared straight in proportion to hunger, but softmax reacts to the *gaps* between scores instead — a wide gap makes it far peakier than plain proportions would be, and a narrow gap makes it flatter. Same scores, different spacing, very different pizza.

#### Three moves, and the pizza is sliced
Let's cut the pizza slowly, using bank's scaled row `[1, 3, 2]` from the last unit. Softmax does the same three things to every score in a row.

%%% steps
step: the stretch — grow each score into a positive amount of "hunger"
why: this does two jobs at once: negative scores become small positive amounts, and a leading score gets stretched ahead of the pack
step: the pot — add all those grown amounts into one total
why: that total is the whole pizza. It exists so we know what "one whole" is worth before we start slicing
step: the share — divide each grown amount by the pot
why: therefore every score becomes a fraction of one pizza, and because we divided by the total, the shares sum to 1 automatically
%%%

%%% insight
This is the knob behind where a model "looks". A row of nearly equal shares is a word that has not made up its mind — attention spread thin across everything (engineers measure that spread with one number called attention <b>entropy</b>: high entropy = vague, low entropy = locked on). A row with one fat share is a word that has locked on. Same machinery, both times; only the scores differ. When a chatbot answers your long question and clearly homes in on the one keyword that mattered, or a translation app lines "cat" up with "chat", this is the step that did the homing.
%%%

So far: stretch, pot, share. Let's put our row through it and check the numbers by eye.

%%% demo id=softmax label="turn bank's row into three shares"
predict: the row is [1, 3, 2]. River's score is 3 and bank's is 2 — a gap of just 1. Will river's share be a little bigger than bank's, or a LOT bigger? Guess before revealing.
code: row = [1, 3, 2]              # the, river, bank  (already scaled)
code: stretch -> [2.72, 20.09, 7.39]
code: pot     -> 2.72 + 20.09 + 7.39 = 30.19
code: shares  -> each stretched value / 30.19
out: the   -> 2.72  / 30.19 = 0.0900
out: river -> 20.09 / 30.19 = 0.6652
out: bank  -> 7.39  / 30.19 = 0.2447
out: check -> 0.0900 + 0.6652 + 0.2447 = 1.0000
take: <b>river takes two-thirds of the budget from a gap of only 1.</b> That's the stretch doing its work: a small lead in score becomes a big lead in share. And look at the last line — the three shares add to exactly 1, one whole pizza, every single time. Rounded to two places they are <b>0.09 / 0.67 / 0.24</b>, and those are what we spend next.
%%%

Now here are the four promises this step always keeps — worth memorising, because every debugging session leans on them.

%%% steps
step: every share is positive
why: the stretch makes everything positive first, so no word can ever get negative attention. There's no such thing as "un-listening"
step: each row's shares sum to 1
why: that's the fixed <b>budget</b>. One whole pizza per asking word, always — which makes rows comparable and easy to check
step: a bigger score always gets a bigger share
why: which means the ranking you computed with dot products survives intact — softmax re-scales, it never re-orders
step: add the same number to EVERY score in a row and nothing changes
why: only the gaps between scores matter, not how high they sit. This surprising promise turns out to be a lifesaver — see the optional box
%%%

#### Play with it — drag the scores and watch the slices move
Here's the live pizza, starting from our row `[1, 3, 2]`. Drag a slider and watch the shares rebalance. Try pulling river far above the others and watch it swallow almost the whole budget — that is a word "focusing". Then pull all three level and watch attention go vague and spread out.

%%% svg
<svg id="d3sm-svg" viewBox="0 0 520 196" role="img" aria-label="Interactive softmax. Drag three scaled scores — the, river and bank — and watch the attention shares, drawn as bars, resize. They always stay positive and add up to exactly one whole pizza."><g font-family="monospace" font-size="11" text-anchor="middle"><rect x="40" y="26" width="440" height="140" rx="6" fill="#FDF9F3" stroke="#E5DFD6"/><line x1="60" y1="150" x2="460" y2="150" stroke="#EFE9DF"/><rect id="d3sm-b0" x="110" y="140" width="60" height="10" fill="#B8AEA2"/><text id="d3sm-t0" x="140" y="132" fill="#6B645E">.09</text><text x="140" y="164" fill="#6B645E">the</text><rect id="d3sm-b1" x="230" y="75" width="60" height="75" fill="#2D8B55"/><text id="d3sm-t1" x="260" y="67" fill="#1a5c38">.67</text><text x="260" y="164" fill="#6B645E">river</text><rect id="d3sm-b2" x="350" y="123" width="60" height="27" fill="#C99A12"/><text id="d3sm-t2" x="380" y="115" fill="#9A7208">.24</text><text x="380" y="164" fill="#6B645E">bank</text><text x="260" y="184" fill="#1a5c38" font-size="10">the three shares always add up to one whole pizza (1.00)</text></g></svg>
<div style="margin-top:8px;font-family:monospace;font-size:.9em;color:#5A544E">
<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">
<label>the <input id="d3sm-s0" type="range" min="-2" max="6" step="0.1" value="1" style="accent-color:#B8AEA2;vertical-align:middle"></label>
<label>river <input id="d3sm-s1" type="range" min="-2" max="6" step="0.1" value="3" style="accent-color:#2D8B55;vertical-align:middle"></label>
<label>bank <input id="d3sm-s2" type="range" min="-2" max="6" step="0.1" value="2" style="accent-color:#C99A12;vertical-align:middle"></label>
</div>
<div id="d3sm-out" style="margin-top:6px;color:#2C2A28">scores [1.0, 3.0, 2.0] → shares <b>0.09 / 0.67 / 0.24</b> (sum 1.00) — crank one score up and it grabs most of the budget.</div>
</div>
<script>(function(){
  var s0=document.getElementById('d3sm-s0');if(!s0)return;
  var s1=document.getElementById('d3sm-s1'),s2=document.getElementById('d3sm-s2'),out=document.getElementById('d3sm-out');
  var BASE=150,MAXH=112;
  function paint(){
    var v=[+s0.value,+s1.value,+s2.value];
    var ex=v.map(function(x){return Math.exp(x);});var Z=ex[0]+ex[1]+ex[2];
    var sh=ex.map(function(e){return e/Z;});
    for(var i=0;i<3;i++){var h=Math.max(2,sh[i]*MAXH);var b=document.getElementById('d3sm-b'+i),t=document.getElementById('d3sm-t'+i);
      if(b){b.setAttribute('y',(BASE-h).toFixed(1));b.setAttribute('height',h.toFixed(1));}
      if(t){t.setAttribute('y',(BASE-h-8).toFixed(1));t.textContent=sh[i].toFixed(2);}}
    out.innerHTML='scores ['+v.map(function(x){return x.toFixed(1);}).join(', ')+'] → shares <b>'+sh.map(function(x){return x.toFixed(2);}).join(' / ')+'</b> (sum '+(sh[0]+sh[1]+sh[2]).toFixed(2)+') — crank one score up and it grabs most of the budget.';
  }
  [s0,s1,s2].forEach(function(el){el.addEventListener('input',paint);});paint();
})();</script>
%%%

!!! c-info 💡
<b>Optional (skippable) — the exact recipe, and one real-world rescue.</b> The "stretch" is <code>exp(score)</code>: raise a special growth number to the power of the score. The pot is the sum of those, and each share is <code>exp(score) / pot</code>. That growth is fierce: <code>exp</code> of a big score can overflow past the largest number a computer can hold, and then dividing one infinity by another gives you NaN — a broken model. The rescue uses the fourth promise: subtract the row's biggest score from every score first. Nothing changes mathematically, and nothing overflows. Libraries call this the <b>stable softmax</b> (or the log-sum-exp trick), and it's on by default. There's one more dial here too — divide every score by a <b>temperature</b> before softmax: small temperature = decisive, big temperature = wishy-washy.
!!!

The victory lap: you now hold the middle of the whole spine — **score every pair, then split one attention budget by those scores.** One step left to make it useful: spending that budget.

@@@ concept id=c6 tag="Spend it" title="Spending the budget — blend the Values into one answer" gotit="Got the blend"
This is the unit where a word stops being one fixed word and becomes a word *in context* — the thing people are really pointing at when they say a transformer "understands" language. And it's a smoothie.

Shares are lovely, but nobody eats a percentage. "bank" now knows *how much* to listen to each neighbour — so let's actually collect what those neighbours have to say. Remember the third role from Day 2: each word also carries a [[Value||The "what do I hand over?" side of a word — the content it contributes when someone listens to it.]], the content it hands over when someone listens. We mix the Values together in exactly the proportions softmax just handed us.

Think of **making a fruit smoothie**.

Your recipe card says two-thirds mango, a quarter banana, a splash of lime. You pour each fruit in *that* amount, blitz it, and one new drink comes out. The shares are the recipe; the fruits are the Values; the smoothie is the word's new, context-aware meaning.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A smoothie analogy. Three jugs labelled the, river and bank pour into one blender in the amounts 0.09, 0.67 and 0.24, and one glass of blended juice comes out labelled the new meaning of the word. The pouring streams are drawn thin, thick and medium to match the share sizes."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Pour each Value by its share → one blended answer</text><rect x="24" y="36" width="66" height="46" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="57" y="58" text-anchor="middle" fill="#9A938A" font-size="9">the</text><text x="57" y="74" text-anchor="middle" fill="#9A938A" font-size="9">0.09</text><rect x="104" y="36" width="66" height="46" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="137" y="58" text-anchor="middle" fill="#276b45" font-size="9">river</text><text x="137" y="74" text-anchor="middle" fill="#276b45" font-size="9">0.67</text><rect x="184" y="36" width="66" height="46" rx="4" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="217" y="58" text-anchor="middle" fill="#8A6D3B" font-size="9">bank</text><text x="217" y="74" text-anchor="middle" fill="#8A6D3B" font-size="9">0.24</text><path d="M57 82 C57 110 150 110 200 122" stroke="#B8AEA2" stroke-width="1.5" fill="none"/><path d="M137 82 C137 108 180 112 205 124" stroke="#2D8B55" stroke-width="9" fill="none" opacity="0.75"/><path d="M217 82 C217 106 200 114 212 126" stroke="#C99A12" stroke-width="4" fill="none" opacity="0.8"/><rect x="196" y="118" width="96" height="66" rx="6" fill="#F4F1EA" stroke="#5E5191" stroke-width="1.5"/><text x="244" y="146" text-anchor="middle" fill="#5E5191" font-size="9">blender</text><text x="244" y="164" text-anchor="middle" fill="#5E5191" font-size="9">(weighted sum)</text><path d="M292 150 L336 150" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#ar6)"/><defs><marker id="ar6" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#8A6D3B"/></marker></defs><path d="M352 120 L392 120 L386 184 L358 184 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="446" y="140" text-anchor="middle" fill="#276b45" font-size="9">one glass:</text><text x="446" y="156" text-anchor="middle" fill="#276b45" font-size="9">the word's new,</text><text x="446" y="172" text-anchor="middle" fill="#276b45" font-size="9">in-context meaning</text></g></svg>
%%%

**What the smoothie picture gets right:** you pour by proportion, everything goes in at once, and one single drink comes out — that is exactly a weighted sum of Values.

**Where it breaks down:** a smoothie loses the fruits forever, but here every word gets its *own* smoothie at the same time, each with a different recipe, and the original words are still sitting there untouched.

#### The blend, one pour at a time
Let's follow the recipe with tiny Values so you can check every number by eye. Each word's Value is a 2-slot list.

%%% steps
step: the recipe — the shares softmax just gave us: the 0.0900, river 0.6652, bank 0.2447
why: this is the <b>budget</b> from the last unit, and it already adds up to one whole. No pouring more than one glass
step: scale each Value by its share — 0.0900 × [0, 6], 0.6652 × [10, 2], 0.2447 × [4, 4]
why: which means a word we barely listen to contributes barely anything, while river's content comes through loud
step: add the scaled Values slot by slot — that sum IS the output
why: therefore "bank"'s new meaning is literally "a bit of everyone, in proportion to how much I listened". That's the whole output of attention
%%%

%%% formula
expr: out = w₁·V₁ + w₂·V₂ + … + wₙ·Vₙ        (all together: out = softmax(Q@Kᵀ/√d_k) @ V)
note: wᵢ = word i's share from softmax (positive, all shares in a row add to 1). Vᵢ = word i's Value list. The result is one list the same length as a Value — the asking word's new, context-aware meaning.
%%%

%%% insight
This is the moment "bank" stops being one fixed word. Its output is now mostly river-flavoured, so downstream the model is holding "bank, the river kind" — not the money kind. Same input word, different neighbours, different meaning. That is what people mean when they say a transformer understands words <i>in context</i>, and it is exactly why a translation app can pick the right foreign word for "bank".
%%%

So far: shares in, one blended list out. Now predict the actual numbers before you look.

%%% demo id=blend label="blend three Values by their shares"
predict: river's Value is [10, 2] and it holds two-thirds of the budget. Will the first slot of the answer land closer to 10, or closer to 0? Guess a number before revealing.
code: shares = [0.0900, 0.6652, 0.2447]     # the, river, bank  (full precision)
code: V = [[0, 6], [10, 2], [4, 4]]         # the, river, bank
code: out = 0.0900*[0,6] + 0.6652*[10,2] + 0.2447*[4,4]
out: slot 1 -> 0.000 + 6.652 + 0.979 = 7.63
out: slot 2 -> 0.540 + 1.330 + 0.979 = 2.85
out: out = [7.63, 2.85]
take: <b>[7.63, 2.85] — pulled hard toward river, nudged by the others.</b> Because river held two-thirds of the budget, the answer sits near river's [10, 2] but not on top of it. Nobody's Value was thrown away; they were mixed. One note so your own code matches: if you round the shares to 0.09 / 0.67 / 0.24 <i>first</i> and then blend, you get [7.66, 2.84] instead — same answer, just rounding earlier. Full-precision shares give <b>[7.63, 2.85]</b>, and that's the number to expect from a computer.
%%%

#### Where the blend can and cannot land
Now here is a genuinely surprising limit, and it's easiest to just *see* it. Plot the three Values as three dots, and the blend always lands inside the triangle they make.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A two dimensional plot of three Value vectors as dots: the at 0 and 6, bank at 4 and 4, river at 10 and 2. A shaded triangle joins them. The blended output at 7.63 and 2.85 sits inside the triangle, closer to river. Text notes that any blend must land inside the triangle and can never land outside it."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Every possible blend lives INSIDE the triangle of Values</text><line x1="50" y1="170" x2="480" y2="170" stroke="#B8AEA2"/><line x1="50" y1="170" x2="50" y2="34" stroke="#B8AEA2"/><text x="470" y="186" fill="#9A938A" font-size="9">slot 1 →</text><text x="30" y="44" fill="#9A938A" font-size="9">slot 2</text><path d="M50 56 L410 132 L194 94 Z" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="4,3"/><circle cx="410" cy="132" r="5" fill="#2D8B55"/><text x="410" y="150" text-anchor="middle" fill="#276b45" font-size="9">river [10, 2]</text><circle cx="50" cy="56" r="5" fill="#B8AEA2"/><text x="72" y="50" fill="#6B645E" font-size="9">the [0, 6]</text><circle cx="194" cy="94" r="5" fill="#C99A12"/><text x="194" y="84" text-anchor="middle" fill="#8A6D3B" font-size="9">bank [4, 4]</text><circle cx="325" cy="116" r="6" fill="#5E5191"/><text x="329" y="108" fill="#5E5191" font-size="9">out [7.63, 2.85]</text><text x="260" y="196" text-anchor="middle" fill="#C0392B" font-size="9">outside the triangle is unreachable — attention can blend, never invent or subtract</text></g></svg>
%%%

%%% steps
step: the blend is a mix with positive shares that add to 1 — so it is always somewhere *between* the Values
why: which means the dot lands inside the triangle. The closer river's share creeps toward the whole budget, the closer the blend creeps toward river's corner — it can get very close, but never quite arrive
step: attention can never land outside that triangle
why: therefore it cannot invent content none of the words carried, and it cannot subtract or negate a word — no share is allowed to be negative
step: and it never picks *only* one word — every unmasked word keeps a sliver
why: that's why we call it a SOFT lookup: instead of opening one drawer, you get a fraction of every drawer, and the fractions add to one
%%%

Victory lap — a big one. You have now walked the entire line on real numbers: score `[2, 6, 4]`, scale to `[1, 3, 2]`, split the budget into `0.09 / 0.67 / 0.24`, blend to `[7.63, 2.85]`. That is `softmax(Q@Kᵀ/√d_k)@V`, start to finish, done by hand. Everything after this makes it robust and honest.

@@@ concept id=c7 tag="The old way" title="How scoring used to work — and why the cheap way won" gotit="Got the history"
Time for a lovely peek at how this field actually moved forward. The first attention models did not use a dot product at all — they scored matches a clumsier way, and seeing it makes today's version feel like the gift it is.

That older method is called [[additive attention||Bahdanau's 2014 scoring: feed the Query and Key into a small neural network that outputs the match score. It works, but it costs many steps and extra learned weights per pair.]], or *Bahdanau attention*, after the researcher who introduced it in 2014.

Think of **two ways to check whether a key fits a lock**.

The careful way: hire a locksmith who studies the key and the lock and hands you a fitness rating out of ten — you pay them, and you wait. The quick way: push the key in and feel whether it turns. Two different judges, both aimed at the same question — and only one of them costs you an afternoon.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="Two ways to judge whether a key fits a lock. Top: a locksmith with a clipboard studies the key and lock and writes a rating of 8 out of 10 — labelled slow and you pay them. Bottom: a hand pushes the key straight into the lock and it turns — labelled instant and free."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Two judges, same question: does this key fit?</text><rect x="20" y="40" width="26" height="7" fill="#C99A12" stroke="#8A6D3B"/><circle cx="16" cy="43" r="7" fill="none" stroke="#8A6D3B" stroke-width="2"/><rect x="40" y="47" width="5" height="5" fill="#8A6D3B"/><rect x="62" y="30" width="34" height="42" rx="5" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><circle cx="79" cy="46" r="5" fill="#5E5191"/><rect x="77" y="50" width="4" height="10" fill="#5E5191"/><circle cx="168" cy="38" r="10" fill="#FBE9E9" stroke="#C0392B" stroke-width="1.5"/><rect x="158" y="50" width="20" height="26" rx="4" fill="#FBE9E9" stroke="#C0392B" stroke-width="1.5"/><rect x="182" y="52" width="20" height="24" rx="2" fill="#FDF9F3" stroke="#C0392B"/><line x1="186" y1="60" x2="198" y2="60" stroke="#C0392B"/><line x1="186" y1="66" x2="198" y2="66" stroke="#C0392B"/><text x="168" y="92" text-anchor="middle" fill="#C0392B" font-size="9">locksmith</text><path d="M110 50 L150 50" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#ar7a)"/><defs><marker id="ar7a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#8A6D3B"/></marker></defs><path d="M212 60 L266 60" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#ar7a)"/><rect x="274" y="42" width="90" height="34" rx="6" fill="#FBE9E9" stroke="#C0392B" stroke-width="1.5"/><text x="319" y="63" text-anchor="middle" fill="#C0392B" font-size="11">fit = 8 / 10</text><text x="440" y="58" text-anchor="middle" fill="#C0392B" font-size="10">slow 🐌</text><text x="440" y="74" text-anchor="middle" fill="#C0392B" font-size="9">and you pay</text><line x1="20" y1="106" x2="500" y2="106" stroke="#E5DED2"/><rect x="20" y="140" width="26" height="7" fill="#C99A12" stroke="#8A6D3B"/><circle cx="16" cy="143" r="7" fill="none" stroke="#8A6D3B" stroke-width="2"/><rect x="40" y="147" width="5" height="5" fill="#8A6D3B"/><rect x="62" y="130" width="34" height="42" rx="5" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><circle cx="79" cy="146" r="5" fill="#5E5191"/><rect x="77" y="150" width="4" height="10" fill="#5E5191"/><path d="M118 140 q14 -14 28 0 q-14 14 -28 0 Z" fill="#FCF3DC" stroke="#8A6D3B"/><text x="132" y="166" text-anchor="middle" fill="#8A6D3B" font-size="9">a push</text><path d="M110 150 L150 150" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#ar7a)"/><path d="M186 150 L266 150" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#ar7a)"/><rect x="274" y="132" width="90" height="34" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="319" y="153" text-anchor="middle" fill="#276b45" font-size="11">it turned! ✓</text><text x="440" y="148" text-anchor="middle" fill="#276b45" font-size="10">instant ⚡</text><text x="440" y="164" text-anchor="middle" fill="#276b45" font-size="9">and free</text></g></svg>
%%%

**What the locksmith picture gets right:** both judges are aimed at the same question, but one is a slow paid step and the other is instant. That price gap is what made the field switch.

**Where it breaks down:** the two judges are not guaranteed to agree. The locksmith has their own training and their own scale, so "8 out of 10" and "it turned" are different verdicts about the same key — and on a tricky key they could genuinely disagree. Same with scoring: additive attention has its own learned weights, so it is a *different* function of the same Query and Key, not a slower copy of the dot product.

#### The clumsy way, one step at a time
Additive attention is the locksmith. Watch how much work it does just to produce one number.

%%% steps
step: glue the Query and the Key into one long list
why: the little network needs both sides in one input before it can judge the fit
step: push that list through a small neural network with its OWN learned weights
why: which means extra parameters to store, and extra multiplies to run, for every single pair of words
step: bend the result (a squash step), then multiply once more
why: that's a second pass of arithmetic before any answer appears — the locksmith taking their time
step: read out one number — the score
why: therefore a whole little network ran, with its own learned weights, to produce one number. Because those weights are its own, it is its own judge: its numbers live on its own scale, and it can rank a pair differently from the dot product
%%%

%%% insight
Here's why the field cared so much. A big model runs this scoring step an astronomical number of times during training. "Many steps versus one step" per pair is not a rounding error at that volume — it decides whether your model trains in a week or a year. Cheap arithmetic is not a detail; it is the reason modern attention exists at all.
%%%

So far: two different judges, wildly different prices. Let's line up the actual work each one does for a single pair.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two computation paths to a match score. Top path, additive attention: Query and Key are glued, pushed through a small neural network with its own learned weights W_q, W_k and v, bent, then multiplied — four passes and three extra weight sets. Bottom path, dot product: Query and Key go straight into one multiply-and-add — one pass and no extra weights."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same input, same job — count the work</text><rect x="14" y="36" width="46" height="22" fill="#EAF0FA" stroke="#5E5191"/><text x="37" y="51" text-anchor="middle" fill="#5E5191" font-size="9">Q</text><rect x="14" y="62" width="46" height="22" fill="#FCF3DC" stroke="#C99A12"/><text x="37" y="77" text-anchor="middle" fill="#8A6D3B" font-size="9">K</text><rect x="86" y="44" width="54" height="32" rx="4" fill="#FBE9E9" stroke="#C0392B"/><text x="113" y="64" text-anchor="middle" fill="#C0392B" font-size="8">glue</text><rect x="152" y="44" width="70" height="32" rx="4" fill="#FBE9E9" stroke="#C0392B"/><text x="187" y="59" text-anchor="middle" fill="#C0392B" font-size="8">W_q, W_k</text><text x="187" y="71" text-anchor="middle" fill="#C0392B" font-size="8">(learned)</text><rect x="234" y="44" width="46" height="32" rx="4" fill="#FBE9E9" stroke="#C0392B"/><text x="257" y="64" text-anchor="middle" fill="#C0392B" font-size="8">bend</text><rect x="292" y="44" width="56" height="32" rx="4" fill="#FBE9E9" stroke="#C0392B"/><text x="320" y="59" text-anchor="middle" fill="#C0392B" font-size="8">× v</text><text x="320" y="71" text-anchor="middle" fill="#C0392B" font-size="8">(learned)</text><rect x="360" y="44" width="52" height="32" rx="4" fill="#FDF9F3" stroke="#B8AEA2"/><text x="386" y="64" text-anchor="middle" fill="#6B645E" font-size="9">score</text><line x1="60" y1="60" x2="86" y2="60" stroke="#C0392B"/><line x1="140" y1="60" x2="152" y2="60" stroke="#C0392B"/><line x1="222" y1="60" x2="234" y2="60" stroke="#C0392B"/><line x1="280" y1="60" x2="292" y2="60" stroke="#C0392B"/><line x1="348" y1="60" x2="360" y2="60" stroke="#C0392B"/><text x="466" y="56" text-anchor="middle" fill="#C0392B" font-size="9">4 passes</text><text x="466" y="70" text-anchor="middle" fill="#C0392B" font-size="9">3 weight sets</text><line x1="14" y1="102" x2="500" y2="102" stroke="#E5DED2"/><rect x="14" y="118" width="46" height="22" fill="#EAF0FA" stroke="#5E5191"/><text x="37" y="133" text-anchor="middle" fill="#5E5191" font-size="9">Q</text><rect x="14" y="144" width="46" height="22" fill="#FCF3DC" stroke="#C99A12"/><text x="37" y="159" text-anchor="middle" fill="#8A6D3B" font-size="9">K</text><rect x="152" y="126" width="196" height="32" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="250" y="146" text-anchor="middle" fill="#276b45" font-size="9">multiply slot by slot &amp; add</text><line x1="60" y1="142" x2="152" y2="142" stroke="#2D8B55"/><line x1="348" y1="142" x2="360" y2="142" stroke="#2D8B55"/><rect x="360" y="126" width="52" height="32" rx="4" fill="#EAF5EE" stroke="#2D8B55"/><text x="386" y="146" text-anchor="middle" fill="#276b45" font-size="9">score</text><text x="466" y="138" text-anchor="middle" fill="#276b45" font-size="9">1 pass</text><text x="466" y="152" text-anchor="middle" fill="#276b45" font-size="9">0 extra weights</text></g></svg>
%%%

%%% demo id=bahdanau label="count the work each scorer does for ONE pair"
predict: bank's Query meets river's Key. One scorer needs learned weights before it can say anything at all; one needs none. Which numbers on this page can you check by hand — and which can't you?
code: Q_bank = [2,1,1,1];  K_river = [2,0,1,1]
code: score_dot(Q, K)       # one multiply-and-add, no extra weights
code: score_additive(Q, K)  # needs W_q, W_k, v — three learned weight sets
out: DOT (modern):    2*2 + 1*0 + 1*1 + 1*1 = 6      <- you can check this yourself
out: ADDITIVE (old):  glue -> W_q,W_k -> bend -> ×v -> one number
out:                  untrained weights -> a meaningless number, any ranking at all
out:                  after training     -> a useful score, on ITS own scale, from ITS own weights
take: <b>Two different judges, one much cheaper.</b> The dot product's 6 you just checked by hand, with zero extra weights. Additive attention is its own learned function of the same Q and K — a well-trained one usually agrees that river beats "the", but an untrained one has no reason to, and its numbers never live on the dot product's scale. Nobody switched because the dot product's numbers were <i>better</i>: they switched because it reaches comparable quality at ONE multiply-and-add.
%%%

#### The family of scorers, side by side
Additive wasn't the only contender. Here's the short history of "how should we score a pair?" in one table.

%%% table
Scorer :: Extra learned weights? :: What it's good at
Additive (Bahdanau, 2014) :: yes — a small neural network :: it works, but it's many steps per pair
Bilinear (Luong, 2015) :: yes — one weight grid in the middle :: fewer steps than additive, still extra weights
Dot product :: none :: one multiply-and-add — fast and chip-friendly
Scaled dot product (today's default) :: none :: same speed, plus the ÷√d_k volume knob
Cosine :: none :: list length can't bully the score — but you lose the confidence long lists carried
%%%

!!! c-info 🔬
<b>Optional (skippable) — the honest trade-off.</b> Additive attention has its own weight matrices, so in principle it can learn match patterns a bare dot product cannot express — it is the <i>more</i> flexible scorer, not the weaker one. What settled it is that Query and Key each already get their own learned projection (Day 2), so the flexibility mostly moves there instead, and the plain dot product reaches comparable quality at a fraction of the cost. Cheaper with no real loss is a rare win — enjoy it. Cosine scoring is the other one worth knowing: shrink both lists to length 1 before the dot product, so a single very long Key can't bully its whole column just by being big. You pay for it by throwing away the confidence that a long list carried.
!!!

@@@ concept id=c8 tag="Puzzle 1 · too loud" title="The spotlight puzzle — when one word takes the whole budget" gotit="Solved puzzle 1"
Here is a little mystery that engineers hit for real, and it is oddly beautiful once you see it. A model trains for hours. Nothing crashes. The loss just… stops getting better. Where did the learning go?

Let's find out with a lamp.

Think of **a spotlight with a dimmer** in a dark room with three friends standing in it.

At a gentle setting you can see all three, one a little brighter than the others — you can tell them apart. Now crank the dimmer all the way up. One face goes blinding white and the other two vanish into black. And here's the eerie part: nudge the dimmer a hair now and *nothing you can see changes*. The picture has stopped responding.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A spotlight with a dimmer, drawn twice. On the left the dimmer is gentle: a soft pool of light shows three friends, one slightly brighter than the others, and you can tell them apart. On the right the dimmer is cranked up: one figure is a blinding white blob and the other two have vanished into black, with a note that nudging the dimmer now changes nothing you can see."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Same three friends, two dimmer settings</text><rect x="14" y="26" width="234" height="172" rx="8" fill="#F4F1EA" stroke="#8Cc4a3" stroke-width="1.5"/><text x="131" y="44" text-anchor="middle" fill="#276b45" font-size="10">dimmer: gentle 🔅</text><path d="M131 52 L64 130 L198 130 Z" fill="#FCF7E4" stroke="#C99A12" stroke-dasharray="3,2"/><circle cx="90" cy="150" r="11" fill="#E4EFE8" stroke="#2D8B55"/><circle cx="131" cy="146" r="13" fill="#C6E4D2" stroke="#2D8B55" stroke-width="2"/><circle cx="176" cy="150" r="11" fill="#E4EFE8" stroke="#2D8B55"/><text x="131" y="180" text-anchor="middle" fill="#276b45" font-size="9">you can tell all three apart</text><text x="131" y="194" text-anchor="middle" fill="#276b45" font-size="8">nudge the dimmer → the picture answers</text><rect x="272" y="26" width="234" height="172" rx="8" fill="#2C2A28" stroke="#C0392B" stroke-width="1.5"/><text x="389" y="44" text-anchor="middle" fill="#F3C9C4" font-size="10">dimmer: cranked 🔆</text><path d="M389 52 L322 130 L456 130 Z" fill="#FFFFFF" opacity="0.25"/><circle cx="348" cy="150" r="11" fill="#2C2A28" stroke="#4A4540"/><circle cx="389" cy="146" r="15" fill="#FFFFFF" stroke="#FFFFFF"/><circle cx="434" cy="150" r="11" fill="#2C2A28" stroke="#4A4540"/><text x="389" y="180" text-anchor="middle" fill="#F3C9C4" font-size="9">one blinding blob, two gone</text><text x="389" y="194" text-anchor="middle" fill="#F3C9C4" font-size="8">nudge the dimmer → nothing visibly changes</text></g></svg>
%%%

**What the spotlight picture gets right:** turning the setting too high does not just look different, it destroys your ability to *compare* the three — and it also stops responding to small nudges, which is exactly what happens to a model's learning signal.

**Where it breaks down:** a room really does get brighter, but here nothing gets bright at all. The "cranking" is scores growing large, and the "blinding" is one word quietly taking the entire attention **budget**.

#### The spike, and the stall
The lamp is our scores; the picture is the softmax row. Let's walk it, with the choir from the scaling unit still singing in the background.

%%% steps
step: the swell — long Query and Key lists mean many products added up, so raw scores grow large
why: nothing is broken yet; the numbers are just loud, exactly as the choir picture showed
step: the spike — feed loud scores into softmax and the stretch step blows the leader far ahead: one share becomes about 1.00 and every other share about 0.00
why: which means the whole <b>budget</b> lands on one word. The pizza is not sliced any more; one friend ate it
step: the stall — that's the cranked dimmer. A tiny change to a score barely moves the shares at all, so there is almost no signal telling the model to look somewhere else
why: therefore the model stops improving at WHO to look at, and nothing crashes while that happens. Engineers call this softmax saturation
step: the fix you already own — divide every score by √d_k before softmax
why: the volume comes down, the shares become readable again, and small nudges start moving them. That is the whole job of scaled dot-product attention
step: two more dials, for when that isn't enough
why: raising the <b>temperature</b> above 1 (divide scores by it) flattens the row further, and big modern models add QK-normalization — a tidy-up applied to Q and K <i>before</i> they are ever scored, so the scores can't run away in the first place
%%%

%%% insight
Notice how quiet this failure is. No error message, no crash, no NaN — just a model that trains forever and learns nothing about where to look. Silent failures are the expensive kind, which is exactly why one small division earned itself a permanent place in every transformer.
%%%

So far: loud → spike → stall, and the cure is the division you already know. Let's catch it in the act with a real `d_k` of 64.

%%% demo id=spike label="the same two scores, loud and then gentle"
predict: two scores, 56 and 40 — a 16-point gap. After softmax, how much of the budget does the loser keep: a fifth? a tenth? almost nothing? Guess, then reveal.
code: d_k = 64  ->  √d_k = 8
code: loud   = [56, 40];       softmax(loud)
code: gentle = [56/8, 40/8];   softmax(gentle)
out: loud   -> [56, 40]  ->  shares [1.00, 0.00]   the loser gets essentially nothing
out: gentle -> [7.0, 5.0] ->  shares [0.88, 0.12]   the loser keeps a real voice
take: <b>Same ranking, completely different attention.</b> Loud scores collapsed the pizza to one slice — and once a share sits at 1.00 and 0.00 there is nothing left for the model to nudge, so it stops learning who to listen to. One division by 8 turned the same two scores back into a sharing that can still move. That is the whole reason √d_k exists.
%%%

Here is that same before-and-after in one picture — the cranked row beside the gentle one.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="Two softmax rows drawn as bars. On the left, loud scores 56 and 40 give shares of 1.00 and 0.00: one full-height bar and one invisible bar, labelled nothing left to nudge. On the right, the same scores divided by 8 give 7.0 and 5.0 and shares of 0.88 and 0.12: a tall bar and a short but clearly visible bar, labelled the shares can still move."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One division by 8 — and the row can breathe again</text><rect x="16" y="28" width="228" height="150" rx="8" fill="#FBE9E9" stroke="#C0392B" stroke-width="1.5"/><text x="130" y="46" text-anchor="middle" fill="#C0392B" font-size="10">loud: scores [56, 40]</text><line x1="46" y1="140" x2="214" y2="140" stroke="#E3B6B1"/><rect x="72" y="58" width="34" height="82" fill="#C0392B"/><text x="89" y="52" text-anchor="middle" fill="#C0392B">1.00</text><text x="89" y="153" text-anchor="middle" fill="#8A2B21" font-size="9">river</text><rect x="152" y="138" width="34" height="2" fill="#E3B6B1"/><text x="169" y="132" text-anchor="middle" fill="#C0392B">0.00</text><text x="169" y="153" text-anchor="middle" fill="#8A2B21" font-size="9">bank</text><text x="130" y="170" text-anchor="middle" fill="#C0392B" font-size="9">nothing left to nudge → learning stalls</text><rect x="276" y="28" width="228" height="150" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="390" y="46" text-anchor="middle" fill="#276b45" font-size="10">gentle: ÷8 → [7.0, 5.0]</text><line x1="306" y1="140" x2="474" y2="140" stroke="#A8D3BB"/><rect x="332" y="68" width="34" height="72" fill="#2D8B55"/><text x="349" y="62" text-anchor="middle" fill="#276b45">0.88</text><text x="349" y="153" text-anchor="middle" fill="#276b45" font-size="9">river</text><rect x="412" y="130" width="34" height="10" fill="#8Cc4a3"/><text x="429" y="124" text-anchor="middle" fill="#276b45">0.12</text><text x="429" y="153" text-anchor="middle" fill="#276b45" font-size="9">bank</text><text x="390" y="170" text-anchor="middle" fill="#276b45" font-size="9">both shares visible → the row can still move</text></g></svg>
%%%

!!! c-ok ✅
<b>Puzzle 1 solved — and you solved it with a tool you already had.</b> That is a genuinely good feeling: the fix wasn't new machinery, it was the same `÷√d_k` from two units ago, now with a reason you can explain out loud. Stretch, take a breath, and let's go do something completely different: making a word *invisible*.
!!!

@@@ concept id=c9 tag="Puzzle 2 · the invisible word" title="Masking — how to make a word impossible to listen to" gotit="Solved puzzle 2"
Now for the puzzle I think is the most fun in the whole day: **how do you make a word invisible to attention** — using nothing but a number? No switches, no if-statements, just arithmetic. It turns out there's a trick so neat it feels like sleight of hand, and it's what makes a chatbot able to *generate* text one word at a time instead of just reading it.

Think of **doing a quiz with the answers printed at the bottom of the same page**.

You cover the answer strip with your hand. The answers are still physically there — you simply cannot use them, so your guesses have to be honest. And if the worksheet has three blank filler lines at the end because it was printed for a longer quiz, you cover those too: there is nothing to read there.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="A quiz worksheet analogy for masking. The top of the page shows three answered questions you are allowed to use. Below them an answer strip is covered by a hand, labelled you may not look at this. At the bottom two blank filler lines are also covered, labelled nothing to read here. A caption says both are still on the page, they are just unusable."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Cover what you are not allowed to use</text><rect x="120" y="26" width="280" height="164" rx="6" fill="#FDFBF6" stroke="#B8AEA2" stroke-width="1.5"/><text x="136" y="48" fill="#276b45" font-size="10">Q1 ....... ✓ allowed</text><text x="136" y="66" fill="#276b45" font-size="10">Q2 ....... ✓ allowed</text><text x="136" y="84" fill="#276b45" font-size="10">Q3 ....... ✓ allowed</text><line x1="132" y1="94" x2="388" y2="94" stroke="#E5DED2"/><rect x="132" y="100" width="256" height="30" rx="4" fill="#FBE9E9" stroke="#C0392B" stroke-dasharray="3,2"/><text x="196" y="120" fill="#C0392B" font-size="10">ANSWERS: b, d, a</text><path d="M300 96 q22 -10 44 4 q10 22 -6 34 q-30 8 -44 -6 Z" fill="#F4E3C8" stroke="#8A6D3B" stroke-width="1.5"/><text x="322" y="122" text-anchor="middle" fill="#8A6D3B" font-size="9">✋</text><text x="260" y="146" text-anchor="middle" fill="#C0392B" font-size="9">you may not look at this (the future)</text><rect x="132" y="154" width="256" height="14" rx="3" fill="#EEEAE2" stroke="#B8AEA2" stroke-dasharray="3,2"/><text x="260" y="180" text-anchor="middle" fill="#9A938A" font-size="9">blank filler lines — nothing to read here (padding)</text><text x="58" y="112" text-anchor="middle" fill="#5E5191" font-size="9">still on</text><text x="58" y="126" text-anchor="middle" fill="#5E5191" font-size="9">the page,</text><text x="58" y="140" text-anchor="middle" fill="#5E5191" font-size="9">just unusable</text><path d="M96 126 L118 122" stroke="#5E5191" stroke-width="1.5"/><text x="454" y="112" text-anchor="middle" fill="#276b45" font-size="9">your guesses</text><text x="454" y="126" text-anchor="middle" fill="#276b45" font-size="9">stay honest</text></g></svg>
%%%

**What the covered-quiz picture gets right:** the forbidden material is still sitting there in the grid, and "masking" is purely about making it unusable — plus the two things worth covering really are the answers ahead of you and the blank filler.

**Where it breaks down:** you cover a page with a hand, which is a hard on/off. Attention has no hand — it mutes a word with a number, and the muting is *almost* perfect rather than perfect. You'll see exactly how in a moment.

#### The leak, the filler, and the trick
Two situations where scoring every pair is simply wrong, then the one-move fix — and one timing trap that catches nearly everybody once.

%%% steps
step: the leak — a word being generated must not peek at FUTURE words
why: the model's whole job is guessing what comes next, so reading the answer first turns training into free lookup and teaches it nothing. That's the answer strip under your hand
step: the filler — sentences in a batch get padded to the same length with blank [[padding||Blank filler slots added so every sentence in a batch is the same length. They hold no real word.]] slots
why: which means softmax would happily spend part of a word's <b>budget</b> on empty filler that holds no meaning at all — the blank lines at the bottom of the worksheet
step: the trick — [[masking||Setting forbidden scores to negative infinity before softmax, so their attention share comes out ~0.]]: set every forbidden score to negative infinity
why: the stretch step turns a score that low into essentially zero hunger, therefore that word's slice comes out ~0 and the freed budget flows to the words that are allowed. That's how you mute a word with nothing but a number
step: the timing trap — mask BEFORE softmax, never after
why: zeroing shares after they have already been shared out leaves a row that no longer sums to one, which quietly shrinks the whole output toward zero. Your test is delightfully simple: every row of shares must still add up to 1
%%%

Remember the dark, blank cells I promised you back at the grid? Here they are — masking drawn straight onto our own score matrix.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="The score grid for the river bank with a padding column added, and masking drawn on it. Rows are the asking words the, river and bank. Columns are the offered words the, river, bank and pad. Allowed cells hold real scores: row the holds 2; row river holds 2 and 4; row bank holds 2, 6 and 4. Every other cell is dark and blank, marked minus infinity — the future words above the diagonal and the whole padding column."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Masking, drawn on the grid: dark cell = switched off before softmax</text><text x="170" y="52" text-anchor="middle" fill="#9A938A" font-size="9">the</text><text x="230" y="52" text-anchor="middle" fill="#9A938A" font-size="9">river</text><text x="290" y="52" text-anchor="middle" fill="#9A938A" font-size="9">bank</text><text x="350" y="52" text-anchor="middle" fill="#C0392B" font-size="9">pad</text><text x="100" y="80" text-anchor="middle" fill="#5E5191" font-size="9">the</text><text x="100" y="110" text-anchor="middle" fill="#5E5191" font-size="9">river</text><text x="100" y="140" text-anchor="middle" fill="#5E5191" font-size="9">bank</text><rect x="140" y="60" width="60" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="170" y="80" text-anchor="middle" fill="#276b45">2</text><rect x="200" y="60" width="60" height="30" fill="#2C2A28" stroke="#6B645E"/><text x="230" y="80" text-anchor="middle" fill="#B8AEA2" font-size="9">−∞</text><rect x="260" y="60" width="60" height="30" fill="#2C2A28" stroke="#6B645E"/><text x="290" y="80" text-anchor="middle" fill="#B8AEA2" font-size="9">−∞</text><rect x="320" y="60" width="60" height="30" fill="#2C2A28" stroke="#6B645E"/><text x="350" y="80" text-anchor="middle" fill="#B8AEA2" font-size="9">−∞</text><rect x="140" y="90" width="60" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="170" y="110" text-anchor="middle" fill="#276b45">2</text><rect x="200" y="90" width="60" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="230" y="110" text-anchor="middle" fill="#276b45">4</text><rect x="260" y="90" width="60" height="30" fill="#2C2A28" stroke="#6B645E"/><text x="290" y="110" text-anchor="middle" fill="#B8AEA2" font-size="9">−∞</text><rect x="320" y="90" width="60" height="30" fill="#2C2A28" stroke="#6B645E"/><text x="350" y="110" text-anchor="middle" fill="#B8AEA2" font-size="9">−∞</text><rect x="140" y="120" width="60" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="170" y="140" text-anchor="middle" fill="#276b45">2</text><rect x="200" y="120" width="60" height="30" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="230" y="140" text-anchor="middle" fill="#276b45" font-weight="bold">6</text><rect x="260" y="120" width="60" height="30" fill="#EAF5EE" stroke="#2D8B55"/><text x="290" y="140" text-anchor="middle" fill="#276b45">4</text><rect x="320" y="120" width="60" height="30" fill="#2C2A28" stroke="#6B645E"/><text x="350" y="140" text-anchor="middle" fill="#B8AEA2" font-size="9">−∞</text><text x="260" y="176" text-anchor="middle" fill="#C0392B" font-size="9">dark above the diagonal = no peeking at future words · dark last column = blank padding</text><text x="260" y="196" text-anchor="middle" fill="#276b45" font-size="9">every surviving row still gets one whole pizza — the budget just has fewer mouths</text></g></svg>
%%%

%%% insight
Look at the very first row for a second. "the" is the first word, so it is allowed to listen to exactly one thing: itself. Its whole <b>budget</b> goes there, and its share is 1.00 by force, not by choice. That is not a bug — it is what "no peeking" costs the first word, and it is the reason text generation can work at all: every word is trained under exactly the conditions it will face when it has to guess the next one.
%%%

%%% hint
t1: Feeling stuck on why setting a score to negative infinity "mutes" a word? Go back to the pizza. Softmax hands out slices by how big each score is — so ask yourself what slice a score way down at the bottom would get.
t2: Try one tiny example. Scores are [river 3.0, FUTURE 2.0]. Now push FUTURE all the way down: [river 3.0, FUTURE −∞]. Softmax turns each score into a slice, and a score of −∞ turns into a slice of essentially 0 — so river keeps the whole pizza, and FUTURE gets nothing.
t3: In one sentence: masking sets a forbidden word's score to −∞ before softmax, so its slice comes out ~0 and the freed budget flows to the words that are allowed.
%%%

Feeling wobbly about negative infinity? That's completely normal — it looks like a hack the first time. Watch it behave once and it stops feeling strange.

%%% demo id=mask label="softmax before vs after masking the forbidden words"
predict: FUTURE and pad are forbidden. Before revealing, guess how much of the budget FUTURE would steal if we DIDN'T mask it — a tiny sliver, or a real chunk?
code: scores = [the 1.0, river 3.0, FUTURE 2.0, pad 0.0]
code: softmax(scores)          vs          softmax(mask(scores))
out: no mask -> the 0.09  river 0.64  FUTURE 0.24  pad 0.03   (sums to 1.00 — FUTURE steals real attention!)
out: masked  -> set FUTURE,pad = −∞
out:         -> the 0.12  river 0.88  FUTURE 0.00  pad 0.00   (still sums to 1.00)
take: <b>Nearly a quarter of the budget was going to a word the model is not allowed to see.</b> Set those scores to −∞ and softmax gives them essentially nothing — and look where that freed budget went: "the" rose from 0.09 to 0.12 and river from 0.64 to 0.88. The row adds up to 1.00 both times. Same budget, honestly spent.
%%%

!!! c-ok ✅
<b>Both puzzles, both fixes, in one breath:</b> scores get too loud → divide by √d_k; a word must not be listened to → mask it (score = −∞, <i>before</i> softmax). And keep that one row-sum check in your pocket — it catches two different mistakes at once. If a row of shares doesn't add to 1, you either masked after softmax, or you ran softmax down the wrong side of the grid (it must run <i>across</i> the offered words, one asking word at a time — never down a column, which would mix different askers together).
!!!

@@@ concept id=c10 tag="Honest limits" title="What a score is — and what it quietly is not" gotit="Got the limits"
Here's a thing that separates people who *use* a tool from people who really understand it: knowing precisely what it cannot do. Two facts about scores are genuinely surprising the first time, and both of them will save you real confusion later.

Think of **a single thermometer reading**: "22". On its own it tells you nothing — 22 degrees? 22 out of 100?

You only know what it means once you see the scale and the other readings. And a thermometer tells you *how warm*, never *what time* — it simply doesn't carry that kind of information at all.

%%% svg
<svg viewBox="0 0 520 190" role="img" aria-label="A thermometer analogy. On the left a thermometer reads 22 with a big question mark beside it and the note 22 what — degrees, or out of a hundred. On the right a clock face is crossed out, with the note that a thermometer carries no sense of time at all."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One reading, on its own, says almost nothing</text><rect x="60" y="44" width="16" height="96" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><rect x="63" y="92" width="10" height="46" fill="#C0392B"/><circle cx="68" cy="150" r="14" fill="#C0392B" stroke="#8A2B21" stroke-width="1.5"/><line x1="80" y1="60" x2="90" y2="60" stroke="#B8AEA2"/><line x1="80" y1="80" x2="90" y2="80" stroke="#B8AEA2"/><line x1="80" y1="100" x2="90" y2="100" stroke="#B8AEA2"/><line x1="80" y1="120" x2="90" y2="120" stroke="#B8AEA2"/><text x="112" y="96" fill="#3A342E" font-size="22">22</text><text x="160" y="96" fill="#9A938A" font-size="26">?</text><text x="112" y="122" fill="#9A938A" font-size="9">22 degrees? 22 out of 100?</text><text x="112" y="138" fill="#9A938A" font-size="9">you can't tell without the others</text><line x1="250" y1="34" x2="250" y2="170" stroke="#E5DED2"/><circle cx="330" cy="92" r="34" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><line x1="330" y1="92" x2="330" y2="70" stroke="#6B645E" stroke-width="2"/><line x1="330" y1="92" x2="348" y2="98" stroke="#6B645E" stroke-width="2"/><line x1="306" y1="68" x2="354" y2="116" stroke="#C0392B" stroke-width="3"/><text x="420" y="86" text-anchor="middle" fill="#C0392B" font-size="10">no sense of time</text><text x="420" y="104" text-anchor="middle" fill="#9A938A" font-size="9">it simply doesn't carry</text><text x="420" y="118" text-anchor="middle" fill="#9A938A" font-size="9">that kind of information</text></g></svg>
%%%

**What the thermometer picture gets right:** a lone reading is meaningless until compared, and it carries no sense of time or place — exactly like a raw score.

**Where it breaks down:** a thermometer could be handed a clock; a bare dot product has no way to sense position at all, which is why word order needs a whole separate mechanism.

#### Limit 1 — a raw score is not a weight yet
Picture bank's winning score on screen: **6**. Is that a lot? Genuinely, you cannot say — and here's the proof, in one picture.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="The same score of six in two different companies. On the left, six sits beside a forty, and after softmax the six's share collapses to about zero while the forty takes the whole budget. On the right, the same six sits beside a zero point five, and after softmax the six takes about the whole budget. Horizontal bars show raw scores on top and shares underneath in each panel."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">The same 6 — two neighbours, two opposite meanings</text><rect x="16" y="28" width="228" height="162" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><text x="130" y="46" text-anchor="middle" fill="#8A6D3B">raw scores: 6 and 40</text><text x="36" y="68" fill="#276b45">6</text><rect x="52" y="58" width="22" height="12" fill="#2D8B55"/><text x="32" y="90" fill="#9A938A">40</text><rect x="52" y="80" width="150" height="12" fill="#B8AEA2"/><text x="130" y="112" text-anchor="middle" fill="#5E5191">↓ softmax the row ↓</text><text x="36" y="136" fill="#276b45">6</text><rect x="52" y="126" width="2" height="12" fill="#2D8B55"/><text x="64" y="136" fill="#C0392B">share 0.00</text><text x="32" y="158" fill="#9A938A">40</text><rect x="52" y="148" width="150" height="12" fill="#B8AEA2"/><text x="208" y="158" fill="#6B645E">1.00</text><text x="130" y="180" text-anchor="middle" fill="#C0392B">the 6 gets nothing</text><rect x="276" y="28" width="228" height="162" rx="8" fill="#FDF9F3" stroke="#E5DFD6"/><text x="390" y="46" text-anchor="middle" fill="#8A6D3B">raw scores: 6 and 0.5</text><text x="296" y="68" fill="#276b45">6</text><rect x="312" y="58" width="150" height="12" fill="#2D8B55"/><text x="288" y="90" fill="#9A938A">0.5</text><rect x="312" y="80" width="13" height="12" fill="#B8AEA2"/><text x="390" y="112" text-anchor="middle" fill="#5E5191">↓ softmax the row ↓</text><text x="296" y="136" fill="#276b45">6</text><rect x="312" y="126" width="150" height="12" fill="#2D8B55"/><text x="468" y="136" fill="#276b45">1.00</text><text x="288" y="158" fill="#9A938A">0.5</text><rect x="312" y="148" width="2" height="12" fill="#B8AEA2"/><text x="330" y="158" fill="#6B645E">share 0.00</text><text x="390" y="180" text-anchor="middle" fill="#276b45">the same 6 takes everything</text></g></svg>
%%%

%%% steps
step: the 6 beside a 40 → its share collapses to about 0.00
why: the row is shared out as one pizza, so a neighbour with a much bigger score simply takes the whole thing
step: the same 6 beside a 0.5 → its share is about 1.00
why: which means the identical number now means "listen to me almost completely". Nothing about the 6 changed — only its company
step: therefore a raw score means nothing on its own
why: it becomes a real share of attention only after softmax weighs the WHOLE row together into slices that sum to 1
%%%

%%% insight
This is the single most common beginner mistake in attention, so let it land: never read one raw score as "how much attention". Until the row is turned into a <b>budget</b>, a score is just a vote that hasn't been counted yet. The good news is you already have the cure — you built it in unit 5. Softmax *is* the fix; this limit is a reading habit to unlearn, not a hole in the machinery.
%%%

So far: a score needs its whole row to mean anything. The second limit is stranger, and unlike the first one it really is a hole — one that takes a whole separate mechanism to fill.

#### Limit 2 — scores carry no sense of order
Let's catch the dot product out. We do it by moving the words around and watching the grid.

%%% steps
step: start with "the river bank" and fill the grid — that's the nine scores you already computed
why: every word brought its own Query and its own Key, and those little lists came from the word itself, not from where it sits
step: now shuffle the sentence into "bank river the" — each word's Query and Key travel with it
why: which means the same nine numbers come out; only the row and column LABELS moved. Nothing in "multiply slot by slot and add" ever asked which word came first
step: careful — swapping WHO ASKS is a different move, and it DOES change a number
why: "river asks bank" is river's Query against bank's Key (our grid says 2), while "bank asks river" is bank's Query against river's Key (a 6). Two different questions, two different answers — the grid is not a mirror
step: so pure scoring is order-blind, and word order has to be added on separately
why: therefore "dog bites man" and "man bites dog" would score identically. This is exactly what the old phrase <b>bag of words</b> points at — a bag doesn't remember which word you dropped in first. The separate trick that fixes it is called <b>positional encoding</b>
%%%

Here are the two grids side by side, so you can watch the same nine scores just change seats.

%%% svg
<svg viewBox="0 0 520 210" role="img" aria-label="Two three by three score grids. The left grid is for the sentence the river bank with rows 2 2 2, then 2 4 2, then 2 6 4. The right grid is for the shuffled sentence bank river the with rows 4 6 2, then 2 4 2, then 2 2 2. The same nine numbers appear in both grids, only in different positions, showing the scores do not know word order."><g font-family="monospace" font-size="10"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">Shuffle the sentence → the same nine scores, new seats</text><text x="118" y="36" text-anchor="middle" fill="#5E5191" font-size="10">"the river bank"</text><text x="60" y="52" fill="#9A938A" font-size="8">the</text><text x="94" y="52" fill="#9A938A" font-size="8">river</text><text x="132" y="52" fill="#9A938A" font-size="8">bank</text><text x="34" y="72" fill="#5E5191" font-size="8">the</text><text x="30" y="102" fill="#5E5191" font-size="8">river</text><text x="30" y="132" fill="#5E5191" font-size="8">bank</text><rect x="56" y="58" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="73" y="74" text-anchor="middle" fill="#9A938A">2</text><rect x="92" y="58" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="109" y="74" text-anchor="middle" fill="#9A938A">2</text><rect x="128" y="58" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="145" y="74" text-anchor="middle" fill="#9A938A">2</text><rect x="56" y="88" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="73" y="104" text-anchor="middle" fill="#9A938A">2</text><rect x="92" y="88" width="34" height="24" fill="#F2F8F4" stroke="#8Cc4a3"/><text x="109" y="104" text-anchor="middle" fill="#276b45">4</text><rect x="128" y="88" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="145" y="104" text-anchor="middle" fill="#9A938A">2</text><rect x="56" y="118" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="73" y="134" text-anchor="middle" fill="#9A938A">2</text><rect x="92" y="118" width="34" height="24" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="109" y="134" text-anchor="middle" fill="#276b45" font-weight="bold">6</text><rect x="128" y="118" width="34" height="24" fill="#F2F8F4" stroke="#8Cc4a3"/><text x="145" y="134" text-anchor="middle" fill="#276b45">4</text><text x="260" y="102" text-anchor="middle" fill="#8A6D3B" font-size="14">→</text><text x="260" y="122" text-anchor="middle" fill="#8A6D3B" font-size="8">shuffle</text><text x="382" y="36" text-anchor="middle" fill="#5E5191" font-size="10">"bank river the"</text><text x="322" y="52" fill="#9A938A" font-size="8">bank</text><text x="358" y="52" fill="#9A938A" font-size="8">river</text><text x="400" y="52" fill="#9A938A" font-size="8">the</text><text x="294" y="72" fill="#5E5191" font-size="8">bank</text><text x="294" y="102" fill="#5E5191" font-size="8">river</text><text x="298" y="132" fill="#5E5191" font-size="8">the</text><rect x="320" y="58" width="34" height="24" fill="#F2F8F4" stroke="#8Cc4a3"/><text x="337" y="74" text-anchor="middle" fill="#276b45">4</text><rect x="356" y="58" width="34" height="24" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="373" y="74" text-anchor="middle" fill="#276b45" font-weight="bold">6</text><rect x="392" y="58" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="409" y="74" text-anchor="middle" fill="#9A938A">2</text><rect x="320" y="88" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="337" y="104" text-anchor="middle" fill="#9A938A">2</text><rect x="356" y="88" width="34" height="24" fill="#F2F8F4" stroke="#8Cc4a3"/><text x="373" y="104" text-anchor="middle" fill="#276b45">4</text><rect x="392" y="88" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="409" y="104" text-anchor="middle" fill="#9A938A">2</text><rect x="320" y="118" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="337" y="134" text-anchor="middle" fill="#9A938A">2</text><rect x="356" y="118" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="373" y="134" text-anchor="middle" fill="#9A938A">2</text><rect x="392" y="118" width="34" height="24" fill="#FDF9F3" stroke="#B8AEA2"/><text x="409" y="134" text-anchor="middle" fill="#9A938A">2</text><text x="260" y="168" text-anchor="middle" fill="#C0392B" font-size="10">every number survived the shuffle — the recipe never knew the order</text><text x="260" y="190" text-anchor="middle" fill="#8A6D3B" font-size="9">but the grid is still NOT a mirror: bank→river is 6, river→bank is 2</text></g></svg>
%%%

Two limits, honestly stated — and neither one is a flaw you have to fix today. Limit 1 you already cured this morning: softmax turns the whole row into a budget, so "a score is not a weight" is a reading habit to keep, not a repair to make. Limit 2 — order-blindness — is a real gap, and it gets its own fix on **Day 5** (positional encoding). Day 4 is about something different and lovely: running several of these meetings at once.

!!! c-info 🗺️
<b>Two curiosities, and where the rest of the story lives.</b> First: a bright cell on an attention map shows where the weight <i>went</i>, not proof that the answer depended on that word — attention weights are not explanations, and researchers argue about this a lot. Second: because the shares must add to 1, a word with nothing useful to look at still has to park its <b>budget</b> somewhere, and heads often dump it on the very first word — an oddity called an <i>attention sink</i>, tamed either by giving heads a spare "register" word to dump into or by letting softmax add a 1 to its pot so "attend to nothing" becomes sayable. <b>Coming up:</b> several score-and-blend meetings at once (multi-head), the cure for one grid holding only ONE notion of relatedness → <b>Day 4</b>. Stamping word order onto the words (positional encoding) → <b>Day 5</b>. Never building the n×n grid at all, so long context gets affordable (FlashAttention) → the inference-systems module.
!!!

@@@ concept id=c11 tag="Recap" title="Today in one page" gotit="Got the recap"
Let's gather everything into one page you can come back to any time. And here's a fresh way to hold the whole day in your hand.

Think of **packing one lunchbox**. The box is a fixed size — that's your attention **budget**, one whole and no more.

First you decide how much you *want* each food (snack high, veggies low) — those wants are the **scores**. Then you pack in proportion to those wants, and when the box is full it's full — that packing step is **softmax**. A food that isn't allowed today gets left out entirely — that's **masking**. And what you actually carry to school is the mixture inside the box — that's the **blended Values**, the output. **Where the lunchbox breaks down:** you pack one box each morning, but a transformer packs a fresh budget for every word at the same time.

%%% svg
<svg viewBox="0 0 520 200" role="img" aria-label="A lunchbox analogy for attention. One fixed-size lunchbox is divided into compartments sized by how much you want each food: a big snack compartment, a smaller veggie one, a small fruit one. The compartments fill the box exactly. Outside the box a candy is crossed out, labelled not allowed today, which is masking."><g font-family="monospace" font-size="11"><text x="260" y="18" text-anchor="middle" fill="#2C2A28" font-size="12">One fixed box, packed in proportion to how much you want each food</text><rect x="56" y="52" width="284" height="112" rx="10" fill="#F4F1EA" stroke="#5E5191" stroke-width="2"/><rect x="64" y="60" width="170" height="96" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="149" y="104" text-anchor="middle" fill="#276b45" font-size="10">🥪 snack</text><text x="149" y="122" text-anchor="middle" fill="#276b45" font-size="9">big want → big share</text><rect x="238" y="60" width="58" height="96" rx="6" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="267" y="104" text-anchor="middle" fill="#8A6D3B" font-size="10">🥕</text><text x="267" y="122" text-anchor="middle" fill="#8A6D3B" font-size="8">small want</text><rect x="300" y="60" width="32" height="96" rx="6" fill="#FDF9F3" stroke="#B8AEA2"/><text x="316" y="104" text-anchor="middle" fill="#9A938A" font-size="10">🍎</text><text x="198" y="184" text-anchor="middle" fill="#5E5191" font-size="10">the box is a fixed size — that IS the budget</text><circle cx="428" cy="88" r="24" fill="#FBE9E9" stroke="#C0392B" stroke-width="1.5"/><text x="428" y="94" text-anchor="middle" fill="#C0392B" font-size="14">🍬</text><line x1="410" y1="70" x2="446" y2="106" stroke="#C0392B" stroke-width="3"/><text x="428" y="132" text-anchor="middle" fill="#C0392B" font-size="9">not allowed today</text><text x="428" y="148" text-anchor="middle" fill="#C0392B" font-size="9">→ left out entirely</text><text x="428" y="166" text-anchor="middle" fill="#C0392B" font-size="9">(that's masking)</text></g></svg>
%%%

#### Rebuild the whole day in five beats — on our own numbers
Before the cheat-sheet, let's re-derive the day once, end to end, on the sentence we've carried all along: **"the river bank"**, with 4-slot Queries and Keys. Watch each beat hand its numbers to the next.

%%% steps
step: beat 1 — one score. Q_bank = [2,1,1,1] meets K_river = [2,0,1,1] → 4 + 0 + 1 + 1 = 6
why: multiply slot by slot, add. One number for one pair — this is the atom, and everything else is just doing it more times
step: beat 2 — every pair. Do that for all nine pairs → rows [2,2,2] / [2,4,2] / [2,6,4]
why: that's the score matrix, one row per asking word. Read bank's row: [2, 6, 4] — it leans on river, exactly where our 6 landed
step: beat 3 — turn the volume down. d_k = 4, so √d_k = 2, and bank's row becomes [1, 3, 2]
why: which means the loudness is gone but the ranking is untouched — river still leads. If any word were forbidden, this is where its score would drop to −∞ instead
step: beat 4 — split the budget. softmax([1, 3, 2]) → 0.0900 / 0.6652 / 0.2447
why: therefore a gap of 1 became a two-thirds share for river, and the three slices add to exactly 1.0000 — one whole pizza
step: beat 5 — spend it. 0.0900×[0,6] + 0.6652×[10,2] + 0.2447×[4,4] → [7.63, 2.85]
why: that is "bank"'s new meaning — mostly river-flavoured, nudged by the rest. From four little lists to one context-aware answer, and you checked every number yourself
%%%

Here is that same rebuild as one picture — the numbers walking left to right through all five beats.

%%% svg
<svg viewBox="0 0 520 220" role="img" aria-label="The five beats of attention drawn as five panels with our running numbers flowing through them. Panel 1: one score, Q dot K equals 6. Panel 2: the three by three score grid with bank's row 2, 6, 4 highlighted. Panel 3: divide by root four, giving 1, 3, 2. Panel 4: softmax bars giving shares 0.09, 0.67 and 0.24. Panel 5: the blended output 7.63 and 2.85. Arrows connect the panels left to right and the full formula is written underneath."><g font-family="monospace" font-size="9"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">score → scale → (mask) → split the budget → blend</text><rect x="8" y="34" width="88" height="120" rx="8" fill="#EAF0FA" stroke="#5E5191" stroke-width="1.5"/><text x="52" y="50" text-anchor="middle" fill="#5E5191" font-size="9">1 · one score</text><text x="52" y="88" text-anchor="middle" fill="#3A342E" font-size="9">Q·K</text><text x="52" y="110" text-anchor="middle" fill="#276b45" font-size="16" font-weight="bold">6</text><text x="52" y="134" text-anchor="middle" fill="#9A938A" font-size="8">bank → river</text><rect x="112" y="34" width="88" height="120" rx="8" fill="#FDF9F3" stroke="#B8AEA2" stroke-width="1.5"/><text x="156" y="50" text-anchor="middle" fill="#8A6D3B" font-size="9">2 · the grid</text><rect x="126" y="70" width="20" height="12" fill="#FDF9F3" stroke="#D8D0C4"/><text x="136" y="79" text-anchor="middle" fill="#9A938A" font-size="7">2</text><rect x="146" y="70" width="20" height="12" fill="#FDF9F3" stroke="#D8D0C4"/><text x="156" y="79" text-anchor="middle" fill="#9A938A" font-size="7">2</text><rect x="166" y="70" width="20" height="12" fill="#FDF9F3" stroke="#D8D0C4"/><text x="176" y="79" text-anchor="middle" fill="#9A938A" font-size="7">2</text><rect x="126" y="82" width="20" height="12" fill="#FDF9F3" stroke="#D8D0C4"/><text x="136" y="91" text-anchor="middle" fill="#9A938A" font-size="7">2</text><rect x="146" y="82" width="20" height="12" fill="#F2F8F4" stroke="#8Cc4a3"/><text x="156" y="91" text-anchor="middle" fill="#276b45" font-size="7">4</text><rect x="166" y="82" width="20" height="12" fill="#FDF9F3" stroke="#D8D0C4"/><text x="176" y="91" text-anchor="middle" fill="#9A938A" font-size="7">2</text><rect x="126" y="94" width="20" height="12" fill="#EAF5EE" stroke="#2D8B55"/><text x="136" y="103" text-anchor="middle" fill="#276b45" font-size="7">2</text><rect x="146" y="94" width="20" height="12" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="156" y="103" text-anchor="middle" fill="#276b45" font-size="7" font-weight="bold">6</text><rect x="166" y="94" width="20" height="12" fill="#EAF5EE" stroke="#2D8B55"/><text x="176" y="103" text-anchor="middle" fill="#276b45" font-size="7">4</text><text x="156" y="126" text-anchor="middle" fill="#276b45" font-size="8">bank's row</text><text x="156" y="138" text-anchor="middle" fill="#276b45" font-size="8">[2, 6, 4]</text><rect x="216" y="34" width="88" height="120" rx="8" fill="#FCF3DC" stroke="#C99A12" stroke-width="1.5"/><text x="260" y="50" text-anchor="middle" fill="#8A6D3B" font-size="9">3 · ÷ √d_k</text><text x="260" y="86" text-anchor="middle" fill="#8A6D3B" font-size="11">÷ √4 = ÷2</text><text x="260" y="112" text-anchor="middle" fill="#3A342E" font-size="11">[1, 3, 2]</text><text x="260" y="136" text-anchor="middle" fill="#9A938A" font-size="8">same winner</text><rect x="320" y="34" width="88" height="120" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="364" y="50" text-anchor="middle" fill="#276b45" font-size="9">4 · softmax</text><line x1="330" y1="118" x2="400" y2="118" stroke="#8Cc4a3"/><rect x="334" y="114" width="14" height="4" fill="#B8AEA2"/><text x="341" y="110" text-anchor="middle" fill="#6B645E" font-size="7">.09</text><rect x="356" y="91" width="14" height="27" fill="#2D8B55"/><text x="363" y="87" text-anchor="middle" fill="#1a5c38" font-size="7">.67</text><rect x="378" y="108" width="14" height="10" fill="#C99A12"/><text x="385" y="104" text-anchor="middle" fill="#9A7208" font-size="7">.24</text><text x="364" y="136" text-anchor="middle" fill="#276b45" font-size="8">sums to 1.00</text><rect x="424" y="34" width="88" height="120" rx="8" fill="#EFEAF7" stroke="#5E5191" stroke-width="1.5"/><text x="468" y="50" text-anchor="middle" fill="#5E5191" font-size="9">5 · blend</text><text x="468" y="86" text-anchor="middle" fill="#5E5191" font-size="8">shares × Values</text><text x="468" y="112" text-anchor="middle" fill="#3A342E" font-size="10">[7.63, 2.85]</text><text x="468" y="136" text-anchor="middle" fill="#9A938A" font-size="8">river-flavoured</text><path d="M96 94 L112 94" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#arR)"/><path d="M200 94 L216 94" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#arR)"/><path d="M304 94 L320 94" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#arR)"/><path d="M408 94 L424 94" stroke="#8A6D3B" stroke-width="1.5" marker-end="url(#arR)"/><defs><marker id="arR" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6 Z" fill="#8A6D3B"/></marker></defs><text x="260" y="182" text-anchor="middle" fill="#5E5191" font-size="12">softmax(Q@Kᵀ / √d_k) @ V — the attention budget, honestly spent</text><text x="260" y="204" text-anchor="middle" fill="#C0392B" font-size="9">forbidden words get −∞ between beats 3 and 4, so their share comes out 0</text></g></svg>
%%%

#### The day in five beats
1. A **score** is one number: how much should this word listen to that word.
2. The **dot product** makes that number — multiply slot by slot, add up. Stack them into the **score matrix**, one row per asking word.
3. **Scale** by √d_k (turn the volume down) and **mask** forbidden words (score = −∞) so raw scores behave.
4. **Softmax** splits one attention **budget** into positive shares that sum to 1 — the biggest score gets the biggest slice.
5. **Blend** the Values by those shares — that weighted mixture is the output, the word's new meaning in context.

That last beat is the one you feel every day: it's why ChatGPT can tell which words in your question belong together, why your phone's keyboard leans on the words that matter instead of the filler, and why a translation app pairs the right two words across two languages.

#### Cheat-sheet — every word from today
%%% jargon
attention score | one number: how much a word should listen to another word
Query (Q) | the "what am I looking for?" side of a word
Key (K) | the "what do I offer?" side of a word
Value (V) | the content a word hands over when someone listens to it
dot product | multiply two lists slot by slot, add up — one match number
score matrix | the grid of all pair scores; one row per asking word
√d_k | the amount we divide scores by to keep them from getting too loud
softmax | turns a row of scores into positive shares that sum to 1
budget | the one whole pot of attention softmax splits into shares
blend (weighted sum) | shares × Values, added up — the output of attention
equal-share averaging | the old baseline: mix every word in equally, so nothing can stand out
additive attention | the older, pricier scorer that used a small neural network
masking | setting forbidden scores to −∞ so their share becomes ~0
padding | blank filler slots that make every sentence in a batch the same length
stable softmax | subtract the row's biggest score first, so exp can't overflow
temperature | a divide-the-scores dial: small = decisive, large = wishy-washy
attention entropy | one number for how spread-out a row of shares is
%%%

#### The fixes and the limits, side by side
%%% table
The move :: What it fixes / means
Divide by √d_k :: scores got too loud → softmax stops slamming into one spike
Mask (→ −∞) :: a word must not be listened to → its share becomes ~0
Row sums to 1 :: your one-line test: if it doesn't, you masked too late or softmaxed the wrong way
Reading habit: score ≠ weight :: a raw score means nothing until softmax normalizes the row — cured today
Limit: order-blind :: shuffle the words and the same set of scores comes back — filled by positional encoding on Day 5
Limit: blend only :: the output always lands between the Values — attention can mix, never invent or subtract
%%%

You just wrote the heart of the transformer. Come back to this page whenever attention feels foggy — the five beats are all of it.

@@@ quiz id=quiz tag="Quiz" title="Four quick checks" gotit="answer all first"
%%% quiz
q: What is an attention score? | a:2 | a learned weight matrix | a whole sentence | one number: how much one word should listen to another | the final output vector | fb: One number per pair — how much to listen.
q: How do you get the score between a Query and a Key? | a:1 | run a big neural network | multiply them slot by slot and add (dot product) | take their average | count matching letters | fb: The dot product — multiply slot-by-slot, add.
q: Why divide scores by √d_k before softmax? | a:0 | to stop long lists making scores too loud (which jams softmax) | to add word order | to make scores negative | to speed up the GPU | fb: Longer lists → bigger scores; √d_k turns the volume down.
q: Softmax gave you positive shares that add up to 1. What happens next? | a:3 | they become the model's answer as they are | they get divided by √d_k again | they are rounded so one word wins | the Values are blended in those proportions | fb: Spend the budget: out = share₁·V₁ + share₂·V₂ + … — a weighted blend of the Values.
%%%

@@@ produce id=produce tag="Produce" title="Build attention scoring by hand" gotit="Done"
Time to make it real with your own hands. In `experiment.py`, build the whole pipeline on our tiny 3-word example and *watch* each step change the numbers.

Start with the three Query lists, three Key lists and three Value lists from today (`Q = [[0,0,1,1],[1,0,1,1],[2,1,1,1]]`, `K = [[0,0,1,1],[2,0,1,1],[0,2,1,1]]`, `V = [[0,6],[10,2],[4,4]]`, so `d_k = 4`). **Predict before you run each step:**
1. Compute the score matrix `Q @ Kᵀ` by hand-style dot products. **Predict** which word will score highest against which — then print the grid and check it against the `[[2,2,2],[2,4,2],[2,6,4]]` we built together.
2. Divide the whole grid by √d_k (here √4 = 2). **Notice** the scores get smaller but the *order* (who beats whom) never changes — bank's row becomes `[1, 3, 2]`.
3. Run softmax on each row. **Observe** that every row now adds up to 1 — print the row sums to prove it. Bank's row should come out `0.09 / 0.67 / 0.24`.
4. Now mask a word: take the 4-score row `[the 1, river 3, FUTURE 2, pad 0]`, set the FUTURE and pad scores to a very large negative number *before* softmax, and **watch** their shares collapse to ~0 while the freed budget flows to the allowed words (`the` rises to 0.12, river to 0.88).
5. Finally spend the budget: multiply each share by that word's Value and add them up. **Predict** which Value the answer will sit closest to, then print it and see whether you land on `[7.63, 2.85]`.

What you should see by the end: a row of raw match numbers turning into a clean budget of shares that sum to 1, and then one blended list of numbers — the exact move at the heart of every transformer. Write it in `experiment.py` and run it.

@@@ fin
