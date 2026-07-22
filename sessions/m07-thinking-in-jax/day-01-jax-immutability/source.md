---
quest_id: w01-d01-jax
mode: concept
donor: v9-base.donor
page_title: "Week 1 · Day 1 — JAX Immutability (The Immutable Array)"
module_label: "Thinking in JAX · Week 1 · Day 1"
title: "JAX and Immutability"
subtitle: "Why a JAX Array Can Never Be Edited in Place"
brand_sub: "Frontier Lab · Week 1 Day 1"
spine: "stone-tablet"
nav_prev_href: "../../m06-cnns-vision-encoders/review.html"
nav_prev_label: "M6 · Review Gate"
nav_next_href: "../day-02-prng-keys/lesson.html"
nav_next_label: "Randomness: Explicit Keys"
fin_title: "Week 1 · Day 1 complete! 🏆"
fin_body: "Welcome to JAX — you've met its first big rule: an array is a <b>stone tablet</b>, carved and never edited in place. To change one, you carve a fresh copy with <code>.at[i].set(v)</code>. That one habit gives you <b>pure functions</b> (same input → same output, no surprise edits) — exactly what lets JAX later compile, vectorize, and differentiate your code automatically.<br>Next up: <b>Randomness: Explicit Keys</b>."
notebook_yardstick: null
---

@@@ hero
@lede Think of a stone tablet: once you carve a word into it, that word is there forever — you cannot rub it out and write a new one. To change the message, you fetch a fresh tablet and carve the whole thing again. That sounds annoying, right? Yet this is the exact rule at the heart of **JAX**, the library used to train some of the biggest AI models around, including work at Google DeepMind. In JAX, a list of numbers (an array) behaves like a **stone-tablet**: once it exists, its numbers never change. And here's the surprise — this one "annoying" rule is the secret that lets JAX later speed your code up, run it on many things at once, and even do calculus on it for you, all automatically. Today we meet that rule, feel why it's actually a gift, and learn the one small habit that replaces the old way of editing numbers.
@goal Together we'll see why a JAX array is a stone-tablet you never edit, feel how that's different from the notebook-style arrays you may know, learn the one move — `.at[i].set(v)` — that "changes" an array by carving a fresh copy, meet the big idea it unlocks (pure functions), dodge the two classic beginner traps, and learn where this rule pays you back later. Every new word gets explained the moment it shows up.

@@@ concept id=c1 tag="The stone tablet" title="A JAX array is a stone tablet — carved, never edited" gotit="Got the tablet"
Let's start with the one idea the whole day rests on. In JAX, when you make a list of numbers — an [[array||A list of numbers stored in a fixed order, like a row of boxes each holding one value. JAX's basic building block.]] — that list is frozen the instant it is born. You can look at the numbers. You can read any one of them. But you can never reach in and *change* one. This property has a name: the array is [[immutable||Cannot be changed after it is created. A JAX array is immutable: its contents are frozen for life.]].

Picture a **stone-tablet** with numbers carved into it. The carving is permanent. You can read the tablet all day long, but you cannot rub out the number in the third slot and carve a new one. It simply is what it was carved to be, forever.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A stone tablet with four carved number slots holding 3, 1, 4, 1. A hand tries to erase the second slot but an eraser is crossed out with a red X, showing you cannot change a carved number. A label reads read yes, edit no."><g font-family="monospace" font-size="11" text-anchor="middle"><text x="260" y="18" fill="#2C2A28" font-size="12">A JAX array = a stone tablet: read it, but never re-carve one slot</text><g transform="translate(120,44)"><rect x="0" y="0" width="280" height="70" rx="10" fill="#E8E2D6" stroke="#8A8172" stroke-width="3"/><rect x="10" y="10" width="60" height="50" rx="4" fill="#D6CEBE" stroke="#8A8172"/><rect x="80" y="10" width="60" height="50" rx="4" fill="#D6CEBE" stroke="#8A8172"/><rect x="150" y="10" width="60" height="50" rx="4" fill="#D6CEBE" stroke="#8A8172"/><rect x="220" y="10" width="60" height="50" rx="4" fill="#D6CEBE" stroke="#8A8172"/><text x="40" y="42" fill="#3A342E" font-size="18">3</text><text x="110" y="42" fill="#3A342E" font-size="18">1</text><text x="180" y="42" fill="#3A342E" font-size="18">4</text><text x="250" y="42" fill="#3A342E" font-size="18">1</text></g><text x="230" y="150" fill="#C93B3B">try to re-carve slot 2 -&gt; not allowed</text><text x="380" y="42" fill="#276b45" font-size="10" text-anchor="start">read: yes</text><text x="380" y="58" fill="#C93B3B" font-size="10" text-anchor="start">edit: no</text></g></svg>
%%%

**What the stone-tablet picture gets right:** once the numbers are carved, they are set — reading is fine, but changing a single slot is off the table, exactly like a JAX array. **Where it breaks down:** a real stone tablet is heavy and slow to re-carve, but JAX is clever behind the scenes — making a "fresh tablet" is often fast and cheap, so you don't pay a stone-mason's price every time.

#### So what CAN you do?
Plenty. You can read the whole array. You can read one slot, like "what number is in position 2?" You can do math that produces a *brand-new* array — add two arrays, double every number, take an average. All of that is welcome. The single thing you cannot do is reach into an existing array and overwrite one of its numbers in place. Hold that line clearly, because the next concept shows you the *old* way that breaks this rule — the way you may already know.

@@@ concept id=c2 tag="The old way" title="The old way: a whiteboard you scribble on" gotit="Got the contrast"
To feel why the stone-tablet rule matters, let's look at the tool it replaces. Many people first meet arrays through a library called [[NumPy||A popular Python library for arrays of numbers. Its arrays are mutable — you can edit a slot in place, unlike JAX.]]. A NumPy array is the opposite of a stone tablet: it's more like a **whiteboard**. You write numbers on it, and any time you like, you wipe one square and scribble a new number in its place. That "wipe and rewrite one square" move is called editing [[in place||Changing a value inside the same object, without making a new one — like erasing one square on a whiteboard and writing a new number there.]].

%%% svg
<svg viewBox="0 0 520 180" role="img" aria-label="Left: a whiteboard labelled NumPy with four squares holding 3, 1, 4, 1, and the second square is being erased and rewritten as 9, showing in-place editing is allowed. Right: a stone tablet labelled JAX with the same numbers, where re-carving one slot is crossed out. A label contrasts whiteboard edit in place versus tablet carve a fresh copy."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Whiteboard (NumPy) vs stone tablet (JAX)</text><text x="130" y="38" fill="#5E5191">NumPy array = whiteboard</text><g transform="translate(40,48)"><rect x="0" y="0" width="180" height="46" rx="3" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"/><rect x="8" y="8" width="38" height="30" fill="#fff" stroke="#B8AEA2"/><rect x="52" y="8" width="38" height="30" fill="#FFF6D6" stroke="#C99A12" stroke-width="2"/><rect x="96" y="8" width="38" height="30" fill="#fff" stroke="#B8AEA2"/><rect x="140" y="8" width="34" height="30" fill="#fff" stroke="#B8AEA2"/><text x="27" y="29" fill="#3A342E">3</text><text x="71" y="29" fill="#8A6D3B">9</text><text x="115" y="29" fill="#3A342E">4</text><text x="157" y="29" fill="#3A342E">1</text></g><text x="130" y="118" fill="#276b45">wipe &amp; rewrite one square -&gt; OK</text><text x="130" y="134" fill="#6B645E" font-size="9">edit "in place"</text><text x="395" y="38" fill="#8A8172">JAX array = stone tablet</text><g transform="translate(305,48)"><rect x="0" y="0" width="180" height="46" rx="6" fill="#E8E2D6" stroke="#8A8172" stroke-width="2.5"/><rect x="8" y="8" width="38" height="30" fill="#D6CEBE" stroke="#8A8172"/><rect x="52" y="8" width="38" height="30" fill="#D6CEBE" stroke="#8A8172"/><rect x="96" y="8" width="38" height="30" fill="#D6CEBE" stroke="#8A8172"/><rect x="140" y="8" width="34" height="30" fill="#D6CEBE" stroke="#8A8172"/><text x="27" y="29" fill="#3A342E">3</text><text x="71" y="29" fill="#3A342E">1</text><text x="115" y="29" fill="#3A342E">4</text><text x="157" y="29" fill="#3A342E">1</text></g><text x="395" y="118" fill="#C93B3B">can't re-carve one slot</text><text x="395" y="134" fill="#6B645E" font-size="9">carve a fresh copy instead</text></g></svg>
%%%

**What the whiteboard picture gets right:** a NumPy array really does let you erase and rewrite any square whenever you want — quick edits in the same spot. **Where it breaks down:** a whiteboard has no memory of what was there before, and if two people scribble on it at once you get a mess — which is exactly the kind of "surprise edit" JAX wants to make impossible.

#### The move that works on a whiteboard but NOT a tablet
In NumPy, changing one number looks like this — and it just works:

%%% demo id=numpy label="the old NumPy way (works)"
code: arr = np.array([3, 1, 4, 1]);  arr[1] = 9;  print(arr)
out: [3 9 4 1]
take: <b>NumPy edits in place.</b> Writing <code>arr[1] = 9</code> wipes square 1 and writes 9 into the very same array. No new array is made — the whiteboard just changed.
%%%

Now try the *exact same move* on a JAX array. The stone-tablet rule kicks in and JAX politely refuses:

%%% demo id=jaxfail label="the same move on JAX (refused)"
code: import jax.numpy as jnp;  arr = jnp.array([3, 1, 4, 1]);  arr[1] = 9
out: TypeError: JAX arrays are immutable and do not support in-place item assignment. Use x = x.at[i].set(v) instead.
take: <b>JAX blocks it — on purpose.</b> The failure is the whole point: you tried to re-carve one slot of a stone tablet, so JAX stops you and even names the right tool (<code>.at[i].set(v)</code>). That tool is the next concept. Feeling stuck here is completely normal — this is the exact moment every JAX beginner meets.
%%%

That error message is not JAX being difficult — it's JAX handing you the map. The cause is simple: you brought a whiteboard habit (`arr[1] = 9`) to a stone tablet. The cure has a name, and we learn it next.

@@@ concept id=c3 tag="Carve a fresh copy" title="How you 'change' a tablet: .at[i].set(v)" gotit="Got the .at move"
So how *do* you get a JAX array with a different number in it? You don't edit the old one — you **carve a fresh copy** that is the same everywhere except the slot you wanted to change. JAX gives you one clean move for this, and it reads almost like a sentence: `arr.at[1].set(9)` means "take `arr`, look **at** slot 1, and give me a new array where that slot is **set** to 9."

Think of a busy office with a **photocopier**. You have a form (the old tablet) and you want to change one line. You don't scribble on the original. You feed it into the copier, and out comes a fresh copy with that one line different — the original still sitting untouched in your hand. That's exactly what `.at[i].set(v)` does: fresh copy out, original unchanged.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="A stone tablet with numbers 3, 1, 4, 1 goes into a copier labelled .at slot 1 set 9. Out comes a fresh new tablet with 3, 9, 4, 1, while the original tablet on the left is unchanged and still shows 3, 1, 4, 1."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">.at[1].set(9) = a fresh copy with slot 1 changed; original untouched</text><text x="70" y="40" fill="#6B645E">original (kept)</text><g transform="translate(20,50)"><rect x="0" y="0" width="110" height="40" rx="6" fill="#E8E2D6" stroke="#8A8172" stroke-width="2"/><text x="18" y="26" fill="#3A342E">3</text><text x="42" y="26" fill="#3A342E">1</text><text x="66" y="26" fill="#3A342E">4</text><text x="90" y="26" fill="#3A342E">1</text></g><g transform="translate(180,40)"><rect x="0" y="0" width="120" height="70" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="60" y="32" fill="#C93B3B" font-size="10">.at[1]</text><text x="60" y="50" fill="#C93B3B" font-size="10">.set(9)</text><text x="60" y="86" fill="#9A938A" font-size="9">the copier</text></g><text x="150" y="74" fill="#9A5A12" font-size="14">-&gt;</text><text x="315" y="74" fill="#9A5A12" font-size="14">-&gt;</text><text x="450" y="40" fill="#276b45">fresh copy (new)</text><g transform="translate(340,50)"><rect x="0" y="0" width="120" height="40" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="18" y="26" fill="#1a5c38">3</text><text x="46" y="26" fill="#1a5c38" font-size="13">9</text><text x="74" y="26" fill="#1a5c38">4</text><text x="102" y="26" fill="#1a5c38">1</text></g><text x="400" y="132" fill="#276b45" font-size="9">only slot 1 differs</text><text x="70" y="132" fill="#6B645E" font-size="9">still 3, 1, 4, 1</text></g></svg>
%%%

**What the photocopier picture gets right:** you get a new document with one line changed, and the original is left exactly as it was — one input, one fresh output. **Where it breaks down:** a photocopier always copies the *whole* page onto new paper, but JAX is smarter and often reuses memory behind the scenes, so "carve a fresh copy" is usually much cheaper than photocopying a whole ream.

#### Reading it left to right
The move comes in two little parts that you always use together. First `.at[i]` picks the slot you care about — it does nothing on its own, it just *points* at slot `i`. Then `.set(v)` says what the new value there should be. Watch the same update happen and confirm the original stays put:

%%% demo id=atset label="predict, then run"
code: arr = jnp.array([3, 1, 4, 1]);  new = arr.at[1].set(9);  print(arr, new)
out: [3 1 4 1]   [3 9 4 1]
take: <b>Two arrays now exist.</b> <code>arr</code> is untouched — still <code>[3 1 4 1]</code>. <code>new</code> is the fresh copy with slot 1 set to 9. The stone tablet was never re-carved; a second tablet was made.
%%%

#### The whole family, one idea
`.set` writes a new value, but the same `.at[i]` pointer works with other verbs too. `.at[i].add(5)` gives a fresh copy with 5 added to slot `i`. `.at[i].get()` just reads slot `i` (reading is always fine). They all share one shape — **point at a slot, then say what to do — and every one hands you back a new array.** (There are a few more verbs like `.multiply`; you don't need them today, just know the pattern.)

%%% svg
<svg viewBox="0 0 520 150" role="img" aria-label="Three rows showing the .at family. Row one: .at slot 1 set 9 makes a new array with slot 1 set to 9. Row two: .at slot 1 add 5 makes a new array with slot 1 increased by 5. Row three: .at slot 1 get just reads slot 1. All rows point at the same slot then say what to do, and set and add return a brand new array."><g font-family="monospace" font-size="11"><text x="260" y="16" text-anchor="middle" fill="#2C2A28" font-size="12">One pattern: point at a slot (.at[i]) then say what to do then get a NEW array</text><text x="20" y="52" fill="#5E5191">.at[1].set(9)</text><text x="185" y="52" fill="#6B645E">-&gt;</text><text x="210" y="52" fill="#276b45">new array, slot 1 = 9</text><text x="20" y="86" fill="#5E5191">.at[1].add(5)</text><text x="185" y="86" fill="#6B645E">-&gt;</text><text x="210" y="86" fill="#276b45">new array, slot 1 += 5</text><text x="20" y="120" fill="#5E5191">.at[1].get()</text><text x="185" y="120" fill="#6B645E">-&gt;</text><text x="210" y="120" fill="#6B645E">just reads slot 1 (no new array needed)</text></g></svg>
%%%

You've just learned the single most important habit in JAX: to change a number, don't edit — **carve a fresh copy** with `.at[i].set(v)`. Take a victory lap. Next we meet the big idea this habit unlocks — pure functions — and then the two traps beginners fall into around the move.

@@@ concept id=c4 tag="Pure functions" title="The big prize: pure functions (no surprise edits)" gotit="Got pure functions"
Now the exciting part — the reason the stone-tablet rule is worth having at all. Because a JAX array can never be edited in place, the little machines you write to work on arrays — [[functions||A small named recipe: give it some inputs, it gives you back an output. Like a vending machine — same button, same snack.]] — get a wonderful property. When you feed the same numbers in, you always get the same numbers out, and the function *never secretly changes the things you handed it*. A function like that has a name: it is a [[pure function||A function that (1) gives the same output every time for the same input, and (2) never secretly changes its inputs or anything outside itself. Predictable, no surprises.]].

Picture a **vending machine**. You press B4 and out drops a chocolate bar. Press B4 again — same chocolate bar. It never gives you a different snack for the same button, and it never reaches back into your pocket and swaps your coins for buttons. Same input, same output, and nothing of yours gets secretly changed. That's a pure function. A machine that sometimes gave you chips instead, or quietly emptied your wallet, would be *impure* — full of surprises.

%%% svg
<svg viewBox="0 0 520 185" role="img" aria-label="Left, a pure vending machine: pressing B4 always drops the same chocolate bar, and the customer's coins are untouched. Right, an impure machine: pressing B4 gives a different snack each time and secretly grabs extra coins from the customer's pocket, marked with a red warning."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">Pure = same input, same output, your stuff untouched</text><text x="130" y="38" fill="#276b45">PURE machine</text><g transform="translate(70,48)"><rect x="0" y="0" width="120" height="80" rx="6" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="60" y="24" fill="#1a5c38" font-size="9">press B4</text><text x="60" y="44" fill="#9A5A12">-&gt;</text><text x="60" y="62" fill="#1a5c38" font-size="9">choc bar</text><text x="60" y="74" fill="#1a5c38" font-size="8">every time</text></g><text x="130" y="150" fill="#276b45" font-size="9">coins untouched, same snack</text><text x="390" y="38" fill="#C93B3B">IMPURE machine</text><g transform="translate(330,48)"><rect x="0" y="0" width="120" height="80" rx="6" fill="#FDECEC" stroke="#C93B3B" stroke-width="2"/><text x="60" y="24" fill="#8a3b3b" font-size="9">press B4</text><text x="60" y="44" fill="#9A5A12">-&gt;</text><text x="60" y="62" fill="#8a3b3b" font-size="9">chips? gum?</text><text x="60" y="74" fill="#8a3b3b" font-size="8">who knows</text></g><text x="390" y="150" fill="#C93B3B" font-size="9">secretly grabs extra coins</text><text x="390" y="164" fill="#8a3b3b" font-size="8">surprise side-effects</text></g></svg>
%%%

**What the vending-machine picture gets right:** a pure function is boring in the best way — press the same button, get the same thing, and none of your other stuff changes. That predictability is the whole point. **Where it breaks down:** a vending machine can run out of stock, but a pure math function never "runs out" — for the same input it gives the same output every single time, with no low-stock surprises.

#### See pure vs impure with your own eyes
Here is a **pure** function. Give it an array, it hands back a *new* array (double every number) and leaves your original exactly as it was. Run it twice with the same input — you get the same answer both times:

%%% demo id=pure label="a PURE function"
code: def double(x): return x * 2
code: a = jnp.array([3, 1, 4]);  print(double(a), double(a), a)
out: [6 2 8]   [6 2 8]   [3 1 4]
take: <b>Same input, same output, input untouched.</b> Both calls give <code>[6 2 8]</code>, and the original <code>a</code> is still <code>[3 1 4]</code>. Nothing was secretly changed — that's a pure function.
%%%

Now an **impure** function — the kind of thing you'd write on a mutable whiteboard. It reaches in and edits its input, so calling it changes `a` behind your back, and the second call gives a *different* answer:

%%% demo id=impure label="an IMPURE function (NumPy whiteboard)"
code: def double_inplace(x): x *= 2; return x   # edits its input!
code: a = np.array([3, 1, 4]);  print(double_inplace(a), double_inplace(a), a)
out: [6 2 8]   [12 4 16]   [12 4 16]
take: <b>Surprise!</b> The second call returns <code>[12 4 16]</code>, not <code>[6 2 8]</code> — because the first call secretly changed <code>a</code>. Same button, different snack. This is exactly the kind of hidden edit JAX's stone-tablet rule makes impossible.
%%%

#### Why you should care
Immutability *forces* your JAX functions to be pure — they cannot edit their inputs even if they wanted to. And pure, predictable functions are the ground JAX's superpowers stand on. Because a JAX function never surprises the system with a hidden edit, JAX can safely speed it up, run it on a whole batch at once, and even do the calculus for training — all automatically. You'll meet those tools in the coming days; today just hold the shape of the prize: **frozen arrays → pure functions → JAX magic.**

@@@ concept id=c5 tag="Trap: catch the copy" title="The rebind trap — catch the fresh copy!" gotit="Got the rebind fix"
Here's the trap that catches almost everyone on day one, and it's a sneaky one because JAX does *not* error — it just quietly does nothing useful. Remember: `.at[1].set(9)` makes a **fresh** array and hands it to you. If you don't *catch* that returned array in a variable, it floats away and is lost. Your original stays exactly as it was, and you're left scratching your head wondering why "nothing changed."

Picture printing a document. You hit "print," a fresh copy slides out of the printer — but if you **walk away without picking it up**, the copy sits in the tray and you go home with only your unchanged original. The update happened; you just didn't grab it.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="Two paths. Top path labelled wrong: arr.at slot 1 set 9 makes a fresh copy but it is dropped on the floor and lost, so arr is still 3, 1, 4, 1. Bottom path labelled right: arr equals arr.at slot 1 set 9 catches the fresh copy into arr, so arr becomes 3, 9, 4, 1."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">You must CATCH the fresh copy in a variable</text><g transform="translate(0,30)"><text x="120" y="14" fill="#C93B3B">wrong -- copy dropped</text><text x="115" y="40" fill="#6B645E" font-size="10">arr.at[1].set(9)</text><text x="250" y="40" fill="#9A5A12">-&gt;</text><rect x="275" y="22" width="90" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-dasharray="3,2"/><text x="320" y="40" fill="#276b45" font-size="9">fresh copy...</text><text x="430" y="40" fill="#C93B3B">...dropped</text><text x="260" y="62" fill="#C93B3B" font-size="10">arr is STILL [3 1 4 1] -- nothing changed</text></g><g transform="translate(0,108)"><text x="120" y="14" fill="#276b45">right -- copy caught</text><text x="135" y="40" fill="#6B645E" font-size="10">arr = arr.at[1].set(9)</text><text x="270" y="40" fill="#9A5A12">-&gt;</text><rect x="295" y="22" width="120" height="28" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="355" y="40" fill="#1a5c38" font-size="9">caught into arr</text><text x="260" y="62" fill="#276b45" font-size="10">arr is now [3 9 4 1]</text></g></g></svg>
%%%

**What the printer picture gets right:** the copy really was made — the only mistake was not picking it up, so you kept the old one. **Where it breaks down:** a printed page waits patiently in the tray, but a dropped JAX array vanishes right away — the computer throws it out the instant nothing is holding onto it.

#### See the trap, then the fix
Watch the wrong way first — the classic "why didn't it change?!" moment:

%%% demo id=drop label="the trap (silent no-op)"
code: arr = jnp.array([3, 1, 4, 1]);  arr.at[1].set(9);  print(arr)
out: [3 1 4 1]
take: <b>Nothing changed — and no error.</b> The fresh copy was made and instantly dropped because we didn't save it. <code>arr</code> is still the original tablet. This silent "no-op" is the #1 beginner surprise.
%%%

The fix is one word: put the result back into a variable — usually the same name. "Catch the copy":

%%% demo id=catch label="the fix (rebind)"
code: arr = jnp.array([3, 1, 4, 1]);  arr = arr.at[1].set(9);  print(arr)
out: [3 9 4 1]
take: <b>Now it "changed."</b> Writing <code>arr = arr.at[1].set(9)</code> throws away the name pointing at the old tablet and points <code>arr</code> at the fresh copy. The old tablet still existed for a moment — we just stopped caring about it.
%%%

!!! c-warn ⚠️
<b>Failure mode (silent):</b> writing <code>arr.at[i].set(v)</code> without <code>arr =</code> in front changes <b>nothing</b> and raises <b>no error</b> — the update is computed and thrown away. <b>Remedy:</b> always <b>rebind</b> — write <code>arr = arr.at[i].set(v)</code> so the name points at the fresh copy.
!!!

The rule of thumb is short and worth memorizing: **every `.at[...].set(...)` should sit on the right of an `=`.** If it doesn't, you dropped the copy.

@@@ concept id=c6 tag="The real limit" title="The honest limit: don't fight the tablet in a loop" gotit="Got the limit"
Time for the honest part — being clear about what this rule *costs* is what separates a real engineer from someone who just memorized a trick. The stone-tablet rule has one genuine limitation: because you can never edit in place, there is **no cheap way to poke one number at a time, over and over, in a tight loop.** Each `.at[i].set(v)` conceptually hands you a whole fresh tablet, so filling a big array one slot at a time with a Python `for` loop means carving thousands of tablets in a row — the slow, wrong pattern.

Imagine you need to write a 500-page book. You *could* carve each page onto its own stone tablet, one at a time, in order — but that's a mountain of separate slabs and a very long day. There's almost always a better tool for "do the same small thing to every item." Reaching for a one-slot-at-a-time loop is a sign you're fighting the tablet instead of working with it.

%%% svg
<svg viewBox="0 0 520 165" role="img" aria-label="Left, labelled slow and wrong: a long line of many separate stone tablets each carved one at a time in a Python for loop, with a tired face. Right, labelled the JAX way: one big operation stamps all slots at once, shown as a single wide tablet filled in one go, with a happy face. An arrow points from the loop toward later tools vmap and scan."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">One slot at a time in a loop = fighting the tablet</text><text x="130" y="38" fill="#C93B3B">slow: Python loop of .at[i].set(v)</text><g transform="translate(30,48)"><rect x="0" y="0" width="30" height="34" rx="3" fill="#E8E2D6" stroke="#8A8172"/><rect x="42" y="0" width="30" height="34" rx="3" fill="#E8E2D6" stroke="#8A8172"/><rect x="84" y="0" width="30" height="34" rx="3" fill="#E8E2D6" stroke="#8A8172"/><rect x="126" y="0" width="30" height="34" rx="3" fill="#E8E2D6" stroke="#8A8172"/><rect x="168" y="0" width="30" height="34" rx="3" fill="#E8E2D6" stroke="#8A8172"/><text x="100" y="24" fill="#8A8172" font-size="9">... slab after slab ...</text></g><text x="130" y="108" fill="#6B645E" font-size="9">carve, copy, carve, copy... slow</text><text x="400" y="38" fill="#276b45">fast: one bulk operation</text><g transform="translate(300,48)"><rect x="0" y="0" width="200" height="34" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="2"/><text x="100" y="23" fill="#1a5c38" font-size="10">all slots filled at once</text></g><text x="400" y="108" fill="#276b45" font-size="9">later tools: vmap, scan</text><text x="260" y="150" fill="#9A938A">the loop still works -- it's just the wrong tool for "same thing to every item"</text></g></svg>
%%%

**What the book-carving picture gets right:** doing one identical job thousands of times, one slab at a time, is exactly the kind of task that begs for a better tool. **Where it breaks down:** you *can* still write the loop and it will run correctly — it's not forbidden like `arr[i]=x` was, it's just slow; the "limit" is about speed and style, not about the answer being wrong.

#### Why the limit exists — and why you'll be glad
Here's the beautiful twist. That same immutability rule — no surprise edits, ever — is exactly what made your functions **pure** two concepts ago, and pure functions are precisely what let JAX do its magic later. Because a JAX function never secretly changes its inputs, JAX can safely rearrange, speed up, and even do calculus on your code without breaking it. So the "limit" and the "superpower" are the same rule seen from two sides.

!!! c-info 🧭
<b>Optional (skippable) — where the loop limit gets fixed.</b> You will not hand-loop over slots for long. When you want to "do the same thing to every item," you'll reach for <b>vmap</b> (auto-run one function across a whole batch) instead of a Python loop. When you truly must build an array step by step (each step depends on the last), the right tool is <b>scan</b>, which folds a loop into one fast operation. And the reasons immutability is worth it at all — <b>jit</b> (compiles your code so it runs fast), <b>grad</b> (does the calculus for training automatically), and treating whole model states as frozen bundles called <b>pytrees</b> — are the next lessons in this module. Today you only need the habit; those tools are what the habit unlocks.
!!!

Here's the same trade seen as a scale — the one rule you give up, and the three gifts you get for it:

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A balance scale. On the left pan sits one small weight labelled give up: no in-place edits. On the right pan sit three larger weights labelled jit for speed, grad for automatic calculus, and vmap for batching. The right pan is heavier, showing the three gifts outweigh the one cost."><g font-family="monospace" font-size="10" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The trade: give up 1 small thing, get 3 big gifts</text><line x1="260" y1="34" x2="260" y2="150" stroke="#8A8172" stroke-width="3"/><line x1="120" y1="52" x2="400" y2="44" stroke="#8A8172" stroke-width="2"/><circle cx="260" cy="34" r="5" fill="#8A8172"/><g transform="translate(70,52)"><line x1="50" y1="0" x2="50" y2="18" stroke="#8A8172"/><rect x="0" y="18" width="100" height="34" rx="4" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="50" y="32" fill="#C93B3B" font-size="9">give up:</text><text x="50" y="45" fill="#8a3b3b" font-size="8">no in-place edits</text></g><g transform="translate(330,44)"><line x1="70" y1="0" x2="70" y2="18" stroke="#8A8172"/><rect x="0" y="18" width="140" height="52" rx="4" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="70" y="33" fill="#1a5c38" font-size="9">jit -&gt; speed</text><text x="70" y="47" fill="#1a5c38" font-size="9">grad -&gt; auto-calculus</text><text x="70" y="61" fill="#1a5c38" font-size="9">vmap -&gt; batching</text></g><text x="260" y="168" fill="#9A938A">the heavier pan wins -- the gifts far outweigh the one rule</text></g></svg>
%%%

You've now met the rule *and* its honest cost — and seen that the cost is really the price of admission for JAX's biggest gifts. That balanced view is exactly how a staff engineer thinks.

@@@ concept id=c7 tag="Recap" title="Today in one page" gotit="Got the recap"
You did it — you've made the single biggest mental shift JAX asks of a newcomer. Here's a clean way to hold the whole day: picture a **swap-not-scribble** rule. When something must "change," you never scribble over the old thing; you make a fresh copy and swap your attention to it. Everything today was that one rule, the pure functions it gives you, plus the two traps around it.

%%% svg
<svg viewBox="0 0 520 175" role="img" aria-label="A five step flow. Step one: a JAX array is a stone tablet, frozen. Step two: to change it, use .at slot i set v which carves a fresh copy. Step three: catch that copy with arr equals arr.at slot i set v, or it is lost. Step four: this gives you pure functions, same input same output. Step five: don't loop one slot at a time, use vmap or scan later. Each step points to the next."><g font-family="monospace" font-size="9" text-anchor="middle"><text x="260" y="16" fill="#2C2A28" font-size="12">The whole day as one path: freeze -&gt; carve copy -&gt; catch it -&gt; pure fns -&gt; don't loop</text><g transform="translate(8,54)"><rect x="0" y="0" width="88" height="58" rx="8" fill="#E8E2D6" stroke="#8A8172" stroke-width="2"/><text x="44" y="26" fill="#3A342E">1 frozen</text><text x="44" y="42" fill="#6B645E" font-size="8">array = tablet</text></g><text x="100" y="86" fill="#9A5A12" font-size="13">-&gt;</text><g transform="translate(116,54)"><rect x="0" y="0" width="88" height="58" rx="8" fill="#FDECEC" stroke="#C93B3B" stroke-width="1.5"/><text x="44" y="26" fill="#C93B3B">2 carve copy</text><text x="44" y="42" fill="#8a3b3b" font-size="8">.at[i].set(v)</text></g><text x="208" y="86" fill="#9A5A12" font-size="13">-&gt;</text><g transform="translate(224,54)"><rect x="0" y="0" width="88" height="58" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="44" y="26" fill="#1a5c38">3 catch it</text><text x="44" y="42" fill="#276b45" font-size="8">arr = arr.at...</text></g><text x="316" y="86" fill="#9A5A12" font-size="13">-&gt;</text><g transform="translate(332,54)"><rect x="0" y="0" width="88" height="58" rx="8" fill="#EAF5EE" stroke="#2D8B55" stroke-width="1.5"/><text x="44" y="26" fill="#1a5c38">4 pure fns</text><text x="44" y="42" fill="#276b45" font-size="8">same in, same out</text></g><text x="424" y="86" fill="#9A5A12" font-size="13">-&gt;</text><g transform="translate(440,54)"><rect x="0" y="0" width="76" height="58" rx="8" fill="#F4F1FB" stroke="#5E5191" stroke-width="1.5"/><text x="38" y="26" fill="#5E5191">5 don't loop</text><text x="38" y="42" fill="#5E5191" font-size="8">vmap/scan later</text></g><text x="260" y="150" fill="#9A938A">immutable = no surprise edits = pure functions = JAX can speed up &amp; differentiate later</text></g></svg>
%%%

The day in six beats:
- **The rule.** A JAX array is [[immutable||frozen after creation; its contents never change]] — a **stone-tablet** you read but never re-carve.
- **The contrast.** NumPy arrays are mutable whiteboards (`arr[i] = x` edits in place). JAX refuses that with a `TypeError` — on purpose.
- **The one move.** To "change" an array, carve a fresh copy: `new = arr.at[i].set(v)`. The original is untouched.
- **The prize.** Because arrays can't be edited in place, your functions become [[pure||same input always gives the same output, and inputs are never secretly changed]] — a vending machine, not a surprise box. That is what JAX's speed and auto-calculus stand on.
- **The two traps.** (1) `arr[i] = x` on a JAX array → `TypeError`; fix = use `.at[i].set(x)`. (2) forgetting `arr =` → silent no-op; fix = rebind, `arr = arr.at[i].set(x)`.
- **The honest limit.** No cheap per-element in-place edits; a tight one-slot loop is the wrong pattern — reach for vmap/scan later.

#### Cheat-sheet · the moves at a glance
%%% table
:: You want to… :: Do this :: Note
Read a slot :: `arr[i]` or `arr.at[i].get()` :: reading is always allowed
Change a slot :: `arr = arr.at[i].set(v)` :: fresh copy; MUST rebind with `arr =`
Add to a slot :: `arr = arr.at[i].add(v)` :: same family, adds instead of sets
Edit in place (NumPy habit) :: `arr[i] = v` :: `TypeError` on JAX — don't
Fill a big array in a loop :: use vmap / scan (later) :: one-slot loops are the slow, wrong pattern
%%%

#### Cheat-sheet · the words you met today
%%% jargon
array | a list of numbers in a fixed order — JAX's basic building block
immutable | cannot be changed after it is created; a JAX array is frozen for life
in place | changing a value inside the same object, without making a new one (the NumPy habit)
NumPy | a popular array library whose arrays ARE mutable — the contrast to JAX
.at[i].set(v) | the JAX move: return a fresh copy of the array with slot i changed to v
pure function | same input always gives the same output, and it never secretly changes its inputs
rebind | catch the fresh copy back into a variable: arr = arr.at[i].set(v)
vmap | (later) auto-run one function across a whole batch instead of a Python loop
scan | (later) fold a step-by-step loop into one fast operation
%%%

That's the whole day. You now think in fresh copies, not in-place edits — the exact mindset the rest of JAX is built on. Next you'll see the same "no hidden surprises" idea applied to *randomness*, where JAX makes you carry your dice around in the open as explicit **keys**.

@@@ quiz id=quiz tag="Quiz" title="Click an answer — instant feedback on each" gotit="answer all four first"
Four quick questions, with instant feedback on each — answer all four to complete today. (If one trips you up, that's a cue to scroll back, not a failing grade.)
%%% quiz
q: What does it mean that a JAX array is "immutable"? | a:2 | It can only hold whole numbers | It must be small | Once created, its contents can never be changed in place | It is always stored on a GPU | fb: Immutable means frozen for life — like a carved stone tablet. You can read it, but you can never reach in and overwrite one of its numbers.
q: You run arr[1] = 9 on a JAX array. What happens, and what should you do instead? | a:2 | It works just like NumPy | It silently does nothing | It raises a TypeError; use arr = arr.at[1].set(9) instead | It deletes the array | fb: JAX forbids in-place item assignment and raises a TypeError. The named fix is the functional update arr = arr.at[1].set(9), which returns a fresh copy.
q: What makes a function "pure"? | a:2 | It uses no numbers | It always returns zero | Same input always gives the same output, and it never secretly changes its inputs | It runs on a GPU | fb: A pure function is like a vending machine: same button, same snack, and it never reaches into your pocket. Immutability forces JAX functions to be pure, which is what lets JAX speed them up and differentiate them.
q: You write arr.at[1].set(9) but forget the "arr =" in front. What happens? | a:1 | It errors loudly | Nothing changes and there is no error — the fresh copy is dropped | It edits arr in place | It doubles every number | fb: This is the silent no-op trap. .at[1].set(9) makes a fresh copy, but with no arr = to catch it, the copy is thrown away and arr is unchanged. Always rebind.
%%%

@@@ produce id=produce tag="Produce" title="Feel the stone tablet with your own hands" gotit="Done"
Time to feel today's rule for yourself. You'll try the forbidden move, watch JAX refuse it, do it the right way, and then check for yourself that a JAX function is *pure* — same input, same output, original untouched. **Predict first:** after `new = arr.at[1].set(9)`, what does the *original* `arr` print — the old numbers or the new ones? Then run it and **watch** which one changed. Pick one path.

#### Option A · write it yourself
Create `sessions/m07-thinking-in-jax/day-01-jax-immutability/experiment.py`. Start with `import jax.numpy as jnp` and `arr = jnp.array([3, 1, 4, 1])`. **Print `arr` at every step so you can see it happen.** (1) Try the NumPy habit `arr[1] = 9` inside a `try/except` and **print** the error message JAX gives — notice it names `.at[i].set()`. (2) Do it the right way: `new = arr.at[1].set(9)`; print both `arr` and `new` and **observe** that `arr` is unchanged while `new` has the 9. (3) Show the silent trap: run `arr.at[1].set(9)` with **no** `new =` in front, print `arr`, and notice nothing changed. Then fix it with `arr = arr.at[1].set(9)` and print again. (4) Write a small pure function `def double(x): return x * 2`, call it twice on the same array, and **notice** you get the same answer both times AND the original array is untouched — that's purity. Run with `python3 sessions/m07-thinking-in-jax/day-01-jax-immutability/experiment.py`.

#### Option B · let Claude build it, then read it
Copy the prompt below back into Claude Code. It triggers the <b>frontier-experiment-lab</b> skill, which creates the file, writes the code, and runs it for you.

%%% prompt id=pp label="triggers <b>frontier-experiment-lab</b>"
Use /frontier-experiment-lab to build my Week 1 Day 1 (JAX immutability) artifact.

Create sessions/m07-thinking-in-jax/day-01-jax-immutability/experiment.py that, with a comment on each step and printing arr at every step:
1. import jax.numpy as jnp and make arr = jnp.array([3, 1, 4, 1]); print arr.
2. In a try/except, attempt the in-place NumPy habit arr[1] = 9 and PRINT the exception message JAX raises (it should be a TypeError mentioning .at[i].set()).
3. Do the functional update the right way: new = arr.at[1].set(9); print BOTH arr and new — arr must still be [3 1 4 1] (unchanged) and new must be [3 9 4 1]. This proves immutability: the original was never edited.
4. Show the silent rebind trap: call arr.at[1].set(9) WITHOUT catching it, then print arr — it is still [3 1 4 1], no error. Then rebind arr = arr.at[1].set(9) and print arr — now [3 9 4 1].
5. Show a PURE function: def double(x): return x * 2. Call double on a jnp array twice and print both results plus the original — the two results must match and the original must be unchanged, proving the function is pure (same input, same output, no secret edits).
6. Also show arr.at[1].add(5) returns a fresh copy with slot 1 increased, leaving the original untouched.
Then run it and paste the output at the bottom as a comment.
%%%

#### What you should see (refuse, carve a fresh copy, stay pure)
- The in-place move `arr[1] = 9` raises a `TypeError` — and the message itself points you to `.at[i].set()`. The error IS the lesson.
- The right move `new = arr.at[1].set(9)` gives you `new = [3 9 4 1]` while the **original `arr` stays `[3 1 4 1]`** — proof the stone tablet was never re-carved.
- The silent trap: `arr.at[1].set(9)` with no `arr =` leaves `arr` **unchanged and throws no error** — the classic no-op. Adding `arr =` fixes it.
- The pure function `double` gives the **same answer both times** and leaves its input **untouched** — same input, same output, no surprises.

!!! c-info 📓
<b>5-minute research log · 5 分钟研究笔记:</b> 关掉页面前，用自己的话写三行 — before you close the tab, write three lines in `sessions/m07-thinking-in-jax/day-01-jax-immutability/log.md`: (1) in one sentence, what "a JAX array is immutable" means, using the stone-tablet picture; (2) the one move that "changes" an array, and why you must write `arr =` in front of it; (3) in one sentence, what makes a function "pure," and why immutability gives you purity for free.

@@@ fin
