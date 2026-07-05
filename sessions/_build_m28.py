#!/usr/bin/env python3
"""Builder for M28 — Formal Methods: TLA+ & Property-Based Testing (author from scratch).
Teaches: state machines (Init/Next), invariants + TLC model checking, refinement,
Hypothesis property-based testing, and when to use which. Builds on M26 (the verify loop).
6 lessons + review. Run: python3 sessions/_build_m28.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m28")
os.makedirs(OUT, exist_ok=True)
def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — What Is TLA+? ----------------
L("day-01-what-is-tlaplus.html", dict(
  qid="m28-d01-tla", title="Module 28 · Day 1 — What Is TLA+?", nav_title="Spiral · M28 Day 1",
  eyebrow="Module 28 · Verify · Day 1", h1="What Is TLA+?",
  lead="Your M26 loop trusts a test suite to say some code is correct. But a test suite only tries a <b>finite</b> list of inputs. It can never try every input, so it can miss a rare timing or ordering that only shows up once in a million runs. Formal methods take a different path: describe the system as <b>states</b> and <b>allowed moves</b>, then prove a property holds for <em>every</em> possible run. TLA+ is the language you write that description in.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain what a formal method is (states + transitions, checked for all runs), say why testing misses rare interleavings, name what TLA+ is, and walk one tiny state example by hand.",
  prev_href="../index.html", prev_label="Curriculum Map",
  next_href="day-02-init-next.html", next_label="States, Init & Next",
  sections=[
    {"title":"Proving, not just sampling","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> your M26 verification loop — code is proposed, then a test suite (pytest) runs to decide if it is correct. Nothing about formal methods yet.</div></div>
     <h4>The gap a test suite can never close</h4>
     <p>A test suite is useful. It raises your confidence. But it only tries a <strong>finite</strong> list of inputs. "My tests passed" and "this is correct for every input" are two different claims. No amount of extra testing turns the first into the second.</p>
     <p>This gap is widest for systems with <span class="term" data-tip="When two or more parts of a program run at the same time, their steps can happen in many different orders; each order is one interleaving.">concurrency</span> — many parts running at once. The steps of each part can happen in many orders. A bug may appear in only one rare order out of billions, so a test almost never hits it.</p>
     <h4>The other path: describe the system as a machine</h4>
     <p>A <span class="term" data-tip="A way to check a system by describing its states and allowed moves, then proving a property holds for every possible run.">formal method</span> describes a system as two things: a set of <strong>states</strong> (all the situations it can be in) and the <strong>allowed moves</strong> from one state to another. Once you have that, a tool can explore <em>every</em> reachable state and check that a property never breaks.</p>
     <h4>TLA+ is the language for that description</h4>
     <p><span class="term" data-tip="A math-based language for describing exactly what a system does — its states and allowed moves — so a tool can check it.">TLA+</span> (Temporal Logic of Actions) is a language for writing that description. You do not <em>run</em> TLA+ like a program. You write a <span class="term" data-tip="A written description of a system: its variables, its allowed start states, and its allowed moves between states.">spec</span> — a precise description — and then a tool checks it.</p>''',
     "gotit":"Got the idea"},
    {"title":"A chess rulebook, not a few played games","body":
     '''<div class="relate">
       <div class="card"><span class="big">♟️🎮</span><h5>Testing = a few played games</h5><p>You play 100 games of chess and never see the king captured. That is reassuring — but you only played 100 games out of a near-infinite number. A losing trap you never tried is still out there.</p></div>
       <div class="card"><span class="big">📖♟️</span><h5>TLA+ = the rulebook + a checker</h5><p>Instead you write down the rules: what a legal board is, what a legal move is, and the rule "the king is never captured." A checker then explores <em>every</em> reachable board and confirms the rule holds — or hands you the exact move sequence that breaks it.</p></div>
     </div>
     <p><strong>In one line:</strong> testing samples a few runs; TLA+ writes the rules of the system so a tool can check a property across <em>all</em> reachable runs.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: real chess has more legal boards than a computer could ever list, so you could not truly check "all" of them. TLA+ works because you deliberately shrink the system to a small, finite model (few variables, small value ranges). Its "all runs" means all runs <em>of that small model</em>, not the full unbounded system.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Walk one tiny state machine","body":'''<p>Watch a two-state light switch: its states, its allowed moves, and why a test that flips it twice can miss a bug. <strong>Click all three.</strong></p>'''},
    {"title":"How a spec proves a property for all runs","body":
     '''<h4>Step 1 — in words</h4>
     <p>A tiny system is a set of states and a set of allowed moves. A run (also called a <span class="term" data-tip="A sequence of states a system passes through, starting from a legal start state and following allowed moves.">behaviour</span>) is a start state, then a legal move, then another, and so on. A property is a rule that must hold in every state of every run.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       States   = all situations the system can be in<br>
       Init     = the allowed start states<br>
       Next     = the allowed moves (state → next state)<br>
       Reachable = { s : s = Init, or s reached from a reachable state via Next }<br>
       Property holds  ⇔  it is true in EVERY reachable state
     </div></div>
     <h4>Step 3 — worked example (a 3-position counter)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.8rem;line-height:1.8">
       States = {0, 1, 2}     Init: x = 0     Next: x' = (x + 1) mod 3<br>
       reachable from 0:  0 → 1 → 2 → 0 → ...  so Reachable = {0, 1, 2}<br>
       Property "x &lt; 3":  true at 0, true at 1, true at 2  → <span class="hl">holds for every reachable state ✓</span><br>
       Property "x &lt; 2":  true at 0, true at 1, <span class="bad">FALSE at 2</span>  → the checker returns the trace 0 → 1 → 2
     </div></div>
     <p>The checker did not guess. It listed all three reachable states and tested the rule on each. That is the whole idea of exhaustive checking on a small model.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — spec drift.</b> The spec is a <em>separate</em> document from the real code. Over time engineers change the code but forget the spec, so the spec now describes a system that no longer exists. The checker keeps printing "no error" against the stale spec, giving false comfort while the real code has a bug the spec would never have allowed. The fix: treat the spec as a living artifact, review it in the same pull request as the code, and re-check it whenever the design changes.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — exhaustive proof vs. modelling effort.</b> A test suite is cheap to write and runs on the real code, but only samples inputs. A TLA+ spec proves a property for every run of a model, but you must first <em>write</em> that model — a real up-front effort, and it describes an abstraction, not the exact code. You pay modelling time to buy exhaustive coverage of a bounded design. Spend it where a rare bug would be catastrophic (money, safety, data loss), not on low-risk glue code.</div></div>''',
     "gotit":"Got the mechanism"},
    {"title":"The idea, built up","body":'''<p>Here is the whole picture, assembled one piece at a time: states, start state, allowed moves, reachable set, and the property checked on every reachable state. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Write your first tiny spec (on paper)","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m28_first_spec.py</code>. In Python, model the 3-position counter as a set of states and a <code>next(x)</code> move. Compute the reachable set by breadth-first search from the start state, then check a property on every reachable state and print a violating trace if one exists. This is a hand-built model checker — it makes the TLA+ idea concrete.</p>
     <h4>Option B · let Claude build it, then read it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 28 Day 1 artifact.
Create experiments/foundations/m28_first_spec.py: model a tiny state machine (a mod-3 counter: Init x=0, Next x'=(x+1)%3) as Python. Compute the reachable state set via BFS from the start state. Then check two invariants — "x < 3" (holds) and "x < 2" (fails) — over every reachable state, and for a failing invariant print the exact trace of states that reaches the violation. Add a comment explaining that this exhaustive check is the core idea TLA+ and the TLC model checker automate.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m28/day-01-log.md</code>: what is one bug class a test suite could miss but an exhaustive state check would catch, and why?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "states":{"btn":"① the states","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># a light switch has exactly two states</span>
States = {<span class="str">"OFF"</span>, <span class="str">"ON"</span>}
Init   = <span class="str">"OFF"</span>   <span class="dim"># the switch starts OFF</span>''',
      "take":"<b>①  Name every state.</b> The whole system lives in one of a small, listed set of situations. Here: just OFF and ON."},
    "moves":{"btn":"② the allowed moves","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># the only legal move is to flip</span>
Next: <span class="kw">if</span> s == <span class="str">"OFF"</span>: s' = <span class="str">"ON"</span>
      <span class="kw">if</span> s == <span class="str">"ON"</span>:  s' = <span class="str">"OFF"</span>
<span class="dim"># s' means "the next state"</span>''',
      "take":"<b>②  List every allowed move.</b> A move takes one state to a next state. Only flips are legal; jumping OFF→OFF is not a move here."},
    "run":{"btn":"③ a run vs. a test","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># one run (behaviour): a chain of states</span>
OFF → ON → OFF → ON → ...
<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># a test flips twice, sees OFF→ON→OFF, says "fine"</span>
<span class="dim"># but it never checked what happens after 3 flips, or</span>
<span class="dim"># two flips racing at once. A spec checks ALL runs.</span>''',
      "take":"<b>③  A run is a chain of states.</b> A test walks one short chain. A formal check walks every reachable chain, so a rare order can't hide."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='120' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='180' y='36' fill='#1F6280'>state 0</text><rect x='290' y='16' width='120' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='350' y='36' fill='#1F6280'>state 1</text></g></svg>",
     "note":"<b>Piece 1 — the states.</b> First you name every situation the system can be in. A tiny counter here has states 0, 1, 2 (two shown). The full list of situations is the system's <b>state space</b>."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><circle cx='90' cy='30' r='9' fill='#2D8B55' stroke='#1a5c38' stroke-width='2'/><rect x='120' y='16' width='150' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='195' y='36' fill='#1a5c38'>Init: x = 0</text><text x='360' y='36' fill='#5A544E'>the allowed start</text></g></svg>",
     "note":"<b>Piece 2 — the start (Init).</b> <code>Init</code> lists which states the system may start in. Here it starts at x = 0. Every run must begin from a legal start state."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='60' y='22' width='90' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='105' y='40' fill='#1F6280'>x=0</text><rect x='215' y='22' width='90' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='40' fill='#1F6280'>x=1</text><rect x='370' y='22' width='90' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='415' y='40' fill='#1F6280'>x=2</text><line x1='150' y1='36' x2='213' y2='36' stroke='#7C6DAA' stroke-width='2' marker-end='url(#a1)'/><line x1='305' y1='36' x2='368' y2='36' stroke='#7C6DAA' stroke-width='2' marker-end='url(#a1)'/><path d='M415 22 C 470 0, 470 60, 105 52' fill='none' stroke='#7C6DAA' stroke-width='2' marker-end='url(#a1)'/><defs><marker id='a1' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0 0 L6 3 L0 6 z' fill='#7C6DAA'/></marker></defs></g></svg>",
     "note":"<b>Piece 3 — the moves (Next).</b> <code>Next</code> is the rule x' = (x+1) mod 3. Each arrow is one allowed move. Following arrows from the start traces out a run."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>Reachable = { 0, 1, 2 }  (all three)</text></g></svg>",
     "note":"<b>Piece 4 — the reachable set.</b> Starting at 0 and following moves, the system can reach 0, 1, and 2 — and nothing else. This finite set is exactly what the checker will inspect."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='60' y='14' width='120' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='120' y='31' fill='#1a5c38'>x&lt;3 at 0 ✓</text><rect x='200' y='14' width='120' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='260' y='31' fill='#1a5c38'>x&lt;3 at 1 ✓</text><rect x='340' y='14' width='120' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='400' y='31' fill='#1a5c38'>x&lt;3 at 2 ✓</text><text x='260' y='58' fill='#1a5c38'>holds in EVERY reachable state → proven for this model</text></g></svg>",
     "note":"<b>Piece 5 — check the property on every state.</b> The property is tested at each reachable state, not sampled. If it holds everywhere the model proves it; if it fails at one state, the checker returns the exact trace that reaches it."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='40' y='16' width='150' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='115' y='35' fill='#C93B3B'>property x&lt;2</text><rect x='220' y='16' width='250' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='345' y='35' fill='#C93B3B'>counterexample: 0 → 1 → 2</text></g></svg>",
     "note":"<b>The payoff — a trace, not just 'false'.</b> When a property fails, the checker hands you the exact sequence of states that breaks it. That trace is a concrete bug report you can act on — something a passing test can never give you."},
  ],
  quiz=[
    {"q":"1. Why can a passing test suite still miss a real bug?","opts":["tests always test the wrong thing","a test suite only tries a finite list of inputs/orders, so a rare interleaving can go untried","tests run too slowly","tests can only check numbers, not text"],"ans":1,"fb":"Testing samples finitely many runs. A bug that only appears in one rare ordering out of billions is almost never hit — that's the gap formal methods close on a bounded model."},
    {"q":"2. What does a TLA+ spec describe?","opts":["the exact bytes of the compiled program","a system as states plus allowed moves between them, so a tool can check a property over all runs","the network latency of a service","the training data of a model"],"ans":1,"fb":"A spec is states + allowed transitions. From that a checker explores every reachable state to verify a property."},
    {"q":"3. For the counter Init: x=0, Next: x'=(x+1) mod 3, which is the reachable set?","opts":["{0}","{0, 1}","{0, 1, 2}","{0, 1, 2, 3}"],"ans":2,"fb":"From 0 you reach 1, then 2, then (2+1) mod 3 = 0 again. So exactly {0, 1, 2}. There is no state 3 because of the mod 3."},
    {"q":"4. For that same counter, does the property 'x < 2' hold in every reachable state?","opts":["yes, x is always below 2","no — it is false at the reachable state x = 2, and the checker returns the trace 0 → 1 → 2","only if you start at x = 1","the checker can't decide"],"ans":1,"fb":"x = 2 is reachable and 2 < 2 is false, so the property fails. The checker returns the exact path 0 → 1 → 2 as the counterexample."},
  ],
  fin={"em":"📐","h3":"Day 1 complete — you can read the map!",
       "p":"You now know what a formal method is: describe a system as states plus allowed moves, then check a property across every reachable state — not just a sampled few. You saw why testing misses rare orderings, what TLA+ is, and walked a tiny counter by hand. Next: <b>Day 2</b> — how a spec names its variables and writes Init and Next to define a state machine precisely."},
))

# ---------------- Day 2 — States, Init & Next ----------------
L("day-02-init-next.html", dict(
  qid="m28-d02-initnext", title="Module 28 · Day 2 — States, Init & Next", nav_title="Spiral · M28 Day 2",
  eyebrow="Module 28 · Verify · Day 2", h1="States, Init & Next",
  lead="Yesterday you saw the shape of a formal spec. Today you write one properly. Every TLA+ spec has three parts: the <b>variables</b> that hold the state, an <b>Init</b> rule saying which start states are legal, and a <b>Next</b> rule saying which moves are legal. Those three pieces together define a state machine — a precise map of every situation the system can be in and every step it can take.",
  goal="<b>🎯 By the end, you'll be able to:</b> model a system as variables, write Init (legal starts) and Next (legal moves), read a behaviour as a sequence of states, and spot when a loose Next allows impossible steps.",
  prev_href="day-01-what-is-tlaplus.html", prev_label="What Is TLA+?",
  next_href="day-03-invariants-tlc.html", next_label="Invariants & Model Checking (TLC)",
  sections=[
    {"title":"Three parts of every spec","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 — a formal method describes a system as states plus allowed moves, and TLA+ is the language you write that in.</div></div>
     <h4>Variables hold the state</h4>
     <p>You choose a few <span class="term" data-tip="Named values that together hold everything the system needs to remember; the current values of all variables are the current state.">variables</span>. Their current values, taken together, ARE the current state. A door lock might use one variable, <code>locked</code>, that is either <code>TRUE</code> or <code>FALSE</code>. A counter might use one variable <code>x</code>.</p>
     <h4>Init says where you may start</h4>
     <p><span class="term" data-tip="A rule that lists the legal starting values of the variables; every run must begin from a state that satisfies Init.">Init</span> is a rule the start state must satisfy. For the lock: <code>Init == locked = TRUE</code> means every run begins locked.</p>
     <h4>Next says which moves are legal</h4>
     <p><span class="term" data-tip="A rule relating the current state to the next state; it lists every legal step the system may take.">Next</span> relates the current values to the next values. TLA+ writes the next value of a variable with a prime: <code>locked'</code> means "the value of <code>locked</code> after this step." A move is legal only if it matches Next.</p>
     <p>Variables + Init + Next = a <span class="term" data-tip="A system defined by its states and the allowed moves between them; formal methods reason about all its behaviours.">state machine</span>. That is the whole object a checker will explore.</p>''',
     "gotit":"Got the three parts"},
    {"title":"A board game's setup and move rules","body":
     '''<div class="relate">
       <div class="card"><span class="big">🎲🏁</span><h5>Init = the setup rule</h5><p>Before a board game starts, the rulebook says where every piece goes: "all tokens on START, score 0." That is Init — the one legal way to begin.</p></div>
       <div class="card"><span class="big">➡️🎲</span><h5>Next = the move rules</h5><p>Then the rulebook lists what a legal turn is: "roll the die, advance that many squares." Any turn that does not follow a rule is illegal. That is Next — the allowed steps.</p></div>
     </div>
     <p><strong>In one line:</strong> the variables are the board's current arrangement, Init is the setup rule, and Next is the list of legal turns.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a board game usually has one player taking clearly ordered turns. A real spec often has several parts that can each move, so many different move orders are possible from the same state. That branching — many possible next states — is exactly what makes the full run set large, and why the checker must explore it carefully.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Build a lock spec, step by step","body":'''<p>Watch a door lock modelled as one variable, with Init and Next, and see one behaviour unfold. <strong>Click all three.</strong></p>'''},
    {"title":"How Init and Next generate every behaviour","body":
     '''<h4>Step 1 — in words</h4>
     <p>Start from any state Init allows. Apply any move Next allows to get a next state. Repeat forever. Every chain you can build this way is a legal behaviour. The set of states you can reach is the reachable set.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       Init  == locked = TRUE                       <span class="dim"># legal start</span><br>
       Next  == \\/ (locked = TRUE  /\\ locked' = FALSE)   <span class="dim"># unlock</span><br>
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\\/ (locked = FALSE /\\ locked' = TRUE)    <span class="dim"># lock</span><br>
       behaviour: s0 (Init) → s1 → s2 → ...  where each sᵢ → sᵢ₊₁ satisfies Next
     </div><div style="font-size:.8rem;color:var(--text2);margin-top:.4rem"><code>\\/</code> means OR, <code>/\\</code> means AND, <code>'</code> marks the next-state value.</div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.8rem;line-height:1.8">
       start:  locked = TRUE                 <span class="dim"># from Init</span><br>
       move 1: unlock  → locked = FALSE       <span class="dim"># matches the first Next branch</span><br>
       move 2: lock    → locked = TRUE        <span class="dim"># matches the second branch</span><br>
       reachable states: <span class="hl">{ TRUE, FALSE }</span>  — exactly two, and both are legal<br>
       <span class="dim"># with one boolean variable there are only 2 possible states total</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — under-constrained Next.</b> If you forget a condition in Next, it allows moves that can never happen in the real system. Suppose you wrote <code>Next == locked' = FALSE</code> with no guard. Now the model says the lock can go from FALSE straight to FALSE (unlock an already-open lock) and, worse, it can never re-lock — a whole legal behaviour is missing and an impossible one is added. The checker then verifies the wrong machine. The fix: for every move, spell out both the guard (what must be true to take it) AND the full next-state of every variable.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — abstraction level.</b> Model the lock as one boolean and the spec is tiny and fast to check, but it cannot express "wrong PIN entered three times." Add PIN attempts, timers, and users and the model captures more real behaviour — but the state space grows fast and the spec gets harder to write and read. Pick the coarsest model that still contains the failure you care about; extra detail that cannot express your target bug only slows the check and hides the logic.</div></div>''',
     "gotit":"Got Init and Next"},
    {"title":"The state machine, built up","body":'''<p>Here it assembles: the variable, the Init start, each Next move as an arrow, a full behaviour, and the complete reachable set. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Model a turnstile as Init + Next","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m28_turnstile.py</code>. Model a turnstile with one variable <code>state</code> in {LOCKED, UNLOCKED}. Write Init (starts LOCKED) and Next as a list of guarded moves: coin unlocks, push through a locked one is refused, push through an unlocked one re-locks. Enumerate the reachable states by BFS and print each legal move as (from, action, to). Then deliberately drop a guard from one move and show it now lets an impossible transition through.</p>
     <h4>Option B · let Claude build it, then read it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 28 Day 2 artifact.
Create experiments/foundations/m28_turnstile.py: model a turnstile as one variable state in {LOCKED, UNLOCKED}. Define Init (state=LOCKED) and Next as a set of guarded transitions (coin: LOCKED->UNLOCKED; push: UNLOCKED->LOCKED; push while LOCKED: no change / refused). Enumerate reachable states via BFS and print every legal (from, action, to) triple. Then include a second "buggy" Next that omits a guard, and print the extra impossible transitions it wrongly allows, to demonstrate an under-constrained Next.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m28/day-02-log.md</code>: which guard did you drop, and what impossible transition did the under-constrained Next then allow?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "var":{"btn":"① the variable = the state","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">VARIABLE</span> locked   <span class="dim"># one boolean holds the whole state</span>
<span class="dim"># current value of locked IS the current state</span>
locked ∈ {TRUE, FALSE}   <span class="dim"># only two possible states</span>''',
      "take":"<b>①  Variables hold the state.</b> Here one boolean, <code>locked</code>, captures everything the lock needs to remember."},
    "init":{"btn":"② Init: legal start","html":'''<span class="prompt">&gt;&gt;&gt;</span> Init == locked = TRUE
<span class="dim"># every run must begin with the lock LOCKED</span>
<span class="ok">start state fixed: locked = TRUE</span>''',
      "take":"<b>②  Init pins the start.</b> The run may only begin from a state Init allows. Here that is exactly one state: locked."},
    "next":{"btn":"③ Next: legal moves + a run","html":'''<span class="prompt">&gt;&gt;&gt;</span> Next == (locked=TRUE /\\ locked'=FALSE)   <span class="dim"># unlock</span>
              \\/ (locked=FALSE /\\ locked'=TRUE)   <span class="dim"># lock</span>
<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># one behaviour:</span>
TRUE → FALSE → TRUE → FALSE → ...''',
      "take":"<b>③  Next lists legal moves.</b> Each move has a guard (what must hold) and a next value. A behaviour is a chain of states, each step matching Next."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='14' width='220' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='33' fill='#5E5191'>VARIABLE locked ∈ {TRUE, FALSE}</text></g></svg>",
     "note":"<b>Piece 1 — the variable.</b> One boolean holds the whole state. With a single boolean there are only two possible states total, so this model is tiny."},
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='11' text-anchor='middle'><circle cx='90' cy='28' r='9' fill='#2D8B55' stroke='#1a5c38' stroke-width='2'/><rect x='125' y='14' width='230' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='240' y='33' fill='#1a5c38'>Init: locked = TRUE</text></g></svg>",
     "note":"<b>Piece 2 — Init.</b> The run must begin from a state Init allows. Here the lock always starts locked (TRUE)."},
    {"viz":"<svg viewBox='0 0 520 72'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='24' width='130' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='155' y='42' fill='#1F6280'>TRUE</text><rect x='300' y='24' width='130' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='365' y='42' fill='#1F6280'>FALSE</text><path d='M220 30 C 260 8, 262 8, 300 30' fill='none' stroke='#7C6DAA' stroke-width='2' marker-end='url(#a2)'/><path d='M300 46 C 262 66, 260 66, 220 46' fill='none' stroke='#7C6DAA' stroke-width='2' marker-end='url(#a2)'/><text x='260' y='12' fill='#5E5191' font-size='9'>unlock</text><text x='260' y='70' fill='#5E5191' font-size='9'>lock</text><defs><marker id='a2' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0 0 L6 3 L0 6 z' fill='#7C6DAA'/></marker></defs></g></svg>",
     "note":"<b>Piece 3 — Next as arrows.</b> Each guarded move is one arrow: TRUE→FALSE (unlock) and FALSE→TRUE (lock). A move is legal only if it matches one Next branch."},
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='16' width='440' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='35' fill='#9A7208'>behaviour: TRUE → FALSE → TRUE → FALSE → ...</text></g></svg>",
     "note":"<b>Piece 4 — a behaviour.</b> Start from Init, then follow Next moves. A behaviour is the full chain of states a run passes through."},
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>Reachable = { TRUE, FALSE } — both legal</text></g></svg>",
     "note":"<b>Piece 5 — the reachable set.</b> Both states are reachable and both are legal. Variables + Init + Next fully define this machine; the checker now has everything it needs."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='30' y='16' width='210' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='135' y='35' fill='#C93B3B'>loose Next: locked' = FALSE</text><rect x='270' y='16' width='220' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='380' y='35' fill='#C93B3B'>allows FALSE→FALSE, blocks lock</text></g></svg>",
     "note":"<b>The trap — under-constrained Next.</b> Drop the guards and the model both adds an impossible move and removes a real one. The checker then verifies the wrong machine — spell out every guard and every variable's next value."},
  ],
  quiz=[
    {"q":"1. In a TLA+ spec, what makes up the current state?","opts":["the last printed line","the current values of all the variables, taken together","the number of moves so far","the name of the spec file"],"ans":1,"fb":"The state is exactly the current values of every variable. Change any variable and you are in a different state."},
    {"q":"2. What does the prime in locked' mean?","opts":["the variable is a secret","the value of locked in the NEXT state, after this step","the derivative of locked","the first variable declared"],"ans":1,"fb":"A primed variable is its value in the next state. Next relates unprimed (now) to primed (next) values."},
    {"q":"3. A lock has one boolean variable. How many possible states does the model have in total?","opts":["1","2","4","unlimited"],"ans":1,"fb":"One boolean can be TRUE or FALSE, so exactly 2 possible states. Keeping variables few and small is how you keep the state space checkable."},
    {"q":"4. You write Next == locked' = FALSE with no guard. What goes wrong?","opts":["nothing, it's a clean spec","it is under-constrained: it allows an impossible move (FALSE→FALSE) and drops a real one (can never lock again), so the checker verifies the wrong machine","it makes the check faster and more accurate","it locks the file"],"ans":1,"fb":"With no guard and no branch to re-lock, the model both adds impossible behaviour and removes real behaviour. Always give each move a guard and the full next-state of every variable."},
  ],
  fin={"em":"🔧","h3":"Day 2 complete — you can write a state machine!",
       "p":"You now write a spec properly: variables hold the state, Init fixes the legal starts, Next lists the legal moves, and a behaviour is the chain of states they generate. You also saw the under-constrained-Next trap and the abstraction-level trade-off. Next: <b>Day 3</b> — invariants and the TLC model checker, which explores every reachable state to find a violating trace."},
))

# ---------------- Day 3 — Invariants & Model Checking (TLC) ----------------
L("day-03-invariants-tlc.html", dict(
  qid="m28-d03-tlc", title="Module 28 · Day 3 — Invariants & Model Checking", nav_title="Spiral · M28 Day 3",
  eyebrow="Module 28 · Verify · Day 3", h1="Invariants & Model Checking (TLC)",
  lead="You have a state machine. Now you say what must never go wrong and let a tool check it. An <b>invariant</b> is a property that must be true in every reachable state. <b>TLC</b> is the model checker that ships with TLA+: it explores every reachable state, and the moment it finds one where the invariant is false, it hands you the exact path that got there. That path is a bug report you could not get from any test.",
  goal="<b>🎯 By the end, you'll be able to:</b> define an invariant, describe how TLC searches the reachable states, read a counterexample trace, explain state-space explosion, and say why an unbounded model never finishes.",
  prev_href="day-02-init-next.html", prev_label="States, Init & Next",
  next_href="day-04-refinement.html", next_label="Refinement: Spec to Code",
  sections=[
    {"title":"Say what must never break","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 2 — a spec has variables, Init (legal starts), and Next (legal moves), which together define a state machine.</div></div>
     <h4>An invariant is a "must always" rule</h4>
     <p>An <span class="term" data-tip="A property that must be true in every reachable state of the system; if it is ever false, the system has a bug.">invariant</span> is a property that must hold in <em>every</em> reachable state. For a bank account: "balance is never negative." For two threads: "at most one holds the lock." You state the invariant; the tool tries to break it.</p>
     <h4>TLC explores every reachable state</h4>
     <p><span class="term" data-tip="The model checker that comes with TLA+; it walks every reachable state of the model and tests the invariant in each one.">TLC</span> is the model checker. It starts from Init, applies Next to find every next state, then every state after that, and so on — a full search of the reachable set. In each state it checks the invariant.</p>
     <h4>A failure is a trace, not just "false"</h4>
     <p>If TLC finds a state where the invariant is false, it does not just say "failed." It prints a <span class="term" data-tip="The exact sequence of states from a legal start to a state where the invariant is broken; a concrete, minimal-ish bug report.">counterexample trace</span> — the exact chain of states from Init to the broken state. You can replay that chain to understand and fix the bug.</p>''',
     "gotit":"Got invariants"},
    {"title":"A maze solver that tries every path","body":
     '''<div class="relate">
       <div class="card"><span class="big">🌀🔦</span><h5>TLC = a maze robot</h5><p>Drop a robot at the maze entrance. It systematically walks every corridor, marking rooms it has already seen so it never loops forever. It is not lucky or clever — it is thorough.</p></div>
       <div class="card"><span class="big">🚩</span><h5>Invariant = "no room has a trap"</h5><p>Your rule is "no reachable room contains a trap." The robot checks each room as it enters. If it finds a trapped room, it shows you the exact turns from the entrance to that room.</p></div>
     </div>
     <p><strong>In one line:</strong> TLC is a thorough robot that visits every reachable state and checks your invariant in each, returning the exact path to any state that breaks it.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a maze has a fixed number of rooms. A real system's reachable states can explode into billions as you add variables or widen their ranges — the robot would never finish. That is <b>state-space explosion</b>, and it is why you must shrink (bound) the model before checking it.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Run TLC and read a counterexample","body":'''<p>Watch TLC check a bounded counter: a passing invariant, a failing one with a trace, and what happens if you forget to bound. <strong>Click all three.</strong></p>'''},
    {"title":"How the search works, and why it must be bounded","body":
     '''<h4>Step 1 — in words</h4>
     <p>TLC keeps a set of states it has seen and a queue of states to explore. It pops a state, generates all Next successors, checks the invariant on each, and adds any new one to the queue. It stops when the queue is empty (all reachable states checked) or when an invariant breaks.</p>
     <h4>Step 2 — the formula (size of the search)</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       one variable with V possible values → up to V states<br>
       k independent variables, each with V values → up to <span class="hl">V<sup>k</sup></span> states<br>
       a sequence of length up to N over an alphabet of size V → up to V<sup>0</sup>+V<sup>1</sup>+...+V<sup>N</sup> states<br>
       <span class="dim"># the count grows EXPONENTIALLY — this is state-space explosion</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.8rem;line-height:1.8">
       3 boolean flags (V=2, k=3):   2³ = <span class="hl">8</span> states  — TLC finishes instantly<br>
       10 boolean flags (V=2, k=10): 2¹⁰ = <span class="hl">1,024</span> states — still fast<br>
       20 flags: 2²⁰ ≈ 1,000,000 — seconds to minutes<br>
       40 flags: 2⁴⁰ ≈ <span class="bad">1,000,000,000,000</span> — <span class="bad">will not finish</span><br>
       <span class="dim"># each added flag DOUBLES the work: 2^k</span>
     </div></div>
     <p>So you set bounds: cap how many values a variable can take, cap sequence length. TLC then checks a small finite model exhaustively instead of an infinite one, forever.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the too-small bound that "passes."</b> You bound values to a set of size 4 and length to 6 so TLC finishes fast, and it prints "No error found." But the bug you feared only appears at length 7 — a step that mishandles inputs above a threshold. TLC never errors: it truthfully and exhaustively checks the tiny model, prints green, and the broken design ships with a "verified" stamp. Catch it by asking of every model: "what is the smallest input that could trigger the failure class I care about — is my bound at least that large?" and by reading the printed <em>state count</em> to confirm the search was not suspiciously small.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — bound size: coverage vs. runtime.</b> Every value you add to a variable and every extra unit of length multiplies the state count. A small bound checks in seconds but proves little; a large bound proves more but can push the check from minutes to hours to "never finishes." Pick the smallest bound that still exercises the failure class you fear, and write down in the README which input classes fall outside it, so the coverage gap is documented, not pretended away.</div></div>''',
     "gotit":"Got the search"},
    {"title":"The checker, built up","body":'''<p>Here it assembles: the reachable states as dots, TLC walking all of them, the invariant checked in each, a red violating state, and the trace back to Init. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Check a bounded spec with a mini model checker","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m28_tlc_mini.py</code>. Build a small state machine (e.g. a bounded counter mod N, or a bank account with deposits/withdrawals capped at a small range). Implement BFS over the reachable states, an <code>invariant(state) -&gt; bool</code> function, and — when the invariant fails — return the trace back to Init using a parent map. Print the state count explored. Then shrink the bound and show the same bug going undetected, to feel the too-small-bound failure.</p>
     <h4>Option B · let Claude build it, then read it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 28 Day 3 artifact.
Create experiments/foundations/m28_tlc_mini.py: a tiny explicit-state model checker. Model a bank account (integer balance, deposit +d and withdraw -d moves, values bounded to a small range). Do BFS from Init over reachable states, tracking a parent map. Check invariant "balance >= 0" in every reachable state; on the first violation, reconstruct and print the trace from Init to the bad state. Print the total number of states explored. Then run it once with a bound too small to reach the negative balance (invariant "passes") and once with a large enough bound (invariant fails with a trace), and comment on the too-small-bound trap.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m28/day-03-log.md</code>: how many states did your model explore, and what bound would have hidden the bug?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "pass":{"btn":"① invariant holds","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># counter mod 3, invariant x &lt; 3</span>
tlc Counter.tla Counter.cfg
<span class="ok">Model checking completed. No error has been found.</span>
<span class="dim">3 states generated, 3 distinct states found.</span>''',
      "take":"<b>①  A clean run.</b> TLC explored all 3 reachable states and the invariant x < 3 held in every one. That is a proof — for this bounded model."},
    "fail":{"btn":"② invariant fails → trace","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># same counter, invariant x &lt; 2</span>
tlc Counter.tla Counter.cfg
<span class="bad">Invariant xLessThan2 is violated.</span>
<span class="bad">Error trace:</span>
<span class="bad">  State 1: x = 0</span>
<span class="bad">  State 2: x = 1</span>
<span class="bad">  State 3: x = 2   ← 2 &lt; 2 is FALSE</span>''',
      "take":"<b>②  A trace, not just 'false'.</b> TLC hands you the exact path 0 → 1 → 2 to the state that breaks x < 2. You can replay it to understand the bug."},
    "explode":{"btn":"③ forget to bound","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># values unbounded: x can be any integer</span>
tlc CounterUnbounded.tla
<span class="hl">1,000,000 states...</span> <span class="hl">10,000,000 states...</span>
<span class="hl">100,000,000 states...</span>
<span class="bad"># never finishes — the reachable set is infinite</span>''',
      "take":"<b>③  Unbounded = never done.</b> With no cap the reachable set is infinite, so TLC runs forever. You must shrink the model to a small, finite bound."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='100' y='16' width='320' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='36' fill='#5E5191'>Invariant: balance ≥ 0 (must always hold)</text></g></svg>",
     "note":"<b>Piece 1 — the invariant.</b> You state a rule that must be true in every reachable state. Here: the balance is never negative. TLC's job is to try to break it."},
    {"viz":"<svg viewBox='0 0 520 90'><g><circle cx='60' cy='45' r='9' fill='#E4F2F7' stroke='#2A7B9B'/><circle cx='130' cy='30' r='9' fill='#E4F2F7' stroke='#2A7B9B'/><circle cx='130' cy='62' r='9' fill='#E4F2F7' stroke='#2A7B9B'/><circle cx='210' cy='45' r='9' fill='#E4F2F7' stroke='#2A7B9B'/><circle cx='210' cy='75' r='9' fill='#E4F2F7' stroke='#2A7B9B'/><circle cx='290' cy='30' r='9' fill='#E4F2F7' stroke='#2A7B9B'/><circle cx='290' cy='62' r='9' fill='#E4F2F7' stroke='#2A7B9B'/><circle cx='370' cy='48' r='9' fill='#E4F2F7' stroke='#2A7B9B'/><line x1='60' y1='45' x2='130' y2='30' stroke='#E5DFD6'/><line x1='60' y1='45' x2='130' y2='62' stroke='#E5DFD6'/><line x1='130' y1='30' x2='210' y2='45' stroke='#E5DFD6'/><line x1='130' y1='62' x2='210' y2='75' stroke='#E5DFD6'/><line x1='210' y1='45' x2='290' y2='30' stroke='#E5DFD6'/><line x1='210' y1='75' x2='290' y2='62' stroke='#E5DFD6'/><line x1='290' y1='30' x2='370' y2='48' stroke='#E5DFD6'/><line x1='290' y1='62' x2='370' y2='48' stroke='#E5DFD6'/></g><text x='240' y='16' text-anchor='middle' font-size='10' font-family='monospace' fill='#6B645E'>every dot = one reachable state</text></svg>",
     "note":"<b>Piece 2 — the reachable states.</b> From Init, following Next, the model reaches a set of states (the dots). TLC will visit all of them — a full search, not a sample."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='40' y='20' width='120' height='28' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='100' y='38' fill='#1a5c38'>bal 5 ✓</text><rect x='200' y='20' width='120' height='28' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='260' y='38' fill='#1a5c38'>bal 0 ✓</text><rect x='360' y='20' width='120' height='28' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='420' y='38' fill='#1a5c38'>bal 2 ✓</text></g></svg>",
     "note":"<b>Piece 3 — check the invariant in each state.</b> As TLC visits a state it tests the invariant there. So far every visited balance is ≥ 0, so the rule holds."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='170' y='16' width='180' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2.5'/><text x='260' y='36' fill='#C93B3B'>bal = -3  ✗ violates ≥ 0</text></g></svg>",
     "note":"<b>Piece 4 — a violating state.</b> TLC reaches a state where balance is -3. The invariant is false here. The search stops and TLC prepares the report."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='30' y='16' width='460' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>trace: bal 0 → withdraw 3 → bal -3   (Init to the bug)</text></g></svg>",
     "note":"<b>Piece 5 — the counterexample trace.</b> TLC prints the exact path from Init to the broken state. That chain is a concrete bug report — replay it to find and fix the flaw."},
    {"viz":"<svg viewBox='0 0 520 100'><line x1='50' y1='84' x2='500' y2='84' stroke='#6B645E'/><line x1='50' y1='8' x2='50' y2='84' stroke='#6B645E'/><rect x='70' y='78' width='40' height='6' fill='#2A7B9B'/><text x='90' y='96' text-anchor='middle' font-size='9' font-family='monospace'>k=3</text><rect x='150' y='70' width='40' height='14' fill='#2A7B9B'/><text x='170' y='96' text-anchor='middle' font-size='9' font-family='monospace'>k=10</text><rect x='230' y='48' width='40' height='36' fill='#2A7B9B'/><text x='250' y='96' text-anchor='middle' font-size='9' font-family='monospace'>k=20</text><rect x='310' y='14' width='40' height='70' fill='#C99A12'/><text x='330' y='96' text-anchor='middle' font-size='9' font-family='monospace'>k=30</text><rect x='390' y='6' width='40' height='12' fill='#C93B3B' opacity='.6'/><text x='410' y='96' text-anchor='middle' font-size='9' font-family='monospace'>k=40</text><text x='410' y='4' text-anchor='middle' font-size='8' fill='#C93B3B' font-family='monospace'>off chart</text></g></svg>",
     "note":"<b>Why you MUST bound.</b> States grow as 2^k for k boolean flags — state-space explosion. That is why you cap values and lengths, so TLC checks a small finite model exhaustively instead of running forever."},
  ],
  quiz=[
    {"q":"1. What is an invariant?","opts":["a variable that never changes name","a property that must be true in every reachable state","the first state of the system","the fastest path through the states"],"ans":1,"fb":"An invariant is a 'must always' rule: true in every reachable state. TLC tries to find a state where it is false."},
    {"q":"2. When TLC finds an invariant violation, what does it give you?","opts":["just the word 'false'","a counterexample trace: the exact chain of states from Init to the broken state","a faster version of the spec","a random state"],"ans":1,"fb":"TLC returns the concrete path from a legal start to the violating state — a replayable bug report no passing test can give."},
    {"q":"3. A model has 20 independent boolean flags. Roughly how many states must TLC explore in the worst case?","opts":["20","40","about 1,000,000 (2^20)","exactly 2"],"ans":2,"fb":"Each flag doubles the count: 2^20 ≈ 1,048,576 ≈ 1,000,000. Each added flag doubles the work — this is state-space explosion."},
    {"q":"4. TLC prints 'No error has been found' on a model bounded to length ≤ 6. What have you proven?","opts":["the design is correct for all inputs of any length","the invariant holds for every reachable state OF THAT BOUNDED MODEL — nothing about length 7+","the code is faster now","the spec compiles"],"ans":1,"fb":"Exhaustive over the bounded model only. A bug that first appears at length 7 was never in the search space — the too-small-bound trap. Match the bound to the smallest input that could trigger the failure you fear."},
  ],
  fin={"em":"🔍","h3":"Day 3 complete — you can run the checker!",
       "p":"You now know invariants (must-always rules), how TLC exhaustively searches the reachable states and returns a counterexample trace, why state-space explosion forces you to bound the model, and how a too-small bound can silently pass. Next: <b>Day 4</b> — refinement, the argument that connects your abstract spec to the real code so the proof actually transfers."},
))

# ---------------- Day 4 — Refinement: Spec to Code ----------------
L("day-04-refinement.html", dict(
  qid="m28-d04-refine", title="Module 28 · Day 4 — Refinement: Spec to Code", nav_title="Spiral · M28 Day 4",
  eyebrow="Module 28 · Verify · Day 4", h1="Refinement: Spec to Code",
  lead="TLC just proved your abstract spec is correct. But you do not ship the spec — you ship Python. So there is a gap: does the real code actually behave like the spec you proved? <b>Refinement</b> is the bridge. It is the claim that every step the concrete code takes is also a step the abstract spec allows. If that holds, the code inherits the spec's guarantee. This is the piece that plugs formal methods into your M26 verify loop.",
  goal="<b>🎯 By the end, you'll be able to:</b> define refinement, describe a refinement map from concrete code state to abstract spec state, explain why this matters for the M26 loop, and spot when a wrong map hides a real bug.",
  prev_href="day-03-invariants-tlc.html", prev_label="Invariants & Model Checking (TLC)",
  next_href="day-05-hypothesis.html", next_label="Property-Based Testing (Hypothesis)",
  sections=[
    {"title":"The proof is about the spec, not the code","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 3 — TLC checks an invariant across every reachable state of an abstract, bounded model. And your M26 loop, where real code is proposed and must be verified.</div></div>
     <h4>What TLC actually proved</h4>
     <p>TLC proved the invariant on the <em>abstract spec</em> — a simplified model. Your real code has extra details the spec left out: data structures, loop counters, error handling. "The spec is correct" does not automatically mean "this code is correct."</p>
     <h4>Refinement is the bridge</h4>
     <p><span class="term" data-tip="A relation saying every step the concrete code takes matches some step the abstract spec allows; if so, the code inherits the spec's proven properties.">Refinement</span> is the claim: every behaviour of the concrete code is also a behaviour the abstract spec allows. The code never does anything the spec forbids. If that claim holds, the code inherits every property you proved about the spec — including the invariant.</p>
     <h4>You build the bridge with a map</h4>
     <p>You connect the two with a <span class="term" data-tip="A function that turns a concrete code state into the abstract state it represents, so you can check each concrete step matches an allowed abstract step.">refinement map</span>: a function from the code's detailed state to the spec's simple state. It answers "when the code is in THIS situation, which abstract state is it in?" Then you check that each concrete step, viewed through the map, is a legal abstract step.</p>''',
     "gotit":"Got the bridge idea"},
    {"title":"An architect's plan vs. the built house","body":
     '''<div class="relate">
       <div class="card"><span class="big">📐🏠</span><h5>Abstract spec = the plan</h5><p>The architect proves the plan is sound: loads balance, exits reachable. But the plan is not the house — it leaves out which nails, which paint.</p></div>
       <div class="card"><span class="big">🔗✅</span><h5>Refinement = the inspection map</h5><p>An inspector maps each built part back to the plan: "this beam here IS the plan's beam B; it is where the plan says and does what the plan says." If every built part maps to a plan part it obeys, the built house inherits the plan's guarantees.</p></div>
     </div>
     <p><strong>In one line:</strong> refinement is the mapping that shows the built thing faithfully follows the proven plan, so the plan's guarantees carry over to the real thing.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a building inspector checks a finished object once. A refinement argument must cover every possible <em>step</em> the code can take, in every state — an ongoing relation over behaviours, not a one-time snapshot. And unlike a physical inspection, a wrong map can look perfectly consistent while quietly excusing a real bug.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Map concrete state to abstract state","body":'''<p>Watch an abstract lock spec, a concrete Python lock with extra fields, and the refinement map connecting them — then a step check. <strong>Click all three.</strong></p>'''},
    {"title":"How a refinement map is checked, and how it can lie","body":
     '''<h4>Step 1 — in words</h4>
     <p>Define a map f that turns any concrete state c into the abstract state f(c). Then require two things: the start maps correctly, and every concrete move maps to a legal abstract move. If both hold, every concrete behaviour becomes an abstract behaviour — so the spec's invariant holds on the code too.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       f : ConcreteState → AbstractState        <span class="dim"># the refinement map</span><br>
       (1)  ConcreteInit(c)  ⇒  AbstractInit(f(c))          <span class="dim"># starts line up</span><br>
       (2)  ConcreteStep(c, c')  ⇒  AbstractStep(f(c), f(c'))  <span class="dim"># each step lines up</span><br>
       if (1) and (2) hold for all c, c':  code REFINES spec  ⇒  spec's invariant holds on code
     </div></div>
     <h4>Step 3 — worked example (a lock)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.8rem;line-height:1.8">
       abstract state: locked ∈ {TRUE, FALSE}<br>
       concrete state: {"is_locked": bool, "fail_count": int, "last_user": str}<br>
       map f(c) = c["is_locked"]           <span class="dim"># drop the extra fields, keep the boolean</span><br>
       concrete unlock step: {is_locked:True, fail_count:0} → {is_locked:False, fail_count:0}<br>
       through f:  TRUE → FALSE            <span class="dim"># which IS a legal abstract Next move</span> <span class="hl">✓</span><br>
       every concrete step maps to a legal abstract step → the code refines the spec
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — a wrong refinement map hides real bugs.</b> If you pick a map that is too forgiving, a buggy step can look legal. Say the concrete lock has a bug: under high <code>fail_count</code> it flips <code>is_locked</code> to False on its own. If your map <em>also</em> secretly reads <code>fail_count</code> and rewrites the abstract state to match, the bad step maps to a "legal" abstract move and the refinement check passes — while the real lock opens itself. The map excused the very bug you were hunting. The fix: keep the map simple and mechanical (a plain projection of fields), never let it depend on outcomes, and review it as carefully as the invariant itself.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — proof strength vs. mapping effort.</b> A full, checked refinement (the map plus a proof or model-check that every step lines up) gives the strongest guarantee: the code truly inherits the spec's properties. But it is real work — you must model the concrete code closely enough to relate it, which for large systems is expensive. A looser, informal argument ("the code clearly follows the spec") is cheap but proves little and can hide a wrong-map bug. Spend the full effort where a defect is catastrophic; use the informal argument, honestly labelled, elsewhere.</div></div>''',
     "gotit":"Got refinement maps"},
    {"title":"The bridge, built up","body":'''<p>Here it assembles: the proven abstract spec, the detailed concrete code, the map projecting one to the other, a step check through the map, and the guarantee carrying over — plus how a bad map breaks it. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Wire a refinement check into the M26 loop","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m28_refinement.py</code>. Define an abstract lock (state = boolean) with its Init and Next. Define a concrete lock as a small Python class with extra fields (fail_count, last_user). Write a refinement map <code>f(concrete) -&gt; abstract</code>. Run the concrete machine over many traces; for each step, assert that <code>(f(before), f(after))</code> is a legal abstract move. Then inject a buggy self-unlock step and show the check catches it — and separately show a too-forgiving map that wrongly lets it pass.</p>
     <h4>Option B · let Claude build it, then read it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 28 Day 4 artifact.
Create experiments/foundations/m28_refinement.py: an abstract lock (state = bool) with Init and a set of legal abstract transitions; a concrete Lock class with extra fields (fail_count, last_user); and a refinement map f(concrete)->abstract that projects to the boolean. Simulate concrete behaviours; for each concrete step assert (f(before), f(after)) is an allowed abstract transition. Then (a) inject a buggy concrete step that self-unlocks and show a correct simple map catches it, and (b) show a too-forgiving map (one that reads fail_count) wrongly excusing the same bug. Add a comment tying this to the M26 verify loop: TLC proves the spec, refinement carries the guarantee to the real code.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m28/day-04-log.md</code>: what made your refinement map "too forgiving," and how did it hide the injected bug?</div></div>''',
     "gotit":"Done — on to Day 5"},
  ],
  demos={
    "abstract":{"btn":"① the proven abstract spec","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># abstract lock — this is what TLC proved</span>
locked ∈ {TRUE, FALSE}
legal moves: TRUE→FALSE (unlock), FALSE→TRUE (lock)
<span class="ok"># invariant held in all reachable states</span>''',
      "take":"<b>①  The proven side.</b> A tiny abstract lock TLC already verified. The whole point of today is to carry this proof onto real code."},
    "concrete":{"btn":"② the real code (extra detail)","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># concrete Python lock: more fields</span>
state = {<span class="str">"is_locked"</span>: True,
         <span class="str">"fail_count"</span>: 0,
         <span class="str">"last_user"</span>: <span class="str">"alice"</span>}
<span class="dim"># richer than the spec — which detail matters?</span>''',
      "take":"<b>②  The real side.</b> The code carries extra state the spec ignored. The proof does not automatically apply to this richer object."},
    "map":{"btn":"③ the map + a step check","html":'''<span class="prompt">&gt;&gt;&gt;</span> f(c) = c[<span class="str">"is_locked"</span>]   <span class="dim"># keep only the boolean</span>
<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># concrete unlock step, seen through f:</span>
   {is_locked:True,...} → {is_locked:False,...}
   f:  TRUE → FALSE   <span class="ok">✓ a legal abstract move</span>''',
      "take":"<b>③  The bridge.</b> The map drops extra fields and keeps the boolean. Each concrete step, seen through f, must be a legal abstract move. If all are, the code refines the spec."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='33' fill='#5E5191'>abstract spec — proven by TLC ✓</text></g></svg>",
     "note":"<b>Piece 1 — the proven abstract spec.</b> TLC verified the invariant here. This is the guarantee you want to carry onto real code."},
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='90' y='14' width='340' height='28' rx='6' fill='#FDF9F3' stroke='#6B645E' stroke-width='2' stroke-dasharray='5 3'/><text x='260' y='32' fill='#5A544E'>concrete code: {is_locked, fail_count, last_user}</text></g></svg>",
     "note":"<b>Piece 2 — the concrete code.</b> The real implementation carries extra detail the spec left out. The proof does not apply to it for free."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='40' y='14' width='190' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA'/><text x='135' y='32' fill='#5E5191'>abstract: TRUE / FALSE</text><rect x='290' y='14' width='190' height='28' rx='6' fill='#FDF9F3' stroke='#6B645E' stroke-dasharray='4 3'/><text x='385' y='32' fill='#5A544E'>concrete: {is_locked,...}</text><line x1='290' y1='58' x2='230' y2='58' stroke='#C99A12' stroke-width='2' marker-end='url(#a4)'/><text x='260' y='78' fill='#9A7208' font-size='9.5'>f(c) = c[is_locked]  — the refinement map</text><defs><marker id='a4' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0 0 L6 3 L0 6 z' fill='#C99A12'/></marker></defs></g></svg>",
     "note":"<b>Piece 3 — the refinement map.</b> f turns a concrete state into the abstract state it represents. Here it just keeps the boolean and drops the extra fields."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='40' y='16' width='440' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>concrete step → through f → TRUE→FALSE = legal abstract move ✓</text></g></svg>",
     "note":"<b>Piece 4 — check every step through the map.</b> Each concrete move, viewed through f, must be a legal abstract move. If every step passes, the code refines the spec."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='80' y='16' width='360' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>guarantee carries over: spec's invariant holds on the code</text></g></svg>",
     "note":"<b>Piece 5 — the guarantee transfers.</b> Because every concrete behaviour maps to an allowed abstract one, the code inherits the invariant TLC proved. This is what plugs formal methods into the M26 loop."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='30' y='16' width='210' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='135' y='35' fill='#C93B3B'>too-forgiving map</text><rect x='270' y='16' width='220' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='380' y='35' fill='#C93B3B'>buggy self-unlock looks 'legal'</text></g></svg>",
     "note":"<b>The trap — a wrong map hides bugs.</b> If the map depends on outcomes, a buggy step can map to a 'legal' abstract move and pass. Keep the map a simple, mechanical projection and review it as carefully as the invariant."},
  ],
  quiz=[
    {"q":"1. TLC proved the invariant on your abstract spec. What does that say about your real Python code?","opts":["the code is automatically correct too","nothing by itself — the proof is about the spec; you need a refinement argument to carry it to the code","the code will run faster","the code has no syntax errors"],"ans":1,"fb":"The proof is about the abstract model. Refinement is the extra argument that every code behaviour is allowed by the spec, so the guarantee transfers."},
    {"q":"2. What does 'the code refines the spec' mean?","opts":["the code is more polished than the spec","every behaviour of the code is also a behaviour the spec allows — the code never does anything the spec forbids","the spec was edited to match the code's bugs","the spec generates the code"],"ans":1,"fb":"Refinement: concrete behaviours are a subset of abstract behaviours. Then the code inherits everything proven about the spec."},
    {"q":"3. What is a refinement map?","opts":["a diagram of the office","a function from a concrete code state to the abstract state it represents, used to check each step lines up","the fastest route through the states","a list of variable names"],"ans":1,"fb":"The map f: ConcreteState → AbstractState lets you view each concrete step as an abstract step and check it is legal."},
    {"q":"4. Your refinement check passes, yet the real lock opens itself under load. What is the most likely cause?","opts":["TLC is broken","the refinement map was too forgiving — it depended on outcomes and mapped the buggy self-unlock to a 'legal' abstract move","the invariant was too strong","the code was too fast"],"ans":1,"fb":"A map that reads outcomes can excuse the very bug you are hunting. Keep the map a simple mechanical projection and review it like the invariant."},
  ],
  fin={"em":"🌉","h3":"Day 4 complete — you built the bridge!",
       "p":"You now know refinement: the claim that every concrete step matches an allowed abstract step, checked with a refinement map, so the code inherits the spec's proven invariant. You saw why the M26 loop needs this bridge and how a too-forgiving map can hide a real bug. Next: <b>Day 5</b> — property-based testing with Hypothesis, the practical tool that runs on the real code and hunts for counterexamples by generating many random inputs."},
))

# ---------------- Day 5 — Property-Based Testing (Hypothesis) ----------------
L("day-05-hypothesis.html", dict(
  qid="m28-d05-hypothesis", title="Module 28 · Day 5 — Property-Based Testing", nav_title="Spiral · M28 Day 5",
  eyebrow="Module 28 · Verify · Day 5", h1="Property-Based Testing (Hypothesis)",
  lead="TLA+ proves things about a model. But you also want to hammer the <em>real</em> code, cheaply, today. That is <b>property-based testing</b>. Instead of writing one example test with fixed inputs, you state a <em>property</em> that should hold for ALL inputs — then a library called <b>Hypothesis</b> generates hundreds of random inputs trying to break it. When it finds a failure, it <b>shrinks</b> it down to the smallest, clearest input that still fails.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain a property vs. an example test, describe how Hypothesis generates and shrinks inputs, write a round-trip or invariant property, and spot a weak property that passes vacuously.",
  prev_href="day-04-refinement.html", prev_label="Refinement: Spec to Code",
  next_href="day-06-synthesis.html", next_label="When to Use Which",
  sections=[
    {"title":"State a rule, not an example","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> your M26 loop runs tests on real code, and Days 1–4 — invariants are "must always" rules. Hypothesis brings that idea to ordinary Python tests.</div></div>
     <h4>Example tests check one input</h4>
     <p>A normal test says: <code>sort([3,1,2]) == [1,2,3]</code>. It checks exactly one input. You picked it, so you probably will not pick the tricky one that breaks your code.</p>
     <h4>A property is a rule for all inputs</h4>
     <p>A <span class="term" data-tip="A rule that should hold for every valid input, not just one example; e.g. 'the sorted list has the same elements as the input.'">property</span> is a rule that should be true for <em>any</em> input. For sorting: "the output has the same elements as the input, and is in non-decreasing order." You do not name the inputs — you state the rule.</p>
     <h4>Hypothesis generates the inputs for you</h4>
     <p><span class="term" data-tip="A Python library for property-based testing; it generates many random inputs to try to break a stated property, then shrinks any failure to a minimal case.">Hypothesis</span> is the library that supplies inputs. You declare the <em>shape</em> of valid inputs (a list of integers, a string, ...) and Hypothesis generates hundreds of examples — including nasty ones like empty lists, duplicates, and huge values — hunting for one that breaks your property.</p>
     <h4>And it shrinks the failure</h4>
     <p>When Hypothesis finds a breaking input, it does not stop. It <span class="term" data-tip="Automatically simplifying a failing input to the smallest one that still fails, so the bug is easy to read.">shrinks</span> it: repeatedly simplifies (shorter list, smaller numbers) while the failure persists, so you get the tiniest, clearest counterexample — e.g. <code>[0, 0]</code> instead of a random 40-element list.</p>''',
     "gotit":"Got the idea"},
    {"title":"A crash-test robot for your code","body":
     '''<div class="relate">
       <div class="card"><span class="big">🤖💥</span><h5>Hypothesis = a tireless crash-tester</h5><p>You do not hand-pick a few crash scenarios. A robot slams the car in hundreds of random ways — angles, speeds, loads — looking for any that break the safety rule.</p></div>
       <div class="card"><span class="big">🔎🚗</span><h5>Shrinking = the minimal repro</h5><p>When one crash breaks the rule, the robot then finds the <em>simplest</em> crash that still breaks it — "just tap the left corner at 5 mph" — so engineers can reproduce and fix it fast.</p></div>
     </div>
     <p><strong>In one line:</strong> Hypothesis throws hundreds of random inputs at your real code to break a stated rule, then hands you the smallest input that still breaks it.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a crash-test robot tries physical scenarios that all really happen. Hypothesis searches a huge input space at <em>random</em> — it can easily miss a rare bug it never happened to generate. Passing means "no counterexample found in N random tries," not "correct for all inputs." That is the key difference from TLA+'s exhaustive check.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Watch Hypothesis break and shrink","body":'''<p>Watch a property stated once, Hypothesis generating inputs to break it, and shrinking the failure to a minimal case. <strong>Click all three.</strong></p>'''},
    {"title":"How generate-and-shrink works, and how a property can lie","body":
     '''<h4>Step 1 — in words</h4>
     <p>You state a property P(x) that should hold for all valid x. Hypothesis draws many random x from the shape you declared and checks P(x). If some x fails, it shrinks x to the simplest failing case and reports it. If none fail after N tries, the test passes — meaning "no counterexample found," not "proven."</p>
     <h4>Step 2 — the formula (two classic property shapes)</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       round-trip:  decode(encode(x)) == x        <span class="dim"># for every x</span><br>
       invariant :  is_sorted(sort(xs)) AND same_multiset(sort(xs), xs)   <span class="dim"># for every xs</span><br>
       Hypothesis draws x₁, x₂, ..., x_N ; passes iff P holds on all of them<br>
       on failure: shrink x to the minimal x* with P(x*) still false
     </div></div>
     <h4>Step 3 — worked example (round-trip on a buggy encoder)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.8rem;line-height:1.8">
       property: decode(encode(s)) == s   <span class="dim"># encode strips trailing spaces (bug!)</span><br>
       random try #37:  s = <span class="str">"hi there  "</span> → decode(encode(s)) = <span class="str">"hi there"</span>  <span class="bad">✗ mismatch</span><br>
       shrink: <span class="str">"hi there  "</span> → <span class="str">"a "</span> → <span class="str">" "</span>  <span class="dim"># still fails, keep simplifying</span><br>
       reported minimal counterexample: s = <span class="hl">" "</span>  (a single space)<br>
       <span class="dim"># one space is the tiniest input that exposes the trailing-space bug</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the vacuous / weak property.</b> A property can pass for the wrong reason. Two ways: (1) <b>vacuous</b> — your input filter is so strict that almost no inputs get generated, so the property is "checked" on nearly nothing and passes while proving almost nothing (Hypothesis warns about this, but only if you look). (2) <b>weak</b> — the property is true but too loose to catch the bug: testing only <code>len(sort(xs)) == len(xs)</code> passes even for a "sort" that returns the list unchanged. The fix: assert the <em>full</em> property (sorted AND same elements), and check how many examples actually ran and were not filtered away.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — random search vs. formal proof.</b> Hypothesis is cheap, runs on the real code, and needs no model — but it is incomplete: passing only means no counterexample was found in N random draws, so a rare bug can survive. TLA+/TLC is exhaustive over a bounded model — complete for that model — but costs modelling effort and proves things about an abstraction, not the exact code. They are complementary: use Hypothesis broadly on real code, and reserve formal proof for the few properties whose failure would be catastrophic.</div></div>''',
     "gotit":"Got generate-and-shrink"},
    {"title":"The test, built up","body":'''<p>Here it assembles: the property stated once, the input shape, random inputs generated, a failure caught, and the shrink down to a minimal counterexample. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Write property-based tests on real code","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m28_hypothesis.py</code>. Install <code>hypothesis</code>. Write two property tests on a real function: a round-trip property (e.g. <code>decode(encode(s)) == s</code>) and an invariant property for a sort (output is sorted AND is a permutation of the input). Introduce a deliberate bug in one function and watch Hypothesis find and <em>shrink</em> a minimal failing input. Then write a deliberately weak property and show it passes even on the buggy code.</p>
     <h4>Option B · let Claude build it, then read it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 28 Day 5 artifact.
Create experiments/foundations/m28_hypothesis.py using the hypothesis library. Write (1) a round-trip property decode(encode(s)) == s over st.text(), and (2) an invariant property for my_sort: the result is sorted AND is a permutation (same multiset) of the input, over st.lists(st.integers()). Add a deliberately buggy encoder (strips trailing spaces) and a buggy sort (drops duplicates); show Hypothesis finds and SHRINKS a minimal failing input for each. Finally add a deliberately weak property (only len(sort(xs))==len(xs)) and show it passes even on a "sort" that returns the input unchanged, demonstrating a vacuous/weak property. Print how many examples ran.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m28/day-05-log.md</code>: what minimal counterexample did Hypothesis shrink to, and how did your weak property miss the bug?</div></div>''',
     "gotit":"Done — on to Day 6"},
  ],
  demos={
    "prop":{"btn":"① state the property","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">from</span> hypothesis <span class="kw">import</span> given, strategies <span class="kw">as</span> st
<span class="prompt">&gt;&gt;&gt;</span> @given(st.lists(st.integers()))
<span class="prompt">&gt;&gt;&gt;</span> <span class="kw">def</span> <span class="fn">test_sort</span>(xs):
        r = my_sort(xs)
        <span class="kw">assert</span> is_sorted(r) <span class="kw">and</span> sorted(xs) == sorted(r)''',
      "take":"<b>①  One rule for all inputs.</b> You state the property (sorted AND same elements). You do NOT pick the inputs — that is Hypothesis's job."},
    "gen":{"btn":"② generate and break","html":'''<span class="prompt">&gt;&gt;&gt;</span> pytest test_sort.py   <span class="dim"># Hypothesis tries many inputs</span>
trying [], [0], [3,1], [5,5,5,-2,5], ...
<span class="bad">Falsifying example: my_sort([5, 5]) dropped a duplicate</span>
<span class="dim"># a buggy sort that removes duplicates is caught</span>''',
      "take":"<b>②  Hundreds of random tries.</b> Hypothesis throws generated inputs — including empties and duplicates — until one breaks the property. Here a dup-dropping bug is found."},
    "shrink":{"btn":"③ shrink to minimal","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># first failure was a messy 12-element list</span>
shrinking: [5,5,-2,9,5,...] → [5,5] → <span class="hl">[0, 0]</span>
<span class="bad">Falsifying example: my_sort([0, 0])</span>
<span class="dim"># the tiniest input that still fails</span>''',
      "take":"<b>③  Shrink to the clearest repro.</b> Hypothesis simplifies the failing input while it still fails, reporting [0, 0] — a minimal, obvious counterexample you can debug in seconds."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='60' y='14' width='400' height='28' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='32' fill='#5E5191'>property: is_sorted(sort(xs)) AND same_elements(sort(xs), xs)</text></g></svg>",
     "note":"<b>Piece 1 — state the property once.</b> A rule that should hold for every input, not a single example. You describe WHAT must be true, not which inputs to try."},
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='14' width='280' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='32' fill='#1F6280'>input shape: st.lists(st.integers())</text></g></svg>",
     "note":"<b>Piece 2 — declare the input shape.</b> You tell Hypothesis the kind of input (a list of integers). It will invent the actual values, including edge cases."},
    {"viz":"<svg viewBox='0 0 520 64'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='18' width='70' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='65' y='35' fill='#1a5c38'>[]</text><rect x='115' y='18' width='90' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='160' y='35' fill='#1a5c38'>[3,1,2]</text><rect x='220' y='18' width='120' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='280' y='35' fill='#1a5c38'>[5,5,-2,5]</text><rect x='355' y='18' width='130' height='26' rx='5' fill='#E8F5EE' stroke='#2D8B55'/><text x='420' y='35' fill='#1a5c38'>[9,9,9,9,0]</text></g></svg>",
     "note":"<b>Piece 3 — generate many random inputs.</b> Hypothesis draws hundreds of examples — empty, sorted, duplicate-heavy — trying to falsify the property on real code."},
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='80' y='14' width='360' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='32' fill='#C93B3B'>[5, 5] → sort drops a duplicate → property FALSE ✗</text></g></svg>",
     "note":"<b>Piece 4 — a failure caught.</b> One generated input breaks the property: a buggy sort that removes duplicates. Hypothesis has found a real counterexample on the actual code."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='40' y='16' width='200' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='140' y='34' fill='#9A7208'>[5,5,-2,9,5,...]</text><text x='260' y='34' fill='#6B645E'>→ shrink →</text><rect x='340' y='16' width='140' height='28' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='410' y='34' fill='#9A7208'>[0, 0]</text></g></svg>",
     "note":"<b>Piece 5 — shrink to minimal.</b> Hypothesis simplifies the messy failure while it still fails, reporting the tiniest counterexample [0, 0] — a clear, fast-to-debug repro."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='16' width='230' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='145' y='35' fill='#C93B3B'>weak: only len(sort(xs))==len(xs)</text><rect x='285' y='16' width='205' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='387' y='35' fill='#C93B3B'>passes on a no-op 'sort' ✗</text></g></svg>",
     "note":"<b>The trap — a weak property passes vacuously.</b> A property too loose to catch the bug (checking only length) passes even for a sort that does nothing. Assert the FULL property and check how many examples actually ran."},
  ],
  quiz=[
    {"q":"1. What is the difference between an example test and a property?","opts":["a property is written in a different language","an example test checks one fixed input; a property states a rule that should hold for ALL inputs, and the library generates them","a property runs slower","there is no difference"],"ans":1,"fb":"Example tests check inputs you picked; a property states a rule and Hypothesis invents hundreds of inputs to try to break it."},
    {"q":"2. What does Hypothesis do after it finds a failing input?","opts":["stops immediately with the raw input","shrinks it to the smallest, simplest input that still fails, so the bug is easy to read","reruns the whole suite","ignores it if it's rare"],"ans":1,"fb":"Shrinking simplifies the counterexample (shorter, smaller) while it still fails — e.g. reporting [0,0] instead of a random 12-element list."},
    {"q":"3. A property-based test passes after 300 random inputs. What have you shown?","opts":["the code is correct for all inputs","no counterexample was found in those random tries — a rare bug could still exist","the code is exhaustively verified like TLC","the property is vacuous"],"ans":1,"fb":"Random search is incomplete: passing means 'no counterexample found', not 'proven'. That is the core difference from TLA+/TLC's exhaustive check."},
    {"q":"4. You test a sort with only assert len(sorted_xs) == len(xs). Why is this dangerous?","opts":["length checks are illegal","it is a weak property: a broken 'sort' that returns the input unchanged still has the right length, so the bug passes silently","it makes Hypothesis slower","it always fails"],"ans":1,"fb":"The property is too loose to catch the bug. Assert the FULL property — output is sorted AND a permutation of the input — so a no-op or dup-dropping sort is caught."},
  ],
  fin={"em":"🎲","h3":"Day 5 complete — you can hammer real code!",
       "p":"You now know property-based testing: state a rule for all inputs, let Hypothesis generate hundreds of random inputs to break it, and shrink any failure to a minimal counterexample. You also saw the vacuous/weak-property trap and the random-vs-exhaustive trade-off. Next: <b>Day 6</b> — when to reach for TLA+ vs. Hypothesis, and how to apply both to the M26 ADRS loop."},
))

# ---------------- Day 6 — When to Use Which ----------------
L("day-06-synthesis.html", dict(
  qid="m28-d06-synth", title="Module 28 · Day 6 — When to Use Which", nav_title="Spiral · M28 Day 6",
  eyebrow="Module 28 · Verify · Day 6", h1="When to Use Which",
  lead="You now have two verification tools with very different shapes. <b>TLA+</b> proves a property for every run of a bounded model — at design time, before code exists. <b>Hypothesis</b> hammers the real running code with random inputs — cheap, practical, but incomplete. The staff-level skill is not knowing either tool; it is choosing the right one <em>per risk</em>, and combining both to guard the M26 ADRS loop.",
  goal="<b>🎯 By the end, you'll be able to:</b> compare TLA+ vs. Hypothesis on completeness, cost, and what they run against; pick a tool per risk; and apply one safety invariant + a refinement + property-based tests to the M26 ADRS loop.",
  prev_href="day-05-hypothesis.html", prev_label="Property-Based Testing (Hypothesis)",
  next_href="review.html", next_label="Module 28 Review",
  sections=[
    {"title":"Two tools, two jobs","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–5 — TLA+/TLC (exhaustive over a bounded model) and Hypothesis (random over real code), and the M26 ADRS loop where code is proposed and must be verified before it ships.</div></div>
     <h4>They answer different questions</h4>
     <p><span class="term" data-tip="Exhaustively checks a property across every reachable state of a small, bounded model at design time.">TLA+/TLC</span> answers "is this <em>design</em> correct for every run of a small model?" — a design-time, exhaustive check. <span class="term" data-tip="Runs many random inputs against the real code to try to falsify a stated property; cheap but incomplete.">Hypothesis</span> answers "does the <em>real code</em> break this property on any of hundreds of random inputs?" — a run-time, practical check.</p>
     <h4>Pick per risk, not per taste</h4>
     <p>The decision is about <span class="term" data-tip="How bad a failure would be times how likely it is; high-risk code justifies more expensive verification.">risk</span>. If a bug is catastrophic (money moves, data is lost, a plane is unsafe) and subtle (concurrency, ordering), the modelling cost of TLA+ is worth it. If a bug is recoverable and the code is ordinary, Hypothesis gives most of the value for a fraction of the effort.</p>
     <h4>The strongest loop uses both</h4>
     <p>They are not rivals. For the M26 loop you can layer them: one small TLA+ <strong>safety invariant</strong> on the design, a <strong>refinement</strong> argument tying it to the code, and broad <strong>property-based tests</strong> that run on every proposed candidate. Cheap checks catch most bugs; the expensive proof guards the one property you cannot afford to get wrong.</p>''',
     "gotit":"Got the split"},
    {"title":"A smoke detector vs. a fire-safety proof","body":
     '''<div class="relate">
       <div class="card"><span class="big">🚨</span><h5>Hypothesis = smoke detectors everywhere</h5><p>Cheap, install one in every room, and they catch most fires fast. But they only alarm once smoke is already present — they cannot prove a fire can never start.</p></div>
       <div class="card"><span class="big">📋🔥</span><h5>TLA+ = a fire-safety proof for the vault</h5><p>Expensive and slow, so you do it only for the one room that must never burn — the vault. There you prove, from the design, that no reachable situation causes a fire.</p></div>
     </div>
     <p><strong>In one line:</strong> put cheap detectors everywhere (Hypothesis on all code) and reserve the expensive proof (TLA+) for the few places a failure is unacceptable.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: smoke detectors and a fire proof guard the same physical building. TLA+ proves an <em>abstract model</em>, not the running code — only the refinement argument (Day 4) carries the proof toward the real thing. Detectors, by contrast, sit directly in the real building. So even the "expensive proof" side needs the refinement bridge to mean anything about production.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Walk a real decision","body":'''<p>Watch three ADRS candidates and decide, per risk, which verification each deserves — then the layered gate that combines both. <strong>Click all three.</strong></p>'''},
    {"title":"How to choose, and the trap of over-verifying","body":
     '''<h4>Step 1 — in words</h4>
     <p>Score each property by two things: how bad a failure would be (severity) and how hard the bug is to catch by sampling (subtlety, e.g. concurrency). High on both → formal proof. Low on both → property-based tests are enough. In between → property tests plus a targeted invariant.</p>
     <h4>Step 2 — the comparison table</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.8rem;line-height:1.8">
       ┌────────────┬──────────────┬───────────────┬─────────────┐<br>
       │            │ TLA+ / TLC   │ Hypothesis    │ example test│<br>
       ├────────────┼──────────────┼───────────────┼─────────────┤<br>
       │ runs on    │ abstract mod │ real code     │ real code   │<br>
       │ coverage   │ exhaustive*  │ random/partial│ one input   │<br>
       │ *of        │ bounded model│ N random tries│ that input  │<br>
       │ cost       │ high (model) │ low           │ very low    │<br>
       │ when       │ design time  │ every run     │ every run   │<br>
       └────────────┴──────────────┴───────────────┴─────────────┘
     </div></div>
     <h4>Step 3 — worked decision (M26 ADRS loop)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.8rem;line-height:1.8">
       property A: "the evolved allocator never double-assigns a job"<br>
       &nbsp;&nbsp;severity HIGH (data corruption), subtlety HIGH (concurrency)  → <span class="hl">TLA+ invariant + refinement</span><br>
       property B: "sort output is a permutation of input, sorted"<br>
       &nbsp;&nbsp;severity MED, subtlety LOW  → <span class="hl">Hypothesis property test on every candidate</span><br>
       property C: "log line is under 200 chars"<br>
       &nbsp;&nbsp;severity LOW, subtlety LOW  → <span class="hl">one example test (or skip)</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — over-verifying low-risk code.</b> A subtle waste: pouring TLA+ modelling effort into code where a bug is cheap and recoverable. It feels rigorous, but it burns weeks writing and maintaining a spec for, say, a log formatter — time stolen from the one concurrency invariant that actually matters. Worse, a big pile of low-value specs drifts out of date (Day 1's spec-drift) and erodes trust in <em>all</em> your specs. The fix: gate formal effort behind an explicit risk check — write the invariant only when severity AND subtlety are both high.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — assurance vs. engineering time.</b> More verification always buys more confidence, but at rising cost: example tests are nearly free, Hypothesis is cheap, a checked TLA+ refinement is expensive. There is no "verify everything to the max" budget in a real team. The staff move is to allocate the assurance budget by risk — cheap checks broadly, the expensive proof narrowly — and to write down, per component, which level it got and why, so the coverage map is honest.</div></div>''',
     "gotit":"Got the decision"},
    {"title":"The layered gate, built up","body":'''<p>Here it assembles: the two tools compared, a risk score per property, the tool chosen for each, and the M26 loop guarded by all three layers. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Design the verification gate for the M26 loop","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m28_verify_gate.py</code>. For the M26 ADRS loop, build a gate that, per proposed candidate, runs: (1) Hypothesis property tests (permutation + sorted, or your domain's invariant) on the real candidate, and (2) records a TLC verdict for one high-risk safety invariant on the bounded model. Log <code>{candidate_id, hypothesis_pass, tlc_verdict, decision}</code> to a JSONL file, and write a short README that states, per property, which verification level it got and why (a risk-based coverage map).</p>
     <h4>Option B · let Claude build it, then read it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 28 Day 6 artifact.
Create experiments/foundations/m28_verify_gate.py: a layered verification gate for the M26 ADRS loop. For each proposed candidate function, (1) run Hypothesis property tests on the real code (e.g. sort: output is sorted AND a permutation of input), and (2) record a TLC-style verdict for ONE high-risk safety invariant checked over a small bounded model (you may simulate TLC with a BFS exhaustive checker over the bounded state set). Combine into a decision (accept only if both pass) and log {candidate_id, hypothesis_pass, tlc_verdict, states_explored, decision} to verification_gate.jsonl over at least 3 candidates (one correct, one with a subtle concurrency-style bug, one with an obvious bug). Also emit a README with a risk-based coverage map: per property, the chosen level (example test / Hypothesis / TLA+) and the severity+subtlety reason, and a note on the over-verification trap.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m28/day-06-log.md</code>: which single property did you deem worth a TLA+ invariant, and which properties you deliberately left to Hypothesis — and why.</div></div>''',
     "gotit":"Done — take the review"},
  ],
  demos={
    "risk":{"btn":"① score three candidates","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># M26 ADRS: three properties, scored by risk</span>
A "no double-assigned job"  severity=HIGH subtle=HIGH
B "sort = permutation+sorted" severity=MED  subtle=LOW
C "log line &lt; 200 chars"     severity=LOW  subtle=LOW''',
      "take":"<b>①  Score by risk.</b> Each property gets a severity and subtlety. High-high needs the strongest tool; low-low needs the cheapest."},
    "pick":{"btn":"② pick the tool per risk","html":'''<span class="prompt">&gt;&gt;&gt;</span> A → <span class="hl">TLA+ invariant + refinement</span>   <span class="dim"># catastrophic + concurrent</span>
<span class="prompt">&gt;&gt;&gt;</span> B → <span class="hl">Hypothesis property test</span>       <span class="dim"># cheap, runs on real code</span>
<span class="prompt">&gt;&gt;&gt;</span> C → <span class="hl">one example test (or skip)</span>    <span class="dim"># low value</span>''',
      "take":"<b>②  Match tool to risk.</b> Reserve the expensive proof for the one property that must never fail; use Hypothesis broadly; barely test the trivial one."},
    "gate":{"btn":"③ the layered gate runs","html":'''<span class="prompt">&gt;&gt;&gt;</span> <span class="dim"># candidate passes the M26 gate only if ALL layers pass</span>
Hypothesis (real code): <span class="ok">PASS</span>
TLC (safety invariant, bounded): <span class="ok">PASS</span>, 4,096 states
refinement map checked: <span class="ok">PASS</span>
<span class="ok">→ ACCEPTED into the loop</span>''',
      "take":"<b>③  Layers combine.</b> Cheap Hypothesis catches most bugs; the targeted TLA+ invariant guards the one that matters; refinement carries it to the code. All must pass."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 78'><g font-family='monospace' font-size='9.5' text-anchor='middle'><rect x='40' y='14' width='200' height='52' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='140' y='34' fill='#5E5191' font-size='11'>TLA+ / TLC</text><text x='140' y='50' fill='#5E5191'>exhaustive · bounded model</text><text x='140' y='62' fill='#5E5191'>design time · high cost</text><rect x='280' y='14' width='200' height='52' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='380' y='34' fill='#1F6280' font-size='11'>Hypothesis</text><text x='380' y='50' fill='#1F6280'>random · real code</text><text x='380' y='62' fill='#1F6280'>every run · low cost</text></g></svg>",
     "note":"<b>Piece 1 — the two tools compared.</b> TLA+ is exhaustive but on an abstract model, at design time. Hypothesis is cheap and runs on real code, but only samples inputs. Different jobs."},
    {"viz":"<svg viewBox='0 0 520 78'><g font-family='monospace' font-size='9.5' text-anchor='middle'><rect x='30' y='14' width='150' height='50' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='105' y='34' fill='#C93B3B' font-size='10'>A: no double-assign</text><text x='105' y='50' fill='#C93B3B'>severity HIGH · subtle HIGH</text><rect x='190' y='14' width='140' height='50' rx='6' fill='#FCF3DC' stroke='#C99A12'/><text x='260' y='34' fill='#9A7208' font-size='10'>B: sort correct</text><text x='260' y='50' fill='#9A7208'>MED · LOW</text><rect x='340' y='14' width='150' height='50' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='415' y='34' fill='#1a5c38' font-size='10'>C: log length</text><text x='415' y='50' fill='#1a5c38'>LOW · LOW</text></g></svg>",
     "note":"<b>Piece 2 — score each property by risk.</b> Rate severity (how bad) and subtlety (how hard to catch by sampling). This score, not personal taste, drives the tool choice."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='9.5' text-anchor='middle'><rect x='30' y='16' width='150' height='36' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='105' y='32' fill='#5E5191'>A → TLA+ invariant</text><text x='105' y='46' fill='#5E5191'>+ refinement</text><rect x='190' y='16' width='140' height='36' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='38' fill='#1F6280'>B → Hypothesis</text><rect x='340' y='16' width='150' height='36' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='415' y='38' fill='#1a5c38'>C → example test</text></g></svg>",
     "note":"<b>Piece 3 — pick the tool per property.</b> High-high gets the formal proof; medium-low gets Hypothesis; low-low gets a single example test. Assurance budget spent by risk."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='9.5' text-anchor='middle'><rect x='140' y='8' width='240' height='22' rx='5' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='23' fill='#1F6280'>Hypothesis tests (every candidate)</text><rect x='140' y='36' width='240' height='22' rx='5' fill='#EDE9F8' stroke='#7C6DAA'/><text x='260' y='51' fill='#5E5191'>TLA+ safety invariant (bounded)</text><rect x='140' y='64' width='240' height='22' rx='5' fill='#FCF3DC' stroke='#C99A12'/><text x='260' y='79' fill='#9A7208'>refinement map (spec → code)</text></g></svg>",
     "note":"<b>Piece 4 — the three layers.</b> Cheap property tests on every candidate, one targeted formal invariant for the critical property, and the refinement bridge so the proof reaches the real code."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10.5' text-anchor='middle'><rect x='60' y='16' width='400' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>M26 loop: accept candidate only if ALL layers pass ✓</text></g></svg>",
     "note":"<b>Piece 5 — the guarded M26 loop.</b> A candidate enters the loop only if the cheap and the expensive checks all pass. Layered verification, allocated by risk — the staff-level answer."},
    {"viz":"<svg viewBox='0 0 520 56'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='14' width='220' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='150' y='32' fill='#C93B3B'>TLA+ spec for a log formatter</text><rect x='290' y='14' width='200' height='28' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='390' y='32' fill='#C93B3B'>effort wasted, spec drifts</text></g></svg>",
     "note":"<b>The trap — over-verifying low-risk code.</b> A formal spec for trivial code burns effort and drifts stale, eroding trust in all your specs. Gate formal effort behind an explicit severity-AND-subtlety check."},
  ],
  quiz=[
    {"q":"1. Which best captures the core difference between TLA+/TLC and Hypothesis?","opts":["TLA+ is newer than Hypothesis","TLA+ is exhaustive over a bounded abstract model at design time; Hypothesis is random/incomplete but runs on the real code","Hypothesis is exhaustive and TLA+ is random","they do exactly the same thing"],"ans":1,"fb":"TLA+ = exhaustive on a bounded model (design time). Hypothesis = random sampling on real code (every run). Complete-but-abstract vs. practical-but-partial."},
    {"q":"2. What should drive whether a property gets a full TLA+ proof?","opts":["how much you enjoy writing TLA+","the risk: high severity AND high subtlety justify the modelling cost; otherwise cheaper checks suffice","the alphabetical order of the module","whether the code compiles"],"ans":1,"fb":"Pick per risk. Reserve expensive formal effort for properties whose failure is catastrophic and hard to catch by sampling."},
    {"q":"3. Property A ('no double-assigned job', severity HIGH, subtlety HIGH from concurrency) — best fit?","opts":["one example test","a TLA+ safety invariant plus a refinement argument","skip it, too rare","a print statement"],"ans":1,"fb":"High severity and high subtlety (concurrency ordering) is exactly where exhaustive model checking earns its cost — plus refinement to reach the real code."},
    {"q":"4. What is the 'over-verification' failure mode?","opts":["running tests twice","pouring expensive formal effort into low-risk code, wasting time and letting specs drift stale — which erodes trust in all specs","using too many print statements","checking a property that is true"],"ans":1,"fb":"Formal verification of trivial code is wasted effort that drifts out of date. Gate formal work behind an explicit risk check — both severity and subtlety high."},
  ],
  fin={"em":"🧭","h3":"Day 6 complete — you can choose wisely!",
       "p":"You can now compare TLA+ (exhaustive, bounded, design-time) with Hypothesis (random, real code, cheap), pick a tool per risk, and layer both to guard the M26 ADRS loop — one safety invariant, a refinement, and broad property tests. You also saw the over-verification trap. Next: the <b>Module 28 review</b> — one self-check per day and a mixed quiz to lock it in."},
))

print("wrote day-01..06")

# ---------------- Review Gate ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m28-review", title="Module 28 · Review — Formal Methods", nav_title="Spiral · M28 Review",
  eyebrow="Module 28 · Verify · Review", h1="Module 28 Review: Formal Methods",
  lead="One gate before you move on. Six days of formal methods: what TLA+ is, states with Init and Next, invariants and TLC model checking, refinement to real code, property-based testing with Hypothesis, and how to choose per risk. Tick each self-check, then score the mixed quiz.",
  goal="<b>🎯 Passing this gate means:</b> you can explain formal verification vs. testing, write a tiny state machine, read an invariant and a counterexample trace, argue a refinement, write a property-based test, and pick the right tool for a given risk.",
  prev_href="day-06-synthesis.html", prev_label="Day 6 · When to Use Which",
  checks=[
    ["Day 1 · What Is TLA+?", "I can explain why a passing test suite is not a proof of correctness, and what a formal method describes (states + allowed moves, checked over all runs of a bounded model)."],
    ["Day 2 · States, Init & Next", "I can model a system as variables and write Init (legal starts) and Next (legal moves), and I can say what goes wrong when Next is under-constrained."],
    ["Day 3 · Invariants & TLC", "I can define an invariant, describe how TLC exhaustively searches reachable states and returns a counterexample trace, and explain why state-space explosion forces me to bound the model."],
    ["Day 4 · Refinement", "I can explain refinement (every concrete step matches an allowed abstract step) and how a refinement map connects code state to spec state — and how a too-forgiving map hides bugs."],
    ["Day 5 · Hypothesis", "I can state a property (round-trip or invariant), describe how Hypothesis generates and shrinks inputs, and recognise a weak or vacuous property that passes for the wrong reason."],
    ["Day 6 · When to Use Which", "I can compare TLA+ (exhaustive, bounded, design-time) with Hypothesis (random, real code, cheap), pick a tool per risk, and layer both onto the M26 loop."],
  ],
  quiz=[
    {"q":"1. Why is 'my test suite passed' a weaker claim than 'this is correct for every input'?","opts":["tests are written in the wrong language","a test suite only samples a finite set of inputs/orderings, so a rare interleaving can go untried","tests always have bugs","correctness cannot be defined"],"ans":1,"fb":"Testing samples finitely many runs. A formal method (on a bounded model) instead checks a property across every reachable state — closing that sampling gap for the model."},
    {"q":"2. For the spec Init: x=0, Next: x'=(x+1) mod 3, the invariant 'x < 2' is checked. What does TLC report?","opts":["no error — x stays below 2","a violation with the counterexample trace 0 → 1 → 2, because x = 2 is reachable and 2 < 2 is false","state-space explosion","that the spec won't parse"],"ans":1,"fb":"x = 2 is reachable, and 2 < 2 is false, so the invariant fails. TLC returns the exact path 0 → 1 → 2 — a concrete counterexample, not just 'false'."},
    {"q":"3. A model has 20 independent boolean flags. Roughly how large is the worst-case reachable state set, and what does that force you to do?","opts":["40 states; nothing special","about 1,000,000 states (2^20); you must bound the model so TLC can finish","2 states; add more flags","infinite; TLC handles it automatically"],"ans":1,"fb":"2^20 ≈ 1,048,576. Each flag doubles the count (state-space explosion), so you cap values and lengths to keep the search finite and feasible."},
    {"q":"4. TLC proved an invariant on your abstract spec. Your real Python still misbehaves. What is the missing piece?","opts":["a faster CPU","a refinement argument — the proof is about the abstract spec, and only refinement carries it to the concrete code","more example tests","a bigger bound only"],"ans":1,"fb":"The proof is about the model. Refinement (every concrete step maps to an allowed abstract step) is what transfers the guarantee to the real code — and a too-forgiving map can hide the very bug."},
    {"q":"5. A candidate needs 'no job is ever double-assigned' guaranteed — a catastrophic, concurrency-subtle property. Best verification choice?","opts":["one example test, since it's rare","a TLA+ safety invariant plus refinement, because severity AND subtlety are both high; Hypothesis alone can miss the rare interleaving","skip verification to save time","only a length check"],"ans":1,"fb":"High severity + high subtlety (concurrency ordering) is exactly where exhaustive model checking earns its cost, with refinement to reach the code. Reserve Hypothesis for the cheaper, lower-risk properties."},
  ],
  verdict_pass="you have working literacy in formal methods. You can state and check an invariant on a bounded model, argue a refinement to real code, write property-based tests, and — most importantly — choose the right tool for a given risk instead of over- or under-verifying. That is the staff-level skill this module builds toward.",
  verdict_fail="re-read the day whose self-check you could not tick honestly. The most common gaps: confusing 'tests passed' with 'proven' (Day 1), forgetting that TLC's proof is only about the <em>bounded</em> model (Day 3), or thinking a spec proof automatically covers the real code without refinement (Day 4).",
  complete_label="Mark Module 28 complete",
  score_target=4,
  fin={"em":"📐","h3":"Module 28 complete — formal methods unlocked!",
       "p":"You can now describe a system as states and allowed moves, prove an invariant holds across every reachable state of a bounded model with TLC, bridge that proof to real code with a refinement argument, hammer real code with property-based tests in Hypothesis, and choose per risk which assurance each property deserves. Your M26 verify loop now has a formal trust layer on top of empirical testing."},
))

print("wrote review — M28 complete")
