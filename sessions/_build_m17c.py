#!/usr/bin/env python3
"""Builder for M17c — State-Space Models & Attention Alternatives.
Authored from scratch. Builds on M3 (attention) and M17b (long-context).
4 lessons + review. Run: python3 sessions/_build_m17c.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m17c")
os.makedirs(OUT, exist_ok=True)
def L(name, spec): write_lesson(os.path.join(OUT, name), spec)
def R(name, spec): write_review(os.path.join(OUT, name), spec)

# ============================================================
# Day 1 — Selective Scan (S6 / Mamba)
# ============================================================
L("day-01-selective-scan.html", dict(
  qid="m17c-d01-s6", title="Module 17c · Day 1 — Selective Scan (S6 / Mamba)", nav_title="Spiral · M17c Day 1",
  eyebrow="Module 17c · Systems · Day 1", h1="Selective Scan (S6 / Mamba)",
  lead="Attention reads every past token again for every new token, so its cost grows with the square of the sequence length. A <b>state-space model</b> takes a different road: it keeps a small fixed-size memory and updates it one token at a time, like a running total. Cost grows only linearly. The Mamba trick — the S6 layer — makes that little memory <em>choose</em> what to remember based on the input. Today you build that running memory from a single equation.",
  goal="<b>🎯 By the end, you'll be able to:</b> write the state-space recurrence <code>h_t = A·h_{t-1} + B·x_t</code>, <code>y_t = C·h_t</code>, explain what \"selective\" (input-dependent A,B,C) adds, say why the cost is O(n) not O(n²), and name the failure mode of a fixed-size state.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-linear-attention.html", next_label="Linear Attention, RWKV, RetNet",
  sections=[
    {"title":"A memory that updates one step at a time","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> you know attention costs O(n²) because every token looks at every other token (M3), and why very long sequences make that painful (M17b). Today's model avoids that entirely.</div></div>
     <h4>The problem attention has</h4>
     <p>To generate token number 10,000, an attention model re-reads all 9,999 tokens before it. Do that for every token and the work grows with the square of the length. Twice the text costs four times the compute.</p>
     <h4>The state-space idea</h4>
     <p>A <span class="term" data-tip="A model that carries a small fixed-size hidden vector (the state) and updates it one token at a time with a linear rule, instead of re-reading all past tokens.">state-space model (SSM)</span> keeps a small memory vector called the <span class="term" data-tip="A fixed-size vector that summarizes everything the model has seen so far. It is updated at each step and never grows with sequence length.">state</span> <code>h</code>. At each step it mixes the old state with the new input, and reads an output from the state. Nothing re-reads the past — the past is <em>already inside</em> the state.</p>
     <h4>What "selective" means (the Mamba part)</h4>
     <p>Classic SSMs use the same update rule for every token. <span class="term" data-tip="The selective state-space layer used in Mamba. The update matrices A, B, C are computed from the current input, so the model can choose what to keep or forget per token.">S6 (Mamba's selective scan)</span> makes the update rule depend on the current input. That lets the model <em>choose</em>, token by token, what to store and what to forget — the missing power that made SSMs finally match attention on language.</p>''',
     "gotit":"Got the idea"},
    {"title":"A running score in a card game","body":
     '''<div class="relate">
       <div class="card"><span class="big">🎲</span><h5>You keep a running total</h5><p>You play cards and keep one number in your head: your score so far. Each new card, you update that one number — you never re-count every card you've ever played. That single number is the <b>state</b> <code>h</code>.</p></div>
       <div class="card"><span class="big">🧠</span><h5>You decide what matters</h5><p>A smart player weights cards differently: a wild card changes how much the next card counts. Reacting to <i>this</i> card when deciding how to update your total is exactly what "selective" (input-dependent) means in S6.</p></div>
     </div>
     <p><strong>In one line:</strong> keep one small memory, update it each step from the new token, and let the token itself decide how the update happens.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: your card total is a single number, but the state <code>h</code> is a whole vector, and the "update" is a matrix multiply, not simple addition. The state can hold several running summaries at once — but it is still a <b>fixed</b> size, no matter how long the game runs.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Run a tiny scan by hand","body":'''<p>We will run the recurrence on three inputs with tiny numbers, then read the outputs, then see why the cost is linear. <strong>Click all three.</strong></p>'''},
    {"title":"The recurrence, built piece by piece","body":
     '''<h4>Step 1 — in words</h4>
     <p>Keep a state <code>h</code>. At each step: shrink the old state a little (that's <code>A</code>), add in the new input (that's <code>B·x</code>), and read an output from the new state (that's <code>C</code>). Two lines, forever.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       h<sub>t</sub> = A · h<sub>t−1</sub> + B · x<sub>t</sub>      <span class="dim"># update the state from old state + new input</span><br>
       y<sub>t</sub> = C · h<sub>t</sub>                    <span class="dim"># read the output from the new state</span><br>
       <span class="dim"># A = keep/forget matrix, B = write-in matrix, C = read-out matrix, x = input</span><br>
       <span class="dim"># SELECTIVE (S6): A, B, C are computed FROM x_t, so they change every token</span>
     </div></div>
     <h4>Step 3 — worked example (scalars, so you can check every number)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       let A = 0.5, B = 1, C = 2, and inputs x = [4, 2, 1], start h<sub>0</sub> = 0<br><br>
       t=1: h = 0.5·0 + 1·4 = <span class="hl">4</span>      y = 2·4 = <span class="hl">8</span><br>
       t=2: h = 0.5·4 + 1·2 = <span class="hl">4</span>      y = 2·4 = <span class="hl">8</span><br>
       t=3: h = 0.5·4 + 1·1 = <span class="hl">3</span>      y = 2·3 = <span class="hl">6</span><br>
       <span class="dim"># notice: the old input 4 still echoes at t=2 and t=3, faded by A each step — that's memory</span>
     </div></div>
     <p>Because each step touches only the small state (not the whole past), the total work is one update per token: <code>O(n)</code> time, and the memory is one fixed-size state, <code>O(1)</code>. Attention is <code>O(n²)</code> time and an <code>O(n)</code> cache. That is the whole efficiency story.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — fixed-size state loses exact recall.</b> The state is one small vector that must summarize everything seen so far. Old details get compressed and overwritten. Ask the model to copy a phone number from 5,000 tokens back and it often can't — the exact digits were squeezed out to make room. The loss curve looks fine; only a recall-heavy test exposes it. Attention, which keeps every token, does not have this problem.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — linear time & long context vs precise copying.</b> The SSM gives you <code>O(n)</code> compute and a constant memory footprint, so you can process very long inputs cheaply. You pay for it with weaker <i>exact</i> retrieval: anything that needs pulling a specific earlier token back verbatim is where a fixed state struggles and attention wins.</div></div>''',
     "gotit":"Got the recurrence"},
    {"title":"Selective scan, assembled","body":'''<p>Here is the S6 layer built up: the fixed state, the update, the read-out, the input-dependent A/B/C, and the linear cost. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Run your own scan","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m17c_selective_scan.py</code>. Implement the recurrence <code>h_t = A·h_{t-1} + B·x_t</code>, <code>y_t = C·h_t</code> as a plain Python loop over a sequence. First use fixed A,B,C; then make them functions of <code>x_t</code> (selective). Print the state at every step and time it against a naive O(n²) attention as n grows.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 17c Day 1 artifact.
Create experiments/foundations/m17c_selective_scan.py: implement the state-space recurrence h_t = A*h_{t-1} + B*x_t, y_t = C*h_t as a scan over a sequence. First with fixed A,B,C, then a "selective" version where A,B,C are computed from x_t. Print the hidden state at each step. Then benchmark runtime vs sequence length n for the scan (should be O(n)) against naive attention (O(n^2)) and plot both.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m17c/day-01-log.md</code>: at what sequence length did the O(n²) attention start to blow up while the O(n) scan stayed flat?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "run":{"btn":"① run the recurrence","html":'''<span class="prompt">&gt;&gt;&gt;</span> A, B, C = 0.5, 1, 2      <span class="dim"># keep-half, write-in, read-out</span>
<span class="prompt">&gt;&gt;&gt;</span> x = [4, 2, 1];  h = 0
<span class="prompt">&gt;&gt;&gt;</span> h = A*h + B*x[0]         <span class="dim"># t=1</span>
<span class="hl">h = 4</span>''',
      "take":"<b>①  One update per token.</b> The state <code>h</code> mixes the faded old state with the new input. Here the first input 4 becomes the state."},
    "read":{"btn":"② read the outputs","html":'''<span class="prompt">&gt;&gt;&gt;</span> # step through all three
t=1: h = 0.5*0 + 4 = 4   y = 2*4 = <span class="hl">8</span>
t=2: h = 0.5*4 + 2 = 4   y = 2*4 = <span class="hl">8</span>
t=3: h = 0.5*4 + 1 = 3   y = 2*3 = <span class="hl">6</span>
<span class="dim"># the first 4 still echoes at t=2 and t=3 — faded memory</span>''',
      "take":"<b>②  The state carries the past.</b> Output <code>y_t = C·h_t</code>. The old input 4 keeps influencing later outputs, shrunk by A each step — that is the model's memory."},
    "cost":{"btn":"③ why it's O(n)","html":'''<span class="prompt">&gt;&gt;&gt;</span> steps_to_generate_n_tokens()
SSM scan:   n updates          <span class="ok"># O(n) time, O(1) state</span>
attention:  1+2+...+n = n(n+1)/2  <span class="bad"># O(n^2) time, O(n) cache</span>
<span class="dim"># n=10000 → scan ~10^4 steps, attention ~5·10^7</span>''',
      "take":"<b>③  Linear vs quadratic.</b> The scan does one update per token; attention re-reads all past tokens each step. For long sequences that is a huge gap."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='210' y='20' width='100' height='34' rx='8' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='36' fill='#5E5191'>state h</text><text x='260' y='50' fill='#6B645E'>fixed size</text></g></svg>",
     "note":"<b>Start with one small state <code>h</code>.</b> A fixed-size memory vector. It never grows, no matter how long the sequence gets."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='40' y='26' width='90' height='30' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='85' y='45' fill='#5A544E'>x_t (input)</text><text x='150' y='45' fill='#9A7208'>→ B·x_t →</text><rect x='250' y='26' width='110' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='305' y='45' fill='#5E5191'>+ A·h_(t-1)</text><text x='395' y='45' fill='#5E5191'>= h_t</text></g></svg>",
     "note":"<b>Update: <code>h_t = A·h_(t-1) + B·x_t</code>.</b> Fade the old state with A, write in the new input with B. This is the whole memory step."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='170' y='20' width='90' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='215' y='39' fill='#5E5191'>h_t</text><text x='285' y='39' fill='#1F6280'>→ C·h_t →</text><rect x='380' y='20' width='90' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='425' y='39' fill='#1F6280'>y_t</text></g></svg>",
     "note":"<b>Read: <code>y_t = C·h_t</code>.</b> The output is just a view of the current state. No looking back at old tokens — everything needed is inside <code>h</code>."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='60' y='30' width='90' height='26' rx='6' fill='#F5F0E8' stroke='#6B645E'/><text x='105' y='47' fill='#5A544E'>x_t</text><text x='185' y='20' fill='#C99A12'>compute</text><rect x='250' y='30' width='210' height='26' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='355' y='47' fill='#9A7208'>A(x_t), B(x_t), C(x_t)</text></g></svg>",
     "note":"<b>Selective (S6): A, B, C come FROM the input.</b> Each token computes its own update rule, so the model can choose what to keep and what to forget — the Mamba upgrade."},
    {"viz":"<svg viewBox='0 0 520 90'><g font-family='monospace' font-size='10'><text x='60' y='30' fill='#1a5c38'>SSM scan:</text><polyline points='120,32 180,32 240,32 300,32 360,32 420,32' fill='none' stroke='#2D8B55' stroke-width='2.5'/><text x='440' y='36' fill='#1a5c38'>O(n)</text><text x='60' y='68' fill='#C93B3B'>attention:</text><polyline points='120,72 180,68 240,60 300,48 360,32 420,10' fill='none' stroke='#C93B3B' stroke-width='2.5'/><text x='440' y='14' fill='#C93B3B'>O(n²)</text></g></svg>",
     "note":"<b>The payoff: linear vs quadratic.</b> One update per token means cost grows with n, not n². The whole reason SSMs are back at frontier labs for long context."},
  ],
  quiz=[
    {"q":"1. What does the state <code>h</code> in an SSM hold?","opts":["a copy of every past token","a small fixed-size summary of everything seen so far","the attention scores","the vocabulary"],"ans":1,"fb":"The state is one fixed-size vector that summarizes the past. It never grows with sequence length — that is what makes the scan cheap."},
    {"q":"2. Run <code>h_t = A·h_(t-1) + B·x_t</code> with A=0.5, B=1, h_0=0 on inputs x=[4, 2]. What is h at t=2?","opts":["6","4","2","8"],"ans":1,"fb":"t=1: h = 0.5·0 + 1·4 = 4. t=2: h = 0.5·4 + 1·2 = 2 + 2 = 4. The faded 4 (→2) plus the new 2 gives 4."},
    {"q":"3. What does \"selective\" (the S6 / Mamba upgrade) change?","opts":["it removes the state","it makes A, B, C depend on the current input, so the model chooses what to keep or forget","it adds softmax back","it makes the state grow with n"],"ans":1,"fb":"Selective means input-dependent A, B, C. Each token picks its own update rule — the missing ingredient that let SSMs match attention on language."},
    {"q":"4. You feed a long document and ask the model to repeat an exact 12-digit code from near the start. It gets it wrong even though training loss looked great. Why?","opts":["the learning rate was too high","the fixed-size state compressed away the exact digits — precise recall is the SSM's weak spot","attention is broken","the code was too short"],"ans":1,"fb":"A single fixed state must summarize everything, so exact old details get overwritten. Precise copying/retrieval is where a fixed state loses to attention."},
  ],
  fin={"em":"🌀","h3":"Day 1 complete — you built a linear-time memory!",
       "p":"You now know the state-space recurrence <code>h_t = A·h_(t-1) + B·x_t</code>, <code>y_t = C·h_t</code>, why it costs O(n) instead of O(n²), and what \"selective\" (input-dependent A,B,C) adds in Mamba's S6. You also met the core weakness: a fixed state loses exact recall. Next: <b>Day 2</b> — how to turn attention itself into a linear-time recurrence."},
))

# ============================================================
# Day 2 — Linear Attention, RWKV, RetNet
# ============================================================
L("day-02-linear-attention.html", dict(
  qid="m17c-d02-linattn", title="Module 17c · Day 2 — Linear Attention, RWKV, RetNet", nav_title="Spiral · M17c Day 2",
  eyebrow="Module 17c · Systems · Day 2", h1="Linear Attention, RWKV, RetNet",
  lead="Yesterday you built a recurrence from scratch. Today you reach the same linear-time destination from the <em>other</em> direction: start with softmax attention and carefully remove the one part that forces O(n²) — the softmax. Drop it the right way and attention turns into a running sum you can update one token at a time. That single trick is the engine behind RWKV and RetNet, models that train like a transformer but run like an RNN.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain how dropping softmax (with kernel feature maps) lets attention be written as a recurrence, what the associativity trick is, why RWKV and RetNet are attention/RNN hybrids, and the quality gap versus true softmax attention.",
  prev_href="day-01-selective-scan.html", prev_label="Selective Scan (S6 / Mamba)", next_href="day-03-hybrids.html", next_label="Hybrids (Jamba, Griffin)",
  sections=[
    {"title":"Attention, but as a running sum","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> softmax attention — each output is a weighted average of value vectors, with weights from <code>softmax(q·k)</code> (M3) — and yesterday's idea that a running state gives O(n).</div></div>
     <h4>Where the O(n²) comes from</h4>
     <p>In standard attention, output for token <code>t</code> is <code>softmax(q<sub>t</sub>·k<sub>1..t</sub>) · v<sub>1..t</sub></code>. The <span class="term" data-tip="A function that turns a list of scores into positive weights that sum to 1. It couples every query with every key, which forces the all-pairs O(n²) cost.">softmax</span> normalizes over <em>all</em> keys, so you cannot compute it without touching every past key. That coupling is what makes attention quadratic.</p>
     <h4>The linear-attention move</h4>
     <p>Replace <code>softmax(q·k)</code> with a plain product of two <span class="term" data-tip="A simple function φ applied to queries and keys (for example, making them positive) so that φ(q)·φ(k) approximates the attention weight without a softmax.">feature maps</span> <code>φ(q)·φ(k)</code>. Now there is no all-key normalizer, so you can reorder the multiplications and keep a <em>running sum</em> of <code>φ(k)·vᵀ</code>. That running sum is a fixed-size state — exactly like yesterday. This is <span class="term" data-tip="Attention rewritten without softmax, using feature maps, so it can be computed as a running sum updated one token at a time: O(n) instead of O(n²).">linear attention</span>.</p>
     <h4>Two models that use it</h4>
     <p><span class="term" data-tip="Receptance Weighted Key Value: a model that trains in parallel like a transformer but runs as an RNN at inference, using a linear-attention-style recurrence with a learned decay.">RWKV</span> and <span class="term" data-tip="Retentive Network: an attention alternative with a 'retention' mechanism that has both a parallel form (for training) and a recurrent form (for O(1)-per-token inference), using a fixed decay.">RetNet</span> are the two best-known. Both are <em>hybrids</em>: they look like attention when training (fast, parallel) but collapse to a cheap recurrence when generating.</p>''',
     "gotit":"Got the idea"},
    {"title":"Two ways to keep a class average","body":
     '''<div class="relate">
       <div class="card"><span class="big">📋</span><h5>Re-add every score (slow)</h5><p>To get the class average after each new test, you re-add all scores and divide. With n students that is n additions every time — the softmax way, O(n²) overall.</p></div>
       <div class="card"><span class="big">➕</span><h5>Keep a running sum (fast)</h5><p>Or keep two numbers: the total so far and the count. Each new score, add it to the total, bump the count, divide. One step per student — the linear-attention way. Same answer, far less work.</p></div>
     </div>
     <p><strong>In one line:</strong> if you can keep a running sum instead of re-summing everything, attention becomes a one-step-per-token update — a fixed-size state you carry forward.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a class average is exact, but linear attention only <i>approximates</i> softmax attention. The feature map φ is a stand-in, not the real thing, so the "average" it keeps is a lossy version of what softmax would compute.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"See the associativity trick","body":'''<p>We will write softmax attention, drop the softmax, then reorder the matrix multiplies to expose the running sum. <strong>Click all three.</strong></p>'''},
    {"title":"Dropping softmax with feature maps","body":
     '''<h4>Step 1 — in words</h4>
     <p>Softmax attention computes weights over all keys, then averages values. If we swap softmax for a product of feature maps, the query no longer needs the whole key list — we can gather the keys and values into one running summary and reuse it.</p>
     <h4>Step 2 — the formula (the associativity trick)</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       softmax form:  y<sub>t</sub> = Σ<sub>i≤t</sub> softmax(q<sub>t</sub>·k<sub>i</sub>) · v<sub>i</sub>   <span class="dim"># needs all keys → O(n²)</span><br>
       linear form:   y<sub>t</sub> = φ(q<sub>t</sub>) · ( Σ<sub>i≤t</sub> φ(k<sub>i</sub>) v<sub>i</sub><sup>ᵀ</sup> )   <span class="dim"># move the sum INSIDE</span><br>
       let S<sub>t</sub> = Σ<sub>i≤t</sub> φ(k<sub>i</sub>) v<sub>i</sub><sup>ᵀ</sup>   →   S<sub>t</sub> = S<sub>t−1</sub> + φ(k<sub>t</sub>) v<sub>t</sub><sup>ᵀ</sup>   <span class="dim"># a running-sum STATE</span><br>
       so:  y<sub>t</sub> = φ(q<sub>t</sub>) · S<sub>t</sub>            <span class="dim"># one update per token → O(n)</span>
     </div></div>
     <p>The whole trick is <span class="term" data-tip="The rule (AB)C = A(BC). Here it lets us compute φ(k)·vᵀ first and sum it into a state, instead of computing q·k for every pair first.">associativity</span>: because there is no softmax gluing q to every k, we may group <code>φ(k)·vᵀ</code> first and accumulate it. The state <code>S</code> is a fixed-size matrix — the past, compressed.</p>
     <h4>Step 3 — worked example (scalar, 1-D)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       take φ(x)=x. keys k=[1, 2], values v=[10, 20], query q_2 = 3<br><br>
       softmax-free numerator = q_2·(k_1·v_1 + k_2·v_2) = 3·(1·10 + 2·20) = 3·50 = <span class="hl">150</span><br>
       via running sum: S = k_1·v_1 + k_2·v_2 = 10 + 40 = <span class="hl">50</span>,  then y = q_2·S = 3·50 = <span class="hl">150</span><br>
       <span class="dim"># same answer — but S was built ONE key at a time and reused, never re-summing</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — the quality gap.</b> Feature maps only approximate softmax's sharp, selective weighting. Softmax can put almost all weight on one key (a laser focus); φ(q)·φ(k) tends to spread weight out (a blur). On tasks needing sharp attention to one exact token, linear attention quietly underperforms — no crash, just lower accuracy that pure perplexity can hide. RWKV and RetNet add learned <b>decay</b> to sharpen recency, but the gap on precise, long-range lookup remains.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — O(n) inference vs expressivity.</b> Linear attention gives constant memory and one cheap update per token at generation time (no growing KV cache), which is fantastic for long, streaming decoding. You trade away some of softmax's expressive, sharp, content-based selection — the ability to lock onto exactly the right past token. Faster and lighter, but a little blurrier.</div></div>''',
     "gotit":"Got the trick"},
    {"title":"Linear attention, assembled","body":'''<p>Here it is built up: softmax attention, dropping softmax, the reorder, the running-sum state, and the RNN-style read-out. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Turn attention into a recurrence","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m17c_linear_attention.py</code>. Implement softmax attention and a linear-attention version (φ(x)=elu(x)+1). Verify the linear version can be computed as a running sum <code>S_t = S_{t-1} + φ(k_t)v_tᵀ</code>, <code>y_t = φ(q_t)·S_t</code>. Compare outputs and time both as n grows.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 17c Day 2 artifact.
Create experiments/foundations/m17c_linear_attention.py: implement (1) standard softmax attention, (2) linear attention with feature map phi(x)=elu(x)+1 in both a parallel form and a recurrent running-sum form S_t = S_{t-1} + phi(k_t) v_t^T, y_t = phi(q_t) . S_t. Show the parallel and recurrent forms give the same output, and benchmark runtime vs sequence length (softmax O(n^2) vs linear O(n)). Add a note on the softmax quality gap.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m17c/day-02-log.md</code>: did the recurrent and parallel linear-attention forms match exactly, and where did softmax attention start losing on speed?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "soft":{"btn":"① softmax attention","html":'''<span class="prompt">&gt;&gt;&gt;</span> y_t = sum_i softmax(q_t . k_i) * v_i
<span class="dim"># the softmax normalizes over ALL keys 1..t</span>
<span class="bad"># cannot skip any past key → O(n^2)</span>''',
      "take":"<b>①  Softmax couples everything.</b> The weight on each key depends on all the others, so you must touch every past key. That coupling is the O(n²) cost."},
    "drop":{"btn":"② drop the softmax","html":'''<span class="prompt">&gt;&gt;&gt;</span> y_t = sum_i (phi(q_t) . phi(k_i)) * v_i
<span class="dim"># replace softmax(q.k) with phi(q).phi(k)</span>
<span class="hl"># no all-key normalizer anymore</span>''',
      "take":"<b>②  Feature maps replace softmax.</b> With φ(q)·φ(k) instead of softmax, the query no longer depends on the full key list. This is the key that unlocks the reorder."},
    "reorder":{"btn":"③ reorder → running sum","html":'''<span class="prompt">&gt;&gt;&gt;</span> S_t = S_(t-1) + phi(k_t) * v_t^T   <span class="dim"># accumulate</span>
<span class="prompt">&gt;&gt;&gt;</span> y_t = phi(q_t) . S_t              <span class="dim"># read out</span>
<span class="ok"># fixed-size state, one update per token → O(n)</span>''',
      "take":"<b>③  Associativity gives a recurrence.</b> Group φ(k)·vᵀ first, sum it into a state S, and read with φ(q). That is a fixed-size memory updated one token at a time — an RNN."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>y_t = Σ softmax(q_t·k_i)·v_i</text></g></svg>",
     "note":"<b>Start with softmax attention.</b> Each output is a softmax-weighted average of all past values. The softmax over every key is what forces O(n²)."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='80' y='16' width='360' height='30' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2' stroke-dasharray='5,3'/><text x='260' y='36' fill='#C93B3B'>softmax(q·k) → φ(q)·φ(k)</text></g></svg>",
     "note":"<b>Drop the softmax.</b> Swap it for a product of feature maps φ. Now nothing normalizes over all keys — the query is free of the full key list."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='100' y='16' width='320' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>y_t = φ(q_t)·( Σ φ(k_i) v_iᵀ )</text></g></svg>",
     "note":"<b>Reorder (associativity).</b> Group φ(k)·vᵀ first, then multiply by φ(q). Same math, but now the inner sum can be built up incrementally."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='36' fill='#5E5191'>S_t = S_(t-1) + φ(k_t) v_tᵀ</text></g></svg>",
     "note":"<b>The running-sum state.</b> The inner sum becomes a fixed-size matrix S, updated by adding one term per token. This is the past, compressed — just like yesterday."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='150' y='18' width='220' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='38' fill='#1a5c38'>y_t = φ(q_t)·S_t  → O(n)</text></g></svg>",
     "note":"<b>Read out like an RNN.</b> Output is the query times the state. Train in parallel like a transformer, generate as a cheap recurrence — that is RWKV and RetNet."},
  ],
  quiz=[
    {"q":"1. What single part of attention is removed to make it linear-time?","opts":["the value vectors","the softmax over all keys","the embeddings","the residual connection"],"ans":1,"fb":"Softmax normalizes over every key, coupling all pairs and forcing O(n²). Replacing it with feature maps φ(q)·φ(k) breaks that coupling."},
    {"q":"2. What does the associativity trick let you do?","opts":["skip the values","compute φ(k)·vᵀ first and accumulate it into a running-sum state, instead of computing q·k for every pair","use two softmaxes","grow the state with n"],"ans":1,"fb":"Without softmax gluing q to all k, you can regroup the products: build S = Σ φ(k)vᵀ once and reuse it — a fixed-size state, one update per token."},
    {"q":"3. With φ(x)=x, keys k=[1,2], values v=[10,20], the running-sum state S = Σ k_i·v_i equals:","opts":["30","50","150","200"],"ans":1,"fb":"S = 1·10 + 2·20 = 10 + 40 = 50. Then a query q would read y = q·S — one multiply, no re-summing."},
    {"q":"4. Why are RWKV and RetNet called attention/RNN hybrids?","opts":["they use no state","they train in parallel like a transformer but generate as a cheap O(1)-per-token recurrence like an RNN","they only work on images","they need O(n²) at inference"],"ans":1,"fb":"They have a parallel form for fast training and an equivalent recurrent form (a running-sum state, often with learned decay) for cheap streaming inference."},
  ],
  fin={"em":"🔁","h3":"Day 2 complete — you linearized attention!",
       "p":"You reached linear time from the attention side: drop softmax, use feature maps, and the associativity trick turns attention into a running-sum state — <code>S_t = S_{t-1} + φ(k_t)v_tᵀ</code>, <code>y_t = φ(q_t)·S_t</code>. That is the engine of RWKV and RetNet, hybrids that train like transformers and run like RNNs. You also saw the softmax quality gap. Next: <b>Day 3</b> — mixing a few attention layers with many SSM layers to get the best of both."},
))

# ============================================================
# Day 3 — Hybrids (Jamba, Griffin)
# ============================================================
L("day-03-hybrids.html", dict(
  qid="m17c-d03-hybrid", title="Module 17c · Day 3 — Hybrids (Jamba, Griffin)", nav_title="Spiral · M17c Day 3",
  eyebrow="Module 17c · Systems · Day 3", h1="Hybrids (Jamba, Griffin)",
  lead="Days 1 and 2 gave you a hard choice: SSMs are cheap on long context but weak at exact recall; attention nails recall but costs O(n²). Real frontier models refuse to choose. They <b>interleave</b> a few attention layers among many cheap recurrent (SSM or linear-attention) layers. The recurrent layers carry the long context for almost free; the sprinkled attention layers restore the precise lookup. That is Jamba and Griffin — and it is why hybrids ship in production today.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why interleaving attention with SSM/recurrent layers gives long-context efficiency AND recall, work out a layer budget, name the failure mode of too few attention layers, and reason about the attention:SSM ratio.",
  prev_href="day-02-linear-attention.html", prev_label="Linear Attention, RWKV, RetNet", next_href="day-04-duality.html", next_label="Recurrence and Attention Duality",
  sections=[
    {"title":"Don't choose — mix","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (SSMs: O(n), fixed state, weak exact recall) and Day 2 (linear attention / RWKV / RetNet, same trade-off). Today: combine them with real attention.</div></div>
     <h4>The two extremes, and their costs</h4>
     <p>A pure-attention model recalls any past token but pays O(n²) and a KV cache that grows with length. A pure-SSM model is O(n) with a tiny fixed state but can't reliably pull back an exact earlier token. Each is missing what the other has.</p>
     <h4>The hybrid idea</h4>
     <p>A <span class="term" data-tip="A model that stacks two kinds of layers: many cheap recurrent (SSM or linear-attention) layers plus a few full-attention layers, to get long-context efficiency and precise recall together.">hybrid</span> stacks both kinds of layer. Most layers are cheap recurrent layers that move the long context along at O(n). A <em>small number</em> of full-attention layers are placed among them, giving the model occasional access to every past token for exact lookup.</p>
     <h4>Two real examples</h4>
     <ul>
       <li><span class="term" data-tip="A production hybrid LLM that interleaves Mamba (SSM) layers with a small fraction of Transformer attention layers, plus mixture-of-experts, for long context at low cost.">Jamba</span> — interleaves Mamba (SSM) blocks with a few Transformer attention blocks (roughly one attention layer for every several Mamba layers).</li>
       <li><span class="term" data-tip="A hybrid architecture mixing gated linear recurrences (the RG-LRU) with local sliding-window attention, for efficient long-context modeling.">Griffin</span> — mixes gated linear recurrences with <em>local</em> (sliding-window) attention, so even its attention is cheap.</li>
     </ul>''',
     "gotit":"Got the idea"},
    {"title":"A crew of runners and a librarian","body":
     '''<div class="relate">
       <div class="card"><span class="big">🏃</span><h5>Runners carry the load</h5><p>Most of a relay team are fast runners who pass a single baton (the state) forward cheaply, step after step. They keep the race moving over any distance. Those are the SSM/recurrent layers — O(n), light.</p></div>
       <div class="card"><span class="big">📚</span><h5>A librarian for exact facts</h5><p>Every so often the baton passes to a librarian who can look up <i>any</i> page from the whole race so far. Slow, but precise. You only need a few of them. Those are the attention layers — O(n²), but rare.</p></div>
     </div>
     <p><strong>In one line:</strong> let cheap recurrent layers do most of the work, and drop in a few attention layers for the moments that need exact, whole-history lookup.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: the runners and librarian are separate people, but in a hybrid the layers are <i>stacked in sequence</i> — every token flows through all of them, recurrent and attention alike. It is not a hand-off; it is a pipeline where each layer type does a different job.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Budget the layers","body":'''<p>We will set a layer budget, compute the cost of all-attention vs a hybrid, then see what happens as we cut attention too far. <strong>Click all three.</strong></p>'''},
    {"title":"Working out the layer budget","body":
     '''<h4>Step 1 — in words</h4>
     <p>Pick a total number of layers. Decide a ratio: how many recurrent layers per attention layer. More attention means better recall but more cost; fewer means cheaper but weaker recall. The right ratio depends on how recall-heavy your task is.</p>
     <h4>Step 2 — the cost picture</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       total layers = L,  ratio = r recurrent layers per 1 attention layer<br>
       attention layers  A = L / (r + 1)<br>
       recurrent layers  = L − A<br>
       <span class="dim"># cost ≈ A·O(n²) + (L−A)·O(n).  Fewer A → cost dominated by the cheap O(n) layers</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       L = 32 layers, ratio r = 7 (7 SSM per 1 attention)<br><br>
       attention layers  A = 32 / (7 + 1) = <span class="hl">4</span><br>
       SSM layers        = 32 − 4 = <span class="hl">28</span><br>
       so only <span class="hl">4 of 32</span> layers (12.5%) pay the O(n²) cost;<br>
       28 layers are cheap O(n) — most of the model runs at linear cost,<br>
       yet those 4 attention layers still give exact whole-history lookup.
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — too few attention layers cripples recall.</b> Push the ratio too high (say 1 attention layer in 32) to save cost and the model loses its ability to do exact retrieval and in-context lookup. It still reads fluently and its perplexity looks fine, so the failure is invisible on generic benchmarks — but a "find the fact buried at position 4,000" test collapses. The cheap layers can't recover the exact token the missing attention layers would have fetched.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — the attention:SSM ratio.</b> This single ratio is the hybrid's main dial. More attention layers → stronger recall and in-context learning, but higher cost and a larger KV cache. More SSM layers → cheaper, longer context, smaller memory, but weaker precise retrieval. Jamba-style designs land around 1 attention per 7–8 SSM layers as a sweet spot; recall-heavy tasks want more attention, streaming/long-doc tasks want less.</div></div>''',
     "gotit":"Got the budget"},
    {"title":"The hybrid stack, assembled","body":'''<p>Here it is built up: the two layer types, the interleaving pattern, the layer budget, the cost split, and the recall it buys. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Design a hybrid stack","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m17c_hybrid.py</code>. Build a small stack that interleaves your Day 1 SSM layer with a real attention layer at a chosen ratio. Run a synthetic "needle in a haystack" recall test at several ratios (e.g. 1:3, 1:7, 1:31) and plot recall accuracy and cost vs ratio.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 17c Day 3 artifact.
Create experiments/foundations/m17c_hybrid.py: build a small layer stack that interleaves an SSM/linear-attention layer with a full-attention layer at a configurable ratio (SSM:attention). Run a synthetic needle-in-a-haystack recall task at ratios like 1:3, 1:7, 1:31 (attention layers per total), and plot recall accuracy vs ratio and approximate FLOPs vs ratio. Show that dropping attention too far collapses recall while barely helping cost.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m17c/day-03-log.md</code>: at what attention:SSM ratio did recall accuracy fall off a cliff?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "budget":{"btn":"① set a layer budget","html":'''<span class="prompt">&gt;&gt;&gt;</span> L = 32; ratio = 7   <span class="dim"># 7 SSM per 1 attention</span>
<span class="prompt">&gt;&gt;&gt;</span> A = L // (ratio + 1)
<span class="hl">A = 4 attention layers, 28 SSM layers</span>''',
      "take":"<b>①  Most layers are cheap.</b> With a 7:1 ratio, only 4 of 32 layers are attention. The other 28 run at O(n) — the model is mostly linear-cost."},
    "cost":{"btn":"② compare the cost","html":'''<span class="prompt">&gt;&gt;&gt;</span> # long sequence, n large
all-attention: 32 layers x O(n^2)   <span class="bad"># expensive</span>
hybrid:         4 x O(n^2) + 28 x O(n)  <span class="ok"># mostly linear</span>
<span class="dim"># the 4 attention layers still see every past token</span>''',
      "take":"<b>②  Best of both.</b> Only 4 layers pay the quadratic cost, so long context is affordable — yet those 4 give exact whole-history lookup the SSM layers lack."},
    "toolow":{"btn":"③ cut attention too far","html":'''<span class="prompt">&gt;&gt;&gt;</span> ratio = 31   <span class="dim"># only 1 attention layer in 32</span>
perplexity: <span class="hl">looks fine</span>
needle-recall @ pos 4000: <span class="bad">FAILS</span>
<span class="dim"># barely cheaper, but recall collapses — invisible on generic evals</span>''',
      "take":"<b>③  Too few attention layers hurts recall.</b> Going from 4 to 1 attention layer saves little cost but breaks exact retrieval. Fluency hides it; a recall test exposes it."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='60' y='24' width='170' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='145' y='43' fill='#5E5191'>SSM layer — O(n)</text><rect x='290' y='24' width='170' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='375' y='43' fill='#1F6280'>attention — O(n²)</text></g></svg>",
     "note":"<b>Two layer types.</b> Cheap recurrent (SSM/linear-attention) layers for long context, and full-attention layers for exact recall. Each fixes the other's weakness."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='20' y='22' width='40' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><rect x='66' y='22' width='40' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><rect x='112' y='22' width='40' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><rect x='158' y='22' width='40' height='24' rx='4' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='178' y='38' fill='#1F6280'>ATT</text><rect x='204' y='22' width='40' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><rect x='250' y='22' width='40' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><rect x='296' y='22' width='40' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><rect x='342' y='22' width='40' height='24' rx='4' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='362' y='38' fill='#1F6280'>ATT</text><rect x='388' y='22' width='40' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/><rect x='434' y='22' width='40' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA'/></g></svg>",
     "note":"<b>Interleave them.</b> Many SSM layers (purple) with an attention layer (blue) dropped in every few blocks. Every token flows through the whole stack in order."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>A = L/(r+1) = 32/8 = 4 attention</text></g></svg>",
     "note":"<b>The layer budget.</b> With 32 layers at a 7:1 ratio, just 4 are attention and 28 are SSM. The ratio r is the dial you tune."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='22' width='200' height='28' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='140' y='40' fill='#1a5c38'>28 × O(n)  (cheap)</text><rect x='280' y='22' width='200' height='28' rx='6' fill='#E4F2F7' stroke='#2A7B9B'/><text x='380' y='40' fill='#1F6280'>4 × O(n²) (recall)</text></g></svg>",
     "note":"<b>The cost split.</b> Most layers are linear-cost; only a few pay the quadratic price. Long context becomes affordable while recall survives."},
    {"viz":"<svg viewBox='0 0 520 70'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='110' y='20' width='300' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='40' fill='#1a5c38'>long context ✓  +  exact recall ✓</text></g></svg>",
     "note":"<b>The payoff.</b> Efficiency from the SSM layers, precise lookup from the few attention layers. This is why Jamba and Griffin ship — but only if the ratio isn't pushed too far."},
  ],
  quiz=[
    {"q":"1. Why do hybrids interleave attention with SSM/recurrent layers?","opts":["to make training slower","the recurrent layers give cheap long context (O(n)) and the few attention layers restore exact recall","to remove the state","to avoid using GPUs"],"ans":1,"fb":"Each layer type fixes the other's weakness: SSM layers are cheap over long context, attention layers give the exact whole-history lookup SSMs lack."},
    {"q":"2. A 24-layer hybrid uses a 5:1 ratio (5 SSM per 1 attention). How many attention layers?","opts":["5","4","6","12"],"ans":1,"fb":"A = L/(r+1) = 24/(5+1) = 24/6 = 4 attention layers, and 20 SSM layers."},
    {"q":"3. You cut a hybrid from 4 attention layers to 1 (out of 32). Perplexity barely moves but a buried-fact retrieval test collapses. Why?","opts":["the SSM layers broke","too few attention layers removes the exact whole-history lookup; the cheap layers can't recover it, and fluency hides the loss","the model got faster so it forgot","attention was never needed"],"ans":1,"fb":"Recall lives in the attention layers. Cutting them too far kills exact retrieval while generic metrics look fine — a silent failure only a recall test exposes."},
    {"q":"4. What does raising the attention:SSM ratio (more SSM per attention) do?","opts":["improves recall and cost together","makes the model cheaper with longer context but weaker at precise retrieval","has no effect","removes the KV cache entirely at no cost"],"ans":1,"fb":"More SSM layers per attention layer means cheaper, longer context and less memory, but weaker exact recall. The ratio is the central hybrid trade-off."},
  ],
  fin={"em":"🧩","h3":"Day 3 complete — you can design a hybrid!",
       "p":"You now know why frontier models like Jamba and Griffin interleave a few attention layers among many cheap SSM/recurrent layers: linear-cost long context from the recurrent layers, exact recall from the sprinkled attention layers. You can work out a layer budget (A = L/(r+1)) and you know that too few attention layers silently kills recall. The attention:SSM ratio is the dial. Next: <b>Day 4</b> — the deep duality that ties all of this together."},
))

# ============================================================
# Day 4 — Recurrence and Attention Duality
# ============================================================
L("day-04-duality.html", dict(
  qid="m17c-d04-duality", title="Module 17c · Day 4 — Recurrence and Attention Duality", nav_title="Spiral · M17c Day 4",
  eyebrow="Module 17c · Systems · Day 4", h1="Recurrence and Attention Duality",
  lead="Here is the idea that ties the whole module together. A selective SSM and a linear-attention layer are not just similar — they are two <em>views of the same computation</em>. One view unrolls in parallel (great for training on a GPU); the other rolls forward one step at a time (great for cheap generation). Understanding this duality tells you exactly when to reach for an SSM and, just as important, when it will lose.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain the parallel-vs-recurrent duality (same math, two forms), read a \"when SSMs lose\" table (copying, exact retrieval, in-context lookup), compare the two forms on a worked example, and avoid picking an SSM for a recall-heavy task.",
  prev_href="day-03-hybrids.html", prev_label="Hybrids (Jamba, Griffin)", next_href="review.html", next_label="Module 17c Review",
  sections=[
    {"title":"One computation, two shapes","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (SSM recurrence), Day 2 (linear attention as a running sum), and Day 3 (hybrids). Today we unify them.</div></div>
     <h4>The surprising equivalence</h4>
     <p>Day 1's selective SSM and Day 2's linear attention look like different models. They are the same thing written two ways. Recent work (the "SSM ⇔ attention duality") shows the state update <code>h_t = A·h_{t-1} + B·x_t</code> can be unrolled into a big matrix multiply that looks exactly like attention with a structured mask.</p>
     <h4>The two forms</h4>
     <ul>
       <li><span class="term" data-tip="Compute all outputs at once as one big matrix multiply. Every step is available in parallel, so a GPU can chew through the whole sequence during training.">Parallel form</span> — expand the recurrence into a single matrix operation. All positions computed together. Fast on a GPU. Used for <b>training</b>.</li>
       <li><span class="term" data-tip="Compute outputs one token at a time by carrying a fixed-size state forward. Each new token costs O(1) — no growing cache. Used for generation.">Recurrent form</span> — carry the state forward, one token per step, O(1) each. Used for <b>inference / generation</b>.</li>
     </ul>
     <h4>Why this matters</h4>
     <p>You get transformer-style parallel training <em>and</em> RNN-style cheap generation from one model — no compromise. But the duality also makes the limit crisp: because both forms funnel the past through a <em>fixed-size state</em>, tasks that need exact token-level lookup are exactly where they lose to full attention.</p>''',
     "gotit":"Got the duality"},
    {"title":"A recipe you can cook two ways","body":
     '''<div class="relate">
       <div class="card"><span class="big">👨‍🍳</span><h5>Prep everything, cook at once</h5><p>A caterer prepping for a party chops all vegetables in parallel across many cutting boards, then combines. Fast when you have many hands (a GPU). That's the parallel form — all positions at once, for training.</p></div>
       <div class="card"><span class="big">🍳</span><h5>Cook to order, one plate</h5><p>At service, the same recipe is cooked one plate at a time as orders arrive — no need to redo the whole batch. That's the recurrent form — one token at a time, cheap, for generation.</p></div>
     </div>
     <p><strong>In one line:</strong> the same recipe (computation) is prepped in parallel for training and cooked one plate at a time for serving — one model, two schedules.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: both cooking styles use the same fresh ingredients, but the parallel form and recurrent form give <i>numerically identical</i> outputs — it is literally the same function, not just the same recipe. And unlike a kitchen, the "fixed-size state" means old ingredients get thrown out to make room, which is the recall limit.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Compare the two forms","body":'''<p>We will run the recurrent form step by step, then show the parallel form gives the identical result, then read where SSMs lose. <strong>Click all three.</strong></p>'''},
    {"title":"The duality, and where SSMs lose","body":
     '''<h4>Step 1 — in words</h4>
     <p>Unroll the recurrence. <code>h_t</code> is a weighted sum of all past inputs, where each input's weight is a product of A's along the way. Collect those weights into a matrix and you have attention with a structured (causal, decaying) mask — the parallel form.</p>
     <h4>Step 2 — the formula (unrolling into a matrix)</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       recurrent:  h<sub>t</sub> = A·h<sub>t−1</sub> + B·x<sub>t</sub><br>
       unroll:     h<sub>t</sub> = Σ<sub>i≤t</sub> A<sup>(t−i)</sup>·B·x<sub>i</sub>      <span class="dim"># each past input, decayed by A^(gap)</span><br>
       parallel:   y = ( M ⊙ (C B<sup>ᵀ</sup>) ) x,  M<sub>t,i</sub> = A<sup>(t−i)</sup> for i≤t else 0<br>
       <span class="dim"># M is a causal, decaying mask — this is attention in disguise</span>
     </div></div>
     <p>The parallel form is one masked matrix multiply (train fast on a GPU); the recurrent form carries <code>h</code> forward (generate cheaply). <b>Same numbers, two schedules.</b></p>
     <h4>Step 3 — worked example (both forms, same answer)</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       A=0.5, B=1, C=1, x=[4, 2], h_0=0<br><br>
       recurrent: h_1 = 4 → y_1 = 4;  h_2 = 0.5·4 + 2 = 4 → y_2 = <span class="hl">4</span><br>
       parallel:  y_2 = A<sup>(2−1)</sup>·x_1 + A<sup>0</sup>·x_2 = 0.5·4 + 1·2 = 2 + 2 = <span class="hl">4</span><br>
       <span class="dim"># identical — the decaying mask (0.5, 1) is exactly what the recurrence produced</span>
     </div></div>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — choosing an SSM for a recall-heavy task.</b> Because both forms compress the past into a fixed state with a <i>decaying</i> mask, they cannot cleanly reach back and copy an exact far-away token. The three classic losers:<br>
       • <b>Copying</b> — reproduce a long input string verbatim.<br>
       • <b>Exact retrieval</b> — pull a specific fact/number from far back.<br>
       • <b>In-context lookup</b> — associative recall ("earlier I said X=7; what is X?").<br>
       A pure SSM looks great on perplexity and general chat, then fails these — and you only notice if you test them. Attention, which keeps every token, handles all three.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — training parallelism vs inference recurrence.</b> The duality is a gift: the parallel form gives GPU-friendly training and the recurrent form gives O(1)-per-token, constant-memory generation from the <i>same</i> weights. The cost is baked into the shared limit — a fixed decaying state means the tasks in the table above are where you must add attention (a hybrid, Day 3) rather than rely on the SSM alone.</div></div>
     <h4>When SSMs lose — the table</h4>
     <div class="callout c-info"><span class="ic">📊</span><div style="font-family:var(--mono);font-size:.8rem;line-height:1.7">
       task                     | SSM alone | attention<br>
       ─────────────────────────┼───────────┼──────────<br>
       long-doc summary (gist)  |   strong  |  strong<br>
       streaming / cheap decode |   strong  |  weak (KV grows)<br>
       verbatim copying         |  <span class="bad">weak</span>    |  strong<br>
       exact retrieval @ pos k  |  <span class="bad">weak</span>    |  strong<br>
       in-context lookup/recall |  <span class="bad">weak</span>    |  strong
     </div></div>''',
     "gotit":"Got the duality & the table"},
    {"title":"The duality, assembled","body":'''<p>Here it is built up: the recurrent form, the unroll, the parallel matrix, the identical result, and the loss table. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Prove the duality","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m17c_duality.py</code>. Implement one SSM both ways — a recurrent loop and a single masked matrix multiply from the unrolled decay mask <code>M[t,i] = A^(t-i)</code>. Assert the two outputs match. Then run copying / retrieval / recall probes on the SSM vs attention and tabulate where the SSM fails.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 17c Day 4 artifact.
Create experiments/foundations/m17c_duality.py: implement one linear SSM two ways — (1) a recurrent loop h_t = A h_{t-1} + B x_t, y_t = C h_t, and (2) a parallel masked matmul using the decay mask M[t,i] = A^(t-i) for i<=t else 0. Assert both give the same y (the duality). Then build small copying, exact-retrieval, and associative-recall probes and compare the SSM against softmax attention, tabulating where the SSM loses.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m17c/day-04-log.md</code>: did the parallel and recurrent forms match to numerical precision, and which recall probe did the pure SSM fail hardest?</div></div>''',
     "gotit":"Done — on to the review"},
  ],
  demos={
    "rec":{"btn":"① recurrent form","html":'''<span class="prompt">&gt;&gt;&gt;</span> A,B,C = 0.5,1,1; x=[4,2]; h=0
t=1: h = 0.5*0 + 4 = 4   y1 = 4
t=2: h = 0.5*4 + 2 = 4   <span class="hl">y2 = 4</span>
<span class="dim"># one token at a time, carry h forward — O(1) each</span>''',
      "take":"<b>①  Roll it forward.</b> The recurrent form carries the state h one step at a time. Cheap per token — ideal for generation."},
    "par":{"btn":"② parallel form","html":'''<span class="prompt">&gt;&gt;&gt;</span> # unroll into a decaying mask M[t,i] = A^(t-i)
y2 = A^1 * x1 + A^0 * x2 = 0.5*4 + 1*2
<span class="hl">y2 = 2 + 2 = 4</span>   <span class="ok"># identical to the recurrent y2</span>
<span class="dim"># all positions at once = one matmul — ideal for GPU training</span>''',
      "take":"<b>②  Same math, all at once.</b> Unrolling gives a masked matrix multiply — attention in disguise. Same answer (4), but computed in parallel for fast training."},
    "lose":{"btn":"③ where SSMs lose","html":'''<span class="prompt">&gt;&gt;&gt;</span> probe(ssm)
verbatim copy long string:  <span class="bad">FAIL</span>
retrieve fact @ position k:  <span class="bad">FAIL</span>
in-context lookup (X=7?):    <span class="bad">FAIL</span>
long-doc gist / streaming:   <span class="ok">PASS</span>''',
      "take":"<b>③  The fixed-state limit.</b> Copying, exact retrieval, and in-context lookup need the exact old token — a decaying fixed state can't keep it. Add attention for these."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='120' y='16' width='280' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='260' y='36' fill='#5E5191'>recurrent: h_t = A·h_(t-1) + B·x_t</text></g></svg>",
     "note":"<b>Start with the recurrence.</b> Carry the state forward one token at a time — the cheap generation form from Day 1."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='36' fill='#9A7208'>unroll: h_t = Σ A^(t−i)·B·x_i</text></g></svg>",
     "note":"<b>Unroll it.</b> Each past input contributes, faded by A raised to the gap. A weighted sum over the whole past — starting to look like attention."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='100' y='16' width='320' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='36' fill='#1F6280'>parallel: y = (M ⊙ CBᵀ)·x, M = decay mask</text></g></svg>",
     "note":"<b>The parallel form.</b> One masked matrix multiply, all positions at once — attention with a causal, decaying mask. This is what a GPU trains fast."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='12' text-anchor='middle'><rect x='150' y='16' width='220' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='36' fill='#1a5c38'>y2 = 4  (both forms agree)</text></g></svg>",
     "note":"<b>Identical result.</b> Recurrent and parallel give the same numbers — one function, two schedules. Train in parallel, generate recurrently."},
    {"viz":"<svg viewBox='0 0 520 78'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='30' y='20' width='150' height='24' rx='5' fill='#FDE8E8' stroke='#C93B3B'/><text x='105' y='36' fill='#C93B3B'>copy · retrieve · lookup</text><text x='250' y='36' fill='#6B645E'>→ need</text><rect x='330' y='20' width='150' height='24' rx='5' fill='#E4F2F7' stroke='#2A7B9B'/><text x='405' y='36' fill='#1F6280'>attention (keep all)</text></g></svg>",
     "note":"<b>Where SSMs lose.</b> Verbatim copying, exact retrieval, and in-context lookup need every exact token. A fixed decaying state can't — so add attention (a hybrid). That's the whole module in one line."},
  ],
  quiz=[
    {"q":"1. What is the recurrence–attention duality?","opts":["they are unrelated","a selective SSM and linear attention are the same computation written two ways: a parallel (matmul) form and a recurrent form","attention is always faster","SSMs can't be trained"],"ans":1,"fb":"Unrolling the recurrence gives a masked matrix multiply — attention with a decaying causal mask. Same function, two schedules."},
    {"q":"2. Which form is used for training, and which for generation?","opts":["recurrent for both","parallel (all positions at once, GPU-friendly) for training; recurrent (one token, O(1)) for generation","parallel for both","neither is used"],"ans":1,"fb":"The parallel form parallelizes across positions for fast GPU training; the recurrent form carries a fixed state for cheap, constant-memory generation."},
    {"q":"3. A=0.5, B=1, C=1, x=[4, 2], h_0=0. Using the parallel unroll y_2 = A^1·x_1 + A^0·x_2, what is y_2?","opts":["6","4","2","8"],"ans":1,"fb":"y_2 = 0.5·4 + 1·2 = 2 + 2 = 4 — matching the recurrent form (h_2 = 0.5·4 + 2 = 4, y_2 = 4). The duality holds."},
    {"q":"4. Which task is a pure SSM most likely to fail, pushing you to add an attention layer?","opts":["summarizing the gist of a long document","reproducing an exact 20-token string from far back (verbatim copying / exact retrieval)","cheap streaming generation","running in constant memory"],"ans":1,"fb":"Copying, exact retrieval, and in-context lookup need the exact old token; a fixed decaying state can't keep it. That's precisely where you add attention."},
  ],
  fin={"em":"🔗","h3":"Day 4 complete — you see the whole picture!",
       "p":"You now understand the duality: a selective SSM and linear attention are one computation with two forms — parallel for training, recurrent for cheap generation. You can read the \"when SSMs lose\" table (copying, exact retrieval, in-context lookup) and you know to add attention (a hybrid) for recall-heavy tasks rather than trust a fixed state. Next: the <b>module review</b> — lock it all in."},
))

print("M17c: wrote day 4")

# ============================================================
# Review gate
# ============================================================
R("review.html", dict(
  qid="m17c-review", title="Module 17c · Review — State-Space Models & Attention Alternatives", nav_title="Spiral · M17c Review",
  eyebrow="Module 17c · Systems · Review",
  h1="Module 17c Review — SSMs & Attention Alternatives",
  lead="You built the linear-time family from the ground up: the selective-scan recurrence, linear attention, hybrids, and the duality that unites them. This gate makes sure it stuck. Tick each self-check you can explain out loud, then clear the mixed quiz.",
  goal="<b>🎯 Passing bar:</b> tick all four self-checks and score 4/5 or better on the mixed quiz. If you fall short, the verdict box tells you exactly which day to revisit.",
  prev_href="day-04-duality.html", prev_label="Day 4 · Recurrence and Attention Duality",
  checks=[
    ["Day 1 · Selective Scan (S6 / Mamba)",
     "I can write the recurrence <b><code>h_t = A·h_(t-1) + B·x_t</code>, <code>y_t = C·h_t</code></b>, explain that \"selective\" means input-dependent A,B,C, say why it's O(n) not O(n²), and name the fixed-state recall failure."],
    ["Day 2 · Linear Attention, RWKV, RetNet",
     "I can explain how dropping softmax (with feature maps) plus the <b>associativity trick</b> turns attention into a running-sum state, why RWKV/RetNet are attention/RNN hybrids, and the softmax quality gap."],
    ["Day 3 · Hybrids (Jamba, Griffin)",
     "I can say why interleaving a few attention layers among many SSM layers gives long context AND recall, compute a layer budget (<b>A = L/(r+1)</b>), and explain why too few attention layers silently kills recall."],
    ["Day 4 · Recurrence and Attention Duality",
     "I can explain the <b>parallel-vs-recurrent duality</b> (same math, two forms), read the \"when SSMs lose\" table (copying, exact retrieval, in-context lookup), and know to add attention for recall-heavy tasks."],
  ],
  quiz=[
    {"q":"1. Why does a state-space model cost O(n) while attention costs O(n²)?","opts":["SSMs use fewer parameters","the SSM updates one small fixed-size state per token instead of re-reading all past tokens each step","attention has no state","SSMs skip most tokens"],"ans":1,"fb":"One update per token over a fixed-size state gives O(n) time and O(1) memory; attention re-reads every past key each step, giving O(n²)."},
    {"q":"2. Run h_t = A·h_(t-1) + B·x_t with A=0.5, B=1, h_0=0 on x=[4, 2, 1]. What is h at t=3?","opts":["4","3","6","2"],"ans":1,"fb":"t=1: h=4. t=2: h=0.5·4+2=4. t=3: h=0.5·4+1=2+1=3. The answer is 3."},
    {"q":"3. What makes linear attention (and RWKV/RetNet) able to run as a recurrence?","opts":["a bigger softmax","dropping softmax for feature maps φ, so associativity lets you keep a running-sum state S_t = S_(t-1) + φ(k_t)v_tᵀ","adding more heads","removing the values"],"ans":1,"fb":"Without softmax coupling every query to every key, you can regroup the products into a fixed-size running-sum state and read it with φ(q) — an RNN-style recurrence."},
    {"q":"4. A 40-layer hybrid uses a 4:1 ratio (4 SSM per 1 attention). How many attention layers, and what's the risk of pushing that ratio much higher?","opts":["8 layers; no risk","8 layers; too few attention layers would silently break exact recall while perplexity looks fine","10 layers; it gets slower","4 layers; recall improves"],"ans":1,"fb":"A = 40/(4+1) = 8 attention layers. Raising the ratio too far removes the exact whole-history lookup, collapsing retrieval while generic metrics stay fine."},
    {"q":"5. You need a model to reproduce an exact 30-token ID string from 6,000 tokens back. Best choice?","opts":["a pure SSM — its fixed state is ideal for exact copying","include attention (a hybrid or full attention) — verbatim copying/exact retrieval is exactly where a fixed decaying state loses","linear attention alone, it never forgets","any of these work equally"],"ans":1,"fb":"Copying and exact retrieval need every exact token; a fixed decaying state can't keep them. This is the classic \"when SSMs lose\" case — reach for attention."},
  ],
  verdict_pass="you have the whole linear-time picture — the selective scan, linear attention, hybrids, and the duality. You know both why SSMs are fast and precisely where they lose. Move on with confidence.",
  verdict_fail="find your weakest self-check and reopen that day. Most stumbles are on the duality (Day 4) or the layer budget (Day 3) — redo the worked examples by hand until the numbers fall out cleanly.",
  complete_label="Mark Module 17c complete ✓",
  score_target=4,
  fin={"em":"🌀","h3":"Module 17c complete — the attention alternatives are yours!",
       "p":"You can now derive the state-space recurrence, linearize attention, design a hybrid layer budget, and reason about the parallel-recurrent duality — including the exact tasks where a fixed state loses to attention. This is live, frontier-lab knowledge (Mamba, Jamba, Griffin, RWKV, RetNet). Head back to the map for what's next."},
))

print("M17c: wrote review")
