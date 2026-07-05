#!/usr/bin/env python3
"""Builder for M16b — Serving Systems in Practice.
Authored from scratch. Builds on M16a (prefill/decode, KV cache). 4 lessons + review.
Run: python3 sessions/_build_m16b.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lesson_gen import write_lesson, write_review

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week-m16b")
os.makedirs(OUT, exist_ok=True)
def L(name, spec): write_lesson(os.path.join(OUT, name), spec)

# ---------------- Day 1 — Deploying a Real Engine ----------------
L("day-01-engines.html", dict(
  qid="m16b-d01-engines", title="Module 16b · Day 1 — Deploying a Real Engine", nav_title="Spiral · M16b Day 1",
  eyebrow="Module 16b · Serve · Day 1", h1="Deploying a Real Engine",
  lead="You can call <code>model.generate()</code> in a notebook and get one answer. But serving thousands of users at once is a different job. That job belongs to an <b>inference engine</b> — a program like vLLM or SGLang whose whole purpose is to run one model for many people, fast and cheaply. Today you learn what the engine adds on top of a plain generate loop, how a request travels through it, and the first failure that bites every beginner: running out of memory because nobody managed the KV cache.",
  goal="<b>🎯 By the end, you'll be able to:</b> say what an inference engine adds over <code>model.generate()</code>, trace a request through the engine's lifecycle, estimate why an engine serves far more tokens per second than a naive loop, and explain the OOM crash from an un-managed KV cache.",
  prev_href="../index.html", prev_label="Curriculum Map", next_href="day-02-paged-attention.html", next_label="PagedAttention & Continuous Batching",
  sections=[
    {"title":"From a generate loop to a serving engine","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> from M16a you know a model runs in two phases — <b>prefill</b> (read the whole prompt at once) and <b>decode</b> (make one token at a time) — and that each request keeps a <span class="term" data-tip="Key-Value cache: the stored attention keys and values for every token already processed, so the model does not recompute them for each new token.">KV cache</span> so it never recomputes past tokens. We build serving on top of that.</div></div>
     <h4>What a plain generate loop does</h4>
     <p>In a notebook you write <code>model.generate(prompt)</code>. It handles <em>one</em> request. It waits for that request to finish before it can start another. The expensive GPU sits idle between tokens. This is fine for you alone; it is a disaster for a service.</p>
     <h4>What an inference engine adds</h4>
     <p>An <span class="term" data-tip="A program built to run one model for many concurrent users efficiently — it schedules requests, batches them, and manages GPU memory. Examples: vLLM, SGLang, TensorRT-LLM.">inference engine</span> wraps the model and adds the missing serving machinery:</p>
     <ul>
       <li><b>A request queue and scheduler</b> — many users send prompts; the engine decides who runs on the GPU each step.</li>
       <li><b>Batching</b> — it runs many requests <em>together</em> in one GPU pass, so the GPU is busy, not idle (Day 2).</li>
       <li><b>KV-cache memory management</b> — it decides how much GPU memory each request's cache may use, and refuses new work before it crashes (today's failure).</li>
       <li><b>An API server</b> — it exposes an endpoint so apps can send requests over the network.</li>
     </ul>
     <p>You can think of it in one line: <b>the model computes; the engine serves.</b></p>''',
     "gotit":"Got what an engine adds"},
    {"title":"A restaurant kitchen","body":
     '''<div class="relate">
       <div class="card"><span class="big">🍳</span><h5>Cooking one dish (generate loop)</h5><p>At home you cook one dish start to finish, then start the next. The stove is often idle while you chop. That is <code>model.generate()</code> — one request at a time, the GPU waiting between steps.</p></div>
       <div class="card"><span class="big">👨‍🍳</span><h5>A restaurant line (the engine)</h5><p>A busy kitchen has a line cook who fires many orders on the same hot grill at once, a host who seats and queues guests, and a rule for when the kitchen is full. That coordination is the inference engine — the grill (GPU) is never idle.</p></div>
     </div>
     <p><strong>In one line:</strong> the engine is the kitchen's coordination — queue, batching, and a "we're full" rule — wrapped around the same stove (the model).</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a real kitchen makes each dish separately, but an inference engine runs many requests in a single fused GPU pass — the orders literally share one trip across the hot grill, not just one cook.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Trace one request","body":'''<p>Follow a single request as it enters the queue, gets batched during decode, and frees its memory on finish. <strong>Click all three.</strong></p>'''},
    {"title":"Why the engine serves so much more","body":
     '''<h4>Step 1 — what throughput means</h4>
     <p><span class="term" data-tip="How many output tokens the whole system produces per second, across all users. The headline number for serving cost.">Throughput</span> is total output tokens per second across every user. It is the number that sets your cost per token. A naive loop's throughput is low because the GPU processes one request's single token per step and idles the rest of the time.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       throughput ≈ batch_size × tokens_per_step_per_request × steps_per_second<br>
       <span class="dim"># a naive loop has batch_size = 1</span><br>
       <span class="dim"># an engine batches many requests, so batch_size ≫ 1 for ~the same step time</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       one decode step takes 20 ms on the GPU → <span class="hl">50 steps/second</span><br><br>
       naive loop: batch = 1 → 1 × 1 × 50 = <span class="hl">50 tokens/sec</span><br>
       engine:     batch = 32 → 32 × 1 × 50 = <span class="hl">1,600 tokens/sec</span><br>
       <span class="dim"># decode is memory-bound (M16a): loading the weights dominates,</span><br>
       <span class="dim"># so serving 32 requests together costs almost the same 20 ms as serving 1 → ~32× more throughput</span>
     </div></div>
     <p>The magic is from M16a: decode is <em>memory-bound</em>. Most of each step is spent reading the model weights from GPU memory, not doing math. If you read the weights once and reuse them for 32 requests, you get 32× the tokens for roughly the same time. That is why batching wins.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — un-managed KV cache OOM.</b> Every active request holds a KV cache that grows with its length. A naive loop that just accepts requests and reserves the worst-case cache size for each will, once enough long requests arrive, ask for more GPU memory than exists and <b>crash with out-of-memory</b> mid-request — dropping every in-flight user, not just the newest one. There is no warning; the queue simply looked fine until one allocation tipped it over. A real engine tracks free KV memory and <em>refuses or queues</em> new requests before that happens.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — managed engine vs raw simplicity.</b> A managed engine (vLLM/SGLang) gives you high throughput, safe memory limits, and batching — but it is a big, complex dependency with its own tuning knobs and version quirks. A raw <code>model.generate()</code> loop is trivial to read and debug, but serves one user slowly and crashes under load. For a demo, raw is fine; for production, the engine's complexity pays for itself many times over.</div></div>''',
     "gotit":"Got why the engine wins"},
    {"title":"The engine, assembled","body":'''<p>Here is the serving path built up: model, then queue, then batched decode, then memory management, then the API. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Serve a model with a real engine","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m16b_engine.py</code>. Start a small model two ways: (1) a plain <code>generate</code> loop over 32 prompts one at a time, and (2) the same 32 prompts sent concurrently to a vLLM server. Time both and print tokens/second for each. Confirm the engine is many times faster.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 16b Day 1 artifact.
Create experiments/foundations/m16b_engine.py: serve a small model (e.g. TinyLlama) two ways — a naive sequential model.generate() loop over 32 prompts, and the same 32 prompts sent concurrently to a vLLM engine. Measure and print total tokens/second for each, and the speedup. Add a comment explaining why decode being memory-bound makes batching give near-linear throughput gains.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m16b/day-01-log.md</code>: what speedup did the engine give over the naive loop, and did you hit any memory limit?</div></div>''',
     "gotit":"Done — on to Day 2"},
  ],
  demos={
    "enqueue":{"btn":"① request enters the queue","html":'''<span class="prompt">&gt;&gt;&gt;</span> POST /generate  {"prompt": "Write a haiku about the sea"}
<span class="dim"># the engine's API server receives it</span>
<span class="hl">request #941 → added to the waiting queue</span>
<span class="dim"># the scheduler will admit it once there is free KV memory</span>''',
      "take":"<b>①  The engine queues, it does not just run.</b> Unlike a bare generate call, the request waits its turn and is only admitted when memory allows."},
    "batch":{"btn":"② batched with others in decode","html":'''<span class="prompt">&gt;&gt;&gt;</span> decode step 118
<span class="hl">running batch: [#938, #939, #941, ... 29 more]</span>
<span class="dim"># all 32 requests generate their next token in ONE GPU pass</span>
<span class="dim"># the weights are read once and shared across the batch</span>''',
      "take":"<b>②  Many requests, one GPU pass.</b> Because decode is memory-bound, adding your request to the batch costs almost nothing — the weights were already loaded."},
    "finish":{"btn":"③ finishes and frees memory","html":'''<span class="prompt">&gt;&gt;&gt;</span> request #941 emitted &lt;eos&gt;
<span class="ok">done — 17 output tokens returned to the caller</span>
<span class="hl">freed its KV cache → memory returned to the pool</span>
<span class="dim"># a queued request can now be admitted in its place</span>''',
      "take":"<b>③  Finish frees the cache.</b> When a request ends, its KV memory goes back to the pool so a waiting request can start. This recycling is what keeps the GPU full."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='170' y='16' width='180' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='260' y='35' fill='#1F6280'>the model (computes)</text></g></svg>",
     "note":"<b>Start with the model.</b> It knows how to do prefill and decode for one sequence. On its own it has no idea how to serve many users."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='30' y='16' width='150' height='30' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='105' y='35' fill='#5E5191'>request queue</text><line x1='180' y1='31' x2='330' y2='31' stroke='#6B645E' stroke-width='2' marker-end='url(#a1)'/><rect x='330' y='16' width='160' height='30' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='410' y='35' fill='#1F6280'>scheduler</text><defs><marker id='a1' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0,0 L6,3 L0,6 Z' fill='#6B645E'/></marker></defs></g></svg>",
     "note":"<b>Add a queue and a scheduler.</b> Users' prompts wait in line. The scheduler decides which requests run on the GPU each step."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='14' width='70' height='16' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='40' y='32' width='70' height='16' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='40' y='50' width='70' height='12' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><text x='150' y='40' fill='#6B645E'>→</text><rect x='190' y='22' width='150' height='26' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='265' y='39' fill='#9A7208'>one fused GPU pass</text></g></svg>",
     "note":"<b>Batch them.</b> Many requests generate their next token together in a single GPU pass. The weights are read once and shared — the throughput win."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='90' y='18' width='340' height='34' rx='6' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><text x='260' y='33' fill='#C93B3B'>KV-cache memory pool (finite!)</text><text x='260' y='47' fill='#C93B3B'>track free space · refuse/queue before OOM</text></g></svg>",
     "note":"<b>Manage the KV memory.</b> The cache pool is finite. The engine tracks free space and refuses or queues new requests before it runs out — the fix for the OOM crash."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='140' y='16' width='240' height='30' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='35' fill='#1a5c38'>API server → apps send requests</text></g></svg>",
     "note":"<b>Expose an API.</b> An HTTP endpoint lets real apps send prompts over the network. Model + queue + batching + memory + API = a serving engine."},
  ],
  quiz=[
    {"q":"1. What does an inference engine add that a plain <code>model.generate()</code> loop lacks?","opts":["a bigger model","a request queue, batching, and KV-cache memory management for many concurrent users","more training data","a new tokenizer"],"ans":1,"fb":"The model computes; the engine serves — it schedules, batches, and manages memory so one model can serve many users fast and safely."},
    {"q":"2. One decode step takes 20 ms. A naive loop (batch 1) gives 50 tokens/sec. An engine batching 40 requests gives about:","opts":["50 tokens/sec","2,000 tokens/sec","20 tokens/sec","400 tokens/sec"],"ans":1,"fb":"40 requests × 50 steps/sec = 2,000 tokens/sec. Decode is memory-bound, so batching 40 costs about the same 20 ms per step as batching 1."},
    {"q":"3. Why does batching give near-linear throughput gains during decode?","opts":["because the math gets faster","because decode is memory-bound — the weights are read once and reused across the whole batch","because the prompt gets shorter","because the GPU overclocks"],"ans":1,"fb":"Decode's time is dominated by loading the model weights from memory. Reading them once and sharing across many requests is the free lunch batching exploits."},
    {"q":"4. A naive server reserves worst-case KV memory per request and keeps accepting them. What happens under load?","opts":["it slows down gracefully","it crashes with out-of-memory once allocations exceed GPU memory, dropping in-flight requests","it uses the CPU instead","nothing, memory is infinite"],"ans":1,"fb":"Un-managed KV growth leads to a silent OOM crash. A real engine tracks free cache memory and refuses or queues new work before it tips over."},
  ],
  fin={"em":"🍳","h3":"Day 1 complete — you can serve a model!",
       "p":"You now know the difference between a generate loop and a serving engine: the engine adds a queue, a scheduler, batching, KV-memory management, and an API. Batching wins because decode is memory-bound, and the classic beginner crash is an un-managed KV cache running out of memory. Next: <b>Day 2</b> — the two ideas that make the engine's memory and batching actually work: PagedAttention and continuous batching."},
))
# ---------------- Day 2 — PagedAttention & Continuous Batching ----------------
L("day-02-paged-attention.html", dict(
  qid="m16b-d02-paged", title="Module 16b · Day 2 — PagedAttention & Continuous Batching", nav_title="Spiral · M16b Day 2",
  eyebrow="Module 16b · Serve · Day 2", h1="PagedAttention & Continuous Batching",
  lead="Yesterday you saw the engine must manage KV-cache memory or it crashes. Today you learn <em>how</em> the best engines do it. Two ideas do the heavy lifting. <b>PagedAttention</b> stores the KV cache in small fixed blocks — like an operating system's virtual memory — so memory is never wasted on unused space. <b>Continuous batching</b> lets a new request join the running batch the instant a slot frees, instead of waiting for a whole batch to finish. Together they are why a real engine serves 2–4× more than a naive one.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain KV-cache fragmentation and how paging fixes it, contrast continuous (in-flight) batching with static batching, compute the memory a paged cache saves, and name the fragmentation failure and the complexity trade-off.",
  prev_href="day-01-engines.html", prev_label="Deploying a Real Engine", next_href="day-03-prefix-caching.html", next_label="Prefix (Radix) Caching",
  sections=[
    {"title":"The problem: wasted and fragmented cache","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 1 (the engine manages a finite KV-cache pool) and the KV cache from M16a. Today: the data structure that makes that pool efficient.</div></div>
     <h4>Why naive caches waste memory</h4>
     <p>A simple engine does not know how long a reply will be, so it reserves one big <em>contiguous</em> block per request — sized for the longest possible answer. Most replies are short, so most of that reserved block sits empty. This is <span class="term" data-tip="Reserving memory for the worst case but using far less, so the unused space inside each reservation is wasted and cannot be given to other requests.">internal fragmentation</span> — memory you paid for but never used.</p>
     <h4>And why contiguous blocks fragment</h4>
     <p>When requests of different sizes start and finish, the free memory left between them comes in odd-sized gaps. A new request needs one <em>contiguous</em> chunk, but no single gap is big enough even though the total free memory is plenty. That is <span class="term" data-tip="Enough total free memory exists, but it is split into scattered gaps, so no single contiguous block is large enough for a new request.">external fragmentation</span>. The pool looks half-empty yet refuses new work.</p>
     <h4>The PagedAttention idea</h4>
     <p><span class="term" data-tip="Storing the KV cache in small fixed-size blocks (pages) that need not be contiguous, with a table mapping each request's tokens to its blocks — like an operating system's virtual memory.">PagedAttention</span> stores the KV cache in small fixed-size <b>blocks</b> (say 16 tokens each) that do <em>not</em> have to sit next to each other. A per-request table maps its tokens to whatever blocks are free. A request grows one block at a time as it generates, so it only ever holds what it actually uses.</p>''',
     "gotit":"Got the fragmentation problem"},
    {"title":"Like an operating system's memory","body":
     '''<div class="relate">
       <div class="card"><span class="big">📦</span><h5>One big reserved shelf (naive)</h5><p>Imagine reserving a whole long shelf for each guest in case they bring many boxes. Most bring two boxes, so most of every shelf is empty, and a guest with many boxes can't fit because the free shelves are all too short. That is contiguous KV allocation.</p></div>
       <div class="card"><span class="big">🗂️</span><h5>Small numbered cubbies (paging)</h5><p>Instead, give everyone small numbered cubbies as they need them, and keep an index card of which cubbies each guest uses. No space is wasted, and any free cubby anywhere can be handed out. That is PagedAttention — exactly how an operating system pages memory.</p></div>
     </div>
     <p><strong>In one line:</strong> stop reserving one long shelf per request; hand out small cubbies on demand and keep a map — no wasted space, no dead gaps.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: cubbies are just storage, but a KV block must be found by the attention math on every token. PagedAttention needs a custom attention kernel that reads keys and values through the block table — the storage trick only works because the compute was rewritten to follow it.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Count the wasted memory","body":'''<p>Watch a naive contiguous reservation, the same request under paging, and how continuous batching backfills a freed slot. <strong>Click all three.</strong></p>'''},
    {"title":"How much paging saves, and continuous batching","body":
     '''<h4>Step 1 — the waste in words</h4>
     <p>A naive engine reserves the max sequence length per request. The wasted memory is the reserved length minus the length actually used, times the KV size per token, times the number of requests.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       naive reserved per request  = max_len × kv_per_token<br>
       paged used per request      = ceil(actual_len / block) × block × kv_per_token<br>
       waste_naive ≈ (max_len − actual_len) × kv_per_token   <span class="dim"># big when replies are short</span><br>
       waste_paged ≈ (block − 1) × kv_per_token   at most   <span class="dim"># only the last, partly-filled block</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       max_len = 2048, actual reply = 100 tokens, block = 16, kv_per_token = 0.5 MB<br><br>
       naive reserves: 2048 × 0.5 = <span class="hl">1024 MB</span>, uses only 100 × 0.5 = 50 MB<br>
       → wasted per request = 1024 − 50 = <span class="hl">974 MB</span> (95% wasted!)<br><br>
       paged uses: ceil(100/16) = 7 blocks → 7 × 16 × 0.5 = <span class="hl">56 MB</span><br>
       → wasted = only 56 − 50 = 6 MB (the half-empty last block)<br>
       <span class="dim"># paging turns ~974 MB of waste into ~6 MB → far more requests fit at once</span>
     </div></div>
     <h4>Continuous batching</h4>
     <p>With <span class="term" data-tip="Old batching: pick a fixed set of requests, run them all together to the end, and only then start the next batch. Short replies wait idle for the longest reply in the batch.">static batching</span>, all requests in a batch must finish before the next batch starts — so a 10-token reply waits idle for a 500-token reply. With <span class="term" data-tip="Also called in-flight batching: as soon as one request in the batch finishes, a waiting request takes its slot on the very next step, so the batch stays full.">continuous batching</span> (also called in-flight batching), the moment one request emits its end token, a queued request fills its slot on the next step. The batch stays full, the GPU stays busy.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — fragmentation without paging.</b> Without paging, an engine can report "only 60% of KV memory in use" and yet <b>refuse new requests</b>, because the free 40% is scattered in gaps too small for any one contiguous reservation. Throughput quietly craters while a memory dashboard says there is room. Nothing errors — the engine just admits fewer requests than the hardware could hold. Paging removes the problem because any free block, anywhere, is usable.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — complexity vs 2–4× throughput.</b> Paging plus continuous batching needs a custom attention kernel, a block table per request, and a scheduler that can add and evict requests every step. That is real engineering complexity and a small per-token bookkeeping cost. In return you fit far more requests in the same memory and keep the batch full, which is the 2–4× throughput gain that makes engines like vLLM worth it.</div></div>''',
     "gotit":"Got paging and continuous batching"},
    {"title":"Paging, built up","body":'''<p>Here it is assembled: the wasteful contiguous block, the split into pages, the block table, the memory saved, and continuous batching backfilling a slot. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Measure the memory paging saves","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m16b_paging.py</code>. Simulate 100 requests with random reply lengths. Compute total KV memory under (1) naive max-length reservation and (2) block paging with block size 16. Print the two totals and how many more requests fit in a fixed pool under paging.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 16b Day 2 artifact.
Create experiments/foundations/m16b_paging.py: simulate 100 requests with reply lengths drawn from a skewed distribution (most short, a few long) and max_len 2048. Compute total KV memory used under (a) naive contiguous max-length reservation and (b) block paging with block=16. Print both totals, the % memory wasted for each, and how many requests fit in an 80 GB pool under each scheme. Add a short note on how continuous batching keeps the running batch full versus static batching.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m16b/day-02-log.md</code>: what percent of KV memory did the naive scheme waste, and how many more requests fit with paging?</div></div>''',
     "gotit":"Done — on to Day 3"},
  ],
  demos={
    "naive":{"btn":"① naive contiguous reservation","html":'''<span class="prompt">&gt;&gt;&gt;</span> reserve(max_len=2048)   <span class="dim"># in case the reply is long</span>
<span class="hl">[■■□□□□□□□□□□□□□□□□□□]</span>  used 100 / reserved 2048 tokens
<span class="bad"># 95% of this block is empty but locked to this request</span>''',
      "take":"<b>①  One big block, mostly empty.</b> The engine reserves the worst case, so short replies waste almost all their reserved memory (internal fragmentation)."},
    "paged":{"btn":"② paged into small blocks","html":'''<span class="prompt">&gt;&gt;&gt;</span> block_table[req] = [p3, p9, p1, p7, p2, p8, p4]
<span class="hl">7 blocks × 16 tokens = 112 slots for a 100-token reply</span>
<span class="dim"># blocks are scattered anywhere free — only the last is partly empty</span>''',
      "take":"<b>②  Small blocks on demand.</b> The request holds only ~7 blocks (112 slots) for 100 tokens. Waste shrinks from ~974 MB to ~6 MB — the half-empty last block."},
    "backfill":{"btn":"③ continuous batching backfills","html":'''<span class="prompt">&gt;&gt;&gt;</span> step 210: request #938 emitted &lt;eos&gt;, freed its blocks
<span class="ok">step 211: queued request #952 admitted into the freed slot</span>
<span class="dim"># the batch stayed full — no waiting for the whole batch to finish</span>''',
      "take":"<b>③  A freed slot is filled at once.</b> Continuous batching admits a waiting request the very next step, so the GPU never idles waiting for the slowest reply."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='60' y='20' width='400' height='24' rx='4' fill='#FDE8E8' stroke='#C93B3B' stroke-width='2'/><rect x='60' y='20' width='24' height='24' fill='#C93B3B'/><text x='270' y='36' fill='#C93B3B'>reserved 2048 · used 100 (rest wasted)</text></g></svg>",
     "note":"<b>Naive: one big contiguous block.</b> Reserved for the longest possible reply. The filled part (dark) is tiny; the rest is locked and wasted."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='40' y='22' width='40' height='20' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='95' y='22' width='40' height='20' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='150' y='22' width='40' height='20' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='205' y='22' width='40' height='20' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='260' y='22' width='40' height='20' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='315' y='22' width='40' height='20' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='370' y='22' width='40' height='20' rx='3' fill='#FCF3DC' stroke='#C99A12' stroke-dasharray='3,2'/><text x='390' y='36' fill='#9A7208'>½</text></g></svg>",
     "note":"<b>Paged: small fixed blocks (16 tokens each).</b> Seven blocks hold a 100-token reply; only the last (amber) is partly empty. Waste is one block at most."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='40' y='16' width='150' height='34' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='115' y='30' fill='#5E5191'>block table</text><text x='115' y='44' fill='#5E5191'>req → [p3,p9,p1,...]</text><text x='215' y='36' fill='#6B645E'>→</text><text x='330' y='30' fill='#6B645E'>blocks scattered</text><text x='330' y='44' fill='#6B645E'>anywhere free</text></g></svg>",
     "note":"<b>A block table maps tokens to blocks.</b> Blocks need not be contiguous — any free block anywhere is usable, so there are no dead gaps (no external fragmentation)."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='22' width='180' height='24' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='130' y='38' fill='#C93B3B'>naive waste ≈ 974 MB</text><rect x='300' y='22' width='180' height='24' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='390' y='38' fill='#1a5c38'>paged waste ≈ 6 MB</text></g></svg>",
     "note":"<b>The memory saved.</b> For a 100-token reply, waste drops from ~974 MB to ~6 MB. That freed memory holds many more concurrent requests."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='40' y='16' width='90' height='16' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><rect x='40' y='34' width='90' height='16' rx='3' fill='#FDE8E8' stroke='#C93B3B' stroke-dasharray='3,2'/><text x='150' y='38' fill='#6B645E'>→</text><rect x='185' y='34' width='90' height='16' rx='3' fill='#E8F5EE' stroke='#2D8B55'/><text x='230' y='46' fill='#1a5c38'>backfilled</text><text x='400' y='34' fill='#6B645E'>batch stays full</text></g></svg>",
     "note":"<b>Continuous batching.</b> When a request finishes (red), a queued one takes its slot on the next step (green). The batch never drains — the GPU stays busy. Paging + continuous batching = 2–4× throughput."},
  ],
  quiz=[
    {"q":"1. What problem does PagedAttention mainly solve?","opts":["slow tokenization","KV-cache memory waste and fragmentation from reserving big contiguous blocks","weak attention scores","training instability"],"ans":1,"fb":"Paging stores the KV cache in small fixed blocks so no big block is reserved and wasted, and any free block anywhere is usable — killing internal and external fragmentation."},
    {"q":"2. max_len=2048, reply=100 tokens, block=16, kv_per_token=0.5 MB. Paged memory used is about:","opts":["1024 MB","56 MB","50 MB","6 MB"],"ans":1,"fb":"ceil(100/16)=7 blocks → 7×16×0.5 = 56 MB. The naive scheme would reserve 2048×0.5 = 1024 MB for the same reply."},
    {"q":"3. How does continuous (in-flight) batching differ from static batching?","opts":["it uses a bigger batch always","a finished request's slot is filled by a queued request on the next step, instead of waiting for the whole batch to end","it disables the KV cache","it runs prefill only"],"ans":1,"fb":"Static batching makes short replies wait for the longest one. Continuous batching backfills freed slots immediately, keeping the batch full and the GPU busy."},
    {"q":"4. Your engine says 60% of KV memory is used but it refuses new requests. Without paging, this is:","opts":["a hardware fault","external fragmentation — free memory is scattered in gaps too small for one contiguous reservation","the model being too large","a tokenizer bug"],"ans":1,"fb":"Contiguous allocation leaves odd-sized gaps; no single gap fits a new request even though total free memory is ample. Paging fixes it — any free block is usable."},
  ],
  fin={"em":"🗂️","h3":"Day 2 complete — you understand the memory engine!",
       "p":"You now know the two ideas behind a real serving engine: PagedAttention stores the KV cache in small non-contiguous blocks (like OS virtual memory) to kill fragmentation, and continuous batching backfills freed slots so the batch never drains. Together they turn ~95% wasted cache into near-zero and deliver 2–4× throughput. Next: <b>Day 3</b> — reusing the KV of a shared prompt prefix across requests with radix caching."},
))
# ---------------- Day 3 — Prefix (Radix) Caching ----------------
L("day-03-prefix-caching.html", dict(
  qid="m16b-d03-prefix", title="Module 16b · Day 3 — Prefix (Radix) Caching", nav_title="Spiral · M16b Day 3",
  eyebrow="Module 16b · Serve · Day 3", h1="Prefix (Radix) Caching",
  lead="Many requests start with the <em>same</em> text. A chatbot sends the same long system prompt every time. A few-shot task repeats the same examples. Computing the KV cache for that shared start again and again is pure waste. <b>Prefix caching</b> computes the shared prefix's KV once and reuses it for every request that begins the same way. A <b>radix tree</b> is the clever index that finds the longest shared prefix instantly. Today you learn the biggest cheap win in modern serving.",
  goal="<b>🎯 By the end, you'll be able to:</b> explain why a shared prompt prefix can reuse its KV cache, describe how a radix tree indexes prefixes, compute the tokens (and time) saved, and name the cache-thrash failure and the memory-vs-recompute trade-off.",
  prev_href="day-02-paged-attention.html", prev_label="PagedAttention & Continuous Batching", next_href="day-04-goodput-routing.html", next_label="TTFT/ITL, Goodput & Routing",
  sections=[
    {"title":"Stop recomputing the same start","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Day 2 (KV cache stored in blocks) and prefill from M16a (the whole prompt's KV is built at the start). Today: reuse that prefill work across requests.</div></div>
     <h4>The observation</h4>
     <p>The KV cache for a token depends only on that token and the tokens <em>before</em> it. So if two requests share the same first 500 tokens (the same system prompt), the KV cache for those 500 tokens is <em>identical</em>. Recomputing it for the second request is wasted work.</p>
     <h4>Prefix caching</h4>
     <p><span class="term" data-tip="Reusing the already-computed KV cache of a prompt prefix that a new request shares, so the model only does prefill for the new, differing part of the prompt.">Prefix caching</span> keeps the KV blocks of common prefixes around. When a new request arrives, the engine finds the longest cached prefix it matches, reuses those blocks, and only runs prefill on the remaining new tokens. The shared start is free.</p>
     <h4>The radix tree index</h4>
     <p>To find the longest shared prefix fast, engines like SGLang use a <span class="term" data-tip="A tree that stores strings by shared prefixes: common beginnings are one shared path, and branches split where requests differ. Looking up the longest cached prefix is a single walk down the tree.">radix tree</span>. Each path from the root spells out a cached token sequence; requests that share a beginning share the same path and split off where they differ. Matching a new request is one walk down the tree — instant, even with thousands of cached prefixes.</p>''',
     "gotit":"Got prefix reuse"},
    {"title":"A shared recipe base","body":
     '''<div class="relate">
       <div class="card"><span class="big">🍲</span><h5>Prep the base once</h5><p>A kitchen makes fifty different soups that all start from the same simmered stock. You do not boil fresh stock for every bowl — you make one big pot of stock (the shared prefix) and ladle from it. Only the toppings differ per bowl.</p></div>
       <div class="card"><span class="big">🌳</span><h5>An index of shared starts</h5><p>A recipe binder organized by shared steps: all tomato soups share the first pages, then branch. To make a new soup you follow the shared pages, then only read the new ending. That branching binder is the radix tree.</p></div>
     </div>
     <p><strong>In one line:</strong> compute the shared start of the prompt once, reuse it for everyone who begins the same way, and index those shared starts in a tree so matching is instant.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: stock, once ladled, is gone, but a cached KV prefix can be reused by any number of requests at the same time — it is read-only shared state, not consumed. It only disappears when the engine evicts it to make room.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Count the tokens saved","body":'''<p>Watch a first request build the prefix, a second request reuse it, and the radix tree grow a branch. <strong>Click all three.</strong></p>'''},
    {"title":"How much a shared prefix saves","body":
     '''<h4>Step 1 — the saving in words</h4>
     <p>Without prefix caching, every request pays prefill for its whole prompt. With it, a request pays prefill only for the tokens <em>after</em> the shared prefix. The saved work is the shared prefix length, for every request that reuses it.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       prefill_no_cache = prompt_len<br>
       prefill_with_cache = prompt_len − shared_prefix_len   <span class="dim"># only the new part</span><br>
       tokens_saved_per_request = shared_prefix_len<br>
       total_saved ≈ shared_prefix_len × (num_requests − 1)   <span class="dim"># first request builds it; the rest reuse</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       system prompt (shared) = 500 tokens, user question = 20 tokens → prompt_len = 520<br>
       1000 requests all share the 500-token system prompt<br><br>
       no cache: 1000 × 520 = <span class="hl">520,000</span> prefill tokens<br>
       with cache: 520 (first) + 999 × 20 = 520 + 19,980 = <span class="hl">20,500</span> prefill tokens<br>
       saved = 520,000 − 20,500 = <span class="hl">499,500</span> tokens (~96% of prefill work gone)<br>
       <span class="dim"># prefill is compute-heavy, so this also slashes time-to-first-token for reused prompts</span>
     </div></div>
     <p>This is why prefix caching is the biggest cheap win in serving: system prompts and few-shot examples are long and repeated on almost every request, so the shared part dominates.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — cache thrash under diverse prefixes.</b> The cache is finite. If every request has a <em>different</em> long prefix (many distinct system prompts, or user text placed <em>before</em> the shared part), each new prefix evicts an old one that never gets reused. You pay the full cost of caching — memory and bookkeeping — and get almost no hits. The hit rate quietly falls near zero and throughput drops <em>below</em> having no prefix cache at all. Nothing errors; the cache just churns. Fix: put the shared, static text <em>first</em> in the prompt so prefixes actually match.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — memory held vs recompute saved.</b> Cached prefix blocks occupy KV memory that could otherwise hold active requests. Keep many prefixes and you save huge recompute but have less room for live decoding; keep few and you free memory but lose hits. Engines tune this with an eviction policy (usually least-recently-used): hold the prefixes that pay off, drop the cold ones. The right balance depends on how repetitive your traffic is.</div></div>''',
     "gotit":"Got the prefix saving"},
    {"title":"Prefix caching, built up","body":'''<p>Here it is assembled: the shared prefix, the first request building it, a second request matching and reusing it, the radix tree, and the tokens saved. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Measure prefix-cache savings","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m16b_prefix.py</code>. Send many requests that share one long system prompt to an engine with prefix caching on, then off. Measure total prefill tokens and time-to-first-token for each. Then send requests with <em>distinct</em> prefixes and show the cache hit rate collapse (thrash).</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 16b Day 3 artifact.
Create experiments/foundations/m16b_prefix.py: with an engine that supports prefix/radix caching (e.g. SGLang or vLLM), send 1000 requests sharing a 500-token system prompt with prefix caching ON vs OFF; report total prefill tokens, cache hit rate, and mean time-to-first-token. Then rerun with a UNIQUE long prefix per request and show the hit rate collapse and throughput dropping (cache thrash). Add a note on why putting shared static text first maximizes hits.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m16b/day-03-log.md</code>: what cache hit rate and TTFT drop did the shared prefix give, and what happened with unique prefixes?</div></div>''',
     "gotit":"Done — on to Day 4"},
  ],
  demos={
    "first":{"btn":"① first request builds the prefix","html":'''<span class="prompt">&gt;&gt;&gt;</span> req A = [SYSTEM 500 tokens] + [question: "reset password?"]
<span class="hl">prefill all 520 tokens → cache the 500-token prefix blocks</span>
<span class="dim"># the shared start is now stored for reuse</span>''',
      "take":"<b>①  Pay once.</b> The first request computes and caches the KV of the 500-token system prompt. Full prefill this one time only."},
    "reuse":{"btn":"② second request reuses it","html":'''<span class="prompt">&gt;&gt;&gt;</span> req B = [SAME SYSTEM 500 tokens] + [question: "change email?"]
<span class="ok">radix match: longest cached prefix = 500 tokens → reuse blocks</span>
<span class="hl">prefill only the new 20 tokens</span>   <span class="dim"># 500 tokens of work skipped</span>''',
      "take":"<b>②  Reuse for free.</b> Request B shares the 500-token start, so the engine reuses those cached blocks and only prefills its 20 new tokens."},
    "tree":{"btn":"③ the radix tree grows a branch","html":'''<span class="prompt">&gt;&gt;&gt;</span> tree:
root ─ [SYSTEM 500 tokens] ─┬─ "reset password?"   (req A)
<span class="hl">                            └─ "change email?"      (req B)</span>
<span class="dim"># shared path once; requests branch where they differ</span>''',
      "take":"<b>③  One shared path, then branches.</b> The radix tree stores the common prefix as a single path and splits where requests differ, so matching is one fast walk down."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='60' y='20' width='300' height='24' rx='4' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='210' y='36' fill='#1F6280'>shared system prompt (500 tokens)</text><rect x='370' y='20' width='90' height='24' rx='4' fill='#FCF3DC' stroke='#C99A12'/><text x='415' y='36' fill='#9A7208'>question</text></g></svg>",
     "note":"<b>The prompt = shared prefix + unique tail.</b> A long system prompt (blue) that repeats on every request, then a short differing question (amber)."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='60' y='20' width='300' height='24' rx='4' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='210' y='36' fill='#5E5191'>req A: prefill all 520 → cache prefix</text></g></svg>",
     "note":"<b>First request pays.</b> It runs full prefill on all 520 tokens and stores the 500-token prefix's KV blocks for later reuse."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='40' y='22' width='210' height='22' rx='4' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='145' y='37' fill='#1a5c38'>reuse cached 500-token prefix</text><rect x='270' y='22' width='120' height='22' rx='4' fill='#FCF3DC' stroke='#C99A12'/><text x='330' y='37' fill='#9A7208'>prefill 20 new</text></g></svg>",
     "note":"<b>Second request reuses.</b> Radix match finds the 500-token prefix already cached, so only the 20 new tokens are prefilled — 500 tokens of work skipped."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='9' text-anchor='middle'><rect x='170' y='10' width='180' height='20' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='24' fill='#1F6280'>SYSTEM 500 (shared path)</text><line x1='260' y1='30' x2='150' y2='50' stroke='#6B645E'/><line x1='260' y1='30' x2='370' y2='50' stroke='#6B645E'/><rect x='90' y='50' width='120' height='18' rx='3' fill='#FCF3DC' stroke='#C99A12'/><text x='150' y='63' fill='#9A7208'>reset password?</text><rect x='310' y='50' width='120' height='18' rx='3' fill='#FCF3DC' stroke='#C99A12'/><text x='370' y='63' fill='#9A7208'>change email?</text></g></svg>",
     "note":"<b>The radix tree.</b> One shared path for the common prefix, branching where requests differ. Finding the longest cached match is a single walk down."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='20' width='190' height='24' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='125' y='36' fill='#C93B3B'>no cache: 520,000 tokens</text><rect x='300' y='20' width='190' height='24' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='395' y='36' fill='#1a5c38'>with cache: 20,500 tokens</text></g></svg>",
     "note":"<b>The tokens saved.</b> Across 1000 requests sharing a 500-token prompt, prefill work drops ~96%. Long, repeated prefixes make this the biggest cheap win in serving."},
  ],
  quiz=[
    {"q":"1. Why can a shared prompt prefix reuse its KV cache across requests?","opts":["because the model is random","a token's KV depends only on it and the tokens before it, so an identical prefix has identical KV","because caching disables attention","because prefixes are always short"],"ans":1,"fb":"KV for a position depends only on that token and everything earlier. Two requests with the same first 500 tokens have the same KV for those 500 tokens — so it can be computed once and reused."},
    {"q":"2. 1000 requests share a 500-token system prompt; each adds a 20-token question. With prefix caching, total prefill tokens ≈","opts":["520,000","20,500","500,000","1,000"],"ans":1,"fb":"520 for the first (builds the prefix) + 999 × 20 for the rest = 520 + 19,980 = 20,500. Without caching it would be 1000 × 520 = 520,000."},
    {"q":"3. What is the radix tree's job in prefix caching?","opts":["to compress the model weights","to index cached prefixes so the longest shared prefix for a new request is found in one walk down the tree","to replace the KV cache","to schedule GPUs"],"ans":1,"fb":"The radix tree stores cached sequences by shared prefix; matching a new request is a single tree walk, fast even with thousands of prefixes."},
    {"q":"4. Every request has a unique long prefix, so hits are rare and throughput drops below having no cache. This is:","opts":["ideal caching","cache thrash — distinct prefixes keep evicting each other, so you pay the caching cost with almost no reuse","a radix tree bug","impossible"],"ans":1,"fb":"With no shared prefixes the cache churns: each entry evicts another before reuse. The fix is to put shared, static text first so prefixes actually match."},
  ],
  fin={"em":"🌳","h3":"Day 3 complete — you unlocked the biggest cheap win!",
       "p":"You now know prefix (radix) caching: a token's KV depends only on earlier tokens, so a shared prompt prefix is computed once and reused by every request that starts the same way, indexed by a radix tree for instant matching. It can cut prefill work ~96% for repeated system prompts — unless prefixes are all distinct, which thrashes the cache. Next: <b>Day 4</b> — the latency metrics (TTFT, ITL), goodput, and how a fleet routes requests to hit its SLO."},
))
# ---------------- Day 4 — TTFT/ITL, Goodput & Routing ----------------
L("day-04-goodput-routing.html", dict(
  qid="m16b-d04-goodput", title="Module 16b · Day 4 — TTFT/ITL, Goodput & Routing", nav_title="Spiral · M16b Day 4",
  eyebrow="Module 16b · Serve · Day 4", h1="TTFT/ITL, Goodput & Routing",
  lead="Raw throughput is not the whole story. A user cares how long they <em>wait</em>: how fast the first word appears, and how smoothly the rest stream out. Those are <b>TTFT</b> and <b>ITL</b>. A service promises limits on them — a <b>Service Level Objective</b>. The number that actually matters is <b>goodput</b>: throughput that <em>meets the SLO</em>. Today you learn these metrics, why chasing throughput alone can be useless, how splitting prefill from decode helps, and how a fleet routes requests to stay fast.",
  goal="<b>🎯 By the end, you'll be able to:</b> define TTFT and ITL, explain goodput and why it can be far below raw throughput, describe disaggregated prefill/decode and fleet routing, work an SLO example, and name the useless-goodput failure and the batch-size trade-off.",
  prev_href="day-03-prefix-caching.html", prev_label="Prefix (Radix) Caching", next_href="review.html", next_label="Module 16b Review",
  sections=[
    {"title":"Two latencies, and the number that counts","body":
     '''<div class="callout c-info"><span class="ic">🧭</span><div><b>What this assumes:</b> Days 1–3 (batching, paging, prefix caching) and prefill/decode from M16a. Today: the metrics that decide whether serving is actually good, not just fast.</div></div>
     <h4>TTFT — time to first token</h4>
     <p><span class="term" data-tip="Time to first token: how long from sending a request until the very first output token appears. Dominated by prefill (reading the whole prompt).">TTFT</span> is how long you wait for the <em>first</em> word. It is set mostly by prefill — the model must read the whole prompt before it can produce anything. Long prompts and busy queues push TTFT up. This is the "loading..." feeling.</p>
     <h4>ITL — inter-token latency</h4>
     <p><span class="term" data-tip="Inter-token latency: the average gap between successive output tokens during streaming. Sets how smoothly and fast the text streams after the first token.">ITL</span> is the gap between each streamed token after the first. It sets how smoothly text flows. Big batches raise throughput but also raise ITL, because each decode step now does more work — every user's tokens come a little slower.</p>
     <h4>Goodput — throughput that meets the promise</h4>
     <p>A service sets an <span class="term" data-tip="Service Level Objective: a promised limit on a metric, e.g. TTFT under 1 second and ITL under 50 ms for 99% of requests.">SLO</span> — for example, "TTFT under 1 s and ITL under 50 ms." <span class="term" data-tip="Goodput: the rate of requests (or tokens) served while still meeting the SLO. Requests that finish but violate the SLO do not count.">Goodput</span> is the request rate you serve <em>while still meeting that SLO</em>. Requests that finish but break the promise do not count. Goodput, not raw throughput, is the honest measure of a serving system.</p>''',
     "gotit":"Got the three metrics"},
    {"title":"A coffee shop's promise","body":
     '''<div class="relate">
       <div class="card"><span class="big">☕</span><h5>First sip vs steady pour</h5><p>TTFT is how long before your coffee even starts pouring. ITL is how steadily it fills your cup after that. A shop can start your drink fast (low TTFT) but dribble it slowly (high ITL) — both matter to how you feel.</p></div>
       <div class="card"><span class="big">⏱️</span><h5>"Served in 3 minutes or it's free"</h5><p>The shop promises drinks in 3 minutes (the SLO). If it makes 200 drinks an hour but half arrive late, only the 100 on-time drinks count as good service. That on-time rate is goodput — not the raw 200.</p></div>
     </div>
     <p><strong>In one line:</strong> customers judge you by the first sip (TTFT), the steady pour (ITL), and whether you kept your time promise (SLO) — and only the promises you kept count as goodput.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div>Where the analogy breaks: a coffee shop serves each drink separately, but a serving engine makes one big batch move faster or slower for <em>everyone</em> at once — pushing throughput up can lift TTFT and ITL for every user in the batch, not just the newest order.</div></div>''',
     "gotit":"Got the picture"},
    {"title":"Work an SLO","body":'''<p>Watch throughput look great, the SLO check reveal violations, and goodput drop to the real number. <strong>Click all three.</strong></p>'''},
    {"title":"Goodput, disaggregation, and routing","body":
     '''<h4>Step 1 — goodput in words</h4>
     <p>Measure throughput, then keep only the requests that met every SLO limit (TTFT and ITL). That surviving rate is goodput. A system can post huge throughput while most of it violates the SLO — so that throughput is useless.</p>
     <h4>Step 2 — the formula</h4>
     <div class="callout c-info"><span class="ic">🧮</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.9">
       goodput = throughput × fraction_of_requests_meeting_SLO<br>
       <span class="dim"># SLO met  ⇔  TTFT ≤ TTFT_limit  AND  ITL ≤ ITL_limit</span><br>
       <span class="dim"># bigger batches ↑ throughput but ↑ TTFT and ITL → can ↓ the fraction that passes</span>
     </div></div>
     <h4>Step 3 — worked example</h4>
     <div class="callout c-info"><span class="ic">🔢</span><div style="font-family:var(--mono);font-size:.82rem;line-height:1.8">
       SLO: TTFT ≤ 1000 ms and ITL ≤ 50 ms<br>
       raw throughput = 100 requests/sec<br><br>
       config A (batch 16): TTFT 800 ms, ITL 40 ms → both pass → 100% meet SLO<br>
       &nbsp;&nbsp;goodput = 100 × 1.00 = <span class="hl">100 req/sec</span><br><br>
       config B (batch 64): raw throughput = 160 req/sec, but ITL = 70 ms &gt; 50 → fails ITL<br>
       &nbsp;&nbsp;only 30% of requests pass → goodput = 160 × 0.30 = <span class="hl">48 req/sec</span><br>
       <span class="dim"># B has higher throughput (160&gt;100) but LOWER goodput (48&lt;100) — the extra work is useless</span>
     </div></div>
     <h4>Disaggregated prefill and decode</h4>
     <p>Prefill is compute-heavy and sets TTFT; decode is memory-bound and sets ITL. Mixing them on one GPU means a big prefill stalls everyone's decode (their tokens stutter). <span class="term" data-tip="Running prefill and decode on separate GPU pools so a heavy prefill does not stall ongoing decode, letting each be tuned for its own latency target.">Disaggregated prefill/decode</span> runs them on separate GPU pools, so each can be tuned for its own metric — steadier ITL and controllable TTFT.</p>
     <h4>Fleet routing</h4>
     <p>With many engine replicas, a <span class="term" data-tip="A load balancer that sends each request to the replica most likely to meet the SLO — e.g. the one whose KV cache already holds the request's prefix, or the least-loaded one.">router</span> picks which replica handles each request — sending it to the least-loaded one, or to the replica that already has its prefix cached (Day 3). Good routing keeps every replica inside the SLO instead of overloading one.</p>
     <div class="callout c-warn"><span class="ic">⚠️</span><div><b>Silent failure — useless goodput (high throughput, SLO-violating).</b> The trap: you tune for the headline throughput number, push the batch size up, and the dashboard shows more tokens/second than ever. But TTFT and ITL crept over the SLO, so users see slow first words and stuttering text. The system is doing <em>more</em> work that <em>fails the promise</em> — goodput fell even as throughput rose. Nothing crashes; the throughput graph looks like a win. You only catch it by measuring TTFT/ITL percentiles against the SLO, not raw throughput.</div></div>
     <div class="callout c-warn"><span class="ic">⚖️</span><div><b>Trade-off — batch size (throughput vs latency).</b> A bigger batch reads the weights once for more requests, so throughput rises — but each step now does more work, so TTFT and ITL rise too. Small batches keep latency low but waste the GPU. The right batch size is the largest one that still keeps TTFT and ITL inside the SLO: it maximizes goodput, which is the number that pays the bills.</div></div>''',
     "gotit":"Got goodput and routing"},
    {"title":"Goodput, built up","body":'''<p>Here it is assembled: TTFT and ITL on the stream, the SLO limits, throughput vs goodput, disaggregated prefill/decode, and the router. <strong>Scroll slowly.</strong></p>'''},
    {"title":"Check yourself","body":'''<p>4 questions, instant feedback. <strong>Answer all four.</strong></p>'''},
    {"title":"Find the goodput-maximizing batch size","body":
     '''<p>Pick one:</p>
     <h4>Option A · write it yourself</h4>
     <p>Create <code>experiments/foundations/m16b_goodput.py</code>. Load a model in an engine and sweep the max batch size. For each, measure raw throughput plus TTFT and ITL percentiles. Apply an SLO (e.g. TTFT ≤ 1 s, ITL ≤ 50 ms) and plot throughput vs goodput. Find the batch size that maximizes goodput, not throughput.</p>
     <h4>Option B · let Claude build it</h4>
     <div class="prompt">
       <div class="prompt-h"><span class="prompt-l">triggers <b>frontier-experiment-lab</b></span><button class="copy" type="button" data-copy="#pp">📋 copy</button></div>
       <pre class="prompt-t" id="pp">Use /frontier-experiment-lab to build my Module 16b Day 4 artifact.
Create experiments/foundations/m16b_goodput.py: drive a served model at increasing load / max batch size; for each config record raw throughput and the p50/p99 of TTFT and ITL. Define an SLO (TTFT<=1000ms, ITL<=50ms), compute goodput = throughput * fraction meeting the SLO, and plot throughput vs goodput vs batch size. Print the batch size that maximizes goodput and show a config where higher throughput gives LOWER goodput. Add a note on disaggregated prefill/decode.</pre>
     </div>
     <div class="callout c-info"><span class="ic">📝</span><div>Write one line in <code>sessions/week-m16b/day-04-log.md</code>: which batch size maximized goodput, and did any config have higher throughput but lower goodput?</div></div>''',
     "gotit":"Done — on to the review"},
  ],
  demos={
    "through":{"btn":"① throughput looks great","html":'''<span class="prompt">&gt;&gt;&gt;</span> config B: batch = 64
<span class="hl">raw throughput = 160 requests/sec</span>
<span class="dim"># the headline number is the highest we've seen — looks like a win</span>''',
      "take":"<b>①  Throughput is the tempting number.</b> Bigger batch, more tokens/second. But this number alone hides whether users are actually served well."},
    "slo":{"btn":"② the SLO check","html":'''<span class="prompt">&gt;&gt;&gt;</span> SLO: TTFT ≤ 1000 ms, ITL ≤ 50 ms
<span class="prompt">&gt;&gt;&gt;</span> measured: TTFT = 900 ms (ok), ITL = <span class="bad">70 ms (FAIL)</span>
<span class="bad"># big batch made each decode step slower → ITL over the limit</span>''',
      "take":"<b>②  Check against the promise.</b> The batch is so large that inter-token latency blew past 50 ms. Most requests now violate the SLO, even though they finish."},
    "goodput":{"btn":"③ goodput is the real number","html":'''<span class="prompt">&gt;&gt;&gt;</span> fraction meeting SLO = 0.30
<span class="prompt">&gt;&gt;&gt;</span> goodput = 160 × 0.30 = <span class="hl">48 req/sec</span>
<span class="dim"># vs config A (batch 16): goodput = 100 × 1.00 = 100 req/sec</span>
<span class="ok"># the smaller batch serves MORE useful requests</span>''',
      "take":"<b>③  Goodput tells the truth.</b> Config B's higher throughput (160) gives lower goodput (48) than the smaller batch (100). The extra work failed the promise — useless."},
  },
  build=[
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='10' text-anchor='middle'><line x1='40' y1='40' x2='480' y2='40' stroke='#6B645E'/><circle cx='120' cy='40' r='5' fill='#2A7B9B'/><text x='120' y='26' fill='#1F6280'>1st token</text><text x='78' y='58' fill='#1F6280'>TTFT</text><circle cx='240' cy='40' r='4' fill='#7C6DAA'/><circle cx='300' cy='40' r='4' fill='#7C6DAA'/><circle cx='360' cy='40' r='4' fill='#7C6DAA'/><text x='300' y='58' fill='#5E5191'>ITL gaps</text></g></svg>",
     "note":"<b>Two latencies on the stream.</b> TTFT is the wait for the first token (blue); ITL is the gap between each token after that (purple). Users feel both."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='11' text-anchor='middle'><rect x='90' y='16' width='340' height='30' rx='6' fill='#FCF3DC' stroke='#C99A12' stroke-width='2'/><text x='260' y='35' fill='#9A7208'>SLO: TTFT ≤ 1000 ms  AND  ITL ≤ 50 ms</text></g></svg>",
     "note":"<b>The SLO is the promise.</b> A request only counts as well-served if it meets every limit. This is the bar throughput must clear to matter."},
    {"viz":"<svg viewBox='0 0 520 60'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='30' y='20' width='200' height='24' rx='6' fill='#FDE8E8' stroke='#C93B3B'/><text x='130' y='36' fill='#C93B3B'>throughput 160 (30% pass)</text><rect x='300' y='20' width='190' height='24' rx='6' fill='#E8F5EE' stroke='#2D8B55'/><text x='395' y='36' fill='#1a5c38'>goodput = 48</text></g></svg>",
     "note":"<b>Throughput vs goodput.</b> Raw throughput of 160 with only 30% meeting the SLO gives goodput 48 — lower than a smaller, all-passing batch. Goodput is what counts."},
    {"viz":"<svg viewBox='0 0 520 66'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='40' y='22' width='180' height='24' rx='6' fill='#E4F2F7' stroke='#2A7B9B' stroke-width='2'/><text x='130' y='38' fill='#1F6280'>prefill pool → TTFT</text><rect x='300' y='22' width='180' height='24' rx='6' fill='#EDE9F8' stroke='#7C6DAA' stroke-width='2'/><text x='390' y='38' fill='#5E5191'>decode pool → ITL</text></g></svg>",
     "note":"<b>Disaggregate prefill and decode.</b> Run them on separate GPU pools so a heavy prefill can't stall ongoing decode. Each pool is tuned for its own metric."},
    {"viz":"<svg viewBox='0 0 520 80'><g font-family='monospace' font-size='10' text-anchor='middle'><rect x='210' y='10' width='100' height='22' rx='6' fill='#E8F5EE' stroke='#2D8B55' stroke-width='2'/><text x='260' y='25' fill='#1a5c38'>router</text><line x1='260' y1='32' x2='120' y2='54' stroke='#6B645E'/><line x1='260' y1='32' x2='260' y2='54' stroke='#6B645E'/><line x1='260' y1='32' x2='400' y2='54' stroke='#6B645E'/><rect x='80' y='54' width='80' height='20' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='120' y='68' fill='#1F6280'>replica 1</text><rect x='220' y='54' width='80' height='20' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='260' y='68' fill='#1F6280'>replica 2</text><rect x='360' y='54' width='80' height='20' rx='4' fill='#E4F2F7' stroke='#2A7B9B'/><text x='400' y='68' fill='#1F6280'>replica 3</text></g></svg>",
     "note":"<b>Fleet routing.</b> A router sends each request to the best replica — least-loaded, or the one that already caches its prefix — so every replica stays inside the SLO. Metrics + disaggregation + routing = high goodput."},
  ],
  quiz=[
    {"q":"1. What does TTFT (time to first token) measure, and what dominates it?","opts":["the total reply length; decode","the wait until the first output token appears; prefill (reading the whole prompt)","the model size; training","the gap between tokens; the tokenizer"],"ans":1,"fb":"TTFT is the wait for the first token, set mostly by prefill — the model must read the whole prompt before producing anything. ITL is the between-token gap."},
    {"q":"2. Raw throughput = 160 req/sec but only 30% meet the SLO. Goodput is:","opts":["160 req/sec","48 req/sec","30 req/sec","0 req/sec"],"ans":1,"fb":"goodput = throughput × fraction meeting SLO = 160 × 0.30 = 48 req/sec. The 70% that violate the SLO don't count, so this config is worse than a smaller all-passing batch."},
    {"q":"3. Why disaggregate prefill and decode onto separate GPU pools?","opts":["to save disk space","so a compute-heavy prefill doesn't stall ongoing decode, letting TTFT and ITL each be tuned separately","to use a smaller model","to skip the KV cache"],"ans":1,"fb":"Prefill (compute-bound, sets TTFT) mixed with decode (memory-bound, sets ITL) makes big prefills stutter everyone's stream. Splitting them lets each pool hit its own metric."},
    {"q":"4. You raise the batch size; throughput climbs but users report slow, stuttering replies. This is:","opts":["ideal tuning","useless goodput — throughput rose but TTFT/ITL crossed the SLO, so the extra work fails the promise","a KV cache bug","a routing error only"],"ans":1,"fb":"Bigger batches lift throughput but also TTFT and ITL; once they cross the SLO, goodput falls even as throughput rises. Measure latency percentiles against the SLO, not raw throughput."},
  ],
  fin={"em":"⏱️","h3":"Day 4 complete — you can serve to an SLO!",
       "p":"You now know the metrics that decide real serving quality: TTFT (first-token wait, set by prefill), ITL (between-token gap, set by decode and batch size), and goodput — throughput that meets the SLO. Bigger batches trade latency for throughput, disaggregating prefill/decode protects both, and a router keeps a fleet inside its SLO. Chasing raw throughput is the classic trap: measure goodput. Next: the <b>review gate</b> — prove you can deploy, page, cache, and serve to an SLO."},
))
print("M16b: Days 1-4 written")

# ---------------- Review ----------------
write_review(os.path.join(OUT, "review.html"), dict(
  qid="m16b-review", title="Module 16b · Review Gate", nav_title="Spiral · M16b Review",
  eyebrow="Module 16b · The Gate — You Can Serve a Model", h1="Module 16b — Review Gate",
  lead="Checkpoint. The gate question: <b>can I take a model from a notebook generate loop to a real serving system — batched, paged, prefix-cached, and tuned to an SLO — and say where each trick quietly fails?</b> Tick the self-check, then answer five mixed questions.",
  goal="<b>🎯 The rule:</b> if a box feels shaky, re-run that day. Later systems modules (quantization, long-context decoding, and the design case studies) assume you can reason about serving cost and latency.",
  prev_href="day-04-goodput-routing.html", prev_label="Day 4 · TTFT/ITL, Goodput & Routing",
  checks=[
    ["Day 1","I can say what an inference engine adds over <code>model.generate()</code>, why batching wins because decode is memory-bound, and how an un-managed KV cache causes an OOM crash."],
    ["Day 2","I can explain KV-cache fragmentation, how PagedAttention (blocks + block table) fixes it, and how continuous batching keeps the batch full."],
    ["Day 3","I can explain prefix (radix) caching — a shared prefix's KV computed once and reused — and when distinct prefixes cause cache thrash."],
    ["Day 4","I can define TTFT and ITL, compute goodput (throughput meeting the SLO), and explain disaggregated prefill/decode and fleet routing."],
  ],
  quiz=[
    {"q":"1. One decode step is 20 ms (50 steps/sec). An engine batching 32 requests serves about how many tokens/sec, and why?","opts":["50 — batching doesn't help decode","1,600 — decode is memory-bound, so the weights are read once and shared across the batch","32 — one token total","20 — limited by step time"],"ans":1,"fb":"32 × 50 = 1,600 tokens/sec. Decode is memory-bound; reading the weights once and reusing them across 32 requests gives near-linear throughput (Day 1)."},
    {"q":"2. max_len=2048, reply=100 tokens, block=16, kv_per_token=0.5 MB. Paged KV memory used ≈","opts":["1024 MB","56 MB","50 MB","2048 MB"],"ans":1,"fb":"ceil(100/16)=7 blocks → 7×16×0.5 = 56 MB, versus 1024 MB a naive contiguous reservation would lock. Paging turns ~974 MB of waste into ~6 MB (Day 2)."},
    {"q":"3. 1000 requests share a 500-token system prompt, each adding 20 new tokens. With prefix caching, total prefill tokens ≈","opts":["520,000","20,500","500,000","1,000"],"ans":1,"fb":"520 (first request builds the prefix) + 999 × 20 = 20,500, versus 520,000 without caching — about 96% of prefill work saved (Day 3)."},
    {"q":"4. Config B has throughput 160 req/sec but only 30% meet the SLO. Its goodput is:","opts":["160 req/sec","48 req/sec","30 req/sec","112 req/sec"],"ans":1,"fb":"goodput = 160 × 0.30 = 48 req/sec — lower than a smaller all-passing batch, so B's higher throughput is useless (Day 4)."},
    {"q":"5. Which failure is shared across these serving tricks — a number looks like success while the system gets worse?","opts":["the model runs out of parameters","a metric that keeps rising while quality/usefulness falls: un-managed KV looks fine until OOM; fragmentation hides free memory; cache thrash churns silently; throughput climbs while goodput drops","the tokenizer changes","the GPU overheats"],"ans":1,"fb":"Every day had a silent failure: OOM from un-managed KV (D1), fragmentation refusing work with memory 'free' (D2), cache thrash with near-zero hits (D3), and useless goodput as throughput rises past the SLO (D4). The discipline is measuring the honest signal — free KV memory, hit rate, and TTFT/ITL against the SLO."},
  ],
  verdict_pass="you can take a model from a generate loop to a real serving stack — a batching engine, PagedAttention, prefix/radix caching, and SLO-driven tuning with disaggregation and routing — and name where each trick quietly fails and how to catch it. You're ready to reason about serving cost and latency in any systems design.",
  verdict_fail="note which day tripped you up and re-run just that lesson. The arc — <code>engine</code> → <code>paging + continuous batching</code> → <code>prefix caching</code> → <code>goodput + routing</code> — should feel familiar before you tackle quantization and long-context decoding.",
  complete_label="Mark Module 16b complete",
  fin={"em":"🏆","h3":"Module 16b — passed!",
       "p":"You can now serve a large model like a real system: an engine that queues, batches, and manages KV memory (Day 1); PagedAttention and continuous batching to kill fragmentation and keep the GPU full (Day 2); prefix/radix caching to reuse shared prompt work (Day 3); and TTFT/ITL/goodput with disaggregation and routing to hit an SLO (Day 4). This is the serving backbone the quantization, long-context, and design modules build on."},
  score_target=4,
))
print("M16b built: 4 lessons + review")




