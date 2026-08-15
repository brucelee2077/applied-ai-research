# m08 coverage reconciliation — READ BEFORE AUTHORING ANY DAY

Output of the pre-build spec pass (6 blind drafters + one reconciler, 2026-08-09).
The coverage blocks it produced are committed in `manifest.yaml`; this file is the
reasoning, and the constraints below are BINDING on every day's author.

**The five that will cost you a rebuild if ignored:**

1. `deferred` is a MAP (`topic: where`), never a list — `coverage_gate._curation_maps()`
   calls `.items()` on it.
2. **Vocabulary is fixed module-wide.** `parameter` (plain handle: knob) · gradient =
   **blame** (the established upstream word, 58 uses in m02-d05/m04-d02 — the drafts'
   invented "report cards"/"fix-list" have ZERO upstream uses, so keep them as ANALOGIES
   but never as the noun) · `head_dim` (gloss "m03 called this d_k" once, on day-02) ·
   `seq_len` · **token**, never "words" · **chip**, never "machine" · `n_heads` /
   `n_kv_heads` · **block**, not "floor" · **MFU** (one name) · **GiB** module-wide.
3. **Day-06 owns no mechanism.** It is a rehearsal day: the checking discipline plus the
   four wait-on-what tools. It had claimed 7 concepts owned by days 01/03; those are now
   back-pointers. Do not re-teach them.
4. **Analogy collisions — the KITCHEN belongs to day-04 only.** Day-04's spine maps
   kitchen stations onto training-loop stages; day-06 must keep its recipe-card spine and
   stay out of the kitchen. K/V are **note cards** (day-03's), not jars. The till/receipt
   is reserved for multiply-accumulate.
5. **One contradiction was merged, keep it merged:** day-01 excludes embeddings from the
   `N` used by the compute rule; day-02 counts every parameter for the file size. That is
   ONE topic on day-02 ("two counts for two jobs"), not two days disagreeing.

Prerequisite fixes already applied to the committed coverage: day-03 no longer claims
m09a "already teaches" arithmetic intensity (m09a is a LATER module — the tool lands
in-module on day-06); 20-tokens-per-parameter and "choosing the best split" point at
**day-05**, not m10a; day-02's activations forward-pointer goes to **day-04**, which owns
them, not day-03, which disowns them.

---

## Findings

**0 — BLOCKER, output shape: the requested `deferred` list shape crashes the build.** `coverage_gate._curation_maps()` does `(d.get('deferred') or {}).items()`; a list of `{topic, where}` raises `AttributeError: 'list' object has no attribute 'items'`. Same call site in `coverage_judge.py:893`. Verified both ways:

```
/Users/ruifengli/Desktop/applied-ai-research/sessions/_compiler/gates/coverage_gate.py:209
    for k, v in (d.get('deferred') or {}).items():
```
So I emit `deferred` as a **map** (`topic: where`) — the shape m07's shipped manifest uses and the only shape the gate accepts. `covers` and `out_of_scope` use the requested list shape (both verified parsing). Validated: 6/6 day keys match disk exactly, all 116 deferred paths resolve, `_spec_from_manifest_covers` + `_curation_maps` parse every day. I also removed three `deferred` keys with an inner `: ` — PyYAML resolved them the way I intended but the split point is ambiguous and a JS YAML lib may differ.

**1 — Duplication: 9 collisions, all assigned.** Biggest: **day-06 claimed 7 concepts that days 01/03 own** (2-per-multiply-accumulate, backward≈2×, C=6ND assembly, "what 6ND leaves out", the KV-cache recipe, decode-waits-on-fetching, the magnitude check). Rule applied: *day-06 is a rehearsal day and owns no mechanism* — it now owns only the checking discipline plus the four wait-on-what tools no earlier m08 day teaches. Its `covers` went 17 → 9 and every mechanism became a back-pointer. Others: MFU (d01 owns concept+name, d04 owns "read your own share off the log line"); width-vs-depth at fixed budget (d02 owns the arithmetic, d05 keeps only "the hardware picks"); sharding (d04 deferred it to m09c while d05 covered it — d05 owns planning depth, d04's limit now points at d05); activations (d02/d04/d05 → d04 owns); batching amortization (d03 deferred to m16a while d06 covered it → d03 now points at d06). One duplication was a **contradiction**: d01 c18 says exclude embeddings from the count, d02 c12 says add them — merged into a single d02 topic, "two counts for two jobs" (non-embedding N for the compute rule, every parameter for the file).

**2 — Prerequisite inversions: 5 real, 2 cleared.** Real: (a) day-03's sources assert "m09a/day-01-rooflines *already* teaches arithmetic intensity" — m09a is a **later** module; the tool now lands in-module on day-06 and day-03 keeps only the felt version its own spec scoped. (b) day-06 defers 20-tokens-per-parameter to m10a/day-02 and day-01 defers "choosing the best split" there too, but **day-05 covers it** — both re-pointed to day-05. (c) day-01's "prices a plan, cannot choose one" pointed at m10a; day-05 answers it four days later. (d) day-01's recomputation→8ND needs the saved-working-out mechanism that day-04 owns (constant stays on d01, d04 now names the 8). (e) day-02 c19 forward-points activations to "next lesson" (=d03), but d03 explicitly disowns them → re-pointed to d04. **Cleared:** day-01's width test may say "the model's width" — `d_model` is taught upstream in m03 day-01/02/04; and day-01's backward-pass callback is grounded (`m04-first-model-mlp/day-02-backward-pass/lesson.html` contains the literal string "Two jobs at each layer").

**3 — Vocabulary: 10 quantities under 2–4 names each.** Canonical set (grep-verified against shipped lessons): `parameter` + plain handle **knob** (m02 uses "weight" ×149, "knob" ×2; drafts add "learned number"/"stored number" — needs one bridge line "m02 called these weights"); gradient = **blame** (m02-d05 + m04-d02 use "blame" ×58; the drafts' "report cards" and "fix-list" have **zero** upstream uses — the analogies can stay, the noun cannot); **head_dim** (m03 uses `d_k` ×41 exclusively, m08 uses `head_dim` exclusively, nothing bridges them → gloss "m03 called this d_k" once on day-02); **seq_len** (`n_ctx` appears in 0 shipped lessons — quote it once as Kaplan's name); **token**, never "words" (day-05's "twenty words of reading per knob" and day-06's "words read" are wrong units); **chip**, never "machine" (day-04 throughout); **n_heads / n_kv_heads**, never "lookers"; **block**, not "floor"; **MFU** one name, not also "effective throughput"/"share of peak"; **GiB** module-wide (day-03's 4.3 GB and day-06's 4.0 GiB are the same bytes — let day-06's unit-slip topic use that pair as its illustration); and one sentence reconciling 2 bytes/parameter (d02, the file) with ~16 bytes/parameter (d05, training).

**4 — Analogy collisions: 7.** Worst is **KITCHEN**: day-04's entire spine maps kitchen stations → loop stages, then day-06 maps the same kitchen to door=bandwidth, cook=FLOPs, 32 pantry shelves=cache layers — adjacent days, same object, different system. Day-06 should drop the kitchen and keep its recipe-card spine. Also: **K/V** = note cards (d03) vs labelled jars (d06) → unify on the cards; **receipt** carries three jobs (exact-vs-napkin d01c3/d02c11, the assembled 2+4=6 bill d01c7, MAC running total d01c4/d06c3) → reserve till/receipt for MAC + the bill, move d02c11 to brick-count-vs-floor-plan (matches its blueprint spine); **van** = "affordable but won't fit" (d01) vs "hauling bytes for tiny math" (d06), both van+memory → d01 keeps doorway+sofa, drops the van; **four different scales** (d02 airport / d03 kitchen / d05 airline / d06 bathroom) → reserve the scale for "is this number believable" (d02c15 + d06c14 as one deliberate motif), d03 → stack height, d05 → suitcase only; **recipe card** = a corrected published rule (d05c5) vs the formula you keep (d06 spine) → d05 switches to two dated notebook pages. One is an *upgrade* not a cut: d01's tile rectangle and d05's carpet roll are the same geometry with area released vs held fixed — make it explicitly one motif.

**5 — Scoping concerns: 24 collected.** *Keep-with-plan:* backward≈2× (raised by 3 drafters — state + check + m04 callback, never re-derive); the twelve in the width test (quote the paper's sentence; the natural English phrase "rule of thumb" is banned by the plain-language rule — use "width test"); the 20:1 ratio (assert as measured); trim-after-average (must ship as the 3-4-5 worked example, never a stated rule). *Do not re-merge:* the O(n²)→O(n) slogan — day-03's single real correctness trap; both halves stay split and the slogan is never written bare. *Defer off m08:* MFU-vs-HFU (2 drafters raised it → m09a/day-06); decode intensity at general batch (day-03 must say "one user at a time" out loud); bandwidth-picks-the-shape (→ m09a/day-01). *Recorded unowned gaps:* gated-FFN mechanism, multi-head latent attention. *Module-wide action:* pin **one** chip speed — the legacy day-01 ships three (`1e15` ×7, `989 TFLOP/s` ×3, `3.96e14` ×1) and day-06 a fourth (`200 TFLOP/s`). Recommend peak 1.0e15 FLOP/s, bandwidth 3.3e12 B/s, MFU 0.4 → effective 4.0e14, ridge ≈ 300 FLOPs/byte, which reconciles all four; author to confirm against the datasheet. Also: day-01's "serving is a third of training" must be scoped to *arithmetic*, or it contradicts day-03's "serving waits on memory".

**6 — Load: every drafted day is over budget.** Shipped V9 covers/day: m03 7.4, m06 7.4, m07 6.0 (m02 seed 14.5); 7–10 concept units per day. Drafts: 21/20/16/13/15/17, avg **17** — every day exceeds the largest normal day (12). After reconciliation: 16/19/15/13/14/9, avg 14.3, achieved by merging failure+remedy pairs into single units (which also defuses the ≥3-consecutive-failure advisory in `concept_structure_gate.py:106` that three drafters independently predicted) and by moving duplicates to back-pointers. **day-01 (16) and day-02 (19) are still over budget.** Days cannot be added — quest-ids `w03-d01-arith … w03-d06-examrehearsal` are frozen in the shipped HTML. If the build needs more cuts, take them in this order: d02 "shape walk" (fold into the four-matrices topic), d02 "the small parts we skip" (fold into the proof beat's residual), d01 "the long way against the one line" (fold into the six-assembled beat).

```yaml
coverage:

  # ===========================================================================
  # DAY 01 — owner of THE COMPUTE RULE (C = 6ND) and everything that prices it
  # ===========================================================================
  day-01-transformer-arithmetic:
    covers:
      - topic: a flop is one step of arithmetic — one multiply or one add
        keywords: [flop, one multiply, one add, unit of work, count the arithmetic]
      - topic: the only two numbers you need — parameters and training tokens
        keywords: [parameter, knob, token, two numbers, model card]
      - topic: using one parameter costs two steps — multiply then add into a running total
        keywords: [multiply-accumulate, multiply then add, running total, two steps per parameter]
      - topic: the forward pass costs two steps per parameter per token
        keywords: [forward pass, prediction, two per pairing, per parameter per token, 2nd]
      - topic: the backward pass costs about four — two blames per parameter
        keywords: [backward pass, backpropagation, two jobs, blame, pass the blame back, 4nd]
      - topic: the six assembled — two forward plus four backward
        keywords: [six steps, add them up, training bill, c = 6nd, 2 + 4 = 6]
      - topic: six is per pairing, not per parameter and not per token — the bill is an area
        keywords: [per pairing, area not a side, rectangle of tiles, grows with both, double either one]
      - topic: the long way against the one line — itemised counting lands within about a tenth
        keywords: [count every operation, layer by layer, itemised, the long way, within about a tenth, napkin estimate]
      - topic: inference is the forward pass only, so serving arithmetic is about a third of training
        keywords: [inference, forward only, one third, serving a model, no backward pass]
      - topic: arithmetic divided by rate gives hours, and many chips divide the waiting
        keywords: [wall-clock time, steps per second, divide, hours, many chips, chip-hours, in parallel]
      - topic: failure — dividing by the speed printed on the box; remedy — shrink it to the share you reach
        keywords: [peak speed, advertised, sticker number, quoted with sparsity, too hopeful, mfu, model flops utilization, share of peak]
      - topic: failure — the rule misses tokens comparing themselves; remedy — the width test
        keywords: [attention term, grows with the text length, undercount, width test, twelve times as wide, safe to ignore]
      - topic: failure — rubbing out the working to save memory turns the six into about eight
        keywords: [activation recomputation, gradient checkpointing, one extra pass, about eight, which constant applies]
      - topic: limit — the rule assumes every parameter works on every token
        keywords: [dense model, every parameter for every token, active parameters, switched off, count only the active ones]
      - topic: two limits — it cannot choose a plan for you and it says nothing about memory
        keywords: [prices but does not choose, same cost different result, memory is a separate budget, affordable but impossible, two budgets]
      - topic: checking the size of the answer before you trust it
        keywords: [order of magnitude, sanity check, anchor runs, count the zeros, add the powers of ten, plain multiplication never complains]
    deferred:
      where the parameter count itself comes from — twelve times d_model squared per block: sessions/m08-transformer-math/day-02-qkv-matrices
      what query key and value do inside attention, and the multi-head split: sessions/m08-transformer-math/day-02-qkv-matrices
      the rounded headline size and which parameters each count includes: sessions/m08-transformer-math/day-02-qkv-matrices
      the exact size of the attention term the width test lets you ignore: sessions/m08-transformer-math/day-03-kv-cache
      the kv cache, and why writing one token at a time waits on memory not arithmetic: sessions/m08-transformer-math/day-03-kv-cache
      why the saved working-out exists and what it costs, which is what recomputation removes: sessions/m08-transformer-math/day-04-production-code
      how a run reads its own share of top speed off its log line: sessions/m08-transformer-math/day-04-production-code
      choosing the best split of a fixed budget between model size and tokens: sessions/m08-transformer-math/day-05-pretraining-logic
      splitting one model across chips so the memory bill divides: sessions/m08-transformer-math/day-05-pretraining-logic
      arithmetic intensity and the ridge point that decide which resource you are waiting on: sessions/m08-transformer-math/day-06-scaling-exercises-rehearsal
      labelling and cancelling units so a rate divided by a rate lands right: sessions/m08-transformer-math/day-06-scaling-exercises-rehearsal
      why serving is not simply two steps per parameter per token in practice: sessions/m16a-inference-economics/day-01-inference-mechanics
      cost per served token and how batching changes it: sessions/m16a-inference-economics/day-03-batching-economics
      why the share of peak you reach is low in the first place: sessions/m09a-hardware-physics/day-01-rooflines
      the training memory budget — weights plus optimizer state plus saved working: sessions/m09a-hardware-physics/day-05-memory-footprint
      the memory-for-compute trade behind rubbing out the working: sessions/m09a-hardware-physics/day-06-distributed-checkpointing
      why splitting a run across more chips is not perfectly proportional: sessions/m09c-sharding-parallelism/day-01-data-parallel-fsdp
      how error falls as a smooth power of compute: sessions/m10a-scaling-laws/day-01-kaplan-paradigm
      how the compute-optimal split is actually measured with equal-compute curves: sessions/m10a-scaling-laws/day-03-isoflops-methodology
      running out of good text to train on: sessions/m10a-scaling-laws/day-05-data-wall
      how a model switches most of its parameters off per token: sessions/m14a-mixture-of-experts/day-01-moe-fundamentals
      how fewer bytes per number changes throughput and memory: sessions/m17a-quantization/day-01-quantization-theory-int8
      rebuilding this derivation cold from a blank page: sessions/m08-transformer-math/day-06-scaling-exercises-rehearsal
    out_of_scope:
      - topic: the full per-operation flop table for every layer type
        reason: the day's whole value is that one line of multiplication replaces the table, and the shapes it depends on are day-02 material
      - topic: the arithmetic of the embedding and de-embedding layers including the vocabulary softmax
        reason: the source papers exclude embeddings from their own parameter count and the term is well under one percent, smaller than the rule's own ten percent wobble
      - topic: the arithmetic the optimizer itself performs
        reason: it happens once per update step rather than once per token, so against six times parameters times tokens it is a rounding error
      - topic: the arithmetic of layer normalisation, the activation bend, biases and residual adds
        reason: these are the sub-leading terms the source papers explicitly omit, and including them triples the vocabulary load for a change smaller than the stated error
      - topic: how fp8 or int8 change what counts as one step of arithmetic
        reason: lower precision changes how fast steps retire and how many bytes a number takes, not how many operations the model performs
      - topic: flash attention and its tiling
        reason: it changes how much data moves, not how many operations run, so it cannot change this day's number at all
      - topic: converting chip-hours into dollars
        reason: hours-to-dollars needs a rental price that changes every few months and would date the lesson; if added, it must be a price the learner types in, never a number baked into the prose
      - topic: comparing specific accelerators against each other
        reason: the method is a single division that works for any chip, and a comparison table dates quickly
      - topic: fine-tuning and adapter compute
        reason: it is the same rule with a smaller token count or fewer unfrozen parameters, and which parameters are frozen needs a module the reader has not reached

  # ===========================================================================
  # DAY 02 — owner of PARAMETER COUNTING, every shape, and every parameter name
  # ===========================================================================
  day-02-qkv-matrices:
    covers:
      - topic: d_model is the token's width and it sets every shape in the block
        keywords: [d_model, token width, one number, every shape follows, 768]
      - topic: a weight matrix holds rows times columns parameters
        keywords: [parameter, rows times columns, weight matrix, grid, count the squares]
      - topic: the four attention matrices are each d_model by d_model, so four times d_model squared together
        keywords: [w_q, w_k, w_v, w_o, four matrices, projection, 4 d_model squared]
      - topic: the shape walk through one block — the last shape matches the first
        keywords: [shape, station, same width out as in, stays the same width, stack]
      - topic: the n by n score grid is the one shape that grows with tokens, not with width
        keywords: [score grid, n by n, every token vs every token, one box per pair, seq_len, four times as many boxes]
      - topic: multi-head is a reshape, not a resize — the parameter count does not change
        keywords: [n_heads, head_dim, d_k, 12 heads of 64, same numbers regrouped, count does not change]
      - topic: failure — d_model does not divide by n_heads; remedy — check it first or pick head_dim first
        keywords: [not divisible, head_dim is not whole, assert, pick head_dim first, 64 or 128, check before you build]
      - topic: the feed-forward part is a wide-then-narrow pair — eight times d_model squared
        keywords: [feed-forward, ffn, up-projection, down-projection, four times wider, d_ff, 8 d_model squared]
      - topic: two of every three parameters in a block live in the feed-forward part
        keywords: [two thirds, most of the weights, ffn dominates, not attention]
      - topic: the napkin rule — about twelve times d_model squared per block, times n_layers for the stack
        keywords: [12 d_model squared, per block, n_layers, 7077888, multiply by the number of blocks]
      - topic: the exact count adds up every matrix, but only once the model exists
        keywords: [add up every matrix, numel, exact count, one at a time, only after you build it, ground truth]
      - topic: silent failure — the napkin rule forgets the word tables; remedy — add vocab times d_model at each end
        keywords: [embedding table, lm head, vocab times d_model, undercount, add the vocabulary terms, nothing errors]
      - topic: weight tying — one shared word table at both doors, paid once instead of twice
        keywords: [weight tying, tied embeddings, shared word table, paid once not twice, same matrix both ends]
      - topic: failure — assuming the four-times widening is a law; remedy — read the model's own stated inner width
        keywords: [convention not a law, intermediate_size, d_ff from the config, gated, three matrices, read the config]
      - topic: two counts for two jobs — the non-embedding count the compute rule needs, and every parameter for the file on disk
        keywords: [non-embedding, 174.6 billion, rounded name, which count did you use, bytes per parameter, 2 bytes, checkpoint size]
      - topic: the small parts we skip, and why the rule honestly says about
        keywords: [bias, layernorm scale and shift, about 4 times d_model, why we write approximately, too small to matter]
      - topic: the proof beat — the rule on gpt-2 small lands within a tenth of a percent
        keywords: [gpt-2 small, 84934656, 124318464, 124439808, within 0.1 percent, real checkpoint]
      - topic: trade-off — on a fixed budget width costs squared while depth costs once
        keywords: [wider vs deeper, double the width is four times, double the blocks is twice, same budget, design choice]
      - topic: limit — a parameter count says nothing about the numbers a model makes while running
        keywords: [activations, temporary numbers, grows with seq_len, not parameters, saved working-out]
    deferred:
      the compute rule this parameter count feeds, and about two steps per parameter per token forward: sessions/m08-transformer-math/day-01-transformer-arithmetic
      which of the projections get stored while a model writes text, and the cache-size formula: sessions/m08-transformer-math/day-03-kv-cache
      grouped-query and multi-query attention, which cut the stored key and value heads: sessions/m08-transformer-math/day-03-kv-cache
      what the saved working-out costs during training, and why it fills memory before the weights do: sessions/m08-transformer-math/day-04-production-code
      reading a model's config in code, and where the weights actually get updated: sessions/m08-transformer-math/day-04-production-code
      how a lab really picks the shape once budgets, data and hardware all press at once: sessions/m08-transformer-math/day-05-pretraining-logic
      whether a given matrix multiply waits on arithmetic or on memory traffic: sessions/m08-transformer-math/day-06-scaling-exercises-rehearsal
      training memory as weights plus gradients plus optimizer states plus activations: sessions/m09a-hardware-physics/day-05-memory-footprint
      the roofline and arithmetic intensity of a single matrix multiply: sessions/m09a-hardware-physics/day-01-rooflines
      how these same four matrices get split across chips: sessions/m09c-sharding-parallelism/day-02-tensor-parallel
      choosing parameters and tokens together for a fixed compute budget: sessions/m10a-scaling-laws/day-02-chinchilla-correction
      solving for d_model and n_layers to hit a parameter budget inside framework code: sessions/m12-addition-transformer/day-07-architecture-sizing
      mixture-of-experts, where total parameters stop equalling the parameters used per token: sessions/m14a-mixture-of-experts/day-01-moe-fundamentals
      fewer bytes per parameter, which changes the file size without changing the count: sessions/m17a-quantization/day-01-quantization-theory-int8
    out_of_scope:
      - topic: what a query-key match means, softmax, and dividing by the square root of head_dim
        reason: taught upstream in m03-attention day-02 and day-03; today re-meets query key and value only as shapes and counts
      - topic: what residual connections and layer normalisation do, and why pre-norm beats post-norm
        reason: taught upstream in m05a-text-transformer day-01, day-02 and day-04; today only counts layernorm's scale and shift so the word about is honest
      - topic: causal masking
        reason: it changes which boxes of the score grid are used, not the grid's shape and not one parameter
      - topic: flash attention and kernel fusion
        reason: they change how the score grid is computed in memory, not a single shape or a single parameter
      - topic: what the gating mechanism inside a gated feed-forward part actually does
        reason: today needs only the counting remedy of reading d_ff; the mechanism cannot be stated in one plain sentence and no day currently owns it, so this is a recorded curriculum gap rather than a settled call
      - topic: per-framework bias bookkeeping and fused qkv layouts
        reason: the word about plus the named residual already gives the honest gap; itemising framework variants adds bookkeeping, not intuition
      - topic: rope and relative-position mechanics
        reason: today needs only the one-line fact that some models add no position parameters at all, so that term can be zero
      - topic: encoder-decoder blocks and cross-attention
        reason: today sizes a decoder-only block, which is what every model in this curriculum is

  # ===========================================================================
  # DAY 03 — owner of THE KV CACHE and the inference memory budget
  # ===========================================================================
  day-03-kv-cache:
    covers:
      - topic: autoregressive decoding writes one token at a time and looks back at all of them
        keywords: [autoregressive, one token at a time, next-token prediction, decode step, look back]
      - topic: the naive baseline redoes the whole prefix every step — a staircase of work you already did
        keywords: [naive recompute, staircase, work you already did, triangle of wasted work, grows every step]
      - topic: keys and values are written once and never change, so reusing them is exact
        keywords: [causal mask, write-once, never change, exact optimization, same output but faster, no quality loss]
      - topic: the kv cache appends one card per step and then reads the whole pile
        keywords: [kv cache, stored keys and values, append one token, read the whole cache, per layer, no recompute]
      - topic: why queries are never cached
        keywords: [a query is used once, no q cache, one query per decode step, keys and values only]
      - topic: what one token costs — n_layers times n_kv_heads times head_dim, twice, times bytes per number
        keywords: [values per token, two cards, keys and values, n_layers, n_kv_heads, head_dim, 262144 values, fixed cost per token]
      - topic: the whole-request cache size, quoted with the shape it belongs to
        keywords: [cache size formula, seq_len, batch, bytes per number, 4.0 gib at 8192 tokens, two bytes per number]
      - topic: one memory pot — the weights sit still while the cache grows with length and with users
        keywords: [memory budget, weights plus kv cache, grows with seq_len, grows with batch, requests per chip, headroom]
      - topic: for one user at a time, decode waits on fetching rather than on arithmetic
        keywords: [memory-bandwidth-bound, bytes read per token, waiting on data, tokens-per-second ceiling, one user at a time]
      - topic: failure — sizing capacity from the weights alone; remedy — budget weights plus peak cache and cap the worst case
        keywords: [out of memory at long context, weights-only sizing, compute sits idle, peak cache not average, admission control, longest context times most users]
      - topic: remedy — grouped-query attention shares the stored cards so the pile shrinks by the sharing factor
        keywords: [grouped-query attention, multi-query attention, shared key and value heads, n_kv_heads, four times smaller cache]
      - topic: failure — counting one set of cards per query head when the heads share
        keywords: [n_heads versus n_kv_heads, over-estimating the cache, sanity check your factors, four to eight times too big]
      - topic: misconception — the cache does not save memory, it spends memory to skip repeated work
        keywords: [trade compute for memory, not a memory saving, the deal you are signing, memory grows and never shrinks]
      - topic: limit — the cache cannot remove the first pass over the prompt
        keywords: [prefill, first pass, fills the cache, removes repeats not first work, long prompts still cost]
      - topic: limit — the cache does nothing during training
        keywords: [inference-only, whole sequence at once, nothing to reuse, switched off while training]
    deferred:
      the parameter counting and the shapes every byte of this cache is built from: sessions/m08-transformer-math/day-02-qkv-matrices
      the compute rule that prices the arithmetic this cache is trading against: sessions/m08-transformer-math/day-01-transformer-arithmetic
      the saved working-out that fills memory during training, which is not this cache: sessions/m08-transformer-math/day-04-production-code
      arithmetic intensity and the ridge point that turn waiting-on-fetching into a number: sessions/m08-transformer-math/day-06-scaling-exercises-rehearsal
      how a bigger batch amortizes the weight read so the chip stops waiting: sessions/m08-transformer-math/day-06-scaling-exercises-rehearsal
      prefill and decode as two distinct performance regimes: sessions/m16a-inference-economics/day-01-inference-mechanics
      bytes-read-per-token arithmetic and tokens-per-second ceilings on real chips: sessions/m16a-inference-economics/day-02-memory-wall
      offloading the cache to host memory or disk: sessions/m16a-inference-economics/day-02-memory-wall
      batching economics — throughput against per-user latency, and cost per token: sessions/m16a-inference-economics/day-03-batching-economics
      paged attention and cache fragmentation: sessions/m16a-inference-economics/day-04-serving-llama3-tpus
      storing the cached keys and values in fewer bits: sessions/m16a-inference-economics/day-05-quantization-intro
      cost per token in dollars: sessions/m16a-inference-economics/day-06-inference-calculator-roofline
      attacking every factor in the cache formula in turn: sessions/m17b-long-context-decoding/day-01-decoding-innovations
      multi-head latent attention — recorded as an UNOWNED curriculum gap, nearest home below: sessions/m17b-long-context-decoding/day-01-decoding-innovations
      cache eviction and sliding-window attention: sessions/m17b-long-context-decoding/day-02-cache-eviction-snapkv
      grouped-query and multi-query attention in depth, including the quality trade-off: sessions/m17b-long-context-decoding/day-03-gqa-mqa
      speculative decoding: sessions/m17b-long-context-decoding/day-04-speculative-decoding
      ring attention and sequence parallelism: sessions/m17b-long-context-decoding/day-05-ring-attention
      splitting the kv cache across several chips: sessions/m09c-sharding-parallelism/day-02-tensor-parallel
      never writing the score matrix down at all: sessions/m15a-custom-kernels-pallas/day-04-flashattention-paradigm
    out_of_scope:
      - topic: re-deriving the attention formula itself
        reason: taught in m03-attention and revisited at the shape level on day-02; today works at the shape-and-size level only
      - topic: rotary position embeddings and how position is baked into a cached key
        reason: it changes no number in today's size arithmetic, and the rotation is applied at the token's own position before the card is filed, so the card is still write-once
      - topic: encoder-decoder cross-attention caching
        reason: the day's setting is a decoder-only model writing one token at a time, and a cross-attention cache is a different static object
      - topic: beam search and several candidate sequences sharing one cache
        reason: a sampling-strategy topic rather than a property of the cache; the day keeps one sequence so every number stays checkable by hand
      - topic: the in-memory layout of the cache tensors inside a kernel
        reason: kernel-engineering detail owned by m15a-custom-kernels-pallas; this day stops at counting bytes

  # ===========================================================================
  # DAY 04 — owner of THE PRODUCTION LOOP, activations, sync placement, saves
  # ===========================================================================
  day-04-production-code:
    covers:
      - topic: the production loop is the five-station loop with extra stations bolted on
        keywords: [training step, the extra stations, shard the batch, accumulate, all-reduce, trim, update, log, periodic save, nanotron, picotron, fixed order]
      - topic: the hidden loop behind one train command
        keywords: [one-line train command, hidden stages, read the code, no abstractions, you cannot price what you cannot see]
      - topic: the saved working-out is what fills the memory, which is what recomputation rubs out
        keywords: [activations, kept until the backward pass, grows with batch size, grows with seq_len, memory runs out before the weights do, why the batch has to be split]
      - topic: gradient accumulation builds the global batch out of small spoonfuls
        keywords: [micro-batch, accumulation steps, global batch, add up the blames, update once at the end, delay the wipe on purpose]
      - topic: the sharing step — many chips add up their blames and divide to agree on one average
        keywords: [all-reduce, each gets its own slice, add up and divide, everyone ends holding the same numbers, collective]
      - topic: where the sharing step has to sit — after the blames, before any weight changes
        keywords: [after working out the blames, before changing any weights, only on the last spoonful, copies drift apart, the order is the correctness]
      - topic: failure — the missing divide makes every update as many times too big as there are chips
        keywords: [sharing adds by default, forgot to divide, too big by that many times, secretly turning the step size up, nothing crashes, average not a total]
      - topic: the one-chip against many-chip check
        keywords: [run both for a few steps, compare the loss numbers line by line, they should match, cheapest test you have]
      - topic: trimming an over-large update after the average, not before
        keywords: [where the trim goes, trim after the average, the average of trimmed is not the trim of the average, 3 and 4 measure 5, every chip trims the same thing]
      - topic: reading your own share of top speed off the loop's log line
        keywords: [loss, step size, seconds per step, tokens per second, six flops per parameter per token, mfu, model flops utilization, about 20 percent]
      - topic: a save has to hold the whole training state, not just the weights
        keywords: [not just the weights, optimizer state, which step you were on, where you had reached in the data, random state, resume exactly, loss jumps after a bad resume]
      - topic: how often to save — weigh what a crash costs against what a save costs
        keywords: [saving stalls training, a crash throws away work, you lose about half the gap, pick the gap on purpose]
      - topic: limit — a full copy on every chip never shrinks the model
        keywords: [full copy per chip, more chips means more data per second, not less memory on any one chip, adding chips does not help, this is what sharding is for]
    deferred:
      the compute rule behind six flops per parameter per token, and the mfu idea itself: sessions/m08-transformer-math/day-01-transformer-arithmetic
      the parameter count and the shapes the loop is updating: sessions/m08-transformer-math/day-02-qkv-matrices
      cutting one model across chips so the memory bill divides, at planning depth: sessions/m08-transformer-math/day-05-pretraining-logic
      how the chips actually carry out the sharing step: sessions/m09a-hardware-physics/day-04-multi-device
      every chip must reach the sharing step the same number of times or the run hangs: sessions/m09a-hardware-physics/day-04-multi-device
      whether a stage waits on arithmetic or on memory traffic: sessions/m09a-hardware-physics/day-01-rooflines
      the full training memory bill of about sixteen bytes per parameter with an adam-style updater: sessions/m09a-hardware-physics/day-05-memory-footprint
      bytes per number, mixed precision, and the full-precision master copy: sessions/m09a-hardware-physics/day-05-memory-footprint
      throwing the saved working-out away and recomputing it instead: sessions/m09a-hardware-physics/day-06-distributed-checkpointing
      saving and loading a checkpoint that is itself split across chips: sessions/m09a-hardware-physics/day-06-distributed-checkpointing
      sharding the weights, gradients and optimizer state so no chip holds a whole copy: sessions/m09c-sharding-parallelism/day-01-data-parallel-fsdp
      how much time the sharing step costs on the wire, and overlapping it: sessions/m09c-sharding-parallelism/day-01-data-parallel-fsdp
      splitting one single layer across chips: sessions/m09c-sharding-parallelism/day-02-tensor-parallel
      splitting the stack of blocks into stages on different chips: sessions/m09c-sharding-parallelism/day-03-pipeline-parallel
      reading picotron piece by piece against its jax sharding equivalent: sessions/m09c-sharding-parallelism/day-06-pytorch-sync-consolidation
      why making the global batch bigger and bigger stops helping: sessions/m10a-scaling-laws/day-01-kaplan-paradigm
      choosing model size and tokens together for a fixed budget: sessions/m10a-scaling-laws/day-02-chinchilla-correction
    out_of_scope:
      - topic: deriving why the four original phases exist and what each one does
        reason: taught in m02 day-06 and run by the reader in m04 day-05; the day still shows the five phases inside its station rail so the reader recognises them
      - topic: what gradient clipping is, and the exploding-gradient failure it fixes
        reason: m04 day-02 already covers it; what only exists once there are many chips is WHERE the trim sits relative to the average, and that placement failure is covered
      - topic: learning-rate warmup and the shapes of decay schedules
        reason: m02 day-08 teaches warmup and decay; here the schedule appears only as the reason a save must store the step number
      - topic: how an adam-style updater computes its update
        reason: m02 day-07 owns the update rules; today needs only the consequence that the updater keeps extra numbers per parameter
      - topic: what zeroing the gradients does and the forgot-to-wipe beginner bug
        reason: m04 day-05 teaches it in full; today it returns only as the trick that gradient accumulation delays on purpose
      - topic: how the backward pass is built, the graph and the chain rule step by step
        reason: m02 day-05 and m04 day-02 own it; today needs only that it uses the saved working-out and costs about twice the forward pass
      - topic: a walkthrough of nanotron's configuration files, launcher and yaml
        reason: the day reviews the loop; a configuration tour teaches one library's spelling rather than what each stage costs
      - topic: tokenization, shuffling and packing sequences inside the data stage
        reason: the first station only needs that a chunk of tokens arrives and gets split across chips
      - topic: fault tolerance beyond saving
        reason: saving is the crash remedy this day teaches, and the surrounding infrastructure is a systems-engineering topic
      - topic: running a validation pass inside the loop, and early stopping
        reason: m02 day-09 owns the split and early stopping; every station production adds is identical whether or not an eval pass runs
      - topic: comparing updaters
        reason: choosing an updater is a different question from reading a loop

  # ===========================================================================
  # DAY 05 — owner of CHOOSING THE PLAN (size, tokens, shape) under three budgets
  # ===========================================================================
  day-05-pretraining-logic:
    covers:
      - topic: pretraining and the base model
        keywords: [pretraining, base model, next-token prediction, one expensive run, post-training comes later]
      - topic: the three budgets that squeeze every pretraining plan
        keywords: [compute budget, good-text budget, chip-space budget, binding constraint, best quality per lifetime cost]
      - topic: at fixed compute you trade model size against tokens — the same rectangle with its area held fixed
        keywords: [fixed compute, size and tokens multiply, same budget different plans, halve the model and double the reading, area held fixed]
      - topic: the compute-optimal split is about twenty tokens per parameter
        keywords: [compute-optimal, chinchilla ratio, twenty tokens per parameter, lowest error for a budget, 70 billion on 1.4 trillion]
      - topic: the earlier rule got it wrong, so the famous models were under-fed
        keywords: [kaplan 2020, under-fed, 1.7 tokens per parameter, re-measurement in 2022, a 70 billion model beat a 280 billion one]
      - topic: serving cost is the bill that arrives on every single answer
        keywords: [inference cost per token, two steps per parameter per token, billions of queries, training paid once, answering paid forever]
      - topic: failure — treating the compute-optimal recipe as the final answer; remedy — train smaller and longer against the whole lifetime
        keywords: [the recipe has no serving term, nothing errors you just overpay, inference-aware budgeting, smaller-but-longer, 1 trillion tokens with 7 billion parameters, 140 tokens per parameter, lifetime cost]
      - topic: failure — the data wall, because good text is finite
        keywords: [finite high-quality tokens, budgets grow but the text does not, we ran out of good data, gain per extra token flattens]
      - topic: remedy — repeating the data, and how many passes still help
        keywords: [epoch, repeated data, about four passes still help, help fades near sixteen passes, training when data is short]
      - topic: remedy — curation, so filter and de-duplicate the text you already have
        keywords: [data quality, de-duplication, throwing out junk, fewer but better tokens]
      - topic: failure — the plan does not fit, because training keeps several copies of every parameter
        keywords: [space bill per parameter, about sixteen bytes to train, gradients and optimizer state, saved working-out, out of memory, the weights are not the whole footprint]
      - topic: remedy — sharding, so cut one model across chips and the space bill divides
        keywords: [sharding, split the model across chips, space divides by the number of chips, how many chips do i need, 80 gigabytes per chip]
      - topic: the hardware picks the shape, so an odd shape is a compromise and not a theory choice
        keywords: [wide versus deep, aspect ratio, n_layers times d_model squared, hardware compromise not theory]
      - topic: limit — these rules predict error, not skills
        keywords: [predicts next-token error, says nothing about new abilities, measured only inside a range, far extrapolation is a guess, a base model is not yet a chat model]
    deferred:
      the compute rule and the rectangle whose area is the training bill: sessions/m08-transformer-math/day-01-transformer-arithmetic
      how the parameter count follows from d_model and n_layers, and why width costs squared: sessions/m08-transformer-math/day-02-qkv-matrices
      the cache that makes serving cost grow with conversation length: sessions/m08-transformer-math/day-03-kv-cache
      what the saved working-out is and when the loop frees it: sessions/m08-transformer-math/day-04-production-code
      the first scaling paradigm in full, including why its fit came out lopsided: sessions/m10a-scaling-laws/day-01-kaplan-paradigm
      the chinchilla correction in depth, with the corrected coefficients: sessions/m10a-scaling-laws/day-02-chinchilla-correction
      how the twenty-to-one ratio was actually measured with equal-compute curves: sessions/m10a-scaling-laws/day-03-isoflops-methodology
      the power-law form of the loss curve and where its exponents come from: sessions/m10a-scaling-laws/day-04-power-law-derivation
      the data wall in depth, and synthetic data as a way past it: sessions/m10a-scaling-laws/day-05-data-wall
      playing with the size and tokens and budget trade interactively: sessions/m10a-scaling-laws/day-06-scaling-simulator-visualization
      byte-by-byte training memory accounting: sessions/m09a-hardware-physics/day-05-memory-footprint
      activation checkpointing when the plan still does not fit: sessions/m09a-hardware-physics/day-06-distributed-checkpointing
      how sharding actually works, splitting arrays across a device mesh: sessions/m09a-hardware-physics/day-03-sharding
      bandwidth, arithmetic intensity and rooflines, which is why today's systems squeeze is memory only: sessions/m09a-hardware-physics/day-01-rooflines
      sharding the optimizer state and gradients: sessions/m09c-sharding-parallelism/day-01-data-parallel-fsdp
      why a wide model pays a communication cost: sessions/m09c-sharding-parallelism/day-02-tensor-parallel
      why a deep model pays a communication cost: sessions/m09c-sharding-parallelism/day-03-pipeline-parallel
      distillation as a second way to a cheap-to-serve small model: sessions/m09c-sharding-parallelism/day-05-distillation
      mixture-of-experts, and how it bends the scaling law: sessions/m14a-mixture-of-experts/day-01-moe-fundamentals
      quantization, serving the same model in fewer bits per number: sessions/m17a-quantization/day-01-quantization-theory-int8
      serving economics in depth, with batching and cost per answer: sessions/m16a-inference-economics/day-03-batching-economics
      post-training, which turns the base model into a chat model: sessions/week-m21a/day-02-sft.html
      how tokens are made, and how the tokenizer changes the token count you budget: sessions/m12-addition-transformer/day-04-custom-tokenizer-design
      mixed precision and where the per-parameter byte count comes from: sessions/week-m9b/day-03-mixed-precision-recipe.html
    out_of_scope:
      - topic: learning-rate schedules, optimizer choice, and hyperparameter transfer
        reason: today's decision is size, tokens and shape under three budgets, and the recipe knobs are a different axis; contestable, because picking a learning rate for a model you have never trained is a real pretraining-logic question that would need its own day
      - topic: dollar costs and specific chip prices
        reason: prices go stale within months while the arithmetic in operations and bytes stays true for years
      - topic: critical batch size and the gradient-noise scale
        reason: batch size changes wall-clock time rather than the size-against-tokens product this day reasons about
      - topic: data licensing, copyright and where the text pile legally comes from
        reason: a genuine constraint but a legal and policy one rather than one of the three technical budgets, and one line would be worse than nothing
      - topic: the debate over whether emergent abilities are real
        reason: the day already states the honest limit that these rules predict error and not skills
      - topic: data mixture weights
        reason: choosing mixture weights needs an evaluation loop and ablation machinery the reader has not met; contestable, because in practice this is arguably the single biggest quality lever

  # ===========================================================================
  # DAY 06 — REHEARSAL. Owns NO mechanism from days 01-05; it re-derives them.
  # New coverage is only the checking discipline plus the wait-on-what tools.
  # ===========================================================================
  day-06-scaling-exercises-rehearsal:
    covers:
      - topic: re-derive from a blank page instead of re-reading
        keywords: [retrieval practice, shut the book, re-derive not re-read, self-test, the illusion of knowing, consolidation, no new mechanism today]
      - topic: keep the recipe, not the remembered number
        keywords: [named knobs, turn one knob at a time, a memorized number is fragile, swap the bytes per number, double the seq_len, one-line multiplication]
      - topic: arithmetic intensity — how much arithmetic you get per byte fetched
        keywords: [flops per byte, bytes moved, read the two inputs and write the output, a property of the work not of the chip]
      - topic: the chip's ridge point — its own break-even ratio of arithmetic to bytes
        keywords: [peak flops divided by peak bandwidth, ridge point, break-even ratio, the per-second parts cancel, a property of the chip not of the work]
      - topic: memory-bound against compute-bound — compare the work's ratio to the chip's
        keywords: [above the break-even, below the break-even, the chip is waiting on data, a faster cook cannot fix a slow door, which knob is worth turning]
      - topic: batching — the numbers you already fetched do far more work
        keywords: [serve many requests in the same trip, same bytes moved far more arithmetic, intensity climbs with batch, why servers wait to fill a batch]
      - topic: failure — the unit slip that flips your answer; remedy — label every unit and cancel them
        keywords: [4 bytes counted for a 2-byte number, peak quoted without its bandwidth, a ridge point from a different chip, gb versus gib, inflated by sparsity, no error message, dimensional analysis, units cancel top and bottom]
      - topic: the turn-a-knob-to-zero check
        keywords: [set the seq_len to 1, set the batch to 0, try a single block, does the recipe collapse to the right thing, 524288 bytes for one token]
      - topic: say the derivation out loud
        keywords: [self-explanation, you feel you understand until you try to explain, full sentences, ask why one step deeper, explain-back]
    deferred:
      two steps per multiply-accumulate, the backward pass costing about twice the forward, and the six assembled — rehearsed here, taught at day-01: sessions/m08-transformer-math/day-01-transformer-arithmetic
      what the compute rule leaves out, the attention term and the width test: sessions/m08-transformer-math/day-01-transformer-arithmetic
      checking the order of magnitude of an answer against known anchor runs: sessions/m08-transformer-math/day-01-transformer-arithmetic
      the parameter count and every shape the recipes are built from: sessions/m08-transformer-math/day-02-qkv-matrices
      the kv-cache recipe and its per-token byte count — rehearsed here, taught at day-03: sessions/m08-transformer-math/day-03-kv-cache
      why writing one token at a time waits on fetching — rehearsed here as the worked example, taught at day-03: sessions/m08-transformer-math/day-03-kv-cache
      the twenty-tokens-per-parameter split and the three budgets: sessions/m08-transformer-math/day-05-pretraining-logic
      the roofline picture itself, with its sloped memory roof and flat compute ceiling: sessions/m09a-hardware-physics/day-01-rooflines
      real chip internals and the memory hierarchy that produce the peak numbers: sessions/m09a-hardware-physics/day-02-tpu-architecture
      communication cost of splitting a model across chips: sessions/m09a-hardware-physics/day-03-sharding
      the full training-time memory bill: sessions/m09a-hardware-physics/day-05-memory-footprint
      shrinking the kv cache by sharing key and value heads: sessions/m17b-long-context-decoding/day-03-gqa-mqa
      mixed precision and quantized bytes-per-number: sessions/m16a-inference-economics/day-05-quantization-intro
      what large batches actually cost, throughput against per-user latency: sessions/m16a-inference-economics/day-03-batching-economics
      isoflop methodology and fitting the power-law loss curve: sessions/m10a-scaling-laws/day-03-isoflops-methodology
      fusing attention to cut memory traffic: sessions/m15a-custom-kernels-pallas/day-04-flashattention-paradigm
      interview logistics and behavioural preparation: sessions/m29-ship-outreach/day-08-interview-preparation
      screencast production craft: sessions/m29-ship-outreach/day-02-screencast-production-math
    out_of_scope:
      - topic: scripted here is how you would say this in an interview monologues
        reason: reader separation — a foundation lesson body is the beginner reader only; the required depth stays as one plain empowerment line and interview scripting belongs in a separate file
      - topic: deriving the exact backward-pass factor layer by layer from the chain rule
        reason: that derivation needs a compute-graph treatment m08 has not built and would swamp a consolidation day; the day rehearses the factor plus the check that catches you dropping it
      - topic: memorizing any specific vendor's peak flops and bandwidth numbers
        reason: the ridge point must land as a division the reader can perform on any two spec numbers, and quoting a vendor peak is the exact trap the unit-slip topic warns about
      - topic: calculator and spreadsheet workflows for these estimates
        reason: hand arithmetic with powers of ten is the skill this day trains, and a tool would hide the exact step the day exists to build
      - topic: screen-recording and microphone setup
        reason: the skill is spoken re-derivation, which needs no equipment; a reader with no microphone must be able to finish the day
      - topic: bilingual chinese glosses carried over from the legacy page
        reason: >
          SUPERSEDED 2026-08-15 by the full Chinese track. The old directive was
          "study material is english, with chinese only when truly necessary", which
          this entry recorded. The learner now asks for a real Chinese VERSION of each
          lesson, selectable from the sidebar. So the legacy inline glosses are still
          out of scope — but because they are the WRONG MECHANISM, not because Chinese
          is. A day gets its Chinese as a ~~~zh twin (see AUTHORING.md s5b) and the
          welded-in glosses are stripped to pure English at that point. m08 is not in
          the current translation scope (m01 and m02 are), so this day stays
          English-only for now.
```

Validated copy at `/tmp/m08_coverage.yaml`. Paste target: `/Users/ruifengli/Desktop/applied-ai-research/sessions/m08-transformer-math/_refactor/manifest.yaml` (does not exist yet — m08 has no `_refactor/` dir; use `/Users/ruifengli/Desktop/applied-ai-research/sessions/m07-thinking-in-jax/_refactor/manifest.yaml` as the surrounding-keys template, and note m08 keeps the original week scheme: labels "Week 3", frozen quest-ids `w03-d01-arith`, `w03-d02-qkv`, `w03-d03-kv`, `w03-d04-prodcode`, `w03-d05-pretrain`, `w03-d06-examrehearsal`).