// shelf.js — Toy Shelf logic. Pure: no DOM, no localStorage, no rendering.
// Inlined into sessions/index.html between the SHELF-LOGIC markers; keep the two
// copies identical (sessions/_shelf_audit.py enforces it).

// A toy unlocks as soon as its owning lesson has ANY completed section.
// This is deliberately looser than the hub's pillStatus(), which requires
// done.produce || done.verdict — a learner meets a toy mid-lesson, so demanding
// completion would lock a toy they have already played with.
export function isUnlocked(raw){
  if (!raw) return false;
  try {
    var s = JSON.parse(raw);
    return !!(s && s.done && Object.keys(s.done).length > 0);
  } catch (e) { return false; }   // corrupt state must never break the hub
}

// ── The shelf table ───────────────────────────────────────────────────────────
// [ name, module label, owning quest-id, what you can do with it, page ]
//
// Owner = the earliest lesson where the learner meets this widget OR AN INLINE
// EQUIVALENT of it. Hand-set, never derived: grep gets neuron-boundary and
// xor-limit wrong (both are held inline by m02 Day 1) and would show them locked.
//
// Excluded: attention-heatmap, attention-multihead, attention-pipeline,
// softmax-scaling, leaky-slope — no tracked page embeds them, so no owner exists.
export var TOYS = [
  ['Broadcasting',        'm01 · Day 3', 'wf1-d03-broadcasting', 'stretch two shapes to fit',        'viz/broadcasting.html'],
  ['Matmul shapes',       'm01 · Day 4', 'wf1-d04-matmul',       'multiply matrices',                'viz/matmul.html'],
  ['Neuron weights',      'm02 · Day 1', 'wf2-d01-neuron',       'tilt a decision boundary',         'viz/neuron-boundary.html'],
  ['XOR limit',           'm02 · Day 1', 'wf2-d01-neuron',       'break a single neuron',            'viz/xor-limit.html'],
  ['Activation curves',   'm02 · Day 2', 'wf2-d02-activations',  'compare ReLU vs sigmoid',          'viz/activation-derivatives.html'],
  ['Gradient descent',    'm02 · Day 6', 'wf3-d03-training-loop','roll downhill at your own pace',   'viz/gradient-descent.html'],
  ['Embedding similarity','m03 · Day 1', 'wf4-d01-embeddings',   'compare meanings by angle',        'viz/embedding-similarity.html'],
  ['Transformer FLOPs',   'm08 · Day 1', 'w03-d01-arith',        'count the cost of a forward pass', 'viz/transformer-flops.html'],
  ['KV cache',            'm08 · Day 3', 'w03-d03-kv',           'watch memory grow per token',      'viz/kv-cache.html'],
  ['Rooflines',           'm09a · Day 1','w02-d01-roofline',     'find the wall your kernel hits',   'viz/roofline.html'],
  ['Parallelism',         'm09c · Day 1','w04-d01-fsdp',         'split a model across devices',     'viz/parallelism.html'],
  ['Scaling laws',        'm10a · Day 3','w05-d03-isoflops',     'trade parameters against data',    'viz/scaling-laws.html'],
  ['MoE routing',         'm14a · Day 1','w07-d01-moefund',      'send tokens to experts',           'viz/moe-routing.html'],
  ['Flash attention',     'm15a · Day 4','w13-d04-flashattn',    'tile attention to fit in SRAM',    'viz/flash-attention.html'],
  ['Quantization',        'm16a · Day 5','w06-d05-llmint8',      'squeeze weights and watch error',  'viz/quantization.html'],
  ['Speculative decoding','m17b · Day 4','w16-d04-specdecode',   'let a small model guess ahead',    'viz/speculative-decoding.html'],
  ['Diffusion noising',   'm22b · Day 1','m22b-d01-forward',     'add noise until it is gone',       'viz/diffusion-noising.html']
];

// Partition the shelf. `read(qid)` returns the raw localStorage string or null —
// injected so this stays testable and never touches the browser itself.
export function shelfSummary(read){
  var unlocked = [], locked = [];
  for (var i = 0; i < TOYS.length; i++){
    var raw = null;
    // localStorage.getItem throws SecurityError when site data is blocked —
    // a locked shelf is fine, a hub that stops rendering is not.
    try { raw = read(TOYS[i][2]); } catch (e) { raw = null; }
    (isUnlocked(raw) ? unlocked : locked).push(TOYS[i]);
  }
  return { total: TOYS.length, unlocked: unlocked, locked: locked };
}

// ── viz iframe height protocol ────────────────────────────────────────────────
// Must stay equal to the type in _compiler/shells/v9-base.donor (line ~606).
// Compiled lessons get that parity free from visual_integrity_gate.py, which reads
// the type from the donor; index.html is NOT compiled, so _shelf_audit.py checks it.
export var VIZ_MSG_TYPE = 'viz-height';

// Same clamp the lesson receiver uses. Returns null for anything not a real number.
export function clampHeight(px){
  if (typeof px !== 'number' || !isFinite(px)) return null;
  return Math.min(Math.max(px, 320), 3200);
}

// Which toy should be open after a click? null means "closed".
// Kept pure so the single-panel rule is testable without a DOM.
export function nextOpen(currentlyOpen, clicked){
  return currentlyOpen === clicked ? null : clicked;
}

// Should this height message be applied, and to what height?
// Returns null to ignore. Senders keep firing for ~1600ms after load
// (viz/xor-limit.html:262 fires at 80/300/800/1600ms, plus 60/420ms per click),
// so a message routinely outlives the panel that requested it.
export function acceptHeight(msg, hasLiveFrame, sourceMatches){
  if (!msg || msg.type !== VIZ_MSG_TYPE) return null;
  if (!hasLiveFrame || !sourceMatches) return null;
  return clampHeight(msg.px);
}
