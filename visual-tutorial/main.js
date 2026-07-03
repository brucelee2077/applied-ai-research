/* =========================================================================
   Attention.Lab — main.js
   Vanilla JS. No dependencies. All interactivity for the tutorial.
   Math (matmul/softmax) is the same logic verified numerically against the
   repo's attention-mechanisms.md worked example.
   ========================================================================= */
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const on = (el, ev, fn) => el && el.addEventListener(ev, fn);

  /* ---------- shared math ---------- */
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const transpose = (m) => m[0].map((_, i) => m.map((row) => row[i]));
  const matmul = (a, b) => { const bT = transpose(b); return a.map((r) => bT.map((c) => dot(r, c))); };
  const softmax = (arr) => {
    const mx = Math.max(...arr);
    const e = arr.map((x) => Math.exp(x - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  };
  const fmtBytes = (b) => {
    if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
    if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
    if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
    return b + " B";
  };

  /* =======================================================================
     NAV · progress bar · mobile toggle
     ===================================================================== */
  function initNav() {
    const links = Array.from(document.querySelectorAll(".nav-link"));
    const sections = links.map((l) => $(l.dataset.target)).filter(Boolean);
    const bar = $("progress-bar");
    const sidebar = $("sidebar");

    links.forEach((l) =>
      on(l, "click", (e) => {
        e.preventDefault();
        const t = $(l.dataset.target);
        if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
        if (sidebar) sidebar.classList.remove("open");
      })
    );

    const setActive = (id) =>
      links.forEach((l) => l.classList.toggle("active", l.dataset.target === id));

    if ("IntersectionObserver" in window) {
      const obs = new IntersectionObserver(
        (entries) => entries.forEach((en) => { if (en.isIntersecting) setActive(en.target.id); }),
        { rootMargin: "-40% 0px -55% 0px" }
      );
      sections.forEach((s) => obs.observe(s));
    }

    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      if (bar) bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + "%";
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    on($("nav-toggle"), "click", () => sidebar && sidebar.classList.toggle("open"));
  }

  /* =======================================================================
     CHECKLIST (sidebar) — persisted
     ===================================================================== */
  function initChecklist() {
    const ul = $("checklist");
    if (!ul) return;
    const KEY = "attnlab.checklist";
    const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
    const items = Array.from(document.querySelectorAll(".nav-link"))
      .filter((l) => l.dataset.target !== "home")
      .map((l) => ({ id: l.dataset.target, label: l.textContent.replace(/^\d+\s·\s/, "") }));

    ul.innerHTML = items
      .map(
        (it) =>
          `<li data-id="${it.id}" class="${saved[it.id] ? "done" : ""}">
             <input type="checkbox" ${saved[it.id] ? "checked" : ""}/>
             <span>${it.label}</span></li>`
      )
      .join("");

    ul.querySelectorAll("li").forEach((li) => {
      const cb = li.querySelector("input");
      on(cb, "change", () => {
        li.classList.toggle("done", cb.checked);
        saved[li.dataset.id] = cb.checked;
        localStorage.setItem(KEY, JSON.stringify(saved));
      });
    });

    on($("reset-progress"), "click", () => {
      localStorage.removeItem(KEY);
      ul.querySelectorAll("li").forEach((li) => {
        li.classList.remove("done");
        li.querySelector("input").checked = false;
      });
    });
  }

  /* =======================================================================
     VIZ ① — click-a-word attention heatmap
     ===================================================================== */
  function initHeatmap() {
    const box = $("heatmap-sentence");
    const cap = $("heatmap-caption");
    if (!box) return;
    // d3-viz.js owns the heatmap (#d3-heatmap) when D3 is present.
    if (window.__ATTN_D3 && window.__ATTN_D3.heatmap) return;
    const words = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"];
    // Illustrative attention rows (each ~sums to 1). "it" (idx 7) points at "cat" (idx 1).
    const A = [
      [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0.1, 0.7, 0.1, 0, 0, 0.1, 0, 0, 0, 0],
      [0, 0.45, 0.4, 0.05, 0, 0.1, 0, 0, 0, 0],
      [0, 0, 0.1, 0.8, 0.1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0.1, 0.8, 0.1, 0, 0, 0, 0],
      [0, 0, 0.1, 0.1, 0.1, 0.7, 0, 0, 0, 0],
      [0, 0.15, 0, 0, 0, 0, 0.65, 0.1, 0.1, 0],
      [0, 0.7, 0, 0, 0, 0.05, 0, 0.05, 0, 0.2],
      [0, 0.1, 0, 0, 0, 0, 0, 0.1, 0.6, 0.2],
      [0, 0.45, 0, 0, 0, 0, 0, 0.1, 0.1, 0.35],
    ];
    const caps = [
      "‘The’ mostly looks at itself and the next word.",
      "‘cat’ looks at itself and its action.",
      "‘sat’ looks back at the cat — who sat?",
      "‘on’ looks ahead to the object.",
      "‘the’ looks at the mat.",
      "‘mat’ ties to ‘sat’ and itself.",
      "‘because’ links the two clauses.",
      "‘it’ looks mostly at ‘cat’. That is how the model knows what ‘it’ means.",
      "‘was’ looks at the subject and the feeling.",
      "‘tired’ looks back at the cat — who is tired?",
    ];

    words.forEach((w, i) => {
      const b = document.createElement("button");
      b.className = "word-chip";
      b.textContent = w;
      on(b, "click", () => render(i));
      box.appendChild(b);
    });
    const chips = Array.from(box.children);

    function render(idx) {
      const w = A[idx];
      chips.forEach((chip, i) => {
        const a = w[i];
        const op = a > 0 ? 0.12 + a * 0.88 : 0;
        chip.style.backgroundColor = a > 0 ? `rgba(37,99,235,${op})` : "#fff";
        chip.style.color = op > 0.5 ? "#fff" : "#1f2735";
        chip.style.borderColor = a > 0 ? `rgba(37,99,235,${op})` : "var(--line)";
        chip.classList.toggle("selected", i === idx);
      });
      if (cap) cap.textContent = caps[idx];
    }
  }

  /* =======================================================================
     VIZ ② — live pipeline step-through ("cat sat", 2-D)
     ===================================================================== */
  function matrixHTML(title, mat, rowLabels, colLabels, cellClass, floats) {
    const cols = mat[0].length;
    let h = `<div class="matrix-wrap"><div class="matrix-title">${title}</div>`;
    h += `<div class="matrix" style="grid-template-columns:auto repeat(${cols},auto)">`;
    h += `<div class="cell lbl"></div>`;
    colLabels.forEach((c) => (h += `<div class="cell lbl">${c}</div>`));
    mat.forEach((row, i) => {
      h += `<div class="cell lbl">${rowLabels[i]}</div>`;
      row.forEach((v) => {
        const val = floats ? (Math.round(v * 100) / 100).toFixed(2) : v;
        h += `<div class="cell ${cellClass || ""}">${val}</div>`;
      });
    });
    return h + `</div></div>`;
  }

  function initPipeline() {
    const visuals = $("pipe-visuals");
    const explain = $("pipe-explain");
    const stepEl = $("pipe-step");
    if (!visuals) return;
    // d3-viz.js owns this when D3 is present; skip the DOM fallback.
    if (window.__ATTN_D3 && window.__ATTN_D3.pipeline) return;

    const W = ["cat", "sat"], D = ["d1", "d2"];
    const X = [[1, 0], [0, 1]], WQ = [[1, 0], [0, 1]], WK = [[0, 1], [1, 0]], WV = [[1, 1], [0, 1]];
    const Q = matmul(X, WQ), K = matmul(X, WK), V = matmul(X, WV);
    const S = matmul(Q, transpose(K));
    const Sc = S.map((r) => r.map((v) => v / Math.sqrt(2)));
    const Wt = Sc.map(softmax);
    const O = matmul(Wt, V);

    const steps = [
      { t: "每个 token 先是一个向量。We learn three projection matrices W_Q, W_K, W_V.",
        r: () => matrixHTML("Embeddings X · 2×2", X, W, D) + matrixHTML("W_Q · 2×2", WQ, ["·", "·"], ["·", "·"], "q") },
      { t: "Project to get Query (looking-for) and Key (advertising). Value is built the same way.",
        r: () => matrixHTML("Query Q · 2×2", Q, W, D, "q") + matrixHTML("Key K · 2×2", K, W, D, "k") },
      { t: "Score every query against every key: S = Q·Kᵀ. This is the n×n object — the O(n²) cost.",
        r: () => matrixHTML("Scores Q·Kᵀ · 2×2", S, W, ["cat", "sat"]) },
      { t: "Divide by √2 to keep softmax gentle, then softmax each row → weights that sum to 1.",
        r: () => matrixHTML("Scaled ÷√2 · 2×2", Sc, W, ["cat", "sat"], "", true) + matrixHTML("Softmax weights · 2×2", Wt, W, ["cat", "sat"], "", true) },
      { t: "Blend the Values with those weights. The routing weight is a volume knob on the payload V.",
        r: () => matrixHTML("Value V (payload) · 2×2", V, W, D, "v") + matrixHTML("Output = W·V · 2×2", O, W, D, "", true) },
    ];

    let step = 0;
    const draw = () => {
      visuals.innerHTML = steps[step].r();
      if (explain) explain.textContent = steps[step].t;
      if (stepEl) stepEl.textContent = `Stage ${step} / ${steps.length - 1}`;
      const p = $("pipe-prev"), n = $("pipe-next");
      if (p) p.disabled = step === 0;
      if (n) n.disabled = step === steps.length - 1;
    };
    on($("pipe-prev"), "click", () => { if (step > 0) { step--; draw(); } });
    on($("pipe-next"), "click", () => { if (step < steps.length - 1) { step++; draw(); } });
    draw();
  }

  /* =======================================================================
     VIZ ③ — √d_k "break it" scaling toggle
     ===================================================================== */
  function initScaling() {
    const toggle = $("scale-toggle");
    if (!toggle) return;
    // d3-viz.js owns this (twin bars + d_k selector) when D3 is present.
    if (window.__ATTN_D3 && window.__ATTN_D3.scaling) return;
    const raw = [10, 5, 2]; // pizza, pasta, salad
    const factor = Math.sqrt(64);
    let scaled = false;
    const set = (id, v) => { const e = $(id); if (e) e.style.width = v; };
    const txt = (id, v) => { const e = $(id); if (e) e.textContent = v; };

    function update() {
      const scores = scaled ? raw.map((v) => v / factor) : raw;
      const w = softmax(scores).map((x) => +(x * 100).toFixed(1));
      ["pizza", "pasta", "salad"].forEach((name, i) => {
        set("bar-" + name, w[i] + "%");
        txt("pct-" + name, w[i] + "%");
      });
      toggle.classList.toggle("on", scaled);
      const lo = $("scale-label-off"), ln = $("scale-label-on");
      if (lo) lo.classList.toggle("active", !scaled);
      if (ln) ln.classList.toggle("active", scaled);
      txt("scale-caption", scaled
        ? "With scaling: scores stay gentle, the model still considers multiple options."
        : "Without scaling: the biggest score crushes everything — softmax collapses to one spike.");
    }
    on(toggle, "click", () => { scaled = !scaled; update(); });
    update();
  }

  /* =======================================================================
     VIZ ④ — O(n²) cost & memory explorer
     ===================================================================== */
  function initCost() {
    const nEl = $("cost-n"), dEl = $("cost-d"), flashEl = $("cost-flash");
    if (!nEl) return;
    const sq = (id) => $(id);
    const sizePx = (bytes) => Math.max(12, Math.min(150, Math.pow(bytes, 0.25) * 1.05));

    function update() {
      const n = +nEl.value, d = +dEl.value, flash = flashEl && flashEl.checked;
      const linBytes = n * d * 2;          // token tensor, fp16  (O(n))
      const quadBytes = n * n * 2;         // attention scores, fp16  (O(n^2))
      const flashBytes = n * 64 * 2;       // FlashAttention: running stats, ~O(n)

      const out = (id, v) => { const e = $(id); if (e) e.textContent = v; };
      out("cost-n-out", n.toLocaleString());
      out("cost-d-out", d.toLocaleString());

      const lin = sq("cost-box-linear"), quad = sq("cost-box-quad");
      if (lin) { const s = sizePx(linBytes); lin.style.width = s + "px"; lin.style.height = s + "px"; }
      if (quad) {
        const eff = flash ? flashBytes : quadBytes;
        const s = sizePx(eff);
        quad.style.width = s + "px"; quad.style.height = s + "px";
        quad.classList.toggle("gone", !!flash);
      }
      out("cost-mem-linear", fmtBytes(linBytes));
      out("cost-mem-quad", flash ? "not materialized" : fmtBytes(quadBytes));

      const ratio = quadBytes / linBytes; // = n/d
      const cap = $("cost-caption");
      if (cap) {
        if (flash) {
          cap.textContent = `FlashAttention: the n×n matrix is never built — attention memory ≈ ${fmtBytes(flashBytes)} (O(n)). Compute is still O(n²): same FLOPs, far less memory IO.`;
        } else if (ratio < 1) {
          cap.textContent = `At n=${n.toLocaleString()}, the token tensor still dominates (n×n is ${ratio.toFixed(2)}× the n×d tensor). Attention memory is not yet the wall.`;
        } else {
          cap.textContent = `At n=${n.toLocaleString()}, the n×n score matrix is ${ratio.toFixed(0)}× the token tensor — attention memory now dominates. This is the long-context wall.`;
        }
      }
    }
    [nEl, dEl].forEach((el) => on(el, "input", update));
    on(flashEl, "change", update);
    update();
  }

  /* =======================================================================
     VIZ ⑤ — KV cache: training (parallel) vs inference (autoregressive)
     ===================================================================== */
  function initKV() {
    const grid = $("kv-grid");
    if (!grid) return;
    // d3-viz.js owns this (animated cache growth + GB readout) when D3 is present.
    if (window.__ATTN_D3 && window.__ATTN_D3.kv) return;
    const D = 4, PROMPT = 3;
    let mode = "train", cache = PROMPT;
    const gen = $("kv-generate"), reset = $("kv-reset"), cap = $("kv-caption");

    const rows = (count, cls, newest) => {
      let h = "";
      for (let r = 0; r < count; r++) {
        const isNew = newest && r === count - 1;
        h += `<div class="kv-cellrow">`;
        for (let c = 0; c < D; c++) h += `<div class="kv-c ${cls} ${isNew ? "new" : ""}"></div>`;
        h += `</div>`;
      }
      return h;
    };
    const col = (title, body) => `<div class="kv-col"><div class="kv-col-title">${title}</div>${body}</div>`;

    function draw(newest) {
      if (mode === "train") {
        const N = 4;
        grid.innerHTML =
          col(`Q · ${N}×${D}`, rows(N, "q")) +
          col(`K · ${N}×${D}`, rows(N, "k")) +
          col(`V · ${N}×${D}`, rows(N, "v"));
        if (cap) cap.textContent = "Training: all N tokens are known. Q, K, V are full N×d matrices and the whole N×N attention grid is computed in parallel — one big matmul. Compute-bound.";
        if (gen) gen.disabled = true;
        if (reset) reset.disabled = true;
      } else {
        grid.innerHTML =
          col(`Q · 1×${D} (new token)`, rows(1, "q", true)) +
          col(`K cache · ${cache}×${D}`, rows(cache, "k", newest)) +
          col(`V cache · ${cache}×${D}`, rows(cache, "v", newest));
        if (cap) cap.textContent = `Decode step ${cache - PROMPT}: Q is a single new row. We append one K and one V row to the cache (now ${cache}) — past keys/values are re-read, never recomputed. You must load the whole cache from HBM each step → memory-bandwidth bound.`;
        if (gen) gen.disabled = cache >= PROMPT + 8;
        if (reset) reset.disabled = false;
      }
    }

    on($("kv-train"), "click", () => {
      mode = "train"; $("kv-train").classList.add("btn-primary"); $("kv-infer").classList.remove("btn-primary"); draw();
    });
    on($("kv-infer"), "click", () => {
      mode = "infer"; cache = PROMPT; $("kv-infer").classList.add("btn-primary"); $("kv-train").classList.remove("btn-primary"); draw();
    });
    on(gen, "click", () => { if (cache < PROMPT + 8) { cache++; draw(true); } });
    on(reset, "click", () => { cache = PROMPT; draw(); });
    draw();
  }

  /* =======================================================================
     MINI QUIZ
     ===================================================================== */
  function initQuiz() {
    const box = $("quiz-container");
    if (!box) return;
    const QS = [
      { q: "What does the softmax in attention operate over?",
        o: ["The entire score matrix at once", "Each row — over the keys, summing to 1", "Each column — over the queries", "The value vectors directly"],
        a: 1, e: "Row-wise over keys: every query gets its own distribution that sums to 1." },
      { q: "If token A attends strongly to token B, what does A actually receive?",
        o: ["B's Key vector", "B's Value (payload), scaled by the Q·K weight", "B's raw embedding, copied verbatim", "Nothing — attention only produces scores"],
        a: 1, e: "Routing ≠ payload. The Q·K match sets the weight; the information transferred is B's Value." },
      { q: "Why divide the scores by √d_k?",
        o: ["To save memory", "To keep large dot products from making softmax razor-sharp and gradients tiny", "To force the matrix to be square", "It is required for causal masking"],
        a: 1, e: "Large d_k inflates dot products; scaling keeps softmax in a usable range so it can blend sources." },
      { q: "What does FlashAttention change?",
        o: ["Makes attention O(n) compute", "Returns an approximation of the output", "Avoids materializing the n×n matrix → O(n) memory, same O(n²) compute", "Removes the need for a KV cache"],
        a: 2, e: "Same exact output and same O(n²) FLOPs — it changes memory IO by never building the full n×n matrix." },
      { q: "Autoregressive decode (one token at a time) is usually bound by…",
        o: ["Compute / FLOPs", "Memory bandwidth — loading the KV cache from HBM each step", "Disk IO", "Network latency"],
        a: 1, e: "Per-token compute is tiny; you re-read the whole KV cache every step, so bandwidth dominates." },
    ];
    let score = 0, answered = 0;
    const scoreEl = $("quiz-score");
    const refresh = () => { if (scoreEl) scoreEl.textContent = `${score} / ${QS.length} correct (${answered}/${QS.length} answered)`; };

    box.innerHTML = QS.map((item, qi) =>
      `<div class="quiz-card"><div class="quiz-question">Q${qi + 1}. ${item.q}</div>
        <div class="quiz-options">${item.o.map((opt, oi) =>
          `<button class="quiz-option" data-q="${qi}" data-o="${oi}">${opt}</button>`).join("")}</div>
        <div class="quiz-feedback" id="qf-${qi}"></div></div>`).join("");
    refresh();

    box.querySelectorAll(".quiz-option").forEach((btn) =>
      on(btn, "click", () => {
        const qi = +btn.dataset.q, oi = +btn.dataset.o, item = QS[qi];
        const card = btn.closest(".quiz-card");
        if (card.dataset.done) return;
        card.dataset.done = "1";
        answered++;
        card.querySelectorAll(".quiz-option").forEach((b, i) => {
          b.disabled = true;
          if (i === item.a) b.classList.add("correct");
          else if (i === oi) b.classList.add("incorrect");
        });
        if (oi === item.a) score++;
        const fb = $("qf-" + qi);
        if (fb) { fb.textContent = (oi === item.a ? "✓ Correct. " : "✗ ") + item.e; fb.classList.add("show"); }
        refresh();
      })
    );
  }

  /* =======================================================================
     ARTIFACT checklist + RESEARCH LOG (persisted)
     ===================================================================== */
  function initArtifact() {
    const list = $("artifact-list");
    if (list) {
      const KEY = "attnlab.artifact";
      const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
      list.querySelectorAll("li").forEach((li) => {
        if (saved[li.dataset.id]) li.classList.add("done");
        on(li, "click", () => {
          li.classList.toggle("done");
          saved[li.dataset.id] = li.classList.contains("done");
          localStorage.setItem(KEY, JSON.stringify(saved));
        });
      });
    }

    const log = $("research-log"), status = $("log-status");
    if (log) {
      const KEY = "attnlab.log";
      log.value = localStorage.getItem(KEY) || "";
      if (status && log.value) status.textContent = "Restored from this browser.";
      let timer = null;
      on(log, "input", () => {
        if (status) status.textContent = "Saving…";
        clearTimeout(timer);
        timer = setTimeout(() => {
          localStorage.setItem(KEY, log.value);
          if (status) status.textContent = "Saved ✓ " + new Date().toLocaleTimeString();
        }, 400);
      });
    }
  }

  /* ---------- boot ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    initNav(); initChecklist(); initHeatmap(); initPipeline();
    initScaling(); initCost(); initKV(); initQuiz(); initArtifact();
  });
})();
