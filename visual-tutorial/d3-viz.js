/* =========================================================================
   Attention.Lab — d3-viz.js
   FIVE D3 (v7) visualizations layered ADDITIVELY on top of main.js.
     A · attention heatmap          (#d3-heatmap)        — replaces word strip
     B · memory growth chart        (#d3-cost-chart)     — complements cost boxes
     C · pipeline step-through      (#pipe-visuals)      — replaces DOM tables
     D · √d_k twin-bar + d_k slider (#d3-scaling)        — replaces CSS bars
     E · KV-cache simulator         (#kv-grid)           — replaces DOM grid

   Takeover, not destruction. At parse time (BEFORE main.js's DOMContentLoaded
   fires) we set window.__ATTN_D3 only IF d3 is present. main.js reads that flag
   and skips its own DOM build for C/D/E. If d3 fails to load (e.g. file missing
   offline), the flag is never set, every init below bails with a console.warn,
   and main.js renders the plain-DOM fallback untouched. So the page is strictly
   more robust than before, never less.

   Pattern A from the d3-viz skill: direct DOM manipulation, responsive viewBox,
   clear-before-redraw, one shared body-level tooltip, accessibility
   (role='img', aria-label, <title>, keyboard), and prefers-reduced-motion.
   ========================================================================= */
(function () {
  "use strict";

  // ---- claim the DOM-fallback visualizations for D3 (only if d3 loaded) ----
  var D3_READY = !!window.d3;
  if (D3_READY) {
    window.__ATTN_D3 = { heatmap: true, pipeline: true, scaling: true, kv: true };
  }

  // ---- honor reduced-motion: collapse every animation duration to 0ms ------
  var REDUCED = !!(window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  function dur(ms) { return REDUCED ? 0 : ms; }

  // ---- palette (mirrors styles.css :root, kept independent) ----------------
  var C = {
    q: "#2563eb",        // Query — routing (cool blue)
    k: "#7c3aed",        // Key   — routing (purple)
    v: "#f97316",        // Value — payload (warm orange)
    accent: "#6366f1",   // indigo
    warn: "#ef4444",     // red
    ok: "#16a34a",       // green
    amber: "#f59e0b",
    line: "#e6e9f2",
    muted: "#64708a",
    ink: "#1f2735"
  };

  /* ---------- shared math (mirrors main.js, kept independent) ---------- */
  function dot(a, b) { return a.reduce(function (s, v, i) { return s + v * b[i]; }, 0); }
  function transpose(m) { return m[0].map(function (_, i) { return m.map(function (row) { return row[i]; }); }); }
  function matmul(a, b) { var bT = transpose(b); return a.map(function (r) { return bT.map(function (c) { return dot(r, c); }); }); }
  function softmax(arr) {
    var mx = Math.max.apply(null, arr);
    var e = arr.map(function (x) { return Math.exp(x - mx); });
    var s = e.reduce(function (a, b) { return a + b; }, 0);
    return e.map(function (x) { return x / s; });
  }

  // fmtBytes mirrors main.js so every memory readout reads identically.
  function fmtBytes(b) {
    if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
    if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
    if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
    return b + " B";
  }

  /* ---------- one shared, body-level tooltip div ---------- */
  var _tooltip = null;
  function getTooltip() {
    if (_tooltip) return _tooltip;
    _tooltip = document.createElement("div");
    _tooltip.className = "d3-tooltip";
    _tooltip.setAttribute("aria-hidden", "true");
    document.body.appendChild(_tooltip);
    return _tooltip;
  }
  function showTooltip(html, event) {
    var tt = getTooltip();
    tt.innerHTML = html;
    tt.style.opacity = "1";
    moveTooltip(event);
  }
  function moveTooltip(event) {
    var tt = getTooltip();
    var x = event.clientX + 14;
    var y = event.clientY + 14;
    var maxX = window.innerWidth - tt.offsetWidth - 8;
    if (x > maxX) x = event.clientX - tt.offsetWidth - 14;
    tt.style.left = x + "px";
    tt.style.top = y + "px";
  }
  function hideTooltip() {
    if (_tooltip) _tooltip.style.opacity = "0";
  }

  /* =======================================================================
     VISUALIZATION A — attention heatmap  (#d3-heatmap)
     Rows = the QUERY word (doing the looking); columns = the KEY word (being
     looked at); cell fill = weight. Enhancements over the first cut: fade-in
     cells, a top-key ring + "Σ = 1.00" badge on the active row (teaching that
     every row is a probability distribution), column-label highlight on hover,
     and full keyboard navigation of the rows.
     ===================================================================== */
  function initD3Heatmap() {
    if (!window.d3) { console.warn("d3 not loaded — heatmap skipped"); return; }
    var container = document.getElementById("d3-heatmap");
    if (!container) return;
    var d3 = window.d3;

    var words = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"];
    var A = [
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
    var caps = [
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

    var n = words.length;
    var caption = document.getElementById("d3-heatmap-caption");
    var defaultCaption = "Hover any cell, or click a row label, to see who that word listens to.";
    var activeRow = -1;

    // Wider right margin than the first cut so the per-row "Σ = 1.00" badge has
    // room to the right of the grid without colliding with the colour legend.
    var margin = { top: 70, right: 110, bottom: 16, left: 88 };
    var gridSize = 360;
    var width = gridSize + margin.left + margin.right;
    var height = gridSize + margin.top + margin.bottom;

    var x = d3.scaleBand().domain(d3.range(n)).range([0, gridSize]).padding(0.06);
    var y = d3.scaleBand().domain(d3.range(n)).range([0, gridSize]).padding(0.06);
    var color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

    var g; // the inner translated group, captured by render() for highlightRow

    function render() {
      d3.select(container).selectAll("*").remove();

      var svg = d3.select(container).append("svg")
        .attr("viewBox", "0 0 " + width + " " + height)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("role", "img")
        .attr("aria-label",
          "Attention heatmap. Rows are query words doing the looking; columns " +
          "are key words being looked at; darker blue means more attention.");
      svg.append("title").text(
        "A 10 by 10 attention matrix for the sentence: " + words.join(" ") +
        ". Each row is one word's attention over all words.");

      g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      // column labels (KEY word) across the top, rotated
      g.selectAll(".col-label").data(words).join("text")
        .attr("class", "d3-axis-label col-label")
        .attr("data-col", function (d, i) { return i; })
        .attr("x", function (d, i) { return x(i) + x.bandwidth() / 2; })
        .attr("y", -10)
        .attr("text-anchor", "start")
        .attr("transform", function (d, i) {
          var cx = x(i) + x.bandwidth() / 2;
          return "rotate(-45," + cx + ",-10)";
        })
        .text(function (d) { return d; });

      g.append("text").attr("class", "d3-axis-title")
        .attr("x", gridSize / 2).attr("y", -margin.top + 16).attr("text-anchor", "middle")
        .text("key word (being looked at) →");
      g.append("text").attr("class", "d3-axis-title").attr("text-anchor", "middle")
        .attr("transform", "translate(" + (-margin.left + 16) + "," + gridSize / 2 + ") rotate(-90)")
        .text("query word (doing the looking) ↓");

      // row labels (QUERY word) — clickable AND keyboard-navigable
      g.selectAll(".row-label").data(words).join("text")
        .attr("class", "d3-axis-label row-label")
        .attr("data-row", function (d, i) { return i; })
        .attr("x", -10)
        .attr("y", function (d, i) { return y(i) + y.bandwidth() / 2; })
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "central")
        .attr("tabindex", 0)
        .attr("role", "button")
        .attr("aria-label", function (d, i) { return "Query word " + d + ", row " + (i + 1) + " of " + n; })
        .style("cursor", "pointer")
        .text(function (d) { return d; })
        .on("mouseenter", function (event, d) { highlightRow(words.indexOf(d)); })
        .on("focus", function (event, d) { highlightRow(words.indexOf(d)); })
        .on("click", function (event, d) { highlightRow(words.indexOf(d)); })
        .on("keydown", function (event, d) {
          var i = words.indexOf(d), ni = i;
          if (event.key === "ArrowDown" || event.key === "ArrowRight") ni = Math.min(n - 1, i + 1);
          else if (event.key === "ArrowUp" || event.key === "ArrowLeft") ni = Math.max(0, i - 1);
          else return;
          event.preventDefault();
          highlightRow(ni);
          // move focus to the newly active row label
          g.selectAll(".row-label").filter(function (dd, ii) { return ii === ni; }).node().focus();
        });

      // the cells
      var cells = [];
      for (var r = 0; r < n; r++)
        for (var c = 0; c < n; c++)
          cells.push({ row: r, col: c, weight: A[r][c] });

      var cellSel = g.selectAll(".d3-cell").data(cells).join("rect")
        .attr("class", "d3-cell mtx-cell")
        .attr("data-row", function (d) { return d.row; })
        .attr("x", function (d) { return x(d.col); })
        .attr("y", function (d) { return y(d.row); })
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .attr("rx", 3)
        .attr("fill", function (d) { return d.weight > 0 ? color(d.weight) : "#fbfcfe"; })
        .attr("stroke", C.line)
        .attr("stroke-width", 1)
        .on("mouseenter", function (event, d) {
          var pct = Math.round(d.weight * 100);
          showTooltip("<b>‘" + words[d.row] + "’</b> attends <b>" + pct + "%</b> to <b>‘" + words[d.col] + "’</b>", event);
          highlightRow(d.row);
          highlightCol(d.col);
        })
        .on("mousemove", function (event) { moveTooltip(event); })
        .on("mouseleave", function () { hideTooltip(); highlightCol(-1); });

      // fade the cells in (staggered down the matrix), unless reduced-motion
      if (!REDUCED) {
        cellSel.attr("opacity", 0).transition().duration(dur(450))
          .delay(function (d) { return d.row * 24 + d.col * 4; })
          .attr("opacity", 1);
      }

      // colour legend (pushed to the far right; Σ badge lives left of it)
      drawLegend(svg, g);

      // overlay group sits ON TOP of the cells for the active-row annotations
      g.append("g").attr("class", "hl-overlay");
    }

    function drawLegend(svg, g) {
      var legendH = 140, legendW = 12, legendX = gridSize + 56, legendTop = 6;
      var defs = svg.append("defs");
      var grad = defs.append("linearGradient").attr("id", "d3-heatmap-grad")
        .attr("x1", "0%").attr("y1", "100%").attr("x2", "0%").attr("y2", "0%");
      d3.range(0, 1.01, 0.1).forEach(function (t) {
        grad.append("stop").attr("offset", t * 100 + "%").attr("stop-color", color(t));
      });
      var legend = g.append("g").attr("transform", "translate(" + legendX + "," + legendTop + ")");
      legend.append("rect").attr("width", legendW).attr("height", legendH).attr("rx", 3)
        .attr("fill", "url(#d3-heatmap-grad)").attr("stroke", C.line);
      legend.append("text").attr("class", "d3-axis-label").attr("x", legendW + 6).attr("y", 4)
        .attr("dominant-baseline", "hanging").text("100%");
      legend.append("text").attr("class", "d3-axis-label").attr("x", legendW + 6).attr("y", legendH)
        .attr("dominant-baseline", "baseline").text("0%");
      legend.append("text").attr("class", "d3-axis-label").attr("x", legendW / 2).attr("y", legendH + 16)
        .attr("text-anchor", "middle").text("weight");
    }

    // highlight one query row: outline its cells, bold its label, draw a ring
    // on its top key, show the Σ badge, and write the row's caption.
    function highlightRow(idx) {
      if (!g) return;
      activeRow = idx;
      g.selectAll(".d3-cell").classed("row-active", function (d) { return d.row === idx; });
      g.selectAll(".row-label").classed("row-active", function (d, i) { return i === idx; });
      if (caption) caption.textContent = caps[idx];

      // find the top key (argmax) for the ring
      var top = 0, sum = 0;
      for (var c = 0; c < n; c++) { sum += A[idx][c]; if (A[idx][c] > A[idx][top]) top = c; }

      var ov = g.select(".hl-overlay");
      ov.selectAll("*").remove();
      // ring on the strongest cell — "this is who the word listens to most"
      ov.append("rect")
        .attr("x", x(top) - 2).attr("y", y(idx) - 2)
        .attr("width", x.bandwidth() + 4).attr("height", y.bandwidth() + 4)
        .attr("rx", 4).attr("fill", "none")
        .attr("stroke", C.accent).attr("stroke-width", 3)
        .attr("opacity", 0).transition().duration(dur(180)).attr("opacity", 1);
      // Σ badge to the right of the active row — every row sums to 1
      var badge = ov.append("g").attr("transform", "translate(" + (gridSize + 6) + "," + (y(idx) + y.bandwidth() / 2) + ")");
      badge.append("rect").attr("x", 0).attr("y", -11).attr("width", 44).attr("height", 22).attr("rx", 6)
        .attr("fill", "#eef0fe").attr("stroke", C.accent).attr("stroke-width", 1);
      badge.append("text").attr("x", 22).attr("y", 1).attr("text-anchor", "middle")
        .attr("dominant-baseline", "central").attr("class", "d3-num")
        .attr("fill", C.accent).style("font-size", "10.5px")
        .text("Σ=" + sum.toFixed(2));
    }

    function highlightCol(idx) {
      if (!g) return;
      g.selectAll(".col-label")
        .attr("fill", function (d, i) { return i === idx ? C.q : null; })
        .attr("font-weight", function (d, i) { return i === idx ? 800 : null; });
    }

    render();
    if (caption) caption.textContent = defaultCaption;
  }

  /* =======================================================================
     VISUALIZATION B — memory growth chart  (#d3-cost-chart)
     Memory in BYTES (log y) vs sequence length n (log x); three curves with the
     SAME formulas main.js uses. Enhancement: a crossover marker showing exactly
     where the n×n score matrix overtakes the n×d token tensor (at n = d).
     ===================================================================== */
  function initD3CostChart() {
    if (!window.d3) { console.warn("d3 not loaded — cost chart skipped"); return; }
    var container = document.getElementById("d3-cost-chart");
    if (!container) return;
    var d3 = window.d3;

    var nEl = document.getElementById("cost-n");
    var dEl = document.getElementById("cost-d");
    var flashEl = document.getElementById("cost-flash");
    if (!nEl || !dEl) return;

    var bytesTokens = function (n, d) { return n * d * 2; };   // n×d token tensor, fp16
    var bytesScores = function (n) { return n * n * 2; };      // n×n attention scores, fp16
    // FlashAttention keeps only running softmax stats, ~O(n). The 64 is an
    // illustrative per-row stat size — the point is the SLOPE (linear, not n²),
    // not the exact constant, which varies with block size in real kernels.
    var bytesFlash = function (n) { return n * 64 * 2; };

    var N_MIN = 128, N_MAX = 131072;
    var nSamples = d3.range(0, 81).map(function (i) {
      return Math.round(N_MIN * Math.pow(N_MAX / N_MIN, i / 80));
    });

    var margin = { top: 24, right: 132, bottom: 48, left: 70 };
    var width = 720, height = 380;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    function readControls() {
      return { n: +nEl.value, d: +dEl.value, flash: !!(flashEl && flashEl.checked) };
    }

    function render() {
      var ctrl = readControls();
      var d = ctrl.d, curN = ctrl.n, flash = ctrl.flash;

      var seriesTokens = nSamples.map(function (nn) { return { n: nn, bytes: bytesTokens(nn, d) }; });
      var seriesScores = nSamples.map(function (nn) { return { n: nn, bytes: bytesScores(nn) }; });
      var seriesFlash = nSamples.map(function (nn) { return { n: nn, bytes: bytesFlash(nn) }; });

      d3.select(container).selectAll("*").remove();

      var svg = d3.select(container).append("svg")
        .attr("viewBox", "0 0 " + width + " " + height)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("role", "img")
        .attr("aria-label",
          "Memory growth chart. Memory in bytes on a log scale versus sequence " +
          "length n on a log scale, for the token tensor, the n by n attention " +
          "scores, and FlashAttention.");
      svg.append("title").text(
        "The n by n attention-score line crosses above the n by d token-tensor " +
        "line and then explodes — the long-context memory wall.");

      var g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      var x = d3.scaleLog().domain([N_MIN, N_MAX]).range([0, innerW]);
      var allBytes = seriesTokens.concat(seriesScores).concat(seriesFlash).map(function (p) { return p.bytes; });
      var y = d3.scaleLog().domain([Math.max(1, d3.min(allBytes)), d3.max(allBytes)]).range([innerH, 0]).nice();

      var xAxis = d3.axisBottom(x).tickValues([128, 512, 2048, 8192, 32768, 131072])
        .tickFormat(function (v) { return v >= 1024 ? v / 1024 + "k" : "" + v; });
      var yAxis = d3.axisLeft(y).ticks(6, function (v) { return fmtBytes(v); });

      g.append("g").attr("class", "d3-axis").attr("transform", "translate(0," + innerH + ")").call(xAxis);
      g.append("g").attr("class", "d3-axis").call(yAxis);

      g.append("text").attr("class", "d3-axis-title").attr("x", innerW / 2).attr("y", innerH + 40)
        .attr("text-anchor", "middle").text("sequence length n (log scale)");
      g.append("text").attr("class", "d3-axis-title").attr("text-anchor", "middle")
        .attr("transform", "translate(" + (-margin.left + 16) + "," + innerH / 2 + ") rotate(-90)")
        .text("memory, bytes (log scale)");

      var line = d3.line().x(function (p) { return x(p.n); }).y(function (p) { return y(p.bytes); });
      var C_TOKEN = C.q, C_SCORES = C.warn, C_FLASH = C.accent;

      g.append("path").attr("d", line(seriesTokens)).attr("fill", "none").attr("stroke", C_TOKEN).attr("stroke-width", 2.5);
      g.append("path").attr("d", line(seriesScores)).attr("fill", "none").attr("stroke", C_SCORES)
        .attr("stroke-width", flash ? 1.8 : 2.8).attr("stroke-dasharray", flash ? "6 5" : null).attr("opacity", flash ? 0.35 : 1);
      g.append("path").attr("d", line(seriesFlash)).attr("fill", "none").attr("stroke", C_FLASH)
        .attr("stroke-width", flash ? 3 : 1.6).attr("stroke-dasharray", flash ? null : "3 4").attr("opacity", flash ? 1 : 0.5);

      // crossover marker: n×n == n×d  ⇔  n == d. Within the slider's range, so
      // always visible. This is the single most important teaching point.
      if (!flash) {
        var xc = Math.max(N_MIN, Math.min(N_MAX, d));
        g.append("line").attr("x1", x(xc)).attr("x2", x(xc)).attr("y1", 0).attr("y2", innerH)
          .attr("stroke", C.amber).attr("stroke-width", 1.4).attr("stroke-dasharray", "2 3").attr("opacity", 0.9);
        g.append("text").attr("class", "d3-axis-label").attr("x", x(xc)).attr("y", innerH - 6)
          .attr("text-anchor", x(xc) > innerW * 0.7 ? "end" : "start")
          .attr("dx", x(xc) > innerW * 0.7 ? -4 : 4)
          .attr("fill", C.amber).style("font-weight", 700)
          .text("n×n overtakes n×d at n = d = " + d.toLocaleString());
      }

      // vertical guide at the current n
      var clampedN = Math.max(N_MIN, Math.min(N_MAX, curN));
      g.append("line").attr("x1", x(clampedN)).attr("x2", x(clampedN)).attr("y1", 0).attr("y2", innerH)
        .attr("stroke", "#94a3b8").attr("stroke-width", 1).attr("stroke-dasharray", "4 4");
      g.append("text").attr("class", "d3-axis-label").attr("x", x(clampedN) + 4).attr("y", 10)
        .text("n = " + curN.toLocaleString());

      var dotData = [
        { bytes: bytesTokens(clampedN, d), color: C_TOKEN, name: "n×d token tensor" },
        { bytes: bytesScores(clampedN), color: C_SCORES, name: "n×n attention scores" },
        { bytes: bytesFlash(clampedN), color: C_FLASH, name: "FlashAttention" },
      ];
      g.selectAll(".d3-dot").data(dotData).join("circle").attr("class", "d3-dot")
        .attr("cx", x(clampedN)).attr("cy", function (p) { return y(p.bytes); }).attr("r", 4.5)
        .attr("fill", function (p) { return p.color; }).attr("stroke", "#fff").attr("stroke-width", 1.5)
        .attr("role", "img")
        .attr("aria-label", function (p) { return p.name + " at n=" + curN.toLocaleString() + ": " + fmtBytes(p.bytes); });

      var legend = g.append("g").attr("transform", "translate(" + (innerW + 18) + ",6)");
      var legendItems = [
        { color: C_TOKEN, text: "n×d token · O(n)", faded: false },
        { color: C_SCORES, text: "n×n scores · O(n²)", faded: flash },
        { color: C_FLASH, text: "FlashAttn · O(n)", faded: !flash },
      ];
      legendItems.forEach(function (item, i) {
        var row = legend.append("g").attr("transform", "translate(0," + i * 22 + ")").attr("opacity", item.faded ? 0.45 : 1);
        row.append("line").attr("x1", 0).attr("x2", 22).attr("y1", 6).attr("y2", 6).attr("stroke", item.color).attr("stroke-width", 3);
        row.append("text").attr("class", "d3-axis-label").attr("x", 28).attr("y", 10).text(item.text);
      });

      var bisect = d3.bisector(function (p) { return p.n; }).left;
      g.append("rect").attr("width", innerW).attr("height", innerH).attr("fill", "transparent")
        .on("mousemove", function (event) {
          var mx = d3.pointer(event, this)[0];
          var i = bisect(seriesTokens, x.invert(mx));
          if (i >= seriesTokens.length) i = seriesTokens.length - 1;
          if (i < 0) i = 0;
          var nn = seriesTokens[i].n;
          showTooltip(
            "<b>n = " + nn.toLocaleString() + "</b><br>" +
            "<span style='color:" + C_TOKEN + "'>■</span> token n×d: " + fmtBytes(bytesTokens(nn, d)) + "<br>" +
            "<span style='color:" + C_SCORES + "'>■</span> scores n×n: " + fmtBytes(bytesScores(nn)) + "<br>" +
            "<span style='color:" + C_FLASH + "'>■</span> flash: " + fmtBytes(bytesFlash(nn)), event);
        })
        .on("mouseleave", function () { hideTooltip(); });
    }

    nEl.addEventListener("input", render);
    dEl.addEventListener("input", render);
    if (flashEl) flashEl.addEventListener("change", render);
    render();
  }

  /* =======================================================================
     SHARED — a labelled matrix block for the pipeline (viz C)
     Draws an r×c grid of rounded cells with numbers, a title above, and a
     "shape" tag below. Returns refs so callers can cross-highlight cells.
     ===================================================================== */
  var CELL = 36, GAP = 5;
  function matW(cols) { return cols * CELL + (cols - 1) * GAP; }
  function matH(rows) { return rows * CELL + (rows - 1) * GAP; }

  function drawMatrix(parent, mat, ox, oy, opts) {
    var d3 = window.d3;
    opts = opts || {};
    var rows = mat.length, cols = mat[0].length;
    var g = parent.append("g").attr("transform", "translate(" + ox + "," + oy + ")");

    if (opts.title)
      g.append("text").attr("class", "d3-axis-title")
        .attr("x", matW(cols) / 2).attr("y", -10).attr("text-anchor", "middle").text(opts.title);

    var data = [];
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) data.push({ r: r, c: c, v: mat[r][c] });

    var mc = g.selectAll("g.mc").data(data).join("g").attr("class", "mc")
      .attr("transform", function (d) { return "translate(" + (d.c * (CELL + GAP)) + "," + (d.r * (CELL + GAP)) + ")"; });
    mc.append("rect").attr("class", "mtx-cell")
      .attr("data-r", function (d) { return d.r; }).attr("data-c", function (d) { return d.c; })
      .attr("width", CELL).attr("height", CELL).attr("rx", 6)
      .attr("fill", function (d) { return opts.fill ? opts.fill(d.v, d.r, d.c) : "#f3f5fb"; })
      .attr("stroke", opts.stroke || C.line).attr("stroke-width", 1);
    mc.append("text").attr("class", "d3-num")
      .attr("x", CELL / 2).attr("y", CELL / 2).attr("text-anchor", "middle").attr("dominant-baseline", "central")
      .attr("fill", function (d) { return opts.textFill ? opts.textFill(d.v) : C.ink; })
      .text(function (d) { return opts.floats ? (Math.round(d.v * 100) / 100).toFixed(2) : d.v; });

    if (opts.shape)
      g.append("text").attr("class", "d3-shape")
        .attr("x", matW(cols) / 2).attr("y", matH(rows) + 14).attr("text-anchor", "middle").text(opts.shape);

    return { g: g, mc: mc, rects: g.selectAll(".mtx-cell"), w: matW(cols), h: matH(rows) };
  }

  // fill helpers shared across pipeline matrices
  function seqFill(maxAbs, baseRGB) { // value→opacity tint of baseRGB
    return function (v) {
      var norm = maxAbs > 0 ? Math.abs(v) / maxAbs : 0;
      return "rgba(" + baseRGB + "," + (0.06 + 0.55 * norm).toFixed(3) + ")";
    };
  }

  /* =======================================================================
     VISUALIZATION C — pipeline step-through  (#pipe-visuals)
     The "cat sat" 2-D worked example, stage by stage, as animated D3 matrices.
     On the two matmul stages (scores, output) hovering a result cell lights up
     the exact row and column that produced it — making "score = query · key"
     and "output = weights · values" physically visible.
     ===================================================================== */
  function initD3Pipeline() {
    if (!window.d3) { console.warn("d3 not loaded — pipeline skipped"); return; }
    var visuals = document.getElementById("pipe-visuals");
    if (!visuals) return;
    var d3 = window.d3;

    var explain = document.getElementById("pipe-explain");
    var stepEl = document.getElementById("pipe-step");
    var prevBtn = document.getElementById("pipe-prev");
    var nextBtn = document.getElementById("pipe-next");

    var W = ["cat", "sat"];
    var X = [[1, 0], [0, 1]], WQ = [[1, 0], [0, 1]], WK = [[0, 1], [1, 0]], WV = [[1, 1], [0, 1]];
    var Q = matmul(X, WQ), K = matmul(X, WK), V = matmul(X, WV);
    var Kt = transpose(K);
    var S = matmul(Q, Kt);
    var Sc = S.map(function (r) { return r.map(function (v) { return v / Math.sqrt(2); }); });
    var Wt = Sc.map(softmax);
    var O = matmul(Wt, V);

    // colour palettes per matrix role
    var qSoft = "#eaf1ff", kSoft = "#f3ecff", vSoft = "#fff1e6", nSoft = "#f3f5fb";
    var fillNeutral = function () { return nSoft; };
    var fillQ = function () { return qSoft; }, fillK = function () { return kSoft; }, fillV = function () { return vSoft; };
    var fillScore = seqFill(1, "239,68,68");      // red tint by magnitude
    var fillWeight = function (v) { return d3.interpolateBlues(v); }; // [0,1] blues
    var weightText = function (v) { return v > 0.6 ? "#fff" : C.ink; };
    var fillOut = seqFill(1, "249,115,22");        // orange tint by magnitude

    var SVG_W = 720, SVG_H = 220, CY = 116;
    var step = 0, NSTEPS = 5;

    var explainText = [
      "Each token is a row of X. We learn three matrices W_Q, W_K, W_V — the only trained parts here.",
      "Project X by each matrix: Q = X·W_Q (what each token is looking for), K (what it advertises), V (its payload).",
      "Score every query against every key: S = Q·Kᵀ. Hover a cell — it is exactly one query row times one key row. This n×n object grows with the square of the sequence length — the source of attention's O(n²) memory and compute cost.",
      "Divide by √d_k = √2 to keep softmax gentle, then softmax EACH ROW → weights that sum to 1. Hover a weight cell to see its row sums to 1.00.",
      "Blend the values: Output = Weights · V. Hover an output cell — the routing weight is a volume knob on the payload V. Output ‘cat’ = [0.33, 1.00].",
    ];

    function layout(parent, items) {
      // items: {kind:'m', w, h, draw(parent,x,y)→ref} | {kind:'op', w, text}
      var total = items.reduce(function (s, it) { return s + it.w; }, 0) + 14 * (items.length - 1);
      var x = Math.max(8, (SVG_W - total) / 2);
      var refs = [];
      items.forEach(function (it) {
        if (it.kind === "m") {
          refs.push(it.draw(parent, x, CY - it.h / 2));
        } else {
          parent.append("text").attr("class", "d3-op").attr("x", x + it.w / 2).attr("y", CY)
            .attr("text-anchor", "middle").attr("dominant-baseline", "central").text(it.text);
          refs.push(null);
        }
        x += it.w + 14;
      });
      return refs;
    }

    function m(mat, opts, w, h) { return { kind: "m", w: matW(mat[0].length), h: matH(mat.length), draw: function (p, x, y) { return drawMatrix(p, mat, x, y, opts); } }; }
    function op(text, w) { return { kind: "op", w: w || 30, text: text }; }

    function clearHi(g) {
      g.selectAll(".mtx-cell").classed("contrib-q", false).classed("contrib-k", false).classed("contrib-v", false).classed("target", false);
    }

    function drawStep() {
      d3.select(visuals).selectAll("*").remove();
      var svg = d3.select(visuals).append("svg")
        .attr("viewBox", "0 0 " + SVG_W + " " + SVG_H)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("role", "img")
        .attr("aria-label", "Attention pipeline, stage " + step + ": " + explainText[step]);
      svg.node().style.width = "100%";
      svg.node().style.height = "auto";
      svg.node().style.maxWidth = SVG_W + "px";
      svg.node().style.display = "block";
      svg.node().style.margin = "0 auto";
      var g = svg.append("g");
      var refs;

      if (step === 0) {
        refs = layout(g, [
          m(X, { title: "X (embeddings)", shape: "2×2", fill: fillNeutral }),
          op("learned →", 70),
          m(WQ, { title: "W_Q", shape: "2×2", fill: fillQ }),
          m(WK, { title: "W_K", shape: "2×2", fill: fillK }),
          m(WV, { title: "W_V", shape: "2×2", fill: fillV }),
        ]);
      } else if (step === 1) {
        refs = layout(g, [
          m(Q, { title: "Q = X·W_Q", shape: "2×2 (looking-for)", fill: fillQ }),
          m(K, { title: "K = X·W_K", shape: "2×2 (advertising)", fill: fillK }),
          m(V, { title: "V = X·W_V", shape: "2×2 (payload)", fill: fillV }),
        ]);
      } else if (step === 2) {
        refs = layout(g, [
          m(Q, { title: "Q", shape: "2×2", fill: fillQ }),
          op("×"),
          m(Kt, { title: "Kᵀ", shape: "2×2", fill: fillK }),
          op("="),
          m(S, { title: "S = Q·Kᵀ", shape: "2×2 scores", fill: fillScore, textFill: function (v) { return v >= 1 ? "#fff" : C.ink; } }),
        ]);
        // hover OR keyboard-focus on a score cell → highlight the query row + key row that built it
        var qRef = refs[0], ktRef = refs[2], sRef = refs[4];
        function hiScore(node, d) {
          clearHi(g);
          qRef.rects.classed("contrib-q", function (dd) { return dd.r === d.r; });
          ktRef.rects.classed("contrib-k", function (dd) { return dd.c === d.c; }); // Kᵀ column j = original key (row j of K)
          d3.select(node).selectAll("rect.mtx-cell").classed("target", true);
        }
        sRef.mc.style("cursor", "crosshair")
          .attr("tabindex", 0).attr("role", "img")
          .attr("aria-label", function (d) { return "Score: query " + W[d.r] + " times key " + W[d.c] + " = " + S[d.r][d.c]; })
          .on("mouseenter", function (event, d) {
            hiScore(this, d);
            showTooltip("<b>S[" + d.r + "][" + d.c + "] = " + S[d.r][d.c] + "</b><br>query ‘" + W[d.r] + "’ · key ‘" + W[d.c] + "’", event);
          })
          .on("mousemove", function (e) { moveTooltip(e); })
          .on("mouseleave", function () { clearHi(g); hideTooltip(); })
          .on("focus", function (event, d) { hiScore(this, d); })
          .on("blur", function () { clearHi(g); });
      } else if (step === 3) {
        refs = layout(g, [
          m(S, { title: "S", shape: "2×2", fill: fillScore, textFill: function (v) { return v >= 1 ? "#fff" : C.ink; } }),
          op("÷√2", 40),
          m(Sc, { title: "Scaled", shape: "2×2", fill: seqFill(1, "239,68,68"), floats: true }),
          op("softmax→", 64),
          m(Wt, { title: "Weights", shape: "2×2 (rows sum to 1)", fill: fillWeight, floats: true, textFill: weightText }),
        ]);
        var wRef = refs[4];
        function hiWeight(d) { clearHi(g); wRef.rects.classed("target", function (dd) { return dd.r === d.r; }); }
        wRef.mc.style("cursor", "crosshair")
          .attr("tabindex", 0).attr("role", "img")
          .attr("aria-label", function (d) { return "Softmax weight row " + W[d.r] + " = [" + Wt[d.r].map(function (x) { return x.toFixed(2); }).join(", ") + "], sums to 1"; })
          .on("mouseenter", function (event, d) {
            hiWeight(d);
            var rowSum = Wt[d.r].reduce(function (a, b) { return a + b; }, 0);
            showTooltip("<b>row ‘" + W[d.r] + "’</b>: softmax over keys → [" +
              Wt[d.r].map(function (x) { return x.toFixed(2); }).join(", ") + "]<br><b>Σ = " + rowSum.toFixed(2) + "</b> (a probability distribution)", event);
          })
          .on("mousemove", function (e) { moveTooltip(e); })
          .on("mouseleave", function () { clearHi(g); hideTooltip(); })
          .on("focus", function (event, d) { hiWeight(d); })
          .on("blur", function () { clearHi(g); });
      } else {
        refs = layout(g, [
          m(Wt, { title: "Weights", shape: "2×2", fill: fillWeight, floats: true, textFill: weightText }),
          op("×"),
          m(V, { title: "V (payload)", shape: "2×2", fill: fillV }),
          op("="),
          m(O, { title: "Output = W·V", shape: "2×2", fill: fillOut, floats: true, textFill: function (v) { return v >= 0.9 ? "#fff" : C.ink; } }),
        ]);
        var w2 = refs[0], vRef = refs[2], oRef = refs[4];
        function hiOut(node, d) {
          clearHi(g);
          w2.rects.classed("contrib-q", function (dd) { return dd.r === d.r; });    // weights row
          vRef.rects.classed("contrib-v", function (dd) { return dd.c === d.c; });  // value column
          d3.select(node).selectAll("rect.mtx-cell").classed("target", true);
        }
        oRef.mc.style("cursor", "crosshair")
          .attr("tabindex", 0).attr("role", "img")
          .attr("aria-label", function (d) { return "Output row " + W[d.r] + ", dim " + (d.c + 1) + " = " + O[d.r][d.c].toFixed(2) + ", a weighted blend of column " + (d.c + 1) + " of V"; })
          .on("mouseenter", function (event, d) {
            hiOut(this, d);
            showTooltip("<b>Output[" + W[d.r] + "][d" + (d.c + 1) + "] = " + O[d.r][d.c].toFixed(2) + "</b><br>blend of column " + (d.c + 1) + " of V, weighted by row ‘" + W[d.r] + "’", event);
          })
          .on("mousemove", function (e) { moveTooltip(e); })
          .on("mouseleave", function () { clearHi(g); hideTooltip(); })
          .on("focus", function (event, d) { hiOut(this, d); })
          .on("blur", function () { clearHi(g); });
      }

      // entrance animation: fade/slide the whole stage in
      if (!REDUCED) {
        g.attr("opacity", 0).attr("transform", "translate(0,8)")
          .transition().duration(dur(260)).attr("opacity", 1).attr("transform", "translate(0,0)");
      }

      if (explain) explain.textContent = explainText[step];
      if (stepEl) stepEl.textContent = "Stage " + step + " / " + (NSTEPS - 1);
      if (prevBtn) prevBtn.disabled = step === 0;
      if (nextBtn) nextBtn.disabled = step === NSTEPS - 1;
    }

    if (prevBtn) prevBtn.addEventListener("click", function () { if (step > 0) { step--; drawStep(); } });
    if (nextBtn) nextBtn.addEventListener("click", function () { if (step < NSTEPS - 1) { step++; drawStep(); } });
    drawStep();
  }

  /* =======================================================================
     VISUALIZATION D — √d_k twin-bar + d_k selector  (#d3-scaling)
     Left bars = the logits (dot-product scores); right bars = the softmax
     weights. A d_k selector shows the core lesson: WITHOUT ÷√d_k the logits
     grow like √d_k and softmax collapses onto one option; WITH it, the
     distribution is stable no matter how large d_k gets.
     ===================================================================== */
  function initD3Scaling() {
    if (!window.d3) { console.warn("d3 not loaded — scaling skipped"); return; }
    var container = document.getElementById("d3-scaling");
    if (!container) return;
    var d3 = window.d3;

    // hide the plain CSS bars; the D3 chart takes their place
    var bars = document.querySelector(".bars");
    if (bars) bars.style.display = "none";

    var toggle = document.getElementById("scale-toggle");
    var labelOff = document.getElementById("scale-label-off");
    var labelOn = document.getElementById("scale-label-on");
    var caption = document.getElementById("scale-caption");

    var foods = [
      { name: "Pizza", emo: "🍕", color: C.v },
      { name: "Pasta", emo: "🍝", color: C.amber },
      { name: "Salad", emo: "🥗", color: C.ok },
    ];
    // `base` is the INTRINSIC (already-scaled) score each option deserves. The
    // RAW dot products a model actually computes are ≈ √d_k times bigger (that's
    // how dot-product magnitude grows with dimension). So: scaling ON shows the
    // gentle `base`; scaling OFF shows the raw `base·√d_k` that softmax sees
    // without the ÷√d_k correction.
    var base = [2.0, 1.0, 0.4];
    var DK_CHOICES = [1, 4, 16, 64];
    var dk = 64;                         // a realistic head dimension
    var scaled = false;                  // start OFF so the collapse is visible first

    // build the d_k selector row (prepended above the chart)
    var dkRow = document.createElement("div");
    dkRow.className = "dk-row";
    dkRow.innerHTML = '<span class="dk-lbl">key dimension d_k:</span>';
    DK_CHOICES.forEach(function (val) {
      var b = document.createElement("button");
      b.className = "btn" + (val === dk ? " btn-primary" : "");
      b.textContent = val;
      b.setAttribute("data-dk", val);
      b.addEventListener("click", function () {
        dk = val;
        Array.prototype.forEach.call(dkRow.querySelectorAll(".btn"), function (bb) {
          bb.classList.toggle("btn-primary", +bb.getAttribute("data-dk") === dk);
        });
        render();
      });
      dkRow.appendChild(b);
    });
    if (container.parentNode) container.parentNode.insertBefore(dkRow, container);

    var SVG_W = 700, SVG_H = 210;
    var margin = { top: 30, right: 16, bottom: 10, left: 96 };
    var rowH = 46, barH = 22;
    var logitX0 = margin.left + 14, logitW = 170;
    var weightX0 = margin.left + 270, weightW = 170;

    function render() {
      var logits = scaled ? base.slice() : base.map(function (b) { return b * Math.sqrt(dk); });
      var weights = softmax(logits).map(function (w) { return w * 100; });
      var maxLogit = Math.max.apply(null, logits.concat([0.001]));

      var xLogit = d3.scaleLinear().domain([0, maxLogit]).range([0, logitW]);
      var xWeight = d3.scaleLinear().domain([0, 100]).range([0, weightW]);

      d3.select(container).selectAll("*").remove();
      var svg = d3.select(container).append("svg")
        .attr("viewBox", "0 0 " + SVG_W + " " + SVG_H)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("role", "img")
        .attr("aria-label", "Logits and softmax weights for three options at d_k " + dk +
          (scaled ? " with" : " without") + " scaling.");

      // section titles
      svg.append("text").attr("class", "d3-axis-title").attr("x", logitX0).attr("y", 18).text("score (logit)");
      svg.append("text").attr("class", "d3-axis-title").attr("x", weightX0).attr("y", 18).text("softmax weight");

      foods.forEach(function (food, i) {
        var cy = margin.top + i * rowH;
        // food label
        svg.append("text").attr("class", "d3-axis-label").attr("x", margin.left - 6).attr("y", cy + barH / 2)
          .attr("text-anchor", "end").attr("dominant-baseline", "central").style("font-size", "13px")
          .text(food.emo + " " + food.name);

        // logit bar track + fill
        svg.append("rect").attr("x", logitX0).attr("y", cy).attr("width", logitW).attr("height", barH).attr("rx", 6).attr("fill", "#eef1f7");
        var lbar = svg.append("rect").attr("x", logitX0).attr("y", cy).attr("width", 0).attr("height", barH).attr("rx", 6).attr("fill", food.color).attr("opacity", 0.85);
        lbar.transition().duration(dur(550)).attr("width", Math.max(0, xLogit(logits[i])));
        svg.append("text").attr("class", "bar-val").attr("x", logitX0 + logitW + 6).attr("y", cy + barH / 2)
          .attr("dominant-baseline", "central").text(logits[i].toFixed(2));

        // weight bar track + fill
        svg.append("rect").attr("x", weightX0).attr("y", cy).attr("width", weightW).attr("height", barH).attr("rx", 6).attr("fill", "#eef1f7");
        var wbar = svg.append("rect").attr("x", weightX0).attr("y", cy).attr("width", 0).attr("height", barH).attr("rx", 6).attr("fill", food.color);
        wbar.transition().duration(dur(550)).attr("width", xWeight(weights[i]));
        svg.append("text").attr("class", "bar-val").attr("x", weightX0 + weightW + 6).attr("y", cy + barH / 2)
          .attr("dominant-baseline", "central").text(weights[i].toFixed(1) + "%");
      });

      // toggle UI mirror
      if (toggle) toggle.classList.toggle("on", scaled);
      if (labelOff) labelOff.classList.toggle("active", !scaled);
      if (labelOn) labelOn.classList.toggle("active", scaled);

      if (caption) {
        if (dk === 1) {
          caption.textContent = "At d_k = 1, √d_k = 1, so scaling does nothing — the two views match. Scaling only matters once d_k grows.";
        } else if (!scaled) {
          caption.textContent = "Scaling OFF, d_k = " + dk + ": the raw dot products are about √" + dk + " = " + Math.sqrt(dk).toFixed(1) +
            "× larger (that is just how dot-product size grows with dimension). The biggest score crushes the rest — softmax collapses to a spike and gradients to the others vanish.";
        } else {
          caption.textContent = "Scaling ON: dividing by √d_k cancels that √d_k growth, keeping logits in a gentle range no matter how big d_k is, so the model can still blend several options.";
        }
      }
    }

    if (toggle) toggle.addEventListener("click", function () { scaled = !scaled; render(); });
    render();
  }

  /* =======================================================================
     VISUALIZATION E — KV-cache simulator  (#kv-grid)
     Training: Q, K, V are full N×d and the whole N×N grid is one matmul.
     Inference: Q is a single new row; each "Generate" APPENDS one K and one V
     row (animated in), and a live readout converts the growing cache to a real
     32-layer / d=4096 model estimate — making "decode is bandwidth-bound" land.
     ===================================================================== */
  function initD3KV() {
    if (!window.d3) { console.warn("d3 not loaded — KV cache skipped"); return; }
    var grid = document.getElementById("kv-grid");
    if (!grid) return;
    var d3 = window.d3;

    var trainBtn = document.getElementById("kv-train");
    var inferBtn = document.getElementById("kv-infer");
    var genBtn = document.getElementById("kv-generate");
    var resetBtn = document.getElementById("kv-reset");
    var caption = document.getElementById("kv-caption");

    var D = 4, PROMPT = 3, MAXGEN = 8;
    var mode = "train", cache = PROMPT;

    // a real-model KV estimate so the abstract grid maps to a felt number
    var PER_TOKEN = 2 * 32 * 4096 * 2; // K&V × 32 layers × d_model 4096 × fp16

    // a live size readout, created once, right after the grid
    var readout = document.getElementById("kv-size-readout");
    if (!readout) {
      readout = document.createElement("div");
      readout.id = "kv-size-readout";
      readout.className = "kv-size-readout";
      grid.parentNode.insertBefore(readout, grid.nextSibling);
    }

    var cw = 26, ch = 20, cgap = 3;        // small cell size
    var colW = D * (cw + cgap);
    var SVG_W = 480, colGap = 56;
    var startX = (SVG_W - (3 * colW + 2 * colGap)) / 2;

    function colX(i) { return startX + i * (colW + colGap); }

    function fillFor(role) { return role === "q" ? "#cfe0ff" : role === "k" ? "#e1d4ff" : "#ffd9be"; }

    function draw(newest) {
      var qRows = mode === "train" ? 4 : 1;
      var kvRows = mode === "train" ? 4 : cache;
      var maxRows = Math.max(qRows, kvRows);
      var SVG_H = 34 + maxRows * (ch + cgap) + 8;

      d3.select(grid).selectAll("*").remove();
      var svg = d3.select(grid).append("svg")
        .attr("viewBox", "0 0 " + SVG_W + " " + SVG_H)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("role", "img")
        .attr("aria-label", mode === "train"
          ? "Training: Q, K and V are full 4 by " + D + " matrices."
          : "Inference: Q is one new row; K and V caches hold " + cache + " rows.");

      var cols = [
        { role: "q", rows: qRows, title: mode === "train" ? "Q · " + qRows + "×" + D : "Q · 1×" + D + " (new token)" },
        { role: "k", rows: kvRows, title: mode === "train" ? "K · " + kvRows + "×" + D : "K cache · " + kvRows + "×" + D },
        { role: "v", rows: kvRows, title: mode === "train" ? "V · " + kvRows + "×" + D : "V cache · " + kvRows + "×" + D },
      ];

      cols.forEach(function (colDef, ci) {
        var cx = colX(ci);
        svg.append("text").attr("class", "d3-axis-title").attr("x", cx + colW / 2).attr("y", 14)
          .attr("text-anchor", "middle").text(colDef.title);
        var cellData = [];
        for (var r = 0; r < colDef.rows; r++)
          for (var c = 0; c < D; c++)
            cellData.push({ r: r, c: c, role: colDef.role, isNew: !!newest && (colDef.role !== "q") && r === colDef.rows - 1 });

        // selectAll(null) (NOT ".kv-rect") is intentional: this runs once per
        // column inside the forEach, so a class selector would re-select the
        // previous column's rects and rebind them. The svg is cleared each draw.
        var rects = svg.selectAll(null).data(cellData).join("rect")
          .attr("class", "kv-rect")
          .attr("x", function (d) { return cx + d.c * (cw + cgap); })
          .attr("y", function (d) { return 26 + d.r * (ch + cgap); })
          .attr("width", cw).attr("height", ch).attr("rx", 5)
          .attr("fill", fillFor(colDef.role))
          .attr("stroke", function (d) { return d.isNew ? C.accent : C.line; })
          .attr("stroke-width", function (d) { return d.isNew ? 2 : 1; });

        // animate the freshly appended row in
        if (!REDUCED) {
          rects.filter(function (d) { return d.isNew; })
            .attr("opacity", 0).attr("transform", "translate(0,10)")
            .transition().duration(dur(360)).attr("opacity", 1).attr("transform", "translate(0,0)");
        }
      });

      // captions + readout
      if (mode === "train") {
        if (caption) caption.textContent = "Training: all N tokens are known. Q, K, V are full N×d matrices and the whole N×N attention grid is computed in parallel — one big matmul. Compute-bound.";
        readout.innerHTML = "No cache during training — the full N×N attention is one parallel matmul. Switch to <b>Inference</b> to watch the KV cache grow one row at a time.";
        if (genBtn) genBtn.disabled = true;
        if (resetBtn) resetBtn.disabled = true;
      } else {
        var stepN = cache - PROMPT;
        if (caption) caption.textContent = "Decode step " + stepN + ": Q is a single new row. We append one K and one V row to the cache (now " + cache +
          ") — past keys/values are re-read, never recomputed. You must load the whole cache from HBM each step → memory-bandwidth bound.";
        readout.innerHTML = "Cache holds <b>" + cache + "</b> token-rows. In a real 32-layer, d=4096 fp16 model that is ≈ <b>" + fmtBytes(PER_TOKEN * cache) +
          "</b> now, growing to ≈ <b>" + fmtBytes(PER_TOKEN * 8192) + "</b> at 8k tokens. Every decode step re-reads the whole cache from HBM.";
        if (genBtn) genBtn.disabled = cache >= PROMPT + MAXGEN;
        if (resetBtn) resetBtn.disabled = false;
      }
    }

    if (trainBtn) trainBtn.addEventListener("click", function () {
      mode = "train"; trainBtn.classList.add("btn-primary"); if (inferBtn) inferBtn.classList.remove("btn-primary"); draw();
    });
    if (inferBtn) inferBtn.addEventListener("click", function () {
      mode = "infer"; cache = PROMPT; inferBtn.classList.add("btn-primary"); if (trainBtn) trainBtn.classList.remove("btn-primary"); draw();
    });
    if (genBtn) genBtn.addEventListener("click", function () { if (cache < PROMPT + MAXGEN) { cache++; draw(true); } });
    if (resetBtn) resetBtn.addEventListener("click", function () { cache = PROMPT; draw(); });
    draw();
  }

  /* ---------- boot (independent of main.js's own DOMContentLoaded) ---------- */
  document.addEventListener("DOMContentLoaded", function () {
    initD3Heatmap();
    initD3CostChart();
    initD3Pipeline();
    initD3Scaling();
    initD3KV();
  });
})();
