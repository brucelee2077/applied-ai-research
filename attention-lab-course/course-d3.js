/* =========================================================================
   Attention, Visualized — course-d3.js
   A single self-contained vanilla-JS file (D3 v7) that draws two
   visualizations into containers placed by the course modules.

   D3 is already loaded (vendored locally as d3.v7.min.js, included before
   this file via <script defer>). This file does not depend on styles.css —
   it injects its own small <style> block at runtime.

   - VIZ A: initCourseHeatmap()  → into #attn-heatmap
   - VIZ B: initCourseCostChart() → into #attn-cost-chart (reads its own
            controls in module 5: #attn-n, #attn-d, #attn-flash)

   Fully defensive: if D3 did not load, or a target container is not on the
   page, the matching init simply returns and the rest of the page keeps
   working untouched.
   ========================================================================= */
(function () {
  "use strict";

  // The course teal accent — matches the --color-accent in _base.html.
  var ACCENT = "#2A7B9B";

  /* ---------- injected styles (do NOT rely on the course styles.css) ------- */
  function injectStyles() {
    // Only inject once, even if this runs again.
    if (document.getElementById("course-d3-style")) return;
    var css =
      /* the two container divs: full width, centered, capped width */
      "#attn-heatmap, #attn-cost-chart {" +
      "  width: 100%; max-width: 720px; margin: 16px auto; display: block;" +
      "}" +
      "#attn-heatmap svg, #attn-cost-chart svg {" +
      "  width: 100%; height: auto; display: block; overflow: visible;" +
      "}" +
      /* axis / label text: muted gray, inherited system font */
      ".course-d3-axis-label, .course-d3-axis-title {" +
      "  fill: #5b6573;" +
      "  font-family: inherit, system-ui, -apple-system, 'Segoe UI', sans-serif;" +
      "}" +
      ".course-d3-axis-label { font-size: 12px; }" +
      ".course-d3-axis-title { font-size: 12px; font-weight: 600; }" +
      ".course-d3-axis path, .course-d3-axis line { stroke: #cdd3dc; }" +
      ".course-d3-axis text { fill: #5b6573; font-family: inherit; font-size: 11px; }" +
      /* row label: clickable, turns teal + bold when active */
      ".course-d3-row-label { transition: fill 120ms ease; }" +
      ".course-d3-row-label.row-active { fill: " + ACCENT + "; font-weight: 700; }" +
      /* cells: an active row gets a teal outline */
      ".course-d3-cell { transition: stroke 120ms ease; }" +
      ".course-d3-cell.row-active { stroke: " + ACCENT + "; stroke-width: 2; }" +
      /* tooltip: fixed, white, light border, rounded, hidden by default */
      ".course-d3-tooltip {" +
      "  position: fixed; top: 0; left: 0;" +
      "  background: #ffffff; border: 1px solid #e2e6ec; border-radius: 6px;" +
      "  padding: 6px 9px; font-size: 12px; line-height: 1.45;" +
      "  font-family: inherit, system-ui, sans-serif; color: #1f2735;" +
      "  box-shadow: 0 4px 14px rgba(20,30,50,0.12);" +
      "  pointer-events: none; opacity: 0; transition: opacity 100ms ease;" +
      "  z-index: 9999; max-width: 260px;" +
      "}" +
      /* the controls block in module 5 */
      "#attn-cost-controls {" +
      "  display: flex; flex-direction: column; gap: 12px;" +
      "  max-width: 720px; margin: 16px auto;" +
      "  font-family: inherit, system-ui, sans-serif;" +
      "}" +
      "#attn-cost-controls label {" +
      "  display: block; font-size: 13px; color: #5b6573; margin-bottom: 4px;" +
      "}" +
      "#attn-cost-controls input[type='range'] {" +
      "  width: 100%; accent-color: " + ACCENT + ";" +
      "}" +
      "#attn-cost-controls input[type='checkbox'] { accent-color: " + ACCENT + "; }" +
      "#attn-cost-controls output { font-weight: 600; color: #1f2735; }";

    var style = document.createElement("style");
    style.id = "course-d3-style";
    style.textContent = css;
    document.head.appendChild(style);
  }

  /* ---------- shared helpers ---------------------------------------------- */

  // fmtBytes mirrors main.js (visual-tutorial) lines 22-27 exactly.
  function fmtBytes(b) {
    if (b >= 1e9) return (b / 1e9).toFixed(2) + " GB";
    if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
    if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
    return b + " B";
  }

  // One shared, body-level tooltip div reused by both visualizations.
  // Created lazily so it only exists if D3 actually runs.
  var _tooltip = null;
  function getTooltip() {
    if (_tooltip) return _tooltip;
    _tooltip = document.createElement("div");
    _tooltip.className = "course-d3-tooltip";
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
    // Offset from the cursor; keep inside the viewport on the right edge.
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
     VIZ A — true 2-D attention heatmap  (#attn-heatmap)
     Rows = the QUERY word (doing the looking), columns = the KEY word
     (being looked at), cell fill = attention weight in [0,1].
     ===================================================================== */
  function initCourseHeatmap() {
    if (!window.d3) { console.warn("d3 missing"); return; }
    if (!document.getElementById("attn-heatmap")) return;

    var d3 = window.d3;
    var container = document.getElementById("attn-heatmap");

    // --- data: copied VERBATIM from visual-tutorial/main.js initHeatmap ---
    var words = ["The", "cat", "sat", "on", "the", "mat", "because", "it", "was", "tired"];
    // Each ROW is one query word's attention over all 10 words. "it" (idx 7)
    // points mostly at "cat" (idx 1).
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
    var caption = document.getElementById("attn-heatmap-caption");
    var defaultCaption =
      "Hover any cell, or click a row label, to see who that word listens to.";

    // --- dimensions and margins ---
    // Generous left margin for row labels, generous top for rotated column
    // labels, and a strip on the right for the color legend.
    var margin = { top: 70, right: 78, bottom: 16, left: 88 };
    var gridSize = 360;
    var width = gridSize + margin.left + margin.right;
    var height = gridSize + margin.top + margin.bottom;

    // --- scales ---
    var x = d3.scaleBand().domain(d3.range(n)).range([0, gridSize]).padding(0.06);
    var y = d3.scaleBand().domain(d3.range(n)).range([0, gridSize]).padding(0.06);
    // Sequential blues, weight in [0,1] — the canonical attention color.
    var color = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

    function render() {
      // clear previous render before redraw
      d3.select(container).selectAll("*").remove();

      var svg = d3
        .select(container)
        .append("svg")
        .attr("viewBox", "0 0 " + width + " " + height)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("role", "img")
        .attr(
          "aria-label",
          "Attention heatmap. Rows are query words doing the looking; " +
            "columns are key words being looked at; darker blue means more attention."
        );

      svg
        .append("title")
        .text(
          "A 10 by 10 attention matrix for the sentence: " +
            words.join(" ") +
            ". Each row is one word's attention over all words."
        );

      var g = svg
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      // --- column labels (KEY word — being looked at), across the top ---
      g.selectAll(".col-label")
        .data(words)
        .join("text")
        .attr("class", "course-d3-axis-label col-label")
        .attr("x", function (d, i) { return x(i) + x.bandwidth() / 2; })
        .attr("y", -10)
        .attr("text-anchor", "start")
        .attr("transform", function (d, i) {
          var cx = x(i) + x.bandwidth() / 2;
          return "rotate(-45," + cx + ",-10)";
        })
        .text(function (d) { return d; });

      // small "key →" hint above the columns
      g.append("text")
        .attr("class", "course-d3-axis-title")
        .attr("x", gridSize / 2)
        .attr("y", -margin.top + 16)
        .attr("text-anchor", "middle")
        .text("key word (being looked at) →");

      // "query ↓" hint on the left, rotated
      g.append("text")
        .attr("class", "course-d3-axis-title")
        .attr("text-anchor", "middle")
        .attr(
          "transform",
          "translate(" + (-margin.left + 16) + "," + gridSize / 2 + ") rotate(-90)"
        )
        .text("query word (doing the looking) ↓");

      // --- row labels (QUERY word — doing the looking), clickable ---
      g.selectAll(".row-label")
        .data(words)
        .join("text")
        .attr("class", "course-d3-axis-label course-d3-row-label row-label")
        .attr("data-row", function (d, i) { return i; })
        .attr("x", -10)
        .attr("y", function (d, i) { return y(i) + y.bandwidth() / 2; })
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "central")
        .style("cursor", "pointer")
        .text(function (d) { return d; })
        .on("mouseenter", function (event, d) { highlightRow(words.indexOf(d)); })
        .on("click", function (event, d) { highlightRow(words.indexOf(d)); });

      // --- the cells: join one rect per (row, col) pair ---
      var cells = [];
      for (var r = 0; r < n; r++) {
        for (var c = 0; c < n; c++) {
          cells.push({ row: r, col: c, weight: A[r][c] });
        }
      }

      g.selectAll(".course-d3-cell")
        .data(cells)
        .join("rect")
        .attr("class", "course-d3-cell")
        .attr("data-row", function (d) { return d.row; })
        .attr("x", function (d) { return x(d.col); })
        .attr("y", function (d) { return y(d.row); })
        .attr("width", x.bandwidth())
        .attr("height", y.bandwidth())
        .attr("rx", 3)
        .attr("fill", function (d) {
          // weight 0 stays nearly white so the matrix reads clean
          return d.weight > 0 ? color(d.weight) : "#fbfcfe";
        })
        .attr("stroke", "#e6e9f2")
        .attr("stroke-width", 1)
        .on("mouseenter", function (event, d) {
          var pct = Math.round(d.weight * 100);
          showTooltip(
            "<b>‘" + words[d.row] + "’</b> attends <b>" + pct +
              "%</b> to <b>‘" + words[d.col] + "’</b>",
            event
          );
          highlightRow(d.row);
        })
        .on("mousemove", function (event) { moveTooltip(event); })
        .on("mouseleave", function () { hideTooltip(); });

      // --- color legend (small vertical gradient bar on the right) ---
      var legendH = 140;
      var legendW = 12;
      var legendX = gridSize + 24;
      var legendTop = 6;

      var defs = svg.append("defs");
      var grad = defs
        .append("linearGradient")
        .attr("id", "course-d3-heatmap-grad")
        .attr("x1", "0%").attr("y1", "100%")
        .attr("x2", "0%").attr("y2", "0%");
      d3.range(0, 1.01, 0.1).forEach(function (t) {
        grad
          .append("stop")
          .attr("offset", t * 100 + "%")
          .attr("stop-color", color(t));
      });

      var legend = g
        .append("g")
        .attr("transform", "translate(" + legendX + "," + legendTop + ")");
      legend
        .append("rect")
        .attr("width", legendW)
        .attr("height", legendH)
        .attr("rx", 3)
        .attr("fill", "url(#course-d3-heatmap-grad)")
        .attr("stroke", "#e6e9f2");
      legend
        .append("text")
        .attr("class", "course-d3-axis-label")
        .attr("x", legendW + 6)
        .attr("y", 4)
        .attr("dominant-baseline", "hanging")
        .text("100%");
      legend
        .append("text")
        .attr("class", "course-d3-axis-label")
        .attr("x", legendW + 6)
        .attr("y", legendH)
        .attr("dominant-baseline", "baseline")
        .text("0%");
      legend
        .append("text")
        .attr("class", "course-d3-axis-label")
        .attr("x", legendW / 2)
        .attr("y", legendH + 16)
        .attr("text-anchor", "middle")
        .text("weight");
    }

    // Highlight a whole query row: outline its cells, bold its label (teal),
    // and write that row's caption into the caption element.
    function highlightRow(idx) {
      var g = d3.select(container).select("svg g");
      g.selectAll(".course-d3-cell")
        .classed("row-active", function (d) { return d.row === idx; });
      g.selectAll(".course-d3-row-label")
        .classed("row-active", function (d, i) { return i === idx; });
      if (caption) caption.textContent = caps[idx];
    }

    render();
    // Set a gentle default caption only if the module did not supply one.
    if (caption && !caption.textContent.trim()) {
      caption.textContent = defaultCaption;
    }
  }

  /* =======================================================================
     VIZ B — memory growth chart  (#attn-cost-chart)
     Self-contained: reads its own controls in module 5 and redraws on their
     events. Plots memory in BYTES (log y) vs sequence length n (log x),
     three curves using the SAME formulas main.js uses.
     ===================================================================== */
  function initCourseCostChart() {
    if (!window.d3) { console.warn("d3 missing"); return; }
    if (!document.getElementById("attn-cost-chart")) return;

    var d3 = window.d3;
    var container = document.getElementById("attn-cost-chart");

    // controls that live in module 5's HTML
    var nEl = document.getElementById("attn-n");
    var dEl = document.getElementById("attn-d");
    var flashEl = document.getElementById("attn-flash");
    if (!nEl || !dEl) return;

    var nOut = document.getElementById("attn-n-out");
    var dOut = document.getElementById("attn-d-out");
    var caption = document.getElementById("attn-cost-caption");

    // --- memory formulas — identical to main.js initCost (lines 267-269) ---
    var bytesTokens = function (n, d) { return n * d * 2; };   // n×d token tensor
    var bytesScores = function (n) { return n * n * 2; };      // n×n attention scores
    var bytesFlash = function (n) { return n * 64 * 2; };      // FlashAttention ~O(n)

    // x sweep from 128 to 131072
    var N_MIN = 128, N_MAX = 131072;
    var nSamples = d3.range(0, 81).map(function (i) {
      // log-spaced points so the lines are smooth on a log x axis
      return Math.round(N_MIN * Math.pow(N_MAX / N_MIN, i / 80));
    });

    // --- dimensions and margins ---
    var margin = { top: 24, right: 140, bottom: 48, left: 70 };
    var width = 720;
    var height = 380;
    var innerW = width - margin.left - margin.right;
    var innerH = height - margin.top - margin.bottom;

    // colors: blue (token, O(n)), red (n×n), teal accent (flash)
    var C_TOKEN = "#2563eb";
    var C_SCORES = "#ef4444";
    var C_FLASH = ACCENT;

    function readControls() {
      return {
        n: +nEl.value,
        d: +dEl.value,
        flash: !!(flashEl && flashEl.checked),
      };
    }

    function updateOutputs(ctrl) {
      if (nOut) nOut.textContent = ctrl.n.toLocaleString();
      if (dOut) dOut.textContent = ctrl.d.toLocaleString();
    }

    function updateCaption(ctrl) {
      if (!caption) return;
      var curN = ctrl.n, d = ctrl.d, flash = ctrl.flash;
      if (flash) {
        caption.textContent =
          "FlashAttention is on: the n×n matrix is never built. Attention " +
          "memory ≈ " + fmtBytes(bytesFlash(curN)) + " (O(n)). Compute is " +
          "still O(n²) — same FLOPs, far less memory.";
      } else {
        var ratio = bytesScores(curN) / bytesTokens(curN, d); // = n / d
        if (ratio < 1) {
          caption.textContent =
            "At n = " + curN.toLocaleString() + ", the token tensor still " +
            "dominates — the n×n matrix is only " + ratio.toFixed(2) +
            "× the token tensor. Attention memory is not yet the wall.";
        } else {
          caption.textContent =
            "At n = " + curN.toLocaleString() + ", the n×n score matrix is " +
            Math.round(ratio).toLocaleString() + "× the token tensor — " +
            "attention memory now dominates. This is the long-context wall.";
        }
      }
    }

    function render() {
      var ctrl = readControls();
      var d = ctrl.d;
      var curN = ctrl.n;
      var flash = ctrl.flash;

      updateOutputs(ctrl);

      // build the three series at the current d
      var seriesTokens = nSamples.map(function (nn) { return { n: nn, bytes: bytesTokens(nn, d) }; });
      var seriesScores = nSamples.map(function (nn) { return { n: nn, bytes: bytesScores(nn) }; });
      var seriesFlash = nSamples.map(function (nn) { return { n: nn, bytes: bytesFlash(nn) }; });

      // clear previous render before redraw
      d3.select(container).selectAll("*").remove();

      var svg = d3
        .select(container)
        .append("svg")
        .attr("viewBox", "0 0 " + width + " " + height)
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("role", "img")
        .attr(
          "aria-label",
          "Memory growth chart. Memory in bytes on a log scale versus sequence " +
            "length n on a log scale, for the token tensor, the n by n attention " +
            "scores, and FlashAttention."
        );

      svg
        .append("title")
        .text(
          "The n by n attention-score line crosses above the n by d token-tensor " +
            "line and then explodes — the long-context memory wall."
        );

      var g = svg
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      // --- scales (log x, log y) ---
      var x = d3.scaleLog().domain([N_MIN, N_MAX]).range([0, innerW]);

      var allBytes = seriesTokens
        .concat(seriesScores)
        .concat(seriesFlash)
        .map(function (p) { return p.bytes; });
      var yMin = d3.min(allBytes);
      var yMax = d3.max(allBytes);
      var y = d3
        .scaleLog()
        .domain([Math.max(1, yMin), yMax])
        .range([innerH, 0])
        .nice();

      // --- axes ---
      var xAxis = d3
        .axisBottom(x)
        .tickValues([128, 512, 2048, 8192, 32768, 131072])
        .tickFormat(function (v) {
          return v >= 1024 ? v / 1024 + "k" : "" + v;
        });
      var yAxis = d3
        .axisLeft(y)
        .ticks(6, function (v) { return fmtBytes(v); });

      g.append("g")
        .attr("class", "course-d3-axis")
        .attr("transform", "translate(0," + innerH + ")")
        .call(xAxis);
      g.append("g").attr("class", "course-d3-axis").call(yAxis);

      // axis titles
      g.append("text")
        .attr("class", "course-d3-axis-title")
        .attr("x", innerW / 2)
        .attr("y", innerH + 40)
        .attr("text-anchor", "middle")
        .text("sequence length n (log scale)");
      g.append("text")
        .attr("class", "course-d3-axis-title")
        .attr("text-anchor", "middle")
        .attr("transform", "translate(" + (-margin.left + 16) + "," + innerH / 2 + ") rotate(-90)")
        .text("memory, bytes (log scale)");

      // --- line generator ---
      var line = d3
        .line()
        .x(function (p) { return x(p.n); })
        .y(function (p) { return y(p.bytes); });

      // n×d token tensor
      g.append("path")
        .attr("class", "course-d3-line")
        .attr("d", line(seriesTokens))
        .attr("fill", "none")
        .attr("stroke", C_TOKEN)
        .attr("stroke-width", 2.5);

      // n×n attention scores — faded + dashed when flash is on
      g.append("path")
        .attr("class", "course-d3-line")
        .attr("d", line(seriesScores))
        .attr("fill", "none")
        .attr("stroke", C_SCORES)
        .attr("stroke-width", flash ? 1.8 : 2.8)
        .attr("stroke-dasharray", flash ? "6 5" : null)
        .attr("opacity", flash ? 0.35 : 1);

      // FlashAttention — emphasized (solid teal) when flash is on
      g.append("path")
        .attr("class", "course-d3-line")
        .attr("d", line(seriesFlash))
        .attr("fill", "none")
        .attr("stroke", C_FLASH)
        .attr("stroke-width", flash ? 3 : 1.6)
        .attr("stroke-dasharray", flash ? null : "3 4")
        .attr("opacity", flash ? 1 : 0.5);

      // --- vertical dashed guide at the current n ---
      var clampedN = Math.max(N_MIN, Math.min(N_MAX, curN));
      g.append("line")
        .attr("class", "course-d3-guide")
        .attr("x1", x(clampedN))
        .attr("x2", x(clampedN))
        .attr("y1", 0)
        .attr("y2", innerH)
        .attr("stroke", ACCENT)
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "4 4")
        .attr("opacity", 0.7);
      g.append("text")
        .attr("class", "course-d3-axis-label")
        .attr("x", x(clampedN) + 4)
        .attr("y", 10)
        .attr("fill", ACCENT)
        .text("n = " + curN.toLocaleString());

      // --- a dot on each curve at the current n ---
      var dotData = [
        { bytes: bytesTokens(clampedN, d), color: C_TOKEN, label: "token" },
        { bytes: bytesScores(clampedN), color: C_SCORES, label: "scores" },
        { bytes: bytesFlash(clampedN), color: C_FLASH, label: "flash" },
      ];
      g.selectAll(".course-d3-dot")
        .data(dotData)
        .join("circle")
        .attr("class", "course-d3-dot")
        .attr("cx", x(clampedN))
        .attr("cy", function (p) { return y(p.bytes); })
        .attr("r", 4.5)
        .attr("fill", function (p) { return p.color; })
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5);

      // --- legend (right strip) ---
      var legend = g
        .append("g")
        .attr("transform", "translate(" + (innerW + 18) + ",6)");
      var legendItems = [
        { color: C_TOKEN, text: "n×d token · O(n)", faded: false },
        { color: C_SCORES, text: "n×n scores · O(n²)", faded: flash },
        { color: C_FLASH, text: "FlashAttn · O(n)", faded: !flash },
      ];
      legendItems.forEach(function (item, i) {
        var row = legend
          .append("g")
          .attr("transform", "translate(0," + i * 22 + ")")
          .attr("opacity", item.faded ? 0.45 : 1);
        row
          .append("line")
          .attr("x1", 0).attr("x2", 22)
          .attr("y1", 6).attr("y2", 6)
          .attr("stroke", item.color)
          .attr("stroke-width", 3);
        row
          .append("text")
          .attr("class", "course-d3-axis-label")
          .attr("x", 28)
          .attr("y", 10)
          .text(item.text);
      });

      // --- invisible hover overlay: bisect to nearest sampled n ---
      var bisect = d3.bisector(function (p) { return p.n; }).left;
      g.append("rect")
        .attr("class", "course-d3-hover-rect")
        .attr("width", innerW)
        .attr("height", innerH)
        .attr("fill", "transparent")
        .on("mousemove", function (event) {
          var mx = d3.pointer(event, this)[0];
          var nVal = x.invert(mx);
          var i = bisect(seriesTokens, nVal);
          if (i >= seriesTokens.length) i = seriesTokens.length - 1;
          if (i < 0) i = 0;
          var nn = seriesTokens[i].n;
          showTooltip(
            "<b>n = " + nn.toLocaleString() + "</b><br>" +
              "<span style='color:" + C_TOKEN + "'>■</span> token n×d: " + fmtBytes(bytesTokens(nn, d)) + "<br>" +
              "<span style='color:" + C_SCORES + "'>■</span> scores n×n: " + fmtBytes(bytesScores(nn)) + "<br>" +
              "<span style='color:" + C_FLASH + "'>■</span> flash: " + fmtBytes(bytesFlash(nn)),
            event
          );
        })
        .on("mouseleave", function () { hideTooltip(); });

      // keep the one-line takeaway in sync with the current slider state
      updateCaption(ctrl);
    }

    // Redraw on this module's own controls.
    nEl.addEventListener("input", render);
    nEl.addEventListener("change", render);
    dEl.addEventListener("input", render);
    dEl.addEventListener("change", render);
    if (flashEl) {
      flashEl.addEventListener("change", render);
      flashEl.addEventListener("input", render);
    }

    render();
  }

  /* ---------- boot (own DOMContentLoaded) --------------------------------- */
  document.addEventListener("DOMContentLoaded", function () {
    injectStyles();
    initCourseHeatmap();
    initCourseCostChart();
  });
})();
