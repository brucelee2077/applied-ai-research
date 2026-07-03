/* =========================================================================
   viz.js — D3 scorecard visualizations (vendored D3, no CDN).
   Exposes window.__INTERVIEW_VIZ and sets window.__INTERVIEW_D3 when D3 loaded,
   mirroring the repo's __ATTN_D3 readiness handshake. app.js degrades to text
   if D3 is unavailable.
   ========================================================================= */
(function () {
  "use strict";
  if (!window.d3) {
    window.__INTERVIEW_D3 = null;
    return;
  }
  const d3 = window.d3;

  // Score -> rubric band (matches backend score_to_band thresholds).
  function band(score) {
    if (score >= 0.85) return "strong";
    if (score >= 0.65) return "hire";
    if (score >= 0.4) return "weak";
    return "no";
  }
  const LEVEL_BAND = { no_hire: "no", weak_hire: "weak", hire: "hire", strong_hire: "strong" };

  function truncate(s, n) {
    s = (s || "").replace(/\s*\(\d+\s*min\)\s*$/, "");
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  }

  /* ---------- horizontal per-question bar chart ---------- */
  function renderBar(svgEl, perQuestion) {
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    const data = perQuestion || [];
    if (!data.length) {
      svg.append("text").attr("x", 210).attr("y", 120).attr("text-anchor", "middle")
        .attr("fill", "#64708a").attr("font-size", 13).text("No questions graded.");
      return;
    }
    const W = 420, rowH = Math.min(34, Math.max(20, 220 / data.length));
    const H = data.length * rowH + 24;
    svg.attr("viewBox", `0 0 ${W} ${H}`);
    const labelW = 150, barX = labelW + 6, barMax = W - barX - 44;

    const rows = svg.selectAll("g.row").data(data).enter().append("g")
      .attr("class", "row").attr("transform", (d, i) => `translate(0,${i * rowH + 12})`);

    rows.append("text").attr("x", labelW).attr("y", rowH / 2).attr("dy", ".35em")
      .attr("text-anchor", "end").attr("font-size", 11).attr("fill", "#37425a")
      .text((d, i) => truncate(d.section_title || `Q${i + 1}`, 22));

    rows.append("rect").attr("x", barX).attr("y", rowH * 0.18).attr("height", rowH * 0.62)
      .attr("rx", 4).attr("width", barMax).attr("fill", "#eef0f5");

    rows.append("rect").attr("x", barX).attr("y", rowH * 0.18).attr("height", rowH * 0.62)
      .attr("rx", 4).attr("class", (d) => "bar-" + (LEVEL_BAND[d.level] || band(d.score_pct)))
      .attr("width", 0)
      .transition().duration(700).delay((d, i) => i * 60)
      .attr("width", (d) => Math.max(2, d.score_pct * barMax));

    rows.append("text").attr("x", W - 4).attr("y", rowH / 2).attr("dy", ".35em")
      .attr("text-anchor", "end").attr("font-size", 11).attr("font-weight", 700)
      .attr("fill", "#37425a").text((d) => Math.round(d.score_pct * 100) + "%");
  }

  /* ---------- criteria coverage radar ---------- */
  function renderRadar(svgEl, coverage) {
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    const entries = Object.entries(coverage || {});
    if (entries.length < 3) {
      // Radar needs >=3 axes to be meaningful.
      return false;
    }
    // Cap axes so labels stay legible.
    const axes = entries.slice(0, 10);
    const W = 420, H = 360, cx = W / 2, cy = H / 2 + 6, R = 120;
    svg.attr("viewBox", `0 0 ${W} ${H}`);
    const n = axes.length, ang = (i) => (Math.PI * 2 * i) / n - Math.PI / 2;

    // rings
    [0.25, 0.5, 0.75, 1].forEach((r) => {
      svg.append("circle").attr("cx", cx).attr("cy", cy).attr("r", R * r)
        .attr("fill", "none").attr("stroke", "#e6e9f2");
    });
    // axes + labels
    axes.forEach(([name], i) => {
      const x = cx + R * Math.cos(ang(i)), y = cy + R * Math.sin(ang(i));
      svg.append("line").attr("x1", cx).attr("y1", cy).attr("x2", x).attr("y2", y)
        .attr("stroke", "#e6e9f2");
      const lx = cx + (R + 14) * Math.cos(ang(i)), ly = cy + (R + 14) * Math.sin(ang(i));
      svg.append("text").attr("x", lx).attr("y", ly).attr("dy", ".35em")
        .attr("text-anchor", lx < cx - 4 ? "end" : lx > cx + 4 ? "start" : "middle")
        .attr("font-size", 10).attr("fill", "#64708a").text(truncate(name, 16));
    });
    // polygon
    const pts = axes.map(([, v], i) => {
      const r = R * Math.max(0, Math.min(1, v));
      return [cx + r * Math.cos(ang(i)), cy + r * Math.sin(ang(i))];
    });
    svg.append("polygon").attr("points", pts.map((p) => p.join(",")).join(" "))
      .attr("fill", "rgba(99,102,241,.18)").attr("stroke", "#6366f1").attr("stroke-width", 2);
    pts.forEach((p) => svg.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 3).attr("fill", "#6366f1"));
    return true;
  }

  window.__INTERVIEW_VIZ = { renderBar, renderRadar, band };
  window.__INTERVIEW_D3 = { radar: true, bar: true };
})();
