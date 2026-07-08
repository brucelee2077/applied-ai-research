# Source-First Authoring Loop (v8)

## Purpose

Give the HTML lesson medium the notebook's one good property — **authoring surface = reader surface** — without requiring a notebook.

## The loop (extends the v7.6 Autonomous Rollout Loop)

```text
1.  Read sessions/_refactor/rollout_tracker.yaml → current_target
2.  Read seed manifest(s) + recent reports
3.  Detect seed propagation risk; stabilize seed if required; update seed manifest/tracker
4.  Select target; create/read target manifest
5.  DETERMINE AUTHORING MODE:
        companion notebook exists → Exemplar
        else papers/docs available → Reference (arXiv MCP / doc fetch)
        else → First-Principles
6.  Source-free coverage discovery → coverage contract (+ capability-limit + geometry rows)
7.  AUTHOR source.md per day in reader-flow order (six rules; anchors single-sourced; spine up front)
8.  COMPILE source.md → lesson.html (compiler owns shell/JS/invariants; behavioral blocks name a viz)
9.  GATES on compiled output:
        Reader Flow · Staff Depth · Shell Invariant · Visual/Evidence(behavioral) · Artifact
        + Notebook Smoothness (Exemplar/pilot only)
        + No-Notebook Authoring (Reference / First-Principles)
10. QA: lesson_audit.py, nav_audit.py, staff_lens_audit.js, node --check, jsdom if available
11. Fix P0 only IN source.md, then RECOMPILE (never hand-edit lesson.html) until 0 P0
12. Update manifest + tracker (mode, gate results, counts, report paths)
13. Stop at pass / pass_with_p1, or write a phase-specific blocker report
```

## No-Notebook Rule

A notebook is a reader-flow calibration input, not a build dependency.

For Reference / First-Principles modules the Notebook Smoothness Gate is **skipped (N/A)**, never failed. All other gates run unchanged.

## Compiler contract (summary)

`source.md → lesson.html` must: emit the frozen shell; map 12 reader-flow blocks onto the 7 sections + hero + `.fin`; generate DEMOS/BUILD/QS; enforce playground ≥3 and quiz `q:4 o:16`; wire quest-id/localStorage/nav/`.fin`; expand single-sourced anchors; insert live iframes for behavioral blocks; be deterministic + idempotent.

Full spec: `sessions/_refactor/v8_source_first_authoring_plan.md`.
