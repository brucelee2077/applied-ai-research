# Batch 3 — naming note (disambiguation)

_Two files in `sessions/` share the `batch_3_` prefix. They are unrelated passes over different modules, with **no overlap**. This note exists so nobody confuses them._

---

| File(s) | Pass type | Scope (modules) |
|---------|-----------|-----------------|
| [`batch_3_rollout_report.md`](./batch_3_rollout_report.md) &nbsp;·&nbsp; [`batch_3_rollout_plan.md`](./batch_3_rollout_plan.md) | **Coach Layer rollout** — added pain-points, Math Ladders, 🎤 interview blocks, bilingual scaffold, explain-back artifacts, and 3 interactive P0 labs (all additive) | **JAX & systems tier:** `m07-thinking-in-jax`, `m08-transformer-math`, `m09a-hardware-physics`, `m09c-sharding-parallelism` |
| [`batch_3_cleanup_report.md`](./batch_3_cleanup_report.md) | **Defect-only cleanup** (older) — no Coach Layer, no new visuals; fixed a title-spacing glitch and scanned for duplicate callouts / stale run commands | **Inference-systems tier:** `m06-cnns-vision-encoders`, `m16a-inference-economics`, `m17a-quantization`, `m17b-long-context-decoding` |

---

## Why they share a number

Each was the **third pass of its own workstream** — one the Coach Layer rollout, the other the defect cleanup — so both landed on "Batch 3" independently. They were **not** coordinated to the same batch number on purpose, and they touch **disjoint** module sets. There is no shared file and no shared module.

**In one line:** `batch_3_rollout_*` = Coach Layer for `m07`/`m08`/`m09a`/`m09c`; `batch_3_cleanup_report.md` = defect cleanup for `m06`/`m16a`/`m17a`/`m17b`.
