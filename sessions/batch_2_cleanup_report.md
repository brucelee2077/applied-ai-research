# Batch 2 — Cleanup Pass Report

_Cleanup-only pass over the Batch 2 Coach Layer rollout (`m03-attention`, `m04-first-model-mlp`, `m05a-text-transformer`). No new visuals, no broad content refactor. Date: 2026-07-07._

Companion docs: [`batch_2_rollout_plan.md`](./batch_2_rollout_plan.md) · [`batch_2_rollout_report.md`](./batch_2_rollout_report.md) (the rollout this cleans up after) · [`lesson_audit.py`](./lesson_audit.py) (the automated gate).

---

## 1. TL;DR

A targeted, additive-safe cleanup of the 19 Batch 2 lessons plus the two rollout `.md` reports. Five defect classes were fixed:

1. **Duplicate trade-off callouts** — the named "coherence vs diversity" dup in `day-07`, **plus 6 more** of the identical copy-paste defect found by an exhaustive scan across the batch.
2. **Stale artifact run commands** — 11 lessons said `Run with python3 <shortname>.py` (e.g. `m05a_sampling.py`) while their own `Create …experiment.py` reference already used the real path.
3. **A title spacing glitch** — `Positional Encoding &the Shuffle Problem`.
4. **Markdown source hygiene** — the two rollout reports had list/heading blank-line lint issues.

Final state: **19/19 OK in `lesson_audit.py`, 0 DEGRADED**; **19/19 jsdom-render clean** with all **7 P0 labs** still interaction-verified; both reports **0 markdownlint issues**. Diff-verified: **no** `data-quest-id` / `data-demo` / `data-sec` / `var QS|DEMOS|BUILD` / nav-`href` / `.gotit` / failure-mode / interview line was removed — every deletion was a stale command, the old `<h1>`, or a byte-identical duplicate.

---

## 2. Task 1 — Duplicate trade-off callouts removed (7 lessons)

The request named one duplicate in `day-07-text-generation` ("coherence vs diversity"). A whole-curriculum exact-duplicate scan showed the **same copy-paste defect in 6 sibling `m05a` lessons**: the `⚖️` Trade-off callout was pasted twice on adjacent, byte-identical lines. Because this is the exact defect class Task 1 targets, is confined to Batch 2, and is pure cleanup (removing an accidental duplicate, no content lost), all 7 were collapsed to a single copy.

| Lesson | Duplicated callout | Action |
|--------|--------------------|--------|
| m05a · day-01 · residual-connections | *Trade-off: trainability now, memory later.* | 2 → 1 |
| m05a · day-02 · layer-norm | *Trade-off: pre-LN stability vs post-LN final quality.* | 2 → 1 |
| m05a · day-04 · full-block | *Trade-off: the block bundles a cheap part with an expensive one.* | 2 → 1 |
| m05a · day-05 · causal-masking | *Trade-off: parallel training speed vs left-to-right-only understanding.* | 2 → 1 |
| m05a · day-06 · encoder-vs-decoder | *Trade-off: bidirectional understanding vs the ability to generate.* | 2 → 1 |
| **m05a · day-07 · text-generation** | *Trade-off: coherence vs diversity, set by one number.* (named in request) | 2 → 1 |
| m05a · day-08 · full-transformer | *Trade-off: weight tying saves parameters but couples two jobs.* | 2 → 1 |

Each duplicate sat directly below a **distinct** `⚠️` failure-mode callout and directly above a **distinct** `🎤` interview callout — only the middle `⚖️` line was doubled, and only that second copy was removed. Post-fix scan: **zero adjacent duplicate lines** anywhere in the 19 lessons; each trade-off callout now appears exactly once.

**Deliberately left untouched (verified false positives):** the whole-file scan also flagged a repeated `🧮` line in `m04/day-05-pytorch-version` and `m05a/day-01-residual-connections`. These are the **opening wrapper line of two distinct multi-line math-ladder boxes** (mono-font `<div style=…>` with different content on the following lines), not duplicated content. Non-adjacent, legitimate — not changed.

---

## 3. Task 2 — Stale artifact run commands fixed (11 lessons)

Each Produce section already opened with the correct `Create <code>sessions/<module>/<day>/experiment.py</code>` path, but the trailing **`Run with …`** command still pointed at an old flat filename. All were rewritten to the canonical form `python3 sessions/<module>/<day>/experiment.py`, matching the `experiment.py` that actually exists in each day's directory.

| Lesson | Was | Now |
|--------|-----|-----|
| m03 · day-02 · qkv | `python3 m04b_qkv.py` | `python3 sessions/m03-attention/day-02-qkv/experiment.py` |
| m03 · day-03 · attention-scores | `python3 m04c_attention.py` | `python3 sessions/m03-attention/day-03-attention-scores/experiment.py` |
| m03 · day-04 · multihead | `python3 m04d_multihead.py` | `python3 sessions/m03-attention/day-04-multihead/experiment.py` |
| m05a · day-01 · residual-connections | `python3 m05a_residual.py` | `python3 sessions/m05a-text-transformer/day-01-residual-connections/experiment.py` |
| m05a · day-02 · layer-norm | `python3 m05a_layernorm.py` | `python3 sessions/m05a-text-transformer/day-02-layer-norm/experiment.py` |
| m05a · day-03 · feed-forward | `python3 m05a_ffn.py` | `python3 sessions/m05a-text-transformer/day-03-feed-forward/experiment.py` |
| m05a · day-04 · full-block | `python3 m05a_block.py` | `python3 sessions/m05a-text-transformer/day-04-full-block/experiment.py` |
| m05a · day-05 · causal-masking | `python3 m05a_causal_mask.py` | `python3 sessions/m05a-text-transformer/day-05-causal-masking/experiment.py` |
| m05a · day-06 · encoder-vs-decoder | `python3 m05a_encoder_vs_decoder.py` | `python3 sessions/m05a-text-transformer/day-06-encoder-vs-decoder/experiment.py` |
| m05a · day-07 · text-generation | `python3 m05a_sampling.py` | `python3 sessions/m05a-text-transformer/day-07-text-generation/experiment.py` |
| m05a · day-08 · full-transformer | `python3 m05a_full_transformer.py` | `python3 sessions/m05a-text-transformer/day-08-full-transformer/experiment.py` |

**Already correct (left as-is):** all 6 `m04-first-model-mlp` lessons and `m03/day-01`, `m03/day-05` — their `Run with` command already used the full `sessions/…/experiment.py` path. After the fix, all **19** run commands are consistent with the file each Produce section says to create.

---

## 4. Task 3 — Title spacing glitch fixed (1 lesson)

`m03-attention/day-05-positional` — the `<h1>` stored `Positional Encoding &amp;<span class="sub">the Shuffle Problem</span>`, which rendered with the ampersand orphaned against the subtitle ("Encoding &the Shuffle Problem"). Moved `&amp;` into the subtitle span so it reads cleanly and matches the established sibling pattern (`m04/day-03`: `<span class="sub">&amp; Overfit-One-Batch</span>`):

```html
<h1>Positional Encoding<span class="sub">&amp; the Shuffle Problem</span></h1>
```

The `<title>` tag (`… Positional Encoding &amp; the Shuffle Problem`) was already correct and was not touched. A scan of all 19 `<title>`/`<h1>`/heading/nav-title strings found **no other** missing-space-around-`&` glitches.

---

## 5. Task 4 — Markdown reports reformatted

`batch_2_rollout_plan.md` and `batch_2_rollout_report.md` render as clean Markdown source. Diagnostics showed **no** CRLF, trailing whitespace, tabs, or malformed tables (all table column counts consistent — nothing that would break rendering). The only defects were blank-line lint issues:

- **MD032** — a bulleted list started immediately after a `**bold:**` paragraph line (plan §4 "Notes / guardrails"; report §8 advisories).
- **MD022** — the seven `### 5.x` P0-spec headings were each followed immediately by their first bullet with no blank line.

Fix was **blank-line insertion only** (9 blank lines: 8 in the plan, 1 in the report). Diff-verified: **0 content lines changed, 0 removed** — every added line is blank. Both files now report **0 markdownlint issues** (MD022/MD032/MD012/MD058 all clear).

---

## 6. Task 5 — Audit results (all 19 Batch 2 lessons)

**`lesson_audit.py` (targeted — `_recover_set.json` left unchanged):**

```
TOTAL lessons audited: 19
  OK: 19   MISSING: 0   LEFTOVER: 0   DEGRADED: 0
```

**jsdom structural + interaction gate** (`/tmp/batch1_verify.js`, jsdom 29.1.1): **19/19 render clean**, 0 runtime errors; each lesson keeps 7 sections / 7 got-it / 3 demos / 4 quiz / BUILD / tooltip controller. The **7 P0 labs** (`--p0`) still re-render on simulated slider/toggle input:

| P0 lab | Interaction re-render |
|--------|:---:|
| m03 · day-02 · qkv | ✓ |
| m03 · day-05 · positional | ✓ |
| m04 · day-02 · backward-pass | ✓ |
| m05a · day-01 · residual-connections | ✓ |
| m05a · day-02 · layer-norm | ✓ |
| m05a · day-05 · causal-masking | ✓ |
| m05a · day-07 · text-generation | ✓ |

**JS syntax:** every inline `<script>` in the 12 edited files parses clean (`vm.Script`).

**Remaining soft advisories (expected, unchanged by this pass):** `math-heavy but no Math Ladder` ×3 on the three intentional non-formula lessons — `m04/day-05-pytorch-version`, `m04/day-06-why-pytorch`, `m05a/day-06-encoder-vs-decoder`. Documented as correct-by-design in the rollout report; this cleanup did not touch their ladders.

---

## 7. Files changed

**12 lesson files** (13 of the 19 were candidates; 12 needed an edit — `day-07` needed both a dedup and a run-command fix):

| Module | Lessons edited | What changed |
|--------|----------------|--------------|
| `m03-attention` | day-02, day-03, day-04 | run command |
| `m03-attention` | day-05 | title `<h1>` spacing |
| `m05a-text-transformer` | day-01, day-02, day-04, day-05, day-06, day-08 | run command + duplicate trade-off callout |
| `m05a-text-transformer` | day-03 | run command |
| `m05a-text-transformer` | day-07 | duplicate trade-off callout + run command |

`git diff --stat` (lessons): **12 files changed, 12 insertions(+), 19 deletions(-)**. Deletions = 11 stale run-command `<p>`s + 1 old `<h1>` + 7 duplicate `⚖️` callouts.

**2 report files:** `batch_2_rollout_plan.md` (+8 blank lines), `batch_2_rollout_report.md` (+1 blank line).

**Not touched:** any non-Batch-2 lesson; every lesson's nav, quest-id, quiz `QS`, `DEMOS`, `BUILD`, tooltip machinery, P0 `.vlab` labs, Staff Lens, failure-mode and interview callouts, and the `experiment.py` files themselves.

---

## 8. Preservation contract — honored

- **Additive/subtractive-safe only.** The only removals were stale text (run commands), one restyled `<h1>`, and byte-identical duplicate lines. No unique concept, failure mode, trade-off, equation, analogy, or interview answer was lost.
- **No new visuals, no broad content refactor** — as scoped.
- **Structure intact** on all 19: 7 sections, 7 got-it, 3 demos, 4 quiz, BUILD, tooltip, and the shell/section boundaries `_shell_migrate.py` relies on.
- **Offline/self-contained** unchanged: no new assets, CDN, or fonts.

---

## 9. Definition of done — Batch 2 cleanup

- [x] Task 1 — duplicate "coherence vs diversity" trade-off in `day-07` reduced to one; exhaustive scan found and fixed 6 more of the same defect; 2 false positives verified and left.
- [x] Task 2 — 11 stale run commands rewritten to `python3 sessions/<module>/<day>/experiment.py`; all 19 now consistent with the created file.
- [x] Task 3 — `Positional Encoding &the Shuffle Problem` fixed; no other title spacing glitches in the batch.
- [x] Task 4 — both rollout reports reformatted to 0-issue Markdown (blank-line-only changes, meaning preserved).
- [x] Task 5 — targeted audit run on all 19: `lesson_audit.py` 19/19 OK / 0 DEGRADED; jsdom 19/19 clean; 7 P0 labs interaction-verified.
- [x] Task 6 — this report written.
- [x] No non-Batch-2 lesson touched; no protected machinery removed (diff-verified).
