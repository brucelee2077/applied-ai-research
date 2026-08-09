---
name: frontier-lesson-builder
description: Use to build or stabilize lessons so learning experience, mechanism, and staff-depth are integrated rather than checklist-driven.
---

# Frontier Lesson Builder

Your job is to make the learner want to continue while still building staff-level depth.

> **Authoritative V9 authoring grammar + compile/gate loop:**
> `sessions/_compiler/AUTHORING.md` (source of truth — matches the shipped compiler).
> Author every `source.md` from that file, not from memory or the obsolete
> `v8_source_first_authoring_plan.md`.

# Communicating with the user
Your text output is what the user reads between tool calls -- write it for a teammate catching up, not a log file. Before your first tool call, say in a sentence what you're about to do; give brief updates when you find something load-bearing or change direction. Lead with the outcome: your first sentence after finishing should answer "what happened," with supporting detail after. Readable matters more than terse -- use complete sentences, not fragments, arrow chains, or codenames you coined mid-run; shorten by including less, not by compressing. Match the response to the question.

## Non-negotiables

No formula before felt intuition.

The opening (hero) leads with a **concrete everyday analogy** — a physical thing the reader has done — before any aspiration or relevance line.

Math restraint: heavy or multi-step math goes in an **optional, skippable box**; the main flow carries intuition + a picture + an analogy, and any main-flow formula is one line, narrated in words.

Visualize the build-up: a heavy or multi-step build-up (worked example, derivation, numeric ladder, matrix/shape change, or math demoted to an optional box) is itself SHOWN in a picture — a **second figure, a run-demo, or a math ladder** — not left as text + equations under only the opening analogy visual. Humans read a graph far faster than a paragraph.

Draw the analogy: each concept's OPENING visual ILLUSTRATES the everyday analogy itself — a picture of the concrete thing (a one-way valve, a dimmer, a see-saw, a pizza sliced into shares, a weather forecaster) — so the *meaning* lands before any math. The equation/mechanism picture comes AFTER, in the build-up. A concept whose opening picture is the math (a curve, an axis plot, an equation) while the analogy lives only in words has it backwards.

Jargon is glossed **define-before-use, inline** — never a front-loaded wall. The cheat-sheet + a one-page recap live at the END.

No checklist before invitation in foundation lessons.

No decorative analogy. Use a narrative spine.

Match the length of written deliverables (especially Markdown files) to what the task needs: cover the substance, but do not pad documents with filler sections, redundant summaries, or boilerplate.

## Plain Language Discipline (the WORDS — write to this, don't just get graded on it)

The registers above govern STRUCTURE: order, visuals, analogies, chunking. Authors
already hit those (measured: a blind authoring probe produced 4 visuals, an interactive
widget, an analogy with its breaks-down beat, and prose runs under 400 chars). What was
never specified is the **words**, and that is the axis that was failing.

Your reader is a curious 12-year-old for whom **English is a second language**. Write to
these eight levers — they are exactly what `judge_plain_language_absolute` grades every
round, with or without a notebook, and it hands back the reader's `hardest_words` as
concrete rewrite targets:

```text
vocabulary_age        everyday words. A word learned after ~age 10 is avoided or
                      glossed in plain words AT first use.
                      ✗ utilize · monotonic · orthogonal · canonical · asymmetry
                      ✓ use · always goes one way · at right angles (90 degrees)
                      A name you must teach (LayerNorm, Xavier init, MNIST) gets an
                      inline [[term||plain gloss]] the first time — never bare.
sentence_simplicity   short sentences, ONE idea each, active voice. Break run-ons.
                      About each SENTENCE — never about the lesson's length.
term_before_use       explained the FIRST time it appears. Not used in concept 2 and
                      glossed in concept 9.
concrete_over_abstract the physical, experienced thing lands before the abstraction.
math_restraint        see the Math Restraint rule.
hero_analogy_scaffold everyday analogy first, and say where it BREAKS DOWN.
warmth_and_pace       brilliant-friend voice; normalize confusion; let ideas breathe.
no_idioms             an idiom excludes a second-language reader. Banned outright:
                      "under the hood" · "out of the box" · "rule of thumb" ·
                      "in a nutshell" · "obviously" · "as you can see" · "trivially" ·
                      "recall that" · "this is just" · "of course," · "needless to say"
```

**LENGTH IS NOT A DEFECT** (user directive 2026-08-06). Never shorten, and never cut
coverage, to satisfy any of this. Fix density by CHUNKING (Build-Up Register) and
readability by simplifying WORDS AND SENTENCES. A long lesson broken into small one-idea
beats is exactly right — verified: 997 chunked words pass the wall check, 157 unbroken
words fail it.

## The Four Beginner Non-Negotiables (user directive 2026-08-05)

Four things matter most for a first-time learner. Everything else in this skill
serves them. **Each axis is enforced by a JUDGE for the quality call and a
deterministic gate for the countable facts** — that split is deliberate. An LLM
judge is the only thing that can see vocabulary age, analogy quality or curiosity;
a regex is the only thing that cannot fail open when the bridge is down. Neither
alone was enough: judge-only returned all-green on 20/20 days that had real
defects, and a word blacklist caught 2 of them.

| # | The ask | Semantic enforcer (the bar) | Deterministic floor (cannot fail open) |
|---|---------|-----------------------------|-----------------------------------------|
| 1 | **Simple language** | `judge_plain_language_absolute` — notebook-FREE, 8 levers incl. vocabulary_age, sentence_simplicity, hero_analogy_scaffold, warmth_and_pace; returns the reader's `hardest_words` | `beginner_language_gate` — banned phrases/idioms, chunking geometry |
| 2 | **Inspire curiosity** | `judge_interest_absolute` (7 levers) + `judge_body_engagement` | `concept_structure_gate` failure-cluster warn (advisory only) |
| 3 | **More visualization** | `judge_concept_structure.buildup_visualized` | `reader_flow_gate` + `concept_shell_gate` + `visual_integrity_gate` — real blocking, exit 2/3/4 |
| 4 | **12-year-old analogies** | `judge_concept_structure.analogy` — GOOD only when the analogy is DRAWN | none (see the blind spot below) |

**Both absolute floors compute `overall` from the LEVER GRADES in code, not from the
model's own word.** The model does not apply its own stated threshold: 6 of 20 m02–m04
days returned FLOOR_MET from the interest floor while carrying ≥2 WEAK levers, and the
plain-language floor did the same on its first run. `_floor_from_levers` does that
arithmetic and keeps the disagreement visible as `overall_stated_by_model`. The
judgment stays with the judge; the threshold cannot drift.

**LENGTH IS NOT A DEFECT** (user directive 2026-08-06). No enforcer here measures total
length, and none should. The chunking check is length-NEUTRAL by construction: verified
997 words in one-idea beats PASSES while 157 words in an unbroken run FAILS. So fix
density by CHUNKING and fix readability by simplifying WORDS AND SENTENCES — never by
cutting coverage.

**Row 1's gate does not block a compile.** `beginner_language_gate` is not in
`compile_lesson.py`'s dispatch (only a `lesson_build.js` prompt line), and it currently
FAILS 40 of 46 shipped V9 days — promoting it to a hard gate needs a corpus fix pass
first. Run it on every new day and treat its FAILs as blocking.

**Rows 2–4's judges fail OPEN.** `BRIDGE_UNAVAILABLE` → `N/A` → the loop PASSES, so
every lens prints an explicit "UNENFORCED this round" note instead of a silent pass.
Row 3 also has one fail-open path: `compile_lesson.py` wraps `visual_integrity_gate` in
a bare `try/except` with `vok` pre-set True.

⚠️ **Row 4's judge cannot see drawings.** `judge_concept_structure` is fed
`_readable_text(html)`, which strips every tag — so a shape-only SVG (a valve drawn with
`<rect>`/`<path>`, no labels) is 100% invisible, while a label-rich equation figure reads
as "drawn". The axis effectively grades SVG *text labels*. Label your analogy figures.

⚠️ **Digestibility measures BUILD-UP prose only.** `_density_scan.buildup_of` starts
after a concept's first visual, so the hero and every concept INTRO are unmeasured
(planted 1259- and 1364-char walls there both passed). The intro is where the felt
intuition lands — keep it chunked on your own judgement.

⚠️ **Do NOT write `notebook_yardstick: null` to make a lesson easier to ship.**
It turns OFF the tone judge (`plain_language`, `analogy_quality` incl. the HERO's
analogy scaffold) and the interest ceiling. `beginner_language_gate` hard-fails a
null yardstick when a companion notebook actually exists — that is exactly how
m04 day-06 shipped with beginner-friendliness ungraded while a 263 KB notebook
sat unused on disk.

Run these before you call a day done:

```bash
python3 sessions/_compiler/gates/beginner_language_gate.py <source.md>   # rows 1 + real play
python3 sessions/_compiler/gates/concept_structure_gate.py <source.md>   # per-concept triad
python3 sessions/_compiler/compile_lesson.py <source.md>                 # row 3 (structural)
python3 sessions/_compiler/gates/coverage_judge.py <lesson.html> --source <source.md>  # rows 2+4
```

The `coverage_judge` CLI always exits 0 — read its verdict lines, do not trust its
exit code.

## Beginner Intuition Register (per concept — the notebook's tone)

Match the register of a strong beginner notebook: warm, plain-language, analogy-first, heavy on *why*. A staff-depth lesson is NOT an excuse to go straight into mechanism — the intuition layer must carry every concept, not just the hero. This is the difference between "correct but cold" and a lesson a beginner actually wants to read.

**Enforcement, stated honestly.** Reading level and warmth are graded by the
**beginner-friendliness judge** (`coverage_judge.judge_tone`: warmth /
analogy_quality / intuition_depth / math_restraint / plain_language /
progressive_disclosure / curiosity / pace) — drive it to MATCHES_NOTEBOOK **when a
notebook exists**. That judge returns `N/A` without one, so on a null-yardstick day
your only plain-language enforcer is `beginner_language_gate` (row 1 above), and
warmth/pace go ungraded. On those days the register below is on YOU, not a gate.

Every concept unit opens with **felt intuition before any formula, range, or notation**, in this order:

```text
1. Plain-words "what is it"   one sentence a 12-year-old gets, no symbols
2. Full analogy SCAFFOLD      not a one-word metaphor. Give all four beats:
                              (a) a concrete everyday thing (switch, dimmer, valve),
                              (b) what the analogy gets RIGHT,
                              (c) the concept restated in plain words,
                              (d) where the analogy BREAKS DOWN (one sentence).
3. Why it matters             the problem it solves, one line
4. THEN the mechanism         narrate EVERY symbol in words before/beside the formula
                              ("1 over 1 plus e-to-the-minus-z" — not a bare glyph); its own visual
5. When to use it             plain-language pros & cons — the notebook's "When to Use This"
```

**Warmth & encouragement (sustained, not just the hero):**
- Voice = a brilliant friend rooting for the reader, never an exam or a textbook. Prefer "let's…", "you", "notice".
- Normalize confusion: when something is hard, SAY so ("this part trips almost everyone up the first time — that's normal"). Do NOT frame the learner as error-prone.
- Victory laps: after a hard idea, tell the reader what they just unlocked ("you just discovered why depth needs a bend").
- Keep required staff/interview depth, but frame it as empowerment ("now you can see why…"), and let it FOLLOW the warm intuition — never lead with "this is the staff-level part" or "say this in an interview" as the concept's opening move.

**Plain language & pace:**
- Short sentences, one idea each. Break run-ons. Active voice. No idioms.
- Introduce each term define-before-use with an inline `[[term||plain gloss]]` the FIRST time it appears — NOT a front-loaded jargon wall the reader hits before any intuition. Collect the full glossary as a reference cheat-sheet in the CLOSING recap unit (see the Recap Rule), not at the top. (This reverses the old "front-load a Jargon Ladder" rule, per direct user review 2026-07-20 — the inline `[[term||…]]` glosses ARE the define-before-use mechanism.)
- Let ideas breathe: a short recap or a one-line "so far…" between big concepts; don't pour them out in one stream.

Mark **P0 (intuition-inverted)** if a concept opens with a formula, a range like `(0,1)`, `max(0,z)`, or notation before the felt picture; if a concept is all mechanism with no analogy and no "why"; if an analogy is given WITHOUT its "where it breaks down" beat; or if the dominant voice is interview/textbook rather than warm-beginner. The inline visual anchors the *intuition*, not the algebra. See repo `CLAUDE.md` §5 (analogy scaffold, simple words, no idioms) and §7 (curiosity hook, normalize confusion, victory laps).

## Build-Up Register (per concept — keep the BODY as engaging as the intro)

The Beginner Intuition Register above governs the concept OPENING. This is its twin for the
BUILD-UP — the mechanism / math / worked example AFTER the intro. The failure it prevents
(direct user review 2026-07-24): the intro is warm and curious, then the body switches into
a flat textbook register — the spine analogy is dropped once the math starts, worked
examples are cold symbol-pushing, and dense blocks run with no re-hook. Structure + a visual
are necessary but NOT sufficient: the body PROSE must be as engaging as the intro. The eval
gate is the **`judge_body_engagement`** floor (`coverage_judge.py`) — per concept it grades
`body_engagement` GOOD / WEAK / MISSING (NA for the recap unit or a one-line definitional
concept); MISSING is a **P0** in the build loop, WEAK a P1. Drive every concept body to GOOD.

Hit these body beats in every concept's build-up:

```text
1. Bridge from the analogy   the build-up's first move re-invokes the opening analogy
                             ("let's go back to the ruler and watch what happens when we
                             stack two") — never a cold switch straight into mechanism.
2. Re-hook / "why this bites" a one-line stakes beat INSIDE the body so the mechanism
                             answers a live question ("this collapse is sneaky — the net
                             runs fine but secretly learns nothing"). Use %%% insight.
3. Narrate the architecture  causal connectors (therefore / which means / that's why) and
                             semantic step NAMES ("the collapse", "the fix") — NOT bare
                             "Step 1 / Step 2 / Step 3" or silent symbol-pushing. Use %%% steps.
4. Predict-then-reveal       invite a guess before the number in a worked example. Use
                             %%% demo with a predict: line.
5. Micro-recap + breath      a one-line "so we've shown that…" landing after a dense run.
6. Normalize + victory lap   DURING the hard part, not only in the intro ("this is the
                             tricky bit — totally normal"; "you just did the hard part").
7. Spine alive to the end    keep the analogy running through the mechanism, and mark
                             where it breaks down — don't abandon it once math starts.
8. Max TWO trap beats        a "trap beat" is any beat whose payload is a failure, limit,
   in a row                  gotcha or warning. Three or four stacked back to back read as a
                             grim march even when each is warm on its own, and the interest
                             floor keeps rating the `momentum` lever WEAK on exactly that
                             shape. Put a win, a payoff or a live widget between them. This
                             is the within-build-up twin of the concept-level rule
                             `concept_structure_gate` warns on (>=3 consecutive failure/limit
                             concept UNITS); that gate cannot see beats inside a single
                             build-up, so this one is on the author. NEVER fix momentum by
                             cutting a failure mode — coverage requires every failure paired
                             with its remedy. Interleave and enrich.
```

**Digestibility is a separate axis from voice, and the judge is blind to it.**
`judge_body_engagement` grades how a body SOUNDS. It cannot see how dense it is, and "hard
to digest" (the user's actual words) is a density complaint. So measure it:
`sessions/_density_scan.py` reports `walls_over_600` (unbroken MAIN-LINE prose paragraphs)
and `asides_over_600` (unbroken runs inside `!!!` callout boxes). Target **zero of each**.
Its `argv[1]` is the OUTPUT json path, not a source — it now refuses any path not ending in
`.json` (it used to overwrite the lesson you pointed it at, which destroyed a `source.md`
twice in one session), and its day list is now a glob over every module rather than a
hardcoded m02+m03 pair. That widening matters: across all 46 concept days it reports **38
with main-line walls over 600** (worst 1766), so the long-standing "zero walls" status only
ever described the 14 days it was pointed at. For a single day prefer
`beginner_language_gate.py <source.md>`, which takes the source as INPUT and reuses this
script's tested `longest_wall` so the two agree.
Note the two have different cures: a main-line wall gets split into paragraphs, a `%%% steps`
ladder or a `%%% insight`; a long "Optional (skippable)" box is already correctly placed by
math-restraint rule 4 and only needs breaking up INSIDE the box, with `<br>` and short bolded
sub-lead-ins. Do not promote a box into main-line prose to fix it — that turns a skippable
aside into a paragraph every reader must cross.

**Tools (see AUTHORING.md):** `%%% insight` (the re-hook / "why this matters" callout),
`%%% demo` with a `predict:` line (predict-then-reveal discovery), `%%% steps` (a narrated
worked-example whose steps assemble as the reader scrolls). Warm a cold body by adding
VOICE, not length — never pad, never cut coverage.

Mark **P0 (body-cold)** if a concept's build-up drops into a flat mechanism/symbol dump
with no voice, no re-hook, no discovery, and the opening analogy abandoned.

## Interest, Reader-Separation & Real Play (2026-07-23 — first-class, not byproducts)

Interest is now gated independent of any notebook (an ALWAYS-ON absolute floor in `coverage_judge.judge_interest_absolute` P0-gates the build loop; the notebook is an added ceiling, not the only bar). So author to these as NON-NEGOTIABLES, not as side-effects of warmth:

- **Aspiration/relevance opening (P1 gate):** each concept OPENS with wonder / what-it-unlocks + a thing the reader already cares about (ChatGPT, face unlock, games) — never problem-first ("without this, X is useless"). Relevance RETURNS mid-lesson, not one mention in the hero.
- **Breadth/payoff tease (P1 gate):** tease the family/landscape ahead; end with a payoff + recap.
- **Reader-separation (P0 — CLAUDE.md §1):** a foundation lesson body is Reader A only. Do NOT write scripted "here's how you'd say it in an interview" monologues, and do NOT drop Reader-B notation (bare matrix identities like `(x·W₁)·W₂=x·(W₁·W₂)`, terms like "non-convex"/"representational ceiling") without a plain gloss. Keep the required staff depth as ONE plain-language empowerment line ("now you can explain *why* deep nets need a bend") — depth stays, interview *scripting* goes to a separate file.
- **Momentum — no failure-mode wall (P1):** teach every failure + remedy (coverage still requires it), but do NOT stack ≥3 "Puzzle→Cause→Remedy"/limit units back-to-back with no play/payoff between — INTERLEAVE a win or a live widget. `concept_structure_gate` warns on a ≥3 failure cluster. **Length is fine — never cut coverage to fix momentum; interleave and enrich instead.**
- **Real play, honestly labelled:** ship ≥2 genuinely interactive `%%% viz` widgets, front-loaded (not one, late). An in-body `<input type="range">` slider counts as real play too — m02 ships it that way, m03 as viz iframes; either form is fine, **zero is not**. A `%%% demo` only reveals a pre-baked output — its button says "reveal", NOT "run it"; do not imply computation the widget doesn't do. Checkable form: `beginner_language_gate.py` hard-fails zero interactive widgets, and hard-fails a `%%% demo label=` that promises a run. (v8lib's DEFAULT label is already honest; the measured problem is authors overriding it — 48 of 103 demo labels in m02–m04 say "run".)

## Three-layer architecture

```text
1. Experience Layer
   reduce fear, build curiosity, familiar human situation, mental picture

2. Mechanism Layer
   definitions, formula, step-traced example, visual/evidence, implementation

3. Staff Depth Layer
   failure mode, scale implication, debugging signal, artifact, interview answer
```

## Seed Stabilization Writing Rule

When stabilizing a seed module, do not over-polish.

Only rewrite enough to prevent a bad pattern from propagating.

Allowed targeted changes:

- move mental picture before technical vocabulary,
- reframe section openings from task/checklist to curiosity,
- carry an analogy through mechanism/staff/artifact sections,
- reframe Produce as discovery,
- delay frontier/staff pressure until learner orientation.

Not allowed unless explicitly needed:

- broad redesign of all lessons,
- unrelated prose polish,
- shell/navigation/quest-id changes,
- changing artifact requirements without QA reason.

## Learning Barrier Gate

Mark P0 if:

- opening feels like work before wonder,
- first technical term appears before mental picture,
- foundation lesson starts engineering-first,
- staff/frontier relevance arrives before orientation,
- artifact feels like homework before curiosity.
- a concept unit that introduces a visual object (curve, distribution, boundary, shape) but is text-only (no inline visual),
- a concept's picture deferred to a later unit, or all visuals dumped in one late "build" section.
- a concept with a HEAVY or multi-step build-up (worked example, derivation, numeric ladder, matrix/shape change, or math demoted to an "Optional (skippable)" box) that is NOT itself visualized — left as text + equations under only the opening analogy figure (violates the Visualize-the-Build-Up Rule).
- a concept whose BUILD-UP goes cold — a flat mechanism/symbol dump with no voice, no re-hook, no discovery, the opening analogy abandoned once the math starts (body-cold — violates the Build-Up Register; the `judge_body_engagement` floor flags `body_engagement` MISSING as P0).

Mark seed-stabilization P1 if this issue exists in a seed module and is likely to be copied forward.

## Artifact as discovery

Produce should feel like:

```text
Run this and observe something surprising.
```

not:

```text
Submit this homework.
```

Acceptance criteria should include observation and explain-back.

## The doing leg (`experiment.py`)

The produce section is the day's PROSE; `sessions/<module>/<day>/experiment.py` is the day's
DOING leg — a **separate deliverable the lesson author owns**. `lesson_build.js` does not write
it, no judge lens reads it, and `concept_shell_gate` only checks that the produce section
MENTIONS the filename. So a green compile plus a clean judge panel says nothing about whether the
learner can practise the day: until 2026-07-27 all but 14 of 115 days shipped a 5-line
"Placeholder. Fill this…" stub behind a fully passing lesson. A day is not done until its
artifact is real and gate-passing.

**Build a module's doing leg with the engine, not by hand:**
`sessions/_compiler/workflows/doing_leg_build.js` via the Workflow tool, `args {module,
days:[...], maxRounds, cumulative, force}`. It triages first — skipping any day whose artifact
already passes the gate, because assuming a whole module is stubs once cost five working
artifacts — then runs one writer per day, then ONE module-wide adversarial reviewer, then a
cross-day pass. Hand-author only when the target is a single day.

- **The spec already exists — never invent requirements.** `python3 sessions/_produce_spec.py
  sessions/<module>/<day>` prints the day's produce block. Its `claude_prompt` — the numbered
  Option-B list you already wrote for the learner — IS the specification, together with the "What
  you should see" acceptance lines. Build exactly that, in the lesson's own spelling, variable
  names and numbers. Those lines address the LEARNER, so "paste the output as a comment" and
  "write three lines into `log.md`" are not requirements on the artifact — where the produce prose
  and the self-check contract conflict, the contract wins. If the extract is not checkable, fix
  the produce section first (`frontier-curriculum-architect`, Produce Spec Rule) rather than
  inventing a requirement the lesson never promised.
- **Shape, length and environment are already specified — do not re-derive them.** The contract
  (header, "Today's big idea in two lines of output", the run command, `# --- Part N ---`
  sections, printed shapes at every step, one boolean per claim, `✅`/`❌`, one assert per claim)
  and the 85-140-line target live in **`sessions/_compiler/AUTHORING.md` section 10**; the binding
  environment and the current library/model inventory live in **`sessions/_experiment_env.md`**.
  Read both, and copy the gold standard
  `sessions/m02-the-neuron/day-02-activations/experiment.py`. Two things those files leave to you:
  confirm the day's imports actually import before you write (a lesson may teach a library this
  machine does not have — declare the substitution in a comment rather than faking the API), and
  know that "no file writes, no `savefig`" is an unenforced convention, so no gate will catch you
  breaking it.
- **Pin every claim; seed everything.** Each expected value must be WRITTEN DOWN in the
  self-check (prefer an exact literal), never re-derived from the code path under test — that is
  an algebraic identity, and it holds while the code is broken. Two runs must print
  byte-identical output.
- **Run the gate before you call the day done:** `python3 sessions/_experiment_check.py
  sessions/<module>/<day>/experiment.py` (structural contract + a REAL run).
  `frontier-refactor-qa` owns what that gate does and does not prove.
- **Then PIN THE GOLDEN REFERENCE, as the last step of authoring the day:**
  `python3 sessions/_experiment_check.py <path> --write-expected`. It stores the day's stdout as
  `<day>/expected_output.txt`, and from then on ANY change to what reaches the screen fails the gate.
  That is the only thing that catches a corruption at the print CALL — an operand swapped inside a
  `%`-tuple, an index shifted, arithmetic wrapped around a name that is already correctly asserted.
  Assertions cannot see those: measured on this corpus, 498 such plants, 4 caught. ⚠️ **Read the
  output before you pin it.** The reference detects DRIFT, not correctness — pin a broken day and the
  gate defends the breakage forever, which makes this the single most consequential step in
  authoring the artifact. Because the reference covers the rendering, do NOT also pin rendered text
  in the file; keep in-file claims on VALUES, and leave the file editable for the learner.

**Writing a self-check that cannot fail is the default failure mode, not an edge case.** Ten
vacuity patterns and five harder no-op traps were each proven on a real day by planting the bug
and watching the script still print `✅ you got it`. Read
**`sessions/_compiler/AUTHORING.md` section 10** for the catalogue with worked examples before you
write your first self-check; the review-side list lives in `frontier-refactor-qa`. The authoring
consequences: assert exception TYPES, never a library's message text; compute any "predict" value
from the inputs instead of hardcoding a string; and choose shapes and values where a wrong
spelling gives a DIFFERENT answer — a parameter initialised to zero, symmetric test data
(`Q = K = V = x`), a uniform output, or floor division each make their own code path untestable.
And **print the same object you assert**: compute a quantity once, bind it to a name, then print
that name and assert that name. Two parallel expressions for one claim means corrupting the printed
one is invisible, and the learner reads a wrong number under a `✅`.
Then PROVE the check bites: plant **at least 4 semantic defects, one at a time, on a copy in
`/tmp`**, and confirm each is caught while diffing the FULL stdout — a plant that reproduces the
same value is a no-op and proves nothing about your check. A writer cannot clear its own
self-check; a separate module-wide adversarial reviewer is required, which is what the engine's
stage 2 exists for.

## Staff courage rule

Staff depth should feel empowering:

```text
You can understand the small mechanism, then scale the reasoning up.
```

## v8 — author in source, not HTML

Under v8 you write `source.md` in reader-flow order; a compiler emits `lesson.html`.

Reordering reader flow is moving a block, not splicing HTML. Do not hand-edit `lesson.html`.

## Reader Flow Blueprint (v9 — concept-driven)

A lesson body is a sequence of **concept units**, not a fixed template. Author `source.md` (mode: concept) as:

    hero (curiosity hook — a concrete everyday analogy)
    → concept unit 1  (plain-words intro → its OWN inline visual → build-up)
    → concept unit 2  (…)
    → … as many concept units as the topic needs
    → recap unit  (tag="Recap"; day summary in a few beats + reference cheat-sheet + its own visual)
    quiz (one section, all questions)
    produce (discovery artifact)
    fin

Each concept unit MUST carry its own inline visual, placed immediately after the intro and before the build-up. Never defer a concept's picture to a later unit or to one shared "build" section. Introduce each term define-before-use with an inline `[[term||gloss]]` — NOT a front-loaded jargon wall — and end with a recap unit that holds the cheat-sheet (see the Jargon Rule + Recap Rule). Keep the narrative spine word running through hero + most concepts.

WITHIN a single concept unit, order the beats like this (this is unit-level ordering, not fixed body sections):

```text
plain-words intro → opening inline visual that DRAWS THE ANALOGY (the everyday object
itself — anchors the intuition) → worked example → mechanism (+ a BUILD-UP visual that
DRAWS the transformation/worked-example/ladder) → frontier payoff → failure/misconception → build-up
```

## Draw the Analogy Rule

Every concept states an everyday analogy in words; **also draw it.** The concept's OPENING visual (the intuition anchor, beat 2) must ILLUSTRATE THE ANALOGY OBJECT ITSELF — a picture of the concrete thing the analogy names — so the reader sees the *meaning* before any math:

- ReLU → a **one-way valve** (positives flow, negatives blocked); sigmoid → a **dimmer knob**; tanh → a **see-saw**; linear collapse → a **ruler / two straight rulers**; softmax → a **pizza sliced into shares**; forward pass → an **assembly-line conveyor**; MSE → a **golf score / dartboard**; cross-entropy → a **weather forecaster's surprise**; dead ReLU → a **jammed valve**; XOR → a **rope that can't split the room**.

Draw the object with labelled parts (a beginner should look and think "oh — it's like a ___"), NOT the equation, a bare axis plot, or a curve — that math picture is the *mechanism* visual and belongs in the build-up (see the Visualize the Build-Up Rule). So a full concept typically carries the analogy picture FIRST, then the math/interactive picture. Interactive explorable widgets (drag/slide) are the gold standard for the mechanism visual; keep the analogy picture as the anchor before it.

Checkable form: the Concept-Structure Judge's `analogy` axis grades GOOD only when the analogy is DRAWN (opening visual pictures the everyday object) — an analogy carried only in words while the opening visual jumps to the math is WEAK (a P0 in the build loop, since analogy WEAK/MISSING gates).

## Visualize the Build-Up Rule

The opening inline visual anchors the concept's *opening intuition* only — it does NOT discharge the duty to visualize the *build-up*. Whenever a concept's build-up is heavy or multi-step — a worked numeric example, a derivation, a growth/decay ladder (e.g. the `0.25×0.25×…` vanishing-signal fade), a matrix/shape transformation, a multi-symbol formula, or anything demoted into an "Optional (skippable)" box — that build-up MUST itself be SHOWN in a picture placed at the build-up (after the opening visual):

- a **second `%%% svg`** that draws the transformation itself — the shape/curve/value changing as each term is added; a **before→after in ONE figure** when a step fixes a problem (CLAUDE.md §6 rule 4);
- a **`%%% demo`** whose printed `out:` makes the numeric build-up watchable (predict-then-run);
- a draggable **`%%% viz`**, or a **`%%% mathladder`** (words→formula→numbers→sanity).

Humans read a graph far faster than a paragraph of algebra — bias toward MORE visuals: graph every transformation you ask the reader to follow, prefer a per-step figure that assembles left-to-right over one busy final diagram, and never leave a numeric ladder or worked example as monospace text when an SVG of bars/curves would let the reader SEE it. Match the notebook yardstick, which plots in about half its code cells. Net: a concept with a real build-up ships **at least two** visual elements (opening anchor + build-up figure); a light/definitional concept needs only its one opening visual, and is never penalized for it.

Checkable form: the **Concept-Structure Judge**'s `buildup_visualized` axis (in `coverage_judge.py`) grades this — MISSING (heavy build-up, no build-up visual) is a **P0** in the build loop, WEAK is P1, and a LIGHT build-up returns NA (never penalized). The deterministic `concept_structure_gate.py` adds an offline advisory warn keyed off `%%% mathladder` / "Optional (skippable)" boxes.

## Jargon Rule (define-before-use inline; cheat-sheet at the bottom)

Introduce each term the first time it appears, define-before-use, with an inline `[[term||plain gloss]]`. No term appears before its gloss.

Do NOT front-load a `%%% jargon` wall in the first concept — a screen of definitions before any intuition overwhelms a beginner (direct user review 2026-07-20). Collect the full glossary as a reference **cheat-sheet in the closing recap unit** instead.

Checkable form: the reader_flow gate HARD-FAILS a `%%% jargon` block in the first concept unit; the tone judge grades `progressive_disclosure` **only when a notebook exists**. ⚠️ Nothing checks "no term appears before its gloss" — the one place the compiler reads `[[` is behind `if strict:` on the v8 non-concept path, which `run()` returns before reaching for every `mode: concept` lesson. So define-before-use is an AUTHOR discipline, not an enforced rule; the measured cost of assuming otherwise was 14 genuinely undefined first uses across 10 shipped days.

## Recap Rule

Every lesson ends with a one-page **recap unit** — the FINAL `@@@ concept` before the quiz, `tag="Recap"`. It gathers the day in a few beats (the arc, not new material), PLUS a reference cheat-sheet: a `%%% jargon` glossary and/or a `%%% table`, and its own recap visual. This is the "recap everything we learned today" the reader comes back to.

Checkable form: the reader_flow gate HARD-FAILS if the final concept is not a recap (recognized by a recap-ish `tag`/`title` — recap|summary|cheat-sheet|review|… — or a `%%% jargon` cheat-sheet in the last unit).

## Curiosity-before-Rigor Rule

First contact is wonder, grounded in a **concrete everyday analogy** — a physical thing the reader has done (drawing a face with a ruler, a light switch, a see-saw) — never a definition, a bare aspiration, or a prerequisites checklist. Ground in the everyday picture FIRST, then connect it to something exciting (e.g. "this is the trick behind ChatGPT").

Prerequisites read as "just curiosity."

## Staff-Depth-after-Orientation Rule

Failure modes, trade-offs, scale, and frontier relevance appear only after hook → picture → mechanism.

Frame depth as empowerment, not pressure. Every staff/interview claim traces to something taught earlier.

## V9 authoring essentials (full grammar in `sessions/_compiler/AUTHORING.md`)

Required front-matter for a `mode: concept` lesson:

```yaml
quest_id: <frozen id>       # replaces the donor __QUEST_ID__ token
mode: concept
donor: v9-base.donor
module_label: "…"           # KeyError if missing; drives sidebar group + hero kicker
title: "…"                  # KeyError if missing
subtitle: "…"               # KeyError if missing
brand_sub: "…"              # set it or the donor's old brand line leaks
spine: "<one word>"         # narrative-spine word; gate needs it in >=3 blocks
page_title / nav_prev_* / nav_next_* / fin_title / fin_body   # optional
notebook_yardstick: <path or null>   # null => Notebook Smoothness gate is N/A
```

Compile + iterate:

```bash
python3 sessions/_compiler/compile_lesson.py <source.md>              # exit 0 = pass
python3 sessions/_compiler/compile_lesson.py <source.md> --check-only # run gates, write nothing
# exit codes: 0 pass · 2 reader-flow fail (nothing written) · 3 shell/concept gate fail · 1 usage/parse error
```

Every concept unit must ship a real visual (`%%% svg` with a closed `<svg>…</svg>`, or
`%%% viz src=…` pointing at an EXISTING `sessions/viz/*.html`) or the gate FAILS. See
AUTHORING.md for the full block grammar, all `%%%` widgets, callouts, and a minimal
compiling example.
