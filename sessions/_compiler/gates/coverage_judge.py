#!/usr/bin/env python3
# =============================================================================
# Coverage Judge (v9.1)  — TIER-2, LLM-as-judge. ADVISORY.
# =============================================================================
# The deterministic coverage_gate (tier-1) can only substring-match: it cannot
# tell "genuinely TAUGHT" from "merely mentioned", cannot tell a concept from an
# analogy label, and cannot judge beginner TONE. This module is the authoritative
# TIER-2 judge: an LLM sub-agent (via the local keyless bridge) reads the lesson,
# the SKILL-DRAFTED spec, the notebook TEST-ORACLE concepts, and the manifest
# curation, and returns structured verdicts on three axes:
#
#   1. execution   — for each spec concept: TAUGHT / MENTIONED / ABSENT
#   2. skill_gaps  — notebook concepts the spec missed (not covered/deferred/oos)
#   3. intuition   — per concept: INTUITION_FIRST / FORMULA_FIRST / NO_ANALOGY
#                    (does it lead with a felt picture before notation? — the
#                     beginner register the notebook models)
#
# Routing mirrors tier-1 / frontier-refactor-qa:
#   MENTIONED/ABSENT (spec)      -> fix the lesson / builder
#   FORMULA_FIRST/NO_ANALOGY     -> fix the lesson / builder (Beginner Intuition Register)
#   skill_gap (notebook)         -> fix the ARCHITECT skill's Coverage Spec Rule,
#                                   re-draft covers, regenerate, re-eval
#
# It is ADVISORY: it never edits lesson.html and never fails a build. It DEGRADES
# GRACEFULLY: if the bridge is unreachable it returns status 'BRIDGE_UNAVAILABLE'
# and the caller falls back to tier-1 (coverage_gate).
#
# Bridge: http://localhost:11211/api/openai/v1 (keyless; reachable in-sandbox).
# Model:  aws:anthropic.claude-opus-4-8 (do NOT pass `temperature` — bridge 400s).
#
# Usage:
#   from coverage_judge import judge
#   result = judge(lesson_text, spec, notebook_concepts, curation, notebook_md=None)
# CLI:
#   python3 gates/coverage_judge.py <lesson.html> --source <source.md>
# =============================================================================
import sys, os, re, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import coverage_gate as cg   # reuse spec/notebook/curation extraction (single source)

ROOT = cg.ROOT
BRIDGE_URL = 'http://localhost:11211/api/openai/v1'
MODEL = 'aws:anthropic.claude-opus-4-8'
# 48000 chars (~12k tokens) comfortably holds a full V9 concept lesson; the older
# 22000 cap silently dropped the LAST ~third of longer lessons, so the final
# concept units (e.g. sigmoid-drift, XOR) were invisible to every judge below.
_MAX_LESSON_CHARS = 48000


def _readable_text(html, max_chars=None):
    """Strip HTML to READABLE prose for the LLM judges — PRESERVING punctuation,
    capitalization, and paragraph/line breaks. This is deliberately different from
    coverage_gate._strip_tags (which lowercases and deletes punctuation for substring
    matching): a tone judge must see real sentences, or it wrongly reads clean prose as
    a punctuation-free run-on. Block tags become newlines so ideas don't run together."""
    import html as _htmlmod
    h = re.sub(r'<(script|style)\b.*?</\1>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    h = re.sub(r'(?i)</(p|div|section|li|h[1-6]|tr|figure|figcaption|blockquote|table)\s*>', '\n', h)
    h = re.sub(r'(?i)<br\s*/?>', '\n', h)
    h = re.sub(r'(?i)</td>', ' | ', h)          # keep table cells scannable
    h = re.sub(r'<[^>]+>', '', h)               # remaining tags
    h = _htmlmod.unescape(h)                     # &amp; -> & etc.
    h = re.sub(r'[ \t]+', ' ', h)
    h = re.sub(r'\n[ \t]+', '\n', h)
    h = re.sub(r'\n{3,}', '\n\n', h).strip()
    return h[:max_chars] if max_chars else h

_SYS = (
    "You are a strict curriculum COVERAGE JUDGE for a beginner-facing ML lesson. "
    "The lesson is compiled from a skill-drafted coverage spec; a companion notebook "
    "is a held-out TEST of whether that spec is complete. You judge three axes and "
    "return STRICT JSON only (no prose, no markdown fences). Be skeptical: a concept "
    "named in one clause is MENTIONED, not TAUGHT. TAUGHT means the lesson gives its "
    "idea in plain words AND some mechanism/example/visual a beginner could follow."
)


def _prompt(lesson_text, spec, notebook_concepts, curation):
    spec_lines = '\n'.join('- %s' % t for t, _ in (spec or [])) or '(none authored)'
    nb_lines = '\n'.join('- %s' % t for t in (notebook_concepts or [])) or '(no notebook)'
    deferred = (curation or {}).get('deferred') or {}
    oos = (curation or {}).get('out_of_scope') or []
    defer_lines = '\n'.join('- %s -> %s' % (k, v) for k, v in deferred.items()) or '(none)'
    oos_lines = '\n'.join('- %s' % (t if isinstance(t, str) else t.get('topic', '')) for t in oos) or '(none)'
    return f"""SKILL-DRAFTED SPEC (concepts this lesson is meant to teach):
{spec_lines}

MANIFEST CURATION — deferred to another day (NOT gaps):
{defer_lines}
MANIFEST CURATION — out of scope (NOT gaps):
{oos_lines}

NOTEBOOK TEST-ORACLE CONCEPTS (what a good beginner treatment of this topic teaches;
ignore items that are analogy labels or pedagogical chrome, not real concepts):
{nb_lines}

LESSON TEXT (plain-text extract of the compiled lesson):
\"\"\"
{lesson_text[:_MAX_LESSON_CHARS]}
\"\"\"

Return STRICT JSON with exactly these keys:
{{
  "execution": [ {{"concept": "<spec concept>", "verdict": "TAUGHT|MENTIONED|ABSENT",
                   "evidence": "<short quote or where>"}} ],
  "skill_gaps": [ {{"concept": "<notebook concept absent from spec AND not covered AND not deferred/oos>",
                    "why": "<why it belongs in a beginner treatment of this topic>"}} ],
  "intuition":  [ {{"concept": "<spec concept>", "verdict": "INTUITION_FIRST|FORMULA_FIRST|NO_ANALOGY",
                    "note": "<one line>"}} ],
  "summary": "<=2 sentences"
}}
Rules: judge every spec concept in both "execution" and "intuition". Put a notebook
concept in "skill_gaps" ONLY if it is a genuine concept, is not TAUGHT/MENTIONED in the
lesson, AND is not in the deferred/out-of-scope lists. A concept that is the REMEDY for a
failure the lesson teaches (e.g. Leaky ReLU for dead ReLU) can never be out of scope."""


def judge(lesson_text, spec, notebook_concepts, curation=None, model=MODEL, timeout=90):
    """Call the LLM judge. Returns a dict:
      {status: 'OK'|'BRIDGE_UNAVAILABLE'|'PARSE_ERROR', execution, skill_gaps,
       intuition, summary, raw?}  — never raises."""
    try:
        from openai import OpenAI
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': 'openai sdk missing: %s' % e,
                'execution': [], 'skill_gaps': [], 'intuition': [], 'summary': ''}
    try:
        client = OpenAI(api_key='not-needed', base_url=BRIDGE_URL, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _SYS},
                      {'role': 'user', 'content': _prompt(lesson_text, spec, notebook_concepts, curation)}],
            # This judge emits THREE arrays at once — execution (one per spec concept),
            # skill_gaps (one per un-accounted notebook concept, which can be dozens), and
            # intuition (one per spec concept). On a full V9 lesson (12 spec concepts +
            # 20-30 notebook headings) that easily exceeds the 2000-token budget the other
            # three single-array judges use, so the response was truncated (finish_reason
            # 'length'), the JSON was cut off mid-object, and the whole call fell back to
            # tier-1 with a spurious PARSE_ERROR. Give it a much larger budget.
            max_tokens=8000,
        )
        content = (resp.choices[0].message.content or '').strip()
        finish = getattr(resp.choices[0], 'finish_reason', None)
    except Exception as e:  # APIConnectionError / APIStatusError / RateLimitError / ...
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'execution': [], 'skill_gaps': [], 'intuition': [], 'summary': ''}

    data = _extract_json(content)
    if data is None and finish == 'length':
        # Response was truncated mid-object; salvage the arrays that DID complete
        # (execution/intuition/skill_gaps entries are self-contained objects) instead
        # of discarding the whole verdict and falling back to tier-1.
        data = _salvage_truncated_json(content)
    if data is None:
        return {'status': 'PARSE_ERROR', 'raw': content,
                'execution': [], 'skill_gaps': [], 'intuition': [], 'summary': ''}
    data.setdefault('execution', [])
    data.setdefault('skill_gaps', [])
    data.setdefault('intuition', [])
    data.setdefault('summary', '')
    data['status'] = 'OK'
    return data


# ===========================================================================
# BEGINNER-FRIENDLINESS / TONE JUDGE — benchmarked against the notebook
# ===========================================================================
# Coverage ("is every concept present + taught") is necessary but NOT sufficient.
# A lesson can cover everything and still be cold, dense, and rush to the formula.
# This judge grades the lesson's TONE **relative to the companion notebook**, which
# is the gold standard for warm, analogy-first, intuition-heavy beginner teaching
# (repo CLAUDE.md §5/§7: analogy scaffold, one idea per sentence, curiosity hooks,
# normalize confusion, victory laps). It returns per-dimension deltas vs the notebook
# and concrete rewrite fixes. Advisory; graceful fallback like the coverage judge.
_TONE_MAX = 48000  # see _MAX_LESSON_CHARS note: hold the whole lesson + notebook
_TONE_SYS = (
    "You are a STRICT BEGINNER-FRIENDLINESS judge for an ML lesson aimed at a curious 12-year-old for "
    "whom English may be a second language. You are given a companion NOTEBOOK that is the GOLD "
    "STANDARD for warm, analogy-first, intuition-heavy teaching, and the LESSON under review. "
    "Grade the LESSON *relative to the notebook*, HARSHLY: it must be AS warm and analogy-rich as the "
    "notebook to score MATCHES. If it reads even somewhat colder, denser, more rushed, or more "
    "textbook/interview-like than the notebook, it is BELOW. Three beginner-mentality expectations weigh "
    "heavily (user directive 2026-07-20): (1) the OPENING must lead with a concrete EVERYDAY analogy before "
    "any aspirational claim; (2) MATH RESTRAINT — intuition, pictures, and analogies carry the lesson while "
    "heavy math sits in optional skippable boxes; (3) NO front-loaded jargon wall — terms are glossed inline "
    "as they arise and the cheat-sheet + a one-page recap come at the END. Be specific and quote. Return STRICT JSON only."
)


def _notebook_markdown(path, max_chars=_TONE_MAX):
    """Concatenate the notebook's markdown-cell prose (the tone gold standard)."""
    try:
        nb = json.load(open(path, encoding='utf-8'))
    except Exception:
        return ''
    out = []
    for c in nb.get('cells', []):
        if c.get('cell_type') == 'markdown':
            out.append(''.join(c.get('source', [])))
    return '\n\n'.join(out)[:max_chars]


def _tone_prompt(lesson_text, notebook_md):
    return f"""GOLD-STANDARD NOTEBOOK (the beginner tone to match — warm, analogy-first, intuition-heavy):
\"\"\"
{notebook_md[:_TONE_MAX]}
\"\"\"

LESSON UNDER REVIEW (plain-text extract):
\"\"\"
{lesson_text[:_TONE_MAX]}
\"\"\"

Judge the LESSON's beginner-friendliness RELATIVE to the notebook on each dimension below.
verdict is one of: MATCHES (as good as the notebook) / BELOW (noticeably colder/denser/rushed) /
WORSE (clearly textbook-like or confusing for a beginner). For each, give a one-line reason that
QUOTES or names a spot, and a concrete FIX.
Dimensions:
- warmth            (encouraging "brilliant friend" voice, not exam/textbook)
- analogy_quality   (a concrete everyday analogy per concept AND in the OPENING hook, WITH what it gets
                     right AND where it breaks down — the full analogy scaffold, not a one-word metaphor.
                     The lesson's very FIRST move should be a physical, experienced thing a 12-year-old
                     knows — drawing with a ruler, a light switch, a see-saw — BEFORE any abstract or
                     aspirational claim. A hero that opens on aspiration/relevance with no everyday analogy is BELOW)
- intuition_depth   (builds the felt idea slowly before/around the mechanism; doesn't sprint to notation)
- math_restraint    (leads with intuition + a picture + an analogy, NOT equations; any heavy or multi-step
                     math — derivations, matrix algebra, closed-form formulas — is DEMOTED into a clearly
                     labelled OPTIONAL / skippable box, and any formula kept in the main flow is single-line
                     and narrated in plain words. Too many equations, or a formula before the felt picture, is BELOW)
- plain_language    (short sentences, one idea each, no undefined jargon, no idioms)
- progressive_disclosure (terms are introduced define-before-use as they arise — via an inline [[term||gloss]]
                     glossary — NOT dumped as a big jargon WALL at the top; a reference cheat-sheet and a
                     one-page RECAP of the day live at the END for the reader to come back to)
- curiosity_encouragement (opening hook, normalizes confusion, victory laps / "you just unlocked…")
- pace              (lets ideas breathe like the notebook; not crammed)

Return STRICT JSON:
{{
  "dimensions": [ {{"name":"warmth","verdict":"MATCHES|BELOW|WORSE","reason":"...","fix":"..."}} ],
  "overall": "MATCHES_NOTEBOOK|BELOW_NOTEBOOK|WORSE_THAN_NOTEBOOK",
  "top_fixes": ["highest-leverage concrete rewrite 1", "..."],
  "summary": "<=2 sentences"
}}"""


def judge_tone(lesson_text, notebook_md, model=MODEL, timeout=90):
    """LLM beginner-friendliness judge, benchmarked against the notebook. Never raises."""
    if not notebook_md:
        return {'status': 'N/A', 'reason': 'no notebook to benchmark tone against',
                'dimensions': [], 'overall': 'N/A', 'top_fixes': [], 'summary': ''}
    try:
        from openai import OpenAI
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'dimensions': [], 'overall': 'N/A', 'top_fixes': [], 'summary': ''}
    try:
        client = OpenAI(api_key='not-needed', base_url=BRIDGE_URL, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _TONE_SYS},
                      {'role': 'user', 'content': _tone_prompt(lesson_text, notebook_md)}],
            max_tokens=2000,
        )
        content = (resp.choices[0].message.content or '').strip()
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'dimensions': [], 'overall': 'N/A', 'top_fixes': [], 'summary': ''}
    data = _extract_json(content)
    if data is None:
        return {'status': 'PARSE_ERROR', 'raw': content,
                'dimensions': [], 'overall': 'N/A', 'top_fixes': [], 'summary': ''}
    data.setdefault('dimensions', [])
    data.setdefault('overall', 'N/A')
    data.setdefault('top_fixes', [])
    data.setdefault('summary', '')
    data['status'] = 'OK'
    return data


# ===========================================================================
# INTEREST / CURIOSITY JUDGE — benchmarked against the notebook
# ===========================================================================
# Warmth and correct coverage are necessary but NOT sufficient. At this early
# stage the make-or-break question is whether the material makes a first-time
# learner WANT MORE — lean in, poke at it, come back tomorrow. A lesson can be
# accurate, friendly, and analogy-rich yet still be a dutiful slog that never
# sparks curiosity. The companion notebook is the gold standard for that PULL
# (user directive 2026-07-16: cultivate interest first; the notebook out-
# cultivates our lessons — port its levers: aspiration-first framing, relevance
# to things the reader has heard of, invitations to play, breadth as a buffet,
# genuine energy, low density). This judge grades the lesson's INTEREST relative
# to the notebook and returns per-lever deltas + concrete fixes. It judges SPARK
# ONLY — not correctness, coverage, or mere warmth. Advisory; graceful fallback.
_INTEREST_MAX = 48000  # see _MAX_LESSON_CHARS note: hold the whole lesson + notebook
_INTEREST_SYS = (
    "You are a STRICT INTEREST & CURIOSITY judge for a beginner-facing ML lesson (a curious 12-year-old for "
    "whom English may be a second language). At this early stage the single most important thing is whether "
    "the material makes a first-time learner WANT MORE — lean in, poke at it, come back tomorrow. You are "
    "given a companion NOTEBOOK that is the GOLD STANDARD for cultivating that pull, and the LESSON under "
    "review. Judge INTEREST / SPARK ONLY — NOT correctness, coverage, or mere warmth: a lesson can be "
    "accurate, friendly, and analogy-rich yet still be a dutiful slog that never sparks curiosity, and it "
    "must still score BELOW. Grade the LESSON *relative to the notebook*, HARSHLY: to score MATCHES it must "
    "make a beginner want to keep going AT LEAST as strongly as the notebook. If it is drier, denser, more "
    "dutiful, more problem/failure-focused, or less inviting to play than the notebook, it is BELOW. Credit "
    "the lesson's OWN ways of sparking interest (interactive drag/slide widgets, predict-then-run demos, "
    "hands-on 'produce' tasks) — you are judging want-more PULL, not style mimicry. Be specific and quote. "
    "Return STRICT JSON only (no prose, no markdown fences)."
)


def _interest_prompt(lesson_text, notebook_md):
    return f"""GOLD-STANDARD NOTEBOOK (the interest/curiosity bar — it makes a first-timer want to keep exploring):
\"\"\"
{notebook_md[:_INTEREST_MAX]}
\"\"\"

LESSON UNDER REVIEW (plain-text extract; %%%viz / %%%demo / @@@produce blocks are INTERACTIVE — runnable or
clickable in the real page, so a caption like "drag z / slide the line" means genuine hands-on play):
\"\"\"
{lesson_text[:_INTEREST_MAX]}
\"\"\"

Judge whether the LESSON cultivates a beginner's INTEREST as strongly as the notebook, on each lever below.
verdict is one of: MATCHES (sparks want-more as much as the notebook) / BELOW (noticeably drier, denser, or
more dutiful) / WORSE (a slog — a beginner would disengage). For each, give a one-line reason that QUOTES or
names a spot, and a concrete FIX.
Levers:
- aspiration_hook   (opens the lesson AND each concept by making it exciting / showing what it unlocks — an
                     invitation, NOT a problem or warning like "without this, depth is useless". BEST when the
                     hook is grounded in a concrete everyday thing the reader has physically experienced —
                     drawing a face with a ruler, flipping a light switch — BEFORE the aspirational payoff)
- relevance         (connects to things the reader has heard of or already cares about — e.g. "this is the
                     knob behind ChatGPT's creativity" — so they think "I want to know this")
- invites_play      (explicitly invites the reader to DO: predict-then-run a demo, drag/slide a widget, try
                     it — agency and play, not just reading)
- momentum          (light and skimmable; low cognitive load; ideas keep moving; NOT a dense wall or a long
                     slog through failure modes — and NOT a front-loaded glossary/jargon WALL or a pile of
                     equations the reader must wade through before the fun starts)
- breadth_spark     (conveys a rich landscape worth exploring — variety, personality, a tease of what's ahead
                     — vs one narrow corridor)
- delight_voice     (genuine enthusiasm and delight in the topic — "the secret ingredient!" energy — not
                     flat, neutral, or dutiful)
- payoff            (concrete wow-moments and victory laps that reward continuing and make the reader want
                     the NEXT unit — including a satisfying one-page RECAP at the end that gathers the day's wins)

Return STRICT JSON:
{{
  "dimensions": [ {{"name":"aspiration_hook","verdict":"MATCHES|BELOW|WORSE","reason":"...","fix":"..."}} ],
  "overall": "MATCHES_NOTEBOOK|BELOW_NOTEBOOK|WORSE_THAN_NOTEBOOK",
  "top_fixes": ["highest-leverage change to make it more compelling 1", "..."],
  "summary": "<=2 sentences"
}}"""


def judge_interest(lesson_text, notebook_md, model=MODEL, timeout=90):
    """LLM interest/curiosity judge, benchmarked against the notebook. Never raises."""
    if not notebook_md:
        return {'status': 'N/A', 'reason': 'no notebook to benchmark interest against',
                'dimensions': [], 'overall': 'N/A', 'top_fixes': [], 'summary': ''}
    try:
        from openai import OpenAI
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'dimensions': [], 'overall': 'N/A', 'top_fixes': [], 'summary': ''}
    try:
        client = OpenAI(api_key='not-needed', base_url=BRIDGE_URL, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _INTEREST_SYS},
                      {'role': 'user', 'content': _interest_prompt(lesson_text, notebook_md)}],
            max_tokens=2000,
        )
        content = (resp.choices[0].message.content or '').strip()
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'dimensions': [], 'overall': 'N/A', 'top_fixes': [], 'summary': ''}
    data = _extract_json(content)
    if data is None:
        return {'status': 'PARSE_ERROR', 'raw': content,
                'dimensions': [], 'overall': 'N/A', 'top_fixes': [], 'summary': ''}
    data.setdefault('dimensions', [])
    data.setdefault('overall', 'N/A')
    data.setdefault('top_fixes', [])
    data.setdefault('summary', '')
    data['status'] = 'OK'
    return data


# ===========================================================================
# CONCEPT-STRUCTURE JUDGE — per-concept intuition-first / analogy / build-up
# ===========================================================================
# The deterministic concept_structure_gate proves the triad is STRUCTURALLY
# present (prose -> visual -> prose). This judge grades whether the unit is
# intuition-first IN SPIRIT: leads with a felt picture, carries a real analogy
# WITH its "where it breaks down" half, and builds up step-by-step. Advisory,
# graceful fallback, never raises. Mirrors judge_tone.
_STRUCT_MAX = 48000  # see _MAX_LESSON_CHARS note: hold the whole lesson (all concepts)
_STRUCT_SYS = (
    "You are a STRICT CONCEPT-STRUCTURE judge for a BEGINNER ML lesson (a curious 12-year-old for whom "
    "English may be a second language) built as concept units. This is the make-or-break beginner-friendliness "
    "bar — grade HARSHLY and default to the LOWER grade when in doubt. For each named concept, judge whether "
    "the unit (1) leads with a felt picture / plain-words intuition BEFORE any formula, notation, or undefined "
    "jargon; (2) carries a CONCRETE, everyday, physically-experienced analogy INCLUDING an explicit 'where it "
    "breaks down' AND DRAWS that analogy — the concept's OPENING visual pictures the everyday THING itself (the "
    "valve, dimmer, see-saw, ruler, pizza, forecaster/surprise, assembly line, scorecard), not the equation or a "
    "bare axis/curve plot; a beginner should look and think 'oh, it's like a ___' before meeting any math; "
    "(3) builds up step-by-step rather than dumping the mechanism; and (4) VISUALIZES ITS BUILD-UP "
    "— a HEAVY build-up (a >=2-step derivation, a multi-step numeric worked example, a value/curve/shape that "
    "CHANGES across the explanation, matrix/vector algebra, or math demoted to an Optional-skippable box) must be "
    "SHOWN by a picture in the build-up region (a second figure that draws the transformation, a run-demo whose "
    "printed output IS the worked example, a drag/step viz, or a Math Ladder), not left as text + equations under "
    "only the opening analogy visual; humans read a graph far faster than a paragraph, so an all-prose+equations "
    "heavy build-up is at best WEAK, while a LIGHT/narrative build-up returns NA and is NEVER penalized. "
    "A generic, abstract, or one-word metaphor is NOT a concrete analogy. Be specific and quote. "
    "Return STRICT JSON only (no prose, no markdown fences)."
)


def _struct_prompt(lesson_text, concept_titles):
    names = '\n'.join('- %s' % c for c in (concept_titles or [])) or '(none)'
    return f"""CONCEPT UNITS TO JUDGE (by name/title):
{names}

LESSON TEXT (plain-text extract):
\"\"\"
{lesson_text[:_STRUCT_MAX]}
\"\"\"

For EACH concept unit above, return a verdict on three axes. verdict is one of:
GOOD (clearly meets it) / WEAK (partially) / MISSING (absent).
Return STRICT JSON:
{{
  "concepts": [ {{"concept":"<name>", "intuition_first":"GOOD|WEAK|MISSING",
                  "analogy":"GOOD|WEAK|MISSING", "buildup":"GOOD|WEAK|MISSING",
                  "buildup_visualized":"GOOD|WEAK|MISSING|NA",
                  "note":"<one line, quote a spot>", "fix":"<concrete rewrite if not GOOD>"}} ],
  "overall": "GOOD|WEAK|MISSING",
  "summary": "<=2 sentences"
}}
Rules: judge every concept, HARSHLY (default to the LOWER grade when in doubt).
"analogy": GOOD requires a CONCRETE everyday analogy (a physical/experienced thing a 12-year-old knows),
an explicit "where it breaks down", AND that the analogy is DRAWN — the concept's OPENING visual ILLUSTRATES
the everyday thing itself (the valve, dimmer, see-saw, ruler, pizza, forecaster/surprise, assembly line,
scorecard…), not the equation or a bare axis/curve plot. A concrete analogy carried only in WORDS while the
opening visual jumps straight to the math is WEAK. A generic/abstract/one-word metaphor, or an analogy with
no breakdown, is WEAK. No real analogy is MISSING.
"intuition_first": GOOD requires the unit to OPEN with a felt picture in plain words before any formula,
notation, or undefined jargon (and before any glossary/jargon wall). Leading with a definition/formula/notation is WEAK or MISSING.
"buildup": GOOD is a step-by-step build a beginner can follow; a dense dump — or heavy multi-step math
(derivations, matrix algebra, closed-form formulas) left in the main flow instead of a clearly-labelled
OPTIONAL / skippable box — is WEAK/MISSING.
"buildup_visualized": judge whether the build-up is DRAWN, not just written. FIRST decide if the build-up is
HEAVY (it derives/manipulates a formula in >=2 steps, walks a multi-step numeric worked example, shows a
value/curve/shape/distribution/boundary that CHANGES across the explanation, does matrix/vector algebra, or
sits in an Optional/skippable box). If HEAVY: GOOD = a visual in the build-up region (AFTER the opening
analogy figure) SHOWS the build-up itself — a second figure that draws the transformation, a run-demo whose
printed output IS the worked example, a drag/step viz, or a Math Ladder (words->formula->numbers->sanity).
WEAK = the only build-up support merely re-draws the opening analogy, or a formula sits with no supporting
picture, or the build-up is mostly prose + equations with a single opening figure. MISSING = the heavy
build-up is a wall of text/equations with NO build-up visual anywhere in the unit beyond the opening figure.
If the build-up is LIGHT (a single narrated one-line formula, or a purely definitional/narrative concept with
no math and no changing shape), return NA with note 'light build-up — opening visual suffices' and NEVER
penalize it. A bare formula callout that only restates an equation is NOT a build-up visual; a Math Ladder or
a predict-then-run demo IS."""


def judge_concept_structure(lesson_text, concept_titles, model=MODEL, timeout=90):
    """LLM per-concept structure judge. Never raises. N/A when no concepts given."""
    if not concept_titles:
        return {'status': 'N/A', 'reason': 'no concepts to judge',
                'concepts': [], 'overall': 'N/A', 'summary': ''}
    try:
        from openai import OpenAI
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'concepts': [], 'overall': 'N/A', 'summary': ''}
    try:
        client = OpenAI(api_key='not-needed', base_url=BRIDGE_URL, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _STRUCT_SYS},
                      {'role': 'user', 'content': _struct_prompt(lesson_text, concept_titles)}],
            # Per-concept objects (name + FOUR axes + note + fix) across ~13 concepts
            # overrun the old 2000-token budget once the 4th axis (buildup_visualized)
            # was added — the response truncated mid-object, _extract_json failed, and the
            # whole verdict fell back to PARSE_ERROR, SILENTLY disabling the structure gate.
            # Give it room + salvage the completed concept entries, exactly like judge().
            max_tokens=8000,
        )
        content = (resp.choices[0].message.content or '').strip()
        finish = getattr(resp.choices[0], 'finish_reason', None)
    except Exception as e:
        return {'status': 'BRIDGE_UNAVAILABLE', 'error': str(e),
                'concepts': [], 'overall': 'N/A', 'summary': ''}
    data = _extract_json(content)
    if data is None and finish == 'length':
        data = _salvage_truncated_json(content)
    if data is None:
        return {'status': 'PARSE_ERROR', 'raw': content,
                'concepts': [], 'overall': 'N/A', 'summary': ''}
    data.setdefault('concepts', [])
    data.setdefault('overall', 'N/A')
    data.setdefault('summary', '')
    data['status'] = 'OK'
    return data


def _extract_json(text):
    """Parse a JSON object from the model output (tolerant of stray fences/prose)."""
    if not text:
        return None
    text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)  # first balanced-ish object
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _salvage_truncated_json(text):
    """Best-effort recovery of a JSON object that was cut off mid-stream (the model
    hit max_tokens). Trims back to the last complete element (at any nesting depth),
    then closes every still-open bracket so the object parses. Returns a dict or None.

    This is deliberately conservative: it only ever DROPS the final, incomplete entry
    — it never invents data — so a coverage verdict truncated after N of M concepts
    still yields the N that completed instead of a total PARSE_ERROR fallback."""
    if not text:
        return None
    s = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()
    start = s.find('{')
    if start == -1:
        return None
    s = s[start:]
    # Walk once, tracking the bracket stack OUTSIDE of strings. Remember the position
    # just after any completed element (a closed `}`/`]`, or a comma that separates
    # two elements) together with the bracket stack AT THAT POINT — that is a place we
    # can safely trim to and then close whatever remained open.
    stack = []
    in_str = False
    esc = False
    best = None  # (index_to_trim_to, stack_snapshot)
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in '{[':
            stack.append('}' if ch == '{' else ']')
        elif ch in '}]':
            if stack:
                stack.pop()
            best = (i + 1, list(stack))          # just closed a complete value
        elif ch == ',' and stack and stack[-1] == ']':
            # A comma is a safe trim point only BETWEEN ARRAY ELEMENTS (innermost
            # open container is an array). A comma between object members would
            # leave a half-populated object, so it is NOT a trim point.
            best = (i, list(stack))
    if best is None:
        return None
    cut, open_stack = best
    head = s[:cut].rstrip().rstrip(',')
    repaired = head + ''.join(reversed(open_stack))
    try:
        return json.loads(repaired)
    except Exception:
        return None


def run_from_paths(lesson_html_path, source_path, root=ROOT):
    """Convenience: gather spec/curation/notebook via coverage_gate, then run ALL
    FOUR judges — coverage, beginner-friendliness (tone), interest/curiosity, and
    concept-structure. Returns {'coverage': ..., 'tone': ..., 'interest': ...,
    'structure': ...}."""
    from v8lib import split_frontmatter
    meta, _ = split_frontmatter(open(source_path, encoding='utf-8').read())
    html = open(lesson_html_path, encoding='utf-8').read()
    src_dir = os.path.dirname(os.path.abspath(source_path))

    day_key, cov_day = cg._load_manifest_curation(src_dir, root)
    deferred, oos = cg._curation_maps(None, cov_day)
    spec = cg._spec_from_frontmatter(meta) or cg._spec_from_manifest_covers(cov_day)

    nb_concepts, notebook_md = [], ''
    yard = meta.get('notebook_yardstick')
    if yard and str(yard).lower() not in ('null', 'none', ''):
        nb_path = os.path.join(root, str(yard))
        nb_concepts, _err = cg._notebook_concept_topics(nb_path)
        nb_concepts = nb_concepts or []
        notebook_md = _notebook_markdown(nb_path)

    lesson_text = _readable_text(html)   # real prose (punctuation + breaks) for the LLM judges
    curation = {'deferred': deferred, 'out_of_scope': list(oos.values())}
    # concept titles come from the source's @@@ concept title="..." args
    import re as _re
    src_text = open(source_path, encoding='utf-8').read()
    concept_titles = _re.findall(r'@@@\s+concept\b[^\n]*\btitle="([^"]+)"', src_text)
    return {'coverage': judge(lesson_text, spec, nb_concepts, curation),
            'tone': judge_tone(lesson_text, notebook_md),
            'interest': judge_interest(lesson_text, notebook_md),
            'structure': judge_concept_structure(lesson_text, concept_titles)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lesson')
    ap.add_argument('--source', required=True)
    a = ap.parse_args()
    out = run_from_paths(a.lesson, a.source)
    res, tone, struct = out['coverage'], out['tone'], out['structure']
    interest = out.get('interest', {'status': 'N/A', 'overall': 'N/A', 'dimensions': [],
                                    'top_fixes': [], 'summary': ''})
    print('== Coverage Judge (tier-2, advisory):', os.path.relpath(a.lesson), '==')
    print('status:', res['status'])
    if res['status'] != 'OK':
        print('  (falling back to tier-1 coverage_gate — bridge unavailable or unparseable)')
        if res.get('error'):
            print('  error:', res['error'])
    else:
        print('\n-- execution (spec concept -> genuinely taught?) --')
        for e in res['execution']:
            print('  [%s] %s — %s' % (e.get('verdict', '?'), e.get('concept', '?'), e.get('evidence', '')))
        print('\n-- intuition (leads with a felt picture before notation?) --')
        for e in res['intuition']:
            print('  [%s] %s — %s' % (e.get('verdict', '?'), e.get('concept', '?'), e.get('note', '')))
        print('\n-- skill gaps (notebook teaches; spec missed) --')
        if not res['skill_gaps']:
            print('  (none — spec covers the notebook)')
        for g in res['skill_gaps']:
            print('  ! %s — %s' % (g.get('concept', '?'), g.get('why', '')))
        print('\nsummary:', res['summary'])

    print('\n== Beginner-Friendliness Judge (vs notebook, advisory) ==')
    print('status:', tone['status'], '| overall:', tone.get('overall'))
    if tone['status'] == 'OK':
        print('\n-- tone dimensions (lesson vs the notebook gold standard) --')
        for d in tone['dimensions']:
            print('  [%s] %s — %s' % (d.get('verdict', '?'), d.get('name', '?'), d.get('reason', '')))
            if d.get('fix'):
                print('        fix: %s' % d['fix'])
        print('\n-- top fixes --')
        for f in tone['top_fixes']:
            print('  * %s' % f)
        print('\nsummary:', tone['summary'])
    elif tone.get('error'):
        print('  error:', tone['error'])

    print('\n== Interest / Curiosity Judge (vs notebook, advisory) ==')
    print('status:', interest['status'], '| overall:', interest.get('overall'))
    if interest['status'] == 'OK':
        print('\n-- interest levers (lesson vs the notebook interest bar) --')
        for d in interest['dimensions']:
            print('  [%s] %s — %s' % (d.get('verdict', '?'), d.get('name', '?'), d.get('reason', '')))
            if d.get('fix'):
                print('        fix: %s' % d['fix'])
        print('\n-- top fixes --')
        for f in interest['top_fixes']:
            print('  * %s' % f)
        print('\nsummary:', interest['summary'])
    elif interest.get('error'):
        print('  error:', interest['error'])

    print('\n== Concept-Structure Judge (per concept, advisory) ==')
    print('status:', struct['status'], '| overall:', struct.get('overall'))
    if struct['status'] == 'OK':
        print('\n-- per concept (intuition_first / analogy / buildup / buildup_visualized) --')
        for c in struct['concepts']:
            print('  [if:%s an:%s bu:%s bv:%s] %s — %s' % (
                c.get('intuition_first', '?'), c.get('analogy', '?'), c.get('buildup', '?'),
                c.get('buildup_visualized', '?'), c.get('concept', '?'), c.get('note', '')))
        print('\nsummary:', struct['summary'])
    elif struct.get('error'):
        print('  error:', struct['error'])
    sys.exit(0)


if __name__ == '__main__':
    main()
