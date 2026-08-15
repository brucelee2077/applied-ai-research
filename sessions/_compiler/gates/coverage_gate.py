#!/usr/bin/env python3
# =============================================================================
# Coverage Gate (v9.1)  — ADVISORY. Never fails the build.
# =============================================================================
# SKILLS ARE THE SOURCE OF TRUTH FOR COVERAGE. THE NOTEBOOK IS A TEST ORACLE.
#
# The curriculum skills draft what a lesson must cover (the manifest
# `coverage.<day>.covers`, or an in-source `coverage_topics`) from domain
# knowledge — NOT by reading a notebook. Many topics (e.g. JAX) have no notebook
# at all, so the notebook can never be the thing that DEFINES required coverage.
# Instead, where a notebook exists it is a held-out TEST that scores whether the
# skill's spec was complete. This gate runs two independent checks:
#
#   CHECK A — EXECUTION (always; needs no notebook)
#     Did the lesson realize the SKILL-DRAFTED spec? For each spec topic:
#     COVERED / DEFERRED / EXEC-GAP. An EXEC-GAP means the builder failed to
#     render a spec item -> fix the lesson / builder skill.
#
#   CHECK B — SKILL EVAL vs TEST ORACLE (only where a notebook exists)
#     Does the notebook teach a concept the SKILL'S SPEC never listed (and the
#     manifest never deferred / scoped out)? That is a SKILL-GAP: the skill's
#     coverage-derivation missed a concept the world considers part of this
#     topic -> fix the ARCHITECT skill, re-draft the spec, regenerate, re-eval.
#     Where there is no notebook (JAX), check B is N/A — the skill is trusted
#     there BECAUSE it was validated by check B on topics that do have notebooks.
#
# The whole point: the notebook must NOT be able to self-certify coverage, and a
# hand-authored `covers` list must NOT be able to silently suppress a notebook
# concept. Check A grades the lesson against the spec; check B grades the SPEC
# against the notebook. Neither ever blocks the build (the caller must not
# sys.exit on the returned status); both are reported and route to a fix.
#
# Signature mirrors the other gates:
#   from coverage_gate import run ; status, msgs = run(html, meta, root=ROOT,
#                                                       source_dir=None, curation=None)
# CLI:
#   python3 gates/coverage_gate.py <lesson.html> --source <source.md>
#   (always exit 0 — advisory)
#
# Pure + deterministic. No LLM — this is the fast TIER-1 pre-filter. The
# authoritative, semantic coverage judgment ("is this concept genuinely TAUGHT,
# not just mentioned?") is the TIER-2 LLM judge in coverage_judge.py, which the
# skill-defined Coverage Review Workflow runs as a sub-agent. Tier-1 never needs
# the bridge, so it always works offline and in CI; tier-2 adds semantic judgment.
# =============================================================================
import sys, os, re, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Generic notebook chrome — pedagogical-structure headings that name no concept.
# Dropped from the check-B probe set so the skill eval is about MISSING CONCEPTS,
# not wording differences. Concept-sounding headings (e.g. "Weight Initialization
# Guide") are kept and cleared by the manifest curation, not by this filter.
_CHROME = {
    'what you ll learn', 'what you will learn', 'jargon buster', 'the problem',
    'the problem why we need activation functions', 'the linear limitation',
    'what happens when we stack linear functions', 'visualizing the problem',
    'analogies', 'analogies understanding activation functions', 'when to use this',
    'when to use it', 'comparative visualization', 'comparative visualization all activation functions',
    'key takeaways', 'quick decision guide', 'quick decision guide choosing the right activation function',
    'how to choose your activation function', 'common mistakes to avoid', 'common mistakes',
    'helpful tips', 'debugging checklist', 'what s next', 'coming up',
    'key insights to remember', 'key insights', 'practice exercises', 'prerequisites',
    'setup', 'imports', 'summary', 'conclusion', 'observations', 'overview',
    'variants to know', 'things to know', 'pros', 'cons',
    'signal processing', 'key property', 'modern best practices', 'formula',
}


# ---------------------------------------------------------------------------
# helpers — all pure
# ---------------------------------------------------------------------------
def _norm(s):
    """Lowercase, collapse whitespace/punctuation to single spaces.

    The keep-set includes Han. With `[^a-z0-9]+` it did not, so every Chinese
    character was DELETED from the covered-text: measured, _norm('注意力 attention 头')
    returned just 'attention'. On a Chinese lesson the covered-text collapsed to the
    English terms alone and tier-1 coverage reported every spec concept as an
    A/EXEC-GAP. Widening a keep-set is purely additive — it can only ever ADD
    matches, so no English verdict can change.
    """
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9㐀-䶿一-鿿]+', ' ', str(s).lower())).strip()


def _strip_tags(html):
    """Compiled lesson HTML -> normalized lowercased plain text (the covered-text)."""
    # drop <script>/<style> bodies so their identifiers don't create phantom coverage
    html = re.sub(r'<(script|style)\b.*?</\1>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = (text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                .replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' '))
    return _norm(text)


def _parse_checklist(items):
    """Normalize a spec list into [(topic, [keywords]), ...].
    Items may be plain strings or {topic, keywords?} mappings. Shared by the
    front-matter `coverage_topics` and the manifest `coverage.<day>.covers`."""
    if not items:
        return None
    out = []
    for item in items:
        if isinstance(item, dict):
            topic = str(item.get('topic', '')).strip()
            if not topic:
                continue
            kws = item.get('keywords') or [topic]
            if isinstance(kws, str):
                kws = [kws]
            out.append((topic, [str(k) for k in kws]))
        else:
            out.append((str(item), [str(item)]))
    return out or None


def _spec_from_frontmatter(meta):
    """SKILL-DRAFTED spec, in-source form: coverage_topics: [ {topic,keywords?} | str ]."""
    return _parse_checklist(meta.get('coverage_topics'))


def _spec_from_manifest_covers(cov_day):
    """SKILL-DRAFTED spec, manifest form: coverage.<day>.covers (str or {topic,keywords})."""
    return _parse_checklist((cov_day or {}).get('covers'))


def _notebook_concept_topics(path):
    """TEST-ORACLE probe set: concept names the notebook teaches. Pulled from
    markdown headings (level >=2 only — the level-1 doc title is not a concept) +
    glossary/table bold terms, with numbering + emoji + chrome + "Label:" lines
    stripped, and compound cells ("Sigmoid / Tanh") split into parts. Deterministic,
    no LLM (tier-1 pre-filter; the LLM judge in coverage_judge.py is authoritative).
    Returns (topics, err)."""
    try:
        nb = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        return None, 'could not read notebook: %s' % e
    topics, seen = [], set()

    def _emit(name):
        title = re.sub(r'\([^)]*\)', '', str(name))          # drop "(Smooth Activation)" descriptors
        title = re.sub(r'^[^\w]+', '', title).strip()        # drop leading emoji/symbols
        key = _norm(title)
        if not title or not key or key in _CHROME or len(key.split()) > 6 or key in seen:
            return
        seen.add(key)
        topics.append(title)

    def _add(raw):
        if not raw:
            return
        full = re.sub(r'^[^\w]+', '', str(raw)).strip()      # strip leading emoji/symbols
        full = re.sub(r'#\s*\d+', '', full).strip()          # drop "#5" numbering
        if _norm(full) in _CHROME:                           # chrome check on the WHOLE heading
            return
        concept = full
        if ':' in concept:                                   # "X #1: Y" / "Label:" -> RHS
            rhs = concept.split(':', 1)[1].strip()
            if not rhs:                                       # "Label:" (empty RHS) -> skip
                return
            concept = rhs
        concept = re.sub(r'^visualizing\s+', '', concept, flags=re.I).strip()
        # de-compound "Sigmoid / Tanh", "ReLU, Leaky ReLU", "A vs B", "A and B" -> parts
        for part in re.split(r'\s*(?:/|&|,|\bvs\b|\band\b)\s*', concept):
            _emit(part)

    for c in nb.get('cells', []):
        if c.get('cell_type') != 'markdown':
            continue
        for ln in ''.join(c.get('source', [])).split('\n'):
            s = ln.strip()
            if s.startswith('##'):                            # level >=2 headings only
                _add(s.lstrip('#').strip())
            elif s.startswith('**') and s.endswith('**') and len(s) > 4:
                _add(s.strip('*').strip())
            elif s.startswith('|'):
                # glossary / comparison table row -> first bolded cell is the term
                m = re.search(r'\*\*(.+?)\*\*', s)
                if m:
                    _add(m.group(1))
    return topics, None


def _load_manifest_curation(source_dir, root=ROOT):
    """Find <repo>/sessions/<module>/_refactor/manifest.yaml from a day source_dir,
    return (day_key, coverage_day_dict) or (None, None). Pure read; no LLM."""
    if not source_dir or yaml is None:
        return None, None
    src = os.path.abspath(source_dir)
    day_key = os.path.basename(src.rstrip(os.sep))
    module_dir = os.path.dirname(src.rstrip(os.sep))
    manifest_path = os.path.join(module_dir, '_refactor', 'manifest.yaml')
    if not os.path.exists(manifest_path):
        return None, None
    try:
        data = yaml.safe_load(open(manifest_path, encoding='utf-8').read()) or {}
    except Exception:
        return None, None
    cov = (data.get('coverage') or {})
    return day_key, cov.get(day_key)


def _curation_maps(curation, cov_day):
    """Merge an injected curation dict (tests) with the manifest coverage day.
    Returns (deferred: {norm_topic: where}, out_of_scope: {norm_topic: reason})."""
    deferred, oos = {}, {}

    def _ingest(d):
        if not d:
            return
        for k, v in (d.get('deferred') or {}).items():
            deferred[_norm(k)] = str(v)
        for t in (d.get('out_of_scope') or []):
            # out_of_scope items may be a bare string or {topic, reason}
            if isinstance(t, dict):
                oos[_norm(t.get('topic', ''))] = str(t.get('reason', '')) or str(t.get('topic', ''))
            else:
                oos[_norm(t)] = str(t)

    _ingest(cov_day)       # manifest first
    _ingest(curation)      # explicit param overrides / augments
    return deferred, oos


def _covered(text, keywords):
    """A topic is covered if any normalized keyword is a substring of the lesson text.
    Keywords are author-chosen (deliberate), so a plain substring match is intended."""
    return any(_norm(k) and _norm(k) in text for k in keywords)


def _phrase_in(phrase, text):
    """Whole-phrase membership in normalized text (word-boundary safe).
    Guards the check-B text fallback against short-token false hits — e.g. 'elu'
    must NOT be judged 'covered' just because the text contains 'relu'."""
    p = _norm(phrase)
    return bool(p) and (' ' + p + ' ') in (' ' + text + ' ')


def _names_at_least(container, contained):
    """True iff `container` names at least as much as `contained` — the contained
    concept's tokens are a subset of the container's. Whole-CONCEPT match, so
    'relu' does NOT satisfy 'leaky relu' (they are different concepts) while
    'leaky relu' DOES satisfy 'relu'. This is what stops a general spec entry
    from silently accounting for a more specific notebook concept."""
    c = set(_norm(container).split())
    d = set(_norm(contained).split())
    return bool(d) and d <= c


def _spec_covers_nb(spec_phrase, nb_norm):
    """Does a spec entry account for a notebook concept? Two safe directions:
      (a) nb ⊆ spec         — spec names at least the whole nb concept, OR
      (b) spec ⊆ nb AND spec is a COMPOUND (>=2 tokens) — spec is a specific named
          concept sitting inside a longer heading (e.g. spec 'leaky relu' inside
          'Leaky ReLU with Different Alphas').
    A 1-token spec ('relu') is NEVER matched by direction (b), so it cannot swallow
    'leaky relu'. Short-token traps ('elu' ⊂ 'relu') are impossible: token sets differ."""
    sp = set(_norm(spec_phrase).split())
    nb = set(nb_norm.split())
    if not sp or not nb:
        return False
    if nb <= sp:
        return True
    return len(sp) >= 2 and sp <= nb


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def run(html, meta, root=ROOT, source_dir=None, curation=None):
    """Advisory two-check coverage comparison. Returns (status, msgs).
    status in {'N/A','PASS','ADVISORY'} — informational only; NEVER a build failure.

    Check A (execution): lesson vs the SKILL-DRAFTED spec (coverage_topics /
    manifest covers). Check B (skill eval): notebook concepts vs the spec+curation
    — a notebook concept the spec never listed/deferred/scoped-out is a SKILL-GAP.

    `curation` (optional) = {'deferred': {topic: where}, 'out_of_scope': [topic,...]}
    injected directly (tests); when absent the module manifest is read from source_dir."""
    msgs = []
    day_key, cov_day = _load_manifest_curation(source_dir, root)
    deferred, oos = _curation_maps(curation, cov_day)

    # -- the EXPECTED coverage is the SKILL-DRAFTED spec, never the notebook --
    spec = _spec_from_frontmatter(meta)
    spec_src = 'front-matter coverage_topics'
    if spec is None:
        spec = _spec_from_manifest_covers(cov_day)
        spec_src = 'manifest coverage.%s.covers' % (day_key or '?')

    yard = meta.get('notebook_yardstick')
    have_yard = bool(yard) and str(yard).lower() not in ('null', 'none', '')

    if spec is None and not have_yard:
        return 'N/A', ['N/A — no skill-drafted spec (coverage_topics / manifest covers) and no notebook']

    text = _strip_tags(html)
    exec_gaps, skill_gaps = [], []

    # ---------------- CHECK A — execution (lesson realizes the spec) ----------------
    if spec is not None:
        msgs.append('note A/execution — spec source: %s (%d topics)' % (spec_src, len(spec)))
        for topic, kws in spec:
            ntopic = _norm(topic)
            if _covered(text, kws):
                msgs.append('A/covered: %s' % topic)
            elif ntopic in deferred:
                msgs.append('A/defer: %s -> %s' % (topic, deferred[ntopic]))
            else:
                exec_gaps.append(topic)
                msgs.append('A/EXEC-GAP: %s (spec says teach it; not in lesson) -> fix lesson/builder' % topic)
    else:
        msgs.append('note A/execution — SKIPPED: no skill-drafted spec authored '
                    '(no coverage_topics, no manifest covers) -> fix architect skill to draft one')

    # ---------------- CHECK B — skill eval vs the notebook TEST ORACLE ----------------
    if not have_yard:
        msgs.append('note B/skill-eval — N/A: no notebook oracle (skill trusted here; '
                    'validated by check B where notebooks exist)')
    else:
        nb_topics, err = _notebook_concept_topics(os.path.join(root, str(yard)))
        if err:
            msgs.append('note B/skill-eval — N/A: %s' % err)
        elif not nb_topics:
            msgs.append('note B/skill-eval — N/A: no concept topics extracted from notebook')
        else:
            # spec keyword phrases (a notebook concept is "in the spec" only if some
            # spec entry names AT LEAST as much — nb tokens ⊆ spec-keyword tokens).
            spec_phrases = set()
            for _t, kws in (spec or []):
                for k in kws:
                    if _norm(k):
                        spec_phrases.add(_norm(k))
                spec_phrases.add(_norm(_t))
            defer_keys = set(deferred.keys())
            oos_keys = set(oos.keys())
            accounted = 0
            for nb in nb_topics:
                nb_norm = _norm(nb)
                if any(_spec_covers_nb(k, nb_norm) for k in spec_phrases):
                    accounted += 1                        # the spec names this concept
                elif _phrase_in(nb_norm, text):
                    accounted += 1                        # covered in the lesson as a whole phrase
                elif any(_names_at_least(nb_norm, dk) for dk in defer_keys):
                    accounted += 1                        # a deferred key covers this concept
                elif any(_names_at_least(nb_norm, ok) for ok in oos_keys):
                    accounted += 1                        # an out-of-scope key covers this concept
                else:
                    skill_gaps.append(nb)
            msgs.append('note B/skill-eval — %d notebook concept-topics; %d accounted (spec/lesson/curated)'
                        % (len(nb_topics), accounted))
            for g in skill_gaps:
                msgs.append('B/SKILL-GAP: %s (notebook teaches it; skill spec missed it) '
                            '-> fix architect skill, re-draft covers, regenerate, re-eval' % g)

    # ---------------- verdict (advisory) ----------------
    status = 'PASS' if not exec_gaps and not skill_gaps else 'ADVISORY'
    if status == 'PASS':
        msgs.append('=> PASS: lesson realizes the spec (check A) and the spec covers the notebook (check B)')
    else:
        parts = []
        if exec_gaps:
            parts.append('%d execution gap(s): %s' % (len(exec_gaps), ', '.join(exec_gaps)))
        if skill_gaps:
            parts.append('%d skill gap(s): %s' % (len(skill_gaps), ', '.join(skill_gaps)))
        msgs.append('=> ADVISORY: ' + ' | '.join(parts))

    if source_dir:
        _write_sidecar(source_dir, spec_src if spec is not None else None, spec, text,
                       deferred, oos, day_key, yard if have_yard else None, exec_gaps, skill_gaps, root)

    return status, msgs


def _write_sidecar(source_dir, spec_src, spec, text, deferred, oos, day_key,
                   yard, exec_gaps, skill_gaps, root=ROOT):
    path = os.path.join(source_dir, '_coverage.md')
    lines = ['# Coverage (advisory)',
             '',
             '_Generated by coverage_gate.py — advisory only. Never modifies lesson.html._',
             '_Skills are the source of truth for coverage; the notebook is a held-out test oracle._',
             '',
             '- Spec source (skill-drafted): %s' % (spec_src or '(none authored)'),
             '- Notebook test oracle: %s' % (yard or '(none — skill trusted here)'),
             '- Curation day: %s' % (day_key or '(none)'),
             '',
             '## Check A — execution (lesson realizes the skill-drafted spec)',
             '',
             '| spec topic | status | note |',
             '|------------|--------|------|']
    if spec is not None:
        for topic, kws in spec:
            ntopic = _norm(topic)
            if _covered(text, kws):
                st, note = 'COVERED', ''
            elif ntopic in deferred:
                st, note = 'DEFERRED', deferred[ntopic]
            else:
                st, note = 'EXEC-GAP', 'spec says teach it; not in lesson -> fix lesson/builder'
            safe = str(topic).replace('|', '\\|')
            lines.append('| %s | %s | %s |' % (safe, st, note))
    else:
        lines.append('| (none) | NO-SPEC | no coverage_topics / manifest covers -> fix architect skill |')

    lines += ['', '## Check B — skill eval (notebook concepts vs the skill-drafted spec)', '']
    if not yard:
        lines.append('_N/A — no notebook oracle. The skill is trusted here; it is validated by '
                     'check B on topics that do have a notebook._')
    else:
        nb_topics, err = _notebook_concept_topics(os.path.join(root, str(yard)))
        if err or not nb_topics:
            lines.append('_N/A — %s_' % (err or 'no concept topics extracted'))
        else:
            gapset = {_norm(g) for g in skill_gaps}
            lines += ['| notebook concept | in spec/lesson/curated? |', '|------------------|------------------------|']
            for nb in nb_topics:
                mark = 'SKILL-GAP -> fix skill' if _norm(nb) in gapset else 'accounted'
                safe = str(nb).replace('|', '\\|')
                lines.append('| %s | %s |' % (safe, mark))

    verdict = 'PASS' if not exec_gaps and not skill_gaps else 'ADVISORY'
    lines += ['', '## Verdict: %s' % verdict]
    if exec_gaps:
        lines.append('- Execution gaps (fix lesson/builder): %s' % ', '.join(exec_gaps))
    if skill_gaps:
        lines.append('- Skill gaps (fix architect skill, re-draft, regenerate, re-eval): %s'
                     % ', '.join(skill_gaps))
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    except Exception:
        pass  # sidecar is best-effort; never fail the build over it


# ---------------------------------------------------------------------------
# CLI  (always exit 0 — advisory)
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('lesson')
    ap.add_argument('--source', required=True)
    a = ap.parse_args()
    from v8lib import split_frontmatter
    meta, _ = split_frontmatter(open(a.source, encoding='utf-8').read())
    html = open(a.lesson, encoding='utf-8').read()
    src_dir = os.path.dirname(os.path.abspath(a.source))
    status, msgs = run(html, meta, source_dir=src_dir)
    print('== Coverage Gate (advisory):', os.path.relpath(a.lesson), '==')
    for m in msgs:
        print('  ', m)
    print('\n' + str(status) + '  (advisory — exit 0 regardless)')
    sys.exit(0)


if __name__ == '__main__':
    main()
