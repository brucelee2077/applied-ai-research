#!/usr/bin/env python3
# =============================================================================
# Coverage Gate (v9)  — ADVISORY. Never fails the build.
# =============================================================================
# Compares a COMPILED lesson to its notebook yardstick on the COVERAGE axis:
# does the lesson cover the topics the notebook teaches? This closes the
# build->evaluate feedback loop. A prior held-out audit (m02) proved that most
# notebook "gaps" are legitimate CURATION (a topic belongs on another day, or is
# out of scope), so this gate is advisory: it classifies and reports, it never
# blocks the build and the caller must NOT sys.exit on its status.
#
# Three findings routes (documented in frontier-refactor-qa):
#   (a) real in-scope gap        -> GAP        -> fix the lesson (add a unit/visual)
#   (b) systemic/format defect   -> (recurs)   -> fix the relevant skill
#   (c) legitimate curation      -> DEFERRED / OUT_OF_SCOPE (recorded in manifest)
#
# The deterministic diff (build checklist -> substring-match -> classify against
# the manifest's curation lists) is pure and reproducible. It does NOT call any
# LLM. A future opt-in LLM adjudication layer would plug in at step 4 (the
# COVERED/MISSING decision) — it is intentionally NOT built here.
#
# Signature mirrors the other gates:
#   from coverage_gate import run ; status, msgs = run(html, meta, root=ROOT,
#                                                       source_dir=None, curation=None)
# CLI:
#   python3 gates/coverage_gate.py <lesson.html> --source <source.md>
#   (always exit 0 — advisory)
# =============================================================================
import sys, os, re, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


# ---------------------------------------------------------------------------
# helpers — all pure
# ---------------------------------------------------------------------------
def _norm(s):
    """Lowercase, collapse whitespace/punctuation to single spaces."""
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', str(s).lower())).strip()


def _strip_tags(html):
    """Compiled lesson HTML -> normalized lowercased plain text (the covered-text)."""
    # drop <script>/<style> bodies so their identifiers don't create phantom coverage
    html = re.sub(r'<(script|style)\b.*?</\1>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', html)
    text = (text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                .replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' '))
    return _norm(text)


def _checklist_from_frontmatter(meta):
    """Author-declared checklist: coverage_topics: [ {topic, keywords?} | str, ... ].
    Returns list of (topic, [keywords]) or None if absent."""
    ct = meta.get('coverage_topics')
    if not ct:
        return None
    out = []
    for item in ct:
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


def _checklist_from_manifest_covers(cov_day):
    """The manifest's coverage.<day>.covers list is an author-declared checklist too."""
    covers = (cov_day or {}).get('covers')
    if not covers:
        return None
    return [(str(t), [str(t)]) for t in covers]


def _checklist_from_notebook(path):
    """Deterministic fallback: candidate topics from notebook markdown HEADINGS.
    Lines starting with #/##/### and bold **...** section titles. No LLM."""
    try:
        nb = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        return None, 'could not read notebook: %s' % e
    topics, seen = [], set()
    for c in nb.get('cells', []):
        if c.get('cell_type') != 'markdown':
            continue
        for ln in ''.join(c.get('source', [])).split('\n'):
            s = ln.strip()
            title = None
            if s.startswith('#'):
                title = s.lstrip('#').strip()
            elif s.startswith('**') and s.endswith('**') and len(s) > 4:
                title = s.strip('*').strip()
            if not title:
                continue
            # drop leading emoji / symbols so the topic text is clean
            title = re.sub(r'^[^\w]+', '', title).strip()
            key = _norm(title)
            if title and key and key not in seen:
                seen.add(key)
                topics.append((title, [title]))
    return topics, None


def _load_manifest_curation(source_dir, root=ROOT):
    """Find <repo>/sessions/<module>/_refactor/manifest.yaml from a day source_dir,
    return (day_key, coverage_day_dict) or (None, None). Pure read; no LLM."""
    if not source_dir or yaml is None:
        return None, None
    src = os.path.abspath(source_dir)
    day_key = os.path.basename(src.rstrip(os.sep))
    # module dir is the parent of the day dir
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
    Returns (deferred: {norm_topic: where}, out_of_scope: {norm_topic: raw})."""
    deferred, oos = {}, {}

    def _ingest(d):
        if not d:
            return
        for k, v in (d.get('deferred') or {}).items():
            deferred[_norm(k)] = str(v)
        for t in (d.get('out_of_scope') or []):
            oos[_norm(t)] = str(t)

    _ingest(cov_day)       # manifest first
    _ingest(curation)      # explicit param overrides / augments
    return deferred, oos


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def run(html, meta, root=ROOT, source_dir=None, curation=None):
    """Advisory coverage comparison. Returns (status, msgs).
    status in {'N/A','PASS','ADVISORY'} — informational only; NEVER a build failure.
    `curation` (optional) = {'deferred': {topic: where}, 'out_of_scope': [topic,...]}
    injected directly (tests); when absent the module manifest is read from source_dir."""
    yard = meta.get('notebook_yardstick')
    msgs = []
    if not yard or str(yard).lower() in ('null', 'none', ''):
        return 'N/A', ['N/A — no notebook yardstick']

    # -- curation: manifest coverage day (+ optional injected override) --
    day_key, cov_day = _load_manifest_curation(source_dir, root)
    deferred, oos = _curation_maps(curation, cov_day)

    # -- build the checklist (deterministic) --
    checklist = _checklist_from_frontmatter(meta)
    checklist_src = 'front-matter coverage_topics'
    if checklist is None:
        checklist = _checklist_from_manifest_covers(cov_day)
        checklist_src = 'manifest coverage.%s.covers' % (day_key or '?')
    if checklist is None:
        checklist, err = _checklist_from_notebook(os.path.join(root, str(yard)))
        checklist_src = 'notebook headings (%s)' % os.path.basename(str(yard))
        if err:
            return 'N/A', ['N/A — ' + err]
    if not checklist:
        return 'N/A', ['N/A — no checklist could be built (empty notebook/front-matter)']

    msgs.append('note checklist source: %s (%d topics)' % (checklist_src, len(checklist)))
    if day_key:
        msgs.append('note curation from manifest day %s: %d deferred, %d out-of-scope'
                    % (day_key, len(deferred), len(oos)))

    # -- covered-text from the compiled lesson --
    text = _strip_tags(html)

    gaps = []
    for topic, keywords in checklist:
        # step 4: COVERED if any normalized keyword substring appears in the lesson.
        # (LLM adjudication of near-misses would replace this decision in future.)
        covered = any(_norm(k) and _norm(k) in text for k in keywords)
        ntopic = _norm(topic)
        if covered:
            msgs.append('pass covered: %s' % topic)
        elif ntopic in deferred:
            msgs.append('defer %s -> %s' % (topic, deferred[ntopic]))
        elif ntopic in oos:
            msgs.append('oos %s' % topic)
        else:
            gaps.append(topic)
            msgs.append('GAP %s (in notebook, not lesson, not curated)' % topic)

    status = 'PASS' if not gaps else 'ADVISORY'
    if gaps:
        msgs.append('=> ADVISORY: %d un-curated gap(s): %s' % (len(gaps), ', '.join(gaps)))
    else:
        msgs.append('=> PASS: every checklist topic covered or curated')

    # -- advisory sidecar report (never touches lesson.html) --
    if source_dir:
        _write_sidecar(source_dir, checklist_src, checklist, text, deferred, oos, day_key)

    return status, msgs


def _classify(topic, keywords, text, deferred, oos):
    ntopic = _norm(topic)
    if any(_norm(k) and _norm(k) in text for k in keywords):
        return 'COVERED', ''
    if ntopic in deferred:
        return 'DEFERRED', deferred[ntopic]
    if ntopic in oos:
        return 'OUT_OF_SCOPE', ''
    return 'GAP', 'in notebook, not lesson, not curated'


def _write_sidecar(source_dir, checklist_src, checklist, text, deferred, oos, day_key):
    path = os.path.join(source_dir, '_coverage.md')
    lines = ['# Coverage (advisory)',
             '',
             '_Generated by coverage_gate.py — advisory only. Never modifies lesson.html._',
             '',
             '- Checklist source: %s' % checklist_src,
             '- Curation day: %s' % (day_key or '(none)'),
             '',
             '| topic | status | note |',
             '|-------|--------|------|']
    for topic, keywords in checklist:
        status, note = _classify(topic, keywords, text, deferred, oos)
        safe = str(topic).replace('|', '\\|')
        lines.append('| %s | %s | %s |' % (safe, status, note))
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
