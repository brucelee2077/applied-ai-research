#!/usr/bin/env python3
# =============================================================================
# Evidence Judge (Plan 2)  — FRONTIER-STAFF bar, LLM-as-judge. ADVISORY.
# =============================================================================
# A portfolio "day" produces an EVIDENCE ARTIFACT: a technical blog + the runnable
# experiment that backs every number/figure in that blog. This module is an LLM
# judge that reads all three (blog prose, experiment code, experiment stdout) and
# grades the artifact the way a frontier-lab staff researcher would: skeptically.
# Any number or figure asserted in the blog that is NOT supported by the actual
# experiment output is a FABRICATION and must be flagged.
#
# Axes graded (each finding is scoped to one):
#   technical_soundness — is the reasoning / method correct?
#   non_triviality      — is there a real result, not a toy restatement?
#   reproducibility     — can someone re-run this and get the same thing?
#   communication       — is the blog clear and honest about what it shows?
#   numbers_match       — does every claimed number/figure trace to the output?
#
# It is ADVISORY: it never edits any artifact and never fails a build. It DEGRADES
# GRACEFULLY: if the bridge/SDK is unreachable it returns status 'BRIDGE_UNAVAILABLE'
# with a conservative WEAK stub. Never raises.
#
# Bridge: http://localhost:11211/api/openai/v1 (keyless; reachable in-sandbox).
# Model:  aws:anthropic.claude-opus-4-8 (do NOT pass `temperature` — bridge 400s).
#
# Usage:
#   from evidence_judge import judge_evidence
#   res = judge_evidence(blog_text, experiment_code, experiment_output)
# CLI:
#   python3 evidence_judge.py <portfolio_day_dir>
# =============================================================================
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gates'))
# Reuse the tolerant JSON parser from coverage_judge (single source, no duplication).
from coverage_judge import _extract_json  # noqa: E402

BRIDGE_URL = 'http://localhost:11211/api/openai/v1'
MODEL = 'aws:anthropic.claude-opus-4-8'
_EVID_MAX = 20000

_EVID_SYS = (
    "You are a FRONTIER-LAB STAFF RESEARCHER reviewing a candidate's evidence artifact "
    "(a technical blog + the runnable experiment that backs it). Be skeptical: any "
    "number/figure in the blog that is NOT supported by the experiment output is a "
    "fabrication. Return STRICT JSON only."
)


def _evidence_prompt(blog_text, experiment_code, experiment_output):
    blog = (blog_text or '')[:_EVID_MAX]
    code = (experiment_code or '')[:_EVID_MAX]
    out = (experiment_output or '')[:_EVID_MAX]
    return f"""BLOG (the technical write-up under review):
\"\"\"
{blog}
\"\"\"

EXPERIMENT CODE (experiment.py — the runnable backing for every claim in the blog):
\"\"\"
{code}
\"\"\"

EXPERIMENT OUTPUT (experiment_out.txt — what the code actually produced):
\"\"\"
{out}
\"\"\"

Grade this evidence artifact at the FRONTIER-STAFF bar. Cross-check EVERY number and
figure claimed in the blog against the experiment output: if a claimed value does not
appear in / follow from the output, that is a fabrication (set numbers_match=false and
add a P0 finding on the "numbers_match" axis).

Return STRICT JSON with exactly these keys:
{{
  "verdict": "STRONG|OK|WEAK",
  "numbers_match": true,
  "findings": [ {{"axis": "technical_soundness|non_triviality|reproducibility|communication|numbers_match",
                  "severity": "P0|P1|P2",
                  "why": "<what is wrong / weak, quote the spot>",
                  "fix": "<concrete change that would fix it>"}} ],
  "summary": "<=2 sentences"
}}
Rules:
- "numbers_match": true ONLY if every number/figure in the blog traces to the experiment output.
- Judge on all five axes: technical_soundness, non_triviality, reproducibility, communication,
  numbers_match. Put each finding under the single axis it belongs to.
- "verdict":"STRONG" requires correct method AND a non-trivial result AND numbers_match=true."""


def judge_evidence(blog_text, experiment_code, experiment_output, model=MODEL, timeout=90):
    """Call the LLM evidence judge. Returns a dict:
      {status: 'OK'|'BRIDGE_UNAVAILABLE'|'PARSE_ERROR', verdict, numbers_match,
       findings, summary, raw?}  — never raises."""
    stub = {'verdict': 'WEAK', 'numbers_match': False, 'findings': [], 'summary': ''}
    try:
        from openai import OpenAI
    except Exception as e:
        return dict(status='BRIDGE_UNAVAILABLE', error='openai sdk missing: %s' % e, **stub)
    try:
        client = OpenAI(api_key='not-needed', base_url=BRIDGE_URL, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _EVID_SYS},
                      {'role': 'user', 'content': _evidence_prompt(blog_text, experiment_code, experiment_output)}],
            max_tokens=2000,
        )
        content = (resp.choices[0].message.content or '').strip()
    except Exception as e:  # APIConnectionError / APIStatusError / RateLimitError / ...
        return dict(status='BRIDGE_UNAVAILABLE', error=str(e), **stub)

    data = _extract_json(content)
    if data is None:
        return dict(status='PARSE_ERROR', raw=content, **stub)
    data.setdefault('verdict', 'WEAK')
    data.setdefault('numbers_match', False)
    data.setdefault('findings', [])
    data.setdefault('summary', '')
    data['status'] = 'OK'
    return data


def _read(path):
    """Read a text file, returning '' if it is missing/unreadable."""
    try:
        return open(path, encoding='utf-8').read()
    except Exception:
        return ''


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('day_dir', help='portfolio day directory containing blog.md / experiment.py / experiment_out.txt')
    a = ap.parse_args()
    blog = _read(os.path.join(a.day_dir, 'blog.md'))
    code = _read(os.path.join(a.day_dir, 'experiment.py'))
    out = _read(os.path.join(a.day_dir, 'experiment_out.txt'))
    res = judge_evidence(blog, code, out)

    print('== Evidence Judge (frontier-staff, advisory) ==')
    print('status:', res['status'], '| verdict:', res.get('verdict'), '| numbers_match:', res.get('numbers_match'))
    if res['status'] != 'OK':
        print('  (advisory — bridge unavailable or unparseable; no build impact)')
        if res.get('error'):
            print('  error:', res['error'])
    else:
        print('\n-- findings (frontier-staff bar) --')
        if not res['findings']:
            print('  (none)')
        for f in res['findings']:
            print('  [%s/%s] %s' % (f.get('severity', '?'), f.get('axis', '?'), f.get('why', '')))
            if f.get('fix'):
                print('        fix: %s' % f['fix'])
        print('\nsummary:', res['summary'])
    sys.exit(0)


if __name__ == '__main__':
    main()
