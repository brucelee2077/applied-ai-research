#!/usr/bin/env python3
"""Extract each lesson day's PRODUCE spec — the contract its experiment.py must meet.

Why this exists. 101 of 115 `experiment.py` files are the 5-line "Placeholder.
Fill this…" stub, so the DOING leg of the learning loop is missing everywhere
except m02+m03. But nothing has to be invented to fix that: every lesson already
states, in its produce section, exactly what the artifact must do — an "Option A"
description, an "Option B" copy-prompt with numbered requirements, and a list of
acceptance criteria. This pulls that out deterministically (no LLM) so a
generator works from the lesson's own contract instead of guessing.

Two shapes are handled:
  - V9 concept lessons  -> `@@@ produce` block in source.md
  - older V7 lessons    -> the produce `.module-section` in lesson.html
                           (32 of the 101 stub days have a source.md; the rest
                            only ever existed as compiled HTML)

Usage:
  python3 sessions/_produce_spec.py                 # summary table of stub days
  python3 sessions/_produce_spec.py --json out.json # full specs
  python3 sessions/_produce_spec.py <day-dir>       # one day, printed
"""
import os, re, sys, json, glob, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLACEHOLDER = 'placeholder. fill this'


def is_stub(experiment_path):
    """True when experiment.py is not yet a real self-checking scaffold.

    Mirrors gates/experiment_contract.py's two load-bearing checks, kept cheap
    (no AST) because this runs over the whole curriculum.
    """
    if not os.path.exists(experiment_path):
        return True
    src = open(experiment_path, encoding='utf-8', errors='replace').read()
    if _PLACEHOLDER in src.lower():
        return True
    return not ('__main__' in src and ('assert' in src or '✅' in src))


def _text(fragment):
    """Visible text of an HTML fragment, one idea per line, blanks dropped."""
    fragment = re.sub(r'(?s)<(script|style)\b.*?</\1>', ' ', fragment)
    fragment = re.sub(r'(?i)<br\s*/?>', '\n', fragment)
    fragment = re.sub(r'<[^>]+>', '\n', fragment)
    lines = [ln.strip() for ln in html.unescape(fragment).split('\n')]
    return '\n'.join(ln for ln in lines if ln)


def from_source(src_md):
    """The `@@@ produce` block of a V9 source.md, as plain text."""
    m = re.search(r'(?ms)^@@@\s+produce\b.*?(?=^@@@\s+\w|\Z)', src_md)
    return m.group(0).strip() if m else None


def from_lesson(lesson_html):
    """The produce section of a compiled lesson, as plain text.

    Anchored on the `s-produce` numeral class the shell gives that section, with
    a text fallback for older shells that predate it.
    """
    body = re.sub(r'(?s)<(script|style)\b.*?</\1>', ' ', lesson_html)
    for m in re.finditer(r'(?is)<(section|div)[^>]*class="[^"]*module-section[^"]*"[^>]*>', body):
        start = m.start()
        nxt = re.search(r'(?is)<(section|div)[^>]*class="[^"]*module-section[^"]*"[^>]*>',
                        body[m.end():])
        chunk = body[start:m.end() + (nxt.start() if nxt else len(body))]
        if 's-produce' in chunk or re.search(r'(?i)>\s*Produce\s*<', chunk):
            return _text(chunk)
    return None


def _claude_prompt(text):
    """The Option-B copy-prompt: the numbered requirement list, verbatim."""
    m = re.search(r'(?is)(Help me build.*?)(?=\nAcceptance criteria|\Z)', text)
    return m.group(1).strip() if m else None


def _acceptance(text):
    m = re.search(r'(?is)Acceptance criteria\s*\n(.*?)(?=\n(?:Mark|Next|Finish|Wrap)\b|\Z)', text)
    return m.group(1).strip() if m else None


def spec_for(day_dir):
    """Everything a generator needs for one day, or None if the day has no produce step."""
    src_path = os.path.join(day_dir, 'source.md')
    lesson_path = os.path.join(day_dir, 'lesson.html')
    raw, origin = None, None
    if os.path.exists(src_path):
        raw = from_source(open(src_path, encoding='utf-8', errors='replace').read())
        origin = 'source.md'
    if not raw and os.path.exists(lesson_path):
        raw = from_lesson(open(lesson_path, encoding='utf-8', errors='replace').read())
        origin = 'lesson.html'
    if not raw:
        return None
    rel = os.path.relpath(day_dir, ROOT)
    return {
        'day': os.path.basename(day_dir),
        'module': rel.split(os.sep)[1] if os.sep in rel else rel,
        'dir': rel,
        'origin': origin,
        'stub': is_stub(os.path.join(day_dir, 'experiment.py')),
        'produce': raw,
        'claude_prompt': _claude_prompt(raw),
        'acceptance': _acceptance(raw),
    }


def all_specs(stub_only=True):
    out = []
    for exp in sorted(glob.glob(os.path.join(ROOT, 'sessions', '**', 'experiment.py'),
                                recursive=True)):
        day_dir = os.path.dirname(exp)
        if stub_only and not is_stub(exp):
            continue
        s = spec_for(day_dir)
        if s:
            out.append(s)
    return out


def main():
    args = [a for a in sys.argv[1:]]
    if args and args[0] not in ('--json',):
        s = spec_for(os.path.join(ROOT, args[0]) if not os.path.isabs(args[0]) else args[0])
        print(json.dumps(s, indent=1, ensure_ascii=False) if s else 'no produce section found')
        return
    specs = all_specs()
    missing = [s['day'] for s in specs if not s['claude_prompt']]
    no_accept = [s['day'] for s in specs if not s['acceptance']]
    print('%d stub days with a produce spec' % len(specs))
    print('  from source.md : %d' % sum(1 for s in specs if s['origin'] == 'source.md'))
    print('  from lesson.html: %d' % sum(1 for s in specs if s['origin'] == 'lesson.html'))
    print('  without an Option-B prompt : %d %s' % (len(missing), missing[:6]))
    print('  without acceptance criteria: %d %s' % (len(no_accept), no_accept[:6]))
    if '--json' in args:
        out = args[args.index('--json') + 1]
        json.dump(specs, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        print('-> %s' % out)


if __name__ == '__main__':
    main()
