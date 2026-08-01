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


# Two spellings of the same heading, each optionally wrapped in markdown bold.
# V7 compiled lessons say "Acceptance criteria"; every V9 source.md says "What
# you should see" instead (46 files use it, 0 use the literal one). Match both or
# the V9 field is null by construction.
_ACCEPT_LABEL = r'\*{0,2}(?:Acceptance criteria|What you should see)(?![A-Za-z])\*{0,2}'

# Where a block ends: the next structural marker of either shape. V9 markup
# (`####` heading, `@@@` block, `%%%` widget fence, `!!!` callout) or the
# compiled page's next part (the Option cards, the research-log box, the Done
# button, the finale, any leading emoji marker). Without this the criteria run to
# end-of-text and swallow the whole rest of the page.
_MARKERS = (r'####|@@@|%%%|!!!|\*{0,2}Option\s+[AB]\b'
            r'|Mark\b|Next\b|Finish\b|Wrap\b|Done[ \t]*(?=\n|\Z)'
            r'|[\U0001F300-\U0001FAFF☀-➿]')
_ACCEPT_END = r'(?i)(?=\n[ \t]*(?:' + _MARKERS + r')|\Z)'
# the prompt additionally ends at the criteria heading that follows it
_PROMPT_END = r'(?=\n[ \t]*(?:' + _ACCEPT_LABEL + r'|' + _MARKERS + r')|\Z)'


def _claude_prompt(text):
    """The Option-B copy-prompt: the numbered requirement list, verbatim."""
    m = re.search(r'(?is)(Help me build.*?)' + _PROMPT_END, text)
    return m.group(1).strip() if m else None


def _acceptance(text):
    """The acceptance list, under either heading spelling.

    Three m03 days state the criteria as prose on the heading line itself ("What
    you should see by the end: <criteria>"), so that line is kept when it carries
    real content after a colon. A bare lead-in ("… when you run it:") or a
    parenthetical aside ("(check your prediction)") is dropped as decoration.
    """
    head = re.search(r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?' + _ACCEPT_LABEL + r'([^\n]*)$', text)
    if not head:
        return None
    tail = head.group(1).strip()
    after_colon = tail.split(':', 1)[1].strip() if ':' in tail else ''
    if len(after_colon) > 15:
        # prose shape: the criteria ARE the heading line, so keep it (minus `####`)
        rest = re.sub(r'\A[ \t]*#{1,6}[ \t]*', '', text[head.start():])
    else:
        rest = text[head.end():]
    stop = re.search(_ACCEPT_END, rest)
    return _list_only((rest[:stop.start()] if stop else rest)).strip() or None


_BULLET = re.compile(r'[-*+•]\s|\d+[.)]\s')


def _list_only(block):
    """If the criteria are written as a list, the list is where they end.

    V9 days often follow the bullet list with a paragraph of optional play
    ("Then poke it — set `keep` to 0.0 …"). That is the next part of the lesson,
    not a criterion, and no markup marker separates the two. Blocks that are not
    lists (the compiled-HTML shape, where `<li>` text is already flattened to
    bare lines) are returned untouched.
    """
    lines = block.split('\n')
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is None or not _BULLET.match(lines[first].lstrip()):
        return block
    kept = lines[:first]
    for ln in lines[first:]:
        if not ln.strip():
            kept.append(ln)          # blank lines may sit inside a list
            continue
        # a bullet, or an indented continuation of the bullet above it
        if _BULLET.match(ln.lstrip()) or ln[:1] in (' ', '\t'):
            kept.append(ln)
            continue
        break
    return '\n'.join(kept)


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
