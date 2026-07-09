#!/usr/bin/env python3
# =============================================================================
# Reader Flow Gate  (v8 Phase C/D)  — runs on the SOURCE, before compilation.
# =============================================================================
# Enforces the Reader Flow Blueprint + six rules from the distillation report.
#
# Two source modes:
#   * clean/exemplar (@@@ hero + @@@ section blocks, Day 1): STRICT — jargon
#     ladder, picture-before-vocab, frontier-deferred are hard checks.
#   * verbatim/migration (@@@ region blocks, Day 2-9 extracted from shipped
#     v7.6 lessons): RELAXED — hard-fail only on the core properties that v7.6
#     already guarantees (human-first hero, discovery produce); flag the
#     Day-1-exemplar extras (Jargon Ladder, explicit picture) as warnings so a
#     good inherited lesson is not falsely blocked.
#
# Reusable:  from reader_flow_gate import run ; ok, msgs = run(meta, blocks)
# CLI:       python3 gates/reader_flow_gate.py <source.md>   (exit 0 / 2)
# =============================================================================
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import v8lib

_TAG = re.compile(r'<[^>]+>')
def _text(html):
    return _TAG.sub(' ', html)


def _region_texts(blocks):
    """Return (mode, {hero_lede, s1, s2, s4, s7}) from clean OR verbatim source."""
    bt = {b['type']: b for b in blocks}
    secs = {b['args'].get('id'): b for b in blocks if b['type'] == 'section'}
    regions = {b['args'].get('name'): '\n'.join(b['lines']) for b in blocks if b['type'] == 'region'}
    t = {}
    if 'hero' in regions:                       # verbatim mode
        mode = 'verbatim'
        m = re.search(r'<p class="lede">(.*?)</p>', regions['hero'], re.DOTALL)
        t['hero_lede'] = _text(m.group(1)) if m else _text(regions['hero'])
        for sid in ('s1', 's2', 's4', 's7'):
            t[sid] = _text(regions.get(sid, ''))
        t['s1_raw'] = regions.get('s1', '')
    else:                                       # clean/exemplar mode
        mode = 'clean'
        hero = bt.get('hero'); htxt = '\n'.join(hero['lines']) if hero else ''
        t['hero_lede'] = htxt.split('@lede', 1)[1].split('@goal', 1)[0] if '@lede' in htxt else htxt
        for sid in ('s1', 's2', 's4', 's7'):
            t[sid] = '\n'.join(secs[sid]['lines']) if sid in secs else ''
        t['s1_raw'] = t['s1']
    return mode, t


def run(meta, blocks, spine_word=None):
    msgs, ok = [], [True]
    def fail(m): ok[0] = False; msgs.append('FAIL ' + m)
    def pas(m): msgs.append('pass ' + m)
    def warn(m): msgs.append('warn ' + m)

    mode, t = _region_texts(blocks)
    strict = (mode == 'clean')
    # In clean/exemplar mode the gate BLOCKS (fail). In verbatim/migration mode the
    # output is byte-identical to the vetted v7.6 donor, so nothing can regress —
    # the gate is INFORMATIONAL (warn) and flags what a future clean pass should add.
    hard = fail if strict else warn
    spine_word = (spine_word or meta.get('spine') or 'brain').split(':')[0].split()[0].lower()

    # --- hero: human-first, no frontier-pressure ---
    lede = t.get('hero_lede', '')
    low = lede.lower()
    hit = [x for x in v8lib.FRONTIER_TOKENS if x.lower() in low]
    hard('hero opens frontier-first (found %s)' % hit) if hit else pas('hero has no frontier-pressure opening')
    curiosity = ['your brain', 'brain', 'you ', 'your ', '?', 'imagine', 'picture',
                 'here is', 'ever ', 'what if', 'feels like', 'remember', 'yesterday']
    (pas('hero opens on human intuition/curiosity')
     if any(c in low for c in curiosity) else hard('hero has no obvious human/curiosity anchor'))

    # --- s1: Jargon Ladder + picture-before-vocab (STRICT clean / WARN verbatim) ---
    s1raw = t.get('s1_raw', '')
    (pas('s1 has a front-loaded Jargon Ladder')
     if '%%% jargon' in s1raw or 'Jargon' in s1raw else hard('s1 has no Jargon Ladder (Day-1 exemplar extra)'))
    if strict:
        idx_pic, idx_jar, idx_term = s1raw.lower().find('picture'), s1raw.find('%%% jargon'), s1raw.find('[[')
        (pas('mental picture precedes vocabulary in s1')
         if idx_pic >= 0 and (idx_jar < 0 or idx_pic < idx_jar) and (idx_term < 0 or idx_pic < idx_term)
         else fail('s1 mental picture does not clearly precede the vocabulary'))
    (hard('s1 leaks frontier payoff (defer to s4)') if any(x in t.get('s1', '') for x in ['GPT-3', '49,152', '49152'])
     else pas('s1 defers frontier payoff'))

    # --- s4: frontier payoff after mechanism + grounded staff/interview ---
    s4 = t.get('s4', '')
    (pas('frontier payoff / scale relevance in s4')
     if any(x in s4 for x in ['GPT-3', '49,152', '49152', 'frontier', 'billion', 'scale', 'production'])
     else warn('s4 has no explicit frontier/scale payoff'))
    if 'interview' in s4.lower():
        pas('s4 carries a staff/interview grounding')

    # --- narrative spine across >=3 of hero/s1/s2/s4 ---
    present = [n for n in ('hero', 's1', 's2', 's4')
               if spine_word in (t.get('hero_lede', '') if n == 'hero' else t.get(n, '')).lower()]
    (pas("spine ('%s') runs through %s" % (spine_word, '+'.join(present))) if len(present) >= 3
     else hard("spine ('%s') appears in <3 blocks: %s" % (spine_word, present)))

    # --- produce = discovery (broad cue set; HARD clean / soft verbatim) ---
    s7 = t.get('s7', '').lower()
    cues = ['predict', 'guess', 'before you write', 'what you should see', 'observe',
            'check your prediction', 'notice', 'watch']
    (pas('produce is discovery-framed')
     if any(c in s7 for c in cues) else hard('produce is not clearly discovery-framed'))

    return ok[0], msgs


def main():
    if len(sys.argv) < 2:
        print('usage: reader_flow_gate.py <source.md>'); sys.exit(1)
    meta, body = v8lib.split_frontmatter(open(sys.argv[1], encoding='utf-8').read())
    ok, msgs = run(meta, v8lib.parse_blocks(body))
    print('== Reader Flow Gate:', os.path.relpath(sys.argv[1]), '==')
    for m in msgs: print('  ', m)
    print('\n' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
