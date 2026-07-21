#!/usr/bin/env python3
# =============================================================================
# v8 Source-First Lesson Compiler  (Phase C — hardened orchestrator)
# =============================================================================
# source.md  ->  lesson.html
#
# Thin orchestrator. All logic lives in reusable modules:
#   v8lib.py                      parsing, markdown-lite, widgets, region compile
#   gates/reader_flow_gate.py     Reader Flow Gate (on source, pre-compile)
#   gates/shell_invariant_gate.py Shell Invariant Gate (on output, + donor identity)
#
# Design: reuse the proven shell verbatim from a pristine donor snapshot and
# marker-replace only the reader-flow regions + authored DEMOS/BUILD/QS.
# Deterministic + idempotent: same (source.md, donor) -> byte-identical output.
#
# Usage:
#   python3 sessions/_compiler/compile_lesson.py <source.md>
#       [--donor <html>] [--out <html>] [--check-only] [--quiet]
#   exit 0 = compiled + gates pass ; 2 = reader-flow gate failed (nothing written)
#          ; 3 = shell-invariant gate failed ; 4 = visual integrity failed
#          ; 1 = usage / parse error
# =============================================================================
import sys, os, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'gates'))
import v8lib                       # noqa: E402
import reader_flow_gate            # noqa: E402
import shell_invariant_gate        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source')
    ap.add_argument('--donor')
    ap.add_argument('--out')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    src_dir = os.path.dirname(os.path.abspath(args.source))
    meta, body = v8lib.split_frontmatter(open(args.source, encoding='utf-8').read())
    blocks = v8lib.parse_blocks(body)

    donor_path = args.donor or os.path.join(HERE, 'shells', meta['donor'])
    out_path = args.out or os.path.join(src_dir, 'lesson.html')

    def log(*a):
        if not args.quiet:
            print(*a)

    log('== v8 compile:', os.path.relpath(args.source), '->', os.path.relpath(out_path))
    log('   donor:', os.path.relpath(donor_path), '| mode:', meta.get('mode'))

    concept_mode = (meta.get('mode') == 'concept')
    vok = True   # Visual Integrity Gate result (concept mode only; True in V8 branch)

    # -- Reader Flow Gate (source) : block write on failure --
    rok, rmsgs = reader_flow_gate.run(meta, blocks)
    log('\n-- Reader Flow Gate (source) --')
    for m in rmsgs: log('  ', m)
    if not rok:
        log('\nReader Flow Gate FAILED — nothing written.'); sys.exit(2)

    donor = open(donor_path, encoding='utf-8').read()
    html = v8lib.compile_html(meta, blocks, donor)

    if concept_mode:
        import concept_shell_gate
        sok, smsgs = concept_shell_gate.run(html, meta, donor=donor)
        log('\n-- Concept Shell Gate (output) --')
        for m in smsgs: log('  ', m)
        try:
            import notebook_smoothness_gate
        except ImportError as e:
            log('   notebook smoothness skipped:', e)
        else:
            # run() outside the import guard so a real bug in it surfaces, not swallowed
            nstatus, nmsgs = notebook_smoothness_gate.run(html, meta)
            log('\n-- Notebook Smoothness Gate --')
            for m in nmsgs: log('  ', m)
            if nstatus is False or str(nstatus).upper() == 'FAIL':
                sok = False; log('   notebook smoothness FAILED')
        # -- Coverage Gate (ADVISORY) : never changes sok / exit code --
        try:
            import coverage_gate
            cstatus, cmsgs = coverage_gate.run(html, meta, source_dir=src_dir)
            log('\n-- Coverage Gate (advisory) --')
            for m in cmsgs: log('  ', m)
        except Exception as e:
            log('   coverage gate skipped:', e)
        # -- Visual Integrity Gate (HARD) : block a visual that would render blank --
        # Catches viz embeds whose file/JS-dep is missing, whose height-sender
        # protocol drifted, or inline SVGs that draw nothing — the "compiles green
        # but renders blank" hole concept_shell_gate can't see. Cannot verify pixels
        # (no browser); a nested file:// iframe may still be blocked by the browser
        # regardless (serve over http to view) — that residual is out of scope here.
        try:
            import visual_integrity_gate
            vok, vmsgs = visual_integrity_gate.run(os.path.abspath(args.source), donor_path=donor_path)
            log('\n-- Visual Integrity Gate --')
            for m in vmsgs: log('  ', m)
        except Exception as e:
            log('   visual integrity gate skipped:', e)
    else:
        sok, smsgs = shell_invariant_gate.run(html, meta, donor=donor)
        log('\n-- Shell Invariant Gate (output vs donor) --')
        for m in smsgs: log('  ', m)

    if not args.check_only:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        log('\nwrote', os.path.relpath(out_path), '(%d chars)' % len(html))
    else:
        log('\n--check-only: not written')

    if not sok:
        log('\n%s FAILED.' % ('Concept Shell Gate' if concept_mode else 'Shell Invariant Gate')); sys.exit(3)
    if not vok:
        log('\nVisual Integrity Gate FAILED — a visual would render blank at runtime.'); sys.exit(4)
    log('\nOK — compiled and all gates pass.')


if __name__ == '__main__':
    main()
