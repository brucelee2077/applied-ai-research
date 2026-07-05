#!/usr/bin/env python3
"""
Coverage audit for the Frontier Lab 24-week curriculum lessons.

Checks that every resource in the companion's "Deduplicated Master Resource Index"
is cited in at least one generated lesson HTML file under sessions/week-*/.

Usage:
    python3 sessions/coverage_audit.py
Exit code 0 if every resource is cited somewhere, 1 otherwise (prints the gaps).
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPANION = ROOT / "frontier_ai_24_week_link_companion.md"
SESSIONS = ROOT / "sessions"


def master_index_urls(text: str):
    """Extract (name, url) pairs from the Deduplicated Master Resource Index section."""
    # The master index lines look like:  - **Name** (type): https://url
    start = text.find("## Deduplicated Master Resource Index")
    if start == -1:
        print("!! Could not find the master resource index section", file=sys.stderr)
        return []
    section = text[start:]
    # stop at the next H2 if any
    nxt = section.find("\n## ", 3)
    if nxt != -1:
        section = section[:nxt]
    pairs = []
    for m in re.finditer(r"- \*\*(.+?)\*\* \((\w+)\): (\S+)", section):
        name, kind, url = m.group(1), m.group(2), m.group(3).rstrip()
        # strip trailing punctuation and any inline note after " — "
        url = url.split(" ")[0].rstrip(".,;")
        pairs.append((name, kind, url))
    return pairs


def all_lesson_text():
    """Concatenate the text of every generated lesson HTML file."""
    blobs = {}
    for html in sorted(list(SESSIONS.glob("week-*/day-*.html")) + list(SESSIONS.glob("m*/day-*/lesson.html"))):
        blobs[html.relative_to(ROOT).as_posix()] = html.read_text(encoding="utf-8")
    return blobs


def norm(url: str) -> str:
    """Normalize a URL for loose matching (ignore trailing slash, http/https, www)."""
    u = url.strip()
    u = re.sub(r"^https?://", "", u)
    u = re.sub(r"^www\.", "", u)
    u = u.rstrip("/")
    return u


def main():
    text = COMPANION.read_text(encoding="utf-8")
    resources = master_index_urls(text)
    blobs = all_lesson_text()
    if not resources:
        print("No resources parsed; aborting.")
        return 2
    if not blobs:
        print("No lesson files found under sessions/week-*/day-*.html or sessions/m*/day-*/lesson.html; aborting.")
        return 2

    # Build one normalized haystack per file plus a global one.
    global_hay = "\n".join(blobs.values())
    global_hay_norm = norm(global_hay) if False else global_hay  # keep raw; match on raw substring

    missing = []
    cited = 0
    for name, kind, url in resources:
        n = norm(url)
        found = any(n in norm(b) for b in blobs.values())
        if found:
            cited += 1
        else:
            missing.append((name, kind, url))

    total = len(resources)
    print(f"Master index resources: {total}")
    print(f"Cited in >=1 lesson:    {cited}")
    print(f"Missing (cited nowhere):{len(missing)}")
    if missing:
        print("\n=== UNCITED RESOURCES ===")
        for name, kind, url in missing:
            print(f"  [{kind}] {name}\n        {url}")
        return 1
    print("\nAll master-index resources are cited in at least one lesson. ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
