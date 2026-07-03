#!/usr/bin/env python3
"""
build_questions.py — parse the repo's interview-prep markdown into a single
normalized question bank at interview_app/data/questions.json.

This is a BUILD step, not a runtime parser. Re-run it whenever the source
markdown or the seed banks change:

    python interview_app/scripts/build_questions.py

Phase 0 scope: ML System Design only (the `staff_interview_guide.md` files).
Later phases register more parsers below (fundamentals, genai, seeds).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `interview_app` importable when run as a plain script from anywhere.
APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from interview_app.backend.parser.staff_guide_md import parse_staff_guide  # noqa: E402
from interview_app.backend.parser.interview_md import parse_interview_md  # noqa: E402
from interview_app.backend.parser.seed_loader import load_seed  # noqa: E402

# COACH module ids that map 1:1 to ML Design directory names. Only these get
# real XP via record_boss_result; others fall back (see coach_bridge).
ML_DESIGN_MODULES = {
    "01-ml-design-prep",
    "02-visual-search",
    "03-google-street-view",
    "04-youtube-video-search",
    "05-harmful-content-detection",
    "06-video-recommendation",
    "07-event-recommendation",
    "08-ad-click-prediction",
    "09-similar-listing",
    "10-personalized-news-feed",
    "11-people-you-may-know",
}

# Foundational module directory -> COACH module id (for the Fundamentals track).
# Ids present in coach.core.ALL_MODULES grant native XP/skill-tree progress;
# others (multimodal, evaluation, ...) use the coach_bridge XP fallback until
# Phase 5 registers them.
FUNDAMENTALS_MODULE_MAP = {
    "00-neural-networks": "neural-networks",
    "01-transformers": "transformers",
    "02-fine-tuning": "fine-tuning",
    "03-rag": "rag",
    "04-prompt-engineering": "prompt-engineering",
    "05-multimodal": "multimodal",
    "06-evaluation": "evaluation",
    "07-deployment": "deployment",
    "08-ai-agents": "ai-agents",
    "10-reinforcement-learning": "rl-fundamentals",
}


def build() -> dict:
    questions: list[dict] = []
    skipped: list[str] = []
    seen_ids: set[str] = set()

    def _ingest(parsed: list[dict], path: Path) -> None:
        for q in parsed:
            if not _valid(q):
                skipped.append(f"{q.get('id', path)}: failed validation")
                continue
            if q["id"] in seen_ids:
                skipped.append(f"{q['id']}: duplicate id")
                continue
            seen_ids.add(q["id"])
            try:
                q["source_file"] = str(path.relative_to(REPO_ROOT))
            except ValueError:
                q["source_file"] = path.name
            questions.append(q)

    # ── ML System Design: staff_interview_guide.md (ML Design + genAI Design) ─
    for design_root in ("ML Design", "genAI design"):
        for path in sorted((REPO_ROOT / design_root).glob("*/staff_interview_guide.md")):
            module_id = path.parent.name
            try:
                parsed = parse_staff_guide(path, module_id=module_id, track="ml_system_design")
            except Exception as exc:  # never let one bad file kill the build
                skipped.append(f"{path}: {exc!r}")
                continue
            if not parsed:
                skipped.append(f"{path}: no gradeable sections found")
                continue
            _ingest(parsed, path)

    # ── Fundamentals: foundational */**/*-interview.md ──────────────────────
    for path in sorted(REPO_ROOT.glob("*/**/*-interview.md")):
        top = path.relative_to(REPO_ROOT).parts[0]
        if top in {"ML Design", "genAI design"}:
            continue  # those use other formats / tracks
        module_id = FUNDAMENTALS_MODULE_MAP.get(top, top.lower())
        try:
            parsed = parse_interview_md(path, module_id=module_id, track="fundamentals")
        except Exception as exc:
            skipped.append(f"{path}: {exc!r}")
            continue
        if not parsed:
            skipped.append(f"{path}: no gradeable questions found")
            continue
        _ingest(parsed, path)

    # ── Net-new tracks: hand-authored seed banks (Frontier Research, Behavioral) ─
    for path in sorted((APP_DIR / "data").glob("seed_*.json")):
        try:
            parsed = load_seed(path)
        except Exception as exc:
            skipped.append(f"{path}: {exc!r}")
            continue
        if not parsed:
            skipped.append(f"{path}: empty seed bank")
            continue
        _ingest(parsed, path)

    bank = {
        "version": 1,
        "tracks": _track_summary(questions),
        "questions": questions,
    }
    _report(bank, skipped)
    return bank


def _valid(q: dict) -> bool:
    """A question is usable if it has a non-empty prompt and >=1 level answer."""
    if not q.get("prompt", "").strip():
        return False
    rubric = q.get("rubric") or {}
    levels = [lv for lv in ("no_hire", "weak_hire", "hire", "strong_hire") if lv in rubric]
    return len(levels) >= 1


def _track_summary(questions: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for q in questions:
        t = q["track"]
        entry = summary.setdefault(t, {"count": 0, "modules": {}})
        entry["count"] += 1
        mod = entry["modules"].setdefault(
            q["module_id"], {"topic": q.get("topic", q["module_id"]), "count": 0}
        )
        mod["count"] += 1
    return summary


def _report(bank: dict, skipped: list[str]) -> None:
    print(f"Parsed {len(bank['questions'])} questions across {len(bank['tracks'])} track(s):")
    for track, info in bank["tracks"].items():
        print(f"  {track}: {info['count']} questions, {len(info['modules'])} modules")
        for mid, mod in info["modules"].items():
            print(f"      {mid}: {mod['count']}  ({mod['topic']})")
    if skipped:
        print(f"\nSkipped {len(skipped)} item(s):")
        for s in skipped:
            print(f"  - {s}")


def main() -> None:
    bank = build()
    out_path = APP_DIR / "data" / "questions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)
    print(f"\nWrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
