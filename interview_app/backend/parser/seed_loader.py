"""
parser/seed_loader.py — load hand-authored seed question banks (the net-new
Frontier Research and Behavioral tracks) and expand them into normalized
questions matching the same schema the markdown parsers produce.

Seed file shape (DRY — track/module/topic declared once at the top):

    {
      "track": "frontier_research",
      "module_id": "frontier-research",
      "topic": "Frontier Research",
      "questions": [
        {
          "order": 1, "difficulty": 4, "title": "RLHF reward hacking",
          "prompt": "...",
          "probes": ["...", "..."],
          "criteria": ["...", "..."],
          "rubric": {
            "no_hire":    {"answer": "...", "interviewer_note": "..."},
            "weak_hire":  {"answer": "...", "interviewer_note": "..."},
            "hire":       {"answer": "...", "interviewer_note": "..."},
            "strong_hire":{"answer": "...", "interviewer_note": "..."}
          }
        }
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_LEVELS = ("no_hire", "weak_hire", "hire", "strong_hire")


def load_seed(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    track = raw["track"]
    module_id = raw["module_id"]
    topic = raw.get("topic", module_id)
    source = path.name

    out: list[dict[str, Any]] = []
    for q in raw.get("questions", []):
        order = q.get("order", len(out) + 1)
        rubric: dict[str, Any] = {}
        for lv in _LEVELS:
            entry = q.get("rubric", {}).get(lv)
            if not entry:
                continue
            rubric[lv] = {
                "answer": entry.get("answer", "").strip(),
                "interviewer_note": entry.get("interviewer_note", "").strip(),
                "criteria_met": entry.get("criteria_met", []),
                "criteria_missing": entry.get("criteria_missing", []),
            }
        out.append(
            {
                "id": f"{module_id}__seed__q{order}",
                "track": track,
                "module_id": module_id,
                "topic": topic,
                "section_title": q.get("title", f"{topic} · Q{order}"),
                "source_file": source,
                "source_format": "seed",
                "difficulty": q.get("difficulty", 4),
                "order": order,
                "prompt": q.get("prompt", "").strip(),
                "probes": q.get("probes", []),
                "rubric": rubric,
                "criteria": q.get("criteria", []),
            }
        )
    return out
