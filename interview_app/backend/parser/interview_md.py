"""
parser/interview_md.py — parse foundational `*-interview.md` files (Layer-2
interview deep dives) into the normalized question schema.

Format (under a "## Staff/Principal Interview Depth" section):

    ### Q1: <prompt>
    ---
    **No Hire**
    *Interviewee:* "<answer>"
    *Interviewer:* <note>
    *Criteria — Met:* a, b / *Missing:* c, d

    **Weak Hire**
    ...
    **Hire**
    ...
    **Strong Hire**
    ...
    ---

The four `criteria_met` sets union into the question's gradeable checklist.
Lenient: requires a prompt + >=1 level with an interviewee answer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_LEVELS = {
    "no hire": "no_hire",
    "weak hire": "weak_hire",
    "strong hire": "strong_hire",
    "hire": "hire",
}
_LEVEL_LINE_RE = re.compile(r"^\*\*\s*(No Hire|Weak Hire|Strong Hire|Hire)\s*\*\*\s*$", re.IGNORECASE)
# Question heading in either form: "### Q1: prompt" or "**Q1: prompt**".
_Q_HEAD_RE = re.compile(r"^(?:#{2,4}\s+|\*\*)\s*Q\s*(\d+)\s*[:.)]\s*(.*?)\s*\*{0,2}\s*$", re.IGNORECASE)
_FIELD_RE = re.compile(r"^\*(Interviewee|Interviewer|Criteria[^:]*)\s*:\*\s*(.*)$", re.IGNORECASE)


def _heading_level(line: str) -> int:
    m = re.match(r"^(#{1,6})\s", line)
    return len(m.group(1)) if m else 0


def _strip_quotes(text: str) -> str:
    return text.strip().strip('"').strip('"').strip('"').strip()


def _classify(level_text: str) -> str | None:
    t = level_text.strip().lower()
    for phrase, key in _LEVELS.items():
        if phrase == t:
            return key
    # fall back to substring (handles "no hire" etc.)
    for phrase, key in (("no hire", "no_hire"), ("weak hire", "weak_hire"),
                        ("strong hire", "strong_hire"), ("hire", "hire")):
        if phrase in t:
            return key
    return None


def _split_criteria(value: str) -> tuple[list[str], list[str]]:
    """Parse '*Criteria — Met:* a, b / *Missing:* c, d' into (met, missing)."""
    met_part, missing_part = value, ""
    m = re.split(r"/\s*\*?\s*Missing\s*:?\*?", value, maxsplit=1, flags=re.IGNORECASE)
    if len(m) == 2:
        met_part, missing_part = m[0], m[1]
    met_part = re.sub(r"^\*?\s*Met\s*:?\*?", "", met_part, flags=re.IGNORECASE)
    return _items(met_part), _items(missing_part)


def _items(text: str) -> list[str]:
    text = text.strip().strip("*").strip()
    if not text or text.lower() in {"none", "n/a", "-"}:
        return []
    parts = re.split(r"[;,]", text)
    return [p.strip(" .—-") for p in parts if p.strip(" .—-")]


def _parse_block(lines: list[str]) -> dict[str, Any]:
    """Parse one level block's Interviewee/Interviewer/Criteria lines."""
    entry = {"answer": "", "interviewer_note": "", "criteria_met": [], "criteria_missing": []}
    for line in lines:
        m = _FIELD_RE.match(line.strip())
        if not m:
            # continuation of the previous answer line
            if entry["answer"] and not entry["interviewer_note"] and line.strip():
                entry["answer"] += " " + line.strip()
            continue
        field = m.group(1).strip().lower()
        value = m.group(2).strip()
        if field.startswith("interviewee"):
            entry["answer"] = _strip_quotes(value)
        elif field.startswith("interviewer"):
            entry["interviewer_note"] = value
        elif field.startswith("criteria"):
            entry["criteria_met"], entry["criteria_missing"] = _split_criteria(value)
    return entry


def parse_interview_md(path: Path, module_id: str, track: str = "fundamentals") -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()

    topic = path.stem.replace("-interview", "").replace("-", " ").title()
    for line in lines:
        if _heading_level(line) == 1:
            topic = re.split(r"\s+[—-]\s+", line.lstrip("#").strip())[0].strip() or topic
            break

    file_slug = path.stem.replace("-interview", "")
    questions: list[dict[str, Any]] = []

    # Collect (prompt, body_lines) for each ### Q heading.
    blocks: list[tuple[int, str, list[str]]] = []
    i = 0
    while i < len(lines):
        m = _Q_HEAD_RE.match(lines[i])
        if m:
            qnum = int(m.group(1))
            prompt = m.group(2).strip()
            body: list[str] = []
            i += 1
            while i < len(lines) and not _Q_HEAD_RE.match(lines[i]) and not (
                _heading_level(lines[i]) and _heading_level(lines[i]) <= 2
            ):
                body.append(lines[i])
                i += 1
            blocks.append((qnum, prompt, body))
        else:
            i += 1

    for qnum, prompt, body in blocks:
        if not prompt:
            continue
        # Split body into level blocks at the **Level** markers.
        rubric: dict[str, Any] = {}
        current_level: str | None = None
        buf: list[str] = []

        def _flush(level, lines_):
            if level and level not in rubric:
                entry = _parse_block(lines_)
                if entry["answer"] or entry["interviewer_note"]:
                    rubric[level] = entry

        for line in body:
            lm = _LEVEL_LINE_RE.match(line.strip())
            if lm:
                _flush(current_level, buf)
                current_level = _classify(lm.group(1))
                buf = []
            elif current_level:
                buf.append(line)
        _flush(current_level, buf)

        if not rubric:
            continue

        criteria: list[str] = []
        for lv in rubric.values():
            criteria.extend(lv.get("criteria_met", []))

        try:
            rel_source = str(path.relative_to(path.parents[len(path.parents) - 1]))
        except (ValueError, IndexError):
            rel_source = path.name

        questions.append(
            {
                "id": f"{module_id}__{file_slug}__q{qnum}",
                "track": track,
                "module_id": module_id,
                "topic": topic,
                "section_title": f"{topic} · Q{qnum}",
                "source_file": str(path),
                "source_format": "interview_md",
                "difficulty": 4,
                "order": qnum,
                "prompt": prompt,
                "probes": [],
                "rubric": rubric,
                "criteria": _dedup(criteria),
            }
        )

    return questions


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it.lower() not in seen:
            seen.add(it.lower())
            out.append(it)
    return out
