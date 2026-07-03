"""
parser/staff_guide_md.py — parse `staff_interview_guide.md` files into the
normalized question schema.

These files (ML Design + genAI Design) share a Section-based layout but drift in
the details. The parser tolerates all observed variants:

- Section heading:  "## Section N: <Title> (<min> min)"  or  "## Section N: <Title>"
- Prompt:           "### Interviewer Prompt" with the prompt as *"..."* or > "..."
                    (synthesized from the section title if the subsection is absent)
- Probes:           "### Follow-up Probes" bullet list, and/or a "What to Clarify"
                    table's "Question" column
- Answers:          a subsection whose title contains "Model Answers" or "Rubric",
                    with level answers as either
                      #### ❌ No Hire Answer        (subheading form)   or
                      **No Hire:** ...               (inline-bold form)
- Level names:      No Hire / Weak Hire / Hire / Strong Hire, plus the variants
                    "Lean No Hire" (-> weak_hire) and "Lean Hire" (-> hire)

Lenient by design: requires a non-empty prompt and >=1 level answer. Sections
without a gradeable answer block are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Level detection. Order matters: the more specific phrases are tested first so
# "Lean No Hire" / "Lean Hire" / "Strong Hire" never get swallowed by a bare
# "Hire" or by "No Hire".
_LEVEL_PATTERNS = [
    ("weak_hire", "lean no hire"),
    ("hire", "lean hire"),
    ("strong_hire", "strong hire"),
    ("weak_hire", "weak hire"),
    ("no_hire", "no hire"),
    ("hire", "hire"),
]

_SECTION_RE = re.compile(r"^##\s+Section\s+(\d+)\s*:\s*(.*?)\s*$", re.IGNORECASE)
_INLINE_LEVEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.*)$")


def _heading_level(line: str) -> int:
    m = re.match(r"^(#{1,6})\s", line)
    return len(m.group(1)) if m else 0


def _split_by_heading(lines: list[str], level: int) -> list[tuple[str, list[str]]]:
    """Split lines into (heading_text, body_lines) chunks at the given heading
    level. A shallower heading ends the current chunk and stops the scan."""
    chunks: list[tuple[str, list[str]]] = []
    title: str | None = None
    body: list[str] = []
    for line in lines:
        lvl = _heading_level(line)
        if lvl == level:
            if title is not None:
                chunks.append((title, body))
            title, body = line.lstrip("#").strip(), []
        elif 0 < lvl < level:
            if title is not None:
                chunks.append((title, body))
                title, body = None, []
        elif title is not None:
            body.append(line)
    if title is not None:
        chunks.append((title, body))
    return chunks


def _classify_level(text: str) -> str | None:
    t = text.lower()
    for key, phrase in _LEVEL_PATTERNS:
        if phrase in t:
            return key
    return None


def _strip_emphasis(text: str) -> str:
    text = text.strip()
    text = re.sub(r'^[>*_"“”\s]+', "", text)
    text = re.sub(r'[*_"“”\s]+$', "", text)
    return text.strip()


def _clean_prompt(body: list[str]) -> str:
    """Join the prompt subsection body into one line, stripping markdown
    emphasis, blockquote markers, and surrounding quotes."""
    text = " ".join(_strip_lead_quote(l) for l in body if l.strip())
    return _strip_emphasis(text)


def _strip_lead_quote(line: str) -> str:
    return re.sub(r"^\s*>\s?", "", line).strip()


def _parse_probe_bullets(body: list[str]) -> list[str]:
    out: list[str] = []
    for line in body:
        m = _BULLET_RE.match(line)
        if m:
            probe = _strip_emphasis(m.group(1))
            if probe:
                out.append(probe)
    return out


def _parse_clarify_table(body: list[str]) -> tuple[list[str], list[str]]:
    """Return (probes, criteria) from a clarify table by locating columns by
    header name. 'Question' -> probes; 'Dimension' -> criteria."""
    rows = [l for l in body if l.strip().startswith("|")]
    if len(rows) < 2:
        return [], []
    header = [c.strip().lower() for c in rows[0].strip().strip("|").split("|")]
    q_idx = next((i for i, h in enumerate(header) if "question" in h), None)
    d_idx = next((i for i, h in enumerate(header) if "dimension" in h), None)
    probes: list[str] = []
    criteria: list[str] = []
    for row in rows[2:]:  # skip header + |---| separator
        cells = [re.sub(r"[*`]", "", c).strip() for c in row.strip().strip("|").split("|")]
        if q_idx is not None and q_idx < len(cells) and cells[q_idx]:
            probes.append(cells[q_idx])
        if d_idx is not None and d_idx < len(cells) and cells[d_idx]:
            criteria.append(cells[d_idx])
    return probes, criteria


def _clean_answer(body: list[str]) -> str:
    kept = [l.rstrip() for l in body if l.strip() and l.strip() != "---"]
    return "\n".join(kept).strip()


def _parse_levels(answers_body: list[str]) -> dict[str, Any]:
    """Parse level answers from an answers subsection, supporting both the
    '#### <Level> Answer' subheading form and the inline '**<Level>:**' form."""
    rubric: dict[str, Any] = {}

    # Form 1: #### subheadings.
    sub_blocks = _split_by_heading(answers_body, 4)
    if sub_blocks:
        for heading, body in sub_blocks:
            level = _classify_level(heading)
            if level and level not in rubric:
                answer = _clean_answer(body)
                if answer:
                    rubric[level] = _level_entry(answer)
        if rubric:
            return rubric

    # Form 2: inline **Level:** markers.
    current: str | None = None
    buffer: list[str] = []

    def _flush() -> None:
        nonlocal current, buffer
        if current and current not in rubric:
            answer = _clean_answer(buffer)
            if answer:
                rubric[current] = _level_entry(answer)
        current, buffer = None, []

    for line in answers_body:
        m = _INLINE_LEVEL_RE.match(line.strip())
        if m:
            level = _classify_level(m.group(1))
            if level:
                _flush()
                current = level
                buffer = [m.group(2)] if m.group(2).strip() else []
                continue
        if current:
            buffer.append(line)
    _flush()
    return rubric


def _level_entry(answer: str) -> dict[str, Any]:
    return {"answer": answer, "interviewer_note": "", "criteria_met": [], "criteria_missing": []}


def _synthesize_prompt(section_title: str) -> str:
    # Drop a trailing "(N min)" and any "— subtitle", then strip a leading verb.
    title = re.sub(r"\s*\(\d+\s*min\)\s*$", "", section_title)
    title = re.split(r"\s+[—-]\s+", title)[0].strip()
    title = re.sub(r"^(Mastering|Master|Understanding)\s+", "", title, flags=re.IGNORECASE)
    return f"Let's go deep on {title or section_title}. Walk me through how you'd approach it."


def parse_staff_guide(path: Path, module_id: str, track: str = "ml_system_design") -> list[dict[str, Any]]:
    """Parse one staff_interview_guide.md into a list of normalized questions."""
    lines = path.read_text(encoding="utf-8").splitlines()

    topic = module_id
    for line in lines:
        if _heading_level(line) == 1:
            h1 = line.lstrip("#").strip()
            topic = re.split(r"\s+[—-]\s+", h1)[0].strip() or h1
            break

    try:
        rel_source = str(path.relative_to(path.parents[2]))
    except (ValueError, IndexError):
        rel_source = path.name

    questions: list[dict[str, Any]] = []
    for heading, body in _split_by_heading(lines, 2):
        m = _SECTION_RE.match("## " + heading)
        if not m:
            continue
        section_num = int(m.group(1))
        section_title = m.group(2).strip()

        subs = _split_by_heading(body, 3)
        prompt_body = _find_sub(subs, "interviewer prompt")
        probe_body = _find_sub(subs, "follow-up probe", "followup probe", "probe")
        clarify_body = _find_sub(subs, "what to clarify", "clarification dimension", "dimensions to clarify")
        answers_body = _find_sub(subs, "model answers", "rubric")

        if answers_body is None:
            continue
        rubric = _parse_levels(answers_body)
        if not rubric:
            continue

        prompt = _clean_prompt(prompt_body) if prompt_body is not None else ""
        if not prompt:
            prompt = _synthesize_prompt(section_title)

        probes = _parse_probe_bullets(probe_body) if probe_body is not None else []
        criteria: list[str] = []
        if clarify_body is not None:
            t_probes, t_criteria = _parse_clarify_table(clarify_body)
            if not probes:
                probes = t_probes
            criteria = t_criteria

        title_l = section_title.lower()
        if "principal" in title_l or "platform" in title_l:
            difficulty = 5
        elif section_num == 1 or "clarif" in title_l:
            difficulty = 3
        else:
            difficulty = 4

        questions.append(
            {
                "id": f"{module_id}__staff-guide__s{section_num}",
                "track": track,
                "module_id": module_id,
                "topic": topic,
                "section_title": section_title,
                "source_file": rel_source,
                "source_format": "staff_guide",
                "difficulty": difficulty,
                "order": section_num,
                "prompt": prompt,
                "probes": _dedup(probes),
                "rubric": rubric,
                "criteria": _dedup(criteria),
            }
        )

    return questions


def _find_sub(subs: list[tuple[str, list[str]]], *needles: str) -> list[str] | None:
    for title, body in subs:
        t = title.lower()
        if any(n in t for n in needles):
            return body
    return None


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it.lower() not in seen:
            seen.add(it.lower())
            out.append(it)
    return out
