#!/usr/bin/env python3
"""Structural validation for Jupyter notebooks.

Usage:
    python3 scripts/validate_notebook.py <notebook_path> <module_id>

Example:
    python3 scripts/validate_notebook.py 03-transformers/01_attention.ipynb transformers

Checks:
    - COACH start/end cells present with correct module_id
    - No banned patterns (savefig, stale ../README.md links)
    - Python syntax validity in all code cells

Does NOT check: runtime errors, import failures, wrong variable names.
Skip validation for auto-generated notebooks (05_interviewer_perspective.ipynb).
"""

import json
import ast
import sys
import os


def validate(nb_path: str, module_id: str) -> list[str]:
    nb = json.load(open(nb_path))
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    issues = []

    # COACH start cell
    if code_cells:
        first = "".join(code_cells[0].get("source", []))
        if "render_session_start" not in first:
            issues.append("Missing COACH start cell")
        elif f'module_id="{module_id}"' not in first:
            issues.append(f'Wrong module_id (expected "{module_id}")')
    else:
        issues.append("No code cells")

    # COACH end cell
    if code_cells:
        last = "".join(code_cells[-1].get("source", []))
        if "render_session_end" not in last:
            issues.append("Missing COACH end cell")

    # Banned patterns
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and "savefig" in src:
            issues.append(f"savefig in cell {i}")
        if "../README.md" in src:
            issues.append(f'Stale link "../README.md" in cell {i}')

    # Syntax
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            try:
                ast.parse("".join(cell.get("source", [])))
            except SyntaxError as e:
                issues.append(f"SyntaxError in cell {i}: {e.msg} (line {e.lineno})")

    return issues


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <notebook_path> <module_id>")
        sys.exit(2)

    nb_path = sys.argv[1]
    module_id = sys.argv[2]
    issues = validate(nb_path, module_id)

    if issues:
        for iss in issues:
            print(f"  FAIL: {iss}")
        sys.exit(1)
    else:
        print(f"PASS {os.path.basename(nb_path)}")
