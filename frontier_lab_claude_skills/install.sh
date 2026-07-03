#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:-$HOME/.claude/skills}"
mkdir -p "$TARGET"
cp -R skills/* "$TARGET/"
echo "Installed Frontier Lab skills to: $TARGET"
echo "Start Claude Code and invoke skills with /frontier-session-coach, /frontier-concept-courseware, etc."
