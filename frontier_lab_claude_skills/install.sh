#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-.claude/skills}"

mkdir -p "$TARGET"
cp -R skills/* "$TARGET"/

echo "Installed Frontier Lab Claude Skills v2 into $TARGET"
find "$TARGET" -maxdepth 2 -name SKILL.md | sort
