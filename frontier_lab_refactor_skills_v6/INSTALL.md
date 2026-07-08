# Install Frontier Lab Refactor Skills v6

```bash
unzip frontier_lab_refactor_skills_v6.zip

mkdir -p .claude/skills
rm -rf .claude/skills/frontier-*
cp -R frontier_lab_refactor_skills_v6/skills/* .claude/skills/

find .claude/skills -maxdepth 2 -name SKILL.md | sort
```

Start Claude Code from repo root:

```bash
claude
```

Recommended model settings for big refactors:

```text
Opus 4.8
xhigh effort
```

Use `max` only for recovery or final deep review.
