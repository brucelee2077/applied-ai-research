# Install / Replace Instructions

## 1. Unzip

```bash
unzip frontier_lab_claude_skills_v2.zip
```

## 2. From your repo root, create a branch

```bash
git checkout -b refactor/coach-skills-v2
```

## 3. Replace current skill pack

```bash
rm -rf frontier_lab_claude_skills
cp -R frontier_lab_claude_skills_v2 frontier_lab_claude_skills
```

## 4. Install as project-local Claude Code skills

```bash
mkdir -p .claude/skills
rm -rf .claude/skills/frontier-*
cp -R frontier_lab_claude_skills/skills/* .claude/skills/
```

## 5. Verify

```bash
find .claude/skills -maxdepth 2 -name SKILL.md | sort
```

Expected: 11 skills.

## 6. Start Claude Code

```bash
claude
```

## 7. First command to run

Paste this into Claude Code:

```text
Use /frontier-curriculum-refactor to inspect the repo and create a safe 4-lesson Coach Layer pilot plan for sessions/. Do not modify files until the plan is written.
```
