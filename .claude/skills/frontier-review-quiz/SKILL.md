---
name: frontier-review-quiz
description: Review a completed Frontier Lab study session, quiz the learner, identify misconceptions, update notes, and decide whether the day counts. Use after the user pastes code, notes, HTML output, experiment results, or a daily research log.
---

# Frontier Review Quiz

Use this after the learner completes a study or build session.

## Review output structure
```markdown
# Session Review

## 1. Verdict
Counts / Partially counts / Does not count yet

## 2. What is correct
Concrete things done well.

## 3. What is shallow or missing
Be direct but supportive.

## 4. Misconceptions
List likely misunderstandings.

## 5. Frontier-lab relevance
Explain why this artifact matters or does not yet matter.

## 6. Quiz
5 questions:
- 2 intuition questions
- 2 mechanism questions
- 1 interview-style question

## 7. Required fix or next step
One concrete next action.

## 8. Log update
Markdown snippet to paste into today's log.
```

## Mark the day done (ledger update)
This skill is the only one that marks a curriculum day complete. After you reach a verdict:

- If the verdict is **Counts** (or Partially counts and the learner has an artifact),
  update `sessions/progress.json`:
  - Add the day id (e.g. `w01-d01`) to the `completed` array if not already there.
  - In `generated`, set that day's `status` to `completed`.
  - Advance `cursor` to the next day id. Ids run `wNN-d01 … wNN-d05`, then roll over to
    the next week: after `wNN-d05` the next cursor is `w(NN+1)-d01` (e.g. `w01-d05` →
    `w02-d01`). Weekdays map Monday=d01 … Friday=d05, matching `frontier-session-coach`.
- If it **Does not count yet**, leave the ledger unchanged and tell the learner what to fix.

The quest page itself (HTML + localStorage) tracks per-step 打卡 for the learner's eyes.
`progress.json` is the source of truth for "which day is next" — keep them consistent.

## Grading standards
A session counts if it produced at least one concrete artifact and the learner can explain the mechanism.

It does not count if:
- The learner only read passively.
- The output is copied without understanding.
- There is no file/result/diagram/log.
- The artifact cannot be rerun/opened/reviewed.

## Tone
Warm but not indulgent. Do not overpraise vague effort. Praise concrete progress.

## Language
Write in **English**. The user prefers English and reads it comfortably. Use Chinese only
where it genuinely clarifies a specific nuance — rare in practice.

Example:

> Explain in one sentence: why does the KV cache reduce computation during autoregressive decoding?
