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

## Grading standards
A session counts if it produced at least one concrete artifact and the learner can explain the mechanism.

It does not count if:
- The learner only read passively.
- The output is copied without understanding.
- There is no file/result/diagram/log.
- The artifact cannot be rerun/opened/reviewed.

## Tone
Warm but not indulgent. Do not overpraise vague effort. Praise concrete progress.

## Bilingual quiz style
Use Chinese for setup, English for key terms and interview phrasing.

Example:

> 你用一句英文解释一下：Why does KV cache reduce computation during autoregressive decoding?
