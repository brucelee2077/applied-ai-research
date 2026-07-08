# Goal + Workflow Playbook

## Core loop

```text
Set /goal
→ read or create manifest
→ run dynamic workflow
→ audit module
→ update manifest
→ fix open P0
→ rerun QA
→ update manifest
→ optional held-out eval
→ update manifest
→ patch skills if systemic
→ update manifest
→ stop when goal condition is satisfied
```

## When to use `/goal`

Use `/goal` for iterative work with a measurable end condition.

Good goal condition:

```text
Module 2 is pass_with_p1, manifest exists, P0 backlog is empty, QA report shows 0 P0, audit checks pass, held-out eval findings are recorded, and skill-gap candidates are either patched or open.
```

Bad goal condition:

```text
Make m02 better.
```

## Dynamic workflow

Assume dynamic workflow is enabled by the user.

Use subagents for:

- coverage traceability
- coach voice / foundation framing
- function family coverage
- visual/evidence
- artifact/anchor consistency
- held-out eval
- adversarial integration

## Stop conditions

Stop the loop if:

- manifest status is pass or pass_with_p1 and open P0 backlog is empty
- two rounds in a row make no progress
- hard JS/nav/audit failures appear
- required fix would modify shared shell files
- held-out eval finds systemic skill failure requiring skill patch
