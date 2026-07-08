# Design Notes

## Source-free by default

The system should be able to produce strong lessons without old notebooks.

Old notebooks are useful as evaluation references, but using them as generation sources creates data leakage. It becomes impossible to tell whether the skills are good or whether they copied a good source.

## Existing sessions are codebase constraints

During refactor, current `sessions/` files should be read for:

- navigation
- quest ids
- section structure
- localStorage
- quiz mechanics
- BUILD / DEMOS
- completion gates
- existing file paths

But current content is not necessarily the teaching source of truth.

## Big refactors need contracts

Before editing, every module needs:

- blueprint
- coverage contract
- visual contract
- artifact contract
- refactor plan

This prevents local polishing from replacing real curriculum design.

## QA stops endless cleanup

QA findings are classified:

- P0: fix immediately
- P1: fix before merge if local/easy
- P2: final polish only

Markdown line wrapping should not block learning quality.
