# Frontier Mock Interview System

A local, single-user web app for practicing **Frontier AI Lab interviews**
(OpenAI / Anthropic / DeepMind style). A live AI interviewer asks a question,
probes adaptively, and grades your answer against a calibrated 4-level rubric
(No Hire / Weak Hire / Hire / Strong Hire). Wins feed your existing COACH XP,
level, and badges.

The live interviewer talks to a local **genAI bridge** (OpenAI-compatible,
keyless) — no API keys to manage. It reuses what this repo already has:
- the **COACH** gamification engine (`coach/core.py`) — XP, levels, badges, the
  45-min "boss battle" that this app turns into a real graded interview;
- ~380 interview questions parsed from the repo's `staff_interview_guide.md`,
  `*-interview.md`, and genAI guides, plus two hand-authored seed banks;
- the repo's vanilla-JS + vendored-D3 frontend style (no framework, no CDN).

---

## Quick start

```bash
# 1. install deps (already present in the repo's Python 3.11 env)
pip install -r interview_app/requirements.txt

# 2. make sure the genAI bridge is running (default: http://localhost:11211).
#    The live interviewer uses it out of the box — no API key needed.
#    To change model/endpoint or force offline, copy the env template:
#      cp interview_app/.env.example interview_app/.env
#    Available models: aws:anthropic.claude-opus-4-8 (default),
#      aws:anthropic.claude-sonnet-4-6, gcp:gemini-3.1-pro-preview, gcp:gemini-3.5-flash
#    If the bridge is down, the app degrades to "offline reveal" (self-grade).

# 3. (re)build the question bank — only needed if you change source content
python interview_app/scripts/build_questions.py

# 4. run the server FROM THE REPO ROOT (so `coach` is importable)
uvicorn interview_app.backend.main:app --host 127.0.0.1 --port 8000

# 5. open the app
#    http://127.0.0.1:8000/
```

---

## Tracks

| Track | Source | Questions |
|-------|--------|-----------|
| **ML System Design** | `ML Design/**` + `genAI design/**` staff guides | ~175 |
| **ML / AI Fundamentals** | foundational `*-interview.md` (transformers, RL, RAG, fine-tuning, eval, multimodal) | ~183 |
| **Frontier Research** | hand-authored `data/seed_frontier_research.json` | 12 |
| **Behavioral & Leadership** | hand-authored `data/seed_behavioral.json` | 10 |

Two session types:
- **Quick drill** — one question + up to one adaptive probe, then instant feedback.
- **Full mock** — a multi-question timed session (e.g. a whole ML-design case study)
  ending in a scorecard. Capped at 8 questions per session.

---

## How it works

```
interview_app/
  backend/                 FastAPI app (run from repo root)
    main.py                routes + static mount
    config.py              reads ANTHROPIC_API_KEY from .env; live/offline switch
    question_bank.py       loads data/questions.json, selection logic
    sessions.py            in-memory live session store
    interviewer.py         AI probing + forced function-call structured grading
    llm_client.py          OpenAI-SDK client -> genAI bridge + degradation switch
    grading.py             score<->level band anchoring (matches COACH widget)
    coach_bridge.py        the ONLY importer of coach.core (XP via record_boss_result)
    finalize.py            aggregates a session -> scorecard, commits XP, writes history
    history.py             reads/writes coach/interview_history/*.json
    parser/                build-time markdown/seed parsers
    schemas.py             Pydantic request/response + the Grade output schema
  data/
    questions.json         BUILD ARTIFACT (committed; regenerate with the script)
    seed_frontier_research.json / seed_behavioral.json   hand-authored banks
  scripts/build_questions.py
  frontend/                index.html, app.js, viz.js, styles.css, d3.v7.min.js (vendored)
  tests/                   pytest (parser, offline flow, live flow with a mocked client)
```

**Grading flow.** The live interviewer presents the question, decides per turn
to probe or grade (probe budget: drill 1, full mock 2), then grades via a forced
`submit_grade` function call to the genAI bridge (OpenAI Chat Completions
protocol). The backend **anchors** the model's `score_pct` to its chosen level's
band (no_hire .25 / weak .5 / hire .75 / strong 1.0) so the score and level
always agree and XP stays consistent with the COACH boss-battle widget.

**Graceful degradation.** Bridge disabled (`INTERVIEW_LLM_ENABLED=0`) → the whole
app runs offline (reveal + self-grade). Bridge enabled but a call fails
(unreachable/error) → that question falls back to offline reveal just-in-time.
The offline path uses the *same* question bank, so the app is always usable.

**XP integration.** On `/end`, the session score and elapsed time go to
`coach_bridge.commit_result` → `record_boss_result`, awarding 200 XP + tokens,
unlocking modules, and checking badges. Transcripts are saved under
`coach/interview_history/` (gitignored). The real `coach/state.json` is the
single source of truth shared with the notebooks.

> **Note on streaming.** The interviewer's probe text is returned in one call and
> "typed out" client-side (a typewriter effect) rather than streamed over SSE.
> This keeps the whole live path testable with a mocked client and avoids
> EventSource/proxy issues, with an equivalent feel.

---

## Tests

```bash
python -m pytest interview_app/tests/ -q
```

- `test_parser.py` — the three markdown formats + built-bank invariants.
- `test_backend_offline.py` — full offline flow; XP lands in a **temp** state.json
  (the real one is never touched); net-new track commits XP natively.
- `test_live_interviewer.py` — live probe loop, structured grading, score
  anchoring, and degradation — all with a **mocked** OpenAI client (never hits
  the bridge).

To smoke-test the live path for real, make sure the genAI bridge is running and
run one drill.
