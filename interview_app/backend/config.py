"""
config.py — runtime configuration for the mock interview backend.

The live interviewer talks to a local **genAI bridge** that speaks the OpenAI
Chat Completions protocol (keyless):

    base_url = http://localhost:11211/api/openai/v1
    models   = aws:anthropic.claude-opus-4-8, aws:anthropic.claude-sonnet-4-6,
               gcp:gemini-3.1-pro-preview, gcp:gemini-3.5-flash

All of these are overridable via interview_app/.env (see .env.example). Never
logs the key (the bridge is keyless anyway).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ───────────────────────────────────────────────────────────────────
# interview_app/backend/config.py  ->  interview_app/
APP_DIR = Path(__file__).resolve().parent.parent
# repo root (one level above interview_app/) — where `coach/` lives
REPO_ROOT = APP_DIR.parent
DATA_DIR = APP_DIR / "data"
FRONTEND_DIR = APP_DIR / "frontend"
QUESTIONS_PATH = DATA_DIR / "questions.json"

# Load interview_app/.env if present. Values already in the environment win.
load_dotenv(APP_DIR / ".env")

# ── Defaults (genAI bridge) ───────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:11211/api/openai/v1"
DEFAULT_MODEL = "aws:anthropic.claude-opus-4-8"


# ── LLM settings ──────────────────────────────────────────────────────────────
def get_base_url() -> str:
    """OpenAI-compatible base URL for the genAI bridge."""
    return os.environ.get("INTERVIEW_LLM_BASE_URL", "").strip() or DEFAULT_BASE_URL


def get_model_id() -> str:
    """Model id passed to the bridge (e.g. aws:anthropic.claude-opus-4-8)."""
    return os.environ.get("INTERVIEW_MODEL_ID", "").strip() or DEFAULT_MODEL


def get_api_key() -> str:
    """API key sent to the bridge. The bridge is keyless, so this defaults to a
    placeholder the OpenAI client requires but the bridge ignores."""
    # ANTHROPIC_API_KEY kept as a fallback name for convenience.
    return (
        os.environ.get("INTERVIEW_LLM_API_KEY", "").strip()
        or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        or "not-needed"
    )


def llm_enabled() -> bool:
    """Whether the live interviewer is enabled. On by default (the bridge is a
    local, keyless default); set INTERVIEW_LLM_ENABLED=0 to force offline mode."""
    return os.environ.get("INTERVIEW_LLM_ENABLED", "1").strip() not in {"0", "false", "no", ""}


# Grading band centers — must match the existing COACH boss-battle widget's
# SCORE_MAP so XP math stays consistent across the app and the notebooks.
#   passed = score_pct >= 0.5  (i.e. "weak_hire or better")
LEVEL_SCORE = {
    "no_hire": 0.25,
    "weak_hire": 0.5,
    "hire": 0.75,
    "strong_hire": 1.0,
}

# Display thresholds for turning an aggregate score_pct back into a band label.
# Mirrors the widget's display bands.
SCORE_BANDS = [
    (0.85, "strong_hire", "Strong Hire"),
    (0.65, "hire", "Hire"),
    (0.40, "weak_hire", "Weak Hire"),
    (0.0, "no_hire", "No Hire"),
]
