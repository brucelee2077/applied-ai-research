"""
llm_client.py — thin wrapper over the OpenAI SDK pointed at the local genAI
bridge, plus the graceful-degradation switch.

The bridge speaks the OpenAI Chat Completions protocol and is keyless:
    OpenAI(api_key="not-needed", base_url="http://localhost:11211/api/openai/v1")
    client.chat.completions.create(model="aws:anthropic.claude-opus-4-8", ...)

Two layers of degradation (see main.py for where they fire):
  (a) LLM disabled (INTERVIEW_LLM_ENABLED=0) -> the app runs offline reveal mode
  (b) enabled but a call fails               -> that question degrades to reveal

`is_live_available()` reflects config only; reachability is discovered per-call
(so a bridge that's down never blocks startup).
"""

from __future__ import annotations

from typing import Any, Optional

from .config import get_api_key, get_base_url, get_model_id, llm_enabled

# Errors that mean "the bridge isn't answering right now" -> degrade to offline.
try:
    import openai

    CONNECTION_ERRORS: tuple = (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APIError,
    )
    _OPENAI_OK = True
except Exception:  # pragma: no cover - only if the package is missing
    openai = None  # type: ignore
    CONNECTION_ERRORS = (Exception,)
    _OPENAI_OK = False

_CLIENT: Optional[Any] = None


def is_live_available() -> bool:
    """True when the live interviewer is enabled and the SDK imported. Does NOT
    probe the network (the bridge may be down; per-call degradation handles it)."""
    return _OPENAI_OK and llm_enabled()


def get_client() -> Optional[Any]:
    """Return a cached OpenAI client bound to the bridge, or None if disabled."""
    global _CLIENT
    if not is_live_available():
        return None
    if _CLIENT is None:
        _CLIENT = openai.OpenAI(api_key=get_api_key(), base_url=get_base_url(), timeout=60.0)
    return _CLIENT


def model_id() -> str:
    return get_model_id()


def base_url() -> str:
    return get_base_url()
