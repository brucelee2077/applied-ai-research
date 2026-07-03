"""
coach_bridge.py — the ONLY module that imports coach.core.

Wraps the gamification engine so the rest of the app never touches state.json
directly. Owns the load -> mutate -> save sequence (re-loading fresh state right
before mutating, exactly as coach.core.end_session does, so a concurrently
running notebook widget doesn't get clobbered).

Guardrail: record_boss_result indexes state["modules"][module_id] with no guard,
so an unknown module_id would raise KeyError. commit_result checks the allowlist
first and falls back to a plain XP award if the module isn't registered.
"""

from __future__ import annotations

from typing import Any

import coach.core as cc


def _badge_views(badge_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bid in badge_ids:
        meta = cc.BADGES.get(bid, {"name": bid, "desc": "", "rarity": "common"})
        out.append(
            {
                "id": bid,
                "name": meta["name"],
                "desc": meta["desc"],
                "rarity": meta["rarity"],
                "emoji": cc.RARITY_EMOJI.get(meta["rarity"], "⚪"),
            }
        )
    return out


def module_is_registered(module_id: str) -> bool:
    return module_id in cc.ALL_MODULES


def _zero_delta(passed: bool) -> dict[str, Any]:
    """A no-XP delta used when the COACH state can't be read/written (e.g. a
    corrupt state.json). Lets /end still return a scorecard and persist history
    instead of 500ing and losing the finished session."""
    return {
        "passed": passed,
        "xp_awarded": 0,
        "tokens_awarded": 0,
        "level_up": False,
        "new_level": 0,
        "new_level_name": "",
        "new_badges": [],
        "newly_unlocked_modules": [],
    }


def commit_result(module_id: str, score_pct: float, time_minutes: float, reason: str) -> dict[str, Any]:
    """Record a finished interview as a boss result and return a normalized
    XP-delta summary. Safe even if module_id isn't in ALL_MODULES, and never
    raises — a broken COACH state degrades to a zero-XP delta so the session's
    scorecard + history are still saved."""
    try:
        return _commit_result_inner(module_id, score_pct, time_minutes, reason)
    except Exception:  # corrupt/unwritable state.json, etc. — never lose the session
        return _zero_delta(score_pct >= 0.5)


def _commit_result_inner(module_id: str, score_pct: float, time_minutes: float, reason: str) -> dict[str, Any]:
    state = cc.load_state()  # fresh load right before mutating

    if module_is_registered(module_id):
        result = cc.record_boss_result(state, module_id, score_pct, time_minutes)
        cc.save_state(state)
        xp_result = result["xp_result"]
        return {
            "passed": result["passed"],
            "xp_awarded": xp_result["xp_awarded"],
            "tokens_awarded": xp_result["tokens_awarded"] + result["tokens_bonus"],
            "level_up": xp_result["level_up"],
            "new_level": xp_result["new_level"],
            "new_level_name": xp_result["new_level_name"],
            "new_badges": _badge_views(result["new_badges"]),
            "newly_unlocked_modules": result["newly_unlocked_modules"],
        }

    # Fallback: unregistered module (e.g. a topic not yet mapped). Award the
    # mock-interview XP directly so a session never crashes the engine.
    xp_result = cc.award_xp(state, cc.XP_MOCK_INTERVIEW, reason)
    state["player"]["tokens"] += cc.TOKENS_BOSS_COMPLETE
    new_badges = cc.check_and_award_badges(state, {"type": "boss_complete", "module": module_id, "score": score_pct})
    cc.save_state(state)
    return {
        "passed": score_pct >= 0.5,
        "xp_awarded": xp_result["xp_awarded"],
        "tokens_awarded": xp_result["tokens_awarded"] + cc.TOKENS_BOSS_COMPLETE,
        "level_up": xp_result["level_up"],
        "new_level": xp_result["new_level"],
        "new_level_name": xp_result["new_level_name"],
        "new_badges": _badge_views(new_badges),
        "newly_unlocked_modules": [],
    }


def get_progress() -> dict[str, Any]:
    """Read-only projection of the player's gamification state."""
    state = cc.load_state()
    xp = state["player"]["xp"]
    level_num, level_name, xp_into, xp_needed = cc.compute_level(xp)
    return {
        "xp": xp,
        "level": level_num,
        "level_name": level_name,
        "xp_into_level": xp_into,
        "xp_needed_for_next": xp_needed,
        "tokens": state["player"]["tokens"],
        "percentile": cc.get_cohort_percentile(state),
        "badges": _badge_views(state.get("badges", [])),
    }
