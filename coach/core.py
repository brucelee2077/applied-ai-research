"""
coach/core.py — XP engine, streak logic, SM-2 spaced repetition, badge system,
boss unlock, module skill tree, session management.

All state is persisted to coach/state.json.
Atomic writes (write to .tmp then rename) prevent corruption on crash.
"""

from __future__ import annotations

import json
import math
import os
import random
import datetime
from pathlib import Path
from typing import Any

# ── Paths ─────────────────────────────────────────────────────────────────────
COACH_DIR = Path(__file__).parent
STATE_PATH = COACH_DIR / "state.json"

# ── XP Awards ─────────────────────────────────────────────────────────────────
XP_READING_NOTEBOOK = 10
XP_QUIZ_PASS = 50
XP_QUIZ_RETRY = 20
XP_MOCK_INTERVIEW = 200
XP_PERFECT_QUIZ = 100
XP_STREAK_BONUS = 25
XP_FIRST_SESSION = 50

# ── Levels (cumulative XP threshold) ──────────────────────────────────────────
LEVELS = [
    (1,  0,     "ML Intern"),
    (2,  150,   "Junior MLE"),
    (3,  400,   "MLE I"),
    (4,  800,   "MLE II"),
    (5,  1500,  "Senior MLE"),
    (6,  2500,  "Staff MLE"),
    (7,  4000,  "Principal MLE"),
    (8,  6000,  "ML Architect"),
    (9,  9000,  "Distinguished MLE"),
    (10, 13000, "AI Legend"),
]

# ── Streak multipliers ─────────────────────────────────────────────────────────
def compute_streak_multiplier(streak_days: int) -> float:
    if streak_days >= 30:
        return 3.0
    if streak_days >= 14:
        return 2.0
    if streak_days >= 7:
        return 1.5
    if streak_days >= 3:
        return 1.25
    return 1.0

# ── Token economy ──────────────────────────────────────────────────────────────
TOKENS_PER_SESSION = 5
TOKENS_QUIZ_PASS = 2
TOKENS_BOSS_COMPLETE = 20
TOKENS_PERFECT_SCORE = 10

COST_STREAK_SHIELD = 15
COST_HINT_REVEAL = 5
COST_SKIP_PREREQ = 50

# ── Boss battle config ─────────────────────────────────────────────────────────
BOSS_EXPIRY_HOURS = 48
BOSS_TIME_LIMIT_MINUTES = 45

# ── SM-2 defaults ─────────────────────────────────────────────────────────────
SR_INITIAL_EASE = 2.5
SR_MIN_EASE = 1.3

# ── Skill tree prerequisite map ────────────────────────────────────────────────
PREREQ_MAP: dict[str, list[str]] = {
    "02-visual-search":             ["01-ml-design-prep"],
    "03-google-street-view":        ["02-visual-search"],
    "04-youtube-video-search":      ["03-google-street-view"],
    "05-harmful-content-detection": ["04-youtube-video-search"],
    "06-video-recommendation":      ["05-harmful-content-detection"],
    "07-event-recommendation":      ["06-video-recommendation"],
    "08-ad-click-prediction":       ["07-event-recommendation"],
    "09-similar-listing":           ["08-ad-click-prediction"],
    "10-personalized-news-feed":    ["09-similar-listing"],
    "11-people-you-may-know":       ["10-personalized-news-feed"],
}

ALL_MODULES = [
    "01-ml-design-prep",
    "02-visual-search",
    "03-google-street-view",
    "04-youtube-video-search",
    "05-harmful-content-detection",
    "06-video-recommendation",
    "07-event-recommendation",
    "08-ad-click-prediction",
    "09-similar-listing",
    "10-personalized-news-feed",
    "11-people-you-may-know",
    # Neural networks / foundations
    "rnn",
    "transformers",
]

# Study notebooks per module (not the boss notebook)
MODULE_STUDY_NOTEBOOKS: dict[str, list[str]] = {
    "01-ml-design-prep": [
        "01_ml_design_framework.ipynb",
        "02_interview_strategy.ipynb",
    ],
    "02-visual-search": [
        "01_visual_search_system_design.ipynb",
        "02_embedding_and_contrastive_learning.ipynb",
        "03_ann_search_and_serving.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    "03-google-street-view": [
        "01_street_view_system_design.ipynb",
        "02_object_detection_deep_dive.ipynb",
        "03_evaluation_and_serving.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    "04-youtube-video-search": [
        "01_video_search_system_design.ipynb",
        "02_multimodal_embeddings.ipynb",
        "03_ranking_and_serving.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    "05-harmful-content-detection": [
        "01_harmful_content_system_design.ipynb",
        "02_multimodal_fusion_and_features.ipynb",
        "03_multi_task_training_and_evaluation.ipynb",
    ],
    "06-video-recommendation": [
        "01_recommendation_system_design.ipynb",
        "02_candidate_generation.ipynb",
        "03_ranking_models.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    "07-event-recommendation": [
        "01_event_recommendation_system_design.ipynb",
        "02_location_and_time_features.ipynb",
        "03_ranking_and_personalization.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    "08-ad-click-prediction": [
        "01_ad_click_system_design.ipynb",
        "02_feature_engineering_deep_dive.ipynb",
        "03_deep_ctr_models.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    "09-similar-listing": [
        "01_similar_listing_system_design.ipynb",
        "02_embedding_techniques.ipynb",
        "03_ranking_and_serving.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    "10-personalized-news-feed": [
        "01_news_feed_system_design.ipynb",
        "02_ranking_and_personalization.ipynb",
        "03_multi_task_and_engagement.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    "11-people-you-may-know": [
        "01_pymk_system_design.ipynb",
        "02_graph_based_approaches.ipynb",
        "03_ranking_and_privacy.ipynb",
        "04_interview_walkthrough.ipynb",
    ],
    # Neural networks / foundations
    "rnn": [
        "01_rnn_fundamentals.ipynb",
        "03_lstm.ipynb",
        "04_gru.ipynb",
        "05_sequence_tasks.ipynb",
        "06_bidirectional_rnns.ipynb",
    ],
    "transformers": [
        "01_attention_mechanisms.ipynb",
        "02_multi_head_attention.ipynb",
        "03_positional_encoding.ipynb",
        "04_transformer_block.ipynb",
    ],
}

# ── Default state factory ──────────────────────────────────────────────────────
def _default_state() -> dict:
    modules = {}
    # Foundations modules are always unlocked; ML Design skill tree starts at module 01
    ALWAYS_UNLOCKED = {"01-ml-design-prep", "rnn", "transformers"}
    for mid in ALL_MODULES:
        modules[mid] = {
            "unlocked": mid in ALWAYS_UNLOCKED,
            "notebooks_completed": [],
            "boss_unlocked": False,
            "boss_attempted": False,
            "boss_passed": False,
            "boss_expires_at": None,
            "boss_score": None,
            "boss_time_minutes": None,
            "mastery_score": 0.0,
        }
    return {
        "version": 1,
        "player": {
            "xp": 0,
            "level": 1,
            "tokens": 0,
            "total_sessions": 0,
            "total_study_minutes": 0,
            "first_session_bonus_given": False,
        },
        "streak": {
            "current": 0,
            "longest": 0,
            "last_study_date": None,
            "shields_remaining": 2,
            "fire_active": False,
        },
        "modules": modules,
        "skill_tree": {
            "unlocked_nodes": ["01-ml-design-prep"],
            "prereq_map": PREREQ_MAP,
        },
        "badges": [],
        "quiz_history": {
            "concept_last_correct": {},
            "concept_next_review": {},
            "concept_ease_factor": {},
            "concept_interval_days": {},
            "concept_review_count": {},
            "perfect_quiz_count": 0,
        },
        "boss_battles": {
            "completed": [],
            "in_progress": None,
        },
        "cohort": {
            "fictional_cohort_size": 1247,
        },
        "session_log": [],
        "meta": {
            "created_at": datetime.datetime.now().isoformat(),
            "last_updated": None,
            "session_start": None,
        },
    }

# ── State I/O ──────────────────────────────────────────────────────────────────
def load_state() -> dict:
    """Load state.json, creating with defaults if absent."""
    if not STATE_PATH.exists():
        state = _default_state()
        save_state(state)
        return state
    with open(STATE_PATH, "r") as f:
        return json.load(f)

def save_state(state: dict) -> None:
    """Atomically write state to state.json."""
    state["meta"]["last_updated"] = datetime.datetime.now().isoformat()
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)

# ── Level engine ───────────────────────────────────────────────────────────────
def compute_level(xp: int) -> tuple[int, str, int, int]:
    """Return (level_number, level_name, xp_into_level, xp_needed_for_next)."""
    current = LEVELS[0]
    for lvl in LEVELS:
        if xp >= lvl[1]:
            current = lvl
        else:
            break
    idx = next(i for i, l in enumerate(LEVELS) if l[0] == current[0])
    xp_into = xp - current[1]
    if idx + 1 < len(LEVELS):
        xp_needed = LEVELS[idx + 1][1] - current[1]
    else:
        xp_needed = 9999  # max level
    return current[0], current[2], xp_into, xp_needed

def get_level_name(xp: int) -> str:
    lvl, _, _, _ = compute_level(xp)
    return next(name for num, _, name in LEVELS if num == lvl)

# ── XP engine ──────────────────────────────────────────────────────────────────
def award_xp(state: dict, base_xp: int, reason: str) -> dict:
    """
    Award XP with streak multiplier applied. Updates state in place.
    Returns summary dict. Caller must save_state().
    """
    streak = state["streak"]["current"]
    multiplier = compute_streak_multiplier(streak)
    xp_awarded = math.floor(base_xp * multiplier)

    prev_xp = state["player"]["xp"]
    state["player"]["xp"] += xp_awarded

    # Token award: ~1 token per 10 XP
    tokens_awarded = max(1, xp_awarded // 10)
    state["player"]["tokens"] += tokens_awarded

    # Level up check
    prev_level = state["player"]["level"]
    new_level_num, _, _, _ = compute_level(state["player"]["xp"])
    level_up = new_level_num > prev_level
    if level_up:
        state["player"]["level"] = new_level_num

    new_level_name = get_level_name(state["player"]["xp"])

    return {
        "xp_awarded": xp_awarded,
        "base_xp": base_xp,
        "multiplier": multiplier,
        "tokens_awarded": tokens_awarded,
        "level_up": level_up,
        "new_level": new_level_num,
        "new_level_name": new_level_name,
        "prev_xp": prev_xp,
        "reason": reason,
    }

# ── Streak engine ──────────────────────────────────────────────────────────────
def check_and_update_streak(state: dict) -> dict:
    """
    Compare today to last_study_date, update streak accordingly.
    Returns result dict. Updates state in place. Caller saves.
    """
    today = datetime.date.today()
    today_str = today.isoformat()
    last_str = state["streak"].get("last_study_date")

    result = {
        "streak_maintained": False,
        "shield_used": False,
        "streak_broken": False,
        "fire_activated": False,
        "current_streak": state["streak"]["current"],
        "already_studied_today": False,
    }

    if last_str == today_str:
        # Already studied today
        result["already_studied_today"] = True
        result["streak_maintained"] = True
        result["current_streak"] = state["streak"]["current"]
        return result

    if last_str is None:
        # First ever session
        state["streak"]["current"] = 1
        state["streak"]["last_study_date"] = today_str
        result["streak_maintained"] = True
        result["current_streak"] = 1
        return result

    last_date = datetime.date.fromisoformat(last_str)
    gap = (today - last_date).days

    if gap == 1:
        # Perfect — consecutive day
        state["streak"]["current"] += 1
        state["streak"]["last_study_date"] = today_str
        result["streak_maintained"] = True
    elif gap == 2 and state["streak"]["shields_remaining"] > 0:
        # One day gap — use shield
        state["streak"]["current"] += 1
        state["streak"]["shields_remaining"] -= 1
        state["streak"]["last_study_date"] = today_str
        result["streak_maintained"] = True
        result["shield_used"] = True
    else:
        # Streak broken
        state["streak"]["current"] = 1
        state["streak"]["last_study_date"] = today_str
        result["streak_broken"] = gap > 1
        result["streak_maintained"] = False

    # Update longest
    if state["streak"]["current"] > state["streak"]["longest"]:
        state["streak"]["longest"] = state["streak"]["current"]

    # Fire status
    fire_was = state["streak"].get("fire_active", False)
    fire_now = state["streak"]["current"] >= 3
    state["streak"]["fire_active"] = fire_now
    if fire_now and not fire_was:
        result["fire_activated"] = True

    result["current_streak"] = state["streak"]["current"]
    return result

def get_streak_warning(state: dict) -> str | None:
    """Return loss-aversion warning if streak is at risk, else None."""
    last_str = state["streak"].get("last_study_date")
    if not last_str:
        return None
    streak = state["streak"]["current"]
    if streak == 0:
        return None

    today = datetime.date.today()
    last_date = datetime.date.fromisoformat(last_str)
    gap = (today - last_date).days

    if gap == 0:
        return None  # Already studied today

    now_hour = datetime.datetime.now().hour
    hours_left = max(0, 24 - now_hour)

    if gap == 1:
        if hours_left <= 6:
            return (
                f"DANGER: Your {streak}-day streak dies in {hours_left}h. "
                f"Open ANY notebook NOW to save it."
            )
        elif hours_left <= 12:
            return (
                f"WARNING: {streak}-day streak at risk — {hours_left}h left today."
            )
    elif gap == 2 and state["streak"]["shields_remaining"] > 0:
        return (
            f"SHIELD ALERT: You missed yesterday. A streak shield will auto-protect "
            f"your {streak}-day streak — but only if you study TODAY."
        )
    return None

# ── Token economy ──────────────────────────────────────────────────────────────
def spend_tokens(state: dict, amount: int, reason: str) -> None:
    """Deduct tokens. Raises ValueError if insufficient."""
    if state["player"]["tokens"] < amount:
        raise ValueError(
            f"Not enough tokens: need {amount}, have {state['player']['tokens']}. "
            f"Keep studying to earn more."
        )
    state["player"]["tokens"] -= amount

# ── Badge engine ───────────────────────────────────────────────────────────────
BADGES: dict[str, dict] = {
    "first_blood":    {"name": "First Blood",      "desc": "Complete your first notebook",      "rarity": "common"},
    "on_fire":        {"name": "On Fire",           "desc": "Maintain a 3-day streak",           "rarity": "uncommon"},
    "week_warrior":   {"name": "Week Warrior",      "desc": "7-day streak",                      "rarity": "rare"},
    "module_master":  {"name": "Module Master",     "desc": "Pass a boss battle",                "rarity": "rare"},
    "speed_runner":   {"name": "Speed Runner",      "desc": "Beat boss in under 30 min",         "rarity": "rare"},
    "comeback_kid":   {"name": "Comeback Kid",      "desc": "Resume after a 3+ day break",       "rarity": "uncommon"},
    "night_owl":      {"name": "Night Owl",         "desc": "Study after 11pm",                  "rarity": "uncommon"},
    "early_bird":     {"name": "Early Bird",        "desc": "Study before 7am",                  "rarity": "uncommon"},
    "half_way":       {"name": "Half Way There",    "desc": "Master 5+ modules",                 "rarity": "epic"},
    "perfectionist":  {"name": "Perfectionist",     "desc": "Score 100% on 3+ quizzes",          "rarity": "epic"},
    "legend_streak":  {"name": "30-Day Legend",     "desc": "30-day streak",                     "rarity": "legendary"},
    "perfect_run":    {"name": "Perfect Run",       "desc": "All bosses at 100%",                "rarity": "legendary"},
    "legend_tier":    {"name": "Legend Tier",       "desc": "Reach level 10 (AI Legend)",        "rarity": "legendary"},
    "interview_ready":{"name": "Interview Ready",   "desc": "Master all 11 modules",             "rarity": "legendary"},
}

RARITY_EMOJI = {"common": "⚪", "uncommon": "🟢", "rare": "🔵", "epic": "🟣", "legendary": "🟡"}

def check_and_award_badges(state: dict, event: dict) -> list[str]:
    """
    Evaluate badge conditions. Returns list of newly awarded badge IDs.
    event: {"type": str, ...extra context}
    """
    already = set(state["badges"])
    new_badges: list[str] = []

    def _grant(badge_id: str) -> None:
        if badge_id not in already:
            state["badges"].append(badge_id)
            already.add(badge_id)
            new_badges.append(badge_id)

    etype = event.get("type", "")
    modules_mastered = [
        m for m, d in state["modules"].items() if d.get("boss_passed")
    ]

    # First Blood
    total_completed = sum(
        len(d["notebooks_completed"]) for d in state["modules"].values()
    )
    if total_completed >= 1:
        _grant("first_blood")

    # Streak badges
    streak = state["streak"]["current"]
    if streak >= 3:
        _grant("on_fire")
    if streak >= 7:
        _grant("week_warrior")
    if streak >= 30:
        _grant("legend_streak")

    # Comeback: streak broken (gap) then resumed
    if etype == "streak_broken":
        _grant("comeback_kid")

    # Boss badges
    if etype == "boss_complete":
        _grant("module_master")
        time_mins = event.get("time_minutes", 999)
        if time_mins < 30:
            _grant("speed_runner")

    # Milestones
    if len(modules_mastered) >= 5:
        _grant("half_way")
    if len(modules_mastered) == 11:
        _grant("interview_ready")

    # Perfect quiz
    if state["quiz_history"]["perfect_quiz_count"] >= 3:
        _grant("perfectionist")

    # All bosses 100%
    all_boss_scores = [
        state["modules"][m].get("boss_score", 0) or 0
        for m in ALL_MODULES
        if state["modules"][m].get("boss_passed")
    ]
    if len(all_boss_scores) == 11 and all(s >= 1.0 for s in all_boss_scores):
        _grant("perfect_run")

    # Level 10
    if state["player"]["level"] >= 10:
        _grant("legend_tier")

    # Time-based
    hour = datetime.datetime.now().hour
    if etype in ("session_start", "session_end"):
        if hour >= 23 or hour < 1:
            _grant("night_owl")
        if 5 <= hour < 7:
            _grant("early_bird")

    return new_badges

# ── Skill tree and boss unlock ─────────────────────────────────────────────────
def check_module_unlock(state: dict) -> list[str]:
    """
    Unlock modules whose prereqs are satisfied (boss_passed on prereq).
    Returns list of newly unlocked module IDs.
    """
    newly_unlocked: list[str] = []
    for mid, prereqs in PREREQ_MAP.items():
        if state["modules"][mid]["unlocked"]:
            continue
        if all(state["modules"][p]["boss_passed"] for p in prereqs):
            state["modules"][mid]["unlocked"] = True
            if mid not in state["skill_tree"]["unlocked_nodes"]:
                state["skill_tree"]["unlocked_nodes"].append(mid)
            newly_unlocked.append(mid)
    return newly_unlocked

def check_boss_unlock(state: dict, module_id: str) -> bool:
    """
    Unlock boss if all study notebooks in the module are completed.
    Sets boss_unlocked and boss_expires_at. Returns True if newly unlocked.
    """
    mod = state["modules"][module_id]
    if mod["boss_unlocked"]:
        return False

    required = set(MODULE_STUDY_NOTEBOOKS.get(module_id, []))
    completed = set(mod["notebooks_completed"])
    if not required.issubset(completed):
        return False

    mod["boss_unlocked"] = True
    expiry = datetime.datetime.now() + datetime.timedelta(hours=BOSS_EXPIRY_HOURS)
    mod["boss_expires_at"] = expiry.isoformat()
    return True

def is_boss_expired(state: dict, module_id: str) -> bool:
    """Check if the boss battle window has closed."""
    mod = state["modules"][module_id]
    if not mod["boss_unlocked"] or mod["boss_attempted"]:
        return False
    exp_str = mod.get("boss_expires_at")
    if not exp_str:
        return False
    return datetime.datetime.now() > datetime.datetime.fromisoformat(exp_str)

def boss_hours_remaining(state: dict, module_id: str) -> float | None:
    """Return hours remaining on boss expiry, or None if not applicable."""
    mod = state["modules"][module_id]
    if not mod["boss_unlocked"] or mod["boss_attempted"]:
        return None
    exp_str = mod.get("boss_expires_at")
    if not exp_str:
        return None
    delta = datetime.datetime.fromisoformat(exp_str) - datetime.datetime.now()
    return max(0.0, delta.total_seconds() / 3600)

def record_boss_result(
    state: dict,
    module_id: str,
    score_pct: float,
    time_minutes: float,
) -> dict:
    """Record boss attempt, award XP and tokens. Returns award summary."""
    mod = state["modules"][module_id]
    mod["boss_attempted"] = True
    passed = score_pct >= 0.5
    mod["boss_passed"] = passed
    mod["boss_score"] = score_pct
    mod["boss_time_minutes"] = time_minutes

    xp_result = award_xp(state, XP_MOCK_INTERVIEW, f"boss_{module_id}")
    state["player"]["tokens"] += TOKENS_BOSS_COMPLETE

    # Check unlocks
    newly_unlocked = check_module_unlock(state)

    # Check badges
    new_badges = check_and_award_badges(state, {
        "type": "boss_complete",
        "module": module_id,
        "score": score_pct,
        "time_minutes": time_minutes,
    })

    state["boss_battles"]["completed"].append({
        "module": module_id,
        "score_pct": score_pct,
        "time_minutes": time_minutes,
        "passed": passed,
        "timestamp": datetime.datetime.now().isoformat(),
    })

    return {
        "passed": passed,
        "score_pct": score_pct,
        "xp_result": xp_result,
        "tokens_bonus": TOKENS_BOSS_COMPLETE,
        "newly_unlocked_modules": newly_unlocked,
        "new_badges": new_badges,
    }

# ── Spaced repetition (SM-2) ───────────────────────────────────────────────────
def get_due_concepts(state: dict, module_id: str | None = None) -> list[dict]:
    """Return concepts due for review today, sorted by most overdue first."""
    today = datetime.date.today()
    due: list[dict] = []

    qh = state["quiz_history"]
    for cid, next_review_str in qh["concept_next_review"].items():
        next_date = datetime.date.fromisoformat(next_review_str)
        if next_date <= today:
            days_overdue = (today - next_date).days
            # Filter by module if specified
            # concept_id format: <module_abbrev>_<concept>
            if module_id:
                # crude prefix check
                pass  # include all for now; quizzes embed module info
            due.append({
                "concept_id": cid,
                "next_review": next_review_str,
                "days_overdue": days_overdue,
                "ease_factor": qh["concept_ease_factor"].get(cid, SR_INITIAL_EASE),
                "interval_days": qh["concept_interval_days"].get(cid, 1),
            })

    due.sort(key=lambda x: x["days_overdue"], reverse=True)
    return due

def record_quiz_answer(state: dict, concept_id: str, quality: int) -> dict:
    """
    Update SM-2 data for a concept.
    quality: 0-5 (0=blackout, 5=perfect).
    Returns dict with new interval and ease_factor.
    """
    qh = state["quiz_history"]
    ease = qh["concept_ease_factor"].get(concept_id, SR_INITIAL_EASE)
    interval = qh["concept_interval_days"].get(concept_id, 1)
    review_count = qh["concept_review_count"].get(concept_id, 0)

    # SM-2 ease update
    new_ease = ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)
    new_ease = max(SR_MIN_EASE, new_ease)

    if quality >= 3:
        if review_count == 0:
            new_interval = 1
        elif review_count == 1:
            new_interval = 6
        else:
            new_interval = round(interval * new_ease)
    else:
        # Wrong answer: reset
        new_interval = 1
        new_ease = SR_INITIAL_EASE  # don't penalize ease too harshly on reset

    next_review = (datetime.date.today() + datetime.timedelta(days=new_interval)).isoformat()

    qh["concept_ease_factor"][concept_id] = round(new_ease, 3)
    qh["concept_interval_days"][concept_id] = new_interval
    qh["concept_next_review"][concept_id] = next_review
    qh["concept_last_correct"][concept_id] = datetime.date.today().isoformat()
    qh["concept_review_count"][concept_id] = review_count + 1

    # XP for quiz
    xp_result = None
    if quality >= 3:
        base = XP_QUIZ_PASS if review_count == 0 else XP_QUIZ_RETRY
        xp_result = award_xp(state, base, f"quiz_{concept_id}")
        state["player"]["tokens"] += TOKENS_QUIZ_PASS

    return {
        "next_review_date": next_review,
        "interval_days": new_interval,
        "ease_factor": new_ease,
        "xp_result": xp_result,
    }

# ── Cohort percentile ──────────────────────────────────────────────────────────
def get_cohort_percentile(state: dict) -> int:
    """Compute percentile vs fictional cohort (mean=400 XP, std=300 XP, N=1247)."""
    xp = state["player"]["xp"]
    mean, std = 400, 300
    # CDF of normal distribution via error function
    z = (xp - mean) / (std * math.sqrt(2))
    percentile = int(50 * (1 + math.erf(z)))
    return max(1, min(99, percentile))

# ── Session management ─────────────────────────────────────────────────────────
def start_session(module_id: str, notebook_name: str) -> dict:
    """
    Call at the top of any notebook. Returns context dict for widgets.
    """
    state = load_state()

    # First session bonus
    xp_result = None
    if not state["player"]["first_session_bonus_given"]:
        state["player"]["first_session_bonus_given"] = True
        xp_result = award_xp(state, XP_FIRST_SESSION, "welcome_bonus")

    streak_result = check_and_update_streak(state)

    # Streak bonus XP
    if not streak_result["already_studied_today"] and state["streak"]["current"] > 1:
        sr = award_xp(state, XP_STREAK_BONUS, "streak_bonus")
        if xp_result:
            xp_result["xp_awarded"] += sr["xp_awarded"]
        else:
            xp_result = sr

    state["player"]["total_sessions"] += 1
    state["meta"]["session_start"] = datetime.datetime.now().isoformat()

    # Check streak break badge
    if streak_result.get("streak_broken"):
        check_and_award_badges(state, {"type": "streak_broken"})

    check_and_award_badges(state, {"type": "session_start"})

    # Boss status for this module
    boss_hours = boss_hours_remaining(state, module_id)
    boss_expired = is_boss_expired(state, module_id)

    due_reviews = get_due_concepts(state)
    level_num, level_thresh, xp_into, xp_needed = compute_level(state["player"]["xp"])
    percentile = get_cohort_percentile(state)

    save_state(state)

    return {
        "state": state,
        "module_id": module_id,
        "notebook_name": notebook_name,
        "streak_result": streak_result,
        "warning": get_streak_warning(state),
        "due_reviews": due_reviews,
        "boss_hours_remaining": boss_hours,
        "boss_expired": boss_expired,
        "level_num": level_num,
        "level_name": get_level_name(state["player"]["xp"]),
        "xp_into_level": xp_into,
        "xp_needed_for_next": xp_needed,
        "percentile": percentile,
        "welcome_xp": xp_result,
        "session_start_time": datetime.datetime.now(),
    }

def end_session(module_id: str, notebook_name: str, context: dict) -> dict:
    """
    Call at the bottom of any notebook. Awards XP, updates completion,
    regenerates dashboard. Returns summary dict.
    """
    from coach.dashboard import write_dashboard  # avoid circular import

    state = context["state"]
    # Reload fresh state in case user ran cells multiple times
    state = load_state()

    # Award reading XP (idempotent: only if notebook not already completed)
    mod = state["modules"][module_id]
    is_new_completion = notebook_name not in mod["notebooks_completed"]
    xp_result = None

    if is_new_completion:
        # Mark as boss notebook or study notebook
        is_boss_nb = "interviewer_perspective" in notebook_name
        if not is_boss_nb:
            mod["notebooks_completed"].append(notebook_name)
            xp_result = award_xp(state, XP_READING_NOTEBOOK, f"completed_{notebook_name}")
            # Check if boss should unlock
            check_boss_unlock(state, module_id)

    # Time spent
    start_time = context.get("session_start_time")
    minutes_spent = 0
    if start_time:
        minutes_spent = int((datetime.datetime.now() - start_time).total_seconds() / 60)
        state["player"]["total_study_minutes"] = (
            state["player"].get("total_study_minutes", 0) + minutes_spent
        )

    # Badges
    new_badges = check_and_award_badges(state, {
        "type": "session_end",
        "module": module_id,
        "notebook": notebook_name,
    })

    # Log session
    state["session_log"].append({
        "module": module_id,
        "notebook": notebook_name,
        "minutes": minutes_spent,
        "xp": xp_result["xp_awarded"] if xp_result else 0,
        "date": datetime.date.today().isoformat(),
    })
    # Keep last 30 sessions
    state["session_log"] = state["session_log"][-30:]

    level_num, _, xp_into, xp_needed = compute_level(state["player"]["xp"])
    level_up = level_num > context.get("level_num", 1)

    save_state(state)
    write_dashboard(state)

    boss_unlocked_now = (
        state["modules"][module_id]["boss_unlocked"]
        and not context["state"]["modules"][module_id]["boss_unlocked"]
    )

    return {
        "state": state,
        "xp_result": xp_result,
        "new_badges": new_badges,
        "level_up": level_up,
        "new_level": level_num,
        "new_level_name": get_level_name(state["player"]["xp"]),
        "xp_into_level": xp_into,
        "xp_needed_for_next": xp_needed,
        "is_new_completion": is_new_completion,
        "boss_unlocked_now": boss_unlocked_now,
        "minutes_spent": minutes_spent,
        "module_id": module_id,
    }
