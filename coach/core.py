"""
coach/core.py — XP engine, SM-2 spaced repetition, badge system,
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

# ── Token economy ──────────────────────────────────────────────────────────────
TOKENS_PER_SESSION = 5
TOKENS_QUIZ_PASS = 2
TOKENS_BOSS_COMPLETE = 20
TOKENS_PERFECT_SCORE = 10

COST_HINT_REVEAL = 5
COST_SKIP_PREREQ = 50

# ── Boss battle config ─────────────────────────────────────────────────────────
BOSS_EXPIRY_HOURS = 48
BOSS_TIME_LIMIT_MINUTES = 45

# ── SM-2 defaults ─────────────────────────────────────────────────────────────
SR_INITIAL_EASE = 2.5
SR_MIN_EASE = 1.3

# ── Build-a-Transformer day system ───────────────────────────────────────────
XP_DAY_COMPLETE = 15
XP_MILESTONE_COMPLETE = 75
TOKENS_DAY_COMPLETE = 2
TOKENS_MILESTONE = 10

# Maps each day to its metadata. component_id links to model_status in state.json.
# phase: 1=Attention, 2=Multi-Head, 3=Position, 4=Full Model, 5=Interview
# milestone days have is_milestone=True.
BUILD_DAYS: dict[int, dict] = {
    # Phase 1: Attention
    1:  {"title": "Words to Numbers",          "component_id": "embedding",          "phase": 1, "is_milestone": False},
    2:  {"title": "Asking Questions",          "component_id": "qkv_projection",     "phase": 1, "is_milestone": False},
    3:  {"title": "Who Matches Who",           "component_id": "dot_product",        "phase": 1, "is_milestone": False},
    4:  {"title": "Keeping Scores Fair",       "component_id": "scaling",            "phase": 1, "is_milestone": False},
    5:  {"title": "Picking Winners",           "component_id": "softmax",            "phase": 1, "is_milestone": False},
    6:  {"title": "Mixing the Answer",         "component_id": "attention_output",   "phase": 1, "is_milestone": False},
    7:  {"title": "Your First Attention",      "component_id": None,                 "phase": 1, "is_milestone": True},
    # Phase 2: Multi-Head
    8:  {"title": "Why One Isn't Enough",      "component_id": None,                 "phase": 2, "is_milestone": False},
    9:  {"title": "Splitting Into Teams",      "component_id": "multi_head_split",   "phase": 2, "is_milestone": False},
    10: {"title": "Parallel Attention",        "component_id": None,                 "phase": 2, "is_milestone": False},
    11: {"title": "The Team Report",           "component_id": "head_concat_project","phase": 2, "is_milestone": False},
    12: {"title": "Multi-Head Attention",      "component_id": None,                 "phase": 2, "is_milestone": True},
    # Phase 3: Position
    13: {"title": "The Shuffle Problem",       "component_id": None,                 "phase": 3, "is_milestone": False},
    14: {"title": "Wave Patterns",             "component_id": "positional_encoding","phase": 3, "is_milestone": False},
    15: {"title": "Adding Position",           "component_id": None,                 "phase": 3, "is_milestone": False},
    16: {"title": "Modern Alternatives",       "component_id": None,                 "phase": 3, "is_milestone": False},
    17: {"title": "Position-Aware Model",      "component_id": None,                 "phase": 3, "is_milestone": True},
    # Phase 4: Full Model
    18: {"title": "Skip Connections",          "component_id": "residual_connection","phase": 4, "is_milestone": False},
    19: {"title": "Keeping Numbers Calm",      "component_id": "layer_norm",         "phase": 4, "is_milestone": False},
    20: {"title": "The Thinking Layer",        "component_id": "ffn",                "phase": 4, "is_milestone": False},
    21: {"title": "Building the Block",        "component_id": "transformer_block",  "phase": 4, "is_milestone": False},
    22: {"title": "No Peeking",               "component_id": "causal_mask",         "phase": 4, "is_milestone": False},
    23: {"title": "Learning to Write",         "component_id": "training_loop",      "phase": 4, "is_milestone": False},
    24: {"title": "Your Transformer Writes!",  "component_id": "text_generation",    "phase": 4, "is_milestone": True},
    # Phase 5: Interview Ready
    25: {"title": "Explain Attention",         "component_id": None,                 "phase": 5, "is_milestone": False},
    26: {"title": "The Cost of Attention",     "component_id": None,                 "phase": 5, "is_milestone": False},
    27: {"title": "Encoder vs Decoder",        "component_id": None,                 "phase": 5, "is_milestone": False},
    28: {"title": "Break It to Prove It",      "component_id": None,                 "phase": 5, "is_milestone": False},
}

TOTAL_BUILD_DAYS = len(BUILD_DAYS)

PHASE_NAMES = {
    1: "Attention",
    2: "Multi-Head",
    3: "Position",
    4: "Full Model",
    5: "Interview Ready",
}

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
    # RAG
    "rag",
    # Fine-tuning
    "fine-tuning",
    # Reinforcement Learning
    "rl-fundamentals",
    # Mock Interview System — fundamentals topics + the two net-new tracks.
    # Added so interview XP flows through record_boss_result like every other
    # module. These have no skill-tree prereqs (see ALWAYS_UNLOCKED).
    "neural-networks",
    "prompt-engineering",
    "multimodal",
    "evaluation",
    "deployment",
    "ai-agents",
    "frontier-research",
    "behavioral",
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
    # RAG
    "rag": [
        "01_what_is_rag.ipynb",
    ],
    # Fine-tuning
    "fine-tuning": [
        "01_what_is_fine_tuning.ipynb",
        "02_full_fine_tuning.ipynb",
        "02_full_fine_tuning_experiments.ipynb",
        "03_lora.ipynb",
        "03_lora_experiments.ipynb",
        "04_qlora.ipynb",
        "05_instruction_tuning.ipynb",
        "05_instruction_tuning_experiments.ipynb",
    ],
    # Reinforcement Learning
    "rl-fundamentals": [
        "01_what_is_reinforcement_learning.ipynb",
        "02_markov_decision_processes.ipynb",
        "03_rewards_and_returns.ipynb",
        "04_policies_and_value_functions.ipynb",
        "05_bellman_equations.ipynb",
    ],
}

# ── Default state factory ──────────────────────────────────────────────────────
def _default_state() -> dict:
    modules = {}
    # Foundations modules are always unlocked; ML Design skill tree starts at module 01
    ALWAYS_UNLOCKED = {"01-ml-design-prep", "rnn", "transformers", "rag", "fine-tuning", "rl-fundamentals",
                       "neural-networks", "prompt-engineering", "multimodal", "evaluation", "deployment",
                       "ai-agents", "frontier-research", "behavioral"}
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
    """Load state.json, creating with defaults if absent.
    Backfills any modules added to ALL_MODULES since the file was created."""
    if not STATE_PATH.exists():
        state = _default_state()
        save_state(state)
        return state
    with open(STATE_PATH, "r") as f:
        state = json.load(f)
    # Backfill modules added after this state file was created
    ALWAYS_UNLOCKED = {"01-ml-design-prep", "rnn", "transformers", "rag", "fine-tuning", "rl-fundamentals",
                       "neural-networks", "prompt-engineering", "multimodal", "evaluation", "deployment",
                       "ai-agents", "frontier-research", "behavioral"}
    changed = False
    for mid in ALL_MODULES:
        if mid not in state.get("modules", {}):
            state.setdefault("modules", {})[mid] = {
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
            changed = True
    if changed:
        save_state(state)
    return state

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
    Award XP. Updates state in place.
    Returns summary dict. Caller must save_state().
    """
    xp_awarded = base_xp

    prev_xp = state["player"]["xp"]
    state["player"]["xp"] += xp_awarded

    tokens_awarded = max(1, xp_awarded // 10)
    state["player"]["tokens"] += tokens_awarded

    prev_level = state["player"]["level"]
    new_level_num, _, _, _ = compute_level(state["player"]["xp"])
    level_up = new_level_num > prev_level
    if level_up:
        state["player"]["level"] = new_level_num

    new_level_name = get_level_name(state["player"]["xp"])

    return {
        "xp_awarded": xp_awarded,
        "base_xp": base_xp,
        "multiplier": 1.0,
        "tokens_awarded": tokens_awarded,
        "level_up": level_up,
        "new_level": new_level_num,
        "new_level_name": new_level_name,
        "prev_xp": prev_xp,
        "reason": reason,
    }

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
    "module_master":  {"name": "Module Master",     "desc": "Pass a boss battle",                "rarity": "rare"},
    "speed_runner":   {"name": "Speed Runner",      "desc": "Beat boss in under 30 min",         "rarity": "rare"},
    "night_owl":      {"name": "Night Owl",         "desc": "Study after 11pm",                  "rarity": "uncommon"},
    "early_bird":     {"name": "Early Bird",        "desc": "Study before 7am",                  "rarity": "uncommon"},
    "half_way":       {"name": "Half Way There",    "desc": "Master 5+ modules",                 "rarity": "epic"},
    "perfectionist":  {"name": "Perfectionist",     "desc": "Score 100% on 3+ quizzes",          "rarity": "epic"},
    "perfect_run":    {"name": "Perfect Run",       "desc": "All bosses at 100%",                "rarity": "legendary"},
    "legend_tier":    {"name": "Legend Tier",       "desc": "Reach level 10 (AI Legend)",        "rarity": "legendary"},
    "interview_ready":{"name": "Interview Ready",   "desc": "Master all 11 modules",             "rarity": "legendary"},
    # Build-a-Transformer badges
    "first_component": {"name": "First Component",  "desc": "Build your first transformer piece", "rarity": "common"},
    "full_transformer":{"name": "Full Transformer",  "desc": "Build all transformer components",   "rarity": "epic"},
    "build_master":    {"name": "Build Master",      "desc": "Complete all build milestones",       "rarity": "legendary"},
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

    # Build-a-Transformer badges
    model_status = state.get("model_status", {}).get("components", {})
    built_count = sum(1 for c in model_status.values() if c.get("built"))
    if built_count >= 1:
        _grant("first_component")
    total_components = sum(1 for d in BUILD_DAYS.values() if d["component_id"])
    if built_count >= total_components:
        _grant("full_transformer")
    milestones_done = state.get("quests", {}).get("milestones_completed", [])
    if len(milestones_done) >= 5:
        _grant("build_master")

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

# ── Build-a-Transformer engine ────────────────────────────────────────────────
def is_day_available(state: dict, day_number: int) -> bool:
    """Check if a build day is available (previous day completed or day 1)."""
    if day_number == 1:
        return True
    completed = set(state.get("quests", {}).get("completed_days", []))
    return (day_number - 1) in completed


def get_build_progress(state: dict) -> dict:
    """Return current build progress: days done, phase, components, pct."""
    completed_days = set(state.get("quests", {}).get("completed_days", []))
    milestones = state.get("quests", {}).get("milestones_completed", [])
    model = state.get("model_status", {}).get("components", {})
    built = [cid for cid, info in model.items() if info.get("built")]

    # Current day = lowest incomplete day
    current_day = 1
    for d in range(1, TOTAL_BUILD_DAYS + 1):
        if d not in completed_days:
            current_day = d
            break
    else:
        current_day = TOTAL_BUILD_DAYS  # all done

    phase = BUILD_DAYS.get(current_day, {}).get("phase", 1)
    pct = len(completed_days) / TOTAL_BUILD_DAYS * 100

    return {
        "current_day": current_day,
        "days_completed": len(completed_days),
        "total_days": TOTAL_BUILD_DAYS,
        "completed_days": sorted(completed_days),
        "phase": phase,
        "phase_name": PHASE_NAMES.get(phase, "Unknown"),
        "components_built": built,
        "milestones_completed": milestones,
        "pct_complete": round(pct, 1),
    }


def complete_build_day(state: dict, day_number: int) -> dict:
    """
    Mark a build day as complete. Awards XP/tokens, marks component,
    checks milestone, checks badges. Returns summary.
    """
    if day_number not in BUILD_DAYS:
        return {"error": f"Day {day_number} does not exist."}

    if not is_day_available(state, day_number):
        prev = day_number - 1
        return {"error": f"Day {prev} must be completed first."}

    # Init quests tracking if absent
    quests = state.setdefault("quests", {})
    completed_days = quests.setdefault("completed_days", [])

    if day_number in completed_days:
        return {"already_done": True, "day": day_number, "title": BUILD_DAYS[day_number]["title"]}

    day_info = BUILD_DAYS[day_number]
    completed_days.append(day_number)
    quests["current_day"] = day_number + 1
    quests["total_quests_completed"] = len(completed_days)

    # Mark component as built
    comp_id = day_info["component_id"]
    if comp_id:
        model = state.setdefault("model_status", {}).setdefault("components", {})
        if comp_id in model:
            model[comp_id]["built"] = True
            model[comp_id]["day_built"] = day_number

    # Determine XP
    is_milestone = day_info["is_milestone"]
    base_xp = XP_MILESTONE_COMPLETE if is_milestone else XP_DAY_COMPLETE
    xp_result = award_xp(state, base_xp, f"build_day_{day_number}")

    # Token bonus
    token_bonus = TOKENS_MILESTONE if is_milestone else TOKENS_DAY_COMPLETE
    state["player"]["tokens"] += token_bonus

    # Record milestone
    if is_milestone:
        milestones = quests.setdefault("milestones_completed", [])
        milestones.append(day_info["phase"])

    # Check badges
    new_badges = check_and_award_badges(state, {"type": "build_day_complete", "day": day_number})

    # Log session
    state["session_log"].append({
        "module": "transformers",
        "notebook": f"build_day_{day_number:02d}",
        "minutes": 0,
        "xp": xp_result["xp_awarded"],
        "date": datetime.date.today().isoformat(),
    })
    state["session_log"] = state["session_log"][-30:]

    # Next day info
    next_day = day_number + 1
    next_info = BUILD_DAYS.get(next_day)

    save_state(state)

    return {
        "day": day_number,
        "title": day_info["title"],
        "phase": day_info["phase"],
        "phase_name": PHASE_NAMES[day_info["phase"]],
        "is_milestone": is_milestone,
        "component_built": comp_id,
        "xp_result": xp_result,
        "token_bonus": token_bonus,
        "new_badges": new_badges,
        "next_day": next_day if next_info else None,
        "next_title": next_info["title"] if next_info else "You're done!",
        "progress": get_build_progress(state),
    }


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

    state["player"]["total_sessions"] += 1
    state["meta"]["session_start"] = datetime.datetime.now().isoformat()

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
        "warning": None,
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
