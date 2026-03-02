"""
coach/dashboard.py — generates DASHBOARD.md from state.
Called automatically at the end of every session.
Never edit DASHBOARD.md by hand — it will be overwritten.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DASHBOARD_PATH = REPO_ROOT / "DASHBOARD.md"

RARITY_SYMBOL = {"common": "⚪", "uncommon": "🟢", "rare": "🔵", "epic": "🟣", "legendary": "🟡"}

MODULE_LABELS = {
    "01-ml-design-prep":            "01  ML Design Prep Framework",
    "02-visual-search":             "02  Visual Search",
    "03-google-street-view":        "03  Google Street View",
    "04-youtube-video-search":      "04  YouTube Video Search",
    "05-harmful-content-detection": "05  Harmful Content Detection",
    "06-video-recommendation":      "06  Video Recommendation",
    "07-event-recommendation":      "07  Event Recommendation",
    "08-ad-click-prediction":       "08  Ad Click Prediction",
    "09-similar-listing":           "09  Similar Listing",
    "10-personalized-news-feed":    "10  Personalized News Feed",
    "11-people-you-may-know":       "11  People You May Know",
}

ALL_MODULES = list(MODULE_LABELS.keys())

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

BADGES_META = {
    "first_blood":    {"name": "First Blood",      "rarity": "common"},
    "on_fire":        {"name": "On Fire",           "rarity": "uncommon"},
    "week_warrior":   {"name": "Week Warrior",      "rarity": "rare"},
    "module_master":  {"name": "Module Master",     "rarity": "rare"},
    "speed_runner":   {"name": "Speed Runner",      "rarity": "rare"},
    "comeback_kid":   {"name": "Comeback Kid",      "rarity": "uncommon"},
    "night_owl":      {"name": "Night Owl",         "rarity": "uncommon"},
    "early_bird":     {"name": "Early Bird",        "rarity": "uncommon"},
    "half_way":       {"name": "Half Way There",    "rarity": "epic"},
    "perfectionist":  {"name": "Perfectionist",     "rarity": "epic"},
    "legend_streak":  {"name": "30-Day Legend",     "rarity": "legendary"},
    "perfect_run":    {"name": "Perfect Run",       "rarity": "legendary"},
    "legend_tier":    {"name": "Legend Tier",       "rarity": "legendary"},
    "interview_ready":{"name": "Interview Ready",   "rarity": "legendary"},
}

BADGE_DESCS = {
    "first_blood":    "Complete your first notebook",
    "on_fire":        "Maintain a 3-day streak",
    "week_warrior":   "7-day streak",
    "module_master":  "Pass a boss battle",
    "speed_runner":   "Beat boss in under 30 min",
    "comeback_kid":   "Resume after a 3+ day break",
    "night_owl":      "Study after 11pm",
    "early_bird":     "Study before 7am",
    "half_way":       "Master 5+ modules",
    "perfectionist":  "Score 100% on 3+ quizzes",
    "legend_streak":  "30-day streak",
    "perfect_run":    "All bosses at 100%",
    "legend_tier":    "Reach level 10 (AI Legend)",
    "interview_ready":"Master all 11 modules",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ascii_bar(filled: int, total: int, width: int = 30) -> str:
    n = round(width * filled / max(total, 1))
    n = max(0, min(n, width))
    return "[" + "#" * n + "." * (width - n) + "]"

def _compute_level(xp: int) -> tuple[int, str, int, int]:
    current = LEVELS[0]
    for lvl in LEVELS:
        if xp >= lvl[1]:
            current = lvl
        else:
            break
    idx = next(i for i, l in enumerate(LEVELS) if l[0] == current[0])
    xp_into = xp - current[1]
    xp_needed = LEVELS[idx + 1][1] - current[1] if idx + 1 < len(LEVELS) else 9999
    return current[0], current[2], xp_into, xp_needed

def _streak_multiplier_label(streak: int) -> str:
    if streak >= 30: return "3.0x LEGENDARY"
    if streak >= 14: return "2.0x"
    if streak >= 7:  return "1.5x"
    if streak >= 3:  return "1.25x"
    return "1.0x"

def _cohort_percentile(xp: int) -> int:
    mean, std = 400, 300
    z = (xp - mean) / (std * math.sqrt(2))
    p = int(50 * (1 + math.erf(z)))
    return max(1, min(99, p))

def _cohort_rank(xp: int, cohort_size: int = 1247) -> int:
    pct = _cohort_percentile(xp)
    rank = round(cohort_size * (1 - pct / 100))
    return max(1, rank)


# ── Section builders ───────────────────────────────────────────────────────────

def _section_header(state: dict) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sessions = state["player"]["total_sessions"]
    xp = state["player"]["xp"]
    level_num, level_name, _, _ = _compute_level(xp)
    return f"""# ML Interview Training Dashboard
> Last updated: {now} | Session #{sessions}

---

## Player Status

**Level {level_num} — {level_name}** | {xp:,} XP total | {state['player']['tokens']} Tokens | {state['streak']['shields_remaining']} Streak Shield(s)
"""


def _section_xp_bar(state: dict) -> str:
    xp = state["player"]["xp"]
    level_num, level_name, xp_into, xp_needed = _compute_level(xp)

    if level_num < 10:
        next_idx = level_num  # 0-indexed LEVELS list: level 1 is index 0
        next_name = LEVELS[next_idx][2]  # LEVELS[level_num] is level level_num+1
        bar = _ascii_bar(xp_into, xp_needed, width=40)
        return f"""**XP Progress → Level {level_num + 1} ({next_name}):**
{bar}  {xp_into:,} / {xp_needed:,} XP
"""
    else:
        return "**MAX LEVEL — AI Legend achieved.**\n"


def _section_streak(state: dict) -> str:
    streak = state["streak"]
    current = streak["current"]
    longest = streak["longest"]
    fire = streak.get("fire_active", False)
    multiplier = _streak_multiplier_label(current)
    fire_str = "  🔥 ON FIRE" if fire else ""

    # Build 21-day calendar
    last_str = streak.get("last_study_date")
    today = datetime.date.today()
    calendar_cells = []
    for i in range(20, -1, -1):
        day = today - datetime.timedelta(days=i)
        day_str = day.isoformat()
        if last_str and day_str <= last_str:
            # Rough approximation: assume studied on days within current streak
            gap = (today - day).days
            if gap < current:
                calendar_cells.append("[✓]")
            else:
                calendar_cells.append("[·]")
        else:
            calendar_cells.append("[·]")
    calendar = "".join(calendar_cells)

    # Loss aversion warning
    warning_line = ""
    if last_str and last_str != today.isoformat() and current > 0:
        hour = datetime.datetime.now().hour
        hours_left = max(0, 24 - hour)
        warning_line = f"\n> ⚠️  STREAK AT RISK — study in the next {hours_left}h or lose your {current}-day streak.\n"

    return f"""---

## Streak{fire_str}

**Current:** {current} days | **Longest:** {longest} days | **XP Multiplier:** {multiplier}

Last 21 days:
{calendar}
{warning_line}"""


def _section_skill_tree(state: dict) -> str:
    lines = ["---", "", "## Skill Tree", ""]
    modules_mastered = 0

    for i, mid in enumerate(ALL_MODULES):
        mod = state["modules"][mid]
        label = MODULE_LABELS[mid]
        unlocked = mod["unlocked"]
        boss_passed = mod.get("boss_passed", False)
        boss_unlocked = mod.get("boss_unlocked", False)
        boss_expires = mod.get("boss_expires_at")
        completed = mod["notebooks_completed"]

        # Determine status
        if boss_passed:
            modules_mastered += 1
            study_nbs = len(completed)
            status = f"[MASTERED ★]  Boss: DEFEATED  ({study_nbs} notebooks)"
        elif boss_unlocked:
            hours = None
            if boss_expires:
                delta = datetime.datetime.fromisoformat(boss_expires) - datetime.datetime.now()
                hours = max(0, delta.total_seconds() / 3600)
            if hours is not None and hours < 24:
                status = f"[BOSS UNLOCKED ⚠️]  Expires in {hours:.0f}h — ATTEMPT NOW"
            else:
                status = f"[BOSS UNLOCKED]  Ready to fight"
        elif unlocked:
            total_study = 4 if mid != "01-ml-design-prep" else 2
            if mid == "05-harmful-content-detection":
                total_study = 3
            done = len(completed)
            bar = _ascii_bar(done, total_study, width=10)
            status = f"[IN PROGRESS]  {done}/{total_study} notebooks  {bar}"
        else:
            status = "[LOCKED 🔒]"

        connector = "    |" if i < len(ALL_MODULES) - 1 else ""
        lines.append(f"  {label:<40} {status}")
        if connector and not boss_passed:
            lines.append(f"  {connector}")
        elif connector:
            lines.append(f"  {connector}")

    total = len(ALL_MODULES)
    overall_bar = _ascii_bar(modules_mastered, total, width=30)
    lines += [
        "",
        f"**Overall Progress:** {overall_bar}  {modules_mastered}/{total} modules mastered",
        "",
    ]
    return "\n".join(lines)


def _section_boss_battles(state: dict) -> str:
    lines = ["---", "", "## Boss Battles", "",
             "| Module | Status | Score | Time |",
             "|--------|--------|-------|------|"]

    scarcity_alerts = []
    now = datetime.datetime.now()

    for mid in ALL_MODULES:
        mod = state["modules"][mid]
        label = MODULE_LABELS[mid].split("  ", 1)[1]  # strip number
        if mod.get("boss_passed"):
            score = f"{mod['boss_score']*100:.0f}%" if mod.get("boss_score") else "—"
            time_s = f"{mod['boss_time_minutes']:.0f} min" if mod.get("boss_time_minutes") else "—"
            status = "DEFEATED ✓"
        elif mod.get("boss_unlocked"):
            expires = mod.get("boss_expires_at")
            if expires:
                delta = datetime.datetime.fromisoformat(expires) - now
                hours = max(0, delta.total_seconds() / 3600)
                status = f"UNLOCKED — {hours:.0f}h left"
                if hours < 24:
                    scarcity_alerts.append(
                        f"> ⚡ **SCARCITY ALERT:** {label} boss expires in {hours:.0f}h — do it now."
                    )
            else:
                status = "UNLOCKED"
            score, time_s = "—", "—"
        elif mod.get("unlocked"):
            status = "In progress"
            score, time_s = "—", "—"
        else:
            status = "Locked 🔒"
            score, time_s = "—", "—"

        lines.append(f"| {label} | {status} | {score} | {time_s} |")

    lines.append("")
    lines.extend(scarcity_alerts)
    lines.append("")
    return "\n".join(lines)


def _section_power_rankings(state: dict) -> str:
    xp = state["player"]["xp"]
    cohort_size = state["cohort"]["fictional_cohort_size"]
    percentile = _cohort_percentile(xp)
    rank = _cohort_rank(xp, cohort_size)
    top_pct = 100 - percentile

    your_bar = _ascii_bar(xp, 8200, width=30)
    avg_bar  = _ascii_bar(400, 8200, width=30)
    top_bar  = _ascii_bar(8200, 8200, width=30)

    weekly_xp = sum(
        s.get("xp", 0)
        for s in state.get("session_log", [])
        if s.get("date", "") >= (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
    )

    return f"""---

## Power Rankings (vs Cohort)

**You are in the top {top_pct}% of ML design studiers.**

Position: **#{rank} of {cohort_size}** active students

```
YOU  ({xp:>6,} XP)  {your_bar}
TOP  ( 8,200 XP)  {top_bar}
AVG  (   400 XP)  {avg_bar}
```

Your XP this week: **+{weekly_xp:,} XP**

"""


def _section_badges(state: dict) -> str:
    earned = set(state["badges"])
    lines = ["---", "", "## Badges", ""]
    for bid, meta in BADGES_META.items():
        sym = RARITY_SYMBOL[meta["rarity"]]
        check = "✓" if bid in earned else "·"
        locked = "" if bid in earned else "  [locked]"
        lines.append(f"  [{check}] {sym} **{meta['name']}** ({meta['rarity']}) — {BADGE_DESCS[bid]}{locked}")
    lines.append(f"\n**{len(earned)} / {len(BADGES_META)} badges earned**\n")
    return "\n".join(lines)


def _section_spaced_repetition(state: dict) -> str:
    today = datetime.date.today()
    qh = state["quiz_history"]
    due = []
    for cid, next_str in qh.get("concept_next_review", {}).items():
        next_date = datetime.date.fromisoformat(next_str)
        if next_date <= today:
            days_over = (today - next_date).days
            due.append((days_over, cid))
    due.sort(reverse=True)

    lines = ["---", "", "## Spaced Repetition Queue", ""]
    if not due:
        lines.append("No concepts due for review today. Keep studying to build your queue!")
    else:
        lines.append(f"**{len(due)} concept(s) due for review:**\n")
        for days_over, cid in due[:10]:
            tag = f"  ← {days_over}d overdue" if days_over > 0 else "  ← due today"
            lines.append(f"  - `{cid}`{tag}")
        if len(due) > 10:
            lines.append(f"  - ... and {len(due)-10} more")
        lines += [
            "",
            "Run a review session in any notebook:",
            "```python",
            "from coach.notebook_widgets import render_quiz_widget",
            "render_quiz_widget(due_only=True)",
            "```",
        ]
    lines.append("")
    return "\n".join(lines)


def _section_session_history(state: dict) -> str:
    log = state.get("session_log", [])
    recent = sorted(log, key=lambda s: s.get("date", ""), reverse=True)[:7]

    lines = ["---", "", "## Session History (Last 7)", "",
             "| Date | Module | Minutes | XP |",
             "|------|--------|---------|-----|"]
    for s in recent:
        date = s.get("date", "—")
        mod = s.get("module", "—").replace("-", " ").title()
        mins = s.get("minutes", 0)
        xp = s.get("xp", 0)
        lines.append(f"| {date} | {mod} | {mins} min | +{xp} XP |")
    lines.append("")
    return "\n".join(lines)


def _section_footer() -> str:
    return (
        "---\n"
        "*Generated by `coach/dashboard.py` — do not edit manually.*\n"
        "*Run any notebook to refresh.*\n"
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_dashboard(state: dict) -> str:
    sections = [
        _section_header(state),
        _section_xp_bar(state),
        _section_streak(state),
        _section_skill_tree(state),
        _section_boss_battles(state),
        _section_power_rankings(state),
        _section_badges(state),
        _section_spaced_repetition(state),
        _section_session_history(state),
        _section_footer(),
    ]
    return "\n".join(sections)


def write_dashboard(state: dict) -> None:
    content = generate_dashboard(state)
    with open(DASHBOARD_PATH, "w") as f:
        f.write(content)
