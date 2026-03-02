"""
coach/notebook_widgets.py — IPython/Jupyter widget renderers.

Usage (add to top of any notebook):
    import sys, os
    sys.path.insert(0, os.path.expanduser('~/Desktop/applied-ai-research'))
    from coach.notebook_widgets import render_session_start
    _SESSION = render_session_start(module_id="02-visual-search",
                                     notebook_name="01_visual_search_system_design.ipynb")

Usage (add to bottom of any notebook):
    from coach.notebook_widgets import render_session_end
    render_session_end(_SESSION)
"""

from __future__ import annotations

import datetime
import importlib
import math
import random
from typing import Any

# ── IPython display helpers ────────────────────────────────────────────────────
try:
    from IPython.display import display, HTML, clear_output
    import ipywidgets as widgets
    _JUPYTER = True
except ImportError:
    _JUPYTER = False

from coach.core import (
    start_session, end_session, load_state, save_state,
    award_xp, get_cohort_percentile, record_quiz_answer,
    get_due_concepts, spend_tokens, record_boss_result,
    BADGES, RARITY_EMOJI, COST_HINT_REVEAL, COST_SKIP_PREREQ, COST_STREAK_SHIELD,
    TOKENS_QUIZ_PASS, XP_QUIZ_PASS, XP_QUIZ_RETRY,
    MODULE_STUDY_NOTEBOOKS, ALL_MODULES,
)

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "xp":      "#4CAF50",
    "streak":  "#FF6B35",
    "warning": "#FF9800",
    "danger":  "#F44336",
    "boss":    "#9C27B0",
    "locked":  "#9E9E9E",
    "gold":    "#FFD700",
    "blue":    "#2196F3",
    "bg":      "#1a1a2e",
    "card":    "#16213e",
    "text":    "#e0e0e0",
}

RARITY_COLOR = {
    "common":    "#9E9E9E",
    "uncommon":  "#4CAF50",
    "rare":      "#2196F3",
    "epic":      "#9C27B0",
    "legendary": "#FFD700",
}

LEVEL_COLORS = {
    1: "#78909C", 2: "#78909C", 3: "#66BB6A",
    4: "#66BB6A", 5: "#FFA726", 6: "#FFA726",
    7: "#EF5350", 8: "#EF5350", 9: "#AB47BC", 10: "#FFD700",
}


def _html(content: str) -> None:
    if _JUPYTER:
        display(HTML(content))
    else:
        print(content)


def _card(content: str, border_color: str = "#2196F3", bg: str = "#16213e") -> str:
    return f"""
<div style="
    background:{bg};
    border-left: 4px solid {border_color};
    border-radius: 8px;
    padding: 14px 18px;
    margin: 8px 0;
    font-family: 'Courier New', monospace;
    color: #e0e0e0;
    font-size: 13px;
    line-height: 1.6;
">{content}</div>"""


def _progress_bar(value: int, max_val: int, color: str = "#4CAF50",
                  height: int = 16, label: str = "") -> str:
    pct = min(100, round(value / max(max_val, 1) * 100))
    return f"""
<div style="margin:4px 0">
  <div style="font-size:11px; color:#aaa; margin-bottom:2px">{label}</div>
  <div style="background:#333; border-radius:8px; height:{height}px; width:100%; position:relative;">
    <div style="background:{color}; width:{pct}%; height:{height}px; border-radius:8px; transition:width 0.5s;"></div>
    <span style="position:absolute; right:6px; top:0; line-height:{height}px; font-size:10px; color:#fff;">{value:,}/{max_val:,}</span>
  </div>
</div>"""


def _ascii_bar(filled: int, total: int, width: int = 20) -> str:
    n = round(width * filled / max(total, 1))
    return "#" * n + "·" * (width - n)


# ── Session Start ──────────────────────────────────────────────────────────────

def render_session_start(module_id: str, notebook_name: str) -> dict:
    """
    Call at the top of any notebook.
    Displays a rich reward widget and returns context dict for render_session_end.
    """
    ctx = start_session(module_id, notebook_name)
    state = ctx["state"]
    player = state["player"]
    streak = state["streak"]

    xp = player["xp"]
    level_num = ctx["level_num"]
    level_name = ctx["level_name"]
    xp_into = ctx["xp_into_level"]
    xp_needed = ctx["xp_needed_for_next"]
    current_streak = streak["current"]
    fire = streak.get("fire_active", False)
    tokens = player["tokens"]
    percentile = ctx["percentile"]
    top_pct = 100 - percentile

    # ── Streak line ──
    fire_tag = ""
    if current_streak >= 30:
        fire_tag = '<span style="color:#FFD700"> 🔥🔥🔥 LEGENDARY 3.0x</span>'
    elif current_streak >= 14:
        fire_tag = '<span style="color:#FF6B35"> 🔥🔥 ON FIRE 2.0x</span>'
    elif current_streak >= 7:
        fire_tag = '<span style="color:#FF6B35"> 🔥 ON FIRE 1.5x</span>'
    elif current_streak >= 3:
        fire_tag = '<span style="color:#FFA726"> 🔥 1.25x</span>'

    streak_html = (
        f'<b style="color:#FF6B35; font-size:18px">{current_streak} DAY STREAK</b>'
        f'{fire_tag}'
        f'<span style="color:#888; font-size:11px; margin-left:10px">Longest: {streak["longest"]}d</span>'
    )

    # ── XP bar ──
    level_color = LEVEL_COLORS.get(level_num, "#2196F3")
    xp_bar = _progress_bar(xp_into, xp_needed, color=level_color, height=18,
                            label=f"Level {level_num} — {level_name}  →  Level {level_num+1}")

    # ── Warning ──
    warning_html = ""
    if ctx["warning"]:
        warning_html = f"""
<div style="background:#b71c1c; border-radius:6px; padding:10px 14px; margin:8px 0;
     font-weight:bold; color:#fff; font-size:13px;">
  ⚠️ {ctx['warning']}
</div>"""

    # ── Cohort status ──
    cohort_html = (
        f'<div style="color:#aaa; font-size:12px; margin-top:4px">'
        f'📊 You are in the <b style="color:#FFD700">top {top_pct}%</b> of ML design studiers '
        f'| <b style="color:#4CAF50">{tokens}</b> tokens</div>'
    )

    # ── Due reviews ──
    due = ctx["due_reviews"]
    review_html = ""
    if due:
        items = "".join(
            f'<li style="color:#FF9800">{d["concept_id"]} '
            f'<span style="color:#666">({d["days_overdue"]}d overdue)</span></li>'
            for d in due[:5]
        )
        review_html = f"""
<div style="margin-top:8px; font-size:12px;">
  <b style="color:#FF9800">📚 Spaced Repetition Due ({len(due)} concepts):</b>
  <ul style="margin:4px 0; padding-left:20px">{items}</ul>
  <code style="font-size:11px; color:#888">render_quiz_widget(due_only=True)</code>
</div>"""

    # ── Boss status ──
    boss_html = ""
    mod = state["modules"].get(module_id, {})
    if mod.get("boss_unlocked") and not mod.get("boss_passed"):
        hours = ctx.get("boss_hours_remaining")
        if hours is not None:
            color = "#F44336" if hours < 12 else "#9C27B0"
            boss_html = f"""
<div style="background:{color}22; border:1px solid {color}; border-radius:6px;
     padding:8px 12px; margin:8px 0; font-size:12px; color:{color}; font-weight:bold;">
  ⚔️ BOSS BATTLE UNLOCKED — {hours:.0f}h remaining
  <span style="color:#aaa; font-weight:normal; margin-left:8px">
    render_boss_battle_widget("{module_id}")
  </span>
</div>"""

    # ── Welcome XP ──
    welcome_html = ""
    if ctx.get("welcome_xp"):
        wx = ctx["welcome_xp"]
        welcome_html = (
            f'<div style="color:#4CAF50; font-size:12px; margin-top:4px">'
            f'✨ +{wx["xp_awarded"]} XP — {wx["reason"].replace("_", " ")}'
            f'</div>'
        )

    # ── Module progress ──
    from coach.core import MODULE_STUDY_NOTEBOOKS
    study_nbs = MODULE_STUDY_NOTEBOOKS.get(module_id, [])
    completed = mod.get("notebooks_completed", [])
    mod_bar = _ascii_bar(len(completed), len(study_nbs))
    is_boss_nb = "interviewer_perspective" in notebook_name
    nb_label = "BOSS BATTLE" if is_boss_nb else notebook_name.replace(".ipynb", "").replace("_", " ").title()

    html = f"""
<div style="background:#0f0f1a; border-radius:10px; padding:16px 20px; margin:10px 0;
     font-family:'Courier New',monospace; border:1px solid #333;">
  <div style="font-size:11px; color:#666; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px">
    ML Interview Coach  ·  {module_id}  ·  {nb_label}
  </div>
  <div style="margin-bottom:10px">{streak_html}</div>
  {xp_bar}
  {cohort_html}
  {warning_html}
  {welcome_html}
  {boss_html}
  {review_html}
  <div style="margin-top:8px; font-size:11px; color:#555;">
    Module progress: [{mod_bar}] {len(completed)}/{len(study_nbs)} notebooks
  </div>
</div>"""

    _html(html)
    return ctx


# ── Session End ────────────────────────────────────────────────────────────────

def render_session_end(context: dict) -> None:
    """
    Call at the bottom of any notebook.
    Awards XP, updates state, shows celebration widget, regenerates DASHBOARD.md.
    """
    summary = end_session(
        context["module_id"],
        context["notebook_name"],
        context,
    )

    state = summary["state"]
    xp_result = summary.get("xp_result")
    new_badges = summary.get("new_badges", [])
    level_up = summary.get("level_up", False)
    minutes = summary.get("minutes_spent", 0)
    boss_unlocked_now = summary.get("boss_unlocked_now", False)

    # ── XP gain line ──
    xp_html = ""
    if xp_result:
        mult_str = f" × {xp_result['multiplier']}x streak" if xp_result["multiplier"] > 1.0 else ""
        xp_html = f"""
<div style="font-size:16px; color:#4CAF50; font-weight:bold; margin:8px 0;">
  +{xp_result['base_xp']} XP{mult_str} = <span style="font-size:20px">+{xp_result['xp_awarded']} XP</span>
  <span style="font-size:12px; color:#888; margin-left:8px">+{xp_result['tokens_awarded']} tokens</span>
</div>"""

    # ── Level up ──
    levelup_html = ""
    if level_up:
        lname = summary["new_level_name"]
        lnum = summary["new_level"]
        color = LEVEL_COLORS.get(lnum, "#FFD700")
        levelup_html = f"""
<div style="background:{color}22; border:2px solid {color}; border-radius:10px;
     padding:14px 20px; margin:10px 0; text-align:center;">
  <div style="font-size:24px; font-weight:bold; color:{color}">
    ⬆️ LEVEL UP!
  </div>
  <div style="font-size:18px; color:{color}; margin-top:4px">
    Level {lnum} — {lname}
  </div>
</div>"""

    # ── New badges ──
    badges_html = ""
    if new_badges:
        items = ""
        for bid in new_badges:
            meta = BADGES.get(bid, {})
            rarity = meta.get("rarity", "common")
            color = RARITY_COLOR.get(rarity, "#888")
            emoji = RARITY_EMOJI.get(rarity, "")
            items += (
                f'<div style="color:{color}; margin:4px 0; font-size:13px;">'
                f'{emoji} <b>{meta.get("name", bid)}</b> ({rarity}) — {meta.get("desc","")}'
                f'</div>'
            )
        badges_html = f"""
<div style="margin:8px 0;">
  <div style="color:#FFD700; font-size:13px; font-weight:bold; margin-bottom:4px">🏅 New Badge(s) Unlocked!</div>
  {items}
</div>"""

    # ── Boss unlocked ──
    boss_html = ""
    if boss_unlocked_now:
        boss_html = f"""
<div style="background:#9C27B022; border:1px solid #9C27B0; border-radius:8px;
     padding:10px 14px; margin:8px 0; color:#CE93D8; font-size:13px; font-weight:bold;">
  ⚔️ BOSS BATTLE UNLOCKED for {context['module_id']}!
  You have 48 hours. Don't let it expire.
  <br><code style="font-size:11px">render_boss_battle_widget("{context['module_id']}")</code>
</div>"""

    # ── XP bar state ──
    xp = state["player"]["xp"]
    level_num = summary["new_level"]
    level_name = summary["new_level_name"]
    xp_into = summary["xp_into_level"]
    xp_needed = summary["xp_needed_for_next"]
    level_color = LEVEL_COLORS.get(level_num, "#2196F3")
    xp_bar = _progress_bar(xp_into, xp_needed, color=level_color, height=16,
                            label=f"Level {level_num} — {level_name}")

    # ── Next up ──
    due = get_due_concepts(state)
    next_due = f"{len(due)} concept(s) due for review" if due else "No reviews due"

    html = f"""
<div style="background:#0f1a0f; border-radius:10px; padding:16px 20px; margin:10px 0;
     font-family:'Courier New',monospace; border:1px solid #1b5e20;">
  <div style="font-size:11px; color:#666; margin-bottom:8px; text-transform:uppercase">
    Session Complete  ·  {minutes} min studied
  </div>
  {xp_html}
  {xp_bar}
  {levelup_html}
  {badges_html}
  {boss_html}
  <div style="margin-top:10px; font-size:11px; color:#555;">
    📋 DASHBOARD.md updated  ·  Next: {next_due}
  </div>
</div>"""

    _html(html)


# ── Quiz Widget ────────────────────────────────────────────────────────────────

def _load_quiz_questions(module_id: str | None = None,
                          concept_ids: list[str] | None = None,
                          due_only: bool = False) -> list[dict]:
    """Load quiz questions from coach/quizzes/ files."""
    from coach.core import load_state
    state = load_state()

    all_questions: list[dict] = []
    module_map = {
        "01-ml-design-prep":            "ml_design_prep",
        "02-visual-search":             "visual_search",
        "03-google-street-view":        "street_view",
        "04-youtube-video-search":      "video_search",
        "05-harmful-content-detection": "harmful_content",
        "06-video-recommendation":      "video_rec",
        "07-event-recommendation":      "event_rec",
        "08-ad-click-prediction":       "ad_click",
        "09-similar-listing":           "similar_listing",
        "10-personalized-news-feed":    "news_feed",
        "11-people-you-know":           "pymk",
        "11-people-you-may-know":       "pymk",
        "rnn":                          "rnn_fundamentals",
        "transformers":                 "transformers",
    }

    targets = [module_id] if module_id else list(module_map.keys())
    for mid in targets:
        mod_name = module_map.get(mid)
        if not mod_name:
            continue
        try:
            mod = importlib.import_module(f"coach.quizzes.{mod_name}")
            all_questions.extend(getattr(mod, "QUESTIONS", []))
        except (ImportError, ModuleNotFoundError):
            pass

    if concept_ids:
        all_questions = [q for q in all_questions if q["concept_id"] in concept_ids]

    if due_only:
        due_ids = {d["concept_id"] for d in get_due_concepts(state)}
        all_questions = [q for q in all_questions if q["concept_id"] in due_ids]
        if not all_questions:
            # Fall back to all questions for the module
            pass

    random.shuffle(all_questions)
    return all_questions


def render_quiz_widget(module_id: str | None = None,
                       concept_ids: list[str] | None = None,
                       due_only: bool = False,
                       max_questions: int = 10) -> None:
    """
    Display an interactive quiz widget.
    module_id: limit to one module, or None for all.
    due_only: only show spaced-repetition due concepts.
    """
    if not _JUPYTER:
        print("Quiz widget requires a Jupyter environment.")
        return

    questions = _load_quiz_questions(module_id, concept_ids, due_only)
    if not questions:
        _html(_card("<b>No questions available.</b> Complete some notebooks first, "
                    "or run <code>render_quiz_widget()</code> without due_only.",
                    border_color="#FF9800"))
        return

    questions = questions[:max_questions]
    state_container = {"q_index": 0, "score": 0, "answered": 0, "hint_used": False}

    # ── Widget components ──
    title = widgets.HTML(value="")
    question_box = widgets.HTML(value="")
    choices_box = widgets.RadioButtons(options=[], layout=widgets.Layout(width="100%"))
    hint_btn = widgets.Button(description="Reveal Hint (5 tokens)", button_style="warning",
                               layout=widgets.Layout(width="200px"))
    submit_btn = widgets.Button(description="Submit Answer", button_style="primary",
                                 layout=widgets.Layout(width="160px"))
    feedback_box = widgets.HTML(value="")
    quality_slider = widgets.IntSlider(
        value=4, min=0, max=5,
        description="How well did you know this? (0=blank, 5=perfect):",
        style={"description_width": "320px"},
        layout=widgets.Layout(width="520px", display="none"),
    )
    next_btn = widgets.Button(description="Next Question →", button_style="success",
                               layout=widgets.Layout(width="160px", display="none"))
    progress_bar = widgets.IntProgress(value=0, min=0, max=len(questions),
                                        description="Progress:",
                                        bar_style="info",
                                        layout=widgets.Layout(width="400px"))

    def load_question(idx: int) -> None:
        if idx >= len(questions):
            _show_summary()
            return
        q = questions[idx]
        state_container["hint_used"] = False
        title.value = (
            f'<div style="font-family:Courier New; font-size:11px; color:#888; margin-bottom:4px">'
            f'Question {idx+1}/{len(questions)} · {q.get("module","—")} · '
            f'Difficulty {"★"*q.get("difficulty",3)}'
            f'</div>'
        )
        question_box.value = (
            f'<div style="background:#16213e; border-left:4px solid #2196F3; '
            f'border-radius:6px; padding:12px 16px; font-family:Courier New; '
            f'color:#e0e0e0; font-size:14px; margin:6px 0;">'
            f'{q["question"]}</div>'
        )
        choices_box.options = q["choices"]
        choices_box.value = q["choices"][0]
        feedback_box.value = ""
        quality_slider.layout.display = "none"
        next_btn.layout.display = "none"
        submit_btn.layout.display = ""
        hint_btn.layout.display = ""

    def _show_summary() -> None:
        total = len(questions)
        pct = round(state_container["score"] / max(total, 1) * 100)
        color = "#4CAF50" if pct >= 70 else "#FF9800" if pct >= 50 else "#F44336"
        summary_html = f"""
<div style="background:#0f1a0f; border:2px solid {color}; border-radius:10px;
     padding:20px; font-family:Courier New; text-align:center; margin:10px 0;">
  <div style="font-size:24px; color:{color}; font-weight:bold;">
    Quiz Complete! {state_container['score']}/{total} ({pct}%)
  </div>
  {'<div style="color:#FFD700; margin-top:8px">PERFECT SCORE! +100 XP bonus incoming...</div>' if pct==100 else ''}
  <div style="color:#aaa; font-size:12px; margin-top:8px">
    Keep going — spaced repetition will resurface the ones you missed.
  </div>
</div>"""
        question_box.value = summary_html
        choices_box.layout.display = "none"
        hint_btn.layout.display = "none"
        submit_btn.layout.display = "none"
        feedback_box.value = ""
        next_btn.layout.display = "none"
        quality_slider.layout.display = "none"

        # Perfect score bonus
        if pct == 100:
            state = load_state()
            state["quiz_history"]["perfect_quiz_count"] = (
                state["quiz_history"].get("perfect_quiz_count", 0) + 1
            )
            award_xp(state, 100, "perfect_quiz_bonus")
            save_state(state)

    def on_hint(_b: Any) -> None:
        if state_container["hint_used"]:
            return
        state = load_state()
        try:
            spend_tokens(state, COST_HINT_REVEAL, "hint_reveal")
            save_state(state)
            state_container["hint_used"] = True
            q = questions[state_container["q_index"]]
            feedback_box.value = (
                f'<div style="color:#FF9800; font-size:12px; font-style:italic; '
                f'padding:6px 10px; background:#FF980011; border-radius:4px;">'
                f'💡 Hint: {q.get("hint","No hint available.")}</div>'
            )
        except ValueError as e:
            feedback_box.value = f'<div style="color:#F44336">❌ {e}</div>'

    def on_submit(_b: Any) -> None:
        idx = state_container["q_index"]
        q = questions[idx]
        chosen = choices_box.value
        correct_letter = q["correct"]
        correct_choice = next((c for c in q["choices"] if c.startswith(correct_letter + ".")), None)

        is_correct = chosen.startswith(correct_letter + ".")
        if is_correct:
            state_container["score"] += 1
            color = "#4CAF50"
            icon = "✓ CORRECT"
        else:
            color = "#F44336"
            icon = f"✗ WRONG — Correct: {correct_letter}"

        state_container["answered"] += 1
        feedback_box.value = f"""
<div style="border-left:4px solid {color}; padding:8px 12px; background:{color}11;
     border-radius:4px; font-family:Courier New; font-size:13px; margin:6px 0;">
  <b style="color:{color}">{icon}</b><br>
  <span style="color:#ccc">{q.get('explanation','')}</span>
</div>"""
        submit_btn.layout.display = "none"
        hint_btn.layout.display = "none"
        quality_slider.layout.display = ""
        next_btn.layout.display = ""
        progress_bar.value = state_container["answered"]

    def on_next(_b: Any) -> None:
        # Record SM-2 result
        idx = state_container["q_index"]
        q = questions[idx]
        quality = quality_slider.value
        state = load_state()
        record_quiz_answer(state, q["concept_id"], quality)
        save_state(state)

        state_container["q_index"] += 1
        load_question(state_container["q_index"])

    hint_btn.on_click(on_hint)
    submit_btn.on_click(on_submit)
    next_btn.on_click(on_next)

    header_html = (
        f'<div style="background:#1a1a2e; border-radius:8px; padding:10px 16px; '
        f'font-family:Courier New; color:#aaa; font-size:12px; margin-bottom:6px;">'
        f'📖 Quiz · {len(questions)} questions'
        f'{"  ·  Due-only mode" if due_only else ""}'
        f'{"  ·  Module: " + module_id if module_id else ""}'
        f'</div>'
    )

    load_question(0)
    display(
        widgets.HTML(value=header_html),
        progress_bar,
        title,
        question_box,
        choices_box,
        widgets.HBox([hint_btn, submit_btn, next_btn]),
        quality_slider,
        feedback_box,
    )


# ── Boss Battle Widget ─────────────────────────────────────────────────────────

def render_boss_battle_widget(module_id: str) -> None:
    """
    Interactive boss battle interface with countdown timer and rubric self-assessment.
    """
    if not _JUPYTER:
        print("Boss battle widget requires a Jupyter environment.")
        return

    state = load_state()
    mod = state["modules"].get(module_id, {})

    if not mod.get("boss_unlocked"):
        _html(_card(
            f"<b>⚔️ Boss not yet unlocked</b> for <code>{module_id}</code>.<br>"
            "Complete all study notebooks to unlock the boss battle.",
            border_color="#9E9E9E",
        ))
        return

    if mod.get("boss_passed"):
        score = mod.get("boss_score", 0) or 0
        time_m = mod.get("boss_time_minutes", 0) or 0
        _html(_card(
            f"<b style='color:#4CAF50'>✓ Boss already defeated!</b><br>"
            f"Score: {score*100:.0f}%  ·  Time: {time_m:.0f} min",
            border_color="#4CAF50",
        ))
        return

    # Rubric criteria (generic — module-specific criteria could be loaded from the notebook)
    criteria = [
        "Problem clarification & scope definition",
        "Data & feature design",
        "ML model / architecture choice with justification",
        "Training pipeline (loss, data splits, offline eval)",
        "Serving & latency design",
        "Metrics (offline vs online, A/B testing)",
        "Edge cases & failure modes",
    ]
    rating_options = ["Not addressed", "No Hire", "Weak Hire", "Hire", "Strong Hire"]
    SCORE_MAP = {"Not addressed": 0, "No Hire": 0.25, "Weak Hire": 0.5, "Hire": 0.75, "Strong Hire": 1.0}

    # Timer state
    timer_state = {"started": False, "start_time": None, "elapsed": 0}
    LIMIT = 45 * 60  # seconds

    timer_label = widgets.HTML(value=_timer_html(0, LIMIT))
    start_btn = widgets.Button(description="⚔️ Start Boss Battle", button_style="danger",
                                layout=widgets.Layout(width="200px"))
    submit_btn = widgets.Button(description="Submit Result", button_style="primary",
                                 layout=widgets.Layout(width="160px", display="none"))

    # Rubric rows
    rubric_dropdowns = []
    rubric_rows = []
    for i, criterion in enumerate(criteria):
        dd = widgets.Dropdown(options=rating_options, value="Not addressed",
                              layout=widgets.Layout(width="200px"))
        rubric_dropdowns.append(dd)
        row = widgets.HBox([
            widgets.HTML(value=f'<div style="font-family:Courier New; font-size:12px; '
                                f'color:#ccc; width:380px; padding:4px 0">'
                                f'{i+1}. {criterion}</div>'),
            dd,
        ])
        rubric_rows.append(row)

    rubric_box = widgets.VBox(rubric_rows, layout=widgets.Layout(display="none"))
    result_box = widgets.HTML(value="")

    # Import for timer updates
    import threading

    def update_timer() -> None:
        import time
        while timer_state["started"]:
            elapsed = int((datetime.datetime.now() - timer_state["start_time"]).total_seconds())
            timer_state["elapsed"] = elapsed
            timer_label.value = _timer_html(elapsed, LIMIT)
            if elapsed >= LIMIT:
                timer_state["started"] = False
                timer_label.value = _timer_html(LIMIT, LIMIT)
                break
            time.sleep(1)

    def on_start(_b: Any) -> None:
        timer_state["started"] = True
        timer_state["start_time"] = datetime.datetime.now()
        rubric_box.layout.display = ""
        submit_btn.layout.display = ""
        start_btn.layout.display = "none"
        t = threading.Thread(target=update_timer, daemon=True)
        t.start()

    def on_submit(_b: Any) -> None:
        timer_state["started"] = False
        elapsed_min = timer_state["elapsed"] / 60
        scores = [SCORE_MAP[dd.value] for dd in rubric_dropdowns]
        score_pct = sum(scores) / len(scores)

        state = load_state()
        result = record_boss_result(state, module_id, score_pct, elapsed_min)
        save_state(state)

        from coach.dashboard import write_dashboard
        write_dashboard(state)

        grade = (
            "STRONG HIRE" if score_pct >= 0.85 else
            "HIRE"        if score_pct >= 0.65 else
            "WEAK HIRE"   if score_pct >= 0.40 else
            "NO HIRE"
        )
        color = (
            "#4CAF50" if score_pct >= 0.65 else
            "#FF9800" if score_pct >= 0.40 else "#F44336"
        )
        xp_r = result["xp_result"]
        badges_html = "".join(
            f'<div style="color:{RARITY_COLOR.get(BADGES.get(b,{}).get("rarity","common"),"#888")}; font-size:12px;">'
            f'🏅 {BADGES.get(b,{}).get("name", b)}</div>'
            for b in result.get("new_badges", [])
        )
        unlocked_html = "".join(
            f'<div style="color:#9C27B0; font-size:12px;">🔓 Unlocked: {m}</div>'
            for m in result.get("newly_unlocked_modules", [])
        )

        result_box.value = f"""
<div style="background:{color}11; border:2px solid {color}; border-radius:10px;
     padding:16px 20px; font-family:Courier New; margin:10px 0;">
  <div style="font-size:22px; color:{color}; font-weight:bold;">{grade}</div>
  <div style="font-size:16px; color:#ccc; margin-top:6px">
    Score: {score_pct*100:.0f}%  ·  Time: {elapsed_min:.1f} min
  </div>
  <div style="color:#4CAF50; font-size:14px; margin-top:8px">
    +{xp_r['xp_awarded']} XP (×{xp_r['multiplier']}x streak)  ·  +20 tokens
  </div>
  {badges_html}
  {unlocked_html}
  <div style="color:#555; font-size:11px; margin-top:8px">DASHBOARD.md updated</div>
</div>"""
        submit_btn.layout.display = "none"

    start_btn.on_click(on_start)
    submit_btn.on_click(on_submit)

    hours_left = None
    exp_str = mod.get("boss_expires_at")
    if exp_str:
        delta = datetime.datetime.fromisoformat(exp_str) - datetime.datetime.now()
        hours_left = max(0, delta.total_seconds() / 3600)

    expiry_html = ""
    if hours_left is not None:
        color = "#F44336" if hours_left < 12 else "#FF9800"
        expiry_html = (
            f'<div style="color:{color}; font-size:12px; margin-bottom:8px; font-weight:bold;">'
            f'⏳ Boss expires in {hours_left:.0f}h — don\'t let it slip!</div>'
        )

    header = widgets.HTML(value=f"""
<div style="background:#1a0a2e; border:2px solid #9C27B0; border-radius:10px;
     padding:14px 20px; font-family:Courier New; margin:10px 0;">
  <div style="font-size:20px; color:#CE93D8; font-weight:bold; margin-bottom:6px">
    ⚔️ BOSS BATTLE — {module_id}
  </div>
  {expiry_html}
  <div style="color:#aaa; font-size:12px;">
    45-minute mock interview · Self-assess against rubric criteria · 200 XP reward
  </div>
</div>""")

    display(
        header,
        timer_label,
        start_btn,
        rubric_box,
        submit_btn,
        result_box,
    )


def _timer_html(elapsed: int, limit: int) -> str:
    remaining = max(0, limit - elapsed)
    m, s = divmod(remaining, 60)
    pct = min(100, round(elapsed / limit * 100))
    color = "#F44336" if remaining < 300 else "#FF9800" if remaining < 600 else "#4CAF50"
    return (
        f'<div style="font-family:Courier New; font-size:28px; color:{color}; '
        f'font-weight:bold; margin:8px 0;">'
        f'⏱ {m:02d}:{s:02d}'
        f'<span style="font-size:12px; color:#666; margin-left:12px">'
        f'{pct}% elapsed</span></div>'
    )


# ── Token Shop ─────────────────────────────────────────────────────────────────

def render_token_shop() -> None:
    """Interactive token shop widget."""
    if not _JUPYTER:
        print("Token shop requires a Jupyter environment.")
        return

    state = load_state()
    balance_label = widgets.HTML(value=_balance_html(state["player"]["tokens"]))
    result_box = widgets.HTML(value="")

    items = [
        ("streak_shield", "Streak Shield",
         f"Protect your streak for 1 missed day", COST_STREAK_SHIELD, "🛡️"),
        ("hint_reveal",   "Hint Reveal",
         f"Reveal a quiz hint without penalty",   COST_HINT_REVEAL,   "💡"),
        ("skip_prereq",   "Skip Prerequisite",
         f"Unlock next module without boss",       COST_SKIP_PREREQ,   "⚡"),
    ]

    btns = []
    for item_id, name, desc, cost, emoji in items:
        btn = widgets.Button(
            description=f"{emoji} {name} ({cost} tokens)",
            button_style="warning",
            layout=widgets.Layout(width="280px", margin="4px"),
        )
        btn._item_id = item_id
        btn._cost = cost
        btn._name = name
        btns.append(btn)

        def _make_handler(iid: str, icost: int, iname: str):
            def handler(_b: Any) -> None:
                s = load_state()
                try:
                    spend_tokens(s, icost, iid)
                    if iid == "streak_shield":
                        s["streak"]["shields_remaining"] += 1
                    save_state(s)
                    balance_label.value = _balance_html(s["player"]["tokens"])
                    result_box.value = (
                        f'<div style="color:#4CAF50; font-size:13px; '
                        f'font-family:Courier New; margin:6px 0;">'
                        f'✓ Purchased: {iname}</div>'
                    )
                except ValueError as e:
                    result_box.value = f'<div style="color:#F44336; font-size:13px;">{e}</div>'
            return handler

        btn.on_click(_make_handler(item_id, cost, name))

    shop_html = """
<div style="background:#1a1208; border:2px solid #FFD700; border-radius:10px;
     padding:14px 20px; font-family:Courier New; margin:10px 0;">
  <div style="font-size:18px; color:#FFD700; font-weight:bold; margin-bottom:8px">
    🏪 Token Shop
  </div>
  <div style="color:#aaa; font-size:12px; margin-bottom:10px">
    Earn tokens by studying and passing quizzes. Spend them wisely.
  </div>
</div>"""

    display(
        widgets.HTML(value=shop_html),
        balance_label,
        widgets.VBox(btns),
        result_box,
    )


def _balance_html(tokens: int) -> str:
    return (
        f'<div style="font-family:Courier New; font-size:14px; color:#FFD700; margin:6px 0;">'
        f'💰 Token Balance: <b>{tokens}</b></div>'
    )
