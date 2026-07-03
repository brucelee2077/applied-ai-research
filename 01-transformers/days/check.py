#!/usr/bin/env python3
"""
check.py — Mark a build day as complete and see your progress.

Usage:
    python check.py <day_number>
    python check.py status          # show current progress

Examples:
    python check.py 1               # complete Day 1
    python check.py status          # see build diagram
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.expanduser("~/Desktop/applied-ai-research"))

from coach.core import (
    load_state, complete_build_day, get_build_progress,
    BUILD_DAYS, PHASE_NAMES, compute_level, get_level_name,
    BADGES, RARITY_EMOJI,
)
from coach.notebook_widgets import render_build_diagram_text


def print_status():
    state = load_state()
    progress = get_build_progress(state)
    xp = state["player"]["xp"]
    level_num, level_name, xp_into, xp_needed = compute_level(xp)
    streak = state["streak"]["current"]
    tokens = state["player"]["tokens"]

    print()
    print(f"  Level {level_num}: {level_name}  |  XP: {xp}  |  Streak: {streak} day(s)  |  Tokens: {tokens}")
    print(render_build_diagram_text(state))

    day = progress["current_day"]
    if day <= len(BUILD_DAYS):
        info = BUILD_DAYS[day]
        print(f"  Next up: Day {day} — {info['title']} (Phase {info['phase']}: {PHASE_NAMES[info['phase']]})")
    else:
        print("  All days complete! You built a transformer from scratch.")
    print()


def complete_day(day_number: int):
    state = load_state()
    result = complete_build_day(state, day_number)

    if "error" in result:
        print(f"\n  {result['error']}\n")
        return

    if result.get("already_done"):
        print(f"\n  Day {day_number} ({result['title']}) is already complete.")
        print_status()
        return

    # Celebration
    print()
    day_info = result
    title = day_info["title"]
    xp_awarded = day_info["xp_result"]["xp_awarded"]
    multiplier = day_info["xp_result"]["multiplier"]
    tokens = day_info["token_bonus"]
    is_milestone = day_info["is_milestone"]

    if is_milestone:
        print("  ===================================")
        print(f"  MILESTONE COMPLETE: Day {day_number} — {title}")
        print("  ===================================")
    else:
        print(f"  Day {day_number} complete: {title}")

    print(f"  +{xp_awarded} XP (x{multiplier:.2f} streak bonus)  |  +{tokens} tokens")

    # Component built
    if day_info["component_built"]:
        print(f"  Component unlocked: {day_info['component_built']}")

    # Level up
    if day_info["xp_result"].get("level_up"):
        new_name = day_info["xp_result"]["new_level_name"]
        print(f"  LEVEL UP! You are now: {new_name}")

    # Badges
    for badge_id in day_info["new_badges"]:
        info = BADGES.get(badge_id, {})
        emoji = RARITY_EMOJI.get(info.get("rarity", "common"), "")
        print(f"  Badge earned: {emoji} {info.get('name', badge_id)} — {info.get('desc', '')}")

    # Show updated diagram
    state = load_state()
    print(render_build_diagram_text(state))

    # Next day teaser
    next_day = day_info["next_day"]
    next_title = day_info["next_title"]
    if next_day:
        print(f"  Tomorrow: Day {next_day} — {next_title}")
    else:
        print("  You built a complete transformer from scratch!")
    print()


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python check.py <day_number>")
        print("       python check.py status\n")
        return

    arg = sys.argv[1]

    if arg == "status":
        print_status()
    else:
        try:
            day = int(arg)
        except ValueError:
            print(f"\n  '{arg}' is not a valid day number. Use 1-28 or 'status'.\n")
            return
        complete_day(day)


if __name__ == "__main__":
    main()
