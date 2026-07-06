"""
Shared derivation logic for the per-lesson kicker / sidebar nav-group-label /
finale heading, used by both `_shell_migrate.py` and `_fix_shell_metadata_sweep.py`.

Ground truth for the module display name is `sessions/index.html`'s `MODULES`
array (cap = short capability word, title = full module title). The week/day
identity always comes from the lesson's own `<title>` tag — never guessed,
never taken from the folder path — because that is the one field every lesson
already displays correctly to the reader elsewhere on its own page.
"""
import re
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(REPO, "sessions/index.html")

TITLE_PREFIX_RE = re.compile(
    r'^(Module\s+\d+(?:-\d+)?[a-z]?|Week\s+\d+[a-z]?)\s*(?:·\s*)?(Day\s+\d+|Weekend Lab)'
)
FOLDER_MODULE_RE = re.compile(r'/m(\w+?)-')
TRAILING_EMOJI_RE = re.compile(
    r'\s*[\U0001F000-\U0001FFFF☀-➿⬀-⯿️]+\s*$'
)


def load_modules():
    """Parse sessions/index.html's `MODULES` array into {num: {cap, title, phase}}."""
    text = open(INDEX_PATH, encoding="utf-8").read()
    m = re.search(r'var MODULES = \[(.*?)\n\];', text, re.S)
    body = m.group(1)
    entries = re.findall(
        r"\{n:'?(\w+)'?,\s*phase:'([^']*)',\s*cap:'([^']*)',\s*title:'([^']*)',\s*goal:",
        body,
    )
    return {n: {"phase": phase, "cap": cap, "title": title} for n, phase, cap, title in entries}


def normalize_folder_num(raw):
    """'01' -> '1', '05a' -> '5a', '10a' -> '10a', '09a' -> '9a'."""
    return re.sub(r'^0+(?=\d)', '', raw)


def module_num_from_path(path):
    m = FOLDER_MODULE_RE.search(path.replace(os.sep, "/"))
    if not m:
        return None
    return normalize_folder_num(m.group(1))


SHORT_NAME_SEPARATORS = [r'\s+\+\s+', r':\s+', r'\s+—\s+']


def clean_module_title(title):
    """Shorten a module's full MODULES[...].title to the concise name used in
    kickers/nav-labels: truncate at the earliest of ' + ', ': ', ' — ', then
    strip trailing decorative emoji. Matches the convention already applied
    by hand to the 2 pilot Week-N lessons (e.g. 'Transformer Math + Search 🎨'
    -> 'Transformer Math')."""
    positions = [m.start() for pat in SHORT_NAME_SEPARATORS for m in [re.search(pat, title)] if m]
    if positions:
        title = title[:min(positions)]
    return TRAILING_EMOJI_RE.sub('', title).strip()


def parse_title_prefix(title):
    """Return (unit_word, unit_num, day_part) from a lesson's own <title>, e.g.
    'Week 3 · Day 1 — Transformer Arithmetic...' -> ('Week', '3', 'Day 1')."""
    m = TITLE_PREFIX_RE.match(title)
    if not m:
        return None
    unit, day_part = m.groups()
    unit_word, unit_num = unit.split(None, 1)
    return unit_word, unit_num, day_part


def derive_metadata(title, folder_module_num, modules):
    """Compute the correct kicker / nav-group-label / finale-h3 for a lesson
    given its own <title> text and its folder-derived module number."""
    parsed = parse_title_prefix(title)
    if parsed is None:
        return None
    unit_word, unit_num, day_part = parsed
    mod = modules.get(folder_module_num)
    if mod is None:
        return None

    if unit_word == "Module":
        name = mod["cap"]
        pad_match = re.match(r'^(\d+)([a-z]?)$', unit_num)
        nav_num = f"{int(pad_match.group(1)):02d}{pad_match.group(2)}" if pad_match else unit_num
        kicker = f"Module {unit_num} · {name} · {day_part}"
        nav_group_label = f"Module {nav_num} · {name}"
    else:  # "Week"
        name = clean_module_title(mod["title"])
        kicker = f"{name} · Week {unit_num} · {day_part}"
        nav_group_label = f"{name} · Week {unit_num}"

    finale_h3 = f"{unit_word} {unit_num} · {day_part} complete! 🏆"
    return {"kicker": kicker, "nav_group_label": nav_group_label, "finale_h3": finale_h3}
