"""NFL Confidence Pool — Streamlit dashboard.

Tactical intelligence aesthetic. 2-column pick grid. All games on one screen.
Run: uv run streamlit run ui/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from ui.styles import STYLES
from ui.utils import (
    _current_season, _api_get, _load_submission, _load_week, _load_odds,
    _load_rerankings, _do_revert, _do_refresh, _model_point_map,
    _review_table_html, swap_dialog, lock_dialog, render_game_card
)

st.set_page_config(
    page_title="NFL Pool",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(STYLES, unsafe_allow_html=True)


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<p style="font-family:\'Syne\',sans-serif;font-size:1.1rem;'
        'font-weight:800;color:#E2E8F0;letter-spacing:0.06em;'
        'text-transform:uppercase;margin-bottom:0.5rem">NFL Pool</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    sidebar_season = st.number_input("Season", min_value=2018,
                                      max_value=_current_season() + 1,
                                      value=_current_season(), step=1, key="sb_season")
    sidebar_week = st.number_input("Week", min_value=1, max_value=22, value=1, step=1, key="sb_week")

    if st.button("↻  Refresh Week Data", type="primary", use_container_width=True, key="sb_refresh"):
        with st.spinner("Fetching games, odds, injuries, weather…"):
            _do_refresh(int(sidebar_season), int(sidebar_week))

    st.divider()
    api_ok = _api_get("/health") is not None
    dot = "●"
    st.caption(
        f"{dot} API connected" if api_ok else f"{dot} offline (direct mode)"
    )

    st.divider()
    if st.button("Admin", type="secondary", use_container_width=True, key="sb_admin"):
        st.session_state["admin_mode"] = True
        st.rerun()

# ── admin mode check ─────────────────────────────────────────────────────────

if st.session_state.get("admin_mode", False):
    from ui.admin import render_admin
    render_admin()
    st.stop()


# ── data & header ────────────────────────────────────────────────────────────

# Use sidebar values as defaults for main inputs if not interacting with main inputs
season = int(st.session_state.get("main_season", sidebar_season))
week = int(st.session_state.get("main_week", sidebar_week))

data = _load_week(season, week)

is_locked = False
if data and data.get("assignments"):
    submission = _load_submission(season, week)
    is_locked = bool(submission) or (season, week) in st.session_state.get("locked", set())

status_class = "locked" if is_locked else "live"
status_text = "● LOCKED IN" if is_locked else "● LIVE"

# Header Row 1: Controls
c1, c2, c3, c4 = st.columns([1.0, 0.8, 6.2, 2.0], vertical_alignment="bottom")
with c1:
    season = st.number_input("Season", min_value=2018,
                             max_value=_current_season() + 1,
                             value=season, step=1, key="main_season")
with c2:
    week = st.number_input("Week", min_value=1, max_value=22, value=week,
                           step=1, key="main_week")
with c3:
    st.empty() # Spacer
with c4:
    if st.button("↻ REFRESH WEEK DATA", type="primary", use_container_width=True,
                 key="main_refresh"):
        with st.spinner("Fetching games, odds, injuries, weather…"):
            _do_refresh(int(season), int(week))

# Header Row 2: Title & Lock
t1, t2 = st.columns([8.0, 2.0], vertical_alignment="bottom")
with t1:
    st.markdown(
        f'<div class="topbar-left">'
        f'<div class="topbar-title">WEEK {week} · {season}</div>'
        f'<div class="topbar-status {status_class}">{status_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with t2:
    if not is_locked:
        if st.button("LOCK IN WEEK ✓", type="primary",
                     use_container_width=True, key="top_lock"):
            lock_dialog(season, week)
    else:
        st.markdown('<div class="locked-badge">✓ PICKS LOCKED IN</div>',
                    unsafe_allow_html=True)

if not data or not data.get("assignments"):
    st.info("No picks yet. Click **↻ Refresh Week Data** to fetch games.")
    st.stop()

season, week = int(season), int(week)
assignments = sorted(
    data["assignments"], key=lambda a: a["confidence_points"], reverse=True
)
games_by_id = {g["espn_id"]: g for g in data.get("games", [])}
odds = _load_odds(season, week)
rerankings = _load_rerankings(season, week)
submission = _load_submission(season, week)

swap_game_id = st.query_params.get("swap")
if isinstance(swap_game_id, list):
    swap_game_id = swap_game_id[0] if swap_game_id else None
if swap_game_id and any(a["game_id"] == swap_game_id for a in assignments):
    swap_dialog(swap_game_id, assignments, games_by_id, season, week)

undo_game_id = st.query_params.get("undo")
if isinstance(undo_game_id, list):
    undo_game_id = undo_game_id[0] if undo_game_id else None
if undo_game_id and any(a["game_id"] == undo_game_id for a in assignments):
    if _do_revert(season, week, undo_game_id):
        del st.query_params["undo"]
        st.rerun()

# ── summary metrics ───────────────────────────────────────────────────────────

n_lock = sum(1 for a in assignments if a["win_probability"] >= 0.70)
n_lean = sum(1 for a in assignments if 0.62 <= a["win_probability"] < 0.70)
n_toss = sum(1 for a in assignments if a["win_probability"] < 0.62)
model_points = _model_point_map(assignments)
n_changed = sum(
    1 for a in assignments
    if a["confidence_points"] != model_points.get(a["game_id"], a["confidence_points"])
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Games", len(assignments))
m2.metric("Locks", n_lock)
m3.metric("Leans", n_lean)
m4.metric("Changed", n_changed)

st.html(_review_table_html(assignments, games_by_id, odds, rerankings))

st.divider()

# ── game grid ─────────────────────────────────────────────────────────────────

left_col, right_col = st.columns(2, gap="medium")

for i, a in enumerate(assignments):
    col = left_col if i % 2 == 0 else right_col
    game_id = a["game_id"]
    g = games_by_id.get(game_id, {})
    
    with col:
        st.html(render_game_card(a, g, odds, rerankings, model_points))
