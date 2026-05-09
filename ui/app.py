"""NFL Confidence Pool — Streamlit dashboard.

Run: uv run streamlit run ui/app.py
Works standalone (direct DB) or with FastAPI on port 8000.
"""
import sqlite3
import subprocess
import sys
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="NFL Confidence Pool",
    page_icon="🏈",
    layout="wide",
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _current_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _load_week(season: int, week: int) -> dict | None:
    data = _api_get(f"/week/{season}/{week}")
    if data is None:
        try:
            db = _config()["db"]["path"]
            from src.db.queries import get_weekly_assignments, get_games_for_week
            data = {
                "season": season, "week": week,
                "assignments": get_weekly_assignments(db, season, week),
                "games": get_games_for_week(db, season, week),
            }
        except Exception as e:
            st.error(f"Could not load data: {e}")
            return None
    return data


def _load_odds(season: int, week: int) -> dict[str, dict]:
    """Return {espn_id: {home_spread, game_total}} for the week."""
    try:
        db = _config()["db"]["path"]
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT go.espn_id, go.home_spread, go.game_total
            FROM game_odds go JOIN games g USING(espn_id)
            WHERE g.season=? AND g.week=?
        """, (season, week)).fetchall()
        conn.close()
        return {r["espn_id"]: dict(r) for r in rows}
    except Exception:
        return {}


def _do_swap(season: int, week: int, game_id: str, new_pts: int, reason: str) -> bool:
    result = _api_post(f"/week/{season}/{week}/override", {
        "game_id": game_id,
        "confidence_points": new_pts,
        "reason": reason or None,
    })
    if result:
        return True
    try:
        db = _config()["db"]["path"]
        from src.db.queries import swap_confidence_points
        swap_confidence_points(db, season, week, game_id, new_pts, reason or None)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


def _format_time(game_date: str) -> str:
    if not game_date:
        return ""
    try:
        dt = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
        offset = -4 if 3 <= dt.month <= 11 else -5
        local = dt + timedelta(hours=offset)
        return local.strftime("%a %b %-d · %-I:%M %p ET")
    except Exception:
        return ""


def _spread_text(espn_id: str, home_team: str, away_team: str, odds: dict) -> str:
    o = odds.get(espn_id)
    if not o:
        return "—"
    spread = o["home_spread"]
    total = o["game_total"]
    if spread < -0.1:
        fav = f"{home_team} {spread:.1f}"
    elif spread > 0.1:
        fav = f"{away_team} -{spread:.1f}"
    else:
        fav = "EVEN"
    return f"{fav}  ·  O/U {total:.1f}"


def _confidence_tier(prob: float) -> tuple[str, str]:
    """Return (badge_text, streamlit_color_hint)."""
    if prob >= 0.70:
        return "🟢 Lock", "green"
    elif prob >= 0.62:
        return "🟡 Lean", "yellow"
    else:
        return "🔴 Toss-up", "red"


def _do_refresh(season: int, week: int) -> None:
    # Run synchronously so data is ready before the UI reloads.
    # The API /refresh uses BackgroundTasks (async), so we call the script directly.
    try:
        result = subprocess.run(
            [sys.executable, "scripts/refresh_weekly.py",
             "--season", str(season), "--week", str(week)],
            check=True, capture_output=True, text=True,
        )
        st.success("Refresh complete.")
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error(f"Refresh failed:\n```\n{e.stderr[:600]}\n```")


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏈 NFL Confidence Pool")

    season = st.number_input("Season", min_value=2018,
                              max_value=_current_season() + 1,
                              value=_current_season(), step=1)
    week = st.number_input("Week", min_value=1, max_value=22, value=1, step=1)

    if st.button("Refresh Week Data", type="primary", use_container_width=True):
        with st.spinner("Fetching games, odds, injuries, weather…"):
            _do_refresh(int(season), int(week))

    st.divider()
    api_ok = _api_get("/health") is not None
    st.caption(f"API: {'🟢 connected' if api_ok else '🔴 offline (direct mode)'}")


# ── load data ─────────────────────────────────────────────────────────────────

season, week = int(season), int(week)
data = _load_week(season, week)

if not data or not data.get("assignments"):
    st.header(f"Week {week} — {season} Season")
    st.info("No picks yet. Click **Refresh Week Data** in the sidebar.")
    st.stop()

assignments = data["assignments"]
games_by_id = {g["espn_id"]: g for g in data.get("games", [])}
odds = _load_odds(season, week)

# Re-derive uncertainty from probability (>= 62% = confident enough)
for a in assignments:
    a["_close_call"] = a["win_probability"] < 0.62

# ── summary metrics ───────────────────────────────────────────────────────────

st.header(f"Week {week} — {season} Season")

n_locks = sum(1 for a in assignments if a["win_probability"] >= 0.70)
n_close = sum(1 for a in assignments if a["_close_call"])
n_overridden = sum(1 for a in assignments if a.get("is_overridden"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", len(assignments))
c2.metric("Locks (≥70%)", n_locks)
c3.metric("Close Calls (<62%)", n_close)
c4.metric("Overridden", n_overridden)

st.divider()

# ── pick cards ────────────────────────────────────────────────────────────────

for a in assignments:
    game_id = a["game_id"]
    g = games_by_id.get(game_id, {})
    home = g.get("home_team") or a.get("home_team", "?")
    away = g.get("away_team") or a.get("away_team", "?")
    winner = a["predicted_winner"]
    loser = away if winner == home else home
    prob = a["win_probability"]
    pts = a["confidence_points"]
    is_indoor = bool(g.get("is_indoor", 0))
    close_call = a["_close_call"]
    overridden = bool(a.get("is_overridden"))

    badge, _ = _confidence_tier(prob)
    venue_icon = "🏟" if is_indoor else "🌤"
    game_time = _format_time(g.get("game_date", ""))
    spread = _spread_text(game_id, home, away, odds)

    with st.container(border=True):
        # ── header row ────────────────────────────────────────────────────────
        h1, h2, h3 = st.columns([1, 5, 2])
        with h1:
            st.metric("Pts", pts)
        with h2:
            st.markdown(f"### {away} @ {home}")
            meta_parts = []
            if game_time:
                meta_parts.append(game_time)
            meta_parts.append(f"{venue_icon} {'Indoor' if is_indoor else 'Outdoor'}")
            if overridden:
                meta_parts.append("✏️ overridden")
            st.caption("  ·  ".join(meta_parts))
        with h3:
            st.markdown(f"**{badge}**")
            st.progress(prob, text=f"{winner} wins {prob:.0%}")

        # ── reasoning row ─────────────────────────────────────────────────────
        r1, r2 = st.columns([3, 2])
        with r1:
            st.markdown(
                f"**Pick:** {winner} over {loser} &nbsp;·&nbsp; "
                f"**Spread:** {spread}"
            )
            if close_call:
                st.warning(
                    f"Close call — model gives {winner} only {prob:.0%}. "
                    "Consider lowering the confidence points or overriding."
                )
            if overridden and a.get("override_reason"):
                st.info(f"Override reason: {a['override_reason']}")

        # ── override / swap form ──────────────────────────────────────────────
        with r2:
            # Build label map: points → "16 pts — KC vs DEN"
            pts_to_label = {}
            for other in assignments:
                og = games_by_id.get(other["game_id"], {})
                oh = og.get("home_team") or other.get("home_team", "?")
                oa = og.get("away_team") or other.get("away_team", "?")
                label = f"{other['confidence_points']} pts — {oa} @ {oh}"
                pts_to_label[other["confidence_points"]] = label

            all_pts = sorted(pts_to_label.keys(), reverse=True)
            current_idx = all_pts.index(pts) if pts in all_pts else 0

            with st.form(key=f"swap_{game_id}", border=False):
                selected_label = st.selectbox(
                    "Swap with",
                    options=[pts_to_label[p] for p in all_pts],
                    index=current_idx,
                    label_visibility="visible",
                    key=f"sel_{game_id}",
                )
                # Resolve selected points from label
                new_pts = next(p for p, lbl in pts_to_label.items() if lbl == selected_label)

                # Show swap preview
                if new_pts != pts:
                    swap_target = pts_to_label[new_pts]
                    st.caption(
                        f"This will swap **{pts} pts** (this game) ↔ **{new_pts} pts** ({swap_target.split('—')[1].strip()})"
                    )

                reason_in = st.text_input(
                    "Reason (optional)", placeholder="injury news, gut feel…",
                    label_visibility="visible",
                )
                if st.form_submit_button("Apply swap", use_container_width=True,
                                         disabled=(new_pts == pts)):
                    if _do_swap(season, week, game_id, new_pts, reason_in):
                        st.rerun()
