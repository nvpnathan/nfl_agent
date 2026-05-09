"""NFL Confidence Pool — Streamlit dashboard.

Run: uv run streamlit run ui/app.py
Requires the FastAPI server to be running on port 8000, OR operates
in standalone mode by importing src directly when the API is unavailable.
"""
import sys
import yaml
from datetime import datetime, timezone
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


# ── helpers ──────────────────────────────────────────────────────────────────

def _current_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _api_get(path: str) -> dict | list | None:
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _api_post(path: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def _load_direct(season: int, week: int) -> dict | None:
    """Fallback: load assignments directly from DB when API is unavailable."""
    try:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        db_path = config["db"]["path"]
        from src.db.queries import get_weekly_assignments, get_games_for_week
        return {
            "season": season,
            "week": week,
            "assignments": get_weekly_assignments(db_path, season, week),
            "games": get_games_for_week(db_path, season, week),
        }
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return None


def _load_week(season: int, week: int) -> dict | None:
    data = _api_get(f"/week/{season}/{week}")
    if data is None:
        data = _load_direct(season, week)
    return data


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏈 NFL Confidence Pool")

    current_season = _current_season()
    season = st.number_input("Season", min_value=2018, max_value=current_season + 1,
                              value=current_season, step=1)
    week = st.number_input("Week", min_value=1, max_value=22, value=1, step=1)

    if st.button("Refresh Week Data", type="primary"):
        with st.spinner("Fetching games, odds, and injuries..."):
            result = _api_post(f"/refresh/{season}/{week}", {})
            if result:
                st.success("Refresh queued — reload in a moment.")
            else:
                # Fallback: run refresh directly
                try:
                    import subprocess, sys
                    subprocess.run(
                        [sys.executable, "scripts/refresh_weekly.py",
                         "--season", str(season), "--week", str(week)],
                        check=True, capture_output=True,
                    )
                    st.success("Refresh complete.")
                    st.rerun()
                except subprocess.CalledProcessError as e:
                    st.error(f"Refresh failed: {e.stderr.decode()[:400]}")

    st.divider()
    api_ok = _api_get("/health") is not None
    st.caption(f"API: {'🟢 connected' if api_ok else '🔴 offline (direct mode)'}")


# ── main panel ────────────────────────────────────────────────────────────────

st.header(f"Week {week} — {season} Season")

data = _load_week(int(season), int(week))

if not data or not data.get("assignments"):
    st.info("No picks for this week yet. Click **Refresh Week Data** in the sidebar.")
    st.stop()

assignments = data["assignments"]
games_by_id = {g["espn_id"]: g for g in data.get("games", [])}

# ── pick sheet ────────────────────────────────────────────────────────────────

st.subheader("Confidence Pick Sheet")

total_pts = sum(a["confidence_points"] for a in assignments)
n_uncertain = sum(1 for a in assignments if a.get("is_uncertain"))
n_overridden = sum(1 for a in assignments if a.get("is_overridden"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Games", len(assignments))
col2.metric("Total Points", total_pts)
col3.metric("Uncertain Picks", n_uncertain)
col4.metric("Overridden", n_overridden)

st.divider()

for a in assignments:
    g = games_by_id.get(a["game_id"], {})
    home = g.get("home_team") or a.get("home_team", "?")
    away = g.get("away_team") or a.get("away_team", "?")
    matchup = f"{away} @ {home}"

    winner = a["predicted_winner"]
    prob = a["win_probability"]
    pts = a["confidence_points"]
    uncertain = a.get("is_uncertain", False)
    overridden = a.get("is_overridden", False)

    flags = []
    if uncertain:
        flags.append("⚠ close call")
    if overridden:
        flags.append("✏ overridden")
    flag_str = "  " + "  ".join(flags) if flags else ""

    with st.expander(
        f"**{pts} pts** — {winner}  ({prob:.0%})  |  {matchup}{flag_str}",
        expanded=False,
    ):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write(f"**Matchup:** {matchup}")
            st.write(f"**Predicted winner:** {winner} ({prob:.1%})")
            if uncertain:
                st.warning("Close call — within 3% of adjacent pick. Consider overriding.")
            if overridden:
                reason = a.get("override_reason") or "No reason given"
                st.info(f"Manually overridden: {reason}")

        with c2:
            st.write("**Override points:**")
            new_pts = st.number_input(
                "Points", min_value=1, max_value=len(assignments),
                value=int(pts), key=f"pts_{a['game_id']}",
                label_visibility="collapsed",
            )
            reason_input = st.text_input(
                "Reason", placeholder="gut feel, injury news...",
                key=f"reason_{a['game_id']}",
                label_visibility="collapsed",
            )
            if st.button("Apply", key=f"apply_{a['game_id']}"):
                result = _api_post(f"/week/{season}/{week}/override", {
                    "game_id": a["game_id"],
                    "confidence_points": new_pts,
                    "reason": reason_input or None,
                })
                if result:
                    st.success("Saved.")
                    st.rerun()
                else:
                    # Direct fallback
                    try:
                        with open("config.yaml") as f:
                            config = yaml.safe_load(f)
                        from src.db.queries import upsert_weekly_assignment
                        upsert_weekly_assignment(config["db"]["path"], {
                            "season": int(season), "week": int(week),
                            "game_id": a["game_id"],
                            "predicted_winner": winner,
                            "confidence_points": new_pts,
                            "win_probability": prob,
                            "is_uncertain": int(a.get("is_uncertain", False)),
                            "is_overridden": 1,
                            "override_reason": reason_input or None,
                        })
                        st.success("Saved.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")

# ── raw table view ────────────────────────────────────────────────────────────

st.divider()
with st.expander("Raw pick table"):
    import pandas as pd
    rows = []
    for a in assignments:
        g = games_by_id.get(a["game_id"], {})
        rows.append({
            "Pts": a["confidence_points"],
            "Pick": a["predicted_winner"],
            "Prob": f"{a['win_probability']:.0%}",
            "Matchup": f"{g.get('away_team','?')} @ {g.get('home_team','?')}",
            "Uncertain": "⚠" if a.get("is_uncertain") else "",
            "Override": "✏" if a.get("is_overridden") else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
