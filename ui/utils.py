"""Utility functions for the NFL Confidence Pool dashboard.

Contains API client logic, database helpers, formatters, and UI components.
"""
import sqlite3
import subprocess
import sys
import yaml
import html as _html
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── helpers ──────────────────────────────────────────────────────────────────

def _current_season() -> int:
    now = datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


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


def _load_submission(season: int, week: int) -> dict | None:
    data = _api_get(f"/week/{season}/{week}/submission")
    if data is not None:
        return data.get("submission")
    try:
        db = _config()["db"]["path"]
        from src.db.queries import get_weekly_submission
        return get_weekly_submission(db, season, week)
    except Exception:
        return None


def _lock_week(season: int, week: int) -> dict | None:
    result = _api_post(f"/week/{season}/{week}/lock", {"source": "streamlit"})
    if result:
        return result
    try:
        db = _config()["db"]["path"]
        from src.db.schema import create_schema
        from src.db.queries import create_weekly_submission
        create_schema(db)
        return create_weekly_submission(db, season, week, source="streamlit")
    except Exception as e:
        st.error(f"Lock failed: {e}")
        return None


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


def _load_rerankings(season: int, week: int) -> dict[str, dict]:
    """Most recent reranking per game for this week."""
    try:
        db = _config()["db"]["path"]
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT game_id, old_points, new_points, reason
            FROM rerankings WHERE season=? AND week=?
            ORDER BY created_at DESC
        """, (season, week)).fetchall()
        conn.close()
        seen: dict[str, dict] = {}
        for r in rows:
            if r["game_id"] not in seen:
                seen[r["game_id"]] = dict(r)
        return seen
    except Exception:
        return {}


def _do_swap(season: int, week: int, game_id: str, new_pts: int, reason: str) -> bool:
    result = _api_post(f"/week/{season}/{week}/override", {
        "game_id": game_id, "confidence_points": new_pts, "reason": reason or None,
    })
    if result:
        if _load_submission(season, week):
            _lock_week(season, week)
        return True
    try:
        db = _config()["db"]["path"]
        from src.db.queries import swap_confidence_points
        swap_confidence_points(db, season, week, game_id, new_pts, reason or None)
        if _load_submission(season, week):
            _lock_week(season, week)
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False


def _do_revert(season: int, week: int, game_id: str) -> bool:
    result = _api_post(f"/week/{season}/{week}/revert", {"game_id": game_id})
    if result:
        if _load_submission(season, week):
            _lock_week(season, week)
        return True
    try:
        db = _config()["db"]["path"]
        from src.db.queries import revert_assignment_to_model
        revert_assignment_to_model(db, season, week, game_id)
        if _load_submission(season, week):
            _lock_week(season, week)
        return True
    except Exception as e:
        st.error(f"Undo failed: {e}")
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


def _spread_text(espn_id: str, home: str, away: str, odds: dict) -> str:
    o = odds.get(espn_id)
    if not o:
        return "—"
    spread = o["home_spread"]
    total = o["game_total"]
    if spread < -0.1:
        fav = f"{home} {spread:.1f}"
    elif spread > 0.1:
        fav = f"{away} -{spread:.1f}"
    else:
        fav = "EVEN"
    return f"{fav}  ·  O/U {total:.1f}"


def _tier(prob: float) -> str:
    if prob >= 0.70:
        return "lock"
    if prob >= 0.62:
        return "lean"
    return "toss"


def _tier_label(tier: str) -> str:
    return {"lock": "LOCK", "lean": "LEAN", "toss": "TOSS-UP"}[tier]


def _model_point_map(assignments: list) -> dict[str, int]:
    model_order = sorted(assignments, key=lambda x: x["win_probability"], reverse=True)
    return {
        a["game_id"]: len(model_order) - i
        for i, a in enumerate(model_order)
    }


def _review_table_html(assignments: list, games_by_id: dict,
                       odds: dict, rerankings: dict) -> str:
    rows = []
    model_points = _model_point_map(assignments)
    n_moved = 0
    for a in assignments:
        game_id = a["game_id"]
        g = games_by_id.get(game_id, {})
        home = g.get("home_team") or a.get("home_team", "?")
        away = g.get("away_team") or a.get("away_team", "?")
        winner = a["predicted_winner"]
        prob = a["win_probability"]
        pts = a["confidence_points"]
        model_pts = model_points.get(game_id, pts)
        delta = pts - model_pts
        n_moved += int(delta != 0)
        tier = _tier(prob)
        spread = _spread_text(game_id, home, away, odds)
        gtime = _format_time(g.get("game_date", ""))
        rerank = rerankings.get(game_id)
        overridden = bool(a.get("is_overridden"))

        notes = []
        if delta != 0:
            direction = "up" if delta > 0 else "dn"
            notes.append(
                f'<span class="review-note {direction}">moved {direction}</span>'
            )
        if rerank:
            delta_pts = rerank["new_points"] - rerank["old_points"]
            direction = "up" if delta_pts >= 0 else "dn"
            sign = "+" if delta_pts > 0 else ""
            notes.append(
                f'<span class="review-note {direction}">{sign}{delta_pts} pts</span>'
            )
        if overridden:
            notes.append('<span class="review-note ovr">override</span>')
        notes_html = " ".join(notes) if notes else '<span class="review-muted">-</span>'

        delta_class = "up" if delta > 0 else "dn" if delta < 0 else ""
        delta_label = f"{delta:+d}" if delta else "same"
        title_parts = [p for p in (gtime, spread, a.get("override_reason") or "") if p]
        row_title = _html.escape(" | ".join(title_parts))
        rows.append(f"""
      <tr class="{tier}" title="{row_title}">
        <td>
          <div class="compare-cell">
            <span class="compare-pts">{model_pts}</span>
            <span class="compare-pick">{_html.escape(winner)}</span>
          </div>
        </td>
        <td>
          <div class="compare-cell {tier}">
            <span class="compare-pts">{pts}</span>
            <span class="compare-pick">{_html.escape(winner)}</span>
          </div>
        </td>
        <td><span class="delta-pill {delta_class}">{delta_label}</span></td>
        <td>{_html.escape(away)} <span class="review-muted">@</span> {_html.escape(home)}</td>
        <td>{int(prob * 100)}%</td>
        <td>{_html.escape(spread)}</td>
        <td class="review-notes-cell">{notes_html}</td>
      </tr>
""")

    return f"""
<div class="review-wrap">
  <div class="review-head">
    <div class="review-title">Model vs Current</div>
    <div class="review-count">{n_moved} moved · {len(assignments)} picks</div>
  </div>
  <table class="review-table">
    <thead>
      <tr>
        <th class="col-model">Model</th>
        <th class="col-current">Current</th>
        <th class="col-delta">Delta</th>
        <th class="col-game">Game</th>
        <th class="col-win">Win</th>
        <th class="col-market">Market</th>
        <th class="col-notes">Notes</th>
      </tr>
    </thead>
    <tbody>
{''.join(rows)}
    </tbody>
  </table>
</div>
"""


def _do_refresh(season: int, week: int) -> None:
    try:
        subprocess.run(
            [sys.executable, "scripts/refresh_weekly.py",
             "--season", str(season), "--week", str(week)],
            check=True, capture_output=True, text=True,
        )
        st.success("Refresh complete.")
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error(f"Refresh failed:\n```\n{e.stderr[:600]}\n```")


# ── dialogs ──────────────────────────────────────────────────────────────────

@st.dialog("⇄  Swap Confidence Points")
def swap_dialog(game_id: str, assignments: list, games_by_id: dict,
                season: int, week: int) -> None:
    a = next(x for x in assignments if x["game_id"] == game_id)
    g = games_by_id.get(game_id, {})
    home = g.get("home_team") or a.get("home_team", "?")
    away = g.get("away_team") or a.get("away_team", "?")
    curr_pts = a["confidence_points"]

    st.markdown(f"**{away} @ {home}** — currently **{curr_pts} pts**")

    pts_to_label: dict[int, str] = {}
    for other in assignments:
        if other["game_id"] == game_id:
            continue
        og = games_by_id.get(other["game_id"], {})
        oh = og.get("home_team") or other.get("home_team", "?")
        oa = og.get("away_team") or other.get("away_team", "?")
        pts_to_label[other["confidence_points"]] = (
            f"{other['confidence_points']} pts — {oa} @ {oh}"
        )

    all_pts = sorted(pts_to_label.keys(), reverse=True)
    options = [pts_to_label[p] for p in all_pts]
    selected = st.selectbox("Swap with", options)
    new_pts = next(p for p, lbl in pts_to_label.items() if lbl == selected)

    reason = st.text_input("Reason (optional)", placeholder="injury, gut feel…")

    target_label = pts_to_label[new_pts].split("—")[1].strip()
    st.caption(f"Swaps **{curr_pts} pts** ↔ **{new_pts} pts** ({target_label})")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("APPLY", type="primary", use_container_width=True,
                     disabled=(new_pts == curr_pts)):
            if _do_swap(season, week, game_id, new_pts, reason):
                if "swap" in st.query_params:
                    del st.query_params["swap"]
                st.rerun()
    with c2:
        if st.button("CANCEL", use_container_width=True):
            if "swap" in st.query_params:
                del st.query_params["swap"]
            st.rerun()


@st.dialog("Lock In Picks")
def lock_dialog(season: int, week: int) -> None:
    st.markdown(f"Finalize your picks for **Week {week}, {season}**?")
    st.caption("This saves a database snapshot of model vs current picks.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("CONFIRM", type="primary", use_container_width=True):
            if _lock_week(season, week):
                if "locked" not in st.session_state:
                    st.session_state.locked = set()
                st.session_state.locked.add((season, week))
                st.rerun()
    with c2:
        if st.button("CANCEL", use_container_width=True):
            st.rerun()


def render_game_card(a: dict, g: dict, odds: dict, rerankings: dict, model_points: dict) -> str:
    game_id  = a["game_id"]
    home     = g.get("home_team") or a.get("home_team", "?")
    away     = g.get("away_team") or a.get("away_team", "?")
    home_eid = g.get("home_espn_id") or ""
    away_eid = g.get("away_espn_id") or ""
    winner   = a["predicted_winner"]
    loser    = away if winner == home else home
    prob     = a["win_probability"]
    pts      = a["confidence_points"]
    is_indoor   = bool(g.get("is_indoor", 0))
    overridden  = bool(a.get("is_overridden"))
    ovr_reason  = a.get("override_reason") or ""

    tier   = _tier(prob)
    t_lbl  = _tier_label(tier)
    gtime  = _format_time(g.get("game_date", ""))
    spread = _spread_text(game_id, home, away, odds)

    rerank     = rerankings.get(game_id)
    is_changed = rerank is not None
    delta_pts  = (rerank["new_points"] - rerank["old_points"]) if rerank else 0
    rsn_text   = (rerank or {}).get("reason") or ovr_reason or ""

    # meta line
    meta_parts = []
    if gtime:
        meta_parts.append(gtime)
    meta_parts.append("Indoor" if is_indoor else "Outdoor")
    meta = "  ·  ".join(meta_parts)

    # badges
    badges_html = ""
    if is_changed:
        direction = "up" if delta_pts >= 0 else "dn"
        arrow     = "↑" if delta_pts >= 0 else "↓"
        sign      = "+" if delta_pts > 0 else ""
        badges_html += (
            f'<span class="badge {direction}">'
            f'{arrow} {sign}{delta_pts} pts</span>'
        )
    if overridden:
        badges_html += '<span class="badge ovr">&#9999; overridden</span>'

    badge_row = f'<div class="badge-row">{badges_html}</div>' if badges_html else ""
    rsn_html  = f'<div class="rsn">{_html.escape(rsn_text)}</div>' if rsn_text else ""

    # escape all text content going into HTML
    away_s   = _html.escape(away)
    home_s   = _html.escape(home)
    winner_s = _html.escape(winner)
    loser_s  = _html.escape(loser)
    meta_s   = _html.escape(meta)
    spread_s = _html.escape(spread)
    swap_href = f"?swap={quote(str(game_id), safe='')}"
    undo_href = f"?undo={quote(str(game_id), safe='')}"
    changed_cls = " changed" if is_changed else ""
    prob_pct    = int(prob * 100)
    model_pts = model_points.get(game_id, pts)
    undo_btn = (
        f'<a class="undo-card-btn" href="{undo_href}" '
        f'title="Undo and restore model points ({model_pts})">↶</a>'
        if pts != model_pts else ""
    )
    away_pick_cls = f" picked {tier}" if winner == away else ""
    home_pick_cls = f" picked {tier}" if winner == home else ""
    winner_eid = away_eid if winner == away else home_eid if winner == home else ""

    away_logo = (
        f'<img class="team-logo" src="https://a.espncdn.com/i/teamlogos/nfl/500/{away_eid}.png"'
        f' onerror="this.style.display=\'none\'">' if away_eid else ""
    )
    home_logo = (
        f'<img class="team-logo" src="https://a.espncdn.com/i/teamlogos/nfl/500/{home_eid}.png"'
        f' onerror="this.style.display=\'none\'">' if home_eid else ""
    )
    pick_logo = (
        f'<img class="pick-logo" src="https://a.espncdn.com/i/teamlogos/nfl/500/{winner_eid}.png"'
        f' onerror="this.style.display=\'none\'">' if winner_eid else ""
    )

    return f"""
<div class="gc {tier}{changed_cls}">
  <div class="gc-inner">
    <div class="gc-head">
      <div class="pick-primary">
        <div class="pick-kicker">Pick</div>
        <div class="pick-title {tier}">
          {pick_logo}<span class="pick-name">{winner_s}</span>
        </div>
        <div class="pick-subline">over <b>{loser_s}</b></div>
      </div>
      <div class="pts-stack">
        <div class="pts-badge {tier}">
          <span class="pts-num">{pts}</span>
          <span class="pts-unit">pts</span>
        </div>
        <a class="swap-card-btn" href="{swap_href}" title="Swap confidence points">⇄</a>
        {undo_btn}
      </div>
    </div>
    <div class="matchup-row">
      <span class="team-seg{away_pick_cls}">{away_logo}<span class="matchup">{away_s}</span></span>
      <span class="at-sep">@</span>
      <span class="team-seg{home_pick_cls}">{home_logo}<span class="matchup">{home_s}</span></span>
    </div>
    <div class="gmeta">{meta_s}</div>
    <div class="support-row">
      <span class="tier-pill {tier}">{t_lbl}</span>
      <span class="spread-line">{spread_s}</span>
    </div>
    <div class="prob-row">
      <div class="prob-track">
        <div class="prob-fill {tier}" style="width:{prob_pct}%"></div>
      </div>
      <span class="prob-pct {tier}">{prob_pct}%</span>
    </div>
    {rsn_html}
    {badge_row}
  </div>
</div>
"""
