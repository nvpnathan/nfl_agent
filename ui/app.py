"""NFL Confidence Pool — Streamlit dashboard.

Tactical intelligence aesthetic. 2-column pick grid. All games on one screen.
Run: uv run streamlit run ui/app.py
"""
import sqlite3
import subprocess
import sys
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path

import html as _html

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="NFL Pool",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={},
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {
  --bg:       #04060E;
  --surf:     #0C1120;
  --surf-hi:  #111D30;
  --bdr:      rgba(255,255,255,0.06);
  --bdr-hi:   rgba(255,255,255,0.13);
  --lock:     #18E08C;
  --lean:     #F0A800;
  --toss:     #FF3050;
  --up:       #22D3EE;
  --dn:       #FB923C;
  --text:     #CBD5E1;
  --dim:      #475569;
  --mono:     'IBM Plex Mono', monospace;
  --disp:     'Syne', sans-serif;
}

/* ── global ── */
.stApp { background: var(--bg) !important; }
header[data-testid="stHeader"],
.stApp > header,
[data-testid="stDecoration"],
[data-testid="stToolbar"],
[data-testid="stStatusWidget"],
#MainMenu { display: none !important; visibility: hidden !important; height: 0 !important; }
section[data-testid="stMain"] > div { padding-top: 1.25rem; }
section[data-testid="stMain"] { padding-top: 0 !important; }
p, div, span, label { font-family: var(--mono) !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surf) !important;
  border-right: 1px solid var(--bdr) !important;
}
[data-testid="stSidebarContent"] { padding: 1.25rem 1rem; }
[data-testid="stSidebar"] label { color: var(--dim) !important; font-size: 0.7rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
[data-testid="stSidebar"] input { background: var(--surf-hi) !important; border-color: var(--bdr-hi) !important; color: var(--text) !important; font-family: var(--mono) !important; }

/* ── metrics ── */
[data-testid="stMetricLabel"] span {
  font-size: 0.6rem !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important; color: var(--dim) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--disp) !important; font-size: 1.9rem !important;
  font-weight: 800 !important; color: var(--text) !important;
}
[data-testid="metric-container"] {
  background: var(--surf) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 4px !important; padding: 0.65rem 0.9rem !important;
}

/* ── buttons — primary ── */
[data-testid="baseButton-primary"] > button,
button[kind="primary"] {
  background: var(--lock) !important;
  color: #020D07 !important;
  font-family: var(--mono) !important; font-weight: 600 !important;
  font-size: 0.68rem !important; letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border: none !important; border-radius: 3px !important;
}

/* ── buttons — secondary (swap) ── */
[data-testid="baseButton-secondary"] > button,
button[kind="secondary"] {
  background: var(--surf) !important;
  color: var(--dim) !important;
  border: 1px solid var(--bdr-hi) !important;
  font-family: var(--mono) !important; font-size: 0.63rem !important;
  letter-spacing: 0.09em !important; text-transform: uppercase !important;
  border-radius: 0 0 4px 4px !important;
  border-top: none !important; margin-top: 0 !important;
  width: 100% !important;
}
[data-testid="baseButton-secondary"] > button:hover,
button[kind="secondary"]:hover {
  color: var(--text) !important;
  border-color: rgba(255,255,255,0.22) !important;
}

/* ── inputs in dialog ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input {
  background: var(--surf-hi) !important;
  border-color: var(--bdr-hi) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important; font-size: 0.8rem !important;
}

/* ── dividers ── */
hr { border-color: var(--bdr) !important; opacity: 1 !important; margin: 0.75rem 0 !important; }

/* ── game card ── */
.gc {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-left: 3px solid transparent;
  border-radius: 4px 4px 0 0;
  border-bottom: none;
  position: relative; overflow: hidden;
}
.gc.lock { border-left-color: var(--lock); }
.gc.lean { border-left-color: var(--lean); }
.gc.toss { border-left-color: var(--toss); }
.gc.changed {
  border-color: rgba(34,211,238,0.2);
  border-left-width: 3px;
  box-shadow: 0 0 20px rgba(34,211,238,0.06);
}
.gc.lock.changed  { border-left-color: var(--lock); }
.gc.lean.changed  { border-left-color: var(--lean); }
.gc.toss.changed  { border-left-color: var(--toss); }

.gc-inner { padding: 0.85rem 1rem 0.8rem 0.85rem; }

/* top row */
.gc-top { display: flex; gap: 0.75rem; align-items: flex-start; }

.pts {
  font-family: var(--disp); font-size: 2.4rem; font-weight: 800;
  line-height: 1; min-width: 2.6rem; text-align: center; padding-top: 0.05rem;
  flex-shrink: 0;
}
.pts.lock { color: var(--lock); }
.pts.lean { color: var(--lean); }
.pts.toss { color: var(--toss); }

.gc-info { flex: 1; min-width: 0; }

.matchup {
  font-family: var(--disp); font-size: 1.05rem; font-weight: 800;
  color: #E2E8F0; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; letter-spacing: 0.02em;
}

.gmeta {
  font-family: var(--mono); font-size: 0.6rem; color: var(--dim);
  margin-top: 0.1rem; letter-spacing: 0.02em;
}

.tier-pill {
  display: inline-block; font-family: var(--mono); font-size: 0.56rem;
  font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  padding: 0.15rem 0.38rem; border-radius: 2px; margin-top: 0.2rem;
}
.tier-pill.lock { background: rgba(24,224,140,0.12); color: var(--lock); }
.tier-pill.lean { background: rgba(240,168,0,0.12); color: var(--lean); }
.tier-pill.toss { background: rgba(255,48,80,0.12); color: var(--toss); }

/* prob bar */
.prob-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.55rem 0 0.38rem; }
.prob-track { flex: 1; height: 2px; background: rgba(255,255,255,0.06); border-radius: 1px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 1px; }
.prob-fill.lock { background: var(--lock); }
.prob-fill.lean { background: var(--lean); }
.prob-fill.toss { background: var(--toss); }
.prob-pct { font-family: var(--mono); font-size: 0.65rem; font-weight: 600; white-space: nowrap; }
.prob-pct.lock { color: var(--lock); }
.prob-pct.lean { color: var(--lean); }
.prob-pct.toss { color: var(--toss); }

/* pick + spread */
.pick-row { font-family: var(--mono); font-size: 0.67rem; color: var(--dim); margin-bottom: 0.28rem; }
.pick-row b { color: var(--text); font-weight: 500; }

/* reasoning */
.rsn { font-family: var(--mono); font-size: 0.63rem; font-style: italic; color: #3D526A; line-height: 1.55; margin-top: 0.22rem; }

/* badges */
.badge-row { display: flex; gap: 0.3rem; margin-top: 0.38rem; flex-wrap: wrap; }
.badge { font-family: var(--mono); font-size: 0.56rem; font-weight: 600; letter-spacing: 0.06em; padding: 0.14rem 0.38rem; border-radius: 2px; }
.badge.up  { background: rgba(34,211,238,0.1); color: var(--up); }
.badge.dn  { background: rgba(251,146,60,0.1); color: var(--dn); }
.badge.ovr { background: rgba(255,255,255,0.05); color: var(--dim); }

/* logos */
.team-logo { width: 22px; height: 22px; border-radius: 50%; object-fit: cover; vertical-align: middle; margin-right: 0.3em; background: var(--surf-hi); flex-shrink: 0; }
.matchup-row { display: flex; align-items: center; gap: 0.15rem; }
.team-seg { display: flex; align-items: center; white-space: nowrap; }
.at-sep { color: var(--dim); font-family: var(--mono); font-size: 0.75rem; margin: 0 0.25rem; }

/* header */
.page-hdr { display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.1rem; }
.page-title { font-family: var(--disp); font-size: 1.5rem; font-weight: 800; color: #E2E8F0; letter-spacing: 0.04em; text-transform: uppercase; }
.page-dot { font-family: var(--mono); font-size: 0.65rem; color: var(--dim); letter-spacing: 0.08em; }
.page-dot.live { color: rgba(24,224,140,0.7); }
.page-dot.locked { color: rgba(240,168,0,0.7); }

.locked-badge {
  text-align: center; font-family: var(--mono); font-size: 0.65rem;
  font-weight: 600; letter-spacing: 0.1em; color: var(--lean);
  background: rgba(240,168,0,0.08); border: 1px solid rgba(240,168,0,0.2);
  border-radius: 3px; padding: 0.45rem 0.75rem;
}
</style>
""", unsafe_allow_html=True)


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
                st.rerun()
    with c2:
        if st.button("CANCEL", use_container_width=True):
            st.rerun()


@st.dialog("Lock In Picks")
def lock_dialog(season: int, week: int) -> None:
    st.markdown(f"Finalize your picks for **Week {week}, {season}**?")
    st.caption("You can still update them before kickoff.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("CONFIRM", type="primary", use_container_width=True):
            if "locked" not in st.session_state:
                st.session_state.locked = set()
            st.session_state.locked.add((season, week))
            st.rerun()
    with c2:
        if st.button("CANCEL", use_container_width=True):
            st.rerun()


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<p style="font-family:\'Syne\',sans-serif;font-size:1.1rem;'
        'font-weight:800;color:#E2E8F0;letter-spacing:0.06em;'
        'text-transform:uppercase;margin-bottom:0.5rem">NFL Pool</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    season = st.number_input("Season", min_value=2018,
                              max_value=_current_season() + 1,
                              value=_current_season(), step=1)
    week = st.number_input("Week", min_value=1, max_value=22, value=1, step=1)

    if st.button("↻  Refresh Week Data", type="primary", use_container_width=True):
        with st.spinner("Fetching games, odds, injuries, weather…"):
            _do_refresh(int(season), int(week))

    st.divider()
    api_ok = _api_get("/health") is not None
    dot = "●"
    st.caption(
        f"{dot} API connected" if api_ok else f"{dot} offline (direct mode)"
    )


# ── data ─────────────────────────────────────────────────────────────────────

season, week = int(season), int(week)
data = _load_week(season, week)

if not data or not data.get("assignments"):
    st.markdown(
        f'<div class="page-hdr"><span class="page-title">WEEK {week} · {season}</span></div>',
        unsafe_allow_html=True,
    )
    st.info("No picks yet. Click **↻ Refresh Week Data** in the sidebar.")
    st.stop()

assignments = sorted(
    data["assignments"], key=lambda a: a["confidence_points"], reverse=True
)
games_by_id = {g["espn_id"]: g for g in data.get("games", [])}
odds = _load_odds(season, week)
rerankings = _load_rerankings(season, week)

# ── page header ──────────────────────────────────────────────────────────────

is_locked = (season, week) in st.session_state.get("locked", set())
status_class = "locked" if is_locked else "live"
status_text = "● LOCKED IN" if is_locked else "● LIVE"

hdr_l, hdr_r = st.columns([4, 1])
with hdr_l:
    st.markdown(
        f'<div class="page-hdr">'
        f'<span class="page-title">WEEK {week} · {season}</span>'
        f'<span class="page-dot {status_class}">{status_text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
with hdr_r:
    if is_locked:
        st.markdown('<div class="locked-badge">✓ PICKS LOCKED IN</div>',
                    unsafe_allow_html=True)
    else:
        if st.button("LOCK IN WEEK ✓", type="primary", use_container_width=True):
            lock_dialog(season, week)

# ── summary metrics ───────────────────────────────────────────────────────────

n_lock = sum(1 for a in assignments if a["win_probability"] >= 0.70)
n_lean = sum(1 for a in assignments if 0.62 <= a["win_probability"] < 0.70)
n_toss = sum(1 for a in assignments if a["win_probability"] < 0.62)
n_changed = len(rerankings)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Games", len(assignments))
m2.metric("Locks", n_lock)
m3.metric("Leans", n_lean)
m4.metric("Changed", n_changed)

st.divider()

# ── game grid ─────────────────────────────────────────────────────────────────

left_col, right_col = st.columns(2, gap="medium")

for i, a in enumerate(assignments):
    col = left_col if i % 2 == 0 else right_col

    game_id  = a["game_id"]
    g        = games_by_id.get(game_id, {})
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
    changed_cls = " changed" if is_changed else ""
    prob_pct    = int(prob * 100)

    away_logo = (
        f'<img class="team-logo" src="https://a.espncdn.com/i/teamlogos/nfl/500/{away_eid}.png"'
        f' onerror="this.style.display=\'none\'">' if away_eid else ""
    )
    home_logo = (
        f'<img class="team-logo" src="https://a.espncdn.com/i/teamlogos/nfl/500/{home_eid}.png"'
        f' onerror="this.style.display=\'none\'">' if home_eid else ""
    )

    card = f"""
<div class="gc {tier}{changed_cls}">
  <div class="gc-inner">
    <div class="gc-top">
      <div class="pts {tier}">{pts}</div>
      <div class="gc-info">
        <div class="matchup-row">
          <span class="team-seg">{away_logo}<span class="matchup">{away_s}</span></span>
          <span class="at-sep">@</span>
          <span class="team-seg">{home_logo}<span class="matchup">{home_s}</span></span>
        </div>
        <div class="gmeta">{meta_s}</div>
        <span class="tier-pill {tier}">{t_lbl}</span>
      </div>
    </div>
    <div class="prob-row">
      <div class="prob-track">
        <div class="prob-fill {tier}" style="width:{prob_pct}%"></div>
      </div>
      <span class="prob-pct {tier}">{prob_pct}%</span>
    </div>
    <div class="pick-row"><b>{winner_s}</b> over {loser_s} &nbsp;·&nbsp; {spread_s}</div>
    {rsn_html}
    {badge_row}
  </div>
</div>
"""

    with col:
        st.markdown(card, unsafe_allow_html=True)
        if st.button("⇄  swap points", key=f"swap_{game_id}",
                     use_container_width=True):
            swap_dialog(game_id, assignments, games_by_id, season, week)
