"""CSS styles for the NFL Confidence Pool dashboard."""

STYLES = """
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

/* ── top bar ── */
.topbar {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 1rem; margin: 0.2rem 0 1rem;
}
.topbar-left { min-width: 0; }
.topbar-title {
  font-family: var(--disp); font-size: 1.75rem; font-weight: 800;
  color: #E2E8F0; letter-spacing: 0.04em; text-transform: uppercase;
}
.topbar-status {
  font-family: var(--mono); font-size: 0.75rem; color: var(--dim);
  margin-left: 0.45rem; letter-spacing: 0.08em;
}
.topbar-status.live { color: rgba(24,224,140,0.7); }
.topbar-status.locked { color: rgba(240,168,0,0.7); }
.topbar-controls {
  display: grid; grid-template-columns: 1.05fr 0.85fr 1.2fr;
  gap: 0.75rem; align-items: end; min-width: 46rem;
}
.topbar-controls label {
  font-family: var(--mono) !important; color: var(--dim) !important;
  font-size: 0.7rem !important; letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
}

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
  font-family: var(--mono) !important; font-size: 0.78rem !important;
  letter-spacing: 0.09em !important; text-transform: uppercase !important;
  border-radius: 4px !important;
  min-height: 2rem !important;
  width: 100% !important;
}
[data-testid="baseButton-secondary"] > button:hover,
button[kind="secondary"]:hover {
  color: var(--text) !important;
  border-color: rgba(255,255,255,0.22) !important;
}

/* ── inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: var(--surf-hi) !important;
  border-color: var(--bdr-hi) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important; font-size: 0.85rem !important;
}
[data-testid="stNumberInput"] label {
  color: var(--dim) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  margin-bottom: 0.15rem !important;
}
[data-testid="stNumberInput"] div[data-baseweb="input"] {
  border-radius: 4px !important;
}

/* ── dividers ── */
hr { border-color: var(--bdr) !important; opacity: 1 !important; margin: 0.75rem 0 !important; }

/* ── game card ── */
.gc {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-left: 3px solid transparent;
  border-radius: 4px;
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

.gc-inner { padding: 1rem 1.1rem 0.95rem 1rem; }

/* pick-first card header */
.gc-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 1rem;
}

.pick-primary { min-width: 0; flex: 1; }
.pick-kicker {
  font-family: var(--mono); font-size: 0.75rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--dim);
  margin-bottom: 0.22rem;
}
.pick-title {
  display: flex; align-items: center; gap: 0.58rem; min-width: 0;
  font-family: var(--disp); font-size: 2.55rem; line-height: 0.95;
  font-weight: 800; letter-spacing: 0.02em; color: #E2E8F0;
}
.pick-title.lock { color: var(--lock); }
.pick-title.lean { color: var(--lean); }
.pick-title.toss { color: var(--toss); }
.pick-logo {
  width: 38px; height: 38px; border-radius: 50%; object-fit: cover;
  background: var(--surf-hi); flex-shrink: 0;
}
.pick-name {
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.pick-subline {
  font-family: var(--mono); font-size: 0.9rem; color: var(--dim);
  margin-top: 0.28rem;
}
.pick-subline b { color: var(--text); font-weight: 600; }

.pts-stack {
  flex-shrink: 0; width: 4.5rem;
  display: flex; flex-direction: column; gap: 0.45rem;
}
.pts-badge {
  min-width: 4.5rem; text-align: center;
  padding: 0.42rem 0.52rem; border-radius: 3px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}
.pts-stack .pts-badge { min-width: 0; }
.pts-badge.lock { border-color: rgba(24,224,140,0.2); background: rgba(24,224,140,0.08); }
.pts-badge.lean { border-color: rgba(240,168,0,0.2); background: rgba(240,168,0,0.08); }
.pts-badge.toss { border-color: rgba(255,48,80,0.2); background: rgba(255,48,80,0.08); }
.pts-num {
  display: block; font-family: var(--disp); font-size: 1.65rem;
  line-height: 0.95; font-weight: 800;
}
.pts-unit {
  display: block; margin-top: 0.18rem;
  font-family: var(--mono); font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--dim);
}
.pts-badge.lock .pts-num { color: var(--lock); }
.pts-badge.lean .pts-num { color: var(--lean); }
.pts-badge.toss .pts-num { color: var(--toss); }

.swap-card-btn {
  display: flex; align-items: center; justify-content: center;
  min-height: 2.1rem; border-radius: 3px;
  border: 1px solid var(--bdr-hi);
  background: rgba(255,255,255,0.02);
  color: var(--dim) !important; text-decoration: none !important;
  font-family: var(--mono); font-size: 0.9rem; font-weight: 700;
}
.swap-card-btn:hover {
  color: var(--text) !important;
  border-color: rgba(255,255,255,0.24);
  background: rgba(255,255,255,0.04);
}
.undo-card-btn {
  display: flex; align-items: center; justify-content: center;
  min-height: 2.1rem; border-radius: 3px;
  border: 1px solid rgba(251,146,60,0.24);
  background: rgba(251,146,60,0.08);
  color: var(--dn) !important; text-decoration: none !important;
  font-family: var(--mono); font-size: 0.9rem; font-weight: 700;
}
.undo-card-btn:hover {
  color: #FDBA74 !important;
  border-color: rgba(251,146,60,0.42);
  background: rgba(251,146,60,0.12);
}

.matchup {
  font-family: var(--disp); font-size: 1.18rem; font-weight: 800;
  color: #E2E8F0; white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; letter-spacing: 0.02em;
}

.gmeta {
  font-family: var(--mono); font-size: 0.78rem; color: var(--dim);
  margin-top: 0.2rem; letter-spacing: 0.02em;
}

.tier-pill {
  display: inline-block; font-family: var(--mono); font-size: 0.72rem;
  font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  padding: 0.2rem 0.48rem; border-radius: 2px;
}
.tier-pill.lock { background: rgba(24,224,140,0.12); color: var(--lock); }
.tier-pill.lean { background: rgba(240,168,0,0.12); color: var(--lean); }
.tier-pill.toss { background: rgba(255,48,80,0.12); color: var(--toss); }

.support-row {
  display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap;
  margin-top: 0.72rem; padding-top: 0.62rem;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.spread-line { font-family: var(--mono); font-size: 0.82rem; color: var(--dim); }

/* prob bar */
.prob-row { display: flex; align-items: center; gap: 0.65rem; margin: 0.72rem 0 0.5rem; }
.prob-track { flex: 1; height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 1px; }
.prob-fill.lock { background: var(--lock); }
.prob-fill.lean { background: var(--lean); }
.prob-fill.toss { background: var(--toss); }
.prob-pct { font-family: var(--mono); font-size: 0.84rem; font-weight: 600; white-space: nowrap; }
.prob-pct.lock { color: var(--lock); }
.prob-pct.lean { color: var(--lean); }
.prob-pct.toss { color: var(--toss); }

/* reasoning */
.rsn { font-family: var(--mono); font-size: 0.8rem; font-style: italic; color: #3D526A; line-height: 1.55; margin-top: 0.32rem; }

/* badges */
.badge-row { display: flex; gap: 0.4rem; margin-top: 0.52rem; flex-wrap: wrap; }
.badge { font-family: var(--mono); font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; padding: 0.18rem 0.48rem; border-radius: 2px; }
.badge.up  { background: rgba(34,211,238,0.1); color: var(--up); }
.badge.dn  { background: rgba(251,146,60,0.1); color: var(--dn); }
.badge.ovr { background: rgba(255,255,255,0.05); color: var(--dim); }

/* logos */
.team-logo { width: 24px; height: 24px; border-radius: 50%; object-fit: cover; vertical-align: middle; margin-right: 0.35em; background: var(--surf-hi); flex-shrink: 0; }
.matchup-row { display: flex; align-items: center; gap: 0.15rem; margin-top: 0.75rem; }
.team-seg {
  display: flex; align-items: center; white-space: nowrap;
  padding: 0.12rem 0.24rem 0.12rem 0.12rem; border-radius: 3px;
}
.team-seg.picked.lock { background: rgba(24,224,140,0.1); }
.team-seg.picked.lean { background: rgba(240,168,0,0.1); }
.team-seg.picked.toss { background: rgba(255,48,80,0.1); }
.team-seg.picked.lock .matchup { color: var(--lock); }
.team-seg.picked.lean .matchup { color: var(--lean); }
.team-seg.picked.toss .matchup { color: var(--toss); }
.at-sep { color: var(--dim); font-family: var(--mono); font-size: 0.95rem; margin: 0 0.28rem; }

/* header */
.page-hdr { display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.1rem; }
.page-title { font-family: var(--disp); font-size: 1.5rem; font-weight: 800; color: #E2E8F0; letter-spacing: 0.04em; text-transform: uppercase; }
.page-dot { font-family: var(--mono); font-size: 0.65rem; color: var(--dim); letter-spacing: 0.08em; }
.page-dot.live { color: rgba(24,224,140,0.7); }
.page-dot.locked { color: rgba(240,168,0,0.7); }

.review-wrap {
  background: var(--surf);
  border: 1px solid var(--bdr);
  border-radius: 4px;
  overflow-x: auto;
  overflow-y: hidden;
  margin: 0.25rem 0 1.1rem;
}
.review-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 1rem; padding: 0.7rem 0.85rem 0.55rem;
  border-bottom: 1px solid var(--bdr);
}
.review-title {
  font-family: var(--disp); font-size: 1rem; font-weight: 800;
  letter-spacing: 0.06em; text-transform: uppercase; color: #E2E8F0;
}
.review-count {
  font-family: var(--mono); font-size: 0.72rem; color: var(--dim);
  letter-spacing: 0.08em; text-transform: uppercase;
}
.review-table {
  width: 100%; min-width: 860px; border-collapse: collapse; table-layout: fixed;
}
.review-table th {
  font-family: var(--mono); font-size: 0.64rem; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--dim);
  text-align: left; padding: 0.45rem 0.65rem;
  border-bottom: 1px solid var(--bdr);
}
.review-table td {
  font-family: var(--mono); font-size: 0.78rem; color: var(--text);
  padding: 0.5rem 0.65rem; border-bottom: 1px solid rgba(255,255,255,0.04);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.review-table tr:last-child td { border-bottom: none; }
.review-table tr.lock { box-shadow: inset 3px 0 0 var(--lock); }
.review-table tr.lean { box-shadow: inset 3px 0 0 var(--lean); }
.review-table tr.toss { box-shadow: inset 3px 0 0 var(--toss); }
.compare-cell { display: flex; align-items: baseline; gap: 0.42rem; min-width: 0; }
.compare-pts {
  font-family: var(--disp); font-size: 1.08rem; line-height: 1;
  font-weight: 800; color: #E2E8F0;
}
.compare-pick {
  font-family: var(--mono); font-size: 0.76rem; font-weight: 700;
  color: var(--text); overflow: hidden; text-overflow: ellipsis;
}
.compare-cell.lock .compare-pts,
.compare-cell.lock .compare-pick { color: var(--lock); }
.compare-cell.lean .compare-pts,
.compare-cell.lean .compare-pick { color: var(--lean); }
.compare-cell.toss .compare-pts,
.compare-cell.toss .compare-pick { color: var(--toss); }
.delta-pill {
  display: inline-block; min-width: 3.6rem; text-align: center;
  font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em;
  text-transform: uppercase; padding: 0.18rem 0.34rem; border-radius: 2px;
  background: rgba(255,255,255,0.05); color: var(--dim);
}
.delta-pill.up { background: rgba(34,211,238,0.1); color: var(--up); }
.delta-pill.dn { background: rgba(251,146,60,0.1); color: var(--dn); }
.review-muted { color: var(--dim) !important; }
.review-note {
  display: inline-block; font-size: 0.66rem; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  padding: 0.16rem 0.34rem; border-radius: 2px;
  background: rgba(255,255,255,0.05); color: var(--dim);
}
.review-note.up { background: rgba(34,211,238,0.1); color: var(--up); }
.review-note.dn { background: rgba(251,146,60,0.1); color: var(--dn); }
.review-note.ovr { background: rgba(255,255,255,0.08); color: var(--text); }
.review-notes-cell { white-space: normal !important; overflow: visible !important; }
.review-table .col-model { width: 8rem; }
.review-table .col-current { width: 8rem; }
.review-table .col-delta { width: 5rem; }
.review-table .col-game { width: 9rem; }
.review-table .col-win { width: 5rem; }
.review-table .col-market { width: 12rem; }
.review-table .col-notes { width: 8rem; }

.locked-badge {
  text-align: center; font-family: var(--mono); font-size: 0.65rem;
  font-weight: 600; letter-spacing: 0.1em; color: var(--lean);
  background: rgba(240,168,0,0.08); border: 1px solid rgba(240,168,0,0.2);
  border-radius: 3px; padding: 0.45rem 0.75rem;
}
</style>
"""
