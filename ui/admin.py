"""NFL Confidence Pool — Admin dashboard.

Shows training run history and deep analytics on model vs user picks.
Import from ui/app.py sidebar, or run standalone: uv run streamlit run ui/admin.py
"""
import json
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.db.queries import (
    get_model_training_runs,
    get_seasons_with_submissions,
    get_weekly_analytics,
    get_season_override_summary,
    get_weekly_overall_stats,
)
from src.db.queries import _conn as db_conn


def render_admin(season: int | None = None, week: int | None = None):
    """Render the admin dashboard. Called from ui/app.py sidebar."""

    with st.sidebar:
        if st.button("← Back to Picks", use_container_width=True):
            st.session_state["admin_mode"] = False
            st.rerun()

    # Get available seasons
    try:
        seasons = get_seasons_with_submissions(_db_path())
        if not seasons:
            conn = db_conn(_db_path())
            rows = conn.execute(
                "SELECT DISTINCT season FROM games ORDER BY season DESC"
            ).fetchall()
            seasons = [int(r[0]) for r in rows]
            conn.close()
    except Exception:
        seasons = []

    if not seasons:
        st.info("No data available yet. Train a model or run a backtest first.")
        return

    if season is None:
        season = seasons[0]
    elif season not in seasons:
        season = seasons[0]

    admin_season = st.selectbox("Season", seasons, key="admin_season")
    tab1, tab2 = st.tabs(["Training Runs", "Historical Results"])

    with tab1:
        _render_training_runs_tab()

    with tab2:
        _render_historical_results_tab(admin_season)


def _db_path() -> str:
    """Get database path from config."""
    import yaml
    with open(Path(__file__).parent.parent / "config.yaml") as f:
        config = yaml.safe_load(f)
    return config["db"]["path"]


def _render_training_runs_tab():
    """Render the training runs tab with chart and table."""
    runs = get_model_training_runs(_db_path())

    if not runs:
        st.info("No training runs recorded yet. Run `train_production.py` to train a model.")
        return

    df = pd.DataFrame(runs)
    df["seasons_list"] = df["seasons_used"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Chart: CV accuracy trend over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["created_at"],
        y=df["cv_accuracy_mean"],
        mode="lines+markers",
        name="CV Accuracy",
        marker=dict(size=8, color="#4ADE80"),
        line=dict(width=2, color="#4ADE80"),
    ))
    fig.update_layout(
        title="Model CV Accuracy Over Time",
        xaxis_title="Training Date",
        yaxis_title="CV Accuracy (Mean)",
        height=350,
        template="plotly_dark",
    )
    st.plotly_chart(fig)

    # Table: all runs
    st.subheader("Training Run History")
    table_data = []
    for r in runs:
        seasons_str = ", ".join(str(s) for s in json.loads(r["seasons_used"]))
        table_data.append({
            "Date": r["created_at"][:10],
            "Version": r["model_version"],
            "CV Accuracy": f"{r['cv_accuracy_mean']:.3f} ± {r['cv_accuracy_std']:.3f}",
            "Samples": r["n_samples"],
            "Seasons": seasons_str,
        })

    st.dataframe(table_data, use_container_width=True, hide_index=True)


def _render_historical_results_tab(season: int):
    """Render the historical results tab with analytics."""

    overall = get_weekly_overall_stats(_db_path(), season)

    if not overall:
        st.info(f"No submission data found for {season}.")
        return

    # Summary cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Weeks Covered", overall["weeks_covered"])
    with c2:
        st.metric("Actual Result %", f"{overall['actual_pct']:.1%}")
    with c3:
        st.metric("Model Prediction %", f"{overall['model_pct']:.1%}")

    if overall["overrides_total"] > 0:
        c4, c5 = st.columns(2)
        with c4:
            st.metric("Overrides Saved You", overall["saved"])
        with c5:
            st.metric("Overrides Hurt You", overall["hurt"])

    # Week selector from override summary
    summary = get_season_override_summary(_db_path(), season)
    weeks_with_data = sorted(set(s["week"] for s in summary))

    if not weeks_with_data:
        st.info(f"No data for {season}.")
        return

    selected_week = st.selectbox("Week", weeks_with_data, key="admin_week")

    analytics = get_weekly_analytics(_db_path(), season, selected_week)

    if not analytics or not analytics["picks"]:
        st.info(f"No pick data for {season} Week {selected_week}.")
        return

    picks = analytics["picks"]
    overrides = [p for p in picks if p["is_overridden"]]

    # Section A: All Picks
    st.subheader(f"All Picks — Week {selected_week}")

    picks_rows = []
    for p in picks:
        correct_icon = "✅" if p["pick_correct"] else "❌"
        conf_str = f"{p['model_confidence']:.0%}" if p["model_confidence"] is not None else "—"
        picks_rows.append({
            f"{p['away_team']} @ {p['home_team']}": "",
            "Pick": p["picked_team"],
            "Source": f"({p['source']})",
            "Result": correct_icon,
            "Model Conf.": conf_str,
            "Conf Pts": p["submitted_points"],
        })

    st.dataframe(picks_rows, use_container_width=True, hide_index=True)

    # Section C: Override Detail
    if overrides:
        st.subheader(f"Override Detail — Week {selected_week}")

        saved = sum(
            1 for o in overrides
            if int(o["pick_correct"]) == 1 and int(o["model_correct"]) == 0
        )
        hurt = sum(
            1 for o in overrides
            if int(o["pick_correct"]) == 0 and int(o["model_correct"]) == 1
        )

        st.caption(f"Overrides: {len(overrides)} made · ✅ Saved you: {saved} · ❌ Hurt you: {hurt}")

        override_rows = []
        for o in overrides:
            if int(o["pick_correct"]) == 1 and int(o["model_correct"]) == 0:
                verdict = "✅ Override saved you"
            else:
                verdict = "❌ Override hurt you"

            override_rows.append({
                f"{o['away_team']} @ {o['home_team']}": "",
                "Model Correct?": "✅" if int(o["model_correct"]) else "❌",
                "User Correct?": "✅" if int(o["pick_correct"]) else "❌",
                "Verdict": verdict,
            })

        st.dataframe(override_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No overrides this week — user agreed with model on all picks.")

    # Section D: Season Override Breakdown
    st.subheader("Season Override Breakdown")

    if not any(s["overrides_made"] > 0 for s in summary):
        st.caption("No overrides made this season.")
    else:
        fig = go.Figure()
        weeks_list = [s["week"] for s in summary if s["overrides_made"] > 0]
        saved_list = [s["saved"] for s in summary if s["overrides_made"] > 0]
        hurt_list = [s["hurt"] for s in summary if s["overrides_made"] > 0]

        fig.add_trace(go.Bar(
            x=weeks_list, y=saved_list, name="Saved You",
            marker_color="#4ADE80",
        ))
        fig.add_trace(go.Bar(
            x=weeks_list, y=hurt_list, name="Hurt You",
            marker_color="#F87171",
        ))
        fig.update_layout(
            title="Overrides: Saved vs Hurt by Week",
            xaxis_title="Week",
            yaxis_title="Count",
            barmode="stack",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

        table_data = [
            {
                "Week": s["week"],
                "Overrides Made": s["overrides_made"],
                "Saved You": s["saved"],
                "Hurt You": s["hurt"],
            }
            for s in summary if s["overrides_made"] > 0
        ]

        st.dataframe(table_data, use_container_width=True, hide_index=True)
