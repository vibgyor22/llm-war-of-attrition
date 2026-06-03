"""Episode Cinema — three crises unfolding side by side."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parents[2]))
from dashboard.app import _CSS_PATH

st.set_page_config(page_title="Cinema — AI Negotiation", page_icon="🎬", layout="wide")
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

st.markdown("# 🎬 EPISODE COMPARISON CINEMA")
st.markdown(
    "<span style='color:#8888aa;font-family:monospace'>"
    "Three debt ceiling crises unfolding simultaneously — spot the patterns"
    "</span>", unsafe_allow_html=True,
)
st.divider()

_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"

@st.cache_data
def _load():
    sp = _RESULTS_DIR / "simulation_results.parquet"
    pp = _RESULTS_DIR / "period_level_data.parquet"
    sim_df = pd.read_parquet(sp) if sp.exists() else pd.DataFrame()
    period_df = pd.read_parquet(pp) if pp.exists() else pd.DataFrame()
    return sim_df, period_df

sim_df, period_df = _load()

if sim_df.empty:
    st.warning("No simulation data. Run `make run` first.")
    st.stop()

EPISODES = ["2011", "2013", "2023"]
EP_COLORS = {"2011": "#FF4D6D", "2013": "#FFE66D", "2023": "#4CC9F0"}
EP_LABELS = {"2011": "2011 Budget Control Act", "2013": "2013 Shutdown", "2023": "2023 Fiscal Responsibility Act"}

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    metric_choice = st.selectbox(
        "Compare metric",
        ["Concession probability", "Delay cost (HAWK)", "Delay cost (DOVE)",
         "Market stress", "Hazard rate"],
    )
with col2:
    cond_filter = st.multiselect("Conditions to include", ["A", "B", "C", "D", "E"],
                                  default=["B", "C", "D"])

st.divider()

# ── Side-by-side concession timing distributions ───────────────────────────────
st.markdown("### Concession Timing — Side by Side")

available_eps = [e for e in EPISODES if e in sim_df["episode_id"].unique()]
if not available_eps:
    st.info("No historical episode data found.")
else:
    fig_timing = make_subplots(
        rows=1, cols=len(available_eps),
        subplot_titles=[EP_LABELS.get(e, e) for e in available_eps],
        shared_yaxes=True,
    )
    for i, ep in enumerate(available_eps, start=1):
        sub = sim_df[sim_df["episode_id"] == ep]
        if cond_filter:
            sub = sub[sub["condition_id"].isin(cond_filter)]
        if sub.empty:
            continue
        times = sub["concession_period"].fillna(30).clip(lower=0).values
        times = np.where(times < 0, 30, times)
        fig_timing.add_trace(
            go.Histogram(
                x=times, nbinsx=15, name=ep,
                marker_color=EP_COLORS.get(ep, "#AAAAAA"),
                opacity=0.85, histnorm="probability",
                showlegend=(i == 1),
            ),
            row=1, col=i,
        )
    fig_timing.update_layout(
        paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white"), height=380,
        title=dict(text="Concession Timing Distribution by Episode", font=dict(color="white")),
    )
    for ann in fig_timing.layout.annotations:
        ann.font.color = "white"
    st.plotly_chart(fig_timing, use_container_width=True)

st.divider()

# ── Period-level metric comparison ────────────────────────────────────────────
st.markdown("### Period-Level Metric Trajectories")

if not period_df.empty:
    metric_col_map = {
        "Concession probability": "hawk_concession_prob",
        "Delay cost (HAWK)": "hawk_delay_cost",
        "Delay cost (DOVE)": "dove_delay_cost",
        "Market stress": "market_stress_index",
    }
    col_name = metric_col_map.get(metric_choice, "hawk_concession_prob")

    if col_name not in period_df.columns:
        st.info(f"Column `{col_name}` not found in period data.")
    else:
        fig_traj = go.Figure()
        for ep in available_eps:
            sub = period_df[period_df["episode_id"] == ep]
            if cond_filter:
                sub = sub[sub["condition_id"].isin(cond_filter)]
            if sub.empty:
                continue
            # Mean across sims at each period
            by_period = sub.groupby("period")[col_name].mean().reset_index()
            std_period = sub.groupby("period")[col_name].std().reset_index()
            fig_traj.add_trace(go.Scatter(
                x=by_period["period"], y=by_period[col_name],
                mode="lines", name=ep, line=dict(color=EP_COLORS.get(ep, "#AAAAAA"), width=2.5),
            ))
            # Confidence band
            fig_traj.add_trace(go.Scatter(
                x=pd.concat([by_period["period"], by_period["period"].iloc[::-1]]),
                y=pd.concat([
                    by_period[col_name] + std_period[col_name].fillna(0),
                    (by_period[col_name] - std_period[col_name].fillna(0)).iloc[::-1],
                ]),
                fill="toself", fillcolor=EP_COLORS.get(ep, "#AAAAAA").replace(")", ",0.15)").replace("rgb(", "rgba("),
                line=dict(color="rgba(0,0,0,0)"), showlegend=False,
            ))

        fig_traj.update_layout(
            title=dict(text=f"{metric_choice} — Mean Across Simulations by Period",
                       font=dict(color="white")),
            xaxis=dict(title="Period", tickfont=dict(color="white"), titlefont=dict(color="white")),
            yaxis=dict(title=metric_choice, tickfont=dict(color="white"), titlefont=dict(color="white")),
            paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
            font=dict(color="white"), height=400,
            legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

st.divider()

# ── Key stats table ────────────────────────────────────────────────────────────
st.markdown("### Episode Comparison Table")
rows = []
for ep in available_eps:
    sub = sim_df[sim_df["episode_id"] == ep]
    if cond_filter:
        sub = sub[sub["condition_id"].isin(cond_filter)]
    if sub.empty:
        continue
    times = sub["concession_period"].fillna(30).clip(lower=0).values
    times = np.where(times < 0, 30, times)
    rows.append({
        "Episode": ep,
        "N sims": len(sub),
        "Median concession period": float(np.median(times)),
        "% censored": f"{100*(times >= 30).mean():.0f}%",
        "% HAWK wins": f"{100*(sub['winner']=='DOVE').mean():.0f}%",
        "% DOVE wins": f"{100*(sub['winner']=='HAWK').mean():.0f}%",
        "Conditions": ", ".join(sorted(sub["condition_id"].unique())),
    })
if rows:
    st.dataframe(pd.DataFrame(rows).set_index("Episode"), use_container_width=True)

st.divider()

# ── Hazard rate by episode ─────────────────────────────────────────────────────
st.markdown("### Empirical Hazard Rate by Episode and Condition")

if not sim_df.empty:
    fig_hz = go.Figure()
    for ep in available_eps:
        sub = sim_df[sim_df["episode_id"] == ep]
        for cid in sorted(sub["condition_id"].unique()):
            if cond_filter and cid not in cond_filter:
                continue
            csub = sub[sub["condition_id"] == cid]
            times = csub["concession_period"].fillna(30).clip(lower=0).values
            times = np.where(times < 0, 30, times)
            n_events = (times < 30).sum()
            hz = n_events / max(times.sum(), 1)
            fig_hz.add_trace(go.Bar(
                name=f"{ep}/{cid}",
                x=[f"{ep}|{cid}"],
                y=[hz],
                marker_color=EP_COLORS.get(ep, "#AAAAAA"),
                opacity=0.8 - 0.1 * "ABCDE".index(cid),
            ))

    fig_hz.update_layout(
        title=dict(text="Empirical Hazard Rate (events / total exposure)", font=dict(color="white")),
        xaxis=dict(tickfont=dict(color="white"), title="Episode | Condition",
                   titlefont=dict(color="white")),
        yaxis=dict(title="Hazard rate", tickfont=dict(color="white"), titlefont=dict(color="white")),
        paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white"), height=380, showlegend=False,
    )
    st.plotly_chart(fig_hz, use_container_width=True)
    st.caption("H2 prediction: hazard should increase from condition A → E (higher deadweight loss → faster concession).")
