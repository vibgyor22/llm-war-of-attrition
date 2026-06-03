"""3D Hazard Landscape — interactive concession probability surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))
from dashboard.app import _CSS_PATH

st.set_page_config(page_title="Hazard — AI Negotiation", page_icon="🌋", layout="wide")
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

st.markdown("# 🌋 3D HAZARD LANDSCAPE")
st.markdown(
    "<span style='color:#8888aa;font-family:monospace'>"
    "Theoretical concession probability surface (Alesina-Drazen) with LLM simulation outcomes overlaid"
    "</span>", unsafe_allow_html=True,
)
st.divider()

_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"

@st.cache_data
def _load_sim_df():
    p = _RESULTS_DIR / "simulation_results.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

sim_df = _load_sim_df()

from src.viz.hazard_surface import build_surface_figure

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    show_theory = st.checkbox("Show theoretical surface", value=True)
with col2:
    show_sims = st.checkbox("Show LLM simulation points", value=not sim_df.empty)
with col3:
    ep_options = ["All"] + (sorted(sim_df["episode_id"].unique().tolist()) if not sim_df.empty else [])
    ep_filter = st.selectbox("Filter episode", ep_options)

resolution = st.slider("Surface resolution", 20, 60, 35, 5,
                        help="Higher = smoother surface but slower render")

st.divider()

# ── 3D Surface ────────────────────────────────────────────────────────────────
with st.spinner("Computing surface..."):
    fig = build_surface_figure(
        sim_df=sim_df if not sim_df.empty else None,
        t_resolution=resolution,
        stress_resolution=resolution,
        show_theory=show_theory,
        show_sims=show_sims and not sim_df.empty,
        episode_filter=None if ep_filter == "All" else ep_filter,
    )

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Cross-section: fixed stress level ─────────────────────────────────────────
st.markdown("### Hazard Cross-Section")
st.caption("Pick a fixed market stress level and see how concession probability evolves over time.")

import numpy as np
import plotly.graph_objects as go

stress_level = st.slider("Market Stress Level", 0.0, 1.0, 0.5, 0.05)
t_arr = np.linspace(0, 30, 100)
mu_eff = 0.12 * (1.0 + 0.25 * stress_level)
z_arr = 1.0 - np.exp(-mu_eff * (30.0 - t_arr))

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=t_arr, y=np.clip(z_arr, 0, 1),
    mode="lines", name="P(concession by period t)",
    line=dict(color="#FF6B35", width=2.5),
))
fig2.add_annotation(
    x=15, y=float(1.0 - np.exp(-mu_eff * 15)),
    text=f"μ_eff = {mu_eff:.3f}", showarrow=True,
    arrowcolor="white", font=dict(color="white"),
)
fig2.update_layout(
    xaxis=dict(title="Period (days from negotiation start)", tickfont=dict(color="white"),
               titlefont=dict(color="white")),
    yaxis=dict(title="Cumulative Concession Probability", tickfont=dict(color="white"),
               titlefont=dict(color="white"), range=[0, 1]),
    paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
    font=dict(color="white"), height=320,
    title=dict(text=f"Concession CDF at stress={stress_level:.2f}", font=dict(color="white")),
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Theory vs LLM comparison ──────────────────────────────────────────────────
if not sim_df.empty:
    st.markdown("### LLM Outcomes vs Theoretical Predictions")
    from src.analysis.theory_benchmark import run_benchmark, plot_survival_comparison
    @st.cache_data
    def _run_bench(sim_hash):
        period_path = _RESULTS_DIR / "period_level_data.parquet"
        if not period_path.exists():
            return None
        period_df = pd.read_parquet(period_path)
        return run_benchmark(sim_df, period_df)
    bench = _run_bench(hash(str(sim_df.shape)))
    if bench:
        c1, c2, c3 = st.columns(3)
        c1.metric("Calibrated μ", f"{bench.calibrated_mu:.3f}", help="MLE from LLM concession times")
        c2.metric("KS p-value", f"{bench.ks_pvalue:.3f}",
                  delta="Pass (>0.05)" if bench.ks_pvalue > 0.05 else "Reject (<0.05)")
        c3.metric("Cost predicts direction", f"{bench.cost_predicts_direction:.0%}",
                  help="Fraction of sims where higher-cost agent conceded first")
        st.caption(bench.ks_interpretation)
        period_path = _RESULTS_DIR / "period_level_data.parquet"
        if period_path.exists():
            surv_fig = plot_survival_comparison(sim_df)
            st.plotly_chart(surv_fig, use_container_width=True)
