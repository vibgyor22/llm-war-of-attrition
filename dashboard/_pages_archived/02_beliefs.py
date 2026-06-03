"""Belief Evolution Theater — what each agent believes about the other's delay cost."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))
from dashboard.app import _CSS_PATH

st.set_page_config(page_title="Beliefs — AI Negotiation", page_icon="🧠", layout="wide")
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

st.markdown("# 🧠 BELIEF EVOLUTION THEATER")
st.markdown(
    "<span style='color:#8888aa;font-family:monospace'>"
    "What each side believes about the opponent's delay cost — and how it evolves"
    "</span>", unsafe_allow_html=True,
)
st.divider()

_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"

@st.cache_data
def _load_period_df():
    p = _RESULTS_DIR / "period_level_data.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

period_df = _load_period_df()

if period_df.empty:
    st.warning("No simulation data found. Run `make run` first.")
    st.stop()

from src.viz.belief_animation import build_belief_animation, build_belief_comparison_figure

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    episodes = sorted(period_df["episode_id"].unique())
    ep = st.selectbox("Episode", episodes)
with col2:
    conds = sorted(period_df[period_df["episode_id"] == ep]["condition_id"].unique())
    cond = st.selectbox("Condition", conds)
with col3:
    agent = st.radio("View beliefs of", ["hawk", "dove"], horizontal=True)

st.divider()

# ── Main animated chart ────────────────────────────────────────────────────────
st.markdown("### Population Belief Distribution (animated)")
st.caption("Each frame = one negotiation period. Bars = distribution of agent beliefs across all simulations.")

fig_anim = build_belief_animation(period_df, episode_id=ep, condition_id=cond, agent=agent)
st.plotly_chart(fig_anim, use_container_width=True)

st.divider()

# ── Single-sim deep dive ───────────────────────────────────────────────────────
st.markdown("### Single-Simulation Belief Trajectory")
sub = period_df[(period_df["episode_id"] == ep) & (period_df["condition_id"] == cond)]
sims = sorted(sub["sim_id"].unique())
if sims:
    chosen_sim = st.selectbox("Select simulation", sims)
    fig_single = build_belief_comparison_figure(period_df, chosen_sim)
    st.plotly_chart(fig_single, use_container_width=True)

    # Reasoning excerpts
    st.markdown("#### Agent Reasoning Excerpts")
    sim_data = sub[sub["sim_id"] == chosen_sim].sort_values("period")
    with st.expander("Show full period data"):
        display_cols = [c for c in ["period", "hawk_action", "dove_action",
                                     "hawk_concession_prob", "dove_concession_prob",
                                     "hawk_delay_cost", "dove_delay_cost",
                                     "hawk_belief_opponent", "dove_belief_opponent",
                                     "market_stress_index"] if c in sim_data.columns]
        st.dataframe(sim_data[display_cols].round(3), use_container_width=True)

st.divider()
st.markdown(
    "<p style='color:#555577;font-size:0.72rem;font-family:monospace'>"
    "Bayesian update rule: HOLD → lower opponent delay cost estimate (they can afford to wait). "
    "SIGNAL_FLEXIBILITY → higher opponent delay cost estimate (they are showing pain)."
    "</p>", unsafe_allow_html=True,
)
