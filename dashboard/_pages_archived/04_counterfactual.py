"""Counterfactual Universe Explorer — run alternate histories."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))
from dashboard.app import _CSS_PATH

st.set_page_config(page_title="Counterfactual — AI Negotiation", page_icon="🔭", layout="wide")
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

st.markdown("# 🔭 COUNTERFACTUAL UNIVERSE EXPLORER")
st.markdown(
    "<span style='color:#8888aa;font-family:monospace'>"
    "Change macroeconomic parameters and instantly see predicted negotiation outcomes"
    "</span>", unsafe_allow_html=True,
)

# Prominent fictional warning
st.error(
    "⚠ **FICTIONAL COUNTERFACTUAL — NOT REAL DATA** ⚠  "
    "This explorer uses fitted regression coefficients to predict outcomes under synthetic parameters. "
    "Predictions are illustrative, not forecasts."
)
st.divider()

_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"

@st.cache_data
def _load_data():
    sim_path = _RESULTS_DIR / "simulation_results.parquet"
    sim_df = pd.read_parquet(sim_path) if sim_path.exists() else pd.DataFrame()
    from src.viz.counterfactual_plot import load_regression_results
    reg = load_regression_results()
    return sim_df, reg

sim_df, reg_results = _load_data()

from src.viz.counterfactual_plot import (
    build_predicted_timing_figure,
    build_sensitivity_figure,
)

# ── Slider controls ────────────────────────────────────────────────────────────
st.markdown("### Configure the Alternate Universe")
col1, col2 = st.columns(2)
with col1:
    vix = st.slider("VIX Level", 10.0, 60.0, 28.5, 0.5,
                    help="Higher = more market fear")
    debt_gdp = st.slider("Debt/GDP (%)", 80.0, 140.0, 124.0, 1.0,
                          help="Federal debt as share of GDP")
    approval = st.slider("Presidential Approval (%)", 25.0, 65.0, 38.0, 1.0,
                          help="Affects political cost of delay")
with col2:
    deficit = st.slider("Annual Deficit ($bn)", 400.0, 3000.0, 2100.0, 50.0,
                         help="Projected annual deficit")
    election_days = st.slider("Days to Next Election", 30, 730, 196, 10,
                               help="Electoral pressure on both sides")
    polarization = st.slider("Political Polarization (0–1)", 0.0, 1.0, 0.75, 0.05,
                              help="Higher polarization = harder to compromise")

st.divider()

# ── Predicted outcome ─────────────────────────────────────────────────────────
col_pred, col_sens = st.columns([3, 2])

with col_pred:
    st.markdown("### Predicted Concession Time Distribution")
    hist_df = sim_df if not sim_df.empty else None
    fig_pred = build_predicted_timing_figure(
        vix_level=vix,
        debt_gdp=debt_gdp,
        approval_pct=approval,
        deficit_bn=deficit,
        election_days=election_days,
        regression_results=reg_results,
        historical_sim_df=hist_df,
        n_samples=2000,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with col_sens:
    st.markdown("### Parameter Sensitivity")
    if reg_results:
        fig_sens = build_sensitivity_figure(reg_results)
        st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.info("Run simulations and analysis to see sensitivity chart.")

st.divider()

# ── Narrative output ───────────────────────────────────────────────────────────
import numpy as np
market_stress_proxy = min(1.0, vix / 40.0)
election_pressure = max(0.0, 1.0 - election_days / 730.0)
urgency = market_stress_proxy * 0.5 + election_pressure * 0.3 + (debt_gdp / 130) * 0.2
urgency_label = "LOW" if urgency < 0.3 else "MEDIUM" if urgency < 0.6 else "HIGH" if urgency < 0.85 else "EXTREME"
urgency_color = {"LOW": "#7BF1A8", "MEDIUM": "#FFE66D", "HIGH": "#FF9A3C", "EXTREME": "#FF4D6D"}[urgency_label]

st.markdown("### Scenario Narrative")
st.markdown(
    f"""
<div style='background:rgba(20,20,40,0.9);border:1px solid rgba(255,255,255,0.1);
border-radius:6px;padding:16px;font-family:monospace;font-size:0.85rem;color:#e8e8f0'>
<b>SCENARIO SUMMARY</b><br><br>
Composite urgency: <span style='color:{urgency_color};font-weight:bold'>{urgency_label}</span>
 ({urgency:.2f})<br><br>
With VIX at <b>{vix:.0f}</b>, debt/GDP at <b>{debt_gdp:.0f}%</b>, and
<b>{election_days}</b> days to the next election, the model predicts
{"moderate pressure on both sides with room for extended deadlock."
 if urgency < 0.5 else
 "elevated urgency — market stress is providing strong incentive to resolve before the X-Date."
 if urgency < 0.8 else
 "CRISIS CONDITIONS — extreme market pressure dramatically shortens the predicted time to concession."}
<br><br>
<span style='color:#555577'>Note: Prediction uses fitted Cox model coefficients.
Electoral proximity amplifies both incentives to deal and incentives to hold for political capital.</span>
</div>
""", unsafe_allow_html=True,
)

st.divider()
st.caption(
    "Counterfactual explorer uses regression coefficients from LLM simulations. "
    "Not suitable for real-world prediction. Research purposes only."
)
