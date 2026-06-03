"""Live Negotiation Arena — watch HAWK vs DOVE in real time."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))

from dashboard.app import _CSS_PATH

st.set_page_config(page_title="Arena — AI Negotiation", page_icon="🥊", layout="wide")
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ── Lazy imports (only when page loads) ───────────────────────────────────────
@st.cache_resource
def _get_cache():
    from src.cache.llm_cache import LLMCache
    db = Path(__file__).parents[2] / "cache" / "llm" / "responses.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    return LLMCache.from_env(db)

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown("# 🥊 LIVE NEGOTIATION ARENA")
st.markdown(
    "<span style='color:#8888aa;font-family:monospace'>"
    "Watch HAWK (fiscal hawk) vs DOVE (fiscal liberal) negotiate in real time"
    "</span>", unsafe_allow_html=True,
)
st.divider()

col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
with col_ctrl1:
    episode_choice = st.selectbox("Episode", ["2011", "2013", "2023", "2025_counterfactual"],
                                  help="Select debt ceiling episode")
with col_ctrl2:
    condition_choice = st.selectbox("Condition", ["A – Low", "B – Medium", "C – High",
                                                    "D – Near-Crisis", "E – Extreme"],
                                     help="Market stress condition")
    cond_id = condition_choice[0]
with col_ctrl3:
    mask_outcome = st.checkbox("Mask historical outcome", value=True,
                                help="Hide the known resolution from agents (reduces memorization)")
with col_ctrl4:
    temp = st.slider("Temperature", 0.7, 1.0, 0.85, 0.05,
                     help="LLM sampling temperature")

run_button = st.button("▶ LAUNCH SIMULATION", use_container_width=True)
st.divider()

# ── Layout: two agent panels + center ─────────────────────────────────────────
hawk_col, center_col, dove_col = st.columns([5, 4, 5])

with hawk_col:
    st.markdown(
        "<h3 style='color:#ff4d6d;font-family:monospace;text-align:center'>🦅 HAWK</h3>",
        unsafe_allow_html=True,
    )
    hawk_action_ph = st.empty()
    hawk_gauge_ph = st.empty()
    hawk_statement_ph = st.empty()
    hawk_cost_ph = st.empty()

with center_col:
    st.markdown(
        "<h3 style='color:#ffe66d;font-family:monospace;text-align:center'>⚔ TERRITORY</h3>",
        unsafe_allow_html=True,
    )
    tug_ph = st.empty()
    period_ph = st.empty()
    stress_ph = st.empty()
    xdate_ph = st.empty()

with dove_col:
    st.markdown(
        "<h3 style='color:#00bfff;font-family:monospace;text-align:center'>🕊 DOVE</h3>",
        unsafe_allow_html=True,
    )
    dove_action_ph = st.empty()
    dove_gauge_ph = st.empty()
    dove_statement_ph = st.empty()
    dove_cost_ph = st.empty()

transcript_ph = st.empty()
market_ph = st.empty()
result_ph = st.empty()

# ── Simulation logic ──────────────────────────────────────────────────────────
def _action_badge(action: str) -> str:
    colors = {"HOLD": "#4CC9F0", "SIGNAL_FLEXIBILITY": "#FFE66D", "CONCEDE": "#FF4D6D"}
    c = colors.get(action, "#AAAAAA")
    return f"<span style='color:{c};font-family:monospace;font-weight:bold'>{action}</span>"

if run_button:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY not set. Add it to your .env file.")
        st.stop()

    # Load episode and condition configs
    import yaml
    import pandas as pd
    from src.viz.arena_components import (
        build_tug_of_war_figure,
        build_agent_status_figure,
        build_market_pressure_figure,
    )
    from src.cache.llm_cache import LLMCache
    from src.agents.hawk_agent import HawkAgent
    from src.agents.dove_agent import DoveAgent
    from src.simulation.game_engine import run_simulation_live

    cfg_base = Path(__file__).parents[2] / "configs"
    episode_cfg_path = cfg_base / "episodes" / f"{episode_choice}.yaml"
    condition_cfg_path = cfg_base / "conditions" / f"condition_{cond_id}.yaml"

    if not episode_cfg_path.exists():
        st.error(f"Episode config not found: {episode_cfg_path}")
        st.stop()

    with open(episode_cfg_path) as f:
        episode_cfg = yaml.safe_load(f)
    with open(condition_cfg_path) as f:
        condition_cfg = yaml.safe_load(f)

    episode_cfg["mask_historical_outcome"] = mask_outcome

    data_path = Path(__file__).parents[2] / "data" / "processed" / f"episode_{episode_choice}.parquet"
    if not data_path.exists():
        st.error(f"Episode data not found. Run `make data` first: {data_path}")
        st.stop()

    episode_df = pd.read_parquet(data_path)

    cache = _get_cache()
    hawk = HawkAgent(cache=cache, temperature=temp)
    dove = DoveAgent(cache=cache, temperature=temp)

    hawk_history: list[dict] = []
    dove_history: list[dict] = []
    period_records: list[dict] = []
    transcript_lines: list[str] = []

    import numpy as np
    rng = np.random.default_rng(42)

    try:
        for period_rec in run_simulation_live(
            episode_df=episode_df,
            episode_config=episode_cfg,
            condition_config=condition_cfg,
            hawk=hawk,
            dove=dove,
            sim_number=0,
            mask_outcome=mask_outcome,
            rng=rng,
        ):
            hd = period_rec.hawk_decision
            dd = period_rec.dove_decision

            hawk_history.append({
                "period": period_rec.period,
                "action": hd.action,
                "concession_probability": hd.concession_probability,
                "delay_cost_implied": hd.delay_cost_implied,
            })
            dove_history.append({
                "period": period_rec.period,
                "action": dd.action,
                "concession_probability": dd.concession_probability,
                "delay_cost_implied": dd.delay_cost_implied,
            })
            period_records.append({
                "period": period_rec.period,
                "vix": period_rec.vix,
                "market_stress_index": period_rec.market_stress_index,
                "tbill_4wk": getattr(period_rec, "tbill_4wk", 4.0),
            })

            # Tug-of-war
            tug_fig = build_tug_of_war_figure(
                hawk_concession_prob=hd.concession_probability,
                dove_concession_prob=dd.concession_probability,
                period=period_rec.period,
                days_to_xdate=period_rec.days_to_xdate,
                hawk_action=hd.action,
                dove_action=dd.action,
            )
            tug_ph.plotly_chart(tug_fig, use_container_width=True, key=f"tug_{period_rec.period}")

            # Agent status panels
            hawk_gauge_ph.plotly_chart(
                build_agent_status_figure(hawk_history, "HAWK"),
                use_container_width=True, key=f"hawk_status_{period_rec.period}",
            )
            dove_gauge_ph.plotly_chart(
                build_agent_status_figure(dove_history, "DOVE"),
                use_container_width=True, key=f"dove_status_{period_rec.period}",
            )

            # Action + statement badges
            hawk_action_ph.markdown(
                f"<div style='text-align:center'>{_action_badge(hd.action)}</div>",
                unsafe_allow_html=True,
            )
            dove_action_ph.markdown(
                f"<div style='text-align:center'>{_action_badge(dd.action)}</div>",
                unsafe_allow_html=True,
            )
            hawk_statement_ph.info(f"*"{hd.public_statement}"*")
            dove_statement_ph.info(f"*"{dd.public_statement}"*")
            hawk_cost_ph.caption(f"Delay cost: {hd.delay_cost_implied:.1f}/10 | P(concede next 5): {hd.concession_probability:.0%}")
            dove_cost_ph.caption(f"Delay cost: {dd.delay_cost_implied:.1f}/10 | P(concede next 5): {dd.concession_probability:.0%}")

            # Period info
            period_ph.metric("Period", f"{period_rec.period}/29")
            xdate_ph.metric("Days to X-Date", period_rec.days_to_xdate)
            stress_ph.metric("Market Stress", f"{period_rec.market_stress_index:.2f}")

            # Market chart
            market_ph.plotly_chart(
                build_market_pressure_figure(period_records),
                use_container_width=True, key=f"market_{period_rec.period}",
            )

            # Transcript
            transcript_lines.append(
                f"[P{period_rec.period:02d}] HAWK:{hd.action} | DOVE:{dd.action} | "
                f"Stress:{period_rec.market_stress_index:.2f} | Days:{period_rec.days_to_xdate}"
            )
            transcript_ph.code("\n".join(transcript_lines[-15:]), language=None)

            if not period_rec.game_continues:
                break

            time.sleep(0.3)

    except Exception as e:
        st.error(f"Simulation error: {e}")
        raise

    # Final result
    winner = getattr(dove, "_last_winner", "unknown")
    st.balloons() if winner != "CENSORED" else None
    result_ph.success(f"Simulation complete. Check the other pages for full analysis.")
