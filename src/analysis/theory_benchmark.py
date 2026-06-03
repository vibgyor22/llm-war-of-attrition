"""
Compare LLM simulation outcomes against Alesina-Drazen theoretical predictions.

Tests:
1. KS test: empirical concession timing CDF vs theoretical Exp(mu) CDF
2. Hazard monotone with market stress (H2)
3. Cost ratio predicts which agent concedes (H3)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from src.theory.alesina_drazen import calibrate_mu

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    calibrated_mu: float
    ks_stat: float
    ks_pvalue: float
    ks_interpretation: str
    hazard_by_condition: dict = field(default_factory=dict)
    hazard_monotone_test: bool = False
    cost_predicts_direction: float = 0.0
    n_sims: int = 0
    n_censored: int = 0


def run_benchmark(
    sim_df: pd.DataFrame, period_df: pd.DataFrame
) -> BenchmarkResults:
    """Run all theoretical benchmark tests."""
    n_sims = len(sim_df)
    n_censored = (sim_df["winner"] == "CENSORED").sum()

    # Concession times (30 if censored)
    times = sim_df["concession_period"].clip(lower=0).fillna(30).values
    times = np.where(times < 0, 30, times)

    # Calibrate mu via MLE
    mu = calibrate_mu(times.tolist(), max_period=30)
    logger.info("Calibrated mu=%.4f from %d observations", mu, n_sims)

    # KS test against theoretical Exp(mu) distribution (uncensored only)
    uncensored = times[times < 30]
    if len(uncensored) > 5:
        ks_stat, ks_p = stats.kstest(uncensored, "expon", args=(0, 1.0 / mu))
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    if not np.isnan(ks_p):
        interp = (
            "LLM outcomes closely match AD theoretical CDF (p={:.3f})".format(ks_p)
            if ks_p > 0.05
            else "Significant divergence from AD theoretical CDF (p={:.3f})".format(ks_p)
        )
    else:
        interp = "Insufficient uncensored data for KS test"

    # Hazard by condition
    cond_order = ["A", "B", "C", "D", "E"]
    hazard_by_cond: dict[str, float] = {}
    for cid in cond_order:
        sub = sim_df[sim_df["condition_id"] == cid]
        if len(sub) == 0:
            continue
        sub_times = sub["concession_period"].clip(lower=0).fillna(30).values
        sub_times = np.where(sub_times < 0, 30, sub_times)
        n_events = (sub_times < 30).sum()
        total_time = sub_times.sum()
        hazard_by_cond[cid] = n_events / total_time if total_time > 0 else 0.0

    hazard_vals = [hazard_by_cond.get(c, 0) for c in cond_order if c in hazard_by_cond]
    monotone = all(x <= y for x, y in zip(hazard_vals, hazard_vals[1:])) if len(hazard_vals) > 1 else False

    # Does higher cost_ratio predict who concedes?
    if "cost_ratio_S" in sim_df.columns and "winner" in sim_df.columns:
        decided = sim_df[sim_df["winner"].isin(["HAWK", "DOVE"])]
        if len(decided) > 0:
            correct = (
                ((decided["cost_ratio_S"] > 1) & (decided["winner"] == "HAWK")) |
                ((decided["cost_ratio_S"] < 1) & (decided["winner"] == "DOVE"))
            )
            cost_predicts = correct.mean()
        else:
            cost_predicts = float("nan")
    else:
        cost_predicts = float("nan")

    return BenchmarkResults(
        calibrated_mu=float(mu),
        ks_stat=float(ks_stat) if not np.isnan(ks_stat) else 0.0,
        ks_pvalue=float(ks_p) if not np.isnan(ks_p) else 1.0,
        ks_interpretation=interp,
        hazard_by_condition=hazard_by_cond,
        hazard_monotone_test=monotone,
        cost_predicts_direction=float(cost_predicts) if not np.isnan(cost_predicts) else 0.0,
        n_sims=n_sims,
        n_censored=int(n_censored),
    )


def plot_survival_comparison(
    sim_df: pd.DataFrame, output_path: Path | None = None
) -> go.Figure:
    """Plot empirical vs theoretical survival curves."""
    times = sim_df["concession_period"].clip(lower=0).fillna(30).values
    times = np.where(times < 0, 30, times).astype(float)
    mu = calibrate_mu(times.tolist(), max_period=30)

    # Empirical survival (Kaplan-Meier style — manual)
    sorted_times = np.sort(times)
    n = len(sorted_times)
    empirical_s = np.array([np.mean(times > t) for t in sorted_times])
    theoretical_s = np.exp(-mu * sorted_times)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sorted_times, y=empirical_s,
        mode="lines", name="LLM Empirical",
        line=dict(color="#00BFFF", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=sorted_times, y=theoretical_s,
        mode="lines", name=f"AD Theory (μ={mu:.3f})",
        line=dict(color="#FF6B35", width=2, dash="dash")
    ))
    fig.update_layout(
        title="Survival Function: LLM vs Alesina-Drazen Theory",
        xaxis_title="Period",
        yaxis_title="P(No concession by period t)",
        paper_bgcolor="rgb(10,10,20)",
        plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white", size=13),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    if output_path:
        fig.write_html(str(output_path))
    return fig
