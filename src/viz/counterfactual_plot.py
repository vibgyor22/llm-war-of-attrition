"""Counterfactual Universe Explorer — predict concession timing from regression coefficients."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"
_FIGURES_DIR = Path(__file__).parents[2] / "outputs" / "figures"

_FICTIONAL_BANNER = (
    "⚠ FICTIONAL COUNTERFACTUAL — NOT REAL DATA ⚠  "
    "Parameters are synthetic constructs, not actual 2025 events."
)


def load_regression_results(results_dir: Path | None = None) -> dict:
    rd = results_dir or _RESULTS_DIR
    path = rd / "regression_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _predict_concession_times(
    vix_level: float,
    debt_gdp: float,
    approval_pct: float,
    deficit_bn: float,
    election_days: int,
    regression_results: dict,
    n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Use Cox coefficients to predict concession time distribution."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Derived predictors
    market_stress = float(np.clip(vix_level / 40.0, 0, 1))
    election_pressure = float(np.clip(1.0 - election_days / 730.0, 0, 1))
    fiscal_stress = float(np.clip(debt_gdp / 130.0, 0, 1))

    # Try to get Cox_S coefficients, fall back to defaults
    cox_key = next((k for k in regression_results if k.startswith("Cox_")), None)
    base_mu = 0.12
    if cox_key:
        coefs = regression_results[cox_key].get("coefs", {})
        # Compose a linear predictor and exponentiate (proportional hazard)
        lp = (
            coefs.get("market_stress_avg", 0.5) * market_stress +
            coefs.get("election_pressure", 0.3) * election_pressure +
            coefs.get("cost_ratio", -0.1) * 1.0  # neutral cost ratio
        )
        effective_mu = base_mu * np.exp(lp) * (1 + 0.3 * fiscal_stress)
    else:
        effective_mu = base_mu * (1 + 0.3 * market_stress + 0.2 * election_pressure + 0.2 * fiscal_stress)

    effective_mu = max(float(effective_mu), 0.01)
    raw_times = rng.exponential(scale=1.0 / effective_mu, size=n_samples)
    return np.clip(raw_times, 1, 30)


def build_predicted_timing_figure(
    vix_level: float = 28.5,
    debt_gdp: float = 124.0,
    approval_pct: float = 38.0,
    deficit_bn: float = 2100.0,
    election_days: int = 196,
    regression_results: dict | None = None,
    historical_sim_df: pd.DataFrame | None = None,
    n_samples: int = 1000,
) -> go.Figure:
    """Predicted concession timing distribution under counterfactual parameters."""
    if regression_results is None:
        regression_results = load_regression_results()

    rng = np.random.default_rng(42)
    times = _predict_concession_times(
        vix_level, debt_gdp, approval_pct, deficit_bn, election_days,
        regression_results, n_samples, rng,
    )
    mean_t = float(np.mean(times))

    fig = go.Figure()

    # Predicted histogram
    fig.add_trace(go.Histogram(
        x=times, nbinsx=30, name="Predicted (Counterfactual)",
        marker_color="#FF6B35", opacity=0.8,
        histnorm="probability",
        hovertemplate="Period %{x:.0f}<br>Probability: %{y:.3f}<extra>Predicted</extra>",
    ))

    # Historical overlay if available
    if historical_sim_df is not None and len(historical_sim_df) > 0:
        hist_times = historical_sim_df["concession_period"].fillna(30).clip(lower=0).values
        hist_times = np.where(hist_times < 0, 30, hist_times)
        fig.add_trace(go.Histogram(
            x=hist_times, nbinsx=30, name="Historical (2011–2023 avg)",
            marker_color="#4CC9F0", opacity=0.5,
            histnorm="probability",
            hovertemplate="Period %{x:.0f}<br>Probability: %{y:.3f}<extra>Historical</extra>",
        ))

    # Mean line
    fig.add_vline(x=mean_t, line_color="white", line_dash="dash", line_width=2,
                  annotation_text=f"Predicted mean: {mean_t:.1f} periods",
                  annotation_font_color="white", annotation_bgcolor="rgba(0,0,0,0.6)")

    # Fictional banner
    fig.add_annotation(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text=_FICTIONAL_BANNER,
        showarrow=False,
        font=dict(color="#FF4D6D", size=11, family="monospace"),
        bgcolor="rgba(60,0,0,0.7)",
        bordercolor="#FF4D6D", borderwidth=1,
    )

    fig.update_layout(
        title=dict(text="Predicted Concession Time Distribution", font=dict(color="white", size=14)),
        xaxis=dict(title="Concession Period (1–30)", tickfont=dict(color="white"),
                   titlefont=dict(color="white")),
        yaxis=dict(title="Probability", tickfont=dict(color="white"),
                   titlefont=dict(color="white")),
        paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white"),
        barmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=440,
        margin=dict(t=80),
    )
    return fig


def build_sensitivity_figure(regression_results: dict) -> go.Figure:
    """Tornado chart: impact of each parameter on predicted concession time."""
    params = {
        "VIX Level": (15, 45, 28.5),
        "Debt/GDP (%)": (90, 135, 110),
        "Approval (%)": (30, 55, 42),
        "Deficit ($bn)": (600, 2500, 1400),
        "Days to Election": (60, 600, 365),
    }

    base_kw = dict(vix_level=28.5, debt_gdp=110, approval_pct=42,
                   deficit_bn=1400, election_days=365)
    base_times = _predict_concession_times(**base_kw, regression_results=regression_results)
    base_mean = float(np.mean(base_times))

    impacts_low: list[float] = []
    impacts_high: list[float] = []
    labels: list[str] = []
    kwmap = {
        "VIX Level": "vix_level", "Debt/GDP (%)": "debt_gdp",
        "Approval (%)": "approval_pct", "Deficit ($bn)": "deficit_bn",
        "Days to Election": "election_days",
    }

    for label, (lo, hi, _) in params.items():
        kw_lo = {**base_kw, kwmap[label]: lo}
        kw_hi = {**base_kw, kwmap[label]: hi}
        mean_lo = float(np.mean(_predict_concession_times(**kw_lo, regression_results=regression_results)))
        mean_hi = float(np.mean(_predict_concession_times(**kw_hi, regression_results=regression_results)))
        impacts_low.append(mean_lo - base_mean)
        impacts_high.append(mean_hi - base_mean)
        labels.append(label)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=impacts_low, orientation="h",
        name="Low value", marker_color="#4CC9F0", opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=impacts_high, orientation="h",
        name="High value", marker_color="#FF6B35", opacity=0.85,
    ))
    fig.add_vline(x=0, line_color="white", line_width=1.5)

    fig.update_layout(
        title=dict(text="Parameter Sensitivity — Impact on Predicted Concession Period",
                   font=dict(color="white", size=13)),
        xaxis=dict(title="Change in predicted concession period (periods)",
                   tickfont=dict(color="white"), titlefont=dict(color="white")),
        yaxis=dict(tickfont=dict(color="white")),
        paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white"), barmode="overlay",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=380,
    )
    return fig
