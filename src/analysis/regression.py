"""
Econometric models for the Alesina-Drazen war of attrition analysis.

Model 1 (OLS): Concession_Time ~ Cost_Ratio + Market_Stress_Avg + Election_Pressure + C(episode_id)
Model 2 (Logit): Pr(Concede_t) ~ Cost_Ratio + Market_Stress_t + Polling_t + Days_To_XDate_t
Model 3 (Cox): hazard ~ Cost_Ratio + Market_Stress_Avg + Election_Pressure
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"


@dataclass
class RegressionResults:
    model_name: str
    n_obs: int
    n_censored: int
    coefs: dict
    se: dict
    pvalues: dict
    r_squared: float
    aic: float
    summary_text: str
    interpretation: str


def _prep_sim_df(sim_df: pd.DataFrame, cost_df: pd.DataFrame, cost_measure: str) -> pd.DataFrame:
    """Merge cost measures into sim_df and build derived variables."""
    ratio_col = f"cost_ratio_{cost_measure}"
    df = sim_df.merge(cost_df[["sim_id", ratio_col]], on="sim_id", how="left")
    df = df.rename(columns={ratio_col: "cost_ratio"})

    # Dependent variable: concession_period (30 if censored)
    df["concession_time"] = df["concession_period"].fillna(30).clip(lower=0)
    df["concession_time"] = np.where(df["concession_time"] < 0, 30, df["concession_time"])

    # Market stress avg (use from sim_df if present, else approximate from condition)
    if "final_market_stress" not in df.columns:
        df["market_stress_avg"] = 0.5
    else:
        df["market_stress_avg"] = df.get("final_market_stress", 0.5).fillna(0.5)

    # Election pressure: closer to election = higher pressure (0=far, 1=imminent)
    if "election_days_out" in df.columns:
        df["election_pressure"] = 1.0 - (df["election_days_out"] / 730.0).clip(0, 1)
    else:
        df["election_pressure"] = 0.3

    df = df.dropna(subset=["cost_ratio", "concession_time"])
    return df


def run_ols(sim_df: pd.DataFrame, cost_df: pd.DataFrame, cost_measure: str = "S") -> RegressionResults:
    """OLS with episode fixed effects. Censored obs assigned concession_time=30 (biased upward)."""
    df = _prep_sim_df(sim_df, cost_df, cost_measure)
    n_censored = int((df["concession_time"] >= 30).sum())

    # Add episode dummies if multiple episodes
    if df["episode_id"].nunique() > 1:
        formula = "concession_time ~ cost_ratio + market_stress_avg + election_pressure + C(episode_id)"
    else:
        formula = "concession_time ~ cost_ratio + market_stress_avg + election_pressure"

    try:
        model = smf.ols(formula, data=df).fit()
        coefs = dict(model.params)
        se = dict(model.bse)
        pvals = dict(model.pvalues)
        r2 = float(model.rsquared)
        aic = float(model.aic)
        summary = model.summary().as_text()

        cr = coefs.get("cost_ratio", 0)
        interp = (
            f"OLS (Measure {cost_measure}): A doubling of cost_ratio is associated with "
            f"{abs(cr):.1f} period {'shorter' if cr < 0 else 'longer'} time to concession "
            f"(β={cr:.3f}, p={pvals.get('cost_ratio', 1):.3f}). "
            f"Note: OLS treats right-censored obs (n={n_censored}) as observed at period 30 — "
            f"see Cox model for censoring-aware estimates."
        )
    except Exception as e:
        logger.warning("OLS failed: %s", e)
        coefs = se = pvals = {}
        r2 = aic = 0.0
        summary = str(e)
        interp = "Model failed — see summary."

    return RegressionResults(
        model_name=f"OLS_{cost_measure}",
        n_obs=len(df),
        n_censored=n_censored,
        coefs=coefs, se=se, pvalues=pvals,
        r_squared=r2, aic=aic,
        summary_text=summary, interpretation=interp,
    )


def run_logit(period_df: pd.DataFrame, sim_df: pd.DataFrame, cost_df: pd.DataFrame, cost_measure: str = "S") -> RegressionResults:
    """Period-level logit with clustered SEs by sim_id."""
    ratio_col = f"cost_ratio_{cost_measure}"
    cost_small = cost_df[["sim_id", ratio_col]].rename(columns={ratio_col: "cost_ratio"})
    df = period_df.merge(cost_small, on="sim_id", how="left")

    # DV: either agent concedes this period
    df["concede_t"] = (
        (df["hawk_action"] == "CONCEDE") | (df["dove_action"] == "CONCEDE")
    ).astype(int)

    df = df.dropna(subset=["cost_ratio", "market_stress_index", "polling_approval_pct", "days_to_xdate"])
    n_obs = len(df)
    n_concede = int(df["concede_t"].sum())

    try:
        formula = "concede_t ~ cost_ratio + market_stress_index + polling_approval_pct + days_to_xdate"
        model = smf.logit(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["sim_id"]}, disp=False
        )
        coefs = dict(model.params)
        se = dict(model.bse)
        pvals = dict(model.pvalues)
        r2 = float(model.prsquared)
        aic = float(model.aic)
        summary = model.summary().as_text()

        cr = coefs.get("cost_ratio", 0)
        interp = (
            f"Logit (Measure {cost_measure}): A unit increase in cost_ratio changes the log-odds "
            f"of concession by {cr:.3f} (p={pvals.get('cost_ratio', 1):.3f}). "
            f"Marginal effect: a 1-unit cost-ratio increase → "
            f"{cr * 0.1:.3f} higher probability of concession per period."
        )
    except Exception as e:
        logger.warning("Logit failed: %s", e)
        coefs = se = pvals = {}
        r2 = aic = 0.0
        summary = str(e)
        interp = "Model failed — see summary."
        n_concede = 0

    return RegressionResults(
        model_name=f"Logit_{cost_measure}",
        n_obs=n_obs,
        n_censored=n_obs - n_concede,
        coefs=coefs, se=se, pvalues=pvals,
        r_squared=r2, aic=aic,
        summary_text=summary, interpretation=interp,
    )


def run_cox(sim_df: pd.DataFrame, cost_df: pd.DataFrame, cost_measure: str = "S") -> RegressionResults:
    """Cox proportional hazard — headline censoring-aware specification."""
    df = _prep_sim_df(sim_df, cost_df, cost_measure)
    df["event"] = (df["winner"] != "CENSORED").astype(int)
    df["duration"] = df["concession_time"].clip(lower=1)

    cox_df = df[["duration", "event", "cost_ratio", "market_stress_avg", "election_pressure"]].dropna()
    n_censored = int((cox_df["event"] == 0).sum())

    try:
        cph = CoxPHFitter()
        cph.fit(cox_df, duration_col="duration", event_col="event")
        summary = cph.summary
        coefs = summary["coef"].to_dict()
        se = summary["se(coef)"].to_dict()
        pvals = summary["p"].to_dict()
        r2 = float(cph.concordance_index_)
        aic = float(cph.AIC_)
        summary_text = str(cph.summary)

        cr = coefs.get("cost_ratio", 0)
        hr = np.exp(cr)
        interp = (
            f"Cox (Measure {cost_measure}) — Headline: A unit increase in cost_ratio multiplies the "
            f"concession hazard by {hr:.2f}x (β={cr:.3f}, p={pvals.get('cost_ratio', 1):.3f}). "
            f"C-index={r2:.3f}. Handles right-censoring correctly (n_censored={n_censored})."
        )
    except Exception as e:
        logger.warning("Cox failed: %s", e)
        coefs = se = pvals = {}
        r2 = aic = 0.0
        summary_text = str(e)
        interp = "Cox model failed — see summary."

    return RegressionResults(
        model_name=f"Cox_{cost_measure}",
        n_obs=len(cox_df),
        n_censored=n_censored,
        coefs=coefs, se=se, pvalues=pvals,
        r_squared=r2, aic=aic,
        summary_text=summary_text, interpretation=interp,
    )


def run_all_models(
    sim_df: pd.DataFrame,
    period_df: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> dict[str, RegressionResults]:
    """Run all models for all available cost measures. Returns results dict."""
    results: dict[str, RegressionResults] = {}
    measures = ["S", "B"]
    if "cost_ratio_J" in cost_df.columns:
        measures.append("J")

    for m in measures:
        results[f"OLS_{m}"] = run_ols(sim_df, cost_df, m)
        results[f"Logit_{m}"] = run_logit(period_df, sim_df, cost_df, m)
        results[f"Cox_{m}"] = run_cox(sim_df, cost_df, m)

    output_path = _RESULTS_DIR / "regression_results.json"
    results_to_json(results, output_path)
    logger.info("Saved regression results to %s", output_path)
    return results


def results_to_json(results_dict: dict[str, RegressionResults], output_path: Path) -> None:
    """Serialize all RegressionResults to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, res in results_dict.items():
        d = asdict(res)
        # Ensure all numeric values are JSON-serializable
        for field in ["coefs", "se", "pvalues"]:
            d[field] = {k: (float(v) if not (isinstance(v, float) and (v != v)) else None)
                        for k, v in d.get(field, {}).items()}
        serializable[key] = d
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
