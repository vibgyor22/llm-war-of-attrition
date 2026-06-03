"""
Three operationally distinct delay cost measures.

S: self-reported  — mean(delay_cost_implied) across active periods
B: behavioral     — 1 / period_of_first_SIGNAL_FLEXIBILITY_or_CONCEDE (sim-specific)
J: judge-extracted — from judge_evaluator.py output
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"


def compute_self_reported(period_df: pd.DataFrame) -> pd.DataFrame:
    """Compute S measure: mean(delay_cost_implied) per agent per sim.

    Returns DataFrame with columns: sim_id, hawk_S, dove_S, cost_ratio_S
    cost_ratio_S = hawk_S / dove_S  (> 1 means HAWK has higher delay cost → concedes first)
    """
    hawk = (
        period_df.groupby("sim_id")["hawk_delay_cost"]
        .mean()
        .rename("hawk_S")
    )
    dove = (
        period_df.groupby("sim_id")["dove_delay_cost"]
        .mean()
        .rename("dove_S")
    )
    df = pd.concat([hawk, dove], axis=1).reset_index()
    df["cost_ratio_S"] = df["hawk_S"] / df["dove_S"].replace(0, np.nan)
    return df


def compute_behavioral(period_df: pd.DataFrame) -> pd.DataFrame:
    """Compute B measure: 1 / first_flexibility_period (sim-specific).

    If agent never signals or concedes, assign B = 1/30 (minimum cost implied).
    Returns DataFrame with columns: sim_id, hawk_B, dove_B, cost_ratio_B
    """
    records: list[dict] = []
    for sim_id, grp in period_df.groupby("sim_id"):
        grp = grp.sort_values("period")

        # Find first period where HAWK deviates from HOLD
        hawk_flex = grp[grp["hawk_action"].isin(["SIGNAL_FLEXIBILITY", "CONCEDE"])]["period"]
        hawk_first = hawk_flex.iloc[0] if len(hawk_flex) > 0 else 30
        hawk_B = 1.0 / max(hawk_first, 1)

        # Same for DOVE
        dove_flex = grp[grp["dove_action"].isin(["SIGNAL_FLEXIBILITY", "CONCEDE"])]["period"]
        dove_first = dove_flex.iloc[0] if len(dove_flex) > 0 else 30
        dove_B = 1.0 / max(dove_first, 1)

        records.append(
            {"sim_id": sim_id, "hawk_B": hawk_B, "dove_B": dove_B}
        )

    df = pd.DataFrame(records)
    df["cost_ratio_B"] = df["hawk_B"] / df["dove_B"].replace(0, np.nan)
    return df


def compute_all_measures(
    period_df: pd.DataFrame,
    judge_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge S, B, and optionally J measures into one sim-level DataFrame."""
    s_df = compute_self_reported(period_df)
    b_df = compute_behavioral(period_df)
    merged = s_df.merge(b_df, on="sim_id", how="outer")
    if judge_df is not None:
        merged = merged.merge(
            judge_df[["sim_id", "hawk_J", "dove_J", "cost_ratio_J"]],
            on="sim_id",
            how="left",
        )
    return merged


def load_and_compute(results_dir: Path | None = None) -> pd.DataFrame:
    """Load period_level_data.parquet and compute all cost measures."""
    rd = results_dir or _RESULTS_DIR
    period_df = pd.read_parquet(rd / "period_level_data.parquet")
    judge_path = rd / "judge_results.parquet"
    judge_df = pd.read_parquet(judge_path) if judge_path.exists() else None
    return compute_all_measures(period_df, judge_df)
