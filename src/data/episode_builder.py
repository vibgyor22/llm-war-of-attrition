"""
episode_builder.py
------------------
Build per-episode state sequences (one row per period) and save as parquet.

Historical episodes (2011, 2013, 2023)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load episode YAML from configs/episodes/{episode_id}.yaml
2. Load market_stress_daily.parquet
3. Load polling CSV from data/raw/polling_{episode_id}.csv
4. Slice market stress to episode window [t0_date, xdate) – 30 calendar days
   (t=0 ↔ t0_date, t=29 ↔ t0_date+29 days, xdate = t0_date+30 days)
5. Map calendar dates → period numbers
6. Linearly interpolate polling data to daily
7. Add episode metadata columns
8. days_to_xdate = 30 - period
9. Save → data/processed/episode_{episode_id}.parquet

Counterfactual episode (2025_counterfactual)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Load episode YAML (is_fictional: true)
2. Generate SYNTHETIC state sequence from YAML parameter values
3. Add Gaussian noise to VIX and stress around synthetic baselines
4. Save → data/processed/episode_2025_counterfactual.parquet
5. Log WARNING every call

Output schema (per row)
-----------------------
episode_id, period, date, days_to_xdate, vix, tbill_4wk, market_stress_index,
polling_approval_pct, debt_gdp_ratio, deficit_projection_bn, election_days_out
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml  # PyYAML; pip install pyyaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs" / "episodes"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_STRESS_FILE = DEFAULT_PROCESSED_DIR / "market_stress_daily.parquet"

HISTORICAL_EPISODE_IDS: list[str] = ["2011", "2013", "2023"]
COUNTERFACTUAL_EPISODE_ID: str = "2025_counterfactual"
N_PERIODS: int = 30  # always 30 calendar days (t=0..29)

# Gaussian noise seed for the counterfactual – use a fixed seed for
# reproducibility; callers can override via np.random.seed() before calling.
_RNG = np.random.default_rng(seed=2025)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(episode_id: str, configs_dir: Path) -> dict:
    """Load and return the episode YAML as a plain dict."""
    path = configs_dir / f"{episode_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Episode config not found: {path}\n"
            "Expected YAML keys: episode_id, t0_date, xdate, "
            "debt_gdp_ratio_at_t0, deficit_projection_bn, election_days_out."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_market_stress(stress_file: Path) -> pd.DataFrame:
    """Load market_stress_daily.parquet, ensuring 'date' is a DatetimeIndex."""
    if not stress_file.exists():
        raise FileNotFoundError(
            f"market_stress_daily.parquet not found: {stress_file}\n"
            "Run src/data/preprocessor.py first."
        )
    df = pd.read_parquet(stress_file, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _slice_stress(
    stress_df: pd.DataFrame,
    t0: pd.Timestamp,
    n_periods: int = N_PERIODS,
) -> pd.DataFrame:
    """
    Return rows of *stress_df* for the window [t0, t0 + n_periods - 1].
    Raises ValueError if fewer than *n_periods* rows are available.
    """
    dates = pd.date_range(t0, periods=n_periods, freq="D")
    sliced = stress_df[stress_df["date"].isin(dates)].copy()

    if len(sliced) < n_periods:
        missing = set(dates.date) - set(sliced["date"].dt.date)
        logger.warning(
            "Episode window [%s … +%d days] is missing %d market-stress "
            "observations.  Missing dates: %s",
            t0.date(),
            n_periods - 1,
            len(missing),
            sorted(str(d) for d in list(missing)[:5])
            + (["…"] if len(missing) > 5 else []),
        )
        # Reindex to the full 30-day window; forward-fill gaps within window
        sliced = sliced.set_index("date").reindex(dates).ffill().bfill().reset_index()
        sliced = sliced.rename(columns={"index": "date"})

    sliced = sliced.sort_values("date").reset_index(drop=True)
    sliced["period"] = range(n_periods)
    return sliced


def _load_polling(episode_id: str, raw_dir: Path) -> pd.DataFrame:
    """
    Load the sparse polling CSV for *episode_id*.

    Expected columns (any approval column name is accepted – we detect it):
        date, <approval_col>, [other cols …]

    Returns a DataFrame with columns: date (datetime), polling_approval_pct.
    The approval column is the first numeric column after 'date'.
    """
    path = raw_dir / f"polling_{episode_id}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Polling CSV not found: {path}\n"
            "Expected columns: date, <approval_pct_column>"
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Identify the approval column: first non-'date' column that contains
    # "_approval_pct" OR "_approval" OR treat the first numeric column.
    approval_col: Optional[str] = None
    for col in df.columns:
        if col == "date":
            continue
        if "approval" in col.lower():
            approval_col = col
            break
    if approval_col is None:
        # Fallback: first numeric column after date
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            approval_col = numeric_cols[0]
            logger.warning(
                "No '*approval*' column found in %s; using '%s' as "
                "polling_approval_pct.",
                path.name,
                approval_col,
            )
        else:
            raise ValueError(
                f"Cannot identify an approval column in {path}. "
                "Column names: " + str(df.columns.tolist())
            )

    return df[["date", approval_col]].rename(
        columns={approval_col: "polling_approval_pct"}
    )


def _interpolate_polling(
    polling_df: pd.DataFrame,
    episode_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Linearly interpolate sparse polling data onto *episode_dates*.

    Extrapolates with the nearest boundary value for dates outside the
    polling range.  Returns a pd.Series indexed by *episode_dates*.
    """
    polling_df = polling_df.sort_values("date").drop_duplicates("date")
    polling_indexed = polling_df.set_index("date")["polling_approval_pct"]

    # Reindex to the episode daily range, then interpolate
    full_range = pd.date_range(episode_dates.min(), episode_dates.max(), freq="D")
    reindexed = polling_indexed.reindex(full_range)
    interpolated = reindexed.interpolate(method="time").ffill().bfill()

    # Select only the episode dates
    result = interpolated.reindex(episode_dates)
    if result.isna().any():
        logger.warning(
            "Polling interpolation left %d NaN values; replacing with "
            "forward/back fill.",
            int(result.isna().sum()),
        )
        result = result.ffill().bfill()

    return result.rename("polling_approval_pct")


def _add_metadata(
    df: pd.DataFrame,
    cfg: dict,
    episode_id: str,
) -> pd.DataFrame:
    """Attach static episode metadata columns to *df*."""
    df = df.copy()
    df["episode_id"] = episode_id
    df["debt_gdp_ratio"] = float(cfg.get("debt_gdp_ratio_at_t0", float("nan")))
    df["deficit_projection_bn"] = float(cfg.get("deficit_projection_bn", float("nan")))
    df["election_days_out"] = int(cfg.get("election_days_out", 0))
    return df


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return *df* with exactly the output columns in the canonical order,
    casting to appropriate dtypes.
    """
    SCHEMA_COLS: list[str] = [
        "episode_id",
        "period",
        "date",
        "days_to_xdate",
        "vix",
        "tbill_4wk",
        "market_stress_index",
        "polling_approval_pct",
        "debt_gdp_ratio",
        "deficit_projection_bn",
        "election_days_out",
    ]
    missing = [c for c in SCHEMA_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required output columns: {missing}")

    out = df[SCHEMA_COLS].copy()
    out["date"] = out["date"].astype(str).str[:10]  # ISO YYYY-MM-DD string
    out["episode_id"] = out["episode_id"].astype(str)
    out["period"] = out["period"].astype(int)
    out["days_to_xdate"] = out["days_to_xdate"].astype(int)
    out["election_days_out"] = out["election_days_out"].astype(int)
    return out


# ---------------------------------------------------------------------------
# Historical episode builder
# ---------------------------------------------------------------------------

def _build_historical_episode(
    episode_id: str,
    market_stress_df: pd.DataFrame,
    configs_dir: Path,
    raw_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Build state sequence for a single historical episode."""
    cfg = _load_yaml(episode_id, configs_dir)

    t0 = pd.Timestamp(cfg["t0_date"])
    xdate = pd.Timestamp(cfg["xdate"])
    window_days = (xdate - t0).days  # expected 30
    if window_days != N_PERIODS:
        logger.warning(
            "Episode %s: xdate - t0_date = %d days (expected %d). "
            "Using N_PERIODS=%d.",
            episode_id,
            window_days,
            N_PERIODS,
            N_PERIODS,
        )

    # Slice and assign period numbers
    sliced = _slice_stress(market_stress_df, t0, N_PERIODS)
    episode_dates = pd.DatetimeIndex(pd.to_datetime(sliced["date"]))

    # Interpolate polling onto daily episode dates
    polling = _load_polling(episode_id, raw_dir)
    approval = _interpolate_polling(polling, episode_dates)
    sliced["polling_approval_pct"] = approval.values

    # Add metadata
    sliced = _add_metadata(sliced, cfg, episode_id)

    # days_to_xdate = 30 - period
    sliced["days_to_xdate"] = N_PERIODS - sliced["period"]

    return _enforce_schema(sliced)


# ---------------------------------------------------------------------------
# Counterfactual episode builder
# ---------------------------------------------------------------------------

def _build_counterfactual_episode(
    cfg: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Generate a synthetic state sequence for the 2025 counterfactual episode.
    ALL values are generated from YAML parameters + Gaussian noise.
    No real FRED data is used.
    """
    logger.warning(
        "LOADING FICTIONAL COUNTERFACTUAL EPISODE - not real data"
    )

    episode_id = cfg["episode_id"]
    t0 = pd.Timestamp(cfg["t0_date"])

    vix_baseline: float = float(cfg.get("vix_synthetic_baseline", 25.0))
    tbill_baseline: float = float(cfg.get("tbill_synthetic_baseline", 5.0))

    # Noise parameters (not in YAML; use sensible defaults)
    vix_noise_sd: float = float(cfg.get("vix_noise_sd", 2.0))
    tbill_noise_sd: float = float(cfg.get("tbill_noise_sd", 0.15))
    stress_baseline: float = float(cfg.get("stress_synthetic_baseline", 0.6))
    stress_noise_sd: float = float(cfg.get("stress_noise_sd", 0.05))

    periods = np.arange(N_PERIODS)
    dates = pd.date_range(t0, periods=N_PERIODS, freq="D")

    # Synthetic VIX: baseline + mild upward drift as xdate approaches + noise
    drift = 0.2 * periods / (N_PERIODS - 1)  # gentle ramp toward xdate
    vix_values = (
        vix_baseline
        + drift * vix_baseline * 0.15
        + _RNG.normal(0.0, vix_noise_sd, N_PERIODS)
    )
    vix_values = np.clip(vix_values, 5.0, None)  # VIX can't go negative

    # Synthetic T-bill: stable near baseline with small noise
    tbill_values = tbill_baseline + _RNG.normal(0.0, tbill_noise_sd, N_PERIODS)
    tbill_values = np.clip(tbill_values, 0.01, None)

    # Synthetic market stress: drifts up toward xdate + noise
    stress_drift = 0.1 * periods / (N_PERIODS - 1)
    stress_values = (
        stress_baseline
        + stress_drift
        + _RNG.normal(0.0, stress_noise_sd, N_PERIODS)
    )
    stress_values = np.clip(stress_values, 0.0, 1.0)

    # Synthetic polling: constant at YAML value
    polling_approval = float(
        cfg.get("polling", {}).get("president_approval_pct", 40.0)
    )

    df = pd.DataFrame(
        {
            "episode_id": episode_id,
            "period": periods,
            "date": dates,
            "days_to_xdate": N_PERIODS - periods,
            "vix": vix_values,
            "tbill_4wk": tbill_values,
            "market_stress_index": stress_values,
            "polling_approval_pct": polling_approval,
            "debt_gdp_ratio": float(cfg.get("debt_gdp_ratio_at_t0", float("nan"))),
            "deficit_projection_bn": float(
                cfg.get("deficit_projection_bn", float("nan"))
            ),
            "election_days_out": int(cfg.get("election_days_out", 0)),
        }
    )

    return _enforce_schema(df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_episode(
    episode_id: str,
    market_stress_df: Optional[pd.DataFrame] = None,
    *,
    configs_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build and save one episode's state sequence.

    Parameters
    ----------
    episode_id:
        One of ``"2011"``, ``"2013"``, ``"2023"``, or
        ``"2025_counterfactual"``.
    market_stress_df:
        Pre-loaded market stress DataFrame (from ``preprocessor.build_market_stress``).
        If *None*, loaded from the default parquet path.
    configs_dir:
        Directory containing episode YAML files.
    raw_dir:
        Directory containing polling CSVs and raw FRED CSVs.
    output_dir:
        Directory where the episode parquet will be written.

    Returns
    -------
    pd.DataFrame
        Episode state sequence with the canonical output schema.
    """
    if configs_dir is None:
        configs_dir = DEFAULT_CONFIGS_DIR
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    if output_dir is None:
        output_dir = DEFAULT_PROCESSED_DIR

    configs_dir = Path(configs_dir)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(episode_id, configs_dir)

    if cfg.get("is_fictional", False):
        df = _build_counterfactual_episode(cfg, output_dir)
    else:
        if market_stress_df is None:
            market_stress_df = _load_market_stress(DEFAULT_STRESS_FILE)
        df = _build_historical_episode(
            episode_id, market_stress_df, configs_dir, raw_dir, output_dir
        )

    dest = output_dir / f"episode_{episode_id}.parquet"
    df.to_parquet(dest, index=False, engine="pyarrow")
    logger.info(
        "Saved episode_%s.parquet → %s  (%d rows × %d cols)",
        episode_id,
        dest,
        len(df),
        len(df.columns),
    )
    return df


def build_all_episodes(
    *,
    configs_dir: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build all four episodes and save each as a parquet file.

    Episodes built:
        2011, 2013, 2023 (historical) + 2025_counterfactual (synthetic).

    Parameters
    ----------
    configs_dir, raw_dir, output_dir:
        Override default directory paths (see :func:`build_episode`).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from episode_id to its state-sequence DataFrame.
    """
    if configs_dir is None:
        configs_dir = DEFAULT_CONFIGS_DIR
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    if output_dir is None:
        output_dir = DEFAULT_PROCESSED_DIR

    # Load market stress once and share across historical episodes
    market_stress_df: Optional[pd.DataFrame] = None
    stress_path = Path(output_dir) / "market_stress_daily.parquet"
    if stress_path.exists():
        try:
            market_stress_df = _load_market_stress(stress_path)
            logger.info(
                "Loaded market_stress_daily.parquet: %d rows", len(market_stress_df)
            )
        except Exception:
            logger.exception(
                "Failed to load market_stress_daily.parquet; historical "
                "episodes will attempt to load it individually."
            )

    all_ids = HISTORICAL_EPISODE_IDS + [COUNTERFACTUAL_EPISODE_ID]
    results: dict[str, pd.DataFrame] = {}

    for eid in all_ids:
        try:
            df = build_episode(
                eid,
                market_stress_df=market_stress_df,
                configs_dir=configs_dir,
                raw_dir=raw_dir,
                output_dir=output_dir,
            )
            results[eid] = df
        except Exception:
            logger.exception("Failed to build episode %s", eid)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results = build_all_episodes()

    if not results:
        print("No episodes were built successfully.")
        sys.exit(1)

    print("\n=== Episode Build Summary ===")
    for eid, df in results.items():
        if df.empty:
            print(f"  {eid}: EMPTY")
            continue
        print(
            f"  {eid:25s}  periods={len(df):2d}"
            f"  dates=[{df['date'].iloc[0]} … {df['date'].iloc[-1]}]"
            f"  stress=[{df['market_stress_index'].min():.3f}"
            f"–{df['market_stress_index'].max():.3f}]"
            f"  vix=[{df['vix'].min():.1f}–{df['vix'].max():.1f}]"
            f"  approval_mean={df['polling_approval_pct'].mean():.1f}%"
        )
