"""
preprocessor.py
---------------
Align all FRED series to daily frequency and compute the market_stress_index.

Steps
-----
1. Load TB4WK.csv      (monthly)   → forward-fill to daily
2. Load VIXCLS.csv     (daily)     → forward-fill gaps; warn if >20 consecutive
3. Load GFDEGDQ188S.csv (quarterly) → forward-fill to daily
4. Load MTSDS133FMS.csv (monthly)  → forward-fill to daily
5. Inner-join all four on date
6. tbill_spread = TB4WK - rolling_30d_mean(TB4WK)
7. vix_norm  = (VIX  - min) / (max - min), clipped [0, 1]
8. spread_norm = (tbill_spread - min) / (max - min), clipped [0, 1]
9. market_stress_index = 0.5 * vix_norm + 0.5 * spread_norm
10. Save to data/processed/market_stress_daily.parquet

Output columns
--------------
date, vix, tbill_4wk, tbill_spread, debt_gdp_ratio, deficit_surplus_bn,
vix_norm, tbill_spread_norm, market_stress_index
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Internal loaders
# ---------------------------------------------------------------------------

def _load_csv(path: Path, value_col: str) -> pd.Series:
    """
    Read a FRED CSV (columns: observation_date, value) and return a daily
    forward-filled pd.Series indexed by date, renamed to *value_col*.

    The CSV may be sparse (monthly, quarterly).  We reindex to a continuous
    daily DatetimeIndex spanning the data's full range and forward-fill.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected FRED data file not found: {path}\n"
            "Run src/data/fred_fetcher.py to download missing series."
        )

    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "value": value_col})
    df = df.dropna(subset=[value_col])
    df = df.sort_values("date").set_index("date")
    series = df[value_col].astype(float)

    # Reindex to daily and forward-fill sparse (monthly/quarterly) data
    daily_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    series = series.reindex(daily_index).ffill()
    series.index.name = "date"
    return series


def _load_vix(path: Path) -> pd.Series:
    """
    Load VIXCLS (daily) with forward-fill and consecutive-gap logging.
    Emits a WARNING for any run of >20 consecutively forward-filled days.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected FRED data file not found: {path}\n"
            "Run src/data/fred_fetcher.py to download VIXCLS."
        )

    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", "value": "vix"})
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    df = df.sort_values("date").set_index("date")
    series = df["vix"].astype(float)

    # Build a full daily index spanning the data range
    daily_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    raw_daily = series.reindex(daily_index)

    # Detect consecutive NaN runs before filling
    is_na = raw_daily.isna()
    if is_na.any():
        # Find run lengths using cumsum trick
        run_id = (~is_na).cumsum()
        run_lengths = is_na.groupby(run_id).sum()
        max_run = int(run_lengths.max())
        if max_run > 20:
            logger.warning(
                "VIX (VIXCLS) has a run of %d consecutive missing days that "
                "will be forward-filled. Verify data quality.",
                max_run,
            )
        else:
            logger.debug(
                "VIX has %d total missing days (max consecutive run: %d); "
                "forward-filling.",
                int(is_na.sum()),
                max_run,
            )

    filled = raw_daily.ffill()
    # If any NaNs remain at the start (no prior value to forward-fill from),
    # back-fill as a last resort and log.
    if filled.isna().any():
        n_remaining = int(filled.isna().sum())
        logger.warning(
            "VIX still has %d NaN values after forward-fill (likely at "
            "series start); back-filling.",
            n_remaining,
        )
        filled = filled.bfill()

    filled.index.name = "date"
    return filled


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _minmax_norm(series: pd.Series) -> pd.Series:
    """
    Normalise *series* to [0, 1] using its global min and max.
    If min == max (constant series), returns a series of 0.0.
    Result is clipped to [0, 1] to guard against floating-point overshoot.
    """
    lo = series.min()
    hi = series.max()
    if hi == lo:
        logger.warning(
            "Series '%s' is constant (value=%s); normalisation returns 0.",
            series.name,
            lo,
        )
        return pd.Series(0.0, index=series.index, name=series.name)
    norm = (series - lo) / (hi - lo)
    return norm.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_market_stress(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build and save ``market_stress_daily.parquet``.

    Parameters
    ----------
    raw_dir:
        Directory containing the FRED CSV files.  Defaults to
        ``<project_root>/data/raw/``.
    output_dir:
        Directory where the parquet will be written.  Defaults to
        ``<project_root>/data/processed/``.

    Returns
    -------
    pd.DataFrame
        Daily DataFrame with columns:
        date, vix, tbill_4wk, tbill_spread, debt_gdp_ratio,
        deficit_surplus_bn, vix_norm, tbill_spread_norm, market_stress_index
    """
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # 1-4. Load each series to a daily pd.Series
    # ------------------------------------------------------------------
    logger.info("Loading TB4WK …")
    tbill = _load_csv(raw_dir / "TB4WK.csv", "tbill_4wk")

    logger.info("Loading VIXCLS …")
    vix = _load_vix(raw_dir / "VIXCLS.csv")

    logger.info("Loading GFDEGDQ188S …")
    debt_gdp = _load_csv(raw_dir / "GFDEGDQ188S.csv", "debt_gdp_ratio")

    logger.info("Loading MTSDS133FMS …")
    deficit = _load_csv(raw_dir / "MTSDS133FMS.csv", "deficit_surplus_bn")

    # ------------------------------------------------------------------
    # 5. Merge all on date (inner join → only dates present in all series)
    # ------------------------------------------------------------------
    logger.info("Merging series on date (inner join) …")
    df = (
        pd.DataFrame({"vix": vix, "tbill_4wk": tbill})
        .join(pd.DataFrame({"debt_gdp_ratio": debt_gdp}), how="inner")
        .join(pd.DataFrame({"deficit_surplus_bn": deficit}), how="inner")
    )
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "date"

    if df.empty:
        raise ValueError(
            "After inner-joining all four FRED series the result is empty. "
            "Check that all CSV files have overlapping date ranges."
        )

    logger.info(
        "Merged dataset: %d daily rows  [%s – %s]",
        len(df),
        df.index.min().date(),
        df.index.max().date(),
    )

    # ------------------------------------------------------------------
    # 6. tbill_spread: deviation from 30-day rolling mean
    # ------------------------------------------------------------------
    rolling_mean = df["tbill_4wk"].rolling(window=30, min_periods=1).mean()
    df["tbill_spread"] = df["tbill_4wk"] - rolling_mean

    # ------------------------------------------------------------------
    # 7-8. Min-max normalisation (global range)
    # ------------------------------------------------------------------
    df["vix_norm"] = _minmax_norm(df["vix"].rename("vix"))
    df["tbill_spread_norm"] = _minmax_norm(df["tbill_spread"].rename("tbill_spread"))

    # ------------------------------------------------------------------
    # 9. market_stress_index
    # ------------------------------------------------------------------
    df["market_stress_index"] = 0.5 * df["vix_norm"] + 0.5 * df["tbill_spread_norm"]
    df["market_stress_index"] = df["market_stress_index"].clip(0.0, 1.0)

    # ------------------------------------------------------------------
    # 10. Save to parquet
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "market_stress_daily.parquet"

    # Reset index so 'date' becomes a column in the parquet
    out = df.reset_index()
    out = out[
        [
            "date",
            "vix",
            "tbill_4wk",
            "tbill_spread",
            "debt_gdp_ratio",
            "deficit_surplus_bn",
            "vix_norm",
            "tbill_spread_norm",
            "market_stress_index",
        ]
    ]

    out.to_parquet(dest, index=False, engine="pyarrow")
    logger.info("Saved market_stress_daily.parquet → %s  (%d rows)", dest, len(out))

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    df = build_market_stress()
    print("\n=== Market Stress Build Summary ===")
    print(f"  Rows:    {len(df)}")
    print(f"  Range:   {df['date'].min()}  →  {df['date'].max()}")
    print(f"  VIX:     min={df['vix'].min():.2f}  max={df['vix'].max():.2f}")
    print(
        f"  Stress:  min={df['market_stress_index'].min():.4f}"
        f"  max={df['market_stress_index'].max():.4f}"
        f"  mean={df['market_stress_index'].mean():.4f}"
    )
