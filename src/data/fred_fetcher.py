"""
fred_fetcher.py
---------------
Download FRED data series and cache them as CSV files.

Tries the `fredapi` library first; falls back to direct FRED API HTTP requests
if fredapi is unavailable.

Required env var: FRED_API_KEY

Series downloaded:
  - GFDEGDQ188S  Federal Debt: Total Public Debt as % of GDP (quarterly)
  - MTSDS133FMS  Federal Surplus or Deficit (monthly, billions)
  - TB4WK        4-Week Treasury Bill Secondary Market Rate (monthly)
  - VIXCLS       CBOE Volatility Index: VIX (daily)

CSV format (matches FRED standard): observation_date,value
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Series the pipeline explicitly needs fetched (TB4WK and VIXCLS may already
# exist from a prior manual download; fetch_all() will skip them if present).
REQUIRED_SERIES: list[str] = [
    "GFDEGDQ188S",  # Federal Debt % GDP, quarterly
    "MTSDS133FMS",  # Federal surplus/deficit, monthly (billions)
    "TB4WK",        # 4-week T-bill rate, monthly
    "VIXCLS",       # VIX, daily
]

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Read FRED_API_KEY from environment; raise ValueError if absent."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "FRED_API_KEY environment variable is not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return key


def _fetch_via_fredapi(series_id: str, api_key: str) -> pd.DataFrame:
    """Fetch a series using the fredapi library."""
    import fredapi  # type: ignore

    fred = fredapi.Fred(api_key=api_key)
    series = fred.get_series(series_id)
    df = series.reset_index()
    df.columns = pd.Index(["observation_date", "value"])
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date.astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _fetch_via_http(series_id: str, api_key: str) -> pd.DataFrame:
    """Fetch a series via direct FRED REST API (no fredapi dependency)."""
    import urllib.request
    import json

    params = (
        f"series_id={series_id}"
        f"&api_key={api_key}"
        f"&file_type=json"
        f"&observation_start=1900-01-01"
    )
    url = f"{FRED_BASE_URL}?{params}"
    logger.debug("HTTP GET %s", url.replace(api_key, "***"))

    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        payload = json.loads(response.read().decode())

    observations = payload.get("observations", [])
    if not observations:
        raise ValueError(f"No observations returned for series {series_id!r}")

    rows = []
    for obs in observations:
        raw_val = obs.get("value", ".")
        value = float("nan") if raw_val == "." else float(raw_val)
        rows.append({"observation_date": obs["date"], "value": value})

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _download_series(series_id: str, api_key: str) -> pd.DataFrame:
    """
    Attempt fredapi first, fall back to direct HTTP.
    Returns a DataFrame with columns [observation_date, value].
    """
    try:
        import fredapi  # noqa: F401 – probe availability
        logger.info("Using fredapi to download %s", series_id)
        return _fetch_via_fredapi(series_id, api_key)
    except ImportError:
        logger.info(
            "fredapi not installed; falling back to direct HTTP for %s", series_id
        )
        return _fetch_via_http(series_id, api_key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_series(
    series_id: str,
    output_dir: Path,
    api_key: str,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download a FRED series and cache it to ``output_dir/{series_id}.csv``.

    Parameters
    ----------
    series_id:
        FRED series identifier (e.g. ``"VIXCLS"``).
    output_dir:
        Directory where the CSV will be written.
    api_key:
        FRED API key.
    force:
        If *True*, re-download even if the CSV already exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``observation_date`` (str, YYYY-MM-DD) and
        ``value`` (float).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{series_id}.csv"

    if dest.exists() and not force:
        logger.info("Cache hit – loading %s from %s", series_id, dest)
        df = pd.read_csv(dest, parse_dates=False)
        logger.info(
            "Loaded %d observations for %s (range %s – %s)",
            len(df),
            series_id,
            df["observation_date"].iloc[0] if len(df) else "N/A",
            df["observation_date"].iloc[-1] if len(df) else "N/A",
        )
        return df

    logger.info("Downloading series %s …", series_id)
    df = _download_series(series_id, api_key)

    # Drop rows where value is NaN (FRED uses "." for missing; already coerced)
    n_before = len(df)
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(
            "Dropped %d missing-value rows from %s", n_dropped, series_id
        )

    df.to_csv(dest, index=False)
    logger.info(
        "Saved %d observations for %s to %s", len(df), series_id, dest
    )
    return df


def fetch_all(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Download all required FRED series.

    Parameters
    ----------
    output_dir:
        Target directory for CSV files. Defaults to ``data/raw/`` relative to
        the project root.
    force:
        Re-download even if CSVs already exist.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from series ID to its DataFrame.
    """
    if output_dir is None:
        output_dir = DEFAULT_RAW_DIR

    api_key = _get_api_key()
    results: dict[str, pd.DataFrame] = {}

    for series_id in REQUIRED_SERIES:
        try:
            df = fetch_series(series_id, output_dir, api_key, force=force)
            results[series_id] = df
        except Exception:
            logger.exception("Failed to fetch series %s", series_id)

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

    results = fetch_all()

    print("\n=== FRED Fetch Summary ===")
    if not results:
        print("No series were fetched (check FRED_API_KEY and network).")
        sys.exit(1)

    for series_id, df in results.items():
        if df.empty:
            print(f"  {series_id}: EMPTY")
            continue
        print(
            f"  {series_id}: {len(df):>5} observations  "
            f"[{df['observation_date'].iloc[0]} – {df['observation_date'].iloc[-1]}]"
        )
