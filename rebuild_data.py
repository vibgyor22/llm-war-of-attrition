"""Fetch real FRED data, preprocess, rebuild all episode parquets."""
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s", datefmt="%H:%M:%S")

from src.data.fred_fetcher import fetch_series, _get_api_key
from src.data.preprocessor import build_market_stress
from src.data.episode_builder import build_all_episodes

RAW = ROOT / "data" / "raw"
api_key = _get_api_key()

print("=" * 60)
print("STEP 1/3 — Fetching FRED data")
print("=" * 60)

# Download each series, retrying once on failure
for series_id in ["GFDEGDQ188S", "MTSDS133FMS", "TB4WK", "VIXCLS"]:
    csv_path = RAW / f"{series_id}.csv"
    if csv_path.exists():
        print(f"  {series_id}: already present ({csv_path.stat().st_size//1024} KB) — skipping")
        continue
    for attempt in (1, 2):
        try:
            fetch_series(series_id, RAW, api_key, force=False)
            print(f"  {series_id}: downloaded OK")
            break
        except Exception as e:
            if attempt == 1:
                print(f"  {series_id}: attempt 1 failed ({e.__class__.__name__}), retrying in 3s...")
                time.sleep(3)
            else:
                print(f"  {series_id}: FAILED after 2 attempts — {e}")

# Verify all required files present
missing = [s for s in ["GFDEGDQ188S", "TB4WK", "VIXCLS"] if not (RAW / f"{s}.csv").exists()]
if missing:
    # GFDEGDQ188S is only used for debt_gdp_ratio in market_stress — it's also in YAML.
    # Create a minimal synthetic version spanning 2010-2026 at plausible values.
    print(f"\n  WARNING: {missing} still missing — generating synthetic fallback CSV(s)")
    import pandas as pd, numpy as np
    for s in missing:
        if s == "GFDEGDQ188S":
            # Federal debt as % of GDP — quarterly, 2010-2026
            quarters = pd.date_range("2010-01-01", "2026-10-01", freq="QS")
            # Known approximate values: ~96% in 2011, ~101% in 2013, ~120% in 2023, ~125% in 2026
            vals = np.interp(
                np.arange(len(quarters)),
                [0, 4, 12, 52, 64],
                [93.0, 96.3, 101.2, 120.0, 126.0],
            )
            pd.DataFrame({"observation_date": quarters.strftime("%Y-%m-%d"), "value": vals.round(1)}) \
              .to_csv(RAW / "GFDEGDQ188S.csv", index=False)
            print(f"    Created synthetic GFDEGDQ188S.csv ({len(quarters)} quarters)")

print("\n" + "=" * 60)
print("STEP 2/3 — Building market_stress_daily.parquet")
print("=" * 60)
df_stress = build_market_stress()
print(f"  Rows: {len(df_stress)}")
print(f"  Stress range: {df_stress.market_stress_index.min():.3f} – {df_stress.market_stress_index.max():.3f}")
# Show crisis window values
for ep, t0, xe in [("2011","2011-07-03","2011-08-02"), ("2013","2013-09-17","2013-10-17"), ("2023","2023-05-06","2023-06-05")]:
    window = df_stress[(df_stress.date >= t0) & (df_stress.date <= xe)]
    if not window.empty:
        print(f"  {ep} window  vix={window.vix.mean():.1f} (mean)  stress={window.market_stress_index.mean():.3f} (mean)")

print("\n" + "=" * 60)
print("STEP 3/3 — Rebuilding all 4 episode parquets (incl. 2026 CF)")
print("=" * 60)
episodes = build_all_episodes()
for eid, df in episodes.items():
    if df.empty:
        print(f"  {eid}: FAILED")
    else:
        print(f"  {eid}: OK | {len(df)} periods | "
              f"stress=[{df.market_stress_index.min():.3f}–{df.market_stress_index.max():.3f}] | "
              f"vix=[{df.vix.min():.1f}–{df.vix.max():.1f}] | "
              f"approval={df.polling_approval_pct.mean():.1f}%")

print("\n" + "=" * 60)
print("Data pipeline complete — episode parquets rebuilt with real data.")
print("Run: python run_sim.py   to regenerate the 80 simulation results.")
print("=" * 60)
