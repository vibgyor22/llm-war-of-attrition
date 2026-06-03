"""Sanity check: FRED API key + data pipeline. Run before simulations."""
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent

# ── Load .env ────────────────────────────────────────────────────────────────
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

FRED_KEY = os.environ.get("FRED_API_KEY", "")
ANTH_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

print("=" * 60)
print("SANITY CHECK")
print("=" * 60)

# ── 1. Check keys present ────────────────────────────────────────────────────
print(f"\n[1] FRED key:      {'SET (' + str(len(FRED_KEY)) + ' chars)' if FRED_KEY else 'MISSING'}")
print(f"    Anthropic key: {'SET (' + str(len(ANTH_KEY)) + ' chars)' if ANTH_KEY else 'MISSING'}")

# ── 2. FRED API call ─────────────────────────────────────────────────────────
print("\n[2] Testing FRED API (VIX, 3 days in July 2011)...")
fred_url = (
    "https://api.stlouisfed.org/fred/series/observations"
    f"?series_id=VIXCLS&observation_start=2011-07-01&observation_end=2011-07-05"
    f"&api_key={FRED_KEY}&file_type=json"
)
try:
    with urllib.request.urlopen(fred_url, timeout=10) as resp:
        data = json.load(resp)
    obs = data.get("observations", [])
    print(f"    OK — {len(obs)} observations returned")
    for o in obs[:3]:
        print(f"      {o['date']}  VIX = {o['value']}")
    fred_ok = True
except Exception as e:
    print(f"    FAILED: {e}")
    fred_ok = False

# ── 3. Anthropic API call (minimal, 1 token) ─────────────────────────────────
print("\n[3] Testing Anthropic API (1-token ping)...")
try:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTH_KEY)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=5,
        messages=[{"role": "user", "content": "Reply: OK"}],
    )
    print(f"    OK — response: {msg.content[0].text!r}")
    anth_ok = True
except Exception as e:
    print(f"    FAILED: {e}")
    anth_ok = False

# ── 4. Episode parquets present ───────────────────────────────────────────────
print("\n[4] Checking episode parquets...")
for ep in ["2011", "2013", "2023", "2025_counterfactual"]:
    p = ROOT / "data" / "processed" / f"episode_{ep}.parquet"
    print(f"    episode_{ep}: {'OK' if p.exists() else 'MISSING'}")

# ── 5. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
print(f"  FRED API:      {'PASS' if fred_ok else 'FAIL'}")
print(f"  Anthropic API: {'PASS' if anth_ok else 'FAIL'}")
if fred_ok and anth_ok:
    print("\n  All checks passed. Ready to run:")
    print("    python -m src.data.fred_fetcher")
    print("    python -m src.data.preprocessor")
    print("    python -m src.data.episode_builder")
    print("    python run_sim.py")
elif fred_ok and not anth_ok:
    print("\n  FRED OK but Anthropic failed.")
    print("  Check your ANTHROPIC_API_KEY in .env.")
    print("  Data pipeline (FRED fetch + preprocess) can still run.")
else:
    print("\n  Fix the failing keys in .env before proceeding.")
