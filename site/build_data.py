"""Export committed transcripts into small JSON files for the static site in docs/.

Run:  python site/build_data.py
No API keys or heavy dependencies needed (stdlib + numpy).
"""
import json, math, glob, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"
(OUT / "runs").mkdir(parents=True, exist_ok=True)

EPISODES = {"2011": "2011 · Budget Control Act", "2013": "2013 · Shutdown standoff",
            "2023": "2023 · Fiscal Responsibility Act", "2025_counterfactual": "2025 · Fictional"}
CONDS = {"A": "Calm", "B": "Baseline", "C": "High stress", "D": "Near crisis", "E": "Extreme"}


def t_pvalue(t, df):
    """Two-sided p-value from Student t via regularized incomplete beta (continued fraction)."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5

    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-30 else 1e-30); h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / c if abs(c) > 1e-30 else 1e-30
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / c if abs(c) > 1e-30 else 1e-30
            de = d * c; h *= de
            if abs(de - 1) < 1e-12: break
        return h
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    return min(1.0, front * betacf(a, b, x))


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; n = len(y); df = n - 2
    s2 = res @ res / df
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    lo, hi = 1.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_pvalue(mid, df) > 0.05 else (lo, mid)
    tc = (lo + hi) / 2
    return {"n": n, "const": beta[0], "beta": beta[1], "se": se[1], "se_const": se[0],
            "t": t[1], "p": t_pvalue(t[1], df), "r2": r2, "ci": [beta[1] - tc * se[1], beta[1] + tc * se[1]]}


def short(d):
    return {"a": d.get("action"), "c": d.get("delay_cost_implied"), "p": d.get("concession_probability"),
            "b": d.get("belief_opponent_delay_cost"), "s": d.get("public_statement"), "r": d.get("reasoning")}


runs = []
for f in sorted(glob.glob(str(ROOT / "outputs/transcripts/*.jsonl"))):
    lines = [json.loads(l) for l in open(f) if l.strip()]
    meta, periods = lines[0], lines[1:]
    hc = [p["hawk_decision"]["delay_cost_implied"] for p in periods if p.get("hawk_decision")]
    dc = [p["dove_decision"]["delay_cost_implied"] for p in periods if p.get("dove_decision")]
    def first_flex(key):
        for p in periods:
            if p[key]["action"] in ("SIGNAL_FLEXIBILITY", "CONCEDE"): return p["period"]
        return None
    run = {
        "id": meta["sim_id"], "ep": meta["episode_id"], "cond": meta["condition_id"],
        "rep": meta["sim_number"], "masked": f.endswith("_masked.jsonl"),
        # winner DOVE means HAWK (Republican) conceded
        "conceder": {"HAWK": "R", "DOVE": "D"}.get(meta.get("conceding_agent")),
        "day": meta["concession_period"],
        "rCost": float(np.mean(hc)), "dCost": float(np.mean(dc)),
        "rFlex": first_flex("hawk_decision"), "dFlex": first_flex("dove_decision"),
    }
    run["ratio"] = run["rCost"] / run["dCost"]
    runs.append(run)
    detail = {**run, "periods": [{
        "t": p["period"], "date": p["date"], "x": p["days_to_xdate"], "vix": p["vix"],
        "stress": p["market_stress_index"], "appr": p["polling_approval_pct"],
        "R": short(p["hawk_decision"]), "D": short(p["dove_decision"])} for p in periods]}
    json.dump(detail, open(OUT / "runs" / f"{run['id']}.json", "w"), separators=(",", ":"))

hist = [r for r in runs if not r["masked"]]
reg = {k: ols([r["ratio"] for r in s], [r["day"] for r in s])
       for k, s in {"historical": hist, "masked": [r for r in runs if r["masked"]], "pooled": runs}.items()}

by_cond = {}
for c in CONDS:
    s = [r for r in runs if r["cond"] == c]
    by_cond[c] = {"n": len(s), "rConcede": sum(r["conceder"] == "R" for r in s),
                  "medianDay": float(np.median([r["day"] for r in s])),
                  "meanDay": float(np.mean([r["day"] for r in s])),
                  "rCost": float(np.mean([r["rCost"] for r in s])), "dCost": float(np.mean([r["dCost"] for r in s]))}

summary = {"episodes": EPISODES, "conditions": CONDS, "runs": runs, "byCondition": by_cond,
           "regression": reg, "n": len(runs), "rConcede": sum(r["conceder"] == "R" for r in runs),
           "medianDay": float(np.median([r["day"] for r in runs]))}
json.dump(summary, open(OUT / "summary.json", "w"), separators=(",", ":"), default=float)
print(f"wrote {len(runs)} runs; R conceded {summary['rConcede']}; reg hist beta={reg['historical']['beta']:.2f} p={reg['historical']['p']:.4f}")
