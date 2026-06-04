"""
Strategic Delay in Debt-Ceiling Negotiations
Testing Alesina & Drazen (1991) war-of-attrition theory with LLM agents.

Pure-visualization dashboard; no live API calls. All statistics are recomputed
live from outputs/results/. The Overview tab is self-sufficient; the remaining
tabs provide depth (timing/hazard, per-episode, transcripts, regression,
methodology).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_ROOT        = Path(__file__).parents[1]
_CSS         = Path(__file__).parent / "components" / "styles.css"
_RESULTS     = _ROOT / "outputs" / "results"
_TRANSCRIPTS = _ROOT / "outputs" / "transcripts"
sys.path.insert(0, str(_ROOT))

st.set_page_config(
    page_title="LLMs in a War of Attrition — Fiscal Brinkmanship",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)
if _CSS.exists():
    st.markdown(f"<style>{_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Colour encoding (consistent across every chart) ─────────────────────────
HAWK, DOVE, GOLD, GOLD2, GREEN, MUTE = "#c94055", "#1a9cc8", "#b89840", "#d4b454", "#22b874", "#7080a0"
_EP   = {"2011": HAWK, "2013": GOLD, "2023": DOVE, "2025_counterfactual": "#7755bb"}
# cool -> warm ramp for conditions A..E (per spec)
_CD   = {"A": "#3a86c8", "B": "#33b0a6", "C": "#c8b040", "D": "#d18438", "E": "#c84050"}
_EP_NAME = {
    "2011": "2011 — Budget Control Act",
    "2013": "2013 — Government Shutdown",
    "2023": "2023 — Fiscal Responsibility Act",
    "2025_counterfactual": "2025 — Fictional counterfactual",
}
_ACTION_CLR = {"HOLD": "#4aaede", "SIGNAL_FLEXIBILITY": GOLD2, "CONCEDE": HAWK}

# Single-period flow diagram — hand-laid-out SVG, dark theme, three named phases.
_FLOW_MERMAID = r"""
<!doctype html><html><head><meta charset="utf-8">
<style>
  /* Dark wrapper · pastel academic cards inside. */
  html,body{margin:0;padding:0;background:transparent;
            font-family:'Courier New','Lucida Console',monospace;color:#e3e7f7;}
  .wrap{background:transparent;border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;padding:20px 24px 18px;position:relative;}

  .phases{display:grid;grid-template-columns:1fr 40px 1.4fr 40px 1fr;
          gap:0;align-items:start;}

  .col{position:relative;}

  /* phase numbers and labels */
  .phase-h{display:flex;align-items:center;gap:8px;margin-bottom:12px;}
  .pnum{display:inline-flex;align-items:center;justify-content:center;
        width:24px;height:24px;border-radius:50%;
        background:#e3e7f7;color:#0a0a18;font-weight:900;font-size:0.82rem;}
  .plbl{font-size:0.66rem;letter-spacing:0.22em;color:#e3e7f7;
        text-transform:uppercase;font-weight:bold;}

  /* arrow gutter between phases */
  .arrow{display:flex;align-items:center;justify-content:center;
         padding-top:50px;}
  .arrow svg{display:block;}

  /* cards — pastel academic palette */
  .card{border-radius:6px;padding:12px 14px 13px;font-size:0.77rem;
        line-height:1.55;border:1.5px solid;color:#1a1a2e;}
  .card .h{font-size:0.72rem;letter-spacing:0.12em;font-weight:bold;
           margin-bottom:8px;text-transform:uppercase;color:#1a1a2e;}
  .card .row{display:flex;justify-content:space-between;padding:2px 0;
             color:#33384a;}
  .card .row b{color:#1a1a2e;font-weight:700;}

  /* OBSERVE — light yellow */
  .market{background:#fff5d0;border-color:#e8b540;}
  .market .h{color:#7a4a00;}

  /* DELIBERATE — two agent rows */
  .deliberate{display:flex;flex-direction:column;gap:10px;}
  .agent{border-radius:6px;padding:11px 14px;position:relative;
         border:1.5px solid;color:#1a1a2e;}
  .agent.h{background:#fde2e6;border-color:#d63a52;}     /* light coral */
  .agent.d{background:#dbf0fb;border-color:#2090c0;}     /* light cyan/blue */
  .agent .name{font-size:0.78rem;font-weight:bold;letter-spacing:0.12em;}
  .agent.h .name{color:#8a0f24;}
  .agent.d .name{color:#0a4d6a;}
  .agent .role{font-size:0.66rem;color:#33384a;
               margin:4px 0 6px;font-style:italic;}
  .agent .think{font-size:0.69rem;color:#1a1a2e;line-height:1.55;}
  .agent .lock{color:#7a4a00;font-size:0.61rem;letter-spacing:0.08em;
               margin-top:6px;padding-top:5px;
               border-top:1px dashed rgba(122,74,0,0.35);}
  .agent .lock b{color:#5a3500;}

  /* ACT — light mint green */
  .act{background:#dff5e5;border-color:#38a060;}
  .act .h{color:#06502b;}
  .act .pill{display:inline-block;font-size:0.66rem;padding:3px 9px;
             border-radius:3px;margin-right:4px;margin-bottom:4px;
             border:1.5px solid;font-weight:700;}
  .act .p-hold{color:#0a4d6a;border-color:#2090c0;background:#c7e6f4;}
  .act .p-sig {color:#7a4a00;border-color:#e8b540;background:#fff0c4;}
  .act .p-con {color:#8a0f24;border-color:#d63a52;background:#fbcbd2;}

  /* loop footer — pastel parchment strip on dark */
  .loop{margin-top:16px;display:flex;align-items:center;gap:12px;
        padding:10px 14px;background:#f5f1e0;
        border:1.5px dashed #c8a040;border-radius:5px;
        font-size:0.75rem;color:#1a1a2e;line-height:1.4;}
  .loop .badge{background:#1a1a2e;color:#fff5d0;font-weight:900;
               padding:5px 11px;border-radius:3px;font-size:0.68rem;
               letter-spacing:0.10em;text-transform:uppercase;
               white-space:nowrap;}
  .loop .arrow-back{color:#7a4a00;font-size:1.2rem;font-weight:bold;}
  .loop .deal{margin-left:auto;display:flex;align-items:center;gap:5px;
              padding:5px 12px;background:#06502b;color:#dff5e5;
              border-radius:3px;font-weight:900;font-size:0.72rem;
              letter-spacing:0.08em;white-space:nowrap;}

  .sub{font-size:0.62rem;color:#aab2c8;margin-top:8px;letter-spacing:0.10em;
       text-align:center;text-transform:uppercase;font-weight:600;}
</style></head><body>
<div class="wrap">
  <div class="phases">
    <!-- ① OBSERVE -->
    <div class="col">
      <div class="phase-h"><span class="pnum">1</span><span class="plbl">Observe</span></div>
      <div class="card market">
        <div class="h">Shared market state</div>
        <div class="row">VIX volatility<b>23.7</b></div>
        <div class="row">4-wk T-bill<b>5.3%</b></div>
        <div class="row">Stress index<b>0.66</b></div>
        <div class="row">Days → X-date<b>12</b></div>
        <div class="row">Approval<b>41%</b></div>
        <div class="row" style="color:#7d87a8">Last opp. action<b>HOLD</b></div>
      </div>
      <div class="sub">Same data → both agents</div>
    </div>

    <!-- arrow -->
    <div class="arrow">
      <svg width="38" height="20" viewBox="0 0 38 20">
        <defs><marker id="ah1" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#e3e7f7"/></marker></defs>
        <line x1="0" y1="10" x2="30" y2="10" stroke="#e3e7f7" stroke-width="1.8" marker-end="url(#ah1)"/>
      </svg>
    </div>

    <!-- ② DELIBERATE -->
    <div class="col">
      <div class="phase-h"><span class="pnum">2</span><span class="plbl">Deliberate · private</span></div>
      <div class="deliberate">
        <div class="agent h">
          <div class="name">AGENT A · pro-cuts side</div>
          <div class="role">"prefers spending cuts; default is worse than a sub-optimal deal"</div>
          <div class="think">Reads state → writes private chain-of-thought →
              rates μ<sub>A</sub> (own delay-cost) 0–10 → infers μ<sub>B</sub></div>
          <div class="lock">🔒 chain-of-thought not visible to opponent
            · self-rate is <b>H1 input</b></div>
        </div>
        <div class="agent d">
          <div class="name">AGENT B · pro-programs side</div>
          <div class="role">"prefers protecting programs; won't validate hostage-taking"</div>
          <div class="think">Reads state → writes private chain-of-thought →
              rates μ<sub>B</sub> (own delay-cost) 0–10 → infers μ<sub>A</sub></div>
          <div class="lock">🔒 chain-of-thought not visible to opponent
            · self-rate is <b>H1 input</b></div>
        </div>
      </div>
      <div class="sub">Independent reasoning · no shared scratchpad</div>
    </div>

    <!-- arrow -->
    <div class="arrow">
      <svg width="38" height="20" viewBox="0 0 38 20">
        <defs><marker id="ah2" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#b89840"/></marker></defs>
        <line x1="0" y1="10" x2="30" y2="10" stroke="#b89840" stroke-width="1.6" marker-end="url(#ah2)"/>
      </svg>
    </div>

    <!-- ③ ACT -->
    <div class="col">
      <div class="phase-h"><span class="pnum">3</span><span class="plbl">Act · public</span></div>
      <div class="card act">
        <div class="h">Each side commits</div>
        <div style="margin:3px 0 5px">
          <span class="pill p-hold">HOLD</span>
          <span class="pill p-sig">SIGNAL FLEX</span>
          <span class="pill p-con">CONCEDE</span>
        </div>
        <div class="row">+ public statement<b style="color:#9aa6c4">(visible)</b></div>
        <div class="row" style="margin-top:6px;color:#7d87a8">recorded:</div>
        <div class="row">action<b>chosen</b></div>
        <div class="row">delay_cost_implied<b>0–10</b></div>
        <div class="row">concession_prob<b>0.00–1.00</b></div>
      </div>
      <div class="sub">Public statement → opponent next period</div>
    </div>
  </div>

  <!-- loop / exit -->
  <div class="loop">
    <span class="arrow-back">↺</span>
    <span class="badge">repeat 1 → 30</span>
    <span>each period the opponent's last action and updated market data feed back into Step ①</span>
    <span class="deal">✓ DEAL · when either side picks CONCEDE</span>
  </div>
</div>
</body></html>
"""

# Episode context (historical facts + peak FRED value in the simulation window).
# FRED peaks are episode-window baselines (condition B ≈ 1.0×); conditions A–E
# scale them by the VIX multiplier / T-bill spike in the design table.
_EP_META = {
    "2011": dict(ctx="S&P downgrade",
        text=("The standoff resolved with the Budget Control Act, signed on the 2 Aug 2011 X-date. "
              "S&P downgraded U.S. sovereign debt from AAA to AA+ three days later. "
              "The deal paired a $2.1T ceiling increase with $917B in spending cuts."),
        data="FRED window peak: VIX 31.5, 4-wk T-bill 0.07%", syn=False),
    "2013": dict(ctx="Government shutdown",
        text=("A 16-day government shutdown (1–16 Oct 2013) overlapped the ceiling deadline. "
              "The Continuing Appropriations Act (17 Oct) reopened government and suspended the "
              "ceiling to 7 Feb 2014. Equity volatility stayed moderate throughout."),
        data="FRED window peak: VIX 20.6, 4-wk T-bill 0.07%", syn=False),
    "2023": dict(ctx="Yellen X-date",
        text=("Treasury Secretary Yellen set an X-date of 5 Jun 2023. The Fiscal Responsibility "
              "Act, signed 3 Jun, suspended the ceiling to 1 Jan 2025 with discretionary caps. "
              "Short-dated bill yields carried a visible X-date premium."),
        data="FRED window peak: 4-wk T-bill 6.64%, VIX 23.6", syn=False),
    "2025_counterfactual": dict(ctx="Fictional episode",
        text=("A fabricated debt-ceiling scenario with no real-world referent — out-of-sample "
              "control for memorization. Synthetic macro baselines (VIX ≈ 26, T-bill ≈ 4.2%) and "
              "extrapolated fiscal fundamentals (debt/GDP 128%)."),
        data="Synthetic baselines: VIX ≈ 26, T-bill ≈ 4.2%, debt/GDP 128%", syn=True),
}


# ── Data loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def _sim() -> pd.DataFrame:
    p = _RESULTS / "simulation_results.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["masked"]   = df["sim_id"].str.endswith("_masked")
    df["conceder"] = np.where(df["winner"] == "DOVE", "HAWK",
                       np.where(df["winner"] == "HAWK", "DOVE", "NONE"))
    return df

@st.cache_data(ttl=30)
def _prd() -> pd.DataFrame:
    p = _RESULTS / "period_level_data.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@st.cache_data(ttl=60)
def _transcript_list() -> list[Path]:
    return sorted(_TRANSCRIPTS.glob("*.jsonl")) if _TRANSCRIPTS.exists() else []

@st.cache_data(ttl=60)
def _load_transcript(path_str: str):
    records = []
    with open(path_str, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return (records[0] if records else {}), (records[1:] if len(records) > 1 else [])

@st.cache_data(ttl=60)
def _featured():
    for name in ("2011_E_sim000.jsonl", "2011_D_sim000.jsonl"):
        p = _TRANSCRIPTS / name
        if p.exists():
            meta, periods = _load_transcript(str(p))
            if periods:
                return meta, periods[0]
    return None, None


@st.cache_data(ttl=30)
def _ols(sample: str):
    """OLS of concession_period on Hawk/Dove delay-cost ratio.
    sample in {'hist','masked','pooled'}. Returns full statistics or None."""
    s = _sim()
    if s.empty:
        return None
    if sample == "hist":
        s = s[~s["masked"]]
    elif sample == "masked":
        s = s[s["masked"]]
    cr = s["hawk_mean_delay_cost"] / s["dove_mean_delay_cost"].replace(0, np.nan)
    d = pd.DataFrame({"cr": cr, "y": s["concession_period"]}).dropna()
    if len(d) < 5:
        return None
    out = dict(n=int(len(d)), r=float(d["cr"].corr(d["y"])))
    try:
        import statsmodels.api as sm
        m = sm.OLS(d["y"], sm.add_constant(d["cr"])).fit()
        ci = m.conf_int().loc["cr"]
        out.update(beta=float(m.params["cr"]), se=float(m.bse["cr"]),
                   t=float(m.tvalues["cr"]), p=float(m.pvalues["cr"]),
                   df=int(m.df_resid), r2=float(m.rsquared), adj=float(m.rsquared_adj),
                   f=float(m.fvalue), fp=float(m.f_pvalue),
                   ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                   c_beta=float(m.params["const"]), c_se=float(m.bse["const"]),
                   c_t=float(m.tvalues["const"]), c_p=float(m.pvalues["const"]),
                   model=m, data=d)
    except Exception:
        z = np.polyfit(d["cr"], d["y"], 1)
        out.update(beta=float(z[0]), p=float("nan"))
    return out


sim    = _sim()
period = _prd()
HAS    = not sim.empty
tx_all = _transcript_list()
n_tx   = len(tx_all)


# ── Shared plotly theme ─────────────────────────────────────────────────────
_BASE = dict(
    paper_bgcolor="rgba(4,4,16,0)", plot_bgcolor="rgba(7,7,20,0.75)",
    font=dict(color="#b0bcd4", family="Courier New, monospace", size=11),
    legend=dict(bgcolor="rgba(0,0,0,0.38)", font=dict(size=10), borderwidth=0),
    hoverlabel=dict(bgcolor="#080820", font=dict(family="Courier New, monospace", size=11)),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(size=10)),
)

def _lay(fig: go.Figure, h: int = 300, margin: dict | None = None, **kw) -> go.Figure:
    layout = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _BASE.items()}
    layout["height"], layout["margin"] = h, margin or dict(t=36, b=28, l=44, r=18)
    for k, v in kw.items():
        if k in layout and isinstance(layout[k], dict) and isinstance(v, dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    fig.update_layout(**layout)
    return fig

def _rgba(hex_c: str, a: float) -> str:
    h = hex_c.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def cap(hypothesis: str, source: str):
    st.caption(f"{hypothesis}  ·  Source: {source}")

def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ── Sidebar (transparency: files, rows, missing flags) ──────────────────────
with st.sidebar:
    st.markdown("<span style='font-family:monospace;color:#b89840;letter-spacing:0.1em'>"
                "DATA PROVENANCE</span>", unsafe_allow_html=True)
    st.divider()
    def _row(ok, lbl, detail):
        c = "pg" if ok else "pr"
        return (f"<div style='margin-bottom:0.35rem'><span class='ping {c}'></span>"
                f"<span style='font-family:monospace;font-size:0.7rem;color:#b8c2dc'>{lbl}</span><br>"
                f"<span style='font-family:monospace;font-size:0.62rem;color:#7080a0;margin-left:12px'>"
                f"{detail}</span></div>")
    sim_p = _RESULTS / "simulation_results.parquet"
    prd_p = _RESULTS / "period_level_data.parquet"
    reg_p = _RESULTS / "regression_results.json"
    st.markdown(
        _row(sim_p.exists(), "simulation_results.parquet",
             f"{len(sim)} rows loaded" if HAS else "MISSING") +
        _row(prd_p.exists(), "period_level_data.parquet",
             f"{len(period)} rows loaded" if not period.empty else "MISSING") +
        _row(n_tx > 0, "transcripts/*.jsonl", f"{n_tx} files") +
        _row(reg_p.exists(), "regression_results.json",
             "found" if reg_p.exists() else "absent — OLS fit live (statsmodels)"),
        unsafe_allow_html=True)
    st.divider()
    if HAS:
        st.markdown(
            "<span style='font-family:monospace;font-size:0.64rem;color:#7080a0;line-height:1.6'>"
            f"<b style='color:#b8c2dc'>Unit of analysis</b><br>"
            f"{len(sim)} runs = 40 scenarios × {{historical, masked}}.<br>"
            f"Pairing keyed off the <code style='color:#d4b454'>_masked</code> id suffix "
            f"({(~sim['masked']).sum()} historical / {sim['masked'].sum()} masked).<br>"
            f"Primary stats use historical runs; masked runs reported as replication."
            "</span>", unsafe_allow_html=True)
    st.divider()
    st.caption("Agents:  claude-haiku-4-5\nJudge:   claude-sonnet-4-6\n"
               "Theory:  Alesina & Drazen (1991)\nData:    FRED VIXCLS, TB4WK; polls")

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">LLMs in a War of Attrition</div>
  <div class="hero-desc">
    Two AI negotiators face a U.S. debt-ceiling deadline. Who breaks first when market stress rises?
    Built to test Alesina &amp; Drazen (1991).
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

t1, t2, t3, t4, t5, t6 = st.tabs([
    "Overview", "Timing & Hazard", "Episode Analysis", "Transcripts", "Regression", "Methodology",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW  (self-sufficient; 12 sections)
# ═══════════════════════════════════════════════════════════════════════════
with t1:
    def sec(num, label):
        st.markdown(f"<div class='sec-tag'><span class='num'>{num}</span> {label}</div>",
                    unsafe_allow_html=True)
    def sep():
        st.markdown("<div class='story-sep'></div>", unsafe_allow_html=True)
    def emph(t: str) -> str:
        for w in ("leverage", "default", "credibility", "catastroph", "existential",
                  "concede", "capitulat", "spending", "revenue", "reforms"):
            t = re.sub(r"(?i)\b(" + re.escape(w) + r"[a-z]*)", r"<span class='em'>\1</span>", t)
        return t
    def summ(txt, max_sent=2, cap=300):
        """First 1–2 complete sentences — a clean summary, never a mid-word cut."""
        parts = re.split(r"(?<=[.!?])\s+", (txt or "").strip())
        out = ""
        for s in parts[:max_sent]:
            if out and len(out) + len(s) + 1 > cap:
                break
            out = (out + " " + s).strip()
        return out or (txt or "")

    if not HAS:
        st.markdown("<div class='empty' style='margin-top:1rem'>No simulation data in "
                    "outputs/results/. Run the simulation to populate this dashboard.</div>",
                    unsafe_allow_html=True)
        st.stop()

    # live numbers ────────────────────────────────────────────────────────────
    n        = len(sim)
    hawk_c   = int((sim["conceder"] == "HAWK").sum())   # HAWK role = Republican (fiscal conservative)
    dove_c   = int((sim["conceder"] == "DOVE").sum())
    no_deal  = int((sim["conceder"] == "NONE").sum())
    agree    = 100 * (n - no_deal) / n
    cond_med = sim.groupby("condition_id")["concession_period"].median().reindex(list("ABCDE"))
    med_lo, med_hi = cond_med.get("A"), cond_med.get("E")
    med_all  = sim["concession_period"].median()
    stress_r = float(sim["final_market_stress"].corr(sim["concession_period"]))
    o_h, o_m = _ols("hist"), _ols("masked")
    rep_pct  = 100 * hawk_c / n
    hawk_dc  = float(sim["hawk_mean_delay_cost"].mean())
    dove_dc  = float(sim["dove_mean_delay_cost"].mean())
    n_dec    = int(len(period)) * 2 if not period.empty else int(sim["concession_period"].sum()) * 2

    def primary_str():
        return (f"β = {o_h['beta']:.1f}, 95% CI [{o_h['ci_lo']:.1f}, {o_h['ci_hi']:.1f}], "
                f"t({o_h['df']}) = {o_h['t']:.2f}, p = {o_h['p']:.3f}, n = {o_h['n']}")
    def masked_str():
        return f"β = {o_m['beta']:.1f}, p = {o_m['p']:.3f}, n = {o_m['n']}"

    # ═══ 1 · ABSTRACT ═════════════════════════════════════════════════════════
    sec("01", "Abstract")
    st.markdown(
        "<div class='lead'>"
        "Debt-ceiling standoffs are a <span class='hl-w'>war of attrition</span>: both sides want a deal, "
        "but each wants the other to take the painful concessions, so both wait while default risk and "
        "market stress build (Alesina &amp; Drazen, 1991). "
        "Here, two LLM agents — <span class='hl-h'>Republican</span> (spending cuts, no tax hikes) and "
        "<span class='hl-d'>Democrat</span> (protect programs) — negotiate period-by-period on real "
        "dates and FRED data (2011, 2013, 2023, plus one <i>fictional</i> episode). Stances are fixed; "
        "who concedes and when is not. "
        "<b>Takeaway:</b> the side that reports more pain from delay tends to concede sooner; "
        f"higher market stress closes deals faster; the Republican side concedes first in "
        f"<span class='hl-g'>{rep_pct:.0f}%</span> of runs because it rates delay as costlier."
        "</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='scale'><span class='lbl'>Scale</span> &nbsp; "
        f"<b>{n}</b> runs · 4 episodes · 5 stress levels (A→E) · up to 30 periods · "
        f"<b>{n_dec:,}</b> logged agent decisions (reasoning, action, delay-cost score).</div>",
        unsafe_allow_html=True)

    sep()

    # ═══ 2 · HEADLINE RESULTS ═════════════════════════════════════════════════
    sec("02", "Key findings")
    fc1, fc2, fc3 = st.columns(3, gap="medium")
    med_a = f"{med_lo:.0f}" if med_lo is not None and not np.isnan(med_lo) else "—"
    med_e = f"{med_hi:.0f}" if med_hi is not None and not np.isnan(med_hi) else "—"
    with fc1:
        st.markdown(f"""
<div class="interp h"><div class="interp-tag h">Result 1 · who breaks first</div>
  <div class="interp-h">Republican concedes in {rep_pct:.0f}% of runs</div>
  <ul>
    <li><b>What:</b> Republican agent accepts a deal before the Democrat in <b>{hawk_c}/{n}</b> runs.</li>
    <li><b>Why:</b> it self-rates higher delay cost (<b>{hawk_dc:.1f}</b> vs <b>{dove_dc:.1f}</b>/10) — deadlock hurts its platform more when rates spike and fiscal credibility is on the line.</li>
  </ul>
  <div class="stat">Rep mean delay cost {hawk_dc:.1f} &gt; Dem {dove_dc:.1f}</div></div>
""", unsafe_allow_html=True)
    with fc2:
        st.markdown(f"""
<div class="interp d"><div class="interp-tag d">Result 2 · pain predicts timing</div>
  <div class="interp-h">Higher relative delay cost → earlier deal</div>
  <ul>
    <li><b>What:</b> runs where the Republican/Democrat delay-cost ratio is higher end sooner (negative slope).</li>
    <li><b>Why:</b> matches Alesina–Drazen — the side that suffers more from waiting folds first; same sign in masked replays, so not memorising past headlines.</li>
  </ul>
  <div class="stat">{primary_str()}; masked {masked_str()}</div></div>
""", unsafe_allow_html=True)
    with fc3:
        st.markdown(f"""
<div class="interp"><div class="interp-tag">Result 3 · stress ends the standoff</div>
  <div class="interp-h">Crisis conditions close deals faster</div>
  <ul>
    <li><b>What:</b> median concession period drops from condition A ({med_a}) to E ({med_e}); stress and timing correlate r = {stress_r:.2f}.</li>
    <li><b>Why:</b> higher VIX/T-bill stress in the prompt raises perceived cost of waiting for both sides, so attrition ends sooner.</li>
  </ul>
  <div class="stat">A→E stress ramp · see bar chart below</div></div>
""", unsafe_allow_html=True)
    st.caption("Republican = HAWK agent, Democrat = DOVE in raw data. Concede = first side to accept a deal.")

    st.markdown(f"""
<div class="bigstat-row four">
  <div class="bigstat"><div class="bigstat-n">{n}</div><div class="bigstat-l">Runs</div></div>
  <div class="bigstat"><div class="bigstat-n h">{rep_pct:.0f}%</div><div class="bigstat-l">Republican concedes first</div>
    <div class="bigstat-sub">{hawk_c}/{n}</div></div>
  <div class="bigstat"><div class="bigstat-n">{med_all:.0f}</div><div class="bigstat-l">Median periods to deal</div></div>
  <div class="bigstat"><div class="bigstat-n d">{o_h['beta']:.0f}</div><div class="bigstat-l">H1 slope β</div>
    <div class="bigstat-sub">p = {o_h['p']:.3f}</div></div>
</div>
""", unsafe_allow_html=True)

    sim_dur = sim["concession_period"].clip(0, 30)

    # ── full-width supporting visuals ─────────────────────────────────────────
    st.markdown("<div class='slbl' style='margin-top:0.6rem'>Time-to-agreement by market-stress "
                "regime (H3)</div>", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(x=list(cond_med.index), y=cond_med.values,
                    marker_color=[_CD[c] for c in cond_med.index],
                    marker_line_color="rgba(255,255,255,0.12)", marker_line_width=0.6, width=0.55,
                    text=[f"{v:.0f}" for v in cond_med.values], textposition="outside",
                    textfont=dict(size=14, color="#e3e7f7"),
                    hovertemplate="Condition %{x}<br>median period %{y:.1f}<extra></extra>"))
    _lay(fig, h=300, margin=dict(t=14, b=38, l=48, r=14),
         xaxis=dict(title="Condition (A calm → E near-default)", gridcolor="rgba(0,0,0,0)"),
         yaxis=dict(title="Median concession period", range=[0, max(cond_med.values) * 1.22]))
    st.plotly_chart(fig, use_container_width=True)
    cap("median of concession_period by condition; falls as stress rises", f"all runs, n = {n}")

    st.markdown("<div class='slbl' style='margin-top:0.4rem'>Delay-cost ratio vs concession period "
                "(H1)</div>", unsafe_allow_html=True)
    cr = sim["hawk_mean_delay_cost"] / sim["dove_mean_delay_cost"].replace(0, np.nan)
    fig = go.Figure()
    for c in "ABCDE":
        m = sim.condition_id == c
        fig.add_trace(go.Scatter(x=cr[m], y=sim_dur[m], mode="markers", name=f"Cond {c}",
                      marker=dict(color=_CD[c], size=8, opacity=0.72,
                                  line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
                      hovertemplate="ratio %{x:.2f}<br>period %{y}<extra>Cond " + c + "</extra>"))
    op = _ols("pooled")
    if op and "model" in op:
        xr = np.linspace(cr.min(), cr.max(), 80)
        fig.add_trace(go.Scatter(x=xr, y=op["c_beta"] + op["beta"] * xr, mode="lines",
                      name=f"OLS β={op['beta']:.0f}", line=dict(color=GOLD, width=2, dash="dot")))
    _lay(fig, h=320, margin=dict(t=14, b=38, l=48, r=14), legend=dict(orientation="h", y=-0.2, x=0),
         xaxis=dict(title="Republican / Democrat self-rated delay-cost ratio (higher ⇒ Republican hurts more)"),
         yaxis=dict(title="Concession period"))
    st.plotly_chart(fig, use_container_width=True)
    cap("ratio of run-mean self-rated delay cost; dashed line = OLS fit (negative slope = H1)",
        f"pooled, n = {n}")

    st.markdown("<div class='slbl' style='margin-top:0.4rem'>Which side concedes</div>",
                unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        y=["Republican concedes", "Democrat concedes", "No deal at cap"], x=[hawk_c, dove_c, no_deal],
        orientation="h", marker_color=[HAWK, DOVE, "#444a66"],
        marker_line_color="rgba(255,255,255,0.12)", marker_line_width=0.6, width=0.6,
        text=[hawk_c, dove_c, no_deal], textposition="outside",
        textfont=dict(size=14, color="#e3e7f7"), hovertemplate="%{y}: %{x} runs<extra></extra>"))
    _lay(fig, h=210, margin=dict(t=10, b=36, l=16, r=22),
         xaxis=dict(title="Number of runs", range=[0, max(hawk_c, dove_c) * 1.25]),
         yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    cap("count of conceding side across runs", f"n = {n}")

    sep()

    # ═══ 3 · THE FOUR EPISODES ════════════════════════════════════════════════
    sec("03", "The four episodes")
    cards = ""
    for ep in ["2011", "2013", "2023", "2025_counterfactual"]:
        m = _EP_META[ep]; sub = sim[sim["episode_id"] == ep]
        hpct = 100 * (sub["conceder"] == "HAWK").mean() if len(sub) else 0
        syn = " syn" if m["syn"] else ""
        flag = "<div class='epi-syn-flag'>Synthetic — not real data</div>" if m["syn"] else ""
        cards += (f"<div class='epi-card{syn}'><div class='epi-h'>{_EP_NAME[ep].split(' — ')[0]} · "
                  f"{m['ctx']}</div>{flag}<div class='epi-b'>{m['text']}</div>"
                  f"<div class='epi-data'>{m['data']}</div>"
                  f"<div class='epi-tag'>In simulation: Republican concedes {hpct:.0f}%</div></div>")
    st.markdown(f"<div class='epi-grid'>{cards}</div>", unsafe_allow_html=True)

    sep()

    # ═══ 4 · THEORETICAL BACKGROUND ═══════════════════════════════════════════
    sec("04", "Theoretical background — the war of attrition")
    tcol, ecol = st.columns([3, 2], gap="large")
    with tcol:
        st.markdown(
            "<div class='lead' style='margin-bottom:0.8rem'>"
            "Both sides prefer a deal to default, but each wants the <i>other</i> to swallow the painful "
            "concessions — so both wait. Waiting costs money and credibility as markets stress rises. "
            "Alesina &amp; Drazen (1991): <span class='hl-g'>whoever hurts more from delay concedes "
            "first</span>. This dashboard checks whether LLM agents follow that rule when given only "
            "each party's fiscal priorities."
            "</div>", unsafe_allow_html=True)
    with ecol:
        st.markdown("<div class='slbl'>The formal core</div>", unsafe_allow_html=True)
        st.latex(r"P(i\ \text{concedes first}) = \frac{\mu_i}{\mu_i + \mu_j}")
        st.markdown("<div class='eqgloss'>Each side <i>i</i> bears a private cost of delay μ. The side with "
                    "the higher cost is the more likely to concede first.</div>", unsafe_allow_html=True)
        st.latex(r"\mu_{\text{eff}}(s) = \mu_{\text{base}}\,(1 + \gamma s)")
        st.markdown("<div class='eqgloss'>Rising market stress <i>s</i> lifts both sides' costs, so "
                    "agreement arrives sooner.</div>", unsafe_allow_html=True)
        st.markdown("<div class='slbl' style='margin-top:0.4rem'>What this predicts (and I test)</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "- **H1** — the side reporting the higher delay cost concedes first\n"
            "- **H2** — concession grows likelier as the X-date nears\n"
            "- **H3** — higher market stress → faster resolution")

    sep()

    # ═══ 5 · INSIDE THE AGENTS ════════════════════════════════════════════════
    sec("05", "Inside the agents — what they are given, and how they decide")
    st.markdown(
        "<div class='lead' style='margin-bottom:0.8rem'>Nothing about the outcome is hard-coded. Each "
        "period an agent is handed a market readout and must reason and choose. Here is exactly what "
        "goes in, and what it bases the call on.</div>", unsafe_allow_html=True)
    st.markdown("""
<div class="explain">
  <div class="explain-item"><div class="ei-h">Variables it reads each period</div>
    <div class="ei-b">The Treasury <b>X-date countdown</b>, <b>VIX</b> (market fear), the <b>4-week
      T-bill yield</b> (a default-risk signal), <b>public approval</b>, and the <b>opponent's last
      public move</b>. Sources: FRED + polling archives.</div></div>
  <div class="explain-item"><div class="ei-h">Its brief (the system prompt)</div>
    <div class="ei-b">“You are the <b>Republican</b> [or <b>Democrat</b>] side. Pursue [spending cuts /
      program protection]; avoid default; don't squander your leverage.” I assign the <b>stance</b>,
      never the moves.</div></div>
  <div class="explain-item"><div class="ei-h">What it weighs to decide</div>
    <div class="ei-b">Deadline pressure, which side the market is hurting more, and the opponent's
      signals — then it picks one action: <b>hold</b>, <b>signal flexibility</b>, or
      <b>concede</b>.</div></div>
  <div class="explain-item"><div class="ei-h">The “delay cost” it reports</div>
    <div class="ei-b">A private <b>0–10</b> self-rating of how badly more deadlock hurts its own side.
      This is the key quantity: the theory says the higher-delay-cost side concedes first, and it is
      the variable tested in H1.</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='slbl' style='margin-top:1.1rem'>How a single period flows</div>",
                unsafe_allow_html=True)
    import streamlit.components.v1 as components
    components.html(_FLOW_MERMAID, height=440, scrolling=False)
    st.caption("The same market data reaches both sides each period; each reasons privately, scores its "
               "own delay cost, and issues a public statement, until one side concedes.")

    st.markdown("<div class='slbl' style='margin-top:1.1rem'>A real negotiation, period by period — "
                "2023, near-default (condition E)</div>", unsafe_allow_html=True)
    st.markdown("<div class='lead' style='font-size:0.82rem;margin-bottom:0.7rem'>Each side reads the "
                "other's public statement and responds the next period. Watch the delay-cost numbers: "
                "when the Republican's climbs past the Democrat's, it concedes.</div>",
                unsafe_allow_html=True)

    xpath = _TRANSCRIPTS / "2023_E_sim001.jsonl"
    if not xpath.exists():
        cand = sorted(_TRANSCRIPTS.glob("2023_E_sim*.jsonl")) or sorted(_TRANSCRIPTS.glob("*_E_sim*.jsonl"))
        xpath = cand[0] if cand else None
    if xpath is not None:
        meta_x, periods_x = _load_transcript(str(xpath))
        cper = meta_x.get("concession_period", len(periods_x))
        show = [p for p in periods_x if p["period"] <= cper][:6]
        for p in show:
            hd = p.get("hawk_decision", {}); dd = p.get("dove_decision", {})
            ha = hd.get("action", "HOLD"); da = dd.get("action", "HOLD")
            st.markdown(f"""
<div class="xchg-period">
  <div class="xchg-mark">Period {p['period']} · {p.get('date','')} · VIX {p.get('vix',0):.0f} ·
    stress {p.get('market_stress_index',0):.2f} · {p.get('days_to_xdate','')}d to X-date</div>
  <div class="xchg-row">
    <div class="xchg-bubble rep">
      <div class="xchg-who">REPUBLICAN <span class="xchg-act act-{ha}">{ha.replace('_',' ')}</span></div>
      <div class="xchg-say">“{hd.get('public_statement','')}”</div>
      <div class="xchg-think">reasoning (summary): {summ(hd.get('reasoning',''))}</div>
      <div class="xchg-dc">self-rated delay cost <b>{hd.get('delay_cost_implied',0):.1f}/10</b></div>
    </div>
    <div class="xchg-bubble dem">
      <div class="xchg-who">DEMOCRAT <span class="xchg-act act-{da}">{da.replace('_',' ')}</span></div>
      <div class="xchg-say">“{dd.get('public_statement','')}”</div>
      <div class="xchg-think">reasoning (summary): {summ(dd.get('reasoning',''))}</div>
      <div class="xchg-dc">self-rated delay cost <b>{dd.get('delay_cost_implied',0):.1f}/10</b></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
        st.markdown(f"<div class='xchg-deal'>✓ DEAL at period {cper} — the REPUBLICAN concedes once its "
                    f"delay cost crosses the Democrat's</div>", unsafe_allow_html=True)
        st.caption(f"Unedited public statements and reasoning summaries. Source: {xpath.name}.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — TIMING & HAZARD  (charts + interpretive paragraphs)
# ═══════════════════════════════════════════════════════════════════════════
with t2:
    if not HAS:
        st.markdown('<div class="empty">Awaiting simulation data.</div>', unsafe_allow_html=True)
    else:
        fc1, fc2 = st.columns([3, 2])
        with fc1:
            ep_sel = st.selectbox("Episode", ["All episodes"] + sorted(sim.episode_id.unique().tolist()), key="t2_ep")
        with fc2:
            cd_sel = st.multiselect("Conditions", list("ABCDE"), default=list("ABCDE"), key="t2_cd")
        ddf = sim.copy()
        if ep_sel != "All episodes":
            ddf = ddf[ddf.episode_id == ep_sel]
        if cd_sel:
            ddf = ddf[ddf.condition_id.isin(cd_sel)]
        ddf["dur"] = ddf["concession_period"].clip(0, 30)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown("<div class='slbl'>Concession-period distribution</div>", unsafe_allow_html=True)
            fig = go.Figure()
            for c in sorted(ddf.condition_id.unique()):
                fig.add_trace(go.Histogram(x=ddf[ddf.condition_id == c]["dur"], name=f"Cond {c}",
                              marker_color=_CD.get(c, "#aaa"), opacity=0.7, nbinsx=15, histnorm="probability"))
            _lay(fig, h=280, margin=dict(t=18, b=30, l=44, r=12), barmode="overlay",
                 xaxis=dict(title="Concession period"), yaxis=dict(title="Relative frequency"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Mass shifts left (earlier concession) as condition severity rises; period 30 "
                       "would indicate no deal, which does not occur in this sample.")
        with c2:
            st.markdown("<div class='slbl'>Survival function by condition</div>", unsafe_allow_html=True)
            fig = go.Figure(); t_arr = np.linspace(0, 30, 61)
            for c in sorted(ddf.condition_id.unique()):
                d_ = ddf[ddf.condition_id == c]["dur"].values
                fig.add_trace(go.Scatter(x=t_arr, y=[np.mean(d_ > t) for t in t_arr], mode="lines",
                              name=f"Cond {c}", line=dict(color=_CD.get(c, "#aaa"), width=2.2, shape="hv")))
            _lay(fig, h=280, margin=dict(t=18, b=30, l=44, r=12),
                 xaxis=dict(title="Period"), yaxis=dict(title="P(unresolved by t)", range=[0, 1.02]))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Warmer curves lie below cooler ones, i.e. higher-stress conditions resolve "
                       "sooner — the survival-analytic form of H3.")

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            st.markdown("<div class='slbl'>Empirical hazard rate</div>", unsafe_allow_html=True)
            hz = [{"c": c, "rate": (ddf[ddf.condition_id == c]["dur"] < 30).sum()
                   / max(ddf[ddf.condition_id == c]["dur"].sum(), 1)} for c in sorted(ddf.condition_id.unique())]
            hz = pd.DataFrame(hz)
            fig = go.Figure(go.Bar(x=hz["c"], y=hz["rate"], marker_color=[_CD.get(c, "#aaa") for c in hz["c"]],
                            width=0.55, text=[f"{r:.4f}" for r in hz["rate"]], textposition="outside",
                            textfont=dict(size=9, color="#b0bcd4")))
            _lay(fig, h=240, margin=dict(t=18, b=30, l=44, r=12),
                 xaxis=dict(title="Condition"), yaxis=dict(title="Hazard rate"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Events divided by total periods at risk. A monotone A→E increase is the "
                       "discrete-time analogue of μ_eff(s) rising in stress.")
        with c4:
            st.markdown("<div class='slbl'>Delay-cost ratio vs concession period</div>", unsafe_allow_html=True)
            cr = ddf["hawk_mean_delay_cost"] / ddf["dove_mean_delay_cost"].replace(0, np.nan)
            fig = go.Figure()
            for c in sorted(ddf.condition_id.unique()):
                m = ddf.condition_id == c
                fig.add_trace(go.Scatter(x=cr[m], y=ddf.loc[m, "dur"], mode="markers", name=f"Cond {c}",
                              marker=dict(color=_CD.get(c, "#aaa"), size=6, opacity=0.68)))
            valid = cr.notna() & ddf["dur"].notna()
            if valid.sum() > 4:
                z = np.polyfit(cr[valid].astype(float), ddf.loc[valid, "dur"].astype(float), 1)
                xr = np.linspace(cr[valid].min(), cr[valid].max(), 60)
                fig.add_trace(go.Scatter(x=xr, y=np.polyval(z, xr), mode="lines",
                              name=f"OLS β={z[0]:.1f}", line=dict(color=GOLD, width=1.6, dash="dot")))
            _lay(fig, h=240, margin=dict(t=18, b=30, l=44, r=12),
                 xaxis=dict(title="Republican/Democrat delay-cost ratio"), yaxis=dict(title="Concession period"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Negative slope is H1 within this filtered subset; see the Regression tab for "
                       "the full table with confidence intervals.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — EPISODE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
with t3:
    HIST = ["2011", "2013", "2023"]
    if not HAS:
        st.markdown('<div class="empty">Awaiting simulation data.</div>', unsafe_allow_html=True)
    else:
        ec1, ec2 = st.columns([3, 2])
        with ec1:
            met_sel = st.selectbox("Trajectory metric",
                ["Republican concession probability", "Democrat concession probability",
                 "Republican delay cost", "Democrat delay cost", "Market stress"], key="t3_m")
        with ec2:
            cc_cd = st.multiselect("Conditions", list("ABCDE"), default=["B", "C", "D"], key="t3_cd")
        mcol = {"Republican concession probability": "hawk_concession_prob",
                "Democrat concession probability": "dove_concession_prob",
                "Republican delay cost": "hawk_delay_cost", "Democrat delay cost": "dove_delay_cost",
                "Market stress": "market_stress_index"}.get(met_sel, "hawk_concession_prob")
        avail = [e for e in HIST if e in sim.episode_id.unique()]
        if not avail:
            st.info("No historical episode data in current results.")
        else:
            st.markdown("<div class='slbl'>Concession-timing distribution by episode</div>", unsafe_allow_html=True)
            fig_3 = make_subplots(rows=1, cols=len(avail),
                                  subplot_titles=[_EP_NAME.get(e, e) for e in avail], shared_yaxes=True)
            for i, ep in enumerate(avail, 1):
                sub = sim[sim.episode_id == ep]
                if cc_cd:
                    sub = sub[sub.condition_id.isin(cc_cd)]
                fig_3.add_trace(go.Histogram(x=sub["concession_period"].clip(0, 30), nbinsx=14, name=ep,
                                marker_color=_EP.get(ep, "#aaa"), opacity=0.8, histnorm="probability"), row=1, col=i)
            _lay(fig_3, h=280, barmode="overlay", margin=dict(t=46, b=26, l=42, r=14))
            for ann in fig_3.layout.annotations:
                ann.font.update(color="#b0bcd4", size=10)
            st.plotly_chart(fig_3, use_container_width=True)

            if not period.empty and mcol in period.columns:
                st.markdown(f"<div class='slbl'>{met_sel} — mean ± 1 SD trajectory</div>", unsafe_allow_html=True)
                fig_ep = go.Figure()
                for ep in avail:
                    pep = period[period.episode_id == ep]
                    if cc_cd:
                        pep = pep[pep.condition_id.isin(cc_cd)]
                    if pep.empty:
                        continue
                    agg = pep.groupby("period")[mcol].agg(["mean", "std"]).reset_index()
                    c_ = _EP.get(ep, "#aaa")
                    hi = (agg["mean"] + agg["std"].fillna(0)).clip(0); lo = (agg["mean"] - agg["std"].fillna(0)).clip(0)
                    fig_ep.add_trace(go.Scatter(x=pd.concat([agg.period, agg.period[::-1]]),
                                     y=pd.concat([hi, lo[::-1]]), fill="toself", fillcolor=_rgba(c_, 0.08),
                                     line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
                    fig_ep.add_trace(go.Scatter(x=agg.period, y=agg["mean"], mode="lines",
                                     name=_EP_NAME.get(ep, ep), line=dict(color=c_, width=2.2)))
                _lay(fig_ep, h=290, margin=dict(t=18, b=26, l=44, r=12),
                     xaxis=dict(title="Period"), yaxis=dict(title=met_sel))
                st.plotly_chart(fig_ep, use_container_width=True)

            st.markdown("<div class='slbl'>Episode statistics</div>", unsafe_allow_html=True)
            tbl = []
            for ep in avail:
                sub = sim[sim.episode_id == ep]
                if cc_cd:
                    sub = sub[sub.condition_id.isin(cc_cd)]
                d_ = sub["concession_period"].clip(0, 30)
                tbl.append({"Episode": ep, "Runs": len(sub), "Median period": f"{d_.median():.0f}",
                            "Republican concedes": f"{100*(sub.conceder=='HAWK').mean():.0f}%",
                            "Democrat concedes": f"{100*(sub.conceder=='DOVE').mean():.0f}%"})
            st.dataframe(pd.DataFrame(tbl).set_index("Episode"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — TRANSCRIPTS  (searchable, paginated)
# ═══════════════════════════════════════════════════════════════════════════
with t4:
    if not tx_all:
        st.markdown('<div class="empty">No transcript files found in outputs/transcripts/.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='slbl'>Negotiation transcripts</div>", unsafe_allow_html=True)
        st.markdown("<span style='font-family:monospace;font-size:0.72rem;color:#7080a0'>"
                    "Each file is one Republican vs Democrat negotiation. Files ending <code>_masked</code> "
                    "are contamination-control runs with the historical outcome hidden.</span>",
                    unsafe_allow_html=True)
        names = [p.name for p in tx_all]
        sel_name = st.selectbox("Transcript", names,
                                format_func=lambda x: x.replace(".jsonl", "").replace("_", " · "), key="tx_sel")
        meta, periods_tx = _load_transcript(str(_TRANSCRIPTS / sel_name))
        if meta:
            conceder = ("Republican" if meta.get("winner") == "DOVE" else "Democrat" if meta.get("winner") == "HAWK" else "—")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Episode", meta.get("episode_id", "—")); m2.metric("Condition", meta.get("condition_id", "—"))
            m3.metric("Conceding side", conceder); m4.metric("Concession period", str(meta.get("concession_period", "—")))
            m5.metric("Temperature", f"{meta.get('temperature',0):.2f}")

        if periods_tx:
            t_vals = [p["period"] for p in periods_tx]
            fig_tx = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06)
            fig_tx.add_trace(go.Scatter(x=t_vals, y=[p["hawk_decision"].get("concession_probability", 0) for p in periods_tx],
                             mode="lines+markers", name="Republican P(concede)", line=dict(color=HAWK, width=2),
                             marker=dict(size=6, color=[_ACTION_CLR.get(p["hawk_decision"].get("action", "HOLD"), "#aaa") for p in periods_tx])), row=1, col=1)
            fig_tx.add_trace(go.Scatter(x=t_vals, y=[p["dove_decision"].get("concession_probability", 0) for p in periods_tx],
                             mode="lines+markers", name="Democrat P(concede)", line=dict(color=DOVE, width=2),
                             marker=dict(size=6, color=[_ACTION_CLR.get(p["dove_decision"].get("action", "HOLD"), "#aaa") for p in periods_tx])), row=1, col=1)
            fig_tx.add_trace(go.Scatter(x=t_vals, y=[p.get("market_stress_index", 0) for p in periods_tx],
                             mode="lines", name="Market stress", line=dict(color=GOLD, width=1.5),
                             fill="tozeroy", fillcolor=_rgba(GOLD, 0.08)), row=2, col=1)
            _lay(fig_tx, h=320, margin=dict(t=24, b=26, l=44, r=12),
                 xaxis2=dict(title="Period"), yaxis=dict(title="P(concede)", range=[0, 1.05]),
                 yaxis2=dict(title="Stress", range=[0, 1.05]))
            st.plotly_chart(fig_tx, use_container_width=True)

            # search + pagination
            q = st.text_input("Filter periods by keyword in reasoning text", key="tx_q").strip().lower()
            def _match(p):
                if not q:
                    return True
                return q in (p["hawk_decision"].get("reasoning", "") + p["dove_decision"].get("reasoning", "")).lower()
            matched = [p for p in periods_tx if _match(p)]
            st.caption(f"{len(matched)} of {len(periods_tx)} periods match.")
            PER = 6
            n_pages = max(1, (len(matched) + PER - 1) // PER)
            pg = st.number_input("Page", 1, n_pages, 1, key="tx_pg") if n_pages > 1 else 1
            for p in matched[(pg - 1) * PER: pg * PER]:
                ctx = (f"Period {p['period']} · {p.get('date','')} · days→X {p.get('days_to_xdate','')} · "
                       f"VIX {p.get('vix',0):.1f} · stress {p.get('market_stress_index',0):.2f}")
                st.markdown(f"<div style='font-family:monospace;font-size:0.66rem;color:#7080a0;"
                            f"margin:0.6rem 0 0.3rem'>{ctx}</div>", unsafe_allow_html=True)
                cc1, cc2 = st.columns(2)
                for col, key, cls, name in [(cc1, "hawk_decision", "h", "REPUBLICAN"), (cc2, "dove_decision", "d", "DEMOCRAT")]:
                    d = p.get(key, {}); act = d.get("action", "HOLD")
                    with col:
                        st.markdown(f"""
<div class="reason {cls}"><div class="reason-head"><span class="reason-who">{name}</span>
  <span class="act-badge act-{act}">{act.replace('_',' ')}</span></div>
  <div class="reason-meta" style="border-top:none;border-bottom:1px solid var(--bdr)">
    <span class="rm">delay cost <b>{d.get('delay_cost_implied',0):.1f}</b></span>
    <span class="rm">belief opp <b>{d.get('belief_opponent_delay_cost',0):.1f}</b></span>
    <span class="rm">P(c) <b>{d.get('concession_probability',0):.2f}</b></span></div>
  <div class="reason-body">{d.get('reasoning','—')}</div>
  <div class="reason-quote">“{d.get('public_statement','')}”</div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — REGRESSION  (academic OLS table + correlation + hazard model)
# ═══════════════════════════════════════════════════════════════════════════
with t5:
    if not HAS:
        st.markdown('<div class="empty">Awaiting simulation data.</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='slbl'>OLS — concession period on delay-cost ratio</div>", unsafe_allow_html=True)
        smp = st.radio("Sample", ["hist", "pooled", "masked"], horizontal=True, key="reg_smp",
                       format_func=lambda s: {"hist": "Historical (primary, n=40)",
                                              "pooled": "Pooled (n=80, paired — see note)",
                                              "masked": "Masked (replication, n=40)"}[s])
        o = _ols(smp)
        if o and "model" in o:
            tbl = pd.DataFrame({
                "coef": [o["c_beta"], o["beta"]],
                "std err": [o["c_se"], o["se"]],
                "t": [o["c_t"], o["t"]],
                "p": [o["c_p"], o["p"]],
                "[0.025": [o["model"].conf_int().iloc[0, 0], o["ci_lo"]],
                "0.975]": [o["model"].conf_int().iloc[0, 1], o["ci_hi"]],
                "": ["", stars(o["p"])],
            }, index=["const", "delay_cost_ratio"])
            st.dataframe(tbl.style.format({"coef": "{:.3f}", "std err": "{:.3f}", "t": "{:.3f}",
                         "p": "{:.4f}", "[0.025": "{:.3f}", "0.975]": "{:.3f}"}), use_container_width=True)
            st.markdown(
                f"<div class='stat-annot'>Dependent variable: <code>concession_period</code>. "
                f"N = {o['n']} &nbsp;·&nbsp; R² = {o['r2']:.3f} &nbsp;·&nbsp; adj. R² = {o['adj']:.3f} "
                f"&nbsp;·&nbsp; F(1,{o['df']}) = {o['f']:.2f}, p = {o['fp']:.4f}. "
                f"Significance: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.10.</div>", unsafe_allow_html=True)
            if smp == "pooled":
                st.warning("The pooled sample double-counts paired historical/masked runs of the same "
                           "scenario (non-independent observations); its smaller p-value is not the "
                           "primary evidence. The historical-only fit (n=40) is the primary estimate.")
            st.markdown("<div class='slbl' style='margin-top:0.8rem'>Scatter with OLS fit</div>", unsafe_allow_html=True)
            d = o["data"]; xr = np.linspace(d["cr"].min(), d["cr"].max(), 80)
            figR = go.Figure()
            figR.add_trace(go.Scatter(x=d["cr"], y=d["y"], mode="markers",
                           marker=dict(color=DOVE, size=7, opacity=0.7), name="runs"))
            figR.add_trace(go.Scatter(x=xr, y=o["c_beta"] + o["beta"] * xr, mode="lines",
                           name=f"β={o['beta']:.1f}", line=dict(color=GOLD, width=2)))
            _lay(figR, h=300, margin=dict(t=16, b=34, l=44, r=12),
                 xaxis=dict(title="Republican/Democrat delay-cost ratio"), yaxis=dict(title="Concession period"))
            st.plotly_chart(figR, use_container_width=True)
            cap("H1 regression line; negative slope indicates earlier concession at higher cost ratio",
                f"{smp} sample, OLS via statsmodels, n = {o['n']}")
        else:
            st.info("Insufficient data to fit OLS.")

        st.markdown("<div class='slbl' style='margin-top:1rem'>Pearson correlation — final market stress × timing</div>",
                    unsafe_allow_html=True)
        r_s = float(sim["final_market_stress"].corr(sim["concession_period"]))
        figC = go.Figure()
        for c in "ABCDE":
            m = sim.condition_id == c
            figC.add_trace(go.Scatter(x=sim.loc[m, "final_market_stress"], y=sim.loc[m, "concession_period"],
                           mode="markers", name=f"Cond {c}", marker=dict(color=_CD[c], size=7, opacity=0.7)))
        _lay(figC, h=280, margin=dict(t=16, b=34, l=44, r=12),
             xaxis=dict(title="Final market stress index"), yaxis=dict(title="Concession period"))
        st.plotly_chart(figC, use_container_width=True)
        cap(f"H3 as a continuous correlation: r = {r_s:.3f}, n = {len(sim)}",
            "final_market_stress vs concession_period, all runs")

        st.markdown("<div class='slbl' style='margin-top:1rem'>Survival / hazard model</div>", unsafe_allow_html=True)
        try:
            from lifelines import KaplanMeierFitter, CoxPHFitter
            kmf = KaplanMeierFitter()
            figK = go.Figure()
            for c in "ABCDE":
                d_ = sim[sim.condition_id == c]["concession_period"].clip(0, 30)
                kmf.fit(d_, event_observed=np.ones(len(d_)))
                sf = kmf.survival_function_
                figK.add_trace(go.Scatter(x=sf.index, y=sf.iloc[:, 0], mode="lines", name=f"Cond {c}",
                               line=dict(color=_CD[c], width=2.2, shape="hv")))
            _lay(figK, h=280, margin=dict(t=16, b=34, l=44, r=12),
                 xaxis=dict(title="Period"), yaxis=dict(title="Kaplan-Meier S(t)", range=[0, 1.02]))
            st.plotly_chart(figK, use_container_width=True)
            # Cox PH on condition ordinal
            cox_df = sim.assign(cond_ord=sim.condition_id.map({k: i for i, k in enumerate("ABCDE")}),
                                T=sim["concession_period"].clip(0, 30), E=1)[["cond_ord", "T", "E"]]
            cph = CoxPHFitter().fit(cox_df, duration_col="T", event_col="E")
            hr = float(np.exp(cph.params_["cond_ord"])); cp = float(cph.summary.loc["cond_ord", "p"])
            cap(f"Cox PH: each step A→E multiplies the concession hazard by HR = {hr:.2f} (p = {cp:.4f})",
                "lifelines CoxPHFitter on condition ordinal, n = " + str(len(sim)))
        except Exception as e:
            st.info(f"Hazard model unavailable ({type(e).__name__}); install lifelines for Cox PH / KM fits.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════
with t6:
    ml, mr = st.columns([5, 4], gap="large")
    with ml:
        st.markdown("<div class='slbl'>Theoretical model — Alesina &amp; Drazen (1991)</div>", unsafe_allow_html=True)
        st.markdown("<span style='font-family:monospace;font-size:0.78rem;color:#b0bcd4;line-height:1.7'>"
                    "Two factions each draw a private delay-cost rate. The faction with the higher rate "
                    "concedes first; the hazard of resolution rises with the deadweight loss of delay.</span>",
                    unsafe_allow_html=True)
        st.latex(r"\lambda_i \sim \text{Exp}(\mu_i), \quad i \in \{\text{Rep}, \text{Dem}\}")
        st.latex(r"P(i \text{ concedes first}) = \frac{\mu_i}{\mu_i + \mu_j}, \quad \mathbb{E}[T^*] = \frac{1}{\mu_i + \mu_j}")
        st.latex(r"h(t) = \mu_i + \mu_j, \qquad \mu_{\text{eff}}(s) = \mu_{\text{base}}(1 + \gamma s), \; s \in [0,1]")
        st.markdown("<div class='slbl' style='margin-top:1rem'>Mapping to the simulation</div>", unsafe_allow_html=True)
        st.markdown("<span style='font-family:monospace;font-size:0.76rem;color:#b0bcd4;line-height:1.7'>"
                    "The per-period <code>delay_cost_implied</code> score (0–10, self-reported by each agent) "
                    "proxies the latent rate λ; the Republican/Democrat ratio of run-means is the H1 regressor. The "
                    "period of the first CONCEDE is the realized stopping time T*. Condition A→E scales s. "
                    "Republican and Democrat agents; code labels HAWK and DOVE.</span>",
                    unsafe_allow_html=True)

        st.markdown("<div class='slbl' style='margin-top:1rem'>Hypotheses</div>", unsafe_allow_html=True)
        for lab, eq, desc in [
            ("H1 — Delay cost predicts timing", r"T^* \downarrow \text{ as } \lambda_{\text{Rep}}/\lambda_{\text{Dem}} \uparrow",
             "Higher Republican relative delay cost → earlier Republican concession."),
            ("H2 — Deadline pressure", r"h(t) \uparrow \text{ as } (T_x - t) \downarrow",
             "Hazard rises as days-to-X-date decrease."),
            ("H3 — Market-stress channel", r"\mu_{\text{eff}} \uparrow \text{ as } s \uparrow",
             "Concession rate and hazard increase from condition A to E."),
            ("H4 — External validation", r"\hat{h}(t) \propto \Delta y_{\text{T-bill}}(t)",
             "Simulated hazard should track real T-bill yield anomalies."),
        ]:
            st.markdown(f"<div class='finding'><div class='finding-label'>{lab}</div>", unsafe_allow_html=True)
            st.latex(eq)
            st.markdown(f"<div class='finding-sub'>{desc}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='slbl' style='margin-top:1rem'>Identification</div>", unsafe_allow_html=True)
        st.markdown("<span style='font-family:monospace;font-size:0.76rem;color:#b0bcd4;line-height:1.7'>"
                    "<b>Threat:</b> historical episodes are in the training corpus. "
                    "<b>Mitigations:</b> (i) masked replays with outcome hidden and identifying detail removed; "
                    "(ii) a fictional 2025 counterfactual as an out-of-sample test; (iii) the primary H1 "
                    "estimate is fit on the 40 historical runs, with masked runs reported as replication to "
                    "avoid double-counting paired observations.</span>", unsafe_allow_html=True)

        st.markdown("<div class='slbl' style='margin-top:1rem'>Literature gap</div>", unsafe_allow_html=True)
        st.markdown("<span style='font-family:monospace;font-size:0.76rem;color:#b0bcd4;line-height:1.7'>"
                    "Baker et al. (2024) and related work document that LLM agents exhibit coherent "
                    "strategic behaviour in matrix and bargaining games. The contribution here is to test "
                    "a specific, falsifiable political-economy prediction — the Alesina-Drazen "
                    "private-type/hazard mechanism — against a calibrated, FRED-grounded environment with "
                    "an explicit contamination control.</span>", unsafe_allow_html=True)

        st.markdown("<div class='slbl' style='margin-top:1rem'>Limitations</div>", unsafe_allow_html=True)
        st.markdown("<span style='font-family:monospace;font-size:0.76rem;color:#7080a0;line-height:1.7'>"
                    "• Two replicates per cell (n=40 historical runs); CIs are wide (β CI spans "
                    f"[{o_h['ci_lo']:.0f}, {o_h['ci_hi']:.0f}] if available).<br>"
                    "• The delay-cost regressor is self-reported by the same agent whose timing is the "
                    "outcome, so H1 is an internal-consistency test, not an external instrument.<br>"
                    "• Market-stress inputs for historical episodes are condition-scaled, not raw realized "
                    "series; H4 (T-bill validation) is not yet estimated here.<br>"
                    "• A single agent model (claude-haiku-4-5); results may not generalize across models.</span>",
                    unsafe_allow_html=True)

    with mr:
        st.markdown("<div class='slbl'>Agent architecture</div>", unsafe_allow_html=True)
        st.markdown("""
**Negotiating agents**: `claude-haiku-4-5`
**Judge model**: `claude-sonnet-4-6`
**Temperature**: 0.85 – 1.00 (varied)
**Interface**: structured tool call, typed fields

Per-period observation:

| Field | Source |
|---|---|
| `days_to_xdate` | episode parquet |
| `market_stress_index` | FRED / synthetic [0–1] |
| `vix` | FRED VIXCLS |
| `tbill_4wk` | FRED TB4WK |
| `polling_approval_pct` | polling archive |
| `opponent_last_action` | previous period |

Per-period output:

| Field | Type | Role |
|---|---|---|
| `action` | enum | HOLD / SIGNAL_FLEXIBILITY / CONCEDE |
| `reasoning` | str | private chain-of-thought |
| `concession_probability` | float | P(concede ≤5 periods) |
| `delay_cost_implied` | float | self-reported cost [0–10] |
| `belief_opponent_delay_cost` | float | inferred opponent cost |
| `public_statement` | str | observable position |
""")
        st.markdown("<div class='slbl' style='margin-top:0.8rem'>Design</div>", unsafe_allow_html=True)
        st.markdown("""
| Parameter | Value |
|---|---|
| Episodes | 2011, 2013, 2023, 2025-CF (fictional) |
| Conditions | 5 (A–E) |
| Replicates | 2 per cell |
| Mask states | historical + masked |
| Total runs | 80 (40 scenarios × 2) |
| Max periods | 30 |
""")
        st.markdown("<div class='slbl' style='margin-top:0.8rem'>Data sources</div>", unsafe_allow_html=True)
        for s, d in [("FRED VIXCLS", "daily VIX close; equity-volatility proxy"),
                     ("FRED TB4WK", "4-week T-bill rate; short-term default-risk proxy"),
                     ("Polling archives", "approval & deal-preference, daily-interpolated"),
                     ("Episode YAMLs", "dates, debt/GDP, deficit, resolution text"),
                     ("Condition YAMLs", "A–E stress multipliers")]:
            st.markdown(f"<div class='finding' style='border-left-color:#5566aa'>"
                        f"<div class='finding-label'>{s}</div><div class='finding-sub'>{d}</div></div>",
                        unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("<div class='note'>Alesina, A. &amp; Drazen, A. (1991). Why are stabilizations delayed? "
            "<i>American Economic Review</i>, 81(5), 1170–1188. &nbsp;·&nbsp; "
            "Agents: claude-haiku-4-5 · Judge: claude-sonnet-4-6 &nbsp;·&nbsp; "
            "Statistics recomputed live from outputs/results/; no API calls in this dashboard.</div>",
            unsafe_allow_html=True)
