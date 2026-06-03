"""Arena visualization components for the live negotiation dashboard."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from lifelines import KaplanMeierFitter
    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False

_ACTION_COLORS = {"HOLD": "#4CC9F0", "SIGNAL_FLEXIBILITY": "#FFE66D", "CONCEDE": "#FF4D6D"}
_CONDITION_COLORS = {"A": "#4CC9F0", "B": "#7BF1A8", "C": "#FFE66D", "D": "#FF9A3C", "E": "#FF4D6D"}


def build_tug_of_war_figure(
    hawk_concession_prob: float,
    dove_concession_prob: float,
    period: int,
    days_to_xdate: int,
    hawk_action: str = "HOLD",
    dove_action: str = "HOLD",
) -> go.Figure:
    """Horizontal tug-of-war showing negotiation territory balance."""
    total = hawk_concession_prob + dove_concession_prob + 1e-9
    # Line position: 0=HAWK wins all territory, 1=DOVE wins all territory
    line_pos = dove_concession_prob / total   # DOVE conceding more → line moves right (HAWK territory)
    # Invert: dove conceding = hawk winning
    hawk_territory = 1.0 - line_pos

    x_line = hawk_territory * 100  # 0–100 scale

    # Color intensity based on who is winning
    hawk_color = f"rgba(255,{int(70 + (1-hawk_territory)*100)},77,0.85)"
    dove_color = f"rgba(0,{int(150 + hawk_territory*100)},255,0.85)"

    fig = go.Figure()

    # HAWK territory (left)
    fig.add_trace(go.Bar(
        x=[x_line], y=["Territory"],
        orientation="h",
        marker_color=hawk_color,
        name="HAWK Territory",
        hoverinfo="skip",
        width=0.4,
    ))
    # DOVE territory (right)
    fig.add_trace(go.Bar(
        x=[100 - x_line], y=["Territory"],
        orientation="h",
        base=[x_line],
        marker_color=dove_color,
        name="DOVE Territory",
        hoverinfo="skip",
        width=0.4,
    ))

    # Center divider line
    fig.add_vline(x=x_line, line_width=4, line_color="white")

    # Labels at extremes
    fig.add_annotation(x=5, y=0.6, text="◀ HAWK", font=dict(color="#FF4D6D", size=14, family="monospace"),
                       showarrow=False, xref="x", yref="paper")
    fig.add_annotation(x=95, y=0.6, text="DOVE ▶", font=dict(color="#4CC9F0", size=14, family="monospace"),
                       showarrow=False, xref="x", yref="paper")

    # Action badges
    hawk_badge_color = _ACTION_COLORS.get(hawk_action, "#AAAAAA")
    dove_badge_color = _ACTION_COLORS.get(dove_action, "#AAAAAA")
    fig.add_annotation(x=5, y=0.1, text=f"HAWK: {hawk_action}", showarrow=False,
                       font=dict(color=hawk_badge_color, size=11), xref="x", yref="paper",
                       bgcolor="rgba(0,0,0,0.5)", bordercolor=hawk_badge_color)
    fig.add_annotation(x=95, y=0.1, text=f"DOVE: {dove_action}", showarrow=False,
                       font=dict(color=dove_badge_color, size=11), xref="x", yref="paper",
                       bgcolor="rgba(0,0,0,0.5)", bordercolor=dove_badge_color)

    fig.update_layout(
        title=dict(
            text=f"NEGOTIATION BALANCE — Period {period} | {days_to_xdate} days to X-Date",
            font=dict(color="white", size=14, family="monospace"),
        ),
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        barmode="stack",
        paper_bgcolor="rgb(10,10,20)",
        plot_bgcolor="rgb(10,10,20)",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=120,
    )
    return fig


def build_agent_status_figure(
    decision_history: list[dict],
    agent_role: str,
) -> go.Figure:
    """Multi-metric panel for one agent: concession prob + delay cost over time."""
    if not decision_history:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="rgb(10,10,20)", height=280)
        return fig

    periods = [d["period"] for d in decision_history]
    probs = [d.get("concession_probability", 0) for d in decision_history]
    costs = [d.get("delay_cost_implied", 0) for d in decision_history]
    actions = [d.get("action", "HOLD") for d in decision_history]
    colors = [_ACTION_COLORS.get(a, "#AAAAAA") for a in actions]

    label = agent_role
    base_color = "#FF4D6D" if agent_role == "HAWK" else "#4CC9F0"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.55, 0.45])

    # Top: concession probability
    fig.add_trace(go.Scatter(
        x=periods, y=probs, mode="lines+markers",
        name="P(Concede)", line=dict(color=base_color, width=2.5),
        marker=dict(size=9, color=colors, line=dict(color="white", width=1)),
        hovertemplate="Period %{x}<br>P(Concede): %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="white", opacity=0.3, row=1, col=1)

    # Bottom: delay cost
    fig.add_trace(go.Bar(
        x=periods, y=costs, name="Delay Cost",
        marker_color=colors, opacity=0.85,
        hovertemplate="Period %{x}<br>Delay Cost: %{y:.1f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{label} — Status Panel", font=dict(color=base_color, size=13)),
        paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white"), showlegend=False,
        margin=dict(l=40, r=10, t=40, b=10), height=280,
        xaxis2=dict(title="Period", tickfont=dict(color="white")),
        yaxis=dict(title="P(Concede)", range=[0, 1], tickfont=dict(color="white")),
        yaxis2=dict(title="Delay Cost (0-10)", tickfont=dict(color="white")),
    )
    return fig


def build_market_pressure_figure(period_records: list[dict]) -> go.Figure:
    """Market storm visualization: VIX, stress, T-bill on one panel."""
    if not period_records:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="rgb(10,10,20)", height=240)
        return fig

    periods = [r.get("period", i) for i, r in enumerate(period_records)]
    vix = [r.get("vix", 20) for r in period_records]
    stress = [r.get("market_stress_index", 0.3) for r in period_records]
    tbill = [r.get("tbill_4wk", 3.0) for r in period_records]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Stress filled area — color shifts with intensity
    stress_colors = [f"rgba({int(50 + s*200)},{int(100 - s*80)},50,0.7)" for s in stress]
    fig.add_trace(go.Scatter(
        x=periods, y=stress, fill="tozeroy",
        mode="lines", name="Market Stress",
        line=dict(color="#FF6B35", width=2),
        fillcolor="rgba(255,107,53,0.25)",
        hovertemplate="Period %{x}<br>Stress: %{y:.2f}<extra></extra>",
    ), secondary_y=False)

    # VIX line
    fig.add_trace(go.Scatter(
        x=periods, y=vix, mode="lines", name="VIX",
        line=dict(color="#FF4D6D", width=2, dash="dash"),
        hovertemplate="Period %{x}<br>VIX: %{y:.1f}<extra></extra>",
    ), secondary_y=True)

    # Danger zone shading
    fig.add_hrect(y0=0.7, y1=1.0, line_width=0, fillcolor="rgba(255,0,0,0.07)",
                  annotation_text="DANGER ZONE", annotation_font_color="#FF4D6D",
                  secondary_y=False)

    fig.update_layout(
        title=dict(text="Market Fear Engine", font=dict(color="#FF6B35", size=13)),
        paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(12,8,18)",
        font=dict(color="white"), showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=50, t=40, b=10), height=240,
        xaxis=dict(title="Period", tickfont=dict(color="white")),
        yaxis=dict(title="Stress Index", range=[0, 1.05], tickfont=dict(color="#FF6B35")),
        yaxis2=dict(title="VIX", tickfont=dict(color="#FF4D6D")),
    )
    return fig


def build_episode_summary_figure(sim_df: pd.DataFrame, episode_id: str) -> go.Figure:
    """4-panel episode summary: KM survival, condition box plot, winner pie, cost scatter."""
    df = sim_df[sim_df["episode_id"] == episode_id].copy()
    if len(df) == 0:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="rgb(10,10,20)", height=600)
        return fig

    df["duration"] = df["concession_period"].fillna(30).clip(lower=0)
    df["duration"] = np.where(df["duration"] < 0, 30, df["duration"])
    df["event"] = (df["winner"] != "CENSORED").astype(int)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Survival Function", "Concession Time by Condition",
                        "Winner Distribution", "Cost Ratio vs Concession Time"],
        horizontal_spacing=0.12, vertical_spacing=0.16,
    )

    # Panel 1: KM survival (manual)
    sorted_t = np.sort(df["duration"].values)
    surv = np.array([np.mean(df["duration"] > t) for t in sorted_t])
    fig.add_trace(go.Scatter(
        x=sorted_t, y=surv, mode="lines", name="Empirical",
        line=dict(color="#00BFFF", width=2.5, shape="hv"),
    ), row=1, col=1)
    # Theoretical
    from src.theory.alesina_drazen import calibrate_mu
    mu = calibrate_mu(df["duration"].tolist(), max_period=30)
    t_th = np.linspace(0, 30, 100)
    fig.add_trace(go.Scatter(
        x=t_th, y=np.exp(-mu * t_th), mode="lines", name=f"AD Theory (μ={mu:.3f})",
        line=dict(color="#FF6B35", width=2, dash="dash"),
    ), row=1, col=1)

    # Panel 2: Box plot by condition
    for cid in sorted(df["condition_id"].unique()):
        sub = df[df["condition_id"] == cid]["duration"]
        color = _CONDITION_COLORS.get(str(cid), "#AAAAAA")
        fig.add_trace(go.Box(
            y=sub, name=f"Cond {cid}", marker_color=color,
            line_color="white", boxmean=True, showlegend=False,
        ), row=1, col=2)

    # Panel 3: Winner pie
    winner_counts = df["winner"].value_counts()
    fig.add_trace(go.Pie(
        labels=winner_counts.index.tolist(),
        values=winner_counts.values.tolist(),
        marker=dict(colors=["#FF4D6D", "#4CC9F0", "#888888"]),
        hole=0.4, showlegend=True,
        textfont=dict(color="white"),
    ), row=2, col=1)

    # Panel 4: Cost ratio scatter
    if "hawk_mean_delay_cost" in df.columns and "dove_mean_delay_cost" in df.columns:
        cost_ratio = df["hawk_mean_delay_cost"] / df["dove_mean_delay_cost"].replace(0, np.nan)
        fig.add_trace(go.Scatter(
            x=cost_ratio, y=df["duration"],
            mode="markers", showlegend=False,
            marker=dict(color="#7BF1A8", size=7, opacity=0.7),
            hovertemplate="Cost Ratio: %{x:.2f}<br>Concession Period: %{y:.0f}<extra></extra>",
        ), row=2, col=2)

    fig.update_layout(
        title=dict(text=f"Episode {episode_id} — Simulation Summary",
                   font=dict(color="white", size=15)),
        paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white"), height=600,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
    )
    for ann in fig.layout.annotations:
        ann.font.color = "white"
    return fig
