"""Belief Evolution Theater — animated distribution of agent beliefs over periods."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_FIGURES_DIR = Path(__file__).parents[2] / "outputs" / "figures"


def build_belief_animation(
    period_df: pd.DataFrame,
    episode_id: str | None = None,
    condition_id: str | None = None,
    agent: str = "hawk",
) -> go.Figure:
    """Animated histogram of belief_opponent_delay_cost over periods."""
    df = period_df.copy()
    if episode_id:
        df = df[df["episode_id"] == episode_id]
    if condition_id:
        df = df[df["condition_id"] == condition_id]

    belief_col = "hawk_belief_opponent" if agent == "hawk" else "dove_belief_opponent"
    if belief_col not in df.columns:
        belief_col = "hawk_delay_cost" if agent == "hawk" else "dove_delay_cost"

    color = "#00BFFF" if agent == "hawk" else "#FFB347"
    label = "HAWK" if agent == "hawk" else "DOVE"

    periods = sorted(df["period"].unique())
    bins = np.linspace(0, 10, 21)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    frames: list[go.Frame] = []
    slider_steps: list[dict] = []

    for t in periods:
        sub = df[df["period"] == t][belief_col].dropna()
        counts, _ = np.histogram(sub, bins=bins)
        density = counts / max(counts.sum(), 1)
        mean_val = sub.mean() if len(sub) > 0 else 0

        frame = go.Frame(
            data=[
                go.Bar(x=bin_centers, y=density, marker_color=color,
                       opacity=0.85, name="Distribution"),
                go.Scatter(x=[mean_val, mean_val], y=[0, density.max() * 1.1],
                           mode="lines", line=dict(color="white", dash="dash", width=2),
                           name=f"Mean: {mean_val:.2f}"),
            ],
            name=str(t),
            layout=go.Layout(
                annotations=[dict(
                    x=0.02, y=0.97, xref="paper", yref="paper",
                    text=f"Period {t} | Mean belief: {mean_val:.2f}",
                    showarrow=False, font=dict(color="white", size=13),
                    bgcolor="rgba(0,0,0,0.5)", bordercolor="white",
                )]
            ),
        )
        frames.append(frame)
        slider_steps.append(dict(
            method="animate",
            args=[[str(t)], {"frame": {"duration": 400, "redraw": True}, "transition": {"duration": 200}}],
            label=str(t),
        ))

    # Initial frame
    t0 = periods[0] if periods else 0
    sub0 = df[df["period"] == t0][belief_col].dropna()
    counts0, _ = np.histogram(sub0, bins=bins)
    density0 = counts0 / max(counts0.sum(), 1)
    mean0 = sub0.mean() if len(sub0) > 0 else 5.0

    fig = go.Figure(
        data=[
            go.Bar(x=bin_centers, y=density0, marker_color=color, opacity=0.85, name="Distribution"),
            go.Scatter(x=[mean0, mean0], y=[0, density0.max() * 1.1],
                       mode="lines", line=dict(color="white", dash="dash", width=2), name=f"Mean"),
        ],
        frames=frames,
    )

    fig.update_layout(
        title=dict(text=f"What {label} believes about opponent's delay cost — Evolution over time",
                   font=dict(size=15, color="white")),
        xaxis=dict(title="Opponent Delay Cost (0–10)", range=[0, 10],
                   tickfont=dict(color="white"), titlefont=dict(color="white")),
        yaxis=dict(title="Normalized Frequency", tickfont=dict(color="white"),
                   titlefont=dict(color="white")),
        paper_bgcolor="rgb(10,10,20)",
        plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        updatemenus=[dict(
            type="buttons", showactive=False, y=1.1, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 500}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
        )],
        sliders=[dict(
            active=0, steps=slider_steps,
            currentvalue=dict(prefix="Period: ", font=dict(color="white")),
            pad=dict(t=50),
            bgcolor="rgba(255,255,255,0.1)",
            font=dict(color="white"),
        )],
        height=500,
    )
    return fig


def build_belief_comparison_figure(period_df: pd.DataFrame, sim_id: str) -> go.Figure:
    """Single-sim belief line chart for both agents with action annotations."""
    df = period_df[period_df["sim_id"] == sim_id].sort_values("period")

    fig = go.Figure()

    if "hawk_belief_opponent" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["period"], y=df["hawk_belief_opponent"],
            mode="lines+markers", name="HAWK's belief about DOVE",
            line=dict(color="#00BFFF", width=2.5),
            marker=dict(size=7),
        ))
    if "dove_belief_opponent" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["period"], y=df["dove_belief_opponent"],
            mode="lines+markers", name="DOVE's belief about HAWK",
            line=dict(color="#FFB347", width=2.5),
            marker=dict(size=7),
        ))

    # Annotate key events
    for _, row in df.iterrows():
        for agent, col, color in [("HAWK", "hawk_action", "#00BFFF"), ("DOVE", "dove_action", "#FFB347")]:
            action = row.get(col, "HOLD")
            if action == "SIGNAL_FLEXIBILITY":
                fig.add_vline(x=row["period"], line_dash="dot", line_color=color, opacity=0.5,
                              annotation_text=f"{agent[0]}:SIGNAL", annotation_font_color=color)
            elif action == "CONCEDE":
                fig.add_vline(x=row["period"], line_dash="solid", line_color=color, opacity=0.9,
                              annotation_text=f"{agent[0]}:CONCEDE", annotation_font_color=color)

    fig.update_layout(
        title=dict(text=f"Belief Evolution — Simulation {sim_id}", font=dict(color="white", size=14)),
        xaxis=dict(title="Period", tickfont=dict(color="white"), titlefont=dict(color="white")),
        yaxis=dict(title="Estimated Opponent Delay Cost (0–10)",
                   tickfont=dict(color="white"), titlefont=dict(color="white")),
        paper_bgcolor="rgb(10,10,20)", plot_bgcolor="rgb(15,15,30)",
        font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        height=420,
    )
    return fig
