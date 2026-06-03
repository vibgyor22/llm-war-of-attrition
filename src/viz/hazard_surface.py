"""3D Hazard Landscape — X=Days-to-XDate, Y=Market-Stress, Z=Concession-Probability."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_CONDITION_COLORS = {"A": "#4CC9F0", "B": "#7BF1A8", "C": "#FFE66D", "D": "#FF9A3C", "E": "#FF4D6D"}
_RESULTS_DIR = Path(__file__).parents[2] / "outputs" / "results"
_FIGURES_DIR = Path(__file__).parents[2] / "outputs" / "figures"


def _compute_z(t_grid: np.ndarray, stress_grid: np.ndarray,
               mu_base: float = 0.12, stress_sensitivity: float = 0.25) -> np.ndarray:
    mu_eff = mu_base * (1.0 + stress_sensitivity * stress_grid)          # shape (S,)
    time_elapsed = (30.0 - t_grid)[:, None]                               # shape (T,1)
    Z = 1.0 - np.exp(-mu_eff[None, :] * time_elapsed)                    # shape (T,S)
    return np.clip(Z, 0.0, 1.0)


def build_surface_figure(
    sim_df: pd.DataFrame | None = None,
    t_resolution: int = 40,
    stress_resolution: int = 40,
    show_theory: bool = True,
    show_sims: bool = True,
    episode_filter: str | None = None,
) -> go.Figure:
    t_grid = np.linspace(0, 30, t_resolution)
    stress_grid = np.linspace(0, 1, stress_resolution)
    Z = _compute_z(t_grid, stress_grid)

    fig = go.Figure()

    if show_theory:
        fig.add_trace(go.Surface(
            x=stress_grid, y=t_grid, z=Z,
            colorscale="Viridis", opacity=0.72,
            name="AD Theory",
            showscale=True,
            colorbar=dict(title="P(Concession)", tickfont=dict(color="white")),
            hovertemplate="Stress: %{x:.2f}<br>Days-to-X: %{y:.0f}<br>P(Concede): %{z:.2f}<extra>Theory</extra>",
        ))

    if show_sims and sim_df is not None and len(sim_df) > 0:
        df = sim_df.copy()
        if episode_filter:
            df = df[df["episode_id"] == episode_filter]
        df["concession_days"] = df["concession_period"].clip(lower=0).fillna(30)
        df["concession_days"] = np.where(df["concession_days"] < 0, 30, df["concession_days"])
        df["conceded"] = (df["winner"] != "CENSORED").astype(float)
        stress_col = df.get("final_market_stress", pd.Series([0.5] * len(df), index=df.index))
        if "final_market_stress" not in df.columns:
            stress_col = pd.Series(0.5, index=df.index)
        else:
            stress_col = df["final_market_stress"].fillna(0.5)

        for cid, grp in df.groupby("condition_id"):
            color = _CONDITION_COLORS.get(str(cid), "#FFFFFF")
            s_vals = stress_col.loc[grp.index] if hasattr(stress_col, "loc") else [0.5] * len(grp)
            fig.add_trace(go.Scatter3d(
                x=s_vals,
                y=grp["concession_days"].values,
                z=grp["conceded"].values + np.random.uniform(-0.02, 0.02, len(grp)),
                mode="markers",
                name=f"Condition {cid}",
                marker=dict(size=5, color=color, opacity=0.85,
                            line=dict(width=0.5, color="white")),
                hovertemplate=f"Condition {cid}<br>Stress: %{{x:.2f}}<br>Day: %{{y:.0f}}<br>Conceded: %{{z:.0f}}<extra></extra>",
            ))

    fig.update_layout(
        title=dict(text="Concession Probability Landscape — Alesina-Drazen Theory vs LLM Simulations",
                   font=dict(size=16, color="white")),
        scene=dict(
            xaxis=dict(title="Market Stress Index", titlefont=dict(color="white"),
                       tickfont=dict(color="white"), gridcolor="rgba(255,255,255,0.15)"),
            yaxis=dict(title="Days to X-Date", titlefont=dict(color="white"),
                       tickfont=dict(color="white"), gridcolor="rgba(255,255,255,0.15)"),
            zaxis=dict(title="P(Concession)", titlefont=dict(color="white"),
                       tickfont=dict(color="white"), gridcolor="rgba(255,255,255,0.15)"),
            bgcolor="rgb(10,10,25)",
        ),
        paper_bgcolor="rgb(10,10,20)",
        legend=dict(bgcolor="rgba(0,0,0,0.6)", font=dict(color="white")),
        margin=dict(l=0, r=0, t=50, b=0),
        height=620,
    )
    return fig


def save_figure(fig: go.Figure, output_dir: Path | None = None,
                name: str = "hazard_surface", formats: list[str] | None = None) -> None:
    out = output_dir or _FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    fmts = formats or ["html"]
    if "html" in fmts:
        fig.write_html(str(out / f"{name}.html"))
    if "png" in fmts:
        try:
            fig.write_image(str(out / f"{name}.png"), scale=2)
        except Exception:
            pass  # kaleido may not be installed


if __name__ == "__main__":
    fig = build_surface_figure(show_sims=False)
    save_figure(fig)
    print("Hazard surface saved.")
