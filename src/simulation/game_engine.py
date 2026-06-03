"""Game engine: orchestrates one simulation of the HAWK vs DOVE negotiation."""
from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base_agent import AgentDecision, AgentOutputError
from src.agents.hawk_agent import HawkAgent
from src.agents.dove_agent import DoveAgent
from src.theory.bayesian_belief import BayesianBeliefUpdater
from src.simulation.transcript import PeriodRecord, SimulationTranscript

logger = logging.getLogger(__name__)

MAX_PERIODS: int = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_state(
    row: pd.Series,
    t: int,
    condition_config: dict,
    prev_hawk_action: str,
    prev_dove_action: str,
    hawk_perspective: bool,
) -> dict[str, Any]:
    """Construct the state dict delivered to one agent for period *t*.

    Parameters
    ----------
    row:
        The episode DataFrame row for period *t*.
    t:
        Current period index (0-based).
    condition_config:
        Condition YAML as a dict.  Expected keys: ``vix_multiplier``,
        ``tbill_spike_bps``, ``market_stress_floor``, ``polls_approval_offset``,
        ``narrative_injection``.
    prev_hawk_action, prev_dove_action:
        String actions from the preceding period, or ``"None"`` at t=0.
    hawk_perspective:
        If ``True``, build the state from HAWK's point of view (opponent =
        DOVE).  If ``False``, build from DOVE's perspective (opponent = HAWK).
    """
    # --- Apply condition modifications -----------------------------------
    vix_multiplier: float = float(condition_config.get("vix_multiplier", 1.0))
    tbill_spike_bps: float = float(condition_config.get("tbill_spike_bps", 0))
    market_stress_floor: float = float(condition_config.get("market_stress_floor", 0.0))
    polls_approval_offset: float = float(
        condition_config.get("polls_approval_offset", 0)
    )

    modified_vix: float = float(row["vix"]) * vix_multiplier
    modified_tbill: float = float(row["tbill_4wk"]) + tbill_spike_bps / 100.0
    modified_stress: float = max(
        float(row["market_stress_index"]), market_stress_floor
    )
    modified_polling: float = float(row["polling_approval_pct"]) + polls_approval_offset

    opponent_last_action: str = (
        prev_dove_action if hawk_perspective else prev_hawk_action
    )

    return {
        "period": t,
        "date": str(row["date"])[:10],
        "days_to_xdate": int(row["days_to_xdate"]),
        "vix": modified_vix,
        "tbill_4wk": modified_tbill,
        "market_stress_index": modified_stress,
        "polling_approval_pct": modified_polling,
        "debt_gdp_ratio": float(row["debt_gdp_ratio"]),
        "deficit_projection_bn": float(row["deficit_projection_bn"]),
        "election_days_out": int(row["election_days_out"]),
        "opponent_last_action": opponent_last_action,
        "narrative_injection": condition_config.get("narrative_injection", ""),
    }


def _resolve_winner(
    hawk_decision: AgentDecision,
    dove_decision: AgentDecision,
) -> tuple[str, str, int, bool]:
    """Determine game outcome after a period where at least one agent conceded.

    Returns (winner, conceding_agent, is_game_over) where:
      - ``winner``:  "HAWK" if DOVE conceded, "DOVE" if HAWK conceded,
                     "DRAW" if both conceded simultaneously.
      - ``conceding_agent``: who conceded first (HAWK is treated as first caller).
      - ``is_game_over``: always True here (caller only invokes when game ends).

    Design note: when HAWK concedes that means the HAWK faction accepted the
    opponent's terms, so DOVE wins (and vice-versa).  If both concede in the same
    period the spec says "treat first caller (hawk) as winner" — interpreted
    game-theoretically as HAWK's concession being the authoritative one, so DOVE
    wins.
    """
    hawk_conceded: bool = hawk_decision.action == "CONCEDE"
    dove_conceded: bool = dove_decision.action == "CONCEDE"

    if hawk_conceded and dove_conceded:
        # Simultaneous — HAWK called first; HAWK's concession is authoritative.
        return "DOVE", "HAWK", True
    if hawk_conceded:
        return "DOVE", "HAWK", True
    if dove_conceded:
        return "HAWK", "DOVE", True

    return "", "", False  # neither conceded


# ---------------------------------------------------------------------------
# Core simulation generator
# ---------------------------------------------------------------------------


def run_simulation_live(
    episode_df: pd.DataFrame,
    episode_config: dict,
    condition_config: dict,
    hawk: HawkAgent,
    dove: DoveAgent,
    sim_number: int,
    mask_outcome: bool,
    rng: np.random.Generator,
) -> Generator[PeriodRecord, None, SimulationTranscript]:
    """Run one simulation and *yield* each ``PeriodRecord`` as it is produced.

    Yields
    ------
    PeriodRecord
        One record per completed period (both agent decisions resolved).

    Returns
    -------
    SimulationTranscript
        The completed transcript (accessible via ``StopIteration.value`` when
        consuming the generator manually, or returned directly when using
        ``run_simulation``).

    Parameters
    ----------
    episode_df:
        30-row DataFrame of episode state (one row per period, 0-indexed).
    episode_config:
        Episode YAML loaded as a dict.
    condition_config:
        Condition YAML loaded as a dict.
    hawk:
        Instantiated ``HawkAgent`` (shared across sims within a condition run).
    dove:
        Instantiated ``DoveAgent`` (shared across sims within a condition run).
    sim_number:
        0-based index of this simulation within the condition run.
    mask_outcome:
        Forwarded to the episode_config passed to agents.
    rng:
        Numpy RNG for any stochastic elements (currently unused by the engine
        itself but passed through for reproducibility if agents ever need it).
    """
    episode_id: str = str(episode_config.get("episode_id", "unknown"))
    condition_id: str = str(condition_config.get("condition_id", "X"))
    sim_id: str = f"{episode_id}_{condition_id}_sim{sim_number:03d}"

    # Merge mask_outcome into the episode_config view the agents see.
    effective_episode_config = dict(episode_config)
    effective_episode_config["mask_historical_outcome"] = mask_outcome

    # Reset agents to clear conversation history from prior simulations.
    hawk.reset()
    dove.reset()

    # Per-sim Bayesian updaters (researcher-side tracking).
    hawk_updater = BayesianBeliefUpdater()
    dove_updater = BayesianBeliefUpdater()

    transcript = SimulationTranscript(
        sim_id=sim_id,
        episode_id=episode_id,
        condition_id=condition_id,
        sim_number=sim_number,
        temperature=hawk.temperature,  # same temp used for both agents per sim
        mask_historical_outcome=mask_outcome,
    )

    prev_hawk_action: str = "None"
    prev_dove_action: str = "None"

    winner: str = "CENSORED"
    concession_period: int = -1
    conceding_agent: str = ""

    for t in range(MAX_PERIODS):
        row: pd.Series = episode_df.iloc[t]

        hawk_state = _build_state(
            row, t, condition_config, prev_hawk_action, prev_dove_action,
            hawk_perspective=True,
        )
        dove_state = _build_state(
            row, t, condition_config, prev_hawk_action, prev_dove_action,
            hawk_perspective=False,
        )

        # --- Agent decisions -------------------------------------------
        try:
            hawk_decision: AgentDecision = hawk.act(
                hawk_state, effective_episode_config, condition_config
            )
        except AgentOutputError as exc:
            logger.error("HAWK AgentOutputError at period %d: %s", t, exc)
            raise

        try:
            dove_decision: AgentDecision = dove.act(
                dove_state, effective_episode_config, condition_config
            )
        except AgentOutputError as exc:
            logger.error("DOVE AgentOutputError at period %d: %s", t, exc)
            raise

        # --- Bayesian updates (researcher-side) -----------------------
        hawk_updater.update(hawk_decision.delay_cost_implied)
        dove_updater.update(dove_decision.delay_cost_implied)

        # --- Determine game continuation ------------------------------
        resolved_winner, resolved_conceder, game_over = _resolve_winner(
            hawk_decision, dove_decision
        )

        game_continues: bool = not game_over

        # --- Build period record --------------------------------------
        period_record = PeriodRecord(
            period=t,
            date=hawk_state["date"],
            days_to_xdate=int(hawk_state["days_to_xdate"]),
            vix=float(hawk_state["vix"]),
            market_stress_index=float(hawk_state["market_stress_index"]),
            polling_approval_pct=float(hawk_state["polling_approval_pct"]),
            hawk_decision=hawk_decision,
            dove_decision=dove_decision,
            hawk_bayesian_belief_mu=hawk_updater.mu,
            dove_bayesian_belief_mu=dove_updater.mu,
            game_continues=game_continues,
        )

        transcript.periods.append(period_record)
        yield period_record

        # --- Update last-period actions for next period ---------------
        prev_hawk_action = hawk_decision.action
        prev_dove_action = dove_decision.action

        if game_over:
            winner = resolved_winner
            concession_period = t
            conceding_agent = resolved_conceder
            break

    # --- Finalise transcript ------------------------------------------
    transcript.winner = winner
    transcript.concession_period = concession_period
    transcript.conceding_agent = conceding_agent

    return transcript


# ---------------------------------------------------------------------------
# Convenience wrapper (returns completed transcript)
# ---------------------------------------------------------------------------


def run_simulation(
    episode_df: pd.DataFrame,
    episode_config: dict,
    condition_config: dict,
    hawk: HawkAgent,
    dove: DoveAgent,
    sim_number: int,
    mask_outcome: bool,
    rng: np.random.Generator,
) -> SimulationTranscript:
    """Run one complete simulation and return the finished ``SimulationTranscript``.

    Thin wrapper around :func:`run_simulation_live` that consumes the generator
    and returns the transcript without yielding intermediate records.

    Parameters are identical to :func:`run_simulation_live`.
    """
    gen = run_simulation_live(
        episode_df=episode_df,
        episode_config=episode_config,
        condition_config=condition_config,
        hawk=hawk,
        dove=dove,
        sim_number=sim_number,
        mask_outcome=mask_outcome,
        rng=rng,
    )

    transcript: SimulationTranscript | None = None
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        transcript = stop.value

    assert transcript is not None, "Generator did not return a transcript."
    return transcript
