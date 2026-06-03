"""Tests for the game engine (no LLM calls — mock agents)."""
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agents.base_agent import AgentDecision
from src.simulation.transcript import SimulationTranscript


def _make_episode_df(n_periods=30):
    """Build a minimal episode DataFrame for testing."""
    rows = []
    for t in range(n_periods):
        rows.append({
            "episode_id": "2011",
            "period": t,
            "date": f"2011-07-{3+t:02d}",
            "days_to_xdate": 30 - t,
            "vix": 25.0 + t * 0.5,
            "tbill_4wk": 0.05,
            "market_stress_index": 0.3 + t * 0.01,
            "polling_approval_pct": 44.0,
            "debt_gdp_ratio": 96.3,
            "deficit_projection_bn": 1300.0,
            "election_days_out": 459 - t,
        })
    return pd.DataFrame(rows)


def _make_decision(period, action="HOLD", prob=0.1):
    return AgentDecision(
        period=period,
        agent_role="HAWK",
        action=action,
        reasoning="test",
        concession_probability=prob,
        delay_cost_implied=3.0,
        belief_opponent_delay_cost=4.0,
        public_statement="Test statement.",
        cached=False,
    )


class TestTranscript:
    def test_round_trip_jsonl(self, tmp_path):
        from src.simulation.transcript import PeriodRecord
        hawk_d = _make_decision(0, "HOLD")
        dove_d = AgentDecision(
            period=0, agent_role="DOVE", action="HOLD",
            reasoning="r", concession_probability=0.2,
            delay_cost_implied=5.0, belief_opponent_delay_cost=3.0,
            public_statement="dove says", cached=False,
        )
        rec = PeriodRecord(
            period=0, date="2011-07-03", days_to_xdate=30,
            vix=25.0, market_stress_index=0.3, polling_approval_pct=44.0,
            hawk_decision=hawk_d, dove_decision=dove_d,
            hawk_bayesian_belief_mu=0.12, dove_bayesian_belief_mu=0.14,
            game_continues=True,
        )
        transcript = SimulationTranscript(
            sim_id="test_0", episode_id="2011", condition_id="C",
            sim_number=0, temperature=0.85, mask_historical_outcome=False,
        )
        transcript.periods.append(rec)
        transcript.winner = "DOVE"
        transcript.concession_period = 0
        transcript.conceding_agent = "HAWK"

        path = tmp_path / "sim.jsonl"
        transcript.to_jsonl(path)
        loaded = SimulationTranscript.from_jsonl(path)

        assert loaded.sim_id == "test_0"
        assert loaded.winner == "DOVE"
        assert len(loaded.periods) == 1
        assert loaded.periods[0].hawk_decision.action == "HOLD"

    def test_game_ends_on_concede(self, tmp_path):
        """Engine should stop when any agent concedes."""
        episode_df = _make_episode_df()
        episode_cfg = {
            "episode_id": "2011", "mask_historical_outcome": False,
            "is_fictional": False, "election_days_out": 459,
            "debt_gdp_ratio_at_t0": 96.3, "deficit_projection_bn": 1300,
            "polling": {"obama_approval_pct": 44},
        }
        condition_cfg = {
            "condition_id": "C", "vix_multiplier": 1.4,
            "tbill_spike_bps": 40, "market_stress_floor": 0.65,
            "polls_approval_offset": -8, "narrative_injection": "Test narrative.",
        }

        # Mock agents: period 5 → HAWK concedes
        mock_hawk = MagicMock()
        mock_dove = MagicMock()

        def hawk_act(state, ep, cond):
            action = "CONCEDE" if state["period"] >= 5 else "HOLD"
            return _make_decision(state["period"], action, 0.9 if action == "CONCEDE" else 0.1)

        mock_hawk.act.side_effect = hawk_act
        mock_hawk.role = "HAWK"
        mock_dove.act.return_value = _make_decision(0, "HOLD", 0.05)
        mock_dove.act.side_effect = lambda state, ep, cond: _make_decision(state["period"], "HOLD", 0.05)
        mock_dove.role = "DOVE"

        from src.simulation.game_engine import run_simulation
        rng = np.random.default_rng(42)
        transcript = run_simulation(
            episode_df=episode_df, episode_config=episode_cfg,
            condition_config=condition_cfg, hawk=mock_hawk, dove=mock_dove,
            sim_number=0, mask_outcome=False, rng=rng,
        )

        assert transcript.concession_period == 5
        assert transcript.conceding_agent == "HAWK"
        assert transcript.winner == "DOVE"
        assert len(transcript.periods) == 6  # periods 0..5

    def test_censored_after_30(self, tmp_path):
        episode_df = _make_episode_df()
        episode_cfg = {
            "episode_id": "2011", "mask_historical_outcome": False, "is_fictional": False,
            "election_days_out": 459, "debt_gdp_ratio_at_t0": 96.3,
            "deficit_projection_bn": 1300, "polling": {"obama_approval_pct": 44},
        }
        condition_cfg = {
            "condition_id": "A", "vix_multiplier": 0.7,
            "tbill_spike_bps": 0, "market_stress_floor": 0.1,
            "polls_approval_offset": 3, "narrative_injection": "",
        }
        mock_hawk = MagicMock()
        mock_dove = MagicMock()
        mock_hawk.act.side_effect = lambda s, e, c: _make_decision(s["period"], "HOLD", 0.01)
        mock_dove.act.side_effect = lambda s, e, c: AgentDecision(
            period=s["period"], agent_role="DOVE", action="HOLD", reasoning="",
            concession_probability=0.01, delay_cost_implied=2.0,
            belief_opponent_delay_cost=3.0, public_statement="holding.", cached=False,
        )
        mock_hawk.role = "HAWK"
        mock_dove.role = "DOVE"

        from src.simulation.game_engine import run_simulation
        rng = np.random.default_rng(1)
        transcript = run_simulation(
            episode_df=episode_df, episode_config=episode_cfg,
            condition_config=condition_cfg, hawk=mock_hawk, dove=mock_dove,
            sim_number=0, mask_outcome=False, rng=rng,
        )
        assert transcript.winner == "CENSORED"
        assert transcript.concession_period == -1
