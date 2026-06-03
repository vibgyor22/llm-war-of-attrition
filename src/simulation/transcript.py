"""Transcript data structures and serialization."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from src.agents.base_agent import AgentDecision


@dataclass
class PeriodRecord:
    """One period's data for both agents."""

    period: int
    date: str
    days_to_xdate: int
    vix: float
    market_stress_index: float
    polling_approval_pct: float
    hawk_decision: AgentDecision
    dove_decision: AgentDecision
    hawk_bayesian_belief_mu: float    # researcher-side Bayesian estimate of HAWK's delay cost
    dove_bayesian_belief_mu: float    # same for DOVE
    game_continues: bool


@dataclass
class SimulationTranscript:
    """Complete record for one simulation run."""

    sim_id: str                # e.g. "2011_C_sim003"
    episode_id: str
    condition_id: str
    sim_number: int
    temperature: float
    mask_historical_outcome: bool
    periods: list[PeriodRecord] = field(default_factory=list)
    winner: str = ""           # "HAWK", "DOVE", or "CENSORED"
    concession_period: int = -1    # -1 if censored
    conceding_agent: str = ""      # who conceded

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_jsonl(self, path: Path) -> None:
        """Write transcript to JSONL.

        Line 1: metadata dict (all scalar fields).
        Lines 2+: one PeriodRecord per line, serialised with ``asdict``.

        ``AgentDecision`` is itself a dataclass so ``asdict`` recurses into it
        correctly without any custom handling.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata: dict = {
            "sim_id": self.sim_id,
            "episode_id": self.episode_id,
            "condition_id": self.condition_id,
            "sim_number": self.sim_number,
            "temperature": self.temperature,
            "mask_historical_outcome": self.mask_historical_outcome,
            "winner": self.winner,
            "concession_period": self.concession_period,
            "conceding_agent": self.conceding_agent,
        }

        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            for period_record in self.periods:
                fh.write(
                    json.dumps(asdict(period_record), ensure_ascii=False) + "\n"
                )

    @classmethod
    def from_jsonl(cls, path: Path) -> "SimulationTranscript":
        """Load a ``SimulationTranscript`` from a JSONL file written by ``to_jsonl``.

        Reconstructs ``AgentDecision`` and ``PeriodRecord`` dataclasses from
        their dict representations.
        """
        with path.open("r", encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh if ln.strip()]

        if not lines:
            raise ValueError(f"Empty transcript file: {path}")

        meta: dict = json.loads(lines[0])

        transcript = cls(
            sim_id=meta["sim_id"],
            episode_id=meta["episode_id"],
            condition_id=meta["condition_id"],
            sim_number=int(meta["sim_number"]),
            temperature=float(meta["temperature"]),
            mask_historical_outcome=bool(meta["mask_historical_outcome"]),
            winner=meta.get("winner", ""),
            concession_period=int(meta.get("concession_period", -1)),
            conceding_agent=meta.get("conceding_agent", ""),
        )

        for line in lines[1:]:
            rec_dict: dict = json.loads(line)

            hawk_dec = AgentDecision(**rec_dict["hawk_decision"])
            dove_dec = AgentDecision(**rec_dict["dove_decision"])

            period_record = PeriodRecord(
                period=int(rec_dict["period"]),
                date=str(rec_dict["date"]),
                days_to_xdate=int(rec_dict["days_to_xdate"]),
                vix=float(rec_dict["vix"]),
                market_stress_index=float(rec_dict["market_stress_index"]),
                polling_approval_pct=float(rec_dict["polling_approval_pct"]),
                hawk_decision=hawk_dec,
                dove_decision=dove_dec,
                hawk_bayesian_belief_mu=float(rec_dict["hawk_bayesian_belief_mu"]),
                dove_bayesian_belief_mu=float(rec_dict["dove_bayesian_belief_mu"]),
                game_continues=bool(rec_dict["game_continues"]),
            )
            transcript.periods.append(period_record)

        return transcript
