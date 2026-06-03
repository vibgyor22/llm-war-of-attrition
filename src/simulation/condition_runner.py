"""Runs N simulations for one (episode, condition) combination."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.agents.hawk_agent import HawkAgent
from src.agents.dove_agent import DoveAgent
from src.cache.llm_cache import LLMCache
from src.simulation.game_engine import run_simulation
from src.simulation.transcript import SimulationTranscript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stable_seed(episode_id: str, condition_id: str, sim_i: int) -> int:
    """Return a deterministic 31-bit seed from the given identifiers.

    Uses SHA-256 (stable across processes) rather than Python's built-in
    ``hash()`` which is non-deterministic when PYTHONHASHSEED is randomised
    (the default since Python 3.3).
    """
    raw = f"{episode_id}_{condition_id}_{sim_i}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    # Take the first 4 bytes as a big-endian uint32, then mask to 31 bits.
    return int.from_bytes(digest[:4], "big") % (2**31)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_condition(
    episode_id: str,
    condition_id: str,
    n_sims: int,
    cache: LLMCache,
    output_dir: Path,
    base_temperature: float = 0.85,
) -> list[SimulationTranscript]:
    """Run *n_sims* simulations for one (episode, condition) combination.

    Parameters
    ----------
    episode_id:
        One of ``"2011"``, ``"2013"``, ``"2023"``, ``"2025_counterfactual"``.
    condition_id:
        One of ``"A"``–``"E"``.
    n_sims:
        Number of independent simulations to run.
    cache:
        Shared ``LLMCache`` instance.
    output_dir:
        Root output directory.  Transcripts are saved to
        ``<output_dir>/transcripts/``.
    base_temperature:
        Base sampling temperature.  Temperature for sim *i* is
        ``base_temperature + (i / n_sims) * 0.2``, spanning
        ``[base_temperature, base_temperature + 0.2)``.

    Returns
    -------
    list[SimulationTranscript]
        All completed transcripts, in sim-index order.
    """
    # --- Load data -------------------------------------------------------
    episode_parquet = Path("data") / "processed" / f"episode_{episode_id}.parquet"
    episode_yaml_path = Path("configs") / "episodes" / f"{episode_id}.yaml"
    condition_yaml_path = (
        Path("configs") / "conditions" / f"condition_{condition_id}.yaml"
    )

    episode_df: pd.DataFrame = pd.read_parquet(episode_parquet)
    # Ensure the DataFrame has exactly MAX_PERIODS rows, sorted by period.
    if "period" in episode_df.columns:
        episode_df = episode_df.sort_values("period").reset_index(drop=True)

    with episode_yaml_path.open("r", encoding="utf-8") as fh:
        episode_config: dict = yaml.safe_load(fh)

    with condition_yaml_path.open("r", encoding="utf-8") as fh:
        condition_config: dict = yaml.safe_load(fh)

    # --- Prepare output directory ----------------------------------------
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # --- Run simulations --------------------------------------------------
    transcripts: list[SimulationTranscript] = []
    mask_outcome: bool = bool(episode_config.get("mask_historical_outcome", False))

    desc = f"episode={episode_id} cond={condition_id}"
    for sim_i in tqdm(range(n_sims), desc=desc, unit="sim"):
        # Spec range is base..base+0.2 but Anthropic's API rejects temperature > 1.0.
        # Clamp at 1.0 to avoid 400 errors on the final few simulations.
        temp = min(1.0, base_temperature + (sim_i / n_sims) * 0.2)

        # Create fresh agents per simulation (temperature differs per sim).
        hawk = HawkAgent(cache=cache, temperature=temp)
        dove = DoveAgent(cache=cache, temperature=temp)

        seed = _stable_seed(episode_id, condition_id, sim_i)
        rng = np.random.default_rng(seed=seed)

        try:
            transcript = run_simulation(
                episode_df=episode_df,
                episode_config=episode_config,
                condition_config=condition_config,
                hawk=hawk,
                dove=dove,
                sim_number=sim_i,
                mask_outcome=mask_outcome,
                rng=rng,
            )
        except Exception:
            logger.exception(
                "Simulation failed: episode=%s condition=%s sim=%d",
                episode_id,
                condition_id,
                sim_i,
            )
            raise

        # Save transcript immediately so partial runs are recoverable.
        jsonl_path = transcripts_dir / f"{transcript.sim_id}.jsonl"
        transcript.to_jsonl(jsonl_path)
        logger.debug("Saved transcript: %s", jsonl_path)

        transcripts.append(transcript)

    return transcripts
