"""Runs all (episode, condition) combinations with budget tracking."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd

from src.cache.llm_cache import LLMCache
from src.simulation.condition_runner import run_condition
from src.simulation.transcript import SimulationTranscript, PeriodRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPISODE_IDS: list[str] = ["2011", "2013", "2023", "2025_counterfactual"]
CONDITION_IDS: list[str] = ["A", "B", "C", "D", "E"]

# Anthropic claude-haiku-4-5 pricing (USD per million tokens, as of mid-2025).
# Used for rough budget estimation; not authoritative.
_INPUT_COST_PER_M: float = 0.25
_OUTPUT_COST_PER_M: float = 1.25

_DEFAULT_BUDGET_USD: float = 100.0
_DEFAULT_OUTPUT_DIR: Path = Path("outputs")
_DEFAULT_CACHE_PATH: Path = Path("cache") / "llm_cache.db"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when the estimated cumulative API spend exceeds the configured budget."""

    def __init__(self, spent_usd: float, budget_usd: float) -> None:
        self.spent_usd = spent_usd
        self.budget_usd = budget_usd
        super().__init__(
            f"Budget exceeded: estimated spend ${spent_usd:.4f} > budget ${budget_usd:.2f}"
        )


# ---------------------------------------------------------------------------
# Budget estimation helper
# ---------------------------------------------------------------------------


def _estimate_cost_usd(cache: LLMCache) -> float:
    """Return a rough estimate of cumulative API spend from cache statistics.

    Only tokens that were actually sent to the API (i.e. cache misses, one call
    per unique cache row) count toward cost.  Cache hits are free.
    """
    stats = cache.stats()
    input_m = stats["total_input_tokens_spent"] / 1_000_000
    output_m = stats["total_output_tokens_spent"] / 1_000_000
    return input_m * _INPUT_COST_PER_M + output_m * _OUTPUT_COST_PER_M


# ---------------------------------------------------------------------------
# Result compilation
# ---------------------------------------------------------------------------


def compile_results(
    transcripts_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all JSONL transcripts and compile sim-level and period-level DataFrames.

    Parameters
    ----------
    transcripts_dir:
        Directory containing ``*.jsonl`` transcript files.
    output_dir:
        Directory where ``simulation_results.parquet`` and
        ``period_level_data.parquet`` will be written.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(sim_level_df, period_level_df)``
    """
    jsonl_files = sorted(transcripts_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL transcripts found in %s", transcripts_dir)
        return pd.DataFrame(), pd.DataFrame()

    sim_rows: list[dict] = []
    period_rows: list[dict] = []

    for jsonl_path in jsonl_files:
        try:
            transcript = SimulationTranscript.from_jsonl(jsonl_path)
        except Exception:
            logger.exception("Failed to load transcript: %s", jsonl_path)
            continue

        if not transcript.periods:
            continue

        # --- Sim-level aggregates ----------------------------------------
        hawk_delay_costs: list[float] = [
            p.hawk_decision.delay_cost_implied for p in transcript.periods
        ]
        dove_delay_costs: list[float] = [
            p.dove_decision.delay_cost_implied for p in transcript.periods
        ]
        hawk_concession_probs: list[float] = [
            p.hawk_decision.concession_probability for p in transcript.periods
        ]
        dove_concession_probs: list[float] = [
            p.dove_decision.concession_probability for p in transcript.periods
        ]
        final_period: PeriodRecord = transcript.periods[-1]

        sim_rows.append(
            {
                "sim_id": transcript.sim_id,
                "episode_id": transcript.episode_id,
                "condition_id": transcript.condition_id,
                "sim_number": transcript.sim_number,
                "winner": transcript.winner,
                "concession_period": transcript.concession_period,
                "mask_historical_outcome": transcript.mask_historical_outcome,
                "hawk_mean_delay_cost": sum(hawk_delay_costs) / len(hawk_delay_costs),
                "dove_mean_delay_cost": sum(dove_delay_costs) / len(dove_delay_costs),
                "hawk_mean_concession_prob": (
                    sum(hawk_concession_probs) / len(hawk_concession_probs)
                ),
                "dove_mean_concession_prob": (
                    sum(dove_concession_probs) / len(dove_concession_probs)
                ),
                "final_market_stress": final_period.market_stress_index,
                "temperature": transcript.temperature,
            }
        )

        # --- Period-level rows -------------------------------------------
        for period_record in transcript.periods:
            period_rows.append(
                {
                    "sim_id": transcript.sim_id,
                    "episode_id": transcript.episode_id,
                    "condition_id": transcript.condition_id,
                    "period": period_record.period,
                    "days_to_xdate": period_record.days_to_xdate,
                    "market_stress_index": period_record.market_stress_index,
                    "hawk_action": period_record.hawk_decision.action,
                    "dove_action": period_record.dove_decision.action,
                    "hawk_concession_prob": period_record.hawk_decision.concession_probability,
                    "dove_concession_prob": period_record.dove_decision.concession_probability,
                    "hawk_delay_cost": period_record.hawk_decision.delay_cost_implied,
                    "dove_delay_cost": period_record.dove_decision.delay_cost_implied,
                    "hawk_belief_opponent": period_record.hawk_decision.belief_opponent_delay_cost,
                    "dove_belief_opponent": period_record.dove_decision.belief_opponent_delay_cost,
                    "hawk_bayesian_mu": period_record.hawk_bayesian_belief_mu,
                    "dove_bayesian_mu": period_record.dove_bayesian_belief_mu,
                }
            )

    sim_level_df = pd.DataFrame(sim_rows)
    period_level_df = pd.DataFrame(period_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not sim_level_df.empty:
        sim_level_df.to_parquet(
            output_dir / "simulation_results.parquet", index=False
        )
        logger.info(
            "Wrote simulation_results.parquet (%d rows)", len(sim_level_df)
        )
    if not period_level_df.empty:
        period_level_df.to_parquet(
            output_dir / "period_level_data.parquet", index=False
        )
        logger.info(
            "Wrote period_level_data.parquet (%d rows)", len(period_level_df)
        )

    return sim_level_df, period_level_df


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------


def run_all(
    n_sims_per_condition: int = 25,
    output_dir: Optional[Path] = None,
    cache: Optional[LLMCache] = None,
) -> pd.DataFrame:
    """Run all (episode, condition) combinations and return simulation results.

    Runs 4 episodes × 5 conditions = 20 standard combinations, plus 20 masked
    variants (``mask_historical_outcome=True``), for 40 total condition runs.
    Each episode's 5 conditions are parallelised with a
    ``ThreadPoolExecutor(max_workers=4)``; episodes run sequentially to bound
    memory pressure.

    Parameters
    ----------
    n_sims_per_condition:
        Number of independent simulations per (episode, condition) pair.
        Default 25.
    output_dir:
        Root output directory.  Defaults to ``outputs/``.
    cache:
        ``LLMCache`` to use.  If ``None``, a default cache at
        ``cache/llm_cache.db`` is created.

    Returns
    -------
    pd.DataFrame
        Simulation-level results (one row per sim).

    Raises
    ------
    BudgetExceededError
        When estimated API spend exceeds ``MAX_API_BUDGET_USD`` (env var,
        default $100).
    """
    resolved_output_dir: Path = (
        output_dir if output_dir is not None else _DEFAULT_OUTPUT_DIR
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    budget_usd: float = float(
        os.environ.get("MAX_API_BUDGET_USD", str(_DEFAULT_BUDGET_USD))
    )

    owns_cache = cache is None
    if owns_cache:
        cache = LLMCache(db_path=_DEFAULT_CACHE_PATH)

    all_transcripts: list[SimulationTranscript] = []

    try:
        for episode_id in EPISODE_IDS:
            for masked in (False, True):
                variant_label = f"{episode_id}{'_masked' if masked else ''}"
                logger.info("Starting episode variant: %s", variant_label)

                # Run all 5 conditions for this (episode, masked) variant in
                # parallel using up to 4 workers.
                futures_map: dict = {}
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for condition_id in CONDITION_IDS:
                        future = executor.submit(
                            _run_condition_with_mask,
                            episode_id=episode_id,
                            condition_id=condition_id,
                            n_sims=n_sims_per_condition,
                            cache=cache,
                            output_dir=resolved_output_dir,
                            masked=masked,
                        )
                        futures_map[future] = (episode_id, condition_id, masked)

                    for future in as_completed(futures_map):
                        ep_id, cond_id, msk = futures_map[future]
                        try:
                            condition_transcripts = future.result()
                        except Exception:
                            logger.exception(
                                "Condition run failed: episode=%s condition=%s masked=%s",
                                ep_id,
                                cond_id,
                                msk,
                            )
                            raise

                        all_transcripts.extend(condition_transcripts)

                        # --- Budget check after each completed condition ---
                        spent = _estimate_cost_usd(cache)
                        logger.info(
                            "Completed episode=%s condition=%s masked=%s | "
                            "cumulative estimated cost: $%.4f / $%.2f",
                            ep_id,
                            cond_id,
                            msk,
                            spent,
                            budget_usd,
                        )
                        if spent > budget_usd:
                            raise BudgetExceededError(
                                spent_usd=spent, budget_usd=budget_usd
                            )

    finally:
        if owns_cache:
            cache.close()

    # --- Compile results -------------------------------------------------
    transcripts_dir = resolved_output_dir / "transcripts"
    sim_level_df, _ = compile_results(
        transcripts_dir=transcripts_dir,
        output_dir=resolved_output_dir / "results",
    )

    return sim_level_df


# ---------------------------------------------------------------------------
# Internal wrapper that injects mask_historical_outcome into episode_config
# ---------------------------------------------------------------------------


def _run_condition_with_mask(
    episode_id: str,
    condition_id: str,
    n_sims: int,
    cache: LLMCache,
    output_dir: Path,
    masked: bool,
) -> list[SimulationTranscript]:
    """Run one condition, patching episode_config with ``mask_historical_outcome``.

    When *masked* is ``True`` the episode YAML's ``mask_historical_outcome``
    field is overridden to ``True``, and transcript sim_ids get an ``_masked``
    suffix to avoid collisions with the unmasked run.

    This wrapper exists so the batch runner can submit masked and unmasked runs
    without duplicating the full ``run_condition`` logic.
    """
    import yaml  # local import to keep module-level imports minimal

    episode_yaml_path = (
        Path("configs") / "episodes" / f"{episode_id}.yaml"
    )
    with episode_yaml_path.open("r", encoding="utf-8") as fh:
        episode_config: dict = yaml.safe_load(fh)

    if masked:
        episode_config["mask_historical_outcome"] = True

    # Temporarily write a patched YAML to a temp file, OR — simpler —
    # call run_condition which reloads from disk and we patch after.
    # Actually, run_condition loads from disk independently.  Instead we
    # use a slightly different approach: duplicate run_condition logic for
    # the masked path using a thin adapter.
    return _run_condition_direct(
        episode_id=episode_id,
        condition_id=condition_id,
        n_sims=n_sims,
        cache=cache,
        output_dir=output_dir,
        episode_config=episode_config,
        masked=masked,
    )


def _run_condition_direct(
    episode_id: str,
    condition_id: str,
    n_sims: int,
    cache: LLMCache,
    output_dir: Path,
    episode_config: dict,
    masked: bool,
) -> list[SimulationTranscript]:
    """Low-level condition runner that accepts an already-loaded episode_config.

    Avoids re-loading YAML from disk (which would lose the masked override) while
    still reusing all the per-sim logic from ``run_condition``.
    """
    import hashlib as _hashlib

    import numpy as np
    import pandas as pd
    import yaml
    from tqdm import tqdm

    from src.agents.hawk_agent import HawkAgent
    from src.agents.dove_agent import DoveAgent
    from src.simulation.condition_runner import _stable_seed
    from src.simulation.game_engine import run_simulation

    condition_yaml_path = (
        Path("configs") / "conditions" / f"condition_{condition_id}.yaml"
    )
    with condition_yaml_path.open("r", encoding="utf-8") as fh:
        condition_config: dict = yaml.safe_load(fh)

    episode_parquet = Path("data") / "processed" / f"episode_{episode_id}.parquet"

    episode_df: pd.DataFrame = pd.read_parquet(episode_parquet)
    if "period" in episode_df.columns:
        episode_df = episode_df.sort_values("period").reset_index(drop=True)

    mask_suffix = "_masked" if masked else ""
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    base_temperature: float = 0.85
    mask_outcome: bool = bool(episode_config.get("mask_historical_outcome", False))

    transcripts: list[SimulationTranscript] = []
    desc = f"episode={episode_id}{mask_suffix} cond={condition_id}"

    for sim_i in tqdm(range(n_sims), desc=desc, unit="sim"):
        # Clamp at 1.0: Anthropic's API rejects temperature > 1.0 with a 400 error.
        temp = min(1.0, base_temperature + (sim_i / n_sims) * 0.2)

        hawk = HawkAgent(cache=cache, temperature=temp)
        dove = DoveAgent(cache=cache, temperature=temp)

        # Use a seed that also incorporates the masked flag so masked/unmasked
        # runs produce different (but reproducible) RNG streams.
        if masked:
            raw = f"{episode_id}_{condition_id}_{sim_i}_masked".encode("utf-8")
            seed = int.from_bytes(_hashlib.sha256(raw).digest()[:4], "big") % (2**31)
        else:
            seed = _stable_seed(episode_id, condition_id, sim_i)
        rng = np.random.default_rng(seed=seed)

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

        # Append mask suffix to sim_id to avoid file-name collisions.
        if masked:
            transcript.sim_id = transcript.sim_id + "_masked"

        jsonl_path = transcripts_dir / f"{transcript.sim_id}.jsonl"
        transcript.to_jsonl(jsonl_path)
        logger.debug("Saved transcript: %s", jsonl_path)
        transcripts.append(transcript)

    return transcripts
