"""
Alesina-Drazen (1991) war of attrition model.

Each faction draws delay cost rate λ ~ Exp(μ). The faction with the HIGHER
delay cost concedes first. In symmetric equilibrium with exponential priors,
the hazard rate is constant at μ (memoryless property of the exponential).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import optimize, stats

logger = logging.getLogger(__name__)


@dataclass
class ADParameters:
    mu_hawk: float
    mu_dove: float


@dataclass
class ADResults:
    mu_hawk: float
    mu_dove: float
    p_hawk_concedes: float   # probability HAWK concedes first (higher rate → concedes sooner)
    p_dove_concedes: float
    expected_concession_period: float
    joint_rate: float        # mu_hawk + mu_dove (combined race hazard)

    def hazard(self, t: float) -> float:
        """Instantaneous concession hazard at period t (constant for exponential)."""
        return self.joint_rate

    def survival(self, t: float) -> float:
        """P(neither has conceded by period t)."""
        return math.exp(-self.joint_rate * t)

    def cdf(self, t: float) -> float:
        """P(at least one has conceded by period t)."""
        return 1.0 - self.survival(t)


def symmetric_model(mu: float) -> ADResults:
    """Both factions have same delay cost rate μ.

    Under symmetric equilibrium, each draws λ ~ Exp(μ). The faction with the
    higher draw concedes first. The joint process is Exp(2μ).
    """
    return ADResults(
        mu_hawk=mu,
        mu_dove=mu,
        p_hawk_concedes=0.5,
        p_dove_concedes=0.5,
        expected_concession_period=1.0 / (2.0 * mu),
        joint_rate=2.0 * mu,
    )


def asymmetric_model(mu_hawk: float, mu_dove: float) -> ADResults:
    """General asymmetric case.

    P(HAWK concedes) = mu_hawk / (mu_hawk + mu_dove) — the agent with the
    higher rate (shorter expected survival) concedes first with higher probability.
    """
    joint = mu_hawk + mu_dove
    return ADResults(
        mu_hawk=mu_hawk,
        mu_dove=mu_dove,
        p_hawk_concedes=mu_hawk / joint,
        p_dove_concedes=mu_dove / joint,
        expected_concession_period=1.0 / joint,
        joint_rate=joint,
    )


def theoretical_concession_times(
    n_samples: int,
    mu_hawk: float,
    mu_dove: float,
    rng: np.random.Generator,
    max_period: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n_samples draws from competing Exp processes.

    Returns (hawk_times, dove_times) where each value is the period at which
    that faction would concede if uncensored. Both are capped at max_period.
    """
    hawk_times = rng.exponential(scale=1.0 / mu_hawk, size=n_samples)
    dove_times = rng.exponential(scale=1.0 / mu_dove, size=n_samples)
    hawk_times = np.clip(hawk_times, 0, max_period)
    dove_times = np.clip(dove_times, 0, max_period)
    return hawk_times, dove_times


def compute_hazard_surface(
    t_grid: np.ndarray,
    stress_grid: np.ndarray,
    mu_base: float = 0.12,
    stress_sensitivity: float = 0.25,
) -> np.ndarray:
    """Compute theoretical concession probability surface for the 3D visualization.

    Axes match the user-specified hero figure:
      t_grid:     days_to_xdate (0..30). Lower = more time has elapsed = higher prob.
      stress_grid: market stress [0, 1].
      Z[i, j]:    cumulative concession probability given t_grid[i] days remain
                  and stress_grid[j] market stress level.

    Formula:
        effective_mu = mu_base * (1 + stress_sensitivity * stress)
        time_elapsed = 30 - days_to_xdate
        Z = 1 - exp(-effective_mu * time_elapsed)
    """
    mu_eff = mu_base * (1.0 + stress_sensitivity * stress_grid)      # shape (S,)
    time_elapsed = (30.0 - t_grid)[:, None]                          # shape (T, 1)
    Z = 1.0 - np.exp(-mu_eff[None, :] * time_elapsed)                # shape (T, S)
    return np.clip(Z, 0.0, 1.0)


def calibrate_mu(
    concession_times: list[float],
    max_period: int = 30,
) -> float:
    """MLE estimate of μ from observed concession times with right-censoring.

    Observations < max_period: full events (contribute Exp PDF).
    Observations == max_period: right-censored (contribute Exp survival).
    """
    times = np.asarray(concession_times, dtype=float)
    events = times < max_period

    n_events = events.sum()
    total_time = times.sum()

    if n_events == 0:
        logger.warning("calibrate_mu: all observations censored — returning minimal rate")
        return 1e-4

    # Closed-form starting point: events / total_time
    mu0 = n_events / total_time

    def neg_log_lik(log_mu: float) -> float:
        mu = math.exp(log_mu)
        ll = (
            n_events * log_mu                          # log(mu) per event
            - mu * total_time                          # survival contribution
        )
        return -ll

    result = optimize.minimize_scalar(
        neg_log_lik,
        bounds=(math.log(1e-4), math.log(10.0)),
        method="bounded",
    )
    return float(math.exp(result.x))


if __name__ == "__main__":
    import sys

    print("=== Alesina-Drazen Model Demo ===\n")

    r = asymmetric_model(mu_hawk=0.18, mu_dove=0.12)
    print(f"Asymmetric model: mu_H={r.mu_hawk}, mu_D={r.mu_dove}")
    print(f"  P(HAWK concedes): {r.p_hawk_concedes:.3f}")
    print(f"  P(DOVE concedes): {r.p_dove_concedes:.3f}")
    print(f"  Expected concession period: {r.expected_concession_period:.2f}")
    print(f"  Survival at t=10: {r.survival(10):.3f}")
    print(f"  CDF at t=10: {r.cdf(10):.3f}")
    print()

    rng = np.random.default_rng(42)
    h_times, d_times = theoretical_concession_times(1000, 0.18, 0.12, rng)
    joint_times = np.minimum(h_times, d_times)

    mu_hat = calibrate_mu(joint_times.tolist(), max_period=30)
    print(f"Calibrated mu from 1000 simulated draws: {mu_hat:.4f} (true joint: {0.18+0.12:.4f})")

    uncensored = joint_times[joint_times < 30]
    if len(uncensored) > 10:
        ks, p = stats.kstest(uncensored, "expon", args=(0, 1.0 / mu_hat))
        print(f"KS test p-value: {p:.4f} ({'pass' if p > 0.05 else 'reject'})")
    print("\nDone.")
