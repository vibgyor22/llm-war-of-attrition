"""Bayesian belief updater for agent delay costs.

Uses a Normal-Normal conjugate model:
  - Prior:       mu_0 ~ N(prior_mu, prior_sigma^2)
  - Likelihood:  x_t | mu ~ N(mu, obs_sigma^2)
  - Posterior:   mu | x_{1..t} is also Normal (closed-form update)

The observations are the *delay_cost_implied* values (0–10 scale) reported
by an agent each period.  A researcher-side tracker maintains a posterior over
the agent's true latent delay cost — updated after every period.

Usage::

    updater = BayesianBeliefUpdater()
    updater.update(3.2)   # after period 0 observation
    updater.update(4.7)   # after period 1 observation
    print(updater.mu)     # posterior mean
    print(updater.sigma)  # posterior standard deviation
"""

from __future__ import annotations

import math


class BayesianBeliefUpdater:
    """Normal-Normal conjugate updater for a latent delay-cost parameter.

    Parameters
    ----------
    prior_mu:
        Prior mean of the latent delay cost (default 5.0 — midpoint of 0-10).
    prior_sigma:
        Prior standard deviation (default 3.0 — diffuse over [0, 10]).
    obs_sigma:
        Standard deviation of each noisy observation (default 2.0).
        Reflects reporting noise / strategic misrepresentation in the agent's
        ``delay_cost_implied`` field.
    """

    def __init__(
        self,
        prior_mu: float = 5.0,
        prior_sigma: float = 3.0,
        obs_sigma: float = 2.0,
    ) -> None:
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be positive.")
        if obs_sigma <= 0:
            raise ValueError("obs_sigma must be positive.")

        # Store originals so reset() can return to prior cleanly.
        self._prior_mu: float = float(prior_mu)
        self._prior_var: float = float(prior_sigma) ** 2

        self._mu: float = self._prior_mu
        self._var: float = self._prior_var          # posterior variance
        self._obs_var: float = float(obs_sigma) ** 2  # fixed observation variance
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mu(self) -> float:
        """Current posterior mean of the latent delay cost."""
        return self._mu

    @property
    def sigma(self) -> float:
        """Current posterior standard deviation."""
        return math.sqrt(self._var)

    @property
    def n_updates(self) -> int:
        """Number of observations incorporated so far."""
        return self._n_updates

    def update(self, observed_value: float) -> None:
        """Incorporate a new noisy observation and update the posterior.

        Parameters
        ----------
        observed_value:
            The ``delay_cost_implied`` value reported by the agent this period
            (expected range 0–10).
        """
        # Normal-Normal conjugate update (known observation variance):
        #   posterior_var  = 1 / (1/prior_var + 1/obs_var)
        #   posterior_mean = posterior_var * (prior_mean/prior_var + obs/obs_var)
        prior_precision = 1.0 / self._var
        obs_precision = 1.0 / self._obs_var

        posterior_precision = prior_precision + obs_precision
        posterior_var = 1.0 / posterior_precision
        posterior_mu = posterior_var * (
            prior_precision * self._mu + obs_precision * float(observed_value)
        )

        self._mu = posterior_mu
        self._var = posterior_var
        self._n_updates += 1

    def reset(self) -> None:
        """Reset to prior — call between simulations to reuse the updater."""
        self._mu = self._prior_mu
        self._var = self._prior_var
        self._n_updates = 0

    def prob_concede_next_period(self) -> float:
        """Heuristic: P(agent concedes next period) given current posterior mean.

        Maps posterior mean (0-10 scale) to [0,1] probability via logistic.
        Higher delay cost → higher concession probability in next period.
        """
        # Logistic: P = 1 / (1 + exp(-(mu - 5) / 2))
        return 1.0 / (1.0 + math.exp(-(self._mu - 5.0) / 2.0))

    def expected_concession_period(self, max_period: int = 30) -> float:
        """Expected period of concession given current posterior mean.

        Higher delay cost → earlier expected concession (inverted mapping).
        """
        p = self.prob_concede_next_period()
        if p <= 0:
            return float(max_period)
        return min(float(max_period), 1.0 / p)

    def summary(self) -> dict:
        """Return a summary dict of current belief state."""
        return {
            "mu": self._mu,
            "uncertainty": self.sigma,
            "n_obs": self._n_updates,
            "implied_cost_high": self._mu > self._prior_mu,
            "implied_cost_low": self._mu < self._prior_mu,
            "prob_concede_next_period": self.prob_concede_next_period(),
            "expected_concession_period": self.expected_concession_period(),
        }
