"""Tests for the Alesina-Drazen theory module."""
import numpy as np
import pytest
from src.theory.alesina_drazen import (
    symmetric_model, asymmetric_model, compute_hazard_surface,
    theoretical_concession_times, calibrate_mu,
)
from src.theory.bayesian_belief import BayesianBeliefUpdater


class TestAlesina:
    def test_symmetric_probabilities(self):
        r = symmetric_model(0.15)
        assert abs(r.p_hawk_concedes - 0.5) < 1e-9
        assert abs(r.p_dove_concedes - 0.5) < 1e-9

    def test_symmetric_joint_rate(self):
        r = symmetric_model(0.15)
        assert abs(r.joint_rate - 0.30) < 1e-9

    def test_asymmetric_higher_rate_concedes(self):
        # HAWK has higher rate → higher probability of conceding
        r = asymmetric_model(mu_hawk=0.3, mu_dove=0.1)
        assert r.p_hawk_concedes > r.p_dove_concedes

    def test_asymmetric_probabilities_sum_one(self):
        r = asymmetric_model(0.2, 0.15)
        assert abs(r.p_hawk_concedes + r.p_dove_concedes - 1.0) < 1e-9

    def test_survival_at_zero(self):
        r = symmetric_model(0.15)
        assert abs(r.survival(0) - 1.0) < 1e-9

    def test_cdf_monotone(self):
        r = symmetric_model(0.12)
        vals = [r.cdf(t) for t in [0, 5, 10, 20, 30]]
        assert all(v1 <= v2 for v1, v2 in zip(vals, vals[1:]))

    def test_hazard_surface_shape(self):
        t = np.linspace(0, 30, 20)
        s = np.linspace(0, 1, 15)
        Z = compute_hazard_surface(t, s)
        assert Z.shape == (20, 15)
        assert Z.min() >= 0.0
        assert Z.max() <= 1.0

    def test_hazard_surface_increases_with_stress(self):
        t = np.array([15.0])
        s = np.linspace(0, 1, 10)
        Z = compute_hazard_surface(t, s)
        assert all(Z[0, i] <= Z[0, i+1] for i in range(len(s)-1))

    def test_theoretical_concession_times_shape(self):
        rng = np.random.default_rng(42)
        h, d = theoretical_concession_times(200, 0.2, 0.15, rng)
        assert len(h) == 200
        assert len(d) == 200
        assert (h >= 0).all()
        assert (h <= 30).all()

    def test_calibrate_mu_roundtrip(self):
        rng = np.random.default_rng(0)
        true_mu = 0.15
        h, d = theoretical_concession_times(500, true_mu, true_mu, rng)
        joint = np.minimum(h, d)
        mu_hat = calibrate_mu(joint.tolist(), max_period=30)
        assert abs(mu_hat - true_mu * 2) < 0.1  # joint rate ≈ 2*mu

    def test_calibrate_mu_all_censored(self):
        # Should not crash, return small value
        times = [30.0] * 50
        mu = calibrate_mu(times, max_period=30)
        assert mu > 0


class TestBayesianBelief:
    def test_high_obs_increases_mu(self):
        """Observing high delay cost shifts posterior mean upward."""
        u = BayesianBeliefUpdater(prior_mu=5.0)
        before = u.mu
        u.update(9.0)   # high observed cost
        assert u.mu > before

    def test_low_obs_decreases_mu(self):
        """Observing low delay cost shifts posterior mean downward."""
        u = BayesianBeliefUpdater(prior_mu=5.0)
        before = u.mu
        u.update(1.0)   # low observed cost
        assert u.mu < before

    def test_sigma_decreases_with_updates(self):
        """Posterior uncertainty should shrink as observations accumulate."""
        u = BayesianBeliefUpdater()
        s0 = u.sigma
        u.update(5.0)
        u.update(5.0)
        assert u.sigma < s0

    def test_reset(self):
        u = BayesianBeliefUpdater(prior_mu=5.0, prior_sigma=3.0)
        u.update(9.0)
        u.reset()
        assert abs(u.mu - 5.0) < 1e-9
        assert u.n_updates == 0

    def test_n_updates_increments(self):
        u = BayesianBeliefUpdater()
        assert u.n_updates == 0
        u.update(4.0)
        assert u.n_updates == 1
        u.update(6.0)
        assert u.n_updates == 2

    def test_prob_concede_next(self):
        u = BayesianBeliefUpdater()
        p = u.prob_concede_next_period()
        assert 0 < p < 1

    def test_prob_high_cost_higher(self):
        """Agent with high cost should have higher concession probability."""
        u_high = BayesianBeliefUpdater(prior_mu=9.0)
        u_low = BayesianBeliefUpdater(prior_mu=1.0)
        assert u_high.prob_concede_next_period() > u_low.prob_concede_next_period()

    def test_summary_keys(self):
        u = BayesianBeliefUpdater()
        s = u.summary()
        for key in ["mu", "uncertainty", "n_obs", "implied_cost_high", "prob_concede_next_period"]:
            assert key in s
