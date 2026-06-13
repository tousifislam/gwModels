"""Tests for gwModels.remnants submodule.

Reference values from tutorials 5_1 and 5_2.
"""

import os
import numpy as np
import pytest
import gwModels


# =========================================================================
# IW aligned-spin kick (gwModel_kick_q200)
# =========================================================================
class TestIWKickAligned:
    """Tests for gwModel_kick_q200 (analytical aligned-spin kick)."""

    def test_tutorial_value(self):
        """Match tutorial 5_1: q=2, chi1z=0.6, chi2z=-0.7 -> 128.97 km/s."""
        vk = gwModels.remnants.gwModel_kick_q200(2, 0.6, -0.7)
        np.testing.assert_allclose(vk, 128.97, atol=1.0)

    def test_return_std(self):
        """Tutorial: uncertainty ~4.29 km/s."""
        vk, vk_std = gwModels.remnants.gwModel_kick_q200(2, 0.6, -0.7, return_std=True)
        np.testing.assert_allclose(vk_std, 4.29, atol=1.0)

    def test_equal_mass_zero_spin_zero_kick(self):
        """q=1, zero spins -> zero kick."""
        vk = gwModels.remnants.gwModel_kick_q200(1.0, 0.0, 0.0)
        np.testing.assert_allclose(vk, 0.0, atol=0.1)

    def test_vectorized(self):
        """Should work with arrays."""
        q = np.array([1.0, 2.0, 4.0])
        vk = gwModels.remnants.gwModel_kick_q200(q, 0.5, 0.0)
        assert vk.shape == (3,)
        assert np.all(np.isfinite(vk))

    def test_q_less_than_1_raises(self):
        with pytest.raises(ValueError, match="q must be >= 1"):
            gwModels.remnants.gwModel_kick_q200(0.5, 0.0, 0.0)

    def test_spin_out_of_range_raises(self):
        with pytest.raises(ValueError, match="chi1z must be in"):
            gwModels.remnants.gwModel_kick_q200(2.0, 1.5, 0.0)

    def test_kick_positive(self):
        """Kick magnitude should be non-negative."""
        q_arr = np.linspace(1, 100, 50)
        vk = gwModels.remnants.gwModel_kick_q200(q_arr, 0.9, -0.9)
        assert np.all(vk >= 0)


# =========================================================================
# IW GPR kick (gwModel_kick_q200_GPR)
# =========================================================================
class TestIWKickGPR:
    """Tests for gwModel_kick_q200_GPR."""

    @pytest.fixture(scope="class")
    def gpr_model(self, data_dir):
        return gwModels.remnants.gwModel_kick_q200_GPR(
            os.path.join(data_dir, 'gwModel_kick_q200_GPR_aligned_spin.pkl'))

    def test_gpr_prediction(self, gpr_model):
        """Tutorial: GPR -> 132.81 km/s."""
        vk, vk_std = gpr_model.predict(q=2, chi1z=0.6, chi2z=-0.7)
        np.testing.assert_allclose(vk[0], 132.81, atol=5.0)
        assert vk_std[0] > 0

    def test_gpr_returns_positive_std(self, gpr_model):
        vk, vk_std = gpr_model.predict(q=2, chi1z=0.6, chi2z=-0.7)
        assert np.all(np.isfinite(vk_std))
        assert np.all(vk_std > 0)


# =========================================================================
# IW precessing kick (normalizing flow)
# =========================================================================
class TestIWKickPrecessing:
    """Tests for gwModel_kick_prec_flow."""

    @pytest.fixture(scope="class")
    def flow_model(self, data_dir):
        return gwModels.remnants.gwModel_kick_prec_flow(data_dir)

    def test_sampling(self, flow_model):
        samples = flow_model.sample(q=3, a1=0.5, a2=0.4, num_samples=1000)
        assert samples.shape == (1000,)
        assert np.all(samples >= 0)
        assert np.all(np.isfinite(samples))

    def test_nonspinning_falls_back(self, flow_model):
        """a1=a2=0 should fall back to aligned-spin model."""
        samples = flow_model.sample(q=2, a1=0, a2=0, num_samples=100)
        assert np.all(samples == samples[0])

    def test_batched_sample_shapes(self, flow_model):
        """Array inputs vectorize with documented shape conventions."""
        q = np.array([2.0, 4.0, 6.0])
        a1 = np.array([0.5, 0.6, 0.7])
        a2 = np.array([0.4, 0.5, 0.6])
        # array, num_samples=1 -> (N,)
        v1 = flow_model.sample(q, a1, a2, num_samples=1)
        assert v1.shape == (3,)
        # array, num_samples>1 -> (N, num_samples)
        vM = flow_model.sample(q, a1, a2, num_samples=50)
        assert vM.shape == (3, 50)
        # scalar input unchanged -> (num_samples,)
        vs = flow_model.sample(2.0, 0.5, 0.4, num_samples=50)
        assert vs.shape == (50,)
        assert np.all(np.isfinite(vM)) and np.all(vM >= 0)

    def test_batched_mixed_nonspinning(self, flow_model):
        """Non-spinning entries in a batch use the deterministic fallback."""
        q = np.array([3.0, 4.0])
        v = flow_model.sample(q, np.array([0.0, 0.5]), np.array([0.0, 0.4]),
                              num_samples=10)
        assert np.all(v[0] == v[0, 0])          # non-spinning row is constant
        assert not np.all(v[1] == v[1, 0])      # spinning row varies

    def test_batched_matches_loop_distribution(self, flow_model):
        """Batched and per-binary sampling agree distributionally (KS test)."""
        rng = np.random.default_rng(0)
        n = 1500
        q = rng.uniform(1, 8, n)
        a1 = rng.uniform(0.05, 1, n)
        a2 = rng.uniform(0.05, 1, n)
        batched = flow_model.sample(q, a1, a2, num_samples=1)
        loop = np.array([flow_model.sample(q[i], a1[i], a2[i], num_samples=1)[0]
                         for i in range(n)])
        from scipy.stats import ks_2samp
        assert ks_2samp(batched, loop).pvalue > 0.01

    def test_log_prob_normalized(self, flow_model):
        """exp(log_prob) integrates to ~1 over the kick magnitude."""
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        vg = np.linspace(0, 3000, 4000)
        lp = flow_model.log_prob(vg, np.full_like(vg, 4.0),
                                 np.full_like(vg, 0.6), np.full_like(vg, 0.5))
        assert abs(trapz(np.exp(lp), vg) - 1.0) < 0.02

    def test_log_prob_scalar_and_shape(self, flow_model):
        """log_prob returns a scalar for scalar input, array for arrays."""
        assert np.ndim(flow_model.log_prob(500.0, 4.0, 0.6, 0.5)) == 0
        out = flow_model.log_prob(np.array([300.0, 600.0]), 4.0, 0.6, 0.5)
        assert out.shape == (2,) and np.all(np.isfinite(out))


# =========================================================================
# HLZ kick models
# =========================================================================
class TestHLZKick:
    """Tests for HLZ kick velocity models."""

    def test_precessing_kick_tutorial(self):
        """Tutorial 5_2: q=1.25 (was q=0.8 with old convention), a1=a2=0.7."""
        vk = gwModels.remnants.bbh_final_kick_precessing_CLZM2007(
            q=1.25, a1=0.7, a2=0.7,
            theta1=np.pi / 4, theta2=3 * np.pi / 4,
            delta_phi=np.pi, Theta=0.0)
        np.testing.assert_allclose(vk, 2000.46, atol=1.0)

    def test_equal_mass_zero_spin(self):
        """Equal mass, zero spins -> zero kick."""
        vk = gwModels.remnants.bbh_final_kick_precessing_CLZM2007(
            q=1.0, a1=0.0, a2=0.0,
            theta1=0.0, theta2=0.0, delta_phi=0.0, Theta=0.0)
        np.testing.assert_allclose(vk, 0.0, atol=0.1)

    def test_aligned_spin_kick_positive(self):
        """HLZ aligned-spin should return non-negative kicks."""
        q_arr = np.linspace(1, 50, 100)
        vk = gwModels.remnants.bbh_final_kick_nonprecessing_HLZ2014(q_arr, 0.9, -0.9)
        assert np.all(vk >= 0)
        assert np.all(np.isfinite(vk))


# =========================================================================
# HBR final mass and spin
# =========================================================================
class TestHBRFinalState:
    """Tests for HBR remnant mass and spin models."""

    def test_final_mass_equal_mass_nonspinning(self):
        mf = gwModels.remnants.bbh_final_mass_precessing_BMR2012(
            q=1.0, a1=0.0, a2=0.0, theta1=0.0, theta2=0.0)
        assert 0.9 < mf < 1.0

    def test_final_mass_bounded(self):
        """Final mass should be less than total mass."""
        q_arr = np.linspace(1.0, 10.0, 50)
        mf = gwModels.remnants.bbh_final_mass_precessing_BMR2012(
            q=q_arr, a1=0.5, a2=0.5, theta1=0.0, theta2=0.0)
        assert np.all(mf > 0)
        assert np.all(mf < 1.0)

    def test_final_spin_equal_mass_nonspinning(self):
        sf = gwModels.remnants.bbh_final_spin_precessing_HBR2016(
            q=1.0, a1=0.0, a2=0.0, theta1=0.0, theta2=0.0, delta_phi=0.0)
        np.testing.assert_allclose(sf, 0.6865, atol=0.01)

    def test_final_spin_bounded(self):
        """Final spin should be in [0, 1]."""
        q_arr = np.linspace(1.0, 10.0, 50)
        sf = gwModels.remnants.bbh_final_spin_precessing_HBR2016(
            q=q_arr, a1=0.9, a2=0.9, theta1=0.0, theta2=0.0, delta_phi=0.0)
        assert np.all(sf >= 0)
        assert np.all(sf <= 1.0)


# =========================================================================
# UIB2016 final mass and spin
# =========================================================================
class TestUIB2016FinalState:
    """Tests for UIB2016 aligned-spin final mass and spin fits."""

    def test_final_mass_equal_mass_nonspinning(self):
        mf = gwModels.remnants.bbh_final_mass_non_precessing_UIB2016(
            q=1.0, chi1z=0.0, chi2z=0.0)
        assert 0.9 < mf < 1.0

    def test_final_mass_positive(self):
        q_arr = np.linspace(1, 10, 50)
        mf = gwModels.remnants.bbh_final_mass_non_precessing_UIB2016(
            q_arr, chi1z=0.5, chi2z=0.5)
        assert np.all(mf > 0)
        assert np.all(mf < 1.0)

    def test_final_spin_equal_mass_nonspinning(self):
        sf = gwModels.remnants.bbh_final_spin_non_precessing_UIB2016(
            q=1.0, chi1z=0.0, chi2z=0.0)
        np.testing.assert_allclose(sf, 0.6865, atol=0.02)

    def test_final_spin_bounded(self):
        q_arr = np.linspace(1, 10, 50)
        sf = gwModels.remnants.bbh_final_spin_non_precessing_UIB2016(
            q_arr, chi1z=0.9, chi2z=0.9)
        assert np.all(sf >= 0)
        assert np.all(sf <= 1.0)

    def test_negative_mass_raises(self):
        """Internal (m1,m2) function should still validate."""
        from gwModels.remnants.UIB2016_mass_spin import _bbh_final_mass_non_precessing
        with pytest.raises(ValueError, match="m1 must not be negative"):
            _bbh_final_mass_non_precessing(-1, 30, 0, 0)

    def test_spin_out_of_range_raises(self):
        with pytest.raises(ValueError, match="chi1z"):
            gwModels.remnants.bbh_final_mass_non_precessing_UIB2016(1.0, 1.5, 0)

    def test_q_wrapper_matches_m1m2(self):
        """q-based wrapper should match (m1, m2) version."""
        from gwModels.remnants.UIB2016_mass_spin import _bbh_final_mass_non_precessing, _bbh_final_spin_non_precessing
        q = 3.0
        m1, m2 = 30.0, 10.0
        mf_m1m2 = _bbh_final_mass_non_precessing(m1, m2, 0.5, -0.3)
        mf_q = gwModels.remnants.bbh_final_mass_non_precessing_UIB2016(q, 0.5, -0.3)
        np.testing.assert_allclose(mf_q, mf_m1m2 / (m1 + m2), rtol=1e-10)

        sf_m1m2 = _bbh_final_spin_non_precessing(m1, m2, 0.5, -0.3)
        sf_q = gwModels.remnants.bbh_final_spin_non_precessing_UIB2016(q, 0.5, -0.3)
        np.testing.assert_allclose(sf_q, sf_m1m2, rtol=1e-10)


# =========================================================================
# remnant_utils
# =========================================================================
class TestRemnantUtils:
    """Tests for shared remnant utility functions."""

    def test_symmetric_mass_ratio_equal_mass(self):
        eta = gwModels.remnants.symmetric_mass_ratio(1.0)
        np.testing.assert_allclose(eta, 0.25)

    def test_symmetric_mass_ratio_extreme(self):
        eta = gwModels.remnants.symmetric_mass_ratio(0.01)
        assert 0 < eta < 0.25

    def test_kerr_isco_schwarzschild(self):
        """a=0 -> r_ISCO = 6."""
        r = gwModels.remnants.kerr_isco_radius(0.0)
        np.testing.assert_allclose(r, 6.0, atol=1e-10)

    def test_kerr_isco_maximal_prograde(self):
        """a=1 -> r_ISCO = 1."""
        r = gwModels.remnants.kerr_isco_radius(1.0)
        np.testing.assert_allclose(r, 1.0, atol=1e-10)

    def test_kerr_isco_maximal_retrograde(self):
        """a=-1 -> r_ISCO = 9."""
        r = gwModels.remnants.kerr_isco_radius(-1.0)
        np.testing.assert_allclose(r, 9.0, atol=1e-10)
