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


# =========================================================================
# gwModelRemS: non-precessing quasi-circular remnant model
# =========================================================================
class TestGwModelRemS:
    """Tests for the aligned-spin gwModelRemS family."""

    def test_combined_matches_individual(self):
        """gwModelRemS returns the same values as the per-quantity functions."""
        args = (3.0, 0.5, -0.2)
        Mf, chif, Lp, wp, vk = gwModels.remnants.gwModelRemS(*args)
        np.testing.assert_allclose(Mf, gwModels.remnants.gwModelRemS_mf(*args), rtol=0, atol=0)
        np.testing.assert_allclose(chif, gwModels.remnants.gwModelRemS_chif(*args), rtol=0, atol=0)
        np.testing.assert_allclose(Lp, gwModels.remnants.gwModelRemS_Lpeak(*args), rtol=0, atol=0)
        np.testing.assert_allclose(wp, gwModels.remnants.gwModelRemS_omega_peak(*args), rtol=0, atol=0)
        np.testing.assert_allclose(vk, gwModels.remnants.gwModelRemS_kick(*args), rtol=0, atol=0)

    def test_equal_mass_equal_spin_no_kick(self):
        """Symmetric binaries recoil zero by construction."""
        for chi in [0.0, 0.5, -0.8]:
            vk = gwModels.remnants.gwModelRemS_kick(1.0, chi, chi)
            np.testing.assert_allclose(vk, 0.0, atol=1e-12)

    def test_equal_mass_closed_form(self):
        """At q=1 the mass reduces to 1 - E_EM/4 with only the EM polynomial."""
        from gwModels.remnants.gwModelRemS import _MASS_PARAMS as p
        chi = 0.6
        E_EM = (p['m0'] + p['m1'] * chi + p['m2'] * chi**2
                + p['m3'] * chi**3 + p['m4'] * chi**4)
        np.testing.assert_allclose(
            gwModels.remnants.gwModelRemS_mf(1.0, chi, chi), 1 - 0.25 * E_EM, rtol=1e-12)

    def test_physical_bounds(self):
        """Mf in (0,1], |chif| <= 1, Lpeak > 0, vkick >= 0 over a wide sweep."""
        rng = np.random.default_rng(0)
        q = np.exp(rng.uniform(0, np.log(1000), 20000))
        c1 = rng.uniform(-1, 1, 20000)
        c2 = rng.uniform(-1, 1, 20000)
        Mf, chif, Lp, wp, vk = gwModels.remnants.gwModelRemS(q, c1, c2)
        assert np.all((Mf > 0) & (Mf <= 1))
        assert np.all(np.abs(chif) <= 1)
        assert np.all(Lp > 0)
        assert np.all(wp > 0)
        assert np.all(vk >= 0)
        assert np.all(np.isfinite(Mf) & np.isfinite(chif) & np.isfinite(vk))

    def test_scalar_and_array_agree(self):
        """Scalar input returns a scalar matching the vectorized evaluation.

        Note that single-element results are unwrapped to python floats, so a
        length-1 array input also returns a scalar; use length 2 here.
        """
        Mf_s = gwModels.remnants.gwModelRemS_mf(3.0, 0.4, -0.2)
        assert np.ndim(Mf_s) == 0

        Mf_a = gwModels.remnants.gwModelRemS_mf(np.array([3.0, 5.0]), 0.4, -0.2)
        assert Mf_a.shape == (2,)
        np.testing.assert_allclose(Mf_s, Mf_a[0], rtol=0, atol=0)

        assert np.ndim(gwModels.remnants.gwModelRemS_mf(np.array([3.0]), 0.4, -0.2)) == 0

    def test_shape_preservation(self):
        """2D input preserves shape across all five quantities."""
        q = np.full((4, 3), 3.0)
        out = gwModels.remnants.gwModelRemS(q, 0.2, 0.1)
        assert all(x.shape == (4, 3) for x in out)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            gwModels.remnants.gwModelRemS_mf(0.5, 0.0, 0.0)
        with pytest.raises(ValueError):
            gwModels.remnants.gwModelRemS_mf(2.0, 1.5, 0.0)

    def test_kick_uses_exchanged_spin_variable(self):
        """The recoil S_tilde is the inherited spin with body labels swapped."""
        from gwModels.remnants.gwModelRemS import _kick_spin_variables
        q, c1, c2 = 4.0, 0.7, -0.3
        St, _ = _kick_spin_variables(np.array(q), np.array(c1), np.array(c2))
        np.testing.assert_allclose(St, (c1 + q**2 * c2) / (1 + q)**2, rtol=1e-14)
        # equals the final-spin S_tilde evaluated at q -> 1/q
        np.testing.assert_allclose(
            St, ((1 / q)**2 * c1 + c2) / (1 + 1 / q)**2, rtol=1e-12)


# =========================================================================
# gwModelRemP: precessing quasi-circular remnant model
# =========================================================================
class TestGwModelRemP:
    """Tests for the precessing gwModelRemP family."""

    def test_reduces_to_aligned_spin(self):
        """S_perp = 0 must reproduce gwModelRemS exactly."""
        Mf_s = gwModels.remnants.gwModelRemS_mf(3.0, 0.5, -0.3)
        chif_s = gwModels.remnants.gwModelRemS_chif(3.0, 0.5, -0.3)
        Lp_s = gwModels.remnants.gwModelRemS_Lpeak(3.0, 0.5, -0.3)
        Mf_p, af_p, thf_p, Lp_p = gwModels.remnants.gwModelRemP(
            3.0, 0.5, 0.3, 0.0, np.pi, 0.0, 0.0)
        np.testing.assert_allclose(Mf_p, Mf_s, rtol=0, atol=0)
        np.testing.assert_allclose(af_p, abs(chif_s), rtol=0, atol=0)
        np.testing.assert_allclose(Lp_p, Lp_s, rtol=0, atol=0)
        np.testing.assert_allclose(thf_p, 0.0, atol=1e-12)

    def test_physical_bounds(self):
        """Mf <= 1 (capped), |af| <= 1, theta_f in [0, pi], Lpeak > 0."""
        rng = np.random.default_rng(0)
        n = 20000
        q = np.exp(rng.uniform(0, np.log(1000), n))
        a1 = rng.uniform(0, 1, n)
        a2 = rng.uniform(0, 1, n)
        t1 = rng.uniform(0, np.pi, n)
        t2 = rng.uniform(0, np.pi, n)
        Mf, af, thf, Lp = gwModels.remnants.gwModelRemP(q, a1, a2, t1, t2, 0.0, 0.0)
        assert np.all((Mf > 0) & (Mf <= 1))
        assert np.all((af >= 0) & (af <= 1))
        assert np.all((thf >= 0) & (thf <= np.pi))
        assert np.all(Lp > 0)

    def test_mass_cap_is_output_only(self):
        """The flow context keeps the uncapped mass, matching training."""
        from gwModels.remnants.gwModelRemP import _mass_aug, spin_projections
        from gwModels.remnants.gwModelRemS import gwModelRemS_mf, gwModelRemS_chif
        q, a1, a2, t1, t2 = 500.0, 0.9, 0.9, np.pi / 2, np.pi / 2
        c1z, c2z, sp, dp = spin_projections(
            np.array(q), np.array(a1), np.array(a2), np.array(t1), np.array(t2))
        eta = q / (1 + q)**2
        raw = _mass_aug(np.atleast_1d(gwModelRemS_mf(q, c1z, c2z)),
                        np.atleast_1d(gwModelRemS_chif(q, c1z, c2z)), sp, dp, eta)
        capped = gwModels.remnants.gwModelRemP(q, a1, a2, t1, t2, 0.0, 0.0)[0]
        assert raw > 1.0
        np.testing.assert_allclose(capped, 1.0, rtol=0, atol=0)

    def test_spin_projections(self):
        """S_perp and Delta_perp match their definitions."""
        q, a1, a2, t1, t2 = 3.0, 0.8, 0.5, np.pi / 3, np.pi / 4
        c1z, c2z, sp, dp = gwModels.remnants.spin_projections(
            np.array(q), np.array(a1), np.array(a2), np.array(t1), np.array(t2))
        p1, p2 = a1 * np.sin(t1), a2 * np.sin(t2)
        np.testing.assert_allclose(c1z, a1 * np.cos(t1), rtol=1e-14)
        np.testing.assert_allclose(sp, np.sqrt(q**4 * p1**2 + p2**2) / (q**2 + 1), rtol=1e-14)
        np.testing.assert_allclose(dp, (q * p1 - p2) / (1 + q), rtol=1e-14)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            gwModels.remnants.gwModelRemP(2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            gwModels.remnants.gwModelRemP(2.0, 0.5, 0.5, -0.1, 0.0, 0.0, 0.0)


# =========================================================================
# gwModelRemSE: eccentric non-precessing remnant model
# =========================================================================
class TestGwModelRemSE:
    """Tests for the eccentric gwModelRemSE family."""

    def test_circular_limit_exact(self):
        """e_ref = 0 must reproduce gwModelRemS bit-for-bit."""
        Mf_s, chif_s, Lp_s, _, vk_s = gwModels.remnants.gwModelRemS(2.0, 0.3, 0.1)
        Mf_e, chif_e, vk_e, Lp_e = gwModels.remnants.gwModelRemSE(2.0, 0.3, 0.1, 0.0, 0.0)
        np.testing.assert_allclose(Mf_e, Mf_s, rtol=0, atol=0)
        np.testing.assert_allclose(chif_e, chif_s, rtol=0, atol=0)
        np.testing.assert_allclose(vk_e, vk_s, rtol=0, atol=0)
        np.testing.assert_allclose(Lp_e, Lp_s, rtol=0, atol=0)

    def test_circular_limit_any_anomaly(self):
        """At e_ref = 0 the result is independent of the mean anomaly."""
        ref = gwModels.remnants.gwModelRemSE_mf(3.0, 0.2, 0.1, 0.0, 0.0)
        for l in [0.5, 2.0, np.pi, 5.5]:
            np.testing.assert_allclose(
                gwModels.remnants.gwModelRemSE_mf(3.0, 0.2, 0.1, 0.0, l), ref, rtol=0, atol=0)

    def test_equal_mass_nonspinning_no_kick(self):
        """Multiplicative correction keeps the q=1 non-spinning recoil at zero."""
        vk = gwModels.remnants.gwModelRemSE_kick(1.0, 0.0, 0.0, 0.2, 1.0)
        np.testing.assert_allclose(vk, 0.0, atol=1e-12)

    def test_anomaly_modulation_is_periodic(self):
        """The correction is 2*pi periodic in the mean anomaly."""
        a = gwModels.remnants.gwModelRemSE_mf(3.0, 0.0, 0.0, 0.15, 0.7)
        b = gwModels.remnants.gwModelRemSE_mf(3.0, 0.0, 0.0, 0.15, 0.7 + 2 * np.pi)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_separatrix_reduces_to_isco(self):
        """The optional separatrix backbone reduces to the Kerr ISCO at e=0."""
        for chi in [0.0, 0.5, -0.7]:
            np.testing.assert_allclose(
                gwModels.remnants.separatrix_energy(0.0, chi),
                gwModels.remnants.kerr_isco_energy(chi), rtol=1e-12)
            np.testing.assert_allclose(
                gwModels.remnants.separatrix_angular_momentum(0.0, chi),
                gwModels.remnants.kerr_isco_angular_momentum(chi), rtol=1e-12)
            np.testing.assert_allclose(
                gwModels.remnants.separatrix_ell(0.0, chi),
                gwModels.remnants.kerr_ell(chi), rtol=1e-12)

    def test_historical_separatrix_aliases(self):
        """E_sep / L_sep / ell_sep still resolve to the Kerr implementations."""
        from gwModels.remnants.gwModelRemSE import E_sep, L_sep, ell_sep
        from gwModels.remnants import Kerr
        assert E_sep is Kerr.separatrix_energy
        assert L_sep is Kerr.separatrix_angular_momentum
        assert ell_sep is Kerr.separatrix_ell

    def test_rejects_bad_eccentricity(self):
        with pytest.raises(ValueError):
            gwModels.remnants.gwModelRemSE(2.0, 0.0, 0.0, 1.0, 0.0)


# =========================================================================
# gwModelEMRI: point-particle remnant model
# =========================================================================
class TestGwModelEMRI:
    """Tests for the point-particle gwModelEMRI model."""

    def test_equatorial_circular_recovers_kerr_isco(self):
        """Prograde and retrograde equatorial circular orbits match the ISCO."""
        chi = np.array([0.0, 0.3, 0.5, 0.7, 0.9, -0.5, -0.9])
        E, Lz, conv = gwModels.remnants.separatrix_EL(
            chi, np.zeros_like(chi), np.ones_like(chi), return_converged=True)
        assert np.all(conv)
        np.testing.assert_allclose(
            E, gwModels.remnants.kerr_isco_energy(chi), atol=1e-12)
        np.testing.assert_allclose(
            Lz, gwModels.remnants.kerr_isco_angular_momentum(chi), atol=1e-12)

    def test_remnant_matches_closed_form(self):
        """Mf and chif follow the leading-order expressions at e=0, theta=0."""
        q, chi = 1000.0, 0.0
        eta = q / (1 + q)**2
        Mf, chif = gwModels.remnants.gwModelEMRI(q, chi, 0.0, 0.0)
        E = float(gwModels.remnants.kerr_isco_energy(chi))
        L = float(gwModels.remnants.kerr_isco_angular_momentum(chi))
        np.testing.assert_allclose(Mf, 1 - eta * (1 - E), rtol=1e-12)
        np.testing.assert_allclose(chif, chi + eta * (L - 2 * chi * (E - 1)), rtol=1e-12)

    def test_reports_convergence(self):
        """The solver exposes a convergence mask and warns when it fails."""
        chi = np.full(9, 0.9)
        th = np.radians(np.linspace(0, 180, 9))
        Mf, chif, conv = gwModels.remnants.gwModelEMRI(
            1e4, chi, th, 0.0, warn_unconverged=False, return_converged=True)
        assert conv.dtype == bool
        assert conv.size == 9
        assert np.all(np.isfinite(Mf))

    def test_extreme_mass_ratio_limit(self):
        """Mf -> 1 and chif -> chi as eta -> 0."""
        Mf, chif = gwModels.remnants.gwModelEMRI(1e6, 0.7, 0.0, 0.0)
        np.testing.assert_allclose(Mf, 1.0, atol=1e-5)
        np.testing.assert_allclose(chif, 0.7, atol=1e-5)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            gwModels.remnants.gwModelEMRI(0.5, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            gwModels.remnants.gwModelEMRI(1000.0, 1.5, 0.0, 0.0)
        with pytest.raises(ValueError):
            gwModels.remnants.gwModelEMRI(1000.0, 0.0, 0.0, 1.0)


# =========================================================================
# gwModelRemP_flow: precessing recoil distribution
# =========================================================================
class TestGwModelRemPFlow:
    """Tests for the precessing recoil normalizing flow."""

    def test_torch_stays_optional(self):
        """Importing gwModels must not pull in torch or nflows."""
        import subprocess
        import sys as _sys
        code = ("import sys; import gwModels; "
                "print('torch' in sys.modules, 'nflows' in sys.modules)")
        out = subprocess.run([_sys.executable, '-c', code],
                             capture_output=True, text=True)
        assert 'False False' in out.stdout

    def test_sample_and_predict(self, data_dir):
        pytest.importorskip("torch")
        pytest.importorskip("nflows")
        flow = gwModels.remnants.gwModelRemP_flow(data_dir)
        s = flow.sample(2.0, 0.7, 0.3, np.pi / 3, np.pi / 4, 0.0, 0.0, n_samples=500)
        assert s.shape == (1, 500)
        assert np.all(s > 0)
        med, p5, p95 = flow.predict(2.0, 0.7, 0.3, np.pi / 3, np.pi / 4, 0.0, 0.0,
                                    n_samples=2000)
        assert p5 <= med <= p95

    def test_context_shape_and_content(self, data_dir):
        pytest.importorskip("torch")
        pytest.importorskip("nflows")
        flow = gwModels.remnants.gwModelRemP_flow(data_dir)
        ctx = flow.compute_context(np.array([2.0, 4.0]), 0.7, 0.3,
                                   np.pi / 3, np.pi / 4)
        assert ctx.shape == (2, 5)
        # columns are (Mf, |chif|, eta, S_perp, Delta_perp)
        assert np.all(ctx[:, 2] > 0) and np.all(ctx[:, 2] <= 0.25)
        assert np.all(ctx[:, 3] >= 0)


# =========================================================================
# Package-level smoke tests
# =========================================================================
class TestPublicAPISmoke:
    """Guard against import shadowing and other whole-module breakage."""

    def test_every_public_callable_evaluates(self):
        """Call every public function in gwModels.remnants once.

        A module-level `def foo` that shadows `from .X import foo` and then
        calls `foo(...)` recurses forever. Package-level lookups can resolve
        to the healthy copy and hide it, so exercise the modules directly.
        """
        import inspect
        from gwModels.remnants import Kerr, gwModelEMRI as emri_mod
        from gwModels.remnants import gwModelRemS, gwModelRemP, gwModelRemSE

        scalar_args = {
            1: (0.5,),
            2: (0.1, 0.5),
            3: (2.0, 0.3, 0.1),
            5: (2.0, 0.3, 0.1, 0.1, 0.5),
            7: (2.0, 0.5, 0.3, 0.4, 0.3, 0.0, 0.0),
        }
        for module in (Kerr, emri_mod, gwModelRemS, gwModelRemP, gwModelRemSE):
            for name, fn in vars(module).items():
                if name.startswith('_') or not inspect.isfunction(fn):
                    continue
                if fn.__module__ != module.__name__:
                    continue  # imported, tested in its own module
                sig = inspect.signature(fn)
                n_req = sum(1 for p in sig.parameters.values()
                            if p.default is inspect.Parameter.empty)
                if n_req not in scalar_args:
                    continue
                try:
                    fn(*scalar_args[n_req])
                except RecursionError:
                    pytest.fail(f"{module.__name__}.{name} recurses infinitely "
                                f"(likely shadows an import of the same name)")
                except (ValueError, TypeError):
                    pass  # wrong sample arguments for this signature, fine

    def test_no_function_shadows_its_own_import(self):
        """No module defines a function with the same name it imports."""
        import ast
        import pathlib
        import gwModels.remnants as pkg

        offenders = []
        for path in pathlib.Path(pkg.__file__).parent.glob('*.py'):
            tree = ast.parse(path.read_text())
            imported = {a.asname or a.name
                        for n in ast.walk(tree)
                        if isinstance(n, (ast.Import, ast.ImportFrom))
                        for a in n.names}
            defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
            clash = imported & defined
            if clash:
                offenders.append(f"{path.name}: {sorted(clash)}")
        assert not offenders, "functions shadow same-named imports: " + "; ".join(offenders)
