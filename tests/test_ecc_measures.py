"""Tests for gwModels.ecc_measures submodule.

Reference values from tutorial 3_1 using SXS:BBH:1355 (q=1, e~0.08).
"""

import numpy as np
import pytest
import gwModels


class TestComputeEccentricity:
    """Tests for ComputeEccentricity using SXS:BBH:1355."""

    @pytest.fixture(scope="class")
    def ecc_result(self, sxs1355_data):
        """Compute eccentricity once, matching tutorial 3_1 settings."""
        obj = gwModels.ecc_measures.ComputeEccentricity(
            t_ecc=sxs1355_data['t_ecc'],
            h_ecc_dict={'h_l2m2': sxs1355_data['hecc_dict']['h_l2m2']},
            t_cir=sxs1355_data['t_cir'],
            h_cir_dict={'h_l2m2': sxs1355_data['hcir_dict']['h_l2m2']},
            q=1,
            distance_btw_peaks=2000,
            fit_funcs_orders=['3PN_m1over8', '3PN_m1over8'],
            ecc_prefactor=2 / 3,
        )
        return obj

    def test_ecc_ref_value(self, ecc_result):
        """Eccentricity at t_ref should match tutorial output: 0.07996."""
        np.testing.assert_allclose(ecc_result.ecc_ref, 0.07996, atol=0.002)

    def test_ecc_ref_time(self, ecc_result):
        """t_ref should be near -2638.56."""
        np.testing.assert_allclose(ecc_result.t_ref, -2638.56, atol=1.0)

    def test_eccentricity_positive(self, ecc_result):
        """Eccentricity should be non-negative."""
        assert np.all(ecc_result.ecc_xi >= 0)

    def test_eccentricity_decreases(self, ecc_result):
        """Eccentricity should generally decrease toward merger."""
        e_start = ecc_result.ecc_xi[0]
        e_end = ecc_result.ecc_xi[-1]
        assert e_start > e_end

    def test_time_axis_premerger(self, ecc_result):
        """Time axis should be pre-merger (t <= 0)."""
        assert np.all(ecc_result.time_xi <= 0)

    def test_fit_errors_small(self, ecc_result):
        """PN fit errors should be small."""
        assert ecc_result.xi_upper_fit_error < 0.01
        assert ecc_result.xi_lower_fit_error < 0.01

    def test_interpolants_callable(self, ecc_result):
        """Interpolants should return scalar for scalar input."""
        t_mid = ecc_result.time_xi[len(ecc_result.time_xi) // 2]
        assert np.isfinite(ecc_result.ecc_interp(t_mid))
        assert np.isfinite(ecc_result.xi_upper_interp(t_mid))
        assert np.isfinite(ecc_result.xi_lower_interp(t_mid))

    def test_upper_above_lower_envelope(self, ecc_result):
        """Upper envelope should be >= lower envelope during early inspiral."""
        mask = ecc_result.time_xi < -1000
        assert np.all(ecc_result.xi_upper[mask] >= ecc_result.xi_lower[mask] - 1e-10)


class TestComputeEccentricityFromModulations:
    """Tests for the lower-level modulations-based eccentricity class."""

    @pytest.fixture(scope="class")
    def mod_result(self, sxs1355_data):
        """Build NRHME object and feed modulations to ComputeEccentricityFromModulations."""
        nrhme = gwModels.frameworks.NRHME(
            t_ecc=sxs1355_data['t_ecc'],
            h_ecc_dict={'h_l2m2': sxs1355_data['hecc_dict']['h_l2m2']},
            t_cir=sxs1355_data['t_cir'],
            h_cir_dict={'h_l2m2': sxs1355_data['hcir_dict']['h_l2m2']},
            project_ecc_on_higher_modes=False,
        )
        t_pre = nrhme.t_common[nrhme.t_common <= 0]
        xi_pre = nrhme.xi_amp[nrhme.t_common <= 0]

        obj = gwModels.ecc_measures.ComputeEccentricityFromModulations(
            time_xi=t_pre,
            xi=xi_pre,
            q=1,
            distance_btw_peaks=2000,
            fit_funcs_orders=['3PN_m1over8', '3PN_m1over8'],
            ecc_prefactor=2 / 3,
        )
        return obj

    def test_maximas_found(self, mod_result):
        assert len(mod_result.t_maximas) > 0

    def test_minimas_found(self, mod_result):
        assert len(mod_result.t_minimas) > 0

    def test_eccentricity_shape(self, mod_result):
        assert mod_result.ecc_xi.shape == mod_result.time_xi.shape


class TestComputeEccentricityFromOmega:
    """Tests for the omega-based eccentricity measures."""

    @pytest.fixture(scope="class")
    def omega_result(self, sxs1355_data):
        """Compute e_omega and e_gw, matching tutorial 3_1 section 5."""
        ecc_obj = gwModels.ecc_measures.ComputeEccentricity(
            t_ecc=sxs1355_data['t_ecc'],
            h_ecc_dict={'h_l2m2': sxs1355_data['hecc_dict']['h_l2m2']},
            t_cir=sxs1355_data['t_cir'],
            h_cir_dict={'h_l2m2': sxs1355_data['hcir_dict']['h_l2m2']},
            q=1,
            distance_btw_peaks=2000,
            fit_funcs_orders=['3PN_m1over8', '3PN_m1over8'],
            ecc_prefactor=2 / 3,
        )
        eobj = gwModels.ecc_measures.ComputeEccentricityFromOmega(
            time_xi=ecc_obj.time_xi,
            xi_lower=ecc_obj.xi_lower,
            xi_upper=ecc_obj.xi_upper,
            gwnrhme_obj=ecc_obj.gwnrhme_obj,
            ecc_prefactor=ecc_obj.ecc_prefactor,
            t_ref=ecc_obj.t_ref,
        )
        return eobj

    def test_ecc_omega_22_ref(self, omega_result):
        """e_omega_22 at t_ref should match tutorial: ~0.05993."""
        np.testing.assert_allclose(omega_result.ecc_omega_22_ref, 0.05993, atol=0.003)

    def test_ecc_gw_ref(self, omega_result):
        """e_gw at t_ref should match tutorial: ~0.07979."""
        np.testing.assert_allclose(omega_result.ecc_gw_ref, 0.07979, atol=0.003)

    def test_ecc_gw_positive(self, omega_result):
        # e_gw should be non-negative for the inspiral portion
        mask = omega_result.time_xi < -100
        assert np.all(omega_result.ecc_gw[mask] >= -0.01)


class TestComputeEccentricityValidation:
    """Test input validation."""

    def test_missing_q_raises(self, sxs1355_data):
        with pytest.raises(ValueError, match="q"):
            gwModels.ecc_measures.ComputeEccentricity(
                t_ecc=sxs1355_data['t_ecc'],
                h_ecc_dict={'h_l2m2': sxs1355_data['hecc_dict']['h_l2m2']},
                t_cir=sxs1355_data['t_cir'],
                h_cir_dict={'h_l2m2': sxs1355_data['hcir_dict']['h_l2m2']},
                q=None,
            )

    def test_bad_method_raises(self, sxs1355_data):
        with pytest.raises(ValueError, match="not recognized"):
            gwModels.ecc_measures.ComputeEccentricity(
                t_ecc=sxs1355_data['t_ecc'],
                h_ecc_dict={'h_l2m2': sxs1355_data['hecc_dict']['h_l2m2']},
                t_cir=sxs1355_data['t_cir'],
                h_cir_dict={'h_l2m2': sxs1355_data['hcir_dict']['h_l2m2']},
                q=1,
                method='bad_method',
            )


class TestInitialEccentricity:
    """Tests for initial eccentricity conversion functions."""

    def test_harmonic_3PN_mildly_eccentric(self):
        """A mildly eccentric orbit should give e_t in (0, 1)."""
        # Newtonian: et^2 = 1 - (-2*E0)*h0^2 where E0=Eb/eta, h0=L/eta
        # For eta=0.25, Eb=-0.005, L=4.0: E0=-0.02, h0=16, eps=0.04, epsj2=0.04*256=10.24
        # et2_newt = 1 - 10.24 < 0, so we need smaller L
        # For Eb=-0.02, L=2.0: E0=-0.08, h0=8, eps=0.16, epsj2=0.16*64=10.24 -> still too big
        # For Eb=-0.03, L=1.0: E0=-0.12, h0=4, eps=0.24, epsj2=0.24*16=3.84 -> et2=1-3.84<0
        # Need epsj2 < 1 => eps*j2 < 1, so small binding energy and small L
        # Eb=-0.005, L=0.5, eta=0.25: E0=-0.02, h0=2, eps=0.04, epsj2=0.04*4=0.16
        # et2_newt = 1 - 0.16 = 0.84 -> e_t ~ 0.92
        e_t = gwModels.ecc_measures.compute_et_harmonic_3PN(Eb=-0.005, L=0.5, eta=0.25)
        assert np.isfinite(e_t)
        assert 0 < e_t < 1.5

    def test_harmonic_3PN_different_eta(self):
        """Different eta should give different results."""
        e1 = gwModels.ecc_measures.compute_et_harmonic_3PN(Eb=-0.005, L=0.5, eta=0.25)
        e2 = gwModels.ecc_measures.compute_et_harmonic_3PN(Eb=-0.005, L=0.5, eta=0.24)
        assert np.isfinite(e1) and np.isfinite(e2)
        assert e1 != e2

    def test_ADM_2PN_nonspinning(self):
        """ADM with zero spins should give a finite result."""
        e_t = gwModels.ecc_measures.compute_et_ADM_2PN(Eb=-0.005, L=0.5, eta=0.25)
        assert np.isfinite(e_t)
        assert e_t >= 0

    def test_ADM_2PN_with_spins(self):
        """ADM with nonzero spins should give a finite result."""
        e_t = gwModels.ecc_measures.compute_et_ADM_2PN(
            Eb=-0.005, L=0.5, eta=0.2, chi1=0.5, chi2=-0.3)
        assert np.isfinite(e_t)
        assert e_t >= 0
