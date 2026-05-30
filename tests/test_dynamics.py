"""Tests for gwModels.dynamics submodule."""

import os
import numpy as np
import pytest
import gwModels


class TestGwEccEvNS:
    """Tests for the gwEccEvNS eccentricity evolution model."""

    def test_basic_evolution(self):
        t = np.linspace(-3000, -1, 3000)
        e = gwModels.dynamics.gwEccEvNS_model(t, q=1, e0=0.07)
        assert e.shape == t.shape
        assert np.all(np.isfinite(e))

    def test_eccentricity_decreases(self):
        """Eccentricity should decrease toward merger."""
        t = np.linspace(-3000, -1, 3000)
        e = gwModels.dynamics.gwEccEvNS_model(t, q=1, e0=0.07)
        assert e[0] > e[-1]

    def test_initial_value(self):
        """Eccentricity at t[0] should equal e0."""
        t = np.linspace(-3000, -1, 3000)
        e0 = 0.07
        e = gwModels.dynamics.gwEccEvNS_model(t, q=1, e0=e0)
        np.testing.assert_allclose(e[0], e0, rtol=1e-6)

    def test_different_mass_ratios(self):
        """Different mass ratios should give different evolutions."""
        t = np.linspace(-3000, -1, 3000)
        e1 = gwModels.dynamics.gwEccEvNS_model(t, q=1, e0=0.05)
        e2 = gwModels.dynamics.gwEccEvNS_model(t, q=4, e0=0.05)
        assert not np.allclose(e1, e2)

    def test_equal_mass_evolution_matches_sxs1355(self, sxs1355_data):
        """gwEccEvNS for q=1 should roughly track the NR eccentricity."""
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
        e_model = gwModels.dynamics.gwEccEvNS_model(
            t=ecc_obj.time_xi, q=1, e0=ecc_obj.ecc_xi[0])
        # should agree within 20% over the inspiral
        ratio = e_model / ecc_obj.ecc_xi
        mask = ecc_obj.time_xi < -200
        np.testing.assert_allclose(ratio[mask], 1.0, atol=0.2)


class TestGwEccEvNSv2:
    """Tests for the gwEccEvNSv2 analytical model."""

    def test_basic_evolution(self):
        t = np.linspace(-3000, -1, 3000)
        e = gwModels.dynamics.gwEccEvNSv2(t, q=1, e0=0.07)
        assert e.shape == t.shape
        assert np.all(np.isfinite(e))

    def test_eccentricity_decreases(self):
        t = np.linspace(-3000, -1, 3000)
        e = gwModels.dynamics.gwEccEvNSv2(t, q=1, e0=0.07)
        # near merger, eccentricity forced to 0
        assert e[0] > e[-1]

    def test_initial_value_with_tref(self):
        """e(t_ref) should be close to e0 when t_ref = t[0]."""
        t = np.linspace(-3000, -1, 3000)
        e0 = 0.07
        e = gwModels.dynamics.gwEccEvNSv2(t, q=1, e0=e0, t_ref=t[0])
        np.testing.assert_allclose(e[0], e0, rtol=1e-6)

    def test_zero_after_merger(self):
        """Eccentricity should be zero for t >= 0 (post-merger)."""
        t = np.linspace(-100, 100, 2000)
        e = gwModels.dynamics.gwEccEvNSv2(t, q=1, e0=0.05)
        assert np.all(e[t >= 0] == 0.0)


class TestGwEccEvolveNoSpinq4:
    """Tests for the SVD surrogate eccentricity evolution model."""

    @pytest.fixture(scope="class")
    def model(self, data_dir):
        path = os.path.join(data_dir, 'gwEccEvolve_NoSpinq4.npy')
        return gwModels.dynamics.gwEccEvolve_NoSpinq4(path)

    def test_model_loads(self, model):
        assert model.n_basis == 2
        np.testing.assert_allclose(model.q_range[0], 1.0, atol=0.01)
        np.testing.assert_allclose(model.q_range[1], 4.0, atol=0.01)

    def test_basic_evaluation(self, model):
        t = np.linspace(-6300, -1, 1000)
        ecc = model(t, q=1.5, e0=0.2)
        assert ecc.shape == t.shape
        assert np.all(np.isfinite(ecc))

    def test_initial_scaling(self, model):
        """e(t[0]) should scale linearly with e0 (model predicts e(t)/e0)."""
        t = np.linspace(-6300, -100, 1000)
        ecc1 = model(t, q=2.0, e0=0.1)
        ecc2 = model(t, q=2.0, e0=0.2)
        ratio = ecc2 / ecc1
        np.testing.assert_allclose(ratio, 2.0, rtol=0.05)

    def test_return_std(self, model):
        t = np.linspace(-6300, -1, 1000)
        ecc, ecc_std = model(t, q=1.5, e0=0.2, return_std=True)
        assert ecc.shape == ecc_std.shape
        assert np.all(ecc_std >= 0)

    def test_eccentricity_decreases(self, model):
        t = np.linspace(-6300, -1, 1000)
        ecc = model(t, q=2.0, e0=0.2)
        assert ecc[0] > ecc[-1]

    def test_different_q_gives_different_result(self, model):
        t = np.linspace(-6300, -1, 1000)
        ecc1 = model(t, q=1.0, e0=0.2)
        ecc2 = model(t, q=4.0, e0=0.2)
        assert not np.allclose(ecc1, ecc2)
