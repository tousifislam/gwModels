"""Tests for gwModels.frameworks submodule using SXS:BBH:1355 data."""

import numpy as np
import pytest
import gwModels


class TestBaseEccentricHM:
    """Tests for the NRHME framework using bundled NR data."""

    @pytest.fixture(scope="class")
    def nrhme_result(self, sxs1355_data):
        """Run NRHME on SXS:BBH:1355 data once for the class."""
        gwnrhme = gwModels.frameworks.NRHME(
            t_ecc=sxs1355_data['t_ecc'],
            h_ecc_dict={'h_l2m2': sxs1355_data['hecc_dict']['h_l2m2']},
            t_cir=sxs1355_data['t_cir'],
            h_cir_dict={'h_l2m2': sxs1355_data['hcir_dict']['h_l2m2']},
            project_ecc_on_higher_modes=False,
        )
        return gwnrhme

    def test_common_time_is_monotonic(self, nrhme_result):
        assert np.all(np.diff(nrhme_result.t_common) > 0)

    def test_amplitude_modulation_shape(self, nrhme_result):
        assert nrhme_result.xi_amp.shape == nrhme_result.t_common.shape

    def test_orbfreq_modulation_shape(self, nrhme_result):
        assert nrhme_result.xi_omega.shape == nrhme_result.t_common.shape

    def test_amplitude_modulation_bounded(self, nrhme_result):
        # amplitude modulation should be small (less than 1 for physical signals)
        assert np.max(np.abs(nrhme_result.xi_amp)) < 5.0


class TestNRHMEMultiMode:
    """Test NRHME with multi-mode circular input (full SXS:BBH:1355 data)."""

    @pytest.fixture(scope="class")
    def nrhme_multimode(self, sxs1355_data):
        gwnrhme = gwModels.frameworks.NRHME(
            t_ecc=sxs1355_data['t_ecc'],
            h_ecc_dict={'h_l2m2': sxs1355_data['hecc_dict']['h_l2m2']},
            t_cir=sxs1355_data['t_cir'],
            h_cir_dict=sxs1355_data['hcir_dict'],
            project_ecc_on_higher_modes=True,
        )
        return gwnrhme

    def test_output_has_all_modes(self, nrhme_multimode):
        expected = {'h_l2m2', 'h_l2m1', 'h_l3m2', 'h_l3m3', 'h_l4m3', 'h_l4m4'}
        assert set(nrhme_multimode.hNRE.keys()) == expected

    def test_22_mode_preserved(self, nrhme_multimode):
        h22 = nrhme_multimode.hNRE['h_l2m2']
        assert len(h22) == len(nrhme_multimode.t_common)
        assert np.max(np.abs(h22)) > 0

    def test_higher_modes_nonzero(self, nrhme_multimode):
        for mode in ['h_l2m1', 'h_l3m3', 'h_l4m4']:
            assert np.max(np.abs(nrhme_multimode.hNRE[mode])) > 0


class TestNRHMESubclass:
    """Test NRHME subclass identity."""

    def test_nrhme_is_base_subclass(self):
        assert issubclass(gwModels.frameworks.NRHME, gwModels.frameworks.BaseEccentricHM)

    def test_nrxhme_is_base_subclass(self):
        assert issubclass(gwModels.frameworks.NRXHME, gwModels.frameworks.BaseEccentricHM)

    def test_nrhme_description(self):
        assert gwModels.frameworks.NRHME._description == "non-spinning"

    def test_nrxhme_description(self):
        assert gwModels.frameworks.NRXHME._description == "non-precessing"


class TestFrameworkValidation:
    """Test input validation in BaseEccentricHM."""

    def test_missing_t_ecc_raises(self):
        with pytest.raises(ValueError, match="t_ecc"):
            gwModels.frameworks.NRHME(t_ecc=None, h_ecc_dict={}, t_cir=np.array([1]), h_cir_dict={})

    def test_missing_h_ecc_dict_raises(self):
        with pytest.raises(ValueError, match="h_ecc_dict"):
            gwModels.frameworks.NRHME(t_ecc=np.array([1]), h_ecc_dict=None, t_cir=np.array([1]), h_cir_dict={})

    def test_missing_t_cir_raises(self):
        with pytest.raises(ValueError, match="t_cir"):
            gwModels.frameworks.NRHME(t_ecc=np.array([1]), h_ecc_dict={}, t_cir=None, h_cir_dict={})

    def test_missing_h_cir_dict_raises(self):
        with pytest.raises(ValueError, match="h_cir_dict"):
            gwModels.frameworks.NRHME(t_ecc=np.array([1]), h_ecc_dict={}, t_cir=np.array([1]), h_cir_dict=None)
