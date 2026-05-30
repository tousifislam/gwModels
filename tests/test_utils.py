"""Tests for gwModels.utils submodule."""

import numpy as np
import pytest
import gwModels


class TestFeatures:
    """Tests for amplitude, phase, frequency extraction."""

    def test_get_amplitude_real_signal(self):
        h = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        amp = gwModels.utils.get_amplitude(h)
        np.testing.assert_allclose(amp, np.abs(h))

    def test_get_phase_returns_unwrapped(self):
        t = np.linspace(0, 100, 1000)
        h = np.exp(1j * 0.1 * t)
        phase = gwModels.utils.get_phase(h)
        # phase should be monotonically increasing
        assert np.all(np.diff(phase) > 0) or np.all(np.diff(phase) < 0)

    def test_get_frequency_constant_freq(self):
        omega0 = 0.05
        t = np.linspace(0, 1000, 10000)
        h = np.exp(1j * omega0 * t)
        freq = gwModels.utils.get_frequency(t, h)
        np.testing.assert_allclose(freq, omega0, atol=1e-6)

    def test_get_gw_frequency_is_freq_over_pi(self):
        omega0 = 0.05
        t = np.linspace(0, 1000, 10000)
        h = np.exp(1j * omega0 * t)
        f_gw = gwModels.utils.get_gw_frequency(t, h)
        freq = gwModels.utils.get_frequency(t, h)
        np.testing.assert_allclose(f_gw, freq / np.pi, atol=1e-10)


class TestMetrics:
    """Tests for waveform comparison metrics."""

    def test_mathcalE_identical_waveforms(self):
        h = np.random.randn(100) + 1j * np.random.randn(100)
        err = gwModels.utils.mathcalE_error(h, h)
        assert abs(err) < 1e-12

    def test_mathcalE_different_waveforms(self):
        h1 = np.ones(100, dtype=complex)
        h2 = 2 * np.ones(100, dtype=complex)
        err = gwModels.utils.mathcalE_error(h1, h2)
        assert err > 0

    def test_simple_mismatch_identical(self):
        h = np.random.randn(100) + 1j * np.random.randn(100)
        mm = gwModels.utils.simple_mismatch(h, h)
        assert abs(mm) < 1e-12

    def test_simple_mismatch_orthogonal(self):
        h1 = np.array([1.0, 0.0], dtype=complex)
        h2 = np.array([0.0, 1.0], dtype=complex)
        mm = gwModels.utils.simple_mismatch(h1, h2)
        np.testing.assert_allclose(mm, 1.0, atol=1e-12)

    def test_simple_mismatch_zero_norm(self):
        h1 = np.zeros(10, dtype=complex)
        h2 = np.ones(10, dtype=complex)
        assert np.isnan(gwModels.utils.simple_mismatch(h1, h2))


class TestPeakFinders:
    """Tests for peak detection utilities."""

    def test_peak_finder_scipy_finds_peaks(self):
        t = np.linspace(-1000, -10, 10000)
        signal = np.sin(0.05 * t)
        pf = gwModels.utils.PeakFinderScipy(t, signal, distance_btw_peaks=100)
        assert len(pf.max_indx) > 0
        assert len(pf.min_indx) > 0

    def test_peak_finder_crude_finds_peaks(self):
        t = np.linspace(0, 100, 1000)
        signal = np.sin(0.5 * t)
        pf = gwModels.utils.PeakFinderCrude(t, signal)
        assert len(pf.max_indx) > 0
        assert len(pf.min_indx) > 0

    def test_peak_finder_scipy_maxima_above_minima(self):
        t = np.linspace(-1000, -10, 10000)
        signal = np.sin(0.05 * t)
        pf = gwModels.utils.PeakFinderScipy(t, signal, distance_btw_peaks=100)
        assert np.all(signal[pf.max_indx] > 0)
        assert np.all(signal[pf.min_indx] < 0)


class TestAlignment:
    """Tests for waveform alignment utilities."""

    def test_get_peak(self):
        t = np.linspace(-100, 100, 10000)
        signal = np.exp(-t**2 / 10)
        tpeak, hpeak = gwModels.utils.get_peak(t, signal)
        assert abs(tpeak) < 0.1
        np.testing.assert_allclose(hpeak, 1.0, atol=0.01)
