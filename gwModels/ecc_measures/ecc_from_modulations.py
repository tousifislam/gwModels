#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: ecc_from_modulations.py
#    Computes eccentricity given eccentric and circular waveforms or
#    given eccentric modulations
#
#       AUTHOR: Tousif Islam
#       CREATED: 08-08-2024
#       REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import logging
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import gwtools

from ..utils.metrics import mathcalE_error
from ..utils.compute_local_peaks import PeakFinderScipy
from ..frameworks.gwnrhme import NRHME
from .pn_eccentricity import Newtonian_e_t, PN2_e_t, PN3_e_t
from ..utils.constants import B_AMP_FREQ
from . import plotting

logger = logging.getLogger(__name__)


def _sanitize_array(arr):
    """Set NaN and negative values to zero in-place."""
    arr[np.isnan(arr)] = 0.0
    arr[arr < 0] = 0.0
    return arr


class ComputeEccentricityFromModulations:
    """
    Class to compute eccentricity using eccentric modulation parameter.
    """

    def __init__(self, time_xi, xi, q, t_ref=None, ecc_prefactor=None,
                 distance_btw_peaks=100, fit_funcs_orders=None,
                 include_zero_zero=False, set_unphysical_xi_to_zero=False,
                 set_unphysical_ecc_to_zero=False, tc=0):
        """
        Parameters:
            time_xi: time axis
            xi: common modulation parameter
            q: mass ratio (q>=1)
            t_ref: reference time to compute eccentricity
            ecc_prefactor: pre-factor in eccentricity definition; default is 2/3
            distance_btw_peaks: distance between peaks for PeakFinderScipy; default: 100
            fit_funcs_orders: list of two strings specifying the PN fit orders
                              for the upper and lower xi envelopes, respectively.
                              Available options: '2PN', '3PN', '3PN_m1over8',
                              '3PN_m7over8', '3PN_m8over8', '3PN_m1over8_m8over8',
                              '3PN_m1over8_m7over8', '3PN_m7over8_m8over8',
                              '3PN_m1over8_m7over8_m8over8'.
                              Default: ['3PN_m1over8', '3PN_m1over8']
            include_zero_zero: if True, include (t=0, y=0) to extrema lists
            set_unphysical_xi_to_zero: if True, set negative/NaN values in fitted xi to zero
            set_unphysical_ecc_to_zero: if True, set negative/NaN values in fitted eccentricity to zero
            tc: time at merger; default is zero
        """
        self.time_xi = time_xi
        self.modulations = xi
        self.q = q
        self.tc = tc

        if t_ref is None:
            self.t_ref = self.time_xi[0] + 10
        else:
            self.t_ref = t_ref
            if self.t_ref <= self.time_xi[0] + 5:
                warnings.warn("t_ref is very close to the start of the waveform. "
                              "Consider using a t_ref at least 10M larger than the earliest time.")

        self.distance_btw_peaks = distance_btw_peaks
        self.include_zero_zero = include_zero_zero
        self.set_unphysical_xi_to_zero = set_unphysical_xi_to_zero
        self.set_unphysical_ecc_to_zero = set_unphysical_ecc_to_zero

        # Resolve fit functions from order strings
        self.fit_funcs_orders = fit_funcs_orders
        if self.fit_funcs_orders is None:
            self.fit_funcs = [self.fit_func_3PN_m1over8, self.fit_func_3PN_m1over8]
        else:
            self.fit_funcs = [self.PNorder_to_func_translation(self.fit_funcs_orders[0]),
                              self.PNorder_to_func_translation(self.fit_funcs_orders[1])]

        # Eccentricity prefactor; Eq (34) of https://arxiv.org/pdf/1702.00872
        self.ecc_prefactor = ecc_prefactor if ecc_prefactor is not None else 2/3

        # Scale modulations so that it represents eccentricity better
        self.modulations = self.ecc_prefactor * self.modulations

        # Perform peak fits and compute eccentricity
        self._get_maximas_minimas()
        self._fit_maximas()
        self._fit_minimas()
        self._get_upper_xi_envelope()
        self._get_lower_xi_envelope()
        self._get_avg_xi_envelope()

        # Time up to which eccentricity estimations are correct
        self.teccmax = max(max(self.t_minimas), max(self.t_maximas))

        self._compute_eccentricity_evolution()
        self._compute_eccentricity_at_tref()

        # Build interpolations
        self.xi_upper_interp = interpolate.interp1d(self.time_xi, self.xi_upper)
        self.xi_lower_interp = interpolate.interp1d(self.time_xi, self.xi_lower)
        self.ecc_interp = interpolate.interp1d(self.time_xi, self.ecc_xi)

        # Fit errors
        self.xi_upper_fit_error = mathcalE_error(self.y_maximas, self.xi_upper_interp(self.t_maximas))
        self.xi_lower_fit_error = mathcalE_error(self.y_minimas, self.xi_lower_interp(self.t_minimas))

    # -------------------------------------------------------------------------
    # PN eccentricity evolution (delegate to standalone functions)
    # -------------------------------------------------------------------------

    def Newtonian_e_t(self, t, e_0, q, tau_0=None):
        """Newtonian eccentricity evolution."""
        t_ref = self.t_ref if tau_0 is None else None
        if tau_0 is not None:
            eta = gwtools.q_to_nu(q)
            t_ref_from_tau0 = self.tc - tau_0 * 5 / eta
            return Newtonian_e_t(t, e_0, q, tc=self.tc, t_ref=t_ref_from_tau0)
        return Newtonian_e_t(t, e_0, q, tc=self.tc, t_ref=t_ref)

    def PN_order2_e_t(self, t, e_0, q, tau_0=None):
        """2PN eccentricity evolution."""
        t_ref = self.t_ref if tau_0 is None else None
        if tau_0 is not None:
            eta = gwtools.q_to_nu(q)
            t_ref_from_tau0 = self.tc - tau_0 * 5 / eta
            return PN2_e_t(t, e_0, q, tc=self.tc, t_ref=t_ref_from_tau0)
        return PN2_e_t(t, e_0, q, tc=self.tc, t_ref=t_ref)

    def PN_e_t(self, t, e_0, q, tau_0=None):
        """3PN eccentricity evolution."""
        t_ref = self.t_ref if tau_0 is None else None
        if tau_0 is not None:
            eta = gwtools.q_to_nu(q)
            t_ref_from_tau0 = self.tc - tau_0 * 5 / eta
            return PN3_e_t(t, e_0, q, tc=self.tc, t_ref=t_ref_from_tau0)
        return PN3_e_t(t, e_0, q, tc=self.tc, t_ref=t_ref)

    # -------------------------------------------------------------------------
    # Fit functions
    # -------------------------------------------------------------------------

    def PNorder_to_func_translation(self, order):
        """Translate fit function PN order string to fit function callable."""
        PNorder_to_func_dict = {
            '2PN': self.fit_func_2PN,
            '3PN': self.fit_func_3PN,
            '3PN_m1over8': self.fit_func_3PN_m1over8,
            '3PN_m7over8': self.fit_func_3PN_m7over8,
            '3PN_m8over8': self.fit_func_3PN_m8over8,
            '3PN_m1over8_m8over8': self.fit_func_3PN_m1over8_m8over8,
            '3PN_m1over8_m7over8': self.fit_func_3PN_m1over8_m7over8,
            '3PN_m7over8_m8over8': self.fit_func_3PN_m7over8_m8over8,
            '3PN_m1over8_m7over8_m8over8': self.fit_func_3PN_m1over8_m7over8_m8over8,
        }
        return PNorder_to_func_dict[order]

    def fit_func_2PN(self, t, e_0):
        return PN2_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)

    def fit_func_3PN(self, t, e_0):
        return PN3_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)

    def fit_func_3PN_m1over8(self, t, e_0, A1):
        eta = gwtools.q_to_nu(self.q)
        tau = (self.tc - t) * (eta / 5)
        tau_0 = (self.tc - self.t_ref) * (eta / 5)
        e_3PN = PN3_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)
        e_m1over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A1 * (tau / tau_0) ** (-1 / 8)
        return e_3PN + e_m1over8

    def fit_func_3PN_m7over8(self, t, e_0, A7):
        eta = gwtools.q_to_nu(self.q)
        tau = (self.tc - t) * (eta / 5)
        tau_0 = (self.tc - self.t_ref) * (eta / 5)
        e_3PN = PN3_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)
        e_m7over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A7 * (tau / tau_0) ** (-7 / 8)
        return e_3PN + e_m7over8

    def fit_func_3PN_m8over8(self, t, e_0, A8):
        eta = gwtools.q_to_nu(self.q)
        tau = (self.tc - t) * (eta / 5)
        tau_0 = (self.tc - self.t_ref) * (eta / 5)
        e_3PN = PN3_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)
        e_m8over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A8 * (tau / tau_0) ** (-8 / 8)
        return e_3PN + e_m8over8

    def fit_func_3PN_m1over8_m8over8(self, t, e_0, A1, A8):
        eta = gwtools.q_to_nu(self.q)
        tau = (self.tc - t) * (eta / 5)
        tau_0 = (self.tc - self.t_ref) * (eta / 5)
        e_3PN = PN3_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)
        e_m1over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A1 * (tau / tau_0) ** (-1 / 8)
        e_m8over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A8 * (tau / tau_0) ** (-1)
        return e_3PN + e_m1over8 + e_m8over8

    def fit_func_3PN_m1over8_m7over8(self, t, e_0, A1, A7):
        eta = gwtools.q_to_nu(self.q)
        tau = (self.tc - t) * (eta / 5)
        tau_0 = (self.tc - self.t_ref) * (eta / 5)
        e_3PN = PN3_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)
        e_m1over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A1 * (tau / tau_0) ** (-1 / 8)
        e_m7over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A7 * (tau / tau_0) ** (-7 / 8)
        return e_3PN + e_m1over8 + e_m7over8

    def fit_func_3PN_m7over8_m8over8(self, t, e_0, A7, A8):
        eta = gwtools.q_to_nu(self.q)
        tau = (self.tc - t) * (eta / 5)
        tau_0 = (self.tc - self.t_ref) * (eta / 5)
        e_3PN = PN3_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)
        e_m7over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A7 * (tau / tau_0) ** (-7 / 8)
        e_m8over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A8 * (tau / tau_0) ** (-1)
        return e_3PN + e_m7over8 + e_m8over8

    def fit_func_3PN_m1over8_m7over8_m8over8(self, t, e_0, A1, A7, A8):
        eta = gwtools.q_to_nu(self.q)
        tau = (self.tc - t) * (eta / 5)
        tau_0 = (self.tc - self.t_ref) * (eta / 5)
        e_3PN = PN3_e_t(t, e_0, self.q, tc=self.tc, t_ref=self.t_ref)
        e_m1over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A1 * (tau / tau_0) ** (-1 / 8)
        e_m7over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A7 * (tau / tau_0) ** (-7 / 8)
        e_m8over8 = (e_0 * (tau / tau_0) ** (19 / 48)) * A8 * (tau / tau_0) ** (-1)
        return e_3PN + e_m1over8 + e_m7over8 + e_m8over8

    # -------------------------------------------------------------------------
    # Peak finding and fitting
    # -------------------------------------------------------------------------

    def _get_maximas_minimas(self):
        """Obtain all maximas and minimas in a xi time series."""
        obj = PeakFinderScipy(time=self.time_xi,
                              signal=self.modulations,
                              distance_btw_peaks=self.distance_btw_peaks)

        self.t_minimas = obj.time[obj.min_indx]
        self.y_minimas = -obj.signal[obj.min_indx]

        self.t_maximas = obj.time[obj.max_indx]
        self.y_maximas = obj.signal[obj.max_indx]

        if self.include_zero_zero:
            self.t_maximas = np.concatenate((self.t_maximas, [0]))
            self.t_minimas = np.concatenate((self.t_minimas, [0]))
            self.y_maximas = np.concatenate((self.y_maximas, [0]))
            self.y_minimas = np.concatenate((self.y_minimas, [0]))

    def _fit_maximas(self):
        """Fit the maximas using scipy.curve_fit."""
        self.popt_maximas, self.pcov_maximas = curve_fit(self.fit_funcs[0], self.t_maximas, self.y_maximas, maxfev=25000)

    def _fit_minimas(self):
        """Fit the minimas using scipy.curve_fit."""
        self.popt_minimas, self.pcov_minimas = curve_fit(self.fit_funcs[1], self.t_minimas, self.y_minimas, maxfev=25000)

    def _get_upper_xi_envelope(self):
        """Obtain upper envelope of the modulation time series."""
        self.xi_upper = self.fit_funcs[0](self.time_xi, *self.popt_maximas)
        if self.set_unphysical_xi_to_zero:
            _sanitize_array(self.xi_upper)

    def _get_lower_xi_envelope(self):
        """Obtain lower envelope of the modulation time series."""
        self.xi_lower = self.fit_funcs[1](self.time_xi, *self.popt_minimas)
        if self.set_unphysical_xi_to_zero:
            _sanitize_array(self.xi_lower)

    def _get_avg_xi_envelope(self):
        """Obtain average envelope of the modulation time series."""
        self.xi_avg = 0.5 * (self.xi_upper + self.xi_lower)

    # -------------------------------------------------------------------------
    # Eccentricity computation
    # -------------------------------------------------------------------------

    def _compute_eccentricity_evolution(self):
        """Compute eccentricity evolution using modulation envelopes."""
        self.ecc_xi = self.xi_avg
        if self.set_unphysical_ecc_to_zero:
            _sanitize_array(self.ecc_xi)

    def _compute_eccentricity_at_tref(self):
        """Compute eccentricity at t_ref."""
        self.ecc_ref = gwtools.interpolate_h(self.time_xi, self.ecc_xi, [self.t_ref])[0]
        print('... gwModels eccentricity at t_ref=%.2f : %.5f' % (self.t_ref, self.ecc_ref))

    # -------------------------------------------------------------------------
    # Plotting (delegate to standalone functions)
    # -------------------------------------------------------------------------

    def plot_xi(self, figsize=(8, 5)):
        plotting.plot_xi(self, figsize)

    def plot_xi_with_peaks(self, figsize=(8, 5)):
        plotting.plot_xi_with_peaks(self, figsize)

    def plot_maximas_fit(self, figsize=(8, 5)):
        plotting.plot_maximas_fit(self, figsize)

    def plot_minimas_fit(self, figsize=(8, 5)):
        plotting.plot_minimas_fit(self, figsize)

    def plot_fit_errors(self, figsize=(8, 5)):
        plotting.plot_fit_errors(self, figsize)

    def plot_xi_with_peaks_and_fits(self, figsize=(8, 5)):
        plotting.plot_xi_with_peaks_and_fits(self, figsize)

    def plot_maximas_and_minimas_fit(self, figsize=(8, 5)):
        plotting.plot_maximas_and_minimas_fit(self, figsize)

    def plot_maximas_minimas_and_avg_fit(self, figsize=(8, 5)):
        plotting.plot_maximas_minimas_and_avg_fit(self, figsize)

    def plot_eccentricity(self, figsize=(8, 5)):
        plotting.plot_eccentricity(self, figsize)


class ComputeEccentricity:
    """
    Class to compute eccentricity using 22 mode eccentric and circular waveforms.

    This is a convenience wrapper that:
    1. Computes modulations via a framework class (NRHME or NRXHME)
    2. Delegates eccentricity extraction to ComputeEccentricityFromModulations
    """
    def __init__(self, t_ecc=None, h_ecc_dict=None, t_cir=None, h_cir_dict=None,
                 q=None, t_ref=None, ecc_prefactor=None, distance_btw_peaks=None,
                 fit_funcs_orders=None, include_zero_zero=False,
                 set_unphysical_xi_to_zero=False, set_unphysical_ecc_to_zero=True,
                 method='xi_amp', use_xi_amp_to_get_xi_freq=False, tc=0, t_buffer=0,
                 framework_cls=None):
        """
        Parameters:
            t_ecc: time array for the eccentric 22 mode waveform
            h_ecc_dict: dictionary of eccentric waveform modes (should contain 22 mode)
            t_cir: time array for the circular waveform modes
            h_cir_dict: dictionary of circular non-spinning waveform modes
            q: mass ratio (q>=1)
            t_ref: reference time to compute eccentricity
            ecc_prefactor: pre-factor in eccentricity definition; default is 2/3
            distance_btw_peaks: distance between peaks for PeakFinderScipy
            fit_funcs_orders: list of two strings specifying the PN fit orders
                              for the upper and lower xi envelopes, respectively.
                              Available options: '2PN', '3PN', '3PN_m1over8',
                              '3PN_m7over8', '3PN_m8over8', '3PN_m1over8_m8over8',
                              '3PN_m1over8_m7over8', '3PN_m7over8_m8over8',
                              '3PN_m1over8_m7over8_m8over8'.
                              Default: ['3PN_m1over8', '3PN_m1over8']
            include_zero_zero: if True, include (t=0, y=0) to extrema lists
            set_unphysical_xi_to_zero: if True, set negative/NaN in fitted xi to zero
            set_unphysical_ecc_to_zero: if True, set negative/NaN in fitted eccentricity to zero
            method: 'xi_amp' or 'xi_freq'
            use_xi_amp_to_get_xi_freq: if True, compute freq modulation from amp modulation
            tc: time at merger; default is zero
            t_buffer: buffer time for common time grid
            framework_cls: framework class to use for modulation extraction.
                           Default is NRHME (non-spinning). Use NRXHME for non-precessing.
        """
        if t_ecc is None:
            raise ValueError("t_ecc must be given as input")
        if h_ecc_dict is None:
            raise ValueError("h_ecc_dict must be given as input")
        if t_cir is None:
            raise ValueError("t_cir must be given as input")
        if h_cir_dict is None:
            raise ValueError("h_cir_dict must be given as input")
        if q is None:
            raise ValueError("q must be given as input")

        if framework_cls is None:
            framework_cls = NRHME

        self.t_ecc = t_ecc
        self.h_ecc_dict = h_ecc_dict
        self.t_cir = t_cir
        self.h_cir_dict = h_cir_dict
        self.q = q
        self.method = method
        self.use_xi_amp_to_get_xi_freq = use_xi_amp_to_get_xi_freq

        # Obtain modulations from eccentric and circular data
        self.gwnrhme_obj = framework_cls(t_ecc=self.t_ecc,
                                 h_ecc_dict={'h_l2m2': self.h_ecc_dict['h_l2m2']},
                                 t_cir=self.t_cir,
                                 h_cir_dict={'h_l2m2': self.h_cir_dict['h_l2m2']},
                                 project_ecc_on_higher_modes=False,
                                 t_buffer=t_buffer)

        # Use only pre-merger data
        t_premerger = self.gwnrhme_obj.t_common[self.gwnrhme_obj.t_common <= 0]

        if self.method == 'xi_amp':
            modulations_premerger = self.gwnrhme_obj.xi_amp[self.gwnrhme_obj.t_common <= 0] / B_AMP_FREQ
        elif self.method == 'xi_freq':
            if self.use_xi_amp_to_get_xi_freq:
                modulations_premerger = self.gwnrhme_obj.xi_amp[self.gwnrhme_obj.t_common <= 0] / B_AMP_FREQ
            else:
                modulations_premerger = self.gwnrhme_obj.xi_omega[self.gwnrhme_obj.t_common <= 0]
        else:
            raise ValueError(f"Method '{method}' not recognized. Use 'xi_amp' or 'xi_freq'.")

        # Reference time
        if t_ref is None:
            t_ref = self.gwnrhme_obj.t_common[0] + 10
        else:
            if t_ref <= self.gwnrhme_obj.t_common[0] + 5:
                warnings.warn("t_ref is very close to the start of the shorter waveform. "
                              "Consider using a t_ref at least 10M larger than the earliest time.")
        self.t_ref = t_ref

        # Auto-compute distance between peaks if not given
        if distance_btw_peaks is None:
            self._compute_auto_peak_distance(t_premerger)
        else:
            self.distance_btw_peaks = distance_btw_peaks

        # Compute eccentricity via ComputeEccentricityFromModulations
        self._ecc_obj = ComputeEccentricityFromModulations(
            time_xi=t_premerger,
            xi=modulations_premerger,
            q=self.q,
            t_ref=self.t_ref,
            distance_btw_peaks=self.distance_btw_peaks,
            ecc_prefactor=ecc_prefactor,
            fit_funcs_orders=fit_funcs_orders,
            include_zero_zero=include_zero_zero,
            set_unphysical_xi_to_zero=set_unphysical_xi_to_zero,
            set_unphysical_ecc_to_zero=set_unphysical_ecc_to_zero,
            tc=tc,
        )

    def __getattr__(self, name):
        """Delegate attribute access to the inner ComputeEccentricityFromModulations object."""
        return getattr(self._ecc_obj, name)

    def _compute_auto_peak_distance(self, t_premerger):
        """Compute distance between peaks from approximate cycle count."""
        start_indx = np.where(self.t_ecc > t_premerger[0])[0][0]
        end_indx = np.where(self.t_ecc > 0)[0][0]

        phase_segment = abs(gwtools.phase(self.h_ecc_dict['h_l2m2'][start_indx:end_indx]))
        phase_change = phase_segment[-1] - phase_segment[0]

        self.approx_n_cycle = phase_change / (2 * np.pi)
        self.approx_n_peaks_xi = self.approx_n_cycle / 3

        waveform_duration = abs(self.t_ecc[end_indx] - self.t_ecc[start_indx])
        dt = np.diff(t_premerger)[0]
        t_cycle_in_time = waveform_duration / self.approx_n_peaks_xi
        t_cycle_in_samples = t_cycle_in_time / dt
        self.distance_btw_peaks = int(0.75 * t_cycle_in_samples)
