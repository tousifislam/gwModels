#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: compute_ecc_xi.py
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

from ..utils.features import get_frequency
from ..utils.alignment import mathcalE_error, phase_align_dict, check_pi_rotation, get_peak
from ..utils.compute_local_peaks import PeakFinderScipy
from ..frameworks.gwnrhme import NRHME

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

    # Available pseudo-PN correction terms: (power of tau/tau_0, parameter name)
    PSEUDO_PN_TERMS = {
        'm1over8': -1/8,
        'm7over8': -7/8,
        'm8over8': -1,
    }

    # Maps order string to which pseudo-PN terms to include
    FIT_ORDER_MAP = {
        '2PN':                          [],
        '3PN':                          [],
        '3PN_m1over8':                  ['m1over8'],
        '3PN_m7over8':                  ['m7over8'],
        '3PN_m8over8':                  ['m8over8'],
        '3PN_m1over8_m8over8':          ['m1over8', 'm8over8'],
        '3PN_m1over8_m7over8':          ['m1over8', 'm7over8'],
        '3PN_m7over8_m8over8':          ['m7over8', 'm8over8'],
        '3PN_m1over8_m7over8_m8over8':  ['m1over8', 'm7over8', 'm8over8'],
    }

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
            fit_funcs_orders: orders of the upper and lower xi fit functions;
                              available options: keys of FIT_ORDER_MAP
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
        if fit_funcs_orders is None:
            self._fit_order_upper = '3PN'
            self._fit_order_lower = '3PN'
        else:
            self._fit_order_upper = fit_funcs_orders[0]
            self._fit_order_lower = fit_funcs_orders[1]

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
    # PN eccentricity evolution
    # -------------------------------------------------------------------------

    def _compute_tau(self, t, tau_0=None):
        """Compute dimensionless time variables tau and tau_0."""
        eta = gwtools.q_to_nu(self.q)
        tau = (self.tc - t) * (eta / 5)
        if tau_0 is None:
            tau_0 = (self.tc - self.t_ref) * (eta / 5)
        return tau, tau_0, eta

    def Newtonian_e_t(self, t, e_0, q, tau_0=None):
        """
        Newtonian eccentricity evolution.
        Page 41, Eq C1 of https://arxiv.org/pdf/1605.00304
        """
        eta = gwtools.q_to_nu(q)
        tau = (self.tc - t) * (eta / 5)
        if tau_0 is None:
            tau_0 = (self.tc - self.t_ref) * (eta / 5)
        return e_0 * (tau / tau_0) ** (19 / 48)

    def PN_order2_e_t(self, t, e_0, q, tau_0=None):
        """2PN eccentricity evolution equation."""
        eta = gwtools.q_to_nu(q)
        tau = (self.tc - t) * (eta / 5)
        if tau_0 is None:
            tau_0 = (self.tc - self.t_ref) * (eta / 5)

        term1 = (tau / tau_0) ** (19 / 48)
        term2 = -4445 / 6912 * (tau ** (-1 / 4) - tau_0 ** (-1 / 4))
        term3 = 854531845 / 4682022912 * tau ** (-1 / 2)
        term4 = 1081754605 / 4682022912 * tau_0 ** (-1 / 2)
        term5 = -19758025 / 47775744 * tau ** (-1 / 4) * tau_0 ** (-1 / 4)
        term6 = -3721 / 33177600 * np.pi ** 2 * tau ** (-3 / 8) * tau_0 ** (-3 / 8)
        term7 = (255918223951763603 / 186891372173721600
                 - 15943 / 80640 * np.euler_gamma
                 - 7926071 / 66355200 * np.pi ** 2) * tau ** (-3 / 4)
        term8 = (-250085444105408603 / 186891372173721600
                 + 15943 / 80640 * np.euler_gamma
                 + 7933513 / 66355200 * np.pi ** 2) * tau_0 ** (-3 / 4)

        g_2pn = term1 * (1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
        return e_0 * g_2pn

    def PN_e_t(self, t, e_0, q, tau_0=None):
        """
        3PN eccentricity evolution equation.
        Page 41, Eq C1 of https://arxiv.org/pdf/1605.00304
        """
        eta = gwtools.q_to_nu(q)
        tau = (self.tc - t) * (eta / 5)
        if tau_0 is None:
            tau_0 = (self.tc - self.t_ref) * (eta / 5)

        term1 = e_0 * (tau / tau_0) ** (19 / 48)
        term2 = (-4445 / 6912 + 185 / 576 * eta) * (tau ** (-1 / 4) - tau_0 ** (-1 / 4))
        term3 = -61 / 5760 * np.pi * (tau ** (-3 / 8) - tau_0 ** (-3 / 8))
        term4 = (854531845 / 4682022912 - 15215083 / 27869184 * eta + 72733 / 663552 * eta ** 2) * tau ** (-1 / 2)
        term5 = (1081754605 / 4682022912 + 3702533 / 27869184 * eta - 4283 / 663552 * eta ** 2) * tau_0 ** (-1 / 2)
        term6 = (-19758025 / 47775744 + 822325 / 1990656 * eta - 34225 / 331776 * eta ** 2) * tau ** (-1 / 4) * tau_0 ** (-1 / 4)
        term7 = (104976437 / 278691840 - 4848113 / 23224320 * eta) * np.pi * tau ** (-5 / 8)
        term8 = (-101180407 / 278691840 + 4690123 / 23224320 * eta) * np.pi * tau_0 ** (-5 / 8)
        term9 = np.pi * (-54229 / 7962624 + 2257 / 663552 * eta) * (tau ** (-1 / 4) * tau_0 ** (-3 / 8) + tau ** (-3 / 8) * tau_0 ** (-1 / 4))
        term10 = (-686914174175 / 4623163195392 - 10094675555 / 898948399104 * eta + 501067585 / 10701766656 * eta ** 2 - 792355 / 382205952 * eta ** 3) * tau ** (-1 / 4) * tau_0 ** (-1 / 2)
        term11 = -3721 / 33177600 * np.pi ** 2 * tau ** (-3 / 8) * tau_0 ** (-3 / 8)
        term12 = (542627721575 / 4623163195392 - 122769222935 / 299649466368 * eta + 2630889335 / 10701766656 * eta ** 2 - 13455605 / 382205952 * eta ** 3) * tau ** (-1 / 2) * tau_0 ** (-1 / 4)
        term13 = (255918223951763603 / 186891372173721600
                  - 15943 / 80640 * np.euler_gamma
                  - 7926071 / 66355200 * np.pi ** 2
                  + (-81120341684927 / 13484225986560 + 12751 / 49152 * np.pi ** 2) * eta
                  - 3929671247 / 32105299968 * eta ** 2
                  + 25957133 / 1146617856 * eta ** 3
                  - 8453 / 15120 * np.log(2)
                  + 26001 / 71680 * np.log(3)
                  + 15943 / 645120 * np.log(tau)) * tau ** (-3 / 4)
        term14 = (-250085444105408603 / 186891372173721600
                  + 15943 / 80640 * np.euler_gamma
                  + 7933513 / 66355200 * np.pi ** 2
                  + (86796376850327 / 13484225986560 - 12751 / 49152 * np.pi ** 2) * eta
                  - 5466199513 / 32105299968 * eta ** 2
                  + 16786747 / 1146617856 * eta ** 3
                  + 8453 / 15120 * np.log(2)
                  - 26001 / 71680 * np.log(3)
                  - 15943 / 645120 * np.log(tau_0)) * tau_0 ** (-3 / 4)

        result = term1 * (1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
                          + term9 + term10 + term11 + term12 + term13 + term14)
        return result

    # -------------------------------------------------------------------------
    # Fit functions — unified via _build_fit_func
    # -------------------------------------------------------------------------

    def _build_fit_func(self, order):
        """
        Build a fit function for the given PN order string.

        Returns a callable with signature f(t, e_0, *pseudo_coeffs) suitable
        for scipy.optimize.curve_fit.
        """
        use_2pn = order.startswith('2PN')
        pseudo_terms = self.FIT_ORDER_MAP[order]
        pseudo_powers = [self.PSEUDO_PN_TERMS[k] for k in pseudo_terms]

        def fit_func(t, e_0, *coeffs):
            eta = gwtools.q_to_nu(self.q)
            tau = (self.tc - t) * (eta / 5)
            tau_0 = (self.tc - self.t_ref) * (eta / 5)

            if use_2pn:
                e_base = self.PN_order2_e_t(t, e_0, self.q, tau_0)
            else:
                e_base = self.PN_e_t(t, e_0, self.q, tau_0)

            # Add pseudo-PN correction terms
            leading = e_0 * (tau / tau_0) ** (19 / 48)
            for coeff, power in zip(coeffs, pseudo_powers):
                e_base = e_base + leading * coeff * (tau / tau_0) ** power

            return e_base

        return fit_func

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
        fit_func = self._build_fit_func(self._fit_order_upper)
        self.popt_maximas, self.pcov_maximas = curve_fit(fit_func, self.t_maximas, self.y_maximas, maxfev=25000)
        self._fit_func_upper = fit_func

    def _fit_minimas(self):
        """Fit the minimas using scipy.curve_fit."""
        fit_func = self._build_fit_func(self._fit_order_lower)
        self.popt_minimas, self.pcov_minimas = curve_fit(fit_func, self.t_minimas, self.y_minimas, maxfev=25000)
        self._fit_func_lower = fit_func

    def _get_upper_xi_envelope(self):
        """Obtain upper envelope of the modulation time series."""
        self.xi_upper = self._fit_func_upper(self.time_xi, *self.popt_maximas)
        if self.set_unphysical_xi_to_zero:
            _sanitize_array(self.xi_upper)

    def _get_lower_xi_envelope(self):
        """Obtain lower envelope of the modulation time series."""
        self.xi_lower = self._fit_func_lower(self.time_xi, *self.popt_minimas)
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
        self.ecc_xi = self.xi_avg.copy()
        if self.set_unphysical_ecc_to_zero:
            _sanitize_array(self.ecc_xi)

    def _compute_eccentricity_at_tref(self):
        """Compute eccentricity at t_ref."""
        self.ecc_ref = gwtools.interpolate_h(self.time_xi, self.ecc_xi, [self.t_ref])[0]
        logger.info('gwModels eccentricity at t_ref=%.2f : %.5f', self.t_ref, self.ecc_ref)

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def plot_xi(self, figsize=(8, 5)):
        """Plot the modulation parameter xi."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.modulations, color='C0', markersize=10, alpha=0.7)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('${\\xi}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show()

    def plot_xi_with_peaks(self, figsize=(8, 5)):
        """Plot xi with periastron and apastron peaks."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.modulations, color='C0', markersize=10, alpha=0.7)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Maximas')
        plt.plot(self.t_minimas, -self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Minimas')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('${\\xi}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()

    def plot_maximas_fit(self, figsize=(8, 5)):
        """Plot upper envelope fits."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Numerical')
        plt.plot(self.time_xi, self.xi_upper, label='PN fit with $e_0=%.3f$' % self.popt_maximas[0], c='C0')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$|{\\xi}_{\\rm upper}^{\\rm env}|$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()

    def plot_minimas_fit(self, figsize=(8, 5)):
        """Plot lower envelope fits."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.t_minimas, self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Numerical')
        plt.plot(self.time_xi, self.xi_lower, label='PN fit with $e_0=%.3f$' % self.popt_minimas[0], c='C1')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$|{\\xi}_{\\rm lower}^{\\rm env}|$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()

    def plot_fit_errors(self, figsize=(8, 5)):
        """Plot errors in upper and lower envelope fits."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.t_maximas, self.y_maximas - self.xi_upper_interp(self.t_maximas), 'o', markersize=4, label='Upper envelop')
        plt.plot(self.t_minimas, self.y_minimas - self.xi_lower_interp(self.t_minimas), 's', markersize=4, label='Lower envelop')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Fit errors', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()

    def plot_xi_with_peaks_and_fits(self, figsize=(8, 5)):
        """Plot xi with upper and lower envelope fits."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.modulations, color='C0', markersize=10, alpha=0.7)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Maximas')
        plt.plot(self.t_minimas, -self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Minimas')
        plt.plot(self.time_xi, self.xi_upper, label='PN fit with $e_0=%.3f$' % self.popt_maximas[0], c='C0')
        plt.plot(self.time_xi, -self.xi_lower, label='PN fit with $e_0=%.3f$' % self.popt_minimas[0], c='C1')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('${\\xi}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()

    def plot_maximas_and_minimas_fit(self, figsize=(8, 5)):
        """Plot both upper and lower envelope fits."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$')
        plt.plot(self.time_xi, self.xi_upper, label='PN fit with $e_0=%.3f$' % self.popt_maximas[0], c='C0')
        plt.plot(self.t_minimas, self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$')
        plt.plot(self.time_xi, self.xi_lower, label='PN fit with $e_0=%.3f$' % self.popt_minimas[0], c='C1')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$|{\\xi}^{\\rm env}|$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()

    def plot_maximas_minimas_and_avg_fit(self, figsize=(8, 5)):
        """Plot upper, lower, and average envelope fits."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$')
        plt.plot(self.time_xi, self.xi_upper, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$ fit with $e_0=%.3f$' % self.popt_maximas[0], c='C0')
        plt.plot(self.t_minimas, self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$')
        plt.plot(self.time_xi, self.xi_lower, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$ fit with $e_0=%.3f$' % self.popt_minimas[0], c='C1')
        plt.plot(self.time_xi, self.xi_avg, label='$|{\\xi}_{\\rm avg}^{\\rm env}|$', c='k', ls='--')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$|{\\xi}^{\\rm env}|$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()

    def plot_eccentricity(self, figsize=(8, 5)):
        """Plot eccentricity evolution."""
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.ecc_xi, c='C0', ls='-')
        plt.axvline(x=self.t_ref, c='k', ls='--')
        plt.text(self.t_ref + 10, self.ecc_ref * 0.5, '$e_{\\rm ref}$', fontsize=14, color='red', rotation=90)
        plt.plot(self.t_ref, self.ecc_ref, 'o', color='red')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$e_{\\xi}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show()


class ComputeEccentricity:
    """
    Class to compute eccentricity using 22 mode eccentric and circular waveforms.

    This is a convenience wrapper that:
    1. Computes modulations via NRHME
    2. Delegates eccentricity extraction to ComputeEccentricityFromModulations
    """
    def __init__(self, t_ecc=None, h_ecc_dict=None, t_cir=None, h_cir_dict=None,
                 q=None, t_ref=None, ecc_prefactor=None, distance_btw_peaks=None,
                 fit_funcs_orders=None, include_zero_zero=False,
                 set_unphysical_xi_to_zero=False, set_unphysical_ecc_to_zero=True,
                 method='xi_amp', use_xi_amp_to_get_xi_freq=False, tc=0, t_buffer=0):
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
            fit_funcs_orders: orders of the upper and lower xi fit functions
            include_zero_zero: if True, include (t=0, y=0) to extrema lists
            set_unphysical_xi_to_zero: if True, set negative/NaN in fitted xi to zero
            set_unphysical_ecc_to_zero: if True, set negative/NaN in fitted eccentricity to zero
            method: 'xi_amp' or 'xi_freq'
            use_xi_amp_to_get_xi_freq: if True, compute freq modulation from amp modulation
            tc: time at merger; default is zero
            t_buffer: buffer time for common time grid
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

        self.t_ecc = t_ecc
        self.h_ecc_dict = h_ecc_dict
        self.t_cir = t_cir
        self.h_cir_dict = h_cir_dict
        self.q = q
        self.method = method
        self.use_xi_amp_to_get_xi_freq = use_xi_amp_to_get_xi_freq

        # Obtain modulations from eccentric and circular data
        self.gwnrhme_obj = NRHME(t_ecc=self.t_ecc,
                                 h_ecc_dict={'h_l2m2': self.h_ecc_dict['h_l2m2']},
                                 t_cir=self.t_cir,
                                 h_cir_dict={'h_l2m2': self.h_cir_dict['h_l2m2']},
                                 project_ecc_on_higher_modes=False,
                                 t_buffer=t_buffer)

        # Amplitude-to-frequency modulation scaling
        B = 0.9

        # Use only pre-merger data
        t_premerger = self.gwnrhme_obj.t_common[self.gwnrhme_obj.t_common <= 0]

        if self.method == 'xi_amp':
            modulations_premerger = self.gwnrhme_obj.xi_amp[self.gwnrhme_obj.t_common <= 0] / B
        elif self.method == 'xi_freq':
            if self.use_xi_amp_to_get_xi_freq:
                modulations_premerger = self.gwnrhme_obj.xi_amp[self.gwnrhme_obj.t_common <= 0] / B
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
        # __getattr__ is only called when normal lookup fails,
        # so this won't intercept attributes set in __init__
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
