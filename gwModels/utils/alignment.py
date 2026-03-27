#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: alignment.py
#    [ This piece of code is a modification of a similar code in gw_remnant package
#      written by Tousif Islam ]
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-02-2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools
from scipy.interpolate import InterpolatedUnivariateSpline as spline


def get_peak(t, func):
    """
    Finds the peak time of a function using spline interpolation.

    Fits the provided function with a 4th degree spline and finds
    the maximum among all critical points and endpoints.

    Parameters:
        t (np.ndarray): An array of times.
        func (np.ndarray): An array of function values.

    Returns:
        tuple: (tpeak, fpeak) — the time and value at the peak.
    """
    spl = spline(t, func, k=4)
    cr_pts = spl.derivative().roots()
    # also check the endpoints of the interval
    cr_pts = np.append(cr_pts, (t[0], t[-1]))
    cr_vals = spl(cr_pts)
    max_index = np.argmax(cr_vals)
    return cr_pts[max_index], cr_vals[max_index]


def check_pi_rotation(h_dict):
    """
    Checks whether a pi rotation is required in waveform mode data.

    Determines the phase factor required for each mode based on
    the phase of the h_l2m1 mode, applying a pi rotation for odd m modes
    if necessary.

    Parameters:
        h_dict (dict): Dictionary of gravitational wave modes.
                       Keys should be "h_l2m2", "h_l2m1", etc.

    Returns:
        dict: Updated dictionary after potential rotation.
    """
    phi = gwtools.phase(h_dict['h_l2m1'])
    pi_rot_factor = -1 if phi[0] > 0 else 1

    for mode in h_dict.keys():
        if mode == 'h_l2m2':
            continue
        m = float(mode.rsplit("m")[1])
        if m % 2 != 0:
            h_dict[mode] = (pi_rot_factor ** m) * h_dict[mode]
    return h_dict


def mathcalE_error(h1, h2):
    """
    Computes the time-domain error between two waveforms.

    Calculates the error according to Equation 21 of
    https://arxiv.org/pdf/1701.00550.pdf by normalizing the difference
    between the two waveforms.

    Parameters:
        h1 (np.ndarray): Reference waveform in the time domain.
        h2 (np.ndarray): Comparison waveform in the time domain.

    Returns:
        np.ndarray: Normalized error for each time sample.
    """
    n1Sqr = np.sum(abs(h1) ** 2)
    n2Sqr = np.sum(abs(h2) ** 2)
    sdot = np.real(np.sum(h1 * np.conj(h2)))
    normed_errs = ((n1Sqr + n2Sqr) - 2 * sdot) / (2 * n1Sqr)
    return normed_errs


def phase_align_dict(hdict):
    """
    Aligns a waveform dictionary to ensure proper phases.

    Modifies the phases of the waveform modes to ensure that the
    initial phase of the (2,2) mode is zero and that the relative phases
    of higher modes are consistent.

    Parameters:
        hdict (dict): Dictionary of gravitational wave modes.
                      Keys should include 'h_l2m2', 'h_l3m3', etc.

    Returns:
        dict: A new dictionary with aligned phases for each mode.
    """
    hdict_out = {}
    phi = np.unwrap(np.angle(hdict['h_l2m2']))
    # enforce correct relative phasing of higher modes
    z_rot = phi[0] / 2.0
    phi = phi - phi[0]
    hdict_out['h_l2m2'] = abs(hdict['h_l2m2']) * np.exp(1j * phi)

    for mode in hdict.keys():
        if mode == 'h_l2m2':
            continue
        phi = np.unwrap(np.angle(hdict[mode]))
        m = float(mode.rsplit("m")[1])
        hdict_out[mode] = abs(hdict[mode]) * np.exp(1j * (phi - z_rot * m))
    return hdict_out


class AlignWFData:
    """
    Align a waveform such that the peak is at t=0 and the initial phase is zero.

    Optionally casts the waveform onto a different time grid.

    Parameters:
        t_input (np.ndarray): Input time array.
        h_input (dict): Input waveform dictionary with mode keys like 'h_l2m2'.
                        Must contain 'h_l2m2'.
        t_common (np.ndarray or None): Target time grid. Default: None.
    """
    def __init__(self, t_input, h_input, t_common=None):
        self.t_input = t_input
        self.h_input = h_input
        self.t_common = t_common

        # time alignment
        self.t_peak = self._find_peak_time()
        self.t_transform = self._align_time()

        # if t_common is given, cast the waveform onto the common time grid
        if self.t_common is not None:
            self.h_transform = self._cast_waveform_on_timegrid()
            self.t_transform = self.t_common
        else:
            self.h_transform = self.h_input

        # phase alignment
        self.z_rot = self._find_offset_orb_phase()
        self._align_phase()

    def _cast_waveform_on_timegrid(self):
        """Cast the waveform onto the common time grid via interpolation."""
        h_transform = {}
        for mode in self.h_input.keys():
            h_transform[mode] = gwtools.gwtools.interpolate_h(
                self.t_transform, self.h_input[mode], self.t_common
            )
        return h_transform

    def _find_peak_time(self):
        """Find the time corresponding to the peak of the (2,2) mode."""
        return np.real(get_peak(self.t_input, abs(self.h_input['h_l2m2']))[0])

    def _align_time(self):
        """Align the waveform so that the peak of the (2,2) mode is at t=0."""
        return self.t_input - self.t_peak

    def _find_offset_orb_phase(self):
        """Find the phase rotation to set the initial (2,2) mode phase to zero."""
        phi = gwtools.phase(self.h_transform['h_l2m2'])
        return phi[0] / 2.0

    def _align_phase(self):
        """Enforce correct relative phasing: initial (2,2) mode phase is zero."""
        for mode in self.h_transform.keys():
            phi = gwtools.phase(self.h_transform[mode])
            m = float(mode.rsplit("m")[-1])
            self.h_transform[mode] = abs(self.h_transform[mode]) * np.exp(1j * (phi - self.z_rot * m))
