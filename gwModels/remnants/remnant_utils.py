#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: remnant_utils.py
#    Shared utility functions for BBH remnant calculations
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 05-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

def validate_q(q):
    """Validate that q = m1/m2 >= 1."""
    q = np.asarray(q, dtype=float)
    if np.any(q < 1):
        raise ValueError("q must be >= 1 (m1/m2).")
    return q


def validate_spin_magnitudes(a1, a2):
    """Validate that spin magnitudes are in [0, 1]."""
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    if np.any((a1 < 0) | (a1 > 1)):
        raise ValueError("a1 must be in [0, 1].")
    if np.any((a2 < 0) | (a2 > 1)):
        raise ValueError("a2 must be in [0, 1].")
    return a1, a2


def validate_spin_z(chi1z, chi2z):
    """Validate that spin z-components are in [-1, 1]."""
    chi1z = np.asarray(chi1z, dtype=float)
    chi2z = np.asarray(chi2z, dtype=float)
    if np.any((chi1z < -1) | (chi1z > 1)):
        raise ValueError("chi1z must be in [-1, 1].")
    if np.any((chi2z < -1) | (chi2z > 1)):
        raise ValueError("chi2z must be in [-1, 1].")
    return chi1z, chi2z


def symmetric_mass_ratio(q):
    """
    Calculate symmetric mass ratio eta(q) = q/(1+q)^2.

    eta = 0.25 for equal masses (q = 1), eta -> 0 for q -> infinity.
    The formula is symmetric under q -> 1/q, so works for any q > 0.

    Parameters:
        q: Mass ratio q = m1/m2 >= 1.

    Returns:
        float or array: Symmetric mass ratio.
    """
    q = np.asarray(q, dtype=float)
    return q / (1.0 + q) ** 2
