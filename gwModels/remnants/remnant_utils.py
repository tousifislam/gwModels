#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: remnant_utils.py
#    Shared utility functions for BBH remnant calculations
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.optimize import minimize_scalar


def symmetric_mass_ratio(q):
    """
    Calculate symmetric mass ratio eta(q) with q = m2/m1 <= 1.

    eta = 0.25 for equal masses (q = 1), eta -> 0 for q -> 0.

    Parameters:
        q: Mass ratio (q <= 1).

    Returns:
        float or array: Symmetric mass ratio.
    """
    q = np.asarray(q, dtype=float)
    return q / (1.0 + q) ** 2


def kerr_isco_radius(a):
    """
    Calculate Boyer-Lindquist r_ISCO(a) for equatorial orbits of a Kerr BH.

    Parameters:
        a: Dimensionless spin parameter (|a| <= 1).
           Positive for prograde, negative for retrograde orbits.

    Returns:
        float or array: ISCO radius in units of GM/c^2.
    """
    a = np.asarray(a, dtype=float)
    Z1 = 1.0 + (1 - a ** 2) ** (1 / 3.0) * ((1 + a) ** (1 / 3.0) + (1 - a) ** (1 / 3.0))
    Z2 = np.sqrt(3 * a ** 2 + Z1 ** 2)
    r_isco = 3 + Z2 - np.sign(a) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    return r_isco


def f_q(q):
    """
    Fitchett mass ratio function f(q) = q^2 * (1-q) / (1+q)^5.

    Parameters:
        q: Mass ratio (q <= 1).

    Returns:
        float or array: Dimensionless function value.
    """
    return q ** 2 * (1 - q) / (1 + q) ** 5


def find_f_max():
    """
    Find the maximum value of the Fitchett function f(q).

    Returns:
        tuple: (q_max, f_max) — mass ratio at maximum and the maximum value.
    """
    result = minimize_scalar(lambda q: -f_q(q), bounds=(0, 1), method='bounded')
    return result.x, -result.fun
