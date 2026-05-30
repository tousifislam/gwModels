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


