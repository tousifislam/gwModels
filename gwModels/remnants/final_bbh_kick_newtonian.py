#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: final_bbh_kick_newtonian.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .remnant_utils import kerr_isco_radius, f_q, find_f_max


def f_spin_orbit(q, a_tilde_1, a_tilde_2):
    """
    Spin-orbit correction function f_SO(q, a1, a2) = q^2 * (a2 - q*a1) / (1+q)^5.

    Parameters:
        q: Mass ratio (q = m1/m2 <= 1, where m1 is the lighter mass).
        a_tilde_1: Dimensionless spin of lighter mass (-1 <= a1 <= 1).
        a_tilde_2: Dimensionless spin of heavier mass (-1 <= a2 <= 1).

    Returns:
        float: Spin-orbit correction function value.
    """
    return q ** 2 * (a_tilde_2 - q * a_tilde_1) / (1 + q) ** 5


def find_f_SO_max():
    """
    Maximum spin-orbit correction.
    Occurs when q=1, a1=-a2=+-1, giving f_SO_max = 1/16.

    Returns:
        float: Maximum value (1/16).
    """
    return 1.0 / 16.0


def bbh_final_kick_nonprecessing_Fitchett1983(q, a, units='km/s'):
    """
    Fitchett (1983) Newtonian kick velocity.
    Eq(1) of https://arxiv.org/pdf/astro-ph/0402056

    Parameters:
        q: Mass ratio (q = m1/m2 <= 1).
        a: Spin value of the larger BH.
        units: Output units ('km/s' or 'm/s').

    Returns:
        float: Fitchett kick velocity.
    """
    r_term = kerr_isco_radius(a)
    q_max, f_max = find_f_max()
    f_ratio = f_q(q) / f_max

    V_F_km_s = 1480.0 * f_ratio * (2.0 / r_term) ** 4

    if units == 'km/s':
        return float(V_F_km_s)
    elif units == 'm/s':
        return float(V_F_km_s * 1000.0)
    else:
        raise ValueError("units must be 'km/s' or 'm/s'")


def bbh_final_kick_nonprecessing_Kidder1995(q, a_tilde_1=0, a_tilde_2=0, units='km/s', all_terms=False):
    """
    Fitchett kick velocity with spin-orbit correction.
    Eq(4) of https://arxiv.org/pdf/astro-ph/0402056

    Parameters:
        q: Mass ratio (q = m1/m2 <= 1).
        a_tilde_1: Dimensionless spin of lighter mass (-1 <= a1 <= 1).
        a_tilde_2: Dimensionless spin of heavier mass (-1 <= a2 <= 1).
        units: Output units ('km/s' or 'm/s').
        all_terms: If True, return (V_total, V_F, V_SO).

    Returns:
        float or tuple: Total kick velocity, or (V_total, V_F, V_SO) if all_terms=True.
    """
    r_term = kerr_isco_radius(a_tilde_2)

    V_F = bbh_final_kick_nonprecessing_Fitchett1983(q, a_tilde_2, units)

    f_SO = f_spin_orbit(q, a_tilde_1, a_tilde_2)
    f_SO_max = find_f_SO_max()

    V_SO_km_s = 1480.0 * (f_SO / f_SO_max) * (2.0 / r_term) ** 4

    if units == 'km/s':
        V_SO = V_SO_km_s
    elif units == 'm/s':
        V_SO = V_SO_km_s * 1000.0
    else:
        raise ValueError("units must be 'km/s' or 'm/s'")

    # Quadrature sum (direction depends on orbital geometry)
    V_total = np.sqrt(V_F ** 2 + V_SO ** 2)

    if all_terms:
        return float(V_total), float(V_F), float(V_SO)
    return float(V_total)
