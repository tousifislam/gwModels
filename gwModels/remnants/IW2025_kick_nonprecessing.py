#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: IW_final_kick_nonprecessing.py
#
#    Kick formula by Islam and Wadekar 2025 for aligned-spin binaries.
#    From Islam & Wadekar (2025), https://arxiv.org/abs/2511.11536
#    Based on RIT formula from arXiv:1406.7295, refitted to expanded dataset
#    including extreme mass ratios from BHPT (q up to ~128).
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 05-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .remnant_utils import validate_q, validate_spin_z

# Fitted parameters (median/best-fit values)
_PARAMS_MEDIAN = {
    'A': 1.177151e+04,
    'B': -9.281482e-01,
    'H': 7.410261e+03,
    'H2a': 5.845639e+00,
    'H2b': -7.440300e-01,
    'H3a': -6.095334e-01,
    'H3b': -1.321148e+00,
    'H3c': -1.442264e+00,
    'H3d': -1.790316e-02,
    'H3e': 6.691620e+00,
    'H4a': -8.580474e-01,
    'H4b': -2.668094e+00,
    'H4c': 3.622004e+00,
    'H4d': -2.214556e+00,
    'H4e': 1.395472e+00,
    'H4f': 2.920338e-01,
    'a_deg': 1.468588e+02,
    'b_deg': 1.107239e+02,
    'c_deg': 1.346647e+02
}

# Parameter uncertainties (standard errors)
_PARAMS_STD = {
    'A': 9.505196e+01,
    'B': 3.629754e-02,
    'H': 2.804217e+01,
    'H2a': 4.100153e-02,
    'H2b': 1.863484e-02,
    'H3a': 3.989919e-02,
    'H3b': 9.692577e-02,
    'H3c': 7.737928e-02,
    'H3d': 6.588784e-03,
    'H3e': 7.858664e-02,
    'H4a': 1.057504e-01,
    'H4b': 1.467482e-01,
    'H4c': 9.840863e-02,
    'H4d': 1.438000e-01,
    'H4e': 1.733390e-01,
    'H4f': 5.419867e-02,
    'a_deg': 3.712547e-01,
    'b_deg': 1.812909e+00,
    'c_deg': 2.957219e+00
}


def _compute_kick(q, chi1z, chi2z, params):
    eta = q / (1.0 + q)**2
    delta_m = (q - 1.0) / (q + 1.0)

    S_tilde_par = (chi1z + q**2 * chi2z) / (1.0 + q)**2
    Delta_tilde_par = (chi1z - q * chi2z) / (1.0 + q)

    poly = (
        Delta_tilde_par
        + params['H2a'] * S_tilde_par * delta_m
        + params['H2b'] * Delta_tilde_par * S_tilde_par
        + params['H3a'] * (Delta_tilde_par**2) * delta_m
        + params['H3b'] * (S_tilde_par**2) * delta_m
        + params['H3c'] * Delta_tilde_par * (S_tilde_par**2)
        + params['H3d'] * (Delta_tilde_par**3)
        + params['H3e'] * Delta_tilde_par * (delta_m**2)
        + params['H4a'] * S_tilde_par * (Delta_tilde_par**2) * delta_m
        + params['H4b'] * (S_tilde_par**3) * delta_m
        + params['H4c'] * S_tilde_par * (delta_m**3)
        + params['H4d'] * Delta_tilde_par * S_tilde_par * (delta_m**2)
        + params['H4e'] * Delta_tilde_par * (S_tilde_par**3)
        + params['H4f'] * S_tilde_par * (Delta_tilde_par**3)
    )
    Vperp = params['H'] * (eta**2) * poly
    Vm = params['A'] * (eta**2) * delta_m * (1.0 + params['B'] * eta)
    xi = np.deg2rad(params['a_deg'] + params['b_deg'] * S_tilde_par
                    + params['c_deg'] * delta_m * Delta_tilde_par)
    V_kick = np.sqrt(Vm**2 + Vperp**2 + 2.0 * Vm * Vperp * np.cos(xi))
    return V_kick


def gwModel_kick_q200(q, chi1z, chi2z, return_std=False):
    """
    Kick velocity for aligned-spin binaries.
    From Islam & Wadekar (2025), https://arxiv.org/abs/2511.11536

    Trained on NR (SXS + RIT, q <= 32) and BHPT data (q <= 200).
    Valid for 1 <= q <= 1000.

    Parameters:
        q: Mass ratio m1/m2 >= 1
        chi1z: Dimensionless spin of primary along z, in [-1, 1]
        chi2z: Dimensionless spin of secondary along z, in [-1, 1]
        return_std: If True, also return parameter uncertainty estimate

    Returns:
        V_kick: Kick velocity in km/s
        V_kick_std (optional): Estimated uncertainty in km/s
    """
    q = validate_q(q)
    chi1z, chi2z = validate_spin_z(chi1z, chi2z)

    V_kick = _compute_kick(q, chi1z, chi2z, _PARAMS_MEDIAN)

    if return_std:
        variance = np.zeros_like(V_kick)
        for param_name in _PARAMS_MEDIAN:
            params_perturbed = _PARAMS_MEDIAN.copy()
            params_perturbed[param_name] = _PARAMS_MEDIAN[param_name] + _PARAMS_STD[param_name]
            delta_V = _compute_kick(q, chi1z, chi2z, params_perturbed) - V_kick
            variance += delta_V**2
        return V_kick, np.sqrt(variance)

    return V_kick
