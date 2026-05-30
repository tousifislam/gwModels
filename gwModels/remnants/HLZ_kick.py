#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: HLZ_final_kick.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 05-30-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .remnant_utils import validate_q, validate_spin_magnitudes, validate_spin_z

# Fitting coefficients from the literature
# Gonzalez et al. (2007b),
# Campanelli et al. (2007b),
# Lousto & Zlochower (2008, 2013),
# Lousto et al. (2012a,b)

# Fitting parameters
A = 1.2e4      # km/s
B = -0.93
H = 6.9e3      # km/s
V11 = 3677.76  # km/s
VA = 2481.21   # km/s
VB = 1792.45   # km/s
VC = 1506.52   # km/s
C2 = 1140.0    # km/s
C3 = 2481.0    # km/s
XI_ANGLE = np.radians(145.0)  # angle between kick components (radians)

def spin_difference_vector(small_q, a1, a2, theta1, theta2, delta_phi):
    """
    Calculate the antisymmetric spin combination Delta = (S2/m2 - S1/m1)/(1+q).

    Parameters:
        small_q: mass ratio m2/m1 <= 1 (internal convention)
        a1, a2: dimensionless spin magnitudes
        theta1, theta2: angles between L and spin vectors (radians)
        delta_phi: azimuthal angle difference between spins (radians)

    Returns:
        Delta_parallel, Delta_perp
    """
    Delta_parallel = abs((a1 * np.cos(theta1) - small_q * a2 * np.cos(theta2)) / (1 + small_q))

    Delta_perp_sq = (a1**2 * np.sin(theta1)**2 + small_q**2 * a2**2 * np.sin(theta2)**2
                    - 2 * small_q * a1 * a2 * np.sin(theta1) * np.sin(theta2) * np.cos(delta_phi)) / (1 + small_q)**2
    Delta_perp = np.sqrt(np.maximum(0, Delta_perp_sq))

    return Delta_parallel, Delta_perp

def total_spin_vector(small_q, a1, a2, theta1, theta2, delta_phi):
    """
    Calculate the symmetric spin combination chi = (S1 + S2)/M^2.

    Parameters:
        small_q: mass ratio m2/m1 <= 1 (internal convention)
        a1, a2: dimensionless spin magnitudes
        theta1, theta2: angles between L and spin vectors (radians)
        delta_phi: azimuthal angle difference between spins (radians)

    Returns:
        chi_tilde_parallel, chi_tilde_perp
    """
    chi_tilde_parallel = (a1 * np.cos(theta1) + small_q**2 * a2 * np.cos(theta2)) / (1 + small_q)**2

    chi_tilde_perp_sq = (a1**2 * np.sin(theta1)**2 + small_q**4 * a2**2 * np.sin(theta2)**2
                        + 2 * small_q**2 * a1 * a2 * np.sin(theta1) * np.sin(theta2) * np.cos(delta_phi)) / (1 + small_q)**4
    chi_tilde_perp = np.sqrt(np.maximum(0, chi_tilde_perp_sq))

    return chi_tilde_parallel, chi_tilde_perp

def calculate_kick_components(small_q, a1, a2, theta1, theta2, delta_phi, Theta=None):
    """
    Calculate the three kick velocity components.

    Parameters:
        small_q: mass ratio m2/m1 <= 1 (internal convention)
        a1, a2: dimensionless spin magnitudes
        theta1, theta2: angles between L and spin vectors (radians)
        delta_phi: azimuthal angle difference between spins (radians)
        Theta: angle between Delta x L and fiducial infall direction (radians).
               If None, a random value between 0 and 2pi is used.

    Returns:
        Vm, Vs_perp, Vs_parallel, Theta_used
    """
    # following https://arxiv.org/pdf/1605.01067
    if Theta is None:
        Theta = np.random.uniform(0, 2*np.pi)

    eta = small_q / (1 + small_q)**2

    Delta_parallel, Delta_perp = spin_difference_vector(small_q, a1, a2, theta1, theta2, delta_phi)
    chi_tilde_parallel, chi_tilde_perp = total_spin_vector(small_q, a1, a2, theta1, theta2, delta_phi)

    Vm = A * eta**2 * (1 - small_q) / (1 + small_q) * (1 + B * eta)

    Vs_perp = H * eta**2 * Delta_parallel

    term1 = Delta_perp * (V11 + 2*VA*chi_tilde_parallel +
                         4*VB*chi_tilde_parallel**2 + 8*VC*chi_tilde_parallel**3)
    term2 = 2 * chi_tilde_perp * Delta_parallel * (C2 + 2*C3*chi_tilde_parallel)
    Vs_parallel = 16 * eta**2 * (term1 + term2) * np.cos(Theta)

    return Vm, Vs_perp, Vs_parallel, Theta

def bbh_final_kick_precessing_CLZM2007(q, a1, a2, theta1, theta2, delta_phi, Theta=None,
                                       debug=False):
    """
    Calculate total kick velocity magnitude.

    Parameters:
        q: mass ratio q = m1/m2 >= 1
        a1, a2: dimensionless spin magnitudes
        theta1, theta2: angles between L and spin vectors (radians)
        delta_phi: azimuthal angle difference between spins (radians)
        Theta: angle between Delta x L and fiducial infall direction (radians).
               If None, a random value between 0 and 2pi is used.
        debug: if True, return all components

    Returns:
        V_kick (float): total kick velocity in km/s
        If debug=True: V_kick, Vm, Vs_perp, Vs_parallel, Theta_used
    """
    validate_q(q)
    validate_spin_magnitudes(a1, a2)

    small_q = 1.0 / q

    Vm, Vs_perp, Vs_parallel, used_Theta = calculate_kick_components(
        small_q, a1, a2, theta1, theta2, delta_phi, Theta)

    V_kick = np.sqrt(Vm**2 + 2*Vm*Vs_perp*np.cos(XI_ANGLE) + Vs_perp**2 + Vs_parallel**2)

    if debug:
        return V_kick, Vm, Vs_perp, Vs_parallel, used_Theta
    else:
        return V_kick

def bbh_final_kick_nonprecessing_HLZ2014(q, chi1z, chi2z):
    """
    RIT aligned-spin recoil (kick) for binaries with spins along +/-z.
    Coefficients from arXiv:1406.7295.

    Parameters:
        q: mass ratio q = m1/m2 >= 1
        chi1z: dimensionless spin of primary along z in [-1, 1]
        chi2z: dimensionless spin of secondary along z in [-1, 1]

    Returns:
        V_kick: kick velocity in km/s
    """
    q = validate_q(q)
    chi1z, chi2z = validate_spin_z(chi1z, chi2z)

    _A = 1.2e4
    _B = -0.93
    _H = 6.9e3
    H2a = 0.185
    H2b = -0.0105
    H3a = -0.283
    H3b = -0.0661
    H3c = 0.153
    H3d = 0.0306
    H3e = 0.00006
    H4a = 0.035
    H4b = -0.030
    H4c = 0.050
    H4d = -0.0013
    H4e = -0.011
    H4f = 0.0057
    a_deg = 145.0
    b_deg = -7.2
    c_deg = 54.0

    eta = q / (1.0 + q)**2
    delta_m = (q - 1.0) / (q + 1.0)

    S_tilde_par = (chi1z + q**2 * chi2z) / (1.0 + q)**2
    Delta_tilde_par = (chi1z - q * chi2z) / (1.0 + q)

    poly = (
        Delta_tilde_par
        + H2a * S_tilde_par * delta_m
        + H2b * Delta_tilde_par * S_tilde_par
        + H3a * (Delta_tilde_par**2) * delta_m
        + H3b * (S_tilde_par**2) * delta_m
        + H3c * Delta_tilde_par * (S_tilde_par**2)
        + H3d * (Delta_tilde_par**3)
        + H3e * Delta_tilde_par * (delta_m**2)
        + H4a * S_tilde_par * (Delta_tilde_par**2) * delta_m
        + H4b * (S_tilde_par**3) * delta_m
        + H4c * S_tilde_par * (delta_m**3)
        + H4d * Delta_tilde_par * S_tilde_par * (delta_m**2)
        + H4e * Delta_tilde_par * (S_tilde_par**3)
        + H4f * S_tilde_par * (Delta_tilde_par**3)
    )
    Vperp = _H * (eta**2) * poly

    Vm = _A * (eta**2) * delta_m * (1.0 + _B * eta)

    xi = np.deg2rad(a_deg + b_deg * S_tilde_par + c_deg * delta_m * Delta_tilde_par)

    V_kick = np.sqrt(Vm**2 + Vperp**2 + 2.0 * Vm * Vperp * np.cos(xi))

    return V_kick
