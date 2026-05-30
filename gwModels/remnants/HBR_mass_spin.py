#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: HBR_final_mass_final_spin.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 05-30-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .remnant_utils import symmetric_mass_ratio, kerr_isco_radius, validate_q, validate_spin_magnitudes

def energy_at_isco(a):
    """
    Dimensionless specific energy at the ISCO: E_ISCO(a).
    """
    r_isco = kerr_isco_radius(a)
    return np.sqrt(1 - 2.0 / (3.0 * r_isco))

def angular_momentum_at_isco(a):
    """
    Dimensionless specific angular momentum at ISCO: L_ISCO(a).
    """
    r = kerr_isco_radius(a)
    return (2.0/(3.0*np.sqrt(3.0))) * (1.0 + 2.0*np.sqrt(3.0*r - 2.0))

def angle_between_spins(theta1, theta2, delta_phi):
    """
    Angle alpha between the two spin vectors using spherical law of cosines.
    """
    cos_alpha = (np.cos(theta1)*np.cos(theta2) +
                 np.sin(theta1)*np.sin(theta2)*np.cos(delta_phi))
    return np.arccos(np.clip(cos_alpha, -1.0, 1.0))

def angle_correction(theta, eps):
    """
    Angle remapping from Eq. (18): tan(theta'/2) = (1+eps) tan(theta/2).
    """
    return 2.0*np.arctan((1.0 + eps) * np.tan(0.5*theta))

def bbh_final_mass_precessing_BMR2012(q, a1, a2, theta1, theta2, verbose=False):
    """
    Final remnant mass using Barausse-Morozova-Rezzolla (2012) fit.

    Parameters:
        q: Mass ratio q = m1/m2 >= 1
        a1, a2: Dimensionless spin magnitudes (0 <= a <= 1)
        theta1, theta2: Angles (radians) between orbital momentum and spins
        verbose: Print intermediate calculations (default: False)

    Returns:
        Mfin: Remnant mass as fraction of initial total mass M_f/M

    Reference: Barausse, Morozova & Rezzolla (2012), ApJ 758, 63
    """
    q = validate_q(q)
    a1, a2 = validate_spin_magnitudes(a1, a2)
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)

    small_q = 1.0 / q
    eta = symmetric_mass_ratio(small_q)

    a_tilde = (a1*np.cos(theta1) + (small_q**2)*a2*np.cos(theta2)) / (1.0 + small_q)**2

    E_isco = energy_at_isco(a_tilde)

    p0 = 0.04827
    p1 = 0.01707

    term1 = eta * (1.0 - E_isco)
    term2 = 4.0 * (eta**2) * (4*p0 + 16*p1*a_tilde*(a_tilde + 1.0) + E_isco - 1.0)
    E_rad = term1 + term2

    M_fin = 1.0 - E_rad

    return M_fin

def bbh_final_spin_precessing_HBR2016(q, a1, a2, theta1, theta2, delta_phi,
                       model="HBR16_34corr", verbose=False):
    """
    Final spin magnitude using Hofmann, Barausse & Rezzolla (2016) fit.

    Parameters:
        q: Mass ratio q = m1/m2 >= 1
        a1, a2: Dimensionless spin magnitudes (0 <= a <= 1)
        theta1, theta2: Angles (radians) between orbital momentum and spins
        delta_phi: Angle between spin projections on orbital plane
        model: Fit model selection (default: "HBR16_34corr")
        verbose: Print intermediate calculations

    Returns:
        chi_final: Final spin magnitude |a_final| <= 1

    Reference: Hofmann, Barausse & Rezzolla (2016), ApJL 825, L19
    """
    q = validate_q(q)
    a1, a2 = validate_spin_magnitudes(a1, a2)
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    delta_phi = np.asarray(delta_phi, dtype=float)

    small_q = 1.0 / q
    eta = symmetric_mass_ratio(small_q)

    coefficient_sets = {
        "HBR16_12": {
            "k": np.array([
                [np.nan, -1.2019, -1.20764],
                [3.79245, 1.18385, 4.90494],
            ]),
            "xi": 0.41616,
        },
        "HBR16_33": {
            "k": np.array([
                [np.nan, 2.87025, -1.53315, -3.78893],
                [32.9127, -62.9901, 10.0068, 56.1926],
                [-136.832, 329.32, -13.2034, -252.27],
                [210.075, -545.35, -3.97509, 368.405],
            ]),
            "xi": 0.463926,
        },
        "HBR16_34": {
            "k": np.array([
                [np.nan, 3.39221, 4.48865, -5.77101, -13.0459],
                [35.1278, -72.9336, -86.0036, 93.7371, 200.975],
                [-146.822, 387.184, 447.009, -467.383, -884.339],
                [223.911, -648.502, -697.177, 753.738, 1166.89],
            ]),
            "xi": 0.474046,
        },
    }

    base_model = model.replace("corr", "")
    if base_model not in coefficient_sets:
        raise ValueError(f"Model must be one of: {list(coefficient_sets.keys())} (with optional 'corr')")

    xi = coefficient_sets[base_model]["xi"]
    k = coefficient_sets[base_model]["k"].copy()

    nu = 0.25
    target = 0.68646
    L_isco_zero = angular_momentum_at_isco(0.0)

    i_indices = np.arange(1, k.shape[0])
    correction_terms = np.sum(k[1:, 0] * (nu ** (2 + i_indices)))
    k[0, 0] = (target - nu * L_isco_zero - correction_terms) / (nu**2)

    use_corrections = "corr" in model
    eps1 = 0.024 if use_corrections else 0.0
    eps2 = 0.024 if use_corrections else 0.0
    eps12 = 0.0

    alpha = angle_between_spins(theta1, theta2, delta_phi)
    beta = theta1
    gamma = theta2

    alpha_corrected = angle_correction(alpha, eps12)
    beta_corrected = angle_correction(beta, eps1)
    gamma_corrected = angle_correction(gamma, eps2)

    a_tot = (a1*np.cos(beta_corrected) + (small_q**2)*a2*np.cos(gamma_corrected)) / (1.0 + small_q)**2
    a_eff = a_tot + xi*eta*(a1*np.cos(beta_corrected) + a2*np.cos(gamma_corrected))
    a_eff = np.clip(a_eff, -1.0, 1.0)

    E_isco = energy_at_isco(a_eff)
    L_isco = angular_momentum_at_isco(a_eff)

    n_M, n_J = k.shape[0] - 1, k.shape[1] - 1

    eta_powers = eta[..., None] ** (1 + np.arange(n_M + 1))
    a_eff_powers = a_eff[..., None] ** (np.arange(n_J + 1))

    correction_sum = np.sum(k * (eta_powers[..., :, None] * a_eff_powers[..., None, :]),
                           axis=(-2, -1))

    ell = np.abs(L_isco - 2.0*a_tot*(E_isco - 1.0) + correction_sum)

    term1 = a1**2 + (a2**2)*(small_q**4) + 2.0*a1*a2*(small_q**2)*np.cos(alpha_corrected)
    term2 = 2.0*(a1*np.cos(beta_corrected) + a2*(small_q**2)*np.cos(gamma_corrected)) * ell * small_q
    term3 = (ell*small_q)**2

    chi_final = np.sqrt(term1 + term2 + term3) / (1.0 + small_q)**2

    chi_final = np.minimum(chi_final, 1.0)

    return chi_final
