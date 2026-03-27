#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: final_bbh_mass_spin_hbr.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .remnant_utils import symmetric_mass_ratio, kerr_isco_radius

def energy_at_isco(a):
    """
    Calculate dimensionless specific energy at the ISCO: Ẽ_ISCO(a).
    
    This is the binding energy per unit mass at the innermost stable orbit.
    """
    r_isco = kerr_isco_radius(a)
    return np.sqrt(1 - 2.0 / (3.0 * r_isco))

def angular_momentum_at_isco(a):
    """
    Dimensionless specific angular momentum at ISCO: L̃_ISCO(a)
    """
    r = kerr_isco_radius(a)
    return (2.0/(3.0*np.sqrt(3.0))) * (1.0 + 2.0*np.sqrt(3.0*r - 2.0))

def angle_between_spins(theta1, theta2, deltaphi):
    """
    Calculate angle α between the two spin vectors.
    Uses spherical law of cosines on the unit sphere.
    """
    cos_alpha = (np.cos(theta1)*np.cos(theta2) + 
                 np.sin(theta1)*np.sin(theta2)*np.cos(deltaphi))
    return np.arccos(np.clip(cos_alpha, -1.0, 1.0))

def angle_correction(theta, eps):
    """
    Angle remapping from Eq. (18): tan(θ'/2) = (1+ε) tan(θ/2).
    This corrects for systematic errors in the fit.
    """
    return 2.0*np.arctan((1.0 + eps) * np.tan(0.5*theta))

def bbh_final_mass_precessing_BMR2012(theta1, theta2, q, chi1, chi2, verbose=False):
    """
    Calculate final remnant mass using Barausse-Morozova-Rezzolla (2012) fit.
    
    This estimates the mass of the final black hole after merger,
    accounting for energy radiated away as gravitational waves.
    
    Parameters:
    - theta1, theta2: Angles (radians) between orbital momentum and spins
    - q: Mass ratio m2/m1 with 0 ≤ q ≤ 1 (m1 ≥ m2)
    - chi1, chi2: Dimensionless spin magnitudes (0 ≤ chi ≤ 1)
    - verbose: Print intermediate calculations (default: False)
    
    Returns:
    - Mfin: Remnant mass as fraction of initial total mass M_f/M
    
    Reference: Barausse, Morozova & Rezzolla (2012), ApJ 758, 63
    """
    # Convert everything to arrays for vectorization
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    q = np.asarray(q, dtype=float)
    chi1 = np.asarray(chi1, dtype=float)
    chi2 = np.asarray(chi2, dtype=float)
    
    # Step 1: Calculate symmetric mass ratio η
    eta = symmetric_mass_ratio(q)
    
    # Step 2: Calculate projected total spin a_tilde (Eq. 17)
    # This represents the effective spin of the final BH
    a_tilde = (chi1*np.cos(theta1) + (q**2)*chi2*np.cos(theta2)) / (1.0 + q)**2
    
    # Step 3: Energy at ISCO for the final spin
    E_isco = energy_at_isco(a_tilde)
    
    # Step 4: BMR2012 fit coefficients (Eq. 12)
    p0 = 0.04827
    p1 = 0.01707
    
    # Step 5: Calculate radiated energy fraction (Eq. 18)
    # This is the fraction of mass-energy radiated away as GWs
    term1 = eta * (1.0 - E_isco)
    term2 = 4.0 * (eta**2) * (4*p0 + 16*p1*a_tilde*(a_tilde + 1.0) + E_isco - 1.0)
    E_rad = term1 + term2
    
    # Step 6: Final mass fraction (what's left after radiation)
    M_fin = 1.0 - E_rad
    
    return M_fin

def bbh_final_spin_precessing_HBR2016(theta1, theta2, deltaphi, q, chi1, chi2, 
                       model="HBR16_34corr", verbose=False):
    """
    Calculate final spin magnitude using Hofmann, Barausse & Rezzolla (2016) fit.
    
    This estimates the spin of the final black hole after merger,
    valid near merger (apply after evolving to r~10M).
    
    Parameters:
    - theta1, theta2: Angles (radians) between orbital momentum and spins
    - deltaphi: Angle between spin projections on orbital plane
    - q: Mass ratio m2/m1 with 0 ≤ q ≤ 1 (m1 ≥ m2)
    - chi1, chi2: Dimensionless spin magnitudes (0 ≤ chi ≤ 1)
    - model: Fit model selection (default: "HBR16_34corr")
             Options: "HBR16_12", "HBR16_12corr", "HBR16_33", "HBR16_33corr",
                     "HBR16_34", "HBR16_34corr"
    - verbose: Print intermediate calculations
    
    Returns:
    - chi_final: Final spin magnitude |a_final| ≤ 1
    
    Reference: Hofmann, Barausse & Rezzolla (2016), ApJL 825, L19
    """
    # Convert to arrays for vectorization
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    deltaphi = np.asarray(deltaphi, dtype=float)
    q = np.asarray(q, dtype=float)
    chi1 = np.asarray(chi1, dtype=float)
    chi2 = np.asarray(chi2, dtype=float)
    
    # Calculate symmetric mass ratio
    eta = symmetric_mass_ratio(q)
    
    # Coefficient sets from Table 1
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
    
    # Select coefficient set
    base_model = model.replace("corr", "")
    if base_model not in coefficient_sets:
        raise ValueError(f"Model must be one of: {list(coefficient_sets.keys())} (with optional 'corr')")
    
    xi = coefficient_sets[base_model]["xi"]
    k = coefficient_sets[base_model]["k"].copy()
    
    # Set k00 for equal mass case (Eq. 11): a_fin(q=1, a1=a2=0) = 0.68646
    nu = 0.25  # eta for equal masses
    target = 0.68646
    L_isco_zero = angular_momentum_at_isco(0.0)  # sqrt(3)/2
    
    i_indices = np.arange(1, k.shape[0])
    correction_terms = np.sum(k[1:, 0] * (nu ** (2 + i_indices)))
    k[0, 0] = (target - nu * L_isco_zero - correction_terms) / (nu**2)
    
    # Apply angle corrections if requested (Eq. 18)
    use_corrections = "corr" in model
    eps1 = 0.024 if use_corrections else 0.0   # ε_β correction
    eps2 = 0.024 if use_corrections else 0.0   # ε_γ correction
    eps12 = 0.0  # ε_α correction (always 0 in the paper)
    
    # Calculate angles
    alpha = angle_between_spins(theta1, theta2, deltaphi)
    beta = theta1
    gamma = theta2
    
    # Apply corrections
    alpha_corrected = angle_correction(alpha, eps12)
    beta_corrected = angle_correction(beta, eps1)
    gamma_corrected = angle_correction(gamma, eps2)
    
    # Calculate effective spins (Eqs. 14-15)
    a_tot = (chi1*np.cos(beta_corrected) + (q**2)*chi2*np.cos(gamma_corrected)) / (1.0 + q)**2
    a_eff = a_tot + xi*eta*(chi1*np.cos(beta_corrected) + chi2*np.cos(gamma_corrected))
    a_eff = np.clip(a_eff, -1.0, 1.0)  # Keep in valid Kerr range
    
    # ISCO quantities at effective spin
    E_isco = energy_at_isco(a_eff)
    L_isco = angular_momentum_at_isco(a_eff)
    
    # Calculate correction term from double sum (Eq. 13)
    n_M, n_J = k.shape[0] - 1, k.shape[1] - 1
    
    # Create power arrays for broadcasting
    eta_powers = eta[..., None] ** (1 + np.arange(n_M + 1))
    a_eff_powers = a_eff[..., None] ** (np.arange(n_J + 1))
    
    # Compute double sum: Σ k_ij η^(1+i) a_eff^j
    correction_sum = np.sum(k * (eta_powers[..., :, None] * a_eff_powers[..., None, :]), 
                           axis=(-2, -1))
    
    # Calculate |ℓ| (Eq. 13)
    ell = np.abs(L_isco - 2.0*a_tot*(E_isco - 1.0) + correction_sum)
    
    # Final spin calculation (Eq. 16)
    term1 = chi1**2 + (chi2**2)*(q**4) + 2.0*chi1*chi2*(q**2)*np.cos(alpha_corrected)
    term2 = 2.0*(chi1*np.cos(beta_corrected) + chi2*(q**2)*np.cos(gamma_corrected)) * ell * q
    term3 = (ell*q)**2
    
    chi_final = np.sqrt(term1 + term2 + term3) / (1.0 + q)**2
    
    # Physical constraint: spin can't exceed 1
    chi_final = np.minimum(chi_final, 1.0)
    
    return chi_final