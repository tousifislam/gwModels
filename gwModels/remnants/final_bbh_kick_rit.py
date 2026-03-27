#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: final_bbh_kick_rit.py
#
#    Project: gwGenealogy
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

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

def spin_difference_vector(theta1, theta2, deltaphi, q, chi1, chi2):
    """
    Calculate the antisymmetric spin combination Δ = (S2/m2 - S1/m1)/(1+q).
    
    This represents the difference in specific angular momenta and drives
    kick asymmetries in the merger.

    Parameters:
    q: mass ratio (q <= 1)
    chi1, chi2: spin magnitudes 
    theta1, theta2: angles between L and spin vectors (degrees)
    deltaphi: azimuthal angle difference between spins (degrees)
    
    Returns:
    Delta_parallel, Delta_perp
    """
    # Delta parallel component
    Delta_parallel = abs((chi1 * np.cos(theta1) - q * chi2 * np.cos(theta2)) / (1 + q))
    
    # Delta perpendicular component
    Delta_perp_sq = (chi1**2 * np.sin(theta1)**2 + q**2 * chi2**2 * np.sin(theta2)**2 
                    - 2 * q * chi1 * chi2 * np.sin(theta1) * np.sin(theta2) * np.cos(deltaphi)) / (1 + q)**2
    Delta_perp = np.sqrt(np.maximum(0, Delta_perp_sq))
       
    return Delta_parallel, Delta_perp

def total_spin_vector(theta1, theta2, deltaphi, q, chi1, chi2):
    """
    Calculate the symmetric spin combination χ = (S1 + S2)/M^2.
    
    This represents the total angular momentum and affects hang-up kicks.

    Parameters:
    q: mass ratio (q <= 1)
    chi1, chi2: spin magnitudes 
    theta1, theta2: angles between L and spin vectors (degrees)
    deltaphi: azimuthal angle difference between spins (degrees)
    
    Returns:
    chi_tilde_parallel, chi_tilde_perp
    """
    # Chi tilde parallel component
    chi_tilde_parallel = (chi1 * np.cos(theta1) + q**2 * chi2 * np.cos(theta2)) / (1 + q)**2
    
    # Chi tilde perpendicular component
    chi_tilde_perp_sq = (chi1**2 * np.sin(theta1)**2 + q**4 * chi2**2 * np.sin(theta2)**2 
                        + 2 * q**2 * chi1 * chi2 * np.sin(theta1) * np.sin(theta2) * np.cos(deltaphi)) / (1 + q)**4
    chi_tilde_perp = np.sqrt(np.maximum(0, chi_tilde_perp_sq))
    
    return chi_tilde_parallel, chi_tilde_perp

def calculate_kick_components(q, chi1, chi2, theta1, theta2, deltaphi, Theta=None):
    """
    Calculate the three kick velocity components
    
    Parameters:
    q: mass ratio (q <= 1)
    chi1, chi2: spin magnitudes 
    theta1, theta2: angles between L and spin vectors (degrees)
    deltaphi: azimuthal angle difference between spins (degrees)
    Theta: angle between Delta x L and fiducial infall direction (radians)
           If None, a random value between 0 and 2π is used
    
    Returns:
    Vm, Vs_perp, Vs_parallel, Theta_used
    """

    # Angle Theta : taken to be random value between 0 and 2pi
    # following https://arxiv.org/pdf/1605.01067
    if Theta is None:
        # Note that, while we sample to randomly, Gerosa1 & Kesden 2016
        # fixes the value to zero;
        Theta = np.random.uniform(0, 2*np.pi)
    
    # Symmetric mass ratio
    eta = q / (1 + q)**2
    
    # Calculate the antisymmetric spin combination
    Delta_parallel, Delta_perp = spin_difference_vector(theta1, theta2, deltaphi, q, chi1, chi2)
    # Calculate the symmetric spin combination
    chi_tilde_parallel, chi_tilde_perp = total_spin_vector(theta1, theta2, deltaphi, q, chi1, chi2)
    
    # Mass asymmetry contribution
    Vm = A * eta**2 * (1 - q) / (1 + q) * (1 + B * eta)
    
    # Spin-orbit contribution (perpendicular)
    Vs_perp = H * eta**2 * Delta_parallel
    
    # Spin-spin contribution (parallel)
    term1 = Delta_perp * (V11 + 2*VA*chi_tilde_parallel + 
                         4*VB*chi_tilde_parallel**2 + 8*VC*chi_tilde_parallel**3)
    term2 = 2 * chi_tilde_perp * Delta_parallel * (C2 + 2*C3*chi_tilde_parallel)
    Vs_parallel = 16 * eta**2 * (term1 + term2) * np.cos(Theta)
    
    return Vm, Vs_perp, Vs_parallel, Theta

def bbh_final_kick_precessing_CLZM2007(q, chi1, chi2, theta1, theta2, deltaphi, Theta=None, 
                                       debug=False):
    """
    Calculate total kick velocity magnitude
    
    Parameters:
    q: mass ratio (q <= 1)
    chi1, chi2: spin magnitudes 
    theta1, theta2: angles between L and spin vectors (degrees)
    deltaphi: azimuthal angle difference between spins (degrees)
    Theta: angle between Delta x L and fiducial infall direction (radians)
           If None, a random value between 0 and 2π is used
    
    Returns:
    V_kick, Vm, Vs_perp, Vs_parallel, Theta_used (if debug is set to True)
    V_kick (if debug is set to False)
    """
    Vm, Vs_perp, Vs_parallel, used_Theta = calculate_kick_components(
        q, chi1, chi2, theta1, theta2, deltaphi, Theta)
    
    # Total kick velocity (Equation A1)
    V_kick = np.sqrt(Vm**2 + 2*Vm*Vs_perp*np.cos(XI_ANGLE) + Vs_perp**2 + Vs_parallel**2)
    
    if debug:
        return V_kick, Vm, Vs_perp, Vs_parallel, used_Theta
    else:
        return V_kick