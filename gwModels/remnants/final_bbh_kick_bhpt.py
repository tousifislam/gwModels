#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: final_bbh_kick_bhpt.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

def bbh_final_kick_from_bhpt_SKH2010(q, a, unit='km/s'):
    """
    Black hole perturbation theory kick velocity from Sundararajan, Khanna, Hughes (2010)
    Eq. 5.7: https://arxiv.org/pdf/1003.0485
    
    Parameters:
    q: small mass ratio (m2/m1)
    a: spin of primary black hole
    unit: 'km/s' or 'c' (speed of light units)
    
    Returns:
    kick velocity in specified units
    """
    # Convert to arrays for vectorized operations
    q, a = np.asarray(q), np.asarray(a)
    
    # Equal mass mapping factor
    equal_mass_factor = q**2 * np.sqrt(1 - 4*q)
    
    # BHPT kick formula (Eq. 5.7)
    v_bhpt = (0.0440 - 0.0099*a - 0.0114*a**2 - 0.0312*a**3) * equal_mass_factor
    
    # Unit conversion
    if unit == 'km/s':
        v_bhpt *= 299792.458  # c in km/s
    
    # Return scalar if input was scalar
    return float(v_bhpt) if np.isscalar(q) and np.isscalar(a) else v_bhpt


def bbh_final_kick_range_from_bhpt_FHH2004(q, a_tilde_2):
    """
    Calculate upper and lower kick velocity estimates from Favata, Hughes & Holz (2004)
    Eqs 1 and 2 of https://arxiv.org/pdf/astro-ph/0402057
    
    Parameters:
    q: mass ratio (q = m1/m2 <= 1)
    a_tilde_2: dimensionless spin of larger BH (-1 <= a_tilde_2 <= 1)
    
    Returns:
    V_upper, V_lower: kick velocity bounds in km/s
    """
    
    # Fitchett scaling function f(q) and its maximum
    f_q = q**2 * (1 - q) / (1 + q)**5
    q_max = (2 + np.sqrt(2)) / 4  # Analytical maximum location
    f_max = q_max**2 * (1 - q_max) / (1 + q_max)**5  # Maximum value
    f_ratio = f_q / f_max
    
    # Convert physical spin to effective spin using Damour (2001) relation
    a_tilde = (1 + 3*q/4) * a_tilde_2 / (1 + q)**2
    
    # Polynomial coefficients for upper limit (Equation 1)
    V_upper = 465.0 * f_ratio * (1 - 0.281*a_tilde - 0.0361*a_tilde**2 
                                - 0.346*a_tilde**3 - 0.374*a_tilde**4 - 0.184*a_tilde**5)
    
    # Polynomial coefficients for lower limit (Equation 2)  
    V_lower = 54.4 * f_ratio * (1 + 1.22*a_tilde + 1.04*a_tilde**2 
                               + 0.977*a_tilde**3 - 0.201*a_tilde**4 - 0.434*a_tilde**5)
    
    # Spin-orbit correction factor for non-zero spins
    q_prime = 0.127  # Reference value from paper
    if abs(1 - q) > 1e-10:  # Avoid division by zero
        correction = abs(1 + (7/29) * a_tilde_2 / (1 - q)) / abs(1 + (7/29) * a_tilde_2 / (1 - q_prime))
        V_upper *= correction
        V_lower *= correction
    
    return float(V_upper), float(V_lower)