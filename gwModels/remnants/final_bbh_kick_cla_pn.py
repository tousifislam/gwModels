#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: final_bbh_kick_cla_pn.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

def bbh_final_kick_aligned_spin_SRGB2019(a, alpha):
    """
    BBH kick velocity for aligned spins from Sperhake, Rosca-Mead, Gerosa, Berti (2019)
    https://arxiv.org/pdf/1910.01598
    
    Parameters:
    a: spin magnitude (same for both BHs)
    alpha: initial spin orientation (radians)
    
    Returns:
    vk_upper, vk_lower: kick velocity bounds in km/s
    """
    # Maximum kick amplitudes (Eq. 4)
    vmax_upper = (365 + 4183) * a  # (243+122) + (4020+163)
    vmax_lower = (121 + 3857) * a  # (243-122) + (4020-163)
    
    # Phase angle and kick calculation (Eq. 3)
    alpha_0 = np.radians(218.7)
    cos_phase = np.cos(alpha - alpha_0)

    vk_avg = abs(float(vmax_upper * cos_phase) + float(vmax_lower * cos_phase))/2
    
    return vk_avg

def bbh_final_kick_nonspinning_eccentric_SYL2006(q, et):
    """
    Non-spinning eccentric BBH kick velocity from Sopuerta, Yunes & Laguna (2006)
    https://arxiv.org/pdf/astro-ph/0611110
    
    Parameters:
    q: mass ratio
    et: eccentricity
    
    Returns:
    kick velocity in km/s
    """
    # Symmetric mass ratio
    eta = q / (1 + q)**2
    
    # Base recoil from close limit approximation
    v_kick = 5232 * np.sqrt(q - 4*eta) * eta**2
    
    # Mass ratio correction
    v_kick *= eta**2 * (1 - 2.621*eta + 3.199*eta**2)
    
    # Eccentricity correction  
    v_kick *= ((1 + et) / (1 - et)) * (1 - 0.942*et + 0.808*et**2 - 0.405*et**3)
    
    # PN correction (assumes r_min = 4GM/c²)
    v_kick *= (1 + 1.357 * (1 + 4.418*eta + 5.33*eta**2))
    
    return float(v_kick)
