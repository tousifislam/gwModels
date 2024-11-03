#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: u_in_x_l_e.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools

def compute_u_in_x_e_l(x, e, l, q):
    """
    Compute the value of u based on the given parameters.
    Refers to Eq 7 of https://arxiv.org/pdf/0806.1037 - converts the equation to get u
    Converted expression is obtained from Eq A2 of https://arxiv.org/pdf/1605.00304
    
    Input:
    ------
    x : float
        Post-Newtonian parameter.
    e : float
        Eccentricity.
    l : float
        Mean anomaly.
    q : float
        Mass ratio.
        
    Output:
    -------
    u : float
        The computed value of u.
    """
    
    # Defined right after Eq. 3.20 in https://arxiv.org/pdf/1605.00304
    xi = x**(3/2)
    
    # symmetric mass ratio
    eta = gwtools.q_to_nu(q)
    
    # Calculate individual terms based on the given formula
    term1 = l

    term2 = (1 + 
             (-15/2 + 9/8 * eta + 1/8 * eta**2) * xi**(4/3) + 
             (-85 + 112153/1680 * eta + 5/4 * eta**2 + 1/24 * eta**3) * xi**2) * e * np.sin(l)

    term3 = (1 + 
             (-75/4 + 15/8 * eta + 3/8 * eta**2) * xi**(4/3) + 
             (-465/2 + (241819/1680 + 41/128 * np.pi**2) * eta + 
              59/4 * eta**2 - 3/8 * eta**3) * xi**2) * (1/2) * e**2 * np.sin(2 * l)

    term4 = (
        (3 * np.sin(3 * l) - np.sin(l) +
        (-(15 * np.sin(l) + 95 * np.sin(3 * l)) + 
         (1/8 * (93 * np.sin(l) + 49 * np.sin(3 * l)) * eta) + 
         (1/8 * (-3 * np.sin(l) + 17 * np.sin(3 * l)) * eta**2) ) * xi**(4/3) +
        (-(770 * np.sin(l) + 1250 * np.sin(3 * l)) + 
         ((310967/336 - 205/64 * np.pi**2) * np.sin(l) + 
          (3106493/5040 + 533/192 * np.pi**2) * np.sin(3 * l)) * eta +
         (-775/8 * np.sin(l) + 3143/24 * np.sin(3 * l)) * eta**2 +
         (1/2 * np.sin(l) - 29/6 * np.sin(3 * l)) * eta**3) * xi**2) * (1/8) * e**3
    )
    
    # Combine all terms to compute u
    u = term1 + term2 + term3 + term4

    return u