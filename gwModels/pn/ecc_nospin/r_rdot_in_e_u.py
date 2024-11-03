#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: r_rdot_in_e_u.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools

def r_0_PN_in_e_u(e, u):
    """
    Calculate r_0_PN.
    Eq. A6 of https://arxiv.org/pdf/0806.1037

    Parameters:
    e (float): Eccentricity.
    u (float): True anomaly.

    Returns:
    float: Value of r_0_PN.
    """
    return 1 - e * np.cos(u)

def r_1_PN_in_e_u(e, u, eta):
    """
    Calculate r_1_PN.
    Eq. A7 of https://arxiv.org/pdf/0806.1037

    Parameters:
    e (float): Eccentricity.
    u (float): True anomaly.
    eta (float): Symmetric mass ratio.

    Returns:
    float: Value of r_1_PN.
    """
    term1 = (2 * (e * np.cos(u) - 1)) / (e**2 - 1)
    term2 = (1 / 6) * (2 * (eta - 9) + e * (7 * eta - 6) * np.cos(u))
    return term1 + term2

def r_2_PN_in_e_u(e, u, eta):
    """
    Calculate r_2_PN.
    Eq. A8 of https://arxiv.org/pdf/0806.1037

    Parameters:
    e (float): Eccentricity.
    u (float): True anomaly.
    eta (float): Symmetric mass ratio.

    Returns:
    float: Value of r_2_PN.
    """
    common_factor = 1 / ((1 - e**2)**2)
    term1 = (1 / 72) * (8 * eta**2 + 30 * eta + 72) * e**4
    term2 = (1 / 72) * (-16 * eta**2 - 876 * eta + 756) * e**2
    term3 = (1 / 72) * (8 * eta**2 + 198 * eta + 360)

    term4 = (1 / 72) * (-35 * eta**2 + 231 * eta - 72) * e**5
    term5 = (1 / 72) * (70 * eta**2 - 150 * eta - 468) * e**3
    term6 = (1 / 72) * (-35 * eta**2 + 567 * eta - 648) * e

    term7 = (1 / 72) * (360 - 144 * eta) * e**2
    term8 = (1 / 72) * (144 * eta - 360)
    term9 = (1 / 72) * (180 - 72 * eta) * e**3
    term10 = (1 / 72) * (72 * eta - 180) * e

    return common_factor * (term1 + term2 + term3 + term4 + term5 + term6 * np.cos(u) + 
                             np.sqrt(1 - e**2) * (term7 + term8 + term9 + term10 * np.cos(u)))

def r_3_PN_in_e_u(e, u, eta):
    """
    Calculate r_3_PN.
    Eq. A9 of https://arxiv.org/pdf/0806.1037

    Parameters:
    e (float): Eccentricity.
    u (float): True anomaly.
    eta (float): Symmetric mass ratio.

    Returns:
    float: Value of r_3_PN.
    """
    common_factor = 1 / (181440 * (1 - e**2)**(7/2))

    term1 = (-665280 * eta**2 + 1753920 * eta - 1814400) * e**6
    term2 = (725760 * eta**2 - 77490 * np.pi**2 * eta + 5523840 * eta - 3628800) * e**4
    term3 = (544320 * eta**2 + 154980 * np.pi**2 * eta - 14132160 * eta + 7257600) * e**2
    term4 = -604800 * eta**2 + 6854400 * eta

    term5 = (302400 * eta**2 - 1254960 * eta + 453600) * e**7
    term6 = (-1542240 * eta**2 - 38745 * np.pi**2 * eta + 6980400 * eta - 453600) * e**5
    term7 = (2177280 * eta**2 + 77490 * np.pi**2 * eta - 12373200 * eta + 4989600) * e**3
    term8 = (-937440 * eta**2 - 38745 * np.pi**2 * eta + 6647760 * eta - 4989600) * e

    term9 = np.sqrt(1 - e**2) * (
        (-4480 * eta**3 - 25200 * eta**2 + 22680 * eta - 120960) * e**6 +
        (13440 * eta**3 + 4404960 * eta**2 + 116235 * np.pi**2 * eta - 12718296 * eta + 5261760) * e**4 +
        (-13440 * eta**3 + 2242800 * eta**2 + 348705 * np.pi**2 * eta - 19225080 * eta + 16148160) * e**2 +
        (4480 * eta**3 + 45360 * eta**2 - 8600904 * eta) +
        ((-6860 * eta**3 + 550620 * eta**2 - 986580 * eta + 120960) * e**7 +
         (20580 * eta**3 - 2458260 * eta**2 + 3458700 * eta - 2358720) * e**5 +
         (-20580 * eta**3 - 3539340 * eta**2 - 116235 * np.pi**2 * eta + 20173860 * eta - 16148160) * e**3 +
         (6860 * eta**3 - 1220940 * eta**2 - 464940 * np.pi**2 * eta + 17875620 * eta - 4717440) * e))

    term10 = 116235 * eta * np.pi**2 + 1814400
    term11 = -77490 * eta * np.pi**2 - 1814400

    return common_factor * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 * np.cos(u) + term9 + term10 + term11)

def compute_r_in_x_e_u(x, e, u, q):
    """
    Calculate radial separation r based on PN expansions.
    Eq. 5 and A5 of https://arxiv.org/pdf/0806.1037
    
    Parameters:
    e (float): Eccentricity.
    u (float): True anomaly.
    q (float): Mass ratio.

    Returns:
    float: Value of r/M.
    """
        
    eta = gwtools.q_to_nu(q)
    r = r_0_PN_in_e_u(e, u) * x**(-1) + r_1_PN_in_e_u(e, u, eta) + r_2_PN_in_e_u(e, u, eta) * x + r_3_PN_in_e_u(e, u, eta) * x**2
        
    return r

def compute_rdot_from_r(t, r):
    """
    Compute the time derivative of r using finite differences.

    Parameters:
    t (np.ndarray): Array of time values.
    r (np.ndarray): Array of radial distance values.

    Returns:
    np.ndarray: Array of radial velocity values (dr/dt).
    """
    return np.gradient(r, t)
