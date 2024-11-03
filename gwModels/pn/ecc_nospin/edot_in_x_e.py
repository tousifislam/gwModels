#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: edot_in_x_e.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.interpolate import splev, splrep
from ...utils import mass_ratio_to_symmetric_mass_ratio
from .approximated_functions import *

# Data for special functions
# Table IV of https://arxiv.org/pdf/0908.3854
data = {
    'e': [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    'varphi_e': [1, 1.013, 1.056, 1.132, 1.248, 1.417, 1.658, 2.002,
                 2.502, 3.243, 4.382, 6.210, 9.310, 14.95, 26.20,
                 51.66, 120.3, 364.4, 1773],
    'psi_e': [1, 0.9845, 0.9338, 0.8352, 0.6629, 0.3708, -0.1233, -0.9688,
              -2.444, -5.092, -10.01, -19.63, -39.58, -84.39, -196.2,
              -519.4, -1670, -7363, -58039],
    'znu_e': [1, 1.024, 1.099, 1.237, 1.456, 1.790, 2.297, 3.076,
               4.304, 6.307, 9.726, 15.89, 27.81, 52.93, 112.1,
               274.7, 827.6, 3451, 25971],
    'kappa_e': [1, 1.027, 1.114, 1.273, 1.532, 1.938, 2.573, 3.590,
                5.266, 8.152, 13.38, 23.50, 44.67, 93.54, 221.6,
                620.2, 2201, 11332, 1.14e5]
}

# Build splines for the special functions
spl_psi_e = splrep(data['e'], data['psi_e'])
spl_varphi_e = splrep(data['e'], data['varphi_e'])
spl_znu_e = splrep(data['e'], data['znu_e'])
spl_kappa_e = splrep(data['e'], data['kappa_e'])


def F_e(e):
    """
    Calculate F_e(e) based on the given formula.
    
    Parameters:
        e (float): The eccentricity value (must be in the range [0, 1)).
    
    Returns:
        float: The computed value of F_e(e).
    """
    if not (0 <= e < 1):
        raise ValueError("e must be in the range [0, 1).")
    
    numerator = (1 + (2782 / 769) * e**2 + (10721 / 6152) * e**4 + (1719 / 24608) * e**6)
    denominator = (1 - e**2)**(11 / 2)
    
    return numerator / denominator


def e_dot_0PN_in_x_e(x, e, nu):
    """
    Compute the time derivative of e at 0PN
    Eqs. A30 of https://arxiv.org/pdf/0806.1037
    
    Input:
    ------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Output:
    -------
    e_dot : float
        0PN term in the time derivative of e
    """

    # 0PN term
    e_dot_0PN = -(e * (121 * e**2 + 304) * nu) / (15 * (1 - e**2)**(5/2))

    # Summing up the terms with respective powers of x
    e_dot = e_dot_0PN * x**4 
    
    return e_dot

def e_dot_1PN_in_x_e(x, e, nu):
    """
    Compute the time derivative of e at 1PN
    Eqs. A30 of https://arxiv.org/pdf/0806.1037
    
    Input:
    ------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Output:
    -------
    e_dot : float
        1PN term in the time derivative of e
    """

    # 1PN term
    e_dot_1PN = (e * nu) / (2520 * (1 - e**2)**(7/2)) * (
        (93184 * nu - 125361) * e**4 
        + 12 * (54271 * nu - 59834) * e**2 
        + 8 * (28588 * nu + 8451)
    )

    # Summing up the terms with respective powers of x
    e_dot = e_dot_1PN * x**5 
    
    return e_dot

    
def e_dot_1_5PN_HT_in_x_e(x, e, nu):
    """
    Compute the time derivative of e at 1.5PN
    It has contributions from hereditary terms
    Eqs. A30 of https://arxiv.org/pdf/0806.1037
    
    Input:
    ------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Output:
    -------
    e_dot : float
        1.5PN term in the time derivative of e
    """

    # 1.5PN term
    kappa_E = compute_aparox_kappa_E(e)
    kappa_J = compute_aparox_kappa_J(e)
    e_dot_1_5PN = (128 * nu * np.pi) / (5 * e) * ((e**2 - 1) * kappa_E + np.sqrt(1 - e**2) * kappa_J)

    # Summing up the terms with respective powers of x
    e_dot = e_dot_1_5PN * x**(11/2)
    
    return e_dot

    
def e_dot_2PN_in_x_e(x, e, nu):
    """
    Compute the time derivative of e at 2PN
    Eqs. A30 of https://arxiv.org/pdf/0806.1037
    
    Input:
    ------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Output:
    -------
    e_dot : float
        2PN term in the time derivative of e
    """

    # 2PN term
    e_dot_2PN = -(e * nu) / (30240 * (1 - e**2)**(9/2)) * (
                                  (2758560 * nu**2 - 4344852 * nu + 3786543) * e**6 
                                + (42810096 * nu**2 - 78112266 * nu + 46579718) * e**4 
                                + (48711348 * nu**2 - 35583228 * nu - 36993396) * e**2 
                                + 4548096 * nu**2 
                                + np.sqrt(1 - e**2) * (
                                                        (2847600 - 1139040 * nu) * e**4 
                                                        + (35093520 - 14037408 * nu) * e**2 
                                                        - 5386752 * nu + 13466880
                                                        ) 
                                + 13509360 * nu - 15198032
                                )

    # Summing up the terms with respective powers of x
    e_dot = e_dot_2PN * x**6
    
    return e_dot

    
def e_dot_3PN_in_x_e(x, e, nu, x_0_reference=1):
    """
    Compute the time derivative of e at 3PN
    Eqs. A39 of https://arxiv.org/pdf/2409.17636
    
    Input:
    ------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Output:
    -------
    e_dot : float
        3PN term in the time derivative of e
    """

    prefactor = nu * e * x**7
    
    # Precompute constants
    sqrt_term = np.sqrt(1 - e**2)
    term1 = 54177075619 / 6237000
    term2 = (7198067 / 22680) + (1283 / 10) * np.pi**2
    term3 = -3000281 / 2520
    term4 = -61001 / 486
    
    # Calculate polynomial contributions
    e2 = e**2
    e4 = e**4
    e6 = e**6
    e8 = e**8
    
    term_e2 = e2 * (6346360709 / 891000 + 
                         ((9569213 / 360) + (54001 / 960) * np.pi**2) * nu + 
                         (12478601 / 15120) * nu**2 - 
                         (86910509 / 19440) * nu**3)
    
    term_e4 = e4 * (-126288160777 / 16632000 + 
                         ((418129451 / 181440) - (254903 / 1920) * np.pi**2) * nu + 
                         (478808759 / 20160) * nu**2 - 
                         (2223241 / 180) * nu**3)
    
    term_e6 = e6 * (5845342193 / 1232000 + 
                         (-98425673 / 10080 - 6519 / 640 * np.pi**2) * nu + 
                         (6538757 / 630) * nu**2 - 
                         (11792069 / 2430) * nu**3)
    
    term_e8 = e8 * (302322169 / 1774080 - 
                         (1921387 / 10080) * nu + 
                         (41179 / 216) * nu**2 - 
                         (193396 / 1215) * nu**3)
    
    # Calculate the combined contribution of the sqrt term
    result = (1 - e**2)**(-11 / 2) * (
        term1 + 
        term2 * nu + 
        term3 * nu**2 + 
        term4 * nu**3 + 
        term_e2 + 
        term_e4 + 
        term_e6 + 
        term_e8 + 
        sqrt_term * (
            -22713049 / 15750 + 
            (-5526991 / 945 + 8323 / 180 * np.pi**2) * nu + 
            (54332 / 45) * nu**2 + 
            e2 * (89395687 / 7875 + 
                     (-38295557 / 1260 + 94177 / 960 * np.pi**2) * nu + 
                     (681989 / 90) * nu**2) + 
            e4 * (531445613 / 378000 + 
                     (-26478311 / 1512 + 2501 / 2880 * np.pi**2) * nu + 
                     (225106 / 45) * nu**2) + 
            e6 * (186961 / 336 - 
                     (289691 / 504) * nu + 
                     (3197 / 18) * nu**2)
        ) + 
        (730168 / 23625) * (1 / (1 + sqrt_term)) + 
        (304 / 15) * (82283 / 1995 + 
                      297674 / 1995 * e2 + 
                      1147147 / 15960 * e4 + 
                      61311 / 21280 * e6) * np.log((x / x_0_reference) * (1 + sqrt_term) / (2 * (1 - e**2)))
    )
    
    return result * prefactor


def e_dot_2_5PN_HT_in_x_e(x, e, nu):
    """
    Compute the time derivative of e at 2.5PN
    It has contributions from hereditary terms
    Eqs. A20 of https://arxiv.org/pdf/1609.05933
    
    Input:
    ------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Output:
    -------
    e_dot : float
        2.5PN term in the time derivative of e
    """

    prefactor = (32/5) * e * nu * x**4
    term1 = (55691 / 1344) * splev(e, spl_psi_e)
    term2 = (19067 / 126) * nu * splev(e, spl_znu_e)
    result = np.pi * (x**(5/2)) * prefactor * (term1 + term2)
    
    return result


def e_dot_3PN_HT_in_x_e(x, e, nu, x_0_reference=1):
    """
    Compute the time derivative of e at 3PN
    It has contributions from hereditary terms
    Eqs. A20 of https://arxiv.org/pdf/1609.05933
    
    Input:
    ------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Output:
    -------
    e_dot : float
        3PN term in the time derivative of e
    """

    # overall prefactor
    prefactor = (32/5) * e * nu * x**4
    
    # Compute the constant terms
    constant_term = (89789209 / 352800) - (87419 / 630) * np.log(2) + (78003 / 560) * np.log(3)
    
    # Calculate the final expression
    result = (
        x**3 * (
            constant_term * splev(e, spl_kappa_e) -
            (769 / 96) * (
                (16 / 3) * np.pi**2 - 
                (1712 / 105) * 0.5772156649015329 -  # Using Euler-Mascheroni constant
                (1712 / 105) * np.log((4 * x**(3 / 2)) / x_0_reference) *
                F_e(e)
            )
        )
    )
    
    return result * prefactor


def e_dot_NS_in_x_e(x, e, q, include_e_terms=None, x_0_reference=1):
    """
    Compute the adiabatic evolution equation for e based on post-Newtonian (PN) approximations.

    This function sums the contributions from various PN orders as defined in 
    Eq A19 of https://arxiv.org/pdf/1609.05933

    Parameters
    ----------
    x : float
        Expansion parameter (dimensionless).
    e : float
        Eccentricity parameter.
    q : float
        Mass ratio.
    include_hereditary_terms: Boolean
                              Whether to include hereditary terms

    Returns
    -------
    float
        The total rate of change of e considering contributions from 0.0 to 3.0 PN orders.
    """

    # Convert mass ratio to symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)

    # compute all terms
    if include_e_terms is None:
        edot_dict =  {
                    '0.0': e_dot_0PN_in_x_e(x, e, nu),
                    '1.0': e_dot_1PN_in_x_e(x, e, nu),
                    '1.5HT': e_dot_1_5PN_HT_in_x_e(x, e, nu),
                    '2.0': e_dot_2PN_in_x_e(x, e, nu),
                    '2.5HT': e_dot_2_5PN_HT_in_x_e(x, e, nu),
                    '3.0': e_dot_3PN_in_x_e(x, e, nu, x_0_reference), 
                    '3.0HT': e_dot_3PN_HT_in_x_e(x, e, nu, x_0_reference)
                    }
    else:
        edot_dict =  {}
        if '0.0' in include_e_terms:
            edot_dict['0.0'] = e_dot_0PN_in_x_e(x, e, nu)
        if '1.0' in include_e_terms:
            edot_dict['1.0'] = e_dot_1PN_in_x_e(x, e, nu)
        if '1.5HT' in include_e_terms:
            edot_dict['1.5HT'] = e_dot_1_5PN_HT_in_x_e(x, e, nu)
        if '2.0' in include_e_terms:
            edot_dict['2.0'] = e_dot_2PN_in_x_e(x, e, nu)
        if '2.5HT' in include_e_terms:
            edot_dict['2.5HT'] = e_dot_2_5PN_HT_in_x_e(x, e, nu)
        if '3.0' in include_e_terms:
            edot_dict['3.0'] = e_dot_3PN_in_x_e(x, e, nu, x_0_reference)
        if '3.0HT' in include_e_terms:
            edot_dict['3.0HT'] = e_dot_3PN_HT_in_x_e(x, e, nu, x_0_reference)
                
    
    # sum up the contributions
    if include_e_terms is None:
        edot = sum(edot_dict.values())
    else:
        edot = sum(edot_dict[key] for key in include_e_terms if key in edot_dict)

    return edot