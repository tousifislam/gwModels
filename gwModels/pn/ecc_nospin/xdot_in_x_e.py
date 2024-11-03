#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: xdot_in_x_e.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .approximated_functions import *
from ...utils import mass_ratio_to_symmetric_mass_ratio

def x_dot_0PN_NS_in_x_e(x, e, nu):
    """
    Compute the time derivative of the expansion parameter x at 0PN order.

    This function calculates the 0PN correction to the time derivative of x,
    as given in Eq. A25 of https://arxiv.org/pdf/0806.1037.

    Parameters
    ----------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Returns
    -------
    x_dot : float
        The computed time derivative of x at 0PN order.
    """
    # 0PN term
    x_dot_0PN = (2 * (37 * e**4 + 292 * e**2 + 96) * nu) / (15 * (1 - e**2)**(7/2))

    # Summing up the terms with respective powers of x
    x_dot = x_dot_0PN * x**5 
    return x_dot


def x_dot_1PN_NS_in_x_e(x, e, nu):
    """
    Compute the time derivative of the expansion parameter x at 1PN order.

    This function calculates the 1PN correction to the time derivative of x,
    as specified in Eq. A25 of https://arxiv.org/pdf/0806.1037.

    Parameters
    ----------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Returns
    -------
    x_dot : float
        The computed time derivative of x at 1PN order.
    """
    # 1PN term
    x_dot_1PN = (nu / (420 * (1 - e**2)**(9/2))) * (
        -(8288 * nu - 11717) * e**6 
        - 14 * (10122 * nu - 12217) * e**4 
        - 120 * (1330 * nu - 731) * e**2 
        - 16 * (924 * nu + 743)
    )
    
    # Summing up the terms with respective powers of x
    x_dot = x_dot_1PN * x**6 
    return x_dot

def x_dot_1p5PN_NS_HT_in_x_e(x, e, nu):
    """
    Compute the hereditary component of the time derivative of the 
    expansion parameter x at 1.5PN order.

    This function calculates the 1.5PN correction to the time derivative of x,
    as described in Eq. A25 of https://arxiv.org/pdf/0806.1037.

    Parameters
    ----------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Returns
    -------
    x_dot : float
        The computed time derivative of x at 1.5PN order.
    """
    # 1.5PN term
    kappa_E = compute_aparox_kappa_E(e)
    x_dot_1_5PN = (256 / 5) * nu * np.pi * kappa_E

    # Summing up the terms with respective powers of x
    x_dot = x_dot_1_5PN * x**(13/2)
    return x_dot
    
    
def x_dot_2PN_NS_in_x_e(x, e, nu):
    """
    Compute the time derivative of the expansion parameter x at 2PN order.

    This function calculates the 2PN correction to the time derivative of x,
    as outlined in Eq. A25 of https://arxiv.org/pdf/0806.1037.

    Parameters
    ----------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.

    Returns
    -------
    x_dot : float
        The computed time derivative of x at 2PN order.
    """
    # 2PN term
    x_dot_2PN = (nu / (45360 * (1 - e**2)**(11/2))) * (
        (1964256 * nu**2 - 3259980 * nu + 3523113) * e**8 
        + (64828848 * nu**2 - 123108426 * nu + 83424402) * e**6 
        + (16650606060 * nu**2 - 207204264 * nu + 783768) * e**4 
        + (61282032 * nu**2 + 15464736 * nu - 92846560) * e**2 
        + 1903104 * nu**2 
        + np.sqrt(1 - e**2) * (
            (2646000 - 1058400 * nu) * e**6 + 
            (64532160 - 25812864 * nu) * e**2 - 
            580608 * nu + 1451520
        ) 
        + 4514976 * nu - 360224
    )

    # Summing up the terms with respective powers of x
    x_dot = x_dot_2PN * x**7
    return x_dot

def x_dot_2p5PN_NS_HT_in_x_e(x, e, nu):
    """
    Compute the HT correction to the rate of change of x
    as specified in Eq. A7 of https://arxiv.org/pdf/1609.05933
    
    Parameters
    ----------
    x : float
        Expansion parameter (dimensionless).
    e : float
        Eccentricity parameter.
    nu : float
        Symmetric mass ratio.

    Returns
    -------
    float
        The HT correction to the time derivative of x at 2.5PN order.
    """
    # Define the HT correction term
    term_1 = (256 * np.pi / (1 - e**2)) * special_function_phi(e) + (2 / 3) * (
        -17599 * np.pi / 35 * special_function_psi(e)
        - 2268 * nu * np.pi / 5 * special_function_znu(e)
        - 788 * np.pi * e**2 / (1 - e**2)**2 * special_function_varphi_e(e)
    )

    # Result calculation
    result = nu * x**(13 / 2) * term_1 * x
    
    return result

def x_dot_3PN_NS_in_x_e(x, e, nu, x_0_reference=1):
    """
    Compute the 3PN correction to the rate of change of x with eccentric effects.

    This function calculates the 3PN correction to the time derivative of x,
    as specified in Eq A6 of https://arxiv.org/pdf/1609.05933.

    Parameters
    ----------
    x : float
        The PN expansion parameter (dimensionless).
    e : float
        The orbital eccentricity.
    nu : float
        The symmetric mass ratio.
    x0 : float
        Initial reference value of x.

    Returns
    -------
    float
        The computed time derivative of x at 3PN order.
    """
    # Define constants
    pi2 = np.pi ** 2
    sqrt_term = np.sqrt(1 - e**2)
    e2, e4, e6, e8, e10 = e**2, e**4, e**6, e**8, e**10
    ln_term = np.log(x / x_0_reference * (1 + sqrt_term) / (2 * (1 - e**2)))

    # Compute the expression
    result = (nu / (598752000 * (1 - e**2)**(13 / 2))) * (
        25 * e10 * (2699947161 - 176 * nu * (4 * nu * (2320640 * nu - 2962791) + 16870887)) +
        32 * e2 * (
            55 * nu * (270 * (7015568 * sqrt_term - 9657701) * nu - 8125851600 * sqrt_term + 
            38745 * pi2 * (1121 * sqrt_term + 1185) - 901169500 * nu**2 + 5387647438) +
            31050413856 * sqrt_term + 358275866598
        ) +
        128 * (
            -275 * nu * (81 * (16073 - 17696 * sqrt_term) * nu - 1066392 * sqrt_term +
            46494 * pi2 * (sqrt_term - 45) + 470820 * nu**2 + 57265081) -
            3950984268 * sqrt_term + 12902173599
        ) +
        e8 * (
            162 * (1240866000 * sqrt_term + 19698134267) -
            1100 * nu * (16 * nu * (-3582684 * sqrt_term + 137570300 * nu - 286933509) +
            27 * (6843728 * sqrt_term + 255717 * pi2 + 173696120))
        ) +
        12 * e6 * (
            55 * nu * (90 * (52007648 * sqrt_term + 311841025) * nu +
            3 * (4305 * pi2 * (14 * sqrt_term - 19113) - 5464335200 * sqrt_term + 767166806) -
            17925404000 * nu**2) +
            742016570592 * sqrt_term + 6005081022
        ) +
        8 * e4 * (
            55 * nu * (270 * (71069152 * sqrt_term + 6532945) * nu -
            74508169680 * sqrt_term + 116235 * pi2 * (1510 * sqrt_term - 4807) -
            23638717900 * nu**2 + 88628306866) +
            6 * (332891836596 * sqrt_term + 8654689873)
        ) +
        40677120 * (891 * e8 + 28016 * e6 + 82736 * e4 + 43520 * e2 + 3072) * ln_term
    )

    return result * x**8


def x_dot_3PN_NS_HT_in_x_e(x, e, nu, x_0_reference=1):
    """
    Compute the HT correction to the rate of change of x
    as specified in Eq. A7 of https://arxiv.org/pdf/1609.05933

    Parameters
    ----------
    x : float
        Expansion parameter (dimensionless).
    e : float
        Eccentricity parameter.
    nu : float
        Symmetric mass ratio.

    Returns
    -------
    float
        The HT correction to the time derivative of x at 3PN order.
    """
    # HT correction terms
    term_3 = (64 / 18375) * (
        -116761 * special_function_kappa(e) + (19600 * np.pi**2 - 59920 * np.euler_gamma - 59920 * np.log(4 * x**(3 / 2) / x_0_reference)) * special_function_F(e)
    )

    # Result calculation
    result = nu * x**(13 / 2) * term_3 * x**(3 / 2)
    
    return result

def x_dot_3p5PN_correction_in_x(x, nu):
    """
    Compute the 3.5PN correction term for d(x)/dt in the circular orbit limit (e -> 0)
    as specified in Eq 16 of https://arxiv.org/pdf/1609.05933
    
    Parameters
    ----------
    x : float
        PN expansion parameter (dimensionless).
    nu : float
        Symmetric mass ratio (dimensionless).

    Returns
    -------
    float
        The 3.5PN correction term the time derivative of x at 3.5PN order.
    """
    # Define the coefficients for the correction term
    term1 = -4415 / 4032
    term2 = 358675 / 6048 * nu
    term3 = 91945 / 1512 * nu**2
    
    # Compute the correction term
    dx_dt_correction = (64 * np.pi / 5) * nu * x**5 * (term1 + term2 + term3) * x**(7 / 2)
    
    return dx_dt_correction

def x_dot_6PN_correction_in_x(x, nu):
    """
    Calculate the correction to the time derivative of x upto 6PN using perturbation theory
    Taken from Eq(A2) of https://arxiv.org/pdf/1609.05933
    
    Parameters:
    nu (float): Symmetric mass ratio.
    x (float): Dimensionless parameter related to the separation.

    Returns:
    float:  correction to the time derivative of x.
    """
    term_1 = compute_a4(nu, x) * x**4 
    term_2 = compute_a9_2(nu, x) * x**(9/2) 
    term_3 = compute_a5(nu, x) * x**5 
    term_4 = compute_a11_2(nu, x) * x**(11/2) 
    term_5 = compute_a6(nu, x) * x**6
    result = (64 * nu * x**5 / 5) * ( term_1 + term_2 + term_3 + term_4 + term_5 )
    return result

def x_dot_NS_in_x_e(x, e, q, include_x_terms=None, x_0_reference=1):
    """
    Compute the adiabatic evolution equation for x based on post-Newtonian (PN) approximations.

    This function sums the contributions from various PN orders as defined in 
    Eq A2 of https://arxiv.org/pdf/1609.05933.

    Parameters
    ----------
    x : float
        Expansion parameter (dimensionless).
    e : float
        Eccentricity parameter.
    q : float
        Mass ratio.

    Returns
    -------
    float
        The total rate of change of x considering contributions from 0.0 to 3.5 PN orders.
    """

    # Convert mass ratio to symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)

    # compute all terms
    if include_x_terms is None:
        xdot_dict =  {
                    '0.0': x_dot_0PN_NS_in_x_e(x, e, nu),
                    '1.0': x_dot_1PN_NS_in_x_e(x, e, nu),
                    '1.5HT': x_dot_1p5PN_NS_HT_in_x_e(x, e, nu),
                    '2.0': x_dot_2PN_NS_in_x_e(x, e, nu),
                    '2.5HT': x_dot_2p5PN_NS_HT_in_x_e(x, e, nu),
                    '3.0': x_dot_3PN_NS_in_x_e(x, e, nu, x_0_reference),
                    '3.0HT': x_dot_3PN_NS_HT_in_x_e(x, e, nu, x_0_reference),
                    '3.5Cir': x_dot_3p5PN_correction_in_x(x, nu),
                    '6.0pp': x_dot_6PN_correction_in_x(x, nu)
                    }
    else:
        xdot_dict =  {}
        if '0.0' in include_x_terms:
            xdot_dict['0.0'] = x_dot_0PN_NS_in_x_e(x, e, nu)
        if '1.0' in include_x_terms:
            xdot_dict['1.0'] = x_dot_1PN_NS_in_x_e(x, e, nu)
        if '1.5HT' in include_x_terms:
            xdot_dict['1.5HT'] = x_dot_1p5PN_NS_HT_in_x_e(x, e, nu)
        if '2.0' in include_x_terms:
            xdot_dict['2.0'] = x_dot_2PN_NS_in_x_e(x, e, nu)
        if '2.5HT' in include_x_terms:
            xdot_dict['2.5HT'] = x_dot_2p5PN_NS_HT_in_x_e(x, e, nu)
        if '3.0' in include_x_terms:
            xdot_dict['3.0'] = x_dot_3PN_NS_in_x_e(x, e, nu, x_0_reference)
        if '3.0HT' in include_x_terms:
            xdot_dict['3.0HT'] = x_dot_3PN_NS_HT_in_x_e(x, e, nu, x_0_reference)
        if '3.5Cir' in include_x_terms:
            xdot_dict['3.5Cir'] = x_dot_3p5PN_correction_in_x(x, nu)
        if '6.0PP' in include_x_terms:
            xdot_dict['6.0pp'] = x_dot_6PN_correction_in_x(x, nu)

    # sum up the contributions
    if include_x_terms is None:
        xdot = sum(xdot_dict.values())
    else:
        xdot = sum(xdot_dict[key] for key in include_x_terms if key in xdot_dict)

    return xdot