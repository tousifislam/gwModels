#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: phi_phidot_in_e_u.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import scipy
import gwtools

def phi_dot_0_PN_in_e_u(e, u):
    """
    Computes the leading-order post-Newtonian contribution to the angular frequency.
    Eq A11 of https://arxiv.org/pdf/0806.1037
    
    Parameters:
    e (float): Eccentricity
    u (float): True anomaly (in radians)

    Returns:
    float: Value of \dot{\phi}_{0 PN}
    """
    return np.sqrt(1 - e**2) / (e * np.cos(u) - 1)**2


def phi_dot_1_PN_in_e_u(e, u, nu):
    """
    Computes the first-order post-Newtonian contribution to the angular frequency.
    Eq A12 of https://arxiv.org/pdf/0806.1037
    
    Parameters:
    e (float): Eccentricity
    u (float): True anomaly (in radians)
    nu (float): Symmetric mass ratio

    Returns:
    float: Value of \dot{\phi}_{1 PN}
    """
    return -e * (nu - 4) * (e - np.cos(u)) / (np.sqrt(1 - e**2) * (e * np.cos(u) - 1)**3)


def phi_dot_2_PN_in_e_u(e, u, nu):
    """
    Computes the second-order post-Newtonian contribution to the angular frequency.
    Eq A13 of https://arxiv.org/pdf/0806.1037
    
    Parameters:
    e (float): Eccentricity
    u (float): True anomaly (in radians)
    nu (float): Symmetric mass ratio

    Returns:
    float: Value of \dot{\phi}_{2 PN}
    """
    sqrt_1_minus_e2 = np.sqrt(1 - e**2)
    cos_u = np.cos(u)

    return (1 / (12 * (1 - e**2)**(3 / 2) * (e * cos_u - 1)**5)) * (
        (-12 * nu**2 - 18 * nu) * e**6 +
        (20 * nu**2 - 26 * nu - 60) * e**4 +
        (-2 * nu**2 + 50 * nu + 75) * e**2 +
        ((-14 * nu**2 + 8 * nu - 147) * e**5 + (8 * nu**2 + 22 * nu + 42) * e**3) * cos_u**3 +
        ((17 * nu**2 - 17 * nu + 48) * e**6 + (-4 * nu**2 - 38 * nu + 153) * e**4 + 
         (5 * nu**2 - 35 * nu + 114) * e**2) * cos_u**2 -
        36 * nu +
        ((-nu**2 + 97 * nu + 12) * e**5 + (-16 * nu**2 - 74 * nu - 81) * e**3 + 
         (-nu**2 + 67 * nu - 246) * e) * cos_u +
        sqrt_1_minus_e2 * (
            e**3 * (36 * nu - 90) * cos_u**3 +
            ((180 - 72 * nu) * e**4 + (90 - 36 * nu) * e**2) * cos_u**2 +
            ((144 * nu - 360) * e**3 + (90 - 36 * nu) * e) * cos_u +
            e**2 * (180 - 72 * nu) + 36 * nu - 90
        ) + 90
    )


def calculate_term_9_in_phi_dot_3_PN(e, u, nu):
    """
    Compute the specified expression involving eccentricity, true anomaly, and mass ratio.
    Eq A14 of https://arxiv.org/pdf/0806.1037
    
    Parameters:
    e (float): Eccentricity of the binary system.
    u (float): True anomaly (in radians).
    nu (float): Symmetric mass ratio (nu = m1*m2 / (m1 + m2)^2).

    Returns:
    float: The computed value of the expression.
    """
    # Calculate the square root term
    sqrt_term = np.sqrt(1 - e**2)
    
    # Calculate polynomial contributions
    term1 = ((-127680 * nu**2 + 544320 * nu - 739200) * e**7 +
              (-53760 * nu**2 - 8610 * np.pi**2 * nu + 674240 * nu - 67200) * e**5) * np.cos(u)**5
    
    term2 = ((161280 * nu**2 - 477120 * nu + 537600) * e**8 +
              (477120 * nu**2 + 17220 * np.pi**2 * nu - 2894080 * nu + 2217600) * e**6 +
              (268800 * nu**2 + 25830 * np.pi**2 * nu - 2721600 * nu + 1276800) * e**4) * np.cos(u)**4
    
    term3 = ((-524160 * nu**2 + 1122240 * nu - 940800) * e**7 +
              (-873600 * nu**2 - 68880 * np.pi**2 * nu + 7705600 * nu - 3897600) * e**5 +
              (-416640 * nu**2 - 17220 * np.pi**2 * nu + 3357760 * nu - 3225600) * e**3) * np.cos(u)**3
    
    term4 = ((604800 * nu**2 - 504000 * nu - 403200) * e**6 +
              (1034880 * nu**2 + 103320 * np.pi**2 * nu - 11195520 * nu + 5779200) * e**4 +
              (174720 * nu**2 - 17220 * np.pi**2 * nu - 486080 * nu + 2688000) * e**2) * np.cos(u)**2
    
    term5 = ((-282240 * nu**2 - 450240 * nu + 1478400) * e**5 +
              (-719040 * nu**2 - 68880 * np.pi**2 * nu + 8128960 * nu - 5040000) * e**3 +
              (94080 * nu**2 + 25830 * np.pi**2 * nu - 1585920 * nu - 470400) * e) * np.cos(u)
    
    # Constant terms
    constant_term = -67200 * nu**2 + 761600 * nu
    
    # Additional polynomial terms
    additional_terms = (e**4 * (40320 * nu**2 + 309120 * nu - 672000) +
                        e**2 * (208320 * nu**2 + 17220 * np.pi**2 * nu - 2289280 * nu + 1680000) -
                        8610 * nu * np.pi**2 - 201600)
    
    # Final expression
    result = sqrt_term * (term1 + term2 + term3 + term4 + term5 + constant_term + additional_terms)

    return result + (8610 * nu * np.pi**2 + 201600)


def phi_dot_3_PN_in_e_u(e, u, nu):
    """
    Compute the 3PN rate of change of the phase for an eccentric binary.
    Eq A10 of https://arxiv.org/pdf/0806.1037
    
    Parameters:
    e (float): Eccentricity of the binary system.
    u (float): True anomaly (in radians).
    nu (float): Symmetric mass ratio (nu = m1*m2 / (m1 + m2)^2).

    Returns:
    float: The value of \dot{phi}_{3PN}.
    """
    # Calculate cosine values
    cos_u = np.cos(u)
    cos2_u = cos_u ** 2
    cos3_u = cos_u ** 3
    cos4_u = cos_u ** 4
    cos5_u = cos_u ** 5

    # Denominator
    denominator = 13440 * (1 - e ** 2) ** (5 / 2) * (e * cos_u - 1) ** 7

    # Calculate the polynomial parts
    term1 = (10080 * nu ** 3 + 40320 * nu ** 2 - 15120 * nu) * e ** 10
    term2 = (-52640 * nu ** 3 - 13440 * nu ** 2 + 483280 * nu) * e ** 8
    term3 = (84000 * nu ** 3 - 190400 * nu ** 2 - 17220 * np.pi ** 2 * nu - 50048 * nu - 241920) * e ** 6
    term4 = (-52640 * nu ** 3 + 516880 * nu ** 2 + 68880 * np.pi ** 2 * nu - 1916048 * nu + 262080) * e ** 4
    term5 = (4480 * nu ** 3 - 412160 * nu ** 2 - 30135 * np.pi ** 2 * nu + 553008 * nu + 342720) * e ** 2
    
    term6 = ((13440 * nu ** 3 + 94640 * nu ** 2 - 113680 * nu - 221760) * e ** 9 +
              (-11200 * nu ** 3 - 112000 * nu ** 2 + 12915 * np.pi ** 2 * nu + 692928 * nu - 194880) * e ** 7 +
              (4480 * nu ** 3 + 8960 * nu ** 2 - 43050 * np.pi ** 2 * nu + 1127280 * nu - 147840) * e ** 5) * cos5_u

    term7 = ((-16240 * nu ** 3 + 12880 * nu ** 2 + 18480 * nu) * e ** 10 +
              (16240 * nu ** 3 - 91840 * nu ** 2 + 17220 * np.pi ** 2 * nu - 652192 * nu + 100800) * e ** 8 +
              (-55440 * nu ** 3 + 34160 * nu ** 2 - 30135 * np.pi ** 2 * nu - 2185040 * nu + 2493120) * e ** 6 +
              (21840 * nu ** 3 + 86800 * nu ** 2 + 163590 * np.pi ** 2 * nu - 5713888 * nu + 228480) * e ** 4) * cos4_u

    term8 = ((560 * nu ** 3 - 137200 * nu ** 2 + 388640 * nu + 241920) * e ** 9 +
              (30800 * nu ** 3 - 264880 * nu ** 2 - 68880 * np.pi ** 2 * nu + 624128 * nu + 766080) * e ** 7 +
              (66640 * nu ** 3 + 612080 * nu ** 2 - 8610 * np.pi ** 2 * nu + 6666080 * nu - 6652800) * e ** 5 +
              (-30800 * nu ** 3 - 294000 * nu ** 2 - 223860 * np.pi ** 2 * nu + 9386432 * nu) * e ** 3) * cos3_u

    term9 = calculate_term_9_in_phi_dot_3_PN(e, u, nu)

    return (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9) / denominator


def compute_phi_dot_in_x_e_u(x, e, u, q):
    """
    Computes the total angular frequency M \dot{\phi} using contributions from various PN orders.
    Eq 6 and A10 of https://arxiv.org/pdf/0806.1037
    
    Parameters:
    x (float): Post-Newtonian parameter.
    e (float): Eccentricity
    u (float): True anomaly (in radians)
    nu (float): Symmetric mass ratio

    Returns:
    float: Total value of \dot{\phi}
    """
    # symmetric mass ratio
    nu = gwtools.q_to_nu(q)
    
    res = phi_dot_0_PN_in_e_u(e, u) * x**(3/2)
    res = res + phi_dot_1_PN_in_e_u(e, u, nu) * x**(5/2)
    res = res + phi_dot_2_PN_in_e_u(e, u, nu) * x**(7/2)
    res = res + phi_dot_3_PN_in_e_u(e, u, nu) * x**(9/2)

    return res

def integrate_phidot_to_get_phi(phi_dot, t, phi_0):
    """
    Integrate the angular velocity (phidot) to get the angle (phi).

    Parameters:
    phi_dot (array-like): The angular velocity values at each time step.
    t (array-like): The time values corresponding to the angular velocity.
    phi_0 (float): The initial angle (phi) at t=0.

    Returns:
    numpy.ndarray: The integrated angle values at each time step.
    """
    # Use cumulative trapezoidal integration to calculate the integral
    try:
        phi = phi_0 + scipy.integrate.cumtrapz(phi_dot, t, initial=0)
    except:
        phi = phi_0 + scipy.integrate.cumulative_trapezoid(phi_dot, t, initial=0)
    return phi