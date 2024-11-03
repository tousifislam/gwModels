#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: l_in_x_e.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools

def compute_mean_motion_n_from_x_e(x, e, nu):
    """
    Calculate the mean motion up to 3PN order.
    Eq A1 of https://arxiv.org/pdf/0806.1037
    
    Input:
    ------
    x : float
        Post-Newtonian parameter.
    e : float
        Eccentricity.
    nu : float
        Symmetric mass ratio.
    
    Output:
    -------
    Mn : float
        Mean motion at 3PN order.
    """
    # PN corrections
    n_1PN = 3 / (e**2 - 1)
    n_2PN = ((26 * nu - 51) * e**2 + 28 * nu - 18) / (4 * (e**2 - 1)**2)

    term1 = (1536 * nu - 3840) * e**4
    term2 = (1920 - 768 * nu) * e**2
    term3 = -768 * nu
    term4 = np.sqrt(1 - e**2) * (
        (1040 * nu**2 - 1760 * nu + 2496) * e**4 +
        (5120 * nu**2 + 123 * np.pi**2 * nu - 17856 * nu + 8544) * e**2 +
        896 * nu**2 - 14624 * nu + 492 * nu * np.pi**2 - 192
    )
    term5 = 1920
    
    n_3PN = -1 / (128 * (1 - e**2)**(7/2)) * (term1 + term2 + term3 + term4 + term5)
    
    # Calculate Mn up to 3PN order
    Mn = x**(3/2) + n_1PN * x**(5/2) + n_2PN * x**(7/2) + n_3PN * x**(9/2)
    
    return Mn


def compute_l_dot_from_x_e(x, e, nu):
    """
    Calculate the time derivative of the mean anomaly up to 3PN order.
    Eq 8 of https://arxiv.org/pdf/0806.1037
    
    Input:
    ------
    x : float
        Post-Newtonian parameter.
    e : float
        Eccentricity.
    nu : float
        Symmetric mass ratio.
    
    Output:
    -------
    dl/dt : float
        Mean motion at 3PN order.
    """
    l_dot = compute_mean_motion_n_from_x_e(x, e, nu)
    
    return l_dot


def integrate_dldt_to_get_l(x, e, t, q, l0=0):
    """
    Estimate the mean anomaly over time using the forward Euler method.
    Integrate Eq 8 of https://arxiv.org/pdf/0806.1037

    Input:
    ------
    x : array
        Array of PN expansion parameters over time.
    e : array
        Array of eccentricities over time.
    t : array
        Time array for the integration.
    q : float
        Mass ratio (m2/m1), with m1 >= m2.
    l0 : float, optional
        Initial value of the mean anomaly (default is 0).

    Output:
    -------
    l : array
        Estimated mean anomaly over time.
    """
    # Initialize the array for the mean anomaly
    l = np.zeros(len(t))
    l[0] = l0  # Set the initial value

    # Time step size
    dt = t[1] - t[0]

    # Perform the integration using forward Euler
    for index in range(1, len(t)):
        ldot = compute_l_dot_from_x_e(x=x[index], e=e[index], nu=gwtools.q_to_nu(q))
        l0 = l0 + ldot * dt  # Update mean anomaly
        l[index] = l0  # Store the updated mean anomaly

    return l