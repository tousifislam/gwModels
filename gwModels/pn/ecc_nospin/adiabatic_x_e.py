#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: adiabatic_x_e.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.integrate import odeint
from .edot_in_x_e import *
from .xdot_in_x_e import *

def update_x_e_Euler(x, e, t, dt, q, include_x_terms=None, include_e_terms=None):
    """
    Perform a single step of 4th-order Runge-Kutta integration for the system (x, e).
    
    Input:
    ------
    x : float
        Current value of the PN expansion parameter.
    e : float
        Current value of the eccentricity.
    t : float
        Current time.
    dt : float
        Time step size.
    q : float
        Mass ratio.
    
    Output:
    -------
    x_new : float
        Updated value of the PN expansion parameter after time step dt.
    e_new : float
        Updated value of the eccentricity after time step dt.
    """
    
    dx = dt * x_dot_NS_in_x_e(x, e, q, include_x_terms)
    de = dt * e_dot_NS_in_x_e(x, e, q, include_e_terms)

    # Update x and e using the RK4 formula
    x_new = x + dx
    e_new = e + de
    
    return x_new, e_new

def adiabatic_x_e_evolution(e0, x0, t0, dt, q, include_x_terms=None, include_e_terms=None, tmax=None):
    """
    Integrates the eccentricity (e) and PN parameter (x) using finite difference method until 
    the eccentricity becomes negative.

    Input:
    ------
    e0 : float
        Initial eccentricity.
    x0 : float
        Initial value of the PN expansion parameter.
    t0 : float
        Initial time.
    dt : float
        Time step size.
    q : float
        Mass ratio.
    
    Output:
    -------
    xarr : list
        Array of x values over time.
    earr : list
        Array of eccentricity values over time.
    tarr : list
        Array of time values.
    """
    
    # Arrays to store the results
    xarr = []
    earr = []
    tarr = []

    # Append initial values
    earr.append(e0)
    xarr.append(x0)
    tarr.append(t0)

    # Run the integration until eccentricity becomes negative or zero
    if tmax is None:
        while e0 > 0.0:
            x0, e0 = update_x_e_Euler(x0, e0, t0, dt, q, include_x_terms, include_e_terms)  # forward Euler
            t0 = t0 + dt  # Update time
            # Append new values to arrays
            earr.append(e0)
            xarr.append(x0)
            tarr.append(t0)
    else:
        while t0 <= tmax:
            x0, e0 = update_x_e_Euler(x0, e0, t0, dt, q, include_x_terms, include_e_terms)  # forward Euler
            t0 = t0 + dt  # Update time
            # Append new values to arrays
            earr.append(e0)
            xarr.append(x0)
            tarr.append(t0)
    
    return np.array(xarr), np.array(earr), np.array(tarr)

#TODO: Fix this function
def integrate_xdot_edot_odeint(q, x0, e0, t):
    """
    Function to perform the numerical integration of the PN parameter x and eccentricity e.

    Input:
    ------
    q : float
        Mass ratio (m2/m1), with m1 >= m2.
    x0 : float
        Initial value of the PN expansion parameter.
    e0 : float
        Initial value of the orbital eccentricity.
    t : array
        Time array over which to integrate the system.
    compute_x_dot_e_dot : function
        Function to compute the time derivatives of x and e.

    Output:
    -------
    x_sol : array
        Integrated solution for the PN expansion parameter x over time.
    e_sol : array
        Integrated solution for the orbital eccentricity e over time.
    """
    
    # Initial conditions: [x0, e0]
    y0 = [x0, e0]
    
    # Define the ODE system that will be passed to odeint
    def odeint_derivative(y, t, q):
        """
        ODE system for coupled derivatives of x and e.
        
        Input:
        ------
        y : list or array
            List of [x, e], where x is the PN parameter and e is the eccentricity.
        t : float
            Current time (not explicitly used, but required for odeint).
        q : float
            Mass ratio.

        Output:
        -------
        dydt : list
            List of time derivatives [dx/dt, de/dt].
        """
        x, e = y
        dxdt, dedt = compute_x_dot_e_dot(x, e, q)
        dydt = [dxdt, dedt]
        return dydt
    
    # Perform the integration using odeint
    sol = odeint(odeint_derivative, y0, t, args=(q,))
    
    # Extract the solutions for x and e from the integrated result
    x_sol = sol[:, 0]
    e_sol = sol[:, 1]
    
    return x_sol, e_sol