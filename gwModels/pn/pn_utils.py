#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: pn_utils.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from ..utils import mass_ratio_to_symmetric_mass_ratio

def x_of_h(t, h):
    """
    Computes the PN expansion parameter x based on the gravitational wave strain h 
    and time t. 

    According to eq 1 of https://arxiv.org/pdf/2304.11185, this function calculates
    the orbital frequency omega from the phase of the gravitational wave strain and
    then derives x, where x is related to the orbital frequency as follows:

    Parameters
    ----------
    t : array-like
        Time values corresponding to the gravitational wave strain h.
    h : array-like
        Gravitational wave strain data.

    Returns
    -------
    x : float
        The PN expansion parameter, computed from the orbital frequency.
    """
    # Get the frequency omega from the strain h
    omega = get_frequency(t, h)
    
    # Orbital frequency is half of the phase frequency
    omega_orb = omega * 0.5
    
    # Calculate the PN expansion parameter x
    x = omega_orb**(2/3)
    
    return x


def tau_of_t(t, q, t0=0.0):
    """
    Computes the post-Newtonian (PN) time variable, tau, from the asymptotic 
    radiative coordinate time t. Based on Eq.(5) of https://arxiv.org/pdf/2304.11185.

    Parameters
    ----------
    t : float
        Coordinate time in the asymptotic radiative coordinate system.
    q : float
        Mass ratio defined as q = m1 / m2, with m1 >= m2.
    t0 : float
        Integration constant, representing the time at merger.
        Default is 0.0

    Returns
    -------
    tau : float
        PN time variable, tau, based on the symmetric mass ratio nu.
    """
    # Convert mass ratio to symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)
    
    # Compute PN time variable tau
    tau = (nu / 5) * (t0 - t)
    
    return tau
