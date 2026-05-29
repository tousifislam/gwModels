#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwEccEvNSv2.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 05-28-2026
#    LAST MODIFIED: 05-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


def gwEccEvNSv2(t, q, e0, t_ref=None):
    """
    Approximate analytical model for eccentricity evolution in non-spinning BBH systems.
    From Islam et al. https://arxiv.org/pdf/2604.17868

    Parameters:
        t (array-like): Time array (in units of M). Should be negative before merger, with merger at t=0.
        q (float): Mass ratio (q >= 1)
        e0 (float): Initial eccentricity at the reference time t_ref
        t_ref (float, optional): Reference time where eccentricity is e0. If None, uses t[0].

    Returns:
        e_t (array-like): Eccentricity evolution at times t
    """
    a = -0.37857487
    b = 18.4726538
    c1 = 0.38397146
    c2 = -1.93699177
    c3 = 2.07262069

    t = np.atleast_1d(t)

    if t_ref is None:
        t_ref = t[0]

    t_c = 0.0

    nu = q / (1.0 + q)**2

    tau = nu * (t_c - t)
    tau0 = nu * (t_c - t_ref)

    n1 = a * q + b
    n2 = 1.0 + c1 * e0 + c2 * e0**2 + c3 * e0**3
    n = n1 * n2

    e_t = e0 * (tau / tau0)**(n / 48.0)

    e_t[tau <= 0] = 0.0

    return e_t
