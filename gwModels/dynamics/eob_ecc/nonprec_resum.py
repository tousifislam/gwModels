#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: nonprec.py
#
#    Unified interface for 3PN eccentric non-precessing BBH dynamics.
#    Combines secular (orbit-averaged) evolution with optional
#    post-adiabatic (PA) corrections.
#
#    AUTHOR: Tousif Islam
#    CREATED: 03-27-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import logging
import numpy as np

from .nonprec_secular_resum import integrate_eob_eccentric_dynamics
from .nonprec_postadiabatic_resum import apply_pa_corrections

logger = logging.getLogger(__name__)


def evolve_eccentric_nonprec(q, chi1=0.0, chi2=0.0, e0=0.1, zeta0=0.0,
                             omega0=0.0085, t_end=None,
                             rtol=1e-10, atol=1e-12,
                             kappa1=1.0, kappa2=1.0,
                             x_stop=None,
                             include_pa=False,
                             alpha=-16.0/3.0, beta=-13.0/2.0):
    """
    Evolve eccentric non-precessing BBH dynamics at 3PN order.

    Integrates the orbit-averaged (secular) ODEs for (e, x, zeta) and
    optionally applies post-adiabatic corrections to the solution.

    Parameters
    ----------
    q : float
        Mass ratio q = m1/m2 >= 1.
    chi1 : float
        Dimensionless spin of the heavier body (aligned with L).
    chi2 : float
        Dimensionless spin of the lighter body (aligned with L).
    e0 : float
        Initial Keplerian eccentricity.
    zeta0 : float
        Initial relativistic anomaly.
    omega0 : float
        Initial orbit-averaged orbital frequency (default 0.0085).
    t_end : float, optional
        End time. If None, estimated from circular inspiral time.
    rtol, atol : float
        ODE solver tolerances.
    kappa1, kappa2 : float
        Spin-induced quadrupole parameters (= 1 for black holes).
    include_pa : bool
        If True, apply post-adiabatic corrections to the secular solution.
    alpha, beta : float
        Gauge parameters for PA corrections (SEOBNRv5EHM defaults).

    Returns
    -------
    dict with keys:
        t : array, time
        e : array, eccentricity (secular or PA-corrected)
        x : array, PN frequency parameter (secular or PA-corrected)
        zeta : array, relativistic anomaly (secular or PA-corrected)
        e_secular : array, secular eccentricity (always present)
        x_secular : array, secular x (always present)
        zeta_secular : array, secular zeta (always present)
        success : bool
        message : str
        include_pa : bool, whether PA was applied

    References
    ----------
    - Gamboa, Khalil & Buonanno, SEOBNRv5EHM (arXiv:2408.02762)
    - PA corrections: arXiv:1904.11814
    """
    nu = q / (1.0 + q)**2

    # Run secular evolution
    secular = integrate_eob_eccentric_dynamics(
        q=q, chi1=chi1, chi2=chi2, e0=e0, zeta0=zeta0,
        omega0=omega0, t_end=t_end, rtol=rtol, atol=atol,
        kappa1=kappa1, kappa2=kappa2, x_stop=x_stop,
    )

    t = secular['t']
    e_sec = secular['e']
    x_sec = secular['x']
    zeta_sec = secular['zeta']

    result = {
        't': t,
        'e_secular': e_sec.copy(),
        'x_secular': x_sec.copy(),
        'zeta_secular': zeta_sec.copy(),
        'success': secular['success'],
        'message': secular['message'],
        'include_pa': include_pa,
    }

    if include_pa:
        logger.info("Applying post-adiabatic corrections")
        x_pa, e_pa, zeta_pa = apply_pa_corrections(
            e_sec, x_sec, zeta_sec, nu, alpha=alpha, beta=beta,
        )
        result['e'] = e_pa
        result['x'] = x_pa
        result['zeta'] = zeta_pa
    else:
        result['e'] = e_sec
        result['x'] = x_sec
        result['zeta'] = zeta_sec

    return result
