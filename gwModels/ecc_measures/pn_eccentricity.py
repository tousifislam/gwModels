#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: pn_eccentricity.py
#    Standalone post-Newtonian eccentricity evolution formulas for
#    non-spinning compact binaries.
#
#    References:
#      - Moore, Favata & Arun (2016), Phys. Rev. D 93, 124061
#        [arXiv:1605.00304], Appendix C, Eq. C1
#
#       AUTHOR: Tousif Islam
#       CREATED: 05-30-2026
#       REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools


def compute_tau(t, q, tc=0, t_ref=None):
    """Compute dimensionless time variables tau and tau_0.

    Parameters:
        t (array_like): Time array.
        q (float): Mass ratio (q >= 1).
        tc (float): Time of coalescence. Default 0.
        t_ref (float): Reference time. If None, uses t[0].

    Returns:
        tau, tau_0, eta
    """
    eta = gwtools.q_to_nu(q)
    tau = (tc - t) * (eta / 5)
    if t_ref is None:
        t_ref = t[0]
    tau_0 = (tc - t_ref) * (eta / 5)
    return tau, tau_0, eta


def Newtonian_e_t(t, e_0, q, tc=0, t_ref=None):
    """
    Newtonian (leading-order) eccentricity evolution.
    Page 41, Eq C1 of arXiv:1605.00304 (Moore, Favata & Arun 2016).

    Parameters:
        t (array_like): Time array.
        e_0 (float): Reference eccentricity at t_ref.
        q (float): Mass ratio (q >= 1).
        tc (float): Time of coalescence. Default 0.
        t_ref (float): Reference time. If None, uses t[0].

    Returns:
        array: Eccentricity evolution e(t).
    """
    tau, tau_0, eta = compute_tau(t, q, tc, t_ref)
    return e_0 * (tau / tau_0) ** (19 / 48)


def PN2_e_t(t, e_0, q, tc=0, t_ref=None):
    """
    2PN eccentricity evolution.
    Page 41, Eq C1 of arXiv:1605.00304 (Moore, Favata & Arun 2016),
    truncated at 2PN order (test-mass limit: eta-independent terms only).

    Parameters:
        t (array_like): Time array.
        e_0 (float): Reference eccentricity at t_ref.
        q (float): Mass ratio (q >= 1).
        tc (float): Time of coalescence. Default 0.
        t_ref (float): Reference time. If None, uses t[0].

    Returns:
        array: Eccentricity evolution e(t).
    """
    tau, tau_0, eta = compute_tau(t, q, tc, t_ref)

    term1 = (tau / tau_0) ** (19 / 48)
    term2 = -4445 / 6912 * (tau ** (-1 / 4) - tau_0 ** (-1 / 4))
    term3 = 854531845 / 4682022912 * tau ** (-1 / 2)
    term4 = 1081754605 / 4682022912 * tau_0 ** (-1 / 2)
    term5 = -19758025 / 47775744 * tau ** (-1 / 4) * tau_0 ** (-1 / 4)
    term6 = -3721 / 33177600 * np.pi ** 2 * tau ** (-3 / 8) * tau_0 ** (-3 / 8)
    term7 = (255918223951763603 / 186891372173721600
             - 15943 / 80640 * np.euler_gamma
             - 7926071 / 66355200 * np.pi ** 2) * tau ** (-3 / 4)
    term8 = (-250085444105408603 / 186891372173721600
             + 15943 / 80640 * np.euler_gamma
             + 7933513 / 66355200 * np.pi ** 2) * tau_0 ** (-3 / 4)

    g_2pn = term1 * (1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
    return e_0 * g_2pn


def PN3_e_t(t, e_0, q, tc=0, t_ref=None):
    """
    3PN eccentricity evolution.
    Page 41, Eq C1 of arXiv:1605.00304 (Moore, Favata & Arun 2016).

    Parameters:
        t (array_like): Time array.
        e_0 (float): Reference eccentricity at t_ref.
        q (float): Mass ratio (q >= 1).
        tc (float): Time of coalescence. Default 0.
        t_ref (float): Reference time. If None, uses t[0].

    Returns:
        array: Eccentricity evolution e(t).
    """
    tau, tau_0, eta = compute_tau(t, q, tc, t_ref)

    term1 = e_0 * (tau / tau_0) ** (19 / 48)
    term2 = (-4445 / 6912 + 185 / 576 * eta) * (tau ** (-1 / 4) - tau_0 ** (-1 / 4))
    term3 = -61 / 5760 * np.pi * (tau ** (-3 / 8) - tau_0 ** (-3 / 8))
    term4 = (854531845 / 4682022912 - 15215083 / 27869184 * eta + 72733 / 663552 * eta ** 2) * tau ** (-1 / 2)
    term5 = (1081754605 / 4682022912 + 3702533 / 27869184 * eta - 4283 / 663552 * eta ** 2) * tau_0 ** (-1 / 2)
    term6 = (-19758025 / 47775744 + 822325 / 1990656 * eta - 34225 / 331776 * eta ** 2) * tau ** (-1 / 4) * tau_0 ** (-1 / 4)
    term7 = (104976437 / 278691840 - 4848113 / 23224320 * eta) * np.pi * tau ** (-5 / 8)
    term8 = (-101180407 / 278691840 + 4690123 / 23224320 * eta) * np.pi * tau_0 ** (-5 / 8)
    term9 = np.pi * (-54229 / 7962624 + 2257 / 663552 * eta) * (tau ** (-1 / 4) * tau_0 ** (-3 / 8) + tau ** (-3 / 8) * tau_0 ** (-1 / 4))
    term10 = (-686914174175 / 4623163195392 - 10094675555 / 898948399104 * eta + 501067585 / 10701766656 * eta ** 2 - 792355 / 382205952 * eta ** 3) * tau ** (-1 / 4) * tau_0 ** (-1 / 2)
    term11 = -3721 / 33177600 * np.pi ** 2 * tau ** (-3 / 8) * tau_0 ** (-3 / 8)
    term12 = (542627721575 / 4623163195392 - 122769222935 / 299649466368 * eta + 2630889335 / 10701766656 * eta ** 2 - 13455605 / 382205952 * eta ** 3) * tau ** (-1 / 2) * tau_0 ** (-1 / 4)
    term13 = (255918223951763603 / 186891372173721600
              - 15943 / 80640 * np.euler_gamma
              - 7926071 / 66355200 * np.pi ** 2
              + (-81120341684927 / 13484225986560 + 12751 / 49152 * np.pi ** 2) * eta
              - 3929671247 / 32105299968 * eta ** 2
              + 25957133 / 1146617856 * eta ** 3
              - 8453 / 15120 * np.log(2)
              + 26001 / 71680 * np.log(3)
              + 15943 / 645120 * np.log(tau)) * tau ** (-3 / 4)
    term14 = (-250085444105408603 / 186891372173721600
              + 15943 / 80640 * np.euler_gamma
              + 7933513 / 66355200 * np.pi ** 2
              + (86796376850327 / 13484225986560 - 12751 / 49152 * np.pi ** 2) * eta
              - 5466199513 / 32105299968 * eta ** 2
              + 16786747 / 1146617856 * eta ** 3
              + 8453 / 15120 * np.log(2)
              - 26001 / 71680 * np.log(3)
              - 15943 / 645120 * np.log(tau_0)) * tau_0 ** (-3 / 4)

    result = term1 * (1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
                      + term9 + term10 + term11 + term12 + term13 + term14)
    return result
