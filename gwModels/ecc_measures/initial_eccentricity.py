#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: initial_eccentricity.py
#    Computes initial eccentricity from orbital binding energy and angular
#    momentum in two different gauges (harmonic and ADM coordinates).
#
#    Based on:
#      - Eq(A120) and Eq(A121) of Wang et al., arXiv:2409.17636
#      - R.-M. Memmesheimer, A. Gopakumar, and G. Schaefer,
#        Phys. Rev. D 70, 104011 (2004), arXiv:gr-qc/0407049
#
#       AUTHOR: Tousif Islam
#       CREATED: 03-27-2026
#       REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


def compute_et_harmonic_3PN(Eb, L, eta):
    """
    Compute the initial time-eccentricity e_t in harmonic coordinates
    at 3PN order for nonspinning compact binaries using the generalized
    quasi-Keplerian parametrization.

    Eq(A120) of arXiv:2409.17636, based on Memmesheimer, Gopakumar,
    and Schaefer, Phys. Rev. D 70, 104011 (2004).

    Parameters:
        Eb (float): Orbital binding energy (negative for bound orbits).
        L (float): Orbital angular momentum.
        eta (float): Symmetric mass ratio, eta = m1*m2/(m1+m2)^2.

    Returns:
        float: The time-eccentricity e_t in harmonic coordinates.
    """
    # Rescaled energy and angular momentum
    E0 = Eb / eta
    h0 = L / eta

    # Shorthand: eps = -2*E0 > 0 for bound orbits
    eps = -2.0 * E0
    j2 = h0**2
    epsj2 = eps * j2           # = -2*E0*h0^2
    sqrt_epsj2 = np.sqrt(epsj2)

    pi2 = np.pi**2

    # ------------------------------------------------------------------
    # Newtonian
    # ------------------------------------------------------------------
    et2 = 1.0 - epsj2

    # ------------------------------------------------------------------
    # 1PN
    # ------------------------------------------------------------------
    et2 += (eps / 4.0) * (
        -8.0 + 8.0 * eta
        - epsj2 * (-17.0 + 7.0 * eta)
    )

    # ------------------------------------------------------------------
    # 2PN
    # ------------------------------------------------------------------
    et2 += (eps**2 / 8.0) * (
        12.0 + 72.0 * eta + 20.0 * eta**2
        - 24.0 * sqrt_epsj2 * (-5.0 + 2.0 * eta)
        - epsj2 * (112.0 - 47.0 * eta + 16.0 * eta**2)
        - (16.0 / epsj2) * (-4.0 + 7.0 * eta)
        + (24.0 / sqrt_epsj2) * (-5.0 + 2.0 * eta)
    )

    # ------------------------------------------------------------------
    # 3PN
    # ------------------------------------------------------------------
    et2 += (eps**3 / 6720.0) * (
        23520.0 - 464800.0 * eta + 179760.0 * eta**2 + 16800.0 * eta**3
        - 2520.0 * sqrt_epsj2 * (265.0 - 193.0 * eta + 46.0 * eta**2)
        - 525.0 * epsj2 * (-528.0 + 200.0 * eta - 77.0 * eta**2 + 24.0 * eta**3)
        - (6.0 / epsj2) * (73920.0 - 260272.0 * eta + 4305.0 * pi2 * eta + 61040.0 * eta**2)
        + (70.0 / sqrt_epsj2) * (16380.0 - 19964.0 * eta + 123.0 * pi2 * eta + 3240.0 * eta**2)
        + (8.0 / epsj2) * (53760.0 - 176024.0 * eta + 4305.0 * pi2 * eta + 15120.0 * eta**2)
        - (70.0 / epsj2**1.5) * (10080.0 - 13952.0 * eta + 123.0 * pi2 * eta + 1440.0 * eta**2)
    )

    return np.sqrt(et2)


def compute_et_ADM_2PN(Eb, L, eta, chi1=0.0, chi2=0.0):
    """
    Compute the initial time-eccentricity e_t in ADM coordinates
    at 2PN order for spin-aligned compact binaries using the
    quasi-Keplerian parametrization.

    Eq(A121) of arXiv:2409.17636, based on Memmesheimer, Gopakumar,
    and Schaefer, Phys. Rev. D 70, 104011 (2004).

    Parameters:
        Eb (float): Orbital binding energy (negative for bound orbits).
        L (float): Orbital angular momentum.
        eta (float): Symmetric mass ratio, eta = m1*m2/(m1+m2)^2.
        chi1 (float): Dimensionless spin of the heavier body. Default 0.
        chi2 (float): Dimensionless spin of the lighter body. Default 0.

    Returns:
        float: The time-eccentricity e_t in ADM coordinates.
    """
    # Rescaled energy and angular momentum
    E0 = Eb / eta
    h0 = L / eta
    absE0 = abs(E0)

    # Mass-difference parameter
    delta = np.sqrt(abs(1.0 - 4.0 * eta))

    # Spin combinations
    chi_s = chi1 + chi2   # symmetric
    chi_a = chi1 - chi2   # antisymmetric

    # Spin-spin coupling constants (nonprecessing case)
    lam1 = -0.5
    lam2 = -0.5
    lam_s = lam1 + lam2   # = -1
    lam_a = lam1 - lam2   # = 0

    sqrt_1m4eta = np.sqrt(abs(1.0 - 4.0 * eta))
    sqrt2 = np.sqrt(2.0)

    # ------------------------------------------------------------------
    # 0PN + 1PN (nonspinning)
    # ------------------------------------------------------------------
    et2 = (1.0
           - 2.0 * h0**2 * absE0
           + absE0 * (h0**2 * absE0 * (17.0 - 7.0 * eta)
                      + 4.0 * (eta - 1.0)))

    # ------------------------------------------------------------------
    # 1.5PN spin-orbit
    # ------------------------------------------------------------------
    et2 += (delta / h0) * absE0 * (
        2.0 * (eta - 2.0) * chi_s
        - 4.0 * sqrt_1m4eta * chi_a
    )

    # ------------------------------------------------------------------
    # 2PN spin-spin
    # ------------------------------------------------------------------
    # Auxiliary: kappa = (eta - 0.5)*lam_s - 0.5*sqrt(1-4eta)*lam_a
    kappa = (eta - 0.5) * lam_s - 0.5 * sqrt_1m4eta * lam_a

    et2 += (delta**2 * absE0 / h0**2) * (
        chi_a**2 * (kappa - eta)
        + chi_s**2 * (eta + kappa)
        + chi_s * chi_a * ((2.0 * eta - 1.0) * lam_a - sqrt_1m4eta * lam_s)
    )

    # ------------------------------------------------------------------
    # 2PN nonspinning
    # ------------------------------------------------------------------
    et2 += (absE0 / h0**2) * (
        -11.0 * eta + 17.0
        + h0**4 * absE0**2 * (-16.0 * eta**2 + 47.0 * eta - 112.0)
        + 12.0 * sqrt2 * h0**3 * absE0**1.5 * (5.0 - 2.0 * eta)
        + 2.0 * h0**2 * absE0 * (5.0 * eta**2 + eta + 2.0)
        + 6.0 * sqrt2 * h0 * np.sqrt(absE0) * (2.0 * eta - 5.0)
    )

    # ------------------------------------------------------------------
    # 2PN spin-orbit
    # ------------------------------------------------------------------
    et2 += (delta * absE0 / (2.0 * h0**3)) * (
        chi_s * (
            -16.0 * sqrt2 * h0**3 * absE0**1.5 * (eta**2 - 8.0 * eta + 6.0)
            + h0**2 * absE0 * (32.0 * eta**2 - 159.0 * eta + 124.0)
            + 8.0 * sqrt2 * h0 * np.sqrt(absE0) * (eta**2 - 8.0 * eta + 6.0)
            - 8.0 * eta**2 + 78.0 * eta - 64.0
        )
        + sqrt_1m4eta * chi_a * (
            32.0 * sqrt2 * h0**3 * absE0**1.5 * (eta - 3.0)
            + h0**2 * absE0 * (124.0 - 59.0 * eta)
            - 16.0 * sqrt2 * h0 * np.sqrt(absE0) * (eta - 3.0)
            + 18.0 * eta - 64.0
        )
    )

    return np.sqrt(et2)
