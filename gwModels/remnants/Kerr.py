#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: Kerr.py
#
#    Kerr geodesic quantities for test particles: the innermost stable circular
#    orbit (ISCO), the eccentric separatrix, and the conserved energy and axial
#    angular momentum at each.
#
#    These are exact properties of Kerr spacetime, independent of any remnant
#    fit. The remnant models in this package use them as point-particle
#    anchors, and they are also useful on their own for BHPT and EMRI work.
#
#    Used by the gwModelRem family of
#    Islam, Wadekar & Khanna (2026), https://arxiv.org/abs/2608.00934
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-03-2026
#    LAST MODIFIED: 08-03-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

# =============================================================================
# Conventions
#
# The spin chi (equivalently a) is dimensionless and signed: positive for
# prograde orbits, negative for retrograde. Radii are in units of the black
# hole mass, energies are specific (per unit test-particle mass), and angular
# momenta are specific and returned as magnitudes.
#
# Reference: Bardeen, Press & Teukolsky (1972), ApJ 178, 347.
# =============================================================================

# Default clipping limit for spins entering the ISCO expressions. See
# clip_spin below for why this exists.
SPIN_CLIP = 0.9999


def clip_spin(chi, limit=SPIN_CLIP):
    """
    Clip a dimensionless spin away from the extremal values +/-1.

    At |chi| = 1 the Kerr ISCO degenerates: r_ISCO -> 1 for prograde orbits,
    and the angular-momentum denominator sqrt(r^1.5 - 3 sqrt(r) + 2 chi)
    vanishes, so kerr_isco_angular_momentum divides by zero. Effective spin
    combinations such as chi_hat reach exactly +/-1 when both component spins
    are extremal, so calibrated fits clip before evaluating the ISCO.

    This is a fit convention, not physics, and is therefore applied explicitly
    by the models rather than being folded into the functions below.

    Parameters:
        chi: Dimensionless spin, scalar or array.
        limit: Clipping magnitude (default SPIN_CLIP = 0.9999).

    Returns:
        float or array: Spin clipped to [-limit, limit].
    """
    return np.clip(np.asarray(chi, dtype=float), -limit, limit)


# =============================================================================
# Innermost stable circular orbit
# =============================================================================

def kerr_isco_radius(a):
    """
    Boyer-Lindquist r_ISCO(a) for equatorial orbits of a Kerr black hole.

        r_ISCO = 3 + Z2 - sign(a) sqrt((3 - Z1)(3 + Z1 + 2 Z2))
        Z1     = 1 + (1 - a^2)^(1/3) [(1 + a)^(1/3) + (1 - a)^(1/3)]
        Z2     = sqrt(3 a^2 + Z1^2)

    Limits: r_ISCO(0) = 6, r_ISCO(1) = 1, r_ISCO(-1) = 9.

    Parameters:
        a: Dimensionless spin parameter (|a| <= 1). Positive for prograde,
           negative for retrograde orbits.

    Returns:
        float or array: ISCO radius in units of GM/c^2.
    """
    a = np.asarray(a, dtype=float)
    Z1 = 1.0 + (1 - a ** 2) ** (1 / 3.0) * ((1 + a) ** (1 / 3.0) + (1 - a) ** (1 / 3.0))
    Z2 = np.sqrt(3 * a ** 2 + Z1 ** 2)
    r_isco = 3 + Z2 - np.sign(a) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    return r_isco


def kerr_isco_energy(a):
    """
    Specific energy of a test particle at the Kerr ISCO.

        E_ISCO = sqrt(1 - 2/(3 r_ISCO))

    Limits: E_ISCO(0) = sqrt(8/9) = 0.942809, E_ISCO(1) = 1/sqrt(3) = 0.577350.

    Parameters:
        a: Dimensionless spin parameter (|a| <= 1).

    Returns:
        float or array: Specific energy at the ISCO.
    """
    return np.sqrt(1.0 - 2.0 / (3.0 * kerr_isco_radius(a)))


def kerr_isco_angular_momentum(a):
    """
    Specific axial angular momentum magnitude at the Kerr ISCO.

        L_ISCO = (r^2 - 2 d a sqrt(r) + a^2)
                 / (r^(3/4) sqrt(r^(3/2) - 3 sqrt(r) + 2 d a))

    with r = r_ISCO(a) and d = sign(a). Limits: L_ISCO(0) = 2 sqrt(3) = 3.464102,
    L_ISCO(1) = 2/sqrt(3) = 1.154701, L_ISCO(-1) = 22/(3 sqrt(3)) = 4.232809.

    Note that the denominator vanishes at |a| = 1, where this expression is a
    removable singularity: evaluating at exactly +/-1 divides by zero. Pass the
    spin through clip_spin first if extremal values are possible. An equivalent
    form regular at the endpoints is (2/(3 sqrt(3)))(1 + 2 sqrt(3 r - 2)); the
    two agree to about 1e-12 and this one is retained because the calibrated
    remnant fits were built on it.

    Parameters:
        a: Dimensionless spin parameter (|a| <= 1).

    Returns:
        float or array: Specific angular momentum magnitude at the ISCO.
    """
    chi = np.asarray(a, dtype=float)
    mag = np.abs(chi)
    d = np.sign(chi)
    r = kerr_isco_radius(chi)
    rh = np.sqrt(r)
    return (r**2 - d * 2.0 * mag * rh + mag**2) / (
        r**0.75 * np.sqrt(r**1.5 - 3.0 * rh + d * 2.0 * mag))


def kerr_ell(a):
    """
    Effective angular momentum combination at the Kerr ISCO.

        ell_Kerr = L_ISCO - 2 a (E_ISCO - 1)

    This is the combination that enters remnant-spin fits anchored to the
    point-particle limit, following Hofmann, Barausse & Rezzolla (2016).

    Parameters:
        a: Dimensionless spin parameter (|a| <= 1).

    Returns:
        float or array: Effective angular momentum at the ISCO.
    """
    return kerr_isco_angular_momentum(a) - 2.0 * a * (kerr_isco_energy(a) - 1.0)


# =============================================================================
# Eccentric separatrix: closed-form Schwarzschild limit with a Kerr extension
#
# The separatrix is the boundary of stable bound motion, where the radial
# potential develops a double root at periastron. For eccentric orbits the
# plunge begins there rather than at the ISCO.
# =============================================================================

_E_ISCO_SCHW = np.sqrt(8.0 / 9.0)
_L_ISCO_SCHW = 2.0 * np.sqrt(3.0)


def separatrix_energy(e_s, chi):
    """
    Specific energy at the separatrix, with a phenomenological Kerr extension.

        E_sep = sqrt(8/(9 - e_s^2)) + [E_ISCO(chi) - E_ISCO(0)] 9/(9 - e_s^2)

    The first term is the exact Schwarzschild separatrix (p_sep = 6 + 2 e_s);
    the second interpolates toward the Kerr ISCO. Reduces to E_ISCO(chi) in the
    circular limit e_s -> 0.

    Parameters:
        e_s: Eccentricity at the separatrix, in [0, 1).
        chi: Dimensionless spin parameter.

    Returns:
        float or array: Specific energy at the separatrix.
    """
    e2 = np.asarray(e_s, dtype=float) ** 2
    return (np.sqrt(8.0 / (9.0 - e2))
            + (kerr_isco_energy(chi) - _E_ISCO_SCHW) * 9.0 / (9.0 - e2))


def separatrix_angular_momentum(e_s, chi):
    """
    Specific angular momentum at the separatrix, with a Kerr extension.

        L_sep = (6 + 2 e_s)/sqrt(3 + 2 e_s - e_s^2)
                + [L_ISCO(chi) - L_ISCO(0)] sqrt(3)/sqrt((3 - e_s)(1 + e_s))

    Reduces to L_ISCO(chi) in the circular limit e_s -> 0.

    Parameters:
        e_s: Eccentricity at the separatrix, in [0, 1).
        chi: Dimensionless spin parameter.

    Returns:
        float or array: Specific angular momentum at the separatrix.
    """
    e_s = np.asarray(e_s, dtype=float)
    d = np.sqrt((3.0 - e_s) * (1.0 + e_s))
    return (2.0 * (3.0 + e_s) / d
            + (kerr_isco_angular_momentum(chi) - _L_ISCO_SCHW) * np.sqrt(3.0) / d)


def separatrix_ell(e_s, chi):
    """
    Effective angular momentum at the separatrix.

        ell_sep = L_sep - 2 chi (E_sep - 1)

    Reduces to kerr_ell(chi) in the circular limit e_s -> 0.

    Parameters:
        e_s: Eccentricity at the separatrix, in [0, 1).
        chi: Dimensionless spin parameter.

    Returns:
        float or array: Effective angular momentum at the separatrix.
    """
    return (separatrix_angular_momentum(e_s, chi)
            - 2.0 * chi * (separatrix_energy(e_s, chi) - 1.0))


# =============================================================================
# Generic Kerr separatrix solver (inclined and eccentric)
#
# For a bound Kerr geodesic with spin a, eccentricity e and inclination
# x = cos(theta_inc), the separatrix is where the radial potential R(r) develops
# a double root at periastron r_p = p/(1+e). The solver imposes
#
#     eccentric (e > 0):  R(r_a) = 0, R(r_p) = 0, R'(r_p) = 0
#     circular  (e = 0):  R(r_p) = 0, R'(r_p) = 0, R''(r_p) = 0
#
# on the unknowns (p, E, L_z) by Newton iteration with an analytic Jacobian
# solved by Cramer's rule. The Carter constant follows from
#
#     Q = (1 - x^2) [a^2 (1 - E^2) + L_z^2/x^2]
#
# For equatorial circular orbits this reproduces the closed forms above to
# machine precision.
#
# Convergence: Newton iteration is not globally convergent here, and where it
# fails it can return a spurious low-energy root rather than diverging visibly.
# Always check the convergence mask. Measured on a 4800-point grid spanning
# chi in [-0.99, 0.99], theta_inc in [0, pi] and e in {0, 0.3, 0.6}, the overall
# failure rate is 5.8% at max_iter = 200 (7.3% at the default 20); raising
# max_iter helps only marginally. Failures concentrate near polar inclination
# (9.2% for 75-105 deg) and at high spin (20.8% for |chi| > 0.95). Equatorial
# orbits always converge.
# =============================================================================

def separatrix_EL(a, e, x, max_iter=20, tol=1e-12, return_converged=False):
    """
    Energy and axial angular momentum at the generic Kerr separatrix.

    Parameters:
        a: Dimensionless Kerr spin, in [-1, 1].
        e: Orbital eccentricity, in [0, 1). Values below 1e-8 are treated as
           circular and use the R = R' = R'' = 0 branch.
        x: Cosine of the orbital inclination, in [-1, 1]. x = 1 is prograde
           equatorial and x = -1 retrograde equatorial.
        max_iter: Maximum Newton iterations.
        tol: Convergence tolerance on the residuals of the defining conditions.
        return_converged: If True, also return a boolean convergence mask.

    Returns:
        tuple: (E, Lz) or (E, Lz, converged) if return_converged is True.

    Note:
        Check the convergence mask before using results, particularly at high
        spin combined with near-polar inclination. See the module header.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    e = np.atleast_1d(np.asarray(e, dtype=float))
    x = np.atleast_1d(np.asarray(x, dtype=float))

    a, e, x = np.broadcast_arrays(a, e, x)
    a = np.array(a, dtype=float)
    e = np.array(e, dtype=float)
    x = np.array(x, dtype=float)

    is_circ = e < 1e-8
    e_safe = np.where(is_circ, 1e-8, e)
    is_eq = np.abs(1 - np.abs(x)) < 1e-10
    zm2 = np.where(is_eq, 0.0, 1.0 - x**2)
    x2 = np.where(is_eq, 1.0, x**2)
    sgn = np.where(x >= 0, 1.0, -1.0)

    # Initial guess from the Kerr circular orbit near the expected separatrix
    a_eff = a * sgn
    r_isco = kerr_isco_radius(clip_spin(a_eff))

    p = np.where(np.abs(a) > 0.01,
                 np.maximum(r_isco * (1 + 0.35 * e), 1 + e + 0.5),
                 6.0 + 2.0 * e)

    r0 = np.maximum(p, 1.5)
    denom = np.abs(1 - 3 / r0 + 2 * a_eff / r0**1.5)
    good = denom > 1e-6
    E = np.where(good,
                 (1 - 2 / r0 + a_eff / r0**1.5) / np.sqrt(np.where(good, denom, 1.0)),
                 np.sqrt(np.clip(8.0 / (9.0 - e**2), 0.01, 0.999)))
    E = np.clip(E, 0.1, 0.9999)

    sr = np.sqrt(r0)
    denom2 = np.abs(r0**1.5 - 3 * sr + 2 * a_eff)
    good2 = denom2 > 1e-6
    Lz = np.where(good2,
                  sgn * (r0**2 - 2 * a_eff * sr + a**2)
                  / (r0**0.75 * np.sqrt(np.where(good2, denom2, 1.0))),
                  sgn * 2 * (3 + e) / np.sqrt(np.clip((3 - e) * (1 + e), 0.01, None)))

    err = np.full(a.shape, np.inf)

    for _ in range(max_iter):
        rp = p / (1 + e)
        ra = p / (1 - e_safe)

        # Carter constant and its derivatives
        Q = zm2 * (a**2 * (1 - E**2) + Lz**2 / x2)
        dQ_dE = -2 * a**2 * E * zm2
        dQ_dLz = 2 * Lz * zm2 / x2

        K = (Lz - a * E) ** 2 + Q
        dK_dE = -2 * a * (Lz - a * E) + dQ_dE
        dK_dLz = 2 * (Lz - a * E) + dQ_dLz

        # Radial potential at apastron (periastron for circular orbits)
        r1 = np.where(is_circ, p, ra)
        D1 = r1**2 - 2 * r1 + a**2
        A1 = r1**2 + a**2
        P1 = E * A1 - a * Lz
        R1 = P1**2 - D1 * (r1**2 + K)
        dR1 = 4 * E * r1 * P1 - (2 * r1 - 2) * (r1**2 + K) - 2 * r1 * D1

        # Radial potential at periastron
        D2 = rp**2 - 2 * rp + a**2
        A2 = rp**2 + a**2
        P2 = E * A2 - a * Lz
        R2 = P2**2 - D2 * (rp**2 + K)
        dR2 = 4 * E * rp * P2 - (2 * rp - 2) * (rp**2 + K) - 2 * rp * D2
        d2R2 = (8 * E**2 * rp**2 + 4 * E * P2 - 2 * (rp**2 + K)
                - 4 * rp * (2 * rp - 2) - 2 * D2)

        f1 = np.where(is_circ, R2, R1)
        f2 = np.where(is_circ, dR2, R2)
        f3 = np.where(is_circ, d2R2, dR2)

        err = np.maximum(np.maximum(np.abs(f1), np.abs(f2)), np.abs(f3))
        if np.all(err < tol):
            break

        # Analytic Jacobian
        dR_dE_1 = 2 * P1 * A1 - D1 * dK_dE
        dR_dE_2 = 2 * P2 * A2 - D2 * dK_dE
        dR_dLz_1 = -2 * a * P1 - D1 * dK_dLz
        dR_dLz_2 = -2 * a * P2 - D2 * dK_dLz
        ddR_dE = 4 * rp * P2 + 4 * E * rp * A2 - (2 * rp - 2) * dK_dE
        ddR_dLz = -4 * a * E * rp - (2 * rp - 2) * dK_dLz
        d2dR_dE = 16 * E * rp**2 + 4 * P2 + 4 * E * A2 - 2 * dK_dE
        d2dR_dLz = -4 * a * E - 2 * dK_dLz
        d3R2 = 24 * (E**2 - 1) * rp + 12

        j00 = np.where(is_circ, dR2, dR1 / (1 - e_safe))
        j01 = np.where(is_circ, dR_dE_2, dR_dE_1)
        j02 = np.where(is_circ, dR_dLz_2, dR_dLz_1)
        j10 = np.where(is_circ, d2R2, dR2 / (1 + e))
        j11 = np.where(is_circ, ddR_dE, dR_dE_2)
        j12 = np.where(is_circ, ddR_dLz, dR_dLz_2)
        j20 = np.where(is_circ, d3R2, d2R2 / (1 + e))
        j21 = np.where(is_circ, d2dR_dE, ddR_dE)
        j22 = np.where(is_circ, d2dR_dLz, ddR_dLz)

        # Cramer's rule: J @ [dp, dE, dLz] = -[f1, f2, f3]
        c00 = j11 * j22 - j12 * j21
        c01 = j10 * j22 - j12 * j20
        c02 = j10 * j21 - j11 * j20
        det = j00 * c00 - j01 * c01 + j02 * c02
        inv_det = 1.0 / np.where(np.abs(det) < 1e-30, 1e-30, det)

        b0, b1, b2 = -f1, -f2, -f3
        m12 = b1 * j22 - j12 * b2
        m02 = b1 * j21 - j11 * b2
        m01 = j10 * b2 - b1 * j20

        dp = (b0 * c00 - j01 * m12 + j02 * m02) * inv_det
        dE = (j00 * m12 - b0 * c01 + j02 * m01) * inv_det
        dLz = (j00 * (j11 * b2 - b1 * j21) - j01 * m01 + b0 * c02) * inv_det

        # Freeze already-converged systems
        conv = err < tol
        dp = np.where(conv, 0.0, dp)
        dE = np.where(conv, 0.0, dE)
        dLz = np.where(conv, 0.0, dLz)

        # Damp large steps
        scale = np.minimum(
            np.where(np.abs(dE) > 0.2, 0.2 / np.maximum(np.abs(dE), 1e-30), 1.0),
            np.where(np.abs(dp) > 2.0, 2.0 / np.maximum(np.abs(dp), 1e-30), 1.0))

        p += dp * scale
        E += dE * scale
        Lz += dLz * scale

        p = np.maximum(p, 1.0 + e + 0.01)
        E = np.clip(E, 0.01, 0.9999)

    converged = err < tol

    if return_converged:
        return E, Lz, converged
    return E, Lz
