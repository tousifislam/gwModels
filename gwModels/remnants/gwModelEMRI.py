#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwModelEMRI.py
#
#    gwModelEMRI: point-particle remnant model for extreme-mass-ratio inspirals.
#    Unlike the other gwModelRem models, this contains no phenomenological
#    corrections fitted to NR. It evaluates the extreme-mass-ratio limit
#    directly from the conserved energy and angular momentum of a test particle
#    at the Kerr geodesic separatrix.
#
#    Intended for the genuine extreme-mass-ratio regime (q >> 1000), where
#    linear black hole perturbation theory describes the dynamics accurately.
#    Also serves as a reference implementation of the point-particle backbone
#    used by the calibrated models.
#
#    From Islam, Wadekar & Khanna (2026), https://arxiv.org/abs/2608.00934
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-02-2026
#    LAST MODIFIED: 08-02-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import warnings

import numpy as np

from .remnant_utils import validate_q, symmetric_mass_ratio
from .Kerr import separatrix_EL

# =============================================================================
# Model
#
#     Mf/M  = 1 - eta (1 - E_sep)                   + O(eta^2)
#     chi_f = chi + eta (L_sep - 2 chi (E_sep - 1))  + O(eta^2)
#
# Separatrix solver:
#   For a bound Kerr geodesic with spin a, eccentricity e and inclination
#   x = cos(theta_inc), the separatrix is where the radial potential R(r)
#   develops a double root at periastron r_p = p/(1+e). The solver imposes
#
#     eccentric (e > 0):  R(r_a) = 0, R(r_p) = 0, R'(r_p) = 0
#     circular  (e = 0):  R(r_p) = 0, R'(r_p) = 0, R''(r_p) = 0
#
#   on the three unknowns (p, E, L_z) by Newton iteration with an analytic
#   Jacobian solved by Cramer's rule. The Carter constant follows from
#
#     Q = (1 - x^2) [a^2 (1 - E^2) + L_z^2/x^2]
#
#   For equatorial circular orbits the solver reproduces the closed-form Kerr
#   ISCO energy and angular momentum to machine precision.
#
# Convergence:
#   Newton iteration is not globally convergent for this system, and where it
#   fails it can return a spurious low-energy root rather than diverging
#   visibly. Every entry point therefore reports convergence: separatrix_EL can
#   return a boolean mask, and gwModelEMRI warns by default when any system
#   fails and can return the mask. Do not use flagged results.
#
#   Measured on a 4800-point grid spanning chi in [-0.99, 0.99], theta_inc in
#   [0, pi] and e in {0, 0.3, 0.6}, the overall failure rate is 5.8% at
#   max_iter = 200 (7.3% at the default 20). Raising max_iter helps only
#   marginally, so these are spurious roots rather than slow convergence.
#
#     inclination band   failure rate      |chi| band   failure rate
#     0-15 deg               2.3%          0.00-0.30        3.1%
#     15-45 deg              3.5%          0.30-0.60        4.1%
#     45-75 deg              7.3%          0.60-0.80        4.8%
#     75-105 deg             9.2%          0.80-0.95       10.7%
#     105-135 deg            8.2%          0.95-1.00       20.8%
#     135-165 deg            3.6%
#     165-180 deg            3.8%
#
#   The worst region is near-polar inclination at high spin, where the Carter
#   constant term L_z^2/x^2 becomes stiff as x -> 0. Equatorial orbits are
#   unaffected: for both prograde and retrograde circular orbits the solver
#   converges everywhere and reproduces the closed-form Kerr ISCO to 4e-16 in
#   E and 5e-15 in L_z.
# =============================================================================


# =============================================================================
# Input handling
# =============================================================================

def _validate_emri_inputs(q, chi, theta_inc, e_sep):
    """
    Validate and broadcast the EMRI inputs.

    Parameters:
        q: Mass ratio m1/m2 >= 1.
        chi: Dimensionless spin of the primary Kerr BH, in [-1, 1].
        theta_inc: Orbital inclination in radians, in [0, pi].
        e_sep: Eccentricity at the separatrix, in [0, 1).

    Returns:
        tuple: Broadcast (q, chi, theta_inc, e_sep) arrays.
    """
    q = np.atleast_1d(validate_q(q))
    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    theta_inc = np.atleast_1d(np.asarray(theta_inc, dtype=float))
    e_sep = np.atleast_1d(np.asarray(e_sep, dtype=float))

    if np.any(np.abs(chi) > 1.0):
        raise ValueError("chi must be in [-1, 1].")
    if np.any(theta_inc < 0.0) or np.any(theta_inc > np.pi):
        raise ValueError("theta_inc must be in [0, pi].")
    if np.any(e_sep < 0.0) or np.any(e_sep >= 1.0):
        raise ValueError("e_sep must be in [0, 1).")

    return np.broadcast_arrays(q, chi, theta_inc, e_sep)


# =============================================================================
# The generic Kerr separatrix solver lives in Kerr.py and is re-exported here
# for backward compatibility, since it is the numerical core of this model.
# =============================================================================


# =============================================================================
# Remnant model
# =============================================================================

def gwModelEMRI(q, chi, theta_inc, e_sep, warn_unconverged=True,
                return_converged=False):
    """
    Point-particle remnant mass and spin for an extreme-mass-ratio inspiral.

    Evaluates the leading-order extreme-mass-ratio limit

        Mf/M  = 1 - eta (1 - E_sep)
        chi_f = chi + eta (L_sep - 2 chi (E_sep - 1))

    with E_sep and L_sep the conserved quantities of a test particle at the
    Kerr geodesic separatrix. No terms are fitted to NR data.

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array. Intended for q >> 1000.
        chi: Dimensionless spin of the primary Kerr BH, in [-1, 1].
        theta_inc: Orbital inclination in radians, in [0, pi]. 0 is prograde
            equatorial and pi retrograde equatorial.
        e_sep: Eccentricity at the separatrix, in [0, 1).
        warn_unconverged: Emit a RuntimeWarning if the solver fails anywhere.
        return_converged: If True, also return the boolean convergence mask.

    Returns:
        tuple: (Mf, chif) or (Mf, chif, converged) if return_converged is True,
            with Mf in units of the total mass and chif the signed final spin.

    Note:
        Results for systems where the solver has not converged are not
        physically meaningful. Check the convergence mask when scanning high
        spin at near-polar inclination; see the module header for rates.

    Example:
        >>> import gwModels
        >>> Mf, chif = gwModels.remnants.gwModelEMRI(1e4, 0.9, 0.0, 0.0)
    """
    q, chi, theta_inc, e_sep = _validate_emri_inputs(q, chi, theta_inc, e_sep)

    eta = symmetric_mass_ratio(q)
    x = np.cos(theta_inc)

    E_sep, Lz_sep, converged = separatrix_EL(chi, e_sep, x, return_converged=True)

    if warn_unconverged and not np.all(converged):
        n_bad = int((~converged).sum())
        warnings.warn(
            f"Kerr separatrix solver did not converge for {n_bad} of "
            f"{converged.size} systems; those results are unreliable. "
            f"Pass return_converged=True to identify them.",
            RuntimeWarning,
            stacklevel=2,
        )

    Mf = 1.0 - eta * (1.0 - E_sep)
    chif = chi + eta * (Lz_sep - 2.0 * chi * (E_sep - 1.0))

    if Mf.size == 1 and not return_converged:
        return Mf.item(), chif.item()
    if return_converged:
        return Mf, chif, converged
    return Mf, chif
