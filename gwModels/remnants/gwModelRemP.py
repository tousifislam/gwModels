#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwModelRemP.py
#
#    gwModelRemP: remnant properties of precessing quasi-circular BBH mergers.
#    Augments the aligned-spin baseline gwModelRemS with corrections built from
#    the in-plane spin components at the reference separation r = 8M.
#
#    Calibrated on 4458 NR + BHPT simulations for mass and spin, and 4062
#    precessing simulations for the peak luminosity.
#
#    From Islam, Wadekar & Khanna (2026), https://arxiv.org/abs/2608.00934
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-02-2026
#    LAST MODIFIED: 08-02-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

from .remnant_utils import (validate_q, validate_spin_magnitudes,
                            symmetric_mass_ratio)
from .gwModelRemS import gwModelRemS_mf, gwModelRemS_chif, gwModelRemS_Lpeak

# =============================================================================
# Model structure
#
# The aligned-spin models are evaluated at the z-projections of the spins at
# r = 8M, then augmented by terms that vanish as the in-plane spin goes to zero:
#
#     Mf      = mf_al + C_m S_perp^2 + C_dM Delta_perp^2
#     |a_f|   = min(sqrt(chif_al^2 + C_a S_perp^2 + C_da Delta_perp^2), 1)
#     a_fz    = chif_al + C_theta S_perp^2
#     theta_f = arccos(a_fz / |a_f|)
#     L_peak  = Lp_al exp(g(S_perp, eta, chi_hat))
#
# In-plane spin variables at r = 8M:
#     chi_i_perp = a_i sin(theta_i)
#     S_perp     = sqrt(q^4 chi1_perp^2 + chi2_perp^2) / (q^2 + 1)
#     Delta_perp = (q chi1_perp - chi2_perp) / (1 + q)
#
# Every augmentation vanishes at S_perp = Delta_perp = 0, so the model reduces
# exactly to gwModelRemS in the non-precessing limit. The spin magnitude is
# capped at the Kerr limit and the mass at Mf <= 1; see gwModelRemP for why the
# mass cap is applied to the output rather than inside the augmentation.
#
# The spin-direction and peak-luminosity fits omit Delta_perp terms, which do
# not improve their validation errors.
#
# Spin inputs are specified at r = 8M. Reference spins are evolved there using
# precession-averaged PN inspiral evolution; this module takes them as given.
# =============================================================================

_Q_MAX = 10000.0


# =============================================================================
# Input handling
# =============================================================================

def _validate_prec_inputs(q, a1, a2, theta1, theta2, phi1, phi2):
    """
    Validate and broadcast the precessing-spin inputs.

    Parameters:
        q: Mass ratio m1/m2 >= 1 (upper bound 10000).
        a1, a2: Spin magnitudes at r = 8M, in [0, 1].
        theta1, theta2: Spin tilt angles at r = 8M in radians, in [0, pi].
        phi1, phi2: Spin azimuthal angles at r = 8M in radians (unbounded).

    Returns:
        tuple: Broadcast (q, a1, a2, theta1, theta2, phi1, phi2) arrays.
    """
    q = np.atleast_1d(validate_q(q))
    a1, a2 = validate_spin_magnitudes(a1, a2)
    a1 = np.atleast_1d(a1)
    a2 = np.atleast_1d(a2)
    theta1 = np.atleast_1d(np.asarray(theta1, dtype=float))
    theta2 = np.atleast_1d(np.asarray(theta2, dtype=float))
    phi1 = np.atleast_1d(np.asarray(phi1, dtype=float))
    phi2 = np.atleast_1d(np.asarray(phi2, dtype=float))

    if np.any(q > _Q_MAX):
        raise ValueError(f"q must be <= {_Q_MAX:.0f}, got maximum {q.max()}.")
    if np.any(theta1 < 0.0) or np.any(theta1 > np.pi):
        raise ValueError("theta1 must be in [0, pi].")
    if np.any(theta2 < 0.0) or np.any(theta2 > np.pi):
        raise ValueError("theta2 must be in [0, pi].")

    return np.broadcast_arrays(q, a1, a2, theta1, theta2, phi1, phi2)


def spin_projections(q, a1, a2, theta1, theta2):
    """
    Aligned projections and in-plane spin combinations at r = 8M.

    Parameters:
        q: Mass ratio m1/m2 >= 1.
        a1, a2: Spin magnitudes at r = 8M.
        theta1, theta2: Spin tilt angles at r = 8M in radians.

    Returns:
        tuple: (chi1z, chi2z, S_perp, Delta_perp).
    """
    chi1z = a1 * np.cos(theta1)
    chi2z = a2 * np.cos(theta2)

    chi1_perp = a1 * np.sin(theta1)
    chi2_perp = a2 * np.sin(theta2)

    S_perp = np.sqrt(q**4 * chi1_perp**2 + chi2_perp**2) / (q**2 + 1.0)
    Delta_perp = (q * chi1_perp - chi2_perp) / (1.0 + q)

    return chi1z, chi2z, S_perp, Delta_perp


# =============================================================================
# Mass augmentation (8 parameters)
# =============================================================================

_MASS_AUG_PARAMS = {
    'a0':  1.42190900e-02,
    'a1': -1.57484090e-01,
    'a2':  4.81225920e-01,
    'a3':  1.18185000e-02,
    'a4': -1.14662040e-01,
    'b0': -1.11779900e-02,
    'b1':  4.75143300e-02,
    'b2': -7.59294000e-03,
}
_MASS_AUG_STDERR = {
    'a0': 2.25339000e-03,
    'a1': 2.00354800e-02,
    'a2': 5.04146200e-02,
    'a3': 3.04468000e-03,
    'a4': 1.44621200e-02,
    'b0': 1.96645000e-03,
    'b1': 9.88860000e-03,
    'b2': 1.58626000e-03,
}


def _mass_aug(mf_al, chif_al, S_perp, Delta_perp, eta):
    """
    Mass augmented by S_perp^2 and Delta_perp^2 corrections, uncapped.

    This is the raw fitted augmentation. It is NOT capped at the physical bound
    Mf <= 1, because it is also the context feature the recoil flow was trained
    on; capping here would shift the flow's conditioning relative to training.
    The cap is applied to the public model output in gwModelRemP instead.
    """
    p = _MASS_AUG_PARAMS
    C_m = (p['a0'] + p['a1'] * eta + p['a2'] * eta**2
           + p['a3'] * chif_al**2 + p['a4'] * eta * chif_al**2)
    C_Delta = p['b0'] + p['b1'] * eta + p['b2'] * chif_al
    return mf_al + C_m * S_perp**2 + C_Delta * Delta_perp**2


# =============================================================================
# Spin magnitude augmentation (7 parameters)
# =============================================================================

_SPIN_AUG_PARAMS = {
    'a0':  7.27519690e-01,
    'a1': -2.83411948e+00,
    'a2':  9.50142840e-01,
    'a3': -1.04717200e-01,
    'b0':  2.50739670e-01,
    'b1': -8.33393990e-01,
    'b2':  7.70934000e-02,
}
_SPIN_AUG_STDERR = {
    'a0': 3.38511200e-02,
    'a1': 3.00328820e-01,
    'a2': 6.93733880e-01,
    'a3': 1.33304300e-02,
    'b0': 2.93943400e-02,
    'b1': 1.51444000e-01,
    'b2': 1.68557300e-02,
}


def _spin_magnitude_aug(chif_al, S_perp, Delta_perp, eta):
    """Spin magnitude augmented by S_perp^2 and Delta_perp^2, capped at 1."""
    p = _SPIN_AUG_PARAMS
    C_a = p['a0'] + p['a1'] * eta + p['a2'] * eta**2 + p['a3'] * chif_al**2
    C_Delta = p['b0'] + p['b1'] * eta + p['b2'] * chif_al

    af_mag = np.sqrt(np.maximum(
        chif_al**2 + C_a * S_perp**2 + C_Delta * Delta_perp**2, 0.0))
    return np.minimum(af_mag, 1.0)


# =============================================================================
# Spin direction augmentation (4 parameters)
# =============================================================================

_THETA_AUG_PARAMS = {
    'a0':  1.54096660e-01,
    'a1':  1.31160201e+00,
    'a2': -3.81403834e+00,
    'a3': -3.98385740e-01,
}
_THETA_AUG_STDERR = {
    'a0': 1.42597900e-02,
    'a1': 1.84080300e-01,
    'a2': 5.29204000e-01,
    'a3': 9.44791000e-03,
}


def _spin_direction_aug(chif_al, S_perp, eta, af_mag):
    """Tilt angle of the remnant spin from the orbital angular momentum."""
    p = _THETA_AUG_PARAMS
    C_theta = p['a0'] + p['a1'] * eta + p['a2'] * eta**2 + p['a3'] * chif_al

    af_z = chif_al + C_theta * S_perp**2
    cos_theta = np.clip(af_z / np.maximum(af_mag, 1e-10), -1.0, 1.0)
    return np.arccos(cos_theta)


# =============================================================================
# Peak luminosity augmentation (5 parameters)
#
# g = S_perp^2 (b0 + b1 chi_hat + b2 (1-4 eta) + b3 chi_hat^2) + b4 S_perp^4
#
# Fitted on 4062 precessing simulations with a quality cut of
# L_peak^NR / L_peak^aligned < 5.
# =============================================================================

_LUMI_AUG_PARAMS = {
    'b0': -1.73406160e-01,
    'b1':  9.50715000e-03,
    'b2': -1.97774310e-01,
    'b3':  1.22131754e+00,
    'b4':  1.23969285e+00,
}
_LUMI_AUG_STDERR = {
    'b0': 3.16113200e-02,
    'b1': 3.37845700e-02,
    'b2': 5.17244100e-02,
    'b3': 1.13559720e-01,
    'b4': 7.44142800e-02,
}


def _luminosity_aug(Lp_al, S_perp, eta, chi_hat):
    """
    Peak luminosity augmented by an exponential in-plane-spin correction.

    The exponential form guarantees L_peak > 0 and reduces to a linear
    correction for small S_perp.
    """
    b = _LUMI_AUG_PARAMS
    delta = 1.0 - 4.0 * eta
    Sp2 = S_perp**2

    g = (Sp2 * (b['b0'] + b['b1'] * chi_hat + b['b2'] * delta
                + b['b3'] * chi_hat**2)
         + b['b4'] * Sp2**2)
    return Lp_al * np.exp(g)


# =============================================================================
# Combined interface
# =============================================================================

def gwModelRemP(q, a1, a2, theta1, theta2, phi1, phi2):
    """
    gwModelRemP remnant properties for a precessing quasi-circular BBH merger.

    The aligned-spin baseline is evaluated at the z-projections of the spins at
    r = 8M, then augmented with in-plane-spin corrections.

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        a1: Spin magnitude of the heavier BH at r = 8M, in [0, 1].
        a2: Spin magnitude of the lighter BH at r = 8M, in [0, 1].
        theta1: Tilt angle of the heavier BH spin at r = 8M in radians, [0, pi].
        theta2: Tilt angle of the lighter BH spin at r = 8M in radians, [0, pi].
        phi1, phi2: Azimuthal spin angles at r = 8M in radians. Accepted for
            interface completeness; the deterministic models do not use them.

    Returns:
        tuple: (Mf, af_mag, theta_f, Lpeak) where

            Mf: final mass in units of the total mass
            af_mag: final spin magnitude, in [0, 1]
            theta_f: final spin tilt from the orbital angular momentum, radians
            Lpeak: peak GW luminosity in geometric units (c^5/G)

    Note:
        The recoil of a precessing binary is not modeled deterministically.
        Use gwModelRemP_flow for the recoil velocity distribution.

    Example:
        >>> import numpy as np, gwModels
        >>> Mf, af, thf, Lp = gwModels.remnants.gwModelRemP(
        ...     2.0, 0.7, 0.3, np.pi/3, np.pi/4, 0.0, 0.0)
    """
    q, a1, a2, theta1, theta2, phi1, phi2 = _validate_prec_inputs(
        q, a1, a2, theta1, theta2, phi1, phi2)

    chi1z, chi2z, S_perp, Delta_perp = spin_projections(
        q, a1, a2, theta1, theta2)

    eta = symmetric_mass_ratio(q)
    chi_hat = (q**2 * chi1z + chi2z) / (q**2 + 1.0)

    mf_al = np.atleast_1d(np.asarray(gwModelRemS_mf(q, chi1z, chi2z), dtype=float))
    chif_al = np.atleast_1d(np.asarray(gwModelRemS_chif(q, chi1z, chi2z), dtype=float))
    Lp_al = np.atleast_1d(np.asarray(gwModelRemS_Lpeak(q, chi1z, chi2z), dtype=float))

    af_mag = _spin_magnitude_aug(chif_al, S_perp, Delta_perp, eta)
    Mf = _mass_aug(mf_al, chif_al, S_perp, Delta_perp, eta)
    theta_f = _spin_direction_aug(chif_al, S_perp, eta, af_mag)
    Lpeak = _luminosity_aug(Lp_al, S_perp, eta, chi_hat)

    # Enforce the physical bound Mf <= 1. The mass augmentation is additive and
    # carries no eta suppression, so at small eta it can push the prediction
    # above the aligned-spin baseline, which itself tends to 1 as eta -> 0.
    # Uncapped, the raw fit returns Mf > 1 for roughly 19% of uniformly sampled
    # configurations, rising to 57% for q > 200 and 3.6% even inside the
    # calibration domain (q <= 100, S_perp <= 0.93). The excess is small,
    # median 6.7e-4 and at most 7.3e-3, so the cap changes nothing where the
    # model is well calibrated. This mirrors the Kerr cap on the spin magnitude.
    Mf = np.minimum(Mf, 1.0)

    if Mf.size == 1:
        return Mf.item(), af_mag.item(), theta_f.item(), Lpeak.item()
    return Mf, af_mag, theta_f, Lpeak


def gwModelRemP_mf(q, a1, a2, theta1, theta2, phi1, phi2):
    """
    Final mass Mf/M of a precessing quasi-circular BBH merger.

    See gwModelRemP for the parameter description.

    Returns:
        float or array: Final mass in units of the total mass.
    """
    return gwModelRemP(q, a1, a2, theta1, theta2, phi1, phi2)[0]


def gwModelRemP_chif(q, a1, a2, theta1, theta2, phi1, phi2):
    """
    Final spin magnitude and tilt of a precessing quasi-circular BBH merger.

    See gwModelRemP for the parameter description.

    Returns:
        tuple: (af_mag, theta_f) with the tilt in radians.
    """
    _, af_mag, theta_f, _ = gwModelRemP(q, a1, a2, theta1, theta2, phi1, phi2)
    return af_mag, theta_f


def gwModelRemP_Lpeak(q, a1, a2, theta1, theta2, phi1, phi2):
    """
    Peak GW luminosity of a precessing quasi-circular BBH merger.

    See gwModelRemP for the parameter description.

    Returns:
        float or array: Peak luminosity in geometric units (c^5/G).
    """
    return gwModelRemP(q, a1, a2, theta1, theta2, phi1, phi2)[3]
