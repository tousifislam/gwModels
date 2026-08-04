#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwModelRemPE.py
#
#    gwModelRemPE: remnant properties of eccentric precessing BBH mergers.
#    Applies the gwModelRemSE eccentric corrections on top of the precessing
#    quasi-circular baseline gwModelRemP, with spins at r = 8M and the
#    eccentricity and mean anomaly at t = -2500M.
#
#    This model is provisional and inherits the gwModelRemSE limitations. See
#    the note below, and do not extrapolate past e_ref ~ 0.3.
#
#    From Islam, Wadekar & Khanna (2026), https://arxiv.org/abs/2608.00934
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-03-2026
#    LAST MODIFIED: 08-03-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

from .remnant_utils import symmetric_mass_ratio
from .gwModelRemP import gwModelRemP
from .gwModelRemSE import (
    _ECC_MASS_PARAMS,
    _ECC_SPIN_PARAMS,
    _ECC_LUMI_PARAMS,
    _ecc_factor,
    _neg_sin,
    _validate_ecc_inputs,
)

# =============================================================================
# Construction
#
#     Mf      = Mf_P      [1 + delta_M(e_ref, l_ref, eta)]
#     |a_f|   = |a_f|_P   [1 + delta_chi(e_ref, l_ref, eta)]
#     L_peak  = L_peak_P  [1 + delta_L(e_ref, l_ref, eta)]
#     theta_f = theta_f_P                       (no eccentric correction)
#
# The correction factors are exactly those of gwModelRemSE, reusing its 28
# calibrated coefficients unchanged; only the baseline differs. Treating
# eccentricity and precession as independent at leading order is the intended
# construction, though the factorization has not been checked against
# precessing eccentric NR.
#
# theta_f carries no eccentric correction because none was calibrated for the
# remnant spin direction. The recoil of a precessing binary is not modeled
# deterministically here; use gwModelRemP_flow, optionally rescaling its
# samples by the gwModelRemSE recoil factor.
#
# Limits, exact by construction:
#     e_ref -> 0                    reduces to gwModelRemP
#     S_perp, Delta_perp -> 0       reduces to gwModelRemSE
#     both -> 0                     reduces to gwModelRemS
#
# Known limitation: gwModelRemSE does not yet improve on its quasi-circular
# baseline on NR data (neutral to a few percent worse inside its calibration
# domain of q <= 4, e0 <= 0.25) and degrades sharply beyond e_ref ~ 0.3, where
# the anomaly modulation alpha_X e_ref becomes large. Those caveats carry over
# here in full. See gwModelRemSE for the measured numbers.
# =============================================================================


def gwModelRemPE(q, a1, a2, theta1, theta2, phi1, phi2, e_ref, l_ref):
    """
    gwModelRemPE remnant properties for an eccentric precessing BBH merger.

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        a1, a2: Spin magnitudes at r = 8M, in [0, 1].
        theta1, theta2: Spin tilt angles at r = 8M in radians, in [0, pi].
        phi1, phi2: Azimuthal spin angles at r = 8M in radians. Accepted for
            interface completeness; the deterministic models do not use them.
        e_ref: Eccentricity at t = -2500M, in [0, 1).
        l_ref: Mean anomaly at t = -2500M in radians.

    Returns:
        tuple: (Mf, af_mag, theta_f, Lpeak) where

            Mf: final mass in units of the total mass
            af_mag: final spin magnitude, in [0, 1]
            theta_f: final spin tilt from the orbital angular momentum, radians
            Lpeak: peak GW luminosity in geometric units (c^5/G)

    Note:
        The eccentric corrections were calibrated on non-spinning aligned-spin
        systems and are applied here to a precessing baseline, treating the two
        effects as independent at leading order. theta_f carries no eccentric
        correction. Use gwModelRemP_flow for the recoil.

    Example:
        >>> import numpy as np, gwModels
        >>> Mf, af, thf, Lp = gwModels.remnants.gwModelRemPE(
        ...     2.0, 0.7, 0.3, np.pi/3, np.pi/4, 0.0, 0.0, 0.1, 0.0)
    """
    # Validate the eccentric arguments against a dummy aligned spin pair; the
    # precessing arguments are validated inside gwModelRemP.
    e_ref = np.atleast_1d(np.asarray(e_ref, dtype=float))
    l_ref = np.atleast_1d(np.asarray(l_ref, dtype=float))
    if np.any(e_ref < 0.0) or np.any(e_ref >= 1.0):
        raise ValueError("e_ref must be in [0, 1).")

    Mf_p, af_p, theta_f, Lp_p = gwModelRemP(q, a1, a2, theta1, theta2, phi1, phi2)

    Mf_p = np.atleast_1d(np.asarray(Mf_p, dtype=float))
    af_p = np.atleast_1d(np.asarray(af_p, dtype=float))
    theta_f = np.atleast_1d(np.asarray(theta_f, dtype=float))
    Lp_p = np.atleast_1d(np.asarray(Lp_p, dtype=float))

    q_arr = np.atleast_1d(np.asarray(q, dtype=float))
    q_arr, e_ref, l_ref, Mf_p, af_p, theta_f, Lp_p = np.broadcast_arrays(
        q_arr, e_ref, l_ref, Mf_p, af_p, theta_f, Lp_p)
    eta = symmetric_mass_ratio(q_arr)

    Mf = Mf_p * _ecc_factor(_ECC_MASS_PARAMS, eta, e_ref, l_ref, np.sin)
    af_mag = af_p * _ecc_factor(_ECC_SPIN_PARAMS, eta, e_ref, l_ref, np.cos)
    Lpeak = Lp_p * _ecc_factor(_ECC_LUMI_PARAMS, eta, e_ref, l_ref, _neg_sin)

    # Re-impose the physical bounds after the eccentric rescaling
    Mf = np.minimum(Mf, 1.0)
    af_mag = np.clip(af_mag, 0.0, 1.0)

    if Mf.size == 1:
        return Mf.item(), af_mag.item(), theta_f.item(), Lpeak.item()
    return Mf, af_mag, theta_f, Lpeak


def gwModelRemPE_mf(q, a1, a2, theta1, theta2, phi1, phi2, e_ref, l_ref):
    """
    Final mass Mf/M of an eccentric precessing BBH merger.

    See gwModelRemPE for the parameter description.

    Returns:
        float or array: Final mass in units of the total mass.
    """
    return gwModelRemPE(q, a1, a2, theta1, theta2, phi1, phi2, e_ref, l_ref)[0]


def gwModelRemPE_chif(q, a1, a2, theta1, theta2, phi1, phi2, e_ref, l_ref):
    """
    Final spin magnitude and tilt of an eccentric precessing BBH merger.

    See gwModelRemPE for the parameter description.

    Returns:
        tuple: (af_mag, theta_f) with the tilt in radians.
    """
    _, af_mag, theta_f, _ = gwModelRemPE(
        q, a1, a2, theta1, theta2, phi1, phi2, e_ref, l_ref)
    return af_mag, theta_f


def gwModelRemPE_Lpeak(q, a1, a2, theta1, theta2, phi1, phi2, e_ref, l_ref):
    """
    Peak GW luminosity of an eccentric precessing BBH merger.

    See gwModelRemPE for the parameter description.

    Returns:
        float or array: Peak luminosity in geometric units (c^5/G).
    """
    return gwModelRemPE(q, a1, a2, theta1, theta2, phi1, phi2, e_ref, l_ref)[3]
