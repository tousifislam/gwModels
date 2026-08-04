#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwModelRemSE.py
#
#    gwModelRemSE: remnant properties of eccentric non-precessing BBH mergers.
#    Extends the quasi-circular aligned-spin baseline gwModelRemS with a
#    multiplicative correction in the reference eccentricity and mean anomaly,
#    both specified at t = -2500M before merger.
#
#    This model is provisional. See the "Known limitation" note below before
#    relying on it, and do not extrapolate past e_ref ~ 0.3.
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-02-2026
#    LAST MODIFIED: 08-02-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

from .remnant_utils import symmetric_mass_ratio
from .Kerr import (separatrix_energy, separatrix_angular_momentum,
                   separatrix_ell)
from .gwModelRemS import (
    gwModelRemS_mf,
    gwModelRemS_chif,
    gwModelRemS_Lpeak,
    gwModelRemS_kick,
    _validate_inputs,
    _unwrap,
)

# =============================================================================
# Model structure
#
# Every quantity X is a relative correction to its circular value,
#
#     X|_ecc  = X^circ(q, chi1z, chi2z) [1 + delta_X]
#     delta_X = P_X(e_ref, eta) [1 + alpha_X e_ref T_X(l_ref + phi_X(eta))]
#     P_X     = (a1 + b1 eta) e_ref + (a2 + b2 eta) e_ref^2
#     phi_X   = phi0 + phi1 eta
#
# with anomaly harmonics T_M = sin, T_chi = cos, T_k = cos, T_L = -sin.
#
# The circular limit is exact: e_ref = 0 gives P_X = 0 for all four quantities.
# The correction is multiplicative throughout, including for the recoil, so no
# (1 - 4 eta) prefactor is needed; the eccentric kick vanishes at q = 1 for
# non-spinning binaries because the circular kick does.
#
# The choice of sin or cos is a phase convention absorbed into phi0. A single
# n = 1 harmonic is used for all four quantities. At leading Newtonian
# quadrupole order the energy and angular-momentum fluxes modulate at the
# orbital frequency, while the instantaneous linear-momentum flux carries an
# n = 2 harmonic; the accumulated recoil is nonetheless better described by
# n = 1, being dominated by the final orbits.
#
# =============================================================================


# =============================================================================
# Input handling
# =============================================================================

def _validate_ecc_inputs(q, chi1z, chi2z, e_ref, l_ref):
    """
    Validate and broadcast the eccentric inputs.

    Parameters:
        q: Mass ratio m1/m2 >= 1 (upper bound 10000).
        chi1z, chi2z: Aligned spin components, in [-1, 1].
        e_ref: Eccentricity at t = -2500M, in [0, 1).
        l_ref: Mean anomaly at t = -2500M in radians (periodic, unbounded).

    Returns:
        tuple: Broadcast (q, chi1z, chi2z, e_ref, l_ref) arrays.
    """
    q, chi1z, chi2z = _validate_inputs(q, chi1z, chi2z)

    e_ref = np.atleast_1d(np.asarray(e_ref, dtype=float))
    l_ref = np.atleast_1d(np.asarray(l_ref, dtype=float))

    if np.any(e_ref < 0.0) or np.any(e_ref >= 1.0):
        raise ValueError("e_ref must be in [0, 1).")

    return np.broadcast_arrays(q, chi1z, chi2z, e_ref, l_ref)


# =============================================================================
# Eccentric correction coefficients (7 parameters per quantity)
#
# a1, a2, b1, b2   : step-1 anomaly-averaged eccentricity polynomial
# alpha, phi0, phi1: step-2 anomaly modulation
# =============================================================================

# Anomaly harmonic sin(l + phi). Step-1 source NR, step-2 source (2,2).
_ECC_MASS_PARAMS = {
    'a1':    -0.0007572772776319899,
    'a2':     0.004301594183940622,
    'b1':     0.006935577219009741,
    'b2':    -0.03661705159796363,
    'alpha':  42.10182837745801,
    'phi0':  -12.946363200587673,
    'phi1':   43.36959330111439,
}

# Anomaly harmonic cos(l + phi). Step-1 source NR, step-2 source (2,2).
_ECC_SPIN_PARAMS = {
    'a1':    -0.022255554898112387,
    'a2':     0.11657204506057114,
    'b1':     0.06385577055933453,
    'b2':    -0.3756145497364606,
    'alpha':  14.843163679499568,
    'phi0':   16.30530074514376,
    'phi1':  -95.49006286772388,
}

# Anomaly harmonic cos(l + phi). Step-1 source gwNRHME, step-2 multi-mode.
_ECC_KICK_PARAMS = {
    'a1':    -0.15611345177160055,
    'a2':     1.017795757058323,
    'b1':     0.6682130732613133,
    'b2':    -4.005937339987845,
    'alpha':  10.000538526481133,
    'phi0':  -6.80211996865876,
    'phi1':   20.498627040552186,
}

# Anomaly harmonic -sin(l + phi). Step-1 source NR, step-2 multi-mode.
# The paper writes this term as [1 - alpha_L e sin(...)] with a negative
# alpha_L; that is identical to [1 + alpha_L e (-sin(...))] used here.
_ECC_LUMI_PARAMS = {
    'a1':    -0.13979433987280632,
    'a2':     0.6543722115792396,
    'b1':     0.41092204249025194,
    'b2':    -2.097417088262585,
    'alpha': -5.568124913188894,
    'phi0':   5.773264710298071,
    'phi1':  -29.665356257048074,
}


def _neg_sin(x):
    """Anomaly harmonic -sin, used by the peak luminosity model."""
    return -np.sin(x)


def _ecc_factor(p, eta, e_ref, l_ref, trig):
    """
    Relative eccentric correction 1 + P [1 + alpha e T(l + phi)].

    Parameters:
        p: Coefficient dict with keys a1, a2, b1, b2, alpha, phi0, phi1.
        eta: Symmetric mass ratio.
        e_ref, l_ref: Eccentricity and mean anomaly at t = -2500M.
        trig: Anomaly harmonic, one of np.sin, np.cos, _neg_sin.

    Returns:
        array: Multiplicative factor applied to the circular value.
    """
    poly = ((p['a1'] + p['b1'] * eta) * e_ref
            + (p['a2'] + p['b2'] * eta) * e_ref**2)
    phi = p['phi0'] + p['phi1'] * eta
    mod = 1.0 + p['alpha'] * e_ref * trig(l_ref + phi)
    return 1.0 + poly * mod


# =============================================================================
# Eccentric remnant quantities
# =============================================================================

def gwModelRemSE_mf(q, chi1z, chi2z, e_ref, l_ref):
    """
    Final mass Mf/M of an eccentric non-precessing BBH merger.

    Ansatz:
        Mf|_ecc = Mf^circ [1 + P_M (1 + alpha_M e sin(l + phi_M))]

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z, chi2z: Aligned spin components, in [-1, 1].
        e_ref: Eccentricity at t = -2500M, in [0, 1).
        l_ref: Mean anomaly at t = -2500M in radians.

    Returns:
        float or array: Final mass in units of the total mass.
    """
    q, chi1z, chi2z, e_ref, l_ref = _validate_ecc_inputs(
        q, chi1z, chi2z, e_ref, l_ref)
    eta = symmetric_mass_ratio(q)

    Mf_circ = np.atleast_1d(np.asarray(gwModelRemS_mf(q, chi1z, chi2z), dtype=float))
    return _unwrap(Mf_circ * _ecc_factor(_ECC_MASS_PARAMS, eta, e_ref, l_ref, np.sin))


def gwModelRemSE_chif(q, chi1z, chi2z, e_ref, l_ref):
    """
    Final spin chi_f,z of an eccentric non-precessing BBH merger.

    Ansatz:
        chif|_ecc = chif^circ [1 + P_chi (1 + alpha_chi e cos(l + phi_chi))]

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z, chi2z: Aligned spin components, in [-1, 1].
        e_ref: Eccentricity at t = -2500M, in [0, 1).
        l_ref: Mean anomaly at t = -2500M in radians.

    Returns:
        float or array: Final dimensionless spin along z (signed).
    """
    q, chi1z, chi2z, e_ref, l_ref = _validate_ecc_inputs(
        q, chi1z, chi2z, e_ref, l_ref)
    eta = symmetric_mass_ratio(q)

    chif_circ = np.atleast_1d(np.asarray(gwModelRemS_chif(q, chi1z, chi2z), dtype=float))
    return _unwrap(chif_circ * _ecc_factor(_ECC_SPIN_PARAMS, eta, e_ref, l_ref, np.cos))


def gwModelRemSE_kick(q, chi1z, chi2z, e_ref, l_ref):
    """
    Recoil velocity of an eccentric non-precessing BBH merger.

    Ansatz:
        vk|_ecc = vk^circ [1 + P_k (1 + alpha_k e cos(l + phi_k))]

    The correction is multiplicative, so it vanishes at q = 1 for non-spinning
    binaries because the circular recoil itself vanishes there.

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z, chi2z: Aligned spin components, in [-1, 1].
        e_ref: Eccentricity at t = -2500M, in [0, 1).
        l_ref: Mean anomaly at t = -2500M in radians.

    Returns:
        float or array: Recoil velocity in km/s.
    """
    q, chi1z, chi2z, e_ref, l_ref = _validate_ecc_inputs(
        q, chi1z, chi2z, e_ref, l_ref)
    eta = symmetric_mass_ratio(q)

    vk_circ = np.atleast_1d(np.asarray(gwModelRemS_kick(q, chi1z, chi2z), dtype=float))
    return _unwrap(vk_circ * _ecc_factor(_ECC_KICK_PARAMS, eta, e_ref, l_ref, np.cos))


def gwModelRemSE_Lpeak(q, chi1z, chi2z, e_ref, l_ref):
    """
    Peak GW luminosity of an eccentric non-precessing BBH merger.

    Ansatz:
        Lp|_ecc = Lp^circ [1 + P_L (1 - alpha_L e sin(l + phi_L))]

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z, chi2z: Aligned spin components, in [-1, 1].
        e_ref: Eccentricity at t = -2500M, in [0, 1).
        l_ref: Mean anomaly at t = -2500M in radians.

    Returns:
        float or array: Peak luminosity in geometric units (c^5/G).
    """
    q, chi1z, chi2z, e_ref, l_ref = _validate_ecc_inputs(
        q, chi1z, chi2z, e_ref, l_ref)
    eta = symmetric_mass_ratio(q)

    Lp_circ = np.atleast_1d(np.asarray(gwModelRemS_Lpeak(q, chi1z, chi2z), dtype=float))
    return _unwrap(Lp_circ * _ecc_factor(_ECC_LUMI_PARAMS, eta, e_ref, l_ref, _neg_sin))


# =============================================================================
# Optional eccentric separatrix backbone
#
# Not used by the calibrated model above, which applies its correction directly
# to the circular baseline (c_e = 0 throughout). Provided for users who wish to
# experiment with a separatrix-anchored point-particle backbone, which would
# require a second eccentricity scale near merger. The implementations live in
# Kerr.py and are aliased here under their historical names.
# =============================================================================

E_sep = separatrix_energy
L_sep = separatrix_angular_momentum
ell_sep = separatrix_ell


# =============================================================================
# Combined interface
# =============================================================================

def gwModelRemSE(q, chi1z, chi2z, e_ref, l_ref):
    """
    All gwModelRemSE remnant properties for an eccentric non-precessing BBH.

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z, chi2z: Aligned spin components, in [-1, 1].
        e_ref: Eccentricity at t = -2500M, in [0, 1).
        l_ref: Mean anomaly at t = -2500M in radians.

    Returns:
        tuple: (Mf, chif, vkick, Lpeak) where

            Mf: final mass in units of the total mass
            chif: final dimensionless spin along z (signed)
            vkick: recoil velocity in km/s
            Lpeak: peak GW luminosity in geometric units (c^5/G)

    Note:
        Calibrated for q <= 4 and e0 <= 0.25 on non-spinning systems, and
        provisional: within that domain it is neutral to slightly worse than
        the circular gwModelRemS baseline on NR data. Do not extrapolate past
        e_ref ~ 0.3. See the module header for measured numbers.

    Example:
        >>> import gwModels
        >>> Mf, chif, vk, Lp = gwModels.remnants.gwModelRemSE(
        ...     2.0, 0.0, 0.0, 0.1, 0.0)
    """
    return (gwModelRemSE_mf(q, chi1z, chi2z, e_ref, l_ref),
            gwModelRemSE_chif(q, chi1z, chi2z, e_ref, l_ref),
            gwModelRemSE_kick(q, chi1z, chi2z, e_ref, l_ref),
            gwModelRemSE_Lpeak(q, chi1z, chi2z, e_ref, l_ref))
