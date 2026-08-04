#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwModelRemS.py
#
#    gwModelRemS: remnant properties of non-precessing quasi-circular BBH
#    mergers. Provides the final mass, final spin, peak luminosity, peak GW
#    frequency and recoil velocity from (q, chi1z, chi2z).
#
#    Calibrated to NR (SXS, RIT, Maya) and BHPT simulations spanning
#    1 <= q <= 1000 and |chi_iz| <= 1.
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

from .remnant_utils import validate_q, validate_spin_z, symmetric_mass_ratio
from .Kerr import clip_spin, kerr_isco_energy, kerr_isco_angular_momentum

# =============================================================================
# Model structure
#
# The mass, spin, luminosity and frequency models share one decomposition,
#
#     Q = Q_PP + Q_EM + Q_departure + Q_asymmetry
#
# where Q_PP is the point-particle (extreme-mass-ratio) limit, Q_EM the
# equal-mass contribution, Q_departure an exchange-symmetric interpolation
# correction, and Q_asymmetry a mixed mass-spin term carrying delta_m * chi_a.
#
# The weights (1 - 4*eta) = delta_m^2 and 4*eta = 1 - delta_m^2 form a partition
# of unity, so the limits are exact by construction: at q = 1 only the
# equal-mass polynomial survives, and as eta -> 0 only the Kerr point-particle
# anchor does. Body-exchange symmetry is enforced by pairing every term odd in
# chi_a with delta_m, so remaining spin-difference terms enter through chi_a^2.
#
# Derived variables:
#     eta     = q / (1+q)^2                  symmetric mass ratio
#     delta_m = (q-1) / (q+1)                mass asymmetry
#     chi_hat = (q^2*chi1z + chi2z)/(q^2+1)  effective aligned spin
#     chi_a   = (chi1z - chi2z)/2            spin asymmetry
# =============================================================================

_Q_MAX = 10000.0


# =============================================================================
# Input handling
# =============================================================================

def _validate_inputs(q, chi1z, chi2z):
    """
    Validate and broadcast the aligned-spin inputs.

    Parameters:
        q: Mass ratio m1/m2 >= 1 (upper bound 10000).
        chi1z: Aligned spin of the heavier BH, in [-1, 1].
        chi2z: Aligned spin of the lighter BH, in [-1, 1].

    Returns:
        tuple: Broadcast (q, chi1z, chi2z) arrays.
    """
    q = np.atleast_1d(validate_q(q))
    chi1z, chi2z = validate_spin_z(chi1z, chi2z)
    chi1z = np.atleast_1d(chi1z)
    chi2z = np.atleast_1d(chi2z)

    if np.any(q > _Q_MAX):
        raise ValueError(f"q must be <= {_Q_MAX:.0f}, got maximum {q.max()}.")

    return np.broadcast_arrays(q, chi1z, chi2z)


def _derived_vars(q, chi1z, chi2z):
    """
    Compute the derived variables shared by all gwModelRemS fits.

    Returns:
        tuple: (eta, delta_m, chi_hat, chi_a).
    """
    eta = symmetric_mass_ratio(q)
    delta_m = (q - 1.0) / (q + 1.0)
    chi_hat = (q**2 * chi1z + chi2z) / (q**2 + 1.0)
    chi_a = (chi1z - chi2z) / 2.0
    return eta, delta_m, chi_hat, chi_a


def _unwrap(result):
    """Return a python float for single-element results, else the array."""
    return result.item() if result.size == 1 else result


# =============================================================================
# Final mass (15 parameters)
# =============================================================================

_MASS_PARAMS = {
    'm0':  0.19413984521746355,
    'm1':  0.10339633261824092,
    'm2':  0.05136562565212282,
    'm3':  0.059259698126912445,
    'm4':  0.04510636078432931,
    'ma':  0.0030355645969721927,
    'r0': -0.24113348795903924,
    'r1': -0.17881428005282166,
    'r2':  0.5842046054685163,
    'r3':  0.6853985593244751,
    'g0':  0.2705202166555664,
    'g1':  0.4692298270916386,
    'u0': -0.03017873175899366,
    'u1': -0.002072973840480129,
    'u2':  0.08393469626762673,
}


def gwModelRemS_mf(q, chi1z, chi2z):
    """
    Final mass Mf/M of a non-precessing quasi-circular BBH merger.

    Ansatz::

        Mf/M = 1 - eta * E_rad_hat
        E_rad_hat = (1-4 eta) [1 - E_ISCO(chi_hat)]
                    + 4 eta E_EM(chi_hat, chi_a)
                    + eta (1-4 eta) R_E(eta, chi_hat)
                    + 4 eta delta_m chi_a (u0 + u1 chi_hat + u2 eta)

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z: Aligned spin of the heavier BH, in [-1, 1].
        chi2z: Aligned spin of the lighter BH, in [-1, 1].

    Returns:
        float or array: Final mass in units of the total mass.
    """
    q, chi1z, chi2z = _validate_inputs(q, chi1z, chi2z)
    eta, dm, ch, ca = _derived_vars(q, chi1z, chi2z)
    p = _MASS_PARAMS

    E_EM = (p['m0'] + p['m1'] * ch + p['m2'] * ch**2 + p['m3'] * ch**3
            + p['m4'] * ch**4 + p['ma'] * ca**2)
    R_E = (p['r0'] + p['r1'] * ch + p['r2'] * ch**2 + p['r3'] * ch**3
           + (1.0 - 4.0 * eta) * (p['g0'] + p['g1'] * ch))

    E_rad_hat = ((1.0 - 4.0 * eta) * (1.0 - kerr_isco_energy(clip_spin(ch)))
                 + 4.0 * eta * E_EM
                 + eta * (1.0 - 4.0 * eta) * R_E
                 + 4.0 * eta * dm * ca * (p['u0'] + p['u1'] * ch + p['u2'] * eta))

    return _unwrap(1.0 - eta * E_rad_hat)


# =============================================================================
# Final spin (15 parameters)
# =============================================================================

_SPIN_PARAMS = {
    'mu0':      2.745810547849273,
    'mu1':     -0.7750975698104806,
    'mu2':     -0.11334962909596354,
    'mu3':     -0.037127939010706344,
    'mu4':     -0.014675008172996296,
    'mua':     -0.01463457129256395,
    'rho0':    -0.6643368337227338,
    'rho1':    -0.5626726070405444,
    'rho2':     0.021601598575932197,
    'rho3':     0.2574953686580504,
    'gamma0':  -1.0035773213322308,
    'gamma1':  -0.34064981249426507,
    'upsilon0': 0.2088618137194638,
    'upsilon1': 0.07562787108301205,
    'upsilon2': 1.2889948902218653,
}


def gwModelRemS_chif(q, chi1z, chi2z):
    """
    Final spin chi_f,z of a non-precessing quasi-circular BBH merger.

    Ansatz::

        chi_f,z = S_tilde + eta * ell_orb
        S_tilde = (q^2 chi1z + chi2z)/(1+q)^2 = (1 - 2 eta) chi_hat
        ell_orb = (1-4 eta) ell_Kerr(chi_hat)
                  + 4 eta ell_EM(chi_hat, chi_a)
                  + eta (1-4 eta) R_chi(eta, chi_hat)
                  + 4 eta delta_m chi_a (v0 + v1 chi_hat + v2 eta)

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z: Aligned spin of the heavier BH, in [-1, 1].
        chi2z: Aligned spin of the lighter BH, in [-1, 1].

    Returns:
        float or array: Final dimensionless spin along z (signed).
    """
    q, chi1z, chi2z = _validate_inputs(q, chi1z, chi2z)
    eta, dm, ch, ca = _derived_vars(q, chi1z, chi2z)
    p = _SPIN_PARAMS

    S_tilde = (q**2 * chi1z + chi2z) / (1.0 + q) ** 2

    ell_EM = (p['mu0'] + p['mu1'] * ch + p['mu2'] * ch**2 + p['mu3'] * ch**3
              + p['mu4'] * ch**4 + p['mua'] * ca**2)
    R_chi = (p['rho0'] + p['rho1'] * ch + p['rho2'] * ch**2 + p['rho3'] * ch**3
             + (1.0 - 4.0 * eta) * (p['gamma0'] + p['gamma1'] * ch))

    # ell_Kerr = L_ISCO - 2 chi (E_ISCO - 1). The fits clip the spin inside
    # L_ISCO and E_ISCO but use the unclipped chi_hat in the linear factor, so
    # that asymmetry is written out explicitly here rather than hidden away.
    ch_clipped = clip_spin(ch)
    ell_kerr = (kerr_isco_angular_momentum(ch_clipped)
                - 2.0 * ch * (kerr_isco_energy(ch_clipped) - 1.0))

    ell_orb = ((1.0 - 4.0 * eta) * ell_kerr
               + 4.0 * eta * ell_EM
               + eta * (1.0 - 4.0 * eta) * R_chi
               + 4.0 * eta * dm * ca
               * (p['upsilon0'] + p['upsilon1'] * ch + p['upsilon2'] * eta))

    return _unwrap(S_tilde + eta * ell_orb)


# =============================================================================
# Peak luminosity (15 parameters), fitted in log space with NR upweighting
# =============================================================================

_LUMI_PARAMS = {
    'pi0':     0.016482515682098183,
    'pi1':     0.007127961814113364,
    'pi2':     0.003233751162432972,
    'pi3':     0.003559149252997133,
    'pi4':     0.0026534054949181552,
    'pia':     0.0002774123866868525,
    'sigma0': -0.008102821869602825,
    'sigma1':  0.005558957126793726,
    'sigma2':  0.007990166659494596,
    'sigma3':  0.0029634531597479705,
    'tau0':    0.005355721023184515,
    'tau1':   -0.0034660661752136616,
    'nu0':    -0.0007548818856671142,
    'nu1':     7.085488954161751e-05,
    'nu2':     0.006371079295776314,
}


def gwModelRemS_Lpeak(q, chi1z, chi2z):
    """
    Peak GW luminosity of a non-precessing quasi-circular BBH merger.

    Ansatz::

        L_peak = eta^2 * L_hat
        L_hat  = L_EM(chi_hat, chi_a) + (1-4 eta) P_L(eta, chi_hat)
                 + delta_m chi_a (nu0 + nu1 chi_hat + nu2 eta)

    The eta^2 prefactor enforces L_peak -> 0 in the point-particle limit.

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z: Aligned spin of the heavier BH, in [-1, 1].
        chi2z: Aligned spin of the lighter BH, in [-1, 1].

    Returns:
        float or array: Peak luminosity in geometric units (c^5/G).
    """
    q, chi1z, chi2z = _validate_inputs(q, chi1z, chi2z)
    eta, dm, ch, ca = _derived_vars(q, chi1z, chi2z)
    p = _LUMI_PARAMS

    L_EM = (p['pi0'] + p['pi1'] * ch + p['pi2'] * ch**2 + p['pi3'] * ch**3
            + p['pi4'] * ch**4 + p['pia'] * ca**2)
    P_L = ((p['sigma0'] + p['sigma1'] * ch + p['sigma2'] * ch**2
            + p['sigma3'] * ch**3)
           + (1.0 - 4.0 * eta) * (p['tau0'] + p['tau1'] * ch))

    L_hat = (L_EM + (1.0 - 4.0 * eta) * P_L
             + dm * ca * (p['nu0'] + p['nu1'] * ch + p['nu2'] * eta))

    return _unwrap(eta**2 * L_hat)


# =============================================================================
# Peak GW frequency (15 parameters)
# =============================================================================

_OMEGA_PEAK_PARAMS = {
    'pi0':     0.35914915855522245,
    'pi1':     0.07636739717665963,
    'pi2':     0.019817361880207437,
    'pi3':     0.006113149210959039,
    'pi4':     0.007590370027139799,
    'pia':    -0.001914348478314409,
    'sigma0': -0.11789417126372968,
    'sigma1':  0.0060426467017228435,
    'sigma2':  0.05478697717512359,
    'sigma3':  0.054768378795899325,
    'tau0':    0.03793616597701851,
    'tau1':    0.016429298438086833,
    'nu0':    -0.011332501040997131,
    'nu1':    -0.004514533369780439,
    'nu2':     0.052956421086909664,
}


def gwModelRemS_omega_peak(q, chi1z, chi2z):
    """
    Peak GW frequency M*omega at the peak of abs(h22).

    Ansatz::

        omega_peak = f_EM + (1-4 eta) P + delta_m chi_a U

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z: Aligned spin of the heavier BH, in [-1, 1].
        chi2z: Aligned spin of the lighter BH, in [-1, 1].

    Returns:
        float or array: M*omega at peak abs(h22) (dimensionless).
    """
    q, chi1z, chi2z = _validate_inputs(q, chi1z, chi2z)
    eta, dm, ch, ca = _derived_vars(q, chi1z, chi2z)
    p = _OMEGA_PEAK_PARAMS

    f_EM = (p['pi0'] + p['pi1'] * ch + p['pi2'] * ch**2 + p['pi3'] * ch**3
            + p['pi4'] * ch**4 + p['pia'] * ca**2)
    P = ((p['sigma0'] + p['sigma1'] * ch + p['sigma2'] * ch**2
          + p['sigma3'] * ch**3)
         + (1.0 - 4.0 * eta) * (p['tau0'] + p['tau1'] * ch))

    result = (f_EM + (1.0 - 4.0 * eta) * P
              + dm * ca * (p['nu0'] + p['nu1'] * ch + p['nu2'] * eta))

    return _unwrap(result)


# =============================================================================
# Recoil velocity (20 parameters)
#
# Refit of the Islam & Wadekar (2025) aligned-spin recoil ansatz,
# https://arxiv.org/abs/2511.11536, on the expanded NR + BHPT dataset used
# here. See gwModel_kick_q200 in IW2025_kick_nonprecessing.py for the
# originally published coefficients, which remain available unchanged.
# =============================================================================

_KICK_PARAMS = {
    'A':     12928.969636623517,
    'B':     -2.2280069717843207,
    'C':      4.396107749707793,
    'H':      7275.0760215351365,
    'H2a':    5.828426224678599,
    'H2b':   -0.7397637437211916,
    'H3a':   -0.7716231215400564,
    'H3b':   -1.6378251706055824,
    'H3c':   -1.1596138943566678,
    'H3d':    0.011561211344815421,
    'H3e':    6.707284850188374,
    'H4a':   -0.7910152398321124,
    'H4b':   -1.7799718660640966,
    'H4c':    3.5296259766800535,
    'H4d':   -2.238476068755506,
    'H4e':    0.5582323184747108,
    'H4f':    0.1273479865083466,
    'a_deg': 147.53940497035262,
    'b_deg': 114.07885472142905,
    'c_deg': 144.6099961215966,
}


def _kick_spin_variables(q, chi1z, chi2z):
    """
    Spin combinations used by the recoil model.

    Returns S_tilde_k and Delta_tilde, where

        S_tilde_k = (m2^2 chi1z + m1^2 chi2z)/M^2 = (chi1z + q^2 chi2z)/(1+q)^2
        Delta_tilde = (chi1z - q chi2z)/(1+q)

    Note that S_tilde_k is NOT the inherited spin S_tilde used by
    gwModelRemS_chif. It is that quantity with the body labels interchanged,
    equivalently S_tilde evaluated at q -> 1/q, following the convention of the
    recoil literature where the mass ratio is defined as m2/m1. The
    coefficients above were fitted with it: substituting the final-spin
    S_tilde raises the calibration-set RMS from about 9.5 km/s to 125 km/s.
    Delta_tilde is used exactly as defined elsewhere; only S_tilde is exchanged.

    Returns:
        tuple: (S_tilde_k, Delta_tilde).
    """
    S_tilde_k = (chi1z + q**2 * chi2z) / (1.0 + q) ** 2
    Delta_tilde = (chi1z - q * chi2z) / (1.0 + q)
    return S_tilde_k, Delta_tilde


def gwModelRemS_kick(q, chi1z, chi2z):
    """
    Recoil velocity of a non-precessing quasi-circular BBH merger.

    Ansatz::

        v_kick   = sqrt(V_mass^2 + V_spin^2 + 2 V_mass V_spin cos(xi))
        V_mass   = A eta^2 delta_m (1 + B eta + C eta^2)
        V_spin   = H eta^2 R_v
        xi       = (pi/180)(a_deg + b_deg S_tilde_k + c_deg delta_m Delta_tilde)

    The point-particle and equal-mass-equal-spin recoils both vanish, so only
    the mass-asymmetry and spin-asymmetry vectors contribute. The eta^2 scaling
    enforces v_kick -> 0 as eta -> 0, and delta_m = 0 removes V_mass at q = 1.

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z: Aligned spin of the heavier BH, in [-1, 1].
        chi2z: Aligned spin of the lighter BH, in [-1, 1].

    Returns:
        float or array: Recoil velocity in km/s.
    """
    q, chi1z, chi2z = _validate_inputs(q, chi1z, chi2z)
    p = _KICK_PARAMS

    eta = symmetric_mass_ratio(q)
    dm = (q - 1.0) / (q + 1.0)
    St, Dt = _kick_spin_variables(q, chi1z, chi2z)

    R_v = (Dt
           + p['H2a'] * St * dm
           + p['H2b'] * Dt * St
           + p['H3a'] * Dt**2 * dm
           + p['H3b'] * St**2 * dm
           + p['H3c'] * Dt * St**2
           + p['H3d'] * Dt**3
           + p['H3e'] * Dt * dm**2
           + p['H4a'] * St * Dt**2 * dm
           + p['H4b'] * St**3 * dm
           + p['H4c'] * St * dm**3
           + p['H4d'] * Dt * St * dm**2
           + p['H4e'] * Dt * St**3
           + p['H4f'] * St * Dt**3)

    V_spin = p['H'] * eta**2 * R_v
    V_mass = p['A'] * eta**2 * dm * (1.0 + p['B'] * eta + p['C'] * eta**2)
    xi = np.deg2rad(p['a_deg'] + p['b_deg'] * St + p['c_deg'] * dm * Dt)

    return _unwrap(np.sqrt(V_mass**2 + V_spin**2
                           + 2.0 * V_mass * V_spin * np.cos(xi)))


# =============================================================================
# Combined interface
# =============================================================================

def gwModelRemS(q, chi1z, chi2z):
    """
    All gwModelRemS remnant properties for a non-precessing quasi-circular BBH.

    Parameters:
        q: Mass ratio m1/m2 >= 1, scalar or array.
        chi1z: Aligned spin of the heavier BH, in [-1, 1].
        chi2z: Aligned spin of the lighter BH, in [-1, 1].

    Returns:
        tuple: (Mf, chif, Lpeak, omega_peak, vkick) where

            Mf: final mass in units of the total mass
            chif: final dimensionless spin along z (signed)
            Lpeak: peak GW luminosity in geometric units (c^5/G)
            omega_peak: M*omega at peak abs(h22) (dimensionless)
            vkick: recoil velocity in km/s

    Example:
        >>> import gwModels
        >>> Mf, chif, Lp, wp, vk = gwModels.remnants.gwModelRemS(3.0, 0.5, -0.2)
    """
    return (gwModelRemS_mf(q, chi1z, chi2z),
            gwModelRemS_chif(q, chi1z, chi2z),
            gwModelRemS_Lpeak(q, chi1z, chi2z),
            gwModelRemS_omega_peak(q, chi1z, chi2z),
            gwModelRemS_kick(q, chi1z, chi2z))
