"""
Shared utilities for eccentric dynamics modules.

Constants, zetadot, ISCO computation, event functions, and parameter helpers.
"""

import math
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

# Physical constants (plain floats for numba)
EULER_GAMMA = 0.5772156649015329  # np.euler_gamma
LOG2 = math.log(2.0)
LOG3 = math.log(3.0)
LOG5 = math.log(5.0)
PI = math.pi


# ---------------------------------------------------------------------------
# zetadot (shared between resummed and non-resummed)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _zetadot_numba(e, x, zeta, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    e6 = e4 * e2
    nu2 = nu * nu
    oome2 = 1.0 - e2
    sqrt_oome2 = math.sqrt(oome2)

    chiA2 = chi_A * chi_A
    chiS2 = chi_S * chi_S
    chiAS = chi_A * chi_S

    cos_z = math.cos(zeta)
    sin_z = math.sin(zeta)
    cos_2z = math.cos(2.0 * zeta)
    cos_3z = math.cos(3.0 * zeta)
    cos_4z = math.cos(4.0 * zeta)

    f = 1.0 + e * cos_z
    f2 = f * f

    # 0PN
    term_0pn = x**1.5 * f2 / oome2**1.5

    # 1PN
    term_1pn = -3.0 * x**2.5 * f2 * (1.0 + e2 + e * cos_z) / oome2**2.5

    # 1.5PN SO
    so_15pn = (x**3 * f2
               * (2.0 * delta * chi_A + (2.0 - nu) * chi_S)
               * (2.0 + e2 + e * cos_z) / oome2**3)

    # 2PN SS
    ss_2pn_inner = (
        (delta * kappa_A + kappa_S - 2.0 * kappa_S * nu)
        * (3.0 + e2 + e * cos_z)
        - (-1.0 + 4.0 * nu) * chiA2 * (3.0 + 2.0 * e2
                                         + 2.0 * e * cos_z)
        + 2.0 * delta * chiAS * (3.0 + 2.0 * e2 + 2.0 * e * cos_z)
        + chiS2 * (3.0 + 2.0 * e2 + 2.0 * e * cos_z)
    )
    ss_2pn = -x**3.5 * f2 * ss_2pn_inner / (2.0 * oome2**3.5)

    # 2PN non-spin
    inner_2pn = (48.0 + 4.0 * e4 * (-6.0 + nu) - 40.0 * nu
                 - 16.0 * e2 * (5.0 + nu)
                 + 6.0 * sqrt_oome2 * (-5.0 + 2.0 * nu)
                 + 4.0 * e * (1.0 + e2 * (-15.0 + nu)
                              - 8.0 * nu) * cos_z
                 - 3.0 * e2 * (1.0 + 2.0 * nu) * cos_2z)
    term_2pn = -x**3.5 * f2 * inner_2pn / (4.0 * oome2**3.5)

    # 2.5PN SO
    so_25pn_inner = (
        oome2**1.5 * (-256.0 * delta * (-3.0 + nu) * chi_A
                      + 128.0 * (6.0 - 8.0 * nu + nu2) * chi_S)
        + delta * chi_A * (
            -1408.0 + 2304.0 * e2 + 768.0 * e4
            + 800.0 * nu + 616.0 * e2 * nu + 44.0 * e4 * nu
            + e * (-384.0 + 872.0 * nu
                   + 5.0 * e2 * (288.0 + 25.0 * nu)) * cos_z
            + 4.0 * e2 * (-32.0 + 59.0 * nu) * cos_2z
            - 32.0 * e3 * cos_3z + 27.0 * e3 * nu * cos_3z
        )
        + chi_S * (
            -1408.0 + 2304.0 * e2 + 768.0 * e4
            + 2848.0 * nu + 568.0 * e2 * nu - 220.0 * e4 * nu
            - 256.0 * nu2 - 272.0 * e2 * nu2 - 40.0 * e4 * nu2
            - e * (8.0 * (48.0 - 295.0 * nu + 50.0 * nu2)
                   + e2 * (-1440.0 + 385.0 * nu
                           + 94.0 * nu2)) * cos_z
            - 4.0 * e2 * (32.0 - 153.0 * nu + 34.0 * nu2) * cos_2z
            - 32.0 * e3 * cos_3z + 73.0 * e3 * nu * cos_3z
            - 18.0 * e3 * nu2 * cos_3z
        )
    )
    so_25pn = -x**4 * f2 * so_25pn_inner / (64.0 * oome2**4)

    # 3PN SS
    ss_3pn_kappa_const = (
        420.0 * delta * kappa_A - 36.0 * e2 * delta * kappa_A
        - 84.0 * e4 * delta * kappa_A
        + 420.0 * kappa_S - 36.0 * e2 * kappa_S - 84.0 * e4 * kappa_S
        - 264.0 * delta * kappa_A * nu
        - 188.0 * e2 * delta * kappa_A * nu
        + 8.0 * e4 * delta * kappa_A * nu
        - 1104.0 * kappa_S * nu - 116.0 * e2 * kappa_S * nu
        + 176.0 * e4 * kappa_S * nu
        + 192.0 * kappa_S * nu2 + 184.0 * e2 * kappa_S * nu2
        - 16.0 * e4 * kappa_S * nu2
    )

    ss_3pn_sqrt_block = oome2**1.5 * (
        12.0 * (delta * kappa_A * (-14.0 + 5.0 * nu)
                + kappa_S * (-14.0 + 33.0 * nu - 6.0 * nu2))
        - 24.0 * (11.0 - 46.0 * nu + 6.0 * nu2) * chiA2
        + 48.0 * delta * (-11.0 + 9.0 * nu) * chiAS
        - 24.0 * (11.0 - 16.0 * nu + 4.0 * nu2) * chiS2
    )

    ss_3pn_kappa_cos = -e * (
        delta * kappa_A * (e2 * (111.0 + 19.0 * nu)
                           + 4.0 * (-72.0 + 53.0 * nu))
        - kappa_S * (e2 * (-111.0 + 203.0 * nu + 38.0 * nu2)
                     + 4.0 * (72.0 - 197.0 * nu + 58.0 * nu2))
    ) * cos_z

    ss_3pn_kappa_cos2 = -6.0 * e2 * (
        delta * kappa_A * (-18.0 + 11.0 * nu)
        + kappa_S * (-18.0 + 47.0 * nu - 18.0 * nu2)
    ) * cos_2z

    ss_3pn_kappa_cos3 = (
        15.0 * e3 * delta * kappa_A
        + 15.0 * e3 * kappa_S
        - 9.0 * e3 * delta * kappa_A * nu
        - 39.0 * e3 * kappa_S * nu
        + 18.0 * e3 * kappa_S * nu2
    ) * cos_3z

    ss_3pn_chiAS = -2.0 * delta * chiAS * (
        -468.0 + 948.0 * e2 + 336.0 * e4
        + 684.0 * nu + 244.0 * e2 * nu - 112.0 * e4 * nu
        - e * (180.0 - 580.0 * nu + e2 * (-576.0 + 79.0 * nu)) * cos_z
        + 12.0 * e2 * (-1.0 + 16.0 * nu) * cos_2z
        + 27.0 * e3 * nu * cos_3z
    )

    ss_3pn_chiS2 = chiS2 * (
        468.0 - 948.0 * e2 - 336.0 * e4
        - 1188.0 * nu - 340.0 * e2 * nu + 208.0 * e4 * nu
        + 336.0 * nu2 + 96.0 * e2 * nu2 - 48.0 * e4 * nu2
        - e3 * (576.0 - 169.0 * nu + 45.0 * nu2) * cos_z
        + 4.0 * e * (45.0 - 247.0 * nu + 63.0 * nu2) * cos_z
        + 12.0 * e2 * (1.0 - 27.0 * nu + 6.0 * nu2) * cos_2z
        - 45.0 * e3 * nu * cos_3z + 9.0 * e3 * nu2 * cos_3z
    )

    ss_3pn_chiA2 = chiA2 * (
        468.0 - 948.0 * e2 - 336.0 * e4
        - 2052.0 * nu + 3644.0 * e2 * nu + 1360.0 * e4 * nu
        + 384.0 * nu2 + 400.0 * e2 * nu2 - 64.0 * e4 * nu2
        + e * (180.0 - 892.0 * nu + 496.0 * nu2
               + e2 * (-576.0 + 2293.0 * nu
                       + 44.0 * nu2)) * cos_z
        + 12.0 * e2 * (1.0 - 9.0 * nu + 18.0 * nu2) * cos_2z
        - 9.0 * e3 * nu * cos_3z + 36.0 * e3 * nu2 * cos_3z
    )

    ss_3pn_total = (ss_3pn_kappa_const + ss_3pn_sqrt_block
                    + ss_3pn_kappa_cos + ss_3pn_kappa_cos2
                    + ss_3pn_kappa_cos3 + ss_3pn_chiAS
                    + ss_3pn_chiS2 + ss_3pn_chiA2)
    ss_3pn = -x**4.5 * f2 * ss_3pn_total / (24.0 * oome2**4.5)

    # 3PN non-spin
    inner_3pn = (
        8.0 * e * (1776.0 + 540.0 * e4 * (-4.0 + nu)
                    + (5212.0 - 123.0 * PI**2) * nu
                    - 656.0 * nu2
                    - 2.0 * e2 * (237.0 + 1264.0 * nu
                                  + 50.0 * nu2)) * cos_z
        + 2.0 * sqrt_oome2 * (
            -4320.0 - 8000.0 * nu + 123.0 * PI**2 * nu
            + 960.0 * nu2
            + 96.0 * e2 * (-90.0 + 34.0 * nu + 5.0 * nu2)
            + 1440.0 * e * (-5.0 + 2.0 * nu) * cos_z)
        + e2 * (1056.0 + (12224.0 - 123.0 * PI**2) * nu
                - 384.0 * nu2
                + 96.0 * e2 * (-21.0 - 40.0 * nu
                               + 4.0 * nu2)) * cos_2z
        + 2.0 * (1728.0 + 10848.0 * e2 - 13344.0 * e4
                  - 1728.0 * e6
                  + 31088.0 * nu + 9040.0 * e2 * nu
                  - 3440.0 * e4 * nu + 1008.0 * e6 * nu
                  - 861.0 * PI**2 * nu - 492.0 * e2 * PI**2 * nu
                  - 2304.0 * nu2 - 2816.0 * e2 * nu2
                  + 80.0 * e4 * nu2
                  + 24.0 * e3 * (1.0 - 8.0 * nu
                                 + 30.0 * nu2) * cos_3z
                  + 72.0 * e4 * nu * (-4.0 + 3.0 * nu) * cos_4z)
    )
    term_3pn = x**4.5 * f2 * inner_3pn / (384.0 * oome2**4.5)

    # 2.5PN RR
    term_rr = (-e * (608.0 + 2370.0 * e2 + 5635.0 * e4) * x**4 * nu
               * (2.0 + e * cos_z) * sin_z / 30.0)

    return (term_0pn + term_1pn + so_15pn + ss_2pn + term_2pn
            + so_25pn + ss_3pn + term_3pn + term_rr)


def zetadot_func(e, x, zeta, nu, chi_S=0.0, chi_A=0.0, delta=None,
                 kappa_S=0.0, kappa_A=0.0):
    """
    Public wrapper for zetadot. Computes delta from nu if not provided.
    """
    if delta is None:
        arg = 1.0 - 4.0 * nu
        if arg < 0:
            arg = 0.0
        delta = math.sqrt(arg)
    return _zetadot_numba(e, x, zeta, nu, chi_S, chi_A, delta,
                          kappa_S, kappa_A)


# ---------------------------------------------------------------------------
# ISCO computation
# ---------------------------------------------------------------------------

def _kerr_isco_radius(a):
    a = np.clip(a, -0.9999, 0.9999)
    Z1 = 1.0 + (1 - a**2)**(1/3.0) * ((1 + a)**(1/3.0) + (1 - a)**(1/3.0))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    return 3 + Z2 - np.sign(a) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))


def _compute_x_isco(chi1, chi2, q):
    a_eff = (chi1 + chi2 * q**2) / (1.0 + q)**2
    r_isco = _kerr_isco_radius(a_eff)
    omega_isco = 1.0 / (r_isco**1.5 + a_eff)
    return omega_isco**(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Parameter preparation
# ---------------------------------------------------------------------------

def prepare_params(q, chi1, chi2, kappa1=1.0, kappa2=1.0):
    """Compute derived parameters from physical inputs."""
    nu = q / (1.0 + q)**2
    delta = (q - 1.0) / (q + 1.0)
    chi_S = 0.5 * (chi1 + chi2)
    chi_A = 0.5 * (chi1 - chi2)
    kappa_S = 0.5 * (kappa1 + kappa2) - 1.0
    kappa_A = 0.5 * (kappa1 - kappa2)
    return nu, delta, chi_S, chi_A, kappa_S, kappa_A


# ---------------------------------------------------------------------------
# Integration driver (shared by both resum and nonresum)
# ---------------------------------------------------------------------------

def integrate_dynamics(rhs_func, rhs_inner_func, q, chi1=0.0, chi2=0.0,
                       e0=0.1, zeta0=0.0, t_eval=None, omega0=0.0085,
                       t_end=None, rtol=1e-10, atol=1e-12, kappa1=1.0,
                       kappa2=1.0, x_stop=None):
    """
    Integrate eccentric dynamics ODEs.

    Parameters
    ----------
    rhs_func : callable
        ODE RHS compatible with solve_ivp (returns list).
    rhs_inner_func : callable
        Inner numba RHS (for warmup).
    q : float
        Mass ratio q = m1/m2 >= 1.
    chi1, chi2 : float
        Dimensionless spins.
    e0 : float
        Initial eccentricity.
    zeta0 : float
        Initial relativistic anomaly.
    t_eval : array-like, optional
        Times at which to store solution.
    omega0 : float
        Initial orbit-averaged orbital frequency.
    t_end : float, optional
        End time.
    rtol, atol : float
        Solver tolerances.
    kappa1, kappa2 : float
        Spin-induced quadrupole parameters (1 for BH).
    x_stop : float or str, optional
        Stopping x value. None -> 0.5, 'isco' -> x_ISCO, float -> that value.

    Returns
    -------
    dict with keys: t, e, x, zeta, x_isco, success, message, sol
    """
    nu, delta_val, chi_S, chi_A, kappa_S_val, kappa_A_val = prepare_params(
        q, chi1, chi2, kappa1, kappa2)

    x_isco = _compute_x_isco(chi1, chi2, q)

    if x_stop == 'isco':
        x_stop_val = x_isco
    elif x_stop is not None:
        x_stop_val = float(x_stop)
    else:
        x_stop_val = 0.5

    def _event_x_stop(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
        return x_stop_val - y[1]
    _event_x_stop.terminal = True
    _event_x_stop.direction = -1

    def _event_e_negative(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
        return y[0] - 1e-10
    _event_e_negative.terminal = True
    _event_e_negative.direction = -1

    def _event_oome2_small(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
        return (1.0 - y[0]**2) - 0.01
    _event_oome2_small.terminal = True
    _event_oome2_small.direction = -1

    x0 = omega0**(2.0 / 3.0)

    if t_end is None:
        t_merge_est = 5.0 / (256.0 * nu * x0**4)
        t_end = 1.05 * t_merge_est

    t_span = (0.0, t_end)
    y0 = [e0, x0, zeta0]

    sol = solve_ivp(
        rhs_func, t_span, y0,
        args=(nu, chi_S, chi_A, delta_val, kappa_S_val, kappa_A_val),
        method='DOP853',
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        dense_output=True,
        events=[_event_x_stop, _event_e_negative, _event_oome2_small],
        max_step=2.0 * PI / (10.0 * omega0),
    )

    return {
        't': sol.t,
        'e': sol.y[0],
        'x': sol.y[1],
        'zeta': sol.y[2],
        'x_isco': x_isco,
        'success': sol.success,
        'message': sol.message,
        'sol': sol,
    }


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup_zetadot():
    """Trigger JIT compilation of zetadot."""
    _zetadot_numba(0.1, 0.01, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0)
