"""
Non-precessing (aligned-spin) eccentric orbital dynamics at 3PN order.

Implements the eccentricity-resummed 3PN evolution ODEs for (e, x, zeta)
from Gamboa, Khalil & Buonanno (SEOBNRv5EHM supplementary data),
including all spin-orbit (SO) and spin-spin (SS) terms.

Spin variables:
    chi_S = (chi1 + chi2) / 2      symmetric spin combination
    chi_A = (chi1 - chi2) / 2      antisymmetric spin combination
    delta = (m1 - m2) / M          mass difference ratio = sqrt(1 - 4*nu)
    kappa_S, kappa_A               tidal deformability combinations
                                   (= 0 for BBH since kappa1 = kappa2 = 1)

Gauge: alpha = -16/3, beta = -13/2.
epsilon = 1 (PN counting parameter) throughout.
SO = 1 (spin order counting parameter) throughout.

References:
    EOB_fluxes.dat.m (edotResum lines 2847-2976,
                      xdotResum lines 3115-3218,
                      zetadot   lines 3226-3304)
"""

import logging
import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# Physical constants
EULER_GAMMA = np.euler_gamma  # 0.5772156649...
LOG2 = np.log(2)
LOG3 = np.log(3)
LOG5 = np.log(5)
PI = np.pi


def _get_delta(nu, delta):
    """Compute delta = sqrt(1 - 4*nu) if not provided."""
    if delta is None:
        arg = 1.0 - 4.0 * nu
        if arg < 0:
            arg = 0.0
        return np.sqrt(arg)
    return delta


# ============================================================================
# de/dt  --  eccentricity-resummed (aligned-spin)
# Source: EOB_fluxes.dat.m lines 2847-2976
# ============================================================================

def edot_resum(e, x, nu, chi_S=0.0, chi_A=0.0, delta=None,
               kappa_S=0.0, kappa_A=0.0):
    """
    Resummed de/dt for aligned-spin eccentric binaries at 3PN order.

    Parameters
    ----------
    e : float
        Keplerian eccentricity.
    x : float
        PN frequency parameter x = (M*Omega)^(2/3).
    nu : float
        Symmetric mass ratio nu = m1*m2/(m1+m2)^2.
    chi_S : float
        Symmetric spin combination (chi1 + chi2) / 2.
    chi_A : float
        Antisymmetric spin combination (chi1 - chi2) / 2.
    delta : float or None
        Mass difference ratio (m1 - m2) / M.  If None, computed as
        sqrt(1 - 4*nu).
    kappa_S : float
        Symmetric tidal parameter (0 for BBH).
    kappa_A : float
        Antisymmetric tidal parameter (0 for BBH).

    Returns
    -------
    float
        Time derivative de/dt in geometric units (G = c = 1, M = 1).
    """
    delta = _get_delta(nu, delta)

    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    nu2 = nu * nu
    nu3 = nu2 * nu
    oome2 = 1.0 - e2           # 1 - e^2
    sqrt_oome2 = np.sqrt(oome2)
    sqrt_x = np.sqrt(x)

    chiA2 = chi_A * chi_A
    chiS2 = chi_S * chi_S
    chiAS = chi_A * chi_S

    # Leading prefactor: -304*e*x^4*nu / (15*(1-e^2)^(5/2))
    prefactor = -304.0 * e * x**4 * nu / (15.0 * oome2**2.5)

    # ------------------------------------------------------------------
    # 0PN: 1 + 121*e^2/304
    # ------------------------------------------------------------------
    term_0pn = 1.0 + 121.0 * e2 / 304.0

    # ------------------------------------------------------------------
    # 1.5PN tail (non-spin): Pi*x^(3/2) * (...)/(29184*(1-e^2)^(3/2))
    # ------------------------------------------------------------------
    tail_15pn = (PI * x**1.5
                 * (189120.0 + 286512.0 * e2 + 24217.0 * e4 - 98.0 * e6)
                 / (29184.0 * oome2**1.5))

    # ------------------------------------------------------------------
    # 1.5PN SO: x^(3/2) * (...) / (1824*(1-e^2)^(3/2))
    #   Lines 2862-2865
    # ------------------------------------------------------------------
    so_15pn_num = (-16232.0 * delta * chi_A
                   + 8.0 * (-2029.0 + 1664.0 * nu) * chi_S
                   + e4 * (1869.0 * delta * chi_A
                           + (1869.0 - 276.0 * nu) * chi_S)
                   + e2 * (-248.0 * delta * chi_A
                           + (-248.0 + 7972.0 * nu) * chi_S))
    so_15pn = x**1.5 * so_15pn_num / (1824.0 * oome2**1.5)

    # ------------------------------------------------------------------
    # 1PN (non-spin): -x * (...) / (51072*(1-e^2))
    # ------------------------------------------------------------------
    pn1_num = (e4 * (94887.0 + 19768.0 * nu)
               + 8.0 * (20547.0 + 24556.0 * nu)
               + e2 * (464376.0 + 257124.0 * nu))
    term_1pn = -x * pn1_num / (51072.0 * oome2)

    # ------------------------------------------------------------------
    # 2.5PN tail (non-spin): -Pi*x^(5/2)*(...) / (2451456*(1-e^2)^(5/2))
    # ------------------------------------------------------------------
    tail_25pn_num = (1372.0 * e8 * (-9.0 + nu)
                     + 192.0 * (263841.0 + 577888.0 * nu)
                     + e6 * (8626445.0 + 1850510.0 * nu)
                     + 25.0 * e4 * (9908007.0 + 4499656.0 * nu)
                     + e2 * (302170212.0 + 319380528.0 * nu))
    tail_25pn = -PI * x**2.5 * tail_25pn_num / (2451456.0 * oome2**2.5)

    # ------------------------------------------------------------------
    # 2.5PN SO tail: Pi*x^3 * (...) / (43776*(1-e^2)^3)
    #   Lines 2866-2871
    # ------------------------------------------------------------------
    so_25pn_tail_num = (-2360832.0 * delta * chi_A
                        + 384.0 * (-6148.0 + 4937.0 * nu) * chi_S
                        + e6 * (132438.0 * delta * chi_A
                                + (132438.0 - 42287.0 * nu) * chi_S)
                        + e2 * (-1882704.0 * delta * chi_A
                                + 48.0 * (-39223.0 + 74954.0 * nu) * chi_S)
                        + e4 * (2251234.0 * delta * chi_A
                                + (2251234.0 + 191122.0 * nu) * chi_S))
    so_25pn_tail = PI * x**3 * so_25pn_tail_num / (43776.0 * oome2**3)

    # ------------------------------------------------------------------
    # 2PN (non-spin): -x^2 * (...) / (612864*(1-e^2)^2)
    # ------------------------------------------------------------------
    pn2_num = (-3.0 * e6 * (1056441.0 + 339608.0 * nu + 60256.0 * nu2)
               - 16.0 * (-765197.0 + 772695.0 * nu + 225792.0 * nu2)
               - 12.0 * e2 * (-949135.0 + 4137363.0 * nu + 987567.0 * nu2)
               - 2.0 * e4 * (11094859.0 + 13799931.0 * nu + 2536968.0 * nu2)
               + sqrt_oome2 * (1532160.0 * (-5.0 + 2.0 * nu)
                               + 609840.0 * e2 * (-5.0 + 2.0 * nu)))
    term_2pn = -x**2 * pn2_num / (612864.0 * oome2**2)

    # ------------------------------------------------------------------
    # 2PN SS: x^2 * (...) / (2432*(1-e^2)^2)
    #   Lines 2872-2880
    # ------------------------------------------------------------------
    ss_2pn_num = (13472.0 * (delta * kappa_A + kappa_S
                             - 2.0 * kappa_S * nu)
                  + (10760.0 - 41600.0 * nu) * chiA2
                  + 21520.0 * delta * chiAS
                  + 40.0 * (269.0 - 36.0 * nu) * chiS2
                  + e4 * (1144.0 * (delta * kappa_A + kappa_S
                                    - 2.0 * kappa_S * nu)
                          + (-963.0 + 4032.0 * nu) * chiA2
                          - 1926.0 * delta * chiAS
                          - 9.0 * (107.0 + 20.0 * nu) * chiS2)
                  + e2 * (14512.0 * (delta * kappa_A + kappa_S
                                     - 2.0 * kappa_S * nu)
                          + 4.0 * (-881.0 + 4064.0 * nu) * chiA2
                          - 7048.0 * delta * chiAS
                          - 4.0 * (881.0 + 540.0 * nu) * chiS2))
    ss_2pn = x**2 * ss_2pn_num / (2432.0 * oome2**2)

    # ------------------------------------------------------------------
    # 2.5PN SO (non-tail): -x^(5/2) * (...) / (1225728*(1-e^2)^(5/2))
    #   Lines 2881-2893
    # ------------------------------------------------------------------
    so_25pn_num = (
        -64.0 * delta * (57681.0 + 952868.0 * nu) * chi_A
        + 64.0 * (-57681.0 - 985610.0 * nu + 670516.0 * nu2) * chi_S
        + e6 * (3.0 * delta * (3441339.0 + 769244.0 * nu) * chi_A
                + (10324017.0 + 3203016.0 * nu - 817656.0 * nu2) * chi_S)
        + e2 * (-24.0 * delta * (4476734.0 + 1745023.0 * nu) * chi_A
                + 24.0 * (-4476734.0 + 3800571.0 * nu
                          + 3015082.0 * nu2) * chi_S)
        + e4 * (4.0 * delta * (7388082.0 + 4790765.0 * nu) * chi_A
                + (29552328.0 + 67302476.0 * nu
                   + 9741032.0 * nu2) * chi_S)
        + sqrt_oome2 * (
            -8171520.0 * delta * (-3.0 + nu) * chi_A
            + 4085760.0 * (6.0 - 8.0 * nu + nu2) * chi_S
            + e2 * (4919040.0 * delta * (-3.0 + nu) * chi_A
                    - 2459520.0 * (6.0 - 8.0 * nu + nu2) * chi_S)
            + e4 * (3252480.0 * delta * (-3.0 + nu) * chi_A
                    - 1626240.0 * (6.0 - 8.0 * nu + nu2) * chi_S)
        )
    )
    so_25pn = -x**2.5 * so_25pn_num / (1225728.0 * oome2**2.5)

    # ------------------------------------------------------------------
    # 3PN SS: x^3 * (...) / (1225728*(1-e^2)^3*(1+sqrt(1-e^2)))
    #   Lines 2894-2934
    # ------------------------------------------------------------------
    # The spin-spin terms at 3PN come with a factor
    # 1/((1-e^2)^3 * (1+sqrt(1-e^2)))
    ss_3pn_num = (
        # e^0 terms
        -192.0 * (delta * kappa_A * (-88517.0 + 333088.0 * nu)
                  + kappa_S * (-88517.0 + 510122.0 * nu
                               - 426748.0 * nu2))
        + 32.0 * (2069461.0 - 9635809.0 * nu + 4395216.0 * nu2) * chiA2
        + 64.0 * delta * (2069461.0 - 3985373.0 * nu) * chiAS
        + 32.0 * (2069461.0 - 6612781.0 * nu + 2014516.0 * nu2) * chiS2
        # e^4
        + e4 * (
            -24.0 * (delta * kappa_A * (2157205.0 + 1979068.0 * nu)
                     + kappa_S * (2157205.0 - 2335342.0 * nu
                                  - 2790004.0 * nu2))
            + 12.0 * (3131923.0 - 13837483.0 * nu
                      + 1693552.0 * nu2) * chiA2
            + 24.0 * delta * (3131923.0 - 2182831.0 * nu) * chiAS
            + 12.0 * (3131923.0 - 3055871.0 * nu
                      + 1466108.0 * nu2) * chiS2
        )
        # e^6
        + e6 * (
            6.0 * (delta * kappa_A * (271948.0 - 604513.0 * nu)
                   + kappa_S * (271948.0 - 1148409.0 * nu
                                + 873600.0 * nu2))
            + (19192791.0 - 80303280.0 * nu
               + 9886464.0 * nu2) * chiA2
            + 6.0 * delta * (6397597.0 - 4003818.0 * nu) * chiAS
            + 3.0 * (6397597.0 - 6830264.0 * nu
                     + 1593536.0 * nu2) * chiS2
        )
        # e^2
        + e2 * (
            -192.0 * (25.0 * delta * kappa_A * (20864.0 + 26341.0 * nu)
                      + kappa_S * (521600.0 - 384675.0 * nu
                                   - 947247.0 * nu2))
            + 16.0 * (-9106018.0 + 34227913.0 * nu
                      + 5762232.0 * nu2) * chiA2
            - 32.0 * delta * (9106018.0 + 3284449.0 * nu) * chiAS
            + 16.0 * (-9106018.0 - 4372739.0 * nu
                      + 3135692.0 * nu2) * chiS2
        )
        # sqrt(1-e^2) block
        + sqrt_oome2 * (
            # e^0 in sqrt block
            -192.0 * (delta * kappa_A * (-88517.0 + 333088.0 * nu)
                      + kappa_S * (-88517.0 + 510122.0 * nu
                                   - 426748.0 * nu2))
            + 32.0 * (2069461.0 - 9635809.0 * nu
                      + 4395216.0 * nu2) * chiA2
            + 64.0 * delta * (2069461.0 - 3985373.0 * nu) * chiAS
            + 32.0 * (2069461.0 - 6612781.0 * nu
                      + 2014516.0 * nu2) * chiS2
            # e^6 in sqrt block
            + e6 * (
                -18.0 * (delta * kappa_A * (225564.0 + 88571.0 * nu)
                         + kappa_S * (225564.0 - 362557.0 * nu
                                      - 155680.0 * nu2))
                + 27.0 * (379573.0 - 1588880.0 * nu
                          + 185472.0 * nu2) * chiA2
                + 18.0 * delta * (1138719.0 - 521486.0 * nu) * chiAS
                + 9.0 * (1138719.0 - 831208.0 * nu
                         + 169792.0 * nu2) * chiS2
            )
            # e^4 in sqrt block
            + e4 * (
                -24.0 * (delta * kappa_A * (2515885.0 + 1850968.0 * nu)
                         + kappa_S * (2515885.0 - 3180802.0 * nu
                                      - 2636284.0 * nu2))
                + 12.0 * (2004643.0 - 9123403.0 * nu
                          + 1078672.0 * nu2) * chiA2
                + 24.0 * delta * (2004643.0 - 1260511.0 * nu) * chiAS
                + 12.0 * (2004643.0 - 1416191.0 * nu
                          + 1056188.0 * nu2) * chiS2
            )
            # e^2 in sqrt block
            + e2 * (
                -576.0 * (45.0 * delta * kappa_A * (3312.0 + 5075.0 * nu)
                          + kappa_S * (149040.0 - 69705.0 * nu
                                       - 326389.0 * nu2))
                + 16.0 * (-7701538.0 + 28354633.0 * nu
                          + 6528312.0 * nu2) * chiA2
                - 32.0 * delta * (7701538.0 + 4433569.0 * nu) * chiAS
                + 16.0 * (-7701538.0 - 6415619.0 * nu
                          + 3646412.0 * nu2) * chiS2
            )
        )
    )
    ss_3pn = (x**3 * ss_3pn_num
              / (1225728.0 * oome2**3 * (1.0 + sqrt_oome2)))

    # ------------------------------------------------------------------
    # 3PN (non-spin): -x^3 * (...) / (169885900800*(1-e^2)^3*(1+sqrt(1-e^2)))
    # ------------------------------------------------------------------
    log_arg = 2.0 * oome2 * sqrt_x / (1.0 + sqrt_oome2)
    log_term = np.log(log_arg)

    # e^0 coefficient (multiplied by 64)
    c0 = 64.0 * (-641828882523.0
                  + 109482468480.0 * EULER_GAMMA
                  + 380870611275.0 * nu
                  + 30985883775.0 * nu2
                  + 12481353500.0 * nu3
                  - 727650.0 * PI**2 * (49216.0 + 17917.0 * nu)
                  + 102648712320.0 * LOG2
                  + 116761130640.0 * LOG3)

    # e^2 coefficient (multiplied by 32)
    c2 = 32.0 * (-6068102350278.0
                  + 792146234880.0 * EULER_GAMMA
                  - 1938077318100.0 * nu
                  + 1564141592475.0 * nu2
                  + 186429656875.0 * nu3
                  + 5821200.0 * PI**2 * (-44512.0 + 8077.0 * nu)
                  + 7937977259520.0 * LOG2
                  - 2860647700680.0 * LOG3)

    # e^4 coefficient (multiplied by 12)
    c4 = 12.0 * (-17189583356238.0
                  + 1017565274880.0 * EULER_GAMMA
                  + 6546104892250.0 * nu
                  + 5184425896650.0 * nu2
                  + 597064314000.0 * nu3
                  + 121275.0 * PI**2 * (-2744576.0 + 790193.0 * nu)
                  - 224809728380160.0 * LOG2
                  + 42510781647180.0 * LOG3
                  + 72413085937500.0 * LOG5)

    # e^6 coefficient (multiplied by 4)
    c6 = 4.0 * (4502397710583.0
                + 122366946240.0 * EULER_GAMMA
                + 11968410346200.0 * nu
                + 1488261297600.0 * nu2
                + 392650720000.0 * nu3
                + 363825.0 * PI**2 * (-110016.0 + 9061.0 * nu)
                - 584404081430400.0 * LOG2
                + 159670846150200.0 * LOG3
                + 144826171875000.0 * LOG5)

    # e^8 coefficient (multiplied by 175)
    c8 = 175.0 * e8 * (46932564429.0
                        - 4051745280.0 * nu
                        - 2937623040.0 * nu2
                        + 178984960.0 * nu3)

    # sqrt(1-e^2) sub-block
    s0 = 64.0 * (-637780831131.0
                  + 109482468480.0 * EULER_GAMMA
                  + 380870611275.0 * nu
                  + 30985883775.0 * nu2
                  + 12481353500.0 * nu3
                  - 727650.0 * PI**2 * (49216.0 + 17917.0 * nu)
                  + 102648712320.0 * LOG2
                  + 116761130640.0 * LOG3)

    s2 = 160.0 * (-1094927702262.0
                   + 158429246976.0 * EULER_GAMMA
                   - 239784537300.0 * nu
                   + 275391017055.0 * nu2
                   + 37285931375.0 * nu3
                   + 291060.0 * PI**2 * (-178048.0 + 28413.0 * nu)
                   + 1587595451904.0 * LOG2
                   - 572129540136.0 * LOG3)

    s4 = 12.0 * (-13196290007886.0
                  + 1017565274880.0 * EULER_GAMMA
                  + 5406255720250.0 * nu
                  + 5078720666250.0 * nu2
                  + 597064314000.0 * nu3
                  + 121275.0 * PI**2 * (-2744576.0 + 865223.0 * nu)
                  - 224809728380160.0 * LOG2
                  + 42510781647180.0 * LOG3
                  + 72413085937500.0 * LOG5)

    s6 = 4.0 * (566190348111.0
                + 122366946240.0 * EULER_GAMMA
                + 9153699093000.0 * nu
                + 3117359044800.0 * nu2
                + 392650720000.0 * nu3
                + 3274425.0 * PI**2 * (-12224.0 + 6519.0 * nu)
                - 584404081430400.0 * LOG2
                + 159670846150200.0 * LOG3
                + 144826171875000.0 * LOG5)

    s8 = 175.0 * e8 * (10539419949.0
                        + 3285893952.0 * nu
                        + 1302605568.0 * nu2
                        + 178984960.0 * nu3)

    sqrt_block = sqrt_oome2 * (s0 + s2 * e2 + s4 * e4 + s6 * e6 + s8)

    # Log term
    log_poly = (24608.0 + 89024.0 * e2 + 42884.0 * e4 + 1719.0 * e6)
    log_block = 284739840.0 * log_poly * (1.0 + sqrt_oome2) * log_term

    pn3_num = c8 + c0 + c2 * e2 + c4 * e4 + c6 * e6 + sqrt_block + log_block

    term_3pn = (-x**3 * pn3_num
                / (169885900800.0 * oome2**3 * (1.0 + sqrt_oome2)))

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    bracket = (term_0pn + tail_15pn + so_15pn + term_1pn
               + tail_25pn + so_25pn_tail + term_2pn + ss_2pn
               + tail_25pn  # already included above -- remove duplicate
               + so_25pn + ss_3pn + term_3pn)

    # Fix: tail_25pn was added twice above.  Correct assembly:
    bracket = (term_0pn + tail_15pn + so_15pn + term_1pn
               + tail_25pn + so_25pn_tail + term_2pn + ss_2pn
               + so_25pn + ss_3pn + term_3pn)

    return prefactor * bracket


# ============================================================================
# dx/dt  --  eccentricity-resummed (aligned-spin)
# Source: EOB_fluxes.dat.m lines 3115-3218
# ============================================================================

def xdot_resum(e, x, nu, chi_S=0.0, chi_A=0.0, delta=None,
               kappa_S=0.0, kappa_A=0.0):
    """
    Resummed dx/dt for aligned-spin eccentric binaries at 3PN order.

    Parameters
    ----------
    e : float
        Keplerian eccentricity.
    x : float
        PN frequency parameter.
    nu : float
        Symmetric mass ratio.
    chi_S : float
        Symmetric spin combination.
    chi_A : float
        Antisymmetric spin combination.
    delta : float or None
        Mass difference ratio.
    kappa_S : float
        Symmetric tidal parameter (0 for BBH).
    kappa_A : float
        Antisymmetric tidal parameter (0 for BBH).

    Returns
    -------
    float
        Time derivative dx/dt.
    """
    delta = _get_delta(nu, delta)

    e2 = e * e
    e4 = e2 * e2
    e6 = e4 * e2
    e8 = e4 * e4
    e10 = e8 * e2
    nu2 = nu * nu
    nu3 = nu2 * nu
    oome2 = 1.0 - e2
    sqrt_oome2 = np.sqrt(oome2)
    sqrt_x = np.sqrt(x)

    chiA2 = chi_A * chi_A
    chiS2 = chi_S * chi_S
    chiAS = chi_A * chi_S

    # Leading prefactor: 2*x^5*nu/3
    prefactor = 2.0 * x**5 * nu / 3.0

    # ------------------------------------------------------------------
    # 0PN
    # ------------------------------------------------------------------
    term_0pn = (96.0 + 292.0 * e2 + 37.0 * e4) / (5.0 * oome2**3.5)

    # ------------------------------------------------------------------
    # 1.5PN tail (non-spin)
    # ------------------------------------------------------------------
    tail_15pn = (PI * x**1.5
                 * (36864.0 + 264000.0 * e2 + 188880.0 * e4 + 10007.0 * e6)
                 / (480.0 * oome2**5))

    # ------------------------------------------------------------------
    # 1.5PN SO: x^(3/2) * (...) / (30*(1-e^2)^5)
    #   Lines 3131-3134
    # ------------------------------------------------------------------
    so_15pn_chiA = ((-5424.0 - 12536.0 * e2 + 2602.0 * e4
                     + 747.0 * e6) * delta * chi_A)
    so_15pn_chiS = ((-9.0 * e6 * (-83.0 + 8.0 * nu)
                     + 48.0 * (-113.0 + 76.0 * nu)
                     + 8.0 * e2 * (-1567.0 + 1670.0 * nu)
                     + e4 * (2602.0 + 4072.0 * nu)) * chi_S)
    so_15pn = x**1.5 * (so_15pn_chiA + so_15pn_chiS) / (30.0 * oome2**5)

    # ------------------------------------------------------------------
    # 1PN (non-spin)
    # ------------------------------------------------------------------
    pn1_num = (16.0 * (743.0 + 924.0 * nu)
               + e6 * (6931.0 + 2072.0 * nu)
               + 14.0 * e4 * (7079.0 + 3690.0 * nu)
               + 8.0 * e2 * (15411.0 + 11158.0 * nu))
    term_1pn = -x * pn1_num / (280.0 * oome2**4.5)

    # ------------------------------------------------------------------
    # 2.5PN tail (non-spin)
    # ------------------------------------------------------------------
    tail_25pn_num = (7.0 * e8 * (-151281.0 + 10007.0 * nu)
                     + 576.0 * (4159.0 + 15876.0 * nu)
                     + 576.0 * e2 * (115991.0 + 171104.0 * nu)
                     + 4.0 * e6 * (17257369.0 + 7471473.0 * nu)
                     + 9.0 * e4 * (18599341.0 + 14964748.0 * nu))
    tail_25pn = -PI * x**2.5 * tail_25pn_num / (20160.0 * oome2**6)

    # ------------------------------------------------------------------
    # 2.5PN SO tail: Pi*x^3 * (...) / (720*(1-e^2)^(13/2))
    #   Lines 3135-3140
    # ------------------------------------------------------------------
    so_25pn_tail_chiA = (-4.0 * (129600.0 + 704832.0 * e2
                                  - 29742.0 * e4 - 339871.0 * e6
                                  + 147.0 * e8) * delta * chi_A)
    so_25pn_tail_chiS = ((294.0 * e8 * (-2.0 + nu)
                          + 2304.0 * (-225.0 + 148.0 * nu)
                          + 192.0 * e2 * (-14684.0 + 13789.0 * nu)
                          + 24.0 * e4 * (4957.0 + 101614.0 * nu)
                          + e6 * (1359484.0 + 214925.0 * nu)) * chi_S)
    so_25pn_tail = (PI * x**3
                    * (so_25pn_tail_chiA + so_25pn_tail_chiS)
                    / (720.0 * oome2**6.5))

    # ------------------------------------------------------------------
    # 2PN (non-spin)
    # ------------------------------------------------------------------
    pn2_num = (6048.0 * sqrt_oome2
               * (-48.0 - 250.0 * e2 + 219.0 * e4 + 79.0 * e6)
               * (-5.0 + 2.0 * nu)
               + 3.0 * e8 * (734703.0 + 290664.0 * nu + 58016.0 * nu2)
               + 32.0 * (-11257.0 + 141093.0 * nu + 59472.0 * nu2)
               + 6.0 * e6 * (5905155.0 + 6204657.0 * nu + 1292312.0 * nu2)
               + 16.0 * e2 * (-2678686.0 + 5601690.0 * nu
                              + 1331295.0 * nu2)
               + 12.0 * e4 * (896914.0 + 11637378.0 * nu
                              + 2585233.0 * nu2))
    term_2pn = x**2 * pn2_num / (30240.0 * oome2**5.5)

    # ------------------------------------------------------------------
    # 2PN SS: x^2 * (...) / (40*(1-e^2)^(11/2))
    #   Lines 3141-3147
    # ------------------------------------------------------------------
    ss_2pn_num = (
        8.0 * (480.0 + 2064.0 * e2 + 1064.0 * e4 + 33.0 * e6)
        * (delta * kappa_A + kappa_S - 2.0 * kappa_S * nu)
        + (48.0 * (81.0 - 320.0 * nu)
           + 3.0 * e6 * (-199.0 + 832.0 * nu)
           - 8.0 * e2 * (-865.0 + 3232.0 * nu)
           + 2.0 * e4 * (-1969.0 + 8704.0 * nu)) * chiA2
        + 2.0 * (3888.0 + 6920.0 * e2 - 3938.0 * e4
                 - 597.0 * e6) * delta * chiAS
        + (48.0 * (81.0 - 4.0 * nu)
           - 3.0 * e6 * (199.0 + 36.0 * nu)
           - 8.0 * e2 * (-865.0 + 228.0 * nu)
           - 2.0 * e4 * (1969.0 + 828.0 * nu)) * chiS2
    )
    ss_2pn = x**2 * ss_2pn_num / (40.0 * oome2**5.5)

    # ------------------------------------------------------------------
    # 2.5PN SO (non-tail): -x^(5/2) * (...) / (6720*(1-e^2)^6)
    #   Lines 3148-3158
    # ------------------------------------------------------------------
    so_25pn_num = (
        delta * (4008832.0 - 6230784.0 * nu
                 - 224.0 * e4 * (80924.0 + 13833.0 * nu)
                 + 9.0 * e8 * (125467.0 + 27916.0 * nu)
                 - 32.0 * e2 * (642396.0 + 696941.0 * nu)
                 + e6 * (9793328.0 + 4249140.0 * nu)) * chi_A
        + (-9.0 * e8 * (-125467.0 - 41528.0 * nu + 8008.0 * nu2)
           + 128.0 * (31319.0 - 91900.0 * nu + 26544.0 * nu2)
           + 448.0 * e4 * (-40462.0 + 79817.0 * nu + 36774.0 * nu2)
           + 4.0 * e6 * (2448332.0 + 3133943.0 * nu + 337722.0 * nu2)
           + 32.0 * e2 * (-642396.0 - 125759.0 * nu
                          + 632758.0 * nu2)) * chi_S
        + oome2**1.5 * (
            -3584.0 * e2 * (304.0 + 121.0 * e2) * delta
            * (-3.0 + nu) * chi_A
            + 1792.0 * e2 * (304.0 + 121.0 * e2)
            * (6.0 - 8.0 * nu + nu2) * chi_S
        )
    )
    so_25pn = -x**2.5 * so_25pn_num / (6720.0 * oome2**6)

    # ------------------------------------------------------------------
    # 3PN SS: -x^3 * (...) / (20160*(1-e^2)^(13/2))
    #   Lines 3159-3188
    # ------------------------------------------------------------------
    # kappa terms
    ss_3pn_kappa = (
        6.0 * delta * kappa_A * (
            192.0 * (-8963.0 + 13706.0 * nu)
            + 3.0 * e8 * (41268.0 + 24115.0 * nu)
            + 96.0 * e2 * (55623.0 + 164080.0 * nu)
            + 4.0 * e6 * (1175865.0 + 952294.0 * nu)
            + 8.0 * e4 * (1721169.0 + 2248456.0 * nu)
        )
        - 6.0 * kappa_S * (
            192.0 * (8963.0 - 31632.0 * nu + 14784.0 * nu2)
            + 3.0 * e8 * (-41268.0 + 58421.0 * nu + 44576.0 * nu2)
            + 96.0 * e2 * (-55623.0 - 52834.0 * nu + 223944.0 * nu2)
            + 8.0 * e4 * (-1721169.0 + 1193882.0 * nu
                          + 3223136.0 * nu2)
            + e6 * (-4703460.0 + 5597744.0 * nu + 5745488.0 * nu2)
        )
    )
    # chiA^2 terms
    ss_3pn_chiA2 = (
        (-576.0 * (55817.0 - 243029.0 * nu + 59136.0 * nu2)
         - 9.0 * e8 * (503031.0 - 2069104.0 * nu + 112000.0 * nu2)
         - 12.0 * e6 * (2460015.0 - 10429817.0 * nu + 589232.0 * nu2)
         - 32.0 * e2 * (-593801.0 + 851591.0 * nu + 4835712.0 * nu2)
         - 8.0 * e4 * (-8736761.0 + 31182647.0 * nu
                       + 8270976.0 * nu2))
        * chiA2
    )
    # delta*chiA*chiS terms
    ss_3pn_chiAS = (
        2.0 * delta * (
            576.0 * (-55817.0 + 68481.0 * nu)
            + 9.0 * e8 * (-503031.0 + 141694.0 * nu)
            + 12.0 * e6 * (-2460015.0 + 897617.0 * nu)
            + 32.0 * e2 * (593801.0 + 3657185.0 * nu)
            + 8.0 * e4 * (8736761.0 + 6220865.0 * nu)
        ) * chiAS
    )
    # chiS^2 terms
    ss_3pn_chiS2 = (
        (-576.0 * (55817.0 - 117201.0 * nu + 33460.0 * nu2)
         - 9.0 * e8 * (503031.0 - 226408.0 * nu + 57792.0 * nu2)
         - 12.0 * e6 * (2460015.0 - 1205477.0 * nu + 797748.0 * nu2)
         - 32.0 * e2 * (-593801.0 - 5790757.0 * nu + 1987972.0 * nu2)
         - 8.0 * e4 * (-8736761.0 - 8677333.0 * nu
                       + 5503540.0 * nu2))
        * chiS2
    )
    # sqrt(1-e^2) block inside 3PN SS
    poly_48 = 48.0 - 358.0 * e2 + 147.0 * e4 + 163.0 * e6
    ss_3pn_sqrt = sqrt_oome2 * (
        -1344.0 * poly_48 * (delta * kappa_A * (-14.0 + 5.0 * nu)
                             + kappa_S * (-14.0 + 33.0 * nu
                                          - 6.0 * nu2))
        + 2688.0 * poly_48 * (11.0 - 46.0 * nu + 6.0 * nu2) * chiA2
        - 5376.0 * poly_48 * delta * (-11.0 + 9.0 * nu) * chiAS
        + 2688.0 * poly_48 * (11.0 - 16.0 * nu + 4.0 * nu2) * chiS2
    )

    ss_3pn_total = (ss_3pn_kappa + ss_3pn_chiA2 + ss_3pn_chiAS
                    + ss_3pn_chiS2 + ss_3pn_sqrt)
    ss_3pn = -x**3 * ss_3pn_total / (20160.0 * oome2**6.5)

    # ------------------------------------------------------------------
    # 3PN (non-spin): -x^3 * (...) / (2794176000*(1-e^2)^(13/2))
    # ------------------------------------------------------------------
    log_arg = 2.0 * oome2 * sqrt_x / (1.0 + sqrt_oome2)
    log_term = np.log(log_arg)

    # e^10 term
    c10 = 175.0 * e10 * (3116030391.0 + 1005041664.0 * nu
                          + 405402624.0 * nu2 + 58347520.0 * nu3)

    # sqrt(1-e^2) block inside 3PN (non-spin)
    sq_inner = (16.0 * (19954466.0 + 75.0 * (-19748.0 + 861.0 * PI**2) * nu
                        - 1990800.0 * nu2)
                + e2 * (2474694912.0 + (964412000.0
                        - 7705950.0 * PI**2) * nu
                        - 341678400.0 * nu2)
                + 3.0 * e4 * (652068196.0
                              + 175.0 * (-960472.0 + 6027.0 * PI**2) * nu
                              + 39620000.0 * nu2)
                + e6 * (-558162656.0
                        + 25.0 * (-18306272.0 + 140343.0 * PI**2) * nu
                        + 239618400.0 * nu2)
                + 300.0 * e8 * (-428445.0 + 70634.0 * nu
                                + 50176.0 * nu2))
    sqrt_3pn = 11088.0 * sqrt_oome2 * sq_inner

    # e^0 (multiplied by 640)
    d0 = 640.0 * (-15399771333.0
                   + 1366751232.0 * EULER_GAMMA
                   + 22047056185.0 * nu
                   + 501236505.0 * nu2
                   + 181265700.0 * nu3
                   - 436590.0 * PI**2 * (1024.0 + 1845.0 * nu)
                   + 2733502464.0 * LOG2)

    # e^2 (multiplied by 32)
    d2 = 32.0 * (-3181561351866.0
                  + 387246182400.0 * EULER_GAMMA
                  - 9693202750.0 * nu
                  + 636423999750.0 * nu2
                  + 75206939500.0 * nu3
                  - 1819125.0 * PI**2 * (69632.0 + 9717.0 * nu)
                  + 72893399040.0 * LOG2
                  + 700566783840.0 * LOG3)

    # e^4 (multiplied by 8)
    d4 = 8.0 * (-28536072962442.0
                 + 2944779425280.0 * EULER_GAMMA
                 + 1156154672750.0 * nu
                 + 7217960245650.0 * nu2
                 + 913931903500.0 * nu3
                 + 363825.0 * PI**2 * (-2647552.0 + 591261.0 * nu)
                 + 65437201589760.0 * LOG2
                 - 31525505272800.0 * LOG3)

    # e^6 (multiplied by 12)
    d6 = 12.0 * (664772613120.0 * EULER_GAMMA
                  + 121275.0 * PI**2 * (-1793024.0 + 783633.0 * nu)
                  + 2.0 * (-3371321595729.0
                           + 2173303676775.0 * nu
                           + 1510891888275.0 * nu2
                           + 207243883000.0 * nu3
                           - 220951471910400.0 * LOG2
                           + 53938777308570.0 * LOG3
                           + 60344238281250.0 * LOG5))

    # e^8 (multiplied by 14)
    d8 = 14.0 * (358019945973.0
                  + 18121656960.0 * EULER_GAMMA
                  + 906482016000.0 * nu
                  + 351807667200.0 * nu2
                  + 50935808000.0 * nu3
                  + 92619450.0 * PI**2 * (-64.0 + 41.0 * nu)
                  + 36243313920.0 * LOG2)

    # Log term
    log_poly = (3072.0 + 43520.0 * e2 + 82736.0 * e4
                + 28016.0 * e6 + 891.0 * e8)
    log_block = 284739840.0 * log_poly * log_term

    pn3_num = (c10 + sqrt_3pn + d0 + d2 * e2 + d4 * e4 + d6 * e6
               + d8 * e8 + log_block)

    term_3pn = -x**3 * pn3_num / (2794176000.0 * oome2**6.5)

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    bracket = (term_0pn + tail_15pn + so_15pn + term_1pn
               + tail_25pn + so_25pn_tail + term_2pn + ss_2pn
               + so_25pn + ss_3pn + term_3pn)

    return prefactor * bracket


# ============================================================================
# dzeta/dt  --  instantaneous (aligned-spin)
# Source: EOB_fluxes.dat.m lines 3226-3304
# ============================================================================

def zetadot_func(e, x, zeta, nu, chi_S=0.0, chi_A=0.0, delta=None,
                 kappa_S=0.0, kappa_A=0.0):
    """
    Instantaneous dzeta/dt for aligned-spin eccentric binaries at 3PN.

    This is NOT orbit-averaged -- it depends explicitly on zeta.

    Parameters
    ----------
    e : float
        Keplerian eccentricity.
    x : float
        PN frequency parameter.
    zeta : float
        Relativistic anomaly.
    nu : float
        Symmetric mass ratio.
    chi_S : float
        Symmetric spin combination.
    chi_A : float
        Antisymmetric spin combination.
    delta : float or None
        Mass difference ratio.
    kappa_S : float
        Symmetric tidal parameter (0 for BBH).
    kappa_A : float
        Antisymmetric tidal parameter (0 for BBH).

    Returns
    -------
    float
        Time derivative dzeta/dt.
    """
    delta = _get_delta(nu, delta)

    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    e6 = e4 * e2
    nu2 = nu * nu
    oome2 = 1.0 - e2
    sqrt_oome2 = np.sqrt(oome2)

    chiA2 = chi_A * chi_A
    chiS2 = chi_S * chi_S
    chiAS = chi_A * chi_S

    cos_z = np.cos(zeta)
    sin_z = np.sin(zeta)
    cos_2z = np.cos(2.0 * zeta)
    cos_3z = np.cos(3.0 * zeta)
    cos_4z = np.cos(4.0 * zeta)

    f = 1.0 + e * cos_z       # (1 + e*cos(zeta))
    f2 = f * f

    # ------------------------------------------------------------------
    # 0PN: x^(3/2) * f^2 / (1-e^2)^(3/2)
    # ------------------------------------------------------------------
    term_0pn = x**1.5 * f2 / oome2**1.5

    # ------------------------------------------------------------------
    # 1PN: -3 * x^(5/2) * f^2 * (1 + e^2 + e*cos(z)) / (1-e^2)^(5/2)
    # ------------------------------------------------------------------
    term_1pn = -3.0 * x**2.5 * f2 * (1.0 + e2 + e * cos_z) / oome2**2.5

    # ------------------------------------------------------------------
    # 1.5PN SO: x^3 * f^2 * (2*delta*chiA + (2-nu)*chiS)
    #           * (2 + e^2 + e*cos(z)) / (1-e^2)^3
    #   Line 3229-3230
    # ------------------------------------------------------------------
    so_15pn = (x**3 * f2
               * (2.0 * delta * chi_A + (2.0 - nu) * chi_S)
               * (2.0 + e2 + e * cos_z) / oome2**3)

    # ------------------------------------------------------------------
    # 2PN SS: -x^(7/2) * f^2 * (...) / (2*(1-e^2)^(7/2))
    #   Lines 3231-3236
    # ------------------------------------------------------------------
    ss_2pn_inner = (
        (delta * kappa_A + kappa_S - 2.0 * kappa_S * nu)
        * (3.0 + e2 + e * cos_z)
        - (-1.0 + 4.0 * nu) * chiA2 * (3.0 + 2.0 * e2
                                         + 2.0 * e * cos_z)
        + 2.0 * delta * chiAS * (3.0 + 2.0 * e2 + 2.0 * e * cos_z)
        + chiS2 * (3.0 + 2.0 * e2 + 2.0 * e * cos_z)
    )
    ss_2pn = -x**3.5 * f2 * ss_2pn_inner / (2.0 * oome2**3.5)

    # ------------------------------------------------------------------
    # 2PN (non-spin): -x^(7/2) * f^2 * (...) / (4*(1-e^2)^(7/2))
    #   Lines 3237-3240
    # ------------------------------------------------------------------
    inner_2pn = (48.0 + 4.0 * e4 * (-6.0 + nu) - 40.0 * nu
                 - 16.0 * e2 * (5.0 + nu)
                 + 6.0 * sqrt_oome2 * (-5.0 + 2.0 * nu)
                 + 4.0 * e * (1.0 + e2 * (-15.0 + nu)
                              - 8.0 * nu) * cos_z
                 - 3.0 * e2 * (1.0 + 2.0 * nu) * cos_2z)
    term_2pn = -x**3.5 * f2 * inner_2pn / (4.0 * oome2**3.5)

    # ------------------------------------------------------------------
    # 2.5PN SO: -x^4 * f^2 * (...) / (64*(1-e^2)^4)
    #   Lines 3241-3253
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 3PN SS: -x^(9/2) * f^2 * (...) / (24*(1-e^2)^(9/2))
    #   Lines 3254-3289
    # ------------------------------------------------------------------
    # kappa terms (constant + cos terms)
    kA_const = (420.0 - 36.0 * e2 - 84.0 * e4) * delta * kappa_A
    kS_const = ((420.0 - 36.0 * e2 - 84.0 * e4) * kappa_S
                + (-264.0 - 188.0 * e2 + 8.0 * e4) * delta * kappa_A * nu
                + (-1104.0 - 116.0 * e2 + 176.0 * e4) * kappa_S * nu
                + (192.0 + 184.0 * e2 - 16.0 * e4) * kappa_S * nu2)
    # Wait -- let me re-read the structure more carefully. The Mathematica
    # expression lists all terms additively. Let me group them properly.

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

    # (1-e^2)^(3/2) block (kappa + chi^2 terms)
    ss_3pn_sqrt_block = oome2**1.5 * (
        12.0 * (delta * kappa_A * (-14.0 + 5.0 * nu)
                + kappa_S * (-14.0 + 33.0 * nu - 6.0 * nu2))
        - 24.0 * (11.0 - 46.0 * nu + 6.0 * nu2) * chiA2
        + 48.0 * delta * (-11.0 + 9.0 * nu) * chiAS
        - 24.0 * (11.0 - 16.0 * nu + 4.0 * nu2) * chiS2
    )

    # kappa cos(z) terms
    ss_3pn_kappa_cos = -e * (
        delta * kappa_A * (e2 * (111.0 + 19.0 * nu)
                           + 4.0 * (-72.0 + 53.0 * nu))
        - kappa_S * (e2 * (-111.0 + 203.0 * nu + 38.0 * nu2)
                     + 4.0 * (72.0 - 197.0 * nu + 58.0 * nu2))
    ) * cos_z

    # kappa cos(2z) terms
    ss_3pn_kappa_cos2 = -6.0 * e2 * (
        delta * kappa_A * (-18.0 + 11.0 * nu)
        + kappa_S * (-18.0 + 47.0 * nu - 18.0 * nu2)
    ) * cos_2z

    # kappa cos(3z) terms
    ss_3pn_kappa_cos3 = (
        15.0 * e3 * delta * kappa_A
        + 15.0 * e3 * kappa_S
        - 9.0 * e3 * delta * kappa_A * nu
        - 39.0 * e3 * kappa_S * nu
        + 18.0 * e3 * kappa_S * nu2
    ) * cos_3z

    # delta*chiA*chiS terms
    ss_3pn_chiAS = -2.0 * delta * chiAS * (
        -468.0 + 948.0 * e2 + 336.0 * e4
        + 684.0 * nu + 244.0 * e2 * nu - 112.0 * e4 * nu
        - e * (180.0 - 580.0 * nu + e2 * (-576.0 + 79.0 * nu)) * cos_z
        + 12.0 * e2 * (-1.0 + 16.0 * nu) * cos_2z
        + 27.0 * e3 * nu * cos_3z
    )

    # chiS^2 terms
    ss_3pn_chiS2 = chiS2 * (
        468.0 - 948.0 * e2 - 336.0 * e4
        - 1188.0 * nu - 340.0 * e2 * nu + 208.0 * e4 * nu
        + 336.0 * nu2 + 96.0 * e2 * nu2 - 48.0 * e4 * nu2
        - e3 * (576.0 - 169.0 * nu + 45.0 * nu2) * cos_z
        + 4.0 * e * (45.0 - 247.0 * nu + 63.0 * nu2) * cos_z
        + 12.0 * e2 * (1.0 - 27.0 * nu + 6.0 * nu2) * cos_2z
        - 45.0 * e3 * nu * cos_3z + 9.0 * e3 * nu2 * cos_3z
    )

    # chiA^2 terms
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

    # ------------------------------------------------------------------
    # 3PN (non-spin): x^(9/2) * f^2 * (...) / (384*(1-e^2)^(9/2))
    #   Lines 3290-3302
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2.5PN RR term: lines 3303-3304
    # ------------------------------------------------------------------
    term_rr = (-e * (608.0 + 2370.0 * e2 + 5635.0 * e4) * x**4 * nu
               * (2.0 + e * cos_z) * sin_z / 30.0)

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    return (term_0pn + term_1pn + so_15pn + ss_2pn + term_2pn
            + so_25pn + ss_3pn + term_3pn + term_rr)


# ============================================================================
# ODE system and integrator
# ============================================================================

def _rhs(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
    """ODE right-hand side: dy/dt = [de/dt, dx/dt, dzeta/dt]."""
    e_val, x_val, zeta_val = y
    if e_val < 0 or x_val <= 0 or x_val > 1.0 or e_val >= 1.0:
        return [0.0, 0.0, 0.0]
    de = edot_resum(e_val, x_val, nu, chi_S, chi_A, delta,
                    kappa_S, kappa_A)
    dx = xdot_resum(e_val, x_val, nu, chi_S, chi_A, delta,
                    kappa_S, kappa_A)
    dz = zetadot_func(e_val, x_val, zeta_val, nu, chi_S, chi_A, delta,
                      kappa_S, kappa_A)
    return [de, dx, dz]


def _kerr_isco_radius(a):
    """Boyer-Lindquist ISCO radius for dimensionless spin a."""
    a = np.clip(a, -0.9999, 0.9999)
    Z1 = 1.0 + (1 - a**2)**(1/3.0) * ((1 + a)**(1/3.0) + (1 - a)**(1/3.0))
    Z2 = np.sqrt(3 * a**2 + Z1**2)
    return 3 + Z2 - np.sign(a) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))


def _compute_x_isco(chi1, chi2, q):
    """
    Estimate x_ISCO from the Kerr ISCO of the final BH.

    Uses the leading-order effective spin to estimate the final spin,
    then computes the ISCO frequency x = Omega_ISCO^{2/3}.
    """
    nu = q / (1.0 + q)**2
    # Leading-order estimate of final spin: test-particle limit
    a_eff = (chi1 + chi2 * q**2) / (1.0 + q)**2
    r_isco = _kerr_isco_radius(a_eff)
    # Kepler: Omega_ISCO = 1 / (r^{3/2} + a)
    omega_isco = 1.0 / (r_isco**1.5 + a_eff)
    x_isco = omega_isco**(2.0 / 3.0)
    return x_isco


def _event_x_large(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
    """Terminate when x > 0.5 (PN breakdown)."""
    return 0.5 - y[1]

_event_x_large.terminal = True
_event_x_large.direction = -1


def _event_e_negative(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
    """Terminate when e < 1e-10."""
    return y[0] - 1e-10

_event_e_negative.terminal = True
_event_e_negative.direction = -1


def _event_oome2_small(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
    """Terminate when 1-e^2 < 0.01."""
    return (1.0 - y[0]**2) - 0.01

_event_oome2_small.terminal = True
_event_oome2_small.direction = -1


def integrate_eob_eccentric_dynamics(q, chi1=0.0, chi2=0.0, e0=0.1,
                                     zeta0=0.0, t_eval=None,
                                     omega0=0.0085, t_end=None,
                                     rtol=1e-10, atol=1e-12,
                                     kappa1=1.0, kappa2=1.0,
                                     x_stop=None):
    """
    Integrate aligned-spin eccentric dynamics ODEs at 3PN order.

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
    t_eval : array-like, optional
        Times at which to store solution. If None, solver chooses.
    omega0 : float
        Initial orbit-averaged orbital frequency (default 0.0085).
    t_end : float, optional
        End time. If None, estimated from circular inspiral time.
    rtol, atol : float
        Solver tolerances.
    kappa1, kappa2 : float
        Spin-induced quadrupole parameters (= 1 for black holes).
    x_stop : float or str, optional
        Stopping value for x. Options:
        - None: stop at x=0.5 (default, PN breakdown)
        - 'isco': compute x_ISCO from Kerr ISCO and stop there
        - float: stop at this specific x value

    Returns
    -------
    dict with keys:
        t : array, time
        e : array, eccentricity
        x : array, PN parameter
        zeta : array, relativistic anomaly
        x_isco : float, estimated x at ISCO
        success : bool
        message : str
        sol : OdeResult
    """
    nu = q / (1.0 + q)**2
    delta_val = (q - 1.0) / (q + 1.0)  # (m1 - m2)/M with m1 >= m2

    chi_S = 0.5 * (chi1 + chi2)
    chi_A = 0.5 * (chi1 - chi2)

    # Tidal / spin-induced quadrupole combinations
    # kappa_S = (kappa1 + kappa2) / 2,  kappa_A = (kappa1 - kappa2) / 2
    # For BBH: kappa1 = kappa2 = 1  =>  kappa_S = 1, kappa_A = 0
    # But the terms in the flux always appear as
    #   delta*kappa_A + kappa_S - 2*kappa_S*nu   etc.
    # which for BH (kappa_S=1, kappa_A=0) gives 1 - 2*nu.
    # In the SEOBNRv5EHM convention the kappa terms are defined so that
    # they VANISH for BBHs.  The Mathematica expressions already separate
    # the kappa-dependent from kappa-independent pieces such that setting
    # kappa_S = kappa_A = 0 recovers the BBH result.  So we follow the
    # same convention:
    #   \[Kappa]S = (kappa1 + kappa2)/2 - 1
    #   \[Kappa]A = (kappa1 - kappa2)/2
    # For BBH this gives kappa_S = kappa_A = 0.
    # Actually, re-reading the Mathematica file more carefully, the
    # kappa terms multiply (delta*kappaA + kappaS - 2*kappaS*nu) which
    # for BH values kappa1=kappa2=1 gives (0 + 1 - 2*nu) = 1-2*nu,
    # but those terms are NOT zero.  However, the user's instructions say
    # "For BBHs, kappa1 = kappa2 = 1, so kappaS = kappaA = 0 and those
    # terms vanish."  This means the convention is:
    #   kappaS = 0 for BBH, kappaA = 0 for BBH
    # and the chi^2 terms that persist for BBH are in the chiA^2, chiS^2
    # pieces that do NOT multiply kappa.  So kappa_S and kappa_A in our
    # code are zero for BBH and nonzero only for NS.
    kappa_S_val = 0.5 * (kappa1 + kappa2) - 1.0
    kappa_A_val = 0.5 * (kappa1 - kappa2)

    # Compute x_ISCO for reference and optional stopping
    x_isco = _compute_x_isco(chi1, chi2, q)

    # Resolve x_stop
    if x_stop == 'isco':
        x_stop_val = x_isco
    elif x_stop is not None:
        x_stop_val = float(x_stop)
    else:
        x_stop_val = 0.5  # default: PN breakdown

    # Build x termination event with the resolved x_stop value
    def _event_x_stop(t, y, nu, chi_S, chi_A, delta, kappa_S, kappa_A):
        return x_stop_val - y[1]
    _event_x_stop.terminal = True
    _event_x_stop.direction = -1

    x0 = omega0**(2.0 / 3.0)

    # Estimate inspiral time from circular 0PN: t_merge ~ 5/(256*nu*x0^4)
    if t_end is None:
        t_merge_est = 5.0 / (256.0 * nu * x0**4)
        t_end = 1.05 * t_merge_est  # go slightly past estimated merger

    t_span = (0.0, t_end)
    y0 = [e0, x0, zeta0]

    logger.info("Integrating EOB eccentric dynamics: q=%.4f, chi1=%.4f, "
                "chi2=%.4f, e0=%.4f, x0=%.6f, x_stop=%.4f, x_isco=%.4f",
                q, chi1, chi2, e0, x0, x_stop_val, x_isco)

    sol = solve_ivp(
        _rhs, t_span, y0,
        args=(nu, chi_S, chi_A, delta_val, kappa_S_val, kappa_A_val),
        method='DOP853',
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        dense_output=True,
        events=[_event_x_stop, _event_e_negative, _event_oome2_small],
        max_step=2.0 * PI / (10.0 * omega0),
    )

    logger.info("Integration finished: success=%s, message=%s",
                sol.success, sol.message)

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


# ============================================================================
# Sanity checks
# ============================================================================

def sanity_check():
    """Run basic sanity checks on the evolution equations."""
    logger.info("Running sanity checks...")

    nu = 0.25  # equal mass
    x = 0.01   # low frequency

    # ------------------------------------------------------------------
    # Check 1: edot(e=0) should be 0 (prefactor is proportional to e)
    # ------------------------------------------------------------------
    de = edot_resum(0.0, x, nu)
    assert abs(de) < 1e-30, f"edot(e=0) = {de}, expected 0"
    logger.info("  [PASS] edot(e=0, x=%s, nu=%s) = %.2e", x, nu, de)

    # ------------------------------------------------------------------
    # Check 2: xdot(e=0) at 0PN should be 64*nu*x^5/5
    # ------------------------------------------------------------------
    dx_full = xdot_resum(0.0, x, nu)
    dx_0pn = 64.0 * nu * x**5 / 5.0
    rel_err = abs(dx_full - dx_0pn) / abs(dx_0pn)
    assert rel_err < 0.05, (
        f"xdot(e=0) = {dx_full}, 0PN = {dx_0pn}, rel_err = {rel_err}")
    logger.info("  [PASS] xdot(e=0, x=%s, nu=%s) = %.6e  "
                "(0PN: %.6e, rel_err: %.4f)", x, nu, dx_full, dx_0pn,
                rel_err)

    # ------------------------------------------------------------------
    # Check 3: zetadot(e=0, 0PN) ~ x^(3/2)
    # ------------------------------------------------------------------
    dz = zetadot_func(0.0, x, 0.0, nu)
    dz_0pn = x**1.5
    rel_err_z = abs(dz - dz_0pn) / abs(dz_0pn)
    assert rel_err_z < 0.1, (
        f"zetadot(e=0) = {dz}, 0PN = {dz_0pn}, rel_err = {rel_err_z}")
    logger.info("  [PASS] zetadot(e=0, x=%s, nu=%s) = %.6e  "
                "(0PN: %.6e, rel_err: %.4f)", x, nu, dz, dz_0pn,
                rel_err_z)

    # ------------------------------------------------------------------
    # Check 4: edot should be negative (eccentricity decreases)
    # ------------------------------------------------------------------
    de2 = edot_resum(0.1, x, nu)
    assert de2 < 0, f"edot(e=0.1) = {de2}, expected negative"
    logger.info("  [PASS] edot(e=0.1, x=%s, nu=%s) = %.6e  (negative)", x,
                nu, de2)

    # ------------------------------------------------------------------
    # Check 5: xdot should be positive (frequency increases)
    # ------------------------------------------------------------------
    dx2 = xdot_resum(0.1, x, nu)
    assert dx2 > 0, f"xdot(e=0.1) = {dx2}, expected positive"
    logger.info("  [PASS] xdot(e=0.1, x=%s, nu=%s) = %.6e  (positive)", x,
                nu, dx2)

    # ------------------------------------------------------------------
    # Check 6: Non-spinning should match spinning with chi=0
    # ------------------------------------------------------------------
    de_nospin = edot_resum(0.3, x, nu)
    de_spin0 = edot_resum(0.3, x, nu, chi_S=0.0, chi_A=0.0)
    assert abs(de_nospin - de_spin0) < 1e-30, (
        f"edot mismatch: nospin={de_nospin}, spin0={de_spin0}")
    logger.info("  [PASS] edot(chi=0) matches non-spinning case")

    # ------------------------------------------------------------------
    # Check 7: Spinning case runs without error
    # ------------------------------------------------------------------
    nu_q3 = 3.0 / 16.0  # q=3
    de_spin = edot_resum(0.1, x, nu_q3, chi_S=0.3, chi_A=0.1)
    dx_spin = xdot_resum(0.1, x, nu_q3, chi_S=0.3, chi_A=0.1)
    dz_spin = zetadot_func(0.1, x, 0.5, nu_q3, chi_S=0.3, chi_A=0.1)
    logger.info("  [PASS] Spinning case (chi_S=0.3, chi_A=0.1): "
                "edot=%.6e, xdot=%.6e, zetadot=%.6e",
                de_spin, dx_spin, dz_spin)

    # ------------------------------------------------------------------
    # Check 8: Spin should modify the fluxes
    # ------------------------------------------------------------------
    de_nospin_q3 = edot_resum(0.1, x, nu_q3)
    assert abs(de_spin - de_nospin_q3) > 1e-30, (
        "Spin terms have no effect -- something is wrong")
    logger.info("  [PASS] Spin terms modify edot (diff = %.6e)",
                de_spin - de_nospin_q3)

    logger.info("All sanity checks passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    sanity_check()
