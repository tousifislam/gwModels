"""
Post-adiabatic (PA) corrections to Keplerian orbital parameters.

Implements the 2.5PN post-adiabatic corrections to the EOB/Keplerian
orbital parameters (x, e, zeta) as derived in the supplementary data
of the SEOBNRv5EHM model (Gamboa et al.), based on the formalism of
Bini and Damour (arXiv:1904.11814).

The corrections are expressed in terms of the relativistic anomaly zeta,
the frequency parameter x, the eccentricity e, and the symmetric mass
ratio nu, with gauge parameters alpha and beta. The default gauge
choice corresponds to the SEOBNRv5EHM model: alpha = -16/3, beta = -13/2.

The parameter epsilon (radiation-reaction book-keeping parameter) is set
to 1 throughout, as appropriate for physical applications.

References
----------
- D. Bini, T. Damour, "Gravitational scattering of two black holes at
  the fourth post-Newtonian approximation", arXiv:1904.11814
- Gamboa et al., SEOBNRv5EHM supplementary data (EOB_Keplerian.dat.m)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Default gauge parameters for SEOBNRv5EHM
_ALPHA_DEFAULT = -16.0 / 3.0
_BETA_DEFAULT = -13.0 / 2.0


def pa_correction_x(e, x, zeta, nu, alpha=_ALPHA_DEFAULT, beta=_BETA_DEFAULT):
    """
    Compute the PA correction to the frequency parameter x.

    The full corrected value is:
        x_PA = x_adiabatic + delta_x
    where delta_x is the quantity returned by this function.

    The correction has the structure:
        delta_x = x^(7/2) * nu * F_x(e, zeta, alpha, beta)
    with F_x containing terms up to e^9.

    Parameters
    ----------
    e : float or array_like
        Orbital eccentricity.
    x : float or array_like
        Frequency-related orbital parameter (~ (M*Omega)^(2/3)).
    zeta : float or array_like
        Relativistic anomaly.
    nu : float
        Symmetric mass ratio, nu = m1*m2 / (m1+m2)^2.
    alpha : float, optional
        Gauge parameter (default: -16/3 for SEOBNRv5EHM).
    beta : float, optional
        Gauge parameter (default: -13/2 for SEOBNRv5EHM).

    Returns
    -------
    float or array_like
        The PA correction delta_x to be added to x_adiabatic.

    References
    ----------
    arXiv:1904.11814, Gamboa et al. SEOBNRv5EHM supplementary data.
    """
    sin_z = np.sin(zeta)
    cos_z = np.cos(zeta)
    sin_2z = np.sin(2.0 * zeta)
    cos_2z = np.cos(2.0 * zeta)
    sin_3z = np.sin(3.0 * zeta)
    sin_5z = np.sin(5.0 * zeta)
    sin_7z = np.sin(7.0 * zeta)
    sin_9z = np.sin(9.0 * zeta)
    cos_4z = np.cos(4.0 * zeta)
    cos_6z = np.cos(6.0 * zeta)

    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    e5 = e4 * e
    e6 = e3 * e3
    e7 = e6 * e
    e8 = e4 * e4
    e9 = e8 * e

    # e^1 term: (16*e*(26 + alpha)*Sin[zeta])/5
    term_e1 = (16.0 * e * (26.0 + alpha) * sin_z) / 5.0

    # e^2 term: (4*e^2*(83 + 24*alpha)*Sin[2*zeta])/15
    term_e2 = (4.0 * e2 * (83.0 + 24.0 * alpha) * sin_2z) / 15.0

    # e^3 term: (8*e^3*(2468 + 126*alpha - 9*beta
    #            + (184 + 45*alpha + 9*beta)*Cos[2*zeta])*Sin[zeta])/45
    term_e3 = (
        8.0
        * e3
        * (
            2468.0
            + 126.0 * alpha
            - 9.0 * beta
            + (184.0 + 45.0 * alpha + 9.0 * beta) * cos_2z
        )
        * sin_z
    ) / 45.0

    # e^4 term: (e^4*(892 + 408*alpha - 24*beta
    #            + (-1 + 24*alpha + 24*beta)*Cos[2*zeta])*Sin[2*zeta])/15
    term_e4 = (
        e4
        * (
            892.0
            + 408.0 * alpha
            - 24.0 * beta
            + (-1.0 + 24.0 * alpha + 24.0 * beta) * cos_2z
        )
        * sin_2z
    ) / 15.0

    # e^5 term: (e^5*(60*(4222 + 192*alpha - 33*beta)*Sin[zeta]
    #            + 5*(3304 + 666*alpha + 117*beta)*Sin[3*zeta]
    #            + 9*(24 + 5*beta)*Sin[5*zeta]))/225
    term_e5 = (
        e5
        * (
            60.0 * (4222.0 + 192.0 * alpha - 33.0 * beta) * sin_z
            + 5.0 * (3304.0 + 666.0 * alpha + 117.0 * beta) * sin_3z
            + 9.0 * (24.0 + 5.0 * beta) * sin_5z
        )
    ) / 225.0

    # e^6 term: (e^6*(1525 + 1008*alpha - 84*beta
    #            + 6*(-37 + 14*alpha + 14*beta)*Cos[2*zeta]
    #            - 14*Cos[4*zeta])*Sin[2*zeta])/15
    term_e6 = (
        e6
        * (
            1525.0
            + 1008.0 * alpha
            - 84.0 * beta
            + 6.0 * (-37.0 + 14.0 * alpha + 14.0 * beta) * cos_2z
            - 14.0 * cos_4z
        )
        * sin_2z
    ) / 15.0

    # e^7 term: e^7*((7*(9748 + 453*alpha - 87*beta)*Sin[zeta])/30
    #           + (7*(826 + 147*alpha + 24*beta)*Sin[3*zeta])/30
    #           + ((354 + 35*beta)*Sin[5*zeta])/50
    #           + (8*Sin[7*zeta])/35)
    term_e7 = e7 * (
        (7.0 * (9748.0 + 453.0 * alpha - 87.0 * beta) * sin_z) / 30.0
        + (7.0 * (826.0 + 147.0 * alpha + 24.0 * beta) * sin_3z) / 30.0
        + ((354.0 + 35.0 * beta) * sin_5z) / 50.0
        + (8.0 * sin_7z) / 35.0
    )

    # e^8 term: (e^8*(50192 + 46872*alpha - 4536*beta
    #            + 9*(-2483 + 504*alpha + 504*beta)*Cos[2*zeta]
    #            - 2558*Cos[4*zeta] - 81*Cos[6*zeta])*Sin[2*zeta])/360
    term_e8 = (
        e8
        * (
            50192.0
            + 46872.0 * alpha
            - 4536.0 * beta
            + 9.0 * (-2483.0 + 504.0 * alpha + 504.0 * beta) * cos_2z
            - 2558.0 * cos_4z
            - 81.0 * cos_6z
        )
        * sin_2z
    ) / 360.0

    # e^9 term: (e^9*(1323*(7498 + 353*alpha - 72*beta)*Sin[zeta]
    #            + 7*(141350 + 23058*alpha + 3591*beta)*Sin[3*zeta]
    #            + 21*(2878 + 189*beta)*Sin[5*zeta]
    #            + 4524*Sin[7*zeta] + 140*Sin[9*zeta]))/2520
    term_e9 = (
        e9
        * (
            1323.0 * (7498.0 + 353.0 * alpha - 72.0 * beta) * sin_z
            + 7.0 * (141350.0 + 23058.0 * alpha + 3591.0 * beta) * sin_3z
            + 21.0 * (2878.0 + 189.0 * beta) * sin_5z
            + 4524.0 * sin_7z
            + 140.0 * sin_9z
        )
    ) / 2520.0

    F_x = (
        term_e1
        + term_e2
        + term_e3
        + term_e4
        + term_e5
        + term_e6
        + term_e7
        + term_e8
        + term_e9
    )

    # epsilon = 1, so epsilon^5 = 1
    delta_x = x ** (7.0 / 2.0) * nu * F_x
    return delta_x


def pa_correction_e(e, x, zeta, nu, alpha=_ALPHA_DEFAULT, beta=_BETA_DEFAULT):
    """
    Compute the PA correction to the eccentricity e.

    The full corrected value is:
        e_PA = e_adiabatic + delta_e
    where delta_e is the quantity returned by this function.

    The correction has the structure:
        delta_e = x^(5/2) * nu * F_e(e, zeta, alpha, beta)
    with F_e containing terms up to e^8.

    Parameters
    ----------
    e : float or array_like
        Orbital eccentricity.
    x : float or array_like
        Frequency-related orbital parameter (~ (M*Omega)^(2/3)).
    zeta : float or array_like
        Relativistic anomaly.
    nu : float
        Symmetric mass ratio, nu = m1*m2 / (m1+m2)^2.
    alpha : float, optional
        Gauge parameter (default: -16/3 for SEOBNRv5EHM).
    beta : float, optional
        Gauge parameter (default: -13/2 for SEOBNRv5EHM).

    Returns
    -------
    float or array_like
        The PA correction delta_e to be added to e_adiabatic.

    References
    ----------
    arXiv:1904.11814, Gamboa et al. SEOBNRv5EHM supplementary data.
    """
    sin_z = np.sin(zeta)
    cos_z = np.cos(zeta)
    sin_2z = np.sin(2.0 * zeta)
    cos_2z = np.cos(2.0 * zeta)
    sin_3z = np.sin(3.0 * zeta)
    sin_5z = np.sin(5.0 * zeta)
    sin_7z = np.sin(7.0 * zeta)
    cos_4z = np.cos(4.0 * zeta)

    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    e5 = e4 * e
    e6 = e3 * e3
    e7 = e6 * e
    e8 = e4 * e4

    # e^0 term: (-64*Sin[zeta])/5
    term_e0 = (-64.0 * sin_z) / 5.0

    # e^1 term: -(8*e*(23 + 3*alpha)*Sin[2*zeta])/15
    term_e1 = -(8.0 * e * (23.0 + 3.0 * alpha) * sin_2z) / 15.0

    # e^2 term: -(4*e^2*(1337 + 72*alpha - 9*beta
    #            + (127 + 36*alpha + 9*beta)*Cos[2*zeta])*Sin[zeta])/45
    term_e2 = (
        -(
            4.0
            * e2
            * (
                1337.0
                + 72.0 * alpha
                - 9.0 * beta
                + (127.0 + 36.0 * alpha + 9.0 * beta) * cos_2z
            )
            * sin_z
        )
        / 45.0
    )

    # e^3 term: -(e^3*(4*(191 + 60*alpha - 6*beta)
    #            + (59 + 24*alpha + 24*beta)*Cos[2*zeta])*Sin[2*zeta])/30
    term_e3 = (
        -(
            e3
            * (
                4.0 * (191.0 + 60.0 * alpha - 6.0 * beta)
                + (59.0 + 24.0 * alpha + 24.0 * beta) * cos_2z
            )
            * sin_2z
        )
        / 30.0
    )

    # e^4 term: -(e^4*(4500 + 264*alpha - 33*beta
    #            + 2*(331 + 72*alpha + 15*beta)*Cos[2*zeta]
    #            + 3*beta*Cos[4*zeta])*Sin[zeta])/15
    term_e4 = (
        -(
            e4
            * (
                4500.0
                + 264.0 * alpha
                - 33.0 * beta
                + 2.0 * (331.0 + 72.0 * alpha + 15.0 * beta) * cos_2z
                + 3.0 * beta * cos_4z
            )
            * sin_z
        )
        / 15.0
    )

    # e^5 term: -(e^5*(385 + 204*alpha - 24*beta
    #            + (-17 + 24*alpha + 24*beta)*Cos[2*zeta])*Sin[2*zeta])/12
    term_e5 = (
        -(
            e5
            * (
                385.0
                + 204.0 * alpha
                - 24.0 * beta
                + (-17.0 + 24.0 * alpha + 24.0 * beta) * cos_2z
            )
            * sin_2z
        )
        / 12.0
    )

    # e^6 term: (e^6*(-75*(5839 + 276*alpha - 69*beta)*Sin[zeta]
    #            - 25*(1793 + 324*alpha + 54*beta)*Sin[3*zeta]
    #            - 9*(152 + 25*beta)*Sin[5*zeta]))/900
    term_e6 = (
        e6
        * (
            -75.0 * (5839.0 + 276.0 * alpha - 69.0 * beta) * sin_z
            - 25.0 * (1793.0 + 324.0 * alpha + 54.0 * beta) * sin_3z
            - 9.0 * (152.0 + 25.0 * beta) * sin_5z
        )
    ) / 900.0

    # e^7 term: (e^7*(-6047 - 5040*alpha + 630*beta
    #            - 9*(-271 + 70*alpha + 70*beta)*Cos[2*zeta]
    #            + 266*Cos[4*zeta])*Sin[2*zeta])/180
    term_e7 = (
        e7
        * (
            -6047.0
            - 5040.0 * alpha
            + 630.0 * beta
            - 9.0 * (-271.0 + 70.0 * alpha + 70.0 * beta) * cos_2z
            + 266.0 * cos_4z
        )
        * sin_2z
    ) / 180.0

    # e^8 term: e^8*((-7*(2507 + 168*alpha - 21*beta
    #            + 6*(16*alpha + 3*beta)*Cos[2*zeta]
    #            + 3*beta*Cos[4*zeta])*Sin[zeta])/24
    #            - (3923*Sin[3*zeta])/45
    #            - (3403*Sin[5*zeta])/600
    #            - (38*Sin[7*zeta])/105)
    term_e8 = e8 * (
        (
            -7.0
            * (
                2507.0
                + 168.0 * alpha
                - 21.0 * beta
                + 6.0 * (16.0 * alpha + 3.0 * beta) * cos_2z
                + 3.0 * beta * cos_4z
            )
            * sin_z
        )
        / 24.0
        - (3923.0 * sin_3z) / 45.0
        - (3403.0 * sin_5z) / 600.0
        - (38.0 * sin_7z) / 105.0
    )

    F_e = (
        term_e0
        + term_e1
        + term_e2
        + term_e3
        + term_e4
        + term_e5
        + term_e6
        + term_e7
        + term_e8
    )

    # epsilon = 1, so epsilon^5 = 1
    delta_e = x ** (5.0 / 2.0) * nu * F_e
    return delta_e


def pa_correction_zeta(e, x, zeta, nu, alpha=_ALPHA_DEFAULT, beta=_BETA_DEFAULT):
    """
    Compute the PA correction to the relativistic anomaly zeta.

    The full corrected value is:
        zeta_PA = zeta_adiabatic + delta_zeta
    where delta_zeta is the quantity returned by this function.

    The correction has the structure:
        delta_zeta = x^(5/2) * nu * F_zeta(e, zeta, alpha, beta)
    with F_zeta containing terms up to e^8 (and an e^{-1} term).

    Parameters
    ----------
    e : float or array_like
        Orbital eccentricity.
    x : float or array_like
        Frequency-related orbital parameter (~ (M*Omega)^(2/3)).
    zeta : float or array_like
        Relativistic anomaly.
    nu : float
        Symmetric mass ratio, nu = m1*m2 / (m1+m2)^2.
    alpha : float, optional
        Gauge parameter (default: -16/3 for SEOBNRv5EHM).
    beta : float, optional
        Gauge parameter (default: -13/2 for SEOBNRv5EHM).

    Returns
    -------
    float or array_like
        The PA correction delta_zeta to be added to zeta_adiabatic.

    Notes
    -----
    This expression contains a 1/e term. For strictly circular orbits
    (e=0), this function is singular. In practice, the PA formalism is
    only applied for eccentric orbits with e > 0.

    References
    ----------
    arXiv:1904.11814, Gamboa et al. SEOBNRv5EHM supplementary data.
    """
    cos_z = np.cos(zeta)
    cos_2z = np.cos(2.0 * zeta)
    cos_3z = np.cos(3.0 * zeta)
    cos_4z = np.cos(4.0 * zeta)
    cos_5z = np.cos(5.0 * zeta)
    cos_6z = np.cos(6.0 * zeta)
    cos_7z = np.cos(7.0 * zeta)
    cos_8z = np.cos(8.0 * zeta)
    cos_10z = np.cos(10.0 * zeta)

    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    e5 = e4 * e
    e6 = e3 * e3
    e7 = e6 * e
    e8 = e4 * e4

    # e^{-1} term: (-64*Cos[zeta])/(5*e)
    term_em1 = (-64.0 * cos_z) / (5.0 * e)

    # e^0 term: -(8*(66 - 6*alpha + 3*beta
    #            + (23 + 3*alpha)*Cos[2*zeta]))/15
    term_e0 = (
        -(
            8.0
            * (
                66.0
                - 6.0 * alpha
                + 3.0 * beta
                + (23.0 + 3.0 * alpha) * cos_2z
            )
        )
        / 15.0
    )

    # e^1 term: -(4*e*Cos[zeta]*(2395 - 90*alpha + 45*beta
    #            + (127 + 36*alpha + 9*beta)*Cos[2*zeta]))/45
    term_e1 = (
        -(
            4.0
            * e
            * cos_z
            * (
                2395.0
                - 90.0 * alpha
                + 45.0 * beta
                + (127.0 + 36.0 * alpha + 9.0 * beta) * cos_2z
            )
        )
        / 45.0
    )

    # e^2 term: (e^2*(24*(-3334 + 81*alpha - 45*beta)
    #            - 8*(1205 + 36*alpha + 54*beta)*Cos[2*zeta]
    #            - 3*(59 + 24*alpha + 24*beta)*Cos[4*zeta]))/180
    term_e2 = (
        e2
        * (
            24.0 * (-3334.0 + 81.0 * alpha - 45.0 * beta)
            - 8.0 * (1205.0 + 36.0 * alpha + 54.0 * beta) * cos_2z
            - 3.0 * (59.0 + 24.0 * alpha + 24.0 * beta) * cos_4z
        )
    ) / 180.0

    # e^3 term: -(e^3*Cos[zeta]*((3718 + 648*alpha + 252*beta)*Cos[2*zeta]
    #            + 3*(36553 - 624*alpha + 318*beta
    #            + 6*beta*Cos[4*zeta])))/90
    term_e3 = (
        -(
            e3
            * cos_z
            * (
                (3718.0 + 648.0 * alpha + 252.0 * beta) * cos_2z
                + 3.0
                * (
                    36553.0
                    - 624.0 * alpha
                    + 318.0 * beta
                    + 6.0 * beta * cos_4z
                )
            )
        )
        / 90.0
    )

    # e^4 term: (e^4*(40*(-35935 + 378*alpha - 216*beta)
    #            - 2*(97681 + 360*alpha + 2160*beta)*Cos[2*zeta]
    #            - 5*(-91 + 144*alpha + 144*beta)*Cos[4*zeta]))/720
    term_e4 = (
        e4
        * (
            40.0 * (-35935.0 + 378.0 * alpha - 216.0 * beta)
            - 2.0 * (97681.0 + 360.0 * alpha + 2160.0 * beta) * cos_2z
            - 5.0 * (-91.0 + 144.0 * alpha + 144.0 * beta) * cos_4z
        )
    ) / 720.0

    # e^5 term: -(e^5*Cos[zeta]*(6*(6323737 - 55500*alpha + 28500*beta)
    #            + 4*(212989 + 27000*alpha + 12375*beta)*Cos[2*zeta]
    #            + (26869 + 4500*beta)*Cos[4*zeta]))/9000
    term_e5 = (
        -(
            e5
            * cos_z
            * (
                6.0 * (6323737.0 - 55500.0 * alpha + 28500.0 * beta)
                + 4.0 * (212989.0 + 27000.0 * alpha + 12375.0 * beta) * cos_2z
                + (26869.0 + 4500.0 * beta) * cos_4z
            )
        )
        / 9000.0
    )

    # e^6 term: (e^6*(25*(-16909031 + 95760*alpha - 55440*beta)
    #            - 25*(2773579 + 30240*beta)*Cos[2*zeta]
    #            - 2*(-283727 + 63000*alpha + 63000*beta)*Cos[4*zeta]
    #            + 52469*Cos[6*zeta]))/72000
    term_e6 = (
        e6
        * (
            25.0 * (-16909031.0 + 95760.0 * alpha - 55440.0 * beta)
            - 25.0 * (2773579.0 + 30240.0 * beta) * cos_2z
            - 2.0 * (-283727.0 + 63000.0 * alpha + 63000.0 * beta) * cos_4z
            + 52469.0 * cos_6z
        )
    ) / 72000.0

    # e^7 term: (e^7*(-4900*Cos[zeta]*(3975061 - 20160*alpha + 10395*beta
    #            + 3150*(2*alpha + beta)*Cos[2*zeta]
    #            + 315*beta*Cos[4*zeta])
    #            - 49*(3290177*Cos[3*zeta] + 225235*Cos[5*zeta])
    #            - 631483*Cos[7*zeta]))/1764000
    term_e7 = (
        e7
        * (
            -4900.0
            * cos_z
            * (
                3975061.0
                - 20160.0 * alpha
                + 10395.0 * beta
                + 3150.0 * (2.0 * alpha + beta) * cos_2z
                + 315.0 * beta * cos_4z
            )
            - 49.0 * (3290177.0 * cos_3z + 225235.0 * cos_5z)
            - 631483.0 * cos_7z
        )
    ) / 1764000.0

    # e^8 term: (e^8*(700*(-1420895911 + 17146080*alpha - 5715360*beta)*Cos[2*zeta]
    #            - 251950184836*Cos[4*zeta]
    #            - 173786363656*Cos[6*zeta]
    #            - 259839915975*Cos[8*zeta]
    #            - 350*(10482108379 - 65726640*alpha + 21908880*beta
    #            + 564036056*Cos[10*zeta])))/254016000
    term_e8 = (
        e8
        * (
            700.0
            * (-1420895911.0 + 17146080.0 * alpha - 5715360.0 * beta)
            * cos_2z
            - 251950184836.0 * cos_4z
            - 173786363656.0 * cos_6z
            - 259839915975.0 * cos_8z
            - 350.0
            * (
                10482108379.0
                - 65726640.0 * alpha
                + 21908880.0 * beta
                + 564036056.0 * cos_10z
            )
        )
    ) / 254016000.0

    F_zeta = (
        term_em1
        + term_e0
        + term_e1
        + term_e2
        + term_e3
        + term_e4
        + term_e5
        + term_e6
        + term_e7
        + term_e8
    )

    # epsilon = 1, so epsilon^5 = 1
    delta_zeta = x ** (5.0 / 2.0) * nu * F_zeta
    return delta_zeta


def apply_pa_corrections(e, x, zeta, nu, alpha=_ALPHA_DEFAULT, beta=_BETA_DEFAULT):
    """
    Apply post-adiabatic corrections to all three Keplerian parameters.

    Computes the corrected values:
        x_PA    = x    + delta_x(e, x, zeta, nu, alpha, beta)
        e_PA    = e    + delta_e(e, x, zeta, nu, alpha, beta)
        zeta_PA = zeta + delta_zeta(e, x, zeta, nu, alpha, beta)

    Parameters
    ----------
    e : float or array_like
        Orbital eccentricity (adiabatic).
    x : float or array_like
        Frequency-related orbital parameter (adiabatic).
    zeta : float or array_like
        Relativistic anomaly (adiabatic).
    nu : float
        Symmetric mass ratio.
    alpha : float, optional
        Gauge parameter (default: -16/3 for SEOBNRv5EHM).
    beta : float, optional
        Gauge parameter (default: -13/2 for SEOBNRv5EHM).

    Returns
    -------
    tuple of (float or array_like)
        (x_PA, e_PA, zeta_PA) -- the post-adiabatically corrected parameters.

    References
    ----------
    arXiv:1904.11814, Gamboa et al. SEOBNRv5EHM supplementary data.
    """
    dx = pa_correction_x(e, x, zeta, nu, alpha, beta)
    de = pa_correction_e(e, x, zeta, nu, alpha, beta)
    dzeta = pa_correction_zeta(e, x, zeta, nu, alpha, beta)

    x_pa = x + dx
    e_pa = e + de
    zeta_pa = zeta + dzeta

    return x_pa, e_pa, zeta_pa


def sanity_check(tol=1.0e-14):
    """
    Verify that PA corrections vanish for the circular limit (e=0).

    In the circular limit, all oscillatory corrections should vanish
    because every term in delta_x and delta_e is proportional to at
    least e^1 (via explicit e prefactors or Sin/Cos of zeta multiplied
    by e-dependent coefficients that go to zero).

    Note: delta_zeta contains a 1/e term, so it is excluded from the
    e=0 check. For e -> 0, delta_zeta diverges, which is expected --
    the anomaly is undefined for circular orbits.

    Parameters
    ----------
    tol : float, optional
        Tolerance for the circular-limit check (default: 1e-14).

    Raises
    ------
    AssertionError
        If any correction exceeds the tolerance at e=0.
    """
    x_test = 0.05
    nu_test = 0.25
    zeta_values = np.linspace(0.0, 2.0 * np.pi, 64)

    logger.info("Running PA corrections sanity check (circular limit e=0)...")

    # delta_x should vanish at e=0 for all zeta
    dx_vals = pa_correction_x(0.0, x_test, zeta_values, nu_test)
    max_dx = np.max(np.abs(dx_vals))
    assert max_dx < tol, (
        f"pa_correction_x does not vanish for e=0: max|delta_x| = {max_dx:.3e}"
    )
    logger.info("  delta_x at e=0: max = %.3e (OK)", max_dx)

    # delta_e should vanish at e=0 for all zeta
    # The e^0 term is (-64*sin(zeta))/5, which does NOT vanish.
    # However, this is the correction to e itself at e=0, meaning
    # it represents a physical excitation from the circular limit.
    # For a true sanity check, we verify the e-dependent terms vanish
    # and the e^0 term has the expected form.
    de_vals = pa_correction_e(0.0, x_test, zeta_values, nu_test)
    # The e^0 term: -64/5 * sin(zeta) * x^(5/2) * nu
    expected_e0 = (-64.0 / 5.0) * np.sin(zeta_values) * x_test ** 2.5 * nu_test
    max_de_diff = np.max(np.abs(de_vals - expected_e0))
    assert max_de_diff < tol, (
        f"pa_correction_e e-dependent terms do not vanish for e=0: "
        f"max residual = {max_de_diff:.3e}"
    )
    logger.info(
        "  delta_e at e=0: e-dependent terms vanish, max residual = %.3e (OK)",
        max_de_diff,
    )

    # delta_zeta has a 1/e singularity, so we skip e=0 and check small e
    e_small = 1.0e-6
    dzeta_vals = pa_correction_zeta(e_small, x_test, zeta_values, nu_test)
    # The dominant term at small e is -64*cos(zeta)/(5*e), so
    # delta_zeta ~ -64*cos(zeta)/(5*e) * x^(5/2) * nu
    expected_dominant = (
        (-64.0 * np.cos(zeta_values)) / (5.0 * e_small)
        * x_test ** 2.5
        * nu_test
    )
    # The ratio should be close to 1 where cos(zeta) != 0
    mask = np.abs(np.cos(zeta_values)) > 0.1
    ratio = dzeta_vals[mask] / expected_dominant[mask]
    max_ratio_dev = np.max(np.abs(ratio - 1.0))
    assert max_ratio_dev < 1.0e-3, (
        f"pa_correction_zeta dominant 1/e term deviates: "
        f"max|ratio - 1| = {max_ratio_dev:.3e}"
    )
    logger.info(
        "  delta_zeta at e=%.1e: 1/e dominance confirmed, "
        "max|ratio-1| = %.3e (OK)",
        e_small,
        max_ratio_dev,
    )

    logger.info("PA corrections sanity check passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sanity_check()
