#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: approximated_functions.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.special import zeta

def compute_aparox_kappa_E(e_t):
    """
    Compute kappa_E as a function of eccentricity e_t up to O(e_t^8).
    Originates from the tail corrections to the orbit-averaged expressions 
    for the far-zone energy and angular momentum fluxes;
    Eq. B4a of https://arxiv.org/pdf/1605.00304
    
    Input:
    ------
    e_t : float
        Eccentricity parameter.
    
    Output:
    -------
    kappa_E : float
        Value of kappa_E.
    """
    # Compute kappa_E using the provided expansion
    kappa_E = (1 
               + (2335 / 192) * e_t**2 
               + (42955 / 768) * e_t**4 
               + (6204647 / 36864) * e_t**6 
               + (352891481 / 884736) * e_t**8)
    
    return kappa_E

def compute_aparox_kappa_J(e_t):
    """
    Compute kappa_J as a function of eccentricity e_t up to O(e_t^8).
    Originates from the tail corrections to the orbit-averaged expressions 
    for the far-zone energy and angular momentum fluxes;
    Eq. B4b of https://arxiv.org/pdf/1605.00304
    
    Input:
    ------
    e_t : float
        Eccentricity parameter.
    
    Output:
    -------
    kappa_J : float
        Value of kappa_J.
    """
    # Compute kappa_J using the provided expansion
    kappa_J = (1 
               + (209 / 32) * e_t**2 
               + (2415 / 128) * e_t**4 
               + (730751 / 18432) * e_t**6 
               + (10355719 / 147456) * e_t**8)
    
    return kappa_J

def special_function_phi(e):
    """
    Compute phi(e) with high precision approximation.

    Parameters
    ----------
    e : float
        Eccentricity parameter.

    Returns
    -------
    float
        The value of phi(e) as described in Eq. A8 of
        https://arxiv.org/pdf/1609.05933.
    """
    E = (1 - e**2)**(-0.5)
    return E**10 * (
        1
        + (18970894028 / 2649026657) * e**2
        + (157473274 / 30734301) * e**4
        + (48176523 / 177473701) * e**6
        + (9293260 / 3542508891) * e**8
        - (5034498 / 7491716851) * e**10
        + (428340 / 9958749469) * e**12
    )

def special_function_tilde_phi(e):
    """
    Compute tilde_phi(e) with high precision approximation.

    Parameters
    ----------
    e : float
        Eccentricity parameter.

    Returns
    -------
    float
        The value of tilde_phi(e) as described in Eq. A8 of
        https://arxiv.org/pdf/1609.05933.
    """
    E = (1 - e**2)**(-0.5)
    return E**7 * (
        1
        + (413137256 / 136292703) * e**2
        + (37570495 / 98143337) * e**4
        - (2640201 / 993226448) * e**6
        - (4679700 / 6316712563) * e**8
        - (328675 / 8674876481) * e**10
    )

def special_function_varphi_e(e):
    """
    Compute varphi_e(e) using phi(e) and tilde_phi(e).

    Parameters
    ----------
    e : float
        Eccentricity parameter.

    Returns
    -------
    float
        The value of varphi_e(e) as described in Eq. A10 of
        https://arxiv.org/pdf/1609.05933.
    """
    E = (1 - e**2)**(-0.5)
    return (192 / 985) * (np.sqrt(1 - e**2) / e**2) * (np.sqrt(1 - e**2) * special_function_phi(e) - special_function_tilde_phi(e))

def special_function_psi(e):
    """
    Compute psi(e) with high precision approximation.

    Parameters
    ----------
    e : float
        Eccentricity parameter.

    Returns
    -------
    float
        The value of psi(e) as described in Eq. A11 of
        https://arxiv.org/pdf/1609.05933.
    """
    E = (1 - e**2)**(-0.5)
    return E**12 * (
        1
        - (185 / 21) * e**2
        - (3733 / 99) * e**4
        - (1423 / 104) * e**6
    )

def special_function_znu(e):
    """
    Compute znu(e) with high precision approximation.

    Parameters
    ----------
    e : float
        Eccentricity parameter.

    Returns
    -------
    float
        The value of znu(e) as described in Eq. A12 of
        https://arxiv.org/pdf/1609.05933.
    """
    E = (1 - e**2)**(-0.5)
    return E**12 * (
        1
        + (2095 / 143) * e**2
        + (1590 / 59) * e**4
        + (977 / 113) * e**6
    )

def special_function_kappa(e):
    """
    Compute kappa(e) with high precision approximation.

    Parameters
    ----------
    e : float
        Eccentricity parameter.

    Returns
    -------
    float
        The value of kappa(e) as described in Eq. A13 of
        https://arxiv.org/pdf/1609.05933.
    """
    E = (1 - e**2)**(-0.5)
    return E**14 * (
        1
        + (1497 / 79) * e**2
        + (7021 / 143) * e**4
        + (997 / 98) * e**6
        + (463 / 51) * e**8
        - (3829 / 120) * e**10
    )

def special_function_F(e):
    """
    Compute F(e) based on the series expansion provided.

    Parameters
    ----------
    e : float
        Eccentricity parameter.

    Returns
    -------
    float
        The value of F(e) as described in Eq. A33 of
        https://arxiv.org/pdf/2409.17636.
    """
    denominator = (1 - e**2)**(13 / 2)
    numerator = (
        1
        + (85 / 6) * e**2
        + (5171 / 192) * e**4
        + (1751 / 192) * e**6
        + (297 / 1024) * e**8
    )
    return numerator / denominator

def compute_a4(nu, x):
    """
    Compute the a_4 term in the PN expansion.
    Taken from Appendix C of https://arxiv.org/pdf/1609.05933

    Parameters:
    - nu: Symmetric mass ratio (dimensionless).
    - x: PN expansion parameter (dimensionless).

    Returns:
    - a4: Value of the a_4 term.
    """

    alpha_0 = 153.8803
    term1 = -5 * nu * alpha_0
    term2 = -97 * nu**4 / 3888
    term3 = -18929389 * nu**3 / 435456
    term4 = -3157 * np.pi**2 * nu**2 / 144
    term5 = 54732199 * nu**2 / 93312
    term6 = -47468 * nu * np.log(x) / 315
    term7 = -31495 * np.pi**2 * nu / 8064
    term8 = -856 *np.euler_gamma* nu / 315
    term9 = 59292668653 * nu / 838252800
    term10 = -1712 * nu * np.log(2) / 315
    term11 = 124741 * np.log(x) / 8820
    term12 = -361 * np.pi**2 / 126
    term13 = 124741 *np.euler_gamma/ 4410
    term14 = 3959271176713 / 25427001600
    term15 = -47385 * np.log(3) / 1568
    term16 = 127751 * np.log(2) / 1470

    a4 = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 +
          term10 + term11 + term12 + term13 + term14 + term15 + term16)
    return a4


def compute_a9_2(nu, x):
    """
    Compute the a_9/2 term in the PN expansion.
    Taken from Appendix C of https://arxiv.org/pdf/1609.05933

    Parameters:
    - nu: Symmetric mass ratio (dimensionless).
    - x: PN expansion parameter (dimensionless).

    Returns:
    - a9_2: Value of the a_9/2 term.
    """
    
    term1 = 9731 * np.pi * nu**3 / 1344
    term2 = 42680611 * np.pi * nu**2 / 145152
    term3 = 205 * np.pi**3 * nu / 6
    term4 = -51438847 * np.pi * nu / 48384
    term5 = -3424 * np.pi * np.log(x) / 105
    term6 = -6848 *np.euler_gamma* np.pi / 105
    term7 = 343801320119 * np.pi / 745113600
    term8 = -13696 * np.pi * np.log(2) / 105

    a9_2 = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
    return a9_2
    

def compute_a5(nu, x):
    """
    Compute the coefficient a5 based on the provided parameters.
    Taken from Appendix C of https://arxiv.org/pdf/1609.05933
    
    Parameters
    ----------
    nu : float
        Symmetric mass ratio (dimensionless).
    x : float
        PN expansion parameter (dimensionless).
    
    Returns
    -------
    float
        The computed value of a5.
    """

    alpha_0 = 153.8803
    alpha_1 = -55.13

    term_1 = (155 * alpha_0 * nu**2) / 12
    term_2 = (1195 * alpha_0 * nu) / 336
    term_3 = -6 * nu * alpha_1
    term_4 = -11567 * nu**5 / 62208
    term_5 = 51474823 * nu**4 / 1741824
    term_6 = 9799 * np.pi**2 * nu**3 / 384
    term_7 = -9007776763 * nu**3 / 11757312
    term_8 = (216619 / 189) * nu**2 * np.log(x)
    term_9 = -126809 * np.pi**2 * nu**2 / 3024
    term_10 = -2354 *np.euler_gamma* nu**2 / 9445
    term_11 = 1362630004933 * nu**2 / 914457600
    term_12 = -4708 * nu**2 * np.log(2) / 945
    term_13 = 53963197 * nu * np.log(x) / 52920
    term_14 = 14555455 * np.pi**2 * nu / 217728
    term_15 = 3090781 *np.euler_gamma* nu / 26460
    term_16 = -847101477593593 * nu / 228843014400
    term_17 = -15795 * nu * np.log(3) / 3136
    term_18 = 2105111 * nu * np.log(2) / 8820
    term_19 = -5910592 * np.log(x) / 1964655
    term_20 = -21512 * np.pi**2 / 1701
    term_21 = -11821184 *np.euler_gamma/ 1964655
    term_22 = 29619150939541789 / 36248733480960
    term_23 = 616005 * np.log(3) / 3136
    term_24 = -107638990 * np.log(2) / 392931
    
    # Sum all terms
    a5 = (term_1 + term_2 + term_3 + term_4 + term_5 + 
           term_6 + term_7 + term_8 + term_9 + term_10 +
           term_11 + term_12 + term_13 + term_14 + term_15 +
           term_16 + term_17 + term_18 + term_19 + term_20 +
           term_21 + term_22 + term_23 + term_24)
    
    return a5

def compute_a11_2(nu, x):
    """
    Compute the coefficient a11/2 based on the provided parameters.
    Taken from Appendix C of https://arxiv.org/pdf/1609.05933
    
    Parameters
    ----------
    nu : float
        Symmetric mass ratio (dimensionless).
    x : float
        PN expansion parameter (dimensionless).
    Returns
    -------
    float
        The computed value of a11/2.
    """
    alpha_0 = 153.8803
    
    term_1 = -20 * np.pi * nu * alpha_0
    term_2 = (49187 * np.pi * nu**4) / 6048
    term_3 = -7030123 * np.pi * nu**3 / 13608
    term_4 = -112955 * np.pi**3 * nu**2 / 576
    term_5 = 1760705531 * np.pi * nu**2 / 290304
    term_6 = -189872 * np.pi * nu * np.log(x) / 315
    term_7 = (26035 * np.pi**3 * nu) / 16128
    term_8 = -3424 *np.euler_gamma* np.pi * nu / 315
    term_9 = -2437749208561 * np.pi * nu / 4470681600
    term_10 = -6848 * np.pi * nu * np.log(2) / 315
    term_11 = (311233 * np.pi * np.log(x)) / 11760
    term_12 = (311233 *np.euler_gamma* np.pi) / 5880
    term_13 = 91347297344213 * np.pi / 81366405120
    term_14 = -142155 * np.pi * np.log(3) / 784
    term_15 = (5069891 * np.pi * np.log(2)) / 17640
    
    # Sum all terms
    a11_2 = (term_1 + term_2 + term_3 + term_4 + term_5 +
              term_6 + term_7 + term_8 + term_9 + term_10 +
              term_11 + term_12 + term_13 + term_14 + term_15)
    
    return a11_2

def compute_a6(nu, x):
    """
    Compute the coefficient a6 based on the provided parameters.
    Taken from Appendix C of https://arxiv.org/pdf/1609.05933
    
    Parameters
    ----------
    nu : float
        Symmetric mass ratio (dimensionless).
    x : float
        PN expansion parameter (dimensionless).
    
    Returns
    -------
    float
        The computed value of a6.
    """

    alpha_0 = 153.8803
    alpha_1 = -55.13
    alpha_2 = 588
    alpha_3 = 1144
    
    term_1 = -535 * alpha_0 * nu**3 / 36
    term_2 = 7295 * alpha_0 * nu**2 / 336
    term_3 = -248065 * alpha_0 * nu / 4536
    term_4 = 31 * alpha_1 * nu**2 / 2
    term_5 = 239 * alpha_1 * nu / 56
    term_6 = -7 * alpha_2 * nu
    term_7 = -7 * alpha_3 * nu * np.log(x)
    term_8 = -alpha_3 * nu
    term_9 = -155377 * nu**6 / 1679616
    term_10 = -152154269 * nu**5 / 10450944
    term_11 = -1039145 * np.pi**2 * nu**4 / 62208
    term_12 = 76527233921 * nu**4 / 94058496
    term_13 = -41026693 * nu**3 * np.log(x) / 17010
    term_14 = 55082725 * np.pi**2 * nu**3 / 217728
    term_15 = -2033 *np.euler_gamma* nu**3 / 1701
    term_16 = -56909847373567 * nu**3 / 7242504192
    term_17 = -4066 * nu**3 * np.log(2) / 1701
    term_18 = -271237829 * nu**2 * np.log(x) / 127008
    term_19 = 92455 * np.pi**4 * nu**2 / 1152
    term_20 = -4061971769 * np.pi**2 * nu**2 / 870912
    term_21 = -21169753 *np.euler_gamma* nu**2 / 317520
    term_22 = 3840832667727673 * nu**2 / 55477094400
    term_23 = -57915 * nu**2 * np.log(3) / 12544
    term_24 = -2724535 * nu**2 * np.log(2) / 21168
    term_25 = -4387 / 63 * np.pi**2 * nu * np.log(x)
    term_26 = -12030840839 * nu * np.log(x) / 37721376
    term_27 = 410 * np.pi**4 * nu / 9
    term_28 = -8774 / 63 *np.euler_gamma* np.pi**2 * nu
    term_29 = 206470485307 * np.pi**2 * nu / 1005903360
    term_30 = 362623282541 *np.euler_gamma* nu / 94303440
    term_31 = -12413297162366594971 * nu / 271865501107200
    term_32 = 3016845 * nu * np.log(3) / 12544
    term_33 = -17548 / 63 * np.pi**2 * nu * np.log(2)
    term_34 = 701463800861 * nu * np.log(2) / 94303440
    term_35 = 366368 * np.log(x)**2 / 11025
    term_36 = 2930944 * np.log(2) * np.log(x) / 11025
    term_37 = -13696 / 315 * np.pi**2 * np.log(x)
    term_38 = 1465472 *np.euler_gamma* np.log(x) / 11025
    term_39 = -155359670313691 * np.log(x) / 157329572400
    term_40 = -27392 * zeta(3) / 105
    term_41 = -256 * np.pi**4 / 45
    term_42 = -27392 *np.euler_gamma* np.pi**2 / 315
    term_43 = 1414520047 * np.pi**2 / 2619540
    term_44 = 1465472 * np.euler_gamma**2 / 11025
    term_45 = -155359670313691 *np.euler_gamma/ 78664786200
    term_46 = 1867705968412371074441833 / 154211174411374080000
    term_47 = 5861888 * np.log(2)**2 / 11025
    term_48 = -37744140625 * np.log(5) / 260941824
    term_49 = -63722699919 * np.log(3) / 112752640
    term_50 = -54784 / 315 * np.pi**2 * np.log(2)
    term_51 = 5861888 *np.euler_gamma* np.log(2) / 11025
    term_52 = -206962178724547 * np.log(2) / 78664786200

    # Sum all terms
    a6 = (term_1 + term_2 + term_3 + term_4 + term_5 +
           term_6 + term_7 + term_8 + term_9 + term_10 +
           term_11 + term_12 + term_13 + term_14 + term_15 +
           term_16 + term_17 + term_18 + term_19 + term_20 +
           term_21 + term_22 + term_23 + term_24 + term_25 +
           term_26 + term_27 + term_28 + term_29 + term_30 +
           term_31 + term_32 + term_33 + term_34 + term_35 +
           term_36 + term_37 + term_38 + term_39 + term_40 +
           term_41 + term_42 + term_43 + term_44 + term_45 +
           term_46 + term_47 + term_48 + term_49 + term_50 +
           term_51 + term_52)
    
    return a6