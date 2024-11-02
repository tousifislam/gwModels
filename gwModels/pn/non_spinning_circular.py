#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: non_spinning_quasicircular.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt

from .pn_utils import *
from ..utils import mass_ratio_to_symmetric_mass_ratio
from ..utils import get_phase, get_frequency

def x_of_tau(tau, q):
    """
    Computes the dimensionless post-Newtonian (PN) frequency expansion parameter x
    given the PN time tau and mass ratio q.
    
    Parameters
    ----------
    tau : float
        The PN time parameter.
    q : float
        Mass ratio defined as q = m1 / m2, with m1 >= m2.
    
    Returns
    -------
    x_results : dict
        A dictionary where each key represents a PN order (e.g., '1.0', '1.5', etc.),
        and the value is the cumulative value of x up to that PN order.
        
        Example:
        x_terms = {
            '1.0': term_1,
            '1.5': term_1 + term_2,
            '2.0': term_1 + term_2 + term_3,
            ...
        }
    
    Notes
    -----
    The PN expansion includes terms up to 4.5 PN order.
    """

    # Convert mass ratio to symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)
    
    # 0 PN term
    term_1 = 1

    # 1 PN term
    term_2 = (743 / 4032 + 11 / 48 * nu) * tau**(-1 / 4)

    # 1.5 PN term
    term_3 = -np.pi / 5 * tau**(-3 / 8)

    # 2 PN term
    term_4 = (19583 / 254016 + 24401 / 193536 * nu + 31 / 288 * nu**2) * tau**(-1 / 2)

    # 2.5 PN term
    term_5 = (-11891 / 53760 + 109 / 1920 * nu) * np.pi * tau**(-5 / 8)

    # 3 PN term
    term_6 = (
        -10052469856691 / 6008596070400 + np.pi**2 / 6 + 107 / 420 * np.euler_gamma - 
        107 / 3360 * np.log(tau / 256) +
        (3147553127 / 780337152 - 451 * np.pi**2 / 3072) * nu - 
        15211 / 442368 * nu**2 + 25565 / 331776 * nu**3
    ) * tau**(-3 / 4)

    # 3.5 PN term
    term_7 = (
        -113868647 / 433520640 - 31821 / 143360 * nu + 294941 / 3870720 * nu**2
    ) * np.pi * tau**(-7 / 8)

    # 4 PN term
    term_8 = (
        -2518977598355703073 / 3779358859513036800 + 9203 / 215040 * np.euler_gamma + 
        9049 / 258048 * np.pi**2 + 14873 / 1128960 * np.log(2) + 
        47385 / 1605632 * np.log(3) - 9203 / 3440640 * np.log(tau) +
        (718143266031997 / 576825222758400 + 244493 / 1128960 * np.euler_gamma - 
         65577 / 1835008 * np.pi**2 + 15761 / 47040 * np.log(2) - 
         47385 / 401408 * np.log(3) - 244493 / 18063360 * np.log(tau)) * nu +
        (-1502014727 / 8323596288 + 2255 / 393216 * np.pi**2) * nu**2 - 
        258479 / 33030144 * nu**3 + 1195 / 262144 * nu**4
    ) * tau**(-1) * np.log(tau)

    # 4.5 PN term
    term_9 = (
        -9965202491753717 / 5768252227584000 + 107 / 600 * np.euler_gamma + 
        23 / 600 * np.pi**2 - 107 / 4800 * np.log(tau / 256) + 
        (8248609881163 / 2746786775040 - 3157 / 30720 * np.pi**2) * nu - 
        3590973803 / 20808990720 * nu**2 - 520159 / 1634992128 * nu**3
    ) * np.pi * tau**(-9 / 8)

    # Summing terms at each PN order and storing in dictionary for easy access
    x_brackets = {
        '0.0': term_1,
        '1.0': term_1 + term_2,
        '1.5': term_1 + term_2 + term_3,
        '2.0': term_1 + term_2 + term_3 + term_4,
        '2.5': term_1 + term_2 + term_3 + term_4 + term_5,
        '3.0': term_1 + term_2 + term_3 + term_4 + term_5 + term_6,
        '3.5': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7,
        '4.0': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8,
        '4.5': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
    }

    # Compute the final phase parameter psi for each PN order
    x_result = {order: (tau**(-1 / 4) / 4) * value for order, value in x_brackets.items()}

    return x_result

def x_of_t(t, q):
    """
    Computes the dimensionless post-Newtonian (PN) frequency expansion parameter x
    given the coordinate time t and mass ratio q.
    
    Parameters
    ----------
    t : float
        Coordinate time.
    q : float
        Mass ratio defined as q = m1 / m2, with m1 >= m2.
    
    Returns
    -------
    x_val : float
        The value of the dimensionless PN parameter x.
    x_terms : dict
        A dictionary where each key represents a PN order (e.g., '1.0', '1.5', etc.),
        and the value is the cumulative value of x up to that PN order.
        
        Example:
        x_terms = {
            '1.0': term_1,
            '1.5': term_1 + term_2,
            '2.0': term_1 + term_2 + term_3,
            ...
        }
    
    Notes
    -----
    The PN expansion includes terms up to 4.5 PN order.
    """
    # PN time
    tau = tau_of_t(t, q)

    # x parameters at different PN orders
    x_results = x_of_tau(tau, q)
    
    return x_results

def E_of_x(x, q):
    """
    Computes the invariant energy of a binary system in the post-Newtonian (PN) approximation at 4PN order.
    
    The energy expression is given by:
    E = - (m * nu * c^2 * x / 2) * {1 + (...)}

    Parameters
    ----------
    x : float
        PN expansion parameter.
    q : float
        Mass ratio defined as q = m1 / m2
    
    Returns
    -------
    E_result : dictionary
        The invariant energy of the binary system at different PN orders.

    Eq(3) of https://arxiv.org/pdf/2304.11185
    """

    # Symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)
    
    # overall prefactor
    prefactor = - (nu * x / 2)

    # Newtonian
    term_1 =  1

    # 1 PN 
    term_2 =  (-3/4 - nu/12) * x 

    # 2 PN
    term_3 =  (-27/8 + 19/8 * nu - nu**2 / 24) * x**2 

    # 3 PN
    term_4 =  (-675/64 + (34445/576 - 205/96 * np.pi**2) * nu - 155/96 * nu**2 - 35/5184 * nu**3) * x**3 

    # 4 PN
    term_5 =  ((-3969/128 + (-123671/5760 + 9037/1536 * np.pi**2 + 896/15 * np.euler_gamma + 448/15 * np.log(16 * x)) * nu + \
                (-498449/3456 + 3157/576 * np.pi**2) * nu**2 + 301/1728 * nu**3 + 77/31104 * nu**4) ) * x**4
    

    # Summing terms at each PN order and storing in dictionary for easy access
    E_brackets = {
        '0.0': term_1,
        '1.0': term_1 + term_2,
        '2.0': term_1 + term_2 + term_3,
        '3.0': term_1 + term_2 + term_3 + term_4,
        '4.0': term_1 + term_2 + term_3 + term_4 + term_5
    }
    
    # Compute the final phase parameter psi for each PN order
    E_result = {order: prefactor * value for order, value in E_brackets.items()}
    
    return E_result


def F_of_x(x, q):
    """
    Computes the energy flux of a binary system in the post-Newtonian (PN) approximation at 4.5PN order.
    
    The flux expression is given by:
    F = (32 * c^5 / (5 * G)) * nu^2 * x^5 * {1 + (...)}

    Parameters
    ----------
    x : float
        PN expansion parameter.
    q : float
        Mass ratio defined as q = m1 / m2
    
    Returns
    -------
    F_result : dictionary
        The energy flux of the binary system at different PN orders.

    Eq(4) of https://arxiv.org/pdf/2304.11185
    """

    # Symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)
    
    # overall prefactor
    prefactor = (32 / 5) * nu**2 * x**5 

    # Newtonian
    term_1 =  1

    # 1 PN 
    term_2 =  (-1247/336 - 35/12 * nu) * x 

    # 1.5 PN
    term_3 = 4 * np.pi * x**(3/2)
    
    # 2 PN
    term_4 =  (-44711/9072 + 9271/504 * nu + 65/18 * nu**2) * x**2

    # 2.5 PN
    term_5 = (-8191/672 - 583/24 * nu) * np.pi * x**(5/2)
    
    # 3 PN
    term_6 =  (6643739519/69854400 + 16/3 * np.pi**2 - 1712/105 * np.euler_gamma - 856/105 * np.log(16 * x) +
         (-134543/7776 + 41/48 * np.pi**2) * nu - 94403/3024 * nu**2 - 775/324 * nu**3) * x**3

    # 3.5 PN
    term_7 =  (-16285/504 + 214745/1728 * nu + 193385/3024 * nu**2) * np.pi * x**(7/2)
    
    # 4 PN
    term_8 = (-323105549467/3178375200 + 232597/4410 * np.euler_gamma - 1369/126 * np.pi**2 +
         39931/294 * np.log(2) - 47385/1568 * np.log(3) + 232597/8820 * np.log(x) +
         (-1452202403629/1466942400 + 41478/245 * np.euler_gamma - 267127/4608 * np.pi**2 +
          479062/2205 * np.log(2) + 47385/392 * np.log(3) + 20739/245 * np.log(x)) * nu +
         (1607125/6804 - 3157/384 * np.pi**2) * nu**2 + 6875/504 * nu**3 + 5/6 * nu**4) * x**4

    # 4.5 PN
    term_9 = (265978667519/745113600 - 6848/105 * np.euler_gamma - 3424/105 * np.log(16 * x) +
         (2062241/22176 + 41/12 * np.pi**2) * nu) * np.pi * x**(9/2)
    
    # Summing terms at each PN order and storing in dictionary for easy access
    F_brackets = {
        '0.0': term_1,
        '1.0': term_1 + term_2,
        '1.5': term_1 + term_2 + term_3,
        '2.0': term_1 + term_2 + term_3 + term_4,
        '2.5': term_1 + term_2 + term_3 + term_4 + term_5,
        '3.0': term_1 + term_2 + term_3 + term_4 + term_5 + term_6,
        '3.5': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7,
        '4.0': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8,
        '4.5': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
    }
    
    # Compute the final phase parameter psi for each PN order
    F_result = {order: prefactor * value for order, value in F_brackets.items()}
    
    return F_result


def H22_of_x(x, q):
    """
    Compute the complex amplitude of the (2,2) mode of the gravitational wave strain
    in the Post-Newtonian (PN) expansion up to the 4.0 PN order.
    
    Parameters
    ----------
    x : float
        Dimensionless PN parameter (related to the orbital velocity).
    q : float
        Mass ratio, defined as q = m1 / m2 
        where m1 and m2 are the component masses.
    
    Returns
    -------
    H_22_result : dict
        A dictionary where each key represents a PN order (e.g., '0.0', '1.0', etc.),
        and the value is the cumulative sum of terms up to that PN order.
        
    Notes
    -----
    The PN expansion includes terms up to 4.0 PN order, with both real and imaginary 
    contributions to account for phase and amplitude effects in the waveform.
    The terms are derived based on equations for the (2,2) mode amplitude in the PN expansion.

    Eq.(11) of https://arxiv.org/pdf/2304.11185
    """
    
    # Symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)

    # 0 PN (Newtonian term)
    term_0 = 1

    # 1 PN term
    term_1 = (-107 / 42 + 55 / 42 * nu) * x

    # 1.5 PN term (first appearance of imaginary component, includes Pi terms)
    term_2 = 2 * np.pi * x**(3 / 2)

    # 2 PN term
    term_3 = (-2173 / 1512 - 1069 / 216 * nu + 2047 / 1512 * nu**2) * x**2

    # 2.5 PN term (imaginary component, complex amplitude term)
    term_4 = (-107 * np.pi / 21 + (34 * np.pi / 21 - 24j) * nu) * x**(5 / 2)

    # 3 PN term
    term_5 = (
        (27027409 / 646800 - 856 / 105 * np.euler_gamma + 428j * np.pi / 105 + 2 * np.pi**2 / 3
        + (-278185 / 33264 + 41 * np.pi**2 / 96) * nu - 20261 / 2772 * nu**2
        + 114635 / 99792 * nu**3 - 428 / 105 * np.log(16 * x)) * x**3
    )

    # 3.5 PN term
    term_6 = (
        (-2173 * np.pi / 756 + (-2495 * np.pi / 378 + 14333j / 162) * nu
         + (40 * np.pi / 27 - 4066j / 945) * nu**2) * x**(7 / 2)
    )

    # 4 PN term (includes real and imaginary contributions, up to nu^4)
    term_7 = (
        (-846557506853 / 12713500800 + 45796 / 2205 * np.euler_gamma - 22898j * np.pi / 2205
         - 107 * np.pi**2 / 63 + 22898 / 2205 * np.log(16 * x)) * x**4
        + (-336005827477 / 4237833600 + 15284 / 441 * np.euler_gamma - 219314j * np.pi / 2205
           - 9755 * np.pi**2 / 32256 + 7642 / 441 * np.log(16 * x)) * nu
        + (256450291 / 7413120 - 1025 * np.pi**2 / 1008) * nu**2
        - 81579187 / 15567552 * nu**3
        + 26251249 / 31135104 * nu**4
    ) * x**4

    # Summing up terms at each PN order and storing in dictionary for easy access
    H_22_result = {
        '0.0': term_0,
        '1.0': term_0 + term_1,
        '1.5': term_0 + term_1 + term_2,
        '2.0': term_0 + term_1 + term_2 + term_3,
        '2.5': term_0 + term_1 + term_2 + term_3 + term_4,
        '3.0': term_0 + term_1 + term_2 + term_3 + term_4 + term_5,
        '3.5': term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6,
        '4.0': term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7
    }
    
    return H_22_result


def psi_of_x(x, q, psi_0):
    """
    Compute the time-domain GW half-phase parameter psi in the Post-Newtonian (PN) expansion up to 4.5 PN order.

    Parameters
    ----------
    x : float
        Dimensionless PN parameter (related to the orbital velocity).
    q : float
        Mass ratio, defined as q = m1 / m2
        m1 and m2 are the component masses.
    psi_0 : float
        Constant phase term that sets the initial phase offset.
    
    Returns
    -------
    psi_result : dict
        A dictionary where each key represents a PN order (e.g., '1.0', '1.5', etc.),
        and the value is the cumulative phase up to that PN order.
        
        Example:
        psi_result = {
            '1.0': term_1,
            '2.0': term_1 + term_2,
            '2.5': term_1 + term_2 + term_3,
            '3.0': term_1 + term_2 + term_3 + term_4,
            '3.5': term_1 + term_2 + term_3 + term_4 + term_5,
            '4.0': term_1 + term_2 + term_3 + term_4 + term_5 + term_6,
            ...
        }
    
    Notes
    -----
    The PN expansion includes terms up to 4.5 PN order, accounting for the real components 
    of the gravitational wave phase.
    
    Eq.(8) of https://arxiv.org/pdf/2304.11185
    """
    
    # Symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)

    # 0 PN term
    term_1 = 1

    # 1 PN term
    term_2 = (3715 / 1008 + 55 / 12 * nu) * x

    # 1.5 PN term (includes Pi term)
    term_3 = -10 * np.pi * x**(3 / 2)

    # 2 PN term
    term_4 = (15293365 / 1016064 + 27145 / 1008 * nu + 3085 / 144 * nu**2) * x**2

    # 2.5 PN term (logarithmic correction)
    term_5 = (38645 / 1344 - 65 / 16 * nu) * np.pi * x**(5 / 2) * np.log(x)

    # 3 PN term (real component with complex logarithmic dependencies)
    term_6 = (
        12348611926451 / 18776862720 - 160 * np.pi**2 / 3 - 1712 / 21 * np.euler_gamma
        - 856 / 21 * np.log(16 * x) 
        + (-15737765635 / 12192768 + 2255 * np.pi**2 / 48) * nu
        + 76055 / 6912 * nu**2 - 127825 / 5184 * nu**3
    ) * x**3

    # 3.5 PN term
    term_7 = (
        (77096675 / 2032128 + 378515 / 12096 * nu - 74045 / 6048 * nu**2) * np.pi * x**(7 / 2)
    )

    # 4 PN term (logarithmic and other complex dependencies up to nu^4)
    term_8 = (
        2550713843998885153 / 2214468081745920 - 9203 / 126 * np.euler_gamma - 45245 * np.pi**2 / 756
        - 252755 / 2646 * np.log(2) - 78975 / 1568 * np.log(3) - 9203 / 252 * np.log(x) 
        + (-680712846248317 / 337983528960 - 488986 / 1323 * np.euler_gamma + 109295 * np.pi**2 / 1792
           - 1245514 / 1323 * np.log(2) + 78975 / 392 * np.log(3) - 244493 / 1323 * np.log(x)) * nu
        + (7510073635 / 24385536 - 11275 * np.pi**2 / 1152) * nu**2
        + 1292395 / 96768 * nu**3 - 5975 / 768 * nu**4
    ) * x**4

    # 4.5 PN term (complex Pi term with logarithmic dependence)
    term_9 = (
        (-93098188434443 / 150214901760 + 1712 / 21 * np.euler_gamma + 80 * np.pi**2 / 3
         + 856 / 21 * np.log(16 * x)) 
        + (1492917260735 / 1072963584 - 2255 * np.pi**2 / 48) * nu 
        - 45293335 / 1016064 * nu**2 - 10323755 / 1596672 * nu**3
    ) * np.pi * x**(9 / 2)

    # Summing terms at each PN order and storing in dictionary for easy access
    psi_brackets = {
        '0.0': term_1,
        '1.0': term_1 + term_2,
        '1.5': term_1 + term_2 + term_3,
        '2.0': term_1 + term_2 + term_3 + term_4,
        '2.5': term_1 + term_2 + term_3 + term_4 + term_5,
        '3.0': term_1 + term_2 + term_3 + term_4 + term_5 + term_6,
        '3.5': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7,
        '4.0': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8,
        '4.5': term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
    }
    
    # Compute the final phase parameter psi for each PN order
    psi_result = {order: psi_0 - x**(-5 / 2) / (32 * nu) * value for order, value in psi_brackets.items()}
    
    return psi_result


def h_of_x(x, q, psi_0=0.0, amplitude_order='4.0', phase_order='4.5'):
    """
    Computes the complex (2,2) gravitational wave mode in the post-Newtonian (PN) 
    approximation for a binary system, given coordinate time t and mass ratio.
    
    Parameters
    ----------
    x : float
        The PN expansion parameter
    q : float
        Mass ratio of the binary system, defined as q = m1 / m2 with m1 >= m2.
    psi_0 : float, optional
        Initial phase parameter (default is 0.0).
    amplitude_order : str, optional
        The PN order for the amplitude calculation (default is '4.0').
    phase_order : str, optional
        The PN order for the phase calculation (default is '4.5').
    
    Returns
    -------
    h22_pn : complex
        The (2,2) mode of the gravitational wave strain in the PN approximation.

    Eq.(10) of https://arxiv.org/pdf/2304.11185
    """
    # Symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)
 
    # Calculate the complex amplitude
    H22_dict = H22_of_x(x, q)
    if amplitude_order in H22_dict.keys():
        H22 = H22_dict[amplitude_order]
    else:
        raise ValueError(f"Amplitude calculations at {amplitude_order} do not exist")
        
    # Calculate the time-domain GW half-phase parameter psi
    psi_dict = psi_of_x(x, q, psi_0)
    if phase_order in psi_dict.keys():
        psi = psi_dict[phase_order]
    else:
        raise ValueError(f"Phase calculations at {phase_order} do not exist")
    
    # Overall amplitude of the GWs
    prefactor = 8 * nu * x * np.sqrt(np.pi / 5)
    
    # Mode index for (2,2) mode
    ell = 2
    h22_pn = prefactor * H22 * np.exp(-1j * ell * psi)

    return h22_pn


def h_of_t(t, q, psi_0=0.0, x_order='4.5', amplitude_order='4.0', phase_order='4.5'):
    """
    Computes the complex (2,2) gravitational wave mode in the post-Newtonian (PN) 
    approximation for a binary system, given coordinate time t and mass ratio.
    
    Parameters
    ----------
    t : float
        The time variable
    q : float
        Mass ratio of the binary system, defined as q = m1 / m2 with m1 >= m2.
    psi_0 : float, optional
        Initial phase parameter (default is 0.0).
    x_order : str, optional
        The PN order for the expansion parameter (default is '4.5').
    amplitude_order : str, optional
        The PN order for the amplitude calculation (default is '4.0').
    phase_order : str, optional
        The PN order for the phase calculation (default is '4.5').
    
    Returns
    -------
    h22_pn : complex
        The (2,2) mode of the gravitational wave strain in the PN approximation.
        
    Eq.(10) of https://arxiv.org/pdf/2304.11185
    """
    # Symmetric mass ratio
    nu = mass_ratio_to_symmetric_mass_ratio(q)

    # PN time
    tau = tau_of_t(t, q)
    
    # Calculate the PN frequency parameter x based on tau and mass ratio
    x_dict = x_of_tau(tau, q)
    if x_order in x_dict.keys():
        x = x_dict[x_order]
    else:
        raise ValueError(f"x calculations at {x_order} do not exist")
        
    # Calculate the time-domain GW (2,2) mode
    h22_pn = h_of_x(x, q, psi_0=psi_0, amplitude_order=amplitude_order, phase_order=phase_order)

    return h22_pn


class Blanchet2024:
    def __init__(self, q, t=None, tc=0.0, psi_0=0.0, x=None):
        """
        Compute energy E, fluxes F, and gravitational wave (2,2) mode ht given mass ratio q and time,
        following https://arxiv.org/pdf/2304.11185.
        
        Parameters
        ----------
        q : float
            Mass ratio.
        t : array-like, optional
            Coordinate time array. If None, defaults to a range from -30000 to 0 with steps of 0.1.
        tc : float, optional
            Time at merger. Defaults to 0.0.
        psi_0 : float, optional
            Initial phase. Defaults to 0.0.
        x : array-like, optional
            Expansion parameter array. If None, it's calculated based on t.
        """
        self.q = q
        self.t = t if t is not None else np.arange(-30000, 0, 0.1)
        self.tc = tc
        self.psi_0 = psi_0
        self.calculate_h_from_x = x is not None
        
        if self.calculate_h_from_x:
            self.x = x
        else:
            self.x = x_of_t(self.t, self.q)

        self.E_dict = self.compute_energy()
        self.F_dict = self.compute_fluxes()
        self.ht_dict = self.compute_waveform()
        self.phase_dict = self.compute_phases()
        self.omega_dict = self.compute_frequencies()

    def compute_energy(self):
        """Compute energy E for different PN orders."""
        E = {key: E_of_x(self.x[key], self.q) for key in ['0.0', '1.0', '2.0', '3.0', '4.0']}
        E['4.5'] = E_of_x(self.x['4.5'], self.q)
        return E

    def compute_fluxes(self):
        """Compute fluxes F for different PN orders."""
        F = {key: F_of_x(self.x[key], self.q) for key in ['0.0', '1.0', '2.0', '3.0', '4.0']}
        F['4.5'] = F_of_x(self.x['4.5'], self.q)
        return F

    def compute_waveform(self):
        """Compute gravitational waveforms ht based on whether x or t is used."""
        ht = {key: np.zeros(len(self.t)) for key in ['0.0', '1.0', '2.0', '3.0', '4.0']}
        
        if self.calculate_h_from_x:
            for key in ht.keys():
                ht[key] = h_of_x(self.t, self.q, psi_0=self.psi_0, amplitude_order=key, phase_order=key)
            ht['4.5'] = h_of_x(self.t, self.q, psi_0=self.psi_0, amplitude_order='4.0', phase_order='4.5')
        else:
            for key in ht.keys():
                ht[key] = h_of_t(self.t, self.q, psi_0=self.psi_0, x_order=key, amplitude_order=key, phase_order=key)
            ht['4.5'] = h_of_t(self.t, self.q, psi_0=self.psi_0, x_order='4.5', amplitude_order='4.0', phase_order='4.5')

        return ht

    def compute_phases(self):
        """Compute the overall phases of gravitational waveforms ht at different PN orders"""
        phase = {}
        for key in self.ht_dict.keys():
            phase[key] = get_phase(self.ht_dict[key])
        return phase

    def compute_frequencies(self):
        """Compute overall frequency of gravitational waveforms ht at different PN orders"""
        omega = {}
        for key in self.ht_dict.keys():
            omega[key] = get_frequency(self.t, self.ht_dict[key])
        return omega

    def plot_waveform(self):
        """Plot the waveform ht for different PN orders."""
        for key in self.ht_dict.keys():
            plt.plot(self.t, self.ht_dict[key], label=f'PN Order {key}')
        plt.ylim(-0.5, 0.5)
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Strain', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=13)
        plt.grid()
        plt.show()