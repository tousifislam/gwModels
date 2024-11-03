#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: quadrupole.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from ...utils import mass_ratio_to_symmetric_mass_ratio

def compute_restricted_quadrupolar_waveform(q, r, rdot, phi, phidot):
    """
    Compute the gravitational waveform h.
    Eq 14 and 15 of https://arxiv.org/pdf/0806.1037
    
    Parameters:
        q (float): Mass ratio.
        r (float): Radial distance.
        rdot (float): Time derivative of r (r_dot/dt).
        phi (float): Angle in radians.
        phidot (float): Time derivative of phi (phi_dot/dt).

    Returns:
        complex: Computed value of h.
    """
    
    nu = mass_ratio_to_symmetric_mass_ratio(q)
    term1 = 1 / r
    term2 = (phidot * r + 1j * rdot) ** 2
    h = -4 * nu * np.exp(-2j * phi) * np.sqrt(np.pi / 5) * (term1 + term2)
    
    return h


def H_0PN_22_NS(r, r_dot, phi_dot):
    """
    Computes the 0PN term (H_{0PN}^{22, NS}) for the gravitatonal waveforms
    Following eq.(A48) of https://arxiv.org/pdf/2409.17636

    Parameters:
    nu : float
        Symmetric mass ratio of the binary system.
    r : float
        Radial separation between the components.
    r_dot : float
        Radial velocity of the system.
    phi_dot : float
        Angular velocity of the system.

    Returns:
    float
        The value of H_{0PN}^{22, NS}.
    """
    # Newtonian part of H_inst^(22)
    term1 = 1 / r
    term2 = r**2 * phi_dot**2
    term3 = 2j * r * r_dot * phi_dot
    term4 = -r_dot**2
    return term1 + term2 + term3 + term4


def H_1PN_22_NS(r, r_dot, phi_dot, nu):
    """
    Computes the 1PN term  H_{1PN}^{22,NS} for gravitational wave calculations
    in geometrized units (G = c = M = 1).
    Following eq.(A49) of https://arxiv.org/pdf/2409.17636
    
    Parameters
    ----------
    r : float
        Radial separation parameter.
    r_dot : float
        Radial velocity parameter (dot r).
    phi_dot : float
        Angular velocity parameter (dot phi).
    nu : float
        Symmetric mass ratio.

    Returns
    -------
    H_1PN : float
        The value of H_{1PN}^{22,NS}
    """
    # First term: (G^2 M^2 / r^2) term
    term1 = (-5 + nu / 2) / r**2
    
    # Term involving (G M * r_dot^2 / r)
    term2 = (-15 / 14 - 16 * nu / 7) * r_dot**2 / r
    
    # Term involving r_dot^4
    term3 = (-9 / 14 + 27 * nu / 14) * r_dot**4
    
    # Term involving r * r_dot^3 * phi_dot
    term4 = (9j / 7 - 27j * nu / 7) * r * r_dot**3 * phi_dot
    
    # Term involving (G M * r * phi_dot^2)
    term5 = (11 / 42 + 26 * nu / 7) * r * phi_dot**2
    
    # Term involving r^4 * phi_dot^4
    term6 = (9 / 14 - 27 * nu / 14) * r**4 * phi_dot**4
    
    # Term involving r_dot * (G M * phi_dot + r^3 * phi_dot^3)
    term7 = (25j / 21 + 45j * nu / 7) * r_dot * phi_dot / r
    term8 = (9j / 7 - 27j * nu / 7) * r_dot * r**3 * phi_dot**3
    
    # Sum all terms
    H_1PN = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8
    return H_1PN


def H_2PN_NS_22(r, r_dot, phi_dot, nu):
    """
    Computes the 2PN term H_{2PN}^{22,NS} for gravitational wave calculations.
    Following eq.(A50) of https://arxiv.org/pdf/2409.17636
    
    Parameters
    ----------
    r : float
        Radial separation parameter.
    r_dot : float
        Radial velocity parameter (dot r).
    phi_dot : float
        Angular velocity parameter (dot phi).
    nu : float
        Symmetric mass ratio.

    Returns
    -------
    H_2PN : float
        The value of H_{2PN}^{22,NS}
    """
    # First term
    term1 = (757/63 + 181 * nu / 36 + 79 * nu**2 / 126) / r**3
    
    # Terms involving r_dot^6 and r * r_dot^5 * phi_dot
    term2 = (-83/168 + 589 * nu / 168 - 1111 * nu**2 / 168) * r_dot**6
    term3 = (83j / 84 - 589j * nu / 84 + 1111j * nu**2 / 84) * r * r_dot**5 * phi_dot

    # Terms involving phi_dot^2 and r * phi_dot^4
    term4 = (-11891 / 1512 - 5225 * nu / 216 + 13133 * nu**2 / 1512) * phi_dot**2
    term5 = (835 / 252 + 19 * nu / 252 - 2995 * nu**2 / 252) * r**3 * phi_dot**4
    
    # Terms involving r^6 * phi_dot^6
    term6 = (83 / 168 - 589 * nu / 168 + 1111 * nu**2 / 168) * r**6 * phi_dot**6

    # Terms involving r_dot^4 * (GM/r + r^2 * phi_dot^2)
    term7 = (-557 / 168 + 83 * nu / 21 + 214 * nu**2 / 21) * r_dot**4 / r
    term8 = (-83 / 168 + 589 * nu / 168 - 1111 * nu**2 / 168) * r_dot**4 * r**2 * phi_dot**2

    # Terms involving r_dot^3 * (GM * phi_dot + r^3 * phi_dot^3)
    term9 = (863j / 126 - 731j * nu / 63 - 211j * nu**2 / 9) * r_dot**3 * phi_dot / r
    term10 = (83j / 42 - 589j * nu / 42 + 1111j * nu**2 / 42) * r_dot**3 * r**3 * phi_dot**3

    # Terms involving r_dot^2 * (GM/r^2 + GM*r*phi_dot^2 + r^4 * phi_dot^4)
    term11 = (619 / 252 - 2789 * nu / 252 - 467 * nu**2 / 126) * r_dot**2 / r**2
    term12 = (11 / 28 - 169 * nu / 14 - 58 * nu**2 / 21) * r_dot**2 * r * phi_dot**2
    term13 = (83 / 168 - 589 * nu / 168 + 1111 * nu**2 / 168) * r_dot**2 * r**4 * phi_dot**4

    # Terms involving r_dot * (GM/r * phi_dot + GM * r^2 * phi_dot^3 + r^5 * phi_dot^5)
    term14 = (-773j / 189 - 3767j * nu / 189 + 2852j * nu**2 / 189) * r_dot * phi_dot / r
    term15 = (433j / 84 + 103j * nu / 12 - 1703j * nu**2 / 84) * r_dot * r**2 * phi_dot**3
    term16 = (83j / 84 - 589j * nu / 84 + 1111j * nu**2 / 84) * r_dot * r**5 * phi_dot**5

    # Sum all terms
    H_2PN = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13 + term14 + term15 + term16
    return H_2PN


def H_2p5PN_22_NS(r, r_dot, phi_dot, nu):
    """
    Compute the 2.5PN component H_{2.5PN}^{22,NS} for gravitational wave calculations
    Following eq.(A51) of https://arxiv.org/pdf/2409.17636
    
    Parameters
    ----------
    r : float
        Radial separation parameter.
    r_dot : float
        Radial velocity parameter (dot r).
    phi_dot : float
        Angular velocity parameter (dot phi).
    nu : float
        Symmetric mass ratio.

    Returns
    -------
    H_2_5PN : complex
        The value of H_{2.5PN}^{22,NS}
    """
    # Term: -122 G^2 M^2 nu r_dot^3 / (35 r^2)
    term1 = -122 * nu * r_dot**3 / (35 * r**2)
    
    # Term: -468 i G^3 M^3 nu phi_dot / (35 r^2)
    term2 = -468j * nu * phi_dot / (35 * r**2)
    
    # Term: 184 i G^2 M^2 nu r_dot^2 phi_dot / (35 r)
    term3 = 184j * nu * r_dot**2 * phi_dot / (35 * r)
    
    # Term: -316 i G^2 M^2 r nu phi_dot^3 / 35
    term4 = -316j * nu * r * phi_dot**3 / 35
    
    # Term involving r_dot * (G^3 M^3 nu / (105 r^3) - 121 G^2 M^2 nu phi_dot^2 / 5)
    term5 = r_dot * (2 * nu / (105 * r**3) - 121 * nu * phi_dot**2 / 5)
    
    # Sum all terms
    H_2_5PN = term1 + term2 + term3 + term4 + term5
    return H_2_5PN

    
def H_3PN_NS_22(r, r_dot, phi_dot, nu):
    """
    Computes the 3PN term (H_{3PN}^{22, NS}) for the gravitatonal waveforms
    Following eq.(A52) of https://arxiv.org/pdf/2409.17636

    Parameters:
    nu : float
        Symmetric mass ratio of the binary system.
    r : float
        Radial separation between the components.
    r_dot : float
        Radial velocity of the system.
    phi_dot : float
        Angular velocity of the system.

    Returns:
    float
        The value of H_{3PN}^{22, NS}.
    """
    # First main term with 1/r^4 dependence
    term1 = (1 / r**4) * (
        -512714/51975 + (-1375951/13860 + 41 * (3.14159)**2 / 16) * nu + 
        1615 * nu**2 / 616 + 2963 * nu**3 / 4158
    )
    
    # Terms with dot(r)^8 and r_dot^7 * phi_dot
    term2 = (
        (-507/1232 + 6101 * nu / 1232 - 12525 * nu**2 / 616 + 34525 * nu**3 / 1232) * r_dot**8 +
        (1 / r) * (507j/616 - 6101j * nu / 616 + 12525j * nu**2 / 308 - 34525j * nu**3 / 616) * r_dot**7 * phi_dot
    )
    
    # Terms with 1/r * phi_dot^2 and r^2 * phi_dot^4
    term3 = (
        (1 / r) * (42188851/415800 + (190703/3465 - 123 * (3.14159)**2 / 64) * nu - 
                   18415 * nu**2 / 308 + 281473 * nu**3 / 16632) * phi_dot**2 +
        r**2 * (328813/55440 - 374651 * nu / 33264 + 249035 * nu**2 / 4158 - 
                1340869 * nu**3 / 33264) * phi_dot**4
    )
    
    # Terms with r^5 * phi_dot^6 and r^8 * phi_dot^8
    term4 = (
        r**5 * (12203/2772 - 36427 * nu / 2772 - 13667 * nu**2 / 1386 + 
                49729 * nu**3 / 924) * phi_dot**6 +
        r**8 * (507/1232 - 6101 * nu / 1232 + 12525 * nu**2 / 616 - 34525 * nu**3 / 1232) * phi_dot**8
    )
    
    # Terms with dot(r)^4
    term5 = (
        (1 / r**2) * (-92567/13860 + 7751 * nu / 396 + 400943 * nu**2 / 11088 + 
                      120695 * nu**3 / 3696) * r_dot**4 +
        r * (-42811/11088 + 6749 * nu / 1386 + 19321 * nu**2 / 693 - 
             58855 * nu**3 / 1386) * r_dot**4 * phi_dot**2
    )
    
    # Terms with dot(r)^6 and dot(r)^5
    term6 = (
        (1 / r) * (-5581/1232 + 4694 * nu / 231 - 3365 * nu**2 / 462 - 
                   1850 * nu**3 / 33) * r_dot**6 +
        r**2 * (-507/616 + 6101 * nu / 616 - 12525 * nu**2 / 308 + 
                34525 * nu**3 / 616) * r_dot**6 * phi_dot**2
    )
    
    term7 = (
        (17233j/1848 - 31532j * nu / 693 + 65575j * nu**2 / 2772 + 
         85145j * nu**3 / 693) * r_dot**5 * phi_dot +
        r**3 * (1521j/616 - 18303j * nu / 616 + 37575j * nu**2 / 308 - 
                103575j * nu**3 / 616) * r_dot**5 * phi_dot**3
    )
    
    # Terms with dot(r)^3
    term8 = (
        (1 / r) * (39052j/3465 - 154114j * nu / 2079 - 246065j * nu**2 / 4158 - 
                   365725j * nu**3 / 4158) * r_dot**3 * phi_dot +
        r**2 * (13867j/792 - 191995j * nu / 2772 - 8741j * nu**2 / 5544 + 
                52700j * nu**3 / 231) * r_dot**3 * phi_dot**3 +
        r**5 * (1521j/616 - 18303j * nu / 616 + 37575j * nu**2 / 308 - 
                103575j * nu**3 / 616) * r_dot**3 * phi_dot**5
    )
    
    # Terms with dot(r)^2
    term9 = (
        (1 / r**3) * (913799/29700 + (174679/2310 + 123 * (3.14159)**2 / 32) * nu - 
                      158215 * nu**2 / 2772 - 12731 * nu**3 / 4158) * r_dot**2 +
        (1 / r) * (20191/18480 - 3879065 * nu / 33264 - 411899 * nu**2 / 8316 - 
                   522547 * nu**3 / 33264) * r_dot**2 * phi_dot**2
    )
    
    term10 = (
        r**3 * (381/77 - 101237 * nu / 2772 + 247505 * nu**2 / 5544 + 
                394771 * nu**3 / 5544) * r_dot**2 * phi_dot**4 +
        r**6 * (507/616 - 6101 * nu / 616 + 12525 * nu**2 / 308 - 
                34525 * nu**3 / 616) * r_dot**2 * phi_dot**6
    )
    
    # Term with dot(r) * phi_dot and higher powers
    term11 = (
        (1 / r**2) * (-68735j/378 + (-57788j / 315 + 123j * (3.14159)**2 / 32) * nu - 
                      701j * nu**2 / 27 + 11365j * nu**3 / 378) * r_dot * phi_dot +
        r * (91229j/13860 + 97861j * nu / 4158 + 919811j * nu**2 / 8316 - 
             556601j * nu**3 / 8316) * r_dot * phi_dot**3
    )
    
    term12 = (
        r**4 * (6299j/792 - 68279j * nu / 5544 - 147673j * nu**2 / 2772 + 
                541693j * nu**3 / 5544) * r_dot * phi_dot**5 +
        r**7 * (507j/616 - 6101j * nu / 616 + 12525j * nu**2 / 308 - 
                34525j * nu**3 / 616) * r_dot * phi_dot**7
    )

    # Sum up all terms
    H_3PN_NS_22 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12

    return H_3PN_NS_22


def H_inst_22_total(nu, r, r_dot, phi, phi_dot):
    """
    Function to compute total instantaneous complex amplitude at 3PN accuracy
    Following Appendix A of https://arxiv.org/pdf/2409.17636

    Parameters:
        nu : float
            Symmetric mass ratio of the binary system.
        r : float
            Radial separation between the components.
        r_dot : float
            Radial velocity of the system.
        phi : float
            Orbital phase of the system.
        phi_dot : float
            Angular velocity of the system.

    Returns:
        float
            The value of H^{22, NS}.
    """
    # Newtonian, 1PN, 2PN and 3PN contributions to H_inst^(22)
    H_0PN = H_0PN_22_NS(r, r_dot, phi_dot)
    H_1PN = H_1PN_22_NS(r, r_dot, phi_dot, nu)
    H_2PN = H_2PN_NS_22(r, r_dot, phi_dot, nu)
    H_2p5PN = H_2p5PN_22_NS(r, r_dot, phi_dot, nu)
    H_3PN = H_3PN_NS_22(r, r_dot, phi_dot, nu)
    
    # Total H_inst^(22)
    H_inst_22_total = H_0PN + H_1PN + H_2PN + H_2p5PN + H_3PN
    
    return H_inst_22_total


def h_inst_22(nu, r, r_dot, phi, phi_dot):
    """
    Function to compute complex waveform h_inst^(22) at 3PN accuracy
    Following Appendix A of https://arxiv.org/pdf/2409.17636

    Parameters:
        nu : float
            Symmetric mass ratio of the binary system.
        r : float
            Radial separation between the components.
        r_dot : float
            Radial velocity of the system.
        phi : float
            Orbital phase of the system.
        phi_dot : float
            Angular velocity of the system.

    Returns:
        float
            The value of H^{22, NS}.
    """
    
    # Total H_inst^(22)
    H_inst_22_total = H_inst_22_total(nu, r, r_dot, phi, phi_dot)
    
    # Instantaneous strain for l=2, m=2 mode
    prefactor = (4 * nu) * np.sqrt(np.pi / 5)
    h_inst =  prefactor * np.exp(2j * phi) * H_inst_22_total
    
    return h_inst