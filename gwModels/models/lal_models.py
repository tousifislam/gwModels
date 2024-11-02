#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#     FILE: lal_models
#
#     AUTHOR: Tousif Islam
#     CREATED: 07-02-2024
#     LAST MODIFIED: Tue Feb  7 17:58:52 2024
#     REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import warnings
warnings.filterwarnings('ignore')

import lal
import lalsimulation as lalsim
import numpy as np
import time
import gwtools

def generate_EccentricTD(mass_ratio, total_mass=10.0, inclination=0.0, 
                         ref_phase=0.0, eccentricity=0.01, Momega0OverM=0.02, 
                         deltaTOverM=0.5, time_wfgeneration=True):
    """
    Generates an eccentric time-domain gravitational waveform using the LAL EccentricTD approximant.

    This function serves as a wrapper for generating gravitational waveforms for binary systems 
    with non-zero eccentricity. The waveform generation is based on LALSuite's EccentricTD model, 
    with additional geometric scaling for convenience in astrophysical analysis.

    Parameters
    ----------
    mass_ratio : float
        The mass ratio of the binary system, defined as `m1/m2`, where `m1` and `m2` 
        are the masses of the individual compact holes.
    total_mass : float, optional
        The total mass of the binary system in solar masses (Msun), by default 10.0.
    inclination : float, optional
        The angle between the binary's angular momentum vector and the observer's line-of-sight, 
        in radians. Default is 0.0 (face-on).
    ref_phase : float, optional
        The reference phase of the waveform at the initial frequency, in radians. Default is 0.0.
    eccentricity : float, optional
        The eccentricity of the binary orbit at the reference frequency. Default is 0.01.
    Momega0OverM : float, optional
        The dimensionless initial orbital frequency of the binary system, defined in 
        geometric units (GM/c^3). Default is 0.02.
    deltaTOverM : float, optional
        The time-step for waveform generation in geometric units, specified as a fraction of 
        the total mass (GM/c^3). Default is 0.5.
    time_wfgeneration : bool, optional
        If True, the function prints the time taken to generate the waveform. Default is True.

    Returns
    -------
    t_geo : numpy.ndarray
        Array of time samples in geometric units (seconds), normalized by the total mass.
    h_geo : numpy.ndarray
        Complex array of the gravitational waveform in geometric units, including both the 
        `h+` and `hx` polarizations as a single complex waveform.

    Notes
    -----
    - This function preprocesses the parameters and internally calls LAL's `SimInspiralChooseTDWaveform` 
      function with the EccentricTD approximant.
    - The waveform output is converted to geometric units to enable consistent scaling 
      across different masses and distances.
    - The waveform is generated at a fiducial distance of 1000 Mpc, which is then normalized 
      to geometric units for ease of interpretation.

    Example
    -------
    >>> t_geo, h_geo = generate_EccentricTD(mass_ratio=1.2, total_mass=30, inclination=np.pi/3)
    >>> print(t_geo, h_geo)

    """
    # approximant
    approx = lalsim.GetApproximantFromString('EccentricTD')

    # fiducial distance
    dist = 1000
    
    # process all params for lal
    deltaT, f_ref, f_low = obtain_time_fequency_inputs(total_mass, deltaTOverM, Momega0OverM)
    m1, m2, s1, s2 = process_intrinsic_params(total_mass, mass_ratio)
    dist_SI, iota, phi_c = process_extrinsic_params(dist, inclination, ref_phase)
    longAscNodes, eccentricity, meanPerAno = process_eccentric_params(eccentricity)
    
    # lal waveform dictionary
    WFdict = create_lal_dict()
    
    if time_wfgeneration:
        t1= time.time()
        
    hplus, hcross = generate_SimInspiralChooseTDWaveform(m1, m2, s1, s2, dist_SI, iota, phi_c, longAscNodes, 
                                         eccentricity, meanPerAno, deltaT, f_ref, f_low, WFdict, approx)
    
    
    if time_wfgeneration:
        t2=time.time()
        print('Time taken: %.5f'%(t2-t1))

    # Data as numpy array
    t, h = convert_lalsim_wf_to_numpyarray(hplus, hcross, deltaT)

    # geometric units; remove iota/phase contribution
    MT = total_mass * lal.MTSUN_SI
    t_geo, h_geo = convert_EccentricTD_physical_wf_to_geometric(t, h, MT, dist, iota, phi_c)
    
    return t_geo, h_geo


def generate_IMRPhenomTHM(mass_ratio, total_mass=10.0, s1z=0.0, s2z=0.0, ref_phase=0.0, Momega0OverM=0.02, 
                         deltaTOverM=0.5, time_wfgeneration=True):
    """
    Generates a precessing inspiral-merger-ringdown waveform using the IMRPhenomTHM model.

    Parameters:
    ----------
    mass_ratio : float
        Mass ratio of the binary system (m1/m2).
    total_mass : float, optional
        Total mass of the binary system in solar masses. Default is 10.0.
    s1z : float, optional
        Spin component along the z-axis for the primary object.
    s2z : float, optional
        Spin component along the z-axis for the secondary object.
    ref_phase : float, optional
        Reference phase at the initial time, in radians.
    Momega0OverM : float, optional
        Initial orbital frequency in geometric units. Default is 0.02.
    deltaTOverM : float, optional
        Time step in geometric units. Default is 0.5.
    time_wfgeneration : bool, optional
        If True, prints the time taken to generate the waveform. Default is True.

    Returns:
    -------
    t_geo : ndarray
        Time array in geometric units.
    h_geo : dict
        Dictionary of waveform modes in geometric units.
    """

    # fiducial distance
    dist = 1000
    
    # process all params for lal
    deltaT, f_ref, f_low = obtain_time_fequency_inputs(total_mass, deltaTOverM, Momega0OverM)
    m1, m2, s1, s2 = process_intrinsic_params(total_mass, mass_ratio, s1z, s2z)
    inclination = 0.0
    dist_SI, iota, phi_c = process_extrinsic_params(dist, inclination, ref_phase)
    
    # lal waveform dictionary
    WFdict = create_lal_dict()
    
    if time_wfgeneration:
        t1= time.time()

    sphtseries = lalsim.SimIMRPhenomTHM_Modes(m1, m2,\
                                             s1[2], s2[2],\
                                             dist_SI, deltaT,\
                                             f_low, f_ref,\
                                             phi_c, WFdict)
    
        
    if time_wfgeneration:
        t2=time.time()
        print('Time taken: %.5f'%(t2-t1))

    MT = total_mass * lal.MTSUN_SI

    # Data as dictionary
    h_dict = process_IMRPheomTHM_output(sphtseries)

    # geometric units; remove iota/phase contribution
    MT = total_mass * lal.MTSUN_SI
    t_geo, h_geo = convert_physical_wf_to_geometric(h_dict, MT, dist, deltaTOverM)
    
    return t_geo, h_geo


def process_IMRPheomTHM_output(sphtseries):
    """
    Processes the raw LALSuite output from IMRPhenomTHM and organizes waveform modes in a dictionary.

    Parameters:
    ----------
    sphtseries : LALSeries
        LALSeries object containing the waveform modes.

    Returns:
    -------
    h_dict : dict
        Dictionary with mode labels (e.g., 'h_l2m2') as keys and complex waveform arrays as values.
    """
    h_dict = {}
    type_struct = type(sphtseries)
    while type(sphtseries) is type_struct:
        l = sphtseries.l
        m = sphtseries.m
        hlm = sphtseries.mode.data.data
        h_dict['h_l%dm%d'%(l,m)] = hlm
        sphtseries = sphtseries.next
    return h_dict


def convert_physical_wf_to_geometric(h_dict, MT, dist, deltaTOverM):
    """
    Converts waveform data from physical to geometric units.

    Parameters:
    ----------
    h_dict : dict
        Dictionary containing waveform data in physical units.
    MT : float
        Total mass in geometric units.
    dist : float
        Distance in megaparsecs.
    deltaTOverM : float
        Time step in geometric units.

    Returns:
    -------
    t_geo : ndarray
        Time array in geometric units.
    h_geo : dict
        Dictionary of waveform modes in geometric units.
    """
    h_geo = {}
    for mode in h_dict.keys():
        h_geo[mode] = h_dict[mode] * (dist * 1e6 * lal.PC_SI) / MT / lal.C_SI
    t_geo = deltaTOverM *np.arange(len(h_dict['h_l2m2']))
    return t_geo, h_geo


def generate_SimInspiralChooseTDWaveform(m1, m2, s1, s2, dist_SI, iota, phi_c, longAscNodes, 
                                         eccentricity, meanPerAno, deltaT, f_ref, f_low, WFdict, 
                                         approx):
    """
    Generates a time-domain waveform using the SimInspiralChooseTDWaveform function from LALSuite.

    Parameters:
    ----------
    m1, m2 : float
        Masses of the two binary components, in SI units (kg).
    s1, s2 : array-like
        Spin vectors for the two binary components.
    dist_SI : float
        Distance to the source in meters.
    iota : float
        Inclination angle between the line-of-sight and binary's angular momentum.
    phi_c : float
        Reference phase at the start of waveform generation.
    longAscNodes : float
        Longitude of ascending nodes (for eccentricity).
    eccentricity : float
        Orbital eccentricity of the binary.
    meanPerAno : float
        Mean anomaly at the start of waveform generation.
    deltaT : float
        Time step for waveform sampling.
    f_ref : float
        Reference frequency for waveform generation.
    f_low : float
        Lower frequency cutoff.
    WFdict : LALDict
        Dictionary of waveform parameters.
    approx : LALApproximation
        Waveform approximation model.

    Returns:
    -------
    hplus : LALSeries
        Plus polarization of the gravitational waveform.
    hcross : LALSeries
        Cross polarization of the gravitational waveform.
    """
    hplus, hcross = lalsim.SimInspiralChooseTDWaveform(m1, m2,\
                                                              s1[0], s1[1], s1[2],\
                                                              s2[0], s2[1], s2[2],\
                                                              dist_SI, iota, phi_c,\
                                                              longAscNodes, eccentricity, meanPerAno,
                                                              deltaT, f_low, f_ref,\
                                                              WFdict, approx)
    return hplus, hcross


def obtain_time_fequency_inputs(total_mass, deltaTOverM, Momega0OverM):
    """
    Converts inputs into time and frequency settings for LALSuite.

    Parameters:
    ----------
    total_mass : float
        Total mass of the binary system in solar masses.
    deltaTOverM : float
        Time step in geometric units.
    Momega0OverM : float
        Initial orbital frequency in geometric units.

    Returns:
    -------
    deltaT : float
        Time step for waveform sampling.
    f_ref : float
        Reference frequency for waveform generation.
    f_low : float
        Lower frequency cutoff.
    """
    # time step
    deltaT = deltaTOverM * lal.MTSUN_SI
    # reference frequency and minimum frequency
    MT = total_mass * lal.MTSUN_SI
    f_low = Momega0OverM/np.pi/MT
    f_ref = f_low
    return deltaT, f_ref, f_low
    
    
def process_intrinsic_params(total_mass, mass_ratio, s1z=0.0, s2z=0.0):
    """
    Processes intrinsic binary parameters for waveform generation.

    Parameters:
    ----------
    total_mass : float
        Total mass of the binary system in solar masses.
    mass_ratio : float
        Mass ratio of the binary system (m1/m2).
    s1z : float, optional
        Spin component along the z-axis for the primary object. Default is 0.0.
    s2z : float, optional
        Spin component along the z-axis for the secondary object. Default is 0.0.

    Returns:
    -------
    m1 : float
        Mass of the primary component in SI units (kg).
    m2 : float
        Mass of the secondary component in SI units (kg).
    s1 : list
        Spin vector for the primary object.
    s2 : list
        Spin vector for the secondary object.
    """
    # component masses of the binary
    m1 = total_mass * lal.MSUN_SI * mass_ratio / (1. + mass_ratio)
    m2 = total_mass * lal.MSUN_SI / (1. + mass_ratio)
    # spins
    s1 = [0, 0, s1z]
    s2 = [0, 0, s2z]
    return m1, m2, s1, s2


def process_extrinsic_params(dist, inclination, ref_phase):
    """
    Processes extrinsic parameters for waveform generation.

    Parameters:
    ----------
    dist : float
        Distance to the source in megaparsecs.
    inclination : float
        Inclination angle between the line-of-sight and binary's angular momentum.
    ref_phase : float
        Reference phase at the start of waveform generation.

    Returns:
    -------
    dist_SI : float
        Distance in meters.
    iota : float
        Inclination angle.
    phi_c : float
        Reference phase.
    """
    # Distance in Mpc
    dist_SI = dist * 1e6 * lal.PC_SI
    # Inclination and overall phase
    iota = inclination 
    phi_c = ref_phase
    return dist_SI, iota, phi_c


def process_eccentric_params(eccentricity):
    """
    Processes parameters related to the binary's orbital eccentricity.

    Parameters:
    ----------
    eccentricity : float
        Orbital eccentricity of the binary.

    Returns:
    -------
    longAscNodes : float
        Longitude of ascending nodes.
    eccentricity : float
        Orbital eccentricity.
    meanPerAno : float
        Mean anomaly at the start of waveform generation.
    """
    longAscNodes = 0
    eccentricity = eccentricity
    meanPerAno = 0
    return longAscNodes, eccentricity, meanPerAno


def create_lal_dict():
    """
    Initializes a dictionary for configuring waveform parameters in LALSuite.

    Returns:
    -------
    WFdict : LALDict
        Dictionary with LALSuite waveform configuration parameters.
    """
    WFdict = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertFrameAxis(WFdict, 2)
    lalsim.SimInspiralWaveformParamsInsertPNSpinOrder(WFdict, -1)
    lalsim.SimInspiralWaveformParamsInsertPNTidalOrder(WFdict, -1)
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(WFdict, -1)
    lalsim.SimInspiralWaveformParamsInsertPNAmplitudeOrder(WFdict, -1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(WFdict, 0.0)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(WFdict, 0.0)
    return WFdict
    
    
def convert_lalsim_wf_to_numpyarray(hplus, hcross, deltaT):
    """
    Converts LALSuite waveform data to numpy arrays for easier processing.

    Parameters:
    ----------
    hplus : LALSeries
        Plus polarization of the gravitational waveform.
    hcross : LALSeries
        Cross polarization of the gravitational waveform.
    deltaT : float
        Time step for waveform sampling.

    Returns:
    -------
    t : ndarray
        Time array.
    h : ndarray
        Complex waveform data array (hplus - i * hcross).
    """
    hplus_timeseries = hplus.data.data 
    hcross_timeseries = hcross.data.data
    h = hplus_timeseries - 1j*hcross_timeseries 
    t = np.array(range(len(hcross_timeseries)))*deltaT
    return t, h


def convert_EccentricTD_physical_wf_to_geometric(t, h, MT, dist, iota, phi_c):
    """
    Converts physical units of the waveform to geometric units.

    Parameters:
    ----------
    t : ndarray
    Time array in physical units.
    h : ndarray
        Complex waveform data in physical units.
    MT : float
        Total mass of the binary system in geometric units.
    dist : float
        Distance to the source in megaparsecs.
    iota : float
        Inclination angle between the line-of-sight and binary's angular momentum.
    phi_c : float
        Reference phase at the start of waveform generation.

    Returns:
    -------
    t_geo : ndarray
        Time array in geometric units.
    h_geo : ndarray
        Complex waveform data in geometric units.
    """
    t_geo = t / MT
    Ylm = gwtools.sYlm(-2, 2, 2, iota, phi_c)
    h_geo = (h * (dist * 1e6 * lal.PC_SI) / MT / lal.C_SI)/Ylm
    return t_geo, h_geo