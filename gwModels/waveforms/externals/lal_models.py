#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#     FILE: lal_models.py
#
#     AUTHOR: Tousif Islam
#     CREATED: 07-02-2024
#     LAST MODIFIED: 05-28-2026
#     REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import logging
import time

import numpy as np
import gwtools

try:
    import lal
    import lalsimulation as lalsim
except ImportError:
    lal = None
    lalsim = None

logger = logging.getLogger(__name__)


def generate_EccentricTD(mass_ratio, total_mass=10.0, inclination=0.0,
                         ref_phase=0.0, eccentricity=0.01, Momega0OverM=0.02,
                         deltaTOverM=0.5, time_wfgeneration=True):
    """
    Generates an eccentric time-domain gravitational waveform using the LAL EccentricTD approximant.

    Parameters
    ----------
    mass_ratio : float
        The mass ratio of the binary system, defined as m1/m2.
    total_mass : float, optional
        The total mass of the binary system in solar masses. Default is 10.0.
    inclination : float, optional
        The inclination angle in radians. Default is 0.0 (face-on).
    ref_phase : float, optional
        The reference phase at the initial frequency, in radians. Default is 0.0.
    eccentricity : float, optional
        The eccentricity at the reference frequency. Default is 0.01.
    Momega0OverM : float, optional
        The dimensionless initial orbital frequency in geometric units. Default is 0.02.
    deltaTOverM : float, optional
        The time-step in geometric units. Default is 0.5.
    time_wfgeneration : bool, optional
        If True, logs the time taken to generate the waveform. Default is True.

    Returns
    -------
    t_geo : numpy.ndarray
        Time array in geometric units.
    h_geo : numpy.ndarray
        Complex waveform in geometric units.
    """
    approx = lalsim.GetApproximantFromString('EccentricTD')

    dist = 1000

    deltaT, f_ref, f_low = obtain_time_fequency_inputs(total_mass, deltaTOverM, Momega0OverM)
    m1, m2, s1, s2 = process_intrinsic_params(total_mass, mass_ratio)
    dist_SI, iota, phi_c = process_extrinsic_params(dist, inclination, ref_phase)
    longAscNodes, eccentricity, meanPerAno = process_eccentric_params(eccentricity)

    WFdict = create_lal_dict()

    if time_wfgeneration:
        t1 = time.time()

    hplus, hcross = generate_SimInspiralChooseTDWaveform(m1, m2, s1, s2, dist_SI, iota, phi_c, longAscNodes,
                                         eccentricity, meanPerAno, deltaT, f_ref, f_low, WFdict, approx)

    if time_wfgeneration:
        t2 = time.time()
        logger.info('EccentricTD waveform generated in %.5f s', t2 - t1)

    t, h = convert_lalsim_wf_to_numpyarray(hplus, hcross, deltaT)

    MT = total_mass * lal.MTSUN_SI
    t_geo, h_geo = convert_EccentricTD_physical_wf_to_geometric(t, h, MT, dist, iota, phi_c)

    return t_geo, h_geo


def generate_IMRPhenomTHM(mass_ratio, total_mass=10.0, s1z=0.0, s2z=0.0, ref_phase=0.0, Momega0OverM=0.02,
                         deltaTOverM=0.5, time_wfgeneration=True):
    """
    Generates an inspiral-merger-ringdown waveform using the IMRPhenomTHM model.

    Parameters
    ----------
    mass_ratio : float
        Mass ratio of the binary system (m1/m2).
    total_mass : float, optional
        Total mass in solar masses. Default is 10.0.
    s1z : float, optional
        Spin component along the z-axis for the primary object. Default is 0.0.
    s2z : float, optional
        Spin component along the z-axis for the secondary object. Default is 0.0.
    ref_phase : float, optional
        Reference phase in radians. Default is 0.0.
    Momega0OverM : float, optional
        Initial orbital frequency in geometric units. Default is 0.02.
    deltaTOverM : float, optional
        Time step in geometric units. Default is 0.5.
    time_wfgeneration : bool, optional
        If True, logs the time taken to generate the waveform. Default is True.

    Returns
    -------
    t_geo : ndarray
        Time array in geometric units.
    h_geo : dict
        Dictionary of waveform modes in geometric units.
    """
    dist = 1000

    deltaT, f_ref, f_low = obtain_time_fequency_inputs(total_mass, deltaTOverM, Momega0OverM)
    m1, m2, s1, s2 = process_intrinsic_params(total_mass, mass_ratio, s1z, s2z)
    inclination = 0.0
    dist_SI, iota, phi_c = process_extrinsic_params(dist, inclination, ref_phase)

    WFdict = create_lal_dict()

    if time_wfgeneration:
        t1 = time.time()

    sphtseries = lalsim.SimIMRPhenomTHM_Modes(m1, m2,
                                             s1[2], s2[2],
                                             dist_SI, deltaT,
                                             f_low, f_ref,
                                             phi_c, WFdict)

    if time_wfgeneration:
        t2 = time.time()
        logger.info('IMRPhenomTHM waveform generated in %.5f s', t2 - t1)

    h_dict = process_IMRPheomTHM_output(sphtseries)

    MT = total_mass * lal.MTSUN_SI
    t_geo, h_geo = convert_physical_wf_to_geometric(h_dict, MT, dist, deltaTOverM)

    return t_geo, h_geo


def process_IMRPheomTHM_output(sphtseries):
    """
    Processes the raw LALSuite output from IMRPhenomTHM into a dictionary.

    Parameters
    ----------
    sphtseries : LALSeries
        LALSeries object containing the waveform modes.

    Returns
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
        h_dict['h_l%dm%d' % (l, m)] = hlm
        sphtseries = sphtseries.next
    return h_dict


def convert_physical_wf_to_geometric(h_dict, MT, dist, deltaTOverM):
    """
    Converts waveform data from physical to geometric units.

    Parameters
    ----------
    h_dict : dict
        Dictionary containing waveform data in physical units.
    MT : float
        Total mass in geometric units.
    dist : float
        Distance in megaparsecs.
    deltaTOverM : float
        Time step in geometric units.

    Returns
    -------
    t_geo : ndarray
        Time array in geometric units.
    h_geo : dict
        Dictionary of waveform modes in geometric units.
    """
    h_geo = {}
    for mode in h_dict.keys():
        h_geo[mode] = h_dict[mode] * (dist * 1e6 * lal.PC_SI) / MT / lal.C_SI
    t_geo = deltaTOverM * np.arange(len(h_dict['h_l2m2']))
    return t_geo, h_geo


def generate_SimInspiralChooseTDWaveform(m1, m2, s1, s2, dist_SI, iota, phi_c, longAscNodes,
                                         eccentricity, meanPerAno, deltaT, f_ref, f_low, WFdict,
                                         approx):
    """
    Generates a time-domain waveform using SimInspiralChooseTDWaveform from LALSuite.

    Parameters
    ----------
    m1, m2 : float
        Masses of the two binary components, in SI units (kg).
    s1, s2 : array-like
        Spin vectors for the two binary components.
    dist_SI : float
        Distance to the source in meters.
    iota : float
        Inclination angle.
    phi_c : float
        Reference phase.
    longAscNodes : float
        Longitude of ascending nodes.
    eccentricity : float
        Orbital eccentricity.
    meanPerAno : float
        Mean anomaly.
    deltaT : float
        Time step for waveform sampling.
    f_ref : float
        Reference frequency.
    f_low : float
        Lower frequency cutoff.
    WFdict : LALDict
        Dictionary of waveform parameters.
    approx : LALApproximation
        Waveform approximation model.

    Returns
    -------
    hplus, hcross : LALSeries
        Plus and cross polarizations of the gravitational waveform.
    """
    hplus, hcross = lalsim.SimInspiralChooseTDWaveform(m1, m2,
                                                       s1[0], s1[1], s1[2],
                                                       s2[0], s2[1], s2[2],
                                                       dist_SI, iota, phi_c,
                                                       longAscNodes, eccentricity, meanPerAno,
                                                       deltaT, f_low, f_ref,
                                                       WFdict, approx)
    return hplus, hcross


def obtain_time_fequency_inputs(total_mass, deltaTOverM, Momega0OverM):
    """
    Converts inputs into time and frequency settings for LALSuite.

    Parameters
    ----------
    total_mass : float
        Total mass in solar masses.
    deltaTOverM : float
        Time step in geometric units.
    Momega0OverM : float
        Initial orbital frequency in geometric units.

    Returns
    -------
    deltaT : float
        Time step for waveform sampling.
    f_ref : float
        Reference frequency.
    f_low : float
        Lower frequency cutoff.
    """
    deltaT = deltaTOverM * lal.MTSUN_SI
    MT = total_mass * lal.MTSUN_SI
    f_low = Momega0OverM / np.pi / MT
    f_ref = f_low
    return deltaT, f_ref, f_low


def process_intrinsic_params(total_mass, mass_ratio, s1z=0.0, s2z=0.0):
    """
    Processes intrinsic binary parameters for waveform generation.

    Parameters
    ----------
    total_mass : float
        Total mass in solar masses.
    mass_ratio : float
        Mass ratio (m1/m2).
    s1z : float, optional
        Spin component along the z-axis for the primary. Default is 0.0.
    s2z : float, optional
        Spin component along the z-axis for the secondary. Default is 0.0.

    Returns
    -------
    m1 : float
        Mass of the primary in SI units (kg).
    m2 : float
        Mass of the secondary in SI units (kg).
    s1 : list
        Spin vector for the primary.
    s2 : list
        Spin vector for the secondary.
    """
    m1 = total_mass * lal.MSUN_SI * mass_ratio / (1. + mass_ratio)
    m2 = total_mass * lal.MSUN_SI / (1. + mass_ratio)
    s1 = [0, 0, s1z]
    s2 = [0, 0, s2z]
    return m1, m2, s1, s2


def process_extrinsic_params(dist, inclination, ref_phase):
    """
    Processes extrinsic parameters for waveform generation.

    Parameters
    ----------
    dist : float
        Distance in megaparsecs.
    inclination : float
        Inclination angle.
    ref_phase : float
        Reference phase.

    Returns
    -------
    dist_SI : float
        Distance in meters.
    iota : float
        Inclination angle.
    phi_c : float
        Reference phase.
    """
    dist_SI = dist * 1e6 * lal.PC_SI
    iota = inclination
    phi_c = ref_phase
    return dist_SI, iota, phi_c


def process_eccentric_params(eccentricity):
    """
    Processes parameters related to the binary's orbital eccentricity.

    Parameters
    ----------
    eccentricity : float
        Orbital eccentricity.

    Returns
    -------
    longAscNodes : float
        Longitude of ascending nodes.
    eccentricity : float
        Orbital eccentricity.
    meanPerAno : float
        Mean anomaly.
    """
    longAscNodes = 0
    eccentricity = eccentricity
    meanPerAno = 0
    return longAscNodes, eccentricity, meanPerAno


def create_lal_dict():
    """
    Initializes a dictionary for configuring waveform parameters in LALSuite.

    Returns
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
    Converts LALSuite waveform data to numpy arrays.

    Parameters
    ----------
    hplus : LALSeries
        Plus polarization.
    hcross : LALSeries
        Cross polarization.
    deltaT : float
        Time step.

    Returns
    -------
    t : ndarray
        Time array.
    h : ndarray
        Complex waveform data (hplus - i * hcross).
    """
    hplus_timeseries = hplus.data.data
    hcross_timeseries = hcross.data.data
    h = hplus_timeseries - 1j * hcross_timeseries
    t = np.array(range(len(hcross_timeseries))) * deltaT
    return t, h


def convert_EccentricTD_physical_wf_to_geometric(t, h, MT, dist, iota, phi_c):
    """
    Converts physical units of the waveform to geometric units.

    Parameters
    ----------
    t : ndarray
        Time array in physical units.
    h : ndarray
        Complex waveform data in physical units.
    MT : float
        Total mass in geometric units.
    dist : float
        Distance in megaparsecs.
    iota : float
        Inclination angle.
    phi_c : float
        Reference phase.

    Returns
    -------
    t_geo : ndarray
        Time array in geometric units.
    h_geo : ndarray
        Complex waveform in geometric units.
    """
    t_geo = t / MT
    Ylm = gwtools.sYlm(-2, 2, 2, iota, phi_c)
    h_geo = (h * (dist * 1e6 * lal.PC_SI) / MT / lal.C_SI) / Ylm
    return t_geo, h_geo
