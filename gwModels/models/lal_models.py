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
    wrapper to generate EccentricTD waveform from lal
    mass_ratio: m1/m2
    total_mass: total mass in Msun
    inclination: angle between line-of-sight and binary angular momentum 
    ref_phase: initial reference phase
    eccentricity: eccentricity value
    Momega0OverM: initial freqquency in geometric unit 
    deltaTOverM: time-step in geometric unit
    time_wfgeneration: if True, it prints the time taken to generate the waveform
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
    wrapper to generate EccentricTD waveform from lal
    mass_ratio: m1/m2
    total_mass: total mass in Msun
    inclination: angle between line-of-sight and binary angular momentum 
    ref_phase: initial reference phase
    Momega0OverM: initial freqquency in geometric unit 
    deltaTOverM: time-step in geometric unit
    time_wfgeneration: if True, it prints the time taken to generate the waveform
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
    process raw lal IMRPhenomTHM output and provide dictionary of modes
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
    convert physical waveform to geometric unit
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
    generate waveform using SimInspiralChooseTDWaveform module from lalsimulation
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
    process initial inputs for lalsimulation
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
    process initial mass, spin inputs for lalsimulation
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
    process initial distance, phase inputs for lalsimulation
    """
    # Distance in Mpc
    dist_SI = dist * 1e6 * lal.PC_SI
    # Inclination and overall phase
    iota = inclination 
    phi_c = ref_phase
    return dist_SI, iota, phi_c


def process_eccentric_params(eccentricity):
    """
    process initial eccentricity inputs for lalsimulation
    """
    longAscNodes = 0
    eccentricity = eccentricity
    meanPerAno = 0
    return longAscNodes, eccentricity, meanPerAno


def create_lal_dict():
    """
    creates a dictionary needed to generate lal waveforms
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
    converts lal output to numpy array for routine use
    """
    hplus_timeseries = hplus.data.data 
    hcross_timeseries = hcross.data.data
    h = hplus_timeseries - 1j*hcross_timeseries 
    t = np.array(range(len(hcross_timeseries)))*deltaT
    return t, h


def convert_EccentricTD_physical_wf_to_geometric(t, h, MT, dist, iota, phi_c):
    """
    convert physical waveform to geometric unit
    """
    t_geo = t / MT
    Ylm = gwtools.sYlm(-2, 2, 2, iota, phi_c)
    h_geo = (h * (dist * 1e6 * lal.PC_SI) / MT / lal.C_SI)/Ylm
    return t_geo, h_geo