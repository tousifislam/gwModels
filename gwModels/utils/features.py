#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: features.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-02-2024
#    LAST MODIFIED: Tue Feb  6 17:58:52 2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools

def get_amplitude(h):
    """
    Computes phase of a given gravitational wave time-series
        t: time array
        h: waveform mode array
    Returns:
        A: phase A=abs(h)
    """
    return abs(h)
    
def get_phase(h):
    """
    Computes phase of a given gravitational wave time-series
        t: time array
        h: waveform mode array
    Returns:
        phi: phase phi=arg(h)
    """
    return gwtools.phase(h)

def get_frequency(t, h):
    """
    Computes orbital frequency of a given gravitational wave time-series
        t: time array
        h: waveform mode array
    Returns:
        omega: frequency omega=dphi/dt
    """
    return abs(np.gradient(get_phase(h), t))

def get_gw_frequency(t, h):
    """
    Computes orbital frequency of a given gravitational wave time-series
        t: time array
        h: waveform mode array
    Returns:
        f: frequency f=omega/np.pi
    """
    return get_frequency(t, h)/np.pi
