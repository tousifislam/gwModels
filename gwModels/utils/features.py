#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: features.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-02-2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools


def get_amplitude(h):
    """
    Computes amplitude of a given gravitational wave time-series.

    Parameters:
        h (np.ndarray): Waveform mode array.

    Returns:
        np.ndarray: Amplitude A = abs(h).
    """
    return abs(h)


def get_phase(h):
    """
    Computes phase of a given gravitational wave time-series.

    Parameters:
        h (np.ndarray): Waveform mode array.

    Returns:
        np.ndarray: Phase phi = arg(h).
    """
    return gwtools.phase(h)


def get_frequency(t, h):
    """
    Computes orbital frequency of a given gravitational wave time-series.

    Parameters:
        t (np.ndarray): Time array.
        h (np.ndarray): Waveform mode array.

    Returns:
        np.ndarray: Frequency omega = dphi/dt.
    """
    return abs(np.gradient(get_phase(h), t))


def get_gw_frequency(t, h):
    """
    Computes gravitational wave frequency of a given time-series.

    Parameters:
        t (np.ndarray): Time array.
        h (np.ndarray): Waveform mode array.

    Returns:
        np.ndarray: Frequency f = omega/pi.
    """
    return get_frequency(t, h) / np.pi
