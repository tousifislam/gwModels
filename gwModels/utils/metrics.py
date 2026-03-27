#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: metrics.py
#    Waveform comparison metrics
#
#    AUTHOR: Tousif Islam
#    CREATED: 03-27-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


def mathcalE_error(h1, h2):
    """
    Computes the time-domain error between two waveforms.

    Calculates the error according to Equation 21 of
    https://arxiv.org/pdf/1701.00550.pdf by normalizing the difference
    between the two waveforms.

    Parameters:
        h1 (np.ndarray): Reference waveform in the time domain.
        h2 (np.ndarray): Comparison waveform in the time domain.

    Returns:
        np.ndarray: Normalized error for each time sample.
    """
    n1Sqr = np.sum(abs(h1) ** 2)
    n2Sqr = np.sum(abs(h2) ** 2)
    sdot = np.real(np.sum(h1 * np.conj(h2)))
    normed_errs = ((n1Sqr + n2Sqr) - 2 * sdot) / (2 * n1Sqr)
    return normed_errs


def simple_mismatch(h1, h2):
    """
    Computes a simple mismatch between two waveforms.

    The mismatch is defined as 1 minus the normalized inner product:
        MM = 1 - Re(<h1, h2>) / (||h1|| * ||h2||)

    Parameters:
        h1 (np.ndarray): First waveform in the time domain.
        h2 (np.ndarray): Second waveform in the time domain.

    Returns:
        float: Mismatch value. Returns NaN if either waveform has zero norm.
    """
    inner = np.real(np.sum(h1 * np.conj(h2)))
    norm1 = np.linalg.norm(h1)
    norm2 = np.linalg.norm(h2)
    if norm1 > 0 and norm2 > 0:
        return 1.0 - inner / (norm1 * norm2)
    return np.nan
