#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: compute_local_peaks.py
#    Computes peaks of an oscillatory function
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-08-2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from scipy.signal import find_peaks


class PeakFinderCrude:
    """
    Find the peaks of a signal using a chunked min/max approach.

    Parameters:
        time (np.ndarray): Time array.
        signal (np.ndarray): Data signal from which to extract envelopes.
        dmin (int): Chunk size for minima. Default: 1.
        dmax (int): Chunk size for maxima. Default: 1.
    """
    def __init__(self, time, signal, dmin=1, dmax=1):
        self.time = time
        self.signal = signal
        self.dmin = dmin
        self.dmax = dmax
        self.min_indx = self._find_local_minimas()
        self.max_indx = self._find_local_maximas()

    def _find_local_minimas(self):
        """Find all local minima."""
        lmin = (np.diff(np.sign(np.diff(self.signal))) > 0).nonzero()[0] + 1
        lmin = lmin[[i + np.argmin(self.signal[lmin[i:i + self.dmin]]) for i in range(0, len(lmin), self.dmin)]]
        return lmin

    def _find_local_maximas(self):
        """Find all local maxima."""
        lmax = (np.diff(np.sign(np.diff(self.signal))) < 0).nonzero()[0] + 1
        lmax = lmax[[i + np.argmax(self.signal[lmax[i:i + self.dmax]]) for i in range(0, len(lmax), self.dmax)]]
        return lmax

    def plot_peaks(self):
        """Plot the signal with detected peaks."""
        import matplotlib.pyplot as plt
        plt.plot(self.time, self.signal, label='Signal')
        plt.plot(self.time[self.max_indx], self.signal[self.max_indx], '.', label='Maximas')
        plt.plot(self.time[self.min_indx], self.signal[self.min_indx], '*', label='Minimas')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        plt.show()


class PeakFinderScipy:
    """
    Find the peaks of a signal using scipy.signal.find_peaks.

    Parameters:
        time (np.ndarray): Time array.
        signal (np.ndarray): Data signal from which to extract envelopes.
        distance_btw_peaks (int): Minimum distance between peaks in samples. Default: 100.
    """
    def __init__(self, time, signal, distance_btw_peaks=100):
        self.time = time
        self.signal = signal
        self.distance_btw_peaks = distance_btw_peaks
        self.max_indx, self.min_indx = self._find_peaks()

    def _find_peaks(self):
        """Find local maxima and minima using only the inspiral part (t <= -10)."""
        indx = np.where(self.time <= -10.0)
        max_peaks_indx = find_peaks(self.signal[indx], distance=self.distance_btw_peaks)[0]
        min_peaks_indx = find_peaks(-self.signal[indx], distance=self.distance_btw_peaks)[0]
        return max_peaks_indx, min_peaks_indx

    def plot_peaks(self):
        """Plot the signal with detected peaks."""
        import matplotlib.pyplot as plt
        plt.plot(self.time, self.signal, label='Signal')
        plt.plot(self.time[self.max_indx], self.signal[self.max_indx], '.', label='Maximas')
        plt.plot(self.time[self.min_indx], self.signal[self.min_indx], '*', label='Minimas')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        plt.show()
