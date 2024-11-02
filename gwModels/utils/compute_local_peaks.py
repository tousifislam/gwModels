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
import warnings
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline

class PeakFinderCrude:
    """
    Class to find the peaks of a signal using a crude way
    """
    def __init__(self, time, signal, dmin=1, dmax=1):
        """
        Input :
        time: 1d-array, time
        signal: 1d-array, data signal from which to extract high and low envelopes
        dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
        
        Output :
        lmin,lmax : high/low envelope idx of input signal s
        """
        self.time = time
        self.signal = signal
        self.dmin = dmin
        self.dmax = dmax
        self.min_indx = self._find_local_minimas()
        self.max_indx = self._find_local_maximas()
        
    def _find_local_minimas(self):
        """
        find all the local minimas
        """
        # locals min      
        lmin = (np.diff(np.sign(np.diff(self.signal))) > 0).nonzero()[0] + 1 
        # global min of dmin-chunks of locals min 
        lmin = lmin[[i+np.argmin(self.signal[lmin[i:i+self.dmin]]) for i in range(0,len(lmin),self.dmin)]]
        return lmin
    
    def _find_local_maximas(self):
        """
        find all the local maximas
        """
        # locals max
        lmax = (np.diff(np.sign(np.diff(self.signal))) < 0).nonzero()[0] + 1 
        # global max of dmax-chunks of locals max 
        lmax = lmax[[i+np.argmax(self.signal[lmax[i:i+self.dmax]]) for i in range(0,len(lmax),self.dmax)]]
        return lmax
    
    def plot_peaks(self):
        """
        plot peaks
        """
        plt.plot(self.time, self.signal, label='Signal')
        plt.plot(self.time[self.max_indx], self.signal[self.max_indx], '.', label='Maximas')
        plt.plot(self.time[self.min_indx], self.signal[self.min_indx], '*', label='Minimas')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        plt.show()
        
class PeakFinderScipy:
    """
    Class to find the peaks of a signal using scipy
    """
    def __init__(self, time, signal, distance_btw_peaks=100):
        """
        Input :
        time: 1d-array, time
        signal: 1d-array, data signal from which to extract high and low envelopes
        distance_btw_peaks : rough distance between peaks in time series g(t)
                             default: 100
        
        Output :
        lmin,lmax : high/low envelope idx of input signal s
        """
        self.time = time
        self.signal = signal
        self.distance_btw_peaks = distance_btw_peaks
        
        self.max_indx, self.min_indx = self._find_peaks()
        
    def _find_peaks(self):
        """
        find all the local minimas
        """
        # use only inspiral part to get the peaks
        indx = np.where(self.time<=-10.0)
        max_peaks_indx = find_peaks(self.signal[indx], distance=self.distance_btw_peaks)[0]
        min_peaks_indx = find_peaks(-self.signal[indx], distance=self.distance_btw_peaks)[0]
        return max_peaks_indx, min_peaks_indx
    
    def plot_peaks(self):
        """
        plot peaks
        """
        plt.plot(self.time, self.signal, label='Signal')
        plt.plot(self.time[self.max_indx], self.signal[self.max_indx], '.', label='Maximas')
        plt.plot(self.time[self.min_indx], self.signal[self.min_indx], '*', label='Minimas')
        plt.xlabel('Time')
        plt.ylabel('Signal')
        plt.legend()
        plt.show() 