#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: compute_ecc_omega_ecc_gw.py
#    Computes eccentricity given eccentric and circular waveforms or
#    given eccentric modulations
#
#       AUTHOR: Tousif Islam
#       CREATED: 08-08-2024
#       REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import timeit
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import gwtools

from scipy.optimize import curve_fit
from scipy import interpolate

from ..utils import *

class ComputeEccentricityFromOmega:
    """
    Class to compute smooth eccentricity using eccentric modulation parameter
    """
    def __init__(self, time_xi, xi_lower, xi_upper, gwnrhme_obj, ecc_prefactor, t_ref):
        
        self.time_xi = time_xi
        self.gwnrhme_obj = gwnrhme_obj
        self.ecc_prefactor = ecc_prefactor
        self.xi_lower = xi_lower
        self.xi_upper = xi_upper
        self.t_ref = t_ref
        
        # compute eccentric and circular omega for the 22 mode
        self.compute_omega_22_cir()
        self.compute_omega_22_ecc()
        
        # compute omega_22 envelops
        self.compute_omega_22_periastron()
        self.compute_omega_22_apastron()
        
        # compute ecc_omega_22
        self.compute_ecc_omega_22()
        # compute ecc_gw
        self.compute_psi()
        self.compute_ecc_gw()
        
        # build interpolation
        self.ecc_omega_22_interp = self._get_ecc_omega_22_interp()
        self.ecc_gw_interp = self._get_ecc_gw_interp()
        
        # reference eccentricities
        self._compute_ecc_omega_22_at_tref()
        self._compute_ecc_gw_at_tref()
        
    def compute_omega_22_ecc(self):
        """
        Compute 22 mode frequency of the eccentric waveform
        """
        omega_22_ecc = abs(get_frequency(self.gwnrhme_obj.ecc_wfobj.t_transform, self.gwnrhme_obj.ecc_wfobj.h_transform['h_l2m2']))
        self.omega_22_ecc = gwtools.interpolate_h(self.gwnrhme_obj.ecc_wfobj.t_transform, omega_22_ecc, self.time_xi)
        
    def compute_omega_22_cir(self):
        """
        Compute 22 mode frequency of the circular waveform
        """
        omega_22_cir = abs(get_frequency(self.gwnrhme_obj.cir_wfobj.t_transform, self.gwnrhme_obj.cir_wfobj.h_transform['h_l2m2']))
        self.omega_22_cir = gwtools.interpolate_h(self.gwnrhme_obj.cir_wfobj.t_transform, omega_22_cir, self.time_xi)
        
    def compute_omega_22_periastron(self):
        """
        Compute omega_22 at periastron from xi parameter
        """
        self.omega_22_p = self.omega_22_cir * (1 + (1/self.ecc_prefactor) * self.xi_upper)
        
    def compute_omega_22_apastron(self):
        """
        Compute omega_22 at apastron from xi parameter
        """
        self.omega_22_a = self.omega_22_cir * (1 - (1/self.ecc_prefactor) * self.xi_lower)
        
    def compute_ecc_omega_22(self):
        """
        Compute ecc_omega_22 eccentricity 
        Based on Eq(5) of https://arxiv.org/pdf/2209.03390
        """
        numerator = np.sqrt(self.omega_22_p) - np.sqrt(self.omega_22_a)
        denominator = np.sqrt(self.omega_22_p) + np.sqrt(self.omega_22_a)
        self.ecc_omega22 = numerator/denominator
        
    def compute_psi(self):
        """
        Compute psi transformation to have correct Newtonian limit at the leading order
        Based on Eq(6b) of https://arxiv.org/pdf/2209.03390
        """
        self.psi = np.arctan2((1-self.ecc_omega22**2), (2*self.ecc_omega22))
        
    def compute_ecc_gw(self):
        """
        Compute e_gw 
        Based on Eq(6a) of https://arxiv.org/pdf/2209.03390
        """
        self.ecc_gw = np.cos(self.psi/3) - np.sqrt(3)*(np.sin(self.psi/3))
        
    def _get_ecc_omega_22_interp(self):
        """
        obtain 1d interpolation function between time and ecc_omega_22
        """
        interp = interpolate.interp1d(self.time_xi, self.ecc_omega22)
        return interp
    
    def _get_ecc_gw_interp(self):
        """
        obtain 1d interpolation function between time and ecc_gw
        """
        interp = interpolate.interp1d(self.time_xi, self.ecc_gw)
        return interp
    
    def _compute_ecc_omega_22_at_tref(self):
        """
        compute eccentricity ecc_omega_22 at tref
        """
        self.ecc_omega_22_ref = self.ecc_omega_22_interp(self.t_ref)
        print('... gwModels ecc_omega_22 at t_ref=%.2f : %.5f'%(self.t_ref, self.ecc_omega_22_ref))
        
    def _compute_ecc_gw_at_tref(self):
        """
        compute eccentricity ecc_gw at tref
        """
        self.ecc_gw_ref = self.ecc_gw_interp(self.t_ref)
        print('... gwModels ecc_gw at t_ref=%.2f : %.5f'%(self.t_ref, self.ecc_gw_ref))
        
    def plot_eccentricities(self, figsize=(8,5)):
        """
        plot eccentricity evolutions
        """
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.ecc_omega22, c='C1', ls='-', label='$e_{\\omega_{22}}$')
        plt.plot(self.time_xi, self.ecc_gw, c='C2', ls='-', label='$e_{\\rm gw}$')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Eccentricities', fontsize=18)
        plt.ylim(ymin=0)
        plt.legend(fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show()
        
    def plot_omega_22_with_peaks_and_fits(self, figsize=(8,5)):
        """
        plot eccentric omega 22 along with upper and lower envelop fits
        """
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.omega_22_ecc, color='C0', markersize=10, alpha=0.7, label='$\\omega_{22}^{\\rm ecc}$')
        plt.plot(self.time_xi, self.omega_22_cir, color='C4', markersize=10, alpha=0.7, label='$\\omega_{22}^{\\rm cir}$')
        plt.plot(self.time_xi, self.omega_22_p, color='C1', ls='--', markersize=10, alpha=0.7, label='$\\omega_{22}^{\\rm p}$')
        plt.plot(self.time_xi, self.omega_22_a, color='C2', ls='-.', markersize=10, alpha=0.7, label='$\\omega_{22}^{\\rm a}$')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('${\\omega_{22}}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=14)
        plt.ylim(0, 0.022*np.pi*2)
        plt.show()