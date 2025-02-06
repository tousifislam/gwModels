#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: compute_ecc_xi.py
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
from ..core import *

class ComputeEccentricityFromModulations():
    """
    Class to compute eccentricity using eccentric modulation parameter
    """
    
    def __init__(self, time_xi, xi, q, t_ref=None, ecc_prefactor=None, 
                 distance_btw_peaks=100, fit_funcs_orders=None,
                 include_zero_zero=False, set_unphysical_xi_to_zero=False,
                 set_unphysical_ecc_to_zero=False,
                 tc=0):
        """
        time_xi: time axis
        xi: common modulation parameter
        q: mass ratio (q>=1)
        t_ref: reference time to compute eccentricity
        ecc_prefactor: pre-factor in eccentricity definition;
                       default is 2/3
        distance_btw_peaks: distance between peaks to be passed to the AvergaeTimeSeriesEstimator;
                            default: 100
        fit_funcs_orders: orders of the upper and lower xi fit functions;
                          available options: ['3PN', '3PN_m1over8', '3PN_m7over8', '3PN_m8over8', '3PN_m1over8_m8over8', 
                                          '3PN_m1over8_m7over8', '3PN_m7over8_m8over8', '3PN_m1over8_m7over8_m8over8']
        include_zero_zero: if True, it will include (t=0,y=0) point to both the list of minimas and maximas;
                           this option sometimes leads to bad fit;
        set_unphysical_xi_to_zero: if True, set all negative xi and NaN values in the fitted xi time-zeries to zero
        set_unphysical_ecc_to_zero: if True, set all negative and NaN values in the fitted eccentricity time-zeries to zero
        tc: time at merger; default is zero
        """
        # basic parameters: time, modulation, mass ratio
        self.time_xi = time_xi
        self.modulations = xi
        self.q = q
        self.tc = tc    
        
        # using a buffer of 50M in time to make sure that interpolation later works fine!
        if t_ref is None:
            self.t_ref = self.time_xi[0] + 10
        else:
            self.t_ref = t_ref
            if self.t_ref <= self.time_xi[0] + 5:
                raise Warning("t_ref is very close to the start of the waveform!")
                raise Warning("It is wise to use a t_ref that is atleast 10M larger than the earliest time in the waveform")
        
        self.distance_btw_peaks = distance_btw_peaks
        
        # fit function orders
        self.fit_funcs_orders = fit_funcs_orders
        if self.fit_funcs_orders is None:
            self.fit_funcs = [self.fit_func_3PN, self.fit_func_3PN]
        else:
            self.fit_funcs = [self.PNorder_to_func_translation(self.fit_funcs_orders[0]), 
                              self.PNorder_to_func_translation(self.fit_funcs_orders[1])]
            
        # whether to include (0,0) points to the list of extremas
        self.include_zero_zero = include_zero_zero
        
        # whether to set negative xi and NaN values in the fitted xi time-zeries to zero
        self.set_unphysical_xi_to_zero = set_unphysical_xi_to_zero
        # whether to set negative and NaN values in the fitted eccentricity time-zeries to zero
        self.set_unphysical_ecc_to_zero = set_unphysical_ecc_to_zero
        
        # obtain eccentricity
        self.ecc_prefactor = ecc_prefactor
        if self.ecc_prefactor is None:
            # Eq (34) of https://arxiv.org/pdf/1702.00872
            self.ecc_prefactor = 2/3
            
        # scale modulations so that it represents eccentricity better
        self.modulations = self.ecc_prefactor * self.modulations
        
        # perform peak fits
        self._get_maximas_minimas()
        self._fit_maximas()
        self._fit_minimas()
        self._get_upper_xi_envelope()
        self._get_lower_xi_envelope()
        self._get_avg_xi_envelope()
        
        # time upto which eccentricity estimations are correct
        self.teccmax = max(max(self.t_minimas), max(self.t_maximas))
        
        self._compute_eccentricity_evolution()
        self._compute_eccentricity_at_tref()
        
        # build interpolation
        self.xi_upper_interp = self._get_xi_upper_interp()
        self.xi_lower_interp = self._get_xi_lower_interp()
        self.ecc_interp = self._get_eccentricity_interp()
        
        # fit errors
        self.xi_upper_fit_error = self._compute_xi_upper_fit_errors()
        self.xi_lower_fit_error = self._compute_xi_lower_fit_errors()
              
    def _get_maximas_minimas(self):
        """
        obtain all maximas and minimas in a xi time series
        """
        # create the average time series estimator
        obj = PeakFinderScipy(time=self.time_xi,
                              signal=self.modulations,
                              distance_btw_peaks=self.distance_btw_peaks)

        # obtain the values of the minimum peaks, convert the values to positive side 
        # and get the corresponding times
        self.t_minimas = obj.time[obj.min_indx]
        self.y_minimas = -obj.signal[obj.min_indx]

        # obtain the values of the maximum peaks and their corresponding times
        self.t_maximas = obj.time[obj.max_indx]
        self.y_maximas = obj.signal[obj.max_indx]
        
        # include (0,0) points to the list of minimas and maximas
        if self.include_zero_zero:
            self.t_maximas = np.concatenate((self.t_maximas, [0]))
            self.t_minimas = np.concatenate((self.t_minimas, [0]))
            self.y_maximas = np.concatenate((self.y_maximas, [0]))
            self.y_minimas = np.concatenate((self.y_minimas, [0]))
       
    def Newtonian_e_t(self, t, e_0, q, tau_0=None):
        """
        Newtonian Eccentricity evolution equation obtained from PAGE 41, Eq C1 of https://arxiv.org/pdf/1605.00304
        e_0: initial eccentricity
        q: mass ratio >=1
        tau_0: reference time
        """
        
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # obtain reference time for tau
        if tau_0 is None:
            # compute tau_0
            tau_0 = (self.tc-self.t_ref) * (eta/5)

        # Calculate the components of the expression
        term1 = e_0 * (tau / tau_0)**(19 / 48)
        # Sum all terms
        result = term1
        return result

    def PN_order2_e_t(self, t, e_0, q, tau_0=None):
        """
        2PN Eccentricity evolution equation
        e_0: initial eccentricity
        q: mass ratio >=1
        tau_0: reference time
        """
        
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # obtain reference time for tau
        if tau_0 is None:
            # compute tau_0
            tau_0 = (self.tc-self.t_ref) * (eta/5)

        # Calculate the components of the expression
        # Compute g_2PN
        term1 = (tau / tau_0)**(19 / 48)
        term2 = -4445 / 6912 * (tau**(-1 / 4) - tau_0**(-1 / 4))
        term3 = 854531845 / 4682022912 * tau**(-1 / 2)
        term4 = 1081754605 / 4682022912 * tau_0**(-1 / 2)
        term5 = -19758025 / 47775744 * tau**(-1 / 4) * tau_0**(-1 / 4)
        term6 = -3721 / 33177600 * np.pi**2 * tau**(-3 / 8) * tau_0**(-3 / 8)
        term7 = (255918223951763603 / 186891372173721600 - 
                  15943 / 80640 * np.euler_gamma  - 
                  7926071 / 66355200 * np.pi**2) * tau**(-3 / 4)
        term8 = (-250085444105408603 / 186891372173721600 + 
                  15943 / 80640 * np.euler_gamma  + 
                  7933513 / 66355200 * np.pi**2) * tau_0**(-3 / 4)

        g_2pn_value = term1 * (1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
    
        et = e_0 * g_2pn_value
        
        return et
    
    def PN_e_t(self, t, e_0, q, tau_0=None):
        """
        3PN Eccentricity evolution equation obtained from PAGE 41, Eq C1 of https://arxiv.org/pdf/1605.00304
        e_0: initial eccentricity
        q: mass ratio >=1
        tau_0: reference time
        """
        
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # obtain reference time for tau
        if tau_0 is None:
            # compute tau_0
            tau_0 = (self.tc-self.t_ref) * (eta/5)

        # Calculate the components of the expression
        term1 = e_0 * (tau / tau_0)**(19 / 48)
        term2 = (-4445 / 6912 + 185 / 576 * eta) * (tau**(-1 / 4) - tau_0**(-1 / 4))
        term3 = -61 / 5760 * np.pi * (tau**(-3 / 8) - tau_0**(-3 / 8))
        term4 = (854531845 / 4682022912 - 15215083 / 27869184 * eta + 72733 / 663552 * eta**2) * tau**(-1 / 2)
        term5 = (1081754605 / 4682022912 + 3702533 / 27869184 * eta - 4283 / 663552 * eta**2) * tau_0**(-1 / 2)
        term6 = (-19758025 / 47775744 + 822325 / 1990656 * eta - 34225 / 331776 * eta**2) * tau**(-1 / 4) * tau_0**(-1 / 4)
        term7 = (104976437 / 278691840 - 4848113 / 23224320 * eta) * np.pi * tau**(-5 / 8)
        term8 = (-101180407 / 278691840 + 4690123 / 23224320 * eta) * np.pi * tau_0**(-5 / 8)
        term9 = np.pi * (-54229 / 7962624 + 2257 / 663552 * eta) * (tau**(-1 / 4) * tau_0**(-3 / 8) + tau**(-3 / 8) * tau_0**(-1 / 4))
        term10 = (-686914174175 / 4623163195392 - 10094675555 / 898948399104 * eta + 501067585 / 10701766656 * eta**2 - 792355 / 382205952 * eta**3) * tau**(-1 / 4) * tau_0**(-1 / 2)
        term11 = -3721 / 33177600 * np.pi**2 * tau**(-3 / 8) * tau_0**(-3 / 8)
        term12 = (542627721575 / 4623163195392 - 122769222935 / 299649466368 * eta + 2630889335 / 10701766656 * eta**2 - 13455605 / 382205952 * eta**3) * tau**(-1 / 2) * tau_0**(-1 / 4)
        term13 = (255918223951763603 / 186891372173721600 
                  - 15943 / 80640 * np.euler_gamma 
                  - 7926071 / 66355200 * np.pi**2 
                  + (-81120341684927 / 13484225986560 + 12751 / 49152 * np.pi**2) * eta
                  - 3929671247 / 32105299968 * eta**2 
                  + 25957133 / 1146617856 * eta**3 
                  - 8453 / 15120 * np.log(2) 
                  + 26001 / 71680 * np.log(3) 
                  + 15943 / 645120 * np.log(tau)) * tau**(-3 / 4)
        term14 = (-250085444105408603 / 186891372173721600 
                  + 15943 / 80640 * np.euler_gamma 
                  + 7933513 / 66355200 * np.pi**2 
                  + (86796376850327 / 13484225986560 - 12751 / 49152 * np.pi**2) * eta
                  - 5466199513 / 32105299968 * eta**2 
                  + 16786747 / 1146617856 * eta**3 
                  + 8453 / 15120 * np.log(2) 
                  - 26001 / 71680 * np.log(3) 
                  - 15943 / 645120 * np.log(tau_0)) * tau_0**(-3 / 4)

        # Sum all terms
        result = term1 * (1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13 + term14)
        return result

    
    def PNorder_to_func_translation(self, order):
        """
        function to translate fit function PN order to fit function name
        """
        PNorder_to_func_dict = { '2PN': self.fit_func_2PN,
                                 '3PN': self.fit_func_3PN,
                                 '3PN_m1over8': self.fit_func_3PN_m1over8,
                                 '3PN_m7over8': self.fit_func_3PN_m7over8,
                                 '3PN_m8over8': self.fit_func_3PN_m8over8,
                                 '3PN_m1over8_m8over8': self.fit_func_3PN_m1over8_m8over8,
                                 '3PN_m1over8_m7over8': self.fit_func_3PN_m1over8_m7over8,
                                 '3PN_m7over8_m8over8': self.fit_func_3PN_m7over8_m8over8,
                                 '3PN_m1over8_m7over8_m8over8': self.fit_func_3PN_m1over8_m7over8_m8over8}
        
        return PNorder_to_func_dict[order]
     
     
    def fit_func_2PN(self, t, e_0):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 2PN terms
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
        
        return self.PN_order2_e_t(t, e_0, self.q, tau_0)
    
    def fit_func_3PN(self, t, e_0):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 3PN terms
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
        
        return self.PN_e_t(t, e_0, self.q, tau_0)
    
    def fit_func_3PN_m1over8(self, t, e_0, A1):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 3PN and pseudo terms upto 3.5PN
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
        
        # compute eccentricity
        e_3PN = self.PN_e_t(t, e_0, self.q, tau_0)
        e_m1over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A1 * (tau / tau_0)**(-1/8)
            
        return e_3PN + e_m1over8
     
    def fit_func_3PN_m7over8(self, t, e_0, A7):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 3PN and pseudo terms upto 3.5PN
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
        
        # compute eccentricity
        e_3PN = self.PN_e_t(t, e_0, self.q, tau_0)
        e_m7over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A7 * (tau / tau_0)**(-7/8)
            
        return e_3PN + e_m7over8
     
    def fit_func_3PN_m8over8(self, t, e_0, A8):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 3PN and pseudo terms upto 3.5PN
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
        
        # compute eccentricity
        e_3PN = self.PN_e_t(t, e_0, self.q, tau_0)
        e_m8over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A8 * (tau / tau_0)**(-8/8)
            
        return e_3PN + e_m8over8
     
    
    def fit_func_3PN_m1over8_m8over8(self, t, e_0, A1, A8):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 3PN and pseudo terms upto 4PN
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
        
        # compute eccentricity
        e_3PN = self.PN_e_t(t, e_0, self.q, tau_0)
        e_m1over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A1 * (tau / tau_0)**(-1/8)
        e_m8over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A8 * (tau / tau_0)**(-1)
            
        return e_3PN + e_m1over8 + e_m8over8
    
    def fit_func_3PN_m1over8_m7over8(self, t, e_0, A1, A7):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 3PN and pseudo terms upto 4PN
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
        
        # compute eccentricity
        e_3PN = self.PN_e_t(t, e_0, self.q, tau_0)
        e_m1over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A1 * (tau / tau_0)**(-1/8)
        e_m7over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A7 * (tau / tau_0)**(-7/8)
            
        return e_3PN + e_m1over8 + e_m7over8
    
    def fit_func_3PN_m7over8_m8over8(self, t, e_0, A7, A8):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 3PN and pseudo terms upto 4.5PN
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
         
        # compute eccentricity
        e_3PN = self.PN_e_t(t, e_0, self.q, tau_0)
        e_m7over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A7 * (tau / tau_0)**(-7/8)
        e_m8over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A8 * (tau / tau_0)**(-1)
            
        return e_3PN + e_m7over8 + e_m8over8
    
    def fit_func_3PN_m1over8_m7over8_m8over8(self, t, e_0, A1, A7, A8):
        """
        function to fit the numerically obtained maximas and minimas
        includes real 3PN and pseudo terms upto 4.5PN
        """
        # compute symmetric mass ratio
        eta = gwtools.q_to_nu(self.q)
        
        # time transformation
        tau = (self.tc-t) * (eta/5)
    
        # compute tau_0
        tau_0 = (self.tc-self.t_ref) * (eta/5)
            
        # compute eccentricity
        e_3PN = self.PN_e_t(t, e_0, self.q, tau_0)
        e_m1over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A1 * (tau / tau_0)**(-1/8)
        e_m7over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A7 * (tau / tau_0)**(-7/8)
        e_m8over8 = (e_0 * (tau / tau_0)**(19 / 48)) * A8 * (tau / tau_0)**(-1)
            
        return e_3PN + e_m1over8 + e_m7over8 + e_m8over8
    
    def _fit_maximas(self):
        """
        fit the maximas using scipy.curve_fit
        """
        self.popt_maximas, self.pcov_maximas = curve_fit(self.fit_funcs[0], self.t_maximas, self.y_maximas, maxfev=25000)
        
    def _fit_minimas(self):
        """
        fit the minimas using scipy.curve_fit
        """
        self.popt_minimas, self.pcov_minimas = curve_fit(self.fit_funcs[1], self.t_minimas, self.y_minimas, maxfev=25000)
        
    def _get_upper_xi_envelope(self):
        """
        obtain upper envelop of the modulation time series
        """
        self.xi_upper = self.fit_funcs[0](self.time_xi, *self.popt_maximas)
        
        if self.set_unphysical_xi_to_zero:
            # change NaN values to zero
            mask = np.where(np.isnan(self.xi_upper))
            self.xi_upper[mask] = np.zeros(len(self.xi_upper[mask]))
            # change negative values to zero
            mask = np.where(self.xi_upper<0)
            self.xi_upper[mask] = np.zeros(len(self.xi_upper[mask]))
        
    def _get_lower_xi_envelope(self):
        """
        obtain lower envelop of the modulation time series
        """
        self.xi_lower = self.fit_funcs[1](self.time_xi, *self.popt_minimas)
        
        if self.set_unphysical_xi_to_zero:
            # change NaN values to zero
            mask = np.where(np.isnan(self.xi_lower))
            self.xi_lower[mask] = np.zeros(len(self.xi_lower[mask]))
            # change negative values to zero
            mask = np.where(self.xi_lower<0)
            self.xi_lower[mask] = np.zeros(len(self.xi_lower[mask]))
        
    def _get_avg_xi_envelope(self):
        """
        obtain avg envelop of the modulation time series
        """
        self.xi_avg = 0.5 * (self.xi_upper + self.xi_lower)
        
    def _compute_eccentricity_evolution(self):
        """
        compute eccentricity evolution using modulation envelops
        """
        self.ecc_xi = self.xi_avg
        
        if self.set_unphysical_ecc_to_zero:
            # change NaN values to zero
            mask = np.where(np.isnan(self.ecc_xi))
            self.ecc_xi[mask] = np.zeros(len(self.ecc_xi[mask]))
            # change negative values to zero
            mask = np.where(self.ecc_xi<0)
            self.ecc_xi[mask] = np.zeros(len(self.ecc_xi[mask]))
        
    def _compute_eccentricity_at_tref(self):
        """
        compute eccentricity at tref
        """
        self.ecc_ref = gwtools.interpolate_h(self.time_xi, self.ecc_xi, [self.t_ref])[0]
        print('... gwModels eccentricity at t_ref=%.2f : %.5f'%(self.t_ref, self.ecc_ref))
        
    def _get_xi_upper_interp(self):
        """
        obtain 1d interpolation function between time and upper envelop
        """
        interp = interpolate.interp1d(self.time_xi, self.xi_upper)
        return interp

    def _get_xi_lower_interp(self):
        """
        obtain 1d interpolation function between time and lower envelop
        """
        interp = interpolate.interp1d(self.time_xi, self.xi_lower)
        return interp
    
    def _get_eccentricity_interp(self):
        """
        obtain 1d interpolation function between time and eccentricity
        """
        interp = interpolate.interp1d(self.time_xi, self.ecc_xi)
        return interp
    
    def _compute_xi_upper_fit_errors(self):
        """
        errors in upper envelop fit
        """
        return mathcalE_error(self.y_maximas, self.xi_upper_interp(self.t_maximas))
    
    def _compute_xi_lower_fit_errors(self):
        """
        errors in upper envelop fit
        """
        return mathcalE_error(self.y_minimas, self.xi_lower_interp(self.t_minimas))

    def plot_xi(self, figsize=(8,5)):
        """
        plot the modulation parameter xi
        """
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.modulations, color='C0', markersize=10, alpha=0.7)
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('${\\xi}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show()
        
    def plot_xi_with_peaks(self, figsize=(8,5)):
        """
        plot the modulation parameter xi along with the periastron and apastron peaks
        """
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.modulations, color='C0', markersize=10, alpha=0.7)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Maximas')
        plt.plot(self.t_minimas, -self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Minimas')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('${\\xi}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()
        
    def plot_maximas_fit(self, figsize=(8,5)):
        """
        plot upper envelop fits
        """
        plt.figure(figsize=figsize)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Numerical')
        plt.plot(self.time_xi, self.xi_upper, label='PN fit with $e_0=%.3f$'%self.popt_maximas[0], c='C0')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$|{\\xi}_{\\rm upper}^{\\rm env}|$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()
        
    def plot_minimas_fit(self, figsize=(8,5)):
        """
        plot lower envelop fits
        """
        plt.figure(figsize=figsize)
        plt.plot(self.t_minimas, self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Numerical')
        plt.plot(self.time_xi, self.xi_lower, label='PN fit with $e_0=%.3f$'%self.popt_minimas[0], c='C1')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$|{\\xi}_{\\rm lower}^{\\rm env}|$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()
        
    def plot_fit_errors(self, figsize=(8,5)):
        """
        plot errors upper and lower envelop fits
        """
        plt.figure(figsize=figsize)
        plt.plot(self.t_maximas, self.y_maximas-self.xi_upper_interp(self.t_maximas), 'o', markersize=4, label='Upper envelop')
        plt.plot(self.t_minimas, self.y_minimas-self.xi_lower_interp(self.t_minimas), 's', markersize=4, label='Lower envelop')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('Fit errors', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()

    def plot_xi_with_peaks_and_fits(self, figsize=(8,5)):
        """
        plot the modulation parameter xi along with upper and lower envelop fits
        """
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.modulations, color='C0', markersize=10, alpha=0.7)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Maximas')
        plt.plot(self.t_minimas, -self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Minimas')
        plt.plot(self.time_xi, self.xi_upper, label='PN fit with $e_0=%.3f$'%self.popt_maximas[0], c='C0')
        plt.plot(self.time_xi, -self.xi_lower, label='PN fit with $e_0=%.3f$'%self.popt_minimas[0], c='C1')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('${\\xi}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()
        
    def plot_maximas_and_minimas_fit(self, figsize=(8,5)):
        """
        plot both upper and lower envelop fits
        """
        plt.figure(figsize=figsize)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$')
        plt.plot(self.time_xi, self.xi_upper, label='PN fit with $e_0=%.3f$'%self.popt_maximas[0], c='C0')
        plt.plot(self.t_minimas, self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$')
        plt.plot(self.time_xi, self.xi_lower, label='PN fit with $e_0=%.3f$'%self.popt_minimas[0], c='C1')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$|{\\xi}^{\\rm env}|$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()
        
    def plot_maximas_minimas_and_avg_fit(self, figsize=(8,5)):
        """
        plot both upper and lower envelop fits along with the average envelop
        """
        plt.figure(figsize=figsize)
        plt.plot(self.t_maximas, self.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$')
        plt.plot(self.time_xi, self.xi_upper, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$ fit with $e_0=%.3f$'%self.popt_maximas[0], c='C0')
        plt.plot(self.t_minimas, self.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$')
        plt.plot(self.time_xi, self.xi_lower, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$ fit with $e_0=%.3f$'%self.popt_minimas[0], c='C1')
        plt.plot(self.time_xi, self.xi_avg, label='$|{\\xi}_{\\rm avg}^{\\rm env}|$', c='k', ls='--')
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$|{\\xi}^{\\rm env}|$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(fontsize=12)
        plt.show()
        
    def plot_eccentricity(self, figsize=(8,5)):
        """
        plot eccentricity evolution
        """
        plt.figure(figsize=figsize)
        plt.plot(self.time_xi, self.ecc_xi, c='C0', ls='-')
        plt.axvline(x=self.t_ref, c='k', ls='--')
        plt.text(self.t_ref+10, self.ecc_ref*0.5, '$e_{\\rm ref}$', fontsize=14, color='red', rotation=90) 
        plt.plot(self.t_ref, self.ecc_ref, 'o', color='red') 
        plt.xlabel('Time', fontsize=15)
        plt.ylabel('$e_{\\xi}$', fontsize=18)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.show()
        
        
        
class ComputeEccentricity():
    """
    Class to compute eccentricity using 22 mode eccentric and circular waveform
    """
    def __init__(self, t_ecc=None, h_ecc_dict=None, t_cir=None, h_cir_dict=None, 
                 q=None, t_ref=None, ecc_prefactor=None, distance_btw_peaks=None,
                 fit_funcs_orders=None, include_zero_zero=False, 
                 set_unphysical_xi_to_zero=False, set_unphysical_ecc_to_zero=True, 
                 method='xi_amp', use_xi_amp_to_get_xi_freq=False, tc=0, t_buffer=0):
        """
        t_ecc: time array for the eccentric 22 mode waveform
        h_ecc_dict: dictionary of eccentric wavefform modes. Should only contain 22 mode
        t_cir: time array for the circular waveform modes
        h_cir_dict: dictionary of circular non-spining waveform modes
                    keys should be 'h_l2m2', 'h_l2m1' and so on
        q: mass ratio (q>=1)
        t_ref: reference time to compute eccentricity
        ecc_prefactor: pre-factor in eccentricity definition;
                       default is 2/3
        distance_btw_peaks: distance between peaks to be passed to the AvergaeTimeSeriesEstimator;
                            default: 100
        include_zero_zero: if True, it will include (t=0,y=0) point to both the list of minimas and maximas;
                           this option sometimes leads to bad fit;
        set_unphysical_xi_to_zero: if True, set all negative xi and NaN values in the fitted xi time-zeries to zero
        set_unphysical_ecc_to_zero: if True, set all negative and NaN values in the fitted eccentricity time-zeries to zero
        method: 'xi_amp' / 'xi_freq'
        use_xi_amp_to_get_xi_freq: if True, it will compute freq modulation from amp modulation
        tc: time at merger; default is zero
        """
        # read eccentric waveform quantities
        self.t_ecc = t_ecc
        self.h_ecc_dict = h_ecc_dict
        if self.t_ecc is None:
            raise ValueError("t_ecc must be given as input")
        if self.h_ecc_dict is None:
            raise ValueError("h_ecc_dict must be given as input")
          
        # read circular waveform quantities
        self.t_cir = t_cir
        self.h_cir_dict = h_cir_dict
        if self.t_cir is None:
            raise ValueError("t_cir must be given as input")
        if self.h_cir_dict is None:
            raise ValueError("h_cir_dict must be given as input")
            
        # mass ratio value; required for the PN inspired fits
        self.q = q
        if self.q is None:
            raise ValueError("q must be given as input")
   
        # ecentricity estimation method
        self.method = method
        self.use_xi_amp_to_get_xi_freq = use_xi_amp_to_get_xi_freq
        
        # obtain modulations from eccentric and circular data
        # do not forget to set 'project_ecc_on_higher_modes=False'
        # otherwise it will give errors
        self.gwnrhme_obj = NRHME(t_ecc = self.t_ecc,
                                 h_ecc_dict = {'h_l2m2': self.h_ecc_dict['h_l2m2']},
                                 t_cir = self.t_cir,
                                 h_cir_dict = {'h_l2m2': self.h_cir_dict['h_l2m2']},
                                 project_ecc_on_higher_modes=False, 
                                 t_buffer=t_buffer)
        
        # amplitude to frequency modulation scaling parameter
        B = 0.9
        
        # use only pre-merger data as modulations after global peak will have noisy features
        t_premerger = self.gwnrhme_obj.t_common[self.gwnrhme_obj.t_common<=0]
        
        # if method is xi_amp, we will use the amplitude modulations as our common modulation parameter
        # for that, we will multiply it with the amp_to_freq modulation scaling
        if self.method == 'xi_amp':
            modulations_premerger = self.gwnrhme_obj.xi_amp[self.gwnrhme_obj.t_common<=0]/B
        # if method is xi_freq, we use modulations computed from the frequencies
        elif self.method == 'xi_freq':
            # however, sometimes, it is nice to use the amplitude modulations and scale it because
            # it has less noise
            if self.use_xi_amp_to_get_xi_freq:
                modulations_premerger = self.gwnrhme_obj.xi_amp[self.gwnrhme_obj.t_common<=0]/B
            # otherwise, directly use the frequency modulations
            else:
                modulations_premerger = self.gwnrhme_obj.xi_freq[self.gwnrhme_obj.t_common<=0]
        else:
            raise ValueError("Method not recognized")
        
        # check reference time input makes sense
        # using a buffer of 50M in time to make sure that interpolation later works fine!
        if t_ref is None:
            self.t_ref = self.gwnrhme_obj.t_common[0] + 10
        else:
            self.t_ref = t_ref
            if self.t_ref <= self.gwnrhme_obj.t_common[0] + 5:
                raise Warning("t_ref is very close to the start of the shorter waveform!")
                raise Warning("It is wise to use a t_ref that is atleast 10M larger than the earliest time in the shorter waveform")
     
        # parameter for peak finding
        if distance_btw_peaks is None:
            self.compute_approx_cycles(t_premerger)
            self.compute_required_distance_btw_peaks(t_premerger)
        else:
            self.distance_btw_peaks = distance_btw_peaks
            
        self.ecc_prefactor = ecc_prefactor
        self.tc = tc
        # include (0,0) to extremas
        self.include_zero_zero = include_zero_zero
        # whether to set negative xi and NaN values in the fitted xi time-zeries to zero
        self.set_unphysical_xi_to_zero = set_unphysical_xi_to_zero
        # whether to set negative and NaN values in the fitted eccentricity time-zeries to zero
        self.set_unphysical_ecc_to_zero = set_unphysical_ecc_to_zero
        
        # compute eccentricity using ComputeEccentricityFromModulations class
        exiobj = ComputeEccentricityFromModulations(time_xi=t_premerger, 
                                                     xi=modulations_premerger, 
                                                     q=self.q, 
                                                     t_ref=self.t_ref,
                                                     distance_btw_peaks=self.distance_btw_peaks,
                                                     ecc_prefactor=self.ecc_prefactor,
                                                     fit_funcs_orders=fit_funcs_orders,
                                                     include_zero_zero=self.include_zero_zero, 
                                                     set_unphysical_xi_to_zero=self.set_unphysical_xi_to_zero,
                                                     set_unphysical_ecc_to_zero=self.set_unphysical_ecc_to_zero,
                                                     tc=self.tc)
        
        
        # PN function
        self.PN_e_t = exiobj.PN_e_t
        self.Newtonian_e_t = exiobj.Newtonian_e_t
        
        # extract relevant quantities to the main class object
        # modulation parameters
        self.modulations = exiobj.modulations
        self.time_xi = exiobj.time_xi
        
        # peak parameters
        self.t_maximas = exiobj.t_maximas
        self.t_minimas = exiobj.t_minimas
        
        self.y_maximas = exiobj.y_maximas
        self.y_minimas = exiobj.y_minimas
        
        # peak fits
        self.pcov_maximas = exiobj.pcov_maximas
        self.popt_maximas = exiobj.popt_maximas
        
        self.pcov_minimas = exiobj.pcov_minimas
        self.popt_minimas = exiobj.popt_minimas
        
        # fitted envelops
        self.xi_avg = exiobj.xi_avg
        self.xi_lower = exiobj.xi_lower
        self.xi_upper = exiobj.xi_upper
        
        # eccentricity parameters
        self.ecc_ref = exiobj.ecc_ref
        self.ecc_xi = exiobj.ecc_xi
        self.teccmax = exiobj.teccmax
        
        # plot modulations
        self.plot_xi = exiobj.plot_xi
        self.plot_xi_with_peaks = exiobj.plot_xi_with_peaks
        
        # plot envelop fits
        self.fit_funcs = exiobj.fit_funcs
        self.plot_maximas_fit = exiobj.plot_maximas_fit
        self.plot_minimas_fit = exiobj.plot_minimas_fit
        self.plot_maximas_and_minimas_fit = exiobj.plot_maximas_and_minimas_fit
        self.plot_maximas_minimas_and_avg_fit = exiobj.plot_maximas_minimas_and_avg_fit
        self.plot_xi_with_peaks_and_fits = exiobj.plot_xi_with_peaks_and_fits
        self.plot_fit_errors = exiobj.plot_fit_errors
        
        # plot eccentricity
        self.plot_eccentricity = exiobj.plot_eccentricity
        
        # get interpolation objects
        self.xi_upper_interp = exiobj.xi_upper_interp
        self.xi_lower_interp = exiobj.xi_lower_interp
        self.ecc_interp = exiobj.ecc_interp
        
        # fit errors
        self.xi_upper_fit_error = exiobj.xi_upper_fit_error
        self.xi_lower_fit_error = exiobj.xi_lower_fit_error
        
        
    def compute_approx_cycles(self, t_premerger):
        """
        function to compute approximated number of cycles in usable eccentric data
        """
        # find time indices such that it is larger than the initial time point
        # in common time grid used for xi parameter and below t=0 merger time
        start_indx = np.where(self.t_ecc>t_premerger[0])[0][0]
        end_indx = np.where(self.t_ecc>0)[0][0]
        phase_at_the_start = abs(gwtools.phase(self.h_ecc_dict['h_l2m2'][start_indx:end_indx]))[0]
        phase_at_merger = abs(gwtools.phase(self.h_ecc_dict['h_l2m2'][start_indx:end_indx]))[-1]
        phase_change = phase_at_merger - phase_at_the_start
        self.approx_n_cycle = phase_change/(2*np.pi)
        self.approx_n_peaks_xi = self.approx_n_cycle/3
        self.waveform_duration_until_merger = abs(self.t_ecc[end_indx]-self.t_ecc[start_indx])

    def compute_required_distance_btw_peaks(self, t_premerger):
        """
        function to compute required distance between peaks to provide to the scipy routine
        in obtain actual peaks in case there is noise in the data
        """
        dt = np.diff(t_premerger)[0]
        # waveform duration in each cycle in time
        self.t_cycle_in_time = self.waveform_duration_until_merger / self.approx_n_peaks_xi
        # waveform duration in each cycle in sample
        self.t_cycle_in_samples = self.t_cycle_in_time / dt
        # approx distance between peaks in samples
        self.distance_btw_peaks = int(0.75 * self.t_cycle_in_samples)
            
        
  
  
        
    