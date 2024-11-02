#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: circular_models.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-01-2024
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import sys, os
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import matplotlib.pyplot as plt
import numpy as np

import gwsurrogate
sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

class genBHPTNRSur1dq1e4:
    """
    Class to generate BHPTNRSur1dq1e4 waveform
    """
    def __init__(self, model):
        """
        model: BHPTNRSur1dq1e4 object
        """
        # instantiate surrogate model
        self.model = model

    def _generate_raw_BHPTNRSur1dq1e4(self, model, params):
        """
        generate BHPTNRSur1dq1e4 waveforms directly from gwsurrogate
        """
        modes, times, hp, hc = model(q=params["q"], 
                                     ell=[2,2,3,3,4,4], 
                                     m=[2,1,3,2,4,3], 
                                     mode_sum=False, 
                                     fake_neg_modes=False)
        return modes, times, hp, hc

    def _process_BHPTNRSur1dq1e4_output(self, modes, hp, hc):
        """
        process BHPTNRSur1dq1e4 output in a way it has similar keys
        that gwNRHME accepts
        """
        hdict = {}
        flag = 0
        for mode in modes:
            hdict['h_l%dm%d'%(mode[0],mode[1])] = hp[:, flag] + 1j*hc[:, flag]
            flag = flag + 1
        return hdict

    def generate_BHPTNRSur1dq1e4(self, model, params):
        """
        generate raw BHPTNRSur1dq1e4 waveforms from gwsurrogate
        """
        # raw output
        modes, times, hp, hc = self._generate_raw_BHPTNRSur1dq1e4(model, params)
        # process output
        t_sur = times
        h_sur = self._process_BHPTNRSur1dq1e4_output(modes, hp, hc)
        return t_sur, h_sur


class genNRHybSur3dq8:
    """
    Class to generate NRHybSur3dq8 waveform
    """
    def __init__(self):
        # instantiate surrogate model
        self.sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

    def _generate_raw_NRHybSur3dq8(self, params):
        """
        generate NRHybSur3dq8 waveforms
        """
        # mass ratio
        q = params["q"]
        # spins
        for key in ["s1z", "s2z"]:
            if key in params:
                pass
            else:
                # If the key doesn't exist, add it with a value of zero
                params[key] = 0
        chiA = [0, 0, params["s1z"]]
        chiB = [0, 0, params["s2z"]]
        # step size in time
        dt = 0.1    
        # initial frequency, Units of cycles/M
        # 0.99 is just to be safe that the circular waveform is a bit longer
        # than the eccentric one
        f_low = 0.99 * (params['x0']**(3/2))/(np.pi)   
        # dyn stands for dynamics and is always None for this model
        t, h, dyn = sur(q, chiA, chiB, dt=dt, f_low=f_low)        
        return t, h
    
    def _process_NRHybSur3dq8_output(self, t, h):
        """
        process NRhybSur3dq8 output in a way it has similar keys
        that gwNRHME accepts
        """
        t_sur = t
        h_sur = {}
        h_sur['h_l2m1'] = h[(2,1)]
        
        for mode in h.keys():
            if mode == (2,1):
                pass
            else:
                if mode[1]>0:
                    h_sur['h_l%dm%d'%(mode[0],mode[1])] = h[mode]
                else:
                    pass
        return t_sur, h_sur

    def generate_NRHybSur3dq8(self, params):
        """
        generate raw NRhybSur3dq8 waveforms from gwsurrogate
        """
        # raw output
        t, h = self._generate_raw_NRHybSur3dq8(params)
        # process output
        t_sur, h_sur = self._process_NRHybSur3dq8_output(t, h)
        return t_sur, h_sur
        