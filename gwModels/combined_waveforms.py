#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: combined_waveforms.py
#
#        AUTHOR: Tousif Islam
#       CREATED: 07-03-2024
# LAST MODIFIED: Tue Feb  6 17:58:52 2024
#      REVISION: ---
#==============================================================================

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import matplotlib.pyplot as plt
import numpy as np

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

# import gwModels
import sys
sys.path.append("/home/tousifislam/Documents/works/gwModels/")
import gwModels
import gwsurrogate
from .eccentric import *

import gwsurrogate
sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

class EccentricIMR_NRHybSur3dq8:
    """
    Class to generate eccentric higher order spherical harmonics using
    EccentricIMR and NRHybSur3dq8 model
    """
    def __init__(self, wolfram_kernel_path, package_directory):
        """
        wolfram_kernel_path: absolute path for your mathematica kernel
        package_directory: absolute path for the EccentricIMR package
        """
        self.wolfram_kernel_path = wolfram_kernel_path
        self.package_directory = package_directory

        # instantiate surrogate model
        self.sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
        
        # instantiate the EccentricIMR class - it may take some time
        self.wf = gwModels.EccentricIMR(self.wolfram_kernel_path, self.package_directory)


    def generate_waveform(self, params): 
        """
        user-friendly function to generate combined eccentric waveform
        params: dictionary with keys "q", "e0", "l0", "x0"
                q: mass ratio
                e0: initial eccentricity at x0
                l0: initial mean anomaly at x0
                x0: initial dimensionless orbital frequency
        """
        # generate eccentricIMR waveform
        tIMR, hIMR = self.wf.generate_waveform(params)
        params["fIMR"] = self._obtain_NRHybSur3dq8_flow(tIMR, hIMR)
        # generate surrogate cicular waveform
        t_sur, h_sur = self._generate_NRHybSur3dq8(params)
        # use gwNRHME to obtain multi-modal
        tNRE, hNRE = self._apply_gwNRHME(t_ecc=tIMR, h_ecc_dict={'h_l2m2': hIMR},
                                    t_cir=t_sur, h_cir_dict=h_sur)
        return tNRE, hNRE

    def _apply_gwNRHME(self, t_ecc, h_ecc_dict, t_cir, h_cir_dict):
        """
        converts circular higher modes to eccentric modes
        """
        hNRE_obj = NRHME(t_ecc, h_ecc_dict,
                                            t_cir, h_cir_dict,
                                            get_orbfreq_mod_from_amp_mod=False)

        tNRE = hNRE_obj.t_common
        hNRE = hNRE_obj.hNRE
        return tNRE, hNRE
        
        
    def _obtain_NRHybSur3dq8_flow(self, tIMR, hIMR):
        """
        obtains start frequency of the eccentric waveform
        this fequency is then passed to the circular waveform as f_low
        """
        fIMR = 0.9 * abs(gwModels.utility.get_frequency(tIMR, hIMR)[0])/(np.pi)
        return fIMR


    def _generate_raw_NRHybSur3dq8(self, params):
        """
        generate NRHybSur3dq8 waveforms
        """
        # mass ratio
        q = params["q"]
        # spins
        chiA = [0, 0, 0.0]
        chiB = [0, 0, 0.0]
        # step size in time
        dt = 0.1    
        # initial frequency, Units of cycles/M
        f_low = 0.95 * (params['x0']**(3/2))/(np.pi)   
        # dyn stands for dynamics and is always None for this model
        t, h, dyn = sur(q, chiA, chiB, dt=dt, f_low=f_low)        
        return t, h
    
    def _process_NRHybSur3dq8_output(self, t, h):
        """
        process NRhybSur3dq8 output in a way it has similar keys
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

    def _generate_NRHybSur3dq8(self, params):
        """
        generate raw NRhybSur3dq8 waveforms from gwsurrogate
        """
        # raw output
        t, h = self._generate_raw_NRHybSur3dq8(params)
        # process output
        t_sur, h_sur = self._process_NRHybSur3dq8_output(t, h)
        return t_sur, h_sur