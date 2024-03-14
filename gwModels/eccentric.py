#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: eccentric.py
#
#        AUTHOR: Tousif Islam
#       CREATED: 07-02-2024
# LAST MODIFIED: Tue Feb  6 17:58:52 2024
#      REVISION: ---
#==============================================================================

import numpy as np
import gwtools
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from .utility import *
import gwtools
import scipy

class NRHME:
    """
    Class to eamlessly convert a multi-modal quasi-circular spherical
    harmonic waveform into eccentric waveform spherical harmonic modes 
    if the quadrupolar eccentric waveform is known
    """
    def __init__(self, t_ecc=None, h_ecc_dict=None, t_cir=None, h_cir_dict=None, 
                 get_orbfreq_mod_from_amp_mod=False, recompute_tpeak=True):
        """
        t_ecc: time array for the eccentric 22 mode waveform
        h_ecc_dict: dictionary of eccentric wavefform modes. Should only contain 22 mode
        t_cir: time array for the circular waveform modes
        h_cir_dict: dictionary of circular non-spining waveform modes
                    keys should be 'h_l2m2', 'h_l2m1' and so on
        get_orbfreq_mod_from_amp_mod: False;
                                      If True, we compute the modulation in the orbital frequency from the amplitude
                                      modulation itself. This is recommended when the 22 mode eccentric amplitude 
                                      model is
        """
        self.t_ecc = t_ecc
        self.h_ecc_dict = h_ecc_dict
        if self.t_ecc is None:
            raise ValueError("t_ecc must be given as input")
        if self.h_ecc_dict is None:
            raise ValueError("h_ecc_dict must be given as input")
            
        self.t_cir = t_cir
        self.h_cir_dict = h_cir_dict
        if self.t_cir is None:
            raise ValueError("t_cir must be given as input")
        if self.h_cir_dict is None:
            raise ValueError("h_cir_dict must be given as input")
            
        self.modelist = list(h_cir_dict.keys())
        if self.modelist[0] in ['h_l2m0', 'h_l2m1','h_l2m2','h_l3m0','h_l3m1','h_l3m2','h_l3m3','h_l4m0','h_l4m1','h_4m2', 'h_l4m3', 'h_l4m4']:
            self.modekeytype = 'h_llmm'
        elif self.modelist[0] in [(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3),(4,0),(4,1),(4,2),(4,3),(4,4)]:
            self.modekeytype = '(l,m)'
        else:
            raise ValueError("Mode key not recognized. Please use dictionary keys as '(2,2)' or 'h_l2m2' format")
            
        # should we compute orbital frequency modulation separately
        self.get_orbfreq_mod_from_amp_mod = get_orbfreq_mod_from_amp_mod
        
        # should we recompute where the peaks are
        self.recompute_tpeak = recompute_tpeak
        
        # align peaks
        if self.recompute_tpeak:
            self.align_peaks()
        
        # cast on common time grid
        self.t_common = self.obtain_common_timegrid()
        
        self.cir_wfobj = AlignWFData(t_input=self.t_cir, h_input=self.h_cir_dict, t_common=self.t_common)
        self.ecc_wfobj = AlignWFData(t_input=self.t_ecc, h_input=self.h_ecc_dict, t_common=self.t_common)
        
        # modulations
        self.xi_amp = self.obtain_amplitude_modulation()
        self.xi_omega = self.obtain_orbfreq_modulation()
        
        # updated eccentric multimodal waveforms
        self.hNRE = self.obtain_eccentricHM()
        
    
    def align_peaks(self):
        """
        align all waveforms so that merger occurs at t=0
        we define merger at the point where 22 mode amplitude is the largest
        """
        tpeak_cir = get_peak(self.t_cir, abs(self.h_cir_dict['h_l2m2']))[0]
        tpeak_ecc = get_peak(self.t_ecc, abs(self.h_ecc_dict['h_l2m2']))[0]
        self.t_cir = self.t_cir - tpeak_cir
        self.t_ecc = self.t_ecc - tpeak_ecc
        
        
    def obtain_common_timegrid(self):
        """
        construct a common time-grid between the circular waveform and the eccentric
        22 mode waveform
        """
        t_buffer = 100
        end_time = 100
        tmin = max(min(self.t_cir),min(self.t_ecc)) + t_buffer
        tmax = min(max(self.t_cir),max(self.t_ecc),end_time)
        tcommon = np.arange(tmin,tmax,0.1)
        return tcommon
    
    
    def obtain_amplitude_modulation(self):
        """
        compute the amplitude modulation from the 22 mode eccentric and 22 mode
        quasicircular waveform
        """
        ecc_quadrupole_amp = abs(self.ecc_wfobj.h_transform['h_l2m2'])
        cir_quadrupole_amp = abs(self.cir_wfobj.h_transform['h_l2m2'])
        ecc_modulation = (ecc_quadrupole_amp-cir_quadrupole_amp)/cir_quadrupole_amp
        return ecc_modulation
    
    
    def obtain_orbfreq_modulation(self):
        """
        compute the orbital frequency modulation from the 22 mode eccentric and 22 mode
        quasicircular waveform;
        when get_orbfreq_mod_from_amp_mod flag is turned on, we will use the amplitude modulation
        and scale it appropritaely to obtain the frequency modulations
        """
        if self.get_orbfreq_mod_from_amp_mod is False:
            ecc_quadrupole_orbfreq = get_frequency(self.t_common, self.ecc_wfobj.h_transform['h_l2m2'])
            cir_quadrupole_orbfreq = get_frequency(self.t_common, self.cir_wfobj.h_transform['h_l2m2'])
            ecc_modulation = (ecc_quadrupole_orbfreq-cir_quadrupole_orbfreq)/cir_quadrupole_orbfreq
        else:
            K = 0.9
            ecc_modulation = self.xi_amp * K
        return ecc_modulation
        
        
    def twist_mode_amplitude(self, mode):
        """
        turn each circular mode amplitude into their corresponding eccentric mode amplitude
        """
        ell = float(mode.rsplit("_l")[-1].rsplit("m")[0])
        scaling_factor = ell/2.0
        return abs(self.cir_wfobj.h_transform[mode]) * (scaling_factor * self.xi_amp + 1)
            
    
    def twist_mode_orbital_frequency(self, mode):
        """
        turn each circular mode orbital frequency into their corresponding eccentric mode amplitude
        """
        cir_frequency = get_frequency(self.t_common, self.cir_wfobj.h_transform[mode])
        return cir_frequency * (1 + self.xi_omega)
    
    
    def twist_mode_phase(self, mode):
        """
        obtain eccentric phase term for a given mode using orbital frequency modulation
        """
        omega =  self.twist_mode_orbital_frequency(mode)
        phase = scipy.integrate.cumtrapz(omega, self.t_common, initial=0)
        return phase
        
    def twist_modes(self):
        """
        twsit a circular waveform modes into eccentric modes
        """
        hNRE = {}
        for mode in self.modelist:
            if mode == 'h_l2m2':
                hNRE[mode] = self.ecc_wfobj.h_transform[mode]
                amplitude = abs(hNRE[mode])
                phase = gwtools.phase(hNRE[mode])
                phase = phase - phase[0] + gwtools.phase(self.cir_wfobj.h_transform[mode])[0]
                hNRE[mode] = amplitude * np.exp(1j*phase)
            else:
                amplitude = self.twist_mode_amplitude(mode)
                phase = self.twist_mode_phase(mode) 
                phase = phase -phase[0]  + gwtools.phase(self.cir_wfobj.h_transform[mode])[0]
                hNRE[mode] = amplitude * np.exp(1j*phase)
        return hNRE
    
    def obtain_eccentricHM(self):
        """
        align eccentric modes properly in phases and times
        """
        gwhNRE = self.twist_modes()
        gwhNRE = phase_align_dict(gwhNRE)
        gwhNRE = check_pi_rotation(gwhNRE)
        return gwhNRE