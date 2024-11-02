#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwnrxhme.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-02-2024
#    LAST MODIFIED: Tue Feb  6 17:58:52 2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ..utils import *
import gwtools
import scipy

class NRXHME:
    """
    Class to seamlessly convert a multi-modal (several spherical
    harmonic modes present) quasi-circular non-precessing waveform 
    into multi-modal eccentric non-precessing waveform if the 
    non-precessing quadrupolar eccentric waveform is known
    """
    def __init__(self, t_ecc=None, h_ecc_dict=None, t_cir=None, h_cir_dict=None, 
                 get_orbfreq_mod_from_amp_mod=False, recompute_tpeak=True,
                 project_ecc_on_higher_modes=True, t_buffer=100, end_time=100):
        """
        Initializes the NRHME class.

        Parameters:
            t_ecc (array): Time array for the eccentric 22 mode waveform.
            h_ecc_dict (dict): Dictionary of eccentric waveform modes, should only contain the 22 mode.
            t_cir (array): Time array for the circular waveform modes.
            h_cir_dict (dict): Dictionary of circular non-precessing waveform modes. 
                               Keys should include 'h_l2m2', 'h_l2m1', etc.
            get_orbfreq_mod_from_amp_mod (bool): If True, computes the modulation in the orbital frequency 
                                                  from the amplitude modulation itself. Default is False.
            recompute_tpeak (bool): If True, recomputes the peaks of the waveforms. Default is True.
            project_ecc_on_higher_modes (bool): If True, projects the effect of eccentricity onto higher-order 
                                                 spherical harmonic modes. Default is True.
            t_buffer (float): Buffer time to exclude at the beginning of the data. Default is 100.
            end_time (float): Final time to keep in the common time grid. Default is 100.
        
        Calculates: 
            Multi-modal eccentric non-precessing waveform.
        """
        # read eccentric waveforms
        self.t_ecc = t_ecc
        self.h_ecc_dict = h_ecc_dict
        if self.t_ecc is None:
            raise ValueError("t_ecc must be given as input")
        if self.h_ecc_dict is None:
            raise ValueError("h_ecc_dict must be given as input")

        # read circular waveforms
        self.t_cir = t_cir
        self.h_cir_dict = h_cir_dict
        if self.t_cir is None:
            raise ValueError("t_cir must be given as input")
        if self.h_cir_dict is None:
            raise ValueError("h_cir_dict must be given as input")

        # read modes and mode name convention
        # TODO: write a conversion function
        self.modelist = list(h_cir_dict.keys())
        if self.modelist[0] in ['h_l2m0', 'h_l2m1','h_l2m2','h_l3m0','h_l3m1','h_l3m2','h_l3m3','h_l4m0','h_l4m1','h_4m2', 'h_l4m3', 'h_l4m4',
                               'h_l2m-1','h_l2m-2','h_l3m-1','h_l3m-2','h_l3m-3','h_l4m-1','h_4m-2', 'h_l4m-3', 'h_l4m-4', 'h_l5m5', 'h_l5m-5']:
            self.modekeytype = 'h_llmm'
        elif self.modelist[0] in [(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3),(4,0),(4,1),(4,2),(4,3),(4,4)]:
            self.modekeytype = '(l,m)'
        else:
            raise ValueError("Mode key not recognized. Please use dictionary keys as '(2,2)' or 'h_l2m2' format")
            
        # get buffer time
        self.t_buffer = t_buffer
        self.end_time = end_time
        
        # should we compute orbital frequency modulation separately
        self.get_orbfreq_mod_from_amp_mod = get_orbfreq_mod_from_amp_mod
        
        # should we recompute where the peaks are
        self.recompute_tpeak = recompute_tpeak
        
        # whether to project eccentricity on higher order modes
        self.project_ecc_on_higher_modes = project_ecc_on_higher_modes
        
        # align peaks
        if self.recompute_tpeak:
            self.align_peaks()
        
        # cast on common time grid
        self.t_common = self.obtain_common_timegrid()

        # create eccentric and circular waveform objects
        self.cir_wfobj = AlignWFData(t_input=self.t_cir, h_input=self.h_cir_dict, t_common=self.t_common)
        self.ecc_wfobj = AlignWFData(t_input=self.t_ecc, h_input=self.h_ecc_dict, t_common=self.t_common)
        
        # modulations
        self.xi_amp = self.obtain_amplitude_modulation()
        self.xi_omega = self.obtain_orbfreq_modulation()
        
        # updated eccentric multimodal waveforms
        if self.project_ecc_on_higher_modes:
            self.hNRE = self.obtain_eccentricHM()
        
    
    def align_peaks(self):
        """
        Aligns all waveforms such that the merger occurs at t=0.
        The merger is defined as the point where the 22 mode amplitude is the largest.
        """
        # find the peak for the circular waveform
        tpeak_cir = get_peak(self.t_cir, abs(self.h_cir_dict['h_l2m2']))[0]
        # find the peak for the eccentric waveform
        tpeak_ecc = get_peak(self.t_ecc, abs(self.h_ecc_dict['h_l2m2']))[0]
        # ensure the peak of the circular waveform is at t=0
        self.t_cir = self.t_cir - tpeak_cir
        # ensure the peak of the eccentric waveform is at t=0
        self.t_ecc = self.t_ecc - tpeak_ecc
        
        
    def obtain_common_timegrid(self):
        """
        Constructs a common time grid between the circular waveform 
        and the eccentric 22 mode waveform.

        Returns:
            array: Common time grid array.
        """
        # minimum time in the common grid
        tmin = max(min(self.t_cir),min(self.t_ecc)) + self.t_buffer
        # maximum time in the common grid
        tmax = min(max(self.t_cir),max(self.t_ecc), self.end_time)
        # time grid
        tcommon = np.arange(tmin,tmax,0.1)
        return tcommon
    
    
    def obtain_amplitude_modulation(self):
        """
        Computes the amplitude modulation from the 22 mode eccentric 
        and circular waveforms.
        """
        # eccentric 22 amplitude
        ecc_quadrupole_amp = abs(self.ecc_wfobj.h_transform['h_l2m2'])
        # circular 22 amplitude
        cir_quadrupole_amp = abs(self.cir_wfobj.h_transform['h_l2m2'])
        # compute the amplitude modulations using Eq(4) of https://arxiv.org/pdf/2403.15506
        ecc_modulation = (ecc_quadrupole_amp-cir_quadrupole_amp)/cir_quadrupole_amp
        return ecc_modulation
    
    
    def obtain_orbfreq_modulation(self):
        """
        Computes the eccentric frequency modulation from the 22 mode eccentric 
        and circular waveforms. If get_orbfreq_mod_from_amp_mod is True, 
        it uses the amplitude modulation to scale frequency modulations.
        """
        if self.get_orbfreq_mod_from_amp_mod is False:
            # obtain frequency modulation from the 22 mode of the waveform data itself
            ecc_quadrupole_orbfreq = get_frequency(self.t_common, self.ecc_wfobj.h_transform['h_l2m2'])
            cir_quadrupole_orbfreq = get_frequency(self.t_common, self.cir_wfobj.h_transform['h_l2m2'])
            ecc_modulation = (ecc_quadrupole_orbfreq-cir_quadrupole_orbfreq)/cir_quadrupole_orbfreq
        else:
            # calculate frequency modulations using known relation between amplitude and frequency
            # modulations; use Eq(6) of https://arxiv.org/pdf/2403.15506
            K = 0.9
            ecc_modulation = self.xi_amp * K
        return ecc_modulation
        
        
    def twist_mode_amplitude(self, mode):
        """
        Convert the amplitude of a circular mode to its corresponding eccentric mode amplitude.
    
        Parameters:
            mode (str): The mode identifier (e.g., 'h_l2m2').
    
        Returns:
            float: The projected amplitude of the corresponding eccentric mode, 
                   calculated using Eq(9) of the referenced paper.
        """
        # find out the ell value of the mode
        ell = float(mode.rsplit("_l")[-1].rsplit("m")[0])
        # calculate the prefactor
        scaling_factor = ell/2.0
        # compute the amplitude of the corresponding mode
        # using Eq(9) of https://arxiv.org/pdf/2403.15506
        projected_amplitude = abs(self.cir_wfobj.h_transform[mode]) * (scaling_factor * self.xi_amp + 1)
        return projected_amplitude
            
    
    def twist_mode_orbital_frequency(self, mode):
        """
        Convert the orbital frequency of a circular mode to its corresponding 
        eccentric mode orbital frequency.
    
        Parameters:
            mode (str): The mode identifier (e.g., 'h_l2m2').
    
        Returns:
            float: The projected orbital frequency of the corresponding eccentric mode, 
                   calculated using Eq(10) of the referenced paper.
        """
        cir_frequency = get_frequency(self.t_common, self.cir_wfobj.h_transform[mode])
        # compute the frequency of the corresponding mode
        # using Eq(10) of https://arxiv.org/pdf/2403.15506
        projected_frequency = cir_frequency * (1 + self.xi_omega)
        return projected_frequency
    
    
    def twist_mode_phase(self, mode):
        """
        Obtain the eccentric phase term for a given mode using orbital frequency modulation.
    
        Parameters:
            mode (str): The mode identifier (e.g., 'h_l2m2').
    
        Returns:
            array: The computed phase of the corresponding eccentric mode, calculated using 
                   Eq(11) of the referenced paper. The integration constant is set to 0.
        """
        omega =  self.twist_mode_orbital_frequency(mode)
        
        # compute the phase of the corresponding mode
        # using Eq(11) of https://arxiv.org/pdf/2403.15506
        # However, we use integration constant=0 to begin with;
        # Integration constant is fixed later
        
        # Our favorite scipy.integrate.cumtrapz is now deprecated; 
        # so provide both the options for now
        try:
            phase = scipy.integrate.cumtrapz(omega, self.t_common, initial=0)
        except:
            phase = scipy.integrate.cumulative_trapezoid(omega, self.t_common, initial=0)
            print("..... scipy.integrate.cumtrapz is no longer supported by your environment; using newer module scipy.integrate.cumulative_trapezoid")
        return phase
        
    def twist_modes(self):
        """
        Transform circular waveform modes into eccentric waveform modes.

        Returns:
            dict: A dictionary containing the transformed eccentric waveform modes.
                  The amplitude and phase of the 'h_l2m2' mode remain unchanged,
                  while other modes are modified according to their respective 
                  projected amplitudes and phases.
        """
        # using Eq(11) of https://arxiv.org/pdf/2403.15506
        hNRE = {}
        for mode in self.modelist:
            if mode == 'h_l2m2':
                hNRE[mode] = self.ecc_wfobj.h_transform[mode]
                # amplitude unchanged
                amplitude = abs(hNRE[mode])
                # phase unchanged
                phase = gwtools.phase(hNRE[mode])
                # phase alignment with appropriate integration constant;
                # initial phase is equal to the initial phase of the circular waveform;
                # using Eq(11) of https://arxiv.org/pdf/2403.15506
                phase = phase - phase[0] + gwtools.phase(self.cir_wfobj.h_transform[mode])[0]
                hNRE[mode] = amplitude * np.exp(1j*phase)
            else:
                # amplitude is computed using universal modulation parameter
                amplitude = self.twist_mode_amplitude(mode)
                # phase is computed using universal modulation parameter
                phase = self.twist_mode_phase(mode) 
                # phase alignment with appropriate integration constant;
                # initial phase is equal to the initial phase of the circular waveform;
                # using Eq(11) of https://arxiv.org/pdf/2403.15506
                phase = phase - phase[0]  + gwtools.phase(self.cir_wfobj.h_transform[mode])[0]
                hNRE[mode] = amplitude * np.exp(1j*phase)
        return hNRE
    
    def obtain_eccentricHM(self):
        """
        Align eccentric modes in phases and times so that the initial phase matches 
        the corresponding circular phase.
    
        Returns:
            dict: A dictionary containing the aligned eccentric higher-order 
                  multipole waveform modes, ensuring initial phase alignment.
        """
        # all higher order spherical harmonics with eccentricity is obtained
        gwhNRE = self.twist_modes()
        # align waveform in phase to make sure initial phase is zero
        gwhNRE = phase_align_dict(gwhNRE)
        # check if a pi rotation is required
        gwhNRE = check_pi_rotation(gwhNRE)
        return gwhNRE