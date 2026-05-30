#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: base.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-02-2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import gwtools
import scipy
from ..utils.alignment import get_peak, AlignWFData, phase_align_dict, check_pi_rotation
from ..utils.features import get_frequency
from ..utils.constants import B_AMP_FREQ


class BaseEccentricHM:
    """
    Base class to seamlessly convert a multi-modal (several spherical
    harmonic modes present) quasi-circular waveform into a multi-modal
    eccentric waveform if the quadrupolar eccentric waveform is known.

    Subclasses: NRHME (non-spinning), NRXHME (non-precessing).
    """
    _description = "base"

    def __init__(self, t_ecc=None, h_ecc_dict=None, t_cir=None, h_cir_dict=None,
                 get_orbfreq_mod_from_amp_mod=False, recompute_tpeak=True,
                 project_ecc_on_higher_modes=True, t_buffer=100, end_time=100):
        """
        Initializes the BaseEccentricHM class.

        Parameters:
            t_ecc (array): Time array for the eccentric 22 mode waveform.
            h_ecc_dict (dict): Dictionary of eccentric waveform modes, should only contain the 22 mode.
            t_cir (array): Time array for the circular waveform modes.
            h_cir_dict (dict): Dictionary of circular waveform modes.
                               Keys should include 'h_l2m2', 'h_l2m1', etc.
            get_orbfreq_mod_from_amp_mod (bool): If True, computes the modulation in the orbital frequency
                                                  from the amplitude modulation itself. Default is False.
            recompute_tpeak (bool): If True, recomputes the peaks of the waveforms. Default is True.
            project_ecc_on_higher_modes (bool): If True, projects the effect of eccentricity onto higher-order
                                                 spherical harmonic modes. Default is True.
            t_buffer (float): Buffer time to exclude at the beginning of the data. Default is 100.
            end_time (float): Final time to keep in the common time grid. Default is 100.

        Calculates:
            Multi-modal eccentric waveform.
        """
        if t_ecc is None:
            raise ValueError("t_ecc must be given as input")
        if h_ecc_dict is None:
            raise ValueError("h_ecc_dict must be given as input")
        if t_cir is None:
            raise ValueError("t_cir must be given as input")
        if h_cir_dict is None:
            raise ValueError("h_cir_dict must be given as input")

        self.t_ecc = t_ecc
        self.h_ecc_dict = h_ecc_dict
        self.t_cir = t_cir
        self.h_cir_dict = h_cir_dict

        self.modelist = list(h_cir_dict.keys())
        if self.modelist[0] in ['h_l2m0', 'h_l2m1','h_l2m2','h_l3m0','h_l3m1','h_l3m2','h_l3m3',
                                'h_l4m0','h_l4m1','h_4m2', 'h_l4m3', 'h_l4m4',
                                'h_l2m-1','h_l2m-2','h_l3m-1','h_l3m-2','h_l3m-3',
                                'h_l4m-1','h_4m-2', 'h_l4m-3', 'h_l4m-4', 'h_l5m5', 'h_l5m-5']:
            self.modekeytype = 'h_llmm'
        elif self.modelist[0] in [(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3),
                                  (4,0),(4,1),(4,2),(4,3),(4,4)]:
            self.modekeytype = '(l,m)'
        else:
            raise ValueError("Mode key not recognized. Please use dictionary keys as '(2,2)' or 'h_l2m2' format")

        self.t_buffer = t_buffer
        self.end_time = end_time
        self.get_orbfreq_mod_from_amp_mod = get_orbfreq_mod_from_amp_mod
        self.recompute_tpeak = recompute_tpeak
        self.project_ecc_on_higher_modes = project_ecc_on_higher_modes

        if self.recompute_tpeak:
            self.align_peaks()

        self.t_common = self.obtain_common_timegrid()

        self.cir_wfobj = AlignWFData(t_input=self.t_cir, h_input=self.h_cir_dict, t_common=self.t_common)
        self.ecc_wfobj = AlignWFData(t_input=self.t_ecc, h_input=self.h_ecc_dict, t_common=self.t_common)

        self.xi_amp = self.obtain_amplitude_modulation()
        self.xi_omega = self.obtain_orbfreq_modulation()

        if self.project_ecc_on_higher_modes:
            self.hNRE = self.obtain_eccentricHM()

    def align_peaks(self):
        """
        Aligns all waveforms such that the merger occurs at t=0.
        The merger is defined as the point where the 22 mode amplitude is the largest.
        """
        tpeak_cir = get_peak(self.t_cir, abs(self.h_cir_dict['h_l2m2']))[0]
        tpeak_ecc = get_peak(self.t_ecc, abs(self.h_ecc_dict['h_l2m2']))[0]
        self.t_cir = self.t_cir - tpeak_cir
        self.t_ecc = self.t_ecc - tpeak_ecc

    def obtain_common_timegrid(self):
        """
        Constructs a common time grid between the circular waveform
        and the eccentric 22 mode waveform.

        Returns:
            array: Common time grid array.
        """
        tmin = max(min(self.t_cir), min(self.t_ecc)) + self.t_buffer
        tmax = min(max(self.t_cir), max(self.t_ecc), self.end_time)
        return np.arange(tmin, tmax, 0.1)

    def obtain_amplitude_modulation(self):
        """
        Computes the amplitude modulation from the 22 mode eccentric
        and circular waveforms using Eq(4) of https://arxiv.org/pdf/2403.15506.
        """
        ecc_quadrupole_amp = abs(self.ecc_wfobj.h_transform['h_l2m2'])
        cir_quadrupole_amp = abs(self.cir_wfobj.h_transform['h_l2m2'])
        return (ecc_quadrupole_amp - cir_quadrupole_amp) / cir_quadrupole_amp

    def obtain_orbfreq_modulation(self):
        """
        Computes the eccentric frequency modulation from the 22 mode eccentric
        and circular waveforms. If get_orbfreq_mod_from_amp_mod is True,
        it uses the amplitude modulation to scale frequency modulations
        via Eq(6) of https://arxiv.org/pdf/2403.15506.
        """
        if self.get_orbfreq_mod_from_amp_mod is False:
            ecc_quadrupole_orbfreq = get_frequency(self.t_common, self.ecc_wfobj.h_transform['h_l2m2'])
            cir_quadrupole_orbfreq = get_frequency(self.t_common, self.cir_wfobj.h_transform['h_l2m2'])
            return (ecc_quadrupole_orbfreq - cir_quadrupole_orbfreq) / cir_quadrupole_orbfreq
        else:
            return self.xi_amp * B_AMP_FREQ

    def twist_mode_amplitude(self, mode):
        """
        Convert the amplitude of a circular mode to its corresponding eccentric mode amplitude.

        Parameters:
            mode (str): The mode identifier (e.g., 'h_l2m2').

        Returns:
            float: The projected amplitude of the corresponding eccentric mode,
                   calculated using Eq(9) of https://arxiv.org/pdf/2403.15506.
        """
        ell = float(mode.rsplit("_l")[-1].rsplit("m")[0])
        scaling_factor = ell / 2.0
        return abs(self.cir_wfobj.h_transform[mode]) * (scaling_factor * self.xi_amp + 1)

    def twist_mode_orbital_frequency(self, mode):
        """
        Convert the orbital frequency of a circular mode to its corresponding
        eccentric mode orbital frequency.

        Parameters:
            mode (str): The mode identifier (e.g., 'h_l2m2').

        Returns:
            float: The projected orbital frequency of the corresponding eccentric mode,
                   calculated using Eq(10) of https://arxiv.org/pdf/2403.15506.
        """
        cir_frequency = get_frequency(self.t_common, self.cir_wfobj.h_transform[mode])
        return cir_frequency * (1 + self.xi_omega)

    def twist_mode_phase(self, mode):
        """
        Obtain the eccentric phase term for a given mode using orbital frequency modulation.

        Parameters:
            mode (str): The mode identifier (e.g., 'h_l2m2').

        Returns:
            array: The computed phase of the corresponding eccentric mode, calculated using
                   Eq(11) of https://arxiv.org/pdf/2403.15506. The integration constant is set to 0.
        """
        omega = self.twist_mode_orbital_frequency(mode)
        phase = scipy.integrate.cumulative_trapezoid(omega, self.t_common, initial=0)
        return phase

    def twist_modes(self):
        """
        Transform circular waveform modes into eccentric waveform modes.

        Returns:
            dict: A dictionary containing the transformed eccentric waveform modes.
                  The amplitude and phase of the 'h_l2m2' mode remain unchanged,
                  while other modes are modified according to their respective
                  projected amplitudes and phases using Eq(11) of
                  https://arxiv.org/pdf/2403.15506.
        """
        hNRE = {}
        for mode in self.modelist:
            if mode == 'h_l2m2':
                hNRE[mode] = self.ecc_wfobj.h_transform[mode]
                amplitude = abs(hNRE[mode])
                phase = gwtools.phase(hNRE[mode])
                phase = phase - phase[0] + gwtools.phase(self.cir_wfobj.h_transform[mode])[0]
                hNRE[mode] = amplitude * np.exp(1j * phase)
            else:
                amplitude = self.twist_mode_amplitude(mode)
                phase = self.twist_mode_phase(mode)
                phase = phase - phase[0] + gwtools.phase(self.cir_wfobj.h_transform[mode])[0]
                hNRE[mode] = amplitude * np.exp(1j * phase)
        return hNRE

    def obtain_eccentricHM(self):
        """
        Align eccentric modes in phases and times so that the initial phase matches
        the corresponding circular phase.

        Returns:
            dict: A dictionary containing the aligned eccentric higher-order
                  multipole waveform modes, ensuring initial phase alignment.
        """
        gwhNRE = self.twist_modes()
        gwhNRE = phase_align_dict(gwhNRE)
        gwhNRE = check_pi_rotation(gwhNRE)
        return gwhNRE
