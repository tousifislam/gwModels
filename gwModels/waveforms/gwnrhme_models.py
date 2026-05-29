#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwnrhme_models.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-03-2024
#    LAST MODIFIED: 05-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import logging
import warnings
import numpy as np

from .circular_models import genNRHybSur3dq8, genBHPTNRSur1dq1e4
from .seob import genSEOBNRv5EHM
from .lal_models import generate_IMRPhenomTHM
from ..frameworks.gwnrhme import NRHME
from ..utils.features import get_frequency

logger = logging.getLogger(__name__)


class IMRHME(genNRHybSur3dq8, genBHPTNRSur1dq1e4):
    """
    Class to generate eccentric non-spinning higher order spherical harmonics using
    a quadrupolar eccentric waveform model and a multi-modal circular model.
    """
    def __init__(self, circular_model, eccentric_model='SEOBNRv5EHM', **kwargs):
        """
        Parameters:
            circular_model (str): Name of the multi-modal circular model
                            (e.g., 'NRHybSur3dq8', 'BHPTNRSur1dq1e4', 'IMRPhenomTHM').
            eccentric_model (str): Name of the quadrupolar eccentric model
                            (e.g., 'SEOBNRv5EHM').
            kwargs (dict): Optional keyword arguments including:
                - model_obj (object): A model object for BHPTNRSur1dq1e4 model.
        """

        self.kwargs = kwargs

        # circular model
        self.circular_model = circular_model
        if self.circular_model not in ['NRHybSur3dq8', 'BHPTNRSur1dq1e4', 'IMRPhenomTHM']:
            raise ValueError(f"Unrecognized circular model '{self.circular_model}'. "
                             "Supported models: 'NRHybSur3dq8', 'BHPTNRSur1dq1e4', 'IMRPhenomTHM'.")
        if self.circular_model == 'BHPTNRSur1dq1e4':
            self.model_obj = kwargs.get("model_obj")
            if self.model_obj is None:
                raise ValueError("A model object for BHPTNRSur1dq1e4 must be provided!")

        # eccentric model
        self.eccentric_model = eccentric_model
        if self.eccentric_model == 'SEOBNRv5EHM':
            self.wf = genSEOBNRv5EHM()
        else:
            raise ValueError(f"Unrecognized eccentric model '{self.eccentric_model}'. "
                             "Supported models: 'SEOBNRv5EHM'.")


    def generate_waveform(self, params):
        """
        Generates a combined eccentric waveform based on the specified parameters.

        Parameters:
            params (dict): Dictionary containing waveform parameters with the following keys:
                - q (float): Mass ratio of the binary system.
                - e0 (float): Initial eccentricity at omega0.
                - l0 (float): Initial mean anomaly at omega0.
                - omega0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: (tNRE, hNRE) — time array and generated eccentric waveform data.
        """

        for key in ["s1z", "s2z"]:
            if key in params:
                if params[key] != 0:
                    warnings.warn(f"Parameter '{key}' should be zero for non-spinning systems. "
                                  f"Setting '{key}' to zero.")
                    params[key] = 0
            else:
                params[key] = 0
                logger.info("%s not found in params. Setting %s to zero.", key, key)

        # generate eccentric waveform
        if self.eccentric_model == 'SEOBNRv5EHM':
            tIMR, hIMR_dict = self.wf.generate_SEOBNRv5EHM(params)
            hIMR = hIMR_dict['h_l2m2']

        # generate circular waveform
        if self.circular_model == 'NRHybSur3dq8':
            t_cir, h_cir = self.generate_NRHybSur3dq8(params)

        elif self.circular_model == 'BHPTNRSur1dq1e4':
            t_cir, h_cir = self.generate_BHPTNRSur1dq1e4(self.model_obj, params)

        elif self.circular_model == 'IMRPhenomTHM':
            fIMR = self._obtain_circular_flow(tIMR, hIMR)
            t_cir, h_cir = generate_IMRPhenomTHM(mass_ratio=params["q"],
                                                 Momega0OverM=fIMR)
        else:
            raise ValueError(f"Circular model '{self.circular_model}' not implemented!")

        # use gwNRHME to obtain multi-modal eccentric waveform
        tNRE, hNRE = self._apply_gwNRHME(t_ecc=tIMR,
                                         h_ecc_dict={'h_l2m2': hIMR},
                                         t_cir=t_cir,
                                         h_cir_dict=h_cir)
        return tNRE, hNRE


    def _obtain_circular_flow(self, tIMR, hIMR):
        """
        Computes the starting frequency of the eccentric waveform.

        Parameters:
            tIMR (numpy.ndarray): Time array for the generated eccentric waveform.
            hIMR (numpy.ndarray): Eccentric waveform data.

        Returns:
            float: The starting frequency of the eccentric waveform.
        """
        fIMR = 0.9 * abs(get_frequency(tIMR, hIMR)[0]) / (np.pi)
        return fIMR


    def _apply_gwNRHME(self, t_ecc, h_ecc_dict, t_cir, h_cir_dict):
        """
        Converts circular higher modes to eccentric modes.

        Parameters:
            t_ecc (numpy.ndarray): Time array for the eccentric waveform.
            h_ecc_dict (dict): Dictionary of eccentric waveform data.
            t_cir (numpy.ndarray): Time array for the circular waveform.
            h_cir_dict (dict): Dictionary of circular waveform data.

        Returns:
            tuple: (tNRE, hNRE) — time array and converted eccentric waveform data.
        """
        hNRE_obj = NRHME(t_ecc, h_ecc_dict,
                         t_cir, h_cir_dict,
                         get_orbfreq_mod_from_amp_mod=False)

        tNRE = hNRE_obj.t_common
        hNRE = hNRE_obj.hNRE
        return tNRE, hNRE


class NRHybSur3dq8_gwNRHME():
    """
    Class to generate eccentric higher order spherical harmonics using
    SEOBNRv5EHM and NRHybSur3dq8 model.
    """
    def __init__(self, eccentric_model='SEOBNRv5EHM', **kwargs):
        """
        Parameters:
            eccentric_model (str): Name of the eccentric quadrupolar model.
            kwargs (dict): Optional keyword arguments.
        """

        self.wf_obj = IMRHME(circular_model='NRHybSur3dq8',
                             eccentric_model=eccentric_model,
                             **kwargs)

    def generate_waveform(self, params):
        """
        Generates a combined eccentric waveform.

        Parameters:
            params (dict): Dictionary containing waveform parameters with the following keys:
                - q (float): Mass ratio of the binary system.
                - e0 (float): Initial eccentricity at omega0.
                - l0 (float): Initial mean anomaly at omega0.
                - omega0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: (tNRE, hNRE) — time array and generated eccentric waveform data.
        """
        tNRE, hNRE = self.wf_obj.generate_waveform(params)
        return tNRE, hNRE


class BHPTNRSur1dq1e4_gwNRHME():
    """
    Class to generate eccentric higher order spherical harmonics using
    SEOBNRv5EHM and BHPTNRSur1dq1e4 model.
    """
    def __init__(self, eccentric_model='SEOBNRv5EHM', **kwargs):
        """
        Parameters:
            eccentric_model (str): Name of the eccentric quadrupolar model.
            kwargs (dict): Optional keyword arguments including:
                - model_obj (object): A model object for BHPTNRSur1dq1e4 model.
        """

        self.wf_obj = IMRHME(circular_model='BHPTNRSur1dq1e4',
                             eccentric_model=eccentric_model,
                             **kwargs)

    def generate_waveform(self, params):
        """
        Generates a combined eccentric waveform.

        Parameters:
            params (dict): Dictionary containing waveform parameters with the following keys:
                - q (float): Mass ratio of the binary system.
                - e0 (float): Initial eccentricity at omega0.
                - l0 (float): Initial mean anomaly at omega0.
                - omega0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: (tNRE, hNRE) — time array and generated eccentric waveform data.
        """
        tNRE, hNRE = self.wf_obj.generate_waveform(params)
        return tNRE, hNRE


class IMRPhenomTHM_gwNRHME():
    """
    Class to generate eccentric higher order spherical harmonics using
    SEOBNRv5EHM and IMRPhenomTHM model.
    """
    def __init__(self, eccentric_model='SEOBNRv5EHM', **kwargs):
        """
        Parameters:
            eccentric_model (str): Name of the eccentric quadrupolar model.
            kwargs (dict): Optional keyword arguments.
        """

        self.wf_obj = IMRHME(circular_model='IMRPhenomTHM',
                             eccentric_model=eccentric_model,
                             **kwargs)

    def generate_waveform(self, params):
        """
        Generates a combined eccentric waveform.

        Parameters:
            params (dict): Dictionary containing waveform parameters with the following keys:
                - q (float): Mass ratio of the binary system.
                - e0 (float): Initial eccentricity at omega0.
                - l0 (float): Initial mean anomaly at omega0.
                - omega0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: (tNRE, hNRE) — time array and generated eccentric waveform data.
        """
        tNRE, hNRE = self.wf_obj.generate_waveform(params)
        return tNRE, hNRE
