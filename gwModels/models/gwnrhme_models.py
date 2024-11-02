#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwnrhme_models.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-03-2024
#    LAST MODIFIED: Tue Feb  6 17:58:52 2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import sys, os
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import matplotlib.pyplot as plt
import numpy as np

from .eccentricimr_wolfram import EccentricIMR
from .circular_models import genNRHybSur3dq8, genBHPTNRSur1dq1e4
from .lal_models import generate_IMRPhenomTHM
from ..core.gwnrhme import NRHME
from ..utils import *

class IMRHME(genNRHybSur3dq8, genBHPTNRSur1dq1e4):
    """
    Class to generate eccentric non-spinning higher order spherical harmonics using
    a quadrupolar eccentric waveform model and a multi-modal circular model.
    """
    def __init__(self, circular_model, eccentric_model, **kwargs):
        """
        Attributes:
        
        circular_model (str): Name of the multi-modal circular model 
                        (e.g., 'NRHybSur3dq8', 'BHPTNRSur1dq1e4', 'IMRPhenomTHM').
        eccentric_model (str): Name of the quadrupolar eccentric model 
                        (e.g., 'EccentricIMR').
        kwargs (dict): Optional keyword arguments including:
            - wolfram_kernel_path (str): Absolute path for the Mathematica kernel.
            - package_directory (str): Absolute path for the EccentricIMR package.
            - model_obj (object): A model object for BHPTNRSur1dq1e4 model.
        """

        # read all the options
        self.kwargs = kwargs

        # circular model
        self.circular_model = circular_model
        if self.circular_model not in ['NRHybSur3dq8', 'BHPTNRSur1dq1e4', 'IMRPhenomTHM']:
            raise ValueError(f"Unrecognized circular model '{self.circular_model}'. Supported models: 'NRHybSur3dq8', 'BHPTNRSur1dq1e4', 'IMRPhenomTHM'.")
        # special treatment for BHPTNRSurroagte
        if self.circular_model == 'BHPTNRSur1dq1e4':
            self.model_obj = kwargs.get("model_obj")
            if self.model_obj is None:
                raise ValueError("... a model object for BHPTNRSur1dq1e4 must be provided!")

        # eccentric model
        self.eccentric_model = eccentric_model
        # raise error if it is not EccentricIMR
        if self.eccentric_model != 'EccentricIMR':
            raise ValueError("... currently gwModels only supports 'EccentricIMR' as its eccentric base model!")
        else:
            # check if a kernel path is provided
            self.wolfram_kernel_path = kwargs.get("wolfram_kernel_path")
            if self.wolfram_kernel_path is None:
                raise ValueError("... a path for the Wolfram Kernel must be provided!")
            
            # check if package directory is provided
            self.package_directory = kwargs.get("package_directory")
            if self.package_directory is None:
                raise ValueError("... a path for the EccentricIMR package directory must be provided!")
                
            # instantiate the EccentricIMR class
            # it may take some time
            self.wf = EccentricIMR(self.wolfram_kernel_path, self.package_directory)
        

    def generate_waveform(self, params): 
        """
        Generates a combined eccentric waveform based on the specified parameters.

        Parameters:
            params (dict): Dictionary containing waveform parameters with the following keys:
                - q (float): Mass ratio of the binary system.
                - e0 (float): Initial eccentricity at x0.
                - l0 (float): Initial mean anomaly at x0.
                - x0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: A tuple containing:
                - tNRE (numpy.ndarray): Time array for the generated eccentric waveform.
                - hNRE (numpy.ndarray): Generated eccentric waveform data.
        """

        # Check for "s1z" and "s2z" keys and ensure they corresponds to the non-spinning case
        for key in ["s1z", "s2z"]:
            if key in params:
                # If the value is not zero, print a message and set it to zero
                if params[key] != 0:
                    print(f"Warning: Parameter '{key}' should be zero for non-spinning systems. Setting '{key}' to zero.")
                    params[key] = 0
            else:
                # If the key doesn't exist, add it with a value of zero
                params[key] = 0
                print(f"{key} not found in params. Setting {key} to zero.")

        
        # generate eccentric waveform
        if self.eccentric_model == 'EccentricIMR':
            tIMR, hIMR = self.wf.generate_waveform(params)
            params["fIMR"] = self._obtain_circular_flow(tIMR, hIMR)

        # generate circular waveform
        if self.circular_model == 'NRHybSur3dq8':
            # generate surrogate cicular waveform
            t_cir, h_cir = self.generate_NRHybSur3dq8(params)

        elif self.circular_model == 'BHPTNRSur1dq1e4':
            # generate surrogate cicular waveform
            t_cir, h_cir = self.generate_BHPTNRSur1dq1e4(self.model_obj, params)

        elif self.circular_model == 'IMRPhenomTHM':
            # generate IMRPhenomTHM model
            t_cir, h_cir = generate_IMRPhenomTHM(mass_ratio=params["q"],
                                                 Momega0OverM=params["fIMR"])
        else:
            raise ValueError("Model not implemented!")
            
        # use gwNRHME to obtain multi-modal eccentric waveform
        tNRE, hNRE = self._apply_gwNRHME(t_ecc=tIMR, 
                                         h_ecc_dict={'h_l2m2': hIMR},
                                         t_cir=t_cir, 
                                         h_cir_dict=h_cir)
        return tNRE, hNRE

    
    def _obtain_circular_flow(self, tIMR, hIMR):
        """
        Computes the starting frequency of the eccentric waveform.

        This frequency is passed to the circular waveform as f_low.

        Parameters:
            tIMR (numpy.ndarray): Time array for the generated eccentric waveform.
            hIMR (numpy.ndarray): Eccentric waveform data.

        Returns:
            float: The starting frequency of the eccentric waveform.
        """
        fIMR = 0.9 * abs(get_frequency(tIMR, hIMR)[0])/(np.pi)
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
            tuple: A tuple containing:
                - tNRE (numpy.ndarray): Time array for the converted eccentric waveform.
                - hNRE (numpy.ndarray): Converted eccentric waveform data.
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
    EccentricIMR and NRHybSur3dq8 model.
    """
    def __init__(self, eccentric_model='EccentricIMR', **kwargs):
        """
        Parameters:
        
        eccentric_model (str): Name of the eccentric quadrupolar model.
        kwargs (dict): Optional keyword arguments including:
            - wolfram_kernel_path (str): Absolute path for the Mathematica kernel.
            - package_directory (str): Absolute path for the EccentricIMR package.
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
                - e0 (float): Initial eccentricity at x0.
                - l0 (float): Initial mean anomaly at x0.
                - x0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: A tuple containing:
                - tNRE (numpy.ndarray): Time array for the generated eccentric waveform.
                - hNRE (numpy.ndarray): Generated eccentric waveform data.
        """
        # waveform
        tNRE, hNRE = self.wf_obj.generate_waveform(params)
        return tNRE, hNRE


class BHPTNRSur1dq1e4_gwNRHME():
    """
    Class to generate eccentric higher order spherical harmonics using
    EccentricIMR and BHPTNRSur1dq1e4 model
    """
    def __init__(self, eccentric_model, **kwargs):
        """
        Parameters:
        
        eccentric_model (str): Name of the eccentric quadrupolar model.
        kwargs (dict): Optional keyword arguments including:
            - wolfram_kernel_path (str): Absolute path for the Mathematica kernel.
            - package_directory (str): Absolute path for the EccentricIMR package.
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
                - e0 (float): Initial eccentricity at x0.
                - l0 (float): Initial mean anomaly at x0.
                - x0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: A tuple containing:
                - tNRE (numpy.ndarray): Time array for the generated eccentric waveform.
                - hNRE (numpy.ndarray): Generated eccentric waveform data.
        """
        # waveform
        tNRE, hNRE = self.wf_obj.generate_waveform(params)
        return tNRE, hNRE


class IMRPhenomTHM_gwNRHME():
    """
    Class to generate eccentric higher order spherical harmonics using
    EccentricIMR and IMRPhenomTHM model
    """
    def __init__(self, eccentric_model, **kwargs):
        """
        Parameters:
        eccentric_model (str): Name of the eccentric quadrupolar model.
        kwargs (dict): Optional keyword arguments including:
            - wolfram_kernel_path (str): Absolute path for the Mathematica kernel.
            - package_directory (str): Absolute path for the EccentricIMR package.
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
                - e0 (float): Initial eccentricity at x0.
                - l0 (float): Initial mean anomaly at x0.
                - x0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: A tuple containing:
                - tNRE (numpy.ndarray): Time array for the generated eccentric waveform.
                - hNRE (numpy.ndarray): Generated eccentric waveform data.
        """
        # waveform
        tNRE, hNRE = self.wf_obj.generate_waveform(params)
        return tNRE, hNRE