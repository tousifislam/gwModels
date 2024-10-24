#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: eccentric.py
#
#        AUTHOR: Tousif Islam
#       CREATED: 07-03-2024
# LAST MODIFIED: Tue Feb  6 17:58:52 2024
#      REVISION: ---
#==============================================================================
"""
Code to generate eccentric BBH merger waveform using a PN+NR IMR model built by Hinder et al.
This is a python wrapper to work efficiently with the original Mathematica code
Author: Tousif Islam
Date: Dec 22, 2023
"""
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from .utility import *

class EccentricIMR():
    """
    EccentricIMR model class based on the following paper -
    Year: 2017
    Link: http://arxiv.org/abs/1709.02007, Hinder, Kidder and Pfeiffer
    Github repo: https://github.com/ianhinder/EccentricIMR/tree/master
    """
    def __init__(self, wolfram_kernel_path, package_directory):
        self.wolfram_kernel_path = wolfram_kernel_path
        self.package_directory = package_directory
        # start a session
        self.session = WolframLanguageSession(self.wolfram_kernel_path) 
        self._load_mathematica_package()
        
    def _load_mathematica_package(self):
        """
        Load Eccentric IMR package
        """
        # Load the EccentricIMR package
        load_package_code = f"AppendTo[$Path, \"{self.package_directory}\"]; << EccentricIMR`;"
        load_package_expr = wlexpr(load_package_code)
        # Set the $Path explicitly
        self.session.evaluate(load_package_expr)
        
    def generate_waveform(self, params):
        """
        Generate the waveform using mathematica package
        converts the mathematica time/hstrain output into python outputs
        """
        # Generate the code for EccentricIMRWaveform
        param_list = ', '.join([f'"{key}" -> {value}' for key, value in params.items()])
        generate_waveform_code = f'hEcc = EccentricIMRWaveform[<|{param_list}|>, {{0, 10000}}]'
        generate_waveform_expr = wlexpr(generate_waveform_code)
        
        # Generate the waveform
        mathematica_res = self.session.evaluate(generate_waveform_expr)
            
        # Separate time and value axes
        time, value_axis = zip(*mathematica_res)
        # change the value_axis from mathematica Complex to Python complex
        complex_value_python = np.transpose(value_axis)[0] + 1j*np.transpose(value_axis)[1]
        time_value_python = np.array(time)
        # align the peak so that merger is at t=0
        tpeak = self.peak_time(time_value_python, complex_value_python)
        time_value_python = time_value_python - tpeak
        
        return time_value_python, complex_value_python

    def peak_time(self, t, modes):
        """
        Finds the peak time quadratically, using 22 mode
        t : an array of times
        modes : a list/array of waveform mode arrays, OR a single mode.
                Each mode should have type numpy.ndarray
        """
        normSqrVsT = abs(modes)**2
        return get_peak(t, normSqrVsT)[0]

    def plot_waveform(self, time, complex_value_python):
        plt.figure(figsize=(8,4))
        plt.plot(time, np.real(complex_value_python), label='real part')
        plt.plot(time, np.imag(complex_value_python), label='imag part')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('h22')
        plt.show()