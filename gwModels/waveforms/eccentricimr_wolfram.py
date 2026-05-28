#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: eccentricimr_wolfram.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-03-2024
#    LAST MODIFIED: Tue Feb  6 17:58:52 2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

try:
    from wolframclient.evaluation import WolframLanguageSession
    from wolframclient.language import wl, wlexpr
except:
    print("ModuleNotFound: 'wolframclient' - EccentricIMR wrapper for Mathematica would not work!")
    
import matplotlib.pyplot as plt
import numpy as np
from ..utils import *

class EccentricIMR():
    """
    Class for generating gravitational waveforms using the EccentricIMR model.

    This model is based on the following work:
    References:
        - Authors: Hinder, Kidder, and Pfeiffer
        - Year: 2017
        - Link: http://arxiv.org/abs/1709.02007
        - GitHub repository: https://github.com/ianhinder/EccentricIMR/tree/master
    """
    def __init__(self, wolfram_kernel_path, package_directory):
        """
        Initializes the EccentricIMR class.

        Parameters:
            wolfram_kernel_path (str): Path to the Wolfram kernel executable.
            package_directory (str): Directory where the EccentricIMR Mathematica package is located.
        """
        self.wolfram_kernel_path = wolfram_kernel_path
        self.package_directory = package_directory
        # start a session
        self.session = WolframLanguageSession(self.wolfram_kernel_path) 
        self._load_mathematica_package()
        
    def _load_mathematica_package(self):
        """
        Loads the EccentricIMR Mathematica package into the current Wolfram session.

        This method appends the package directory to the Wolfram $Path and
        loads the EccentricIMR package, making its functions available for use.
        """
        # Load the EccentricIMR package
        load_package_code = f"AppendTo[$Path, \"{self.package_directory}\"]; << EccentricIMR`;"
        load_package_expr = wlexpr(load_package_code)
        # Set the $Path explicitly
        self.session.evaluate(load_package_expr)
        
    def generate_waveform(self, params):
        """
        Generates the gravitational waveform using the EccentricIMR Mathematica package.

        This method converts the Mathematica output for time and strain into
        NumPy arrays for further analysis.

        Parameters:
            params (dict): A dictionary of parameters required by the EccentricIMRWaveform function.
                Expected keys include physical parameters that define the binary system.

        Returns:
            tuple: A tuple containing:
                - time (np.ndarray): An array of time values corresponding to the waveform.
                - complex_value_python (np.ndarray): A NumPy array of complex strain values (h22 mode).
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
        Finds the peak time of the waveform using the 22 mode.

        This method calculates the time at which the waveform strain reaches its maximum,
        based on the norm of the modes.

        Parameters:
            t (np.ndarray): An array of time values.
            modes (np.ndarray): An array of waveform mode arrays, or a single mode array.
                Each mode should be of type numpy.ndarray.

        Returns:
            float: The time at which the peak occurs.
        """
        normSqrVsT = abs(modes)**2
        return get_peak(t, normSqrVsT)[0]

    def plot_waveform(self, time, complex_value_python):
        """
        Plots the EccentricIMR waveform.

        This method visualizes the real and imaginary parts of the waveform.

        Parameters:
            time (np.ndarray): An array of time values.
            complex_value_python (np.ndarray): A NumPy array of complex strain values (h22 mode).
        """
        plt.figure(figsize=(8,4))
        plt.plot(time, np.real(complex_value_python), label='real part')
        plt.plot(time, np.imag(complex_value_python), label='imag part')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('h22')
        plt.show()