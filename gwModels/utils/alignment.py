#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: alignment.py
#    [ This piece of code is a modification of a similar code in gw_remnant package 
#      written by Tousif Islam ] 
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

def get_peak(t, func):
    """"
    Finds the peak time of a function using spline.
    
    Fits the provided function over the five points
    closest to the argmax of the function to accurately find the peak.

    Parameters:
    t (np.ndarray): An array of times.
    func (np.ndarray): An array of function values.

    Returns:
    tuple: A tuple containing:
        - tpeak (float): The time at which the peak occurs.
        - fpeak (float): The function value at the peak.
    """
    # Use a 4th degree spline for interpolation, so that the roots of its derivative can be found easily.
    spl = spline(t, func, k=4)
    # find the critical points
    cr_pts = spl.derivative().roots()
    # also check the endpoints of the interval
    cr_pts = np.append(cr_pts, (t[0], func[-1]))  
    # critial values
    cr_vals = spl(cr_pts)
    # we only care about the maximas
    max_index = np.argmax(cr_vals)
    return cr_pts[max_index], cr_vals[max_index]


def check_pi_rotation(h_dict):
    """
    Checks whether a pi rotation is required in waveform mode data.

    This function determines the phase factor required for each mode based on 
    the phase of the `h_l2m1` mode, applying a pi rotation for odd m modes 
    if necessary.

    Parameters:
    h_dict (dict): Dictionary of gravitational wave modes. Keys should be "h_l2m2", "h_l2m1", etc.

    Returns:
    dict: Updated dictionary of gravitational wave modes after potential rotation.
    """
    phi = gwtools.phase(h_dict['h_l2m1'])
    # decide whether to perform a physical \pi rotation
    if phi[0]>0:
        pi_rot_factor=-1
    else:
        pi_rot_factor=1
        
    for mode in h_dict.keys():
        # do nothing for the 22 mode
        if mode == 'h_l2m2':
            pass
        # for other modes, rotate odd m modes only
        else:
            m = float(mode.rsplit("m")[1])
            if m%2 ==0:
                pass
            else:
                h_dict[mode] =((pi_rot_factor)**m) * h_dict[mode]
    return h_dict
             
    
def mathcalE_error(h1, h2):
    """
    Computes the time-domain error between two waveforms.

    This function calculates the error according to Equation 21 from the paper 
    (https://arxiv.org/pdf/1701.00550.pdf) by normalizing the difference between 
    the two waveforms.

    Parameters:
    h1 (np.ndarray): Reference waveform in the time domain.
    h2 (np.ndarray): Comparison waveform in the time domain.

    Returns:
    np.ndarray: Normalized error for each time sample.
    """
    n1Sqr = np.sum(abs(h1**2))
    n2Sqr = np.sum(abs(h2**2))

    dots = np.array([(h1[i]*(h2[i].conjugate())) for i in range(len(h1))])
    sdot = np.real(np.sum(dots))
    # Assume h1 is the reference waveform and normalize by its magnitude
    normed_errs = ((n1Sqr + n2Sqr) - 2*sdot)/(2*n1Sqr)
    summed_err = np.sum(normed_errs)
    return normed_errs

def phase_align_dict(hdict):
    """
    Aligns a waveform dictionary to ensure proper phases.

    This function modifies the phases of the waveform modes to ensure that the 
    initial phase of the (2,2) mode is zero and that the relative phases 
    of higher modes are consistent.

    Parameters:
    hdict (dict): Dictionary of gravitational wave modes. Keys should include 
                  'h_l2m2', 'h_l3m3', etc.

    Returns:
    dict: A new dictionary with aligned phases for each mode.
    """
    hdict_out = {}
    phi=np.unwrap(np.angle(hdict['h_l2m2']))
    # Added to enforce correct relative phasing of higher modes (see "else" block)
    z_rot = phi[0]/2.0
    phi=phi-phi[0]
    hdict_out['h_l2m2']=abs(hdict['h_l2m2'])*np.exp(1j*phi)

    for mode in hdict.keys():
        if mode=='h_l2m2':
            pass
        else:
            phi=np.unwrap(np.angle(hdict[mode]))
            m = float(mode.rsplit("m")[1])
            # modified to enforce correct relative phasing
            hdict_out[mode]=abs(hdict[mode])*np.exp(1j*(phi-z_rot*m)) # phase -> phase + z_rot*m
    return hdict_out

class AlignWFData:
    """
    Class to align a waveform such that the peak is at t=0 and the initial phase is zero.

    This class also casts the waveform onto a different time grid if specified.

    Attributes:
    t_input (np.ndarray): Input time array.
    h_input (dict): Input waveform dictionary with keys as spherical harmonics modes.
    t_common (np.ndarray or None): Target time grid on which waveform data should be cast.

    Methods:
    _find_peak_time(): Finds the time of the peak for the (2,2) mode.
    _align_time(): Aligns the waveform to ensure the peak is at t=0.
    _find_offset_orb_phase(): Finds the phase rotation to set the initial (2,2) mode phase to zero.
    _align_phase(): Aligns the phases of the waveform modes.
    _cast_waveform_on_timegrid(): Casts the waveform onto the specified time grid.
    """
    def __init__(self, t_input, h_input, t_common=None):
        """
        t_input: float
        h_input: dictionary whose keys are sphereical harmonics modes
                 dictionary keys should be 'h_l2m2', 'h_l3m3' and so on
                 dictionary must contain 'h_l2m2'
        t_common: target time grid on which waveform data should be cast
                 default: None
        """
        self.t_input = t_input
        self.h_input = h_input
        self.t_common = t_common
        
        # time alignment
        self.t_peak = self._find_peak_time()
        self.t_transform = self._align_time()
        
        # if t_common is given, cast the waveform in a common time grid
        if self.t_common is not None:
            self.h_transform = self._cast_waveform_on_timegrid()
            self.t_transform = self.t_common
        else:
            self.h_transform = self.h_input
            
        # phase alignment
        self.z_rot = self._find_offset_orb_phase()
        self._align_phase()
        
    def _cast_waveform_on_timegrid(self):
        """
        Casts the waveform onto a new time grid.
        This method should be called after the time alignment has been completed.

        Returns:
        dict: A dictionary of waveforms aligned on the new time grid.
        """
        h_transform = {}
        for mode in self.h_input.keys():
            h_transform[mode] = gwtools.gwtools.interpolate_h(self.t_transform, self.h_input[mode], self.t_common)
        return h_transform
    
    def _find_peak_time(self):
        """
        Finds the time corresponding to the peak of the (2,2) mode.

        Returns:
        float: The time at which the peak of the (2,2) mode occurs.
        """
        return np.real(get_peak(self.t_input, abs(self.h_input['h_l2m2']))[0])
        
    def _align_time(self):
        """
        Aligns the waveform such that the peak of the (2,2) mode is at t=0.

        Returns:
        np.ndarray: The transformed time array with the peak aligned to zero.
        """
        return self.t_input - self.t_peak
    
    def _find_offset_orb_phase(self):
        """
        Finds the phase rotation required to make the initial (2,2) mode phase zero.

        Returns:
        float: The phase rotation to apply.
        """
        phi = gwtools.phase(self.h_transform['h_l2m2'])
        return phi[0]/2.0
    
    def _align_phase(self):
        """
        Enforces correct relative phasing such that the initial (2,2) mode phase is zero.
        This modifies each mode in the waveform to ensure consistent phase alignment.
        """
        for mode in self.h_transform.keys():
            phi = gwtools.phase(self.h_transform[mode])
            m = float(mode.rsplit("m")[-1])
            # phase -> phase + z_rot*m
            self.h_transform[mode] = abs(self.h_transform[mode] ) * np.exp(1j*(phi-self.z_rot*m))