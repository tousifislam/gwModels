#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: utility.py
#    [ This piece of code is a modification of a similar code in gw_remnant package ] 
# 
#
#        AUTHOR: Tousif Islam
#       CREATED: 07-02-2024
# LAST MODIFIED: Tue Feb  6 17:58:52 2024
#      REVISION: ---
#==============================================================================

import numpy as np
import gwtools
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def get_peak(t, func):
    """
    Finds the peak time of a function quadratically
    Fits the function to a quadratic over the 5 points closest to the argmax func.
    t : an array of times
    func : array of function values
    Returns: tpeak, fpeak
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


def get_frequency(t, h):
    """
    Computes orbital frequency of a given gravitational wave time-series
    t: time array
    h: waveform mode array
    """
    return np.gradient(gwtools.phase(h), t)


def check_pi_rotation(h_dict):
    """
    Checks whether a pi rotation is required in waveform mode data
    h_dict: dictionary of gravitational waves modes. Keys should be "h_l2m2" and so on
    """
    for mode in h_dict.keys():
        # do nothing for the 22 mode
        if mode == 'h_l2m2':
            pass
        # check whether you need to perform a pi rotation based on the phase 
        # of the 21 mode
        elif mode == 'h_l2m1':
            phi = gwtools.phase(h_dict['h_l2m1'])
            # decide whether to perform a physical \pi rotation
            if phi[0]>0:
                pi_rot_factor=-1
            else:
                pi_rot_factor=1
            m = 1
            # rotate 21 mode
            h_dict['h_l2m1'] = ((pi_rot_factor)**m) * h_dict['h_l2m1']
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
    time-domain error via Eq 21 https://arxiv.org/pdf/1701.00550.pdf 
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
    align waveform dict to have proper phases
    """
    phi=np.unwrap(np.angle(hdict['h_l2m2']))
    # Added to enforce correct relative phasing of higher modes (see "else" block)
    z_rot = phi[0]/2.0
    phi=phi-phi[0]
    hdict['h_l2m2']=abs(hdict['h_l2m2'])*np.exp(1j*phi)

    for mode in hdict.keys():
        if mode=='h_l2m2':
            pass
        else:
            phi=np.unwrap(np.angle(hdict[mode]))
            m = float(mode.rsplit("m")[1])
            # modified to enforce correct relative phasing
            hdict[mode]=abs(hdict[mode])*np.exp(1j*(phi-z_rot*m)) # phase -> phase + z_rot*m
    return hdict

class AlignWFData:
    """
    Class to align a waveform such that peak is at t=0 and the initial phase is zero;
    It also cast the waveform in a different time-grid
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
        cast the waveform onto a new time grid;
        this step should be done after the time alignement
        """
        h_transform = {}
        for mode in self.h_input.keys():
            h_transform[mode] = gwtools.gwtools.interpolate_h(self.t_transform, self.h_input[mode], self.t_common)
        return h_transform
    
    def _find_peak_time(self):
        """
        find the time corresponding to the peak of the (2,2) mode
        """
        return np.real(get_peak(self.t_input, abs(self.h_input['h_l2m2']))[0])
        
    def _align_time(self):
        """
        align the waveform such that the peak of the (2,2) mode is at t=0
        """
        return self.t_input - self.t_peak
    
    def _find_offset_orb_phase(self):
        """
        find the phase rotation required to make the initial (2,2) mode phase to be zero
        """
        phi = gwtools.phase(self.h_transform['h_l2m2'])
        return phi[0]/2.0
    
    def _align_phase(self):
        """
        enforce correct relative phasing such that initial (2,2) mode phase is zero
        """
        for mode in self.h_transform.keys():
            phi = gwtools.phase(self.h_transform[mode])
            m = float(mode.rsplit("m")[-1])
            # phase -> phase + z_rot*m
            self.h_transform[mode] = abs(self.h_transform[mode] ) * np.exp(1j*(phi-self.z_rot*m))