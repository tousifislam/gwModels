#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: x_model_hinder.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-02-2024
#    LAST MODIFIED:
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.rcParams['mathtext.fontset'] ='stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral' 
matplotlib.rcParams['axes.linewidth'] = 0.7 #set the value globally
plt.rcParams["figure.figsize"] = (6,5)
plt.rcParams['font.size'] = '14'
plt.rc('text', usetex=True)

import gwsurrogate
sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

from scipy.interpolate import splrep, splev
from scipy import integrate

from .approximated_functions import *
from .xdot_in_x_e import *
from .edot_in_x_e import *
from .adiabatic_x_e import *
from .l_in_x_e import *
from .u_in_x_l_e import *
from .phi_phidot_in_e_u import *
from .r_rdot_in_e_u import *
from .quadrupole import *

from ...utils import mass_ratio_to_symmetric_mass_ratio

class EccentricInspiralHinder2008:
    """
    Employs Eccentric PN model as described in Hinder+ 2008 (https://arxiv.org/pdf/0806.1037)

    x-model
    - 2PN adiabatic evolution of x and e 
    - 3PN conservative dynamics for l, phi, phidot, r, rdot
    """
    def __init__(self, q, x0, e0, t0, phi0=0.0, l0=0.0, dt=0.1, adiabatic_integration_type='euler'):
        """
        Initialize the PNW (Post-Newtonian Waveform) class with default values.
        
        Parameters:
        q (float): Mass ratio.
        x0 (float): Initial value of x.
        e0 (float): Initial eccentricity.
        l0 (float): Initial mean anomaly. default 0.0
        t0 (float): Initial time.
        phi0 (float): Initial phase angle. Default 0.0
        dt (float): Time step for Euler integration. Default 0.1
        """
        self.q = q
        self.nu = mass_ratio_to_symmetric_mass_ratio(q)
        self.x0 = x0
        self.e0 = e0
        self.l0 = l0
        self.t0 = t0
        self.phi0 = phi0
        self.dt = dt
        
        self.adiabatic_integration_type = adiabatic_integration_type
        
        self.pnw_scheme()

    def pnw_scheme(self):
        """
        Run the full sequence of pn waveform calculations.
        Eq 5 to Eq 15 of https://arxiv.org/pdf/0806.1037
        """
        # Step 1: Integrate xdot and edot using Euler method
        include_x_terms = ['0.0', '1.0', '1.5HT', '2.0']
        include_e_terms = ['0.0', '1.0', '1.5HT', '2.0']
        if self.adiabatic_integration_type == 'euler':
            self.x_t, self.e_t, self.t = adiabatic_x_e_evolution(self.e0, self.x0, self.t0, self.dt, self.q, include_x_terms, include_e_terms)
        elif self.adiabatic_integration_type == 'scipy-odeint':
            self.x_t, self.e_t, self.t = integrate_xdot_edot_Euler(self.e0, self.x0, self.t0, self.dt, self.q)
        
        # Step 2: Integrate dl/dt to get l(t)
        self.l_t = integrate_dldt_to_get_l(x=self.x_t, e=self.e_t, t=self.t, q=self.q, l0=self.l0)
        
        # Step 3: Compute u(t)
        self.u_t = compute_u_in_x_e_l(x=self.x_t, e=self.e_t, l=self.l_t, q=self.q)
        
        # Step 4: Compute phidot(t)
        self.phidot_t = compute_phi_dot_in_x_e_u(x=self.x_t, e=self.e_t, u=self.u_t, q=self.q)
        
        # Step 5: Integrate phidot to get phi(t)
        self.phi_t = integrate_phidot_to_get_phi(phi_dot=self.phidot_t, t=self.t, phi_0=self.phi0)
        
        # Step 6: Compute r(t)
        self.r_t = compute_r_in_x_e_u(x=self.x_t, e=self.e_t, u=self.u_t, q=self.q)
        
        # Step 7: Compute rdot(t)
        self.rdot_t = compute_rdot_from_r(t=self.t, r=self.r_t)
        
        # Step 8: Compute the restricted quadrupolar waveform h(t)
        self.h_t = compute_restricted_quadrupolar_waveform(q=self.q, 
                                                           r=self.r_t, 
                                                           rdot=self.rdot_t, 
                                                           phi=self.phi_t, 
                                                           phidot=self.phidot_t)
        
        # Additional: compute mean motion n(t)
        self.n_t = compute_mean_motion_n_from_x_e(x=self.x_t, e=self.e_t, nu=self.nu)
        
        # find time where PN diverges
        self.t_diverge = self.find_pn_divergence_time()
        # time series to keep
        index = np.where(self.t<self.t_diverge)[0][-1]
        self.truncate_arrays_at_index(index)
        
    def truncate_arrays_at_index(self, index):
        """
        Truncate the time-dependent arrays at a specific index.
        
        Parameters:
        index (int): The index at which to truncate all time-dependent arrays.
        """
        self.t = self.t[:index]
        self.x_t = self.x_t[:index]
        self.e_t = self.e_t[:index]
        self.l_t = self.l_t[:index]
        self.u_t = self.u_t[:index]
        self.phi_t = self.phi_t[:index]
        self.phidot_t = self.phidot_t[:index]
        self.r_t = self.r_t[:index]
        self.rdot_t = self.rdot_t[:index]
        self.h_t = self.h_t[:index]
        self.n_t = self.n_t[:index]

        
    def find_pn_divergence_time(self):
        """
        Finds where PN expressions are not reliable anymore and they diverge
        """
        change_threshold = 1.05
        # find where mean anomaly suddenly changes direction
        t1 = self.t[np.where(np.diff(self.l_t)<0)][0]
        # find where mean motion suddenly changes direction
        t2 = self.t[np.where(np.diff(self.n_t)<0)][0]
        # find where x(t) changes too much 
        r = self.x_t[1:]/self.x_t[:-1]
        t3 = self.t[np.where(r>change_threshold)][0]
        # figure out the time after which PN blows up
        t_diverge = min(t1, t2, t3)
        
        return t_diverge
    
    def plot_waveform(self):
        """
        Plot the inspiral only waveform
        """
        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.plot(self.t, self.h_t)
        plt.xlabel('$t$')
        plt.ylabel('$h(t)$')
        plt.subplot(122)
        plt.plot(self.t, self.e_t)
        plt.xlabel('$t$')
        plt.ylabel('$e(t)$')
        plt.show()
        
        
class EccentricIMRHinder2017:
    """
    PN-NR Hybrid waveform model for eccentric non-spinning binaries;
    Developed in https://arxiv.org/pdf/1709.02007
    This is a Python Implementation of the original mathematica package 
    [ https://github.com/ianhinder/EccentricIMR/ ]
    
    With only two exceptions: 
        (i) they use a custom merger ringdown model - we replaced it with NRHybSur3dq8
        (ii) to compute kappa_E and kappa_J, they use interpolation to the infinite sum of 
        Bessel functions; We rather use an analytical approximation as a function of 
        eccentricity;
    """
    def __init__(self, q, x0, e0, t0, l0, phi0, dt=0.1, x_ref=0.11, x_blend=0.12):
        """
        Initialize the IMRHybridWaveform class with binary parameters.

        Input:
        - q : Mass ratio
        - x0, e0 : Initial dimensionless frequency and eccentricity
        - t0, l0 : Initial time and mean anomaly
        - phi0 : Initial phase
        - x_ref : Reference internal dimensionless frequency to build IMR model
                  Default is 0.11
        - x_blend : Reference dimensionless frequency for blending
                    Default is 0.12
        This initialization sets up the inspiral waveform model and extracts
        parameters corresponding to x_ref and x_blend.
        """
        # inputs
        self.q = q
        self.x0 = x0
        self.e0 = e0
        self.t0 = t0
        self.l0 = l0
        self.phi0 = phi0
        self.dt = dt
        self.x_ref = x_ref
        self.x_blend = x_blend

        # Instantiate the inspiral waveform
        self.wf = EccentricInspiralHinder2008(q=self.q, x0=self.x0, e0=self.e0, t0=self.t0, l0=self.l0, phi0=self.phi0, dt=self.dt)

        # Interpolation objects for waveform quantities
        self.spl_x_to_t = splrep(self.wf.x_t, self.wf.t)
        self.spl_t_to_e = splrep(self.wf.t, self.wf.e_t)
        self.spl_t_to_l = splrep(self.wf.t, self.wf.l_t)

        # Extract time, eccentricity, and mean anomaly at x_ref
        self.t_ref = splev([self.x_ref], self.spl_x_to_t)[0]
        self.e_ref = splev([self.t_ref], self.spl_t_to_e)[0]
        self.l_ref = splev([self.t_ref], self.spl_t_to_l)[0]

        # Extract time corresponding to x_blend
        self.t_blend = splev([self.x_blend], self.spl_x_to_t)[0]

        # Time to merger from x_ref
        self.delta_t = self.time_to_merge(q=self.q, e=self.e_ref, l=self.l_ref)
        self.t_peak = self.t_ref + self.delta_t

        # Transform time axis so t_peak = 0
        self.t_transform = self.wf.t - self.t_peak
        self.t_blend = self.t_blend - self.t_peak
        self.t_ref = self.t_ref - self.t_peak   
        
        # Approximate circularization time
        self.t_cir = self.t_transform[-1]
        
        # Time grid for the hybrid inspiral-merger-ringdown waveform
        self.t_imr = np.linspace(self.t_transform[0], 100, 10000)
        
        # obtain cmm waveform
        self.get_cmm_waveform_pieces()
        
        # obtain eccentric inspiral waveform
        self.get_inspiral_waveform_pieces()
        
        # builf hybrid waveform
        self.build_hybrid_waveform()

    def time_to_merge(self, q, e, l):
        """
        Compute the time difference Δt between the initial time and merger time as a function of mass ratio (q), 
        eccentricity (e), and mean anomaly parameter (l)
        """
        return 391.196 + 3.13391 * e - 2492.95 * e**2 + 2.77212 * q - 17.92 * e * q + 8.11842 * q**2 + 76.4944 * e * np.cos(0.626653 + l)

    def transition_function(self, t, t1, t2):
        """
        Compute the transition function for hybridization.
    
        Input:
        - t : float or array-like : Time value(s)
        - t1 : float : Lower time bound for the transition window
        - t2 : float : Upper time bound for the transition window

        Output:
        - T : float or array-like : Value of the function T at time t
        """
        if np.isscalar(t):
            if t <= t1:
                return 0
            elif t1 < t < t2:
                return 1 / (np.exp((t2 - t1) / (t - t1) + (t2 - t1) / (t - t2)) + 1)
            else:
                return 1
        else:
            # For array-like input
            T_values = np.zeros_like(t)
            for i, ti in enumerate(t):
                if ti <= t1:
                    T_values[i] = 0
                elif t1 < ti < t2:
                    T_values[i] = 1 / (np.exp((t2 - t1) / (ti - t1) + (t2 - t1) / (ti - t2)) + 1)
                else:
                    T_values[i] = 1
            return T_values

    def get_cmm_waveform_pieces(self):
        """
        Obtain the circular merger model (CMM) waveform
        """
        # Generate circular waveform from the surrogate
        self.t_cmm, h_cir, _ = sur(self.q, [0, 0, 0], [0, 0, 0], times=self.t_imr, f_low=0)
        self.h_cmm = h_cir[(2, 2)]
        
        # interpolate to imr time grid
        self.h_cmm = gwtools.interpolate_h(self.t_cmm, self.h_cmm, self.t_imr)
        # amplitude
        self.amp_cmm = np.abs(self.h_cmm)
        # phase
        self.phase_cmm = gwtools.phase(self.h_cmm)
        # frequency
        self.omega_cmm = abs(np.gradient(self.phase_cmm, self.t_imr))

        # Set waveform, amplitude, and frequencies to zero in the circular model
        # for times smaller than t=-300
        mask = self.t_cmm <= -300
        self.h_cmm[mask] = np.zeros_like(self.h_cmm[mask], dtype=complex)
        self.amp_cmm[mask] = np.zeros_like(self.amp_cmm[mask], dtype=complex)
        self.omega_cmm[mask] = np.zeros_like(self.omega_cmm[mask], dtype=complex)
        
    def get_inspiral_waveform_pieces(self):
        """
        Obtain the eccentric PN inspiral waveform
        """
        
        # Interpolate inspiral waveform onto IMR time grid
        self.h_pn = np.zeros(len(self.t_imr), dtype=complex)
        self.h_pn[self.t_imr < self.t_cir] = gwtools.interpolate_h(self.t_transform, self.wf.h_t, self.t_imr[self.t_imr < self.t_cir])

        # Amplitude hybridization
        self.amp_pn = np.abs(self.h_pn)
        
        # Frequency hybridization
        self.phase_pn = gwtools.phase(self.h_pn)
        self.omega_pn = abs(np.gradient(self.phase_pn, self.t_imr))
        
    def build_hybrid_waveform(self):
        """
        Build the hybrid inspiral-merger-ringdown (IMR) waveform using eccentric PN inspiral and
        circular merger model
        """
        
        # Compute transition function
        self.alpha_t = self.transition_function(t=self.t_imr, t1=self.t_blend, t2=self.t_cir)

        # Amplitude hybridization
        self.amp_imr = (1 - self.alpha_t) * self.amp_pn + self.alpha_t * self.amp_cmm

        # Frequency hybridization
        self.omega_imr = (1 - self.alpha_t) * self.omega_pn + self.alpha_t * self.omega_cmm

        # Compute phase for the hybrid waveform
        try:
            self.phase_imr = integrate.cumtrapz(self.omega_imr, self.t_imr, initial=0)
        except:
            self.phase_imr = integrate.cumulative_trapezoid(self.omega_imr, self.t_imr, initial=0)
            
        # Hybrid waveform
        self.h_imr = self.amp_imr * np.exp(1j * self.phase_imr)
        
        # black magic to ensure correct phase evolution
        self.h_imr = - self.h_imr
        
    def plot_waveform(self):
        """
        Plot the IMR waveform and the inspiral only waveform
        """
        plt.figure(figsize=(10,3))
        plt.plot(self.t_imr, self.h_imr, label='IMR')
        plt.plot(self.t_imr, self.h_pn, '--', label='Inspiral')
        plt.legend(fontsize=12)
        plt.xlabel('$t$')
        plt.ylabel('$h(t)$')
        plt.tight_layout()
        plt.show()