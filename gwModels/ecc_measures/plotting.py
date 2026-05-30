#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: plotting.py
#    Plotting functions for eccentricity measure classes.
#
#       AUTHOR: Tousif Islam
#       CREATED: 05-30-2026
#       REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import matplotlib.pyplot as plt


# =========================================================================
# Plots for ComputeEccentricityFromModulations / ComputeEccentricity
# =========================================================================

def plot_xi(obj, figsize=(8, 5)):
    """Plot the modulation parameter xi."""
    plt.figure(figsize=figsize)
    plt.plot(obj.time_xi, obj.modulations, color='C0', markersize=10, alpha=0.7)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('${\\xi}$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()


def plot_xi_with_peaks(obj, figsize=(8, 5)):
    """Plot xi with periastron and apastron peaks."""
    plt.figure(figsize=figsize)
    plt.plot(obj.time_xi, obj.modulations, color='C0', markersize=10, alpha=0.7)
    plt.plot(obj.t_maximas, obj.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Maximas')
    plt.plot(obj.t_minimas, -obj.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Minimas')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('${\\xi}$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.show()


def plot_maximas_fit(obj, figsize=(8, 5)):
    """Plot upper envelope fits."""
    plt.figure(figsize=figsize)
    plt.plot(obj.t_maximas, obj.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Numerical')
    plt.plot(obj.time_xi, obj.xi_upper, label='PN fit with $e_0=%.3f$' % obj.popt_maximas[0], c='C0')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('$|{\\xi}_{\\rm upper}^{\\rm env}|$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.show()


def plot_minimas_fit(obj, figsize=(8, 5)):
    """Plot lower envelope fits."""
    plt.figure(figsize=figsize)
    plt.plot(obj.t_minimas, obj.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Numerical')
    plt.plot(obj.time_xi, obj.xi_lower, label='PN fit with $e_0=%.3f$' % obj.popt_minimas[0], c='C1')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('$|{\\xi}_{\\rm lower}^{\\rm env}|$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.show()


def plot_fit_errors(obj, figsize=(8, 5)):
    """Plot errors in upper and lower envelope fits."""
    plt.figure(figsize=figsize)
    plt.plot(obj.t_maximas, obj.y_maximas - obj.xi_upper_interp(obj.t_maximas), 'o', markersize=4, label='Upper envelop')
    plt.plot(obj.t_minimas, obj.y_minimas - obj.xi_lower_interp(obj.t_minimas), 's', markersize=4, label='Lower envelop')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Fit errors', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.show()


def plot_xi_with_peaks_and_fits(obj, figsize=(8, 5)):
    """Plot xi with upper and lower envelope fits."""
    plt.figure(figsize=figsize)
    plt.plot(obj.time_xi, obj.modulations, color='C0', markersize=10, alpha=0.7)
    plt.plot(obj.t_maximas, obj.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='Maximas')
    plt.plot(obj.t_minimas, -obj.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='Minimas')
    plt.plot(obj.time_xi, obj.xi_upper, label='PN fit with $e_0=%.3f$' % obj.popt_maximas[0], c='C0')
    plt.plot(obj.time_xi, -obj.xi_lower, label='PN fit with $e_0=%.3f$' % obj.popt_minimas[0], c='C1')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('${\\xi}$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.show()


def plot_maximas_and_minimas_fit(obj, figsize=(8, 5)):
    """Plot both upper and lower envelope fits."""
    plt.figure(figsize=figsize)
    plt.plot(obj.t_maximas, obj.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$')
    plt.plot(obj.time_xi, obj.xi_upper, label='PN fit with $e_0=%.3f$' % obj.popt_maximas[0], c='C0')
    plt.plot(obj.t_minimas, obj.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$')
    plt.plot(obj.time_xi, obj.xi_lower, label='PN fit with $e_0=%.3f$' % obj.popt_minimas[0], c='C1')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('$|{\\xi}^{\\rm env}|$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.show()


def plot_maximas_minimas_and_avg_fit(obj, figsize=(8, 5)):
    """Plot upper, lower, and average envelope fits."""
    plt.figure(figsize=figsize)
    plt.plot(obj.t_maximas, obj.y_maximas, 'o', color='C0', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$')
    plt.plot(obj.time_xi, obj.xi_upper, label='$|{\\xi}_{\\rm upper}^{\\rm env}|$ fit with $e_0=%.3f$' % obj.popt_maximas[0], c='C0')
    plt.plot(obj.t_minimas, obj.y_minimas, 's', color='C1', markersize=10, alpha=0.5, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$')
    plt.plot(obj.time_xi, obj.xi_lower, label='$|{\\xi}_{\\rm lower}^{\\rm env}|$ fit with $e_0=%.3f$' % obj.popt_minimas[0], c='C1')
    plt.plot(obj.time_xi, obj.xi_avg, label='$|{\\xi}_{\\rm avg}^{\\rm env}|$', c='k', ls='--')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('$|{\\xi}^{\\rm env}|$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=12)
    plt.show()


def plot_eccentricity(obj, figsize=(8, 5)):
    """Plot eccentricity evolution."""
    plt.figure(figsize=figsize)
    plt.plot(obj.time_xi, obj.ecc_xi, c='C0', ls='-')
    plt.axvline(x=obj.t_ref, c='k', ls='--')
    plt.text(obj.t_ref + 10, obj.ecc_ref * 0.5, '$e_{\\rm ref}$', fontsize=14, color='red', rotation=90)
    plt.plot(obj.t_ref, obj.ecc_ref, 'o', color='red')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('$e_{\\xi}$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()


# =========================================================================
# Plots for ComputeEccentricityFromOmega
# =========================================================================

def plot_eccentricities_omega(obj, figsize=(8, 5)):
    """Plot eccentricity evolutions (e_omega_22 and e_gw)."""
    plt.figure(figsize=figsize)
    plt.plot(obj.time_xi, obj.ecc_omega22, c='C1', ls='-', label='$e_{\\omega_{22}}$')
    plt.plot(obj.time_xi, obj.ecc_gw, c='C2', ls='-', label='$e_{\\rm gw}$')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Eccentricities', fontsize=18)
    plt.ylim(ymin=0)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.show()


def plot_omega_22_with_envelopes(obj, figsize=(8, 5)):
    """Plot eccentric omega 22 along with upper and lower envelope fits."""
    plt.figure(figsize=figsize)
    plt.plot(obj.time_xi, obj.omega_22_ecc, color='C0', markersize=10, alpha=0.7, label='$\\omega_{22}^{\\rm ecc}$')
    plt.plot(obj.time_xi, obj.omega_22_cir, color='C4', markersize=10, alpha=0.7, label='$\\omega_{22}^{\\rm cir}$')
    plt.plot(obj.time_xi, obj.omega_22_p, color='C1', ls='--', markersize=10, alpha=0.7, label='$\\omega_{22}^{\\rm p}$')
    plt.plot(obj.time_xi, obj.omega_22_a, color='C2', ls='-.', markersize=10, alpha=0.7, label='$\\omega_{22}^{\\rm a}$')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('${\\omega_{22}}$', fontsize=18)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=14)
    plt.ylim(0, 0.022 * np.pi * 2)
    plt.show()
