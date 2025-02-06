#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwEccEvNS.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 11-01-2024
#    LAST MODIFIED: 
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import numpy as np
import gwtools

def gwEccEvNS_model(t, q, e0):
    """
    Simple model of eccentricity evolution based on SXS and RIT NR data for non-spinning binaries;
    Described in Sev IV E of https://arxiv.org/pdf/2502.02739 by Islam and Venumadhav
    """
    tc = 0
    eta = gwtools.q_to_nu(q)
    t_ref = t[0]
    tau = (tc-t) * (eta/5)
    tau_0 = (tc-t_ref) * (eta/5)
    
    [m, c ] = [-0.37857487, 18.4726538 ]
    term1 = m*q + c
    
    [A, B, C ] = [ 0.15346999, -1.38867977,  2.45635187]
    term2 = (1 + A*e0 + B*e0*e0 + C*e0**3)
    n = term1 * term2
    
    efit = e0 * (tau / tau_0)**(n / 48)

    return efit