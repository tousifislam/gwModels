#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: conversion.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-02-2024
#    LAST MODIFIED: Tue Feb  6 17:58:52 2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np

def mass_ratio_to_symmetric_mass_ratio(mass_ratio):
    """
    Convert the mass ratio (q >= 1) to the symmetric mass ratio (nu).

    Parameters
    ----------
    mass_ratio : float
                 q = m1 / m2, where m1 >= m2.
    
    Returns
    -------
    Symmetric mass ratio : float
                           nu = m1 * m2 / (m1 + m2)^2.
    """
    total_mass_squared = (1 + mass_ratio) ** 2
    return mass_ratio / total_mass_squared