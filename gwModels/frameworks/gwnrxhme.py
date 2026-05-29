#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwnrxhme.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-02-2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

from .base import BaseEccentricHM


class NRXHME(BaseEccentricHM):
    """
    Convert a multi-modal quasi-circular non-precessing waveform
    into a multi-modal eccentric non-precessing waveform given a known
    non-precessing quadrupolar eccentric waveform.
    """
    _description = "non-precessing"
