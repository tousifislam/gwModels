#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwnrhme.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 07-02-2024
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

from .base import BaseEccentricHM


class NRHME(BaseEccentricHM):
    """
    Convert a multi-modal quasi-circular non-spinning waveform
    into a multi-modal eccentric non-spinning waveform given a known
    non-spinning quadrupolar eccentric waveform.
    """
    _description = "non-spinning"
