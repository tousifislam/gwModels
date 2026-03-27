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
    into a multi-modal eccentric non-spinning waveform, given
    the non-spinning quadrupolar eccentric waveform.

    This implements the gwNRHME framework described in:
        - arXiv:2403.15506
        - arXiv:2408.02762
    """
    pass
