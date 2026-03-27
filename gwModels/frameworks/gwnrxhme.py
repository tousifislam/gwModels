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
    Convert a multi-modal quasi-circular non-precessing (spin-aligned)
    waveform into a multi-modal eccentric non-precessing waveform,
    given the non-precessing quadrupolar eccentric waveform.

    This implements the gwNRXHME framework described in:
        - arXiv:2403.15506
    """
    pass
