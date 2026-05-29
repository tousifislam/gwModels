#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: seob.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 05-28-2026
#    LAST MODIFIED: 05-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from pyseobnr.generate_waveform import generate_modes_opt
except ImportError:
    generate_modes_opt = None
    logger.info("pyseobnr not installed; SEOBNRv5EHM will not be available")


class genSEOBNRv5EHM:
    """
    Class to generate waveforms using the SEOBNRv5EHM model via pyseobnr.
    """

    def _generate_raw_SEOBNRv5EHM(self, params):
        """
        Generate raw SEOBNRv5EHM waveforms from pyseobnr.

        Parameters:
            params (dict): Dictionary containing parameters for waveform generation:
                - q (float): Mass ratio (>= 1).
                - s1z (float): z-component of spin for primary (default 0).
                - s2z (float): z-component of spin for secondary (default 0).
                - e0 (float): Initial eccentricity.
                - l0 (float): Initial relativistic anomaly in radians (default 0).
                - omega0 (float): Initial dimensionless orbital frequency.

        Returns:
            tuple: (t, modes) — time array and dictionary of waveform modes
                   with string keys like '2,2'.
        """
        if generate_modes_opt is None:
            raise ImportError("pyseobnr is required for SEOBNRv5EHM. "
                              "Install via: pip install pyseobnr")

        q = params["q"]
        s1z = params.get("s1z", 0.0)
        s2z = params.get("s2z", 0.0)
        eccentricity = params["e0"]
        rel_anomaly = params.get("l0", 0.0)
        omega_start = params["omega0"]

        chi1 = [0.0, 0.0, s1z]
        chi2 = [0.0, 0.0, s2z]

        t, modes = generate_modes_opt(q, chi1, chi2,
                                      omega_start,
                                      eccentricity=eccentricity,
                                      rel_anomaly=rel_anomaly,
                                      approximant="SEOBNRv5EHM")
        return t, modes

    def _process_SEOBNRv5EHM_output(self, modes):
        """
        Process SEOBNRv5EHM output into a format compatible with gwModels.

        Parameters:
            modes (dict): Dictionary of waveform modes with string keys (e.g. '2,2').

        Returns:
            dict: Dictionary mapping mode labels (e.g. 'h_l2m2') to complex
                  strain waveforms.
        """
        h_dict = {}
        for key, hlm in modes.items():
            l, m = key.split(",")
            h_dict['h_l%dm%d' % (int(l), int(m))] = hlm
        return h_dict

    def generate_SEOBNRv5EHM(self, params):
        """
        Generate SEOBNRv5EHM waveforms from pyseobnr.

        Parameters:
            params (dict): Dictionary containing parameters for waveform generation.

        Returns:
            tuple: (t, h_dict) — time array and dictionary of waveform modes.
        """
        t, modes = self._generate_raw_SEOBNRv5EHM(params)
        h_dict = self._process_SEOBNRv5EHM_output(modes)
        return t, h_dict
