#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: IW_final_kick_gpr.py
#
#    GPR-based aligned-spin kick velocity model (gwModel_kick_q200_GPR).
#    From Islam & Wadekar (2025), https://arxiv.org/abs/2511.11536
#    Gaussian Process Regression trained on kick data with features
#    [log2(q), chi_hat, chi_a].
#
#    Requires: scikit-learn (for GPR predict and StandardScaler)
#
#    AUTHOR: Tousif Islam
#    CREATED: 05-28-2026
#    LAST MODIFIED: 05-30-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import warnings
import numpy as np
import pickle
from .remnant_utils import validate_q, validate_spin_z


class gwModel_kick_q200_GPR:
    """
    GPR-based aligned-spin kick velocity model.
    From Islam & Wadekar (2025), https://arxiv.org/abs/2511.11536

    Gaussian Process trained on kick data with features
    [log2(q), chi_hat, chi_a].

    For the analytical kick model, use gwModel_kick_q200 from
    IW2025_kick_nonprecessing.

    Parameters:
        model_path (str): Path to the .pkl model file.
    """

    def __init__(self, model_path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*InconsistentVersionWarning.*")
            warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
        self._gpr = data['gpr_model']
        self._scaler_X = data['scaler_X']
        self._scaler_y = data['scaler_y']

    def predict(self, q, chi1z, chi2z):
        """
        GPR kick prediction.

        Parameters:
            q: Mass ratio m1/m2 >= 1
            chi1z: Dimensionless spin of primary along z, in [-1, 1]
            chi2z: Dimensionless spin of secondary along z, in [-1, 1]

        Returns:
            vk: Kick velocity in km/s
            vk_std: GPR uncertainty in km/s
        """
        q = np.atleast_1d(validate_q(q))
        chi1z, chi2z = validate_spin_z(chi1z, chi2z)
        chi1z = np.atleast_1d(chi1z)
        chi2z = np.atleast_1d(chi2z)

        eta = q / (1 + q)**2
        chi_eff = (q * chi1z + chi2z) / (1 + q)
        chi_hat = (chi_eff - 38 * eta * (chi1z + chi2z) / 113) / (1 - 76 * eta / 113)
        chi_a = 0.5 * (chi1z - chi2z)

        X = np.column_stack([np.log2(q), chi_hat, chi_a])
        X_scaled = self._scaler_X.transform(X)

        y_pred_scaled, y_std_scaled = self._gpr.predict(X_scaled, return_std=True)
        vk = self._scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        vk_std = y_std_scaled * self._scaler_y.scale_

        return vk, vk_std

    def info(self):
        print("gwModel_kick_q200_GPR — GPR aligned-spin kick model")
        print("  Islam & Wadekar (2025), https://arxiv.org/abs/2511.11536")
        print("  GPR features: [log2(q), chi_hat, chi_a]")
        print(f"  Valid range : {{'q': [1, 1000], 'chi1z': [-1, 1], 'chi2z': [-1, 1]}}")
