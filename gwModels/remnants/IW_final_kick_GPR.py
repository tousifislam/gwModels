#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: IW_final_kick_GPR.py
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
#    LAST MODIFIED: 05-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
import pickle


class gwModel_kick_q200_GPR:
    """
    GPR-based aligned-spin kick velocity model.
    From Islam & Wadekar (2025), https://arxiv.org/abs/2511.11536

    The model has two components:
    1. Analytical model: refitted RIT formula with 20 coefficients
    2. GPR model: Gaussian Process trained on kick data with features
       [log2(q), chi_hat, chi_a]

    Parameters:
        model_path (str): Path to the .pkl model file.
    """

    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self._gpr = data['gpr_model']
        self._scaler_X = data['scaler_X']
        self._scaler_y = data['scaler_y']
        self._params = data['analytical_params']
        self._params_std = data['analytical_params_std']

    def predict_analytical(self, q, s1z, s2z, return_std=False):
        """
        Analytical aligned-spin kick (refitted RIT formula).

        Parameters:
            q: Mass ratio m1/m2 >= 1
            s1z: Dimensionless spin of primary along z, in [-1, 1]
            s2z: Dimensionless spin of secondary along z, in [-1, 1]
            return_std: If True, also return parameter uncertainty estimate

        Returns:
            V_kick: Kick velocity in km/s
            V_kick_std (optional): Uncertainty in km/s
        """
        q = np.asarray(q, dtype=float)
        s1z = np.asarray(s1z, dtype=float)
        s2z = np.asarray(s2z, dtype=float)

        V_kick = self._compute_analytical(q, s1z, s2z, self._params)

        if return_std:
            variance = np.zeros_like(V_kick)
            for pname in self._params:
                p_pert = self._params.copy()
                p_pert[pname] = self._params[pname] + self._params_std[pname]
                delta_V = self._compute_analytical(q, s1z, s2z, p_pert) - V_kick
                variance += delta_V**2
            return V_kick, np.sqrt(variance)
        return V_kick

    def predict_gpr(self, q, s1z, s2z):
        """
        GPR kick prediction.

        Parameters:
            q: Mass ratio m1/m2 >= 1
            s1z: Dimensionless spin of primary along z, in [-1, 1]
            s2z: Dimensionless spin of secondary along z, in [-1, 1]

        Returns:
            vk: Kick velocity in km/s
            vk_std: GPR uncertainty in km/s
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        s1z = np.atleast_1d(np.asarray(s1z, dtype=float))
        s2z = np.atleast_1d(np.asarray(s2z, dtype=float))

        eta = q / (1 + q)**2
        chi_eff = (q * s1z + s2z) / (1 + q)
        chi_hat = (chi_eff - 38 * eta * (s1z + s2z) / 113) / (1 - 76 * eta / 113)
        chi_a = 0.5 * (s1z - s2z)

        X = np.column_stack([np.log2(q), chi_hat, chi_a])
        X_scaled = self._scaler_X.transform(X)

        y_pred_scaled, y_std_scaled = self._gpr.predict(X_scaled, return_std=True)
        vk = self._scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        vk_std = y_std_scaled * self._scaler_y.scale_

        return vk, vk_std

    @staticmethod
    def _compute_analytical(q, s1z, s2z, p):
        eta = q / (1.0 + q)**2
        delta_m = (q - 1.0) / (q + 1.0)
        S_tilde_par = (s1z + q**2 * s2z) / (1.0 + q)**2
        Delta_tilde_par = (s1z - q * s2z) / (1.0 + q)

        poly = (
            Delta_tilde_par
            + p['H2a'] * S_tilde_par * delta_m
            + p['H2b'] * Delta_tilde_par * S_tilde_par
            + p['H3a'] * (Delta_tilde_par**2) * delta_m
            + p['H3b'] * (S_tilde_par**2) * delta_m
            + p['H3c'] * Delta_tilde_par * (S_tilde_par**2)
            + p['H3d'] * (Delta_tilde_par**3)
            + p['H3e'] * Delta_tilde_par * (delta_m**2)
            + p['H4a'] * S_tilde_par * (Delta_tilde_par**2) * delta_m
            + p['H4b'] * (S_tilde_par**3) * delta_m
            + p['H4c'] * S_tilde_par * (delta_m**3)
            + p['H4d'] * Delta_tilde_par * S_tilde_par * (delta_m**2)
            + p['H4e'] * Delta_tilde_par * (S_tilde_par**3)
            + p['H4f'] * S_tilde_par * (Delta_tilde_par**3)
        )
        Vperp = p['H'] * (eta**2) * poly
        Vm = p['A'] * (eta**2) * delta_m * (1.0 + p['B'] * eta + p['C'] * eta**2)
        xi = np.deg2rad(p['a_deg'] + p['b_deg'] * S_tilde_par
                        + p['c_deg'] * delta_m * Delta_tilde_par)
        return np.sqrt(Vm**2 + Vperp**2 + 2.0 * Vm * Vperp * np.cos(xi))

    def info(self):
        print("gwModel_kick_q200_GPR — GPR aligned-spin kick model")
        print("  Islam & Wadekar (2025), https://arxiv.org/abs/2511.11536")
        print("  Components  : analytical (refitted RIT) + GPR")
        print("  GPR features: [log2(q), chi_hat, chi_a]")
        print(f"  Valid range : {{'q': [1, 1000], 's1z': [-1, 1], 's2z': [-1, 1]}}")
