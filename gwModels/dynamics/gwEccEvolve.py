#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwEccEvolve.py
#
#    AUTHOR: Tousif Islam
#    CREATED: 05-28-2026
#    LAST MODIFIED: 05-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.linalg import cho_solve


def _rbf_kernel(X1, X2, constant_value, length_scale):
    """Compute C * RBF kernel: k(x1, x2) = sigma^2 * exp(-0.5 * sum((x1-x2)^2 / l^2))"""
    X1_scaled = X1 / length_scale
    X2_scaled = X2 / length_scale
    dists_sq = (
        np.sum(X1_scaled**2, axis=1, keepdims=True)
        - 2.0 * X1_scaled @ X2_scaled.T
        + np.sum(X2_scaled**2, axis=1)
    )
    return constant_value * np.exp(-0.5 * dists_sq)


def _gpr_predict(gpr_data, X_new, return_std=False):
    """Pure-numpy GPR prediction using stored kernel params, alpha, and Cholesky factor."""
    K_star = _rbf_kernel(X_new, gpr_data['X_train'],
                         gpr_data['constant_value'], gpr_data['length_scale'])
    y_mean = K_star @ gpr_data['alpha']

    if return_std:
        K_star_star = np.full(X_new.shape[0], gpr_data['constant_value'])
        V = cho_solve((gpr_data['L'], True), K_star.T)
        y_var = K_star_star - np.einsum('ij,ji->i', K_star, V)
        y_var = np.maximum(y_var, 0.0)
        return y_mean, np.sqrt(y_var)

    return y_mean


class gwEccEvolve_NoSpinq4:
    """
    SVD-based surrogate model for eccentricity evolution in non-spinning BBH systems.
    From Islam et al. https://arxiv.org/pdf/2604.17868

    The model decomposes e(t)/e0 via SVD, then uses Gaussian Process Regression
    to predict the SVD coefficients as functions of (q, e0).

    Parameters:
        model_path (str, optional): Path to the model .npy file.
            If None, loads the bundled model from the package data directory.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__),
                                      'data', 'gwEccEvolve_NoSpinq4.npy')
        data = np.load(model_path, allow_pickle=True)[()]
        self.tcommon = data['tcommon']
        self.Vh = data['Vh']
        self.coeffs0 = data['coeffs0']
        self._gpr1 = data['gpr1']
        self._gpr2 = data['gpr2']
        self.q_range = data['q_range']
        self.e0_range = data['e0_range']
        self.n_basis = data['n_basis']

    def __call__(self, t, q, e0, return_std=False):
        """
        Evaluate the eccentricity evolution model.

        Parameters:
            t (array_like): Time array in units of M (negative before merger, merger at t=0).
            q (float): Mass ratio (q >= 1).
            e0 (float): Initial eccentricity at t[0].
            return_std (bool): If True, also return the GPR uncertainty estimate.

        Returns:
            ecc (ndarray): Eccentricity as a function of time.
            ecc_std (ndarray): Estimated 1-sigma uncertainty (only if return_std=True).
        """
        X = np.array([[q, e0]])

        if return_std:
            c1, s1 = _gpr_predict(self._gpr1, X, return_std=True)
            c2, s2 = _gpr_predict(self._gpr2, X, return_std=True)
        else:
            c1 = _gpr_predict(self._gpr1, X)
            c2 = _gpr_predict(self._gpr2, X)

        ecc_scaled = self.Vh[0] * (self.coeffs0 + c1[0]) + self.Vh[1] * c2[0]

        t = np.atleast_1d(t)
        spline = InterpolatedUnivariateSpline(self.tcommon, ecc_scaled, k=3)
        ecc = e0 * spline(t)

        if return_std:
            std_scaled = np.sqrt((self.Vh[0] * s1[0])**2 + (self.Vh[1] * s2[0])**2)
            std_spline = InterpolatedUnivariateSpline(self.tcommon, std_scaled, k=3)
            ecc_std = e0 * std_spline(t)
            return ecc, ecc_std

        return ecc

    def info(self):
        """Print model summary."""
        print("gwEccEvolve_NoSpinq4 — SVD surrogate for eccentricity evolution")
        print(f"  Time domain : [{self.tcommon[0]:.0f}, {self.tcommon[-1]:.0f}] M")
        print(f"  q range     : [{self.q_range[0]:.3f}, {self.q_range[1]:.3f}]")
        print(f"  e0 range    : [{self.e0_range[0]:.3f}, {self.e0_range[1]:.3f}]")
        print(f"  SVD basis   : {self.n_basis} components")
