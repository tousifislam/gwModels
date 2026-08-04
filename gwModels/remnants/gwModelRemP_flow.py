#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: gwModelRemP_flow.py
#
#    gwModelRemP_flow: probabilistic recoil model for precessing BBH mergers.
#    A conditional Rational-Quadratic Neural Spline Flow (RQ-NSF) models
#    P(log10(1/v_kick) | c) with a five-dimensional context built from the
#    gwModelRemP remnant predictions and the in-plane spin combinations.
#
#    Requires torch and nflows (install with the "kicks" extra), and the
#    checkpoint gwModelRemP_flow.pt in the gwModels data directory.
#
#    From Islam, Wadekar & Khanna (2026), https://arxiv.org/abs/2608.00934
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-02-2026
#    LAST MODIFIED: 08-02-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import os

import numpy as np

from .remnant_utils import symmetric_mass_ratio
from .gwModelRemS import gwModelRemS_mf, gwModelRemS_chif
from .gwModelRemP import spin_projections, _mass_aug, _spin_magnitude_aug

# =============================================================================
# Model description
#
# The precessing recoil is not modeled deterministically. A single-valued map
# v_kick = f(q, chi1_vec, chi2_vec) would have to resolve the detailed
# dependence on all four spin-orientation angles, and no such formulation was
# found that matches the NR data at an accuracy comparable to the mass, spin
# and luminosity models. Instead the flow learns the recoil distribution
# marginalized over the orientation degrees of freedom not retained in the
# context
#
#     c = (Mf_model, |chi_f|_model, eta, S_perp, Delta_perp)
#
# evaluated with the gwModelRemP models at the spins at r = 8M. Both target and
# context are standardized to zero mean and unit variance; the standardization
# constants ship inside the checkpoint.
#
# =============================================================================

_D_TARGET = 1
_D_CONTEXT = 5

_DEFAULT_CONFIG = {
    'd_hidden': 64,
    'n_layers': 8,
    'num_bins': 8,
    'dropout': 0.05,
}

_CHECKPOINT_NAME = 'gwModelRemP_flow.pt'


def _build_flow(d_hidden=64, n_layers=8, num_bins=8, dropout=0.05):
    """
    Build the RQ-NSF architecture matching the saved checkpoint.

    Parameters:
        d_hidden: Hidden features per residual block.
        n_layers: Number of masked autoregressive spline transforms.
        num_bins: Number of spline bins.
        dropout: Dropout probability inside the residual blocks.

    Returns:
        nflows Flow: Untrained flow with the target architecture.
    """
    from nflows.flows.base import Flow
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.autoregressive import (
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
    from nflows.transforms.permutations import ReversePermutation

    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=_D_TARGET))
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=_D_TARGET,
                hidden_features=d_hidden,
                context_features=_D_CONTEXT,
                num_blocks=2,
                num_bins=num_bins,
                tails='linear',
                tail_bound=6.0,
                dropout_probability=dropout,
                use_residual_blocks=True,
            )
        )
    return Flow(CompositeTransform(transforms), StandardNormal(shape=[_D_TARGET]))


class gwModelRemP_flow:
    """
    Conditional normalizing-flow model for the precessing BBH recoil velocity.

    Samples recoil velocities for a given precessing configuration, marginalizing
    over the spin-orientation information not retained in the context vector.

    Context features are computed with the gwModelRemS and gwModelRemP models in
    this package, so the flow stays consistent with the deterministic remnant
    models rather than carrying private copies of them.

    Parameters:
        datadir (str): Directory containing gwModelRemP_flow.pt. Defaults to the
                       data directory shipped with gwModels.

    Attributes:
        best_nll (float or None): Test negative log likelihood from the checkpoint.

    Example:
        >>> import numpy as np, gwModels
        >>> flow = gwModels.remnants.gwModelRemP_flow()
        >>> med, p5, p95 = flow.predict(2.0, 0.7, 0.3, np.pi/3, np.pi/4, 0.0, 0.0)
    """

    def __init__(self, datadir=None):
        import torch

        if datadir is None:
            datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   '..', 'data')
        model_path = os.path.join(datadir, _CHECKPOINT_NAME)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Flow checkpoint not found at {model_path}. Run "
                f"gwmodels_setup_data.py to verify the gwModels data directory."
            )

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        self._ctx_mean = checkpoint['ctx_mean']
        self._ctx_std = checkpoint['ctx_std']
        self._tgt_mean = checkpoint['tgt_mean']
        self._tgt_std = checkpoint['tgt_std']
        self.best_nll = checkpoint.get('best_nll')

        config = checkpoint.get('config') or _DEFAULT_CONFIG
        flow_keys = set(_DEFAULT_CONFIG)
        self._flow = _build_flow(
            **{k: v for k, v in config.items() if k in flow_keys})
        self._flow.load_state_dict(checkpoint['state_dict'])
        self._flow.eval()

    def compute_context(self, q, a1, a2, theta1, theta2):
        """
        Build the five-dimensional context vector for the flow.

        Parameters:
            q: Mass ratio m1/m2 >= 1, scalar or array.
            a1, a2: Spin magnitudes at r = 8M, in [0, 1].
            theta1, theta2: Spin tilt angles at r = 8M in radians, in [0, pi].

        Returns:
            array: Shape (N, 5), columns Mf, abs(chi_f), eta, S_perp, Delta_perp.
        """
        q = np.atleast_1d(np.asarray(q, dtype=float))
        a1 = np.atleast_1d(np.asarray(a1, dtype=float))
        a2 = np.atleast_1d(np.asarray(a2, dtype=float))
        theta1 = np.atleast_1d(np.asarray(theta1, dtype=float))
        theta2 = np.atleast_1d(np.asarray(theta2, dtype=float))

        q, a1, a2, theta1, theta2 = np.broadcast_arrays(q, a1, a2, theta1, theta2)

        chi1z, chi2z, S_perp, Delta_perp = spin_projections(
            q, a1, a2, theta1, theta2)

        eta = symmetric_mass_ratio(q)

        mf_al = np.atleast_1d(np.asarray(gwModelRemS_mf(q, chi1z, chi2z), dtype=float))
        chif_al = np.atleast_1d(np.asarray(gwModelRemS_chif(q, chi1z, chi2z), dtype=float))

        mf = _mass_aug(mf_al, chif_al, S_perp, Delta_perp, eta)
        af = _spin_magnitude_aug(chif_al, S_perp, Delta_perp, eta)

        return np.column_stack([mf, af, eta, S_perp, Delta_perp])

    def sample(self, q, a1, a2, theta1, theta2, phi1, phi2, n_samples=1000):
        """
        Draw recoil velocity samples from the conditional distribution.

        Parameters:
            q: Mass ratio m1/m2 >= 1, scalar or array.
            a1, a2: Spin magnitudes at r = 8M, in [0, 1].
            theta1, theta2: Spin tilt angles at r = 8M in radians, in [0, pi].
            phi1, phi2: Spin azimuthal angles at r = 8M in radians. Accepted for
                interface completeness; the context marginalizes over them.
            n_samples: Number of samples to draw per binary.

        Returns:
            array: Shape (N, n_samples) recoil velocities in km/s, where N is
                the number of input systems (1 for scalar inputs).
        """
        import torch

        ctx_raw = self.compute_context(q, a1, a2, theta1, theta2)
        ctx_norm = (ctx_raw - self._ctx_mean) / self._ctx_std
        ctx_tensor = torch.tensor(ctx_norm, dtype=torch.float32)

        with torch.no_grad():
            # nflows returns shape (N, n_samples, D_TARGET)
            raw = self._flow.sample(n_samples, context=ctx_tensor).numpy()

        # Undo standardization, then map log10(1/v) back to km/s
        log10_inv_vk = raw[:, :, 0] * self._tgt_std[0] + self._tgt_mean[0]
        return np.abs(10.0 ** (-log10_inv_vk))

    def predict(self, q, a1, a2, theta1, theta2, phi1, phi2, n_samples=5000):
        """
        Point estimate: median recoil velocity and 90% credible interval.

        Parameters:
            q: Mass ratio m1/m2 >= 1, scalar or array.
            a1, a2: Spin magnitudes at r = 8M, in [0, 1].
            theta1, theta2: Spin tilt angles at r = 8M in radians, in [0, pi].
            phi1, phi2: Spin azimuthal angles at r = 8M in radians. Not used.
            n_samples: Number of samples used to compute the statistics.

        Returns:
            tuple: (median, p5, p95) recoil velocities in km/s.
        """
        samples = self.sample(q, a1, a2, theta1, theta2, phi1, phi2,
                              n_samples=n_samples)
        median = np.median(samples, axis=1)
        p5 = np.percentile(samples, 5, axis=1)
        p95 = np.percentile(samples, 95, axis=1)

        if median.size == 1:
            return median.item(), p5.item(), p95.item()
        return median, p5, p95
