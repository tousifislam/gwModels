#! /usr/bin/env python
#-*- coding: utf-8 -*-
#==============================================================================
#
#    FILE: IW_final_kick_precessing.py
#
#    Normalizing-flow model for precessing-spin kick velocity distributions.
#    From Islam & Wadekar (2025), https://arxiv.org/abs/2511.11536
#    Given (q, a1, a2), the flow marginalizes over spin angles and returns
#    samples from the kick distribution.
#
#    sample() and log_prob() accept scalar OR array (q, a1, a2): array inputs
#    are evaluated in a single batched call over the flow, which is orders of
#    magnitude faster than looping one binary at a time.
#
#    Requires: torch, nflows
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 06-13-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np
from .remnant_utils import validate_q, validate_spin_magnitudes


def _build_flow(d_in=3, d_hidden=16, d_context=2, n_layers=4):
    import torch  # noqa: F811
    from nflows.distributions.normal import StandardNormal
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
    from nflows.transforms.permutations import ReversePermutation
    from nflows.flows.base import Flow

    base_dist = StandardNormal(shape=[d_in])
    transforms = []
    for _ in range(n_layers):
        transforms.append(ReversePermutation(features=d_in))
        transforms.append(MaskedAffineAutoregressiveTransform(
            features=d_in, hidden_features=d_hidden, context_features=d_context))
    return Flow(CompositeTransform(transforms), base_dist)


class gwModel_kick_prec_flow:
    """
    Normalizing-flow model for precessing-spin kick velocity distributions.

    Samples kick velocities for given (q, a1, a2), marginalizing over
    spin orientation angles.

    Parameters:
        datadir (str): Directory containing gwModel_kick_prec_flow.pt
                       and gwModel_kick_prec_flow_config.npy
    """

    def __init__(self, datadir):
        import torch
        if not datadir.endswith('/'):
            datadir += '/'
        cfg = np.load(datadir + 'gwModel_kick_prec_flow_config.npy',
                      allow_pickle=True).item()
        self._flow = _build_flow(d_in=cfg['d_in'], d_hidden=cfg['d_hidden'],
                                 d_context=cfg['d_context'], n_layers=cfg['n_layers'])
        self._flow.load_state_dict(
            torch.load(datadir + 'gwModel_kick_prec_flow.pt', weights_only=False))
        self._flow.eval()
        self._cfg = cfg

    def sample(self, q, a1, a2, num_samples=5000):
        """
        Sample kick velocities from the flow model.

        Accepts scalar or array (q, a1, a2). Array inputs (length N) are
        evaluated in a single batched call over the flow rather than one
        binary at a time. For non-spinning systems (a1==0 and a2==0), falls
        back to the aligned-spin model gwModel_kick_q200 (applied per element).

        Parameters:
            q: Mass ratio(s) (m1/m2 >= 1), scalar or array
            a1: Primary spin magnitude(s), scalar or array
            a2: Secondary spin magnitude(s), scalar or array
            num_samples: Number of samples to draw per binary

        Returns:
            samples: kick velocities in km/s. Shape conventions:
              - scalar (q, a1, a2)               -> (num_samples,)
              - array of length N, num_samples=1 -> (N,)
              - array of length N, num_samples>1 -> (N, num_samples)

        Note:
            Batched and per-binary calls are statistically equivalent but not
            element-wise identical (the RNG is consumed in a different order).
        """
        validate_q(q)
        validate_spin_magnitudes(a1, a2)
        import torch

        scalar_input = (np.ndim(q) == 0 and np.ndim(a1) == 0
                        and np.ndim(a2) == 0)
        q_arr, a1_arr, a2_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(q, dtype=float)),
            np.atleast_1d(np.asarray(a1, dtype=float)),
            np.atleast_1d(np.asarray(a2, dtype=float)))
        n = len(q_arr)

        out = np.empty((n, num_samples), dtype=float)
        nonspin = (a1_arr == 0) & (a2_arr == 0)
        spin = ~nonspin

        # Non-spinning systems: deterministic aligned-spin fallback
        if nonspin.any():
            from .IW2025_kick_nonprecessing import gwModel_kick_q200
            v_q = gwModel_kick_q200(q_arr[nonspin], chi1z=a1_arr[nonspin],
                                    chi2z=a2_arr[nonspin])
            out[nonspin, :] = np.asarray(v_q, dtype=float).reshape(-1, 1)

        # Spinning systems: one batched flow sample over all contexts
        if spin.any():
            contexts = (np.column_stack([np.log2(q_arr[spin]), a1_arr[spin],
                                         a2_arr[spin]])
                        - self._cfg['context_mean']) / self._cfg['context_std']
            contexts = torch.tensor(contexts, dtype=torch.float32)
            with torch.no_grad():
                s = self._flow.sample(num_samples=num_samples, context=contexts)
            s = s.detach().numpy().reshape(int(spin.sum()), num_samples)
            out[spin, :] = s * self._cfg['x_std'] + self._cfg['x_mean']

        out = np.abs(out)

        if scalar_input:
            return out[0]              # (num_samples,) — backward compatible
        if num_samples == 1:
            return out[:, 0]           # (N,) — one kick per binary
        return out                     # (N, num_samples)

    def log_prob(self, v_kick, q, a1, a2):
        """
        Conditional log-density log p(v_kick | q, a1, a2) of the kick magnitude.

        Evaluates the flow's density of the recoil magnitude v = abs(x) by
        folding the two signed branches of the underlying variable and
        including the normalization Jacobian, so that exp(log_prob) integrates
        to 1 over v >= 0. All inputs broadcast to a common shape and are scored
        in a single batched call.

        This is the primitive for inverting the model (inferring progenitor
        q, a1, a2 from an observed kick via Bayes). It represents the smooth
        flow density and does not apply the non-spinning delta-function
        fallback used by sample(); intended for spinning systems.

        Parameters:
            v_kick: Observed kick magnitude(s) [km/s], scalar or array
            q, a1, a2: Progenitor parameters, scalar or array

        Returns:
            log-density, scalar if all inputs are scalar else a 1-D array.
        """
        validate_q(q)
        validate_spin_magnitudes(a1, a2)
        import torch

        scalar_input = (np.ndim(v_kick) == 0 and np.ndim(q) == 0
                        and np.ndim(a1) == 0 and np.ndim(a2) == 0)
        v, q_arr, a1_arr, a2_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(v_kick, dtype=float)),
            np.atleast_1d(np.asarray(q, dtype=float)),
            np.atleast_1d(np.asarray(a1, dtype=float)),
            np.atleast_1d(np.asarray(a2, dtype=float)))

        contexts = (np.column_stack([np.log2(q_arr), a1_arr, a2_arr])
                    - self._cfg['context_mean']) / self._cfg['context_std']
        contexts = torch.tensor(contexts, dtype=torch.float32)

        x_mean = float(np.asarray(self._cfg['x_mean']).reshape(-1)[0])
        x_std = float(np.asarray(self._cfg['x_std']).reshape(-1)[0])
        x_plus = torch.tensor(((v - x_mean) / x_std).reshape(-1, 1),
                              dtype=torch.float32)
        x_minus = torch.tensor(((-v - x_mean) / x_std).reshape(-1, 1),
                               dtype=torch.float32)
        with torch.no_grad():
            lp = self._flow.log_prob(x_plus, context=contexts).numpy()
            lm = self._flow.log_prob(x_minus, context=contexts).numpy()
        logp = np.logaddexp(lp, lm) - np.log(x_std)

        return float(logp[0]) if scalar_input else logp

    def info(self):
        print("gwModel_kick_prec_flow — normalizing-flow kick model for precessing BBH")
        print(f"  Inputs      : q, a1, a2 (spin angles marginalized)")
        print(f"  Flow config : d_in={self._cfg['d_in']}, d_hidden={self._cfg['d_hidden']}, "
              f"d_context={self._cfg['d_context']}, n_layers={self._cfg['n_layers']}")
