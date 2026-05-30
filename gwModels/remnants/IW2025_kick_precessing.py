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
#    Requires: torch, nflows
#
#    AUTHOR: Tousif Islam
#    CREATED: 08-11-2025
#    LAST MODIFIED: 05-28-2026
#    REVISION: ---
#==============================================================================
__author__ = "Tousif Islam"

import numpy as np


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

        For non-spinning systems (a1==0 and a2==0), falls back to the
        aligned-spin model gwModel_kick_q200.

        Parameters:
            q: Mass ratio (m1/m2 >= 1)
            a1: Spin magnitude of primary BH
            a2: Spin magnitude of secondary BH
            num_samples: Number of samples to draw

        Returns:
            samples: 1-D array of kick velocities in km/s
        """
        if a1 == 0 and a2 == 0:
            from .IW2025_kick_nonprecessing import gwModel_kick_q200
            return np.full(num_samples, gwModel_kick_q200(q, chi1z=a1, chi2z=a2))

        contexts_raw = np.c_[np.log2(q), a1, a2]
        import torch
        contexts = (contexts_raw - self._cfg['context_mean']) / self._cfg['context_std']
        contexts = torch.tensor(contexts, dtype=torch.float32)

        samples_list = []
        for context in contexts:
            s = self._flow.sample(num_samples=num_samples,
                                  context=context.unsqueeze(0))
            samples_list.append(s.detach().numpy().squeeze())

        samples = np.array(samples_list) * self._cfg['x_std'] + self._cfg['x_mean']
        return np.abs(samples.flatten())

    def info(self):
        print("gwModel_kick_prec_flow — normalizing-flow kick model for precessing BBH")
        print(f"  Inputs      : q, a1, a2 (spin angles marginalized)")
        print(f"  Flow config : d_in={self._cfg['d_in']}, d_hidden={self._cfg['d_hidden']}, "
              f"d_context={self._cfg['d_context']}, n_layers={self._cfg['n_layers']}")
