"""Shared fixtures for gwModels tests."""

import os
import pytest
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'gwModels', 'data')


@pytest.fixture(scope="session")
def sxs1355_data():
    """Load SXS:BBH:1355 waveform data used across multiple test modules."""
    path = os.path.join(DATA_DIR, 'SXSBBH1355.npy')
    return np.load(path, allow_pickle=True)[()]


@pytest.fixture(scope="session")
def data_dir():
    """Return the path to the gwModels/data directory."""
    return DATA_DIR
