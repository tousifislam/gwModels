[![arXiv](https://img.shields.io/badge/arXiv-2403.15506-b31b1b.svg)](https://arxiv.org/abs/2403.15506)
[![arXiv](https://img.shields.io/badge/arXiv-2403.03487-b31b1b.svg)](https://arxiv.org/abs/2403.03487)
[![arXiv](https://img.shields.io/badge/arXiv-2408.02762-b31b1b.svg)](https://arxiv.org/abs/2408.02762)
[![arXiv](https://img.shields.io/badge/arXiv-2502.02739-b31b1b.svg)](https://arxiv.org/abs/2502.02739)
[![arXiv](https://img.shields.io/badge/arXiv-2604.17868-b31b1b.svg)](https://arxiv.org/abs/2604.17868)
[![arXiv](https://img.shields.io/badge/arXiv-2511.11536-b31b1b.svg)](https://arxiv.org/abs/2511.11536)
![gwModels](https://raw.githubusercontent.com/tousifislam/gwModels/main/gwModels.png)
![Visitors](https://komarev.com/ghpvc/?username=tousifislam-gwModels&label=visits&color=brightgreen&base=1547)
[![PyPI](https://img.shields.io/pypi/v/gwModels)](https://pypi.org/project/gwModels/)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen)](https://tousifislam.com/gwModels/)
[![License](https://img.shields.io/github/license/tousifislam/gwModels)](https://github.com/tousifislam/gwModels/blob/main/LICENSE)
![Created](https://img.shields.io/badge/created-March%202024-blue)
![Last Updated](https://img.shields.io/badge/last%20updated-May%202026-blue)

## **gwModels**
This package is intended to host a variety of data-driven and phenomenological models for the gravitational radiation (waveforms) emitted from binary black hole mergers. For questions, suggestions or collaborations, please feel free to drop an email to tousifislam24@gmail.com. Detailed documentation is available at https://tousifislam.com/gwModels/

## Getting the package

### From PyPI
```bash
pip install gwModels
```

### From source
```bash
git clone https://github.com/tousifislam/gwModels
cd gwModels
pip install -e .
```

### Data files
Model data files are stored in the `gwModels/data/` directory. After cloning from source, verify all data files are present:
```bash
python gwmodels_setup_data.py
```

## Available Models

### 1. Waveform Frameworks

Frameworks for converting quasi-circular waveforms into eccentric waveforms using known quadrupolar eccentric waveforms.

| Model | Description | Reference | Tutorial |
|-------|-------------|-----------|----------|
| **gwNRHME** | Non-spinning quasi-circular HM waveform → eccentric | [2403.15506](https://arxiv.org/abs/2403.15506) | [1_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/1_1_framework_gwNRHME_example.ipynb) |
| **gwNRXHME** | Non-precessing quasi-circular HM waveform → eccentric | [2502.02739](https://arxiv.org/abs/2502.02739) | [1_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/1_1_framework_gwNRHME_example.ipynb) |

### 2. Eccentric Higher-Mode Waveforms

Eccentric waveform models obtained by combining circular surrogates with an eccentric model through gwNRHME.

| Model | Components | Reference | Tutorial |
|-------|------------|-----------|----------|
| **NRHybSur3dq8-gwNRHME** | NRHybSur3dq8 + SEOBNRv5EHM | [2408.02762](https://arxiv.org/abs/2408.02762) | [2_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/2_1_NRHybSur3dq8-gwNRHME_example.ipynb) |
| **BHPTNRSur1dq1e4-gwNRHME** | BHPTNRSur1dq1e4 + SEOBNRv5EHM | [2408.02762](https://arxiv.org/abs/2408.02762) | [2_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/2_2_BHPTNRSur1dq1e4-gwNRHME_example.ipynb) |

### 3. Eccentricity Estimation

| Model | Description | Reference | Tutorial |
|-------|-------------|-----------|----------|
| **eccentricity_estimation** | Computes $e_{\xi}$, $e_{\omega}$, $e_{\rm gw}$ | [2502.02739](https://arxiv.org/abs/2502.02739) | [3_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/3_1_eccentricity_estimation_nonprecessing.ipynb) |

### 4. Dynamics: Eccentricity Evolution

| Model | Type | Parameter Range | Reference | Tutorial |
|-------|------|-----------------|-----------|----------|
| **gwEccEvNS** | NR-based approximate | Non-spinning | [2502.02739](https://arxiv.org/abs/2502.02739) | [4_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/4_1_dynamics_gwEccEvNS.ipynb) |
| **gwEccEvNSv2** | Analytical | Non-spinning | [2604.17868](https://arxiv.org/abs/2604.17868) | [4_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/4_2_dynamics_gwEccEvNSv2.ipynb) |
| **gwEccEvolve_NoSpinq4** | SVD surrogate + GPR | $1 \leq q \leq 4$, $0.003 \leq e_0 \leq 0.443$ | [2604.17868](https://arxiv.org/abs/2604.17868) | [4_3](https://github.com/tousifislam/gwModels/blob/main/tutorials/4_3_dynamics_gwEccEvolve_NoSpinq4.ipynb) |

### 5. Remnant Properties: Final Mass, Spin, and Kick

#### Kick velocity models

| Model | Type | Valid Range | Extra Deps | Reference | Tutorial |
|-------|------|-------------|------------|-----------|----------|
| **gwModel_kick_q200** | Analytical (aligned-spin) | $1 \leq q \leq 1000$ | — | [2511.11536](https://arxiv.org/abs/2511.11536) | [5_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_1_gwModels_kicks.ipynb) |
| **gwModel_kick_q200_GPR** | GPR (aligned-spin) | $1 \leq q \leq 1000$ | `scikit-learn` | [2511.11536](https://arxiv.org/abs/2511.11536) | [5_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_1_gwModels_kicks.ipynb) |
| **gwModel_kick_prec_flow** | Normalizing flow (precessing) | $q \leq 100$ | `torch`, `nflows` | [2511.11536](https://arxiv.org/abs/2511.11536) | [5_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_1_gwModels_kicks.ipynb) |
| **HLZ_2014_aligned_spin** | Analytical (aligned-spin) | — | — | [1406.7295](https://arxiv.org/abs/1406.7295) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |
| **bbh_final_kick_precessing_CLZM2007** | Analytical (precessing) | — | — | Gonzalez+ 2007, Campanelli+ 2007 | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |

#### Final mass and spin models

| Model | Quantity | Spin Type | Reference | Tutorial |
|-------|----------|-----------|-----------|----------|
| **bbh_final_mass_precessing_BMR2012** | Final mass | Precessing | Barausse, Morozova & Rezzolla (2012) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |
| **bbh_final_spin_precessing_HBR2016** | Final spin | Precessing | Hofmann, Barausse & Rezzolla (2016) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |
| **bbh_final_mass_non_precessing_UIB2016** | Final mass | Aligned-spin | [1611.00332](https://arxiv.org/abs/1611.00332) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |
| **bbh_final_spin_non_precessing_UIB2016** | Final spin | Aligned-spin | [1611.00332](https://arxiv.org/abs/1611.00332) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |

## Requirements
This package requires Python 3 and gwtools.

```bash
pip install gwtools
```

Optional dependencies for specific models:
- `scikit-learn` — for `gwModel_kick_q200_GPR`
- `torch`, `nflows` — for `gwModel_kick_prec_flow`
- `gwsurrogate` — for NRHybSur3dq8-gwNRHME and BHPTNRSur1dq1e4-gwNRHME tutorials

Install optional groups with:
```bash
pip install gwModels[kicks]       # scikit-learn, torch, nflows for remnant kick models
pip install gwModels[all]         # all optional dependencies
```

## Issue tracker
Known bugs are recorded in the project bug tracker:
https://github.com/tousifislam/gwModels/issues

## License
This code is distributed under the MIT License. Details can be found in the LICENSE file.

## Maintainer
Tousif Islam

## Citation guideline
If you make use of the gwModels framework, please cite the relevant papers:

```
@article{Islam:2024rhm,
    author = "Islam, Tousif",
    title = "{Straightforward mode hierarchy in eccentric binary black hole mergers and associated waveform model}",
    eprint = "2403.15506",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "3",
    year = "2024"
}
```

```
@article{Islam:2024tcs,
    author = "Islam, Tousif",
    title = "{Study of eccentric binary black hole mergers using numerical relativity and an inspiral-merger-ringdown model}",
    eprint = "2403.03487",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "3",
    year = "2024"
}
```

```
@article{Islam:2024zqo,
    author = "Islam, Tousif and Khanna, Gaurav and Field, Scott E.",
    title = "{Adding higher-order spherical harmonics in non-spinning eccentric binary black hole merger waveform models}",
    eprint = "2408.02762",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "8",
    year = "2024"
}
```

```
@article{Islam:2025oiv,
    author = "Islam, Tousif and Venumadhav, Tejaswi",
    title = "{Post-Newtonian theory-inspired framework for characterizing eccentricity in gravitational waveforms}",
    eprint = "2502.02739",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "2",
    year = "2025"
}
```

```
@article{Islam:2026blk,
    author = "Islam, Tousif and others",
    title = "{Including higher-order modes in a quadrupolar eccentric numerical relativity surrogate using universal eccentric modulation functions}",
    eprint = "2604.17868",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "4",
    year = "2026"
}
```

```
@article{Islam:2025drw,
    author = "Islam, Tousif and Wadekar, Digvijay",
    title = "{Accurate models for recoil velocity distribution in black hole mergers with comparable to extreme mass-ratios and their astrophysical implications}",
    eprint = "2511.11536",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/4jvv-qg4h",
    journal = "Phys. Rev. D",
    volume = "113",
    number = "10",
    pages = "104017",
    year = "2026"
}
```
