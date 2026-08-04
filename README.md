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

## Conventions

Throughout the package, we adopt the following conventions for binary black hole parameters:

| Symbol | Description | Range |
|--------|-------------|-------|
| `q` | Mass ratio $m_1/m_2$ (body 1 is always the more massive) | $q \geq 1$ |
| `a1`, `a2` | Dimensionless spin magnitudes | $[0, 1]$ |
| `chi1`, `chi2` | Dimensionless spin vectors `[chi1x, chi1y, chi1z]` | — |
| `chi1z`, `chi2z` | Spin components along orbital angular momentum | $[-1, 1]$ |
| `theta1`, `theta2` | Zenith angle between spin and orbital angular momentum | — |
| `phi1`, `phi2` | Azimuthal angle of spin projection onto the orbital plane | — |
| `delta_phi` | Azimuthal angle difference between spin projections | — |
| `e_ref` | Eccentricity at reference | — |
| `ano_ref` | Anomaly at reference | — |

## Getting the package

This package requires Python 3 and [gwtools](https://pypi.org/project/gwtools/).

### From PyPI
```bash
pip install gwModels                # core (numpy, scipy, matplotlib, gwtools)
pip install gwModels[surrogates]    # + gwsurrogate
pip install gwModels[lal]           # + lalsuite
pip install gwModels[seob]          # + pyseobnr
pip install gwModels[kicks]         # + scikit-learn, torch, nflows (remnant kick models)
pip install gwModels[all]           # all optional dependencies
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

#### gwModelRem family (unified remnant models)

A single framework covering aligned-spin, precessing and eccentric binaries,
plus the point-particle limit. All fitted coefficients are inline in the source.

| Model | Regime | Inputs | Outputs | Valid Range | Extra Deps | Tutorial |
|-------|--------|--------|---------|-------------|------------|----------|
| **gwModelRemS** | Aligned-spin, quasi-circular | $q, \chi_{1z}, \chi_{2z}$ | $M_f, \chi_f, L_{\rm peak}, M\omega_{\rm peak}, v_{\rm kick}$ | $1 \leq q \leq 1000$ | — | [6_1](https://github.com/tousifislam/gwModels/blob/main/tutorials/6_1_gwModelRemS.ipynb) |
| **gwModelRemP** | Precessing, quasi-circular | $q, a_i, \theta_i, \phi_i$ at $r=8M$ | $M_f, \|\chi_f\|, \theta_f, L_{\rm peak}$ | $q \leq 1000$, $S_\perp \leq 0.93$ | — | [6_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/6_2_gwModelRemP.ipynb) |
| **gwModelRemSE** | Eccentric, non-precessing | $q, \chi_{iz}, e_{\rm ref}, \ell_{\rm ref}$ at $t=-2500M$ | $M_f, \chi_f, v_{\rm kick}, L_{\rm peak}$ | $q \leq 4$, $e_0 \leq 0.25$, non-spinning | — | [6_3](https://github.com/tousifislam/gwModels/blob/main/tutorials/6_3_gwModelRemSE.ipynb) |
| **gwModelRemPE** | Eccentric, precessing | $q, a_i, \theta_i, \phi_i$ at $r=8M$; $e_{\rm ref}, \ell_{\rm ref}$ at $t=-2500M$ | $M_f, |\chi_f|, \theta_f, L_{\rm peak}$ | $q \leq 4$, $e_0 \leq 0.25$ | -- | [6_4](https://github.com/tousifislam/gwModels/blob/main/tutorials/6_4_gwModelRemPE.ipynb) |
| **gwModelRemP_flow** | Precessing recoil distribution | $q, a_i, \theta_i, \phi_i$ at $r=8M$ | $P(v_{\rm kick})$ | $q \leq 1000$ | `torch`, `nflows` | [6_5](https://github.com/tousifislam/gwModels/blob/main/tutorials/6_5_gwModelRemP_flow.ipynb) |
| **gwModelEMRI** | Point-particle limit | $q, \chi, \theta_{\rm inc}, e_{\rm sep}$ | $M_f, \chi_f$ | $q \gg 1000$ | — | -- |

```python
import numpy as np
import gwModels

Mf, chif, Lpeak, wpeak, vkick = gwModels.remnants.gwModelRemS(3.0, 0.5, -0.2)
Mf, af, theta_f, Lpeak = gwModels.remnants.gwModelRemP(
    2.0, 0.7, 0.3, np.pi/3, np.pi/4, 0.0, 0.0)

flow = gwModels.remnants.gwModelRemP_flow()
median, p5, p95 = flow.predict(2.0, 0.7, 0.3, np.pi/3, np.pi/4, 0.0, 0.0)
```

Per-quantity functions are also available: `gwModelRemS_mf`, `gwModelRemS_chif`,
`gwModelRemS_Lpeak`, `gwModelRemS_omega_peak`, `gwModelRemS_kick`, and the
corresponding `gwModelRemP_*` and `gwModelRemSE_*` entry points.

Two caveats worth reading before use:

- **gwModelRemPE applies the gwModelRemSE corrections to a precessing
  baseline**, treating eccentricity and precession as independent at leading
  order. This is the intended construction, but the factorization has not been
  checked against precessing eccentric NR, of which very little exists, so
  treat its eccentric corrections with the same caution as gwModelRemSE.
- **gwModelRemSE is provisional.** The circular limit is exact, but at
  $e_{\rm ref} > 0$ the correction does not yet improve on the quasi-circular
  baseline on NR data (neutral to a few percent worse inside its calibration
  domain). Do not extrapolate past $e_{\rm ref} \sim 0.3$, where the anomaly
  modulation becomes an order of magnitude larger than the residual it corrects.
- **gwModelEMRI's separatrix solver does not always converge**, failing for
  about 6% of a grid over $(\chi, \theta_{\rm inc}, e)$, worst near polar
  inclination at high spin. It warns by default; pass `return_converged=True`
  for the mask. Equatorial orbits always converge.

The `gwModelRemS` recoil is a refit of `gwModel_kick_q200` on an expanded
dataset, and `gwModelRemP_flow` supersedes `gwModel_kick_prec_flow`. Both
earlier models remain available and unchanged.

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
| **bbh_final_mass_precessing_BMR2012** | Final mass | Precessing | [Barausse, Morozova & Rezzolla (2012)](https://arxiv.org/abs/1206.3803) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |
| **bbh_final_spin_precessing_HBR2016** | Final spin | Precessing | [Hofmann, Barausse & Rezzolla (2016)](https://arxiv.org/abs/1605.01938) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |
| **bbh_final_mass_non_precessing_UIB2016** | Final mass | Aligned-spin | [1611.00332](https://arxiv.org/abs/1611.00332) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |
| **bbh_final_spin_non_precessing_UIB2016** | Final spin | Aligned-spin | [1611.00332](https://arxiv.org/abs/1611.00332) | [5_2](https://github.com/tousifislam/gwModels/blob/main/tutorials/5_2_other_remnant_models.ipynb) |

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
