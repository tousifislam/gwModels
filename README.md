![alt text](gwModels.png)

## **gwModels**
This package is intended to host a variety of data-driven and phenomenological models for the gravitational radiation (waveforms) emitted from binary black hole mergers. For questions, suggestions or collaborations, please feel free to drop an email to Tousif Islam (tousifislam24@gmail.com)

## Getting the package
The latest development version will always be available from the project git repository:
```bash
git clone https://github.com/tousifislam/gwModels
```

## Available Models

#### 1. BHPTNRSur1dq1e4

This model can generate waveforms from a merging non-spinning black hole binary 
systems with mass-ratios varying from 2.5 to 10000. It supports a total of 50 
modes : [(2,2),(2,1),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4),(5,3),
(5,4),(5,5),(6,4),(6,5),(6,6),(7,5),(7,6),(7,7),(8,6),(8,7),(8,8),(9,7),(9,8),(9,9),(10,8),(10,9)]
and their m<0 counterparts. Uncalibrated raw ppBHPT waveforms are ~30,500M long.
Modes up to $\ell=5$ are NR-calibrated. The model has been further tested against
state-of-art NR simulations at mass ratio $q=15,16,30,32$.

Model details can be found in the following paper:
[Surrogate model for gravitational wave signals from non-spinning, comparable- to
large-mass-ratio black hole binaries built on black hole perturbation theory waveforms
calibrated to numerical relativity](https://arxiv.org/pdf/2204.01972.pdf)

#### 2. EMRISur1dq1e4 (deprecated)

EMRISur1dq1e4 is the predecessor of the BHPTNRSur1dq1e4 model, which includes numerous
numerous upgrades (please see [table 1](https://arxiv.org/pdf/2204.01972.pdf)). EMRISur1dq1e4 is
a non-spinning model trained for mass ratios $q=3$ to $q=10000$ and the dominant $(2,2)$ 
mode was calibrated to NR in the comparable mass ratios. The EMRISur1dq1e4 model is NOT supported in 
this package but can be accessed from [EMRISurrogate](https://bhptoolkit.org/EMRISurrogate/).
**CAUTION :** This model is outdated and we advise for using BHPTNRSurrogate(s).

# Requirements
This package requires Python 3, and gwtools.

```bash
pip install gwtools
```

Parts of the accompanying Jupyter notebook will require gwsurrogate, 
which can be installed with either pip

```bash
pip install gwsurrogate
```

or conda

```bash
conda install -c conda-forge gwsurrogate
```

Note that you do not need gwsurrogate to evalulate the EMRI surrogate model or 
run most parts of the notebook.


## Issue tracker
Known bugs are recorded in the project bug tracker:
https://github.com/tousifislam/gwModels/issues

## License
This code is distributed under the MIT License. Details can be found in the LICENSE file.

## Maintainer
Tousif Islam

## Citation guideline
If you make use of the gwModels framework, please cite the following paper:

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














