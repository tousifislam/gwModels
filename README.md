[![arXiv](https://img.shields.io/badge/arXiv-2403.15506-b31b1b.svg)](https://arxiv.org/abs/2403.15506)
[![arXiv](https://img.shields.io/badge/arXiv-2403.03487-b31b1b.svg)](https://arxiv.org/abs/2403.03487)
[![arXiv](https://img.shields.io/badge/arXiv-2408.02762-b31b1b.svg)](https://arxiv.org/abs/2408.02762)
![alt text](gwModels.png)

## **gwModels**
This package is intended to host a variety of data-driven and phenomenological models for the gravitational radiation (waveforms) emitted from binary black hole mergers. For questions, suggestions or collaborations, please feel free to drop an email to tousifislam24@gmail.com. Detailed documentation of the package is provided at http://tousifislam.com/gwModels/gwModels.html

## Getting the package
The latest development version will always be available from the project git repository:
```bash
git clone https://github.com/tousifislam/gwModels
```

## Available Models

#### 1. Frameworks

##### 1a. gwNRHME
A framework to seamlessly convert a multi-modal (i.e with several spherical harmonic modes) non-spinning quasi-circular waveform into multi-modal eccentric waveform if the quadrupolar eccentric waveform is known (https://arxiv.org/abs/2403.15506).

##### 1b. gwNRXHME
A framework to seamlessly convert a multi-modal (i.e with several spherical harmonic modes) non-precessing quasi-circular waveform into multi-modal eccentric waveform if the quadrupolar eccentric waveform is known (https://arxiv.org/abs/2403.15506).

#### 2. EccentricIMR 
Python wrapper for the PN based quadrupolar eccentric waveform model (https://arxiv.org/abs/0806.1037). Example use is here: https://github.com/tousifislam/gwModels/blob/main/tutorials/EccentricIMR_example.ipynb

#### 3. Higher modes model with eccentricity
It has two variants based on the constituent circular model. These variants are obtained by combining the following circular and eccentric model through gwNRHME.
- 3a. NRHybSur3dq8-gwNRHME = NRHybSur3dq8 (https://arxiv.org/abs/1812.07865) + EccentricIMR (https://arxiv.org/abs/0806.1037)
- 3b. BHPTNRSur1dq1e4-gwNRHME = BHPTNRSur1dq1e4 (https://arxiv.org/abs/2204.01972) + EccentricIMR (https://arxiv.org/abs/0806.1037)
- 3c. IMRPhenomTHM-gwNRHME = IMRPhenomTHM (https://arxiv.org/abs/2012.11923) + EccentricIMR (https://arxiv.org/abs/0806.1037)

Example use is here: https://github.com/tousifislam/gwModels/blob/main/tutorials/

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
If you make use of the gwModels framework, please cite the following papers:

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













