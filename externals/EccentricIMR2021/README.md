# EccentricIMR

## About

A Mathematica package for generating gravitational waveforms for
nonspinning eccentric binary black hole mergers.

EccentricIMR was originally written by Ian Hinder. This is a Git-forked version modified by Sashwat Tanay. 

Copyright (C) Ian Hinder, 2017.

This Mathematica package is based on this paper: [Generalized quasi-Keplerian solution for eccentric, non-spinning compact binaries at 4PN order and the associated IMR waveform, 2021](https://arxiv.org/abs/2110.09608).

## Installation

You can install EccentricIMR by downloading a zip file.

- Download [master.zip](https://github.com/sashwattanay/EccentricIMR/archive/master.zip),
- Extract the zip file
- There is a directory 'EccentricIMR' in '4PN_IMR_package' (plots the full IMR; does not work for large eccentricities) and also in '4PN_Inspiral_only_package'
(uses only the post-Newtonian equations; works for large eccentricities too).
- Move the 'EccentricIMR' directory (from either the '4PN_IMR_package' directory or the '4PN_Inspiral_only_package' directory) into your Mathematica applications directory

LINUX: 

for all users (requires root priviledges): /usr/share/Mathematica/Applications/

for one user: $HOME/.Mathematica/Applications/
   
MAC OS: 

for all users (requires root priviledges): /Library/Mathematica/Applications/

for one user: /Users/<user>/Library/Mathematica/Applications/
  
WINDOWS: 
   
for all users: C:\Program Files\Wolfram Research\Mathematica\<version>\AddOns\Applications\
   
for one user: C:\Users\<user>\AppData\Roaming\Mathematica\Applications\
   
   
- The file 'How to run the packages.nb' has commands that will run the packages and plot the waveforms. Alternatively, see the Sec. "How to run the packages" below. 
- The file 'Lengthy_expressions.nb' and 'Lengthy_expressions.pdf' contains some lengthy expressions which we did not present in the paper for the sake of brevity. 

## How to run the packages
Once the Mathematica package directory has been moved to the correct location (as explained above), the following codes are to be run to generate the waveforms

For the full IMR package
```
<<EccentricIMR`;
params=<|"q"->1,"x0"->0.07,"e0"->0.1,"l0"->0,"phi0"->0,"t0"->0|>;
hEcc=EccentricIMRWaveform[params,{0,10000}];
ListLinePlot[Re[hEcc]]
```

For the Inspiral-only package
```
<< EccentricIMR`;
finTime1 = 
  10000;                              (*final time; end of simulation*)

params = <|"q" -> 1, "x0" -> 0.07, "e0" -> 0.25, "l0" -> 0, 
   "phi0" -> 0, "t0" -> 0|>;
dt =  1 ;                                 
hEcc = EccentricIMRWaveform[params, {0,   finTime1     , dt}];
ifun = hEcc["h"]  ;     
InterFun = hEcc["h"] ;
iniTime  =    ((InterFun@Domain)[[1]])[[1]]   ;
finTime =      ((InterFun@Domain)[[1]])[[2]]   ;
nDiv = Round[(finTime - iniTime)/dt];                  
timeArr = Array[# &, nDiv + 1, {iniTime, finTime}];     
sigArr = Re[ifun[timeArr]];                       
ListLinePlot[sigArr]
```

## Documentation

_parameters_ is an Association with the following entries:

Parameter | Meaning
--------- | ---
q         | Mass ratio of the binary (q=m1/m2)
t0        | _Reference time_ at which the remaining parameters are quoted
x0        | Dimensionless frequency parameter (x = (M om_orb)^(2/3)) evaluated at the reference time
e0		   | Eccentricity at the reference time
l0		   | Mean anomaly at the reference time
phi0	   | Orbital phase at the reference time

See [arXiv:2110.09608](https://arxiv.org/abs/2110.09608) for full details about the meaning of the parameters.

The returned waveform is expressed as a list of {t, h22} pairs, where t is the retarted time coordinate and h22 is the l=2, m=2 spin-weighted spherical harmonic coefficient of the waveform.
