# UNITE Toolbox
###  Unified diagnostic evaluation of physics-based, data-driven and hybrid hydrological models based on information theory

This repository contains code for the UNITE set of tools based on information theory for the diagnostic evaluation of hydrological models. In the UNITE tools we have functions to calculate different quantities used in information theory: entropy $H(X)$, Kullback-Leibler divergence $D_{KL}(p||q)$, mutual information $I(X; Y)$, using different methods. More specifically, the methods implemented are:

 - Kernel density based estimation (KDE)
 - Binning using histograms
 - *k*-nearest neighbor based estimation (*k*NN)

## Installation
Although the code is still highly experimental and in very active development, a release version is hosted in PyPI and can be installed using `pip`. Check the `pyproject.toml` for requirements. The current state of the toolbox can be installed directly from this repository using `git`.

```
pip install unite_toolbox
```

## How-to

In the folder `examples\` please find a tutorial on the general usage of the toolbox.

