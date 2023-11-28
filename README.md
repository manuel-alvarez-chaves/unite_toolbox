# UNITE Toolbox
###  Unified diagnostic evaluation of physics-based, data-driven and hybrid hydrological models based on information theory

This repository contains code for the UNITE set of tools based on information theory for the diagnostic evaluation of hydrological models. In the UNITE tools we have functions to calculate different quantities used in information theory: entropy $H(X)$, Kullback-Leibler divergence $D_{KL}(p||q)$, mutual information $I(X; Y)$, using different methods. More specifically, the methods implemented are:

 - Kernel density based estimation (KDE)
 - Binning using histograms
 - *k*-nearest neighbor based estimation (*k*NN)

## Installation
Currently, the best way to experiment with the code is to install it directly from Github using `pip`. The code was written in Python 3.10.12 but should work in earlier versions starting with 3.6. The only dependencies are `numpy` >= 1.25 and `scipy` >= 1.10.1 but also `git` is required for installation.

```
pip install "git+https://github.com/manuel-alvarez-chaves/unite_toolbox"
```

## How-to

In the folder `examples\` please find a tutorial on the general usage of the toolbox.

