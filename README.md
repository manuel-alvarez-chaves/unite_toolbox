# UNITE Toolbox

### Unified diagnostic evaluation of scientific models based on information theory

The **UNITE Toolbox** is a Python library for incorporating _Information Theory_
into data analysis and modeling workflows.
The toolbox collects different methods of estimating information-theoretic quantities
in one easy-to-use Python package.
Currently, UNITE includes functions to calculate entropy $H(X)$,
Kullback-Leibler divergence $D_{KL}(p||q)$, and mutual information $I(X; Y)$,
using three methods:

- Kernel density-based estimation (KDE)
- Binning using histograms
- _k_-nearest neighbor-based estimation (_k_-NN)

## Installation

Although the code is still highly experimental and in very active development,
a release version is available on PyPI and can be installed using `pip`.

```
pip install unite_toolbox
```

Alternatively, the latest updates can be installed directly from this repository

```
pip install git+https://github.com/manuel-alvarez-chaves/unite_toolbox
```

Check the `pyproject.toml` for requirements.

## How-to

In the [documentation](https://unite-toolbox.readthedocs.io/) please find
[tutorials](https://unite-toolbox.readthedocs.io/en/latest/tutorials.html) on
the general usage of the toolbox and some applications.
