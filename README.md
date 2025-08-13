# UNITE Toolbox

### Unified diagnostic evaluation of scientific models based on information theory

![PyPI - Version](https://img.shields.io/pypi/v/unite_toolbox) ![Tests Badge](https://github.com/manuel-alvarez-chaves/unite_toolbox/actions/workflows/run-tests.yml/badge.svg) [![Coverage](https://codecov.io/gh/manuel-alvarez-chaves/unite_toolbox/graph/badge.svg?token=MWNDWXLZ9B)](https://codecov.io/gh/manuel-alvarez-chaves/unite_toolbox) [![Identifier](<https://img.shields.io/badge/DOI-10.18419/darus--4188-blue>)](https://doi.org/10.18419/darus-4188)

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

Or `uv`.

```
uv add unite_toolbox
```

Check the `pyproject.toml` for requirements.

## How-to

In the [documentation](https://unite-toolbox.readthedocs.io/) please find
[tutorials](https://unite-toolbox.readthedocs.io/en/latest/tutorials.html) on
the general usage of the toolbox and some applications.
