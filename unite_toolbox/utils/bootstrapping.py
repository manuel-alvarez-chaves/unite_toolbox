from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm import trange

from unite_toolbox.utils.data_validation import add_noise_to_data


def density_bootstrap(
    x: np.ndarray,
    data: np.ndarray,
    estimator: Callable,
    n_bootstraps: int,
    significance: float,
    seed: int | None = None,
    *,
    add_noise: bool = False,
    **kwargs: dict[str, Any],
) -> tuple[float, list[float]]:
    """Calculate density with bootstrap confidence intervals.

    Calculates density at `x` with confidence intervals defined by
    `significance`. An `estimator` to calculate density has to be
    passed as a callable function. `add_noise` is required for a *k*-NN
    based estimator.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (n_samples, d_features)
    data : np.ndarray
        Array of shape (n_samples, d_features)
    estimator : Callable function
        Density estimator function
    n_bootstraps : int
        No. of bootstraps to perform
    significance : float
        Statistical significance for confidence intervals
    seed : int, optional
        Seed for random number generator
    add_noise : bool, optional
        Flag to add noise to data if required
    **kwargs: dict[str, Any]
        Keyword arguments for the estimator

    Returns
    -------
    bs_mean : float
        Mean density from the bootstrap
    bs_ci : list[float]
        Lower and upper quantiles of the bootstrap

    """
    rng = np.random.default_rng(seed)
    n, _ = data.shape
    res = np.empty(shape=(x.shape[0], 1, n_bootstraps))
    for i in trange(int(n_bootstraps), ascii=True, unit="boot"):
        sub_idx = rng.choice(n, size=n, replace=True)
        subsample = data[sub_idx, :].copy()
        subsample = add_noise_to_data(subsample) if add_noise else subsample
        res[:, :, i] = estimator(x, subsample, **kwargs).reshape(-1, 1)
    bs_mean = res.mean(axis=2)
    bs_ci = np.quantile(
        res,
        [significance / 2, 1 - (significance / 2)],
        axis=2,
    )
    return bs_mean, bs_ci


def one_sample_bootstrap(
    data: np.ndarray,
    estimator: Callable,
    n_bootstraps: int,
    significance: float,
    seed: int | None = None,
    *,
    add_noise: bool = False,
    **kwargs: dict[str, Any],
) -> tuple[float, list[float]]:
    """Calculate entropy with bootstrap confidence intervals.

    Calculates a bootstrap result of the `estimator` with confidence intervals
    defined by `significance`. `add_noise` is required for a *k*-NN. The
    `estimator` must applicable to only one sample, i.e., `data`.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, d_features)
    estimator : Callable function
        Density estimator function
    n_bootstraps : int
        No. of bootstraps to perform
    significance : float
        Statistical significance for confidence intervals
    seed : int, optional
        Seed for random number generator
    add_noise : bool, optional
        Flag to add noise to data if required
    **kwargs: dict[str, Any]
        Keyword arguments for the estimator

    Returns
    -------
    bs_mean : float
        Mean density from the bootstrap
    bs_ci : list[float]
        Lower and upper quantiles of the bootstrap

    """
    rng = np.random.default_rng(seed)
    n, _ = data.shape
    res = np.empty(n_bootstraps)
    for i in trange(int(n_bootstraps), ascii=True, unit="boot"):
        sub_idx = rng.choice(n, size=n, replace=True)
        subsample = data[sub_idx, :].copy()
        subsample = add_noise_to_data(subsample) if add_noise else subsample
        res[i] = estimator(subsample, **kwargs)
    bs_mean = res.mean()
    bs_ci = np.quantile(res, [significance / 2, 1 - (significance / 2)])
    return bs_mean, bs_ci
