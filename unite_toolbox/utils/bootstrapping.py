import numpy as np
from numpy.typing import ArrayLike, NDArray

from scipy import stats
from tqdm import trange


def find_repeats(data: ArrayLike) -> NDArray:
    """Returns a boolean mask for repeat rows in data
    where True is a repeated row.

    Parameters
    ----------
    data : array_like
        2D array like of shape (n_samples, d_features)

    Returns
    -------
    mask : numpy.ndarray
        Boolean array of shape (n_samples,)"""

    data = np.asarray(data, dtype=np.float64)
    _, inv, counts = np.unique(data, return_inverse=True, return_counts=True, axis=0)
    mask = np.where(counts[inv] > 1, True, False)
    return mask


def add_noise_to_data(data: ArrayLike) -> NDArray:
    """Adds noise to repeated rows in data.

    Adds Gaussian noise to only the repeated rows in a 2D array.
    The noise added is one order of magnitude below the order of magnitude
    of the std. dev. of each specific column in the data. This was empirically
    determined to be adequate for distance based measures.

    Parameters
    ----------
    data : array_like
        2D array like of shape (n_samples, d_features)

    Returns
    -------
    noisy_data : numpy.ndarray
        2D array of shape (n_samples, d_features)"""

    data = np.asarray(data, dtype=np.float64)
    _, d = data.shape

    # Determine the scale of the noise
    data_std = np.std(data, axis=0).reshape(1, -1) / 10
    noise_scale = 10 ** (np.floor(np.log10(np.abs(data_std))))

    # Generate only the required noise for the data
    mask = find_repeats(data)
    noise = stats.multivariate_normal.rvs(cov=np.diag(noise_scale.flatten()), size=mask.sum(), random_state=42).reshape(-1, d)

    # Add noise to specific rows
    noisy_data = data.copy()
    noisy_data[mask] += noise

    return noisy_data


def density_bootstrap(x, data, estimator, n_bootstraps, significance, add_noise=False, **kwargs):
    n, d = data.shape
    res = np.empty(shape=(x.shape[0], x.shape[1], n_bootstraps))
    for i in trange(int(n_bootstraps), ascii=True, unit="boot"):
        sub_idx = np.random.choice(n, size=n, replace=True)
        subsample = data[sub_idx, :].copy()
        subsample = add_noise_to_data(subsample) if add_noise else subsample
        res[:, :, i] = estimator(x, subsample, **kwargs).reshape(-1, d)
    bs_mean = res.mean(axis=2)
    bs_ci = np.quantile(res, [significance / 2, 1 - (significance / 2)], axis=2)
    return bs_mean, bs_ci


def one_sample_bootstrap(data, estimator, n_bootstraps, significance, add_noise=False, **kwargs):
    n, _ = data.shape
    res = np.empty(n_bootstraps)
    for i in trange(int(n_bootstraps), ascii=True, unit="boot"):
        sub_idx = np.random.choice(n, size=n, replace=True)
        subsample = data[sub_idx, :].copy()
        subsample = add_noise_to_data(subsample) if add_noise else subsample
        res[i] = estimator(subsample, **kwargs)
    bs_mean = res.mean()
    bs_ci = np.quantile(res, [significance / 2, 1 - (significance / 2)])
    return bs_mean, bs_ci
