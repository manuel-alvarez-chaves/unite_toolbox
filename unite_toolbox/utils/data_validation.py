from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def validate_array(arr: ArrayLike) -> np.ndarray:
    """Validate array.

    Simple function to validate the dimensions and transform into a type
    suitable for usage in the UNITE toolbox.

    Parameters
    ----------
    arr : array_like
        Array like object that can be transformed into a 2d Numpy array

    Returns
    -------
    arr : numpy.ndarray
        Array of shape (n_samples, d_features)

    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        err = "UNITE only uses 2d arrays!"
        raise TypeError(err)
    return arr


def find_repeats(data: np.ndarray) -> np.ndarray:
    """Find repeats.

    Returns a boolean mask for repeat rows in `data` where True is a repeated
    row.

    Parameters
    ----------
    data : array_like
        2D array like of shape (n_samples, d_features)

    Returns
    -------
    mask : numpy.ndarray
        Boolean array of shape (n_samples,)

    """
    _, inv, counts = np.unique(
        data, return_inverse=True, return_counts=True, axis=0,
    )
    mask = np.where(counts[inv] > 1, True, False)
    return mask


def add_noise_to_data(data: np.ndarray) -> np.ndarray:
    """Add noise to repeated rows in `data`.

    Adds Gaussian noise to only the repeated rows in a 2D array.
    The noise added is one order of magnitude below the order of magnitude of
    the std. dev. of each specific column in the data. This was empirically
    determined to be adequate for distance based measures.

    Parameters
    ----------
    data : array_like
        2D array like of shape (n_samples, d_features)

    Returns
    -------
    noisy_data : numpy.ndarray
        2D array of shape (n_samples, d_features)

    """
    data = np.asarray(data, dtype=np.float64)
    _, d = data.shape

    # Determine the scale of the noise
    data_std = np.std(data, axis=0).reshape(1, -1) / 10
    noise_scale = 10 ** (np.floor(np.log10(np.abs(data_std))))

    # Generate only the required noise for the data
    mask = find_repeats(data)
    noise = stats.multivariate_normal.rvs(
        cov=np.diag(noise_scale.flatten()), size=mask.sum(), random_state=42,
    ).reshape(-1, d)

    # Add noise to specific rows
    noisy_data = data.copy()
    noisy_data[mask] += noise

    return noisy_data


def valida_data_kld(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray]:
    """Validate data for kNN-based KLD.

    Eliminates repeated values from a and the joint array a-b to perform a
    distance based calculation of KLD, or other method which requires only
    unique values in two arrays.

    The code finds repeats in `a`, deletes them, and adds back the unique
    values that were repreated so that not a lot of the data is lost. A
    similar procedure is applied to `b` by first concatenating it with `a`
    so that no value in `a` is repeated in `b`.

    Parameters
    ----------
    a : numpy.ndarray
        2D array of shape (n_samples, d_features)
    b : numpy.ndarray
        2D array of shape (m_samples, d_features)

    Returns
    -------
    p : numpy.ndarray
        2D array of shape (<=n_samples, d_features)
    q : numpy.ndarray
        2D array of shape (<=m_samples, d_features)

    """
    # Clean a
    mask_a = np.argwhere(find_repeats(a)).flatten()
    repeats = a[mask_a]
    p = np.delete(a, mask_a, axis=0)
    unique_repeats = np.unique(repeats, axis=0)
    p = np.vstack((p, unique_repeats))

    # Clean b
    nrows, _ = p.shape
    aux = np.vstack((p, b))
    mask_b = find_repeats(aux)[nrows:]
    q = b[~mask_b]

    return p, q
