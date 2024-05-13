from pathlib import Path

import numpy as np

from tests.aux_functions import rng
from unite_toolbox.bin_estimators import calc_bin_kld, estimate_ideal_bins
from unite_toolbox.kde_estimators import calc_kde_kld
from unite_toolbox.knn_estimators import calc_knn_kld
from unite_toolbox.utils.data_validation import validate_data_kld

data_path = Path("tests/data/test_data.npy")
data = np.load(data_path)
samples1, samples2 = data[:, :, 0], data[:, :, 1]

def test_validate_data_kld() -> None:
    """Test to validate arrays for distance-based KLD estimation.

    Checks that the implemented function to validate the arrays for
    KLD estimation using k-NN correctly finds repeats in both of the arrays
    being validated.
    """
    # Copy data
    a = samples1.copy()
    b = samples2.copy()

    # Repeat values
    repeats = rng.choice(range(1, samples1.shape[0]), size=5, replace=False)
    a[repeats] = a[0] # 5 repeats
    b[repeats[0]] = b[0] # 1 repeat from `a`

    # Assert only unique values in both arrays
    p, q = validate_data_kld(a, b)
    assert np.unique(p, axis=0).shape == p.shape
    assert np.unique(q, axis=0).shape == q.shape

def test_differential_entropy() -> None:
    """Test for KLD.

    Test for the implemented methods to estimate Kullback-Leibler divergence.
    Each estimate should be close to the expected result given a fixed set of
    samples from known Normal distributions.
    """
    # Estimates
    bins = estimate_ideal_bins(samples2, counts=False)["scott"]
    kld_bin = calc_bin_kld(samples1, samples2, bins)
    kld_kde = calc_kde_kld(samples1, samples2)
    kld_ikde = calc_kde_kld(samples1, samples2, mode="integral")
    kld_knn = calc_knn_kld(samples1, samples2)

    # Assertions
    assert np.isclose(kld_bin, 0.66, atol=0.01)
    assert np.isclose(kld_kde, 1.51, atol=0.01)
    assert np.isclose(kld_ikde, 0.52, atol=0.01)
    assert np.isclose(kld_knn, 0.18, atol=0.01)