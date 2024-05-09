import numpy as np

from tests.aux_functions import generate_samples
from unite_toolbox.bin_estimators import (
    calc_bin_entropy,
    calc_qs_entropy,
    calc_uniform_bin_entropy,
    estimate_ideal_bins,
)
from unite_toolbox.kde_estimators import calc_kde_entropy
from unite_toolbox.knn_estimators import calc_knn_entropy
from unite_toolbox.utils.bootstrapping import one_sample_bootstrap

data, _ = generate_samples()


def test_differential_entropy() -> None:
    """Test differential entropy.

    Test for the implemented methods to estimate differential entropy. Each
    estimate should be close to the expected result give a fixed set of
    samples from the Normal distribution.
    """
    # Estimates
    h_nonuniform_bin, _ = calc_bin_entropy(data, [10, 10])
    bins = estimate_ideal_bins(data, counts=False)["scott"]
    h_uniform_bin, _ = calc_uniform_bin_entropy(data, bins)
    h_qs = calc_qs_entropy(data, seed=42)
    h_knn = calc_knn_entropy(data)
    h_kde = calc_kde_entropy(data)
    h_ikde = calc_kde_entropy(data, mode="integral")

    # Assertions
    assert np.isclose(h_nonuniform_bin, 1.31, atol=0.01)
    assert np.isclose(h_uniform_bin, 1.44, atol=0.01)
    assert np.isclose(h_qs, 0.79, atol=0.01)
    assert np.isclose(h_knn, 1.97, atol=0.01)
    assert np.isclose(h_kde, 1.62, atol=0.01)
    assert np.isclose(h_ikde, 1.63, atol=0.01)


def test_one_sample_bootstrap() -> None:
    """Test for one sample bootstrap.

    Test for the implementation of bootstrapping for one sample (typically
    for entropy). Uses the kNN estimator to check if the option to add noise
    is also working correctly, although any estimator could be used.
    """
    bs_h, bs_h_ci = one_sample_bootstrap(
        data, calc_knn_entropy, 100, 0.05, 42, add_noise=True, k=3,
    )
    assert np.isclose(bs_h, 1.47, atol=0.01)
    assert np.isclose(bs_h_ci[0], 1.24, atol=0.01)
    assert np.isclose(bs_h_ci[1], 1.73, atol=0.01)
