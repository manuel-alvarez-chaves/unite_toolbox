import numpy as np

from tests.aux_functions import generate_samples
from unite_toolbox.bin_estimators import calc_bin_entropy, estimate_ideal_bins
from unite_toolbox.kde_estimators import calc_ikde_entropy, calc_kde_entropy
from unite_toolbox.knn_estimators import calc_knn_entropy

data, _ = generate_samples()


def test_differential_entropy() -> None:
    """Test differential entropy.

    Test for the implemented methods to estimate differential entropy. Each
    estimate should be close to the expected result give a fixed set of
    samples from the Gamma-Exponential distribution.
    """
    # Estimates
    bins = estimate_ideal_bins(data, counts=False)["scott"]
    h_bin, _ = calc_bin_entropy(data, bins)
    h_knn = calc_knn_entropy(data)
    h_kde = calc_kde_entropy(data)
    h_ikde = calc_ikde_entropy(data)

    # Assertions
    assert np.isclose(h_bin,  1.42, 0.01)
    assert np.isclose(h_knn,  1.97, 0.01)
    assert np.isclose(h_kde,  1.62, 0.01)
    assert np.isclose(h_ikde, 1.63, 0.01)
