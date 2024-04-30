import numpy as np
from unite_toolbox.bin_estimators import calc_bin_entropy, estimate_ideal_bins
from unite_toolbox.kde_estimators import calc_kde_entropy, calc_ikde_entropy
from unite_toolbox.knn_estimators import calc_knn_entropy

from aux_functions import generate_gexp_samples


data, _ = generate_gexp_samples()


def test_differential_entropy():
    """Test for the implemented methods to estimate differential entropy. Each estimate should be close to the expected
    result give a fixed set of samples from The Gamma-Exponential distribution.
    """
    # Estimates
    bins = estimate_ideal_bins(data, counts=False)["scott"]
    h_bin = sum(calc_bin_entropy(data, bins))
    h_knn = calc_knn_entropy(data)
    h_kde = calc_kde_entropy(data)
    h_ikde = calc_ikde_entropy(data)

    # Assertions
    assert np.isclose(h_bin, 1.84, 0.01)
    assert np.isclose(h_knn, 1.94, 0.01)
    assert np.isclose(h_kde, 2.14, 0.01)
    assert np.isclose(h_ikde, 1.91, 0.01)
