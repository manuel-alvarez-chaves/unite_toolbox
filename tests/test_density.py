from pathlib import Path

import numpy as np

from unite_toolbox.bin_estimators import calc_bin_density, estimate_ideal_bins
from unite_toolbox.kde_estimators import calc_kde_density
from unite_toolbox.knn_estimators import calc_knn_density
from unite_toolbox.utils.bootstrapping import density_bootstrap

data_path = Path("tests/data/test_data.npy")
data = np.load(data_path)
data = data[:, :, 0]
x = np.array([[0.0, 0.0]])

def test_density() -> None:
    """Test density estimation.

    Test for the implemented methods to estimate probability density. Each
    estimate should be close to the expected result given a specific point:
    [0.0, 0.0].
    """

    # Estimates
    ideal_bins = estimate_ideal_bins(data, counts=False)["scott"]
    px_bin = calc_bin_density(x, data, ideal_bins)
    px_kde = calc_kde_density(x, data)
    px_knn = calc_knn_density(x, data)

    # Assertions
    assert np.isclose(px_bin, 0.26, atol=0.01)
    assert np.isclose(px_kde, 0.33, atol=0.01)
    assert np.isclose(px_knn, 0.07, atol=0.01)

def test_density_bootstrap() -> None:
    """Test for one sample density bootstrap.

    Test for the implementation of bootstrapping for one sample density. Uses
    the k-NN estimator to check if the option to add noise is also working
    correctly, although any estimator could be used.
    """
    # Estimate
    bs_px, bs_px_ci = density_bootstrap(
        x, data, calc_knn_density, 100, 0.05, 42, add_noise=True, k=3
        )
    
    # Assertions
    assert np.isclose(bs_px, 0.05, atol=0.01)
    assert np.isclose(bs_px_ci[0], 0.03, atol=0.01)
    assert np.isclose(bs_px_ci[1], 0.09, atol=0.01)

