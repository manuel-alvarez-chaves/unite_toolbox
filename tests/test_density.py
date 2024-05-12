import numpy as np

from unite_toolbox.bin_estimators import calc_bin_density, estimate_ideal_bins
from unite_toolbox.kde_estimators import calc_kde_density
from unite_toolbox.knn_estimators import calc_knn_density

data = np.load("tests/data/test_data.npy")
data = data[:, :, 0]
x = np.array([[0.0, 0.0]])

def test_density() -> None:
    # Estimates
    ideal_bins = estimate_ideal_bins(data, counts=False)["scott"]
    px_bin = calc_bin_density(x, data, ideal_bins)
    px_kde = calc_kde_density(x, data)
    px_knn = calc_knn_density(x, data)

    # Assertions
    assert np.isclose(px_bin, 0.26, atol=0.01)
    assert np.isclose(px_kde, 0.33, atol=0.01)
    assert np.isclose(px_knn, 0.07, atol=0.01)
