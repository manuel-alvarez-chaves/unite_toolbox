from pathlib import Path

import numpy as np

from unite_toolbox.bin_estimators import (
    calc_bin_mutual_information,
    estimate_ideal_bins,
)
from unite_toolbox.kde_estimators import calc_kde_mutual_information
from unite_toolbox.knn_estimators import calc_knn_mutual_information

data_path = Path("tests/data/test_data.npy")
data = np.load(data_path)
samples1, samples2 = data[:, :, 0], data[:, :, 1]

def test_mutual_information() -> None:
    """Test mutual information.

    Test for the implemented methods to estimate mutual information. Each
    estimate should be close to the expected result given a fixed set of
    samples.
    """
    # Estimates
    bins1 = estimate_ideal_bins(samples1, counts=False)["scott"]
    bins2 = estimate_ideal_bins(samples2, counts=False)["scott"]
    mi_bin = calc_bin_mutual_information(samples1, samples2, [bins1, bins2])
    mi_kde = calc_kde_mutual_information(samples1, samples2)
    mi_ikde = calc_kde_mutual_information(
        samples1[:, [0]], samples2[:, [1]], mode="integral"
    )
    mi_knn = calc_knn_mutual_information(samples1, samples2, k=110)


    # Assertions
    assert np.isclose(mi_bin, 2.07, atol=0.01)
    assert np.isclose(mi_kde, 0.41, atol=0.01)
    assert np.isclose(mi_ikde, 0.02, atol=0.01)
    assert np.isclose(mi_knn, 0.08, atol=0.01)