import numpy as np

from tests.aux_functions import generate_samples, pdf_mnorm, rng
from unite_toolbox.bin_estimators import calc_vol_array, estimate_ideal_bins
from unite_toolbox.knn_estimators import vol_lp_ball
from unite_toolbox.utils.data_validation import (
    add_noise_to_data,
    find_repeats,
    validate_array,
)
from unite_toolbox.utils.marginal_scores import calc_marginal_scores, power_set
from unite_toolbox.utils.sampling import get_samples

samples, _ = generate_samples()
labels = ["a", "b", "c"]


def test_ideal_bins() -> None:
    """Test for rules-of-thumb.

    Checks the estimated rules-of-thumb against previously calculated number
    of bins for a fixed data set.
    """
    rules = ["fd", "scott", "sturges", "doane"]  # numpy.histogram_bin_edges
    true_ideal_bins = [[11, 10], [9, 8], [9, 9], [10, 10]]
    res_dict = {r: b for (r, b) in zip(rules, true_ideal_bins, strict=True)}
    ideal_bins = estimate_ideal_bins(samples.copy())
    assert ideal_bins == res_dict


def test_vol_array() -> None:
    """Test for area (volume) of non-uniform array.

    Calculates the area of each cell in a 2D array defined by the location of
    its bin edges. Compares the result against the manually computed bin areas.
    """
    edges = [[0.0, 1.0, 3.0, 7.0, 12.0], [4.0, 8.0, 10.0]]
    area_array = calc_vol_array(edges)
    true_area_array = np.array(
        [[4.0, 2.0], [8.0, 4.0], [16.0, 8.0], [20.0, 10.0]]
    )
    assert (area_array == true_area_array).all()


def test_vol_ball() -> None:
    """Test for volume of lp ball.

    Calculates the volume of two lp balls using different norms and compares
    against the theoretical result.
    """
    radius = 2.0
    assert np.isclose(vol_lp_ball(radius, d=2, p_norm=2), 12.57, atol=0.01)
    assert np.isclose(vol_lp_ball(radius, d=3, p_norm=np.inf), 64.0, atol=0.01)


def test_validate_array() -> None:
    """Test for basic array validation.

    Creates a copy of the `samples` array and transforms it into a list to test
    the input as an ArrayLike in the validate_array function.
    """
    arr = samples.copy().tolist()
    unite_arr = validate_array(arr)
    assert type(unite_arr) == np.ndarray
    assert unite_arr.ndim == 2


def test_repeats() -> None:
    """Test for finding repeated rows in array.

    Repeats 5 random rows in the `samples` array and tests that the
    `find_repeats` function finds the correct row indices for the repeated
    rows.
    """
    arr = samples.copy()
    repeats = rng.choice(range(1, samples.shape[0]), size=5, replace=False)
    arr[repeats] = arr[0]
    repeat_matches = find_repeats(arr).nonzero()[0][1:]
    assert np.array_equal(sorted(repeat_matches), sorted(repeats))


def test_add_noise_to_samples() -> None:
    """Test for adding noise to repeated rows.

    Repeats 4 random rows in the `samples` array, adds noise to the repeated
    rows and creates a new array in which noise has been added to the repeated
    rows. Checks if the number of unique rows in the `noisy_arr` is the same as
    in the original array.
    """
    arr = samples.copy()
    repeats = rng.choice(range(1, samples.shape[0]), size=5, replace=False)
    arr[repeats] = arr[0]
    noisy_arr = add_noise_to_data(arr)
    assert np.unique(noisy_arr, axis=0).shape[0] == samples.shape[0]


def test_power_set() -> None:
    """Test for creating a power set.

    Creates a power set for a predefined list of labels.
    """
    res = [
        (),
        ("a",),
        ("b",),
        ("c",),
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("a", "b", "c"),
    ]
    assert power_set(labels) == res


def test_marginal_scores() -> None:
    """Test for calculating a marginal score.

    Calculates marginal scores for a predefined list of labels and given scores
    for combinations of them.
    """
    scores = [0.00, 0.20, 0.25, 0.10, 0.70, 0.50, 0.55, 1.00]
    scores = dict(zip(power_set(labels), scores, strict=True))
    true_marginal_scores = {"a": 0.36, "b": 0.41, "c": 0.23}
    marginal_scores = calc_marginal_scores(scores)
    for feature in labels:
        assert np.isclose(
            marginal_scores[feature], true_marginal_scores[feature], atol=0.005
        )


def test_get_samples() -> None:
    """Test for rejection sampling.

    Obtains samples from a predefined 2D normal mixture distribution and checks
    that they are the same as a fixed set of samples.
    """
    mnorm1_params = [
        [[-2, 0], [[1, -0.5], [-0.5, 1]], 0.5],
        [[2, 0], [[1, 0.5], [0.5, 1]], 0.5],
    ]
    mnorm_lims = [[-10, 10], [-10, 10]]

    samples = get_samples(
        func=pdf_mnorm,
        limits=mnorm_lims,
        n_samples=10,
        seed=42,
        params=mnorm1_params,
    )
    fixed_samples = np.array(
        [
            [-1.92, -0.52],
            [-2.79, 0.25],
            [1.73, -0.13],
            [-2.67, 0.79],
            [1.81, -2.06],
            [-0.90, 0.32],
            [-1.61, 0.29],
            [0.40, -1.84],
            [-2.42, 0.34],
            [2.34, 0.85],
        ]
    )
    tol = 0.01
    assert (np.abs(samples - fixed_samples) < tol).all()
