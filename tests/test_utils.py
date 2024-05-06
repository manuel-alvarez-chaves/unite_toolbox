import numpy as np

from tests.aux_functions import generate_samples, rng
from unite_toolbox.utils.data_validation import (
    add_noise_to_data,
    find_repeats,
    validate_array,
)
from unite_toolbox.utils.marginal_scores import calc_marginal_scores, power_set

samples, _ = generate_samples()
labels = ["a", "b", "c"]


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
