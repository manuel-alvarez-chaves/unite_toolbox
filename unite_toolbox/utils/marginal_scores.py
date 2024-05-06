from collections.abc import Iterable
from itertools import chain, combinations
from typing import Any

import numpy as np


def power_set(items: Iterable) -> list[tuple]:
    """Create power set.

    Creates a list of tuples which are the power set (all possible
    combinations) of the input items.
    Based on the itertools recipe:
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    Parameters
    ----------
    items : Iterable
        Iterable object containing individual labels

    Returns
    -------
    power_set : list[tuples]
        List of tuples containing each element of the power set

    """
    s = list(items)
    power_set = chain.from_iterable(
        combinations(s, r) for r in range(len(s) + 1)
    )
    return list(power_set)


def calc_marginal_scores(scores: dict[tuple, float]) -> dict[Any, float]:
    """Calculate marginal scores due to adding more features.

    Calculates the marginal score of adding a specific feature to a callable
    that generates a score. That callable can be a function (e.g. mutual
    information) or a model. The result is a marginal gain (or loss) of
    adding an individual feature as input. The combinations of features are
    the tuples and each tuple has a specific score associated. The result is
    a dict of individual features as keys and their marginal score as values.

    Parameters
    ----------
    scores : dict
        Dict. of tuples of feature combinations and individual scores
        i.e. {(): 0.0, ("a"): 0.3, ("b"): 0.5, ("a", "b"): 0.8}

    Returns
    -------
    marginal_scores : dict
        Dict. of individual labels and their marginal scores
        i.e. {"a": 0.15, "b": 0.25}

    """
    # Preparation data
    combs = list(scores.keys())
    features = max(scores, key=len)
    n = len(features)
    sizes = [len(comb) for comb in combs]
    counts = np.bincount(sizes)
    weight_factors = [counts[:-1][i] * (n - i) for i in range(n)]

    # Calculate marginal scores
    marginal_scores = {}
    for feat in features:
        combs_with_feat = [item for item in combs if feat in item]
        marginal_score = 0.0
        for comb in combs_with_feat:
            factor = 1 / weight_factors[len(comb) - 1]
            origin = tuple(item for item in comb if item != feat)
            if origin in scores:
                marginal_score += factor * (scores[comb] - scores[origin])
        marginal_scores[feat] = marginal_score

    return marginal_scores
