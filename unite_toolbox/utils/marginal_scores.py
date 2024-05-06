import numpy as np
from itertools import chain, combinations


def power_set(items: list) -> list[tuple]:
    """Create power set.

    Creates a list of tuples which are the power set of the input items.
    Based on the itertools recipe:
    https://docs.python.org/3/library/itertools.html#itertools-recipes

    Parameters
    ----------
    items : list
        List of items

    Returns
    -------
    power_set : list[tuples]
        List of tuples containing each element of the power set
    """
    power_set = chain.from_iterable(
        combinations(items, r) for r in range(len(items) + 1)
    )
    return list(power_set)


def calc_marginal_scores(scores: dict) -> dict:
    """Calculate marginal scores due to adding more features.

    (blah)
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
