import numpy as np


def power_set(items):
    """Creates a list of lists which are the power set of the input
    items.

    Parameters
    ----------
    items : list
        List of items

    Returns
    -------
    subsets : list
        List of list containing each element of the power set"""

    N = len(items)
    if N == 0:
        return [[]]

    subsets = []
    for subset in power_set(items[1:]):
        subsets.append(subset)
        subsets.append(subset[:] + [items[0]])

    return subsets


def calc_marginal_information(models):
    """(placeholder)

    Parameters
    ----------
    models : dictionary
        (placeholder)

    Returns
    -------
    mi : dict
        (placeholder)"""

    combs = [eval(key) for key in models.keys()]
    features = max(combs, key=len)
    N = len(features)
    lengths = [len(item) for item in combs]
    counts = np.bincount(lengths)
    factors = [counts[:-1][id] * (N - id) for id in range(N)]

    mi = {}
    for feat in features:
        operations = [item for item in combs if feat in item]
        sv = 0
        for item in operations:
            factor = 1 / factors[len(item) - 1]
            bottom_name = str(item)
            top_name = str([element for element in item if element != feat])
            sv += factor * (models[bottom_name] - models[top_name])
        mi[feat] = sv

    return mi
