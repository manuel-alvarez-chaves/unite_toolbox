import numpy as np

def get_samples(func, limits, n_samples, **kwargs):
    """
    """
    
    ids = []
    acc_rate = 1.0
    while len(ids) < n_samples:
        d = len(limits)
        f = np.array(limits)[:, 0] # floor
        s = np.array(limits)[:, 1] - f # scale
        r = np.random.uniform(size=(int(n_samples / acc_rate), d))
        r = f + s * r
            
        F = func(*(np.hsplit(r, d)), **kwargs).flatten()
        G = 1 / np.prod(s)
        M = F.max() / G

        U = np.random.uniform(0.0, 1.0, size=F.shape)
        ids = np.argwhere(U < F/(M * G)).flatten()
        acc_rate = acc_rate * (len(ids) / n_samples)

    samples = r[ids][:n_samples, :]
    return samples


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
    factors = [counts[:-1][id]*(N-id) for id in range(N)]

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