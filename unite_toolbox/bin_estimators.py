import numpy as np


def estimate_ideal_bins(data, counts=True):
    """
    Estimates the ideal number of bins for each column of a 2D data array using
    three different methods: Scott, Freedman-Diaconis, and Sturges.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, n_features)
    counts : bool, optional
        Whether to return the number of bins (True) or the bin edges (False).

    Returns
    -------
    dict
        A dictionary with a key for each method, and the values are lists of
        number of bins or bin edges for each feature of the data (if counts=False).
    """

    _, n_features = data.shape

    methods = ["scott", "fd", "sturges"]
    ideal_bins = []

    for m in methods:
        d_bins = []
        for d in range(n_features):
            num_bins = np.histogram_bin_edges(data[:, d], bins=m)
            num_bins = len(num_bins) if counts is True else num_bins
            d_bins.append(num_bins)
        ideal_bins.append(d_bins)

    return dict(zip(methods, ideal_bins))


def calc_bin_density(x, data, edges):
    """
    Calculates the density of every point of the 2D array x within the d-dimensional
    histogram created from data and edges.
    
    Similar to a lookup operation where the
    entries in x are replaced by the bin indices in which they would fall given the
    binning scheme defined in edges. Then the indices are used to "look up" each value
    of x in the d-dimensional histogram created from data and edges.

    arameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d_features)
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    edges : list or int
        A list of length d_features which contains arrays describing the bin edges
        along each dimension or a list of ints describing the number of bins to use
        in each dimension.

    Returns
    -------
    fx : numpy.ndarray
        Array of shape (n_samples, 1)
    """
    fi, edges = np.histogramdd(data, edges, density=True)
    res = []
    # Loop over dimensions
    for i in range(len(edges)):
        dimbins = np.digitize(x[:, i], edges[i])
        dimbins = (
            np.where((dimbins > 0) & (dimbins < len(edges[i])), dimbins, -9998) - 1
        )
        res.append(dimbins)

    # Loop over elements
    indexes = np.vstack(res).T
    fx = np.zeros(shape=(x.shape[0], 1))
    for i, idx in enumerate(indexes):
        if -9999 in idx:
            pass
        else:
            fx[i, 0] += fi[tuple(idx)]
    return fx


def calc_vol_array(edges):
    """ """

    res = edges[0]
    for e in edges[1:]:
        res = np.stack([res] * len(e), axis=-1)
        for id, val in enumerate(e):
            res[..., id] = res[..., id] * val
    return res


def calc_bin_entropy(data, edges):
    """
    Calculates the (joint) entropy of the input data after binning it along each
    dimension using specified bin edges or number of bins.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    edges : list or int
        A list of length n_features which contains arrays describing the bin edges
        along each dimension or a list of ints describing the number of bins to use
        in each dimension. Input can also be a single int and the histogram will be
        created with the same number of bins for each dimension.

    Returns
    -------
    h : float
        The (joint) entropy of the input data after binning.
    cf : float
        The correction factor due to bin spacing. See Cover &
        Thomas (2006) Eq. 8.28 ISBN: 978-0-471-24195-9
    """

    # binning
    fi, edges = np.histogramdd(data, bins=edges, density=True)

    # volume
    edges = [np.diff(e) for e in edges]
    volume = calc_vol_array(edges)

    # entropy
    ids = fi.nonzero()
    delta = volume[ids]
    fi = fi[ids]
    h = -1.0 * np.sum(delta * fi * np.log(fi * delta))
    cf = np.sum(fi * delta * np.log(delta))

    return h, cf


def calc_bin_kld(data_f, data_g, edges):
    """ """

    fi, _ = np.histogramdd(data_f, bins=edges, density=True)
    gi, edges = np.histogramdd(data_g, bins=edges, density=True)

    edges = [np.diff(e) for e in edges]
    volume = calc_vol_array(edges)

    ids = fi.nonzero()
    fi, gi, delta = fi[ids], gi[ids], volume[ids]
    ids = gi.nonzero()
    fi, gi, delta = fi[ids], gi[ids], delta[ids]

    kld = np.sum(fi * delta * np.log(fi / gi))

    return kld


def calc_bin_mutual_information(data, edges):
    _, d = data.shape

    data_x = data[:, : d - 1].reshape(-1, d - 1)
    edges_x = edges[: d - 1]
    data_y = data[:, d - 1].reshape(-1, 1)
    edges_y = edges[-1]

    fxy, bin_edges = np.histogramdd(data, bins=edges, density=True)
    fx, _ = np.histogramdd(data_x, bins=edges_x, density=True)
    fy, _ = np.histogram(data_y, bins=edges_y, density=True)

    bin_edges = [np.diff(e) for e in bin_edges]
    volume = calc_vol_array(bin_edges)

    mi = 0.0
    for idxy in np.ndindex(fxy.shape):
        if fxy[idxy] != 0.0:
            mi += (
                fxy[idxy]
                * volume[idxy]
                * np.log(fxy[idxy] / (fx[idxy[: d - 1]] * fy[idxy[-1]]))
            )

    return max(0, mi)


def calc_qs_entropy(sample, alpha=0.25, N_k=500):
    """Calculates the 1-D entropy of the input data.

    Calculates the 1-D entropy of the input data [in nats] using
    the  quantile spacing (QS) estimator proposed by: Gupta et al.
    (2021) https://doi.org/10.3390/e23060740

    Adapted from: https://github.com/rehsani/Entropy


    Parameters
    ----------
    data : numpy.ndarray
        Flat array
    alpha : float, optional
        percent of the instances from the sample used for estimation of
        entropy (i.e., number of quantile-spacings).
    N_k : int, optional
        number of sample subsets, used to estimate the sample distribution
        for each quantile empirically

    Returns
    -------
    h : float
        Entropy [in nats] of 'alpha' percent of the instances"""

    sample = sample.flatten()
    n = int(np.ceil(alpha * sample.size))
    x_min = sample.min()
    x_max = sample.max()
    sample.sort()

    sample_b = np.random.choice(sample, sample.size, replace=True)
    X_alpha = [
        np.random.choice(sample_b[1:-1], n - 1, replace=False) for _ in range(N_k)
    ]
    X_alpha = np.vstack(X_alpha)
    X_alpha.sort(axis=1)
    Z = np.hstack([x_min, X_alpha.mean(axis=0), x_max])
    dZ = np.diff(Z)
    h = 1 / (n + 1) * np.log((n + 1) * dZ).sum()
    return h
