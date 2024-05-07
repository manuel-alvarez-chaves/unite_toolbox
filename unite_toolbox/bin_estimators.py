import numpy as np


def estimate_ideal_bins(
    data: np.ndarray,
    *,
    counts: bool | None = True,
) -> dict[str, int | list[float]]:
    """Estimate the number of ideal bins.

    Estimates the ideal number of bins for each feature (column) of a 2D data
    array using different methods. See numpy.histogram_bin_edges for a list
    of available methods.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    counts : bool, optional
        Whether to return the number of bins (True) or the bin edges (False).

    Returns
    -------
    dict
        A dictionary with a key for each method, and the values are lists of
        number of bins or bin edges for each feature of the data
        (if counts=False).

    """
    _, d_features = data.shape

    methods = ["fd", "scott", "sturges", "doane"]
    ideal_bins = []

    for m in methods:
        d_bins = []
        for d in range(d_features):
            num_bins = np.histogram_bin_edges(data[:, d], bins=m)
            num_bins = len(num_bins) if counts is True else num_bins
            d_bins.append(num_bins)
        ideal_bins.append(d_bins)

    return dict(zip(methods, ideal_bins, strict=True))


def calc_bin_density(
    x: np.ndarray,
    data: np.ndarray,
    edges: list[int | list[float]],
) -> np.ndarray:
    """Calculate density using binning.

    Calculates the density of every point of the 2D array x within the
    d-dimensional histogram created from data and edges.

    Similar to a lookup operation where the entries in x are replaced by the
    bin indices in which they would fall given the binning scheme defined in
    edges. Then the indices are used to "look up" each value of x in the
    d-dimensional histogram created from data and edges.

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d_features)
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    edges : list or int
        A list of length d_features which contains arrays describing the bin
        edges along each dimension or a list of ints describing the number of
        bins to use in each dimension.

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
            np.where((dimbins > 0) & (dimbins < len(edges[i])), dimbins, -9998)
            - 1
        )
        res.append(dimbins)

    # Loop over elements
    indexes = np.column_stack(res)
    fx = np.zeros(shape=(x.shape[0], 1))
    for i, idx in enumerate(indexes):
        if -9999 not in idx:
            fx[i, 0] += fi[tuple(idx)]
    return fx


def calc_vol_array(edges: list[np.ndarray]) -> np.ndarray:
    """Calculate the volume of a multidimensional array.

    Calculates the volume of each cell of the multidimensional array defined
    by `edges` where `edges` is a list of arrays. As an example, if `edges`
    contains two arrays, this functions returns a 2D grid where each element
    in the grid contains the value for the area of that specific cell.
    If `edges` contains three arrays, the returned grid is 3D where each
    element of the grid is the volume of the cell, and so on.

    As this is done to calculate the volume of each bin of a multidimensional
    histogram, the returned grid can be indexed by the same indices as a
    `histogramdd` from NumPy.

    Parameters
    ----------
    edges : list[np.ndarray]
        List of 1D NumPy arrays

    Returns
    -------
    vol : numpy.ndarray
        Array of shape (len(arr0) - 1, len(arr1) - 1, ..., len(arrn) - 1)

    Example
    -------
        >>> a = np.array([0.0, 1.0, 3.0, 7.0, 12.0])
        >>> b = np.array([4.0, 8.0, 10.0])
        >>> calc_vol_array([a, b])
        array([[ 4.,  2.],
               [ 8.,  4.],
               [16.,  8.],
               [20., 10.]])

    """
    vol = np.diff(edges[0])
    for e in edges[1:]:
        vol = np.stack([vol] * (len(e) - 1), axis=-1)
        for idx, val in enumerate(np.diff(e)):
            vol[..., idx] = vol[..., idx] * val
    return vol


def calc_bin_entropy(
    data: np.ndarray,
    edges: list[int | float] | int,
) -> tuple[float]:
    """Calculate entropy using binning.

    Calculates the (joint) entropy of the input data after binning it along
    each dimension using specified bin edges or number of bins.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    edges : list or int
        A list of length d_features which contains arrays describing the bin
        edges along each dimension or a list of ints describing the number of
        bins to use in each dimension. Input can also be a single int and the
        histogram will be created with the same number of bins for each
        dimension.

    Returns
    -------
    h : float
        The (joint) entropy of the input data after binning.
    cf : float
        The correction factor due to bin spacing. See Cover &
        Thomas (2006) Eq. 8.28 ISBN: 978-0-471-24195-9

    """
    # Binning
    f, edges = np.histogramdd(data, bins=edges, density=True)

    # Volume
    volume = calc_vol_array(edges)

    # Entropy
    idx = f.nonzero()
    delta = volume[idx]
    f = f[idx]
    h = -1.0 * np.sum(delta * f * np.log(f))
    corr_fact = -1.0 * np.sum(f * delta * np.log(delta))

    return h, corr_fact


def calc_uniform_bin_entropy(
    data: np.ndarray,
    edges: list[list[float]],
) -> tuple[float]:
    """Alternative method to calculate entropy using binning.

    Calculates the (joint) entropy of the input data. Using this method, every
    data point is substituted by the specific bin it occupies in `edges`.
    Therefore limiting the required memory to only store the number of entries
    in data.

    NOTE: this only works for uniform binning schemes as the correction factor
    for differential entropy is calculated as assuming that every bin is of the
    same size.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    edges : list or int
        A list of length d_features which contains arrays describing the bin
        edges along each dimension

    Returns
    -------
    h : float
        The (joint) entropy of the input data after binning.
    corr_fact : float
        The correction factor due to bin spacing. See Cover &
        Thomas (2006) Eq. 8.28 ISBN: 978-0-471-24195-9

    """
    # Digitize data and get count of unique rows
    data_binned = np.empty(shape=data.shape, dtype=np.int64)
    for idy in range(data.shape[1]):
        data_binned[:, idy] = np.digitize(data[:, idy], edges[idy])
    _, counts = np.unique(data_binned, return_counts=True, axis=0)

    # Calculate the uniform delta
    delta = 1.0
    for e in edges:
        delta *= np.diff(e)[0]  # first item in every edge

    # Calculate density from counts
    density = counts / (data.shape[0] * delta)

    # Calculate entropy and correction factor
    h = -1 * np.sum(delta * density * np.log(density))
    corr_fact = -1 * np.sum(density * delta * np.log(delta))

    return h, corr_fact


def calc_bin_kld(
    p: np.ndarray,
    q: np.ndarray,
    edges: list[list[float]],
) -> float:
    """Calculate Kullback-Leibler divergence (relative entropy) using binning.

    Calculates the Kullback-Leibler divergence (relative entropy) between `p`
    and `q` [in nats] by approximating both distributions using some binning
    scheme defined by `edges`. `edges` *must* be able to support the values in
    `q`.

    Parameters
    ----------
    p : numpy.ndarray
        Array of shape (n_samples, d_features)
    q : numpy.ndarray
        Array of shape (m_samples, d_features)
    edges : list or int
        A list of length d_features which contains arrays describing the bin
        edges along each dimension or a list of ints describing the number of
        bins to use in each dimension.

    Returns
    -------
    kld : float
        Kullback-Leibler divergence between p and q [in nats]

    """
    # Bin according to support of q!
    p_binned = np.empty(shape=p.shape, dtype=np.int64)
    q_binned = np.empty(shape=q.shape, dtype=np.int64)
    for idy in range(q.shape[1]):
        p_binned[:, idy] = np.digitize(p[:, idy], edges[idy])
        q_binned[:, idy] = np.digitize(q[:, idy], edges[idy])

    # Find uniques
    bins_p, counts_p = np.unique(p_binned, return_counts=True, axis=0)
    bins_q, counts_q = np.unique(q_binned, return_counts=True, axis=0)

    set_p = set(tuple(x) for x in bins_p)
    set_q = set(tuple(x) for x in bins_q)
    matching_bins = np.array([x for x in set_p & set_q])

    # Calculate density (here equal to frequency)
    density_p = counts_p / p.shape[0]
    density_q = counts_q / q.shape[0]

    # Evaluate KLD only in matching bins
    kld = 0.0
    for idx in matching_bins:
        a = np.where((bins_p == idx).all(axis=1))[0][0]
        b = np.where((bins_q == idx).all(axis=1))[0][0]
        kld += density_p[a] * np.log(density_p[a] / density_q[b])
    return kld


def calc_bin_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    edges: list[list[float]],
) -> float:
    """Calculate mutual information between `x` and `y` using binning.

    Calculates the mutual information between an array `x` and an array `y`.
    Both `x` and `y` don't necesarily need the same number of samples as
    binning is used. This approach builds multivariate histograms for `x`, `y`
    and `x`-`y` using the specified edges, and evaluates I(X; Y) in every bin
    where the density of `x`-`y` is not zero. This is a resubstitution
    estimate.

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d1_features)
    y : numpy.ndarray
        Array of shape (m_samples, d2_features)
    edges : list
        A list of two lists each containing either integers for
        the number of bins in each axis or arrays of the edges for
        the binning scheme of each axis.

    Returns
    -------
    mi : float
        Mutual information between x and y [in nats]

    """
    _, d1 = x.shape
    _, d2 = y.shape
    data = np.hstack((x, y))
    fxy, joint_edges = np.histogramdd(
        data,
        bins=edges[0] + edges[1],
        density=True,
    )
    fx, _ = np.histogramdd(x, bins=edges[0], density=True)
    fy, _ = np.histogramdd(y, bins=edges[1], density=True)

    volume = calc_vol_array(joint_edges)

    mi = 0.0
    for idx in np.ndindex(fxy.shape):
        if fxy[idx] != 0.0:
            mi += (
                fxy[idx]
                * volume[idx]
                * np.log(fxy[idx] / (fx[idx[:d1]] * fy[idx[-d2:]]))
            )
    return max(0.0, mi)


def calc_qs_entropy(
    sample: np.ndarray,
    alpha: float = 0.25,
    N_k: int = 500,
    seed: int | None = None,
) -> float:
    """Calculate the 1-D entropy of the input data.

    Calculates the 1-D entropy of the input data [in nats] using
    the  quantile spacing (QS) estimator proposed by: Gupta et al.
    (2021) https://doi.org/10.3390/e23060740

    Adapted from: https://github.com/rehsani/Entropy


    Parameters
    ----------
    sample : numpy.ndarray
        Flat array
    alpha : float, optional
        percent of the instances from the sample used for estimation of
        entropy (i.e., number of quantile-spacings).
    N_k : int, optional
        number of sample subsets, used to estimate the sample distribution
        for each quantile empirically
    seed: int, optional
        random seed, if None: default is used

    Returns
    -------
    h : float
        Entropy [in nats] of 'alpha' percent of the instances

    """
    rng = np.random.default_rng(seed=seed)
    sample = sample.flatten()
    n = int(np.ceil(alpha * sample.size))
    x_min = sample.min()
    x_max = sample.max()
    sample.sort()

    sample_b = rng.choice(sample, sample.size, replace=True)
    X_alpha = [
        rng.choice(sample_b[1:-1], n - 1, replace=False) for _ in range(N_k)
    ]
    X_alpha = np.vstack(X_alpha)
    X_alpha.sort(axis=1)
    Z = np.hstack([x_min, X_alpha.mean(axis=0), x_max])
    dZ = np.diff(Z)
    h = 1 / (n + 1) * np.log((n + 1) * dZ).sum()
    return h
