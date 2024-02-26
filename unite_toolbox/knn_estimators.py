import numpy as np
from scipy.spatial import KDTree
from scipy.special import gamma, digamma


def vol_lp_ball(r, d, p_norm):
    r"""Calculates the volume of a :math:`L^p` ball of radius *R* in d-
    dimensional space.

    .. math::
         V_d^p(R) = \frac{(2\,\Gamma(\frac{1}{p}+1))^d}{\Gamma(\frac{d}{p}+1)}\,R^d

    Where:
        * :math:`\Gamma` is the Gamma function.
        * :math:`p` is the order of the Minkowski distance.
        * :math:`d` is the number of dimensions of the d-ball.
        * :math:`R` is the radius of the d-ball.

    Parameters
    ----------
    r : float
        radius of the d-ball
    d : int
        dimension of the d-ball
    p_norm : int
        p (Minkowski) distance. p = 1 is the Manhattan distance,
        p = 2 is the Euclidian distance, etc.

    Returns
    -------
    vol : float
        Volume of the L^p ball"""

    if p_norm == np.inf:  # faster
        vol = (2**d) * (r**d)
        return vol

    a = (2 * gamma(1 / p_norm + 1)) ** d
    b = gamma(d / p_norm + 1)
    c = r**d
    vol = c * a / b
    return vol


def calc_knn_density(x, data, k=5, p_norm=2):
    r"""Calculates the probability density of every point in x.

    Calculates the probability density of every point in x based on
    the proximity to the data using k-nearest neighbors and the equation
    proposed by Wang et al. (2009). Note: x and data must have the
    same number of d_features. 10.1109/TIT.2009.2016060

    .. math::
        \hat{p}_k(x_i) = \frac{k}{N-1} \cdot \frac{1}{c_1(d) \cdot \rho^{d}_k(i)}

    Where:
        * :math:`k` is the number of neighbors used.
        * :math:`N` is the number of samples in the data.
        * :math:`c_1(d)` is the volume of a d-dimensional unit :math:`L^p` ball.
        * :math:`d` is the number of dimensions of the data.
        * :math:`\rho^{d}_k(i)` is the distance between a point :math:`i` and its *k*-th nearest neighbor.

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d_features)
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    k : int, optional
        no. of nearest neighbors to use
    p_norm : int
        p (Minkowski) distance. p = 1 is the Manhattan distance,
        p = 2 is the Euclidian distance, etc.

    Returns
    -------
    p : numpy.array
        Array of shape (n_samples, d_features)
    """
    n, d = data.shape
    vol = vol_lp_ball(r=1.0, d=d, p_norm=p_norm)

    knn_tree = KDTree(data)
    radius = knn_tree.query(x, k + 1, p=p_norm)[0][:, k]
    p = (k / (n - 1)) * (1 / (vol * radius))
    return p


def calc_knn_entropy(data, k=1, p_norm=2):
    r"""Calculates the (joint) entropy of the input n-d array.

    Calculates the (joint) entropy of the input data [in nats] using
    the  Kozachenko and Leonenko (KL) (1987) estimator which is an approach
    based on k-nearest neighbors (k-NN). By default, the Euclidean norm
    distance (p-norm = 2) is used to calculate distances.
    http://mi.mathnet.ru/ppi797

    .. math::
        \hat{H}(X) = \psi(N) - \psi(k) + log(c_1(d)) + \frac{d}{N}\sum_{i=1}^{N}\log(\rho^{d}_k(i))

    Where:
        * :math:`\psi` is the digamma function.
        * :math:`N` is the number of samples in the data.
        * :math:`k` is the number of neighbors used.
        * :math:`c_1(d)` is the volume of a d-dimensional unit :math:`L^p` ball.
        * :math:`d` is the number of dimensions of the data.
        * :math:`\rho^{d}_k(i)` is the distance between a point :math:`i` and its *k*-th nearest neighbor.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    k : int, optional
        no. of nearest neighbors to use

    Returns
    -------
    h : float
        Entropy of the data set [in nats]"""

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    n, d = data.shape

    knn_tree = KDTree(data)
    radius = knn_tree.query(data, k + 1, p=p_norm)[0][:, k]
    h = (
        digamma(n)
        - digamma(k)
        + np.log(vol_lp_ball(1.0, d, p_norm))
        + d * np.mean(np.log(radius))
    )
    return h


def calc_knn_kld(p, q, k=1, p_norm=2):
    r"""Calculates the the Kullback-Leibler divergence (relative entropy) between
    two n-d arrays of data.

    Calculates the Kullback-Leibler divergence (relative entropy) between
    two data sets (p and q) [in nats] using the estimation method proposed
    by Wang et al. (2009). Both p and q are n-d arrays where d >= 1 which means
    they can have multiple features. Typically p represents the true distribution
    while q is the approximate distribution. Different number of total
    samples in p and q is acceptable, 10.1109/TIT.2009.2016060

    .. math::
        \hat{D}_{KL\,n,m}(p||q) = \frac{d}{n} \sum_{i=1}^{n} \log \left ( \frac{\nu_k(i)}{\rho_k(i)} \right ) + \log\left (\frac{m}{n-1}\right )

    Where:
        * :math:`d` is the number of dimensions of the data.
        * :math:`n` is the number of samples in *p*.
        * :math:`\rho_k(i)` is the distance between :math:`x_i` and its *k*-NN in :math:`{x_j}_(j \neq i)`.
        * :math:`\nu_k(i)` is the distance between :math:`x_i` and its *k*-NN in :math:`y_j`.
        * :math:`m` is the number of points in *q*.

    Parameters
    ----------
    p : numpy.ndarray
        Array of shape (n_samples, d_features)
    q : numpy.ndarray
        Array of shape (m_samples, d_features)
    k : int, optional
        no. of nearest neighbors to use
    p_norm : int, optional
        p (Minkowski) distance. p = 1 is the Manhattan distance,
        p = 2 is the Euclidian distance, etc.

    Returns
    -------
    kld : float
        Kullback-Leibler divergence between p and q [in nats]"""

    n, m = len(p), len(q)
    d = len(p[0])

    rho, _ = KDTree(p).query(p, k + 1, p=p_norm)
    nu, _ = KDTree(q).query(p, k, p=p_norm)
    rho = rho.reshape(-1, k + 1)[:, -1]
    nu = nu.reshape(-1, k)[:, -1]

    kld = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1))

    return max(0.0, kld)


def calc_knn_mutual_information(x, y, k=15):
    r"""Calculates the mutual information between two n-d arrays of data.

    Estimates the mutual information between two data sets (x and y) [in nats] using
    the method of estimation proposed Kraskov et al. (2004). Both x and y
    can have d >= 1 which means they can have multiple features. By default, the maximum
    norm (p-norm = ∞) and three neighbors (k = 3) are used. 10.1103/PhysRevE.69.066138

    .. math::
        \hat{I}(X;Y) = \psi(k) - \frac{1}{N} \sum_{i=1}^{N} \mathbb{E} \left[\psi(n_{i,x} + 1) + \psi(n_{i,y} + 1) \right] + \psi(N)

    Where:
        * :math:`\psi` is the digamma function.
        * :math:`N` is the number of samples.
        * :math:`\mathbb{E}` is the expectation operation.
        * :math:`n_{i,x}` and :math:`n_{i,y}` are the number of neighbors of every point within a given radius calculated as the distance to the *k*-th nearest neighbor in the joint X-Y space.

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d_features)
    y : numpy.ndarray
        Array of shape (n_samples, d_features)
    k : int, optional
        no. of nearest neighbors to use

    Returns
    -------
    mi : float
        Mutual information between x and y [in nats]"""

    assert len(x) == len(y), "x and y must have the same number of samples."

    n_samples = len(x)
    xy = np.hstack((x, y))

    xy_tree = KDTree(xy)
    x_tree = KDTree(x)
    y_tree = KDTree(y)

    radius = xy_tree.query(xy, k=[k + 1], p=np.inf)[0].flatten()
    nx = x_tree.query_ball_point(x, radius - 1e-12, p=np.inf, return_length=True)
    ny = y_tree.query_ball_point(y, radius - 1e-12, p=np.inf, return_length=True)

    mi = digamma(n_samples) + digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1))

    return max(0.0, mi)
