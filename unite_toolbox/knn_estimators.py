import numpy as np
from scipy.spatial import KDTree
from scipy.special import gamma, digamma


def vol_lp_ball(r, d, p_norm):
    """Calculates the volume of a L^p ball of radius r in d dimensional
    space.

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

    if p_norm == np.inf:
        vol = (2**d) * (r**d)
        return vol

    a = (2 * gamma(1 / p_norm + 1)) ** d
    b = gamma(d / p_norm + 1)
    c = r**d
    vol = c * a / b
    return vol
    
def calc_knn_density(x, data, k=5, p_norm=2):
    N, d = data.shape
    vol = vol_lp_ball(r=1, d=d, p_norm=p_norm)

    knn_tree = KDTree(data)
    radius = knn_tree.query(x, k+1, p=p_norm)[0][:, k]
    p = (k / (N - 1)) * (1 / (vol * radius))
    return p


def calc_knn_entropy(data, k=3, p_norm=2):
    """Calculates the (joint) entropy of the input n-d array.

    Calculates the (joint) entropy of the input data [in nats] using
    the  Kozachenko and Leonenko (KL) (1987) estimator which is an approach
    based on k-nearest neighbors (k-NN) to approximate probability
    densities. By default, the Euclidean norm distance (p-norm = 2) is used
    to calculate distances. http://mi.mathnet.ru/ppi797


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

    N, d = data.shape
    

    knn_tree = KDTree(data)
    radius = knn_tree.query(data, k+1, p=p_norm)[0][:, k]
    h = (
        digamma(N)
        - digamma(k)
        + np.log(vol_lp_ball(1.0, d, p_norm))
        + d * np.mean(np.log(radius))
    )
    return h

def calc_knn_kld(data_p, data_q, k=3, p_norm=2):
    """Calculates the the Kullback-Leibler divergence (relative entropy) between
    two n-d arrays of data.
    
    Calculates the Kullback-Leibler divergence (relative entropy) between
    two data sets (p and q) [in nats] using the estimation method proposed 
    by Wang et al. (2009). Both p and q are n-d arrays where d >= 1 which means
    they can have multiple features. Typically p represents the true distribution
    while q is the approximate distribution. Different number of total
    samples in p and q is acceptable, 10.1109/TIT.2009.2016060

    Parameters
    ----------
    data_p : numpy.ndarray
        Array of shape (n_samples, d_features)
    data_q : numpy.ndarray
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

    n, m = len(data_p), len(data_q)
    d = len(data_p[0])

    rho, _ = KDTree(data_p).query(data_p, k + 1, p=p_norm)
    nu, _ = KDTree(data_q).query(data_p, k, p=p_norm)
    rho = rho.reshape(-1, k + 1)[:, -1]
    nu = nu.reshape(-1, k)[:, -1]

    kld = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1))
    
    return kld

def calc_knn_mutual_information(x, y, k=3):
    """Calculates the mutual information between two n-d arrays of data.

    Estimates the mutual information between two data sets (x and y) [in nats] using
    the method of estimation proposed Kraskov et al. (2004). Both x and y
    can have d >= 1 which means they can have multiple features. By default, the maximum
    norm (p-norm = ∞) and three neighbors (k = 3) are used. 10.1103/PhysRevE.69.066138

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d_features)
    y : numpy.ndarray
        Array of shape (n_samples, d_features)
    k : int, optional
        no. of nearest neighbors to use, k <= 2

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

    radius = xy_tree.query(xy, k=[k+1], p=np.inf)[0].flatten()
    nx = x_tree.query_ball_point(x, radius - 1e-12, p=np.inf, return_length=True)
    ny = y_tree.query_ball_point(y, radius - 1e-12, p=np.inf, return_length=True)

    mi = (
        digamma(n_samples)
        + digamma(k)
        - np.mean(digamma(nx + 1) + digamma(ny + 1))
        )
    
    return max(0, mi)