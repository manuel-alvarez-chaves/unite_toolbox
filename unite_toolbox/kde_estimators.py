import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import nquad


def calc_kde_density(x, data, bandwidth=None):
    """Calculates density using KDE

    Calculates the density of every point of the 2D array `x` within KDE
    representation of `data`. Simply, every point in `x` is evaluated
    in a KDE-based distribution of `data`.

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d_features)
    data : numpy.ndarray
        Array of shape (m_samples, d_features)

    Returns
    -------
    p : numpy.ndarray
        Array of shape (n_samples, 1)
    """
    kde = gaussian_kde(data.T, bw_method=bandwidth)
    p = kde.evaluate(x.T).reshape(-1, 1)
    return p


def calc_kde_entropy(data, bandwidth=None):
    """Calculates entropy using KDE

    Calculates the (joint) entropy of the input data [in nats] by
    approximating the (joint) density of the distribution using a
    Gaussian kernel density estimator (KDE). By defaul the Scott
    estimate for the bandwith is used for the Gaussian kernel.
    This is a resubstitution estimate.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    bandwidth : float, optional
        bandwidth of the gaussian kernel

    Returns
    -------
    h : float
        Entropy of the data set [in nats]
    """

    kde = gaussian_kde(data.T, bw_method=bandwidth)
    p = kde.evaluate(data.T)
    h = -1 * np.mean(np.log(p))
    return h


def calc_ikde_entropy(data, bandwidth=None):
    """Calculates entropy using numerical integration and KDE

    Calculates the (joint) entropy of the input data [in nats] by
    approximating the (joint) density of the distribution using a
    Gaussian kernel density estimator (KDE).
    The method creates a helper function that as the basis
    for numerical integration. The integration limits are set as the maximum
    and minimum values of X and Y, plus and minus one magnitude of the bandwidth
    respectively.
    This is an integral estimate.
    """

    lims = np.vstack((data.min(axis=0), data.max(axis=0))).T
    kde = gaussian_kde(data.T, bw_method=bandwidth)

    def eval_entropy(*args):
        p = kde.evaluate(np.vstack(args))
        return -1 * p * np.log(p)

    h = nquad(eval_entropy, ranges=lims)[0]
    return h


def calc_kde_kld(p, q, bandwidth=None):
    """Calculates Kullback-Leibler divergence (relative entropy) using KDE

    Calculates the Kullback-Leibler divergence (relative entropy) between
    two data sets (f and g) [in nats] by approximating both distributions using
    a Gaussian kernel density estimate (KDE). The divergence is measured between
    both of the estimated densities. Both density estimates are independent, therefore
    a different number of total samples in p and q is valid.

    Parameters
    ----------
    data_p : numpy.ndarray
        Array of shape (n_samples, d_features)
    data_q : numpy.ndarray
        Array of shape (m_samples, d_features)
    bw : float, optional
        bandwith of the gaussian kernel

    Returns
    -------
    kld : float
        Kullback-Leibler divergence between f and g [in nats]"""

    p_kde = gaussian_kde(p.T, bw_method=bandwidth)
    q_kde = gaussian_kde(q.T, bw_method=bandwidth)

    pi_kde = p_kde.evaluate(p.T)
    qi_kde = q_kde.evaluate(q.T)

    kld = np.abs(np.mean(np.log(pi_kde / qi_kde)))

    return kld


def calc_ikde_kld(p, q, bandwidth=None):
    """Calculates Kullback-Leibler divergence (relative entropy) using
    numerical integration and KDE

    """

    p_kde = gaussian_kde(p.T, bw_method=bandwidth)
    q_kde = gaussian_kde(q.T, bw_method=bandwidth)
    bw = q_kde.factor

    lims = np.vstack((q.min(axis=0) - bw, q.max(axis=0) + bw)).T

    def eval_kld(*args):
        pi = p_kde.evaluate(np.vstack(args))
        qi = q_kde.evaluate(np.vstack(args))
        if pi == 0.0 or qi == 0.0:
            return 0.0
        else:
            return pi * np.log(pi / qi)

    kld = nquad(eval_kld, ranges=lims)[0]
    return kld


def calc_kde_mutual_information(x, y, bandwidth=None):
    """Calculates mutual information between X and Y using KDE

    Calculates the mutual information between X and Y [in nats] using KDE.
    This method uses a multivariate Gaussian kernel so, both X an Y can have
    multivariate data. The method evaluates density at every point in X, Y and
    X-Y, therefore, X and Y must have the same number of entries.
    This is a resubstition method.

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d1_features)
    y : numpy.ndarray
        Array of shape (n_samples, d2_features)
    bandwidth : float, optional
        bandwith of the gaussian kernel, "scott" by default

    Returns
    -------
    mi : float
        Mutual information between x and y [in nats]
    """

    xy = np.hstack((x, y))

    kde_x = gaussian_kde(x.T, bw_method=bandwidth)
    kde_y = gaussian_kde(y.T, bw_method=bandwidth)
    kde_xy = gaussian_kde(xy.T, bw_method=bandwidth)

    px_kde = kde_x.evaluate(x.T)
    py_kde = kde_y.evaluate(y.T)
    pxy_kde = kde_xy.evaluate(xy.T)

    mi = np.mean(np.log(pxy_kde / (px_kde * py_kde)))
    return max(0.0, mi)


def calc_ikde_mutual_information(x, y, bandwidth=None):
    """Calculates mutual information between X and Y using numerical
    integration and KDE

    Calculates the mutual information between X and Y [in nats] using numerical
    integration and KDE.
    This method uses a multivariate Gaussian kernel so, both X an Y can have
    multivariate data. The method creates a helper function that as the basis
    for numerical integration. The integration limits are set as the maximum
    and minimum values of X and Y, plus and minus one magnitude of the bandwidth
    respectively.
    This is an integral estimate.

    Parameters
    ----------
    x : numpy.ndarray
        Array of shape (n_samples, d1_features)
    y : numpy.ndarray
        Array of shape (n_samples, d2_features)
    bandwidth : float, optional
        bandwith of the gaussian kernel, "scott" by default

    Returns
    -------
    mi : float
        Mutual information between x and y [in nats]
    """

    xy = np.hstack((x, y))
    _, d = xy.shape

    kde_x = gaussian_kde(x.T, bw_method=bandwidth)
    kde_y = gaussian_kde(y.T, bw_method=bandwidth)
    kde_xy = gaussian_kde(xy.T, bw_method=bandwidth)
    bw = kde_xy.factor

    lims = np.vstack((xy.min(axis=0) - bw, xy.max(axis=0) + bw)).T

    def eval_mi(*args):  # helper function
        px = kde_x.evaluate(np.vstack(args[: d - 1]))
        py = kde_y.evaluate(np.vstack((args[d - 1],)))
        pxy = kde_xy.evaluate(np.vstack(args))
        return pxy * np.log(pxy / (px * py))

    mi = nquad(eval_mi, ranges=lims)[0]
    return max(0.0, mi)
