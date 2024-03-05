import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import nquad

def calc_kde_density(x, data, bandwidth=None):
    """
    """
    
    kde = gaussian_kde(data.T, bw_method=bandwidth)
    p = kde.evaluate(x.T).reshape(-1, 1)
    return p

def calc_kde_entropy(data, bandwidth=None):
    """Calculates the (joint) entropy of the input n-d array.

    Calculates the (joint) entropy of the input data [in nats] by
    approximating the (joint) density of the distribution using a
    Gaussian kernel density estimator (KDE). By defaul the Scott
    estimate for the bandwith is used for the Gaussian kernel.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    bandwidth : float, optional
        bandwidth of the gaussian kernel
    
    Returns
    -------
    h : float
        Entropy of the data set [in nats]"""
    
    kde = gaussian_kde(data.T, bw_method=bandwidth)
    p_kde = kde.evaluate(data.T)
    h = -1 * np.mean(np.log(p_kde))
    return h

def calc_ikde_entropy(data, bandwidth=None):
    """
    """
    
    lims = np.vstack((data.min(axis=0), data.max(axis=0))).T
    kde = gaussian_kde(data.T, bw_method=bandwidth)

    def eval_entropy(*args):
        p = kde.evaluate(np.vstack(args))
        return -1*p*np.log(p)
    
    h = nquad(eval_entropy, ranges=lims)[0]
    return h
    
def calc_kde_kld(data_f, data_g, bandwidth=None):
    """Calculates the the Kullback-Leibler divergence (relative entropy) between
    two n-d arrays of data.
    
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
    
    f_kde = gaussian_kde(data_f.T, bw_method=bandwidth)
    g_kde = gaussian_kde(data_g.T, bw_method=bandwidth)
    
    pf_kde = f_kde.evaluate(data_f.T)
    pg_kde = g_kde.evaluate(data_f.T)
    
    kld = np.abs(np.mean(np.log(pf_kde / pg_kde)))
    
    return kld
    
def calc_ikde_kld(data_f, data_g, bandwidth=None):
    """
    """
    
    f_kde = gaussian_kde(data_f.T, bw_method=bandwidth)
    g_kde = gaussian_kde(data_g.T, bw_method=bandwidth)
    bw = g_kde.factor

    lims = np.vstack((data_g.min(axis=0) - bw, data_g.max(axis=0) + bw)).T

    def eval_kld(*args):
        f = f_kde.evaluate(np.vstack(args))
        g = g_kde.evaluate(np.vstack(args))
        if f == 0.0 or g == 0.0:
            return 0.0
        else:
            return f * np.log(f / g) 

    kld = nquad(eval_kld, ranges=lims)[0]
    return kld
    
def calc_kde_mutual_information(x, y, bandwidth=None):
    """
    """
    
    xy = np.hstack((x, y))
    
    kde_x = gaussian_kde(x.T, bw_method=bandwidth)
    kde_y = gaussian_kde(y.T, bw_method=bandwidth)
    kde_xy = gaussian_kde(xy.T, bw_method=bandwidth)
    
    px_kde = kde_x.evaluate(x.T)
    py_kde = kde_y.evaluate(y.T)
    pxy_kde = kde_xy.evaluate(xy.T)
    
    mi = np.mean(np.log(pxy_kde / (px_kde * py_kde)))
    return max(0, mi)
    
def calc_ikde_mutual_information(x, y, bandwidth=None):
    """
    """
    
    xy = np.hstack((x, y))
    d = xy.shape[1]

    kde_x = gaussian_kde(x.T, bw_method=bandwidth)
    kde_y = gaussian_kde(y.T, bw_method=bandwidth)
    kde_xy = gaussian_kde(xy.T, bw_method=bandwidth)
    bw = kde_xy.factor
    
    lims = np.vstack((xy.min(axis=0) - bw, xy.max(axis=0) + bw)).T

    def eval_mi(*args):
        px = kde_x.evaluate(np.vstack(args[:d-1]))
        py = kde_y.evaluate(np.vstack((args[d-1],)))
        pxy = kde_xy.evaluate(np.vstack(args))
        return pxy * np.log(pxy/(px * py))
    
    mi = nquad(eval_mi, ranges=lims)[0]
    return max(0, mi)