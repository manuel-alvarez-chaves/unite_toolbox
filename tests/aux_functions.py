import numpy as np
from scipy.special import gamma

from unite_toolbox.utils.sampling import get_samples


def pdf_gamma_exponential(x, y, params):
    r"""PDF of the Gamma-Exponential distribution.

    .. math::
        p(x_1, x_2) = \frac{x_{1}^\theta e^{-x_{1} - x_{1} \cdot x_{2}}}{\Gamma\left ( \theta \right )}

    Parameters
    ----------
        x : numpy.ndarray
            Array of shape (n_samples, 1)
        y : numpy.ndarray
            Array of shape (n_samples, 1)
        params : List[List[float]]
            List of sublists containing the number of distributions to mix,
            with each sublist containing floats for :math:`\theta`, the
    """
    z = 0.0
    for dist in params:
        t, w = dist
        z += (1 / gamma(t)) * (x**t) * np.exp(-x - x * y) * w
    return z


def generate_gexp_samples():
    r"""Static function to generate samples from the Gamma-Exponential distribution for testing.

    Returns
    -------
    data1 : numpy.ndarray
        Array of shape (10_000, 2) with samples from the Gamma-Exponential distribution
        with :math:`\theta = 3`
    data2 : numpy.ndarray
        Array of shape (10_000, 2) with samples from the Gamma-Exponential distribution
        with :math:`\theta = 4`
    """

    gexp1_params = [[3, 1]]
    gexp2_params = [[4, 1]]
    gexp_lims = [[0, 15], [0, 12]]

    samples1 = get_samples(func=pdf_gamma_exponential, limits=gexp_lims, n_samples=1_000, seed=42, params=gexp1_params)
    samples2 = get_samples(func=pdf_gamma_exponential, limits=gexp_lims, n_samples=1_000, seed=42, params=gexp2_params)

    return samples1, samples2
