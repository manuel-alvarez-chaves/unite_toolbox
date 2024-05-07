import numpy as np
from scipy import stats

rng = np.random.default_rng(seed=42)


def generate_samples() -> tuple[np.ndarray]:
    """Generate samples.

    Simple function to generate some data for testing purposes. Two samples
    are generate with data coming from two different distributions, the first
    from N(0.0, 0.6577) and the second from N(0.0, 1.0). These are multi-
    variate samples with ndim = 2.
    """
    n_samples = 100
    samples1 = rng.normal(loc=0.0, scale=0.6577, size=(n_samples, 2))
    samples2 = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 2))
    return samples1, samples2


def pdf_mnorm(x: float, y: float, params: list[list[float]]) -> float:
    """Evaluate the PDF of a 2D normal distribution.

    Evaluates `x` and `y` in the PDF of a 2D normal distribution defined by the
    contents of `params`. `params` should contain sublists with means,
    covariances and weights of 2D normal distributions in the format:
    [[mean1, cov1, weight1], [mean2, cov2, weight2]] for a mixture of two 2D
    normal distributions.

    Parameters:
    ----------
    x : float
        point to evaluate the PDF in the first axis
    y : float
        point to evaluate the PDF in the second axis
    params : list[lists[float]]
        list of lists containing parameters for mean, cov and weight of the
        distribution
    """
    z = 0.0
    for dist in params:
        loc, scale, weight = dist
        z += (
            stats.multivariate_normal(mean=loc, cov=scale).pdf(
                np.dstack((x, y))
            )
            * weight
        )
    return z
