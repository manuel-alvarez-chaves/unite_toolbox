import numpy as np

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
