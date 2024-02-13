import numpy as np
from scipy import stats


def find_repeats(data):
    # Returns a boolean mask for repeat rows in data
    _, inv, counts = np.unique(data, return_inverse=True, return_counts=True, axis=0)
    mask = np.where(counts[inv] > 1, True, False)
    return mask


def add_noise_to_data(data):
    data = data.astype("float64")
    _, d = data.shape
    # Scale of noise: two orders of magnitude below the order of magnitude
    # of the std. dev. of the data
    data_std = np.std(data, axis=0).reshape(1, -1) / 100
    noise_scale = 10 ** (np.floor(np.log10(np.abs(data_std))))

    # Generate only required noise for the data
    mask = find_repeats(data)
    noise = stats.multivariate_normal.rvs(
        cov=np.diag(noise_scale), size=mask.sum()
    ).reshape(-1, d)
    noisy_data = data.copy()
    noisy_data[mask] += noise

    return noisy_data


def one_sample_bootstrap(
    data, estimator, n_bootstraps, confidence, add_noise=False, **kwargs
):
    n, _ = data.shape
    res = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        sub_idx = np.random.choice(n, size=n, replace=True)
        subsample = data[sub_idx, :].copy()
        subsample = add_noise_to_data(subsample) if add_noise else subsample
        res[i] = estimator(subsample, **kwargs)
    bs_mean = res.mean()
    bs_ci = np.percentile(res, [confidence * 100, 100 - confidence * 100])
    return bs_mean, bs_ci
