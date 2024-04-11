import numpy as np
from tqdm import trange

from unite_toolbox.utils.data_validation import add_noise_to_data


def density_bootstrap(x, data, estimator, n_bootstraps, significance, add_noise=False, **kwargs):
    n, d = data.shape
    res = np.empty(shape=(x.shape[0], x.shape[1], n_bootstraps))
    for i in trange(int(n_bootstraps), ascii=True, unit="boot"):
        sub_idx = np.random.choice(n, size=n, replace=True)
        subsample = data[sub_idx, :].copy()
        subsample = add_noise_to_data(subsample) if add_noise else subsample
        res[:, :, i] = estimator(x, subsample, **kwargs).reshape(-1, d)
    bs_mean = res.mean(axis=2)
    bs_ci = np.quantile(res, [significance / 2, 1 - (significance / 2)], axis=2)
    return bs_mean, bs_ci


def one_sample_bootstrap(data, estimator, n_bootstraps, significance, add_noise=False, **kwargs):
    n, _ = data.shape
    res = np.empty(n_bootstraps)
    for i in trange(int(n_bootstraps), ascii=True, unit="boot"):
        sub_idx = np.random.choice(n, size=n, replace=True)
        subsample = data[sub_idx, :].copy()
        subsample = add_noise_to_data(subsample) if add_noise else subsample
        res[i] = estimator(subsample, **kwargs)
    bs_mean = res.mean()
    bs_ci = np.quantile(res, [significance / 2, 1 - (significance / 2)])
    return bs_mean, bs_ci
