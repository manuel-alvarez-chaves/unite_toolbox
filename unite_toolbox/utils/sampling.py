import numpy as np


def get_samples(func, limits, n_samples, seed=None, **kwargs):
    rng = np.random.default_rng(seed)

    ids = []
    acc_rate = 1.0
    while len(ids) < n_samples:
        d = len(limits)
        f = np.array(limits)[:, 0]  # floor
        s = np.array(limits)[:, 1] - f  # scale
        r = rng.uniform(size=(int(n_samples / acc_rate), d))
        r = f + s * r

        F = func(*(np.hsplit(r, d)), **kwargs).flatten()
        G = 1 / np.prod(s)
        M = F.max() / G

        U = rng.uniform(0.0, 1.0, size=F.shape)
        ids = np.argwhere(U < F / (M * G)).flatten()
        acc_rate = acc_rate * (len(ids) / n_samples)

    samples = r[ids][:n_samples, :]
    return samples
