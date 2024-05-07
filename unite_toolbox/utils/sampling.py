from collections.abc import Callable
from typing import Any

import numpy as np


def get_samples(
    func: Callable,
    limits: list[list[float]],
    n_samples: int,
    seed: int | None = None,
    **kwargs: dict[str, Any],
) -> np.ndarray:
    r"""Get samples.

    The algorithm obtains samples :math:`y` from the distribution :math:`X`
    defined in `func` with density :math:`f`, using samples from :math:`Y` with
    density :math:`g` using rejection sampling with :math:`Y` being a uniform
    proposal distribution. In the first step samples from `Y` are generated.
    Then those samples are evaluated in :math:`X` using `func`. We use a naive
    scaling value :math:`M` by taking the ratio of highest sampled density in
    :math:`X` and dividing it by the density in :math:`Y` which, being a
    uniform distribution, is a constant. Samples are accepted only if:

    .. math::
        u < \frac{f(y)}{M\,g(y)}

    NOTE: This started as a function for Monte Carlo integration (main reason
    for uniform proposal distribution) and ended up being used for directly
    taking the samples. Therefore the implementation is not ideal.

    Parameters
    ----------
    func : Callable
        function defining a distribution to sample from
    limits : list[lists[floats]]
        a list of lists with contains the limits to sample from, the number
        of lists should match the dimensionality of `func`
    n_samples : int
        number of samples to obtain
    seed : int, optional
        seed for random number generator
    **kwargs : dict[str, Any]
        additional arguments for `func`

    Returns
    -------
    samples : np.ndarray
        samples of `func`

    """
    rng = np.random.default_rng(seed)

    ids = []
    acc_rate = 1.0 # acceptance rate
    while len(ids) < n_samples:
        d = len(limits)
        f = np.array(limits)[:, 0]  # floor
        s = np.array(limits)[:, 1] - f  # scale
        r = rng.uniform(size=(int(n_samples / acc_rate), d)) # samples from Y
        r = f + s * r # uniformly distributed samples within `limits`

        F = func(*(np.hsplit(r, d)), **kwargs).flatten() # evaluates `func`
        G = 1 / np.prod(s) # uniform density -> proposal distribution
        M = F.max() / G # scaling value (proposal G > F)

        U = rng.uniform(0.0, 1.0, size=F.shape)
        ids = np.argwhere(U < F / (M * G)).flatten() # decision rule
        acc_rate = acc_rate * (len(ids) / n_samples) # update acc. rate

    samples = r[ids][:n_samples, :] # return only n_samples
    return samples
