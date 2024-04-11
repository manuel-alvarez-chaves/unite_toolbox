import numpy as np
from unite_toolbox.utils.bootstrapping import find_repeats


def valida_data_kld(a, b, verbose=False):
    """Validate data for kNN-based KLD

    Eliminates repeated values from a and the joint array a-b to perform a
    distance based calculation of KLD, or other method which requires only
    unique values in two arrays.

    The code finds repeats in `a`, deletes them, and adds back the unique
    values that were repreated so that not a lot of the data is lost. A
    similar procedure is applied to `b` by first concatenating it with `a`
    so that no value in `a` is repeated in `b`.

    Parameters
    ----------
    a : numpy.ndarray
        2D array of shape (n_samples, d_features)
    b : numpy.ndarray
        2D array of shape (m_samples, d_features)

    Returns
    -------
    p : numpy.ndarray
        2D array of shape (<=n_samples, d_features)
    q : numpy.ndarray
        2D array of shape (<=m_samples, d_features)
    """
    # Clean a
    mask_a = np.argwhere(find_repeats(a)).flatten()
    repeats = a[mask_a]
    p = np.delete(a, mask_a, axis=0)
    unique_repeats = np.unique(repeats, axis=0)
    p = np.vstack((p, unique_repeats))

    # Clean b
    nrows, _ = p.shape
    aux = np.vstack((p, b))
    mask_b = find_repeats(aux)[nrows:]
    q = b[~mask_b]
    if verbose and (len(mask_a) > 0 or len(mask_b) > 0):
        counts_a = len(mask_a) - unique_repeats.shape[0]
        counts_b = len(np.argwhere(mask_b))
        print(f"Removed: {counts_a} from `p` and {counts_b} from `q`")
    return p, q
