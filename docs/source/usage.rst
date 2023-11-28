Usage
=====

.. _installation:

Installation
------------

As the toolbox is still in very active development, installation can only be done through `git` and `pip`:

.. code-block:: console

   $ pip install "git+https://github.com/manuel-alvarez-chaves/unite_toolbox"


Estimation
----------

A typical example would involve the calculation of the entropy :math:`H(X)` of a dataset. This can be accomplished through the *k*-NN estimator:

.. code-block:: Python

    from scipy import stats
    from unite_toolbox import knn_estimators

    # Generate data
    dist = stats.norm(loc=0.0, scale=0.6577)
    samples = dist.rvs(size=(10_000, 1), random_state=42)

    # Estimate entropy and print results
    est_h = knn_estimators.calc_knn_entropy(samples)
    print(f"Est. H = {est_h:.3f} nats")
    print(f"True H = {dist.entropy():.3f} nats")

Output:

.. code-block:: console

    Est. H = 1.002 nats
    True H = 1.000 nats

Success!