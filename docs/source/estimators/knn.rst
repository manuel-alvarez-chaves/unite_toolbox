*k*-NN
------
Different authors have proposed equations to calculate entropy, KL divergence and
mutual information using *k*-nearest neighbors (*k*-NN). In principle one could make
an estimate of density :math:`p(x)` at :math:`x_i` using ``calc_knn_density``
and then use a **resubstitution estimate** of a desired quantity but this is not recommended
for *k*-NN-based methods. See more in the tutorial: `Non-parametric Density Estimation <https://unite-toolbox.readthedocs.io/en/latest/tutorials/Density-Estimation.html>`_.

See ``calc_knn_entropy``, ``calc_knn_kld`` and ``calc_knn_mutual_information`` for details
specific for each equation as well as their source.

Distance is a specific topic which requires a few words. Distance is the length between two points :math:`x` and
:math:`y`, and is given by a :math:`p`-norm function where :math:`p \geq 1` as follows:

.. math::
   \left\|x - y\right\|_{p} = (|x_{1} - y_{1}|^{p} + |x_{2} - y_{2}|^{p} + \cdots + |x_{n} - y_{n}|^{p})^{\frac{1}{p}}

with :math:`p=2` suggested for both estimating density, entropy and KL divergence. For mutual
information, Kraskov, et al. propose the usage of :math:`p=\inf` or the infinite norm.

.. automodule:: unite_toolbox.knn_estimators
   :members:
