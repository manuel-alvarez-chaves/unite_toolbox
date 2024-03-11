KDE
---
KDE-based estimation consists of estimating a PDF based on kernels as weights,
with the kernel being a non-negative window function. The density :math:`p(x)`
at a point :math:`x` is estimated as:

.. math::
   \hat{p}(x) = \frac{1}{n}\sum_{i=1}^d K(u)

where:

.. math::
   u = \frac{\left (x-x_i  \right )^\intercal \Sigma^{-1}\left ( x-x_i \right )}{h^2}

and :math:`n` is the total number of samples, :math:`K` is a multivariate kernel
function, :math:`x_i = [x_{1,i}, x_{2,i}, \dots, x_{d,i}]^\intercal` is a :math:`d`-
dimensional vector of samples, :math:`\Sigma` is the covariance matrix of the samples,
and :math:`h` is a smoothing parameter.

The **UNITE** toolbox uses a multivariate Gaussian kernel by default:

.. math::
   K(u)=\frac{1}{\left (2\pi  \right )^{d/2} h^d \det{\left ( \Sigma  \right )}^{1/2}} e^{-u/2}

with Silverman's bandwidth estimate:

.. math::
   h=\left ( \frac{n\left ( d+2 \right )}{4} \right )^{-1/\left ( d+4 \right )}

Having an estimate of density :math:`\hat{p}(x)` at :math:`x` makes it so entropy, KL divergence
and mututal information can be calculated directly as **resubstitution estimates** using the
following equations:

.. math::
   H(X) = -\frac{1}{n} \sum_{i=1}^{n} {\log{\left ( \hat{p}\left ( x_i \right ) \right )}}

.. math::
   D_{KL}(p||q) = \frac{1}{n} \sum_{i=1}^{n} {\log{\left ( \frac{\hat{p}(x_i)}{\hat{q}(x_i)} \right )}}

.. math::
   I(X;Y) = \frac{1}{n} \sum_{i=1}^{n} {\log{\left ( \frac{\hat{p}(x_i,y_i)}{\hat{p}(x_i)\hat{p}(y_i)} \right )}}

Further, **integral estimates** can also be calculated using numerical integration. For example,
using numerical integration, entropy is estimated as:

.. math::
   H(X) = -\int_\mathcal{X} \hat{p}(x) \log \hat{p}(x)\,\text{d}x

.. automodule:: unite_toolbox.kde_estimators
   :members:
