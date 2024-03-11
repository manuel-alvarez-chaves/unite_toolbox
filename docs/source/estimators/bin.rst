Binning
-------
The binning method consists of obtainin an estimate of the PDF using a
histogram. Frequency at a point :math:`x_i` from histogram can be calculated
as:

.. math::
   \hat{f}(x_i) = \frac{c_i}{n\Delta}.

Where :math:`\hat{f}` is an estimate of the frequency count, :math:`c_i` is
the number of observations in the same bin as :math:`x_i`, :math:`n` is the
total number of observations and :math:`\Delta` is the bin width.

To get probability density from this frequency count, the width of the bin
must be accounted for and, then, the density estimate can be calculated as
:math:`p(x) = \Delta\,f(x)`.

The ideal :math:`\Delta` varies depending on the ``data``, but several
"rules-of-thumb" exist that can be used as guidance. These have been
implemented in the **UNITE** toolbox taking advantage and following the same
notation as in ``numpy``'s ``histogram_bin_edges``. See: ``estimate_ideal_bins``.

Having an estimate of density :math:`\hat{p}(x)` at :math:`x` makes it so entropy, KL divergence
and mututal information can be calculated directly as **resubstitution estimates** using the
following equations:

.. math::
   H(X) = -\frac{1}{n} \sum_{i=1}^{n} {\log{\left ( \hat{p}\left ( x_i \right ) \right )}}

.. math::
   D_{KL}(p||q) = \frac{1}{n} \sum_{i=1}^{n} {\log{\left ( \frac{\hat{p}(x_i)}{\hat{q}(x_i)} \right )}}

.. math::
   I(X;Y) = \frac{1}{n} \sum_{i=1}^{n} {\log{\left ( \frac{\hat{p}(x_i,y_i)}{\hat{p}(x_i)\hat{p}(y_i)} \right )}}


.. automodule:: unite_toolbox.bin_estimators
   :members:
