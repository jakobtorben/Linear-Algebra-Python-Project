==========================================
Welcome to acse_la's documentation!
==========================================

``acse_la`` is a linear algebra library that implements Gaussian Elimination, matrix multiplication and calculating the determinant.


A Gaussian Elimination routine
------------------------------

This package implements Gaussian elimination [1]_ for list lists and :obj:`numpy.ndarray` objects, along with hand-written matrix multiplication.

See :func:`acse_la.gauss` and :func:`acse_la.gauss.matmul` for more information.

.. automodule:: acse_la
  :members: gauss

.. automodule:: acse_la.gauss
  :members: matmul, zeromat
  :noindex: gauss

A determinant routine
---------------------

.. automodule:: acse_la
  :members: det

.. automodule:: acse_la.det
  :members: det
  :noindex: gauss


Citation
--------

[1] https://mathworld.wolfram.com/GaussianElimination.html
[2] Propp, J. G., Wilson, D. B. 'Numerical Recipes', Cambridge University Press. (1996)
