"""Determinant calculation timer

This script allows the user to time different methods to calculate the
determinant of matrices.

Methods tested:
- Gaussian elemination
- numpy.linalg.det (Using  LU factorisation using the LAPACK routine)
- My own implementation using LU decomposition (Crout's method)

The methods are tested on random dense square matrices with size 10 to 500.
The result are written to a results/timings.txt.
"""

import os
from timeit import Timer

import numpy as np

from acse_la import gauss, det


sizes = [10, 100]#, 200]  # , 400]#, 500]
n_times = 1
timings = np.zeros((len(sizes), 4))
timings[:, 0] = sizes

for i, size in enumerate(sizes):
    M = np.random.rand(size, size)
    b = np.eye(size)

    gauss_t = Timer(lambda: gauss(M, b))
    timings[i, 1] = gauss_t.timeit(number=n_times)/n_times

    linalg_t = Timer(lambda: np.linalg.det(M))
    timings[i, 2] = linalg_t.timeit(number=n_times)/n_times

    det_t = Timer(lambda: det(M))
    timings[i, 3] = det_t.timeit(number=n_times)/n_times

fmt = '%d', '%3g', '%3g', '%3g'
header = 'Mat. size  gauss t(s)      linalg t(s)     own t(s)'
filename = os.path.join("results", "timings.txt")
np.savetxt(filename, timings,
           header=header, comments='', fmt=fmt, delimiter='\t')
