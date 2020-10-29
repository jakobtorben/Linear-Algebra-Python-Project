import os
from timeit import Timer

import numpy as np

from acse_la import gauss


if __name__ == '__main__':

    # Random
    sizes = [10, 100, 150]
    timings = np.zeros((len(sizes), 4))
    timings[:, 0] = sizes
    np.seterr(all='raise')

    for i, size in enumerate(sizes):
        M = np.random.rand(size, size)
        b = np.eye(size)
        #b = np.random.rand(size, 1)

        gauss_t = Timer(lambda: gauss(M, b))
        timings[i, 1] = gauss_t.timeit(number=2)

        linalg_t = Timer(lambda: np.linalg.det(M))
        timings[i, 2] = linalg_t.timeit(number=2)

    fmt = '%-3d', '%-1.1f', '%-1.1f', '%-1.1f'
    header = 'Mat. dim   gauss t(s)  linalg t(s) my impl t(s)'
    filename = os.path.join("results", "timings.txt")
    np.savetxt(filename, timings, header=header, comments='',
                                             fmt=fmt, delimiter='\t       ')

    # Make method for sparse as well ( maybe triangular)