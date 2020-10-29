import os
from timeit import Timer

import numpy as np

from acse_la import gauss





def timearr(size):
    M = np.random.randomint(size)
    b = np.array(size)
    gauss_det, b = gauss(M, b)

    linalg_det = np.linalg.det(M)


if __name__ == '__main__':

    # Random
    sizes = [10, 100, 150]#, 200]
    timings = np.zeros((len(sizes)+1, 4))
    #timings[0, :] = ['abs', 'ac', 'ac']
    timings[1:, 0] = sizes

    for i, size in enumerate(sizes):
        M = np.random.randn(size, size)
        b = np.random.randn(size, 1)

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