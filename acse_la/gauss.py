import numpy as np
import copy
from fractions import Fraction

__all__ = ['gauss']


def gauss(a, b):
    """
    Given two matrices, `a` and `b`, with `a` square, the determinant
    of `a` and a matrix `x` such that a*x = b are returned.
    If `b` is the identity, then `x` is the inverse of `a`.

    Parameters
    ----------
    a : np.array or list of lists
        'n x n' array
    b : np. array or list of lists
        'm x n' array

    Examples
    --------
    >>> a = [[2, 0, -1], [0, 5, 6], [0, -1, 1]]
    >>> b = [[2], [1], [2]]
    >>> det, x = gauss(a, b)
    >>> det
    22.0
    >>> x
    [[1.5], [-1.0], [1.0]]
    >>> A = [[1, 0, -1], [-2, 3, 0], [1, -3, 2]]
    >>> I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> Det, Ainv = gauss(A, I)
    >>> Det
    3.0
    >>> Ainv
    [[2.0, 1.0, 1.0],
    [1.3333333333333333, 1.0, 0.6666666666666666],
    [1.0, 1.0, 1.0]]

    Notes
    -----
    See https://en.wikipedia.org/wiki/Gaussian_elimination for further details.
    """
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    n = len(a)
    p = len(b[0])
    det = np.ones(1, dtype=np.int32)
    for i in range(n - 1):
        k = i
        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[k][i]):
                k = j
        if k != i:
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
            det = -det

        for j in range(i + 1, n):
            t = a[j][i]/a[i][i]
            for k in range(i + 1, n):
                a[j][k] -= t*a[i][k]
            for k in range(p):
                b[j][k] -= t*b[i][k]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            t = a[i][j]
            for k in range(p):
                b[i][k] -= t*b[j][k]
        t = 1/a[i][i]
        det *= a[i][i]
        for j in range(p):
            b[i][j] *= t
    return det, b


def matmul(a, b): 
    #from IPython import embed; embed()
    n, p = len(a), len(a[0])
    p1, q = len(b), len(b[0])
    if p != p1:
        raise ValueError("Incompatible dimensions")
    c = zeromat(n, q)
    for i in range(n):
        for j in range(q):
            c[i][j] = sum(a[i][k]*b[j][k] for k in range(p))
    return c

def zeromat(p, q):
    return [[0]*q for i in range(p)]
