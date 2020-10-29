import numpy as np
import copy
# from fractions import Fraction
# Make sure we don't need this before deleting

__all__ = ['gauss', 'matmul', 'zeromat']


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
    >>> Ainv # doctest: +NORMALIZE_WHITESPACE
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
    det = 1
    for i in range(n - 1):
        k = i
        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[k][i]):
                k = j
        if k != i:
            a_i_temp = copy.deepcopy(a[i])
            a[i], a[k] = a[k], a_i_temp
            b_i_temp = copy.deepcopy(b[i])
            b[i], b[k] = b[k], b_i_temp
            det = -det

        for j in range(i + 1, n):
            try:
                t = a[j][i]/a[i][i]
            except ZeroDivisionError:
                return 0, None

            for k in range(i + 1, n):
                a[j][k] -= t*a[i][k]
            for k in range(p):
                b[j][k] -= t*b[i][k]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            t = a[i][j]
            for k in range(p):
                b[i][k] -= t*b[j][k]
        try:
            t = 1/a[i][i]
        except ZeroDivisionError:
            return 0, None

        det *= a[i][i]
        for j in range(p):
            b[i][j] *= t
    return det, b


def matmul(a, b):
    """
    Matrix product of matrix a and  matrix b.

    Parameters
    ----------
    a : np.array or list of lists
        'n x m' array
    b : np. array or list of lists
        'm x l' array

    Returns
    -------
    out : list of lists
        The matrix product of the inputs.

    Raises
    ------
    ValueError
        If the number of columns of `a` is not the same as
        the number of rows `b`.

    Notes
    -----
    The output dimension depends on the dimensions of the input.

    - For input matrices a = n x m and b = m x l, the output matrix will have
      dimensions n x l.

    Examples
    --------
    For 2-D arrays:

    >>> a = np.array([[1, 2],
    ...               [3, 4]])
    >>> b = np.array([[5, 1],
    ...               [6, 2]])
    >>> matmul(a, b)
    [[17, 5], [39, 11]]

    For a 2-D array and 1-D array:
    >>> a = np.array([[1, 2],
    ...               [3, 4]])
    >>> b = np.array([5, 6])
    >>> matmul(a, b)
    [17, 39]
    """
    a = np.array(a)
    b = np.array(b)
    # Simplify this                        ##### Remove Line  ##########
    if a.ndim == 1:
        n = 1
        p = a.shape[0]
    else:
        n, p = a.shape
    if b.ndim == 1:
        p1 = 1
        q = b.shape[0]
    else:
        p1, q = b.shape
    if p != q:
        raise ValueError("Incompatible dimensions")
    if b.ndim == 1:
        c = zeromat((n,))
        for i in range(n):
            c[i] = sum(a[i][k]*b[k] for k in range(p))
    else:
        c = zeromat((n, q))
        for i in range(n):
            for j in range(q):
                c[i][j] = sum(a[i][k]*b[k][j] for k in range(p))
    return c


def zeromat(shape):
    """
    Returns an array with dimension shape, filled with zeros.

    Parameters
    ----------
    shape : tuple
        Shape of output matrix

    Returns
    -------
    out : list of lists
        Matrix with dimension p x q

    Examples
    --------
    >>> zeromat((2, 3))
    [[0, 0, 0], [0, 0, 0]]
    >>> zeromat((1, 3))
    [[0, 0, 0]]
    >>> zeromat((2, 1))
    [[0], [0]]
    """
    if len(shape) == 1:
        return [0]*shape[0]
    else:
        return [[0]*shape[1] for i in range(shape[0])]
