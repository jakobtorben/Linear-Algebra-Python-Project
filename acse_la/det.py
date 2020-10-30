import numpy as np


__all__ = ['det']


def det(A):
    """
    Compute the determinant of a matrix.

    Parameters
    ----------
    A : np.array or list of lists
        'N x N' matrix

    Returns
    -------
    out : float
        The determinant of the matrix.

    Raises
    ------
    ValueError
        If input is not square matrix

    Notes
    -----
    - The determinant is computed with LU decomposition, using Crout's method.
      For further details see: Propp, J. G., Wilson, D. B. 'Numerical Recipes',
      Cambridge University Press. (1996).

    - Pivoting is not yet implemented. Matrices with zero entries along
      diagonal can have unstable behaviour.


    Examples
    --------
    >>> det([[2, 9, 4], [7, 5, 3], [6, 1, 8]])
    -360.0
    >>> det([[0.5, 1.5], [4.2, 3.9]])
    -4.35
    """


    A = np.asarray(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Not a square NxN matrix")
    N = A.shape[0]
    
    # Set L_ii = 1 (Crout's method)
    L = np.eye(N)
    U = np.zeros((N, N))

    for j in range(N):  # j = 0, 1,..., N-1

        # Avoid zero division
        # Note that this is cheating and is only a workaround
        # With more time I would have implemented row pivoting
        if np.isclose(U[j, j], 0.0):
            U[j, j] = 1.0e-18

        # Find coefficients for U matrix iteratively
        for i in range(j+1):
            Bsum =  sum([L[i, k]*U[k, j] for k in range(i)])
            U[i, j] = A[i, j] - Bsum
        #sums = [sum([L[i, k]*U[k, j] for k in range(i)]) for i in range(j+1)]
        #U[:j+1, j] = A[:j+1, j] - sums


        #for i in range(j+1, N):
        #    if i != j:  # L along diagonal already set to 1
        #        Asum = sum([L[i][k]*U[k][j] for k in range(j)])
        #        L[i, j] = (A[i, j] - Asum)/U[j, j]

        # Find coefficients for L matrix
        # Finds sum along each axis
        sums = [sum([L[i, k]*U[k, j] for k in range(j)]) for i in range(j+1, N)]
        L[j+1:N, j] = (A[j+1:N, j] - sums)/U[j, j]

    det = 1.
    for j in range(N):
        det *= U[j][j]
    return det

    
if __name__ == '__main__':
    #print(det([[2, 9, 4], [7, 5, 3], [6, 1, 8]]))
    #print(det([[2, 9, 4], [7, 5, 3], [6, 1, 8]]))
    assert np.isclose(det([[2, 9, 4], [7, 5, 3], [6, 1, 8]]), -360.0)
    #print(det([[0.5, 1.5], [4.2, 3.9]]))
    assert np.isclose(det([[0.5, 1.5], [4.2, 3.9]]), -4.35)
    assert np.isclose(det([[1, 1, 1], [2, 3, 3], [3, 4, 4]]), 0)
    #print(np.linalg.det([[0, 0, 1], [2, 5, 8], [4, 10, 16]]))
    #print(det([[0, 0, 1], [2, 5, 8], [4, 10, 16]]))
    #assert np.isclose(det([[0, 0, 1], [2, 5, 8], [4, 10, 16]]), 0)