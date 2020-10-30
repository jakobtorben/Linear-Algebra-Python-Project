import numpy as np
import pytest

from acse_la import gauss, matmul, zeromat


class TestGauss(object):
    """
    Class for testing the Gaussian elimination algorithm Gauss
    and its associated functions.
    """

    @pytest.mark.parametrize('a, b, dete, xe', [
        ([[2, 9, 4], [7, 5, 3], [6, 1, 8]],
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]], -360.0,
         [[-0.10277777777777776, 0.18888888888888888, -0.019444444444444438],
          [0.10555555555555554, 0.02222222222222223, -0.061111111111111116],
          [0.0638888888888889, -0.14444444444444446, 0.14722222222222223]]),
        ([[2, 0, -1], [0, 5, 6], [0, -1, 1]],
         [[2], [1], [2]], 22.0,
         [[1.5], [-1.0], [1.0]]),
        ([[0.5, 1.5], [4.2, 3.9]],
         [[1], [2]], -4.35,
         [[-0.206896551724137931], [0.73563218390804597701]])
    ])
    def test_gauss(self, a, b, dete, xe):
        """ Test the gauss function """
        det, x = gauss(a, b)
        print(det, x)

        assert np.isclose(det, dete)
        assert np.allclose(x, xe)

    @pytest.mark.parametrize('a, b, exp', [
        ([[1, 2], [3, 4]], [[5, 1], [6, 2]], [[17, 5], [39, 11]]),
        ([[1, 2], [3, 4]], [5, 6], [17, 39])
    ])
    def test_matmul(self, a, b, exp):
        M = matmul(a, b)
        assert (M == exp)

    @pytest.mark.parametrize('p, q, exp', [
        (2, 3, [[0, 0, 0], [0, 0, 0]]),
        (1, 3, [[0, 0, 0]]),
        (2, 1, [[0], [0]])
        ])
    def test_zeromat(self, p, q, exp):
        M = zeromat(p, q)

        assert M == exp

