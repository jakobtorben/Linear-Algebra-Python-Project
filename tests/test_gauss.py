import numpy as np
import pytest

from acse_la import gauss



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
          [0.0638888888888889, -0.14444444444444446, 0.14722222222222223]])
    ])
    def test_gauss(self, a, b, dete, xe):
        """ Test the gauss function """
        det, x = gauss(a, b)

        assert np.isclose(det, dete)
        assert np.allclose(x, xe)



