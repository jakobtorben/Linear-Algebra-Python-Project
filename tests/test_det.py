import numpy as np
import pytest

from acse_la import det


class TestDet(object):
    """
    Class for testing the Gaussian elimination algorithm Gauss
    and its associated functions.
    """
    M = np.random.rand(100, 100)

    @pytest.mark.parametrize('a, dete', [
        ([[2, 9, 4], [7, 5, 3], [6, 1, 8]], -360.0),
        ([[2, 0, -1], [0, 5, 6], [0, -1, 1]], 22.0),
        ([[0.5, 1.5], [4.2, 3.9]], -4.35),
        ([[1, 1, 1], [2, 3, 3], [3, 4, 4]], 0.),
        (M, np.linalg.det(M))
    ])
    def test_det(self, a, dete):
        """ Test the gauss function """
        detc = det(a)
        assert np.isclose(detc, dete)

