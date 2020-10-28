import pytest
import doctest


@pytest.mark.parametrize('func', [
    'gauss', 'matmul', 'zeromat'
])
def test_docstrings(func):
    print("hei")
    # module = import_module('acse_la.gauss.%s' % func)
    doctest.testmod(name=func)
    print("hade")
