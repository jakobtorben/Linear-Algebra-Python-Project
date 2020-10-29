from importlib import import_module

import pytest
import doctest


def test_docstrings():
    module = import_module('acse_la.gauss')
    assert doctest.testmod(module).failed == 0
