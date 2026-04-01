from functools import partial

import numpy as np
import pytest
import thewalrus
from scipy.stats import unitary_group

from perm_rs import permanent

random_unitary = unitary_group.rvs

approximate = partial(pytest.approx, abs=1e-6)


class TestPermanentCalculation:
    """
    For testing the correctness of the permanent calculation function.
    """

    def test_correct_1(self):
        """Checks the correct permanent value is calculated for a 1x1 unitary"""
        mat = random_unitary(1)
        p = permanent(mat)
        assert p == approximate(mat[0, 0])

    def test_correct_2(self):
        """Checks the correct permanent value is calculated for a 1x1 unitary"""
        mat = random_unitary(2)
        p = permanent(mat)
        assert p == approximate(mat[0, 0] * mat[1, 1] + mat[0, 1] * mat[1, 0])

    @pytest.mark.parametrize("n", [3, 10, 11])
    def test_compare_walrus(self, n):
        """
        Checks the correct permanent value is calcualted for a range of values
        of n by comparing to those calculated by thewalrus permanent
        calculation.
        """
        mat = random_unitary(n)
        p1 = permanent(mat)
        p2 = thewalrus.perm(mat)
        assert p1 == approximate(p2)

    @pytest.mark.parametrize("n", [3, 10, 11])
    def test_compare_walrus_float(self, n):
        """
        Checks the correct permanent value is calcualted for a range of values
        of n by comparing to those calculated by thewalrus permanent calculation
        when the matrix is a float.
        """
        rng = np.random.RandomState()
        mat = rng.random((n, n))
        p1 = permanent(mat)
        p2 = thewalrus.perm(mat)
        assert p1 == approximate(p2)
