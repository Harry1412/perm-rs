from functools import partial

import numpy as np
import numpy.typing as npt
import pytest
import thewalrus
from scipy.stats import unitary_group

from perm_rs import permanent


def random_unitary(
    n: int, seed: int | None = None
) -> npt.NDArray[np.complex128]:
    """Generates a random unitary of size nxn."""
    if n == 1:
        rng = np.random.RandomState(seed=seed)
        return np.exp(2j * np.pi * rng.random((1, 1)))
    return unitary_group.rvs(n, random_state=seed)


approximate = partial(pytest.approx, abs=1e-6)


@pytest.fixture(params=[1, 2, 5, 10, 15, 20])
def size(request):
    """Size of matrices to use in permanent testing"""
    return request.param


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

    def test_compare_walrus(self, size):
        """
        Checks the correct permanent value is calcualted for a range of values
        of n by comparing to those calculated by thewalrus permanent
        function.
        """
        mat = random_unitary(size)
        p1 = permanent(mat)
        p2 = thewalrus.perm(mat)
        assert p1 == approximate(p2)

    def test_compare_walrus_single(self, size):
        """
        Checks the correct permanent value is calcualted for a range of values
        of n when specifically using single-threading by comparing to those
        calculated by thewalrus permanent function.
        """
        mat = random_unitary(size)
        p1 = permanent(mat)
        p2 = thewalrus.perm(mat)
        assert p1 == approximate(p2)

    def test_compare_walrus_multi(self, size):
        """
        Checks the correct permanent value is calcualted for a range of values
        of n when specifically using multi-threading by comparing to those
        calculated by thewalrus permanent function.
        """
        mat = random_unitary(size)
        p1 = permanent(mat)
        p2 = thewalrus.perm(mat)
        assert p1 == approximate(p2)

    def test_compare_walrus_float(self, size):
        """
        Checks the correct permanent value is calcualted for a range of values
        of n by comparing to those calculated by thewalrus permanent calculation
        when the matrix is a float.
        """
        rng = np.random.RandomState()
        mat = rng.random((size, size))
        mat /= mat.sum(axis=0)
        p1 = permanent(mat)
        p2 = thewalrus.perm(mat)
        assert p1 == approximate(p2)
