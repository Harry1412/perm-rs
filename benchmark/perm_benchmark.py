import pytest
import thewalrus
from scipy.stats import unitary_group

from perm_rs import permanent

CONFIG = {"warmup_rounds": 1, "rounds": 25}


@pytest.fixture(params=[5, 10, 15, 20])
def size(request):
    return request.param


def test_permanent(benchmark, size):
    """
    Perform benchmarking of the perm_rs permanent function for a given matrix
    size.
    """

    def setup():
        return (unitary_group.rvs(size),), {}

    benchmark.pedantic(permanent, setup=setup, **CONFIG)


def test_walrus(benchmark, size):
    """
    Perform benchmarking of the walrus permanent function for a given matrix
    size.
    """

    def setup():
        return (unitary_group.rvs(size),), {}

    benchmark.pedantic(thewalrus.perm, setup=setup, **CONFIG)
