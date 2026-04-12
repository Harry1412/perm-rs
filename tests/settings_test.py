import numpy as np
import pytest

from perm_rs.__settings import Settings


class TestPermanentCalculation:
    """
    For testing the correctness of the permanent calculation function.
    """

    def test_multi_threaded_threshold_update(self):
        """
        Checks the multi-threaded threshold is correctly updated when being
        supplied with a valud value
        """
        settings = Settings()
        n = np.random.RandomState().randint(0, 100)
        settings.multi_threaded_threshold = n
        assert settings.multi_threaded_threshold == n

    @pytest.mark.parametrize("value", [1.5, True, [2], -1])
    def test_invalid_multi_threaded_threshold(self, value):
        """
        Checks an exception is raised when invalid thresholds are set.
        """
        settings = Settings()
        with pytest.raises((ValueError, TypeError)):
            settings.multi_threaded_threshold = value
