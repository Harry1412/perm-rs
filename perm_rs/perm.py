from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .perm_rs import (
    _permanent_cf32,
    _permanent_cf64,
    _permanent_f32,
    _permanent_f64,
)

T = TypeVar("T", bound=np.float32 | np.float64 | np.complex64 | np.complex128)


def permanent(matrix: npt.NDArray[T]) -> T:
    """
    Calculates the permanent for a provided matrix.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Provided matrix should be a numpy array.")
    if matrix.dtype == np.float32:
        return _permanent_f32(matrix)
    if matrix.dtype == np.float64:
        return _permanent_f64(matrix)
    if matrix.dtype == np.complex64:
        return _permanent_cf32(matrix)
    return _permanent_cf64(matrix)
