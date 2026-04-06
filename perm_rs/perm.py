from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .perm_rs import (
    _permanent_multi_cf32,
    _permanent_multi_cf64,
    _permanent_multi_f32,
    _permanent_multi_f64,
    _permanent_single_cf32,
    _permanent_single_cf64,
    _permanent_single_f32,
    _permanent_single_f64,
)

T = TypeVar("T", bound=np.float32 | np.float64 | np.complex64 | np.complex128)


def _validate_matrix(matrix: npt.NDArray[T]) -> None:
    """Validates a matrix is suitable for permanent calculation."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Matrix should be provided as a numpy array.")
    if matrix.ndim != 2:
        raise ValueError("Matrix should be two dimensional.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix should be square.")


def permanent(matrix: npt.NDArray[T]) -> T:
    """
    Calculates the permanent for a provided matrix.
    """
    if matrix.shape[0] < 17:
        return permanent_single(matrix)
    return permanent_multi(matrix)


def permanent_single(matrix: npt.NDArray[T]) -> T:
    """
    Calculates the permanent for a provided matrix using a standard
    single-threaded approach.
    """
    _validate_matrix(matrix)
    if matrix.dtype == np.float32:
        return _permanent_single_f32(matrix)
    if matrix.dtype == np.float64:
        return _permanent_single_f64(matrix)
    if matrix.dtype == np.complex64:
        return _permanent_single_cf32(matrix)
    return _permanent_single_cf64(matrix)


def permanent_multi(matrix: npt.NDArray[T]) -> T:
    """
    Calculates the permanent for a provided matrix using all available threads
    on a system.
    """
    _validate_matrix(matrix)
    if matrix.dtype == np.float32:
        return _permanent_multi_f32(matrix)
    if matrix.dtype == np.float64:
        return _permanent_multi_f64(matrix)
    if matrix.dtype == np.complex64:
        return _permanent_multi_cf32(matrix)
    return _permanent_multi_cf64(matrix)
