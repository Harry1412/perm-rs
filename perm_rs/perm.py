from .perm_rs import (
    _permanent_f32,
    _permanent_f64,
    _permanent_cf32,
    _permanent_cf64,
)
import numpy as np

def permanent(matrix):
    if isinstance(matrix, np.ndarray):
        if matrix.dtype == np.float32:
            return _permanent_f32(matrix)
        if matrix.dtype == np.float64:
            return _permanent_f64(matrix)
        if matrix.dtype == np.complex64:
            return _permanent_cf32(matrix)
        else:
            return _permanent_cf64(matrix)
    return _permanent_f64(matrix)