use num_complex::Complex;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

mod permanent;
mod profiler;
mod random_unitary;

pub use permanent::{permanent, permanent_multi, permanent_single};
pub use profiler::performance_profiler;

#[pyfunction]
fn _permanent_single_f32<'py>(matrix: PyReadonlyArray2<'py, f32>) -> PyResult<f32> {
    Ok(crate::permanent_single(matrix.as_array()))
}

#[pyfunction]
fn _permanent_single_f64<'py>(matrix: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
    Ok(crate::permanent_single(matrix.as_array()))
}

#[pyfunction]
fn _permanent_single_cf32<'py>(
    matrix: PyReadonlyArray2<'py, Complex<f32>>,
) -> PyResult<Complex<f32>> {
    Ok(crate::permanent_single(matrix.as_array()))
}

#[pyfunction]
fn _permanent_single_cf64<'py>(
    matrix: PyReadonlyArray2<'py, Complex<f64>>,
) -> PyResult<Complex<f64>> {
    Ok(crate::permanent_single(matrix.as_array()))
}

#[pyfunction]
fn _permanent_multi_f32<'py>(matrix: PyReadonlyArray2<'py, f32>) -> PyResult<f32> {
    Ok(crate::permanent_multi(matrix.as_array()))
}

#[pyfunction]
fn _permanent_multi_f64<'py>(matrix: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
    Ok(crate::permanent_multi(matrix.as_array()))
}

#[pyfunction]
fn _permanent_multi_cf32<'py>(
    matrix: PyReadonlyArray2<'py, Complex<f32>>,
) -> PyResult<Complex<f32>> {
    Ok(crate::permanent_multi(matrix.as_array()))
}

#[pyfunction]
fn _permanent_multi_cf64<'py>(
    matrix: PyReadonlyArray2<'py, Complex<f64>>,
) -> PyResult<Complex<f64>> {
    Ok(crate::permanent_multi(matrix.as_array()))
}

/// Runs a comparison between single & multi-threaded performance for a range of
/// n values to find the optimal threshold for switching between the two.
#[pyfunction]
#[pyo3(name = "performance_profiler")]
fn py_performance_profiler() -> PyResult<usize> {
    Ok(crate::performance_profiler())
}

#[pymodule]
fn perm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_permanent_single_f32, m)?)?;
    m.add_function(wrap_pyfunction!(_permanent_single_f64, m)?)?;
    m.add_function(wrap_pyfunction!(_permanent_single_cf32, m)?)?;
    m.add_function(wrap_pyfunction!(_permanent_single_cf64, m)?)?;
    m.add_function(wrap_pyfunction!(_permanent_multi_f32, m)?)?;
    m.add_function(wrap_pyfunction!(_permanent_multi_f64, m)?)?;
    m.add_function(wrap_pyfunction!(_permanent_multi_cf32, m)?)?;
    m.add_function(wrap_pyfunction!(_permanent_multi_cf64, m)?)?;
    m.add_function(wrap_pyfunction!(py_performance_profiler, m)?)?;
    Ok(())
}
