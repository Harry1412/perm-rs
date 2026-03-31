use pyo3::prelude::*;
pub mod permanent;

#[pymodule]
mod perm_rs {
    use num_complex::Complex;
    use numpy::PyReadonlyArray2;
    use pyo3::prelude::*;

    #[pyfunction]
    fn _permanent_f32<'py>(matrix: PyReadonlyArray2<'py, f32>) -> PyResult<f32> {
        Ok(crate::permanent::permanent(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_f64<'py>(matrix: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        Ok(crate::permanent::permanent(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_cf32<'py>(matrix: PyReadonlyArray2<'py, Complex<f32>>) -> PyResult<Complex<f32>> {
        Ok(crate::permanent::permanent(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_cf64<'py>(matrix: PyReadonlyArray2<'py, Complex<f64>>) -> PyResult<Complex<f64>> {
        Ok(crate::permanent::permanent(matrix.as_array()))
    }
}
