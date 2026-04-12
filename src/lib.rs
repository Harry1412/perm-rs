use pyo3::prelude::*;
mod permanent;

pub use permanent::{permanent, permanent_multi, permanent_single};

#[pymodule]
mod perm_rs {
    use num_complex::Complex;
    use numpy::PyReadonlyArray2;
    use pyo3::prelude::*;

    #[pyfunction]
    fn _permanent_single_f32<'py>(matrix: PyReadonlyArray2<'py, f32>) -> PyResult<f32> {
        Ok(crate::permanent::permanent_single(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_single_f64<'py>(matrix: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        Ok(crate::permanent::permanent_single(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_single_cf32<'py>(
        matrix: PyReadonlyArray2<'py, Complex<f32>>,
    ) -> PyResult<Complex<f32>> {
        Ok(crate::permanent::permanent_single(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_single_cf64<'py>(
        matrix: PyReadonlyArray2<'py, Complex<f64>>,
    ) -> PyResult<Complex<f64>> {
        Ok(crate::permanent::permanent_single(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_multi_f32<'py>(matrix: PyReadonlyArray2<'py, f32>) -> PyResult<f32> {
        Ok(crate::permanent::permanent_multi(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_multi_f64<'py>(matrix: PyReadonlyArray2<'py, f64>) -> PyResult<f64> {
        Ok(crate::permanent::permanent_multi(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_multi_cf32<'py>(
        matrix: PyReadonlyArray2<'py, Complex<f32>>,
    ) -> PyResult<Complex<f32>> {
        Ok(crate::permanent::permanent_multi(matrix.as_array()))
    }

    #[pyfunction]
    fn _permanent_multi_cf64<'py>(
        matrix: PyReadonlyArray2<'py, Complex<f64>>,
    ) -> PyResult<Complex<f64>> {
        Ok(crate::permanent::permanent_multi(matrix.as_array()))
    }
}
