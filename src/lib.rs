use pyo3::prelude::*;
pub mod permanent;

#[pymodule]
mod perm_rs {
    use num_complex::Complex;
    use pyo3::prelude::*;

    #[pyfunction]
    fn _permanent_f32(matrix: Vec<Vec<f32>>) -> f32 {
        crate::permanent::permanent(matrix)
    }

    #[pyfunction]
    fn _permanent_f64(matrix: Vec<Vec<f64>>) -> f64 {
        crate::permanent::permanent(matrix)
    }

    #[pyfunction]
    fn _permanent_cf32(matrix: Vec<Vec<Complex<f32>>>) -> Complex<f32> {
        crate::permanent::permanent(matrix)
    }

    #[pyfunction]
    fn _permanent_cf64(matrix: Vec<Vec<Complex<f64>>>) -> Complex<f64> {
        crate::permanent::permanent(matrix)
    }
}
