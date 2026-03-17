use pyo3::prelude::*;
pub mod permanent;

#[pymodule]
mod perm_rs {
    use crate::permanent::NumType;
    use pyo3::prelude::*;

    #[pyfunction]
    fn permanent(matrix: Vec<Vec<NumType>>) -> NumType {
        crate::permanent::permanent_rs(matrix)
    }
}
