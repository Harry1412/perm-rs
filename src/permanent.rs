use ndarray::ArrayView2;
use num_complex::ComplexFloat;
use num_traits::{FromPrimitive, One, Zero};
use rayon::prelude::*;
use std::{cmp, iter, ops};

/// Custom trait for wrapping all traits required for an object to be compatible
/// with the permanent functions.
pub trait SupportsPermanent:
    ComplexFloat
    + iter::Sum
    + iter::Product
    + ops::Mul<Output = Self>
    + ops::MulAssign
    + ops::AddAssign
    + FromPrimitive
    + Zero
    + One
{
}
impl<T> SupportsPermanent for T where
    T: ComplexFloat
        + iter::Sum
        + iter::Product
        + ops::Mul<Output = Self>
        + ops::MulAssign
        + ops::AddAssign
        + FromPrimitive
        + Zero
        + One
{
}

/// Runs loops required for grey code across the provided range of indices.
fn gray_loops<T>(
    matrix: &ArrayView2<T>,
    row_comb: &mut [T],
    range: ops::RangeInclusive<u64>,
    start_positive_sign: bool,
) -> T
where
    T: SupportsPermanent,
{
    let n = matrix.nrows();

    let mut sign = match start_positive_sign {
        true => T::one(),
        false => -T::one(),
    };
    let mut total = T::zero();
    let mut old_gray = 0;

    for bin_index in range {
        let reduced: T = row_comb.iter().copied().product();
        total += sign * reduced;
        let new_gray = bin_index ^ (bin_index >> 1);
        let gray_diff = old_gray ^ new_gray;
        let gray_diff_index = gray_diff.trailing_zeros() as usize;

        let new_vector = matrix.row(gray_diff_index);
        let direction = T::from_i8(match old_gray.cmp(&new_gray) {
            cmp::Ordering::Greater => 2,
            cmp::Ordering::Less => -2,
            cmp::Ordering::Equal => 0,
        })
        .unwrap();

        for i in 0..n {
            row_comb[i] += new_vector[i] * direction;
        }

        sign = -sign;
        old_gray = new_gray;
    }
    total
}

/// Compute the permanent using single-threading. This is a recreation of code
/// from thewalrus Python module.
pub fn permanent_single<T>(matrix: ArrayView2<T>) -> T
where
    T: SupportsPermanent,
{
    match matrix.nrows().cmp(&3) {
        cmp::Ordering::Greater => _permanent_single(matrix),
        _ => permanent_exact(matrix),
    }
}

pub fn _permanent_single<T>(matrix: ArrayView2<T>) -> T
where
    T: SupportsPermanent,
{
    let num_loops = 2_u64.pow(matrix.nrows() as u32 - 1);

    let mut row_comb = matrix.sum_axis(ndarray::Axis(0)).to_vec();
    let total = gray_loops(&matrix, &mut row_comb, 1..=num_loops, true);

    total / T::from_u64(num_loops).unwrap()
}

/// Compute the permanent using multi-threading, modifying the single-threaded
/// code using the techniques discussed in
/// https://doi.org/10.48550/arXiv.2602.10141 to improve performance.
pub fn permanent_multi<T>(matrix: ArrayView2<T>) -> T
where
    T: SupportsPermanent + Send + Sync,
{
    let n = matrix.nrows();
    let num_loops = 2_u64.pow(n as u32 - 1);
    let num_threads = rayon::current_num_threads() as u64;
    let chunk_size = num_loops.div_ceil(num_threads);
    let matrix_t = matrix.t();

    let total: T = (0..num_threads.min(num_loops))
        .into_par_iter()
        .map(|thread_id| {
            let start = thread_id * chunk_size + 1;
            let end = ((thread_id + 1) * chunk_size).min(num_loops);

            let init_old_gray = (start - 1) ^ ((start - 1) >> 1);
            let mut row_comb: Vec<T> = matrix_t
                .outer_iter()
                .map(|col| {
                    col.iter()
                        .enumerate()
                        .map(|(row, val)| {
                            if init_old_gray & (1 << row) != 0 {
                                -*val
                            } else {
                                *val
                            }
                        })
                        .sum()
                })
                .collect();

            gray_loops(
                &matrix,
                &mut row_comb,
                start..=end,
                !start.is_multiple_of(2), // Determine initial sign based on starting bin_index
            )
        })
        .sum();

    total / T::from_u64(num_loops).unwrap()
}

/// Exact implementations of the permanent calculations for n <= 3.
pub fn permanent_exact<T>(matrix: ArrayView2<T>) -> T
where
    T: SupportsPermanent,
{
    match matrix.nrows() {
        1 => matrix[[0, 0]],
        2 => matrix[[0, 0]] * matrix[[1, 1]] + matrix[[0, 1]] * matrix[[1, 0]],
        3 => {
            matrix[[0, 0]] * matrix[[1, 1]] * matrix[[2, 2]]
                + matrix[[0, 1]] * matrix[[1, 2]] * matrix[[2, 0]]
                + matrix[[0, 2]] * matrix[[1, 0]] * matrix[[2, 1]]
                + matrix[[0, 2]] * matrix[[1, 1]] * matrix[[2, 0]]
                + matrix[[0, 1]] * matrix[[1, 0]] * matrix[[2, 2]]
                + matrix[[0, 0]] * matrix[[1, 2]] * matrix[[2, 1]]
        }
        _ => panic!("Exact permanent not implemented for n > 3."),
    }
}

/// Computes the permanent of a provided matrix, switching between single &
/// multi-threading based on an internally set threshold to optimise
/// performance.
pub fn permanent<T>(matrix: ArrayView2<T>) -> T
where
    T: SupportsPermanent + Send + Sync,
{
    if matrix.nrows() < 17 {
        _permanent_single(matrix)
    } else {
        permanent_multi(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;
    use num_complex::Complex;
    use rand::prelude::*;

    const EPSILON: f64 = 0.000001;

    fn assert_equal_complex(v1: Complex<f64>, v2: Complex<f64>) {
        assert_abs_diff_eq!(v1.re, v2.re, epsilon = EPSILON);
        assert_abs_diff_eq!(v1.im, v2.im, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_1() {
        let mut rng = rand::rng();
        let a: f64 = rng.random();
        let res = _permanent_single(Array2::from_elem((1, 1), a).view());
        assert_abs_diff_eq!(res, a, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_1_cmplx() {
        let mut rng = rand::rng();
        let a: Complex<f64> = Complex::new(rng.random(), rng.random());
        let res = _permanent_single(Array2::from_elem((1, 1), a).view());
        assert_equal_complex(res, a);
    }

    #[test]
    fn test_perm_2() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((2, 2), |_| rng.random::<f64>());
        let res = _permanent_single(matrix.view());
        let expected = permanent_exact(matrix.view());
        assert_abs_diff_eq!(res, expected, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_2_cmplx() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((2, 2), |_| {
            Complex::new(rng.random::<f64>(), rng.random::<f64>())
        });
        let res = _permanent_single(matrix.view());
        let expected = permanent_exact(matrix.view());
        assert_equal_complex(res, expected);
    }

    #[test]
    fn test_perm_3() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((3, 3), |_| rng.random::<f64>());
        let res = _permanent_single(matrix.view());
        let expected = permanent_exact(matrix.view());
        assert_abs_diff_eq!(res, expected, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_3_cmplx() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((3, 3), |_| {
            Complex::new(rng.random::<f64>(), rng.random::<f64>())
        });
        let res = _permanent_single(matrix.view());
        let expected = permanent_exact(matrix.view());
        assert_equal_complex(res, expected);
    }

    #[test]
    fn test_perm_multi_1() {
        let mut rng = rand::rng();
        let a: f64 = rng.random();
        let res = permanent_multi(Array2::from_elem((1, 1), a).view());
        assert_abs_diff_eq!(res, a, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_multi_1_cmplx() {
        let mut rng = rand::rng();
        let a: Complex<f64> = Complex::new(rng.random(), rng.random());
        let res = permanent_multi(Array2::from_elem((1, 1), a).view());
        assert_equal_complex(res, a);
    }

    #[test]
    fn test_perm_multi_2() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((2, 2), |_| rng.random::<f64>());
        let res = permanent_multi(matrix.view());
        let expected = permanent_exact(matrix.view());
        assert_abs_diff_eq!(res, expected, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_multi_2_cmplx() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((2, 2), |_| {
            Complex::new(rng.random::<f64>(), rng.random::<f64>())
        });
        let res = permanent_multi(matrix.view());
        let expected = permanent_exact(matrix.view());
        assert_equal_complex(res, expected);
    }

    #[test]
    fn test_perm_multi_3() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((3, 3), |_| rng.random::<f64>());
        let res = permanent_multi(matrix.view());
        let expected = permanent_exact(matrix.view());
        assert_abs_diff_eq!(res, expected, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_multi_3_cmplx() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((3, 3), |_| {
            Complex::new(rng.random::<f64>(), rng.random::<f64>())
        });
        let res = permanent_multi(matrix.view());
        let expected = permanent_exact(matrix.view());
        assert_equal_complex(res, expected);
    }
}
