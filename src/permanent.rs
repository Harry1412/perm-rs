use ndarray::ArrayView2;
use num_complex::ComplexFloat;
use num_traits::{FromPrimitive, One, Zero};
use rayon::prelude::*;
use std::{iter, ops};

fn gray_loops<T>(
    matrix: &ArrayView2<T>,
    row_comb: &mut [T],
    range: ops::RangeInclusive<u64>,
    mut sign: T,
) -> T
where
    T: ComplexFloat + ops::AddAssign + iter::Product + FromPrimitive,
{
    let n = matrix.ncols();

    let mut total = T::zero();
    let mut old_gray = 0;
    for bin_index in range {
        let reduced: T = row_comb.iter().copied().product();
        total += sign * reduced;
        let new_gray = bin_index ^ (bin_index >> 1);
        let gray_diff = old_gray ^ new_gray;
        let gray_diff_index = gray_diff.trailing_zeros() as usize;

        let new_vector = matrix.row(gray_diff_index);
        let direction =
            T::from_isize(2 * ((old_gray > new_gray) as isize - (old_gray < new_gray) as isize))
                .unwrap();
        for i in 0..n {
            row_comb[i] += new_vector[i] * direction;
        }
        sign = -sign;
        old_gray = new_gray;
    }
    total
}

pub fn permanent_single<T>(matrix: ArrayView2<T>) -> T
where
    T: ComplexFloat
        + iter::Sum
        + iter::Product
        + ops::Mul<Output = T>
        + ops::MulAssign
        + ops::AddAssign
        + FromPrimitive
        + Zero
        + One,
{
    let n = matrix.ncols();
    let num_loops = 2_u64.pow(n as u32 - 1);

    let mut row_comb: Vec<T> = (0..n)
        .map(|i| (0..n).map(|j| matrix[[j, i]]).sum())
        .collect();

    let total = gray_loops(&matrix, &mut row_comb, 1..=num_loops, T::one());

    total / T::from_u64(num_loops).unwrap()
}

pub fn permanent_multi<T>(matrix: ArrayView2<T>) -> T
where
    T: ComplexFloat
        + iter::Sum
        + iter::Product
        + ops::Mul<Output = T>
        + ops::MulAssign
        + ops::AddAssign
        + FromPrimitive
        + Zero
        + One
        + Send
        + Sync,
{
    let n = matrix.ncols();
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
            let mut row_comb: Vec<T> = (0..n)
                .map(|col| {
                    (0..n)
                        .map(|row| {
                            let val = matrix_t[[col, row]];
                            if (init_old_gray >> row) & 1 == 1 {
                                -val
                            } else {
                                val
                            }
                        })
                        .sum()
                })
                .collect();

            // Determine initial sign based on starting bin_index
            let sign = match (start - 1).is_multiple_of(2) {
                true => T::one(),
                false => -T::one(),
            };

            gray_loops(&matrix, &mut row_comb, start..=end, sign)
        })
        .sum();

    total / T::from_u64(num_loops).unwrap()
}

pub fn permanent<T>(matrix: ArrayView2<T>) -> T
where
    T: ComplexFloat
        + iter::Sum
        + iter::Product
        + ops::Mul<Output = T>
        + ops::MulAssign
        + ops::AddAssign
        + FromPrimitive
        + Zero
        + One
        + Send
        + Sync,
{
    if matrix.ncols() < 20 {
        permanent_single(matrix)
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
        let res = permanent_single(Array2::from_elem((1, 1), a).view());
        assert_abs_diff_eq!(res, a, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_1_cmplx() {
        let mut rng = rand::rng();
        let a: Complex<f64> = Complex::new(rng.random(), rng.random());
        let res = permanent_single(Array2::from_elem((1, 1), a).view());
        assert_equal_complex(res, a);
    }

    #[test]
    fn test_perm_2() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((2, 2), |_| rng.random::<f64>());
        let res = permanent_single(matrix.view());
        let expected = matrix[[0, 0]] * matrix[[1, 1]] + matrix[[0, 1]] * matrix[[1, 0]];
        assert_abs_diff_eq!(res, expected, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_2_cmplx() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((2, 2), |_| {
            Complex::new(rng.random::<f64>(), rng.random::<f64>())
        });
        let res = permanent_single(matrix.view());
        let expected = matrix[[0, 0]] * matrix[[1, 1]] + matrix[[0, 1]] * matrix[[1, 0]];
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
        let expected = matrix[[0, 0]] * matrix[[1, 1]] + matrix[[0, 1]] * matrix[[1, 0]];
        assert_abs_diff_eq!(res, expected, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_multi_2_cmplx() {
        let mut rng = rand::rng();
        let matrix = Array2::from_shape_fn((2, 2), |_| {
            Complex::new(rng.random::<f64>(), rng.random::<f64>())
        });
        let res = permanent_multi(matrix.view());
        let expected = matrix[[0, 0]] * matrix[[1, 1]] + matrix[[0, 1]] * matrix[[1, 0]];
        assert_equal_complex(res, expected);
    }
}
