use num_complex::ComplexFloat;
use num_traits::{FromPrimitive, One, Zero};
use std::{iter, ops};

pub fn permanent<T>(matrix: Vec<Vec<T>>) -> T
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
    let n = matrix.len();

    let mut row_comb: Vec<T> = (0..n).map(|i| (0..n).map(|j| matrix[j][i]).sum()).collect();

    let mut total = T::zero();
    let mut old_gray = 0;
    let mut sign = T::one();
    let num_loops = 2_u64.pow(n as u32 - 1);

    for bin_index in 1..=num_loops {
        let reduced: T = row_comb.iter().copied().product();
        total += sign * reduced;
        let new_gray = bin_index ^ (bin_index / 2);
        let gray_diff = old_gray ^ new_gray;
        let gray_diff_index = gray_diff.trailing_zeros() as usize;

        let new_vector = &matrix[gray_diff_index];
        let direction =
            T::from_isize(2 * ((old_gray > new_gray) as isize - (old_gray < new_gray) as isize))
                .unwrap();
        for i in 0..n {
            row_comb[i] += new_vector[i] * direction;
        }
        sign = -sign;
        old_gray = new_gray;
    }

    total / T::from_u64(num_loops).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
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
        let res = permanent(vec![vec![a]]);
        assert_abs_diff_eq!(res, a, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_1_cmplx() {
        let mut rng = rand::rng();
        let a: Complex<f64> = Complex::new(rng.random(), rng.random());
        let res = permanent(vec![vec![a]]);
        assert_equal_complex(res, a);
    }

    #[test]
    fn test_perm_2() {
        let mut rng = rand::rng();
        let values: Vec<f64> = (0..4).map(|_| rng.random()).collect();
        let res = permanent(vec![vec![values[0], values[1]], vec![values[2], values[3]]]);
        let expected = values[0] * values[3] + values[1] * values[2];
        assert_abs_diff_eq!(res, expected, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_2_cmplx() {
        let mut rng = rand::rng();
        let values: Vec<Complex<f64>> = (0..4)
            .map(|_| Complex::new(rng.random(), rng.random()))
            .collect();
        let res = permanent(vec![vec![values[0], values[1]], vec![values[2], values[3]]]);
        let expected = values[0] * values[3] + values[1] * values[2];
        assert_equal_complex(res, expected);
    }
}
