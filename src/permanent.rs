use num_complex::Complex;

pub type NumType = Complex<f64>;

pub fn permanent_rs(matrix: Vec<Vec<NumType>>) -> NumType {
    let n = matrix.len();

    let mut row_comb: Vec<NumType> = (0..n).map(|i| (0..n).map(|j| matrix[j][i]).sum()).collect();

    let mut total: NumType = Complex::new(0.0, 0.0);
    let mut old_gray = 0;
    let mut sign = 1.0;
    let num_loops = 2_u64.pow(n as u32 - 1);

    for bin_index in 1..=num_loops {
        let reduced = row_comb.iter().product::<NumType>();
        total += sign * reduced;
        let new_gray = bin_index ^ (bin_index / 2);
        let gray_diff = old_gray ^ new_gray;
        let gray_diff_index = gray_diff.trailing_zeros() as usize;

        let new_vector = &matrix[gray_diff_index];
        let direction =
            (2 * ((old_gray > new_gray) as isize - (old_gray < new_gray) as isize)) as f64;

        for i in 0..n {
            row_comb[i] += new_vector[i] * direction;
        }
        sign = -sign;
        old_gray = new_gray;
    }

    total / Complex::new(num_loops as f64, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::prelude::*;

    const EPSILON: f64 = 0.000001;

    #[test]
    fn test_perm_1() {
        let mut rng = rand::rng();
        let a = Complex::new(rng.random(), 0.0);
        let res = permanent_rs(vec![vec![a]]);
        assert_abs_diff_eq!(res.re, a.re, epsilon = EPSILON);
        assert_abs_diff_eq!(res.im, a.im, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_1_cmplx() {
        let mut rng = rand::rng();
        let a = Complex::new(rng.random(), rng.random());
        let res = permanent_rs(vec![vec![a]]);
        assert_abs_diff_eq!(res.re, a.re, epsilon = EPSILON);
        assert_abs_diff_eq!(res.im, a.im, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_2() {
        let mut rng = rand::rng();
        let values: Vec<Complex<f64>> = (0..4).map(|_| Complex::new(rng.random(), 0.0)).collect();
        let res = permanent_rs(vec![vec![values[0], values[1]], vec![values[2], values[3]]]);
        let expected = values[0] * values[3] + values[1] * values[2];
        assert_abs_diff_eq!(res.re, expected.re, epsilon = EPSILON);
        assert_abs_diff_eq!(res.im, expected.im, epsilon = EPSILON);
    }

    #[test]
    fn test_perm_2_cmplx() {
        let mut rng = rand::rng();
        let values: Vec<Complex<f64>> = (0..4)
            .map(|_| Complex::new(rng.random(), rng.random()))
            .collect();
        let res = permanent_rs(vec![vec![values[0], values[1]], vec![values[2], values[3]]]);
        let expected = values[0] * values[3] + values[1] * values[2];
        assert_abs_diff_eq!(res.re, expected.re, epsilon = EPSILON);
        assert_abs_diff_eq!(res.im, expected.im, epsilon = EPSILON);
    }
}
