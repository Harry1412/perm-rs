use ndarray::Array2;
use num_complex::Complex64;
use rand::RngExt;

/// A function to generate a random unitary matrix of size nxn. Note: this is
/// AI generated as it was intended entirely for use within the performance
/// profiler without adding additional dependencies to the library. As such,
/// true randomness is not guaranteed.
pub fn random_unitary(n: usize) -> Array2<Complex64> {
    let mut rng = rand::rng();

    let mut mat = Array2::<Complex64>::zeros((n, n));
    for mut row in mat.rows_mut() {
        for v in row.iter_mut() {
            let u1: f64 = rng.random::<f64>().max(1e-10);
            let u2: f64 = rng.random::<f64>();
            let r = (-2.0_f64 * u1.ln()).sqrt() / 2.0_f64.sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            *v = Complex64::new(r * theta.cos(), r * theta.sin());
        }
    }

    let mut q = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        let mut v = mat.row(i).to_owned();
        for j in 0..i {
            let q_j = q.row(j);
            let proj: Complex64 = q_j.iter().zip(v.iter()).map(|(x, y)| x.conj() * y).sum();
            v -= &(q_j.mapv(|x| x * proj));
        }
        let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        q.row_mut(i).assign(&v.mapv(|x| x / norm));
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_is_unitary() {
        let n = 10;
        let mat = random_unitary(n);
        let mat_conj = mat.mapv(|x| x.conj());
        let product = mat.dot(&mat_conj.t());
        assert_abs_diff_eq!(product.mapv(|x| x.re), Array2::eye(n), epsilon = 1e-9);
        assert_abs_diff_eq!(
            product.mapv(|x| x.im),
            Array2::zeros((n, n)),
            epsilon = 1e-9
        );
    }
}
