use super::random_unitary::random_unitary;
use super::{permanent_multi, permanent_single};
use std::time::Instant;

const N_REPS: usize = 5;
const N_KEEP: usize = 3;

/// Returns the relative execution speed of single to multi-threaded
/// computation, by taking the fastest n samples from N runs. The ratio returned
/// will be > 1 if single threaded is faster, otherwise < 1.
fn _single_to_multi_ratio(n: usize) -> f64 {
    let mat = random_unitary(n);
    let mut single_times: Vec<f64> = (0..N_REPS)
        .map(|_| {
            let t0 = Instant::now();
            permanent_single(mat.view());
            let t1 = Instant::now();
            (t1 - t0).as_secs_f64()
        })
        .collect();
    single_times.sort_by(f64::total_cmp);
    single_times.truncate(N_KEEP);
    let mut multi_times: Vec<f64> = (0..N_REPS)
        .map(|_| {
            let t0 = Instant::now();
            permanent_multi(mat.view());
            let t1 = Instant::now();
            (t1 - t0).as_secs_f64()
        })
        .collect();
    multi_times.sort_by(f64::total_cmp);
    multi_times.truncate(N_KEEP);
    // (t2 - t1).as_secs_f32() / (t1 - t0).as_secs_f32()
    multi_times.iter().sum::<f64>() / single_times.iter().sum::<f64>()
}

/// Runs a comparison between single & multi-threaded performance for a range of
/// n values to find the optimal threshold for switching between the two.
pub fn performance_profiler() -> Result<usize, &'static str> {
    let mut n = 15;
    let mut rel_speed = _single_to_multi_ratio(n);
    if rel_speed > 1.0 {
        while rel_speed > 1.0 {
            n += 1;
            if n > 25 {
                return Err("Unable to find a threshold for which \
                multi-threading is faster than single-threading within a \
                reasonable matrix size (n <= 25).");
            }
            rel_speed = _single_to_multi_ratio(n)
        }
    } else {
        while rel_speed < 1.0 {
            n -= 1;
            if n == 0 {
                return Err("Unable to find a theshold for which \
                single-threading is faster than multi-threading.");
            }
            rel_speed = _single_to_multi_ratio(n)
        }
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validates that the performance profiler is correctly working (at least
    /// not generating any exceptions).
    #[test]
    fn test_performance_profile() {
        assert!(performance_profiler().is_ok());
    }
}
