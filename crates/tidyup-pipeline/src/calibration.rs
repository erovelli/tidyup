//! Offline confidence-calibration fitting + measurement.
//!
//! The runtime *apply* lives in [`tidyup_domain::Calibration`] (a no-op by
//! default). This module fits a [`Calibration::Platt`] from labeled
//! `(score, correct)` samples and measures calibration quality via Expected
//! Calibration Error (ECE). It's the bridge from the eval corpus (Stage 1) to a
//! calibrated confidence: `cargo xtask eval --calibrate` collects the samples,
//! calls [`fit_platt`], and reports the fitted parameters plus ECE before/after.
//!
//! Pure math — no models, no I/O — so it runs in CI and carries its own tests.

use tidyup_domain::Calibration;

const FIT_EPOCHS: usize = 500;
const FIT_LR: f64 = 0.5;

/// Fit Platt (logistic) scaling `sigmoid(a·score + b)` to labeled samples by
/// gradient descent on log-loss.
///
/// Returns [`Calibration::Identity`] when there's nothing meaningful to fit —
/// fewer than two samples, or a single class (all correct or all wrong) — since
/// a calibrator can't be estimated from those.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn fit_platt(samples: &[(f32, bool)]) -> Calibration {
    if samples.len() < 2 {
        return Calibration::Identity;
    }
    let positives = samples.iter().filter(|(_, correct)| *correct).count();
    if positives == 0 || positives == samples.len() {
        return Calibration::Identity;
    }

    let sample_count = samples.len() as f64;
    let (mut a, mut b) = (1.0_f64, 0.0_f64);
    for _ in 0..FIT_EPOCHS {
        let (mut grad_a, mut grad_b) = (0.0_f64, 0.0_f64);
        for (score, correct) in samples {
            let score_f = f64::from(*score);
            let target = f64::from(u8::from(*correct));
            let prob = sigmoid(a.mul_add(score_f, b));
            let err = prob - target;
            grad_a += err * score_f;
            grad_b += err;
        }
        a -= FIT_LR * grad_a / sample_count;
        b -= FIT_LR * grad_b / sample_count;
    }
    Calibration::Platt { a, b }
}

/// Expected Calibration Error over `bins` equal-width confidence bins.
///
/// The support-weighted mean gap between each bin's mean confidence and its
/// accuracy; `0.0` is perfectly calibrated. Returns `0.0` for empty input.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn expected_calibration_error(samples: &[(f32, bool)], cal: &Calibration, bins: usize) -> f32 {
    if samples.is_empty() || bins == 0 {
        return 0.0;
    }
    let total = samples.len() as f64;
    let mut conf_sum = vec![0.0_f64; bins];
    let mut correct_sum = vec![0.0_f64; bins];
    let mut count = vec![0.0_f64; bins];

    for (score, correct) in samples {
        let p = f64::from(cal.calibrate(*score)).clamp(0.0, 1.0);
        let idx = ((p * bins as f64) as usize).min(bins - 1);
        conf_sum[idx] += p;
        correct_sum[idx] += f64::from(u8::from(*correct));
        count[idx] += 1.0;
    }

    let mut ece = 0.0_f64;
    for i in 0..bins {
        if count[i] > 0.0 {
            let avg_conf = conf_sum[i] / count[i];
            let accuracy = correct_sum[i] / count[i];
            ece += (count[i] / total) * (avg_conf - accuracy).abs();
        }
    }
    ece as f32
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_precision_loss, clippy::float_cmp)]
mod tests {
    use super::*;

    /// Build a separable sample set: scores below 0.5 are wrong, above are right
    /// (with the boundary respected), so a fit should be monotonically
    /// increasing.
    fn separable() -> Vec<(f32, bool)> {
        let mut v = Vec::new();
        for i in 0..50 {
            let s = i as f32 / 50.0; // 0.0 .. 1.0
            v.push((s, s >= 0.5));
        }
        v
    }

    #[test]
    fn fit_platt_is_monotonically_increasing_on_separable_data() {
        let cal = fit_platt(&separable());
        match cal {
            Calibration::Platt { a, .. } => assert!(a > 0.0, "slope should be positive, got {a}"),
            Calibration::Identity => panic!("expected a fitted Platt calibrator"),
        }
        // Higher raw score => higher calibrated probability.
        assert!(cal.calibrate(0.9) > cal.calibrate(0.1));
        // And probabilities stay in range.
        assert!((0.0..=1.0).contains(&cal.calibrate(0.9)));
        assert!((0.0..=1.0).contains(&cal.calibrate(-2.0)));
    }

    #[test]
    fn fit_platt_returns_identity_for_degenerate_input() {
        assert_eq!(fit_platt(&[]), Calibration::Identity);
        assert_eq!(fit_platt(&[(0.5, true)]), Calibration::Identity);
        // Single class (all correct) — nothing to separate.
        let all_true: Vec<_> = (0..10).map(|i| (i as f32 / 10.0, true)).collect();
        assert_eq!(fit_platt(&all_true), Calibration::Identity);
    }

    #[test]
    fn fitting_reduces_calibration_error() {
        // Raw scores are systematically over-confident: score 0.9 is right only
        // ~50% of the time. Identity ECE should be high; the fit lowers it.
        let mut samples = Vec::new();
        for i in 0..100 {
            let score = 0.9_f32;
            samples.push((score, i % 2 == 0)); // 50% correct at confidence 0.9
        }
        // Add some low-confidence true-negatives so there are two classes overall
        // and a non-trivial mapping to learn.
        for i in 0..100 {
            samples.push((0.1_f32, i % 10 == 0)); // 10% correct at confidence 0.1
        }

        let identity_ece = expected_calibration_error(&samples, &Calibration::Identity, 10);
        let cal = fit_platt(&samples);
        let fitted_ece = expected_calibration_error(&samples, &cal, 10);

        assert!(
            fitted_ece <= identity_ece,
            "fit should not worsen ECE: identity={identity_ece}, fitted={fitted_ece}",
        );
    }

    #[test]
    fn ece_is_zero_for_empty_or_no_bins() {
        assert_eq!(
            expected_calibration_error(&[], &Calibration::Identity, 10),
            0.0
        );
        assert_eq!(
            expected_calibration_error(&[(0.5, true)], &Calibration::Identity, 0),
            0.0
        );
    }
}
