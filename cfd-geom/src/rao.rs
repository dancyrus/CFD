//! Digitized Rao table for parabolic-bell wall angles.
//!
//! The table was digitized at a specific gamma (around 1.23–1.25). Keep that in
//! mind when combining theta_e with the divergence factor lambda = (1+cos
//! theta_e)/2 — using it at a very different gamma is quietly inconsistent
//! (physics-reference §6, §10).

/// Abscissa: area ratio epsilon.
const EPS: [f64; 8] = [4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0];
/// Ordinate rows: bell percent.
const BELL: [f64; 3] = [0.60, 0.80, 0.90];

/// theta_n rows for 60/80/90% bells, degrees.
/// The commonly circulated 60% row is non-monotonic (37.1 at eps=40 -> 35.0 at
/// eps=50 — a digitization typo). The eps=50 value here is corrected to 37.5.
const TN: [[f64; 8]; 3] = [
    [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 37.5, 40.0],
    [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5],
    [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0],
];
/// theta_e rows for 60/80/90% bells, degrees.
const TE: [[f64; 8]; 3] = [
    [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2],
    [14.0, 13.0, 11.0, 9.0, 8.5, 8.0, 7.5, 7.0],
    [11.5, 10.5, 8.0, 7.0, 6.5, 6.0, 6.0, 6.0],
];

/// Interpolate one row log-linearly in epsilon. `eps` must already be clamped
/// to the table range.
fn interp_eps(row: &[f64; 8], eps: f64) -> f64 {
    let k = EPS.partition_point(|&e| e <= eps).clamp(1, EPS.len() - 1);
    let (e0, e1) = (EPS[k - 1], EPS[k]);
    let t = (eps.ln() - e0.ln()) / (e1.ln() - e0.ln());
    row[k - 1] + t * (row[k] - row[k - 1])
}

/// `(theta_n, theta_e)` in DEGREES for a Rao parabolic bell.
///
/// Log-linear interpolation in area ratio, linear in bell percent — there is no
/// published polynomial fit; every implementation uses a table. Inputs are
/// clamped to `area_ratio` in [4, 100] and `bell_percent` in [0.6, 0.9]; the
/// published data does not support extrapolation (in particular not 100% bells).
pub fn rao_angles(area_ratio: f64, bell_percent: f64) -> (f64, f64) {
    let eps = area_ratio.clamp(EPS[0], EPS[EPS.len() - 1]);
    let bp = bell_percent.clamp(BELL[0], BELL[BELL.len() - 1]);

    let k = BELL.partition_point(|&b| b <= bp).clamp(1, BELL.len() - 1);
    let (b0, b1) = (BELL[k - 1], BELL[k]);
    let t = (bp - b0) / (b1 - b0);

    let tn0 = interp_eps(&TN[k - 1], eps);
    let tn1 = interp_eps(&TN[k], eps);
    let te0 = interp_eps(&TE[k - 1], eps);
    let te1 = interp_eps(&TE[k], eps);
    (tn0 + t * (tn1 - tn0), te0 + t * (te1 - te0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The session brief's discriminating test: at eps = 25, 80% bell, log-linear
    /// interpolation gives 29.4604 / 8.7248. Linear-in-eps would give
    /// 29.400 / 8.750, which is how you tell the two apart.
    #[test]
    fn log_linear_at_eps_25_80_percent() {
        let (tn, te) = rao_angles(25.0, 0.8);
        assert!((tn - 29.4604).abs() < 5e-4, "theta_n = {tn}");
        assert!((te - 8.7248).abs() < 5e-4, "theta_e = {te}");
        // Explicitly not the linear-in-eps values.
        assert!((tn - 29.400).abs() > 5e-2);
        assert!((te - 8.750).abs() > 2e-2);
    }

    #[test]
    fn anchors_reproduce_exactly() {
        let (tn, te) = rao_angles(20.0, 0.8);
        assert!((tn - 28.8).abs() < 1e-12 && (te - 9.0).abs() < 1e-12);
        let (tn, te) = rao_angles(4.0, 0.6);
        assert!((tn - 26.5).abs() < 1e-12 && (te - 20.5).abs() < 1e-12);
        let (tn, te) = rao_angles(100.0, 0.9);
        assert!((tn - 32.0).abs() < 1e-12 && (te - 6.0).abs() < 1e-12);
    }

    #[test]
    fn inputs_clamp_to_table_range() {
        assert_eq!(rao_angles(1.5, 0.8), rao_angles(4.0, 0.8));
        assert_eq!(rao_angles(500.0, 0.8), rao_angles(100.0, 0.8));
        assert_eq!(rao_angles(25.0, 0.2), rao_angles(25.0, 0.6));
        assert_eq!(rao_angles(25.0, 1.0), rao_angles(25.0, 0.9));
    }

    /// theta_n > theta_e everywhere reachable — the bell construction divides by
    /// tan(theta_n) - tan(theta_e).
    #[test]
    fn tn_exceeds_te_everywhere() {
        for i in 0..40 {
            let eps = 4.0 * (25.0f64).powf(i as f64 / 39.0); // log sweep 4..100
            for j in 0..7 {
                let bp = 0.6 + 0.3 * j as f64 / 6.0;
                let (tn, te) = rao_angles(eps, bp);
                assert!(tn > te + 1.0, "tn {tn} te {te} at eps {eps} bp {bp}");
            }
        }
    }

    /// Guard against reintroducing the circulated non-monotonic 60% row.
    #[test]
    fn tn_60_row_is_monotone_in_eps() {
        let mut prev = 0.0;
        for i in 0..60 {
            let eps = 4.0 * (25.0f64).powf(i as f64 / 59.0);
            let (tn, _) = rao_angles(eps, 0.6);
            assert!(tn >= prev - 1e-12, "tn not monotone at eps {eps}");
            prev = tn;
        }
    }
}
