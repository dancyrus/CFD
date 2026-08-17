//! Parametric nozzle contours: conical and Rao parabolic bell.
//!
//! All construction happens in units of r_t (throat radius = 1) and is scaled to
//! metres at the end. z = 0 is the chamber head; the throat sits at
//! z_t = chamber + converging cone + upstream arc, and is always an exact sample
//! point of the returned polyline.
//!
//! The geometry itself now lives on [`NozzleCurve`](crate::NozzleCurve) — this
//! module is the fixed-sample-count entry point that `cfd-core`'s acceptance
//! ladder and diagnostics call, kept bit-identical on purpose.

use crate::curve::NozzleCurve;
use crate::{NozzleSpec, WallProfile};
use cfd_contract::Result;

/// Generate the full wall contour as a polyline with roughly `samples` points
/// (clamped to at least 64). The throat is exactly a vertex; `throat_index`
/// points at it.
///
/// A thin wrapper over [`NozzleCurve::tessellate_fixed`] on the standard 2 r_t
/// chamber straight — bit-identical output to the pre-curve generator, which is
/// what keeps `cfd-core`'s recorded ladder results comparable across this
/// change. Callers that want the wall sized to their mesh use
/// [`NozzleCurve::tessellate`] instead.
pub fn generate_contour(spec: &NozzleSpec, samples: usize) -> Result<WallProfile> {
    NozzleCurve::new(*spec).tessellate_fixed(samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::l_c15;
    use crate::rao::rao_angles;
    use crate::ContourKind;

    fn spec(contour: ContourKind, eps: f64) -> NozzleSpec {
        NozzleSpec {
            throat_radius_m: 0.05,
            area_ratio: eps,
            contraction_ratio: 4.0,
            converge_half_angle_deg: 30.0,
            throat_arc_up: 1.5,
            throat_arc_down: 0.382,
            contour,
        }
    }

    /// Wall slope at polyline index i (central where possible).
    fn slope_at(p: &WallProfile, i: usize) -> f64 {
        let pts = &p.points;
        let (a, b) = if i == 0 {
            (pts[0], pts[1])
        } else if i + 1 == pts.len() {
            (pts[i - 1], pts[i])
        } else {
            (pts[i - 1], pts[i + 1])
        };
        (b[1] - a[1]) / (b[0] - a[0])
    }

    #[test]
    fn conical_throat_station_matches_reference() {
        // physics-reference §8: 2 chamber + converging cone + arcs = 4.13 r_t.
        let p = generate_contour(
            &spec(
                ContourKind::Conical {
                    half_angle_deg: 15.0,
                },
                8.0,
            ),
            256,
        )
        .unwrap();
        let rt = 0.05;
        let throat = p.points[p.throat_index];
        assert!(
            (throat[1] / rt - 1.0).abs() < 1e-12,
            "throat r = {}",
            throat[1] / rt
        );
        assert!(
            (throat[0] / rt - 4.1339746).abs() < 1e-3,
            "z_t = {}",
            throat[0] / rt
        );
    }

    #[test]
    fn conical_exit_lands_on_exact_length_and_radius() {
        let s = spec(
            ContourKind::Conical {
                half_angle_deg: 15.0,
            },
            8.0,
        );
        let p = generate_contour(&s, 256).unwrap();
        let rt = s.throat_radius_m;
        let last = *p.points.last().unwrap();
        assert!((last[1] - s.exit_radius_m()).abs() < 1e-12 * rt);
        // L_n identity: exit z - throat z.
        let alpha = 15.0f64.to_radians();
        let l_n = ((8.0f64.sqrt() - 1.0) + 0.382 * (1.0 / alpha.cos() - 1.0)) / alpha.tan();
        let z_span = (last[0] - p.points[p.throat_index][0]) / rt;
        assert!(
            (z_span - l_n).abs() < 1e-12,
            "L_n = {z_span}, expected {l_n}"
        );
        // Exit wall slope is tan(alpha).
        let m = slope_at(&p, p.points.len() - 1);
        assert!((m - alpha.tan()).abs() < 1e-9);
    }

    #[test]
    fn bell_end_slopes_match_rao_angles() {
        let s = spec(ContourKind::ParabolicBell { bell_percent: 0.8 }, 25.0);
        let p = generate_contour(&s, 4096).unwrap();
        let rt = s.throat_radius_m;
        let (tn, te) = rao_angles(25.0, 0.8);
        // Exit point and slope.
        let last = *p.points.last().unwrap();
        assert!((last[1] / rt - 5.0).abs() < 1e-12); // sqrt(25)
        let m_exit = slope_at(&p, p.points.len() - 1);
        assert!(
            (m_exit - te.to_radians().tan()).abs() < 2e-3,
            "exit slope {m_exit} vs tan(theta_e) {}",
            te.to_radians().tan()
        );
        // Slope just downstream of the arc/Bezier junction is ~tan(theta_n).
        // Find the vertex closest to N_z = z_t + 0.382 sin(tn).
        let zt = p.points[p.throat_index][0] / rt;
        let n_z = zt + 0.382 * tn.to_radians().sin();
        let i = p
            .points
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let da = (a.1[0] / rt - n_z).abs();
                let db = (b.1[0] / rt - n_z).abs();
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
            .0;
        let m_n = slope_at(&p, i);
        assert!(
            (m_n - tn.to_radians().tan()).abs() < 2e-2,
            "junction slope {m_n} vs tan(theta_n) {}",
            tn.to_radians().tan()
        );
        // 80% bell is shorter than the equivalent 15-degree cone. The
        // reference cone is Huzel–Huang's: 1.5 R_t downstream throat arc, NOT
        // this nozzle's 0.382 (the two differ by 0.98% at eps = 25).
        let s15 = 15.0f64.to_radians();
        let l_c15 = ((5.0 - 1.0) + 1.5 * (1.0 / s15.cos() - 1.0)) / s15.tan();
        let l_n = (last[0] - p.points[p.throat_index][0]) / rt;
        assert!(
            (l_n - 0.8 * l_c15).abs() < 1e-9,
            "L_n = {l_n}, L_c15 = {l_c15}"
        );
    }

    /// The H&H reference cone is a fixed definition (1.5 R_t arc): changing
    /// the nozzle's OWN downstream arc must not move the length "80% bell"
    /// means, or the percentage is not comparable between engines.
    #[test]
    fn bell_length_reference_is_independent_of_the_nozzle_throat_arc() {
        let mut a = spec(ContourKind::ParabolicBell { bell_percent: 0.8 }, 25.0);
        a.throat_arc_down = 0.382;
        let mut b = a;
        b.throat_arc_down = 0.300;
        let len = |s: &NozzleSpec| {
            let p = generate_contour(s, 512).unwrap();
            (p.points.last().unwrap()[0] - p.points[p.throat_index][0]) / s.throat_radius_m
        };
        assert!((len(&a) - len(&b)).abs() < 1e-12, "{} vs {}", len(&a), len(&b));
        // And it is the 1.5 R_t form, 0.98% longer than the 0.382 R_t one.
        let s15 = 15.0f64.to_radians();
        let with_own_arc = 0.8 * ((4.0) + 0.382 * (1.0 / s15.cos() - 1.0)) / s15.tan();
        assert!(
            (len(&a) / with_own_arc - 1.00983).abs() < 1e-4,
            "ratio {}",
            len(&a) / with_own_arc
        );
    }

    /// The direct-angle path: wall angles are inputs, the throat arc is the
    /// spec's (Raptor 2's 0.300 R_t, not the Rao construction's 0.382), and
    /// the produced wall meets both angles exactly.
    #[test]
    fn direct_bell_reproduces_its_input_angles() {
        let mut s = spec(
            ContourKind::DirectBell {
                theta_n_deg: 32.0,
                theta_e_deg: 6.0,
                length_fraction: 0.76,
            },
            34.3,
        );
        s.throat_arc_down = 0.300;
        let p = generate_contour(&s, 4096).unwrap();
        let rt = s.throat_radius_m;
        let last = *p.points.last().unwrap();
        assert!((last[1] / rt - 34.3f64.sqrt()).abs() < 1e-12);
        // Exit slope is tan(theta_e) — the whole point of the path.
        let m_exit = slope_at(&p, p.points.len() - 1);
        assert!(
            (m_exit.atan().to_degrees() - 6.0).abs() < 0.05,
            "exit angle {} deg",
            m_exit.atan().to_degrees()
        );
        // Slope at the arc/Bezier junction N is tan(theta_n).
        let zt = p.points[p.throat_index][0] / rt;
        let n_z = zt + 0.300 * 32.0f64.to_radians().sin();
        let i = p
            .points
            .iter()
            .enumerate()
            .min_by(|a, b| {
                let (da, db) = ((a.1[0] / rt - n_z).abs(), (b.1[0] / rt - n_z).abs());
                da.partial_cmp(&db).unwrap()
            })
            .unwrap()
            .0;
        let m_n = slope_at(&p, i).atan().to_degrees();
        assert!((m_n - 32.0).abs() < 0.5, "junction angle {m_n} deg");
        // Length is the requested fraction of the H&H reference cone.
        let l_n = (last[0] - p.points[p.throat_index][0]) / rt;
        assert!((l_n - 0.76 * l_c15(34.3)).abs() < 1e-9, "L_n = {l_n}");
    }

    #[test]
    fn direct_bell_rejects_impossible_angles() {
        // theta_e >= theta_n: the Bezier control point divides by their
        // tangent difference.
        assert!(generate_contour(
            &spec(
                ContourKind::DirectBell {
                    theta_n_deg: 6.0,
                    theta_e_deg: 32.0,
                    length_fraction: 0.76
                },
                34.3
            ),
            256
        )
        .is_err());
        // A length far too short for the area ratio puts Q outside [N, E].
        assert!(generate_contour(
            &spec(
                ContourKind::DirectBell {
                    theta_n_deg: 32.0,
                    theta_e_deg: 6.0,
                    length_fraction: 0.2
                },
                34.3
            ),
            256
        )
        .is_err());
        assert!(generate_contour(
            &spec(
                ContourKind::DirectBell {
                    theta_n_deg: 32.0,
                    theta_e_deg: 6.0,
                    length_fraction: -1.0
                },
                34.3
            ),
            256
        )
        .is_err());
        // theta_e = 0 is rejected because it UNBOUNDS the length: Q_z moves
        // with E_z at rate tan(theta_e), so at zero the N_z < Q_z < E_z guard
        // stops constraining the length at all and a 1e9 r_t "bell" validates.
        for lf in [0.76, 1.0, 1.5] {
            assert!(
                generate_contour(
                    &spec(
                        ContourKind::DirectBell {
                            theta_n_deg: 32.0,
                            theta_e_deg: 0.0,
                            length_fraction: lf
                        },
                        34.3
                    ),
                    256
                )
                .is_err(),
                "theta_e = 0 accepted at length_fraction {lf}"
            );
        }
        // …and an absurd length is rejected outright, so a near-zero exit
        // angle cannot make the ordering guard toothless either.
        assert!(generate_contour(
            &spec(
                ContourKind::DirectBell {
                    theta_n_deg: 32.0,
                    theta_e_deg: 0.001,
                    length_fraction: 1e9
                },
                34.3
            ),
            256
        )
        .is_err());
    }

    /// The rejection message is user-visible now (cfd-ui shows it when the
    /// generator falls back to the cone), so it must point the right way: Q_z
    /// behind the tangency means too LONG, Q_z past the exit means too SHORT.
    #[test]
    fn bell_rejection_message_says_which_way_to_move() {
        let msg = |eps: f64, tn: f64, te: f64, lf: f64| {
            generate_contour(
                &spec(
                    ContourKind::DirectBell {
                        theta_n_deg: tn,
                        theta_e_deg: te,
                        length_fraction: lf,
                    },
                    eps,
                ),
                256,
            )
            .unwrap_err()
            .to_string()
        };
        // Q_z past the exit: too short for this area ratio.
        let short = msg(34.3, 32.0, 6.0, 0.2);
        assert!(short.contains("too short"), "{short}");
        // Q_z behind the throat-arc tangency: too long. Reached at a small
        // area ratio, where the reference cone is short and the exit radius
        // is close to the throat.
        let long = msg(1.2, 20.0, 6.0, 1.5);
        assert!(long.contains("too long"), "{long}");
    }

    #[test]
    fn bell_monotone_in_z_and_r() {
        let p = generate_contour(
            &spec(ContourKind::ParabolicBell { bell_percent: 0.8 }, 25.0),
            512,
        )
        .unwrap();
        // z strict monotone is enforced by validate(); r must be monotone
        // non-decreasing downstream of the throat.
        for i in p.throat_index..p.points.len() - 1 {
            assert!(p.points[i + 1][1] >= p.points[i][1] - 1e-15);
        }
        // ...and non-increasing upstream of it.
        for i in 0..p.throat_index {
            assert!(p.points[i + 1][1] <= p.points[i][1] + 1e-15);
        }
    }

    #[test]
    fn invalid_specs_are_rejected() {
        assert!(generate_contour(
            &spec(
                ContourKind::Conical {
                    half_angle_deg: 15.0
                },
                0.9
            ),
            128
        )
        .is_err()); // area ratio <= 1
        assert!(generate_contour(
            &spec(ContourKind::ParabolicBell { bell_percent: 0.5 }, 25.0),
            128
        )
        .is_err()); // bell percent below table
        let mut s = spec(
            ContourKind::Conical {
                half_angle_deg: 15.0,
            },
            8.0,
        );
        s.throat_radius_m = -1.0;
        assert!(generate_contour(&s, 128).is_err());
        s = spec(
            ContourKind::Conical {
                half_angle_deg: 15.0,
            },
            8.0,
        );
        s.contraction_ratio = 1.05; // chamber inside the upstream arc tangency
        assert!(generate_contour(&s, 128).is_err());
    }
}
