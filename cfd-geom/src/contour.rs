//! Parametric nozzle contours: conical and Rao parabolic bell.
//!
//! All construction happens in units of r_t (throat radius = 1) and is scaled to
//! metres at the end. z = 0 is the chamber head; the throat sits at
//! z_t = chamber + converging cone + upstream arc, and is always an exact sample
//! point of the returned polyline.

use crate::rao::rao_angles;
use crate::{ContourKind, NozzleSpec, WallProfile};
use cfd_contract::{CfdError, Result};

/// Chamber straight length, in units of r_t (session brief §1).
const CHAMBER_LEN_RT: f64 = 2.0;

/// Downstream throat-arc radius of Huzel–Huang's REFERENCE 15° cone, in r_t.
///
/// This is a fixed property of the definition of "N% bell" — H&H measure bell
/// length against a 15° conical nozzle of the same throat area and area ratio
/// **with a 1.5 R_t downstream throat arc** — not a property of the nozzle
/// being generated. It happens to equal the family's `throat_arc_up`, but it
/// is deliberately NOT read from the spec: an engine with a tighter throat arc
/// (Raptor 2, 0.300 R_t) must still be measured against the same reference, or
/// "80% bell" means a different length for every spec.
const L_C15_REF_ARC_RT: f64 = 1.5;

/// Append a point, dropping an exact-duplicate z (shared piece endpoints get
/// pushed twice).
fn push(pts: &mut Vec<[f64; 2]>, p: [f64; 2]) {
    if pts.last().is_none_or(|q| p[0] - q[0] > 1e-12) {
        pts.push(p);
    }
}

/// Huzel–Huang reference length: the 15° cone this family measures bell
/// percent against, in r_t, measured from the throat.
fn l_c15(area_ratio: f64) -> f64 {
    let s15 = 15.0f64.to_radians();
    ((area_ratio.sqrt() - 1.0) + L_C15_REF_ARC_RT * (1.0 / s15.cos() - 1.0)) / s15.tan()
}

/// The diverging bell: downstream throat arc to wall angle `tn_deg`, then a
/// quadratic Bézier to the exit lip at axial distance `l_n` from the throat
/// meeting it at `te_deg`. Shared by the Rao-table and direct-angle paths —
/// they differ only in where the two angles and the length come from.
#[allow(clippy::too_many_arguments)]
fn push_bell(
    pts: &mut Vec<[f64; 2]>,
    z_t: f64,
    r2: f64,
    re: f64,
    l_n: f64,
    tn_deg: f64,
    te_deg: f64,
    n_arc_dn: usize,
    n_div: usize,
) -> Result<()> {
    let (tn, te) = (tn_deg.to_radians(), te_deg.to_radians());
    if tn <= te {
        // Cannot happen with the shipped table; guards division by zero in the
        // Bézier control point below.
        return Err(CfdError::Geometry(format!(
            "theta_n ({tn_deg}) must exceed theta_e ({te_deg})"
        )));
    }
    // Bézier endpoints (z relative to throat) and control point.
    let nz = r2 * tn.sin();
    let nr = 1.0 + r2 * (1.0 - tn.cos());
    let (ez, er) = (l_n, re);
    let (m1, m2) = (tn.tan(), te.tan());
    let c1 = nr - m1 * nz;
    let c2 = er - m2 * ez;
    let qz = (c2 - c1) / (m1 - m2);
    let qr = (m1 * c2 - m2 * c1) / (m1 - m2);
    if !(nz < qz && qz < ez) {
        return Err(CfdError::Geometry(format!(
            "bell control point out of order (N_z={nz:.4}, Q_z={qz:.4}, E_z={ez:.4}): \
             length {l_n:.4} r_t is too short for area ratio {:.4} at theta_n {tn_deg}",
            re * re
        )));
    }
    // Downstream arc, wall angle 0 -> theta_n; ends exactly at N.
    for k in 1..=n_arc_dn {
        let phi = tn * k as f64 / n_arc_dn as f64;
        push(pts, [z_t + r2 * phi.sin(), 1.0 + r2 * (1.0 - phi.cos())]);
    }
    // Quadratic Bézier N -> E. t is NOT proportional to z; the polyline just
    // needs monotone z, which N_z < Q_z < E_z guarantees.
    for k in 1..=n_div {
        let t = k as f64 / n_div as f64;
        let a = (1.0 - t) * (1.0 - t);
        let b = 2.0 * t * (1.0 - t);
        let c = t * t;
        push(
            pts,
            [z_t + a * nz + b * qz + c * ez, a * nr + b * qr + c * er],
        );
    }
    Ok(())
}

/// Generate the full wall contour as a polyline with roughly `samples` points
/// (clamped to at least 64). The throat is exactly a vertex; `throat_index`
/// points at it.
pub fn generate_contour(spec: &NozzleSpec, samples: usize) -> Result<WallProfile> {
    spec.validate()?;

    let rt = spec.throat_radius_m;
    let eps = spec.area_ratio;
    let re = eps.sqrt(); // exit radius, r_t units
    let rc = spec.contraction_ratio.sqrt(); // chamber radius, r_t units
    let beta = spec.converge_half_angle_deg.to_radians();
    let r1 = spec.throat_arc_up;
    let r2 = spec.throat_arc_down;

    // Upstream arc tangency (converging side), parameterized by wall angle phi:
    //   z(phi) = z_t - r1*sin(phi),  r(phi) = 1 + r1*(1 - cos(phi)),  phi in [0, beta]
    let ra_up = 1.0 + r1 * (1.0 - beta.cos());
    let z_t = CHAMBER_LEN_RT + (rc - ra_up) / beta.tan() + r1 * beta.sin();

    let n = samples.max(64);
    let n_arc_up = (n / 8).max(8);
    let n_arc_dn = (n / 8).max(8);
    let n_div = n.saturating_sub(n_arc_up + n_arc_dn + 4).max(16);

    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(n + 8);

    // 1. Chamber straight.
    push(&mut pts, [0.0, rc]);
    push(&mut pts, [CHAMBER_LEN_RT, rc]);
    // 2. Converging cone (a straight segment; two endpoints are exact).
    push(&mut pts, [z_t - r1 * beta.sin(), ra_up]);
    // 3. Upstream throat arc, wall angle beta -> 0. Ends exactly at (z_t, 1).
    for k in 1..=n_arc_up {
        let phi = beta * (1.0 - k as f64 / n_arc_up as f64);
        push(
            &mut pts,
            [z_t - r1 * phi.sin(), 1.0 + r1 * (1.0 - phi.cos())],
        );
    }
    let throat_index = pts.len() - 1;

    // 4. Diverging section.
    match spec.contour {
        ContourKind::Conical { half_angle_deg } => {
            let alpha = half_angle_deg.to_radians();
            // Exact closed form for the cone length measured from the throat
            // (identity verified symbolically; residual 4.2e-15 over 20k draws).
            let l_n = ((re - 1.0) + r2 * (1.0 / alpha.cos() - 1.0)) / alpha.tan();
            // Downstream arc, wall angle 0 -> alpha; tangency is automatic.
            for k in 1..=n_arc_dn {
                let phi = alpha * k as f64 / n_arc_dn as f64;
                push(
                    &mut pts,
                    [z_t + r2 * phi.sin(), 1.0 + r2 * (1.0 - phi.cos())],
                );
            }
            // Straight cone to the exit lip.
            push(&mut pts, [z_t + l_n, re]);
        }
        // Huzel–Huang reference length, INCLUDING the throat-arc term, which
        // that reference cone takes at 1.5 R_t (`L_C15_REF_ARC_RT`) — NOT at
        // this nozzle's own downstream arc. The Aspirespace / bell_nozzle.py
        // form drops the term entirely and comes out short by an ε-dependent
        // amount (2.89% at ε = 8, 1.32% at ε = 25, 0.72% at ε = 69, 0.45% at
        // ε = 165), which is what makes an unqualified "80% bell" ambiguous.
        // This is the pinned definition (physics-reference §10).
        ContourKind::ParabolicBell { bell_percent } => {
            let (tn_deg, te_deg) = rao_angles(eps, bell_percent);
            push_bell(
                &mut pts,
                z_t,
                r2,
                re,
                bell_percent * l_c15(eps),
                tn_deg,
                te_deg,
                n_arc_dn,
                n_div,
            )?;
        }
        // Measured geometry: the wall angles are inputs, not table lookups.
        ContourKind::DirectBell {
            theta_n_deg,
            theta_e_deg,
            length_fraction,
        } => {
            push_bell(
                &mut pts,
                z_t,
                r2,
                re,
                length_fraction * l_c15(eps),
                theta_n_deg,
                theta_e_deg,
                n_arc_dn,
                n_div,
            )?;
        }
    }

    // Scale to metres.
    for p in &mut pts {
        p[0] *= rt;
        p[1] *= rt;
    }
    let profile = WallProfile {
        points: pts,
        throat_index,
    };
    profile.validate()?;
    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
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
