//! Golden test for the curve split: wrapping `NozzleSpec` in `NozzleCurve` must
//! not move a single bit of the generated polyline.
//!
//! `legacy_generate_contour` below is the pre-curve `generate_contour`, copied
//! verbatim and frozen. It is deliberately a duplicate rather than a call into
//! the crate: a golden test that shares code with the thing it guards guards
//! nothing. If a future change to the curve is MEANT to move the polyline, this
//! test is the one that has to be updated deliberately, and updating it is the
//! signal that `cfd-core`'s recorded ladder results need re-running (CLAUDE.md's
//! contract-change rule).
//!
//! **What it does and does not freeze.** The copy is arithmetically verbatim,
//! but it still calls three live functions: `rao_angles`, `NozzleSpec::validate`
//! and `WallProfile::validate`. The first would otherwise move BOTH sides of the
//! comparison together — a change to the digitized table would slide the five
//! `ParabolicBell` specs' walls and this test would not notice — so
//! `rao_table_anchors_are_frozen` pins the exact angles those specs resolve to.
//! The two validators are gates, not arithmetic: they can only turn a pass into
//! an error, never into a different polyline.
//!
//! **What it guards, after the adaptive-tessellation change.** `generate_contour`
//! is the FIXED-sample entry point, and it is now called only by `cfd-core`'s
//! ladder and diagnostics — the app tessellates adaptively. That is exactly why
//! this test matters: it is the reason the committed ladder results stay
//! comparable across a change that rewrote the geometry layer underneath them.
//! The app's own wall is covered separately, by `cfd-ui`'s
//! `halving_the_chord_tolerance_does_not_move_the_throat_area`.

use cfd_geom::{generate_contour, ContourKind, NozzleCurve, NozzleSpec, WallProfile};

// ---------------------------------------------------------------------------
// The frozen pre-curve implementation. DO NOT refactor to share code.
// ---------------------------------------------------------------------------

const CHAMBER_LEN_RT: f64 = 2.0;
const L_C15_REF_ARC_RT: f64 = 1.5;

fn push(pts: &mut Vec<[f64; 2]>, p: [f64; 2]) {
    if pts.last().is_none_or(|q| p[0] - q[0] > 1e-12) {
        pts.push(p);
    }
}

fn l_c15(area_ratio: f64) -> f64 {
    let s15 = 15.0f64.to_radians();
    ((area_ratio.sqrt() - 1.0) + L_C15_REF_ARC_RT * (1.0 / s15.cos() - 1.0)) / s15.tan()
}

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
) -> Result<(), String> {
    let (tn, te) = (tn_deg.to_radians(), te_deg.to_radians());
    if tn <= te {
        return Err("theta_n must exceed theta_e".into());
    }
    let nz = r2 * tn.sin();
    let nr = 1.0 + r2 * (1.0 - tn.cos());
    let (ez, er) = (l_n, re);
    let (m1, m2) = (tn.tan(), te.tan());
    let c1 = nr - m1 * nz;
    let c2 = er - m2 * ez;
    let qz = (c2 - c1) / (m1 - m2);
    let qr = (m1 * c2 - m2 * c1) / (m1 - m2);
    if !(nz < qz && qz < ez) {
        return Err("bell control point out of order".into());
    }
    for k in 1..=n_arc_dn {
        let phi = tn * k as f64 / n_arc_dn as f64;
        push(pts, [z_t + r2 * phi.sin(), 1.0 + r2 * (1.0 - phi.cos())]);
    }
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

fn legacy_generate_contour(spec: &NozzleSpec, samples: usize) -> Result<WallProfile, String> {
    spec.validate().map_err(|e| e.to_string())?;

    let rt = spec.throat_radius_m;
    let eps = spec.area_ratio;
    let re = eps.sqrt();
    let rc = spec.contraction_ratio.sqrt();
    let beta = spec.converge_half_angle_deg.to_radians();
    let r1 = spec.throat_arc_up;
    let r2 = spec.throat_arc_down;

    let ra_up = 1.0 + r1 * (1.0 - beta.cos());
    let z_t = CHAMBER_LEN_RT + (rc - ra_up) / beta.tan() + r1 * beta.sin();

    let n = samples.max(64);
    let n_arc_up = (n / 8).max(8);
    let n_arc_dn = (n / 8).max(8);
    let n_div = n.saturating_sub(n_arc_up + n_arc_dn + 4).max(16);

    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(n + 8);

    push(&mut pts, [0.0, rc]);
    push(&mut pts, [CHAMBER_LEN_RT, rc]);
    push(&mut pts, [z_t - r1 * beta.sin(), ra_up]);
    for k in 1..=n_arc_up {
        let phi = beta * (1.0 - k as f64 / n_arc_up as f64);
        push(
            &mut pts,
            [z_t - r1 * phi.sin(), 1.0 + r1 * (1.0 - phi.cos())],
        );
    }
    let throat_index = pts.len() - 1;

    match spec.contour {
        ContourKind::Conical { half_angle_deg } => {
            let alpha = half_angle_deg.to_radians();
            let l_n = ((re - 1.0) + r2 * (1.0 / alpha.cos() - 1.0)) / alpha.tan();
            for k in 1..=n_arc_dn {
                let phi = alpha * k as f64 / n_arc_dn as f64;
                push(
                    &mut pts,
                    [z_t + r2 * phi.sin(), 1.0 + r2 * (1.0 - phi.cos())],
                );
            }
            push(&mut pts, [z_t + l_n, re]);
        }
        ContourKind::ParabolicBell { bell_percent } => {
            let (tn_deg, te_deg) = cfd_geom::rao_angles(eps, bell_percent);
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

    for p in &mut pts {
        p[0] *= rt;
        p[1] *= rt;
    }
    let profile = WallProfile {
        points: pts,
        throat_index,
    };
    profile.validate().map_err(|e| e.to_string())?;
    Ok(profile)
}

// ---------------------------------------------------------------------------
// The specs under test.
// ---------------------------------------------------------------------------

fn spec(
    throat_radius_m: f64,
    area_ratio: f64,
    throat_arc_down: f64,
    contour: ContourKind,
) -> NozzleSpec {
    NozzleSpec {
        throat_radius_m,
        area_ratio,
        contraction_ratio: 4.0,
        converge_half_angle_deg: 30.0,
        throat_arc_up: 1.5,
        throat_arc_down,
        contour,
    }
}

/// The demo case plus the six original engine presets, as `(name, spec)` —
/// between them every contour SHAPE the app can build: a cone, a Rao-table
/// bell, and a measured-angle bell.
///
/// The preset rows MIRROR `cfd_ui::case::PRESETS` — `cfd-geom` cannot depend on
/// `cfd-ui`, so the mirror is guarded from the other side by `cfd-ui`'s
/// `preset_specs_match_the_geom_golden_table`, which fails if one of these six
/// moves away from these numbers, AND fails if any newer preset (the eight
/// historical engines, and whatever comes after) introduces a shape this table
/// does not cover.
pub fn golden_specs() -> Vec<(&'static str, NozzleSpec)> {
    vec![
        (
            "demo (15 deg cone)",
            spec(
                0.05,
                8.0,
                0.382,
                ContourKind::Conical {
                    half_angle_deg: 15.0,
                },
            ),
        ),
        (
            "Merlin 1D",
            spec(
                0.131,
                16.0,
                0.382,
                ContourKind::ParabolicBell { bell_percent: 0.78 },
            ),
        ),
        (
            "F-1",
            spec(
                0.465,
                16.0,
                0.382,
                ContourKind::ParabolicBell { bell_percent: 0.75 },
            ),
        ),
        (
            "Raptor 2",
            spec(
                0.115,
                34.3,
                0.300,
                ContourKind::DirectBell {
                    theta_n_deg: 32.0,
                    theta_e_deg: 6.0,
                    length_fraction: 0.76,
                },
            ),
        ),
        (
            "AJ10-190",
            spec(
                0.073,
                55.0,
                0.382,
                ContourKind::ParabolicBell { bell_percent: 0.78 },
            ),
        ),
        (
            "RS-25",
            spec(
                0.138,
                69.0,
                0.382,
                ContourKind::ParabolicBell { bell_percent: 0.80 },
            ),
        ),
        (
            "Merlin Vac",
            spec(
                0.128,
                165.0,
                0.382,
                ContourKind::ParabolicBell { bell_percent: 0.75 },
            ),
        ),
    ]
}

/// THE STEP 1 GATE: the curve wrapper must reproduce the legacy polyline bit
/// for bit — `==` on f64, not a tolerance — for the demo spec and all six
/// preset specs, through both entry points.
#[test]
fn curve_tessellation_is_bit_identical_to_the_legacy_generator() {
    for (name, s) in golden_specs() {
        // 512 and 1024 are what cfd-core/tests/diag.rs asks for; 256 was the
        // app's old constant; 64 and 4096 are the clamp and a stress point.
        for samples in [64, 128, 256, 512, 1024, 4096] {
            let want = legacy_generate_contour(&s, samples)
                .unwrap_or_else(|e| panic!("{name}: legacy generator rejected the spec: {e}"));
            let got = generate_contour(&s, samples).unwrap();

            assert_eq!(
                want.throat_index, got.throat_index,
                "{name} @ {samples}: throat_index moved"
            );
            assert_eq!(
                want.points.len(),
                got.points.len(),
                "{name} @ {samples}: point count moved ({} -> {})",
                want.points.len(),
                got.points.len()
            );
            for (i, (a, b)) in want.points.iter().zip(&got.points).enumerate() {
                assert_eq!(
                    a.map(f64::to_bits),
                    b.map(f64::to_bits),
                    "{name} @ {samples}: point {i} differs: {a:?} vs {b:?}"
                );
            }
        }
    }
}

/// The wrapper's own default has to BE the §10 family's chamber straight, or
/// the bit-identity above is vacuous (it would just prove two copies of the
/// same wrong constant agree).
#[test]
fn chamber_len_default_is_the_family_value() {
    assert_eq!(cfd_geom::CHAMBER_LEN_RT, 2.0);
    let s = golden_specs()[0].1;
    let c = NozzleCurve::new(s);
    assert_eq!(c.chamber_len_rt, 2.0);
    // And it is load-bearing: a different chamber straight moves the throat.
    let mut long = c;
    long.chamber_len_rt = 3.0;
    let (a, b) = (
        c.tessellate_fixed(256).unwrap(),
        long.tessellate_fixed(256).unwrap(),
    );
    let dz = (b.points[b.throat_index][0] - a.points[a.throat_index][0]) / s.throat_radius_m;
    assert!((dz - 1.0).abs() < 1e-12, "throat moved by {dz} r_t, want 1.0");
}

/// Close the one real hole in the freeze: five of the seven golden specs are
/// `ParabolicBell`, whose wall angles the frozen copy and the live code BOTH
/// take from `rao_angles`. A change to the digitized table would slide want and
/// got together and the bit-identity assert would sail through.
///
/// So pin the angles themselves. These literals are the digitized table's
/// values at each preset's (ε, bell percent) as of the curve split; if the table
/// moves, this fails and the golden walls above are known to have moved with it.
#[test]
fn rao_table_anchors_are_frozen() {
    // (eps, bell_percent, theta_n_deg, theta_e_deg)
    const ANCHORS: [(f64, f64, f64, f64); 5] = [
        (16.0, 0.78, 28.599_083_358_037_227, 10.177_759_785_030_357),
        (16.0, 0.75, 29.504_938_750_920_676, 10.978_615_177_913_806),
        (55.0, 0.78, 32.381_882_223_687_37, 7.963_372_780_037_542),
        (69.0, 0.80, 32.429_336_534_006_886, 7.267_665_866_498_278),
        (165.0, 0.75, 35.125, 8.05),
    ];

    for (eps, bp, tn, te) in ANCHORS {
        let (gn, ge) = cfd_geom::rao_angles(eps, bp);
        assert!(
            (gn - tn).abs() < 1e-12 && (ge - te).abs() < 1e-12,
            "rao_angles({eps}, {bp}) = ({gn}, {ge}), frozen at ({tn}, {te}) — \
             the golden walls above moved with it"
        );
    }
}

/// `NozzleCurve::throat_z_rt` and the frozen `tessellate_fixed` compute the
/// throat station from the same formula written out twice. Only the
/// `tessellate_fixed` copy is bit-frozen by the test above, so editing
/// `throat_z_rt` could silently desync `pieces()`/`radius_at` from the polyline
/// with nothing failing. This is the assert that catches that.
#[test]
fn throat_station_agrees_between_the_curve_and_the_frozen_tessellation() {
    for (name, s) in golden_specs() {
        let c = NozzleCurve::new(s);
        let p = c.tessellate_fixed(256).unwrap();
        let from_polyline = p.points[p.throat_index][0] / s.throat_radius_m;
        assert_eq!(
            from_polyline.to_bits(),
            c.throat_z_rt().to_bits(),
            "{name}: throat_z_rt() = {} but the frozen tessellation puts it at {from_polyline}",
            c.throat_z_rt()
        );
        // …and the throat really is r = r_t there, on both representations.
        assert!((p.points[p.throat_index][1] / s.throat_radius_m - 1.0).abs() < 1e-15);
        assert!(
            (c.radius_at(c.throat_z_rt() * s.throat_radius_m).unwrap() / s.throat_radius_m - 1.0)
                .abs()
                < 1e-15
        );
    }
}

/// `BellGeometry` is public API and the editor's whole reason for existing (Q is
/// the point it refuses to let anyone drag), so assert what it promises: the
/// §10 ordering, and G1 tangency — the wall leaves N at exactly theta_n and
/// reaches E at exactly theta_e, which is what makes theta_n physically
/// meaningful and what dragging Q would destroy.
#[test]
fn bell_geometry_is_ordered_and_tangent_to_both_angles() {
    for (name, s) in golden_specs() {
        let c = NozzleCurve::new(s);
        let Ok(b) = c.bell() else {
            continue; // the cone has no Bezier, by design
        };
        assert!(
            b.n[0] < b.q[0] && b.q[0] < b.e[0],
            "{name}: N_z {} < Q_z {} < E_z {} violated",
            b.n[0],
            b.q[0],
            b.e[0]
        );
        // Q is DERIVED: it is the intersection of the two tangent lines, so the
        // chord N->Q has slope tan(theta_n) and Q->E has slope tan(theta_e).
        let slope = |a: [f64; 2], z: [f64; 2]| (z[1] - a[1]) / (z[0] - a[0]);
        assert!(
            (slope(b.n, b.q) - b.theta_n_deg.to_radians().tan()).abs() < 1e-12,
            "{name}: N->Q leg is not tangent at theta_n"
        );
        assert!(
            (slope(b.q, b.e) - b.theta_e_deg.to_radians().tan()).abs() < 1e-12,
            "{name}: Q->E leg is not tangent at theta_e"
        );
        // The exit lip IS the area ratio.
        assert!((b.e[1] - s.area_ratio.sqrt()).abs() < 1e-12, "{name}: exit radius");
    }
}
