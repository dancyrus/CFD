//! T0 — rasterizer area, no flow, under a second (physics-reference §12).
//!
//! 1. A disk of radius R cells centred on a cell corner must recover pi*R^2.
//!    The error is SIGNED (point sampling measures +13.18% at R=3, +3.45% at
//!    R=8, -0.97% at R=12), so the assertion takes the absolute value.
//! 2. For the parametric nozzle at six slider settings — cone, Rao-table bell
//!    and measured-angle bell, the last on its own 0.300 r_t throat arc — the
//!    throat area recovered from the solid field is within 0.5% of the
//!    analytic value.

use cfd_contract::{GasModel, Grid, RefScales};
use cfd_geom::{
    generate_contour, quantize_dr_to_throat, rasterize, rasterize_solid_polygon, ContourKind,
    NozzleSpec,
};
use cfd_results::{record_test, TestResult, Value};
use std::f64::consts::PI;

/// Results get committed, not reported in chat (CLAUDE.md).
fn record(id: &str, name: &str, expected: impl Into<Value>, actual: impl Into<Value>,
          units: &str, pass: bool) {
    record_test("ladder", TestResult {
        id: id.into(), name: name.into(), expected: expected.into(),
        actual: actual.into(), units: units.into(), pass,
    });
}

/// Regular n-gon whose AREA is exactly pi*R^2 (circumradius slightly above R),
/// so the comparison isolates the rasterizer instead of polygonization error.
fn disk(center: [f64; 2], radius: f64, n: usize) -> Vec<[f64; 2]> {
    let step = 2.0 * PI / n as f64;
    let r_eff = radius * (step / step.sin()).sqrt();
    (0..n)
        .map(|k| {
            let a = step * k as f64;
            [center[0] + r_eff * a.cos(), center[1] + r_eff * a.sin()]
        })
        .collect()
}

#[test]
fn t0_disk_area_is_exact() {
    let g = Grid::uniform(72, 72, 1.0, 1.0);
    let mut worst = 0.0f64;
    for radius in [3.0, 4.0, 6.0, 8.0, 12.0, 20.0] {
        let poly = disk([32.0, 32.0], radius, 2048); // centred on a cell corner
        let s = rasterize_solid_polygon(&poly, &g).unwrap();
        // f64 accumulation, as every reduction must be.
        let mut area: f64 = 0.0;
        for f in &s.fraction {
            area += *f as f64;
        }
        let exact = PI * radius * radius;
        let rel = (area - exact).abs() / exact; // SIGNED error -> abs()
        worst = worst.max(rel);
        assert!(
            rel <= 5e-3,
            "R={radius}: area {area}, exact {exact}, rel {rel}"
        );
        // Exact sub-cell fractions should be limited only by f32 fraction
        // storage, orders of magnitude below the 0.5% gate.
        assert!(
            rel <= 1e-5,
            "R={radius}: rel {rel} — rasterizer is not exact"
        );
    }
    record("T0-disk", "rasterizer disk area, R in {3..20} cells", "<= 5e-3",
           worst, "worst relative area error", worst <= 5e-3);
}

/// Point sampling is the documented failure mode: keep a record of how wrong it
/// is so nobody "simplifies" the rasterizer back to it.
#[test]
fn t0_point_sampling_is_disqualified() {
    let radius = 3.0f64;
    let center = [32.0, 32.0];
    let mut area = 0.0;
    for iz in 0..72 {
        for ir in 0..72 {
            let d2 = (iz as f64 + 0.5 - center[0]).powi(2) + (ir as f64 + 0.5 - center[1]).powi(2);
            if d2 <= radius * radius {
                area += 1.0;
            }
        }
    }
    let rel = (area - PI * radius * radius) / (PI * radius * radius);
    assert!(rel > 0.08, "expected ~+13% error at R=3, got {rel}");
}

#[test]
fn t0_nozzle_throat_area_within_half_percent() {
    let mut worst = 0.0f64;
    // Demo-case reference scales: r_t = 50 mm, p0 = 5 MPa, T0 = 3200 K.
    let gas = GasModel {
        gamma: 1.24,
        r_specific_si: 378.0,
    };
    let refs = RefScales::from_chamber(0.05, 5e6, 3200.0, &gas);

    // Measure-like grid; dr quantized so the throat (r = 1) lies on a cell face.
    let dr = quantize_dr_to_throat(0.025);
    assert!((1.0 / dr as f64 - 40.0).abs() < 1e-4); // N_throat = 40 (f32 dr)
    let g = Grid::uniform(240, 220, 0.0724, dr);

    // Six slider settings across all three contour kinds — including the
    // measured-angle bell, which brings its own (tighter) throat arc: the
    // throat is what this test measures, so the arc must be part of the sweep
    // rather than pinned at the parametric family's 0.382.
    let settings: [(ContourKind, f64, f64); 6] = [
        (
            ContourKind::Conical {
                half_angle_deg: 15.0,
            },
            4.0,
            0.382,
        ),
        (
            ContourKind::Conical {
                half_angle_deg: 15.0,
            },
            8.0,
            0.382,
        ),
        (
            ContourKind::Conical {
                half_angle_deg: 18.0,
            },
            16.0,
            0.382,
        ),
        (ContourKind::ParabolicBell { bell_percent: 0.8 }, 8.0, 0.382),
        (ContourKind::ParabolicBell { bell_percent: 0.8 }, 25.0, 0.382),
        (
            // Raptor 2's published geometry (FAA AR 2019-001b Table 1).
            ContourKind::DirectBell {
                theta_n_deg: 32.0,
                theta_e_deg: 6.0,
                length_fraction: 0.76,
            },
            34.3,
            0.300,
        ),
    ];

    for (contour, area_ratio, throat_arc_down) in settings {
        let spec = NozzleSpec {
            throat_radius_m: 0.05,
            area_ratio,
            contraction_ratio: 4.0,
            converge_half_angle_deg: 30.0,
            throat_arc_up: 1.5,
            throat_arc_down,
            contour,
        };
        let profile = generate_contour(&spec, 512).unwrap();
        let solid = rasterize(&profile, &g, &refs).unwrap();

        // Open radius per column, r-weighted (physics-reference §5):
        //   r_w(i)^2 = 2*dr*sum_j (1 - frac[i][j]) * r_j
        // The throat is the minimum over columns fully covered by the profile.
        let z_end = profile.points.last().unwrap()[0] / refs.l_m;
        let mut min_rw2 = f64::MAX;
        for iz in 0..g.nz {
            if g.z_edges()[iz + 1] > z_end {
                break; // past the exit lip the column is open to the far field
            }
            let mut acc: f64 = 0.0; // f64 reduction, mandatory
            for ir in 0..g.nr {
                acc += (1.0 - solid.fraction[g.idx(iz, ir)] as f64)
                    * g.r_center(ir) as f64
                    * g.dr(ir) as f64;
            }
            min_rw2 = min_rw2.min(2.0 * acc);
        }
        // Analytic throat area is pi * r_t^2, i.e. r_w^2 = 1 non-dimensionally.
        let rel = (min_rw2 - 1.0).abs();
        worst = worst.max(rel);
        assert!(
            rel <= 5e-3,
            "{contour:?} eps={area_ratio}: recovered A/A_t = {min_rw2:.5}, rel err {rel:.2e}"
        );
    }
    record("T0-throat", "rasterized nozzle throat area, 6 slider settings", "<= 5e-3",
           worst, "worst relative area error", worst <= 5e-3);
}
