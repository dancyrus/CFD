//! T0 — rasterizer area, no flow, under a second (physics-reference §12).
//!
//! 1. A disk of radius R cells centred on a cell corner must recover pi*R^2.
//!    The error is SIGNED (point sampling measures +13.18% at R=3, +3.45% at
//!    R=8, -0.97% at R=12), so the assertion takes the absolute value.
//! 2. For the parametric nozzle at five slider settings, the throat area
//!    recovered from the solid field is within 0.5% of the analytic value.

use cfd_contract::{GasModel, Grid, RefScales};
use cfd_geom::{
    generate_contour, quantize_dr_to_throat, rasterize, rasterize_solid_polygon, ContourKind,
    NozzleSpec,
};
use std::f64::consts::PI;

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

    // Five slider settings across both contour kinds.
    let settings: [(ContourKind, f64); 5] = [
        (
            ContourKind::Conical {
                half_angle_deg: 15.0,
            },
            4.0,
        ),
        (
            ContourKind::Conical {
                half_angle_deg: 15.0,
            },
            8.0,
        ),
        (
            ContourKind::Conical {
                half_angle_deg: 18.0,
            },
            16.0,
        ),
        (ContourKind::ParabolicBell { bell_percent: 0.8 }, 8.0),
        (ContourKind::ParabolicBell { bell_percent: 0.8 }, 25.0),
    ];

    for (contour, area_ratio) in settings {
        let spec = NozzleSpec {
            throat_radius_m: 0.05,
            area_ratio,
            contraction_ratio: 4.0,
            converge_half_angle_deg: 30.0,
            throat_arc_up: 1.5,
            throat_arc_down: 0.382,
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
        assert!(
            rel <= 5e-3,
            "{contour:?} eps={area_ratio}: recovered A/A_t = {min_rw2:.5}, rel err {rel:.2e}"
        );
    }
}
