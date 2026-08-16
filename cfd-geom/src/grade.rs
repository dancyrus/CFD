//! The spacing rule (grid-grading work order, item 2). The rule lives HERE,
//! in the geometry layer — `cfd-core` accepts a list of cell edges and must
//! not know why they are spaced that way.
//!
//! For each axis independently: find the span where solid exists in the
//! rasterized field, hold base resolution across it plus a margin, then grade
//! geometrically beyond. Works on arbitrary drawn geometry — nothing here
//! references nozzle exit planes or exit radii.
//!
//! Two deliberate refinements of the rule, both on the safe side:
//!
//! - Grading happens only on the HIGH side of each axis (downstream in z,
//!   outward in r). The low side is the inlet / the axis: the region between
//!   the axis and the bore solid is the jet core itself, and "beyond the
//!   solid span" must never coarsen it.
//! - Cell widths are capped at `max_ratio` times base. Pure geometric growth
//!   never stops; a cap keeps the far plume readable (a shock cell several
//!   throat radii long still spans many cells) at a tiny cell-count cost.
//!
//! Growth ratio 1.05 is a margin, not a correctness requirement — the
//! weighted non-uniform stencil in `cfd-core` is second order at any ratio
//! (the acceptance guards run at 1.15–1.2).

use cfd_contract::{CfdError, Grid, Result, SolidField};

/// How to grade a domain around a rasterized solid field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GradeSpec {
    /// Base (finest) spacing per axis — held across the solid span.
    pub base_dz: f64,
    pub base_dr: f64,
    /// Target domain extents.
    pub lz: f64,
    pub lr: f64,
    /// Geometric growth ratio beyond the held span. 1.0 = uniform.
    pub growth: f64,
    /// Base cells held beyond the solid span before grading starts.
    pub margin_cells: f64,
    /// Cap on any cell width, as a multiple of base.
    pub max_ratio: f64,
}

impl GradeSpec {
    pub fn new(base_dz: f64, base_dr: f64, lz: f64, lr: f64) -> GradeSpec {
        GradeSpec { base_dz, base_dr, lz, lr, growth: 1.05, margin_cells: 8.0, max_ratio: 6.0 }
    }

    fn validate(&self) -> Result<()> {
        let ok = self.base_dz > 0.0
            && self.base_dr > 0.0
            && self.lz >= self.base_dz
            && self.lr >= self.base_dr
            && self.growth >= 1.0
            && self.margin_cells >= 0.0
            && self.max_ratio >= 1.0
            && [self.base_dz, self.base_dr, self.lz, self.lr, self.growth, self.max_ratio]
                .iter()
                .all(|v| v.is_finite());
        if ok {
            Ok(())
        } else {
            Err(CfdError::Parameter(format!("invalid GradeSpec: {self:?}")))
        }
    }
}

/// One axis: hold `base` from 0 through `hold_end`, then grow geometrically
/// (capped) until `len`. Held cells are EXACTLY `base` wide, so a `base_dr`
/// quantized to put the throat on a cell face keeps that property; the graded
/// block is rescaled as a whole to land exactly on `len`, which preserves the
/// width ratios.
fn graded_axis(hold_end: f64, base: f64, len: f64, growth: f64, max_ratio: f64) -> Vec<f64> {
    let hold_end = hold_end.clamp(0.0, len);
    let n_base = (hold_end / base).ceil().max(1.0) as usize;
    let held = n_base as f64 * base;
    // Too little room left to grade (or uniform requested): uniform axis of
    // near-base cells covering `len` exactly.
    let rest = len - held;
    if rest < base * growth || growth <= 1.0 {
        let n = (len / base).round().max(1.0) as usize;
        let w = len / n as f64;
        return (0..=n).map(|i| i as f64 * w).collect();
    }
    let mut edges: Vec<f64> = (0..=n_base).map(|i| i as f64 * base).collect();
    let mut widths: Vec<f64> = Vec::new();
    let cap = base * max_ratio;
    let mut w = base * growth;
    let mut acc = 0.0;
    while acc + w <= rest {
        widths.push(w);
        acc += w;
        w = (w * growth).min(cap);
    }
    // Stretch the block that FITS onto `rest` (ratios kept). Scaling UP, never
    // down: a shrink could push graded cells below `base`, silently tightening
    // the timestep — the held base spacing must stay the finest on the axis.
    let scale = rest / acc; // >= 1 by construction of the loop
    let mut e = held;
    for w in widths {
        e += w * scale;
        edges.push(e);
    }
    *edges.last_mut().unwrap() = len; // exact endpoint
    edges
}

/// The spacing rule. Finds the per-axis span covered by ANY solid fraction in
/// `solid` (its own grid supplies the physical coordinates), holds base
/// resolution from the axis origin through that span plus the margin, and
/// grades geometrically beyond, out to the `spec` extents.
pub fn grade_from_solid(solid: &SolidField, spec: &GradeSpec) -> Result<Grid> {
    spec.validate()?;
    let g = &solid.grid;
    let (mut z_hi, mut r_hi) = (0.0f64, 0.0f64);
    for ir in 0..g.nr {
        for iz in 0..g.nz {
            if solid.fraction[g.idx(iz, ir)] > 0.0 {
                z_hi = z_hi.max(g.z_edges()[iz + 1]);
                r_hi = r_hi.max(g.r_edges()[ir + 1]);
            }
        }
    }
    let z_edges = graded_axis(
        z_hi + spec.margin_cells * spec.base_dz,
        spec.base_dz,
        spec.lz,
        spec.growth,
        spec.max_ratio,
    );
    let r_edges = graded_axis(
        r_hi + spec.margin_cells * spec.base_dr,
        spec.base_dr,
        spec.lr,
        spec.growth,
        spec.max_ratio,
    );
    Grid::from_edges(z_edges, r_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn widths(edges: &[f64]) -> Vec<f64> {
        edges.windows(2).map(|w| w[1] - w[0]).collect()
    }

    #[test]
    fn axis_holds_base_then_grows_capped() {
        let e = graded_axis(2.0, 0.1, 30.0, 1.05, 4.0);
        let w = widths(&e);
        assert_eq!(e[0], 0.0);
        assert_eq!(*e.last().unwrap(), 30.0);
        // Held region: base width (to f64 edge roundoff).
        for k in 0..20 {
            assert!((w[k] - 0.1).abs() < 1e-12, "held cell {k}: {}", w[k]);
        }
        // Graded region: monotone growth, ratio <= growth, width <= cap.
        for k in 21..w.len() {
            let ratio = w[k] / w[k - 1];
            assert!(ratio <= 1.05 + 1e-9, "ratio {ratio} at {k}");
            assert!(ratio >= 1.0 - 1e-9, "shrinking at {k}");
            // The closing stretch may exceed the cap by its scale factor
            // (a few percent); the cap is a readability bound, not exact.
            assert!(w[k] <= 0.4 * 1.1, "cell {k} above cap: {}", w[k]);
        }
        // Massively fewer cells than uniform.
        assert!(w.len() < 200, "graded axis has {} cells vs 300 uniform", w.len());
        // Edges strictly increasing (Grid::from_edges would reject otherwise).
        for k in 1..e.len() {
            assert!(e[k] > e[k - 1]);
        }
    }

    #[test]
    fn axis_uniform_when_hold_covers_domain() {
        let e = graded_axis(9.99, 0.1, 10.0, 1.05, 4.0);
        let w = widths(&e);
        assert_eq!(w.len(), 100);
        for x in &w {
            assert!((x - 0.1).abs() < 1e-12);
        }
        // growth = 1.0 is uniform regardless of hold.
        let e = graded_axis(1.0, 0.1, 10.0, 1.0, 4.0);
        assert_eq!(widths(&e).len(), 100);
    }

    #[test]
    fn grade_from_solid_holds_base_across_the_solid_span() {
        // Solid block on a uniform base grid: z in [1.0, 3.0], r in [0.5, 2.0].
        let base = Grid::uniform(100, 40, 0.1, 0.05);
        let mut s = SolidField::empty(base.clone());
        for ir in 10..40 {
            for iz in 10..30 {
                s.fraction[base.idx(iz, ir)] = 1.0;
            }
        }
        let spec = GradeSpec::new(0.1, 0.05, 40.0, 8.0);
        let g = grade_from_solid(&s, &spec).unwrap();
        assert!((g.lz() - 40.0).abs() < 1e-9);
        assert!((g.lr() - 8.0).abs() < 1e-9);
        // Base resolution held from 0 through the span + 8-cell margin.
        let z_hold = 3.0 + 0.8;
        let r_hold = 2.0 + 0.4;
        for iz in 0..g.nz {
            if g.z_edges()[iz + 1] <= z_hold + 1e-9 {
                assert!((g.dz(iz) as f64 - 0.1).abs() < 1e-6, "dz({iz}) = {}", g.dz(iz));
            }
        }
        for ir in 0..g.nr {
            if g.r_edges()[ir + 1] <= r_hold + 1e-9 {
                assert!((g.dr(ir) as f64 - 0.05).abs() < 1e-6, "dr({ir}) = {}", g.dr(ir));
            }
        }
        // The graded tail exists and saves cells: uniform would be 400 x 160.
        assert!(g.nz < 250, "nz = {}", g.nz);
        assert!(g.nr < 110, "nr = {}", g.nr);
        // No solid at all still produces a valid graded grid.
        let empty = SolidField::empty(base);
        let g = grade_from_solid(&empty, &spec).unwrap();
        assert!((g.lz() - 40.0).abs() < 1e-9);
    }

    /// The same rule on a real rasterized BELL, not a synthetic block or the
    /// legacy cone: the bell is the wall every engine preset actually
    /// produces, it is shorter than the cone, and its lip sits at a different
    /// station — which is precisely what the held span is read from. Both bell
    /// paths (Rao table and measured angles) are exercised.
    #[test]
    fn grade_from_solid_holds_base_across_a_rasterized_bell() {
        use crate::{generate_contour, rasterize, ContourKind, NozzleSpec};
        use cfd_contract::{GasModel, RefScales};

        let gas = GasModel { gamma: 1.2, r_specific_si: 616.0 };
        let refs = RefScales::from_chamber(0.138, 2.06e7, 3600.0, &gas);
        let bells = [
            (
                "80% table bell, eps 25",
                25.0,
                0.382,
                ContourKind::ParabolicBell { bell_percent: 0.8 },
            ),
            (
                "measured bell, eps 34.3 (theta_n 32 / theta_e 6)",
                34.3,
                0.300,
                ContourKind::DirectBell {
                    theta_n_deg: 32.0,
                    theta_e_deg: 6.0,
                    length_fraction: 0.76,
                },
            ),
        ];
        for (name, area_ratio, arc, contour) in bells {
            let spec_n = NozzleSpec {
                throat_radius_m: 0.138,
                area_ratio,
                contraction_ratio: 4.0,
                converge_half_angle_deg: 30.0,
                throat_arc_up: 1.5,
                throat_arc_down: arc,
                contour,
            };
            let profile = generate_contour(&spec_n, 512).unwrap();
            // Base grid big enough to hold the whole bell, then grade out.
            let (dz, dr) = (0.145, 0.05);
            let base = Grid::uniform(260, 260, dz as f32, dr as f32);
            let solid = rasterize(&profile, &base, &refs).unwrap();
            assert!(solid.fraction.iter().any(|&f| f > 0.0), "{name}: nothing rasterized");
            let spec = GradeSpec::new(dz, dr, 60.0, 26.0);
            let g = grade_from_solid(&solid, &spec).unwrap();
            assert!((g.lz() - 60.0).abs() < 1e-9 && (g.lr() - 26.0).abs() < 1e-9, "{name}");
            // Base resolution held across the bell itself (its z-extent and
            // its exit radius, both in r_t = non-dimensional units).
            let z_end = profile.points.last().unwrap()[0] / refs.l_m;
            let r_end = area_ratio.sqrt();
            for iz in 0..g.nz {
                if g.z_edges()[iz + 1] <= z_end {
                    assert!((g.dz(iz) as f64 - dz).abs() < 1e-6, "{name}: dz({iz}) {}", g.dz(iz));
                }
            }
            for ir in 0..g.nr {
                if g.r_edges()[ir + 1] <= r_end {
                    assert!((g.dr(ir) as f64 - dr).abs() < 1e-6, "{name}: dr({ir}) {}", g.dr(ir));
                }
            }
            // The graded tail saves cells against the uniform equivalent.
            let uniform = (60.0 / dz).round() as usize * (26.0 / dr).round() as usize;
            assert!(g.len() < uniform, "{name}: graded {} vs uniform {uniform}", g.len());
        }
    }
}
