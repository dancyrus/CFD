//! Exact sub-cell solid area fractions.
//!
//! Point sampling ("is the cell centre inside?") gives +13.18% area error at a
//! radius of 3 cells, +3.45% at 8, -0.97% at 12 — it oscillates, it does not
//! converge monotonically, and the picture looks completely normal either way
//! (physics-reference §12, T0). Every geometry-dependent number downstream
//! inherits the choice, so this module computes the EXACT polygon/cell
//! intersection area instead: each cell column pre-clips the solid polygon to
//! its z-strip (Sutherland–Hodgman against two half-planes), then every cell in
//! the column clips the strip piece to its r-strip and takes the shoelace area.
//! Clipping a simple polygon against a convex window and applying the shoelace
//! formula is exact in real arithmetic; everything here runs in f64 and only the
//! final fraction is stored as f32.

use crate::WallProfile;
use cfd_contract::{CfdError, Grid, RefScales, Result, SolidField};

/// Clip `input` (closed polygon) against the half-plane `dist(p) >= 0`.
/// `dist` must be affine (signed distance up to scale); output lands in `out`.
fn clip_half(input: &[[f64; 2]], out: &mut Vec<[f64; 2]>, dist: impl Fn(&[f64; 2]) -> f64) {
    out.clear();
    let n = input.len();
    for i in 0..n {
        let a = input[i];
        let b = input[(i + 1) % n];
        let da = dist(&a);
        let db = dist(&b);
        if da >= 0.0 {
            out.push(a);
        }
        if (da > 0.0 && db < 0.0) || (da < 0.0 && db > 0.0) {
            let t = da / (da - db);
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
}

/// Unsigned shoelace area of a closed polygon.
fn polygon_area(poly: &[[f64; 2]]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        s += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * s.abs()
}

/// Rasterize a closed simple polygon (non-dimensional grid units) into exact
/// per-cell solid area fractions. This is the core engine behind [`rasterize`]
/// and is public so closed drawn shapes and the T0 disk test can use it
/// directly. Vertex order does not matter.
pub fn rasterize_solid_polygon(poly: &[[f64; 2]], g: &Grid) -> Result<SolidField> {
    if poly.len() < 3 {
        return Err(CfdError::Geometry(format!(
            "polygon needs at least 3 vertices, has {}",
            poly.len()
        )));
    }
    for (i, p) in poly.iter().enumerate() {
        if !(p[0].is_finite() && p[1].is_finite()) {
            return Err(CfdError::Geometry(format!(
                "polygon vertex {i} not finite: {p:?}"
            )));
        }
    }

    let z_edges = g.z_edges();
    let r_edges = g.r_edges();
    let mut fraction = vec![0.0f32; g.len()];

    // Grid-overlapping part of the polygon bounding box. Edge lists are
    // strictly increasing, so cells come from a binary search — the grid may
    // be graded and index arithmetic on a single spacing would be wrong.
    let (mut zmin, mut zmax, mut rmin, mut rmax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in poly {
        zmin = zmin.min(p[0]);
        zmax = zmax.max(p[0]);
        rmin = rmin.min(p[1]);
        rmax = rmax.max(p[1]);
    }
    if zmax <= z_edges[0] || zmin >= z_edges[g.nz] || rmax <= r_edges[0] || rmin >= r_edges[g.nr] {
        return Ok(SolidField { grid: g.clone(), fraction });
    }
    // First cell whose upper edge exceeds the bound; last cell whose lower
    // edge is below it.
    let iz0 = z_edges[1..=g.nz].partition_point(|&e| e <= zmin);
    let iz1 = z_edges[..g.nz].partition_point(|&e| e < zmax);
    let ir0 = r_edges[1..=g.nr].partition_point(|&e| e <= rmin);
    let ir1 = r_edges[..g.nr].partition_point(|&e| e < rmax);

    let mut strip: Vec<[f64; 2]> = Vec::with_capacity(poly.len() + 4);
    let mut tmp: Vec<[f64; 2]> = Vec::with_capacity(poly.len() + 4);
    let mut cell: Vec<[f64; 2]> = Vec::new();

    for iz in iz0..iz1 {
        let za = z_edges[iz];
        let zb = z_edges[iz + 1];
        // Pre-clip the whole polygon to this column's z-strip.
        clip_half(poly, &mut tmp, |p| p[0] - za);
        clip_half(&tmp, &mut strip, |p| zb - p[0]);
        if strip.len() < 3 {
            continue;
        }
        // Rows actually touched by the strip piece.
        let (mut prmin, mut prmax) = (f64::MAX, f64::MIN);
        for p in &strip {
            prmin = prmin.min(p[1]);
            prmax = prmax.max(p[1]);
        }
        let jr0 = r_edges[1..=g.nr].partition_point(|&e| e <= prmin).clamp(ir0, ir1);
        let jr1 = r_edges[..g.nr].partition_point(|&e| e < prmax).clamp(jr0, ir1);
        for ir in jr0..jr1 {
            let ra = r_edges[ir];
            let rb = r_edges[ir + 1];
            clip_half(&strip, &mut tmp, |p| p[1] - ra);
            clip_half(&tmp, &mut cell, |p| rb - p[1]);
            let a = polygon_area(&cell);
            if a > 0.0 {
                let cell_area = (zb - za) * (rb - ra);
                fraction[g.idx(iz, ir)] = ((a / cell_area).clamp(0.0, 1.0)) as f32;
            }
        }
    }
    Ok(SolidField { grid: g.clone(), fraction })
}

/// Rasterize a wall profile (SI metres) into exact solid area fractions on the
/// non-dimensional grid. The solid region is everything at or above the wall
/// polyline, `r >= r_w(z)` for z inside the profile's extent; z outside it
/// (upstream of the first point, downstream of the exit lip) is fluid. The
/// profile's z = 0 maps to the grid's z = 0; `refs.l_m` (= r_t) converts metres
/// to grid units. Drawn and generated walls both come through here — the solver
/// cannot tell them apart.
pub fn rasterize(p: &WallProfile, g: &Grid, refs: &RefScales) -> Result<SolidField> {
    p.validate()?;
    if !(refs.l_m.is_finite() && refs.l_m > 0.0) {
        return Err(CfdError::Parameter(format!("RefScales.l_m = {}", refs.l_m)));
    }
    let inv = 1.0 / refs.l_m;
    let mut poly: Vec<[f64; 2]> = p.points.iter().map(|q| [q[0] * inv, q[1] * inv]).collect();
    // Close the solid region above the wall, past the top of the grid.
    let mut r_top = g.lr();
    for q in &poly {
        r_top = r_top.max(q[1]);
    }
    r_top += 1.0;
    let z_first = poly[0][0];
    let z_last = poly[poly.len() - 1][0];
    poly.push([z_last, r_top]);
    poly.push([z_first, r_top]);
    rasterize_solid_polygon(&poly, g)
}

// Grid validity (positive counts, strictly increasing edges) is enforced by
// `Grid`'s own constructors, so the rasterizer no longer re-checks it.

#[cfg(test)]
mod tests {
    use super::*;

    fn refs_lm(l_m: f64) -> RefScales {
        RefScales {
            l_m,
            p_pa: 5e6,
            rho_kg_m3: 4.13,
            u_m_s: 1100.0,
            t_k: 3200.0,
            time_s: l_m / 1100.0,
        }
    }

    /// Flat wall at r = 2.25 grid units with dr = 0.5: rows below fluid, the
    /// cut row exactly half solid, rows above fully solid.
    #[test]
    fn flat_wall_gives_exact_half_fraction() {
        let g = Grid::uniform(8, 8, 1.0, 0.5);
        let l_m = 0.05; // r_t; profile in metres
        let p = WallProfile {
            points: vec![[0.0, 2.25 * l_m], [8.0 * l_m, 2.25 * l_m]],
            throat_index: 0,
        };
        let s = rasterize(&p, &g, &refs_lm(l_m)).unwrap();
        for iz in 0..8 {
            for ir in 0..4 {
                assert_eq!(s.fraction[g.idx(iz, ir)], 0.0, "iz {iz} ir {ir}");
            }
            assert!((s.fraction[g.idx(iz, 4)] - 0.5).abs() < 1e-6); // cell [2.0, 2.5]
            for ir in 5..8 {
                assert_eq!(s.fraction[g.idx(iz, ir)], 1.0, "iz {iz} ir {ir}");
            }
        }
    }

    /// A sloped wall: fraction summed over a column equals the exact trapezoid
    /// area above the line.
    #[test]
    fn sloped_wall_column_sums_are_exact() {
        let g = Grid::uniform(10, 12, 1.0, 1.0);
        let l_m = 1.0;
        let p = WallProfile {
            points: vec![[0.0, 2.0], [10.0, 7.0]], // r_w(z) = 2 + 0.5 z
            throat_index: 0,
        };
        let s = rasterize(&p, &g, &refs_lm(l_m)).unwrap();
        for iz in 0..10 {
            let mut solid: f64 = 0.0;
            for ir in 0..12 {
                solid += s.fraction[g.idx(iz, ir)] as f64;
            }
            // Exact solid area in the column: 12 - mean(r_w) over the column.
            let z_mid = iz as f64 + 0.5;
            let expect = 12.0 - (2.0 + 0.5 * z_mid);
            assert!(
                (solid - expect).abs() < 1e-5,
                "iz {iz}: {solid} vs {expect}"
            );
        }
    }

    /// z outside the profile extent is fluid.
    #[test]
    fn beyond_profile_extent_is_fluid() {
        let g = Grid::uniform(10, 4, 1.0, 1.0);
        let p = WallProfile {
            points: vec![[2.0, 1.5], [6.0, 1.5]],
            throat_index: 0,
        };
        let s = rasterize(&p, &g, &refs_lm(1.0)).unwrap();
        for ir in 0..4 {
            assert_eq!(s.fraction[g.idx(0, ir)], 0.0);
            assert_eq!(s.fraction[g.idx(1, ir)], 0.0);
            assert_eq!(s.fraction[g.idx(7, ir)], 0.0);
            assert_eq!(s.fraction[g.idx(9, ir)], 0.0);
        }
        assert_eq!(s.fraction[g.idx(3, 3)], 1.0);
        assert_eq!(s.fraction[g.idx(3, 0)], 0.0);
    }

    #[test]
    fn degenerate_inputs_are_rejected() {
        // A zero-cell or non-monotone grid can no longer be constructed at
        // all — Grid::from_edges rejects it before the rasterizer runs.
        assert!(Grid::from_edges(vec![0.0], vec![0.0, 1.0]).is_err());
        let g = Grid::uniform(4, 4, 1.0, 1.0);
        assert!(rasterize_solid_polygon(&[[0.0, 0.0], [1.0, 0.0]], &g).is_err());
        assert!(rasterize_solid_polygon(&[[0.0, 0.0], [1.0, 0.0], [f64::NAN, 1.0]], &g).is_err());
    }
}
