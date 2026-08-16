//! Nozzle contours, rasterizer, editor data model. Session C owns this crate.
//! No egui dependency — ever. See docs/sessions/C-geom.md and docs/contract.md.
//!
//! Lengths in `NozzleSpec`, `WallProfile` and `Editor` are SI metres.
//! `rasterize` converts to the non-dimensional grid (units of throat radius)
//! through `RefScales`; nothing else in this crate touches units.

#![forbid(unsafe_code)]

mod contour;
mod editor;
mod grade;
mod rao;
mod raster;

pub use contour::generate_contour;
pub use editor::Editor;
pub use grade::{grade_from_solid, GradeSpec};
pub use rao::rao_angles;
pub use raster::{rasterize, rasterize_solid_polygon};

use cfd_contract::{CfdError, Real, Result};

/// Diverging-section shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContourKind {
    Conical {
        half_angle_deg: f64,
    },
    /// Rao parabolic-bell approximation; theta_n, theta_e come from the Rao table.
    ParabolicBell {
        bell_percent: f64,
    },
    /// Parabolic bell with MEASURED wall angles: theta_n and theta_e are taken
    /// straight from published geometry instead of the Rao table, and the
    /// downstream throat arc comes from `NozzleSpec::throat_arc_down` like any
    /// other spec input. For engines the table cannot represent — Raptor 2's
    /// published 32.0 deg / 6.0 deg pair sits on no single bell-percent row,
    /// and its 0.300 R_t throat arc is not the 0.382 R_t of the Rao/TOP
    /// construction.
    ///
    /// `length_fraction` is still the divergent length as a fraction of the
    /// Huzel-Huang 15 deg reference cone: two wall angles and an exit radius
    /// do not by themselves fix a length, so a bell always needs one. It is a
    /// LENGTH parameter here, not a claim that the contour is a Rao bell.
    DirectBell {
        theta_n_deg: f64,
        theta_e_deg: f64,
        length_fraction: f64,
    },
}

/// Parametric nozzle description. All lengths SI; arc radii in units of r_t.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NozzleSpec {
    pub throat_radius_m: f64,
    pub area_ratio: f64,
    pub contraction_ratio: f64,       // 4.0
    pub converge_half_angle_deg: f64, // 30.0
    pub throat_arc_up: f64,           // 1.5  (x r_t)
    pub throat_arc_down: f64,         // 0.382
    pub contour: ContourKind,
}

impl NozzleSpec {
    pub fn exit_radius_m(&self) -> f64 {
        self.throat_radius_m * self.area_ratio.sqrt()
    }

    pub fn throat_area_m2(&self) -> f64 {
        std::f64::consts::PI * self.throat_radius_m * self.throat_radius_m
    }

    pub fn validate(&self) -> Result<()> {
        fn bad(msg: String) -> CfdError {
            CfdError::Geometry(msg)
        }
        if !(self.throat_radius_m.is_finite() && self.throat_radius_m > 0.0) {
            return Err(bad(format!("throat_radius_m = {}", self.throat_radius_m)));
        }
        if !(self.area_ratio.is_finite() && self.area_ratio > 1.0) {
            return Err(bad(format!(
                "area_ratio = {} (must be > 1)",
                self.area_ratio
            )));
        }
        if !(self.contraction_ratio.is_finite() && self.contraction_ratio > 1.0) {
            return Err(bad(format!(
                "contraction_ratio = {} (must be > 1)",
                self.contraction_ratio
            )));
        }
        if !(self.converge_half_angle_deg > 0.0 && self.converge_half_angle_deg < 90.0) {
            return Err(bad(format!(
                "converge_half_angle_deg = {}",
                self.converge_half_angle_deg
            )));
        }
        if !(self.throat_arc_up.is_finite() && self.throat_arc_up > 0.0) {
            return Err(bad(format!("throat_arc_up = {}", self.throat_arc_up)));
        }
        if !(self.throat_arc_down.is_finite() && self.throat_arc_down > 0.0) {
            return Err(bad(format!("throat_arc_down = {}", self.throat_arc_down)));
        }
        // The converging cone must have positive length: the chamber radius has to
        // sit above the upstream-arc tangency radius.
        let beta = self.converge_half_angle_deg.to_radians();
        let ra_up = 1.0 + self.throat_arc_up * (1.0 - beta.cos());
        let rc = self.contraction_ratio.sqrt();
        if rc <= ra_up {
            return Err(bad(format!(
                "chamber radius {rc:.4} r_t does not clear the upstream throat arc \
                 (tangency at {ra_up:.4} r_t); increase contraction_ratio or reduce throat_arc_up"
            )));
        }
        match self.contour {
            ContourKind::Conical { half_angle_deg } => {
                if !(half_angle_deg > 0.0 && half_angle_deg < 90.0) {
                    return Err(bad(format!("cone half_angle_deg = {half_angle_deg}")));
                }
                let alpha = half_angle_deg.to_radians();
                let ra_dn = 1.0 + self.throat_arc_down * (1.0 - alpha.cos());
                if self.area_ratio.sqrt() <= ra_dn {
                    return Err(bad(format!(
                        "area_ratio {} too small: exit radius sits inside the downstream throat arc",
                        self.area_ratio
                    )));
                }
            }
            ContourKind::ParabolicBell { bell_percent } => {
                // The Rao table is digitized for 60-90% bells only; do not extrapolate.
                if !(0.6..=0.9).contains(&bell_percent) {
                    return Err(bad(format!(
                        "bell_percent = {bell_percent} (supported range 0.6..=0.9)"
                    )));
                }
            }
            ContourKind::DirectBell {
                theta_n_deg,
                theta_e_deg,
                length_fraction,
            } => {
                // No table to stay inside: the angles are the measurement. What
                // has to hold is that the geometry closes — theta_n above
                // theta_e (the Bezier control point divides by their tangent
                // difference), both wall angles strictly inside the first
                // quadrant, and a length in a range that is still a bell.
                if !(theta_n_deg > 0.0 && theta_n_deg < 90.0) {
                    return Err(bad(format!("theta_n_deg = {theta_n_deg}")));
                }
                // theta_e = 0 is excluded on purpose and is NOT merely
                // degenerate-looking: the contour's only length bound is the
                // N_z < Q_z < E_z ordering, and Q_z moves with E_z at rate
                // tan(theta_e). At theta_e = 0 that coupling vanishes, Q_z
                // stops depending on the length, and a 1e9 r_t "bell" passes
                // every check. An axial exit is not a bell anyway.
                if !(theta_e_deg > 0.0 && theta_e_deg < theta_n_deg) {
                    return Err(bad(format!(
                        "theta_e_deg = {theta_e_deg} (must be in 0 < theta_e < theta_n = \
                         {theta_n_deg}; a zero exit angle leaves the length unbounded)"
                    )));
                }
                // Absolute length bound, so a near-zero theta_e cannot make the
                // ordering guard toothless either. Past the 15 deg reference
                // cone's own length there is no bell left to speak of.
                if !(length_fraction.is_finite() && length_fraction > 0.0 && length_fraction <= 1.5)
                {
                    return Err(bad(format!(
                        "length_fraction = {length_fraction} (supported range 0.0..=1.5 of the \
                         15 deg reference cone)"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Polyline in the (z, r) half-plane, increasing z, all r > 0, SI metres. Also the
/// EDITABLE representation: dragging a control point edits `points` directly, so a
/// hand-edited wall and a generated wall are the same type and the solver cannot
/// tell them apart.
#[derive(Debug, Clone, PartialEq)]
pub struct WallProfile {
    pub points: Vec<[f64; 2]>,
    pub throat_index: usize,
}

impl WallProfile {
    /// Wall radius at axial station `z` (metres), linearly interpolated.
    /// `None` outside the profile's z-extent.
    pub fn radius_at(&self, z: f64) -> Option<f64> {
        let pts = &self.points;
        if pts.len() < 2 || z < pts[0][0] || z > pts[pts.len() - 1][0] {
            return None;
        }
        // First point with p.z > z; the segment [k-1, k] brackets z.
        let k = pts.partition_point(|p| p[0] <= z);
        if k == 0 {
            return Some(pts[0][1]);
        }
        if k == pts.len() {
            return Some(pts[pts.len() - 1][1]);
        }
        let (a, b) = (pts[k - 1], pts[k]);
        let t = (z - a[0]) / (b[0] - a[0]);
        Some(a[1] + t * (b[1] - a[1]))
    }

    pub fn validate(&self) -> Result<()> {
        if self.points.len() < 2 {
            return Err(CfdError::Geometry(format!(
                "profile needs at least 2 points, has {}",
                self.points.len()
            )));
        }
        for (i, p) in self.points.iter().enumerate() {
            if !(p[0].is_finite() && p[1].is_finite()) {
                return Err(CfdError::Geometry(format!(
                    "point {i} is not finite: {p:?}"
                )));
            }
            if p[1] <= 0.0 {
                return Err(CfdError::Geometry(format!(
                    "point {i} has r = {} <= 0",
                    p[1]
                )));
            }
        }
        // Strictly increasing z. For a function-like polyline this also rules out
        // self-intersection.
        for i in 1..self.points.len() {
            if self.points[i][0] <= self.points[i - 1][0] {
                return Err(CfdError::Geometry(format!(
                    "z not strictly increasing at point {i}: {} -> {}",
                    self.points[i - 1][0],
                    self.points[i][0]
                )));
            }
        }
        if self.throat_index >= self.points.len() {
            return Err(CfdError::Geometry(format!(
                "throat_index {} out of range (len {})",
                self.throat_index,
                self.points.len()
            )));
        }
        Ok(())
    }
}

/// Snap `dr` so the throat radius (exactly r = 1 in non-dimensional units, since
/// L_ref = r_t) lands on a cell face: returns the nearest `dr` with 1/dr integer.
/// This collapses the +/-1/N_throat systematic mass-flow error to the second-order
/// arc-curvature term (physics-reference §8). **Parametric nozzle only** — do not
/// apply it to drawn geometry, which has no theoretical thrust to match.
pub fn quantize_dr_to_throat(dr: Real) -> Real {
    let n = (1.0 / dr as f64).round().max(1.0);
    (1.0 / n) as Real
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_dr_snaps_to_integer_throat_cells() {
        assert_eq!(quantize_dr_to_throat(0.05), 0.05); // already 1/20
        let q = quantize_dr_to_throat(0.0509);
        assert!((q - 0.05).abs() < 1e-9); // rounds to 1/20
        let q = quantize_dr_to_throat(0.026);
        assert!((q as f64 - 1.0 / 38.0).abs() < 1e-9);
        assert_eq!(quantize_dr_to_throat(2.0), 1.0); // degenerate: at least 1 cell
    }

    #[test]
    fn radius_at_interpolates_and_bounds() {
        let p = WallProfile {
            points: vec![[0.0, 2.0], [1.0, 1.0], [3.0, 2.0]],
            throat_index: 1,
        };
        assert!(p.validate().is_ok());
        assert_eq!(p.radius_at(-0.1), None);
        assert_eq!(p.radius_at(3.1), None);
        assert!((p.radius_at(0.5).unwrap() - 1.5).abs() < 1e-12);
        assert!((p.radius_at(1.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((p.radius_at(2.0).unwrap() - 1.5).abs() < 1e-12);
        assert!((p.radius_at(3.0).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn profile_validation_rejects_bad_input() {
        let mut p = WallProfile {
            points: vec![[0.0, 2.0], [1.0, 1.0]],
            throat_index: 5,
        };
        assert!(p.validate().is_err()); // throat_index out of range
        p.throat_index = 1;
        assert!(p.validate().is_ok());
        p.points[1][0] = -1.0;
        assert!(p.validate().is_err()); // z not increasing
        p.points[1] = [1.0, -0.5];
        assert!(p.validate().is_err()); // r <= 0
    }
}
