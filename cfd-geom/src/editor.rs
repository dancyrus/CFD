//! Editor data model — plain Rust, no egui. Session D wraps this in widgets and
//! converts screen to world coordinates; everything here is world space
//! (z, r in SI metres), matching `WallProfile`.

use crate::WallProfile;
use cfd_contract::Result;

/// Control points, selection, hit radius. Mutating operations keep z strictly
/// monotone and r positive so that `to_profile` succeeds after any sequence of
/// drags/inserts/removes; `to_profile` still validates as the final gate.
#[derive(Debug, Clone, PartialEq)]
pub struct Editor {
    pub points: Vec<[f64; 2]>,
    pub selection: Option<usize>,
    /// Default hit-test tolerance in world units (metres); D may pass its own
    /// zoom-dependent `tol` to `hit_test` instead.
    pub hit_radius: f64,
}

/// Minimum radius a control point may be dragged to, as a fraction of the
/// profile z-extent. Keeps r strictly positive (the rasterizer and the
/// axisymmetric solver both require r > 0).
const MIN_R_FRAC: f64 = 1e-6;
/// Minimum z-gap between neighbouring control points, as a fraction of extent.
const MIN_DZ_FRAC: f64 = 1e-6;

impl Editor {
    pub fn from_profile(p: &WallProfile) -> Self {
        let extent = Self::extent_of(&p.points);
        Editor {
            points: p.points.clone(),
            selection: None,
            hit_radius: 0.02 * extent,
        }
    }

    /// Build a `WallProfile`, recomputing `throat_index` as the minimum-radius
    /// control point (editing can move the throat), then validate: monotone z,
    /// all r > 0, no self-intersection, throat_index in range.
    pub fn to_profile(&self) -> Result<WallProfile> {
        let throat_index = self
            .points
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1[1]
                    .partial_cmp(&b.1[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        let p = WallProfile {
            points: self.points.clone(),
            throat_index,
        };
        p.validate()?;
        Ok(p)
    }

    /// Nearest control point within `tol` (world units) of `world`, or None.
    pub fn hit_test(&self, world: [f64; 2], tol: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, p) in self.points.iter().enumerate() {
            let d2 = (p[0] - world[0]).powi(2) + (p[1] - world[1]).powi(2);
            if d2 <= tol * tol && best.is_none_or(|(_, b)| d2 < b) {
                best = Some((i, d2));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Move control point `i` to `world`, clamped so z stays strictly between
    /// its neighbours and r stays positive. Selects the point. Out-of-range `i`
    /// is a no-op.
    pub fn drag(&mut self, i: usize, world: [f64; 2]) {
        if i >= self.points.len() {
            return;
        }
        let gap = MIN_DZ_FRAC * self.extent();
        let lo = (i > 0).then(|| self.points[i - 1][0] + gap);
        let hi = (i + 1 < self.points.len()).then(|| self.points[i + 1][0] - gap);
        let z = match (lo, hi) {
            // Neighbours closer than two gaps: pin to their strict midpoint.
            (Some(lo), Some(hi)) if hi < lo => {
                0.5 * (self.points[i - 1][0] + self.points[i + 1][0])
            }
            (lo, hi) => {
                let mut z = world[0];
                if let Some(lo) = lo {
                    z = z.max(lo);
                }
                if let Some(hi) = hi {
                    z = z.min(hi);
                }
                z
            }
        };
        let r = world[1].max(MIN_R_FRAC * self.extent());
        self.points[i] = [z, r];
        self.selection = Some(i);
    }

    /// Insert a control point at `world`, placed in z-order (clamped like
    /// `drag` if it collides with a neighbour), and select it.
    pub fn insert(&mut self, world: [f64; 2]) {
        let k = self.points.partition_point(|p| p[0] <= world[0]);
        let r = world[1].max(MIN_R_FRAC * self.extent());
        self.points.insert(k, [world[0], r]);
        // Re-use drag's clamping to guarantee strict monotonicity.
        self.drag(k, [world[0], r]);
    }

    /// Remove control point `i`. Refuses to shrink below 2 points (the minimum
    /// valid profile); out-of-range `i` is a no-op.
    pub fn remove(&mut self, i: usize) {
        if i >= self.points.len() || self.points.len() <= 2 {
            return;
        }
        self.points.remove(i);
        self.selection = match self.selection {
            Some(s) if s == i => None,
            Some(s) if s > i => Some(s - 1),
            other => other,
        };
    }

    fn extent(&self) -> f64 {
        Self::extent_of(&self.points)
    }

    fn extent_of(points: &[[f64; 2]]) -> f64 {
        match (points.first(), points.last()) {
            (Some(a), Some(b)) if b[0] > a[0] => b[0] - a[0],
            _ => 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> WallProfile {
        WallProfile {
            points: vec![[0.0, 0.10], [0.10, 0.05], [0.30, 0.12]],
            throat_index: 1,
        }
    }

    #[test]
    fn round_trip_preserves_points_and_recomputes_throat() {
        let e = Editor::from_profile(&profile());
        let p = e.to_profile().unwrap();
        assert_eq!(p.points, profile().points);
        assert_eq!(p.throat_index, 1);
    }

    #[test]
    fn drag_moves_throat_and_to_profile_tracks_it() {
        let mut e = Editor::from_profile(&profile());
        // Pull the last point below the old throat.
        e.drag(2, [0.30, 0.01]);
        let p = e.to_profile().unwrap();
        assert_eq!(p.throat_index, 2);
        assert_eq!(e.selection, Some(2));
    }

    #[test]
    fn hit_test_picks_nearest_within_tol() {
        let e = Editor::from_profile(&profile());
        assert_eq!(e.hit_test([0.101, 0.049], 0.01), Some(1));
        assert_eq!(e.hit_test([0.2, 0.2], 0.01), None);
        // Between points 0 and 1: picks the closer one.
        assert_eq!(e.hit_test([0.02, 0.09], 1.0), Some(0));
    }

    #[test]
    fn drag_clamps_z_between_neighbours_and_r_positive() {
        let mut e = Editor::from_profile(&profile());
        e.drag(1, [-5.0, -3.0]); // way past the left neighbour, negative r
        let p = e.to_profile().unwrap(); // still valid
        assert!(p.points[1][0] > p.points[0][0]);
        assert!(p.points[1][1] > 0.0);
        e.drag(1, [99.0, 0.05]); // way past the right neighbour
        assert!(e.to_profile().is_ok());
        assert!(e.points[1][0] < e.points[2][0]);
    }

    #[test]
    fn insert_keeps_z_order_and_remove_keeps_two_points() {
        let mut e = Editor::from_profile(&profile());
        e.insert([0.20, 0.08]);
        assert_eq!(e.points.len(), 4);
        assert!(e.to_profile().is_ok());
        assert_eq!(e.selection, Some(2)); // inserted between old points 1 and 2
                                          // Insert at a duplicate z still yields a valid (strictly monotone) profile.
        e.insert([0.20, 0.09]);
        assert!(e.to_profile().is_ok());
        // Remove down to the floor of 2 points.
        for _ in 0..10 {
            e.remove(0);
        }
        assert_eq!(e.points.len(), 2);
        assert!(e.to_profile().is_ok());
    }
}
