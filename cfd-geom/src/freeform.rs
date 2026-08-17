//! The freeform (drawn) wall editor — a plain polyline whose points ARE the
//! control points, for geometry that is not a member of any contour family.
//!
//! **This is a merge of two editors that both existed and neither of which was
//! a superset of the other.** `cfd_geom::Editor` had the validation gate — a
//! `to_profile` that ran `WallProfile::validate` and recomputed the throat as
//! the minimum-radius point — but no idea the domain had edges, so a point
//! could be dragged into the radial sponge or off the end of the grid.
//! `cfd_ui::StubEditor` had the domain clamp, `R_MIN`, `R_MARGIN` and endpoint
//! protection, but no validation gate at all, so it could hand the rasterizer a
//! polyline it had never checked. The union of the two is below, and the bounds
//! the app used to hard-code are now INJECTED, which is what let the two
//! become one type.
//!
//! Units are the caller's, and `cfd-ui` passes r_t (the units the canvas, the
//! rasterizer and `CaseParams`-derived walls all use). The class of checks here
//! — ordering, positivity, separation — is unit-independent; only `bounds`
//! carries a scale, and it is supplied by the caller in the same units.

use crate::WallProfile;
use cfd_contract::Result;

/// The box a drawn wall must stay inside. Injected, because `cfd-geom` does not
/// know how big the domain is and `cfd-ui` does.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreeformBounds {
    /// Domain extents.
    pub lz: f64,
    pub lr: f64,
    /// Keep the wall off the axis: `r >= r_min`. The axisymmetric solver and
    /// the rasterizer both require r > 0, and a wall ON the axis is a plug.
    pub r_min: f64,
    /// Keep the wall clear of the radial sponge, `r <= lr - r_margin`. A wall
    /// inside the sponge is being relaxed toward ambient while it reflects.
    pub r_margin: f64,
    /// Minimum z separation between neighbouring points.
    pub z_gap: f64,
}

impl FreeformBounds {
    /// The values `cfd-ui` used to hard-code, in r_t.
    pub const R_MIN: f64 = 0.15;
    pub const R_MARGIN: f64 = 1.5;
    pub const Z_GAP: f64 = 1e-3;

    pub fn for_domain(lz: f64, lr: f64) -> Self {
        FreeformBounds {
            lz,
            lr,
            r_min: Self::R_MIN,
            r_margin: Self::R_MARGIN,
            z_gap: Self::Z_GAP,
        }
    }

    /// The clamped radius for a drawn point. Degenerate domains (a radius under
    /// the margin) collapse to `r_min` rather than inverting the interval.
    fn clamp_r(&self, r: f64) -> f64 {
        let hi = (self.lr - self.r_margin).max(self.r_min);
        r.clamp(self.r_min, hi)
    }
}

/// A drawn wall: control points, selection, and the bounds they live in.
///
/// Every mutating operation keeps z strictly increasing and r positive, so
/// [`to_profile`](FreeformEditor::to_profile) succeeds after any sequence of
/// drags, inserts and removes — and `to_profile` still validates, as the final
/// gate. Belt and braces on purpose: the invariants are what make the editor
/// usable, the gate is what stops a bug in them reaching the solver.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeformEditor {
    points: Vec<[f64; 2]>,
    selection: Option<usize>,
    bounds: FreeformBounds,
}

/// Minimum points a drawn wall may be reduced to. Three, not two: the endpoints
/// are protected from removal, so two would leave nothing removable and a wall
/// with no interior at all.
const MIN_POINTS: usize = 3;

impl FreeformEditor {
    /// Take over from a polyline — the parametric wall's tessellation, at the
    /// moment of the break. The shape does not move: these are exactly the
    /// points that were on screen.
    pub fn new(points: Vec<[f64; 2]>, bounds: FreeformBounds) -> Self {
        FreeformEditor {
            points,
            selection: None,
            bounds,
        }
    }

    pub fn points(&self) -> &[[f64; 2]] {
        &self.points
    }

    pub fn selection(&self) -> Option<usize> {
        self.selection
    }

    pub fn bounds(&self) -> FreeformBounds {
        self.bounds
    }

    /// Presets and the domain fields resize the box. Points outside the new
    /// box are pulled into it.
    ///
    /// Re-clamping moves the wall, which is unwelcome — but the alternative is
    /// worse in a way the user cannot see. `rasterize_solid_polygon` clips
    /// against the grid, so a point left outside the domain produces a SOLVED
    /// wall that differs from the DRAWN one with nothing on screen saying so.
    /// It also produced a spike: the first drag of any out-of-box point snapped
    /// it to the new ceiling while its neighbours stayed where they were.
    pub fn set_bounds(&mut self, bounds: FreeformBounds) {
        self.bounds = bounds;
        for p in &mut self.points {
            p[0] = p[0].clamp(0.0, bounds.lz);
            p[1] = bounds.clamp_r(p[1]);
        }
        // Clamping z can collapse neighbours; restore strict monotonicity by
        // spreading any run that lands on the same station.
        for i in 1..self.points.len() {
            let lo = self.points[i - 1][0] + bounds.z_gap;
            if self.points[i][0] < lo {
                self.points[i][0] = lo;
            }
        }
    }

    /// Nearest control point within `tol`, or None.
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

    /// Move point `i` to `world`, clamped so z stays strictly between its
    /// neighbours (and inside the domain at the ends) and r stays inside the
    /// bounds. Selects the point. Out-of-range `i` is a no-op.
    pub fn drag(&mut self, i: usize, world: [f64; 2]) {
        if i >= self.points.len() {
            return;
        }
        let gap = self.bounds.z_gap;
        let lo = if i > 0 {
            self.points[i - 1][0] + gap
        } else {
            0.0
        };
        let hi = if i + 1 < self.points.len() {
            self.points[i + 1][0] - gap
        } else {
            self.bounds.lz
        };
        let z = if hi < lo && i > 0 && i + 1 < self.points.len() {
            // Neighbours closer together than two gaps: pin to their strict
            // midpoint, which is still between them.
            0.5 * (self.points[i - 1][0] + self.points[i + 1][0])
        } else {
            world[0].clamp(lo.min(hi), hi.max(lo))
        };
        self.points[i] = [z, self.bounds.clamp_r(world[1])];
        self.selection = Some(i);
    }

    /// Insert a control point at `world`, placed in z-order, and select it.
    /// Returns its index.
    pub fn insert(&mut self, world: [f64; 2]) -> usize {
        let z = world[0].clamp(0.0, self.bounds.lz);
        let k = self.points.partition_point(|p| p[0] <= z);
        self.points.insert(k, [z, self.bounds.clamp_r(world[1])]);
        // Re-use drag's clamping to guarantee strict monotonicity.
        self.drag(k, [z, world[1]]);
        k
    }

    /// Remove point `i`. Endpoints are protected and the wall never drops below
    /// [`MIN_POINTS`]. Returns whether anything was removed.
    pub fn remove(&mut self, i: usize) -> bool {
        if i == 0 || i + 1 >= self.points.len() || self.points.len() <= MIN_POINTS {
            return false;
        }
        self.points.remove(i);
        self.selection = match self.selection {
            Some(s) if s == i => None,
            Some(s) if s > i => Some(s - 1),
            other => other,
        };
        true
    }

    /// Build a `WallProfile`, recomputing `throat_index` as the minimum-radius
    /// control point (drawing can move the throat anywhere), then validate:
    /// monotone z, all r > 0, throat_index in range.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn editor() -> FreeformEditor {
        FreeformEditor::new(
            vec![[0.0, 2.0], [4.0, 1.0], [10.0, 2.8]],
            FreeformBounds::for_domain(46.4, 10.0),
        )
    }

    #[test]
    fn round_trip_preserves_points_and_finds_the_throat() {
        let e = editor();
        let p = e.to_profile().unwrap();
        assert_eq!(p.points, e.points());
        assert_eq!(p.throat_index, 1);
    }

    /// The half `cfd_geom::Editor` had and `StubEditor` did not: every mutation
    /// leaves a polyline that passes the gate.
    #[test]
    fn every_mutation_leaves_a_valid_profile() {
        let mut e = editor();
        e.drag(1, [-500.0, -500.0]); // hostile
        assert!(e.to_profile().is_ok());
        e.drag(1, [500.0, 500.0]);
        assert!(e.to_profile().is_ok());
        e.insert([2.0, 1.5]);
        assert!(e.to_profile().is_ok());
        e.insert([2.0, 1.6]); // duplicate z
        assert!(e.to_profile().is_ok());
        for i in (0..e.points().len()).rev() {
            e.remove(i);
            assert!(e.to_profile().is_ok());
        }
        // Dragging the last point below the old throat moves the throat.
        let n = e.points().len() - 1;
        e.drag(n, [e.points()[n][0], 0.2]);
        assert_eq!(e.to_profile().unwrap().throat_index, n);
    }

    /// The half `StubEditor` had and `cfd_geom::Editor` did not: the domain box.
    #[test]
    fn points_stay_inside_the_injected_bounds() {
        let mut e = editor();
        let b = e.bounds();
        e.drag(1, [4.0, 1e9]);
        assert!((e.points()[1][1] - (b.lr - b.r_margin)).abs() < 1e-12);
        e.drag(1, [4.0, -1e9]);
        assert!((e.points()[1][1] - b.r_min).abs() < 1e-12);
        // Endpoints are held inside [0, lz].
        e.drag(0, [-99.0, 2.0]);
        assert!(e.points()[0][0] >= 0.0);
        let n = e.points().len() - 1;
        e.drag(n, [1e9, 2.0]);
        assert!(e.points()[n][0] <= b.lz + 1e-12);
        // Interior points stay strictly between their neighbours.
        for w in e.points().windows(2) {
            assert!(w[1][0] > w[0][0]);
        }
        // A domain smaller than the margin collapses to r_min rather than
        // inverting the clamp interval (which would panic).
        let mut tiny = FreeformEditor::new(e.points().to_vec(), FreeformBounds::for_domain(46.4, 1.0));
        tiny.drag(1, [4.0, 5.0]);
        assert!((tiny.points()[1][1] - FreeformBounds::R_MIN).abs() < 1e-12);
    }

    #[test]
    fn endpoints_are_protected_and_the_wall_keeps_a_floor() {
        let mut e = editor();
        e.insert([2.0, 1.5]);
        e.insert([6.0, 1.5]);
        let n = e.points().len();
        assert!(!e.remove(0), "first point must be protected");
        assert!(!e.remove(n - 1), "last point must be protected");
        assert_eq!(e.points().len(), n);
        // Interior points come off until the floor.
        while e.remove(1) {}
        assert_eq!(e.points().len(), MIN_POINTS);
        assert!(e.to_profile().is_ok());
    }

    /// A domain that shrinks under a drawn wall must pull the wall in, and the
    /// result must still be a valid profile — the rasterizer clips silently, so
    /// a wall left hanging outside the box is solved as something other than
    /// what is drawn.
    #[test]
    fn shrinking_the_domain_pulls_the_wall_inside_it() {
        let mut e = FreeformEditor::new(
            vec![[0.0, 8.3], [4.0, 1.0], [10.0, 8.3], [40.0, 8.3]],
            FreeformBounds::for_domain(46.4, 27.0),
        );
        assert!(e.to_profile().is_ok());
        e.set_bounds(FreeformBounds::for_domain(12.0, 5.0));
        let b = e.bounds();
        for p in e.points() {
            assert!(p[0] >= 0.0 && p[0] <= b.lz + 1e-9, "z {} outside [0, {}]", p[0], b.lz);
            assert!(
                p[1] >= b.r_min - 1e-12 && p[1] <= b.lr - b.r_margin + 1e-12,
                "r {} outside the box",
                p[1]
            );
        }
        // Two points were at z 10 and 40, both now clamped to lz = 12 — they
        // must not have collapsed onto each other.
        for w in e.points().windows(2) {
            assert!(w[1][0] > w[0][0], "z collapsed at {:?}", w);
        }
        assert!(e.to_profile().is_ok(), "still a valid profile after the shrink");
    }

    #[test]
    fn hit_test_picks_the_nearest_within_tolerance() {
        let e = editor();
        assert_eq!(e.hit_test([4.05, 0.95], 0.2), Some(1));
        assert_eq!(e.hit_test([7.0, 7.0], 0.2), None);
        assert_eq!(e.hit_test([0.5, 1.9], 5.0), Some(0));
    }

    #[test]
    fn selection_follows_removal() {
        let mut e = editor();
        e.insert([2.0, 1.5]);
        e.insert([6.0, 1.5]);
        assert_eq!(e.selection(), Some(3));
        assert_eq!(e.points().len(), 5);
        // Removing BEFORE the selection shifts it down.
        assert!(e.remove(1));
        assert_eq!(e.selection(), Some(2));
        // Removing the selection itself clears it.
        e.drag(1, [3.0, 1.2]);
        assert_eq!(e.selection(), Some(1));
        assert!(e.remove(1));
        assert_eq!(e.selection(), None);
        // At the floor, removal is refused and the selection is untouched.
        assert_eq!(e.points().len(), MIN_POINTS);
        e.drag(1, [3.0, 1.2]);
        assert!(!e.remove(1));
        assert_eq!(e.selection(), Some(1));
    }
}
