//! Geometry-editor data model, behind a trait per docs/sessions/D-ui.md §5.
//! The UI codes against `EditorBackend`; `StubEditor` is a straight polyline
//! model that keeps the app alive until session C's `cfd_geom::Editor` lands
//! (same operations — swap the concrete type in `app.rs`, nothing else moves).
//! The UI supplies picking, drag, hover and the world/screen transform; the
//! backend owns the points and their invariants.

use crate::case::{LR, LZ};

pub trait EditorBackend {
    fn points(&self) -> &[[f64; 2]];
    /// Nearest control point within `tol` (world units), if any.
    fn hit_test(&self, world: [f64; 2], tol: f64) -> Option<usize>;
    /// Move point `i`, keeping the polyline valid (z strictly increasing,
    /// r inside the domain).
    fn drag(&mut self, i: usize, world: [f64; 2]);
    /// Insert a point on the nearest segment; returns its index.
    fn insert(&mut self, world: [f64; 2]) -> usize;
    /// Remove point `i`. Endpoints and a minimal polyline are protected.
    fn remove(&mut self, i: usize);
    /// Replace the whole polyline (parametric regeneration).
    fn set_points(&mut self, pts: Vec<[f64; 2]>);
}

pub struct StubEditor {
    points: Vec<[f64; 2]>,
    /// Domain extents in r_t; presets resize the domain, so these are set
    /// alongside `set_points` when the case changes.
    lz: f64,
    lr: f64,
}

/// Keep walls off the axis and clear of the radial sponge.
const R_MIN: f64 = 0.15;
const R_MARGIN: f64 = 1.5;
const Z_GAP: f64 = 1e-3;

impl StubEditor {
    pub fn new(points: Vec<[f64; 2]>) -> Self {
        StubEditor {
            points,
            lz: LZ,
            lr: LR,
        }
    }

    pub fn set_domain(&mut self, lz: f64, lr: f64) {
        self.lz = lz;
        self.lr = lr;
    }
}

impl EditorBackend for StubEditor {
    fn points(&self) -> &[[f64; 2]] {
        &self.points
    }

    fn hit_test(&self, world: [f64; 2], tol: f64) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, p) in self.points.iter().enumerate() {
            let d2 = (p[0] - world[0]).powi(2) + (p[1] - world[1]).powi(2);
            if d2 <= tol * tol && best.is_none_or(|(_, bd)| d2 < bd) {
                best = Some((i, d2));
            }
        }
        best.map(|(i, _)| i)
    }

    fn drag(&mut self, i: usize, world: [f64; 2]) {
        if i >= self.points.len() {
            return;
        }
        let z_lo = if i == 0 {
            0.0
        } else {
            self.points[i - 1][0] + Z_GAP
        };
        let z_hi = if i + 1 == self.points.len() {
            self.lz
        } else {
            self.points[i + 1][0] - Z_GAP
        };
        self.points[i] = [
            world[0].clamp(z_lo, z_hi),
            world[1].clamp(R_MIN, self.lr - R_MARGIN),
        ];
    }

    fn insert(&mut self, world: [f64; 2]) -> usize {
        let z = world[0].clamp(0.0, self.lz);
        let i = self.points.partition_point(|p| p[0] < z);
        self.points
            .insert(i, [z, world[1].clamp(R_MIN, self.lr - R_MARGIN)]);
        i
    }

    fn remove(&mut self, i: usize) {
        if self.points.len() > 3 && i > 0 && i + 1 < self.points.len() {
            self.points.remove(i);
        }
    }

    fn set_points(&mut self, pts: Vec<[f64; 2]>) {
        self.points = pts;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::conical_contour;

    #[test]
    fn drag_preserves_z_order_and_bounds() {
        let mut e = StubEditor::new(conical_contour(8.0));
        let n = e.points().len();
        e.drag(3, [-100.0, 100.0]); // hostile input
        for w in e.points().windows(2) {
            assert!(w[1][0] > w[0][0]);
        }
        assert!(e.points()[3][1] <= LR - R_MARGIN && e.points()[3][1] >= R_MIN);
        e.remove(0);
        e.remove(n - 1);
        assert_eq!(e.points().len(), n, "endpoints must be protected");
    }

    #[test]
    fn insert_keeps_sorted() {
        let mut e = StubEditor::new(conical_contour(8.0));
        let i = e.insert([5.0, 1.4]);
        assert!(i > 0 && e.points()[i] == [5.0, 1.4]);
        for w in e.points().windows(2) {
            assert!(w[1][0] >= w[0][0]);
        }
    }
}
