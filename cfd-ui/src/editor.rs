//! The wall-editing model the canvas codes against.
//!
//! **What changed and why.** This file used to expose `EditorBackend`, a
//! polyline model whose control points WERE the solver's input: 256 of them,
//! one drag handle every ~0.1 r_t, and every one of them a way to put a kink in
//! a nozzle wall. The handle model replaces it. A parametric wall has nine
//! markers for a bell and six for a cone — one per editable degree of freedom —
//! and the polyline underneath is regenerated, never dragged.
//!
//! The UI still owns picking, hover and the world/screen transform, and it now
//! owns ALL of the pixels: the backend hands out anchors in r_t and tangent
//! DIRECTIONS, and the canvas decides how many pixels along a direction a dot
//! goes. `cfd-geom` has no idea what a pixel is.

use cfd_geom::{HandleKind, NozzleCurve, Tessellation};

use crate::case::CaseParams;

/// A marker the canvas draws, and usually picks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditHandle {
    /// Where it attaches to the wall, r_t, folded frame (r >= 0).
    pub anchor: [f64; 2],
    pub kind: HandleKind,
    pub label: &'static str,
    /// The parameter's current value, for the sidebar readout.
    pub value: f64,
    pub unit: &'static str,
    /// False only for the derived Bézier control point Q.
    pub pickable: bool,
}

/// Whatever is currently editing the wall.
///
/// Two implementations: [`ParametricEditor`] (handles on a curve) and, once the
/// user breaks the parametric link, the freeform point editor.
pub trait WallEditor {
    /// The polyline the solver and the rasterizer get, r_t.
    fn polyline(&self) -> &[[f64; 2]];
    /// Every marker to draw. Index into this is the handle index used by
    /// `drag` and `remove`.
    fn handles(&self) -> &[EditHandle];
    /// Move handle `i` toward `folded` (r_t, r >= 0). Returns the clamp reason
    /// when the request was infeasible and the handle stopped short.
    fn drag(&mut self, i: usize, folded: [f64; 2]) -> Option<String>;
    /// Ctrl+click. `false` means this editor has no notion of inserting a point
    /// — the app turns that into the one-way break to freeform.
    fn insert(&mut self, folded: [f64; 2]) -> bool;
    /// Right-click. `false` as above.
    fn remove(&mut self, i: usize) -> bool;
    /// The dashed control polygon (N–Q–E for a bell), if this editor has one.
    fn control_polygon(&self) -> Option<Vec<[f64; 2]>>;
    /// Domain extents in r_t; presets resize the domain.
    fn set_domain(&mut self, lz: f64, lr: f64);
}

/// A parametric wall edited through CAD handles.
pub struct ParametricEditor {
    curve: NozzleCurve,
    tess: Tessellation,
    /// Throat radius in metres — `cfd_geom` speaks SI, the canvas speaks r_t.
    rt: f64,
    points: Vec<[f64; 2]>,
    handles: Vec<EditHandle>,
}

impl ParametricEditor {
    pub fn new(p: &CaseParams) -> Self {
        let mut e = ParametricEditor {
            curve: crate::case::nozzle_curve(p),
            tess: crate::case::tessellation(p),
            rt: p.r_throat_m,
            points: Vec::new(),
            handles: Vec::new(),
        };
        e.rebuild();
        e
    }

    /// Re-point at a different case (preset applied, area ratio slider moved,
    /// mesh resolution changed).
    pub fn reset(&mut self, p: &CaseParams) {
        self.curve = crate::case::nozzle_curve(p);
        self.tess = crate::case::tessellation(p);
        self.rt = p.r_throat_m;
        self.rebuild();
    }

    pub fn curve(&self) -> &NozzleCurve {
        &self.curve
    }

    /// Regenerate the polyline and the handles from the curve. Called after
    /// every accepted drag — the polyline is DERIVED, so nothing else has to
    /// keep the two in step.
    fn rebuild(&mut self) {
        let inv = 1.0 / self.rt;
        if let Ok(p) = self.curve.tessellate(&self.tess) {
            self.points = p
                .points
                .iter()
                .map(|q| [q[0] * inv, q[1] * inv])
                .collect();
        }
        self.handles = self
            .curve
            .handles()
            .unwrap_or_default()
            .into_iter()
            .map(|h| EditHandle {
                anchor: h.anchor,
                kind: h.kind,
                label: h.id.label(),
                value: h.value,
                unit: h.unit,
                pickable: h.pickable,
            })
            .collect();
    }
}

impl WallEditor for ParametricEditor {
    fn polyline(&self) -> &[[f64; 2]] {
        &self.points
    }

    fn handles(&self) -> &[EditHandle] {
        &self.handles
    }

    fn drag(&mut self, i: usize, folded: [f64; 2]) -> Option<String> {
        let ids = self.curve.handles().unwrap_or_default();
        let Some(h) = ids.get(i).filter(|h| h.pickable) else {
            return None;
        };
        let out = self.curve.drag_handle(h.id, folded);
        self.curve = out.curve;
        self.rebuild();
        out.clamped
    }

    /// A parametric wall has no points to insert between. The app reads the
    /// `false` as a request to break the parametric link.
    fn insert(&mut self, _folded: [f64; 2]) -> bool {
        false
    }

    fn remove(&mut self, _i: usize) -> bool {
        false
    }

    fn control_polygon(&self) -> Option<Vec<[f64; 2]>> {
        self.curve.control_polygon().map(|p| p.to_vec())
    }

    /// The parametric wall is bounded by its own feasibility, not by the
    /// domain: every handle drag is clamped to a wall that closes, and a nozzle
    /// longer than the domain is a sizing mistake the cost readout already
    /// shows. Only the freeform editor clamps to the box.
    fn set_domain(&mut self, _lz: f64, _lr: f64) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{ContourKind, PRESETS};

    #[test]
    fn a_bell_gets_nine_handles_and_a_dense_polyline() {
        let p = PRESETS.iter().find(|p| p.name == "RS-25").unwrap().case(0.0, false);
        let e = ParametricEditor::new(&p);
        assert_eq!(e.handles().len(), 9);
        assert_eq!(e.handles().iter().filter(|h| h.pickable).count(), 8);
        // The polyline is dense and NOT the handle set — the whole point.
        assert!(e.polyline().len() > 60, "{} points", e.polyline().len());
        assert!(e.control_polygon().is_some());
        // Demo cone: six handles, all pickable, no control polygon.
        let e = ParametricEditor::new(&CaseParams::default());
        assert_eq!(e.handles().len(), 6);
        assert!(e.handles().iter().all(|h| h.pickable));
        assert!(e.control_polygon().is_none());
    }

    /// Dragging theta_n flips the case to a measured bell and the case round
    /// trips through `apply_curve` — this is the path the report's live angle
    /// readout depends on.
    #[test]
    fn a_theta_drag_lands_in_the_case_as_a_measured_bell() {
        let mut p = PRESETS.iter().find(|p| p.name == "RS-25").unwrap().case(0.0, false);
        let mut e = ParametricEditor::new(&p);
        let i = e
            .handles()
            .iter()
            .position(|h| h.label == "theta_n")
            .unwrap();
        let h = e.handles()[i];
        let want = (h.value - 3.0).to_radians();
        assert_eq!(
            e.drag(i, [h.anchor[0] + want.cos(), h.anchor[1] + want.sin()]),
            None
        );
        crate::case::apply_curve(&mut p, e.curve());
        match p.contour_kind {
            ContourKind::MeasuredBell { theta_n_deg, .. } => {
                assert!((theta_n_deg - (h.value - 3.0)).abs() < 1e-9, "{theta_n_deg}")
            }
            k => panic!("expected a measured bell, got {k:?}"),
        }
        // …and rebuilding the curve from the case reproduces the edited wall.
        let again = ParametricEditor::new(&p);
        assert_eq!(again.polyline(), e.polyline());
    }

    /// **THE STEP 3 GATE, headless.** Sweep every handle of every preset over
    /// its full clamped range, and after each drag take the SAME round trip the
    /// app takes on commit — `apply_curve` into the case, then `nozzle_contour`
    /// back out — asserting the wall never falls back to the 15° cone and never
    /// stops being a bell.
    ///
    /// This is the assertion behind "the wall never vanishes mid-drag". The old
    /// code returned an error from the `N_z < Q_z < E_z` guard, `nozzle_contour`
    /// answered a rejected spec with the fallback cone, and the user's nozzle
    /// was replaced by a cone for as long as the cursor stayed there.
    #[test]
    fn no_reachable_handle_position_falls_back_to_the_cone() {
        let mut swept = 0;
        for pre in PRESETS.iter() {
            let base = pre.case(0.0, false);
            let n = ParametricEditor::new(&base).handles().len();
            for i in 0..n {
                if !ParametricEditor::new(&base).handles()[i].pickable {
                    continue;
                }
                for k in 0..12 {
                    let ang = std::f64::consts::TAU * k as f64 / 12.0;
                    for step in 1..=9 {
                        let d = 0.08 * (1.7f64).powi(step);
                        let mut e = ParametricEditor::new(&base);
                        let a = e.handles()[i].anchor;
                        e.drag(i, [a[0] + d * ang.cos(), (a[1] + d * ang.sin()).max(0.0)]);
                        let mut p = base;
                        crate::case::apply_curve(&mut p, e.curve());
                        let w = crate::case::nozzle_contour(&p);
                        let at = format!("{} handle {i} dir {k} step {step}", pre.name);
                        assert_eq!(w.fallback, None, "{at}: fell back to the cone");
                        assert!(w.kind.is_bell(), "{at}: stopped being a bell ({:?})", w.kind);
                        assert!(w.points.len() > 20, "{at}: {} points", w.points.len());
                        // The committed polyline is what the editor showed.
                        assert_eq!(w.points, e.polyline(), "{at}: commit changed the wall");
                        swept += 1;
                    }
                }
            }
        }
        assert!(swept > 3000, "only {swept} positions swept");
    }

    /// Ctrl+click and right-click are refused, which is the signal the app uses
    /// to offer the one-way break instead of silently doing nothing.
    #[test]
    fn point_insert_and_remove_are_refused_on_a_parametric_wall() {
        let mut e = ParametricEditor::new(&CaseParams::default());
        assert!(!e.insert([5.0, 1.4]));
        assert!(!e.remove(0));
        assert_eq!(e.handles().len(), 6);
    }
}
