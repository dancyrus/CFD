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

use cfd_geom::{FreeformBounds, FreeformEditor, HandleKind, NozzleCurve, Tessellation};

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
/// Two implementations: [`ParametricEditor`] (handles on a curve) and
/// [`FreeformWall`] (every point a handle), reached by the one-way break.
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

    /// **THE STEP 4 GATE, part 1.** The break keeps the shape EXACTLY, is
    /// one-way, and is the only thing that raises the sandbox badge.
    #[test]
    fn the_break_is_one_way_and_the_shape_does_not_move() {
        for pre in PRESETS.iter() {
            let p = pre.case(0.0, false);
            let mut w = WallState::parametric(&p);
            let before = w.editor().polyline().to_vec();
            assert!(!w.is_freeform(), "{}: starts parametric", pre.name);

            // A handle drag does NOT raise the badge — that is the provenance
            // rule: a hand-tuned bell is still a member of the contour family.
            let n = w.editor().handles().len();
            for i in 0..n {
                let a = w.editor().handles()[i].anchor;
                if w.editor().handles()[i].pickable {
                    w.editor_mut().drag(i, [a[0] + 0.4, a[1] + 0.4]);
                }
                assert!(!w.is_freeform(), "{}: a drag raised the sandbox badge", pre.name);
            }

            // The break preserves the polyline it was given, vertex for vertex.
            let at_break = w.editor().polyline().to_vec();
            assert!(w.break_to_freeform(p.lz_rt, p.lr_rt));
            assert!(w.is_freeform());
            assert_eq!(
                w.editor().polyline(),
                at_break,
                "{}: the shape moved on conversion",
                pre.name
            );
            // One-way: a second break is a no-op, and there is no way back
            // except regenerating (which is a different wall by definition).
            assert!(!w.break_to_freeform(p.lz_rt, p.lr_rt));
            assert!(w.is_freeform());
            // Freeform has no handles-vs-points distinction and no control
            // polygon: every point is a control point again, deliberately.
            assert_eq!(w.editor().handles().len(), at_break.len());
            assert!(w.editor().control_polygon().is_none());
            assert!(w.editor_mut().insert([before[1][0] + 0.01, 1.5]));
            assert!(w.editor().polyline().len() == at_break.len() + 1);
        }
    }

    /// **THE STEP 4 GATE, part 2.** Regenerating from a preset restores a
    /// parametric wall — the one escape hatch, and the reason the app asks
    /// before taking it.
    #[test]
    fn regenerating_restores_a_parametric_wall() {
        let p = PRESETS[2].case(0.0, false); // Raptor 2
        let mut w = WallState::parametric(&p);
        w.break_to_freeform(p.lz_rt, p.lr_rt);
        assert!(w.is_freeform());
        w = WallState::parametric(&p);
        assert!(!w.is_freeform());
        assert_eq!(w.editor().handles().len(), 9);
        assert!(w.editor().control_polygon().is_some());
        assert_eq!(w.editor().polyline(), crate::case::nozzle_contour(&p).points);
    }

    /// **THE STEP 4 GATE, part 3.** A hand-tuned bell stops showing the
    /// Rao-table clamp warning: its angles came from the user, not the table,
    /// so a warning about table extrapolation would describe a lookup that did
    /// not happen.
    #[test]
    fn a_hand_tuned_bell_stops_claiming_a_clamped_rao_table() {
        use crate::case::rao_clamp_warning;
        // Merlin Vac: eps 165, past the table's eps = 100 end, so the warning
        // fires while it is a table bell.
        let mut p = PRESETS[5].case(0.0, false);
        assert_eq!(p.contour_kind, ContourKind::ParabolicBell);
        assert!(rao_clamp_warning(Some(p.contour_kind), p.area_ratio).is_some());

        // Drag theta_e; the wall becomes a measured bell and the warning goes.
        let mut e = ParametricEditor::new(&p);
        let i = e.handles().iter().position(|h| h.label == "theta_e").unwrap();
        let h = e.handles()[i];
        let want = (h.value + 1.0).to_radians();
        e.drag(i, [h.anchor[0] - want.cos(), h.anchor[1] - want.sin()]);
        crate::case::apply_curve(&mut p, e.curve());
        assert!(matches!(p.contour_kind, ContourKind::MeasuredBell { .. }));
        assert!(
            rao_clamp_warning(Some(p.contour_kind), p.area_ratio).is_none(),
            "a measured bell must not claim a clamped table lookup"
        );
        // A drawn wall has no contour kind at all, so it cannot either.
        assert!(rao_clamp_warning(None, p.area_ratio).is_none());
        // …and the warning still fires for a genuine table bell out of range.
        assert_eq!(
            rao_clamp_warning(Some(ContourKind::ParabolicBell), 2.0),
            Some((4.0, "starts at"))
        );
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

// ---------------------------------------------------------------------------
// Freeform
// ---------------------------------------------------------------------------

/// A drawn wall: `cfd_geom::FreeformEditor` behind the same trait, with one
/// handle per control point.
///
/// The polyline IS the model here — which is exactly the conflation the
/// parametric path exists to avoid, and exactly what is wanted once the user
/// has decided their shape is not a member of any contour family. The
/// difference is that it is now a deliberate MODE with a badge on it, not the
/// only representation there is.
pub struct FreeformWall {
    inner: FreeformEditor,
    handles: Vec<EditHandle>,
}

impl FreeformWall {
    /// Take over from a polyline. **The shape does not move**: these are
    /// exactly the points that were on screen when the break happened.
    pub fn new(points: Vec<[f64; 2]>, lz: f64, lr: f64) -> Self {
        let mut w = FreeformWall {
            inner: FreeformEditor::new(points, FreeformBounds::for_domain(lz, lr)),
            handles: Vec::new(),
        };
        w.rebuild();
        w
    }

    /// The validation gate, for the app to refuse to commit a broken wall.
    pub fn validate(&self) -> Result<(), String> {
        self.inner.to_profile().map(|_| ()).map_err(|e| e.to_string())
    }

    fn rebuild(&mut self) {
        self.handles = self
            .inner
            .points()
            .iter()
            .map(|p| EditHandle {
                anchor: *p,
                kind: HandleKind::Point,
                label: "point",
                value: p[1],
                unit: "r_t",
                pickable: true,
            })
            .collect();
    }
}

impl WallEditor for FreeformWall {
    fn polyline(&self) -> &[[f64; 2]] {
        self.inner.points()
    }

    fn handles(&self) -> &[EditHandle] {
        &self.handles
    }

    fn drag(&mut self, i: usize, folded: [f64; 2]) -> Option<String> {
        self.inner.drag(i, folded);
        self.rebuild();
        None // a drawn point has no parametric constraint to clamp against
    }

    fn insert(&mut self, folded: [f64; 2]) -> bool {
        self.inner.insert(folded);
        self.rebuild();
        true
    }

    fn remove(&mut self, i: usize) -> bool {
        let ok = self.inner.remove(i);
        self.rebuild();
        ok
    }

    fn control_polygon(&self) -> Option<Vec<[f64; 2]>> {
        None
    }

    fn set_domain(&mut self, lz: f64, lr: f64) {
        self.inner.set_bounds(FreeformBounds::for_domain(lz, lr));
    }
}

/// Which representation the wall is in.
///
/// **The break is ONE-WAY.** There is no fit-a-curve-to-a-polyline step and
/// there should not be: fitting is a research problem with no unique answer,
/// and a bad fit silently changes the user's shape while claiming to preserve
/// it. Regenerating from a preset or the area-ratio slider is the escape hatch
/// — and because that DISCARDS the drawing, the app asks first.
pub enum WallState {
    Parametric(ParametricEditor),
    Freeform(FreeformWall),
}

impl WallState {
    pub fn parametric(p: &CaseParams) -> Self {
        WallState::Parametric(ParametricEditor::new(p))
    }

    pub fn is_freeform(&self) -> bool {
        matches!(self, WallState::Freeform(_))
    }

    pub fn editor(&self) -> &dyn WallEditor {
        match self {
            WallState::Parametric(e) => e,
            WallState::Freeform(e) => e,
        }
    }

    pub fn editor_mut(&mut self) -> &mut dyn WallEditor {
        match self {
            WallState::Parametric(e) => e,
            WallState::Freeform(e) => e,
        }
    }

    /// Break the parametric link, keeping the wall EXACTLY where it is. A no-op
    /// if already freeform.
    pub fn break_to_freeform(&mut self, lz: f64, lr: f64) -> bool {
        match self {
            WallState::Freeform(_) => false,
            WallState::Parametric(e) => {
                *self = WallState::Freeform(FreeformWall::new(e.polyline().to_vec(), lz, lr));
                true
            }
        }
    }
}
