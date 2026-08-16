//! CAD handles for a parametric nozzle wall.
//!
//! Nine markers for a bell, six for a cone — one per editable degree of
//! freedom, instead of one per tessellated vertex. That is the whole point of
//! the curve split: the polyline the solver eats has 63–100 points and none of
//! them is draggable.
//!
//! **No pixels here.** This module returns anchors in r_t and tangent
//! DIRECTIONS; `cfd-ui` decides how many pixels along a direction the dot goes.
//! The day `cfd-geom` knows the size of a screen pixel is the day it grows an
//! egui dependency.
//!
//! **The folded frame.** Every world coordinate in and out of this module has
//! `r >= 0`. The canvas mirrors the half-plane about the axis, so a cursor on
//! the lower copy arrives with negative r; an angle computed from
//! cursor-minus-anchor there comes out INVERTED. Folding is the caller's job
//! and it must happen before the call, not after.

use crate::curve::NozzleCurve;
use crate::ContourKind;
use cfd_contract::Result;

/// The editable degrees of freedom of a parametric wall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HandleId {
    /// Chamber radius -> contraction ratio.
    ChamberRadius,
    /// Where the chamber straight ends and the converging cone starts.
    ChamberEnd,
    /// Converging half-angle beta.
    ConvergeAngle,
    /// Upstream throat-arc radius R1.
    ThroatArcUp,
    /// Downstream throat-arc radius R2.
    ThroatArcDown,
    /// Wall angle at the throat-arc / Bezier tangency, theta_n.
    ThetaN,
    /// Wall angle at the exit lip, theta_e.
    ThetaE,
    /// The exit lip: radially the area ratio, axially the bell length (or, on a
    /// cone, the half-angle).
    ExitLip,
    /// The Bezier control point Q. **Derived, never draggable** — see
    /// [`Handle::pickable`].
    ControlPoint,
}

impl HandleId {
    pub fn label(self) -> &'static str {
        match self {
            HandleId::ChamberRadius => "chamber radius",
            HandleId::ChamberEnd => "chamber end",
            HandleId::ConvergeAngle => "converge half-angle",
            HandleId::ThroatArcUp => "throat arc R1",
            HandleId::ThroatArcDown => "throat arc R2",
            HandleId::ThetaN => "theta_n",
            HandleId::ThetaE => "theta_e",
            HandleId::ExitLip => "exit lip",
            HandleId::ControlPoint => "control point Q",
        }
    }
}

/// How a handle is drawn and picked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandleKind {
    /// Drawn at its anchor; dragged in (z, r).
    Point,
    /// Drawn a fixed number of PIXELS from its anchor along `dir` (a unit
    /// vector in world units), and dragged by angle about the anchor. A wall
    /// angle has no natural length, so an angle handle cannot live at a world
    /// distance without changing size as the nozzle does.
    Tangent { dir: [f64; 2] },
    /// Drawn, never picked.
    Derived,
}

/// One CAD handle: where it attaches, how it is drawn, and what number it
/// currently reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Handle {
    pub id: HandleId,
    pub kind: HandleKind,
    /// Attachment point on the wall, r_t, in the folded (r >= 0) frame.
    pub anchor: [f64; 2],
    /// The parameter's current value, for the sidebar readout.
    pub value: f64,
    pub unit: &'static str,
    /// False for [`HandleId::ControlPoint`] and nothing else.
    ///
    /// Q is the intersection of the two tangent lines: it has ZERO remaining
    /// degrees of freedom once theta_n, theta_e, N and E are fixed. Making it
    /// draggable would mean breaking the G1 tangency at N, and theta_n is only
    /// physically meaningful because of that tangency — a bell whose wall does
    /// not leave the throat arc at theta_n is not a bell with that theta_n, it
    /// is a wall with a kink.
    pub pickable: bool,
}

/// The outcome of a handle drag: always a VALID curve, plus the reason it could
/// not go further when the request was infeasible.
#[derive(Debug, Clone, PartialEq)]
pub struct DragOutcome {
    pub curve: NozzleCurve,
    /// `Some` when the drag was clamped short of the requested position. The
    /// text is the generator's own rejection message, which already says which
    /// direction is infeasible ("too long" / "too short" for the area ratio);
    /// the UI shows it as the clamp tooltip.
    pub clamped: Option<String>,
}

impl NozzleCurve {
    /// The handles for this wall: nine for a bell, six for a cone.
    pub fn handles(&self) -> Result<Vec<Handle>> {
        self.validate()?;
        let s = self.spec;
        let rc = s.contraction_ratio.sqrt();
        let beta = s.converge_half_angle_deg.to_radians();
        let (r1, r2) = (s.throat_arc_up, s.throat_arc_down);
        let z_t = self.throat_z_rt();
        let ra_up = 1.0 + r1 * (1.0 - beta.cos());
        let z_a = z_t - r1 * beta.sin();
        let cl = self.chamber_len_rt;

        let point = |id: HandleId, anchor: [f64; 2], value: f64, unit: &'static str| Handle {
            id,
            kind: HandleKind::Point,
            anchor,
            value,
            unit,
            pickable: true,
        };

        let mut v = vec![
            point(HandleId::ChamberRadius, [0.0, rc], s.contraction_ratio, "ε_c"),
            point(HandleId::ChamberEnd, [cl, rc], cl, "r_t"),
            // Anchored at the converging cone's midpoint, so it never collides
            // with the two chamber handles at its ends.
            point(
                HandleId::ConvergeAngle,
                [0.5 * (cl + z_a), 0.5 * (rc + ra_up)],
                s.converge_half_angle_deg,
                "°",
            ),
            // Arc handles sit at the arc's own mid-angle: the only place on the
            // arc whose radial position moves monotonically with R and is clear
            // of both tangency points.
            point(
                HandleId::ThroatArcUp,
                arc_point(z_t, r1, -1.0, 0.5 * beta),
                r1,
                "R_t",
            ),
        ];

        match s.contour {
            ContourKind::Conical { half_angle_deg } => {
                let alpha = half_angle_deg.to_radians();
                v.push(point(
                    HandleId::ThroatArcDown,
                    arc_point(z_t, r2, 1.0, 0.5 * alpha),
                    r2,
                    "R_t",
                ));
                v.push(point(
                    HandleId::ExitLip,
                    [z_t + self.divergent_len_rt()?, s.area_ratio.sqrt()],
                    s.area_ratio,
                    "ε",
                ));
            }
            _ => {
                let b = self.bell()?;
                let (tn, te) = (b.theta_n_deg.to_radians(), b.theta_e_deg.to_radians());
                let n = [z_t + b.n[0], b.n[1]];
                let e = [z_t + b.e[0], b.e[1]];
                v.push(point(
                    HandleId::ThroatArcDown,
                    arc_point(z_t, r2, 1.0, 0.5 * tn),
                    r2,
                    "R_t",
                ));
                // theta_n's dot goes DOWNSTREAM along the wall tangent at N,
                // theta_e's goes UPSTREAM along the tangent at E — otherwise it
                // would float off the end of the nozzle into open plume.
                v.push(Handle {
                    id: HandleId::ThetaN,
                    kind: HandleKind::Tangent {
                        dir: [tn.cos(), tn.sin()],
                    },
                    anchor: n,
                    value: b.theta_n_deg,
                    unit: "°",
                    pickable: true,
                });
                v.push(Handle {
                    id: HandleId::ThetaE,
                    kind: HandleKind::Tangent {
                        dir: [-te.cos(), -te.sin()],
                    },
                    anchor: e,
                    value: b.theta_e_deg,
                    unit: "°",
                    pickable: true,
                });
                v.push(point(HandleId::ExitLip, e, s.area_ratio, "ε"));
                v.push(Handle {
                    id: HandleId::ControlPoint,
                    kind: HandleKind::Derived,
                    anchor: [z_t + b.q[0], b.q[1]],
                    value: b.q[0],
                    unit: "r_t",
                    pickable: false,
                });
            }
        }
        Ok(v)
    }

    /// The dashed N–Q–E control polygon, r_t, folded frame. `None` for a cone.
    pub fn control_polygon(&self) -> Option<[[f64; 2]; 3]> {
        let b = self.bell().ok()?;
        let z_t = self.throat_z_rt();
        Some([
            [z_t + b.n[0], b.n[1]],
            [z_t + b.q[0], b.q[1]],
            [z_t + b.e[0], b.e[1]],
        ])
    }

    /// Drag `id` toward `world` (r_t, **folded frame**), clamping to the
    /// furthest position that still produces a valid wall.
    ///
    /// **Why a clamp and not an error.** The `N_z < Q_z < E_z` guard used to
    /// reject the whole spec, and the app's response to a rejected spec is to
    /// fall back to the 15° cone — so mid-drag the user's nozzle would VANISH
    /// and be replaced by a cone, then come back when they dragged the other
    /// way. Bisecting to the boundary instead means the wall follows the cursor
    /// until it physically cannot, and then stops, which is what every CAD
    /// package does and what a constraint is supposed to feel like.
    pub fn drag_handle(&self, id: HandleId, world: [f64; 2]) -> DragOutcome {
        let Some(want) = self.param_from_world(id, world) else {
            return DragOutcome {
                curve: *self,
                clamped: None,
            };
        };
        let have = self.handle_param(id);
        let at = |t: f64| {
            self.with_param(
                id,
                [
                    have[0] + t * (want[0] - have[0]),
                    have[1] + t * (want[1] - have[1]),
                ],
            )
        };
        let full = at(1.0);
        match full.feasible() {
            Ok(()) => DragOutcome {
                curve: full,
                clamped: None,
            },
            Err(why) => {
                // The current curve is feasible by construction, the requested
                // one is not: bisect for the boundary. 40 halvings resolve the
                // parameter to 1e-12 of the drag distance, far below a pixel.
                let (mut lo, mut hi) = (0.0f64, 1.0f64);
                let mut best = *self;
                for _ in 0..40 {
                    let mid = 0.5 * (lo + hi);
                    let c = at(mid);
                    if c.feasible().is_ok() {
                        best = c;
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                DragOutcome {
                    curve: best,
                    clamped: Some(why.to_string()),
                }
            }
        }
    }

    /// Everything a wall must satisfy to be editable AND solvable: a valid
    /// spec, a closing bell, pieces that advance in z, and a wall that falls to
    /// the throat and rises after it.
    ///
    /// Stricter than `validate()` on purpose. `validate()` guards the spec;
    /// this guards the SHAPE, which is what the drag clamp has to bisect
    /// against — a curve that validates but doubles back is still a wall the
    /// rasterizer and the solver cannot use.
    pub fn feasible(&self) -> Result<()> {
        let pieces = self.pieces()?;
        for (i, p) in pieces.iter().enumerate() {
            let (a, b) = (p.start(), p.end());
            if !(b[0] > a[0]) {
                return Err(cfd_contract::CfdError::Geometry(format!(
                    "wall piece {i} does not advance in z ({} -> {})",
                    a[0], b[0]
                )));
            }
            // Upstream of the throat (pieces 0..=2) the wall may only fall;
            // downstream it may only rise.
            let falling = i <= 2;
            if falling && b[1] > a[1] + 1e-12 {
                return Err(cfd_contract::CfdError::Geometry(format!(
                    "converging wall piece {i} widens ({} -> {})",
                    a[1], b[1]
                )));
            }
            if !falling && b[1] < a[1] - 1e-12 {
                return Err(cfd_contract::CfdError::Geometry(format!(
                    "diverging wall piece {i} narrows ({} -> {})",
                    a[1], b[1]
                )));
            }
        }
        Ok(())
    }

    /// The current value of the parameter(s) `id` controls. The second slot is
    /// unused (and zero) for the seven single-DOF handles.
    fn handle_param(&self, id: HandleId) -> [f64; 2] {
        let s = self.spec;
        match id {
            HandleId::ChamberRadius => [s.contraction_ratio, 0.0],
            HandleId::ChamberEnd => [self.chamber_len_rt, 0.0],
            HandleId::ConvergeAngle => [s.converge_half_angle_deg, 0.0],
            HandleId::ThroatArcUp => [s.throat_arc_up, 0.0],
            HandleId::ThroatArcDown => [s.throat_arc_down, 0.0],
            HandleId::ThetaN => [self.bell().map(|b| b.theta_n_deg).unwrap_or(0.0), 0.0],
            HandleId::ThetaE => [self.bell().map(|b| b.theta_e_deg).unwrap_or(0.0), 0.0],
            HandleId::ExitLip => match s.contour {
                ContourKind::Conical { half_angle_deg } => [s.area_ratio, half_angle_deg],
                ContourKind::ParabolicBell { bell_percent } => [s.area_ratio, bell_percent],
                ContourKind::DirectBell {
                    length_fraction, ..
                } => [s.area_ratio, length_fraction],
            },
            HandleId::ControlPoint => [0.0, 0.0],
        }
    }

    /// The parameter(s) a cursor at `world` (r_t, folded) is asking for.
    /// `None` for the non-pickable control point.
    fn param_from_world(&self, id: HandleId, world: [f64; 2]) -> Option<[f64; 2]> {
        let s = self.spec;
        let (z, r) = (world[0], world[1].abs());
        let beta = s.converge_half_angle_deg.to_radians();
        let z_t = self.throat_z_rt();
        Some(match id {
            // r_c = sqrt(contraction ratio), so the cursor's radius squared IS
            // the parameter.
            HandleId::ChamberRadius => [(r * r).max(1.0 + 1e-9), 0.0],
            HandleId::ChamberEnd => [z.max(1e-6), 0.0],
            // Angle of the chamber-end -> cursor chord, in the folded frame.
            HandleId::ConvergeAngle => {
                let dz = z - self.chamber_len_rt;
                let dr = s.contraction_ratio.sqrt() - r;
                [dr.atan2(dz.max(1e-9)).to_degrees().clamp(0.1, 89.9), 0.0]
            }
            // The handle sits at the arc's mid-angle, where
            // r = 1 + R(1 - cos(phi_mid)): invert for R.
            HandleId::ThroatArcUp => {
                let k = 1.0 - (0.5 * beta).cos();
                [((r - 1.0) / k.max(1e-12)).max(1e-6), 0.0]
            }
            HandleId::ThroatArcDown => {
                let phi = 0.5 * self.divergent_angles_deg().ok()?.0.to_radians();
                let k = 1.0 - phi.cos();
                [((r - 1.0) / k.max(1e-12)).max(1e-6), 0.0]
            }
            // Angle of the anchor -> cursor chord. N is DOWNSTREAM-facing, E is
            // upstream-facing, matching where each dot is drawn.
            HandleId::ThetaN => {
                let b = self.bell().ok()?;
                let (nz, nr) = (z_t + b.n[0], b.n[1]);
                [
                    (r - nr)
                        .atan2((z - nz).max(1e-9))
                        .to_degrees()
                        .clamp(0.2, 89.0),
                    0.0,
                ]
            }
            HandleId::ThetaE => {
                let b = self.bell().ok()?;
                let (ez, er) = (z_t + b.e[0], b.e[1]);
                [
                    (er - r)
                        .atan2((ez - z).max(1e-9))
                        .to_degrees()
                        .clamp(0.1, 89.0),
                    0.0,
                ]
            }
            HandleId::ExitLip => {
                let eps = (r * r).max(1.0 + 1e-9);
                let l_n = (z - z_t).max(1e-6);
                match s.contour {
                    // On a cone the exit STATION is the half-angle: invert
                    // L_n(alpha) numerically, since it is monotone decreasing
                    // in alpha and has no closed-form inverse.
                    ContourKind::Conical { .. } => [eps, cone_angle_for_length(eps, s.throat_arc_down, l_n)],
                    // On a bell it is the length fraction, directly.
                    _ => [eps, l_n / crate::curve::l_c15(eps)],
                }
            }
            HandleId::ControlPoint => return None,
        })
    }

    /// The curve this parameter value produces. May be infeasible; the caller
    /// bisects.
    fn with_param(&self, id: HandleId, p: [f64; 2]) -> NozzleCurve {
        let mut c = *self;
        match id {
            HandleId::ChamberRadius => c.spec.contraction_ratio = p[0],
            HandleId::ChamberEnd => c.chamber_len_rt = p[0],
            HandleId::ConvergeAngle => c.spec.converge_half_angle_deg = p[0],
            HandleId::ThroatArcUp => c.spec.throat_arc_up = p[0],
            HandleId::ThroatArcDown => c.spec.throat_arc_down = p[0],
            // A theta drag takes the wall OFF the Rao table: the table maps
            // (area ratio, bell percent) to a pair of angles, and a wall with a
            // hand-set theta_n is no longer that pair. It becomes a measured
            // bell — the same contour kind Raptor 2 already flies — keeping the
            // other angle and the length.
            HandleId::ThetaN | HandleId::ThetaE => {
                if let Ok(b) = self.bell() {
                    let (tn, te) = if id == HandleId::ThetaN {
                        (p[0], b.theta_e_deg)
                    } else {
                        (b.theta_n_deg, p[0])
                    };
                    c.spec.contour = ContourKind::DirectBell {
                        theta_n_deg: tn,
                        theta_e_deg: te,
                        length_fraction: self.length_fraction(),
                    };
                }
            }
            HandleId::ExitLip => {
                c.spec.area_ratio = p[0];
                c.spec.contour = match self.spec.contour {
                    ContourKind::Conical { .. } => ContourKind::Conical {
                        half_angle_deg: p[1],
                    },
                    ContourKind::ParabolicBell { .. } => ContourKind::ParabolicBell {
                        bell_percent: p[1],
                    },
                    ContourKind::DirectBell {
                        theta_n_deg,
                        theta_e_deg,
                        ..
                    } => ContourKind::DirectBell {
                        theta_n_deg,
                        theta_e_deg,
                        length_fraction: p[1],
                    },
                };
            }
            HandleId::ControlPoint => {}
        }
        c
    }

    /// Divergent length as a fraction of the H&H reference cone, whichever
    /// contour this is.
    pub fn length_fraction(&self) -> f64 {
        match self.spec.contour {
            ContourKind::ParabolicBell { bell_percent } => bell_percent,
            ContourKind::DirectBell {
                length_fraction, ..
            } => length_fraction,
            ContourKind::Conical { .. } => self
                .divergent_len_rt()
                .map(|l| l / crate::curve::l_c15(self.spec.area_ratio))
                .unwrap_or(1.0),
        }
    }
}

/// A point on a throat arc at local wall angle `phi`, r_t.
fn arc_point(z_t: f64, radius: f64, sign: f64, phi: f64) -> [f64; 2] {
    [
        z_t + sign * radius * phi.sin(),
        1.0 + radius * (1.0 - phi.cos()),
    ]
}

/// The cone half-angle whose divergent length is `l_n` at this area ratio and
/// downstream arc. `L_n(alpha)` is strictly decreasing on (0, 90°), so bisect.
fn cone_angle_for_length(area_ratio: f64, r2: f64, l_n: f64) -> f64 {
    let len = |deg: f64| {
        let a = deg.to_radians();
        ((area_ratio.sqrt() - 1.0) + r2 * (1.0 / a.cos() - 1.0)) / a.tan()
    };
    let (mut lo, mut hi) = (0.2f64, 89.0f64);
    if l_n >= len(lo) {
        return lo;
    }
    if l_n <= len(hi) {
        return hi;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if len(mid) > l_n {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Tessellation;
    use crate::NozzleSpec;

    fn spec(eps: f64, rt: f64, arc_dn: f64, contour: ContourKind) -> NozzleSpec {
        NozzleSpec {
            throat_radius_m: rt,
            area_ratio: eps,
            contraction_ratio: 4.0,
            converge_half_angle_deg: 30.0,
            throat_arc_up: 1.5,
            throat_arc_down: arc_dn,
            contour,
        }
    }

    fn walls() -> Vec<(&'static str, NozzleCurve)> {
        vec![
            (
                "demo cone",
                NozzleCurve::new(spec(
                    8.0,
                    0.05,
                    0.382,
                    ContourKind::Conical {
                        half_angle_deg: 15.0,
                    },
                )),
            ),
            (
                "RS-25 table bell",
                NozzleCurve::new(spec(
                    69.0,
                    0.138,
                    0.382,
                    ContourKind::ParabolicBell { bell_percent: 0.80 },
                )),
            ),
            (
                "Raptor 2 measured bell",
                NozzleCurve::new(spec(
                    34.3,
                    0.115,
                    0.300,
                    ContourKind::DirectBell {
                        theta_n_deg: 32.0,
                        theta_e_deg: 6.0,
                        length_fraction: 0.76,
                    },
                )),
            ),
        ]
    }

    #[test]
    fn a_bell_has_nine_handles_and_a_cone_six() {
        let h = walls()[1].1.handles().unwrap();
        assert_eq!(h.len(), 9, "bell handles: {:?}", h.iter().map(|x| x.id).collect::<Vec<_>>());
        assert_eq!(h.iter().filter(|x| x.pickable).count(), 8);
        // Exactly one non-pickable marker, and it is Q.
        let q = h.iter().find(|x| !x.pickable).unwrap();
        assert_eq!(q.id, HandleId::ControlPoint);
        assert_eq!(q.kind, HandleKind::Derived);
        // Q lies strictly between N and E in z — the §10 ordering, drawn.
        let poly = walls()[1].1.control_polygon().unwrap();
        assert!(poly[0][0] < poly[1][0] && poly[1][0] < poly[2][0]);
        assert_eq!(poly[1], q.anchor);

        let c = walls()[0].1.handles().unwrap();
        assert_eq!(c.len(), 6, "cone handles: {:?}", c.iter().map(|x| x.id).collect::<Vec<_>>());
        assert!(c.iter().all(|x| x.pickable));
        assert!(walls()[0].1.control_polygon().is_none());
    }

    /// Every handle's anchor is ON the wall it is attached to (except Q, which
    /// is deliberately off it — that is what a control point is).
    #[test]
    fn handle_anchors_sit_on_the_wall() {
        for (name, c) in walls() {
            let rt = c.spec.throat_radius_m;
            for h in c.handles().unwrap() {
                if h.id == HandleId::ControlPoint {
                    continue;
                }
                let r = c.radius_at(h.anchor[0] * rt).unwrap() / rt;
                assert!(
                    (r - h.anchor[1]).abs() < 1e-9,
                    "{name}: {:?} anchor {:?} is {:.3e} r_t off the wall",
                    h.id,
                    h.anchor,
                    (r - h.anchor[1]).abs()
                );
            }
        }
    }

    /// Dragging a handle to exactly where it already is must be a no-op. If it
    /// is not, the world->parameter map and the anchor disagree and every drag
    /// starts with a jump.
    #[test]
    fn dragging_a_handle_to_its_own_anchor_changes_nothing() {
        for (name, c) in walls() {
            for h in c.handles().unwrap() {
                if !h.pickable {
                    continue;
                }
                // A tangent handle is picked at a pixel offset along its
                // direction, so "where it already is" means a point along dir.
                let at = match h.kind {
                    HandleKind::Tangent { dir } => {
                        [h.anchor[0] + 0.3 * dir[0], h.anchor[1] + 0.3 * dir[1]]
                    }
                    _ => h.anchor,
                };
                let out = c.drag_handle(h.id, at);
                assert_eq!(out.clamped, None, "{name}: {:?} clamped on a null drag", h.id);
                let after = out
                    .curve
                    .handles()
                    .unwrap()
                    .into_iter()
                    .find(|x| x.id == h.id)
                    .unwrap();
                assert!(
                    (after.value - h.value).abs() <= 1e-6 * h.value.abs().max(1.0),
                    "{name}: {:?} jumped {} -> {} on a null drag",
                    h.id,
                    h.value,
                    after.value
                );
            }
        }
    }

    /// **THE STEP 3 GATE.** Sweep every handle over its full clamped range in
    /// both directions and assert the wall is always solvable: `tessellate()`
    /// is Ok, z is strictly monotone, and the throat is a STRICT argmin in r.
    ///
    /// The old behaviour this replaces: the `N_z < Q_z < E_z` guard returned an
    /// error, the app fell back to the 15° cone, and the nozzle vanished
    /// mid-drag.
    #[test]
    fn every_handle_sweeps_its_clamped_range_without_producing_a_bad_wall() {
        let mut clamps = 0;
        for (name, c0) in walls() {
            let rt = c0.spec.throat_radius_m;
            let t = Tessellation::from_cell_size(0.145 * rt, 0.05 * rt);
            for h0 in c0.handles().unwrap() {
                if !h0.pickable {
                    continue;
                }
                // Push the handle far past anything reachable on screen, in
                // eight directions, at eleven magnitudes. Anything infeasible
                // must come back clamped, never broken.
                for k in 0..8 {
                    let ang = std::f64::consts::TAU * k as f64 / 8.0;
                    for step in 1..=11 {
                        let d = 0.05 * (1.6f64).powi(step);
                        let want = [
                            h0.anchor[0] + d * ang.cos(),
                            (h0.anchor[1] + d * ang.sin()).max(0.0),
                        ];
                        let out = c0.drag_handle(h0.id, want);
                        clamps += out.clamped.is_some() as usize;
                        let at = format!("{name}/{:?} dir {k} step {step}", h0.id);
                        let p = out
                            .curve
                            .tessellate(&t)
                            .unwrap_or_else(|e| panic!("{at}: tessellate failed: {e}"));
                        for w in p.points.windows(2) {
                            assert!(w[1][0] > w[0][0], "{at}: z not strictly increasing");
                        }
                        let rmin = p.points.iter().map(|q| q[1]).fold(f64::MAX, f64::min);
                        assert_eq!(
                            p.points.iter().filter(|q| q[1] <= rmin + 1e-15 * rt).count(),
                            1,
                            "{at}: throat is not a strict minimum"
                        );
                        assert!(
                            (p.points[p.throat_index][1] - rmin).abs() < 1e-15 * rt,
                            "{at}: throat_index does not point at argmin r"
                        );
                        // The throat radius is r_t by construction and must
                        // stay there: no handle may move it.
                        assert!(
                            (rmin / rt - 1.0).abs() < 1e-9,
                            "{at}: throat radius moved to {} r_t",
                            rmin / rt
                        );
                    }
                }
            }
        }
        // The sweep has to actually REACH infeasible territory, or it is
        // asserting that nothing bad happens while never going anywhere.
        assert!(clamps > 50, "only {clamps} drags were clamped — sweep too timid");
    }

    /// A theta drag takes a table bell off the table and onto measured angles,
    /// and the angle it lands on is the one that was asked for.
    #[test]
    fn a_theta_drag_converts_a_table_bell_to_a_measured_one() {
        let c = walls()[1].1; // RS-25, ParabolicBell
        let (tn0, te0) = c.divergent_angles_deg().unwrap();
        let h = c
            .handles()
            .unwrap()
            .into_iter()
            .find(|x| x.id == HandleId::ThetaN)
            .unwrap();
        // Ask for theta_n two degrees shallower.
        let want = (tn0 - 2.0).to_radians();
        let out = c.drag_handle(
            HandleId::ThetaN,
            [h.anchor[0] + want.cos(), h.anchor[1] + want.sin()],
        );
        assert_eq!(out.clamped, None);
        let (tn1, te1) = out.curve.divergent_angles_deg().unwrap();
        assert!((tn1 - (tn0 - 2.0)).abs() < 1e-9, "theta_n {tn0} -> {tn1}");
        assert!((te1 - te0).abs() < 1e-12, "theta_e must not move: {te0} -> {te1}");
        assert!(matches!(
            out.curve.spec.contour,
            ContourKind::DirectBell { .. }
        ));
        // The length is carried across unchanged.
        assert!((out.curve.length_fraction() - 0.80).abs() < 1e-12);
    }

    /// The clamp message is the generator's own, and it says which way to move.
    #[test]
    fn the_clamp_reports_the_infeasible_direction() {
        let c = walls()[2].1; // Raptor 2 measured bell
        let h = c
            .handles()
            .unwrap()
            .into_iter()
            .find(|x| x.id == HandleId::ExitLip)
            .unwrap();
        // Drag the lip far upstream: the bell becomes far too short for its
        // area ratio and Q leaves [N, E].
        let out = c.drag_handle(HandleId::ExitLip, [h.anchor[0] * 0.55, h.anchor[1]]);
        let why = out.clamped.expect("dragging the lip to the throat must clamp");
        assert!(
            why.contains("too short") || why.contains("does not advance"),
            "clamp message must name the direction: {why}"
        );
        assert!(out.curve.feasible().is_ok());
        // …and it stopped SHORT of the request, not at it.
        assert!(out.curve.divergent_len_rt().unwrap() < c.divergent_len_rt().unwrap());
    }

    /// The cone's exit lip carries the half-angle in z and the area ratio in r,
    /// and both round-trip.
    #[test]
    fn the_cone_exit_lip_carries_the_half_angle_and_the_area_ratio() {
        let c = walls()[0].1;
        let z_t = c.throat_z_rt();
        // Ask for a 20 deg cone at eps 12.
        let eps = 12.0f64;
        let alpha = 20.0f64.to_radians();
        let l_n = ((eps.sqrt() - 1.0) + 0.382 * (1.0 / alpha.cos() - 1.0)) / alpha.tan();
        let out = c.drag_handle(HandleId::ExitLip, [z_t + l_n, eps.sqrt()]);
        assert_eq!(out.clamped, None);
        assert!((out.curve.spec.area_ratio - eps).abs() < 1e-9);
        assert!(
            (out.curve.exit_angle_deg().unwrap() - 20.0).abs() < 1e-6,
            "half angle {}",
            out.curve.exit_angle_deg().unwrap()
        );
    }
}
