//! The nozzle wall as a PARAMETRIC CURVE — arcs and a quadratic Bézier — and
//! the tessellation that turns it into the dense polyline the solver eats.
//!
//! This is the split the whole editor rests on. Before it, the 256-point
//! polyline WAS the nozzle: it was simultaneously the solver's input and the
//! editor's control points, so the canvas painted a drag handle every ~0.1 r_t
//! and wall editing was unusable. The curve is now the model; the polyline is
//! a derived artifact that only the solver and the rasterizer ever see.
//!
//! Units: everything is constructed in r_t (throat radius = 1) and scaled to
//! SI metres on the way out, exactly as `generate_contour` always did. z = 0 is
//! the chamber head.

use crate::rao::rao_angles;
use crate::{ContourKind, NozzleSpec, WallProfile};
use cfd_contract::{CfdError, Result};

/// Chamber straight length, in units of r_t (session brief §1). The default of
/// `NozzleCurve::chamber_len_rt`.
pub const CHAMBER_LEN_RT: f64 = 2.0;

/// Downstream throat-arc radius of Huzel–Huang's REFERENCE 15° cone, in r_t.
///
/// This is a fixed property of the definition of "N% bell" — H&H measure bell
/// length against a 15° conical nozzle of the same throat area and area ratio
/// **with a 1.5 R_t downstream throat arc** — not a property of the nozzle
/// being generated. It happens to equal the family's `throat_arc_up`, but it
/// is deliberately NOT read from the spec: an engine with a tighter throat arc
/// (Raptor 2, 0.300 R_t) must still be measured against the same reference, or
/// "80% bell" means a different length for every spec.
const L_C15_REF_ARC_RT: f64 = 1.5;

/// Huzel–Huang reference length: the 15° cone this family measures bell
/// percent against, in r_t, measured from the throat.
pub(crate) fn l_c15(area_ratio: f64) -> f64 {
    let s15 = 15.0f64.to_radians();
    ((area_ratio.sqrt() - 1.0) + L_C15_REF_ARC_RT * (1.0 / s15.cos() - 1.0)) / s15.tan()
}

/// Append a point, dropping an exact-duplicate z (shared piece endpoints get
/// pushed twice).
fn push(pts: &mut Vec<[f64; 2]>, p: [f64; 2]) {
    if pts.last().is_none_or(|q| p[0] - q[0] > 1e-12) {
        pts.push(p);
    }
}

/// A parametric nozzle wall: a `NozzleSpec` plus the one shape parameter that
/// has no home on it.
///
/// **Why `chamber_len_rt` lives here and not on `NozzleSpec`.** Nine test files
/// across `cfd-core`, `cfd-geom` and `cfd-ui` construct `NozzleSpec` as a
/// struct literal with no `..rest`, so adding a field to it is a breaking
/// change that ripples into the acceptance ladder. The curve wrapper is the
/// place new shape parameters go.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NozzleCurve {
    pub spec: NozzleSpec,
    /// Chamber straight length in units of r_t.
    pub chamber_len_rt: f64,
}

impl NozzleCurve {
    /// The §10 family: this spec on the standard 2 r_t chamber straight.
    pub fn new(spec: NozzleSpec) -> Self {
        NozzleCurve {
            spec,
            chamber_len_rt: CHAMBER_LEN_RT,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        if !(self.chamber_len_rt.is_finite() && self.chamber_len_rt > 0.0) {
            return Err(CfdError::Geometry(format!(
                "chamber_len_rt = {} (must be > 0)",
                self.chamber_len_rt
            )));
        }
        Ok(())
    }

    /// Throat axial station in r_t, z measured from the chamber head. The
    /// upstream arc is parameterized by wall angle phi —
    /// `z(phi) = z_t - r1 sin phi`, `r(phi) = 1 + r1 (1 - cos phi)` for
    /// phi in [0, beta] — so tangency to the converging cone is automatic.
    pub fn throat_z_rt(&self) -> f64 {
        let beta = self.spec.converge_half_angle_deg.to_radians();
        let rc = self.spec.contraction_ratio.sqrt();
        let r1 = self.spec.throat_arc_up;
        let ra_up = 1.0 + r1 * (1.0 - beta.cos());
        self.chamber_len_rt + (rc - ra_up) / beta.tan() + r1 * beta.sin()
    }

    /// The wall as an ordered list of analytic pieces, in r_t with z measured
    /// from the chamber head. This is the curve; everything else in this file
    /// is either a query on it or a way of turning it into points.
    pub fn pieces(&self) -> Result<Vec<Piece>> {
        self.validate()?;
        let spec = &self.spec;
        let re = spec.area_ratio.sqrt();
        let rc = spec.contraction_ratio.sqrt();
        let beta = spec.converge_half_angle_deg.to_radians();
        let r1 = spec.throat_arc_up;
        let r2 = spec.throat_arc_down;
        let ra_up = 1.0 + r1 * (1.0 - beta.cos());
        let z_t = self.throat_z_rt();

        let mut v = vec![
            // 1. Chamber straight.
            Piece::Line {
                a: [0.0, rc],
                b: [self.chamber_len_rt, rc],
            },
            // 2. Converging cone.
            Piece::Line {
                a: [self.chamber_len_rt, rc],
                b: [z_t - r1 * beta.sin(), ra_up],
            },
            // 3. Upstream throat arc, wall angle beta -> 0, ending at (z_t, 1).
            Piece::ThroatArc {
                z_t,
                radius: r1,
                sign: -1.0,
                phi0: beta,
                phi1: 0.0,
            },
        ];
        // 4. Diverging section.
        match spec.contour {
            ContourKind::Conical { half_angle_deg } => {
                let alpha = half_angle_deg.to_radians();
                let l_n = ((re - 1.0) + r2 * (1.0 / alpha.cos() - 1.0)) / alpha.tan();
                v.push(Piece::ThroatArc {
                    z_t,
                    radius: r2,
                    sign: 1.0,
                    phi0: 0.0,
                    phi1: alpha,
                });
                v.push(Piece::Line {
                    a: [
                        z_t + r2 * alpha.sin(),
                        1.0 + r2 * (1.0 - alpha.cos()),
                    ],
                    b: [z_t + l_n, re],
                });
            }
            _ => {
                let b = self.bell()?;
                v.push(Piece::ThroatArc {
                    z_t,
                    radius: r2,
                    sign: 1.0,
                    phi0: 0.0,
                    phi1: b.theta_n_deg.to_radians(),
                });
                v.push(Piece::Bezier {
                    n: [z_t + b.n[0], b.n[1]],
                    q: [z_t + b.q[0], b.q[1]],
                    e: [z_t + b.e[0], b.e[1]],
                });
            }
        }
        Ok(v)
    }

    /// The bell's N/Q/E triple (z measured FROM THE THROAT, r_t). `Err` for a
    /// conical contour — a cone has no Bézier.
    pub fn bell(&self) -> Result<BellGeometry> {
        let spec = &self.spec;
        let (l_n, tn, te) = match spec.contour {
            ContourKind::Conical { .. } => {
                return Err(CfdError::Geometry(
                    "a conical contour has no Bezier control point".into(),
                ))
            }
            ContourKind::ParabolicBell { bell_percent } => {
                let (tn, te) = rao_angles(spec.area_ratio, bell_percent);
                (bell_percent * l_c15(spec.area_ratio), tn, te)
            }
            ContourKind::DirectBell {
                theta_n_deg,
                theta_e_deg,
                length_fraction,
            } => (
                length_fraction * l_c15(spec.area_ratio),
                theta_n_deg,
                theta_e_deg,
            ),
        };
        BellGeometry::solve(
            spec.throat_arc_down,
            spec.area_ratio.sqrt(),
            l_n,
            tn,
            te,
        )
    }

    /// `(theta_n, theta_e)` in degrees, read off the CURVE — the exact wall
    /// angles the analytic contour meets, with no tessellation bias whatsoever.
    /// A cone reports its half-angle for both: its steepest wall angle and its
    /// exit angle are the same number.
    pub fn divergent_angles_deg(&self) -> Result<(f64, f64)> {
        match self.spec.contour {
            ContourKind::Conical { half_angle_deg } => {
                self.validate()?;
                Ok((half_angle_deg, half_angle_deg))
            }
            _ => {
                let b = self.bell()?;
                Ok((b.theta_n_deg, b.theta_e_deg))
            }
        }
    }

    /// Exit wall angle in degrees, analytic.
    pub fn exit_angle_deg(&self) -> Result<f64> {
        Ok(self.divergent_angles_deg()?.1)
    }

    /// Steepest downstream wall angle in degrees, analytic (theta_n for a bell).
    pub fn max_wall_angle_deg(&self) -> Result<f64> {
        Ok(self.divergent_angles_deg()?.0)
    }

    /// Divergent length in r_t, throat to exit plane.
    pub fn divergent_len_rt(&self) -> Result<f64> {
        match self.spec.contour {
            ContourKind::Conical { half_angle_deg } => {
                self.validate()?;
                let alpha = half_angle_deg.to_radians();
                let r2 = self.spec.throat_arc_down;
                Ok(((self.spec.area_ratio.sqrt() - 1.0) + r2 * (1.0 / alpha.cos() - 1.0))
                    / alpha.tan())
            }
            _ => Ok(self.bell()?.e[0]),
        }
    }

    /// Total axial extent in metres, chamber head to exit plane.
    pub fn length_m(&self) -> Result<f64> {
        Ok((self.throat_z_rt() + self.divergent_len_rt()?) * self.spec.throat_radius_m)
    }

    /// **The analytic wall radius at axial station `z`** (both SI metres),
    /// `None` outside the curve's extent.
    ///
    /// This is the query the polyline used to stand in for. It has no
    /// tessellation error at all, which is why the equivalence and angle tests
    /// assert against it rather than against chord slopes of whatever sample
    /// grid the tessellator happened to pick.
    pub fn radius_at(&self, z: f64) -> Option<f64> {
        let rt = self.spec.throat_radius_m;
        let zr = z / rt;
        let pieces = self.pieces().ok()?;
        let (z0, z1) = (pieces[0].start()[0], pieces[pieces.len() - 1].end()[0]);
        if !(zr >= z0 - 1e-12 && zr <= z1 + 1e-12) {
            return None;
        }
        let zc = zr.clamp(z0, z1);
        for p in &pieces {
            if zc <= p.end()[0] + 1e-12 {
                return Some(p.radius_at(zc)? * rt);
            }
        }
        Some(pieces[pieces.len() - 1].end()[1] * rt)
    }

    /// Unit wall tangent (dz, dr) at axial station `z` (metres), pointing
    /// downstream. `None` outside the curve.
    ///
    /// Returned as a DIRECTION, never as pixels: `cfd-geom` has no idea how big
    /// a screen pixel is, and the day it does is the day this crate grows an
    /// egui dependency. `cfd-ui` places its tangent dots at a fixed pixel
    /// offset along this vector.
    pub fn tangent_at(&self, z: f64) -> Option<[f64; 2]> {
        let rt = self.spec.throat_radius_m;
        let zr = z / rt;
        let pieces = self.pieces().ok()?;
        let (z0, z1) = (pieces[0].start()[0], pieces[pieces.len() - 1].end()[0]);
        let zc = zr.clamp(z0, z1);
        for p in &pieces {
            if zc <= p.end()[0] + 1e-12 {
                return Some(p.tangent_at(zc));
            }
        }
        pieces.last().map(|p| p.tangent_at(z1))
    }

    /// **Adaptive tessellation**: the polyline the solver and the rasterizer
    /// eat, sized to the mesh instead of to a magic constant.
    ///
    /// Every curved piece is split until its chord sagitta is under
    /// `t.chord_tol_m` AND no segment turns by more than `t.max_turn_deg`, with
    /// a floor of 2 segments per curved piece (one segment cannot represent
    /// curvature at all) and a global cap of `t.max_points`. Straight pieces
    /// are one segment, exactly, because they are exact.
    pub fn tessellate(&self, t: &Tessellation) -> Result<WallProfile> {
        let rt = self.spec.throat_radius_m;
        let pieces = self.pieces()?;
        let tol = (t.chord_tol_m / rt).max(f64::MIN_POSITIVE);

        // Per-piece segment counts, then one proportional shrink if the total
        // would blow the cap. Shrinking is a LAST resort and it is honest: the
        // caller asked for a tolerance and gets told (through the point count)
        // that the cap bound it instead.
        let mut counts: Vec<usize> = pieces
            .iter()
            .map(|p| p.segments(tol, t.max_turn_deg))
            .collect();
        let floors: Vec<usize> = pieces.iter().map(Piece::min_segments).collect();
        // n segments per piece plus the opening vertex.
        let cap = t.max_points.saturating_sub(1).max(floors.iter().sum());
        let mut total: usize = counts.iter().sum();
        if total > cap {
            let scale = cap as f64 / total as f64;
            for ((c, &f), _) in counts.iter_mut().zip(&floors).zip(&pieces) {
                *c = ((*c as f64 * scale).floor() as usize).max(f);
            }
            total = counts.iter().sum();
            // Flooring each piece at its own minimum can still overshoot, so
            // shave the largest piece until it fits. Terminates: the floors sum
            // to at most 2 per piece and `cap` is at least that.
            while total > cap {
                let Some(i) = (0..counts.len())
                    .filter(|&i| counts[i] > floors[i])
                    .max_by_key(|&i| counts[i])
                else {
                    break;
                };
                counts[i] -= 1;
                total -= 1;
            }
        }

        let mut pts: Vec<[f64; 2]> = Vec::with_capacity(counts.iter().sum::<usize>() + 2);
        push(&mut pts, pieces[0].start());
        let mut throat_index = 0;
        for (i, (p, &n)) in pieces.iter().zip(&counts).enumerate() {
            for k in 1..=n {
                push(&mut pts, p.point_at(k as f64 / n as f64));
            }
            // The upstream throat arc is piece 2; its end IS the throat.
            if i == 2 {
                throat_index = pts.len() - 1;
            }
        }
        for p in &mut pts {
            p[0] *= rt;
            p[1] *= rt;
        }
        let profile = WallProfile {
            points: pts,
            throat_index,
        };
        profile.validate()?;
        Ok(profile)
    }

    /// Tessellate at a fixed sample budget — the ORIGINAL `generate_contour`
    /// loops, unchanged, so a caller that asks for `samples` points gets the
    /// polyline this build has always produced, bit for bit.
    ///
    /// `cfd-core`'s acceptance ladder and diagnostics call this path through
    /// `generate_contour`; their recorded results are only comparable across
    /// time while it stays bit-identical. New callers want
    /// [`NozzleCurve::tessellate`] instead, which sizes itself from the mesh.
    pub fn tessellate_fixed(&self, samples: usize) -> Result<WallProfile> {
        self.validate()?;

        let spec = &self.spec;
        let rt = spec.throat_radius_m;
        let eps = spec.area_ratio;
        let re = eps.sqrt(); // exit radius, r_t units
        let rc = spec.contraction_ratio.sqrt(); // chamber radius, r_t units
        let beta = spec.converge_half_angle_deg.to_radians();
        let r1 = spec.throat_arc_up;
        let r2 = spec.throat_arc_down;

        let ra_up = 1.0 + r1 * (1.0 - beta.cos());
        let z_t = self.chamber_len_rt + (rc - ra_up) / beta.tan() + r1 * beta.sin();

        let n = samples.max(64);
        let n_arc_up = (n / 8).max(8);
        let n_arc_dn = (n / 8).max(8);
        let n_div = n.saturating_sub(n_arc_up + n_arc_dn + 4).max(16);

        let mut pts: Vec<[f64; 2]> = Vec::with_capacity(n + 8);

        // 1. Chamber straight.
        push(&mut pts, [0.0, rc]);
        push(&mut pts, [self.chamber_len_rt, rc]);
        // 2. Converging cone (a straight segment; two endpoints are exact).
        push(&mut pts, [z_t - r1 * beta.sin(), ra_up]);
        // 3. Upstream throat arc, wall angle beta -> 0. Ends exactly at (z_t, 1).
        for k in 1..=n_arc_up {
            let phi = beta * (1.0 - k as f64 / n_arc_up as f64);
            push(
                &mut pts,
                [z_t - r1 * phi.sin(), 1.0 + r1 * (1.0 - phi.cos())],
            );
        }
        let throat_index = pts.len() - 1;

        // 4. Diverging section.
        match spec.contour {
            ContourKind::Conical { half_angle_deg } => {
                let alpha = half_angle_deg.to_radians();
                // Exact closed form for the cone length measured from the throat
                // (identity verified symbolically; residual 4.2e-15 over 20k draws).
                let l_n = ((re - 1.0) + r2 * (1.0 / alpha.cos() - 1.0)) / alpha.tan();
                // Downstream arc, wall angle 0 -> alpha; tangency is automatic.
                for k in 1..=n_arc_dn {
                    let phi = alpha * k as f64 / n_arc_dn as f64;
                    push(
                        &mut pts,
                        [z_t + r2 * phi.sin(), 1.0 + r2 * (1.0 - phi.cos())],
                    );
                }
                // Straight cone to the exit lip.
                push(&mut pts, [z_t + l_n, re]);
            }
            // Huzel–Huang reference length, INCLUDING the throat-arc term, which
            // that reference cone takes at 1.5 R_t (`L_C15_REF_ARC_RT`) — NOT at
            // this nozzle's own downstream arc. The Aspirespace / bell_nozzle.py
            // form drops the term entirely; the two differ by an ε-dependent
            // amount (H&H longer by 2.89% at ε = 8, 1.32% at ε = 25, 0.72% at
            // ε = 69, 0.45% at ε = 165), which makes "80% bell" ambiguous unqualified.
            // This is the pinned definition (physics-reference §10).
            ContourKind::ParabolicBell { bell_percent } => {
                let (tn_deg, te_deg) = rao_angles(eps, bell_percent);
                push_bell(
                    &mut pts,
                    z_t,
                    r2,
                    re,
                    bell_percent * l_c15(eps),
                    tn_deg,
                    te_deg,
                    n_arc_dn,
                    n_div,
                )?;
            }
            // Measured geometry: the wall angles are inputs, not table lookups.
            ContourKind::DirectBell {
                theta_n_deg,
                theta_e_deg,
                length_fraction,
            } => {
                push_bell(
                    &mut pts,
                    z_t,
                    r2,
                    re,
                    length_fraction * l_c15(eps),
                    theta_n_deg,
                    theta_e_deg,
                    n_arc_dn,
                    n_div,
                )?;
            }
        }

        // Scale to metres.
        for p in &mut pts {
            p[0] *= rt;
            p[1] *= rt;
        }
        let profile = WallProfile {
            points: pts,
            throat_index,
        };
        profile.validate()?;
        Ok(profile)
    }
}

/// How dense a polyline the solver gets. All lengths SI metres.
///
/// The defaults are the work order's: chord tolerance 1% of the base cell
/// spacing, a 2° turn cap per segment, 4096 points maximum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tessellation {
    /// Maximum distance between the chord and the true curve, metres.
    pub chord_tol_m: f64,
    /// Maximum wall-angle change across one segment, degrees.
    pub max_turn_deg: f64,
    /// Hard cap on the produced point count.
    pub max_points: usize,
}

impl Tessellation {
    pub const MAX_TURN_DEG: f64 = 2.0;
    pub const MAX_POINTS: usize = 4096;
    /// Chord tolerance as a fraction of the base cell spacing.
    pub const TOL_FRACTION: f64 = 0.01;

    /// Sized to the mesh: 1% of the smaller base cell dimension.
    ///
    /// The wall's discretization error should be an order of magnitude below
    /// the cell it lands in, and no finer — the rasterizer computes EXACT
    /// sub-cell area fractions from the polyline, so chord error a hundredth of
    /// a cell is already invisible in the solid field, and paying for more is
    /// paying for nothing.
    pub fn from_cell_size(dz_m: f64, dr_m: f64) -> Self {
        Tessellation {
            chord_tol_m: Self::TOL_FRACTION * dz_m.min(dr_m).abs().max(f64::MIN_POSITIVE),
            max_turn_deg: Self::MAX_TURN_DEG,
            max_points: Self::MAX_POINTS,
        }
    }
}

/// One analytic piece of the wall, in r_t with z measured from the chamber
/// head. Every piece is a function of z (strictly increasing z), which is what
/// makes `radius_at` a closed-form lookup rather than a search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Piece {
    /// Straight segment. Exact at one segment; never subdivided.
    Line { a: [f64; 2], b: [f64; 2] },
    /// A throat arc, parameterized by LOCAL WALL ANGLE phi (which is what makes
    /// tangency to the neighbouring pieces automatic):
    /// `z = z_t + sign*R*sin(phi)`, `r = 1 + R*(1 - cos(phi))`, phi: phi0 -> phi1.
    /// `sign` is −1 upstream of the throat (phi falling), +1 downstream.
    ThroatArc {
        z_t: f64,
        radius: f64,
        sign: f64,
        phi0: f64,
        phi1: f64,
    },
    /// Quadratic Bézier N -> E through the derived control point Q, absolute z.
    Bezier {
        n: [f64; 2],
        q: [f64; 2],
        e: [f64; 2],
    },
}

impl Piece {
    pub fn is_straight(&self) -> bool {
        matches!(self, Piece::Line { .. })
    }

    /// Point at curve parameter `t` in [0, 1]. NOT arc length, and for the
    /// Bézier not proportional to z either — only the endpoints are pinned.
    pub fn point_at(&self, t: f64) -> [f64; 2] {
        match *self {
            Piece::Line { a, b } => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])],
            Piece::ThroatArc {
                z_t,
                radius,
                sign,
                phi0,
                phi1,
            } => {
                let phi = phi0 + t * (phi1 - phi0);
                [
                    z_t + sign * radius * phi.sin(),
                    1.0 + radius * (1.0 - phi.cos()),
                ]
            }
            Piece::Bezier { n, q, e } => {
                let (a, b, c) = ((1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t), t * t);
                [
                    a * n[0] + b * q[0] + c * e[0],
                    a * n[1] + b * q[1] + c * e[1],
                ]
            }
        }
    }

    pub fn start(&self) -> [f64; 2] {
        self.point_at(0.0)
    }

    pub fn end(&self) -> [f64; 2] {
        self.point_at(1.0)
    }

    /// Total wall-angle change across the piece, radians.
    pub fn turn(&self) -> f64 {
        match *self {
            Piece::Line { .. } => 0.0,
            Piece::ThroatArc { phi0, phi1, .. } => (phi1 - phi0).abs(),
            Piece::Bezier { n, q, e } => {
                let d0 = [q[0] - n[0], q[1] - n[1]];
                let d1 = [e[0] - q[0], e[1] - q[1]];
                (d1[1].atan2(d1[0]) - d0[1].atan2(d0[0])).abs()
            }
        }
    }

    /// The floor on this piece's segment count: 1 for a straight piece (it is
    /// exact), 2 for a curved one — a curved piece drawn with one segment is a
    /// straight line through its endpoints, which is not a discretization of a
    /// curve at all.
    pub fn min_segments(&self) -> usize {
        if self.is_straight() {
            1
        } else {
            2
        }
    }

    /// Segment count meeting BOTH the chord tolerance (r_t) and the turn cap.
    ///
    /// **The tolerance is RADIAL** — the gap in r between the chord and the
    /// curve at the same z — not the perpendicular distance, because the
    /// rasterizer integrates exact sub-cell areas under this polyline and an
    /// area error is `∫|Δr| dz`. The two differ by a `sec(wall angle)` factor
    /// that reaches 1.2 on a 35° bell, so bounding the perpendicular distance
    /// and calling it the chord tolerance overshoots by up to 20%. (Measured:
    /// the perpendicular form let RS-25's Bézier sit at 1.12× the requested
    /// tolerance.)
    ///
    /// Bounds, both exact for the shapes involved:
    /// - **circular arc**, n equal-angle pieces: perpendicular sagitta
    ///   `R(1 − cos(Δφ/2n)) ≤ R Δφ²/(8n²)`, divided by `cos φ_max` to convert
    ///   to a radial gap.
    /// - **quadratic Bézier**, n equal-t pieces: each sub-piece is itself a
    ///   quadratic Bézier whose second difference is `D/n²`, `D = N − 2Q + E`,
    ///   and `B(t) − chord(t) = −t(1−t)·D/n²` exactly, peaking at t = ½. The
    ///   radial component of that offset relative to a chord of slope m is
    ///   `|D_r − m·D_z|/(4n²) ≤ (|D_r| + m_max|D_z|)/(4n²)`.
    pub fn segments(&self, tol_rt: f64, max_turn_deg: f64) -> usize {
        if self.is_straight() {
            return 1;
        }
        let turn_cap = (self.turn().to_degrees() / max_turn_deg.max(1e-6)).ceil() as usize;
        let chord = match *self {
            Piece::Line { .. } => 1.0,
            Piece::ThroatArc {
                radius, phi0, phi1, ..
            } => {
                let d = (phi1 - phi0).abs();
                let sec = 1.0 / phi0.abs().max(phi1.abs()).cos().max(1e-6);
                d * (radius * sec / (8.0 * tol_rt)).sqrt()
            }
            Piece::Bezier { n, q, e } => {
                let d = [n[0] - 2.0 * q[0] + e[0], n[1] - 2.0 * q[1] + e[1]];
                // Steepest chord slope on the piece is one of its two control
                // legs (the tangent turns monotonically between them).
                let m = |a: [f64; 2], b: [f64; 2]| ((b[1] - a[1]) / (b[0] - a[0])).abs();
                let m_max = m(n, q).max(m(q, e));
                ((d[1].abs() + m_max * d[0].abs()) / (4.0 * tol_rt)).sqrt()
            }
        };
        let chord_cap = if chord.is_finite() {
            chord.ceil().max(1.0) as usize
        } else {
            2
        };
        chord_cap.max(turn_cap).max(self.min_segments())
    }

    /// Analytic radius at axial station `z` (r_t), `None` if z is outside.
    pub fn radius_at(&self, z: f64) -> Option<f64> {
        match *self {
            Piece::Line { a, b } => {
                if b[0] <= a[0] {
                    return Some(a[1]);
                }
                let t = ((z - a[0]) / (b[0] - a[0])).clamp(0.0, 1.0);
                Some(a[1] + t * (b[1] - a[1]))
            }
            Piece::ThroatArc {
                z_t, radius, sign, ..
            } => {
                // z = z_t + sign*R*sin(phi)  =>  sin(phi) = sign*(z - z_t)/R.
                let s = (sign * (z - z_t) / radius).clamp(-1.0, 1.0);
                let phi = s.asin();
                Some(1.0 + radius * (1.0 - phi.cos()))
            }
            Piece::Bezier { n, q, e } => {
                let t = bezier_t_at_z(n[0], q[0], e[0], z)?;
                let (a, b, c) = ((1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t), t * t);
                Some(a * n[1] + b * q[1] + c * e[1])
            }
        }
    }

    /// Unit downstream tangent at axial station `z` (r_t).
    pub fn tangent_at(&self, z: f64) -> [f64; 2] {
        let d = match *self {
            Piece::Line { a, b } => [b[0] - a[0], b[1] - a[1]],
            Piece::ThroatArc {
                z_t, radius, sign, ..
            } => {
                // The wall angle IS the parameter. Differentiating
                //   z = z_t + sign*R*sin(phi),  r = 1 + R*(1 - cos phi)
                // gives dr/dz = (R sin phi)/(sign*R cos phi) = sign*tan(phi).
                // So the sign of the slope is the sign of the arc: the
                // upstream arc (sign = -1) descends toward the throat, the
                // downstream arc (sign = +1) climbs away from it.
                let s = (sign * (z - z_t) / radius).clamp(-1.0, 1.0);
                let phi = s.asin();
                [1.0, sign * phi.tan()]
            }
            Piece::Bezier { n, q, e } => {
                let t = bezier_t_at_z(n[0], q[0], e[0], z).unwrap_or(0.0);
                [
                    2.0 * ((1.0 - t) * (q[0] - n[0]) + t * (e[0] - q[0])),
                    2.0 * ((1.0 - t) * (q[1] - n[1]) + t * (e[1] - q[1])),
                ]
            }
        };
        let m = d[0].hypot(d[1]).max(f64::MIN_POSITIVE);
        [d[0] / m, d[1] / m]
    }
}

/// Invert a quadratic Bézier's z component: the `t` in [0, 1] with `z(t) = z`.
///
/// `N_z < Q_z < E_z` (the §10 ordering guard) makes z(t) strictly increasing,
/// so the root is unique. Solved in the numerically stable form — the naive
/// quadratic formula loses the small root to cancellation when `4AC ≪ B²`,
/// which is exactly the near-straight bell.
fn bezier_t_at_z(nz: f64, qz: f64, ez: f64, z: f64) -> Option<f64> {
    let a = nz - 2.0 * qz + ez;
    let b = 2.0 * (qz - nz);
    let c = nz - z;
    if a.abs() < 1e-14 * b.abs().max(1.0) {
        // Degenerate: z(t) is linear.
        return (b.abs() > 0.0).then(|| (-c / b).clamp(0.0, 1.0));
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let h = -0.5 * (b + if b >= 0.0 { sq } else { -sq });
    let (t0, t1) = (h / a, if h != 0.0 { c / h } else { h / a });
    let pick = |t: f64| (-1e-9..=1.0 + 1e-9).contains(&t);
    if pick(t0) {
        Some(t0.clamp(0.0, 1.0))
    } else if pick(t1) {
        Some(t1.clamp(0.0, 1.0))
    } else {
        None
    }
}

/// The diverging bell: downstream throat arc to wall angle `tn_deg`, then a
/// quadratic Bézier to the exit lip at axial distance `l_n` from the throat
/// meeting it at `te_deg`. Shared by the Rao-table and direct-angle paths —
/// they differ only in where the two angles and the length come from.
#[allow(clippy::too_many_arguments)]
fn push_bell(
    pts: &mut Vec<[f64; 2]>,
    z_t: f64,
    r2: f64,
    re: f64,
    l_n: f64,
    tn_deg: f64,
    te_deg: f64,
    n_arc_dn: usize,
    n_div: usize,
) -> Result<()> {
    let b = BellGeometry::solve(r2, re, l_n, tn_deg, te_deg)?;
    let tn = tn_deg.to_radians();
    // Downstream arc, wall angle 0 -> theta_n; ends exactly at N.
    for k in 1..=n_arc_dn {
        let phi = tn * k as f64 / n_arc_dn as f64;
        push(pts, [z_t + r2 * phi.sin(), 1.0 + r2 * (1.0 - phi.cos())]);
    }
    // Quadratic Bézier N -> E. t is NOT proportional to z; the polyline just
    // needs monotone z, which N_z < Q_z < E_z guarantees.
    for k in 1..=n_div {
        let t = k as f64 / n_div as f64;
        let a = (1.0 - t) * (1.0 - t);
        let bb = 2.0 * t * (1.0 - t);
        let c = t * t;
        push(
            pts,
            [
                z_t + a * b.n[0] + bb * b.q[0] + c * b.e[0],
                a * b.n[1] + bb * b.q[1] + c * b.e[1],
            ],
        );
    }
    Ok(())
}

/// The three Bézier points of a bell's diverging section, in r_t with z
/// measured FROM THE THROAT. `q` is fully derived from the other two and the
/// two wall angles — it has no remaining degree of freedom, which is why the
/// editor draws it and refuses to drag it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BellGeometry {
    /// Throat-arc / Bézier tangency point.
    pub n: [f64; 2],
    /// Derived control point.
    pub q: [f64; 2],
    /// Exit lip.
    pub e: [f64; 2],
    pub theta_n_deg: f64,
    pub theta_e_deg: f64,
}

impl BellGeometry {
    /// Solve N, Q, E for a downstream arc `r2`, exit radius `re` and divergent
    /// length `l_n` (all r_t), meeting the two wall angles. The two guards
    /// (physics-reference §10) live here so every caller inherits them.
    pub fn solve(r2: f64, re: f64, l_n: f64, tn_deg: f64, te_deg: f64) -> Result<Self> {
        let (tn, te) = (tn_deg.to_radians(), te_deg.to_radians());
        if tn <= te {
            // Cannot happen with the shipped table; guards division by zero in the
            // Bézier control point below.
            return Err(CfdError::Geometry(format!(
                "theta_n ({tn_deg}) must exceed theta_e ({te_deg})"
            )));
        }
        // Bézier endpoints (z relative to throat) and control point.
        let nz = r2 * tn.sin();
        let nr = 1.0 + r2 * (1.0 - tn.cos());
        let (ez, er) = (l_n, re);
        let (m1, m2) = (tn.tan(), te.tan());
        let c1 = nr - m1 * nz;
        let c2 = er - m2 * ez;
        let qz = (c2 - c1) / (m1 - m2);
        let qr = (m1 * c2 - m2 * c1) / (m1 - m2);
        if !(nz < qz && qz < ez) {
            // Which side failed says which way to move, and this message is now
            // user-visible (the app shows it when it falls back to the cone, and
            // the parametric editor uses it as the clamp tooltip). Q_z past the
            // exit: the bell is too SHORT for the area ratio. Q_z behind the
            // throat-arc tangency: too LONG. Telling someone to lengthen an
            // already-overlong nozzle sends them the wrong way.
            let which = if qz <= nz { "too long" } else { "too short" };
            return Err(CfdError::Geometry(format!(
                "bell control point out of order (N_z={nz:.4}, Q_z={qz:.4}, E_z={ez:.4}): \
                 length {l_n:.4} r_t is {which} for area ratio {:.4} at theta_n {tn_deg} / \
                 theta_e {te_deg}",
                re * re
            )));
        }
        Ok(BellGeometry {
            n: [nz, nr],
            q: [qz, qr],
            e: [ez, er],
            theta_n_deg: tn_deg,
            theta_e_deg: te_deg,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContourKind;

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

    fn curves() -> Vec<(&'static str, NozzleCurve)> {
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
                "RS-25 80% bell",
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
            (
                "Merlin Vac 75% bell",
                NozzleCurve::new(spec(
                    165.0,
                    0.128,
                    0.382,
                    ContourKind::ParabolicBell { bell_percent: 0.75 },
                )),
            ),
        ]
    }

    /// The tessellation of a curve must lie ON the curve: every vertex is a
    /// point of `radius_at` to roundoff, and every chord midpoint is within the
    /// requested tolerance of it. The second half is the one that bites — it is
    /// the definition of "chord tolerance", measured rather than assumed.
    #[test]
    fn tessellation_lies_on_the_analytic_curve_within_tolerance() {
        for (name, c) in curves() {
            let t = Tessellation::from_cell_size(0.145 * c.spec.throat_radius_m, 0.05 * c.spec.throat_radius_m);
            let p = c.tessellate(&t).unwrap();
            let mut worst_vertex = 0.0f64;
            let mut worst_mid = 0.0f64;
            for w in p.points.windows(2) {
                let r = c.radius_at(w[0][0]).unwrap();
                worst_vertex = worst_vertex.max((r - w[0][1]).abs());
                let zm = 0.5 * (w[0][0] + w[1][0]);
                let rm = 0.5 * (w[0][1] + w[1][1]);
                worst_mid = worst_mid.max((c.radius_at(zm).unwrap() - rm).abs());
            }
            let last = *p.points.last().unwrap();
            worst_vertex = worst_vertex.max((c.radius_at(last[0]).unwrap() - last[1]).abs());
            println!(
                "{name}: {} points, worst vertex {:.2e} m, worst chord {:.2e} m (tol {:.2e})",
                p.points.len(), worst_vertex, worst_mid, t.chord_tol_m
            );
            assert!(
                worst_vertex < 1e-12 * c.spec.throat_radius_m,
                "{name}: vertices are not on the curve ({worst_vertex:.3e} m)"
            );
            // The sagitta bounds are upper bounds on the true deviation, and
            // the chord midpoint measures a lower bound on it; the tolerance
            // must hold with room to spare either way.
            assert!(
                worst_mid <= t.chord_tol_m,
                "{name}: chord deviation {worst_mid:.3e} m exceeds tolerance {:.3e}",
                t.chord_tol_m
            );
        }
    }

    /// Structural invariants the solver depends on, at every tolerance the UI
    /// can produce (8..160 cells/r_t, aspect 1..8).
    #[test]
    fn tessellation_is_monotone_with_the_throat_at_argmin_r() {
        for (name, c) in curves() {
            for cells_per_rt in [8.0, 14.0, 20.0, 40.0, 160.0] {
                for aspect in [1.0, 2.9, 8.0] {
                    let rt = c.spec.throat_radius_m;
                    let dr = rt / cells_per_rt;
                    let t = Tessellation::from_cell_size(aspect * dr, dr);
                    let p = c.tessellate(&t).unwrap();
                    let at = format!("{name} @ {cells_per_rt} cells/r_t, aspect {aspect}");
                    for w in p.points.windows(2) {
                        assert!(w[1][0] > w[0][0], "{at}: z not strictly increasing");
                    }
                    // Throat is the STRICT argmin: exactly one vertex at r_t.
                    let rmin = p.points.iter().map(|q| q[1]).fold(f64::MAX, f64::min);
                    assert!(
                        (rmin / rt - 1.0).abs() < 1e-12,
                        "{at}: min radius {} r_t",
                        rmin / rt
                    );
                    assert_eq!(
                        p.points.iter().filter(|q| q[1] <= rmin + 1e-15 * rt).count(),
                        1,
                        "{at}: throat is not a strict minimum"
                    );
                    assert!(
                        (p.points[p.throat_index][1] - rmin).abs() < 1e-15 * rt,
                        "{at}: throat_index does not point at the minimum"
                    );
                    assert!(p.points.len() <= t.max_points, "{at}: over the point cap");
                }
            }
        }
    }

    /// The analytic angles ARE the requested angles, exactly — no chord slope,
    /// no tessellation bias. This is what the app's angle readouts and the
    /// published-geometry tests should be reading.
    #[test]
    fn analytic_angles_reproduce_their_inputs_exactly() {
        let c = NozzleCurve::new(spec(
            34.3,
            0.115,
            0.300,
            ContourKind::DirectBell {
                theta_n_deg: 32.0,
                theta_e_deg: 6.0,
                length_fraction: 0.76,
            },
        ));
        assert_eq!(c.divergent_angles_deg().unwrap(), (32.0, 6.0));
        // …and the curve's own tangent at the exit plane agrees with them.
        let z_e = c.length_m().unwrap();
        let tg = c.tangent_at(z_e).unwrap();
        assert!(
            (tg[1].atan2(tg[0]).to_degrees() - 6.0).abs() < 1e-9,
            "exit tangent {:?}",
            tg
        );
        // A cone reports its half-angle for both.
        let cone = NozzleCurve::new(spec(
            8.0,
            0.05,
            0.382,
            ContourKind::Conical {
                half_angle_deg: 15.0,
            },
        ));
        assert_eq!(cone.divergent_angles_deg().unwrap(), (15.0, 15.0));
    }

    /// The polyline reproduces the analytic angles to within the turn cap. This
    /// is the honest statement about tessellation: it does not reproduce them
    /// exactly, it reproduces them to the cap, and the cap is the knob.
    #[test]
    fn tessellation_reproduces_the_analytic_angles_within_the_turn_cap() {
        for (name, c) in curves() {
            let rt = c.spec.throat_radius_m;
            let t = Tessellation::from_cell_size(0.145 * rt, 0.05 * rt);
            let p = c.tessellate(&t).unwrap();
            let (tn, te) = c.divergent_angles_deg().unwrap();
            // Exit: the last chord's slope.
            let n = p.points.len();
            let (a, b) = (p.points[n - 2], p.points[n - 1]);
            let te_chord = ((b[1] - a[1]) / (b[0] - a[0])).atan().to_degrees();
            // theta_n: steepest CENTRAL chord downstream of the throat (the
            // junction vertex straddles two pieces that are both tangent to
            // theta_n there, so a central chord recovers it to second order).
            let tn_chord = (p.throat_index + 1..n - 1)
                .map(|i| {
                    let (u, v) = (p.points[i - 1], p.points[i + 1]);
                    ((v[1] - u[1]) / (v[0] - u[0])).atan().to_degrees()
                })
                .fold(f64::NEG_INFINITY, f64::max);
            println!(
                "{name}: analytic ({tn:.3}, {te:.3}) vs chord ({tn_chord:.3}, {te_chord:.3}) deg"
            );
            assert!(
                (te_chord - te).abs() <= t.max_turn_deg,
                "{name}: exit chord {te_chord:.3} vs analytic {te:.3} deg"
            );
            assert!(
                (tn_chord - tn).abs() <= t.max_turn_deg,
                "{name}: junction chord {tn_chord:.3} vs analytic {tn:.3} deg"
            );
        }
    }

    /// Halving the tolerance must not move the wall: the tessellation has
    /// CONVERGED at the default tolerance, which is the whole claim.
    #[test]
    fn halving_the_tolerance_does_not_move_the_wall() {
        for (name, c) in curves() {
            let rt = c.spec.throat_radius_m;
            let base = Tessellation::from_cell_size(0.145 * rt, 0.05 * rt);
            let mut fine = base;
            fine.chord_tol_m *= 0.5;
            fine.max_turn_deg *= 0.5;
            let (a, b) = (c.tessellate(&base).unwrap(), c.tessellate(&fine).unwrap());
            // Sample both walls on a common grid and compare radii.
            let z0 = a.points[0][0];
            let z1 = a.points.last().unwrap()[0];
            let mut worst = 0.0f64;
            for k in 0..=2000 {
                let z = z0 + (z1 - z0) * k as f64 / 2000.0;
                if let (Some(ra), Some(rb)) = (a.radius_at(z), b.radius_at(z)) {
                    worst = worst.max((ra - rb).abs() / rt);
                }
            }
            println!(
                "{name}: {} -> {} points on halving, worst |dr| {:.3e} r_t",
                a.points.len(), b.points.len(), worst
            );
            assert!(worst < 2.0 * base.chord_tol_m / rt, "{name}: {worst:.3e} r_t");
        }
    }

    /// The point cap binds without producing an invalid wall: an absurdly tight
    /// tolerance must still tessellate, still be monotone, and still stop at
    /// the cap.
    #[test]
    fn the_point_cap_binds_and_the_wall_stays_valid() {
        for (name, c) in curves() {
            let t = Tessellation {
                chord_tol_m: 1e-12,
                max_turn_deg: 0.01,
                max_points: Tessellation::MAX_POINTS,
            };
            let p = c.tessellate(&t).unwrap();
            assert!(
                p.points.len() <= t.max_points && p.points.len() > 1000,
                "{name}: {} points",
                p.points.len()
            );
            for w in p.points.windows(2) {
                assert!(w[1][0] > w[0][0], "{name}: z not strictly increasing at the cap");
            }
        }
    }

    /// The analytic tangent must agree with the derivative of the analytic
    /// radius, on EVERY piece. The two are independent code paths — one
    /// differentiates the parameterization, the other inverts it — and the
    /// throat arcs are where they can disagree in sign without anything else
    /// noticing, because the app reads its wall angles off the divergent
    /// section and nothing there touches the converging arc.
    #[test]
    fn the_tangent_is_the_derivative_of_the_radius() {
        for (name, c) in curves() {
            let rt = c.spec.throat_radius_m;
            let z1 = c.length_m().unwrap();
            let h = 1e-7 * z1;
            let mut worst = 0.0f64;
            for k in 1..2000 {
                let z = z1 * k as f64 / 2000.0;
                let (Some(a), Some(b)) = (c.radius_at(z - h), c.radius_at(z + h)) else {
                    continue;
                };
                let fd = (b - a) / (2.0 * h);
                let t = c.tangent_at(z).unwrap();
                let an = t[1] / t[0];
                // Skip the piece joins, where a central difference straddles a
                // curvature discontinuity and measures the average of two
                // different slopes.
                if (fd - an).abs() > 0.05 && is_near_a_join(&c, z / rt) {
                    continue;
                }
                worst = worst.max((fd - an).abs());
            }
            println!("{name}: worst |dr/dz analytic - finite difference| = {worst:.2e}");
            assert!(worst < 1e-4, "{name}: tangent disagrees with radius by {worst:.3e}");
            // The converging wall descends and the diverging wall climbs — the
            // sign error this test exists to catch.
            let z_t = c.throat_z_rt() * rt;
            assert!(
                c.tangent_at(0.5 * z_t).unwrap()[1] <= 0.0,
                "{name}: the converging wall must not climb"
            );
            assert!(
                c.tangent_at(0.5 * (z_t + z1)).unwrap()[1] > 0.0,
                "{name}: the diverging wall must climb"
            );
            // …and just inside each throat arc, specifically.
            let r1 = c.spec.throat_arc_up;
            assert!(
                c.tangent_at(z_t - 0.3 * r1 * rt).unwrap()[1] < 0.0,
                "{name}: the upstream throat arc must descend"
            );
            let r2 = c.spec.throat_arc_down;
            assert!(
                c.tangent_at(z_t + 0.3 * r2 * rt).unwrap()[1] > 0.0,
                "{name}: the downstream throat arc must climb"
            );
        }
    }

    /// Within a hair of a piece boundary, in r_t.
    fn is_near_a_join(c: &NozzleCurve, z_rt: f64) -> bool {
        c.pieces()
            .unwrap()
            .iter()
            .any(|p| (p.end()[0] - z_rt).abs() < 1e-3 || (p.start()[0] - z_rt).abs() < 1e-3)
    }

    /// `radius_at` outside the curve is `None`, and the Bézier inversion is
    /// exact: round-tripping z -> t -> z reproduces z.
    #[test]
    fn radius_at_is_bounded_and_the_bezier_inverts_exactly() {
        let c = curves()[1].1; // RS-25 bell
        let rt = c.spec.throat_radius_m;
        assert_eq!(c.radius_at(-1e-6 * rt - 1e-9), None);
        assert_eq!(c.radius_at(c.length_m().unwrap() + 1e-6), None);
        assert!((c.radius_at(0.0).unwrap() / rt - 2.0).abs() < 1e-12); // chamber, CR 4
        let b = c.bell().unwrap();
        let z_t = c.throat_z_rt();
        let mut worst = 0.0f64;
        for k in 0..=1000 {
            let t = k as f64 / 1000.0;
            let (a, bb, cc) = ((1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t), t * t);
            let z = a * b.n[0] + bb * b.q[0] + cc * b.e[0] + z_t;
            let r = a * b.n[1] + bb * b.q[1] + cc * b.e[1];
            worst = worst.max((c.radius_at(z * rt).unwrap() / rt - r).abs());
        }
        assert!(worst < 1e-10, "Bezier inversion worst |dr| {worst:.3e} r_t");
    }
}
