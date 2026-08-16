//! Surface pressure integrals over immersed bodies.
//!
//! The solver's engineering output is an exit-plane control volume
//! (docs/physics-reference.md §11) because that is the defensible definition
//! for a nozzle. It is meaningless for an arbitrary drawn body: there is no
//! exit plane, no lip and no throat. This module is the other half — the force
//! a body feels, obtained by summing the wall momentum flux the solver already
//! applies at every fluid/solid face.
//!
//! **The integrand is not a re-derivation.** `physics::wall_flux_z` /
//! `wall_flux_r` return the oriented momentum flux the sweeps subtract from
//! the adjacent fluid cell; summing exactly that quantity, area-weighted, is
//! the discrete reaction to it. So the force reported here and the momentum
//! the fluid actually loses at the wall are the same number by construction —
//! which is what makes a momentum balance over a domain containing a body
//! checkable at all (ladder rung G3 does exactly that).
//!
//! Everything here is non-dimensional and chamber-referenced, like the rest of
//! the solver. `SurfaceForce::to_si` is the only conversion.

use cfd_contract::{GasModel, Geometry, Prim, RefScales, SolidField};

/// Pressure force on a set of solid cells, plus the wetted area it was
/// integrated over. Axisymmetric forces carry the full 2*pi (they are whole
/// rings); planar forces are per unit depth.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SurfaceForce {
    /// Axial component, positive downstream (drag on a body in +z flow).
    pub f_z: f64,
    /// Radial component, positive outward. Zero by symmetry for a body
    /// straddling the axis in Axisymmetric mode; lift for a planar body.
    pub f_r: f64,
    /// Wetted area of the faces summed (same 2*pi / unit-depth convention).
    pub area: f64,
    /// Number of fluid/solid faces summed. Zero means the selection touched no
    /// fluid — a fully buried body, or an empty selection.
    pub faces: usize,
}

impl SurfaceForce {
    /// Non-dimensional -> SI. Force scales as `p_ref * L_ref^2`, area as
    /// `L_ref^2`.
    pub fn to_si(self, refs: &RefScales) -> SurfaceForce {
        let l2 = refs.l_m * refs.l_m;
        SurfaceForce {
            f_z: self.f_z * refs.p_pa * l2,
            f_r: self.f_r * refs.p_pa * l2,
            area: self.area * l2,
            faces: self.faces,
        }
    }

    /// Force coefficient on a reference area, e.g. `C_d = f_z / (q * A_ref)`.
    /// `q` is the dynamic pressure in the same units as the force.
    pub fn coefficient(component: f64, q: f64, a_ref: f64) -> f64 {
        component / (q * a_ref)
    }
}

/// Connected components of the solid field — the sandbox may hold any number
/// of disconnected bodies and a per-body force needs to know which cells are
/// which. 4-connectivity on the interior index space (the same neighbourhood
/// the fluxes use); diagonal-only contact is two bodies, matching the fact
/// that no flux couples across a corner.
#[derive(Debug, Clone)]
pub struct Bodies {
    /// One entry per interior cell: `usize::MAX` for fluid, else the body id.
    label: Vec<usize>,
    counts: Vec<usize>,
}

/// Sentinel stored for fluid cells.
const FLUID: usize = usize::MAX;

impl Bodies {
    /// Number of disconnected solid bodies.
    pub fn count(&self) -> usize {
        self.counts.len()
    }

    /// Body id of an interior cell, or `None` if the cell is fluid.
    pub fn label(&self, idx: usize) -> Option<usize> {
        match self.label[idx] {
            FLUID => None,
            b => Some(b),
        }
    }

    /// Cell count of a body. Bodies are numbered in order of first
    /// appearance scanning row-major, so id 0 is the body containing the
    /// lowest interior index.
    pub fn cells(&self, body: usize) -> usize {
        self.counts.get(body).copied().unwrap_or(0)
    }

    /// Predicate ready to hand to [`surface_pressure_force`].
    pub fn selector(&self, body: usize) -> impl Fn(usize) -> bool + '_ {
        move |idx| self.label[idx] == body
    }

    /// Bodies sorted by cell count, largest first. The identity of "the body"
    /// in a case with walls plus an obstacle is usually "the biggest one that
    /// is not the wall", and callers need a stable way to say so.
    pub fn by_size(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..self.count()).collect();
        ids.sort_by_key(|&b| std::cmp::Reverse(self.counts[b]));
        ids
    }
}

/// Label the solid field's connected components. Cost is one BFS over the
/// interior; call it once per geometry, not per step.
pub fn label_bodies(solid: &SolidField) -> Bodies {
    let g = &solid.grid;
    let n = g.len();
    let mut label = vec![FLUID; n];
    let mut counts: Vec<usize> = Vec::new();
    let mut queue: Vec<(usize, usize)> = Vec::new();
    for ir0 in 0..g.nr {
        for iz0 in 0..g.nz {
            let seed = g.idx(iz0, ir0);
            if !solid.is_solid(seed) || label[seed] != FLUID {
                continue;
            }
            let id = counts.len();
            counts.push(0);
            label[seed] = id;
            queue.clear();
            queue.push((iz0, ir0));
            while let Some((iz, ir)) = queue.pop() {
                counts[id] += 1;
                let mut visit = |jz: usize, jr: usize, q: &mut Vec<(usize, usize)>| {
                    let j = g.idx(jz, jr);
                    if solid.is_solid(j) && label[j] == FLUID {
                        label[j] = id;
                        q.push((jz, jr));
                    }
                };
                if iz > 0 { visit(iz - 1, ir, &mut queue); }
                if iz + 1 < g.nz { visit(iz + 1, ir, &mut queue); }
                if ir > 0 { visit(iz, ir - 1, &mut queue); }
                if ir + 1 < g.nr { visit(iz, ir + 1, &mut queue); }
            }
        }
    }
    Bodies { label, counts }
}

/// Pressure force on the solid cells `body` selects, integrated over their
/// fluid/solid faces.
///
/// `w` is interior-only canonical primitives `[rho, u_z, u_r, p]`,
/// `grid.len()` long — what `EulerSolver::primitives` returns. Solid entries
/// are never read. `body` takes an interior cell index; pass `|_| true` for
/// the whole solid field, or `bodies.selector(id)` for one component.
///
/// Faces on the domain boundary are not fluid/solid faces and contribute
/// nothing, exactly as in the sweeps.
pub fn surface_pressure_force(
    w: &[Prim],
    solid: &SolidField,
    gas: &GasModel,
    geometry: Geometry,
    body: impl Fn(usize) -> bool,
) -> SurfaceForce {
    let g = &solid.grid;
    assert_eq!(w.len(), g.len(), "primitives must be interior-only, grid.len() long");
    let gamma = gas.gamma;
    let axisym = geometry == Geometry::Axisymmetric;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut out = SurfaceForce::default();

    // Axial faces: area is the exact annulus 2*pi*r_center*dr (r_center is the
    // arithmetic face mean, so r_center*dr IS the true shell area), or dr per
    // unit depth in planar mode.
    for ir in 0..g.nr {
        let da = if axisym {
            two_pi * g.r_center(ir) as f64 * g.dr(ir) as f64
        } else {
            g.dr(ir) as f64
        };
        for iz in 0..g.nz.saturating_sub(1) {
            let (lo, hi) = (g.idx(iz, ir), g.idx(iz + 1, ir));
            let (ls, hs) = (solid.is_solid(lo), solid.is_solid(hi));
            if ls == hs {
                continue;
            }
            // sgn = +1 when the FLUID is on the low side of the face, matching
            // physics::wall_flux_z. Its [1] component is already sgn*ps, and
            // sgn*ps is the force per unit area ON THE BODY in +z: the fluid
            // cell loses exactly that much z-momentum through the face.
            let (fluid, sgn) = if hs { (lo, 1.0) } else { (hi, -1.0) };
            let solid_cell = if hs { hi } else { lo };
            if !body(solid_cell) {
                continue;
            }
            out.f_z += crate::physics::wall_flux_z(w[fluid], sgn, gamma)[1] as f64 * da;
            out.area += da;
            out.faces += 1;
        }
    }

    // Radial faces: area is 2*pi*r_face*dz (the face radius, not the cell
    // centre), or dz per unit depth. The r = 0 face carries zero area in
    // Axisymmetric mode and is a symmetry plane in Planar mode; neither is a
    // fluid/solid face, so neither appears here.
    for ir in 0..g.nr.saturating_sub(1) {
        let r_f = g.r_face(ir + 1) as f64;
        for iz in 0..g.nz {
            let (lo, hi) = (g.idx(iz, ir), g.idx(iz, ir + 1));
            let (ls, hs) = (solid.is_solid(lo), solid.is_solid(hi));
            if ls == hs {
                continue;
            }
            let da = if axisym { two_pi * r_f * g.dz(iz) as f64 } else { g.dz(iz) as f64 };
            let (fluid, sgn) = if hs { (lo, 1.0) } else { (hi, -1.0) };
            let solid_cell = if hs { hi } else { lo };
            if !body(solid_cell) {
                continue;
            }
            out.f_r += crate::physics::wall_flux_r(w[fluid], sgn, gamma)[2] as f64 * da;
            out.area += da;
            out.faces += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cfd_contract::Grid;

    /// A box of quiescent gas at p = 1 around a rectangular block: every face
    /// sees the same static pressure, so both force components must cancel to
    /// roundoff and the wetted area must be the block's perimeter.
    #[test]
    fn quiescent_pressure_gives_zero_net_force() {
        let g = Grid::uniform(20, 12, 0.1, 0.1);
        let mut solid = SolidField::empty(g.clone());
        for ir in 3..7 {
            for iz in 5..11 {
                solid.fraction[g.idx(iz, ir)] = 1.0;
            }
        }
        let w = vec![[1.0, 0.0, 0.0, 1.0]; g.len()];
        let gas = GasModel { gamma: 1.4, r_specific_si: 287.0 };
        let f = surface_pressure_force(&w, &solid, &gas, Geometry::Planar, |_| true);
        assert_eq!(f.faces, 2 * 4 + 2 * 6, "perimeter faces");
        assert!(f.f_z.abs() < 1e-12, "f_z = {}", f.f_z);
        assert!(f.f_r.abs() < 1e-12, "f_r = {}", f.f_r);
        // 6 cells wide x 0.1 twice, 4 cells tall x 0.1 twice. The widths are
        // f32, so compare relatively.
        let expect = 2.0 * 6.0 * 0.1 + 2.0 * 4.0 * 0.1;
        assert!((f.area - expect).abs() / expect < 1e-6, "{} vs {expect}", f.area);
    }

    /// A pressure difference across the block reproduces (p_hi - p_lo) * A.
    /// Signs: higher pressure upstream pushes the body downstream.
    #[test]
    fn pressure_difference_gives_the_projected_force() {
        let g = Grid::uniform(20, 12, 0.1, 0.1);
        let mut solid = SolidField::empty(g.clone());
        for ir in 3..7 {
            for iz in 5..11 {
                solid.fraction[g.idx(iz, ir)] = 1.0;
            }
        }
        // Upstream of the block p = 2, everywhere else p = 1, gas at rest so
        // the wall star pressure is exactly the cell pressure.
        let mut w = vec![[1.0f32, 0.0, 0.0, 1.0f32]; g.len()];
        for ir in 0..g.nr {
            for iz in 0..5 {
                w[g.idx(iz, ir)] = [1.0, 0.0, 0.0, 2.0];
            }
        }
        let gas = GasModel { gamma: 1.4, r_specific_si: 287.0 };
        let f = surface_pressure_force(&w, &solid, &gas, Geometry::Planar, |_| true);
        // Front face: 4 cells x 0.1 at p = 2 pushing +z; back face 4 x 0.1 at
        // p = 1 pushing -z. Top and bottom are r-faces and cancel in z.
        assert!((f.f_z - (2.0 - 1.0) * 0.4).abs() < 1e-6, "f_z = {}", f.f_z);
        assert!(f.f_r.abs() < 1e-6, "f_r = {}", f.f_r);
    }

    /// Two blocks that touch only at a corner are two bodies: no flux couples
    /// across a diagonal, so neither may the labelling.
    #[test]
    fn corner_contact_is_two_bodies() {
        let g = Grid::uniform(12, 12, 0.1, 0.1);
        let mut solid = SolidField::empty(g.clone());
        for (iz, ir) in [(3, 3), (3, 4), (4, 3), (5, 5), (5, 6), (6, 6)] {
            solid.fraction[g.idx(iz, ir)] = 1.0;
        }
        let b = label_bodies(&solid);
        assert_eq!(b.count(), 2);
        assert_eq!(b.cells(0), 3);
        assert_eq!(b.cells(1), 3);
        assert_eq!(b.label(g.idx(3, 3)), Some(0));
        assert_eq!(b.label(g.idx(5, 5)), Some(1));
        assert_eq!(b.label(g.idx(0, 0)), None);
    }

    /// Per-body selection isolates one component of a two-body mask: with
    /// pressure raised only around the second block, the first reports zero.
    #[test]
    fn per_body_selection_isolates_one_component() {
        let g = Grid::uniform(24, 12, 0.1, 0.1);
        let mut solid = SolidField::empty(g.clone());
        for iz in 3..6 {
            solid.fraction[g.idx(iz, 5)] = 1.0;
        }
        for iz in 15..18 {
            solid.fraction[g.idx(iz, 5)] = 1.0;
        }
        let bodies = label_bodies(&solid);
        assert_eq!(bodies.count(), 2);
        let mut w = vec![[1.0f32, 0.0, 0.0, 1.0f32]; g.len()];
        for ir in 0..g.nr {
            for iz in 0..15 {
                w[g.idx(iz, ir)] = [1.0, 0.0, 0.0, 3.0];
            }
        }
        let gas = GasModel { gamma: 1.4, r_specific_si: 287.0 };
        let f0 = surface_pressure_force(&w, &solid, &gas, Geometry::Planar,
                                        bodies.selector(0));
        let f1 = surface_pressure_force(&w, &solid, &gas, Geometry::Planar,
                                        bodies.selector(1));
        assert!(f0.f_z.abs() < 1e-6, "body 0 sits in uniform pressure: {}", f0.f_z);
        assert!((f1.f_z - (3.0 - 1.0) * 0.1).abs() < 1e-6, "body 1 f_z = {}", f1.f_z);
        let all = surface_pressure_force(&w, &solid, &gas, Geometry::Planar, |_| true);
        assert_eq!(all.faces, f0.faces + f1.faces);
    }

    /// Axisymmetric areas carry the full 2*pi and use the FACE radius on
    /// radial faces, the arithmetic-mean cell radius on axial ones.
    #[test]
    fn axisymmetric_areas_use_the_right_radius() {
        let g = Grid::uniform(10, 10, 0.1, 0.1);
        let mut solid = SolidField::empty(g.clone());
        solid.fraction[g.idx(5, 4)] = 1.0;
        let w = vec![[1.0, 0.0, 0.0, 1.0]; g.len()];
        let gas = GasModel { gamma: 1.4, r_specific_si: 287.0 };
        let f = surface_pressure_force(&w, &solid, &gas, Geometry::Axisymmetric, |_| true);
        let tp = 2.0 * std::f64::consts::PI;
        let expect = 2.0 * tp * 0.45 * 0.1              // two axial faces
            + tp * 0.4 * 0.1 + tp * 0.5 * 0.1;          // inner and outer radial faces
        assert!((f.area - expect).abs() / expect < 1e-6, "{} vs {}", f.area, expect);
        assert!(f.f_z.abs() < 1e-12);
        // Uniform pressure on a ring gives a net INWARD force: the outer face
        // is the larger one. That is geometry, not an error — it is the same
        // p*dA imbalance the axisymmetric source term cancels in the solver.
        assert!((f.f_r + tp * 0.1 * (0.5 - 0.4)).abs() < 1e-6, "f_r = {}", f.f_r);
    }
}
