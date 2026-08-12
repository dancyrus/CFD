//! Walls, boundary conditions, sponge, carbuncle sensor, initialization,
//! geometry flips. **Session B owns this file and nothing else.** Signatures
//! are frozen — they are what `step.rs` and session A's sweeps call. If you
//! believe one is wrong, print `CONTRACT CHANGE REQUEST:` with the exact diff
//! and stop.
//!
//! Layout: all padded slices are `g.glen()` long, indexed with `Grid::gidx`
//! (see `lib.rs`). Every reduction accumulates in f64. All state is
//! non-dimensional, chamber-referenced: R = 1, p = rho*T, chamber (1, 0, 0, 1).

#![allow(unused_variables)] // remove when implementing

use cfd_contract::{Ambient, Chamber, Cons, GasModel, Grid, Numerics, Prim, Real, SolidField};

/// Wall flux at a fluid/solid face. `w` is the fluid cell's CANONICAL
/// primitives. `sgn` is +1 when the fluid is on the low side of the face,
/// -1 on the high side. Returns [0, sgn*ps, 0, 0] — mass and energy flux are
/// bit-exactly zero. Special-cased, NOT computed by calling hllc_flux with a
/// mirrored state (the S_M = 0 cancellation is exact in real arithmetic but
/// not under FMA contraction). See docs/physics-reference.md §3.
pub fn wall_flux_z(w: Prim, sgn: Real, gamma: Real) -> Cons {
    todo!()
}

/// Returns [0, 0, sgn*ps, 0].
pub fn wall_flux_r(w: Prim, sgn: Real, gamma: Real) -> Cons {
    todo!()
}

/// Axis mirror (two ghost rows, (rho, u_z, -u_r, p) — applies at r-min in
/// Planar mode too), stagnation inlet on open z-min cells, supersonic/subsonic
/// outflow at z-max, radial far field at r-max. See docs/physics-reference.md
/// §4 — including the MANDATORY `D = max(D, 0)` clamp in the inlet.
pub fn fill_ghosts(u: &mut [Cons], g: &Grid, solid: &[bool], gas: &GasModel,
                   chamber: &Chamber, ambient: &Ambient, n: &Numerics) {
    todo!()
}

/// dt-based, sigma_max = 12*a_ambient/L_sponge. NOT the per-step form (4-6x
/// too weak, resolution-dependent). Radial far-field rows only — never the
/// downstream boundary, where the core is supersonic. `cells` = 0 disables.
pub fn apply_sponge(u: &mut [Cons], g: &Grid, dt: Real, ambient: &Ambient,
                    gas: &GasModel, cells: usize) {
    todo!()
}

/// Omega = min/max of p over a +/-2 cell axial window; where Omega < 0.7 set
/// mask on cells i-1..=i+1 of that row. The caller uses HLL for the RADIAL
/// fluxes of masked cells; axial fluxes always use HLLC.
pub fn carbuncle_mask(w: &[Prim], g: &Grid, mask: &mut [bool]) {
    todo!()
}

/// Quasi-1D isentropic in the nozzle, ambient elsewhere, blended over 4 cells
/// at the exit plane. Open radius per column: r_w(i) = sqrt(2*dr*sum_j
/// (1-frac[i][j])*r_j) — r-weighted, with the square root. If the area has no
/// interior minimum (arbitrary drawn blob), fall back to ambient everywhere;
/// do not crash.
pub fn quasi1d_init(u: &mut [Cons], g: &Grid, solid: &SolidField,
                    gas: &GasModel, chamber: &Chamber, ambient: &Ambient) {
    todo!()
}

/// Area-Mach inversion, NASA b4wind, 8 fixed Newton iterations.
/// Guard |ar - 1| < 1e-6 -> return 1.0 (double root: Newton NaNs at ar = 1).
pub fn mach_from_area_ratio(ar: f64, gamma: f64, supersonic: bool) -> f64 {
    todo!()
}

/// What separates "conservation drift" from "the user drew a hole". Test T2
/// asserts against this, not against zero.
#[derive(Default, Debug, Clone, Copy)]
pub struct FlipLedger { pub mass: f64, pub energy: f64 }

/// Flip bookkeeping plus BFS refill (at most 8 passes) of newly opened cells
/// AT REST (u = v = 0 — an opened cell inheriting a Mach-3 neighbour's
/// velocity fires a shock into the new cavity). Remaining unreached cells get
/// ambient. Called from the top of `step()`, never inside an RK stage.
pub fn apply_geometry_change(u: &mut [Cons], g: &Grid, old: &SolidField,
                             new: &SolidField, gas: &GasModel,
                             ambient: &Ambient, ledger: &mut FlipLedger) {
    todo!()
}
