//! Sweeps, timestep, positivity, reductions. **Session A owns this file and
//! nothing else.** Signatures are frozen — they are what `step.rs` calls. If
//! you believe one is wrong, print `CONTRACT CHANGE REQUEST:` with the exact
//! diff and stop.
//!
//! Layout (set by `step.rs`, which allocates every buffer):
//! - All slices are padded, `g.glen()` long, indexed with `Grid::gidx`.
//! - `w` holds CANONICAL primitives `[rho, u_z, u_r, p]`. Rotate to the sweep
//!   direction at the point of use; `hllc_flux` expects `[rho, u_n, u_t, p]`.
//! - `solid` is the thresholded solid mask, edge-replicated into the ghost
//!   band. `mask` is the carbuncle mask, false in ghosts.
//! - `rhs` is zeroed by `step.rs` before each stage. `sweep_z` and `sweep_r`
//!   both ACCUMULATE (+=) into it, interior cells only.
//! - Every reduction accumulates in f64. Ping-pong: read immutably, write
//!   through `par_chunks_mut` over rows. Never mutate in place.

#![allow(unused_variables)] // remove when implementing

use cfd_contract::{Cons, GasModel, Grid, Numerics, Prim, Real};

/// Canonical unrotated primitives for every padded cell (ghosts included —
/// `fill_ghosts` ran first).
pub fn compute_primitives(u: &[Cons], w: &mut [Prim], gamma: Real) {
    todo!()
}

/// max over FLUID interior cells of (|u|+a)/dz + (|v|+a)/dr. f64 accumulator.
pub fn max_wave_speed(w: &[Prim], solid: &[bool], g: &Grid, gamma: Real) -> Real {
    todo!()
}

/// Axial sweep: MUSCL face states via `kernels::muscl_face_states`, flux via
/// `kernels::hllc_flux` (axial faces ALWAYS use HLLC — contact resolution is
/// what keeps the plume boundary sharp — except under `FluxMode::Hll`).
/// At a fluid/solid face call `crate::physics::wall_flux_z`. Accumulates
/// -dF/dz into `rhs`. Face-level positivity fallback increments `floors`.
pub fn sweep_z(w: &[Prim], solid: &[bool], mask: &[bool], g: &Grid,
               n: &Numerics, gas: &GasModel, rhs: &mut [Cons], floors: &mut u64) {
    todo!()
}

/// Radial sweep. Carries the axisymmetric machinery: the r-weighted radial
/// flux difference AND the axisymmetric pressure source, written inside a
/// SINGLE bracket:
///
///   rhs_r[j] -= [ (r_{j+1/2}*G_{j+1/2} - r_{j-1/2}*G_{j-1/2}) - p_j*dr ] / (dr * r_j)
///
/// In f32 the separated form leaves a residue that accumulates into a faint
/// axis artifact; inside one bracket the identical operands cancel bit-exactly.
/// See docs/physics-reference.md §1. Under `Geometry::Planar` drop the
/// r-weighting and the source entirely. Radial faces use HLL where `mask` is
/// set (or under `FluxMode::Hll`/`HllRadial`), HLLC otherwise.
pub fn sweep_r(w: &[Prim], solid: &[bool], mask: &[bool], g: &Grid,
               n: &Numerics, gas: &GasModel, rhs: &mut [Cons], floors: &mut u64) {
    todo!()
}

/// u1 = u0 + dt*rhs on fluid cells; solid cells copy u0 unchanged. Ghost
/// entries copy u0 (rhs is zero there), so ghosts pass through unchanged.
pub fn accumulate(out: &mut [Cons], u0: &[Cons], rhs: &[Cons], dt: Real, solid: &[bool]) {
    todo!()
}

/// SSP-RK2 stage 2: out = 0.5*(u0 + u1 + dt*rhs) on fluid cells; solid cells
/// copy u0 unchanged.
pub fn accumulate2(out: &mut [Cons], u0: &[Cons], u1: &[Cons], rhs: &[Cons], dt: Real, solid: &[bool]) {
    todo!()
}

/// Cell-level positivity pass (the face-level fallback lives in the sweeps):
/// clamp rho and p to the floors, incrementing `floors` per clamped cell.
/// See docs/physics-reference.md §5.
pub fn enforce_positivity(u: &mut [Cons], gas: &GasModel, floors: &mut u64) {
    todo!()
}

/// L2 of (rho_new - rho_old) over fluid cells, f64 accumulator. `step.rs`
/// guarantees the ghost bands of both arrays are identical before this call.
pub fn density_residual_f64(u0: &[Cons], u1: &[Cons], solid: &[bool]) -> f64 {
    todo!()
}
