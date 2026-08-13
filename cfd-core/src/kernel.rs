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

use cfd_contract::kernels::{cons_to_prim, hll_flux, hllc_flux, muscl_face_states};
use cfd_contract::{
    Cons, FluxMode, GasModel, Geometry, Grid, Numerics, Prim, Real, Reconstruction, WallMode,
    NG, P_MIN_ABS, RHO_MIN,
};
use rayon::prelude::*;

/// Canonical unrotated primitives for every padded cell (ghosts included —
/// `fill_ghosts` ran first).
pub fn compute_primitives(u: &[Cons], w: &mut [Prim], gamma: Real) {
    w.par_iter_mut()
        .zip(u.par_iter())
        .for_each(|(wc, uc)| *wc = cons_to_prim(*uc, gamma));
}

/// max over FLUID interior cells of (|u|+a)/dz + (|v|+a)/dr. f64 accumulator.
pub fn max_wave_speed(w: &[Prim], solid: &[bool], g: &Grid, gamma: Real) -> Real {
    let inv_dz = 1.0 / g.dz as f64;
    let inv_dr = 1.0 / g.dr as f64;
    let gamma = gamma as f64;
    let wmax = (0..g.nr)
        .into_par_iter()
        .map(|ir| {
            let mut mx = 0.0f64;
            for iz in 0..g.nz {
                let i = g.gidx(iz as isize, ir as isize);
                if solid[i] {
                    continue;
                }
                let [rho, uz, ur, p] = w[i];
                let a = (gamma * p as f64 / rho as f64).sqrt();
                let s = ((uz as f64).abs() + a) * inv_dz + ((ur as f64).abs() + a) * inv_dr;
                if s > mx {
                    mx = s;
                }
            }
            mx
        })
        .reduce(|| 0.0, f64::max);
    wmax as Real
}

/// Exact analytic flux of a single state in the rotated frame — the exact
/// Riemann solution when both face states are bitwise identical. Taking it
/// there is not a speed hack: it keeps uniform regions bit-exactly uniform
/// (T1, well-balanced) instead of picking up per-face rounding residue from
/// the approximate solver's wave-speed algebra.
#[inline]
fn analytic_flux(w: Prim, gamma: Real) -> Cons {
    let [rho, un, ut, p] = w;
    let e = p / (gamma - 1.0) + 0.5 * rho * (un * un + ut * ut);
    [rho * un, rho * un * un + p, rho * un * ut, un * (e + p)]
}

#[inline]
#[allow(clippy::neg_cmp_op_on_partial_ord)] // deliberate: NaN must read as nonphysical
fn nonphysical(q: &Prim) -> bool {
    !(q[0] > RHO_MIN) || !(q[3] > P_MIN_ABS)
}

/// Fluid/fluid face flux in the rotated frame. Face-level positivity fallback:
/// if either reconstructed state is at/below the floors, BOTH sides revert to
/// their cell averages — a one-sided fallback makes the flux multivalued and
/// breaks conservation (physics-reference §5) — and `floors` is incremented.
#[inline]
fn fluid_face_flux(
    s: [Prim; 4],
    sol: [bool; 4],
    use_hll: bool,
    n: &Numerics,
    gamma: Real,
    floors: &mut u64,
) -> Cons {
    let (mut ql, mut qr) = muscl_face_states(s, sol, n.reconstruction, n.limiter);
    if n.reconstruction == Reconstruction::Muscl && (nonphysical(&ql) || nonphysical(&qr)) {
        ql = s[1];
        qr = s[2];
        *floors += 1;
    }
    if ql == qr {
        return analytic_flux(ql, gamma);
    }
    if use_hll {
        hll_flux(ql, qr, gamma)
    } else {
        hllc_flux(ql, qr, gamma)
    }
}

/// `wall_flux_{z,r}` return the OUTWARD-oriented flux `[0, sgn*ps, 0, 0]` /
/// `[0, 0, sgn*ps, 0]` (sgn = outward normal direction). The sweeps difference
/// fluxes in the fixed +z/+r orientation, so multiply by sgn once more: the
/// oriented wall momentum flux is +ps on both sides of a wall.
#[inline]
fn oriented(f: Cons, sgn: Real) -> Cons {
    [sgn * f[0], sgn * f[1], sgn * f[2], sgn * f[3]]
}

/// Axial sweep: MUSCL face states via `kernels::muscl_face_states`, flux via
/// `kernels::hllc_flux` (axial faces ALWAYS use HLLC — contact resolution is
/// what keeps the plume boundary sharp — except under `FluxMode::Hll`).
/// At a fluid/solid face call `crate::physics::wall_flux_z`. Accumulates
/// -dF/dz into `rhs`. Face-level positivity fallback increments `floors`.
#[allow(clippy::too_many_arguments)] // frozen signature
pub fn sweep_z(w: &[Prim], solid: &[bool], mask: &[bool], g: &Grid,
               n: &Numerics, gas: &GasModel, rhs: &mut [Cons], floors: &mut u64) {
    let _ = mask; // carbuncle switch applies to radial fluxes only (§2)
    let snz = g.snz();
    let gamma = gas.gamma;
    let inv_dz = 1.0 / g.dz;
    let use_hll = n.flux_mode == FluxMode::Hll;
    let new_floors: u64 = rhs
        .par_chunks_mut(snz)
        .enumerate()
        .map(|(pr, row)| {
            let ir = pr as isize - NG as isize;
            if ir < 0 || ir >= g.nr as isize {
                return 0u64;
            }
            let mut local = 0u64;
            // Faces f = 0..=nz sit between cells f-1 and f of this row. The
            // z-sweep needs no rotation: canonical order is already
            // [rho, u_n = u_z, u_t = u_r, p].
            for f in 0..=g.nz {
                let (lc, rc) = (f as isize - 1, f as isize);
                let (li, ri) = (g.gidx(lc, ir), g.gidx(rc, ir));
                let (ls, rs) = (solid[li], solid[ri]);
                if ls && rs {
                    continue;
                }
                // NOTE: axial fluid/solid faces stay WALL faces in every
                // wall mode. ColumnReflect applies only to the radial bore
                // face; making these transparent lets mass flow axially
                // through the wall region and the nozzle cannot choke
                // (measured: mdot 7.6x ideal).
                let flux: Cons = if !ls && !rs {
                    let s = [w[g.gidx(lc - 1, ir)], w[li], w[ri], w[g.gidx(rc + 1, ir)]];
                    let sol = [solid[g.gidx(lc - 1, ir)], ls, rs, solid[g.gidx(rc + 1, ir)]];
                    fluid_face_flux(s, sol, use_hll, n, gamma, &mut local)
                } else if !ls {
                    oriented(crate::physics::wall_flux_z(w[li], 1.0, gamma), 1.0)
                } else {
                    oriented(crate::physics::wall_flux_z(w[ri], -1.0, gamma), -1.0)
                };
                if lc >= 0 && !ls {
                    let c = &mut row[NG + lc as usize];
                    for k in 0..4 {
                        c[k] -= flux[k] * inv_dz;
                    }
                }
                if (rc as usize) < g.nz && !rs {
                    let c = &mut row[NG + rc as usize];
                    for k in 0..4 {
                        c[k] += flux[k] * inv_dz;
                    }
                }
            }
            local
        })
        .sum();
    *floors += new_floors;
}

/// Flux through the radial face between rows `j_low` and `j_low + 1` at
/// column `iz`, returned in the CANONICAL frame [mass, z-mom, r-mom, energy].
#[inline]
#[allow(clippy::too_many_arguments)]
fn radial_face_flux(
    w: &[Prim],
    solid: &[bool],
    mask: &[bool],
    g: &Grid,
    n: &Numerics,
    hll_mode: bool,
    gamma: Real,
    iz: isize,
    j_low: isize,
    floors: &mut u64,
) -> Cons {
    let li = g.gidx(iz, j_low);
    let ri = g.gidx(iz, j_low + 1);
    let (ls, rs) = (solid[li], solid[ri]);
    if ls && rs {
        return [0.0; 4];
    }
    if (!ls && !rs) || n.wall_mode == WallMode::ColumnReflect {
        // ColumnReflect: solid cells hold mirror-filled states, wall faces
        // run the regular solver (first-order — solid flags zero the slopes).
        // Rotate to the r-sweep frame: u_n = u_r, u_t = u_z.
        let rot = |q: Prim| -> Prim { [q[0], q[2], q[1], q[3]] };
        let s = [
            rot(w[g.gidx(iz, j_low - 1)]),
            rot(w[li]),
            rot(w[ri]),
            rot(w[g.gidx(iz, j_low + 2)]),
        ];
        let sol = [solid[g.gidx(iz, j_low - 1)], ls, rs, solid[g.gidx(iz, j_low + 2)]];
        let use_hll = hll_mode || mask[li] || mask[ri];
        let f = fluid_face_flux(s, sol, use_hll, n, gamma, floors);
        [f[0], f[2], f[1], f[3]] // rotate back
    } else if !ls {
        oriented(crate::physics::wall_flux_r(w[li], 1.0, gamma), 1.0)
    } else {
        oriented(crate::physics::wall_flux_r(w[ri], -1.0, gamma), -1.0)
    }
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
/// set (or under `FluxMode::Hll`/`HllRadial`), HLLC otherwise. The ±1 axial
/// widening of the carbuncle switch (§2) lives inside `physics::carbuncle_mask`
/// — its declaration sets the mask on cells i-1..=i+1 — so this caller only
/// tests the two cells adjacent to each radial face.
#[allow(clippy::too_many_arguments)] // frozen signature
pub fn sweep_r(w: &[Prim], solid: &[bool], mask: &[bool], g: &Grid,
               n: &Numerics, gas: &GasModel, rhs: &mut [Cons], floors: &mut u64) {
    let snz = g.snz();
    let gamma = gas.gamma;
    let inv_dr = 1.0 / g.dr;
    let hll_mode = matches!(n.flux_mode, FluxMode::Hll | FluxMode::HllRadial);
    // Each row recomputes both of its face fluxes, so every interior face is
    // evaluated twice on identical inputs — bit-identical results, hence still
    // conservative, and no cross-row writes under par_chunks_mut. (A face-level
    // fallback on a shared face is therefore counted twice in `floors`;
    // the ladder only ever asserts the count is zero.)
    let new_floors: u64 = rhs
        .par_chunks_mut(snz)
        .enumerate()
        .map(|(pr, row)| {
            let ir = pr as isize - NG as isize;
            if ir < 0 || ir >= g.nr as isize {
                return 0u64;
            }
            let mut local = 0u64;
            let axisym = n.geometry == Geometry::Axisymmetric;
            for iz in 0..g.nz {
                let izs = iz as isize;
                let ci = g.gidx(izs, ir);
                if solid[ci] {
                    continue;
                }
                // Row 0's lower face is the axis: r_face(0) = 0 exactly, so in
                // axisymmetric mode its flux is weighted by zero regardless of
                // value. Skip the Riemann solve instead of multiplying it away.
                let g_lo = if axisym && ir == 0 {
                    [0.0; 4]
                } else {
                    radial_face_flux(w, solid, mask, g, n, hll_mode, gamma, izs,
                                     ir - 1, &mut local)
                };
                let g_hi = radial_face_flux(w, solid, mask, g, n, hll_mode, gamma, izs,
                                            ir, &mut local);
                let c = &mut row[NG + iz];
                match n.geometry {
                    Geometry::Planar => {
                        for k in 0..4 {
                            c[k] -= (g_hi[k] - g_lo[k]) * inv_dr;
                        }
                    }
                    Geometry::Axisymmetric => {
                        // The single bracket, with the source entering as
                        // S_j = [0, 0, p_j, 0] subtracted from each face flux
                        // before the r-weighting:
                        //
                        //   [ r_hi*(G_hi - S_j) - r_lo*(G_lo - S_j) ] / (dr*r_j)
                        //
                        // Algebraically identical (r_hi - r_lo = dr), and it
                        // makes the uniform-p cancellation bit-exact even
                        // though fl((ir+1)*dr) - fl(ir*dr) need not round to
                        // dr in f32. Row 0's lower face has r_lo = 0 exactly:
                        // the axis carries zero flux by construction.
                        let pj = w[ci][3];
                        let sj: Cons = [0.0, 0.0, pj, 0.0];
                        let r_lo = g.r_face(ir as usize);
                        let r_hi = g.r_face(ir as usize + 1);
                        let inv = 1.0 / (g.dr * g.r_center(ir as usize));
                        for k in 0..4 {
                            c[k] -= (r_hi * (g_hi[k] - sj[k]) - r_lo * (g_lo[k] - sj[k])) * inv;
                        }
                    }
                }
            }
            local
        })
        .sum();
    *floors += new_floors;
}

/// u1 = u0 + dt*rhs on fluid cells; solid cells copy u0 unchanged. Ghost
/// entries copy u0 (rhs is zero there), so ghosts pass through unchanged.
pub fn accumulate(out: &mut [Cons], u0: &[Cons], rhs: &[Cons], dt: Real, solid: &[bool]) {
    out.par_iter_mut()
        .zip(u0.par_iter().zip(rhs.par_iter().zip(solid.par_iter())))
        .for_each(|(o, (u, (r, &s)))| {
            *o = if s {
                *u
            } else {
                [u[0] + dt * r[0], u[1] + dt * r[1], u[2] + dt * r[2], u[3] + dt * r[3]]
            };
        });
}

/// SSP-RK2 stage 2: out = 0.5*(u0 + u1 + dt*rhs) on fluid cells; solid cells
/// copy u0 unchanged.
pub fn accumulate2(out: &mut [Cons], u0: &[Cons], u1: &[Cons], rhs: &[Cons], dt: Real, solid: &[bool]) {
    out.par_iter_mut()
        .zip(u0.par_iter().zip(u1.par_iter().zip(rhs.par_iter().zip(solid.par_iter()))))
        .for_each(|(o, (a, (b, (r, &s))))| {
            *o = if s {
                *a
            } else {
                [
                    0.5 * (a[0] + b[0] + dt * r[0]),
                    0.5 * (a[1] + b[1] + dt * r[1]),
                    0.5 * (a[2] + b[2] + dt * r[2]),
                    0.5 * (a[3] + b[3] + dt * r[3]),
                ]
            };
        });
}

/// Cell-level positivity pass (the face-level fallback lives in the sweeps):
/// clamp rho and p to the floors, incrementing `floors` per clamped cell.
/// See docs/physics-reference.md §5.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // deliberate: NaN must read as at/below floor
pub fn enforce_positivity(u: &mut [Cons], gas: &GasModel, floors: &mut u64) {
    let gm1 = gas.gamma - 1.0;
    let clamped: u64 = u
        .par_iter_mut()
        .map(|c| {
            let mut bad = false;
            if !(c[0] > RHO_MIN) || !c[0].is_finite() {
                c[0] = RHO_MIN;
                bad = true;
            }
            if !c[1].is_finite() || !c[2].is_finite() {
                c[1] = 0.0;
                c[2] = 0.0;
                bad = true;
            }
            let ke = 0.5 * (c[1] * c[1] + c[2] * c[2]) / c[0];
            let p = gm1 * (c[3] - ke);
            if !(p > P_MIN_ABS) || !p.is_finite() {
                c[3] = P_MIN_ABS / gm1 + ke; // clamp p, keep density and momentum
                bad = true;
            }
            bad as u64
        })
        .sum();
    *floors += clamped;
}

/// L2 of (rho_new - rho_old) over fluid cells, f64 accumulator. `step.rs`
/// guarantees the ghost bands of both arrays are identical before this call,
/// so ghost entries contribute exactly zero and need no grid arithmetic.
pub fn density_residual_f64(u0: &[Cons], u1: &[Cons], solid: &[bool]) -> f64 {
    let ss: f64 = u0
        .par_iter()
        .zip(u1.par_iter().zip(solid.par_iter()))
        .map(|(a, (b, &s))| {
            if s {
                0.0
            } else {
                let d = b[0] as f64 - a[0] as f64;
                d * d
            }
        })
        .sum();
    ss.sqrt()
}

// ---------------------------------------------------------------------------
// Unit tests that exercise ONLY this file plus the frozen contract kernels.
// No test below reaches physics.rs (its bodies are session B's): every grid is
// all-fluid, ghosts are filled by a local transmissive/mirror helper, and the
// carbuncle mask is supplied explicitly.

#[cfg(test)]
mod tests {
    use super::*;
    use cfd_contract::kernels::prim_to_cons;
    use cfd_contract::Limiter;

    const GAMMA: Real = 1.4;

    fn numerics(geometry: Geometry) -> Numerics {
        Numerics { geometry, quasi1d_init: false, sponge_cells: 0, ..Numerics::default() }
    }

    /// Fill the ghost band by edge replication: z ghosts copy the nearest
    /// interior column, then whole padded rows replicate outward (covering
    /// corners). Mirror-symmetric for uniform states, transmissive otherwise.
    fn fill_ghosts_transmissive(u: &mut [Cons], g: &Grid) {
        let (nz, nr) = (g.nz as isize, g.nr as isize);
        for ir in 0..nr {
            let first = u[g.gidx(0, ir)];
            let last = u[g.gidx(nz - 1, ir)];
            for k in 1..=NG as isize {
                u[g.gidx(-k, ir)] = first;
                u[g.gidx(nz - 1 + k, ir)] = last;
            }
        }
        let snz = g.snz();
        for k in 1..=NG as isize {
            // Copy padded interior row 0 -> ghost row -k, row nr-1 -> nr-1+k.
            for pz in 0..snz {
                let src = (NG as isize) * snz as isize + pz as isize;
                let dst = (NG as isize - k) * snz as isize + pz as isize;
                u[dst as usize] = u[src as usize];
                let src = (NG as isize + nr - 1) * snz as isize + pz as isize;
                let dst = (NG as isize + nr - 1 + k) * snz as isize + pz as isize;
                u[dst as usize] = u[src as usize];
            }
        }
    }

    fn uniform_state(g: &Grid, prim: Prim) -> Vec<Cons> {
        vec![prim_to_cons(prim, GAMMA); g.glen()]
    }

    fn sweep_both(u: &[Cons], g: &Grid, n: &Numerics) -> (Vec<Cons>, u64) {
        let gas = GasModel { gamma: GAMMA, r_specific_si: 287.0 };
        let mut w = vec![[0.0; 4]; g.glen()];
        compute_primitives(u, &mut w, GAMMA);
        let solid = vec![false; g.glen()];
        let mask = vec![false; g.glen()];
        let mut rhs = vec![[0.0; 4]; g.glen()];
        let mut floors = 0u64;
        sweep_z(&w, &solid, &mask, g, n, &gas, &mut rhs, &mut floors);
        sweep_r(&w, &solid, &mask, g, n, &gas, &mut rhs, &mut floors);
        (rhs, floors)
    }

    #[test]
    fn primitives_round_trip() {
        let g = Grid { nz: 4, nr: 3, dz: 0.1, dr: 0.1 };
        let prim: Prim = [0.7, 1.3, -0.4, 0.9];
        let u = uniform_state(&g, prim);
        let mut w = vec![[0.0; 4]; g.glen()];
        compute_primitives(&u, &mut w, GAMMA);
        for c in &w {
            for k in 0..4 {
                assert!((c[k] - prim[k]).abs() <= 1e-6 * prim[k].abs().max(1.0));
            }
        }
    }

    #[test]
    fn wave_speed_matches_hand_value() {
        let g = Grid { nz: 4, nr: 3, dz: 0.5, dr: 0.25 };
        let prim: Prim = [1.0, 2.0, -1.0, 1.0];
        let u = uniform_state(&g, prim);
        let mut w = vec![[0.0; 4]; g.glen()];
        compute_primitives(&u, &mut w, GAMMA);
        let solid = vec![false; g.glen()];
        let a = (GAMMA as f64).sqrt();
        let expect = (2.0 + a) / 0.5 + (1.0 + a) / 0.25;
        let got = max_wave_speed(&w, &solid, &g, GAMMA) as f64;
        assert!((got - expect).abs() <= 1e-5 * expect, "got {got}, want {expect}");
    }

    #[test]
    fn wave_speed_skips_solid_cells() {
        let g = Grid { nz: 4, nr: 3, dz: 0.5, dr: 0.25 };
        let prim: Prim = [1.0, 2.0, -1.0, 1.0];
        let u = uniform_state(&g, prim);
        let mut w = vec![[0.0; 4]; g.glen()];
        compute_primitives(&u, &mut w, GAMMA);
        // A solid cell with a huge (bogus) wave speed must not win the max.
        let mut solid = vec![false; g.glen()];
        let hot = g.gidx(1, 1);
        w[hot] = [1.0, 1000.0, 0.0, 1.0];
        solid[hot] = true;
        let a = (GAMMA as f64).sqrt();
        let expect = (2.0 + a) / 0.5 + (1.0 + a) / 0.25;
        let got = max_wave_speed(&w, &solid, &g, GAMMA) as f64;
        assert!((got - expect).abs() <= 1e-5 * expect, "got {got}, want {expect}");
    }

    /// A uniform moving state must produce bitwise-zero rhs in Planar mode:
    /// every face sees identical reconstructed states, so every flux is the
    /// identical analytic flux and the differences cancel exactly.
    #[test]
    fn uniform_stream_gives_zero_rhs_planar() {
        let g = Grid { nz: 8, nr: 6, dz: 0.1, dr: 0.07 };
        let u = uniform_state(&g, [1.0, 0.3, 0.2, 1.0]);
        let (rhs, floors) = sweep_both(&u, &g, &numerics(Geometry::Planar));
        assert_eq!(floors, 0);
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                let c = rhs[g.gidx(iz as isize, ir as isize)];
                assert_eq!(c, [0.0; 4], "rhs at ({iz},{ir}) = {c:?}");
            }
        }
    }

    /// The well-balanced property at kernel level: quiescent uniform pressure
    /// in Axisymmetric mode gives bitwise-zero rhs — the single-bracket form
    /// cancels the source against the r-weighted flux difference exactly,
    /// including at row 0 where r_face = 0.
    #[test]
    fn quiescent_axisymmetric_rhs_is_bitwise_zero() {
        let g = Grid { nz: 8, nr: 6, dz: 0.1, dr: 0.1 };
        let u = uniform_state(&g, [1.0, 0.0, 0.0, 1.0]);
        let (rhs, floors) = sweep_both(&u, &g, &numerics(Geometry::Axisymmetric));
        assert_eq!(floors, 0);
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                let c = rhs[g.gidx(iz as isize, ir as isize)];
                assert_eq!(c, [0.0; 4], "rhs at ({iz},{ir}) = {c:?}");
            }
        }
    }

    /// Uniform axial flow (v = 0) in axisymmetric mode is also an exact steady
    /// state: radial fluxes reduce to pressure only and the bracket cancels.
    #[test]
    fn uniform_axial_flow_axisymmetric_rhs_is_bitwise_zero() {
        let g = Grid { nz: 8, nr: 6, dz: 0.1, dr: 0.1 };
        let u = uniform_state(&g, [1.0, 2.0, 0.0, 1.0]);
        let (rhs, _) = sweep_both(&u, &g, &numerics(Geometry::Axisymmetric));
        for ir in 0..g.nr {
            for iz in 0..g.nz {
                let c = rhs[g.gidx(iz as isize, ir as isize)];
                assert_eq!(c, [0.0; 4], "rhs at ({iz},{ir}) = {c:?}");
            }
        }
    }

    #[test]
    fn accumulate_and_accumulate2_respect_solid_and_ghosts() {
        let g = Grid { nz: 4, nr: 3, dz: 0.1, dr: 0.1 };
        let u0 = uniform_state(&g, [1.0, 1.0, 0.0, 1.0]);
        let mut rhs = vec![[0.0; 4]; g.glen()];
        let mut solid = vec![false; g.glen()];
        let a = g.gidx(1, 1);
        let b = g.gidx(2, 1);
        rhs[a] = [1.0, 2.0, 3.0, 4.0];
        rhs[b] = [1.0, 1.0, 1.0, 1.0];
        solid[b] = true;
        let dt: Real = 0.5;
        let mut u1 = vec![[9.0; 4]; g.glen()];
        accumulate(&mut u1, &u0, &rhs, dt, &solid);
        for k in 0..4 {
            assert_eq!(u1[a][k], u0[a][k] + dt * rhs[a][k]);
            assert_eq!(u1[b][k], u0[b][k]); // solid: copied, rhs ignored
        }
        assert_eq!(u1[g.gidx(-1, -1)], u0[g.gidx(-1, -1)]); // ghost passthrough
        let mut u2 = vec![[9.0; 4]; g.glen()];
        accumulate2(&mut u2, &u0, &u1, &rhs, dt, &solid);
        for k in 0..4 {
            assert_eq!(u2[a][k], 0.5 * (u0[a][k] + u1[a][k] + dt * rhs[a][k]));
            assert_eq!(u2[b][k], u0[b][k]);
        }
    }

    #[test]
    fn positivity_clamps_and_counts() {
        let g = Grid { nz: 4, nr: 2, dz: 0.1, dr: 0.1 };
        let gas = GasModel { gamma: GAMMA, r_specific_si: 287.0 };
        let mut u = uniform_state(&g, [1.0, 0.5, 0.0, 1.0]);
        let healthy = u[g.gidx(0, 0)];
        let neg_p = g.gidx(1, 0);
        let neg_rho = g.gidx(2, 0);
        u[neg_p] = [1.0, 0.0, 0.0, -0.5]; // E < 0 => p < 0
        u[neg_rho] = [-1.0, 0.0, 0.0, 2.5];
        let mut floors = 0u64;
        enforce_positivity(&mut u, &gas, &mut floors);
        assert_eq!(floors, 2);
        assert_eq!(u[g.gidx(0, 0)], healthy, "healthy cell must be untouched bit-exactly");
        let wp = cons_to_prim(u[neg_p], GAMMA);
        assert!(wp[3] >= P_MIN_ABS * 0.99 && wp[3] <= P_MIN_ABS * 1.01);
        assert!(u[neg_rho][0] == RHO_MIN);
        // Second pass: already clamped cells sit exactly AT the floor, which
        // still reads as "at/below" — the counter may tick again, but values
        // must be stable.
        let snapshot = u.clone();
        let mut floors2 = 0u64;
        enforce_positivity(&mut u, &gas, &mut floors2);
        assert_eq!(u, snapshot);
    }

    #[test]
    fn residual_is_f64_and_skips_solid() {
        let g = Grid { nz: 3, nr: 2, dz: 0.1, dr: 0.1 };
        let u0 = uniform_state(&g, [1.0, 0.0, 0.0, 1.0]);
        let mut u1 = u0.clone();
        let mut solid = vec![false; g.glen()];
        let a = g.gidx(0, 0);
        let b = g.gidx(1, 0);
        u1[a][0] += 3e-4;
        u1[b][0] += 4e-4;
        // The perturbations round on storage into f32 (ulp(1.0) ~ 6e-8); the
        // tolerance covers that storage rounding, not the f64 accumulation.
        assert!((density_residual_f64(&u0, &u1, &solid) - 5e-4).abs() < 2e-7);
        solid[b] = true;
        assert!((density_residual_f64(&u0, &u1, &solid) - 3e-4).abs() < 2e-7);
    }

    // ------------------------------------------------------------------
    // Sod shock tube driven entirely by this file: local transmissive ghosts,
    // local SSP-RK2 loop, no physics.rs. Mirrors acceptance test T4.

    const P_STAR: f64 = 0.3031301781;
    const U_STAR: f64 = 0.9274526200;
    const RHO_STAR_L: f64 = 0.4263194282;
    const RHO_STAR_R: f64 = 0.2655737117;
    const SHOCK_SPEED: f64 = 1.7521557320;

    fn sod_exact_rho(x: f64, t: f64) -> f64 {
        let g = 1.4f64;
        let a_l = g.sqrt(); // rho_l = p_l = 1
        let a_star_l = a_l * P_STAR.powf((g - 1.0) / (2.0 * g));
        let xi = (x - 0.5) / t;
        if xi < -a_l {
            1.0
        } else if xi < U_STAR - a_star_l {
            (2.0 / (g + 1.0) - (g - 1.0) / ((g + 1.0) * a_l) * xi).powf(2.0 / (g - 1.0))
        } else if xi < U_STAR {
            RHO_STAR_L
        } else if xi < SHOCK_SPEED {
            RHO_STAR_R
        } else {
            0.125
        }
    }

    /// Run Sod to t >= 0.2 with the given numerics; return (L1(rho), time).
    fn run_sod(n: &Numerics) -> (f64, f64) {
        let nz = 200usize;
        let g = Grid { nz, nr: 1, dz: 1.0 / nz as f32, dr: 1.0 / nz as f32 };
        let gas = GasModel { gamma: GAMMA, r_specific_si: 287.0 };
        let solid = vec![false; g.glen()];
        let mask = vec![false; g.glen()];
        let mut u_old = uniform_state(&g, [0.125, 0.0, 0.0, 0.1]);
        for iz in 0..nz / 2 {
            u_old[g.gidx(iz as isize, 0)] = prim_to_cons([1.0, 0.0, 0.0, 1.0], GAMMA);
        }
        let mut u1 = u_old.clone();
        let mut u_new = u_old.clone();
        let mut w = vec![[0.0; 4]; g.glen()];
        let mut rhs = vec![[0.0; 4]; g.glen()];
        let mut floors = 0u64;
        let mut t = 0.0f64;
        while t < 0.2 {
            fill_ghosts_transmissive(&mut u_old, &g);
            compute_primitives(&u_old, &mut w, GAMMA);
            let dt = n.cfl / max_wave_speed(&w, &solid, &g, GAMMA);
            rhs.iter_mut().for_each(|c| *c = [0.0; 4]);
            sweep_z(&w, &solid, &mask, &g, n, &gas, &mut rhs, &mut floors);
            sweep_r(&w, &solid, &mask, &g, n, &gas, &mut rhs, &mut floors);
            accumulate(&mut u1, &u_old, &rhs, dt, &solid);
            enforce_positivity(&mut u1, &gas, &mut floors);
            fill_ghosts_transmissive(&mut u1, &g);
            compute_primitives(&u1, &mut w, GAMMA);
            rhs.iter_mut().for_each(|c| *c = [0.0; 4]);
            sweep_z(&w, &solid, &mask, &g, n, &gas, &mut rhs, &mut floors);
            sweep_r(&w, &solid, &mask, &g, n, &gas, &mut rhs, &mut floors);
            accumulate2(&mut u_new, &u_old, &u1, &rhs, dt, &solid);
            enforce_positivity(&mut u_new, &gas, &mut floors);
            std::mem::swap(&mut u_old, &mut u_new);
            t += dt as f64;
        }
        assert_eq!(floors, 0, "Sod must not trip the positivity floors");
        let mut l1 = 0.0f64;
        let mut rho_max = 0.0f64;
        let mut rv_max = 0.0f64;
        for iz in 0..nz {
            let x = (iz as f64 + 0.5) / nz as f64;
            let wc = cons_to_prim(u_old[g.gidx(iz as isize, 0)], GAMMA);
            l1 += (wc[0] as f64 - sod_exact_rho(x, t)).abs() * g.dz as f64;
            rho_max = rho_max.max(wc[0] as f64);
            rv_max = rv_max.max((wc[0] as f64 * wc[2] as f64).abs());
        }
        assert!(rho_max <= 1.001, "max rho = {rho_max}");
        assert!(rv_max <= 1e-8, "max|rho*v| = {rv_max} — transverse flux leakage");
        (l1, t)
    }

    /// Second-order MUSCL lands in 2.4-4.1e-3; the 6.0e-3 threshold sits in
    /// the gap below first order's 1.32e-2 (physics-reference §12 T4).
    #[test]
    fn sod_muscl_is_second_order_accurate() {
        let (l1, _) = run_sod(&numerics(Geometry::Planar));
        assert!(l1 <= 6.0e-3, "L1(rho) = {l1:.4e} (2nd order: 2.4-4.1e-3, 1st: 1.32e-2)");
    }

    /// The FirstOrder abort-ladder rung must run and land near its own
    /// reference accuracy, clearly worse than MUSCL.
    #[test]
    fn sod_first_order_flag_is_honoured() {
        let n = Numerics {
            reconstruction: Reconstruction::FirstOrder,
            ..numerics(Geometry::Planar)
        };
        let (l1, _) = run_sod(&n);
        assert!(l1 > 6.0e-3 && l1 < 2.5e-2, "first-order L1(rho) = {l1:.4e}");
    }

    /// Every FluxMode variant must produce a sane Sod solution (abort ladder).
    #[test]
    fn sod_flux_mode_flags_are_honoured() {
        for fm in [FluxMode::Hll, FluxMode::HllRadial] {
            let n = Numerics { flux_mode: fm, ..numerics(Geometry::Planar) };
            let (l1, _) = run_sod(&n);
            assert!(l1 <= 8.0e-3, "{fm:?}: L1(rho) = {l1:.4e}");
        }
    }

    /// Limiter::None + Muscl on Sod exercises the face-level positivity
    /// fallback path without demanding it fires; the run must stay finite.
    #[test]
    fn sod_unlimited_survives() {
        let n = Numerics { limiter: Limiter::None, ..numerics(Geometry::Planar) };
        let nz = 200usize;
        let g = Grid { nz, nr: 1, dz: 1.0 / nz as f32, dr: 1.0 / nz as f32 };
        let gas = GasModel { gamma: GAMMA, r_specific_si: 287.0 };
        let solid = vec![false; g.glen()];
        let mask = vec![false; g.glen()];
        let mut u_old = uniform_state(&g, [0.125, 0.0, 0.0, 0.1]);
        for iz in 0..nz / 2 {
            u_old[g.gidx(iz as isize, 0)] = prim_to_cons([1.0, 0.0, 0.0, 1.0], GAMMA);
        }
        let mut u1 = u_old.clone();
        let mut u_new = u_old.clone();
        let mut w = vec![[0.0; 4]; g.glen()];
        let mut rhs = vec![[0.0; 4]; g.glen()];
        let mut floors = 0u64;
        for _ in 0..50 {
            fill_ghosts_transmissive(&mut u_old, &g);
            compute_primitives(&u_old, &mut w, GAMMA);
            let dt = n.cfl / max_wave_speed(&w, &solid, &g, GAMMA);
            assert!(dt.is_finite() && dt > 0.0);
            rhs.iter_mut().for_each(|c| *c = [0.0; 4]);
            sweep_z(&w, &solid, &mask, &g, &n, &gas, &mut rhs, &mut floors);
            sweep_r(&w, &solid, &mask, &g, &n, &gas, &mut rhs, &mut floors);
            accumulate(&mut u1, &u_old, &rhs, dt, &solid);
            enforce_positivity(&mut u1, &gas, &mut floors);
            fill_ghosts_transmissive(&mut u1, &g);
            compute_primitives(&u1, &mut w, GAMMA);
            rhs.iter_mut().for_each(|c| *c = [0.0; 4]);
            sweep_z(&w, &solid, &mask, &g, &n, &gas, &mut rhs, &mut floors);
            sweep_r(&w, &solid, &mask, &g, &n, &gas, &mut rhs, &mut floors);
            accumulate2(&mut u_new, &u_old, &u1, &rhs, dt, &solid);
            enforce_positivity(&mut u_new, &gas, &mut floors);
            std::mem::swap(&mut u_old, &mut u_new);
        }
        for c in &u_old {
            for k in 0..4 {
                assert!(c[k].is_finite());
            }
        }
    }
}
